"""The ONE Scharfetter-Gummel edge flux (dynameta.carriers.sg_flux, 2026-11-04): two Bernoulli calls
in EVERY flux family, accurate at any drift. docs/audit/2026-11-04-sg-flux-two-bernoulli.md.

  (a) value + Jacobian vs a 50-digit mpmath oracle at |v| up to 40, every family: unipolar DD
      electrons, bipolar DD electrons / holes, unipolar DG electrons, bipolar DG electrons / holes.
      ON ee0ad04 (the kahan3 B(v) + v spelling) the four kahan3 families FAIL it -- measured numbers
      in the test docstring; the two bipolar DG families passed there already (e36e808).
  (b) one kernel: every family's Bernoulli and current models are created inside sg_flux (run-time spy).
  (c) the cross-version guard: the defaults-off definition fixtures of tests/_bipolar_calllog.py and
      tests/_dg_calllog.py, SOLVED, equal the arrays recorded on ee0ad04 (tests/data/
      sgflux_6595bde.json) to the measured round-off of the flux change -- and a 1e-6 change of one
      physics parameter fails that gate (the gate still catches physics).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _sgflux_reference as REF  # noqa: E402

_V = (-40.0, -33.0, -27.6, -21.0, -10.0, -0.3, 1e-3, 0.3, 10.0, 21.0, 27.6, 33.0, 40.0)


def _drop(dev, mesh):
    from dynameta.carriers import eq_registry as _R
    _R.clear(dev)                          # a rebuilt device must not inherit the old equation records
    for d in list(ds.get_device_list()):
        if d == dev:
            ds.delete_device(device=d)
    for m in list(ds.get_mesh_list()):
        if m == mesh:
            ds.delete_mesh(mesh=m)


# family -> (bipolar?, DG?, carrier, current model, Bernoulli-argument model, g model, mu parameter)
_FAMILIES = {
    "dd_electrons": (False, False, "electrons", "ElectronCurrent", "vdiff_g", "g_enh", "mu_n"),
    "bipolar_electrons": (True, False, "electrons", "ElectronCurrent", "vdiff_g", "g_enh", "mu_n"),
    "bipolar_holes": (True, False, "holes", "HoleCurrent", "vdiff_gp", "g_enh_p", "mu_p"),
    "dg_electrons": (False, True, "electrons", "ElectronCurrentDG", "vdiff_g_dg", "g_enh", "mu_n"),
    "bipolar_dg_electrons": (True, True, "electrons", "ElectronCurrentDG", "vdiff_g_dg", "g_enh",
                             "mu_n"),
    "bipolar_dg_holes": (True, True, "holes", "HoleCurrentDG", "vdiff_gp_dg", "g_enh_p", "mu_p"),
}


def _one_edge(dev, mesh, bipolar, dg):
    """A single 1 nm edge carrying the family's flux (no solve: models are evaluated directly)."""
    from dynameta.constants import M_E
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_density_gradient as DG
    from dynameta.carriers import physics_drift_diffusion as DD
    _drop(dev, mesh)
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="a")
    ds.add_1d_mesh_line(mesh=mesh, pos=1e-9, ps=1e-9, tag="c")
    ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=mesh, name="c", tag="c", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="a", tag2="c")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    if bipolar:
        ds.node_model(device=dev, region="s", name="NetDoping", equation="1e22")
        BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-7,
                                tau_p_s=1e-7, fd_enhancement=False)
        if dg:
            DG.setup_bipolar_dg(dev, "s", m_eff_n_kg=0.26 * M_E, m_eff_p_kg=0.49 * M_E)
    else:                                     # the unipolar DD always carries the FD g-factor
        DD.setup_semiconductor_region_dd(dev, "s", n_bg_m3=1e22, eps_static=11.7,
                                         dos_mass_kg=0.35 * M_E, mobility_m2Vs=0.01)
        if dg:
            DG.setup_dg_quantum_correction(dev, "s", m_eff_kg=0.35 * M_E)
    for nm in ("QSqrtN", "QSqrtP"):
        if nm in ds.get_node_model_list(device=dev, region="s"):
            ds.set_node_values(device=dev, region="s", name=nm, values=[1.0, 1.0])
    for nm in ("QLambda", "QLambdaP", "Holes", "Electrons"):
        if nm in ds.get_node_model_list(device=dev, region="s"):
            ds.set_node_values(device=dev, region="s", name=nm, values=[0.0, 0.0])


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_sg_flux_value_and_jacobian_are_accurate_at_large_drift(family):
    """Every SG flux family on one 1 nm edge, the drift set through Potential (|v| up to 40; an
    insulator-wall edge sits at |v| ~ 20-28 by construction, a depletion edge of a coarse mesh can
    too), each Bernoulli term isolated by zeroing the other node's density. Oracles at the v DEVSIM
    computed, 50-digit mpmath B and B':
        electrons: J / P = c@n1 B(-v) - c@n0 B(v),  dJ/dpsi@n0 / P = -(c@n1 B'(-v) + c@n0 B'(v)) / (V_t g)
        holes:     J / P = c@n1 B(v) - c@n0 B(-v),  dJ/dpsi@n0 / P =  (c@n1 B'(v) + c@n0 B'(-v)) / (V_t g)
    with P = +-q mu V_t EdgeInverseLength g. Gate 1e-13 on both (measured on this tree, 2026-11-04:
    value <= 3.3e-16, Jacobian <= 8.9e-16, every family). The Jacobian is gated for |v| >= 0.1 only:
    DEVSIM 2.11.2's dBdx itself loses accuracy as v -> 0 (vs mpmath: 2.1e-14 at |v| = 1e-2, 4.1e-13 at
    1e-3, 2.0e-12 at 1e-4, 2.4e-10 at 1e-6, 4.8e-5 at 1e-12), identically in both flux spellings and
    only in the Jacobian (Newton rate, not the solution) -- out of this change's scope (audit note,
    open items).
    ON ee0ad04 (kahan3, B(-v) spelled B(v) + v) the four kahan3 families FAIL both (measured, worst
    over the two isolated terms, value / Jacobian): |v| = 40: 1.0 / 0.18-0.64; 33: 1.1-2.3e-2 /
    4.4e-5-9.7e-4; 27.6: 0.7-1.6e-4 / 4.1e-8-4.6e-7; 21: 0.6-1.3e-7 / 3.2e-10-8.0e-9; 10: 0.5-2.3e-12 /
    1.1-2.3e-13 (dd_electrons, bipolar_electrons, bipolar_holes, dg_electrons; electrons at v < 0,
    holes at v > 0). The bipolar DG families passed there already (two Bernoulli since e36e808)."""
    pytest.importorskip("mpmath")
    worst_v, worst_d = flux_accuracy(family)
    assert worst_v < 1e-13, (family, worst_v)
    assert worst_d < 1e-13, (family, worst_d)


def flux_accuracy(family, vs=_V):
    """(worst value error, worst Jacobian error) of `family` against mpmath (the gate above; also
    run by validation/sg_flux_cross_version.py GATE A). The Jacobian only for |v| >= 0.1."""
    import mpmath as mp
    bipolar, dg, carrier, cur, vd, gname, mu = _FAMILIES[family]
    el = carrier == "electrons"
    c = "Electrons" if el else "Holes"
    dev, mesh = "sgf_" + family, "sgf_m_" + family
    worst_v = worst_d = 0.0

    def B(x):
        with mp.workdps(50):
            return x / mp.expm1(x) if x != 0 else mp.mpf(1)

    def dB(x):
        with mp.workdps(50):
            if x == 0:
                return mp.mpf(-0.5)
            e = mp.exp(x)
            return (e - 1 - x * e) / (e - 1) ** 2

    def edge(nm):
        return float(ds.get_edge_model_values(device=dev, region="s", name=nm)[0])

    try:
        _one_edge(dev, mesh, bipolar, dg)
        vt = float(ds.get_parameter(device=dev, region="s", name="V_t"))
        q = float(ds.get_parameter(device=dev, region="s", name="ElectronCharge"))
        mob = float(ds.get_parameter(device=dev, region="s", name=mu))
        for v in vs:
            ds.set_node_values(device=dev, region="s", name="Potential", values=[v * vt, 0.0])
            for c0, c1 in ((0.0, 1e20), (1e20, 0.0)):
                ds.set_node_values(device=dev, region="s", name=c, values=[c0, c1])
                g = edge(gname)
                pref = (1.0 if el else -1.0) * q * mob * vt * edge("EdgeInverseLength") * g
                vv = mp.mpf(edge(vd))
                if el:
                    ref = c1 * B(-vv) - c0 * B(vv)
                    dref = -(c1 * dB(-vv) + c0 * dB(vv)) / (vt * g)
                else:
                    ref = c1 * B(vv) - c0 * B(-vv)
                    dref = (c1 * dB(vv) + c0 * dB(-vv)) / (vt * g)
                worst_v = max(worst_v, float(abs(edge(cur) / pref / ref - 1)))
                if abs(v) >= 0.1:             # DEVSIM's own dBdx near 0: docstring
                    worst_d = max(worst_d, float(abs(edge(cur + ":Potential@n0") / pref / dref - 1)))
    finally:
        _drop(dev, mesh)
    return worst_v, worst_d


def test_every_flux_family_is_built_by_the_one_sg_kernel(monkeypatch):
    """House rule (one kernel per formula), checked at RUN time: building every family (_FAMILIES),
    every DEVSIM edge model whose expression calls B( / dBdx( or kahan3s a Bern* product, and every
    current model, is created INSIDE sg_flux.install_sg_current (a spy on devsim.edge_model, flagged
    by a wrapper around the kernel). On ee0ad04 it fails: the classical and unipolar-DG fluxes were
    spelled in their own modules."""
    import re
    from dynameta.carriers import sg_flux as SG
    real_em, real_k = ds.edge_model, SG.install_sg_current
    made, depth = [], [0]

    def em(*a, **kw):
        made.append((kw.get("name", ""), kw.get("equation", ""), depth[0] > 0))
        return real_em(*a, **kw)

    def kernel(*a, **kw):
        depth[0] += 1
        try:
            return real_k(*a, **kw)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(ds, "edge_model", em)
    monkeypatch.setattr(SG, "install_sg_current", kernel)
    currents = {f[3] for f in _FAMILIES.values()}
    for family, (bipolar, dg, *_rest) in _FAMILIES.items():
        dev, mesh = "sgk_" + family, "sgk_m_" + family
        try:
            _one_edge(dev, mesh, bipolar, dg)
        finally:
            _drop(dev, mesh)
    bern = re.compile(r"(?<![A-Za-z_])(B|dBdx)\(|kahan3\([^)]*Bern")
    outside = [(n, e) for n, e, inside in made
               if not inside and ":" not in n and (bern.search(e) or n in currents)]
    assert not outside, outside
    assert {n for n, _e, inside in made if inside and ":" not in n} >= currents


# ---- (c) the cross-version numeric guard ---------------------------------------------------------

def _solve_in_child(perturb=None, statistics="legacy"):
    """REF.solve_fixtures() in a child process (the cl_bz circuit cannot be removed from this one --
    tests/_bipolar_calllog.py), on this tree. statistics (v0.15.0): 'legacy' -- the path the ee0ad04 arrays
    pin (the pre-v0.15.0 default); None = this tree's default ('fermi_dirac')."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
    if statistics:
        env["SGFLUX_STATISTICS"] = statistics
    else:
        env.pop("SGFLUX_STATISTICS", None)
    if perturb:
        env["SGFLUX_PERTURB"] = perturb
    else:
        env.pop("SGFLUX_PERTURB", None)
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        r = subprocess.run([sys.executable, REF.__file__, "--emit", out], env=env,
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.remove(out)


@pytest.fixture(scope="module")
def solved():
    return _solve_in_child()


def test_defaults_off_fixtures_equal_6595bde_to_flux_round_off(solved):
    """The defaults-off DEFINITION fixtures (tests/_bipolar_calllog.py: FD diode with DC contacts,
    Boltzmann diode with a circuit contact; tests/_dg_calllog.py: the unipolar DG ITO bar), solved,
    against the arrays recorded on ee0ad04 (kahan3 everywhere, tests/data/sgflux_6595bde.json).
    The definition digests were RE-RECORDED for the two-Bernoulli flux (2026-11-04), so this is the
    gate that still pins the physics: every field and terminal current within REF.TOL of the old
    tree -- the measured round-off of the flux change times ~10 (REF module docstring). REVISED 2026-11-07
    (v0.15.0 default statistics 'fermi_dirac'): the gate pins statistics='legacy', the path the arrays were
    recorded on; the default's move is measured by test_v0150_default_moves_the_fd_and_dg_fixtures."""
    rows = REF.compare(REF.load_reference(), solved)
    bad = [r for r in rows if not r[3]]
    assert not bad, bad
    assert len(rows) == REF.N_COMPARED


def test_v0150_default_moves_the_fd_and_dg_fixtures():
    """MEASUREMENT of the v0.15.0 default change on these fixtures (statistics unset = 'fermi_dirac' vs the
    ee0ad04 arrays = the legacy statistics): the Boltzmann (bz) rows are untouched -- they are within the
    gate (their statistics did not change) -- and the FD (fd) and DG ITO (dg) rows move by physics (the FD
    detailed balance, the quasi-field transport). MEASURED values: audit note
    2026-11-06-v0.15-fermi-dirac-mode-complete.md, section 10."""
    rows = REF.compare(REF.load_reference(), _solve_in_child(statistics=None))
    moved = {r[0] for r in rows if not r[3]}
    assert all(not k.startswith("bz") for k in moved), moved
    assert any(k.startswith("fd") for k in moved) and any(k.startswith("dg") for k in moved), rows
    print("\n".join("{} {:.3e} (tol {:.0e})".format(k, m, t) for k, m, t, ok in rows))


def test_cross_version_gate_still_catches_physics():
    """The same gate on the same fixtures with mu_n x (1 + 1e-6) (SGFLUX_PERTURB, applied after setup
    in the child) FAILS: a real physics change of one part per million is ~100x the tolerance on the
    currents. So converting the byte-identity guard to a tolerance kept its purpose."""
    rows = REF.compare(REF.load_reference(), _solve_in_child(perturb="mu_n"))
    assert [r for r in rows if not r[3]], rows
