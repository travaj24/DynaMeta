"""The opt-in UNIFORM junction mesh band of LayeredDevsimBuilder (v0.15.1, junction_band=; carriers.junction_mesh; user
decision 2026-11-08 "please add it as an opt-in and see how it compares"). Audit note with the measured comparison:
docs/audit/2026-11-08-v0.15-junction-mesh-band.md. The devsim-free kernels: tests/test_junction_mesh.py.

Gates
  D1 DEFAULTS OFF, cross-commit: the ordered DEVSIM mesh-definition call log of build_device (create_2d_mesh,
     add_2d_mesh_line, add_2d_region, add_2d_interface, add_2d_contact with every argument's repr) hashes to the
     digest RECORDED ON 098553f (the v0.15.0 tip this branch was cut from) for five builder fixtures -- the
     GaN-like E4 diode, the Si lateral diode, the reference modulator (drift-diffusion), the N-AlGaAs / p-GaAs
     stack and the in-region z-junction GaAs layer -- with junction_band omitted, None and False. (Measured as well
     on 098553f vs this branch, same scratch recorder, Windows: the node coordinates and the solved digests of the
     four solved fixtures are identical too -- audit note.)
  D2 DEFAULTS OFF, same tree: the Si lateral diode solved with junction_band omitted / None / False is BITWISE equal
     (node fields + contact currents); devsim_isolated (a bitwise comparison of separate solves).
  G1 the band: the junction a node with equal edges, spacing h uniform over the half-width, and the report's
     W_0 / V_t / E_0 equal to the depletion approximation RE-DERIVED here (Sze Eqs. 2.2-2.10, scipy.constants);
     the default h = min(V_t / E_0, the spacing of the line already on the junction); no probe device left behind.
  A1 THE MOTIVATING CASE (fast; the 1-D extended-precision twin on the builder's OWN x nodes, which reproduces the
     2-D builder to 4 digits -- docs/audit/2026-11-06-v0.15-followups.md item 3): the GaN-like E4 diode at its
     nominal 1 nm junction spacing. Banded: |I / I_SNS - 1| <= 3e-3 (the D2b gate-A budget E4 inherits; the oracle
     is the Shockley + Sah-Noyce-Shockley current re-derived in tests/test_extended_precision_design.py).
     Band OFF (the graded mesh, the old code's only mesh): > 3e-3 (MEASURED +4.944e-3, the 0.7716 nm remainder edge).
     ADVERSARIAL: the banded node set with the edge ending at the junction shortened to 0.5 nm misses by > 3e-3.
  A2 (@slow, 88 s MEASURED) the same on the 2-D builder itself, extended precision, at +1.0 V.
  Z1 detection: the AlGaAs / GaAs stack gets ONE z band at the interface and its x lines are untouched; the in-region
     z expression gets its band at the centre of its undoped sliver; an n-p-n lateral layer gets two x bands; a
     junction near the edge-metal line has its band clipped there (the line stays, no sliver edge).
  Z2 no junction (reference modulator): one AdvisoryWarning, and the call log is the graded one.
FAILS ON THE OLD CODE (098553f): TypeError (no junction_band keyword) / ImportError (no junction_mesh) everywhere
but D1, an IDENTITY gate (its digests were recorded there), and A1's band-off half, which IS the old mesh.
Runtime (MEASURED, Windows dev box, serial, -m "not slow"): see the solvers-devsim entry in .github/workflows/ci.yml.
"""
import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
from scipy.constants import e as QE, epsilon_0 as EPS0, k as KB                  # noqa: E402

from dynameta.carriers import devsim_layered as DL                               # noqa: E402
from dynameta.carriers import junction_mesh as JM                                # noqa: E402
from dynameta.sweep import BiasPoint                                             # noqa: E402
from dynameta.warning_types import AdvisoryWarning                               # noqa: E402
from test_extended_precision_design import (                                     # noqa: E402
    GAN, SI, _diode, _gan_1d_miss, _sns_oracle_A_per_m2)

VT = KB * 300.0 / QE
_LOGGED = ("create_2d_mesh", "add_2d_mesh_line", "add_2d_region", "add_2d_interface", "add_2d_contact")


def _clean_process():
    from dynameta.carriers import eq_registry as _R
    for d in list(devsim.get_device_list()):
        _R.clear(d)
        devsim.delete_device(device=d)
    for m in list(devsim.get_mesh_list()):
        devsim.delete_mesh(mesh=m)


@pytest.fixture(autouse=True)
def _clean():
    _clean_process()
    yield
    _clean_process()


def _gaas_stack():
    from validation.devsim_heterojunction import ALGAAS, GAAS, stack_design, transport
    return stack_design([("pgaas", "GaAs_p", 400e-9, transport(GAAS, n_bg=1e23, physics="bipolar_dd", acceptor=True)),
                         ("nalgaas", "AlGaAs_n", 400e-9, transport(ALGAAS, n_bg=1e23, physics="bipolar_dd"))])


ZJ_HOMO = None


def _gaas_homo_one():
    """tests/test_devsim_heterojunction.py's single-layer homojunction: an in-region z expression with an undoped
    sliver of width 0.5 nm centred on zj."""
    global ZJ_HOMO
    from validation.devsim_heterojunction import GAAS, T_METAL, stack_design, transport
    zj, H = T_METAL + 400e-9, 1e-9
    ZJ_HOMO = zj
    expr = "ifelse(y < {a:.10e}, -{na:.6e}, ifelse(y > {b:.10e}, {nd:.6e}, 0.0))".format(
        a=zj - H / 4, b=zj + H / 4, na=1e23, nd=1e23)
    return stack_design([("pn", "GaAs_pn", 800e-9, transport(GAAS, n_bg=1e23, physics="bipolar_dd",
                                                             net_doping_expr=expr))], ps_if=H, coarse=H, zone=100e-9)


def _refmod():
    from validation._reference_device import build_reference_modulator
    return build_reference_modulator("drift_diffusion")


FIXTURES = {
    "gan_e4": lambda: _diode(dict(GAN, mid=1e-9), True, name="gan"),
    "si_lat": lambda: _diode(SI, None, name="si"),
    "refmod_dd": _refmod,
    "gaas_pn": _gaas_stack,
    "gaas_homo_one": _gaas_homo_one,
}
# RECORDED ON 098553f (2026-11-08, Windows, devsim 2.11.2) with the same hashing (sha256 of the JSON of
# [(call name, sorted [(kwarg, repr(value))])], first 16 hex digits). Python floats' repr is platform-independent.
CALL_LOG_0554410 = {"gan_e4": ("2222f9fa11e3b536", 14), "si_lat": ("88ab8e4edd6b754a", 14),
                    "refmod_dd": ("c0ffd08506038d10", 52), "gaas_pn": ("5e37a0478f5b930a", 24),
                    "gaas_homo_one": ("a9477d37a15a4029", 20)}


def _call_log(monkeypatch, name, **kw):
    calls = []
    for nm in _LOGGED:
        f = getattr(DL.ds, nm)

        def w(*a, _f=f, _nm=nm, **k):
            calls.append((_nm, sorted((kk, repr(v)) for kk, v in k.items())))
            return _f(*a, **k)
        monkeypatch.setattr(DL.ds, nm, w)
    b = DL.LayeredDevsimBuilder(FIXTURES[name](), mesh_name="m_" + name, device_name="d_" + name, **kw)
    try:
        b.build_device()
    finally:
        b.teardown()
        monkeypatch.undo()
    return hashlib.sha256(json.dumps(calls).encode()).hexdigest()[:16], len(calls), b


@pytest.mark.parametrize("name", sorted(CALL_LOG_0554410))
def test_d1_defaults_off_mesh_call_log_is_the_0554410_digest(monkeypatch, name):
    for kw in ({}, {"junction_band": None}, {"junction_band": False}):
        digest, n, b = _call_log(monkeypatch, name, **kw)
        assert (digest, n) == CALL_LOG_0554410[name], (name, kw)
        assert b.junction_band is None and b.junction_band_report == []
    # ADVERSARIAL (sensitivity): the band on changes the log wherever there is a junction
    if name != "refmod_dd":
        assert _call_log(monkeypatch, name, junction_band=True)[0] != CALL_LOG_0554410[name][0]


def _solved_digest(tag, **kw):
    b = DL.LayeredDevsimBuilder(_diode(SI, None, name="si"), mesh_name="m" + tag, device_name="d" + tag, **kw)
    try:
        cf = b.solve(BiasPoint({"anode": 0.3}, "v"))
        h = hashlib.sha256()
        for r in sorted(cf.regions):
            for nm, a in sorted(cf.regions[r].node_fields.items()):
                h.update(nm.encode())
                h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        for e in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            h.update(np.float64(devsim.get_contact_current(device=b.device, contact="anode", equation=e)).tobytes())
        return h.hexdigest()
    finally:
        b.teardown()


@pytest.mark.devsim_isolated("bitwise comparison of three separate solves")
def test_d2_defaults_off_same_tree_solve_is_bitwise():
    ref = _solved_digest("o")
    assert _solved_digest("n", junction_band=None) == ref
    assert _solved_digest("f", junction_band=False) == ref
    assert _solved_digest("t", junction_band=True) != ref                  # sensitivity


# ------------------------------------------------------------------------------------------------ G1 geometry
def _depletion(N_A, N_D, eps_r, n_i):
    """Sze Eqs. 2.2-2.10 (re-derived here): V_bi = V_t ln(N_A N_D / n_i^2), W = sqrt(2 eps V_bi (N_A + N_D) /
    (q N_A N_D)), E_max = 2 V_bi / W."""
    vbi = VT * math.log(N_A * N_D / n_i ** 2)
    W = math.sqrt(2 * eps_r * EPS0 * vbi * (N_A + N_D) / (QE * N_A * N_D))
    return vbi, W, 2 * vbi / W


def _xnodes(dev, region="pn"):
    return np.unique(np.round(np.asarray(devsim.get_node_model_values(device=dev, region=region, name="x")), 18))


@pytest.mark.parametrize("g,mid", [(GAN, 1e-9), (GAN, 0.5e-9), (SI, 5e-9)], ids=["gan_1nm", "gan_05nm", "si_5nm"])
def test_g1_band_geometry_and_the_physical_defaults(g, mid):
    b = DL.LayeredDevsimBuilder(_diode(dict(g, mid=mid), None, name="g"), mesh_name="g1m", device_name="g1d",
                                junction_band=True)
    try:
        b.build_device()
        assert sorted(devsim.get_device_list()) == ["g1d"] and sorted(devsim.get_mesh_list()) == ["g1m"]   # no probe
        xs = _xnodes("g1d")
    finally:
        b.teardown()
    (e,) = b.junction_band_report
    vbi, W, Emax = _depletion(g["NA"], g["ND"], g["eps_r"], g["n_i"])
    # budget: CODATA 2018 (library) vs 2022 (scipy) eps0, 6.8e-10 relative -> 1e-9
    assert e["axis"] == "x" and e["regions"] == ("pn",) and e["position_m"] == g["P"] / 2
    assert e["V_bi"] == pytest.approx(vbi, rel=1e-9) and e["W0_m"] == pytest.approx(W, rel=1e-9)
    assert e["peak_width_m"] == pytest.approx(VT / Emax, rel=1e-9)
    assert e["half_width_m"] == e["W0_m"] and e["line_spacing_m"] == mid
    h = min(VT / Emax, mid)
    assert e["spacing_m"] == pytest.approx(h, rel=1e-9)
    xj = g["P"] / 2
    k = int(np.argmin(np.abs(xs - xj)))
    assert xs[k] == pytest.approx(xj, abs=1e-18)
    m = math.ceil(W / e["spacing_m"] - 1e-9)
    band = xs[k - m:k + m + 1]
    # node positions are read back rounded to 1e-18 m: |error| <= 1e-18 / h relative
    assert np.allclose(np.diff(band), e["spacing_m"], rtol=0, atol=2e-18)
    assert band[0] <= xj - W and band[-1] >= xj + W
    # ADVERSARIAL: the graded mesh fails the uniformity (the 0.7716 nm remainder edge at 1 nm, GaN)
    if g is GAN and mid == 1e-9:
        bg = DL.LayeredDevsimBuilder(_diode(dict(g, mid=mid), None, name="g"), mesh_name="g0m", device_name="g0d")
        try:
            bg.build_device()
            xg = _xnodes("g0d")
        finally:
            bg.teardown()
        kg = int(np.argmin(np.abs(xg - xj)))
        assert abs((xg[kg] - xg[kg - 1]) - 0.7716e-9) < 1e-13


def test_g2_explicit_half_width_and_spacing_win():
    jb = JM.JunctionBand(half_width_m=20e-9, spacing_m=0.25e-9)
    b = DL.LayeredDevsimBuilder(_diode(SI, None, name="si"), mesh_name="g2m", device_name="g2d", junction_band=jb)
    try:
        b.build_device()
        xs = _xnodes("g2d")
    finally:
        b.teardown()
    (e,) = b.junction_band_report
    assert e["half_width_m"] == 20e-9 and e["spacing_m"] == 0.25e-9
    xj = SI["P"] / 2
    band = xs[(xs >= xj - 20e-9 - 1e-15) & (xs <= xj + 20e-9 + 1e-15)]
    assert band.size == 161 and np.allclose(np.diff(band), 0.25e-9, rtol=0, atol=2e-18)


# ------------------------------------------------------------------------------------------------ A1 / A2 accuracy
def _gan_nodes(**kw):
    b = DL.LayeredDevsimBuilder(_diode(dict(GAN, mid=1e-9), True, name="gan"), mesh_name="a1m", device_name="a1d",
                                **kw)
    try:
        b.build_device()
        return _xnodes("a1d"), b.junction_band_report
    finally:
        b.teardown()


def test_a1_band_fixes_the_motivating_e4_mesh_error_and_graded_does_not():
    """The 1-D extended-precision twin on the builder's x nodes, +1.0 V, vs the Sah-Noyce-Shockley oracle (budget
    3e-3, D2b gate A). MEASURED (Windows, 2026-11-08): graded +4.944e-3 (FAILS), banded -6.4e-4 (h = 1 nm),
    adversarial +4.6e-3."""
    xs_off, rep_off = _gan_nodes()
    xs_on, (e,) = _gan_nodes(junction_band=True)
    assert rep_off == [] and e["spacing_m"] == 1e-9
    miss_off, miss_on = _gan_1d_miss(xs_off), _gan_1d_miss(xs_on)
    assert miss_off > 3e-3, miss_off                              # the old (graded) mesh misses the budget
    assert abs(miss_on) <= 3e-3, miss_on                          # the band meets it
    assert abs(miss_on) < abs(miss_off)
    # ADVERSARIAL: the band's p-side lattice shifted 0.5 nm towards the junction, i.e. ONE 0.5 nm edge ending at
    # the junction inside an otherwise uniform 1 nm band (the follow-up's (d) on the builder's nodes)
    xj = GAN["P"] / 2
    sel = (xs_on < xj) & (xs_on >= xj - e["half_width_m"] - 1e-15)
    bad = xs_on + np.where(sel, 0.5e-9, 0.0)
    assert _gan_1d_miss(bad) > 3e-3


@pytest.mark.slow
def test_a2_2d_extended_gan_diode_banded_meets_the_budget():
    """The 2-D builder itself (extended precision, 1131 / 2181 nodes graded / banded), +1.0 V. MEASURED 2026-11-08:
    graded +4.94e-3 (fails), banded -6.1e-4; 88 s on the Windows dev box (+1.0 V only)."""
    from test_extended_precision_design import _anode_current
    oracle = _sns_oracle_A_per_m2(1.0, GAN) * GAN["T"]
    got = {}
    for tag, kw in (("off", {}), ("on", {"junction_band": True})):
        b = DL.LayeredDevsimBuilder(_diode(dict(GAN, mid=1e-9), True, name="gan"), mesh_name="a2m" + tag,
                                    device_name="a2d", **kw)
        try:
            b.solve(BiasPoint({"anode": 1.0}, "v"), rel_tol=1e-12, abs_tol=1e8)
            got[tag] = _anode_current("a2d") / oracle - 1.0
        finally:
            b.teardown()
    assert got["off"] > 3e-3, got
    assert abs(got["on"]) <= 3e-3, got


# ------------------------------------------------------------------------------------------------ Z detection
def _built(design, tag, **kw):
    b = DL.LayeredDevsimBuilder(design, mesh_name="zm" + tag, device_name="zd" + tag, **kw)
    b.build_device()
    out = {r: (np.asarray(devsim.get_node_model_values(device=b.device, region=r, name="x")),
               np.asarray(devsim.get_node_model_values(device=b.device, region=r, name="y")))
           for r in devsim.get_region_list(device=b.device)}
    b.teardown()
    return b, out


def test_z1_stacked_heterojunction_gets_one_z_band_and_keeps_its_x_lines():
    b0, n0 = _built(_gaas_stack(), "0")
    b1, n1 = _built(_gaas_stack(), "1", junction_band=True)
    (e,) = b1.junction_band_report
    assert e["axis"] == "z" and e["regions"] == ("pgaas", "nalgaas") and e["N_A_m3"] == e["N_D_m3"] == 1e23
    zj = 20e-9 + 400e-9                                         # T_METAL + the p layer
    assert e["position_m"] == pytest.approx(zj, abs=1e-18) and e["line_spacing_m"] == 0.5e-9
    assert e["spacing_m"] == 0.5e-9                             # the interface's own 0.5 nm is finer than V_t/E_0
    for r in ("pgaas", "nalgaas"):
        assert np.array_equal(np.unique(n0[r][0]), np.unique(n1[r][0]))      # x lines untouched
    z = np.unique(np.round(np.concatenate([n1[r][1] for r in ("pgaas", "nalgaas")]), 18))
    band = z[(z >= zj - e["half_width_m"]) & (z <= zj + e["half_width_m"])]
    assert np.allclose(np.diff(band), 0.5e-9, rtol=0, atol=2e-18) and zj in np.round(z, 18)


def test_z1_in_region_z_expression_band_sits_on_the_sliver_centre():
    b, n = _built(_gaas_homo_one(), "h", junction_band=True)
    (e,) = b.junction_band_report
    assert e["axis"] == "z" and abs(e["position_m"] - ZJ_HOMO) <= 1e-6 * e["spacing_m"]
    assert e["spacing_m"] == 1e-9                               # the fixture's uniform 1 nm (finer than V_t/E_0)


def _lateral(expr, name, P=300e-9, mid=5e-9):
    """The Si lateral diode of tests/test_extended_precision_design.py with another NetDoping expression."""
    d = _diode(dict(SI, P=P, mid=mid), None, name=name)
    from dynameta.materials import Material
    m = d.materials.get("semi")
    d.materials.add(Material("semi2", m.optical, transport=dataclasses.replace(m.transport, net_doping_expr=expr)))
    st = dataclasses.replace(d.stack, layers=[dataclasses.replace(d.stack.layers[0], background_material="semi2")])
    return dataclasses.replace(d, stack=st)


def test_z1_npn_gets_two_bands_and_a_band_is_clipped_at_the_edge_metal_line():
    expr = "ifelse(x < 1.0e-07, 1e24, ifelse(x < 2.0e-07, -1e24, 1e24))"
    b, n = _built(_lateral(expr, "npn"), "n", junction_band=True)
    assert [e["position_m"] for e in b.junction_band_report] == pytest.approx([1e-7, 2e-7], abs=1e-15)
    xs = np.unique(np.round(n["pn"][0], 18))
    for e in b.junction_band_report:
        k = int(np.argmin(np.abs(xs - e["position_m"])))
        assert xs[k - 1:k + 2] - xs[k] == pytest.approx([-e["spacing_m"], 0.0, e["spacing_m"]], abs=1e-17)
    # a junction 6 nm from the anode's edge-metal line (x = 1.5 nm): the band is clipped there and the line stays
    b2, n2 = _built(_lateral("ifelse(x < 7.5e-09, -1e24, 1e24)", "clip"), "c", junction_band=True)
    (e,) = b2.junction_band_report
    xs2 = np.unique(np.round(n2["pn"][0], 18))
    assert 1.5e-9 in np.round(xs2, 18)
    left = xs2[(xs2 >= 1.5e-9 - 1e-18) & (xs2 <= e["position_m"] + 1e-18)]
    d = np.diff(left)
    assert np.allclose(d[1:], e["spacing_m"], rtol=0, atol=2e-17) and 0.25 * e["spacing_m"] <= d[0] < 1.25 * e["spacing_m"]


def test_z2_no_junction_warns_once_and_keeps_the_graded_mesh(monkeypatch):
    with pytest.warns(AdvisoryWarning, match="no p-n junction found") as rec:
        digest, n, b = _call_log(monkeypatch, "refmod_dd", junction_band=True)
    assert sum("no p-n junction" in str(w.message) for w in rec) == 1
    assert (digest, n) == CALL_LOG_0554410["refmod_dd"] and b.junction_band_report == []
