"""v0.15 plan item 4.1.2 -- BTBT parameter sets in DEVSIM devices (carriers.tunneling): per-set wiring and
units, several processes per region (a tuple of sets), the FDT with two BTBT terms, the opt-in temperature
scaling of B in a device and through the builders, and defaults off. Devsim-free gates (the printed-table
pins, the Kane function, the scaling law): tests/test_tunneling_presets.py. Audit note:
docs/audit/2026-11-06-v0.15-btbt-presets.md.

Gates
  P4  1-D abrupt symmetric Zener, q Int G vs the integrated analytic estimate (depletion profile, exact
      first-integral Emax, flat quasi-Fermi Hurkx D -- the Z1 oracle of tests/test_devsim_tunneling.py
      re-coded here for a material eps, n_i and a SUM of sets) within 1 % + delta_D (Z1's budget), for Ge
      direct, Ge indirect, the Ge tuple and the computed InGaAs set. A 100x unit error or a P = 2 <-> 2.5
      swap misses by >= 0.5 (asserted on the oracle). The material eps / n_i below are FIXTURE values (the
      oracle uses the same numbers), not cited constants.
  P4c the Si Kao sets peak within 1/k = (V_bi + V_r)/B = 0.14 / 0.55 nm of the junction, so their Zener gate is
      a CONVERGENCE gate: observed order >= 1.7 on three uniform meshes and the Richardson limit within
      1 % + delta_D. ROOT CAUSE of the earlier non-convergence (audit note): the add_1d_mesh_line ps-mesh put
      a FIXED 0.128 nm edge on the field peak at every ps; the edge-midpoint quadrature then carries an
      h-independent error (~ k h_e/2 - 1 + exp(-k h_e/2) = +8.8 % for an e^(-k|x|) rate; MEASURED +11 %).
      The fixtures now use explicit uniform nodes (_nodes); that edge put back is the adversarial.
  P4s the installed rate == numpy A E^P exp(-B/E) D with the PRINTED SI literals for the Si Kao sets on a
      frozen state (rel 1e-12, the Z5 budget), which catches a unit or P error exactly.
  P5  composition: GERMANIUM_KAO as a tuple at a FROZEN state gives term rates and integrals BIT-equal to the
      single-set devices and a total == I_direct + I_indirect exactly; both terms carry a noise pair; the
      solved tuple current exceeds each process alone.
  FDT S_I = 4 kT Re Y with two BTBT terms (direct + phonon-assisted shapes) to 1e-4 (the D2a budget).
  T   a 350 K region with btbt_at_temperature: the installed rate uses B (E_g(350)/E_g(300))^1.5 (oracle:
      re-typed Vurgaftman Eq. (4.1)), no warning; a flagged set without a band model warns and says why;
      the 2-D LayeredDevsimBuilder and Stacked3DSpec.from_design pass the scaled set.
  P7  defaults off, three ways: (ii) the DEVSIM definition call log of a single SILICON_HURKX set hashes to
      the digest recorded on 66f6185; (i) same tree: the bare v0.14 BTBTParams and a flagged set at 300 K
      give the identical log; (iii) is tests/test_devsim_sg_flux.py (run with this change, audit note).

FAILS ON THE OLD CODE (66f6185, MEASURED by running this file there): collection error, AttributeError
(GERMANIUM_KAO_DIRECT and the other sets, btbt_at_temperature, tuples of sets do not exist). P7's digest is
an IDENTITY gate: the helper tests/_btbt_calllog.py run against 66f6185 RECORDED it (306 calls), so it holds
there by construction -- its sensitivity is the tuple-of-one adversarial.
Runtime ~70 s.
"""
import contextlib
import dataclasses
import math
import os
import sys
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

from dynameta.carriers import eq_registry as _R                              # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP                       # noqa: E402
from dynameta.carriers import tunneling as TN                                # noqa: E402
from dynameta.warning_types import AdvisoryWarning                           # noqa: E402

Q = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
VT = KB * 300.0 / Q
MU_N, MU_P = 0.135, 0.048
# printed SI values, hand-converted (tests/test_tunneling_presets.py PRINTED): Kao Table III via Jiao Table 1
SI_DIR = (1.35e22, 1.01e10, 2.0)
SI_IND = (3.29e16, 2.38e9, 2.5)


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _wipe():
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
        _R.clear(dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)


def _solve(abs_err=1e12, rel_err=1e-10):
    with _quiet():
        ds.solve(type="dc", absolute_error=abs_err, relative_error=rel_err, maximum_iterations=100)


# ------------------------------------------------------------------------------------------------
# P4: the 1-D Zener per set
# ------------------------------------------------------------------------------------------------
def _nodes(L, xj, W, h, junction_edge=None, h_out=2e-9, grow=1.15):
    """Explicit 1-D node positions: UNIFORM spacing h over [xj - m h, xj + m h], m = round(W/h) (the
    junction xj is a node), graded geometrically (x grow, capped at h_out) out to the contacts at 0 and L.
    Explicit nodes, not add_1d_mesh_line(ps=...): DEVSIM's line mesher puts the remainder of a non-integer
    interval / ps ratio into one edge next to the following line (MEASURED: a 0.128 nm edge left of the
    junction at ps = 0.2, 0.1 and 0.05 nm alike), and that edge sits on the field PEAK (root cause in the
    audit note). junction_edge (ADVERSARIAL only): replace the nodes inside (xj - junction_edge, xj) by one
    node at xj - junction_edge, i.e. a FIXED-length edge on the peak that does not shrink with h."""
    m = int(round(W / h))
    inner = xj + h * np.arange(-m, m + 1)

    def side(x0, sgn, stop):
        out, s = [], h
        while True:
            s = min(s * grow, h_out)
            nx = (out[-1] if out else x0) + sgn * s
            if (nx <= 0.5 * s) if sgn < 0 else (nx >= stop - 0.5 * s):
                return out + [0.0 if sgn < 0 else stop]
            out.append(nx)
    xs = np.concatenate([np.array(side(inner[0], -1.0, L)[::-1]), inner, np.array(side(inner[-1], 1.0, L))])
    if junction_edge:
        keep = (xs <= xj - junction_edge - 1e-15) | (xs >= xj - 1e-15)
        xs = np.sort(np.concatenate([xs[keep], [xj - junction_edge]]))
    return xs


def _zener(dev, btbt, N, ni, epsr, W, hj=0.2e-9, junction_edge=None):
    """Abrupt symmetric p+n+ diode, Na = Nd = N, junction node at L/2, uniform spacing hj over about
    [xj - W, xj + W] (_nodes)."""
    _wipe()
    mesh = dev + "_m"
    L = max(200e-9, 6.0 * W)
    xj = 0.5 * L
    xs = _nodes(L, xj, W, hj, junction_edge)
    ds.create_1d_mesh(mesh=mesh)
    for k, x in enumerate(xs):                       # ps larger than any gap: no subdivision
        ds.add_1d_mesh_line(mesh=mesh, pos=float(x), ps=1.0,
                            **({"tag": "pc"} if k == 0 else {"tag": "nc"} if k == xs.size - 1 else {}))
    ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
    ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="b", name="NetDoping", equation="ifelse(x < {}, {}, {})".format(xj, -N, N))
    BP.setup_bipolar_region(dev, "b", eps_static=epsr, n_dos_m3=2.8e25, n_i_m3=ni, mobility_n_m2Vs=MU_N,
                            mobility_p_m2Vs=MU_P, tau_n_s=1e-3, tau_p_s=1e-3, fd_enhancement=False, btbt=btbt)
    BP.setup_equilibrium_seed_models(dev, "b")
    for c in ("pc", "nc"):
        BP.setup_contact_ohmic_bipolar(dev, c)
    ds.node_model(device=dev, region="b", name="_seed", equation=BP.equilibrium_seed_psi_expr(None))
    for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region="b", name=v,
                           values=ds.get_node_model_values(device=dev, region="b", name=m))
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(dev, ceq)
    _solve(1e10, 1e-8)                                       # Poisson-only pre-solve
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(dev, ceq)
    _solve()


def _ramp(dev, v_from, v_to, steps=5):
    for v in np.linspace(v_from, v_to, steps + 1)[1:]:
        ds.set_parameter(device=dev, name="pc_bias", value=float(v))
        _solve()


def _analytic(sets, N, ni, epsr, Vr):
    """(J [A/m^2], Emax, delta_D, depletion width): q Int sum_s A_s E^P_s exp(-B_s/E) (-D) dx over the
    depletion profile E = Emax - (qN/eps)|x|, Emax^2 = (qN/eps)(V_bi + V_r - 2 V_t), flat quasi-Fermi levels
    split by V_r (Z1 of tests/test_devsim_tunneling.py, re-coded for eps, n_i and a sum of sets)."""
    from scipy.integrate import quad
    eps = epsr * EPS0
    vbi = 2.0 * VT * math.log(N / ni)
    emax = math.sqrt(Q * N / eps * (vbi + Vr - 2.0 * VT))
    a = Q * N / eps
    xn = emax / a

    def g(x):
        E = emax - a * x
        dpsi = emax * x - 0.5 * a * x * x
        n = ni * math.exp((dpsi - 0.5 * Vr) / VT)
        p = ni * math.exp((-0.5 * Vr - dpsi) / VT)
        D = (n * p - ni ** 2) / ((n + ni) * (p + ni))
        return sum(s.A_SI * E ** s.P * math.exp(-s.B_V_per_m / E) for s in sets) * (-D)
    J = 2.0 * Q * quad(g, 0.0, xn, limit=500, epsrel=1e-11)[0]
    return J, emax, J / (Q * MU_P * emax * ni), 2.0 * xn


GE = (16.0, 2.0e19)              # fixture eps_r, n_i [m^-3] (Ge-like; oracle uses the same)
INGAAS = (13.9, 6.3e17)          # fixture eps_r, n_i (InGaAs-like)
# name: (sets, N, (eps_r, n_i), reverse biases). MEASURED dev on the uniform explicit mesh at 0.2 / 0.1 nm (audit
# note), max over the biases: Ge direct 2.3e-3 / 7.5e-4, Ge indirect 1.6e-3 / 3.9e-4, Ge tuple 2.3e-3 / 7.5e-4,
# InGaAs 8.1e-4 / 2.0e-4 -- the 1 % quadrature allowance of Z1 holds with >= 4x margin ((k h)^2/12 <= 6e-4 here).
P4_CASES = {
    "ge_direct": ((TN.GERMANIUM_KAO_DIRECT,), 1e24, GE, (0.3, 0.6, 1.0)),
    "ge_indirect": ((TN.GERMANIUM_KAO_INDIRECT,), 1e24, GE, (0.6, 0.8, 1.0)),
    "ge_tuple": (TN.GERMANIUM_KAO, 1e24, GE, (0.3, 0.6, 1.0)),
    "ingaas_kane": ((TN.IN053GA047AS_KANE,), 3e23, INGAAS, (0.6, 0.8, 1.0)),
}


@pytest.mark.parametrize("case", list(P4_CASES))
def test_p4_zener_matches_the_analytic_estimate_per_set(case):
    sets, N, (epsr, ni), vrs = P4_CASES[case]
    dev = "tbp_" + case
    W = _analytic(sets, N, ni, epsr, max(vrs))[3]
    btbt = sets if len(sets) > 1 else sets[0]
    try:
        _zener(dev, btbt, N, ni, epsr, W)
        v, rows = 0.0, []
        for Vr in vrs:
            _ramp(dev, v, -Vr)
            v = -Vr
            rows.append((Vr, TN.btbt_generation_current(dev, "b")))
    finally:
        _wipe()
    for Vr, I in rows:
        J, _em, dD, _w = _analytic(sets, N, ni, epsr, Vr)
        assert J > 1e-2                                          # a real current (not a clamp floor)
        assert abs(I / J - 1.0) <= 0.01 + dD, (case, Vr, I, J, dD)
        # ADVERSARIAL (on the oracle): a 100x unit error, or P swapped 2 <-> 2.5, misses by far more
        bad_A = [dataclasses.replace(s, A_SI=s.A_SI / 100.0) for s in sets]
        bad_P = [dataclasses.replace(s, P=4.5 - s.P) for s in sets]
        assert abs(I / _analytic(bad_A, N, ni, epsr, Vr)[0] - 1.0) > 0.5
        assert abs(I / _analytic(bad_P, N, ni, epsr, Vr)[0] - 1.0) > 0.5


# The Si Kao sets peak within 1/k = (V_bi + V_r)/B = 0.14 nm (direct) / 0.55 nm (indirect) of the junction, so
# the edge-midpoint quadrature's error, ~(k h)^2/12 for a rate e^(-k|x|) whose kink sits at an edge midpoint
# (exactly: (kh/2) coth(kh/2) - 1), is NOT below 1 % at h = 0.1 nm. The gate is therefore a CONVERGENCE gate (as
# L1): observed order >= 1.7 on three uniform meshes and the Richardson limit within Z1's 1 % + delta_D of the
# analytic estimate. MEASURED dev at the three h (audit note): Si direct +1.017e-2 / +1.600e-3 / -5.50e-4
# (orders 1.98, 1.99; limit -1.27e-3; (kh)^2/12 = 1.03e-2 / 2.6e-3 / 6.4e-4); Si indirect -1.374e-3 /
# -3.782e-3 / -4.385e-3 (order 2.0; limit -4.59e-3, the generated-carrier D loss, delta_D = 5.5e-3).
P4C_CASES = {
    "si_direct": (TN.SILICON_KAO_DIRECT, 1e26, (0.05e-9, 0.025e-9, 0.0125e-9)),
    "si_indirect": (TN.SILICON_KAO_INDIRECT, 1e25, (0.1e-9, 0.05e-9, 0.025e-9)),
}
SI = (11.7, 1.0e16)


def _si_zener_current(prm, N, h, junction_edge=None, Vr=0.3):
    epsr, ni = SI
    W = math.ceil(_analytic((prm,), N, ni, epsr, 0.4)[3] / 0.2e-9) * 0.2e-9
    try:
        _zener("tbp_si_c", prm, N, ni, epsr, W, hj=h, junction_edge=junction_edge)
        _ramp("tbp_si_c", 0.0, -Vr)
        # a convergence study, under-resolved BY DESIGN on its coarse mesh (h = 0.05 nm Si direct) and on the
        # adversarial fixed junction edge: the resolution advisory is gated in tests/test_devsim_btbt_resolution.py
        return TN.btbt_generation_current("tbp_si_c", "b", check_resolution=False)
    finally:
        _wipe()


@pytest.mark.parametrize("case", list(P4C_CASES))
def test_p4c_si_kao_zener_converges_to_the_analytic_estimate(case):
    """GATE (root-caused 2026-11-06, audit note): on uniform meshes the Si Kao Zener current converges at second
    order to the analytic estimate. FAILS on the previous fixture (add_1d_mesh_line ps-mesh, a fixed 0.128 nm
    edge on the field peak): MEASURED Si direct (n_i 1e19) +7.9 / +15.1 / +17.1 % at 0.2 / 0.1 / 0.05 nm.
    ADVERSARIAL: that fixed-length junction edge put back into the uniform mesh pins the error at +11 % (below)."""
    prm, N, hs = P4C_CASES[case]
    epsr, ni = SI
    J, _em, dD, _w = _analytic((prm,), N, ni, epsr, 0.3)
    I = [_si_zener_current(prm, N, h) for h in hs]
    e1, e2 = I[0] - I[1], I[1] - I[2]
    order = math.log(abs(e1 / e2)) / math.log(2.0)
    assert e1 * e2 > 0 and order >= 1.7, (I, order)
    lim = I[2] + (I[2] - I[1]) / 3.0                           # Richardson, p = 2
    assert abs(lim / J - 1.0) <= 0.01 + dD, (lim / J - 1.0, dD)
    assert abs(I[2] / J - 1.0) <= 0.01 + dD                     # the finest mesh is inside the budget too
    if case == "si_direct":
        bad = [_si_zener_current(prm, N, h, junction_edge=0.12827e-9) for h in hs[1:]]
        assert all(b / J - 1.0 > 5.0 * (0.01 + dD) for b in bad), [b / J - 1.0 for b in bad]


def _slab(dev, btbt, *, options=None, n=8, L=8e-9):
    """1-D bipolar slab with no solve: models only (for frozen-state checks)."""
    _wipe()
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=L / n, tag="a")
    ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=L / n, tag="b")
    ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="r", tag1="a", tag2="b")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="r", name="NetDoping", equation="1e22")
    BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16, mobility_n_m2Vs=MU_N,
                            mobility_p_m2Vs=MU_P, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=False,
                            options=options, btbt=btbt)


def _frozen_state(dev, E0, seed=3):
    x = np.asarray(ds.get_node_model_values(device=dev, region="r", name="x"))
    rng = np.random.default_rng(seed)
    st = {"Potential": -E0 * x, "Electrons": 10 ** rng.uniform(13, 17, x.size),
          "Holes": 10 ** rng.uniform(13, 17, x.size)}
    for v, val in st.items():
        ds.set_node_values(device=dev, region="r", name=v, values=[float(a) for a in val])
    return x, st


def _numpy_rate(x, st, A, B, P, ni=1e16):
    E = np.sqrt((np.diff(st["Potential"]) / np.diff(x)) ** 2 + 1.0)
    n, p = st["Electrons"], st["Holes"]
    D = (n * p - ni ** 2) / ((n + ni) * (p + ni))
    return A * E ** P * np.exp(-B / E) * 0.5 * (D[1:] + D[:-1])


def test_p4s_si_kao_sets_installed_rate_equals_the_printed_numbers():
    """The Si Kao sets (and both as a tuple) installed: BTBT_U(_k) == numpy A E^P exp(-B/E) D(mean) with the
    PRINTED SI literals, rel 1e-12 (Z5's budget), at E = 5e8 V/m (where both rates are O(1e30))."""
    dev = "tbp_si"
    try:
        for btbt, names, lits in ((TN.SILICON_KAO_DIRECT, ("BTBT_U",), (SI_DIR,)),
                                  (TN.SILICON_KAO_INDIRECT, ("BTBT_U",), (SI_IND,)),
                                  ((TN.SILICON_KAO_DIRECT, TN.SILICON_KAO_INDIRECT), ("BTBT_U_0", "BTBT_U_1"),
                                   (SI_DIR, SI_IND))):
            _slab(dev, btbt)
            x, st = _frozen_state(dev, 5e8)
            for nm, (A, B, P) in zip(names, lits):
                u = np.asarray(ds.get_edge_model_values(device=dev, region="r", name=nm))
                ref = _numpy_rate(x, st, A, B, P)
                assert np.max(np.abs(ref)) > 1e25
                assert np.max(np.abs(u - ref)) <= 1e-12 * np.max(np.abs(ref)), nm
                # ADVERSARIAL: the cm^-1 unit reading (A / 1e4) is off by 1e4
                assert np.max(np.abs(u - ref / 1e4)) > 0.5 * np.max(np.abs(ref))
    finally:
        _wipe()


# ------------------------------------------------------------------------------------------------
# P5: a tuple of sets SUMS
# ------------------------------------------------------------------------------------------------
def test_p5_ge_tuple_equals_the_sum_of_separate_runs():
    N, (epsr, ni) = 1e24, GE
    W = _analytic(TN.GERMANIUM_KAO, N, ni, epsr, 1.0)[3]
    res = {}
    try:
        _zener("tbp_tup", TN.GERMANIUM_KAO, N, ni, epsr, W)
        _ramp("tbp_tup", 0.0, -1.0)
        st = {v: np.asarray(ds.get_node_model_values(device="tbp_tup", region="b", name=v))
              for v in ("Potential", "Electrons", "Holes")}
        res["tot"] = TN.btbt_generation_current("tbp_tup", "b")
        res["i0"] = TN.btbt_generation_current("tbp_tup", "b", index=0)
        res["i1"] = TN.btbt_generation_current("tbp_tup", "b", index=1)
        res["u"] = [np.asarray(ds.get_edge_model_values(device="tbp_tup", region="b", name=nm)).tobytes()
                    for nm in ("BTBT_U_0", "BTBT_U_1")]
        terms = BP.recombination_terms("tbp_tup", "b")["edge"]
        assert set(terms) == {"btbt_0", "btbt_1"} and all(pm is not None for _r, pm in terms.values())
        with pytest.raises(ValueError):
            TN.btbt_generation_current("tbp_tup", "b", index=2)
        for k, prm in enumerate(TN.GERMANIUM_KAO):
            dev = "tbp_one{}".format(k)
            _zener(dev, prm, N, ni, epsr, W)
            _ramp(dev, 0.0, -1.0)
            res["solved{}".format(k)] = TN.btbt_generation_current(dev, "b")
            for v, val in st.items():                        # the tuple run's state, frozen
                ds.set_node_values(device=dev, region="b", name=v, values=[float(a) for a in val])
            res["frozen{}".format(k)] = TN.btbt_generation_current(dev, "b")
            res["uf{}".format(k)] = np.asarray(ds.get_edge_model_values(device=dev, region="b",
                                                                        name="BTBT_U")).tobytes()
            assert "BTBT_U_0" not in ds.get_edge_model_list(device=dev, region="b")
    finally:
        _wipe()
    assert res["u"] == [res["uf0"], res["uf1"]]                 # term rates bit-equal to the single-set runs
    assert res["i0"] == res["frozen0"] and res["i1"] == res["frozen1"]
    assert res["tot"] == res["frozen0"] + res["frozen1"]        # EXACT (same summation order)
    assert res["i1"] / res["tot"] > 1e-3                        # ADVERSARIAL: dropping a term misses by > 1e-3
    assert res["tot"] > res["solved0"] and res["tot"] > res["solved1"]


def test_p5_fdt_holds_with_two_btbt_terms():
    """FDT S_I = 4 kT Re Y(f) on the D2a diode with TWO BTBT terms (a P = 2.5 and a P = 2 set; the test-only
    small-B sets that make BTBT a visible share at the FDT diode's ~1e7 V/m), 1 Hz - 1 GHz, budget 1e-4."""
    V = pytest.importorskip("validation.devsim_noise")
    from scipy.constants import e as QE, k as KBS
    sets = (TN.BTBTParams(A_SI=1.0e12, B_V_per_m=1.0e7, P=2.5), TN.BTBTParams(A_SI=1.0e14, B_V_per_m=2.0e7, P=2.0))
    ff = [1.0, 1e5, 1e7, 1e9]
    try:
        V.set_bias._v = {}
        V.bipolar_1d("fdt2b", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, btbt=sets)
        terms = BP.recombination_terms("fdt2b", "bulk")["edge"]
        assert set(terms) == {"btbt_0", "btbt_1"}
        sn = V.noise(ff)
        Y = V.ssac_Y(ff)
        T = V.ds.get_parameter(device="fdt2b", region="bulk", name="V_t") * QE / KBS
    finally:
        V.fresh()
        V.set_bias._v = {}
    assert np.max(np.abs(sn.S / (4 * KBS * T * Y.real) - 1)) < 1e-4


# ------------------------------------------------------------------------------------------------
# T: the opt-in scaling of B in a device and through the builders
# ------------------------------------------------------------------------------------------------
GAAS = (1.519, 0.5405e-3, 204.0)     # Vurgaftman Table I
INAS = (0.417, 0.276e-3, 93.0)       # Table III


def _eg_ingaas(T):
    v = [e0 - a * T * T / (T + b) for e0, a, b in (GAAS, INAS)]
    return 0.47 * v[0] + 0.53 * v[1] - 0.47 * 0.53 * 0.477          # Eq. (4.1), Table XIII


def test_t_scaled_set_in_a_350K_region_installs_the_scaled_B_and_does_not_warn():
    from dynameta.carriers.bipolar_options import BipolarOptions
    T = 350.0
    ratio = _eg_ingaas(T) / _eg_ingaas(300.0)                       # oracle (no band model needed here)
    flagged = dataclasses.replace(TN.IN053GA047AS_KANE, scale_B_with_gap=True)
    scaled = TN.btbt_at_temperature(flagged, T, ratio)
    dev = "tbp_T"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")                          # no advisory: the set IS at 350 K
            _slab(dev, scaled, options=BipolarOptions(T_K=T))
        x, st = _frozen_state(dev, 2e8)
        u = np.asarray(ds.get_edge_model_values(device=dev, region="r", name="BTBT_U"))
        B_T = flagged.B_V_per_m * ratio ** 1.5
        ref = _numpy_rate(x, st, flagged.A_SI, B_T, 2.0)
        assert np.max(np.abs(u - ref)) <= 1e-12 * np.max(np.abs(ref))
        # ADVERSARIAL: the unscaled B is off by far more (B/E ~ 2.8, dB/B = 3.6e-2 -> ~10 % in the rate)
        ref0 = _numpy_rate(x, st, flagged.A_SI, flagged.B_V_per_m, 2.0)
        assert np.max(np.abs(u - ref0)) > 1e-2 * np.max(np.abs(ref))
        # the flagged set NOT scaled (no band model / direct call) still warns, and says why
        with pytest.warns(AdvisoryWarning, match=r"T = 350 K.*BTBT parameters.*T_REF = 300.*scale_B_with_gap is "
                                                 r"set on 1 of them"):
            _slab(dev, flagged, options=BipolarOptions(T_K=T))
        # a set QUOTED at 350 K (fitted there) does not warn at 350 K, and does at 300 K
        at350 = dataclasses.replace(TN.SILICON_HURKX, T_K=T)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _slab(dev, at350, options=BipolarOptions(T_K=T))
        with pytest.warns(AdvisoryWarning, match=r"T = 300 K.*as at T = 350 K"):
            _slab(dev, at350)
    finally:
        _wipe()


def _ingaas_tr(btbt, T_K):
    from dynameta.carriers.bipolar_options import BipolarOptions
    from dynameta.constants import M_E
    from dynameta.materials import BandTemperature, TransportModel
    return TransportModel(n_bg_m3=1e24, eps_static=13.9, dos_mass_kg_of_n_m3=lambda n: np.full_like(
        np.asarray(n, dtype=float), 0.043 * M_E), band_gap_eV=_eg_ingaas(300.0), chi_eV=4.5,
        physics="bipolar_dd", mobility_m2Vs_of_n_m3=lambda n: np.full_like(np.asarray(n, dtype=float), MU_N),
        hole_mobility_m2Vs_of_n_m3=lambda n: np.full_like(np.asarray(n, dtype=float), MU_P), tau_srh_s=1e-3,
        n_i_m3=6.3e17, dos_mass_p_kg=0.05 * M_E, btbt=btbt,
        bipolar_options=BipolarOptions(T_K=T_K), band_temperature=BandTemperature("InGaAs_LM", 1.0))


class _Captured(Exception):
    pass


def test_t_builders_pass_the_scaled_set(monkeypatch):
    """The 2-D LayeredDevsimBuilder hands setup_bipolar_region B (E_g(350)/E_g(300))^1.5 (captured before any
    solve); Stacked3DSpec.from_design carries the same scaled set; an unflagged set passes as the SAME object."""
    T = 350.0
    flagged = dataclasses.replace(TN.IN053GA047AS_KANE, scale_B_with_gap=True)
    want = flagged.B_V_per_m * (_eg_ingaas(T) / _eg_ingaas(300.0)) ** 1.5
    got = {}
    real = BP.setup_bipolar_region

    def capture(*a, **kw):
        got["btbt"] = kw.get("btbt")
        raise _Captured()
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.sweep import BiasPoint
    for btbt in (flagged, TN.IN053GA047AS_KANE):
        reg = MaterialRegistry()
        reg.add(Material("air", ConstantOptical(1.0 + 0j)))
        reg.add(Material("cmetal", ConstantOptical(-50 + 5j), is_metal=True))
        reg.add(Material("InGaAs", ConstantOptical(12 + 0j), transport=_ingaas_tr(btbt, T)))
        d = Design(name="tbp_l", unit_cell=UnitCell.square(120e-9),
                   stack=Stack(layers=[Layer("s", 4e-9, "InGaAs")], superstrate_material="air",
                               substrate_material="air"),
                   electrodes=[Electrode("anode", "s", "x_lo", role="biased"),
                               Electrode("cathode", "s", "x_hi", role="ground", fixed_voltage_V=0.0)],
                   materials=reg, mesh_2d=Mesh2DSpec(x_spacing_feature_mid_m=2e-9, coarse_ps_m=2e-9))
        b = LayeredDevsimBuilder(d, mesh_name="tbp_lm", device_name="tbp_ld")
        monkeypatch.setattr(BP, "setup_bipolar_region", capture)
        try:
            with _quiet(), pytest.raises(_Captured):
                b.solve(BiasPoint({"anode": 0.0}, "v"))
        finally:
            monkeypatch.setattr(BP, "setup_bipolar_region", real)
            _wipe()
        if btbt is flagged:
            assert abs(got["btbt"].B_V_per_m / want - 1.0) <= 1e-14 and got["btbt"].A_SI == flagged.A_SI
            assert got["btbt"].T_K == T
        else:
            assert got["btbt"] is btbt
    # 3-D: from_design (devsim-free spec construction) on a bipolar body
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    from validation.carriers_3d_multisemi import _design
    layers = [("mirror", 70e-9, "Al-Nd"), ("lower_hfo2", 7e-9, "HfO2"), ("ito_bot", 12e-9, "ITO"),
              ("upper_hfo2", 7e-9, "HfO2"), ("patch", 50e-9, "air")]
    d3 = _design(layers)
    ito = d3.materials.get("ITO")
    body = dataclasses.replace(ito, name="InGaAsB", transport=_ingaas_tr(flagged, T))
    d3.materials.add(body)
    d3 = dataclasses.replace(d3, stack=dataclasses.replace(d3.stack, layers=[
        dataclasses.replace(L, background_material="InGaAsB") if L.name == "ito_bot" else L
        for L in d3.stack.layers]))
    spec = Stacked3DSpec.from_design(d3)
    assert abs(spec.btbt.B_V_per_m / want - 1.0) <= 1e-14 and spec.btbt.T_K == T


# ------------------------------------------------------------------------------------------------
# P7: defaults off
# ------------------------------------------------------------------------------------------------
# sha256 of tests/_btbt_calllog.record_call_log() (306 calls: a 1-D and a 2-D slab, btbt=SILICON_HURKX),
# RECORDED on 66f6185 (the v0.15 integration commit this work was cut from) by running the helper against
# that tree; this tree gives the same 306-call log (MEASURED: the two JSON logs compare equal).
# 2026-11-10 (v0.15.1): now the digest of the CANONICAL log (correctly rounded transcendentals: tests/_calllog.py,
# tests/_cr_libm.py), re-recorded on an export of 66f6185 with the current recorder + helpers copied in: the
# SAME value, raw and canonical, on Windows AND in WSL (glibc), = this tree's on both (no libm result of this log
# is a hard rounding case there); pinned canonically so that holds on any libm.
# docs/audit/2026-11-10-v0.15.1-digest-svd-fragility.md.
_BTBT_CALLLOG_DIGEST = "61543ffbfa156a6a7566fd20481455cd5438f6702ccdb71487b814edecbeeb7d"


def _calllog():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import _btbt_calllog as CL
    finally:
        sys.path.pop(0)
    return CL


def test_p7_single_set_definition_is_byte_identical_to_66f6185():
    """(ii) the recorded digest; (i) same-tree A/B: the bare v0.14 constructor and a FLAGGED set passed through
    btbt_for_material at 300 K (with a band model) give the identical log. ADVERSARIAL: the same set as a
    1-tuple (indexed names) changes the digest. CANONICAL log since 2026-11-10 (v0.15.1). STILL BITES
    (MEASURED on Windows and in WSL, canonical): A_SI = A_cgs (1e6 100^-P) instead of (A_cgs 1e6) 100^-P --
    mathematically equal -- moves A_SI by 1 ulp (4.0000000000000005e15 -> 4e15, 30 literals) and changes the
    digest; the field-magnitude string 'pow(ElectricField^2 + 1, 0.5)' (for '+ 1.0') changes it. NOT
    COVERED: 100.0 ** -P is Python's C pow, which no context can intercept (bitwise equal on UCRT and glibc
    here: the raw digests agree)."""
    CL = _calllog()
    _wipe()
    log = CL.record_call_log()
    assert len(log) == 306 and CL.digest(log) == _BTBT_CALLLOG_DIGEST
    bare = TN.BTBTParams(4000000000000000.5, 1.9e9, 2.5)
    flagged = dataclasses.replace(TN.SILICON_HURKX, scale_B_with_gap=True)
    via_hook = TN.btbt_for_material(flagged, _ingaas_tr(flagged, 300.0))
    assert via_hook.B_V_per_m == flagged.B_V_per_m and via_hook.T_K == 300.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert CL.record_call_log(bare) == log
        assert CL.record_call_log(via_hook) == log
    assert CL.digest(CL.record_call_log((TN.SILICON_HURKX,))) != _BTBT_CALLLOG_DIGEST
