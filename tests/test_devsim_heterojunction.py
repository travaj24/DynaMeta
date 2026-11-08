"""DEVSIM semiconductor/semiconductor HETEROINTERFACES (v0.14.0, workstream D1a).

Formulation + measured numbers: docs/audit/2026-11-03-devsim-heterojunctions.md; device factories and the
oracle helpers (independent Fermi-Dirac quadrature, Richardson law, band-diagram algebra):
validation/devsim_heterojunction.py.

Every gate here FAILS on v0.13.0: physics_heterojunction did not exist (ImportError), the 2-D builder
REFUSED any carrier-transporting semiconductor/semiconductor interface (NotImplementedError, audit regrade
7.1), Stacked3DSpec refused a drift-diffusion heterostack (ValueError), and the 2-D equilibrium builder put
every region at psi = 0 in its own frame (zero built-in potential -- the equilibrium gate's 1.4e-1 V
miss). DEVSIM keeps global state, so every device is built, measured and torn down inside ONE module
fixture; the gates then assert on plain numbers.
"""
import math

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

from validation.devsim_heterojunction import (                               # noqa: E402
    ALGAAS, DEC, DEV, GAAS, P_CELL, T_METAL, column, current_density, fd_eta, nc_m3, richardson,
    stack_design, transport)
from dynameta.constants import EPS0, Q_E, T_REF, V_T                          # noqa: E402


def _builder(design, tag, **kw):
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    return LayeredDevsimBuilder(design, mesh_name="hjm_" + tag, device_name="hjd_" + tag, **kw)


def _bias(v):
    from dynameta.sweep import BiasPoint
    return BiasPoint(label="v{:+.4f}".format(v), voltages={"anode": float(v)})


def _trap(y, x):
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


# ---------------------------------------------------------------- pure algebra (no device)
def test_richardson_constant_free_electron_value():
    from dynameta.carriers.physics_heterojunction import A_RICHARDSON_FREE, richardson_constant_Am2K2
    # 4 pi q m0 k^2 / h^3 = 1.20173e6 A m^-2 K^-2 (120.17 A cm^-2 K^-2); oracle h = exact SI value
    # 1.8e-9: the library h = 2 pi HBAR uses the CODATA-rounded HBAR literal (shared with N_c)
    assert A_RICHARDSON_FREE == pytest.approx(richardson(1.0), rel=1e-8)
    assert A_RICHARDSON_FREE == pytest.approx(1.20173e6, rel=1e-5)
    assert richardson_constant_Am2K2(0.067 * 9.1093837015e-31) == pytest.approx(0.067 * A_RICHARDSON_FREE)
    with pytest.raises(ValueError):
        richardson_constant_Am2K2(0.0)


def test_anderson_alignment_algebra_and_refusal():
    from dynameta.carriers.physics_heterojunction import anderson_alignment
    al = anderson_alignment({"a": 0.1, "b": -0.05}, {"a": 4.07, "b": 3.827}, "a")
    assert al["a"] == (0.1, 0.0)
    assert al["b"][0] == pytest.approx(0.1 + 0.243)                 # E_c(b) higher by chi_a - chi_b
    assert al["b"][1] == pytest.approx(0.1 + 0.243 + 0.05)          # s(b) = Phi_c(b) - own(b)
    with pytest.raises(ValueError, match="no neutral default"):
        anderson_alignment({"a": 0.0, "b": 0.0}, {"a": 4.0, "b": None}, "a")


def test_closed_form_delta_node_model_matches_quadrature():
    """The DEVSIM node model of the FD activity L = ln u + Delta(u) (closed form, arctan by half-angle
    reduction) against physics_bipolar_dd.fd_potential_shift (96-point Gauss-Legendre quadrature of the
    same g fit), u over 1e-8 .. 1e4, and its derivative model g(u)/n against a finite difference."""
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    from dynameta.carriers.physics_heterojunction import setup_log_activity
    u = np.array([1e-8, 1e-4, 1e-2, 0.14, 1.0, 7.7, 23.8, 77.0, 136.0, 1e3, 1e4])
    m, dv = "hj_cf_m", "hj_cf_d"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=0.1, tag="a")
    ds.add_1d_mesh_line(mesh=m, pos=1.0, ps=0.1, tag="b")
    ds.add_1d_region(mesh=m, material="x", region="r", tag1="a", tag2="b")
    ds.add_1d_contact(mesh=m, name="ca", tag="a", material="metal")
    ds.add_1d_contact(mesh=m, name="cb", tag="b", material="metal")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dv)
    try:
        nn = len(ds.get_node_model_values(device=dv, region="r", name="x"))
        nc = 4.35e23
        ds.set_parameter(device=dv, region="r", name="N_c", value=nc)
        ds.node_solution(device=dv, region="r", name="Electrons")
        vals = np.resize(u * nc, nn)
        ds.set_node_values(device=dv, region="r", name="Electrons", values=[float(v) for v in vals])
        name = setup_log_activity(dv, "r", "Electrons", "N_c", fd=True)
        L = np.asarray(ds.get_node_model_values(device=dv, region="r", name=name))
        dL = np.asarray(ds.get_node_model_values(device=dv, region="r", name=name + ":Electrons"))
        ref = np.log(vals / nc) + fd_potential_shift(vals / nc)
        # MEASURED 2026-11-03: max |L - ref| = 2.7e-12 (at u = 1e4 the two agree to 13 digits of 2e2)
        assert np.max(np.abs(L - ref)) < 1e-10
        # derivative model vs a centred difference of the quadrature form
        h = 1e-6
        fd = (np.log(vals * (1 + h) / nc) + fd_potential_shift(vals * (1 + h) / nc)
              - np.log(vals * (1 - h) / nc) - fd_potential_shift(vals * (1 - h) / nc)) / (2 * h * vals)
        assert np.max(np.abs(dL / fd - 1.0)) < 1e-6
    finally:
        ds.delete_device(device=dv)
        ds.delete_mesh(mesh=m)


# ---------------------------------------------------------------- GATE B / D: abrupt N-AlGaAs / p-GaAs
NA = ND = 1.0e23                                                   # 1e17 cm^-3 each side


@pytest.fixture(scope="module")
def pn_equilibrium():
    d = stack_design([("pgaas", "GaAs_p", 400e-9, transport(GAAS, n_bg=NA, physics="bipolar_dd",
                                                            acceptor=True)),
                      ("nalgaas", "AlGaAs_n", 400e-9, transport(ALGAAS, n_bg=ND, physics="bipolar_dd"))])
    b = _builder(d, "pn")
    try:
        b.solve(_bias(0.0), rel_tol=1e-8)
        out = {"frames": b.band_frames(), "ifaces": b.heterointerfaces(),
               "j0": current_density(b, None)}
        for r in ("pgaas", "nalgaas"):
            for nm in ("Potential", "Electrons", "Holes"):
                out[(r, nm)] = column(b, r, nm)
        fwd = {}
        for v in (0.7, 0.8):
            b.solve(_bias(v), rel_tol=1e-8)
            fwd[v] = (current_density(b, None, "cathode"), current_density(b, None, "anode"))
        out["fwd"] = fwd
    finally:
        b.teardown()
    return out


def test_gate_b_wiring_and_auto_model(pn_equilibrium):
    assert pn_equilibrium["ifaces"] == {"pgaas__nalgaas": "thermionic"}     # offsets -> auto picks TE
    fr = pn_equilibrium["frames"]
    assert fr["nalgaas"][1] == 0.0                                          # the ground's region = reference
    assert fr["nalgaas"][0] - fr["pgaas"][0] == pytest.approx(DEC, abs=1e-12)


def test_gate_b_built_in_potential_matches_band_diagram(pn_equilibrium):
    """q V_bi = W_p - W_n, W = chi + (E_c - E_F): W_n = chi_AlGaAs - kT eta(N_D/N_c), W_p = chi_GaAs + E_g
    + kT eta(N_A/N_v), eta from DIRECT Fermi-Dirac quadrature. MEASURED: 1.502262 V solved vs 1.502232
    analytic (+3.1e-5 V -- the rational g fit's Delta vs exact FD at u = 0.14 / 0.011). Tolerance 2e-4 V."""
    _, psip = pn_equilibrium[("pgaas", "Potential")]
    _, psin = pn_equilibrium[("nalgaas", "Potential")]
    eta_n = fd_eta(ND / nc_m3(ALGAAS["me"]))
    eta_p = fd_eta(NA / nc_m3(GAAS["mh"]))
    vbi = (GAAS["chi"] + GAAS["eg"] + eta_p * V_T) - (ALGAAS["chi"] - eta_n * V_T)
    assert 1.45 < vbi < 1.55
    assert psin[-1] - psip[0] == pytest.approx(vbi, abs=2e-4)


def test_gate_b_depletion_widths_and_potential_split(pn_equilibrium):
    """Anderson depletion approximation for an abrupt heterojunction with eps_n != eps_p:
    x_n = sqrt(2 eps_n eps_p N_A (V_bi - 2kT/q) / (q N_D (eps_n N_D + eps_p N_A))), x_p = x_n N_D/N_A;
    the 2kT/q is the standard majority-carrier-tail correction (one kT/q per side). Widths are measured as
    CHARGE-equivalent widths int (N - majority + minority) dz / N. MEASURED: x_n = 99.815 nm vs 99.835 nm
    (-0.02 %); V_n/V_p = 1.0732 vs eps_p N_A/(eps_n N_D) = 1.0750 (-0.17 %), 1.0759 with the per-side
    kT/q tail removed (-0.08 %). Tolerances 1 % (widths) and 0.5 % (split) -- the residual of the tail
    correction is O((kT/qV_bi)^2) ~ 3e-4 and the FD majority statistics shift it by ~1e-3."""
    zp, psip = pn_equilibrium[("pgaas", "Potential")]
    zn, psin = pn_equilibrium[("nalgaas", "Potential")]
    _, n_p = pn_equilibrium[("pgaas", "Electrons")]
    _, p_p = pn_equilibrium[("pgaas", "Holes")]
    _, n_n = pn_equilibrium[("nalgaas", "Electrons")]
    _, p_n = pn_equilibrium[("nalgaas", "Holes")]
    vbi = psin[-1] - psip[0]
    en, ep = ALGAAS["eps"] * EPS0, GAAS["eps"] * EPS0
    xn = math.sqrt(2 * en * ep * NA * (vbi - 2 * V_T) / (Q_E * ND * (en * ND + ep * NA)))
    xp = xn * ND / NA
    xn_s = _trap(ND - n_n + p_n, zn) / ND
    xp_s = _trap(NA - p_p + n_p, zp) / NA
    assert xn_s == pytest.approx(xn, rel=1e-2)
    assert xp_s == pytest.approx(xp, rel=1e-2)
    vn, vp = psin[-1] - psin[0], psip[-1] - psip[0]
    assert vn + vp == pytest.approx(vbi, abs=1e-12)                       # psi continuous at the interface
    assert vn / vp == pytest.approx(ep * NA / (en * ND), rel=5e-3)
    assert (vn - V_T) / (vp - V_T) == pytest.approx(ep * NA / (en * ND), rel=2e-3)


def test_gate_b_band_offsets_from_solved_densities(pn_equilibrium):
    """The conduction-band SPIKE and valence step at the interface, read from the SOLVED carrier densities
    alone (E_c - E_F = -kT eta(n/N_c), E_v - E_F = kT eta(p/N_v), exact FD quadrature; E_F = 0 at
    equilibrium) on the two node copies of the interface: they must equal the Anderson offsets
    dE_c = chi_GaAs - chi_AlGaAs = 0.2431 eV and dE_v = 0.1309 eV. MEASURED: dE_c to 4.8e-13 eV. This is
    the interface carrier equation's detailed balance -- the potential-only v0.13.0 wiring had no such
    relation (and was refused)."""
    _, n_p = pn_equilibrium[("pgaas", "Electrons")]
    _, n_n = pn_equilibrium[("nalgaas", "Electrons")]
    _, p_p = pn_equilibrium[("pgaas", "Holes")]
    _, p_n = pn_equilibrium[("nalgaas", "Holes")]
    ec_gaas = -V_T * fd_eta(n_p[-1] / nc_m3(GAAS["me"]))
    ec_algaas = -V_T * fd_eta(n_n[0] / nc_m3(ALGAAS["me"]))
    ev_gaas = V_T * fd_eta(p_p[-1] / nc_m3(GAAS["mh"]))
    ev_algaas = V_T * fd_eta(p_n[0] / nc_m3(ALGAAS["mh"]))
    assert ec_algaas - ec_gaas == pytest.approx(DEC, abs=1e-6)
    assert ev_gaas - ev_algaas == pytest.approx(DEV, abs=1e-6)


def test_gate_b_flat_fermi_level_at_zero_bias(pn_equilibrium):
    """Zero bias: the electron and hole quasi-Fermi levels are flat at 0 THROUGH both regions and across the
    interface (exact FD quadrature on the solution). This is the frame-consistency gate: a wrong contact
    re-reference, a wrong Phi_c or an interface activity inconsistent with the bulk shows up here as an
    offset of tens of mV. Tolerance 2e-4 eV (the g fit vs exact FD; MEASURED max |E_F| ~ 3e-5 eV)."""
    fr = pn_equilibrium["frames"]
    for r, mat in (("pgaas", GAAS), ("nalgaas", ALGAAS)):
        _, psi = pn_equilibrium[(r, "Potential")]
        _, n = pn_equilibrium[(r, "Electrons")]
        _, p = pn_equilibrium[(r, "Holes")]
        ec = fr[r][0] - psi
        ev = ec - mat["eg"]
        for k in range(0, len(psi), max(1, len(psi) // 25)):
            # a quasi-Fermi level is only DEFINED to round-off where its carrier exists: below 1 m^-3
            # (1e-6 cm^-3; minority holes in n-AlGaAs sit at ~5e-5 m^-3) it is set by the absolute
            # Newton floor, not by physics.
            if n[k] > 1.0:
                assert abs(ec[k] + V_T * fd_eta(n[k] / nc_m3(mat["me"]))) < 2e-4, (r, k, "n")
            if p[k] > 1.0:
                assert abs(ev[k] - V_T * fd_eta(p[k] / nc_m3(mat["mh"]))) < 2e-4, (r, k, "p")


def test_gate_b_forward_current_conserved_and_rectifying(pn_equilibrium):
    """Forward bias: carriers CROSS the heterointerface (v0.13.0 refused the device; a potential-only
    interface would block them: ~0 current) and the two terminal currents balance."""
    j0 = pn_equilibrium["j0"]
    for v, (jc, ja) in pn_equilibrium["fwd"].items():
        assert abs(jc + ja) < 1e-6 * abs(jc) + 1e-3
        assert abs(jc) > 300.0 * abs(j0)      # j0 = the FD-SG discretization floor (~4e-4 A/m^2)
    # electron injection into the p-GaAs base (n_i^2/N_A = 46 m^-3; hole injection into the wide-gap
    # emitter is ~exp(-dE_g/kT) smaller), SHORT base: J = q D_n (n_i^2/N_A) exp(qV/kT) / W_b with the
    # neutral width W_b = 400 nm - x_p(V) (depletion approximation). MEASURED 2026-11-03: see the note.
    vbi = 1.50226
    en, ep = ALGAAS["eps"] * EPS0, GAAS["eps"] * EPS0
    ni2 = nc_m3(GAAS["me"]) * nc_m3(GAAS["mh"]) * math.exp(-GAAS["eg"] / V_T)
    for v, (jc, _ja) in pn_equilibrium["fwd"].items():
        xn = math.sqrt(2 * en * ep * NA * (vbi - v - 2 * V_T) / (Q_E * ND * (en * ND + ep * NA)))
        wb = 400e-9 - xn * ND / NA
        j_ideal = Q_E * GAAS["mu_n"] * V_T * ni2 / NA * math.exp(v / V_T) / wb
        assert abs(jc) == pytest.approx(j_ideal, rel=0.15), (v, jc, j_ideal)
    r = pn_equilibrium["fwd"][0.8][0] / pn_equilibrium["fwd"][0.7][0]
    assert r == pytest.approx(math.exp(0.1 / V_T), rel=0.15)


def test_gate_d_charge_neutrality_and_gauss_law_at_interface(pn_equilibrium):
    """No interface charge: the total space charge of the device integrates to zero (N_D x_n = N_A x_p),
    and the discrete Gauss law at the interface node -- D(right edge) - D(left edge) = the charge of the
    two half-boxes -- closes, i.e. the CONTINUUM D-field jump is zero. MEASURED: neutrality 1.3e-11 of the
    depletion charge; Gauss residual ~1e-9 of the peak D."""
    zp, psip = pn_equilibrium[("pgaas", "Potential")]
    zn, psin = pn_equilibrium[("nalgaas", "Potential")]
    _, n_p = pn_equilibrium[("pgaas", "Electrons")]
    _, p_p = pn_equilibrium[("pgaas", "Holes")]
    _, n_n = pn_equilibrium[("nalgaas", "Electrons")]
    _, p_n = pn_equilibrium[("nalgaas", "Holes")]
    rho_p = Q_E * (p_p - n_p - NA)
    rho_n = Q_E * (p_n - n_n + ND)
    q_total = _trap(rho_p, zp) + _trap(rho_n, zn)
    assert abs(q_total) < 1e-8 * Q_E * ND * 1e-7
    ep, en = GAAS["eps"] * EPS0, ALGAAS["eps"] * EPS0
    d_left = -ep * (psip[-1] - psip[-2]) / (zp[-1] - zp[-2])           # D = -eps dpsi/dz on the last p edge
    d_right = -en * (psin[1] - psin[0]) / (zn[1] - zn[0])              # first n edge
    box = rho_p[-1] * (zp[-1] - zp[-2]) / 2 + rho_n[0] * (zn[1] - zn[0]) / 2
    d_peak = max(abs(d_left), abs(d_right))
    assert d_peak > 1e-3                                               # a real junction field (C/m^2)
    assert abs((d_right - d_left) - box) < 1e-6 * d_peak


# ---------------------------------------------------------------- GATE A: homojunction limit
HJ_H = 1e-9                                                        # uniform 1 nm mesh -> identical nodes


@pytest.fixture(scope="module")
def homojunction():
    """The SAME GaAs p-n junction two ways on the same uniform 1 nm node set: (i) the existing path -- ONE
    bipolar layer with an in-region net_doping_expr (junction node at NetDoping = 0, i.e. the average of
    its two half-boxes for N_A = N_D); (ii) two layers of the same crystal (p, n materials with identical
    band parameters) joined by the new interface ('auto' -> continuous: zero offsets)."""
    zj = T_METAL + 400e-9
    expr = "ifelse(y < {a:.10e}, -{na:.6e}, ifelse(y > {b:.10e}, {nd:.6e}, 0.0))".format(
        a=zj - HJ_H / 4, b=zj + HJ_H / 4, na=NA, nd=ND)
    one = stack_design([("pn", "GaAs_pn", 800e-9, transport(GAAS, n_bg=ND, physics="bipolar_dd",
                                                            net_doping_expr=expr))],
                       ps_if=HJ_H, coarse=HJ_H, zone=100e-9)
    two = stack_design([("pgaas", "GaAs_p", 400e-9, transport(GAAS, n_bg=NA, physics="bipolar_dd",
                                                              acceptor=True)),
                        ("ngaas", "GaAs_n", 400e-9, transport(GAAS, n_bg=ND, physics="bipolar_dd"))],
                       ps_if=HJ_H, coarse=HJ_H, zone=100e-9)
    out = {}
    for tag, d in (("one", one), ("two", two)):
        b = _builder(d, "homo_" + tag)
        try:
            res = {}
            for v in (0.0, 0.7):
                b.solve(_bias(v), rel_tol=1e-10)
                cols = {}
                regs = ["pn"] if tag == "one" else ["pgaas", "ngaas"]
                for nm in ("Potential", "Electrons", "Holes"):
                    zs, vs = zip(*(column(b, r, nm) for r in regs))
                    z = np.concatenate(zs)
                    val = np.concatenate(vs)
                    _, keep = np.unique(np.round(z / 1e-13), return_index=True)   # drop the duplicate copy
                    cols[nm] = (z[keep], val[keep])
                res[v] = (cols, current_density(b, None))
            out[tag] = res
            if tag == "two":
                out["ifaces"] = b.heterointerfaces()
                out["frames"] = b.band_frames()
        finally:
            b.teardown()
    return out


def test_gate_a_homojunction_limit_reproduces_single_region_solve(homojunction):
    """Zero offsets + the same crystal: the two-region device reproduces the single-region in-region
    junction to SOLVER tolerance (not bit-identical: the assembly order of the split interface node
    differs). Same 801-node column. MEASURED 2026-11-03: potential to 1.1e-14 V, carrier densities to
    4.4e-13 relative, forward (0.7 V) terminal current to 1.9e-16 -- i.e. round-off; gated at 1e-9 V /
    1e-8."""
    assert homojunction["ifaces"] == {"pgaas__ngaas": "continuous"}
    assert all(s == 0.0 for (_phi, s) in homojunction["frames"].values())   # same frame: no shift at all
    for v in (0.0, 0.7):
        c1, j1 = homojunction["one"][v]
        c2, j2 = homojunction["two"][v]
        z1, psi1 = c1["Potential"]
        z2, psi2 = c2["Potential"]
        assert z1.shape == z2.shape and np.max(np.abs(z1 - z2)) < 1e-15
        assert np.max(np.abs(psi1 - psi2)) < 1e-9                  # MEASURED 1.1e-14 V
        for nm in ("Electrons", "Holes"):
            a, bb = c1[nm][1], c2[nm][1]
            assert np.max(np.abs(a - bb) / np.maximum(np.abs(a), 1.0)) < 1e-8, nm   # MEASURED 4.4e-13
        if v > 0:
            assert j2 == pytest.approx(j1, rel=1e-8)                 # MEASURED 1.9e-16



# ---------------------------------------------------------------- GATE C / E: isotype n-N thermionic emission
N_ISO = 1.0e23
V_LR = 5e-4                                                        # linear-response probe (+-0.5 mV)
# (tag, top-layer material): the real Al0.3Ga0.7As, then GaAs-mass layers with smaller affinity steps
ISO_CASES = [("algaas", ALGAAS)] + [("x{:03d}".format(int(round(de * 1000))), dict(GAAS, chi=GAAS["chi"] - de))
                                    for de in (0.10, 0.03, 0.0)]


@pytest.fixture(scope="module")
def isotype():
    """n-GaAs (bottom, 400 nm) / N-top (400 nm), both 1e17 cm^-3, unipolar drift-diffusion, under both
    interface models; linear-response conductance from J(+0.5 mV) - J(-0.5 mV), plus the zero-bias
    profile (identical for both models: equilibrium carries no current)."""
    out = {}
    for tag, top in ISO_CASES:
        for model, a_scale in (("thermionic", 1.0), ("continuous", 1.0), ("thermionic", 1e6)):
            if a_scale != 1.0 and tag not in ("algaas", "x000"):
                continue
            d = stack_design([("ngaas", "GaAs_n", 400e-9,
                               transport(GAAS, n_bg=N_ISO, physics="drift_diffusion",
                                         richardson_n=richardson(GAAS["me"]) * a_scale)),
                              ("ntop", "Top_" + tag, 400e-9,
                               transport(top, n_bg=N_ISO, physics="drift_diffusion",
                                         richardson_n=richardson(top["me"]) * a_scale))])
            b = _builder(d, "iso_{}_{}_{:g}".format(tag, model[0], a_scale), heterointerface=model)
            try:
                b.solve(_bias(0.0), rel_tol=1e-8)
                prof = {r: (column(b, r, "Potential"), column(b, r, "Electrons")) for r in ("ngaas", "ntop")}
                j0 = current_density(b, None, "cathode")
                b.solve(_bias(+V_LR), rel_tol=1e-8)
                jplus = current_density(b, None, "cathode")
                b.solve(_bias(-V_LR), rel_tol=1e-8)
                jminus = current_density(b, None, "cathode")
                out[(tag, model, a_scale)] = dict(prof=prof, g=abs(jplus - jminus) / (2 * V_LR), j0=j0,
                                                  ifaces=b.heterointerfaces())
            finally:
                b.teardown()
    return out


def _iso_oracle(prof, top, a_scale=1.0):
    """Linear response of the series device, from the ZERO-BIAS solution only: the bulk drift-diffusion
    resistance int dz / (q mu n_eq(z)) of both layers (J = mu n grad E_Fn holds exactly for the FD
    model) plus the thermionic-emission interface resistance kT / (q J_s), J_s = A* T^2 exp(-phi_B/kT),
    A* of the barrier side (Richardson law, h exact SI) and phi_B = E_c,top - E_F read from the solved
    electron density at the barrier-side interface node (exact FD quadrature). Returns
    (R_bulk, R_TE, phi_B)."""
    (zg, _), (_, ng) = prof["ngaas"]
    (zt, _), (_, nt) = prof["ntop"]
    r_bulk = (_trap(1.0 / (Q_E * GAAS["mu_n"] * ng), zg) + _trap(1.0 / (Q_E * top["mu_n"] * nt), zt))
    # barrier side = the higher conduction edge at the interface = the larger E_c - E_F of the two copies
    phi_g = -V_T * fd_eta(ng[-1] / nc_m3(GAAS["me"]))
    phi_t = -V_T * fd_eta(nt[0] / nc_m3(top["me"]))
    if abs(phi_t - phi_g) < 1e-9:
        phi_b, a = phi_t, min(richardson(top["me"]), richardson(GAAS["me"]))
    elif phi_t > phi_g:
        phi_b, a = phi_t, richardson(top["me"])
    else:
        phi_b, a = phi_g, richardson(GAAS["me"])
    js = a * a_scale * T_REF ** 2 * math.exp(-phi_b / V_T)
    return r_bulk, V_T / js, phi_b


def test_gate_c_isotype_thermionic_emission_matches_richardson_law(isotype):
    """n-GaAs / N-Al0.3Ga0.7As: the low-bias conductance equals the series combination of the bulk
    drift-diffusion resistance and the Richardson-law interface resistance kT/(q A* T^2 e^{-phi_B/kT})
    (the linearisation of J = A* T^2 e^{-phi_B/kT}(e^{qV/kT} - 1) at the interface), phi_B from the SOLVED
    barrier. MEASURED 2026-11-03: phi_B = 0.2177 eV, R_TE = 1.181e-8 vs R_bulk = 1.05e-9 Ohm m^2 (the
    interface is 92 % of the resistance), G_TE / G_oracle - 1 = +9.4e-5. The CONTINUOUS-quasi-Fermi model
    has no interface resistance and misses the same oracle by 12x (G_cont R_bulk = 1.001) -- the
    discriminating gate (a naive continuous-QFL heterojunction cannot pass it)."""
    r = isotype[("algaas", "thermionic", 1.0)]
    assert r["ifaces"] == {"ngaas__ntop": "thermionic"}
    r_bulk, r_te, phi_b = _iso_oracle(r["prof"], ALGAAS)
    assert 0.15 < phi_b < DEC + 0.01                                  # a real spike, below the bare offset
    assert r_te > 5.0 * r_bulk                                        # interface-limited: a sharp test
    assert r["g"] == pytest.approx(1.0 / (r_bulk + r_te), rel=2e-3)    # MEASURED +9.4e-5
    c = isotype[("algaas", "continuous", 1.0)]
    assert c["g"] > 5.0 / (r_bulk + r_te)                             # continuous QFL misses by > 5x
    assert c["g"] == pytest.approx(1.0 / r_bulk, rel=2e-2)           # ... and is the bare bulk resistor


def test_gate_e_continuous_and_thermionic_converge_as_the_barrier_vanishes(isotype):
    """For every offset (0.243, 0.10, 0.03, 0 eV) the two models share the zero-bias solution (equilibrium
    carries no current), so 1/G_TE - 1/G_cont must be EXACTLY the Richardson interface resistance
    kT/(q J_s) -- which shrinks as the barrier vanishes (exp(-phi_B/kT)) and vanishes as A* -> infinity:
    with A* x 1e6 the two models agree to < 1e-4 even across the 0.243 eV AlGaAs step. MEASURED: G_TE/G_cont
    = 0.082 / 0.285 / 0.688 / 0.804 at phi_B = 0.218 / 0.098 / 0.052 / 0.036 eV, the identity to <= 2.4e-3,
    A* x 1e6: -1.1e-5 (AlGaAs) and -2.4e-7 (zero offset)."""
    ratios = []
    for tag, top in ISO_CASES:
        te, ct = isotype[(tag, "thermionic", 1.0)], isotype[(tag, "continuous", 1.0)]
        _r_bulk, r_te, _phi = _iso_oracle(te["prof"], top)
        assert (1.0 / te["g"] - 1.0 / ct["g"]) == pytest.approx(r_te, rel=1e-2), tag   # MEASURED <= 2.4e-3
        ratios.append(te["g"] / ct["g"])
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios      # monotone toward 1 as dE_c -> 0
    for tag in ("algaas", "x000"):
        big, ct = isotype[(tag, "thermionic", 1e6)], isotype[(tag, "continuous", 1.0)]
        assert big["g"] == pytest.approx(ct["g"], rel=1e-4), tag


# ---------------------------------------------------------------- the 2-D equilibrium built-in potential
EQ_NP, EQ_NN = 1.0e24, 1.0e22                # n+ GaAs (u = 2.3, eta ~ +1.2) | n AlGaAs (non-degenerate)


@pytest.fixture(scope="module")
def equilibrium_stack():
    layers = [("ngaas", "GaAs_np", 300e-9, transport(GAAS, n_bg=EQ_NP, physics="equilibrium")),
              ("nalgaas", "AlGaAs_n", 300e-9, transport(ALGAAS, n_bg=EQ_NN, physics="equilibrium"))]
    b = _builder(stack_design(layers), "eq")
    try:
        b.solve(_bias(0.0), rel_tol=1e-10)    # Poisson-only: converge the charge, not just the update
        return {ln: (column(b, ln, "Potential"), column(b, ln, "Electrons")) for ln, *_ in layers}
    finally:
        b.teardown()


# Not @slow since 2026-11-06: the "~8 min/region" DEVSIM simplify() of the F_1/2 closure is replayed from
# its stored result since 2026-11-03 (physics_equilibrium); MEASURED 0.8 s on the Windows dev box, 2.4 s
# under WSL (ci.yml solvers-devsim).
def test_equilibrium_2d_stack_carries_its_built_in_potential(equilibrium_stack):
    """audit S1-9's 2-D sibling: adjacent EQUILIBRIUM semiconductors now sit in one Anderson frame, so a
    neutral bulk of region r sits at psi_bulk(r) = const - chi_r/q + eta_r kT/q (eta = (E_F - E_c)/kT by
    exact Fermi-Dirac quadrature), i.e. psi(AlGaAs) - psi(GaAs) = dE_c/q + (eta_A - eta_G) kT/q -- both the
    affinity step and the doping (Fermi-level) step. v0.13.0 returned 0 (every region neutral at psi = 0
    in its own frame). Tolerance 2e-3 V: the Aymerich-Humet F_1/2 fit of the equilibrium closure against
    exact FD near eta ~ +1 (the n+ side)."""
    (zg, psig), (_, ng) = equilibrium_stack["ngaas"]
    (za, psia), (_, na) = equilibrium_stack["nalgaas"]
    vbi = DEC + V_T * (fd_eta(EQ_NN / nc_m3(ALGAAS["me"])) - fd_eta(EQ_NP / nc_m3(GAAS["me"])))
    assert abs(vbi) > 0.05
    print("[D1a equilibrium gate] V_bi solved {:.6f} V, oracle {:.6f} V".format(psia[-1] - psig[0], vbi))
    assert psia[-1] - psig[0] == pytest.approx(vbi, abs=2e-3)
    # electrons spill from the high-E_c AlGaAs into the GaAs: depletion on the AlGaAs side. The device is
    # NOT exactly neutral -- the 1e16 cm^-3 AlGaAs (Debye length 41 nm) still carries a residual field at its
    # top contact 300 nm from the interface (MEASURED net charge 3.7e-3 of the depletion charge, ~e^-5) --
    # so the gate is the discrete GAUSS LAW over the whole stack: the space charge of all non-contact nodes
    # (box weights; the interface node's two half-boxes) equals D(top contact edge) - D(bottom contact edge)
    # with D continuous across the heterointerface (no interface charge).
    rho_g = Q_E * (EQ_NP - ng)
    rho_a = Q_E * (EQ_NN - na)
    q_int = (_trap(rho_g, zg) - rho_g[0] * (zg[1] - zg[0]) / 2
             + _trap(rho_a, za) - rho_a[-1] * (za[-1] - za[-2]) / 2)
    d_bot = -GAAS["eps"] * EPS0 * (psig[1] - psig[0]) / (zg[1] - zg[0])
    d_top = -ALGAAS["eps"] * EPS0 * (psia[-1] - psia[-2]) / (za[-1] - za[-2])
    q_dep = Q_E * _trap(np.abs(na - EQ_NN), za)
    print("[D1a equilibrium gate] net charge / depletion charge = {:.2e}; Gauss residual / depletion "
          "charge = {:.2e}".format(q_int / q_dep, (q_int - (d_top - d_bot)) / q_dep))
    assert abs(q_int - (d_top - d_bot)) < 1e-6 * q_dep
    assert abs(q_int) < 1e-2 * q_dep                                    # neutral up to the e^-5 contact tail
    assert na[0] < EQ_NN


# ---------------------------------------------------------------- refusals that remain (named, not silent)
def _mixed_or_bad(kind):
    if kind == "mixed":
        layers = [("ngaas", "GaAs_n", 200e-9, transport(GAAS, n_bg=1e23, physics="drift_diffusion")),
                  ("pgaas", "GaAs_p", 200e-9, transport(GAAS, n_bg=1e23, physics="bipolar_dd", acceptor=True))]
    else:                                     # band_gap_eV inconsistent with n_i (n_i from a 0.1 eV wider gap)
        bad = dict(ALGAAS, eg=ALGAAS["eg"] + 0.1)
        tm = transport(bad, n_bg=1e23, physics="bipolar_dd")
        tm.band_gap_eV = ALGAAS["eg"]
        layers = [("pgaas", "GaAs_p", 200e-9, transport(GAAS, n_bg=1e23, physics="bipolar_dd", acceptor=True)),
                  ("nalgaas", "AlGaAs_bad", 200e-9, tm)]
    return stack_design(layers, name="bad_" + kind)


@pytest.mark.parametrize("kind,exc,match", [("mixed", NotImplementedError, "hole boundary condition"),
                                            ("eg", ValueError, "band_gap_eV")])
def test_remaining_refusals_name_what_is_missing(kind, exc, match):
    b = _builder(_mixed_or_bad(kind), "bad_" + kind)
    try:
        with pytest.raises(exc, match=match):
            b.build_device()
    finally:
        b.teardown()


def test_builder_rejects_unknown_interface_model():
    with pytest.raises(ValueError, match="heterointerface"):
        _builder(_mixed_or_bad("mixed"), "badmodel", heterointerface="tunnelling")


def _defaulted(tm, **kw):
    """`tm` with the given fields put back to the TransportModel DEFAULT (read from the dataclass, so the
    test exercises whatever the default is: None since v0.14, 4.5 / 3.6 on 6595bde)."""
    import dataclasses
    from dynameta.materials import TransportModel
    dflt = {f.name: f.default for f in dataclasses.fields(TransportModel)}
    return dataclasses.replace(tm, **{k: dflt[k] for k in kw})


def _build_or_raise(layers, tag):
    b = _builder(stack_design(layers), tag)
    try:
        b.build_device()
        return b._frames
    finally:
        b.teardown()


def test_transport_model_has_no_default_affinity_or_gap():
    """User decision 2026-11-04: chi_eV and band_gap_eV default to None (they were 4.5 / 3.6 -- a band
    offset of ZERO by default, not by data). Fails on 6595bde."""
    from dynameta.materials import TransportModel
    tm = TransportModel(n_bg_m3=1e23, eps_static=12.9, dos_mass_kg_of_n_m3=lambda n: 0.067 * 9.109e-31)
    assert tm.chi_eV is None and tm.band_gap_eV is None


@pytest.mark.parametrize("which", ["one", "both"])
def test_defaulted_affinity_at_a_heterointerface_refuses(which):
    """A GaAs / AlGaAs drift-diffusion heterointerface with a DEFAULTED affinity must refuse to build,
    naming the field to set. MEASURED on 6595bde (defaults 4.5 / 3.6): both cases build SILENTLY --
    'both' with a zero offset (the old AdvisoryWarning needed BOTH gaps at the 3.6 default too, and
    these gaps are explicit), 'one' with AlGaAs at chi = 4.5 against GaAs 4.07, i.e. a 0.43 eV
    conduction-band offset of the WRONG sign taken from the default -- "DID NOT RAISE" there."""
    a = transport(GAAS, n_bg=1e23, physics="drift_diffusion")
    c = _defaulted(transport(ALGAAS, n_bg=1e23, physics="drift_diffusion"), chi_eV=None)
    if which == "both":
        a = _defaulted(a, chi_eV=None)
    with pytest.raises(ValueError, match=r"set TransportModel\.chi_eV"):
        _build_or_raise([("l0", "M0", 200e-9, a), ("l1", "M1", 200e-9, c)], "defchi_" + which)


def test_defaulted_band_gap_at_a_bipolar_heterojunction_refuses():
    """The bipolar heterojunction reads band_gap_eV (its valence offset is checked against it): a
    defaulted gap refuses, naming band_gap_eV. On 6595bde the 3.6 default met the gap check and raised
    the 'disagrees with its statistics' ValueError instead -- whose message does not say the gap was a
    default; the match below requires the new wording, so it fails there."""
    p = transport(GAAS, n_bg=1e23, physics="bipolar_dd", acceptor=True)
    n = _defaulted(transport(ALGAAS, n_bg=1e23, physics="bipolar_dd"), band_gap_eV=None)
    with pytest.raises(ValueError, match="has no band_gap_eV"):
        _build_or_raise([("pgaas", "GaAs_p", 200e-9, p), ("nalgaas", "AlGaAs_n", 200e-9, n)], "defgap")


def test_defaults_still_build_where_no_offset_is_read():
    """Consumers that do not need the values keep working with the defaults: (a) a same-material
    two-layer stack (zero offset by identity -- the frame is the one 4.5 - 4.5 gave, bit for bit);
    (b) a UNIPOLAR heterojunction with explicit affinities and defaulted gaps (the gap is read only by
    the bipolar check)."""
    g = _defaulted(transport(GAAS, n_bg=1e23, physics="drift_diffusion"), chi_eV=None, band_gap_eV=None)
    fr = _build_or_raise([("l0", "M0", 200e-9, g), ("l1", "M0", 200e-9, g)], "defhomo")
    assert fr["l0"][1] == 0.0 and fr["l1"][1] == 0.0
    a = _defaulted(transport(GAAS, n_bg=1e23, physics="drift_diffusion"), band_gap_eV=None)
    c = _defaulted(transport(ALGAAS, n_bg=1e23, physics="drift_diffusion"), band_gap_eV=None)
    fr = _build_or_raise([("l0", "M0", 200e-9, a), ("l1", "M1", 200e-9, c)], "defgap_uni")
    assert fr["l1"][0] - fr["l0"][0] == pytest.approx(DEC, abs=1e-12)    # dE_c from the explicit chi


def test_3d_from_design_refuses_a_defaulted_heterostack_affinity():
    """Stacked3DSpec.from_design: an ITO / AZO-like heterostack whose AZO carries the defaulted chi_eV
    refuses at spec construction (band_alignment), naming TransportModel.chi_eV; an ITO / ITO stack
    with defaults builds (same-material rule). On 6595bde both affinities defaulted to 4.5 and the
    heterostack built SILENTLY with zero offset -- fails there."""
    import dataclasses
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    from dynameta.materials import DrudeOptical, Material, TransportModel
    from dynameta.constants import M_E
    from validation.carriers_3d_multisemi import N_BG, _design
    layers = [("mirror", 70e-9, "Al-Nd"), ("lower_hfo2", 7e-9, "HfO2"), ("ito_bot", 6e-9, "ITO"),
              ("azo_top", 5e-9, "AZO"), ("upper_hfo2", 7e-9, "HfO2"), ("patch", 50e-9, "air")]
    homo_d = _design([(n, t, "ITO" if m == "AZO" else m) for n, t, m in layers])
    homo_d.materials.add(Material("AZO", DrudeOptical(eps_inf=3.8, m_opt_kg=0.3 * M_E, gamma_rad_s=1e14),
                                  transport=TransportModel(n_bg_m3=N_BG / 4, eps_static=9.0,
                                                           dos_mass_kg_of_n_m3=lambda n: 0.25 * M_E)))
    d = dataclasses.replace(homo_d, stack=dataclasses.replace(
        homo_d.stack, layers=[dataclasses.replace(L, background_material="AZO") if L.name == "azo_top" else L
                              for L in homo_d.stack.layers]))
    with pytest.raises(ValueError, match=r"electron affinities.*TransportModel\.chi_eV"):
        Stacked3DSpec.from_design(d)
    homo = Stacked3DSpec.from_design(homo_d)
    assert homo.chi_semi_eV is None and homo.extra_semiconductors[0][5] is None
    assert homo.band_alignment()["azo_top"][0] == homo.band_alignment()["semi"][0]


# ---------------------------------------------------------------- 3-D: drift-diffusion heterostack
@pytest.fixture(scope="module")
def gated_3d(tmp_path_factory):
    """validation.devsim_heterojunction.spec_3d: degenerate ITO (4e26, chi 4.50) | AZO-like (1e26, chi 4.62,
    m* 0.25) | 4 nm HfO2 | gate, unipolar drift-diffusion with thermionic emission at the ITO/AZO interface.
    The gate draws no DC current, so at ANY gate voltage the electron quasi-Fermi level must be flat at the
    body's (E_F = 0) through BOTH semiconductors and across the heterointerface."""
    pytest.importorskip("gmsh")
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium
    from dynameta.sweep import BiasPoint
    from validation.devsim_heterojunction import spec_3d
    spec = spec_3d("drift_diffusion", "auto")
    sv = Devsim3DEquilibrium(spec, mesh_name="hj3t_m", device_name="hj3t_d",
                             msh_path=str(tmp_path_factory.mktemp("hj3d") / "hj.msh"))
    out = {"band": spec.band_alignment()}
    try:
        sv.build_device()
        out["ifaces"] = sv.heterointerfaces()
        for vg in (0.0, 0.5):
            sv.solve(BiasPoint(label="g", voltages={"gate": vg}))
            out[vg] = {r: (np.asarray(ds.get_node_model_values(device="hj3t_d", region=r, name="Potential")),
                           np.asarray(ds.get_node_model_values(device="hj3t_d", region=r, name="Electrons")))
                       for r in ("semi", "xs")}
            out[("z", vg)] = np.asarray(ds.get_node_model_values(device="hj3t_d", region="xs", name="z"))
    finally:
        sv.teardown()
    return out


def test_3d_dd_heterostack_wires_thermionic_emission_and_keeps_a_flat_fermi_level(gated_3d):
    """v0.13.0 REFUSED Stacked3DSpec(physics='drift_diffusion', extra_semiconductors=...). Now: the
    ITO/AZO interface carries thermionic emission ('auto', dE_c = 0.12 eV) and, with no DC path, the solved
    E_Fn = E_c + kT eta(n/N_c) (exact FD quadrature, E_c = q(Phi_c - psi) from the Anderson alignment) is
    0 at every node of both regions at Vg = 0 and +0.5 V. Tolerance 5e-3 eV: the rational g fit's Delta
    vs exact FD at the degenerate eta ~ 10-20 of these layers (+0.8 % of Delta ~ 7-14 -> ~1.5-3 mV)."""
    assert gated_3d["ifaces"] == {"if_1": "thermionic"}
    band = gated_3d["band"]
    assert band["xs"][0] - band["semi"][0] == pytest.approx(-0.12, abs=1e-12)    # chi_xs larger: E_c lower
    nc = {"semi": nc_m3(0.35), "xs": nc_m3(0.25)}
    for vg in (0.0, 0.5):
        worst = 0.0
        for r in ("semi", "xs"):
            psi, n = gated_3d[vg][r]
            idx = np.linspace(0, len(n) - 1, min(len(n), 40)).astype(int)
            for k in idx:
                worst = max(worst, abs((band[r][0] - psi[k]) + V_T * fd_eta(n[k] / nc[r])))
        assert worst < 5e-3, (vg, worst)
    # the gate accumulates the AZO layer at the OXIDE side (its top face): more electrons there at +0.5 V
    # (the AZO's maximum density sits at the ITO side -- electrons spill from the higher-E_c ITO)
    z = gated_3d[("z", 0.5)]
    top = z > z.max() - 1e-12
    assert np.mean(gated_3d[0.5]["xs"][1][top]) > 1.1 * np.mean(gated_3d[0.0]["xs"][1][top])


# =====================================================================================================
# TEMPERATURE (D1a follow-up): the heterointerface runs at the bulk's temperature (BipolarOptions.T_K ->
# the region V_t, ONE source). At e46bd25 a T_K heterostack without BGN ran the interface, the Anderson
# frame and the band-gap check at 300 K while the bulk ran at T_K.
# =====================================================================================================
T_GATES = (250.0, 350.0)


def _T_iso(T, model):
    """n-GaAs / N-Al0.3Ga0.7As, 1e17 cm^-3 each, bipolar (the physics that carries T_K) at T."""
    return stack_design([("ngaas", "GaAs_n", 400e-9, transport(GAAS, n_bg=N_ISO, physics="bipolar_dd", T_K=T)),
                         ("ntop", "Top_algaas", 400e-9, transport(ALGAAS, n_bg=N_ISO, physics="bipolar_dd",
                                                                  T_K=T))],
                        name="iso_T{:g}_{}".format(T, model))


@pytest.fixture(scope="module")
def isotype_T():
    out = {}
    for T in T_GATES:
        for model in ("thermionic", "continuous"):
            b = _builder(_T_iso(T, model), "isoT{:g}{}".format(T, model[0]), heterointerface=model)
            try:
                b.solve(_bias(0.0), rel_tol=1e-8)
                prof = {r: (column(b, r, "Potential"), column(b, r, "Electrons")) for r in ("ngaas", "ntop")}
                b.solve(_bias(+V_LR), rel_tol=1e-8)
                jplus = current_density(b, None, "cathode")
                b.solve(_bias(-V_LR), rel_tol=1e-8)
                jminus = current_density(b, None, "cathode")
                cf = b.solve(_bias(0.0), rel_tol=1e-8)
                out[(T, model)] = dict(prof=prof, g=abs(jplus - jminus) / (2 * V_LR), ifaces=b.heterointerfaces(),
                                       T_cf=cf.temperature_K)
            finally:
                b.teardown()
    return out


def _iso_oracle_T(prof, top, T):
    """The isotype linear-response oracle at temperature T: bulk int dz/(q mu n_eq) (holes negligible) +
    kT/(q J_s), J_s = A* T^2 exp(-phi_B/kT), phi_B from the solved barrier-side density with N_c(T)."""
    from validation.devsim_heterojunction import thermal_voltage
    vt = thermal_voltage(T)
    (zg, _), (_, ng) = prof["ngaas"]
    (zt, _), (_, nt) = prof["ntop"]
    r_bulk = _trap(1.0 / (Q_E * GAAS["mu_n"] * ng), zg) + _trap(1.0 / (Q_E * top["mu_n"] * nt), zt)
    phi_b = -vt * fd_eta(nt[0] / nc_m3(top["me"], T))                 # AlGaAs side holds the top
    assert phi_b > -vt * fd_eta(ng[-1] / nc_m3(GAAS["me"], T))
    js = richardson(top["me"]) * T ** 2 * math.exp(-phi_b / vt)
    return r_bulk, vt / js, phi_b


@pytest.mark.parametrize("T", T_GATES)
def test_t_gate_a_isotype_te_conductance_follows_the_richardson_law_at_T(isotype_T, T):
    """GATE T-a. The low-bias TE conductance at 250 K and 350 K equals 1/(R_bulk + kT/(q A* T^2
    e^{-phi_B/kT})) at THAT T (phi_B from the solve, N_c(T) from the masses, A* T^2 with the real T).
    At e46bd25 the builder evaluated N_c at 300 K while the bulk ran at T_K, so the band-gap check
    refused the device (with an n_i at 300 K it would have run the interface silently at 300 K: a
    (T/300)^2 Richardson prefactor and a 300 K kT in every barrier exponent)."""
    r = isotype_T[(T, "thermionic")]
    assert r["ifaces"] == {"ngaas__ntop": "thermionic"}
    assert r["T_cf"] == pytest.approx(T, rel=1e-14)                  # the CarrierField reports the solved T
    r_bulk, r_te, phi_b = _iso_oracle_T(r["prof"], ALGAAS, T)
    assert r_te > 3.0 * r_bulk                                        # interface-limited
    assert r["g"] == pytest.approx(1.0 / (r_bulk + r_te), rel=2e-3)
    c = isotype_T[(T, "continuous")]
    assert c["g"] > 3.0 / (r_bulk + r_te)                             # discriminating, as at 300 K
    # the 300 K interface would miss by the prefactor alone: (T/300)^2 = 0.69 / 1.36
    assert abs((T / T_REF) ** 2 - 1.0) > 0.3


@pytest.fixture(scope="module")
def pn_T():
    out = {}
    for T in T_GATES:
        d = stack_design([("pgaas", "GaAs_p", 400e-9, transport(GAAS, n_bg=NA, physics="bipolar_dd",
                                                                acceptor=True, T_K=T)),
                          ("nalgaas", "AlGaAs_n", 400e-9, transport(ALGAAS, n_bg=ND, physics="bipolar_dd",
                                                                    T_K=T))])
        b = _builder(d, "pnT{:g}".format(T))
        try:
            b.solve(_bias(0.0), rel_tol=1e-8)
            out[T] = {(r, nm): column(b, r, nm) for r in ("pgaas", "nalgaas")
                      for nm in ("Potential", "Electrons", "Holes")}
            out[(T, "frames")] = b.band_frames()
        finally:
            b.teardown()
    return out


@pytest.mark.parametrize("T", T_GATES)
def test_t_gate_b_equilibrium_built_in_voltage_at_T_matches_the_band_diagram(pn_T, T):
    """GATE T-b. N-AlGaAs / p-GaAs at 250 K and 350 K: q V_bi = W_p - W_n at T (N_c(T), N_v(T) from the
    masses, eta by exact FD quadrature, kT at T); the conduction spike read from the SOLVED densities is
    dE_c; E_Fn and E_Fp are flat at zero bias. Tolerances as at 300 K (2e-4 V)."""
    from validation.devsim_heterojunction import thermal_voltage
    vt = thermal_voltage(T)
    r = pn_T[T]
    _, psip = r[("pgaas", "Potential")]
    _, psin = r[("nalgaas", "Potential")]
    eta_n = fd_eta(ND / nc_m3(ALGAAS["me"], T))
    eta_p = fd_eta(NA / nc_m3(GAAS["mh"], T))
    vbi = (GAAS["chi"] + GAAS["eg"] + eta_p * vt) - (ALGAAS["chi"] - eta_n * vt)
    assert psin[-1] - psip[0] == pytest.approx(vbi, abs=2e-4)
    assert abs(vbi - 1.502232) > 0.01                                   # really a different T
    _, n_p = r[("pgaas", "Electrons")]
    _, n_n = r[("nalgaas", "Electrons")]
    ec_g = -vt * fd_eta(n_p[-1] / nc_m3(GAAS["me"], T))
    ec_a = -vt * fd_eta(n_n[0] / nc_m3(ALGAAS["me"], T))
    assert ec_a - ec_g == pytest.approx(DEC, abs=1e-6)
    fr = pn_T[(T, "frames")]
    for reg, mat in (("pgaas", GAAS), ("nalgaas", ALGAAS)):
        _, psi = r[(reg, "Potential")]
        _, n = r[(reg, "Electrons")]
        _, p = r[(reg, "Holes")]
        ec = fr[reg][0] - psi
        for k in range(0, len(psi), max(1, len(psi) // 25)):
            # QFL defined above 1e3 m^-3: minority holes in the AlGaAs sit at ~1 m^-3 at 350 K, where the
            # absolute Newton floor sets E_Fp to ~2e-4 eV (MEASURED 2.03e-4 at p = 1.45 m^-3)
            if n[k] > 1e3:
                assert abs(ec[k] + vt * fd_eta(n[k] / nc_m3(mat["me"], T))) < 2e-4, (reg, k, "n")
            if p[k] > 1e3:
                assert abs(ec[k] - mat["eg"] - vt * fd_eta(p[k] / nc_m3(mat["mh"], T))) < 2e-4, (reg, k, "p")


def _T_refusal_design(kind):
    if kind == "mixed_T":
        layers = [("pgaas", "GaAs_p", 200e-9, transport(GAAS, n_bg=1e23, physics="bipolar_dd", acceptor=True,
                                                        T_K=250.0)),
                  ("nalgaas", "AlGaAs_n", 200e-9, transport(ALGAAS, n_bg=1e23, physics="bipolar_dd"))]
    elif kind == "dd_next_to_T":               # unipolar DD has no T_K: it runs at T_REF (fast to set up,
        layers = [("ngaas", "GaAs_n", 200e-9, transport(GAAS, n_bg=1e23, physics="drift_diffusion")),
                  ("nalgaas", "AlGaAs_n", 200e-9, transport(ALGAAS, n_bg=1e23, physics="bipolar_dd", T_K=250.0))]
    else:                                     # n_i supplied at 300 K for a 250 K bipolar heterojunction
        layers = [("pgaas", "GaAs_p", 200e-9, transport(GAAS, n_bg=1e23, physics="bipolar_dd", acceptor=True,
                                                        T_K=250.0, n_i_T_K=T_REF)),
                  ("nalgaas", "AlGaAs_n", 200e-9, transport(ALGAAS, n_bg=1e23, physics="bipolar_dd", T_K=250.0,
                                                            n_i_T_K=T_REF))]
    return stack_design(layers, name="badT_" + kind)


@pytest.mark.parametrize("kind,exc,match", [
    ("mixed_T", NotImplementedError, "NON-ISOTHERMAL"),
    ("dd_next_to_T", NotImplementedError, "NON-ISOTHERMAL"),
    ("ni_at_300K", ValueError, "AT that temperature")])
def test_t_gate_c_temperature_refusals_name_what_is_missing(kind, exc, match):
    """GATE T-c. What is not implemented is refused, naming it: a mixed-temperature stack (two bipolar
    regions at 250 K / 300 K; a unipolar-DD region -- no T_K, like an equilibrium one -- next to a 250 K
    bipolar one; the frame refuses before the unipolar/bipolar interface refusal) --
    no non-isothermal model; and an n_i given at 300 K for a 250 K heterojunction -- its statistics
    contradict band_gap_eV at T."""
    b = _builder(_T_refusal_design(kind), "badT_" + kind)
    try:
        with pytest.raises(exc, match=match):
            b.build_device()
    finally:
        b.teardown()


def test_t_gate_d_300K_interface_arithmetic_is_the_T_REF_arithmetic():
    """GATE T-d (unit half; the solution-hash half is in the design note): at V_t = V_T the interface
    temperature is EXACTLY T_REF and interface_models returns the same floats as the T_REF formula."""
    from dynameta.carriers.physics_heterojunction import HeteroSide, interface_models, temperature_of_V_t
    assert temperature_of_V_t(V_T) == T_REF
    a, b = HeteroSide("a", 0.1, "N_c", 8.0e4), HeteroSide("b", 0.3431, "N_c", 1.1e5)
    k0, k1, at2 = interface_models(a, b, "Electrons")
    assert (k0, k1, at2) == ((0.3431 - 0.1) / V_T, 0.0, 1.1e5 * T_REF ** 2)
    assert interface_models(a, b, "Electrons", V_T) == (k0, k1, at2)
    k0t, _, at2t = interface_models(a, b, "Electrons", V_T * 250.0 / T_REF)
    assert at2t == pytest.approx(1.1e5 * 250.0 ** 2, rel=1e-14) and k0t == pytest.approx(k0 * 1.2, rel=1e-13)
