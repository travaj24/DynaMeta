"""v0.14 temperature fixes (consolidated plan section 3.2, rows 3.2-7 .. 3.2-10; P3 items H1 / H2-guard /
H4). Audit note: docs/audit/2026-11-06-v0.14-temperature-fixes.md.

  3.2-7  the 3-D bipolar builder evaluated N_c, N_v at T_REF under BipolarOptions.T_K (body region AND the
         gate frame phi_bi) and reported CarrierField.temperature_K = T_REF. Gate: the 3-D body and the 2-D
         layered bar at 250 / 350 K against N_c(T) = N_c(300) (T/300)^(3/2) (re-derived here from the DOS
         mass with exact-SI h) and the degenerate-body potential psi = V_t (eta + ln(N_c(T)/n_i)) with eta
         from scipy quadrature of F_1/2.
  3.2-8  the default interface-trap bins were the 300 K kT at any T. Gate: at 77 K the installed sheet
         levels are <= kT(77 K) wide and their trapped charge matches scipy quadrature of the N&B integral
         to the midpoint rule's derived aliasing bound.
  3.2-9  BTBT runs with T_REF parameters under T_K: an AdvisoryWarning now says so.
  3.2-10 chi_eV is used unchanged at any T: a heterostack at T != the temperature chi was quoted at
         (TransportModel.chi_reference_T_K, None = T_REF) refuses by default; on_chi_temperature="warn" /
         "off" proceed.

Every device gate FAILS on 189c648 (the v0.14 integration tip before this branch); the measured misses
are in each docstring.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

from validation.devsim_heterojunction import (ALGAAS, GAAS, n_i_m3, stack_design,  # noqa: E402
                                              transport)
from dynameta.warning_types import AdvisoryWarning                            # noqa: E402

# ---- the oracle's own constants (exact 2019 SI; m_e CODATA 2018) -------------------------------------
KB_SI = 1.380649e-23
Q_SI = 1.602176634e-19
H_SI = 6.62607015e-34
ME_SI = 9.1093837015e-31


def _nc_closed_form(m_rel, T):
    """N(T) = N(300) (T/300)^(3/2), N(300) = 2 (2 pi m k 300 / h^2)^(3/2) -- textbook form with h (not hbar)."""
    n300 = 2.0 * (2.0 * math.pi * m_rel * ME_SI * KB_SI * 300.0 / H_SI ** 2) ** 1.5
    return n300 * (T / 300.0) ** 1.5


def _dos_tol():
    """Budget for the DOS comparison: the library's HBAR is the CODATA-ROUNDED literal (1.054571817e-34),
    not h/2pi, and N ~ hbar^-3, so the two forms differ by 3 |HBAR 2pi/h - 1| (1.8e-9), plus round-off."""
    from dynameta.constants import HBAR
    return 3.0 * abs(HBAR * 2.0 * math.pi / H_SI - 1.0) + 1e-13


def _F12(eta):
    """Normalised F_1/2(eta) = (2/sqrt(pi)) int_0^inf sqrt(e)/(1 + exp(e - eta)) de (scipy quad)."""
    from scipy.integrate import quad
    v, _ = quad(lambda e: math.sqrt(e) / (1.0 + math.exp(min(e - eta, 700.0))), 0.0, eta + 60.0,
                points=[eta] if eta > 0 else None, epsabs=0.0, epsrel=1e-13, limit=200)
    return 2.0 / math.sqrt(math.pi) * v


def _eta_of_u(u):
    from scipy.optimize import brentq
    return brentq(lambda e: _F12(e) - u, -30.0, 60.0, xtol=1e-14, rtol=1e-14)


def test_band_dos_at_T_is_the_closed_form():
    """Row 3.2-7, the shared helper: band_dos_at_T(m_n, m_p, options) = N(300) (T/300)^1.5 at 77/250/350 K,
    T_REF when T_K is None (bit-identical to the T_REF-default effective_dos_m3 call), N_v = N_c when
    m_p is None. Adversarial: ignoring T_K (the 189c648 3-D rule) misses at 250 K by (300/250)^1.5 - 1 =
    31 %."""
    from dynameta.carriers.bipolar_options import BipolarOptions, band_dos_at_T
    from dynameta.carriers.physics_equilibrium import effective_dos_m3
    me, mh = GAAS["me"] * ME_SI, GAAS["mh"] * ME_SI
    tol = _dos_tol()
    for T in (77.0, 250.0, 350.0):
        nc, nv = band_dos_at_T(me, mh, BipolarOptions(T_K=T))
        assert nc == pytest.approx(_nc_closed_form(GAAS["me"], T), rel=tol)
        assert nv == pytest.approx(_nc_closed_form(GAAS["mh"], T), rel=tol)
    assert band_dos_at_T(me, mh, None) == (effective_dos_m3(me), effective_dos_m3(mh))
    assert band_dos_at_T(me, mh, BipolarOptions()) == (effective_dos_m3(me), effective_dos_m3(mh))
    assert band_dos_at_T(me, None, None) == (effective_dos_m3(me), effective_dos_m3(me))
    stale = band_dos_at_T(me, mh, None)[0]                         # adversarial: T_K ignored
    assert abs(stale / _nc_closed_form(GAAS["me"], 250.0) - 1.0) > 0.3


# =====================================================================================================
# 3.2-7: 3-D bipolar body at T_K vs the 2-D layered bar vs the closed form
# =====================================================================================================
T_GATES = (250.0, 350.0)
ETA_BODY = 10.0                      # degenerate n body: eta = (E_F - E_c)/kT at the gate temperature


def _nd_for(T):
    return _nc_closed_form(GAAS["me"], T) * _F12(ETA_BODY)


def _psi_oracle(T, nd):
    """The neutral degenerate body's potential in the bipolar (intrinsic-level) frame:
    psi = (E_F - E_i)/q = V_t [eta + ln(N_c(T)/n_i)], eta = F_1/2^-1(N_D/N_c(T)) by quadrature."""
    vt = KB_SI * T / Q_SI
    nc = _nc_closed_form(GAAS["me"], T)
    eta = _eta_of_u(nd / nc)
    return vt * (eta + math.log(nc / n_i_m3(GAAS, T))), vt, eta, nc


def _solve_3d(T, nd, tag):
    from dynameta.carriers.bipolar_options import BipolarOptions
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    from dynameta.sweep import BiasPoint
    spec = Stacked3DSpec(semi_material="GaAs", oxide_material="SiO2", lateral_m=4e-9, semi_thk_m=12e-9,
                         oxide_thk_m=4e-9, n_bg_m3=nd, eps_semi=GAAS["eps"], eps_oxide=3.9,
                         dos_mass_kg=GAAS["me"] * ME_SI, dos_mass_p_kg=GAAS["mh"] * ME_SI,
                         physics="bipolar_dd", n_i_m3=n_i_m3(GAAS, T), mobility_m2Vs=GAAS["mu_n"],
                         mobility_p_m2Vs=GAAS["mu_p"], tau_srh_s=1e-7, grid_n=(3, 3, 5),
                         mesh_min_nm=1.0, mesh_max_nm=3.0, bipolar_options=BipolarOptions(T_K=T))
    s = Devsim3DEquilibrium(spec, device_name="tf3d_" + tag, mesh_name="tf3m_" + tag)
    try:
        cf = s.solve(BiasPoint({"gate": 0.0}, "flat"))
        psi = np.asarray(ds.get_node_model_values(device=s.device, region="semi", name="Potential"))
        nc = ds.get_parameter(device=s.device, region="semi", name="N_dos")
        nv = ds.get_parameter(device=s.device, region="semi", name="N_dos_p")
    finally:
        s.teardown()
    return dict(T_cf=cf.temperature_K, psi=psi, nc=nc, nv=nv)


def _solve_layered(T, nd, tag):
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    d = stack_design([("ngaas", "GaAs_n", 100e-9, transport(GAAS, n_bg=nd, physics="bipolar_dd", T_K=T))],
                     name="tfl_" + tag, coarse=10e-9)
    b = LayeredDevsimBuilder(d, mesh_name="tflm_" + tag, device_name="tfld_" + tag)
    try:
        cf = b.solve(BiasPoint(label="eq", voltages={"anode": 0.0}))
        psi = np.asarray(ds.get_node_model_values(device=b.device, region="ngaas", name="Potential"))
        nc = ds.get_parameter(device=b.device, region="ngaas", name="N_dos")
        nv = ds.get_parameter(device=b.device, region="ngaas", name="N_dos_p")
    finally:
        b.teardown()
    return dict(T_cf=cf.temperature_K, psi=psi, nc=nc, nv=nv)


@pytest.fixture(scope="module")
def body_at_T():
    out = {}
    for T in T_GATES:
        nd = _nd_for(T)
        out[(T, "3d")] = _solve_3d(T, nd, "{:g}".format(T))
        out[(T, "2d")] = _solve_layered(T, nd, "{:g}".format(T))
        out[(T, "nd")] = nd
    return out


@pytest.mark.parametrize("T", T_GATES)
@pytest.mark.parametrize("builder", ["3d", "2d"])
def test_bipolar_body_dos_potential_and_reported_T_at_T(body_at_T, T, builder):
    """GATE 3.2-7 (P3 H1-a/H1-b). A degenerate n-GaAs body (eta = 10 at T) at the flat-band gate, at 250 and
    350 K, in the 3-D Devsim3DEquilibrium and the 2-D LayeredDevsimBuilder:
      * the region N_dos / N_dos_p = N(300) (T/300)^1.5 (closed form, exact-SI h; budget = the library's
        rounded-HBAR literal, 1.8e-9);
      * every node Potential = V_t [eta + ln(N_c(T)/n_i)] with eta from scipy quadrature of F_1/2. Budget:
        the FD built-in correction Delta(u) is the rational g fit's, +0.78 % of the exact eta - ln u at
        eta = 10 (physics_bipolar_dd, "ACCURACY OF Delta", exact frame; MEASURED here +0.780 %), so tol =
        0.01 V_t (eta - ln u) + 1e-9 V (the Newton). eta = 10, not lower: the fit error grows below the
        documented anchors (MEASURED 1.72 % at eta = 5, 3.3 % at eta = 2 -- see the audit note);
      * CarrierField.temperature_K == T.
    MEASURED on 189c648 (3-D): N_dos 1.3145x / 0.7936x the closed form at 250 / 350 K, reported T = 300 K,
    and the body potential off by 29.6 / 45.8 mV against budgets of 1.47 / 2.06 mV (fixed: 1.15 / 1.60 mV,
    the fit's 0.78 %). The 2-D builder already passed its own checks there (it fails only the 3-D == 2-D
    consistency line); it is the reference the 3-D one must agree with."""
    r = body_at_T[(T, builder)]
    nd = body_at_T[(T, "nd")]
    tol = _dos_tol()
    assert r["nc"] == pytest.approx(_nc_closed_form(GAAS["me"], T), rel=tol)
    assert r["nv"] == pytest.approx(_nc_closed_form(GAAS["mh"], T), rel=tol)
    psi_o, vt, eta, nc = _psi_oracle(T, nd)
    delta_exact = eta - math.log(nd / nc)
    assert delta_exact > 2.0                                       # really degenerate: the DOS is load-bearing
    budget = 0.01 * vt * delta_exact + 1e-9
    assert np.max(np.abs(r["psi"] - psi_o)) < budget, (np.max(np.abs(r["psi"] - psi_o)), budget)
    assert r["T_cf"] == pytest.approx(T, rel=1e-14)
    # 3-D == 2-D (consistency, not the gate): the same body in both builders
    other = body_at_T[(T, "2d" if builder == "3d" else "3d")]
    assert r["nc"] == other["nc"] and r["nv"] == other["nv"]


def test_3d_body_at_T_REF_is_unchanged_by_the_T_K_route():
    """Adversarial / identity half of 3.2-7: T_K = T_REF and T_K = None give the SAME region DOS floats
    (the helper's T_REF default call), so no 300 K result moves."""
    from dynameta.carriers.bipolar_options import BipolarOptions, band_dos_at_T
    me, mh = GAAS["me"] * ME_SI, GAAS["mh"] * ME_SI
    assert band_dos_at_T(me, mh, BipolarOptions(T_K=300.0)) == band_dos_at_T(me, mh, None)


# =====================================================================================================
# 3.2-8: interface-trap bins at the region kT (77 K)
# =====================================================================================================
T77 = 77.0
DIT, EG = 1e16, 1.12


def _install_traps_at(T, tag):
    """A 1-D p-Si / SiO2 stack (bipolar region at T via BipolarOptions.T_K); install a uniform split D_it
    and return the installed sheet levels. No solve: the bins are an install-time quantity."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_traps as PT
    from dynameta.carriers.bipolar_options import BipolarOptions
    dev, m = "tft_" + tag, "tftm_" + tag
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=20e-9, tag="body")
    ds.add_1d_mesh_line(mesh=m, pos=0.2e-6, ps=1e-9, tag="i")
    ds.add_1d_mesh_line(mesh=m, pos=0.21e-6, ps=1e-9, tag="gate")
    ds.add_1d_contact(mesh=m, name="body", tag="body", material="metal")
    ds.add_1d_contact(mesh=m, name="gate", tag="gate", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="si", tag1="body", tag2="i")
    ds.add_1d_region(mesh=m, material="SiO2", region="ox", tag1="i", tag2="gate")
    ds.add_1d_interface(mesh=m, name="si_ox", tag="i")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)
    try:
        ds.node_model(device=dev, region="si", name="NetDoping", equation="-1e23")
        BP.setup_bipolar_region(dev, "si", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7,
                                fd_enhancement=False, options=BipolarOptions(T_K=T))
        PE.setup_dielectric_region(dev, "ox", 3.9)
        PE.setup_interface(dev, "si_ox")
        res = PT.install_interface_traps(dev, "si_ox", PT.InterfaceTraps(dit_eV_m2=DIT, band_gap_eV=EG,
                                                                         recombination=False))
        vt = ds.get_parameter(device=dev, region="si", name="V_t")
    finally:
        ds.delete_device(device=dev)
        ds.delete_mesh(mesh=m)
        _R.clear(dev)
    return res["levels"], vt


def _fermi(x):
    """1/(1 + e^x) without cancellation on either side (1 - f is evaluated as _fermi(-x))."""
    return math.exp(-x) / (1.0 + math.exp(-x)) if x > 0.0 else 1.0 / (1.0 + math.exp(x))


def _charge_sum(levels, ef, kt):
    """Q_it/q [m^-2] of the sheet levels at Fermi level ef (E - E_i coordinates): donor N(1-f), acceptor -N f."""
    q = 0.0
    for e, n, kind in levels:
        q += n * _fermi((ef - e) / kt) if kind == "donor" else -n * _fermi((e - ef) / kt)
    return q


def _charge_quad(ef, kt):
    """The Nicollian-Brews integral by scipy quad: int_{-Eg/2}^0 D (1 - f) dE - int_0^{Eg/2} D f dE, each
    piece split at E_F and E_F +- 40 kT (absolute error 1 m^-2 requested, the estimate is returned)."""
    from scipy.integrate import quad

    def piece(g, a, b):
        cuts = sorted({p for p in (ef - 40.0 * kt, ef, ef + 40.0 * kt) if a < p < b})
        tot = err = 0.0
        for lo, hi in zip([a] + cuts, cuts + [b]):
            v, e = quad(g, lo, hi, epsabs=1.0, epsrel=1e-13, limit=200)
            tot, err = tot + v, err + e
        return tot, err
    a, ea = piece(lambda e: DIT * _fermi((ef - e) / kt), -0.5 * EG, 0.0)
    b, eb = piece(lambda e: DIT * _fermi((e - ef) / kt), 0.0, 0.5 * EG)
    return a - b, ea + eb


def _alias_bound(kt, h):
    """Midpoint-rule error bound on uniform bins of width h for D x (a Fermi step of width kT), flat at the
    band ends (E_F >= 15 kT inside them) and with continuous derivatives across the neutral-level edge (the
    donor N(1-f) and acceptor -N f pieces have the same derivative -D f'): Poisson summation leaves only
    |err| <= 2 pi D kT sum_m 1/sinh(2 pi^2 m kT/h) (the Fourier transform of -f' is
    pi w kT / sinh(pi w kT))."""
    return 2.0 * math.pi * DIT * kt * sum(1.0 / math.sinh(2.0 * math.pi ** 2 * m * kt / h) for m in range(1, 8))


EF_GRID = (-0.4, -0.2, -0.05, 0.0, 0.031, 0.17, 0.4)               # E_F - E_i [eV], >= 0.16 eV off the band ends


def test_interface_trap_bins_follow_the_region_kT_at_77K():
    """GATE 3.2-8 (P3 H1-c). install_interface_traps on a 77 K region: every default bin is <= kT(77 K) =
    6.635 meV wide, and the trapped charge sum_k N_k f_k matches scipy quadrature of the N&B integral at
    77 K, at seven Fermi levels across the gap, within the derived midpoint aliasing bound (_alias_bound at
    the installed bin width, 1.94e6 m^-2 = 6e-10 of the 3e15 m^-2 band charge) + the quad error estimate.
    MEASURED on 189c648: the installed bins were 25.45 meV = 3.84 kT(77 K) wide (the 300 K kT) and the
    charge missed quadrature by up to 4.75e12 m^-2 = 2.1e6x this tolerance -- inside the same bound
    evaluated at THOSE bins (4.89e12), which checks the bound. Fixed: bins 6.588 meV = 0.993 kT, worst miss
    1.89e6 m^-2 against the 1.94e6 bound (the m = 1 alias term is nearly sharp)."""
    levels, vt = _install_traps_at(T77, "77")
    kt = KB_SI * T77 / Q_SI
    assert vt == pytest.approx(kt, rel=1e-14)
    e = np.sort(np.array([lv[0] for lv in levels]))
    widths = np.diff(e)
    assert np.all(widths <= kt * (1.0 + 1e-12)), widths.max() / kt
    h = float(np.max(widths))
    tot = sum(lv[1] for lv in levels)
    assert tot == pytest.approx(DIT * EG, rel=1e-13)               # the bins tile the gap exactly
    worst = 0.0
    for ef in EF_GRID:
        qq, qerr = _charge_quad(ef, kt)
        tol = _alias_bound(kt, h) + qerr + 1e-13 * DIT * EG
        miss = abs(_charge_sum(levels, ef, kt) - qq)
        worst = max(worst, miss / tol)
        assert miss < tol, (ef, miss, tol)
    # adversarial: bins twice as wide (2 kT) must break the 1-kT bound somewhere on the grid
    from dynameta.carriers import physics_traps as PT
    coarse = PT.InterfaceTraps(dit_eV_m2=DIT, band_gap_eV=EG, recombination=False).sheet_levels(kT_eV=2.0 * kt)
    misses = [abs(_charge_sum(coarse, ef, kt) - _charge_quad(ef, kt)[0]) for ef in EF_GRID]
    assert max(misses) > 10.0 * (_alias_bound(kt, h) + 1e-13 * DIT * EG)


def test_sheet_levels_kT_argument():
    """Devsim-free half of 3.2-8: kT_eV sets the DEFAULT width only (an explicit bin_eV wins), None is the
    T_REF kT (the shipped 300 K bins, unchanged), and a non-positive kT is refused."""
    from dynameta.carriers import physics_traps as PT
    from dynameta.constants import V_T
    it = PT.InterfaceTraps(dit_eV_m2=DIT, band_gap_eV=EG)
    assert it.sheet_levels() == it.sheet_levels(kT_eV=V_T)
    k77 = KB_SI * T77 / Q_SI
    lv = it.sheet_levels(kT_eV=k77)
    assert len(lv) == 2 * math.ceil(0.5 * EG / k77 - 1e-12)
    fixed = PT.InterfaceTraps(dit_eV_m2=DIT, band_gap_eV=EG, bin_eV=0.05)
    assert fixed.sheet_levels(kT_eV=k77) == fixed.sheet_levels()
    with pytest.raises(ValueError, match="kT_eV"):
        it.sheet_levels(kT_eV=0.0)


# =====================================================================================================
# 3.2-9: BTBT under T_K warns
# =====================================================================================================
def _btbt_slab(tag, options):
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.tunneling import SILICON_HURKX
    dev, m = "tfb_" + tag, "tfbm_" + tag
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=5e-9, tag="a")
    ds.add_1d_mesh_line(mesh=m, pos=50e-9, ps=5e-9, tag="b")
    ds.add_1d_contact(mesh=m, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=m, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="r", tag1="a", tag2="b")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)
    try:
        ds.node_model(device=dev, region="r", name="NetDoping", equation="1e24")
        BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7,
                                fd_enhancement=False, options=options, btbt=SILICON_HURKX)
    finally:
        ds.delete_device(device=dev)
        ds.delete_mesh(mesh=m)
        _R.clear(dev)


def test_btbt_under_T_K_warns_that_its_parameters_are_at_T_REF():
    """GATE 3.2-9 (decision D-H4). setup_btbt on a 250 K region warns (AdvisoryWarning naming the T and
    T_REF); at T_K = None and T_K = 300.0 (V_t == V_T exactly) it is silent (warnings are errors here).
    189c648: no warning ("DID NOT WARN")."""
    from dynameta.carriers.bipolar_options import BipolarOptions
    with pytest.warns(AdvisoryWarning, match=r"T = 250 K.*BTBT parameters.*T_REF = 300"):
        _btbt_slab("250", BipolarOptions(T_K=250.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _btbt_slab("none", None)
        _btbt_slab("300", BipolarOptions(T_K=300.0))


# =====================================================================================================
# 3.2-10: chi at T != its quoted temperature
# =====================================================================================================
def test_algaas_gaas_assumption_size_quoted_in_the_warning():
    """The 2.15 meV / 100 K the "warn" message quotes, re-derived from Vurgaftman, Meyer & Ram-Mohan, J. Appl.
    Phys. 89, 5815 (2001): eq. (2.13) E_g(T) = E_g(0) - a T^2/(T + b); Table I GaAs (1.519 eV, 0.5405 meV/K,
    204 K), Table II AlAs Gamma (3.099 eV, 0.885 meV/K, 530 K), Table XII bowing C = -0.127 + 1.310 x
    (T-independent). Al0.3Ga0.7As vs GaAs, 250 -> 350 K."""
    from dynameta.carriers.physics_heterojunction import _ALGAAS_GAAS_DEG_MEV_PER_100K

    def v(e0, a, b, T):
        return e0 - a * 1e-3 * T * T / (T + b)
    x = 0.3

    def alloy(T):
        return (1 - x) * v(1.519, 0.5405, 204.0, T) + x * v(3.099, 0.885, 530.0, T) - x * (1 - x) * (-0.127 + 1.310 * x)
    d_alloy = alloy(350.0) - alloy(250.0)
    d_gaas = v(1.519, 0.5405, 204.0, 350.0) - v(1.519, 0.5405, 204.0, 250.0)
    assert d_alloy * 1e3 == pytest.approx(-47.26, abs=0.005)       # P3 plan arithmetic
    assert d_gaas * 1e3 == pytest.approx(-45.11, abs=0.005)
    assert _ALGAAS_GAAS_DEG_MEV_PER_100K == pytest.approx(abs(d_alloy - d_gaas) * 1e3, abs=0.005)


def test_check_affinity_temperature_modes():
    """The kernel, devsim-free: consistent materials pass silently in every mode; an unstated chi
    (None -> T_REF) or a stated wrong T refuses ("refuse"), warns with the size ("warn"), or passes ("off");
    an unknown mode is refused."""
    from dynameta.carriers.physics_heterojunction import (AffinityTemperatureError,
                                                          check_affinity_temperature)
    T = 250.0
    T_rt = T * KB_SI / Q_SI * Q_SI / KB_SI                          # the round trip through V_t
    for mode in ("refuse", "warn", "off"):
        check_affinity_temperature({"A": T, "B": T_rt}, T, mode=mode)
        check_affinity_temperature({"A": None, "B": None}, 300.0, mode=mode)
    for chis in ({"A": None, "B": T}, {"A": 300.0, "B": T}):
        with pytest.raises(AffinityTemperatureError) as ei:
            check_affinity_temperature(chis, T)
        assert isinstance(ei.value, NotImplementedError)
        msg = str(ei.value)
        assert "on_chi_temperature" in msg and "chi_reference_T_K" in msg and "A (chi_eV quoted at 300 K" in msg
        with pytest.warns(AdvisoryWarning, match=r"Size of the assumption.*2\.15 meV per 100 K.*0\.5-1\.1 meV"):
            check_affinity_temperature(chis, T, mode="warn")
        check_affinity_temperature(chis, T, mode="off")
    with pytest.raises(ValueError, match="on_chi_temperature"):
        check_affinity_temperature({"A": None}, T, mode="ignore")


def _pn_250(tag, tag_chi=True, same_material=False, chi_T=None):
    """N-AlGaAs / p-GaAs (or p-GaAs / n-GaAs) bipolar at 250 K; tag_chi=False drops chi_reference_T_K,
    chi_T states another temperature."""
    top_mat, top_name = (GAAS, "GaAs_n2") if same_material else (ALGAAS, "AlGaAs_n")
    lay = []
    for ln, nm, mat, acc in (("pgaas", "GaAs_p", GAAS, True), ("ntop", top_name, top_mat, False)):
        tm = transport(mat, n_bg=1e23, physics="bipolar_dd", acceptor=acc, T_K=250.0)
        if not tag_chi or chi_T is not None:
            tm = dataclasses.replace(tm, chi_reference_T_K=chi_T)
        lay.append((ln, nm, 200e-9, tm))
    return stack_design(lay, name="tfc_" + tag)


def _build(design, tag, **kw):
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    b = LayeredDevsimBuilder(design, mesh_name="tfcm_" + tag, device_name="tfcd_" + tag, **kw)
    try:
        b.build_device()
    finally:
        b.teardown()


def test_heterostack_at_T_refuses_an_untagged_affinity_by_default():
    """GATE 3.2-10 (user decision 3, modified). A 250 K bipolar N-AlGaAs / p-GaAs whose materials do not
    state chi's temperature: the default refuses (AffinityTemperatureError, a NotImplementedError, naming
    the toggle and the field and the material); an unknown toggle value is refused at construction.
    189c648: the untagged device built silently (MEASURED: frames ntop 0.8701 V / pgaas 0.6270 V, no
    warning) -- the 300 K affinities used at 250 K."""
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.carriers.physics_heterojunction import AffinityTemperatureError
    with pytest.raises(AffinityTemperatureError, match=r"on_chi_temperature") as ei:
        _build(_pn_250("ref", tag_chi=False), "ref")
    assert isinstance(ei.value, NotImplementedError)
    assert "chi_reference_T_K" in str(ei.value) and "AlGaAs_n" in str(ei.value)
    with pytest.raises(ValueError, match="on_chi_temperature"):
        LayeredDevsimBuilder(_pn_250("bad"), on_chi_temperature="never")


@pytest.mark.slow
def test_heterostack_at_T_toggle_settings_and_the_tag():
    """3.2-10, the other settings (~30 s: four bipolar heterostack builds): "warn" builds with an
    AdvisoryWarning stating the size; "off" builds silently; materials tagged chi_reference_T_K = 250 build
    silently under the default; a p/n stack of ONE semiconductor (equal chi_eV and band_gap_eV, two
    material names) is not a heterostack and is never checked. Adversarial: a tag at the WRONG temperature
    (300 K, stated) still refuses."""
    from dynameta.carriers.physics_heterojunction import AffinityTemperatureError
    with pytest.warns(AdvisoryWarning, match="Size of the assumption"):
        _build(_pn_250("warn", tag_chi=False), "warn", on_chi_temperature="warn")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build(_pn_250("off", tag_chi=False), "off", on_chi_temperature="off")
        _build(_pn_250("tag"), "tag")
        _build(_pn_250("same", tag_chi=False, same_material=True), "same")
    with pytest.raises(AffinityTemperatureError, match=r"quoted at 300 K\)"):
        _build(_pn_250("wrong", chi_T=300.0), "wrong")
