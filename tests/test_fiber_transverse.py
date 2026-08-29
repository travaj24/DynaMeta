"""Discrimination-proven gates for the radially-resolved inversion n2(r, phi, z) --
transverse spatial hole burning (dynameta.optics.fiber_amp.transverse). Oracles and tolerances
come from docs/fiber_transverse_grounding_2026_07_29.md sec.1; the heavier end-to-end sweeps
(the full kappa table, the 400 x 12 bracket sweep, the dossier sec.1.5 benchmark) live in
validation/fiber_radial_inversion.py. Pure numpy/scipy; fixtures are tiny so the file runs in
seconds."""

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp import (
    AseBand, ConcentrationModel, CrossSectionModel, FiberAmplifier, FiberSpec, Pump,
    RadialGrid, RamanStokes, RareEarthIon, ResolvedFiberAmplifier, Signal, analyze_noise,
    cladding_pump_overlap, dopant_overlap, erbium, mean_field_equivalent, mode_field_radius_m,
    saturation_correction_kappa, solve_lp_modes, tshb_closed_form_J, tshb_mean_field_J,
    ytterbium,
)

# Smith & Smith, Opt. Express 21:15168 (2013) Table 1 (PINNED) -- see the same fixture in
# validation/fiber_radial_inversion.py; both are small and local on purpose.
SA_P, SE_P = 2.47e-24, 2.44e-24
SA_S, SE_S = 5.80e-27, 5.00e-25
LAM_P, LAM_S = 0.976e-6, 1.032e-6
TAU_S, N_T = 901e-6, 3.0e25

YB = ytterbium("aluminosilicate")


def _smith_yb():
    """Two-wavelength Table-1 fixture: narrow lines so each anchor is exact at its own line."""
    sa = CrossSectionModel(((LAM_P, 4e-9, SA_P), (LAM_S, 6e-9, SA_S)))
    se = CrossSectionModel(((LAM_P, 4e-9, SE_P), (LAM_S, 6e-9, SE_S)))
    return RareEarthIon("Yb3+(SmithTable1)", sa, se, tau_s=TAU_S, zero_line_m=LAM_P, host="test")


def _fiber(length_m=0.5, a=3e-6, na=0.12, clad=50e-6):
    return FiberSpec(core_radius_m=a, na=na, n_t_m3=N_T, length_m=length_m, clad_radius_m=clad)


def _amp(ion=YB, fiber=None, pump_W=5.0, sig_W=1e-6, ase=None, **kw):
    fiber = _fiber() if fiber is None else fiber
    lam_s = LAM_S if ion is not YB else 1.030e-6
    lam_p = LAM_P if ion is not YB else 0.976e-6
    return ResolvedFiberAmplifier(ion, fiber, [Pump(pump_W, lam_p, "fwd", cladding=True)],
                                  [Signal(sig_W, lam_s)], ase, **kw)


# ============================ the local balance (F1.1) =====================================

def test_signal_free_inversion_clamps_at_the_cross_section_ratio():
    """dossier T1 (PINNED Smith & Smith sec.2): with no signal and a pump far above
    h nu_p/(tau (sa_p+se_p)) the balance clamps at sa_p/(sa_p+se_p) = 0.50305, a ZERO-parameter
    check -- it depends on no geometry, no density, no lifetime."""
    amp = ResolvedFiberAmplifier(_smith_yb(), _fiber(),
                                 [Pump(1.0e12, LAM_P, "fwd", cladding=True)], [], None)
    nb = amp.local_nbar2([1.0e12])
    target = SA_P / (SA_P + SE_P)
    assert target == pytest.approx(0.50305, abs=1e-5)
    assert float(np.max(np.abs(nb - target))) / target < 1e-12
    assert float(np.ptp(nb)) == 0.0            # a uniform pump gives a uniform inversion


def test_inversion_is_burned_hardest_where_the_mode_is_brightest():
    """The signature of TSHB: with a saturating core-guided signal, nbar2 must INCREASE
    monotonically outward across the doped region (the on-axis ions are the depleted ones)."""
    amp = _amp(sig_W=1.0)
    nb = amp.local_nbar2([5.0, 1.0])
    r = amp.grid.r_m
    inside = r <= amp.fiber.b_dope_m
    nb_in = nb[inside][np.argsort(r[inside])]
    assert np.all(np.diff(nb_in) > 0.0)
    assert nb_in[-1] / nb_in[0] > 1.05         # a real hole, not round-off


# ============================ profiles and the (F1.0) contract =============================

def test_every_profile_is_normalized_to_unit_area():
    amp = _amp(ase=AseBand(1.01e-6, 1.07e-6, 4))
    norms = np.array([amp.grid.integrate(p) for p in amp.profiles])
    assert norms == pytest.approx(np.ones_like(norms), abs=1e-13)
    assert amp.profiles.shape[0] == 1 + 1 + 2 * 4       # pump + signal + fwd/bwd ASE bins


def test_cladding_pump_overlap_emerges_from_the_quadrature():
    """(F2.8) limit check: a flat pump over the inner cladding integrates to A_dope/A_clad, i.e.
    waveguide.cladding_pump_overlap becomes an OUTPUT of the model instead of an input."""
    fib = _fiber()
    amp = _amp(fiber=fib)
    assert float(amp.gamma_quad[0]) == pytest.approx(cladding_pump_overlap(fib), rel=1e-13)


def test_exact_lp_overlap_matches_the_independent_lma_routine():
    """Above the LP11 cutoff the profile is the exact LP field (dossier sec.1.5, binding), so the
    quadrature overlap must agree with lma.dopant_overlap -- a different algorithm on a
    different grid."""
    a, na = 25e-6, 0.054
    fib = FiberSpec(core_radius_m=a, na=na, n_t_m3=N_T, length_m=1.0, clad_radius_m=200e-6)
    m01 = solve_lp_modes(a, na, LAM_S)[0]
    assert m01.V > 2.405                                  # the exact-LP branch really is taken
    amp = ResolvedFiberAmplifier(_smith_yb(), fib, [Pump(1.0, LAM_P, "fwd", cladding=True)],
                                 [Signal(1e-6, LAM_S)], None)
    assert float(amp.gamma_quad[1]) == pytest.approx(dopant_overlap(m01, a), rel=1e-5)


def test_radial_grid_weights_tile_the_cross_section():
    grid = RadialGrid.build([0.0, 3e-6, 5e-5], n_nodes_per_panel=8, n_azimuthal=4)
    assert grid.integrate(np.ones(grid.size)) == pytest.approx(np.pi * (5e-5) ** 2, rel=1e-14)
    assert grid.n_azimuthal == 4 and grid.size == 8 * 2 * 4


# ============================ the closed forms (F1.5)-(F1.8) ===============================

def test_closed_form_matches_the_solver_quadrature():
    """dossier T2: for a uniform (cladding) pump + one Gaussian-mode signal the resolved overlap
    J is the exact Moebius integral (F1.5)/(F1.6). Read out of the PUBLIC local gain
    g = nt[(sa+se) J - sa Gamma]."""
    P_p, P_s = 20.0, 0.5
    fib = _fiber()
    amp = ResolvedFiberAmplifier(_smith_yb(), fib, [Pump(P_p, LAM_P, "fwd", cladding=True)],
                                 [Signal(P_s, LAM_S)], None)
    Ip = P_p / (np.pi * fib.clad_radius_m ** 2)
    hnu_p, hnu_s = H_PLANCK * C_LIGHT / LAM_P, H_PLANCK * C_LIGHT / LAM_S
    den = Ip * (SA_P + SE_P) / hnu_p + 1.0 / TAU_S
    n2_0, n2_inf = (Ip * SA_P / hnu_p) / den, SA_S / (SA_S + SE_S)
    Isat = hnu_s / (SA_S + SE_S) * den
    w = float(mode_field_radius_m(fib.core_radius_m, fib.na, LAM_S))
    gam = 1.0 - np.exp(-2.0 * fib.b_dope_m ** 2 / w ** 2)
    s0 = 2.0 * P_s / (np.pi * w * w * Isat)
    assert 0.5 < s0 < 5.0                                  # genuinely saturated, not a soft case
    J_quad = ((amp.local_gain_per_m([P_p, P_s])[1] + N_T * SA_S * float(amp.gamma_quad[1]))
              / (N_T * (SA_S + SE_S)))
    assert J_quad == pytest.approx(float(tshb_closed_form_J(n2_0, n2_inf, s0, gam)), rel=1e-12)


def test_kappa_is_tanh_over_x_squared_with_the_right_limits():
    """dossier (F1.8). kappa < 1 for every x > 0 IS the direction of the whole feature."""
    assert float(saturation_correction_kappa(1.0)) == pytest.approx(0.7615942, abs=1e-7)
    assert float(saturation_correction_kappa(0.0)) == 1.0
    assert float(saturation_correction_kappa(1e-6)) == pytest.approx(1.0, abs=1e-12)
    x = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    k = saturation_correction_kappa(x)
    assert np.all(k < 1.0) and np.all(np.diff(k) < 0.0)    # monotone decreasing
    # kappa -> 1/x^2 as x -> inf; at x = 3 the residual is 1 - tanh(9) = 3.1e-8
    assert float(k[-1]) == pytest.approx(1.0 / 9.0, rel=1e-7)


def test_resolved_bracket_is_never_positive():
    """dossier T6, the gate that would catch a sign error in the whole feature: the resolved
    integral never EXCEEDS the mean-field one, anywhere on the (Gamma, s0) plane."""
    gs, ss = np.linspace(0.02, 0.9999, 60), np.logspace(-6.0, 8.0, 12)
    GG, SS = np.meshgrid(gs, ss, indexing="ij")
    assert float(np.max(tshb_closed_form_J(1.0, 0.0, SS, GG)
                        - tshb_mean_field_J(1.0, 0.0, SS, GG))) <= 1e-9


def test_closed_forms_reconverge_in_both_limits():
    """dossier sec.1.3: resolved and mean-field agree as s0 -> 0 (nothing is saturated) AND as
    s0 -> inf (both extract everything). The disagreement is a bump in between."""
    gam = 0.8647
    assert float(tshb_closed_form_J(1.0, 0.0, 1e-9, gam)) == pytest.approx(gam, rel=1e-8)
    for s0, tol in ((1e-9, 1e-8), (1e4, 1e-3)):
        assert (float(tshb_closed_form_J(1.0, 0.0, s0, gam))
                / float(tshb_mean_field_J(1.0, 0.0, s0, gam))) == pytest.approx(1.0, rel=tol)
    mid = (float(tshb_closed_form_J(1.0, 0.0, 2.64, gam))
           / float(tshb_mean_field_J(1.0, 0.0, 2.64, gam)))
    assert mid == pytest.approx(0.92539, rel=1e-4)         # dossier sec.1.4 worst-case row


# ============================ reduction to the mean-field solver ===========================

def test_unsaturated_solve_reduces_to_the_mean_field_solver():
    """dossier T3: with a 1 pW signal and no ASE, nothing burns a hole and the resolved solver
    must reproduce steady_state.FiberAmplifier to the ODE tolerance."""
    amp = _amp(sig_W=1e-12)
    res, mf = amp.solve(), mean_field_equivalent(amp).solve()
    assert res.meta["converged"] and mf.meta["converged"]
    assert float(res.signal_gain_dB[0]) == pytest.approx(float(mf.signal_gain_dB[0]), abs=1e-9)
    assert float(res.nbar2_z[-1]) == pytest.approx(float(mf.nbar2_z[-1]), rel=1e-9)


@pytest.mark.parametrize("P_s", [1e-6, 0.1, 20.0])
def test_flat_profile_reproduces_the_mean_field_solver_at_every_saturation(P_s):
    """dossier T10. The 'flat' spec IS the mean-field closure written as a transverse profile,
    so the resolved solver must contain steady_state as an exact special case -- including in
    deep saturation, where the gain is compressed by ~65x."""
    amp = _amp(sig_W=P_s, signal_modes=["flat"])
    res, mf = amp.solve(), mean_field_equivalent(amp).solve()
    assert float(res.signal_gain_dB[0]) == pytest.approx(float(mf.signal_gain_dB[0]), abs=1e-9)


def test_cladding_pumped_resolved_gain_never_exceeds_the_mean_field_gain():
    """dossier T6 at the SOLVER level, and the headline claim -- SCOPED to the case the dossier
    actually proves: a UNIFORM (cladding) pump, where nbar2 stays Moebius in the one saturating
    signal intensity. Every amplifier this test builds is cladding-pumped (see `_amp`). The
    deficit vanishes at both ends and peaks in between. The core-pumped case has the OPPOSITE
    sign -- see the companion test below."""
    deltas = []
    for P_s in (1e-9, 1e-2, 0.1, 10.0):
        amp = _amp(sig_W=P_s)
        assert all(p.cladding for p in amp.pumps)       # the scope of the claim, asserted
        deltas.append(float(amp.solve().signal_gain_dB[0]
                            - mean_field_equivalent(amp).solve().signal_gain_dB[0]))
    assert max(deltas) <= 1e-8                          # never positive
    assert deltas[0] == pytest.approx(0.0, abs=1e-7)    # unsaturated: no difference
    assert min(deltas) < -1e-2                          # and a real, multi-hundredth-dB deficit
    assert deltas[2] < deltas[3]                        # deficit re-shrinks in deep saturation


def test_core_pumped_resolved_gain_exceeds_the_mean_field_gain():
    """The SIGN EXCEPTION (dossier sec.1.4 caveat (b)): with a CORE-GUIDED pump the pump's own
    absorption saturates non-uniformly -- it is bleached hardest on axis where it is brightest --
    so the resolved pump absorption coefficient is SMALLER than mean-field, more pump survives
    downstream, and the resolved END-TO-END gain comes out ABOVE the mean-field gain. The
    direction is asserted tolerance-free; the magnitude only loosely bracketed."""
    fib = FiberSpec(core_radius_m=7e-6, na=0.12, n_t_m3=N_T, length_m=1.0, clad_radius_m=50e-6)
    assert 2.0 * np.pi * 7e-6 * 0.12 / 0.976e-6 == pytest.approx(5.4077, abs=1e-3)  # V = 5.41
    amp = ResolvedFiberAmplifier(YB, fib, [Pump(0.2, 0.976e-6, "fwd", cladding=False)],
                                 [Signal(1e-3, 1.030e-6)], None)
    assert not any(p.cladding for p in amp.pumps)
    res, mf = amp.solve(), mean_field_equivalent(amp).solve()
    assert res.meta["converged"] and mf.meta["converged"]
    delta = float(res.signal_gain_dB[0] - mf.signal_gain_dB[0])
    assert delta > 0.0                                  # the sign, with NO tolerance
    assert 0.0 < delta < 1.5                            # measured +0.798 dB on this fixture


def test_mean_field_equivalent_carries_the_whole_plan():
    amp = _amp(ase=AseBand(1.01e-6, 1.07e-6, 3))
    mf = mean_field_equivalent(amp)
    assert isinstance(mf, FiberAmplifier)
    assert mf.ion is amp.ion and mf.fiber is amp.fiber and mf.ase is amp.ase
    assert mf.pumps == amp.pumps and mf.signals == amp.signals
    assert mf.concentration is None and mf.raman is None


# ============================ multi-mode competition (F1.9) ================================

def test_lp01_lp11_modal_gain_ordering_reverses_with_saturation():
    """dossier T7: two modes on ONE shared nbar2(r,phi,z). At small signal LP01 wins because it
    is better confined; once it burns an on-axis hole, LP11 -- which peaks off-axis -- gains
    MORE. A mean-field model gives g_11 - g_01 = const < 0 and can never show this."""
    a, na = 25e-6, 0.054
    fib = FiberSpec(core_radius_m=a, na=na, n_t_m3=N_T, length_m=1.0, clad_radius_m=200e-6)
    modes = solve_lp_modes(a, na, LAM_S)
    m01 = [m for m in modes if m.l == 0 and m.m == 1][0]
    m11 = [m for m in modes if m.l == 1 and m.m == 1][0]
    amp = ResolvedFiberAmplifier(_smith_yb(), fib, [Pump(1000.0, LAM_P, "fwd", cladding=True)],
                                 [Signal(1.0, LAM_S), Signal(1e-6, LAM_S)], None,
                                 signal_modes=[m01, m11])
    assert amp.grid.n_azimuthal > 1                    # an l>=1 mode switched on the phi grid
    assert float(amp.gamma_quad[1]) > float(amp.gamma_quad[2])     # LP01 better confined
    d = {}
    for P01 in (1.0, 10.0, 100.0, 1000.0):
        g = amp.local_gain_per_m([1000.0, P01, 1e-6])
        d[P01] = float((g[2] - g[1]) * 10.0 / np.log(10.0))
    assert d[1.0] < 0.0 < d[10.0]                      # the crossover, sign-exact
    assert d[100.0] == max(d.values()) and d[100.0] > 2.0
    assert d[1000.0] < d[100.0]                        # and it decays again
    assert d[1.0] == pytest.approx(-0.287542, rel=1e-3)
    assert d[100.0] == pytest.approx(2.843726, rel=1e-3)


def test_beat_length_of_the_lma_fixture():
    """dossier T8: L_beat(LP01, LP11) = 9.895 mm for the Table-1 fiber."""
    modes = solve_lp_modes(25e-6, 0.054, LAM_S)
    m01 = [m for m in modes if m.l == 0 and m.m == 1][0]
    m11 = [m for m in modes if m.l == 1 and m.m == 1][0]
    assert 2.0 * np.pi / (m01.beta - m11.beta) == pytest.approx(9.895e-3, rel=1e-4)


# ============================ quadrature convergence =======================================

def test_doubling_the_quadrature_nodes_does_not_move_the_answer():
    """dossier T9 / sec.1.7 item 5, evaluated in DEEP saturation where the integrand is least
    polynomial-like."""
    fib = _fiber()
    P = [5.0, 20.0]
    d24 = _amp(fiber=fib, sig_W=20.0, n_quad=24).dP_dz(P)
    d48 = _amp(fiber=fib, sig_W=20.0, n_quad=48).dP_dz(P)
    assert float(np.max(np.abs(d24 - d48) / np.abs(d48))) < 1e-10


def _two_l_mode_amp(n_azimuthal):
    """LP11 + LP21 on the Smith & Smith LMA fiber, BOTH driven to 200 W -- the hardest azimuthal
    integrand the module supports (two different l >= 1 profiles burning one shared hole)."""
    a, na = 25e-6, 0.054
    fib = FiberSpec(core_radius_m=a, na=na, n_t_m3=N_T, length_m=1.0, clad_radius_m=200e-6)
    modes = solve_lp_modes(a, na, LAM_S)
    m11 = [m for m in modes if m.l == 1 and m.m == 1][0]
    m21 = [m for m in modes if m.l == 2 and m.m == 1][0]
    return ResolvedFiberAmplifier(_smith_yb(), fib, [Pump(1000.0, LAM_P, "fwd", cladding=True)],
                                  [Signal(200.0, LAM_S), Signal(200.0, LAM_S)], None,
                                  signal_modes=[m11, m21], n_azimuthal=n_azimuthal)


def test_doubling_the_azimuthal_nodes_does_not_move_the_answer():
    """The AZIMUTHAL twin of the radial convergence gate, which the feature shipped without: with
    TWO different l >= 1 modes both saturating, the phi integrand is a product of cos^2(phi) and
    cos^2(2 phi) against a nonlinear balance, and 16 nodes (the original default) misses the
    repo's 1e-10 bar by three orders -- 4.98e-7 relative. At the raised default of 32 the
    32 -> 64 doubling moves the modal gains by 1.5e-13."""
    amp32, amp64 = _two_l_mode_amp(32), _two_l_mode_amp(64)
    assert amp32.grid.n_azimuthal == 32 and amp64.grid.n_azimuthal == 64
    P = [1000.0, 200.0, 200.0]
    g32, g64 = amp32.local_gain_per_m(P), amp64.local_gain_per_m(P)
    assert float(np.max(np.abs(g32 - g64) / np.abs(g64))) < 1e-10
    d32, d64 = amp32.dP_dz(P), amp64.dP_dz(P)
    assert float(np.max(np.abs(d32 - d64) / np.abs(d64))) < 1e-10
    # and the discrimination: the OLD default really was above the bar on this fixture
    g16 = _two_l_mode_amp(16).local_gain_per_m(P)
    assert float(np.max(np.abs(g16 - g64) / np.abs(g64))) > 1e-8


def test_the_shipped_azimuthal_default_is_the_converged_value():
    """The default must BE the converged value, not merely be reachable by an opt-in."""
    a, na = 25e-6, 0.054
    fib = FiberSpec(core_radius_m=a, na=na, n_t_m3=N_T, length_m=1.0, clad_radius_m=200e-6)
    m11 = [m for m in solve_lp_modes(a, na, LAM_S) if m.l == 1 and m.m == 1][0]
    amp = ResolvedFiberAmplifier(_smith_yb(), fib, [Pump(1000.0, LAM_P, "fwd", cladding=True)],
                                 [Signal(1.0, LAM_S)], None, signal_modes=[m11])
    assert amp.grid.n_azimuthal == 32
    # and the axis is still switched OFF entirely when every mode has l = 0
    assert _amp().grid.n_azimuthal == 1


# ============================ result object ================================================

def test_result_carries_the_resolved_extras_and_duck_types_for_noise():
    amp = _amp(ase=AseBand(1.01e-6, 1.07e-6, 4))
    res = amp.solve()
    n_z, n_r, K = res.z_m.size, amp.grid.size, res.lambda_m.size
    assert res.nbar2_rz.shape == (n_z, n_r) and res.gain_per_m.shape == (K, n_z)
    assert res.n2_rz.shape == (n_z, n_r)
    outside = amp.grid.r_m > amp.fiber.b_dope_m
    assert np.all(res.n2_rz[:, outside] == 0.0)        # nt(r) is a top-hat
    assert np.all((res.nbar2_rz >= 0.0) & (res.nbar2_rz <= 1.0))
    # nbar2_z is the DOPANT-AREA average of the resolved field, by contract
    w = amp.grid.dA_m2 * (~outside)
    assert res.nbar2_z[10] == pytest.approx(
        float(np.dot(w, res.nbar2_rz[10]) / amp.fiber.a_dope_m2), rel=1e-12)
    assert res.meta["resolved"] is True and res.meta["mcc"] is None
    assert res.nbar2_mode_z.shape == (n_z,)
    # the SteadyStateResult meta contract is honoured, so the noise helpers consume it
    nz = analyze_noise(res, 1.030e-6)
    assert nz.gain_dB == pytest.approx(float(res.signal_gain_dB[0]), abs=1e-3)
    assert nz.n_sp_local_min >= 1.0 - 1e-9


def test_mode_weighted_inversion_is_the_average_the_gain_actually_used():
    """nbar2_mode_z = J/Gamma of the first signal -- the reduction (F1.2) integrates -- as
    opposed to nbar2_z, the dopant-AREA average that the SteadyStateResult contract (and hence
    noise.local_inversion_factor / n_sp_local_*) reads. They coincide when nothing saturates and
    separate once the mode burns its hole."""
    amp = _amp(sig_W=1e-9)
    res = amp.solve()
    # rel 3e-8, was 1e-9 then 5e-9: the identity is asymptotic (the residual radial structure
    # comes from pump/ASE saturation, not the 1 nW signal); the ytterbium(mccumber_refit)
    # default moved the fixture to 1.8e-9 relative separation, the Melkumov-table default
    # (2026-08-28, stronger ASE) to 1.35e-8.  Still ~1e6x tighter than the saturated case
    # below, so the gate keeps its discrimination.
    assert res.nbar2_mode_z == pytest.approx(res.nbar2_z, rel=3e-8)     # unsaturated: identical
    amp = _amp(sig_W=1.0)
    res = amp.solve()
    assert np.all(res.nbar2_mode_z < res.nbar2_z)      # the mode samples the burned region
    assert float(np.min(res.nbar2_mode_z / res.nbar2_z)) < 0.99        # by a real margin
    # it IS the J/Gamma the gain coefficient used: invert g = nt[(sa+se) J - sa Gamma] (this
    # fixture has no background loss) at the signal channel and compare.
    i = int(np.argmin(np.abs(res.lambda_m - 1.030e-6)))
    sa, se = float(res.meta["sigma_a"][i]), float(res.meta["sigma_e"][i])
    gam = float(res.gamma_quad[i])
    J = (res.gain_per_m[i] + N_T * sa * gam) / (N_T * (sa + se))
    assert res.nbar2_mode_z == pytest.approx(J / gam, rel=1e-10)


def test_no_signal_leaves_the_mode_weighted_inversion_undefined():
    amp = ResolvedFiberAmplifier(_smith_yb(), _fiber(),
                                 [Pump(1.0, LAM_P, "fwd", cladding=True)], [], None)
    assert np.all(np.isnan(amp.solve().nbar2_mode_z))


# ============================ scope refusals ===============================================

def test_refuses_raman():
    with pytest.raises(ValueError, match="RamanStokes"):
        _amp(raman=RamanStokes())


def test_refuses_concentration_and_upconversion():
    with pytest.raises(ValueError, match="concentration"):
        _amp(concentration=ConcentrationModel())
    with pytest.raises(ValueError, match="upconversion_C_up"):
        _amp(upconversion_C_up=1e-24)


def test_refuses_excited_state_absorption():
    with pytest.raises(ValueError, match="excited-state absorption"):
        ResolvedFiberAmplifier(erbium(esa=True), FiberSpec(1.4e-6, 0.24, 1e25, 1.0),
                               [Pump(0.1, 0.980e-6)], [Signal(1e-6, 1.560e-6)], None)
    # an ESA-carrying ion whose ESA is ZERO at every planned wavelength is NOT refused
    ResolvedFiberAmplifier(erbium(esa=True), FiberSpec(1.4e-6, 0.24, 1e25, 1.0),
                           [Pump(0.1, 1.480e-6)], [Signal(1e-6, 1.560e-6)], None)


def test_refuses_overlap_override():
    with pytest.raises(ValueError, match="overlap_override"):
        _amp(fiber=FiberSpec(3e-6, 0.12, N_T, 0.5, overlap_override=0.5))


def test_refuses_a_temperature_profile():
    with pytest.raises(ValueError, match="set_temperature_profile"):
        _amp().set_temperature_profile([0.0, 1.0], [300.0, 400.0])


def test_refuses_a_non_rare_earth_ion():
    with pytest.raises(ValueError, match="RareEarthIon"):
        ResolvedFiberAmplifier("not-an-ion", _fiber(), [], [Signal(1e-6, 1.03e-6)], None)


def test_rejects_malformed_signal_modes():
    with pytest.raises(ValueError, match="PARALLEL list"):
        _amp(signal_modes=[None, None])
    with pytest.raises(ValueError, match="string spelling"):
        _amp(signal_modes=["uniform"])
    with pytest.raises(ValueError, match="solved at"):
        _amp(signal_modes=[solve_lp_modes(3e-6, 0.12, 1.06e-6)[0]])
    with pytest.raises(ValueError, match="core radius"):
        _amp(signal_modes=[solve_lp_modes(9e-6, 0.12, 1.030e-6)[0]])


def test_refuses_a_mode_solved_for_a_different_na():
    """a and lambda match, so ONLY the NA can make V disagree -- and an LP01 solved on NA = 0.20
    is a tighter field than an NA = 0.12 fiber supports. Unchecked it shipped +1.00 dB silently."""
    with pytest.raises(ValueError, match="different NA"):
        _amp(signal_modes=[solve_lp_modes(3e-6, 0.20, 1.030e-6)[0]])
    # the matching mode is accepted, so the check is not simply refusing everything
    _amp(signal_modes=[solve_lp_modes(3e-6, 0.12, 1.030e-6)[0]])


def test_refuses_the_unspellable_degenerate_cos_sin_pair():
    """LPMode carries no orientation, so two LP11s can only mean cos + cos -- a different
    physical state from the cos/sin degenerate pair (-16.5% modal gain at 200 W). Refused."""
    a, na = 25e-6, 0.054
    fib = FiberSpec(core_radius_m=a, na=na, n_t_m3=N_T, length_m=1.0, clad_radius_m=200e-6)
    modes = solve_lp_modes(a, na, LAM_S)
    m11 = [m for m in modes if m.l == 1 and m.m == 1][0]
    m21 = [m for m in modes if m.l == 2 and m.m == 1][0]
    with pytest.raises(ValueError, match="degenerate cos/sin PAIR"):
        ResolvedFiberAmplifier(_smith_yb(), fib, [Pump(1000.0, LAM_P, "fwd", cladding=True)],
                               [Signal(1.0, LAM_S), Signal(1.0, LAM_S)], None,
                               signal_modes=[m11, m11])
    # two DIFFERENT l >= 1 modes are the supported multi-mode case and stay accepted
    ResolvedFiberAmplifier(_smith_yb(), fib, [Pump(1000.0, LAM_P, "fwd", cladding=True)],
                           [Signal(1.0, LAM_S), Signal(1.0, LAM_S)], None,
                           signal_modes=[m11, m21])


def test_refuses_an_r_max_inside_the_inner_cladding():
    """A cladding pump is FLAT over the inner cladding and normalized ON the grid, so an r_max
    inside R_clad does not clip it -- it renormalizes it over the smaller disc, inflating
    Gamma_p by (R_clad/r_max)^2 (4x / 16x / 100x at 100 / 50 / 20 um on a 200 um cladding)."""
    fib = FiberSpec(core_radius_m=3e-6, na=0.12, n_t_m3=N_T, length_m=0.5, clad_radius_m=200e-6)
    for r_max in (100e-6, 50e-6, 20e-6):
        amp = ResolvedFiberAmplifier(YB, fib, [Pump(5.0, 0.976e-6, "fwd", cladding=True)],
                                     [Signal(1e-6, 1.030e-6)], None, r_max_m=r_max)
        with pytest.raises(ValueError, match="inside the inner-cladding radius"):
            amp.solve()
    # at or above R_clad it is accepted and reproduces cladding_pump_overlap exactly
    amp = ResolvedFiberAmplifier(YB, fib, [Pump(5.0, 0.976e-6, "fwd", cladding=True)],
                                 [Signal(1e-6, 1.030e-6)], None, r_max_m=200e-6)
    assert float(amp.gamma_quad[0]) == pytest.approx(cladding_pump_overlap(fib), rel=1e-13)
    # and a CORE-pumped plan is free to use a small r_max -- the guard is cladding-pump-only
    core = ResolvedFiberAmplifier(YB, fib, [Pump(5.0, 0.976e-6, "fwd", cladding=False)],
                                  [Signal(1e-6, 1.030e-6)], None, r_max_m=20e-6)
    assert core.grid.r_max_m == pytest.approx(20e-6)
