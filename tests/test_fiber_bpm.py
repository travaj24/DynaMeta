"""Discrimination-proven gates for the scalar gain-BPM (dynameta.optics.fiber_amp.gain_bpm).
Oracles and tolerances come from docs/fiber_transverse_grounding_2026_07_29.md sec. 2; the heavy
end-to-end work -- the full dz-refinement order study, the cross-solver parity sweep, the thermal
loop -- lives in validation/fiber_gain_bpm.py. Fixtures here are deliberately tiny (48-96 grids,
mm-to-cm fibers) so the file runs in seconds."""

import warnings

import numpy as np
import pytest

from dynameta.optics.fiber_amp import (AseBand, BPMResult, ConcentrationModel, CrossSectionModel,
                                       FiberSpec, GainBPM, LPMode, Pump, RamanStokes,
                                       RareEarthIon, ResolvedFiberAmplifier, Signal, ThermalLoop,
                                       ThermalModel, erbium, quadratic_duct_period_m,
                                       quadratic_duct_radius_m, radial_temperature_rise,
                                       solve_lp_modes, thermal_lens_focal_power_per_m)

# Smith & Smith, Opt. Express 21:15168 (2013) Table 1 (PINNED) -- the same fixture
# tests/test_fiber_transverse.py and validation/fiber_gain_bpm.py use.
SA_P, SE_P = 2.47e-24, 2.44e-24
SA_S, SE_S = 5.80e-27, 5.00e-25
LAM_P, LAM_S = 0.976e-6, 1.032e-6
TAU_S, N_T = 901e-6, 3.0e25
A_CORE, NA, N_CLAD = 25e-6, 0.054, 1.45


def _smith_yb():
    sa = CrossSectionModel(((LAM_P, 4e-9, SA_P), (LAM_S, 6e-9, SA_S)))
    se = CrossSectionModel(((LAM_P, 4e-9, SE_P), (LAM_S, 6e-9, SE_S)))
    return RareEarthIon("Yb3+(SmithTable1)", sa, se, tau_s=TAU_S, zero_line_m=LAM_P, host="test")


def _fiber(length_m=0.02, clad=100e-6):
    return FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=length_m,
                     clad_radius_m=clad)


def _passive(length_m, **kw):
    """gain_model = 0 REPLACES the rate-equation medium, so no ion and no pump are needed."""
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=length_m)
    return GainBPM(None, fib, [], [Signal(1.0, LAM_S)], gain_model=lambda I: 0.0 * I, **kw)


def _absorbing():
    """The STEP-INDEX fixtures below shed discretization radiation into the boundary well above the
    module's 1e-6 converged-window bar, so solve() raises the per-step-absorber advisory (gain_bpm
    scope limitation 1: the mask is a dz-DEPENDENT operator). Assert it HERE rather than filtering
    it away in pyproject -- the repo's no-warnings policy stays intact, and "this fixture still
    exercises the absorber" becomes a checked fact instead of a hope."""
    return pytest.warns(UserWarning, match="boundary absorber removed")


D_PRIME = 1.0e5
ALPHA = float(np.sqrt(D_PRIME))
W_M = quadratic_duct_radius_m(D_PRIME, LAM_S, N_CLAD)
Z_P = quadratic_duct_period_m(D_PRIME)
G_SAT, I_SAT = 300.0, 3.76e8                 # the saturable duct medium of the order gate


def _duct(length_m, n_grid=64, extent_m=200e-6, gain_model=None):
    """The SMOOTH quadratic duct of dossier sec. 2.7 -- the ONLY fixture the order gate may use
    (on the step-index fiber the dossier measured a ragged 0.90/0.92/2.03/4.55 because the hard
    discontinuity contributes an error that does not scale with dz)."""
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=length_m)
    return GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=n_grid, extent_m=extent_m,
                   gain_model=(lambda I: 0.0 * I) if gain_model is None else gain_model,
                   absorber=False, n_ref=N_CLAD,
                   index_profile=lambda X, Y: N_CLAD * (1.0 - 0.5 * ALPHA ** 2 * (X ** 2 + Y ** 2)))


def _duct_orders(gain_model, n_grid=48, nzs=(24, 48, 96), n_ref_steps=768):
    """log2 error ratios of the duct march under dz refinement, against the same solver at
    n_ref_steps."""
    b = _duct(0.5 * Z_P, n_grid=n_grid, gain_model=gain_model)
    E0 = b.launch_field(w_m=1.5 * W_M)
    ref = b.solve(n_steps=n_ref_steps, n_samples=2, E_in=E0).E_out
    errs = [float(np.linalg.norm(b.solve(n_steps=nz, n_samples=2, E_in=E0).E_out - ref)
                  / np.linalg.norm(ref)) for nz in nzs]
    return b, E0, [float(np.log2(errs[i - 1] / errs[i])) for i in range(1, len(errs))]


# ============================ operators and conventions ==================================

def test_passive_march_conserves_power_to_round_off():
    """dossier B9. D is a pure phase in Fourier space and the index leg a pure phase in real
    space, so with the absorber off the ONLY power change available is round-off."""
    b = _passive(1e-3, n_grid=64, extent_m=150e-6, absorber=False)
    r = b.solve(n_steps=100, n_samples=6)
    assert float(np.max(np.abs(r.P_signal_W / r.P_signal_W[0] - 1.0))) < 1e-12


def test_uniform_gain_reproduces_exp_gL_with_no_stray_factor_of_two():
    """dossier B8 (measured there 2.4e-14). g is the INTENSITY gain, so the amplitude factor is
    exp(g dz/2); a stray 2 would show up as exp(2 g L) or exp(g L/2), both orders of magnitude
    away from this tolerance."""
    g0, L = 2.5, 1.3
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=L)
    b = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=48, extent_m=400e-6,
                gain_model=lambda I: np.full_like(I, g0), absorber=False,
                index_profile=lambda X, Y: np.full_like(X, N_CLAD))
    r = b.solve(n_steps=8, n_samples=3, E_in=b.launch_field(w_m=40e-6))
    assert r.P_signal_W[-1] / r.P_signal_W[0] == pytest.approx(np.exp(g0 * L), rel=1e-12)


def test_split_step_is_globally_second_order_on_the_smooth_duct():
    """dossier B7: halving dz must divide the error by 4. NOTE what this case does NOT do: with
    gain_model = 0 the midpoint corrector of _nonlinear is a no-op -- a gain that is identically
    zero is the same at the step entry as at the half-advanced field -- so a frozen-gain
    (exponential-Euler) leg reproduces these numbers BITWISE. The gate that catches a frozen leg is
    the saturable one below."""
    _, _, orders = _duct_orders(None)
    assert all(abs(o - 2.0) < 0.05 for o in orders), orders


def test_split_step_stays_second_order_with_a_saturable_gain():
    """THE discriminating order gate. With a saturable medium the entry gain and the midpoint gain
    genuinely differ, so freezing g at the step entry drops the whole Strang split to O(dz):
    MEASURED with a frozen leg patched in place of the midpoint one (verifier, 2026-08-04), the
    orders collapse to 1.32/1.18/1.15 against 2.00 here. The launch enters at peak I/I_sat = 1.05,
    the knee where dg/dI is largest; validation gate C carries the full series and the frozen
    twin. Bar 2.00 +/- 0.15 rather than +/- 0.05 -- the medium's own nonlinearity puts more spread
    on the measured order than the passive duct does."""
    b, E0, orders = _duct_orders(lambda I: G_SAT / (1.0 + I / I_SAT))
    assert float(np.max(np.abs(E0) ** 2)) / I_SAT == pytest.approx(1.05, abs=0.01)
    assert b.solve(n_steps=768, n_samples=2, E_in=E0).P_signal_W[-1] > 3.0   # it really amplifies
    assert all(abs(o - 2.0) < 0.15 for o in orders), orders


def test_quadratic_duct_matched_gaussian_and_period():
    """dossier B3/B4/B5 -- the tight oracle that pins the diffraction factor, the index-phase
    factor and both signs at once. w_m = sqrt(2/(k0 n0 alpha)) is invariant, the field self-images
    at z_p = 2 pi/alpha, and a mismatched launch reaches its first extremum at z_p/4."""
    b = _duct(Z_P, n_grid=96, extent_m=250e-6)
    E0 = b.launch_field(w_m=W_M)
    r = b.solve(n_steps=300, n_samples=31, E_in=E0)
    assert float(np.ptp(r.spot_radius_m) / W_M) < 1e-4
    fid = abs(np.sum(np.conj(E0) * r.E_out) * b.dA) ** 2 / (b.power_W(E0) * b.power_W(r.E_out))
    assert fid > 1.0 - 1e-8
    b2 = _duct(0.5 * Z_P, n_grid=96, extent_m=250e-6)
    r2 = b2.solve(n_steps=300, n_samples=301, E_in=b2.launch_field(w_m=1.5 * W_M))
    assert r2.z_m[int(np.argmin(r2.spot_radius_m))] == pytest.approx(0.25 * Z_P, rel=2e-3)


def test_duct_parameter_is_thermal_lens_focal_power_per_m():
    """dossier B6: alpha^2 IS thermal.thermal_lens_focal_power_per_m's D', so the BPM and
    thermal.py are gated against each other for free."""
    Q, k_core = 664.0, 1.38
    d_prime = thermal_lens_focal_power_per_m(Q, A_CORE, dndt=1.2e-5, k_core_W_mK=k_core,
                                             n_core=N_CLAD)
    _, dT = radial_temperature_rise(Q, A_CORE, 125e-6, ThermalModel(),
                                    r_m=np.array([0.0, A_CORE]))
    # the in-core temperature rise is exactly parabolic, dT(0) - dT(a) = Q/(4 pi k), so the duct
    # curvature read off the temperature solution must BE D'
    assert 2.0 * 1.2e-5 * float(dT[0] - dT[1]) / (A_CORE ** 2 * N_CLAD) == pytest.approx(
        d_prime, rel=1e-12)
    assert quadratic_duct_period_m(d_prime) == pytest.approx(2.0 * np.pi / np.sqrt(d_prime),
                                                             rel=1e-15)
    assert quadratic_duct_radius_m(d_prime, LAM_S, N_CLAD) == pytest.approx(
        np.sqrt(2.0 / (2.0 * np.pi / LAM_S * N_CLAD * np.sqrt(d_prime))), rel=1e-15)


# ============================ the normalization contract ==================================

def test_field_normalization_contract_is_integral_E2_dA_equals_power():
    """[E] = sqrt(W)/m and INT |E|^2 dA = P_signal. Everything downstream (the local intensity fed
    to the balance, the energy ledger, the modal powers) is only in SI because of this."""
    b = GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                [Signal(3.0, LAM_S)], n_grid=64, extent_m=150e-6)
    assert b.power_W(b.launch_field()) == pytest.approx(3.0, rel=1e-12)
    assert b.power_W(b.launch_field(power_W=0.25)) == pytest.approx(0.25, rel=1e-12)
    assert b.power_W(b.launch_field(w_m=12e-6)) == pytest.approx(3.0, rel=1e-12)
    # the intensity really is W/m^2: a top-hat-free check via the peak of a known Gaussian
    E = b.launch_field(power_W=1.0, w_m=15e-6)
    assert float(np.max(np.abs(E) ** 2)) == pytest.approx(2.0 / (np.pi * 15e-6 ** 2), rel=2e-3)


def test_mode_fields_are_unit_power_and_decomposition_is_orthonormal():
    """The decomposition helper is only meaningful if the LP fields are orthonormal ON THE GRID.
    lma.solve_lp_modes gives the exact modes; sampling them on a Cartesian grid is what could
    break it, so that is what is measured."""
    b = _passive(1e-3, n_grid=96, extent_m=150e-6)
    modes = solve_lp_modes(A_CORE, NA, LAM_S, n_clad=N_CLAD)[:3]
    gram = np.array([[float(np.sum(b.mode_field_2d(mi) * b.mode_field_2d(mj)) * b.dA)
                      for mj in modes] for mi in modes])
    assert np.allclose(gram, np.eye(len(modes)), atol=2e-3)
    assert np.allclose(np.diag(gram), 1.0, atol=1e-12)
    m01, m11 = modes[0], modes[1]
    # cos/sin partners of the SAME LP11 are orthogonal too -- a field solver can spell the pair
    # that transverse.py's intensity-only profiles cannot
    assert abs(float(np.sum(b.mode_field_2d(m11, "cos") * b.mode_field_2d(m11, "sin"))
                     * b.dA)) < 1e-12
    E = b.launch_field(0.7, m01) + b.launch_field(0.3, m11)
    assert b.power_W(E) == pytest.approx(1.0, rel=5e-3)
    powers = np.abs(b.mode_overlaps(E, [m01, m11])) ** 2
    assert powers[0] == pytest.approx(0.7, rel=5e-3)
    assert powers[1] == pytest.approx(0.3, rel=5e-3)


def test_result_mode_powers_helper_reads_the_output_field():
    b = _passive(2e-4, n_grid=96, extent_m=150e-6, index_taper_cells=2.0)
    m01 = solve_lp_modes(A_CORE, NA, LAM_S, n_clad=N_CLAD)[0]
    r = b.solve(n_steps=20, n_samples=3, E_in=b.launch_field(1.0, m01), track_modes=[m01])
    assert isinstance(r, BPMResult)
    assert float(r.mode_powers_W([m01])[0]) == pytest.approx(1.0, rel=1e-2)
    assert abs(r.mode_amp_sqrtW[0, -1]) ** 2 == pytest.approx(float(r.mode_powers_W([m01])[0]),
                                                              rel=1e-12)


# ============================ energy accounting ===========================================

def test_energy_ledger_closes_on_a_gain_run():
    """dossier B10. NOT a tautology: the exchange term is a closed form in g and the field
    ENTERING the N leg, so closure tests that D is unitary and that the index leg is a pure real
    phase."""
    b = GainBPM(_smith_yb(), _fiber(0.01), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                [Signal(50.0, LAM_S)], n_grid=64, extent_m=150e-6, index_taper_cells=2.0)
    with _absorbing():
        r = b.solve(n_steps=120, n_samples=11)
    assert float(np.max(np.abs(r.energy_residual_W)) / r.P_signal_W[0]) < 1e-12
    assert r.P_signal_W[-1] > r.P_signal_W[0]                    # it really amplified
    assert r.medium_exchange_W[-1] > 0.0
    assert r.absorbed_boundary_W[-1] >= 0.0
    assert r.P_pump_W[0, -1] < 100.0                             # and the pump really depleted


def test_background_loss_is_applied_exactly_once_on_the_gain_support():
    """The leg no fixture in this file used to exercise: every other fiber here has
    background_loss_per_m = 0. _medium returns g (F2.7) ALREADY NET of l_s while _march folds the
    scalar amplitude factor exp(-l_s dz/2) into the WHOLE-GRID index phase, so multiplying the gain
    support by exp(g dz/2) on top applied the background loss TWICE inside the doped disc --
    intensity factor exp(g_med dz - 2 l_s dz). Off the support it was right, which is exactly why
    nothing caught it. With the gain off the whole march is then a pure scalar, so exp(-l_s L) is
    an exact oracle: the defect lands on exp(-2 l_s L) over the core and misses this by 6%."""
    loss, L = 3.0, 0.02
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=L,
                    background_loss_per_m=loss)
    b = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=64, extent_m=150e-6,
                gain_model=lambda I: 0.0 * I, absorber=False, index_taper_cells=2.0)
    r = b.solve(n_steps=100, n_samples=5)
    assert r.P_signal_W[-1] / r.P_signal_W[0] == pytest.approx(np.exp(-loss * L), rel=1e-12)
    assert float(np.max(np.abs(r.energy_residual_W)) / r.P_signal_W[0]) < 1e-12
    # and with the rate-equation medium on, the ledger (whose exchange term is a closed form in g
    # and the entering field) must still close, and the loss must cost the lossless twin its dB
    kw = dict(n_grid=64, extent_m=150e-6, index_taper_cells=2.0)
    pumps = [Pump(100.0, LAM_P, "fwd", cladding=True)]
    lossy = GainBPM(_smith_yb(), FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=L,
                                           background_loss_per_m=loss, clad_radius_m=100e-6),
                    pumps, [Signal(1.0, LAM_S)], None, **kw)
    clean = GainBPM(_smith_yb(), _fiber(L), pumps, [Signal(1.0, LAM_S)], None, **kw)
    with _absorbing():
        r_l = lossy.solve(n_steps=150, n_samples=5)
    with _absorbing():
        r_c = clean.solve(n_steps=150, n_samples=5)
    assert float(np.max(np.abs(r_l.energy_residual_W)) / r_l.P_signal_W[0]) < 1e-12
    # not exact: the lossy run carries less signal, so it saturates less and wins a little back
    assert r_c.gain_dB - r_l.gain_dB == pytest.approx(10.0 * np.log10(np.exp(loss * L)), abs=1e-3)
    assert 0.0 < r_l.medium_exchange_W[-1] < r_c.medium_exchange_W[-1]


def test_absorber_advisory_fires_only_above_the_converged_window_bar():
    """The absorber is a per-STEP mask, so it has no dz -> 0 limit (module scope limitation 1):
    MEASURED with light parked in the annulus, P_out/P_in falls 0.612 -> 0.271 over nz = 2 -> 64 at
    a FIXED physical length. That is tolerable only while the absorber eats nothing, so a run that
    crosses the dossier's 1e-6 converged-window bar has to SAY so. A window that comfortably holds
    the mode must stay silent -- an advisory that fired on every run would be filtered away within
    a week and would then warn about nothing."""
    # a SMOOTH (flat) index so nothing is shed by a discretized step, a 25 um Gaussian and a 300 um
    # window: the absorber layer starts 5 mode radii out, where the field is exp(-50)
    fib = FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=1e-3)
    wide = GainBPM(None, fib, [], [Signal(1.0, LAM_S)], n_grid=64, extent_m=300e-6,
                   gain_model=lambda I: 0.0 * I, absorber=True,
                   index_profile=lambda X, Y: np.full_like(X, N_CLAD))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # any warning at all fails this half
        r = wide.solve(n_steps=20, n_samples=2, E_in=wide.launch_field(w_m=25e-6))
    assert r.meta["absorbed_fraction"] <= 1e-6
    narrow = _passive(1e-3, n_grid=64, extent_m=60e-6, absorber=True)
    with pytest.warns(UserWarning, match="dz-DEPENDENT operator"):
        r2 = narrow.solve(n_steps=100, n_samples=2)
    assert r2.meta["absorbed_fraction"] > 1e-6


def test_boundary_absorption_is_tracked_not_discarded():
    """A window narrow enough to clip the mode must REPORT what it ate (dossier B11's contract);
    silently dropping it is the failure mode this exists to prevent."""
    b = _passive(1e-3, n_grid=64, extent_m=60e-6, absorber=True)
    with _absorbing():
        r = b.solve(n_steps=100, n_samples=6)
    assert r.absorbed_boundary_W[-1] / r.P_signal_W[0] > 1e-3
    assert float(np.max(np.abs(r.energy_residual_W)) / r.P_signal_W[0]) < 1e-12


# ============================ cross-solver parity =========================================

def test_small_signal_gain_matches_the_channel_model():
    """The BPM apportions gain by diffraction, ResolvedFiberAmplifier by a fixed modal overlap, so
    they can only agree while the mode stays LP01 -- which at small signal it does. The pin is
    loose on purpose (validation/fiber_gain_bpm.py gate F measures +2e-5 dB at a finer grid and
    carries the refinement series); this is the regression guard, not the measurement."""
    fib, pumps = _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)]
    sig = [Signal(1e-3, LAM_S)]
    b = GainBPM(_smith_yb(), fib, pumps, sig, None, n_grid=64, extent_m=150e-6,
                index_taper_cells=2.0)
    with _absorbing():
        r = b.solve(n_steps=200, n_samples=3)
    ch = ResolvedFiberAmplifier(_smith_yb(), fib, pumps, sig, None).solve(n_nodes=41)
    assert float(ch.signal_gain_dB[0]) > 0.3                     # the fixture has real gain
    assert r.gain_dB == pytest.approx(float(ch.signal_gain_dB[0]), abs=0.05)


# ============================ sampling / truncation guards =================================

def test_undersampled_core_is_refused_not_answered():
    """The truncation-sensitive default: with a cladding pump the window is auto-sized to the
    inner cladding, so a small core in a wide cladding lands on dx coarser than the dossier's
    dx <= a/16 rule without the caller doing anything wrong. Refuse rather than answer -- an
    unresolved index step does not converge away, it sheds power (gate F measures -0.28 dB from a
    hard step at 2 cm)."""
    fib = FiberSpec(core_radius_m=3e-6, na=0.12, n_t_m3=N_T, length_m=0.01, clad_radius_m=200e-6)
    with pytest.raises(ValueError, match="coarser than a/8"):
        GainBPM(erbium(), fib, [Pump(1.0, 0.98e-6, "fwd", cladding=True)],
                [Signal(1e-3, 1.55e-6)], n_grid=64)
    ok = GainBPM(erbium(), fib, [Pump(1.0, 0.98e-6, "fwd", cladding=True)],
                 [Signal(1e-3, 1.55e-6)], n_grid=64, extent_m=20e-6)
    assert ok.dx <= 3e-6 / 8.0


def test_default_window_covers_the_pump_cladding_and_the_taper_is_reported():
    b = GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                [Signal(1.0, LAM_S)], n_grid=256, index_taper_cells=2.0)
    assert b.extent_m == pytest.approx(2.0 * 100e-6, rel=1e-12)
    with _absorbing():
        r = b.solve(n_steps=2, n_samples=2)
    assert r.meta["index_taper_cells"] == 2.0
    assert r.meta["n_ref"] == pytest.approx(
        solve_lp_modes(A_CORE, NA, LAM_S, n_clad=N_CLAD)[0].n_eff(), rel=1e-12)
    assert r.meta["absorber"] is True
    # the taper is OFF by default: it is a numerical smoothing of a physical discontinuity
    b0 = GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                 [Signal(1.0, LAM_S)], n_grid=256)
    assert b0.index_taper_cells == 0.0
    assert float(np.ptp(np.unique(b0.n_xy))) == pytest.approx(
        np.sqrt(N_CLAD ** 2 + NA ** 2) - N_CLAD, rel=1e-12)


def test_automatic_step_count_is_refused_when_it_would_be_absurd():
    """The default step comes from the 0.3-rad index-phase and L_beat/20 criteria; on a long
    fiber that is tens of thousands of steps, and running it silently for minutes is worse than
    saying so."""
    b = GainBPM(_smith_yb(), _fiber(length_m=4.0), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                [Signal(1.0, LAM_S)], n_grid=128, extent_m=150e-6)
    assert 0.0 < b.suggested_step_m() < 1e-3
    with pytest.raises(ValueError, match="max_auto_steps"):
        b.solve()
    rep = b.sampling_report()
    assert rep["dx_over_core_radius"] < 1.0 / 16.0
    assert rep["theta_max_rad"] > NA / N_CLAD                    # Nyquist holds the fiber's cone


def test_transverse_refinement_tightens_the_suggested_longitudinal_step():
    """dx and dz are NOT independent knobs (module scope limitation 2). The highest transverse
    frequency the grid carries, kx_max = pi/dx, accumulates a diffraction phase pi^2 dz/(2 k dx^2)
    per step, so the split aliases at the corner of the k-grid unless dz <= 2 k dx^2/pi -- a bound
    that QUARTERS when dx halves. suggested_step_m() must carry it, or refining N at a fixed step
    count silently walks off the criterion the step was chosen under."""
    args = (_smith_yb(), _fiber(length_m=4.0), [Pump(100.0, LAM_P, "fwd", cladding=True)],
            [Signal(1.0, LAM_S)])
    b1 = GainBPM(*args, n_grid=128, extent_m=150e-6)
    b2 = GainBPM(*args, n_grid=256, extent_m=150e-6)
    assert b1.transverse_nyquist_step_m() == pytest.approx(
        2.0 * b1.k * b1.dx ** 2 / np.pi, rel=1e-12)
    assert b2.transverse_nyquist_step_m() == pytest.approx(
        b1.transverse_nyquist_step_m() / 4.0, rel=1e-12)
    assert b2.suggested_step_m() < b1.suggested_step_m()
    rep = b1.sampling_report()
    assert rep["transverse_nyquist_dz_m"] == pytest.approx(b1.transverse_nyquist_step_m(),
                                                           rel=1e-15)
    assert rep["suggested_dz_m"] <= rep["transverse_nyquist_dz_m"]


def test_nonpositive_dz_is_refused_not_answered():
    """dz_m = -1.0 used to fall through max(ceil(L/dz), 1) = 1 and run a single L-long step -- a
    plausible-looking number from a sign typo -- and dz_m = 0.0 raised a bare ZeroDivisionError
    from inside the marcher."""
    b = _passive(1e-3, n_grid=48, extent_m=150e-6, absorber=False)
    for bad in (0.0, -1.0, -1e-6):
        with pytest.raises(ValueError, match="dz_m must be > 0"):
            b.solve(dz_m=bad)
    assert b.solve(dz_m=1e-5, n_samples=2).meta["n_steps"] == 100


# ============================ scope refusals ==============================================

def test_backward_pump_is_refused():
    with pytest.raises(ValueError, match="backward"):
        GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "bwd", cladding=True)],
                [Signal(1.0, LAM_S)], n_grid=64, extent_m=150e-6)


def test_ase_is_refused_and_says_where_it_lives():
    with pytest.raises(ValueError, match="ASE is not supported"):
        GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                [Signal(1.0, LAM_S)], AseBand(1.01e-6, 1.07e-6, 4), n_grid=64, extent_m=150e-6)
    # an explicitly empty band is not ASE and must be accepted
    GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)],
            [Signal(1.0, LAM_S)], AseBand(1.01e-6, 1.07e-6, 0), n_grid=64, extent_m=150e-6)


def test_multiple_signals_are_refused():
    with pytest.raises(ValueError, match="exactly ONE Signal"):
        GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                [Signal(1.0, LAM_S), Signal(1.0, 1.06e-6)], n_grid=64, extent_m=150e-6)


def test_raman_concentration_esa_and_overlap_override_are_refused():
    fib, pumps, sig = _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)], [Signal(1.0, LAM_S)]
    with pytest.raises(ValueError, match="RamanStokes"):
        GainBPM(_smith_yb(), fib, pumps, sig, n_grid=64, extent_m=150e-6, raman=RamanStokes())
    with pytest.raises(ValueError, match="concentration"):
        GainBPM(_smith_yb(), fib, pumps, sig, n_grid=64, extent_m=150e-6,
                concentration=ConcentrationModel())
    with pytest.raises(ValueError, match="concentration"):
        GainBPM(_smith_yb(), fib, pumps, sig, n_grid=64, extent_m=150e-6, upconversion_C_up=1e-24)
    with pytest.raises(ValueError, match="excited-state absorption"):
        GainBPM(erbium(esa=True),
                FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=0.02,
                          clad_radius_m=100e-6),
                [Pump(1.0, 0.98e-6, "fwd", cladding=True)], [Signal(1e-3, 1.55e-6)],
                n_grid=64, extent_m=150e-6)
    with pytest.raises(ValueError, match="overlap_override"):
        GainBPM(_smith_yb(),
                FiberSpec(core_radius_m=A_CORE, na=NA, n_t_m3=N_T, length_m=0.02,
                          clad_radius_m=100e-6, overlap_override=0.8),
                pumps, sig, n_grid=64, extent_m=150e-6)
    with pytest.raises(ValueError, match="must be a RareEarthIon"):
        GainBPM("not-an-ion", fib, pumps, sig, n_grid=64, extent_m=150e-6)


def test_gain_model_replaces_the_medium_and_refuses_to_share():
    with pytest.raises(ValueError, match="gain_model REPLACES"):
        GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                [Signal(1.0, LAM_S)], n_grid=64, extent_m=150e-6,
                gain_model=lambda I: 0.0 * I)


def test_set_temperature_profile_is_refused():
    b = GainBPM(_smith_yb(), _fiber(), [Pump(100.0, LAM_P, "fwd", cladding=True)],
                [Signal(1.0, LAM_S)], n_grid=64, extent_m=150e-6)
    with pytest.raises(ValueError, match="set_temperature_profile"):
        b.set_temperature_profile(np.array([0.0, 1.0]), np.array([300.0, 310.0]))


# ============================ thermal loop ================================================

def test_thermal_loop_is_off_by_default_and_byte_identical_when_off():
    kw = dict(n_grid=64, extent_m=150e-6, index_taper_cells=2.0)
    args = (_smith_yb(), _fiber(0.01), [Pump(500.0, LAM_P, "fwd", cladding=True)],
            [Signal(20.0, LAM_S)], None)
    with _absorbing():
        a = GainBPM(*args, **kw).solve(n_steps=60, n_samples=3)
    with _absorbing():
        b = GainBPM(*args, thermal=ThermalLoop(b_outer_m=125e-6, max_iter=1, tol_K=1e9),
                    **kw).solve(n_steps=60, n_samples=3)
    # max_iter=1 runs exactly one march, whose first iterate carries ZERO heat -> identical field
    assert np.array_equal(a.E_out, b.E_out)
    assert a.meta["thermal_iterations"] == 0 and b.meta["thermal_iterations"] == 1


def test_thermal_loop_converges_and_focuses():
    b = GainBPM(_smith_yb(), _fiber(0.01), [Pump(1000.0, LAM_P, "fwd", cladding=True)],
                [Signal(20.0, LAM_S)], None, n_grid=64, extent_m=150e-6, index_taper_cells=2.0,
                thermal=ThermalLoop(b_outer_m=125e-6, model=ThermalModel(), max_iter=6,
                                    tol_K=0.05))
    with _absorbing():
        r = b.solve(n_steps=100, n_samples=21)
    assert r.meta["thermal_converged"] and 1 <= r.meta["thermal_iterations"] <= 6
    assert float(np.min(r.heat_per_m_W)) > 0.0                   # the fiber is dissipating
    # dn = (dn/dT) dT(r) must be monotonically DECREASING outward -- a focusing (positive) lens.
    # A sign slip here would defocus the mode and would not otherwise show up in the powers.
    assert b._dn_shape[b.i_axis, b.i_axis] > b._dn_shape[b.i_axis, -1] > 0.0


def test_heat_source_is_the_quantum_defect_times_the_pump_absorption():
    """dossier B14 / (F2.9). The marched pump profile and the reported heat are computed from the
    same integrand but through different code paths (a log-derivative vs an energy density), so
    d/dz of the marched pump power is a genuine consistency check on both."""
    b = GainBPM(_smith_yb(), _fiber(0.02), [Pump(1000.0, LAM_P, "fwd", cladding=True)],
                [Signal(20.0, LAM_S)], None, n_grid=64, extent_m=150e-6, index_taper_cells=2.0)
    with _absorbing():
        r = b.solve(n_steps=200, n_samples=51)
    qd = 1.0 - LAM_P / LAM_S
    # INTEGRAL form: Q(z) is recorded at each step's MIDPOINT while P_p(z) is sampled at the step
    # boundary, so a pointwise d/dz comparison carries a half-step offset plus np.gradient's own
    # O(h^2) error (measured 1.4e-3 -- real, and NOT a physics defect). Integrating removes both:
    # total heat must be the quantum defect times the pump the fiber actually swallowed.
    from dynameta.core.numerics import trapz   # audit X-1: floor-safe (np.trapezoid needs numpy>=2.0)
    assert float(trapz(r.heat_per_m_W, r.z_m)) == pytest.approx(
        qd * float(r.P_pump_W[0, 0] - r.P_pump_W[0, -1]), rel=1e-3)


# ============================ scope honesty ===============================================

def test_module_docstring_states_the_TMI_boundary():
    """dossier B16. A steady-state BPM puts the index grating exactly IN PHASE with the modal
    interference pattern, which transfers zero net energy between modes -- so this module must SAY
    it cannot predict TMI rather than let a caller read mode-content evolution as a threshold."""
    import dynameta.optics.fiber_amp.gain_bpm as mod
    doc = mod.__doc__
    assert "TMI" in doc and "IN PHASE" in doc and "ZERO" in doc
    assert "Jauregui" in doc and "12(2):429 (2020)" in doc
    assert "nonlinear_limits.tmi_threshold_W" in doc
    assert "Smith" in doc and "21(13):15168 (2013)" in doc
    assert "Feit" in doc and "exp(+i omega t)" in doc             # the sign-convention caveat
    assert isinstance(GainBPM.__doc__, str) and "Gamma is an OUTPUT" in mod.__doc__


def test_local_balance_is_called_unbound_and_matches_the_hand_formula():
    """(F1.1) has EXACTLY ONE implementation in this package and gain_bpm must not re-type it, so
    it binds transverse.ResolvedFiberAmplifier._nbar2_nodes as a plain function and calls it with
    self = None. That is only correct while the method never touches `self` -- a mechanical pin,
    because the failure mode is silent: an added `self.` would raise AttributeError deep inside a
    marcher step, or (worse) a @staticmethod/@property refactor would change what the name binds
    to and the call would answer with something else entirely."""
    import dynameta.optics.fiber_amp.gain_bpm as mod

    assert mod._LOCAL_BALANCE is ResolvedFiberAmplifier.__dict__["_nbar2_nodes"]
    c = {"fa": np.array([2.0, 0.5]), "fe": np.array([0.25, 1.5]), "tau": 3.0,
         "idope": np.array([[1.0, 2.0, 0.5], [4.0, 0.25, 3.0]])}
    P = np.ones(2)
    R_a = (c["fa"] * P) @ c["idope"]
    R_e = (c["fe"] * P) @ c["idope"]
    hand = c["tau"] * R_a / (1.0 + c["tau"] * (R_a + R_e))
    assert np.array_equal(mod._LOCAL_BALANCE(None, c, P), hand)
    # and the physics it encodes: nbar2 is a fraction, monotone in the absorption rate, and clamps
    # at fa/(fa+fe) as the pump intensity diverges
    huge = {**c, "idope": c["idope"] * 1e12}
    clamp = (c["fa"] @ c["idope"]) / ((c["fa"] + c["fe"]) @ c["idope"])
    assert np.all((mod._LOCAL_BALANCE(None, c, P) >= 0.0)
                  & (mod._LOCAL_BALANCE(None, c, P) <= 1.0))
    assert mod._LOCAL_BALANCE(None, huge, P) == pytest.approx(clamp, rel=1e-9)
