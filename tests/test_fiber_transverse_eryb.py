"""Discrimination-proven gates for the RADIALLY RESOLVED Er:Yb co-doped amplifier
(dynameta.optics.fiber_amp.transverse_eryb.ResolvedErYbAmplifier) -- the co-doped counterpart of
transverse.ResolvedFiberAmplifier and the pre-reduction form of eryb.ErYbAmplifier.

The two REDUCTION gates are the spine: forcing flat illumination must reproduce the scalar
ErYbAmplifier, and taking N_Yb -> 0 with k_tr = 0 must reproduce ResolvedFiberAmplifier's TSHB
answer on the same fiber. Between them they pin the new module against both of its parents, so a
defect in the ring balance, the ring integrals, the channel table or the relaxation shows up in
one of them. Bidirectional-adversarial: the reductions are proven out AND the transverse effect
is proven to be REAL (non-zero, correctly signed, and vanishing in the small-signal limit).

Fixtures are tiny on purpose (short fibers, few ASE bins, coarse z meshes) so the file runs in
~1 min; the study-scale numbers live in docs/audit/2026-09-15-eryb-transverse.md."""

import numpy as np
import pytest

from dynameta.optics.fiber_amp import (AseBand, ConcentrationModel, FiberSpec, Pump,
                                       ResolvedFiberAmplifier, Signal, erbium,
                                       mean_field_equivalent, ytterbium)
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier
from dynameta.optics.fiber_amp.transverse_eryb import (RadialDopant, ResolvedErYbAmplifier,
                                                       _bracketed_newton_nodes,
                                                       eryb_mean_field_equivalent)

ER = erbium("aluminosilicate")
YB = ytterbium("phosphosilicate")               # phospho Yb: tau_Yb = 1.45 ms (the k_tr host)


def _clad_fiber(length_m=2.0, a=3.0e-6, na=0.20, n_er=1.0e25, clad=62.5e-6, **kw):
    return FiberSpec(core_radius_m=a, na=na, n_t_m3=n_er, length_m=length_m,
                     clad_radius_m=clad, **kw)


def _plan(n_er_bins=4, n_yb_bins=3):
    return (AseBand(1.530e-6, 1.565e-6, n_er_bins), AseBand(1.000e-6, 1.080e-6, n_yb_bins))


# ============================ Gate 1: the uniform-illumination limit ========================

@pytest.mark.parametrize("opts", [
    pytest.param({}, id="defaults"),
    pytest.param({"yb_sigma_e_scale": 0.40}, id="yb_sigma_e_scale"),
    pytest.param({"yb_sigma_e_scale": 1.85, "er_4i11_2_zero_line_m": 972e-9},
                 id="scale_up+zero_line"),
])
def test_uniform_illumination_reproduces_the_scalar_amplifier_with_the_v0_11_4_options(opts):
    """GATE 1c. PR #30 added `yb_sigma_e_scale` (one scalar on the Yb emission cross-section,
    fitted to a measured 1-um ASE) and `er_4i11_2_zero_line_m`. The first must reach EVERY
    Yb-emission term of the ring solver -- the modal gain, the per-node stimulated-emission rate,
    the ASE spontaneous source, `eta_tr`'s R_e_Yb and the 1030 nm diagnostic -- and the way that
    is achieved here is composition: `ErYbAmplifier._plan()` applies the scale once to `se_yb`
    and this class reads that plan. Composition is only a claim until the reduction is re-run
    with the option ON, which is this gate. The second is inert without `tau32_s` and must stay
    inert. Bar: the same 1e-3 dB.

    Discrimination: 0.40 is Morasse's value and 1.85 is deliberately the other side of 1.0, so a
    solver that had cached its own unscaled `se_yb` anywhere would miss by far more than the bar
    (the Yb ASE source and the 1-um gain move by the scale itself)."""
    fib = _clad_fiber(length_m=1.5)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-4, 1.550e-6)]
    ase, yb_ase = _plan(3, 3)
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22, yb_ase=yb_ase, yb_coupled_fraction=0.9,
              k_tr2_m3_s=2e-22, **opts)
    s = ErYbAmplifier(ER, YB, fib, pumps, sig, ase, **kw).solve(n_nodes=61)
    r = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, uniform_illumination=True,
                              **kw).solve(n_nodes=61)
    assert s.meta["converged"] and r.meta["converged"]
    assert abs(float(r.signal_gain_dB[0]) - float(s.signal_gain_dB[0])) < 1e-3
    assert abs(float(r.meta["eta_transfer"]) - float(s.meta["eta_transfer"])) < 1e-6
    assert abs(float(r.meta["yb_parasitic_gain_dB"])
               - float(s.meta["yb_parasitic_gain_dB"])) < 1e-3
    assert np.max(np.abs(r.power_W - s.power_W)) / np.max(s.power_W) < 1e-6
    # the scale really did reach the solver: meta['sigma_e_yb'] is the SCALED array, and the
    # resolved solve's 1-um parasitic gain moves with it (so the gate above is not vacuous)
    scale = opts.get("yb_sigma_e_scale", 1.0)
    base = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, uniform_illumination=True,
                                 **{k: v for k, v in kw.items()
                                    if k != "yb_sigma_e_scale"}).solve(n_nodes=61)
    moved = abs(float(r.meta["yb_parasitic_gain_dB"])
                - float(base.meta["yb_parasitic_gain_dB"]))
    assert (moved > 1e-2) if scale != 1.0 else (moved == 0.0)


def test_the_resolved_solver_honours_the_yb_emission_scale_when_actually_resolving():
    """The same option, with the illumination RESOLVED rather than flattened -- i.e. on the path
    that has no scalar twin to lean on. Scaling the Yb emission down must lower the 1-um
    parasitic gain and raise the Yb inversion (less stimulated emission draining it)."""
    fib = _clad_fiber(length_m=1.5)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-3, 1.550e-6)]
    ase, yb_ase = _plan(3, 3)
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22, yb_ase=yb_ase)
    hi = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, **kw).solve(n_nodes=41)
    lo = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, yb_sigma_e_scale=0.40,
                               **kw).solve(n_nodes=41)
    assert float(lo.meta["yb_parasitic_gain_dB"]) < float(hi.meta["yb_parasitic_gain_dB"]) - 1e-2
    assert float(lo.meta["beta_yb_z"][0]) > float(hi.meta["beta_yb_z"][0])
    # and it is carried by the re-seed protocol rather than dropped on a metrics.* sweep
    assert lo.meta["sigma_e_yb"][0] == pytest.approx(0.40 * hi.meta["sigma_e_yb"][0], rel=1e-12)
    amp = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, yb_sigma_e_scale=0.40, **kw)
    assert amp.with_pumps([Pump(0.5, 0.976e-6, "fwd", cladding=True)]).yb_sigma_e_scale == 0.40


def test_the_explicit_4i11_2_level_is_refused_by_name():
    """PR #30's explicit Er 4I11/2 (`tau32_s`) carries n3 as a THIRD erbium unknown; the node
    kernel here solves the ADIABATIC reduction, whose whole numerical argument is that the
    co-doped system collapses to ONE bracketed scalar per ring. Refused by name rather than
    silently answering the reduced model under the explicit model's name."""
    fib = _clad_fiber(length_m=1.0)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-3, 1.550e-6)]
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22)
    with pytest.raises(ValueError, match="tau32_s"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, tau32_s=1e-5, **kw)
    with pytest.raises(ValueError, match="tau32_s"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, tau32_s=1e-5,
                              upconversion_via_4i11_2=True, **kw)
    # eryb.py itself refuses the routing without the level, and that refusal must reach the port
    with pytest.raises(ValueError, match="upconversion_via_4i11_2"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, upconversion_via_4i11_2=True, **kw)
    # the zero line alone is INERT (it only splits level-2 from level-3 channels), so it is
    # accepted and must change nothing
    a = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw).solve(n_nodes=31)
    b = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                              er_4i11_2_zero_line_m=960e-9, **kw).solve(n_nodes=31)
    assert np.array_equal(a.power_W, b.power_W)


@pytest.mark.parametrize("two_pop", [False, True])
def test_uniform_illumination_reproduces_the_scalar_amplifier(two_pop):
    """GATE 1. With every core channel forced to its MEAN-FIELD profile Gamma_k/A_dope (and the
    cladding pump already flat), the ring solver computes the scalar model by a completely
    different route -- K x N profile quadratures and a per-node root find instead of one
    area-averaged intensity and one scalar root. Gain and eta_tr must therefore agree to the
    relaxation's own convergence, NOT approximately. Bar: 1e-3 dB.

    This is the gate that discriminates: it fails if the channel table, the overlap quadrature,
    the spontaneous prefactor, the transfer coefficients, the dark-erbium absorption or the
    population algebra differs anywhere from eryb.py's."""
    fib = _clad_fiber()
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-4, 1.550e-6)]
    ase, yb_ase = _plan()
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22, yb_ase=yb_ase)
    if two_pop:
        kw.update(yb_coupled_fraction=0.85, k_tr2_m3_s=2e-22,
                  yb_migration_rate_per_s=50.0)
    s = ErYbAmplifier(ER, YB, fib, pumps, sig, ase, **kw).solve(n_nodes=61)
    r = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, uniform_illumination=True,
                              **kw).solve(n_nodes=61)
    assert s.meta["converged"] and r.meta["converged"]
    assert abs(float(r.signal_gain_dB[0]) - float(s.signal_gain_dB[0])) < 1e-3
    assert abs(float(r.meta["eta_transfer"]) - float(s.meta["eta_transfer"])) < 1e-6
    assert abs(float(r.meta["yb_parasitic_gain_dB"])
               - float(s.meta["yb_parasitic_gain_dB"])) < 1e-3
    # the population reductions and the whole power profile follow, not just the endpoints
    assert np.max(np.abs(r.nbar2_z - s.nbar2_z)) / np.max(s.nbar2_z) < 1e-6
    assert np.max(np.abs(r.meta["beta_yb_z"] - s.meta["beta_yb_z"])) < 1e-8
    assert np.max(np.abs(r.power_W - s.power_W)) / np.max(s.power_W) < 1e-6


def test_uniform_illumination_honours_an_overlap_override_and_resolving_refuses_it():
    """The uniform limit is exactly the configuration in which FiberSpec.overlap_override is
    meaningful (the solver is reproducing the mean-field model, and the override IS that model's
    Gamma), so it is accepted there and refused everywhere else -- and accepting it must still
    land on the scalar answer, override and all."""
    fib = _clad_fiber(overlap_override=0.55)
    pumps = [Pump(0.5, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-4, 1.550e-6)]
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22)
    with pytest.raises(ValueError, match="overlap_override"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw)
    s = ErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw).solve(n_nodes=41)
    r = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, uniform_illumination=True,
                              **kw).solve(n_nodes=41)
    assert abs(float(r.signal_gain_dB[0]) - float(s.signal_gain_dB[0])) < 1e-3


# ============================ Gate 2: the single-ion limit =================================

def test_single_ion_limit_reproduces_the_resolved_two_level_solver():
    """GATE 2. N_Yb -> 0 and k_tr = 0 removes the sensitizer entirely, so the co-doped ring
    solver must reproduce transverse.ResolvedFiberAmplifier's TSHB result on the SAME fiber,
    including the hole burning -- not merely the mean-field answer. Bar: 1e-3 dB.

    Discrimination: the same fixture's mean-field gain is ~0.02 dB away (asserted below), so a
    solver that had silently area-averaged its intensities would fail this gate while still
    passing gate 1."""
    fib = FiberSpec(1.4e-6, 0.24, 1.0e25, 6.0)
    pumps = [Pump(100e-3, 0.980e-6, "fwd")]
    sig = [Signal(1e-3, 1.560e-6)]
    ase = AseBand(1.52e-6, 1.575e-6, 4)
    single = ResolvedFiberAmplifier(ER, fib, pumps, sig, ase)
    r_single = single.solve(n_nodes=81)
    r_co = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, n_yb_m3=1e-30,
                                 k_tr_m3_s=0.0).solve(n_nodes=81)
    assert r_single.meta["converged"] and r_co.meta["converged"]
    assert abs(float(r_co.signal_gain_dB[0]) - float(r_single.signal_gain_dB[0])) < 1e-3
    assert np.max(np.abs(r_co.nbar2_z - r_single.nbar2_z)) / np.max(r_single.nbar2_z) < 1e-6
    # and the TSHB the gate is really about is NOT zero on this fixture
    g_mf = float(mean_field_equivalent(single).solve(n_nodes=81).signal_gain_dB[0])
    assert abs(float(r_single.signal_gain_dB[0]) - g_mf) > 1e-2


# ============================ Gate 3: energy closure per ring ==============================

@pytest.mark.parametrize("two_pop", [False, True])
def test_ring_summed_energy_closure(two_pop):
    """GATE 3. q_opt(z) = dU/dt + D_loss + D_Er + D_Yb + D_tr + D_K2 with EVERY term a ring
    integral. The identity is an algebraic rearrangement of one LOCAL rate balance, so beyond
    round-off a departure is an implementation defect. Bar: 1e-5 relative."""
    fib = _clad_fiber(length_m=1.5)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-3, 1.550e-6)]
    ase, yb_ase = _plan(3, 2)
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22, yb_ase=yb_ase, upconversion_C_up=1.1e-24)
    if two_pop:
        kw.update(yb_coupled_fraction=0.9, k_tr2_m3_s=2e-22)
    amp = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, **kw)
    res = amp.solve(n_nodes=41)
    assert amp.closure_residual(res) < 1e-5
    t = amp.energy_terms_from_result(res)
    # the transfer defect really is the dominant dissipation channel of an EYDFA
    assert np.max(t["transfer_defect"]) > np.max(t["background_loss"])
    assert np.all(t["dissipation"] > 0.0)
    # a steady state stores nothing: dU/dt is round-off against q_opt
    assert np.max(np.abs(t["d_stored_dt"])) / np.max(np.abs(t["q_optical"])) < 1e-10


def test_wall_plug_efficiency_returns_a_finite_residual():
    """GATE 3b. efficiency._dissipated_power_W looks for `_rate_balance_dissipation_W` FIRST;
    without that hook it reports NaN for a resolved class (which is exactly what
    ResolvedFiberAmplifier still does). This class supplies it, so the wall-plug budget closes."""
    from dynameta.optics.fiber_amp import PumpSource, wall_plug_efficiency
    fib = _clad_fiber(length_m=1.0)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-3, 1.550e-6)]
    amp = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, n_yb_m3=8e25, k_tr_m3_s=2e-22)
    res = amp.solve(n_nodes=41)
    wp = wall_plug_efficiency(amp, res, [PumpSource(wallplug_efficiency=0.45)])
    assert np.isfinite(wp.energy_balance_residual_W)
    assert abs(wp.energy_balance_residual_W) < 1e-3 * float(np.sum(res.power_W[:, 0]) + 1.0)
    assert 0.0 < wp.eta_wallplug < 0.45
    # the thermal layer's Q(z) hook is the same ring-summed dissipation, and its INTEGRAL must
    # agree with thermal.heat_load_per_m's independent flux-gradient form (a different
    # computation: np.gradient of the net forward flux, not the analytic term sum)
    from dynameta.optics.fiber_amp import heat_load_per_m
    from dynameta.core.numerics import trapz
    q_rate = trapz(amp._heat_profile_W_per_m(res), res.z_m)
    q_flux = trapz(heat_load_per_m(res), res.z_m)
    assert abs(q_rate - q_flux) / q_rate < 1e-4


# ============================ Gate 4: the RAM watchdog =====================================

def test_ring_state_is_reported_and_refused_above_the_package_bar():
    """GATE 4. The ring-resolved state is (z-node x ring) per population; `state_bytes` reports
    it and `solve` refuses above the package's existing 2 GiB bar, naming the
    ring x z-node x channel shape. The message must carry all three, because 'too big' without
    the shape does not tell a caller which axis to coarsen."""
    from dynameta.optics.fiber_amp.dynamics import _STORE_PROFILES_MAX_BYTES
    fib = _clad_fiber(length_m=1.0)
    amp = ResolvedErYbAmplifier(ER, YB, fib, [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                                [Signal(1e-3, 1.550e-6)], None, n_yb_m3=8e25)
    rings = amp.grid.size
    K = amp.channel_plan().lambda_m.size
    assert amp.state_bytes(201) == 8 * (4 * 201 * rings + K * 201 + K * rings)
    assert amp.state_bytes(201) < _STORE_PROFILES_MAX_BYTES        # a normal solve is tiny
    huge = 1 + _STORE_PROFILES_MAX_BYTES // (8 * 4 * rings)
    with pytest.raises(ValueError, match="rings"):
        amp.solve(n_nodes=int(huge))
    # the guard fires BEFORE any allocation, so it is instant even at an absurd request
    res = amp.solve(n_nodes=41)
    assert res.meta["state_bytes"] == amp.state_bytes(41)


# ============================ Gate 5: the transverse effect is REAL ========================

def test_resolved_and_scalar_differ_under_saturation_and_converge_in_the_small_signal_limit():
    """GATE 5. The whole point of the module: the resolved and the mean-field answers must
    DIFFER once anything saturates, and must CONVERGE as the signal goes to zero (where the
    inversion is flat over the mode and mean-field is exact). Under a uniform cladding pump the
    saturated difference is also strictly NEGATIVE -- a real mode burns a hole where it is
    intense while the area average never sees the peak (transverse.py's signed statement, which
    the second ion does not overturn, because the pump that sets the Yb inversion is flat).

    THE FIXTURE IS LOW-NA ON PURPOSE (V = 1.95 at 1550 nm, below the 2.405 cutoff), so both
    solvers use the SAME Marcuse profile and what is measured here is hole burning ALONE. On a
    V > 2.405 fiber the resolved solver also switches to the exact LP field, which moves the
    overlap by a couple of percent and would swamp this gate with an effect that is a mode-shape
    correction rather than TSHB -- see the next test."""
    fib = _clad_fiber(length_m=1.5, na=0.16)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    ase, _ = _plan(3, 2)
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22)
    deltas = {}
    for P_s in (1e-9, 1e-2):
        sig = [Signal(P_s, 1.550e-6)]
        s = ErYbAmplifier(ER, YB, fib, pumps, sig, ase, **kw).solve(n_nodes=41)
        r = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, ase, **kw).solve(n_nodes=41)
        deltas[P_s] = float(r.signal_gain_dB[0]) - float(s.signal_gain_dB[0])
    assert abs(deltas[1e-9]) < 5e-3                      # small-signal: the two agree
    assert deltas[1e-2] < -0.05                          # saturated: resolved gain is LOWER
    assert abs(deltas[1e-2]) > 20.0 * abs(deltas[1e-9])


def test_exact_lp_profile_switch_is_a_separate_effect_from_hole_burning():
    """BIDIRECTIONAL-ADVERSARIAL. A resolved-minus-scalar delta on a V > 2.405 fiber is NOT all
    transverse hole burning: above cutoff the resolved solver takes the exact LP01 field while
    the mean-field one keeps the Marcuse Gaussian, and the two overlaps differ by a couple of
    percent. This gate separates the two so neither can be reported as the other --
    `signal_modes=["flat"]` switches the signal's OWN hole burning off while leaving the core
    pump resolved, so the two contributions are read independently."""
    from dynameta.optics.fiber_amp import overlap_gamma
    fib = _clad_fiber(length_m=1.0, a=2.0e-6, n_er=4e25)
    pumps = [Pump(0.3, 0.976e-6, "fwd", cladding=False)]     # CORE pump, V(976) = 2.575
    sig = [Signal(2e-3, 1.550e-6)]
    kw = dict(n_yb_m3=4e26, k_tr_m3_s=1.11e-21)

    def g(amp):
        return float(amp.solve(n_nodes=41).signal_gain_dB[0])

    g_scalar = g(ErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw))
    g_unif = g(ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                                     uniform_illumination=True, **kw))
    g_pump_only = g(ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                                          signal_modes=["flat"], **kw))
    g_full = g(ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw))
    assert abs(g_unif - g_scalar) < 1e-3                     # the uniform limit IS the scalar
    # the pump's exact-LP overlap is genuinely larger than its Marcuse one above cutoff
    amp = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw)
    assert float(amp.gamma_quad[0]) > float(overlap_gamma(fib, 0.976e-6)) * 1.005
    # and BOTH contributions are individually resolvable, neither invisible behind the other
    assert abs(g_pump_only - g_unif) > 1e-3                  # the profile switch
    assert abs(g_full - g_pump_only) > 1e-3                  # the signal's hole burning


def test_the_transfer_is_local_so_the_covariance_is_nonzero():
    """The transfer integral is INT k_tr n_Yb2(r) n_Er1(r) dA, not k_tr <n_Yb2><n_Er1> A. The
    difference -- `transfer_covariance` -- is exactly what resolving bought, and it is nonzero
    precisely because the flat cladding pump and the confined C-band signal give the Yb and the
    Er inversions DIFFERENT radial shapes."""
    fib = _clad_fiber(length_m=1.0)
    amp = ResolvedErYbAmplifier(ER, YB, fib, [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                                [Signal(1e-2, 1.550e-6)], None, n_yb_m3=8e25, k_tr_m3_s=2e-22)
    res = amp.solve(n_nodes=41)
    t = amp.energy_terms_from_result(res)
    frac = np.abs(t["transfer_covariance"]) / np.maximum(t["transfer_rate_per_m"], 1e-300)
    assert float(np.max(frac)) > 1e-3
    # and it vanishes when the illumination is made flat (nothing left to covary with)
    flat = ResolvedErYbAmplifier(ER, YB, fib, [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                                 [Signal(1e-2, 1.550e-6)], None, n_yb_m3=8e25, k_tr_m3_s=2e-22,
                                 uniform_illumination=True)
    tf = flat.energy_terms_from_result(flat.solve(n_nodes=41))
    assert float(np.max(np.abs(tf["transfer_covariance"])
                        / np.maximum(tf["transfer_rate_per_m"], 1e-300))) < 1e-12


def test_the_inversion_is_burned_hardest_where_the_signal_is_brightest():
    """The sign of the radial structure, with no reference to any other solver: a confined,
    saturating C-band signal depletes the Er metastable fraction ON AXIS relative to the dopant
    edge, while the flat cladding pump leaves the Yb inversion far more uniform."""
    fib = _clad_fiber(length_m=1.0)
    amp = ResolvedErYbAmplifier(ER, YB, fib, [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                                [Signal(5e-2, 1.550e-6)], None, n_yb_m3=8e25, k_tr_m3_s=2e-22)
    res = amp.solve(n_nodes=41)
    keep = res.meta["doped_mask"]
    f2 = res.meta["f2_rz"][-1, keep]                     # output end, where the signal is largest
    bb = res.meta["bbar_rz"][-1, keep]
    assert f2[0] < f2[-1]                                # hole burnt on axis
    spread_f2 = (f2[-1] - f2[0]) / f2[-1]
    spread_bb = abs(bb[-1] - bb[0]) / max(bb[-1], 1e-300)
    assert spread_f2 > 1e-3
    assert spread_bb < spread_f2


# ============================ the node kernel vs its scalar original =======================

@pytest.mark.parametrize("cfg", [
    dict(),
    dict(yb_coupled_fraction=0.85, k_tr2_m3_s=2e-22, yb_migration_rate_per_s=200.0),
    dict(k_back_m3_s=1e-24, a32_per_s=3e5),
    dict(upconversion_C_up=3e-24),
])
def test_the_vectorized_node_solver_matches_the_scalar_root_find(cfg):
    """The node kernel is the per-node generalization of ErYbAmplifier._solve_fbb. Fed IDENTICAL
    rates and uniform densities it must return the scalar's own root; a drift here is the one
    failure mode gate 1 could hide behind a compensating quadrature error. Bar: 1e-13 in every
    population, over four decades of pump rate."""
    from dynameta.optics.fiber_amp.transverse_eryb import _solve_populations_nodes
    fib = _clad_fiber()
    amp = ErYbAmplifier(ER, YB, fib, [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                        [Signal(1e-3, 1.550e-6)], None, n_yb_m3=8e25, k_tr_m3_s=2e-22, **cfg)
    rates = [(1e2, 1e1, 1e4, 1e3), (1e4, 1e3, 1e6, 1e5), (1e6, 1e5, 1e8, 1e7),
             (1e0, 0.0, 1e2, 0.0), (0.0, 0.0, 0.0, 0.0)]
    n = len(rates)
    arr = np.asarray(rates, float)
    p = {"n_er": np.full(n, amp._n_er), "n_yb": np.full(n, amp._n_yb),
         "fc": amp._fc, "k_tr": amp._k_tr, "k_tr2": amp._k_tr2, "w_mig": amp._w_mig,
         "k_back": amp._k_back, "a32": amp._a32, "inv_tau_er": 1.0 / amp._tau_er,
         "inv_tau_yb": 1.0 / amp._tau_yb, "c_up": amp.upconversion_C_up}
    f2, b2c, b2nc = _solve_populations_nodes(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], p)
    for i, (ra_e, re_e, ra_y, re_y) in enumerate(rates):
        if amp._two_pop:
            sf, sbc, sbn = amp._solve_fbb(ra_e, re_e, ra_y, re_y)
        else:
            sf, sbc = amp._solve_fb(ra_e, re_e, ra_y, re_y)
            sbn = None
        assert abs(f2[i] - sf) < 1e-13, (i, f2[i], sf)
        assert abs(b2c[i] - sbc) < 1e-13, (i, b2c[i], sbc)
        if sbn is not None:
            assert abs(b2nc[i] - sbn) < 1e-13, (i, b2nc[i], sbn)


def test_the_node_newton_finds_the_root_of_a_deliberately_awkward_residual():
    """The vectorized bracket must be as unconditionally robust as the scalar: H(0) >= 0,
    H(1) < 0, roots spread over the whole interval INCLUDING the ends, and one node with no
    drive at all (pinned at 0). A quadratic H is used so the Newton path is exercised rather
    than solved exactly in one step."""
    roots = np.array([1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9])

    def hdh(f):
        # H = (root - f)(1 + f), strictly decreasing on [0, 1], H(0) = root >= 0, H(1) < 0
        return (roots - f) * (1.0 + f), -(1.0 + f) + (roots - f)

    got = _bracketed_newton_nodes(hdh, roots.size)
    assert np.max(np.abs(got - roots)) < 1e-12

    def hdh_dead(f):
        out = np.full(f.shape, -1.0)
        return out, out

    assert np.array_equal(_bracketed_newton_nodes(hdh_dead, 3), np.zeros(3))


# ============================ separate Er and Yb radial profiles ===========================

def test_separate_dopant_profiles_change_the_answer_and_default_to_the_scalar_geometry():
    """Both profiles default to a top hat at the fiber's dopant radius, which is the scalar
    model's geometry -- so the default is not a new model. Widening the YTTERBIUM alone must
    then change the answer in the physically necessary direction: the extra Yb absorbs more pump
    but sits where there is no erbium to hand it to, so it stores excitation it can only lose to
    fluorescence and 1-um gain."""
    fib = _clad_fiber(length_m=2.0)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-3, 1.550e-6)]
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22)
    base = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw)
    explicit = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                                     er_profile=fib.b_dope_m, yb_profile=fib.b_dope_m, **kw)
    assert base.state_bytes(41) == explicit.state_bytes(41)
    r0 = base.solve(n_nodes=41)
    r1 = explicit.solve(n_nodes=41)
    assert np.array_equal(r0.power_W, r1.power_W)          # the same amplifier, spelled twice
    wide = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                                 yb_profile=2.0 * fib.b_dope_m, **kw).solve(n_nodes=41)
    assert wide.meta["yb_radius_m"] == 2.0 * fib.b_dope_m
    # the ytterbium beyond the erbium absorbs pump but has no acceptor to hand it to, so the
    # transfer efficiency COLLAPSES (0.176 -> 0.051 on this fixture) and the C-band gain follows.
    assert float(wide.meta["eta_transfer"]) < 0.5 * float(r0.meta["eta_transfer"])
    assert float(wide.signal_gain_dB[0]) < float(r0.signal_gain_dB[0])
    # MEASURED, and the opposite of the naive reading: the 1-um parasitic gain FALLS too
    # (19.47 -> 15.84 dB), because 4x the ytterbium shares the same launched pump, so the
    # inversion PER ION at the output end drops faster than the extra density adds gain. A
    # wide-Yb geometry at FIXED pump is therefore not a 1-um trade -- it is just more absorber.
    assert float(wide.meta["beta_yb_z"][-1]) < float(r0.meta["beta_yb_z"][-1])
    assert float(wide.meta["yb_parasitic_gain_dB"]) < float(r0.meta["yb_parasitic_gain_dB"])


def test_a_graded_dopant_shape_is_honoured_and_validated():
    fib = _clad_fiber(length_m=1.0)
    amp = ResolvedErYbAmplifier(
        ER, YB, fib, [Pump(1.0, 0.976e-6, "fwd", cladding=True)], [Signal(1e-3, 1.550e-6)],
        None, n_yb_m3=8e25, k_tr_m3_s=2e-22,
        er_profile=RadialDopant(fib.b_dope_m, lambda r: np.exp(-(r / fib.b_dope_m) ** 2)))
    n_er = amp.er_density_r_m3
    r = amp.grid.r_m
    inside = r <= fib.b_dope_m
    assert np.all(n_er[~inside] == 0.0)
    assert np.allclose(n_er[inside], fib.n_t_m3 * np.exp(-(r[inside] / fib.b_dope_m) ** 2))
    with pytest.raises(ValueError, match="non-negative"):
        RadialDopant(1e-6, lambda rr: -np.ones_like(rr)).density(1e25, np.array([0.0]))
    with pytest.raises(ValueError, match="radius_m must be > 0"):
        RadialDopant(-1.0)


# ============================ result contract and refusals =================================

def test_result_duck_types_a_steady_state_result_and_carries_the_ring_extras():
    fib = _clad_fiber(length_m=1.0)
    ase, yb_ase = _plan(3, 2)
    amp = ResolvedErYbAmplifier(ER, YB, fib, [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                                [Signal(1e-3, 1.550e-6)], ase, yb_ase=yb_ase, n_yb_m3=8e25,
                                k_tr_m3_s=2e-22, yb_coupled_fraction=0.9, k_tr2_m3_s=2e-22)
    res = amp.solve(n_nodes=41)
    from dynameta.optics.fiber_amp import analyze_noise
    nz = analyze_noise(res, 1.550e-6)                    # the noise layer reads it unchanged
    assert np.isfinite(nz.nf_dB)
    grid = res.meta["grid"]
    for key in ("f2_rz", "b2c_rz", "b2nc_rz", "bbar_rz"):
        assert res.meta[key].shape == (41, grid.size)
    assert res.meta["resolved"] is True
    assert res.meta["beta_yb_uncoupled_z"] is not None
    # the uncoupled pool sits HIGHER than the coupled one: it is pumped but never drained
    assert np.all(res.meta["beta_yb_uncoupled_z"] >= res.meta["beta_yb_coupled_z"] - 1e-15)
    # the quadrature cladding-pump overlap IS the analytic one, by construction
    from dynameta.optics.fiber_amp import cladding_pump_overlap
    assert amp.gamma_quad[0] == pytest.approx(cladding_pump_overlap(fib), rel=1e-12)
    assert eryb_mean_field_equivalent(amp).fiber is fib
    assert np.allclose(np.sum(amp.profiles * grid.dA_m2[None, :], axis=1), 1.0, atol=1e-12)


def test_reseed_protocol_is_type_preserving_and_drives_the_metrics_layer():
    """metrics.* and chain.AmplifierChain rebuild a stage through with_signals / with_pumps /
    without_ase; a class missing them raises AttributeError there (the audit A-3 failure mode).
    The clone must preserve THIS class and every opt-in of both layers -- the co-doped physics
    and the resolved geometry."""
    from dynameta.optics.fiber_amp import gain_spectrum, slope_efficiency
    fib = _clad_fiber(length_m=1.0)
    ase, yb_ase = _plan(3, 2)
    amp = ResolvedErYbAmplifier(ER, YB, fib, [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                                [Signal(1e-3, 1.550e-6)], ase, yb_ase=yb_ase, n_yb_m3=8e25,
                                k_tr_m3_s=2e-22, yb_coupled_fraction=0.9, k_tr2_m3_s=2e-22,
                                er_profile=2.0e-6, n_quad=16)
    for clone in (amp.with_pumps([Pump(0.5, 0.976e-6, "fwd", cladding=True)]),
                  amp.with_signals([Signal(1e-3, 1.560e-6)]),
                  amp.without_ase()):
        assert isinstance(clone, ResolvedErYbAmplifier)
        assert clone.er_profile == amp.er_profile and clone.yb_profile == amp.yb_profile
        assert clone.n_quad == 16
        assert eryb_mean_field_equivalent(clone)._fc == 0.9
        assert eryb_mean_field_equivalent(clone)._k_tr2 == 2e-22
    assert amp.without_ase().ase is None and amp.without_ase().yb_ase is None
    gs = gain_spectrum(amp, np.array([1.545e-6, 1.555e-6]))
    assert np.all(np.isfinite(gs.gain_dB))
    se = slope_efficiency(amp, np.array([0.5, 1.0]))
    assert np.isfinite(se.slope)


def test_scope_refusals():
    fib = _clad_fiber(length_m=1.0)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-3, 1.550e-6)]
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22)
    with pytest.raises(ValueError, match="photodarkening"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                              concentration=ConcentrationModel(pd_loss_per_m=0.1), **kw)
    # eryb.py's two 2026-09-15 temperature opt-ins act only through an axial T(z) profile, which
    # this class refuses, so they are refused BY NAME rather than carried and silently ignored
    from dynameta.optics.fiber_amp import (RATE_ARRHENIUS_CHENG_2022, RateTemperatureLaw,
                                           YbStarkThermal)
    with pytest.raises(ValueError, match="rate_temperature"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                              rate_temperature=RATE_ARRHENIUS_CHENG_2022, **kw)
    with pytest.raises(ValueError, match="yb_stark_thermal"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                              yb_stark_thermal=YbStarkThermal(delta_E_over_k_K=600.0),
                              **kw)
    # ... but an all-zero law IS the identity and must be accepted, exactly as eryb.py treats it
    ok = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None,
                               rate_temperature=RateTemperatureLaw(), **kw)
    assert ok.rate_temperature is None
    assert np.isfinite(ok.solve(n_nodes=21).signal_gain_dB[0])
    amp = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw)
    with pytest.raises(ValueError, match="set_temperature_profile"):
        amp.set_temperature_profile([0.0, 1.0], [300.0, 350.0])
    with pytest.raises(ValueError, match="PARALLEL list"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, signal_modes=[None, None], **kw)
    with pytest.raises(ValueError, match="string spelling"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, signal_modes=["fat"], **kw)
    # the two GEOMETRY refusals fire when the grid is built (the plan is lazy here exactly as it
    # is in ResolvedFiberAmplifier), so they are reached through .grid, not the constructor
    with pytest.raises(ValueError, match="inside the inner-cladding radius"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, r_max_m=20e-6, **kw).grid
    with pytest.raises(ValueError, match="does not exceed"):
        ResolvedErYbAmplifier(ER, YB, FiberSpec(3e-6, 0.2, 1e25, 1.0), pumps, sig, None,
                              r_max_m=1e-6, **kw).grid
    with pytest.raises(ValueError, match="er_profile"):
        ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, er_profile="wide", **kw)
    # the single-ion resolved class still refuses co-doping, and now says where to go
    with pytest.raises(ValueError, match="ResolvedErYbAmplifier"):
        ResolvedFiberAmplifier(object(), fib, pumps, sig, None)


def test_concentration_quenching_and_upconversion_reduce_exactly():
    """C_up and pair-induced quenching are LOCAL (a quadratic in the node's own f2, and an
    unbleachable absorption at the node's own dark density), so they must reduce to eryb.py
    exactly under uniform illumination -- which is why they are supported rather than refused."""
    fib = _clad_fiber(length_m=1.5)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-3, 1.550e-6)]
    conc = ConcentrationModel(c_up_m3_s=1.1e-24, pair_fraction=0.016,
                              pair_convention="delevaque")
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22, concentration=conc)
    s = ErYbAmplifier(ER, YB, fib, pumps, sig, None, **kw).solve(n_nodes=41)
    r = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, uniform_illumination=True,
                              **kw).solve(n_nodes=41)
    assert abs(float(r.signal_gain_dB[0]) - float(s.signal_gain_dB[0])) < 1e-3
    assert float(r.meta["n_er_dark_m3"]) > 0.0


def test_quadrature_refinement_does_not_move_the_answer():
    """Convergence: doubling the nodes per radial panel must not move the gain. The bar is the
    single-ion module's own (1e-10 relative on a local quantity); on an end-to-end gain through
    a relaxation it is stated in dB."""
    fib = _clad_fiber(length_m=1.0)
    pumps = [Pump(1.0, 0.976e-6, "fwd", cladding=True)]
    sig = [Signal(1e-2, 1.550e-6)]
    kw = dict(n_yb_m3=8e25, k_tr_m3_s=2e-22)
    g24 = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, n_quad=24,
                                **kw).solve(n_nodes=41).signal_gain_dB[0]
    g48 = ResolvedErYbAmplifier(ER, YB, fib, pumps, sig, None, n_quad=48,
                                **kw).solve(n_nodes=41).signal_gain_dB[0]
    assert abs(float(g48) - float(g24)) < 1e-6
