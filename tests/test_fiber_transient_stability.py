"""Transient-march gates of 2026-11-02 (workstream eryb_fix): three defects of the co-doped Er:Yb
march found while measuring its z_refine work, and the 2026-07-17 audit item S3-42.

DEFECTS (each gate FAILS on the code before the fix -- measured against HEAD 1b1f394's
dynamics.py + eryb.py):

  1. PHOTODARKENING MISSING FROM THE MARCH. simulate_transient_eryb's frozen-population gain g_s
     omitted the Yb photodarkening gray loss that the steady operator eryb._dP subtracts, so a
     march seeded at amp.solve() with a ConcentrationModel(pd_loss_per_m > 0) started OFF its own
     operator. Old: first frame 24.600 dB against solve() 23.877 dB (+0.72 dB) on the 4 m / 1 W
     cladding-pumped reference at 0.4 /m; new: +1.2e-4 dB.

  2. LAGGED FIELD COUPLING RINGS, UNFLAGGED. The quasi-static step holds the powers of the START
     of a step while the populations move; past a dt set by the operating point (0.25 ms on the
     4 m / 1 W cladding-pumped reference at 81 nodes) the march rings with a GROWING swing (x1.18
     per step at 0.8 ms, period 2). Old: quasi_static_valid / march_valid True, no warning, and
     assert_valid() passed a +-1.4 dB ring. New: dynamics._OscillationMonitor flags it
     (UnreliableResultWarning, march_valid False, assert_valid raises); ase_mode="auto" latches
     the field-implicit self-consistent step on detection and holds the march.

  3. UNCONVERGED SEED / SILENT NON-CONVERGENCE. ErYbAmplifier.solve() defaulted to relax = 1.0,
     which on the core-pumped 2 m reference at 1 W stops after 200 iterations at an endpoint
     residual of 1.3e6 and returns -4.98 dB (converged: +20.75 dB) with no warning; the co-doped
     march seeded from it. New: solve() defaults to relax="auto" (byte-identical wherever 1.0
     converges -- it is the first rung), warns (UnreliableResultWarning) on ANY unconverged
     return, and both marches seed from a converged solve or refuse.

S3-42 (2026-07-17 audit: "transient dynamics validated only ASE-free / single-signal /
co-pumped"). Gates on ASE-on, multi-signal, COUNTER-pumped marches of BOTH the single-ion and the
co-doped amplifier, each against an oracle independent of the march:
  * the STEADY LIMIT: a march from a cold start at constant drive lands on amp.solve() (a
    different algorithm: LSODA relaxation vs. exponential stepping + trapezoid propagation);
  * PER-STEP CONSERVATION: photon number (single ion) / energy (co-doped, via energy_terms)
    closes step by step, FIRST ORDER in dt (the quasi-static approximation's order) -- the
    population side from the integrator, the field side from the propagated endpoint fluxes;
  * a KNOWN ANALYTIC TRANSIENT: the pump-induced gain build-up rate 1/tau_eff = 1/tau +
    Gamma_p (sigma_a + sigma_e)_p P_p / (h nu_p A) of an unsaturated, undepleted amplifier,
    computed by hand from the ion's spectra;
  * the step-stability monitor on the counter-pumped points (period-2 co-doped, period-~4
    single-ion), which is where the lagged coupling turned out to bite hardest.
"""

import warnings

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.core.numerics import trapz
from dynameta.optics.fiber_amp import (AseBand, ConcentrationModel, FiberAmplifier, FiberSpec, Pump,
                                       Signal, erbium, simulate_transient, ytterbium)
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier
from dynameta.optics.fiber_amp.waveguide import overlap_gamma
from dynameta.warning_types import UnreliableResultWarning

ER_AL = erbium("aluminosilicate")
YB_P = ytterbium("phosphosilicate")
ER = erbium()


def _clad(*, bins=8, pump_dir="fwd", signals=(1.0e-3,), conc=None, yb_ase=None, L=4.0):
    """The cladding-pumped Er:Yb transient reference of test_fiber_eryb_transient._clad_amp
    (2.8 um, NA 0.20, N_Er 2e25, N_Yb 2e26, k_tr 2e-22, 125 um cladding, 1 W at 976 nm)."""
    lams = (1.550e-6, 1.535e-6)
    return ErYbAmplifier(ER_AL, YB_P, FiberSpec(2.8e-6, 0.20, 2.0e25, L, clad_radius_m=62.5e-6),
                         [Pump(1.0, 0.976e-6, pump_dir, cladding=True)],
                         [Signal(p, lams[i]) for i, p in enumerate(signals)],
                         AseBand(1.520e-6, 1.570e-6, bins), n_yb_m3=2.0e26, k_tr_m3_s=2.0e-22,
                         concentration=conc, yb_ase=yb_ase)


def _core(P=0.6, L=2.0):
    """The core-pumped Er:Yb reference (test_fiber_eryb_transient._core_amp) at a higher pump."""
    return ErYbAmplifier(ER_AL, YB_P, FiberSpec(2.8e-6, 0.20, 2.0e25, L),
                         [Pump(P, 0.976e-6, "fwd")], [Signal(0.5e-3, 1.550e-6)],
                         AseBand(1.520e-6, 1.570e-6, 24), n_yb_m3=2.0e26, k_tr_m3_s=2.0e-22,
                         yb_ase=AseBand(1.000e-6, 1.100e-6, 8))


def _si_counter(pump=0.1):
    """Single-ion Er, COUNTER-pumped, two signals, ASE on: 1.6 um / NA 0.22 / 8e24 / 8 m, 100 mW
    at 980 nm backward, 0.1 mW at 1550 and at 1535 nm, 12 C-band bins."""
    return FiberAmplifier(ER, FiberSpec(1.6e-6, 0.22, 8.0e24, 8.0), [Pump(pump, 0.98e-6, "bwd")],
                          [Signal(1e-4, 1.55e-6), Signal(1e-4, 1.535e-6)],
                          AseBand(1.52e-6, 1.57e-6, 12))


def _co_counter(pump=1.0):
    """Co-doped, COUNTER-pumped, two signals, C-band AND Yb-band ASE: _clad's fiber, 1 W at
    976 nm backward, 0.5 mW at 1550 and 1535 nm, 8 C-band bins, 4 bins over 1000-1100 nm."""
    a = _clad(pump_dir="bwd", signals=(0.5e-3, 0.5e-3), yb_ase=AseBand(1.0e-6, 1.1e-6, 4))
    return a if pump == 1.0 else a.with_pumps([Pump(pump, 0.976e-6, "bwd", cladding=True)])


def _cold_grid(h, T):
    """A cold-start grid: 30 log-spaced points to h, then uniform h to T."""
    return np.concatenate([[0.0], np.logspace(-6, np.log10(h), 30), np.arange(2.0 * h, T, h)])


# ============================== defect 1: photodarkening ======================================

def test_codoped_march_carries_the_photodarkening_loss_of_its_steady_operator():
    pd = _clad(bins=4, conc=ConcentrationModel(pd_loss_per_m=0.4, pd_exponent=1.0))
    ss = pd.solve(n_nodes=81)
    # PREMISE: the photodarkening is a real term here (MEASURED: solve 23.877 dB vs 24.601 dB
    # with the ConcentrationModel absent), so a march that drops it cannot hide.
    assert float(_clad(bins=4).solve(n_nodes=81).signal_gain_dB[0] - ss.signal_gain_dB[0]) > 0.5
    tr = simulate_transient(pd, np.arange(101) * 0.2e-3, n_nodes=81)
    assert tr.meta["march_valid"] and tr.meta["z_refine"] == 1
    # the first frame IS the solve's state propagated by the march's operator: OLD +0.722 dB
    # (24.600 vs 23.877), NEW +1.2e-4 dB (the 81-node trapezoid vs LSODA)
    assert abs(float(tr.signal_gain_dB[0, 0] - ss.signal_gain_dB[0])) < 2e-3
    # and it holds there: NEW +5.96e-4 dB over 20 ms, the same +5.94e-4 the pd-free march
    # drifts (mesh), i.e. nothing left from the photodarkening
    assert abs(float(tr.signal_gain_dB[-1, 0] - tr.signal_gain_dB[0, 0])) < 1.5e-3
    assert abs(float(tr.pump_out_W[0, 0]) - float(ss.power_W[ss.kind.index("pump"), -1])) \
        < 1e-3 * float(ss.power_W[ss.kind.index("pump"), -1])     # OLD 0.1959 vs 0.16588 W


def test_photodarkening_free_concentration_model_leaves_the_march_bit_identical():
    # pd_loss_per_m = 0 subtracts an exact 0.0 -- the pre-fix arithmetic, bit for bit
    t = np.arange(11) * 0.2e-3
    a = simulate_transient(_clad(bins=4), t, n_nodes=41)
    b = simulate_transient(_clad(bins=4, conc=ConcentrationModel(pd_loss_per_m=0.0)), t,
                           n_nodes=41)
    assert np.array_equal(a.nbar2_zt, b.nbar2_zt)
    assert np.array_equal(a.signal_gain_dB, b.signal_gain_dB)


# ============================== defect 2: lagged-coupling ring ================================

def test_lagged_coupling_ring_is_flagged_and_assert_valid_refuses_it():
    amp = _clad()
    # PREMISE (no false positive): at dt = 0.2 ms the same march decays onto its fixed point
    # (alternation x0.92 per step) and holds within 5e-4 dB.
    ok = simulate_transient(amp, np.arange(121) * 0.2e-3, n_nodes=81)
    assert ok.meta["march_valid"] and ok.meta["oscillation_step"] is None
    assert float(np.ptp(ok.signal_gain_dB[:, 0])) < 1e-3
    # at 0.8 ms it rings with a growing swing (MEASURED x1.18 per step, +-0.18 dB by 40 ms)
    with pytest.warns(UnreliableResultWarning, match="GROWING swing"):
        bad = simulate_transient(amp, np.arange(51) * 0.8e-3, n_nodes=81)
    g = bad.signal_gain_dB[:, 0]
    assert float(np.ptp(g[-10:])) > 0.1                       # the ring is real and large
    assert bad.meta["march_valid"] is False and bad.meta["quasi_static_valid"] is False
    assert 1.1 < bad.meta["oscillation_growth_per_step"] < 1.3
    assert bad.meta["oscillation_step"] <= 12                 # caught early (MEASURED 8)
    # the A-7 margins are NOT what caught it -- they sit far inside their limits
    assert bad.meta["max_ase_to_launched"] < 0.1 and bad.meta["max_ase_gain_integral"] < 10.0
    with pytest.raises(RuntimeError, match="oscillation"):
        bad.assert_valid()


def test_auto_mode_latches_the_implicit_step_and_holds_the_march():
    amp = _clad()
    ref = simulate_transient(amp, np.arange(121) * 0.2e-3, n_nodes=81)      # the stable march
    tr = simulate_transient(amp, np.arange(51) * 0.8e-3, n_nodes=81, ase_mode="auto")
    assert tr.meta["march_valid"] and tr.meta["oscillation_latched_self_consistent"]
    assert tr.meta["quasi_static_valid"] is False           # the quasi-static step DID fail
    assert tr.meta["n_self_consistent_steps"] >= 40          # latched from step ~8 on
    # MEASURED: 24.4919 dB at the end, the dt = 0.2 ms quasi-static march's own fixed point
    assert abs(float(tr.signal_gain_dB[-1, 0] - ref.signal_gain_dB[-1, 0])) < 1e-4
    assert float(np.ptp(tr.signal_gain_dB[:, 0])) < 2e-3
    tr.assert_valid()


def test_ring_monitor_abstains_on_steps_and_nyquist_forcing():
    # a pump STEP and a signal toggled EVERY step (a genuine period-2 forcing) at a stable dt:
    # neither may be flagged. The toggled drive is the one case where a period-2 state history
    # is physics, and the monitor's drive check is what keeps it quiet.
    amp = _clad(bins=4)
    t = np.arange(121) * 1e-4
    step = simulate_transient(amp, t, n_nodes=41,
                              pump_drive=lambda tt: np.array([1.0 if tt < 5e-3 else 0.3]))
    assert step.meta["march_valid"] and step.meta["oscillation_step"] is None
    assert float(step.signal_gain_dB[0, 0] - step.signal_gain_dB[-1, 0]) > 2.0   # it moved
    tog = simulate_transient(amp, t, n_nodes=41, signal_drive=lambda tt: np.array(
        [1e-3 if int(round(tt / 1e-4)) % 2 == 0 else 0.5e-3]))
    assert tog.meta["march_valid"] and tog.meta["oscillation_step"] is None


# ============================== defect 3: converged seed ======================================

def test_codoped_solve_defaults_to_a_converged_answer_and_warns_when_it_is_not():
    amp = _core(P=0.6)
    # relax = 1.0 (the OLD default) does not converge here, and now says so
    with pytest.warns(UnreliableResultWarning, match="did NOT converge"):
        r1 = amp.solve(n_nodes=21, relax=1.0)
    assert r1.meta["converged"] is False
    assert float(r1.signal_gain_dB[0]) < 0.0                 # MEASURED -6.5 dB: a non-answer
    # the default walks the ladder and converges at relax = 0.5 (MEASURED +19.10 dB)
    r = amp.solve(n_nodes=21)
    assert r.meta["converged"] and r.meta["relax_attempts"] == (1.0, 0.5)
    assert float(r.signal_gain_dB[0]) > 15.0


def test_codoped_march_seeds_from_a_converged_solve():
    # OLD: the march seeded from solve()'s relax = 1.0 default and started at -6.5 dB. (21
    # nodes on this core-pumped fiber are deliberately under-resolved -- jump 17.7 -- to keep the
    # relaxation solves cheap; that z-mesh warning is expected and is not what is under test.)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*z mesh does NOT resolve",
                                category=UnreliableResultWarning)
        tr = simulate_transient(_core(P=0.6), np.array([0.0, 1e-5]), n_nodes=21, z_refine=1)
    assert tr.meta["mesh_resolved"] is False
    assert float(tr.signal_gain_dB[0, 0]) > 15.0             # MEASURED 19.18 dB


def test_march_refuses_an_unconverged_seed(monkeypatch):
    amp = _clad(bins=2)
    real = ErYbAmplifier._solve_once

    def never_converges(self, **kw):
        res = real(self, **kw)
        res.meta["converged"] = False
        return res
    monkeypatch.setattr(ErYbAmplifier, "_solve_once", never_converges)
    with pytest.raises(RuntimeError, match="did NOT converge"):
        simulate_transient(amp, np.linspace(0.0, 1e-4, 3), n_nodes=21)
    # an explicit seed needs no solve and is unaffected
    tr = simulate_transient(amp, np.linspace(0.0, 1e-4, 3), n_nodes=21, nbar2_0=(0.3, 0.1))
    assert np.all(np.isfinite(tr.signal_gain_dB))


# ============================== S3-42: analytic transient =====================================

@pytest.mark.parametrize("p_pump", [3e-3, 10e-3])
def test_single_ion_pump_buildup_rate_matches_the_analytic_tau_eff(p_pump):
    # A 0.2 m, N = 1e23 Er fiber absorbs 0.4% of the pump (undepleted) and a 10 nW signal is
    # 3e-5 of its saturation power, so every node sees the same pump and the inversion builds up
    # from n2 = 0 as n2_ss (1 - exp(-B t)) with the textbook
    #     B = 1/tau + Gamma_p (sigma_a + sigma_e)(lambda_p) P_p / (h nu_p A_dope)
    # and the dB gain (linear in n2) relaxes at the same B. B is computed HERE from the ion's
    # spectra; the march never sees this formula.
    fib = FiberSpec(2.0e-6, 0.20, 1.0e23, 0.2)
    lp = 0.98e-6
    amp = FiberAmplifier(ER, fib, [Pump(p_pump, lp, "fwd")], [Signal(1e-8, 1.55e-6)], None)
    B = (float(overlap_gamma(fib, lp)) * (ER.sigma_a.sigma(lp) + ER.sigma_e.sigma(lp)) * p_pump
         / (H_PLANCK * C_LIGHT / lp * fib.a_dope_m2) + 1.0 / ER.tau_s)
    assert B * ER.tau_s > 3.0          # PREMISE: the pump term dominates 1/tau (3.4 and 9.2)
    t = np.arange(81) * (0.05 / B)
    tr = simulate_transient(amp, t, n_nodes=21, nbar2_0=0.0)
    assert tr.meta["march_valid"]
    g = tr.signal_gain_dB[:, 0]
    g_ss = float(amp.solve(n_nodes=21).signal_gain_dB[0])
    y = np.log((g_ss - g) / (g_ss - g[0]))
    m = t <= 3.0 / B
    B_fit = -np.polyfit(t[m], y[m], 1)[0]
    # MEASURED 2026-11-02: -1.16e-3 (3 mW) and -8.9e-4 (10 mW) -- the 0.2% mean pump depletion
    # along the 0.2 m. Dropping sigma_e from B (the classic error) would be ~-50%.
    assert abs(B_fit / B - 1.0) < 3e-3, (B_fit, B)
    assert float(np.max(np.abs(np.polyval(np.polyfit(t[m], y[m], 1), t[m]) - y[m]))) < 2e-3


# ============================== S3-42: counter-pumped, multi-signal, ASE on ===================

def _photon_rates(amp, tr, i):
    """(dN_ex/dt from the populations over step i -> i+1, the step-averaged net stimulated photon
    absorption minus spontaneous decay from the FIELDS) [1/s] for a single-ion march. Field side:
    every channel's photon flux in minus out (endpoint fluxes of the propagated profile), minus
    what background loss and ESA removed, plus the spontaneous photons the ASE channels captured
    -- i.e. the net stimulated exchange with the ions, by photon bookkeeping alone."""
    ch, _bc, u, is_ase, _kind = amp._plan()
    A, na, z = amp.fiber.a_dope_m2, amp._n_active, tr.z_m
    hnu = H_PLANCK * C_LIGHT / ch.lambda_m
    m = amp.ase.m_modes if amp.ase else 2

    def field(k):
        P, n2 = tr.power_zt[k], tr.nbar2_zt[k]
        Phi = P / hnu[:, None]
        phi_in = sum(Phi[c, 0] if u[c] > 0 else Phi[c, -1] for c in range(len(u)))
        phi_out = sum(Phi[c, -1] if u[c] > 0 else Phi[c, 0] for c in range(len(u)))
        lost = (ch.gamma[:, None] * na * ch.sigma_esa[:, None] * n2[None, :]
                + ch.loss_per_m[:, None]) * Phi
        spont = (is_ase[:, None] * ch.gamma[:, None] * na * ch.sigma_e[:, None] * m
                 * ch.dnu_hz[:, None]) * n2[None, :]
        stim = phi_in - phi_out - trapz(np.sum(lost, 0), z) + trapz(np.sum(spont, 0), z)
        return stim - A * na * trapz(n2, z) / ch.tau_s, phi_in

    dt = tr.t_s[i + 1] - tr.t_s[i]
    dN = A * na * (trapz(tr.nbar2_zt[i + 1], z) - trapz(tr.nbar2_zt[i], z)) / dt
    a, phi_in = field(i)
    b, _ = field(i + 1)
    return dN, 0.5 * (a + b), phi_in


def test_single_ion_counter_pumped_march_reaches_the_steady_state_from_cold():
    amp = _si_counter()
    ss = amp.solve(n_nodes=81)
    assert ss.meta["converged"]
    tr = simulate_transient(amp, _cold_grid(0.1e-3, 80e-3), n_nodes=81, nbar2_0=0.05)
    assert tr.meta["march_valid"] and tr.meta["oscillation_step"] is None
    assert np.all(tr.signal_gain_dB[0] < -15.0)              # really started cold (-18.5 / -42.4)
    # MEASURED -6.8e-5 / -1.3e-4 dB: both signals, through counter-propagating pump and ASE
    assert np.all(np.abs(tr.signal_gain_dB[-1] - ss.signal_gain_dB) < 1e-3)
    assert np.all(tr.ase_bwd_W[-1] > 0.0) and np.all(tr.ase_fwd_W[-1] > 0.0)


def test_single_ion_counter_pumped_photon_number_closes_first_order_in_dt():
    # Pump step 30 -> 100 mW (counter-propagating), seeded at the 30 mW steady state, so the
    # reservoir is moving fast (dN/dt ~ 0.5 of the launched photon flux) at the sampled step.
    amp = _si_counter()
    seed = np.asarray(_si_counter(pump=0.03).solve(n_nodes=81).nbar2_z, float)
    out = []
    for dt in (4e-6, 2e-6, 1e-6):
        tr = simulate_transient(amp, np.arange(int(round(60e-6 / dt)) + 1) * dt, n_nodes=81,
                                nbar2_0=seed, store_profiles=True, z_refine=1)
        assert tr.meta["march_valid"]
        dN, fld, phi_in = _photon_rates(amp, tr, int(round(40e-6 / dt)))
        out.append((abs(dN - fld) / phi_in, abs(dN) / phi_in))
    assert min(o[1] for o in out) > 0.3, out                  # PREMISE: really moving
    r = [o[0] for o in out]
    # MEASURED 2026-11-02 at 161 nodes: 1.12e-3 / 5.03e-4 / 2.29e-4 -- ratios 2.2, 2.2
    assert r[0] < 3e-3 and r[-1] < 6e-4, r
    assert r[0] / r[1] > 1.7 and r[1] / r[2] > 1.7, r


def test_single_ion_counter_pumped_period_four_ring_is_flagged():
    # The lagged coupling of a COUNTER-pumped single-ion amplifier rings with period ~4, not 2,
    # at dt = 3.2 ms (18.11 dB -> 15.6..21.6 dB within 80 steps); the monitor's windowed
    # (any-period) test catches it, the period-2 fast path alone would not.
    amp = _si_counter()
    with pytest.warns(UnreliableResultWarning, match="GROWING swing"):
        tr = simulate_transient(amp, np.arange(41) * 3.2e-3, n_nodes=81)
    assert tr.meta["march_valid"] is False and tr.meta["oscillation_step"] is not None
    d = np.diff(tr.signal_gain_dB[:, 0])
    assert np.sum(d[1:] * d[:-1] < 0) < 0.8 * (d.size - 1)   # NOT a pure period-2 alternation


def _co_closure(amp, tr, i0, launched):
    """Step-averaged energy residuals of a co-doped march step i0 -> i0+1 (the
    test_fiber_eryb_transient._step_closure bookkeeping): 'flux' from the propagated endpoints,
    'rate' from INT q_opt dz; both against d(stored)/dt."""
    def exiting(P, d):
        return float(np.sum([P[k, -1] if d[k] > 0 else P[k, 0] for k in range(P.shape[0])]))
    dt = tr.t_s[i0 + 1] - tr.t_s[i0]
    U, net, qn = {}, {}, {}
    for i in (i0, i0 + 1):
        P = tr.power_zt[i]
        te = amp.energy_terms(P, tr.nbar2_zt[i], tr.meta["beta_yb"][i])
        D = float(trapz(te["dissipation"], tr.z_m))
        U[i] = amp.stored_energy_J(tr.nbar2_zt[i], tr.meta["beta_yb"][i], tr.z_m)
        net[i] = launched - exiting(P, tr.plan.direction) - D
        qn[i] = float(trapz(te["q_optical"], tr.z_m)) - D
    dU = (U[i0 + 1] - U[i0]) / dt
    return (abs(0.5 * (net[i0] + net[i0 + 1]) - dU) / launched,
            abs(0.5 * (qn[i0] + qn[i0 + 1]) - dU) / launched, abs(dU) / launched)


def test_codoped_counter_pumped_march_reaches_the_steady_state_from_cold():
    amp = _co_counter()
    ss = amp.solve(n_nodes=81)
    assert ss.meta["converged"]
    tr = simulate_transient(amp, _cold_grid(0.1e-3, 80e-3), n_nodes=81, nbar2_0=(0.05, 0.05))
    assert tr.meta["march_valid"] and tr.meta["oscillation_step"] is None
    assert np.all(tr.signal_gain_dB[0] < -40.0)              # really cold (-44.0 / -98.7)
    # MEASURED +1.2e-4 / +2.2e-4 dB
    assert np.all(np.abs(tr.signal_gain_dB[-1] - ss.signal_gain_dB) < 1e-3)
    assert np.all(tr.ase_bwd_W[-1] > 0.0) and float(np.max(tr.meta["beta_yb"][-1])) > 0.0


def test_codoped_counter_pumped_energy_closes_first_order_in_dt():
    amp = _co_counter()
    r0 = _co_counter(pump=0.3).solve(n_nodes=81)
    seed = (np.asarray(r0.nbar2_z, float), np.asarray(r0.meta["beta_yb_z"], float))
    launched = 1.0 + 1.0e-3
    out = []
    for dt in (1e-6, 5e-7, 2.5e-7):
        tr = simulate_transient(amp, np.arange(int(round(3e-5 / dt)) + 1) * dt, n_nodes=81,
                                nbar2_0=seed, store_profiles=True, z_refine=1)
        assert tr.meta["march_valid"]
        out.append(_co_closure(amp, tr, int(round(2e-5 / dt)), launched))
    assert min(o[2] for o in out) > 0.3, out                  # PREMISE: dU/dt ~ 0.58 launched
    rate = [o[1] for o in out]
    # MEASURED 2026-11-02: 5.65e-5 / 2.86e-5 / 1.44e-5 -- ratios 1.97, 1.99
    assert rate[0] < 1.5e-4 and rate[-1] < 4e-5, rate
    assert rate[0] / rate[1] > 1.7 and rate[1] / rate[2] > 1.7, rate
    # the endpoint (flux) version adds only the 81-node mesh error (MEASURED <= 2.7e-5)
    assert all(o[0] < 1e-4 for o in out), out


def test_codoped_counter_pumped_ring_is_flagged_and_auto_holds():
    amp = _co_counter()
    ss = amp.solve(n_nodes=81)
    ok = simulate_transient(amp, np.arange(81) * 0.2e-3, n_nodes=81)
    assert ok.meta["march_valid"] and ok.meta["oscillation_step"] is None     # premise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnreliableResultWarning)
        bad = simulate_transient(amp, np.arange(41) * 0.8e-3, n_nodes=81)
    # MEASURED: +45 / -83 dB swings by 32 ms, previously with every flag True
    assert bad.meta["march_valid"] is False and bad.meta["oscillation_step"] is not None
    assert float(np.max(np.abs(bad.signal_gain_dB - ss.signal_gain_dB))) > 10.0
    fix = simulate_transient(amp, np.arange(41) * 0.8e-3, n_nodes=81, ase_mode="auto")
    assert fix.meta["march_valid"] and fix.meta["oscillation_latched_self_consistent"]
    # MEASURED +1.2e-4 / +2.2e-4 dB at the end (the dt-free fixed point of the march)
    assert np.all(np.abs(fix.signal_gain_dB[-1] - ss.signal_gain_dB) < 1e-3)
