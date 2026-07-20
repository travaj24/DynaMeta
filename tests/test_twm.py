"""Gates for the three-wave-mixing coupled-wave reference (roadmap 4.1).

Pure numpy/scipy; runs in CI. Gates:
  (1) Manley-Rowe photon-flux invariants N1+N3, N2+N3 conserved to 1e-12 across strongly
      depleted propagation incl. back-conversion;
  (2) undepleted limits: twm_propagate (weak inputs) == sfg_undepleted sinc^2(dk) and
      opa_gain cosh^2/sinh^2 to 1e-6;
  (3) DEGENERATE LIMIT: omega1 == omega2 reproduces the fdtd_chi2_shg_raman coupled-wave
      SHG oracle's efficiency for ITS exact configuration (the d_eff = chi2/2 + 1/2
      degeneracy-factor convention, documented in twm_reference);
  (4) first-order QPM efficiency == (2/pi)^2 x phase-matched (1%); a wrong period kills it;
  (5) total EM power conserved to 1e-10 in the lossless depleted integrator.
"""
import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.optics.twm_reference import (
    TWMSpec, sfg_undepleted, shg_undepleted, opa_gain, twm_propagate, qpm_period_for,
)

W1 = 2.0 * np.pi * 2.5e14      # 250 THz
W2 = 2.0 * np.pi * 1.9e14      # 190 THz (nondegenerate)
DEFF = 1.0e-11


# ------------------------------------------------------------------ gate 1: Manley-Rowe
def test_manley_rowe_invariants_strong_conversion():
    # Strong, comparable inputs -> A3 grows to (near-)full conversion then BACK-converts
    # (SFG<->DFG oscillation); the two photon-flux invariants must hold through it all to 1e-12.
    spec = TWMSpec(omega1=W1, omega2=W2, d_eff=DEFF, length=8.0e-5,
                   n1=1.5, n2=1.5, n3=1.5, dk_override=0.0)
    res = twm_propagate(spec, 8.0e8, 8.0e8, 0.0 + 0j, n_out=257)
    conv = float(np.max(res.N3) / res.N1[0])
    assert conv > 0.9                                  # near-complete conversion
    assert res.N3[-1] < np.max(res.N3) * 0.999         # back-conversion occurred (N3 turns over)
    assert res.mr13_residual < 1e-12
    assert res.mr23_residual < 1e-12


# ------------------------------------------------------------------ gate 2: undepleted limits
def test_undepleted_sfg_sinc2_vs_dk():
    L = 3.0e-3
    amp = 1.0                                          # weak: depletion ~1e-12, undepleted holds
    for dk in (0.0, 400.0, 900.0, 1500.0):
        spec = TWMSpec(omega1=W1, omega2=W2, d_eff=DEFF, length=L,
                       n1=1.5, n2=1.5, n3=1.5, dk_override=dk)
        cf = sfg_undepleted(spec, amp, amp)
        res = twm_propagate(spec, amp, amp, 0.0 + 0j, n_out=201)
        rel = abs(abs(res.A3[-1]) - abs(cf["A3_L"])) / abs(cf["A3_L"])
        assert rel < 1e-6, (dk, rel)
        # the SFG intensity really is sinc^2 in dk (spot-check the closed form itself)
        expect = (np.sinc(dk * L / 2.0 / np.pi)) ** 2
        assert abs(cf["sinc2"] - expect) < 1e-12


def test_undepleted_opa_cosh_sinh():
    # strong undepleted pump A3, weak signal A1, no idler; phase matched -> cosh^2 / sinh^2.
    L = 3.0e-3
    spec = TWMSpec(omega1=W1, omega2=W2, d_eff=DEFF, length=L,
                   n1=1.5, n2=1.5, n3=1.5, dk_override=0.0)
    pump, sig0 = 2.0e7, 1.0e2
    og = opa_gain(spec, pump, sig0, 0.0)
    res = twm_propagate(spec, sig0, 0.0 + 0j, pump, n_out=201)
    gain_prop = abs(res.A1[-1]) ** 2 / sig0 ** 2
    assert abs(gain_prop - og["signal_gain"]) / og["signal_gain"] < 1e-6
    # cosh^2/sinh^2 forms: signal cosh^2(gL), idler (k2/k1) sinh^2(gL)
    gL = og["gL"].real
    assert abs(og["signal_gain"] - np.cosh(gL) ** 2) / np.cosh(gL) ** 2 < 1e-9
    idler_prop = abs(res.A2[-1]) ** 2
    idler_cf = (spec.kappa(2) / spec.kappa(1)) * sig0 ** 2 * np.sinh(gL) ** 2
    assert abs(idler_prop - idler_cf) / idler_cf < 1e-6
    assert og["above_threshold"] and og["manley_rowe_residual"] < 1e-12


def test_opa_below_threshold_is_oscillatory():
    # dk large enough that kappa1 kappa2 |Ap|^2 < (dk/2)^2 -> g imaginary -> bounded (cos^2/sin^2),
    # NO exponential gain (the trig<->hyperbolic crossover).
    spec = TWMSpec(omega1=W1, omega2=W2, d_eff=DEFF, length=3.0e-3,
                   n1=1.5, n2=1.5, n3=1.5, dk_override=6.0e4)
    og = opa_gain(spec, 2.0e7, 1.0e2, 0.0)
    assert not og["above_threshold"]
    assert og["signal_gain"] < 2.0                     # bounded, not exponential


# ------------------------------------------------------------------ gate 3: degenerate SHG oracle
def test_degenerate_limit_matches_shg_oracle():
    # EXACT configuration of validation/fdtd_chi2_shg_raman.py (its GATE B coupled-wave SHG):
    #   f0 = 250 THz, n = sqrt(2), chi2 = 2e-11 m/V, L = 400 nm, pump peak A0 = 5e8 V/m.
    # Its undepleted SH field is (chi2 w0 L / (2 n c)) A0^2 (real-peak amplitude). twm uses
    # d_eff = chi2/2 and the 1/2 degeneracy factor (module docstring); shg_undepleted must
    # reproduce that field and the implied efficiency (I_SH/I_pump).
    f0 = 2.5e14
    w0 = 2.0 * np.pi * f0
    chi2 = 2.0e-11
    A0 = 5.0e8
    L = 400e-9
    n = np.sqrt(2.0)
    spec = TWMSpec(omega1=w0, omega2=w0, d_eff=chi2 / 2.0, length=L, n1=n, n2=n, n3=n)

    r = shg_undepleted(spec, A0)
    oracle_field = (chi2 * w0 * L / (2.0 * n * C_LIGHT)) * A0 ** 2
    assert abs(abs(r["A_s_L"]) - oracle_field) / oracle_field < 1e-9

    oracle_eff = (chi2 * w0 * L / (2.0 * n * C_LIGHT) * A0) ** 2
    assert abs(r["efficiency"] - oracle_eff) / oracle_eff < 1e-9

    # the DEPLETED degenerate integrator reproduces the undepleted closed form (depletion here
    # is ~eta ~ 5e-5), validating the degenerate coupled-wave equations + their 1/2 factor.
    res = twm_propagate(spec, A0, 0.0 + 0j, 0.0 + 0j, degenerate=True, n_out=65)
    assert abs(abs(res.A3[-1]) - abs(r["A_s_L"])) / abs(r["A_s_L"]) < 1e-3
    assert res.mr13_residual < 1e-12 and res.power_residual < 1e-10


# ------------------------------------------------------------------ gate 4: QPM
def test_qpm_two_over_pi_squared_and_wrong_period_kills():
    dk0 = 800.0                                        # deliberate phase mismatch
    Lam = qpm_period_for(dk0)                          # first-order poling period 2 pi / dk0
    ndom = 20
    L = ndom * Lam                                     # integer number of poling periods
    amp = 1.0                                          # undepleted so the (2/pi)^2 law is clean
    common = dict(omega1=W1, omega2=W2, d_eff=DEFF, length=L, n1=1.5, n2=1.5, n3=1.5)

    spec_qpm = TWMSpec(dk_override=dk0, qpm_period=Lam, **common)
    spec_pm = TWMSpec(dk_override=0.0, **common)       # true phase matching, uniform d_eff
    spec_wrong = TWMSpec(dk_override=dk0, qpm_period=Lam * 1.5, **common)

    eff_qpm = abs(twm_propagate(spec_qpm, amp, amp, 0.0 + 0j, n_out=8 * ndom + 1).A3[-1]) ** 2
    eff_pm = abs(twm_propagate(spec_pm, amp, amp, 0.0 + 0j, n_out=201).A3[-1]) ** 2
    eff_wrong = abs(twm_propagate(spec_wrong, amp, amp, 0.0 + 0j, n_out=8 * ndom + 1).A3[-1]) ** 2

    assert abs(eff_qpm / eff_pm - (2.0 / np.pi) ** 2) / (2.0 / np.pi) ** 2 < 1e-2
    assert eff_wrong / eff_pm < 0.05                   # wrong period -> conversion killed
    # closed form agrees with the integrator for the QPM efficiency too
    cf = sfg_undepleted(spec_qpm, amp, amp)
    assert abs(abs(cf["A3_L"]) ** 2 - eff_qpm) / eff_qpm < 1e-3


# ------------------------------------------------------------------ gate 5: energy conservation
def test_total_power_conserved_lossless():
    spec = TWMSpec(omega1=W1, omega2=W2, d_eff=DEFF, length=0.04,
                   n1=2.0, n2=1.7, n3=2.3, dk_override=250.0)
    res = twm_propagate(spec, 7.0e8, 5.0e8, 1.0e8, n_out=193)
    assert res.power_residual < 1e-10
    # explicit: max deviation of the summed intensity from its start
    assert np.max(np.abs(res.total_power - res.total_power[0])) / res.total_power[0] < 1e-10
