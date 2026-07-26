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
import pytest

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


def test_undepleted_sfg_complex_phase_vs_brute_force_integral():
    # PHASE gate (a sign flip of phase_matching_sinc's exp(-i dk L/2) is invisible to every
    # |A3|/sinc^2 gate above, yet corrupts the non-separable JSA phase -> Schmidt number).
    # Pin the FULL COMPLEX A3 of sfg_undepleted against the brute-force closed form
    # A3 = i kappa3 A1 A2 integral_0^L exp(-i dk z) dz, evaluated by raw quadrature.
    L = 3.0e-3
    a1, a2 = 1.0 * np.exp(0.4j), 1.0 * np.exp(-1.2j)     # complex inputs expose the phase
    z = np.linspace(0.0, L, 400001)
    for dk in (400.0, 900.0, -1500.0):
        spec = TWMSpec(omega1=W1, omega2=W2, d_eff=DEFF, length=L,
                       n1=1.5, n2=1.5, n3=1.5, dk_override=dk)
        cf = sfg_undepleted(spec, a1, a2)
        g = np.exp(-1j * dk * z)                          # exp(-i dk z), NOT +i (the sign under test)
        dz = z[1] - z[0]
        integ = dz * (g.sum() - 0.5 * g[0] - 0.5 * g[-1])  # trapezoid (numpy-version independent)
        a3_bf = 1j * spec.kappa(3) * a1 * a2 * integ
        assert abs(cf["A3_L"] - a3_bf) / abs(a3_bf) < 1e-5, (dk, cf["A3_L"], a3_bf)


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


# --------------------------------------------------- gate 4b: QPM SIGN convention (audit Q-7)
def test_qpm_closed_form_matches_integrator_for_BOTH_signs_of_dk():
    """Audit Q-7. ``qpm_period_for`` returns a POSITIVE period for either sign of dk, because the
    poling square wave carries BOTH first grating orders m = +-1 (equal weight 2/pi). The closed
    form (``phase_matching_sinc`` -> ``sfg_undepleted``) must therefore agree with what
    ``twm_propagate`` actually integrates (the square wave) at dk < 0 to the SAME tolerance as the
    shipped dk > 0 gate above -- before the fix the two oracles disagreed by 229x (an intensity
    factor 1.9e-5) with every Manley-Rowe / power diagnostic clean. dk < 0 is reachable from plain
    indices whenever n3 < n1, n2.

    Also pins that the SIGNED period Lambda = 2 pi m / dk is the SAME physical grating: the square
    wave is even in Lambda, so both oracles (and the integrator's max_step) must accept it and
    return the identical answer."""
    ndom = 20
    amp = 1.0
    for dk0 in (800.0, -800.0):
        Lam = qpm_period_for(dk0)
        assert Lam > 0.0                               # the physical (mask) period, either sign
        L = ndom * Lam
        common = dict(omega1=W1, omega2=W2, d_eff=DEFF, length=L, n1=1.5, n2=1.5, n3=1.5)
        spec_qpm = TWMSpec(dk_override=dk0, qpm_period=Lam, **common)
        spec_pm = TWMSpec(dk_override=0.0, **common)

        res = twm_propagate(spec_qpm, amp, amp, 0.0 + 0j, n_out=8 * ndom + 1)
        eff_qpm = abs(res.A3[-1]) ** 2
        eff_pm = abs(twm_propagate(spec_pm, amp, amp, 0.0 + 0j, n_out=201).A3[-1]) ** 2
        cf = sfg_undepleted(spec_qpm, amp, amp)

        # (a) same (2/pi)^2 first-order QPM law at both signs
        assert abs(eff_qpm / eff_pm - (2.0 / np.pi) ** 2) / (2.0 / np.pi) ** 2 < 1e-2, dk0
        # (b) the two shipped oracles agree -- the Q-7 gate, same 1e-3 tolerance as dk > 0
        assert abs(abs(cf["A3_L"]) ** 2 - eff_qpm) / eff_qpm < 1e-3, (dk0, cf["A3_L"], eff_qpm)
        # (c) SIGNED period == same physical grating (integrator included: max_step uses |Lambda|)
        spec_signed = TWMSpec(dk_override=dk0, qpm_period=2.0 * np.pi / dk0, **common)
        cf_s = sfg_undepleted(spec_signed, amp, amp)
        eff_s = abs(twm_propagate(spec_signed, amp, amp, 0.0 + 0j, n_out=8 * ndom + 1).A3[-1]) ** 2
        assert cf_s["A3_L"] == cf["A3_L"]                                  # byte-identical
        assert abs(eff_s - eff_qpm) / eff_qpm < 1e-12


def test_qpm_both_grating_orders_vs_brute_force_square_wave():
    """Audit Q-7 (oracle-independent). |Phi| from ``phase_matching_sinc`` vs a raw quadrature of
    the ACTUAL poling profile ``_qpm_sign`` -- the m = -1 band at dk = -2 pi/Lambda is physically
    present with the SAME strength as the m = +1 band, which is why one positive period matches
    either sign of dk."""
    from dynameta.optics.twm_reference import phase_matching_sinc, _qpm_sign

    dk0 = 800.0
    Lam = qpm_period_for(dk0)
    L = 20 * Lam                                       # integer number of poling periods
    z = np.linspace(0.0, L, 400001)
    dz = z[1] - z[0]
    s = _qpm_sign(z, Lam)
    for dk in (dk0, -dk0):
        g = s * np.exp(-1j * dk * z)
        brute = dz * (g.sum() - 0.5 * g[0] - 0.5 * g[-1]) / L
        phi = phase_matching_sinc(dk, L, Lam)
        assert abs(abs(phi) - abs(brute)) <= 1e-5 * abs(brute), (dk, phi, brute)
        assert abs(abs(brute) - 2.0 / np.pi) <= 1e-5 * (2.0 / np.pi)       # both bands (2/pi)


def _phi_square_wave_exact(dk, L, Lam):
    """EXACT (1/L) int_0^L sign(cos(2 pi z/Lam)) exp(-i dk z) dz, piecewise-analytic (the square
    wave is piecewise constant, so each segment integrates in closed form -- no quadrature error
    and no reference to the module under test)."""
    import math
    Lam = abs(float(Lam))
    js = np.arange(0, int(math.ceil(2.0 * L / Lam)) + 4)
    brk = Lam * (0.25 + 0.5 * js)
    edges = np.concatenate(([0.0], brk[(brk > 0.0) & (brk < L)], [L]))
    tot = 0.0 + 0.0j
    for a, b in zip(edges[:-1], edges[1:]):
        s = 1.0 if math.cos(2.0 * math.pi * (0.5 * (a + b)) / Lam) >= 0.0 else -1.0
        seg = ((b - a) + 0.0j if dk == 0.0
               else (np.exp(-1j * dk * b) - np.exp(-1j * dk * a)) / (-1j * dk))
        tot += s * seg
    return tot / L


def test_qpm_selects_the_nearest_ODD_order_not_just_m_plus_minus_1():
    """FIX-VERIFY W1 kill 3.  ``phase_matching_sinc`` selected only m = +-1, so a grating driven on
    a HIGHER odd order -- exactly what ``qpm_period_for(dk, order=3)`` and
    ``effective_deff_qpm(d, 3)`` exist to build -- fell in a hole: at dk = +-3 * 2 pi/Lambda the
    exact square-wave response is 2/(3 pi) = 0.212207 and the truncated form returned 2.4e-16, a
    silent 100% error, and the two order-3 helpers disagreed with it by fifteen orders of
    magnitude.  The nearest ODD order is now chosen elementwise with its own (2/(pi|m|))
    (-1)^((|m|-1)/2) Fourier weight."""
    from dynameta.optics.twm_reference import phase_matching_sinc, effective_deff_qpm

    dk0 = 800.0
    Lam = qpm_period_for(dk0)
    g = 2.0 * np.pi / Lam
    L = 20 * Lam
    # (a) every odd order, magnitude AND phase, against the piecewise-exact integral
    for m in (1, 3, 5, 7, 9, -1, -3, -5, -7):
        got = complex(phase_matching_sinc(m * g, L, Lam))
        ex = _phi_square_wave_exact(m * g, L, Lam)
        assert abs(got - ex) <= 1e-9 * abs(ex), (m, got, ex)
        assert abs(got) == pytest.approx(2.0 / (np.pi * abs(m)), rel=1e-9)

    # (b) the m-th-order helpers now agree with the function they are meant to describe
    for order in (1, 3, 5, 7):
        Lm = qpm_period_for(dk0, order=order)
        got = abs(complex(phase_matching_sinc(dk0, 20 * Lm, Lm)))
        assert got == pytest.approx(effective_deff_qpm(1.0, order), rel=1e-9)

    # (c) 0 <= dk < 2g is BYTE-IDENTICAL to the m = +-1 form (the pinned 120k-value golden)
    def m1_only(dkv, length, period):
        from dynameta.optics.twm_reference import _sinc
        dkv = np.asarray(dkv, dtype=float)
        gg = 2.0 * np.pi / float(period)
        dp, dm = dkv - gg, dkv + gg
        de = np.where(np.abs(dm) < np.abs(dp), dm, dp)[()]
        arg = de * length / 2.0
        return (2.0 / np.pi) * _sinc(arg) * np.exp(-1j * arg)

    rng = np.random.default_rng(0)
    for u in (np.linspace(-2 * g, 2 * g, 120001), rng.uniform(0.0, 2.0 * g, 20000)):
        for Ll in (L, 0.7 * L, 13.37 * Lam):
            assert np.array_equal(np.asarray(phase_matching_sinc(u, Ll, Lam)),
                                  np.asarray(m1_only(u, Ll, Lam)))
    # ... and |dk| > 2g is exactly where it must NOT be
    hi = np.linspace(2.001 * g, 6 * g, 5000)
    assert not np.any(np.asarray(phase_matching_sinc(hi, L, Lam))
                      == np.asarray(m1_only(hi, L, Lam)))

    # (d) reality/symmetry: Phi(-dk) == conj(Phi(dk)) and a SIGNED period is the same grating,
    #     bitwise, everywhere except the documented dk == 0 tie (m = +1 by convention).
    u = np.linspace(-8 * g, 8 * g, 8001)
    u = u[u != 0.0]
    a = np.asarray(phase_matching_sinc(u, L, Lam))
    assert np.array_equal(a, np.conj(np.asarray(phase_matching_sinc(-u, L, Lam))))
    assert np.array_equal(a, np.asarray(phase_matching_sinc(u, L, -Lam)))

    # (e) the residual the docstring quotes: <= 1.1e-2 relative inside whichever main lobe is
    #     matched (pre-fix this band held a 100%-error hole at dk/g ~ +-3)
    uu = np.linspace(-8 * g, 8 * g, 4001)
    ex = np.array([_phi_square_wave_exact(d, L, Lam) for d in uu])
    got = np.array([complex(phase_matching_sinc(d, L, Lam)) for d in uu])
    for thr, tol in ((0.5, 1.1e-2), (0.2, 3.3e-2)):
        msk = np.abs(ex) >= thr
        assert np.max(np.abs(got[msk] - ex[msk]) / np.abs(ex[msk])) <= tol, thr


def test_qpm_order_three_design_closed_form_matches_the_integrator():
    """FIX-VERIFY W1 kill 3, end to end: a THIRD-order poling mask (period 3x coarser, the reason
    higher-order QPM is used at all) must give the same A3(L) from ``sfg_undepleted``'s closed form
    as from ``twm_propagate``'s ODE march over the actual square wave.  Pre-fix the closed form
    returned ~0 for every order > 1."""
    dk0 = 800.0
    for order in (1, 3, 5):
        Lam = qpm_period_for(dk0, order=order)
        L = 20 * Lam
        spec = TWMSpec(dk_override=dk0, qpm_period=Lam, omega1=W1, omega2=W1, d_eff=DEFF,
                       length=L, n1=1.5, n2=1.5, n3=1.5)
        march = abs(twm_propagate(spec, 1.0, 1.0, 0.0 + 0j, n_out=241,
                                  rtol=1e-12, atol=1e-14).A3[-1]) ** 2
        closed = abs(sfg_undepleted(spec, 1.0, 1.0)["A3_L"]) ** 2
        assert closed == pytest.approx(march, rel=2e-2), order


# ------------------------------------------------------------------ gate 5: energy conservation
def test_total_power_conserved_lossless():
    spec = TWMSpec(omega1=W1, omega2=W2, d_eff=DEFF, length=0.04,
                   n1=2.0, n2=1.7, n3=2.3, dk_override=250.0)
    res = twm_propagate(spec, 7.0e8, 5.0e8, 1.0e8, n_out=193)
    assert res.power_residual < 1e-10
    # explicit: max deviation of the summed intensity from its start
    assert np.max(np.abs(res.total_power - res.total_power[0])) / res.total_power[0] < 1e-10
