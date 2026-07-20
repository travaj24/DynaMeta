"""Gates for AAA rational pole extraction from real-axis sweeps (roadmap item 5.5).

Gates (see docs/enz_bic_nonlinear_roadmap.md 5.5):
  0. AAA algorithm sanity: interpolates at support points; recovers a KNOWN rational function's
     poles / residues / zeros exactly (pure-algorithm test, no physics).
  6. Q convention: q_from_pole == optics.resonance.pole_q exactly.
  1. FP ETALON: real-axis transmission of the n=3.5 slab over modes m=3..6 -> AAA poles vs the
     optics.resonance exact poles (Re rel < 1e-6, Q rel < 1e-3, clean data). Complex-t (analytic)
     path places physical poles in Im<0 with NO in-band upper-half poles; the real-T path shows
     the conjugate-pair mirror and the Im<0 selection.
  2. NOISE: 0.1% multiplicative noise -> Q within a few %, count == number of true modes across
     10 seeds (no spurious high-Q pole survives the filter).
  3. FROISSART: deliberate over-fit on clean data -> raw poles contain doublets, the filter
     removes them, final count exact.
  4. LORENTZIAN/FANO + BACKGROUND: synthetic Fano + slowly-varying (transcendental) background:
     AAA (omega0, Q) vs construction (< 1e-4), and fano_fit agrees within a few %.
  5. RCWA BRIDGE (importorskip-guarded): a real 2-D lumenairy RCWA wavelength sweep with a
     guided-mode resonance -> AAA Q vs fano_fit Q within ~5%.

Run: python -m pytest tests/test_aaa_poles.py -q
"""
import importlib.util
import math

import numpy as np
import pytest

from dynameta.constants import C_LIGHT
from dynameta.optics.resonance import (
    layered_smatrix_complex, newton_refine, pole_q, smatrix_pole_func,
)
from dynameta.optics.aaa_poles import (
    AAAResult, aaa, find_resonances, q_from_pole, sweep_and_extract,
)
from dynameta.analysis import fano_fit

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None


# ------------------------------------------------------------------------------------------------
# Fabry-Perot closed form (symmetric slab in vacuum, s-pol, normal incidence) -- same as
# tests/test_resonance.py; the AAA poles are checked against the resonance-module exact poles.
# ------------------------------------------------------------------------------------------------
def _fp_pole(n, L, m):
    r12 = abs((n - 1.0) / (n + 1.0))
    return (C_LIGHT / (n * L)) * (m * math.pi - 1j * math.log(1.0 / r12))


def _fp_exact_poles(n, L, ms):
    """Exact scattering poles of the slab from optics.resonance (Newton on the pole function,
    seeded by the FP closed form) -- the ground truth AAA must reproduce."""
    func = smatrix_pole_func([(complex(n) ** 2, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)
    return {m: newton_refine(func, _fp_pole(n, L, m), tol=1e-13) for m in ms}


def _slab_t(omega, n, L):
    return layered_smatrix_complex(omega, [(complex(n) ** 2, L)], pol="s",
                                   n_super=1.0, n_sub=1.0).t


def _slab_T(omega, n, L):
    return layered_smatrix_complex(omega, [(complex(n) ** 2, L)], pol="s",
                                   n_super=1.0, n_sub=1.0).T


# ------------------------------------------------------------------------------------------------
# Gate 0a: AAA barycentric interpolation property
# ------------------------------------------------------------------------------------------------
def test_aaa_interpolates_at_support_points():
    z = np.linspace(-3.0, 3.0, 121).astype(complex)
    f = np.exp(z) / (1.0 + z ** 2)                          # smooth-ish complex test function
    res = aaa(z, f, tol=1e-13)
    assert isinstance(res, AAAResult)
    # Exact interpolation at each support point.
    vals = res(res.support_points)
    assert np.allclose(vals, res.support_values, rtol=0, atol=1e-12)
    # Uniform accuracy on the sample set.
    assert res.max_error <= 1e-11 * float(np.max(np.abs(f)))


# ------------------------------------------------------------------------------------------------
# Gate 0b: recover a KNOWN rational's poles / residues / zeros exactly (pure algorithm)
# ------------------------------------------------------------------------------------------------
def test_aaa_recovers_known_rational_poles_residues_zeros():
    p1, r1 = (2.0 - 1.0j), 3.0
    p2, r2 = (-1.0 - 0.5j), (1.0 - 2.0j)

    def f(z):
        return r1 / (z - p1) + r2 / (z - p2)

    z = np.linspace(-6.0, 6.0, 201).astype(complex)
    res = aaa(z, f(z), tol=1e-13)

    # Two finite poles, matched to the constructed ones.
    got1 = min(res.poles, key=lambda p: abs(p - p1))
    got2 = min(res.poles, key=lambda p: abs(p - p2))
    assert abs(got1 - p1) < 1e-9 and abs(got2 - p2) < 1e-9
    # Residues at those poles.
    k1 = int(np.argmin(np.abs(res.poles - p1)))
    k2 = int(np.argmin(np.abs(res.poles - p2)))
    assert abs(res.residues[k1] - r1) < 1e-8
    assert abs(res.residues[k2] - r2) < 1e-8
    # The single zero of f (numerator degree 1) is recovered and r(zero) ~ 0.
    z0_true = (r1 * p2 + r2 * p1) / (r1 + r2)               # solve r1(z-p2)+r2(z-p1)=0
    got0 = min(res.zeros, key=lambda z_: abs(z_ - z0_true))
    assert abs(got0 - z0_true) < 1e-8
    assert abs(res(np.array([z0_true]))[0]) < 1e-7


# ------------------------------------------------------------------------------------------------
# Gate 6: Q convention identical to optics.resonance.pole_q
# ------------------------------------------------------------------------------------------------
def test_q_from_pole_matches_resonance_pole_q():
    samples = [1.0e15 - 5.0e13j, 8.0e14 - 1.0e12j, 3.3 - 0.1j, 2.0 - 2.0j,
               -1.0e15 - 3.0e13j, 5.0 + 0.0j]
    for w in samples:
        assert q_from_pole(w) == pole_q(w)                 # byte-identical
    assert math.isinf(q_from_pole(4.2 + 0.0j))             # real pole -> inf


# ------------------------------------------------------------------------------------------------
# Gate 1: FP etalon poles from real-axis COMPLEX transmission (clean-data limit)
# ------------------------------------------------------------------------------------------------
def test_fp_etalon_complex_t_poles_vs_resonance_exact():
    n, L = 3.5, 1.0e-6
    ms = (3, 4, 5, 6)
    exact = _fp_exact_poles(n, L, ms)
    base = C_LIGHT / (n * L)
    omega = np.linspace(0.85 * 3 * math.pi * base, 1.12 * 6 * math.pi * base, 400)
    t = np.array([_slab_t(w, n, L) for w in omega])

    res = aaa(omega.astype(complex), t, tol=1e-13)
    for m in ms:
        ex = exact[m]
        got = min(res.poles, key=lambda p, e=ex: abs(p - e))
        assert abs(got.real - ex.real) <= 1e-6 * abs(ex.real)          # Re rel < 1e-6
        assert abs(q_from_pole(got) - pole_q(ex)) <= 1e-3 * pole_q(ex)  # Q rel < 1e-3
        assert got.imag < 0.0                                          # decaying (exp(-i w t))

    # PHYSICALITY (complex analytic data): clean data places every IN-BAND physical pole in the
    # lower half plane; there is NO in-band upper-half counterpart.
    lo, hi = float(omega.min()), float(omega.max())
    in_band_upper = [p for p in res.poles if lo <= p.real <= hi and p.imag > 0.0]
    assert in_band_upper == []

    # find_resonances returns exactly the 4 in-band modes, physical + sorted.
    reso = find_resonances(omega, t, tol=1e-13)
    assert len(reso) == 4
    assert all(r.omega_tilde.imag < 0.0 for r in reso)
    for r, m in zip(reso, ms):
        assert abs(q_from_pole(r.omega_tilde) - pole_q(exact[m])) <= 1e-3 * pole_q(exact[m])


def test_fp_etalon_real_transmittance_conjugate_pairs():
    # REAL transmittance T = |t|^2: real data forces conjugate-symmetric poles; the physical
    # (Im<0) member is selected and matches the exact pole Q, and its Im>0 mirror is present.
    n, L = 3.5, 1.0e-6
    ms = (4, 5)
    exact = _fp_exact_poles(n, L, ms)
    base = C_LIGHT / (n * L)
    omega = np.linspace(3.4 * math.pi * base, 5.6 * math.pi * base, 320)
    T = np.array([_slab_T(w, n, L) for w in omega])

    res = aaa(omega.astype(complex), T, tol=1e-12)
    assert res.real_data is True
    for m in ms:
        ex = exact[m]
        lower = min(res.poles, key=lambda p, e=ex: abs(p - e))
        assert lower.imag < 0.0
        # its conjugate mirror exists in the pole set (real data => conjugate symmetry)
        mirror = min(res.poles, key=lambda p, e=ex: abs(p - np.conj(e)))
        assert mirror.imag > 0.0
        assert abs(mirror - np.conj(lower)) <= 1e-4 * abs(lower)

    reso = find_resonances(omega, T, tol=1e-12)
    assert len(reso) == 2
    for r, m in zip(reso, ms):
        assert r.omega_tilde.imag < 0.0
        assert abs(q_from_pole(r.omega_tilde) - pole_q(exact[m])) <= 5e-3 * pole_q(exact[m])


# ------------------------------------------------------------------------------------------------
# Gate 2: noise robustness -- count == number of true modes, Q to a few %, no spurious high-Q
# ------------------------------------------------------------------------------------------------
def test_noise_robustness_count_and_q():
    n, L = 3.5, 1.0e-6
    ms = (4, 5)
    exact = _fp_exact_poles(n, L, ms)
    q_exact = {m: pole_q(exact[m]) for m in ms}
    base = C_LIGHT / (n * L)
    omega = np.linspace(3.5 * math.pi * base, 5.5 * math.pi * base, 300)
    T = np.array([_slab_T(w, n, L) for w in omega])

    counts = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        Tn = T * (1.0 + 1e-3 * rng.standard_normal(T.size))            # 0.1% multiplicative noise
        reso = find_resonances(omega, Tn, tol=1e-6, max_degree=30)
        counts.append(len(reso))
        # every surviving pole is a true mode within a few %, none spurious/high-Q
        for r in reso:
            m_near = min(ms, key=lambda mm: abs(exact[mm].real - r.omega_tilde.real))
            assert abs(r.omega_tilde.real - exact[m_near].real) <= 5e-3 * exact[m_near].real
            assert abs(r.Q - q_exact[m_near]) <= 0.05 * q_exact[m_near]  # Q within a few %
    assert counts == [2] * 10                                          # exact mode count, all seeds


# ------------------------------------------------------------------------------------------------
# Gate 3: Froissart doublets -- overfit clean data, filter removes the doublets, count exact
# ------------------------------------------------------------------------------------------------
def test_froissart_doublets_manufactured_then_filtered():
    n, L = 3.5, 1.0e-6
    ms = (3, 4, 5, 6)
    exact = _fp_exact_poles(n, L, ms)
    base = C_LIGHT / (n * L)
    omega = np.linspace(0.85 * 3 * math.pi * base, 1.12 * 6 * math.pi * base, 400)
    t = np.array([_slab_t(w, n, L) for w in omega])
    lo, hi = float(omega.min()), float(omega.max())

    # Deliberate over-fit (tol=0 forces AAA to max_degree) => Froissart doublets appear.
    raw = aaa(omega.astype(complex), t, tol=0.0, max_degree=40)
    raw_in_band_lower = [p for p in raw.poles if p.imag < 0.0 and lo <= p.real <= hi]
    assert len(raw_in_band_lower) > 4                                  # spurious doublets present

    # The genuine poles carry a residue orders of magnitude above the doublets'.
    def residue_at(p):
        d = p - raw.support_points
        return abs(np.sum(raw.weights * raw.support_values / d)
                   / (-np.sum(raw.weights / d ** 2)))
    mags = sorted(residue_at(p) for p in raw_in_band_lower)
    assert mags[-4] > 1e6 * mags[-5]                                   # clear gap: 4 real >> rest

    # The filter removes every doublet and returns exactly the 4 true modes.
    filt = find_resonances(omega, t, tol=0.0, max_degree=40)
    assert len(filt) == 4
    for r, m in zip(filt, ms):
        assert abs(q_from_pole(r.omega_tilde) - pole_q(exact[m])) <= 1e-3 * pole_q(exact[m])


# ------------------------------------------------------------------------------------------------
# Gate 4: Lorentzian / Fano + slowly-varying background; AAA vs construction, fano_fit agrees
# ------------------------------------------------------------------------------------------------
def test_fano_plus_background_vs_construction_and_fano_fit():
    w0 = 1.0e15
    gamma = 4.0e12                                          # FWHM
    q0 = 2.5
    q_true = w0 / gamma                                     # pole Q = omega0 / gamma (= 250)
    wg = np.linspace(w0 - 30 * gamma, w0 + 30 * gamma, 400)
    u = (wg - w0) / (30 * gamma)
    eps_r = 2.0 * (wg - w0) / gamma
    fano = (q0 + eps_r) ** 2 / (1.0 + eps_r ** 2)
    # gentle, NON-polynomial (transcendental) background -> AAA truly approximates it
    bg = 0.4 + 0.08 * np.cos(0.8 * u) + 0.03 * np.exp(0.3 * u)
    resp = bg + 0.25 * fano

    reso = find_resonances(wg, resp, tol=1e-10, max_degree=40)
    assert len(reso) >= 1
    r = min(reso, key=lambda rr: abs(rr.omega_tilde.real - w0))
    assert abs(r.omega_tilde.real - w0) <= 1e-4 * w0                   # omega0 rel < 1e-4
    assert abs(r.Q - q_true) <= 1e-3 * q_true                         # Q rel < 1e-3 (AAA is exact)

    ff = fano_fit(wg, resp)
    assert abs(ff.omega0 - w0) <= 1e-3 * w0
    assert abs(ff.Q - q_true) <= 0.06 * q_true                        # fano_fit within a few %
    assert abs(r.Q - ff.Q) <= 0.06 * ff.Q                             # AAA and fano_fit consistent


# ------------------------------------------------------------------------------------------------
# sweep_and_extract convenience: adaptive sampling + extraction on the FP etalon
# ------------------------------------------------------------------------------------------------
def test_sweep_and_extract_adaptive_fp_etalon():
    n, L = 3.5, 1.0e-6
    ms = (3, 4, 5, 6)
    exact = _fp_exact_poles(n, L, ms)
    base = C_LIGHT / (n * L)

    def solver(omega):
        return _slab_t(omega, n, L)

    sw = sweep_and_extract(solver, 0.85 * 3 * math.pi * base, 1.12 * 6 * math.pi * base,
                           n_initial=65, tol=1e-12)
    assert len(sw.resonances) == 4
    assert sw.omega.size >= 65
    for r, m in zip(sw.resonances, ms):
        assert abs(r.omega_tilde.real - exact[m].real) <= 1e-6 * abs(exact[m].real)
        assert abs(q_from_pole(r.omega_tilde) - pole_q(exact[m])) <= 1e-3 * pole_q(exact[m])


# ------------------------------------------------------------------------------------------------
# Gate 7 (adversarial ATTACK 1): the spurious-pole filter must NOT false-kill a GENUINE weak pole.
# A weakly-coupled resonance (residue 1e-6 of the dominant) and a close doublet (spacing ~ half a
# linewidth) are physical; a residue-magnitude floor relative to the global max residue killed the
# weak one (a dominant resonance set the residue scale).  The Froissart discriminator is the
# pole-zero COINCIDENCE, not the residue magnitude -- a genuine weak pole has no near-coincident
# zero, so it survives.  Regression for the aaa_poles.find_resonances filter fix.
# ------------------------------------------------------------------------------------------------
def test_weak_pole_and_doublet_not_false_killed():
    p_dom = 1.00e15 - (1.00e15 / (2 * 100)) * 1j                # Q = 100, dominant
    p_weak = 1.30e15 - (1.30e15 / (2 * 500)) * 1j               # Q = 500, well separated
    w_d, Q_d = 0.75e15, 300.0
    gam_d = w_d / Q_d
    half_lw = 0.5 * gam_d                                       # doublet spacing ~ half a linewidth
    p_d1 = (w_d - 0.5 * half_lw) - 0.5 * gam_d * 1j
    p_d2 = (w_d + 0.5 * half_lw) - 0.5 * gam_d * 1j
    poles = [p_dom, p_weak, p_d1, p_d2]
    residues = [1.0, 1.0e-6, 0.3, 0.3]                          # weak pole: residue 1e-6 of dominant

    def f(omega):
        z = np.asarray(omega, dtype=complex)
        return sum(r / (z - p) for p, r in zip(poles, residues))

    omega = np.linspace(0.6e15, 1.5e15, 1200)
    reso = find_resonances(omega, f(omega), tol=1e-13)
    assert len(reso) == 4                                       # ALL four kept (weak + doublet)
    # weak pole present with its true (high) Q
    weak = min(reso, key=lambda rr: abs(rr.omega_tilde.real - p_weak.real))
    assert abs(weak.omega_tilde.real - p_weak.real) <= 1e-6 * p_weak.real
    assert abs(weak.Q - pole_q(p_weak)) <= 1e-2 * pole_q(p_weak)
    assert abs(weak.residue) <= 1e-3                            # genuinely weak residue, still kept
    # doublet resolved into two distinct poles half a linewidth apart
    dbl = [rr for rr in reso if abs(rr.omega_tilde.real - w_d) < gam_d]
    assert len(dbl) == 2 and abs(dbl[0].omega_tilde.real - dbl[1].omega_tilde.real) > 0.3 * half_lw


# ------------------------------------------------------------------------------------------------
# Gate 8 (adversarial ATTACK 2): NO sweep-window bias -- clean-data Q is exact at any span (the AAA
# barycentric fit is an analytic continuation), and a resonance that nearly fills the window (Q is
# then a high-variance extrapolation) triggers the narrow-window RuntimeWarning.
# ------------------------------------------------------------------------------------------------
def test_no_window_bias_and_narrow_window_warns():
    import warnings
    w0, Q_true = 1.0e15, 200.0
    gamma = w0 / Q_true
    p = w0 - 0.5 * gamma * 1j
    A = 0.18 * (0.5 * gamma)                                    # ~30% contrast on the background

    def fn(w):
        w = np.asarray(w, dtype=complex)
        u = (w - w0) / (10 * gamma)
        return (0.6 + 0.08 * np.cos(0.7 * u) + 0.03 * np.exp(0.2 * u)) + A / (w - p)

    # clean-data Q is exact from a very narrow (0.25 FWHM) to a wide (40 FWHM) window
    for m in (0.25, 1.0, 4.0, 40.0):
        half = 0.5 * m * gamma
        w = np.linspace(w0 - half, w0 + half, 400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reso = find_resonances(w, fn(w), tol=1e-13)
        best = min(reso, key=lambda rr: abs(rr.omega_tilde.real - w0))
        assert abs(best.Q - Q_true) <= 1e-3 * Q_true           # unbiased at every span

    # a span of ~1 FWHM makes the resonance fill the window -> narrow-window warning fires
    half = 0.5 * 1.0 * gamma
    w = np.linspace(w0 - half, w0 + half, 400)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        find_resonances(w, fn(w), tol=1e-13)
    assert any("high-variance" in str(x.message) for x in rec)
    # a wide span (20 FWHM) does NOT warn
    half = 0.5 * 20.0 * gamma
    w = np.linspace(w0 - half, w0 + half, 400)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        find_resonances(w, fn(w), tol=1e-13)
    assert not any("high-variance" in str(x.message) for x in rec)


# ------------------------------------------------------------------------------------------------
# Gate 9 (adversarial ATTACK 3): GAIN pole (Im > 0 under exp(-i w t)) convention.  Complex analytic
# gain data must NOT yield a false DECAYING resonance (the physical Im<0 cut returns nothing); real
# intensity data is conjugate-ambiguous, so the Im<0 mirror is reported -- but it obeys the stated
# contract (Im < 0, Q > 0 with |Q| matching the gain pole), never a silent wrong-sign Q.
# ------------------------------------------------------------------------------------------------
def test_gain_pole_convention_no_wrong_sign_Q():
    w0, Qg = 1.0e15, 150.0
    gam_g = w0 / Qg
    p_gain = w0 + 0.5 * gam_g * 1j                             # Im > 0 = amplifying mode
    w = np.linspace(w0 - 10 * gam_g, w0 + 10 * gam_g, 400)

    # (a) COMPLEX analytic gain data: AAA places the pole in the UPPER half; no physical decaying
    # resonance -> find_resonances returns nothing (no false Im<0 pole, no wrong-sign Q).
    resp_c = 1.0 / (w.astype(complex) - p_gain)
    ac = aaa(w.astype(complex), resp_c, tol=1e-13)
    near = min(ac.poles, key=lambda z: abs(z.real - w0))
    assert near.imag > 0.0                                     # the true gain pole is Im > 0
    assert find_resonances(w, resp_c, tol=1e-13) == []

    # (b) REAL intensity data: conjugate-symmetric; the Im<0 mirror is returned with |Q| = gain Q,
    # and every returned pole honours the Im < 0 / Q > 0 contract (no silent sign error).
    resp_r = 1.0 / ((w - w0) ** 2 + (0.5 * gam_g) ** 2)
    reso = find_resonances(w, resp_r, tol=1e-12)
    assert len(reso) == 1
    r = reso[0]
    assert r.omega_tilde.imag < 0.0 and r.Q > 0.0
    assert abs(r.Q - Qg) <= 5e-3 * Qg


# ------------------------------------------------------------------------------------------------
# Gate 5: RCWA bridge -- real lumenairy 2-D RCWA guided-mode-resonance sweep, AAA Q vs fano_fit Q
# ------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")
def test_rcwa_bridge_gmr_aaa_vs_fano():
    """A grating-on-waveguide guided-mode resonance solved by the LIVE lumenairy RCWA bridge
    (no upstream complex-frequency support). AAA on the reflectance sweep and a Fano fit of the
    SAME data must agree on Q within ~5% -- the no-upstream-changes RCWA-pole demonstration."""
    from dynameta.optics.lumenairy_bridge import rcwa_stack_RT

    n_orders = 20
    Sx = 4 * n_orders + 1
    period = 900e-9
    duty = 0.5
    cell = np.full(Sx, 2.0, dtype=complex)                 # binary grating: eps 4 ridge / 2 groove
    cell[:int(round(duty * Sx))] = 4.0
    d_grating, d_wg, eps_wg, n_sub = 120e-9, 250e-9, 4.0, 1.5

    def R_of_lambda(lam):
        layers = [(cell, d_grating), (eps_wg, d_wg)]
        R, _T = rcwa_stack_RT(layers, n_sub, 1.0, lam, period_x=period, theta=0.0,
                              n_orders=n_orders, row=0)
        return float(R)

    lam = np.linspace(1388.0e-9, 1418.0e-9, 140)
    R = np.array([R_of_lambda(l) for l in lam])
    assert R.max() > 0.9 and R.min() < 0.1                 # a strong (high-contrast) GMR is present

    omega = 2.0 * math.pi * C_LIGHT / lam
    reso = find_resonances(omega, R, tol=1e-9, max_degree=40)
    assert len(reso) >= 1
    ff = fano_fit(omega, R)
    best = min(reso, key=lambda r: abs(r.omega_tilde.real - ff.omega0))
    assert best.omega_tilde.imag < 0.0                     # decaying pole
    assert abs(best.omega_tilde.real - ff.omega0) <= 1e-3 * ff.omega0   # same resonance frequency
    assert abs(best.Q - ff.Q) <= 0.05 * ff.Q               # AAA Q vs fano_fit Q within ~5%
