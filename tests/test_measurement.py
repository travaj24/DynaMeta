"""Measurement averaging (dynameta.measurement, v0.16): cone, fabrication scatter, spectrometer,
composition and the broadening fit.

Every gate compares the module with an INDEPENDENT calculation written here: an Airy thin-film
formula integrated over the cone by scipy's adaptive 2-D quadrature, the closed-form angular
average of a displaced mirror, the Faddeeva (wofz) closed forms of the Gaussian-averaged
Lorentzian amplitude and power, brute-force numerical convolutions, and synthetic data with known
parameters. A gate that compares the module with itself is labelled a consistency check.

All gates are new with the module: on v0.15.1 each fails at `import dynameta.measurement`
(ModuleNotFoundError). Pure numpy/scipy (plus `tmm` for one adapter gate); no solver.
"""
import cmath
import math
import warnings

import numpy as np
import pytest
from scipy import integrate
from scipy.special import wofz

from dynameta.core.numerics import trapz
from dynameta.measurement import (InstrumentFunction, MeasurementSetup, ParameterQuadrature,
                                  combine_parameters, design_plane_wave_model, fit_broadening,
                                  instrument_convolve, measure, normalize_to_reference,
                                  parameter_quadrature, patch_average, plane_wave_incidence,
                                  pupil_quadrature, scalar_model)
from dynameta.warning_types import UnreliableResultWarning

LAM = np.array([600e-9, 633e-9, 700e-9])


# ------------------------------------------------------------------------------------------------
# independent thin-film oracle (Airy, Byrnes-tmm p sign: r_p = -r_s at normal incidence)
# ------------------------------------------------------------------------------------------------
def _cos(n0, n, sin0):
    c = cmath.sqrt(1.0 - (n0 * sin0 / n) ** 2)
    return c if c.imag >= 0 else -c


def airy(lam, theta, pol, n0=1.0, n1=2.0, d=300e-9, n2=1.5):
    s0 = math.sin(theta)
    c0, c1, c2 = math.cos(theta), _cos(n0, n1, s0), _cos(n0, n2, s0)
    if pol == "s":
        r01 = (n0 * c0 - n1 * c1) / (n0 * c0 + n1 * c1)
        r12 = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    else:
        r01 = (n1 * c0 - n0 * c1) / (n1 * c0 + n0 * c1)
        r12 = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    ph = cmath.exp(2j * 2 * math.pi / lam * n1 * d * c1)
    return (r01 + r12 * ph) / (1 + r01 * r12 * ph)


def film_model(wavelengths_m, theta_deg=0.0, phi_deg=0.0, pol="s", **kw):
    """The planar film as a model (azimuth-independent)."""
    return np.array([airy(lam, math.radians(theta_deg), pol, **kw) for lam in wavelengths_m])


def _ray(lam, theta, phi, alpha, quantity, r_fn=airy):
    """Per-ray detected quantity, written out from the Richards-Wolf decomposition."""
    rs, rp = r_fn(lam, theta, "s"), r_fn(lam, theta, "p")
    a_s, a_p = math.sin(alpha - phi), math.cos(alpha - phi)
    e_par = a_s * a_s * rs - a_p * a_p * rp
    e_perp = a_s * a_p * (rs + rp)
    return {"power": abs(e_par) ** 2 + abs(e_perp) ** 2, "parallel": abs(e_par) ** 2,
            "crossed": abs(e_perp) ** 2, "re": e_par.real, "im": e_par.imag}[quantity]


def cone_oracle(lam, na, alpha, quantity, weight=lambda th: math.cos(th) * math.sin(th),
                na_inner=0.0, n=1.0, r_fn=airy):
    """INT over the cone (theta, phi) of weight(theta) x ray quantity, / INT weight. The default
    weight is the filled-pupil one: cos(theta) sin(theta) dtheta dphi = d^2 s / n^2."""
    th_hi, th_lo = math.asin(na / n), math.asin(na_inner / n)
    num = integrate.dblquad(lambda th, ph: weight(th) * _ray(lam, th, ph, alpha, quantity, r_fn),
                            0.0, 2 * math.pi, th_lo, th_hi, epsabs=1e-12, epsrel=1e-11)[0]
    den = integrate.dblquad(lambda th, ph: weight(th), 0.0, 2 * math.pi, th_lo, th_hi,
                            epsabs=1e-14, epsrel=1e-12)[0]
    return num / den


def test_tmm_library_oracle_sign_convention():
    """The test's Airy oracle uses the library's p sign (Byrnes tmm): checked against `tmm` itself
    at 35 deg, both polarizations."""
    tmm = pytest.importorskip("tmm")
    for pol in ("s", "p"):
        ref = tmm.coh_tmm(pol, [1.0, 2.0, 1.5], [np.inf, 300.0, np.inf], math.radians(35.0), 633.0)
        assert abs(airy(633e-9, math.radians(35.0), pol) - ref["r"]) < 1e-12


# ------------------------------------------------------------------------------------------------
# item 1: the cone
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("alpha_deg", [0.0, 90.0, 30.0])
def test_na_average_of_thin_film_matches_direct_cone_integral(alpha_deg):
    """Filled pupil NA 0.5, power / parallel / crossed analyzer and the coherent amplitude, against
    scipy dblquad over (theta, phi) with the cos(theta) sin(theta) Jacobian. Error budget: the
    default 4 x 16 rule integrates this smooth Airy cone to 1.9e-11 (measured, all quantities and
    wavelengths; dblquad runs at epsrel 1e-11); tolerance 1e-9. The Jacobian is load-bearing:
    weighting per solid angle instead (sin(theta) only) moves the power average by 1.1e-4 to
    3.4e-4 (measured), 1e5 x the tolerance."""
    na = 0.5
    p = pupil_quadrature(na, polarization_deg=alpha_deg)
    out = {}
    for an in (None, "parallel", "crossed"):
        out[an] = measure(film_model, LAM, pupil=p, analyzer=an)
    amp = measure(film_model, LAM, pupil=p, detection="amplitude")
    a = math.radians(alpha_deg)
    for i, lam in enumerate(LAM):
        assert abs(out[None].signal[i] - cone_oracle(lam, na, a, "power")) < 1e-9
        assert abs(out["parallel"].signal[i] - cone_oracle(lam, na, a, "parallel")) < 1e-9
        assert abs(out["crossed"].signal[i] - cone_oracle(lam, na, a, "crossed")) < 1e-9
        r_coh = cone_oracle(lam, na, a, "re") + 1j * cone_oracle(lam, na, a, "im")
        assert abs(amp.amplitude[i] - r_coh) < 1e-9
        assert abs(amp.signal[i] - abs(r_coh) ** 2) < 1e-9
    # the crossed (depolarized) power through a filled cone is physical and non-zero
    assert np.all(out["crossed"].signal > 1e-5)
    # sensitivity: the solid-angle weight is a different (wrong) pupil
    wrong = [cone_oracle(lam, na, a, "power", weight=lambda th: math.sin(th)) for lam in LAM]
    assert np.min(np.abs(out[None].signal - np.asarray(wrong))) > 1e-4


def test_na_zero_returns_the_plane_wave_exactly_and_converges_as_na_squared():
    """NA = 0 is one normal-incidence ray with weight 1: equal to r_s(0) and |r_s(0)|^2 to rounding, for any
    input polarization and both detection models. Rounding bound: the module forms E_par = a_s (a_s r_s) -
    a_p (a_p r_p) with r_p = -r_s exactly (this oracle) and a_s, a_p = sin, cos of the polarization angle,
    i.e. (a_s^2 + a_p^2) r_s. sin and cos carry <= 1 ulp each, so a_s^2 + a_p^2 = 1 within 2 eps; the four
    real x complex products and the sum add <= 2.5 eps of |r|: |amplitude - r_s| <= 4.5 eps |r_s|, and the
    squared modulus doubles that plus its own rounding, <= 10.5 eps |r_s|^2. Bars 8 eps |r| and 16 eps
    |r|^2 (fused multiply-add only removes roundings). MEASURED worst 0.67 eps |r| and 1.03 eps |r|^2 (at
    45 deg; 0 at 0 / 90 deg, where one of a_s, a_p is exactly 1). For small NA the cone correction is
    O(NA^2): the deviation ratio between NA 0.04 and 0.02 is 4 to within 1 % at 633 and 700 nm (measured
    4.0037, 3.99998). At 600 nm this film's NA^2 coefficient happens to vanish and the ratio is 16.0 (an
    O(NA^4) correction) -- checked too, since it confirms the scaling rather than contradicting it."""
    pw = np.array([airy(lam, 0.0, "s") for lam in LAM])
    eps = np.finfo(float).eps
    tol_amp = 8.0 * eps * np.abs(pw)
    tol_pow = 16.0 * eps * np.abs(pw) ** 2
    for alpha in (0.0, 45.0, 90.0):
        res = measure(film_model, LAM, pupil=pupil_quadrature(0.0, polarization_deg=alpha))
        assert np.all(np.abs(res.signal - np.abs(pw) ** 2) <= tol_pow), alpha
        assert np.all(np.abs(res.amplitude - pw) <= tol_amp), alpha
        amp = measure(film_model, LAM, pupil=pupil_quadrature(0.0, polarization_deg=alpha),
                      detection="amplitude")
        assert np.all(np.abs(amp.signal - np.abs(pw) ** 2) <= tol_pow), alpha
    res = measure(film_model, LAM)                                     # pupil None == NA 0
    assert np.all(np.abs(res.signal - np.abs(pw) ** 2) <= tol_pow)
    dev = [measure(film_model, LAM, pupil=pupil_quadrature(na)).signal - np.abs(pw) ** 2
           for na in (0.02, 0.04)]
    ratio = dev[1] / dev[0]
    assert np.all(np.abs(ratio[1:] - 4.0) < 0.04), ratio
    assert abs(ratio[0] - 16.0) < 0.2, ratio


def test_gaussian_beam_weights_fixed_pupil_and_fixed_waist():
    """Gaussian pupil (power per d^2 s ~ exp(-2 s^2/s0^2)) truncated at NA 0.6, against dblquad
    with weight exp(-2 sin^2/s0^2) cos sin. Fixed waist: s0 = lambda/(pi w0) differs per
    wavelength, so the weights are per-wavelength rows. Measured 5.4e-11 (fixed s0) and 3.0e-9
    (fixed waist, whose narrower s0 at short wavelength is harder for 8 radial nodes); 1e-6."""
    na, s0 = 0.6, 0.3
    res = measure(film_model, LAM, pupil=pupil_quadrature(na, profile="gaussian", na_e2=s0,
                                                          n_radial=8))
    w0 = 0.8e-6
    resw = measure(film_model, LAM, pupil=pupil_quadrature(na, profile="gaussian", waist_m=w0,
                                                           n_radial=8))
    for i, lam in enumerate(LAM):
        g = cone_oracle(lam, na, 0.0, "power",
                        weight=lambda th: math.exp(-2 * math.sin(th) ** 2 / s0 ** 2) * math.cos(th) * math.sin(th))
        assert abs(res.signal[i] - g) < 1e-6
        sw = lam / (math.pi * w0)
        gw = cone_oracle(lam, na, 0.0, "power",
                         weight=lambda th: math.exp(-2 * math.sin(th) ** 2 / sw ** 2) * math.cos(th) * math.sin(th))
        assert abs(resw.signal[i] - gw) < 1e-6
    W = pupil_quadrature(na, profile="gaussian", waist_m=w0).weights(LAM)
    # s0 = lambda / (pi w0) changes by 17 % from 600 to 700 nm: the weight rows differ by 5.2e-3 (MEASURED max)
    assert W.shape[0] == LAM.size and np.max(np.abs(W[0] - W[-1])) > 1e-3
    assert np.allclose(W.sum(axis=1), 1.0, rtol=1e-14, atol=0.0)          # normalized rows: round-off


def test_annular_pupil_and_dense_incidence_medium():
    """Central obscuration (NA 0.2..0.65) in an n = 1.5 immersion medium vs dblquad."""
    n0 = 1.5

    def r_fn(lam, th, pol):
        return airy(lam, th, pol, n0=n0)

    def model(wl, theta_deg=0.0, phi_deg=0.0, pol="s"):
        return np.array([r_fn(lam, math.radians(theta_deg), pol) for lam in wl])

    res = measure(model, LAM, pupil=pupil_quadrature(0.65, na_inner=0.2, n_medium=n0, n_radial=6))
    for i, lam in enumerate(LAM):
        assert abs(res.signal[i] - cone_oracle(lam, 0.65, 0.0, "power", na_inner=0.2, n=n0,
                                               r_fn=r_fn)) < 1e-6


def test_in_plane_marginal_is_exact_for_kx_only_responses():
    """Consistency of the reduction: for a response that depends on k_x alone the in-plane 1-D
    rule must equal the full 2-D cone (uniform, gaussian, annulus, callable profiles). Measured
    max difference 8.0e-7 (the 24 x 64 conical rule's own error on the e^{7i s_x} phase); 2e-6."""
    def kx_model(wl, theta_deg=0.0, phi_deg=0.0, pol="s"):
        sx = math.sin(math.radians(theta_deg)) * math.cos(math.radians(phi_deg))
        r = np.exp(1j * 7.0 * sx) * (0.6 + 0.3 * sx * sx) * np.ones(len(wl))
        return r if pol == "s" else -r
    cases = [dict(), dict(profile="gaussian", na_e2=0.35), dict(na_inner=0.25),
             dict(profile=lambda s: 1.0 + 2.0 * s * s)]
    for kw in cases:
        full = measure(kx_model, LAM[:1], pupil=pupil_quadrature(0.6, n_radial=24, n_azimuth=64, **kw))
        inp = measure(kx_model, LAM[:1], pupil=pupil_quadrature(0.6, mode="in_plane", n_kx=24, **kw))
        assert abs(full.amplitude[0] - inp.amplitude[0]) < 2e-6, kw
        assert abs(full.signal[0] - inp.signal[0]) < 2e-6, kw


def test_in_plane_approximation_is_labelled_and_second_order():
    """The same-k_x approximation on the thin film (x input = TM): labelled in notes; its error
    against the exact conical average scales as NA^2 (ratio 0.2 -> 0.1 within 4 +- 0.4) and is
    stated here for scale: at NA 0.5 it is 5.8e-3 in reflectance on R = 0.043 (13 % relative; the
    film's TE and TM responses differ strongly off normal, which is the worst case for dropping
    the s part of each ray)."""
    errs = []
    for na in (0.1, 0.2, 0.5):
        ex = measure(film_model, LAM[:1], pupil=pupil_quadrature(na))
        ap = measure(film_model, LAM[:1], pupil=pupil_quadrature(na, mode="in_plane",
                                                                fold_symmetric=True))
        assert any("in-plane-only approximation" in s for s in ap.notes)
        errs.append(abs(ap.signal[0] - ex.signal[0]))
    assert 3.6 < errs[1] / errs[0] < 4.4, errs
    assert 4e-3 < errs[2] < 8e-3, errs


def test_detection_models_differ_under_phase_dispersion():
    """A perfect mirror displaced by d below the reference plane: r_s = -exp(i psi), r_p =
    +exp(i psi), psi = 2 k0 d sqrt(n^2 - s^2). Power detection must give exactly 1 at every NA
    (lossless, Parseval). The coherent mode-matched amplitude is the pupil average of exp(i psi):
    closed form (2/NA^2) [exp(i a c)(c/(i a) + 1/a^2)] from c1 = sqrt(1 - NA^2) to 1, a = 2 k0 d.
    Its power |r_coh|^2 < 1 (dephasing, not loss). Measured at NA 0.5, d = 1 um: power 1 to 1e-16,
    amplitude error 1.8e-14, |r_coh|^2 = 0.49 / 0.53 / 0.60 at 600 / 633 / 700 nm."""
    d = 1.0e-6

    def mirror(wl, theta_deg=0.0, phi_deg=0.0, pol="s"):
        c = math.cos(math.radians(theta_deg))
        e = np.exp(2j * 2 * np.pi / np.asarray(wl) * d * c)
        return -e if pol == "s" else e

    na = 0.5
    p = pupil_quadrature(na, n_radial=12, n_azimuth=4)
    pw = measure(mirror, LAM, pupil=p)
    am = measure(mirror, LAM, pupil=p, detection="amplitude")
    assert np.max(np.abs(pw.signal - 1.0)) < 1e-12
    for i, lam in enumerate(LAM):
        a = 2 * 2 * math.pi / lam * d
        c1 = math.sqrt(1 - na * na)

        def F(c):
            return cmath.exp(1j * a * c) * (c / (1j * a) + 1 / a ** 2)
        avg = 2.0 / na ** 2 * (F(1.0) - F(c1))
        assert abs(am.amplitude[i] - (-avg)) < 1e-9
    assert np.all(am.signal < 0.9)


def test_conical_notes_and_cross_terms_enter_when_supplied():
    """A model returning (r_co, r_cross) has its conversion terms used (the power changes by the
    cross power); one returning co-pol only is flagged in notes. Consistency check of the bookkeeping
    against the written-out per-ray formula at one off-axis ray."""
    rc = 0.1 + 0.05j

    def conv(wl, theta_deg=0.0, phi_deg=0.0, pol="s"):
        r = np.full(len(wl), 0.5 + 0.2j)
        return (r if pol == "s" else -r * 0.9, np.full(len(wl), rc if pol == "s" else -rc))

    def co(wl, theta_deg=0.0, phi_deg=0.0, pol="s"):
        return conv(wl, theta_deg, phi_deg, pol)[0]

    p = plane_wave_incidence(20.0, 30.0, polarization_deg=10.0)
    a_s, a_p = math.sin(math.radians(-20.0)), math.cos(math.radians(-20.0))
    rss, rpp, rps, rsp = 0.5 + 0.2j, -(0.5 + 0.2j) * 0.9, rc, -rc
    es, ep = a_s * rss + a_p * rsp, a_p * rpp + a_s * rps
    res = measure(conv, LAM[:1], pupil=p)
    assert abs(res.signal[0] - (abs(es) ** 2 + abs(ep) ** 2)) < 1e-14
    cone = measure(co, LAM[:1], pupil=pupil_quadrature(0.3))
    assert any("cross-polarization" in s for s in cone.notes)
    cone2 = measure(conv, LAM[:1], pupil=pupil_quadrature(0.3))
    assert not any("cross-polarization" in s for s in cone2.notes)


def test_period_warning_when_diffracted_orders_enter_the_pupil():
    """lambda / period <= 2 NA: an UnreliableResultWarning (the specular model omits collected
    diffracted orders); none when the period is fine enough."""
    p = pupil_quadrature(0.5, mode="in_plane")
    with pytest.warns(UnreliableResultWarning, match="non-specular diffraction order"):
        measure(film_model, LAM, pupil=p, period_m=650e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        measure(film_model, LAM, pupil=p, period_m=500e-9)


# ------------------------------------------------------------------------------------------------
# item 2: fabrication scatter
# ------------------------------------------------------------------------------------------------
# amplitude resonance r = 1 - i kappa / (Delta + i g), Delta = w - w0(p), w0 linear in p:
# |r|^2 = 1 - (2 g kappa - kappa^2) / (Delta^2 + g^2)  (a Lorentzian dip, HWHM g)
G, KAPPA, W0, CW, P0, SIGP = 1.0, 0.8, 0.0, 2.0, 100.0, 0.6
WGRID = np.linspace(-6.0, 6.0, 61)


def lorentz_r(w, p):
    return 1.0 - 1j * KAPPA / (w - (W0 + CW * (p - P0)) + 1j * G)


def voigt_closed_forms(w):
    """<r> and <|r|^2> over p ~ N(P0, SIGP^2) via the Faddeeva function: the centre is
    N(W0, s^2) with s = CW SIGP; z = (w - W0 + i g)/(s sqrt 2);
    <1/(w - w0 + i g)> = -i sqrt(pi) w(z) / (s sqrt 2); <1/((w-w0)^2 + g^2)> = sqrt(pi) Re w(z)/(g s sqrt 2)."""
    s = CW * SIGP
    z = (w - W0 + 1j * G) / (s * math.sqrt(2.0))
    fz = wofz(z)
    mean_r = 1.0 - 1j * KAPPA * (-1j * math.sqrt(math.pi) * fz / (s * math.sqrt(2.0)))
    mean_inv = math.sqrt(math.pi) * fz.real / (G * s * math.sqrt(2.0))
    mean_R = 1.0 - (2 * G * KAPPA - KAPPA ** 2) * mean_inv
    return mean_r, mean_R


def test_scatter_average_of_lorentzian_gives_the_voigt_closed_form():
    """Gauss-Hermite (200 nodes) patch average vs the Faddeeva closed forms: <|r|^2> is the Voigt
    dip, <r> its coherent partner. Error budget: GH on a Lorentzian whose HWHM g is 0.83 of the
    centre spread converges only root-exponentially (pole near the real axis); 200 nodes measure
    1.3e-10 (<r>) and 1.6e-10 (<|r|^2>); tolerance 1e-8. Plus the
    identities |<r>|^2 <= <|r|^2> and diffuse = the weighted variance sum w |r - <r>|^2 (computed
    here independently)."""
    q = parameter_quadrature("p", "normal", mean=P0, std=SIGP, n_nodes=200)
    r = np.array([lorentz_r(WGRID, q.nodes[i, 0]) for i in range(q.n_nodes)])
    pa = patch_average(r, q.weights)
    mean_r, mean_R = voigt_closed_forms(WGRID)
    assert np.max(np.abs(pa.mean - mean_r)) < 1e-8
    assert np.max(np.abs(pa.incoherent - mean_R)) < 1e-8
    assert np.all(pa.coherent <= pa.incoherent + 1e-15)
    var = np.sum(q.weights[:, None] * np.abs(r - pa.mean[None, :]) ** 2, axis=0)
    assert np.max(np.abs(pa.diffuse - var)) < 1e-12
    # the composer gives the same numbers through a scalar model at normal incidence
    model = scalar_model(lambda wl, theta_deg=0.0, phi_deg=0.0, p=P0: lorentz_r(np.asarray(wl), p))
    spec = measure(model, WGRID, parameters=q, scatter_collection="specular")
    allc = measure(model, WGRID, parameters=q, scatter_collection="all")
    assert np.max(np.abs(spec.signal - np.abs(mean_r) ** 2)) < 1e-8
    assert np.max(np.abs(allc.signal - mean_R)) < 1e-8
    assert np.max(np.abs(spec.diffuse - (mean_R - np.abs(mean_r) ** 2))) < 1e-8


def test_parameter_quadrature_convergence():
    """Gauss-Hermite error against the closed form falls monotonically, root-exponentially for
    this pole-near-axis case (measured 1.3e-2, 7.0e-5, 9.0e-7, 1.9e-9 at n = 10, 40, 80, 160 --
    the numbers quoted in parameter_quadrature's docstring), and Gauss-Legendre on a
    uniform spread converges to the arctan closed form. The convergence re-run warns when the
    rule is coarse and stays silent when converged."""
    _, mean_R = voigt_closed_forms(WGRID)
    errs = []
    for n in (10, 40, 80, 160):
        q = parameter_quadrature("p", "normal", mean=P0, std=SIGP, n_nodes=n)
        r = np.array([lorentz_r(WGRID, q.nodes[i, 0]) for i in range(n)])
        errs.append(float(np.max(np.abs(patch_average(r, q.weights).incoherent - mean_R))))
    assert errs[0] > errs[1] > errs[2] > errs[3] and errs[3] < 1e-8 and errs[1] < 1e-4, errs
    h = 1.5                                                       # uniform half width of p
    q = parameter_quadrature("p", "uniform", mean=P0, half_width=h, n_nodes=40)   # 2e-9 at 30, 3e-12 at 40
    r = np.array([lorentz_r(WGRID, q.nodes[i, 0]) for i in range(q.n_nodes)])
    lo, hi = WGRID - W0 - CW * h, WGRID - W0 + CW * h             # Delta range
    closed = 1.0 - (2 * G * KAPPA - KAPPA ** 2) / G * (np.arctan(hi / G) - np.arctan(lo / G)) / (2 * CW * h)
    assert np.max(np.abs(patch_average(r, q.weights).incoherent - closed)) < 1e-9
    model = scalar_model(lambda wl, theta_deg=0.0, phi_deg=0.0, p=P0: lorentz_r(np.asarray(wl), p))
    q5 = parameter_quadrature("p", "normal", mean=P0, std=SIGP, n_nodes=5)
    with pytest.warns(UnreliableResultWarning, match="not converged"):
        measure(model, WGRID, parameters=q5, scatter_collection="all", convergence_nodes=9)
    q120 = parameter_quadrature("p", "normal", mean=P0, std=SIGP, n_nodes=120)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = measure(model, WGRID, parameters=q120, scatter_collection="all", convergence_nodes=160)
    assert res.convergence_delta < 1e-6


def test_combined_parameters_tensor_and_sparse_exactness():
    """Normal x uniform x normal moments: the tensor rule (3 nodes each) is exact for per-variable
    degree <= 5; the Smolyak rule at level L is exact for total degree <= 2 (L - d) + 1 (= 5 at
    d = 3, L = 5), checked on every monomial up to that degree against the analytic moments, and
    it is NOT exact at total degree 8 (the claim is not vacuous)."""
    qa = parameter_quadrature("a", "normal", mean=1.0, std=0.5)
    qb = parameter_quadrature("b", "uniform", mean=-2.0, half_width=0.7)
    qc = parameter_quadrature("c", "normal", mean=0.0, std=2.0)

    def mom_normal(m, s, k):
        return sum(math.comb(k, j) * m ** (k - j) * s ** j * (0 if j % 2 else math.prod(range(j - 1, 0, -2)))
                   for j in range(k + 1))

    def mom_unif(m, h, k):
        return ((m + h) ** (k + 1) - (m - h) ** (k + 1)) / ((k + 1) * 2 * h)

    def exact(e):
        return mom_normal(1.0, 0.5, e[0]) * mom_unif(-2.0, 0.7, e[1]) * mom_normal(0.0, 2.0, e[2])

    def rule(q, e):
        return float(np.sum(q.weights * np.prod(q.nodes ** np.asarray(e), axis=1)))

    tens = combine_parameters(qa.refine(3), qb.refine(3), qc.refine(3))
    assert tens.n_nodes == 27
    sp = combine_parameters(qa, qb, qc, method="sparse", level=5)
    assert sp.n_nodes == 25
    for e in [(i, j, k) for i in range(6) for j in range(6) for k in range(6)]:
        if max(e) <= 5:
            assert abs(rule(tens, e) - exact(e)) < 1e-9 * max(1.0, abs(exact(e)))
        if sum(e) <= 5:
            assert abs(rule(sp, e) - exact(e)) < 1e-9 * max(1.0, abs(exact(e))), e
    assert abs(rule(sp, (0, 0, 8)) - exact((0, 0, 8))) > 1.0


# ------------------------------------------------------------------------------------------------
# item 3: instrument and normalization
# ------------------------------------------------------------------------------------------------
def _nonuniform_grid(lo, hi, n, centre):
    u = np.linspace(-1.0, 1.0, n)
    x = centre + np.sign(u) * np.abs(u) ** 1.7 * max(hi - centre, centre - lo)
    return np.unique(np.clip(x, lo, hi))


def test_gaussian_instrument_on_lorentzian_gives_voigt():
    """A Lorentzian line (HWHM 2 nm) on a non-uniform grid convolved with a Gaussian of FWHM 3 nm
    equals the closed-form Voigt (wofz) to 2e-5 of the peak away from the grid ends (piecewise-
    linear interpolation error of the grid, measured 3.6e-6; within ~6 FWHM of an end the
    truncated Lorentzian tail is genuinely missing from the data). The integral is conserved with
    edge='zero' when the line and kernel lie inside the grid: a Gaussian line under each kernel,
    trapz of output vs input to 2e-5 (measured 5.7e-6: the trapz discretization of the sampled
    output; the convolution itself conserves the interpolant's integral exactly)."""
    c, g = 1000e-9, 2e-9
    lam = _nonuniform_grid(900e-9, 1100e-9, 3001, c)
    L = g / math.pi / ((lam - c) ** 2 + g ** 2)
    fw = 3e-9
    sig = fw / (2 * math.sqrt(2 * math.log(2)))
    out = instrument_convolve(lam, L, shape="gaussian", fwhm_m=fw, edge="zero")
    voigt = wofz(((lam - c) + 1j * g) / (sig * math.sqrt(2))).real / (sig * math.sqrt(2 * math.pi))
    inner = np.abs(lam - c) < 80e-9
    assert np.max(np.abs(out - voigt)[inner]) < 2e-5 * L.max()
    line = np.exp(-0.5 * ((lam - c) / 2e-9) ** 2)
    for shape in ("gaussian", "triangle", "rectangle"):
        conv = instrument_convolve(lam, line, shape=shape, fwhm_m=fw, edge="zero")
        assert abs(trapz(conv, lam) / trapz(line, lam) - 1.0) < 2e-5, shape


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian", "triangle", "rectangle"])
def test_instrument_shapes_against_brute_force(shape):
    """Every shape against an independent adaptive-quadrature convolution of the piecewise-linear
    interpolant at a few output points (1e-7 of the scale), constants preserved by edge
    'renormalize' (1e-12), and a wavenumber-axis convolution equal to the wavelength-axis one
    performed on 1/lambda by hand."""
    lam = _nonuniform_grid(1400e-9, 1600e-9, 401, 1500e-9)
    S = 0.5 + 0.4 * np.exp(-((lam - 1500e-9) / 6e-9) ** 2) * np.cos((lam - 1500e-9) / 3e-9)
    fw = 4e-9
    out = instrument_convolve(lam, S, shape=shape, fwhm_m=fw, edge="zero")

    def K(t):
        if shape == "gaussian":
            s = fw / (2 * math.sqrt(2 * math.log(2)))
            return math.exp(-0.5 * (t / s) ** 2) / (s * math.sqrt(2 * math.pi))
        if shape == "lorentzian":
            return (fw / 2) / math.pi / (t * t + (fw / 2) ** 2)
        if shape == "triangle":
            return max(fw - abs(t), 0.0) / fw ** 2
        return (1.0 / fw) if abs(t) < fw / 2 else 0.0

    for i in (50, 200, 333):
        x0 = lam[i]
        brute = sum(integrate.quad(lambda xp: K(x0 - xp) * np.interp(xp, lam, S), lam[j], lam[j + 1],
                                   epsabs=1e-13, limit=200)[0] for j in range(lam.size - 1)
                    if shape == "lorentzian" or abs(lam[j] - x0) < 8 * fw)
        assert abs(out[i] - brute) < 1e-7, (shape, out[i], brute)
    ones = instrument_convolve(lam, np.ones_like(lam), shape=shape, fwhm_m=fw)
    assert np.max(np.abs(ones - 1.0)) < 1e-12
    nu = 0.01 / lam
    fw_cm = 25.0
    a = instrument_convolve(lam, S, shape=shape, fwhm_per_cm=fw_cm)
    o = np.argsort(nu)
    b = np.empty_like(a)
    b[o] = instrument_convolve(nu[o] * 1e-9, S[o], shape=shape, fwhm_m=fw_cm * 1e-9)
    assert np.max(np.abs(a - b)) < 1e-10


def test_source_weighting_and_reference_ratio_order():
    """With a non-flat lamp the spectrometer reports conv(R I0)/conv(I0), not conv(R); and the
    reflectance is the ratio of the two CONVOLVED signals. A sharp lamp line next to a sample dip
    and a structured reference mirror (a 0.5-deep feature) make the wrong orders differ by > 1e-2
    (measured 0.064 for ignoring the lamp and 0.034 for convolving the ratio)."""
    lam = np.linspace(990e-9, 1010e-9, 801)
    R = 0.9 - 0.6 / (1 + ((lam - 1000e-9) / 0.6e-9) ** 2)
    I0 = 1.0 + 5.0 * np.exp(-((lam - 1001e-9) / 0.3e-9) ** 2)
    fw = 2e-9
    rep = instrument_convolve(lam, R, fwhm_m=fw, source=I0)
    manual = instrument_convolve(lam, R * I0, fwhm_m=fw) / instrument_convolve(lam, I0, fwhm_m=fw)
    assert np.max(np.abs(rep - manual)) < 1e-12
    assert np.max(np.abs(rep - instrument_convolve(lam, R, fwhm_m=fw))) > 1e-2
    Rm = 0.95 - 0.5 * np.exp(-((lam - 1000.5e-9) / 0.4e-9) ** 2)  # a structured reference mirror
    right = normalize_to_reference(instrument_convolve(lam, R, fwhm_m=fw),
                                   instrument_convolve(lam, Rm, fwhm_m=fw), 1.0)
    wrong = instrument_convolve(lam, R / Rm, fwhm_m=fw)
    assert np.max(np.abs(right - wrong)) > 1e-2
    with pytest.raises(TypeError, match="DETECTED POWER"):
        instrument_convolve(lam, R + 0j, fwhm_m=fw)


# ------------------------------------------------------------------------------------------------
# item 5: composition, cost, parallel map
# ------------------------------------------------------------------------------------------------
class _Counting:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self.fn(*a, **k)


def test_composition_order_and_reference_normalization():
    """The composer = (fields averaged over the parameter nodes at each wavelength) -> detection
    (|<E>|^2) -> instrument -> ratio to the reference detected under the same cone, reproduced
    here step by step from patch_average and instrument_convolve (consistency check), and
    different from the two wrong orders: squaring before averaging (the incoherent signal) and
    convolving the amplitude before squaring (both > 1e-2 here)."""
    q = parameter_quadrature("p", "normal", mean=P0, std=SIGP, n_nodes=20)
    model = scalar_model(lambda wl, theta_deg=0.0, phi_deg=0.0, p=P0: lorentz_r(np.asarray(wl), p))
    mirror = scalar_model(lambda wl, theta_deg=0.0, phi_deg=0.0: 0.95 * np.exp(0.1j * np.asarray(wl)))
    inst = InstrumentFunction("gaussian", fwhm_m=1.5)
    res = measure(model, WGRID, parameters=q, scatter_collection="specular", instrument=inst,
                  reference=mirror)
    r = np.array([lorentz_r(WGRID, q.nodes[i, 0]) for i in range(q.n_nodes)])
    pa = patch_average(r, q.weights)
    ref = np.abs(0.95 * np.exp(0.1j * WGRID)) ** 2
    manual = (instrument_convolve(WGRID, pa.coherent, fwhm_m=1.5)
              / instrument_convolve(WGRID, ref, fwhm_m=1.5) * ref)
    assert np.max(np.abs(res.signal - manual)) < 1e-12
    assert np.max(np.abs(res.detected - pa.coherent)) < 1e-12
    wrong1 = instrument_convolve(WGRID, pa.incoherent, fwhm_m=1.5)
    re_ = instrument_convolve(WGRID, pa.mean.real, fwhm_m=1.5)
    im_ = instrument_convolve(WGRID, pa.mean.imag, fwhm_m=1.5)
    wrong2 = re_ ** 2 + im_ ** 2
    assert np.max(np.abs(res.signal - wrong1)) > 1e-2
    assert np.max(np.abs(res.signal - wrong2)) > 1e-2
    # reference + convergence re-run together (the plane-wave reference must not inherit the
    # sample's parameter settings), and the amplitude normalized by the reference amplitude
    # times its plane-wave amplitude (here the same normal-incidence ray, so it returns <r>)
    res2 = measure(model, WGRID, parameters=q, scatter_collection="specular", reference=mirror,
                   convergence_nodes=24, convergence_tol=1.0)
    assert res2.plan.n_reference_calls == 2 and res2.convergence_delta is not None
    assert np.max(np.abs(res2.amplitude - pa.mean)) < 1e-12


def test_numeric_reference_reflectance_keeps_the_amplitude_phase_or_refuses():
    """A power reflectance R_ref fixes |r_ref| but not its phase. Reference mirror r = -0.98 (metal-like),
    sample r = 0.3 + 0.4i, normal incidence: the normalized amplitude amplitude / r_ref,model x r_ref is
      * 0.3 + 0.4i with 'plane_wave' (r_ref from the reference model) and with a numeric R_ref = 0.98^2 plus
        reference_amplitude = -0.98 (a few ulp: one complex division and product, bar 1e-15 on |r| = 0.5);
      * refused (ValueError) for detection='amplitude' with a numeric R_ref alone, and NaN for power
        detection, whose power signal is unaffected (0.25 = |r|^2 R_ref / R_ref,detected, a few ulp);
      * the raw ratio (None): the phase relative to the reference, -(0.3 + 0.4i)/0.98.
    Taking sqrt(R_ref) = +0.98 as the reference amplitude would return -0.3 - 0.4i (a pi error) with no
    warning."""
    mirror = scalar_model(lambda wl, theta_deg=0.0, phi_deg=0.0: np.full(np.size(wl), -0.98 + 0j))
    samp = scalar_model(lambda wl, theta_deg=0.0, phi_deg=0.0: np.full(np.size(wl), 0.3 + 0.4j))
    lam = [800e-9, 900e-9]
    want = 0.3 + 0.4j
    pw = MeasurementSetup(reference=mirror, detection="amplitude").run(samp, lam)
    assert np.max(np.abs(pw.amplitude - want)) < 1e-15
    num = MeasurementSetup(reference=mirror, reference_reflectance=0.98 ** 2, reference_amplitude=-0.98,
                           detection="amplitude").run(samp, lam)
    assert np.max(np.abs(num.amplitude - want)) < 1e-15
    assert np.max(np.abs(num.signal - abs(want) ** 2)) < 1e-15
    raw = MeasurementSetup(reference=mirror, reference_reflectance=None, detection="amplitude").run(samp, lam)
    assert np.max(np.abs(raw.amplitude - want / -0.98)) < 1e-15
    with pytest.raises(ValueError, match="phase"):
        MeasurementSetup(reference=mirror, reference_reflectance=0.98 ** 2, detection="amplitude")
    pwr = MeasurementSetup(reference=mirror, reference_reflectance=0.98 ** 2).run(samp, lam)
    assert np.all(np.isnan(pwr.amplitude)) and np.max(np.abs(pwr.signal - 0.25)) < 1e-15
    with pytest.raises(ValueError, match="abs\\(reference_amplitude\\)"):
        MeasurementSetup(reference=mirror, reference_reflectance=0.9, reference_amplitude=-0.98)
    with pytest.raises(ValueError, match="numeric"):
        MeasurementSetup(reference=mirror, reference_amplitude=-0.98)
    with pytest.raises(ValueError, match="POWER"):
        MeasurementSetup(reference=mirror, reference_reflectance=-0.98 + 0j)


def test_plan_counts_parallel_map_and_refusals():
    """plan() predicts the exact number of model calls (counted), n_workers=2 and a user map_fn
    give bit-identical results to the serial run, fold_symmetric / azimuthal_invariant cut the
    calls as stated, max_solves refuses before solving, and scatter without a collection choice
    is refused."""
    q = parameter_quadrature("p", "normal", mean=P0, std=SIGP, n_nodes=5)

    def fn(wl, theta_deg=0.0, phi_deg=0.0, pol="s", p=P0):
        r = lorentz_r(np.asarray(wl) + 0.01 * theta_deg + 0.003 * phi_deg, p)
        return r if pol == "s" else -0.97 * r

    pup = pupil_quadrature(0.3, n_radial=3, n_azimuth=8, polarization_deg=20.0)
    setup = MeasurementSetup(pupil=pup, parameters=q, scatter_collection="all")
    plan = setup.plan(WGRID)
    assert plan.n_model_calls == 3 * 8 * 2 * 5 and plan.n_solves == plan.n_model_calls * WGRID.size
    cm = _Counting(fn)
    serial = setup.run(cm, WGRID)
    assert cm.calls == plan.n_model_calls
    thr = setup.run(fn, WGRID, n_workers=2)
    mapped = setup.run(fn, WGRID, map_fn=map)
    assert np.array_equal(serial.signal, thr.signal) and np.array_equal(serial.signal, mapped.signal)
    assert serial.plan.n_solves == plan.n_solves
    with pytest.raises(ValueError, match="max_solves"):
        setup.run(fn, WGRID, max_solves=100)
    assert measure(fn, WGRID, dry_run=True, pupil=pup).n_model_calls == 3 * 8 * 2
    fold = pupil_quadrature(0.3, mode="in_plane", n_kx=9, fold_symmetric=True)
    assert MeasurementSetup(pupil=fold).plan(WGRID).n_model_calls == 5          # |theta'| distinct, p only
    inv = pupil_quadrature(0.3, n_radial=3, n_azimuth=8, azimuthal_invariant=True)
    assert MeasurementSetup(pupil=inv).plan(WGRID).n_model_calls == 3 * 2
    a = measure(film_model, LAM, pupil=inv)
    b = measure(film_model, LAM, pupil=pupil_quadrature(0.3, n_radial=3, n_azimuth=8))
    assert np.max(np.abs(a.signal - b.signal)) < 1e-14
    with pytest.raises(ValueError, match="scatter_collection"):
        MeasurementSetup(parameters=q)
    assert ParameterQuadrature().n_nodes == 1


def test_design_adapter_through_the_layered_tmm_solver():
    """design_plane_wave_model with the run_pipeline TMM seam on a planar Design: r per (theta,
    pol) equals the Airy oracle of the same film, and a design_for rebuild reaches the solver
    (thickness scatter changes the result as the oracle predicts)."""
    pytest.importorskip("tmm")
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.optics.tmm_reference import make_layered_tmm_solver

    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("film", ConstantOptical(4.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(2.25 + 0j)))

    def make(d_m=300e-9):
        return Design(name="film", unit_cell=UnitCell(period_x_m=400e-9, period_y_m=400e-9),
                      stack=Stack(layers=[Layer("film", d_m, "film")], superstrate_material="air",
                                  substrate_material="glass"),
                      electrodes=[], materials=reg, optical=OpticalSpec(polarization="x"))

    model = design_plane_wave_model(make(), make_layered_tmm_solver(), design_for=make)
    for th in (0.0, 25.0):
        for pol in ("s", "p"):
            got = model(LAM, theta_deg=th, pol=pol)
            ref = np.array([airy(lam, math.radians(th), pol) for lam in LAM])
            assert np.max(np.abs(got - ref)) < 1e-12
    got = model(LAM, theta_deg=10.0, pol="p", d_m=320e-9)
    ref = np.array([airy(lam, math.radians(10.0), "p", d=320e-9) for lam in LAM])
    assert np.max(np.abs(got - ref)) < 1e-12


# ------------------------------------------------------------------------------------------------
# item 6: fitting
# ------------------------------------------------------------------------------------------------
def _voigt_dip(lam, centre, gamma, sigma, depth, base):
    z = ((lam - centre) + 1j * gamma) / (sigma * math.sqrt(2.0))
    v = wofz(z).real / (sigma * math.sqrt(2.0 * math.pi))
    v0 = wofz(1j * gamma / (sigma * math.sqrt(2.0))).real / (sigma * math.sqrt(2.0 * math.pi))
    return base - depth * v / v0


def test_fit_recovers_intrinsic_damping_and_gaussian_broadening():
    """Synthetic Voigt dip (intrinsic Lorentzian HWHM 4 nm + Gaussian broadening sigma 6 nm,
    noise 2e-3): every parameter is recovered within 3 reported standard errors. The two widths
    are strongly correlated (measured -0.963 -- a Gaussian and a Lorentzian width trade off on one
    dip) yet both are resolved to ~2 %, so the verdict is "correlated but resolved", not
    "degenerate"; the null test gamma = 0 (no intrinsic damping, all broadening Gaussian) is
    excluded by a large delta chi2, and the profile of gamma has its minimum at the fit."""
    rng = np.random.default_rng(7)
    lam = np.linspace(1250.0, 1350.0, 201)
    true = dict(centre=1300.0, gamma=4.0, sigma=6.0, depth=0.5, base=0.9)
    y = _voigt_dip(lam, **true) + rng.normal(0.0, 2e-3, lam.size)
    p0 = dict(centre=1298.0, gamma=6.0, sigma=4.0, depth=0.4, base=0.85)
    fit = fit_broadening(lam, y, _voigt_dip, p0, sigma=2e-3,
                         bounds={"gamma": (0.0, 50.0), "sigma": (0.1, 50.0)}, profile=("gamma",),
                         null_values={"gamma": 0.0})
    for k, v in true.items():
        assert abs(fit.params[k] - v) < 3 * fit.stderr[k], (k, fit.params[k], fit.stderr[k])
    assert not fit.degenerate and not fit.null_combinations and not fit.undetermined
    assert "distinguished" in fit.message
    assert [(a, b) for a, b, _ in fit.correlated] == [("gamma", "sigma")]
    assert fit.correlated[0][2] < -0.9
    assert fit.null_tests["gamma"] > 100.0 and "excluded" in fit.message
    vals, dchi = fit.profiles["gamma"]
    assert np.argmin(dchi) == len(vals) // 2 and dchi.max() > 4.0
    assert 0.5 < fit.reduced_chi2 < 1.5


def test_fit_flags_exact_and_data_limited_degeneracy():
    """(1) Lorentzian instrument on a Lorentzian line: the measured width is exactly gamma +
    gamma_inst, so only the sum is measurable -- reported as an exactly degenerate combination.
    (2) Voigt with sigma >> gamma (truth gamma 0.5, sigma 8) and noisy, sparse data: the fit lands
    on an almost Lorentzian explanation (measured gamma 7.9 with sigma unresolved) -- the two widths
    are degenerate FOR THESE DATA (|corr| >= 0.95, one unresolved), and the null test gamma = 0
    passes (measured delta chi2 1.8 < 4): the line does not require intrinsic damping."""
    lam = np.linspace(1250.0, 1350.0, 121)

    def lor(lam, centre, gamma, gamma_inst, depth, base):
        g = gamma + gamma_inst
        return base - depth / (1 + ((lam - centre) / g) ** 2)

    y = lor(lam, 1300.0, 3.0, 5.0, 0.5, 0.9)
    fit = fit_broadening(lam, y + np.random.default_rng(1).normal(0, 1e-3, lam.size), lor,
                         dict(centre=1300.0, gamma=2.0, gamma_inst=4.0, depth=0.5, base=0.9),
                         sigma=1e-3)
    assert any(set(c) >= {"gamma", "gamma_inst"} for c in fit.null_combinations)
    assert "EXACTLY DEGENERATE" in fit.message
    assert math.isinf(fit.stderr["gamma"]) and math.isinf(fit.stderr["gamma_inst"])
    rng = np.random.default_rng(3)
    lam2 = np.linspace(1270.0, 1330.0, 25)
    y2 = _voigt_dip(lam2, 1300.0, 0.5, 8.0, 0.3, 0.9) + rng.normal(0, 2e-2, lam2.size)
    fit2 = fit_broadening(lam2, y2, _voigt_dip, dict(centre=1300.0, gamma=1.0, sigma=7.0, depth=0.3,
                                                      base=0.9), sigma=2e-2,
                          bounds={"gamma": (0.0, 50.0), "sigma": (0.1, 50.0)},
                          null_values={"gamma": 0.0})
    assert {"gamma", "sigma"} & set(fit2.undetermined)            # measured: the fit chose a
    assert any({a, b} == {"gamma", "sigma"} for a, b, _ in fit2.degenerate)   # Lorentzian (gamma 7.9)
    assert "cannot be told apart" in fit2.message
    assert fit2.null_tests["gamma"] < 4.0 and "does not require it" in fit2.message
