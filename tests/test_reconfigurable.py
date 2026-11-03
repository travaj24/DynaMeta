"""Unit coverage for the Phase-4 reconfigurable EffectModels + graphene sheet: PCMModel (Bruggeman
EMA) and LiquidCrystalModel (uniaxial tensor) in core.effects, and graphene_sigma / sheet_rt in
core.graphene. Pure numpy. Run: python -m pytest tests/test_reconfigurable.py -q
"""
import numpy as np
import pytest

from dynameta.constants import HBAR, C_LIGHT, Q_E as Q
from dynameta.core.effects import PCMModel, LiquidCrystalModel
from dynameta.core.backend import JAX_AVAILABLE
from dynameta.core.graphene import graphene_sigma, sheet_rt, SIGMA0, Z0

LAM = 1.55e-6


def test_pcm_reduces_to_end_states():
    ea, ec = complex(16.0, 0.5), complex(36.0, 6.0)
    pcm = PCMModel(eps_amorphous=ea, eps_crystalline=ec)
    assert pcm.eps({"crystalline_fraction": 0.0}, LAM) == pytest.approx(ea, abs=1e-12)  # amorphous
    assert pcm.eps({"crystalline_fraction": 1.0}, LAM) == pytest.approx(ec, abs=1e-12)  # crystalline
    assert pcm.eps({}, LAM) == pytest.approx(ea, abs=1e-12)                             # default f=0
    # lossless negative-real endpoint (ENZ/metal idealization): exact by short-circuit, NOT the
    # wrong tie-break root (-eps_a/2) the bare quadratic would pick at the boundary (audit PCM-1)
    enz = PCMModel(eps_amorphous=complex(6.0, 0.0), eps_crystalline=complex(-10.0, 0.0))
    assert enz.eps({"crystalline_fraction": 1.0}, LAM) == pytest.approx(complex(-10.0, 0.0), abs=1e-12)
    assert enz.eps({"crystalline_fraction": 0.0}, LAM) == pytest.approx(complex(6.0, 0.0), abs=1e-12)


def test_pcm_monotonic_passive_and_bounded():
    ea, ec = complex(16.0, 0.5), complex(36.0, 6.0)
    pcm = PCMModel(eps_amorphous=ea, eps_crystalline=ec)
    fs = np.linspace(0, 1, 9)
    es = np.array([pcm.eps({"crystalline_fraction": f}, LAM) for f in fs])
    assert np.all(np.diff(np.sqrt(es).real) > 0)              # Re(n) rises with crystallinity
    assert np.all(es.imag >= -1e-12)                          # passive (Im(eps) >= 0)
    for f, e in zip(fs, es):                                  # within Wiener bounds (on Re)
        ser = 1.0 / (f / ec + (1 - f) / ea)
        par = f * ec + (1 - f) * ea
        assert ser.real - 1e-9 <= e.real <= par.real + 1e-9


def test_pcm_rejects_fraction_out_of_range():
    pcm = PCMModel(eps_amorphous=complex(16.0, 0.5), eps_crystalline=complex(36.0, 6.0))
    with pytest.raises(ValueError):
        pcm.eps({"crystalline_fraction": 1.5}, LAM)
    with pytest.raises(ValueError):
        pcm.eps({"crystalline_fraction": -0.1}, LAM)


def test_liquid_crystal_planar_tensor_and_isotropic_reduction():
    no, ne = 1.53, 1.71
    lc = LiquidCrystalModel(n_o=no, n_e=ne)
    e0 = lc.eps({"director_angle_rad": 0.0}, LAM)             # optic axis along x
    assert np.allclose(np.diag(e0), [ne ** 2, no ** 2, no ** 2])
    assert np.allclose(e0 - np.diag(np.diag(e0)), 0.0)        # diagonal at theta=0
    assert np.allclose(LiquidCrystalModel(1.6, 1.6).eps({"director_angle_rad": 0.7}, LAM),
                       1.6 ** 2 * np.eye(3))                  # n_e=n_o -> isotropic


def test_liquid_crystal_uniaxial_eigenvalues_are_rotation_invariant():
    no, ne = 1.53, 1.71
    lc = LiquidCrystalModel(n_o=no, n_e=ne)
    for th in (0.0, 0.4, 1.0, np.pi / 2):
        eps = lc.eps({"director_angle_rad": th}, LAM)
        assert np.allclose(eps, eps.T)                        # symmetric tensor
        ev = np.sort(np.linalg.eigvals(eps).real)
        # AUDIT T-6: rtol=0.0 -- eigenvalues are O(3); numpy's default rtol=1e-5 would relax this
        # rotation-invariance gate to ~3e-5 absolute.
        assert np.allclose(ev, np.sort([no ** 2, no ** 2, ne ** 2]), rtol=0.0, atol=1e-9)
        # tilt mixes x-z: off-diagonal eps_xz = (ne^2-no^2) cos(theta) sin(theta)
        assert eps[0, 2] == pytest.approx((ne ** 2 - no ** 2) * np.cos(th) * np.sin(th), abs=1e-9)


def test_graphene_universal_conductivity_and_pauli_blocking():
    # well below the interband threshold (E_F=0), Re(sigma) ~ the universal sigma0 = e^2/4hbar
    assert graphene_sigma(0.0, LAM).real == pytest.approx(SIGMA0, rel=0.05)
    hw = 2.0 * np.pi * HBAR * C_LIGHT / LAM                    # photon energy ~0.80 eV
    re = [graphene_sigma(EF * Q, LAM).real / SIGMA0 for EF in (0.0, 0.3, 0.5, 0.7)]
    assert re[0] > 0.9 and re[-1] < 0.1                       # Pauli-blocked above 2E_F = hbar omega
    assert np.all(np.diff(re) < 0)                            # monotone collapse with gating
    assert graphene_sigma(2.0 * Q, LAM).real > 0              # passive (Re(sigma) > 0)
    assert hw / Q == pytest.approx(0.80, abs=0.02)
    # the EXACT finite-T (sinh/cosh) interband form gives a VANISHING blocked-tail deep in the
    # Pauli-blocked region (2E_F >> hbar omega) -- an arctan smoothing would leave ~1.3% (audit
    # GRAPH-1). Use a long tau so the intraband Drude loss is negligible and the interband tail shows.
    assert graphene_sigma(1.0 * Q, LAM, tau_s=1e-11).real / SIGMA0 < 1e-3


def _kubo_interband_oracle(EF_eV, lam, T):
    """sigma_inter / sigma0 by DIRECT QUADRATURE of the finite-T Kubo integral (Falkovsky &
    Varlamov, Eur. Phys. J. B 56, 281 (2007)), energies in eV, exp(-i omega t):
        sigma_inter / sigma0 = G(hw/2) + i (4 hw / pi) Int_0^inf [G(e) - G(hw/2)] / (hw^2 - 4 e^2) de,
        G(e) = f(-e) - f(e),  f the Fermi function at mu = |E_F|.
    G is built from the two Fermi FUNCTIONS (not the sinh/cosh closed form the module uses), and
    the principal-value-free integrand (numerator vanishes at e = hw/2) is split at hw/2 and mu
    (+-30 kT) so quad sees every feature. Self-checks measured 2026-11-02: -> the T = 0 log
    -(1/2 pi) ln((hw + 2 mu)^2 / (hw - 2 mu)^2) away from resonance as T -> 0 (E_F 0.8 eV, 10 K:
    -0.34965 vs -0.34970), and at E_F = 0 the analytic finite-T term -(8 ln2 / pi)(kT / hw)
    (300 K: -0.0577 vs -0.0570, next order)."""
    from scipy.integrate import quad
    h = HBAR * 2.0 * np.pi * C_LIGHT / lam / Q
    from dynameta.constants import KB
    a, kT, mu = 0.5 * h, KB * T / Q, abs(EF_eV)

    def fermi(x):
        return np.exp(-x) / (1.0 + np.exp(-x)) if x > 0 else 1.0 / (np.exp(x) + 1.0)

    def G(e):
        return fermi((-e - mu) / kT) - fermi((e - mu) / kT)
    Ga = G(a)

    def ig(e):
        return 0.0 if abs(e - a) <= 1e-9 else (G(e) - Ga) / (h * h - 4.0 * e * e)
    w = 30.0 * kT
    edges = sorted({0.0, max(0.0, a - w), a, a + w, max(0.0, mu - w), mu, mu + w,
                    4.0 * (a + mu + w)})
    I = sum(quad(ig, lo, hi, limit=400, epsabs=1e-14, epsrel=1e-11)[0]
            for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo)
    I += quad(ig, edges[-1], np.inf, limit=400, epsabs=1e-14, epsrel=1e-11)[0]
    return Ga, (4.0 * h / np.pi) * I


def _inter(EF_eV, T, *, lam=LAM, tau=1e-13, model="exact"):
    """sigma_inter / sigma0 = (graphene_sigma - the Drude intraband term) / sigma0 (complex)."""
    from dynameta.constants import KB
    EF, kT = abs(EF_eV) * Q, KB * T
    W = EF + 2.0 * kT * np.log1p(np.exp(-EF / kT))
    w = 2.0 * np.pi * C_LIGHT / lam
    intra = 1j * Q ** 2 * W / (np.pi * HBAR ** 2 * (w + 1j / tau))
    return (complex(graphene_sigma(EF_eV * Q, lam, tau_s=tau, T_K=T, model=model)) - intra) / SIGMA0


def test_s4_22_interband_kubo_vs_direct_quadrature():
    """audit S4-22 (wave-7, 2026-11-02): an INDEPENDENT numerical oracle for the interband Kubo
    terms (the FEM validation consumed the same graphene_sigma, so it could not see them).
    MEASURED at 1.55 um, E_F 0..0.8 eV, T 10 / 30 / 77 / 300 K:
      * Re: exact -- |dRe| <= 3.6e-15 sigma0 everywhere (both models);
      * Im, model="exact" (the default since the 2026-11-02 exact-Im follow-up): |dIm| <= 6.2e-15
        sigma0 (10 K: 1.3e-15, 300 K: 6.2e-15);
      * Im, model="T0" (the pre-2026-11-02 thermally SMOOTHED log, exact only as T -> 0, kept as
        an explicit option): |dIm| / sigma0 = 0.1001 at 300 K (2 E_F within ~0.1 eV of hbar
        omega), <= 0.075 at 77 K, 0.040 at the resonance point for T <= 30 K; away from it
        O((kT / |hw - 2 E_F|)^2) (10 K, E_F 0.1..0.8: <= 2e-4). Pinned below as before.
    Sign-sensitive: a conjugated Im (the exp(+i omega t) slip) misses the oracle by 2 |Im| >= 0.32
    sigma0 at every non-zero E_F tested. Also the LOW-T overflow this oracle exposed: at T_K = 3 the
    bare sinh/cosh overflowed to NaN (with a RuntimeWarning); the scaled form is finite and
    Pauli-steps correctly (the exact Im is finite there too: digamma arguments ~1e3)."""
    pytest.importorskip("scipy.integrate")
    hwJ = 2.0 * np.pi * HBAR * C_LIGHT / LAM
    for T, im_bar in ((300.0, 0.11), (77.0, 0.08), (30.0, 0.045)):
        for EF in (0.0, 0.1, 0.3, 0.35, 0.4, 0.45, 0.5, 0.8):
            re_o, im_o = _kubo_interband_oracle(EF, LAM, T)
            s, s0 = _inter(EF, T), _inter(EF, T, model="T0")
            assert abs(s.real - re_o) < 1e-12 and s0.real == s.real
            assert abs(s.imag - im_o) < 1e-12                 # exact (measured <= 6.2e-15)
            assert abs(s0.imag - im_o) < im_bar               # the documented T0 approximation
            if EF > 0.0:
                assert abs(-s.imag - im_o) > 0.3              # the conjugate is caught
    for EF in (0.1, 0.3, 0.5, 0.8):                           # T -> 0 away from resonance
        re_o, im_o = _kubo_interband_oracle(EF, LAM, 10.0)
        s, s0 = _inter(EF, 10.0), _inter(EF, 10.0, model="T0")
        assert abs(s.imag - im_o) < 1e-12 and abs(s.real - re_o) < 1e-12
        assert abs(s0.imag - im_o) < 3e-4
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lo = complex(graphene_sigma(0.3 * Q, LAM, T_K=3.0, tau_s=1e-11))   # 2 E_F < hw: absorbing
        hi = complex(graphene_sigma(0.5 * Q, LAM, T_K=3.0, tau_s=1e-11))   # blocked
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo.real / SIGMA0 == pytest.approx(1.0, abs=1e-3) and hi.real / SIGMA0 < 1e-3
    assert hwJ / Q == pytest.approx(0.80, abs=0.01)


def _kubo_im_mpmath(mu, t):
    """Im(sigma_inter)/sigma0 by a 30-digit mpmath quadrature of the Falkovsky-Varlamov (2007, Eq.
    (11)) / Hanson (2008, Eq. (1), Gamma -> 0) principal-value Kubo integral, energies in units of
    hbar omega (mu = |E_F|/hw, t = kT/hw), in the singularity-SUBTRACTED form
        Im = (4/pi) int_0^inf [G(e) - G(1/2)] / (1 - 4 e^2) de,
        G(e) = sinh(e/t) / (cosh(mu/t) + cosh(e/t))
    (PV int_0^inf de/(1 - 4e^2) = 0, so the subtraction changes nothing and removes the pole).
    Independent of the module's Maldague / digamma / Gauss-Legendre route; tanh-sinh quadrature
    split at e = 1/2, mu and +-(1, 3, 10, 40) t around both."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(30):
        mu, t, a = mp.mpf(mu), mp.mpf(t), mp.mpf(1) / 2

        def G(e):
            return mp.sinh(e / t) / (mp.cosh(mu / t) + mp.cosh(e / t))
        Ga = G(a)
        # removable point e = 1/2: the limit -G'(1/2)/4, G'(e) = (1 + ch_mu ch_e)/(t (ch_mu + ch_e)^2)
        # (tanh-sinh nodes of a tiny [mu, 1/2] panel can round onto e = 1/2 exactly at 30 digits)
        lim = -(1 + mp.cosh(mu / t) * mp.cosh(a / t)) / (4 * t * (mp.cosh(mu / t) + mp.cosh(a / t)) ** 2)

        def ig(e):
            d = 1 - 4 * e * e
            return lim if d == 0 else (G(e) - Ga) / d
        pts = sorted({mp.mpf(0), a, mu} | {p + k * t for p in (a, mu)
                                            for k in (-40, -10, -3, -1, 1, 3, 10, 40)
                                            if p + k * t > 0})
        return float(4 / mp.pi * mp.quad(ig, pts + [mp.inf]))


def test_graphene_exact_interband_im_vs_mpmath_principal_value():
    """S4-22 exact-Im follow-up (wave-7, 2026-11-02): the default interband Im is the EXACT
    finite-T principal-value Kubo integral. Gate against _kubo_im_mpmath over T 4..600 K, E_F
    -0.6..0.6 eV (only |E_F| enters), hbar omega 0.1 / 0.8 / 2.0 eV, incl. the resonance 2E_F = hw.
    MEASURED (2026-11-02): max |dIm| = 1.8e-14 sigma0 here (8.3e-16 over a 96-point superset with
    the raw dimensionless kernel). The old smoothed log (model="T0") misses the same grid by up
    to 0.56 sigma0 (hw 0.1 eV, 600 K, E_F 0) and by 0.040 at the T <= 77 K resonance, so this gate
    fails on the pre-2026-11-02 code. Also: tau_s enters the Drude term only -- the extracted
    interband part is tau-independent (1e-15 .. 1e-10 s; measured spread 1.1e-16 sigma0)."""
    from dynameta.constants import KB
    worst = worst_T0 = 0.0
    for hw_eV, EFs in ((0.1, (0.0, -0.05, 0.05, 0.6)), (0.8, (-0.4, 0.0, 0.35, 0.4, 0.41, -0.6)),
                       (2.0, (0.0, 0.6))):
        lam = 2.0 * np.pi * HBAR * C_LIGHT / (hw_eV * Q)
        hw = 2.0 * np.pi * HBAR * C_LIGHT / lam / Q             # the module's own hbar omega, eV
        for T in (4.0, 77.0, 300.0, 600.0):
            for EF in EFs:
                o = _kubo_im_mpmath(abs(EF) / hw, KB * T / Q / hw)
                worst = max(worst, abs(_inter(EF, T, lam=lam).imag - o))
                worst_T0 = max(worst_T0, abs(_inter(EF, T, lam=lam, model="T0").imag - o))
    assert worst < 1e-11
    assert worst_T0 > 0.5                                     # the defect this replaces
    spread = 0.0
    for T in (4.0, 300.0):
        for EF in (0.0, 0.4, 0.6):
            v = [_inter(EF, T, tau=tau) for tau in (1e-15, 1e-13, 1e-10)]
            spread = max(spread, max(abs(x - v[0]) for x in v))
    assert spread < 1e-11


def test_graphene_exact_im_reduces_to_T0_closed_form():
    """T -> 0: away from 2|E_F| = hw the exact Im converges to the closed form (the model="T0"
    smoothed log, whose own regularizer is O(T^2)) at O(T^2): MEASURED (1.55 um, 2026-11-02)
    |exact - T0| at E_F 0.3 eV = 8.0e-6 / 2.0e-6 / 5.0e-7 sigma0 at 4 / 2 / 1 K, ratio
    4.0002 / 4.00005 per halving. AT the resonance E_F = hw/2 both diverge like ln T, and the exact
    finite-T value is (1/pi)[ln(pi kT/(2 hw)) - gamma_E] + O(T^2) (Maldague: Re psi(1/2) = -gamma_E
    - 2 ln 2) -- measured residual 6.1e-9 at 1 K, 9.7e-8 at 4 K. The T0 log reads (1/pi) ln(kT/hw)
    there: exact - T0 = (ln(pi/2) - gamma_E)/pi = -0.03999 at every T, the documented 0.040
    resonance offset (and fails this gate on the old code)."""
    from dynameta.constants import KB
    hw = 2.0 * np.pi * HBAR * C_LIGHT / LAM / Q
    for EF in (0.1, 0.3, 0.5, 0.6):
        d = [abs(_inter(EF, T).imag - _inter(EF, T, model="T0").imag) for T in (4.0, 2.0, 1.0)]
        assert d[0] < 1e-5 and d[2] < 6e-7
        assert d[0] / d[1] == pytest.approx(4.0, abs=0.01)
        assert d[1] / d[2] == pytest.approx(4.0, abs=0.01)
    for T, bar in ((1.0, 1e-8), (4.0, 2e-7)):
        t = KB * T / Q / hw
        pred = (np.log(np.pi * t / 2.0) - np.euler_gamma) / np.pi
        assert abs(_inter(hw / 2.0, T).imag - pred) < bar
        assert _inter(hw / 2.0, T, model="T0").imag - pred == pytest.approx(
            -(np.log(np.pi / 2.0) - np.euler_gamma) / np.pi, abs=1e-6)


def test_graphene_interband_kramers_kronig_consistent():
    """Kramers-Kronig consistency of the module's OWN Re and Im (exp(-i omega t), sheet
    conductivity analytic in the upper half plane):
        Im(E) / sigma0 = -(2E/pi) PV int_0^inf [Re(E')/sigma0 - 1] / (E'^2 - E^2) dE'
    (the -1 is free: PV int_0^inf dE'/(E'^2 - E^2) = 0; it makes the integrand decay, so a finite
    grid E' in [0, hw + 2E_F + 90 kT] has no tail). The PV is the repo's Maclaurin KK transform
    (core.effects.base.kramers_kronig_dn_rows, dalpha := (Re/sigma0 - 1)/(hbar c) so that
    Im = -2E dn), on grids with hw ON a node, N = 4001 and 8001, Richardson-extrapolated.
    MEASURED (1.55 um, 300 K, 2026-11-02): |KK - Im_exact| = 5.1e-10 sigma0 at E_F 0 (8001-point
    raw: 3.9e-7, the E' -> 0 corner) and <= 1.3e-12 at E_F 0.35 / 0.40 / 0.45, whereas the T0 log misses KK by 0.058 (E_F 0) / 0.10 (E_F 0.35, 0.45) /
    0.039 (E_F 0.40) -- this gate fails on the pre-2026-11-02 code."""
    from dynameta.core.effects.base import kramers_kronig_dn_rows
    from dynameta.constants import KB
    T = 300.0
    E0 = 2.0 * np.pi * HBAR * C_LIGHT / LAM
    for EF in (0.0, 0.35, 0.4, 0.45):
        kk = []
        for N in (4001, 8001):
            h = (2.0 * EF * Q + 90.0 * KB * T + E0) / (N - 1)
            k0 = int(round(E0 / h))
            h = E0 / k0
            E = h * np.arange(N) + 1e-9 * h                   # E' = 0 node nudged (lambda = inf)
            re = np.array([_inter(EF, T, lam=2.0 * np.pi * HBAR * C_LIGHT / e).real for e in E])
            dn = kramers_kronig_dn_rows(E, ((re - 1.0) / (HBAR * C_LIGHT))[None, :],
                                        e_eval_J=E[k0])[0]
            kk.append(-2.0 * E[k0] * dn)
        kk = (4.0 * kk[1] - kk[0]) / 3.0
        assert abs(kk - _inter(EF, T).imag) < 5e-9
        assert abs(kk - _inter(EF, T, model="T0").imag) > 0.035


@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_graphene_exact_im_jax_grad_matches_fd_and_kubo_slope():
    """graphene_sigma stays JAX-traceable in E_F with the exact Im (fixed 336-node rule, no adaptive
    control flow): jit(grad) of Im(sigma)/sigma0 wrt E_F (eV) vs (a) a central FD of the numpy path
    and (b) the slope of the mpmath Kubo integral (interband only: the Drude term is subtracted in
    jnp). MEASURED (1.55 um, 2026-11-02): (a) rel 1.6e-8 at 4 K (FD truncation), <= 2.4e-10 at
    300 K; (b) <= 2.6e-12 per eV vs the Richardson (O(h^4)) mpmath slope at 300 K, E_F 0.35 / 0.40
    / 0.45 (slopes -5.970 / -0.389 / +5.168 per eV), where the T0 log's slope is off by 0.52 / 0.015 / 0.52 per eV (fails on the old code).
    jax-vs-numpy values agree to <= 3.4e-16 sigma0."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dynameta.constants import KB

    def im_total(e, T, model="exact"):
        return jnp.imag(graphene_sigma(e * Q, LAM, T_K=T, model=model)) / SIGMA0

    def im_inter(e, T, model="exact"):
        kT, EF = KB * T, jnp.abs(e) * Q
        W = EF + 2.0 * kT * jnp.log1p(jnp.exp(-EF / kT))
        w = 2.0 * np.pi * C_LIGHT / LAM
        intra = 1j * Q ** 2 * W / (np.pi * HBAR ** 2 * (w + 1j / 1e-13))
        return jnp.imag(graphene_sigma(e * Q, LAM, T_K=T, model=model) - intra) / SIGMA0

    def np_im(e, T):
        return complex(graphene_sigma(e * Q, LAM, T_K=T)).imag / SIGMA0
    for T in (4.0, 300.0):
        g = jax.jit(jax.grad(lambda e: im_total(e, T)))
        for EF in (-0.3, 0.05, 0.39, 0.4, 0.41, 0.6):
            h = 1e-7 if T < 10 else 1e-6
            fd = (np_im(EF + h, T) - np_im(EF - h, T)) / (2.0 * h)
            assert float(g(jnp.asarray(EF))) == pytest.approx(fd, rel=1e-6, abs=1e-6)
            assert complex(graphene_sigma(jnp.asarray(EF * Q), LAM, T_K=T)) == pytest.approx(
                complex(graphene_sigma(EF * Q, LAM, T_K=T)), rel=1e-13, abs=0.0)
    hw = 2.0 * np.pi * HBAR * C_LIGHT / LAM / Q
    t = KB * 300.0 / Q / hw
    g = jax.jit(jax.grad(lambda e: im_inter(e, 300.0)))
    g0 = jax.grad(lambda e: im_inter(e, 300.0, "T0"))
    for EF in (0.35, 0.4, 0.45):
        def D(hm):                                            # central FD of the mpmath oracle
            return (_kubo_im_mpmath((EF + hm) / hw, t)
                    - _kubo_im_mpmath((EF - hm) / hw, t)) / (2.0 * hm)
        slope = (4.0 * D(5e-5) - D(1e-4)) / 3.0               # Richardson: O(h^4)
        assert float(g(jnp.asarray(EF))) == pytest.approx(slope, abs=1e-9)
        assert abs(float(g0(jnp.asarray(EF))) - slope) > 0.01


def test_graphene_sheet_energy_conservation_and_fresnel_limit():
    n1, n2 = 1.0, 1.5
    # INDEPENDENT energy balance (not the tautological A:=1-R-T): the flux deficit equals the Ohmic
    # sheet dissipation A_poynting = Z0 Re(sigma) |t|^2 / n1 computed separately (audit F1).
    s = graphene_sigma(0.0, LAM)
    r, t, R, T, A = sheet_rt(n1, n2, s)
    assert A == pytest.approx(Z0 * s.real * abs(t) ** 2 / n1, abs=1e-9) and A >= 0.0
    # gate-tunable absorption: ON (E_F=0) >> OFF (Pauli-blocked at E_F=0.6 eV)
    A_off = sheet_rt(n1, n2, graphene_sigma(0.6 * Q, LAM))[4]
    assert A > 5.0 * A_off
    # sigma -> 0 recovers the bare Fresnel reflection
    assert sheet_rt(n1, n2, 0.0)[0] == pytest.approx((n1 - n2) / (n1 + n2), abs=1e-12)


def test_graphene_sheet_warns_on_gain_sheet():
    import warnings
    # a gain sheet (Re(sigma) < 0) violates passivity for exp(-iwt): sheet_rt must WARN (audit GRAPH-2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, _, R, T, A = sheet_rt(1.0, 1.0, complex(-1e-4, 0.0))
    assert any(issubclass(x.category, RuntimeWarning) for x in w)
    assert A < 0.0 and T > 1.0                                 # active sheet -> negative absorption


def test_graphene_input_guards():
    with pytest.raises(ValueError):
        graphene_sigma(0.4 * Q, LAM, tau_s=-1e-13)            # non-positive relaxation time
    with pytest.raises(ValueError):
        sheet_rt(complex(1.0, 0.1), 1.5, 0.0)                 # lossy incidence medium
    with pytest.raises(ValueError):
        sheet_rt(1.0, 1.5, 0.0, theta_deg=10.0)              # oblique not supported
