"""Unit coverage for the Phase-4 reconfigurable EffectModels + graphene sheet: PCMModel (Bruggeman
EMA) and LiquidCrystalModel (uniaxial tensor) in core.effects, and graphene_sigma / sheet_rt in
core.graphene. Pure numpy. Run: python -m pytest tests/test_reconfigurable.py -q
"""
import numpy as np
import pytest

from dynameta.constants import HBAR, C_LIGHT, Q_E as Q
from dynameta.core.effects import PCMModel, LiquidCrystalModel
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


def test_s4_22_interband_kubo_vs_direct_quadrature():
    """audit S4-22 (wave-7, 2026-11-02): an INDEPENDENT numerical oracle for the interband Kubo
    terms (the FEM validation consumed the same graphene_sigma, so it could not see them).
    MEASURED at 1.55 um, E_F 0..0.8 eV, T 10 / 30 / 77 / 300 K:
      * Re: exact -- |dRe| <= 3.6e-15 sigma0 everywhere;
      * Im: the module's thermally SMOOTHED log is exact only as T -> 0. |dIm| / sigma0 <= 0.10 at
        300 K (2 E_F within ~0.1 eV of hbar omega), <= 0.075 at 77 K, 0.040 at the resonance
        point for T <= 30 K; away from it O((kT / |hw - 2 E_F|)^2) (10 K, E_F 0.1..0.8: <= 2e-4).
    The bars below pin both statements, sign-sensitively: a conjugated Im (the exp(+i omega t)
    slip) misses the oracle by 2 |Im| >= 0.32 sigma0 at every non-zero E_F tested. Also the
    LOW-T overflow this oracle exposed: at T_K = 3 the bare sinh/cosh overflowed to NaN (with a
    RuntimeWarning); the scaled form is finite and Pauli-steps correctly."""
    pytest.importorskip("scipy.integrate")
    hwJ = 2.0 * np.pi * HBAR * C_LIGHT / LAM

    def inter(EF_eV, T):
        from dynameta.constants import KB
        EF, kT = EF_eV * Q, KB * T
        W = EF + 2.0 * kT * np.log1p(np.exp(-EF / kT))
        w = 2.0 * np.pi * C_LIGHT / LAM
        intra = 1j * Q ** 2 * W / (np.pi * HBAR ** 2 * (w + 1j / 1e-13))
        return (complex(graphene_sigma(EF, LAM, tau_s=1e-13, T_K=T)) - intra) / SIGMA0
    for T, im_bar in ((300.0, 0.11), (77.0, 0.08), (30.0, 0.045)):
        for EF in (0.0, 0.1, 0.3, 0.35, 0.4, 0.45, 0.5, 0.8):
            s, (re_o, im_o) = inter(EF, T), _kubo_interband_oracle(EF, LAM, T)
            assert abs(s.real - re_o) < 1e-12
            assert abs(s.imag - im_o) < im_bar
            if EF > 0.0:
                assert abs(-s.imag - im_o) > 0.3              # the conjugate is caught
    for EF in (0.1, 0.3, 0.5, 0.8):                           # T -> 0 away from resonance
        s, (re_o, im_o) = inter(EF, 10.0), _kubo_interband_oracle(EF, LAM, 10.0)
        assert abs(s.imag - im_o) < 3e-4 and abs(s.real - re_o) < 1e-12
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lo = complex(graphene_sigma(0.3 * Q, LAM, T_K=3.0, tau_s=1e-11))   # 2 E_F < hw: absorbing
        hi = complex(graphene_sigma(0.5 * Q, LAM, T_K=3.0, tau_s=1e-11))   # blocked
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo.real / SIGMA0 == pytest.approx(1.0, abs=1e-3) and hi.real / SIGMA0 < 1e-3
    assert hwJ / Q == pytest.approx(0.80, abs=0.01)


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
