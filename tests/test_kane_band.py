"""v0.16: the Kane band of the equilibrium electrostatics (one band object for the density closure, the DOS
mass and the optical mass) -- pure-math gates (no DEVSIM).

carriers/kane_band.py evaluates n = N_c(m0) G(eta, b), G = F_AH(eta) + Delta(eta, b), b = alpha kT, with the
Kane increment Delta = G - F_1/2 from a generalized Gauss-Laguerre rule (eta <= 8) and a Sommerfeld expansion
(eta >= 6), blended on [6, 8]. ORACLES (written here, independent of the library): scipy quadrature of the
defining integral G = (1/Gamma(5/2)) Int f(x - eta) K'(x) dx, K = [x (1 + b x)]^(3/2); the T = 0 Kane Fermi
sphere; the band dispersion itself for the optical mass.

ERROR BUDGET (MEASURED 2026-11-14, eta in [-10, 60] step 0.25):
  increment |Delta - (G - F_1/2)_quad| / G_quad <= 1.5e-6 (b = 0.0103) / 3.5e-6 (b = 0.03), worst at eta = 7.5
  (inside the blend zone). Gate: 1e-5.
  total |G_closure - G_quad| <= |F_AH - F_1/2,quad| + 1e-5 G_quad: the Aymerich-Humet error of the parabolic
  part (up to 3.8e-3 near eta = -3 at b = 0) is the closure's whole error; the Kane part adds < 4e-6.
On v0.15.1 every gate here fails at import: carriers.kane_band and KaneBand do not exist.
Film numbers below: the generic film n_bg 3e26 m^-3, eps_static 9.0, with a Kane band m0 0.22 m_e,
alpha 0.4 eV^-1.
"""
import math
import warnings

import numpy as np
import pytest
from scipy import integrate
from scipy.special import roots_genlaguerre

from dynameta.carriers import kane_band as K
from dynameta.carriers import physics_equilibrium as PE
from dynameta.constants import HBAR, KB, M_E, Q_E, T_REF, V_T
from dynameta.materials.scattering import KaneOpticalMass
from dynameta.materials.transport_model import KaneBand, TransportModel
from dynameta.warning_types import AdvisoryWarning

G52 = math.gamma(2.5)


def _f(x, eta):
    return 0.5 * (1.0 - math.tanh(0.5 * (x - eta)))


def G_quad(eta, b, p=1.5):
    """(1/Gamma(p + 1)) Int_0^inf f(x - eta) d/dx [x (1 + b x)]^p dx by adaptive quadrature."""
    def dK(x):
        u = x + b * x * x
        return p * u ** (p - 1.0) * (1.0 + 2.0 * b * x) if u > 0 else 0.0
    top = max(eta, 0.0) + 60.0
    pts = [q for q in (eta - 5.0, eta, eta + 5.0) if 0.0 < q < top]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # quad's round-off notice at the 1e-13 target
        v, _e = integrate.quad(lambda x: _f(x, eta) * dK(x), 0.0, top, points=pts or None, limit=400,
                               epsabs=0.0, epsrel=1e-13)
    return v / math.gamma(p + 1.0)


def F_quad(eta, j):
    """Normalized complete Fermi-Dirac integral F_j(eta) = (1/Gamma(j+1)) Int x^j f(x - eta) dx."""
    return G_quad(eta, 0.0, p=j + 1.0)


ETAS = np.round(np.arange(-10.0, 60.001, 0.5), 6)


def test_laguerre_constants_are_the_rule():
    """The literal nodes / weights are the generalized Gauss-Laguerre rule (weight x^(1/2) e^-x, N = 32,
    nodes below 45) to round-off, and the dropped nodes contribute < 2e-15 of Delta for eta <= 8 (MEASURED
    1.6e-15 at b = 0.03: below the round-off of G)."""
    x, w = roots_genlaguerre(32, 0.5)
    np.testing.assert_allclose(K._LAG_X, x[:22], rtol=1e-13)
    np.testing.assert_allclose(K._LAG_W, w[:22], rtol=1e-12)
    assert x[21] < 45.0 < x[22]
    np.testing.assert_allclose(K._LAG_D, np.exp(-np.asarray(K._LAG_X)), rtol=1e-15)
    b = 0.03
    g = lambda t: np.sqrt(1 + t) * (1 + 2 * t) - 1
    terms = 1.5 * w * g(b * x) * math.exp(8.0) / (1 + np.exp(8.0 - x)) / G52
    assert np.sum(terms[22:]) / np.sum(terms) < 2e-15


@pytest.mark.parametrize("b", [0.0103, 0.02, 0.03])
def test_increment_and_closure_against_quadrature(b):
    """ORACLE: scipy quadrature of G and of F_1/2. Increment gate 1e-5 relative to G (MEASURED worst 3.5e-6
    at b = 0.03, eta = 7.5); the total closure error is bounded by the Aymerich-Humet error of its parabolic
    part plus that. Sign-sensitive: Delta > 0 and grows with b."""
    worst = (0.0, None)
    for eta in ETAS:
        gq, fq = G_quad(eta, b), G_quad(eta, 0.0)
        d = K.kane_increment(eta, b)
        err = abs(d - (gq - fq)) / gq
        if err > worst[0]:
            worst = (err, eta)
        assert d > 0.0
        assert abs(K.kane_closure(eta, b) - gq) <= abs(PE.F12_aymerich_humet(eta) - fq) + 1e-5 * gq
    assert worst[0] < 1e-5, worst
    assert np.all(np.diff(K.kane_increment(ETAS, b)) > 0.0)                  # monotone in eta
    assert np.all(K.kane_increment(ETAS, 1.1 * b) > K.kane_increment(ETAS, b))


def test_alpha_zero_is_the_parabolic_closure_bit_for_bit():
    """b = 0: Delta = 0 exactly, the closure IS F12_aymerich_humet, the inverse IS invert_F12, and a KaneBand
    with alpha = 0 reports its m0 as the DOS mass (constant)."""
    eta = np.linspace(-30.0, 90.0, 1201)
    assert np.array_equal(K.kane_increment(eta, 0.0), np.zeros_like(eta))
    assert np.array_equal(K.kane_closure(eta, 0.0), np.array([PE.F12_aymerich_humet(e) for e in eta]))
    for t in (1e-6, 0.3, 7.0, 200.0):
        assert K.invert_kane_closure(t, 0.0) == PE.invert_F12(t)
    kb = KaneBand(0.3 * M_E, 0.0)
    assert np.array_equal(kb.dos_mass_kg(np.array([1e24, 3e26, 1e28])), np.full(3, 0.3 * M_E))
    assert kb.eta_of_density(3e26) == PE.invert_F12(3e26 / PE.effective_dos_m3(0.3 * M_E))


def test_first_order_in_b_is_the_series_coefficient():
    """As b -> 0, Delta / b -> (15/4) F_3/2(eta) (the d/db of G at b = 0: (3/2) Int (-df/dx) x^(5/2) dx /
    Gamma(5/2)); F_3/2 by quadrature. The increment's error is linear in b, so this holds to ~1e-5 relative
    of Delta at any small b -- a check of the formula, not only of its size. Gate 1e-4."""
    b = 1e-7
    for eta in (-10.0, -2.0, 0.0, 3.0, 6.5, 7.5, 10.0, 25.0, 60.0):
        ratio = K.kane_increment(eta, b) / (b * 3.75 * F_quad(eta, 1.5))
        assert abs(ratio - 1.0) < 1e-4, (eta, ratio)


def test_zero_temperature_limit_is_the_kane_fermi_sphere():
    """T -> 0 at fixed E_F: n = N_c(m0, T) G(E_F/kT, alpha kT) -> k_F^3/(3 pi^2) with hbar^2 k_F^2/(2 m0) =
    E_F (1 + alpha E_F). E_F = 0.7 eV, alpha = 0.4 /eV, m0 = 0.22 m_e; at 3 K (eta = 2708) to 1e-5. At 300 K
    the EXACT Kane density (quadrature) exceeds the Fermi sphere by +3.97e-3, the first Sommerfeld correction
    (pi^2/6) [(3/4) u^-2 + 6 b/u] (u = eta (1 + b eta)) to 2 % of itself (sign-sensitive: thermal smearing
    adds states on the convex Kane DOS); the closure's +3.27e-3 differs from it by the Aymerich-Humet error of
    its parabolic part at eta = 27 (-1.0e-3 of F_1/2), bounded as in the quadrature gate."""
    m0, a, EF = 0.22 * M_E, 0.4, 0.7
    kF = math.sqrt(2.0 * m0 * EF * Q_E * (1.0 + a * EF)) / HBAR
    n0 = kF ** 3 / (3.0 * math.pi ** 2)
    kT = KB * 3.0 / Q_E
    n3 = PE.effective_dos_m3(m0, 3.0) * K.kane_closure(EF / kT, K.kane_b(a, 3.0))
    assert abs(n3 / n0 - 1.0) < 1e-5, n3 / n0
    eta, b = EF / V_T, K.kane_b(a)
    Nc = PE.effective_dos_m3(m0)
    u = eta * (1 + b * eta)
    som = math.pi ** 2 / 6 * (0.75 / u ** 2 + 6 * b / u)
    assert (Nc * G_quad(eta, b) / n0 - 1.0) == pytest.approx(som, rel=2e-2)
    ah_err = abs(PE.F12_aymerich_humet(eta) - G_quad(eta, 0.0))
    assert abs(Nc * K.kane_closure(eta, b) - Nc * G_quad(eta, b)) <= Nc * ah_err + 1e-5 * n0


def test_inversion_and_bracket():
    for b in (0.0103, 0.03):
        for eta in (-15.0, -3.0, 0.0, 7.0, 7.9, 30.0, 70.0):
            assert abs(K.invert_kane_closure(K.kane_closure(eta, b), b) - eta) < 1e-8
    with pytest.raises(ValueError, match="invert_kane_closure: target .* exceeds G_Kane"):
        K.invert_kane_closure(1e6, 0.01)
    with pytest.raises(ValueError, match="b = alpha kT must be finite and >= 0"):
        K.kane_closure(1.0, -0.01)


def test_band_dos_mass_definition_and_limits():
    """KaneBand.dos_mass_kg(n) = m0 (G/F_AH)^(2/3) at the band's eta(n): the parabolic closure with that mass
    holds n at the SAME Fermi level (to the inversion tolerance); T -> 0 it is the counting mass m0 (1 +
    alpha E_F) (0.2727 m_e at 3e26 m^-3, m0 = 0.22 m_e, alpha = 0.4); at 300 K 0.27317 m_e."""
    kb = KaneBand(0.22 * M_E, 0.4)
    n = 3e26
    md = kb.dos_mass_kg(n)
    eta = kb.eta_of_density(n)
    assert abs(PE.effective_dos_m3(md) * PE.F12_aymerich_humet(eta) / n - 1.0) < 1e-9
    assert md / M_E == pytest.approx(0.27317, abs=5e-5)
    gF = HBAR ** 2 * (3 * math.pi ** 2 * n) ** (2.0 / 3.0) / (2 * kb.m0_kg) / Q_E
    EF0 = (-1 + math.sqrt(1 + 4 * 0.4 * gF)) / (2 * 0.4)
    assert kb.dos_mass_kg(n, T_K=3.0) / (kb.m0_kg * (1 + 0.4 * EF0)) == pytest.approx(1.0, abs=1e-4)
    arr = kb.dos_mass_kg(np.array([1e25, 3e26, 2e27]))
    assert arr.shape == (3,) and np.all(np.diff(arr) > 0)


def test_optical_mass_is_the_fermi_surface_mass_of_the_same_band():
    """band.optical_mass() is KaneOpticalMass(m0, alpha) (one kernel) and equals hbar^2 k_F/(dE/dk) at the
    Fermi wavevector k_F = (3 pi^2 n)^(1/3) of the dispersion E (1 + alpha E) = hbar^2 k^2/(2 m0) (numerical
    dE/dk here, 1e-7). The f-sum mass at 300 K (carrier_heating.kane_mass_of_Te) is 0.20 % heavier at 3e26
    m^-3 (thermal occupation of heavier states): the
    Drude side uses the degenerate T = 0 mass."""
    from dynameta.carriers.carrier_heating import kane_mass_of_Te
    m0, a = 0.22 * M_E, 0.4
    kb = KaneBand(m0, a)
    om = kb.optical_mass()
    assert om == KaneOpticalMass(m0_kg=m0, alpha_eV=a) == KaneOpticalMass.from_band(kb)

    def E_of_k(k):
        g = HBAR ** 2 * k ** 2 / (2 * m0) / Q_E
        return (-1 + math.sqrt(1 + 4 * a * g)) / (2 * a) * Q_E
    for n in (1e26, 3e26, 1.5e27):
        kF = (3 * math.pi ** 2 * n) ** (1.0 / 3.0)
        h = kF * 1e-5
        dEdk = (E_of_k(kF + h) - E_of_k(kF - h)) / (2 * h)
        assert float(om(n)) == pytest.approx(HBAR ** 2 * kF / dEdk, rel=1e-7, abs=0)
    r = float(kane_mass_of_Te(m0, a, 3e26, 300.0)) / float(om(3e26))
    assert 1.0005 < r < 1.003


def test_sub_band_model_is_a_different_band():
    """The Schrodinger-Poisson in-plane-Kane bulk limit N_c [F_1/2 + 2 b F_3/2] is NOT this isotropic band
    (first-order coefficient 15/4): at the 3-D band's own E_F (3e26 m^-3, m0 0.22 m_e, alpha 0.4, 300 K) the
    sub-band model holds 0.8633 n_bg, and its own bulk E_F is 55.3 meV higher (MEASURED). Consistency check of the stated
    difference (both from library kernels)."""
    from dynameta.carriers.schrodinger_poisson import sp_bulk_density_m3, sp_bulk_fermi_level_J
    kb = KaneBand(0.22 * M_E, 0.4)
    EF = kb.eta_of_density(3e26) * V_T * Q_E
    r = sp_bulk_density_m3(EF, kb.m0_kg, T_REF, alpha_np_per_eV=0.4) / 3e26
    assert r == pytest.approx(0.8633, abs=5e-4)
    d = sp_bulk_fermi_level_J(3e26, kb.m0_kg, T_REF, alpha_np_per_eV=0.4) / Q_E - EF / Q_E
    assert d == pytest.approx(0.0553, abs=5e-4)
    b = 1e-7                                                            # the two first-order coefficients
    eta = 5.0
    sp = sp_bulk_density_m3(eta * V_T * Q_E, kb.m0_kg, T_REF, alpha_np_per_eV=b / V_T)
    par = sp_bulk_density_m3(eta * V_T * Q_E, kb.m0_kg, T_REF)
    assert (sp / par - 1.0) / b * F_quad(eta, 0.5) / F_quad(eta, 1.5) == pytest.approx(2.0, rel=1e-3)
    assert K.kane_increment(eta, b) / b / F_quad(eta, 1.5) == pytest.approx(3.75, rel=1e-4)


def test_transport_model_band_api_and_guards():
    kb = KaneBand(0.22 * M_E, 0.4)
    tr = TransportModel(3e26, 9.0, band=kb)
    assert tr.dos_mass_kg_of_n_m3 == kb.dos_mass_kg                       # derived from the band
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # the density-dependent-mass advisory must NOT fire
        assert tr.dos_mass_at_bg("here") == kb.dos_mass_kg(3e26)
    import dataclasses
    tr2 = dataclasses.replace(tr, band=KaneBand(0.3 * M_E, 0.4))         # re-derived from the new band
    assert tr2.dos_mass_kg_of_n_m3.__self__.m0_kg == 0.3 * M_E
    assert TransportModel(3e26, 9.0, dos_mass_kg_of_n_m3=lambda n: 0.3 * M_E).band == "parabolic"
    with pytest.raises(ValueError, match="omit dos_mass_kg_of_n_m3"):
        TransportModel(3e26, 9.0, dos_mass_kg_of_n_m3=lambda n: 0.3 * M_E, band=kb)
    with pytest.raises(ValueError, match="dos_mass_kg_of_n_m3 is required"):
        TransportModel(3e26, 9.0)
    with pytest.raises(TypeError, match="band must be 'parabolic' or a KaneBand"):
        TransportModel(3e26, 9.0, dos_mass_kg_of_n_m3=lambda n: 0.3 * M_E, band="kane")
    for phys, extra in (("drift_diffusion", {}), ("bipolar_dd", dict(hole_mobility_m2Vs_of_n_m3=lambda n: 1e-3,
                                                                     n_i_m3=1e16, tau_srh_s=1e-7))):
        with pytest.raises(NotImplementedError, match="EQUILIBRIUM closure only"):
            TransportModel(3e26, 9.0, band=kb, physics=phys, mobility_m2Vs_of_n_m3=lambda n: 4e-3, **extra)
    with pytest.raises(ValueError, match="m0_kg must be finite and > 0"):
        KaneBand(0.0, 0.4)
    with pytest.raises(ValueError, match="alpha_per_eV must be finite and >= 0"):
        KaneBand(0.22 * M_E, -0.1)


def test_density_dependent_mass_without_band_still_warns():
    """No band + a density-dependent DOS mass: the density-dependent-mass advisory still fires (the inconsistency is not
    resolved). The band's own DOS-mass callable passed as a plain parabolic mass is such a case too."""
    kb = KaneBand(0.22 * M_E, 0.4)
    tr = TransportModel(3e26, 9.0, dos_mass_kg_of_n_m3=lambda n: kb.dos_mass_kg(n))
    with pytest.warns(AdvisoryWarning, match="density-dependent"):
        tr.dos_mass_at_bg("here")


def test_screening_length_follows_the_band():
    """screening_length_m with a KaneBand: L_s^2 = eps eps0 kT/(q^2 N_c(m0) G'(eta_bg)). ORACLE: G' = F_AH'
    (central difference of the fit, the parabolic part the closure keeps) + dDelta/deta from the QUADRATURE
    increment (central difference, step 1e-3, error ~1e-7). Gate 1e-5. MEASURED (3e26, eps 9.0, m0 0.22 m_e,
    alpha 0.4): 0.7456 nm, against 0.8137 nm for the parabolic band frozen at the band's DOS mass at n_bg and
    0.9064 nm for the band-edge mass: the Kane band screens 8.4 % shorter than the frozen-mass model."""
    from dynameta.carriers.screening_mesh import screening_length_m
    from dynameta.constants import EPS0
    kb = KaneBand(0.22 * M_E, 0.4)
    tr = TransportModel(3e26, 9.0, band=kb)
    b = kb.b()
    eta = kb.eta_of_density(3e26)
    h = 1e-3
    dF = (PE.F12_aymerich_humet(eta + 1e-4) - PE.F12_aymerich_humet(eta - 1e-4)) / 2e-4
    dD = ((G_quad(eta + h, b) - G_quad(eta + h, 0.0)) - (G_quad(eta - h, b) - G_quad(eta - h, 0.0))) / (2 * h)
    N = PE.effective_dos_m3(kb.m0_kg)
    Ls = math.sqrt(9.0 * EPS0 * KB * T_REF / (Q_E ** 2 * N * (dF + dD)))
    assert screening_length_m(tr) == pytest.approx(Ls, rel=1e-5, abs=0)
    assert screening_length_m(tr) * 1e9 == pytest.approx(0.7456, abs=1e-4)
    md = kb.dos_mass_kg(3e26)
    frozen = TransportModel(3e26, 9.0, dos_mass_kg_of_n_m3=lambda n: md + 0.0 * np.asarray(n))
    assert screening_length_m(frozen) * 1e9 == pytest.approx(0.8137, abs=1e-4)
    # alpha = 0 band: the parabolic number bit for bit
    par = TransportModel(3e26, 9.0, dos_mass_kg_of_n_m3=lambda n: 0.22 * M_E + 0.0 * np.asarray(n))
    assert screening_length_m(TransportModel(3e26, 9.0, band=KaneBand(0.22 * M_E, 0.0))) == screening_length_m(par)
