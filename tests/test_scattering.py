"""Fast unit tests for the resolved Drude scattering/mass closures (materials/scattering.py, roadmap R2).
Pure numpy; the eps-level + Park gates live in validation/drude_matthiessen_kane.py."""
import numpy as np
import pytest

from dynameta.constants import Q_E, M_E, HBAR
from dynameta.materials import DrudeOptical, KaneOpticalMass, MatthiessenGamma

N = np.array([1e26, 4e26, 1e27, 2e27])
LAM = 1500e-9


def _park_ito_dos_mass(n, m_low=0.27 * M_E, alpha=0.5):
    """Independent reference: the exact bulk Kane DOS-mass closure used in examples/park_2021."""
    n = np.maximum(np.asarray(n, float), 1e10)
    kF = (3.0 * np.pi ** 2 * n) ** (1.0 / 3.0)
    E_F = HBAR ** 2 * kF ** 2 / (2.0 * m_low)
    return m_low * np.sqrt(1.0 + 2.0 * alpha * E_F / Q_E)


def test_kane_alpha_zero_is_constant():
    m = KaneOpticalMass(m0_kg=0.225 * M_E, alpha_eV=0.0)
    assert np.allclose(m(N), 0.225 * M_E, rtol=0, atol=0)        # exactly m0


def test_kane_matches_park_reference():
    m = KaneOpticalMass(m0_kg=0.27 * M_E, alpha_eV=0.5, exponent=0.5)
    assert np.allclose(m(N), _park_ito_dos_mass(N), rtol=1e-12)


def test_kane_monotone_and_wp2_sublinear():
    m = KaneOpticalMass(m0_kg=0.27 * M_E, alpha_eV=0.5)
    mm = m(N)
    assert np.all(np.diff(mm) > 0)                              # heavier carriers at higher n
    wp2 = N * Q_E ** 2 / (8.8541878128e-12 * mm)
    dwp2 = np.diff(wp2) / np.diff(N)                            # d(wp^2)/dn must DECREASE (sub-linear)
    assert np.all(np.diff(dwp2) < 0)


def test_matthiessen_off_switch_is_constant():
    g = MatthiessenGamma(gamma_const_rad_s=1.1e14)
    assert np.allclose(g(N), 1.1e14, rtol=0, atol=0)            # all other channels 0 -> exact constant


def test_matthiessen_additivity_and_temperature_trend():
    base = dict(gamma_const_rad_s=5.0e13, gamma_phonon_300K_rad_s=4.0e13)
    g300 = MatthiessenGamma(T_K=300.0, **base)
    g200 = MatthiessenGamma(T_K=200.0, **base)
    g400 = MatthiessenGamma(T_K=400.0, **base)
    # additivity: at 300 K phonon term == its 300 K value, total = const + phonon
    assert float(g300(1e27)) == pytest.approx(5.0e13 + 4.0e13, rel=1e-12)
    # linear high-T phonon: Gamma(400) > Gamma(300) > Gamma(200)
    assert float(g400(1e27)) > float(g300(1e27)) > float(g200(1e27))
    # ionized-impurity channel rises with n (Brooks-Herring scaling), off by default
    gii = MatthiessenGamma(gamma_const_rad_s=1e13, bh_prefactor_rad_s=3e13, bh_n_ref_m3=1e27,
                           m_opt=KaneOpticalMass(m0_kg=0.27 * M_E, alpha_eV=0.5))
    vals = gii(N)
    assert np.all(np.diff(vals) > 0)                           # more scattering at higher doping


def test_drude_byte_identical_when_neutral():
    # the resolved closures plugged into DrudeOptical reproduce the constant-Drude eps EXACTLY when
    # the new knobs are neutral (alpha=0, only the const damping channel).
    d_const = DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14)
    d_resolved = DrudeOptical(eps_inf=4.25,
                              m_opt_kg=KaneOpticalMass(m0_kg=0.225 * M_E, alpha_eV=0.0),
                              gamma_rad_s=MatthiessenGamma(gamma_const_rad_s=1.1e14))
    for lam in (1200e-9, 1500e-9, 2000e-9):
        a = d_const.eps(lam, n_m3=N)
        b = d_resolved.eps(lam, n_m3=N)
        assert np.allclose(a, b, rtol=0, atol=1e-15)


def test_drude_passivity_with_resolved_gamma():
    d = DrudeOptical(eps_inf=4.25,
                     m_opt_kg=KaneOpticalMass(m0_kg=0.27 * M_E, alpha_eV=0.5),
                     gamma_rad_s=MatthiessenGamma(gamma_const_rad_s=8e13, gamma_phonon_300K_rad_s=3e13))
    eps = d.eps(1500e-9, n_m3=N)
    assert np.all(np.imag(eps) >= 0.0)                         # exp(-i w t) passive
