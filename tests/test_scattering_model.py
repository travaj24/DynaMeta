"""Fast unit tests for the shared ScatteringModel link (materials/scattering.py + Material wiring,
roadmap R3): ONE tau(n;T) law derives BOTH the optical Drude gamma(n)=1/tau and the transport drift
mobility mu(n)=q/(m_cond 1/tau). Pure numpy."""
import numpy as np
import pytest

from dynameta.constants import Q_E, M_E
from dynameta.materials import (DrudeOptical, TransportModel, Material, KaneOpticalMass,
                                MatthiessenGamma, ScatteringModel)

N = np.array([1e26, 4e26, 1e27, 2e27])


def test_constant_tau_gamma_and_tau():
    sm = ScatteringModel(one_over_tau=1.1e14, m_cond_kg=0.35 * M_E)
    g = sm.gamma_optical_of_n()
    assert np.allclose(g(N), 1.1e14, rtol=0, atol=0)
    assert np.allclose(sm.tau_s(N), 1.0 / 1.1e14)


def test_mobility_tau_identity():
    # the q/(m mu) identity: gamma_opt * m_cond * mu / q == hall_factor (==1) exactly
    sm = ScatteringModel(one_over_tau=MatthiessenGamma(gamma_const_rad_s=2.0e14, gamma_phonon_300K_rad_s=1e14),
                         m_cond_kg=0.3 * M_E)
    g = sm.gamma_optical_of_n()(N)
    mu = sm.mobility_of_n()(N)
    assert np.allclose(g * (0.3 * M_E) * mu / Q_E, 1.0, rtol=1e-12)


def test_dc_vs_optical_mass_not_conflated():
    # gamma is mass-INDEPENDENT (= 1/tau); only mu uses m_cond. Two different masses -> gamma unchanged.
    ot = MatthiessenGamma(gamma_const_rad_s=1.0e14)
    sm_a = ScatteringModel(one_over_tau=ot, m_cond_kg=0.25 * M_E)
    sm_b = ScatteringModel(one_over_tau=ot, m_cond_kg=0.50 * M_E)
    assert np.allclose(sm_a.gamma_optical_of_n()(N), sm_b.gamma_optical_of_n()(N))   # gamma identical
    assert np.allclose(sm_a.mobility_of_n()(N), 2.0 * sm_b.mobility_of_n()(N))       # mu ~ 1/m_cond


def test_material_without_scattering_is_unchanged():
    opt = DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14)
    tr = TransportModel(n_bg_m3=4e26, eps_static=9.5, dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E)
    m = Material("ito", optical=opt, transport=tr)            # scattering=None
    assert m.optical is opt and m.transport is tr             # untouched, same objects


def test_material_link_derives_gamma_and_mobility():
    sm = ScatteringModel(one_over_tau=MatthiessenGamma(gamma_const_rad_s=1.1e14), m_cond_kg=0.35 * M_E)
    opt = DrudeOptical(eps_inf=4.25, m_opt_kg=KaneOpticalMass(m0_kg=0.225 * M_E, alpha_eV=0.5),
                       gamma_rad_s=9.9e9)                     # placeholder gamma -> overridden by link
    tr = TransportModel(n_bg_m3=4e26, eps_static=9.5, dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E)
    m = Material("ito", optical=opt, transport=tr, scattering=sm)
    # optical gamma now == 1/tau; the link replaced the placeholder, and on a FRESH object (opt untouched)
    assert np.allclose(m.optical.gamma_rad_s(N), 1.1e14)
    assert opt.gamma_rad_s == 9.9e9                           # original not mutated
    # the linked eps equals a constant-gamma Drude with the SAME m_opt
    ref = DrudeOptical(eps_inf=4.25, m_opt_kg=KaneOpticalMass(m0_kg=0.225 * M_E, alpha_eV=0.5),
                       gamma_rad_s=1.1e14)
    assert np.allclose(m.optical.eps(1500e-9, n_m3=N), ref.eps(1500e-9, n_m3=N), rtol=0, atol=1e-15)
    # transport mobility now n-callable = q/(m_cond 1/tau)
    assert np.allclose(m.transport.mobility_m2Vs_of_n_m3(N), Q_E / (0.35 * M_E * 1.1e14))


def test_link_requires_drude_and_transport():
    sm = ScatteringModel(one_over_tau=1e14)
    from dynameta.materials import ConstantOptical
    with pytest.raises(ValueError):                          # no transport
        Material("x", optical=DrudeOptical(4.0, 0.3 * M_E, 1e14), transport=None, scattering=sm)
    tr = TransportModel(n_bg_m3=4e26, eps_static=9.5, dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E)
    with pytest.raises(ValueError):                          # non-Drude optical
        Material("y", optical=ConstantOptical(2.0 + 0j), transport=tr, scattering=sm)
