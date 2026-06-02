"""Unit coverage for the Phase-4 reconfigurable EffectModels + graphene sheet: PCMModel (Bruggeman
EMA) and LiquidCrystalModel (uniaxial tensor) in core.effects, and graphene_sigma / sheet_rt in
core.graphene. Pure numpy. Run: python -m pytest tests/test_reconfigurable.py -q
"""
import numpy as np
import pytest

from dynameta.constants import HBAR, C_LIGHT, Q_E as Q
from dynameta.core.effects import PCMModel, LiquidCrystalModel
from dynameta.core.graphene import graphene_sigma, sheet_rt, SIGMA0

LAM = 1.55e-6


def test_pcm_reduces_to_end_states():
    ea, ec = complex(16.0, 0.5), complex(36.0, 6.0)
    pcm = PCMModel(eps_amorphous=ea, eps_crystalline=ec)
    assert pcm.eps({"crystalline_fraction": 0.0}, LAM) == pytest.approx(ea, abs=1e-12)  # amorphous
    assert pcm.eps({"crystalline_fraction": 1.0}, LAM) == pytest.approx(ec, abs=1e-12)  # crystalline
    assert pcm.eps({}, LAM) == pytest.approx(ea, abs=1e-12)                             # default f=0


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
        assert np.allclose(ev, np.sort([no ** 2, no ** 2, ne ** 2]), atol=1e-9)
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


def test_graphene_sheet_energy_conservation_and_fresnel_limit():
    n1, n2 = 1.0, 1.5
    # energy budget R + T + A = 1, A >= 0 for a passive sheet
    r, t, R, T, A = sheet_rt(n1, n2, graphene_sigma(0.0, LAM))
    assert R + T + A == pytest.approx(1.0, abs=1e-9) and A >= 0.0
    # gate-tunable absorption: ON (E_F=0) >> OFF (Pauli-blocked at E_F=0.6 eV)
    A_off = sheet_rt(n1, n2, graphene_sigma(0.6 * Q, LAM))[4]
    assert A > 5.0 * A_off
    # sigma -> 0 recovers the bare Fresnel reflection
    assert sheet_rt(n1, n2, 0.0)[0] == pytest.approx((n1 - n2) / (n1 + n2), abs=1e-12)


def test_graphene_input_guards():
    with pytest.raises(ValueError):
        graphene_sigma(0.4 * Q, LAM, tau_s=-1e-13)            # non-positive relaxation time
    with pytest.raises(ValueError):
        sheet_rt(complex(1.0, 0.1), 1.5, 0.0)                 # lossy incidence medium
    with pytest.raises(ValueError):
        sheet_rt(1.0, 1.5, 0.0, theta_deg=10.0)              # oblique not supported
