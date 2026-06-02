"""Unit coverage for the Phase-3 QCSE / MQW electro-absorption path: the QuantumWell Stark driver
(carriers.qcse) + the ElectroAbsorptionModel and Kramers-Kronig helper (core.effects). Pure
numpy/scipy (a 1D BenDaniel-Duke eigenproblem; no FEM/devsim). Run: python -m pytest tests/test_qcse.py -q
"""
import numpy as np
import pytest

from dynameta.constants import HBAR, M_E, C_LIGHT, Q_E as Q
from dynameta.carriers.qcse import QuantumWell, StarkState, INFINITE_WELL_STARK_BETA
from dynameta.core.effects import ElectroAbsorptionModel, kramers_kronig_dn

ME, MHH = 0.067 * M_E, 0.34 * M_E


def _gaas(nz=801):
    return QuantumWell(well_width_m=10e-9, barrier_e_J=0.25 * Q, barrier_h_J=0.15 * Q,
                       m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q,
                       exciton_binding_J=0.010 * Q, nz=nz, n_pad=2.0)


def test_infinite_well_stark_beta_constant():
    assert INFINITE_WELL_STARK_BETA == pytest.approx(2.1944e-3, rel=2e-3)   # textbook coefficient


def test_quantum_well_flat_band_state():
    s = _gaas().solve(0.0)
    assert isinstance(s, StarkState)
    assert s.E_e1_J > 0 and s.E_hh1_J > 0                      # confinement energies above edges
    assert 0.9 < s.overlap <= 1.0                              # symmetric well -> near-unity overlap
    assert s.E_transition_J == pytest.approx(1.42 * Q + s.E_e1_J + s.E_hh1_J - 0.010 * Q)


def test_quantum_well_deep_limit_matches_analytic():
    L = 15e-9
    qw = QuantumWell(well_width_m=L, barrier_e_J=200.0 * Q, barrier_h_J=200.0 * Q,
                     m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q, nz=1501, n_pad=1.5)
    E1 = qw.solve(0.0).E_e1_J
    E1_ana = HBAR ** 2 * np.pi ** 2 / (2.0 * ME * L ** 2)
    assert E1 == pytest.approx(E1_ana, rel=0.05)               # infinite-well ground energy
    # quadratic Stark coefficient vs analytic beta q^2 m F^2 L^4 / hbar^2
    Fs = np.array([0.0, 1e6, 2e6, 3e6])
    dE = np.array([qw.solve(F).E_e1_J for F in Fs]); dE -= dE[0]
    C_num = -np.polyfit(Fs ** 2, dE, 1)[0]
    C_ana = INFINITE_WELL_STARK_BETA * Q ** 2 * ME * L ** 4 / HBAR ** 2
    assert C_num / C_ana == pytest.approx(1.0, abs=0.15)       # finite-barrier converges toward 1


def test_quantum_well_quadratic_redshift_and_overlap_drop():
    qw = _gaas()
    s0, s1, s2 = qw.solve(0.0), qw.solve(3e6), qw.solve(6e6)
    assert s1.E_transition_J < s0.E_transition_J               # redshift under field
    assert s2.E_transition_J < s1.E_transition_J
    # ~quadratic: doubling-ish the field grows the redshift super-linearly
    red1, red2 = s0.E_transition_J - s1.E_transition_J, s0.E_transition_J - s2.E_transition_J
    assert red2 > 2.0 * red1                                   # 2x field -> >2x shift (quadratic)
    assert s2.overlap < s1.overlap < s0.overlap                # e-h overlap falls with field


def test_quantum_well_rejects_bad_input():
    with pytest.raises(ValueError):
        QuantumWell(well_width_m=0.0, barrier_e_J=0.25 * Q, barrier_h_J=0.15 * Q,
                    m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q)
    with pytest.raises(ValueError):
        QuantumWell(well_width_m=10e-9, barrier_e_J=0.0, barrier_h_J=0.15 * Q,
                    m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q)    # non-positive barrier


def test_kramers_kronig_sign_structure():
    # a positive Gaussian d-alpha peak -> dn>0 just below the peak, dn<0 just above (KK kernel)
    E = np.linspace(1.0, 2.0, 2001) * Q
    E0, sig = 1.5 * Q, 0.01 * Q
    da = 1e6 * np.exp(-0.5 * ((E - E0) / sig) ** 2)
    dn = kramers_kronig_dn(E, da)
    below = dn[np.argmin(np.abs(E - (E0 - 3 * sig)))]
    above = dn[np.argmin(np.abs(E - (E0 + 3 * sig)))]
    assert below > 0 and above < 0


def test_kramers_kronig_rejects_nonuniform_grid():
    E = np.array([1.0, 1.1, 1.3, 1.6]) * Q                    # non-uniform
    with pytest.raises(ValueError):
        kramers_kronig_dn(E, np.ones_like(E))
    with pytest.raises(ValueError):
        kramers_kronig_dn(np.linspace(1, 2, 5) * Q, np.ones(4))   # shape mismatch


def test_electroabsorption_flat_band_reduces_to_background():
    qw = _gaas()
    ET0 = qw.transition_energy_J(0.0); sig = 0.006 * Q
    eps_bg = complex(3.6 ** 2, 0.01)
    eam = ElectroAbsorptionModel(qw=qw, eps_bg=eps_bg, alpha0_per_m=1e6, broadening_J=sig,
                                 e_grid_J=(ET0 - 0.3 * Q, ET0 + 0.3 * Q, 2001))
    lam = 2.0 * np.pi * HBAR * C_LIGHT / (ET0 - 2.0 * sig)
    assert eam.eps({"E": np.zeros(3)}, lam) == pytest.approx(eps_bg, abs=1e-9)   # F=0 -> eps_bg


def test_electroabsorption_field_turns_on_absorption():
    qw = _gaas()
    ET0 = qw.transition_energy_J(0.0); sig = 0.006 * Q
    eps_bg = complex(3.6 ** 2, 0.01)
    eam = ElectroAbsorptionModel(qw=qw, eps_bg=eps_bg, alpha0_per_m=1e6, broadening_J=sig,
                                 e_grid_J=(ET0 - 0.3 * Q, ET0 + 0.3 * Q, 2001))
    lam = 2.0 * np.pi * HBAR * C_LIGHT / (ET0 - 2.0 * sig)     # probe below the F=0 edge
    F = np.array([0.0, 0.0, 7e6])
    da = eam.delta_alpha_per_m({"E": F}, lam)
    eps = eam.eps({"E": F}, lam)
    assert da > 0                                              # redshift turns ON absorption
    assert eps.imag > eps_bg.imag and eps.imag > 0            # Im(eps)>0 absorber (exp(-iwt))


def test_electroabsorption_requires_field_and_straddling_grid():
    qw = _gaas()
    ET0 = qw.transition_energy_J(0.0)
    eam = ElectroAbsorptionModel(qw=qw, eps_bg=complex(13.0, 0.0), alpha0_per_m=1e6,
                                 broadening_J=0.006 * Q, e_grid_J=(ET0 - 0.3 * Q, ET0 + 0.3 * Q, 1001))
    lam = 2.0 * np.pi * HBAR * C_LIGHT / ET0
    with pytest.raises(ValueError):
        eam.eps({}, lam)                                      # no fields['E']
    bad = ElectroAbsorptionModel(qw=qw, eps_bg=complex(13.0, 0.0), alpha0_per_m=1e6,
                                 broadening_J=0.006 * Q, e_grid_J=(0.1 * Q, 0.2 * Q, 1001))
    with pytest.raises(ValueError):
        bad.eps({"E": np.zeros(3)}, lam)                      # grid does not straddle E_T
