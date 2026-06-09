"""Fast unit tests for the carrier-heating two-temperature ENZ driver (carriers.carrier_heating, R9).
Pure numpy/scipy (no devsim/ngsolve/fdtd). The rigorous oracle is validation/carrier_heating_enz.py."""
import numpy as np
import pytest

from dynameta.constants import M_E, KB
from dynameta.materials import DrudeOptical
from dynameta.carriers.carrier_heating import (TwoTempParams, two_temperature_response,
                                               carrier_heating_transient, kane_mass_of_Te,
                                               gamma_of_Te, fermi_energy_J)
from dynameta.transient_optics import optical_transient_response

M0, ALPHA_EV, GAMMA0, N = 0.35 * M_E, 0.5, 1.0e14, 1.0e27
DRUDE0 = DrudeOptical(eps_inf=3.9, m_opt_kg=M0, gamma_rad_s=GAMMA0)
GAMMA_E = (np.pi ** 2 / 2.0) * N * KB ** 2 / float(fermi_energy_J(N, M0, ALPHA_EV))
PARAMS = TwoTempParams(C_e=lambda Te: GAMMA_E * Te, C_l=2.4e6, G_e_l=6.0e15, alpha_abs=1.0)


def test_kane_mass_off_switch_and_monotone():
    # alpha=0 -> m0 EXACTLY (the byte-identical off-switch)
    assert kane_mass_of_Te(M0, 0.0, N, 5000.0) == M0
    assert float(kane_mass_of_Te(M0, 0.0, N, 300.0)) == M0
    # <m*> rises with Te (hot electrons climb the nonparabolic band)
    m_cold = float(kane_mass_of_Te(M0, ALPHA_EV, N, 300.0))
    m_hot = float(kane_mass_of_Te(M0, ALPHA_EV, N, 3000.0))
    assert m_hot > m_cold > M0


def test_gamma_off_switch():
    assert gamma_of_Te(GAMMA0, 5000.0, p=0.0) == GAMMA0          # p=0 -> gamma0 exactly
    assert gamma_of_Te(GAMMA0, 600.0, p=1.0) == pytest.approx(GAMMA0 * 2.0)   # linear in Te


def test_two_temperature_no_pump_stays_at_T0():
    t = np.linspace(0.0, 2e-12, 200)
    _t, Te, Tl = two_temperature_response(t, lambda tt: 0.0, PARAMS, T0_K=300.0)
    assert np.max(np.abs(Te - 300.0)) < 1e-9 and np.max(np.abs(Tl - 300.0)) < 1e-9


def test_two_temperature_monotone_rise_then_fall():
    t = np.linspace(0.0, 3e-12, 400)
    pump = lambda tt: 3e20 * np.exp(-((tt - 0.4e-12) / 6e-14) ** 2)
    _t, Te, _Tl = two_temperature_response(t, pump, PARAMS, T0_K=300.0)
    ipk = int(np.argmax(Te))
    assert Te[ipk] > 800.0                                       # the pump heats the electrons
    assert np.all(np.diff(Te[:ipk + 1]) >= -1e-6)                # monotone up to the peak
    assert np.all(np.diff(Te[ipk:]) <= 1e-6)                     # monotone cooling after


def test_two_temperature_energy_conservation():
    # no pump after the pulse: total thermal energy = integral of absorbed power (no loss term)
    t = np.linspace(0.0, 4e-12, 800)
    pump = lambda tt: 2e20 * np.exp(-((tt - 0.4e-12) / 5e-14) ** 2)
    _t, Te, Tl = two_temperature_response(t, pump, PARAMS, T0_K=300.0)
    U_in = np.trapezoid(np.array([pump(tt) for tt in t]), t) if hasattr(np, "trapezoid") else \
        np.trapz(np.array([pump(tt) for tt in t]), t)
    U_e = 0.5 * GAMMA_E * (Te[-1] ** 2 - 300.0 ** 2)             # electron energy (C_e = gamma_e Te)
    U_l = PARAMS.C_l * (Tl[-1] - 300.0)
    assert abs((U_e + U_l) - U_in) / U_in < 0.05                # conserved (G only redistributes)


def test_carrier_heating_reduces_to_fixed_drude():
    # alpha_per_eV=0, gamma_p=0 -> per-instant Drude collapses to drude0 -> byte-identical R(t)
    t = np.linspace(0.0, 2e-12, 150)
    pump = lambda tt: 3e20 * np.exp(-((tt - 0.4e-12) / 6e-14) ** 2)
    n_of_t = lambda tt: N
    _t, R_fix, _T, _e = optical_transient_response(t, n_of_t, 1500e-9, drude_model=DRUDE0)
    _th, R_h, _Th, _eh, _Te, _Tl = carrier_heating_transient(t, pump, 1500e-9, drude0=DRUDE0,
                                                             ttm_params=PARAMS, n_m3=N,
                                                             alpha_per_eV=0.0, gamma_p=0.0)
    assert np.max(np.abs(R_h - R_fix)) < 1e-12


def test_optical_transient_requires_exactly_one_drude():
    t = np.linspace(0.0, 1e-12, 10)
    with pytest.raises(ValueError):
        optical_transient_response(t, lambda tt: N, 1500e-9)                       # neither
    with pytest.raises(ValueError):
        optical_transient_response(t, lambda tt: N, 1500e-9, drude_model=DRUDE0,
                                   drude_of_t=lambda tt: DRUDE0)                    # both
