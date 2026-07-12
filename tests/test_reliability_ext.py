"""Fast unit tests for the reliability-axis completion (REL3/5/6/7/8/9). Pure numpy/scipy; the
rigorous oracles live in validation/reliability_{em,lidt,fatigue,stressmig,hci,corrosion}.py."""
import numpy as np
import pytest

from dynameta.reliability import (EmParams, black_mttf_s, blech_immortal, current_density_A_m2,
                                  miner_time_to_failure_s, ThermalNode, lidt_fluence_J_m2,
                                  cw_steady_temperature_K, cw_critical_intensity_W_m2,
                                  MechanicalProps, biaxial_stress_Pa, coffin_manson_nf,
                                  norris_landzberg_af, brittle_survival, cycles_to_failure,
                                  korhonen_kappa_m2_s, korhonen_relax, void_nucleates,
                                  soret_flux_per_m2_s, trap_generation_rate_per_m2_s,
                                  hci_time_to_failure_s, deal_grove_thickness_m,
                                  peck_time_to_failure_s, peck_af)


# ---- REL3 electromigration ----

def test_em_black_and_blech():
    assert np.isinf(black_mttf_s(0.0, 350.0, A_s=1.0))                  # no current -> immortal
    assert black_mttf_s(2e9, 350.0, A_s=1.0) < black_mttf_s(1e9, 350.0, A_s=1.0)
    assert blech_immortal(1e10, 5e-6) and not blech_immortal(1e10, 50e-6)
    assert current_density_A_m2(1e-3, 1e-6, 100e-9) == pytest.approx(1e10)
    with pytest.raises(ValueError):
        black_mttf_s(-1.0, 350.0, A_s=1.0)
    with pytest.raises(ValueError):
        current_density_A_m2(1e-3, 0.0, 1e-7)


def test_em_miner_constant_stress_reduces():
    p = EmParams.calibrated(J_A_m2=2e9, T_K=350.0, mttf_s=100.0)
    t = np.linspace(0.0, 300.0, 30001)
    tf = miner_time_to_failure_s(t, lambda tt: 2e9, lambda tt: 350.0, p)
    assert tf == pytest.approx(100.0, rel=1e-6)                         # constant -> MTTF itself
    assert np.isinf(miner_time_to_failure_s(t, lambda tt: 0.0, lambda tt: 350.0, p))


# ---- REL5 LIDT / runaway ----

def test_lidt_zero_drive_and_sqrt_law():
    node = ThermalNode(R_th_K_W=1e4, C_th_J_K=1e-9, area_m2=1e-10)
    assert cw_steady_temperature_K(lambda T: 0.5, 0.0, node) == 300.0
    assert float(lidt_fluence_J_m2(4e-9, F_ref_J_m2=1.0, tau_ref_s=1e-9)) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        ThermalNode(R_th_K_W=0.0, C_th_J_K=1e-9, area_m2=1e-10)


def test_lidt_runaway_detection():
    node = ThermalNode(R_th_K_W=1e4, C_th_J_K=1e-9, area_m2=1e-10)
    absorbed = lambda T: 0.01 + 1e-4 * (T - 300.0)
    I_cr = cw_critical_intensity_W_m2(absorbed, node, I_lo=1e6, I_hi=1e13)
    assert 1e8 < I_cr < 1e10                                            # near 1/(a1 S R) ~ 1e10 scale
    with pytest.raises(RuntimeError):
        cw_steady_temperature_K(absorbed, 10.0 * I_cr, node)            # over threshold -> runaway


# ---- REL6 fatigue ----

def test_fatigue_split_and_guards():
    cu = MechanicalProps(E_Pa=110e9, nu=0.34, cte_per_K=16.5e-6)
    ito = MechanicalProps(E_Pa=115e9, nu=0.35, cte_per_K=6e-6, sigma_crit_Pa=0.3e9)
    assert float(biaxial_stress_Pa(cu, 2.6e-6, 0.0)) == 0.0
    assert np.isinf(coffin_manson_nf(0.0))
    assert cycles_to_failure(ito, 2.6e-6, 600.0) == 0.0                 # brittle over-stress
    assert 0.0 < cycles_to_failure(cu, 2.6e-6, 600.0) < np.inf          # ductile finite
    assert float(brittle_survival(0.0, sigma0_Pa=1e9, m_weibull=8.0)) == 1.0
    assert norris_landzberg_af(f_use_Hz=1e-4, f_test_Hz=1e-4, dT_use_K=50.0, dT_test_K=50.0,
                               Tmax_use_K=358.0, Tmax_test_K=358.0) == pytest.approx(1.0)
    # audit C4-1 direction pin: from Nf ~ f^(1/3), a test differing ONLY by cycling 8x
    # FASTER is less damaging per cycle, so AF = (1/8)^(1/3) = 0.5 exactly (the pre-audit
    # inverted ratio returned 2.0 -- non-conservative field-life inflation)
    assert norris_landzberg_af(f_use_Hz=1e-4, f_test_Hz=8e-4, dT_use_K=50.0, dT_test_K=50.0,
                               Tmax_use_K=358.0, Tmax_test_K=358.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        MechanicalProps(E_Pa=110e9, nu=0.6, cte_per_K=1e-6)             # Poisson out of range
    with pytest.raises(ValueError):
        coffin_manson_nf(1e-3, c_ductility=1.5)                         # c in (0, 1]


# ---- REL7 stress migration ----

def test_stressmig_relax_and_thresholds():
    x = np.linspace(0.0, 10e-6, 101)
    sig = korhonen_relax(x, 1.0, sigma0_Pa=1e8, kappa_m2_s=1e-13)
    assert sig[0] == pytest.approx(0.0, abs=1.0) and sig[-1] == pytest.approx(1e8, rel=1e-3)
    assert not void_nucleates(sig, float("inf"))
    assert void_nucleates(sig, 0.5e8)
    assert korhonen_kappa_m2_s(400.0, D0_m2_s=1e-6, Q_eV=0.9, B_Pa=1e11, Omega_m3=1.18e-29) > 0.0
    assert soret_flux_per_m2_s(8e28, 350.0, 1e7, D_a_m2_s=1e-18, Qstar_eV=0.8) < 0.0


# ---- REL8 HCI ----

def test_hci_limits_and_q_dimension():
    assert np.isinf(hci_time_to_failure_s(0.0, 300.0, C_s=1.0, width_m=1e-6))
    rate = trap_generation_rate_per_m2_s(5e-6, 1e-6, 2e-6, A_it=1e-3)
    assert rate * 1.602176634e-19 * 1e-6 * 2e-6 == pytest.approx(1e-3 * 5e-6, rel=1e-12)
    # negative Ea (the physical HCI quirk) accepted; worse cold
    assert (hci_time_to_failure_s(1e-6, 250.0, C_s=1.0, width_m=1e-6, Ea_eV=-0.1)
            < hci_time_to_failure_s(1e-6, 350.0, C_s=1.0, width_m=1e-6, Ea_eV=-0.1))


# ---- REL9 corrosion ----

def test_corrosion_limits_and_peck():
    assert float(deal_grove_thickness_m(0.0, A_m=50e-9, B_m2_s=1e-19, x0_m=2e-9)) == pytest.approx(2e-9)
    x = deal_grove_thickness_m(np.array([0.0, 1e3, 1e6]), A_m=50e-9, B_m2_s=1e-19)
    assert np.all(np.diff(x) > 0.0)                                     # oxide only grows
    assert peck_af(RH_use=40.0, RH_stress=85.0, T_use_K=298.15, T_stress_K=358.15) > 10.0
    with pytest.raises(ValueError):
        peck_time_to_failure_s(0.0, 300.0, A_s=1.0)                     # RH in (0, 100]
    with pytest.raises(ValueError):
        peck_time_to_failure_s(120.0, 300.0, A_s=1.0)


# ---- driver D3: MechanicalProps promoted onto the Material schema ----

def test_mechanical_props_on_material_schema():
    from dynameta.materials import Material, ConstantOptical, MechanicalProps as MatMech
    from dynameta.reliability import MechanicalProps as RelMech
    assert MatMech is RelMech                                # one class, re-exported (back-compat)
    m_plain = Material("oxide", ConstantOptical(4.0 + 0j))
    assert m_plain.mechanical is None                        # default None = byte-identical
    mech = MatMech(E_Pa=70e9, nu=0.17, cte_per_K=0.5e-6, sigma_crit_Pa=0.8e9)
    m = Material("oxide2", ConstantOptical(4.0 + 0j), mechanical=mech)
    assert m.mechanical.E_Pa == 70e9 and np.isfinite(m.mechanical.sigma_crit_Pa)
    assert biaxial_stress_Pa(m.mechanical, 2.6e-6, 100.0) != 0.0   # consumable by the REL6 functions


# ---- R16: gate-oxide tunneling leakage --------------------------------------------------------

def test_leakage_fn_linearity_and_anchor():
    from dynameta.reliability.leakage import fn_coefficients, fowler_nordheim_current
    a_fn, b_fn = fn_coefficients(0.42, 3.1)
    assert 2.3e10 <= b_fn <= 2.6e10                      # SiO2 literature band [V/m]
    E1, E2 = 6e8, 1.2e9
    r = fowler_nordheim_current(E2) / fowler_nordheim_current(E1)
    assert r == pytest.approx((E2 / E1) ** 2 * np.exp(-b_fn * (1 / E2 - 1 / E1)), rel=1e-12)
    assert fowler_nordheim_current(0.0) == 0.0


def test_leakage_dt_joins_fn_exactly_and_off_switch():
    from dynameta.reliability.leakage import (OxideLeakageParams, direct_tunneling_current,
                                              fowler_nordheim_current)
    t = 3e-9
    assert direct_tunneling_current(3.1, t) == fowler_nordheim_current(3.1 / t)
    off = OxideLeakageParams(t_ox_m=t)
    assert off.leakage_J_A_m2(2.0) == 0.0
    on = OxideLeakageParams(t_ox_m=t, enabled=True)
    assert on.joule_W_m3(2.0) == on.leakage_J_A_m2(2.0) * 2.0 / t > 0.0
    with pytest.raises(ValueError):
        OxideLeakageParams(t_ox_m=0.0)
