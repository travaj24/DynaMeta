"""Smoke + guard tests for the spatially-resolved two-temperature FEM (thermal_fem, roadmap R14).
Skipped when ngsolve is absent (numpy-only CI leg); the rigorous oracle lives in
validation/thermal_fem_twotemp.py."""
import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.carriers.thermal_fem import (ThermalLayer, ThermalLayerTwoTemp,
                                           solve_thermal_twotemp_fem,
                                           solve_thermal_transient_twotemp_fem)


def _layer(**kw):
    p = dict(name="m", thickness_m=100e-9, k_thermal=1.0, rho_kg_m3=2.0e3, Cp_J_kgK=1.0e3,
             G_e_l=1.0e16, C_e_J_m3K=1.0e4, k_electron=10.0)
    p.update(kw)
    return ThermalLayerTwoTemp(**p)


def test_twotemp_insulated_uniform_matches_lumped_closed_form():
    C_e, C_l, G, S, T0 = 1.0e4, 2.0e6, 1.0e16, 1.0e18, 300.0
    r = solve_thermal_transient_twotemp_fem(
        [_layer()], period_x_m=100e-9, period_y_m=100e-9, t_end_s=2e-12, dt_s=2e-14,
        source_e_W_m3=S, T_sink_K=T0, theta=0.5, maxh_m=50e-9, bottom_bc="insulated",
        store_every=20)
    t = r.t_s
    lam = G * (1.0 / C_e + 1.0 / C_l)
    d = (S / (C_e * lam)) * (1.0 - np.exp(-lam * t))
    Te_cf = T0 + (S * t + C_l * d) / (C_e + C_l)
    rise = float(np.max(Te_cf - T0))
    assert np.max(np.abs(r.mean_Te_per_layer_t[:, 0] - Te_cf)) < 1e-3 * rise


def test_twotemp_g_zero_keeps_lattice_at_sink():
    r = solve_thermal_transient_twotemp_fem(
        [_layer(G_e_l=0.0)], period_x_m=100e-9, period_y_m=100e-9, t_end_s=1e-10, dt_s=5e-12,
        source_e_W_m3=1.0e15, T_sink_K=300.0, maxh_m=50e-9, store_every=5)
    assert np.max(np.abs(r.mean_Tl_per_layer_t - 300.0)) < 1e-8
    assert r.mean_Te_per_layer_t[-1, 0] > 300.0 + 1e-3


def test_twotemp_steady_electron_hotter_and_sink_pinned():
    r = solve_thermal_twotemp_fem([_layer(G_e_l=1.0e12, thickness_m=1e-6)], period_x_m=100e-9,
                                  period_y_m=100e-9, source_e_W_m3=1.0e15, T_sink_K=300.0,
                                  maxh_m=100e-9)
    assert r.Te_at(50e-9, 50e-9, 0.999e-6) > r.Tl_at(50e-9, 50e-9, 0.999e-6)  # source in e-channel
    assert abs(r.Te_at(50e-9, 50e-9, 0.0) - 300.0) < 1e-9                      # Dirichlet sink
    assert abs(r.Tl_at(50e-9, 50e-9, 0.0) - 300.0) < 1e-9


def test_twotemp_guards():
    with pytest.raises(ValueError):                            # C_e missing
        solve_thermal_twotemp_fem([_layer(C_e_J_m3K=0.0)], period_x_m=1e-7, period_y_m=1e-7)
    with pytest.raises(ValueError):                            # negative coupling
        solve_thermal_twotemp_fem([_layer(G_e_l=-1.0)], period_x_m=1e-7, period_y_m=1e-7)
    with pytest.raises(ValueError):                            # plain layer rejected
        solve_thermal_twotemp_fem([ThermalLayer("m", 1e-7, 1.0)], period_x_m=1e-7,
                                  period_y_m=1e-7)
    with pytest.raises(ValueError):                            # bad bottom_bc
        solve_thermal_transient_twotemp_fem([_layer()], period_x_m=1e-7, period_y_m=1e-7,
                                            t_end_s=1e-12, dt_s=1e-13, bottom_bc="open")
    with pytest.raises(ValueError):                            # lattice C_l missing
        solve_thermal_transient_twotemp_fem([_layer(rho_kg_m3=0.0)], period_x_m=1e-7,
                                            period_y_m=1e-7, t_end_s=1e-12, dt_s=1e-13)
