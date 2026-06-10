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


# ---- R21: k(T) Kirchhoff transform ------------------------------------------------------------

def test_kirchhoff_constant_k_matches_linear_solver():
    from dynameta.carriers.thermal_fem import solve_thermal_fem, solve_thermal_kirchhoff_fem
    lays = [ThermalLayer("s", 100e-9, 5.0)]
    rl = solve_thermal_fem(lays, period_x_m=1e-7, period_y_m=1e-7, flux_W_m2=1e8,
                           T_sink_K=300.0, maxh_m=25e-9)
    rk = solve_thermal_kirchhoff_fem(lays, lambda T: 5.0, period_x_m=1e-7, period_y_m=1e-7,
                                     flux_W_m2=1e8, T_sink_K=300.0, maxh_m=25e-9)
    assert abs(rk.T_at(5e-8, 5e-8, 9.9e-8) - rl.T_at(5e-8, 5e-8, 9.9e-8)) < 1e-9


def test_kirchhoff_roundtrip_and_guards():
    from dynameta.carriers.thermal_fem import (kirchhoff_theta, invert_kirchhoff,
                                               solve_thermal_kirchhoff_fem)
    k = lambda T: 148.0 * (T / 300.0) ** (-1.3)
    assert abs(invert_kirchhoff(k, kirchhoff_theta(k, 600.0, 300.0), 300.0) - 600.0) < 1e-8
    assert invert_kirchhoff(k, 0.0, 300.0) == 300.0
    with pytest.raises(ValueError):
        solve_thermal_kirchhoff_fem([ThermalLayer("s", 1e-7, 1.0)], lambda T: 0.0,
                                    period_x_m=1e-7, period_y_m=1e-7)


# ---- R21 follow-on: per-layer k(T) exact 1D + transient k(T(x)) -------------------------------

def test_kirchhoff_layered_1d_constant_k_exact():
    from dynameta.carriers.thermal import steady_layered_temperature
    from dynameta.carriers.thermal_fem import solve_thermal_kirchhoff_layered_1d
    lays = [ThermalLayer("a", 100e-9, 140.0), ThermalLayer("b", 60e-9, 1.4)]
    r = solve_thermal_kirchhoff_layered_1d(lays, {"a": lambda T: 140.0, "b": lambda T: 1.4},
                                           flux_W_m2=1e8)
    ref = steady_layered_temperature([140.0, 1.4], [100e-9, 60e-9], 1e8, 300.0)
    mids = 0.5 * (r.interface_z_m[:-1] + r.interface_z_m[1:])
    assert max(abs(r.T_at_z(z) - t) for z, t in zip(mids, ref)) < 1e-9


def test_kirchhoff_layered_1d_guards_and_continuity():
    from dynameta.carriers.thermal_fem import solve_thermal_kirchhoff_layered_1d
    lays = [ThermalLayer("a", 100e-9, 8.0), ThermalLayer("b", 60e-9, 2.0)]
    k_by = {"a": lambda T: 8.0 * (300.0 / T) ** 1.3, "b": lambda T: 2.0}
    r = solve_thermal_kirchhoff_layered_1d(lays, k_by, flux_W_m2=5e8)
    assert r.interface_T_K[0] == 300.0
    assert np.all(np.diff(r.interface_T_K) > 0)          # heating toward the top
    # T continuity at the interface: the probes sit 2h apart across a real ~j/k gradient
    # (~2.5e8 K/m here), so bound by gradient*2h + the brentq inversion tolerance
    h = 1e-13
    bound = (5e8 / 2.0) * 2 * h * 3.0 + 1e-7
    assert abs(r.T_at_z(100e-9 - h) - r.T_at_z(100e-9 + h)) < bound
    with pytest.raises(ValueError):
        solve_thermal_kirchhoff_layered_1d(lays, {"a": lambda T: 8.0})   # missing coverage
