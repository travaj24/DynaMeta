"""Fast unit test for the unified modulator figure-of-merit helper (pure arithmetic, no solver)."""
import numpy as np

from dynameta.analysis import modulator_figure_of_merit


def test_modulator_figure_of_merit():
    C_area = 7.65e-3            # F/m^2
    V = 2.0
    rho_s = 100.0              # Ohm/sq
    path, pad = 5e-6, 1e-6
    cell = (370e-9) ** 2
    spec = modulator_figure_of_merit(
        optical_contrast=0.26, contrast_lambda_nm=1350.0, gate_C_per_area_F_m2=C_area,
        voltage_swing_V=V, sheet_resistance_ohm_sq=rho_s, path_length_m=path, pad_width_m=pad,
        cell_area_m2=cell)
    R = rho_s * path / pad
    C_cell = C_area * cell
    f3 = 1.0 / (2.0 * np.pi * R * C_cell)
    E_fj = 0.5 * C_area * V ** 2 * cell * 1e15
    assert abs(spec["R_access_ohm"] - R) < 1e-9
    assert abs(spec["gate_C_fF"] - C_cell * 1e15) < 1e-12 * C_cell * 1e15
    assert abs(spec["f_3dB_GHz"] - f3 * 1e-9) < 1e-9 * f3 * 1e-9
    assert abs(spec["switching_energy_fJ"] - E_fj) < 1e-9 * E_fj
    assert abs(spec["contrast_per_fJ"] - 0.26 / E_fj) < 1e-12
