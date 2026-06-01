"""
Spectral analysis helpers for reflection / transmission sweeps.

Generic post-processing utilities for the (wavelength, |r|^2) spectra that
Stage 3 produces -- not tied to any particular design.
"""

from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple

import numpy as np

from dynameta.core.carrier_field import ELECTRON_DENSITY

_Q_E = 1.602176634e-19   # C


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral (np.trapz was removed in NumPy 2.x)."""
    y = np.asarray(y, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def gate_cv(carrier_fields: Sequence, region: str, *, voltage_key: str,
            density_field: str = ELECTRON_DENSITY):
    """DC gate charge Q(Vg) and differential capacitance C = dQ/dVg from a VOLTAGE SWEEP
    of CarrierFields (the per-bias fields run_pipeline / a Sweep already produces -- NO new
    solver). For an n-type accumulation gate the mirror gate charge per unit cell area is
    Q(Vg) = q * INT (n(z) - n_bg) dz (the accumulated excess electrons), laterally averaged;
    C(Vg) = dQ/dVg is the small-signal gate capacitance and is the first dynamic-adjacent
    figure of merit (with an access/sheet resistance it sets the RC modulation bandwidth).

    Args:
      carrier_fields : iterable of CarrierField at different gate biases (same device).
      region         : the semiconductor region key.
      voltage_key    : the electrode-name key into CarrierField.voltages for the gate bias.

    Returns (Vg, Q, Vmid, C): Vg (V, sorted) and Q (C/m^2) per bias; Vmid (V) and
    C (F/m^2) on the midpoints (central differences need >= 2 biases).

    Note: Q is the EXCESS sheet charge over the flat doping n_bg. For a
    Schrodinger-Poisson carrier the hard-wall dead layer makes n < n_bg near the body
    contact at every bias, so the absolute Q(0) is slightly negative and only the
    DIFFERENTIAL C = dQ/dVg (in which that constant offset cancels) is physically
    meaningful for SP fields; a classical/DD carrier gives Q(0) ~ 0.
    """
    Vg: List[float] = []
    Q: List[float] = []
    for cf in carrier_fields:
        reg = cf.regions[region]
        if reg.grid_fields is None or density_field not in reg.grid_fields:
            raise ValueError("region '{}' has no gridded '{}'".format(region, density_field))
        n = np.asarray(reg.grid_fields[density_field], dtype=np.float64)
        n_bg = float(cf.n_bg_by_region[region])
        if n.ndim == 3:                                   # (Nx, Ny, Nz) -> n(z)
            zc = np.asarray(reg.grid_axes_m["z"], dtype=np.float64)
            prof = n.mean(axis=(0, 1))
        elif n.ndim == 2:                                 # (Nx, Nv) -> n(v); 2D vertical axis = 'y'
            zc = np.asarray(reg.grid_axes_m["y"], dtype=np.float64)
            prof = n.mean(axis=0)
        else:
            raise ValueError("density grid must be 2D or 3D")
        Q.append(_Q_E * _trapz(prof - n_bg, zc))          # C/m^2 (excess electron sheet charge)
        Vg.append(float(cf.voltages[voltage_key]))
    Vg_a = np.asarray(Vg, dtype=np.float64)
    Q_a = np.asarray(Q, dtype=np.float64)
    order = np.argsort(Vg_a)
    Vg_a, Q_a = Vg_a[order], Q_a[order]
    if Vg_a.size < 2:
        return Vg_a, Q_a, np.array([]), np.array([])
    if np.any(np.diff(Vg_a) == 0.0):                  # coincident biases -> dQ/dVg = 0/0 = NaN
        dup = np.unique(Vg_a[:-1][np.diff(Vg_a) == 0.0]).tolist()
        raise ValueError("gate_cv: duplicate gate-bias point(s) {} V -> dQ/dVg undefined; "
                         "supply one CarrierField per distinct voltage".format(dup))
    Vmid = 0.5 * (Vg_a[1:] + Vg_a[:-1])
    C = np.diff(Q_a) / np.diff(Vg_a)
    return Vg_a, Q_a, Vmid, C


def sheet_resistance_ohm_sq(n_m3, mobility_m2Vs, thickness_m):
    """Sheet resistance [Ohm/sq] of a conductive layer: rho_s = 1/(q n mu t)."""
    sigma = _Q_E * np.asarray(n_m3, dtype=np.float64) * float(mobility_m2Vs)   # S/m
    return 1.0 / (sigma * float(thickness_m))


def lumped_rc_bandwidth(C_F_per_m2, sheet_resistance_ohm_sq, *,
                        path_length_m, pad_width_m, cell_area_m2):
    """Intrinsic single-cell electrical RC 3-dB bandwidth of a gated modulator cell
    (ported from the Metasurface_Modulator Stage-4 lumped model).

    The gate charges through the in-plane ACCESS resistance
    R_access = rho_sheet * (path_length / pad_width) into the per-cell capacitance
    C_cell = C_per_area * cell_area, so f_3dB = 1 / (2 pi R_access C_cell). Pair it with
    gate_cv()'s differential capacitance (which may be an array over bias) for f_3dB(Vg).
    The access path geometry is a MODELING CHOICE -- sweep a few plausible (path, pad).

    Returns (R_access_ohm, C_cell_F, f_3dB_Hz), each broadcasting over C_F_per_m2.
    """
    C_per = np.asarray(C_F_per_m2, dtype=np.float64)
    if np.any(C_per <= 0.0):
        warnings.warn("lumped_rc_bandwidth: non-positive capacitance encountered "
                      "(depletion branch?); f_3dB is NaN there. Pass the accumulation-"
                      "branch (C>0) capacitance for a physical bandwidth.", stacklevel=2)
    R = float(sheet_resistance_ohm_sq) * float(path_length_m) / float(pad_width_m)
    C_cell = C_per * float(cell_area_m2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f3db = 1.0 / (2.0 * np.pi * R * C_cell)
    if np.ndim(f3db) == 0:                            # preserve scalar-in -> scalar-out
        f3db = float(f3db) if C_cell > 0.0 else float("nan")
    else:
        f3db = np.where(C_cell > 0.0, f3db, np.nan)
    return R, C_cell, f3db


def switching_energy_per_area(C_F_per_m2, voltage_swing_V):
    """Dynamic switching energy per unit area of a capacitive gate: E = 0.5 C V^2 [J/m^2]
    (C the gate areal capacitance, V the full drive swing). x cell_area -> J/event."""
    return 0.5 * np.asarray(C_F_per_m2, dtype=np.float64) * float(voltage_swing_V) ** 2


def resonance_dip(wavelengths_nm: Sequence[float],
                  spectrum: Sequence[float]) -> Tuple[float, float]:
    """Locate a resonance dip (minimum) in a spectrum with sub-grid accuracy.

    Fits a parabola to the three points around the discrete minimum and
    returns the interpolated (wavelength, value) of the vertex. This recovers
    the dip position to better than the wavelength sampling step -- important
    when comparing small bias-induced resonance shifts that are a fraction of
    the scan spacing.

    Args:
      wavelengths_nm : 1D array of wavelengths (need not be sorted)
      spectrum       : 1D array of the quantity to minimize (e.g. |r|^2),
                       same length as wavelengths_nm

    Returns:
      (dip_wavelength_nm, dip_value). Falls back to the discrete minimum if
      there are < 3 points or the minimum is at an array edge (no parabola).

    Example:
      lam = [1250, 1300, 1350, 1400]; R = [0.30, 0.05, 0.18, 0.40]
      dip_nm, dip_val = resonance_dip(lam, R)   # ~1297 nm, ~0.04
    """
    lam = np.asarray(wavelengths_nm, dtype=np.float64).ravel()
    y = np.asarray(spectrum, dtype=np.float64).ravel()
    if lam.shape != y.shape:
        raise ValueError("wavelengths_nm and spectrum must have equal length")
    finite = np.isfinite(lam) & np.isfinite(y)
    if not finite.any():
        raise ValueError("spectrum has no finite values to locate a dip")
    if not finite.all():
        # a NaN/inf sample (e.g. a failed solve point) must not hijack argmin or the
        # parabolic fit; locate the dip among the finite samples only.
        lam, y = lam[finite], y[finite]
    order = np.argsort(lam)
    lam, y = lam[order], y[order]

    i = int(np.argmin(y))
    if len(lam) < 3 or i == 0 or i == len(lam) - 1:
        return float(lam[i]), float(y[i])

    xs = lam[i - 1:i + 2]
    ys = y[i - 1:i + 2]
    # Exact parabola through the 3 (possibly unequally-spaced) points: y = a x^2
    # + b x + c. np.polyfit on exactly 3 points is the exact interpolant; the
    # vertex is -b/2a. (The old symmetric-step formula was exact only on a
    # uniform grid -- it biased the dip by up to a full step on a non-uniform one.)
    a, b, c = np.polyfit(xs, ys, 2)
    if a <= 1e-30:                       # not an upward parabola -> no interior min
        return float(lam[i]), float(y[i])
    lam_dip = -b / (2.0 * a)
    if not (xs[0] <= lam_dip <= xs[-1]):  # vertex outside the bracket -> fall back
        return float(lam[i]), float(y[i])
    val_dip = a * lam_dip ** 2 + b * lam_dip + c
    return float(lam_dip), float(val_dip)


def resonance_shift(wavelengths_nm: Sequence[float],
                    spectrum_ref: Sequence[float],
                    spectrum_test: Sequence[float]) -> float:
    """Spectral shift (nm) of a resonance dip between two spectra on the same
    wavelength grid: dip(test) - dip(ref). Positive = redshift.
    """
    ref_nm, _ = resonance_dip(wavelengths_nm, spectrum_ref)
    test_nm, _ = resonance_dip(wavelengths_nm, spectrum_test)
    return float(test_nm - ref_nm)
