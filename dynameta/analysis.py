"""
Spectral analysis helpers for reflection / transmission sweeps.

Generic post-processing utilities for the (wavelength, |r|^2) spectra that
Stage 3 produces -- not tied to any particular design.
"""

from __future__ import annotations

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
    Vmid = 0.5 * (Vg_a[1:] + Vg_a[:-1])
    C = np.diff(Q_a) / np.diff(Vg_a)
    return Vg_a, Q_a, Vmid, C


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
        return float(x1), float(y1)
    lam_dip = -b / (2.0 * a)
    if not (xs[0] <= lam_dip <= xs[-1]):  # vertex outside the bracket -> fall back
        return float(x1), float(y1)
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
