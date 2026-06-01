"""
Spectral analysis helpers for reflection / transmission sweeps.

Generic post-processing utilities for the (wavelength, |r|^2) spectra that
Stage 3 produces -- not tied to any particular design.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


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
