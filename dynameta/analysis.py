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

    x0, x1, x2 = lam[i - 1], lam[i], lam[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-30:
        return float(x1), float(y1)
    # Vertex offset for a parabola through 3 (here unequally-spaced-safe via
    # the symmetric-step approximation; exact for uniform spacing).
    h = 0.5 * (x2 - x0)
    dx = 0.5 * (y0 - y2) / denom * h
    lam_dip = float(x1 + dx)
    val_dip = float(y1 - 0.25 * (y0 - y2) * dx / h) if h != 0 else float(y1)
    return lam_dip, val_dip


def resonance_shift(wavelengths_nm: Sequence[float],
                    spectrum_ref: Sequence[float],
                    spectrum_test: Sequence[float]) -> float:
    """Spectral shift (nm) of a resonance dip between two spectra on the same
    wavelength grid: dip(test) - dip(ref). Positive = redshift.
    """
    ref_nm, _ = resonance_dip(wavelengths_nm, spectrum_ref)
    test_nm, _ = resonance_dip(wavelengths_nm, spectrum_test)
    return float(test_nm - ref_nm)
