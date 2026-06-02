"""Small shared numeric helpers (pure numpy). Kept here (not in analysis.py) so the carrier
modules can reuse them without importing the analysis layer; pure, so no import-cycle risk."""

from __future__ import annotations

import numpy as np


def trapz(y, x) -> float:
    """Trapezoidal integral of y over x. (np.trapz was removed in NumPy 2.x, so the library
    uses this single implementation everywhere instead of re-rolling the sum.)"""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))
