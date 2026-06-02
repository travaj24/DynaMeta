"""Generalized-Einstein degeneracy factor g(x)=F_1/2(eta)/F_-1/2(eta) for Fermi-Dirac
drift-diffusion -- the SINGLE source of the rational-fit coefficients (shared by
physics_drift_diffusion and physics_bipolar_dd, which previously duplicated them) plus a
pure-numpy evaluator so the load-bearing fit can be unit-tested without DEVSIM.

  g(x) = 1 + (a x + c x^(4/3)) / (1 + b x^(1/3) + d x^(2/3)),   x = n/N_c (or carrier/N_dos).

The Boltzmann VALUE limit is exact (g(0)=1); the fit tracks the exact F_1/2/F_-1/2 ratio to
~1.1% peak and <0.5% across ITO's regime (eta>=10). The degenerate leading coefficient is NOT
anchored (c/d=0.69 vs the true (2/3)c1^(2/3)=0.81), so the fit is valid to ~eta=32 and
undershoots beyond. See physics_drift_diffusion.py for the full derivation. Pure numpy, no
devsim. (audit F1 / DD-1 / DD-2 / DD-3)
"""

from __future__ import annotations

import numpy as np

# Least-squares fit to the exact Fermi-Dirac ratio F_1/2/F_-1/2 (the audited values).
GA, GB, GC, GD = 0.33717, 0.13356, 0.14143, 0.20570
_P13, _P23, _P43 = 1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0


def g_einstein(x):
    """Generalized-Einstein factor g(x) for x = n/N_c >= 0 (array-broadcasting); g(0)=1."""
    x = np.asarray(x, dtype=np.float64)
    return 1.0 + (GA * x + GC * np.power(x, _P43)) / (
        1.0 + GB * np.power(x, _P13) + GD * np.power(x, _P23))
