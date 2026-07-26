"""Small shared numeric helpers (pure numpy). Kept here (not in analysis.py) so the carrier
modules can reuse them without importing the analysis layer; pure, so no import-cycle risk."""

from __future__ import annotations

import numpy as np


def trapz(y, x) -> float:
    """Trapezoidal integral of a REAL 1-D y over x -- the library's floor-safe trapezoid.

    NEITHER numpy spelling works across the declared `numpy>=1.24` floor: `np.trapz` was
    REMOVED in NumPy 2.x, and `np.trapezoid` does not exist before 2.0 (the identifier is
    absent from the 1.24 wheel entirely). Calling either directly therefore breaks on one
    side of the floor -- audit X-1, which found 8 such `np.trapezoid` sites in
    `carriers/density_gradient`, `fiber_amp/eryb` and `fiber_amp/nonlinear_limits`; they
    now route here. This is the ONE home for the scalar case: import it, do not re-roll the
    sum and do not call numpy's version directly.

    Sole documented exception: `optics/spdc_design.py` keeps a `getattr(np, "trapezoid",
    np.trapz)` shim because it needs the 2-D `axis=` form this scalar helper does not offer
    (that shim is itself floor-correct on both sides). Cumulative variants live with their
    consumers (`fiber_amp/dynamics._cumtrapz`).

    Byte-note: `0.5*(y[i]+y[i+1])*dx` and numpy's `dx*(y[i]+y[i+1])/2.0` differ only by an
    EXACT power-of-two scaling, so this reproduces `np.trapezoid` bit-for-bit on normal
    float64 input.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))
