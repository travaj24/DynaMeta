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
    sum and do not call numpy's (or scipy's) version directly. The claim is machine-checked
    by `tests/test_numerics.py::test_no_direct_numpy_trapezoid_in_library`, which walks the
    AST of every shipped module for the attribute form (`np.trapezoid`), the from-import form
    (`from numpy import trapezoid`), any ALIASED numpy import it reads out of the file's own
    import statements (`import numpy as onp; onp.trapezoid`), the `getattr(np, "trapezoid")`
    form, and the scipy spelling (`scipy.integrate.trapezoid`).

    EXCEPTIONS, exhaustive (the guard's allow-list; both are floor-correct as written):
      * `optics/spdc_design.py` keeps a `getattr(np, "trapezoid", np.trapz)` shim because it
        needs the 2-D `axis=` form this scalar helper does not offer.
      * `scipy.integrate.cumulative_trapezoid` (`fiber_amp/lma.py`) is a CUMULATIVE variant,
        not this scalar reduction, and scipy has carried that name since 1.6 -- well under the
        declared `scipy>=1.10` floor. Other cumulative variants live with their consumers
        (`fiber_amp/dynamics._cumtrapz` / `_cumtrapz2`).

    REAL ONLY, enforced. A complex integrand raises TypeError rather than being silently cast:
    the `dtype=np.float64` conversion below discards the imaginary part with a numpy
    ComplexWarning (which is easy to filter away or to miss), where `np.trapezoid` would have
    returned a complex result -- a silent halving of the information for any caller that
    reached here with a complex array (audit X-1). Integrate the real and imaginary parts
    separately if that is what you meant.

    Byte-note: `0.5*(y[i]+y[i+1])*dx` and numpy's `dx*(y[i]+y[i+1])/2.0` differ only by an
    EXACT power-of-two scaling, so this reproduces `np.trapezoid` bit-for-bit on normal
    float64 input.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    if np.iscomplexobj(y) or np.iscomplexobj(x):
        raise TypeError(
            "core.numerics.trapz integrates a REAL integrand; got a complex {} (dtype {}). "
            "Casting it to float64 would DISCARD the imaginary part (numpy ComplexWarning) "
            "where np.trapezoid returns a complex value -- audit X-1. Pass y.real / y.imag "
            "separately, or combine them yourself: trapz(y.real, x) + 1j*trapz(y.imag, x)."
            .format("y" if np.iscomplexobj(y) else "x",
                    (y if np.iscomplexobj(y) else x).dtype))
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))
