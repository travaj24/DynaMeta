"""Small shared numeric helpers (pure numpy/scipy). Kept here (not in analysis.py) so the carrier
modules can reuse them without importing the analysis layer; pure, so no import-cycle risk."""

from __future__ import annotations

import numpy as np
from scipy.integrate import BDF


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


class ZeroInitBDF(BDF):
    """scipy's BDF with its backward-difference array fully initialised. Nothing else differs.

    scipy allocates that array as `D = np.empty((MAX_ORDER + 3, n))` and writes only D[0] = y0
    and D[1] = f h (scipy/integrate/_ivp/bdf.py:249). On the FIRST accepted step the order is
    still 1, so its difference update

        D[order + 2] = d - D[order + 1]        # bdf.py:421 (scipy 1.18.1); :416 in 1.17

    READS D[2], which nobody ever wrote -- an uninitialised-memory read of n doubles, once per
    BDF instance. The VALUE is inert: D[order+2] is recomputed from a real D[order+1] at the end
    of the second step, before the order-selection heuristic first looks at it, so no result
    depends on the garbage (verified bit-identical with D[2:] filled with 0, 1, -1e300, +inf and
    1e12*randn). The SUBTRACT is not inert. IEEE-754 makes `finite - x` signal INVALID when x is
    a SIGNALLING NaN, and about 1 random bit pattern in 4096 is one (11 exponent bits set, top
    mantissa bit clear, mantissa nonzero) -- so whenever the recycled heap block behind that
    np.empty happens to spell an sNaN, numpy raises "RuntimeWarning: invalid value encountered
    in subtract" from a line of scipy's arithmetic that has nothing to do with the caller's model.

    Under this repo's `filterwarnings = ["error"]` that warning is a test failure. Observed as a
    ~9% (1 in 11) flake of tests/test_soa.py::test_eh_numba_parity on the `numba kernels
    (windows, py3.12)` leg -- CI job 99382994280, 2026-08-31 -- where QDGainModel.steady_state
    runs two BDF legs over an n = 86 state, i.e. 172 uninitialised doubles per call -- 4.1% for
    uniformly random bytes, and 19% over the 14 BDF instances (868 doubles) the leg's six SOA
    tests build in one process. Those are UPPER bounds: a recycled heap is mostly zeros, pointers
    and real float64 data, none of which are sNaN, so the measured 9% sits below them, and the
    test that failed is the LAST of the six (most-churned heap). It is NOT a property of the
    QD-SOA rate equations, which the numbers above might otherwise be blamed on: that
    integration is a monotone relaxation to a fixed point, and 420 solves from seeds perturbed by
    1e-16 to 0.3 relative all landed on the same steady state with no non-finite value anywhere.
    The line number alone settles it -- reaching this update means the step was ACCEPTED, and a
    solution that really does go non-finite signals somewhere else entirely (dy/dt = y^3 from a
    huge seed warns first as "overflow encountered in dot", then a scalar divide; dy/dt = y^2
    integrated straight through its blow-up warns not at all, it just fails on step size).

    Zero is the right filling, not merely a safe one: the higher backward differences of a
    solution known at a single point ARE zero. Use this in place of method="BDF" library-wide;
    `resolve_ivp_method` does that for an API whose method arrives as a caller-supplied string.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.D[2:] = 0.0                                      # never read scipy's np.empty rows


def resolve_ivp_method(method):
    """Map a solve_ivp `method` to ZeroInitBDF when the caller asked for BDF; pass anything
    else through untouched. For the public APIs that take the method as a string (so the
    documented default stays the string "BDF") -- see ZeroInitBDF for what it fixes."""
    if isinstance(method, str) and method.strip().upper() == "BDF":
        return ZeroInitBDF
    return method
