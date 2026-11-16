"""Lateral structure of a gridded permittivity: the ONE test of lateral uniformity and the ONE lateral-axis
guard every optical route uses (one implementation, so the routes cannot disagree on what is uniform).

Why one helper. Each optical route used to decide for itself whether a gridded EpsField is laterally uniform,
with different references and tolerances (the slicer compared with the first lateral sample against the
slice's max |eps|; the bridges used np.allclose against the first sample, relative to that sample; the 2-D
finite-element adapter kept the lateral axis whenever it had more than one sample), and only the Fourier and
modal bridges checked the unit and span of a lateral axis. The same laterally uniform carrier film -- which the
bridge always delivers with 256 identical lateral samples -- was then 1-D on one route, structured on another
and refused on a third. Every route now asks this module.

THE RULE. Along the lateral axes asked about, every sample of a z slice must equal the FIRST lateral sample of
that slice to within LATERAL_UNIFORM_RTOL times the slice's largest |eps| (all tensor components included).
The first sample is exact for identical samples (the lateral MEAN is not: over 256 x 256 samples numpy's mean
carries ~1.75e-12 of summation round-off, measured); the slice's max |eps| rather than each sample's own |eps|
keeps solver round-off near a zero of eps (|eps| ~ 0.27 inside an accumulation layer) from reading as
structure; 1e-12 keeps any physical grating (relative modulation far above round-off) structured.

Conventions: values_zyx is (Nz, Ny, Nx) scalar or (Nz, Ny, Nx, 3, 3) tensor (the EpsField layout); a lateral
cell is (Sx, Sy) or (Sx, Sy, 3, 3) (the layered-slab layout). Pure numpy.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = ["LATERAL_UNIFORM_RTOL", "uniform_along", "is_laterally_uniform", "is_y_invariant",
           "is_x_invariant", "is_uniform_cell", "tensor_couples_y", "check_periodic_axis",
           "lateral_axis_m"]

#: Relative tolerance of the lateral-uniformity test (against the z slice's largest |eps|).
LATERAL_UNIFORM_RTOL = 1e-12


def uniform_along(values_zyx, axes: Iterable[int], *, rtol: float = LATERAL_UNIFORM_RTOL) -> bool:
    """True when `values_zyx` does not vary along the lateral `axes` (1 = y, 2 = x) of every z slice,
    by THE RULE of the module docstring."""
    v = np.asarray(values_zyx)
    if v.ndim not in (3, 5) or (v.ndim == 5 and v.shape[-2:] != (3, 3)):
        raise ValueError("lateral uniformity: values_zyx must be (Nz, Ny, Nx) or (Nz, Ny, Nx, 3, 3); got "
                         "shape {}".format(v.shape))
    axes = tuple(sorted(set(int(a) for a in axes)))
    if any(a not in (1, 2) for a in axes):
        raise ValueError("lateral uniformity: axes must be 1 (y) and/or 2 (x), got {}".format(axes))
    if v.shape[0] == 0:
        return True
    ref = v[tuple(slice(0, 1) if a in axes else slice(None) for a in range(v.ndim))]
    scale = np.max(np.abs(v).reshape(v.shape[0], -1), axis=1).reshape((-1,) + (1,) * (v.ndim - 1))
    return bool(np.all(np.abs(v - ref) <= float(rtol) * scale))


def is_laterally_uniform(values_zyx, *, rtol: float = LATERAL_UNIFORM_RTOL) -> bool:
    """No variation along x or y in any z slice."""
    return uniform_along(values_zyx, (1, 2), rtol=rtol)


def is_y_invariant(values_zyx, *, rtol: float = LATERAL_UNIFORM_RTOL) -> bool:
    """No variation along y (lines along y: a 1-D-periodic cell)."""
    return uniform_along(values_zyx, (1,), rtol=rtol)


def is_x_invariant(values_zyx, *, rtol: float = LATERAL_UNIFORM_RTOL) -> bool:
    """No variation along x."""
    return uniform_along(values_zyx, (2,), rtol=rtol)


def is_uniform_cell(cell, *, rtol: float = LATERAL_UNIFORM_RTOL) -> bool:
    """A lateral cell (Sx, Sy) or (Sx, Sy, 3, 3) that is constant across the cell (one z slice)."""
    c = np.asarray(cell)
    if c.ndim not in (2, 4):
        raise ValueError("lateral uniformity: a cell must be (Sx, Sy) or (Sx, Sy, 3, 3); got shape "
                         "{}".format(c.shape))
    return is_laterally_uniform(np.swapaxes(c, 0, 1)[None], rtol=rtol)


def tensor_couples_y(tensor, *, rtol: float = LATERAL_UNIFORM_RTOL) -> bool:
    """True when a (..., 3, 3) permittivity couples y to x or z (eps_xy, eps_yx, eps_yz or eps_zy nonzero
    beyond rtol * max(1, max |eps|)): the cell is then not a y-decoupled (TE / TM separable) problem."""
    t = np.asarray(tensor)
    if t.shape[-2:] != (3, 3):
        raise ValueError("tensor_couples_y: expected (..., 3, 3), got shape {}".format(t.shape))
    off = np.abs(t[..., [0, 1, 1, 2], [1, 0, 2, 1]])
    return bool(np.max(off, initial=0.0) > float(rtol) * max(1.0, float(np.max(np.abs(t), initial=0.0))))


def check_periodic_axis(x, period, *, where: str = "lateral profile") -> np.ndarray:
    """Validate the lateral sample positions `x` of one periodic cell [0, period] (any one length unit,
    the same for both) and return them clipped into [0, period]. Raises ValueError when the positions are
    not strictly increasing, lie outside the cell, span more than one period, or span less than half of it
    (every periodic sampling with >= 2 points spans at least half the period; less means the axis does not
    describe this cell -- most often a unit slip: metre-valued positions read in nm span ~1e-9 of the
    period, and a periodic interpolation would silently stretch the last sample across the rest of the
    cell). Needs >= 2 samples."""
    x = np.asarray(x, dtype=np.float64).ravel()
    P = float(period)
    if x.size < 2:
        raise ValueError("{}: a laterally structured profile needs >= 2 x samples".format(where))
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("{}: x positions must be strictly increasing".format(where))
    tol = 1e-9 * P
    if x[0] < -tol or x[-1] > P + tol:
        raise ValueError("{}: x positions [{:.6g}, {:.6g}] m lie outside the cell [0, {:.6g}] m "
                         "(check the axis unit)".format(where, x[0], x[-1], P))
    x = np.clip(x, 0.0, P)
    if x[-1] - x[0] > P * (1.0 + 1e-12):
        raise ValueError("{}: x positions span more than one period".format(where))
    span = float(x[-1] - x[0])
    if span < 0.5 * P:
        ratio = span / P
        hint = (" (ratio {:.3g}: the x axis looks like it is in METRES where nm are read)"
                .format(ratio) if 0 < ratio < 1e-6 else "")
        raise ValueError("{}: x samples span {:.6g} m of the {:.6g} m period (< half){}; sample "
                         "the profile across the whole cell".format(where, span, P, hint))
    return x


def lateral_axis_m(axis_u, period_m: float, metres_per_unit: float, *,
                   where: str = "lateral profile") -> np.ndarray:
    """An EpsField lateral axis (in solver units, e.g. nm) in metres, validated by check_periodic_axis
    against the cell period. Raises when the axis is missing."""
    if axis_u is None:
        raise ValueError("{}: a laterally structured gridded EpsField needs its lateral axis "
                         "(x_axis_u / y_axis_u, in solver units)".format(where))
    x = np.asarray(axis_u, dtype=np.float64).ravel() * float(metres_per_unit)
    return check_periodic_axis(x, period_m, where=where)
