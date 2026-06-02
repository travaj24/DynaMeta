"""
FieldLift: reconstruct a 3D carrier field n(x, y, z) from a lower-dimensional
DEVSIM solve. This is where the 2D-DEVSIM -> 3D-optics step is made explicit
(it used to be a hidden branch + the xy-product inside eps_loader). Lifting the
CARRIER DENSITY (not the eps) means the background automatically maps to the
correct n_bg -> eps via the NToEpsMap, fixing the old grid-corner background bug.

Lifts operate on n_2d shaped (Nx_lateral, Nv_vertical):
  IdentityLift     : source already 3D (native 3D DEVSIM) -> pass through
  ExtrudeLift      : invariant along the 2nd lateral axis (1D gratings) -> repeat
  SeparableXYLift  : xy-product reconstruction; VALID ONLY for a centered,
                      4-fold (c4v) device on a SQUARE cell with a SINGLE-SIGN
                      lateral deviation. apply() now enforces the square-cell and
                      single-sign preconditions (a sign-changing lateral profile
                      makes the outer product spuriously positive in the
                      (neg)x(neg) quadrant); evenness/centering remain the
                      caller's responsibility.

`choose_lift` picks the lift from the device symmetry string (the design path
also enforces a square cell transitively via the unit-cell lattice symmetry), so
a wrong lift is rejected at orchestration time; the apply()-level guards below
are the backstop for direct construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class FieldLift:
    def apply(self, n_2d, x_m, v_m, *, n_bg):
        """Return (n_3d (Nx,Ny,Nz), x_m (Nx), y_m (Ny), z_m (Nz=Nv))."""
        raise NotImplementedError


@dataclass
class IdentityLift(FieldLift):
    """For a native 3D carrier field already shaped (Nx, Ny, Nz). n_2d here is
    actually the 3D array; axes are passed straight through."""

    def apply(self, n_3d, x_m, v_m, *, n_bg):
        arr = np.asarray(n_3d, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError("IdentityLift expects a 3D carrier array")
        ny = arr.shape[1]
        y_m = np.linspace(0.0, 1.0, ny)   # caller overrides axes if needed
        return arr, np.asarray(x_m), y_m, np.asarray(v_m)


@dataclass
class ExtrudeLift(FieldLift):
    """Invariant along the synthesized 2nd lateral (y) axis. Honest only for a
    y-translationally-invariant device (e.g. a 1D grating)."""
    period_y_m: float

    def apply(self, n_2d, x_m, v_m, *, n_bg):
        n_2d = np.asarray(n_2d, dtype=np.float64)        # (Nx, Nv)
        n_3d = np.repeat(n_2d[:, None, :], 2, axis=1)    # (Nx, 2, Nv)
        y_m = np.array([0.0, self.period_y_m], dtype=np.float64)
        return n_3d, np.asarray(x_m, dtype=np.float64), y_m, np.asarray(v_m, dtype=np.float64)


@dataclass
class SeparableXYLift(FieldLift):
    """xy-product reconstruction of the carrier-deviation from n_bg. Valid only
    for a centered, 4-fold-symmetric (c4v) device on a square cell with a
    single-sign lateral deviation (see module docstring)."""
    period_y_m: float
    ny: int = 256

    def apply(self, n_2d, x_m, v_m, *, n_bg):
        n_2d = np.asarray(n_2d, dtype=np.float64)        # (Nx, Nv)
        x_m = np.asarray(x_m, dtype=np.float64)
        v_m = np.asarray(v_m, dtype=np.float64)
        # Precondition 1: SQUARE cell -- the x-profile is reused as the y-profile,
        # so the lateral x-span must match the synthesized y-period.
        x_span = float(x_m[-1] - x_m[0])
        if self.period_y_m > 0 and abs(x_span - self.period_y_m) > 0.02 * self.period_y_m:
            raise ValueError(
                "SeparableXYLift requires a square cell: carrier x-span {:.4g} m != "
                "period_y {:.4g} m. Use ExtrudeLift (y-invariant) or a 3D carrier "
                "solve.".format(x_span, self.period_y_m))
        dn = n_2d - n_bg                                  # deviation (Nx, Nv)
        # Precondition 2: SINGLE-SIGN lateral deviation. The outer product makes a
        # (neg)x(neg) quadrant spuriously positive, so a profile that both
        # accumulates and depletes laterally cannot be separably reconstructed.
        peak = float(np.max(np.abs(dn))) if dn.size else 0.0
        if peak > 0:
            tol = 1e-3 * peak
            if dn.max() > tol and dn.min() < -tol:
                raise ValueError(
                    "SeparableXYLift requires a single-sign lateral carrier deviation "
                    "(found both +{:.3g} and {:.3g} about n_bg); the xy outer product "
                    "would inject a spurious positive (neg)x(neg) quadrant. Use a 3D "
                    "carrier solve for a sign-changing lateral profile.".format(
                        dn.max(), dn.min()))
        y_m = np.linspace(0.0, self.period_y_m, self.ny)
        # y-profile from the x-profile (separable): interp dn(x) onto the y grid
        dn_y = np.empty((self.ny, dn.shape[1]), dtype=np.float64)
        for k in range(dn.shape[1]):
            dn_y[:, k] = np.interp(y_m, x_m, dn[:, k])
        # peak deviation per vertical level (avoid divide-by-zero)
        dn_peak = np.empty(dn.shape[1], dtype=np.float64)
        for k in range(dn.shape[1]):
            idx = int(np.argmax(np.abs(dn[:, k])))
            dn_peak[k] = dn[idx, k] if abs(dn[idx, k]) > 0 else 1.0
        dn_3d = np.einsum("ik,jk,k->ijk", dn, dn_y, 1.0 / dn_peak)   # (Nx, Ny, Nv)
        n_3d = n_bg + dn_3d
        return n_3d, x_m, y_m, v_m


def choose_lift(device_symmetry: str, setting: str, *,
                  period_y_m: float, ny: int = 256) -> FieldLift:
    """Pick + validate the lift. setting in {auto, separable_xy, extrude, identity}."""
    if setting == "identity":
        return IdentityLift()
    if setting in ("separable_xy",) or (setting == "auto" and device_symmetry == "c4v"):
        if device_symmetry != "c4v":
            raise ValueError(
                "SeparableXYLift requires c4v device symmetry; device is '{}'. "
                "Use lift='extrude' (y-invariant) or a 3D carrier solve.".format(
                    device_symmetry))
        return SeparableXYLift(period_y_m=period_y_m, ny=ny)
    if setting in ("extrude", "auto"):
        return ExtrudeLift(period_y_m=period_y_m)
    raise ValueError("unknown lift setting {!r}".format(setting))
