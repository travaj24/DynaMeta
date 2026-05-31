"""
4-fold-symmetric (xy-product) extension of a 2D (x, z) carrier-density
field to a 3D (x, y, z) field for square-patch geometries.

Formula:
    dn(x, z)    = n(x, z) - n_bg
    dn_peak(z)  = max_over_x of dn(x, z)         (with sign preserved)
    dn_3D(x,y,z) = dn(x, z) * dn(y, z) / dn_peak(z)
    n_3D(x,y,z)  = n_bg + dn_3D(x, y, z)

Properties:
  * At patch center (P/2, P/2): dn_3D = dn_peak.                    (peak preserved)
  * At (0, P/2) or (P/2, 0): dn_3D = 0.                              (background outside patch in x or y)
  * Soft-square footprint matching the 1D x-profile in both axes.
  * Symmetric under x <-> y by construction; works for accumulation
    (dn > 0) AND depletion (dn < 0) without special-casing.

See Metasurface_Modulator/stage1_carriers/symmetrize_xy.py for the
ad-hoc implementation this replaces.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def symmetrize_n_xz_to_xyz(n_xz: np.ndarray,
                              x_axis_m: np.ndarray,
                              n_bg_m3: float,
                              y_axis_m_out: Optional[np.ndarray] = None,
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Build a 4-fold-symmetric 3D carrier field from a 2D (x, z) input.

    Args:
      n_xz          : (Nx, Nz)  carrier density on a regular x-z grid [m^-3]
      x_axis_m      : (Nx,)     regular x grid [m]
      n_bg_m3       : scalar    background carrier density [m^-3]
      y_axis_m_out  : optional (Ny,) y grid; defaults to x_axis_m (square cell)

    Returns:
      n_3d          : (Nx, Ny, Nz)  symmetric carrier density [m^-3]
      y_axis_used   : (Ny,)         the y grid used for n_3d
    """
    if n_xz.ndim != 2:
        raise ValueError("n_xz must be 2D (Nx, Nz)")
    Nx, Nz = n_xz.shape
    if x_axis_m.shape != (Nx,):
        raise ValueError("x_axis_m shape must be (Nx,)")
    if y_axis_m_out is None:
        y_axis_m_out = x_axis_m.copy()
    Ny = y_axis_m_out.size

    dn_xz = n_xz - n_bg_m3                          # (Nx, Nz)

    # Sample dn at the requested y axis by interpolation
    dn_yz = np.empty((Ny, Nz), dtype=np.float64)
    for k in range(Nz):
        dn_yz[:, k] = np.interp(y_axis_m_out, x_axis_m, dn_xz[:, k])

    # Per-z signed peak
    dn_peak = np.empty(Nz, dtype=np.float64)
    for k in range(Nz):
        idx_peak = int(np.argmax(np.abs(dn_xz[:, k])))
        dn_peak[k] = dn_xz[idx_peak, k]
    safe_peak = np.where(np.abs(dn_peak) > 0.0, dn_peak, 1.0)

    dn_3d = np.einsum("ik,jk,k->ijk",
                       dn_xz.astype(np.float64),
                       dn_yz.astype(np.float64),
                       1.0 / safe_peak)
    zero_mask = np.abs(dn_peak) == 0.0
    if np.any(zero_mask):
        dn_3d[:, :, zero_mask] = 0.0

    n_3d = float(n_bg_m3) + dn_3d
    return n_3d, y_axis_m_out
