"""
Optical-side configuration: incidence, polarization, solver, and the
xy-product symmetrization toggle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Polarization = Literal["x", "y"]
LinearSolver = Literal["umfpack", "bddc_cg", "bddc_gmres"]


@dataclass
class OpticalSpec:
    """
    Args:
      polarization     : 'x' or 'y'. For a square patch with 4-fold
                          symmetric eps both give the same result; only
                          matters when the carrier-density field breaks
                          x<->y symmetry.
      incidence_angle_deg : 0 = normal incidence (Bloch k_parallel = 0).
                              Non-zero not yet supported.
      use_symmetrization  : True  -> apply xy-product symmetrization to
                                      the 2D DEVSIM n(x,z) for a 4-fold
                                      symmetric 3D eps(x,y,z).
                              False -> extrude eps(x,z) along y (y-invariant;
                                      breaks x<->y symmetry; faster).
      ny_sym            : number of y samples when use_symmetrization=True.
                          Default 256 (~ 1.4 nm at P = 370 nm).
      linear_solver     : 'umfpack' (direct LU, high memory),
                          'bddc_cg' or 'bddc_gmres' (iterative, low memory).
                          Default 'bddc_gmres'.
      gmres_rtol        : convergence tolerance for GMRes.
      gmres_max_iter    : iteration cap.
    """
    polarization:            Polarization = "x"
    incidence_angle_deg:     float = 0.0
    use_symmetrization:      bool = True
    ny_sym:                  int = 256
    linear_solver:           LinearSolver = "bddc_gmres"
    gmres_rtol:              float = 1e-6
    gmres_max_iter:          int = 800

    def __post_init__(self) -> None:
        if abs(self.incidence_angle_deg) > 1e-6:
            raise NotImplementedError(
                "Non-normal incidence (incidence_angle_deg={}) not yet "
                "supported.".format(self.incidence_angle_deg))
        if self.polarization not in ("x", "y"):
            raise ValueError("polarization must be 'x' or 'y'")
        if self.ny_sym < 4:
            raise ValueError("ny_sym must be >= 4")
