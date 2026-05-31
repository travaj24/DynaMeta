"""
Stage 3 NGSolve solver: scattered-field formulation, x-pol normal-
incidence plane wave, complex-symmetric HCurl, BDDC + GMRes by default.

This is a focused clean-room version of the existing solve_fem_3d. It
assumes the geometry was built by ngsolve_build.build_unit_cell_3d so
the boundary tags 'periodic_x_lo/hi/y_lo/hi' are present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import ngsolve as ng

from dynameta.design import OpticalSpec
from dynameta.stage3_optical.ngsolve_build import Cell3DGeometry, UNIT_SCALE


@dataclass
class FEMResult:
    r:               complex                  # complex reflection coefficient
    R:               float                    # |r|^2
    a_dn:            complex                  # downward amplitude (sanity)
    a_up:            complex                  # upward amplitude
    solve_time_s:    float


def solve_fem(geo: Cell3DGeometry, lambda_m: float,
                eps_cf: ng.CoefficientFunction,
                optical: OpticalSpec,
                *, order: int = 2,
                verbose: bool = False) -> FEMResult:
    """Solve the scattered-field Maxwell equation at one (lambda, eps_cf)
    and return the complex reflection coefficient.
    """
    import time
    S    = UNIT_SCALE
    k0_nm = 2.0 * math.pi / (lambda_m * S)

    mesh = geo.mesh

    # --- Cartesian HalfSpace PML in z on the top/bottom PML bands. Stretches
    # ONLY z (a full-box PML would stretch x,y and break the periodic
    # identifications). alpha=1j attenuates outgoing exp(-i k0 z) waves under
    # the exp(-i omega t) convention. Matches the proven Modulator driver_3d. ---
    z_air_top     = geo.z_intervals["air_buffer"][1]
    z_pml_bot_top = geo.z_intervals["pml_bot"][1]
    try:
        mesh.UnSetPML("pml_top"); mesh.UnSetPML("pml_bot")
    except Exception:
        pass
    pml_alpha = 1j
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_air_top),
                                   normal=(0, 0, 1), alpha=pml_alpha), "pml_top")
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_pml_bot_top),
                                   normal=(0, 0, -1), alpha=pml_alpha), "pml_bot")

    # Scattered-field formulation: incident plane wave is known; solve for E_s.
    # E_inc = x_hat * exp(-i*k0*z)  (or y_hat if optical.polarization == "y")
    if optical.polarization == "x":
        E_inc = ng.CoefficientFunction((ng.exp(-1j * k0_nm * ng.z), 0.0, 0.0))
    else:
        E_inc = ng.CoefficientFunction((0.0, ng.exp(-1j * k0_nm * ng.z), 0.0))

    # Background eps used for the scattered-field source term
    eps_bg = 1.0       # background = air

    # Periodic HCurl. NO Dirichlet: the z PML absorbs outgoing waves; clamping
    # E_tangent on the outer z-faces would make a reflection cavity. The x/y
    # Bloch periodicity comes from ng.Periodic + the OCC identifications built
    # in ngsolve_build (NOT a periodic= kwarg, which HCurl silently ignores).
    fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""))
    u = fes.TrialFunction()
    v = fes.TestFunction()
    a = ng.BilinearForm(fes, symmetric=False)
    a += (ng.curl(u) * ng.curl(v) - k0_nm**2 * eps_cf * u * v) * ng.dx
    f = ng.LinearForm(fes)
    f += (k0_nm**2 * (eps_cf - eps_bg) * E_inc * v) * ng.dx

    if optical.linear_solver.startswith("bddc"):
        pre = ng.Preconditioner(a, "bddc")

    with ng.TaskManager():
        a.Assemble()
        f.Assemble()

    gfu = ng.GridFunction(fes)
    t0 = time.time()
    with ng.TaskManager():
        if optical.linear_solver == "umfpack":
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(),
                                           inverse="umfpack") * f.vec
        elif optical.linear_solver == "bddc_cg":
            inv = ng.solvers.CGSolver(mat=a.mat, pre=pre.mat,
                                         tol=optical.gmres_rtol,
                                         maxiter=optical.gmres_max_iter)
            gfu.vec.data = inv * f.vec
        elif optical.linear_solver == "bddc_gmres":
            ng.solvers.GMRes(A=a.mat, b=f.vec, pre=pre.mat, x=gfu.vec,
                                tol=optical.gmres_rtol,
                                maxsteps=optical.gmres_max_iter,
                                printrates=verbose)
        else:
            raise ValueError("unknown linear_solver: " + optical.linear_solver)
    dt = time.time() - t0

    # Probe scattered field on two horizontal planes to extract a_up / a_dn
    a_up, a_dn = _extract_amplitudes(mesh, gfu, k0_nm, geo, optical)
    r = a_up      # by convention, incident has amplitude 1
    R = float(abs(r) ** 2)
    return FEMResult(r=r, R=R, a_dn=a_dn, a_up=a_up, solve_time_s=dt)


def _extract_amplitudes(mesh: ng.Mesh, gfu: ng.GridFunction,
                          k0_nm: float, geo: Cell3DGeometry,
                          optical: OpticalSpec
                          ) -> Tuple[complex, complex]:
    """0-order reflection amplitude by least-squares fit over MULTIPLE
    z-planes in the air buffer (above the structure, below the top PML):
        E_s(z) ~ a_up*exp(+i k0 z) + a_dn*exp(-i k0 z)
    r = a_up (incident has unit amplitude). a_dn should be small if the PML
    absorbs the residual downward scatter. At each z, average the in-plane
    E_s component (pol) over an Nx x Ny grid to isolate the 0-order Fourier
    term. Matches the proven Modulator driver_3d extraction (a single-plane
    probe is fragile: a missed point set silently gives r=0)."""
    P_nm = geo.period_nm
    z_struct_top = geo.z_intervals["air_buffer"][0]   # top of structure
    z_air_top    = geo.z_intervals["air_buffer"][1]   # air/PML interface
    z_probe_lo = z_struct_top + 50.0                  # 50 nm above structure
    z_probe_hi = z_air_top - 50.0                     # 50 nm below PML
    if z_probe_hi <= z_probe_lo:                      # thin air buffer fallback
        z_probe_lo = z_struct_top + 0.2 * (z_air_top - z_struct_top)
        z_probe_hi = z_struct_top + 0.8 * (z_air_top - z_struct_top)
    Nz_probe = 7
    z_probes = np.linspace(z_probe_lo, z_probe_hi, Nz_probe)
    Nx, Ny = 6, 6
    xs = np.linspace(0.0, P_nm, Nx, endpoint=False)
    ys = np.linspace(0.0, P_nm, Ny, endpoint=False)
    pol = 0 if optical.polarization == "x" else 1

    Es_at_z = []
    for zv in z_probes:
        vals = []
        for xv in xs:
            for yv in ys:
                try:
                    e = gfu(mesh(float(xv), float(yv), float(zv)))
                    vals.append(complex(e[pol]))
                except Exception:
                    pass
        Es_at_z.append(complex(np.mean(vals)) if vals else 0 + 0j)
    Es_at_z = np.array(Es_at_z)

    M = np.column_stack([np.exp(+1j * k0_nm * z_probes),
                           np.exp(-1j * k0_nm * z_probes)])
    coeffs, *_ = np.linalg.lstsq(M, Es_at_z, rcond=None)
    a_up, a_dn = complex(coeffs[0]), complex(coeffs[1])
    return (a_up, a_dn)
