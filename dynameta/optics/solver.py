"""
Default optical solver: scattered-field complex HCurl on the periodic unit cell,
HalfSpace PML in z, BDDC+GMRes (or UMFPACK). Implements OpticalSolver.

Incidence: a plane wave in the x-z plane (azimuth phi=0). Normal incidence
(theta=0) is the proven path (x- or y-pol). Oblique incidence (theta != 0) is
supported for s-polarization (E along y, perpendicular to the x-z plane) via
Floquet-Bloch periodicity: the x-periodic faces carry a phase exp(i*kx*Px), the
incident field carries the transverse phase exp(i*kx*x), and the reflection/
transmission fits are demodulated by exp(-i*kx*x) and use the medium-correct
normal wavevectors kz = sqrt((n*k0)^2 - kx^2). Oblique assumes a vacuum/air
incidence medium (the scattered-field background eps_bg = 1). Validated against
the `tmm` library for layered stacks (see C:\\tmp\\lib_p5_oblique_tmm.py).
"""

from __future__ import annotations

import math
import time

import numpy as np
import ngsolve as ng

from dynameta.core.interfaces import OpticalResult
from dynameta.optics.ngsolve_layered import OpticalGeometry, S

_OBLIQUE_WARNED = False


def solve_fem(geo: OpticalGeometry, lambda_m: float,
                eps_cf: ng.CoefficientFunction, optical,
                *, order: int = 2, n_super: complex = 1.0 + 0j,
                n_sub: complex = 1.0 + 0j, verbose: bool = False) -> OpticalResult:
    """Solve and extract reflection r/R and (if a transmitted wave reaches the
    substrate) transmission t/T plus absorption A = 1 - R - T. n_super/n_sub are
    the semi-infinite superstrate/substrate refractive indices = sqrt(eps)."""
    k0 = 2.0 * math.pi / (lambda_m * S)        # nm^-1
    mesh = geo.mesh

    z_air_top = geo.z_super_interface_nm
    z_sub_top = geo.z_sub_interface_nm
    try:
        mesh.UnSetPML("pml_top"); mesh.UnSetPML("pml_bot")
    except Exception:
        pass
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_air_top), normal=(0, 0, 1), alpha=1j), "pml_top")
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_sub_top), normal=(0, 0, -1), alpha=1j), "pml_bot")

    # ---- incidence: plane wave in the x-z plane (phi=0) ----
    theta = math.radians(float(getattr(optical, "incidence_angle_deg", 0.0) or 0.0))
    oblique = abs(theta) > 1e-9
    if oblique:
        global _OBLIQUE_WARNED
        if not _OBLIQUE_WARNED:
            print("[DynaMeta WARNING] oblique incidence uses an angle-INDEPENDENT "
                  "HalfSpace PML; energy conservation degrades with angle. Validated vs "
                  "the tmm library at NORMAL incidence (R to 0.4%); R+T energy error is "
                  "~12%/27% at 15/30deg. Treat oblique R/T as QUALITATIVE pending an "
                  "angle-aware PML (see docs/roadmap_phase5_stretch.md).", flush=True)
            _OBLIQUE_WARNED = True
    if oblique and optical.polarization != "y":
        raise NotImplementedError(
            "oblique incidence is implemented for s-polarization only "
            "(OpticalSpec.polarization='y'; E perpendicular to the x-z plane of "
            "incidence). p-pol oblique is a documented follow-up "
            "(docs/roadmap_phase5_stretch.md).")
    kx = k0 * math.sin(theta)          # transverse wavevector (vacuum incidence medium)
    kz_s = k0 * math.cos(theta)        # normal wavevector in the vacuum background
    inc_phase = ng.exp(1j * kx * ng.x - 1j * kz_s * ng.z)
    if optical.polarization == "x":
        E_inc = ng.CoefficientFunction((inc_phase, 0.0, 0.0))
    else:
        E_inc = ng.CoefficientFunction((0.0, inc_phase, 0.0))
    eps_bg = 1.0

    # ---- Floquet-Bloch periodic HCurl space ----
    Px_nm, Py_nm = geo.period_x_nm, geo.period_y_nm
    if oblique and (geo.n_px or geo.n_py):
        # phase list in identification-creation order: all px (x-translation by Px),
        # then all py (y-translation by Py, ky=0 -> phase 1).
        phases = [ng.exp(1j * kx * Px_nm)] * geo.n_px + [1.0 + 0j] * geo.n_py
        fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""),
                            phase=phases)
    else:
        fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""))

    u, v = fes.TrialFunction(), fes.TestFunction()
    a = ng.BilinearForm(fes, symmetric=True)
    a += (ng.curl(u) * ng.curl(v) - k0 ** 2 * eps_cf * (u * v)) * ng.dx
    f = ng.LinearForm(fes)
    f += (k0 ** 2 * (eps_cf - eps_bg) * (E_inc * v)) * ng.dx
    pre = ng.Preconditioner(a, "bddc") if optical.linear_solver.startswith("bddc") else None

    gfu = ng.GridFunction(fes)
    t0 = time.time()
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        if optical.linear_solver == "umfpack":
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec
        elif optical.linear_solver == "bddc_cg":
            inv = ng.solvers.CGSolver(mat=a.mat, pre=pre.mat, tol=optical.gmres_rtol,
                                        maxiter=optical.gmres_max_iter)
            gfu.vec.data = inv * f.vec
        else:  # bddc_gmres
            ng.solvers.GMRes(A=a.mat, b=f.vec, pre=pre.mat, x=gfu.vec,
                                tol=optical.gmres_rtol, maxsteps=optical.gmres_max_iter,
                                printrates=verbose)
    dt = time.time() - t0

    r = _reflection(mesh, gfu, kx, kz_s, geo, optical)
    R = float(abs(r) ** 2)
    # Transmission: substrate normal wavevector via Snell (kx conserved across
    # interfaces): kz_sub = sqrt((n_sub k0)^2 - kx^2).
    kz_sub = complex(np.sqrt(complex((complex(n_sub) * k0) ** 2 - kx ** 2)))
    t = _transmission(mesh, gfu, kx, kz_s, kz_sub, geo, optical)
    if t is None:
        T = A = None
    else:
        # s-pol Poynting-z ratio: T = |t|^2 Re(kz_sub)/Re(kz_super_medium),
        # kz_super_medium = n_super k0 cos(theta). Reduces to |t|^2 Re(n_sub)/Re(n_super)
        # at normal incidence.
        kz_sup_med = complex(n_super) * k0 * math.cos(theta)
        T = float(abs(t) ** 2 * (kz_sub.real / max(kz_sup_med.real, 1e-12)))
        A = float(1.0 - R - T)
    return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                          solve_time_s=dt, t=t, T=T, A=A)


def _reflection(mesh, gfu, kx, kz_s, geo: OpticalGeometry, optical) -> complex:
    """0-order reflection by least-squares fit of the (transverse-demodulated)
    scattered field over z-planes in the superstrate buffer."""
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z_struct_top = geo.z_intervals_nm["superstrate"][0]
    z_air_top = geo.z_intervals_nm["superstrate"][1]
    z_lo = z_struct_top + 50.0
    z_hi = z_air_top - 50.0
    if z_hi <= z_lo:
        z_lo = z_struct_top + 0.2 * (z_air_top - z_struct_top)
        z_hi = z_struct_top + 0.8 * (z_air_top - z_struct_top)
    z_probes = np.linspace(z_lo, z_hi, 7)
    xs = np.linspace(0.0, Px, 6, endpoint=False)
    ys = np.linspace(0.0, Py, 6, endpoint=False)
    pol = 0 if optical.polarization == "x" else 1

    Es = []
    for zv in z_probes:
        vals = []
        for xv in xs:
            demod = np.exp(-1j * kx * xv)       # remove the transverse Bloch phase
            for yv in ys:
                try:
                    vals.append(complex(gfu(mesh(float(xv), float(yv), float(zv)))[pol]) * demod)
                except Exception:
                    pass
        Es.append(complex(np.mean(vals)) if vals else 0 + 0j)
    Es = np.array(Es)
    # upward (reflected) exp(+i kz_s z) + residual downward exp(-i kz_s z)
    M = np.column_stack([np.exp(+1j * kz_s * z_probes), np.exp(-1j * kz_s * z_probes)])
    coeffs, *_ = np.linalg.lstsq(M, Es, rcond=None)
    return complex(coeffs[0])


def _transmission(mesh, gfu, kx, kz_s, kz_sub, geo: OpticalGeometry, optical):
    """0-order transmission amplitude t: fit the (demodulated) TOTAL field in the
    substrate buffer to a downward wave exp(-i kz_sub z). Returns None if there is
    no usable substrate buffer."""
    if "substrate" not in geo.z_intervals_nm:
        return None
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z_sub_lo, z_sub_hi = geo.z_intervals_nm["substrate"]
    pad = 0.1 * (z_sub_hi - z_sub_lo)
    z_lo, z_hi = z_sub_lo + pad, z_sub_hi - pad
    if z_hi <= z_lo:
        return None
    z_probes = np.linspace(z_lo, z_hi, 7)
    xs = np.linspace(0.0, Px, 6, endpoint=False)
    ys = np.linspace(0.0, Py, 6, endpoint=False)
    pol = 0 if optical.polarization == "x" else 1

    Et = []
    for zv in z_probes:
        vals = []
        for xv in xs:
            demod = np.exp(-1j * kx * xv)
            inc = np.exp(1j * kx * xv - 1j * kz_s * zv)    # vacuum incident (background)
            for yv in ys:
                try:
                    e_s = complex(gfu(mesh(float(xv), float(yv), float(zv)))[pol])
                    vals.append((e_s + inc) * demod)        # total field, demodulated
                except Exception:
                    pass
        Et.append(complex(np.mean(vals)) if vals else 0 + 0j)
    Et = np.array(Et)
    # downward (transmitted) exp(-i kz_sub z) + any upward residual exp(+i kz_sub z)
    M = np.column_stack([np.exp(-1j * kz_sub * z_probes), np.exp(+1j * kz_sub * z_probes)])
    coeffs, *_ = np.linalg.lstsq(M, Et, rcond=None)
    return complex(coeffs[0])
