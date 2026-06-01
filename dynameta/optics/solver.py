"""
Default optical solver: scattered-field complex HCurl on the periodic unit cell,
HalfSpace PML in z, BDDC+GMRes (or UMFPACK). Implements OpticalSolver.

Incidence: a plane wave in the x-z plane (azimuth phi=0). Oblique incidence
(theta != 0) is supported for s-polarization (E along y) by two selectable
formulations (module global _OBLIQUE_FORMULATION), both reducing EXACTLY to the
proven normal-incidence curl-curl form at theta=0:

  "phase_in_space" (default): solve for the PHYSICAL field E on a QUASI-periodic
      HCurl space ng.Periodic(HCurl, phase=[exp(i*kx*Px)]*n_px + [1]*n_py). The
      identification order is f0(x=0).Identify(p(x=Px), tx=+Px), so master=x=0,
      minion=x=Px and the Bloch rule phase[idnr]=exp(i*k_par.(r_minion-r_master))
      = exp(+i*kx*Px). The curl operator is the GENUINE curl, so the standard
      stretched-coordinate HalfSpace z-PML transforms it exactly (this is the key
      advantage over the envelope route, whose algebraic transverse-phase term the
      z-PML cannot transform consistently). R/T are demodulated by exp(-i*kx*x).

  "envelope": solve for the envelope u with E=u*exp(i*kx*x), u plain-periodic, the
      transverse phase carried by a modified curl curl(u)+i*k_par x u. Cleaner
      periodicity but the z-PML mishandles the algebraic cross term at oblique
      (R is under-captured) -- kept for reference/diagnostics.

PML is the ordinary normal z-stretch (alpha=1j constant; a 1/cos(theta) rescaling
only changes absorption length, not conservation). Oblique assumes a vacuum/air
incidence medium (scattered-field background eps_bg=1). Validated against the `tmm`
library (validation/oblique_vs_tmm.py).
"""

from __future__ import annotations

import cmath
import math
import time

import numpy as np
import ngsolve as ng

from dynameta.core.interfaces import OpticalResult
from dynameta.optics.ngsolve_layered import OpticalGeometry, S

# Oblique formulation: "phase_in_space" (physical field, genuine curl, standard
# PML -- the validated route) or "envelope" (plain-periodic envelope, modified
# curl -- diagnostic only). Both are identical at normal incidence.
_OBLIQUE_FORMULATION = "phase_in_space"
_NONVAC_SUB_WARNED = False
# Sign of the transverse-phase term on the TEST envelope's modified curl (envelope
# route only): trial carries exp(+i k_par.r) (+kcross), test the conjugate (-kcross).
_TEST_KCROSS_SIGN = -1.0


def _detect_bloch_dirs(geo: OpticalGeometry):
    """Return a per-identification direction list (['x'|'y'] of length n_px+n_py)
    mapping each periodic idnr to its translation axis. CRITICAL: ng.Periodic keys
    its `phase` list per identification in IDNR order, and netgen does NOT keep that
    in creation order -- for a glued multi-layer stack the x- and y-face idnrs come
    out INTERLEAVED (x,y,x,y,...), one x/y pair per z-layer. A wrong mapping silently
    puts phase=1 on the x-faces (plain-periodic in x) -> the solver returns the
    NORMAL-incidence field at every angle (a particularly nasty silent failure, since
    normal incidence still validates). We resolve each idnr's axis by toggling a
    marker phase on that idnr alone and measuring whether it perturbs the x- or the
    y-boundary, then assert the recovered x/y counts (anti-silent-failure)."""
    cached = getattr(geo, "_bloch_dirs", None)
    if cached is not None:
        return cached
    mesh = geo.mesh
    n_px, n_py = geo.n_px, geo.n_py
    N = n_px + n_py
    Px, Py = geo.period_x_nm, geo.period_y_nm
    if N == 0 or n_py == 0:
        dirs = ["x"] * n_px
    elif n_px == 0:
        dirs = ["y"] * n_py
    else:
        zvals = [z for iv in geo.z_intervals_nm.values() for z in iv]
        zlo, zhi = min(zvals), max(zvals)
        th = cmath.exp(1j * 0.7853981634)           # marker phase != 1
        zfr = np.linspace(0.03, 0.97, 18)           # dense in z to hit every layer's face

        def viol(i):                                # x- vs y-boundary perturbation of idnr i
            phases = [(th if j == i else 1.0 + 0j) for j in range(N)]
            fes = ng.Periodic(ng.H1(mesh, order=1, complex=True), phase=phases)
            gf = ng.GridFunction(fes)
            gf.Set(ng.exp(0.01j * ng.z) * (1.0 + 0.3 * ng.y / Py + 0.25 * ng.x / Px + 0.2j))
            xv = yv = 0.0
            for fz in zfr:
                z = zlo + fz * (zhi - zlo)
                for fy in (0.3, 0.6):
                    try:
                        a = complex(gf(mesh(0.0, fy * Py, z))); b = complex(gf(mesh(Px, fy * Py, z)))
                        xv = max(xv, abs(b - a))
                    except Exception:
                        pass
                for fx in (0.3, 0.6):
                    try:
                        c = complex(gf(mesh(fx * Px, 0.0, z))); d = complex(gf(mesh(fx * Px, Py, z)))
                        yv = max(yv, abs(d - c))
                    except Exception:
                        pass
            return xv, yv
        dirs = []
        for i in range(N):
            xv, yv = viol(i)
            dirs.append("x" if xv > yv else "y")
        if dirs.count("x") != n_px or dirs.count("y") != n_py:
            raise RuntimeError(
                "Bloch periodic-phase idnr direction detection inconsistent: resolved "
                "{} x / {} y, expected {} / {}. Oblique incidence cannot be trusted.".format(
                    dirs.count("x"), dirs.count("y"), n_px, n_py))
    try: geo._bloch_dirs = dirs
    except Exception: pass
    return dirs


def _bloch_phase_list(geo: OpticalGeometry, kx_per_nm: float):
    """Floquet-Bloch phase list (one entry per periodic identification, in idnr
    order): exp(i*kx*Px) on x-identifications, 1 on y (ky=0 for x-z-plane incidence)."""
    dirs = _detect_bloch_dirs(geo)
    px_phase = cmath.exp(1j * kx_per_nm * geo.period_x_nm)
    return [(px_phase if d == "x" else 1.0 + 0j) for d in dirs]


def solve_fem(geo: OpticalGeometry, lambda_m: float,
                eps_cf: ng.CoefficientFunction, optical,
                *, order: int = 2, n_super: complex = 1.0 + 0j,
                n_sub: complex = 1.0 + 0j, verbose: bool = False) -> OpticalResult:
    """Solve and extract reflection r/R and (if a transmitted wave reaches the
    substrate) transmission t/T plus absorption A = 1 - R - T. n_super/n_sub are
    the semi-infinite superstrate/substrate refractive indices = sqrt(eps)."""
    k0 = 2.0 * math.pi / (lambda_m * S)        # nm^-1
    mesh = geo.mesh

    # ---- incidence: plane wave in the x-z plane (phi=0) ----
    theta = math.radians(float(getattr(optical, "incidence_angle_deg", 0.0) or 0.0))
    oblique = abs(theta) > 1e-9
    if oblique and optical.polarization != "y":
        raise NotImplementedError(
            "oblique incidence is implemented for s-polarization only "
            "(OpticalSpec.polarization='y'; E perpendicular to the x-z plane of "
            "incidence). p-pol oblique is a documented follow-up "
            "(docs/roadmap_phase5_stretch.md).")
    envelope = oblique and _OBLIQUE_FORMULATION == "envelope"
    kx = k0 * math.sin(theta)          # transverse wavevector (vacuum incidence medium)
    kz_s = k0 * math.cos(theta)        # normal wavevector in the vacuum background

    # PML: ordinary normal HalfSpace z-stretch, alpha=1j CONSTANT.
    pml_alpha = 1j
    z_air_top = geo.z_super_interface_nm
    z_sub_top = geo.z_sub_interface_nm
    try:
        mesh.UnSetPML("pml_top"); mesh.UnSetPML("pml_bot")
    except Exception:
        pass
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_air_top), normal=(0, 0, 1), alpha=pml_alpha), "pml_top")
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_sub_top), normal=(0, 0, -1), alpha=pml_alpha), "pml_bot")

    # Incident field. phase_in_space solves for the physical field -> full plane
    # wave exp(i kx x - i kz_s z). envelope solves for u -> envelope exp(-i kz_s z)
    # (the exp(i kx x) is divided out). Both equal exp(-i kz_s z)*pol at theta=0.
    inc_x_phase = (1.0 if envelope else ng.exp(1j * kx * ng.x))
    inc_field = inc_x_phase * ng.exp(-1j * kz_s * ng.z)
    if optical.polarization == "x":
        E_inc = ng.CoefficientFunction((inc_field, 0.0, 0.0))
    else:
        E_inc = ng.CoefficientFunction((0.0, inc_field, 0.0))
    eps_bg = 1.0

    # ---- periodic HCurl space ----
    if oblique and not envelope and (geo.n_px or geo.n_py):
        # quasi-periodic: u(minion=x=Px) = exp(+i kx Px) u(master=x=0). The phase
        # list is keyed per identification in idnr order, which netgen does NOT keep
        # in creation order -- _bloch_phase_list resolves + verifies the true order.
        phases = _bloch_phase_list(geo, kx)
        fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""), phase=phases)
    else:
        fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""))
    u, v = fes.TrialFunction(), fes.TestFunction()

    if envelope:
        # modified curl: C(w)=curl(w)+i k_par x w, k_par=(kx,0,0) -> kcross(w)=
        # 1j*(0,-kx w_z, kx w_y). Trial +kcross, test -kcross (conjugate phase).
        def kcross(w):
            return 1j * ng.CoefficientFunction((0.0, -kx * w[2], kx * w[1]))
        curlE = ng.curl(u) + kcross(u)
        curlV = ng.curl(v) + _TEST_KCROSS_SIGN * kcross(v)
    else:
        curlE, curlV = ng.curl(u), ng.curl(v)

    a = ng.BilinearForm(fes, symmetric=not envelope)
    a += (curlE * curlV - k0 ** 2 * eps_cf * (u * v)) * ng.dx
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

    # Demodulation: phase_in_space holds the physical field E=u exp(i kx x) -> demod
    # by exp(-i kx x) to recover the 0-order envelope. envelope already holds u.
    demod_kx = 0.0 if envelope else kx
    r = _reflection(mesh, gfu, kz_s, demod_kx, geo, optical)
    R = float(abs(r) ** 2)
    kz_sub = complex(np.sqrt(complex((complex(n_sub) * k0) ** 2 - kx ** 2)))
    t = _transmission(mesh, gfu, kz_s, kz_sub, demod_kx, geo, optical)
    if t is None:
        T = A = None
    else:
        if abs(complex(n_sub) - 1.0) > 0.01:
            global _NONVAC_SUB_WARNED
            if not _NONVAC_SUB_WARNED:
                print("[DynaMeta WARNING] exit medium is non-vacuum (n_sub={:.3f}). The "
                      "uniform-background scattered-field formulation (eps_bg=1) drives a "
                      "large volumetric source through the dense substrate at the vacuum "
                      "wavevector; transmission (and R for transmissive stacks) is "
                      "inaccurate/mesh-fragile at ALL angles. Reflection-mode stacks with a "
                      "bottom mirror are fine. See docs/roadmap_phase5_stretch.md (layered-"
                      "background-field fix).".format(complex(n_sub).real), flush=True)
                _NONVAC_SUB_WARNED = True
        kz_sup_med = complex(n_super) * k0 * math.cos(theta)
        T = float(abs(t) ** 2 * (kz_sub.real / max(kz_sup_med.real, 1e-12)))
        A = float(1.0 - R - T)
    return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                          solve_time_s=dt, t=t, T=T, A=A)


def _cell_average(mesh, gfu, z_probes, Px, Py, pol, demod_kx):
    """Transverse (x,y) cell-average of the 0-order envelope at each z. For the
    physical field the demod factor exp(-i*kx*x) removes the transverse Bloch phase
    (recovering the plain-periodic envelope, whose cell-average IS the 0-order
    Fourier coefficient); for the envelope formulation demod_kx=0 (it is already u)."""
    xs = np.linspace(0.0, Px, 6, endpoint=False)
    ys = np.linspace(0.0, Py, 6, endpoint=False)
    out = []
    for zv in z_probes:
        vals = []
        for xv in xs:
            demod = np.exp(-1j * demod_kx * xv)
            for yv in ys:
                try:
                    vals.append(complex(gfu(mesh(float(xv), float(yv), float(zv)))[pol]) * demod)
                except Exception:
                    pass
        out.append(complex(np.mean(vals)) if vals else 0 + 0j)
    return np.array(out)


def _reflection(mesh, gfu, kz_s, demod_kx, geo: OpticalGeometry, optical) -> complex:
    """0-order reflection: least-squares fit of the cell-averaged scattered envelope
    over z-planes in the superstrate buffer."""
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z_struct_top = geo.z_intervals_nm["superstrate"][0]
    z_air_top = geo.z_intervals_nm["superstrate"][1]
    z_lo = z_struct_top + 50.0
    z_hi = z_air_top - 50.0
    if z_hi <= z_lo:
        z_lo = z_struct_top + 0.2 * (z_air_top - z_struct_top)
        z_hi = z_struct_top + 0.8 * (z_air_top - z_struct_top)
    z_probes = np.linspace(z_lo, z_hi, 7)
    pol = 0 if optical.polarization == "x" else 1
    Es = _cell_average(mesh, gfu, z_probes, Px, Py, pol, demod_kx)
    # upward (reflected) exp(+i kz_s z) + residual downward exp(-i kz_s z)
    M = np.column_stack([np.exp(+1j * kz_s * z_probes), np.exp(-1j * kz_s * z_probes)])
    coeffs, *_ = np.linalg.lstsq(M, Es, rcond=None)
    return complex(coeffs[0])


def _transmission(mesh, gfu, kz_s, kz_sub, demod_kx, geo: OpticalGeometry, optical):
    """0-order transmission amplitude t: fit the cell-averaged TOTAL envelope in the
    substrate buffer to a downward wave exp(-i kz_sub z). Returns None if there is no
    usable substrate buffer."""
    if "substrate" not in geo.z_intervals_nm:
        return None
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z_sub_lo, z_sub_hi = geo.z_intervals_nm["substrate"]
    pad = 0.1 * (z_sub_hi - z_sub_lo)
    z_lo, z_hi = z_sub_lo + pad, z_sub_hi - pad
    if z_hi <= z_lo:
        return None
    z_probes = np.linspace(z_lo, z_hi, 7)
    pol = 0 if optical.polarization == "x" else 1
    # scattered envelope (cell-averaged, demodulated) + incident envelope exp(-i kz_s z)
    Es = _cell_average(mesh, gfu, z_probes, Px, Py, pol, demod_kx)
    Et = Es + np.exp(-1j * kz_s * z_probes)
    # downward (transmitted) exp(-i kz_sub z) + any upward residual exp(+i kz_sub z)
    M = np.column_stack([np.exp(-1j * kz_sub * z_probes), np.exp(+1j * kz_sub * z_probes)])
    coeffs, *_ = np.linalg.lstsq(M, Et, rcond=None)
    return complex(coeffs[0])
