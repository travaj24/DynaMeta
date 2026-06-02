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

p-pol (TM, E in the x-z plane) oblique and conical s-pol (azimuth phi != 0) are also
implemented (each tmm-validated at phi=0 / phi-invariance).

PML is the ordinary normal z-stretch (alpha=1j CONSTANT; it is NOT angle-aware, so
energy conservation degrades with angle -- validated to ~1% through 30 deg). solve_fem
emits a runtime warning at oblique incidence and the OpticalSpec caps the polar angle.
Oblique REQUIRES a vacuum/air incidence medium (n_super=1): the in-plane wavevector
kx=k0 sin(theta) uses the vacuum dispersion, so solve_fem RAISES on a non-vacuum
superstrate at angle rather than returning a silently-wrong result. Validated against
the `tmm` library (validation/oblique_vs_tmm.py).
"""

from __future__ import annotations

import cmath
import math
import re
import time
import warnings

import numpy as np
import ngsolve as ng

from typing import TYPE_CHECKING

from dynameta.core.interfaces import OpticalResult
from dynameta.optics.ngsolve_layered import OpticalGeometry, S

if TYPE_CHECKING:                       # type-only; OpticalSpec lives in geometry (no runtime dep)
    from dynameta.geometry.specs import OpticalSpec

# Oblique formulation: "phase_in_space" (physical field, genuine curl, standard
# PML -- the validated route) or "envelope" (plain-periodic envelope, modified
# curl -- diagnostic only). Both are identical at normal incidence.
_OBLIQUE_FORMULATION = "phase_in_space"
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


def _bloch_phase_list(geo: OpticalGeometry, kx_per_nm: float, ky_per_nm: float = 0.0):
    """Floquet-Bloch phase list (one entry per periodic identification, in idnr order):
    exp(i*kx*Px) on x-identifications, exp(i*ky*Py) on y-identifications. ky=0 (x-z-plane
    incidence) gives phase 1 on y; ky!=0 is CONICAL incidence (2D Bloch phase)."""
    dirs = _detect_bloch_dirs(geo)
    px_phase = cmath.exp(1j * kx_per_nm * geo.period_x_nm)
    py_phase = cmath.exp(1j * ky_per_nm * geo.period_y_nm)
    return [(px_phase if d == "x" else py_phase) for d in dirs]


def _incidence_geometry(optical, n_super):
    """Derive (theta, phi, oblique, conical) [radians/bools] from the OpticalSpec and VALIDATE
    the incidence against solve_fem's implemented regime: top-side only; vacuum superstrate if
    oblique; pol constraints for oblique/conical. Raises NotImplementedError on an unsupported
    combination and warns on a non-angle-aware oblique solve. (Audit OPT-1/OPT-3/OPT-4 guards.)"""
    theta = math.radians(float(getattr(optical, "incidence_angle_deg", 0.0) or 0.0))
    phi = math.radians(float(getattr(optical, "azimuth_deg", 0.0) or 0.0))
    oblique = abs(theta) > 1e-9
    conical = abs(phi) > 1e-9
    if oblique and optical.polarization == "x":
        raise NotImplementedError(
            "oblique incidence requires polarization='y' (s-pol) or 'p' (p-pol); "
            "'x' (E along x) is not transverse to an oblique x-z-plane wavevector.")
    if conical and optical.polarization != "y":
        raise NotImplementedError("conical incidence (azimuth != 0) is s-pol only")
    if getattr(optical, "incidence_side", "top") != "top":
        raise NotImplementedError(
            "solve_fem implements top-side incidence only; incidence_side='{}' is not "
            "supported (the source + R/T extraction are hardwired to top illumination)."
            .format(getattr(optical, "incidence_side", "top")))
    if oblique and abs(complex(n_super) - 1.0) > 1e-6:
        # The in-plane wavevector uses kx = k0 sin(theta) (the VACUUM dispersion). A dense
        # incidence medium would need kx = Re(n_super) k0 sin(theta) and a matched
        # T-normalization; without it the result is silently wrong (audit OPT-1).
        raise NotImplementedError(
            "oblique incidence assumes a vacuum/air incidence medium (n_super=1); got "
            "n_super={:.4g}. Use normal incidence for a non-vacuum superstrate, or extend "
            "solve_fem with the dense-incidence dispersion.".format(complex(n_super)))
    if oblique:
        # The fixed-alpha HalfSpace PML is not angle-aware; energy conservation degrades
        # with angle (audit OPT-3 / cross-cutting F5 -- the warning the README promised).
        warnings.warn(
            "oblique incidence: the fixed-alpha HalfSpace z-PML is not angle-aware, so "
            "R/T/energy-conservation degrade with angle (validated to ~1% through 30 deg). "
            "Treat oblique R/T as approximate.", stacklevel=2)
    return theta, phi, oblique, conical


def solve_fem(geo: OpticalGeometry, lambda_m: float,
                eps_cf: ng.CoefficientFunction, optical: "OpticalSpec",
                *, order: int = 2, n_super: complex = 1.0 + 0j,
                n_sub: complex = 1.0 + 0j, verbose: bool = False) -> OpticalResult:
    """Solve and extract reflection r/R and (if a transmitted wave reaches the
    substrate) transmission t/T. n_super/n_sub are the semi-infinite superstrate/
    substrate refractive indices = sqrt(eps).

    A = 1 - R - T is the energy-budget CLOSURE (it is identically 1-R-T, not an
    independent measurement). result.A_independent is the INDEPENDENTLY measured
    absorbed fraction (volumetric Im(eps)|E|^2 integral); |A - A_independent| is the
    genuine, non-tautological energy/numerics diagnostic."""
    k0 = 2.0 * math.pi / (lambda_m * S)        # nm^-1
    mesh = geo.mesh

    # ---- incidence: plane wave, polar angle theta, azimuth phi (derived + validated) ----
    theta, phi, oblique, conical = _incidence_geometry(optical, n_super)
    pol_p = optical.polarization == "p"        # p-pol: E in the x-z plane (Ex, Ez)
    envelope = oblique and _OBLIQUE_FORMULATION == "envelope" and not pol_p and not conical
    # in-plane wavevector k_par = (kx, ky); kz_s = k0 cos(theta) (vacuum incidence medium)
    kx = k0 * math.sin(theta) * math.cos(phi)
    ky = k0 * math.sin(theta) * math.sin(phi)
    kz_s = k0 * math.cos(theta)
    # s-pol unit E (perpendicular to the plane of incidence); reduces to +y at phi=0
    es_x, es_y = -math.sin(phi), math.cos(phi)

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

    # ---- layered (Fresnel two-region) background ----
    # eps_bg(z) = superstrate medium above the substrate-top interface z_int, substrate
    # medium below. E_bg = the analytic bare air/substrate Fresnel field (incident +
    # background reflection R0 above, background transmission T0 below). The scattered
    # source k0^2 (eps - eps_bg) E_bg is then nonzero ONLY in the structure layers
    # (slab/oxide/patch/carrier) -- the substrate carries NO volumetric source
    # (eps==eps_bg there). This is what makes a dense (non-vacuum) substrate accurate:
    # the old uniform eps_bg=1 drove a huge source through the whole substrate at the
    # WRONG (vacuum) wavevector. Reduces exactly to the plain incident wave when
    # n_sub==n_super==1 (R0=0, T0=1). Incidence medium assumed vacuum (kx=k0 sin th).
    kz_sub = complex(np.sqrt(complex((complex(n_sub) * k0) ** 2 - kx ** 2 - ky ** 2)))
    kz_s_c = complex(kz_s)
    z_int = (geo.z_intervals_nm["substrate"][1] if "substrate" in geo.z_intervals_nm
              else geo.z_sub_interface_nm)                       # substrate-top interface (nm)
    eps_sup_c, eps_sub_c = complex(n_super) ** 2, complex(n_sub) ** 2
    # transverse Bloch phase exp(i(kx x + ky y)); reduces to exp(i kx x) at phi=0
    inc_x_phase = (1.0 if envelope else ng.exp(1j * (kx * ng.x + ky * ng.y)))

    if pol_p:
        # p-pol: E in the x-z plane (Ex, Ez). The background reflection/transmission
        # E-vector amplitudes (rho, tau) come from the physical interface BCs at z_int
        # -- tangential Ex and Hy continuity -- solved NUMERICALLY (no Fresnel sign-
        # convention ambiguity). Hy ~ Ex*eps/qz (qz the z-wavevector: -kz down, +kz up).
        cth, sth = kz_s_c / (complex(n_super) * k0), kx / (complex(n_super) * k0)
        cth_t, sth_t = kz_sub / (complex(n_sub) * k0), kx / (complex(n_sub) * k0)
        A = cmath.exp(-1j * kz_s_c * z_int); B = cmath.exp(1j * kz_s_c * z_int)
        C = cmath.exp(-1j * kz_sub * z_int)
        M = np.array([[cth * B,                   -cth_t * C],
                      [cth * B * eps_sup_c / kz_s_c, cth_t * C * eps_sub_c / kz_sub]], dtype=complex)
        rhs = np.array([-cth * A, cth * A * eps_sup_c / kz_s_c], dtype=complex)
        pp_rho, pp_tau = (complex(val) for val in np.linalg.solve(M, rhs))
        ex_sup = cth * ng.exp((-1j * kz_s_c) * ng.z) + pp_rho * cth * ng.exp((1j * kz_s_c) * ng.z)
        ez_sup = sth * ng.exp((-1j * kz_s_c) * ng.z) - pp_rho * sth * ng.exp((1j * kz_s_c) * ng.z)
        ex_sub = pp_tau * cth_t * ng.exp((-1j * kz_sub) * ng.z)
        ez_sub = pp_tau * sth_t * ng.exp((-1j * kz_sub) * ng.z)
        E_bg = ng.CoefficientFunction((inc_x_phase * ng.IfPos(ng.z - z_int, ex_sup, ex_sub),
                                        0.0,
                                        inc_x_phase * ng.IfPos(ng.z - z_int, ez_sup, ez_sub)))
        R0 = T0 = None                          # p-pol extracts the total field directly
    else:
        # s-pol (E along y) / x-pol: scalar tangential field. Fresnel R0/T0 (field
        # amplitude), z=0 reference. The extractor returns the scattered amplitude and
        # the caller adds R0/T0 back.
        r_f = (kz_s_c - kz_sub) / (kz_s_c + kz_sub)
        t_f = 2.0 * kz_s_c / (kz_s_c + kz_sub)
        R0 = r_f * cmath.exp(-2j * kz_s_c * z_int)
        T0 = t_f * cmath.exp(-1j * (kz_s_c - kz_sub) * z_int)
        sup_bg = ng.exp((-1j * kz_s_c) * ng.z) + R0 * ng.exp((1j * kz_s_c) * ng.z)
        sub_bg = T0 * ng.exp((-1j * kz_sub) * ng.z)
        bg_field = inc_x_phase * ng.IfPos(ng.z - z_int, sup_bg, sub_bg)
        if optical.polarization == "x":
            E_bg = ng.CoefficientFunction((bg_field, 0.0, 0.0))
        else:
            # s-pol along the (conical) in-plane direction E_s = (-sin phi, cos phi, 0);
            # reduces to (0, bg_field, 0) at phi=0.
            E_bg = ng.CoefficientFunction((es_x * bg_field, es_y * bg_field, 0.0))
    eps_bg_cf = ng.IfPos(ng.z - z_int, eps_sup_c, eps_sub_c)

    # ---- periodic HCurl space ----
    if oblique and not envelope and (geo.n_px or geo.n_py):
        # quasi-periodic: u(minion=x=Px) = exp(+i kx Px) u(master=x=0). The phase
        # list is keyed per identification in idnr order, which netgen does NOT keep
        # in creation order -- _bloch_phase_list resolves + verifies the true order.
        phases = _bloch_phase_list(geo, kx, ky)
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
    f += (k0 ** 2 * (eps_cf - eps_bg_cf) * (E_bg * v)) * ng.dx
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

    # OS-1: bddc_gmres/bddc_cg can stop at maxsteps without reaching tol (the ill-conditioned
    # ENZ / lossy-metal regime) and silently return a wrong field. Independently measure the
    # relative residual ||b - A x|| / ||b|| and warn so a non-converged solve is not trusted.
    if optical.linear_solver != "umfpack":
        rvec = gfu.vec.CreateVector()
        rvec.data = f.vec - a.mat * gfu.vec
        rn = float(np.linalg.norm(rvec.FV().NumPy()))
        bn = float(np.linalg.norm(f.vec.FV().NumPy()))
        relres = rn / bn if bn > 0.0 else rn
        if relres > 1e-3:
            warnings.warn(
                "solve_fem: iterative solver '{}' did not converge (relative residual {:.2e} "
                "> 1e-3 after {} steps); R/T/A are unreliable. Use linear_solver='umfpack', "
                "raise gmres_max_iter, or refine the mesh.".format(
                    optical.linear_solver, relres, optical.gmres_max_iter), stacklevel=2)

    # Demodulation: phase_in_space holds the physical field E=u exp(i(kx x+ky y)) -> demod
    # by exp(-i(kx x+ky y)) to recover the 0-order Fourier coefficient. envelope already
    # holds u (kx=ky=0 in the demod).
    kx_d = 0.0 if envelope else kx
    ky_d = 0.0 if envelope else ky
    if pol_p:
        # p-pol: reconstruct the TOTAL field (E_bg + scattered gfu) and extract from the
        # tangential Ex up/down ratio (convention-robust). T carries the p-pol Poynting
        # factor (Sz ~ |Ex|^2 eps/kz). (p-pol is phi=0 only.)
        r, R, t, T = _ppol_extract(mesh, E_bg + gfu, kz_s, kz_sub, kx, geo,
                                     eps_sup_c, eps_sub_c)
        A = None if T is None else float(1.0 - R - T)
    else:
        # project the scattered field onto the extraction polarization: tangential Ex
        # (1,0,0) for x-pol, or the (conical) s-pol unit vector Es=(es_x,es_y,0) for 'y'.
        proj = (1.0, 0.0, 0.0) if optical.polarization == "x" else (es_x, es_y, 0.0)
        # total amplitude = background (analytic R0/T0) + scattered (fitted from gfu)
        r = complex(R0) + _reflection(mesh, gfu, kz_s, proj, kx_d, ky_d, geo)
        R = float(abs(r) ** 2)
        t_scat = _transmission(mesh, gfu, kz_sub, proj, kx_d, ky_d, geo)
        if t_scat is None:
            t = T = A = None
        else:
            t = complex(T0) + t_scat
            kz_sup_med = complex(n_super) * k0 * math.cos(theta)
            T = float(abs(t) ** 2 * (kz_sub.real / max(kz_sup_med.real, 1e-12)))
            A = float(1.0 - R - T)
    # Independent absorption diagnostic (audit OPT-2): the normalized volumetric loss
    # integral, computed from the reconstructed TOTAL field. Best-effort -- a diagnostic
    # must not break the solve, so a failure warns (not silent) and yields None.
    try:
        A_independent = _absorbed_fraction(mesh, E_bg + gfu, eps_cf, k0, theta,
                                            geo.period_x_nm, geo.period_y_nm)
    except Exception as _e:                                   # noqa: BLE001 (diagnostic)
        warnings.warn("independent absorption diagnostic unavailable: {}".format(_e))
        A_independent = None
    return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                          solve_time_s=dt, t=t, T=T, A=A, A_independent=A_independent)


def _cell_average(mesh, field, z_probes, Px, Py, proj, kx, ky):
    """Transverse (x,y) cell-average of (proj . field), demodulated by exp(-i(kx x+ky y)),
    at each z. `proj` is a length-3 weight vector projecting the vector field onto the
    polarization of interest (the s-pol unit vector, or (1,0,0) for tangential Ex). The
    demod removes the transverse Bloch phase so the cell-average IS the 0-order Fourier
    coefficient; kx=ky=0 (envelope formulation) leaves it unchanged."""
    # cell-centred probe grid: offset off the x=0 / y=0 periodic-boundary lines (where
    # quasi-periodic point evaluation can fail) by half a step (audit OPT-7).
    xs = (np.arange(6) + 0.5) * (Px / 6.0)
    ys = (np.arange(6) + 0.5) * (Py / 6.0)
    p0, p1, p2 = proj
    out = []
    for zv in z_probes:
        vals = []
        for xv in xs:
            for yv in ys:
                try:
                    E = field(mesh(float(xv), float(yv), float(zv)))
                    proj_val = p0 * complex(E[0]) + p1 * complex(E[1]) + p2 * complex(E[2])
                    vals.append(proj_val * np.exp(-1j * (kx * xv + ky * yv)))
                except Exception:
                    pass
        if not vals:
            # A whole z-plane with zero valid samples means the probe points all fell
            # outside the mesh (a geometry/units contract break) -- fail loudly instead
            # of feeding a silent 0+0j into the least-squares R/T fit (audit OPT-7/F6).
            raise RuntimeError(
                "cell-average got no valid field samples at z={:.3f} nm; the probe grid "
                "missed the mesh (check the geometry/units and z-interval bounds).".format(
                    float(zv)))
        out.append(complex(np.mean(vals)))
    return np.array(out)


def _reflection(mesh, gfu, kz_s, proj, kx, ky, geo: OpticalGeometry) -> complex:
    """0-order reflection: least-squares fit of the cell-averaged (proj-projected,
    demodulated) scattered field over z-planes in the superstrate buffer."""
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z_struct_top = geo.z_intervals_nm["superstrate"][0]
    z_air_top = geo.z_intervals_nm["superstrate"][1]
    z_lo = z_struct_top + 50.0
    z_hi = z_air_top - 50.0
    if z_hi <= z_lo:
        z_lo = z_struct_top + 0.2 * (z_air_top - z_struct_top)
        z_hi = z_struct_top + 0.8 * (z_air_top - z_struct_top)
    z_probes = np.linspace(z_lo, z_hi, 7)
    Es = _cell_average(mesh, gfu, z_probes, Px, Py, proj, kx, ky)
    # upward (reflected) exp(+i kz_s z) + residual downward exp(-i kz_s z)
    M = np.column_stack([np.exp(+1j * kz_s * z_probes), np.exp(-1j * kz_s * z_probes)])
    coeffs, *_ = np.linalg.lstsq(M, Es, rcond=None)
    return complex(coeffs[0])


def _transmission(mesh, gfu, kz_sub, proj, kx, ky, geo: OpticalGeometry):
    """0-order SCATTERED transmission amplitude: fit the cell-averaged scattered field
    in the substrate buffer to a downward wave exp(-i kz_sub z). The background
    transmission T0 is added analytically by the caller. Returns None if there is no
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
    Es = _cell_average(mesh, gfu, z_probes, Px, Py, proj, kx, ky)
    # downward (transmitted) exp(-i kz_sub z) + any upward residual exp(+i kz_sub z)
    M = np.column_stack([np.exp(-1j * kz_sub * z_probes), np.exp(+1j * kz_sub * z_probes)])
    coeffs, *_ = np.linalg.lstsq(M, Es, rcond=None)
    return complex(coeffs[0])


def _ppol_extract(mesh, E_tot, kz_s, kz_sub, kx, geo: OpticalGeometry, eps_sup, eps_sub):
    """p-pol R/T from the reconstructed TOTAL field (E_bg + scattered) tangential Ex.
    R = |Ex_up/Ex_down|^2 in the superstrate (the eps/kz Poynting factors cancel in the
    same medium); T = |Ex_down_sub/Ex_down_super|^2 * Re((eps_sub/eps_sup)(kz_s/kz_sub))
    -- the p-pol z-flux Sz ~ |Ex|^2 eps/kz. Returns (r, R, t, T)."""
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z0, z1 = geo.z_intervals_nm["superstrate"]
    zlo, zhi = z0 + 50.0, z1 - 50.0
    if zhi <= zlo:
        zlo, zhi = z0 + 0.2 * (z1 - z0), z0 + 0.8 * (z1 - z0)
    zr = np.linspace(zlo, zhi, 7)
    Exr = _cell_average(mesh, E_tot, zr, Px, Py, (1.0, 0.0, 0.0), kx, 0.0)   # tangential Ex (p-pol: phi=0)
    Mr = np.column_stack([np.exp(-1j * kz_s * zr), np.exp(+1j * kz_s * zr)])
    cr, *_ = np.linalg.lstsq(Mr, Exr, rcond=None)
    a_d, a_u = complex(cr[0]), complex(cr[1])                     # incident-dir, reflected-dir Ex
    r = a_u / a_d if abs(a_d) > 1e-30 else 0j
    R = float(abs(r) ** 2)
    if "substrate" not in geo.z_intervals_nm:
        return r, R, None, None
    zs_lo, zs_hi = geo.z_intervals_nm["substrate"]
    pad = 0.1 * (zs_hi - zs_lo)
    if zs_hi - pad <= zs_lo + pad:
        return r, R, None, None
    zt = np.linspace(zs_lo + pad, zs_hi - pad, 7)
    Ext = _cell_average(mesh, E_tot, zt, Px, Py, (1.0, 0.0, 0.0), kx, 0.0)
    Mt = np.column_stack([np.exp(-1j * kz_sub * zt), np.exp(+1j * kz_sub * zt)])
    ct, *_ = np.linalg.lstsq(Mt, Ext, rcond=None)
    t = complex(ct[0]) / a_d if abs(a_d) > 1e-30 else 0j          # transmitted Ex / incident Ex
    factor = (eps_sub / eps_sup) * (kz_s / kz_sub)
    T = float(abs(t) ** 2 * factor.real)
    return r, R, t, T


def _absorbed_fraction(mesh, E_tot, eps_cf, k0, theta, Px, Py):
    """Independently measured absorbed fraction (audit OPT-2): the normalized
    volumetric loss integral A = k0 * Int_V Im(eps) |E|^2 dV / (cos(theta) * cell_area),
    over the PHYSICAL (non-PML) domain, for a unit-amplitude incident plane wave. This
    is a genuine measurement (not 1-R-T): comparing it to the budget closure 1-R-T
    catches energy/numerics errors that the R/T extraction alone cannot. The PML
    materials are excluded -- their stretched-coordinate eps would corrupt the integral
    (and a lossy substrate makes the bottom-PML contribution spurious). NOTE: a lossy
    super/substrate BUFFER (between the structure and its PML) is still integrated, so A is
    a clean measurement only for LOSSLESS cladding media (the validated cases); for a lossy
    cladding treat A as qualitative (audit OS-4)."""
    # Exclude PML regions only. Use the 'pml_' prefix (the builder names PML 'pml_top'/
    # 'pml_bot') so a physical material like 'pmlayer' is NOT dropped (OS-3), and re.escape
    # each name so a regex-metacharacter material name (e.g. 'ito.n+') is matched literally
    # rather than silently missed by mesh.Materials' regex (OS-2).
    non_pml = [m for m in dict.fromkeys(mesh.GetMaterials()) if not m.startswith("pml_")]
    if not non_pml:
        return None
    defon = mesh.Materials("|".join(re.escape(m) for m in non_pml))
    im_eps = (eps_cf - ng.Conj(eps_cf)) / 2j                  # Im(eps) as a real CF
    e2 = ng.InnerProduct(E_tot, ng.Conj(E_tot))               # |E|^2 (real)
    integ = ng.Integrate(im_eps * e2, mesh, definedon=defon)
    area = float(Px) * float(Py)
    if area <= 0:
        return None
    return float(complex(integ).real * k0 / (max(math.cos(theta), 1e-12) * area))
