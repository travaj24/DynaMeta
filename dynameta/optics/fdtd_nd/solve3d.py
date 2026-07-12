"""3D solve front-ends: grid fill, coefficient builders, dispatch, R/T extraction.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from dynameta.constants import C_LIGHT, EPS0
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd.backends import _resolve_backend
from dynameta.optics.fdtd_nd.results import FDTD2DObliqueResult, FDTD3DMOResult, FDTD3DResult, _flux3d
from dynameta.optics.fdtd_nd.cpml import _cpml_z
from dynameta.optics.fdtd_nd.kernels3d import _run_3d, _run_3d_mo, _run_3d_oblique
from dynameta.optics.fdtd_nd.kernels3d_numba import _run_3d_oblique_numba, _te3d_numba
from dynameta.optics.fdtd_nd.kernels3d_jax import _run_3d_jax, _run_3d_oblique_jax

def _dispatch_3d(name, eps_inf, wp, gam, chi3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np,
                 lor=None, chi2=None, raman=None, gain=None):
    """Run ONE 3D pass on the named backend, returning the eight probe planes as NumPy arrays (so the
    downstream FFT / R-T extraction is backend-agnostic). 'numba' = the fused threaded CPU kernel (the
    fast 3D path); 'numpy'/'cupy' = the vectorized reference loop. `lor`=(C1,C2,C3) per-cell Lorentz ADE
    coefficients or None. (The jax 3D kernel does not carry the Lorentz ADE yet -> guarded upstream.)
    chi2/raman/gain (R15/R20) run on the numpy/cupy vectorized path only in 3D; numba/jax raise."""
    (ke, be, ce), (kh, bh, ch) = cpml
    nonlinear3 = chi2 is not None or raman is not None or gain is not None
    if nonlinear3 and name not in ("numpy", "cupy"):
        raise NotImplementedError("3D chi2/Raman/gain run on backend='numpy' (or cupy) only; got "
                                  "backend={!r}".format(name))
    if name == "numba":
        has_lor = lor is not None
        z = np.zeros_like(eps_inf)
        C1, C2, C3 = (lor if has_lor else (z, z, z))
        return _te3d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dy, dz, dt,
                           nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor)
    if name == "jax":
        if lor is not None:
            raise NotImplementedError("the jax 3D backend does not carry the Lorentz ADE yet; use "
                                      "backend='numba' or 'numpy' for a 3D Lorentz material.")
        out = _run_3d_jax(eps_inf, wp, gam, chi3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml)
        return tuple(np.asarray(v) for v in out)            # JAX arrays -> NumPy for the FFT/R-T stage
    if name == "cupy" and xp is np:
        import cupy as xp
    a = tuple(xp.asarray(v) for v in (eps_inf, wp, gam, chi3))
    out = _run_3d(*a, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, xp.asarray(src), cpml, xp, lor,
                  chi2=chi2, raman=raman, gain=gain)
    to_np = (lambda v: np.asarray(v.get()) if hasattr(v, "get") else np.asarray(v))
    return tuple(to_np(v) for v in out)


def solve_fdtd_3d(layers: List[FDTDLayer], *, period_x_m: float, period_y_m: float,
                  nx: Optional[int] = None, ny: Optional[int] = None,
                  lateral_eps_inf: Optional[np.ndarray] = None,
                  lambda_min_m: float, lambda_max_m: float, resolution: int = 24,
                  courant: float = 0.5, n_pad_wave: float = 4.0, settle: float = 12.0,
                  kerr: bool = False, source_amp: float = 1.0, npml: int = 12,
                  n_super: float = 1.0, n_sub: float = 1.0,
                  backend: str = "numpy", xp=np) -> FDTD3DResult:
    """Broadband R(f)/T(f) of a doubly-periodic (period_x_m x period_y_m) unit cell at NORMAL incidence,
    y-polarized. `layers` = the through-stack (z) profile; supply `lateral_eps_inf` (an (nx,ny,nz) array,
    or a callable(nx,ny,nz,zc,pad,zstruct)->(nx,ny,nz)) to make a 2D-periodic structure, else the stack is
    laterally UNIFORM (and the result reduces to 1D/TMM). Returns both the specular 0-order and the
    total-flux (all (kx,ky) orders) R/T.

    n_super / n_sub (default 1 = vacuum): the lossless semi-infinite superstrate / substrate indices
    (metasurface-on-glass), filling the z-pads with n_super^2 / n_sub^2, with an impedance-matched CPML
    and a homogeneous-superstrate incident reference (T carries the n_sub/n_super flux ratio). Reduces
    byte-identically to vacuum at n_super=n_sub=1.

    backend: 'auto'/'numba' (the fused threaded CPU kernel = the fast 3D path), 'numpy' (reference),
    'jax' (the differentiable XLA scan, for 3D inverse design), or 'cupy'/xp for the GPU. A Lorentz pole
    (lorentz_delta_eps) is carried by the numpy/numba/cupy 3D kernels (per E-component); the jax 3D backend
    does not yet, and raises if a Lorentz layer is run on it."""
    if abs(complex(n_super).imag) > 1e-9 or abs(complex(n_sub).imag) > 1e-9:   # mirror the FEM/2D guard
        raise NotImplementedError("solve_fdtd_3d: R/T and the energy budget are defined only for LOSSLESS "
                                  "end media (Im(n)=0); got n_super={}, n_sub={}.".format(n_super, n_sub))
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)

    def _n_band_max(L):
        return max(abs(np.sqrt(L.eps_at(w))) for w in w_band)
    n_max = max(1.0, n_super, n_sub, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    if nx is None:
        nx = max(4, int(round(period_x_m / dz)))
    if ny is None:
        ny = max(4, int(round(period_y_m / dz)))
    dx = period_x_m / nx
    dy = period_y_m / ny
    # 3D CFL: dt <= courant / (c sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2))

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1

    shape = (nx, ny, nz)
    eps_inf = np.ones(shape); wp = np.zeros(shape); gam = np.zeros(shape); chi3 = np.zeros(shape)
    lw0 = np.zeros(shape); lgam = np.zeros(shape); ldeps = np.zeros(shape)   # Lorentz pole per cell
    chi2g = np.zeros(shape)                                                  # R15 SHG chi2 [m/V]
    chi3R = np.zeros(shape); rw = np.zeros(shape); rgam = np.zeros(shape)    # R15 Raman pole
    gw = np.zeros(shape); gdw = np.zeros(shape); gkdn = np.zeros(shape)      # R20 gain line
    zc = (np.arange(nz) + 0.5) * dz
    eps_inf[:, :, zc < pad] = n_super ** 2                   # fill the semi-infinite super/substrate pads
    eps_inf[:, :, zc >= pad + z_struct] = n_sub ** 2
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, :, m] = L.eps_inf
        wp[:, :, m] = L.drude_wp_rad_s
        gam[:, :, m] = L.drude_gamma_rad_s
        lw0[:, :, m] = L.lorentz_w0_rad_s
        lgam[:, :, m] = L.lorentz_gamma_rad_s
        ldeps[:, :, m] = L.lorentz_delta_eps
        if kerr:
            chi3[:, :, m] = L.chi3_m2_V2
        chi2g[:, :, m] = L.chi2_m_V
        chi3R[:, :, m] = L.raman_chi3_m2_V2
        rw[:, :, m] = L.raman_w_rad_s
        rgam[:, :, m] = L.raman_gamma_rad_s
        gw[:, :, m] = L.gain_w_rad_s
        gdw[:, :, m] = L.gain_dw_rad_s
        gkdn[:, :, m] = L.gain_kappa_C2_kg * L.gain_dN_m3
        z += L.thickness_m
    if lateral_eps_inf is not None:
        lat = lateral_eps_inf(nx, ny, nz, zc, pad, z_struct) if callable(lateral_eps_inf) else np.asarray(lateral_eps_inf)
        eps_inf = np.asarray(lat, dtype=float)
        # GRID-SIZING GUARD: dz was derived from `layers` (+ end media) before this override; a
        # higher-index lateral pattern would be silently under-resolved. Raise rather than mis-solve --
        # size `layers` eps_inf to the lateral pattern max (the make_structured_lateral seam does this).
        _n_lat = float(np.sqrt(max(1.0, float(np.max(np.real(eps_inf))))))
        if _n_lat > n_max * (1.0 + 1e-9):
            raise NotImplementedError(
                "solve_fdtd_3d: lateral pattern peak index {:.3f} exceeds the grid-sizing index {:.3f} "
                "from `layers` (+ end media) -- dz is under-resolved by {:.0%}. Size the `layers` "
                "eps_inf to the lateral pattern max so n_max/dz are derived from it.".format(
                    _n_lat, n_max, _n_lat / n_max - 1.0))

    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))

    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    # Lorentz ADE coefficients (central difference, per E-component); applied to the STRUCTURE run only.
    # d_eps=0 everywhere -> lor=None -> the path is byte-identical to the no-Lorentz solve.
    lor = None
    if np.any(ldeps != 0.0):
        den = 1.0 + lgam * dt / 2.0
        C1 = (2.0 - lw0 ** 2 * dt ** 2) / den
        C2 = (lgam * dt / 2.0 - 1.0) / den
        C3 = (EPS0 * ldeps * lw0 ** 2 * dt ** 2) / den
        lor = (C1, C2, C3)

    # R15/R20 nonlinear grids -> coefficient tuples (all-zero -> None -> pre-R15 path)
    chi2_arrs = chi2g if np.any(chi2g != 0.0) else None
    raman_arrs = None
    if np.any(chi3R != 0.0):
        if np.any((chi3R != 0.0) & (rw <= 0.0)):
            raise ValueError("Raman chi3 needs raman_w_rad_s > 0 on every Raman-active layer")
        den_r = 1.0 + rgam * dt / 2.0
        raman_arrs = ((2.0 - rw ** 2 * dt ** 2) / den_r, (rgam * dt / 2.0 - 1.0) / den_r,
                      (rw ** 2 * dt ** 2) / den_r, chi3R)
    gain_arrs = None
    if np.any(gkdn != 0.0):
        if np.any((gkdn != 0.0) & ((gw <= 0.0) | (gdw <= 0.0))):
            raise ValueError("gain line needs gain_w_rad_s > 0 and gain_dw_rad_s > 0 on every "
                             "gain-active layer")
        den_g = 1.0 + gdw * dt / 2.0
        gain_arrs = ((2.0 - gw ** 2 * dt ** 2) / den_g, (gdw * dt / 2.0 - 1.0) / den_g,
                     (-gkdn * dt ** 2) / den_g)
    cpml_struct = _cpml_z(nz, dz, dt, npml, n_super, n_sub)  # PML matched super (low z) + sub (high z)
    cpml_ref = _cpml_z(nz, dz, dt, npml, n_super, n_super)   # homogeneous-superstrate reference
    name = _resolve_backend(backend)                        # 'auto'/'cpu' -> numba (the fast 3D path)
    one = np.ones(shape); zero = np.zeros(shape)

    def run(ei, w, g_, c3, cpml, lor=None, chi2=None, raman=None, gain=None):
        return _dispatch_3d(name, ei, w, g_, c3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp,
                            lor, chi2, raman, gain)

    exL_i, eyL_i, hxL_i, hyL_i, exR_i, eyR_i, hxR_i, hyR_i = run(
        n_super ** 2 * one, zero, zero, zero, cpml_ref)                  # homogeneous-superstrate reference
    exL_t, eyL_t, hxL_t, hyL_t, exR_t, eyR_t, hxR_t, hyR_t = run(eps_inf, wp, gam, chi3, cpml_struct, lor,
                                                                 chi2_arrs, raman_arrs, gain_arrs)  # struct

    f = np.fft.rfftfreq(nsteps, dt)
    # 0-order specular co-pol (E_y) from the x,y-MEAN field (== the 1D two-run method)
    mL_inc = np.fft.rfft(eyL_i.mean(axis=(1, 2))); mR_inc = np.fft.rfft(eyR_i.mean(axis=(1, 2)))
    mRefl = np.fft.rfft((eyL_t - eyL_i).mean(axis=(1, 2))); mTrans = np.fft.rfft(eyR_t.mean(axis=(1, 2)))
    k0 = 2.0 * np.pi * f / C_LIGHT
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(mRefl / mL_inc) ** 2
        T0 = np.abs(mTrans / mR_inc) ** 2 * (n_sub / n_super)   # Snell power-flux ratio (incident in super)
        # COMPLEX co-pol 0-order coeffs: conjugate (rfft is exp(+iwt); convention is exp(-iwt)), then
        # de-embed the propagation phase (superstrate phase velocity c/n_super): r0c to the front face
        # z=pad (probe k_pL). t0c (audit C3-4): interface-referenced t carries n_sub*z_struct plus the
        # (n_super-n_sub) mismatch over the face->probe distance (the incident reference travels
        # n_super all the way); the old bare exp(1j*k0*z_struct) was vacuum-only. Byte-identical at
        # n_super=n_sub=1.
        r0c = np.conj(mRefl / mL_inc) * np.exp(-2j * n_super * k0 * (pad - k_pL * dz))
        t0c = np.conj(mTrans / mR_inc) * np.exp(1j * k0 * (n_sub * z_struct
                                                           + (n_super - n_sub) * (k_pR * dz - pad)))
    # total R/T from the full Poynting flux (all (kx,ky) diffraction orders)
    P_inc = _flux3d(exL_i, eyL_i, hxL_i, hyL_i)
    P_refl = _flux3d(exL_t - exL_i, eyL_t - eyL_i, hxL_t - hxL_i, hyL_t - hyL_i)
    P_trans = _flux3d(exR_t, eyR_t, hxR_t, hyR_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_flux = np.abs(P_refl) / np.abs(P_inc)
        T_flux = np.abs(P_trans) / np.abs(P_inc)
    band = (f >= f_min) & (f <= f_max) & (np.abs(mL_inc) > 0.05 * np.max(np.abs(mL_inc)))
    return FDTD3DResult(freqs_Hz=f, R0=R0, T0=T0, R_flux=R_flux, T_flux=T_flux, band=band, r0=r0c, t0=t0c)


def solve_fdtd_3d_oblique(layers: List[FDTDLayer], *, period_x_m: float, period_y_m: float,
                          angle_deg: float, azimuth_deg: float = 0.0,
                          lambda_min_m: float, lambda_max_m: float, resolution: int = 36,
                          courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                          source_amp: float = 1.0, npml: int = 12, nx: int = 6,
                          ny: int = 6, backend: str = "numpy") -> FDTD2DObliqueResult:
    """Broadband s-pol reflectance/transmittance of a laterally-uniform stack at OBLIQUE incidence in the
    FULL-VECTOR 3D engine, via the complex-envelope Bloch method with a 2D transverse wavevector
    k_par=(kx,ky), |k_par| = (2 pi/lambda_c) sin(angle_deg), azimuth phi=azimuth_deg (kx=|k_par|cos phi,
    ky=|k_par|sin phi). The s-pol E-vector (-sin phi, cos phi, 0) is injected; the R/T come from its
    x,y-mean projection. Fixed k_par -> theta(f)=asin(k_par c/w). For a uniform stack the result is
    azimuth-INVARIANT and equals tmm(theta(f),'s') -- this exercises the genuine 2D transverse Bloch
    envelope (kx AND ky). Vacuum ends; Drude only. angle_deg=0 reduces to normal incidence."""
    if any(L.lorentz_delta_eps != 0.0 for L in layers):
        raise NotImplementedError("solve_fdtd_3d_oblique supports Drude dispersion only (no Lorentz pole).")
    # audit C5-7: the oblique kernel carries NO chi3/chi2/Raman/gain ADEs -- refuse rather
    # than silently solve the passive stack (the 1-D entry point raises for the same set)
    _dropped = [t for t in ("chi3_m2_V2", "chi2_m_V", "raman_chi3_m2_V2", "gain_dN_m3")
                if any(getattr(L, t, 0.0) != 0.0 for L in layers)]
    if _dropped:
        raise NotImplementedError(
            "solve_fdtd_3d_oblique: the oblique kernel carries no {} terms -- they would be "
            "silently ignored (audit C5-7); use the normal-incidence solver or split the "
            "problem.".format("/".join(_dropped)))
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)
    n_max = max(1.0, max(max(abs(np.sqrt(L.eps_at(w))) for w in w_band) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    dx = period_x_m / nx; dy = period_y_m / ny
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2))
    k_par = (2.0 * np.pi * f_c / C_LIGHT) * np.sin(np.radians(angle_deg))
    phi = np.radians(azimuth_deg)
    kx, ky = k_par * np.cos(phi), k_par * np.sin(phi)
    sx, sy = -np.sin(phi), np.cos(phi)                           # s-pol in-plane E direction

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1
    eps_inf = np.ones((nx, ny, nz)); wp = np.zeros((nx, ny, nz)); gam = np.zeros((nx, ny, nz))
    zc = (np.arange(nz) + 0.5) * dz
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, :, m] = L.eps_inf; wp[:, :, m] = L.drude_wp_rad_s; gam[:, :, m] = L.drude_gamma_rad_s
        z += L.thickness_m
    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))
    cpml = _cpml_z(nz, dz, dt, npml)
    one = np.ones((nx, ny, nz)); zero = np.zeros((nx, ny, nz))
    bk = _resolve_backend(backend)

    def _obl3d(ei, w, g):
        if bk == "numba":
            (ke, be, ce), (kh, bh, ch) = cpml
            return _run_3d_oblique_numba(np.asarray(ei, float), np.asarray(w, float), np.asarray(g, float),
                                         ke, be, ce, kh, bh, ch, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR,
                                         np.asarray(src, float), kx, ky, sx, sy)
        if bk == "jax":
            out = _run_3d_oblique_jax(ei, w, g, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml,
                                      kx, ky, sx, sy)
            return tuple(np.asarray(v) for v in out)        # JAX -> NumPy for the FFT/R-T stage
        return _run_3d_oblique(ei, w, g, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, kx, ky, sx, sy)

    exL_i, eyL_i, exR_i, eyR_i = _obl3d(one, zero, zero)
    exL_t, eyL_t, exR_t, eyR_t = _obl3d(eps_inf, wp, gam)
    nf = nsteps // 2 + 1
    f = np.fft.rfftfreq(nsteps, dt)
    proj = (lambda ex, ey: sx * ex.mean(axis=(1, 2)) + sy * ey.mean(axis=(1, 2)))   # s-pol of the x,y-mean
    meanf = (lambda a: np.fft.fft(a)[:nf])
    inc_L = meanf(proj(exL_i, eyL_i)); inc_R = meanf(proj(exR_i, eyR_i))
    refl = meanf(proj(exL_t - exL_i, eyL_t - eyL_i)); trans = meanf(proj(exR_t, eyR_t))
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(refl / inc_L) ** 2
        T0 = np.abs(trans / inc_R) ** 2
    sin_t = np.divide(k_par * C_LIGHT, 2.0 * np.pi * np.maximum(f, 1e-30))
    theta = np.degrees(np.arcsin(np.clip(sin_t, -1.0, 1.0)))
    band = (f >= f_min) & (f <= f_max) & (sin_t < 0.999) & (np.abs(inc_L) > 0.05 * np.max(np.abs(inc_L)))
    return FDTD2DObliqueResult(freqs_Hz=f, theta_deg=theta, R0=R0, T0=T0, band=band)






def solve_fdtd_3d_mo(layers, *, period_x_m: float, period_y_m: float, lambda_min_m: float,
                     lambda_max_m: float, resolution: int = 40, courant: float = 0.5,
                     n_pad_wave: float = 6.0, settle: float = 14.0, pol: str = "y",
                     source_amp: float = 1.0, nx: int = 4, ny: int = 4, npml: int = 12,
                     lateral_tensor=None) -> FDTD3DMOResult:
    """Broadband co/cross R/T + Faraday rotation of a 3D anisotropic / magneto-optic stack (the full-vector
    gyrotropic engine, _run_3d_mo). `layers` are MO layers (duck-typed: .thickness_m, .eps_xx, .eps_yy,
    optional .eps_zz, .drude_wp_rad_s, .drude_gamma_rad_s, .cyclotron_wc_rad_s); the magnetized-Drude
    cyclotron term gives the gyrotropy. For a laterally-uniform stack the result reduces to the 1-D
    fdtd_mo (validated vs the circular-eigenmode Jones-TMM); nx,ny small. Vacuum ends, convention
    exp(-i w t).

    STRUCTURED (laterally-patterned) TENSOR: pass `lateral_tensor` to make a 3-D patterned anisotropic
    metasurface (the per-layer eps_xx/eps_yy/eps_zz are the laterally-UNIFORM default; lateral_tensor
    OVERRIDES them cell-by-cell). It is either an (nx,ny,nz)->keyed dict {'exx','eyy','ezz'[, 'wp','gam',
    'wc']} of arrays, or a callable lateral_tensor(nx, ny, nz, zc, pad, z_struct) returning that dict;
    any key present overwrites the corresponding field in the structure region. With wc==0 everywhere the
    engine is a plain per-cell DIAGONAL-anisotropic Yee solve (the cyclotron 2x2 collapses to diagonal),
    so this is the structured diagonal-tensor 3-D FDTD; nx,ny must resolve the lateral pattern. Reduces to
    the scalar solve_fdtd_3d when exx==eyy==ezz, and to the 1-D anisotropic TMM when laterally uniform."""
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)

    def _ncell(L):
        wpL = getattr(L, "drude_wp_rad_s", 0.0)
        if wpL > 0:                                              # circular-mode index bound for grid sizing
            # audit C3-3: max over BOTH circular branches (w -/+ wc) -- the single (w - wc)
            # branch is resonant only for wc > 0, silently under-sizing dz for reversed
            # magnetization; floor by the background birefringent indices too.
            eps_inf = 0.5 * (L.eps_xx + L.eps_yy)
            wcL = L.cyclotron_wc_rad_s
            n_circ = max(abs(np.sqrt(eps_inf - wpL ** 2 / (w * (w - s * wcL) + 1j * w * L.drude_gamma_rad_s)))
                         for w in w_band for s in (+1, -1))
            return max(n_circ, np.sqrt(L.eps_xx), np.sqrt(L.eps_yy), 1.0)
        return max(np.sqrt(L.eps_xx), np.sqrt(L.eps_yy), 1.0)
    n_max = max(1.0, max(_ncell(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    dx = period_x_m / nx; dy = period_y_m / ny
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2))
    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1
    o = np.ones((nx, ny, nz)); zr = np.zeros((nx, ny, nz))
    exx, eyy, ezz = o.copy(), o.copy(), o.copy()
    wp, gam, wc = zr.copy(), zr.copy(), zr.copy()
    zc = (np.arange(nz) + 0.5) * dz
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        exx[:, :, m] = L.eps_xx; eyy[:, :, m] = L.eps_yy
        _ezz = getattr(L, "eps_zz", None)        # `is not None`, NOT truthiness: eps_zz=0 (z-ENZ) is valid
        ezz[:, :, m] = _ezz if _ezz is not None else 0.5 * (L.eps_xx + L.eps_yy)
        wp[:, :, m] = getattr(L, "drude_wp_rad_s", 0.0)
        gam[:, :, m] = getattr(L, "drude_gamma_rad_s", 0.0)
        wc[:, :, m] = getattr(L, "cyclotron_wc_rad_s", 0.0)
        z += L.thickness_m
    if lateral_tensor is not None:
        # STRUCTURED override: a laterally-patterned per-cell tensor replaces the uniform per-layer fill.
        pat = lateral_tensor(nx, ny, nz, zc, pad, z_struct) if callable(lateral_tensor) else lateral_tensor
        fields = {"exx": exx, "eyy": eyy, "ezz": ezz, "wp": wp, "gam": gam, "wc": wc}
        for key, arr in dict(pat).items():
            if key not in fields:
                raise ValueError("lateral_tensor key {!r} not in {}".format(key, sorted(fields)))
            a = np.asarray(arr, dtype=float)
            if a.shape != (nx, ny, nz):
                raise ValueError("lateral_tensor[{!r}] shape {} != (nx,ny,nz)={}".format(
                    key, a.shape, (nx, ny, nz)))
            fields[key][...] = a
        # GRID-SIZING GUARD: dz was derived from `layers` before this tensor override; a higher-index
        # lateral pattern would be silently under-resolved. Use the TRANSVERSE index (exx/eyy) only --
        # matching _ncell, which sets dz from eps_xx/eps_yy (the transverse E a z-propagating wave
        # sees); eps_zz drives Ez, not the dz resolution, so including it would false-trip on a
        # faithful high-eps_zz override. Raise rather than mis-solve -- size `layers` eps_xx/eps_yy to
        # the lateral pattern max.
        _n_lat = float(np.sqrt(max(1.0, float(np.max(exx)), float(np.max(eyy)))))
        if _n_lat > n_max * (1.0 + 1e-9):
            raise NotImplementedError(
                "solve_fdtd_3d_mo: lateral_tensor transverse peak index {:.3f} exceeds the grid-sizing "
                "index {:.3f} from `layers` -- dz is under-resolved by {:.0%}. Size the `layers` "
                "eps_xx/eps_yy to the lateral pattern max so n_max/dz are derived from it.".format(
                    _n_lat, n_max, _n_lat / n_max - 1.0))
    k_src = max(2, int(round(0.35 * pad / dz)))
    k_pL = int(round(0.7 * pad / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + 4.0 * Lz / C_LIGHT + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))
    cpml = _cpml_z(nz, dz, dt, npml)
    exL_i, eyL_i, exR_i, eyR_i = _run_3d_mo(o, o, o, zr, zr, zr, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR,
                                            src, cpml, pol)                                  # vacuum
    exL_t, eyL_t, exR_t, eyR_t = _run_3d_mo(exx, eyy, ezz, wp, gam, wc, dx, dy, dz, dt, nsteps, k_src,
                                            k_pL, k_pR, src, cpml, pol)                       # structure
    f = np.fft.rfftfreq(nsteps, dt)
    m_ = (lambda a: np.conj(np.fft.rfft(a.mean(axis=(1, 2)))))   # x,y-mean 0-order; conj -> exp(-iwt)
    if pol == "y":
        coL_i, coR_i, coL_t, coR_t, crL_t, crR_t = eyL_i, eyR_i, eyL_t, eyR_t, exL_t, exR_t
    else:
        coL_i, coR_i, coL_t, coR_t, crL_t, crR_t = exL_i, exR_i, exL_t, exR_t, eyL_t, eyR_t
    inc_L, inc_R = m_(coL_i), m_(coR_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_co = m_(coL_t - coL_i) / inc_L; r_cr = m_(crL_t) / inc_L
        t_co = m_(coR_t) / inc_R; t_cr = m_(crR_t) / inc_R
        R = np.abs(r_co) ** 2 + np.abs(r_cr) ** 2
        T = np.abs(t_co) ** 2 + np.abs(t_cr) ** 2
        far = 0.5 * np.arctan2(2.0 * np.real(t_co * np.conj(t_cr)), np.abs(t_co) ** 2 - np.abs(t_cr) ** 2)
    band = (f >= f_min) & (f <= f_max) & (np.abs(inc_L) > 0.05 * np.max(np.abs(inc_L)))
    return FDTD3DMOResult(freqs_Hz=f, band=band, t_co=t_co, t_cross=t_cr, r_co=r_co, r_cross=r_cr,
                          R=R, T=T, faraday_deg=np.degrees(far))
