"""2D solve front-ends: grid fill, coefficient builders, dispatch, R/T extraction.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from dynameta.constants import C_LIGHT, EPS0
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd.backends import _HAVE_NUMBA, _have_jax, _resolve_backend
from dynameta.optics.fdtd_nd.kernels2d_numba import _te2d_cuda, _te2d_numba
from dynameta.optics.fdtd_nd.results import FDTD2DObliqueResult, FDTD2DResult, _flux
from dynameta.optics.fdtd_nd.cpml import _cpml_z
from dynameta.optics.fdtd_nd.kernels2d import _run_2d_te
from dynameta.optics.fdtd_nd.kernels2d_jax import _run_2d_te_jax
from dynameta.optics.fdtd_nd.oblique2d import _run_oblique



def _ring_time_s(layers) -> float:
    """Material-memory ring-down time (audit C3-6): the fixed 200*tau DFT window predates
    the Lorentz/gain ADEs -- a high-loaded-Q in-band pole rings past it, truncating the
    rfft with O(0.1) silent R0/T0 bias (probe: |dT0| = 0.102 vs the TMM oracle for a
    Q~600 line, no warning possible since the band mask checks excitation only). Returns
    the (2/Gamma) ln(1/1e-4) ~ 18.4/Gamma memory of the NARROWEST active Lorentz/gain
    pole (0.0 when no pole is active -> the legacy window, byte-identical)."""
    t_ring = 0.0
    for L in layers:
        if getattr(L, "lorentz_delta_eps", 0.0) != 0.0 and getattr(L, "lorentz_gamma_rad_s", 0.0) > 0.0:
            t_ring = max(t_ring, 18.4 / float(L.lorentz_gamma_rad_s))
        if getattr(L, "gain_dN_m3", 0.0) != 0.0 and getattr(L, "gain_dw_rad_s", 0.0) > 0.0:
            t_ring = max(t_ring, 18.4 / float(L.gain_dw_rad_s))
    return t_ring


def _dispatch_2d_te(name, eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np,
                    lor=None, chi2=None, raman=None, gain=None):
    """Run ONE 2D-TE pass on the named backend and return the four probe x-lines as NumPy arrays, so the
    downstream FFT / R-T extraction stays backend-agnostic. 'numba' = the fused threaded CPU kernel;
    'jax' = the differentiable XLA scan; 'numpy'/'cupy' = the vectorized reference loop on the chosen
    array module (an explicit power-user `xp` is honored even for 'numpy', preserving the old xp=cupy API).
    `lor` = (C1,C2,C3) per-cell Lorentz ADE coefficients or None (no Lorentz pole). chi2/raman/gain
    (R15/R20) run on EVERY backend: the GPU kernels carry the same cell-local recurrences
    (numba-cuda in the cooperative kernel; cupy through the xp-parameterized reference loop),
    validated GPU==CPU in validation/fdtd_gpu_nonlinear.py. None keeps every backend
    byte-identical."""
    (ke, be, ce), (kh, bh, ch) = cpml
    if name in ("numba", "numba-cuda"):
        has_lor = lor is not None
        z = np.zeros_like(eps_inf)
        C1, C2, C3 = (lor if has_lor else (z, z, z))
        chi2g = chi2 if chi2 is not None else z
        R1, R2, R3, chi3R = (raman if raman is not None else (z, z, z, z))
        G1, G2, G3 = (gain if gain is not None else (z, z, z))
        if name == "numba-cuda":
            return _te2d_cuda(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
                              nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor,
                              chi2g, chi2 is not None, R1, R2, R3, chi3R, raman is not None,
                              G1, G2, G3, gain is not None)
        return _te2d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
                           nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor,
                           chi2g, chi2 is not None, R1, R2, R3, chi3R, raman is not None,
                           G1, G2, G3, gain is not None)
    if name == "jax":
        out = _run_2d_te_jax(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml,
                             lor, chi2=chi2, raman=raman, gain=gain)
        return tuple(np.asarray(v) for v in out)            # JAX arrays -> NumPy for the FFT/R-T stage
    if name == "cupy" and xp is np:
        import cupy as xp                                    # backend='cupy' auto-selects the device module
    a = tuple(xp.asarray(v) for v in (eps_inf, wp, gam, chi3))
    out = _run_2d_te(*a, dx, dz, dt, nsteps, k_src, k_pL, k_pR, xp.asarray(src), cpml, xp, lor,
                     chi2=chi2, raman=raman, gain=gain)
    to_np = (lambda v: np.asarray(v.get()) if hasattr(v, "get") else np.asarray(v))
    return tuple(to_np(v) for v in out)


def solve_fdtd_2d(layers: List[FDTDLayer], *, period_x_m: float, nx: Optional[int] = None,
                  lateral_eps_inf: Optional[np.ndarray] = None,
                  lateral_wp: Optional[np.ndarray] = None, lateral_gam: Optional[np.ndarray] = None,
                  lambda_min_m: float, lambda_max_m: float, resolution: int = 40,
                  courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                  kerr: bool = False, source_amp: float = 1.0, npml: int = 12,
                  n_super: float = 1.0, n_sub: float = 1.0,
                  backend: str = "numpy", xp=np) -> FDTD2DResult:
    """Broadband R(f)/T(f) of a periodic (period_x_m) 2D-TE unit cell at NORMAL incidence. `layers`
    is the through-stack (z) profile; supply `lateral_eps_inf` (shape (nx, n_layer_cells) or a callable
    building the (nx,nz) eps_inf) to make a laterally-structured grating, else the stack is laterally
    UNIFORM (and the result reduces to the 1D solver / TMM). Returns both the 0-order (specular, x-mean)
    and the total-flux (all-diffraction-order) R/T.

    n_super / n_sub (default 1 = vacuum) are the lossless semi-infinite superstrate / substrate indices
    (metasurface-on-glass etc.): the z-pad regions are filled with n_super^2 / n_sub^2, the CPML is
    impedance-matched per end, and the incident reference is a homogeneous-superstrate run so R/T are
    correctly normalized (T carries the n_sub/n_super flux ratio). Reduces byte-identically to vacuum
    at n_super=n_sub=1.

    backend selects the compute kernel (see available_backends()): 'auto' (default-fastest CPU present),
    'numpy' (reference), 'numba' (fused threaded CPU -- fastest for unit cells), 'cupy' (NVIDIA GPU),
    'jax' (differentiable XLA), or the 'cpu'/'gpu' aliases. All backends are byte-for-byte equivalent on
    R/T (validation/fdtd_2d_reduces.py GATE D); xp is an advanced override for a custom array module."""
    if abs(complex(n_super).imag) > 1e-9 or abs(complex(n_sub).imag) > 1e-9:   # mirror the FEM guard
        raise NotImplementedError("solve_fdtd_2d: R/T and the energy budget are defined only for LOSSLESS "
                                  "end media (Im(n)=0); got n_super={}, n_sub={} (use the FEM/TMM solver "
                                  "for an absorbing incidence/exit medium).".format(n_super, n_sub))
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)      # sample the band (a Lorentz peak may be in-band)

    def _n_band_max(L):
        return max(abs(np.sqrt(L.eps_at(w))) for w in w_band)
    n_max = max(1.0, n_super, n_sub, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    if nx is None:
        nx = max(4, int(round(period_x_m / dz)))
    dx = period_x_m / nx
    # 2D CFL: dt <= courant / (c sqrt(1/dx^2 + 1/dz^2))
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1

    # z-profile, replicated over nx columns (laterally uniform unless lateral_eps_inf given)
    eps_inf = np.ones((nx, nz)); wp = np.zeros((nx, nz)); gam = np.zeros((nx, nz)); chi3 = np.zeros((nx, nz))
    lw0 = np.zeros((nx, nz)); lgam = np.zeros((nx, nz)); ldeps = np.zeros((nx, nz))  # Lorentz pole per cell
    chi2g = np.zeros((nx, nz))                                                       # R15 SHG chi2 [m/V]
    chi3R = np.zeros((nx, nz)); rw = np.zeros((nx, nz)); rgam = np.zeros((nx, nz))   # R15 Raman pole
    gw = np.zeros((nx, nz)); gdw = np.zeros((nx, nz)); gkdn = np.zeros((nx, nz))     # R20 gain line
    zc = (np.arange(nz) + 0.5) * dz
    # fill the semi-infinite super/substrate pads with the end-media permittivity (so the incident wave is
    # truly in n_super and the structure sees the n_sub backing); vacuum (n=1) leaves this as ones
    eps_inf[:, zc < pad] = n_super ** 2
    eps_inf[:, zc >= pad + z_struct] = n_sub ** 2
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, m] = L.eps_inf
        wp[:, m] = L.drude_wp_rad_s
        gam[:, m] = L.drude_gamma_rad_s
        lw0[:, m] = L.lorentz_w0_rad_s
        lgam[:, m] = L.lorentz_gamma_rad_s
        ldeps[:, m] = L.lorentz_delta_eps
        if kerr:
            chi3[:, m] = L.chi3_m2_V2
        chi2g[:, m] = L.chi2_m_V
        chi3R[:, m] = L.raman_chi3_m2_V2
        rw[:, m] = L.raman_w_rad_s
        rgam[:, m] = L.raman_gamma_rad_s
        gw[:, m] = L.gain_w_rad_s
        gdw[:, m] = L.gain_dw_rad_s
        gkdn[:, m] = L.gain_kappa_C2_kg * L.gain_dN_m3
        z += L.thickness_m
    if lateral_eps_inf is not None:
        # a laterally-structured grating: overwrite eps_inf in the structure band with the (nx, *)
        # lateral pattern (callable(nx,nz)->array, or an (nx,nz) array applied in the structure region)
        lat = lateral_eps_inf(nx, nz, zc, pad, z_struct) if callable(lateral_eps_inf) else np.asarray(lateral_eps_inf)
        eps_inf = np.asarray(lat, dtype=float)
        # GRID-SIZING GUARD: dz was derived from `layers` (+ end media) BEFORE this override. If the
        # lateral pattern's peak index exceeds the sizing index, dz is too coarse and R/T are silently
        # under-resolved. Raise rather than mis-solve -- size `layers` eps_inf to the pattern's max index
        # (the make_structured_lateral seam already does this) so n_max/dz are derived correctly.
        _n_lat = float(np.sqrt(max(1.0, float(np.max(np.real(eps_inf))))))
        if _n_lat > n_max * (1.0 + 1e-9):
            raise NotImplementedError(
                "solve_fdtd_2d: lateral pattern peak index {:.3f} exceeds the grid-sizing index {:.3f} "
                "from `layers` (+ end media) -- dz is under-resolved by {:.0%}. Size the `layers` "
                "eps_inf to the lateral pattern max so n_max/dz are derived from it.".format(
                    _n_lat, n_max, _n_lat / n_max - 1.0))
    # PER-CELL LOSSY/graded eps (R4): a Drude (wp,gam) grid alongside eps_inf lets a slow drive (gate E,
    # T, PCM fraction) paint a graded ABSORBING eps the eps_inf-only lateral seam cannot carry. Each is a
    # callable(nx,nz,zc,pad,z_struct)->array or an (nx,nz) array (zero in the pads). Default None -> the
    # wp/gam grids stay zeros -> byte-identical to the dielectric path.
    if lateral_wp is not None:
        wp = np.asarray(lateral_wp(nx, nz, zc, pad, z_struct) if callable(lateral_wp) else lateral_wp,
                        dtype=float)
    if lateral_gam is not None:
        gam = np.asarray(lateral_gam(nx, nz, zc, pad, z_struct) if callable(lateral_gam) else lateral_gam,
                         dtype=float)

    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))

    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    t_ring = _ring_time_s(layers)                            # audit C3-6: pole memory
    if t_ring > 200 * tau:
        import warnings
        warnings.warn("FDTD window extended {:.1f}x for a narrow Lorentz/gain line "
                      "(material memory {:.2e} s > the 200*tau source window; audit "
                      "C3-6)".format(1.0 + t_ring / (200 * tau), t_ring),
                      RuntimeWarning, stacklevel=2)
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau + t_ring) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    # Lorentz ADE coefficients (central difference): PL^{n+1} = C1 PL^n + C2 PL^{n-1} + C3 E^n, where the
    # pole eps += d_eps w0^2/(w0^2 - w^2 - i gl w). lor is applied to the STRUCTURE run only (the reference
    # is the bare superstrate). With d_eps=0 everywhere lor=None -> the path is byte-identical to before.
    lor = None
    if np.any(ldeps != 0.0):
        den = 1.0 + lgam * dt / 2.0
        C1 = (2.0 - lw0 ** 2 * dt ** 2) / den
        C2 = (lgam * dt / 2.0 - 1.0) / den
        C3 = (EPS0 * ldeps * lw0 ** 2 * dt ** 2) / den
        lor = (C1, C2, C3)

    # R15 chi2 / Raman flags + Raman vibrational-ADE coefficients (same central-difference template
    # as the Lorentz pole; Q'' + gam_R Q' + W_R^2 Q = W_R^2 E^2 -> Q^{n+1} = R1 Q + R2 Q^{n-1} + R3 E^2;
    # the polarization is P_R = eps0 chi3R E Q, formed in the kernel). All-zero grids -> None -> the
    # kernels take the pre-R15 code path byte-identically.
    chi2_arrs = chi2g if np.any(chi2g != 0.0) else None
    raman_arrs = None
    if np.any(chi3R != 0.0):
        if np.any((chi3R != 0.0) & (rw <= 0.0)):
            raise ValueError("Raman chi3 needs raman_w_rad_s > 0 on every Raman-active layer")
        if float(np.max(rw)) * dt > 1.0:                     # central-diff resonance stability margin
            raise ValueError("Raman resonance under-resolved: raman_w_rad_s*dt = {:.2f} > 1 -- raise "
                             "`resolution` (or lower the Raman frequency)".format(float(np.max(rw)) * dt))
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
    cpml_struct = _cpml_z(nz, dz, dt, npml, n_super, n_sub)  # PML matched to super (low z) + sub (high z)
    cpml_ref = _cpml_z(nz, dz, dt, npml, n_super, n_super)   # homogeneous-superstrate reference -> super both ends
    name = _resolve_backend(backend)                         # 'auto'/'cpu'/'gpu'/explicit -> concrete backend
    one = np.ones((nx, nz)); zero = np.zeros((nx, nz))

    def run(ei, w, g_, c3, cpml, lor=None, chi2=None, raman=None, gain=None):
        return _dispatch_2d_te(name, ei, w, g_, c3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp,
                               lor, chi2, raman, gain)

    # reference = homogeneous superstrate (no structure, no substrate) so the probe sees the pure incident
    # wave in n_super and the reflection subtraction is exact (same incident medium as the structure run)
    eyL_i, hxL_i, eyR_i, hxR_i = run(n_super ** 2 * one, zero, zero, zero, cpml_ref)
    eyL_t, hxL_t, eyR_t, hxR_t = run(eps_inf, wp, gam, chi3, cpml_struct, lor,
                                     chi2_arrs, raman_arrs, gain_arrs)  # structure run

    f = np.fft.rfftfreq(nsteps, dt)
    # ---- 0-order (specular) R/T from the x-MEAN field (== the 1D two-run method) ----
    mL_inc = np.fft.rfft(eyL_i.mean(axis=1)); mR_inc = np.fft.rfft(eyR_i.mean(axis=1))
    mRefl = np.fft.rfft((eyL_t - eyL_i).mean(axis=1)); mTrans = np.fft.rfft(eyR_t.mean(axis=1))
    k0 = 2.0 * np.pi * f / C_LIGHT
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(mRefl / mL_inc) ** 2
        # power transmittance carries the n_sub/n_super impedance (flux) ratio: the incident reference is
        # measured in n_super, the transmitted field in n_sub (Snell power continuity)
        T0 = np.abs(mTrans / mR_inc) ** 2 * (n_sub / n_super)
        # COMPLEX 0-order coeffs. np.fft.rfft yields exp(+i w t) phasors, but the library convention is
        # exp(-i w t), so conjugate to get the physical complex amplitudes; then de-embed the probe<->face
        # propagation phase. The superstrate phase velocity is c/n_super, so r0c (referenced to the front
        # face z=pad, probe at k_pL) carries n_super in k. t0c (audit C3-4): the incident reference
        # travels n_super the WHOLE way to the right probe while the transmitted leg is n_sub past the
        # back face, so the interface-referenced t carries n_sub*z_struct PLUS the (n_super-n_sub)
        # mismatch over the face->probe distance D = k_pR*dz - pad (the old bare exp(1j*k0*z_struct)
        # was vacuum-only: ~100 deg phase error on glass, frequency-dependent; |t| untouched).
        r0c = np.conj(mRefl / mL_inc) * np.exp(-2j * n_super * k0 * (pad - k_pL * dz))
        t0c = np.conj(mTrans / mR_inc) * np.exp(1j * k0 * (n_sub * z_struct
                                                           + (n_super - n_sub) * (k_pR * dz - pad)))
    # ---- TOTAL R/T from the Poynting flux (all diffraction orders) ----
    P_inc = _flux(eyL_i, hxL_i)
    P_refl = _flux(eyL_t - eyL_i, hxL_t - hxL_i)
    P_trans = _flux(eyR_t, hxR_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_flux = np.abs(P_refl) / np.abs(P_inc)
        T_flux = np.abs(P_trans) / np.abs(P_inc)
    band = (f >= f_min) & (f <= f_max) & (np.abs(mL_inc) > 0.05 * np.max(np.abs(mL_inc)))
    return FDTD2DResult(freqs_Hz=f, R0=R0, T0=T0, R_flux=R_flux, T_flux=T_flux, band=band, r0=r0c, t0=t0c)




def solve_fdtd_2d_oblique(layers: List[FDTDLayer], *, period_x_m: float, angle_deg: float,
                          lambda_min_m: float, lambda_max_m: float, resolution: int = 40,
                          courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                          source_amp: float = 1.0, npml: int = 12, nx: int = 8,
                          backend: str = "numpy", pol: str = "s") -> FDTD2DObliqueResult:
    """Broadband reflectance/transmittance of a laterally-uniform stack at OBLIQUE incidence, via the
    complex-envelope Bloch method with a FIXED transverse wavevector k_par = (2 pi / lambda_c)
    sin(angle_deg) (angle_deg the physical angle at the band centre). pol='s' = TE (Ey,Hx,Hz); pol='p' =
    TM (Hy,Ex,Ez) -- the p-pol R/T come from the tangential-Ex up/down ratio. Because k_par is fixed, the
    physical angle varies with frequency: theta(f) = asin(k_par c/(2 pi f)); the result carries theta_deg(f)
    and the band mask excludes frequencies below the light line (k_par > w/c, evanescent). Vacuum ends.
    angle_deg=0 reduces to the normal-incidence solver. backend selects the TE kernel (numpy/numba); TM is
    the NumPy reference."""
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' (TE) or 'p' (TM); got {!r}".format(pol))
    if any(L.lorentz_delta_eps != 0.0 for L in layers):     # the oblique kernel carries Drude only
        raise NotImplementedError("solve_fdtd_2d_oblique supports Drude dispersion only (no Lorentz pole "
                                  "yet); use solve_fdtd_2d at normal incidence for a Lorentz material.")
    # audit C5-7: the oblique kernel also carries NO chi3/chi2/Raman/gain ADEs -- these terms
    # used to be silently DROPPED (an amplifying/SHG/Raman stack at 20 deg returned R0/T0
    # bit-identical to the passive layer), while the 1-D entry point raises for the same set
    _dropped = [t for t in ("chi3_m2_V2", "chi2_m_V", "raman_chi3_m2_V2", "gain_dN_m3")
                if any(getattr(L, t, 0.0) != 0.0 for L in layers)]
    if _dropped:
        raise NotImplementedError(
            "solve_fdtd_2d_oblique: the oblique kernel carries no {} terms -- they would be "
            "silently ignored (audit C5-7); use the normal-incidence solver or split the "
            "problem.".format("/".join(_dropped)))
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)

    def _n_band_max(L):
        return max(abs(np.sqrt(L.eps_at(w))) for w in w_band)
    n_max = max(1.0, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    dx = period_x_m / nx
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    kx = (2.0 * np.pi * f_c / C_LIGHT) * np.sin(np.radians(angle_deg))   # fixed transverse wavevector

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1
    eps_inf = np.ones((nx, nz)); wp = np.zeros((nx, nz)); gam = np.zeros((nx, nz))
    zc = (np.arange(nz) + 0.5) * dz
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, m] = L.eps_inf; wp[:, m] = L.drude_wp_rad_s; gam[:, m] = L.drude_gamma_rad_s
        z += L.thickness_m

    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    t_ring = _ring_time_s(layers)                            # audit C3-6: pole memory
    if t_ring > 200 * tau:
        import warnings
        warnings.warn("FDTD window extended {:.1f}x for a narrow Lorentz/gain line "
                      "(material memory {:.2e} s > the 200*tau source window; audit "
                      "C3-6)".format(1.0 + t_ring / (200 * tau), t_ring),
                      RuntimeWarning, stacklevel=2)
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau + t_ring) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    cpml = _cpml_z(nz, dz, dt, npml)
    one = np.ones((nx, nz)); zero = np.zeros((nx, nz))
    # 'numba' = the fused threaded complex-envelope kernel; 'auto'/'cpu' pick it when present; everything else
    # falls back to the vectorized NumPy reference (the oblique path is normal-incidence-free of jax/cupy).
    rb = _resolve_backend(backend)
    # _run_oblique carries fused numba + differentiable jax kernels for BOTH s-pol (TE) and p-pol (TM);
    # pick the requested fast/diff backend when available, else the NumPy reference.
    if rb == "jax" and _have_jax():
        name = "jax"                                         # differentiable oblique scan (s + p)
    elif rb == "numba" and _HAVE_NUMBA:
        name = "numba"                                       # fused JIT oblique kernel (s + p)
    else:
        name = "numpy"
    eyL_i, hxL_i, eyR_i, hxR_i = _run_oblique(name, one, zero, zero, dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                              src, cpml, kx, pol)
    eyL_t, hxL_t, eyR_t, hxR_t = _run_oblique(name, eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                              src, cpml, kx, pol)
    # complex envelope -> full FFT; take the positive-frequency half (the forward exp(-iwt) response).
    # rfftfreq gives the monotonic positive-frequency axis matching fft(...)[:nf] within the band.
    nf = nsteps // 2 + 1
    f = np.fft.rfftfreq(nsteps, dt)
    mean = (lambda a: np.fft.fft(a.mean(axis=1))[:nf])
    inc_L = mean(eyL_i); inc_R = mean(eyR_i)
    refl = mean(eyL_t - eyL_i); trans = mean(eyR_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(refl / inc_L) ** 2
        T0 = np.abs(trans / inc_R) ** 2
    sin_t = np.divide(kx * C_LIGHT, 2.0 * np.pi * np.maximum(f, 1e-30))   # sin theta(f) = k_par c / w
    theta = np.degrees(np.arcsin(np.clip(sin_t, -1.0, 1.0)))
    # audit C3-5: the old sin_t < 0.999 mask trusted points up to theta ~ 87 deg, where
    # the z-CPML's grazing round-trip echo reaches 0.1-0.5 FIELD (the absorber sees the
    # z-wavevector shrink as cos theta): the validation geometry re-run at 76 deg carried
    # band=True points with |R0 - TMM| = 0.39 and R0+T0-1 up to +0.38. Trust only
    # sin_t < 0.95 (theta < ~72 deg -- the measured error onset for the shipped npml=12);
    # warn when the mask removes otherwise-excited in-band points so the truncation is
    # visible rather than silent.
    _excited = (f >= f_min) & (f <= f_max) & (np.abs(inc_L) > 0.05 * np.max(np.abs(inc_L)))
    band = _excited & (sin_t < 0.95)
    _cut = _excited & (sin_t >= 0.95)
    if np.any(_cut):
        import warnings
        warnings.warn(
            "solve_fdtd_2d_oblique: {} excited in-band points at theta(f) >= 71.8 deg were "
            "EXCLUDED from the trusted band -- the grazing-incidence CPML echo corrupts R0/T0 "
            "there (audit C3-5); narrow the band, lower angle_deg, or strengthen npml."
            .format(int(np.sum(_cut))), RuntimeWarning, stacklevel=2)
    return FDTD2DObliqueResult(freqs_Hz=f, theta_deg=theta, R0=R0, T0=T0, band=band)


# =====================================================================================================
# 3D: full-vector Yee engine for a 2D-periodic (x AND y) unit cell at normal incidence.
# The 2D-TE engine above is the (d/dy = 0, {Ey,Hx,Hz}) reduction of this; this carries all six field
# components so a genuinely 2D-periodic structure (pillars/holes/crosses) couples into every order.
# =====================================================================================================
