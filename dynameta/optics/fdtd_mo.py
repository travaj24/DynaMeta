"""
1-D magneto-optic / anisotropic FDTD: a normal-incidence, z-propagation, FULL-transverse-polarization
(Ex, Ey) time-domain solver carrying a per-cell DIAGONAL anisotropy (eps_xx, eps_yy) AND a gyrotropic
MAGNETO-OPTIC response via a magnetized-Drude auxiliary-differential-equation (the cyclotron coupling
wc * (zhat x J) that mixes Jx <-> Jy). This is the PHYSICALLY-CORRECT time-domain route to gyrotropic
optics: the frequency-domain off-diagonal i*g is a stand-in for exactly this TIME-DERIVATIVE coupling, so
a real-time FDTD must carry the cyclotron ADE (a complex algebraic E = inv(eps) @ D on real fields would
be unphysical). Faraday rotation falls out for free -- the two circular eigenmodes see eps_pm = eps_inf -
wp^2/(w(w -/+ wc) + i w gamma) and accumulate a differential phase.

Method: staggered Ex/Ey (E-grid) and Hx/Hy (H-grid) leapfrog; 1st-order Mur ABC at both vacuum ends; a
soft modulated-Gaussian plane-wave source (a chosen input polarization). The magnetized-Drude ADE is a
per-cell 2x2 Crank-Nicolson solve coupled semi-implicitly to the E-update (so the E^{n+1} <-> J^{n+1}
coupling is resolved with one precomputed 2x2 inverse per cell). The TWO-RUN reference method gives the
broadband complex co- and cross-polarized reflection / transmission, from which R/T and the Faraday
rotation angle are read. Convention exp(-i w t), SI, Im(eps) > 0 = loss.

Validated (validation/fdtd_mo_vs_tmm.py) vs (a) per-polarization scalar TMM for a birefringent
(eps_xx != eps_yy) slab and (b) the circular-eigenmode Jones-TMM Faraday rotation for a gyrotropic slab.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, MU0  # MU0 single-sourced in constants (was re-derived here)
from dynameta.optics.fdtd_nd import HAVE_NUMBA, njit, resolve_backend


@dataclass
class MOLayer:
    """One layer of the 1-D magneto-optic / anisotropic stack. Diagonal background eps_xx / eps_yy (the
    transverse principal permittivities -> birefringence), an optional magnetized-Drude free-carrier pole
    (plasma frequency wp, damping gamma) and its cyclotron frequency wc = q B0 / m* (the gyration; wc=0 ->
    a plain anisotropic Drude). eps_pm(w) = eps_inf - wp^2/(w(w -/+ wc) + i w gamma) for the +/- circular
    modes."""
    thickness_m: float
    eps_xx: float = 1.0
    eps_yy: float = 1.0
    drude_wp_rad_s: float = 0.0
    drude_gamma_rad_s: float = 0.0
    cyclotron_wc_rad_s: float = 0.0

    def eps_circular(self, w_rad_s, sign):
        """The +/- circular-eigenmode permittivity (sign = +1 for the '+' mode, -1 for '-'); the diagonal
        background is taken isotropic at 0.5(eps_xx+eps_yy) for this scalar circular model (used only as
        the gyrotropic oracle, where eps_xx == eps_yy)."""
        eps_inf = 0.5 * (self.eps_xx + self.eps_yy)
        if self.drude_wp_rad_s <= 0.0:
            return complex(eps_inf)
        wc = self.cyclotron_wc_rad_s
        return eps_inf - self.drude_wp_rad_s ** 2 / (w_rad_s * (w_rad_s - sign * wc) + 1j * w_rad_s * self.drude_gamma_rad_s)


@dataclass
class FDTDMOResult:
    freqs_Hz: np.ndarray
    band: np.ndarray
    # complex co/cross 0-order coefficients at the probes (input along `pol`); co = same axis as input
    t_co: np.ndarray
    t_cross: np.ndarray
    r_co: np.ndarray
    r_cross: np.ndarray
    R: np.ndarray               # total reflectance |r_co|^2 + |r_cross|^2
    T: np.ndarray               # total transmittance |t_co|^2 + |t_cross|^2
    faraday_deg: np.ndarray     # polarization-ellipse major-axis rotation of the transmitted wave [deg]


def _mo_band_index_bound(L, w_band) -> float:
    """Grid-sizing index bound for one MOLayer over the band (audit C3-3): the max |n|
    over BOTH circular branches eps_circular(w, +/-1) -- the (w, +1) branch is resonant
    only for wc > 0, so the old one-branch max silently under-sized dz ~6x near resonance
    for reversed magnetization / electron-signed wc < 0 (the MOLayer docstring's own
    convention) -- floored by the background birefringent indices, which the
    Drude-depressed circular index can undercut. wc-SIGN-INVARIANT by construction."""
    bg = max(np.sqrt(L.eps_xx), np.sqrt(L.eps_yy), 1.0)
    if L.drude_wp_rad_s <= 0:
        return float(bg)
    n_circ = max(abs(np.sqrt(L.eps_circular(w, s))) for w in w_band for s in (+1, -1))
    return float(max(n_circ, bg))


def _2x2_inv(M):
    """Vectorized inverse of a stack of 2x2 matrices, shape (...,2,2)."""
    a, b, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    det = a * d - b * c
    out = np.empty_like(M)
    out[..., 0, 0] = d / det; out[..., 0, 1] = -b / det
    out[..., 1, 0] = -c / det; out[..., 1, 1] = a / det
    return out


@njit(fastmath=True, cache=True)
def _mo_loop_numba(iv00, iv01, iv10, iv11, ep00, ep01, ep10, ep11, jc00, jc01, jc10, jc11,
                   ma00, ma01, ma10, ma11, mb00, mb01, mb10, mb11,
                   dz, dt, nsteps, i_src, i_pL, i_pR, src, src_on_y):
    """JIT-compiled bi-polarization 1-D MO time loop -- the SAME physics as the _run_mo NumPy loop
    (staggered Ex/Ey, Hx/Hy leapfrog; per-cell magnetized-Drude 2x2 Crank-Nicolson via the precomputed
    scalar components iv/ep/jc/ma/mb; soft source on `src_on_y`; 1st-order Mur ABC both ends), fused into
    one compiled pass with no per-step temporary arrays. Returns the (Ex,Ey) probe time series eL/eR."""
    nz = iv00.size
    Ex = np.zeros(nz); Ey = np.zeros(nz)
    Hx = np.zeros(nz - 1); Hy = np.zeros(nz - 1)
    J0 = np.zeros(nz); J1 = np.zeros(nz)
    Exn = np.empty(nz); Eyn = np.empty(nz)
    eL = np.empty((nsteps, 2)); eR = np.empty((nsteps, 2))
    cmu = dt / (MU0 * dz)
    mur = (C_LIGHT * dt - dz) / (C_LIGHT * dt + dz)
    for n in range(nsteps):
        ex_oL0 = Ex[0]; ex_oL1 = Ex[1]; ey_oL0 = Ey[0]; ey_oL1 = Ey[1]
        ex_oR0 = Ex[nz - 1]; ex_oR1 = Ex[nz - 2]; ey_oR0 = Ey[nz - 1]; ey_oR1 = Ey[nz - 2]
        for k in range(nz - 1):                                # H update
            Hx[k] += cmu * (Ey[k + 1] - Ey[k])
            Hy[k] -= cmu * (Ex[k + 1] - Ex[k])
        for k in range(nz):                                    # E + J update (curl=0 at the ends -> Mur)
            if k == 0 or k == nz - 1:
                c0 = 0.0; c1 = 0.0
            else:
                c0 = -(Hy[k] - Hy[k - 1]) / dz
                c1 = (Hx[k] - Hx[k - 1]) / dz
            r0 = ep00[k] * Ex[k] + ep01[k] * Ey[k] + c0 - (jc00[k] * J0[k] + jc01[k] * J1[k])
            r1 = ep10[k] * Ex[k] + ep11[k] * Ey[k] + c1 - (jc10[k] * J0[k] + jc11[k] * J1[k])
            exn = iv00[k] * r0 + iv01[k] * r1
            eyn = iv10[k] * r0 + iv11[k] * r1
            Exn[k] = exn; Eyn[k] = eyn
            sx = exn + Ex[k]; sy = eyn + Ey[k]                 # J^{n+1} uses OLD J0/J1 of THIS cell only
            j0n = ma00[k] * J0[k] + ma01[k] * J1[k] + mb00[k] * sx + mb01[k] * sy
            j1n = ma10[k] * J0[k] + ma11[k] * J1[k] + mb10[k] * sx + mb11[k] * sy
            J0[k] = j0n; J1[k] = j1n
        if src_on_y:
            Eyn[i_src] += src[n]
        else:
            Exn[i_src] += src[n]
        Exn[0] = ex_oL1 + mur * (Exn[1] - ex_oL0); Eyn[0] = ey_oL1 + mur * (Eyn[1] - ey_oL0)
        Exn[nz - 1] = ex_oR1 + mur * (Exn[nz - 2] - ex_oR0)
        Eyn[nz - 1] = ey_oR1 + mur * (Eyn[nz - 2] - ey_oR0)
        for k in range(nz):
            Ex[k] = Exn[k]; Ey[k] = Eyn[k]
        eL[n, 0] = Ex[i_pL]; eL[n, 1] = Ey[i_pL]
        eR[n, 0] = Ex[i_pR]; eR[n, 1] = Ey[i_pR]
    return eL, eR


def _run_mo(exx, eyy, wp, gam, wc, dz, dt, nsteps, i_src, i_pL, i_pR, src, pol, backend="numpy"):
    """One 1-D bi-polarization pass. Per-cell diagonal eps (exx,eyy) + magnetized-Drude ADE (wp,gam,wc).
    Returns the Ex,Ey time series at the left (reflection) and right (transmission) probes. backend='numba'
    runs the JIT-compiled time loop (the precompute below is identical); any other value runs the NumPy
    reference loop."""
    nz = exx.size
    Ex = np.zeros(nz); Ey = np.zeros(nz)
    Hx = np.zeros(nz - 1); Hy = np.zeros(nz - 1)
    J0 = np.zeros(nz); J1 = np.zeros(nz)                    # magnetized-Drude current (Jx,Jy), split
    # per-cell magnetized-Drude Crank-Nicolson + E-update matrices (all (nz,2,2)):
    #   A J^{n+1} = B J^n + eps0 wp^2 (E^{n+1}+E^n)/2 ; (gamma I + Wg), Wg = [[0,-wc],[wc,0]]
    I2 = np.broadcast_to(np.eye(2), (nz, 2, 2)).copy()
    Wg = np.zeros((nz, 2, 2)); Wg[:, 0, 1] = -wc; Wg[:, 1, 0] = wc
    G = gam[:, None, None] * I2 + Wg
    A = I2 / dt + 0.5 * G
    B = I2 / dt - 0.5 * G
    Ainv = _2x2_inv(A)
    Ma = np.einsum("zij,zjk->zik", Ainv, B)                 # J^{n+1} = Ma J^n + Mb (E^{n+1}+E^n)
    Mb = Ainv * (EPS0 * wp ** 2 * 0.5)[:, None, None]
    D = np.zeros((nz, 2, 2)); D[:, 0, 0] = EPS0 * exx / dt; D[:, 1, 1] = EPS0 * eyy / dt
    Einv = _2x2_inv(D + 0.5 * Mb)                           # E^{n+1} = Einv [ (D-Mb/2)E^n + curl - (Ma+I)/2 J^n ]
    Epre = D - 0.5 * Mb
    Jc = 0.5 * (Ma + I2)
    # split every per-cell 2x2 into its four (nz,) components -- the inner loop then uses explicit scalar
    # arithmetic (NO per-step einsum/stack; ~2.6x faster than einsum for these tiny 2x2 matvecs).
    (iv00, iv01, iv10, iv11) = (Einv[:, 0, 0], Einv[:, 0, 1], Einv[:, 1, 0], Einv[:, 1, 1])
    (ep00, ep01, ep10, ep11) = (Epre[:, 0, 0], Epre[:, 0, 1], Epre[:, 1, 0], Epre[:, 1, 1])
    (jc00, jc01, jc10, jc11) = (Jc[:, 0, 0], Jc[:, 0, 1], Jc[:, 1, 0], Jc[:, 1, 1])
    (ma00, ma01, ma10, ma11) = (Ma[:, 0, 0], Ma[:, 0, 1], Ma[:, 1, 0], Ma[:, 1, 1])
    (mb00, mb01, mb10, mb11) = (Mb[:, 0, 0], Mb[:, 0, 1], Mb[:, 1, 0], Mb[:, 1, 1])
    if backend == "numba" and HAVE_NUMBA:                       # JIT the hot time loop (same precompute)
        ac = (lambda a: np.ascontiguousarray(a, dtype=np.float64))
        return _mo_loop_numba(ac(iv00), ac(iv01), ac(iv10), ac(iv11), ac(ep00), ac(ep01), ac(ep10), ac(ep11),
                              ac(jc00), ac(jc01), ac(jc10), ac(jc11), ac(ma00), ac(ma01), ac(ma10), ac(ma11),
                              ac(mb00), ac(mb01), ac(mb10), ac(mb11), dz, dt, nsteps, i_src, i_pL, i_pR,
                              ac(src), 1 if pol == "y" else 0)
    c = C_LIGHT
    mur = (c * dt - dz) / (c * dt + dz)
    eL = np.empty((nsteps, 2)); eR = np.empty((nsteps, 2))
    cmu = dt / (MU0 * dz)
    cz0 = np.zeros(nz)
    for n in range(nsteps):
        ex_oL0, ex_oL1 = Ex[0], Ex[1]; ey_oL0, ey_oL1 = Ey[0], Ey[1]
        ex_oR0, ex_oR1 = Ex[-1], Ex[-2]; ey_oR0, ey_oR1 = Ey[-1], Ey[-2]
        # H update: dHx/dt = (1/mu) dEy/dz ; dHy/dt = -(1/mu) dEx/dz
        Hx += cmu * (Ey[1:] - Ey[:-1])
        Hy -= cmu * (Ex[1:] - Ex[:-1])
        # curl H at E points (interior): curl_x = -dHy/dz, curl_y = dHx/dz
        c0 = cz0.copy(); c1 = cz0.copy()
        c0[1:-1] = -(Hy[1:] - Hy[:-1]) / dz
        c1[1:-1] = (Hx[1:] - Hx[:-1]) / dz
        # rhs = Epre @ [Ex;Ey] + curl - Jc @ [J0;J1] ; Enew = Einv @ rhs   (explicit 2x2)
        r0 = ep00 * Ex + ep01 * Ey + c0 - (jc00 * J0 + jc01 * J1)
        r1 = ep10 * Ex + ep11 * Ey + c1 - (jc10 * J0 + jc11 * J1)
        Exn = iv00 * r0 + iv01 * r1
        Eyn = iv10 * r0 + iv11 * r1
        # J^{n+1} = Ma @ J + Mb @ (Enew + Eold) ; J1 needs the OLD J0, so write to fresh names
        sx = Exn + Ex; sy = Eyn + Ey
        J0n = ma00 * J0 + ma01 * J1 + mb00 * sx + mb01 * sy
        J1n = ma10 * J0 + ma11 * J1 + mb10 * sx + mb11 * sy
        J0, J1 = J0n, J1n
        if pol == "y":                                      # soft source on the input axis
            Eyn[i_src] += src[n]
        else:
            Exn[i_src] += src[n]
        # 1st-order Mur ABC at both vacuum ends (each component independently)
        Exn[0] = ex_oL1 + mur * (Exn[1] - ex_oL0); Eyn[0] = ey_oL1 + mur * (Eyn[1] - ey_oL0)
        Exn[-1] = ex_oR1 + mur * (Exn[-2] - ex_oR0); Eyn[-1] = ey_oR1 + mur * (Eyn[-2] - ey_oR0)
        Ex, Ey = Exn, Eyn
        eL[n] = (Ex[i_pL], Ey[i_pL]); eR[n] = (Ex[i_pR], Ey[i_pR])
    return eL, eR


def solve_fdtd_mo_1d(layers: List[MOLayer], *, lambda_min_m: float, lambda_max_m: float,
                     resolution: int = 60, courant: float = 0.5, n_pad_wave: float = 6.0,
                     settle: float = 14.0, pol: str = "y", source_amp: float = 1.0,
                     backend: str = "numpy") -> FDTDMOResult:
    """Broadband co/cross R/T + Faraday rotation of a 1-D anisotropic / magneto-optic stack (vacuum ends),
    input linearly polarized along `pol` ('x' or 'y'). The two-run reference method gives the COMPLEX
    co-pol (same axis as input) and cross-pol (orthogonal) reflection/transmission; the Faraday angle is
    the major-axis rotation of the transmitted polarization ellipse. backend='numba' JITs the time loop
    (~same answer to ~1e-12); 'auto'/'cpu' select it when present, else the NumPy reference runs."""
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)

    n_max = max(1.0, max(_mo_band_index_bound(L, w_band) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    dt = courant * dz / C_LIGHT
    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1

    exx = np.ones(nz); eyy = np.ones(nz); wp = np.zeros(nz); gam = np.zeros(nz); wc = np.zeros(nz)
    zc = np.arange(nz) * dz
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        exx[m] = L.eps_xx; eyy[m] = L.eps_yy
        wp[m] = L.drude_wp_rad_s; gam[m] = L.drude_gamma_rad_s; wc[m] = L.cyclotron_wc_rad_s
        z += L.thickness_m

    i_src = max(2, int(round(0.35 * pad / dz)))
    i_pL = int(round(0.7 * pad / dz))
    i_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    nsteps = int(round((2.0 * t0 + 4.0 * Lz / C_LIGHT + 200 * tau) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    one = np.ones(nz); zero = np.zeros(nz)
    rb = resolve_backend(backend)                               # 'numba' = JIT loop; else NumPy reference
    bk = "numba" if (rb == "numba" and HAVE_NUMBA) else "numpy"
    eL_i, eR_i = _run_mo(one, one, zero, zero, zero, dz, dt, nsteps, i_src, i_pL, i_pR, src, pol, bk)   # vacuum
    eL_t, eR_t = _run_mo(exx, eyy, wp, gam, wc, dz, dt, nsteps, i_src, i_pL, i_pR, src, pol, bk)         # structure

    f = np.fft.rfftfreq(nsteps, dt)
    co, cr = (1, 0) if pol == "y" else (0, 1)               # co = input axis index, cr = orthogonal
    # rfft is exp(+iwt); conjugate to the exp(-iwt) convention. Incident reference = the co-pol input field.
    inc_L = np.conj(np.fft.rfft(eL_i[:, co])); inc_R = np.conj(np.fft.rfft(eR_i[:, co]))
    refl_co = np.conj(np.fft.rfft(eL_t[:, co] - eL_i[:, co])); refl_cr = np.conj(np.fft.rfft(eL_t[:, cr]))
    tr_co = np.conj(np.fft.rfft(eR_t[:, co])); tr_cr = np.conj(np.fft.rfft(eR_t[:, cr]))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_co = refl_co / inc_L; r_cr = refl_cr / inc_L
        t_co = tr_co / inc_R; t_cr = tr_cr / inc_R
        R = np.abs(r_co) ** 2 + np.abs(r_cr) ** 2
        T = np.abs(t_co) ** 2 + np.abs(t_cr) ** 2
        # Faraday: major-axis rotation of the transmitted (E_co, E_cross) ellipse, signed toward +cross
        far = 0.5 * np.arctan2(2.0 * np.real(t_co * np.conj(t_cr)), np.abs(t_co) ** 2 - np.abs(t_cr) ** 2)
    band = (f >= f_min) & (f <= f_max) & (np.abs(inc_L) > 0.05 * np.max(np.abs(inc_L)))
    return FDTDMOResult(freqs_Hz=f, band=band, t_co=t_co, t_cross=t_cr, r_co=r_co, r_cross=r_cr,
                        R=R, T=T, faraday_deg=np.degrees(far))
