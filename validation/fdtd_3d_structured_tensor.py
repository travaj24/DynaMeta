"""Validate the STRUCTURED (laterally-patterned) 3-D DIAGONAL-TENSOR FDTD: solve_fdtd_3d_mo with the new
`lateral_tensor` override patterns a per-cell anisotropic tensor (exx,eyy,ezz)(x,y,z) over a 2-D-periodic
unit cell (here a square dielectric pillar array), wc=0 so the engine is a plain diagonal-anisotropic Yee
solve. Independent oracle = grcwa (RCWA, frequency-domain). The check is built to DEFEAT THE LOSSLESS TRAP
(energy closure alone does not prove a correct per-order split): a per-polarization grcwa comparison pins
each tensor component independently.

KEY DECOUPLING: at NORMAL incidence on an x- and y-mirror-symmetric pattern, an x-polarized wave excites
only (Ex, Ez) -- it depends on eps_xx and eps_zz but NOT eps_yy (Ey stays zero by symmetry). So a tensor
with eps_zz == eps_xx makes the x-pol problem IDENTICAL to a fully isotropic eps = eps_xx medium, whose
RCWA answer is the oracle; eps_yy is set DIFFERENT and must not affect it. The mirror (y-pol vs eps_yy)
pins the other in-plane component.

GATE 0 (oracle sanity): grcwa on a uniform slab == analytic Airy.
GATE A (isotropic structured): exx=eyy=ezz patterned -> FDTD R/T == grcwa, R+T = 1.
GATE B (x-pol -> eps_xx): anisotropic (exx=hi, eyy=lo, ezz=exx), x-pol -> matches grcwa(eps=hi) and is
        CLEARLY closer to grcwa(hi) than grcwa(lo) (the per-order discriminator; eps_yy correctly ignored).
GATE C (y-pol -> eps_yy): anisotropic (exx=lo, eyy=hi, ezz=eyy), y-pol -> matches grcwa(eps=hi); and the
        x-pol vs y-pol responses differ (genuine anisotropy), each pinned to its own component.

Run: python -m validation.fdtd_3d_structured_tensor
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grcwa

from dynameta.optics.fdtd_mo import MOLayer
from dynameta.optics.fdtd_nd import solve_fdtd_3d_mo

C = 299792458.0
LAM = 1300e-9
PERIOD = 600e-9          # sub-wavelength: 0-order only -> small fast 3D grid, still exercises the tensor raster
PILLAR = 300e-9
THICK = 250e-9
EPS_HI = 6.25            # n = 2.5 pillar
EPS_LO_IN = 2.25         # n = 1.5 (the "other" in-plane component inside the pillar -- the anisotropy)
EPS_BG = 1.0             # air background
RES, NX, NY = 11, 14, 14


def airy(f, n, d):
    k0 = 2 * np.pi * f / C
    b = n * k0 * d
    r1 = (1.0 - n) / (1.0 + n)
    e2 = np.exp(2j * b)
    r = r1 * (1.0 - e2) / (1.0 - r1 ** 2 * e2)
    t = (1.0 - r1 ** 2) * np.exp(1j * b) / (1.0 - r1 ** 2 * e2)
    return float(np.abs(r) ** 2), float(np.abs(t) ** 2)


def _pillar_mask(nx, ny):
    xs = (np.arange(nx) + 0.5) * PERIOD / nx
    ys = (np.arange(ny) + 0.5) * PERIOD / ny
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return (np.abs(X - PERIOD / 2) <= PILLAR / 2) & (np.abs(Y - PERIOD / 2) <= PILLAR / 2)


def _tensor(exx_in, eyy_in, ezz_in):
    """A lateral_tensor callable: pillar = (exx_in,eyy_in,ezz_in) inside, air outside, vacuum in the pads."""
    def build(nx, ny, nz, zc, pad, z_struct):
        m = _pillar_mask(nx, ny)[:, :, None]
        band = ((zc >= pad) & (zc < pad + z_struct))[None, None, :]

        def fld(ein):
            patt = np.where(m, ein, EPS_BG)
            return np.where(band, patt, 1.0)
        return {"exx": fld(exx_in), "eyy": fld(eyy_in), "ezz": fld(ezz_in)}
    return build


def _grcwa_grid_rt(eps_grid, nG=121):
    """grcwa total R/T for ONE patterned layer (eps_grid, ng x ng) in vacuum, normal incidence."""
    ng = eps_grid.shape[0]
    lam_nm, p_nm = LAM * 1e9, PERIOD * 1e9
    o = grcwa.obj(nG, [p_nm, 0.0], [0.0, p_nm], 1.0 / lam_nm, 0.0, 0.0, verbose=0)
    o.Add_LayerUniform(lam_nm, 1.0)
    o.Add_LayerGrid(THICK * 1e9, ng, ng)
    o.Add_LayerUniform(lam_nm, 1.0)
    o.Init_Setup()
    o.GridLayer_geteps(eps_grid.flatten().astype(complex))
    o.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    R, T = o.RT_Solve(normalize=1)
    return float(np.real(R)), float(np.real(T))


def _grcwa_rt(eps_in, ng=96, nG=121):
    xs = (np.arange(ng) + 0.5) * PERIOD / ng
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    ep = np.where((np.abs(X - PERIOD / 2) <= PILLAR / 2) & (np.abs(Y - PERIOD / 2) <= PILLAR / 2),
                  eps_in, EPS_BG)
    return _grcwa_grid_rt(ep, nG=nG)


def _fdtd_rt(exx, eyy, ezz, pol):
    L = [MOLayer(thickness_m=THICK, eps_xx=EPS_HI, eps_yy=EPS_HI, drude_wp_rad_s=0.0,
                 drude_gamma_rad_s=0.0, cyclotron_wc_rad_s=0.0)]   # eps=EPS_HI sets n_max for grid sizing
    r = solve_fdtd_3d_mo(L, period_x_m=PERIOD, period_y_m=PERIOD, lambda_min_m=1200e-9,
                         lambda_max_m=1400e-9, resolution=RES, n_pad_wave=2.5, settle=10.0, pol=pol,
                         nx=NX, ny=NY, npml=10, lateral_tensor=_tensor(exx, eyy, ezz))
    b = r.band
    i = int(np.argmin(np.abs(r.freqs_Hz[b] - C / LAM)))
    return float(r.R[b][i]), float(r.T[b][i])


def main():
    print("[st] === FDTD 3-D structured diagonal-TENSOR vs grcwa (RCWA) ===", flush=True)

    # GATE 0: grcwa on a genuinely UNIFORM eps grid == analytic Airy (the RCWA oracle is set up right)
    Ru, Tu = _grcwa_grid_rt(np.full((8, 8), EPS_HI), nG=21)
    Ra, Ta = airy(C / LAM, np.sqrt(EPS_HI), THICK)
    g0 = max(abs(Ru - Ra), abs(Tu - Ta)) < 2e-3
    print("[st] 0 grcwa uniform == Airy: R {:.4f}/{:.4f} T {:.4f}/{:.4f} -> {}".format(
        Ru, Ra, Tu, Ta, "OK" if g0 else "FAIL"), flush=True)

    Rg_hi, Tg_hi = _grcwa_rt(EPS_HI)
    Rg_lo, Tg_lo = _grcwa_rt(EPS_LO_IN)

    # GATE A: isotropic structured (exx=eyy=ezz=hi) == grcwa(hi)
    Ra_f, Ta_f = _fdtd_rt(EPS_HI, EPS_HI, EPS_HI, "y")
    gA = abs(Ra_f - Rg_hi) < 4e-2 and abs(Ta_f - Tg_hi) < 4e-2 and abs(Ra_f + Ta_f - 1.0) < 4e-2
    print("[st] A isotropic: FDTD R={:.4f} T={:.4f} | grcwa R={:.4f} T={:.4f} | dR={:.3f} dT={:.3f} "
          "R+T={:.3f} -> {}".format(Ra_f, Ta_f, Rg_hi, Tg_hi, abs(Ra_f - Rg_hi), abs(Ta_f - Tg_hi),
                                    Ra_f + Ta_f, "OK" if gA else "FAIL"), flush=True)

    # GATE B: x-pol on (exx=hi, eyy=lo, ezz=exx) -> grcwa(hi), and clearly closer to hi than lo
    Rbx, Tbx = _fdtd_rt(EPS_HI, EPS_LO_IN, EPS_HI, "x")
    d_hi, d_lo = abs(Rbx - Rg_hi), abs(Rbx - Rg_lo)
    gB = abs(Rbx - Rg_hi) < 4e-2 and abs(Tbx - Tg_hi) < 4e-2 and d_hi < 0.5 * d_lo
    print("[st] B x-pol->exx: FDTD R={:.4f} T={:.4f} | grcwa(hi) R={:.4f} (d={:.3f}) vs grcwa(lo) R={:.4f}"
          " (d={:.3f}) discriminator {:.2f}x -> {}".format(Rbx, Tbx, Rg_hi, d_hi, Rg_lo, d_lo,
                                                           d_lo / max(d_hi, 1e-6), "OK" if gB else "FAIL"),
          flush=True)

    # GATE C: y-pol on (exx=lo, eyy=hi, ezz=eyy) -> grcwa(hi); x vs y differ (genuine anisotropy)
    Rcy, Tcy = _fdtd_rt(EPS_LO_IN, EPS_HI, EPS_HI, "y")
    gC = abs(Rcy - Rg_hi) < 4e-2 and abs(Tcy - Tg_hi) < 4e-2 and abs(Rcy - Rbx) < 4e-2
    print("[st] C y-pol->eyy: FDTD R={:.4f} T={:.4f} | grcwa(hi) R={:.4f} | y-vs-x |dR|={:.3f} (both pin "
          "their hi component) -> {}".format(Rcy, Tcy, Rg_hi, abs(Rcy - Rbx), "OK" if gC else "FAIL"),
          flush=True)

    ok = g0 and gA and gB and gC
    print("[st] *** FDTD 3-D STRUCTURED TENSOR vs RCWA (per-component decoupling; energy): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
