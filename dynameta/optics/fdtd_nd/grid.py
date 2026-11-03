"""Shared z-grid sizing, probe placement and source-pulse builders for every FDTD front end.

audit D-15 (2026-11-02): the source-envelope / time-window, probe-placement and z-grid-sizing
blocks were re-typed verbatim in SEVEN front ends -- solve_fdtd_1d (fdtd.py), solve_fdtd_mo_1d
(fdtd_mo.py), solve_fdtd_2d / solve_fdtd_2d_oblique (solve2d.py) and solve_fdtd_3d /
solve_fdtd_3d_oblique / solve_fdtd_3d_mo (solve3d.py) -- so audit D-3 (probe guard), D-10 (window
memory) and D-11 (CFL guard) each needed the same edit in 4-8 places, and two of the copies had
already drifted in spelling (`4.0 * Lz / C_LIGHT` vs `(Lz / C_LIGHT) * 4.0`, `0.35 * pad / dz` vs
`(0.35 * pad) / dz`). The window-memory and CFL halves were single-sourced in wave 5
(`solve2d._window_memory_s`, `spec.courant_guard`); these three helpers close the rest. Each one
is the old inline arithmetic operand-for-operand, so every front end is BIT-IDENTICAL to the
pre-refactor code (checked on all seven, 204 result arrays; the two drifted spellings are the
same float because x4 is exact in binary floating point). The per-front-end sizing INDEX stays a
parameter (`n_max`): the 1-D band-edge Drude index, the 2-D/3-D `eps_at` band maximum, the
oblique copies that omit the end media (audit X-18) and the MO circular-branch bound genuinely
differ and are not unified here.
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import C_LIGHT


def z_grid(layers, lambda_min_m, lambda_max_m, resolution, n_pad_wave, n_max):
    """(dz, pad, z_struct, Lz, nz) of the uniform z grid: `resolution` cells per lambda_min in the
    index-`n_max` medium, `n_pad_wave` * lambda_max of pad each side of the stack."""
    dz = lambda_min_m / (resolution * n_max)
    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1
    return dz, pad, z_struct, Lz, nz


def material_z(nz, dz):
    """Positions [m] at which EVERY uniform-grid FDTD engine registers its materials: the Yee E-NODES
    k*dz, k = 0..nz-1 (audit S2-14 engine siblings, 2026-11-02).

    WHY THE NODES. Every engine shares one z staggering: the tangential E (Ex/Ey -- the field the
    planar wave, the soft source, the probes and the PEC/Mur walls all live on) at k*dz and the
    tangential H at (k+1/2)*dz; `probe_cells` rounds z/dz to an E index and the front ends
    de-embed r0 in that node frame (exp(-2i n k0 (pad - k_pL*dz))). The eps that multiplies E_k
    represents E_k's control volume [(k-1/2), (k+1/2)]*dz, which is CENTRED ON THE NODE (Taflove &
    Hagness, Computational Electrodynamics 3rd ed., the 1-D/3-D Yee material assignment; Hwang &
    Cangellaris, IEEE MWCL 11:158 (2001): for the tangential E the second-order effective eps is the
    control-volume arithmetic mean). Point-sampling eps AT the node therefore puts the effective
    interface (the H node between the last outside and the first inside E node) within +-dz/2 of the
    true one with ZERO MEAN over the interface's sub-cell offset. The old (k + 1/2)*dz "cell-centre"
    sampling of solve_fdtd_1d's uniform path, solve_fdtd_2d(_oblique) and solve_fdtd_3d(_oblique/
    _mo) shifted every structure by -dz/2 on average: a reflection-phase bias of 2 n k0 (dz/2) =
    n k0 dz that |R| and |T| cannot see (a rigid translation). Measured 2026-11-02 on a 180 nm
    n = 2 slab, lambda 1.4 um, 8 midpoint-uniform sub-cell pad offsets, res 30
    (docs/audit/2026-11-02-fdtd-registration.md): mean dphi_r -0.94..-0.95 k0dz (1-D, 2-D, 3-D,
    3-D MO) before; +0.05..+0.06 k0dz after, shrinking 4x per dz halving = the O((k dz)^2) Yee
    dispersion. solve_fdtd_mo_1d always used the nodes (the S2-14 gate) and is bit-identical; this
    helper is now its source too.

    NORMAL COMPONENT. Ez lives at (k+1/2)*dz. The p-pol oblique 2-D kernels take SEPARATE Ez maps
    registered at `material_z_half` (with the node maps on Ez too, p-pol at 30 deg kept a -0.3 k0dz
    bias). The 3-D kernels keep ONE eps array for Ex/Ey/Ez: Ez is identically zero on every
    laterally uniform 3-D case they solve (normal incidence; the 3-D oblique engine is s-pol), so
    their r0/r_co phases are exact to O((k dz)^2); in a STRUCTURED 3-D cell Ez is dz/2 off, the
    vertical twin of the lateral single-array convention (rasterize.layer_eps_cell paints Ex at
    (i+1/2) and Ey at (j+1/2) from the same cell sample) -- a documented limit, not changed here.
    A node lying EXACTLY on an interface takes the layer that starts there (the half-open
    [z, z + t) rule of `layer_masks`): unbiased only on average over offsets (see
    `interface_node_fractions` for the one grid where the tie is systematic)."""
    return np.arange(nz) * dz


def material_z_half(nz, dz):
    """Positions [m] of the NORMAL component Ez: the Yee half nodes (k+1/2)*dz. Its own registration
    where a kernel carries separate Ez maps (the p-pol oblique 2-D kernels, 2026-11-02): the
    per-component Yee assignment of Taflove & Hagness -- each E component takes the material at ITS
    OWN position. (This is exactly the old cell-centre rule: right for Ez, wrong for Ex/Ey.)"""
    return (np.arange(nz) + 0.5) * dz


def layer_masks(z_m, z_start, layers):
    """[(layer, bool mask)] of the half-open [z, z + thickness) z-slab of each layer on the material
    positions `z_m` (normally `material_z`; the nonuniform 1-D grid passes its primal nodes), the
    stack starting at `z_start`. The running `z += thickness` accumulation is the exact float
    sequence of the inline loops it replaces, so a node-registered engine is bit-identical."""
    out = []
    z = z_start
    for L in layers:
        out.append((L, (z_m >= z) & (z_m < z + L.thickness_m)))
        z += L.thickness_m
    return out


def interface_node_fractions(z_nodes, z_start, layers, rtol=1e-9):
    """Tie rule for a node lying EXACTLY ON a material interface (audit S2-14 siblings, 2026-11-02).

    Returns {node index j: [(region, fraction), ...]} for every node j within rtol x its local cell
    of an interface, where region -1 = the superstrate pad, 0..n-1 = `layers[i]`, n = the substrate
    pad, and the fractions are the overlap of E_j's control volume [z_j - h_{j-1}/2, z_j + h_j/2]
    with each region (they sum to 1). The half-open point rule of `layer_masks` gives such a node
    wholly to the region that STARTS there, i.e. moves that interface by -h_{j-1}/2; the
    control-volume arithmetic mean (Hwang & Cangellaris, IEEE MWCL 11:158 (2001) -- the
    second-order effective eps for the tangential E; Taflove & Hagness 3rd ed. on interface
    averaging) puts it back on the node. Only solve_fdtd_1d's NONUNIFORM branch uses this: its grid
    builder (_march_graded) SNAPS every interface onto a node, so there the tie is systematic
    (measured -0.88 k0*h_fine mean r-phase bias, refine 2 and 4, res 30 and 60 -> +-0.03 k0*h with
    the average). On the uniform grids a tie needs pad/dz to be an exact integer -- measure zero
    over the sub-cell offset, and the uniform engines carry Lorentz/chi2/Raman/gain/hot-carrier
    maps with no exact single-pole mixing rule -- so they keep the plain point rule."""
    z = np.asarray(z_nodes, dtype=float)
    n = len(layers)
    bounds = [float(z_start)]
    zz = float(z_start)
    for L in layers:
        zz += L.thickness_m
        bounds.append(zz)
    mid = 0.5 * (z[1:] + z[:-1])
    lo = np.concatenate(([z[0]], mid))
    hi = np.concatenate((mid, [z[-1]]))
    h_loc = np.maximum(hi - lo, 1e-300)
    edges = [-np.inf] + bounds + [np.inf]             # region r spans edges[r+1] .. edges[r+2]
    out = {}
    for b in bounds:
        for j in np.nonzero(np.abs(z - b) <= rtol * h_loc)[0]:
            j = int(j)
            if j in out or hi[j] <= lo[j]:
                continue
            fr = []
            for r in range(-1, n + 1):
                ov = min(hi[j], edges[r + 2]) - max(lo[j], edges[r + 1])
                if ov > 0.0:
                    fr.append((r, ov / (hi[j] - lo[j])))
            out[j] = fr
    return out


def pad_masks(z_m, pad, z_struct):
    """(superstrate, substrate) pad masks on the material positions: z < pad, z >= pad + z_struct."""
    return z_m < pad, z_m >= pad + z_struct


def layer_band_index_max(layers, w_band, n_ends=()):
    """max(1, *n_ends, max over layers and w_band of |sqrt(L.eps_at(w))|) -- the grid-sizing index
    of the 2-D/3-D front ends (the oblique ones pass no `n_ends`; see the module note)."""
    return max(1.0, *n_ends, max(max(abs(np.sqrt(L.eps_at(w))) for w in w_band) for L in layers))


def probe_cells(pad, z_struct, dz):
    """(k_src, k_pL, k_pR): the soft source at 0.35 pad, the reflection probe at 0.7 pad (between
    source and structure) and the transmission probe 0.3 pad past the structure. These are pad
    FRACTIONS against a CELL-count CPML -- the reason `solve2d._check_probe_placement` exists."""
    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    return k_src, k_pL, k_pR


def pulse_width(f_min, f_max, settle):
    """(tau, t0) of the modulated-Gaussian source covering [f_min, f_max]: tau ~ the inverse
    bandwidth, the centre `settle` pulse widths into the run."""
    tau = 1.0 / (np.pi * (f_max - f_min))
    return tau, settle * tau


def source_pulse(f_c, tau, t0, *, source_amp, Lz, dt, t_ring=0.0, t_end_s=None, tail_tau_s=None):
    """(nsteps, tgrid, src): the run length and the soft-source waveform
    source_amp exp(-((t - t0)/tau)^2) cos(2 pi f_c (t - t0)).

    Run length = the pulse (2 t0) + four box transits + a 200 tau tail + `t_ring` (the material
    memory from `solve2d._window_memory_s`; 0 for the 1-D engines). `t_end_s` / `tail_tau_s`
    (default None = the single-pulse window, bit-identical) let a caller that injects a SECOND,
    later or longer pulse extend the window to cover it: the pulse term becomes max(2 t0, t_end_s)
    and the tail 200 max(tau, tail_tau_s) -- solve_fdtd_1d's bichromatic `second_source`, audit
    D-19."""
    t_pulse = 2.0 * t0 if t_end_s is None else max(2.0 * t0, float(t_end_s))
    tail = 200 * tau if tail_tau_s is None else 200 * max(tau, float(tail_tau_s))
    nsteps = int(round((t_pulse + (Lz / C_LIGHT) * 4.0 + tail + t_ring) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))
    return nsteps, tgrid, src
