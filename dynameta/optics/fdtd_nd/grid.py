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


def pad_masks(z_m, pad, z_struct, margin=0.0):
    """(superstrate, substrate) pad masks on the material positions: z < pad, z >= pad + z_struct.
    `margin` > 0 keeps only the nodes whose control volume [z - margin, z + margin] lies WHOLLY in
    the pad (the z-subcell-averaged grids, where the node next to a face is a mixed cell)."""
    if margin == 0.0:
        return z_m < pad, z_m >= pad + z_struct
    return z_m + margin <= pad, z_m - margin >= pad + z_struct


# =================================================================================================
# z-SUBCELL AVERAGING (2026-11-02, the default since this date; z_subcell="point" opts out)
# =================================================================================================
# Point sampling (`material_z` + `layer_masks`) realizes every layer as a whole number of cells, so a
# layer's thickness is rounded by up to one cell and |R|/|T| carry an O(k dz) per-geometry error that
# depends on where the faces fall inside their cells (measured on the fdtd_seam_vs_tmm dielectric
# stack at res 28: point-sampled R = 0.0132 / 0.0308 / 0.0711 against TMM 0.0305, depending only on
# frac(pad/dz)). Subpixel smoothing removes it. Each E node k stands for its control volume
# V_k = [z_k - h_-, z_k + h_+] (h_- = h_+ = dz/2 on a uniform grid); let f_r,k be the fraction of
# V_k occupied by region r (superstrate pad, each layer, substrate pad).
#
#   TANGENTIAL E (Ex/Ey -- every normal-incidence engine, s-pol, the p-pol Ex): the field is
#   continuous across a z-interface, so the cell's D = sum_r f_r eps_r E: the ARITHMETIC mean
#       eps_k = sum_r f_r,k eps_r
#   (Farjadpour et al., Opt. Lett. 31, 2972 (2006); Oskooi et al., Opt. Lett. 34, 2778 (2009): the
#   parallel-field component of the anisotropic smoothing tensor; Hwang & Cangellaris, IEEE MWCL 11,
#   158 (2001): the second-order effective eps of a tangential E at a dielectric interface).
#   NORMAL E (the p-pol Ez half-node map): D is continuous, so 1/eps_k = sum_r f_r,k / eps_r -- the
#   HARMONIC mean (the same papers' perpendicular component).
#
#   DISPERSIVE / NONLINEAR / GAIN LAYERS on the arithmetic channel: the polarization densities of
#   parallel sub-layers add with their volume fractions, P = sum_r f_r P_r, so every term enters the
#   cell scaled by its fill: eps_inf averages, and a pole's STRENGTH scales -- Drude wp^2, Lorentz
#   delta_eps, the gain line's kappa*dN, chi2, chi3, Raman chi3R -- with its shape (gamma, w0, ...)
#   kept. Exact whenever a cell holds at most one pole of each kind (every layer/dielectric face).
#   A cell mixing two DIFFERENT poles of one kind (two Drude slices of a graded profile, say) gets
#   ONE pole with the summed strength and the strength-weighted shape: for Drude,
#   wp^2 = sum f wp_r^2 and gamma = sum f wp_r^2 gamma_r / wp^2, which matches
#   sum_r f_r wp_r^2 / (w^2 + i gamma_r w) through first order in gamma/w -- the error is
#   O(f(1-f) (dgamma/w)^2) of the cell's Drude term (measured in docs/audit/2026-11-02-fdtd-
#   registration.md). The harmonic (Ez) channel has no exact local ADE for a mixed dispersive cell
#   (the mean of 1/eps of a pole is not a pole), so a mixed Ez cell containing any Drude region
#   falls back to POINT sampling at the Ez node (documented + gated, `mix_profile(harmonic=...)`).
#   Lateral inclusion edges of structured cells are NOT smoothed (rasterize.layer_eps_cell is a
#   point raster; recorded follow-up).

Z_SUBCELL_MODES = ("average", "point")


def check_z_subcell(where, mode):
    """Validate the `z_subcell` opt-out kwarg ('average' default; 'point' = the 2026-11-02 2b7a002
    E-node point sampling, bit-for-bit)."""
    if mode not in Z_SUBCELL_MODES:
        raise ValueError("{}: z_subcell must be one of {}, got {!r}".format(where, Z_SUBCELL_MODES, mode))
    return mode


def _region_bounds(z_start, layers):
    """Layer boundaries with the EXACT float accumulation of `layer_masks`."""
    b = [z_start]
    z = z_start
    for L in layers:
        z = z + L.thickness_m
        b.append(z)
    return b


def z_fill_fractions(z_m, z_start, layers, *, dz=None, lo=None, hi=None):
    """F[r, k]: the fraction of node k's control volume occupied by region r (r = 0 superstrate pad,
    1..n the layers, n+1 substrate pad; each column sums to 1). Uniform grid: pass `dz` (control
    volume [z - dz/2, z + dz/2]); nonuniform: pass the volume ends `lo`, `hi` per node."""
    z = np.asarray(z_m, dtype=float)
    if lo is None:
        lo, hi = z - 0.5 * dz, z + 0.5 * dz
    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    b = _region_bounds(z_start, layers)
    edges = [-np.inf] + b + [np.inf]
    width = hi - lo
    F = np.empty((len(layers) + 2, z.size))
    for r in range(len(layers) + 2):
        ov = np.minimum(hi, edges[r + 1]) - np.maximum(lo, edges[r])
        F[r] = np.where(ov > 0.0, ov, 0.0) / width
    return F


def point_regions(z_m, z_start, layers):
    """Region index (0 = superstrate pad .. n+1 = substrate pad) holding each node under the
    half-open point rule of `layer_masks`."""
    return np.searchsorted(np.asarray(_region_bounds(z_start, layers)), np.asarray(z_m, dtype=float),
                           side="right")


ND_POLES = (("wp", 2, ("gam",)),                 # Drude: wp^2 fill-weighted, gamma shape
            ("ldeps", 1, ("lw0", "lgam")),       # Lorentz: delta_eps fill-weighted, (w0, gamma) shape
            ("chi3R", 1, ("rw", "rgam")),        # Raman: chi3R fill-weighted, (W_R, gamma_R) shape
            ("gkdn", 1, ("gw", "gdw")))          # gain line: kappa*dN fill-weighted, (w_a, dw_a) shape


def nd_region_values(layers, n_super, n_sub, kerr):
    """Per-region (superstrate pad, layers..., substrate pad) values of every 2-D/3-D FDTDLayer
    material channel, for `mix_profile(..., poles=ND_POLES)` (z-subcell averaging, 2026-11-02)."""
    def col(get, pad_lo=0.0, pad_hi=0.0):
        return [pad_lo] + [float(get(L)) for L in layers] + [pad_hi]
    return {"eps_inf": col(lambda L: L.eps_inf, float(n_super) ** 2, float(n_sub) ** 2),
            "wp": col(lambda L: L.drude_wp_rad_s), "gam": col(lambda L: L.drude_gamma_rad_s),
            "lw0": col(lambda L: L.lorentz_w0_rad_s), "lgam": col(lambda L: L.lorentz_gamma_rad_s),
            "ldeps": col(lambda L: L.lorentz_delta_eps),
            "chi3": col(lambda L: L.chi3_m2_V2 if kerr else 0.0), "chi2": col(lambda L: L.chi2_m_V),
            "chi3R": col(lambda L: L.raman_chi3_m2_V2), "rw": col(lambda L: L.raman_w_rad_s),
            "rgam": col(lambda L: L.raman_gamma_rad_s), "gw": col(lambda L: L.gain_w_rad_s),
            "gdw": col(lambda L: L.gain_dw_rad_s),
            "gkdn": col(lambda L: L.gain_kappa_C2_kg * L.gain_dN_m3)}


def mix_profile(F, values, *, poles=(), harmonic=(), point_index=None):
    """Per-node material maps from region values and fill fractions -- THE one implementation of the
    subcell rule above, consumed by every FDTD engine.

    values: {name: per-region sequence (len n+2)}; an entry may itself be an array (e.g. a lateral
            (nx, ny) raster per slab): the result then has shape value.shape + (nz,).
    poles:  ((strength_name, power, shape_names), ...): the region strength enters as value**power
            (Drude wp: power 2), is fill-weighted and mapped back (**1/power); each shape parameter
            becomes the |strength|-weighted mean (exact when one region carries the pole).
    harmonic: names averaged as 1/sum(f/v) (the normal-E channel); a mixed node whose regions carry
            any nonzero pole strength falls back to the point value values[point_index[k]].
    Nodes with a single region (max F == 1) take that region's values EXACTLY (no rounding)."""
    F = np.asarray(F, dtype=float)
    nreg, nz = F.shape
    pure = F.max(axis=0) == 1.0
    pure_r = np.argmax(F, axis=0)
    mixed = ~pure
    out = {}
    pole_of = {}
    for sname, power, shapes in poles:
        pole_of[sname] = (power, shapes)
    has_pole = np.zeros(nz, dtype=bool)                   # node touching a nonzero pole strength
    for sname in pole_of:
        nzr = np.array([bool(np.any(np.asarray(x, dtype=float) != 0.0)) for x in values[sname]])
        if nzr.any():
            has_pole |= (F[nzr] > 0.0).any(axis=0)
    for name, vals in values.items():
        V = np.asarray([np.asarray(x, dtype=float) for x in vals])          # (nreg, ...)
        res = np.moveaxis(V[pure_r], 0, -1).copy()                          # pure: exact region value
        if mixed.any():
            Fm = F[:, mixed]                                                # (nreg, nmix)
            Vx = np.moveaxis(V, 0, -1)                                      # (..., nreg)
            if name in harmonic:
                with np.errstate(divide="ignore"):
                    inv = np.where(Vx != 0.0, 1.0 / np.where(Vx != 0.0, Vx, 1.0), np.inf)
                mval = 1.0 / (inv @ Fm)
            elif name in pole_of:                                         # pole STRENGTH: fill-weighted
                mval = np.sqrt((Vx ** 2) @ Fm) if pole_of[name][0] == 2 else Vx @ Fm
            else:
                shape_of = [s for s, (p, shp) in pole_of.items() if name in shp]
                if shape_of:
                    sname = shape_of[0]
                    S = np.abs(np.moveaxis(np.asarray([np.asarray(x, dtype=float)
                                                       for x in values[sname]]), 0, -1)) ** pole_of[sname][0]
                    w = S @ Fm
                    num = (S * Vx) @ Fm
                    mval = np.where(w > 0.0, num / np.where(w > 0.0, w, 1.0), Vx @ Fm)
                else:
                    mval = Vx @ Fm
            res[..., mixed] = mval
            if harmonic:
                # normal-E channel: a mixed cell touching a dispersive region has no exact local ADE
                # (mean of 1/eps of a pole is not a pole) -> EVERY field of that cell is POINT-sampled
                # at the node (the documented fallback; gated in tests/test_fdtd_registration.py)
                fb = mixed & has_pole
                if fb.any():
                    res[..., fb] = np.moveaxis(V[np.asarray(point_index)[fb]], 0, -1)
        out[name] = res
    return out


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
