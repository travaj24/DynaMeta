"""Laminate averaging of a sampled permittivity onto PMM segments (the PMM bridge's lateral
partition and its z-slab merge). Pure numpy; no lumenairy import.

WHY A LAMINATE TENSOR, AND WHICH ONE. The 1-D PMM represents a layer as walls and segments with a
CONSTANT permittivity (scalar or 3x3) per segment. A permittivity sampled on a grid (a carrier
profile that differs under a metal strip and between strips) has to be averaged onto cells. In a
cell that is THIN in one direction compared with its other extent (and with the wavelength), the
field component normal to the thin direction keeps its displacement continuous and sees the
series (harmonic) mean; the components along the thin direction's planes see the parallel
(arithmetic) mean. So:

  a z slab (always thin against its lateral extent)          ->  diag(arith, arith, harm)
  a lateral interval NARROWER than the structured layer       ->  diag(harm, arith, arith)
  a lateral interval WIDER than the layer (a thin film)       ->  arithmetic for every component

The last line is why the lateral rule depends on the aspect ratio: in a film thinner than the
interval, E_x is pinned by tangential continuity with the layers above and below and is nearly
uniform across the interval (measured, TM, sixteen 25 nm lamellae of a Drude film over a mirror,
one segment per period against the resolved solve: in a 6 nm film the arithmetic mean is 2.8x closer
than the harmonic one, in a 50 nm film the harmonic one is 4.0x closer; tests/test_pmm_lateral_partition
.py). lateral_average='auto' applies that rule per interval. For a partition that
resolves the variation the choice is second order (measured 4e-6 between the two on a ramped
film). The interpolant is the periodic piecewise-LINEAR one through the samples (for a Drude-type
permittivity, linear in the carrier density); both means integrate that same function exactly (the
harmonic one via the closed form of int dx / (a + b x)).

ADAPTIVE PARTITION. `partition_edges` starts from seed edges (0, 1 and every wall the other layers
own, so it adds no union-grid walls of its own where it can avoid it) and cuts the interval with
the largest spread  max |eps - mean| / max(|mean|, spread_floor)  (over the interval and over every
z slab of the layer) until every interval is within `rtol`, the interval count reaches
`max_segments`, or the interval would fall below `min_width_frac` of the period. The spread is
exact for the piecewise-linear interpolant (its extremes sit on breakpoints). The floor (default
1, the vacuum value) keeps an interval whose mean passes near zero from demanding an unbounded
refinement. split='balanced' (default) cuts at the sample that minimizes the two-piece constant
misfit (the cut lands where the profile varies; a lateral ramp is isolated in a few cuts);
split='bisect' cuts at the midpoint. Measured on a ramped two-profile film under a strip:
13 balanced segments reach the partition error of ~26 bisected ones.

The one implementation of the lateral partition average (one kernel, shared). Conventions: exp(-i omega t),
Im(eps) >= 0 for loss; lateral positions in metres in [0, period]; widths as period fractions.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from dynameta.core.lateral import check_periodic_axis

__all__ = ["periodic_samples", "interval_means", "partition_edges", "laminate_segments",
           "interval_domain_size", "z_interval_harmonic", "z_merge_groups", "laminate_z"]

_EDGE_TOL = 1e-12          # period fractions: edges closer than this are one edge
_SERIES_U = 1e-3           # |b/a - 1| below which the harmonic piece uses its Taylor series


def periodic_samples(x_m, values, period_m: float, *, where: str = "lateral profile"
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """(xs, vs): breakpoints covering [0, period] and the values there, for the periodic
    piecewise-linear interpolant through the samples.

    x_m    : (Nx,) strictly increasing sample positions inside [0, period] (metres).
    values : (..., Nx) complex samples (leading axes are independent profiles).
    A sample is added at 0 (and at the period) by periodic wrap when the grid does not reach that
    end; an endpoint-inclusive grid (samples at 0 AND at the period) is used as given, so a
    duplicated periodic end point costs nothing and a non-periodic pair of end values is a step
    on the cell boundary (a wall of every segment list anyway)."""
    v = np.asarray(values, dtype=np.complex128)
    P = float(period_m)
    if v.shape[-1] != np.asarray(x_m).size:
        raise ValueError("{}: {} lateral samples but {} positions".format(where, v.shape[-1],
                                                                          np.asarray(x_m).size))
    # the unit / span / ordering guard is core.lateral.check_periodic_axis, shared by every optical
    # route that reads a lateral axis (the 2-D FEM had none before)
    x = check_periodic_axis(x_m, P, where=where)
    tol = 1e-9 * P
    xs, vs = [x], [v]
    if x[0] > tol:                     # wrap the last sample to x - P, interpolate to 0
        xl, vl = x[-1] - P, v[..., -1:]
        w = (0.0 - xl) / (x[0] - xl)
        xs.insert(0, np.array([0.0]))
        vs.insert(0, (1.0 - w) * vl + w * v[..., :1])
    else:
        x[0] = 0.0
    if x[-1] < P - tol:
        xr, vr = x[0] + P, v[..., :1]
        w = (P - x[-1]) / (xr - x[-1])
        xs.append(np.array([P]))
        vs.append((1.0 - w) * v[..., -1:] + w * vr)
    else:
        x[-1] = P
    return np.concatenate(xs), np.concatenate(vs, axis=-1)


def _harmonic_piece(a, b):
    """int_0^1 dt / (a + (b - a) t) for complex arrays a, b (the straight segment a -> b must not
    pass through 0)."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    if np.any(a == 0) or np.any(b == 0):
        raise ValueError("laminate average: a permittivity sample is exactly zero; the harmonic "
                         "mean is undefined (use lateral_average='arithmetic')")
    cross = np.conj(a) * b
    through0 = (np.abs(cross.imag) <= 1e-12 * np.abs(cross)) & (cross.real < 0.0)
    if np.any(through0):
        raise ValueError("laminate average: the interpolated permittivity passes through zero "
                         "between two samples (a lossless sign change); its harmonic mean is "
                         "singular there (use lateral_average='arithmetic', or add loss)")
    u = b / a - 1.0
    small = np.abs(u) < _SERIES_U
    us = np.where(small, u, 0.0)
    series = (1.0 - us / 2.0 + us ** 2 / 3.0 - us ** 3 / 4.0 + us ** 4 / 5.0) / a
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.log(np.where(small, 1.0 + 0j, b / a)) / np.where(small, 1.0 + 0j, b - a)
    return np.where(small, series, exact)


def interval_means(xs, vs, edges_frac, period_m: float, spread_floor: float = 0.0):
    """Exact interval averages of the piecewise-linear interpolant (xs, vs) from periodic_samples.

    Returns (arith, harm, spread), each (..., K) for the K intervals between consecutive
    `edges_frac` (period fractions, ascending, first 0 and last 1):
      arith  = (1/w) int eps dx
      harm   = w / int (1/eps) dx
      spread = max over the interval of |eps - arith| / max(|arith|, spread_floor)."""
    P = float(period_m)
    e = np.asarray(edges_frac, dtype=np.float64) * P
    e[0], e[-1] = 0.0, P
    xs = np.asarray(xs, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.complex128)
    # an edge within round-off of a sample IS that sample (a fraction -> metre round trip is off
    # by ~1 ulp; interpolating there would mix in the neighbouring segment's slope)
    j = np.clip(np.searchsorted(xs, e), 1, xs.size - 1)
    near = np.where(np.abs(xs[j] - e) < np.abs(xs[j - 1] - e), xs[j], xs[j - 1])
    e = np.where(np.abs(near - e) <= _EDGE_TOL * P, near, e)
    B = np.union1d(xs, e)
    # drop breakpoints within round-off of an edge (keep the edge itself)
    near = np.min(np.abs(B[:, None] - e[None, :]), axis=1) <= _EDGE_TOL * P
    is_edge = np.isin(B, e)
    B = B[~near | is_edge]
    j = np.clip(np.searchsorted(xs, B, side="right") - 1, 0, xs.size - 2)
    w = (B - xs[j]) / (xs[j + 1] - xs[j])
    vb = vs[..., j] * (1.0 - w) + vs[..., j + 1] * w                # (..., nB)
    h = np.diff(B)                                                  # piece widths
    a_piece = 0.5 * h * (vb[..., :-1] + vb[..., 1:])
    r_piece = h * _harmonic_piece(vb[..., :-1], vb[..., 1:])
    start = np.searchsorted(B, e[:-1])                              # first piece of each interval
    # the interval width as the sum of its own pieces (not e[i+1] - e[i]): a constant interval
    # then averages to its value up to one rounding, whatever the piece count
    widths = np.add.reduceat(h, start)
    arith = np.add.reduceat(a_piece, start, axis=-1) / widths
    harm = widths / np.add.reduceat(r_piece, start, axis=-1)
    piece_iv = np.searchsorted(e, 0.5 * (B[:-1] + B[1:]), side="right") - 1
    m = arith[..., piece_iv]
    dev = (np.maximum(np.abs(vb[..., :-1] - m), np.abs(vb[..., 1:] - m))
           / np.maximum(np.abs(m), float(spread_floor)))
    spread = np.maximum.reduceat(dev, start, axis=-1)
    return arith, harm, spread


def interval_domain_size(xs, vs, edges_frac, period_m: float) -> np.ndarray:
    """Lateral size of the domains the profile forms inside each interval: width / (1 + the
    number of times the profile crosses its interval mean, the larger of the real- and
    imaginary-part counts, maximized over rows). A single ramp gives width / 2, n alternating
    lamellae about width / n. Used by lateral_average='auto' (the aspect-ratio rule)."""
    P = float(period_m)
    e = np.asarray(edges_frac, dtype=np.float64) * P
    e[0], e[-1] = 0.0, P
    xs = np.asarray(xs, dtype=np.float64)
    vs = np.atleast_2d(np.asarray(vs, dtype=np.complex128))
    arith, _, _ = interval_means(xs, vs, edges_frac, P)
    out = np.empty(e.size - 1)
    for k in range(e.size - 1):
        sel = (xs >= e[k]) & (xs <= e[k + 1])
        d = vs[:, sel] - arith[:, k:k + 1]
        tiny = 1e-12 * np.abs(arith[:, k:k + 1])          # round-off is not a crossing
        n = 0
        for part in (d.real, d.imag):
            sg = np.where(np.abs(part) <= tiny, 0.0, np.sign(part))
            nz = [row[row != 0] for row in sg]
            n = max([n] + [int(np.sum(r[1:] * r[:-1] < 0)) for r in nz])
        out[k] = (e[k + 1] - e[k]) / (1 + n)
    return out


def _clean_edges(edges) -> np.ndarray:
    e = np.sort(np.concatenate([np.asarray(edges, dtype=np.float64).ravel(), [0.0, 1.0]]))
    e = e[(e >= -_EDGE_TOL) & (e <= 1.0 + _EDGE_TOL)]
    e = np.clip(e, 0.0, 1.0)
    keep = [e[0]]
    for x in e[1:]:
        if x - keep[-1] > _EDGE_TOL:
            keep.append(x)
    keep[0], keep[-1] = 0.0, 1.0
    if len(keep) < 2:
        keep = [0.0, 1.0]
    return np.asarray(keep)


def _split_point(xf, vb, a: float, b: float, min_w: float, floor: float = 0.0) -> float:
    """Where to cut the interval [a, b] (period fractions): the breakpoint (inside, at least
    min_w from both ends) that minimizes the summed relative squared misfit of a two-piece
    constant fit to the profiles (rows of vb sampled at the breakpoints xf), i.e. the cut that
    best separates where the profile varies from where it does not. Falls back to the midpoint.
    The stopping test stays the exact spread; this only places the cut."""
    sel = (xf >= a) & (xf <= b)
    xi, vi = xf[sel], vb[:, sel]
    mid = 0.5 * (a + b)
    if xi.size < 3:
        return mid
    xe = np.concatenate([[a], 0.5 * (xi[1:] + xi[:-1]), [b]])
    w = np.diff(xe)                                                  # cell weight per sample
    a2 = np.abs(vi) ** 2
    c0, c1, c2 = np.cumsum(w), np.cumsum(w * vi, axis=1), np.cumsum(w * a2, axis=1)
    h0, h1, h2 = 0.5 * w, 0.5 * w * vi, 0.5 * w * a2                 # the cut sample counts half
    W_L, S1_L, S2_L = c0 - h0, c1 - h1, c2 - h2
    W_R, S1_R, S2_R = c0[-1] - W_L, c1[:, -1:] - S1_L, c2[:, -1:] - S2_L
    ok = (xi > a + min_w) & (xi < b - min_w) & (W_L > 0) & (W_R > 0)
    if not np.any(ok):
        return mid
    with np.errstate(divide="ignore", invalid="ignore"):
        sse = (S2_L - np.abs(S1_L) ** 2 / W_L) + (S2_R - np.abs(S1_R) ** 2 / W_R)
        scale = np.maximum(np.abs(c1[:, -1] / c0[-1]), floor) ** 2   # spread scale^2 per row
        cost = np.sum(sse / scale[:, None], axis=0)
    cost = np.where(ok & np.isfinite(cost), cost, np.inf)
    k = int(np.argmin(cost))
    return float(xi[k]) if np.isfinite(cost[k]) else mid


def partition_edges(profiles, x_m, period_m: float, seed_edges_frac: Sequence[float] = (), *,
                    rtol: float = 0.1, max_segments: int = 24,
                    min_width_frac: float = 1.0 / 4096, refine: bool = True,
                    split: str = "balanced", spread_floor: float = 1.0):
    """Adaptive lateral partition shared by every profile (z slab) of one layer.

    profiles : (Ns, Nx) complex lateral profiles sampled at x_m (one row per z slab).
    Returns (edges_frac, report) with report = {'n_segments', 'max_spread', 'converged',
    'capped'}: max_spread is the largest relative spread left in any interval of any slab;
    converged = max_spread <= rtol; capped = refinement stopped at max_segments or at the
    minimum width with intervals still above rtol. refine=False evaluates the seed partition
    as given (an explicit user partition)."""
    if not (float(rtol) > 0.0):
        raise ValueError("partition_rtol must be > 0, got {!r}".format(rtol))
    if int(max_segments) < 1:
        raise ValueError("max_segments must be >= 1, got {!r}".format(max_segments))
    if split not in ("balanced", "bisect"):
        raise ValueError("split must be 'balanced' or 'bisect', got {!r}".format(split))
    xs, vs = periodic_samples(x_m, np.atleast_2d(profiles), period_m)
    edges = _clean_edges(seed_edges_frac)
    capped = False
    while True:
        _, _, spread = interval_means(xs, vs, edges, period_m, spread_floor)
        sp = np.max(spread, axis=0)                                  # per interval
        if not refine:
            break
        widths = np.diff(edges)
        cand = (sp > float(rtol)) & (widths >= 2.0 * float(min_width_frac))
        if not np.any(cand):
            capped = bool(np.any(sp > float(rtol)))
            break
        if edges.size - 1 >= int(max_segments):
            capped = True
            break
        k = int(np.argmax(np.where(cand, sp, -1.0)))
        if split == "balanced":
            cut = _split_point(xs / float(period_m), vs, edges[k], edges[k + 1],
                               float(min_width_frac), float(spread_floor))
        else:
            cut = 0.5 * (edges[k] + edges[k + 1])
        edges = np.insert(edges, k + 1, cut)
    max_spread = float(np.max(sp)) if sp.size else 0.0
    return edges, {"n_segments": int(edges.size - 1), "max_spread": max_spread,
                   "converged": bool(max_spread <= float(rtol)), "capped": capped}


def laminate_segments(profiles, x_m, period_m: float, edges_frac, mode: str = "auto",
                      zz_profiles=None, layer_thickness_m: Optional[float] = None):
    """Per-slab, per-interval averaged components on the partition `edges_frac`.

    Returns (widths (K,), comps (Ns, K, 3)) with comps[..., :] = (eps_xx, eps_yy, eps_zz):
      mode='laminate'   : (harm, arith, arith) -- the field across the x variation (E_x) sees the
                          series mean, E_y / E_z the parallel one: right for an interval NARROW
                          against the structured layer's thickness (tall lamellae);
      mode='arithmetic' : (arith, arith, arith) -- right for an interval WIDE against the layer
                          thickness: in a thin film E_x is pinned by tangential continuity with
                          the layers above and below, so it is nearly uniform across the interval
                          and sees the parallel mean (measured on a 6 nm film with 25 nm
                          lamellae: 2.8x closer than the laminate);
      mode='auto'       : per interval, 'laminate' when its width is below layer_thickness_m,
                          else 'arithmetic' (the aspect-ratio rule; needs layer_thickness_m).
    zz_profiles (optional, same shape as profiles): the slab's z-component profile when it
    differs from `profiles` (a z-laminate slab: the harmonic mean along z of each column); its
    lateral ARITHMETIC mean is eps_zz (E_z is tangential to the lateral variation)."""
    if mode not in ("auto", "laminate", "arithmetic"):
        raise ValueError("lateral_average must be 'auto', 'laminate' or 'arithmetic', got "
                         "{!r}".format(mode))
    if mode == "auto" and not (layer_thickness_m is not None and float(layer_thickness_m) > 0):
        raise ValueError("lateral_average='auto' needs the layer thickness")
    xs, vs = periodic_samples(x_m, np.atleast_2d(profiles), period_m)
    edges = np.asarray(edges_frac, dtype=np.float64)
    arith, harm, _ = interval_means(xs, vs, edges, period_m)
    if mode == "auto":
        narrow = interval_domain_size(xs, vs, edges, period_m) < float(layer_thickness_m)
        xx = np.where(narrow, harm, arith)
    else:
        xx = harm if mode == "laminate" else arith
    zz = arith
    if zz_profiles is not None:
        xz, vz = periodic_samples(x_m, np.atleast_2d(zz_profiles), period_m)
        zz, _, _ = interval_means(xz, vz, edges, period_m)
    comps = np.stack([xx, arith, zz], axis=-1)
    return np.diff(edges), comps


def z_interval_harmonic(values_z, z_axis):
    """Harmonic means of the z-linear interpolant over each native z interval, ascending z.

    values_z : (Nz, ...) complex samples along z (leading axis); z_axis : (Nz,) strictly
    monotonic (a descending axis is reversed, as slice_eps_field does). Returns (Nz - 1, ...)."""
    v = np.asarray(values_z, dtype=np.complex128)
    z = np.asarray(z_axis, dtype=np.float64).ravel()
    if z.size >= 2 and z[0] > z[-1]:
        v = v[::-1]
    return 1.0 / _harmonic_piece(v[:-1], v[1:])


def z_merge_groups(comps, thickness_m, rtol: Optional[float]) -> List[Tuple[int, int]]:
    """Greedy grouping of consecutive z slabs (axis 0 of comps, shape (Ns, ...)): a group grows
    while every component of every slab in it stays within `rtol` of the group's thickness-
    weighted mean, relative to |mean|. None or 0 -> one group per slab (no merge). Returns
    [(start, stop)) index pairs, bottom (index 0) first."""
    c = np.asarray(comps, dtype=np.complex128)
    t = np.asarray(thickness_m, dtype=np.float64)
    n = c.shape[0]
    if rtol is None or float(rtol) <= 0.0 or n <= 1:
        return [(i, i + 1) for i in range(n)]
    tol = float(rtol)
    flat = c.reshape(n, -1)
    groups, i = [], 0
    while i < n:
        j = i + 1
        while j < n:
            blk = flat[i:j + 1]
            tt = t[i:j + 1, None]
            m = np.sum(tt * blk, axis=0) / np.sum(tt)
            dev = np.max(np.abs(blk - m[None, :]) / np.abs(m)[None, :])
            if not (dev <= tol):
                break
            j += 1
        groups.append((i, j))
        i = j
    return groups


def laminate_z(comps, thickness_m, groups):
    """Merge z slabs per group as a z-laminate: (xx, yy) thickness-weighted arithmetic, zz
    thickness-weighted harmonic (D_z is continuous through the stack). comps (Ns, ..., 3),
    thickness_m (Ns,). Returns (thickness (G,), comps (G, ..., 3))."""
    c = np.asarray(comps, dtype=np.complex128)
    t = np.asarray(thickness_m, dtype=np.float64)
    out_t, out_c = [], []
    for a, b in groups:
        tt = t[a:b]
        T = float(np.sum(tt))
        w = (tt / T).reshape((-1,) + (1,) * (c.ndim - 2))
        blk = c[a:b]
        if b - a == 1:
            out_t.append(float(t[a]))
            out_c.append(blk[0])
            continue
        xx = np.sum(w * blk[..., 0], axis=0)
        yy = np.sum(w * blk[..., 1], axis=0)
        zz = 1.0 / np.sum(w / blk[..., 2], axis=0)
        out_t.append(T)
        out_c.append(np.stack([xx, yy, zz], axis=-1))
    return np.asarray(out_t), np.stack(out_c, axis=0)
