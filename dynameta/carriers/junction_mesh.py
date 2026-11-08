"""
Opt-in UNIFORM mesh band around p-n junctions for the 2-D DEVSIM builder (v0.15.1; user decision 2026-11-08:
"please add it as an opt-in and see how it compares"). Audit note: docs/audit/2026-11-08-v0.15-junction-mesh-band.md.

WHY. DEVSIM's mesh-line mesher (add_2d_mesh_line ns / ps) fills each interval between two lines with a
geometric progression that starts at the LEFT line's ps and puts the leftover length into ONE edge that ends
on the RIGHT line (MEASURED: lines 0 / 10.3 nm at 1 nm give ten 1 nm edges and one 0.3 nm edge). When the
right line is a junction, that odd ("remainder") edge sits on the junction: 0.7716 nm next to the 1 nm
junction line of the extended-precision GaN-like diode, which put +4.9e-3 on its recombination current
(docs/audit/2026-11-06-v0.15-followups.md item 3). The recombination integrand (SRH: a peak of width
V_t / E at the n = p point) and the BTBT rate (a cusp e^(-k|x|) at the field maximum) are both narrow peaks
in the junction, and the node / edge quadrature of a narrow peak is accurate on a UNIFORM grid around it and
not across one irregular edge (the same family as the BTBT fixed-edge error of tunneling's MESH RESOLUTION).

WHAT. The band replaces the builder's lines inside [x_j - w, x_j + w] (and likewise in z) by lines at
x_j + k h, every one with ns = ps = h, so DEVSIM adds nothing between them and the junction x_j is a node
with two equal edges. DEVSIM's own grading (and its remainder edge) is left OUTSIDE the band, a half-width
w away from the peak.

DEFAULTS (JunctionBand(), both None): physical lengths of the zero-bias ABRUPT-junction depletion
approximation on the doping found on each side (abrupt_junction_scales):
  half-width w = W_0 = x_p0 + x_n0, the whole zero-bias depletion width, on EACH side of the junction. The
      SRH peak sits at the n = p point, which lies inside the zero-bias depletion region at every forward
      bias (it moves into the lighter side, at most ~0.3 x_dep from the junction for a one-sided junction);
      the BTBT / field peak sits at the junction at every bias. W_0 on both sides covers the depletion
      region however the doping splits it.
  spacing  h = min(V_t / E_0, sqrt(12 tol) / k_0): V_t / E_0 is the width of the SRH peak at its NARROWEST
      (E_0 the zero-bias peak field; forward bias lowers E and widens the peak), and for a region with
      local BTBT the second term is the tunnelling module's resolution rule h <= 0.35 / k
      ((k h)^2 / 12 <= tunneling.BTBT_QUADRATURE_TOL) with k_0 = B (dE/dx) / E_0^2 on the heavier side.
      The builder caps h at the spacing of a mesh line already sitting ON the junction (the cell mid line,
      a layer interface), so the default never coarsens what the design asked for there.
  A triangle field gives E_0 = 2 V_bi / W_0, so the band holds 2 W_0 / h ~ 4 V_bi / V_t nodes per line
  (~150 for Si, ~500 for a GaN-like junction) independently of the doping level.
The estimate is Boltzmann (V_bi = V_t ln(N_A N_D / (n_i,p n_i,n))) and ignores band offsets; for degenerate
doping it under-estimates V_bi by a few V_t (INFERRED). Override either field with a length when you know
better (e.g. the junction-line spacing you already validated).

The band is a MESH option: it changes the discretization, never the physics. Off (the builder's default,
junction_band=None) every mesh line, node and result is byte-identical to the builder without it.

Everything here is devsim-free (numpy only): the probe that FINDS the junctions on the device's own
NetDoping expression lives in devsim_layered.LayeredDevsimBuilder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from dynameta.constants import EPS0, KB, Q_E

__all__ = ["JunctionBand", "JunctionScales", "abrupt_junction_scales", "band_lines", "resolve_junction_band",
           "sign_changes"]


@dataclass(frozen=True)
class JunctionBand:
    """Settings of the opt-in uniform junction band (LayeredDevsimBuilder(junction_band=...)).

    half_width_m: band half-width [m] on each side of every junction; None (default) = W_0, the zero-bias
        depletion width of the junction (abrupt_junction_scales).
    spacing_m: uniform spacing [m] inside the band; None (default) = min(V_t / E_0, the BTBT resolution
        length when the region has local BTBT, the spacing of a mesh line already on the junction).
    WHICH TO USE (measured, audit note): the defaults for a forward SRH current or a BTBT current at an in-region
    (lateral) junction; halve spacing_m (or keep the defaults and refine twice) when you want a Richardson
    estimate -- on the band the junction-peak error is cleanly second order. The band does not improve a
    node-sampled peak rate (that needs a smaller spacing), nor a diffusion-limited current.
    Pass True to the builder for JunctionBand() (both defaults)."""
    half_width_m: Optional[float] = None
    spacing_m: Optional[float] = None

    def __post_init__(self):
        for nm in ("half_width_m", "spacing_m"):
            v = getattr(self, nm)
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0):
                raise ValueError("JunctionBand.{} must be a positive length in metres or None; got {!r}"
                                 .format(nm, v))


def resolve_junction_band(value) -> Optional[JunctionBand]:
    """The builder argument: None / False -> None (off, the default); True -> JunctionBand() (both physical
    defaults); a JunctionBand -> itself. Anything else raises TypeError."""
    if value is None or value is False:
        return None
    if value is True:
        return JunctionBand()
    if isinstance(value, JunctionBand):
        return value
    raise TypeError("junction_band must be None, a bool or a junction_mesh.JunctionBand; got {!r}".format(value))


@dataclass(frozen=True)
class JunctionScales:
    """Zero-bias abrupt-junction lengths (depletion approximation), SI units."""
    V_bi: float          # built-in potential [V] (Boltzmann; floored at V_t)
    x_p: float           # depletion depth into the p side [m]
    x_n: float           # depletion depth into the n side [m]
    E0: float            # peak field at the junction [V/m]
    peak_width: float    # V_t / E0: width of the SRH recombination peak at zero bias [m]
    btbt_width: Optional[float]   # sqrt(12 tol) / k_0 for local BTBT (None without BTBT) [m]

    @property
    def W0(self) -> float:
        """x_p + x_n, the zero-bias depletion width [m] (the default band half-width)."""
        return self.x_p + self.x_n

    @property
    def spacing(self) -> float:
        """The default band spacing [m]: min(peak_width, btbt_width)."""
        return self.peak_width if self.btbt_width is None else min(self.peak_width, self.btbt_width)


def abrupt_junction_scales(N_A: float, N_D: float, *, eps_r_p: float, eps_r_n: float, n_i_p: float,
                           n_i_n: float, T_K: float, btbt_B_V_per_m: Optional[float] = None,
                           btbt_tol: Optional[float] = None) -> JunctionScales:
    """Depletion approximation of an abrupt p-n junction at zero bias (Sze, Physics of Semiconductor
    Devices, 3rd ed., Eqs. 2.2-2.10 for a homojunction; the two-permittivity form of the heterojunction,
    Eq. 2.88, without band offsets): V_bi = V_t ln(N_A N_D / (n_i,p n_i,n)) (floored at V_t),
    charge balance N_A x_p = N_D x_n and V_bi = q N_A x_p^2 / (2 eps_p) + q N_D x_n^2 / (2 eps_n), so
    x_n = sqrt(2 V_bi eps_p eps_n N_A / (q N_D (eps_p N_A + eps_n N_D))), E_0 = q N_D x_n / eps_n.
    btbt_B_V_per_m (largest B of the region's local BTBT sets): k_0 = B q max(N_A/eps_p, N_D/eps_n) / E_0^2,
    btbt_width = sqrt(12 btbt_tol) / k_0 (btbt_tol None -> tunneling.BTBT_QUADRATURE_TOL)."""
    for nm, v in (("N_A", N_A), ("N_D", N_D), ("n_i_p", n_i_p), ("n_i_n", n_i_n), ("eps_r_p", eps_r_p),
                  ("eps_r_n", eps_r_n), ("T_K", T_K)):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError("abrupt_junction_scales: {} must be finite and > 0; got {!r}".format(nm, v))
    vt = KB * float(T_K) / Q_E
    vbi = max(vt, vt * math.log(N_A * N_D / (n_i_p * n_i_n)))
    ep, en = eps_r_p * EPS0, eps_r_n * EPS0
    x_n = math.sqrt(2.0 * vbi * ep * en * N_A / (Q_E * N_D * (ep * N_A + en * N_D)))
    x_p = x_n * N_D / N_A
    E0 = Q_E * N_D * x_n / en
    bw = None
    if btbt_B_V_per_m is not None:
        if btbt_tol is None:
            from dynameta.carriers.tunneling import BTBT_QUADRATURE_TOL as btbt_tol
        k0 = float(btbt_B_V_per_m) * Q_E * max(N_A / ep, N_D / en) / E0 ** 2
        bw = math.sqrt(12.0 * float(btbt_tol)) / k0
    return JunctionScales(V_bi=vbi, x_p=x_p, x_n=x_n, E0=E0, peak_width=vt / E0, btbt_width=bw)


def band_lines(lines: Mapping[float, Tuple[float, float]], fixed: Iterable[float],
               junctions: Sequence[Tuple[float, float, float]]) -> Dict[float, Tuple[float, float]]:
    """Mesh lines {position: (ns, ps)} with a uniform band around each junction (position, half_width,
    spacing). `fixed` are the lines that must stay (region boundaries, the cell edges; they bound the domain);
    every other line strictly inside a band is dropped.

    Rules: a junction within 1e-3 h of an existing line is snapped onto it. Bands that overlap form one
    cluster; its ANCHORS are its junctions plus the fixed lines inside it, and each anchor-to-anchor span is
    cut into n = ceil(D / h) equal cells (h = the cluster's smallest spacing). The two end spans continue the
    adjacent span's spacing from the anchor outwards (anchor -/+ k h, so the junction keeps two equal edges)
    up to the half-width, or up to the first fixed line, where a remainder < h / 4 is merged into the last
    cell. Each band line gets ns / ps = the spacing of the cell on that side; the outermost line's outward
    spacing is the band spacing, from which DEVSIM grades to the next line as before."""
    out = {float(p): (float(v[0]), float(v[1])) for p, v in lines.items()}
    fx = sorted({float(f) for f in fixed})
    if not junctions:
        return out
    lo, hi = fx[0], fx[-1]
    js = []
    for p, w, h in sorted((float(p), float(w), float(h)) for p, w, h in junctions):
        if not (lo < p < hi or any(abs(p - f) <= 1e-3 * h for f in (lo, hi))):
            raise ValueError("band_lines: junction {!r} outside the domain [{!r}, {!r}]".format(p, lo, hi))
        near = min(out, key=lambda q: abs(q - p))
        p = near if abs(near - p) <= 1e-3 * h else p
        if js and p - js[-1][0] <= 1e-3 * min(h, js[-1][2]):   # the same junction seen twice (probe rows)
            q, w0, h0 = js.pop()
            p, w, h = q, max(w, w0), min(h, h0)
        js.append((p, w, h))
    clusters, cur = [], [js[0]]
    for j in js[1:]:
        if j[0] - j[1] < max(c[0] + c[1] for c in cur):
            cur.append(j)
        else:
            clusters.append(cur)
            cur = [j]
    clusters.append(cur)
    for cl in clusters:
        h = min(c[2] for c in cl)
        first, last = cl[0][0], cl[-1][0]
        f_left = max([f for f in fx if f < first - 1e-3 * h] or [lo])
        f_right = min([f for f in fx if f > last + 1e-3 * h] or [hi])
        a = max(min(c[0] - c[1] for c in cl), f_left)
        b = min(max(c[0] + c[1] for c in cl), f_right)
        anchors = sorted({c[0] for c in cl} | {f for f in fx if first < f < last})
        nodes: Dict[float, list] = {q: [None, None] for q in anchors}     # pos -> [ns, ps]
        for q0, q1 in zip(anchors[:-1], anchors[1:]):
            n = max(1, int(math.ceil((q1 - q0) / h - 1e-9)))
            s = (q1 - q0) / n
            for k in range(1, n):
                nodes[q0 + k * (q1 - q0) / n] = [s, s]
            nodes[q0][1], nodes[q1][0] = s, s
        for anchor, sgn, end, end_fixed in ((anchors[0], -1.0, a, a == f_left),
                                            (anchors[-1], 1.0, b, b == f_right)):
            s = nodes[anchor][1 if sgn < 0 else 0] or h               # the adjacent span's spacing
            if sgn < 0:
                nodes[anchor][0] = s
            else:
                nodes[anchor][1] = s
            m = int(math.ceil(abs(end - anchor) / s - 1e-9))
            pts = [anchor + sgn * k * s for k in range(1, m + 1)]
            if end_fixed:
                pts = [q for q in pts if sgn * (end - q) > 1e-3 * s]  # stop before the fixed line ...
                if pts and abs(end - pts[-1]) < 0.25 * s:              # ... and merge a tiny remainder
                    pts.pop()
            for q in pts:
                nodes[q] = [s, s]
        span_lo = min(nodes) if a != f_left else f_left
        span_hi = max(nodes) if b != f_right else f_right
        for q in [q for q in out if span_lo < q < span_hi and q not in fx and q not in nodes]:
            del out[q]
        for q, (ns, ps) in nodes.items():
            out[q] = (ns, ps)
    return out


def sign_changes(pos: Sequence[float], val: Sequence[float]) -> list:
    """Sign changes of a sampled net doping along one line: [(a, b, left_sign, N_left, N_right)], where a is the
    last sample of the left sign and b the first of the right one (zeros in between are skipped: a junction
    with an undoped sliver), and N_left / N_right are the MEDIAN |N| over the nonzero samples of the span on
    each side (from the previous change, or the line's start, to the next one, or its end) -- the abrupt
    doping of a step junction, a representative level of a graded one."""
    nz = [(float(p), float(v)) for p, v in zip(pos, val) if v != 0.0 and math.isfinite(v)]
    cuts = [k for k in range(1, len(nz)) if (nz[k][1] > 0.0) != (nz[k - 1][1] > 0.0)]
    bounds = [0] + cuts + [len(nz)]

    def med(i0, i1):
        a = sorted(abs(v) for _p, v in nz[i0:i1])
        n = len(a)
        return 0.5 * (a[(n - 1) // 2] + a[n // 2])
    return [(nz[k - 1][0], nz[k][0], 1.0 if nz[k - 1][1] > 0.0 else -1.0, med(bounds[i], k), med(k, bounds[i + 2]))
            for i, k in enumerate(cuts)]
