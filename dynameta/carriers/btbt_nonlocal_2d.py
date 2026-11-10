"""
2-D path machinery of the nonlocal band-to-band tunnelling model (carriers/btbt_nonlocal.py, module doc;
v0.15 plan item 4.1.3, user decisions 21-25). The rate, the closed-form segment integrals, the chain rule
and the DEVSIM assembly are the dimension-free ones of btbt_nonlocal (_rates_core on a PathSet); this module
supplies what is 2-D: the triangle mesh, the nodal gradient, and the path WALKER.

MESH. A region's P1 triangles (DEVSIM's element node list) and node coordinates: barycentric gradients per
triangle, the neighbour across each edge (-1 on the region boundary), the triangles around each node, the
boundary edges.

NODAL GRADIENT (the start node's |grad E_V| and direction, and the slopes ds/dv of the band-variable
prefactor). A least-squares QUADRATIC fit to psi over the node's 1-ring (2-ring if the 1-ring cannot
determine a quadratic), centred on the node; its gradient there. It is exact for a quadratic potential
(second order), and it is a fixed LINEAR operator of psi (gradient_ops_2d: two sparse N x N matrices, built
once per mesh) -- so its Jacobian is exact. On the region boundary the patch is completed by REFLECTING it
across the boundary line(s) through the node (an even, Neumann-consistent extension). On a tensor mesh with
a laterally uniform potential the fit is the quadratic through the three x-abscissae of the patch, i.e.
EXACTLY the 1-D 3-point formula (btbt_nonlocal.nodal_gradient_1d), interior and boundary rows alike: that is
what makes a laterally uniform 2-D device reproduce the 1-D model (gate).

WALKER (numba REQUIRED: user decision 23; trace_paths_2d raises without it). From each start node a straight
ray along the tunnelling direction (the nodal gradient of psi -- dynamic -- or a frozen unit vector), walked
through the triangles: in a triangle the exit is the first barycentric coordinate that reaches 0; the point
is then re-solved exactly as the intersection of the ray with that edge (2 x 2 system: path coordinate s and
edge coordinate mu), so nothing accumulates. A ray that passes through a node, or runs along an edge (a
field aligned with a tensor mesh), continues from that node in the triangle whose angular sector contains
the direction. Each point records (n1, n2, mu, s) and ds/dtheta, dmu/dtheta (the derivative with respect to
the direction angle, from the same 2 x 2 system; 0 at a node). The walk stops at the first point with
v = (psi - psi(x0))/E_g >= 1 (the path end lies on the last segment, inside one triangle) and DISCARDS the
path when it leaves the region (region boundary, a Neumann side, an interface -- decision 24: no specular
reflection yet), when v stops rising, at max_length, or when the end triangle touches a contact node.
A pure-Python run of the same kernel (`_walk2d.py_func`) is used only as a test oracle-twin; the gates use an
independent brute-force ray/edge intersection (tests/test_btbt_nonlocal_2d_kernels.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dynameta.numba_env import have_numba, njit

__all__ = ["Mesh2D", "build_mesh2d", "gradient_ops_2d", "trace_paths_2d", "require_numba"]

STATUS = ("ok", "boundary", "reentrant", "too_long")


def require_numba(who: str) -> None:
    """User decision 23: the 2-D walker needs numba (a pure-Python walk is minutes per assembly)."""
    if not have_numba():
        raise ImportError("{}: the 2-D nonlocal BTBT walker requires numba (user decision 23; "
                          "pip install dynameta[numba]). Use the local model (tunneling.setup_btbt) "
                          "without it.".format(who))


@dataclass
class Mesh2D:
    """A region's triangle mesh (module doc). xy (N, 2) [m]; tri (T, 3) node indices; nbr (T, 3): the
    triangle across the edge opposite local vertex k (-1: region boundary); gl (T, 3, 2): barycentric
    gradients; n2t_ptr / n2t: CSR list of the triangles of each node; bnd_edges (B, 2); h_min: the smallest
    triangle altitude [m]."""
    xy: np.ndarray
    tri: np.ndarray
    nbr: np.ndarray
    gl: np.ndarray
    n2t_ptr: np.ndarray
    n2t: np.ndarray
    bnd_edges: np.ndarray
    h_min: float

    @property
    def n_nodes(self) -> int:
        return self.xy.shape[0]


def build_mesh2d(xy, tri) -> Mesh2D:
    xy = np.ascontiguousarray(np.asarray(xy, float))
    tri = np.ascontiguousarray(np.asarray(tri, np.int64))
    T = tri.shape[0]
    p = xy[tri]                                                  # (T, 3, 2)
    M = np.ones((T, 3, 3))
    M[:, 0, :] = p[:, :, 0]
    M[:, 1, :] = p[:, :, 1]
    Minv = np.linalg.inv(M)                                      # lambda = Minv @ (x, y, 1)
    gl = np.ascontiguousarray(Minv[:, :, :2])
    # edges: local edge k joins vertices (k+1, k+2)
    ea = np.stack([tri[:, 1], tri[:, 2], tri[:, 0]], axis=1).ravel()
    eb = np.stack([tri[:, 2], tri[:, 0], tri[:, 1]], axis=1).ravel()
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    key = lo * (xy.shape[0] + 1) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    nbr = -np.ones(3 * T, np.int64)
    same = ks[1:] == ks[:-1]
    for j in np.flatnonzero(same):                               # interior edges appear twice
        u, w = order[j], order[j + 1]
        nbr[u], nbr[w] = w // 3, u // 3
    nbr = nbr.reshape(T, 3)
    single = np.ones(ks.size, bool)
    single[1:][same] = False
    single[:-1][same] = False
    bnd = np.stack([lo[order][single], hi[order][single]], axis=1)
    flat = tri.ravel()
    o = np.argsort(flat, kind="stable")
    ptr = np.zeros(xy.shape[0] + 1, np.int64)
    np.add.at(ptr, flat + 1, 1)
    ptr = np.cumsum(ptr)
    n2t = (o // 3).astype(np.int64)
    a2 = np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    emax = np.max(np.stack([np.hypot(*(p[:, 1] - p[:, 0]).T), np.hypot(*(p[:, 2] - p[:, 1]).T),
                            np.hypot(*(p[:, 0] - p[:, 2]).T)], 1), axis=1)
    return Mesh2D(xy, tri, nbr, gl, ptr, n2t, bnd, float(np.min(a2 / emax)))


# =====================================================================================================
# nodal gradient: least-squares quadratic recovery (module doc)
# =====================================================================================================

def _rings(mesh: Mesh2D):
    """The 1-ring neighbour sets."""
    N = mesh.n_nodes
    nb = [set() for _ in range(N)]
    for a, b, c in mesh.tri:
        nb[a].update((b, c))
        nb[b].update((a, c))
        nb[c].update((a, b))
    return nb


_MONOMIAL_SETS = ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5))      # 1 x y x^2 xy y^2; without xy


def gradient_ops_2d(mesh: Mesh2D, neumann_nodes=None):
    """[Gx, Gy]: sparse N x N operators of the nodal gradient (module doc): g = (Gx psi, Gy psi).

    The fit: a least-squares quadratic over the node's 1-ring, centred on the node; when the 1-ring cannot
    carry the xy term (a valence-4 node of a union-jack mesh: xy vanishes on every patch point) it is
    dropped -- its gradient at the node is zero, so nothing is lost --; when the 1-ring cannot carry
    1, x, y, x^2, y^2 either, the 2-ring with the full quadratic. Exact for a quadratic psi.

    neumann_nodes (bool mask, default: every boundary node): boundary nodes whose boundary edges are
    homogeneous-NEUMANN sides for psi (DEVSIM's natural boundary of Poisson: an outer side that is neither a
    contact nor an interface). Their patch is completed by its mirror image across the boundary line(s): the
    even extension IS the solution's continuation there (d psi/dn = 0), and it keeps the 1-ring stencil
    (a laterally uniform potential on a tensor mesh then gives the 1-D 3-point formula exactly). Any other
    boundary node (a contact, an interface, a side the field crosses) gets a ONE-SIDED fit (no mirror)."""
    import scipy.sparse as sp
    N = mesh.n_nodes
    xy = mesh.xy
    ring = _rings(mesh)
    bnd_at = [[] for _ in range(N)]
    for a_, b_ in mesh.bnd_edges:
        bnd_at[a_].append(b_)
        bnd_at[b_].append(a_)
    neu = np.ones(N, bool) if neumann_nodes is None else np.asarray(neumann_nodes, bool)
    rows, cols, vx, vy = [], [], [], []
    for n in range(N):
        mirror = bool(bnd_at[n]) and neu[n]
        lines = []
        if mirror:
            for m in bnd_at[n]:
                u = xy[m] - xy[n]
                u = u / np.hypot(*u)
                if not any(abs(u[0] * w[1] - u[1] * w[0]) < 1e-9 for w in lines):
                    lines.append(u)
        found = None
        for depth in (1, 2):
            patch = set(ring[n]) | {n}
            if depth == 2:
                for m in list(ring[n]):
                    patch |= ring[m]
            idx = np.array(sorted(patch), np.int64)
            d = xy[idx] - xy[n]
            pts, own = [d], [idx]
            refl = [lambda q, u=u: 2.0 * (q @ u)[:, None] * u[None, :] - q for u in lines]
            if len(refl) == 2:
                refl.append(lambda q, f=refl[0], g=refl[1]: g(f(q)))
            for f in refl:
                pts.append(f(d))
                own.append(idx)
            Q = np.concatenate(pts)
            J = np.concatenate(own)
            hs = float(np.max(np.hypot(Q[:, 0], Q[:, 1])))
            q = Q / hs
            full = np.stack([np.ones(len(q)), q[:, 0], q[:, 1], q[:, 0] ** 2, q[:, 0] * q[:, 1], q[:, 1] ** 2], 1)
            for cols_ in (_MONOMIAL_SETS if depth == 1 else _MONOMIAL_SETS[:1]):
                A = full[:, cols_]
                if np.linalg.matrix_rank(A) == len(cols_):
                    found = (A, J, hs)
                    break
            if found is not None:
                break
        if found is None:                                        # pragma: no cover - degenerate patch
            found = (full[:, :3], J, hs)
        A, J, hs = found
        Pinv = np.linalg.pinv(A)
        rows.append(np.full(J.size, n))
        cols.append(J)
        vx.append(Pinv[1] / hs)
        vy.append(Pinv[2] / hs)
    r, c = np.concatenate(rows), np.concatenate(cols)
    Gx = sp.csr_matrix((np.concatenate(vx), (r, c)), shape=(N, N))
    Gy = sp.csr_matrix((np.concatenate(vy), (r, c)), shape=(N, N))
    Gx.sum_duplicates()
    Gy.sum_duplicates()
    return [Gx, Gy]


# =====================================================================================================
# the walker (numba)
# =====================================================================================================

@njit(cache=False)
def _sector(i, d0, d1, xy, tri, ptr, n2t):
    """The triangle around node i whose angular sector contains direction (d0, d1); -1 if none (the ray
    leaves the region at i)."""
    best = -1e300
    bt = -1
    for m in range(ptr[i], ptr[i + 1]):
        t = n2t[m]
        a = 0
        for k in range(3):
            if tri[t, k] == i:
                a = k
        b = tri[t, (a + 1) % 3]
        c = tri[t, (a + 2) % 3]
        ebx = xy[b, 0] - xy[i, 0]
        eby = xy[b, 1] - xy[i, 1]
        ecx = xy[c, 0] - xy[i, 0]
        ecy = xy[c, 1] - xy[i, 1]
        cr = ebx * ecy - eby * ecx
        sg = 1.0 if cr > 0.0 else -1.0
        c1 = sg * (ebx * d1 - eby * d0) / np.sqrt(ebx * ebx + eby * eby)
        c2 = sg * (d0 * ecy - d1 * ecx) / np.sqrt(ecx * ecx + ecy * ecy)
        mm = c1 if c1 < c2 else c2
        if mm > best:
            best = mm
            bt = t
    if best < -1e-10:
        return -1
    return bt


@njit(cache=False)
def _walk2d(xy, tri, nbr, gl, ptr, n2t, psi, starts, dirs, Eg_V, Lmax, cap, store, offsets,
            cnt, status, o_n1, o_n2, o_mu, o_s, o_ds, o_dmu):
    lam = np.zeros(3)
    rate = np.zeros(3)
    for q in range(starts.shape[0]):
        i = starts[q]
        d0 = dirs[q, 0]
        d1 = dirs[q, 1]
        x0 = xy[i, 0]
        y0 = xy[i, 1]
        p0 = psi[i]
        base = offsets[q]
        if store:
            o_n1[base] = i
            o_n2[base] = i
            o_mu[base] = 0.0
            o_s[base] = 0.0
            o_ds[base] = 0.0
            o_dmu[base] = 0.0
        npt = 1
        st = 0
        t = _sector(i, d0, d1, xy, tri, ptr, n2t)
        if t < 0:
            cnt[q] = npt
            status[q] = 1
            continue
        for k in range(3):
            lam[k] = 1.0 if tri[t, k] == i else 0.0
        v_prev = 0.0
        while True:
            best = 1e300
            kx = -1
            for k in range(3):
                gx = gl[t, k, 0]
                gy = gl[t, k, 1]
                rate[k] = gx * d0 + gy * d1
                if rate[k] < -1e-12 * np.sqrt(gx * gx + gy * gy):
                    sk = lam[k] / (-rate[k])
                    if sk < best:
                        best = sk
                        kx = k
            if kx < 0:
                st = 1
                break
            ka = (kx + 1) % 3
            kb = (kx + 2) % 3
            na = tri[t, ka]
            nb_ = tri[t, kb]
            la = lam[ka] + best * rate[ka]
            lb = lam[kb] + best * rate[kb]
            vertex = -1
            if la <= 1e-10:
                vertex = nb_
            elif lb <= 1e-10:
                vertex = na
            if vertex >= 0:
                n1 = vertex
                n2 = vertex
                mu = 0.0
                s_new = (xy[vertex, 0] - x0) * d0 + (xy[vertex, 1] - y0) * d1
                dsd = 0.0
                dmd = 0.0
            else:
                ex = xy[nb_, 0] - xy[na, 0]
                ey = xy[nb_, 1] - xy[na, 1]
                rx = xy[na, 0] - x0
                ry = xy[na, 1] - y0
                det = ex * d1 - d0 * ey
                s_new = (ex * ry - rx * ey) / det
                mu = (d0 * ry - d1 * rx) / det
                hx = s_new * d1                                  # d/dtheta: [d, -e][s', mu'] = -s d_perp
                hy = -s_new * d0
                dsd = (ex * hy - hx * ey) / det
                dmd = (d0 * hy - d1 * hx) / det
                if mu < 0.0:
                    mu = 0.0
                if mu > 1.0:
                    mu = 1.0
                n1 = na
                n2 = nb_
            v = ((1.0 - mu) * psi[n1] + mu * psi[n2] - p0) / Eg_V
            if v <= v_prev:
                st = 2
                break
            if store:
                j = base + npt
                o_n1[j] = n1
                o_n2[j] = n2
                o_mu[j] = mu
                o_s[j] = s_new
                o_ds[j] = dsd
                o_dmu[j] = dmd
            npt += 1
            if v >= 1.0:
                st = 0
                break
            if s_new >= Lmax or npt >= cap:
                st = 3
                break
            if vertex >= 0:
                t = _sector(vertex, d0, d1, xy, tri, ptr, n2t)
                if t < 0:
                    st = 1
                    break
                for k in range(3):
                    lam[k] = 1.0 if tri[t, k] == vertex else 0.0
            else:
                tn = nbr[t, kx]
                if tn < 0:
                    st = 1
                    break
                for k in range(3):
                    nd = tri[tn, k]
                    lam[k] = (1.0 - mu) if nd == na else (mu if nd == nb_ else 0.0)
                t = tn
            v_prev = v
        cnt[q] = npt
        status[q] = st


def trace_paths_2d(mesh: Mesh2D, psi, gvec, Eg_V: float, max_length_m: float, start_ok, end_ok,
                   v_star: Optional[float] = None, direction=None):
    """The 2-D walker (module doc) -> btbt_nonlocal.PathSet. gvec (N, 2): the nodal gradient of psi;
    direction None: each path along its start node's gradient (dynamic), else a frozen unit 2-vector."""
    from dynameta.carriers.btbt_nonlocal import PathSet
    require_numba("trace_paths_2d")
    psi = np.ascontiguousarray(np.asarray(psi, float))
    gvec = np.asarray(gvec, float)
    stats = {"starts": 0, "paths": 0, "boundary": 0, "reentrant": 0, "contact": 0, "too_long": 0}
    gn = np.hypot(gvec[:, 0], gvec[:, 1])
    cand = np.flatnonzero(np.asarray(start_ok, bool) & (gn > 0.0) & (psi.max() - psi >= Eg_V))
    if direction is None:
        dirs = gvec[cand] / gn[cand, None]
    else:
        dd = np.asarray(direction, float)
        dirs = np.tile(dd / np.hypot(*dd), (cand.size, 1))
    dirs = np.ascontiguousarray(dirs)
    stats["starts"] = int(cand.size)
    cap = int(10.0 * max_length_m / mesh.h_min) + 1000
    S = cand.size
    cnt = np.zeros(S, np.int64)
    status = np.zeros(S, np.int64)
    zi, zf = np.zeros(1, np.int64), np.zeros(1)
    args = (mesh.xy, mesh.tri, mesh.nbr, mesh.gl, mesh.n2t_ptr, mesh.n2t, psi, cand.astype(np.int64), dirs,
            float(Eg_V), float(max_length_m), cap)
    _walk2d(*args, False, np.zeros(S, np.int64), cnt, status, zi, zi, zf, zf, zf, zf)
    off = np.concatenate(([0], np.cumsum(cnt)[:-1])).astype(np.int64)
    tot = int(cnt.sum())
    o_n1, o_n2 = np.zeros(tot, np.int64), np.zeros(tot, np.int64)
    o_mu, o_s, o_ds, o_dmu = np.zeros(tot), np.zeros(tot), np.zeros(tot), np.zeros(tot)
    _walk2d(*args, True, off, cnt, status, o_n1, o_n2, o_mu, o_s, o_ds, o_dmu)
    # normalise node points (mu 0 or 1) to n1 == n2, mu = 0
    one = o_mu >= 1.0
    o_n1[one] = o_n2[one]
    o_mu[one] = 0.0
    zero = o_mu <= 0.0
    o_n2[zero] = o_n1[zero]
    o_dmu[zero | one] = 0.0
    for k, nm in enumerate(STATUS[1:], start=1):
        stats[nm] += int(np.sum(status == k))
    keep = np.zeros(S, bool)
    end_ok = np.asarray(end_ok, bool)
    vstar_seg = np.full(S, -1, np.int64)
    for q in np.flatnonzero(status == 0):
        s0, n = off[q], cnt[q]
        a, b = s0 + n - 2, s0 + n - 1
        if not (end_ok[o_n1[a]] and end_ok[o_n2[a]] and end_ok[o_n1[b]] and end_ok[o_n2[b]]):
            stats["contact"] += 1
            continue
        sl = slice(s0, s0 + n)
        w1, mu = 1.0 - o_mu[sl], o_mu[sl]
        gp = w1[:, None] * gvec[o_n1[sl]] + mu[:, None] * gvec[o_n2[sl]]
        if np.any(gp @ dirs[q] <= 0.0):                          # the prefactor's ds/dv must be > 0
            stats["reentrant"] += 1
            continue
        v = (w1 * psi[o_n1[sl]] + mu * psi[o_n2[sl]] - psi[cand[q]]) / Eg_V
        va, vb = v[-2], v[-1]
        tt = (1.0 - va) / (vb - va)
        if o_s[a] + tt * (o_s[b] - o_s[a]) >= max_length_m:
            stats["too_long"] += 1
            continue
        if v_star is not None:
            vstar_seg[q] = int(np.flatnonzero(v[1:] >= v_star)[0])
        keep[q] = True
    kq = np.flatnonzero(keep)
    stats["paths"] = int(kq.size)
    P = kq.size
    if P == 0:
        e, f = np.zeros(0, np.int64), np.zeros(0)
        return PathSet(e, np.zeros((0, 2)), e, e, e, e, e, f, e.astype(bool), e, e, f, e.astype(bool),
                       e.astype(bool), e, stats, f, f, f)
    npt = cnt[kq]
    sel = np.concatenate([np.arange(off[q], off[q] + cnt[q]) for q in kq])
    poff = np.concatenate(([0], np.cumsum(npt)[:-1]))
    pt_path = np.repeat(np.arange(P), npt)
    first = np.zeros(sel.size, bool)
    first[poff] = True
    nseg = npt - 1
    seg_path = np.repeat(np.arange(P), nseg)
    j = np.arange(int(nseg.sum())) - np.repeat(np.concatenate(([0], np.cumsum(nseg)[:-1])), nseg)
    seg_lo = np.repeat(poff, nseg) + j
    s_pts = o_s[sel]
    seg_L = s_pts[seg_lo + 1] - s_pts[seg_lo]
    seg_first = j == 0
    seg_end = j == np.repeat(nseg - 1, nseg)
    phase = np.zeros(seg_path.size, np.int64)
    if v_star is not None:
        spl = np.repeat(vstar_seg[kq], nseg)
        phase = np.where(j < spl, 0, np.where(j == spl, 1, 2))
    end_b = poff + npt - 1
    return PathSet(cand[kq].astype(np.int64), dirs[kq], end_b - 1, end_b, pt_path, o_n1[sel], o_n2[sel],
                   o_mu[sel], first, seg_path, seg_lo, seg_L, seg_first, seg_end, phase, stats, s_pts,
                   o_ds[sel], o_dmu[sel], dynamic=direction is None)
