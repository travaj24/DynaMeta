"""2-D nonlocal band-to-band tunnelling, the KERNELS (carriers/btbt_nonlocal_2d.py + the dimension-free
btbt_nonlocal._rates_core) -- no DEVSIM; numba REQUIRED (user decision 23).

References are independent of the code under test: closed forms re-derived here, a brute-force ray / edge
intersection written here (no walking), the 1-D model (tests/test_btbt_nonlocal_kernels.py gates it against
Sentaurus' equations) and that file's scipy continuum oracle.

OLD CODE (22b1ba7): carriers/btbt_nonlocal_2d does not exist and btbt_nonlocal has no path_rates_2d -- every
test fails at import (MEASURED by running this file against an export of 22b1ba7).

  W1   WALKER: the numba walker's points (path coordinate s and band value v) == a brute-force intersection
       of the ray with EVERY mesh edge (sorted; ray-inside checks by point location), on random Delaunay
       meshes and random directions, to 1e-12 (relative to the domain size / to 1); the exit status agrees.
  W2   the numba kernel == its own pure-Python run (`.py_func`) bit for bit (plan N13).
  U1   UNIFORM OBLIQUE FIELD (plan N11): every interior path at 0 / 30 / 45 deg on a union-jack mesh and on a
       Delaunay mesh gives A F^P exp(-B/F) within the N1 round-off budget (<= 1e-12).
  U2   LATERALLY UNIFORM: a tensor-mesh 2-D profile psi(x) gives the 1-D model's rate at every start node,
       boundary rows included (Neumann mirror), to the round-off budget: the gradient recovery reduces to the
       1-D 3-point formula there.
  R1   ROTATED JUNCTION (30, 45 deg) on a mesh NOT aligned with the field: per-path rates at nested start
       nodes converge to the 1-D continuum oracle (Eqs. 480-484 / 490-493 by scipy on the junction
       coordinate) at order >= 1.7, and the 0.1 nm error <= 2 C_inf h^2 (C_inf from the finest mesh).
       Refutation: a one-element (first-order) nodal gradient drops the order to ~1.
  J1   JACOBIAN of the assembled residual (hole rows at the starts, electron rows at the end triangles) ==
       4-point central differences, DYNAMIC direction (paths rotate with the field: the exact rotation terms)
       and frozen direction, both processes, 30 and 45 deg. Refutation: without the rotation terms the
       dynamic-direction Jacobian misses by O(1) of its largest entry.
  N0   refusals: no numba -> ImportError naming decision 23.
"""
import math
import sys

import numpy as np
import pytest

pytest.importorskip("numba")

from dynameta.carriers import btbt_nonlocal as NL  # noqa: E402
from dynameta.carriers import btbt_nonlocal_2d as B2  # noqa: E402
from dynameta.carriers.tunneling import BTBTParams  # noqa: E402

EPS = np.finfo(float).eps
EG = 1.12
DIRECT = BTBTParams(A_SI=3.5e21, B_V_per_m=2.2e9, P=2.0)
PHONON = BTBTParams.from_cgs(4.0e14, 1.9e7, 2.5)
RATIO = 1.7


def _spec(params, Lmax=1.0):
    return NL.NonlocalBTBT(params, Eg_tun_eV=EG, mass_ratio_V_over_C=RATIO, max_length_m=Lmax)


def tensor_mesh(xs, ys, union_jack=True):
    """Rectangles split into 2 triangles; union_jack alternates the diagonal (valence-4 and -8 nodes)."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    ny = len(ys)
    tri = []
    for i in range(len(xs) - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            if (i + j) % 2 == 0 or not union_jack:
                tri += [(a, b, c), (a, c, d)]
            else:
                tri += [(a, b, d), (b, c, d)]
    return xy, np.array(tri)


def delaunay_mesh(n, L, seed):
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    g = np.linspace(0, L, 6)
    border = np.array([(u, 0) for u in g] + [(u, L) for u in g] + [(0, u) for u in g[1:-1]]
                      + [(L, u) for u in g[1:-1]])
    xy = np.concatenate([border, rng.uniform(0.02 * L, 0.98 * L, (n, 2))])
    return xy, Delaunay(xy).simplices.astype(np.int64)


# =====================================================================================================
# W1 / W2 walker
# =====================================================================================================

def _oracle_ray(xy, tri, psi, i, d, Eg):
    """Brute force: every edge the ray x_i + s d crosses (2x2 solve per edge), sorted by s; the ray must be
    INSIDE the mesh between consecutive crossings (midpoint inside some triangle). Returns (s list, v list,
    status) up to and including the first point with v >= 1."""
    E = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    a, b = xy[E[:, 0]], xy[E[:, 1]]
    e = b - a
    r = a - xy[i]
    det = e[:, 0] * d[1] - d[0] * e[:, 1]
    ok = np.abs(det) > 1e-30
    s = np.where(ok, (e[:, 0] * r[:, 1] - r[:, 0] * e[:, 1]) / np.where(ok, det, 1), np.nan)
    mu = np.where(ok, (d[0] * r[:, 1] - d[1] * r[:, 0]) / np.where(ok, det, 1), np.nan)
    hit = ok & (s > 1e-15) & (mu >= -1e-12) & (mu <= 1 + 1e-12)
    order = np.argsort(s[hit])
    S = s[hit][order]
    MU = np.clip(mu[hit][order], 0, 1)
    EE = E[hit][order]
    vals = (1 - MU) * psi[EE[:, 0]] + MU * psi[EE[:, 1]]
    keep = np.concatenate(([True], np.diff(S) > 1e-14 * (S[-1] if S.size else 1)))    # vertex: many edges
    S, vals = S[keep], vals[keep]
    P = xy[tri]
    out_s, out_v = [0.0], [0.0]
    prev = 0.0
    for sk, pv in zip(S, vals):
        mid = xy[i] + 0.5 * (prev + sk) * d                     # is the ray inside the mesh before sk?
        lam = _bary(P, mid)
        if not np.any(np.all(lam >= -1e-9, axis=1)):
            return out_s, out_v, "boundary"
        v = (pv - psi[i]) / Eg
        out_s.append(sk)
        out_v.append(v)
        if v >= 1.0:
            return out_s, out_v, "ok"
        prev = sk
    return out_s, out_v, "boundary"


def _bary(P, q):
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    l1 = ((q[0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (q[1] - a[:, 1])) / det
    l2 = ((b[:, 0] - a[:, 0]) * (q[1] - a[:, 1]) - (q[0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
    return np.stack([1 - l1 - l2, l1, l2], axis=1)


def _walk_raw(mesh, psi, starts, dirs, Eg, Lmax, pyfunc=False):
    S = len(starts)
    cnt, status = np.zeros(S, np.int64), np.zeros(S, np.int64)
    zi, zf = np.zeros(1, np.int64), np.zeros(1)
    f = B2._walk2d.py_func if pyfunc else B2._walk2d
    args = (mesh.xy, mesh.tri, mesh.nbr, mesh.gl, mesh.n2t_ptr, mesh.n2t, psi, np.asarray(starts, np.int64),
            np.ascontiguousarray(dirs), Eg, Lmax, 100000)
    f(*args, False, np.zeros(S, np.int64), cnt, status, zi, zi, zf, zf, zf, zf)
    off = np.concatenate(([0], np.cumsum(cnt)[:-1])).astype(np.int64)
    n = int(cnt.sum())
    o = [np.zeros(n, np.int64), np.zeros(n, np.int64)] + [np.zeros(n) for _ in range(4)]
    f(*args, True, off, cnt, status, *o)
    return cnt, status, off, o


def test_w1_walker_matches_brute_force_intersections():
    worst_s = worst_v = 0.0
    n_ok = n_bnd = 0
    for seed in range(3):
        L = 20e-9
        xy, tri = delaunay_mesh(300, L, seed)
        mesh = B2.build_mesh2d(xy, tri)
        rng = np.random.default_rng(100 + seed)
        psi = 0.3 * np.sin(xy[:, 0] / 7e-9) + 1.4 * xy[:, 0] / L + 0.2 * xy[:, 1] / L
        starts = rng.choice(np.arange(20, len(xy)), 40, replace=False)
        ang = rng.uniform(-0.6, 0.6, 40)
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        cnt, status, off, (n1, n2, mu, s, _, _) = _walk_raw(mesh, psi, starts, dirs, 0.5, 1.0)
        for q, i in enumerate(starts):
            os_, ov, ost = _oracle_ray(xy, tri, psi, i, dirs[q], 0.5)
            st = B2.STATUS[status[q]]
            if st == "reentrant" or ost == "boundary" and st != "boundary":
                # the walker discards a non-rising band; the oracle reports where the ray goes regardless
                assert st in ("reentrant", "boundary")
                continue
            assert st == ost, (seed, q, st, ost)
            k = slice(off[q], off[q] + cnt[q])
            ws = s[k]
            wv = ((1 - mu[k]) * psi[n1[k]] + mu[k] * psi[n2[k]] - psi[i]) / 0.5
            m = min(len(ws), len(os_))
            if st == "ok":
                assert len(ws) == len(os_)
                n_ok += 1
            else:
                n_bnd += 1
            worst_s = max(worst_s, float(np.max(np.abs(ws[:m] - np.array(os_[:m])))) / L)
            worst_v = max(worst_v, float(np.max(np.abs(wv[:m] - np.array(ov[:m])))))
    print("W1: ok paths", n_ok, "boundary", n_bnd, "worst s/L", worst_s, "worst v", worst_v)
    assert n_ok > 30 and n_bnd > 5
    assert worst_s < 1e-12 and worst_v < 1e-12


def test_w2_numba_equals_python_bitwise():
    xy, tri = delaunay_mesh(200, 20e-9, 7)
    mesh = B2.build_mesh2d(xy, tri)
    psi = 1.5 * xy[:, 0] / 20e-9 + 0.1 * np.cos(xy[:, 1] / 5e-9)
    rng = np.random.default_rng(1)
    starts = rng.choice(len(xy), 30, replace=False)
    ang = rng.uniform(-0.5, 0.5, 30)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    a = _walk_raw(mesh, psi, starts, dirs, 0.5, 1.0)
    b = _walk_raw(mesh, psi, starts, dirs, 0.5, 1.0, pyfunc=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    for u, w in zip(a[3], b[3]):
        assert np.array_equal(u, w)


# =====================================================================================================
# U1 / U2
# =====================================================================================================

@pytest.mark.parametrize("params", [DIRECT, PHONON], ids=["direct", "phonon"])
@pytest.mark.parametrize("mesh_kind", ["union_jack", "delaunay"])
def test_u1_uniform_oblique_field_identity(params, mesh_kind):
    """Budget per path: the N1 budget of the 1-D file, 8 eps nseg (1 + B/F) x 3 (nseg: the path's segments;
    the factor 3 covers the 2x2 edge solve per point), PLUS the round-off the gradient recovery amplifies:
    the start gradient and the prefactor slopes are sums G_ij psi_j, exact to eps sum_j |G_ij| |psi_j| /
    |g| (computed here per node; tiny on a tensor mesh, up to 4e-11 at the worst node of the Delaunay mesh,
    whose patches hold near-coincident nodes), entering R linearly -- 4x the largest over the path's nodes.
    On the union-jack mesh additionally < 1e-12 (the plan's figure). MEASURED <= 5e-14 (union-jack),
    1.2e-12 (Delaunay)."""
    F = 1e8
    ph = NL.physical_from_fitted(_spec(params))
    if mesh_kind == "union_jack":
        xs = np.linspace(0, 40e-9, 81)
        xy, tri = tensor_mesh(xs, xs)
    else:
        xy, tri = delaunay_mesh(2500, 40e-9, 3)
    mesh = B2.build_mesh2d(xy, tri)
    N = len(xy)
    gops = B2.gradient_ops_2d(mesh, neumann_nodes=np.zeros(N, bool))     # the field crosses every side
    ok = np.ones(N, bool)
    ref = params.A_SI * F ** params.P * math.exp(-params.B_V_per_m / F)
    absG = abs(gops[0]) + abs(gops[1])
    for th in (0.0, 30.0, 45.0):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        psi = F * (xy[:, 0] * c + xy[:, 1] * s) - 2.0
        res = NL.path_rates_2d(mesh, psi, np.full(N, -300.0), np.full(N, -300.0), ph, 0.0, 1.0, ok, ok,
                               gops=gops, derivatives=False)
        ps = res["pathset"]
        amp = EPS * (absG @ np.abs(psi)) / F                    # relative round-off of each nodal gradient
        amp_p = np.zeros(ps.start.size)
        np.maximum.at(amp_p, ps.pt_path, np.maximum(amp[ps.pt_n1], amp[ps.pt_n2]))
        nseg = np.bincount(ps.seg_path)
        dev = np.abs(-res["R"] / ref - 1.0)
        budget = 8 * EPS * nseg * (1 + params.B_V_per_m / F) * 3 + 4 * amp_p
        print(mesh_kind, th, "paths", ps.start.size, "max dev", dev.max(), "max dev/budget",
              float(np.max(dev / budget)))
        assert ps.start.size > 1000
        assert np.all(dev <= budget)
        if mesh_kind == "union_jack":
            assert dev.max() < 1e-12


@pytest.mark.parametrize("params", [DIRECT, PHONON], ids=["direct", "phonon"])
def test_u2_laterally_uniform_equals_1d(params):
    """psi(x), Ln(x), Lp(x) on a union-jack tensor mesh with non-uniform x spacing; the x-ends are contacts
    (no starts, one-sided gradient), the y-sides Neumann (mirror). Every 2-D path == the 1-D path of the same
    x node. Budget: both sides are the same closed forms on the same values; the 2-D values pass through
    the gradient recovery (pinv of a 6 x 6 design: condition ~1e2 x eps) and the edge solve: 1e-13
    relative, x B/F for the exponent: 4e-12. MEASURED <= 6e-14."""
    ph = NL.physical_from_fitted(_spec(params))
    rng = np.random.default_rng(2)
    xs = np.cumsum(np.r_[0, rng.uniform(0.15e-9, 0.3e-9, 199)])
    xs -= xs[len(xs) // 2]
    ys = np.array([0.0, 1e-9, 2.5e-9, 3e-9])
    xy, tri = tensor_mesh(xs, ys)
    mesh = B2.build_mesh2d(xy, tri)
    N = len(xy)

    def f(x):
        return 1.0 * (1 + np.tanh(x / 5e-9))

    def lnf(x):
        return -2 + 0.5 * np.sin(x / 7e-9)

    def lpf(x):
        return -1 + 0.4 * np.cos(x / 9e-9)
    ends = np.isin(xy[:, 0], [xs[0], xs[-1]])
    gops = B2.gradient_ops_2d(mesh, neumann_nodes=~ends)
    r2 = NL.path_rates_2d(mesh, f(xy[:, 0]), lnf(xy[:, 0]), lpf(xy[:, 0]), ph, 0.3, 1.0, ~ends,
                          np.ones(N, bool), gops=gops, derivatives=False)
    so1 = np.ones(len(xs), bool)
    so1[0] = so1[-1] = False
    r1 = NL.path_rates_1d(xs, f(xs), lnf(xs), lpf(xs), ph, 0.3, 1.0, so1, np.ones(len(xs), bool),
                          derivatives=False)
    R1 = dict(zip(r1["paths"].start.tolist(), r1["R"]))
    ix = r2["pathset"].start // len(ys)
    assert sorted(set(ix.tolist())) == sorted(R1)                # the same start nodes have paths
    assert r2["pathset"].start.size == len(ys) * len(R1)
    dev = max(abs(r2["R"][k] / R1[ix[k]] - 1) for k in range(len(ix)))
    print("U2 dev", dev)
    assert dev < 1e-13 * (1 + params.B_V_per_m / 1e8) * 4


# =====================================================================================================
# R1 rotated junction vs the continuum oracle
# =====================================================================================================
sys.path.insert(0, __import__("os").path.dirname(__file__))
import test_btbt_nonlocal_kernels as K1  # noqa: E402

HS = [0.4e-9, 0.2e-9, 0.1e-9, 0.05e-9]
XI0 = np.array([-5.6e-9, -4.0e-9, -2.4e-9, -0.8e-9])          # start positions on the y = 0 line (nested nodes)


def _rotated_errors(process, theta_deg, hs, linear_recovery=False):
    """Max relative per-path error vs the 1-D continuum oracle (K1._oracle_rate on the junction coordinate
    xi = x cos + y sin) over the nested starts (x = XI0/cos, y = 0)."""
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    params = K1.DIRECT if process == "direct" else K1.PHONON
    ph = NL.physical_from_fitted(K1._spec(params, max_length_m=1.0))
    consts = K1._consts(process)
    out = []
    for h in hs:
        xs = np.round(np.arange(-12e-9, 12e-9 + h / 2, h) / h) * h
        ys = np.round(np.arange(-3.2e-9, 8.0e-9 + h / 2, h) / h) * h
        xy, tri = tensor_mesh(xs, ys)
        mesh = B2.build_mesh2d(xy, tri)
        N = len(xy)
        xi = xy[:, 0] * c + xy[:, 1] * s
        psi = np.array([K1._psi(v) for v in xi])
        Ln = np.array([K1._Ln(v) for v in xi])
        Lp = np.array([K1._Lp(v) for v in xi])
        gops = B2.gradient_ops_2d(mesh, neumann_nodes=np.zeros(N, bool))
        if linear_recovery:                                     # refutation: a first-order recovery
            gops = _one_element_gradient_ops(mesh)
        x0 = np.round(XI0 / c / h) * h                          # nested nodes on y = 0
        starts = [int(np.argmin(np.hypot(xy[:, 0] - u, xy[:, 1]))) for u in x0]
        so = np.zeros(N, bool)
        so[starts] = True
        res = NL.path_rates_2d(mesh, psi, Ln, Lp, ph, K1.DELTA, 1.0, so, np.ones(N, bool), gops=gops,
                               derivatives=False)
        assert np.array_equal(np.sort(res["pathset"].start), np.sort(starts))
        ref = {i: K1._oracle_rate(xi[i], process, consts) for i in starts}
        out.append(float(max(abs(res["R"][k] / ref[i] - 1) for k, i in enumerate(res["pathset"].start))))
    return out


def _one_element_gradient_ops(mesh):
    """The gradient of ONE adjacent triangle (the first listed for the node): a one-sided, FIRST-ORDER
    nodal gradient. (The area-weighted mean of the adjacent element gradients is superconvergent on this
    union-jack mesh -- MEASURED order 1.91 -- so it cannot serve as the refutation.)"""
    import scipy.sparse as sp
    N = mesh.n_nodes
    t0 = mesh.n2t[mesh.n2t_ptr[:-1]]
    ops = []
    for c in range(2):
        rows = np.repeat(np.arange(N), 3)
        cols = mesh.tri[t0].ravel()
        vals = mesh.gl[t0, :, c].ravel()
        ops.append(sp.csr_matrix((vals, (rows, cols)), shape=(N, N)))
    return ops


@pytest.mark.slow
@pytest.mark.parametrize("process", ["direct", "phonon"])
@pytest.mark.parametrize("theta", [30.0, 45.0])
def test_r1_rotated_junction_converges_to_the_continuum(process, theta):
    """Budget: O(h^2) -- P1 band per triangle, second-order (quadratic-exact) gradient recovery for the start
    gradient and the prefactor slopes, linear end root and E_Fn interpolation; the walker adds no error (W1).
    Gate: least-squares order over 0.4 .. 0.05 nm >= 1.7 and err(0.4 nm) <= 2 C_inf h^2 x 16 (C_inf from the
    0.05 nm mesh). MEASURED: see the printed orders (~2)."""
    err = _rotated_errors(process, theta, HS)
    order = float(np.polyfit(np.log([h * 1e9 for h in HS]), np.log(err), 1)[0])
    C_inf = err[-1] / (HS[-1] * 1e9) ** 2
    print(process, theta, "errors", err, "order", order)
    assert order >= 1.7
    assert err[1] <= 2.0 * C_inf * (HS[1] * 1e9) ** 2


@pytest.mark.slow
def test_r1_refute_linear_gradient_recovery():
    err = _rotated_errors("direct", 30.0, HS[:3], linear_recovery=True)
    order = float(np.polyfit(np.log([h * 1e9 for h in HS[:3]]), np.log(err), 1)[0])
    print("one-element gradient errors", err, "order", order)          # MEASURED 1.29 (30 deg, direct)
    assert order < 1.5


# =====================================================================================================
# J1 Jacobian
# =====================================================================================================

def _graded_case(theta_deg, frozen, params):
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    ph = NL.physical_from_fitted(_spec(params, Lmax=12e-9))
    xs = np.linspace(-15e-9, 15e-9, 41)
    xy, tri = tensor_mesh(xs, xs)
    mesh = B2.build_mesh2d(xy, tri)
    N = len(xy)
    xi = xy[:, 0] * c + xy[:, 1] * s
    psi0 = 1.2 * (1 + np.tanh(xi / 4e-9)) + 0.05 * np.sin(xy[:, 1] / 3e-9)
    Ln = -2 + 0.5 * np.sin(xi / 7e-9)
    Lp = -1 + 0.4 * np.cos(xi / 9e-9)
    gops = B2.gradient_ops_2d(mesh, neumann_nodes=np.zeros(N, bool))
    ok = np.ones(N, bool)
    dirn = (c, s) if frozen else None

    def resid(psi, deriv=False):
        r = NL.path_rates_2d(mesh, psi, Ln, Lp, ph, 0.3, 12e-9, ok, ok, gops=gops, direction=dirn,
                             derivatives=deriv)
        F = np.zeros(2 * N)
        np.add.at(F, r["pathset"].start, r["R"])
        for k in range(4):
            np.add.at(F, N + r["e_nodes"][:, k], -r["R"] * r["e_w"][:, k])
        return F, r
    return psi0, resid, N


def _assembled_jacobian(r, N, rotation=True):
    """The Jacobian of the residual of _graded_case.resid, assembled the way _assemble_entries does."""
    ps = r["pathset"]
    J = np.zeros((2 * N, N))
    jr, jt = r["jac_R"].toarray(), r["jac_t"].toarray()
    R = r["R"]
    if not rotation and "jac_theta" in r:                       # refutation: the frozen-geometry part only
        jr = jr - np.diag(r["dR_dtheta"]) @ r["jac_theta"].toarray()
    for p_ in range(ps.start.size):
        J[ps.start[p_]] += jr[p_]
        for k in range(4):
            J[N + r["e_nodes"][p_, k]] -= r["e_w"][p_, k] * jr[p_] + R[p_] * r["e_dw"][p_, k] * jt[p_]
    if rotation and "jac_theta" in r:
        jth = r["jac_theta"].toarray()
        for p_ in range(ps.start.size):
            for k in range(4):
                J[N + r["e_nodes"][p_, k]] -= R[p_] * r["e_dwdth"][p_, k] * jth[p_]
    return J


def _fd_worst(psi0, resid, J, N, ncols=40):
    rng = np.random.default_rng(0)
    scale = np.max(np.abs(J))
    _, r0 = resid(psi0)
    worst, n_used = 0.0, 0
    for j in rng.choice(N, ncols, replace=False):
        hh = 2e-7
        F = []
        for sg in (1, -1, 2, -2):
            p = psi0.copy()
            p[j] += sg * hh
            Fx, rx = resid(p)
            if not np.array_equal(rx["pathset"].start, r0["pathset"].start):
                break
            F.append(Fx)
        if len(F) < 4:
            continue
        n_used += 1
        fd = (8 * (F[0] - F[1]) - (F[2] - F[3])) / (12 * hh)
        worst = max(worst, float(np.max(np.abs(fd - J[:, j]))) / scale)
    return worst, n_used


@pytest.mark.parametrize("params", [DIRECT, PHONON], ids=["direct", "phonon"])
@pytest.mark.parametrize("frozen", [False, True], ids=["dynamic", "frozen"])
@pytest.mark.parametrize("theta", [30.0, 45.0])
def test_j1_jacobian_matches_central_differences(params, frozen, theta):
    """Budget: the 4-point stencil's truncation h^4 f^(5)/30 with h = 2e-7 V against the ~4 V/E_g scale of
    the exponent's psi-dependence (B/F ~ 20: f^(5)/f ~ (20/1.12)^5) -> ~1e-14, plus round-off
    1.5 eps |F|/h relative to the largest entry ~ 1e-9 (|F|/(h |J|) ~ 1e6): tolerance 1e-8 of the largest
    Jacobian entry. MEASURED <= 1.1e-9 (dynamic and frozen)."""
    psi0, resid, N = _graded_case(theta, frozen, params)
    _, r = resid(psi0, True)
    assert r["pathset"].start.size > 300 and r["pathset"].dynamic == (not frozen)
    J = _assembled_jacobian(r, N)
    worst, n = _fd_worst(psi0, resid, J, N)
    print(theta, "frozen" if frozen else "dynamic", params.P, "worst", worst, "columns", n)
    assert n >= 30 and worst < 1e-8


def test_j1_refute_dynamic_without_rotation_terms():
    psi0, resid, N = _graded_case(30.0, False, PHONON)
    _, r = resid(psi0, True)
    worst, n = _fd_worst(psi0, resid, _assembled_jacobian(r, N, rotation=False), N)
    print("without rotation terms: worst", worst)
    assert worst > 1e-3


def test_n0_numba_required(monkeypatch):
    import dynameta.numba_env as NE
    monkeypatch.setattr(NE, "_NUMBA_OK", False)
    monkeypatch.setattr(B2, "have_numba", NE.have_numba)
    with pytest.raises(ImportError, match="decision 23"):
        B2.require_numba("test")
