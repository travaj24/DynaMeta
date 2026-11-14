"""Hooge 1/f noise in 3-D: one per-element kernel for 2-D and 3-D (v0.14.0 blocker, 2026-11-06).

The Hooge source is a fluctuation of the local conductivity, delta j = xi J with ONE scalar xi per
point (noise_analysis module doc, items 2-3); it is assembled per element as
    S_T = (alpha_H / f) |Sum_{e in T} c_eT a_e (y_0 - y_1)|^2 / V_T ,   a_e = J_e / sqrt(n_eff),
which equals the continuum V_T alpha_H (J . grad y)^2 / (n f) for a uniform J and n iff the element
weights satisfy the TENSOR identity Sum_e c_eT L_e t_e t_e^T = V_T I. Until this fix c_eT was DEVSIM's
circumcentric ElementEdgeCouple: in 2-D that is the cotangent (P1) weight and the identity holds, but in
3-D only its TRACE does -- a Kuhn path tet (steps x, y, z) gives diag(0.75, 1.5, 0.75) V_T, so an
axis-aligned current sees <m^2> = (2 x 1.5^2 + 4 x 0.75^2) / 6 = 1.125 over the six path tets of a cell.
The fix uses the P1 element couples -V_T L_e grad(lambda_i) . grad(lambda_j) (exact per element in any
dimension; = the circumcentric couple in 2-D).

THE ORACLE, re-derived here. A uniform ohmic bar (length L along the current, cross-section A, carrier
density n, current I): the adjoint field of the terminal current is y = (x . u)/L (a unit current source
at x reaches the contacts in proportion to its position), J = I/A along u, so
    S_I = Int alpha_H (J . grad y)^2 / (n f) dV = alpha_H (I/A)^2 (1/L)^2 (A L) / (n f) = alpha_H I^2 / (N f),
N = n A L, at ANY orientation of the bar against the mesh. The fixtures check their own exactness first
(n uniform, I = q mu n V A/L), so the closed form holds on the DISCRETE device.

ERROR BUDGET (derived, not tuned). With uniform n and a linear potential the SG edge current is exact
and the SG edge density of noise_analysis is n_eff = n (v/2) coth(v/2) <= n (1 + v^2/12) (v the edge's SG
argument vdiff_g; x coth x <= 1 + x^2/3), so a_e carries a relative error eps_e = 1 - sqrt(n/n_eff) in
[0, v_max^2/24]. Per element A_T = A_T,exact (1 - delta_T) with |delta_T| <= eps_max kappa_T, kappa_T =
Sum_k |c_kT| L_k (t_k . u)^2 / V_T (= 1 when no couple is negative; > 1 on non-well-centred elements, whose
P1 couples mix signs), and S_T ~ V_T, so
    |S f N / (alpha I^2) - 1| <= Sum_T V_T (2 eps_max kappa_T + (eps_max kappa_T)^2) / Sum_T V_T
plus 4 |I/I_ohm - 1| (S ~ J^2 against the measured I^2) and a 1e-11 summation floor (~1e3 elements).

MEASURED (dev box, 2026-11-06; S f N / (alpha I^2) - 1 at 1 Hz; the derived budget):
                                                 be719d5 (circumcentric)   this fix     budget
  Kuhn bar 10x4x4 cells, current along x            +0.1249999935        -5.8015e-9   5.812e-9
  the same device rigidly rotated                   +0.1249999935        -5.8015e-9   5.812e-9
  rotated-lattice Delaunay 8x4x4, exact geometry    +0.0547369783        +2.7101e-9   1.155e-7
    (1031 tets, 51 negative assembled couples, kappa mean 5.1 / max 2.7e4)
  2-D rotated-lattice Delaunay 16x8, exact geom.    -1.7699383e-9        -1.7699383e-9 4.590e-9
    (2-D does not move: the P1 couple IS the signed circumcentric couple there; diff 2.2e-16)
  wrong-edge perturbation (_broken_couples)         Kuhn +0.770, Delaunay 3-D +1.299, 2-D +1.341
Without apply_exact_geometry the Delaunay fixtures are invalid (I/I_ohm - 1 = 0.338 in 3-D, 0.036 in 2-D)
and read -0.244 / -0.048 (circumcentric: -0.198 / -0.022): DEVSIM's own solve is off there, not the
Hooge kernel -- the reason the gates apply the exact geometry.
"""
import itertools
import warnings

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
scipy_spatial = pytest.importorskip("scipy.spatial")
# NOT importorskip: a missing noise_analysis must FAIL these gates, not skip them
import validation.devsim_noise as V                                         # noqa: E402
from dynameta.carriers import devsim_geometry as DG                         # noqa: E402
from dynameta.carriers import noise_analysis as NA                          # noqa: E402
from dynameta.carriers import physics_drift_diffusion as DD                 # noqa: E402
from dynameta.carriers.devsim_env import import_devsim                      # noqa: E402
from dynameta.carriers.physics_equilibrium import M_E                       # noqa: E402
from dynameta.warning_types import UnreliableResultWarning                  # noqa: E402

ds = import_devsim()

H = 20e-9                   # cell size [m]
N_BG, MU = 4e26, 0.004      # uniform donor density [m^-3], mobility [m^2/Vs] (the gate-A ITO bar)
ALPHA, V_BIAS = 2e-3, 1e-3
FLOOR = 1e-11               # summation round-off floor of the ratio (~1e3 elements, ~1e-16 each)


# ------------------------------------------------------------------------------------------------
# in-memory meshes (independent of the library meshers)
# ------------------------------------------------------------------------------------------------

def _facets(cells):
    cnt = {}
    for c in cells:
        for f in itertools.combinations(sorted(int(v) for v in c), len(c) - 1):
            cnt[f] = cnt.get(f, 0) + 1
    return [f for f, k in cnt.items() if k == 1]


def _end_facets(P, T, ext):
    """Boundary facets on the two planes x = 0 (lo) and x = ext[0] (hi) of the box."""
    f = _facets(T)
    lo = [q for q in f if np.allclose(P[list(q), 0], 0.0, atol=1e-9 * ext[0])]
    hi = [q for q in f if np.allclose(P[list(q), 0], ext[0], atol=1e-9 * ext[0])]
    return lo, hi


def kuhn_box(n):
    """n[0] x n[1] x n[2] unit cubes, 6 Kuhn path tets each (all 6 axis orders, so every axis plays
    every step: the x-current result holds for y and z by the construction's symmetry)."""
    g = [np.arange(k + 1, dtype=float) for k in n]
    P = np.array([[x, y, z] for x in g[0] for y in g[1] for z in g[2]])
    idx = lambda i, j, k: (i * (n[1] + 1) + j) * (n[2] + 1) + k          # noqa: E731
    T = []
    for i, j, k in itertools.product(range(n[0]), range(n[1]), range(n[2])):
        for perm in itertools.permutations(range(3)):
            v = [i, j, k]
            path = [idx(*v)]
            for ax in perm:
                v[ax] += 1
                path.append(idx(*v))
            T.append(path)
    return P, np.array(T), np.asarray(n, dtype=float)


def rotated_lattice_box(n, seed=7, dim=3):
    """Qhull Delaunay of a box [0, n] (unit spacing): the INTERIOR points are a ROTATED, slightly
    jittered cubic lattice (so the current along x is oblique to essentially every edge), the boundary
    points a tangentially jittered grid on each face / edge (planar faces, obtuse boundary triangles ->
    non-well-centred elements and negative boundary couples)."""
    rng = np.random.default_rng(seed)
    ext = np.asarray(n, dtype=float)
    ang = (0.37, 0.61, 0.23) if dim == 3 else (0.41,)
    if dim == 3:
        a, b, c = ang
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
        R = Rz @ Ry @ Rx
    else:
        R = np.array([[np.cos(ang[0]), -np.sin(ang[0])], [np.sin(ang[0]), np.cos(ang[0])]])
    m = int(np.ceil(np.linalg.norm(ext))) + 2
    G = np.array(list(itertools.product(*([np.arange(-m, m + 1, dtype=float)] * dim))))
    Q = (G @ R.T) + 0.5 * ext + rng.uniform(-0.05, 0.05, G.shape)
    Q = Q[np.all((Q > 0.3) & (Q < ext - 0.3), axis=1)]
    B = np.array(list(itertools.product(*[np.arange(k + 1, dtype=float) for k in n])))
    on = np.isclose(B, 0.0) | np.isclose(B, ext)
    B = B[on.any(axis=1)]
    on = np.isclose(B, 0.0) | np.isclose(B, ext)
    B = B + np.where(on, 0.0, rng.uniform(-0.2, 0.2, B.shape))      # move only ALONG the boundary
    P = np.vstack([B, Q])
    T = scipy_spatial.Delaunay(P).simplices
    X = P[T]
    vol = np.abs(np.linalg.det(X[:, 1:] - X[:, :1])) / (6.0 if dim == 3 else 2.0)
    assert vol.min() > 1e-6 and abs(vol.sum() / np.prod(ext) - 1.0) < 1e-12, (vol.min(), vol.sum())
    return P, T, ext


def _create(name, P_m, T, lo, hi):
    """DEVSIM device from arrays (create_gmsh_mesh(coordinates=, elements=): no file)."""
    dim = T.shape[1] - 1
    coords = [float(c) for p in P_m for c in (list(p) + [0.0] * (3 - len(p)))]
    elems = []
    for k, cells in enumerate((T, lo, hi)):
        for c in cells:
            elems += [{2: 1, 3: 2, 4: 3}[len(c)], k] + [int(v) for v in c]
    ds.create_gmsh_mesh(mesh=name, coordinates=coords, physical_names=["bulk", "slo", "shi"],
                        elements=elems)
    ds.add_gmsh_region(mesh=name, gmsh_name="bulk", region="bulk", material="ITO")
    ds.add_gmsh_contact(mesh=name, gmsh_name="slo", region="bulk", name="lo", material="metal")
    ds.add_gmsh_contact(mesh=name, gmsh_name="shi", region="bulk", name="hi", material="metal")
    ds.finalize_mesh(mesh=name)
    ds.create_device(mesh=name, device=name)
    assert int(ds.get_dimension(device=name)) == dim


def _hooge_ratio(name, P, T, ext, *, exact=False, rotate=None):
    """Build + bias the uniform bar (current along x of the box frame; `rotate` turns the whole device);
    returns (S f N / (alpha I^2) - 1 at 1 Hz and 1 kHz, its derived budget, the fixture checks)."""
    lo, hi = _end_facets(P, T, ext)
    P_m = P * H if rotate is None else (P * H) @ np.asarray(rotate).T
    _create(name, P_m, T, lo, hi)
    n_neg = None
    if exact:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = DG.apply_exact_geometry(name, "bulk", negative="warn")
        assert all(issubclass(x.category, UnreliableResultWarning) for x in w), [str(x.message) for x in w]
        assert len(w) == int(rep.has_negatives)
        n_neg = rep.n_negative_couples
    with V.quiet():
        DD.setup_semiconductor_region_dd(name, "bulk", n_bg_m3=N_BG, eps_static=9.5,
                                         dos_mass_kg=0.35 * M_E, mobility_m2Vs=MU)
        DD.setup_contact_ohmic_dd(name, "lo")
        DD.setup_contact_ohmic_dd_circuit(name, "hi", node_name="vhi", source_name="V1")
    ae = max(1e10, N_BG * 1e-12)
    V._solve(ae, 1e-9)
    V.set_bias(V_BIAS, ae)
    I = V.source_current()
    n = np.asarray(ds.get_node_model_values(device=name, region="bulk", name="Electrons"))
    v = np.asarray(ds.get_edge_model_values(device=name, region="bulk", name="vdiff_g"))
    Lx, area = ext[0] * H, float(np.prod(ext[1:])) * H ** (len(ext) - 1)
    N = N_BG * Lx * area                                       # closed form (per unit depth in 2-D)
    ohm = float(I / (V.QE * MU * N_BG * V_BIAS * area / Lx) - 1.0)
    ff = np.array([1.0, 1.0e3])
    sn = V.noise(ff, hooge_alpha=ALPHA, maps=True)
    tot = sn.maps["bulk"]["hooge_n"].sum(axis=1)               # the per-node map sums to the total
    assert np.allclose(tot, sn.by_mechanism["hooge_n"], rtol=1e-12, atol=0.0)
    eps = float(np.max(v * v)) / 24.0                         # >= 1 - sqrt(n / n_eff) on every edge
    kap, Vt = _kappa(P_m, T, rotate)
    budget = float(np.sum(Vt * (2.0 * eps * kap + (eps * kap) ** 2)) / np.sum(Vt)) + 4.0 * abs(ohm) + FLOOR
    checks = {"n_spread": float(n.max() / n.min() - 1.0), "n_bg": float(n.mean() / N_BG - 1.0),
              "ohm": ohm, "negative_couples": n_neg, "kappa_max": float(kap.max()),
              "kappa_mean": float(np.sum(Vt * kap) / np.sum(Vt))}
    return sn.by_mechanism["hooge_n"] * ff * N / (ALPHA * I * I) - 1.0, budget, checks


def _kappa(P_m, T, rotate):
    """(kappa_T, V_T): kappa_T = Sum_k |W_k| L_k (t_k . u)^2 / V_T for the current direction u (box x axis,
    rotated with the device) -- how much sign-mixed element weights amplify the per-edge density error
    (= 1 when every W >= 0, since Sum_k W_k L_k (t_k . u)^2 = V_T). W from the kernel that
    test_p1_couples_tensor_identity_and_2d_cotangent pins against independent oracles (the unpatched
    reference, so a perturbation cannot widen the budget)."""
    X = np.asarray(P_m, dtype=float)[T]
    d = T.shape[1] - 1
    u = np.eye(d)[0] if rotate is None else np.asarray(rotate)[:, 0]
    pairs = list(itertools.combinations(range(d + 1), 2))
    a = np.tile([p[0] for p in pairs], (len(T), 1))
    b = np.tile([p[1] for p in pairs], (len(T), 1))
    W, Vt = _p1_ref(X, a, b)
    e = X[np.arange(len(T))[:, None], b] - X[np.arange(len(T))[:, None], a]
    L = np.linalg.norm(e, axis=2)
    return np.sum(np.abs(W) * (e @ u) ** 2 / L, axis=1) / Vt, Vt


def _assert_fixture(chk):
    """The closed form holds on the DISCRETE fixture: uniform n = N_D, ohmic current."""
    assert abs(chk["n_spread"]) < 1e-12 and abs(chk["n_bg"]) < 1e-12, chk
    assert abs(chk["ohm"]) < 1e-9, chk


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()
    V.set_bias._v = {}


def _rot3(a=0.5, b=-0.8, c=1.3):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return (np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]) @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            @ np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]]))


def _broken_couples(X, a, b):
    """Deliberate perturbation: the element's couples assigned to the wrong edges (reversed local edge
    order -- opposite edges swapped in a tet). Same weights, same volume, wrong tensor."""
    W, Vt = _p1_ref(X, a, b)
    return W[:, ::-1], Vt


_p1_ref = NA._p1_element_couples


def test_hooge_3d_kuhn_bar_closed_form(monkeypatch):
    """The DEFAULT 3-D mesh (the Kuhn tensor bar of validation.devsim_noise.unipolar_3d_resistor:
    10 x 4 x 4 cells of 20 nm, 960 tets), axis-aligned current, and the SAME device rigidly rotated
    (the P1 couples come from the node coordinates: frame invariance). MEASURED: -5.80e-9 both (budget
    5.81e-9: v_max^2/12 of the degenerate-FD SG argument -- the deviation IS that term, kappa = 1);
    be719d5 (circumcentric couples): +0.125 both. Perturbation (couples on the wrong edges): +0.770."""
    P, T, ext = kuhn_box((10, 4, 4))
    out = {}
    for case, rot in (("aligned", None), ("rotated", _rot3())):
        r, budget, chk = _hooge_ratio("hk3_" + case, P, T, ext, rotate=rot)
        _assert_fixture(chk)
        assert np.max(np.abs(r)) <= budget, (case, r, budget)
        out[case] = r
        V.fresh()
        V.set_bias._v = {}
    assert np.max(np.abs(out["rotated"] - out["aligned"])) < 1e-11, out
    monkeypatch.setattr(NA, "_p1_element_couples", _broken_couples)
    r, budget, _ = _hooge_ratio("hk3_broken", P, T, ext)
    assert np.min(np.abs(r)) > 1e-2, r


def test_hooge_3d_oblique_non_well_centred_exact_geometry(monkeypatch):
    """Current OBLIQUE to every edge on a non-well-centred 3-D mesh: rotated-lattice Delaunay box
    (8 x 4 x 4 cells of 20 nm, boundary faces with obtuse triangles), with devsim_geometry.
    apply_exact_geometry(negative='warn') so DEVSIM's own DC / adjoint solve is exact (patch test)
    and the fixture checks hold. MEASURED: +2.71e-9 (budget 1.16e-7); be719d5 (circumcentric couples,
    signed after the correction): +0.0547; perturbation (couples on the wrong edges): +1.299."""
    P, T, ext = rotated_lattice_box((8, 4, 4))
    r, budget, chk = _hooge_ratio("hk3_obl", P, T, ext, exact=True)
    _assert_fixture(chk)
    assert np.max(np.abs(r)) <= budget, (r, budget, chk)
    V.fresh()
    V.set_bias._v = {}
    monkeypatch.setattr(NA, "_p1_element_couples", _broken_couples)
    r, budget, _ = _hooge_ratio("hk3_obl_broken", P, T, ext, exact=True)
    assert np.min(np.abs(r)) > 1e-2, r


def test_hooge_2d_oblique_obtuse_exact_geometry(monkeypatch):
    """The SAME kernel in 2-D, current oblique to every edge on an obtuse (non-well-centred) Delaunay
    mesh with the exact geometry: the P1 couple is the signed cotangent weight there, so the 2-D
    result is the pre-fix one to round-off: MEASURED -1.7699383e-9 both (budget 4.59e-9; diff 2.2e-16).
    Perturbation: couples on the wrong edges, +1.341."""
    P, T, ext = rotated_lattice_box((16, 8), dim=2)
    r, budget, chk = _hooge_ratio("hk2_obl", P, T, ext, exact=True)
    _assert_fixture(chk)
    assert np.max(np.abs(r)) <= budget, (r, budget, chk)
    V.fresh()
    V.set_bias._v = {}
    monkeypatch.setattr(NA, "_p1_element_couples", _broken_couples)
    r, budget, _ = _hooge_ratio("hk2_obl_broken", P, T, ext, exact=True)
    assert np.min(np.abs(r)) > 1e-2, r


def test_p1_couples_tensor_identity_and_2d_cotangent():
    """Kernel gate, independent oracles: on random simplices (any shape, obtuse included)
    Sum_k W_k (x_b - x_a)(x_b - x_a)^T / L_k = |det|/d! I; in 2-D W = L cot(theta)/2 with theta the
    ANGLE opposite the edge (arccos of edge-vector dot products). Perturbation: the circumcentric
    tetrahedron couple (devsim_geometry, signed) violates the 3-D identity. MEASURED: identity error /
    budget <= 0.013 (2-D) / 0.0058 (3-D); circumcentric worst deviation 67.9 V_T (asserted > 1e-2)."""
    rng = np.random.default_rng(3)
    for d in (2, 3):
        X = rng.normal(size=(400, d + 1, d))
        pairs = list(itertools.combinations(range(d + 1), 2))
        a = np.tile([p[0] for p in pairs], (400, 1))
        b = np.tile([p[1] for p in pairs], (400, 1))
        W, Vt = NA._p1_element_couples(X, a, b)
        e = X[np.arange(400)[:, None], b] - X[np.arange(400)[:, None], a]
        L = np.linalg.norm(e, axis=2)
        M = np.einsum("tk,tki,tkj->tij", W / L, e, e)
        vol = np.abs(np.linalg.det(X[:, 1:] - X[:, :1])) / (2.0 if d == 2 else 6.0)
        assert np.allclose(Vt, vol, rtol=1e-13, atol=0.0)
        # round-off budget: the gradients carry ~eps cond(D) relative error, each term of the sum is
        # bounded by |W_k| L_k, so |M - V I| <= 64 eps cond(D) Sum_k |W_k| L_k (64: summation + inversion)
        D = X[:, 1:] - X[:, :1]
        tol = 64.0 * np.finfo(float).eps * np.linalg.cond(D) * np.sum(np.abs(W) * L, axis=1)
        err = np.max(np.abs(M - vol[:, None, None] * np.eye(d)), axis=(1, 2))
        assert np.all(err <= tol), np.max(err / tol)
        if d == 2:
            for k, (i, j) in enumerate(pairs):
                o = 3 - i - j
                u, w = X[:, i] - X[:, o], X[:, j] - X[:, o]
                th = np.arccos(np.einsum("ti,ti->t", u, w) / np.linalg.norm(u, axis=1) / np.linalg.norm(w, axis=1))
                # arccos near 0 / pi amplifies its eps by 1/sin; cot' = -1/sin^2
                tol_c = 64.0 * np.finfo(float).eps * (np.linalg.cond(D) * np.abs(W[:, k])
                                                      + L[:, k] / (2.0 * np.sin(th) ** 3))
                assert np.all(np.abs(W[:, k] - L[:, k] / np.tan(th) / 2.0) <= tol_c)
        else:
            sg, _us, Lc = DG.tetrahedron_dual_pieces(X)        # local edges (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
            Mc = np.einsum("tk,tki,tkj->tij", sg / Lc, e, e)   # same edge order as `pairs`
            dev = np.max(np.abs(Mc - vol[:, None, None] * np.eye(3)) / vol[:, None, None], axis=(1, 2))
            assert np.max(dev) > 1e-2, np.max(dev)
