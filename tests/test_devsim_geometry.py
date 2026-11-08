"""Exact (signed) box-method geometry for DEVSIM (dynameta.carriers.devsim_geometry, v0.14.0 opt-in
workaround; docs/audit/2026-11-06-devsim-geometry-workaround.md).

DEVSIM 2.11.2 sums the MAGNITUDES of the circumcentric dual pieces (TriangleEdgeCouple.cc:74,
TetrahedronEdgeCouple.cc:99, SurfaceAreaUtil.cc processTriangle), so on any element that is not
well-centred its control volumes / couples / surface areas over-count. apply_exact_geometry writes the
SIGNED pieces into DEVSIM's own ElementEdgeCouple (DEVSIM rebuilds the rest) and, in 3-D, SurfaceArea /
ContactSurfaceArea.

Oracles (independent of the module's kernels):
  * 2-D: the cotangent weight L cot(theta)/2 from the triangle's ANGLES (arccos of edge dot products),
    i.e. the P1-FEM stiffness, and the triangle area from the cross product;
  * 3-D: the tetrahedron volume |det|/6, and Qhull's Voronoi diagram (scipy.spatial.Voronoi + ConvexHull)
    for interior node cells and interior edge facets of a Delaunay mesh;
  * closed-form fields: the parallel-plate capacitor (phi = z exactly, C = eps A / d), Gauss's law
    for a uniform charge (Q_lo + Q_hi = rho V), and a uniform drift-diffusion resistor (G = q n mu A / L).
Every gate is shown failing on DEVSIM's own (uncorrected) geometry, by the MEASURED amount in its
docstring, and carries one deliberate perturbation that must break it.

All meshes are built in memory (create_gmsh_mesh(coordinates=, elements=)); device / mesh names are
unique per test (DEVSIM state is process-global).
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("devsim")
scipy_spatial = pytest.importorskip("scipy.spatial")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers import devsim_geometry as G                                  # noqa: E402
from dynameta.carriers.devsim_env import import_devsim                              # noqa: E402
from dynameta.warning_types import UnreliableResultWarning                          # noqa: E402

ds = import_devsim()
EPS = np.finfo(float).eps


def _drop_all():
    """ds.solve() solves EVERY device in the process: keep only the test's own."""
    for d in list(ds.get_device_list()):
        ds.delete_device(device=d)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)


@pytest.fixture(autouse=True)
def _clean():
    _drop_all()
    yield
    _drop_all()


# ------------------------------------------------------------------------------------------------
# in-memory meshes
# ------------------------------------------------------------------------------------------------

def _facets(cells):
    """Boundary facets (faces of exactly one cell) of a simplicial mesh, as sorted tuples."""
    cnt = {}
    for c in cells:
        for f in itertools.combinations(sorted(int(v) for v in c), len(c) - 1):
            cnt[f] = cnt.get(f, 0) + 1
    return [f for f, k in cnt.items() if k == 1]


def _device(name, P, regions, surfaces=(), contacts=(), interfaces=()):
    """DEVSIM device from arrays, no file. regions {name: cells}; surfaces {name: facets};
    contacts [(name, surface, region)]; interfaces [(name, surface, r0, r1)]."""
    P = np.asarray(P, dtype=float)
    dim = len(next(iter(regions.values()))[0]) - 1
    coords = [float(c) for p in P for c in (list(p) + [0.0] * (3 - len(p)))]
    names, elems = [], []
    for k, (nm, cells) in enumerate(list(regions.items()) + list(dict(surfaces).items())):
        names.append(nm)
        for c in cells:
            et = {2: 1, 3: 2, 4: 3}[len(c)]
            elems += [et, k] + [int(v) for v in c]
    ds.create_gmsh_mesh(mesh=name, coordinates=coords, physical_names=names, elements=elems)
    for nm in regions:
        ds.add_gmsh_region(mesh=name, gmsh_name=nm, region=nm, material="X")
    for (cn, surf, reg) in contacts:
        ds.add_gmsh_contact(mesh=name, gmsh_name=surf, region=reg, name=cn, material="metal")
    for (iname, surf, r0, r1) in interfaces:
        ds.add_gmsh_interface(mesh=name, gmsh_name=surf, region0=r0, region1=r1, name=iname)
    ds.finalize_mesh(mesh=name)
    ds.create_device(mesh=name, device=name)
    assert dim == int(ds.get_dimension(device=name))
    return name


def _lattice(n, h=1.0):
    g = np.arange(n + 1) * h
    P = np.array([[x, y, z] for z in g for y in g for x in g])
    idx = lambda i, j, k: i + (n + 1) * (j + (n + 1) * k)              # noqa: E731
    return P, idx


def five_tet_lattice(n):
    """n^3 unit cubes, each split into 5 tets (4 trirectangular corner tets + 1 regular), parity-
    alternated so the face diagonals match (all diagonals join globally-even vertices): conforming, and
    80 % of the tets are NOT well-centred (DEVSIM: 2x per corner tet, 1.667x the cube)."""
    P, idx = _lattice(n)
    T = []
    for i, j, k in itertools.product(range(n), repeat=3):
        v = {(a, b, c): idx(i + a, j + b, k + c) for a, b, c in itertools.product((0, 1), repeat=3)}
        even = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        odd = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        cen, cor = (even, odd) if (i + j + k) % 2 == 0 else (odd, even)
        T.append([v[q] for q in cen])
        for q in cor:
            T.append([v[q]] + [v[r] for r in cen if sum(abs(np.subtract(q, r))) == 1])
    return P, np.array(T)


def kuhn_lattice(n):
    """n^3 unit cubes, 6 path (Kuhn) tets each: every piece >= 0, DEVSIM exact (the well-centred control)."""
    P, idx = _lattice(n)
    T = []
    for i, j, k in itertools.product(range(n), repeat=3):
        for perm in itertools.permutations(range(3)):
            v = [i, j, k]
            path = [idx(*v)]
            for ax in perm:
                v[ax] += 1
                path.append(idx(*v))
            T.append(path)
    return P, np.array(T)


def jittered_delaunay(n, jit, seed, dim=3, split=None):
    """Qhull Delaunay of a jittered lattice of [0,1]^dim; boundary points move only ALONG their boundary
    (so the faces stay planar but their triangles are obtuse), likewise points on the plane z = split
    (then each half is triangulated separately; their restrictions to the plane are the same 2-D Delaunay
    triangulation, so the two regions conform). Returns P and {region: cells}."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, 1.0, n + 1)
    P = np.array(list(itertools.product(*([g] * dim))))[:, ::-1]
    d = rng.uniform(-jit / n, jit / n, P.shape)
    fixed = np.isclose(P, 0.0) | np.isclose(P, 1.0)
    if split is not None:
        fixed[:, -1] |= np.isclose(P[:, -1], split)
    d[fixed] = 0.0
    P = P + d
    if split is None:
        groups = {"r": np.arange(len(P))}
    else:
        groups = {"a": np.flatnonzero(P[:, -1] <= split + 1e-12), "b": np.flatnonzero(P[:, -1] >= split - 1e-12)}
    regions = {}
    for nm, ids in groups.items():
        T = ids[scipy_spatial.Delaunay(P[ids]).simplices]
        regions[nm] = T
    return P, regions


def _true_volume(P, T):
    P = np.asarray(P, float)
    X = P[T]
    d = X[:, 1:] - X[:, :1]
    if X.shape[1] == 3:
        return np.abs(d[:, 0, 0] * d[:, 1, 1] - d[:, 0, 1] * d[:, 1, 0]) / 2.0
    return np.abs(np.linalg.det(d)) / 6.0


def _plates(P, T, axis=-1):
    """Facets on the two end planes of `axis` (lo at 0, hi at 1)."""
    P = np.asarray(P, float)
    f = _facets(T)
    lo = [q for q in f if np.allclose(P[list(q), axis], 0.0)]
    hi = [q for q in f if np.allclose(P[list(q), axis], 1.0)]
    return lo, hi


def nv(dev, r, name="NodeVolume"):
    return np.asarray(ds.get_node_model_values(device=dev, region=r, name=name))


def ev(dev, r, name):
    return np.asarray(ds.get_edge_model_values(device=dev, region=r, name=name))


def elv(dev, r, name):
    return np.asarray(ds.get_element_model_values(device=dev, region=r, name=name))


# ------------------------------------------------------------------------------------------------
# the capacitor / Gauss problem (eps = 1, plates lo at axis = 0 and hi at axis = 1)
# ------------------------------------------------------------------------------------------------

def _laplace(dev, regions, contacts, interfaces=()):
    ds.set_parameter(device=dev, name="rho", value=0.0)
    ds.set_parameter(device=dev, name="Vhi", value=1.0)
    for r in regions:
        ds.node_solution(device=dev, region=r, name="Potential")
        ds.edge_from_node_model(device=dev, region=r, node_model="Potential")
        ds.edge_model(device=dev, region=r, name="DField",
                      equation="(Potential@n0 - Potential@n1)*EdgeInverseLength")
        ds.edge_model(device=dev, region=r, name="DField:Potential@n0", equation="EdgeInverseLength")
        ds.edge_model(device=dev, region=r, name="DField:Potential@n1", equation="-EdgeInverseLength")
        ds.node_model(device=dev, region=r, name="NegRho", equation="-rho")
        ds.node_model(device=dev, region=r, name="NegRho:Potential", equation="0")
        ds.equation(device=dev, region=r, name="PotentialEquation", variable_name="Potential",
                    edge_model="DField", node_model="NegRho", variable_update="default")
    for (c, v) in contacts:
        ds.contact_node_model(device=dev, contact=c, name="bc_" + c, equation="Potential - " + v)
        ds.contact_node_model(device=dev, contact=c, name="bc_{}:Potential".format(c), equation="1")
        ds.contact_equation(device=dev, contact=c, name="PotentialEquation", node_model="bc_" + c,
                            edge_charge_model="DField", node_charge_model="NegRho")
    for i in interfaces:
        ds.interface_model(device=dev, interface=i, name="cont", equation="Potential@r0 - Potential@r1")
        ds.interface_model(device=dev, interface=i, name="cont:Potential@r0", equation="1")
        ds.interface_model(device=dev, interface=i, name="cont:Potential@r1", equation="-1")
        ds.interface_equation(device=dev, interface=i, name="PotentialEquation", interface_model="cont",
                              type="continuous")


def _solve():
    ds.solve(type="dc", absolute_error=1e-14, relative_error=1e-14, maximum_iterations=6)


def _capacitor_metrics(dev, regions, axis=-1):
    """(max |phi - x_axis|, C = |Q_hi| at V = 1, |Q_lo + Q_hi| / (rho V) at rho = 1, V = 0)."""
    ds.set_parameter(device=dev, name="rho", value=0.0)
    ds.set_parameter(device=dev, name="Vhi", value=1.0)
    _solve()
    coord = "xyz"[axis] if axis >= 0 else ("y" if int(ds.get_dimension(device=dev)) == 2 else "z")
    err = max(float(np.max(np.abs(nv(dev, r, "Potential") - nv(dev, r, coord)))) for r in regions)
    C = abs(ds.get_contact_charge(device=dev, contact="hi", equation="PotentialEquation"))
    ds.set_parameter(device=dev, name="rho", value=1.0)
    ds.set_parameter(device=dev, name="Vhi", value=0.0)
    _solve()
    Q = abs(ds.get_contact_charge(device=dev, contact="hi", equation="PotentialEquation")
            + ds.get_contact_charge(device=dev, contact="lo", equation="PotentialEquation"))
    return err, C, Q


def _cap_device(name, P, regions, axis=-1):
    T_all = np.vstack(list(regions.values()))
    lo, hi = _plates(P, T_all, axis)
    reg_of = {}
    for r, T in regions.items():
        for f in _facets(T):
            reg_of.setdefault(f, r)
    contacts = [("lo", "slo", reg_of[lo[0]]), ("hi", "shi", reg_of[hi[0]])]
    surfaces = {"slo": lo, "shi": hi}
    interfaces = []
    if len(regions) == 2:
        fa, fb = set(_facets(regions["a"])), set(_facets(regions["b"]))
        surfaces["sif"] = sorted(fa & fb)
        interfaces = [("if0", "sif", "a", "b")]
    _device(name, P, regions, surfaces, contacts, interfaces)
    _laplace(name, list(regions), [("lo", "0"), ("hi", "Vhi")], [i[0] for i in interfaces])
    return name


# ================================================================================================
# kernels vs independent geometry (no DEVSIM)
# ================================================================================================

def test_triangle_kernel_is_the_cotangent_weight():
    """2-D: the signed piece of edge j is L_j cot(theta_j)/2 (theta_j the angle OPPOSITE edge j, from
    arccos of the two edge vectors -- the P1-FEM stiffness weight, an independent construction), the
    pieces partition the triangle (sum s L / 2 = area), and DEVSIM's |s| does not: the obtuse triangle
    (0,0),(2,0),(1,0.3) gives sum |s| L / 2 = 3.333 = 11.1 x its area 0.3 (MEASURED here and in DEVSIM)."""
    rng = np.random.default_rng(7)
    X = np.concatenate([rng.uniform(0, 1, (200, 3, 2)), [[[0, 0], [2, 0], [1, 0.3]]]])
    s, L, _c = G.triangle_dual_pieces(X)
    area = 0.5 * np.abs(np.cross(np.pad(X[:, 1] - X[:, 0], ((0, 0), (0, 1))),
                                 np.pad(X[:, 2] - X[:, 0], ((0, 0), (0, 1))))[:, 2])
    keep = area > 1e-3 * L.max(axis=1) ** 2                        # drop near-degenerate random ones
    X3 = np.pad(X, ((0, 0), (0, 0), (0, 1)))
    cot = np.empty_like(s)
    for j in range(3):
        u, v = X3[:, (j + 1) % 3] - X3[:, j], X3[:, (j + 2) % 3] - X3[:, j]
        cot[:, j] = 0.5 * L[:, j] * np.einsum("ij,ij->i", u, v) / np.linalg.norm(np.cross(u, v), axis=1)
    # budget: both sides are a few flops of O(L + |cot weight|) magnitude each -> 64 eps of that
    # (MEASURED max 6.4 eps on the kept triangles)
    assert np.all((np.abs(s - cot) <= 64 * EPS * (L + 2.0 * np.abs(cot)))[keep])
    part = (s * L).sum(axis=1) / 2.0
    assert np.max(np.abs(part / area - 1.0)[keep]) < 1e-11
    obt = (np.abs(s) * L).sum(axis=1)[-1] / 2.0
    assert obt == pytest.approx(10.0 / 3.0, rel=1e-12) and obt / area[-1] == pytest.approx(100.0 / 9.0)
    # adversarial: DEVSIM's magnitudes miss the partition on every obtuse triangle in the sample
    obtuse = keep & np.any(s < 0, axis=1)
    assert obtuse.sum() > 20
    assert np.min((np.abs(s) * L).sum(axis=1)[obtuse] / 2.0 / area[obtuse]) > 1.0 + 1e-6


def test_tetrahedron_pieces_partition_every_tet():
    """3-D: sum_e c_e L_e / 3 = |det|/6 for every tet (the 12 pyramids tile it), incl. the trirectangular
    corner tet and all five tets of the co-spherical cube split. DEVSIM's construction (the `unsigned`
    output, re-derived here) gives exactly 2 x on the corner tet and 5/3 x on the 5-tet cube (MEASURED;
    DEVSIM reports the same, test_devsim_volume_defect_and_exact_partition)."""
    rng = np.random.default_rng(11)
    X = rng.uniform(0, 1, (300, 4, 3))
    V = np.abs(np.linalg.det(X[:, 1:] - X[:, :1])) / 6.0
    X, V = X[V > 1e-3], V[V > 1e-3]
    sg, us, L = G.tetrahedron_dual_pieces(X)
    assert np.max(np.abs((sg * L).sum(axis=1) / 3.0 / V - 1.0)) < 1e-11
    assert np.any(sg < 0) and np.min((us * L).sum(axis=1) / 3.0 / V) >= 1.0 - 1e-12
    corner = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], float)
    sg, us, L = G.tetrahedron_dual_pieces(corner)
    assert (sg * L).sum() / 3.0 == pytest.approx(1.0 / 6.0, rel=1e-14)
    assert (us * L).sum() / 3.0 == pytest.approx(2.0 / 6.0, rel=1e-14)
    P, T = five_tet_lattice(1)
    sg, us, L = G.tetrahedron_dual_pieces(P[T])
    assert (sg * L).sum() / 3.0 == pytest.approx(1.0, rel=1e-14)
    assert (us * L).sum() / 3.0 == pytest.approx(5.0 / 3.0, rel=1e-14)


def test_interior_duals_are_the_voronoi_cells():
    """On a Delaunay mesh the ASSEMBLED signed node volume of an interior node is its Voronoi cell volume
    and the assembled signed couple of an interior edge is the area of its Voronoi facet -- checked
    against Qhull (scipy.spatial.Voronoi: cell = ConvexHull of the region's vertices, facet = the ridge
    polygon), an independent construction. Budget 1e-10 relative (Qhull's own vertices are computed to
    ~1e-13; the polygon areas sum ~20 triangles). DEVSIM's unsigned assembly misses (asserted: some
    interior cell > 1.05 x its Voronoi volume)."""
    P, R = jittered_delaunay(4, 0.25, 3)
    T = R["r"]
    sg, us, L = G.tetrahedron_dual_pieces(P[T])
    vol_s, vol_u = np.zeros(len(P)), np.zeros(len(P))
    cpl_s, cpl_u = {}, {}
    for t, (a, b) in itertools.product(range(len(T)), [(k, e) for k, e in enumerate(G._TET_EDGES)]):
        i, j = T[t, b[0]], T[t, b[1]]
        key = (min(i, j), max(i, j))
        for (vol, cpl, c) in ((vol_s, cpl_s, sg), (vol_u, cpl_u, us)):
            vol[i] += c[t, a] * L[t, a] / 6.0
            vol[j] += c[t, a] * L[t, a] / 6.0
            cpl[key] = cpl.get(key, 0.0) + c[t, a]
    vor = scipy_spatial.Voronoi(P)
    interior = np.flatnonzero(np.all((P > 1e-9) & (P < 1 - 1e-9), axis=1))
    checked = 0
    worst_u = 1.0
    for i in interior:
        reg = vor.regions[vor.point_region[i]]
        if -1 in reg or not reg:
            continue
        vc = scipy_spatial.ConvexHull(vor.vertices[reg]).volume
        assert vol_s[i] == pytest.approx(vc, rel=1e-10)
        worst_u = max(worst_u, vol_u[i] / vc)
        checked += 1
    assert checked >= 20 and worst_u > 1.05
    nf = 0
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in rv or p not in interior or q not in interior:
            continue
        key = (min(p, q), max(p, q))
        V3 = vor.vertices[rv]
        cen = V3.mean(axis=0)
        area = 0.5 * sum(np.linalg.norm(np.cross(V3[k] - cen, V3[(k + 1) % len(V3)] - cen))
                         for k in range(len(V3)))           # ridge vertices come in polygon order
        if key not in cpl_s:                                # a degenerate (zero-area) ridge: no edge
            assert area < 1e-10
            continue
        assert cpl_s[key] == pytest.approx(area, rel=1e-9, abs=1e-12)
        nf += 1
    assert nf >= 50


# ================================================================================================
# DEVSIM: volumes, well-centred identity, patch test / closed forms
# ================================================================================================

def _volume_identities(dev, r, Vtrue):
    d = int(ds.get_dimension(device=dev))
    return (nv(dev, r).sum() / Vtrue, 2.0 * elv(dev, r, "ElementNodeVolume").sum() / Vtrue,
            (ev(dev, r, "EdgeCouple") * ev(dev, r, "EdgeLength")).sum() / d / Vtrue)


def test_devsim_volume_defect_and_exact_partition():
    """sum NodeVolume, 2 sum ElementNodeVolume and sum EdgeCouple L / d all equal the region volume after
    apply_exact_geometry, to <= 1e-13 (budget: <= 1000 pieces, each exact to ~4 eps of its own size).
    DEVSIM before (MEASURED, asserted): corner tet 2.000, one 5-tet cube 1.6667, 5-tet lattice 3^3
    1.6667, obtuse triangle 11.111, jittered 2-D Delaunay square 1.0519 (asserted > 1.01).
    Adversarial: the same identities with the UNSIGNED kernel output written instead fail."""
    cases = {
        "gv_corner": ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], {"r": np.array([[0, 1, 2, 3]])}, 2.0),
        "gv_cube5": (lambda: (five_tet_lattice(1)[0], {"r": five_tet_lattice(1)[1]}), None, 5.0 / 3.0),
        "gv_lat5": (lambda: (five_tet_lattice(3)[0], {"r": five_tet_lattice(3)[1]}), None, 5.0 / 3.0),
        "gv_obtuse": ([[0, 0], [2, 0], [1, 0.3]], {"r": np.array([[0, 1, 2]])}, 100.0 / 9.0),
        "gv_jit2": (lambda: jittered_delaunay(8, 0.3, 5, dim=2), None, None),
    }
    for name, (P, R, before) in cases.items():
        if callable(P):
            P, R = P()
        _device(name, P, R)
        Vt = _true_volume(P, R["r"]).sum()
        dv = _volume_identities(name, "r", Vt)
        if before is not None:
            assert dv[0] == pytest.approx(before, rel=1e-12)
        else:
            assert dv[0] > 1.01
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = G.apply_exact_geometry(name, "r", negative="warn")
        # a lone non-well-centred element is all boundary: the corner tet keeps 3 NEGATIVE assembled
        # couples (-0.0295 L^2), the obtuse triangle 1 couple (-0.758 L) and 2 node volumes (MEASURED)
        assert rep.has_negatives == (name in ("gv_corner", "gv_obtuse"))
        assert [x.category for x in w] == ([UnreliableResultWarning] if rep.has_negatives else [])
        assert rep.devsim_volume_ratio == pytest.approx(dv[0], rel=1e-14) and rep.applied
        for x in _volume_identities(name, "r", Vt):
            assert abs(x - 1.0) < 1e-13, (name, x)
        assert G.measure_dual_geometry(name, "r").devsim_volume_ratio == pytest.approx(1.0, abs=1e-13)
        assert G.exact_geometry_applied(name, "r")
    # adversarial: DEVSIM's unsigned values written back -> the identities fail again
    P, T = five_tet_lattice(1)
    R = G._Region("gv_cube5", "r")
    _sg, us, _L, _v = G._signed_slots(R)
    ds.set_element_values(device="gv_cube5", region="r", name="ElementEdgeCouple",
                          values=[float(v) for v in us.ravel()])
    assert _volume_identities("gv_cube5", "r", 1.0)[0] == pytest.approx(5.0 / 3.0, rel=1e-12)


def test_well_centred_meshes_unchanged_to_round_off():
    """On meshes where DEVSIM is already exact -- Kuhn (path) tets, rectilinear right triangles -- the
    correction moves every geometric value by round-off only (the zero pieces of diagonals / hypotenuses
    come out as +-few eps h^k, h the longest edge, k = d for volumes / d-1 for couples; budget 64 eps h^k)
    and the capacitor solve is the same to 1e-13 V (budget: that perturbation times the condition number, < 1e3 here).
    MEASURED: max geometry change 1.15 eps h^k (2-D) / 0.75 eps h^k (3-D), potential 2.1e-17 V."""
    out = {}
    P3, T3 = kuhn_lattice(3)
    for tag, (P, T) in {"3d": (P3 / 3.0, T3), "2d": (None, None)}.items():
        if tag == "2d":
            gx, gy = np.linspace(0, 1, 6), np.array([0, 0.1, 0.35, 0.5, 0.8, 1.0])
            P = np.array([[x, y] for y in gy for x in gx])
            T = []
            for j, i in itertools.product(range(5), range(5)):
                a, b, c, d = j * 6 + i, j * 6 + i + 1, (j + 1) * 6 + i + 1, (j + 1) * 6 + i
                T += [[a, b, d], [b, c, d]]
            T = np.array(T)
        res = []
        for k, apply in enumerate((False, True)):
            name = "wc_{}_{}".format(tag, k)
            _cap_device(name, P, {"r": T})
            geo0 = [nv(name, "r"), ev(name, "r", "EdgeCouple"), elv(name, "r", "ElementNodeVolume")]
            if apply:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    rep = G.apply_exact_geometry(name, "r")
                assert rep.n_not_well_centred == 0 and not rep.has_negatives
            geo = [nv(name, "r"), ev(name, "r", "EdgeCouple"), elv(name, "r", "ElementNodeVolume")]
            h = float(ev(name, "r", "EdgeLength").max())
            d = len(T[0]) - 1
            for a, b, p in zip(geo0, geo, (d, d - 1, d)):
                assert np.max(np.abs(a - b)) <= 64 * EPS * h ** p
                out.setdefault(tag + "_geo_eps", []).append(float(np.max(np.abs(a - b)) / (EPS * h ** p)))
            res.append((_capacitor_metrics(name, ["r"]), nv(name, "r", "Potential")))
            _drop_all()
        out[tag] = float(np.max(np.abs(res[0][1] - res[1][1])))
        assert out[tag] < 1e-13
        assert res[0][0][1] == pytest.approx(res[1][0][1], rel=1e-13)
    print("MEASURED well-centred: geometry change / (eps h^k), potential change [V]:", out)


@pytest.mark.parametrize("case", ["lat5", "jit3", "jit2", "two_region"])
def test_capacitor_patch_test_and_gauss(case):
    """Parallel-plate capacitor, eps = 1, plates at axis = 0 / 1, insulating side walls: the exact
    solution phi = axis coordinate is linear, so the box method with exact geometry reproduces it to
    round-off (patch test: every interior dual cell closes, and the side-wall nodes' open cells close on
    the wall), C = eps A / d = 1, and Gauss: |Q_lo + Q_hi| = rho V = 1. Budgets: 1e-11 (direct solve,
    cond(K) <~ 1e4 for these meshes, times the 1e-14 solver tolerance and the geometry round-off).

    DEVSIM's own geometry FAILS (MEASURED; asserted: one of the three off by > 1 %), as
    (max|phi - z|, C, Q): 5-tet lattice 3^3 (5.6e-17, 1.6667, 1.6667) -- phi stays exact there, the
    over-count scales every z-edge alike; jittered 3-D Delaunay (3.3e-2, 1.2553, 1.2640); jittered 2-D
    Delaunay (1.5e-2, 1.0389, 1.0519); two regions, interface z = 0.5 (3.2e-2, 1.2295, 1.2342).
    Corrected (warn): phi <= 1.1e-16, |C - 1| and |Q - 1| <= 2.2e-16.
    Adversarial: the EdgeCouple carrying the most flux scaled by 1.5 after the correction breaks the
    patch test."""
    if case == "lat5":
        P, T = five_tet_lattice(3)
        P = P / 3.0
        R = {"r": T}
    elif case == "jit3":
        P, R = jittered_delaunay(4, 0.3, 21)
    elif case == "jit2":
        P, R = jittered_delaunay(8, 0.3, 5, dim=2)
    else:
        P, R = jittered_delaunay(4, 0.3, 8, split=0.5)
    Vt = sum(_true_volume(P, T).sum() for T in R.values())
    assert Vt == pytest.approx(1.0, rel=1e-12)
    before = _cap_device("cp0_" + case, P, R)
    e0, C0, Q0 = _capacitor_metrics(before, list(R))
    _drop_all()
    assert max(e0, abs(C0 - 1.0), abs(Q0 - 1.0)) > 0.01           # DEVSIM uncorrected: shown failing
    name = _cap_device("cp1_" + case, P, R)
    for r in R:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = G.apply_exact_geometry(name, r, negative="warn")
        assert [x.category for x in w] == ([UnreliableResultWarning] if rep.has_negatives else [])
    e1, C1, Q1 = _capacitor_metrics(name, list(R))
    print("MEASURED {}: DEVSIM err {:.3e} C {:.6f} Q {:.6f} | exact err {:.3e} C-1 {:.2e} Q-1 {:.2e}".format(
        case, e0, C0, Q0, e1, C1 - 1, Q1 - 1))
    assert e1 < 1e-11 and abs(C1 - 1.0) < 1e-11 and abs(Q1 - 1.0) < 1e-11, (e1, C1, Q1)
    # adversarial: perturb one interior edge couple -> the linear field is no longer reproduced
    r = list(R)[0]
    ec = ev(name, r, "EdgeCouple")
    k = int(np.argmax(ec * np.abs(ev(name, r, "Potential@n0") - ev(name, r, "Potential@n1"))))
    ec[k] *= 1.5
    ds.set_edge_values(device=name, region=r, name="EdgeCouple", values=[float(v) for v in ec])
    e2, _C2, _Q2 = _capacitor_metrics(name, list(R))
    assert e2 > 1e-4


def test_surface_areas_3d_interface_and_contacts():
    """3-D SurfaceArea (interface triangles) and ContactSurfaceArea (contact triangles) are DEVSIM's
    unsigned circumcentric quarter-pieces (SurfaceAreaUtil.cc processTriangle): with obtuse surface
    triangles (tangentially jittered faces) their sums exceed the true areas (MEASURED: SurfaceArea
    1.0549, ContactSurfaceArea 1.0285 / 1.0681; asserted > 1.001). After apply_exact_geometry:
    sum SurfaceArea = interface area = 1 and sum ContactSurfaceArea = plate area = 1 per region (oracle:
    the cross-product area of the facets), to 1e-13.
    2-D SurfaceArea is half edge lengths (exact) and is not touched."""
    P, R = jittered_delaunay(4, 0.35, 8, split=0.5)
    name = _cap_device("sa3", P, R)
    true_if = sum(0.5 * np.linalg.norm(np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]]))
                  for f in set(_facets(R["a"])) & set(_facets(R["b"])))
    assert true_if == pytest.approx(1.0, rel=1e-12)
    before = {r: (nv(name, r, "SurfaceArea").sum(), nv(name, r, "ContactSurfaceArea").sum()) for r in R}
    assert min(min(v) for v in before.values()) > 1.001, before
    for r in R:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rep = G.apply_exact_geometry(name, r, negative="warn")
        assert rep.devsim_surface_ratio == pytest.approx(sum(before[r]) / 2.0, rel=1e-12)   # 1 + 1
        assert nv(name, r, "SurfaceArea").sum() == pytest.approx(1.0, abs=1e-13)
        assert nv(name, r, "ContactSurfaceArea").sum() == pytest.approx(1.0, abs=1e-13)
    print("MEASURED DEVSIM surface sums (SurfaceArea, ContactSurfaceArea):", before)


def test_drift_diffusion_resistor_closed_form():
    """Uniform n-type unipolar DD resistor (physics_drift_diffusion, Fermi-Dirac, SG currents), 100 nm
    cube, ohmic plates at z = 0 / 100 nm, insulating walls, 1 mV: n stays N_D and the potential linear,
    which the SG box scheme reproduces exactly when the geometry is exact, so the terminal current is
    G V with G = q N_D mu A / L. Budget 1e-8 relative (Newton stopped at relative update 1e-12; DEVSIM's
    contact current is the assembled edge flux). DEVSIM's own geometry: G off by the capacitor's factor
    (MEASURED 1.6667 on the 5-tet lattice, 1.2553 on the jittered Delaunay mesh; asserted > 1 %); corrected
    MEASURED 1 - 1.1e-16 / 1 + 2.2e-16."""
    from dynameta.carriers import physics_drift_diffusion as DD
    from dynameta.constants import M_E, Q_E
    nd, mu, v, Ls = 4e26, 0.004, 1e-3, 100e-9
    out = {}
    for case in ("lat5", "jit3"):
        if case == "lat5":
            P, T = five_tet_lattice(3)
            P = P / 3.0
        else:
            P, R = jittered_delaunay(4, 0.3, 21)
            T = R["r"]
        lo, hi = _plates(P, T)
        res = []
        for apply in (False, True):
            name = "dd_{}_{}".format(case, int(apply))
            _device(name, P * Ls, {"r": T}, {"slo": lo, "shi": hi}, [("lo", "slo", "r"), ("hi", "shi", "r")])
            if apply:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    G.apply_exact_geometry(name, "r", negative="warn")
            DD.setup_semiconductor_region_dd(name, "r", n_bg_m3=nd, eps_static=9.5, dos_mass_kg=0.35 * M_E,
                                             mobility_m2Vs=mu)
            DD.setup_contact_ohmic_dd(name, "lo")
            DD.setup_contact_ohmic_dd(name, "hi")
            ds.solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=30)
            ds.set_parameter(device=name, name="hi_bias", value=v)
            ds.solve(type="dc", absolute_error=1e10, relative_error=1e-12, maximum_iterations=30)
            I = abs(ds.get_contact_current(device=name, contact="hi", equation="ElectronContinuityEquation"))
            res.append(I / (Q_E * nd * mu * Ls * v))                  # G V = q n mu (A / L) V, A / L = Ls
            _drop_all()
        out[case] = res
        assert abs(res[0] - 1.0) > 0.01, res                         # DEVSIM uncorrected: shown failing
        assert abs(res[1] - 1.0) < 1e-8, res
    print("MEASURED G/G_exact (DEVSIM, corrected):", out)


def _random_square_2d(name, seed=1, n=60):
    rng = np.random.default_rng(seed)
    P = np.vstack([rng.uniform(0, 1, (n, 2)), [[0, 0], [1, 0], [0, 1], [1, 1]]])
    T = scipy_spatial.Delaunay(P).simplices
    _device(name, P, {"r": T})
    return P, T


def test_negative_policies_raise_warn_clip():
    """A random-point Delaunay square (its long convex-hull edges face obtuse angles) has negative
    ASSEMBLED couples and node volumes (MEASURED: 4 couples, worst -23.8 L; 4 node volumes, worst -6.65
    h^2; clip removes 0.284 of sum|EC| and adds 14.1 x the volume) -- the boundary caveat.
    'raise' leaves the device untouched; 'warn' (the default since the 2026-11-06 user decision) writes
    the exact signed geometry (volume exact) with UnreliableResultWarning; re-applying is idempotent; 'clip' leaves no negative EdgeCouple /
    NodeVolume, keeps 2 sum ENV exact, and its clipped fractions equal a recount from the exact values
    (a consistency check: the module is the only source of the exact values)."""
    P, T = _random_square_2d("np_r")
    eec0 = elv("np_r", "r", "ElementEdgeCouple")
    with pytest.raises(G.NegativeDualGeometryError) as ei:
        G.apply_exact_geometry("np_r", "r", negative="raise")
    rep = ei.value.report
    assert rep.n_negative_couples > 0 and rep.n_negative_volumes > 0 and not rep.applied
    assert np.array_equal(elv("np_r", "r", "ElementEdgeCouple"), eec0)
    assert not G.exact_geometry_applied("np_r", "r")
    Vt = _true_volume(P, T).sum()
    with pytest.warns(UnreliableResultWarning, match="NO M-matrix guarantee"):
        assert G.apply_exact_geometry("np_r", "r").policy == "warn"     # the default policy
    exact_nv = nv("np_r", "r")
    exact_ec = ev("np_r", "r", "EdgeCouple")
    L = ev("np_r", "r", "EdgeLength")
    assert exact_nv.sum() / Vt == pytest.approx(1.0, abs=1e-13)
    assert (exact_ec < -1e-9 * L).sum() == rep.n_negative_couples
    with pytest.warns(UnreliableResultWarning):
        G.apply_exact_geometry("np_r", "r", negative="warn")
    assert np.array_equal(nv("np_r", "r"), exact_nv)
    _random_square_2d("np_c")
    with pytest.warns(UnreliableResultWarning, match="Clipped to 0"):
        rc = G.apply_exact_geometry("np_c", "r", negative="clip")
    ec, nvc = ev("np_c", "r", "EdgeCouple"), nv("np_c", "r")
    assert np.all(ec >= -1e-9 * L) and np.all(nvc >= 0.0)
    assert 2.0 * elv("np_c", "r", "ElementNodeVolume").sum() / Vt == pytest.approx(1.0, abs=1e-13)
    neg_ec = exact_ec < -1e-9 * L
    assert rc.clipped_couple_fraction == pytest.approx(np.abs(exact_ec[neg_ec]).sum() / np.abs(exact_ec).sum(),
                                                       rel=1e-12)
    assert rc.clipped_volume_fraction == pytest.approx(-exact_nv[exact_nv < 0].sum() / Vt, rel=1e-12)
    assert nvc.sum() / Vt == pytest.approx(1.0 + rc.clipped_volume_fraction, rel=1e-12)
    print("MEASURED random square: {} neg couples (worst {:.3g} L), {} neg volumes (worst {:.3g} h^2); clip "
          "removes {:.3g} of sum|EC| and adds {:.3g} V".format(rep.n_negative_couples, rep.min_couple_rel,
                                                             rep.n_negative_volumes, rep.min_volume_rel,
                                                             rc.clipped_couple_fraction, rc.clipped_volume_fraction))


def test_refusals_and_one_d():
    """Cylindrical / user geometry models on the region are refused (ValueError, nothing written); an
    unknown policy is refused; a 1-D region is returned untouched (DEVSIM's 1-D geometry is exact)."""
    _device("rf_c", [[0, 0], [1, 0], [0.4, 0.9]], {"r": np.array([[0, 1, 2]])})
    with pytest.raises(ValueError, match="negative must be"):
        G.apply_exact_geometry("rf_c", "r", negative="ignore")
    eec0 = elv("rf_c", "r", "ElementEdgeCouple")
    ds.set_parameter(device="rf_c", region="r", name="node_volume_model", value="CylindricalNodeVolume")
    with pytest.raises(ValueError, match="cylindrical"):
        G.apply_exact_geometry("rf_c", "r")
    assert np.array_equal(elv("rf_c", "r", "ElementEdgeCouple"), eec0)
    ds.create_1d_mesh(mesh="rf_1")
    ds.add_1d_mesh_line(mesh="rf_1", pos=0.0, ps=0.1, tag="a")
    ds.add_1d_mesh_line(mesh="rf_1", pos=1.0, ps=0.3, tag="b")
    ds.add_1d_contact(mesh="rf_1", name="ca", tag="a", material="metal")
    ds.add_1d_contact(mesh="rf_1", name="cb", tag="b", material="metal")
    ds.add_1d_region(mesh="rf_1", material="X", region="r", tag1="a", tag2="b")
    ds.finalize_mesh(mesh="rf_1")
    ds.create_device(mesh="rf_1", device="rf_1")
    nv0 = nv("rf_1", "r")
    rep = G.apply_exact_geometry("rf_1", "r")
    assert rep.dimension == 1 and rep.applied and np.array_equal(nv("rf_1", "r"), nv0)
    assert rep.true_volume == pytest.approx(1.0, rel=1e-14)


def test_element_field_and_element_volume_consumers():
    """What element-based consumers see after the correction:
      * 2-D element_from_edge_model of the edge field of a P1 potential (TriangleElementField weights each
        edge's two node-based vectors by the OTHER edges' couples, now signed) is unchanged to round-off on
        an obtuse Delaunay mesh -- every node-based vector of a P1 field IS the element gradient, and the
        signed weight pairs stay > 0 (module doc);
      * 3-D 5-tet lattice: 2 sum ElementNodeVolume == the true volume -- the condition impact_ionization's
        3-D path (and the shared exact-volume check) requires before it integrates: it fails on DEVSIM's
        values (1.6667, the adversarial half) and passes after the correction."""
    P, R = jittered_delaunay(6, 0.3, 5, dim=2)
    fields = []
    for apply in (False, True):
        name = "ef_{}".format(int(apply))
        _device(name, P, R)
        if apply:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                G.apply_exact_geometry(name, "r", negative="warn")
        x, y = nv(name, "r", "x"), nv(name, "r", "y")
        ds.node_solution(device=name, region="r", name="Psi")
        ds.set_node_values(device=name, region="r", name="Psi", values=[float(v) for v in np.sin(3 * x) * y ** 2])
        ds.edge_from_node_model(device=name, region="r", node_model="Psi")
        ds.edge_model(device=name, region="r", name="EF", equation="(Psi@n0 - Psi@n1)*EdgeInverseLength")
        ds.element_from_edge_model(device=name, region="r", edge_model="EF")
        fields.append(np.stack([elv(name, "r", "EF_x"), elv(name, "r", "EF_y")]))
    assert np.max(np.abs(fields[1] - fields[0])) <= 1e-12 * np.max(np.abs(fields[0]))
    P, T = five_tet_lattice(2)
    for apply in (False, True):
        name = "ev_{}".format(int(apply))
        _device(name, P, {"r": T})
        if apply:
            G.apply_exact_geometry(name, "r")
        ratio = 2.0 * elv(name, "r", "ElementNodeVolume").sum() / _true_volume(P, T).sum()
        assert (abs(ratio - 1.0) < 1e-9) == apply, ratio            # impact_ionization's own condition


class _WriteRecorder:
    """Records every devsim call that is not a get_* read (create / set / delete / solve ...)."""

    def __init__(self, monkeypatch):
        import devsim as _dsm
        self.calls = []
        for nm in dir(_dsm):
            fn = getattr(_dsm, nm)
            if nm.startswith("_") or nm.startswith("get_") or not callable(fn) or isinstance(fn, type):
                continue
            monkeypatch.setattr(_dsm, nm, self._wrap(nm, fn))

    def _wrap(self, nm, fn):
        def inner(*a, **kw):
            self.calls.append(nm)
            return fn(*a, **kw)
        return inner


def test_already_exact_meshes_are_only_read(monkeypatch):
    """BYTE-IDENTITY of the library meshes under the automatic correction: on the 3-D Kuhn lattice (with
    contacts, so the surface shares are checked too) and the 2-D rectilinear right-triangle mesh,
    apply_exact_geometry makes NO devsim call other than get_* reads (every other devsim function is wrapped
    and recorded) and returns written=False, so DEVSIM's state is untouched. Adversarial: the same recorder
    on the 5-tet lattice sees the write (set_element_values) and written=True."""
    P, T = kuhn_lattice(2)
    P = P / 2.0
    lo, hi = _plates(P, T)
    _device("ro_k", P, {"r": T}, {"slo": lo, "shi": hi}, [("lo", "slo", "r"), ("hi", "shi", "r")])
    gx, gy = np.linspace(0, 1, 5), np.array([0, 0.2, 0.3, 0.7, 1.0])
    P2 = np.array([[x, y] for y in gy for x in gx])
    T2 = np.array([t for j, i in itertools.product(range(4), range(4))
                   for t in ([j * 5 + i, j * 5 + i + 1, (j + 1) * 5 + i],
                             [j * 5 + i + 1, (j + 1) * 5 + i + 1, (j + 1) * 5 + i])])
    _device("ro_2", P2, {"r": T2})
    P5, T5 = five_tet_lattice(1)
    _device("ro_5", P5, {"r": T5})
    rec = _WriteRecorder(monkeypatch)
    for name in ("ro_k", "ro_2"):
        assert G.geometry_already_exact(name, "r")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rep = G.apply_exact_geometry(name, "r", negative="warn")
        assert rep.applied and not rep.written and rep.devsim_volume_ratio == pytest.approx(1.0, abs=1e-13)
    assert rec.calls == []
    assert not G.exact_geometry_applied("ro_k", "r")
    assert not G.geometry_already_exact("ro_5", "r")
    rep = G.apply_exact_geometry("ro_5", "r", negative="warn")
    assert rep.written and "set_element_values" in rec.calls


def test_correct_device_geometry_policies():
    """The builders' device-level entry: exact_geometry=True means 'warn' and merges the per-region
    negative-couple warnings into ONE UnreliableResultWarning; 'raise' measures every region first and
    leaves the WHOLE device unchanged on a refusal; False does nothing; anything else is refused."""
    assert G.resolve_policy(True) == "warn" and G.resolve_policy(False) is None
    assert G.resolve_policy(None) is None and G.resolve_policy("clip") == "clip"
    with pytest.raises(ValueError, match="exact_geometry must be"):
        G.resolve_policy("yes")
    P, R = jittered_delaunay(4, 0.3, 8, split=0.5)               # region b has a negative couple
    _device("cd_r", P, R)
    eec = {r: elv("cd_r", r, "ElementEdgeCouple") for r in R}
    with pytest.raises(G.NegativeDualGeometryError):
        G.correct_device_geometry("cd_r", "raise")
    assert all(np.array_equal(elv("cd_r", r, "ElementEdgeCouple"), eec[r]) for r in R)
    assert G.correct_device_geometry("cd_r", False) == {}
    with pytest.warns(UnreliableResultWarning) as w:
        reps = G.correct_device_geometry("cd_r")
    assert len(w) == 1 and "region 'b'" in str(w[0].message)
    assert set(reps) == set(R) and all(rp.written for rp in reps.values())
    for r, T in R.items():
        assert nv("cd_r", r).sum() == pytest.approx(_true_volume(P, T).sum(), rel=1e-13)
