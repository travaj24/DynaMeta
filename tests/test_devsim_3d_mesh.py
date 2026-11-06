"""D2c (v0.14.0): the devsim_3d tensor (Kuhn) mesh gives DEVSIM exact box-method geometry.

The pre-D2c gmsh mesh had 42-57 % non-well-centred tetrahedra; DEVSIM 2.11 sums the MAGNITUDES of each
tet's circumcentric dual pieces, so sum NodeVolume over-counted (1.196 x semi, 1.293 x oxide) and
integrated / sheet charge was ~20 % wrong. The heavy oracles (five topologies, the D1c trap gate G,
the e46bd25 failure table) are validation/devsim_3d_mesh.py; docs/audit/2026-11-03-devsim-3d-mesh.md."""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("devsim")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers import devsim_3d as D3                                       # noqa: E402
from dynameta.carriers import physics_traps as PT                                   # noqa: E402
from dynameta.warning_types import UnreliableResultWarning                          # noqa: E402
import validation.devsim_3d_mesh as V                                                # noqa: E402


def test_graded_axis_hits_every_break_and_follows_the_threshold_rule():
    z = D3._graded_axis([0.0, 30.0, 38.0], [30.0], 0.5, 3.0)
    assert z[0] == 0.0 and z[-1] == 38.0 and np.any(z == 30.0)
    h = np.diff(z)
    mid = 0.5 * (z[1:] + z[:-1])
    assert np.all(h > 0.0) and h.max() <= 3.0 + 1e-9
    assert np.all(h[np.abs(mid - 30.0) < 0.5] <= 0.5 + 1e-6)         # the gmsh SizeMin band (DistMin 1 nm)
    u = D3._graded_axis([0.0, 12.0], [], 0.5, 3.0)                   # no refinement: uniform <= h_max
    assert np.allclose(np.diff(u), 3.0)


def _all_path_tets(P, T, rtol=1e-12):
    """True iff every tet is a PATH ("Kuhn") tet: its vertices, ordered by x + y + z, step along three
    DISTINCT coordinate axes, each step a positive move along exactly one axis. On such a tet every
    circumcentric dual piece is >= 0, so DEVSIM 2.11's sum of |piece| (TetrahedronEdgeCouple.cc:99) is the
    exact signed sum. Being right-angled is not enough: the trirectangular corner tet (0,0,0), (1,0,0),
    (0,1,0), (0,0,1) is NOT a path tet and DEVSIM gives it exactly 2x its volume (verification 2026-11-04)."""
    X = np.asarray(P, dtype=float)[np.asarray(T, dtype=int)]               # (m, 4, 3)
    X = np.take_along_axis(X, np.argsort(X.sum(axis=2), axis=1)[..., None], axis=1)
    d = np.diff(X, axis=1)                                                 # (m, 3, 3) steps
    scale = np.abs(X).max() + 1e-300
    nz = np.abs(d) > rtol * scale
    one_axis = nz.sum(axis=2) == 1
    positive = np.all(np.where(nz, d > 0, True), axis=(1, 2))
    axes = np.argmax(nz, axis=2)
    distinct = (axes[:, 0] != axes[:, 1]) & (axes[:, 0] != axes[:, 2]) & (axes[:, 1] != axes[:, 2])
    return bool(np.all(one_axis.all(axis=1) & positive & distinct))


def test_path_tet_checker_accepts_kuhn_and_rejects_the_corner_tet():
    cube = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=float)
    kuhn = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    assert _all_path_tets(cube, kuhn)
    corner = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert not _all_path_tets(corner, [(0, 1, 2, 3)])                     # right-angled, NOT a path tet
    assert not _all_path_tets(cube, [(0, 1, 2, 4)])                       # the 5-tet split's corner tet


def test_tensor_mesher_emits_only_path_tets_and_the_2d_builder_only_non_obtuse_triangles():
    """Exactness pins (coordinator scope (b), 2026-11-04) so a future change to the splitting cannot
    silently break the box method. DEVSIM sums UNSIGNED circumcentric pieces in 3-D AND 2-D
    (TetrahedronEdgeCouple.cc:99, TriangleEdgeCouple.cc:74), so sum(NodeVolume) equals the geometric
    volume only when no piece is negative.
    3-D: every tet the tensor mesher emits, over the five validation topologies (full gate, patch,
    multi-dielectric, buried-oxide inclusion, semi inclusion + patch), is a path tet.
    2-D: LayeredDevsimBuilder's graded rectilinear mesh (the D1a GaAs / AlGaAs stack, 0.5 -> 4 nm grading)
    has only non-obtuse triangles (DEVSIM splits each rectangle into two right triangles: all pieces >= 0)
    and sum(NodeVolume) == the geometric area of every region to round-off (MEASURED |ratio - 1| <= 2.2e-16
    on all four regions, 2026-11-04)."""
    import devsim as ds
    from dynameta.carriers.devsim_3d import Inclusion3D
    hfo2 = [("HfO2", 4e-9, 18.0)]
    cases = [{}, dict(gate_patch_frac=0.5), dict(extra_dielectrics=hfo2),
             dict(extra_dielectrics=hfo2, inclusions=[Inclusion3D(name="pill", material="HfO2", eps=18.0,
                                                                  in_layer="oxide", x_frac=0.5, y_frac=0.5)]),
             dict(gate_patch_frac=0.6, inclusions=[Inclusion3D(name="sx", material="SiO2", eps=3.9,
                                                               in_layer="semi", x_frac=0.3, y_frac=0.5)])]
    for kw in cases:
        sol = D3.Devsim3DEquilibrium(V._spec(physics="drift_diffusion", **kw), device_name="t3x",
                                     mesh_name="t3xm")
        sol._build_mesh()                                 # the default (in-memory) route: no file
        P, groups3, _tri = sol._mesh_data
        for reg, T in groups3.items():
            assert len(T) and _all_path_tets(P, T), (kw, reg)
    # 2-D: the library's own rectilinear builder
    from validation.devsim_heterojunction import ALGAAS, GAAS, stack_design, transport
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    b = LayeredDevsimBuilder(stack_design([("l0", "M0", 200e-9, transport(GAAS, n_bg=1e23, physics="drift_diffusion")),
                                           ("l1", "M1", 200e-9, transport(ALGAAS, n_bg=1e23, physics="drift_diffusion"))]),
                             mesh_name="t2xm", device_name="t2x")
    b.build_device()
    try:
        for reg in ds.get_region_list(device="t2x"):
            Q = np.column_stack([np.asarray(ds.get_node_model_values(device="t2x", region=reg, name=c))
                                 for c in ("x", "y")])
            tri = np.asarray([list(e) for e in ds.get_element_node_list(device="t2x", region=reg)], dtype=int)
            A = Q[tri]
            e = [A[:, (k + 1) % 3] - A[:, k] for k in range(3)]
            area = 0.5 * np.abs(e[0][:, 0] * e[1][:, 1] - e[0][:, 1] * e[1][:, 0])
            # non-obtuse: at every vertex the two outgoing edges have a non-negative dot product
            dots = [-(e[k] * e[(k + 2) % 3]).sum(axis=1) for k in range(3)]
            lmax = max(float(np.max(np.sum(x * x, axis=1))) for x in e)
            assert min(float(np.min(dk)) for dk in dots) >= -1e-12 * lmax, reg
            nv = float(np.sum(ds.get_node_model_values(device="t2x", region=reg, name="NodeVolume")))
            assert abs(nv / float(area.sum()) - 1.0) < 1e-12, (reg, nv / float(area.sum()))
    finally:
        b.teardown()


def test_kuhn_tets_conform_tile_the_box_and_are_closed_well_centred():
    xs, ys, zs = np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 2.5]), np.array([0.0, 0.5, 2.0, 4.0])
    P, T, cell = D3._kuhn_tets(xs, ys, zs)
    X = P[T]
    D = X[:, 1:] - X[:, :1]
    vol = np.linalg.det(D) / 6.0
    assert np.all(np.abs(vol) > 0.0) and np.isclose(np.abs(vol).sum(), 3.0 * 2.5 * 4.0, rtol=1e-14)
    faces = {}
    for t in T:
        for o in range(4):
            f = tuple(sorted(int(t[m]) for m in range(4) if m != o))
            faces[f] = faces.get(f, 0) + 1
    assert max(faces.values()) == 2                                   # conforming: no face shared 3 times
    # circumcentre of every tet on or inside it (barycentric >= 0). NECESSARY but NOT SUFFICIENT for
    # DEVSIM's |piece| sum to equal the signed (exact) one (correction 2026-11-04, verification files in
    # docs/audit/2026-11-03-devsim-3d-mesh.md section 8): an inside-circumcentre tet with an OBTUSE face
    # still over-counts 4.53x. What makes these tets exact is that they are PATH (Kuhn) tets -- pinned
    # by test_tensor_mesher_emits_only_path_tets_and_the_2d_builder_only_non_obtuse_triangles.
    cc = np.linalg.solve(D, 0.5 * np.einsum("mid,mid->mi", D, D)[..., None])[..., 0]
    lam = np.linalg.solve(np.transpose(D, (0, 2, 1)), cc[..., None])[..., 0]
    lam = np.concatenate([1.0 - lam.sum(1, keepdims=True), lam], axis=1)
    assert lam.min() > -1e-12
    assert _all_path_tets(P, T)


def test_triangle_node_areas_circumcentric_for_right_thirds_for_obtuse():
    r = PT._triangle_node_areas(np.array([[0.0, 0, 0], [2.0, 0, 0], [2.0, 3.0, 0]]))
    assert np.allclose(r, [0.75, 1.5, 0.75])                          # ab/8, ab/4, ab/8: the box quarters
    o = PT._triangle_node_areas(np.array([[0.0, 0, 0], [4.0, 0, 0], [2.0, 0.5, 0]]))
    assert np.allclose(o, 1.0 / 3.0)                                  # obtuse -> thirds, total kept


def test_spec_validates_mesher_and_sizes():
    with pytest.raises(ValueError):
        V._spec(mesher="delaunay")
    with pytest.raises(ValueError):
        V._spec(mesh_min_nm=4.0, mesh_max_nm=3.0)
    assert V._spec().mesher == "tensor"                              # the D2c default


def test_volumes_couplings_and_interface_areas_exact_on_the_builder(tmp_path):
    import devsim as ds
    spec = V._spec(physics="drift_diffusion", gate_patch_frac=0.5)
    sol = D3.Devsim3DEquilibrium(spec, device_name="t3m", mesh_name="t3mm", msh_path=str(tmp_path / "t.msh"))
    sol.build_device()
    try:
        for r, thk in (("semi", V.TSI), ("oxide", V.TOX)):
            vol = V.L * V.L * thk
            nv = V._v("t3m", r, "NodeVolume")
            ec = np.asarray(ds.get_edge_model_values(device="t3m", region=r, name="EdgeCouple"))
            el = np.asarray(ds.get_edge_model_values(device="t3m", region=r, name="EdgeLength"))
            env = np.asarray(ds.get_element_model_values(device="t3m", region=r, name="ElementNodeVolume"))
            assert abs(nv.sum() / vol - 1) < 1e-12 and abs(2 * env.sum() / vol - 1) < 1e-12
            assert abs((ec * el).sum() / (3 * vol) - 1) < 1e-12 and ec.min() >= 0.0
        a = PT.interface_node_area("t3m", "semi", "if_1")
        assert abs(a.sum() / V.L ** 2 - 1) < 1e-12
        sa = V._v("t3m", "semi", "SurfaceArea")
        top = V._v("t3m", "semi", "z") >= V.TSI - 1e-15                # SurfaceArea also holds the body face
        assert np.allclose(a[top], sa[top], rtol=1e-12, atol=0.0)    # the FV control-volume face, node by node
        ng = ds.get_contact_list(device="t3m")
        assert set(ng) == {"gate", "body"}
    finally:
        sol.teardown()


def test_carrier_free_sheet_charge_equals_the_series_capacitor_closed_form():
    r = V.sheet_charge()                                             # pre-D2c: mean 0.827, worst node 0.19
    assert r["worst_rel"] < 1e-9 and abs(r["total_area_rel"]) < 1e-12


def test_gauss_balance_and_lateral_invariance_of_the_bipolar_cap():
    r = V.gauss()                                                    # pre-D2c: 0.23 / 0.23 / 5.0e-3 V
    assert r["gauss_rel"] < 1e-6 and r["gate_charge_rel"] < 1e-6
    assert r["psi_s_spread_V"] < 1e-9


# ---- task 7 (user decision 2026-11-04): no shared default mesh file ---------------------------------
_RACE_CHILD = r"""
import os, sys, tempfile, time
sys.path.insert(0, sys.argv[1])
from dynameta.carriers.devsim_env import configure_math_libs, import_devsim
configure_math_libs()
ds = import_devsim()
from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
from dynameta.constants import M_E
nz, iters, go = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
spec = Stacked3DSpec(semi_material="Si", oxide_material="SiO2", lateral_m=4e-9, semi_thk_m=8e-9,
                     oxide_thk_m=2e-9, n_bg_m3=1e23, eps_semi=11.7, eps_oxide=3.9, dos_mass_kg=1.08 * M_E,
                     mobility_m2Vs=0.135, grid_n=(4, 4, nz), mesh_min_nm=0.25 + 0.05 * (nz % 7), mesh_max_nm=1.0)
def build(sol):                       # build_device's mesh/device step without the region physics
    if hasattr(sol, "_create_mesh_device"):
        return sol._create_mesh_device()
    sol._build_mesh()                 # 6595bde: the same first lines of build_device, verbatim
    ds.create_gmsh_mesh(mesh=sol.mesh_name, file=sol.msh_path)
    for (name, mat, _r, _e) in sol._mesh_regions:
        ds.add_gmsh_region(mesh=sol.mesh_name, gmsh_name=name, region=name, material=mat)
    ds.add_gmsh_contact(mesh=sol.mesh_name, gmsh_name="gate", region=sol._top_diel, name="gate", material="metal")
    ds.add_gmsh_contact(mesh=sol.mesh_name, gmsh_name="body", region="semi", name="body", material="metal")
    for (ifn, r0, r1) in sol._iface_pairs:
        ds.add_gmsh_interface(mesh=sol.mesh_name, gmsh_name=ifn, region0=r0, region1=r1, name=ifn)
    ds.finalize_mesh(mesh=sol.mesh_name)
    ds.create_device(mesh=sol.mesh_name, device=sol.device)
def nodes_and_clear():
    n = sum(len(ds.get_node_model_values(device=d, region=r, name="x"))
            for d in ds.get_device_list() for r in ds.get_region_list(device=d))
    for d in list(ds.get_device_list()):
        ds.delete_device(device=d)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    return n
build(Devsim3DEquilibrium(spec, mesh_name="m", device_name="d",
                          msh_path=os.path.join(tempfile.mkdtemp(), "own.msh")))
expected = nodes_and_clear()
while not os.path.exists(go):
    time.sleep(0.01)
bad = 0
for i in range(iters):
    try:
        build(Devsim3DEquilibrium(spec, mesh_name="m{}".format(i), device_name="d{}".format(i)))
        bad += nodes_and_clear() != expected
    except Exception:
        bad += 1
        nodes_and_clear()
print("BAD", bad, "OF", iters, "EXPECTED", expected, flush=True)
"""


def test_two_processes_building_default_path_3d_devices_get_their_own_meshes(tmp_path):
    """The default msh_path was ONE fixed file (~/.dynameta/_devsim3d.msh) for every process. MEASURED
    on 6595bde (this harness, 100 builds per process): 1 build read the OTHER process's mesh and 199/200
    died with 'DEVSIM FATAL'; with 25 builds each this gate fails there. Now the default tensor mesh goes
    to DEVSIM in memory, so two concurrent processes with different devices both get their own mesh in
    every build (MEASURED 0/100 + 0/100)."""
    import subprocess
    import time
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child = tmp_path / "race_child.py"
    child.write_text(_RACE_CHILD, encoding="utf-8")
    go = tmp_path / "go.flag"
    procs = [subprocess.Popen([sys.executable, str(child), repo, str(nz), "25", str(go)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for nz in (21, 40)]
    time.sleep(1.0)
    go.write_text("go")
    outs = [p.communicate(timeout=600)[0] for p in procs]
    for out in outs:
        line = [ln for ln in out.splitlines() if ln.startswith("BAD ")]
        assert line and line[-1].startswith("BAD 0 OF 25"), out[-2000:]


@pytest.mark.devsim_isolated("bitwise comparison of separate DEVSIM solves: a device left live by another test is iterated by the same joint Newton and moves the answer by ulps")
def test_default_mesh_path_is_in_memory_and_bit_identical_to_the_file_route(tmp_path, monkeypatch):
    """Default msh_path: the tensor mesh reaches DEVSIM in memory (create_gmsh_mesh(coordinates=,
    elements=, physical_names=)) -- no file is written or read -- and the device is BIT-IDENTICAL to the
    explicit-file route (node coordinates, NodeVolume, EdgeCouple, the solved Potential; MEASURED 12 arrays
    / 5864 values array_equal). An explicit msh_path still writes and keeps its file. (The no-file check
    spies on the writer and on create_gmsh_mesh's arguments rather than on ~/.dynameta/_devsim3d.msh's
    mtime: other checkouts on a shared box still write that file -- the first version of this test
    failed exactly that way in a full run, 2026-11-04.)"""
    import devsim as ds
    from dynameta.sweep import BiasPoint
    calls = {"write": 0, "file": []}
    real_write, real_create = D3._write_msh22, D3.ds.create_gmsh_mesh

    def spy_write(*a, **k):
        calls["write"] += 1
        return real_write(*a, **k)

    def spy_create(**k):
        calls["file"].append(k.get("file"))
        return real_create(**k)
    monkeypatch.setattr(D3, "_write_msh22", spy_write)
    monkeypatch.setattr(D3.ds, "create_gmsh_mesh", spy_create)
    got = []
    for tag, kw in (("f", dict(msh_path=str(tmp_path / "f.msh"))), ("m", {})):
        sol = D3.Devsim3DEquilibrium(V._spec(physics="drift_diffusion"), device_name="t3p" + tag,
                                     mesh_name="t3pm" + tag, **kw)
        n_write = calls["write"]
        sol.build_device()
        try:
            assert calls["file"][-1] == kw.get("msh_path")              # None: the in-memory route
            assert calls["write"] - n_write == (1 if kw else 0)          # no file written by default
            sol.solve(BiasPoint(label="g", voltages={"gate": 0.5}))
            dev = "t3p" + tag
            got.append({(r, nm): np.asarray(ds.get_node_model_values(device=dev, region=r, name=nm))
                         for r in ds.get_region_list(device=dev)
                         for nm in ("x", "y", "z", "NodeVolume", "Potential")})
            got[-1].update({(r, "EdgeCouple"): np.asarray(ds.get_edge_model_values(device=dev, region=r,
                                                                                    name="EdgeCouple"))
                            for r in ds.get_region_list(device=dev)})
            assert sol._last_msh_path == (kw.get("msh_path"))           # None: no file for the default
        finally:
            sol.teardown()
    assert (tmp_path / "f.msh").exists()                                  # explicit path: written + kept
    assert got[0].keys() == got[1].keys()
    assert all(np.array_equal(got[0][k], got[1][k]) for k in got[0])


def test_legacy_gmsh_mesher_keeps_the_defect_and_warns(tmp_path):
    spec = V._spec(physics="drift_diffusion", mesher="gmsh")
    sol = D3.Devsim3DEquilibrium(spec, device_name="t3g", mesh_name="t3gm", msh_path=str(tmp_path / "g.msh"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol.build_device()
    try:
        msgs = [str(x.message) for x in w if issubclass(x.category, UnreliableResultWarning)]
        assert msgs
        # user decision 2026-11-04: the warning states the legacy mesher's fate (fails on 6595bde)
        assert any("pre-v0.14" in m and "REMOVED in v0.15" in m for m in msgs), msgs
        assert V._v("t3g", "semi", "NodeVolume").sum() / (V.L * V.L * V.TSI) > 1.1
    finally:
        sol.teardown()


# ---- 3-D element-based impact ionization (D2c optional; impact_ionization._impact_generation_elements_3d)
NI, MU_N, MU_P, TAU = 1e16, 0.135, 0.048, 1e-7


def _bipolar_column(dev, z_lines, doping, lateral_m=None, tau=TAU):
    """A bipolar Si column between ohmic contacts 'bot' (z = 0) and 'top' (z = z_lines[-1]): 1-D when
    lateral_m is None, else a lateral_m^2 3-D Kuhn-tensor bar (2 x 2 cells laterally) on the SAME z-lines.
    doping: a DEVSIM expression in the variable `z`."""
    import devsim as ds
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.constants import M_E
    V._clear()
    m = dev + "_m"
    zl = np.asarray(z_lines, dtype=float)
    if lateral_m is None:
        ds.create_1d_mesh(mesh=m)
        for k, zk in enumerate(zl):
            tag = {0: "bot", len(zl) - 1: "top"}.get(k)
            ds.add_1d_mesh_line(mesh=m, pos=float(zk), ps=1.0, **({"tag": tag} if tag else {}))
        ds.add_1d_contact(mesh=m, name="bot", tag="bot", material="metal")
        ds.add_1d_contact(mesh=m, name="top", tag="top", material="metal")
        ds.add_1d_region(mesh=m, material="Si", region="si", tag1="bot", tag2="top")
        ds.finalize_mesh(mesh=m)
        coord = "x"
    else:
        lat = np.linspace(0.0, lateral_m, 3)
        ax = (lat, lat, zl)
        P, T, _c = D3._kuhn_tets(*ax)
        rid = np.zeros((2, 2, len(zl) - 1), dtype=int)
        tri = {"bot": D3._face_triangles(ax, 2, D3._cell_faces(rid, 2, "bottom", 0), plus=False),
               "top": D3._face_triangles(ax, 2, D3._cell_faces(rid, 2, "top", 0), plus=True)}
        D3.create_mesh_in_memory(m, P, {"si": T}, tri)      # no fixed ~/.dynameta file (2026-11-04)
        ds.add_gmsh_region(mesh=m, gmsh_name="si", region="si", material="Si")
        for c in ("bot", "top"):
            ds.add_gmsh_contact(mesh=m, gmsh_name=c, region="si", name=c, material="metal")
        ds.finalize_mesh(mesh=m)
        coord = "z"
    ds.create_device(mesh=m, device=dev)
    ds.node_model(device=dev, region="si", name="NetDoping", equation=doping.replace("z", coord))
    nc = PE.effective_dos_m3(1.08 * M_E)
    BP.setup_bipolar_region(dev, "si", eps_static=11.7, n_dos_m3=nc, n_i_m3=NI, mobility_n_m2Vs=MU_N,
                            mobility_p_m2Vs=MU_P, tau_n_s=tau, tau_p_s=tau, fd_enhancement=True, n_dos_p_m3=nc)
    BP.setup_equilibrium_seed_models(dev, "si")
    for c in ("bot", "top"):
        BP.setup_contact_ohmic_bipolar(dev, c)
    ds.node_model(device=dev, region="si", name="_seed",
                  equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "si")))
    for var, mod in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region="si", name=var, values=V._v(dev, "si", mod).tolist())


def _ramp(dev, contact, v_end, steps):
    import devsim as ds
    for k in range(steps + 1):
        ds.set_parameter(device=dev, name=contact + "_bias", value=v_end * k / steps)
        ds.solve(type="dc", absolute_error=1e10, relative_error=1e-11, maximum_iterations=60)


def test_3d_impact_ionization_uniform_field_bar_matches_alpha_E_J_volume():
    from dynameta.carriers.impact_ionization import ChynowethParams, substrate_current
    nd, length, v_bar, lat = 1e22, 100e-9, 0.1, 20e-9
    pars = ChynowethParams(a_n_per_m=1.0e6, b_n_V_per_m=2.0e5, a_p_per_m=0.0, b_p_V_per_m=1.0)
    _bipolar_column("ii3b", np.linspace(0.0, length, 21), repr(nd), lateral_m=lat)
    try:
        _ramp("ii3b", "top", v_bar, 2)
        i3 = substrate_current("ii3b", "si", pars)               # [A]: the 3-D mesh is in metres
    finally:
        V._clear()
    q = 1.602176634e-19
    e = v_bar / length
    i_cf = pars.alpha_n(e) * q * nd * MU_N * e * (lat * lat * length)
    assert abs(i3 / i_cf - 1.0) < 1e-6, (i3, i_cf)


def test_3d_impact_ionization_equals_1d_on_a_laterally_uniform_reverse_diode():
    import devsim as ds
    from dynameta.carriers.impact_ionization import SILICON_VANOVERSTRAETEN, substrate_current
    length, lat = 200e-9, 20e-9
    z = D3._graded_axis([0.0, 100.0, 200.0], [100.0], 1.0, 8.0, dist_min=30.0, dist_max=80.0) * 1e-9
    dop = "1e24 * (2 * step(z - 100e-9) - 1)"                     # p (bot) | n (top), abrupt at 100 nm
    res = {}
    for name, lm in (("1d", None), ("3d", lat)):
        _bipolar_column("ii_" + name, z, dop, lateral_m=lm, tau=1e-10)   # short tau: a well-conditioned J
        try:
            _ramp("ii_" + name, "top", 3.0, 30)                    # n side up: reverse bias
            res[name] = substrate_current("ii_" + name, "si", SILICON_VANOVERSTRAETEN)
            res[name + "_I"] = ds.get_contact_current(device="ii_" + name, contact="top",
                                                      equation="ElectronContinuityEquation")
        finally:
            V._clear()
    per_area_3d = res["3d"] / (lat * lat)
    # MEASURED: I_sub/A 0.1556608697 A/m^2 both, rel 5e-10; contact J 0.42584 A/m^2, rel 2e-16 (the 3-D and
    # 1-D solutions are the same discrete system). At tau = 1e-7 the net J is a ~1e-6 residue of the SG
    # drift/diffusion terms and the element reconstruction's round-off shows (rel 1.4e-6) -- hence tau 1e-10.
    assert abs(res["3d_I"] / (lat * lat) / res["1d_I"] - 1.0) < 1e-12
    assert res["1d"] > 0.0
    assert abs(per_area_3d / res["1d"] - 1.0) < 1e-8, (per_area_3d, res["1d"])


def test_3d_impact_ionization_refuses_an_inexact_mesh(tmp_path):
    from dynameta.carriers.impact_ionization import SILICON_VANOVERSTRAETEN, substrate_current
    spec = V._spec(physics="drift_diffusion", mesher="gmsh")
    sol = D3.Devsim3DEquilibrium(spec, device_name="t3r", mesh_name="t3rm", msh_path=str(tmp_path / "r.msh"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol.build_device()
    try:
        with pytest.raises(NotImplementedError, match="mesher='tensor'"):
            substrate_current("t3r", "semi", SILICON_VANOVERSTRAETEN)
    finally:
        sol.teardown()
