"""The ONE exact-element-volume check (dynameta.carriers.element_geometry, v0.14.0, consolidated plan row
3.2-13) and its three consumers: impact ionization (2-D / 3-D element quadrature), BTBT (setup_btbt, 2-D /
3-D element assembly) and the Hooge 1/f source (noise_analysis, per-element assembly).

WHY. DEVSIM 2.11.2 builds ElementNodeVolume (ENV) from the MAGNITUDES of the circumcentric dual pieces
(TriangleEdgeCouple.cc:74, TetrahedronEdgeCouple.cc:99), so 2 sum ENV over an element equals its volume
only when no piece is negative (a well-centred / closed element). Before this check only the 3-D impact
ionization refused such a mesh (a region-total check); 2-D impact ionization, 2-D/3-D BTBT and the Hooge
source integrated with the over-counted weights silently (the gmsh-tet bar's Hooge read 1.055).

ORACLES (independent of DEVSIM and of the module under test):
  * true simplex volumes |det|/d! from the node coordinates (numpy, here);
  * DEVSIM's over-count re-derived here from the geometry: per triangle edge the unsigned dual piece
    |m - c| L / 2 (m the edge midpoint, c the circumcentre), so the ratio is sum |s_e| L_e / sum s_e L_e
    (repro_2d.py construction); the trirectangular corner tet is exactly 2x (verification 2026-11-04).

GATES
  E1  library meshes pass with max |ratio - 1| <= 1e-13 (MEASURED 6.7e-15 2-D graded tensor, 1.1e-14 3-D
      Kuhn graded): k = 3 / 6 pieces per element at eps each. Adversarial: moving ONE interior node of
      the 2-D tensor triangulation by 0.3 h makes obtuse triangles and the check must refuse.
  E2  inexact meshes are measured exactly (vs the unsigned re-derivation, 1e-12) and refused with
      InexactElementGeometryError (a NotImplementedError); raise_error=False only measures.
  E3  consumers: setup_btbt refuses the 2-D obtuse pair and the 3-D 5-tet cube BEFORE creating any model,
      and still installs on the exact meshes; impact ionization refuses both; the Hooge source refuses the
      gmsh-tet bar (MEASURED 2 sum ENV = 1.2249x its volume, 544 of 932 tets
      inexact, worst 3.81x) while the same bar's diffusion noise (edge-assembled, not
      an element quantity) is still computed.
FAILS ON THE OLD CODE: dynameta.carriers.element_geometry does not exist (ImportError) and, measured on
3ba3a2a, setup_btbt installs on the 2-D obtuse mesh and on the 5-tet cube, the 2-D impact ionization
integrates the obtuse mesh, and noise_spectrum(hooge_alpha=...) returns the biased Hooge spectrum on the
gmsh bar (S f N / (alpha I^2) = 1.0546) -- only the 3-D impact ionization refused (its old region-total
check, now this one) (docs/audit/2026-11-06-v0.14-impact-ionization-2d.md).
"""
import contextlib
import itertools
import os
import sys

import numpy as np
import pytest

pytest.importorskip("devsim")

from dynameta.carriers.devsim_env import import_devsim                        # noqa: E402
from dynameta.carriers import devsim_3d as D3                                  # noqa: E402
from dynameta.carriers import eq_registry as _R                                # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP                         # noqa: E402
from dynameta.carriers.element_geometry import (InexactElementGeometryError,   # noqa: E402
                                                check_element_geometry)

ds = import_devsim()


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _drop(dev):
    if dev in ds.get_device_list():
        ds.delete_device(device=dev)
    if dev + "_m" in ds.get_mesh_list():
        ds.delete_mesh(mesh=dev + "_m")
    _R.clear(dev)


def _mesh_device(dev, P, cells, region="si"):
    """A device from explicit coordinates and triangles / tetrahedra (in memory, no contacts)."""
    _drop(dev)
    P = np.asarray(P, dtype=float)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    el = []
    for c in cells:
        el += [len(c) - 1, 0] + [int(v) for v in c]
    with _quiet():
        ds.create_gmsh_mesh(mesh=dev + "_m", coordinates=[float(v) for v in P.ravel()],
                            physical_names=[region], elements=el)
        ds.add_gmsh_region(mesh=dev + "_m", gmsh_name=region, region=region, material="Si")
        ds.finalize_mesh(mesh=dev + "_m")
        ds.create_device(mesh=dev + "_m", device=dev)
    return dev


def _bipolar(dev, region="si", btbt=None):
    ds.node_model(device=dev, region=region, name="NetDoping", equation="1e22")
    with _quiet():
        BP.setup_bipolar_region(dev, region, eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7,
                                fd_enhancement=False, btbt=btbt)


def _tri_grid(n, h, move=None):
    """n x n squares of side h, each split into two right triangles; move = (node index, (dx, dy))."""
    g = np.arange(n + 1) * h
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel()])
    if move is not None:
        P[move[0]] += np.asarray(move[1])
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            tris += [(a, b, c), (a, c, d)]
    return P, tris


def _unsigned_ratio_2d(P, tri):
    """DEVSIM's 2-D over-count re-derived from the geometry (module docstring): sum |s| L / sum s L."""
    X = np.asarray(P, dtype=float)[list(tri)]
    a, b, c = X
    dd = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    cc = np.array([((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / dd,
                   ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / dd])
    sgn = uns = 0.0
    for i in range(3):
        p, q, o = X[i], X[(i + 1) % 3], X[(i + 2) % 3]
        m, L = 0.5 * (p + q), np.linalg.norm(q - p)
        nrm = np.array([-(q - p)[1], (q - p)[0]]) / L
        if np.dot(o - m, nrm) < 0.0:
            nrm = -nrm
        s = float(np.dot(cc - m, nrm))
        sgn += 0.5 * L * s
        uns += 0.5 * L * abs(s)
    return uns / sgn


OBTUSE_P = [[0.0, 0.0], [2e-8, 0.0], [1e-8, 3e-9], [1e-8, -4e-8]]   # Delaunay pair, one 147-deg triangle
OBTUSE_T = [(0, 1, 2), (0, 3, 1)]
CUBE = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=float) * 1e-8
CUBE5 = [(0, 1, 2, 4), (1, 2, 3, 7), (1, 4, 5, 7), (2, 4, 6, 7), (1, 2, 4, 7)]   # 4 corner tets + 1 regular


# ---- E1 library meshes ----------------------------------------------------------------------------
def test_e1_library_meshes_are_exact():
    dev = "eg_rect"
    _drop(dev)
    with _quiet():                                    # DEVSIM's own 2-D tensor mesher, graded
        ds.create_2d_mesh(mesh=dev + "_m")
        for x, ps in ((0.0, 1e-9), (50e-9, 3e-9), (100e-9, 1e-9)):
            ds.add_2d_mesh_line(mesh=dev + "_m", dir="x", pos=x, ps=ps)
        for y, ps in ((0.0, 2e-9), (40e-9, 5e-9)):
            ds.add_2d_mesh_line(mesh=dev + "_m", dir="y", pos=y, ps=ps)
        ds.add_2d_region(mesh=dev + "_m", material="Si", region="si")
        ds.finalize_mesh(mesh=dev + "_m")
        ds.create_device(mesh=dev + "_m", device=dev)
    try:
        r2 = check_element_geometry(dev, "si")
    finally:
        _drop(dev)
    assert r2["dim"] == 2 and r2["n_inexact"] == 0 and r2["max_element_deviation"] < 1e-13, r2
    # 3-D Kuhn tensor mesh on GRADED axes (the Stacked3DSpec(mesher='tensor') construction)
    ax = (D3._graded_axis([0.0, 12.0], [6.0], 0.5, 3.0) * 1e-9, D3._graded_axis([0.0, 8.0], [], 2.0, 2.0) * 1e-9,
          D3._graded_axis([0.0, 20.0, 30.0], [20.0], 0.5, 4.0) * 1e-9)
    P, T, _c = D3._kuhn_tets(*ax)
    dev = _mesh_device("eg_kuhn", P, T)
    try:
        r3 = check_element_geometry(dev, "si")
    finally:
        _drop(dev)
    assert r3["dim"] == 3 and r3["n_inexact"] == 0 and r3["max_element_deviation"] < 1e-13, r3
    # adversarial: one interior node of a right-triangle grid moved by 0.3 h -> obtuse triangles, refused
    h = 5e-9
    P2, T2 = _tri_grid(4, h, move=(2 * 5 + 2, (0.3 * h, 0.1 * h)))
    dev = _mesh_device("eg_move", P2, T2)
    try:
        with pytest.raises(InexactElementGeometryError):
            check_element_geometry(dev, "si")
    finally:
        _drop(dev)
    P2, T2 = _tri_grid(4, h)                          # the same grid unmoved (hand-built arrays): exact
    dev = _mesh_device("eg_grid", P2, T2)
    try:
        assert check_element_geometry(dev, "si")["max_element_deviation"] < 1e-13
    finally:
        _drop(dev)


def test_e1_one_d_is_trivially_exact():
    dev = "eg_1d"
    _drop(dev)
    with _quiet():
        ds.create_1d_mesh(mesh=dev + "_m")
        ds.add_1d_mesh_line(mesh=dev + "_m", pos=0.0, ps=1e-9, tag="a")
        ds.add_1d_mesh_line(mesh=dev + "_m", pos=10e-9, ps=1e-9, tag="b")
        ds.add_1d_contact(mesh=dev + "_m", name="a", tag="a", material="metal")
        ds.add_1d_contact(mesh=dev + "_m", name="b", tag="b", material="metal")
        ds.add_1d_region(mesh=dev + "_m", material="Si", region="si", tag1="a", tag2="b")
        ds.finalize_mesh(mesh=dev + "_m")
        ds.create_device(mesh=dev + "_m", device=dev)
    try:
        r = check_element_geometry(dev, "si")
    finally:
        _drop(dev)
    assert r["dim"] == 1 and r["n_inexact"] == 0 and r["volume_ratio"] == 1.0


# ---- E2 inexact meshes: measured exactly, refused ----------------------------------------------------
def test_e2_obtuse_triangle_measured_against_the_unsigned_rederivation_and_refused():
    dev = _mesh_device("eg_obt", OBTUSE_P, OBTUSE_T)
    try:
        r = check_element_geometry(dev, "si", raise_error=False)            # measure only
        with pytest.raises(InexactElementGeometryError, match="my_caller") as ei:
            check_element_geometry(dev, "si", caller="my_caller")
    finally:
        _drop(dev)
    worst = max(_unsigned_ratio_2d(OBTUSE_P, t) for t in OBTUSE_T)
    assert worst == pytest.approx(100.0 / 9.0, rel=1e-12)                   # 11.1x (repro_2d.py)
    assert r["max_element_ratio"] == pytest.approx(worst, rel=1e-12)
    def area(t):
        (x0, y0), (x1, y1), (x2, y2) = (OBTUSE_P[v] for v in t)
        return 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    a = [area(t) for t in OBTUSE_T]
    tot = sum(ai * _unsigned_ratio_2d(OBTUSE_P, t) for ai, t in zip(a, OBTUSE_T)) / sum(a)
    assert r["volume_ratio"] == pytest.approx(tot, rel=1e-12)                # 1.7054 (repro_2d.py)
    assert r["n_inexact"] == 1 and r["n_elements"] == 2
    assert isinstance(ei.value, NotImplementedError) and ei.value.ratios == r


def test_e2_five_tet_cube_corner_tets_are_two_x_and_refused():
    dev = _mesh_device("eg_c5", CUBE, CUBE5)
    try:
        r = check_element_geometry(dev, "si", raise_error=False)
        with pytest.raises(InexactElementGeometryError):
            check_element_geometry(dev, "si")
    finally:
        _drop(dev)
    # 4 corner tets of volume 1/6 at 2x (trirectangular: DEVSIM exactly 2x) + the regular tet (1/3) exact
    assert r["n_inexact"] == 4 and r["max_element_ratio"] == pytest.approx(2.0, rel=1e-12)
    assert r["volume_ratio"] == pytest.approx((4 * 2.0 / 6.0 + 1.0 / 3.0) / 1.0, rel=1e-12)


# ---- E3 consumers ----------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["obtuse_2d", "cube5_3d"])
def test_e3_btbt_and_impact_ionization_refuse_inexact_meshes(case):
    from dynameta.carriers.impact_ionization import SILICON_VANOVERSTRAETEN, substrate_current
    from dynameta.carriers.tunneling import SILICON_HURKX, setup_btbt
    P, T = (OBTUSE_P, OBTUSE_T) if case == "obtuse_2d" else (CUBE, CUBE5)
    dev = _mesh_device("eg_c_" + case, P, T)
    try:
        _bipolar(dev)
        before = sorted(ds.get_element_model_list(device=dev, region="si"))
        with pytest.raises(InexactElementGeometryError, match="setup_btbt"):
            setup_btbt(dev, "si", SILICON_HURKX)
        # refused BEFORE any model was created
        assert sorted(ds.get_element_model_list(device=dev, region="si")) == before
        with pytest.raises(InexactElementGeometryError, match="impact_generation_edges"):
            substrate_current(dev, "si", SILICON_VANOVERSTRAETEN)
    finally:
        _drop(dev)


@pytest.mark.parametrize("dim", [2, 3])
def test_e3_btbt_still_installs_on_exact_meshes(dim):
    from dynameta.carriers.tunneling import SILICON_HURKX
    if dim == 2:
        P, T = _tri_grid(3, 5e-9)
    else:
        g = np.linspace(0.0, 1e-8, 3)
        P, T, _c = D3._kuhn_tets(g, g, g)
    dev = _mesh_device("eg_ok{}".format(dim), P, T)
    try:
        _bipolar(dev, btbt=SILICON_HURKX)
        assert "BTBT_U0" in ds.get_element_model_list(device=dev, region="si")
    finally:
        _drop(dev)


def _gmsh_bar(mesh):
    """The pre-2026-11-04 3-D noise bar (validation/devsim_noise.py gate A): an occ box 200 x 80 x 80 nm,
    gmsh MeshSizeMin/Max 14/20 nm, contacts on the x faces -- loaded IN MEMORY (no file)."""
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("eg_bar")
        occ = gmsh.model.occ
        occ.addBox(0, 0, 0, 200.0, 80.0, 80.0)
        occ.synchronize()
        faces = {"left": [], "right": []}
        for dim, tag in gmsh.model.getEntities(2):
            xc = occ.getCenterOfMass(dim, tag)[0]
            if abs(xc) < 1e-4:
                faces["left"].append(tag)
            elif abs(xc - 200.0) < 1e-4:
                faces["right"].append(tag)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 14.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 20.0)
        gmsh.model.mesh.generate(3)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        pos = {int(t): k for k, t in enumerate(tags)}
        P = np.asarray(xyz, dtype=float).reshape(-1, 3) * 1e-9
        _t, ntags = gmsh.model.mesh.getElementsByType(4)
        T = np.array([pos[int(v)] for v in ntags], dtype=int).reshape(-1, 4)
        tri = {}
        for nm, fl in faces.items():
            rows = []
            for f in fl:
                _t, nt = gmsh.model.mesh.getElementsByType(2, f)
                rows.append(np.array([pos[int(v)] for v in nt], dtype=int).reshape(-1, 3))
            tri[nm] = np.concatenate(rows)
    finally:
        gmsh.finalize()
    D3.create_mesh_in_memory(mesh, P, {"semi": T}, tri)


def test_e3_hooge_refuses_the_gmsh_tet_bar_but_diffusion_noise_still_runs():
    import validation.devsim_noise as V
    from dynameta.carriers import physics_drift_diffusion as DD
    from dynameta.carriers.physics_equilibrium import M_E
    V.fresh()
    V.set_bias._v = {}
    dev = "eg_hg"
    try:
        with V.quiet():
            _gmsh_bar(dev + "_m")
            ds.add_gmsh_region(mesh=dev + "_m", gmsh_name="semi", region="bulk", material="ITO")
            ds.add_gmsh_contact(mesh=dev + "_m", gmsh_name="left", region="bulk", name="lo", material="metal")
            ds.add_gmsh_contact(mesh=dev + "_m", gmsh_name="right", region="bulk", name="hi", material="metal")
            ds.finalize_mesh(mesh=dev + "_m")
            ds.create_device(mesh=dev + "_m", device=dev)
            DD.setup_semiconductor_region_dd(dev, "bulk", n_bg_m3=4e26, eps_static=9.5,
                                             dos_mass_kg=0.35 * M_E, mobility_m2Vs=0.004)
            DD.setup_contact_ohmic_dd(dev, "lo")
            DD.setup_contact_ohmic_dd_circuit(dev, "hi", node_name="vhi", source_name="V1")
        ae = max(1e10, 4e26 * 1e-12)
        V._solve(ae, 1e-9)
        V.set_bias(1e-3, ae)
        r = check_element_geometry(dev, "bulk", raise_error=False)
        # the defect the refusal is about: the region total is over-counted (MEASURED 1.2249 on this bar)
        assert r["volume_ratio"] > 1.1 and r["n_inexact"] > 0, r
        with pytest.raises(InexactElementGeometryError, match="hooge"):
            V.noise([1.0], hooge_alpha=2e-3)
        sn = V.noise([1.0])                                   # diffusion (edge-assembled): not refused
        assert sn.by_mechanism["diffusion_n"][0] > 0.0
    finally:
        V.fresh()
        V.set_bias._v = {}
