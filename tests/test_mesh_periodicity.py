"""Mesh-builder gates for the periodic unit cell and the per-region refinement knobs
(audit F-1 / F-2 / F-23). Run: python -m pytest tests/test_mesh_periodicity.py -q

F-1  a full-width inclusion stripe next to a cavity-split full-cell layer produced a mesh whose
     periodic ENTITY TABLE was incomplete (netgen expanded one per-face Face.Identify to that
     face's bottom edge only), so ng.Periodic left boundary dofs unconstrained and the solve was
     silently non-periodic -- T wrong by 9.4 points, no warning, and the error does not converge
     away. Gated here two ways: the trigger geometry must now pair COMPLETELY, and the
     post-build gate must RAISE when handed the old per-face identification.
F-2  a boundary-crossing inclusion made the cavity refinement box overhang the cell, so its side
     faces had no partner and _identify_periodic dropped them without a word.
F-23 every per-region `solid.maxh` was written to a throwaway OCCGeometry.shape copy, so six
     refinement knobs were inert. Gated by sweeping each knob and watching mesh.nv.

Everything here is meshing only (no FEM solve), on deliberately coarse cells.
"""
import numpy as np
import pytest

pytest.importorskip("ngsolve")

import ngsolve as ng                                                            # noqa: E402
from ngsolve.fem import NODE_TYPE                                               # noqa: E402
import netgen.occ as occ                                                        # noqa: E402
from netgen.meshing import IdentificationType                                   # noqa: E402

from dynameta.materials import Material, MaterialRegistry, ConstantOptical      # noqa: E402
from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design         # noqa: E402
from dynameta.geometry.cross_section import Rectangle                           # noqa: E402
from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec                     # noqa: E402
from dynameta.optics import ngsolve_layered as NL                               # noqa: E402
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder               # noqa: E402

PX, PY = 400.0, 200.0          # nm
W, THK = 160.0, 220.0          # stripe width / grating thickness (nm)


# ------------------------------------------------------------------ designs
def _mesh_spec(**kw):
    base = dict(pml_thk_m=300e-9, superstrate_buffer_m=300e-9, substrate_buffer_m=300e-9,
                maxh_superstrate_m=250e-9, maxh_substrate_m=250e-9, maxh_pml_m=250e-9,
                maxh_background_m=70e-9, maxh_inclusion_m=70e-9)
    base.update(kw)
    return Mesh3DSpec(**base)


def _grating_design(cx_nm, cap, pol="x", **mesh_kw):
    """A full-y n=2.5 stripe centred at cx_nm, optionally with a full-cell dielectric layer on
    top (`cap='dielectric'` -- the cavity-split band that is half of the F-1 trigger)."""
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(complex(6.25, 0.0))))
    reg.add(Material("mid", ConstantOptical(complex(4.0, 0.0))))
    stripe = Rectangle(cx_m=cx_nm * 1e-9, cy_m=PY / 2 * 1e-9, width_m=W * 1e-9, height_m=PY * 1e-9)
    layers = [Layer("grating", THK * 1e-9, "air", inclusions=[Inclusion(stripe, "hi")])]
    if cap == "dielectric":
        layers.append(Layer("cap", 100e-9, "mid"))
    return Design(name="periodic_gate", unit_cell=UnitCell(period_x_m=PX * 1e-9,
                                                           period_y_m=PY * 1e-9),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="air"),
                  electrodes=[], materials=reg, mesh_3d=_mesh_spec(**mesh_kw),
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=0.0))


# ------------------------------------------------------- periodicity measures
def _unpaired_counts(mesh, Px, Py):
    """(vertex, edge, face) entities on a periodic boundary plane that are ABSENT from netgen's
    periodic pair table, per axis. Independent re-implementation of the shipped gate (it walks
    the same tables but is written here from the mesh side), so a regression in the gate itself
    cannot mask a regression in the mesh."""
    pts = {v.nr: np.array(v.point) for v in mesh.vertices}
    ent = {"VERTEX": {v.nr: pts[v.nr] for v in mesh.vertices},
           "EDGE": {e.nr: np.mean([pts[v.nr] for v in e.vertices], axis=0) for e in mesh.edges},
           "FACE": {f.nr: np.mean([pts[v.nr] for v in f.vertices], axis=0) for f in mesh.faces}}
    tol = 1e-6 * max(Px, Py)
    need = {"x": {k: set() for k in ent}, "y": {k: set() for k in ent}}
    for el in mesh.Elements(ng.BND):
        vs = [pts[v.nr] for v in el.vertices]
        for ax, i, P in (("x", 0, Px), ("y", 1, Py)):
            if all(abs(p[i]) < tol for p in vs) or all(abs(p[i] - P) < tol for p in vs):
                need[ax]["VERTEX"].update(v.nr for v in el.vertices)
                need[ax]["EDGE"].update(e.nr for e in el.edges)
                need[ax]["FACE"].update(f.nr for f in el.faces)
    have = {"x": {k: set() for k in ent}, "y": {k: set() for k in ent}}
    for k, nt in (("VERTEX", NODE_TYPE.VERTEX), ("EDGE", NODE_TYPE.EDGE), ("FACE", NODE_TYPE.FACE)):
        for (a, b), _idnr in mesh.GetPeriodicNodePairs(nt):
            d = ent[k][b] - ent[k][a]
            if abs(abs(d[0]) - Px) < 1e-3 * Px and abs(d[1]) < 1e-3 * Py:
                have["x"][k].update((a, b))
            elif abs(abs(d[1]) - Py) < 1e-3 * Py and abs(d[0]) < 1e-3 * Px:
                have["y"][k].update((a, b))
    return {ax: {k: len(need[ax][k] - have[ax][k]) for k in ent} for ax in ("x", "y")}


def _h1_periodicity_violation(mesh, Px, Py, order=2, ns=7):
    """Relative jump of a Periodic(H1) GridFunction ACROSS the x-boundary, sampled EXACTLY on
    the two planes (an offset probe compares different interior points and reads ~1e-5 even on a
    perfect mesh -- that is the false positive the audit re-grade warned about). Secondary to
    the entity-table count, kept because it is what the solver actually feels."""
    gf = ng.GridFunction(ng.Periodic(ng.H1(mesh, order=order)))
    gf.Set(ng.x * ng.y + ng.exp(-((ng.x - 0.3 * Px) ** 2 + (ng.y - 0.4 * Py) ** 2) / (0.1 * Px) ** 2))
    bb = mesh.ngmesh.bounding_box
    z0, z1 = bb[0][2], bb[1][2]
    worst = scale = 0.0
    for fz in np.linspace(0.15, 0.85, ns):
        for fy in np.linspace(0.15, 0.85, ns):
            try:
                a = gf(mesh(0.0, fy * Py, z0 + fz * (z1 - z0)))
                b = gf(mesh(Px, fy * Py, z0 + fz * (z1 - z0)))
            except Exception:                        # point outside the mesh -> skip
                continue
            worst = max(worst, abs(a - b))
            scale = max(scale, abs(a), abs(b))
    return worst / max(scale, 1e-30)


def _old_per_face_identify(shape, Px, Py, sym_x=False, sym_y=False):
    """The PRE-FIX identification: one Face.Identify per face pair, each with its own name (and
    so its own idnr). Kept verbatim as the negative control for the post-build gate."""
    tol = max(Px, Py) * 1e-4
    x0, xP, y0, yP = [], [], [], []
    for f in shape.faces:
        c = f.center
        if not sym_x and abs(c.x) < tol:        x0.append(f)
        elif not sym_x and abs(c.x - Px) < tol: xP.append(f)
        elif not sym_y and abs(c.y) < tol:      y0.append(f)
        elif not sym_y and abs(c.y - Py) < tol: yP.append(f)
    sig_yz = lambda f: (round(f.center.y * 1e3), round(f.center.z * 1e3))    # noqa: E731
    sig_xz = lambda f: (round(f.center.x * 1e3), round(f.center.z * 1e3))    # noqa: E731
    xb, yb = {sig_yz(f): f for f in xP}, {sig_xz(f): f for f in yP}
    tx = occ.gp_Trsf.Translation(occ.Vec(Px, 0, 0))
    ty = occ.gp_Trsf.Translation(occ.Vec(0, Py, 0))
    n_px = n_py = 0
    for f0 in x0:
        p = xb.get(sig_yz(f0))
        if p is not None:
            f0.Identify(p, "px_{}".format(n_px), IdentificationType.PERIODIC, tx); n_px += 1
    for f0 in y0:
        p = yb.get(sig_xz(f0))
        if p is not None:
            f0.Identify(p, "py_{}".format(n_py), IdentificationType.PERIODIC, ty); n_py += 1
    return n_px, n_py


# ------------------------------------------------------------------- F-1/F-2
@pytest.mark.parametrize("cx_nm,cap,label", [
    (PX / 2, "none", "interior stripe (clean sibling)"),
    (PX / 2, "dielectric", "F-1 trigger: full-y stripe + cavity-split cap"),
    (0.0, "dielectric", "F-2: boundary-crossing stripe + cavity-split cap"),
    (0.0, "none", "boundary-crossing stripe alone")])
def test_periodic_entity_table_is_complete(cx_nm, cap, label):
    geo = LayeredOpticalBuilder(_grating_design(cx_nm, cap)).build()
    miss = _unpaired_counts(geo.mesh, geo.period_x_nm, geo.period_y_nm)
    assert miss == {"x": {"VERTEX": 0, "EDGE": 0, "FACE": 0},
                    "y": {"VERTEX": 0, "EDGE": 0, "FACE": 0}}, \
        "{}: unpaired periodic entities {}".format(label, miss)
    # and the space the solver actually builds is periodic to round-off
    v = _h1_periodicity_violation(geo.mesh, geo.period_x_nm, geo.period_y_nm)
    assert v < 1e-10, "{}: Periodic(H1) jump across x-boundary {:.3e}".format(label, v)
    assert geo.n_px == 1 and geo.n_py == 1     # one identification per axis == the phase-list order


def test_post_build_gate_catches_the_pre_fix_identification(monkeypatch):
    """The F-1 defect is invisible upstream (every face pairs, the counts agree), so the value of
    the post-build gate is precisely that it fires on the old identification. Negative control:
    put the per-face Identify back and the build must RAISE, naming the periodic table."""
    monkeypatch.setattr(NL, "_identify_periodic", _old_per_face_identify)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        LayeredOpticalBuilder(_grating_design(PX / 2, "dielectric")).build()


def test_pre_fix_identification_is_clean_on_the_untriggered_geometry(monkeypatch):
    """Guard against the gate being a blanket refusal: the same pre-fix identification on the
    clean sibling (no cavity-split cap) pairs completely and must still build."""
    monkeypatch.setattr(NL, "_identify_periodic", _old_per_face_identify)
    geo = LayeredOpticalBuilder(_grating_design(PX / 2, "none")).build()
    assert geo.n_px > 0 and geo.n_py > 0


def test_no_geometry_overhangs_the_unit_cell():
    """F-2 root cause: the cavity refinement box was built from the RAW (unclipped) inclusion
    bbox, so for a boundary-crossing stripe it stuck out of the cell (mesh points at x=-80 nm)."""
    geo = LayeredOpticalBuilder(_grating_design(0.0, "dielectric")).build()
    xy = np.array([v.point[:2] for v in geo.mesh.vertices])
    assert xy[:, 0].min() > -1e-6 and xy[:, 0].max() < geo.period_x_nm + 1e-6
    assert xy[:, 1].min() > -1e-6 and xy[:, 1].max() < geo.period_y_nm + 1e-6


def test_unpairable_periodic_face_raises():
    """An unpairable periodic face used to be dropped silently (F-2), leaving that band at a
    natural BC. Hand _identify_periodic a shape with a lone x=0 face and it must refuse."""
    a = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(PX, PY, 100))         # the cell
    b = occ.Box(occ.Pnt(0, 0, 100), occ.Pnt(0.5 * PX, PY, 200))  # a half-width block on top
    glued = occ.Glue([a, b])                                     # its x=0 face has no x=Px partner
    with pytest.raises(RuntimeError, match="no partner|NO partner"):
        NL._identify_periodic(glued, PX, PY)


# --------------------------------------------------------------------- F-23
def _knob_design(metal_split, **mesh_kw):
    """A 3-layer cell exercising every region class: metal band (split or not), a cavity-split
    dielectric band (_inpatch/_outside), and an inclusion band (__incl0 + background)."""
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(complex(6.25, 0.0))))
    reg.add(Material("au", ConstantOptical(complex(-90.0, 8.0)), is_metal=True))
    P = 400.0
    box = Rectangle(cx_m=P / 2 * 1e-9, cy_m=P / 2 * 1e-9, width_m=160e-9, height_m=160e-9)
    layers = [Layer("metal", 60e-9, "au"), Layer("cavity", 80e-9, "hi"),
              Layer("grating", 100e-9, "air", inclusions=[Inclusion(box, "hi")])]
    base = dict(pml_thk_m=300e-9, superstrate_buffer_m=200e-9, substrate_buffer_m=200e-9,
                maxh_superstrate_m=120e-9, maxh_substrate_m=120e-9, maxh_pml_m=200e-9,
                maxh_metal_m=110e-9, maxh_metal_bulk_m=110e-9, maxh_metal_skin_m=110e-9,
                maxh_inclusion_m=110e-9, maxh_background_m=110e-9,
                metal_skin_thk_m=(20e-9 if metal_split else 0.0))
    base.update(mesh_kw)
    return Design(name="knobs", unit_cell=UnitCell.square(P * 1e-9),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="air"),
                  electrodes=[], materials=reg, mesh_3d=Mesh3DSpec(**base),
                  optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0))


@pytest.mark.parametrize("knob,split", [("maxh_inclusion_m", True),
                                        ("maxh_background_m", True),
                                        ("maxh_metal_skin_m", True),
                                        ("maxh_metal_bulk_m", True),
                                        ("maxh_substrate_m", True),
                                        ("maxh_metal_m", False)])
def test_per_region_maxh_knob_actually_refines(knob, split):
    """F-23: all six of these moved mesh.nv by EXACTLY zero across 25x parameter swings, because
    the loop wrote `solid.maxh` through OCCGeometry.shape -- a fresh wrapper on every access, so
    the values landed on a temporary discarded before GenerateMesh. `maxh_superstrate_m` and
    `maxh_pml_m` were the only two that appeared to work, and only because they also feed the
    GLOBAL maxh. Each knob must now measurably refine its own region."""
    ref = LayeredOpticalBuilder(_knob_design(split)).build().mesh
    fine = LayeredOpticalBuilder(_knob_design(split, **{knob: 35e-9})).build().mesh
    assert fine.nv > 1.3 * ref.nv, \
        "{}: 110 nm -> 35 nm moved mesh.nv only {} -> {}".format(knob, ref.nv, fine.nv)


def test_maxh_routing_sends_each_region_class_to_its_own_knob():
    """Routing contract (a plain function, no meshing). The catch-all must be the BACKGROUND
    knob: while the loop was inert nothing distinguished it from maxh_inclusion_m, whose 5 nm
    default would otherwise land on every plain dielectric film. A metal region follows
    maxh_metal_m whether it is a background band or an inclusion."""
    d = _knob_design(True, maxh_metal_m=11e-9, maxh_metal_skin_m=12e-9, maxh_metal_bulk_m=13e-9,
                     maxh_inclusion_m=14e-9, maxh_background_m=15e-9, maxh_substrate_m=16e-9,
                     maxh_superstrate_m=17e-9, maxh_pml_m=18e-9)
    mh = LayeredOpticalBuilder(d)._maxh
    assert mh("pml_top", "air") == pytest.approx(18.0)
    assert mh("substrate", "air") == pytest.approx(16.0)
    assert mh("superstrate", "air") == pytest.approx(17.0)
    assert mh("metal_skin", "au") == pytest.approx(12.0)
    assert mh("metal_bulk", "au") == pytest.approx(13.0)
    assert mh("grating__incl0", "hi") == pytest.approx(14.0)        # dielectric inclusion -> inclusion knob
    assert mh("grating__incl0", "au") == pytest.approx(11.0)        # metal inclusion -> metal knob
    assert mh("grating", "air") == pytest.approx(15.0)   # inclusion-layer background -> background
    assert mh("film", "hi") == pytest.approx(15.0)       # plain full-cell band -> background
    assert mh("cavity_inpatch", "hi") == pytest.approx(15.0)   # cavity halves are background
    assert mh("cavity_outside", "hi") == pytest.approx(15.0)
