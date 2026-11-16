"""Solver-free gates for optics/fem_2d_tm.py: material input, geometry / mesh planning, the analytic
port background, the zeroth-order extraction, the Design adapter and the run_pipeline seam.

The FE assembly and solve need NGSolve (solver leg: tests/test_fem_2d_tm*.py). Everything before the
mesh object is created -- refusals, the structured-mesh plan (mesh lines, element sizes, corner
grading, buffers, PML strength, probe lines, region of every cell), coefficient records -- and the
post-processing helpers run on numpy. build_tm_cell's last steps hand points and quads to the netgen
mesher; here a RECORDING mesher (_RecMesh, a plain container standing in for netgen.meshing.Mesh)
receives them, so the plan can be inspected cell by cell. Nothing is solved.

Oracles (all written in this file):
  * geometry: the region of every quad from its centroid and the layer / block list;
  * sizes: lambda/(8 n) in dielectrics, the skin depth 1/(k0 Im n) in a metal;
  * buffers: lambda/(2n) and 8 decay lengths of the least-evanescent order, kappa_m =
    sqrt(q_m^2 - eps k0^2), q_m = kx + 2 pi m / P; opaque substrate >= 15 decay lengths;
  * PML strength alpha = 3 ln(1/att) / (2 kz_min d) (quadratic profile round trip);
  * Fresnel / layered r and t: the `tmm` package;
  * Fourier projection: an analytic sum of up- and down-going diffraction orders;
  * resampling: a function exactly representable by the bilinear periodic interpolant.
"""
import cmath
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from dynameta.core.interfaces import OpticalResult
from dynameta.optics import fem_2d_tm as F
from dynameta.optics.fem_2d_tm import (GriddedEps2D, TMBlock, TMLayer, TMMeshSpec, TMSolution,
                                       build_tm_cell, design_to_tm_layers,
                                       make_fem_2d_tm_optical_solver, solve_design_2d_tm,
                                       tm_convergence_study)
from dynameta.warning_types import AdvisoryWarning, UnreliableResultWarning

LAM = 1200e-9
LAM_NM = 1200.0
K0 = 2.0 * math.pi / LAM_NM                           # 1/nm
P_NM = 300.0


# ------------------------------------------------------------------------------------------------
# a recording stand-in for the netgen mesher (no NGSolve, nothing solved)
# ------------------------------------------------------------------------------------------------

class _Pnt:
    def __init__(self, x, y, z):
        self.xyz = (x, y, z)


class _MeshPoint:
    def __init__(self, p):
        self.xyz = p.xyz


class _El2:
    def __init__(self, index, vertices):
        self.index, self.vertices = index, list(vertices)


class _El1:
    def __init__(self, vertices, index=0):
        self.index, self.vertices = index, list(vertices)


class _RecMesh:
    """Records the points, quads, boundary segments, regions and periodic identifications that
    build_tm_cell creates (1-based ids, as netgen returns)."""

    def __init__(self, dim=2):
        self.points, self.quads, self.segs, self.ident = [], [], [], []
        self.regions = {1: [], 2: []}

    def Add(self, obj):
        if isinstance(obj, _MeshPoint):
            self.points.append(obj.xyz)
            return len(self.points)
        (self.quads if isinstance(obj, _El2) else self.segs).append(obj)
        return None

    def AddRegion(self, name, dim):
        self.regions[dim].append(name)
        return len(self.regions[dim])

    def AddPointIdentification(self, a, b, identnr, type):
        self.ident.append((a, b, identnr, type))

    @property
    def ne(self):
        return len(self.quads)

    def quad_table(self):
        """[(region name, x centre, z centre, width, height)] of every quad."""
        out = []
        for q in self.quads:
            xs = [self.points[v - 1][0] for v in q.vertices]
            zs = [self.points[v - 1][1] for v in q.vertices]
            out.append((self.regions[2][q.index - 1], 0.5 * (min(xs) + max(xs)),
                        0.5 * (min(zs) + max(zs)), max(xs) - min(xs), max(zs) - min(zs)))
        return out


@pytest.fixture
def mesher(monkeypatch):
    """Route build_tm_cell's `from netgen.meshing import ...` / `import ngsolve` to the recorder (the
    real modules, when installed, are restored after the test)."""
    meshing = types.ModuleType("netgen.meshing")
    meshing.Element1D, meshing.Element2D, meshing.MeshPoint = _El1, _El2, _MeshPoint
    meshing.Pnt, meshing.Mesh = _Pnt, _RecMesh
    netgen = types.ModuleType("netgen")
    netgen.meshing = meshing
    ng = types.ModuleType("ngsolve")
    ng.Mesh = lambda m: m
    monkeypatch.setitem(sys.modules, "netgen", netgen)
    monkeypatch.setitem(sys.modules, "netgen.meshing", meshing)
    monkeypatch.setitem(sys.modules, "ngsolve", ng)
    return build_tm_cell


def _sq(z):
    w = cmath.sqrt(complex(z))
    return -w if (w.imag < 0 or (w.imag == 0 and w.real < 0)) else w


def _strip_layers(eps_strip=-40 + 3j, lo=90e-9, w=120e-9):
    return [TMLayer("sp", 20e-9, 2.1), TMLayer("gr", 40e-9, 1.0, (TMBlock(lo, lo + w, eps_strip, "s"),))]


def _expected_region(xc, zc, cell, layers):
    """Region of a point from the geometry alone (independent of build_tm_cell's lookup)."""
    zi = cell.z_intervals_nm
    for nm in ("pml_bot", "substrate", "superstrate", "pml_top"):
        if zi[nm][0] < zc < zi[nm][1]:
            return nm
    z = 0.0
    for L in layers:
        t = L.thickness_m * 1e9
        if z < zc < z + t:
            for j, b in enumerate(L.blocks):
                lo, hi = b.x_lo_m * 1e9, b.x_hi_m * 1e9
                for sh in (-P_NM, 0.0, P_NM):
                    if lo + sh < xc < hi + sh:
                        return b.name or "{}__blk{}".format(L.name, j)
            return L.name
        z += t
    raise AssertionError("point outside the cell")


# ------------------------------------------------------------------------------------------------
# build_tm_cell: the mesh plan
# ------------------------------------------------------------------------------------------------

def test_interfaces_and_block_edges_are_mesh_lines_and_every_cell_has_its_region(mesher):
    """Every layer interface is a z line and every block edge an x line (no material boundary cuts an
    element); each quad's region is the one its centre lies in; the lateral boundary points are
    paired x = 0 <-> x = P on every z line (Bloch periodicity)."""
    lays = _strip_layers()
    cell = mesher(P_NM * 1e-9, lays, lambda_m=LAM, n_sub=1.5)
    xs, zs = cell.x_lines_nm, cell.z_lines_nm
    for zv in (0.0, 20.0, 60.0):
        assert np.min(np.abs(zs - zv)) < 1e-9
    for xv in (0.0, 90.0, 210.0, P_NM):
        assert np.min(np.abs(xs - xv)) < 1e-9
    assert xs[0] == 0.0 and xs[-1] == pytest.approx(P_NM, rel=1e-15, abs=0)     # P = 300e-9 m * 1e9
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(zs) > 0)
    m = cell.mesh
    assert cell.n_elements == (xs.size - 1) * (zs.size - 1)
    for name, xc, zc, _w, _h in m.quad_table():
        assert name == _expected_region(xc, zc, cell, lays), (name, xc, zc)
    assert len(m.ident) == zs.size
    for a, b, nr, typ in m.ident:
        pa, pb = m.points[a - 1], m.points[b - 1]
        assert pa[0] == 0.0 and pb[0] == xs[-1] and pa[1] == pb[1] and (nr, typ) == (1, 2)
    assert cell.z_ref_nm == pytest.approx(60.0) and cell.regions["s"].role == "block"
    assert cell.regions["s"].layer == "gr" and cell.regions["sp"].role == "layer"
    assert sorted(m.regions[1]) == ["bottom", "left", "right", "top"]
    # boundary segments: nx on top and bottom, nz on each side
    assert len(m.segs) == 2 * (xs.size - 1) + 2 * (zs.size - 1)


def test_element_sizes_resolve_the_wavelength_and_the_metal_skin_depth(mesher):
    """Auto sizes: lambda/(8 n_max) in dielectrics (n_max over the non-metal regions), the skin depth
    1/(k0 Im n) inside a metal (in x across the strip and in z through its layer); h_scale scales them."""
    cell = mesher(P_NM * 1e-9, _strip_layers(), lambda_m=LAM, n_sub=1.5, mesh_spec=TMMeshSpec(corner_levels=0))
    h_diel = LAM_NM / (8.0 * 1.5)
    skin = 1.0 / (K0 * _sq(-40 + 3j).imag)
    tab = cell.mesh.quad_table()
    for name, xc, zc, w, h in tab:
        assert w <= h_diel * (1 + 1e-9)
        if name == "s":
            assert w <= skin * (1 + 1e-9) and h <= skin * (1 + 1e-9)
        if name == "sp":
            assert h <= LAM_NM / (8.0 * math.sqrt(2.1)) * (1 + 1e-9)
    assert max(w for n_, _x, _z, w, _h in tab if n_ == "s") > 0.5 * skin      # not over-refined
    fine = mesher(P_NM * 1e-9, _strip_layers(), lambda_m=LAM, n_sub=1.5,
                  mesh_spec=TMMeshSpec(corner_levels=0, h_scale=0.5))
    assert max(w for n_, _x, _z, w, _h in fine.mesh.quad_table() if n_ == "s") <= 0.5 * skin * (1 + 1e-9)
    capped = mesher(P_NM * 1e-9, [TMLayer("sp", 20e-9, 2.1, maxh_z_m=2e-9)], lambda_m=LAM,
                    mesh_spec=TMMeshSpec(maxh_m=50e-9, corner_levels=0))
    for name, _x, _z, w, h in capped.mesh.quad_table():
        assert w <= 50.0 * (1 + 1e-9)
        if name == "sp":
            assert h <= 2.0 * (1 + 1e-9)


def test_metal_corners_get_geometric_grading(mesher):
    """Metal strip: the four strip corners (edges x at both faces z) are refined, with nodes at h1 r^l
    (l = 1..levels) from each corner; 'none' refines nothing. A dielectric strip of eps 4 in air on an
    eps 2.1 spacer (corner contrast min(c_v, c_h) = 2.1, above the default corner_contrast 2) is refined
    by the default rule 'contrast' and by 'all', not by 'metal'."""
    spec = TMMeshSpec(corner_levels=4, corner_ratio=0.25)
    cell = mesher(P_NM * 1e-9, _strip_layers(), lambda_m=LAM, n_sub=1.5, mesh_spec=spec)
    assert sorted(cell.corners_nm) == [(90.0, 20.0), (90.0, 60.0), (210.0, 20.0), (210.0, 60.0)]
    xs = cell.x_lines_nm
    for xe, side in ((90.0, +1), (210.0, -1)):
        d = np.sort(side * (xs - xe))
        d = d[d > 1e-9]
        h1 = d[4]                                           # first uniform node past the graded ones
        assert np.allclose(d[:4], h1 * 0.25 ** np.arange(4, 0, -1), rtol=1e-9, atol=0)
    zs = cell.z_lines_nm
    for zf in (20.0, 60.0):                                 # graded on both sides of each face
        for side in (+1, -1):
            d = np.sort(side * (zs - zf))
            d = d[d > 1e-9]
            assert np.allclose(d[1:4] / d[:3], 4.0, rtol=1e-9, atol=0), (zf, side)
    none = mesher(P_NM * 1e-9, _strip_layers(), lambda_m=LAM, n_sub=1.5,
                  mesh_spec=TMMeshSpec(corner_mode="none"))
    assert none.corners_nm == [] and none.x_lines_nm.size < xs.size
    diel = _strip_layers(eps_strip=4.0)
    assert mesher(P_NM * 1e-9, diel, lambda_m=LAM, n_sub=1.5,
                  mesh_spec=TMMeshSpec(corner_mode="metal")).corners_nm == []
    assert sorted(mesher(P_NM * 1e-9, diel, lambda_m=LAM, n_sub=1.5).corners_nm) == sorted(cell.corners_nm)
    assert len(mesher(P_NM * 1e-9, diel, lambda_m=LAM, n_sub=1.5,
                      mesh_spec=TMMeshSpec(corner_mode="all")).corners_nm) == 4
    # strips in the bottom layer touch the substrate: corners at z = 0 and at the layer top
    bot = mesher(P_NM * 1e-9, [TMLayer("gr", 40e-9, 1.0, (TMBlock(90e-9, 210e-9, -40 + 3j, "s"),))],
                 lambda_m=LAM, n_sub=1.5)
    assert sorted(bot.corners_nm) == [(90.0, 0.0), (90.0, 40.0), (210.0, 0.0), (210.0, 40.0)]


def _graded_film_blocks(nblk=6, t=8e-9):
    """A thin film whose Re eps < 0 only near its TOP face (a z-graded accumulation layer), cut into nblk
    lateral blocks of slightly different profiles; the last segment is the layer's own background."""
    z = np.linspace(0.0, t, 11)
    xs = np.linspace(0.0, P_NM * 1e-9, nblk + 1)

    def g(j):
        a = 1.0 + 0.05 * j
        return GriddedEps2D(values=4.4 - a * 12.0 * np.exp(-(t - z) / 0.8e-9) + 0.3j, z_m=z)
    blocks = tuple(TMBlock(float(xs[j]), float(xs[j + 1]), g(j), "f{}".format(j)) for j in range(nblk - 1))
    return TMLayer("film", t, g(nblk - 1), blocks)


def test_metal_corner_rule_judges_each_region_at_the_corner(mesher):
    """corner_mode 'metal' grades a corner where the metal / non-metal boundary BENDS (one, three or two
    diagonal quadrants metallic, each region judged at the corner itself). A film metallic only near its top
    face and cut into lateral blocks is a flat metal / dielectric interface at every block edge (both
    in-layer quadrants metallic at the top face, both above it not; nothing metallic at its bottom face):
    no corner. Grading every block edge at both faces instead (what a rule asking whether a region has
    Re eps < 0 SOMEWHERE does, and what 'all' does) is costly: MEASURED on the strip cell of
    tests/_strip_cell_fixture with its film cut into 31 z-graded blocks, 66 corners and 49434 elements (the
    order-4 solve above 11 GB) against 6 corners, 7695 elements, 0.14 GB and 2.8 s with the metal rule. The
    default rule 'contrast' adds no corner here either (adjacent blocks differ by at most 8 % at the top
    face, far below corner_contrast). A metal strip ON that film keeps its four corners (three quadrants
    metallic below it)."""
    film = _graded_film_blocks()
    flat = mesher(P_NM * 1e-9, [TMLayer("sp", 10e-9, 2.1), film, TMLayer("ox", 9e-9, 2.1)],
                  lambda_m=LAM, n_sub=1.45)
    assert flat.corners_nm == []
    flat_m = mesher(P_NM * 1e-9, [TMLayer("sp", 10e-9, 2.1), film, TMLayer("ox", 9e-9, 2.1)],
                    lambda_m=LAM, n_sub=1.45, mesh_spec=TMMeshSpec(corner_mode="metal"))
    assert flat_m.corners_nm == []
    on = mesher(P_NM * 1e-9, [TMLayer("sp", 10e-9, 2.1), film,
                              TMLayer("st", 30e-9, 1.0, (TMBlock(110e-9, 190e-9, -40 + 3j, "strip"),))],
                lambda_m=LAM, n_sub=1.45)
    assert np.allclose(sorted(on.corners_nm), [(110.0, 18.0), (110.0, 48.0), (190.0, 18.0), (190.0, 48.0)],
                       rtol=0, atol=1e-9)
    # 'all' still grades every block edge at both faces of the film
    allc = mesher(P_NM * 1e-9, [TMLayer("sp", 10e-9, 2.1), film, TMLayer("ox", 9e-9, 2.1)],
                  lambda_m=LAM, n_sub=1.45, mesh_spec=TMMeshSpec(corner_mode="all"))
    assert len(allc.corners_nm) == 2 * 6
    # straight metal walls: a metal block above a metal block of the same x span is a vertical interface
    wall = mesher(P_NM * 1e-9, [TMLayer("a", 20e-9, 2.1, (TMBlock(100e-9, 200e-9, -40 + 3j, "m1"),)),
                                TMLayer("b", 20e-9, 2.1, (TMBlock(100e-9, 200e-9, -30 + 2j, "m2"),))],
                  lambda_m=LAM, n_sub=1.45)
    assert sorted(wall.corners_nm) == [(100.0, 0.0), (100.0, 40.0), (200.0, 0.0), (200.0, 40.0)]


def test_contrast_corner_rule_grades_where_two_permittivity_jumps_meet(mesher):
    """corner_mode 'contrast' (the default): a dielectric corner is graded when the permittivity jump
    across the block edge (c_v) and the one across the layer face (c_h) both exceed corner_contrast,
    min(c_v, c_h) > corner_contrast, each quadrant judged at the corner. Oracle: the four quadrant values
    written out per case (ratio of moduli)."""
    def corners(lays, **kw):
        return sorted(mesher(P_NM * 1e-9, lays, lambda_m=LAM, n_sub=1.5,
                             mesh_spec=TMMeshSpec(**kw)).corners_nm)
    rib = [TMLayer("sp", 20e-9, 2.1), TMLayer("gr", 40e-9, 1.0, (TMBlock(90e-9, 210e-9, 4.0, "s"),))]
    four = [(90.0, 20.0), (90.0, 60.0), (210.0, 20.0), (210.0, 60.0)]
    # bottom corners: air | 4 over 2.1 | 2.1 -> c_v = 4, c_h = max(2.1, 1.9) = 2.1; top: c_v = c_h = 4
    assert corners(rib) == four
    assert corners(rib, corner_contrast=2.05) == four
    assert corners(rib, corner_contrast=2.15) == [(90.0, 60.0), (210.0, 60.0)]   # only the top ones
    assert corners(rib, corner_contrast=4.0) == []                               # strict '>'
    # a low-contrast rib (eps 1.5 in air: c_v = 1.5) is not graded; 'all' grades it
    low = [rib[0], TMLayer("gr", 40e-9, 1.0, (TMBlock(90e-9, 210e-9, 1.5, "s"),))]
    assert corners(low) == [] and corners(low, corner_mode="all") == four
    # a straight wall: the eps-4 block continues into an eps-4 block of the layer above (c_h = 1 at the
    # shared face z = 40), so only the two ends of the wall are corners
    wall = [TMLayer("sp", 20e-9, 2.1), TMLayer("a", 20e-9, 1.0, (TMBlock(90e-9, 210e-9, 4.0, "s1"),)),
            TMLayer("b", 20e-9, 1.0, (TMBlock(90e-9, 210e-9, 4.0, "s2"),))]
    assert corners(wall) == four
    # a lossy block is judged by moduli: |4 + 3i| = 5 against air at the top corners (c_v = c_h = 5), and
    # min(5, max(2.1, 5 / 2.1)) = 2.38 at the bottom ones
    lossy = [rib[0], TMLayer("gr", 40e-9, 1.0, (TMBlock(90e-9, 210e-9, 4.0 + 3.0j, "s"),))]
    assert corners(lossy, corner_contrast=2.3) == four
    assert corners(lossy, corner_contrast=4.9) == [(90.0, 60.0), (210.0, 60.0)]
    assert corners(lossy, corner_contrast=5.1) == []
    with pytest.raises(ValueError, match="corner_contrast"):
        corners(rib, corner_contrast=0.5)


def test_block_names_cannot_alias_a_layer_or_another_block(mesher):
    """A block named like an existing layer, or like a block of another layer, used to be merged into that
    region silently and meshed with ITS permittivity (a strip of eps 4 became eps 2; a metal strip in a second
    layer became the first layer's dielectric strip). Refused now; the two spans of one block split at the
    seam still share their region."""
    with pytest.raises(ValueError, match="duplicate or reserved block name 'a'"):
        mesher(P_NM * 1e-9, [TMLayer("a", 10e-9, 2.0), TMLayer("b", 10e-9, 1.0, (TMBlock(0, 100e-9, 4.0, "a"),))],
               lambda_m=LAM)
    with pytest.raises(ValueError, match="duplicate or reserved block name 's'"):
        mesher(P_NM * 1e-9, [TMLayer("a", 10e-9, 2.0, (TMBlock(0, 100e-9, 4.0, "s"),)),
                             TMLayer("b", 10e-9, 1.0, (TMBlock(0, 100e-9, -40 + 3j, "s"),))], lambda_m=LAM)
    seam = mesher(P_NM * 1e-9, [TMLayer("a", 10e-9, 2.0, (TMBlock(250e-9, 350e-9, 4.0, "s"),))], lambda_m=LAM)
    assert seam.regions["s"].eps.const[0] == 4.0
    with pytest.raises(TypeError, match="TMLayer instances"):
        mesher(P_NM * 1e-9, [("a", 10e-9, 2.0)], lambda_m=LAM)


def test_block_wrapping_across_the_cell_edge_is_a_translation(mesher):
    """A strip crossing x = 0 is one region in two mesh pieces with the same area and the same number
    of refined corners as the centred strip."""
    a = mesher(P_NM * 1e-9, _strip_layers(lo=90e-9), lambda_m=LAM, n_sub=1.5)
    b = mesher(P_NM * 1e-9, _strip_layers(lo=-50e-9), lambda_m=LAM, n_sub=1.5)
    assert len(a.corners_nm) == len(b.corners_nm) == 4
    for c in (a, b):
        area = sum(w * h for n_, _x, _z, w, h in c.mesh.quad_table() if n_ == "s")
        assert area == pytest.approx(120.0 * 40.0, rel=1e-9)
    xb = [x for n_, x, _z, _w, _h in b.mesh.quad_table() if n_ == "s"]
    assert min(xb) < 70.0 and max(xb) > 250.0
    assert np.allclose(sorted(x for x, _ in b.corners_nm), [70.0, 70.0, 250.0, 250.0], rtol=0, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 30.0])
def test_cladding_buffers_pml_strength_and_probe_lines(mesher, theta):
    """Buffers: max(lambda/(2n), 8 decay lengths of the least-evanescent order) in each cladding; PML
    thickness lambda/n with the quadratic-profile strength alpha = 3 ln(1/att)/(2 kz_min d) (clamped to
    [1, 60]); the two probe lines of each cladding lie inside its buffer, <= lambda/(8n) apart."""
    cell = mesher(P_NM * 1e-9, _strip_layers(), lambda_m=LAM, n_sub=1.5, theta_deg=theta)
    kx = K0 * math.sin(math.radians(theta))
    zi = cell.z_intervals_nm
    for nm, n in (("superstrate", 1.0), ("substrate", 1.5)):
        kap = min(math.sqrt(q * q - n * n * K0 * K0) for q in
                  (kx + 2 * math.pi * m / P_NM for m in range(-50, 51)) if abs(q) > n * K0)
        buf = max(LAM_NM / (2 * n), 8.0 / kap)
        assert zi[nm][1] - zi[nm][0] == pytest.approx(buf, rel=1e-12), nm
    for nm, n, k in (("pml_top", 1.0, 0), ("pml_bot", 1.5, 1)):
        d = zi[nm][1] - zi[nm][0]
        assert d == pytest.approx(LAM_NM / n, rel=1e-12)
        kz_min = min(math.sqrt(n * n * K0 * K0 - q * q) for q in
                     (kx + 2 * math.pi * m / P_NM for m in range(-5, 6)) if abs(q) < n * K0)
        alpha = min(max(3 * math.log(1e10) / (2 * max(kz_min, 0.2 * n * K0) * d), 1.0), 60.0)
        assert cell.pml_alpha[k] == pytest.approx(alpha, rel=1e-12, abs=0)
    t0, t1 = zi["superstrate"]
    p0, p1 = cell.probe_super_nm
    assert t0 < p0 < p1 < t1 and p1 - p0 <= LAM_NM / 8 + 1e-9
    b0, b1 = zi["substrate"]
    q0, q1 = cell.probe_sub_nm
    assert b0 < q1 < q0 < b1 and q0 - q1 <= LAM_NM / (8 * 1.5) + 1e-9
    assert zi["pml_bot"][1] == zi["substrate"][0] and zi["superstrate"][1] == zi["pml_top"][0]
    assert not cell.opaque_substrate and cell.theta_deg == theta


def test_explicit_buffers_and_pml_are_honoured(mesher):
    spec = TMMeshSpec(superstrate_buffer_m=200e-9, substrate_buffer_m=300e-9, pml_thickness_m=500e-9,
                      pml_elements=4, pml_attenuation=1e-6)
    cell = mesher(P_NM * 1e-9, [TMLayer("a", 10e-9, 2.0)], lambda_m=LAM, n_sub=1.5, mesh_spec=spec)
    zi = cell.z_intervals_nm
    assert zi["superstrate"][1] - zi["superstrate"][0] == pytest.approx(200.0)
    assert zi["substrate"][1] - zi["substrate"][0] == pytest.approx(300.0)
    assert zi["pml_top"][1] - zi["pml_top"][0] == pytest.approx(500.0)
    assert zi["pml_bot"][1] - zi["pml_bot"][0] == pytest.approx(500.0)
    assert cell.pml_alpha[0] == pytest.approx(3 * math.log(1e6) / (2 * K0 * 500.0), rel=1e-12, abs=0)
    tab = cell.mesh.quad_table()
    x_first = min(x for _n, x, _z, _w, _h in tab)
    assert len([1 for n_, x, _z, _w, _h in tab if n_ == "pml_top" and x == x_first]) == 4


def test_opaque_metal_substrate(mesher):
    """A substrate whose zeroth order decays within a wavelength is opaque: buffer >= 15 decay lengths,
    PML min(lambda, 5 decay lengths) at unit strength. A weakly lossy one stays transparent."""
    n_al = _sq(-120 + 20j)
    cell = mesher(P_NM * 1e-9, [TMLayer("a", 10e-9, 2.0)], lambda_m=LAM, n_sub=n_al)
    kap = (n_al * K0).imag
    zi = cell.z_intervals_nm
    assert cell.opaque_substrate and cell.pml_alpha[1] == 1.0
    assert zi["substrate"][1] - zi["substrate"][0] >= 15.0 / kap * (1 - 1e-12)
    assert zi["pml_bot"][1] - zi["pml_bot"][0] == pytest.approx(min(LAM_NM, 5.0 / kap), rel=1e-12)
    weak = mesher(P_NM * 1e-9, [TMLayer("a", 10e-9, 2.0)], lambda_m=LAM, n_sub=1.5 + 1e-3j)
    assert not weak.opaque_substrate


def test_slowest_decay_is_the_minimum_over_all_orders():
    """_slowest_decay: min over m of Im kz(kx + 2 pi m / P) (full complex kz); oracle: a brute-force minimum
    over |m| <= 400. Zero when an order propagates in a lossless cladding (even with order 0 evanescent);
    for a metal or a fully evanescent cladding it is the order with the smallest |kx_m|, not always 0."""
    def brute(eps, kx, P):
        return min(F._kz(complex(eps), K0, kx + 2 * math.pi * m / P).imag for m in range(-400, 401))
    kx_tir = 1.5 * K0 * math.sin(math.radians(60.0))                     # 1.299 k0
    cases = [(1.0, kx_tir, LAM_NM / 0.9),        # order 0 evanescent, orders -1 / -2 propagate: 0
             (1.0, kx_tir, LAM_NM / 3.0),        # every order evanescent, order 0 slowest
             (1.0, kx_tir, LAM_NM / 2.45),       # every order evanescent, order -1 slowest
             (-120 + 20j, 0.0, 300.0), (-120 + 20j, 0.3 * K0, 300.0), (2.25 + 0.5j, kx_tir, 700.0),
             (2.25, 0.2 * K0, 900.0)]
    got = [F._slowest_decay(complex(e), K0, kx, P) for e, kx, P in cases]
    for g_, (e, kx, P) in zip(got, cases):
        assert g_ == brute(e, kx, P), (e, kx, P)
    assert got[0] == 0.0 and got[6] == 0.0
    assert got[1] == F._kz(1.0 + 0j, K0, kx_tir).imag
    assert got[2] == F._kz(1.0 + 0j, K0, kx_tir - 2 * math.pi / (LAM_NM / 2.45)).imag < got[1]


def test_grating_coupled_tir_substrate_is_not_opaque(mesher):
    """n_super 1.5, air substrate, 60 deg, P = lambda / 0.9: order 0 decays within lambda in the substrate,
    orders -1 / -2 propagate. The substrate is transparent: its buffer is the automatic one (8 decay lengths
    of the slowest EVANESCENT order, here order 0), its PML lambda / n thick with the strength alpha sized on
    the slowest propagating order (order -2, kz = 0.866 k0), not the opaque unit strength. With every order
    evanescent (P = lambda / 2.45, order -1 the slowest) it is opaque and its buffer is >= 15 decay lengths
    of that slowest order, not of order 0."""
    kx = 1.5 * K0 * math.sin(math.radians(60.0))
    P = LAM_NM / 0.9
    lays = [TMLayer("g", 60e-9, 2.25, (TMBlock(0.3 * P * 1e-9, 0.7 * P * 1e-9, 4.0, "blk"),))]
    cell = mesher(P * 1e-9, lays, lambda_m=LAM, n_super=1.5, n_sub=1.0, theta_deg=60.0)
    zi = cell.z_intervals_nm
    assert not cell.opaque_substrate
    kap0 = math.sqrt(kx * kx - K0 * K0)
    assert zi["substrate"][1] - zi["substrate"][0] == pytest.approx(max(LAM_NM / 2, 8.0 / kap0), rel=1e-12, abs=0)
    d = zi["pml_bot"][1] - zi["pml_bot"][0]
    assert d == pytest.approx(LAM_NM, rel=1e-12, abs=0)
    kz_min = min(math.sqrt(K0 * K0 - q * q) for q in (kx + 2 * math.pi * m / P for m in range(-5, 6)) if abs(q) < K0)
    assert kz_min == math.sqrt(K0 * K0 - (kx - 2 * 2 * math.pi / P) ** 2)          # order -2
    alpha = min(max(3 * math.log(1e10) / (2 * max(kz_min, 0.2 * K0) * d), 1.0), 60.0)
    assert cell.pml_alpha[1] == pytest.approx(alpha, rel=1e-12, abs=0) and cell.pml_alpha[1] != 1.0
    P2 = LAM_NM / 2.45
    lays2 = [TMLayer("g", 60e-9, 2.25, (TMBlock(0.3 * P2 * 1e-9, 0.7 * P2 * 1e-9, 4.0, "blk"),))]
    opq = mesher(P2 * 1e-9, lays2, lambda_m=LAM, n_super=1.5, n_sub=1.0, theta_deg=60.0)
    kap_m1 = math.sqrt((kx - 2 * math.pi / P2) ** 2 - K0 * K0)
    assert opq.opaque_substrate and opq.pml_alpha[1] == 1.0 and kap_m1 < kap0
    assert opq.z_intervals_nm["substrate"][1] - opq.z_intervals_nm["substrate"][0] >= 15.0 / kap_m1 * (1 - 1e-12)


def test_near_cutoff_order_caps_the_buffer_with_a_warning(mesher):
    """An order just past its cutoff decays over many wavelengths; the buffer is capped at 30 lambda/n
    and an UnreliableResultWarning says why."""
    P = LAM_NM * (1 - 1e-7)
    with pytest.warns(UnreliableResultWarning, match="cutoff"):
        cell = mesher(P * 1e-9, [TMLayer("a", 10e-9, 2.0)], lambda_m=LAM,
                      mesh_spec=TMMeshSpec(substrate_buffer_m=500e-9))
    zi = cell.z_intervals_nm
    assert zi["superstrate"][1] - zi["superstrate"][0] == pytest.approx(30 * LAM_NM, rel=1e-12)


def test_gridded_layers_put_their_samples_on_the_mesh(mesher):
    """A z-graded GriddedEps2D puts every native z sample on a mesh line; a laterally structured one
    finer than max_grid_lines is not aligned and says so (AdvisoryWarning)."""
    z = np.array([0.0, 0.7, 1.9, 3.0, 4.0]) * 1e-9
    g = GriddedEps2D(values=np.linspace(2.0, 4.0 + 0.5j, z.size), z_m=z)
    cell = mesher(P_NM * 1e-9, [TMLayer("lo", 9e-9, 2.0), TMLayer("g", 4e-9, g)], lambda_m=LAM)
    for zv in 9.0 + z * 1e9:
        assert np.min(np.abs(cell.z_lines_nm - zv)) < 1e-9
    x = np.linspace(0.0, P_NM, 16, endpoint=False) * 1e-9
    v = 2.0 + 0.5 * np.cos(2 * math.pi * x / (P_NM * 1e-9))[None, :] * np.ones((2, 1))
    lat = GriddedEps2D(values=v, z_m=[0.0, 5e-9], x_m=x)
    with pytest.warns(AdvisoryWarning, match="max_grid_lines"):
        mesher(P_NM * 1e-9, [TMLayer("g", 5e-9, lat)], lambda_m=LAM, mesh_spec=TMMeshSpec(max_grid_lines=4))
    ok = mesher(P_NM * 1e-9, [TMLayer("g", 5e-9, lat)], lambda_m=LAM)
    for xv in np.linspace(0.0, P_NM, 17):
        assert np.min(np.abs(ok.x_lines_nm - xv)) < 1e-9


def test_hydrodynamic_conductor_layer_gets_its_own_lines(mesher):
    """A HydroConductor2D layer: the region keeps the spec (the solver adds the extension), its H_y
    operator is eps_inf, and its conductor z lines (rows of h_z) are mesh lines; in the exact local
    limit no extension is attached and the profile samples are the lines."""
    from dynameta.constants import M_E
    from dynameta.optics.fem_2d_tm_hydro import hydro_conductor_from_profile
    zp = np.linspace(0.0, 6e-9, 11)
    npf = 3e26 + 1.2e27 * np.exp(-zp / 1e-9)
    cond = hydro_conductor_from_profile(zp, npf, m_eff_kg=0.30 * M_E, eps_inf=4.0, gamma=1.2e14,
                                        check_validity=False, h_z_m=0.5e-9)
    lays = [TMLayer("lo", 6e-9, 4.0), TMLayer("f", 6e-9, cond), TMLayer("up", 6e-9, 4.0)]
    cell = mesher(100e-9, lays, lambda_m=LAM, theta_deg=60.0, n_sub=1.45)
    r = cell.regions["f"]
    assert r.eps.hydro is cond and r.eps.const == (4.0 + 0j, 0j, 0j, 4.0 + 0j)
    for zv in 6.0 + np.linspace(0.0, 6.0, 11):
        assert np.min(np.abs(cell.z_lines_nm - zv)) < 1e-9
    loc = mesher(100e-9, [TMLayer("lo", 6e-9, 4.0), TMLayer("f", 6e-9, cond.with_options(beta_scale=0.0)),
                          TMLayer("up", 6e-9, 4.0)], lambda_m=LAM, theta_deg=60.0, n_sub=1.45)
    assert loc.regions["f"].eps.hydro is None and loc.regions["f"].eps.grid is not None
    with pytest.raises(NotImplementedError, match="blocks inside"):
        mesher(100e-9, [TMLayer("f", 6e-9, cond, (TMBlock(0.0, 5e-8, -10 + 1j),))], lambda_m=LAM)
    with pytest.raises(ValueError, match="adjacent"):
        mesher(100e-9, [TMLayer("f1", 6e-9, cond), TMLayer("f2", 6e-9, cond)], lambda_m=LAM)
    with pytest.raises(NotImplementedError, match="not as a block"):
        mesher(100e-9, [TMLayer("a", 6e-9, 2.0, (TMBlock(0.0, 5e-8, cond),))], lambda_m=LAM)
    with pytest.raises(ValueError, match="thick"):
        mesher(100e-9, [TMLayer("f", 7e-9, cond)], lambda_m=LAM)


@pytest.mark.parametrize("kwargs, layers, exc, msg", [
    (dict(mesh_spec=TMMeshSpec(corner_mode="some")), None, ValueError, "corner_mode"),
    (dict(mesh_spec=TMMeshSpec(corner_ratio=1.0)), None, ValueError, "corner_ratio"),
    (dict(mesh_spec=TMMeshSpec(corner_contrast=0.9)), None, ValueError, "corner_contrast"),
    (dict(mesh_spec=TMMeshSpec(h_scale=0.0)), None, ValueError, "h_scale"),
    (dict(period=0.0), None, ValueError, "period_x_m"),
    (dict(lambda_m=-1.0), None, ValueError, "lambda_m"),
    (dict(), [], ValueError, "at least one layer"),
    (dict(n_super=1.0 + 0.1j), None, NotImplementedError, "LOSSLESS"),
    (dict(theta_deg=90.0), None, ValueError, "theta_deg"),
    (dict(), [SimpleNamespace(name="a", thickness_m=1e-8, eps=2.0, blocks=())], TypeError,
     "TMLayer instances"),
    (dict(), [TMLayer("a", 1e-8, 2.0), TMLayer("a", 1e-8, 2.0)], ValueError, "duplicate or reserved"),
    (dict(), [TMLayer("substrate", 1e-8, 2.0)], ValueError, "duplicate or reserved"),
    (dict(), [TMLayer("a", 0.0, 2.0)], ValueError, "thickness must be positive"),
    (dict(), [TMLayer("a", 1e-8, 2.0, (TMBlock(5e-8, 5e-8, 4.0),))], ValueError, "non-positive width"),
    (dict(), [TMLayer("a", 1e-8, 2.0, (TMBlock(0.0, 4e-7, 4.0),))], ValueError, "wider than the period"),
    (dict(), [TMLayer("a", 1e-8, 2.0, (TMBlock(0, 1e-7, 4.0), TMBlock(5e-8, 2e-7, 4.0)))], ValueError,
     "overlap"),
    (dict(), [TMLayer("b", 1e-8, 2.0, (TMBlock(0, 1e-7, 4.0, "pml_top"),))],
     ValueError, "duplicate or reserved block name"),
    (dict(), [TMLayer("a", 1e-8, 2.0, (TMBlock(1e-7, 1e-7 + 1e-14, 4.0),))], ValueError,
     "received no mesh cell"),
    (dict(), [TMLayer("a", 1e-8, [2.0, 2.0, 2.0])], TypeError, "complex scalar"),
    (dict(), [TMLayer("a", 1e-8, complex("nan"))], ValueError, "non-finite"),
    (dict(), [TMLayer("a", 1e-8, SimpleNamespace(beta=1e6, wp=1e16))], NotImplementedError, "nonlocal"),
])
def test_build_refusals_name_the_reason(mesher, kwargs, layers, exc, msg):
    kw = dict(kwargs)
    period = kw.pop("period", P_NM * 1e-9)
    kw.setdefault("lambda_m", LAM)
    lays = [TMLayer("a", 10e-9, 2.0)] if layers is None else layers
    with pytest.raises(exc, match=msg):
        mesher(period, lays, **kw)


# ------------------------------------------------------------------------------------------------
# material records
# ------------------------------------------------------------------------------------------------

def test_region_eps_scales_and_flags():
    """_RegionEps: n_max / kappa_max from the passive root of the diagonal components; metal flag from
    Re eps < 0; isotropic only for a scalar; tuples and 2x2 tensors are (exx, exz, ezx, ezz)."""
    r = F._RegionEps(-40 + 3j, "m")
    n = _sq(-40 + 3j)
    assert r.is_metal and r.isotropic and r.const == (-40 + 3j, 0j, 0j, -40 + 3j)
    assert r.n_max == pytest.approx(abs(n.real)) and r.kappa_max == pytest.approx(n.imag)
    u = F._RegionEps((2.0, 3.0 + 0.1j), "u")
    assert not u.isotropic and not u.is_metal and u.const == (2.0, 0j, 0j, 3.0 + 0.1j)
    assert u.n_max == pytest.approx(_sq(3.0 + 0.1j).real)
    t = F._RegionEps(np.array([[2.0, 0.1], [0.2, 3.0]]), "t")
    assert t.const == (2.0, 0.1, 0.2, 3.0) and not t.isotropic
    with pytest.raises(TypeError, match="2x2 x-z tensor"):
        F._RegionEps(np.ones(3), "bad")
    gu = GriddedEps2D.uniaxial([2.0, 2.5], [3.0, 3.5], [0.0, 1e-9])
    assert gu.values.shape == (2, 2, 2) and gu.values[1, 0, 0] == 2.5 and gu.values[1, 1, 1] == 3.5
    assert gu.values[0, 0, 1] == 0 and gu.values[0, 1, 0] == 0
    with pytest.raises(ValueError, match="shapes differ"):
        GriddedEps2D.uniaxial([2.0, 2.5], [3.0], [0.0, 1e-9])


def test_gridded_record_orientation_collapse_and_refusals():
    """A descending z axis is flipped with its data; identical lateral columns collapse to the
    laterally uniform (native-z) record; malformed grids are refused with the reason."""
    z = np.array([5e-9, 2e-9, 0.0])
    v = np.array([3.0, 2.0, 1.0])
    r = F._RegionEps(GriddedEps2D(values=v, z_m=z), "g", thickness_nm=5.0, period_nm=P_NM)
    assert np.allclose(r.grid["z"], [0.0, 2.0, 5.0], rtol=0, atol=1e-12)
    assert np.array_equal(r.grid["comps"][0][:, 0], [1, 2, 3])
    xl, zl = r.mesh_lines(10.0, P_NM, 256)
    assert r.grid["x"] is None and xl == [] and np.allclose(zl, [10.0, 12.0, 15.0], rtol=0, atol=1e-12)
    cols = np.repeat(v[:, None], 8, axis=1)
    rc = F._RegionEps(GriddedEps2D(values=cols, z_m=z, x_m=np.arange(8) * 30e-9), "g",
                      thickness_nm=5.0, period_nm=P_NM)
    assert rc.grid["x"] is None and rc.grid["comps"][0].shape == (3, 1)
    rt = F._RegionEps(GriddedEps2D.uniaxial(v, 2 * v, z), "g", thickness_nm=5.0)
    assert np.allclose(rt.grid["comps"][3][:, 0], [2, 4, 6]) and rt.n_max == pytest.approx(math.sqrt(6))
    bad = [
        (GriddedEps2D(values=np.ones((2, 3, 4)), z_m=[0, 5e-9]), "must be"),
        (GriddedEps2D(values=[1.0, 2.0, 3.0], z_m=[0, 5e-9]), ">= 2 z samples"),
        (GriddedEps2D(values=[1.0, 2.0, 3.0], z_m=[0, 4e-9, 1e-9]), "monotonic"),
        (GriddedEps2D(values=[1.0, 2.0], z_m=[0, 5.0]), "check the length unit"),
        (GriddedEps2D(values=[[1.0, 2.0], [1.0, 3.0]], z_m=[0, 5e-9]), "needs x_m"),
        (GriddedEps2D(values=[[1.0, 2.0], [1.0, 3.0]], z_m=[0, 5e-9], x_m=[0.0]), "Nx entries"),
        (GriddedEps2D(values=[[1.0, 2.0], [1.0, 3.0]], z_m=[0, 5e-9], x_m=[0.0, 150.0]), "outside the cell"),
    ]
    for g, msg in bad:
        with pytest.raises(ValueError, match=msg):
            F._RegionEps(g, "g", thickness_nm=5.0, period_nm=P_NM)


def test_lateral_resample_is_exact_for_a_bilinear_periodic_profile():
    """The voxel resample of a laterally structured grid (periodic linear in x, linear in z) reproduces
    exactly a profile the interpolant can represent: a triangle wave in x with kinks only at native
    samples, times a linear function of z, on non-uniform native axes. Its x/z nodes are mesh lines
    while within max_grid_lines."""
    def g(x):
        x = np.mod(x, P_NM)
        return np.where(x <= 150.0, 1.0 + x / 75.0, 3.0 - (x - 150.0) / 75.0)
    xs = np.array([0.0, 50.0, 150.0, 225.0])
    zs = np.array([0.0, 2.0, 5.0, 10.0])
    f = lambda zz, xx: (1.0 + 0.1j) * (1.0 + zz / 10.0) * g(xx)       # noqa: E731
    vals = f(zs[:, None], xs[None, :])
    r = F._RegionEps(GriddedEps2D(values=vals, z_m=zs * 1e-9, x_m=xs * 1e-9), "g", thickness_nm=10.0,
                     period_nm=P_NM)
    xu, zu, comps = r.resampled(P_NM, 256)
    assert np.allclose(xu, np.linspace(0.0, P_NM, 7), rtol=0, atol=1e-12)
    assert np.allclose(zu, np.linspace(0.0, 10.0, 6), rtol=0, atol=1e-12)
    assert np.allclose(comps[0], f(zu[:, None], xu[None, :]), rtol=1e-13, atol=0)
    assert np.allclose(comps[3], comps[0], rtol=0, atol=0) and not np.any(comps[1])
    xl, zl = r.mesh_lines(4.0, P_NM, 256)
    assert np.array_equal(xl, xu) and np.allclose(zl, 4.0 + zu, rtol=0, atol=1e-12)
    assert r.mesh_lines(4.0, P_NM, 3) == ([], [])
    # a uniformly sampled z axis is used natively (no resampling in z)
    r2 = F._RegionEps(GriddedEps2D(values=vals[[0, 2]], z_m=[0.0, 10e-9], x_m=xs * 1e-9), "g")
    assert np.allclose(r2.voxel_axes(P_NM, 256)[1], [0.0, 10.0], rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------------------
# small kernels
# ------------------------------------------------------------------------------------------------

def test_wrap_blocks_and_graded_axis():
    """_wrap_blocks: a full-width block covers [0, P]; a block crossing x = P splits at the seam; a
    block starting at x = P starts at 0. _graded_axis: every break kept, interval k cut into
    ceil(L/h_k) cells, geometric nodes h1 r^l next to each graded break."""
    P = 300.0
    L = TMLayer("a", 1e-8, 1.0, (TMBlock(0.0, 300e-9, 4.0),))
    assert F._wrap_blocks(L, P, "w") == [(0.0, P, 0)]
    L2 = TMLayer("a", 1e-8, 1.0, (TMBlock(250e-9, 350e-9, 4.0), TMBlock(300e-9, 320e-9 + 30e-9, 4.0)))
    with pytest.raises(ValueError, match="overlap"):
        F._wrap_blocks(L2, P, "w")
    L3 = TMLayer("a", 1e-8, 1.0, (TMBlock(250e-9, 350e-9, 4.0), TMBlock(660e-9, 680e-9, 4.0)))
    sp = F._wrap_blocks(L3, P, "w")
    assert [s[2] for s in sp] == [0, 1, 0]
    assert np.allclose([s[:2] for s in sp], [(0.0, 50.0), (60.0, 80.0), (250.0, P)], rtol=0, atol=1e-9)
    L4 = TMLayer("a", 1e-8, 1.0, (TMBlock(300e-9, 320e-9, 4.0),))
    (s4,) = F._wrap_blocks(L4, P, "w")
    assert abs(s4[0]) < 1e-9 and abs(s4[1] - 20.0) < 1e-9
    breaks = [0.0, 10.0, 100.0]
    ax = F._graded_axis(breaks, [3.0, 20.0], [10.0], 3, 0.5, 1e-9)
    assert ax[0] == 0.0 and ax[-1] == 100.0 and np.all(np.diff(ax) > 0)
    uni = [0.0 + 10.0 * i / 4 for i in range(5)] + [10.0 + 90.0 * i / 5 for i in range(1, 6)]
    graded = [10.0 - 2.5 * 0.5 ** l for l in (1, 2, 3)] + [10.0 + 18.0 * 0.5 ** l for l in (1, 2, 3)]
    assert np.allclose(ax, np.sort(uni + graded), rtol=0, atol=1e-12)
    assert np.allclose(F._graded_axis(breaks, [3.0, 20.0], [10.0], 0, 0.5, 1e-9), np.sort(uni), rtol=0, atol=1e-12)


def test_passive_sqrt_and_propagating_orders():
    """_passive_sqrt picks Im >= 0 (and +real on the cut); _propagating lists exactly the orders with
    |kx + 2 pi m/P| < n k0 (always including 0)."""
    assert F._passive_sqrt(-4.0) == 2j and F._passive_sqrt(4.0) == 2.0
    assert F._passive_sqrt(-1 - 1e-30j).imag > 0
    for n, th, P in ((1.0, 0.0, 300.0), (1.5, 20.0, 2000.0), (1.45, 70.0, 1700.0)):
        kx = K0 * math.sin(math.radians(th))
        want = [m for m in range(-60, 61) if abs(kx + 2 * math.pi * m / P) < n * K0]
        assert F._propagating(n * n, K0, kx, P) == sorted(set(want) | {0})


def test_gauss_line_integrates_polynomials_and_harmonics_exactly():
    xl = np.array([0.0, 30.0, 45.0, 110.0, 300.0])
    xq, wq = F._gauss_line(xl, 6)
    assert xq.size == 24 and np.sum(wq) == pytest.approx(300.0, rel=1e-14, abs=0)
    assert np.dot(wq, xq ** 11) == pytest.approx(300.0 ** 12 / 12, rel=1e-12)
    xq, wq = F._gauss_line(np.linspace(0.0, 300.0, 21), 10)
    assert abs(np.dot(wq, np.exp(2j * math.pi * 3 * xq / 300.0))) < 1e-12


def test_background_is_the_two_half_space_fresnel_solution():
    """_Background (unit incident H_y at the top interface): rF and t = tF n1/n2 are tmm's p-hat
    Fresnel r and t; the reflection referenced at z_ref is tmm's r of the same interface seen through
    z_ref of superstrate; grazing incidence is refused."""
    tmm = pytest.importorskip("tmm")
    n1, n2, zr = 1.0, 1.5 + 0.02j, 60.0
    for th in (0.0, 40.0):
        kx = n1 * K0 * math.sin(math.radians(th))
        bg = F._Background(SimpleNamespace(n_super=n1, n_sub=n2, z_ref_nm=zr), K0, kx)
        ref = tmm.coh_tmm("p", [n1, n2], [np.inf, np.inf], math.radians(th), LAM_NM)
        assert abs(bg.rF - ref["r"]) < 1e-13
        assert abs(bg.tF * n1 / n2 - ref["t"]) < 1e-13
        shifted = tmm.coh_tmm("p", [n1, n1, n2], [np.inf, zr, np.inf], math.radians(th), LAM_NM)
        assert abs(bg.R0_ref - shifted["r"]) < 1e-12
        assert abs(bg.T0_bot - bg.tF * cmath.exp(1j * bg.kz1 * zr)) < 1e-15
    with pytest.raises(NotImplementedError, match="grazing"):
        F._Background(SimpleNamespace(n_super=1.0, n_sub=1.5, z_ref_nm=0.0), K0, K0)


class _FieldMesh:
    """Stands in for an NGSolve mesh + GridFunction: mesh(x, z) returns the points, gf(points) an
    analytic superposition of up- and down-going diffraction orders."""

    def __init__(self, inside=True):
        self.inside = inside

    def __call__(self, x, z):
        return {"nr": np.zeros(x.size, int) if self.inside else -np.ones(x.size, int), "x": x, "z": z}


def test_order_extraction_recovers_up_and_down_going_amplitudes():
    """_extract_orders projects the field on two lines onto each order (Gauss quadrature on the mesh
    x cells) and splits it into the outgoing and incoming parts referenced at z0, for propagating and
    evanescent orders, in either cladding orientation."""
    P, eps, kx = 300.0, 2.25, K0 * 0.3
    amps = {-1: (0.3 - 0.1j, 0.05j), 0: (0.7 + 0.2j, -0.02 + 0.01j), 1: (-0.2j, 0.1)}
    z0 = 80.0
    for sign in (+1, -1):
        def gf(pts, sign=sign):
            x, z = pts["x"], pts["z"]
            out = np.zeros(x.size, complex)
            for m, (a, b) in amps.items():
                kz = cmath.sqrt(eps * K0 ** 2 - (kx + 2 * math.pi * m / P) ** 2)
                out += (a * np.exp(sign * 1j * kz * (z - z0)) + b * np.exp(-sign * 1j * kz * (z - z0))) \
                    * np.exp(1j * (kx + 2 * math.pi * m / P) * x)
            return out
        cell = SimpleNamespace(period_nm=P, x_lines_nm=np.linspace(0.0, P, 13), mesh=_FieldMesh())
        got = F._extract_orders(cell, gf, K0, kx, eps, (z0 + 40.0 * sign, z0 + 55.0 * sign), z0, sign,
                                [-1, 0, 1], 10)
        for m, (a, b) in amps.items():
            assert abs(got[m][0] - a) < 1e-11 and abs(got[m][1] - b) < 1e-11, (sign, m)
    off = SimpleNamespace(period_nm=P, x_lines_nm=np.linspace(0.0, P, 4), mesh=_FieldMesh(False))
    with pytest.raises(RuntimeError, match="left the mesh"):
        F._extract_orders(off, lambda p: np.zeros(p["x"].size), K0, kx, eps, (1.0, 2.0), 0.0, 1, [0], 4)
    on = SimpleNamespace(period_nm=P, x_lines_nm=np.linspace(0.0, P, 4), mesh=_FieldMesh())
    with pytest.raises(NotImplementedError, match="grazing cutoff"):
        F._extract_orders(on, lambda p: np.zeros(p["x"].size), K0, 1.5 * K0, eps, (1.0, 2.0), 0.0, 1, [0], 4)


def test_flux_and_inplane_curl_close_the_constitutive_relation():
    """Formulation (1): with Lambda = eps^T / det(eps) (components (xx, xz, zx, zz) as the assembly
    stores them), E ~ J Lambda grad H and D ~ J grad H, so eps (J Lambda g) = J g for ANY in-plane
    tensor and gradient (the 2x2 identity A J A^T = det(A) J); for a diagonal tensor the x-derivative
    is weighted by 1/ezz and the z-derivative by 1/exx."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        eps = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) + 3 * np.eye(2)
        lam = eps.T / np.linalg.det(eps)
        L = (lam[0, 0], lam[0, 1], lam[1, 0], lam[1, 1])
        g = rng.normal(size=2) + 1j * rng.normal(size=2)
        e_dir = np.array(F._AssemblyContext.inplane_curl(F._flux(L, g)))
        d_dir = np.array(F._AssemblyContext.inplane_curl(g))
        assert np.allclose(eps @ e_dir, d_dir, rtol=1e-12, atol=0)
    fx, fz = F._flux((1 / 4.0, 0.0, 0.0, 1 / 2.0), (1.0, 1.0))       # eps = diag(exx=2, ezz=4)
    assert (fx, fz) == (0.25, 0.5)


def test_refuse_nonlocal_material_input():
    """Hydrodynamic records (by class name or by carrying beta and wp) and a finite-k Mermin model are
    refused naming the supported route; the local Mermin limit and plain numbers pass."""
    class HydroLayer:
        pass

    class MerminDrudeOptical:
        def __init__(self, k):
            self.k_per_m = k
    for obj in (HydroLayer(), SimpleNamespace(beta=1e6, wp=1e16), MerminDrudeOptical(1e7)):
        with pytest.raises(NotImplementedError, match="HydroConductor2D"):
            F._refuse_nonlocal(obj, "here")
    F._refuse_nonlocal(MerminDrudeOptical(0.0), "here")
    F._refuse_nonlocal(2.0 + 0.1j, "here")


# ------------------------------------------------------------------------------------------------
# helpers around the solver: mesh-spec overrides, convergence table, pipeline seam
# ------------------------------------------------------------------------------------------------

def test_spec_overrides():
    base = TMMeshSpec(corner_levels=5, corner_ratio=0.2)
    assert F._spec_with(base, None, None) is base
    assert F._spec_with(None, 3e-8, None) == TMMeshSpec(maxh_m=3e-8)
    assert F._spec_with(base, None, False).corner_levels == 0
    assert F._spec_with(base, None, 0).corner_levels == 0
    s = F._spec_with(base, None, (3, 0.4))
    assert (s.corner_levels, s.corner_ratio) == (3, 0.4)
    assert F._spec_with(base, None, 8) == TMMeshSpec(corner_levels=8, corner_ratio=0.2)


def test_convergence_table_differences():
    """tm_convergence_study: |r - r_prev| and |r - r_last| per row (the last setting is the reference),
    with ndof / energy_residual carried from a TMSolution and None from a bare OpticalResult."""
    rs = {1: 0.30 + 0.10j, 2: 0.31 + 0.12j, 4: 0.3105 + 0.1201j}

    def solve(s):
        res = OpticalResult(r=rs[s], R=abs(rs[s]) ** 2, phase_deg=0.0, solve_time_s=0.0)
        if s == 4:
            return TMSolution(result=res, r_orders={0: rs[s]}, t_orders={}, R_orders={}, T_orders={},
                              energy_residual=1e-9, pml_reflection=0.0, ndof=1234, n_elements=10,
                              assemble_solve_s=0.0, cell=None)
        return res
    rows = tm_convergence_study(solve, [1, 2, 4])
    assert [r_["setting"] for r_ in rows] == [1, 2, 4]
    assert rows[0]["d_prev"] is None
    assert rows[1]["d_prev"] == abs(rs[2] - rs[1]) and rows[2]["d_prev"] == abs(rs[4] - rs[2])
    assert [r_["d_last"] for r_ in rows] == [abs(rs[k] - rs[4]) for k in (1, 2, 4)]
    assert rows[2]["ndof"] == 1234 and rows[2]["energy_residual"] == 1e-9
    assert rows[0]["ndof"] is None and rows[0]["R"] == pytest.approx(abs(rs[1]) ** 2)


def test_pipeline_seam_honours_the_passed_claddings_in_both_paths(monkeypatch):
    """make_fem_2d_tm_optical_solver: the per-wavelength call and solve_sweep both hand the caller's
    claddings, order, mesh spec and threads to solve_design_2d_tm (the sweep path used to re-derive
    the design's claddings); the seam declares the native carrier grid and fingerprints its settings."""
    calls = []

    def fake(design, ebr, lam, **kw):
        calls.append((design, ebr, lam, kw))
        return lam
    monkeypatch.setattr(F, "solve_design_2d_tm", fake)
    spec = TMMeshSpec(corner_levels=3)
    seam = make_fem_2d_tm_optical_solver(order=3, mesh_spec=spec, num_threads=2)
    assert seam.native_carrier_grid is True
    assert "order=3" in seam.cache_fingerprint and repr(spec) in seam.cache_fingerprint
    assert seam("D", None, {"a": 1}, 1.25e-6, 1.0, 1.45) == 1.25e-6
    out = seam.solve_sweep("D", None, lambda lam: {"lam": lam}, [1.2e-6, 1.4e-6], 1.1, 1.6)
    assert out == [1.2e-6, 1.4e-6]
    want = dict(order=3, mesh_spec=spec, num_threads=2)
    assert calls[0] == ("D", {"a": 1}, 1.25e-6, dict(want, n_super=1.0, n_sub=1.45))
    assert calls[1] == ("D", {"lam": 1.2e-6}, 1.2e-6, dict(want, n_super=1.1, n_sub=1.6))
    assert calls[2] == ("D", {"lam": 1.4e-6}, 1.4e-6, dict(want, n_super=1.1, n_sub=1.6))


# ------------------------------------------------------------------------------------------------
# the Design adapter
# ------------------------------------------------------------------------------------------------

P_A, W_A = 400e-9, 160e-9


def _design(incl_shape=None, eps_film=4.0, optical=None):
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0), ("glass", 1.45 ** 2), ("oxB", 2.00 ** 2), ("metal", -60 + 4j),
                  ("mirror_metal", -120 + 20j), ("FILM", eps_film)):
        reg.add(Material(nm, ConstantOptical(complex(e))))
    shape = incl_shape if incl_shape is not None else Rectangle(P_A / 2, P_A / 2, W_A, P_A)
    layers = [Layer("mirror", 80e-9, "mirror_metal"), Layer("lo", 6e-9, "oxB"), Layer("film", 4e-9, "FILM"),
              Layer("grating", 40e-9, "air", inclusions=[Inclusion(shape, "metal")])]
    return Design(name="strip", unit_cell=UnitCell(period_x_m=P_A, period_y_m=P_A),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg, optical=optical or OpticalSpec(polarization="x"))


def test_design_adapter_translates_layers_blocks_and_claddings():
    """design_to_tm_layers: period, layers bottom first with their materials' eps, the Rectangle as a
    block over its x extent named '<layer>__incl<j>', and the end-media indices."""
    P, lays, ns, nb = design_to_tm_layers(_design(), None, LAM)
    assert P == P_A and ns == 1.0 and nb == pytest.approx(1.45, abs=1e-12)
    assert [L.name for L in lays] == ["mirror", "lo", "film", "grating"]
    assert [L.thickness_m for L in lays] == [80e-9, 6e-9, 4e-9, 40e-9]
    assert lays[0].eps == -120 + 20j and lays[2].eps == 4.0 and lays[3].eps == 1.0
    (blk,) = lays[3].blocks
    assert blk.name == "grating__incl0" and blk.eps == -60 + 4j
    assert blk.x_lo_m == pytest.approx(P_A / 2 - W_A / 2) and blk.x_hi_m == pytest.approx(P_A / 2 + W_A / 2)


def test_design_adapter_eps_overrides():
    """eps_by_region overrides a layer's background: a scalar EpsField as is, a uniform 3x3 tensor as
    its x-z block, a gridded y-invariant field as a GriddedEps2D with z from the layer bottom (a
    descending axis flipped) and the endpoint-duplicated x sample dropped."""
    from dynameta.core.eps_field import EpsField
    d = _design()
    _, lays, _, _ = design_to_tm_layers(d, {"film": EpsField(scalar=4.5 + 0.3j)}, LAM)
    assert lays[2].eps == 4.5 + 0.3j
    t = np.array([[2.0, 0, 0.1], [0, 2.5, 0], [0.2, 0, 3.0 + 0.1j]], dtype=complex)
    _, lays, _, _ = design_to_tm_layers(d, {"film": EpsField(tensor=t)}, LAM)
    assert np.array_equal(lays[2].eps, np.array([[2.0, 0.1], [0.2, 3.0 + 0.1j]]))
    ty = t.copy()
    ty[0, 1] = 0.3
    with pytest.raises(ValueError, match="couples y"):
        design_to_tm_layers(d, {"film": EpsField(tensor=ty)}, LAM)
    z = np.array([5.0, 2.5, 0.0])                                    # nm, descending
    x = np.linspace(0.0, P_A * 1e9, 5)                               # endpoint-inclusive
    v = (np.array([3.0, 2.0, 1.0])[:, None, None] * (1.0 + 0.1 * np.arange(5))[None, None, :]).astype(complex)
    ef = EpsField(values_zyx=v, z_axis_u=z, x_axis_u=x, y_axis_u=np.array([0.0]))
    _, lays, _, _ = design_to_tm_layers(d, {"film": ef}, LAM)
    g = lays[2].eps
    assert isinstance(g, GriddedEps2D)
    assert np.allclose(g.z_m, [0.0, 2.5e-9, 5e-9], rtol=0, atol=1e-21)
    # the adapter divides the nm axis by 1e9, this oracle multiplies by 1e-9: equal to an ulp
    assert np.allclose(g.x_m, x[:-1] * 1e-9, rtol=1e-15, atol=0) and g.values.shape == (3, 4)
    assert np.allclose(g.values[:, 0], [1.0, 2.0, 3.0])
    vz = v[:, :, :1]
    efz = EpsField(values_zyx=vz, z_axis_u=z, x_axis_u=x[:1], y_axis_u=np.array([0.0]))
    gz = design_to_tm_layers(d, {"film": efz}, LAM)[1][2].eps
    assert gz.values.shape == (3,) and gz.x_m is None
    vy = np.concatenate([v, 2 * v], axis=1)
    efy = EpsField(values_zyx=vy, z_axis_u=z, x_axis_u=x, y_axis_u=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="varies along y"):
        design_to_tm_layers(d, {"film": efy}, LAM)
    vt = np.zeros(v.shape + (3, 3), complex)
    vt[..., 0, 0] = vt[..., 1, 1] = vt[..., 2, 2] = v
    vt[..., 0, 2] = 0.05
    gt = design_to_tm_layers(d, {"film": EpsField(values_zyx=vt, z_axis_u=z, x_axis_u=x,
                                                 y_axis_u=np.array([0.0]))}, LAM)[1][2].eps
    assert gt.values.shape == (3, 4, 2, 2) and np.allclose(gt.values[..., 0, 1], 0.05)
    vt[..., 1, 2] = 0.05
    with pytest.raises(ValueError, match="couples y"):
        design_to_tm_layers(d, {"film": EpsField(values_zyx=vt, z_axis_u=z, x_axis_u=x,
                                                y_axis_u=np.array([0.0]))}, LAM)
    with pytest.raises(ValueError, match="values_zyx must be"):
        design_to_tm_layers(d, {"film": EpsField(values_zyx=np.ones((3, 1, 4, 2)), z_axis_u=z,
                                                x_axis_u=x[:4], y_axis_u=np.array([0.0]))}, LAM)


def test_design_adapter_refusals():
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry.cross_section import Circle, Rectangle
    with pytest.raises(ValueError, match="positive wavelength"):
        design_to_tm_layers(_design(), None, None)
    with pytest.raises(ValueError, match="not a lamellar Rectangle"):
        design_to_tm_layers(_design(Circle(P_A / 2, P_A / 2, 50e-9)), None, LAM)
    with pytest.raises(ValueError, match="full y period"):
        design_to_tm_layers(_design(Rectangle(P_A / 2, P_A / 2, W_A, P_A / 2)), None, LAM)
    with pytest.raises(NotImplementedError, match="nonlocal"):
        design_to_tm_layers(_design(), {"film": SimpleNamespace(beta=1e6, wp=1e16)}, LAM)
    bad = EpsField(scalar=4.0, time_convention="exp(+iwt)")
    with pytest.raises(ValueError, match="time_convention"):
        solve_design_2d_tm(_design(), {"film": bad}, LAM)
    from dynameta.geometry.specs import OpticalSpec
    with pytest.raises(NotImplementedError, match="conical"):           # refused before any mesh
        solve_design_2d_tm(_design(optical=OpticalSpec(polarization="p", incidence_angle_deg=20.0,
                                                       azimuth_deg=30.0)), None, LAM)


def test_design_incidence():
    """Incidence from design.optical: angle and polarization; bottom-side and conical incidence are
    not this 2-D problem."""
    def d(**kw):
        return SimpleNamespace(optical=SimpleNamespace(**kw))
    assert F._design_incidence(d(polarization="x")) == (0.0, "x")
    assert F._design_incidence(d(incidence_angle_deg=25.0, polarization="p")) == (25.0, "p")
    with pytest.raises(NotImplementedError, match="top-side"):
        F._design_incidence(d(incidence_side="bottom"))
    with pytest.raises(NotImplementedError, match="conical"):
        F._design_incidence(d(azimuth_deg=10.0))


def test_mesher_stub_leaves_no_trace():
    """Hygiene guard for this file's recorder: once the tests that used it are done, sys.modules
    holds no stand-in that a later test (or a solver-leg importorskip) could pick up."""
    assert getattr(sys.modules.get("netgen.meshing"), "Mesh", None) is not _RecMesh
    assert not isinstance(getattr(sys.modules.get("ngsolve"), "Mesh", None), type(lambda: 0))
