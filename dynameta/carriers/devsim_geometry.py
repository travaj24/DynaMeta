"""Exact (signed) box-method geometry for DEVSIM 2-D / 3-D regions -- an OPT-IN workaround for the
DEVSIM 2.11.2 unsigned-dual defect (v0.14.0, docs/audit/2026-11-06-devsim-geometry-workaround.md).

THE DEFECT (verified 2026-11-04 against the DEVSIM source, tag v2.11.2rc5). DEVSIM builds every
box-method quantity from ONE element-level model, ElementEdgeCouple (the area / length of the piece of
an edge's circumcentric dual facet inside one element):

    2-D  TriangleEdgeCouple.cc:74      c_e = |m_e - c_T|                       (edge midpoint to circumcentre)
    3-D  TetrahedronEdgeCouple.cc:99   c_e = 0.5 (|v0 x v1| + |v0 x v2|)       (two right triangles)

and derives the rest from it: ElementNodeVolume = c_e L_e / 4 (2-D) or / 6 (3-D), EdgeCouple = sum over
the elements of an edge, EdgeNodeVolume and NodeVolume = sums of ElementNodeVolume. Both formulas take
MAGNITUDES. A circumcentric dual piece has an orientation: it is negative when the element's (or the
face's) circumcentre lies on the far side of the edge (face), and a Voronoi / perpendicular-bisector
dual needs it to subtract. With magnitudes the pieces only add, so on any element that is not
well-centred the control volumes over-count -- NOT only on non-Delaunay meshes: one obtuse triangle
gives sum(NodeVolume) = 11.1 x its area, the trirectangular corner tetrahedron 2 x, the co-spherical
5-tet split of the unit cube 1.667 x, jittered Delaunay cubes 1.24-1.26 x (and a parallel-plate C 1.24 x
too large). It is exact only where every piece is >= 0: rectilinear right triangles (the 2-D builder)
and path ("Kuhn") tetrahedra (devsim_3d mesher='tensor'), which is why DynaMeta's own meshes and DEVSIM's
regression meshes do not see it. 3-D SurfaceArea (SurfaceAreaUtil.cc processTriangle) splits boundary
triangles the same way (magnitudes of the three circumcentric quarter-pieces), so interface / contact
areas over-count on obtuse surface triangles; the 2-D SurfaceArea (half edge lengths) is exact.

WHAT THIS MODULE DOES (apply_exact_geometry). It recomputes, from the node coordinates and the element
connectivity DEVSIM reports, the SIGNED version of exactly DEVSIM's construction -- the same edge
midpoints, face and element circumcentres and the same right-triangle pieces, each with its orientation
sign -- and writes it into DEVSIM's OWN built-in ElementEdgeCouple model (set_element_values). DEVSIM's
dependency chain then rebuilds EdgeCouple, ElementNodeVolume, EdgeNodeVolume and NodeVolume from it with
its own code (RegisterCallback("ElementEdgeCouple") in EdgeCouple.cc / TetrahedronNodeVolume.cc /
TriangleNodeVolume.cc, then EdgeNodeVolume.cc and NodeVolume.cc), and in 3-D it overwrites SurfaceArea
and ContactSurfaceArea with the signed circumcentric shares of the interface / contact triangles. With
signed pieces the element pieces partition every element exactly (sum NodeVolume = region volume to
round-off on ANY conforming mesh), interior edge couples / node volumes are the Voronoi facet areas /
cell volumes on a Delaunay mesh, and the box method reproduces a linear potential exactly (patch test:
every interior dual cell is a closed polyhedron with facets normal to their edges). In 2-D the signed
couple is the cotangent weight, c_e = L_e cot(theta_e) / 2, so the 2-D box Laplacian IS the P1 FEM one.

WHY THE BUILT-INS AND NOT DEVSIM'S REGION PARAMETERS. DEVSIM also lets a region point its assembly at
user models (set_parameter node_volume_model / edge_couple_model / edge_node0_volume_model /
edge_node1_volume_model / element_edge_couple_model / element_node0_volume_model /
element_node1_volume_model -- the cylindrical mechanism; Region.cc:1457-1534, read by Equation.cc and
ExprContactEquation.cc, MEASURED honoured in 2.11.2 for 2-D and 3-D). That route corrects only what
DEVSIM assembles: every DynaMeta kernel that reads the geometry by its built-in name would keep the
wrong values -- the per-element Hooge source (noise_analysis: ElementEdgeCouple, EdgeCouple,
NodeVolume), the 3-D impact-ionization quadrature (ElementNodeVolume), the interface-trap weight
ItArea = A/NodeVolume (physics_traps), BTBT and tunneling (ElementNodeVolume, EdgeNodeVolume,
SurfaceArea). Overwriting the built-in values corrects DEVSIM's assembly AND all of those at once, so the
discretisation and every post-processor stay consistent. The only other DEVSIM reader of
ElementEdgeCouple is the 2-D element-field reconstruction (TriangleElementField.cc GetFieldPairs:
element_from_edge_model / element_pair_from_edge_model weight each edge's two node-based vectors by the
couples of the two OTHER edges of the triangle). With signed couples those two weights still sum to
R (cos theta_j + cos theta_k) = 2 R sin(theta_i/2) cos((theta_j - theta_k)/2) > 0 (no division by
zero), and for a field that is the gradient of the P1 interpolant (ElectricField) every node-based vector
IS that gradient, so the reconstruction is unchanged; for other edge fields (currents) it moves on
obtuse triangles only, between two first-order-consistent averages. The 3-D element field
(TetrahedronElementField.cc) does not use couples. The 3-D "COUPLE" node interpolation
(TetrahedronEdgeModel.cc:199, element values shown on nodes for output) does, and skips a node whose
couple sum is <= 0.

WHAT IT DOES NOT FIX: element-TENSOR consumers in 3-D. The signed pieces make every ASSEMBLED quantity
exact, and per element they still partition the volume (sum_e c_eT L_e / 3 = V_T). They do NOT satisfy the
per-element tensor identity sum_e c_eT L_e t_e t_e^T = V_T I in 3-D (only its trace) -- not even on a Kuhn
tet (diag(0.75, 1.5, 0.75) V_T for the path x, y, z) or a well-centred one (eigenvalues 0.88-1.11 V_T,
MEASURED). That identity holds per element for the P1-FEM edge weights -V_T L_e grad(l_i).grad(l_j), which
equal the circumcentric couples in 2-D only. A kernel that squares a per-element sum (noise_analysis's
per-element Hooge source) is therefore biased in 3-D on every mesh: MEASURED S f N / (alpha I^2) = 1.125 on
the devsim_3d Kuhn bar (validation/devsim_noise.unipolar_3d_resistor), 1.000 on a 5-tet bar, with or
without this correction (audit note).

WHEN TO USE IT. On any user mesh that is not made of rectilinear right triangles (2-D) or path
tetrahedra (3-D): imported gmsh meshes, Delaunay meshes, devsim_3d mesher='gmsh'. Call it once per
region right after create_device (or the builder's mesh build) and BEFORE any physics setup that reads
the geometry (install_interface_traps computes ItArea from NodeVolume at install time), and re-apply
after anything that makes DEVSIM rebuild its geometry (adding a contact / interface, reloading a device).
On a mesh where DEVSIM is already exact it changes nothing beyond round-off (gate:
tests/test_devsim_geometry.py). 1-D regions are exact already (returned untouched). Cylindrical or other
user geometry models already selected on the region are refused: not covered here (DEVSIM's 2-D
cylindrical couple takes the same magnitude, TriangleCylindricalEdgeCouple.cc:153; no DynaMeta module
uses DEVSIM's cylindrical mode).

THE BOUNDARY CAVEAT -- negative couples / volumes, and the `negative=` policy. Per-ELEMENT pieces of a
non-well-centred element are negative by construction; that is expected and harmless (they cancel in the
sums). What matters is the ASSEMBLED quantities:
  * EdgeCouple < 0 on an interior edge means a non-Delaunay edge; on a boundary edge it means the dual
    facet is clipped by a boundary that is not "conforming" (2-D: the angle opposite a boundary edge is
    obtuse; 3-D: a non-Gabriel boundary face, slivers at the boundary). A negative couple flips the sign
    of an off-diagonal of the Scharfetter-Gummel Jacobian: the discrete maximum principle (M-matrix)
    is lost, so densities can undershoot / go negative and Newton can stall. (Poisson alone stays
    consistent -- in 2-D it is the P1 FEM matrix, which is a valid convergent discretisation.)
  * NodeVolume < 0 (at boundary nodes of such meshes) flips the sign of that node's charge /
    recombination / time-derivative term.
  * SurfaceArea < 0 (3-D, an obtuse interface / contact triangle at the edge of the surface).
DEVSIM's magnitudes hide these by making everything positive -- at the price of the wrong volumes. The
policy, applied BEFORE anything is written:
  'raise' (DEFAULT): refuse with NegativeDualGeometryError (counts, worst values; `.report`), leaving the
          device unchanged, so the first call on a new mesh shows how many negatives it has instead of
          silently choosing for the caller. A boundary-conforming (Gabriel / non-obtuse-boundary)
          Delaunay remesh removes them and is the only fix that keeps every property.
  'warn'  (RECOMMENDED once the report has been read): write the exact signed values anyway and emit
          UnreliableResultWarning: volumes and the patch test are exact, the M-matrix property is not
          guaranteed. MEASURED on the legacy devsim_3d gmsh stack (validation/devsim_geometry.py; ~1.3 %
          of the edges negative, worst -0.12 L^2, no negative node volumes): the series capacitor is
          exact to 6e-16 (DEVSIM's geometry: 1.203 x), the bipolar FD-SG MOS cap converges and its Gauss
          residual drops from 0.228 to 1.3e-5 -- 60 x better than 'clip'.
  'clip'  : set every negative assembled EdgeCouple to 0 (and all element pieces of that edge, so the
          edge and element assembly agree), keep ElementNodeVolume as the exact signed partition, and
          clip negative NodeVolume / SurfaceArea to 0. Keeps the M-matrix property; costs the patch test
          on the clipped edges and the volume the clipped nodes carried -- both MEASURED and returned
          (ExactGeometryReport.clipped_*), and warned as UnreliableResultWarning. MEASURED on the same
          gmsh stack: capacitor 6.8e-4 off, Gauss residual 7.6e-4. The fallback when a drift-diffusion
          Newton does not converge under 'warn'; useless on pathological meshes (a random-point Delaunay
          square: the clipped node volumes add 14 x the region volume).
A tolerance of 1e-9 x the local length scale (L_e^(d-1) for a couple, h^d for a node volume, h the node's
longest edge) separates a genuine negative from round-off of a zero piece (the face / body diagonals of a
Kuhn tet and the hypotenuses of right triangles carry exactly 0, computed as +-1e-17 relative): the
circumcentre solve loses ~ eps x cond, cond ~ 1/sin(smallest angle) <= 1e6 on any mesh DEVSIM can
solve, so its error is < 2e-10 relative. Values inside the tolerance are written as computed.

ROUND-OFF / PRECISION. The signed values are computed in float64 from the float64 node coordinates
DEVSIM returns and written as float64; on an extended-precision DEVSIM build (geometry models built in
float128) the corrected geometry therefore carries float64 round-off (~1e-16 relative) instead of
float128's. measure_dual_geometry() reports everything without changing the device (a diagnostic for the
shared well-centredness check, see exact_geometry_applied()).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np

from dynameta.warning_types import UnreliableResultWarning

__all__ = ["apply_exact_geometry", "measure_dual_geometry", "exact_geometry_applied",
           "geometry_already_exact", "correct_device_geometry", "resolve_policy",
           "ExactGeometryReport", "NegativeDualGeometryError", "triangle_dual_pieces",
           "tetrahedron_dual_pieces", "MARKER_PARAMETER"]

MARKER_PARAMETER = "dynameta_exact_geometry"     # region parameter set by apply_exact_geometry
NEGATIVE_POLICIES = ("raise", "warn", "clip")
_NEG_RTOL = 1e-9                                 # genuine negative vs round-off (module doc)
_LAYOUT_RTOL = 1e-9                              # DEVSIM unsigned values vs our |pieces| re-derivation
# ALREADY-EXACT bound (geometry_already_exact): DEVSIM's |piece| equals the signed piece unless the piece is
# negative; a piece that is mathematically ZERO (Kuhn face / body diagonals, right-triangle hypotenuses)
# comes out of the circumcentre arithmetic as +-(a few eps) h_T^(d-1), h_T the element's longest edge
# (MEASURED <= 1.15 eps h^k on the Kuhn / rectilinear gates). 64 eps leaves a 50x margin over that and is
# still 7 decades below the smallest genuine negative measured anywhere (-1.5e-4 L^2, a two-region
# Delaunay test mesh).
_EXACT_EPS = 64.0 * np.finfo(np.float64).eps
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_GEOM_PARAMS = {"node_volume_model": "NodeVolume", "edge_couple_model": "EdgeCouple",
                "element_edge_couple_model": "ElementEdgeCouple",
                "edge_node0_volume_model": "EdgeNodeVolume", "edge_node1_volume_model": "EdgeNodeVolume",
                "element_node0_volume_model": "ElementNodeVolume",
                "element_node1_volume_model": "ElementNodeVolume"}


class NegativeDualGeometryError(ValueError):
    """The exact (signed) box-method geometry of a region has negative ASSEMBLED edge couples, node
    volumes or surface areas (non-Delaunay edges or a non-conforming boundary; module doc). Raised by
    apply_exact_geometry(negative='raise') before the device is modified. `.report` carries the
    ExactGeometryReport with the counts and worst values."""

    def __init__(self, msg: str, report: "ExactGeometryReport"):
        super().__init__(msg)
        self.report = report


@dataclass
class ExactGeometryReport:
    """What apply_exact_geometry / measure_dual_geometry measured on one region (all MEASURED from the
    node coordinates; volumes in m^d, couples in m^(d-1), areas in m^(d-1))."""
    device: str
    region: str
    dimension: int
    n_elements: int = 0
    n_not_well_centred: int = 0          # elements with at least one negative signed piece
    true_volume: float = 0.0             # sum |det| / d! over the elements
    devsim_volume_ratio: float = 1.0     # DEVSIM's own sum(NodeVolume) / true_volume (before)
    exact_volume_ratio: float = 1.0      # sum of the signed node volumes / true_volume (== 1)
    n_negative_couples: int = 0          # assembled EdgeCouple < -tol
    min_couple_rel: float = 0.0          # most negative assembled couple / L_e^(d-1)
    negative_couple_edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    n_negative_volumes: int = 0          # assembled NodeVolume < -tol
    min_volume_rel: float = 0.0          # most negative node volume / h^d
    negative_volume_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    surface_true_area: float = 0.0       # 3-D: interface + contact triangle area (each triangle once)
    devsim_surface_ratio: float = 1.0    # 3-D: DEVSIM's (SurfaceArea + ContactSurfaceArea) / true
    n_negative_areas: int = 0            # 3-D: signed SurfaceArea / ContactSurfaceArea shares < -tol
    applied: bool = False                # the region's geometry is exact after the call
    written: bool = False                # apply_exact_geometry wrote DEVSIM models (False: already exact)
    policy: Optional[str] = None
    clipped_couple_fraction: float = 0.0  # clip: sum |clipped couples| / sum |couples|
    clipped_volume_fraction: float = 0.0  # clip: sum |clipped node volumes| / true_volume
    clipped_area_fraction: float = 0.0    # clip: sum |clipped areas| / surface_true_area

    @property
    def has_negatives(self) -> bool:
        return bool(self.n_negative_couples or self.n_negative_volumes or self.n_negative_areas)

    def summary(self) -> str:
        s = ("region {!r} ({}-D, {} elements, {} not well-centred): DEVSIM sum(NodeVolume) = {:.6g} x the "
             "true volume, exact signed = {:.15g} x".format(self.region, self.dimension, self.n_elements,
                                                           self.n_not_well_centred, self.devsim_volume_ratio,
                                                           self.exact_volume_ratio))
        if self.dimension == 3 and self.surface_true_area > 0.0:
            s += "; DEVSIM surface area = {:.6g} x the true".format(self.devsim_surface_ratio)
        if self.has_negatives:
            s += ("; NEGATIVE assembled geometry: {} edge couples (worst {:.3g} L^(d-1)), {} node volumes "
                  "(worst {:.3g} h^d), {} surface shares".format(
                      self.n_negative_couples, self.min_couple_rel, self.n_negative_volumes,
                      self.min_volume_rel, self.n_negative_areas))
        return s


# --------------------------------------------------------------------------------------------------
# kernels (numpy only; one formula each -- the triangle kernel is reused by the tetrahedron one)
# --------------------------------------------------------------------------------------------------

def triangle_dual_pieces(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed circumcentric dual pieces of triangles (2-D, or embedded in 3-D).

    X: (n, 3, 2|3) vertex coordinates. Returns (s, L, c): s[t, j] the SIGNED distance from the midpoint
    of edge j (the edge OPPOSITE vertex j) to the circumcentre c[t], positive toward vertex j (into the
    triangle); L[t, j] that edge's length; c (n, 3) the circumcentres. |s| is DEVSIM's 2-D
    ElementEdgeCouple (TriangleEdgeCouple.cc:74); s = L cot(theta_j) / 2 = R cos(theta_j). Each endpoint
    of edge j owns s L / 4 of the triangle (the 2-D ElementNodeVolume, and the 3-D SurfaceArea share)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[-1] == 2:
        X = np.concatenate([X, np.zeros(X.shape[:-1] + (1,))], axis=-1)
    a = X[:, 0]
    ab, ac = X[:, 1] - a, X[:, 2] - a
    nrm = np.cross(ab, ac)
    n2 = np.einsum("ij,ij->i", nrm, nrm)
    c = a + ((np.einsum("ij,ij->i", ac, ac)[:, None] * np.cross(nrm, ab)
              + np.einsum("ij,ij->i", ab, ab)[:, None] * np.cross(ac, nrm)) / (2.0 * n2[:, None]))
    s = np.empty(X.shape[:2])
    L = np.empty(X.shape[:2])
    for j in range(3):
        p, q, o = X[:, (j + 1) % 3], X[:, (j + 2) % 3], X[:, j]
        m = 0.5 * (p + q)
        t = q - p
        t2 = np.einsum("ij,ij->i", t, t)
        u = o - m
        u = u - (np.einsum("ij,ij->i", u, t) / t2)[:, None] * t     # in-plane, normal to the edge, inward
        u /= np.linalg.norm(u, axis=1)[:, None]
        s[:, j] = np.einsum("ij,ij->i", c - m, u)
        L[:, j] = np.sqrt(t2)
    return s, L, c


def tetrahedron_dual_pieces(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed and unsigned circumcentric edge couples of tetrahedra.

    X: (n, 4, 3). Returns (signed, unsigned, L), each (n, 6) over the local edges _TET_EDGES
    ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)). The couple of edge ij in a tet is the area of the part of its
    dual facet inside the tet: two right triangles (edge midpoint m, face circumcentre c_f, tet
    circumcentre c_T), one per face f = (i, j, k) containing the edge, each of area 0.5 s1 s2 with
    s1 = the signed in-face distance m -> c_f (positive toward k; triangle_dual_pieces) and s2 = the
    signed distance c_f -> c_T along the face normal (positive toward the tet's fourth vertex l).
    `unsigned` is DEVSIM's 0.5 (|v0 x v1| + |v0 x v2|) = 0.5 sum |s1| |s2| (TetrahedronEdgeCouple.cc:99;
    v0 = c_T - m, v1/v2 = c_f - c_T are orthogonal decompositions of the same triangles)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    A = X[:, 1:] - X[:, :1]
    cT = X[:, 0] + np.linalg.solve(A, 0.5 * np.einsum("nij,nij->ni", A, A)[..., None])[..., 0]
    # face f = the face opposite vertex f; its vertices in increasing local order
    s1 = {}                                       # (face, i, j) -> signed in-face distance, (n,)
    s2 = np.empty((n, 4))
    for f in range(4):
        vs = [v for v in range(4) if v != f]
        s, _L, cf = triangle_dual_pieces(X[:, vs])
        for jj in range(3):                       # edge opposite vs[jj] within the face
            e = tuple(sorted(v for k, v in enumerate(vs) if k != jj))
            s1[(f,) + e] = s[:, jj]
        nrm = np.cross(X[:, vs[1]] - X[:, vs[0]], X[:, vs[2]] - X[:, vs[0]])
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        sgn = np.sign(np.einsum("ij,ij->i", X[:, f] - X[:, vs[0]], nrm))
        s2[:, f] = sgn * np.einsum("ij,ij->i", cT - cf, nrm)
    signed = np.empty((n, 6))
    unsigned = np.empty((n, 6))
    L = np.empty((n, 6))
    for k, (i, j) in enumerate(_TET_EDGES):
        f_a, f_b = [v for v in range(4) if v not in (i, j)]   # face opposite f_a contains i, j, f_b
        pa, pb = s1[(f_a, i, j)] * s2[:, f_a], s1[(f_b, i, j)] * s2[:, f_b]
        signed[:, k] = 0.5 * (pa + pb)
        unsigned[:, k] = 0.5 * (np.abs(pa) + np.abs(pb))
        L[:, k] = np.linalg.norm(X[:, j] - X[:, i], axis=1)
    return signed, unsigned, L


# --------------------------------------------------------------------------------------------------
# DEVSIM plumbing
# --------------------------------------------------------------------------------------------------

def _ds():
    from dynameta.carriers.devsim_env import import_devsim
    return import_devsim()


def _arr(vals) -> np.ndarray:
    return np.asarray(vals, dtype=np.float64)


def exact_geometry_applied(device: str, region: str) -> bool:
    """True iff apply_exact_geometry wrote the exact geometry into this region (it sets the region
    parameter MARKER_PARAMETER). For the shared well-centredness check: on such a region DEVSIM's
    ElementNodeVolume / NodeVolume are the exact signed partition, so the exact-volume test passes."""
    ds = _ds()
    # region-level only: get_parameter would fall back to a device / global value
    if MARKER_PARAMETER not in ds.get_parameter_list(device=device, region=region):
        return False
    return bool(ds.get_parameter(device=device, region=region, name=MARKER_PARAMETER))


class _Region:
    """Coordinates, elements and the element-edge slot layout of one DEVSIM region (all DEVSIM ids)."""

    def __init__(self, device: str, region: str):
        ds = _ds()
        self.device, self.region = device, region
        self.dim = int(ds.get_dimension(device=device))
        self.xyz = np.column_stack([_arr(ds.get_node_model_values(device=device, region=region, name=c))
                                    for c in ("x", "y", "z")])
        self.n_nodes = self.xyz.shape[0]
        self.n_edges = _arr(ds.get_edge_model_values(device=device, region=region, name="EdgeLength")).size
        if self.dim == 1:
            return
        self.elems = np.asarray(ds.get_element_node_list(device=device, region=region), dtype=np.int64)
        k = 3 if self.dim == 2 else 6
        self.k = k
        # per element-edge slot: the global edge id and both end nodes (temporary element models)
        have = set(ds.get_element_model_list(device=device, region=region))
        tmp_e = "_dm_geom_edge_index"
        made = [tmp_e]
        try:
            ds.element_model(device=device, region=region, name=tmp_e, equation="edge_index")
            if "node_index@en0" not in have:
                ds.element_from_node_model(device=device, region=region, node_model="node_index")
                made += [m for m in ds.get_element_model_list(device=device, region=region)
                         if m.startswith("node_index@en") and m not in have]
            eid = _arr(ds.get_element_model_values(device=device, region=region, name=tmp_e))
            e0 = _arr(ds.get_element_model_values(device=device, region=region, name="node_index@en0"))
            e1 = _arr(ds.get_element_model_values(device=device, region=region, name="node_index@en1"))
        finally:
            left = set(ds.get_element_model_list(device=device, region=region))
            for m in made:
                if m in left:
                    ds.delete_element_model(device=device, region=region, name=m)
        n_el = self.elems.shape[0]
        self.slot_edge = np.rint(eid).astype(np.int64).reshape(n_el, k)
        e0 = np.rint(e0).astype(np.int64).reshape(n_el, k)
        e1 = np.rint(e1).astype(np.int64).reshape(n_el, k)
        # local vertex positions of each slot's end nodes, CHECKED (DEVSIM layout, not trusted)
        la = np.argmax(self.elems[:, :, None] == e0[:, None, :], axis=1)
        lb = np.argmax(self.elems[:, :, None] == e1[:, None, :], axis=1)
        ok = (np.all(np.take_along_axis(self.elems, la, 1) == e0)
              and np.all(np.take_along_axis(self.elems, lb, 1) == e1) and np.all(la != lb))
        if not ok:                                                 # pragma: no cover - DEVSIM invariant
            raise RuntimeError("devsim_geometry: DEVSIM's element-edge layout on region {!r} is not the "
                               "documented one; refusing to guess.".format(region))
        lo, hi = np.minimum(la, lb), np.maximum(la, lb)
        if self.dim == 2:                                          # local edge index = the opposite vertex
            self.slot_local = 3 - lo - hi
        else:
            table = {e: i for i, e in enumerate(_TET_EDGES)}
            lut = np.full((4, 4), -1, dtype=np.int64)
            for (i, j), v in table.items():
                lut[i, j] = v
            self.slot_local = lut[lo, hi]
        self.slot_nodes = np.stack([e0, e1], axis=-1)              # (n_el, k, 2)


def _signed_slots(R: _Region):
    """(signed, unsigned, L) per element-edge slot, (n_el, k), plus per-element volumes."""
    X = R.xyz[R.elems]
    if R.dim == 2:
        s, L, _c = triangle_dual_pieces(X[:, :, :2])
        sg, us = s, np.abs(s)
        d = X[:, 1:, :2] - X[:, :1, :2]
        vol = 0.5 * np.abs(d[:, 0, 0] * d[:, 1, 1] - d[:, 0, 1] * d[:, 1, 0])
    else:
        sg, us, L = tetrahedron_dual_pieces(X)
        vol = np.abs(np.linalg.det(X[:, 1:] - X[:, :1])) / 6.0
    take = lambda a: np.take_along_axis(a, R.slot_local, axis=1)  # noqa: E731
    return take(sg), take(us), take(L), vol


def _surfaces(device: str, region: str, xyz: np.ndarray):
    """3-D: signed / unsigned circumcentric node shares of the interface triangles (SurfaceArea) and of
    the contact triangles (ContactSurfaceArea) of `region`, each triangle once (as DEVSIM dedups)."""
    ds = _ds()
    out = {}
    for kind in ("interface", "contact"):
        names = (ds.get_interface_list(device=device) if kind == "interface"
                 else ds.get_contact_list(device=device))
        tris = {}
        for nm in names:
            if region not in ds.get_region_list(device=device, **{kind: nm}):
                continue
            for t in ds.get_element_node_list(device=device, region=region, **{kind: nm}):
                tris.setdefault(tuple(sorted(int(i) for i in t)), tuple(int(i) for i in t))
        sg = np.zeros(xyz.shape[0])
        us = np.zeros(xyz.shape[0])
        area = 0.0
        if tris:
            T = np.asarray(list(tris.values()), dtype=np.int64)
            s, L, _c = triangle_dual_pieces(xyz[T])
            area = float(0.5 * np.linalg.norm(np.cross(xyz[T[:, 1]] - xyz[T[:, 0]],
                                                       xyz[T[:, 2]] - xyz[T[:, 0]]), axis=1).sum())
            for j in range(3):
                for v in ((j + 1) % 3, (j + 2) % 3):
                    np.add.at(sg, T[:, v], 0.25 * L[:, j] * s[:, j])
                    np.add.at(us, T[:, v], 0.25 * L[:, j] * np.abs(s[:, j]))
        out[kind] = (sg, us, area)
    return out


def _check_default_models(device: str, region: str) -> None:
    ds = _ds()
    for p, default in _GEOM_PARAMS.items():
        v = ds.get_parameter(device=device, region=region, name=p)
        if v != default:
            raise ValueError(
                "apply_exact_geometry: region {!r} assembles with {}={!r} (cylindrical or user geometry "
                "models), which this correction does not cover; it corrects DEVSIM's Cartesian built-ins "
                "only.".format(region, p, v))


def geometry_already_exact(device: str, region: str) -> bool:
    """READ-ONLY test (only devsim get_* calls; nothing is created, set or deleted): True iff DEVSIM's
    built-in geometry of `region` already IS the exact signed one to round-off, so apply_exact_geometry has
    nothing to write. That is the case iff no circumcentric piece is negative beyond the round-off of a zero
    piece (_EXACT_EPS h_T^(d-1), h_T the element's longest edge) -- the DynaMeta library meshes (2-D
    rectilinear right triangles, 3-D Kuhn tets) -- and, in 3-D, the same holds for every interface /
    contact triangle share. DEVSIM's current ElementEdgeCouple must also re-derive from the coordinates
    (per element, as a multiset, to _LAYOUT_RTOL): a region whose values were already overwritten, or that
    does not match, returns False and goes through the full (checked) path. 1-D: True."""
    ds = _ds()
    dim = int(ds.get_dimension(device=device))
    if dim == 1:
        return True
    _check_default_models(device, region)
    xyz = np.column_stack([_arr(ds.get_node_model_values(device=device, region=region, name=c))
                           for c in ("x", "y", "z")])
    els = np.asarray(ds.get_element_node_list(device=device, region=region), dtype=np.int64)
    if dim == 2:
        s, L, _c = triangle_dual_pieces(xyz[els][:, :, :2])
        sg, us = s, np.abs(s)
    else:
        sg, us, L = tetrahedron_dual_pieces(xyz[els])
    hT = L.max(axis=1, keepdims=True) ** (dim - 1)
    cur = _arr(ds.get_element_model_values(device=device, region=region, name="ElementEdgeCouple"))
    if cur.size != us.size:                                     # pragma: no cover - DEVSIM invariant
        return False
    cur = np.sort(cur.reshape(us.shape), axis=1)
    if not np.all(np.abs(cur - np.sort(us, axis=1)) <= _LAYOUT_RTOL * hT):
        return False
    if not np.all(us - sg <= _EXACT_EPS * hT):
        return False
    if dim == 3:
        h2 = float(L.max()) ** 2
        for sg_a, us_a, _area in _surfaces(device, region, xyz).values():
            if not np.all(us_a - sg_a <= _EXACT_EPS * h2):
                return False
    return True


def _measure(device: str, region: str):
    ds = _ds()
    R = _Region(device, region)
    rep = ExactGeometryReport(device=device, region=region, dimension=R.dim)
    if R.dim == 1:
        L = _arr(ds.get_edge_model_values(device=device, region=region, name="EdgeLength"))
        rep.true_volume = float(L.sum())
        return R, rep, None
    _check_default_models(device, region)
    sg, us, L, vol = _signed_slots(R)
    d = R.dim
    vfac = 1.0 / (2.0 * d)                                       # ElementNodeVolume = c L / 4 | / 6
    rep.n_elements = int(R.elems.shape[0])
    tol_piece = _NEG_RTOL * L ** (d - 1)
    rep.n_not_well_centred = int(np.any(sg < -tol_piece, axis=1).sum())
    rep.true_volume = float(vol.sum())
    nv_dev = _arr(ds.get_node_model_values(device=device, region=region, name="NodeVolume"))
    rep.devsim_volume_ratio = float(nv_dev.sum() / rep.true_volume)
    env = sg * L * vfac
    nv = np.zeros(R.n_nodes)
    for side in (0, 1):
        np.add.at(nv, R.slot_nodes[:, :, side].ravel(), env.ravel())
    rep.exact_volume_ratio = float(nv.sum() / rep.true_volume)
    ec = np.bincount(R.slot_edge.ravel(), weights=sg.ravel(), minlength=R.n_edges)
    Le = _arr(ds.get_edge_model_values(device=device, region=region, name="EdgeLength"))
    neg_e = np.flatnonzero(ec < -_NEG_RTOL * Le ** (d - 1))
    rep.n_negative_couples = int(neg_e.size)
    rep.negative_couple_edges = neg_e
    rep.min_couple_rel = float(min(0.0, (ec / Le ** (d - 1)).min()))
    h = np.zeros(R.n_nodes)
    for side in (0, 1):
        np.maximum.at(h, R.slot_nodes[:, :, side].ravel(), L.ravel())
    neg_n = np.flatnonzero(nv < -_NEG_RTOL * h ** d)
    rep.n_negative_volumes = int(neg_n.size)
    rep.negative_volume_nodes = neg_n
    rep.min_volume_rel = float(min(0.0, (nv / np.where(h > 0, h, 1.0) ** d).min()))
    surf = None
    if d == 3:
        surf = _surfaces(device, region, R.xyz)
        tot = surf["interface"][2] + surf["contact"][2]
        rep.surface_true_area = tot
        if tot > 0.0:
            dev = (_arr(ds.get_node_model_values(device=device, region=region, name="SurfaceArea")).sum()
                   + _arr(ds.get_node_model_values(device=device, region=region,
                                                   name="ContactSurfaceArea")).sum())
            rep.devsim_surface_ratio = float(dev / tot)
            hs = np.where(h > 0, h, 1.0)
            rep.n_negative_areas = int(sum(np.count_nonzero(surf[k][0] < -_NEG_RTOL * hs ** 2)
                                           for k in ("interface", "contact")))
    return R, rep, dict(sg=sg, us=us, L=L, env=env, nv=nv, ec=ec, h=h, surf=surf)


def measure_dual_geometry(device: str, region: str) -> ExactGeometryReport:
    """Diagnostic only: the ExactGeometryReport of a region (DEVSIM's own volume / surface ratio, the
    number of non-well-centred elements, the negative assembled couples / volumes the exact geometry would
    have) WITHOUT changing the device. After apply_exact_geometry, devsim_volume_ratio reads the
    corrected values (== 1)."""
    return _measure(device, region)[1]


def apply_exact_geometry(device: str, region: str, *, negative: str = "raise") -> ExactGeometryReport:
    """OPT-IN: replace DEVSIM's unsigned box-method geometry of `region` by the exact signed one (module
    doc), so every equation DEVSIM assembles -- edge, element, node, contact and (3-D) interface
    terms -- and every DynaMeta kernel that reads NodeVolume / EdgeCouple / EdgeNodeVolume /
    ElementEdgeCouple / ElementNodeVolume / SurfaceArea uses the correct control volumes.

    Use it on any 2-D / 3-D user mesh that is not rectilinear-right-triangle / path-tetrahedron (gmsh,
    Delaunay, devsim_3d mesher='gmsh'); call it per region right after the device is created and before
    physics setup; re-apply after DEVSIM rebuilds its geometry. On meshes where DEVSIM is already exact
    it changes the geometry by round-off only.

    negative: what to do when the exact geometry has negative ASSEMBLED couples / node volumes / surface
      shares (non-Delaunay edges, non-conforming boundary): 'raise' (default: NegativeDualGeometryError
      with the counts in `.report`, device untouched), 'warn' (the recommended choice once the counts
      are known: the exact values, UnreliableResultWarning -- exact volumes + patch test, no M-matrix
      guarantee), 'clip' (negatives -> 0, exact element volumes kept; the clipped fractions are measured
      into the report and warned -- the fallback if a drift-diffusion Newton fails under 'warn'). See the
      module doc for the trade-offs and the measured gmsh-stack numbers.

    ALREADY-EXACT REGIONS ARE NOT TOUCHED: when geometry_already_exact() holds (the library meshes) the call
    makes only devsim get_* reads and returns with written=False, so a device built on them is
    byte-identical with and without the correction (gate: tests/test_devsim_geometry.py).

    Returns the ExactGeometryReport (applied=True; written says whether DEVSIM was modified). Raises ValueError for a region that already uses
    cylindrical / user geometry models, RuntimeError if DEVSIM's values do not re-derive from the
    coordinates (layout check) or do not propagate after the write (read-back check)."""
    if negative not in NEGATIVE_POLICIES:
        raise ValueError("apply_exact_geometry: negative must be one of {}, got {!r}"
                         .format(NEGATIVE_POLICIES, negative))
    ds = _ds()
    if not exact_geometry_applied(device, region) and geometry_already_exact(device, region):
        return _exact_report(device, region, negative)            # nothing to write: byte-identical
    R, rep, g = _measure(device, region)
    rep.policy = negative
    if R.dim == 1:                                               # pragma: no cover - caught above
        rep.applied = True
        return rep
    # layout check: DEVSIM's current ElementEdgeCouple must re-derive from the coordinates
    cur = _arr(ds.get_element_model_values(device=device, region=region, name="ElementEdgeCouple"))
    ref = (g["sg"] if exact_geometry_applied(device, region) else g["us"]).ravel()
    scale = (g["L"] ** (R.dim - 1)).ravel()
    if cur.size != ref.size or not np.all(np.abs(cur - ref) <= _LAYOUT_RTOL * scale):
        if not np.all(np.abs(cur - g["sg"].ravel()) <= _LAYOUT_RTOL * scale):
            raise RuntimeError(
                "devsim_geometry: DEVSIM's ElementEdgeCouple on region {!r} does not re-derive from the "
                "node coordinates (max |diff| / L^(d-1) = {:.3g}); refusing to overwrite it.".format(
                    region, float(np.max(np.abs(cur - ref) / scale)) if cur.size == ref.size else np.inf))
    if rep.has_negatives:
        where = ("{} assembled edge couples < 0 (worst {:.3g} x L^(d-1)), {} node volumes < 0 (worst {:.3g} "
                 "x h^d), {} surface shares < 0".format(rep.n_negative_couples, rep.min_couple_rel,
                                                        rep.n_negative_volumes, rep.min_volume_rel,
                                                        rep.n_negative_areas))
        if negative == "raise":
            raise NegativeDualGeometryError(
                "apply_exact_geometry: the exact box-method geometry of region {!r} has {} -- non-Delaunay "
                "edges or a non-conforming (obtuse / non-Gabriel) boundary. Negative couples break the "
                "M-matrix property Scharfetter-Gummel relies on. Remesh with a conforming boundary, or "
                "pass negative='warn' (exact values, Poisson-safe) or negative='clip' (measured volume "
                "error). The device is unchanged.".format(region, where), rep)
    sg, env = g["sg"].copy(), g["env"]
    clip = negative == "clip" and rep.has_negatives
    if clip:
        bad = np.zeros(R.n_edges, dtype=bool)
        bad[rep.negative_couple_edges] = True
        rep.clipped_couple_fraction = float(np.abs(g["ec"][bad]).sum() / max(np.abs(g["ec"]).sum(), 1e-300))
        sg[bad[R.slot_edge]] = 0.0
    ds.set_element_values(device=device, region=region, name="ElementEdgeCouple",
                          values=[float(v) for v in sg.ravel()])
    if clip:                                                     # keep the exact element volumes
        ds.set_element_values(device=device, region=region, name="ElementNodeVolume",
                              values=[float(v) for v in env.ravel()])
    # force DEVSIM's chain to rebuild NOW (a lazy rebuild after a direct NodeVolume write would undo it)
    ds.get_edge_model_values(device=device, region=region, name="EdgeCouple")
    ds.get_edge_model_values(device=device, region=region, name="EdgeNodeVolume")
    nv_dev = _arr(ds.get_node_model_values(device=device, region=region, name="NodeVolume"))
    nv = g["nv"]
    if clip and rep.n_negative_volumes:
        nv = nv.copy()
        nv[rep.negative_volume_nodes] = 0.0                      # only the genuine negatives
        rep.clipped_volume_fraction = float(np.abs(g["nv"][rep.negative_volume_nodes]).sum()
                                            / rep.true_volume)
        ds.set_node_values(device=device, region=region, name="NodeVolume", values=[float(v) for v in nv])
        nv_dev = _arr(ds.get_node_model_values(device=device, region=region, name="NodeVolume"))
    if R.dim == 3 and g["surf"] is not None and rep.surface_true_area > 0.0:
        ds.get_node_model_values(device=device, region=region, name="SurfaceArea")   # normals computed
        clipped = 0.0
        for kind, model in (("interface", "SurfaceArea"), ("contact", "ContactSurfaceArea")):
            a = g["surf"][kind][0]
            if clip:
                clipped += float(np.abs(a[a < 0.0]).sum())
                a = np.where(a < 0.0, 0.0, a)
            ds.set_node_values(device=device, region=region, name=model, values=[float(v) for v in a])
        rep.clipped_area_fraction = clipped / rep.surface_true_area
    # read-back: DEVSIM must now hold the values we derived
    ec_dev = _arr(ds.get_edge_model_values(device=device, region=region, name="EdgeCouple"))
    ec_ref = np.bincount(R.slot_edge.ravel(), weights=sg.ravel(), minlength=R.n_edges)
    Le = _arr(ds.get_edge_model_values(device=device, region=region, name="EdgeLength"))
    if not (np.all(np.abs(ec_dev - ec_ref) <= _LAYOUT_RTOL * Le ** (R.dim - 1))
            and np.all(np.abs(nv_dev - nv) <= _LAYOUT_RTOL * np.where(g["h"] > 0, g["h"], 1.0) ** R.dim)):
        raise RuntimeError("devsim_geometry: DEVSIM did not rebuild EdgeCouple / NodeVolume from the "
                           "written ElementEdgeCouple on region {!r} (DEVSIM behaviour changed?)".format(region))
    ds.set_parameter(device=device, region=region, name=MARKER_PARAMETER, value=negative)
    rep.applied = rep.written = True
    if rep.has_negatives:
        warnings.warn(
            "apply_exact_geometry(negative={!r}) on region {!r}: {}.{}".format(
                negative, region, where,
                (" Clipped to 0: {:.3g} of sum|EdgeCouple|, {:.3g} of the volume, {:.3g} of the surface "
                 "area (the patch test fails on the clipped edges).".format(
                     rep.clipped_couple_fraction, rep.clipped_volume_fraction, rep.clipped_area_fraction))
                if clip else " Written as computed: exact volumes, NO M-matrix guarantee for "
                             "drift-diffusion."), UnreliableResultWarning, stacklevel=2)
    return rep


def _exact_report(device: str, region: str, policy: str) -> ExactGeometryReport:
    """The report of an already-exact region, from reads only (no temporary models)."""
    ds = _ds()
    dim = int(ds.get_dimension(device=device))
    rep = ExactGeometryReport(device=device, region=region, dimension=dim, applied=True, policy=policy)
    if dim == 1:
        rep.true_volume = float(_arr(ds.get_edge_model_values(device=device, region=region,
                                                              name="EdgeLength")).sum())
        return rep
    xyz = np.column_stack([_arr(ds.get_node_model_values(device=device, region=region, name=c))
                           for c in ("x", "y", "z")])[:, :dim]
    els = np.asarray(ds.get_element_node_list(device=device, region=region), dtype=np.int64)
    rep.n_elements = int(els.shape[0])
    rep.true_volume = float(np.abs(np.linalg.det(xyz[els[:, 1:]] - xyz[els[:, :1]])).sum()
                            / (2.0 if dim == 2 else 6.0))
    rep.devsim_volume_ratio = rep.exact_volume_ratio = float(
        _arr(ds.get_node_model_values(device=device, region=region, name="NodeVolume")).sum() / rep.true_volume)
    return rep


def resolve_policy(exact_geometry: Union[bool, str, None]) -> Optional[str]:
    """The builders' `exact_geometry` option -> the apply_exact_geometry `negative` policy, or None (off).

      True (DEFAULT) / 'warn' -> 'warn'   exact geometry; negative assembled couples (gmsh / Delaunay
                                          meshes) are written as computed and warned (recommended);
      'raise'                 -> 'raise'  refuse a mesh with negative assembled couples / volumes;
      'clip'                  -> 'clip'   clip them to 0 (keeps the M-matrix property, measured cost);
      False / None            -> None     DEVSIM's uncorrected geometry (over-counts on non-well-centred
                                          elements; reproduces pre-v0.14 numbers).
    Library meshes are exact already and are never written, whatever the value."""
    if exact_geometry is True:
        return "warn"
    if exact_geometry is False or exact_geometry is None:
        return None
    if isinstance(exact_geometry, str) and exact_geometry in NEGATIVE_POLICIES:
        return exact_geometry
    raise ValueError("exact_geometry must be True, False, or one of {}; got {!r}".format(
        NEGATIVE_POLICIES, exact_geometry))


def correct_device_geometry(device: str, exact_geometry: Union[bool, str, None] = True,
                            *, stacklevel: int = 2) -> Dict[str, ExactGeometryReport]:
    """apply_exact_geometry on EVERY region of `device` -- what LayeredDevsimBuilder and
    devsim_3d.Devsim3DEquilibrium call right after ds.create_device (option `exact_geometry`, see
    resolve_policy). Regions that are already exact are only read (byte-identical). With 'raise' every
    region is measured first, so a refusal leaves the whole device unchanged. The per-region
    negative-couple warnings of 'warn' / 'clip' are merged into ONE UnreliableResultWarning per device.
    Returns {region: ExactGeometryReport} ({} when the option is off)."""
    policy = resolve_policy(exact_geometry)
    if policy is None:
        return {}
    ds = _ds()
    regions = list(ds.get_region_list(device=device))
    if policy == "raise":
        for r in regions:
            if not geometry_already_exact(device, r):
                rep = measure_dual_geometry(device, r)
                if rep.has_negatives:
                    raise NegativeDualGeometryError(
                        "exact_geometry='raise': region {!r} of device {!r} has negative assembled geometry "
                        "({}); nothing was written. Use exact_geometry=True ('warn', recommended) or "
                        "'clip'.".format(r, device, rep.summary()), rep)
    out, notes = {}, []
    for r in regions:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out[r] = apply_exact_geometry(device, r, negative=policy)
        for x in w:
            if issubclass(x.category, UnreliableResultWarning):
                notes.append(str(x.message))
            else:                                               # pragma: no cover - none raised today
                warnings.warn_explicit(x.message, x.category, x.filename, x.lineno)
    if notes:
        warnings.warn("exact geometry on device {!r} (devsim_geometry; DEVSIM's own geometry over-counts here): "
                      "{}".format(device, " | ".join(notes)), UnreliableResultWarning, stacklevel=stacklevel + 1)
    return out
