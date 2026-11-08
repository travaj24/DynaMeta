"""The ONE exact-element-volume check for every ELEMENT-based quantity on a DEVSIM region (v0.14.0,
consolidated plan row 3.2-13; docs/audit/2026-11-06-v0.14-impact-ionization-2d.md).

WHAT IT GUARDS. A quantity assembled or integrated per element edge with DEVSIM's ElementNodeVolume
(ENV) weight -- impact ionization in 2-D/3-D (impact_ionization), BTBT generation in 2-D/3-D
(tunneling.setup_btbt), the rank-one Hooge 1/f source (noise_analysis, element volume V_T = sum_k c_eT
L_k / d = 2 sum ENV) -- integrates correctly only if, in every element,

    2 * sum_{element edges} ENV  ==  |element volume|     (triangle |det|/2, tetrahedron |det|/6)

DEVSIM 2.11.2 builds ENV from the MAGNITUDES of the circumcentric dual pieces
(TriangleEdgeCouple.cc:74, TetrahedronEdgeCouple.cc:99), so the identity holds exactly on an element
whose pieces all have one sign (a well-centred element, or a closed one such as a right triangle /
Kuhn tet with the circumcentre on its boundary) and OVER-counts on any other: one obtuse triangle
reads 11x, the trirectangular corner tet 2x, a gmsh bar 1.225x (MEASURED, devsim_verify repro scripts
and devsim_3d._build_mesh_tensor). Because sum |piece| >= |sum piece| = the true volume, the per-element
ratio is >= 1 on DEVSIM's own weights, with equality exactly when no piece is negative -- so the check
is also the well-centredness test, measured on the weights the consumers actually use.

WHICH MESHES PASS. The library meshes: the 2-D rectilinear tensor mesh (LayeredDevsimBuilder,
create_2d_mesh: right triangles, zero-couple hypotenuses) and the 3-D tensor/Kuhn mesh
(devsim_3d.Stacked3DSpec(mesher='tensor'), kuhn_bar_mesh) -- both exact to round-off (MEASURED max
|ratio - 1| = 6.7e-15 / 1.1e-14 on graded meshes, tests/test_devsim_element_geometry.py). Uncorrected
unstructured meshes (raw-API gmsh / Delaunay meshes, Stacked3DSpec(mesher='gmsh', exact_geometry=False))
with obtuse triangles / non-well-centred tets FAIL and are refused with InexactElementGeometryError (MEASURED: the gmsh-tet noise bar 1.2249 x, 544 of
932 tets inexact, worst 3.81 x).

THRESHOLD rtol = 1e-9 (unchanged from the 3-D impact-ionization check it replaces): the exact meshes sit
at <= 1.1e-14 (k = 3 or 6 pieces per element at a few eps each), the defect starts at the percent level,
so any value between ~1e-12 and ~1e-3 separates them; 1e-9 leaves >= 3 decades of headroom either side.

The geometry correction (devsim_geometry.apply_exact_geometry -- applied AUTOMATICALLY by both builders
since v0.14.0, exact_geometry=True) replaces DEVSIM's ElementNodeVolume values with the signed (exact) ones
and makes this check pass without any change here: it reads the region's ElementNodeVolume as DEVSIM
reports it. So this refusal now fires only on raw-API devices (or exact_geometry=False builds).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


__all__ = ["InexactElementGeometryError", "ELEMENT_VOLUME_RTOL", "check_element_geometry"]

ELEMENT_VOLUME_RTOL = 1e-9


class InexactElementGeometryError(NotImplementedError):
    """DEVSIM's element volumes on a region are not the true element volumes (a non-well-centred mesh),
    so an element-based quantity on it would be biased. Subclasses NotImplementedError (the class the
    3-D impact-ionization refusal raised before v0.14.0), so `except NotImplementedError` keeps
    working. `.ratios` holds the measured check_element_geometry result."""

    def __init__(self, message: str, ratios: Dict[str, float]):
        super().__init__(message)
        self.ratios = ratios


def check_element_geometry(device: str, region: str, *, caller: Optional[str] = None,
                           rtol: float = ELEMENT_VOLUME_RTOL, raise_error: bool = True) -> Dict[str, float]:
    """Measure, per element of `region`, 2 * sum(ElementNodeVolume) / true simplex volume (module
    docstring), and REFUSE the region (InexactElementGeometryError, naming `caller`) when any element
    deviates from 1 by more than `rtol`; raise_error=False only measures.

    Returns {"dim", "n_elements", "volume_ratio" (2 sum ENV / sum true volume over the region),
    "max_element_deviation" (max |ratio_T - 1|), "max_element_ratio", "n_inexact" (elements with
    |ratio_T - 1| > rtol)}. 1-D regions have no element models in DEVSIM and their edge volumes are
    exact (L/2 per end): the result is the trivial exact one, nothing is read."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    dim = int(ds.get_dimension(device=device))
    if dim == 1:
        return {"dim": 1, "n_elements": 0, "volume_ratio": 1.0, "max_element_deviation": 0.0,
                "max_element_ratio": 1.0, "n_inexact": 0}
    if dim not in (2, 3):                                     # pragma: no cover - devsim dims are 1..3
        raise NotImplementedError("check_element_geometry: dimension {}".format(dim))
    xyz = np.column_stack([np.asarray(ds.get_node_model_values(device=device, region=region, name=c),
                                      dtype=np.float64) for c in "xyz"[:dim]])
    els = np.asarray(ds.get_element_node_list(device=device, region=region), dtype=np.int64)
    env = np.asarray(ds.get_element_model_values(device=device, region=region, name="ElementNodeVolume"),
                     dtype=np.float64)
    k = dim * (dim + 1) // 2                                  # edges per triangle / tetrahedron
    if els.ndim != 2 or els.shape[1] != dim + 1 or env.size != els.shape[0] * k:  # pragma: no cover
        raise RuntimeError("check_element_geometry: DEVSIM element layout on region {!r} is not the "
                           "documented one (blocks of {} element-edge values per element)".format(region, k))
    v_true = np.abs(np.linalg.det(xyz[els[:, 1:]] - xyz[els[:, :1]])) / math.factorial(dim)
    v_devsim = 2.0 * env.reshape(els.shape[0], k).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = v_devsim / v_true
    dev = np.abs(ratio - 1.0)
    dev[~np.isfinite(dev)] = np.inf                           # a degenerate (zero-volume) element
    out = {"dim": dim, "n_elements": int(els.shape[0]),
           "volume_ratio": float(v_devsim.sum() / v_true.sum()),
           "max_element_deviation": float(dev.max()) if dev.size else 0.0,
           "max_element_ratio": float(np.max(ratio)) if dev.size else 1.0,
           "n_inexact": int(np.count_nonzero(dev > rtol))}
    if raise_error and out["n_inexact"]:
        raise InexactElementGeometryError(
            "{}: DEVSIM's element volumes on region {!r} are not the true {} volumes -- {} of {} elements "
            "are off by more than {:.0e} (worst ratio {:.4f}; 2 sum ElementNodeVolume = {:.4f} x the region "
            "volume). DEVSIM 2.11 sums the MAGNITUDES of the circumcentric dual pieces, exact only on "
            "well-centred elements (no obtuse triangle / no tetrahedron with its circumcentre or a face "
            "circumcentre outside), so any element-based quantity here would be biased. The DynaMeta builders "
            "(LayeredDevsimBuilder, Devsim3DEquilibrium) correct this automatically (exact_geometry=True, the "
            "default); on a device built with the raw DEVSIM API call "
            "dynameta.carriers.devsim_geometry.apply_exact_geometry(device, region, negative='warn') right "
            "after create_device, or use an exact mesh: the 2-D rectilinear tensor mesh (create_2d_mesh) or "
            "the 3-D Kuhn tensor mesh (Stacked3DSpec(mesher='tensor'), devsim_3d.kuhn_bar_mesh) -- "
            "dynameta.carriers.element_geometry.".format(
                caller or "check_element_geometry", region, "triangle" if dim == 2 else "tetrahedron",
                out["n_inexact"], out["n_elements"], rtol, out["max_element_ratio"], out["volume_ratio"]),
            out)
    return out
