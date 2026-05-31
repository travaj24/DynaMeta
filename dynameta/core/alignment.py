"""
The KEYSTONE of the bridge: the coordinate/identity contract between the
carrier solver and the optical geometry. This is the information that lives
in NEITHER solver alone -- it says which optical mesh region corresponds to
which carrier region, where that region sits in space (so the carrier grid can
be affine-placed into it), and what length unit the optical mesh uses.

A bring-your-own-NGSolve-mesh user implements OpticalGeometryBuilder.alignment()
to return a GeometryAlignment; a bring-your-own-DEVSIM user names regions to
match `source_region`. No shared object graph -- just string keys + bboxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from dynameta.core.units import UnitScale


@dataclass(frozen=True)
class RegionAlignment:
    """How ONE optical (mesh) region maps to a carrier-field source region."""
    mesh_region:   str                                   # optical mesh material name
    source_region: str                                   # CarrierField region key
    bbox_m: Tuple[float, float, float, float, float, float]  # xlo,xhi,ylo,yhi,zlo,zhi (SI)
    stack_axis: Literal["x", "y", "z"] = "y"             # carrier-field through-stack axis


@dataclass
class GeometryAlignment:
    unit_scale:         UnitScale                # length unit of the optical mesh
    region_alignments:  List[RegionAlignment]    # carrier-derived (spatial-eps) regions
    fixed_eps_regions:  Dict[str, str]           # every OTHER mesh region -> material name

    def validate_coverage(self, mesh_regions: List[str]) -> None:
        """Fail loudly if any mesh region is unmapped or double-mapped (the old
        code silently defaulted unmapped regions to vacuum)."""
        spatial = {ra.mesh_region for ra in self.region_alignments}
        fixed = set(self.fixed_eps_regions)
        dup = spatial & fixed
        if dup:
            raise ValueError("Regions mapped both spatial and fixed: {}".format(sorted(dup)))
        mapped = spatial | fixed
        mesh_set = set(mesh_regions)
        missing = mesh_set - mapped
        extra = mapped - mesh_set
        if missing:
            raise ValueError("Mesh regions with no eps mapping: {}".format(sorted(missing)))
        if extra:
            raise ValueError("Alignment names regions not in the mesh: {}".format(sorted(extra)))
