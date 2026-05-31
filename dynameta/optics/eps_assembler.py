"""
The NGSolve half of the bridge: turn the bridge's per-region EpsField dict into
a single domain CoefficientFunction over the mesh. Uniform EpsFields become
constant CFs; gridded ones become VoxelCoefficients (axes already in nm, values
already in (Nz,Ny,Nx) order). Solver-specific -- keeps NGSolve out of core/.
"""

from __future__ import annotations

from typing import Dict

import ngsolve as ng

from dynameta.core.eps_field import EpsField
from dynameta.optics.ngsolve_layered import OpticalGeometry


def assemble_eps_cf(geo: OpticalGeometry,
                      eps_by_region: Dict[str, EpsField]) -> ng.CoefficientFunction:
    mats = list(geo.mesh.GetMaterials())
    region_cf = {}
    for region in mats:
        if region not in eps_by_region:
            raise ValueError("no EpsField for mesh region '{}' (bridge/alignment "
                              "coverage gap)".format(region))
        ef = eps_by_region[region]
        if ef.is_uniform:
            region_cf[region] = ng.CoefficientFunction(complex(ef.scalar))
        else:
            start, end = ef.voxel_bounds_u()
            region_cf[region] = ng.VoxelCoefficient(start=start, end=end,
                                                       values=ef.values_zyx, linear=True)
    return ng.CoefficientFunction([region_cf[m] for m in mats])
