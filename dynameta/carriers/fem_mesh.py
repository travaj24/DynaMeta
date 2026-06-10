"""Shared layered-box OCC meshing for the carrier-side NGSolve FEM drivers.

Single source of the mesh-unit scale _S (coordinate = metres * _S, i.e. nm) and of the
layered-box geometry/meshing used identically by the thermal and electrostatic solvers.
Validation of per-layer physical properties (k_thermal vs eps_static) stays at the call
sites -- this module is geometry only, so both solvers mesh byte-identically.
"""

from __future__ import annotations

import netgen.occ as occ
import ngsolve as ng

_S = 1.0e9                       # mesh unit: coordinate = metres * _S (nm)


def build_layered_box_mesh(layers, period_x_m, period_y_m, maxh_m):
    """Layered-box OCC mesh in nm coordinates with 'top'/'bot' faces named. `layers` is any
    sequence of objects with .name and .thickness_m (stacked bottom -> top along z). The maxh
    default is the shared heuristic min(thinnest layer, total/6)."""
    Px, Py = float(period_x_m) * _S, float(period_y_m) * _S
    total = float(sum(L.thickness_m for L in layers))
    maxh = (maxh_m if maxh_m is not None else min(min(L.thickness_m for L in layers),
                                                  total / 6.0)) * _S

    solids, z = [], 0.0
    for L in layers:
        b = occ.Box(occ.Pnt(0, 0, z * _S), occ.Pnt(Px, Py, (z + L.thickness_m) * _S))
        b.name = L.name
        solids.append(b)
        z += L.thickness_m
    glued = occ.Glue(solids)
    glued.faces.Max(occ.Z).name = "top"
    glued.faces.Min(occ.Z).name = "bot"
    return ng.Mesh(occ.OCCGeometry(glued).GenerateMesh(maxh=maxh))
