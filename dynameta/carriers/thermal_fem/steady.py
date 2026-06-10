"""Steady single-temperature heat solve: div(k grad T) = -Q.

Split from the former monolithic thermal_fem.py; see the package __init__ docstring
for unit conventions (the _S nm-scaling derivation). Bodies are verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import ngsolve as ng

from dynameta.carriers.thermal_fem.common import ThermalLayer, _S, _build_thermal_forms, _mean_T_per_layer

@dataclass
class ThermalResult:
    mesh: object                 # ng.Mesh (coordinates in nm)
    T: object                    # ng.GridFunction, temperature [K]
    layers: List[ThermalLayer]

    def mean_T_per_layer(self) -> np.ndarray:
        """Volume-averaged temperature [K] in each layer (sink-first order)."""
        return _mean_T_per_layer(self.mesh, self.T, self.layers)

    def T_at(self, x_m: float, y_m: float, z_m: float) -> float:
        return float(np.real(self.T(self.mesh(x_m * _S, y_m * _S, z_m * _S))))


def solve_thermal_fem(layers: List[ThermalLayer], *, period_x_m: float, period_y_m: float,
                      flux_W_m2: float = 0.0, T_sink_K: float = 300.0,
                      joule_W_m3: Optional[Union[float, Dict[str, float], object]] = None,
                      maxh_m: Optional[float] = None, order: int = 2,
                      linear_solver: str = "umfpack") -> ThermalResult:
    """Steady heat equation div(k grad T) = -Q on the layered box (period_x_m x period_y_m x sum-of-
    thicknesses): bottom face Dirichlet T = T_sink_K; top face Neumann inflow `flux_W_m2`; lateral
    faces natural (insulated). `joule_W_m3` adds a volumetric source Q [W/m^3] -- a float (uniform),
    a {layer_name: Q} dict (per-layer), or an NGSolve CF in mesh (nm) coordinates. Returns the
    ThermalResult (T field + mean_T_per_layer)."""
    if linear_solver not in ("umfpack", "sparsecholesky"):       # no silent substitution
        raise ValueError("linear_solver must be 'umfpack' or 'sparsecholesky', got {!r}".format(
            linear_solver))
    mesh, fes, u, v, a, f, k_cf = _build_thermal_forms(
        layers, period_x_m, period_y_m, flux_W_m2, T_sink_K, joule_W_m3, maxh_m, order)
    T = ng.GridFunction(fes)
    T.Set(ng.CoefficientFunction(float(T_sink_K)), definedon=mesh.Boundaries("bot"))
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        res = f.vec - a.mat * T.vec
        inv = a.mat.Inverse(fes.FreeDofs(), inverse=linear_solver)
        T.vec.data += inv * res
    return ThermalResult(mesh=mesh, T=T, layers=list(layers))
