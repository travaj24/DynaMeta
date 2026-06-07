"""
FEM heat-equation thermal driver + electro-thermal Joule coupling -- the volumetric/lateral
generalization of carriers.thermal (which is the exact series-thermal-resistance profile for a 1D
flux-driven stack). Solves the steady heat equation div(k grad T) = -Q on a layered box with the
bottom face at the sink temperature (Dirichlet), a heat flux into the top face (Neumann), and an
optional volumetric Joule source Q [W/m^3] (e.g. sigma|E|^2 from the electrical solve -- the
electro-thermal coupling). Returns the temperature field T [K] for the field bundle that
ThermoOpticModel reads. Reduces EXACTLY to carriers.thermal.steady_layered_temperature when Q = 0,
and to the uniform-Joule slab profile T_mean = T_sink + Q L^2/(3k) for a single heated layer.
Requires NGSolve.

Layers are ordered from the SINK (index 0, at z = 0 / the bottom Dirichlet face) outward; the top
face (z = sum thicknesses) receives `flux_W_m2`. Units: the mesh is built in nm (coordinate =
metres * _S); with k in W/(m K), T in K, the SI weak form maps to mesh coordinates as
  int k gradT.gradv dV'  =  int (Q/_S^2) v dV'  +  int_top (flux/_S) v dS'
(the _S powers convert the SI source [W/m^3] and flux [W/m^2] into the nm-coordinate integrals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import netgen.occ as occ
import ngsolve as ng

_S = 1.0e9                       # mesh unit: coordinate = metres * _S (nm)


@dataclass
class ThermalLayer:
    """One layer of the conduction stack (ordered SINK-side first, index 0 at the bottom sink)."""
    name: str
    thickness_m: float
    k_thermal: float             # W/(m K)


@dataclass
class ThermalResult:
    mesh: object                 # ng.Mesh (coordinates in nm)
    T: object                    # ng.GridFunction, temperature [K]
    layers: List[ThermalLayer]

    def mean_T_per_layer(self) -> np.ndarray:
        """Volume-averaged temperature [K] in each layer (sink-first order)."""
        out = []
        for L in self.layers:
            dom = self.mesh.Materials(L.name)
            vol = ng.Integrate(ng.CoefficientFunction(1.0), self.mesh, definedon=dom)
            tt = ng.Integrate(self.T, self.mesh, definedon=dom)
            out.append(float((tt / vol).real) if abs(vol) > 0 else 0.0)
        return np.asarray(out, dtype=np.float64)

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
    if not layers:
        raise ValueError("layers must be non-empty")
    if any(L.thickness_m <= 0 or L.k_thermal <= 0 for L in layers):
        raise ValueError("every layer needs thickness_m > 0 and k_thermal > 0")
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
    mesh = ng.Mesh(occ.OCCGeometry(glued).GenerateMesh(maxh=maxh))

    k_by = {L.name: L.k_thermal for L in layers}
    missing = [m for m in mesh.GetMaterials() if m not in k_by]
    if missing:
        raise RuntimeError("thermal_fem: mesh materials {} have no k_thermal".format(sorted(set(missing))))
    k_cf = ng.CoefficientFunction([k_by[m] for m in mesh.GetMaterials()])

    fes = ng.H1(mesh, order=order, dirichlet="bot")
    u, v = fes.TnT()
    T = ng.GridFunction(fes)
    T.Set(ng.CoefficientFunction(float(T_sink_K)), definedon=mesh.Boundaries("bot"))
    a = ng.BilinearForm(fes)
    a += k_cf * ng.grad(u) * ng.grad(v) * ng.dx
    f = ng.LinearForm(fes)
    if flux_W_m2:
        f += (float(flux_W_m2) / _S) * v * ng.ds(definedon=mesh.Boundaries("top"))
    if joule_W_m3 is not None:
        if isinstance(joule_W_m3, dict):
            q_cf = ng.CoefficientFunction([float(joule_W_m3.get(m, 0.0)) for m in mesh.GetMaterials()])
        elif isinstance(joule_W_m3, (int, float)):
            q_cf = ng.CoefficientFunction(float(joule_W_m3))
        else:
            q_cf = joule_W_m3                                 # an ng CF (mesh coords)
        f += (q_cf / _S ** 2) * v * ng.dx
    if linear_solver not in ("umfpack", "sparsecholesky"):       # no silent substitution
        raise ValueError("linear_solver must be 'umfpack' or 'sparsecholesky', got {!r}".format(
            linear_solver))
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        res = f.vec - a.mat * T.vec
        inv = a.mat.Inverse(fes.FreeDofs(), inverse=linear_solver)
        T.vec.data += inv * res
    return ThermalResult(mesh=mesh, T=T, layers=list(layers))
