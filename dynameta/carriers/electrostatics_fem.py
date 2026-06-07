"""
FEM electrostatics driver -- the Laplace/Poisson generalization of carriers.electrostatics (which is
the EXACT series-capacitor field for a 1D layered stack). Solves div(eps_static grad phi) = 0 on a
layered dielectric box with Dirichlet electrodes (top at the applied voltage, bottom grounded) and
returns E = -grad(phi). Unlike the analytic series-cap, this handles laterally NON-UNIFORM geometry
(a split / patterned gate) where E is not purely along z; it reduces EXACTLY to the series-cap field
for a laterally uniform stack (validation/electrostatics_fem.py).

It PRODUCES the applied static E-field for the field bundle that a field-effect EffectModel
(PockelsEffect, KerrEffect, FranzKeldyshEffect) reads as fields['E'] -> a tensor eps. Requires
NGSolve. Convention: E = -grad(phi); with the top electrode at +applied_V over a grounded bottom, E
points along -z (E_z < 0 for applied_V > 0), matching carriers.electrostatics.

Units: the mesh is built in nm (coordinate = metres * _S) for conditioning; phi is in volts, so a
mesh-coordinate gradient is V/nm and the physical field is E[V/m] = -grad_mesh(phi) * _S.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import netgen.occ as occ
import ngsolve as ng

_S = 1.0e9                       # mesh unit: coordinate = metres * _S (nm); E[V/m] = -grad_mesh * _S


@dataclass
class ElectrostaticLayer:
    """One dielectric layer of the stack (ordered bottom -> top along z)."""
    name: str
    thickness_m: float
    eps_static: float


@dataclass
class ElectrostaticResult:
    mesh: object             # ng.Mesh (coordinates in nm)
    phi: object              # ng.GridFunction, electrostatic potential [V]
    E_cf: object             # ng.CoefficientFunction, E = -grad(phi) [V/m]
    layers: List[ElectrostaticLayer]

    def mean_Ez_per_layer(self) -> np.ndarray:
        """Volume-averaged E_z [V/m] in each layer (bottom->top order). For a laterally uniform stack
        this is the per-layer series-capacitor field; for a patterned gate it is the layer mean."""
        out = []
        for L in self.layers:
            dom = self.mesh.Materials(L.name)
            vol = ng.Integrate(ng.CoefficientFunction(1.0), self.mesh, definedon=dom)
            ez = ng.Integrate(self.E_cf[2], self.mesh, definedon=dom)
            out.append(float((ez / vol).real) if abs(vol) > 0 else 0.0)
        return np.asarray(out, dtype=np.float64)

    def E_at(self, x_m: float, y_m: float, z_m: float) -> np.ndarray:
        """E = (Ex, Ey, Ez) [V/m] at a physical point (metres)."""
        val = self.E_cf(self.mesh(x_m * _S, y_m * _S, z_m * _S))
        return np.asarray(val, dtype=np.complex128).real.reshape(3)


def solve_electrostatics_fem(layers: List[ElectrostaticLayer], applied_V: float, *,
                             period_x_m: float, period_y_m: float,
                             maxh_m: Optional[float] = None, order: int = 2,
                             top_voltage_cf: Optional[object] = None,
                             linear_solver: str = "umfpack") -> ElectrostaticResult:
    """Solve div(eps_static grad phi) = 0 on the layered box (period_x_m x period_y_m x sum-of-
    thicknesses), Dirichlet phi = applied_V on the top face and 0 on the bottom; lateral faces are
    natural (Neumann). Returns an ElectrostaticResult with phi and E = -grad(phi) [V/m].

    `top_voltage_cf` (optional) sets a laterally-VARYING top electrode (e.g. a split gate) as an
    NGSolve CoefficientFunction in MESH coordinates (nm): use ng.x / 1e9, ng.y / 1e9 to express it in
    metres. When None the top is the uniform `applied_V`."""
    if not layers:
        raise ValueError("layers must be non-empty")
    if any(L.thickness_m <= 0 or L.eps_static <= 0 for L in layers):
        raise ValueError("every layer needs thickness_m > 0 and eps_static > 0")
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

    eps_by = {L.name: L.eps_static for L in layers}
    missing = [m for m in mesh.GetMaterials() if m not in eps_by]
    if missing:                                                # anti-silent-failure
        raise RuntimeError("electrostatics_fem: mesh materials {} have no eps_static (layer-name "
                           "drift)".format(sorted(set(missing))))
    eps_cf = ng.CoefficientFunction([eps_by[m] for m in mesh.GetMaterials()])

    fes = ng.H1(mesh, order=order, dirichlet="top|bot")
    u, v = fes.TnT()
    phi = ng.GridFunction(fes)
    top_cf = top_voltage_cf if top_voltage_cf is not None else ng.CoefficientFunction(float(applied_V))
    phi.Set(top_cf, definedon=mesh.Boundaries("top"))          # top = V(x,y); bottom stays 0
    a = ng.BilinearForm(fes)
    a += eps_cf * ng.grad(u) * ng.grad(v) * ng.dx
    f = ng.LinearForm(fes)
    if linear_solver not in ("umfpack", "sparsecholesky"):       # no silent substitution
        raise ValueError("linear_solver must be 'umfpack' or 'sparsecholesky', got {!r}".format(
            linear_solver))
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        res = f.vec - a.mat * phi.vec
        inv = a.mat.Inverse(fes.FreeDofs(), inverse=linear_solver)
        phi.vec.data += inv * res
    E_cf = -_S * ng.grad(phi)                                  # V/m
    return ElectrostaticResult(mesh=mesh, phi=phi, E_cf=E_cf, layers=list(layers))
