"""Shared mesh-unit scale _S, layer dataclasses, mesh builder, and forms helpers.

Split from the former monolithic thermal_fem.py; see the package __init__ docstring
for unit conventions (the _S nm-scaling derivation). Bodies are verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import ngsolve as ng

from dynameta.carriers.fem_mesh import _S, build_layered_box_mesh


@dataclass
class ThermalLayer:
    """One layer of the conduction stack (ordered SINK-side first, index 0 at the bottom sink).
    rho_kg_m3 / Cp_J_kgK default 0.0 and are read ONLY by the transient solver -- the steady path
    ignores them, so existing steady callers are byte-identical."""
    name: str
    thickness_m: float
    k_thermal: float             # W/(m K)
    rho_kg_m3: float = 0.0       # mass density [kg/m^3]    (transient only; required > 0 there)
    Cp_J_kgK: float = 0.0        # specific heat [J/(kg K)] (transient only; required > 0 there)


@dataclass
class ThermalLayerTwoTemp(ThermalLayer):
    """Two-temperature layer (roadmap R14): the lattice channel reuses ThermalLayer's k_thermal /
    rho_kg_m3 / Cp_J_kgK; the ELECTRON channel adds a volumetric heat capacity C_e_J_m3K [J/(m^3 K)]
    (volumetric because the degenerate-gas C_e = gamma_e*T_e is naturally volumetric -- the SAME
    convention as carrier_heating.TwoTempParams, the lumped reference), the electron-phonon coupling
    G_e_l [W/(m^3 K)], and an optional electron conductivity k_electron (None -> k_thermal).
    G_e_l = 0 decouples the fields (each evolves as an independent single-T problem)."""
    G_e_l: float = 0.0           # electron-phonon coupling [W/(m^3 K)]
    C_e_J_m3K: float = 0.0       # volumetric electron heat capacity [J/(m^3 K)] (required > 0)
    k_electron: Optional[float] = None   # electron thermal conductivity [W/(m K)]; None -> k_thermal

    def k_e(self) -> float:
        return self.k_thermal if self.k_electron is None else float(self.k_electron)


def _mean_T_per_layer(mesh, T, layers) -> np.ndarray:
    out = []
    for L in layers:
        dom = mesh.Materials(L.name)
        vol = ng.Integrate(ng.CoefficientFunction(1.0), mesh, definedon=dom)
        tt = ng.Integrate(T, mesh, definedon=dom)
        out.append(float((tt / vol).real) if abs(vol) > 0 else 0.0)
    return np.asarray(out, dtype=np.float64)


def _add_load_terms(f, v, mesh, flux_W_m2, joule_W_m3):
    """Add the top-face flux Neumann + volumetric Joule source to a LinearForm (mesh-nm scaling).
    Shared by the steady and transient paths so the load is built identically."""
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
    return f


def _build_layered_mesh(layers, period_x_m, period_y_m, maxh_m):
    """Layered-box OCC mesh in nm coordinates with 'top'/'bot' faces named (shared by the single-T
    and two-temperature paths). The geometry/meshing is the shared carriers.fem_mesh builder (also
    used by electrostatics_fem), so both solvers mesh byte-identically; only the k_thermal
    validation is thermal-specific."""
    if not layers:
        raise ValueError("layers must be non-empty")
    if any(L.thickness_m <= 0 or L.k_thermal <= 0 for L in layers):
        raise ValueError("every layer needs thickness_m > 0 and k_thermal > 0")
    return build_layered_box_mesh(layers, period_x_m, period_y_m, maxh_m)


def _per_material_cf(mesh, by_name: Dict[str, float], what: str):
    """Per-material piecewise-constant CoefficientFunction; raises if a mesh material is missing."""
    missing = [m for m in mesh.GetMaterials() if m not in by_name]
    if missing:
        raise RuntimeError("thermal_fem: mesh materials {} have no {}".format(
            sorted(set(missing)), what))
    return ng.CoefficientFunction([by_name[m] for m in mesh.GetMaterials()])


def _build_thermal_forms(layers, period_x_m, period_y_m, flux_W_m2, T_sink_K, joule_W_m3,
                         maxh_m, order) -> Tuple:
    """Build the shared mesh + H1 space + stiffness BilinearForm a + load LinearForm f (UNASSEMBLED)
    used by BOTH the steady and transient solvers. Returns (mesh, fes, u, v, a, f, k_cf). Factoring
    this out keeps solve_thermal_fem byte-identical -- it assembles the same a, f it always did."""
    mesh = _build_layered_mesh(layers, period_x_m, period_y_m, maxh_m)

    k_by = {L.name: L.k_thermal for L in layers}
    missing = [m for m in mesh.GetMaterials() if m not in k_by]
    if missing:
        raise RuntimeError("thermal_fem: mesh materials {} have no k_thermal".format(sorted(set(missing))))
    k_cf = ng.CoefficientFunction([k_by[m] for m in mesh.GetMaterials()])

    fes = ng.H1(mesh, order=order, dirichlet="bot")
    u, v = fes.TnT()
    a = ng.BilinearForm(fes)
    a += k_cf * ng.grad(u) * ng.grad(v) * ng.dx
    f = ng.LinearForm(fes)
    _add_load_terms(f, v, mesh, flux_W_m2, joule_W_m3)
    return mesh, fes, u, v, a, f, k_cf
