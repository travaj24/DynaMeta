"""Shared mesh-unit scale MESH_SCALE (_S), layer dataclasses, mesh builder, and forms helpers.

Split from the former monolithic thermal_fem.py; see the package __init__ docstring
for unit conventions (the _S nm-scaling derivation). Bodies are verbatim.

SINGLE-HOMED NAMES (audit X-11). The split "promoted" three helpers to public names but kept the
pre-promotion private spellings as aliases, and the two spellings then split by caller: 18
in-package call sites on `_mean_T_per_layer` / `_add_load_terms` / `_build_thermal_forms` against 3
on the public names, with the package facade re-exporting both -- two names for one object, each
half-used, which is exactly how a later edit lands on one spelling and misses the other. The
CANONICAL spelling is now the PUBLIC one (`mean_T_per_layer`, `add_load_terms`,
`build_thermal_forms`): every call site inside the package uses it, and the underscored spellings
survive only as thin, deprecated, one-line aliases bound to the SAME function object (gated by
tests/test_audit_2026_07_25_cleanup.py, X-11) so anything importing them from the facade keeps
working. Do not add a call site on an underscored alias.
`_build_layered_mesh` / `_per_material_cf` are NOT doubled -- they are private-only, one name each.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import ngsolve as ng

# audit X-11: `_S` is the in-formula symbol for the nm mesh scale and MESH_SCALE is its spelled-out
# name; BOTH are single-sourced in carriers.fem_mesh (this module used to re-declare `_S = MESH_SCALE`,
# a second home for the same float). Import them; do not re-bind them here.
from dynameta.carriers.fem_mesh import MESH_SCALE      # noqa: F401 -- pass-through re-export: the
from dynameta.carriers.fem_mesh import _S, build_layered_box_mesh   # package facade exports it
from dynameta.warning_types import DynaMetaDeprecationWarning


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
    G_e_l [W/(m^3 K)], and the electron conductivity k_electron [W/(m K)].
    G_e_l = 0 decouples the fields (each evolves as an independent single-T problem).

    k_electron: PASS IT (audit S1-11, 2026-07-17; fixed wave-7 2026-11-02). The two channels conduct
    IN PARALLEL -- the G -> infinity limit is the single-T solve with k = k_e + k_l -- so the old
    silent default k_electron=None -> k_e = k_thermal (the LATTICE value) doubled the stack's
    conductivity in that limit (k = 2 k_thermal; measured on a 200 nm layer, k_thermal = 1, 1e8
    W/m^2 top flux, G = 1e20: steady mean rise 5.00 K two-temp vs 10.0 K single-T -- exactly half)
    with nothing to say a choice had been made. There is no physically neutral default (in a metal
    the electrons carry most of the heat; in a dielectric k_e ~ 0), so k_electron=None now emits a
    DynaMetaDeprecationWarning at construction; the legacy value is unchanged, so existing results
    are byte-identical. Every in-repo caller already passed it explicitly. A provided k_electron
    must be finite and >= 0 (0 = a non-conducting electron gas)."""
    G_e_l: float = 0.0           # electron-phonon coupling [W/(m^3 K)]
    C_e_J_m3K: float = 0.0       # volumetric electron heat capacity [J/(m^3 K)] (required > 0)
    k_electron: Optional[float] = None   # electron thermal conductivity [W/(m K)] -- pass it (S1-11)

    def __post_init__(self):
        if self.k_electron is None:
            warnings.warn(
                "ThermalLayerTwoTemp('{}'): k_electron=None defaults the ELECTRON conductivity to the "
                "LATTICE k_thermal={} W/(m K), so the two parallel channels conduct 2*k_thermal in the "
                "strong-coupling limit. This implicit default is deprecated (audit S1-11): pass "
                "k_electron explicitly.".format(self.name, self.k_thermal),
                DynaMetaDeprecationWarning, stacklevel=3)
        elif not (np.isfinite(float(self.k_electron)) and float(self.k_electron) >= 0.0):
            raise ValueError("ThermalLayerTwoTemp('{}'): k_electron must be finite and >= 0 W/(m K), "
                             "got {!r} (audit S1-11)".format(self.name, self.k_electron))

    def k_e(self) -> float:
        return self.k_thermal if self.k_electron is None else float(self.k_electron)


def mean_T_per_layer(mesh, T, layers) -> np.ndarray:
    """Volume-averaged temperature [K] in each layer of `layers` (same order; 0.0 if zero volume)."""
    out = []
    for L in layers:
        dom = mesh.Materials(L.name)
        vol = ng.Integrate(ng.CoefficientFunction(1.0), mesh, definedon=dom)
        tt = ng.Integrate(T, mesh, definedon=dom)
        out.append(float((tt / vol).real) if abs(vol) > 0 else 0.0)
    return np.asarray(out, dtype=np.float64)


_mean_T_per_layer = mean_T_per_layer         # DEPRECATED alias of the canonical public name (X-11)


def add_load_terms(f, v, mesh, flux_W_m2, joule_W_m3):
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


_add_load_terms = add_load_terms             # DEPRECATED alias of the canonical public name (X-11)


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


def build_thermal_forms(layers, period_x_m, period_y_m, flux_W_m2, T_sink_K, joule_W_m3,
                        maxh_m, order) -> Tuple:
    """Build the shared mesh + H1 space + stiffness BilinearForm a + load LinearForm f (UNASSEMBLED)
    used by BOTH the steady and transient solvers, plus the SINK LIFT. Returns
    (mesh, fes, u, v, a, f, k_cf, T_lift). Factoring this out keeps solve_thermal_fem byte-identical
    -- it assembles the same a, f it always did.

    T_lift is a GridFunction holding T_sink_K on the Dirichlet sink face ("bot") and 0 on every free
    dof: the Dirichlet data of the problem. A steady solve is
        T = GridFunction(fes); T.vec.data = T_lift.vec; T.vec.data += inv * (f.vec - a.mat * T.vec)
    AUDIT X-17 (2026-11-02): T_sink_K used to be accepted and IGNORED (a and f do not depend on
    it; measured: the returned forms were identical for T_sink_K = 300 and 1000), and each of the
    four callers re-applied the sink by hand -- a caller that started from a zero GridFunction
    solved with T_sink = 0 K (measured on a 200 nm / 100 nm stack at 1e8 W/m^2: mean layer T 7.1 /
    14.3 K instead of 307.1 / 314.3 K). The lift now comes FROM T_sink_K, and every in-package
    caller takes its sink values from it. T_sink_K must be finite and > 0 (kelvin)."""
    T_sink = float(T_sink_K)
    if not (np.isfinite(T_sink) and T_sink > 0.0):
        raise ValueError("build_thermal_forms: T_sink_K must be a finite temperature > 0 K, got "
                         "{!r} (audit X-17)".format(T_sink_K))
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
    add_load_terms(f, v, mesh, flux_W_m2, joule_W_m3)
    T_lift = ng.GridFunction(fes)
    T_lift.Set(ng.CoefficientFunction(T_sink), definedon=mesh.Boundaries("bot"))
    return mesh, fes, u, v, a, f, k_cf, T_lift


_build_thermal_forms = build_thermal_forms   # DEPRECATED alias of the canonical public name (X-11)
