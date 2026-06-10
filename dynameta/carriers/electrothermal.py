"""
Self-consistent electro-thermo-optic coupling (roadmap R6): a fixed-point (Picard) driver that closes
the loop E -> Joule(sigma, n, T) -> T -> sigma(T) over the THREE existing, separately-validated steady
FEM solvers (electrostatics_fem, thermal_fem) plus the eps(T) optical leg (ThermoOpticModel via the
bridge). The applied bias sets the static field E (electrostatics_fem); the Joule source
Q = sigma(n,T) |E|^2 [W/m^3] heats the stack (thermal_fem); the temperature rise feeds back into a
caller-supplied sigma(T) (and, optionally, n(T)/mu(T)); iterate until the per-layer mean temperature
stops moving. The converged T is the operating point a single weak-coupling pass mis-predicts when the
conductivity is temperature-dependent.

This is a NEW entry point -- nothing else imports it, and it makes NO behavioral change to the three
solvers or the bridge, so every existing path is byte-identical. It REUSES the documented Joule seam
(solve_thermal_fem joule_W_m3={layer: Q}) and the already-wired bridge T-seam (assemble_eps
extra_fields), modifying neither.

SCOPE: steady fixed point on the existing steady thermal solver (the transient theta-method from R5 is
not required here). The three meshes (electrostatic, thermal, optical) are DISTINCT; fields couple by
PHYSICAL per-layer means (mean_Ez/mean_Esq/mean_T), never by DOF index. sigma is an injected callable
(or constant) -- no ITO model is baked in. Requires NGSolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import ngsolve as ng

from dynameta.carriers.electrostatics_fem import ElectrostaticLayer, solve_electrostatics_fem
from dynameta.carriers.thermal_fem import (ThermalLayer, ThermalResult, _S, _build_thermal_forms,
                                           _add_load_terms, _mean_T_per_layer)

SigmaSpec = Union[float, Callable[[float], float]]      # constant S/m, or sigma(T_K) -> S/m


@dataclass
class ElectroThermalLayer:
    """One layer feeding BOTH the electrostatic and thermal solves (shared `name` prevents the
    {layer: Q} dict from silently mapping to the wrong layer). sigma_S_m is the DC conductivity used
    for the Joule source Q = sigma |E|^2 -- a constant (weak coupling) or a callable sigma(T_K)
    (electro-thermal feedback); 0.0 -> no Joule from this layer.

    This class is a COMPOSITION facade over the per-physics layer schemas, not a third schema:
    .electrostatic and .thermal are the canonical conversions consumed by the Picard driver, and
    compose() builds one from existing per-physics layers."""
    name: str
    thickness_m: float
    eps_static: float            # for the electrostatic solve
    k_thermal: float             # W/(m K), for the thermal solve
    sigma_S_m: SigmaSpec = 0.0   # DC conductivity [S/m]: float or callable of T_K

    @property
    def electrostatic(self) -> ElectrostaticLayer:
        """The ElectrostaticLayer view (single source of truth for the E-solve stack)."""
        return ElectrostaticLayer(self.name, self.thickness_m, self.eps_static)

    @property
    def thermal(self) -> ThermalLayer:
        """The ThermalLayer view (single source of truth for the T-solve stack)."""
        return ThermalLayer(self.name, self.thickness_m, self.k_thermal)

    @classmethod
    def compose(cls, electrostatic: ElectrostaticLayer, thermal: ThermalLayer,
                sigma_S_m: SigmaSpec = 0.0) -> "ElectroThermalLayer":
        """Build from existing per-physics layers (name + thickness must agree)."""
        if electrostatic.name != thermal.name:
            raise ValueError("compose: layer names disagree ({!r} vs {!r})".format(
                electrostatic.name, thermal.name))
        if electrostatic.thickness_m != thermal.thickness_m:
            raise ValueError("compose: layer thicknesses disagree for {!r} ({} vs {})".format(
                electrostatic.name, electrostatic.thickness_m, thermal.thickness_m))
        return cls(electrostatic.name, electrostatic.thickness_m, electrostatic.eps_static,
                   thermal.k_thermal, sigma_S_m)


@dataclass
class ElectroThermalResult:
    E_result: object             # ElectrostaticResult (final E field)
    T_result: object             # ThermalResult (final T field)
    layers: List[ElectroThermalLayer]
    T_per_layer: np.ndarray      # converged per-layer mean temperature [K]
    joule_per_layer: np.ndarray  # converged per-layer Joule density Q [W/m^3]
    n_iter: int
    converged: bool
    residual_history: List[float]
    total_joule_W: float         # integral of Q dV  (power in)
    total_sink_outflux_W: float  # conductive power out the sink face (energy-balance cross-check)
    applied_V: float
    T_sink_K: float


def _sigma_at(spec: SigmaSpec, T_K: float) -> float:
    return float(spec(T_K)) if callable(spec) else float(spec)


def _mean_scalar_per_layer(mesh, cf, layers) -> np.ndarray:
    out = []
    for L in layers:
        dom = mesh.Materials(L.name)
        vol = ng.Integrate(ng.CoefficientFunction(1.0), mesh, definedon=dom)
        val = ng.Integrate(cf, mesh, definedon=dom)
        out.append(float((val / vol).real) if abs(vol) > 0 else 0.0)
    return np.asarray(out, dtype=np.float64)


def _sink_outflux_W(t_result, layers) -> float:
    """Conductive power [W] leaving the bottom (sink) face: integral over 'bot' of (-k grad T).n with
    outward normal -z, i.e. integral of k dT/dz_phys dS_phys. The mesh is in nm: grad_phys = _S*grad_mesh
    and dS_phys = dS_mesh/_S^2, so the assembled boundary integral carries an overall 1/_S."""
    mesh, T = t_result.mesh, t_result.T
    k_by = {L.name: L.k_thermal for L in layers}
    k_cf = ng.CoefficientFunction([k_by[m] for m in mesh.GetMaterials()])
    qz = ng.BoundaryFromVolumeCF(k_cf * ng.grad(T)[2])          # k dT/dz in mesh-nm units
    integ = ng.Integrate(qz, mesh, definedon=mesh.Boundaries("bot"))
    return float(integ.real) / _S                              # dS_mesh -> dS_phys (1/_S^2) * _S(grad) = 1/_S


def solve_electrothermal_picard(layers: List[ElectroThermalLayer], applied_V: float, *,
                                period_x_m: float, period_y_m: float, T_sink_K: float = 300.0,
                                flux_W_m2: float = 0.0, max_iter: int = 50, tol_T_K: float = 1e-3,
                                relax: float = 1.0, maxh_m: Optional[float] = None, order: int = 2,
                                linear_solver: str = "umfpack") -> ElectroThermalResult:
    """Fixed-point electro-thermal solve. The static field E is set once by `applied_V` (eps_static is
    temperature-independent here); each iteration recomputes the per-layer Joule Q = sigma(T) |E|^2,
    solves the steady heat equation, and feeds the new per-layer mean T back into sigma(T). Damped
    update T <- (1-relax)*T_old + relax*T_new (relax in (0,1]; 1.0 = pure Picard). Converged when
    max|T_new - T_old| < tol_T_K. Raises if not converged within max_iter (when max_iter > 1);
    max_iter=1 is an explicit single weak-coupling pass (the limit GATE A checks)."""
    if not layers:
        raise ValueError("layers must be non-empty")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not (0.0 < relax <= 1.0):
        raise ValueError("relax must be in (0, 1]")
    if not (tol_T_K > 0.0):
        raise ValueError("tol_T_K must be > 0")
    if not (T_sink_K > 0.0):
        raise ValueError("T_sink_K must be > 0")
    if not np.isfinite(applied_V):
        raise ValueError("applied_V must be finite")

    estack = [L.electrostatic for L in layers]
    tstack = [L.thermal for L in layers]
    Px, Py = float(period_x_m), float(period_y_m)

    # E-field: eps_static is T-independent in this spec, so solve ONCE outside the loop. |E|^2 per layer.
    eres = solve_electrostatics_fem(estack, applied_V, period_x_m=Px, period_y_m=Py,
                                    maxh_m=maxh_m, order=order, linear_solver=linear_solver)
    Esq_cf = eres.E_cf[0] ** 2 + eres.E_cf[1] ** 2 + eres.E_cf[2] ** 2     # |E|^2 [V^2/m^2], real
    Esq = _mean_scalar_per_layer(eres.mesh, Esq_cf, layers)

    # Thermal mesh + stiffness factored ONCE; each Picard iteration rebuilds only the Joule load and
    # re-solves on the SAME mesh/factorization -> the const-sigma loop is a machine-exact no-op (no
    # re-meshing noise between iterations) and the solve is cheap.
    mesh_t, fes_t, _u, v_t, a_t, _f0, _k = _build_thermal_forms(
        tstack, Px, Py, flux_W_m2, T_sink_K, None, maxh_m, order)
    T_layer = np.full(len(layers), float(T_sink_K), dtype=np.float64)
    residual_history: List[float] = []
    converged = False
    Q = np.zeros(len(layers))
    n_iter = 0
    Tg = ng.GridFunction(fes_t)
    with ng.TaskManager():
        a_t.Assemble()
        inv = a_t.mat.Inverse(fes_t.FreeDofs(), inverse=linear_solver)
        for it in range(1, max_iter + 1):
            n_iter = it
            sigma = np.array([_sigma_at(L.sigma_S_m, T_layer[i]) for i, L in enumerate(layers)])
            if np.any(sigma < 0.0):
                raise ValueError("sigma must be >= 0 (a negative conductivity would give Q < 0)")
            Q = sigma * Esq                                                # W/m^3 per layer (>= 0)
            Qdict = {L.name: float(Q[i]) for i, L in enumerate(layers)}
            f = ng.LinearForm(fes_t)
            _add_load_terms(f, v_t, mesh_t, flux_W_m2, Qdict)
            f.Assemble()
            Tg.Set(ng.CoefficientFunction(float(T_sink_K)), definedon=mesh_t.Boundaries("bot"))
            res = f.vec - a_t.mat * Tg.vec
            Tg.vec.data += inv * res
            T_new = _mean_T_per_layer(mesh_t, Tg, tstack)
            T_upd = (1.0 - relax) * T_layer + relax * T_new
            resid = float(np.max(np.abs(T_upd - T_layer)))
            residual_history.append(resid)
            T_layer = T_upd
            if resid < tol_T_K:
                converged = True
                break
    if (not converged) and max_iter > 1:                                   # anti-silent-failure
        raise RuntimeError("electro-thermal Picard did not converge in {} iters (last residual "
                           "{:.3e} K > tol {:.3e}); try relax < 1 or more iters".format(
                               max_iter, residual_history[-1], tol_T_K))

    tres = ThermalResult(mesh=mesh_t, T=Tg, layers=tstack)
    total_joule_W = float(np.sum(Q * np.array([L.thickness_m for L in layers]) * Px * Py))
    try:
        total_outflux_W = _sink_outflux_W(tres, layers)
    except Exception:                                                      # boundary-grad eval optional
        total_outflux_W = float("nan")
    return ElectroThermalResult(
        E_result=eres, T_result=tres, layers=list(layers), T_per_layer=T_layer, joule_per_layer=Q,
        n_iter=n_iter, converged=converged, residual_history=residual_history,
        total_joule_W=total_joule_W, total_sink_outflux_W=total_outflux_W,
        applied_V=float(applied_V), T_sink_K=float(T_sink_K))


def electrothermal_extra_fields(layers: List[ElectroThermalLayer], *, period_x_m: float,
                                period_y_m: float, voltage_of_bias: Callable[[object], float],
                                optical_region: str, T_sink_K: float = 300.0, flux_W_m2: float = 0.0,
                                **picard_kw) -> Callable[[object], dict]:
    """Build an extra_fields closure fn(bias_point) -> {'T': <mean T of `optical_region`>} for
    pipeline.run_pipeline / core.bridge.assemble_eps. Pairs with an EffectEpsMap whose `optical_region`
    material uses a ThermoOpticModel (wrap it in a DeltaEffect if composed with a Drude(n) model, to
    avoid double-counting the background eps). The closure runs the steady electro-thermal Picard solve
    at the bias voltage and returns the converged mean temperature of the optical region as a scalar T
    (broadcast against the carrier grid by the bridge). Coordinate/per-layer transfer -- the thermal
    mesh is distinct from the optical mesh."""
    names = [L.name for L in layers]
    if optical_region not in names:
        raise ValueError("optical_region {!r} not among layer names {}".format(optical_region, names))
    idx = names.index(optical_region)

    def _fields(bias_point) -> dict:
        V = float(voltage_of_bias(bias_point))
        res = solve_electrothermal_picard(layers, V, period_x_m=period_x_m, period_y_m=period_y_m,
                                          T_sink_K=T_sink_K, flux_W_m2=flux_W_m2, **picard_kw)
        return {"T": float(res.T_per_layer[idx])}

    return _fields
