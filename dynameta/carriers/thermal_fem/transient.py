"""Transient single-temperature heat solve (implicit Euler).

Split from the former monolithic thermal_fem.py; see the package __init__ docstring
for unit conventions (the _S nm-scaling derivation). Bodies are verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import ngsolve as ng

from dynameta.carriers.thermal_fem.common import ThermalLayer, _S, _add_load_terms, _build_thermal_forms, _mean_T_per_layer

@dataclass
class ThermalTransientResult:
    """Trace of the transient heat solve. mean_T_per_layer_t has shape (n_times, n_layers); t_s the
    sample times [s]; T_final the last temperature field; T_snapshots the optional full-field copies."""
    mesh: object
    layers: List[ThermalLayer]
    t_s: np.ndarray
    mean_T_per_layer_t: np.ndarray
    T_final: object
    flux_W_m2: float
    T_sink_K: float
    joule_W_m3: object
    T_snapshots: Optional[List[object]] = None

    def mean_T_per_layer(self) -> np.ndarray:
        """Volume-averaged temperature [K] per layer at the FINAL time (sink-first order)."""
        return _mean_T_per_layer(self.mesh, self.T_final, self.layers)

    def T_at(self, x_m: float, y_m: float, z_m: float) -> float:
        return float(np.real(self.T_final(self.mesh(x_m * _S, y_m * _S, z_m * _S))))

    @property
    def steady_limit_T(self) -> Optional[np.ndarray]:
        """The analytic series-resistance steady limit (carriers.thermal) for the pure-flux,
        no-Joule case -- the t -> infinity target. Returns None when a Joule source is present
        (no closed form here; compare against solve_thermal_fem instead)."""
        if self.joule_W_m3 is not None:
            return None
        from dynameta.carriers.thermal import steady_layered_temperature
        return steady_layered_temperature([L.k_thermal for L in self.layers],
                                          [L.thickness_m for L in self.layers],
                                          self.flux_W_m2, self.T_sink_K)


def solve_thermal_transient_fem(layers: List[ThermalLayer], *, period_x_m: float, period_y_m: float,
                                t_end_s: float, dt_s: float, flux_W_m2: float = 0.0,
                                T_sink_K: float = 300.0,
                                joule_W_m3: Optional[Union[float, Dict[str, float], object]] = None,
                                T_init_K: Optional[Union[float, object]] = None, theta: float = 1.0,
                                maxh_m: Optional[float] = None, order: int = 2,
                                linear_solver: str = "umfpack",
                                flux_of_t: Optional[Callable[[float], float]] = None,
                                joule_of_t: Optional[Callable[[float], object]] = None,
                                store_every: int = 1, store_fields: bool = False
                                ) -> ThermalTransientResult:
    """Transient heat equation rho*Cp*dT/dt = div(k grad T) + Q on the layered box, integrated by the
    theta-method from t=0 to t_end_s. theta=1 is backward-Euler (unconditionally stable, monotone --
    the default); theta=0.5 is Crank-Nicolson (2nd-order). The step is uniform dt = t_end_s/round
    (t_end_s/dt_s) so it lands exactly on t_end_s and the system matrix is factored once.

    EVERY layer must have rho_kg_m3 > 0 and Cp_J_kgK > 0 (the transient cannot run without rho*Cp;
    this is an explicit precondition, NOT an off-switch -- the off-switch is that the steady solver
    never reads them). Boundary conditions match the steady solve (bottom Dirichlet T_sink, top
    Neumann flux, lateral insulated). `flux_of_t` / `joule_of_t`, if given, make the flux / Joule
    source time-dependent (the load is reassembled each step); otherwise the load is constant.

    Returns a ThermalTransientResult with the per-layer mean-T trace (sampled every `store_every`
    steps, plus t=0 and the final step) and the final field; set store_fields=True to also keep
    full-field GridFunction copies at the sampled times (memory-heavy on fine meshes)."""
    if not (t_end_s > 0.0):
        raise ValueError("t_end_s must be > 0")
    if not (dt_s > 0.0):
        raise ValueError("dt_s must be > 0")
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must be in [0, 1] (1=backward-Euler, 0.5=Crank-Nicolson)")
    if any(L.rho_kg_m3 <= 0.0 or L.Cp_J_kgK <= 0.0 for L in layers):
        raise ValueError("transient requires every layer rho_kg_m3 > 0 and Cp_J_kgK > 0 "
                         "(set them on ThermalLayer; the steady solver does not need them)")
    if linear_solver not in ("umfpack", "sparsecholesky"):
        raise ValueError("linear_solver must be 'umfpack' or 'sparsecholesky', got {!r}".format(
            linear_solver))
    if store_every < 1:
        raise ValueError("store_every must be >= 1")

    n_steps = max(1, int(round(t_end_s / dt_s)))
    dt = t_end_s / n_steps
    time_dependent = (flux_of_t is not None) or (joule_of_t is not None)

    mesh, fes, u, v, a, f, k_cf = _build_thermal_forms(
        layers, period_x_m, period_y_m, flux_W_m2, T_sink_K, joule_W_m3, maxh_m, order)

    # MASS term: int (rho*Cp/_S^2) u v dV' = _S * M_phys (matches the _S * (K, f) scaling; see header)
    rhoCp_by = {L.name: float(L.rho_kg_m3) * float(L.Cp_J_kgK) for L in layers}
    rhoCp_cf = ng.CoefficientFunction([rhoCp_by[m] for m in mesh.GetMaterials()])
    m = ng.BilinearForm(fes)
    m += (rhoCp_cf / _S ** 2) * u * v * ng.dx

    def _load_at(t):
        ff = ng.LinearForm(fes)
        fl = flux_of_t(t) if flux_of_t is not None else flux_W_m2
        jo = joule_of_t(t) if joule_of_t is not None else joule_W_m3
        _add_load_terms(ff, v, mesh, fl, jo)
        ff.Assemble()
        return ff

    # initial condition: set the whole domain to T_init (default sink), then PATCH the constrained
    # sink dofs to T_sink. (A second Set(..., definedon=Boundaries) would ZERO the interior -- the
    # NGSolve boundary-projection semantics -- so instead copy only the non-free dofs from a
    # boundary-Set helper, leaving the interior IC intact.)
    T = ng.GridFunction(fes)
    if T_init_K is None:
        T.Set(ng.CoefficientFunction(float(T_sink_K)))
    elif isinstance(T_init_K, (int, float)):
        T.Set(ng.CoefficientFunction(float(T_init_K)))
    else:
        T.Set(T_init_K)
    g_bot = ng.GridFunction(fes)
    g_bot.Set(ng.CoefficientFunction(float(T_sink_K)), definedon=mesh.Boundaries("bot"))
    free = fes.FreeDofs()
    tvec = T.vec.FV().NumPy()
    gvec = g_bot.vec.FV().NumPy()
    mask = np.array([not free[i] for i in range(len(free))])
    tvec[mask] = gvec[mask]                 # constrained (sink) dofs -> T_sink; free dofs keep T_init

    t_list = [0.0]
    mean_list = [_mean_T_per_layer(mesh, T, layers)]
    snaps = None
    if store_fields:
        s0 = ng.GridFunction(fes); s0.vec.data = T.vec; snaps = [s0]

    with ng.TaskManager():
        a.Assemble(); m.Assemble()
        if not time_dependent:
            f.Assemble()
        # combined system S = M + theta*dt*K (M, K share the FES sparsity -> AsVector combine valid)
        S = m.mat.CreateMatrix()
        S.AsVector().data = m.mat.AsVector() + (theta * dt) * a.mat.AsVector()
        Sinv = S.Inverse(fes.FreeDofs(), inverse=linear_solver)
        rhs = T.vec.CreateVector()
        res = T.vec.CreateVector()

        f_old = f.vec if not time_dependent else _load_at(0.0).vec
        t = 0.0
        for step in range(1, n_steps + 1):
            t_new = t + dt
            f_new = f.vec if not time_dependent else _load_at(t_new).vec
            # (M + theta dt K) T^{n+1} = (M - (1-theta) dt K) T^n + dt[(1-theta) f^n + theta f^{n+1}]
            rhs.data = m.mat * T.vec - ((1.0 - theta) * dt) * (a.mat * T.vec) \
                + (dt * (1.0 - theta)) * f_old + (dt * theta) * f_new
            res.data = rhs - S * T.vec               # residual-correction keeps the sink Dirichlet fixed
            T.vec.data += Sinv * res
            t = t_new
            f_old = f_new
            if (step % store_every == 0) or (step == n_steps):
                t_list.append(t)
                mean_list.append(_mean_T_per_layer(mesh, T, layers))
                if store_fields:
                    sc = ng.GridFunction(fes); sc.vec.data = T.vec; snaps.append(sc)

    return ThermalTransientResult(
        mesh=mesh, layers=list(layers), t_s=np.asarray(t_list, dtype=np.float64),
        mean_T_per_layer_t=np.asarray(mean_list, dtype=np.float64), T_final=T,
        flux_W_m2=float(flux_W_m2), T_sink_K=float(T_sink_K), joule_W_m3=joule_W_m3,
        T_snapshots=snaps)
