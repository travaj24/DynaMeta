"""Transient single-temperature heat solve (implicit Euler).

Split from the former monolithic thermal_fem.py; see the package __init__ docstring
for unit conventions (the _S nm-scaling derivation). Bodies are verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import ngsolve as ng

from dynameta.carriers.thermal_fem.common import ThermalLayer, _S, add_load_terms, build_thermal_forms, mean_T_per_layer

@dataclass
class ThermalTransientResult:
    """Trace of the transient heat solve. mean_T_per_layer_t has shape (n_times, n_layers); t_s the
    sample times [s]; T_final the last temperature field; T_snapshots the optional full-field copies.

    PICKLING (audit C-9 residual). `k_of_T`, `flux_of_t` and `joule_of_t` hold the CALLABLES the run
    was driven with, because that is the only way `steady_limit_T` can tell when it must stay silent
    (a boolean would answer "was it time-dependent?" but not "with what?", and callers do read them
    back). A result built from a LAMBDA therefore does not pickle -- measured:
    `PicklingError: Can't pickle <function <lambda>>` -- while a result with all three None (the
    constant-k, constant-load solve) pickles fine, mesh and GridFunction included. Pass a
    MODULE-LEVEL function instead of a lambda if the result has to be pickled, or drop the callables
    with `dataclasses.replace(result, k_of_T=repr(result.k_of_T), ...)` before dumping -- the guards
    only test `is not None`, so a repr string keeps `steady_limit_T` honest."""
    mesh: object
    layers: List[ThermalLayer]
    t_s: np.ndarray
    mean_T_per_layer_t: np.ndarray
    T_final: object
    flux_W_m2: float
    T_sink_K: float
    joule_W_m3: object
    T_snapshots: Optional[List[object]] = None
    # audit C-9: what the RUN actually used, so `steady_limit_T` can tell when it must stay silent.
    # The k(T) twin (thermal_fem.kirchhoff.solve_thermal_transient_kt_fem) assembles the stiffness
    # from per-ELEMENT k_of_T -- L.k_thermal is then only a positivity placeholder -- and it can be
    # run with a pure-Neumann bottom (bottom_bc='insulated'), which has no series-resistance limit at
    # all. The result object could see neither, so the property returned the closed form for a
    # DIFFERENT material / a BC the run never imposed, as a plausible-looking array with no signal
    # that it was wrong. Defaults keep the constant-k solver's result byte-identical.
    k_of_T: object = None                  # the k_of_T_by the run used (None = constant L.k_thermal)
    bottom_bc: str = "sink"                # 'sink' (Dirichlet T_sink) or 'insulated' (pure Neumann)
    # audit C-9 (wave-5 residual): the TIME-DEPENDENT load hooks. `flux_W_m2` / `joule_W_m3` record
    # only the STATIC arguments, so a run driven by flux_of_t / joule_of_t recorded the defaults
    # (0.0 / None) and steady_limit_T evaluated the closed form for a load the run never used --
    # measured: flux_of_t = 1e8 with the default flux_W_m2 = 0.0 claimed a steady limit of 300.0 K
    # while the run settles at 300.5 K, and joule_of_t = 1e16 was invisible to the Joule guard, which
    # returned 300.5 K against an actual 303.8 K. Same honesty rule: if the result cannot describe
    # the load, it says None rather than a plausible-looking number.
    flux_of_t: object = None               # the flux_of_t the run used (None = constant flux_W_m2)
    joule_of_t: object = None              # the joule_of_t the run used (None = constant joule_W_m3)

    def mean_T_per_layer(self) -> np.ndarray:
        """Volume-averaged temperature [K] per layer at the FINAL time (sink-first order)."""
        return mean_T_per_layer(self.mesh, self.T_final, self.layers)

    def T_at(self, x_m: float, y_m: float, z_m: float) -> float:
        return float(np.real(self.T_final(self.mesh(x_m * _S, y_m * _S, z_m * _S))))

    @property
    def steady_limit_T(self) -> Optional[np.ndarray]:
        """The analytic series-resistance steady limit (carriers.thermal) for the pure-flux,
        no-Joule, constant-k, constant-load, sink-bottom case -- the t -> infinity target. Returns
        None whenever that closed form does not describe the run (audit C-9):
          * a Joule source is present (no closed form here; compare against solve_thermal_fem),
          * the run used a temperature-dependent k_of_T (the stiffness came from per-element k(T),
            NOT from L.k_thermal, which the k(T) solver reads only for its > 0 validation),
          * bottom_bc='insulated' (a pure-Neumann box has no steady limit -- it stores the energy),
          * a TIME-DEPENDENT load hook (flux_of_t / joule_of_t) drove the run: `flux_W_m2` and
            `joule_W_m3` then hold the unused static arguments, and evaluating the closed form on
            them describes a load the march never applied (measured: 300.0 K claimed against 300.5 K
            reached for flux_of_t, 300.5 K against 303.8 K for joule_of_t).
        A steady limit for a time-varying drive would in any case need the t -> infinity value of
        the hook, which this object cannot know; call solve_thermal_fem with that value instead.
        The constant-k, constant-load solver leaves every new field at its default, so it is
        unaffected."""
        if self.joule_W_m3 is not None:
            return None
        if self.k_of_T is not None or self.bottom_bc != "sink":
            return None
        if self.flux_of_t is not None or self.joule_of_t is not None:
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
    source time-dependent (the load is reassembled each step); otherwise the load is constant. A
    time-dependent hook is RECORDED on the result and makes `steady_limit_T` return None, because
    the static `flux_W_m2` / `joule_W_m3` arguments are then unused and the closed form evaluated on
    them describes a load the march never applied (audit C-9).

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

    mesh, fes, u, v, a, f, k_cf = build_thermal_forms(
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
        add_load_terms(ff, v, mesh, fl, jo)
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
    mean_list = [mean_T_per_layer(mesh, T, layers)]
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
                mean_list.append(mean_T_per_layer(mesh, T, layers))
                if store_fields:
                    sc = ng.GridFunction(fes); sc.vec.data = T.vec; snaps.append(sc)

    return ThermalTransientResult(
        mesh=mesh, layers=list(layers), t_s=np.asarray(t_list, dtype=np.float64),
        mean_T_per_layer_t=np.asarray(mean_list, dtype=np.float64), T_final=T,
        flux_W_m2=float(flux_W_m2), T_sink_K=float(T_sink_K), joule_W_m3=joule_W_m3,
        T_snapshots=snaps,
        # audit C-9: record the load hooks so steady_limit_T can refuse (both None on the
        # constant-load path, which keeps that result byte-identical).
        flux_of_t=flux_of_t, joule_of_t=joule_of_t)
