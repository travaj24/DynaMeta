"""Temperature-dependent k(T): Kirchhoff transform FEM, layered 1D, transient k(T(x)) (R21).

Split from the former monolithic thermal_fem.py; see the package __init__ docstring
for unit conventions (the _S nm-scaling derivation). Bodies are verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import ngsolve as ng

from dynameta.carriers.thermal_fem.common import ThermalLayer, _S, _add_load_terms, _build_layered_mesh, _build_thermal_forms, _mean_T_per_layer
from dynameta.carriers.thermal_fem.transient import ThermalTransientResult

# ---- R21: temperature-dependent k(T) via the EXACT Kirchhoff transform ------------------------

def kirchhoff_theta(k_of_T, T_K, T_ref_K: float):
    """Kirchhoff potential theta(T) = int_Tref^T k(T') dT' [W/m] (adaptive quadrature; SI)."""
    from scipy.integrate import quad
    val, _ = quad(lambda t: float(k_of_T(t)), float(T_ref_K), float(T_K), limit=200)
    return float(val)


def invert_kirchhoff(k_of_T, theta_W_m: float, T_ref_K: float, *, T_max_K: float = 5000.0):
    """T such that int_Tref^T k dT' = theta (brentq; theta monotone since k > 0). theta >= 0
    (heating above the reference) up to theta(T_max_K); below-reference theta inverts too."""
    from scipy.optimize import brentq
    th = float(theta_W_m)
    if th == 0.0:
        return float(T_ref_K)
    lo, hi = (float(T_ref_K), float(T_max_K)) if th > 0.0 else (1.0, float(T_ref_K))
    f = lambda t: kirchhoff_theta(k_of_T, t, T_ref_K) - th
    if f(lo) * f(hi) > 0.0:
        raise ValueError("invert_kirchhoff: theta={} outside the [{}, {}] K bracket".format(
            th, lo, hi))
    return float(brentq(f, lo, hi, xtol=1e-9))


@dataclass
class ThermalKirchhoffResult:
    """Steady k(T) solve via the exact Kirchhoff transform (R21). theta is the LINEAR potential
    field (GridFunction); temperatures come from the pointwise inversion T = theta^-1."""
    mesh: object
    theta: object                # ng.GridFunction, Kirchhoff potential [W/m]; theta(T_sink) = 0
    k_of_T: object
    T_sink_K: float
    layers: List[ThermalLayer]

    def theta_at(self, x_m: float, y_m: float, z_m: float) -> float:
        return float(np.real(self.theta(self.mesh(x_m * _S, y_m * _S, z_m * _S))))

    def T_at(self, x_m: float, y_m: float, z_m: float) -> float:
        """Temperature [K] at a point: the EXACT pointwise Kirchhoff inversion."""
        return invert_kirchhoff(self.k_of_T, self.theta_at(x_m, y_m, z_m), self.T_sink_K)

    def T_profile(self, z_points_m, x_m: float, y_m: float) -> np.ndarray:
        """T [K] along z at lateral position (x, y) -- the quasi-1D stack profile."""
        return np.array([self.T_at(x_m, y_m, float(z)) for z in np.asarray(z_points_m)])


def solve_thermal_kirchhoff_fem(layers: List[ThermalLayer], k_of_T, *, period_x_m: float,
                                period_y_m: float, flux_W_m2: float = 0.0,
                                T_sink_K: float = 300.0,
                                joule_W_m3: Optional[Union[float, Dict[str, float],
                                                           object]] = None,
                                maxh_m: Optional[float] = None, order: int = 2,
                                linear_solver: str = "umfpack") -> ThermalKirchhoffResult:
    """STEADY heat equation with temperature-dependent conductivity, div(k(T) grad T) = -Q,
    solved EXACTLY (no Picard) by the Kirchhoff transform: theta = int_{T_sink}^T k dT' turns it
    into the LINEAR problem div(grad theta) = -Q with theta = 0 on the sink face and the SAME
    Neumann flux/volumetric loads (-k dT/dn = -dtheta/dn). One linear solve, then the pointwise
    inversion T = theta^-1 (ThermalKirchhoffResult.T_at). k_of_T must be > 0 over the operating
    range (theta strictly monotone <=> invertible).

    SCOPE (v1, exact within it): ONE k(T) for the whole meshed stack -- a single-material
    Kirchhoff potential. Different k(T) per layer makes theta JUMP at interfaces (continuity of
    T, not theta) and needs interface jump conditions -- raise-and-defer rather than silently
    wrong. The TRANSIENT k(T) problem does not linearize this way (the C/k(theta) diffusivity
    stays nonlinear) and keeps the constant-k path. k_of_T = const c reduces to the linear solver
    with k = c exactly (theta = c (T - T_sink) -- the off-switch oracle)."""
    if linear_solver not in ("umfpack", "sparsecholesky"):
        raise ValueError("linear_solver must be 'umfpack' or 'sparsecholesky', got {!r}".format(
            linear_solver))
    if not callable(k_of_T):
        raise ValueError("k_of_T must be a callable T_K -> k [W/(m K)]")
    for t_probe in (float(T_sink_K), float(T_sink_K) + 100.0, float(T_sink_K) + 500.0):
        if not (float(k_of_T(t_probe)) > 0.0):
            raise ValueError("k_of_T must be > 0 over the operating range (k({:.0f} K) = {})"
                             .format(t_probe, float(k_of_T(t_probe))))
    # theta-problem: unit conductivity everywhere on the SAME mesh/loads as the linear path
    theta_layers = [ThermalLayer(name=L.name, thickness_m=L.thickness_m, k_thermal=1.0)
                    for L in layers]
    mesh, fes, u, v, a, f, _ = _build_thermal_forms(
        theta_layers, period_x_m, period_y_m, flux_W_m2, T_sink_K, joule_W_m3, maxh_m, order)
    th = ng.GridFunction(fes)                    # theta(T_sink) = 0 on the sink face (Dirichlet 0)
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        res = f.vec - a.mat * th.vec
        inv = a.mat.Inverse(fes.FreeDofs(), inverse=linear_solver)
        th.vec.data += inv * res
    return ThermalKirchhoffResult(mesh=mesh, theta=th, k_of_T=k_of_T, T_sink_K=float(T_sink_K),
                                  layers=list(layers))


# ---- R21 follow-on: PER-LAYER k(T) (exact 1D) + transient k(T) FEM ----------------------------

@dataclass
class ThermalKirchhoffLayeredResult:
    """Exact 1D multi-layer k(T) steady solution (sequential per-layer Kirchhoff inversion).
    interface_T_K[i] is the temperature at the i-th layer boundary (sink-first: index 0 is the
    sink face at T_sink, index n is the top face); T_at_z gives the in-layer profile."""
    layers: List[ThermalLayer]
    interface_z_m: np.ndarray
    interface_T_K: np.ndarray
    k_of_T_by: Dict[str, object]
    flux_W_m2: float
    joule_by: Dict[str, float]
    T_sink_K: float

    def T_at_z(self, z_m: float) -> float:
        """Exact T at height z above the sink (piecewise Kirchhoff inversion)."""
        z = float(z_m)
        if not (self.interface_z_m[0] <= z <= self.interface_z_m[-1]):
            raise ValueError("z_m outside the stack")
        i = min(int(np.searchsorted(self.interface_z_m, z, side="right")) - 1,
                len(self.layers) - 1)
        L = self.layers[i]
        zlo = float(self.interface_z_m[i])
        above = sum(self.joule_by.get(Lj.name, 0.0) * Lj.thickness_m
                    for Lj in self.layers[i + 1:])
        q_i = self.joule_by.get(L.name, 0.0)
        zhi = float(self.interface_z_m[i + 1])
        dth = ((self.flux_W_m2 + above + q_i * zhi) * (z - zlo)
               - q_i * (z ** 2 - zlo ** 2) / 2.0)
        if dth == 0.0:
            return float(self.interface_T_K[i])
        return invert_kirchhoff(self.k_of_T_by[L.name], dth, float(self.interface_T_K[i]))


def solve_thermal_kirchhoff_layered_1d(layers: List[ThermalLayer], k_of_T_by, *,
                                       flux_W_m2: float = 0.0, T_sink_K: float = 300.0,
                                       joule_W_m3: Optional[Dict[str, float]] = None
                                       ) -> ThermalKirchhoffLayeredResult:
    """EXACT steady 1D multi-layer stack with PER-LAYER temperature-dependent conductivity --
    the interface theta-jump case the single-material Kirchhoff FEM defers. In 1D the heat
    current j(z) = flux + int_z^L Q dz' is KNOWN in closed form (energy conservation), so within
    each layer d theta_i/dz = j(z) integrates exactly (quadratic in z for uniform per-layer Q)
    and the interface condition is simply T-continuity: invert each layer's OWN Kirchhoff
    potential theta_i = int k_i dT' (referenced to its lower-interface temperature), walking
    sink -> top. theta itself jumps at interfaces (different k_i(T)); temperature and normal
    flux are continuous by construction. k_of_T_by: ONE callable for all layers or a dict
    {layer_name: callable} covering EVERY layer; joule_W_m3: optional {layer_name: Q} uniform
    per-layer sources. Constant-k callables reduce EXACTLY to the series-resistance solution
    (carriers.thermal.steady_layered_temperature)."""
    if not layers:
        raise ValueError("layers must be non-empty")
    if callable(k_of_T_by):
        k_by = {L.name: k_of_T_by for L in layers}
    else:
        k_by = dict(k_of_T_by)
        missing = [L.name for L in layers if L.name not in k_by]
        if missing:
            raise ValueError("k_of_T_by must cover every layer (missing {}) -- pass "
                             "lambda T: k_const for constant-k layers".format(missing))
    joule_by = dict(joule_W_m3 or {})
    for L in layers:
        for t_probe in (float(T_sink_K), float(T_sink_K) + 200.0):
            if not (float(k_by[L.name](t_probe)) > 0.0):
                raise ValueError("k_of_T for layer {!r} must be > 0 (k({:.0f} K) = {})".format(
                    L.name, t_probe, float(k_by[L.name](t_probe))))
    z_if = np.concatenate([[0.0], np.cumsum([L.thickness_m for L in layers])])
    T_if = np.empty(len(layers) + 1)
    T_if[0] = float(T_sink_K)
    res = ThermalKirchhoffLayeredResult(layers=list(layers), interface_z_m=z_if,
                                        interface_T_K=T_if, k_of_T_by=k_by,
                                        flux_W_m2=float(flux_W_m2), joule_by=joule_by,
                                        T_sink_K=float(T_sink_K))
    for i in range(len(layers)):
        T_if[i + 1] = res.T_at_z(float(z_if[i + 1]) * (1.0 - 1e-15))  # top of layer i
    return res


def solve_thermal_transient_kt_fem(layers: List[ThermalLayer], k_of_T_by, *, period_x_m: float,
                                   period_y_m: float, t_end_s: float, dt_s: float,
                                   flux_W_m2: float = 0.0, T_sink_K: float = 300.0,
                                   joule_W_m3: Optional[Union[float, Dict[str, float],
                                                              object]] = None,
                                   T_init_K: Optional[float] = None, theta: float = 1.0,
                                   maxh_m: Optional[float] = None, order: int = 2,
                                   linear_solver: str = "umfpack", store_every: int = 1
                                   ) -> ThermalTransientResult:
    """TRANSIENT heat equation with temperature-dependent conductivity k(T(x)) (R21 follow-on):
    rho Cp dT/dt = div(k(T) grad T) + Q via the theta-method with a POINTWISE elementwise
    coefficient -- each step projects T onto piecewise-constant L2 element means, evaluates the
    layer's k(T) per element (no layer lumping), and reassembles the stiffness (lagged/Picard
    coefficient k(T^n): first-order in dt on top of the theta error; the t -> infinity steady
    limit is the TRUE nonlinear solution since T stops changing). k_of_T_by: one callable or a
    per-layer dict covering every layer. C(T) is NOT modeled (rho Cp constant -- documented v1).
    Cost note: the system matrix is refactored EVERY step (k changes); use the constant-k
    solve_thermal_transient_fem when k is constant. Constant callables reproduce it to solver
    roundoff."""
    if not (t_end_s > 0.0):
        raise ValueError("t_end_s must be > 0")
    if not (dt_s > 0.0):
        raise ValueError("dt_s must be > 0")
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must be in [0, 1]")
    if any(L.rho_kg_m3 <= 0.0 or L.Cp_J_kgK <= 0.0 for L in layers):
        raise ValueError("transient requires every layer rho_kg_m3 > 0 and Cp_J_kgK > 0")
    if linear_solver not in ("umfpack", "sparsecholesky"):
        raise ValueError("linear_solver must be 'umfpack' or 'sparsecholesky'")
    if store_every < 1:
        raise ValueError("store_every must be >= 1")
    if callable(k_of_T_by):
        k_by = {L.name: k_of_T_by for L in layers}
    else:
        k_by = dict(k_of_T_by)
        missing = [L.name for L in layers if L.name not in k_by]
        if missing:
            raise ValueError("k_of_T_by must cover every layer (missing {})".format(missing))

    n_steps = max(1, int(round(t_end_s / dt_s)))
    dt = t_end_s / n_steps
    mesh = _build_layered_mesh(layers, period_x_m, period_y_m, maxh_m)
    fes = ng.H1(mesh, order=order, dirichlet="bot")
    u, v = fes.TnT()
    # element-wise coefficient space: L2 order-0 dof index == element index (probed), so the
    # per-element material map lets each element evaluate ITS layer's k at ITS mean temperature
    Vk = ng.L2(mesh, order=0)
    mats = list(mesh.GetMaterials())
    el_mat = np.empty(mesh.ne, dtype=int)
    for el in mesh.Elements(ng.VOL):
        el_mat[Vk.GetDofNrs(el)[0]] = mats.index(el.mat)
    k_funcs = [k_by.get(m) for m in mats]
    if any(f is None for f in k_funcs):
        raise RuntimeError("mesh material without a k_of_T entry")
    kg = ng.GridFunction(Vk)
    Tmean = ng.GridFunction(Vk)

    rhoCp_by = {L.name: float(L.rho_kg_m3) * float(L.Cp_J_kgK) for L in layers}
    rhoCp_cf = ng.CoefficientFunction([rhoCp_by[m] for m in mesh.GetMaterials()])
    m = ng.BilinearForm(fes)
    m += (rhoCp_cf / _S ** 2) * u * v * ng.dx
    a = ng.BilinearForm(fes)
    a += kg * ng.grad(u) * ng.grad(v) * ng.dx
    f = ng.LinearForm(fes)
    _add_load_terms(f, v, mesh, flux_W_m2, joule_W_m3)

    T = ng.GridFunction(fes)
    T.Set(ng.CoefficientFunction(float(T_sink_K if T_init_K is None else T_init_K)))
    g_bot = ng.GridFunction(fes)
    g_bot.Set(ng.CoefficientFunction(float(T_sink_K)), definedon=mesh.Boundaries("bot"))
    free = fes.FreeDofs()
    mask = np.array([not free[i] for i in range(len(free))])
    T.vec.FV().NumPy()[mask] = g_bot.vec.FV().NumPy()[mask]

    def _refresh_k():
        Tmean.Set(T)                                   # element-mean temperatures
        tv = Tmean.vec.FV().NumPy()
        kv = kg.vec.FV().NumPy()
        for im, fk in enumerate(k_funcs):
            sel = el_mat == im
            kv[sel] = fk(tv[sel])
        if not np.all(kv > 0.0):
            raise ValueError("k_of_T returned a non-positive conductivity during the transient")

    t_list = [0.0]
    mean_list = [_mean_T_per_layer(mesh, T, layers)]
    with ng.TaskManager():
        m.Assemble()
        f.Assemble()
        rhs = T.vec.CreateVector()
        res = T.vec.CreateVector()
        t = 0.0
        for step in range(1, n_steps + 1):
            _refresh_k()
            a.Assemble()                               # k(T^n) -> stiffness changes every step
            S = m.mat.CreateMatrix()
            S.AsVector().data = m.mat.AsVector() + (theta * dt) * a.mat.AsVector()
            Sinv = S.Inverse(fes.FreeDofs(), inverse=linear_solver)
            rhs.data = m.mat * T.vec - ((1.0 - theta) * dt) * (a.mat * T.vec) + dt * f.vec
            res.data = rhs - S * T.vec
            T.vec.data += Sinv * res
            t += dt
            if (step % store_every == 0) or (step == n_steps):
                t_list.append(t)
                mean_list.append(_mean_T_per_layer(mesh, T, layers))

    return ThermalTransientResult(
        mesh=mesh, layers=list(layers), t_s=np.asarray(t_list, dtype=np.float64),
        mean_T_per_layer_t=np.asarray(mean_list, dtype=np.float64), T_final=T,
        flux_W_m2=float(flux_W_m2), T_sink_K=float(T_sink_K), joule_W_m3=joule_W_m3)
