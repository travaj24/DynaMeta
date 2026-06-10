"""Two-temperature (electron/lattice) steady + transient solves (R14).

Split from the former monolithic thermal_fem.py; see the package __init__ docstring
for unit conventions (the _S nm-scaling derivation). Bodies are verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import ngsolve as ng

from dynameta.carriers.thermal_fem.common import ThermalLayerTwoTemp, _S, _build_layered_mesh, _mean_T_per_layer, _per_material_cf

@dataclass
class ThermalTwoTempResult:
    """Steady two-temperature solve (R14): electron + lattice fields on the shared mesh."""
    mesh: object
    Te: object                   # ng.GridFunction on the component H1 space [K]
    Tl: object
    layers: List[ThermalLayerTwoTemp]

    def mean_Te_per_layer(self) -> np.ndarray:
        return _mean_T_per_layer(self.mesh, self.Te, self.layers)

    def mean_Tl_per_layer(self) -> np.ndarray:
        return _mean_T_per_layer(self.mesh, self.Tl, self.layers)

    def Te_at(self, x_m: float, y_m: float, z_m: float) -> float:
        return float(np.real(self.Te(self.mesh(x_m * _S, y_m * _S, z_m * _S))))

    def Tl_at(self, x_m: float, y_m: float, z_m: float) -> float:
        return float(np.real(self.Tl(self.mesh(x_m * _S, y_m * _S, z_m * _S))))


@dataclass
class ThermalTransientTwoTempResult:
    """Trace of the spatially-resolved two-temperature transient (R14). mean_Te/Tl_per_layer_t have
    shape (n_times, n_layers); Te/Tl_final are component-space GridFunction copies; snapshots are
    optional (Te, Tl) GridFunction pairs at the sampled times."""
    mesh: object
    layers: List[ThermalLayerTwoTemp]
    t_s: np.ndarray
    mean_Te_per_layer_t: np.ndarray
    mean_Tl_per_layer_t: np.ndarray
    Te_final: object
    Tl_final: object
    flux_W_m2: float
    T_sink_K: float
    snapshots: Optional[List[Tuple[object, object]]] = None

    def Te_at(self, x_m: float, y_m: float, z_m: float) -> float:
        return float(np.real(self.Te_final(self.mesh(x_m * _S, y_m * _S, z_m * _S))))

    def Tl_at(self, x_m: float, y_m: float, z_m: float) -> float:
        return float(np.real(self.Tl_final(self.mesh(x_m * _S, y_m * _S, z_m * _S))))


def _twotemp_space_and_forms(layers: List[ThermalLayerTwoTemp], mesh, order: int,
                             dirichlet_bot: bool):
    """Compound (V x V) space + stiffness for the two-temperature system (R14). The weak form of

        C_e dTe/dt = div(k_e grad Te) - G (Te - Tl) + S_e
        C_l dTl/dt = div(k_l grad Tl) + G (Te - Tl) + S_l

    moves the spatial terms to the stiffness K so M du/dt = -K u + f:

        K = k_e grad(ue).grad(ve) + k_l grad(ul).grad(vl) + G (ue - ul)(ve - vl)

    -- the coupling block is the SYMMETRIC positive-semidefinite form G (ue-ul)(ve-vl) (equivalently
    +G(ue-ul) ve - G(ue-ul) vl: moving -G(Te-Tl) from the RHS of the electron equation into K flips
    its sign). The sign matters: the PSD orientation makes the coupling DISSIPATIVE (heat flows
    electron -> lattice when Te > Tl); the flipped orientation is anti-dissipative and blows up.
    The volumetric coupling carries the same 1/_S^2 mesh scaling as the mass/source terms (see the
    module header) so every assembled operator is _S * (its SI counterpart) and the _S cancels."""
    if any(not isinstance(L, ThermalLayerTwoTemp) for L in layers):
        raise ValueError("two-temperature solvers need ThermalLayerTwoTemp layers")
    if any(L.G_e_l < 0.0 for L in layers):
        raise ValueError("G_e_l must be >= 0 (0 decouples the fields)")
    if any(L.C_e_J_m3K <= 0.0 for L in layers):
        raise ValueError("two-temperature solvers require every layer C_e_J_m3K > 0")
    V = ng.H1(mesh, order=order, dirichlet="bot") if dirichlet_bot else ng.H1(mesh, order=order)
    VV = V * V
    (ue, ul), (ve, vl) = VV.TnT()
    k_e_cf = _per_material_cf(mesh, {L.name: L.k_e() for L in layers}, "k_e")
    k_l_cf = _per_material_cf(mesh, {L.name: L.k_thermal for L in layers}, "k_thermal")
    G_cf = _per_material_cf(mesh, {L.name: float(L.G_e_l) for L in layers}, "G_e_l")
    a = ng.BilinearForm(VV)
    a += k_e_cf * ng.grad(ue) * ng.grad(ve) * ng.dx
    a += k_l_cf * ng.grad(ul) * ng.grad(vl) * ng.dx
    a += (G_cf / _S ** 2) * (ue - ul) * (ve - vl) * ng.dx
    return V, VV, ue, ul, ve, vl, a


def _twotemp_load(VV, ve, vl, mesh, flux_W_m2, source_e_W_m3, source_l_W_m3):
    """One compound LinearForm carrying both channels: the top-face flux drives the ELECTRON field
    (optical power is absorbed by the carriers); volumetric sources go to their own channel."""
    f = ng.LinearForm(VV)
    if flux_W_m2:
        f += (float(flux_W_m2) / _S) * ve * ng.ds(definedon=mesh.Boundaries("top"))
    for src, vtest in ((source_e_W_m3, ve), (source_l_W_m3, vl)):
        if src is not None:
            if isinstance(src, dict):
                q_cf = ng.CoefficientFunction([float(src.get(m, 0.0)) for m in mesh.GetMaterials()])
            elif isinstance(src, (int, float)):
                q_cf = ng.CoefficientFunction(float(src))
            else:
                q_cf = src                                    # an ng CF (mesh coords)
            f += (q_cf / _S ** 2) * vtest * ng.dx
    return f


def _copy_component(V, u, idx):
    out = ng.GridFunction(V)
    out.vec.data = u.components[idx].vec
    return out


def solve_thermal_twotemp_fem(layers: List[ThermalLayerTwoTemp], *, period_x_m: float,
                              period_y_m: float, flux_W_m2: float = 0.0, T_sink_K: float = 300.0,
                              source_e_W_m3: Optional[Union[float, Dict[str, float], object]] = None,
                              source_l_W_m3: Optional[Union[float, Dict[str, float], object]] = None,
                              maxh_m: Optional[float] = None, order: int = 2,
                              linear_solver: str = "umfpack") -> ThermalTwoTempResult:
    """STEADY two-temperature solve (R14): 0 = div(k_e grad Te) - G(Te - Tl) + S_e and
    0 = div(k_l grad Tl) + G(Te - Tl) + S_l on the layered box; both fields Dirichlet T_sink_K on
    the bottom face, top-face Neumann flux into the ELECTRON field, lateral faces insulated. Heat
    capacities are never read by the steady solve. The G -> infinity limit collapses to the
    single-T solve with k = k_e + k_l; G = 0 decouples into two independent single-T problems."""
    if linear_solver not in ("umfpack", "sparsecholesky"):
        raise ValueError("linear_solver must be 'umfpack' or 'sparsecholesky', got {!r}".format(
            linear_solver))
    mesh = _build_layered_mesh(layers, period_x_m, period_y_m, maxh_m)
    V, VV, ue, ul, ve, vl, a = _twotemp_space_and_forms(layers, mesh, order, dirichlet_bot=True)
    f = _twotemp_load(VV, ve, vl, mesh, flux_W_m2, source_e_W_m3, source_l_W_m3)
    u = ng.GridFunction(VV)
    for comp in (0, 1):           # boundary Set per component (Set is per-component on compounds)
        u.components[comp].Set(ng.CoefficientFunction(float(T_sink_K)),
                               definedon=mesh.Boundaries("bot"))
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        res = f.vec - a.mat * u.vec
        inv = a.mat.Inverse(VV.FreeDofs(), inverse=linear_solver)
        u.vec.data += inv * res
    return ThermalTwoTempResult(mesh=mesh, Te=_copy_component(V, u, 0),
                                Tl=_copy_component(V, u, 1), layers=list(layers))


def solve_thermal_transient_twotemp_fem(layers: List[ThermalLayerTwoTemp], *, period_x_m: float,
                                        period_y_m: float, t_end_s: float, dt_s: float,
                                        flux_W_m2: float = 0.0, T_sink_K: float = 300.0,
                                        source_e_W_m3: Optional[Union[float, Dict[str, float],
                                                                      object]] = None,
                                        source_l_W_m3: Optional[Union[float, Dict[str, float],
                                                                      object]] = None,
                                        Te_init_K: Optional[float] = None,
                                        Tl_init_K: Optional[float] = None, theta: float = 1.0,
                                        maxh_m: Optional[float] = None, order: int = 2,
                                        linear_solver: str = "umfpack",
                                        flux_of_t: Optional[Callable[[float], float]] = None,
                                        source_e_of_t: Optional[Callable[[float], object]] = None,
                                        source_l_of_t: Optional[Callable[[float], object]] = None,
                                        bottom_bc: str = "sink", store_every: int = 1,
                                        store_fields: bool = False
                                        ) -> ThermalTransientTwoTempResult:
    """Spatially-resolved two-temperature transient (R14): theta-method on the compound (V x V)
    space for

        C_e dTe/dt = div(k_e grad Te) - G (Te - Tl) + S_e(t)
        C_l dTl/dt = div(k_l grad Tl) + G (Te - Tl) + S_l(t)

    C_e is the layer's volumetric C_e_J_m3K, C_l = rho_kg_m3 * Cp_J_kgK (both required > 0). The
    top-face flux (constant or flux_of_t) heats the ELECTRON field; volumetric sources are per
    channel. bottom_bc = 'sink' (both fields Dirichlet T_sink_K, the default) or 'insulated' (pure
    Neumann everywhere -- the uniform-field configuration that reduces EXACTLY to the lumped
    carrier_heating.two_temperature_response ODE, the R9 oracle seam). theta = 1 backward-Euler,
    0.5 Crank-Nicolson. The G coupling makes the system stiff on the tau = 1/(G(1/C_e + 1/C_l))
    scale; backward-Euler is unconditionally stable across it. Sampled every store_every steps
    (plus t = 0 and the final step)."""
    if not (t_end_s > 0.0):
        raise ValueError("t_end_s must be > 0")
    if not (dt_s > 0.0):
        raise ValueError("dt_s must be > 0")
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must be in [0, 1] (1=backward-Euler, 0.5=Crank-Nicolson)")
    if any(L.rho_kg_m3 <= 0.0 or L.Cp_J_kgK <= 0.0 for L in layers):
        raise ValueError("two-temperature transient requires every layer rho_kg_m3 > 0 and "
                         "Cp_J_kgK > 0 (the lattice C_l)")
    if linear_solver not in ("umfpack", "sparsecholesky"):
        raise ValueError("linear_solver must be 'umfpack' or 'sparsecholesky', got {!r}".format(
            linear_solver))
    if bottom_bc not in ("sink", "insulated"):
        raise ValueError("bottom_bc must be 'sink' or 'insulated', got {!r}".format(bottom_bc))
    if store_every < 1:
        raise ValueError("store_every must be >= 1")

    n_steps = max(1, int(round(t_end_s / dt_s)))
    dt = t_end_s / n_steps
    time_dependent = (flux_of_t is not None) or (source_e_of_t is not None) \
        or (source_l_of_t is not None)

    mesh = _build_layered_mesh(layers, period_x_m, period_y_m, maxh_m)
    V, VV, ue, ul, ve, vl, a = _twotemp_space_and_forms(layers, mesh, order,
                                                        dirichlet_bot=(bottom_bc == "sink"))
    Ce_cf = _per_material_cf(mesh, {L.name: float(L.C_e_J_m3K) for L in layers}, "C_e_J_m3K")
    Cl_cf = _per_material_cf(mesh, {L.name: float(L.rho_kg_m3) * float(L.Cp_J_kgK)
                                    for L in layers}, "rho*Cp")
    m = ng.BilinearForm(VV)
    m += (Ce_cf / _S ** 2) * ue * ve * ng.dx
    m += (Cl_cf / _S ** 2) * ul * vl * ng.dx

    f = _twotemp_load(VV, ve, vl, mesh, flux_W_m2, source_e_W_m3, source_l_W_m3)

    def _load_at(t):
        fl = flux_of_t(t) if flux_of_t is not None else flux_W_m2
        se = source_e_of_t(t) if source_e_of_t is not None else source_e_W_m3
        sl = source_l_of_t(t) if source_l_of_t is not None else source_l_W_m3
        ff = _twotemp_load(VV, ve, vl, mesh, fl, se, sl)
        ff.Assemble()
        return ff

    # IC per component (default sink temperature); then patch the constrained sink dofs exactly as
    # the single-T path does (a constant is NOT all-dofs-equal for order >= 2 -- vertex dofs carry
    # the value, edge/face dofs are 0 -- so the patch copies from a boundary-Set helper, never
    # assigns the raw constant into every constrained slot).
    u = ng.GridFunction(VV)
    u.components[0].Set(ng.CoefficientFunction(float(T_sink_K if Te_init_K is None else Te_init_K)))
    u.components[1].Set(ng.CoefficientFunction(float(T_sink_K if Tl_init_K is None else Tl_init_K)))
    if bottom_bc == "sink":
        g_bot = ng.GridFunction(V)
        g_bot.Set(ng.CoefficientFunction(float(T_sink_K)), definedon=mesh.Boundaries("bot"))
        freeV = V.FreeDofs()
        maskV = np.array([not freeV[i] for i in range(len(freeV))])
        gvec = g_bot.vec.FV().NumPy()
        for comp in (0, 1):
            cvec = u.components[comp].vec.FV().NumPy()
            cvec[maskV] = gvec[maskV]

    t_list = [0.0]
    mean_te = [_mean_T_per_layer(mesh, u.components[0], layers)]
    mean_tl = [_mean_T_per_layer(mesh, u.components[1], layers)]
    snaps = [(_copy_component(V, u, 0), _copy_component(V, u, 1))] if store_fields else None

    with ng.TaskManager():
        a.Assemble(); m.Assemble()
        if not time_dependent:
            f.Assemble()
        S = m.mat.CreateMatrix()
        S.AsVector().data = m.mat.AsVector() + (theta * dt) * a.mat.AsVector()
        Sinv = S.Inverse(VV.FreeDofs(), inverse=linear_solver)
        rhs = u.vec.CreateVector()
        res = u.vec.CreateVector()

        f_old = f.vec if not time_dependent else _load_at(0.0).vec
        t = 0.0
        for step in range(1, n_steps + 1):
            t_new = t + dt
            f_new = f.vec if not time_dependent else _load_at(t_new).vec
            rhs.data = m.mat * u.vec - ((1.0 - theta) * dt) * (a.mat * u.vec) \
                + (dt * (1.0 - theta)) * f_old + (dt * theta) * f_new
            res.data = rhs - S * u.vec           # residual correction keeps the sink Dirichlet fixed
            u.vec.data += Sinv * res
            t = t_new
            f_old = f_new
            if (step % store_every == 0) or (step == n_steps):
                t_list.append(t)
                mean_te.append(_mean_T_per_layer(mesh, u.components[0], layers))
                mean_tl.append(_mean_T_per_layer(mesh, u.components[1], layers))
                if store_fields:
                    snaps.append((_copy_component(V, u, 0), _copy_component(V, u, 1)))

    return ThermalTransientTwoTempResult(
        mesh=mesh, layers=list(layers), t_s=np.asarray(t_list, dtype=np.float64),
        mean_Te_per_layer_t=np.asarray(mean_te, dtype=np.float64),
        mean_Tl_per_layer_t=np.asarray(mean_tl, dtype=np.float64),
        Te_final=_copy_component(V, u, 0), Tl_final=_copy_component(V, u, 1),
        flux_W_m2=float(flux_W_m2), T_sink_K=float(T_sink_K), snapshots=snaps)
