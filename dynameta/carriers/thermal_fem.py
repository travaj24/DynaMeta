"""
FEM heat-equation thermal driver + electro-thermal Joule coupling -- the volumetric/lateral
generalization of carriers.thermal (which is the exact series-thermal-resistance profile for a 1D
flux-driven stack). Solves the steady heat equation div(k grad T) = -Q on a layered box with the
bottom face at the sink temperature (Dirichlet), a heat flux into the top face (Neumann), and an
optional volumetric Joule source Q [W/m^3] (e.g. sigma|E|^2 from the electrical solve -- the
electro-thermal coupling). Returns the temperature field T [K] for the field bundle that
ThermoOpticModel reads. Reduces EXACTLY to carriers.thermal.steady_layered_temperature when Q = 0,
and to the uniform-Joule slab profile T_mean = T_sink + Q L^2/(3k) for a single heated layer.

ALSO provides the TRANSIENT heat equation rho*Cp*dT/dt = div(k grad T) + Q (roadmap R5) via a
theta-method time integrator (solve_thermal_transient_fem). The transient path requires every layer
to carry rho_kg_m3 > 0 and Cp_J_kgK > 0 (mass density and specific heat); the STEADY path never
reads them, so adding them is byte-identical for all existing steady callers (they default 0.0).
Typical material values (NOT stored as constants -- ThermalLayer is the home): Si rho=2329 Cp=700,
SiO2 rho=2200 Cp=730, ITO rho=7140 Cp=340 (kg/m^3, J/(kg K)). Requires NGSolve.

Layers are ordered from the SINK (index 0, at z = 0 / the bottom Dirichlet face) outward; the top
face (z = sum thicknesses) receives `flux_W_m2`. Units: the mesh is built in nm (coordinate =
metres * _S); with k in W/(m K), T in K, the SI weak form maps to mesh coordinates as
  int k gradT.gradv dV'  =  int (Q/_S^2) v dV'  +  int_top (flux/_S) v dS'
(the _S powers convert the SI source [W/m^3] and flux [W/m^2] into the nm-coordinate integrals).
The stiffness/load thus assemble as _S * (K_phys, f_phys); for the transient the mass term must
assemble as _S * M_phys too so the common _S cancels and dt stays in SI seconds. A plain
int rho*Cp*u*v*dV' integral equals _S^3 * M_phys, so the mass coefficient carries 1/_S^2:
  int (rho*Cp/_S^2) u v dV'  =  _S * int rho*Cp u v dV_phys  =  _S * M_phys.   (verified by R5 gates)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import netgen.occ as occ
import ngsolve as ng

_S = 1.0e9                       # mesh unit: coordinate = metres * _S (nm)


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
    and two-temperature paths; extracted verbatim from the original _build_thermal_forms body so the
    single-T solvers mesh byte-identically)."""
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
    return ng.Mesh(occ.OCCGeometry(glued).GenerateMesh(maxh=maxh))


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
