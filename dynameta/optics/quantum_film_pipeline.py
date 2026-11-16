"""Pipeline wiring of the self-consistent quantum-film response (optics.quantum_film).

What it connects
----------------
    carriers.DualGateSchrodingerPoisson  --solve(bias)-->  CarrierField (the film's n(z))
            |                                                   |
            | QuantumFilmCoupling.state(bias)                   | core.bridge.assemble_eps
            v                                                   v
    QuantumFilmState (complete sub-band set,  ----->  QuantumFilmOptical.eps_fields
    one QuantumFilmResponse per option set)           -> uniform uniaxial (3, 3) tensor
                                                      -> EpsField(tensor=diag(exx, exx, ezz))
                                                      -> any optical route that takes a tensor

The film material of the Design carries a QuantumFilmOptical (eps_inf, damping, exchange-correlation
choice, ...): that is the flag that selects the quantum response. Its permittivity is not a function of
the density alone -- the response needs the film's whole ground state (every sub-band of the converged
potential) -- so the per-bias ground state reaches it through the assembler's field bundle under the key
QUANTUM_FILM_FIELD, supplied by QuantumFilmCoupling.extra_fields (run_pipeline's `extra_fields`
callable). The coupling also IS a carrier solver (it wraps the dual-gate solver), so the documented call
is

    coupling = QuantumFilmCoupling(dual_gate_solver)
    rows = run_pipeline(design, sweep, carrier_solver=coupling, extra_fields=coupling.extra_fields,
                        optical_builder=film_layer_optics(design, "film", coupling),
                        optical_solver=make_fem_2d_tm_optical_solver())

or, equivalently, run_quantum_film_pipeline(design, sweep, coupling, film_layer="film",
optical_solver=...). Done by hand, the same numbers are

    sol = solver.solve_voltages(Vb, Vt)
    film = QuantumFilm.from_dual_gate(solver, sol)
    resp = QuantumFilmResponse(film, eps_inf, gamma, xc=...)
    exx, ezz, L = resp.effective_layer(2 pi c / lambda)
    eps_by_region = {"film": EpsField(tensor=np.diag([exx, exx, ezz]))}
    optical_solver(design, None, eps_by_region, lambda, n_super, n_sub)

(tests/test_quantum_film_pipeline.py checks that the two agree bit for bit).

What the optics receives: the EQUIVALENT LAYER. QuantumFilmResponse.effective_layer is the uniform
uniaxial layer of the film's thickness carrying the same sheet responses -- exact to first order in the
thickness. In a planar stack its error against the exact film element is 3e-7 .. 7e-5 in r on the
generic gated film (tests/test_quantum_film_pipeline.py, validation/quantum_film_rpa.py). Under lateral
structure the near field contains large in-plane wavevectors q, and the first-order layer degrades with
q L (docs/audit/2026-11-14-v0.16-quantum-film-wiring.md measures it on a strip cell against an exact
per-harmonic treatment, optics.quantum_film_cell). The exact response object stays available:
QuantumFilmCoupling.response(bias, optical) -> QuantumFilmResponse (transfer_matrix, alpha_perp, ...).

Caching. Every bias's ground state (the dual-gate solution, cached by the solver itself, and the complete
sub-band set) is built once; every (bias, option set) builds its QuantumFilmResponse once (the
frequency-independent part: the collective modes); each wavelength then costs only the
frequency-dependent sheet responses. The option set is the frozen QuantumFilmOptical itself, so its repr
-- which the optical-solver cache (dynameta.cache) folds into its key -- names eps_inf, both damping
rates, the exchange-correlation model and the numerical route.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.materials.optical_model import STACK_AXIS_FIELD, OpticalModel

__all__ = ["QUANTUM_FILM_FIELD", "QuantumFilmOptical", "QuantumFilmState", "QuantumFilmCoupling",
           "film_layer_optics", "run_quantum_film_pipeline"]

#: Field-bundle key under which QuantumFilmCoupling.extra_fields hands the assembler the film's ground
#: state at the current bias (a QuantumFilmState).
QUANTUM_FILM_FIELD = "quantum_film_state"

_XC = ("none", "x", "lda")
_ROUTES = ("auto", "modes", "real_space")


@dataclass(frozen=True)
class QuantumFilmOptical(OpticalModel):
    """Optical model of a film whose response is the self-consistent quantum (RPA / adiabatic-LDA)
    response of its sub-band system (optics.quantum_film.QuantumFilmResponse). Putting it on the film's
    Material is what selects the quantum response in the pipeline.

    Fields (the QuantumFilmResponse options): eps_inf (background permittivity of the film at optical
    frequency, real > 0); gamma_rad_s (damping of every transition); gamma_inplane_rad_s (in-plane Drude
    damping, None = gamma_rad_s); xc ('none' = RPA, 'x', 'lda'); eps_xc (background permittivity of the
    exchange-correlation interaction, None = eps_inf); occupation_tol; route ('auto', 'modes',
    'real_space'); max_pairs_modes. thickness_rtol: tolerance of the check that the carrier grid spans
    the film the ground state was computed on.

    The assembler (core.bridge.assemble_eps, through core.effects.OpticalModelEffect) calls eps_fields
    with the field bundle, which must carry fields[QUANTUM_FILM_FIELD] (QuantumFilmCoupling.extra_fields
    supplies it); the result is the equivalent uniaxial layer as ONE uniform (3, 3) tensor
    diag(eps_xx,eq, eps_xx,eq, eps_zz,eq) (z = the film normal = the stack axis). eps() without a ground
    state is refused, so a quantum film in a fixed-eps region fails loudly instead of being solved as
    something else."""
    eps_inf: float
    gamma_rad_s: float
    gamma_inplane_rad_s: Optional[float] = None
    xc: str = "none"
    eps_xc: Optional[float] = None
    occupation_tol: float = 1e-12
    route: str = "auto"
    max_pairs_modes: int = 3000
    thickness_rtol: float = 1e-9

    def __post_init__(self):
        if not (np.isfinite(float(self.eps_inf)) and float(self.eps_inf) > 0.0):
            raise ValueError("QuantumFilmOptical: eps_inf must be real and > 0, got {!r}".format(self.eps_inf))
        for nm in ("gamma_rad_s", "gamma_inplane_rad_s"):
            v = getattr(self, nm)
            if v is not None and not (np.isfinite(float(v)) and float(v) >= 0.0):
                raise ValueError("QuantumFilmOptical: {} must be finite and >= 0, got {!r}".format(nm, v))
        if self.xc not in _XC:
            raise ValueError("QuantumFilmOptical: xc must be one of {}, got {!r}".format(_XC, self.xc))
        if self.route not in _ROUTES:
            raise ValueError("QuantumFilmOptical: route must be one of {}, got {!r}".format(_ROUTES, self.route))

    # ---- the field-bundle surface the assembler uses (materials.optical_model conventions) ----
    @property
    def field_response(self) -> bool:
        """True: the assembler passes the whole field bundle to eps_fields."""
        return True

    @property
    def needs_position(self) -> bool:
        """True: the assembler supplies the carrier region's stack-axis coordinates (checked against
        the film the ground state belongs to)."""
        return True

    def response_kwargs(self) -> dict:
        """Keyword arguments of QuantumFilmResponse(film, **response_kwargs())."""
        return dict(eps_inf=float(self.eps_inf), gamma_rad_s=float(self.gamma_rad_s),
                    gamma_inplane_rad_s=(None if self.gamma_inplane_rad_s is None
                                         else float(self.gamma_inplane_rad_s)),
                    xc=self.xc, eps_xc=None if self.eps_xc is None else float(self.eps_xc),
                    occupation_tol=float(self.occupation_tol), route=self.route,
                    max_pairs_modes=int(self.max_pairs_modes))

    def eps(self, lambda_m: float, *, n_m3=None):
        raise ValueError(
            "QuantumFilmOptical.eps: the quantum-film response needs the film's ground state (its "
            "complete sub-band set), not a density alone. Run the permittivity assembler with "
            "extra_fields=QuantumFilmCoupling(...).extra_fields (run_pipeline / "
            "run_quantum_film_pipeline), or build the layer by hand with "
            "QuantumFilmResponse(...).effective_layer(omega). A quantum film in a fixed-eps region "
            "is not supported.")

    def eps_fields(self, fields: dict, lambda_m: float):
        """The equivalent uniaxial layer at lambda_m as one uniform (3, 3) tensor (see the class
        docstring)."""
        state = fields.get(QUANTUM_FILM_FIELD)
        if state is None:
            raise ValueError(
                "QuantumFilmOptical.eps_fields: the field bundle carries no {!r}; pass "
                "extra_fields=QuantumFilmCoupling(solver).extra_fields to run_pipeline / assemble_eps "
                "(or use run_quantum_film_pipeline).".format(QUANTUM_FILM_FIELD))
        if not isinstance(state, QuantumFilmState):
            raise TypeError("QuantumFilmOptical.eps_fields: fields[{!r}] must be a QuantumFilmState, got "
                            "{!r}".format(QUANTUM_FILM_FIELD, type(state).__name__))
        state.check_carrier_grid(fields.get("n"), fields.get(STACK_AXIS_FIELD),
                                 rtol=float(self.thickness_rtol))
        return state.eps_tensor(self, lambda_m)


class QuantumFilmState:
    """The film's ground state at one bias: the complete sub-band set (a QuantumFilm), the carrier
    solution it came from (optional), and one QuantumFilmResponse per option set (built on first use).

    film: optics.quantum_film.QuantumFilm; solution: the DualGateSolution (or None); key: a label
    (the gate-voltage pair when built by QuantumFilmCoupling)."""

    def __init__(self, film, solution=None, key=None):
        self.film = film
        self.solution = solution
        self.key = key
        self._responses: Dict[QuantumFilmOptical, object] = {}
        self._layers: "OrderedDict[Tuple[QuantumFilmOptical, float], Tuple[complex, complex, float]]" = \
            OrderedDict()
        self.stats = {"responses_built": 0, "layer_evaluations": 0, "layer_hits": 0}

    def __repr__(self) -> str:
        return "QuantumFilmState(key={!r}, thickness_m={!r}, sheet_density_m2={!r})".format(
            self.key, self.film.thickness_m, self.film.sheet_density_total_m2)

    def response(self, optical: QuantumFilmOptical):
        """The QuantumFilmResponse for this option set (built once, then reused)."""
        if not isinstance(optical, QuantumFilmOptical):
            raise TypeError("QuantumFilmState.response: options must be a QuantumFilmOptical")
        r = self._responses.get(optical)
        if r is None:
            from dynameta.optics.quantum_film import QuantumFilmResponse
            r = QuantumFilmResponse(self.film, **optical.response_kwargs())
            self._responses[optical] = r
            self.stats["responses_built"] += 1
        return r

    def effective_layer(self, optical: QuantumFilmOptical, lambda_m: float):
        """(eps_xx,eq, eps_zz,eq, L) at the vacuum wavelength lambda_m (cached per option set and
        wavelength)."""
        key = (optical, float(lambda_m))
        hit = self._layers.get(key)
        if hit is not None:
            self.stats["layer_hits"] += 1
            return hit
        omega = 2.0 * math.pi * C_LIGHT / float(lambda_m)
        out = self.response(optical).effective_layer(omega)
        self._layers[key] = out
        self.stats["layer_evaluations"] += 1
        while len(self._layers) > 4096:
            self._layers.popitem(last=False)
        return out

    def eps_tensor(self, optical: QuantumFilmOptical, lambda_m: float) -> np.ndarray:
        """diag(eps_xx,eq, eps_xx,eq, eps_zz,eq) as a complex (3, 3) array."""
        exx, ezz, _L = self.effective_layer(optical, lambda_m)
        return np.diag(np.array([exx, exx, ezz], dtype=np.complex128))

    def check_carrier_grid(self, n_grid, z_axis_m, *, rtol: float = 1e-9) -> None:
        """Refuse a state that does not belong to the carrier solution being assembled: the stack-axis
        span must equal the film's NOMINAL thickness (the carrier region is the film; an element on finite
        barriers reaches beyond it, and its optics describes the nominal film), and (when the state carries
        its solution) the density column must equal the solution's density. Catches a state handed in for
        the wrong bias or the wrong film."""
        L = float(self.film.nominal_thickness_m)
        if z_axis_m is not None:
            z = np.asarray(z_axis_m, dtype=np.float64).ravel()
            span = float(abs(z[-1] - z[0])) if z.size >= 2 else 0.0
            if abs(span - L) > rtol * L:
                raise ValueError(
                    "QuantumFilmOptical: the carrier region spans {:.9g} m along the stack axis but the "
                    "quantum-film state is for a {:.9g} m film (state {!r}); the state does not belong "
                    "to this carrier region".format(span, L, self.key))
        if n_grid is not None and self.solution is not None:
            n = np.asarray(n_grid, dtype=np.float64)
            col = n.reshape(-1, n.shape[-1])[0] if n.ndim >= 1 else n
            ref = np.asarray(self.solution.n_m3, dtype=np.float64)
            if col.shape != ref.shape or not np.allclose(col, ref, rtol=1e-12, atol=1e-12 * float(np.max(ref))):
                raise ValueError(
                    "QuantumFilmOptical: the density being assembled does not match the quantum-film "
                    "state {!r} (a state from another bias or another solver was supplied; build "
                    "extra_fields from the same QuantumFilmCoupling that solved the carriers)".format(
                        self.key))


class QuantumFilmCoupling:
    """Couples a carriers.DualGateSchrodingerPoisson solve to the quantum-film optical response.

    It is a CarrierSolver (regions() / solve(bias) delegate to the dual-gate solver, whose bias cache
    and warm starts are used unchanged) that also builds, per bias, the film's QuantumFilmState
    (QuantumFilm.from_dual_gate: the complete sub-band set of the converged potential), cached by the
    gate-voltage pair. extra_fields(bias) -> {QUANTUM_FILM_FIELD: state} is the run_pipeline
    `extra_fields` callable.

    solver: a DualGateSchrodingerPoisson (anything with solve_voltages, solve, regions, bias_keys,
    bottom, top and the attributes QuantumFilm.from_dual_gate reads); max_states: LRU size of the
    per-bias states (each holds the complete sub-band set and, once used, its responses);
    optical_penetration_m: QuantumFilm.from_dual_gate's (None = its default). With finite oxide barriers
    the film states' optics describes the NOMINAL film (QuantumFilm class docstring), so the Design is
    given as designed, its film layer at the film's nominal thickness."""

    def __init__(self, solver, *, max_states: int = 256, optical_penetration_m: Optional[float] = None):
        for attr in ("solve_voltages", "solve", "regions", "bias_keys", "E_F_J", "m_eff_kg"):
            if not hasattr(solver, attr):
                raise TypeError("QuantumFilmCoupling: solver lacks {!r}; pass a "
                                "carriers.DualGateSchrodingerPoisson".format(attr))
        self.solver = solver
        self.optical_penetration_m = optical_penetration_m      # QuantumFilm.from_dual_gate (None = default)
        self.max_states = max(1, int(max_states))
        self._states: "OrderedDict[Tuple[float, float], QuantumFilmState]" = OrderedDict()
        self.stats = {"states_built": 0, "state_hits": 0}

    # ---- CarrierSolver protocol (delegation) ----
    def regions(self):
        return self.solver.regions()

    def solve(self, bias, **kwargs):
        """The dual-gate CarrierField at this bias; the film's state is built at the same time."""
        if kwargs:
            raise TypeError("QuantumFilmCoupling.solve: the dual-gate solver takes no solve keywords, "
                            "got {}".format(sorted(kwargs)))
        cf = self.solver.solve(bias)
        self.state(bias)
        return cf

    def teardown(self) -> None:
        td = getattr(self.solver, "teardown", None)
        if td is not None:
            td()

    # ---- the film states ----
    @property
    def region_name(self) -> str:
        return str(getattr(self.solver, "region_name", "semi"))

    @property
    def film_material(self) -> str:
        return str(getattr(self.solver, "semi_material", ""))

    @property
    def cache_fingerprint(self) -> str:
        """The carrier-side inputs of the film states (the response options are on the material)."""
        s = self.solver
        parts = []
        for nm in ("t_film_m", "n_bg_m3", "m_eff_kg", "eps_static", "T_K", "alpha_np", "g_s", "g_v",
                   "nz", "bottom", "top", "gauss_tol_C_m2", "inner_tol_V"):
            parts.append("{}={!r}".format(nm, getattr(s, nm, None)))
        if self.optical_penetration_m is not None:
            parts.append("optical_penetration_m={!r}".format(self.optical_penetration_m))
        return "quantum_film_coupling(" + ",".join(parts) + ")"

    def voltages(self, bias) -> Tuple[float, float]:
        """(V_bottom, V_top) of a bias point (BiasPoint, or a (V_bottom, V_top) pair), normalized as
        the solver's cache key (an ungated face reads 0)."""
        s = self.solver
        if isinstance(bias, (tuple, list)) and len(bias) == 2:
            vb, vt = float(bias[0]), float(bias[1])
        else:
            vals = []
            for key, gate in zip(s.bias_keys, (getattr(s, "bottom", None), getattr(s, "top", None))):
                if gate is None:
                    vals.append(0.0)
                    continue
                if key not in bias.voltages:
                    raise ValueError("QuantumFilmCoupling: bias {!r} has no voltage for gate {!r} "
                                     "(voltages: {})".format(getattr(bias, "label", "?"), key,
                                                              sorted(bias.voltages)))
                vals.append(float(bias.voltages[key]))
            vb, vt = vals
        if getattr(s, "bottom", 1) is None:
            vb = 0.0
        if getattr(s, "top", 1) is None:
            vt = 0.0
        return vb, vt

    def state(self, bias) -> QuantumFilmState:
        """The film's ground state at this bias (built once per gate-voltage pair)."""
        key = self.voltages(bias)
        st = self._states.get(key)
        if st is not None:
            self._states.move_to_end(key)
            self.stats["state_hits"] += 1
            return st
        from dynameta.optics.quantum_film import QuantumFilm
        sol = self.solver.solve_voltages(*key)
        st = QuantumFilmState(QuantumFilm.from_dual_gate(self.solver, sol,
                                                         optical_penetration_m=self.optical_penetration_m),
                              solution=sol, key=key)
        self._states[key] = st
        self.stats["states_built"] += 1
        while len(self._states) > self.max_states:
            self._states.popitem(last=False)
        return st

    def extra_fields(self, bias) -> dict:
        """run_pipeline's extra_fields callable: {QUANTUM_FILM_FIELD: state at this bias}."""
        return {QUANTUM_FILM_FIELD: self.state(bias)}

    def response(self, bias, optical: QuantumFilmOptical):
        """The exact response object (QuantumFilmResponse) of the film at this bias for these options:
        transfer_matrix (exact planar element), alpha_perp, alpha_par, sigma_zz_kernel, validity, ..."""
        return self.state(bias).response(optical)

    def effective_layer(self, bias, optical: QuantumFilmOptical, lambda_m: float):
        """(eps_xx,eq, eps_zz,eq, L) at this bias and vacuum wavelength -- what the optics receives."""
        return self.state(bias).effective_layer(optical, lambda_m)


class _FilmLayerOptics:
    """OpticalGeometryBuilder stub for the layered / structured optical solvers (TMM, Berreman, RCWA,
    PMM, 2-D TM FEM), which take every non-film layer from the Design: the alignment maps the film
    layer to the carrier region; no mesh is built."""

    class _Geo:
        class mesh:
            ne = 0
            nv = 0

    def __init__(self, design, film_layer: str, source_region: str):
        self.design = design
        self.film_layer = str(film_layer)
        self.source_region = str(source_region)

    def build(self):
        return self._Geo()

    def alignment(self):
        from dynameta.core.alignment import GeometryAlignment, RegionAlignment
        from dynameta.core.units import NM
        P = float(self.design.unit_cell.period_x_m)
        Py = float(self.design.unit_cell.period_y_m)
        zlo, zhi = self.design.z_intervals()[self.film_layer]
        return GeometryAlignment(unit_scale=NM, region_alignments=[
            RegionAlignment(self.film_layer, self.source_region, (0.0, P, 0.0, Py, zlo, zhi),
                            stack_axis="y")], fixed_eps_regions={})

    def mesh_regions(self):
        return [self.film_layer]


def film_layer_optics(design, film_layer: str, carrier) -> _FilmLayerOptics:
    """An optical_builder for run_pipeline with a layered / structured optical solver: the film layer
    `film_layer` of `design` is fed by the carrier region of `carrier` (a QuantumFilmCoupling or a
    DualGateSchrodingerPoisson); all other layers come from the Design. Refuses a film layer whose
    thickness differs from the carrier film's (the assembler would otherwise stretch the profile onto
    the layer silently)."""
    names = [L.name for L in design.stack.layers]
    if film_layer not in names:
        raise ValueError("film_layer_optics: design has no layer {!r} (layers: {})".format(film_layer, names))
    solver = getattr(carrier, "solver", carrier)
    t = float(getattr(solver, "t_film_m", float("nan")))
    L = float(next(lay.thickness_m for lay in design.stack.layers if lay.name == film_layer))
    if not (abs(t - L) <= 1e-9 * L):
        raise ValueError("film_layer_optics: the carrier film is {!r} m thick but design layer {!r} is "
                         "{!r} m".format(t, film_layer, L))
    region = str(getattr(solver, "region_name", "semi"))
    return _FilmLayerOptics(design, film_layer, region)


def run_quantum_film_pipeline(design, sweep, carrier, *, film_layer: str,
                              optical_solver: Callable, optical_builder=None, extra_fields=None,
                              **run_kwargs):
    """run_pipeline for a dual-gate film: wraps `carrier` (a DualGateSchrodingerPoisson or a
    QuantumFilmCoupling) in a QuantumFilmCoupling, builds film_layer_optics(design, film_layer, ...)
    unless an optical_builder is given, and merges the coupling's extra_fields with the caller's
    (a dict or a callable of the bias). Returns (rows, coupling) -- the coupling keeps every bias's
    state, so coupling.response(bias, design.materials.get(name).optical) is the exact response
    behind each row.

    The film material is the carrier region's material (solver.semi_material): a QuantumFilmOptical
    gives the quantum response; any other optical model (e.g. DrudeOptical) gets the density profile as
    usual and ignores the extra field."""
    from dynameta.pipeline import run_pipeline
    coupling = carrier if isinstance(carrier, QuantumFilmCoupling) else QuantumFilmCoupling(carrier)
    mat = coupling.film_material
    if mat and mat not in design.materials:
        raise ValueError("run_quantum_film_pipeline: the carrier film's material {!r} is not in the "
                         "design's registry (set DualGateSchrodingerPoisson(semi_material=...))".format(mat))
    if optical_builder is None:
        optical_builder = film_layer_optics(design, film_layer, coupling)

    def _extra(bp):
        user = extra_fields(bp) if callable(extra_fields) else (extra_fields or {})
        if QUANTUM_FILM_FIELD in user:
            raise ValueError("run_quantum_film_pipeline: extra_fields key {!r} is reserved".format(
                QUANTUM_FILM_FIELD))
        out = dict(user)
        out.update(coupling.extra_fields(bp))
        return out

    rows = run_pipeline(design, sweep, carrier_solver=coupling, optical_builder=optical_builder,
                        optical_solver=optical_solver, extra_fields=_extra, **run_kwargs)
    return rows, coupling
