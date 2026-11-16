"""
Dual-gate Schrodinger-Poisson carrier solver for a thin degenerate semiconductor film gated from BOTH
faces through oxide stacks.

Geometry (z through the stack):

    bottom gate | bottom oxide stack | film, z in [0, t] | top oxide stack | top gate

The film is contacted laterally, so its electron Fermi level is the grounded reference: E_F is fixed at
the level at which the sub-band model's own flat-band bulk holds the background density n_bg
(schrodinger_poisson.sp_bulk_fermi_level_J; parabolic z dispersion, in-plane Kane DOS with
`alpha_np_per_eV`, finite temperature). The electrostatic potential phi(z) is measured so that the band
edge is E_c(z) = -q phi(z) with E_F - E_c = E_F_J where phi = 0. The net film charge is whatever the two
gates (and the fixed interface charges) induce; neutrality is NOT imposed.

Inside the film the solver is the validated SchrodingerPoisson1D slab-mode solve (hard walls psi = 0 on
both faces -- the oxide barriers -- all sub-bands kept, Dirichlet Poisson ends phi(0) = phi_b,
phi(t) = phi_t). The two face potentials are the unknowns of an outer 2 x 2 Newton that enforces Gauss's
law at both faces.

Boundary conditions (derivation)
--------------------------------
Each oxide stack is a series of charge-free layers (thickness t_i, static permittivity eps_i). In a
charge-free layer the displacement is uniform, and with no charge between the layers it is the same in
every layer, so the potential drop across the stack is G sum_i t_i / (eps0 eps_i) = G / C with
G = eps0 eps_i dphi/dz (the same in every layer) and C = 1 / sum_i t_i / (eps0 eps_i) the series
capacitance per area. The gate metal's electrostatic potential is phi_g = V - V_wf, with V the applied
voltage (gate Fermi level -qV relative to the film's) and V_wf = (W_gate - W_film) / q the work-function
offset (W_film = chi + E_c - E_F of the flat-band film). With a fixed sheet charge sigma (C/m^2) at the
film/oxide interface, Gauss's law for a pillbox on the interface, D_z(face+) - D_z(face-) = sigma with
D_z = -eps0 eps dphi/dz, gives

    top face (z = t, oxide above):     F_t - C_t (phi_gt - phi_t) - sigma_t = 0
    bottom face (z = 0, oxide below):  C_b (phi_b - phi_gb) - F_b - sigma_b = 0

where F = eps0 eps_s dphi/dz on the FILM side of the face (signed along +z at both faces). F is taken
from the conservative flux of the solver's own Poisson stencil (schrodinger_poisson.sp_face_flux_C_m2:
the half control volume of the end node, eps_s (phi_1 - phi_0) / h + q (h/2)(Nd_0 - n_0) at z = 0 and
eps_s (phi_N - phi_{N-1}) / h - q (h/2)(Nd_N - n_N) at z = t), so the discrete Gauss law across the
film, F_t - F_b = q trapz(n - Nd), is exact and the totals balance:

    Q_gate_b + Q_gate_t + sigma_b + sigma_t + Q_film = 0,
    Q_gate = C (phi_g - phi_face),  Q_film = q trapz(Nd - n).

Both residuals increase with their own face potential (dR/dphi_face = C + |dF/dphi_face| > 0). A face
with no gate (`None`) has C = 0: Gauss's law then reads F = -sigma at the bottom (+sigma at the top),
i.e. a bare dielectric face with zero displacement beyond it. Charge placed inside an oxide stack is not
modelled; a known amount enters as a shift of V_wf.

Outer solve: Newton on (phi_b, phi_t) with a forward-difference Jacobian (two extra inner solves),
Broyden updates between refreshes, a step cap and backtracking on max|R|; it stops at
max|R| < gauss_tol_C_m2 with every inner solve converged. Bias sweeps reuse cached solutions: each solved
bias is kept (LRU, `cache_size`) with its face potentials, profile and Jacobian; a new bias starts from
the nearest cached one with the tangent predictor dx = J^-1 diag(C_b, C_t) dV and a warm-started inner
solve, and walks there by adaptive continuation (step halving) if the direct Newton fails. If even
the smallest step fails, DualGateConvergenceError is raised -- never a silently wrong answer.

Output: a CarrierField(ndim=3) exactly like SchrodingerPoissonCarrier's: the through-film profile on the
solver's native uniform z grid, broadcast over the lateral cell, which the bridge (core.bridge.
assemble_eps, native 3D branch) consumes directly.

Optional physics (v0.16, all OFF by default; with every option off the solve is the one above, bit
for bit)
------------------------------------------------------------------------------------------------------
1. Finite oxide barriers (GateStack(barrier=OxideBarrier(height_eV, mass_kg, penetration_m))). The
   Schrodinger-Poisson domain is extended a distance d (a whole number of film grid cells) into the oxide
   on that side. Material properties are per CELL (permittivity, z mass) and per node (conduction-band
   offset), with the film/oxide interface ON a node: the kinetic operator is the finite-volume
   BenDaniel-Duke form (BenDaniel and Duke 1966), whose discrete flux (1/m) dpsi/dz is single-valued at
   every cell face (the interface condition), with the interface node's potential the control-volume
   average of the two sides; the matrix is symmetric and the eigenvalues converge at O(h^2). psi = 0 at
   the far end of the penetration region (where the tail has decayed to round-off by default). Poisson
   runs over the whole domain in control-volume form, so electron charge in the oxide is a Poisson source
   there; the interface node holds half the film doping, and the fixed interface charge and the interface
   traps become sheet sources ON the interface node (they sit at the interface plane, now inside the
   domain). The Gauss matching plane moves to the outer end of the penetration region: beyond it the
   remaining oxide (possibly several layers, possibly partial) is charge-free, so its series capacitance
   C_rem carries the displacement to the gate and
       bottom:  C_rem,b (phi_m,b - phi_gb) - F_m,b = 0,     top:  F_m,t + C_rem,t (phi_m,t - phi_gt) = 0
   with F_m the conservative flux of the domain's end cell (oxide permittivity; n = 0 and N_d = 0 at the
   wall node). The outer Newton unknowns are the two matching-plane potentials. When the penetration
   reaches the gate (C_rem infinite) the matching plane IS the gate: phi_m = phi_g. The totals balance
   exactly on the discrete grid:
       Q_gate_b + Q_gate_t + sigma_b + sigma_t + Q_it,b + Q_it,t + q trapz_domain(N_d - n) = 0.
   With `penetration_m=0` (or no barrier) a face keeps the hard wall and the face law above.
2. Exchange-correlation in the ground state (xc="x" or "lda"): the local-density potential of the electron
   gas (exchange, or exchange + Perdew-Zunger 1981 correlation) in effective atomic units of the host
   (film mass m_eff_kg, permittivity xc_eps_r; default the film's static eps_static -- the choice of which
   permittivity screens the exchange interaction is ambiguous for a polar host: the static value includes
   the lattice polarization, which responds on the phonon time scale and is the usual choice for
   ground-state calculations (Ando, Fowler and Stern 1982), the high-frequency value is the electronic
   screening only; pass xc_eps_r to choose), added to the Schrodinger potential energy and carried with
   its derivative in the inner Newton (SchrodingerPoisson1D.solve_self_consistent(xc_potential=...)).
   The functional is optics.quantum_film's (lda_xc_potential / lda_xc_kernel share ONE implementation of
   the xc energy density and its derivatives), so the optical response kernel is the same functional.
   The potential used is V_xc(n(z)) - V_xc(n_bg): the uniform-gas shift V_xc(n_bg) is a constant that
   only renormalizes the band edge of the flat-band bulk, so subtracting it keeps the model's bulk density
   at E_F equal to n_bg (E_F stays sp_bulk_fermi_level_J; the measured film work function, and so V_wf,
   already contain the shift). The xc potential is evaluated with the film's units in the oxide
   penetration regions as well.
   Barrier height convention with xc and a finite barrier. Every potential energy of the model is measured
   from the RENORMALIZED conduction-band edge of the flat-band film (the bare edge plus V_xc(n_bg) < 0, the
   band-gap renormalization of the degenerate gas). In the oxide, where n -> 0 and V_xc(n) -> 0, the
   Hamiltonian therefore carries OxideBarrier.height_eV - V_xc(n_bg)/q, i.e. the barrier seen from the film
   is RAISED by |V_xc(n_bg)| (114 meV for 3e26 m^-3, 0.30 m_e, eps 9, LDA; 80 meV at 1e26 m^-3). That is the correct barrier when
   height_eV is the offset between the band edges WITHOUT the film's electron gas -- the usual tabulated
   conduction-band offset, or Anderson's rule with the undoped film's electron affinity -- because the gas
   lowers the film's edge and not the oxide's; dropping the term would lose it. If instead a barrier was
   measured on the degenerate film itself relative to its own (renormalized) band edge, it already contains
   the shift: pass height_eV = H_measured + V_xc(n_bg)/q (V_xc(n_bg) =
   optics.quantum_film.lda_xc_potential([n_bg], m_eff_kg, xc_eps_r, model=xc)[0], negative), or the
   shift is counted twice. With xc=None no such shift exists and height_eV is used as given.
3. Interface traps (GateStack(traps=physics_traps.FilmInterfaceTraps(...)), the classical path's spec and
   closed-form occupancy reused by import). The trapped sheet charge Q_it follows the Fermi-Dirac
   occupancy of the film's Fermi level at the ELECTROSTATIC potential of the interface plane, phi_s:
   eta = (E_F - E_c,s)/kT = (E_F_J + q phi_s)/kT, with trap energies measured from the film's conduction
   band edge at the interface (the flat-band bulk edge the solver's E_F_J refers to, moved rigidly by
   -q phi_s; neither the band offset nor the local V_xc moves the trap levels). With hard walls phi_s is
   the face potential, the outer unknown: Q_it(phi_s) enters that face's Gauss residual and its exact
   derivative dQ_it/dphi_s the outer Jacobian. With a finite barrier phi_s is the interface-node potential
   inside the domain: Q_it is an inner-Newton sheet source there with its exact derivative.

The film/oxide split of the reported density (finite barriers): DualGateSolution.z_m / phi_V / n_m3 and
the CarrierField region hold the FILM part, z in [0, t], on the film grid (the face nodes now carry
nonzero density); the full profile on the extended Schrodinger domain is z_ext_m / phi_ext_V / n_ext_m3,
and the electrons in the two penetration regions are oxide_electron_charge_* (also in the CarrierField
extras). The film part integrates (trapz on [0, t]) to the in-film charge; the oxide parts integrate on
[-d_b, 0] and [t, t + d_t]; the three add to the domain total exactly.
"""

from __future__ import annotations

import math
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from dynameta.constants import EPS0, HBAR, KB, Q_E as Q, T_REF
from dynameta.core.carrier_field import CarrierField, CarrierRegion, ELECTRON_DENSITY, POTENTIAL
from dynameta.core.interfaces import RegionInfo
from dynameta.core.numerics import trapz
from dynameta.carriers.schrodinger_poisson import (
    SchrodingerPoisson1D, SubbandResult, sp_bulk_fermi_level_J, sp_face_flux_C_m2)
from dynameta.carriers.physics_traps import FilmInterfaceTraps
from dynameta.warning_types import AdvisoryWarning, UnreliableResultWarning

__all__ = ["GateStack", "OxideBarrier", "DualGateSolution", "DualGateConvergenceError",
           "DualGateSchrodingerPoisson", "trap_charge_and_derivative"]

# the inner solver's non-convergence message (solve_self_consistent); recognised so that a failed
# inner solve becomes a rejected Newton step / continuation retry instead of a stray warning
_INNER_NONCONV = "SchrodingerPoisson.solve_self_consistent did NOT converge"
# its sub-band completeness guard (density()); in the option mode the solver checks completeness itself
_INNER_TRUNC = "SchrodingerPoisson.density: highest solved sub-band"
# tail decay of the default penetration depth: the density falls by e^-(2 * 18.42) = 1e-16 (round-off)
_TAIL_EXPONENT = 0.5 * math.log(1e16)


class DualGateConvergenceError(RuntimeError):
    """The dual-gate Schrodinger-Poisson solve did not reach Gauss's law at both faces (or an inner
    self-consistent solve failed) even with the smallest continuation step."""


def _per_layer(v, n, name):
    if np.isscalar(v):
        return (float(v),) * n
    v = tuple(float(x) for x in v)
    if len(v) != n:
        raise ValueError("OxideBarrier.{}: {} values for {} oxide layers".format(name, len(v), n))
    return v


@dataclass(frozen=True)
class OxideBarrier:
    """Finite conduction-band barrier that an oxide stack presents to the film's electrons (v0.16).

    height_eV: conduction-band offset E_c(oxide) - E_c(film) [eV] > 0, one value for every layer of the
        stack or a tuple per layer (layers listed from the film outward). With xc on, the offset between
        the band edges without the film's electron gas (module docstring, "Barrier height convention": the
        solver adds |V_xc(n_bg)| to it).
    mass_kg: the oxide's z effective mass [kg] > 0 (BenDaniel-Duke), scalar or per layer.
    penetration_m: how far the Schrodinger-Poisson domain extends into the stack [m]; it is rounded to a
        whole number of film grid cells and may span several layers, up to the whole stack (then the
        matching plane is the gate). None (default) = automatic: the depth over which the evanescent
        tail of a state at E_F + 10 kT decays in density by 1e-16 (int kappa dz = ln(1e16)/2, kappa =
        sqrt(2 m (V_b - E)) / hbar per layer at flat band), capped at the stack. 0 = hard wall (the
        solver then takes its hard-wall code path on that face)."""
    height_eV: Union[float, Tuple[float, ...]]
    mass_kg: Union[float, Tuple[float, ...]]
    penetration_m: Optional[float] = None

    def __post_init__(self):
        for name in ("height_eV", "mass_kg"):
            v = getattr(self, name)
            vals = (v,) if np.isscalar(v) else tuple(v)
            if not vals or not all(np.isfinite(float(x)) and float(x) > 0.0 for x in vals):
                raise ValueError("OxideBarrier.{} must be finite and > 0 (scalar or per layer), got "
                                 "{!r}".format(name, v))
            object.__setattr__(self, name, float(v) if np.isscalar(v) else tuple(float(x) for x in vals))
        if self.penetration_m is not None:
            d = float(self.penetration_m)
            if not (np.isfinite(d) and d >= 0.0):
                raise ValueError("OxideBarrier.penetration_m must be None or finite >= 0, got "
                                 "{!r}".format(self.penetration_m))
            object.__setattr__(self, "penetration_m", d)


@dataclass(frozen=True)
class GateStack:
    """One gate and the oxide stack between it and the film.

    layers: the oxide layers as ((thickness_m, eps_r_static), ...); a single (thickness_m, eps_r) pair is
        accepted. Without a barrier only the series capacitance C = 1 / sum t_i / (eps0 eps_i) enters, so
        the order does not matter; with a barrier the layers are listed FROM THE FILM OUTWARD.
    work_function_offset_V: V_wf = (W_gate - W_film) / q; the gate metal sits at the electrostatic
        potential V - V_wf. A flat-band voltage quoted for a gate without interface charge is this
        number.
    interface_charge_C_m2: fixed sheet charge at the film/oxide interface (C/m^2, positive = positive
        charge, e.g. fixed oxide charge).
    barrier: OxideBarrier or None (default: hard wall, psi = 0 at the interface).
    traps: physics_traps.FilmInterfaceTraps or None (default: no interface traps); energies E - E_c of
        the film's conduction-band edge at the interface plane."""
    layers: Tuple[Tuple[float, float], ...]
    work_function_offset_V: float = 0.0
    interface_charge_C_m2: float = 0.0
    barrier: Optional[OxideBarrier] = None
    traps: Optional[FilmInterfaceTraps] = None

    def __post_init__(self):
        lay = self.layers
        if (isinstance(lay, (tuple, list)) and len(lay) == 2
                and all(np.isscalar(v) for v in lay)):
            lay = (tuple(lay),)
        try:
            lay = tuple((float(t), float(e)) for t, e in lay)
        except (TypeError, ValueError):
            raise ValueError("GateStack.layers must be ((thickness_m, eps_r), ...), got "
                             "{!r}".format(self.layers)) from None
        if not lay:
            raise ValueError("GateStack.layers must name at least one oxide layer")
        for t, e in lay:
            if not (np.isfinite(t) and t > 0.0 and np.isfinite(e) and e > 0.0):
                raise ValueError("GateStack: every layer needs a finite thickness > 0 (m) and a "
                                 "static permittivity > 0, got ({!r}, {!r})".format(t, e))
        object.__setattr__(self, "layers", lay)
        for name in ("work_function_offset_V", "interface_charge_C_m2"):
            v = float(getattr(self, name))
            if not np.isfinite(v):
                raise ValueError("GateStack.{} must be finite, got {!r}".format(name, v))
            object.__setattr__(self, name, v)
        if self.barrier is not None:
            if not isinstance(self.barrier, OxideBarrier):
                raise TypeError("GateStack.barrier must be an OxideBarrier or None")
            _per_layer(self.barrier.height_eV, len(lay), "height_eV")
            _per_layer(self.barrier.mass_kg, len(lay), "mass_kg")
            if (self.barrier.penetration_m is not None
                    and self.barrier.penetration_m > sum(t for t, _e in lay) * (1.0 + 1e-12)):
                raise ValueError("OxideBarrier.penetration_m = {!r} m exceeds the oxide stack ({!r} m)".format(
                    self.barrier.penetration_m, sum(t for t, _e in lay)))
        if self.traps is not None and not isinstance(self.traps, FilmInterfaceTraps):
            raise TypeError("GateStack.traps must be a physics_traps.FilmInterfaceTraps or None")

    @property
    def capacitance_F_m2(self) -> float:
        """Series capacitance per area of the oxide stack, 1 / sum t_i / (eps0 eps_i) [F/m^2]."""
        return 1.0 / sum(t / (EPS0 * e) for t, e in self.layers)


def trap_charge_and_derivative(traps: FilmInterfaceTraps, phi_s_V: float, E_F_J: float,
                               T_K: float) -> Tuple[float, float]:
    """(Q_it [C/m^2], dQ_it/dphi_s [F/m^2]) of an interface-trap spec at the electrostatic potential
    phi_s of the interface plane, for a film whose Fermi level sits E_F_J above its flat-band conduction
    band edge: eta = (E_F_J + q phi_s) / kT. Charge and exact derivative are the spec's own kernel
    (FilmInterfaceTraps.sheet_charge_and_derivative, which the classical path's sheet_charge_C_m2 also
    uses -- one implementation), dQ/dphi_s = dQ/deta / V_t."""
    kT = KB * float(T_K)
    Vt = kT / Q
    eta = (float(E_F_J) + Q * float(phi_s_V)) / kT
    q_it, dq_deta = traps.sheet_charge_and_derivative(eta, V_t=Vt)
    return float(q_it), float(dq_deta) / Vt


@dataclass
class DualGateSolution:
    """One converged dual-gate bias point. Charges per area [C/m^2]; potentials [V].

    With finite barriers the film fields (z_m, phi_V, n_m3) are the in-film part on the film grid, and
    the *_ext fields carry the whole Schrodinger-Poisson domain (film + oxide penetration regions); the
    face potentials are those of the film/oxide interface planes and face_flux_* / gauss_residual_* refer
    to the Gauss matching planes (the domain ends; the faces themselves without a barrier)."""
    V_bottom: float
    V_top: float
    z_m: np.ndarray                 # native uniform grid, z = 0 bottom face, z = t top face
    phi_V: np.ndarray               # electrostatic potential (E_c = -q phi; phi = 0 is the flat-band bulk)
    n_m3: np.ndarray                # electron density (0 at the two hard-wall face nodes)
    phi_bottom_V: float
    phi_top_V: float
    face_flux_bottom_C_m2: float    # eps0 eps_s dphi/dz on the film side, along +z
    face_flux_top_C_m2: float
    gate_charge_bottom_C_m2: float  # C_b (phi_gb - phi_b); 0 for an ungated face
    gate_charge_top_C_m2: float
    interface_charge_bottom_C_m2: float
    interface_charge_top_C_m2: float
    film_charge_C_m2: float         # q trapz(Nd - n)
    gauss_residual_bottom_C_m2: float
    gauss_residual_top_C_m2: float
    newton_iterations: int
    inner_solves: int
    continuation_steps: int
    solve_time_s: float
    subbands: Optional[SubbandResult] = None
    from_cache: bool = False
    # ---- v0.16 options (defaults = no option active) ----
    trap_charge_bottom_C_m2: float = 0.0          # interface-trap sheet charge Q_it at each face
    trap_charge_top_C_m2: float = 0.0
    oxide_electron_charge_bottom_C_m2: float = 0.0  # -q int n over the penetration region (finite barrier)
    oxide_electron_charge_top_C_m2: float = 0.0
    phi_match_bottom_V: Optional[float] = None     # potential at the Gauss matching planes (finite barrier)
    phi_match_top_V: Optional[float] = None
    z_ext_m: Optional[np.ndarray] = None           # the Schrodinger-Poisson domain (any option active)
    phi_ext_V: Optional[np.ndarray] = None
    n_ext_m3: Optional[np.ndarray] = None
    U_ext_J: Optional[np.ndarray] = None           # Schrodinger potential energy: band offset - q phi + V_xc
    xc_potential_ext_J: Optional[np.ndarray] = None
    inv_mass_ext_per_kg: Optional[np.ndarray] = None   # per-cell inverse z mass (finite barrier)
    tail_density_ratio: Optional[float] = None     # n next to the far walls / max n (finite barrier)

    @property
    def neutrality_residual_C_m2(self) -> float:
        """Q_gate_b + Q_gate_t + sigma_b + sigma_t + Q_it,b + Q_it,t + Q_film + Q_oxide electrons: zero for
        a self-consistent solution (the discrete Gauss law across the domain; its size measures the inner
        Schrodinger-Poisson self-consistency, not the face law)."""
        return (self.gate_charge_bottom_C_m2 + self.gate_charge_top_C_m2
                + self.interface_charge_bottom_C_m2 + self.interface_charge_top_C_m2
                + self.film_charge_C_m2 + self.trap_charge_bottom_C_m2 + self.trap_charge_top_C_m2
                + self.oxide_electron_charge_bottom_C_m2 + self.oxide_electron_charge_top_C_m2)

    @property
    def sheet_excess_m2(self) -> float:
        """Electron sheet excess over the uniform background IN THE FILM, trapz(n - Nd) on [0, t] [m^-2]."""
        return -self.film_charge_C_m2 / Q

    @property
    def sheet_excess_total_m2(self) -> float:
        """Electron sheet excess including the electrons in the oxide penetration regions [m^-2]."""
        return -(self.film_charge_C_m2 + self.oxide_electron_charge_bottom_C_m2
                 + self.oxide_electron_charge_top_C_m2) / Q


@dataclass
class _State:
    """A converged point kept for warm starts: bias, face potentials, profile, Jacobian."""
    V: np.ndarray
    x: np.ndarray
    J: np.ndarray
    phi: np.ndarray
    sol: DualGateSolution = field(repr=False, default=None)
    n: Optional[np.ndarray] = field(repr=False, default=None)


def _as_gate(g):
    if g is None or isinstance(g, GateStack):
        return g
    return GateStack(layers=g)


def _piecewise_mean(edges, values, a, b):
    """Mean over [a, b] of a piecewise-constant profile (values[i] on [edges[i], edges[i+1]])."""
    lo = np.clip(edges[:-1], a, b)
    hi = np.clip(edges[1:], a, b)
    return float(np.sum((hi - lo) * values) / (b - a))


class DualGateSchrodingerPoisson:
    """Self-consistent Schrodinger-Poisson CarrierSolver for a film gated through oxide on both faces.

    Implements regions() + solve(bias) -> CarrierField(ndim=3), the interface SchrodingerPoissonCarrier
    satisfies; `solve_voltages(V_bottom, V_top)` returns the full DualGateSolution. bias.voltages must
    carry both gate voltages under `bias_keys` (default ("bottom", "top")); an ungated face (None) needs
    no key. Solutions are cached per bias (LRU) and every new bias is warm-started from the nearest
    cached one, so a sweep costs a few warm inner solves per point.

    Parameters (SI): t_film_m film thickness; bottom / top a GateStack, a (thickness_m, eps_r) pair or a
    sequence of them, or None for an ungated face; n_bg_m3 uniform ionized donor density (= the bulk
    electron density at flat band); m_eff_kg effective mass (z dispersion parabolic); eps_static the
    film's static permittivity; alpha_np_per_eV in-plane Kane nonparabolicity [eV^-1] (0 = parabolic);
    nz uniform grid nodes over the film (h = t / (nz - 1); 0.0125 nm resolves the dead layer of a
    1e26-1e27 m^-3 film to ~1e-5 V); n_states sub-bands per solve (None = a rigorous count from the
    potential: every state below E_F + 30 kT plus one); gauss_tol_C_m2 the stop on max|R| of the two
    face laws (the default 1e-9 C/m^2 leaves the face potentials ~3e-9 V and the density ~4e-9 n_bg from the
    converged solution; 1e-11 brings that to 1e-10 V / 1.4e-10 n_bg for ~10 % more inner solves in a warm
    sweep -- MEASURED on a 6 nm, 3e26 m^-3 film, nine biases); inner_tol_V the inner self-consistency
    tolerance.

    v0.16 options (module docstring): finite barriers and interface traps per gate (GateStack.barrier,
    GateStack.traps); xc = None (default) | "x" | "lda" exchange-correlation in the ground state, with
    xc_eps_r the permittivity of its effective atomic units (default eps_static).

    The cache is keyed by the options too (a GateStack or xc change after construction rebuilds the
    derived grids and uses a separate cache); other parameters must not be mutated after construction.
    """

    def __init__(self, *, t_film_m: float, bottom, top,
                 n_bg_m3: float, m_eff_kg: float, eps_static: float,
                 T_K: float = T_REF, alpha_np_per_eV: float = 0.0, g_s: int = 2, g_v: int = 1,
                 nz: int = 401, n_states: Optional[int] = None,
                 bias_keys: Tuple[str, str] = ("bottom", "top"),
                 lateral_m: float = 12e-9, n_lateral: int = 4, semi_material: str = "film",
                 region_name: str = "semi", gauss_tol_C_m2: float = 1e-9, inner_tol_V: float = 1e-11,
                 max_newton: int = 30, max_halvings: int = 10, max_step_V: float = 0.5,
                 cache_size: int = 128, xc: Optional[str] = None,
                 xc_eps_r: Optional[float] = None) -> None:
        self.t_film_m = float(t_film_m)
        if not (np.isfinite(self.t_film_m) and self.t_film_m > 0.0):
            raise ValueError("DualGateSchrodingerPoisson: t_film_m must be > 0, got {!r}".format(t_film_m))
        self.bottom = _as_gate(bottom)
        self.top = _as_gate(top)
        if self.bottom is None and self.top is None:
            raise ValueError("DualGateSchrodingerPoisson: at least one face needs a gate")
        self.n_bg_m3 = float(n_bg_m3)
        self.m_eff_kg = float(m_eff_kg)
        self.eps_static = float(eps_static)
        if not (np.isfinite(self.eps_static) and self.eps_static > 0.0):
            raise ValueError("DualGateSchrodingerPoisson: eps_static (the film's static relative permittivity) must "
                             "be finite and > 0, got {!r}".format(eps_static))
        self.T_K = float(T_K)
        self.alpha_np = float(alpha_np_per_eV)
        if self.alpha_np < 0.0:
            raise ValueError("DualGateSchrodingerPoisson: alpha_np_per_eV must be >= 0")
        self.g_s, self.g_v = int(g_s), int(g_v)
        self.nz = int(nz)
        if self.nz < 11:
            raise ValueError("DualGateSchrodingerPoisson: nz must be >= 11, got {}".format(nz))
        self.n_states = None if n_states is None else int(n_states)
        if len(bias_keys) != 2 or bias_keys[0] == bias_keys[1]:
            raise ValueError("DualGateSchrodingerPoisson: bias_keys must be two distinct names")
        self.bias_keys = (str(bias_keys[0]), str(bias_keys[1]))
        self.lateral_m = float(lateral_m)
        self.n_lateral = int(n_lateral)
        self.semi_material = semi_material
        self.region_name = str(region_name)
        self.gauss_tol_C_m2 = float(gauss_tol_C_m2)
        self.inner_tol_V = float(inner_tol_V)
        self.max_newton = int(max_newton)
        self.max_halvings = int(max_halvings)
        self.max_step_V = float(max_step_V)
        self.cache_size = max(1, int(cache_size))
        self.xc = xc
        self.xc_eps_r = xc_eps_r

        self.z_m = np.linspace(0.0, self.t_film_m, self.nz)
        self._h = float(self.z_m[1] - self.z_m[0])
        self._sp = SchrodingerPoisson1D(self.z_m, self.m_eff_kg, T_K=self.T_K, g_s=self.g_s, g_v=self.g_v)
        self._Nd = np.full(self.nz, self.n_bg_m3)
        # the ONE bulk Fermi-level kernel of the Schrodinger-Poisson path (the model's own flat-band bulk)
        self.E_F_J = sp_bulk_fermi_level_J(self.n_bg_m3, self.m_eff_kg, self.T_K, g_s=self.g_s,
                                           g_v=self.g_v, alpha_np_per_eV=self.alpha_np)
        # exact hard-wall kinetic eigenvalues of the discrete operator (interior nodes), for the
        # rigorous sub-band count: E_j(T + U) >= E_j(T) + min U (min-max)
        c = HBAR ** 2 / (2.0 * self.m_eff_kg * self._h ** 2)
        k = np.arange(1, self.nz - 1)
        self._E_kin = 4.0 * c * np.sin(k * np.pi / (2.0 * (self.nz - 1))) ** 2
        self._caches: Dict[str, "OrderedDict[Tuple[float, float], _State]"] = {}
        self.stats = {"inner_solves": 0, "biases_solved": 0, "cache_hits": 0}
        self._setup()

    # ---- options ----
    def _fingerprint(self):
        return (self.bottom, self.top, self.xc, self.xc_eps_r)

    def _setup(self) -> None:
        """Derived per-option state: gate arrays and, when an option is active, the extended
        Schrodinger-Poisson domain (grid, cell permittivities and masses, node band offsets and doping,
        interface-node indices, matching-plane capacitances) and the xc functional."""
        gates = (self.bottom, self.top)
        if self.bottom is None and self.top is None:
            raise ValueError("DualGateSchrodingerPoisson: at least one face needs a gate")
        self._C = np.array([0.0 if g is None else g.capacitance_F_m2 for g in gates])
        self._wf = np.array([0.0 if g is None else g.work_function_offset_V for g in gates])
        self._sigma = np.array([0.0 if g is None else g.interface_charge_C_m2 for g in gates])
        self._traps = [None if (g is None or g.traps is None or g.traps.is_empty()) else g.traps
                       for g in gates]
        xc = None if self.xc in (None, "none") else self.xc
        if xc not in (None, "x", "lda"):
            raise ValueError("DualGateSchrodingerPoisson: xc must be None, 'x' or 'lda', got {!r}".format(self.xc))
        if xc is not None and (self.g_s, self.g_v) != (2, 1):
            # the LDA functional (optics.quantum_film) is that of the spin-unpolarized single-valley gas. For
            # g = g_s g_v equally populated species, exchange acts within each species of density n / g, i.e.
            # eps_x(n; g) = eps_x(2 n / g; 2) -- 2^(1/3) weaker than the formula used here at g_v = 2 -- and
            # the correlation energy is not that of the unpolarized gas at all; neither is implemented.
            raise ValueError("DualGateSchrodingerPoisson: xc={!r} is the spin-unpolarized single-valley LDA "
                             "(g_s = 2, g_v = 1); got g_s = {}, g_v = {}, for which that functional is wrong "
                             "(exchange scales as (2 / (g_s g_v))^(1/3) and the correlation differs). Use xc=None "
                             "for this band.".format(xc, self.g_s, self.g_v))
        self._xc_model = xc
        self._xc_eps = self.eps_static if self.xc_eps_r is None else float(self.xc_eps_r)
        if not (np.isfinite(self._xc_eps) and self._xc_eps > 0.0):
            raise ValueError("DualGateSchrodingerPoisson: xc_eps_r must be finite and > 0")
        h = self._h
        kT = KB * self.T_K
        pen = [0, 0]
        self._barrier_layers = [None, None]
        for side, g in enumerate(gates):
            if g is None or g.barrier is None or g.barrier.penetration_m == 0.0:
                continue
            nl = len(g.layers)
            hts = _per_layer(g.barrier.height_eV, nl, "height_eV")
            ms = _per_layer(g.barrier.mass_kg, nl, "mass_kg")
            total = sum(t for t, _e in g.layers)
            if g.barrier.penetration_m is None:
                E_ref = max(self.E_F_J, 0.0) + 10.0 * kT
                acc, d = 0.0, 0.0
                for (t, _e), V, m in zip(g.layers, hts, ms):
                    if V * Q <= E_ref:
                        raise ValueError("DualGateSchrodingerPoisson: oxide barrier {} eV is not above "
                                         "E_F + 10 kT = {:.4f} eV: no confinement".format(V, E_ref / Q))
                    kap = math.sqrt(2.0 * m * (V * Q - E_ref)) / HBAR
                    if acc + kap * t >= _TAIL_EXPONENT:
                        d += (_TAIL_EXPONENT - acc) / kap
                        acc = _TAIL_EXPONENT
                        break
                    acc += kap * t
                    d += t
                ncell = min(int(math.ceil(d / h - 1e-9)), int(math.floor(total / h + 1e-9)))
            else:
                ncell = int(round(g.barrier.penetration_m / h))
                ncell = min(ncell, int(math.floor(total / h + 1e-9)))
            if ncell > 0:
                pen[side] = ncell
                self._barrier_layers[side] = (g.layers, hts, ms)
        self._pen = pen
        self._general = bool(pen[0] or pen[1] or xc is not None or self._traps[0] is not None
                             or self._traps[1] is not None)
        self._barrier = bool(pen[0] or pen[1])
        self._Ceff = self._C
        self._xc_fn = None
        if xc is not None:
            from dynameta.optics.quantum_film import lda_xc_kernel, lda_xc_potential
            m, ex = self.m_eff_kg, self._xc_eps
            v_ref = float(lda_xc_potential(np.array([self.n_bg_m3]), m, ex, model=xc)[0])
            self._xc_ref_J = v_ref

            def _xc_fn(n, m=m, ex=ex, xc=xc, v_ref=v_ref):        # bound now (no late binding)
                n = np.asarray(n, dtype=np.float64)
                return (lda_xc_potential(n, m, ex, model=xc) - v_ref, lda_xc_kernel(n, m, ex, model=xc))
            self._xc_fn = _xc_fn
        if not self._general:
            self._fp = self._fingerprint()
            return
        # ---- the extended domain (equal to the film grid when no barrier is active) ----
        nb, nt = pen
        nzf = self.nz
        zf = self.z_m
        z_ext = np.concatenate([-h * np.arange(nb, 0, -1), zf, zf[-1] + h * np.arange(1, nt + 1)])
        N = z_ext.size
        self._kb, self._kt = nb, nb + nzf - 1
        self.z_ext_m = z_ext
        Nd = np.zeros(N)
        Nd[nb:nb + nzf] = self.n_bg_m3
        # piecewise-constant material profile over [z_ext[0], z_ext[-1]] (segments ascending)
        edges, eps_v, m_v, U_v = [], [], [], []
        if nb:
            lay, hts, ms = self._barrier_layers[0]
            depth = 0.0
            seg = []
            for (t, e), V, m in zip(lay, hts, ms):
                seg.append((-(depth + t), -depth, e, m, V * Q))
                depth += t
            for a, b, e, m, U in reversed(seg):
                edges.append(a); eps_v.append(e); m_v.append(m); U_v.append(U)
            Nd[nb] = 0.5 * self.n_bg_m3                    # half the interface control volume is film
        edges.append(0.0); eps_v.append(self.eps_static); m_v.append(self.m_eff_kg); U_v.append(0.0)
        if nt:
            lay, hts, ms = self._barrier_layers[1]
            depth = 0.0
            for (t, e), V, m in zip(lay, hts, ms):
                edges.append(self.t_film_m + depth); eps_v.append(e); m_v.append(m); U_v.append(V * Q)
                depth += t
            Nd[self._kt] = 0.5 * self.n_bg_m3
            edges.append(self.t_film_m + depth)
        else:
            edges.append(self.t_film_m)
        edges = np.array(edges)
        edges[0] = min(edges[0], z_ext[0])
        edges[-1] = max(edges[-1], z_ext[-1])
        eps_v, m_v, U_v = np.array(eps_v), np.array(m_v), np.array(U_v)
        inv_eps = 1.0 / eps_v
        self._Nd_ext = Nd
        if self._barrier:
            # cells: 1/eps and m averaged over the cell (series flux), nodes: U averaged over the control volume
            eps_c = np.empty(N - 1)
            m_c = np.empty(N - 1)
            for j in range(N - 1):
                a, b = z_ext[j], z_ext[j + 1]
                if nb <= j < self._kt:                         # film cell: exact, no averaging
                    eps_c[j], m_c[j] = self.eps_static, self.m_eff_kg
                else:
                    eps_c[j] = 1.0 / _piecewise_mean(edges, inv_eps, a, b)
                    m_c[j] = _piecewise_mean(edges, m_v, a, b)
            U_n = np.zeros(N)
            for i in range(N):
                if nb < i < self._kt:
                    continue
                a = z_ext[i] - (0.5 * h if i > 0 else 0.0)
                b = z_ext[i] + (0.5 * h if i < N - 1 else 0.0)
                U_n[i] = _piecewise_mean(edges, U_v, a, b)
            self._eps_cell = eps_c
            self._inv_m_cell = 1.0 / m_c
            self._U_band = U_n
            # matching-plane capacitance of the remaining oxide (inf = the penetration reaches the gate)
            for side in (0, 1):
                if not pen[side]:
                    continue
                lay = self._barrier_layers[side][0]
                d = pen[side] * h
                rem, depth = 0.0, 0.0
                for t, e in lay:
                    part = max(0.0, min(t, depth + t - d))
                    rem += part / (EPS0 * e)
                    depth += t
                end_eps = eps_c[0] if side == 0 else eps_c[-1]
                if rem <= 1e-6 * h / (EPS0 * end_eps):
                    self._Ceff = self._Ceff.copy()
                    self._Ceff[side] = EPS0 * end_eps / h          # residual scale of phi_m = phi_g
                    setattr(self, "_gate_at_plane_{}".format(side), True)
                else:
                    self._Ceff = self._Ceff.copy()
                    self._Ceff[side] = 1.0 / rem
                    setattr(self, "_gate_at_plane_{}".format(side), False)
        else:
            self._eps_cell = None
            self._inv_m_cell = None
            self._U_band = None
        self._sp_ext = (self._sp if N == nzf else
                        SchrodingerPoisson1D(z_ext, self.m_eff_kg, T_K=self.T_K, g_s=self.g_s, g_v=self.g_v))
        self._ns_guess = None
        self._fp = self._fingerprint()

    @property
    def _cache(self) -> "OrderedDict[Tuple[float, float], _State]":
        """The bias cache of the CURRENT options (a separate LRU per option set)."""
        if self._fingerprint() != self._fp:
            self._setup()
        return self._caches.setdefault(repr(self._fp), OrderedDict())

    @property
    def penetration_m(self) -> Tuple[float, float]:
        """(bottom, top) depth of the Schrodinger-Poisson domain into the oxide stacks [m] (0 = hard wall)."""
        _ = self._cache
        return (self._pen[0] * self._h, self._pen[1] * self._h)

    # ---- CarrierSolver protocol ----
    def regions(self) -> List[RegionInfo]:
        L, t = self.lateral_m, self.t_film_m
        return [RegionInfo(name=self.region_name, role="semiconductor", material=self.semi_material,
                           bbox_m=(0.0, L, 0.0, L, 0.0, t), ndim=3)]

    def teardown(self) -> None:
        pass

    def clear_cache(self) -> None:
        """Forget every cached bias point (the next solve starts cold)."""
        self._caches.clear()

    # ---- pieces ----
    def gate_potentials_V(self, V_bottom: float, V_top: float) -> np.ndarray:
        """Electrostatic potentials of the two gate metals, V - V_wf (meaningless for an ungated face)."""
        return np.array([float(V_bottom), float(V_top)]) - self._wf

    def face_fluxes_C_m2(self, phi_V: np.ndarray, n_m3: np.ndarray) -> Tuple[float, float]:
        """(F_b, F_t): eps0 eps_s dphi/dz on the film side of the bottom / top face, along +z, from the
        conservative flux of the solver's Poisson stencil (schrodinger_poisson.sp_face_flux_C_m2)."""
        Fb = sp_face_flux_C_m2(phi_V, n_m3, self._Nd, self._h, self.eps_static, side="left")
        Ft = sp_face_flux_C_m2(phi_V, n_m3, self._Nd, self._h, self.eps_static, side="right")
        return Fb, Ft

    def _end_fluxes(self, phi, n) -> Tuple[float, float]:
        """Conservative fluxes at the two ends of the (extended) domain, along +z."""
        eb = self.eps_static if not self._pen[0] else float(self._eps_cell[0])
        et = self.eps_static if not self._pen[1] else float(self._eps_cell[-1])
        Fb = sp_face_flux_C_m2(phi, n, self._Nd_ext, self._h, eb, side="left")
        Ft = sp_face_flux_C_m2(phi, n, self._Nd_ext, self._h, et, side="right")
        return Fb, Ft

    def _trap(self, side: int, phi_s: float) -> Tuple[float, float]:
        tr = self._traps[side]
        if tr is None:
            return 0.0, 0.0
        return trap_charge_and_derivative(tr, phi_s, self.E_F_J, self.T_K)

    def _hard_trap_terms(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """(R contribution, exact diagonal dR/dx) of the traps on hard-wall faces (outer residual)."""
        r = np.zeros(2)
        d = np.zeros(2)
        for side, sgn in ((0, -1.0), (1, -1.0)):
            if self._traps[side] is None or self._pen[side]:
                continue
            q_it, dq = self._trap(side, float(x[side]))
            r[side] = sgn * q_it
            d[side] = sgn * dq
        return r, d

    def _residual(self, x, phi, n, pg) -> np.ndarray:
        if self._general:
            return self._residual_general(x, phi, n, pg)
        Fb, Ft = self.face_fluxes_C_m2(phi, n)
        Cb, Ct = self._C
        sb, st = self._sigma
        Rb = Cb * (x[0] - pg[0]) - Fb - sb
        Rt = Ft + Ct * (x[1] - pg[1]) - st
        return np.array([Rb, Rt])

    def _residual_general(self, x, phi, n, pg) -> np.ndarray:
        Fb, Ft = self._end_fluxes(phi, n)
        Cb, Ct = self._Ceff
        if self._pen[0]:
            Rb = Cb * (x[0] - pg[0]) - (0.0 if self._gate_at_plane_0 else Fb)
        else:
            Rb = Cb * (x[0] - pg[0]) - Fb - self._sigma[0]
        if self._pen[1]:
            Rt = Ct * (x[1] - pg[1]) + (0.0 if self._gate_at_plane_1 else Ft)
        else:
            Rt = Ft + Ct * (x[1] - pg[1]) - self._sigma[1]
        return np.array([Rb, Rt]) + self._hard_trap_terms(x)[0]

    def _n_states_for(self, phi_max: float) -> int:
        if self.n_states is not None:
            return self.n_states
        kT = KB * self.T_K
        U_min = -Q * (max(phi_max, 0.0) + 0.25)            # 0.25 V margin for in-film overshoot
        cnt = int(np.count_nonzero(self._E_kin + U_min <= self.E_F_J + 30.0 * kT)) + 1
        return int(min(max(cnt, 2), self.nz - 2))

    def _sheet_sources(self):
        """Inner-solve sheet charges on the interface nodes of barrier faces: sigma + Q_it(phi_node)."""
        out = []
        for side, k in ((0, getattr(self, "_kb", 0)), (1, getattr(self, "_kt", 0))):
            if not self._pen[side]:
                continue
            sig = float(self._sigma[side])
            if self._traps[side] is None and sig == 0.0:
                continue

            def fn(phi_node, side=side, sig=sig):
                q_it, dq = self._trap(side, phi_node)
                return sig + q_it, dq
            out.append((k, fn))
        return out or None

    def _inner(self, x, phi_warm, n_warm=None):
        """One self-consistent film solve at face potentials x = (phi_b, phi_t). Returns
        (phi, n, SubbandResult) or None when the inner solve does not converge (after a damped retry)."""
        if self._general:
            return self._inner_general(x, phi_warm, n_warm)
        phi_max = max(float(x[0]), float(x[1]), float(np.max(phi_warm)) if phi_warm is not None else 0.0)
        ns = self._n_states_for(phi_max)
        U0 = None if phi_warm is None else -Q * phi_warm
        for relax in (1.0, 0.5):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                phi, n, res = self._sp.solve_self_consistent(
                    eps_r=self.eps_static, doping_m3=self._Nd, E_F_J=self.E_F_J, U_init_J=U0,
                    phi_left_V=float(x[0]), phi_right_V=float(x[1]), n_states=ns, bound_tol=1e9,
                    max_outer=150, tol_V=self.inner_tol_V, relax=relax,
                    alpha_np_per_eV=self.alpha_np)
            self.stats["inner_solves"] += 1
            self._count += 1
            other = [r for r in rec if not (issubclass(r.category, UnreliableResultWarning)
                                             and str(r.message).startswith(_INNER_NONCONV))]
            if res.converged:
                for r in other:
                    warnings.warn(str(r.message), r.category, stacklevel=4)
                return phi, n, res
        return None

    def _inner_general(self, x, phi_warm, n_warm):
        """The inner solve with the v0.16 options on the (extended) domain. Sub-band completeness is
        checked directly (top solved state >= E_F + 30 kT, else the count is doubled and the solve
        repeated), which is the rigorous form of the hard-wall count when band offsets, the oxide field
        and V_xc make the min-max bound loose."""
        kT = KB * self.T_K
        N = self._sp_ext.z.size
        if self.n_states is not None:
            ns = self.n_states
        else:
            pw = 0.0 if phi_warm is None else float(np.max(phi_warm[self._kb:self._kt + 1]))
            ns = self._n_states_for(max(float(x[0]), float(x[1]), pw) if not self._barrier else pw)
            if self._ns_guess is not None:
                ns = max(ns, self._ns_guess)
        U0 = None if phi_warm is None else -Q * phi_warm
        kw = dict(eps_cell_r=self._eps_cell, inv_mass_cell_per_kg=self._inv_m_cell,
                  band_offset_J=self._U_band, node_sheet_charges=self._sheet_sources(),
                  xc_potential=self._xc_fn,
                  xc_density_init_m3=(n_warm if (self._xc_fn is not None and n_warm is not None) else None))
        relaxes = [1.0, 0.5]
        while relaxes:
            relax = relaxes[0]
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                phi, n, res = self._sp_ext.solve_self_consistent(
                    eps_r=self.eps_static, doping_m3=self._Nd_ext, E_F_J=self.E_F_J, U_init_J=U0,
                    phi_left_V=float(x[0]), phi_right_V=float(x[1]), n_states=ns, bound_tol=1e9,
                    max_outer=150, tol_V=self.inner_tol_V, relax=relax,
                    alpha_np_per_eV=self.alpha_np, **kw)
            self.stats["inner_solves"] += 1
            self._count += 1
            top_ok = res.energies_J.size >= N - 2 or res.energies_J[-1] >= self.E_F_J + 30.0 * kT
            if self.n_states is None and not top_ok:
                ns = min(2 * ns, N - 2)                    # incomplete ladder: more states, same relax
                continue
            other = [r for r in rec if not (issubclass(r.category, UnreliableResultWarning)
                                             and (str(r.message).startswith(_INNER_NONCONV)
                                                  or (self.n_states is None
                                                      and str(r.message).startswith(_INNER_TRUNC))))]
            if res.converged:
                if self.n_states is None:
                    self._ns_guess = ns
                for r in other:
                    warnings.warn(str(r.message), r.category, stacklevel=4)
                return phi, n, res
            relaxes.pop(0)
        return None

    def _evaluate(self, x, phi_warm, pg, n_warm=None):
        out = self._inner(x, phi_warm, n_warm) if self._general else self._inner(x, phi_warm)
        if out is None:
            return None
        phi, n, res = out
        return self._residual(x, phi, n, pg), phi, n, res

    def _fd_jacobian(self, x, R, phi, pg, n=None):
        J = np.empty((2, 2))
        if self._general:
            r0, _d0 = self._hard_trap_terms(x)
        for j in range(2):
            d = 1e-5
            xp = x.copy()
            xp[j] += d
            ev = self._evaluate(xp, phi, pg, n) if self._general else self._evaluate(xp, phi, pg)
            if ev is None:
                return None
            if self._general:
                # film part by forward difference, the hard-wall trap term with its exact derivative
                rp, _dp = self._hard_trap_terms(xp)
                J[:, j] = ((ev[0] - rp) - (R - r0)) / d
            else:
                J[:, j] = (ev[0] - R) / d
        if self._general:
            J[np.diag_indices(2)] += self._hard_trap_terms(x)[1]
        return J

    def _newton(self, V, x0, J0, phi_warm, n_warm=None):
        """Newton on the two face laws at bias V from (x0, J0). Returns a _State or None."""
        pg = self.gate_potentials_V(*V)
        x = np.array(x0, dtype=np.float64)
        gen = self._general
        ev = self._evaluate(x, phi_warm, pg, n_warm) if gen else self._evaluate(x, phi_warm, pg)
        if ev is None:
            return None
        R, phi, n, res = ev
        J = None if J0 is None else np.array(J0, dtype=np.float64)
        fresh = False
        it = 0
        while True:
            if np.max(np.abs(R)) < self.gauss_tol_C_m2:
                break
            if it >= self.max_newton:
                return None
            if J is None:
                J = self._fd_jacobian(x, R, phi, pg, n) if gen else self._fd_jacobian(x, R, phi, pg)
                if J is None:
                    return None
                fresh = True
            try:
                dx = -np.linalg.solve(J, R)
            except np.linalg.LinAlgError:
                dx = None
            ok = False
            if dx is not None and np.all(np.isfinite(dx)):
                big = np.max(np.abs(dx))
                if big > self.max_step_V:
                    dx = dx * (self.max_step_V / big)
                r0 = np.max(np.abs(R))
                lam = 1.0
                for _ in range(8):
                    xt = x + lam * dx
                    evt = self._evaluate(xt, phi, pg, n) if gen else self._evaluate(xt, phi, pg)
                    if evt is not None and np.max(np.abs(evt[0])) < (1.0 - 1e-4 * lam) * r0:
                        ok = True
                        break
                    lam *= 0.5
            it += 1
            if not ok:
                if fresh:
                    return None                       # a fresh Jacobian could not reduce |R|
                J = None                              # refresh the Jacobian and retry
                continue
            Rt, phit, nt, rest = evt
            s = lam * dx
            J = J + np.outer((Rt - R) - J @ s, s) / float(s @ s)   # Broyden rank-1 update
            fresh = False
            x, R, phi, n, res = xt, Rt, phit, nt, rest
        self._iters += it
        if J is None:
            J = self._fd_jacobian(x, R, phi, pg, n) if gen else self._fd_jacobian(x, R, phi, pg)
            if J is None:
                return None
        sol = self._make_solution(V, x, R, phi, n, res, pg)
        return _State(V=np.array(V, dtype=np.float64), x=x, J=J, phi=phi, sol=sol, n=n)

    def _make_solution(self, V, x, R, phi, n, res, pg) -> DualGateSolution:
        if self._general:
            return self._make_solution_general(V, x, R, phi, n, res, pg)
        Fb, Ft = self.face_fluxes_C_m2(phi, n)
        Qgb = self._C[0] * (pg[0] - x[0])
        Qgt = self._C[1] * (pg[1] - x[1])
        Qfilm = Q * trapz(self._Nd - n, self.z_m)
        return DualGateSolution(
            V_bottom=float(V[0]), V_top=float(V[1]), z_m=self.z_m.copy(), phi_V=phi, n_m3=n,
            phi_bottom_V=float(x[0]), phi_top_V=float(x[1]),
            face_flux_bottom_C_m2=float(Fb), face_flux_top_C_m2=float(Ft),
            gate_charge_bottom_C_m2=float(Qgb), gate_charge_top_C_m2=float(Qgt),
            interface_charge_bottom_C_m2=float(self._sigma[0]),
            interface_charge_top_C_m2=float(self._sigma[1]),
            film_charge_C_m2=float(Qfilm),
            gauss_residual_bottom_C_m2=float(R[0]), gauss_residual_top_C_m2=float(R[1]),
            newton_iterations=0, inner_solves=0, continuation_steps=0, solve_time_s=0.0,
            subbands=res)

    def _make_solution_general(self, V, x, R, phi, n, res, pg) -> DualGateSolution:
        kb, kt = self._kb, self._kt
        z = self.z_ext_m
        Fb, Ft = self._end_fluxes(phi, n)
        Qg = np.empty(2)
        for side, F, sgn in ((0, Fb, -1.0), (1, Ft, 1.0)):
            if self._pen[side] and getattr(self, "_gate_at_plane_{}".format(side)):
                Qg[side] = sgn * F                          # the gate sits at the matching plane
            else:
                Qg[side] = self._Ceff[side] * (pg[side] - x[side])
        phi_f, n_f = phi[kb:kt + 1].copy(), n[kb:kt + 1].copy()
        Qfilm = Q * trapz(self._Nd - n_f, self.z_m)
        q_ox = [0.0, 0.0]
        if self._pen[0]:
            q_ox[0] = -Q * trapz(n[:kb + 1], z[:kb + 1])
        if self._pen[1]:
            q_ox[1] = -Q * trapz(n[kt:], z[kt:])
        q_it = [self._trap(0, float(phi[kb]))[0], self._trap(1, float(phi[kt]))[0]]
        U_ext = -Q * phi
        if self._U_band is not None:
            U_ext = U_ext + self._U_band
        vx = getattr(res, "xc_potential_J", None)
        if vx is not None:
            U_ext = U_ext + vx
        tail = None
        if self._barrier:
            nmax = float(np.max(n))
            ends = [float(n[1]) if self._pen[0] else 0.0, float(n[-2]) if self._pen[1] else 0.0]
            tail = max(ends) / nmax if nmax > 0.0 else 0.0
        return DualGateSolution(
            V_bottom=float(V[0]), V_top=float(V[1]), z_m=self.z_m.copy(), phi_V=phi_f, n_m3=n_f,
            phi_bottom_V=float(phi[kb]), phi_top_V=float(phi[kt]),
            face_flux_bottom_C_m2=float(Fb), face_flux_top_C_m2=float(Ft),
            gate_charge_bottom_C_m2=float(Qg[0]), gate_charge_top_C_m2=float(Qg[1]),
            interface_charge_bottom_C_m2=float(self._sigma[0]),
            interface_charge_top_C_m2=float(self._sigma[1]),
            film_charge_C_m2=float(Qfilm),
            gauss_residual_bottom_C_m2=float(R[0]), gauss_residual_top_C_m2=float(R[1]),
            newton_iterations=0, inner_solves=0, continuation_steps=0, solve_time_s=0.0,
            subbands=res,
            trap_charge_bottom_C_m2=float(q_it[0]), trap_charge_top_C_m2=float(q_it[1]),
            oxide_electron_charge_bottom_C_m2=float(q_ox[0]), oxide_electron_charge_top_C_m2=float(q_ox[1]),
            phi_match_bottom_V=float(x[0]) if self._pen[0] else None,
            phi_match_top_V=float(x[1]) if self._pen[1] else None,
            z_ext_m=z.copy(), phi_ext_V=phi, n_ext_m3=n, U_ext_J=U_ext,
            xc_potential_ext_J=None if vx is None else np.asarray(vx).copy(),
            inv_mass_ext_per_kg=None if self._inv_m_cell is None else self._inv_m_cell.copy(),
            tail_density_ratio=tail)

    def _check_validity(self, sol: DualGateSolution) -> None:
        """Finite-barrier sanity at a converged bias: the occupied states must stay evanescent over the
        whole penetration region (the oxide band above E_F + 5 kT) and the tail must have decayed at the
        far wall; otherwise the closed one-Fermi-level model with a wall there is not valid."""
        if not self._barrier:
            return
        kT = KB * self.T_K
        U_ox = self._U_band - Q * sol.phi_ext_V          # electrostatic oxide band edge (no V_xc)
        idx = np.r_[np.arange(0, self._kb), np.arange(self._kt + 1, self.z_ext_m.size)]
        low = float(np.min(U_ox[idx] - self.E_F_J)) / kT
        if low < 5.0:              # (the tail advisory below is implied and not repeated)
            warnings.warn(
                "DualGateSchrodingerPoisson: at V = ({:.4g}, {:.4g}) V the oxide conduction band inside the "
                "penetration region comes within {:.1f} kT of the film's Fermi level (tunnelling regime): "
                "states there are not evanescent and the hard wall at the matching plane confines them "
                "artificially. Reduce OxideBarrier.penetration_m or the bias.".format(
                    sol.V_bottom, sol.V_top, low), UnreliableResultWarning, stacklevel=3)
        elif sol.tail_density_ratio is not None and sol.tail_density_ratio > 1e-6:
            warnings.warn(
                "DualGateSchrodingerPoisson: at V = ({:.4g}, {:.4g}) V the density next to the far wall of "
                "the penetration region is {:.2g} of its maximum; increase OxideBarrier.penetration_m for "
                "a converged tail.".format(sol.V_bottom, sol.V_top, sol.tail_density_ratio),
                AdvisoryWarning, stacklevel=3)

    def _remember(self, st: _State) -> None:
        key = (float(st.V[0]), float(st.V[1]))
        cache = self._cache
        cache[key] = st
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def _nearest(self, V) -> Optional[_State]:
        cache = self._cache
        if not cache:
            return None
        Vt = np.asarray(V, dtype=np.float64)
        return min(cache.values(), key=lambda s: float(np.sum((s.V - Vt) ** 2)))

    def _cold_state(self) -> _State:
        """First solve without any cache: the anchor bias at which both gate metals sit at the film's
        flat-band potential (V = V_wf), started from flat face potentials."""
        V0 = self._wf.copy()
        st = self._newton(V0, np.zeros(2), None, None)
        if st is None:
            raise DualGateConvergenceError(
                "DualGateSchrodingerPoisson: the anchor solve at V = ({:.4g}, {:.4g}) V (gates at the "
                "film's flat-band potential) did not converge; check the grid (nz), the inputs and the "
                "interface charges.".format(V0[0], V0[1]))
        return st

    def _march(self, start: _State, V_target) -> _State:
        """Adaptive continuation from a converged state to V_target (step halving on failure)."""
        Vt = np.asarray(V_target, dtype=np.float64)
        s = start
        lam, dlam = 0.0, 1.0
        halvings = 0
        V0 = start.V.copy()
        while lam < 1.0:
            step = min(dlam, 1.0 - lam)
            lam_n = 1.0 if step >= 1.0 - lam else lam + step
            Vn = Vt if lam_n == 1.0 else V0 + lam_n * (Vt - V0)
            dV = Vn - s.V
            try:
                x0 = s.x + np.linalg.solve(s.J, self._Ceff * dV)    # tangent predictor
            except np.linalg.LinAlgError:
                x0 = s.x.copy()
            if not np.all(np.isfinite(x0)):
                x0 = s.x.copy()
            st = (self._newton(Vn, x0, s.J, s.phi, s.n) if self._general
                  else self._newton(Vn, x0, s.J, s.phi))
            if st is None:
                halvings += 1
                if halvings > self.max_halvings:
                    raise DualGateConvergenceError(
                        "DualGateSchrodingerPoisson: no convergence toward V = ({:.6g}, {:.6g}) V: stuck at "
                        "V = ({:.6g}, {:.6g}) V after {} step halvings (last step {:.3g} of the path). Gauss's "
                        "law at the faces or an inner self-consistent solve failed; refine nz, check "
                        "n_states, or approach the bias in smaller sweep steps.".format(
                            Vt[0], Vt[1], s.V[0], s.V[1], self.max_halvings, step))
                dlam = 0.5 * step
                continue
            self._steps += 1
            s = st
            lam = lam_n
            if lam < 1.0:
                self._remember(s)
                dlam = min(2.0 * step, 1.0)
        return s

    # ---- public solves ----
    def solve_voltages(self, V_bottom: float, V_top: float) -> DualGateSolution:
        """Solve one bias point (gate voltages relative to the film's Fermi level). Cached."""
        key = (float(V_bottom), float(V_top))
        if not all(np.isfinite(key)):
            raise ValueError("DualGateSchrodingerPoisson: gate voltages must be finite, got {}".format(key))
        if self.bottom is None:
            key = (0.0, key[1])
        if self.top is None:
            key = (key[0], 0.0)
        cache = self._cache                       # (re)builds the option state if the options changed
        if key in cache:
            cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return self._copy(cache[key].sol, from_cache=True)
        t0 = time.perf_counter()
        self._count, self._iters, self._steps = 0, 0, 0
        start = self._nearest(key)
        if start is None:
            start = self._cold_state()
            self._remember(start)
        st = start if tuple(start.V) == key else self._march(start, key)
        self._remember(st)
        self.stats["biases_solved"] += 1
        sol = st.sol
        sol.newton_iterations = self._iters
        sol.inner_solves = self._count
        sol.continuation_steps = self._steps
        sol.solve_time_s = time.perf_counter() - t0
        if self._barrier:
            self._check_validity(sol)
        return self._copy(sol)

    @staticmethod
    def _copy(sol: DualGateSolution, **over) -> DualGateSolution:
        d = dict(sol.__dict__)
        for k in ("z_m", "phi_V", "n_m3", "z_ext_m", "phi_ext_V", "n_ext_m3", "U_ext_J",
                  "xc_potential_ext_J", "inv_mass_ext_per_kg"):
            if d.get(k) is not None:
                d[k] = d[k].copy()                    # the cached arrays stay private
        d.update(over)
        return DualGateSolution(**d)

    def _bias_voltages(self, bias) -> Tuple[float, float]:
        vals = []
        for key, gate in zip(self.bias_keys, (self.bottom, self.top)):
            if gate is None:
                vals.append(0.0)
                continue
            if key not in bias.voltages:
                raise ValueError("DualGateSchrodingerPoisson: bias '{}' has no voltage for gate '{}' "
                                 "(voltages: {})".format(getattr(bias, "label", "?"), key,
                                                          sorted(bias.voltages)))
            vals.append(float(bias.voltages[key]))
        return vals[0], vals[1]

    def solve(self, bias) -> CarrierField:
        """CarrierSolver entry point: the laterally uniform through-film profile at this bias.

        With finite barriers the region holds the IN-FILM density (z in [0, t]); the electrons in the
        oxide penetration regions are reported in the extras ('oxide_electron_sheet_bottom_m2' / top,
        'z_ext_m', 'electron_density_ext_m3', 'potential_ext_V'), not in any region."""
        Vb, Vt = self._bias_voltages(bias)
        sol = self.solve_voltages(Vb, Vt)
        z = sol.z_m
        xs = np.linspace(0.0, self.lateral_m, self.n_lateral)
        ys = np.linspace(0.0, self.lateral_m, self.n_lateral)
        nx, ny, nz = xs.size, ys.size, z.size
        n3d = np.broadcast_to(sol.n_m3[None, None, :], (nx, ny, nz)).copy()
        pot3d = np.broadcast_to(sol.phi_V[None, None, :], (nx, ny, nz)).copy()
        X, Y, Z = np.meshgrid(xs, ys, z, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        reg = CarrierRegion(
            name=self.region_name, role="semiconductor", material=self.semi_material,
            nodes_m=nodes, node_fields={ELECTRON_DENSITY: n3d.ravel(), POTENTIAL: pot3d.ravel()},
            grid_axes_m={"x": xs, "y": ys, "z": z},
            grid_fields={ELECTRON_DENSITY: n3d, POTENTIAL: pot3d})
        extras: Dict[str, object] = {
            "quantum": True, "dual_gate": True, "E_F_eV": self.E_F_J / Q,
            "phi_bottom_V": sol.phi_bottom_V, "phi_top_V": sol.phi_top_V,
            "gate_charge_bottom_C_m2": sol.gate_charge_bottom_C_m2,
            "gate_charge_top_C_m2": sol.gate_charge_top_C_m2,
            "interface_charge_bottom_C_m2": sol.interface_charge_bottom_C_m2,
            "interface_charge_top_C_m2": sol.interface_charge_top_C_m2,
            "film_charge_C_m2": sol.film_charge_C_m2, "sheet_excess_m2": sol.sheet_excess_m2,
            "gauss_residual_bottom_C_m2": sol.gauss_residual_bottom_C_m2,
            "gauss_residual_top_C_m2": sol.gauss_residual_top_C_m2,
            "newton_iterations": sol.newton_iterations, "inner_solves": sol.inner_solves,
            "continuation_steps": sol.continuation_steps, "from_cache": sol.from_cache}
        if self._general:
            extras.update({
                "xc": self._xc_model, "barrier_penetration_m": self.penetration_m,
                "trap_charge_bottom_C_m2": sol.trap_charge_bottom_C_m2,
                "trap_charge_top_C_m2": sol.trap_charge_top_C_m2,
                "oxide_electron_sheet_bottom_m2": -sol.oxide_electron_charge_bottom_C_m2 / Q,
                "oxide_electron_sheet_top_m2": -sol.oxide_electron_charge_top_C_m2 / Q,
                "sheet_excess_total_m2": sol.sheet_excess_total_m2,
                "density_reporting": ("region '{}' = in-film density on [0, t]; electrons in the oxide "
                                      "penetration regions only in the *_ext extras".format(self.region_name)),
                "z_ext_m": sol.z_ext_m, "electron_density_ext_m3": sol.n_ext_m3,
                "potential_ext_V": sol.phi_ext_V})
        return CarrierField(
            bias_label=bias.label, voltages=dict(bias.voltages), ndim=3, temperature_K=self.T_K,
            regions={self.region_name: reg}, n_bg_by_region={self.region_name: self.n_bg_m3},
            unit_cell_m=(self.lateral_m, self.lateral_m), extras=extras)
