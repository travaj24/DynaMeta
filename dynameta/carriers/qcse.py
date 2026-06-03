"""
QCSE quantum-well Stark driver (roadmap Phase 3a). Builds a finite square quantum well -- a
conduction-band electron well + a valence-band heavy-hole well -- tilts both with a perpendicular
DC field F, and solves the ground-subband energies with the existing BenDaniel-Duke Schrodinger
kernel (SchrodingerPoisson1D.solve_schrodinger). The field REDSHIFTS the interband transition

    E_T(F) = E_g + E_e1(F) + E_hh1(F) - E_exciton

(the quantum-confined Stark effect: both confinement energies fall ~quadratically in F) and pushes
the electron and hole envelopes to OPPOSITE walls (the e-h overlap, hence the oscillator strength,
drops). The driver PRODUCES E_T(F) + overlap(F); an ElectroAbsorptionModel (core/effects) turns
those into a field-dependent complex eps. Pure numpy/scipy, SI.

Convention: z in metres, energies in Joules; the field tilt is +q F z for the electron PE and
-q F z for the hole envelope (opposite signs -> opposite-wall displacement). The ground state is
chosen by IN-WELL localization (the lowest-index state whose in-well probability exceeds 0.5, else
the most-localized of the solved set) rather than merely the lowest eigenvalue. VALID-FIELD CAVEAT:
at strong tilt the downhill barrier drops below the well floor and the state FIELD-IONIZES; even the
most-localized state then delocalizes and its energy becomes dependent on the grid padding (n_pad)
-- a Dirichlet-wall artifact, not a true bound state. The solver emits a RuntimeWarning when the
picked state's in-well probability falls below 0.5, flagging the result as an unreliable
quasi-bound resonance; keep the field below that onset for trustworthy E_T(F)/overlap.

Oracle (validation/qcse_electroabsorption.py): the small-field electron shift matches the analytic
infinite-well second-order Stark coefficient

    dE_1 = - INFINITE_WELL_STARK_BETA * q^2 * m_eff * F^2 * L^4 / hbar^2 ,
    INFINITE_WELL_STARK_BETA = (128 / pi^6) * sum_{n=2,4,...} n^2 / (n^2 - 1)^5  ~= 2.1944e-3 ,

the textbook ground-state polarizability of an infinite square well (2nd-order perturbation theory).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from dynameta.constants import HBAR, Q_E as Q
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D

# Analytic ground-state Stark coefficient of an infinite square well (2nd-order PT):
# dE_1 = -beta * q^2 m* F^2 L^4 / hbar^2, beta = (128/pi^6) sum_{n even} n^2/(n^2-1)^5.
_n_even = np.arange(2, 200, 2, dtype=np.float64)
INFINITE_WELL_STARK_BETA = float((128.0 / np.pi ** 6) * np.sum(_n_even ** 2 / (_n_even ** 2 - 1.0) ** 5))


@dataclass
class StarkState:
    """Result of a QuantumWell.solve(F): ground-subband energies + interband transition + overlap."""
    field_V_per_m: float
    E_e1_J: float          # electron ground confinement energy (above E_c), J
    E_hh1_J: float         # heavy-hole ground confinement energy (below E_v, into the gap), J
    E_transition_J: float  # E_g + E_e1 + E_hh1 - E_exciton, J
    overlap: float         # |<psi_e|psi_hh>|^2 in [0,1] (oscillator-strength factor)
    z_m: np.ndarray        # interior z nodes
    psi_e: np.ndarray      # electron ground envelope (interior, normalized)
    psi_h: np.ndarray      # hole ground envelope (interior, normalized)
    ionized: bool = False  # True if a carrier field-ionized (in-well prob < 0.5): E_T/overlap then
                           # become a Dirichlet-box artifact (n_pad-dependent) -- a warning was raised
    p_in_min: float = 1.0  # the smaller of the two carriers' in-well probabilities (localization)


@dataclass
class QuantumWell:
    """A single finite square quantum well for QCSE electro-absorption.

    well_width_m   : well thickness L (m)
    barrier_e_J    : conduction-band offset (electron barrier height) (J)
    barrier_h_J    : valence-band offset (heavy-hole barrier height) (J)
    m_e_kg, m_h_kg : electron / heavy-hole effective masses in the well (kg)
    E_g_J          : well-material bandgap (J)
    exciton_binding_J : exciton binding energy subtracted from the edge (J; 0 to ignore)
    n_pad          : barrier padding each side, in well widths (the bound state must decay before
                     the Dirichlet grid ends; too large + strong tilt drops the downhill barrier)
    nz             : number of grid nodes
    n_solve        : number of lowest eigenstates scanned for the in-well ground (must exceed the
                     count of field-ionized triangular-corner states that accumulate below the
                     in-well level at strong tilt -- the HEAVY-HOLE channel needs the most; 40 covers
                     well past the onset of ionization, where results are flagged unreliable anyway)
    """
    well_width_m: float
    barrier_e_J: float
    barrier_h_J: float
    m_e_kg: float
    m_h_kg: float
    E_g_J: float
    exciton_binding_J: float = 0.0
    n_pad: float = 2.5
    nz: int = 1501
    n_solve: int = 40
    _solve_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if not (self.well_width_m > 0 and self.nz >= 21):
            raise ValueError("well_width_m must be > 0 and nz >= 21")
        if self.barrier_e_J <= 0 or self.barrier_h_J <= 0:
            raise ValueError("barrier heights must be > 0 (finite confining well)")
        if self.m_e_kg <= 0 or self.m_h_kg <= 0:
            raise ValueError("m_e_kg and m_h_kg must be > 0 (effective masses)")
        if self.E_g_J <= 0:
            raise ValueError("E_g_J must be > 0")

    def _grid(self) -> np.ndarray:
        pad = float(self.n_pad) * self.well_width_m
        return np.linspace(-pad, self.well_width_m + pad, int(self.nz))

    def _well(self, z: np.ndarray, barrier_J: float) -> np.ndarray:
        inside = (z >= 0.0) & (z <= self.well_width_m)
        return np.where(inside, 0.0, float(barrier_J))

    def _in_well_mask(self, zi: np.ndarray) -> np.ndarray:
        """Interior-node mask of the conducting well region(s) (single well: [0, L]). Overridden by
        MultiQuantumWell to span all wells of the stack."""
        return (zi >= 0.0) & (zi <= self.well_width_m)

    def _well_centre(self) -> float:
        """z of the well-region centre -- the reference for the linear-tilt removal in solve()
        (single well: L/2). Overridden by MultiQuantumWell to the stack centre."""
        return 0.5 * self.well_width_m

    def _ground_localized(self, sp: SchrodingerPoisson1D, U_J: np.ndarray,
                           in_well_interior: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """Lowest-INDEX state that is actually localized IN the well (in-well probability > 0.5),
        scanning the n_solve lowest eigenstates; falls back to the most-localized of them. Returns
        (E, psi, p_in) where p_in is the picked state's in-well probability. Because field-ionized
        triangular-corner states accumulate BELOW the in-well level as the tilt grows, the in-well
        ground can sit well above index 0; n_solve must exceed that count (the heavy hole needs the
        most). A low returned p_in flags the field-ionized regime (handled by the caller)."""
        n = min(int(self.n_solve), in_well_interior.size)
        E, psi, _zi = sp.solve_schrodinger(U_J, n_states=n)
        p_in = np.sum((psi ** 2)[in_well_interior, :], axis=0) * sp.h   # in-well probability/state
        cand = np.where(p_in > 0.5)[0]
        k = int(cand[0]) if cand.size else int(np.argmax(p_in))
        return float(E[k]), psi[:, k], float(p_in[k])

    _IONIZE_TOL = 0.5      # in-well probability below this => field-ionized / box-corner artifact

    def solve(self, field_V_per_m: float) -> StarkState:
        """Solve the electron + heavy-hole ground subbands under a perpendicular field F and
        return the (redshifted) interband transition energy + the e-h overlap. Results are CACHED
        by field value: an EAM sweep recomputes the F=0 baseline and repeated fields for free
        (bit-identical -- the same StarkState object is returned, and callers only read it)."""
        F = float(field_V_per_m)
        cached = self._solve_cache.get(F)
        if cached is not None:
            return cached
        z = self._grid()
        zi = z[1:-1]
        in_well = self._in_well_mask(zi)
        # electron PE: well + (+q F z); hole envelope: well + (-q F z) -> opposite-wall displacement
        U_e = self._well(z, self.barrier_e_J) + Q * F * z
        U_h = self._well(z, self.barrier_h_J) - Q * F * z
        spe = SchrodingerPoisson1D(z, self.m_e_kg)
        sph = SchrodingerPoisson1D(z, self.m_h_kg)
        E_e1_raw, psi_e, p_e = self._ground_localized(spe, U_e, in_well)
        E_hh1_raw, psi_h, p_h = self._ground_localized(sph, U_h, in_well)
        p_in_min = min(p_e, p_h)
        ionized = p_in_min < self._IONIZE_TOL
        if ionized:
            warnings.warn(
                "QuantumWell.solve(F={:.3g} V/m): a carrier FIELD-IONIZED (min in-well "
                "probability {:.2f} < {:.1f}) -- the downhill barrier has dropped below the well "
                "floor, so E_T(F)/overlap are a Dirichlet-box artifact (n_pad-dependent), NOT a "
                "true bound state. Reduce the field below the ionization onset for a trustworthy "
                "result.".format(F, p_in_min, self._IONIZE_TOL), RuntimeWarning, stacklevel=2)
        # Reference each confinement energy to its band floor at the WELL CENTRE (z = L/2): the raw
        # eigenvalue carries a LINEAR tilt offset (+qF*L/2 for U_e = well + qFz, -qF*L/2 for U_h =
        # well - qFz). The linear parts cancel in the SUM E_e1 + E_hh1, but subtracting them
        # per-carrier leaves the pure (quadratic) Stark shift, so E_e1 is the true confinement
        # energy -- validated against the analytic infinite-well 2nd-order coefficient -- and E_T(F)
        # is still the correctly redshifted edge.
        zc = self._well_centre()
        E_e1 = E_e1_raw - Q * F * zc
        E_hh1 = E_hh1_raw + Q * F * zc
        h = float(z[1] - z[0])
        overlap = float(abs(np.sum(psi_e * psi_h) * h) ** 2)           # |<psi_e|psi_hh>|^2
        E_T = float(self.E_g_J + E_e1 + E_hh1 - self.exciton_binding_J)
        state = StarkState(field_V_per_m=F, E_e1_J=E_e1, E_hh1_J=E_hh1, E_transition_J=E_T,
                           overlap=overlap, z_m=zi, psi_e=psi_e, psi_h=psi_h,
                           ionized=ionized, p_in_min=p_in_min)
        self._solve_cache[F] = state
        return state

    def transition_energy_J(self, field_V_per_m: float) -> float:
        """Interband transition energy E_T(F) (J) -- the QCSE-redshifted absorption edge."""
        return self.solve(field_V_per_m).E_transition_J


@dataclass
class MultiQuantumWell(QuantumWell):
    """A MULTI-quantum-well stack: `n_wells` identical wells of width well_width_m separated by
    barriers of width barrier_width_m (barrier heights = the single-well offsets). The ground
    MINIBAND state is solved with the SAME BenDaniel-Duke kernel + in-well-localization picker as
    QuantumWell, on the full N-well potential -- so a thick barrier gives an UNCOUPLED stack (ground
    ~ the single-well E_1, an N-fold near-degenerate manifold) and a thin barrier gives a COUPLED
    miniband (the ground subband drops as the wells hybridize). Reduces EXACTLY to QuantumWell when
    n_wells == 1 (barrier_width_m irrelevant). The interband edge E_T(F) + e-h overlap feed the same
    ElectroAbsorptionModel; the MQW carries n_wells x the absorption (more wells in the optical path).

    A wider stack needs a denser grid -- raise nz for n_wells > ~2 (the default 1501 over a longer
    extent thins the per-well resolution)."""
    n_wells: int = 1
    barrier_width_m: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if int(self.n_wells) < 1:
            raise ValueError("n_wells must be >= 1")
        if int(self.n_wells) > 1 and not (self.barrier_width_m > 0.0):
            raise ValueError("barrier_width_m must be > 0 for n_wells > 1")

    def _period(self) -> float:
        return self.well_width_m + self.barrier_width_m

    def _stack_len(self) -> float:
        return int(self.n_wells) * self.well_width_m + (int(self.n_wells) - 1) * self.barrier_width_m

    def _well_starts(self):
        p = self._period()
        return [i * p for i in range(int(self.n_wells))]          # left edge of each well

    def _grid(self) -> np.ndarray:
        pad = float(self.n_pad) * self.well_width_m
        return np.linspace(-pad, self._stack_len() + pad, int(self.nz))

    def _well(self, z: np.ndarray, barrier_J: float) -> np.ndarray:
        inside = np.zeros(np.shape(z), dtype=bool)
        for s in self._well_starts():
            inside = inside | ((z >= s) & (z <= s + self.well_width_m))
        return np.where(inside, 0.0, float(barrier_J))

    def _in_well_mask(self, zi: np.ndarray) -> np.ndarray:
        m = np.zeros(np.shape(zi), dtype=bool)
        for s in self._well_starts():
            m = m | ((zi >= s) & (zi <= s + self.well_width_m))
        return m

    def _well_centre(self) -> float:
        return 0.5 * self._stack_len()
