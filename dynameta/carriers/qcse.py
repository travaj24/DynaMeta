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
chosen by IN-WELL localization (not merely the lowest eigenvalue), so a strong tilt that drops the
downhill barrier below the well floor does not return a spurious field-ionized edge state.

Oracle (validation/qcse_electroabsorption.py): the small-field electron shift matches the analytic
infinite-well second-order Stark coefficient

    dE_1 = - INFINITE_WELL_STARK_BETA * q^2 * m_eff * F^2 * L^4 / hbar^2 ,
    INFINITE_WELL_STARK_BETA = (128 / pi^6) * sum_{n=2,4,...} n^2 / (n^2 - 1)^5  ~= 2.1944e-3 ,

the textbook ground-state polarizability of an infinite square well (2nd-order perturbation theory).
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def __post_init__(self):
        if not (self.well_width_m > 0 and self.nz >= 21):
            raise ValueError("well_width_m must be > 0 and nz >= 21")
        if self.barrier_e_J <= 0 or self.barrier_h_J <= 0:
            raise ValueError("barrier heights must be > 0 (finite confining well)")

    def _grid(self) -> np.ndarray:
        pad = float(self.n_pad) * self.well_width_m
        return np.linspace(-pad, self.well_width_m + pad, int(self.nz))

    def _well(self, z: np.ndarray, barrier_J: float) -> np.ndarray:
        inside = (z >= 0.0) & (z <= self.well_width_m)
        return np.where(inside, 0.0, float(barrier_J))

    @staticmethod
    def _ground_localized(sp: SchrodingerPoisson1D, U_J: np.ndarray,
                           in_well_interior: np.ndarray) -> Tuple[float, np.ndarray]:
        """Lowest-energy state that is actually localized IN the well (in-well probability > 0.5),
        falling back to the most-localized state. Guards against a field-ionized downhill-wall
        state being returned as the 'ground' under a strong tilt."""
        E, psi, _zi = sp.solve_schrodinger(U_J, n_states=8)
        p_in = np.sum((psi ** 2)[in_well_interior, :], axis=0) * sp.h   # in-well probability/state
        cand = np.where(p_in > 0.5)[0]
        k = int(cand[0]) if cand.size else int(np.argmax(p_in))
        return float(E[k]), psi[:, k]

    def solve(self, field_V_per_m: float) -> StarkState:
        """Solve the electron + heavy-hole ground subbands under a perpendicular field F and
        return the (redshifted) interband transition energy + the e-h overlap."""
        F = float(field_V_per_m)
        z = self._grid()
        zi = z[1:-1]
        in_well = (zi >= 0.0) & (zi <= self.well_width_m)
        # electron PE: well + (+q F z); hole envelope: well + (-q F z) -> opposite-wall displacement
        U_e = self._well(z, self.barrier_e_J) + Q * F * z
        U_h = self._well(z, self.barrier_h_J) - Q * F * z
        spe = SchrodingerPoisson1D(z, self.m_e_kg)
        sph = SchrodingerPoisson1D(z, self.m_h_kg)
        E_e1_raw, psi_e = self._ground_localized(spe, U_e, in_well)
        E_hh1_raw, psi_h = self._ground_localized(sph, U_h, in_well)
        # Reference each confinement energy to its band floor at the WELL CENTRE (z = L/2): the raw
        # eigenvalue carries a LINEAR tilt offset (+qF*L/2 for U_e = well + qFz, -qF*L/2 for U_h =
        # well - qFz). The linear parts cancel in the SUM E_e1 + E_hh1, but subtracting them
        # per-carrier leaves the pure (quadratic) Stark shift, so E_e1 is the true confinement
        # energy -- validated against the analytic infinite-well 2nd-order coefficient -- and E_T(F)
        # is still the correctly redshifted edge.
        zc = 0.5 * self.well_width_m
        E_e1 = E_e1_raw - Q * F * zc
        E_hh1 = E_hh1_raw + Q * F * zc
        h = float(z[1] - z[0])
        overlap = float(abs(np.sum(psi_e * psi_h) * h) ** 2)           # |<psi_e|psi_hh>|^2
        E_T = float(self.E_g_J + E_e1 + E_hh1 - self.exciton_binding_J)
        return StarkState(field_V_per_m=F, E_e1_J=E_e1, E_hh1_J=E_hh1, E_transition_J=E_T,
                          overlap=overlap, z_m=zi, psi_e=psi_e, psi_h=psi_h)

    def transition_energy_J(self, field_V_per_m: float) -> float:
        """Interband transition energy E_T(F) (J) -- the QCSE-redshifted absorption edge."""
        return self.solve(field_V_per_m).E_transition_J
