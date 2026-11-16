"""Self-consistent (RPA / adiabatic-LDA) optical response of a quantum-confined degenerate electron film.

A degenerate electron film a few nanometres thick is one quantum system along z: its states are
sub-bands psi_i(z) with free in-plane motion. Its out-of-plane response is NONLOCAL in z (the induced
polarization at z depends on the field everywhere in the film) and Landau-damped (the particle-hole
transitions of the confined gas). The hydrodynamic model (optics.nonlocal_tmm) is the long-wavelength
expansion of that response and fails where the longitudinal wavevector is comparable to omega / v_F
(hydro_validity's u ~ 1); the local effective eps_zz (core.effects.IntersubbandEffect) replaces the
depolarization kernel by the uniform-field one. This module builds the full linear response of the
sub-band system in the self-consistent field (RPA = time-dependent Hartree, optionally + adiabatic LDA
exchange-correlation) and couples it EXACTLY (within the model) to Maxwell's equations for planar
p-polarized light.

Formulation (exp(-i omega t); z across the film, x in-plane; in-plane wavevector q)
---------------------------------------------------------------------------------
Ground state: states psi_i, energies E_i, sheet densities n_i (spin and valley included) of the
SchrodingerPoisson1D slab model (hard walls on both faces; parabolic z dispersion with mass m; in-plane
dispersion parabolic with mass m or Kane, E (1 + alpha E) = hbar^2 k^2 / 2m, the solver's own option),
on a uniform grid z_0..z_{N-1} (states on the N-2 interior nodes, Riemann-normalized).

Electronic response at q -> 0 (q << k_F and q v_F << omega; the dropped terms are relative
O((q/k_F)^2, (q v_F/omega)^2) -- 4e-6 and 7e-6 at 60 deg, 1200 nm, 3e26 m^-3):
  * out-of-plane: the intersubband transitions. With the transition density zeta_p = psi_i psi_j and
    its running integral Xi_p(z) = int_0^z zeta_p (Xi_p vanishes at both faces: orthogonality), the
    induced polarization of the independent electrons driven by an effective field E is
        P(z) = eps0 sum_p g_p(omega) Xi_p(z) int Xi_p E dz',
        g_p = s_p / (w_p^2 - Omega^2),  s_p = e^2 (n_i - n_j) 2 w_p / (hbar eps0),  w_p = (E_j - E_i)/hbar,
    i.e. the nonlocal conductivity sigma_zz(z, z') = -i omega eps0 sum_p g_p Xi_p(z) Xi_p(z')
    (sigma_zz_kernel). Pairs p = (i < j) with i occupied, j over the COMPLETE discrete basis.
  * in-plane: the intraband Drude current J_x(z) = (i / (omega + i gamma)) sum_i D_i psi_i(z)^2 E_x(z)
    with the Drude weight D_i = e^2 n_i / m (parabolic) or the Fermi-surface average of v_x^2 for the
    Kane band, D_i = (e^2 g_s g_v / 2 pi hbar^2) int (-df/dE) E (1 + aE) / (1 + 2aE) dE. Local in z: the
    exact q -> 0 in-plane current of subband i is psi_i^2 times its response to <E_x>_i, which differs
    from this only through the variation of E_x across the film (second order in qL; the sheet current
    integral int J_x dz is identical).
  * damping: Omega^2 = omega (omega + i gamma), i.e. damping on the velocity of each transition
    oscillator (the relaxation drives the electrons toward the equilibrium IN the instantaneous field,
    not toward the unperturbed one). This keeps the exact static limit, the f-sum rule for every gamma
    and particle number (the transition densities integrate to 0) -- the properties Mermin's
    number-conserving correction restores in the Lindhard function (Mermin 1970). The in-plane Mermin
    correction is O((q v_F / omega)^2) and is dropped. A single gamma is used for every transition.

Self-consistent field. Maxwell's equations carry the Hartree field exactly: inside the film
D_z = eps0 eps_inf E_z + P with D_z = -(q/omega) H_y from Ampere's law, so the Coulomb (depolarization)
interaction, its retardation and the dielectric environment (the claddings enter through the field
boundary conditions) are all in the electromagnetic solution, not in a separately chosen kernel. At
q -> 0 (D_z uniform across the film) the Hartree kernel reduces to M_pq = int Xi_p Xi_q dz / eps_inf,
independent of the claddings (the depolarization integrals of Ando, Fowler and Stern 1982; Allen, Tsui
and Vinter 1976). Exchange-correlation in the adiabatic LDA adds V_xc = f_xc(n0(z)) dn(z), f_xc = d v_xc /
dn of the exchange + Perdew-Zunger (1981) correlation energy of the electron gas in a background of
permittivity eps_xc (default eps_inf), in effective atomic units: X_pq = int zeta_p f_xc zeta_q dz.
For a drive that is uniform in D_z the response then follows from the symmetric eigenproblem

    B = diag(w_p^2) + S^1/2 K S^1/2,   K = M + (eps0/e^2) X,   B = U Lambda U^T,

whose eigenvalues Lambda_m are the squared frequencies of the multisubband plasmons (the coupled
intersubband plasmons of the RPA) and whose oscillator strengths F_m = (U^T S^1/2 z)_m^2 / eps_inf sum
EXACTLY to the bare ones: sum_m F_m = (e^2/(eps0 eps_inf m)) sum_{i<j} (n_i - n_j) f_ij (f-sum rule of
the full response; equal to (e^2/(eps0 eps_inf m)) sum_i n_i kappa_i for the complete discrete set,
kappa_i = h sum_k psi_i,k psi_i,k+1 -> 1 as h -> 0). Replacing M by its rank-one uniform-field part
z z^T / (L eps_inf) and dropping X gives back the local effective eps_zz of IntersubbandEffect
exactly; M - z z^T/(L eps_inf) is positive semidefinite (a covariance), so without xc the nonlocal
response shifts every collective mode UP from the local prediction (Weyl's inequality) (single transition: depolarization integral
S = int Xi^2 vs z_12^2 / L, Cauchy-Schwarz).

Two routes to the same response (cross-checked in the tests to ~1e-10):
  * "modes": the eigenproblem above (cost (number of pairs)^3 once; then any frequency is cheap; gives
    modes and the f-sum decomposition) -- for thin films;
  * "real_space": the independent-particle density response chi0(z, z') from the Sternheimer equations
    (E_i +- hbar Omega - H) dpsi = V psi_i solved for the occupied states only (no sum over empty
    states; tridiagonal solves on the grid), then the self-consistent equations solved on the grid
    (cost (occupied states) x N^2 per frequency) -- for any thickness.

Outputs
-------
(a) sigma_zz_kernel (the independent-particle nonlocal conductivity on the half-node grid),
    response_kernel (the self-consistent map from D_z(z) to P(z)), induced_charge (rho(z) under a
    uniform D_z).
(b) transfer_matrix: the EXACT 2x2 (E_x, H_y/(omega eps0)) transfer matrix of the film for planar
    p-polarization: Maxwell's equations with the nonlocal P (through response_kernel) and the local
    in-plane current, discretized on the film grid by a staggered (Yee-type) scheme whose discrete
    Poynting theorem holds exactly; same basis and orientation as optics.nonlocal_tmm (maps the field
    at the superstrate-side face to the substrate-side face). stack_rt composes it with
    nonlocal_tmm layers; its r is in the library's p-hat convention r = r_H = -E_x(refl)/E_x(inc)
    (as fem_2d_tm, the bridges and quantum_film_cell), the negative of nonlocal_tmm.stack_rt's
    tangential-E r (film_stack_rt).
    Thin-film boundary conditions: to FIRST order in the film thickness (relative corrections of order
    (k0 L)^2 |eps_xx / eps_zz,eq| sin^2 -- measured by tests/test_quantum_film.py and
    validation/quantum_film_rpa.py), the film is equivalent to the
    uniform anisotropic layer of the same thickness L:
        eps_xx,eq = eps_inf + alpha_par / L,     1 / eps_zz,eq = (1 - alpha_perp / L) / eps_inf,
    with the two sheet response lengths (dimension of length, like Feibelman's d-parameters):
        alpha_par = i sigma_s / (omega eps0)          (sigma_s the in-plane sheet conductivity),
        alpha_perp = p_s / D_z                         (p_s = int P dz the induced dipole per area under a
                                                        uniform D_z; the response to the EXTERNAL-type
                                                        field, hence Im alpha_perp >= 0 = passivity).
    Equivalently, replacing a host slab (eps_h, thickness L) by a zero-thickness sheet (generalized sheet
    transition conditions; jumps from the superstrate-side to the substrate-side face, D_z = -(q/omega)
    H_y): [H_y] = i omega eps0 (alpha_par + (eps_inf - eps_h) L) E_x and
    [E_x] = i (q / eps0) ((L - alpha_perp) / eps_inf - L / eps_h) D_z.
(c) effective_layer (eps_xx,eq, eps_zz,eq, L) for laterally structured solvers (RCWA / PMM / FEM): a
    uniaxial thin layer. Valid when every lateral Fourier component that carries field obeys
    q_x L << 1, q_x << k_F and q_x v_F << omega (the film must look laterally uniform over its own
    thickness and over the Fermi wavelength), and the lateral structure leaves the film's ground state
    laterally uniform (else use per-column films). validity() reports these numbers.

References: T. Ando, A. B. Fowler, F. Stern, Rev. Mod. Phys. 54, 437 (1982), secs. II.D and IV.B (sub-band
response, depolarization and exciton-like shifts); S. J. Allen, D. C. Tsui, B. Vinter, Solid State
Commun. 20, 425 (1976); M. Zaluzny and C. Nalewajko, Phys. Rev. B 59, 13043 (1999) (intersubband
absorption with depolarization in a layered-medium transfer-matrix setting); F. Alpeggiani and L. C.
Andreani, Phys. Rev. B 90, 115311 (2014) (nonlocal electrodynamics of multisubband plasmons);
P. J. Feibelman, Prog. Surf. Sci. 12, 287 (1982); T. Christensen et al., Phys. Rev. Lett. 118, 157402
(2017); P. A. D. Goncalves et al., Nat. Commun. 11, 366 (2020) (d-parameters, mesoscopic boundary
conditions); N. D. Mermin, Phys. Rev. B 1, 2362 (1970); J. P. Perdew and A. Zunger, Phys. Rev. B 23,
5048 (1981); S. Baroni et al., Rev. Mod. Phys. 73, 515 (2001) (Sternheimer linear response).

Not modelled: in-plane wavevector dispersion of the electronic response (q -> 0, above), a z-dependent
mass or band offsets inside the film, nonparabolicity along z (the solver's z dispersion is parabolic;
its Kane option is in-plane only, and the same separable model is used here, so intersubband energies
do not depend on k), frequency-dependent or transition-dependent damping, local-field effects of the
lattice, and electron spill-out through the walls (hard walls, as in the Schrodinger-Poisson solve).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, HBAR, KB, Q_E, T_REF
from dynameta.warning_types import UnreliableResultWarning

__all__ = ["QuantumFilm", "QuantumFilmResponse", "QuantumFilmValidity", "lda_xc_kernel", "lda_xc_potential",
           "film_transfer_matrix", "uniaxial_layer_matrix", "film_stack_rt", "DEFAULT_OPTICAL_PENETRATION_M"]

_XC_MODELS = ("none", "x", "lda")
_ROUTES = ("auto", "modes", "real_space")

#: Default depth [m] to which QuantumFilm.from_dual_gate lets the response element reach into each oxide
#: (capped by the solver's own penetration); see from_dual_gate.
DEFAULT_OPTICAL_PENETRATION_M = 2e-9
#: Solved density (relative to its peak) above which cutting the element there is flagged.
TAIL_WARN = 1e-6


# ------------------------------------------------------------------------------------------------
# Ground state
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class QuantumFilm:
    """Ground state of a hard-wall film on a uniform grid, with the COMPLETE set of discrete sub-bands.

    z_m: full grid (both hard-wall face nodes), ascending, uniform; U_J: electron potential energy (band
    edge) on z_m; fermi_level_J; m_eff_kg (z mass, and in-plane band-edge mass); T_K; alpha_np_per_eV
    (in-plane Kane nonparabolicity, as SchrodingerPoisson1D.density); g_s, g_v. The states are computed
    by SchrodingerPoisson1D.density (all n_states, all kept): energies_J (n,), psi (N-2, n) on the
    interior nodes with sum psi^2 h = 1, sheet_density_m2 (n,), density_m3 on z_m (0 at the faces).
    Build with from_potential, flat_band or from_dual_gate. inv_mass_cell_per_kg (optional, N-1): the
    per-cell inverse z mass of a heterostructure domain (SchrodingerPoisson1D.solve_schrodinger), e.g. a
    film whose wavefunctions penetrate finite oxide barriers (from_dual_gate); the in-plane mass stays
    m_eff_kg.

    nominal_m (optional): (z_lo, z_hi) of the NOMINAL film (the design layer) inside z_m, when the grid
    extends beyond it into the neighbouring layers (from_dual_gate with finite oxide barriers); None = the
    whole grid is the film. extension_m = (z_lo - z_m[0], z_m[-1] - z_hi) is then how far the response
    element reaches into the layer below (z_min side) and above (z_max side).

    How the optics treats an extended film (the penetration regions used to be counted twice, once in
    the element and once in the designed oxides). Every optical output of
    QuantumFilmResponse describes the NOMINAL film, so every route takes the stack exactly as designed and
    the total optical thickness is the designed one by construction:
      * transfer_matrix maps the nominal film's superstrate face to its substrate face: the element's
        penetration regions are folded back by propagating through a pure eps_inf layer of minus the
        extension on each side. That removes exactly the background eps_inf the element puts where the
        oxide is (the designed oxide keeps its own permittivity) and keeps the response of the electrons
        that sit in the oxide;
      * effective_layer is the equivalent layer of the NOMINAL thickness carrying the element's sheet
        responses (first order in the thickness, like the layer itself).
    Measured on the generic gated film (2 nm + 6 nm oxide per face, 3.0 eV / 0.4 m_e barrier, 1200 nm,
    p 60 deg; tests/test_quantum_film_barrier_optics.py): r does not depend on the element's depth into
    the oxide (<= 2.7e-8 between 2 nm and the solver's whole 3.85 nm, against a bias modulation of 1.6e-3),
    whereas shortening the designed oxides instead would leave eps_inf in place of the oxide permittivity
    over the extension depth: an error of first order in (eps_inf - eps_oxide) x extension that does not
    depend on the bias."""
    z_m: np.ndarray
    U_J: np.ndarray
    fermi_level_J: float
    m_eff_kg: float
    T_K: float = T_REF
    alpha_np_per_eV: float = 0.0
    g_s: int = 2
    g_v: int = 1
    inv_mass_cell_per_kg: Optional[np.ndarray] = None
    nominal_m: Optional[Tuple[float, float]] = None
    energies_J: np.ndarray = field(default=None, init=False, repr=False)
    psi: np.ndarray = field(default=None, init=False, repr=False)
    sheet_density_m2: np.ndarray = field(default=None, init=False, repr=False)
    density_m3: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
        z = np.asarray(self.z_m, dtype=np.float64)
        U = np.asarray(self.U_J, dtype=np.float64)
        if z.ndim != 1 or U.shape != z.shape or z.size < 7:
            raise ValueError("QuantumFilm: z_m and U_J must be 1-D arrays of the same length >= 7")
        object.__setattr__(self, "z_m", z)
        object.__setattr__(self, "U_J", U)
        if self.nominal_m is not None:
            lo, hi = (float(v) for v in self.nominal_m)
            tol = 1e-9 * float(z[-1] - z[0])
            if not (z[0] - tol <= lo < hi <= z[-1] + tol):
                raise ValueError("QuantumFilm: nominal_m = ({!r}, {!r}) must be an interval inside the grid "
                                 "[{!r}, {!r}]".format(lo, hi, float(z[0]), float(z[-1])))
            object.__setattr__(self, "nominal_m", (max(lo, float(z[0])), min(hi, float(z[-1]))))
        sp = SchrodingerPoisson1D(z, float(self.m_eff_kg), T_K=float(self.T_K), g_s=int(self.g_s),
                                  g_v=int(self.g_v))
        mkw = {}
        if self.inv_mass_cell_per_kg is not None:
            mkw["inv_mass_cell_per_kg"] = np.asarray(self.inv_mass_cell_per_kg, dtype=np.float64)
            object.__setattr__(self, "inv_mass_cell_per_kg", mkw["inv_mass_cell_per_kg"])
        res = sp.density(U, float(self.fermi_level_J), n_states=None, bound_tol=np.inf,
                         alpha_np_per_eV=float(self.alpha_np_per_eV), **mkw)
        n_full = np.zeros(z.size)
        n_full[1:-1] = res.density_m3
        object.__setattr__(self, "energies_J", np.asarray(res.energies_J))
        object.__setattr__(self, "psi", np.asarray(res.psi))
        object.__setattr__(self, "sheet_density_m2", np.asarray(res.sheet_density_m2))
        object.__setattr__(self, "density_m3", n_full)

    @property
    def h(self) -> float:
        return float(self.z_m[1] - self.z_m[0])

    @property
    def thickness_m(self) -> float:
        """Thickness of the response element (the whole grid, extension included)."""
        return float(self.z_m[-1] - self.z_m[0])

    @property
    def nominal_thickness_m(self) -> float:
        """Thickness of the nominal film (the design layer); equals thickness_m without an extension."""
        if self.nominal_m is None:
            return self.thickness_m
        return float(self.nominal_m[1] - self.nominal_m[0])

    @property
    def extension_m(self) -> Tuple[float, float]:
        """(below, above): how far the response element reaches beyond the nominal film on its z_min and
        z_max sides [m] ((0, 0) without an extension)."""
        if self.nominal_m is None:
            return (0.0, 0.0)
        return (max(float(self.nominal_m[0] - self.z_m[0]), 0.0),
                max(float(self.z_m[-1] - self.nominal_m[1]), 0.0))

    @property
    def is_extended(self) -> bool:
        b, a = self.extension_m
        return b > 0.0 or a > 0.0

    @property
    def sheet_density_total_m2(self) -> float:
        return float(np.sum(self.sheet_density_m2))

    @classmethod
    def from_potential(cls, z_m, phi_V, E_F_J: float, m_eff_kg: float, *, T_K: float = T_REF,
                       alpha_np_per_eV: float = 0.0, g_s: int = 2, g_v: int = 1) -> "QuantumFilm":
        """From an electrostatic potential phi(z) [V] on the full grid (band edge U = -q phi) and the
        Fermi level [J], e.g. a converged Schrodinger-Poisson solution."""
        return cls(np.asarray(z_m, dtype=float), -Q_E * np.asarray(phi_V, dtype=float), float(E_F_J),
                   float(m_eff_kg), T_K=float(T_K), alpha_np_per_eV=float(alpha_np_per_eV), g_s=g_s, g_v=g_v)

    @classmethod
    def flat_band(cls, t_film_m: float, n_bg_m3: float, m_eff_kg: float, *, nz: int = 201,
                  T_K: float = T_REF, alpha_np_per_eV: float = 0.0, g_s: int = 2,
                  g_v: int = 1) -> "QuantumFilm":
        """Flat-band film (U = 0) at the model's own bulk Fermi level for n_bg_m3
        (schrodinger_poisson.sp_bulk_fermi_level_J): the wide-film interior holds n_bg."""
        from dynameta.carriers.schrodinger_poisson import sp_bulk_fermi_level_J
        z = np.linspace(0.0, float(t_film_m), int(nz))
        ef = sp_bulk_fermi_level_J(float(n_bg_m3), float(m_eff_kg), float(T_K), g_s=g_s, g_v=g_v,
                                   alpha_np_per_eV=alpha_np_per_eV)
        return cls(z, np.zeros_like(z), ef, float(m_eff_kg), T_K=float(T_K),
                   alpha_np_per_eV=float(alpha_np_per_eV), g_s=g_s, g_v=g_v)

    @classmethod
    def from_dual_gate(cls, solver, solution, *,
                       optical_penetration_m: Optional[float] = None) -> "QuantumFilm":
        """Accessor for carriers.DualGateSchrodingerPoisson: recompute the COMPLETE sub-band set from the
        converged potential solution.phi_V (the solver keeps only the states it needs to fill the
        density; the response needs every state). Same grid, mass, Fermi level, temperature, Kane
        parameter and degeneracies as the solve. When the solve used finite oxide barriers or an
        exchange-correlation potential (solution.U_ext_J is set), the film is built on the solver's
        Schrodinger domain solution.z_ext_m (film plus the oxide penetration regions) with its full
        potential energy U_ext_J (band offsets, -q phi, V_xc) and per-cell masses, so the response sees
        the ground-state states. With finite barriers that domain extends into the oxides and the film
        records its nominal interval (nominal_m, extension_m); every optical output then describes the
        nominal film (class docstring), so optical stacks are given as designed.

        optical_penetration_m: how far the RESPONSE element may reach into each oxide [m]. None (default)
        = DEFAULT_OPTICAL_PENETRATION_M (2 nm), capped by the solver's own penetration and rounded to the
        grid; a value >= the solver's penetration keeps its whole domain. The element is cut at the grid
        node nearest that depth with the hard wall there (the solver's own boundary condition at the end of
        its domain); an UnreliableResultWarning names a cut where the solved density still exceeds
        TAIL_WARN (1e-6) of its peak. Why 2 nm: the state tails beyond a few decay lengths carry nothing
        (the solver's automatic domain ends where the density tail reaches 1e-16, 4-6 nm), the folded
        optics does not depend on the depth once the tail is resolved (measured: 1.6e-9 in r between 2 nm
        and the whole 3.86 nm domain at 3.0 eV, 2.8e-6 at 2.0 eV in the tunnelling regime; 1e-6 / 6e-5 at
        1 nm), and the response costs grow with the cube of the element's node count."""
        U_ext = getattr(solution, "U_ext_J", None)
        if U_ext is not None:
            z = np.asarray(solution.z_ext_m, dtype=float)
            U = np.asarray(U_ext, dtype=float)
            im = getattr(solution, "inv_mass_ext_per_kg", None)
            im = None if im is None else np.asarray(im, dtype=float)
            zf = np.asarray(solution.z_m, dtype=float)
            lo, hi = float(zf[0]), float(zf[-1])
            h = float(zf[1] - zf[0])
            pen = float(DEFAULT_OPTICAL_PENETRATION_M if optical_penetration_m is None
                        else optical_penetration_m)
            if not (np.isfinite(pen) and pen >= 0.0):
                raise ValueError("QuantumFilm.from_dual_gate: optical_penetration_m must be finite and >= 0, "
                                 "got {!r}".format(optical_penetration_m))
            ncut = int(round(pen / h))                   # keep [lo - ncut h, hi + ncut h]
            i0 = max(0, int(np.searchsorted(z, lo - 0.5 * h)) - ncut)
            i1 = min(z.size - 1, int(np.searchsorted(z, hi + 0.5 * h)) - 1 + ncut)
            if i0 > 0 or i1 < z.size - 1:
                n_ext = getattr(solution, "n_ext_m3", None)
                if n_ext is not None:
                    n_ext = np.asarray(n_ext, dtype=float)
                    peak = float(np.max(n_ext))
                    tail = max(float(n_ext[i0]) if i0 > 0 else 0.0,
                               float(n_ext[i1]) if i1 < z.size - 1 else 0.0)
                    if peak > 0.0 and tail > TAIL_WARN * peak:
                        warnings.warn(
                            "QuantumFilm.from_dual_gate: the optical element is cut {:.3g} nm into the oxide, "
                            "where the solved density is still {:.2g} of its peak (> {:g}); the cut states "
                            "differ from the solved ones -- pass a larger optical_penetration_m".format(
                                ncut * h * 1e9, tail / peak, TAIL_WARN), UnreliableResultWarning, stacklevel=2)
                z, U = z[i0:i1 + 1], U[i0:i1 + 1]
                im = None if im is None else im[i0:i1]
            nominal = (lo, hi) if (z[0] < lo - 1e-3 * h or z[-1] > hi + 1e-3 * h) else None
            return cls(z, U, solver.E_F_J, solver.m_eff_kg, T_K=solver.T_K, alpha_np_per_eV=solver.alpha_np,
                       g_s=solver.g_s, g_v=solver.g_v, inv_mass_cell_per_kg=im, nominal_m=nominal)
        return cls.from_potential(solution.z_m, solution.phi_V, solver.E_F_J, solver.m_eff_kg,
                                  T_K=solver.T_K, alpha_np_per_eV=solver.alpha_np, g_s=solver.g_s,
                                  g_v=solver.g_v)

    def drude_weights(self) -> np.ndarray:
        """In-plane Drude weight D_i [C^2 / (kg m^2)] per sub-band (sigma_i = i D_i / (omega + i gamma)): e^2 n_i / m
        for the parabolic band; for Kane, (e^2 g_s g_v / 2 pi hbar^2) int (-df/dE) E (1 + aE)/(1 + 2aE) dE
        (the Fermi-surface average of v_x^2 times the density of states)."""
        ns = np.asarray(self.sheet_density_m2, dtype=float)
        a = float(self.alpha_np_per_eV) / Q_E
        if a <= 0.0:
            return Q_E * Q_E * ns / float(self.m_eff_kg)
        kT = KB * float(self.T_K)
        pref = Q_E * Q_E * int(self.g_s) * int(self.g_v) / (2.0 * math.pi * HBAR ** 2)
        out = np.zeros_like(ns)
        x = np.linspace(0.0, 1.0, 4001)
        for k, Ei in enumerate(self.energies_J):
            eta = (float(self.fermi_level_J) - float(Ei)) / kT
            top = max(eta, 0.0) + 60.0
            if top <= 0.0 or ns[k] <= 0.0:
                continue
            e = x * top * kT                                     # kinetic energy grid [J]
            u = (e / kT) - eta
            mfp = 0.25 / np.cosh(0.5 * np.clip(u, -700.0, 700.0)) ** 2 / kT   # -df/dE
            g = mfp * e * (1.0 + a * e) / (1.0 + 2.0 * a * e)
            # Simpson on the uniform grid (odd point count): smooth integrand, 4000 intervals
            w = np.ones_like(x)
            w[1:-1:2] = 4.0
            w[2:-1:2] = 2.0
            out[k] = pref * float(np.sum(w * g)) * (e[1] - e[0]) / 3.0
        return out


# ------------------------------------------------------------------------------------------------
# Exchange-correlation kernel (adiabatic LDA)
# ------------------------------------------------------------------------------------------------
_CX = 0.75 * (9.0 / (4.0 * math.pi ** 2)) ** (1.0 / 3.0)        # exchange: eps_x = -_CX / r_s (Ha)
_PZ_HI = (-0.1423, 1.0529, 0.3334)                               # PZ81 unpolarized, r_s >= 1
_PZ_LO = (0.0311, -0.048, 0.0020, -0.0116)                       # PZ81 unpolarized, r_s < 1


def _eps_xc_terms(rs: np.ndarray, correlation: bool):
    """(eps, eps', eps'') of the per-particle xc energy in Hartree units and its r_s derivatives: the ONE
    implementation of the LDA functional, shared by the response kernel (lda_xc_kernel) and the
    ground-state potential (lda_xc_potential, used by carriers.DualGateSchrodingerPoisson)."""
    e = -_CX / rs
    d1 = _CX / rs ** 2
    d2 = -2.0 * _CX / rs ** 3
    if correlation:
        g, b1, b2 = _PZ_HI
        A, B, C, D = _PZ_LO
        hi = rs >= 1.0
        r = np.where(hi, rs, 1.0)
        sq = np.sqrt(r)
        den = 1.0 + b1 * sq + b2 * r
        t = b1 / (2.0 * sq) + b2
        c1_hi = -g * t / den ** 2
        c2_hi = g * (b1 / (4.0 * r ** 1.5) * den + 2.0 * t * t) / den ** 3
        rl = np.where(hi, 1.0, rs)
        c1_lo = A / rl + C * (np.log(rl) + 1.0) + D
        c2_lo = -A / rl ** 2 + C / rl
        e = e + np.where(hi, g / den, A * np.log(rl) + B + C * rl * np.log(rl) + D * rl)
        d1 = d1 + np.where(hi, c1_hi, c1_lo)
        d2 = d2 + np.where(hi, c2_hi, c2_lo)
    return e, d1, d2


def _eps_xc_derivs(rs: np.ndarray, correlation: bool):
    """(eps', eps'') of the per-particle xc energy in Hartree units, w.r.t. r_s."""
    return _eps_xc_terms(rs, correlation)[1:]


def _xc_units(m_eff_kg: float, eps_r: float):
    """(a*, Ha*): effective Bohr radius [m] and Hartree [J] of a host of mass m_eff_kg, permittivity eps_r."""
    a_star = 4.0 * math.pi * EPS0 * float(eps_r) * HBAR ** 2 / (float(m_eff_kg) * Q_E ** 2)
    return a_star, Q_E ** 2 / (4.0 * math.pi * EPS0 * float(eps_r) * a_star)


def lda_xc_potential(n_m3, m_eff_kg: float, eps_r: float, *, model: str = "lda") -> np.ndarray:
    """LDA exchange-correlation potential v_xc = d(n eps_xc)/dn [J] of a spin-unpolarized electron gas of
    effective mass m_eff_kg in a background of relative permittivity eps_r, at density n_m3 (array; 0
    where n <= 0): v_xc = eps_xc - (r_s/3) eps_xc' in effective atomic units (see lda_xc_kernel; the same
    functional, so d lda_xc_potential / dn = lda_xc_kernel). model 'x': exchange only,
    v_x = -(3 n / pi)^(1/3) e^2 / (4 pi eps0 eps_r); 'lda': + Perdew-Zunger (1981) correlation."""
    if model not in ("x", "lda"):
        raise ValueError("lda_xc_potential: model must be 'x' or 'lda', got {!r}".format(model))
    n = np.asarray(n_m3, dtype=float)
    a_star, ha_star = _xc_units(m_eff_kg, eps_r)
    out = np.zeros_like(n)
    pos = n > 0.0
    if np.any(pos):
        rs = (3.0 / (4.0 * math.pi * n[pos])) ** (1.0 / 3.0) / a_star
        e, d1, _d2 = _eps_xc_terms(rs, model == "lda")
        out[pos] = (e - (rs / 3.0) * d1) * ha_star
    return out


def lda_xc_kernel(n_m3, m_eff_kg: float, eps_r: float, *, model: str = "lda") -> np.ndarray:
    """Adiabatic LDA kernel f_xc = d v_xc / dn [J m^3] of a spin-unpolarized electron gas of effective
    mass m_eff_kg in a background of relative permittivity eps_r, at density n_m3 (array; 0 where
    n <= 0). model 'x': exchange only (closed form f_x = -(1/3)(3/pi)^(1/3) n^(-2/3) e^2/(4 pi eps0 eps_r));
    'lda': exchange + Perdew-Zunger (1981) correlation. Effective atomic units: a* = 4 pi eps0 eps_r
    hbar^2 / (m e^2), Ha* = e^2 / (4 pi eps0 eps_r a*), r_s = (3 / 4 pi n)^(1/3) / a*;
    v_xc = eps_xc - (r_s/3) eps_xc', f_xc = -(r_s/(3n)) [(2/3) eps_xc' - (r_s/3) eps_xc'']."""
    if model not in ("x", "lda"):
        raise ValueError("lda_xc_kernel: model must be 'x' or 'lda', got {!r}".format(model))
    n = np.asarray(n_m3, dtype=float)
    a_star, ha_star = _xc_units(m_eff_kg, eps_r)
    out = np.zeros_like(n)
    pos = n > 0.0
    if np.any(pos):
        nn = n[pos]
        rs = (3.0 / (4.0 * math.pi * nn)) ** (1.0 / 3.0) / a_star
        d1, d2 = _eps_xc_derivs(rs, model == "lda")
        out[pos] = -(rs / (3.0 * nn)) * ((2.0 / 3.0) * d1 - (rs / 3.0) * d2) * ha_star
    return out


# ------------------------------------------------------------------------------------------------
# Planar helpers (basis and orientation of optics.nonlocal_tmm)
# ------------------------------------------------------------------------------------------------
def uniaxial_layer_matrix(eps_xx: complex, eps_zz: complex, thickness_m: float, k0: complex,
                          k_par: complex) -> np.ndarray:
    """p-pol characteristic matrix of a uniform uniaxial layer (optic axis z) in nonlocal_tmm's
    (E_x, H_y / (omega eps0)) basis, superstrate face -> substrate face: generator
    [[0, i (k0^2 - q^2/eps_zz)], [i eps_xx, 0]], k_z^2 = eps_xx (k0^2 - q^2/eps_zz). Written without
    any division by k_z (finite at k_z = 0)."""
    exx, ezz, d = complex(eps_xx), complex(eps_zz), float(thickness_m)
    kz2 = exx * (complex(k0) ** 2 - complex(k_par) ** 2 / ezz)
    kz = np.sqrt(kz2 + 0j)
    p = kz * d
    sinc = np.sin(p) / p if abs(p) > 1e-6 else 1.0 - p * p / 6.0
    s_over_kz = d * sinc
    return np.array([[np.cos(p), 1j * kz2 * s_over_kz / exx],
                     [1j * exx * s_over_kz, np.cos(p)]], dtype=np.complex128)


def film_transfer_matrix(omega: float, k_par: float, h_m: float, eps_xx_nodes, eps_inf: float,
                         R, *, return_solver: bool = False):
    """Exact (staggered-grid) 2x2 transfer matrix of a planar film for p-polarization, in
    nonlocal_tmm's (E_x, Ht = H_y/(omega eps0)) basis, from its first grid face (s = 0, superstrate
    side) to the last (s = L), given:

      eps_xx_nodes (N,): the LOCAL in-plane permittivity at the N grid nodes (background + in-plane
          current); R (N-1, N-1): the out-of-plane response kernel on the N-1 half nodes, P/eps0 =
          R @ (D_z/eps0), any linear (nonlocal) map (a local eps_zz(s) is R = diag(1 - eps_inf/eps_zz)).

    Equations (s downward; E_x at nodes, Ht, D_z, E_z, P at half nodes, Ht also at the two faces):
        (E_{a+1} - E_a)/h = i k0^2 Ht_a + i q E_z,a,   E_z = ((I - R) d)/eps_inf,   d = D_z/eps0 = -q Ht,
        (Ht_{k+1/2} - Ht_{k-1/2})/h = i eps_xx,k E_k (interior nodes), half cells at the two faces.
    Its discrete Poynting theorem is exact: (omega eps0/2) Re(E_0 conj(Ht_0) - E_L conj(Ht_L)) =
    (omega eps0/2) [sum_k w_k h Im(eps_xx,k) |E_k|^2 + sum_a h Im(P_a conj(E_z,a))/eps0] (w = 1/2 at
    the face nodes). One dense solve of size 2N - 1 for two right-hand sides.

    return_solver=True also returns a callable (E_0, Ht_0) -> dict of the internal fields."""
    exx = np.asarray(eps_xx_nodes, dtype=np.complex128)
    N = exx.size
    Rm = np.asarray(R, dtype=np.complex128)
    if Rm.shape != (N - 1, N - 1):
        raise ValueError("film_transfer_matrix: R must be (N-1, N-1) for N = {} nodes".format(N))
    h = float(h_m)
    k0 = complex(omega) / C_LIGHT
    q = complex(k_par)
    # unknowns: E_1..E_{N-1} (N-1) | Ht_half 0..N-2 (N-1) | Ht_L (1)
    nE, nH = N - 1, N - 1
    nu = nE + nH + 1
    iE = lambda k: k - 1                                   # noqa: E731  (k = 1..N-1)
    iH = lambda a: nE + a                                  # noqa: E731
    iL = nE + nH
    A = np.zeros((nu, nu), dtype=np.complex128)
    rhs = np.zeros((nu, 2), dtype=np.complex128)           # columns: (E_0, Ht_0) = (1, 0), (0, 1)
    row = 0
    G = (q * q / complex(eps_inf)) * (np.eye(N - 1) - Rm)  # i q E_z = -i G @ Ht
    for a in range(N - 1):                                 # E_{a+1} - E_a - h i k0^2 Ht_a + h i (G Ht)_a = 0
        A[row, iE(a + 1)] += 1.0
        if a == 0:
            rhs[row, 0] += 1.0                             # + E_0 moves to the rhs
        else:
            A[row, iE(a)] -= 1.0
        A[row, nE:nE + nH] += 1j * h * G[a, :]
        A[row, iH(a)] -= 1j * h * k0 * k0
        row += 1
    for k in range(1, N - 1):                              # Ht_{k} - Ht_{k-1} - i h exx_k E_k = 0
        A[row, iH(k)] += 1.0
        A[row, iH(k - 1)] -= 1.0
        A[row, iE(k)] -= 1j * h * exx[k]
        row += 1
    A[row, iH(0)] = 1.0                                    # Ht_1/2 - Ht_0 - i (h/2) exx_0 E_0 = 0
    rhs[row, 1] += 1.0
    rhs[row, 0] += 1j * 0.5 * h * exx[0]
    row += 1
    A[row, iL] = 1.0                                       # Ht_L - Ht_{N-3/2} - i (h/2) exx_{N-1} E_{N-1} = 0
    A[row, iH(N - 2)] = -1.0
    A[row, iE(N - 1)] = -1j * 0.5 * h * exx[N - 1]
    row += 1
    assert row == nu
    from scipy.linalg import lu_factor, lu_solve
    X = lu_solve(lu_factor(A, check_finite=False), rhs, check_finite=False)
    T = np.array([[X[iE(N - 1), 0], X[iE(N - 1), 1]],
                  [X[iL, 0], X[iL, 1]]], dtype=np.complex128)
    if not return_solver:
        return T

    def fields(E0: complex, H0: complex) -> dict:
        x = X @ np.array([E0, H0], dtype=np.complex128)
        E = np.concatenate([[E0], x[:nE]])
        Ht = x[nE:nE + nH]
        d = -q * Ht
        P_eps0 = Rm @ d
        Ez = (d - P_eps0) / complex(eps_inf)
        return {"E_x": E, "Ht_half": Ht, "Ht_faces": (complex(H0), complex(x[iL])), "E_z": Ez,
                "P_over_eps0": P_eps0, "D_over_eps0": d}
    return T, fields


def film_stack_rt(omega: float, film_matrix, above: Sequence = (), below: Sequence = (), *,
                  n_super=1.0, n_sub=1.0, theta_rad: float = 0.0, k_par_m=None,
                  check_energy: bool = True):
    """r, t, R, T, A of super | above | FILM | below | sub for p-polarization, with the film given as its
    2x2 transfer matrix (nonlocal_tmm basis and orientation) or a callable (omega, k_par) -> matrix, and
    above / below lists of nonlocal_tmm layers (DielectricLayer / HydroLayer), superstrate side first.
    Returns optics.nonlocal_tmm.RTA.

    SIGN OF r (p-polarization): the library's p-hat convention, r = r_H = H_y(refl) / H_y(inc) =
    -E_x(refl) / E_x(inc) -- the convention of optics.fem_2d_tm (polarization 'p'), the lumenairy
    bridges, optics.tmm_reference and optics.quantum_film_cell.solve_film_cell, so the same planar
    stack gives the same r through every one of them. optics.nonlocal_tmm.stack_rt (unchanged since
    v0.15.1) returns the tangential-E ratio for p, i.e. the NEGATIVE of this r; the two are related by
    r_here = -nonlocal_tmm.stack_rt(...).r exactly (tests/test_quantum_film.py pins it). t is the
    tangential-E ratio E_x(trans) / E_x(inc) (the basis's own, as in nonlocal_tmm.stack_rt; T uses it),
    which differs from fem_2d_tm's t = t_H n_super / n_sub by the positive factor
    cos(theta_sub) / cos(theta_super). R, T, A and D_pole do not depend on the sign."""
    from dynameta.optics import nonlocal_tmm as nt
    from dynameta.optics.tmm_reference import _check_energy_budget
    w = complex(omega)
    k0 = w / C_LIGHT
    eps_super, eps_sub = complex(n_super) ** 2, complex(n_sub) ** 2
    k_par = (complex(n_super) * k0 * math.sin(theta_rad)) if k_par_m is None else complex(k_par_m)
    Tf = film_matrix(omega, k_par) if callable(film_matrix) else np.asarray(film_matrix, dtype=complex)
    M = nt._stack_matrix(w, list(below), k0, k_par, "p") @ Tf @ nt._stack_matrix(w, list(above), k0,
                                                                                    k_par, "p")
    Y_super = nt._admittance(eps_super, nt._kz(eps_super, k0, k_par), "p")
    Y_sub = nt._admittance(eps_sub, nt._kz(eps_sub, k0, k_par), "p")
    r_E, t, D_pole = nt._rt_from_matrix(M, Y_super, Y_sub)       # r_E: tangential-E ratio
    r = -r_E                                                     # p-hat convention (docstring)
    R = float(abs(r) ** 2)
    T = float(abs(t) ** 2 * (Y_sub.real / Y_super.real)) if Y_super.real != 0.0 else float("nan")
    if check_energy and w.imag == 0.0 and Y_super.real > 0.0 and eps_super.imag == 0.0:
        _check_energy_budget(R, T, where="quantum_film.film_stack_rt")
    return nt.RTA(r=r, t=t, R=R, T=T, A=1.0 - R - T, D_pole=D_pole)


# ------------------------------------------------------------------------------------------------
# The response
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantumFilmValidity:
    """Validity numbers of the q -> 0 electronic response (q_over_kF, q_vF_over_omega; corrections ~ their
    squares) and of the thin-film equivalent layer (qL, k0L_eps = k0 L sqrt(max |eps|)); ok and message
    refer to the first two."""
    q_over_kF: float = 0.0
    q_vF_over_omega: float = 0.0
    qL: float = 0.0
    k0L_eps: float = 0.0
    ok: bool = True
    message: str = ""


class QuantumFilmResponse:
    """Self-consistent linear optical response of a QuantumFilm (see the module docstring).

    Parameters: film (QuantumFilm); eps_inf (background permittivity of the film at optical frequency,
    real > 0); gamma_rad_s (damping of every transition, Omega^2 = omega (omega + i gamma));
    gamma_inplane_rad_s (in-plane Drude damping, default gamma_rad_s); xc ('none' = RPA, 'x' = + adiabatic
    LDA exchange, 'lda' = + exchange and PZ81 correlation); eps_xc (background permittivity of the
    exchange-correlation interaction, default eps_inf); occupation_tol (sub-bands with
    n_i <= occupation_tol x total are treated as empty: pairs among empty states carry
    |n_i - n_j| below it); max_states (keep only the lowest states: a TRUNCATED basis, for model studies
    such as the two-level depolarization shift; None = the complete discrete basis); route ('auto',
    'modes', 'real_space'; see the module docstring; 'auto' uses the modes for alpha_perp when the pair
    count is <= max_pairs_modes and the real-space route for the kernels and the transfer matrix);
    max_pairs_modes (default 3000)."""

    def __init__(self, film: QuantumFilm, eps_inf: float, gamma_rad_s: float, *,
                 gamma_inplane_rad_s: Optional[float] = None, xc: str = "none",
                 eps_xc: Optional[float] = None, occupation_tol: float = 1e-12,
                 max_states: Optional[int] = None, route: str = "auto",
                 max_pairs_modes: int = 3000):
        if xc not in _XC_MODELS:
            raise ValueError("QuantumFilmResponse: xc must be one of {}, got {!r}".format(_XC_MODELS, xc))
        if route not in _ROUTES:
            raise ValueError("QuantumFilmResponse: route must be one of {}, got {!r}".format(_ROUTES, route))
        self.film = film
        self.eps_inf = float(eps_inf)
        if not (np.isfinite(self.eps_inf) and self.eps_inf > 0.0):
            raise ValueError("QuantumFilmResponse: eps_inf must be real and > 0")
        self.gamma = float(gamma_rad_s)
        self.gamma_inplane = self.gamma if gamma_inplane_rad_s is None else float(gamma_inplane_rad_s)
        if self.gamma < 0.0 or self.gamma_inplane < 0.0:
            raise ValueError("QuantumFilmResponse: damping rates must be >= 0")
        self.xc = xc
        self.eps_xc = self.eps_inf if eps_xc is None else float(eps_xc)
        self.route = route
        self.max_pairs_modes = int(max_pairs_modes)
        h = film.h
        self.h = h
        self.N = film.z_m.size
        E = np.asarray(film.energies_J, dtype=float)
        psi = np.asarray(film.psi, dtype=float)
        ns = np.asarray(film.sheet_density_m2, dtype=float)
        if max_states is not None:
            E, psi, ns = E[:int(max_states)], psi[:, :int(max_states)], ns[:int(max_states)]
        self.truncated = max_states is not None and int(max_states) < film.energies_J.size
        self.E, self.psi, self.ns = E, psi, ns
        tot = float(np.sum(ns))
        if not (tot > 0.0):
            raise ValueError("QuantumFilmResponse: the film holds no electrons")
        self.occ = np.flatnonzero(ns > float(occupation_tol) * tot)
        # pairs (i occupied, j > i)
        ii, jj = [], []
        for i in self.occ:
            j = np.arange(i + 1, E.size)
            ii.append(np.full(j.size, i))
            jj.append(j)
        self.pi = np.concatenate(ii) if ii else np.zeros(0, int)
        self.pj = np.concatenate(jj) if jj else np.zeros(0, int)
        dn = ns[self.pi] - ns[self.pj]
        if np.any(dn < -1e-12 * tot):
            raise ValueError("QuantumFilmResponse: inverted sub-band populations are not supported "
                             "(the response would contain gain)")
        self.dn = np.maximum(dn, 0.0)
        self.wp = (E[self.pj] - E[self.pi]) / HBAR
        self.s = Q_E * Q_E * self.dn * 2.0 * self.wp / (HBAR * EPS0)
        n_int = self.N - 2
        # Xi_p at the N-1 half nodes: Xi_{a} = h sum_{l <= a} zeta_l (interior node l = 1..N-2)
        self._zeta = None
        self._Xi = None
        self._n_int = n_int
        self._modes = None
        self.f_xc = (np.zeros(n_int) if xc == "none"
                     else lda_xc_kernel(film.density_m3[1:-1], film.m_eff_kg, self.eps_xc, model=xc))
        D = film.drude_weights()
        if max_states is not None:
            D = D[:int(max_states)]
        self.drude_weights = D
        # in-plane Drude weight density on the full grid (0 at the faces)
        dw = np.zeros(self.N)
        dw[1:-1] = (psi ** 2) @ D
        self._drude_density = dw

    # ---- transition-basis ingredients ----
    @property
    def n_pairs(self) -> int:
        return int(self.pi.size)

    def _pair_arrays(self):
        if self._Xi is None:
            zeta = self.psi[:, self.pi] * self.psi[:, self.pj]                 # (N-2, P)
            Xi = np.zeros((self.N - 1, zeta.shape[1]))
            Xi[1:, :] = self.h * np.cumsum(zeta, axis=0)
            self._zeta, self._Xi = zeta, Xi
        return self._zeta, self._Xi

    def dipoles_m(self) -> np.ndarray:
        """z_p = <psi_i| z |psi_j> of every pair [m] (= -int Xi_p dz, exactly the Riemann dipole)."""
        _, Xi = self._pair_arrays()
        return -self.h * np.sum(Xi, axis=0)

    def modes(self) -> dict:
        """The collective (RPA / TDDFT) modes of the uniform-D_z response: eigen-decomposition of
        B = diag(w_p^2) + S^1/2 K S^1/2. Returns {'omega2' (Lambda_m, rad^2/s^2), 'strength' (F_m, m*s^-2
        so that alpha_perp = sum F_m / (Lambda_m - Omega^2)), 'U', 'Phi' (mode polarization shapes on the
        half nodes)}. Cached."""
        if self._modes is not None:
            return self._modes
        zeta, Xi = self._pair_arrays()
        h = self.h
        K = (h / self.eps_inf) * (Xi.T @ Xi)
        if self.xc != "none":
            K = K + (EPS0 / Q_E ** 2) * h * (zeta.T @ (zeta * self.f_xc[:, None]))
        sq = np.sqrt(self.s)
        B = sq[:, None] * K * sq[None, :]
        B[np.diag_indices_from(B)] += self.wp ** 2
        lam, U = np.linalg.eigh(B)
        if lam.size and not (lam[0] > 0.0):
            raise ValueError("QuantumFilmResponse: the self-consistent response is unstable (a mode "
                             "with Omega^2 = {:.3g} <= 0; the exchange-correlation kernel overwhelms "
                             "the Hartree term)".format(lam[0]))
        v = sq * self.dipoles_m()
        F = (U.T @ v) ** 2 / self.eps_inf
        Phi = Xi @ (sq[:, None] * U)
        self._modes = {"omega2": lam, "strength": F, "U": U, "Phi": Phi}
        return self._modes

    def fsum(self) -> dict:
        """f-sum bookkeeping (the oracle is the discrete identity): 'bare' = (e^2/(eps0 eps_inf m))
        sum_p (n_i - n_j) f_p, 'identity' = (e^2/(eps0 eps_inf m)) sum_i n_i kappa_i (complete discrete
        basis), 'ratio_bare' = bare / (e^2 N_s/(eps0 eps_inf m)), and 'modes' = sum_m F_m when the modes
        are already built (call modes() first to include them)."""
        m = float(self.film.m_eff_kg)
        pref = Q_E * Q_E / (EPS0 * self.eps_inf * m)
        z = self.dipoles_m()
        f = 2.0 * m * self.wp * z * z / HBAR
        bare = pref * float(np.sum(self.dn * f))
        kappa = self.h * np.sum(self.psi[:-1, :] * self.psi[1:, :], axis=0)
        out = {"bare": bare, "identity": pref * float(np.sum(self.ns * kappa)),
               "ratio_bare": bare / (pref * float(np.sum(self.ns)))}
        if self._modes is not None:
            out["modes"] = float(np.sum(self._modes["strength"]))
        return out

    # ---- frequency helpers ----
    def _Omega2(self, omega) -> complex:
        w = complex(omega)
        return w * (w + 1j * self.gamma)

    def _use_modes(self, for_kernel: bool = False) -> bool:
        if self.truncated or self.route == "modes":
            return True
        if self.route == "real_space" or for_kernel:
            return False
        return self.n_pairs <= self.max_pairs_modes

    # ---- real-space (Sternheimer) route ----
    def _hamiltonian_bands(self):
        h = self.h
        U = self.film.U_J[1:-1]
        if self.film.inv_mass_cell_per_kg is not None:            # heterostructure: per-cell masses
            tc = (HBAR ** 2 / (2.0 * h * h)) * self.film.inv_mass_cell_per_kg
            return tc[:-1] + tc[1:] + U, -tc[1:-1]
        t = HBAR ** 2 / (2.0 * float(self.film.m_eff_kg) * h * h)
        return 2.0 * t + U, -t

    def chi0(self, omega) -> np.ndarray:
        """Independent-particle density response chi0(z_l, z_m; omega) on the interior nodes [m^-3 / J]
        (dn_l = sum_m chi0_lm V_m, V a potential energy), from the Sternheimer equations
        (E_i +- hbar Omega - H) X = diag(psi_i) for the occupied states (no empty states needed; the
        complete discrete basis is implicit in the resolvent). Requires the complete basis (max_states
        None)."""
        if self.truncated:
            raise ValueError("QuantumFilmResponse.chi0: the real-space route needs the complete basis "
                             "(max_states=None)")
        from scipy.linalg import solve_banded
        Om = np.sqrt(self._Omega2(omega))
        if Om.real < 0.0:
            Om = -Om
        diag, off = self._hamiltonian_bands()
        n = diag.size
        ab = np.zeros((3, n), dtype=np.complex128)
        ab[0, 1:] = -off                                      # (E - H): off-diagonals +t
        ab[2, :-1] = -off
        chi = np.zeros((n, n), dtype=np.complex128)
        for i in self.occ:
            p = self.psi[:, i]
            rhs = np.diag(p).astype(np.complex128)
            acc = np.zeros((n, n), dtype=np.complex128)
            for sgn in (1.0, -1.0):
                ab[1, :] = self.E[i] + sgn * HBAR * Om - diag
                acc += solve_banded((1, 1), ab, rhs, check_finite=False)
            chi += self.ns[i] * (p[:, None] * acc)
        return chi

    def _cumulative(self) -> np.ndarray:
        """C (N-1, N-2): C[a, l] = 1 for interior node l (1-based) <= half node a."""
        n_int = self.N - 2
        a = np.arange(self.N - 1)[:, None]
        l_ = np.arange(1, n_int + 1)[None, :]
        return (l_ <= a).astype(float)

    def _real_space_R(self, omega) -> np.ndarray:
        chi = self.chi0(omega)
        C = self._cumulative()
        h = self.h
        c = Q_E * Q_E * h * h / (EPS0 * self.eps_inf)
        A = np.eye(chi.shape[0], dtype=np.complex128) - chi @ (c * (C.T @ C) + np.diag(self.f_xc))
        Y = np.linalg.solve(A, chi @ C.T)                     # dn = -(e h / eps_inf) Y d
        return -c * (C @ Y)

    # ---- the response ----
    def response_kernel(self, omega) -> np.ndarray:
        """Self-consistent out-of-plane response R (N-1, N-1) on the half nodes: P/eps0 = R @ (D_z/eps0)
        for an arbitrary D_z(z) profile (q -> 0 electronic response, Hartree + xc). Modes route:
        R = (h/eps_inf) Phi diag(1/(Lambda - Omega^2)) Phi^T; real-space route: from chi0."""
        if self._use_modes(for_kernel=True):
            md = self.modes()
            g = 1.0 / (md["omega2"] - self._Omega2(omega))
            return (self.h / self.eps_inf) * (md["Phi"] * g[None, :]) @ md["Phi"].T
        return self._real_space_R(omega)

    def alpha_perp(self, omega) -> complex:
        """Out-of-plane sheet response length alpha_perp = p_s / D_z [m] under a uniform D_z (Im >= 0)."""
        if self._use_modes():
            md = self.modes()
            return complex(np.sum(md["strength"] / (md["omega2"] - self._Omega2(omega))))
        R = self._real_space_R(omega)
        return complex(self.h * np.sum(R))

    def sheet_conductivity(self, omega) -> complex:
        """In-plane sheet conductivity sigma_s = i sum_i D_i / (omega + i gamma_inplane) [S]."""
        w = complex(omega)
        return complex(1j * float(np.sum(self.drude_weights)) / (w + 1j * self.gamma_inplane))

    def alpha_par(self, omega) -> complex:
        """In-plane sheet response length alpha_par = i sigma_s / (omega eps0) [m]."""
        return complex(1j * self.sheet_conductivity(omega) / (complex(omega) * EPS0))

    def eps_xx_nodes(self, omega) -> np.ndarray:
        """Local in-plane permittivity on the grid nodes, eps_inf + i sigma(z) / (omega eps0)."""
        w = complex(omega)
        sig = 1j * self._drude_density / (w + 1j * self.gamma_inplane)
        return self.eps_inf + 1j * sig / (w * EPS0)

    def effective_layer(self, omega):
        """(eps_xx,eq, eps_zz,eq, L): the uniform uniaxial layer of the film's thickness that carries the
        same sheet responses (first order in the thickness; see the module docstring). L is the NOMINAL
        film thickness: an element extended into the oxides (QuantumFilm.extension_m) puts its sheet
        responses into the nominal layer and leaves the oxides as designed (QuantumFilm class docstring)."""
        L = self.film.nominal_thickness_m
        exx = self.eps_inf + self.alpha_par(omega) / L
        ezz = self.eps_inf / (1.0 - self.alpha_perp(omega) / L)
        return complex(exx), complex(ezz), L

    def local_eps_zz(self, omega) -> complex:
        """The LOCAL (uniform-field) effective eps_zz of the same sub-band set, eps_inf + sum g_p z_p^2 / L
        (IntersubbandEffect's 'subbands' form with Leff = L), for comparison."""
        z = self.dipoles_m()
        g = self.s / (self.wp ** 2 - self._Omega2(omega))
        return complex(self.eps_inf + np.sum(g * z * z) / self.film.nominal_thickness_m)

    def sigma_zz_kernel(self, omega) -> np.ndarray:
        """Independent-particle (Kohn-Sham) nonlocal conductivity sigma_zz(z_a, z_b) [S/m^2] on the half
        nodes (J_z,a = h sum_b sigma_ab E_eff,b): -i omega eps0 sum_p g_p Xi_p(z_a) Xi_p(z_b)."""
        _, Xi = self._pair_arrays()
        g = self.s / (self.wp ** 2 - self._Omega2(omega))
        return -1j * complex(omega) * EPS0 * (Xi * g[None, :]) @ Xi.T

    def induced_charge(self, omega, D_z: float = 1.0):
        """Self-consistent induced charge density rho(z) [C/m^3] on the interior nodes and P(z) [C/m^2]
        on the half nodes, under a uniform D_z [C/m^2]. Returns (z_interior, rho, z_half, P)."""
        R = self.response_kernel(omega)
        P = EPS0 * (R @ np.full(self.N - 1, float(D_z) / EPS0))
        rho = -(P[1:] - P[:-1]) / self.h
        z = self.film.z_m
        return z[1:-1], rho, 0.5 * (z[:-1] + z[1:]), P

    def transfer_matrix(self, omega, k_par, *, superstrate_face: str = "top", R=None):
        """Exact planar p-pol transfer matrix of the film (film_transfer_matrix with this response).
        superstrate_face 'top' means z_max faces the superstrate (the dual-gate solver's top gate side),
        'bottom' means z = z_min does. It maps the NOMINAL film's faces (an extended element is folded back,
        QuantumFilm class docstring), so stacks around it are given as designed."""
        if superstrate_face not in ("top", "bottom"):
            raise ValueError("superstrate_face must be 'top' or 'bottom'")
        Rk = self.response_kernel(omega) if R is None else R
        exx = self.eps_xx_nodes(omega)
        if superstrate_face == "top":
            Rk = Rk[::-1, ::-1]
            exx = exx[::-1]
        T = film_transfer_matrix(omega, k_par, self.h, exx, self.eps_inf, Rk)
        below, above = self.film.extension_m                  # z_min side, z_max side
        if below > 0.0 or above > 0.0:
            # fold the penetration regions back onto the nominal film (QuantumFilm class docstring): the
            # inverse of a pure eps_inf layer of each extension depth, so the matrix maps the nominal faces
            up, down = (above, below) if superstrate_face == "top" else (below, above)
            k0 = complex(omega) / C_LIGHT
            T = (uniaxial_layer_matrix(self.eps_inf, self.eps_inf, -down, k0, k_par) @ T
                 @ uniaxial_layer_matrix(self.eps_inf, self.eps_inf, -up, k0, k_par))
        return T

    def stack_rt(self, omega, above: Sequence = (), below: Sequence = (), *, n_super=1.0, n_sub=1.0,
                 theta_rad: float = 0.0, k_par_m=None, model: str = "rpa",
                 superstrate_face: str = "top", check_energy: bool = True):
        """p-pol r, t, R, T, A of super | above | film | below | sub (film_stack_rt: r in the library's
        p-hat convention r = r_H = -E_x(refl)/E_x(inc), as fem_2d_tm, the bridges and solve_film_cell;
        the negative of nonlocal_tmm.stack_rt's r). model 'rpa' = the exact film element
        (transfer_matrix); 'equivalent' = the uniform uniaxial effective layer (effective_layer). above /
        below are the DESIGNED neighbours of the nominal film, also for an element extended into them."""
        if model == "rpa":
            fm = lambda w, kp: self.transfer_matrix(w, kp, superstrate_face=superstrate_face)  # noqa: E731
        elif model == "equivalent":
            exx, ezz, L = self.effective_layer(omega)
            fm = lambda w, kp: uniaxial_layer_matrix(exx, ezz, L, complex(w) / C_LIGHT, kp)  # noqa: E731
        else:
            raise ValueError("model must be 'rpa' or 'equivalent'")
        return film_stack_rt(omega, fm, above, below, n_super=n_super, n_sub=n_sub, theta_rad=theta_rad,
                             k_par_m=k_par_m, check_energy=check_energy)

    def validity(self, omega, k_par, *, tol: float = 0.1) -> QuantumFilmValidity:
        """Numbers that bound the approximations: q/k_F and q v_F/omega (the q -> 0 electronic
        response; corrections ~ their squares), q L and k0 L sqrt(max|eps|) (the first-order
        equivalent layer; the exact element has no such limit). ok when the first two are < tol (and
        the equivalent-layer pair is reported, not gated)."""
        nmax = float(np.max(self.film.density_m3))
        kF = (3.0 * math.pi ** 2 * nmax) ** (1.0 / 3.0)
        vF = HBAR * kF / float(self.film.m_eff_kg)
        q = abs(complex(k_par))
        w = abs(complex(omega))
        exx, ezz, L = self.effective_layer(omega)
        vals = {"q_over_kF": q / kF, "q_vF_over_omega": q * vF / w}
        bad = [k for k, v in vals.items() if v >= tol]
        msg = "" if not bad else "q -> 0 electronic response questionable: " + ", ".join(
            "{} = {:.3g}".format(k, vals[k]) for k in bad)
        if bad:
            warnings.warn("QuantumFilmResponse.validity: " + msg, UnreliableResultWarning, stacklevel=2)
        return QuantumFilmValidity(q_over_kF=vals["q_over_kF"], q_vF_over_omega=vals["q_vF_over_omega"],
                                   qL=q * L, k0L_eps=w / C_LIGHT * L * math.sqrt(max(abs(exx), abs(ezz), 1.0)),
                                   ok=not bad, message=msg)
