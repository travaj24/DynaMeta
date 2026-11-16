"""
1D effective-mass Schrodinger-Poisson for the degenerate ITO accumulation layer.

The classical drift-diffusion / Poisson Stage 1 misses the sub-band quantization of
the ~1 nm electron accumulation layer in degenerate ITO. This module provides the
self-consistent quantum reference: a 1D BenDaniel-Duke Schrodinger solve through the
stack, filled by the DEGENERATE 2D sub-band density (NOT Boltzmann), iterated against
Poisson by the Trellakis predictor-corrector (a nonlinear-Poisson inner solve, which
converges where naive Picard sloshes).

It is solver-agnostic (pure numpy/scipy, SI units) and validates against analytic
square- and triangular-well sub-band energies. See docs/implementation_notes.md for
the derivation + sources (Tan/Snider 1990, Trellakis 1997, Gao 2014).

Conventions: z in metres, energies in Joules internally (helpers accept/return eV at
the boundary where noted), potential energy U(z) = -q*phi(z) (electron PE), densities
in m^-3, sheet densities in m^-2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Physical constants (SI): single source in core/constants. `Q` is this module's local alias
# for the elementary charge (used pervasively in the band-physics expressions, e.g. -Q*phi).
from dynameta.constants import EPS0, KB, HBAR, M_E, Q_E as Q  # noqa: F401
from dynameta.warning_types import UnreliableResultWarning
from dynameta.constants import T_REF   # audit X-24: one reference temperature


def _fermi_log(x: np.ndarray) -> np.ndarray:
    """ln(1+exp(x)) with an overflow-safe large-x branch (-> x for x >> 1). The 2D
    degenerate occupation integral; for strongly degenerate ITO x can be >> 1. This is the
    complete Fermi-Dirac integral of order 0, F_0(x) = Int_0^inf 1/(1+e^(t-x)) dt."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    big = x > 40.0
    out[big] = x[big]
    out[~big] = np.log1p(np.exp(np.clip(x[~big], -700.0, 40.0)))
    return out


_FD1_SERIES_K = np.arange(1, 41, dtype=np.float64)   # x < -1: e^x < 0.368, 0.368^41 ~ 1e-18


def _fd1(x: np.ndarray) -> np.ndarray:
    """Complete Fermi-Dirac integral of order 1: F_1(x) = Int_0^inf t/(1+e^(t-x)) dt = -Li_2(-e^x).
    Three branches, each accurate to a few ulp (gated against a 40-digit mpmath polylog and the
    dF_1/dx = F_0 identity in tests/test_audit_2026_07_17_wave7_carriers.py, audit S1-18):
      * x < -1: the alternating series -Li_2(-w) = sum_k (-1)^(k+1) w^k / k^2, w = e^x < 0.37 (40 terms);
      * -1 <= x <= 40: the dilogarithm closed form Li_2(w) = scipy.special.spence(1-w), F_1 = -spence(1+e^x);
      * x > 40: the Sommerfeld form x^2/2 + pi^2/6 (exact up to -F_1(-x) ~ e^-40 ~ 4e-18 relative x^2).
    AUDIT S1-18 (wave-7, 2026-11-02): the spence branch used to run down to x = -700. Its argument
    1 + e^x keeps e^x only to 2.2e-16 ABSOLUTE, so the relative error grew as 2.2e-16 e^-x: measured
    1.0e-3 at x = -30, and F_1 = 0 EXACTLY for x < -36.7 (1 + e^x == 1.0) -- the "verified to ~1e-14"
    claim held only for x >~ -5. Used by the nonparabolic (Kane) 2D sub-band sheet density, where the
    m*(eps)=m*0(1+2 alpha eps) DOS gives n_s = pref0 (kT F_0 + 2 alpha kT^2 F_1): the non-degenerate
    (high) sub-bands lost their whole 2 alpha kT nonparabolic share there."""
    from scipy.special import spence
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    big = x > 40.0
    small = x < -1.0
    mid = ~(big | small)
    out[big] = 0.5 * x[big] ** 2 + (np.pi ** 2) / 6.0
    out[mid] = -spence(1.0 + np.exp(x[mid]))
    if np.any(small):
        w = np.exp(x[small])[..., None]
        k = _FD1_SERIES_K
        out[small] = np.sum(np.where(k % 2.0 == 1.0, 1.0, -1.0) * w ** k / (k * k), axis=-1)
    return out


# ---- bulk (flat-band) limit of the sub-band model: the ONE Fermi-level kernel -----------------------
#
# The sub-band fill of SchrodingerPoisson1D is separable: a PARABOLIC dispersion along z (mass m) and,
# per sub-band, an in-plane 2D DOS with Kane nonparabolicity m*(eps) = m (1 + 2 alpha eps), giving the
# sheet density n_s(E_i) = pref0 [kT F_0(eta) + 2 alpha kT^2 F_1(eta)], eta = (E_F - E_i)/kT,
# pref0 = g_s g_v m / (2 pi hbar^2). In a wide flat-band slab the sub-band floors become
# E(k_z) = hbar^2 k_z^2 / (2 m) with k_z = i pi / L and |psi_i|^2 -> 1/L on average, so the centre density
# tends to
#     n_sep(E_F) = (1/pi) Int_0^inf dk_z n_s(hbar^2 k_z^2 / 2m).
# Substituting u = hbar^2 k_z^2 / (2 m kT) and exchanging the order of integration gives the closed form
#     n_sep = N_c [F_1/2(eta_F) + 2 alpha kT F_3/2(eta_F)],  N_c = g_s g_v (m kT / (2 pi hbar^2))^(3/2),
# with the normalized Fermi-Dirac integrals F_j(x) = Int_0^inf t^j / (1 + e^(t - x)) dt / Gamma(j + 1).
# At alpha = 0 this is the textbook parabolic 3D density; for alpha > 0 it is NOT the 3D Kane density
# (whose dispersion is nonparabolic along z too): at T -> 0 the ratio of the two at one E_F is
# (1 + 0.8 alpha E_F) / (1 + alpha E_F)^(3/2). The Fermi level that makes the model's own bulk equal a
# background density is therefore the root of n_sep(E_F) = n_bg, computed here by quadrature of the SAME
# sheet-density kernels (_fermi_log, _fd1) the solver fills with.

def sp_bulk_density_m3(E_F_J: float, m_kg: float, T_K: float, *, g_s: int = 2, g_v: int = 1,
                       alpha_np_per_eV: float = 0.0) -> float:
    """Flat-band bulk electron density [m^-3] of the separable sub-band model at Fermi level E_F_J [J]
    (band edge E_c = 0): n_sep = (1/pi) Int_0^inf dk_z pref0 [kT F_0(eta) + 2 a kT^2 F_1(eta)],
    eta = (E_F - hbar^2 k_z^2 / 2m) / kT. The z dispersion is PARABOLIC with mass m_kg; alpha_np_per_eV
    is the in-plane Kane nonparabolicity [eV^-1] of SchrodingerPoisson1D.density. Quadrature to the
    wavevector where the floor reaches max(E_F, 0) + 60 kT (the neglected tail is ~e^-60 relative)."""
    from scipy.integrate import quad
    m = float(m_kg)
    kT = KB * float(T_K)
    a = float(alpha_np_per_eV) / Q if alpha_np_per_eV else 0.0          # J^-1
    pref0 = int(g_s) * int(g_v) * m / (2.0 * np.pi * HBAR ** 2)          # m^-2 J^-1
    EF = float(E_F_J)
    ck = HBAR ** 2 / (2.0 * m)                                           # E(k) = ck k^2

    def integrand(k):
        eta = np.array([(EF - ck * k * k) / kT])
        val = kT * _fermi_log(eta)
        if a > 0.0:
            val = val + 2.0 * a * kT ** 2 * _fd1(eta)
        return float(pref0 * val[0])

    k_max = np.sqrt((max(EF, 0.0) + 60.0 * kT) / ck)
    points = [np.sqrt(EF / ck)] if EF > 0.0 else None                    # the Fermi edge eta = 0
    val, _err = quad(integrand, 0.0, k_max, points=points, epsabs=0.0, epsrel=1e-13, limit=400)
    return float(val / np.pi)


def sp_bulk_fermi_level_J(n_bg_m3: float, m_kg: float, T_K: float, *, g_s: int = 2, g_v: int = 1,
                          alpha_np_per_eV: float = 0.0) -> float:
    """Fermi level E_F [J] (relative to E_c) at which the separable sub-band model's own flat-band bulk
    density sp_bulk_density_m3 equals n_bg_m3, at the finite temperature T_K. This is the ONE bulk-E_F
    kernel of the Schrodinger-Poisson path (every caller imports it). Root by brentq (xtol 1e-12 kT).

    Bracket: n_sep increases monotonically with E_F. Upper end: the T = 0 parabolic Fermi energy
    E0 = (hbar^2 / 2m)(6 pi^2 n / (g_s g_v))^(2/3) plus 10 kT (finite T and alpha >= 0 both RAISE n_sep
    at fixed E_F above its T = 0 parabolic value, so n_sep(E0 + 10 kT) > n_bg). Lower end: since
    F_j(x) <= e^x, n_sep <= N_c e^eta (1 + 2 alpha kT); E_F = kT ln(n_bg / (N_c (1 + 2 alpha kT))) - kT
    gives n_sep <= n_bg / e. Both ends are checked and widened if ever needed."""
    from scipy.optimize import brentq
    n_bg = float(n_bg_m3)
    if not (np.isfinite(n_bg) and n_bg > 0.0):
        raise ValueError("sp_bulk_fermi_level_J: n_bg_m3 must be a finite positive density, got "
                         "{!r}".format(n_bg_m3))
    m = float(m_kg)
    kT = KB * float(T_K)
    a = float(alpha_np_per_eV) / Q if alpha_np_per_eV else 0.0
    if a < 0.0:
        raise ValueError("sp_bulk_fermi_level_J: alpha_np_per_eV must be >= 0, got "
                         "{!r}".format(alpha_np_per_eV))
    gsv = int(g_s) * int(g_v)
    alpha_eV = a * Q

    def f(E):
        return sp_bulk_density_m3(E, m, T_K, g_s=g_s, g_v=g_v, alpha_np_per_eV=alpha_eV) / n_bg - 1.0

    from dynameta.carriers.physics_equilibrium import effective_dos_m3   # the one N_c kernel
    E0 = (HBAR ** 2 / (2.0 * m)) * (6.0 * np.pi ** 2 * n_bg / gsv) ** (2.0 / 3.0)
    Nc = 0.5 * gsv * effective_dos_m3(m, float(T_K))           # g_s g_v (m kT / 2 pi hbar^2)^(3/2)
    lo = min(kT * np.log(n_bg / (Nc * (1.0 + 2.0 * a * kT))) - kT, E0 - kT)
    hi = E0 + 10.0 * kT
    f_lo, f_hi = f(lo), f(hi)
    for _ in range(60):
        if f_lo < 0.0:
            break
        lo -= 10.0 * kT
        f_lo = f(lo)
    for _ in range(60):
        if f_hi > 0.0:
            break
        hi += max(10.0 * kT, hi - lo)
        f_hi = f(hi)
    if not (f_lo < 0.0 < f_hi):
        raise RuntimeError("sp_bulk_fermi_level_J: could not bracket the bulk Fermi level for "
                           "n_bg={:.4g} m^-3".format(n_bg))
    # xtol = 1e-12 kT: the density error is dn/n ~ dE/kT (non-degenerate) or (3/2) dE/E_F (degenerate),
    # so this keeps n_sep(E_F) = n_bg to ~1e-12 at any degeneracy (a fixed 1e-28 J would allow ~2e-8)
    return float(brentq(f, lo, hi, xtol=1e-12 * kT, maxiter=200))


def bulk_fermi_level_3d_T0_J(n_bg_m3: float, m_kg: float, *, g_s: int = 2, g_v: int = 1,
                             alpha_np_per_eV: float = 0.0) -> float:
    """LEGACY bulk Fermi level [J]: the T = 0 3D relation gamma_F = (hbar^2 / 2m)(6 pi^2 n / (g_s g_v))^(2/3),
    Kane-inverted (alpha E_F^2 + E_F = gamma_F) for alpha > 0. It is NOT the bulk limit of the sub-band
    model (it ignores temperature, and its Kane band is nonparabolic along z too), so a flat-band slab
    filled at this E_F does not hold n_bg (+0.28 % at alpha = 0, -10.4 % at alpha = 0.4 eV^-1 for
    3e26 m^-3, 0.30 m_e, 300 K). Kept bit-identical to the pre-0.16 SchrodingerPoissonCarrier formula as
    the `fermi_level="bulk3d_T0"` option; use sp_bulk_fermi_level_J for the model-consistent level."""
    gamma_F = (HBAR ** 2 / (2.0 * float(m_kg))) * (
        6.0 * np.pi ** 2 * float(n_bg_m3) / (int(g_s) * int(g_v))) ** (2.0 / 3.0)
    if alpha_np_per_eV and float(alpha_np_per_eV) > 0.0:
        a = float(alpha_np_per_eV) / Q                       # eV^-1 -> J^-1
        return (-1.0 + np.sqrt(1.0 + 4.0 * a * gamma_F)) / (2.0 * a)
    return gamma_F


# ---- Gauss's law at a Dirichlet end of the solver's Poisson grid: the ONE face-flux kernel -----------

def sp_face_flux_C_m2(phi_V: np.ndarray, n_m3: np.ndarray, doping_m3: np.ndarray, h_m: float,
                      eps_r: float, *, side: str = "right") -> float:
    """eps0 eps_r dphi/dz [C/m^2] at an END of the uniform grid of solve_self_consistent, from the
    conservative (discrete Gauss law) balance of the solver's own Poisson stencil.

    The interior stencil eps (phi_{i-1} - 2 phi_i + phi_{i+1}) / h^2 = -q (Nd_i - n_i) is the flux balance
    F_{i+1/2} - F_{i-1/2} = -q h (Nd_i - n_i) of the control volume [z_i - h/2, z_i + h/2], with
    F_{i+1/2} = eps (phi_{i+1} - phi_i) / h. Closing the HALF control volume of an end node gives
        right end (z = z[-1]):  F = eps (phi[-1] - phi[-2]) / h - q (h/2) (Nd[-1] - n[-1])
        left end  (z = z[0]):   F = eps (phi[1] - phi[0]) / h + q (h/2) (Nd[0] - n[0])
    (second-order accurate for eps phi' at the face), and summing all control volumes gives the exact
    discrete Gauss law F_right - F_left = q trapz(n - Nd). The returned quantity is eps phi' along +z at
    both ends (the z component of the displacement is its negative).

    Gauss's law at a semiconductor/oxide face without interface charge equates it with the oxide's
    eps_ox eps0 dphi/dz. For a gate at potential Vg beyond an oxide of capacitance C_ox on the RIGHT:
    Vg = psi_s + F_right / C_ox; on the LEFT: Vg = psi_s - F_left / C_ox."""
    phi = np.asarray(phi_V, dtype=np.float64)
    n = np.asarray(n_m3, dtype=np.float64)
    Nd = np.asarray(doping_m3, dtype=np.float64)
    h = float(h_m)
    ee = float(eps_r) * EPS0
    if side == "right":
        return float(ee * (phi[-1] - phi[-2]) / h - Q * (0.5 * h) * (Nd[-1] - n[-1]))
    if side == "left":
        return float(ee * (phi[1] - phi[0]) / h + Q * (0.5 * h) * (Nd[0] - n[0]))
    raise ValueError("sp_face_flux_C_m2: side must be 'right' or 'left', got {!r}".format(side))


def _warm_start_phi(U_init_J: np.ndarray, phi_left_V: float, phi_right_V: float) -> np.ndarray:
    """Initial potential of solve_self_consistent from a warm-start band profile U_init_J [J]:
    phi = -U / q plus the linear end-correction ramp that moves its two ends onto the requested Dirichlet
    values (the Newton updates interior nodes only, so the ends of the initial phi ARE the boundary
    conditions). The interior keeps the warm start's shape; the ends are then written exactly (a + (b - a)
    need not round to b). When the warm start's ends already equal the requested values the ramp is
    identically zero and phi is bit-identical to -U / q."""
    phi = -np.asarray(U_init_J, dtype=np.float64) / Q
    dl = float(phi_left_V) - phi[0]
    dr = float(phi_right_V) - phi[-1]
    if dl != 0.0 or dr != 0.0:
        phi = phi + np.linspace(dl, dr, phi.size)
        phi[0] = phi_left_V
        phi[-1] = phi_right_V
    return phi


@dataclass
class SubbandResult:
    energies_J: np.ndarray       # bound-state energies (sorted), Joules
    psi: np.ndarray              # (n_interior, n_states) normalized so sum |psi|^2 dz = 1
    z_m: np.ndarray              # interior z nodes (Dirichlet ends excluded)
    sheet_density_m2: np.ndarray # per-subband 2D sheet density n_s,i (m^-2)
    density_m3: Optional[np.ndarray] = None   # n(z), filled by density() post-construction
    fermi_level_J: Optional[float] = None     # E_F used, filled by density()
    converged: Optional[bool] = None          # set by solve_self_consistent (None if N/A)
    xc_potential_J: Optional[np.ndarray] = None   # exchange-correlation potential energy on the full grid
                                                  # (solve_self_consistent with xc_potential; else None)


class SchrodingerPoisson1D:
    """Effective-mass 1D solver on a uniform z-grid. ITO defaults: single Gamma valley
    (g_v=1), spin g_s=2, parabolic m* (caveat: ITO is nonparabolic at 1e26-1e27 m^-3 --
    pass a density-dependent m* or accept sub-band-spacing error). The z dispersion is always
    parabolic; `alpha_np_per_eV` adds Kane nonparabolicity to the in-plane sub-band DOS only. The
    Fermi level that makes this model's own flat-band bulk hold a given density is bulk_fermi_level()."""

    def __init__(self, z_m: np.ndarray, m_eff_kg: float, *, T_K: float = T_REF,
                 g_s: int = 2, g_v: int = 1):
        z = np.asarray(z_m, dtype=np.float64)
        if z.ndim != 1 or z.size < 5:
            raise ValueError("z_m must be a 1D grid with >= 5 nodes")
        self.h = float(z[1] - z[0])
        # atol=0: np.allclose's default atol=1e-8 would swamp nm-scale spacings (~1e-10)
        # and silently accept a non-uniform grid.
        if self.h <= 0 or not np.allclose(np.diff(z), self.h, rtol=1e-6, atol=0.0):
            raise ValueError("z_m must be uniformly spaced (BenDaniel-Duke here assumes it)")
        self.z = z
        self.m = float(m_eff_kg)
        self.T = float(T_K)
        self.g_s = int(g_s)
        self.g_v = int(g_v)
        # Anti-silent-failure: a negative/zero/NaN effective mass inverts the kinetic operator
        # (well -> barrier) and returns a meaningless spectrum; T<=0 makes the Fermi occupation
        # 0*inf=NaN. Guard here (the sibling QuantumWell already guards its inputs).
        if not (np.isfinite(self.m) and self.m > 0.0):
            raise ValueError("m_eff_kg must be a finite positive effective mass (kg), got "
                             "{!r}".format(m_eff_kg))
        if not (np.isfinite(self.T) and self.T > 0.0):
            raise ValueError("T_K must be a finite positive temperature (K), got {!r}".format(T_K))
        if self.g_s < 1 or self.g_v < 1:
            raise ValueError("g_s and g_v must be >= 1 (degeneracy factors), got g_s={}, "
                             "g_v={}".format(self.g_s, self.g_v))

    # ---- bulk limit of this model ----
    def bulk_density(self, E_F_J: float, *, alpha_np_per_eV: float = 0.0) -> float:
        """Flat-band bulk density [m^-3] this model fills at E_F_J (see sp_bulk_density_m3): parabolic
        z dispersion with mass self.m, in-plane Kane DOS with alpha_np_per_eV, temperature self.T."""
        return sp_bulk_density_m3(E_F_J, self.m, self.T, g_s=self.g_s, g_v=self.g_v,
                                  alpha_np_per_eV=alpha_np_per_eV)

    def bulk_fermi_level(self, n_bg_m3: float, *, alpha_np_per_eV: float = 0.0) -> float:
        """Fermi level E_F [J] (relative to E_c) at which this model's own flat-band bulk density equals
        n_bg_m3, at the solver temperature (see sp_bulk_fermi_level_J). The z dispersion is parabolic
        with mass self.m (a position-dependent m_eff_z_kg is not part of this bulk limit); with
        alpha_np_per_eV > 0 the in-plane DOS is Kane, as in density(). Use it as E_F_J for a slab that
        must hold n_bg_m3 in its flat-band interior."""
        return sp_bulk_fermi_level_J(n_bg_m3, self.m, self.T, g_s=self.g_s, g_v=self.g_v,
                                     alpha_np_per_eV=alpha_np_per_eV)

    # ---- Schrodinger ----
    def solve_schrodinger(self, U_J: np.ndarray, *,
                            m_eff_z_kg: Optional[np.ndarray] = None,
                            n_states: Optional[int] = None,
                            neumann_left: bool = False,
                            inv_mass_cell_per_kg: Optional[np.ndarray] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bound states of -hbar^2/2 d/dz(1/m dpsi/dz) + U psi = E psi (BenDaniel-Duke; mass at
        half-nodes for position-dependent m). Default: Dirichlet psi=0 at BOTH ends. neumann_left=True:
        a zero-flux (Neumann, dpsi/dz=0) BODY boundary at z=0 -- node 0 is then a free unknown (a
        half-cell BenDaniel-Duke operator with only the right face), so psi need NOT vanish at the body
        (the bulk side is a contact into the semiconductor, not an infinite wall). Removes the ~0.4 nm
        body-side Dirichlet dead layer. Returns (E_J sorted ascending, psi normalized, z_nodes).

        `inv_mass_cell_per_kg` (length N-1): the inverse mass 1/m of every CELL [z_j, z_j+1] instead of node
        masses, for a heterostructure whose material interfaces sit ON grid nodes (an abrupt mass step).
        The operator is then the finite-volume (lumped linear finite-element) form of the BenDaniel-Duke
        operator: row j is -(hbar^2 / 2 h^2) [(psi_j+1 - psi_j) / m_j+1/2 - (psi_j - psi_j-1) / m_j-1/2]
        + U_j psi_j, i.e. the discrete flux (1/m) dpsi/dz is single-valued at every cell face, which is the
        BenDaniel-Duke interface condition (1/m) psi' continuous at a node-centred interface; U_j at an
        interface node is the control-volume average of the two sides (the caller's band_offset). The
        matrix is real symmetric (Hermitian), and the eigenvalues converge at second order in h (gated
        against the analytic finite well in tests/test_sp_barrier_xc_traps.py). Not combinable with
        m_eff_z_kg or neumann_left."""
        from scipy.linalg import eigh_tridiagonal
        U = np.asarray(U_J, dtype=np.float64)
        if inv_mass_cell_per_kg is not None:
            if m_eff_z_kg is not None or neumann_left:
                raise ValueError("solve_schrodinger: inv_mass_cell_per_kg excludes m_eff_z_kg and neumann_left")
            inv_m_half = np.asarray(inv_mass_cell_per_kg, dtype=np.float64)
            if (inv_m_half.shape != (self.z.size - 1,) or np.any(~np.isfinite(inv_m_half))
                    or np.any(inv_m_half <= 0.0)):
                raise ValueError("inv_mass_cell_per_kg must hold N-1 finite positive inverse masses (one "
                                 "per cell)")
        else:
            m_node = (np.full_like(self.z, self.m) if m_eff_z_kg is None
                       else np.asarray(m_eff_z_kg, dtype=np.float64))
            if np.any(~np.isfinite(m_node)) or np.any(m_node <= 0.0):
                raise ValueError("m_eff_z_kg must be finite and > 0 at every node (a non-positive "
                                 "mass inverts the BenDaniel-Duke kinetic operator -> nonsense spectrum)")
            # half-node inverse masses at faces j+1/2 (between node j and j+1), length N-1
            inv_m_half = 2.0 / (m_node[:-1] + m_node[1:])
        c = HBAR ** 2 / (2.0 * self.h ** 2)
        if neumann_left:
            # unknown nodes 0..N-2 (Dirichlet psi=0 only at z=L); node 0 keeps ONLY its right face
            # (zero-flux left boundary) -> diag[0] = c inv_m_{1/2} + U_0.
            Ui = U[:-1]; zi = self.z[:-1]
            diag = np.empty(self.z.size - 1)
            diag[0] = c * inv_m_half[0] + Ui[0]
            diag[1:] = c * (inv_m_half[:-1] + inv_m_half[1:]) + Ui[1:]   # nodes 1..N-2
            offd = -c * inv_m_half[:-1]                          # off (0,1),(1,2),...,(N-3,N-2); length N-2
        else:
            # interior nodes 1..N-2 (Dirichlet at 0 and N-1)
            Ui = U[1:-1]
            zi = self.z[1:-1]
            # diagonal_i = c*(inv_m_{i-1/2} + inv_m_{i+1/2}) + U_i ; offdiag = -c*inv_m_{i+1/2}
            diag = c * (inv_m_half[:-1] + inv_m_half[1:]) + Ui    # length N-2
            offd = -c * inv_m_half[1:-1]                           # length N-3
        # Compute ONLY the lowest n_states eigenpairs (select='i', bisection + inverse iteration)
        # instead of the full spectrum -- ~4x faster for n_states << N and identical to machine
        # precision (eigenvalues bit-equal; eigenvectors equal up to sign, and all downstream use
        # here is sign-invariant: |psi|^2 densities and |<psi_e|psi_h>|^2 overlaps).
        if n_states is not None and int(n_states) < diag.size:
            E, V = eigh_tridiagonal(diag, offd, select="i",
                                    select_range=(0, int(n_states) - 1))
        else:
            E, V = eigh_tridiagonal(diag, offd)
        # normalize columns so sum |psi|^2 * h = 1
        norm = np.sqrt(np.sum(np.abs(V) ** 2, axis=0) * self.h)
        V = V / norm
        return E, V, zi

    # ---- degenerate 2D filling ----
    def density(self, U_J: np.ndarray, E_F_J: float, *,
                 m_eff_z_kg: Optional[np.ndarray] = None,
                 n_states: Optional[int] = None,
                 bound_tol: float = 1e-3,
                 alpha_np_per_eV: float = 0.0,
                 neumann_left: bool = False,
                 inv_mass_cell_per_kg: Optional[np.ndarray] = None) -> SubbandResult:
        """Electron density n(z) from degenerate 2D sub-bands filled to E_F:
            n(z) = sum_i (g_s g_v m* kT / (2 pi hbar^2)) ln(1+exp((E_F-E_i)/kT)) |psi_i(z)|^2
        Unbound states (psi not ~0 at the domain edge) are discarded. Returns a
        SubbandResult; n(z) on the interior grid is `result.density_m3` (attached).

        `alpha_np_per_eV`: Kane in-plane nonparabolicity (eV^-1). The 2D DOS per sub-band
        becomes m*(eps)/(2 pi hbar^2) with the energy-dependent mass m*(eps)=m*0(1+2 alpha
        eps), so the sheet density n_s,i = (g_s g_v m*0/2 pi hbar^2) Int (1+2 alpha eps)
        f(E_i+eps) deps (numerically). alpha=0 reduces to the parabolic kT*ln(1+e^eta).
        Captures the band flattening of a nonparabolic film (heavier DOS mass at high density). `inv_mass_cell_per_kg`: per-cell
        inverse z mass (solve_schrodinger); the in-plane DOS keeps the solver mass self.m."""
        E, psi, zi = self.solve_schrodinger(U_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states,
                                            neumann_left=neumann_left,
                                            inv_mass_cell_per_kg=inv_mass_cell_per_kg)
        # Completeness guard (anti-silent-failure): if n_states truncated the ladder, the HIGHEST
        # SOLVED sub-band is still at/below E_F, so occupied states above it were never solved and
        # n(z) is silently UNDER-counted (verified ~27% at 150 nm / 1e27 m^-3 with n_states=80).
        # Require the top solved state to sit > 5 kT above E_F (Fermi factor < 0.7%). In isolated-
        # well mode the top solved state is an unbound continuum state far above E_F, so this
        # never false-fires there.
        if n_states is not None and E.size and (E_F_J - E[-1]) > -5.0 * KB * self.T:
            warnings.warn(
                "SchrodingerPoisson.density: highest solved sub-band E[{}]={:.4f} eV is not "
                ">5kT above E_F={:.4f} eV (E_F-E_top={:.1f} kT); n_states={} likely truncates "
                "the occupied sub-band ladder and UNDER-counts the density. Increase "
                "n_states.".format(E.size - 1, E[-1] / Q, E_F_J / Q,
                                   (E_F_J - E[-1]) / (KB * self.T), n_states), UnreliableResultWarning, stacklevel=2)
        # keep states that are actually localized (small amplitude at the Dirichlet edge[s]). With a
        # Neumann body (node 0 free), psi need NOT vanish at z=0, so only the RIGHT (gate) edge is a wall.
        if neumann_left:
            edge = np.abs(psi[-1, :]) * np.sqrt(self.h)
        else:
            edge = np.maximum(np.abs(psi[0, :]), np.abs(psi[-1, :])) * np.sqrt(self.h)
        keep = edge < bound_tol
        if not np.any(keep):
            keep = np.zeros(E.size, dtype=bool); keep[0] = True   # keep ground state at least
        E, psi = E[keep], psi[:, keep]
        if alpha_np_per_eV and alpha_np_per_eV > 0.0:
            # Kane in-plane nonparabolicity, m*(eps) = m*0 (1 + 2 alpha eps). The per-sub-band 2D
            # sheet density n_s,i = (g_s g_v m*0 / 2 pi hbar^2) Int (1 + 2 alpha eps) f(E_i+eps) deps
            # is CLOSED-FORM in the complete Fermi-Dirac integrals: pref0 [ kT F_0(eta) + 2 alpha kT^2
            # F_1(eta) ], eta=(E_F-E_i)/kT. alpha=0 reduces to pref0 kT F_0 (the parabolic branch);
            # T->0 it gives pref0 (dE + alpha dE^2) (the validated closed form). Exact + fast (was an
            # 800-pt numerical integral per sub-band), and IDENTICAL to the self-consistent Newton's
            # a-priori density so the converged fill is consistent.
            a = float(alpha_np_per_eV) / Q                        # J^-1
            kT = KB * self.T
            pref0 = self.g_s * self.g_v * self.m / (2.0 * np.pi * HBAR ** 2)  # m^-2 J^-1
            eta = (E_F_J - E) / kT
            ns = pref0 * (kT * _fermi_log(eta) + 2.0 * a * kT ** 2 * _fd1(eta))
        else:
            pref = self.g_s * self.g_v * self.m * KB * self.T / (2.0 * np.pi * HBAR ** 2)  # m^-2
            ns = pref * _fermi_log((E_F_J - E) / (KB * self.T))   # per-subband sheet density (m^-2)
        n_z = (np.abs(psi) ** 2) @ ns                              # (n_interior,) m^-3
        res = SubbandResult(energies_J=E, psi=psi, z_m=zi, sheet_density_m2=ns)
        res.density_m3 = n_z
        res.fermi_level_J = float(E_F_J)
        return res

    # ---- self-consistent Schrodinger-Poisson (Trellakis predictor-corrector) ----
    def solve_self_consistent(self, *, eps_r: float, doping_m3: np.ndarray,
                                E_F_J: float, U_init_J: Optional[np.ndarray] = None,
                                phi_left_V: float = 0.0, phi_right_V: float = 0.0,
                                m_eff_z_kg: Optional[np.ndarray] = None,
                                max_outer: int = 60, tol_V: float = 1e-4,
                                n_states: Optional[int] = None, bound_tol: float = 1e-3,
                                relax: float = 0.7, alpha_np_per_eV: float = 0.0,
                                neumann_left: bool = False, bulk_buffer_m: float = 0.0,
                                verbose: bool = False,
                                eps_cell_r: Optional[np.ndarray] = None,
                                inv_mass_cell_per_kg: Optional[np.ndarray] = None,
                                band_offset_J: Optional[np.ndarray] = None,
                                node_sheet_charges=None,
                                xc_potential=None,
                                xc_density_init_m3: Optional[np.ndarray] = None):
        """Self-consistent solve on the FULL grid (Dirichlet phi at both ends).
        Poisson: d/dz(eps eps0 dphi/dz) = -q (N_D+ - n), electron PE U = -q*phi + U_band.
        The Trellakis predictor-corrector folds an a-priori quantum density that rigidly
        shifts each sub-band floor with the local potential into a NONLINEAR Poisson
        Newton solve (exact Jacobian = Fermi function), then re-solves Schrodinger --
        far more robust than naive Picard. Returns (phi_V, n_m3, SubbandResult).
        `doping_m3` is the ionized net donor profile N_D+ (m^-3) on the full grid.

        `bound_tol`: edge-amplitude threshold for keeping a state. The default 1e-3
        rejects unbound states (isolated quantum well). For a DEGENERATE-bulk slab
        (e.g. ITO, E_F far above many sub-bands) pass a LARGE value (e.g. 1e9) so ALL
        sub-bands up to E_F are kept -- they carry the bulk continuum, and rejecting
        them collapses the bulk density to ~0 (use n_states >= the # of sub-bands < E_F).

        `relax`: outer-loop under-relaxation factor in (0, 1]. The potential update is
        damped: phi <- last_phi + relax*(phi_new - last_phi); relax=1 recovers the undamped
        update. CAVEAT (audit SP-RELAX-2): damping phi does NOT cure the kept-state-SET churn
        across the bound_tol edge threshold, so the isolated-well default mode (small
        bound_tol) is NOT expected to converge at any relax -- it limit-cycles and stays
        max_outer-parity-sensitive. For the degenerate-bulk slab use a LARGE bound_tol (slab
        mode): it keeps all sub-bands, has no set churn, and converges; `.converged` flags
        the isolated-well non-convergence.

        `alpha_np_per_eV`: Kane in-plane nonparabolicity (eV^-1). When > 0 the Trellakis inner
        Newton's a-priori density AND its Jacobian use the nonparabolic 2D DOS (m*(eps)=m*0(1+2
        alpha eps)) in the SAME closed form as density() -- so the self-consistent potential is
        nonparabolic-CONSISTENT (not a post-hoc re-fill of a parabolic potential). alpha=0 is the
        parabolic solve, byte-identical to before.

        `neumann_left`: a zero-flux (Neumann) BODY boundary at z=0 for the SCHRODINGER only (psi need
        not vanish at the body -> removes the ~0.4 nm Dirichlet dead layer; the body is a contact into
        the bulk, not an infinite wall). The POISSON stays Dirichlet (phi[0]=phi_left_V pins the bulk
        reference, so the body density is the bulk n_bg, not a floating value); the freed body node's
        density is filled from the Neumann eigenstates. Default False = Dirichlet psi at both ends,
        byte-identical to before.

        `bulk_buffer_m`: an OPEN / TRANSPARENT body boundary (replaces the hard Neumann wall). A
        zero-flux Neumann body forces a wavefunction ANTINODE at z=0 (density PILE-UP), and a Dirichlet
        body forces a node (DEAD LAYER) -- neither recovers the bulk smoothly. With bulk_buffer_m > 0 a
        field-free bulk buffer of that length (same grid spacing, doping = the body doping, flat band) is
        prepended to the body side and the solve runs on the EXTENDED grid (Dirichlet at the FAR buffer
        end = the bulk reference). The physical body node is then INTERIOR, far from any wall, so its
        density relaxes to the bulk n_bg with NO boundary layer (the buffer mimics the semi-infinite
        bulk -- the task's "bulk buffer" open BC). phi and n are returned on the ORIGINAL grid (buffer
        stripped); the SubbandResult spans the extended grid. Mutually exclusive with neumann_left (the
        buffer IS the open body boundary). Use ~5-15 nm for a degenerate accumulation layer.

        `U_init_J`: optional warm-start electron potential energy on the grid (e.g. -q phi of a previous
        bias). Its interior shape seeds the iteration; its two ends are moved onto (phi_left_V,
        phi_right_V) by a linear correction ramp, so the requested boundary potentials always hold
        (a warm start whose ends already match is used unchanged).

        Returns (phi_V, n_m3, SubbandResult). The result carries `.converged` (bool):
        if the outer loop did not reach tol_V in max_outer iterations, `.converged` is
        False and a warning is emitted -- the returned (phi, n) is NOT trustworthy and
        is sensitive to max_outer (the kept-state set can churn between iterations). Callers should check it.

        Heterostructure / many-body options (v0.16; every one None by default = the solve above, bit for
        bit). They need Dirichlet psi at both ends (not neumann_left / bulk_buffer_m):
          * `eps_cell_r` (N-1): relative permittivity per CELL; Poisson becomes the control-volume form
            (eps_j+1/2 (phi_j+1 - phi_j) - eps_j-1/2 (phi_j - phi_j-1)) / h^2 = -rho_j (a node on a dielectric
            interface takes one flux from each side). `doping_m3` is then the control-volume average (a
            film/oxide interface node carries half the film doping).
          * `inv_mass_cell_per_kg` (N-1): per-cell inverse z mass (solve_schrodinger); `band_offset_J` (N):
            conduction-band offset U_band on the nodes (control-volume average at an interface node).
          * `node_sheet_charges`: sequence of (node_index, fn) with fn(phi_node_V) -> (Q [C/m^2], dQ/dphi
            [F/m^2]): a sheet charge on that interior node (e.g. a fixed interface charge, or interface
            traps whose occupancy follows the local electrostatic potential); it enters the Poisson row as
            Q / h, with dQ/dphi / h in the inner Newton Jacobian (exact derivative).
          * `xc_potential`: fn(n_m3) -> (V_xc [J], dV_xc/dn [J m^3]), a LOCAL density functional added to
            the Schrodinger potential energy, U = -q phi + U_band + V_xc(n). The predictor-corrector then
            carries the local band shift s(z) as a second unknown: n~ = sum_i |psi_i|^2 n_s(E_i + s),
            s = -q (phi - phi_k) + V_xc(n~) - V_xc(n_k); eliminating ds per node gives the tridiagonal
            Newton system with dn/dphi -> dn/dphi / (1 + V_xc' dn/dphi / q), so the inner Newton stays
            quadratic with the exchange-correlation derivative included. The density entering V_xc is
            under-relaxed like phi and convergence also requires max |dV_xc| / q < tol_V.
            `xc_density_init_m3` (N): warm-start density for V_xc (default: the fill of the initial phi
            without xc). The returned SubbandResult carries `xc_potential_J`."""
        general = (eps_cell_r is not None or inv_mass_cell_per_kg is not None or band_offset_J is not None
                   or node_sheet_charges is not None or xc_potential is not None)
        if general and (neumann_left or (bulk_buffer_m and bulk_buffer_m > 0.0)):
            raise ValueError("solve_self_consistent: eps_cell_r / inv_mass_cell_per_kg / band_offset_J / "
                             "node_sheet_charges / xc_potential need Dirichlet psi at both ends (not "
                             "neumann_left or bulk_buffer_m)")
        if bulk_buffer_m and bulk_buffer_m > 0.0:
            # OPEN / TRANSPARENT body BC: prepend a field-free bulk buffer on the body side, solve on the
            # extended grid (Dirichlet far end = bulk reference), and strip the buffer from the returned
            # phi/n. The physical body node becomes interior -> recovers n_bg with no pile-up/dead-layer.
            if neumann_left:
                raise ValueError("solve_self_consistent: bulk_buffer_m and neumann_left are mutually "
                                  "exclusive (the bulk buffer IS the open body boundary; the Neumann "
                                  "wall is the pile-up BC it replaces).")
            hb = self.h
            n_buf = max(1, int(round(float(bulk_buffer_m) / hb)))
            z_buf = self.z[0] - hb * np.arange(n_buf, 0, -1)       # n_buf nodes below z[0], spacing h
            z_ext = np.concatenate([z_buf, self.z])
            Nd0 = np.asarray(doping_m3, dtype=np.float64)
            Nd_ext = np.concatenate([np.full(n_buf, Nd0[0]), Nd0])  # field-free bulk doping in the buffer
            m_ext = None
            if m_eff_z_kg is not None:
                me = np.asarray(m_eff_z_kg, dtype=np.float64)
                m_ext = np.concatenate([np.full(n_buf, me[0]), me])
            U_ext = None
            if U_init_J is not None:
                Ui = np.asarray(U_init_J, dtype=np.float64)
                U_ext = np.concatenate([np.full(n_buf, -Q * phi_left_V), Ui])
            ext = SchrodingerPoisson1D(z_ext, self.m, T_K=self.T, g_s=self.g_s, g_v=self.g_v)
            phi_e, n_e, res_e = ext.solve_self_consistent(
                eps_r=eps_r, doping_m3=Nd_ext, E_F_J=E_F_J, U_init_J=U_ext,
                phi_left_V=phi_left_V, phi_right_V=phi_right_V, m_eff_z_kg=m_ext,
                max_outer=max_outer, tol_V=tol_V, n_states=n_states, bound_tol=bound_tol,
                relax=relax, alpha_np_per_eV=alpha_np_per_eV, neumann_left=False,
                bulk_buffer_m=0.0, verbose=verbose)
            return phi_e[n_buf:], n_e[n_buf:], res_e   # strip the buffer -> the ORIGINAL grid
        N = self.z.size
        h = self.h
        ee = eps_r * EPS0
        lo = 0 if neumann_left else 1                         # first UNKNOWN node (0 = Neumann body)
        Nd = np.asarray(doping_m3, dtype=np.float64)
        U_band = np.zeros(N)                                  # band-edge offset (0 for single material)
        ec_half = None                                        # eps0 eps per cell (eps_cell_r) or None
        sheets = []
        if general:
            if eps_cell_r is not None:
                ec_half = EPS0 * np.asarray(eps_cell_r, dtype=np.float64)
                if ec_half.shape != (N - 1,) or np.any(~np.isfinite(ec_half)) or np.any(ec_half <= 0.0):
                    raise ValueError("solve_self_consistent: eps_cell_r must hold N-1 finite positive values")
            if band_offset_J is not None:
                U_band = np.asarray(band_offset_J, dtype=np.float64).copy()
                if U_band.shape != (N,) or np.any(~np.isfinite(U_band)):
                    raise ValueError("solve_self_consistent: band_offset_J must hold N finite values")
            for idx, fn in (node_sheet_charges or ()):
                if not (1 <= int(idx) <= N - 2) or not callable(fn):
                    raise ValueError("solve_self_consistent: node_sheet_charges entries are (interior node "
                                     "index 1..N-2, callable), got ({!r}, {!r})".format(idx, fn))
                sheets.append((int(idx), fn))
        # initial potential: linear ramp between the Dirichlet values; a warm start keeps its interior
        # shape and has its ends moved onto (phi_left_V, phi_right_V) by a linear correction ramp (the
        # Newton below updates interior nodes only, so the initial ends ARE the boundary conditions)
        if U_init_J is not None and np.asarray(U_init_J).shape != (N,):
            raise ValueError("solve_self_consistent: U_init_J must have the grid's shape ({},), got "
                             "{}".format(N, np.asarray(U_init_J).shape))
        phi = (np.linspace(phi_left_V, phi_right_V, N) if U_init_J is None
                else _warm_start_phi(U_init_J, phi_left_V, phi_right_V))
        kT = KB * self.T

        # (dead laplacian_matrix helper removed -- audit S1-17: the Newton loop builds its
        # banded Jacobian directly and never read it)
        last_phi = phi.copy()
        result = None
        dV = float("inf")          # guard: max_outer<1 -> loop body never runs (report non-converged)
        dVxc = 0.0                 # max |dV_xc| / q of the last outer iteration (xc_potential only)
        inner_ok = True            # tracks whether the LAST outer iteration's inner Newton converged
        a_np = float(alpha_np_per_eV) / Q if (alpha_np_per_eV and alpha_np_per_eV > 0.0) else 0.0  # J^-1
        mkw = {} if inv_mass_cell_per_kg is None else {"inv_mass_cell_per_kg": inv_mass_cell_per_kg}
        n_xc = None
        Vx_full = None
        if xc_potential is not None:
            if xc_density_init_m3 is not None:
                n_xc = np.array(xc_density_init_m3, dtype=np.float64)
                if n_xc.shape != (N,):
                    raise ValueError("solve_self_consistent: xc_density_init_m3 must have shape ({},)".format(N))
            else:
                r0 = self.density(-Q * phi + U_band, E_F_J, n_states=n_states, bound_tol=bound_tol,
                                  alpha_np_per_eV=alpha_np_per_eV, **mkw)
                n_xc = np.zeros(N)
                n_xc[1:-1] = r0.density_m3
        for it in range(max_outer):
            U = -Q * phi + U_band
            if n_xc is not None:
                Vx_full = np.asarray(xc_potential(n_xc)[0], dtype=np.float64)
                U = U + Vx_full
            res = self.density(U, E_F_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states, bound_tol=bound_tol,
                               alpha_np_per_eV=alpha_np_per_eV, neumann_left=neumann_left, **mkw)
            E_k, psi_k = res.energies_J.copy(), res.psi.copy()
            ns_pref = self.g_s * self.g_v * self.m * kT / (2.0 * np.pi * HBAR ** 2)
            phi_k = phi.copy()

            # --- nonlinear Poisson Newton (predictor-corrector): n_tilde(phi) shifts each
            #     bound sub-band floor rigidly with the local potential change ---
            from scipy.linalg import solve_banded
            phi_in = phi.copy()
            inner_ok = True
            # POISSON stays Dirichlet at the body (phi[0]=phi_left pins the bulk reference, so n[0]=n_bg);
            # a Neumann body only frees the SCHRODINGER psi (no dead layer), so the eigenstates carry an
            # extra body row 0 -- use its INTERIOR rows (1..N-2) for the Poisson a-priori density.
            pk = psi_k[1:] if neumann_left else psi_k          # interior rows (drop the body node)
            pk2 = np.abs(pk) ** 2                              # audit S6-15: |psi|^2 is invariant

            def _occupancy(arg):
                F0 = _fermi_log(arg)                           # complete FD order 0 (n_int, n_states)
                f = 1.0 / (1.0 + np.exp(-np.clip(arg, -700, 700)))   # Fermi function = dF_0/dx
                if a_np > 0.0:
                    # nonparabolic (Kane) a-priori density + Jacobian, the SAME closed form as
                    # density(): n_s = pref0(kT F_0 + 2a kT^2 F_1); dn_s/dphi = q pref0(f + 2a kT F_0)
                    # (ns_pref = pref0 kT). Reduces to the parabolic branch below at a_np=0.
                    F1 = _fd1(arg)
                    occ = ns_pref * (F0 + 2.0 * a_np * kT * F1)
                    dns_dphi = (ns_pref * Q / kT) * (f + 2.0 * a_np * kT * F0)
                else:
                    occ = ns_pref * F0                         # (n_int, n_states)
                    dns_dphi = (ns_pref * Q / kT) * f          # d(arg)/dphi = q/kT
                return np.sum(pk2 * occ, axis=1), np.sum(pk2 * dns_dphi, axis=1)

            if n_xc is not None:
                s_in = np.zeros(N - 2)                         # local band shift relative to E_k (J)
                Vx_k = Vx_full[1:-1]
            for _newton in range(40):                          # across the inner Newton iterations
                if n_xc is None:
                    shift = -Q * (phi_in - phi_k)              # E_i(phi) ~ E_i^k + (U-U^k); U=-q phi
                    arg = (E_F_J - (E_k[None, :] + shift[1:-1, None])) / kT   # interior nodes 1..N-2
                else:
                    arg = (E_F_J - (E_k[None, :] + s_in[:, None])) / kT
                n_int, dn_dphi = _occupancy(arg)
                # residual of eps0 eps phi'' = -q(Nd - n) on interior nodes
                ab = np.zeros((3, N - 2))
                if ec_half is None:
                    lap = (phi_in[:-2] - 2.0 * phi_in[1:-1] + phi_in[2:]) / h ** 2
                    R = ee * lap + Q * (Nd[1:-1] - n_int)
                    # Jacobian: ee * D2 - q * dn/dphi  (D2 tin/diag)
                    ab[0, 1:] = ee / h ** 2                      # super-diagonal
                    ab[2, :-1] = ee / h ** 2                     # sub-diagonal
                    diag = -2.0 * ee / h ** 2                    # diagonal of eps D2
                else:
                    fl = ec_half * (phi_in[1:] - phi_in[:-1]) / h ** 2       # cell fluxes / h
                    R = (fl[1:] - fl[:-1]) + Q * (Nd[1:-1] - n_int)
                    ab[0, 1:] = ec_half[1:-1] / h ** 2
                    ab[2, :-1] = ec_half[1:-1] / h ** 2
                    diag = -(ec_half[:-1] + ec_half[1:]) / h ** 2
                if sheets:
                    diag = diag + np.zeros(N - 2)
                    for idx, fn in sheets:
                        q_s, dq_s = fn(float(phi_in[idx]))
                        R[idx - 1] += float(q_s) / h
                        diag[idx - 1] += float(dq_s) / h
                if n_xc is None:
                    ab[1, :] = diag - Q * dn_dphi                # diagonal (dR/dphi)
                    dphi = solve_banded((1, 1), ab, -R)
                    phi_in[1:-1] += dphi
                    if np.max(np.abs(dphi)) < 1e-9:
                        break
                else:
                    Vn, dVn = xc_potential(n_int)
                    r2 = s_in + Q * (phi_in[1:-1] - phi_k[1:-1]) - (np.asarray(Vn) - Vx_k)
                    den = np.maximum(1.0 + np.asarray(dVn) * dn_dphi / Q, 0.05)
                    dnx = dn_dphi / den
                    ab[1, :] = diag - Q * dnx
                    dphi = solve_banded((1, 1), ab, -R + dnx * r2)
                    ds = (-r2 - Q * dphi) / den
                    phi_in[1:-1] += dphi
                    s_in += ds
                    if np.max(np.abs(dphi)) < 1e-9 and np.max(np.abs(ds)) < 1e-9 * Q:
                        break
            else:
                inner_ok = False     # inner nonlinear-Poisson Newton never reached 1e-9 in 40 its
            # outer under-relaxation: damp the kept-state-set churn (audit SP-1)
            phi = last_phi + float(relax) * (phi_in - last_phi)
            if n_xc is not None:
                n_pred = _occupancy((E_F_J - (E_k[None, :] + s_in[:, None])) / kT)[0]
                n_new = n_xc.copy()
                n_new[1:-1] = n_xc[1:-1] + float(relax) * (n_pred - n_xc[1:-1])
                dVxc = float(np.max(np.abs(np.asarray(xc_potential(n_new)[0])[1:-1] - Vx_k))) / Q
                n_xc = n_new
            result = res
            dV = float(np.max(np.abs(phi - last_phi)))
            if verbose:
                print("[t] SP outer {:2d}: max|dphi|={:.3e} V, n_states_bound={}".format(
                    it, dV, E_k.size), flush=True)
            last_phi = phi.copy()
            if dV < tol_V and dVxc < tol_V:
                break
        converged = (dV < tol_V) and (dVxc < tol_V) and inner_ok
        if not converged:
            why = []
            if dV >= tol_V:
                why.append("outer max|dphi|={:.3e} V >= tol_V={:.1e}".format(dV, tol_V))
            if dVxc >= tol_V:
                why.append("outer max|dV_xc|/q={:.3e} V >= tol_V={:.1e}".format(dVxc, tol_V))
            if not inner_ok:
                why.append("the inner nonlinear-Poisson Newton did not reach 1e-9 in 40 steps")
            warnings.warn(
                "SchrodingerPoisson.solve_self_consistent did NOT converge after {} outer "
                "iterations ({}). The returned (phi, n) is unreliable and sensitive to "
                "max_outer; try a smaller `relax`, a larger bound_tol (slab mode), or more "
                "iterations.".format(max_outer, "; ".join(why)), UnreliableResultWarning, stacklevel=2)
        U = -Q * phi + U_band
        if n_xc is not None:
            Vx_full = np.asarray(xc_potential(n_xc)[0], dtype=np.float64)
            U = U + Vx_full
        result = self.density(U, E_F_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states, bound_tol=bound_tol,
                              alpha_np_per_eV=alpha_np_per_eV, neumann_left=neumann_left, **mkw)
        result.converged = converged
        if n_xc is not None:
            result.xc_potential_J = Vx_full
        n_full = np.zeros(N)
        n_full[lo:-1] = result.density_m3                      # nodes lo..N-2 (lo=0 keeps the body node)
        return phi, n_full, result
