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

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

HBAR = 1.054571817e-34       # J s
M_E = 9.1093837015e-31       # kg
KB = 1.380649e-23            # J/K
Q = 1.602176634e-19          # C
EPS0 = 8.8541878128e-12      # F/m


def _fermi_log(x: np.ndarray) -> np.ndarray:
    """ln(1+exp(x)) with an overflow-safe large-x branch (-> x for x >> 1). The 2D
    degenerate occupation integral; for strongly degenerate ITO x can be >> 1."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    big = x > 40.0
    out[big] = x[big]
    out[~big] = np.log1p(np.exp(np.clip(x[~big], -700.0, 40.0)))
    return out


@dataclass
class SubbandResult:
    energies_J: np.ndarray       # bound-state energies (sorted), Joules
    psi: np.ndarray              # (n_interior, n_states) normalized so sum |psi|^2 dz = 1
    z_m: np.ndarray              # interior z nodes (Dirichlet ends excluded)
    sheet_density_m2: np.ndarray # per-subband 2D sheet density n_s,i (m^-2)


class SchrodingerPoisson1D:
    """Effective-mass 1D solver on a uniform z-grid. ITO defaults: single Gamma valley
    (g_v=1), spin g_s=2, parabolic m* (caveat: ITO is nonparabolic at 1e26-1e27 m^-3 --
    pass a density-dependent m* or accept sub-band-spacing error)."""

    def __init__(self, z_m: np.ndarray, m_eff_kg: float, *, T_K: float = 300.0,
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

    # ---- Schrodinger ----
    def solve_schrodinger(self, U_J: np.ndarray, *,
                            m_eff_z_kg: Optional[np.ndarray] = None,
                            n_states: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bound states of -hbar^2/2 d/dz(1/m dpsi/dz) + U psi = E psi with Dirichlet
        psi=0 at both ends (BenDaniel-Duke; mass at half-nodes for position-dependent
        m). Returns (E_J sorted ascending, psi interior-normalized, z_interior)."""
        from scipy.linalg import eigh_tridiagonal
        U = np.asarray(U_J, dtype=np.float64)
        m_node = (np.full_like(self.z, self.m) if m_eff_z_kg is None
                   else np.asarray(m_eff_z_kg, dtype=np.float64))
        # interior nodes 1..N-2 (Dirichlet at 0 and N-1)
        Ui = U[1:-1]
        zi = self.z[1:-1]
        # half-node inverse masses
        inv_m_half = 2.0 / (m_node[:-1] + m_node[1:])         # at i+1/2, length N-1
        c = HBAR ** 2 / (2.0 * self.h ** 2)
        # diagonal_i = c*(inv_m_{i-1/2} + inv_m_{i+1/2}) + U_i ; offdiag = -c*inv_m_{i+1/2}
        diag = c * (inv_m_half[:-1] + inv_m_half[1:]) + Ui    # length N-2
        offd = -c * inv_m_half[1:-1]                           # length N-3
        E, V = eigh_tridiagonal(diag, offd)
        if n_states is not None:
            E, V = E[:n_states], V[:, :n_states]
        # normalize columns so sum |psi|^2 * h = 1
        norm = np.sqrt(np.sum(np.abs(V) ** 2, axis=0) * self.h)
        V = V / norm
        return E, V, zi

    # ---- degenerate 2D filling ----
    def density(self, U_J: np.ndarray, E_F_J: float, *,
                 m_eff_z_kg: Optional[np.ndarray] = None,
                 n_states: Optional[int] = None,
                 bound_tol: float = 1e-3,
                 alpha_np_per_eV: float = 0.0) -> SubbandResult:
        """Electron density n(z) from degenerate 2D sub-bands filled to E_F:
            n(z) = sum_i (g_s g_v m* kT / (2 pi hbar^2)) ln(1+exp((E_F-E_i)/kT)) |psi_i(z)|^2
        Unbound states (psi not ~0 at the domain edge) are discarded. Returns a
        SubbandResult; n(z) on the interior grid is `result.density_m3` (attached).

        `alpha_np_per_eV`: Kane in-plane nonparabolicity (eV^-1). The 2D DOS per sub-band
        becomes m*(eps)/(2 pi hbar^2) with the energy-dependent mass m*(eps)=m*0(1+2 alpha
        eps), so the sheet density n_s,i = (g_s g_v m*0/2 pi hbar^2) Int (1+2 alpha eps)
        f(E_i+eps) deps (numerically). alpha=0 reduces to the parabolic kT*ln(1+e^eta).
        Captures ITO's band flattening (heavier DOS mass at high density)."""
        E, psi, zi = self.solve_schrodinger(U_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states)
        # keep states that are actually localized (small amplitude at both edges)
        edge = np.maximum(np.abs(psi[0, :]), np.abs(psi[-1, :])) * np.sqrt(self.h)
        keep = edge < bound_tol
        if not np.any(keep):
            keep = np.zeros(E.size, dtype=bool); keep[0] = True   # keep ground state at least
        E, psi = E[keep], psi[:, keep]
        if alpha_np_per_eV and alpha_np_per_eV > 0.0:
            a = float(alpha_np_per_eV) / Q                        # J^-1
            kT = KB * self.T
            pref0 = self.g_s * self.g_v * self.m / (2.0 * np.pi * HBAR ** 2)  # m^-2 J^-1
            eg = np.linspace(0.0, max(0.0, float(E_F_J - float(E.min()))) + 30.0 * kT, 800)
            dos = 1.0 + 2.0 * a * eg                              # m*(eps)/m*0 (nonparabolic DOS)
            de = np.diff(eg)
            ns = np.empty(E.size)
            for i in range(E.size):
                occ = 1.0 / (1.0 + np.exp(np.clip((E[i] + eg - E_F_J) / kT, -700.0, 700.0)))
                g = dos * occ
                ns[i] = pref0 * float(np.sum(0.5 * (g[:-1] + g[1:]) * de))
        else:
            pref = self.g_s * self.g_v * self.m * KB * self.T / (2.0 * np.pi * HBAR ** 2)  # m^-2
            ns = pref * _fermi_log((E_F_J - E) / (KB * self.T))   # per-subband sheet density (m^-2)
        n_z = (np.abs(psi) ** 2) @ ns                              # (n_interior,) m^-3
        res = SubbandResult(energies_J=E, psi=psi, z_m=zi, sheet_density_m2=ns)
        res.density_m3 = n_z                                       # type: ignore[attr-defined]
        res.fermi_level_J = float(E_F_J)                           # type: ignore[attr-defined]
        return res

    # ---- self-consistent Schrodinger-Poisson (Trellakis predictor-corrector) ----
    def solve_self_consistent(self, *, eps_r: float, doping_m3: np.ndarray,
                                E_F_J: float, U_init_J: Optional[np.ndarray] = None,
                                phi_left_V: float = 0.0, phi_right_V: float = 0.0,
                                m_eff_z_kg: Optional[np.ndarray] = None,
                                max_outer: int = 60, tol_V: float = 1e-4,
                                n_states: Optional[int] = None, bound_tol: float = 1e-3,
                                verbose: bool = False):
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
        them collapses the bulk density to ~0 (use n_states >= the # of sub-bands < E_F)."""
        N = self.z.size
        h = self.h
        ee = eps_r * EPS0
        Nd = np.asarray(doping_m3, dtype=np.float64)
        U_band = np.zeros(N)                                  # band-edge offset (0 for single material)
        # initial potential: linear ramp between the Dirichlet values
        phi = (np.linspace(phi_left_V, phi_right_V, N) if U_init_J is None
                else -np.asarray(U_init_J) / Q)
        kT = KB * self.T

        def laplacian_matrix():
            # second-difference operator on interior nodes (uniform grid), Dirichlet ends
            import numpy as _np
            n = N - 2
            main = -2.0 * _np.ones(n)
            off = _np.ones(n - 1)
            return main, off

        main, off = laplacian_matrix()
        last_phi = phi.copy()
        result = None
        for it in range(max_outer):
            U = -Q * phi + U_band
            res = self.density(U, E_F_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states, bound_tol=bound_tol)
            E_k, psi_k = res.energies_J.copy(), res.psi.copy()
            ns_pref = self.g_s * self.g_v * self.m * kT / (2.0 * np.pi * HBAR ** 2)
            phi_k = phi.copy()

            # --- nonlinear Poisson Newton (predictor-corrector): n_tilde(phi) shifts each
            #     bound sub-band floor rigidly with the local potential change ---
            from scipy.linalg import solve_banded
            phi_in = phi.copy()
            for _newton in range(40):
                Uloc = -Q * phi_in + U_band
                # a-priori quantum density with potential-shifted sub-band energies
                shift = -Q * (phi_in - phi_k)                  # E_i(phi) ~ E_i^k + (U-U^k); U=-q phi
                arg = (E_F_J - (E_k[None, :] + shift[1:-1, None])) / kT
                occ = ns_pref * _fermi_log(arg)                # (n_int, n_states)
                n_int = np.sum((np.abs(psi_k) ** 2) * occ, axis=1)   # (n_int,)
                # d n / d phi  (chain rule through the Fermi function): d(arg)/dphi = q/kT
                f = 1.0 / (1.0 + np.exp(-np.clip(arg, -700, 700)))   # Fermi function = d/dx ln(1+e^x)
                dn_dphi = np.sum((np.abs(psi_k) ** 2) * (ns_pref * (Q / kT)) * f, axis=1)
                # residual of eps0 eps phi'' = -q(Nd - n)  on interior nodes
                lap = (phi_in[:-2] - 2.0 * phi_in[1:-1] + phi_in[2:]) / h ** 2
                R = ee * lap + Q * (Nd[1:-1] - n_int)
                # Jacobian: ee * D2 - q * dn/dphi  (D2 tin/diag)
                ab = np.zeros((3, N - 2))
                ab[0, 1:] = ee / h ** 2                          # super-diagonal
                ab[2, :-1] = ee / h ** 2                         # sub-diagonal
                ab[1, :] = -2.0 * ee / h ** 2 - Q * dn_dphi      # diagonal (dR/dphi)
                dphi = solve_banded((1, 1), ab, -R)
                phi_in[1:-1] += dphi
                if np.max(np.abs(dphi)) < 1e-9:
                    break
            phi = phi_in
            result = res
            dV = float(np.max(np.abs(phi - last_phi)))
            if verbose:
                print("[t] SP outer {:2d}: max|dphi|={:.3e} V, n_states_bound={}".format(
                    it, dV, E_k.size), flush=True)
            last_phi = phi.copy()
            if dV < tol_V:
                break
        U = -Q * phi + U_band
        result = self.density(U, E_F_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states, bound_tol=bound_tol)
        n_full = np.zeros(N)
        n_full[1:-1] = result.density_m3   # type: ignore[attr-defined]
        return phi, n_full, result
