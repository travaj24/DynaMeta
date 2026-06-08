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


def _fd1(x: np.ndarray) -> np.ndarray:
    """Complete Fermi-Dirac integral of order 1: F_1(x) = Int_0^inf t/(1+e^(t-x)) dt = -Li_2(-e^x).
    Closed form via the dilogarithm: Li_2(w) = scipy.special.spence(1-w), so F_1(x) = -spence(1+e^x);
    the large-x branch uses F_1(x) -> x^2/2 + pi^2/6 (Sommerfeld). Verified vs direct quadrature to
    ~1e-14 and dF_1/dx = F_0 to ~1e-9. Used by the nonparabolic (Kane) 2D sub-band sheet density:
    the m*(eps)=m*0(1+2 alpha eps) DOS gives n_s = pref0 (kT F_0 + 2 alpha kT^2 F_1)."""
    from scipy.special import spence
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    big = x > 40.0
    out[big] = 0.5 * x[big] ** 2 + (np.pi ** 2) / 6.0
    out[~big] = -spence(1.0 + np.exp(np.clip(x[~big], -700.0, 40.0)))
    return out


@dataclass
class SubbandResult:
    energies_J: np.ndarray       # bound-state energies (sorted), Joules
    psi: np.ndarray              # (n_interior, n_states) normalized so sum |psi|^2 dz = 1
    z_m: np.ndarray              # interior z nodes (Dirichlet ends excluded)
    sheet_density_m2: np.ndarray # per-subband 2D sheet density n_s,i (m^-2)
    density_m3: Optional[np.ndarray] = None   # n(z), filled by density() post-construction
    fermi_level_J: Optional[float] = None     # E_F used, filled by density()
    converged: Optional[bool] = None          # set by solve_self_consistent (None if N/A)


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

    # ---- Schrodinger ----
    def solve_schrodinger(self, U_J: np.ndarray, *,
                            m_eff_z_kg: Optional[np.ndarray] = None,
                            n_states: Optional[int] = None,
                            neumann_left: bool = False
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bound states of -hbar^2/2 d/dz(1/m dpsi/dz) + U psi = E psi (BenDaniel-Duke; mass at
        half-nodes for position-dependent m). Default: Dirichlet psi=0 at BOTH ends. neumann_left=True:
        a zero-flux (Neumann, dpsi/dz=0) BODY boundary at z=0 -- node 0 is then a free unknown (a
        half-cell BenDaniel-Duke operator with only the right face), so psi need NOT vanish at the body
        (the bulk side is a contact into the semiconductor, not an infinite wall). Removes the ~0.4 nm
        body-side Dirichlet dead layer. Returns (E_J sorted ascending, psi normalized, z_nodes)."""
        from scipy.linalg import eigh_tridiagonal
        U = np.asarray(U_J, dtype=np.float64)
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
                 neumann_left: bool = False) -> SubbandResult:
        """Electron density n(z) from degenerate 2D sub-bands filled to E_F:
            n(z) = sum_i (g_s g_v m* kT / (2 pi hbar^2)) ln(1+exp((E_F-E_i)/kT)) |psi_i(z)|^2
        Unbound states (psi not ~0 at the domain edge) are discarded. Returns a
        SubbandResult; n(z) on the interior grid is `result.density_m3` (attached).

        `alpha_np_per_eV`: Kane in-plane nonparabolicity (eV^-1). The 2D DOS per sub-band
        becomes m*(eps)/(2 pi hbar^2) with the energy-dependent mass m*(eps)=m*0(1+2 alpha
        eps), so the sheet density n_s,i = (g_s g_v m*0/2 pi hbar^2) Int (1+2 alpha eps)
        f(E_i+eps) deps (numerically). alpha=0 reduces to the parabolic kT*ln(1+e^eta).
        Captures ITO's band flattening (heavier DOS mass at high density)."""
        E, psi, zi = self.solve_schrodinger(U_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states,
                                            neumann_left=neumann_left)
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
                                   (E_F_J - E[-1]) / (KB * self.T), n_states), stacklevel=2)
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

        Returns (phi_V, n_m3, SubbandResult). The result carries `.converged` (bool):
        if the outer loop did not reach tol_V in max_outer iterations, `.converged` is
        False and a warning is emitted -- the returned (phi, n) is NOT trustworthy and
        is sensitive to max_outer (audit SP-1). Callers should check it."""
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
        # initial potential: linear ramp between the Dirichlet values
        phi = (np.linspace(phi_left_V, phi_right_V, N) if U_init_J is None
                else -np.asarray(U_init_J) / Q)
        kT = KB * self.T

        def laplacian_matrix():
            # second-difference operator on interior nodes (uniform grid), Dirichlet ends
            n = N - 2
            main = -2.0 * np.ones(n)
            off = np.ones(n - 1)
            return main, off

        main, off = laplacian_matrix()
        last_phi = phi.copy()
        result = None
        dV = float("inf")          # guard: max_outer<1 -> loop body never runs (report non-converged)
        inner_ok = True            # tracks whether the LAST outer iteration's inner Newton converged
        a_np = float(alpha_np_per_eV) / Q if (alpha_np_per_eV and alpha_np_per_eV > 0.0) else 0.0  # J^-1
        for it in range(max_outer):
            U = -Q * phi + U_band
            res = self.density(U, E_F_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states, bound_tol=bound_tol,
                               alpha_np_per_eV=alpha_np_per_eV, neumann_left=neumann_left)
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
            for _newton in range(40):
                pk = psi_k[1:] if neumann_left else psi_k      # interior rows (drop the body node)
                shift = -Q * (phi_in - phi_k)                  # E_i(phi) ~ E_i^k + (U-U^k); U=-q phi
                arg = (E_F_J - (E_k[None, :] + shift[1:-1, None])) / kT   # interior nodes 1..N-2
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
                n_int = np.sum((np.abs(pk) ** 2) * occ, axis=1)   # (n_int,)
                dn_dphi = np.sum((np.abs(pk) ** 2) * dns_dphi, axis=1)
                # residual of eps0 eps phi'' = -q(Nd - n) on interior nodes
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
            else:
                inner_ok = False     # inner nonlinear-Poisson Newton never reached 1e-9 in 40 its
            # outer under-relaxation: damp the kept-state-set churn (audit SP-1)
            phi = last_phi + float(relax) * (phi_in - last_phi)
            result = res
            dV = float(np.max(np.abs(phi - last_phi)))
            if verbose:
                print("[t] SP outer {:2d}: max|dphi|={:.3e} V, n_states_bound={}".format(
                    it, dV, E_k.size), flush=True)
            last_phi = phi.copy()
            if dV < tol_V:
                break
        converged = (dV < tol_V) and inner_ok
        if not converged:
            why = []
            if dV >= tol_V:
                why.append("outer max|dphi|={:.3e} V >= tol_V={:.1e}".format(dV, tol_V))
            if not inner_ok:
                why.append("the inner nonlinear-Poisson Newton did not reach 1e-9 in 40 steps")
            warnings.warn(
                "SchrodingerPoisson.solve_self_consistent did NOT converge after {} outer "
                "iterations ({}). The returned (phi, n) is unreliable and sensitive to "
                "max_outer; try a smaller `relax`, a larger bound_tol (slab mode), or more "
                "iterations.".format(max_outer, "; ".join(why)), stacklevel=2)
        U = -Q * phi + U_band
        result = self.density(U, E_F_J, m_eff_z_kg=m_eff_z_kg, n_states=n_states, bound_tol=bound_tol,
                              alpha_np_per_eV=alpha_np_per_eV, neumann_left=neumann_left)
        result.converged = converged
        n_full = np.zeros(N)
        n_full[lo:-1] = result.density_m3                      # nodes lo..N-2 (lo=0 keeps the body node)
        return phi, n_full, result
