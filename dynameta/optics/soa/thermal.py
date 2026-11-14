"""Spatially-resolved (z) steady thermal profile for the QD-SOA -- the reduced 1-D heat-conduction
upgrade to the lumped Rth/Cth self-heating. The active stripe is a 1-D fin: longitudinal conduction
along z (kappa A) plus a distributed sink to the heat-sink/substrate (per-length thermal conductance
g_sub = 1/Rth_prime), driven by the per-slice dissipated power q(z) [W/m]:

    kappa A d2T/dz2 - (T - T0)/Rth_prime = -q(z).

ends='sunk' (Dirichlet T=T0, mounted/heat-sunk facets) gives the classic DOME profile; ends=
'insulated' (Neumann) lets every slice relax to its local lumped value T0 + q Rth_prime when
conduction is negligible. A spatially-resolved thermal FEM (carriers/thermal_fem) can ALSO supply a
T(z) through the SAME per-slice interface that feeds the gain (QDGainModel.gain_per_m_thermal), via
sample_T_along_axis on its ThermalResult.T_at -- this reduced 1-D fin and that FEM are alternative
T(z) sources for the same gain seam (this module itself contains NO FEM; it is a tridiagonal fin
solve plus a passive sampler).

SI; ASCII. (1-D fin / Ning-Lippi distributed-thermal SOA model.)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lapack

__all__ = ["thermal_profile_steady_1d", "thermal_profile_transient_1d", "sample_T_along_axis",
           "dome_analytic"]


# TRIDIAGONAL fin operator (audit S3-22, wave-7 2026-11-02). Both solvers used to assemble a DENSE
# (n, n) matrix in a Python loop and solve it densely (np.linalg.solve / lu_factor + lu_solve:
# O(n^2) memory, O(n^3) factor, O(n^2) per implicit step) although the docstring promised a
# tridiagonal solve. They now build only the three diagonals and use LAPACK's banded LU with
# partial pivoting (dgbtrf once + dgbtrs per right-hand side: O(n) memory and work).
# Measured vs the old dense path (HEAD 7b9481a copy) and vs an EXACT rational (fractions.Fraction)
# Thomas solve, 2026-11-02, kappaA 2e-5, Rth' 5e-4, L 1 mm, random q:
#   * 'insulated': old vs new max|dT|/max|T - T0| 3.5e-15 (n = 60) .. 7.6e-15 (n = 2000, 400
#     implicit steps) -- rounding only; kappaA = 0 bit-identical; n = 2, 3 bit-identical.
#   * 'sunk': the new solve is EXACT to 1.0e-15 (n = 60) / 9.9e-15 (n = 300) / 6.1e-14 (n = 1000)
#     where the OLD dense solve was off by 1.0e-12 / 3.1e-10 / 1.1e-9 (transient run to steady:
#     old 2.6e-11 / 8.3e-10, new 1.0e-15 / 7.7e-15) -- see the Dirichlet elimination note in
#     _fin_tridiag. So 'sunk' old-vs-new differs at the OLD error level (up to 1e-7 at n = 2000 x 400
#     steps); the new value is the accurate one.
#   * time (same box, loaded): steady n = 60 46 -> 13 us, n = 2000 79 ms -> 0.22 ms; transient
#     n = 60 x 4000 steps 47 -> 14 ms, n = 2000 x 400 steps 455 -> 35 ms.
def _fin_tridiag(n, diag, c, ends):
    """(dl, d, du) of the fin operator with diagonal `diag` and couplings `c` (sub = super = c in
    the interior). ends 'sunk' -> identity rows 0 and n-1 (Dirichlet theta = 0); 'insulated' ->
    the 2nd-order mirror ghost doubles the single neighbour coupling (Neumann)."""
    dl = np.full(n - 1, c, dtype=np.float64)
    d = np.full(n, diag, dtype=np.float64)
    du = np.full(n - 1, c, dtype=np.float64)
    if ends == "sunk":
        d[0] = 1.0; du[0] = 0.0                              # row 0:   theta_0 = rhs_0 (= 0)
        d[-1] = 1.0; dl[-1] = 0.0                            # row n-1: theta_{n-1} = rhs_{n-1} (= 0)
        # the facet values are KNOWN (theta = 0), so their couplings into rows 1 and n-2 multiply
        # zero: drop them (symmetric Dirichlet elimination; mathematically identical). Kept, they
        # make partial pivoting swap row 1 (|c| ~ 1e7-1e8) above the unit Dirichlet row, which cost
        # the old dense solve 6.1e-12 (n = 60) and 3.8e-10 (n = 300) relative error vs an exact
        # rational (Fraction) Thomas solve; eliminated, the banded solve is 1e-15-level.
        if n > 2:
            dl[0] = 0.0                                      # A[1, 0]
            du[-1] = 0.0                                     # A[n-2, n-1]
    elif ends == "insulated":
        du[0] = 2.0 * c                                      # theta[-1] = theta[1] -> neighbour doubles
        dl[-1] = 2.0 * c                                     # (exact lumped reduction at c=0 preserved)
    else:
        raise ValueError("ends must be 'sunk' or 'insulated'")
    return dl, d, du


def _tridiag_factor(dl, d, du, who):
    """Banded LU (LAPACK dgbtrf, kl = ku = 1, partial pivoting) of the tridiagonal (dl, d, du).
    (dgbtrf rather than the dedicated dgttrf: scipy 1.17's dgttrf wrapper rejects n = 2, which the
    fin solvers accept.) Raises LinAlgError on an exactly singular pivot."""
    n = d.size
    ab = np.zeros((4, n))                                    # LAPACK band storage, kl extra fill row
    ab[1, 1:] = du                                           # ab[kl + ku + i - j, j] = A[i, j]
    ab[2, :] = d
    ab[3, :-1] = dl
    lu, piv, info = lapack.dgbtrf(ab, 1, 1)
    if info != 0:
        raise np.linalg.LinAlgError("{}: singular fin operator (dgbtrf info {})".format(who, info))
    return lu, piv


def _tridiag_solve(fac, b):
    lu, piv = fac
    x, info = lapack.dgbtrs(lu, 1, 1, np.asarray(b, dtype=np.float64), piv)
    return x


def thermal_profile_steady_1d(q_per_m, dz_m, kappaA_W_m_K, Rth_prime_K_m_W, T0_K, *, ends="sunk"):
    """Steady T(z) [K] from the 1-D fin equation kappa A T'' - (T-T0)/Rth' = -q(z). q_per_m is the
    per-slice line dissipation [W/m] (length n). kappaA_W_m_K = kappa*A [W m / K] (longitudinal
    conduction); Rth_prime_K_m_W = the per-length thermal resistance to the sink [K m / W] (g_sub =
    1/Rth'). ends 'sunk' -> T(0)=T(L)=T0 (Dirichlet); 'insulated' -> dT/dz=0 (Neumann). Reduces to the
    lumped per-slice T0 + q Rth' when kappaA -> 0 (insulated). Tridiagonal solve (LAPACK banded dgbtrf/
    dgbtrs, O(n); audit S3-22).

    1-D LONGITUDINAL fin: conduction is resolved along z ONLY; transverse/vertical conduction to the
    sink is LUMPED into Rth_prime (per-length). For a true 2-D/3-D temperature field use the thermal
    FEM (carriers/thermal_fem) and feed its T(z) via sample_T_along_axis. ('sunk' pins the Dirichlet
    facets at the FIRST/LAST sample point -- use a node grid dz=L/(n-1) for the dome to span [0,L].)"""
    q = np.atleast_1d(np.asarray(q_per_m, dtype=np.float64))
    n = q.size
    if n < 2:
        raise ValueError("thermal_profile_steady_1d: need >= 2 slices")
    c = float(kappaA_W_m_K) / (float(dz_m) ** 2)              # conduction coupling [W/(m K)]
    s = 1.0 / float(Rth_prime_K_m_W)                          # distributed sink [W/(m K)]
    if ends not in ("sunk", "insulated"):
        raise ValueError("thermal_profile_steady_1d: ends must be 'sunk' or 'insulated'")
    b = -q.copy()                                            # theta = T - T0; sink/conduction on LHS
    if ends == "sunk":                                       # Dirichlet theta=0 at both facets
        b[0] = 0.0; b[-1] = 0.0
    fac = _tridiag_factor(*_fin_tridiag(n, -(2.0 * c + s), c, ends), "thermal_profile_steady_1d")
    theta = _tridiag_solve(fac, b)
    return float(T0_K) + theta


def thermal_profile_transient_1d(q_per_m, dz_m, kappaA_W_m_K, Rth_prime_K_m_W, T0_K, C_line_J_m_K,
                                 dt_s, n_steps, *, ends="sunk", T_init=None, return_history=False):
    """Transient 1-D fin C' dT/dt = kappa A T'' - (T-T0)/Rth' + q(z), marched by IMPLICIT EULER
    (unconditionally stable). C_line_J_m_K = C' is the per-length heat capacity [J/(m K)] (= rho Cp
    A_cross); tau_th = C' Rth' the lumped thermal time constant. Returns T(z) after n_steps (or, with
    return_history, the (n_steps+1, n) array including the initial profile). Reduces to
    thermal_profile_steady_1d as n_steps dt -> inf; the lumped limit (kappa A -> 0, insulated, uniform
    q) is the RC charge-up T(t) = T0 + q Rth'(1 - exp(-t/tau_th)). T_init defaults to the uniform T0.

    Implicit step: (C'/dt I - L) theta^{n+1} = C'/dt theta^n + q, theta = T - T0, L the steady fin
    operator (so the fixed point is L theta = -q, identical to the steady solve)."""
    q = np.atleast_1d(np.asarray(q_per_m, dtype=np.float64))
    n = q.size
    if n < 2:
        raise ValueError("thermal_profile_transient_1d: need >= 2 slices")
    if C_line_J_m_K <= 0.0 or dt_s <= 0.0:
        raise ValueError("thermal_profile_transient_1d: C_line and dt must be > 0")
    c = float(kappaA_W_m_K) / (float(dz_m) ** 2)
    s = 1.0 / float(Rth_prime_K_m_W)
    a = float(C_line_J_m_K) / float(dt_s)                     # backward-Euler accumulation [W/(m K)]
    if ends not in ("sunk", "insulated"):
        raise ValueError("thermal_profile_transient_1d: ends must be 'sunk' or 'insulated'")
    # M = (C'/dt) I - L, tridiagonal (mirror ghost on 'insulated' matches the steady solve);
    # factored ONCE (dgbtrf), O(n) dgbtrs per step (audit S3-22)
    fac = _tridiag_factor(*_fin_tridiag(n, a + 2.0 * c + s, -c, ends), "thermal_profile_transient_1d")
    theta = (np.asarray(T_init, dtype=np.float64) - T0_K) if T_init is not None else np.zeros(n)
    hist = [T0_K + theta.copy()] if return_history else None
    for _ in range(int(n_steps)):
        rhs = a * theta + q
        if ends == "sunk":
            rhs[0] = 0.0; rhs[-1] = 0.0
        theta = _tridiag_solve(fac, rhs)
        if return_history:
            hist.append(T0_K + theta.copy())
    return np.array(hist) if return_history else (float(T0_K) + theta)


def sample_T_along_axis(T_at, s_centers_m, *, axis="x", a_fixed_m=0.0, b_fixed_m=0.0):
    """Sample an external 3-D temperature field T_at(x,y,z) [K] along the SOA propagation axis to
    produce the per-slice T(s) for QDGainModel.gain_per_m_thermal. T_at is ANY callable returning K at
    a point in metres -- a thermal-FEM ThermalResult.T_at, an analytic field, or an interpolant. `axis`
    in {'x','y','z'} is the propagation direction; the other two coordinates are held at a_fixed_m,
    b_fixed_m. This is the external-field coupling SEAM -- a ONE-WAY point-sampler (no solve, no
    feedback loop): it just evaluates T_at at the slice centres. The reduced thermal_profile_steady_1d
    and an external FEM both feed the SAME gain interface through the resulting plain T(s) array (no
    NGSolve import here -- the caller runs the FEM and passes its ThermalResult.T_at)."""
    s = np.atleast_1d(np.asarray(s_centers_m, dtype=np.float64))
    out = np.empty(s.size, dtype=np.float64)
    for i, sv in enumerate(s):
        if axis == "x":
            out[i] = float(T_at(sv, a_fixed_m, b_fixed_m))
        elif axis == "y":
            out[i] = float(T_at(a_fixed_m, sv, b_fixed_m))
        elif axis == "z":
            out[i] = float(T_at(a_fixed_m, b_fixed_m, sv))
        else:
            raise ValueError("sample_T_along_axis: axis must be 'x', 'y' or 'z'")
    return out


def dome_analytic(q_W_m, L_m, kappaA_W_m_K, Rth_prime_K_m_W, T0_K, z):
    """Analytic sunk-ends dome for UNIFORM q: T(z) = T0 + q Rth'[1 - cosh((z-L/2)/Lc)/cosh(L/2Lc)],
    Lc = sqrt(kappa A Rth') the thermal healing length. The independent oracle for the numerical solve."""
    Lc = np.sqrt(float(kappaA_W_m_K) * float(Rth_prime_K_m_W))
    zc = np.asarray(z) - 0.5 * L_m
    return T0_K + q_W_m * Rth_prime_K_m_W * (1.0 - np.cosh(zc / Lc) / np.cosh(0.5 * L_m / Lc))
