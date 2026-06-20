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
from scipy.linalg import lu_factor, lu_solve


def thermal_profile_steady_1d(q_per_m, dz_m, kappaA_W_m_K, Rth_prime_K_m_W, T0_K, *, ends="sunk"):
    """Steady T(z) [K] from the 1-D fin equation kappa A T'' - (T-T0)/Rth' = -q(z). q_per_m is the
    per-slice line dissipation [W/m] (length n). kappaA_W_m_K = kappa*A [W m / K] (longitudinal
    conduction); Rth_prime_K_m_W = the per-length thermal resistance to the sink [K m / W] (g_sub =
    1/Rth'). ends 'sunk' -> T(0)=T(L)=T0 (Dirichlet); 'insulated' -> dT/dz=0 (Neumann). Reduces to the
    lumped per-slice T0 + q Rth' when kappaA -> 0 (insulated). Tridiagonal solve.

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
    A = np.zeros((n, n))
    b = -q.copy()                                            # theta = T - T0; sink/conduction on LHS
    for k in range(n):
        A[k, k] = -(2.0 * c + s)
        if k > 0:
            A[k, k - 1] = c
        if k < n - 1:
            A[k, k + 1] = c
    if ends == "sunk":                                       # Dirichlet theta=0 at both facets
        A[0, :] = 0.0; A[0, 0] = 1.0; b[0] = 0.0
        A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0
    elif ends == "insulated":                               # Neumann dT/dz=0: 2nd-order mirror ghost
        A[0, 0] = -(2.0 * c + s); A[0, 1] = 2.0 * c          # theta[-1]=theta[1] -> neighbour doubles
        A[-1, -1] = -(2.0 * c + s); A[-1, -2] = 2.0 * c      # (exact lumped reduction at c=0 preserved)
    else:
        raise ValueError("thermal_profile_steady_1d: ends must be 'sunk' or 'insulated'")
    theta = np.linalg.solve(A, b)
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
    M = np.zeros((n, n))                                      # M = (C'/dt) I - L
    for k in range(n):
        M[k, k] = a + 2.0 * c + s
        if k > 0:
            M[k, k - 1] = -c
        if k < n - 1:
            M[k, k + 1] = -c
    if ends == "sunk":
        M[0, :] = 0.0; M[0, 0] = 1.0
        M[-1, :] = 0.0; M[-1, -1] = 1.0
    elif ends == "insulated":
        M[0, 0] = a + 2.0 * c + s; M[0, 1] = -2.0 * c        # mirror ghost (matches the steady solve)
        M[-1, -1] = a + 2.0 * c + s; M[-1, -2] = -2.0 * c
    else:
        raise ValueError("thermal_profile_transient_1d: ends must be 'sunk' or 'insulated'")
    lu = lu_factor(M)                                        # factor once; reuse every step
    theta = (np.asarray(T_init, dtype=np.float64) - T0_K) if T_init is not None else np.zeros(n)
    hist = [T0_K + theta.copy()] if return_history else None
    for _ in range(int(n_steps)):
        rhs = a * theta + q
        if ends == "sunk":
            rhs[0] = 0.0; rhs[-1] = 0.0
        theta = lu_solve(lu, rhs)
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
