"""R19: density-gradient quantum correction -- the POST-HOC frozen-potential closure.

The density-gradient (DG) model augments drift-diffusion with the quantum potential

    Lambda = b (d^2 sqrt(n)/dz^2) / sqrt(n),      b = gamma hbar^2 / (6 m q)   [V],

the lowest-order Wigner expansion of quantum confinement. In equilibrium at a FROZEN
electrostatic potential, n satisfies V_t ln(n/n_cl) = Lambda, i.e. with u = sqrt(n) the 1D
boundary-value problem

    b u'' = V_t u ln(u^2 / n_cl(z)),    u(0) = 0 (oxide hard wall),  u(L) = sqrt(n_cl(L)),

whose characteristic length L_q = sqrt(b/V_t) = hbar sqrt(gamma/(6 m kB T)) ~ 1.2 nm for ITO
(m = 0.35 m0, 300 K) -- EXACTLY the quantum dead-layer scale the in-house Schrodinger-Poisson
solver shows (the accumulation peak displaced ~1 nm off the oxide interface where classical
DD peaks AT it). dg_correct_density_1d solves this BVP on a CLASSICAL profile n_cl(z) and
returns the quantum-corrected n(z); gamma = 0 returns n_cl EXACTLY (off-switch).

SCOPE (honest): the electrostatic potential is FROZEN (no Poisson feedback), so this is the
post-hoc correction sanctioned as the R19 fallback -- quantitative for the dead-layer SHAPE
near the interface, perturbative for the total charge (pass conserve_charge=True to rescale).
The in-Newton DG-DD (u and Lambda as DEVSIM solution variables with a Poisson-like u-equation;
a 5-variable Newton) is the documented follow-on -- DEVSIM node models cannot reference
neighbor nodes, so the discrete Laplacian must be assembled as an equation, not a model.
Pure numpy/scipy. Oracle: validation/density_gradient_dead_layer.py (vs Schrodinger-Poisson).
"""

from __future__ import annotations

import numpy as np

from dynameta.constants import HBAR, KB, M_E, Q_E

__all__ = ["quantum_potential_V", "dg_correct_density_1d", "dg_length_m"]


def dg_length_m(m_eff_kg: float, *, gamma: float = 1.0, T_K: float = 300.0) -> float:
    """The DG dead-layer length L_q = hbar sqrt(gamma/(6 m kB T)) [m]."""
    if not (m_eff_kg > 0.0 and gamma >= 0.0 and T_K > 0.0):
        raise ValueError("density_gradient: m_eff_kg > 0, gamma >= 0, T_K > 0 required")
    return float(HBAR * np.sqrt(gamma / (6.0 * m_eff_kg * KB * T_K)))


def quantum_potential_V(z_m, n_m3, m_eff_kg: float, *, gamma: float = 1.0) -> np.ndarray:
    """Lambda(z) = b (sqrt(n))'' / sqrt(n) [VOLTS] on a solved density profile (second-order
    finite differences; non-uniform z supported). gamma = 0 -> exactly zeros."""
    z = np.asarray(z_m, dtype=np.float64)
    n = np.asarray(n_m3, dtype=np.float64)
    if z.ndim != 1 or z.shape != n.shape or z.size < 5:
        raise ValueError("density_gradient: z_m and n_m3 must be matching 1D arrays (>= 5 pts)")
    if np.any(n <= 0.0):
        raise ValueError("density_gradient: n_m3 must be > 0 (floor it before calling)")
    if gamma == 0.0:
        return np.zeros_like(n)
    b = gamma * HBAR ** 2 / (6.0 * m_eff_kg * Q_E)
    u = np.sqrt(n)
    upp = np.gradient(np.gradient(u, z), z)
    return b * upp / u


def dg_correct_density_1d(z_m, n_cl_m3, m_eff_kg: float, *, gamma: float = 1.0,
                          T_K: float = 300.0, hard_wall: str = "left",
                          conserve_charge: bool = False, tol: float = 1e-6,
                          max_nodes: int = 100000) -> np.ndarray:
    """Quantum-corrected density n_dg(z) from the classical profile n_cl(z) via the frozen-
    potential DG boundary-value problem (module header). hard_wall = 'left'|'right' marks the
    insulating (oxide) end where u = sqrt(n) -> 0; the other end is pinned to the classical
    bulk. conserve_charge=True rescales n_dg so int n dz matches the classical profile (the
    frozen-potential closure otherwise trades interface charge for the dead layer). gamma = 0
    returns n_cl EXACTLY."""
    z = np.asarray(z_m, dtype=np.float64)
    n_cl = np.asarray(n_cl_m3, dtype=np.float64)
    if z.ndim != 1 or z.shape != n_cl.shape or z.size < 5:
        raise ValueError("density_gradient: z_m and n_cl_m3 must be matching 1D arrays")
    if np.any(n_cl <= 0.0):
        raise ValueError("density_gradient: n_cl_m3 must be > 0 everywhere")
    if hard_wall not in ("left", "right"):
        raise ValueError("density_gradient: hard_wall must be 'left' or 'right'")
    if gamma == 0.0:
        return n_cl.copy()
    from scipy.integrate import solve_bvp
    from scipy.interpolate import interp1d

    flip = hard_wall == "right"
    zz = z[::-1] * -1.0 if flip else z                       # canonical: wall at zz[0]
    nn = n_cl[::-1] if flip else n_cl
    zz = zz - zz[0]
    b = gamma * HBAR ** 2 / (6.0 * m_eff_kg * Q_E)
    v_t = KB * T_K / Q_E
    lq = np.sqrt(b / v_t)
    # NONDIMENSIONALIZE (v = u/sqrt(n_ref), x = z/L_q): u ~ 1e13 with u' ~ 1e22 defeats
    # solve_bvp's mixed-component tolerance; the scaled problem is O(1) in both components.
    n_ref = float(np.max(nn))
    x_grid = zz / lq
    r_of_x = interp1d(x_grid, nn / n_ref, kind="cubic", fill_value=(nn[0] / n_ref, nn[-1] / n_ref),
                      bounds_error=False)

    def rhs(x, y):
        # the wall end v -> 0 carries a log singularity in the Jacobian (d/dv = ln(v^2/r) + 2
        # -> -inf) that drives endless mesh refinement; cap the log at a PHYSICAL floor
        # (densities below 1e-12 of the peak are outside the DG model's meaning anyway)
        r = r_of_x(x)
        return np.vstack([y[1], y[0] * np.log(np.maximum(y[0] ** 2, 1e-12) / r)])

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - np.sqrt(nn[-1] / n_ref)])

    v0 = np.sqrt(nn / n_ref) * np.tanh(x_grid)               # dead-layer-shaped initial guess
    y0 = np.vstack([v0, np.gradient(v0, x_grid)])
    sol = solve_bvp(rhs, bc, x_grid, y0, tol=tol, max_nodes=max_nodes)
    if not sol.success:
        raise RuntimeError("density_gradient: DG BVP did not converge ({})".format(sol.message))
    u = sol.sol(x_grid)[0] * np.sqrt(n_ref)
    n_dg = np.maximum(u, 0.0) ** 2
    if flip:
        n_dg = n_dg[::-1]
    if conserve_charge:
        n_dg = n_dg * (np.trapezoid(n_cl, z) / max(np.trapezoid(n_dg, z), 1e-300))
    return n_dg
