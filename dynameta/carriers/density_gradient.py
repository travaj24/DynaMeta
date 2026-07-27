"""R19: density-gradient quantum correction -- the POST-HOC frozen-potential closure.

The density-gradient (DG) model augments drift-diffusion with the quantum potential

    Lambda = b (d^2 sqrt(n)/dz^2) / sqrt(n),      b = gamma hbar^2 / (6 m q)   [V],

the lowest-order Wigner expansion of quantum confinement. In equilibrium at a FROZEN
electrostatic potential, n satisfies V_t ln(n/n_cl) = Lambda, i.e. with u = sqrt(n) the 1D
boundary-value problem

    b u'' = V_t u ln(u^2 / n_cl(z)),    u(0) = 0 (oxide hard wall),  u(L) = sqrt(n_cl(L)),

whose characteristic length L_q = sqrt(b/V_t) = hbar sqrt(gamma/(6 m kB T)) ~ 1.2 nm for ITO
(m = 0.35 m0, 300 K) -- the quantum dead-layer scale the in-house Schrodinger-Poisson solver
shows (the accumulation peak displaced ~1 nm off the oxide interface where classical DD peaks
AT it). dg_correct_density_1d solves this BVP on a CLASSICAL profile n_cl(z) and returns the
quantum-corrected n(z); gamma = 0 returns n_cl EXACTLY (off-switch).

STATISTICS -- read before quoting L_q (audit C-8). That BVP is the BOLTZMANN closure: it follows
from n = n_cl exp(Lambda/V_t). Degenerate ITO (eta ~ 10-20, which the rest of this subsystem
insists on) obeys n = N_c F_1/2(eta_cl + Lambda/V_t), which linearises to
n = n_cl (1 + Lambda/(g V_t)) with the generalized-Einstein factor g = F_1/2/F_-1/2
(carriers.einstein.g_einstein) -- i.e. V_t -> g V_t and L_q -> sqrt(b/(g V_t)), a factor sqrt(g)
SHORTER: 1.1847 nm becomes 0.4543 nm at g = 6.8 (eta ~ 10) and 0.3236 nm at g = 13.4 (eta ~ 20),
2.6-3.7x. The in-Newton twin (physics_density_gradient.py:85-100) already carries the FD form via
vdiff_dg/g_enh. So `gamma` here is a FITTED parameter that absorbs the degeneracy factor
(gamma_eff = gamma g): the ~1.2 nm agreement with Schrodinger-Poisson in
validation/density_gradient_dead_layer.py GATE C is a CALIBRATION at gamma = 1, not a derivation
from FD statistics. Pass dg_length_m(..., degeneracy_g=g_einstein(n/N_c)) for the FD-consistent
length, and read `gamma` as a shape knob rather than a first-principles 1.

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

from dynameta.constants import HBAR, KB, Q_E
from dynameta.core.numerics import trapz          # audit X-1: floor-safe (np.trapezoid needs numpy>=2.0)

__all__ = ["quantum_potential_V", "dg_correct_density_1d", "dg_length_m"]


def dg_length_m(m_eff_kg: float, *, gamma: float = 1.0, T_K: float = 300.0,
                degeneracy_g: float = 1.0) -> float:
    """The DG dead-layer length L_q = sqrt(b/(g V_t)) = hbar sqrt(gamma/(6 m kB T g)) [m].

    degeneracy_g is the generalized-Einstein factor g = F_1/2(eta)/F_-1/2(eta) of the statistics
    the profile actually obeys -- `carriers.einstein.g_einstein(n/N_c)`. The default g = 1 is the
    BOLTZMANN limit and the byte-identical off-switch (what this function always returned, and
    what the BVP in dg_correct_density_1d closes with; see the module header on audit C-8). For
    degenerate ITO the FD-consistent length is sqrt(g) SHORTER -- 1.1847 nm -> 0.3236 nm at
    g = 13.4 -- and the difference is exactly what the fitted `gamma` absorbs."""
    if not (m_eff_kg > 0.0 and gamma >= 0.0 and T_K > 0.0 and degeneracy_g > 0.0):
        raise ValueError("density_gradient: m_eff_kg > 0, gamma >= 0, T_K > 0, degeneracy_g > 0 "
                         "required")
    return float(HBAR * np.sqrt(gamma / (6.0 * m_eff_kg * KB * T_K * degeneracy_g)))


def _fd_weights_d2(nodes, x0) -> np.ndarray:
    """Finite-difference weights w with sum_j w_j f(nodes_j) = f''(x0) + O(h^(k-2)) for a k-node
    stencil (Fornberg), from the local Vandermonde system sum_j w_j (z_j - x0)^p = 2! delta_{p,2}.
    The offsets are scaled by the stencil half-width before the solve and the result unscaled, so
    the 4x4 system is conditioned on O(1) numbers rather than on 1e-9-sized metres."""
    d = np.asarray(nodes, dtype=np.float64) - float(x0)
    hs = float(np.max(np.abs(d)))
    if hs <= 0.0 or np.unique(d).size != d.size:
        raise ValueError("density_gradient: degenerate FD stencil (repeated or coincident z nodes)")
    k = d.size
    v = np.vander(d / hs, k, increasing=True).T          # V[p, j] = (d_j/hs)**p
    rhs = np.zeros(k)
    rhs[2] = 2.0
    return np.linalg.solve(v, rhs) / hs ** 2


def _second_derivative(f: np.ndarray, z: np.ndarray) -> np.ndarray:
    """f''(z) on a 1-D (possibly non-uniform) grid, SECOND-ORDER AT EVERY NODE -- ends included.

    audit C-3 / N3 (= 2026-07-17 ledger S1-6): this used to be `np.gradient(np.gradient(f, z), z)`,
    which is NOT a second-difference stencil. `np.gradient`'s default `edge_order=1` makes the INNER
    pass a one-sided first-order slope at nodes 0 and N-1, and the outer centred pass then mixes
    that biased slope into nodes 1 and N-2, so FOUR nodes converge to the wrong limit:

        node 0, N-1 -> (1/2) f''        node 1, N-2 -> (3/4) f''

    i.e. 50% and 25% errors that are INDEPENDENT of h -- refining the grid does not remove them.
    (The interior was convergent but used the WIDE composition (f[i+2] - 2 f[i] + f[i-2])/(2h)^2,
    whose leading error is 4x the 3-point stencil's.)

    Interior: the 3-point non-uniform second difference, exact for quadratics. Ends: a 4-point
    one-sided stencil, exact for cubics -> O(h^2) one-sided, with the weights solved from the local
    Vandermonde system so a non-uniform mesh at the ends is handled too. Gated at all four
    previously-broken nodes by tests/test_density_gradient.py (observed order ~2)."""
    f = np.asarray(f, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    d = np.diff(z)
    if not (np.all(d > 0.0) or np.all(d < 0.0)):
        raise ValueError("density_gradient: z_m must be strictly monotonic (increasing or "
                         "decreasing) -- the second-difference stencils divide by the spacings")
    h1 = z[1:-1] - z[:-2]
    h2 = z[2:] - z[1:-1]
    out = np.empty_like(z)
    out[1:-1] = 2.0 * (h2 * f[:-2] - (h1 + h2) * f[1:-1] + h1 * f[2:]) / (h1 * h2 * (h1 + h2))
    out[0] = float(_fd_weights_d2(z[:4], z[0]) @ f[:4])
    out[-1] = float(_fd_weights_d2(z[-4:], z[-1]) @ f[-4:])
    return out


def quantum_potential_V(z_m, n_m3, m_eff_kg: float, *, gamma: float = 1.0) -> np.ndarray:
    """Lambda(z) = b (sqrt(n))'' / sqrt(n) [VOLTS] on a solved density profile (second-order
    finite differences AT EVERY NODE, ends included -- audit C-3/N3; non-uniform z supported).
    gamma = 0 -> exactly zeros."""
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
    return b * _second_derivative(u, z) / u


def dg_correct_density_1d(z_m, n_cl_m3, m_eff_kg: float, *, gamma: float = 1.0,
                          T_K: float = 300.0, hard_wall: str = "left",
                          conserve_charge: bool = False, tol: float = 1e-6,
                          max_nodes: int = 100000) -> np.ndarray:
    """Quantum-corrected density n_dg(z) from the classical profile n_cl(z) via the frozen-
    potential DG boundary-value problem (module header). hard_wall = 'left'|'right' marks the
    insulating (oxide) end where u = sqrt(n) -> 0; the other end is pinned to the classical
    bulk. conserve_charge=True rescales n_dg so int n dz matches the classical profile (the
    frozen-potential closure otherwise trades interface charge for the dead layer). gamma = 0
    returns n_cl EXACTLY.

    The closure is BOLTZMANN (n = n_cl exp(Lambda/V_t)); on a degenerate profile `gamma` is a
    fitted parameter absorbing the generalized-Einstein factor g -- see the module header
    (audit C-8) before quoting the dead-layer length as derived."""
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
        n_dg = n_dg * (trapz(n_cl, z) / max(trapz(n_dg, z), 1e-300))
    return n_dg
