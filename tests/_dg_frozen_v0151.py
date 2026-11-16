"""FROZEN copy of dg_correct_density_1d as shipped in v0.15.1 (dynameta/carriers/density_gradient.py,
lines 264-350 of that release), docstring dropped, for the same-tree A/B
gate tests/test_density_gradient.py::test_hard_wall_left_right_are_bit_identical_to_v0151.
Test fixture only: never imported by the library."""
import numpy as np

from dynameta.constants import T_REF
from dynameta.core.numerics import trapz
from dynameta.carriers.density_gradient import dg_length_m


def dg_correct_density_1d_v0151(z_m, n_cl_m3, m_eff_kg: float, *, gamma: float = 1.0,
                          T_K: float = T_REF, hard_wall: str = "left",
                          conserve_charge: bool = False, tol: float = 1e-6,
                          max_nodes: int = 100000) -> np.ndarray:
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
    # audit C-16: the length is the module's own public dg_length_m (Boltzmann g = 1, which is what
    # this BVP closes with), not an inline re-derivation of the same sqrt(b/V_t) -- it also brings
    # that function's m_eff/T_K validation. Equal to the old inline value to 1 ulp.
    lq = dg_length_m(m_eff_kg, gamma=gamma, T_K=T_K)
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
