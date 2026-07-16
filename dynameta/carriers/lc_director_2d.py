"""
Two-dimensional nematic director theta(x, z) for LATERALLY PATTERNED liquid-crystal cells -- the axis the
1-D solvers (lc_director.py) lack. A pixelated SLM / metasurface gate applies a laterally non-uniform top
voltage V_top(x); near a pixel boundary the field acquires a lateral (fringing) component Ex and the
director tilt theta becomes a 2-D field theta(x, z) (in-plane director n = (sin theta, 0, cos theta), the
FIELD-AXIS convention of lc_director.py: theta = 0 homeotropic along z, theta = pi/2 planar along x).

Physics (one-constant Frank elastic K, planar cell of thickness d, lateral period/width Lx):
  - FIELD: a 2-D electrostatic potential V(x, z) from Laplace div(eps grad V) = 0 (constant eps; the
    director-anisotropy back-coupling to the field is a documented second-order extension), Dirichlet
    V(x, 0) = 0 / V(x, d) = V_top(x), periodic or Neumann in x. E = -grad V -> (Ex, Ez). 'uniform_columns'
    instead sets Ez = V_top(x)/d, Ex = 0 (each column independent -> the 1-D limit).
  - DIRECTOR: the 2-D Euler-Lagrange torque balance (minimize the Frank + dielectric free energy)
        K (theta_xx + theta_zz) + eps0 dEps (Ex sin th + Ez cos th)(Ex cos th - Ez sin th) = 0,
    dEps = eps_para - eps_perp, strong anchoring theta = theta_b at z = 0, d; periodic/Neumann in x.
    Reduces EXACTLY to the 1-D one-constant director equation when Ex = 0 (K theta_zz - eps0 dEps Ez^2
    sin th cos th = 0). Solved by under-relaxed nonlinear Gauss-Seidel, seeded column-by-column from the
    1-D director_profile_bvp at the local voltage (which also lands the Freedericksz-tilted branch).

Emits theta(x, z), the field (Ex, Ez, V) and a per-column optical n_eff(x). SI units; pure numpy/scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from dynameta.constants import EPS0
from dynameta.carriers.lc_director import (
    director_profile_bvp, n_eff_from_theta_profile)

__all__ = ["LC2DResult", "director_profile_2d"]


@dataclass
class LC2DResult:
    """2-D static director solution theta(x, z) (FIELD-AXIS theta) on a rectangular grid."""
    x_m: np.ndarray                 # (nx,)
    z_m: np.ndarray                 # (nz,)
    theta_field_rad: np.ndarray     # (nx, nz) tilt from the field/z axis
    Ex: np.ndarray                  # (nx, nz) lateral field
    Ez: np.ndarray                  # (nx, nz) normal field
    V: np.ndarray                   # (nx, nz) electrostatic potential
    n_eff_of_x: np.ndarray          # (nx,) per-column OPL n_eff (nan if n_o/n_e not given)
    theta_b_rad: float
    iters: int
    residual: float
    x_boundary: str = "periodic"
    success: bool = True
    message: str = ""


def _solve_laplace_2d(V_top_x: np.ndarray, nx: int, nz: int, dx: float, dz: float,
                      x_boundary: str) -> np.ndarray:
    """Solve Laplace V_xx + V_zz = 0 with V(:,0)=0, V(:,nz-1)=V_top_x, periodic or Neumann in x.
    Returns V (nx, nz). Direct sparse solve over the interior z-rows (j = 1..nz-2)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    nj = nz - 2                                          # interior z rows
    N = nx * nj
    ix2 = 1.0 / (dx * dx); iz2 = 1.0 / (dz * dz)
    rows = []; cols = []; vals = []; b = np.zeros(N)

    def idx(i, j):                                       # j in 1..nz-2 -> interior index
        return i * nj + (j - 1)

    periodic = (str(x_boundary).lower() == "periodic")
    for i in range(nx):
        for j in range(1, nz - 1):
            k = idx(i, j)
            rows.append(k); cols.append(k); vals.append(-2.0 * (ix2 + iz2))
            # x-neighbours
            if periodic:
                ip = (i + 1) % nx; im = (i - 1) % nx
                for nb in (ip, im):
                    rows.append(k); cols.append(idx(nb, j)); vals.append(ix2)
            else:                                        # Neumann (zero lateral gradient at x ends)
                ip = i + 1 if i < nx - 1 else i - 1
                im = i - 1 if i > 0 else i + 1
                rows.append(k); cols.append(idx(ip, j)); vals.append(ix2)
                rows.append(k); cols.append(idx(im, j)); vals.append(ix2)
            # z-neighbours (Dirichlet rows folded into rhs)
            if j + 1 <= nz - 2:
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(iz2)
            else:
                b[k] -= iz2 * float(V_top_x[i])          # V(:, nz-1) = V_top
            if j - 1 >= 1:
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(iz2)
            # else V(:,0)=0 contributes nothing
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    sol = spla.spsolve(A, b)
    V = np.zeros((nx, nz))
    V[:, -1] = np.asarray(V_top_x, float)
    for i in range(nx):
        for j in range(1, nz - 1):
            V[i, j] = sol[idx(i, j)]
    return V


def director_profile_2d(*, K: float, eps_para: float, eps_perp: float,
                        V_top: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]],
                        d_planar: float, Lx_m: float, nx: int = 49, nz: int = 41,
                        theta_b_rad: float = math.radians(89.9),
                        field: str = "laplace", x_boundary: str = "periodic",
                        n_o: Optional[float] = None, n_e: Optional[float] = None,
                        opt_model: str = "extra_k_radial",
                        max_iter: int = 4000, tol: float = 1e-7, omega: float = 0.8) -> LC2DResult:
    """Solve the 2-D one-constant nematic director theta(x, z) for a laterally patterned planar cell.
    V_top sets the top-electrode voltage vs x: a scalar (uniform), an (nx,) array, or a callable
    V_top(x_array). field='laplace' solves the 2-D potential (fringing fields); 'uniform_columns' makes
    each column independent (Ez = V_top(x)/d, Ex = 0 -> the per-column 1-D limit). x_boundary in
    {'periodic','neumann'}. Returns an LC2DResult with theta(x,z), (Ex,Ez,V) and per-column n_eff(x).

    Reduces to lc_director.director_profile_bvp(K11=K33=K) at every column when V_top is uniform."""
    K = float(K); dEps = float(eps_para) - float(eps_perp)
    if not (K > 0):
        raise ValueError("K must be > 0")
    if not (d_planar > 0 and Lx_m > 0):
        raise ValueError("d_planar and Lx_m must be > 0")
    if nx < 4 or nz < 5:
        raise ValueError("need nx >= 4, nz >= 5")
    theta_b = float(theta_b_rad)
    periodic = (str(x_boundary).lower() == "periodic")
    z = np.linspace(0.0, float(d_planar), nz)
    dz = z[1] - z[0]
    dx = (float(Lx_m) / nx) if periodic else (float(Lx_m) / (nx - 1))
    x = (np.arange(nx) * dx) if periodic else np.linspace(0.0, float(Lx_m), nx)

    if callable(V_top):
        V_top_x = np.asarray(V_top(x), dtype=float).reshape(nx)
    else:
        arr = np.asarray(V_top, dtype=float)
        V_top_x = np.full(nx, float(arr)) if arr.ndim == 0 else arr.reshape(nx)

    # ---- field
    fld = str(field).lower()
    if fld == "uniform_columns":
        V = np.tile(z / float(d_planar), (nx, 1)) * V_top_x[:, None]   # linear in z per column
        Ez = np.tile(-V_top_x[:, None] / float(d_planar), (1, nz))     # E = -grad V (consistent w/ laplace)
        Ex = np.zeros((nx, nz))
    elif fld == "laplace":
        V = _solve_laplace_2d(V_top_x, nx, nz, dx, dz, x_boundary)
        # E = -grad V (central in interior; one-sided at z ends; periodic/edge in x)
        Ez = np.zeros((nx, nz)); Ex = np.zeros((nx, nz))
        Ez[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2.0 * dz)
        Ez[:, 0] = -(V[:, 1] - V[:, 0]) / dz
        Ez[:, -1] = -(V[:, -1] - V[:, -2]) / dz
        if periodic:
            Ex = -(np.roll(V, -1, axis=0) - np.roll(V, 1, axis=0)) / (2.0 * dx)
        else:
            Ex[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2.0 * dx)
            Ex[0, :] = 0.0; Ex[-1, :] = 0.0              # Neumann walls
    else:
        raise ValueError("field must be 'laplace' or 'uniform_columns'")

    # ---- seed theta column-by-column from the 1-D solver at the local column voltage (lands the
    #      Freedericksz-tilted branch; for a uniform column field this seed is already the solution).
    th = np.empty((nx, nz))
    seen = {}
    for i in range(nx):
        key = round(float(V_top_x[i]), 12)
        if key not in seen:
            r = director_profile_bvp(V_app=float(V_top_x[i]), K11=K, K33=K, eps_para=eps_para,
                                     eps_perp=eps_perp, d_planar=float(d_planar), nz=nz,
                                     theta_b_rad=theta_b, field_model="uniform")
            seen[key] = np.asarray(r.theta_field_rad, float)
        th[i, :] = seen[key]

    # ---- under-relaxed nonlinear Gauss-Seidel on the interior (j = 1..nz-2), all x.
    #      Vectorized RED-BLACK ordering: two interleaved (i+j)-parity half-sweeps, each a flat numpy
    #      gather/update over that colour's points with the SAME per-point equation and relaxation as
    #      the former lexicographic triple loop -- the iterates differ (ordering) but the fixed point
    #      is identical, and the 5-point stencil makes every neighbour of one colour the other colour
    #      (a periodic ring with odd nx leaves one same-colour seam pair, merely Jacobi-coupled there).
    ix2 = 1.0 / (dx * dx); iz2 = 1.0 / (dz * dz); S = 2.0 * (ix2 + iz2)
    cD = EPS0 * dEps
    # precompute flat neighbour indices (th is C-ordered (nx, nz): flat k = i*nz + j) per colour
    ii = np.arange(nx)[:, None]; jj = np.broadcast_to(np.arange(nz)[None, :], (nx, nz))
    if periodic:
        ip = (ii + 1) % nx; im = (ii - 1) % nx
    else:                                                # Neumann walls: both neighbours fold inward
        ip = np.where(ii < nx - 1, ii + 1, ii - 1); im = np.where(ii > 0, ii - 1, ii + 1)
    kip = (ip * nz + jj).ravel(); kim = (im * nz + jj).ravel()
    kjp = (ii * nz + jj + 1).ravel(); kjm = (ii * nz + jj - 1).ravel()
    interior = (jj >= 1) & (jj <= nz - 2)
    colors = []                                          # red, black: (self, x+, x-, z+, z-) indices
    for p in (0, 1):
        k = np.flatnonzero((((ii + jj) % 2 == p) & interior).ravel())
        colors.append((k, kip[k], kim[k], kjp[k], kjm[k], Ex.ravel()[k], Ez.ravel()[k]))
    thf = th.ravel()                                     # flat VIEW -- writes land in th
    res = float("inf"); it = 0
    err_est = float("inf")
    _res_hist = []
    for it in range(1, int(max_iter) + 1):
        dmax = 0.0
        for k, kxp, kxm, kzp, kzm, Exc, Ezc in colors:   # red half-sweep, then black
            t = thf[k]; s = np.sin(t); c = np.cos(t)
            nE = Exc * s + Ezc * c
            torque = cD * nE * (Exc * c - Ezc * s)
            neigh = (thf[kxp] + thf[kxm]) * ix2 + (thf[kzp] + thf[kzm]) * iz2
            t_new = (K * neigh + torque) / (K * S)
            t_upd = (1.0 - omega) * t + omega * t_new
            dmax = max(dmax, float(np.max(np.abs(t_upd - t))))
            thf[k] = t_upd
        res = dmax
        # audit C6-1: the per-sweep UPDATE size under-states the true error of a
        # Gauss-Seidel iterate by 1/(1-rho) (rho = the iteration's convergence factor:
        # measured amplification x200 at nz=41 up to x4340 at nz=161), so breaking --
        # and certifying success -- on `res` alone masked errors up to ~1.5 deg on fine
        # grids. Estimate rho from consecutive update ratios (the standard geometric-
        # tail bound) and gate on the implied ERROR estimate res*rho/(1-rho).
        _res_hist.append(res)
        if len(_res_hist) >= 4 and res > 0.0 and _res_hist[-2] > 0.0 and _res_hist[-3] > 0.0:
            _rho = min(max(0.5 * (_res_hist[-1] / _res_hist[-2]
                                  + _res_hist[-2] / _res_hist[-3]), 0.0), 0.999999)
            err_est = res * _rho / (1.0 - _rho)
        else:
            err_est = res
        if err_est < tol:
            break
    th[:, 0] = theta_b; th[:, -1] = theta_b              # enforce strong anchoring exactly

    n_eff_of_x = np.full(nx, float("nan"))
    if n_o is not None and n_e is not None:
        for i in range(nx):
            n_eff_of_x[i] = n_eff_from_theta_profile(th[i, :], z, n_o, n_e, model=opt_model,
                                                     d_lc=float(d_planar))
    # audit C6-1: success is certified on the geometric-tail ERROR estimate, not the raw
    # update size (which is rho-fold smaller); `residual` now reports the error estimate.
    success = err_est < max(tol, 1e-5)
    return LC2DResult(x_m=x, z_m=z, theta_field_rad=th, Ex=Ex, Ez=Ez, V=V, n_eff_of_x=n_eff_of_x,
                      theta_b_rad=theta_b, iters=it, residual=float(err_est),
                      x_boundary=x_boundary, success=success,
                      message=("ok" if success else
                               "did not reach tol (error estimate {:.2e}, last update "
                               "{:.2e})".format(err_est, res)))
