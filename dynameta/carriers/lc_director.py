"""
Liquid-crystal director driver for reconfigurable (LC) metasurfaces (roadmap Phase 4b). A nematic
cell of thickness d with PLANAR anchoring (director in the plate plane, tilt theta = 0, at both
surfaces z = 0 and z = d) and a voltage V across it. For a positive dielectric anisotropy
(dEps_static = eps_par - eps_perp > 0) the director tilts toward the field (z) above the
Freedericksz threshold

    V_th = pi * sqrt(K / (eps0 * dEps_static))                 (1-constant elastic, INDEPENDENT of d)

It PRODUCES the equilibrium tilt profile theta(z); a LiquidCrystalModel (core/effects) turns the
director into a UNIAXIAL optical tensor eps along n-hat = (cos theta, 0, sin theta) (reusing the
tensor-eps FEM path, incl. the off-diagonal UPML solve for an intermediate tilt). The bridge does
not yet auto-assemble the director field -- the driver produces theta for the caller to place in the
bundle (a tracked seam; the Freedericksz threshold + uniaxial response are validated in
validation/reconfigurable_modulators.py and tests/test_lc_director.py).

Method: the 1-constant Frank torque balance has a first integral (1/2 K theta'^2 = g(theta) -
g(theta_m), g = D^2/(2 eps0 eps(theta)), eps(theta) = eps_perp + dEps_static sin^2 theta) whose
midplane tilt theta_m fixes the voltage through a single non-singular elliptic quadrature

    V(theta_m) = 2 sqrt(K/eps0) * B(theta_m),
    B = int_0^{pi/2} sqrt( eps(theta_m) / (dEps * eps(theta)) ) / sqrt(1 - sin^2(theta_m) sin^2 phi) dphi,

with sin(theta) = sin(theta_m) sin(phi) (the substitution that removes the turning-point
singularity). V(theta_m) increases monotonically from V_th (theta_m -> 0) to infinity, so theta_m
is recovered by bisection -- robust right through the pitchfork (where a fixed-seed energy
minimizer stalls on the quartic-flat landscape). Below V_th there is no tilted solution
(theta = 0). SI units (K in J/m i.e. N; eps_* are STATIC relative permittivities; V in volts;
d in metres). Pure numpy/scipy; no devsim/ngsolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import EPS0
from dynameta.core.numerics import trapz


def freedericksz_threshold_V(K_elastic: float, dEps_static: float) -> float:
    """Freedericksz threshold voltage V_th = pi sqrt(K/(eps0 dEps)) for a planar nematic cell with
    positive dielectric anisotropy (independent of cell thickness). Raises for dEps <= 0 (no
    threshold instability -- the field does not tilt a negative/zero-anisotropy planar cell)."""
    if not (K_elastic > 0.0):
        raise ValueError("K_elastic must be > 0")
    if not (dEps_static > 0.0):
        raise ValueError("dEps_static must be > 0 (positive anisotropy required for a planar-cell "
                         "Freedericksz transition)")
    return float(np.pi * np.sqrt(float(K_elastic) / (EPS0 * float(dEps_static))))


@dataclass
class DirectorProfile:
    z_m: np.ndarray         # grid through the cell, 0..d
    theta_rad: np.ndarray   # director tilt from the plate plane (0 = planar, pi/2 = homeotropic)
    theta_max_rad: float    # midplane (maximum) tilt
    V_applied: float        # applied voltage
    V_th: float             # Freedericksz threshold


def _B(theta_m: float, dEps: float, ep: float, nphi: int = 1201) -> float:
    """The elliptic quadrature B(theta_m) (see module docstring); bounded for theta_m < pi/2."""
    s2 = np.sin(theta_m) ** 2
    phi = np.linspace(0.0, 0.5 * np.pi, int(nphi))
    sp2 = np.sin(phi) ** 2
    eps_th = ep + dEps * s2 * sp2                          # eps(theta(phi))
    eps_thm = ep + dEps * s2                               # eps(theta_m)
    integrand = np.sqrt(eps_thm / (dEps * eps_th)) / np.sqrt(1.0 - s2 * sp2)
    return float(trapz(integrand, phi))


def _V_of_theta_m(theta_m: float, K: float, dEps: float, ep: float) -> float:
    return 2.0 * np.sqrt(K / EPS0) * _B(theta_m, dEps, ep)


def director_profile(K_elastic: float, dEps_static: float, eps_perp: float, d_m: float,
                      applied_V: float, *, nz: int = 201) -> DirectorProfile:
    """Equilibrium director tilt theta(z) for a planar nematic cell under `applied_V`. Below V_th
    theta = 0 (planar); above, the midplane tilt theta_m is recovered by bisecting the monotone
    V(theta_m) relation, and theta(z) is reconstructed from the parametric turning-point map.
    Returns a DirectorProfile."""
    K = float(K_elastic); dEps = float(dEps_static); ep = float(eps_perp)
    d = float(d_m); V = float(applied_V)
    if not (K > 0 and dEps > 0 and ep > 0 and d > 0 and int(nz) >= 11):
        raise ValueError("require K>0, dEps>0, eps_perp>0, d>0, nz>=11")
    Vth = freedericksz_threshold_V(K, dEps)
    z = np.linspace(0.0, d, int(nz))
    if V <= Vth * (1.0 + 1e-9):                            # below threshold: planar (no tilt)
        return DirectorProfile(z_m=z, theta_rad=np.zeros_like(z), theta_max_rad=0.0,
                               V_applied=V, V_th=Vth)
    # bisect the monotone V(theta_m) for theta_m in (0, pi/2)
    lo, hi = 1e-6, 0.5 * np.pi - 1e-6
    if _V_of_theta_m(hi, K, dEps, ep) < V:                # extreme over-drive: saturate near homeotropic
        theta_m = hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _V_of_theta_m(mid, K, dEps, ep) < V:
                lo = mid
            else:
                hi = mid
        theta_m = 0.5 * (lo + hi)
    # reconstruct theta(z): the cumulative A(phi) maps phi in [0, pi/2] to z in [0, d/2]
    s = np.sin(theta_m); s2 = s * s
    phi = np.linspace(0.0, 0.5 * np.pi, 4001)
    th_phi = np.arcsin(np.clip(s * np.sin(phi), -1.0, 1.0))
    eps_th = ep + dEps * np.sin(th_phi) ** 2
    eps_thm = ep + dEps * s2
    integ = np.sqrt(eps_th * eps_thm / dEps) / np.sqrt(1.0 - s2 * np.sin(phi) ** 2)
    A = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(phi))])
    z_half = A / A[-1] * (0.5 * d)                        # z from 0 (surface) to d/2 (midplane)
    nh = int(nz) // 2
    zu = np.linspace(0.0, 0.5 * d, nh + 1)
    th_u = np.interp(zu, z_half, th_phi)                  # tilt on the half grid
    theta_full = np.concatenate([th_u, th_u[-2::-1]])     # mirror about the midplane
    z_full = np.concatenate([zu, d - zu[-2::-1]])
    theta = np.interp(z, z_full, theta_full)              # onto the requested uniform z grid
    return DirectorProfile(z_m=z, theta_rad=theta, theta_max_rad=float(np.max(theta)),
                           V_applied=V, V_th=Vth)
