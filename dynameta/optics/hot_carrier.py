"""
Per-cell hot-carrier two-temperature ADE parameters + precomputed lookup tables (roadmap 2.1).

The 0-D physics OWNER is carriers.carrier_heating; this module REUSES its coefficient functions
(kane_mass_of_Te, gamma_of_Te, TwoTempParams / the C_e integral) to build per-material
m*(T_e) -> wp(T_e) and gamma(T_e) RATIO tables that the 2D-TE NumPy FDTD reference kernel
(optics.fdtd_nd.kernels2d.run_2d_te) interpolates per heated Drude cell. No physics is duplicated
here -- only the table assembly + the SI energy bookkeeping the field solver needs.

Physics (Alam / De Leon / Boyd, Science 352:795 (2016) class): an absorbed optical pump heats the
Drude electron gas LOCALLY, where the field concentrates,
    dU_e/dt = p_abs(r,t) - G (T_e - T_l),    p_abs = J_drude . E     (Drude Joule dissipation),
and T_e is recovered from the electron energy density U_e via the electron heat-capacity integral
    U_e(T_e) = integral_{T_e0}^{T_e} C_e(T') dT'                       (inverted by interpolation).
The hot electrons climb the nonparabolic Kane band so <m*(T_e)> RISES -> wp^2 ~ n/<m*> DROPS
(Re(eps) moves toward eps_inf, through ENZ) and scatter more so gamma(T_e) RISES. wp and gamma are
carried as material-only RATIO tables anchored at the cold electron temperature T_e0, so a cell's
cold (wp0, gamma0) reproduce EXACTLY at T_e = T_e0 (off-switch consistency):
    wp(T_e)    = wp0    * sqrt(<m*(T_e0)> / <m*(T_e)>),
    gamma(T_e) = gamma0 * (T_e / T_e0)^p.
Off-switches:
    alpha_per_eV == 0 -> <m*> == m0 constant -> wp ratio == 1 (no plasma shift, exact);
    gamma_p == 0      -> gamma ratio == 1 (no damping shift, exact).
Either collapses the Drude cell to its cold (wp0, gamma0) at every T_e.

TIER SCOPE: the lattice temperature T_l is a FIXED bath (the C_l -> infinity limit of the two-
temperature model); a co-evolving lattice (finite C_l) is a documented follow-on. The uniform-film
UNIFORMITY oracle therefore drives carrier_heating.two_temperature_response with a large C_l (so its
T_l stays pinned) and the SAME absorbed-power-density history extracted from the FDTD run.

Pure numpy/scipy; SI; exp(-i omega t), Im(eps) > 0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from dynameta.constants import T_REF
from dynameta.carriers.carrier_heating import (TwoTempParams, gamma_of_Te,
                                               kane_mass_of_Te)

__all__ = ["HotCarrierParams", "build_hot_carrier_tables"]


@dataclass(frozen=True)
class HotCarrierParams:
    """Opt-in per-material hot-carrier two-temperature parameters for a Drude FDTDLayer.

    `ttm` is a carriers.carrier_heating.TwoTempParams REUSED verbatim as the electron-gas energy
    model: C_e(T_e) (float or callable, e.g. gamma_e*T_e for a degenerate gas), C_l, G_e_l
    [W/m^3/K] and alpha_abs (the absorbed-power coupling; default 1 because p_abs = J.E is already a
    volumetric power density W/m^3). `n_m3`, `m0_kg`, `alpha_per_eV` set the Kane band average that
    fixes the m*(T_e) -> wp(T_e) shift; the cell's COLD (wp0, gamma0) come from the FDTDLayer's own
    drude_wp_rad_s / drude_gamma_rad_s and are the anchor at T_e = T_e0.

    Fields (SI):
      ttm           two-temperature-model coefficients (REUSED; C_l unused this fixed-bath tier)
      n_m3          carrier density (fixed; sets E_F for the Kane band average)
      m0_kg         band-edge effective mass
      alpha_per_eV  Kane nonparabolicity (0 -> m* == m0, no plasma shift)
      T_l_K         lattice bath temperature (FIXED this tier)
      T_e0_K        cold electron temperature -- the wp0/gamma0 anchor (U_e == 0 here)
      gamma_p       gamma(T_e) = gamma0 (T_e/T_e0)^gamma_p exponent (0 -> no damping shift)
      mass_exponent kane_mass_of_Te exponent
      g_s, g_v      spin / valley degeneracy for the Kane Fermi energy
      n_update      FDTD steps between (wp, gamma) refreshes from the T_e tables (>= 1; 1 = every step)
      Te_max_K      upper bound of the T_e lookup table (interp clamps above it -> wp/gamma saturate)
      n_table       lookup-table resolution
    """
    ttm: TwoTempParams
    n_m3: float
    m0_kg: float
    alpha_per_eV: float = 0.0
    T_l_K: float = T_REF
    T_e0_K: float = T_REF
    gamma_p: float = 1.0
    mass_exponent: float = 1.0
    g_s: int = 2
    g_v: int = 1
    n_update: int = 1
    Te_max_K: float = 6000.0
    n_table: int = 1024

    def __post_init__(self):
        if self.n_m3 <= 0.0 or self.m0_kg <= 0.0:
            raise ValueError("HotCarrierParams: n_m3 and m0_kg must be > 0")
        if self.T_e0_K <= 0.0 or self.Te_max_K <= self.T_e0_K:
            raise ValueError("HotCarrierParams: require 0 < T_e0_K < Te_max_K")
        if self.T_l_K <= 0.0:
            raise ValueError("HotCarrierParams: T_l_K must be > 0")
        if int(self.n_table) < 8:
            raise ValueError("HotCarrierParams: n_table must be >= 8")
        if int(self.n_update) < 1:
            raise ValueError("HotCarrierParams: n_update must be >= 1")
        if self.ttm.G_e_l < 0.0:
            raise ValueError("HotCarrierParams: ttm.G_e_l must be >= 0")


def build_hot_carrier_tables(hc: HotCarrierParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the (Te_grid, U_grid, wp_ratio, gam_ratio) lookup arrays for ONE material, calling the
    carrier_heating 0-D coefficient functions ONCE (never per step, never per cell). Returns arrays for
    np.interp:
      Te_grid  (n_table,)  strictly increasing, T_e0 .. Te_max,
      U_grid   (n_table,)  electron energy density above T_e0 [J/m^3], strictly increasing:
                           U(T_e) = integral_{T_e0}^{T_e} C_e(T') dT' (cumulative trapezoid; U[0] = 0),
      wp_ratio (n_table,)  = sqrt(<m*(T_e0)>/<m*(T_e)>)  (multiplies the cell's cold wp0),
      gam_ratio(n_table,)  = (T_e/T_e0)^gamma_p           (multiplies the cell's cold gamma0).
    U_grid[0] == 0 at T_e0, so an unpumped (U_e == 0) cell maps to (wp0, gamma0) EXACTLY; the interp
    inversion T_e(U_e) then feeds wp/gamma."""
    Te = np.linspace(float(hc.T_e0_K), float(hc.Te_max_K), int(hc.n_table))
    # electron heat capacity C_e(T_e) sampled from the REUSED TwoTempParams (constant or callable)
    Ce = np.array([float(hc.ttm.C_e_of(float(t))) for t in Te], dtype=np.float64)
    if np.any(Ce <= 0.0):
        raise ValueError("build_hot_carrier_tables: C_e(T_e) must be > 0 across the table range")
    # U(T_e) = integral_{T_e0}^{T_e} C_e dT' by cumulative trapezoid (strictly increasing since C_e > 0)
    U = np.concatenate(([0.0], np.cumsum(0.5 * (Ce[1:] + Ce[:-1]) * np.diff(Te))))
    # <m*(T_e)> from the Kane band average (carriers.carrier_heating); the high-T tail of the TABLE can
    # exceed the Sommerfeld validity edge (kB*Te > E_F) -- that is a table-construction detail, not a
    # physics event (the warning fires at RUNTIME only if a cell actually heats that far, which the interp
    # does not re-trigger), so suppress the per-sample warning during the one-shot precompute.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m_ref = float(kane_mass_of_Te(hc.m0_kg, hc.alpha_per_eV, hc.n_m3, hc.T_e0_K,
                                      g_s=hc.g_s, g_v=hc.g_v, exponent=hc.mass_exponent))
        m_star = np.broadcast_to(
            np.asarray(kane_mass_of_Te(hc.m0_kg, hc.alpha_per_eV, hc.n_m3, Te,
                                       g_s=hc.g_s, g_v=hc.g_v, exponent=hc.mass_exponent),
                       dtype=np.float64), Te.shape).astype(np.float64)
    wp_ratio = np.sqrt(m_ref / m_star)                       # wp ~ 1/sqrt(<m*>) -> DROPS as T_e rises
    gam_ratio = np.broadcast_to(
        np.asarray(gamma_of_Te(1.0, Te, p=hc.gamma_p, T_ref_K=hc.T_e0_K), dtype=np.float64),
        Te.shape).astype(np.float64)
    return Te, U, wp_ratio, gam_ratio
