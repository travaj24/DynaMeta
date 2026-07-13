"""Resolved free-carrier scattering / mass closures for the Drude optical model (roadmap R2):
a Kane nonparabolic optical mass m_opt(n) and a Matthiessen damping Gamma(n; T) = sum of phonon,
ionized-impurity and grain-boundary channels. Both are plain callables-of-n that plug THROUGH the
existing DrudeOptical m_opt_kg / gamma_rad_s callable seam (DrudeOptical itself is unchanged), so the
default constant-Drude behavior is byte-identical when the new knobs are neutral. Pure numpy, SI units;
spell out omega/Gamma/tau. (materials/scattering.py is also the home for the R3 shared ScatteringModel.)

Off-switches:
  KaneOpticalMass(m0_kg=m, alpha_eV=0.0)            -> m_opt(n) == m  exactly (constant).
  MatthiessenGamma(gamma_const_rad_s=g, ...=0)      -> Gamma(n) == g  exactly (constant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from dynameta.constants import Q_E, M_E, HBAR

_N_FLOOR = 1.0e10        # mirror the ITO DOS-mass closure floor (avoid 0**(1/3) edge)


@dataclass(frozen=True)
class KaneOpticalMass:
    """Density-dependent Kane (nonparabolic) optical/conductivity mass.

    For the Kane band E (1 + alpha E) = hbar^2 k^2 / (2 m0), the degenerate optical
    (Drude/plasma) mass is the Fermi-surface value
        m_opt = hbar k_F / v_F = m0 (1 + 2 alpha E_F),
    with k_F = (3 pi^2 n)^(1/3) fixed by counting and E_F the TRUE Kane Fermi level
    solving E_F (1 + alpha E_F) = gamma_F,  gamma_F = hbar^2 k_F^2 / (2 m0).
    The quadratic-inversion identity  1 + 2 alpha E_F = sqrt(1 + 4 alpha gamma_F)
    is what the default computes -- so alpha_eV here is the STANDARD Kane
    nonparabolicity (literature ITO ~0.4 eV^-1), the same alpha the library's
    schrodinger_poisson and carrier_heating closures use.

    audit C7b-1: the previous default evaluated sqrt(1 + 2 alpha gamma_F) -- the Kane
    alpha silently HALVED to leading order (m_opt 8.6-22% low over n = 1e26-2e27 m^-3
    with literature ITO alpha; lambda_ENZ ~4-12% short) -- and its gate/test compared
    against a copy of the same formula. legacy=True keeps the old closure for
    back-compat with alphas CALIBRATED under it (alpha_legacy = 2 alpha_Kane
    reproduces the old numbers exactly, since sqrt(1 + 2*(2a)*g) = sqrt(1 + 4 a g)).

    alpha_eV=0 -> exactly m0 (constant) on both branches. Callable of n -> kg.
    `exponent` is a LEGACY-branch fit knob only (the exact Kane mass has none)."""
    m0_kg: float
    alpha_eV: float = 0.0
    exponent: float = 0.5
    legacy: bool = False

    def __post_init__(self):
        if not self.legacy and self.exponent != 0.5:
            raise ValueError(
                "KaneOpticalMass: `exponent` is a legacy-closure fit knob; the exact Kane "
                "optical mass has no exponent freedom. Pass legacy=True to use the old "
                "(1 + 2 alpha gamma_F)^exponent closure (audit C7b-1).")

    def __call__(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        kF = np.power(3.0 * np.pi ** 2 * n, 1.0 / 3.0)
        g_F = HBAR ** 2 * kF ** 2 / (2.0 * self.m0_kg)                 # J (parabolic-form energy)
        a_g = self.alpha_eV * g_F / Q_E                                # alpha * gamma_F, dimensionless
        if self.legacy:
            return self.m0_kg * np.power(1.0 + 2.0 * a_g, self.exponent)
        return self.m0_kg * np.sqrt(1.0 + 4.0 * a_g)                   # = m0 (1 + 2 alpha E_F)


def _bose(x):
    """Bose occupation 1/(exp(x)-1) = 1/expm1(x); x = T_debye/T."""
    return 1.0 / np.expm1(np.asarray(x, dtype=np.float64))


@dataclass(frozen=True)
class MatthiessenGamma:
    """Matthiessen free-carrier damping Gamma(n) = optical_dc_ratio * (1/tau_gb + 1/tau_phonon(T) +
    1/tau_ii(n)), a callable of carrier density n -> rad/s. Temperature is model STATE (T_K), set by the
    caller (e.g. an electro-thermo loop does dataclasses.replace(gamma, T_K=...)); the per-call T/omega
    signature widening is deferred (roadmap R3/R6). All channels are >= 0 so Gamma >= 0 (passive);
    Gamma is exactly 0 only if EVERY channel is left at its zero default (a deliberate lossless
    configuration -- DrudeOptical accepts gamma = 0 but rejects gamma < 0).

      1/tau_gb       = gamma_const_rad_s                         (grain-boundary + any T,n-independent floor)
      1/tau_phonon   = gamma_phonon_300K_rad_s * f_T             (LO/acoustic phonons; f_T below)
      1/tau_ii(n)    = bh_prefactor_rad_s * (n/bh_n_ref_m3)^p * (m_opt(n)/m_ref)      (degenerate Brooks-
                       Herring RATE scaling; audit 7b: the previous (m_ref/m)^2 factor was the
                       MOBILITY mass law mu_ii ~ 1/m*^2, but the damping RATE 1/tau = q/(m* mu)
                       scales ~ m*^{+1} x a screening log -- a Born probe at ITO parameters showed
                       the true rate RISING +28% over m* = 0.30->0.50 me while the old factor FELL
                       to 0.36, a ~3.5x trend distortion. The absolute prefactor stays
                       CALIBRATION-bearing, default 0 = off; re-fit it after this change.)
      f_T            = (T/300)              if debye_T_K <= 0 (linear high-T)
                       bose(Td/T)/bose(Td/300)  otherwise (LO-phonon Bose occupation)
    """
    gamma_const_rad_s: float = 0.0
    gamma_phonon_300K_rad_s: float = 0.0
    T_K: float = 300.0
    debye_T_K: float = 0.0
    bh_prefactor_rad_s: float = 0.0
    bh_n_ref_m3: float = 1.0e27
    bh_exponent: float = 1.0
    m_opt: Optional[Union[float, "KaneOpticalMass"]] = None
    optical_dc_ratio: float = 1.0

    def __post_init__(self):
        # T_K is mutable model state (set via dataclasses.replace by an electro-thermo loop); guard it
        # here so a non-positive temperature fails loudly rather than producing a divide-warning /
        # negative (gain) phonon rate in the Bose branch.
        if not (self.T_K > 0.0):
            raise ValueError("MatthiessenGamma: T_K must be > 0 (K), got {!r}".format(self.T_K))
        # the optical/DC ratio is a positive scale factor; a non-positive value would flip the whole
        # damping to zero/negative (gain under exp(-i omega t)) through an otherwise-valid channel sum.
        if not (self.optical_dc_ratio > 0.0):
            raise ValueError("MatthiessenGamma: optical_dc_ratio must be > 0, got {!r}".format(
                self.optical_dc_ratio))

    def _phonon(self) -> float:
        if self.gamma_phonon_300K_rad_s <= 0.0:
            return 0.0
        if self.debye_T_K <= 0.0:
            return self.gamma_phonon_300K_rad_s * (self.T_K / 300.0)
        return self.gamma_phonon_300K_rad_s * float(_bose(self.debye_T_K / self.T_K) /
                                                    _bose(self.debye_T_K / 300.0))

    def _m_opt(self, n):
        if self.m_opt is None:
            return M_E
        return self.m_opt(n) if callable(self.m_opt) else float(self.m_opt)

    def _ii(self, n):
        if self.bh_prefactor_rad_s <= 0.0:
            return 0.0
        m = self._m_opt(n)
        m_ref = self._m_opt(self.bh_n_ref_m3)
        # audit 7b: RATE ~ m*^{+1} (1/tau = q/(m* mu), mu_ii ~ 1/m*^2), not the old
        # mobility-law (m_ref/m)^2 which moved the channel the WRONG WAY with mass
        return (self.bh_prefactor_rad_s * np.power(n / self.bh_n_ref_m3, self.bh_exponent)
                * (m / m_ref))

    def __call__(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), _N_FLOOR)
        return self.optical_dc_ratio * (self.gamma_const_rad_s + self._phonon() + self._ii(n))


@dataclass(frozen=True)
class ScatteringModel:
    """ONE momentum-relaxation law tau(n;T) shared by BOTH the optical Drude damping and the transport
    mobility (roadmap R3), removing the hidden inconsistency of fitting them independently. The shared
    quantity is 1/tau (rad/s), supplied as `one_over_tau` -- typically a MatthiessenGamma (R2) so the
    SAME density/temperature scattering law feeds both sides:

        optical Drude gamma(n) = 1/tau(n)                          (mass enters wp, NOT gamma)
        drift mobility  mu(n)  = hall_factor * q / (m_cond(n) * 1/tau(n))   [m^2/Vs]

    Build the link around tau, never around mu -- writing gamma = q/(m_opt mu) is circular (it just gives
    1/tau back). CAVEATS, encoded as separate inputs so they are not conflated:
      - m_cond is the CONDUCTIVITY (DC) effective mass, DISTINCT from the optical mass m_opt (which sets
        wp) and from the DOS mass; in a nonparabolic conductor all three differ.
      - hall_factor r_H: a measured Hall mobility is r_H * drift mobility; the tau<->mu link uses the
        DRIFT mobility, so set hall_factor only if you want the produced mu to be a Hall mobility
        (default 1.0 = drift). A measured Hall mu fed back to calibrate tau should be divided by r_H.
    Pure numpy; attach to a Material via Material(scattering=...) (opt-in; default unset = byte-identical).
    """
    one_over_tau: Union[float, Callable]
    m_cond_kg: Union[float, Callable] = M_E
    hall_factor: float = 1.0

    def inv_tau(self, n_m3):
        ot = self.one_over_tau
        return np.asarray(ot(n_m3), dtype=np.float64) if callable(ot) else float(ot)

    def tau_s(self, n_m3):
        return 1.0 / self.inv_tau(n_m3)

    def gamma_optical_of_n(self):
        """Optical Drude gamma callable gamma(n) = 1/tau(n) (reuses the MatthiessenGamma directly)."""
        if callable(self.one_over_tau):
            return self.one_over_tau
        g = float(self.one_over_tau)
        return lambda n: g

    def mobility_of_n(self):
        """Drift mobility callable mu(n) = hall_factor * q / (m_cond(n) * 1/tau(n)) [m^2/Vs]."""
        ot, mc, rh = self.one_over_tau, self.m_cond_kg, self.hall_factor

        def mu(n_m3):
            n = np.asarray(n_m3, dtype=np.float64)
            m = np.asarray(mc(n), dtype=np.float64) if callable(mc) else float(mc)
            g = np.asarray(ot(n), dtype=np.float64) if callable(ot) else float(ot)
            if np.any(np.asarray(m) <= 0.0) or np.any(np.asarray(g) <= 0.0):   # mu ~ 1/(m * 1/tau)
                raise ValueError("ScatteringModel.mobility_of_n: m_cond and 1/tau must be > 0 (a "
                                 "callable returned a non-positive value -> mu would be inf/NaN/negative).")
            return rh * Q_E / (m * g)
        return mu
