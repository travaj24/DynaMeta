"""REL2: NBTI/PBTI bias-temperature instability -- the PARAMETRIC companion to TDDB. Sustained gate
bias + temperature drifts the flat-band/threshold voltage, creeping the bias needed to reach the ENZ
accumulation point (the modulator de-tunes without ever shorting).

Model: the standard power-law closure (reaction-diffusion limit):

    dVth(t) = A * D^p_duty * (E_ox / E_ref)^gamma * t^n * exp(-Ea / (kB * T))

with n ~ 1/6 (H2-diffusion-limited; up to ~0.25 process-dependent), Ea ~ 0.1-0.15 eV (NOTE the SIGN:
exp(-Ea/kBT) is the activated PROCESS rate, monotonically INCREASING with T -- hotter degrades
faster; the resulting LIFETIME falls with T), gamma ~ 0.3-0.4 (field acceleration), and an AC
duty-cycle weighting D^p_duty (p_duty ~ 0.5; D = 1 is DC). Time-to-spec inverts the law:

    t_fail = ( dVth_max / (A * D^p_duty * (E/E_ref)^gamma * exp(-Ea/kBT)) )^(1/n)

A is calibration-bearing (anchor on one measured (t, E, T, dVth) point via BtiParams.calibrated()).
Recovery during the off-phase is FOLDED into the duty factor at this level; an explicit
stress/relax state model is the documented follow-on. Pure numpy; oracles in
validation/reliability_bti.py (log-slope == n, numeric-root inversion cross-check, sane 10-year band).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

KB_EV_K = 8.617333262e-5


def _check(A_V, n_exp, gamma, Ea_eV, duty, p_duty, E_ref_V_m):
    if not (A_V > 0.0):
        raise ValueError("BTI: A_V must be > 0")
    if not (0.0 < n_exp < 1.0):
        raise ValueError("BTI: time exponent n must be in (0, 1) (typ. 1/6..0.25)")
    if gamma < 0.0:
        raise ValueError("BTI: field exponent gamma must be >= 0")
    if Ea_eV < 0.0:
        raise ValueError("BTI: Ea_eV must be >= 0")
    if not (0.0 < duty <= 1.0):
        raise ValueError("BTI: duty must be in (0, 1] (1 = DC)")
    if p_duty < 0.0:
        raise ValueError("BTI: p_duty must be >= 0")
    if not (E_ref_V_m > 0.0):
        raise ValueError("BTI: E_ref_V_m must be > 0")


def dvth_power_law(t_s, E_ox_V_m, T_K, *, A_V: float, n_exp: float = 1.0 / 6.0,
                   gamma: float = 0.35, Ea_eV: float = 0.12, E_ref_V_m: float = 1.0e8,
                   duty: float = 1.0, p_duty: float = 0.5):
    """Threshold/flat-band drift dVth(t) [V] (see module docstring). Broadcasts over t/E/T arrays;
    t = 0 -> exactly 0 (no drift before stress)."""
    _check(A_V, n_exp, gamma, Ea_eV, duty, p_duty, E_ref_V_m)
    t = np.asarray(t_s, dtype=np.float64)
    if np.any(t < 0.0):
        raise ValueError("BTI: t_s must be >= 0")
    E = np.asarray(E_ox_V_m, dtype=np.float64)
    if np.any(E < 0.0):
        raise ValueError("BTI: E_ox_V_m must be >= 0 (use the field magnitude)")
    T = np.asarray(T_K, dtype=np.float64)
    if np.any(T <= 0.0):
        raise ValueError("BTI: T_K must be > 0")
    pref = A_V * duty ** p_duty * (E / E_ref_V_m) ** gamma * np.exp(-Ea_eV / (KB_EV_K * T))
    return pref * t ** n_exp


def time_to_dvth(dvth_max_V: float, E_ox_V_m: float, T_K: float, *, A_V: float,
                 n_exp: float = 1.0 / 6.0, gamma: float = 0.35, Ea_eV: float = 0.12,
                 E_ref_V_m: float = 1.0e8, duty: float = 1.0, p_duty: float = 0.5) -> float:
    """Time [s] for the drift to reach the spec limit dVth_max (the BTI parametric lifetime)."""
    _check(A_V, n_exp, gamma, Ea_eV, duty, p_duty, E_ref_V_m)
    if not (dvth_max_V > 0.0):
        raise ValueError("BTI: dvth_max_V must be > 0")
    if not (E_ox_V_m > 0.0):
        raise ValueError("BTI: E_ox_V_m must be > 0 for a finite lifetime")
    pref = A_V * duty ** p_duty * (E_ox_V_m / E_ref_V_m) ** gamma \
        * float(np.exp(-Ea_eV / (KB_EV_K * float(T_K))))
    return float((dvth_max_V / pref) ** (1.0 / n_exp))


@dataclass(frozen=True)
class BtiParams:
    """Power-law BTI parameter set; A_V is calibration-bearing (prefer .calibrated())."""
    A_V: float = 1.0
    n_exp: float = 1.0 / 6.0
    gamma: float = 0.35
    Ea_eV: float = 0.12
    E_ref_V_m: float = 1.0e8
    duty: float = 1.0
    p_duty: float = 0.5

    def dvth_V(self, t_s, E_ox_V_m, T_K):
        return dvth_power_law(t_s, E_ox_V_m, T_K, A_V=self.A_V, n_exp=self.n_exp, gamma=self.gamma,
                              Ea_eV=self.Ea_eV, E_ref_V_m=self.E_ref_V_m, duty=self.duty,
                              p_duty=self.p_duty)

    def time_to_s(self, dvth_max_V, E_ox_V_m, T_K):
        return time_to_dvth(dvth_max_V, E_ox_V_m, T_K, A_V=self.A_V, n_exp=self.n_exp,
                            gamma=self.gamma, Ea_eV=self.Ea_eV, E_ref_V_m=self.E_ref_V_m,
                            duty=self.duty, p_duty=self.p_duty)

    @classmethod
    def calibrated(cls, *, t_s: float, E_ox_V_m: float, T_K: float, dvth_V: float,
                   **kw) -> "BtiParams":
        """Anchor A_V on ONE measured stress point (t, E, T, dVth)."""
        base = cls(A_V=1.0, **kw)
        d_unit = float(base.dvth_V(t_s, E_ox_V_m, T_K))      # drift with A = 1
        if not (dvth_V > 0.0 and d_unit > 0.0):
            raise ValueError("BTI: calibration point must give a positive drift")
        return replace(base, A_V=dvth_V / d_unit)
