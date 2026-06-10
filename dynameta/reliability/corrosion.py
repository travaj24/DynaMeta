"""REL9: environmental corrosion / oxidation / humidity life -- GENERAL packaging-level mechanisms
(low priority for a hermetically-sealed photonic chip, real for unencapsulated test structures).

Deal-Grove oxide growth (metal/Si oxidation kinetics):

    x^2 + A x = B (t + tau)  =>  x(t) = (A/2) (sqrt(1 + 4 B (t + tau)/A^2) - 1)
    thin/short-time LINEAR limit  x ~ (B/A) t;  thick/long-time PARABOLIC limit  x ~ sqrt(B t)
    (B and B/A are Arrhenius in T; tau = (x0^2 + A x0)/B carries any initial oxide x0.)

Peck humidity-life (electrochemical migration / CAF class):

    t_fail = A_p * RH^(-n) * exp(Ea / (kB T)),   n ~ 2.5-3, Ea ~ 0.7-0.9 eV

DRIVER NOTE: ambient humidity / contamination are EXTERNAL inputs (not computable from the operating
solve) -- the roadmap-flagged boundary. Pure numpy/scipy; oracles in
validation/reliability_corrosion.py (closed form vs the rate ODE dx/dt = B/(2x + A), the two limits,
and the Peck acceleration ratios).
"""

from __future__ import annotations

import numpy as np

KB_EV_K = 8.617333262e-5


def deal_grove_thickness_m(t_s, *, A_m: float, B_m2_s: float, x0_m: float = 0.0):
    """Oxide thickness x(t) from the Deal-Grove closed form (see module docstring). Broadcasts over
    t arrays; t = 0 -> x0 exactly."""
    if not (A_m > 0.0 and B_m2_s > 0.0):
        raise ValueError("Deal-Grove: A and B must be > 0")
    if x0_m < 0.0:
        raise ValueError("Deal-Grove: x0 must be >= 0")
    t = np.asarray(t_s, dtype=np.float64)
    if np.any(t < 0.0):
        raise ValueError("Deal-Grove: t must be >= 0")
    tau = (x0_m ** 2 + A_m * x0_m) / B_m2_s
    return (A_m / 2.0) * (np.sqrt(1.0 + 4.0 * B_m2_s * (t + tau) / A_m ** 2) - 1.0)


def deal_grove_rate_arrhenius(T_K: float, *, B0_m2_s: float, Ea_B_eV: float,
                              BA0_m_s: float, Ea_BA_eV: float):
    """The Arrhenius rate constants at T: returns (B, B/A) -> A = B / (B/A). Typical Si/SiO2 (dry):
    Ea_B ~ 1.23 eV (parabolic, O2 diffusion), Ea_B/A ~ 2.0 eV (linear, interface reaction)."""
    if not (T_K > 0.0 and B0_m2_s > 0.0 and BA0_m_s > 0.0):
        raise ValueError("Deal-Grove Arrhenius: T, B0, (B/A)0 must be > 0")
    B = B0_m2_s * np.exp(-Ea_B_eV / (KB_EV_K * T_K))
    BA = BA0_m_s * np.exp(-Ea_BA_eV / (KB_EV_K * T_K))
    return float(B), float(BA)


def peck_time_to_failure_s(RH_percent, T_K, *, A_s: float, n_exp: float = 2.7,
                           Ea_eV: float = 0.8):
    """Peck humidity life t_fail = A * RH^(-n) * exp(Ea/kBT) (RH in percent, 0 < RH <= 100).
    Broadcasts over RH/T arrays."""
    if not (A_s > 0.0):
        raise ValueError("Peck: A_s must be > 0")
    if not (n_exp > 0.0):
        raise ValueError("Peck: humidity exponent n must be > 0 (typ. 2.5-3)")
    if Ea_eV < 0.0:
        raise ValueError("Peck: Ea_eV must be >= 0")
    RH = np.asarray(RH_percent, dtype=np.float64)
    if np.any(RH <= 0.0) or np.any(RH > 100.0):
        raise ValueError("Peck: RH must be in (0, 100] percent")
    T = np.asarray(T_K, dtype=np.float64)
    if np.any(T <= 0.0):
        raise ValueError("Peck: T_K must be > 0")
    return A_s * RH ** (-n_exp) * np.exp(Ea_eV / (KB_EV_K * T))


def peck_af(*, RH_use: float, RH_stress: float, T_use_K: float, T_stress_K: float,
            n_exp: float = 2.7, Ea_eV: float = 0.8) -> float:
    """Peck acceleration factor AF = t_use / t_stress = (RH_stress/RH_use)^n * exp((Ea/kB)
    (1/T_use - 1/T_stress)) (> 1 for a harsher stress chamber: 85/85 vs office ambient)."""
    return float((peck_time_to_failure_s(RH_use, T_use_K, A_s=1.0, n_exp=n_exp, Ea_eV=Ea_eV)
                  / peck_time_to_failure_s(RH_stress, T_stress_K, A_s=1.0, n_exp=n_exp,
                                           Ea_eV=Ea_eV)))
