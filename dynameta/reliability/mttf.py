"""REL10: acceleration factors + system-MTTF aggregation -- the umbrella that makes the per-mechanism
lifetimes comparable, extrapolates accelerated-stress measurements to use conditions, and aggregates
them into one device lifetime / FIT.

    AF_T = exp( (Ea / kB) * (1/T_use - 1/T_stress) )          (Arrhenius; note the PARENTHESIZATION:
                                                               (Ea/kB)*(1/Tu - 1/Ts), NOT
                                                               Ea/(kB*(1/Tu - 1/Ts)) which overflows)
    AF_X = exp( gamma * (X_stress - X_use) )                  (field/voltage exponential acceleration)
    MTTF_use = MTTF_stress * AF
    1/MTTF_sys = sum_i 1/MTTF_i                               (competing risks, exponential lives;
                                                               immortal mechanisms = inf are dropped)
    t63_min(N)  = t63_single / N^(1/beta)                     (the earliest failure of N iid Weibull
                                                               elements -- an array's weakest link)
    FIT = 1e9 / MTTF_hours

Pure numpy; oracles in validation/reliability_mttf.py include MONTE CARLO competing-risks and
order-statistics cross-checks (independent numeric references, not algebraic self-inversions).
"""

from __future__ import annotations

import numpy as np

from dynameta.constants import KB_EV_K   # eV/K, single source (audit 6.3)


def arrhenius_af(Ea_eV: float, T_use_K: float, T_stress_K: float) -> float:
    """Arrhenius acceleration factor AF = MTTF_use / MTTF_stress = exp((Ea/kB)(1/T_use - 1/T_stress)).

    Ea may be SIGNED (audit C5-9): Ea > 0 = the usual thermally-activated aging (AF > 1
    when the stress is HOTTER than use); Ea < 0 = cold-worse mechanisms -- REL8/HCI ships
    a documented-physical default Ea = -0.1 eV, for which AF > 1 requires stressing
    COLDER than use. The formula is exact for signed Ea (matches hci.py's own ratio to
    machine precision); the old Ea >= 0 guard blocked the REL10-over-REL8 extrapolation
    outright, and the natural |Ea| workaround silently returned the RECIPROCAL AF
    (probe: 0.266 where 3.77 is correct -- a 14.2x silent error)."""
    if not np.isfinite(Ea_eV):
        raise ValueError("arrhenius_af: Ea_eV must be finite (signed is allowed; see docstring)")
    if not (T_use_K > 0.0 and T_stress_K > 0.0):
        raise ValueError("arrhenius_af: temperatures must be > 0 K")
    return float(np.exp((Ea_eV / KB_EV_K) * (1.0 / T_use_K - 1.0 / T_stress_K)))


def field_af(gamma_per_unit: float, X_use: float, X_stress: float) -> float:
    """Exponential field/voltage acceleration AF = exp(gamma * (X_stress - X_use)) (> 1 when the
    stress drive exceeds use). X in whatever unit gamma is per (V, MV/cm, ...)."""
    if gamma_per_unit < 0.0:
        raise ValueError("field_af: gamma must be >= 0")
    return float(np.exp(gamma_per_unit * (X_stress - X_use)))


def mttf_use_from_stress(mttf_stress_s: float, af: float) -> float:
    """Extrapolate a measured stress-condition MTTF to use conditions: MTTF_use = MTTF_stress * AF."""
    if not (mttf_stress_s > 0.0) or not (af > 0.0):
        raise ValueError("mttf_use_from_stress: MTTF and AF must be > 0")
    return float(mttf_stress_s * af)


def system_mttf(mttfs_s) -> float:
    """Competing-risks (series / weakest-link) system MTTF for EXPONENTIAL per-mechanism lives:
    1/MTTF_sys = sum_i 1/MTTF_i. Immortal mechanisms (inf, e.g. Blech-immortal EM) drop out; an
    all-immortal system returns inf."""
    m = np.asarray(mttfs_s, dtype=np.float64)
    if m.size == 0:
        raise ValueError("system_mttf: need at least one mechanism MTTF")
    if np.any(m <= 0.0) or np.any(np.isnan(m)):
        raise ValueError("system_mttf: every MTTF must be > 0 (inf = immortal is allowed)")
    rates = np.where(np.isinf(m), 0.0, 1.0 / m)
    tot = float(np.sum(rates))
    return float("inf") if tot == 0.0 else 1.0 / tot


def fit_per_1e9_hours(mttf_s: float) -> float:
    """Failures-in-time: FIT = 1e9 / MTTF[hours] (0 for an immortal mechanism)."""
    if not (mttf_s > 0.0):
        raise ValueError("fit_per_1e9_hours: MTTF must be > 0")
    return 0.0 if np.isinf(mttf_s) else float(1.0e9 / (mttf_s / 3600.0))


def weibull_earliest_t63(t63_single_s: float, n_elements: int, beta: float) -> float:
    """Characteristic (63.2%) life of the FIRST failure among n iid Weibull(beta, t63_single)
    elements: the minimum of n iid Weibulls is Weibull(beta, t63_single / n^(1/beta)) EXACTLY -- the
    array weakest-link statistic (an N-pixel metasurface fails at its first pixel)."""
    if not (t63_single_s > 0.0):
        raise ValueError("weibull_earliest_t63: t63 must be > 0")
    if not (isinstance(n_elements, (int, np.integer)) and n_elements >= 1):
        raise ValueError("weibull_earliest_t63: n_elements must be an integer >= 1")
    if not (beta > 0.0):
        raise ValueError("weibull_earliest_t63: Weibull shape beta must be > 0")
    return float(t63_single_s / n_elements ** (1.0 / beta))
