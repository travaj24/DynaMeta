"""Field- and density-dependent carrier mobility mu(E, N): Caughey-Thomas / Canali velocity saturation
times an optional Masetti ionized-impurity low-field model. This is the SINGLE source of the closure --
a pure-numpy evaluator (for unit tests and analytic sigma references) AND a DEVSIM edge-model string
emitter that shares the same coefficients, so the two forms cannot drift (mirrors carriers/einstein.py).
Pure numpy, no devsim; SI units throughout (m^2/Vs, m^-3, m/s, V/m). (roadmap R1)

  Caughey-Thomas (field):   mu(E) = mu_low / (1 + (mu_low |E| / v_sat)^b)^(1/b)
  Masetti (low-field, N):   mu_low(N) = mu_min1*exp(-Pc/N) + (mu_max - mu_min2)/(1 + (N/Cr)^a)
                                        - mu1 / (1 + (Cs/N)^be)

The velocity-saturation factor -> 1 as E -> 0 (recovers mu_low exactly) and the drift velocity
mu(E) E -> v_sat as E -> inf (current saturates). The DEVSIM emitter uses a SMOOTHED magnitude
|E| = sqrt(E^2 + E_smooth^2) rather than abs(): abs() has a kink at E=0 (the near-equilibrium edge
operating point) whose discontinuous Jacobian stalls Newton, and it also keeps pow(.,1/b) off the
negative branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MasettiParams:
    """Masetti low-field mobility coefficients (SI). Defaults are a SILICON-electron template
    (Masetti 1983, converted from cm-units); recalibrate per material -- for ITO the low-field mobility
    is usually supplied directly (mu_low constant) and only the velocity-saturation field factor is
    used. mu1=0 / Pc=0 drop the corresponding terms."""
    mu_max: float = 0.1414       # m^2/Vs  (1414 cm^2/Vs)
    mu_min1: float = 0.00685     # m^2/Vs  (68.5 cm^2/Vs)
    mu_min2: float = 0.00685
    mu1: float = 0.0             # m^2/Vs  (set >0 to enable the high-doping upturn term)
    Pc_m3: float = 0.0           # m^-3    (set >0 to enable the exp(-Pc/N) term)
    Cr_m3: float = 9.2e22        # m^-3    (9.2e16 cm^-3)
    Cs_m3: float = 3.41e26       # m^-3    (3.41e20 cm^-3)
    alpha: float = 0.711
    beta: float = 1.98


def masetti_mu_low(N, p: "MasettiParams" = MasettiParams()):
    """Masetti low-field mobility mu_low(N), N in m^-3 (array-broadcasting). Monotone-decreasing in N;
    -> mu_max as N -> 0 (when Pc=mu1=0)."""
    N = np.asarray(N, dtype=np.float64)
    Nsafe = np.where(N > 0.0, N, 1.0)
    t_exp = p.mu_min1 * np.exp(-p.Pc_m3 / Nsafe) if p.Pc_m3 > 0.0 else p.mu_min1
    body = (p.mu_max - p.mu_min2) / (1.0 + np.power(N / p.Cr_m3, p.alpha))
    last = p.mu1 / (1.0 + np.power(p.Cs_m3 / Nsafe, p.beta)) if p.mu1 > 0.0 else 0.0
    return t_exp + body - last


def caughey_thomas(mu_low, E, v_sat_ms, beta=2.0):
    """Velocity-saturated mobility mu(E) = mu_low/(1+(mu_low|E|/v_sat)^beta)^(1/beta). v_sat=inf returns
    mu_low (the off-switch). Array-broadcasting."""
    mu_low = np.asarray(mu_low, dtype=np.float64)
    Eabs = np.abs(np.asarray(E, dtype=np.float64))
    if not np.isfinite(v_sat_ms):
        return mu_low * np.ones_like(Eabs)
    x = mu_low * Eabs / float(v_sat_ms)
    return mu_low / np.power(1.0 + np.power(x, beta), 1.0 / beta)


def mu_field_density(E, N, *, mu_low_const=None, masetti=None, v_sat_ms=float("inf"), beta=2.0):
    """Full mu(E, N): low-field part = mu_low_const (constant) OR masetti_mu_low(N) if `masetti` given,
    then the Caughey-Thomas field factor. At least one of mu_low_const / masetti must be supplied."""
    if masetti is not None:
        mu_low = masetti_mu_low(N, masetti)
    elif mu_low_const is not None:
        mu_low = np.full(np.shape(np.asarray(E, float)), float(mu_low_const))
    else:
        raise ValueError("mu_field_density needs mu_low_const or masetti")
    return caughey_thomas(mu_low, E, v_sat_ms, beta)


def drift_velocity(E, mu_low, v_sat_ms, beta=2.0):
    """v(E) = mu(E) E -- saturates to v_sat as E -> inf. Convenience for the J = q n v(E) reference."""
    return caughey_thomas(mu_low, E, v_sat_ms, beta) * np.abs(np.asarray(E, dtype=np.float64))


def mu_edge_expr_devsim(*, mu_low="mu_n", efield="ElectricField", v_sat="v_sat",
                        e_smooth="E_smooth", beta=2.0):
    """The Caughey-Thomas field factor as a DEVSIM edge-model STRING (ASCII only): mu_low and v_sat and
    e_smooth are DEVSIM parameters/edge-models; the smoothed field magnitude is sqrt(efield^2+e_smooth^2)
    so diff() stays continuous through E=0. Shares the beta exponent with caughey_thomas so the DEVSIM
    and numpy forms cannot drift. `mu_low` may be a scalar parameter name (field-only) or an edge model
    (density-dependent)."""
    Eabs = "pow({ef}^2 + {es}^2, 0.5)".format(ef=efield, es=e_smooth)
    x = "({ml}*{Ea}/{vs})".format(ml=mu_low, Ea=Eabs, vs=v_sat)
    return "({ml}/pow(1.0 + pow({x},{b}), {ib}))".format(ml=mu_low, x=x, b=float(beta),
                                                         ib=1.0 / float(beta))
