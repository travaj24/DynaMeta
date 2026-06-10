"""R16: gate-oxide tunneling leakage -- Fowler-Nordheim (high field) + direct tunneling (thin
oxide), the static gate-leakage pathway of the biased metasurface stack and a Joule source for
the electro-thermal loop.

Fowler-Nordheim (triangular-barrier WKB):

    J_FN(E) = A_FN E^2 exp(-B_FN / E)
    A_FN = q^3 m0 / (8 pi h m_ox phi_J)        [A/V^2]   (h = 2 pi hbar -- the PLANCK constant;
    B_FN = (4/3) sqrt(2 m_ox) phi_J^(3/2) / (q hbar)     [V/m]  the classic h-vs-hbar trap)

    SiO2 (phi_b = 3.1 eV, m_ox = 0.42 m0): A_FN = 1.18e-6 A/V^2 (the textbook
    1.54e-6/(m_r phi_eV)), B_FN = 2.42e10 V/m = 242 MV/cm (literature 230-260).

Direct tunneling (TRAPEZOIDAL-barrier WKB, Schuegraf-Hu exponent): for V_ox < phi_b the
electron exits the barrier before it reaches zero, removing the (phi_b - qV)^(3/2) tail:

    J_DT(V, t_ox) = A_FN E^2 exp(-B_FN g(V/phi_b) / E),   E = V/t_ox
    g(u) = 1 - (1 - min(u, 1))^(3/2)

g(u >= 1) = 1 reduces J_DT to J_FN EXACTLY (continuous at V_ox = phi_b by construction --
the oracle checks identity, not closeness). g < 1 at low V means J_DT EXCEEDS the naive FN
extrapolation -- the well-known thin-oxide leakage excess. Tunneling is ~temperature-
insensitive (no Arrhenius factor by design -- contrast TDDB/BTI).

Off-switch: OxideLeakageParams(enabled=False) (the default) returns EXACTLY 0.0. Pure numpy;
SI in/out (E in V/m, V in V, J in A/m^2; phi_b in eV converted internally). Oracle:
validation/reliability_leakage.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import HBAR, M_E, Q_E

__all__ = ["OxideLeakageParams", "fn_coefficients", "fowler_nordheim_current",
           "direct_tunneling_current"]


def fn_coefficients(m_ox_over_m0: float, phi_b_eV: float):
    """(A_FN [A/V^2], B_FN [V/m]) from the barrier mass and height (module-header formulas)."""
    if not (m_ox_over_m0 > 0.0 and phi_b_eV > 0.0):
        raise ValueError("leakage: m_ox_over_m0 and phi_b_eV must be > 0")
    m_ox = m_ox_over_m0 * M_E
    phi_J = phi_b_eV * Q_E
    h = 2.0 * np.pi * HBAR
    a_fn = Q_E ** 3 * M_E / (8.0 * np.pi * h * m_ox * phi_J)
    b_fn = (4.0 / 3.0) * np.sqrt(2.0 * m_ox) * phi_J ** 1.5 / (Q_E * HBAR)
    return float(a_fn), float(b_fn)


def fowler_nordheim_current(E_V_m, *, m_ox_over_m0: float = 0.42, phi_b_eV: float = 3.1):
    """J_FN(E) [A/m^2] for field magnitude E [V/m]; E <= 0 -> exactly 0 (SiO2 defaults)."""
    a_fn, b_fn = fn_coefficients(m_ox_over_m0, phi_b_eV)
    E = np.asarray(E_V_m, dtype=np.float64)
    out = np.zeros_like(E)
    pos = E > 0.0
    with np.errstate(under="ignore"):
        out[pos] = a_fn * E[pos] ** 2 * np.exp(-b_fn / E[pos])
    return out if out.ndim else float(out)


def direct_tunneling_current(V_ox_V, t_ox_m: float, *, m_ox_over_m0: float = 0.42,
                             phi_b_eV: float = 3.1):
    """Trapezoidal-WKB tunneling J(V_ox) [A/m^2] through an oxide of thickness t_ox_m -- the
    direct-tunneling regime for V_ox < phi_b, reducing EXACTLY to Fowler-Nordheim at
    V_ox >= phi_b. V_ox is the magnitude of the oxide drop; V <= 0 -> exactly 0."""
    if not (t_ox_m > 0.0):
        raise ValueError("leakage: t_ox_m must be > 0 (a zero-thickness oxide is not a "
                         "tunneling barrier)")
    a_fn, b_fn = fn_coefficients(m_ox_over_m0, phi_b_eV)
    V = np.asarray(V_ox_V, dtype=np.float64)
    out = np.zeros_like(V)
    pos = V > 0.0
    u = np.minimum(V[pos] / float(phi_b_eV), 1.0)
    g = 1.0 - (1.0 - u) ** 1.5
    E = V[pos] / float(t_ox_m)
    with np.errstate(under="ignore"):
        out[pos] = a_fn * E ** 2 * np.exp(-b_fn * g / E)
    return out if out.ndim else float(out)


@dataclass(frozen=True)
class OxideLeakageParams:
    """Gate-oxide tunneling-leakage post-processor (R16). enabled=False (default) -> 0.0 EXACTLY.
    leakage_J(V_ox) gives the tunneling current density through the oxide [A/m^2];
    joule_W_m3(V_ox) the volumetric heat it deposits, Q = J V_ox / t_ox = J E [W/m^3], the
    thermal_fem/electrothermal seam (pass it straight into joule_W_m3 -- SI, no _S scaling)."""
    t_ox_m: float
    phi_b_eV: float = 3.1                    # SiO2 barrier height [eV]
    m_ox_over_m0: float = 0.42               # SiO2 tunneling effective mass
    enabled: bool = False

    def __post_init__(self):
        if not (self.t_ox_m > 0.0):
            raise ValueError("OxideLeakageParams: t_ox_m must be > 0")
        if not (self.phi_b_eV > 0.0 and self.m_ox_over_m0 > 0.0):
            raise ValueError("OxideLeakageParams: phi_b_eV and m_ox_over_m0 must be > 0")

    def leakage_J_A_m2(self, V_ox_V):
        if not self.enabled:
            return 0.0 if np.ndim(V_ox_V) == 0 else np.zeros(np.shape(V_ox_V))
        return direct_tunneling_current(V_ox_V, self.t_ox_m, m_ox_over_m0=self.m_ox_over_m0,
                                        phi_b_eV=self.phi_b_eV)

    def joule_W_m3(self, V_ox_V):
        """Volumetric leakage heating Q = J * E = J * V_ox / t_ox [W/m^3] (0.0 when disabled)."""
        J = self.leakage_J_A_m2(V_ox_V)
        return J * (np.asarray(V_ox_V, dtype=np.float64) / self.t_ox_m) if np.ndim(J) \
            else float(J) * float(V_ox_V) / self.t_ox_m
