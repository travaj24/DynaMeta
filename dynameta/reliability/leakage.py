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

D1d (2026-11-03, DEVSIM tunneling) -- two more kernels, both ADDITIVE (the three functions above
are unchanged, byte for byte):

* schuegraf_hu_current -- the FULL Schuegraf-Hu closed form, prefactor included:

      J_SH = A_FN E^2 exp(-B_FN g(u)/E) / (1 - sqrt(1 - u))^2 ,   u = min(qV_ox/phi_b, 1)

  (K. F. Schuegraf, C. C. King, C. Hu, VLSI Technol. Symp. 1992, p. 18). It is what the
  linearised-WKB derivation of FN gives for a TRAPEZOID: the exponent's slope at the Fermi level
  drops from c = 2 sqrt(2 m_ox phi)/(q hbar) to c (1 - sqrt(1 - u)), and the supply integral
  Int eps exp(-c eps/E) d eps = (E/c)^2 carries the square of that drop. u >= 1 makes the prefactor
  exactly 1, so J_SH == J_FN IDENTICALLY above the barrier. direct_tunneling_current above keeps
  only the exponent -- the "2-32x below full S-H" the 2026-07-05 audit disclosed is exactly this
  factor, (1 - sqrt(1-u))^-2 = 2.1x at u = 0.9 and 32x at u = 0.33. The linearisation needs
  qV_ox >> E/c' and FAILS at small V_ox (J_SH tends to a nonzero constant as V -> 0); use the
  Tsu-Esaki quadrature there.

* tsu_esaki_current -- the reference: the Tsu-Esaki supply integral with WKB transmission through
  the trapezoidal barrier (R. Tsu, L. Esaki, Appl. Phys. Lett. 22, 562 (1973); the WKB trapezoid as
  in Lenzlinger & Snow, J. Appl. Phys. 40, 278 (1969)):

      J = q m_s kT/(2 pi^2 hbar^3) Int T(E_x) ln[(1 + e^((E_F - E_x)/kT))/(1 + e^((E_F - qV - E_x)/kT))] dE_x
      T = exp(-theta),  theta = 4 sqrt(2 m_ox)/(3 hbar q F) [U0^(3/2) - (U0 - qV)_+^(3/2)],
      U0 = phi_b + E_F - E_x  (the barrier above the electron at the cathode),  F = V_ox/t_ox,

  T = 1 above the barrier top (U0 <= 0, thermionic), and the T -> 0 limit of kT ln(...) is
  clip(E_F - E_x, 0, qV). Energies are measured from the cathode SUPPLY band bottom; phi_b is the
  barrier above the cathode FERMI level (the Lenzlinger-Snow metal convention; for an accumulated
  Si surface E_F ~ E_c and phi_b ~ the 3.1-3.2 eV Si/SiO2 conduction-band offset). Image-force
  lowering and the 2-D quantisation of an accumulation-layer supply are NOT modelled.

  Reduction to FN (T = 0, E_F >> E/c, V_ox > phi_b): expanding theta to second order about E_F,
      J_TE / J_FN = 1 - E/B_FN + O((E/B_FN)^2)      (the curvature term; -4.1 % at 10 MV/cm in SiO2)
  with m_s = m0 -- the Lenzlinger-Snow A_FN carries m0 for the supply and m_ox in the barrier. At
  300 K the Murphy-Good thermal factor (pi c kT/E)/sin(pi c kT/E) RAISES J above FN (+17 % at
  10 MV/cm in SiO2) -- physics the closed form does not have, not error.

  QUADRATURE: composite 12-point Gauss-Legendre in eps = E_F - E_x, with breakpoints at the barrier
  top (eps = -phi_b), the Fermi level (0), the trapezoid/triangle switch (qV - phi_b) and the supply
  knee (qV), and a panel width of 1/4 of the SMALLER of kT and the WKB decay length 1/theta'. The
  deep tail is cut where theta rises 70 above its Fermi-level value (e^-70 ~ 4e-31 relative).
  Accuracy against an independent scipy.integrate.quad: see tests/test_tunneling_kernels.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import H_PLANCK, HBAR, KB, M_E, Q_E, T_REF

__all__ = ["OxideLeakageParams", "fn_coefficients", "fowler_nordheim_current",
           "direct_tunneling_current", "schuegraf_hu_current", "wkb_trapezoid_exponent",
           "tsu_esaki_current"]


def fn_coefficients(m_ox_over_m0: float, phi_b_eV: float):
    """(A_FN [A/V^2], B_FN [V/m]) from the barrier mass and height (module-header formulas)."""
    if not (m_ox_over_m0 > 0.0 and phi_b_eV > 0.0):
        raise ValueError("leakage: m_ox_over_m0 and phi_b_eV must be > 0")
    m_ox = m_ox_over_m0 * M_E
    phi_J = phi_b_eV * Q_E
    a_fn = Q_E ** 3 * M_E / (8.0 * np.pi * H_PLANCK * m_ox * phi_J)
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



def schuegraf_hu_current(V_ox_V, t_ox_m: float, *, m_ox_over_m0: float = 0.42,
                         phi_b_eV: float = 3.1):
    """FULL Schuegraf-Hu J(V_ox) [A/m^2] (module header): the trapezoid exponent of
    direct_tunneling_current PLUS the (1 - sqrt(1 - u))^-2 prefactor; == J_FN EXACTLY for
    V_ox >= phi_b. Valid while qV_ox >> E/c' (linearised WKB); V <= 0 -> exactly 0."""
    if not (t_ox_m > 0.0):
        raise ValueError("leakage: t_ox_m must be > 0 (a zero-thickness oxide is not a "
                         "tunneling barrier)")
    a_fn, b_fn = fn_coefficients(m_ox_over_m0, phi_b_eV)
    V = np.asarray(V_ox_V, dtype=np.float64)
    out = np.zeros_like(V)
    pos = V > 0.0
    u = np.minimum(V[pos] / float(phi_b_eV), 1.0)
    root = np.sqrt(1.0 - u)
    g = 1.0 - (1.0 - u) * root                                 # == 1 - (1 - u)^(3/2)
    E = V[pos] / float(t_ox_m)
    with np.errstate(under="ignore"):
        out[pos] = a_fn * E ** 2 * np.exp(-b_fn * g / E) / (1.0 - root) ** 2
    return out if out.ndim else float(out)


def wkb_trapezoid_exponent(eps_J, V_ox_V: float, t_ox_m: float, *, phi_b_eV: float,
                           m_ox_over_m0: float = 0.42):
    """WKB exponent theta = 2 Int kappa dx [dimensionless] through the trapezoidal oxide barrier for an
    electron eps_J [J] BELOW the cathode Fermi level (eps = E_F - E_x; U0 = phi_b + eps is its barrier
    height at the cathode). theta = 0 above the barrier top (U0 <= 0). V_ox_V must be > 0. The
    3/2-power difference is evaluated in the cancellation-free form
    a^1.5 - b^1.5 = (a - b)(a + sqrt(ab) + b)/(sqrt(a) + sqrt(b)), so V_ox -> 0 is stable (theta tends
    to the rectangular-barrier 2 sqrt(2 m_ox U0) t_ox / hbar)."""
    if not (V_ox_V > 0.0 and t_ox_m > 0.0):
        raise ValueError("wkb_trapezoid_exponent: V_ox_V and t_ox_m must be > 0")
    m_ox = m_ox_over_m0 * M_E
    qV = Q_E * float(V_ox_V)
    F = float(V_ox_V) / float(t_ox_m)
    U0 = phi_b_eV * Q_E + np.asarray(eps_J, dtype=np.float64)
    a = np.maximum(U0, 0.0)
    b = np.maximum(U0 - qV, 0.0)
    sa, sb = np.sqrt(a), np.sqrt(b)
    trap = b > 0.0
    diff = np.where(trap, qV * (a + sa * sb + b) / np.where(trap, sa + sb, 1.0), a * sa)
    return 4.0 * np.sqrt(2.0 * m_ox) / (3.0 * HBAR * Q_E * F) * diff


_TE_GL_X, _TE_GL_W = np.polynomial.legendre.leggauss(12)
_TE_THETA_CUT = 70.0              # deep-tail cut: theta - theta(E_F) > 70 -> e^-70 ~ 4e-31 relative
_TE_MAX_NODES = 4_000_000         # hard ceiling on quadrature nodes per bias point (memory guard)


def _tsu_esaki_one(V: float, t_ox: float, phi: float, EF: float, m_ox_r: float, kT: float) -> float:
    """Int T(eps) S(eps) d eps [J^2] for ONE V_ox > 0 (module header; eps = E_F - E_x, in J)."""
    qV = Q_E * V
    F = V / t_ox
    m_ox = m_ox_r * M_E

    def theta(e):
        return wkb_trapezoid_exponent(e, V, t_ox, phi_b_eV=phi / Q_E, m_ox_over_m0=m_ox_r)

    def dtheta(e):                                            # d theta / d eps [1/J], U0 > 0
        U0 = phi + e
        return 2.0 * np.sqrt(2.0 * m_ox) / (HBAR * Q_E * F) * (
            np.sqrt(np.maximum(U0, 0.0)) - np.sqrt(np.maximum(U0 - qV, 0.0)))

    lo = -(phi + 40.0 * kT) if kT > 0.0 else 0.0               # 40 kT above the barrier top
    hi = EF                                                    # the supply band bottom
    # deep-tail cut: theta INCREASES with eps below the barrier top (dtheta > 0), so scan a geometric
    # grid in (0, hi] for the first eps where theta exceeds theta(E_F) + _TE_THETA_CUT
    th0 = float(theta(0.0))
    probe = hi * np.geomspace(1e-9, 1.0, 400)
    over = np.nonzero(theta(probe) - th0 > _TE_THETA_CUT)[0]
    cut = float(probe[over[0]]) if over.size else hi
    # panel width: 1/4 of min(kT, the smallest WKB decay length 1/theta') over the integrated range
    e_dec = np.linspace(max(lo, -phi) + 1e-6 * phi, cut, 257)
    ell = 1.0 / float(np.max(dtheta(e_dec)))
    h = 0.25 * (min(ell, kT) if kT > 0.0 else ell)
    bps = sorted({lo, cut} | {b for b in (-phi, 0.0, qV - phi, qV) if lo < b < cut})
    total, nodes = 0.0, 0
    for a, b in zip(bps[:-1], bps[1:]):
        n = max(1, int(np.ceil((b - a) / h)))
        nodes += n * _TE_GL_X.size
        if nodes > _TE_MAX_NODES:
            raise RuntimeError("tsu_esaki_current: quadrature needs > {} nodes at V_ox = {:.4g} V "
                               "(kT or the WKB decay length is tiny against the energy range)"
                               .format(_TE_MAX_NODES, V))
        edges = np.linspace(a, b, n + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
        half = 0.5 * (edges[1:] - edges[:-1])[:, None]
        e = (mid + half * _TE_GL_X).ravel()
        w = (half * _TE_GL_W).ravel()
        if kT > 0.0:
            S = kT * (np.logaddexp(0.0, e / kT) - np.logaddexp(0.0, (e - qV) / kT))
        else:
            S = np.clip(e, 0.0, qV)
        with np.errstate(under="ignore"):
            total += float(np.sum(w * np.exp(-theta(e)) * S))
    return total


def tsu_esaki_current(V_ox_V, t_ox_m: float, *, phi_b_eV: float, E_F_eV: float,
                      m_ox_over_m0: float = 0.42, m_supply_over_m0: float = 1.0,
                      T_K: float = T_REF):
    """Tsu-Esaki tunneling J(V_ox) [A/m^2] with WKB trapezoidal transmission -- the reference for BOTH
    regimes (direct tunneling V_ox < phi_b, Fowler-Nordheim V_ox > phi_b); see the module header for
    the formula and conventions. E_F_eV is the cathode Fermi energy above its supply band bottom and
    m_supply_over_m0 the supply (cathode) transverse mass; the classic FN limit is m_supply = m0,
    E_F >> E/c. T_K = 0 takes the exact zero-temperature supply. V <= 0 -> exactly 0 (a MAGNITUDE;
    the caller owns the direction). NO default for E_F_eV: the supply is a property of the cathode
    (a metal gate ~ 3-11 eV, an accumulated Si surface ~ 0.05-0.2 eV) and must be chosen."""
    if not (t_ox_m > 0.0):
        raise ValueError("leakage: t_ox_m must be > 0 (a zero-thickness oxide is not a "
                         "tunneling barrier)")
    if not (phi_b_eV > 0.0 and E_F_eV > 0.0 and m_ox_over_m0 > 0.0 and m_supply_over_m0 > 0.0):
        raise ValueError("tsu_esaki_current: phi_b_eV, E_F_eV, m_ox_over_m0 and m_supply_over_m0 "
                         "must be > 0")
    if not (T_K >= 0.0):
        raise ValueError("tsu_esaki_current: T_K must be >= 0")
    V = np.asarray(V_ox_V, dtype=np.float64)
    out = np.zeros(V.shape, dtype=np.float64)
    pref = Q_E * m_supply_over_m0 * M_E / (2.0 * np.pi ** 2 * HBAR ** 3)
    kT = KB * float(T_K)
    flat_in, flat_out = V.reshape(-1), out.reshape(-1)
    for i, v in enumerate(flat_in):
        if v > 0.0:
            flat_out[i] = pref * _tsu_esaki_one(float(v), float(t_ox_m), phi_b_eV * Q_E,
                                                E_F_eV * Q_E, float(m_ox_over_m0), kT)
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
