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

v0.15 (2026-11-06, plan item 4.1.1) -- the TWO-SIDED Tsu-Esaki kernel for the SELF-CONSISTENT gate-oxide
tunnelling term (carriers/tunneling.py, coupling="self_consistent"). Additive; the functions above keep
their results (wkb_trapezoid_exponent is now the one-layer call of wkb_layer_exponent -- the same algebra,
re-associated: <= 4 ulp in theta, gated in tests/test_tunneling_kernels.py).

* wkb_layer_exponent -- THE WKB exponent of one dielectric layer whose conduction edge is LINEAR across it
  (a charge-free layer), for an electron of normal energy E: with A = U_a - E and B = U_b - E the barrier
  heights at its two faces (either may be negative: a turning point inside the layer),

      theta = 2 Int kappa dx = c f(A, B),   c = 4 sqrt(2 m) t / (3 hbar),
      f = (A_+^(3/2) - B_+^(3/2)) / (A - B)
        = (A + sqrt(AB) + B) / (sqrt A + sqrt B)          A, B > 0 (cancellation-free; A = B is the
                                                          rectangular barrier 3 sqrt(A)/2 -> theta =
                                                          2 sqrt(2 m A) t / hbar)
        = A^(3/2) / (A - B)                                A > 0 >= B (the trapezoid exits inside)
      with exact partial derivatives f_A = (sqrt A + 2 sqrt B) / (2 (sqrt A + sqrt B)^2) (both > 0), which
      are CONTINUOUS across every turning-point switch (theta is C^1 in A and B; its second derivative
      jumps there -- the reason the fixed grid below is fine but needs ~kT/4 panels near a kink).
  A stack is the SUM over its layers, each with its own mass, thickness and Anderson band offset.

* tsu_esaki_two_sided -- the two-sided supply integral (Tsu & Esaki, Appl. Phys. Lett. 22, 562 (1973);
  the form re-derived in docs/audit/2026-11-06-v0.15-oxide-tunnelling.md section 2, and checked against
  the Silvaco ATLAS manual's Eq. 3-144/3-145, a SECONDARY source -- the primary's equation number was not
  read): with eps = E_x - E_c,s >= 0 the electron normal energy above the semiconductor conduction edge
  at the interface, x_s = (E_Fs - E_c,s)/kT and x_g = (E_Fg - E_c,s)/kT the two Fermi levels,

      J = (q m_s / (2 pi^2 hbar^3)) Int_0^inf T(eps) [N_s(eps) - N_g(eps)] d eps,
      N_a(eps) = kT ln(1 + exp(x_a - eps/kT)) = Int_0^inf f(E_x + E_t - E_Fa) dE_t,

  J > 0 for electrons flowing semiconductor -> gate. It is ZERO POINTWISE when x_s == x_g (detailed
  balance by construction). Its shot noise, for T << 1 (the Landauer-Buttiker partition
  f_s (1 - f_g) + f_g (1 - f_s), integrated over the transverse energy in closed form, audit note
  section 2.3), is EXACTLY

      S = 2 q J coth(Delta/2),   Delta = x_s - x_g     (per unit area)

  -- 4 kT dJ/dV at equilibrium (the Nyquist value) and 2 q |J| for |Delta| >> 1. It is evaluated on a
  FIXED energy grid (TunnelEnergyGrid: composite 12-point Gauss-Legendre panels of width step_kT * kT on
  [0, eps_max]) so that J is a smooth (C^1) function of the state at every node of the grid, and the
  analytic derivatives returned are the EXACT derivatives of the discrete sum (the Newton Jacobian and
  the ssac / noise matrices are then consistent; the adaptive one-sided kernel above changes its panel
  count with V and steps J(V) at the 1e-10 level).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import H_PLANCK, HBAR, KB, M_E, Q_E, T_REF

__all__ = ["OxideLeakageParams", "fn_coefficients", "fowler_nordheim_current",
           "direct_tunneling_current", "schuegraf_hu_current", "wkb_trapezoid_exponent",
           "tsu_esaki_current", "wkb_layer_exponent", "TunnelLayer", "TunnelEnergyGrid",
           "tunnel_energy_grid", "tsu_esaki_two_sided", "TwoSidedTE"]


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
    U0 = phi_b_eV * Q_E + np.asarray(eps_J, dtype=np.float64)
    # v0.15: the one-layer call of THE layer kernel (A = U0 at the cathode face, B = U0 - qV at the anode)
    return wkb_layer_exponent(U0, U0 - Q_E * float(V_ox_V), float(t_ox_m), m_ox_over_m0)


def wkb_layer_exponent(A_J, B_J, t_m: float, m_over_m0: float, *, derivatives: bool = False):
    """THE WKB exponent theta = 2 Int kappa dx [dimensionless] of ONE dielectric layer of thickness t_m
    with a LINEAR conduction edge, for barrier heights A_J = U_a - E and B_J = U_b - E [J] above the
    electron at the layer's two faces (module header, v0.15; either may be <= 0 -- a turning point inside
    the layer, or the whole layer below E -> 0). derivatives=True also returns (dtheta/dA, dtheta/dB)
    [1/J], continuous across every turning-point switch. Arrays broadcast."""
    A = np.asarray(A_J, dtype=np.float64)
    B = np.asarray(B_J, dtype=np.float64)
    A, B = np.broadcast_arrays(A, B)
    if not (t_m > 0.0 and m_over_m0 > 0.0):
        raise ValueError("wkb_layer_exponent: t_m and m_over_m0 must be > 0")
    c = 4.0 * np.sqrt(2.0 * m_over_m0 * M_E) * float(t_m) / (3.0 * HBAR)
    a = np.maximum(A, 0.0)
    b = np.maximum(B, 0.0)
    sa, sb = np.sqrt(a), np.sqrt(b)
    both = (a > 0.0) & (b > 0.0)
    only_a = (a > 0.0) & ~(b > 0.0)
    only_b = (b > 0.0) & ~(a > 0.0)
    s = np.where(both, sa + sb, 1.0)
    dab = np.where(only_a, A - B, 1.0)                      # > 0 there (A > 0 >= B)
    dba = np.where(only_b, B - A, 1.0)
    f = np.where(both, (a + sa * sb + b) / s,
                 np.where(only_a, a * sa / dab, np.where(only_b, b * sb / dba, 0.0)))
    if not derivatives:
        return c * f
    s2 = s * s
    fA = np.where(both, (sa + 2.0 * sb) / (2.0 * s2),
                  np.where(only_a, sa * (0.5 * A - 1.5 * B) / (dab * dab),
                           np.where(only_b, b * sb / (dba * dba), 0.0)))
    fB = np.where(both, (sb + 2.0 * sa) / (2.0 * s2),
                  np.where(only_a, a * sa / (dab * dab),
                           np.where(only_b, sb * (0.5 * B - 1.5 * A) / (dba * dba), 0.0)))
    return c * f, c * fA, c * fB


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


# ---------------------------------------------------------------------------------------------------
# v0.15: the two-sided Tsu-Esaki kernel on a FIXED energy grid (module header)
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TunnelLayer:
    """One dielectric layer of a tunnelling stack, semiconductor side first: its conduction-band offset
    above the SEMICONDUCTOR conduction edge at a common potential [eV] (Anderson: chi_s - chi_layer; for
    SiO2 on Si the Lenzlinger-Snow fit 3.1 eV), thickness [m] and tunnelling mass [m0]."""
    offset_eV: float
    thickness_m: float
    m_over_m0: float

    def __post_init__(self):
        if not (self.offset_eV > 0.0 and self.thickness_m > 0.0 and self.m_over_m0 > 0.0):
            raise ValueError("TunnelLayer: offset_eV, thickness_m and m_over_m0 must be > 0 (got {!r}, {!r}, "
                             "{!r})".format(self.offset_eV, self.thickness_m, self.m_over_m0))


@dataclass(frozen=True)
class TunnelEnergyGrid:
    """A FIXED quadrature grid in eps = E_x - E_c,s on [0, eps_max_J]: composite 12-point Gauss-Legendre
    panels of width step_J (<= step_kT * kT). Build with tunnel_energy_grid."""
    eps_J: np.ndarray
    w_J: np.ndarray
    kT_J: float
    eps_max_J: float
    step_J: float


def tunnel_energy_grid(kT_J: float, eps_max_J: float, *, step_kT: float = 0.125) -> TunnelEnergyGrid:
    """The fixed energy grid of tsu_esaki_two_sided: n = ceil(eps_max / (step_kT kT)) equal panels of the
    12-point Gauss-Legendre rule on [0, eps_max]. step_kT = 0.25 (default) integrates the Fermi supply to
    ~1e-15 and the C^1 turning-point kinks of theta to <= 1e-9 relative (MEASURED, K6 in
    tests/test_tunneling_kernels.py); halve it for a tighter kink error (it falls as step^3.5)."""
    if not (kT_J > 0.0 and eps_max_J > 0.0 and step_kT > 0.0):
        raise ValueError("tunnel_energy_grid: kT_J, eps_max_J and step_kT must be > 0 (a T = 0 supply is not "
                         "supported by the fixed grid; use tsu_esaki_current)")
    n = int(np.ceil(float(eps_max_J) / (float(step_kT) * float(kT_J))))
    if n * _TE_GL_X.size > _TE_MAX_NODES:
        raise ValueError("tunnel_energy_grid: {} nodes exceed the {} ceiling (eps_max / (step_kT kT) too large)"
                         .format(n * _TE_GL_X.size, _TE_MAX_NODES))
    edges = np.linspace(0.0, float(eps_max_J), n + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    return TunnelEnergyGrid(eps_J=(mid + half * _TE_GL_X).ravel(), w_J=(half * _TE_GL_W).ravel(),
                            kT_J=float(kT_J), eps_max_J=float(eps_max_J), step_J=float(edges[1] - edges[0]))


@dataclass(frozen=True)
class TwoSidedTE:
    """Result of tsu_esaki_two_sided, per interface node (arrays of shape (n,) unless stated).
    J        net current density [A/m^2], > 0 for electrons semiconductor -> gate
    J_fwd    the semiconductor -> gate one-way part (the N_s term) [A/m^2]; J_bwd the gate -> semiconductor
             part (N_g); J = J_fwd - J_bwd
    S        shot-noise PSD 2 q J coth(Delta/2) [A^2/(Hz m^2)] (module header), >= 0
    dJ_dxs, dJ_dxg   dJ / dx_s, dJ / dx_g [A/m^2] (x = (E_F - E_c,s)/kT)
    dJ_dd    (n, L) dJ / d d_k [A/m^2/V], d_k = psi_s - psi(face k), k = 1..L (face L = the gate face)
    tail     integrand at the grid's top node times kT, over J_fwd + J_bwd (truncation diagnostic)"""
    J: np.ndarray
    J_fwd: np.ndarray
    J_bwd: np.ndarray
    S: np.ndarray
    dJ_dxs: np.ndarray
    dJ_dxg: np.ndarray
    dJ_dd: np.ndarray
    tail: np.ndarray


def _softplus(z):
    return np.logaddexp(0.0, z)


def _logistic(z):
    return np.exp(-np.logaddexp(0.0, -z))


_NOISE_SMALL_DELTA = 1e-4     # |Delta| below this: (N_s - N_g) coth(Delta/2) by its O(Delta^2) series


def tsu_esaki_two_sided(grid: TunnelEnergyGrid, x_s, x_g, d_V, layers, *, m_supply_over_m0: float = 1.0,
                        chunk_nodes: int = 2_000_000) -> TwoSidedTE:
    """Two-sided Tsu-Esaki current through a dielectric STACK on the fixed `grid` (module header, v0.15),
    vectorised over n interface nodes, with the EXACT derivatives of the discrete sum.

    x_s, x_g   (n,) (E_Fs - E_c,s)/kT and (E_Fg - E_c,s)/kT at each node (E_c,s: the semiconductor
               conduction edge at the interface).
    d_V        (n, L) psi_s - psi(face k), k = 1..L: the potential drop from the interface to the far face
               of each layer (face L touches the gate). Layer j's conduction edge relative to E_c,s is
               then q d_j + offset_j at its near face (d_0 = 0) and q d_(j+1) + offset_j at its far face.
    layers     sequence of L TunnelLayer, semiconductor side first.
    m_supply_over_m0  the transverse (supply) mass of the prefactor q m kT/(2 pi^2 hbar^3) -- one mass for
               BOTH sides, as Tsu-Esaki's conserved transverse momentum requires (detailed balance).
    The integral starts at eps = 0 (the semiconductor band bottom): the gate's band bottom must lie below
    E_c,s (the caller checks; a metal gate's does). kT is the grid's."""
    layers = tuple(layers)
    L = len(layers)
    if L < 1 or not all(isinstance(l, TunnelLayer) for l in layers):
        raise ValueError("tsu_esaki_two_sided: layers must be a non-empty sequence of TunnelLayer")
    xs = np.atleast_1d(np.asarray(x_s, dtype=np.float64))
    xg = np.atleast_1d(np.asarray(x_g, dtype=np.float64))
    d = np.asarray(d_V, dtype=np.float64).reshape(xs.size, L)
    if xg.shape != xs.shape:
        raise ValueError("tsu_esaki_two_sided: x_s and x_g must have the same shape")
    kT = grid.kT_J
    pref = Q_E * float(m_supply_over_m0) * M_E / (2.0 * np.pi ** 2 * HBAR ** 3)
    e = grid.eps_J[None, :]
    w = grid.w_J[None, :]
    n = xs.size
    J, Jf, Jb, S = (np.zeros(n) for _ in range(4))
    dxs, dxg, tail = np.zeros(n), np.zeros(n), np.zeros(n)
    dd = np.zeros((n, L))
    step = max(1, int(chunk_nodes // max(e.size, 1)))
    for lo in range(0, n, step):
        sl = slice(lo, min(n, lo + step))
        a_s = xs[sl, None] - e / kT
        a_g = xg[sl, None] - e / kT
        theta = np.zeros((a_s.shape[0], e.size))
        dth = []                                               # dtheta/d d_k, k = 1..L
        prev_fb = None
        for j, lay in enumerate(layers):
            off = lay.offset_eV * Q_E
            Ua = off + (Q_E * d[sl, j - 1, None] if j > 0 else 0.0)
            Ub = off + Q_E * d[sl, j, None]
            th, tA, tB = wkb_layer_exponent(Ua - e, Ub - e, lay.thickness_m, lay.m_over_m0, derivatives=True)
            theta += th
            if j > 0:                                          # face j: far face of j-1, near face of j
                dth.append(Q_E * (prev_fb + tA))
            prev_fb = tB
        dth.append(Q_E * prev_fb)                              # face L (the gate face)
        with np.errstate(under="ignore"):
            T = np.exp(-theta)
        Ns = kT * _softplus(a_s)
        Ng = kT * _softplus(a_g)
        diff = Ns - Ng
        wT = w * T
        J[sl] = pref * np.sum(wT * diff, axis=1)
        Jf[sl] = pref * np.sum(wT * Ns, axis=1)
        Jb[sl] = pref * np.sum(wT * Ng, axis=1)
        dxs[sl] = pref * np.sum(wT * kT * _logistic(a_s), axis=1)
        dxg[sl] = -pref * np.sum(wT * kT * _logistic(a_g), axis=1)
        for k in range(L):
            dd[sl, k] = -pref * np.sum(wT * dth[k] * diff, axis=1)
        delta = xs[sl] - xg[sl]
        small = np.abs(delta) < _NOISE_SMALL_DELTA
        big = ~small
        Sv = np.zeros(delta.size)
        Sv[big] = 2.0 * Q_E * J[sl][big] / np.tanh(0.5 * delta[big])
        if np.any(small):                                      # (N_s - N_g) coth(D/2) -> 2 kT sigma(a_s - D/2)
            mid = a_s[small] - 0.5 * delta[small, None]
            Sv[small] = 2.0 * Q_E * pref * np.sum(wT[small] * 2.0 * kT * _logistic(mid), axis=1)
        S[sl] = Sv
        top = T[:, -1] * (Ns[:, -1] + Ng[:, -1]) * kT
        tot = np.sum(wT * (Ns + Ng), axis=1)
        tail[sl] = np.where(tot > 0.0, top / np.where(tot > 0.0, tot, 1.0), 0.0)
    return TwoSidedTE(J=J, J_fwd=Jf, J_bwd=Jb, S=S, dJ_dxs=dxs, dJ_dxg=dxg, dJ_dd=dd, tail=tail)

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
