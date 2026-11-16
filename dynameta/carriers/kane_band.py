"""
Kane (nonparabolic) conduction band in the equilibrium electrostatics (v0.16: the closure solves the
same band whose DOS and optical masses the other consumers report).

Pure math (numpy only, no DEVSIM import): the density closure of a Kane band, its DEVSIM expression twin,
its inverse and its slope. The band object the user passes is materials.transport_model.KaneBand; the
DEVSIM wiring is physics_equilibrium.setup_semiconductor_region(band=...).

1. BAND AND DENSITY
   Dispersion E (1 + alpha E) = hbar^2 k^2 / (2 m0) (E above the band edge, m0 the band-edge mass,
   alpha [1/eV] the nonparabolicity). The electron density of a spin-degenerate isotropic band at
   temperature T and reduced Fermi level eta = (E_F - E_c)/kT, integrating by parts in E:
       n = (1/(3 pi^2)) Int_0^inf (-df/dE) k(E)^3 dE ,   f = 1/(1 + exp((E - E_F)/kT)) ,
   and with x = E/kT, b = alpha kT, k^3 = (2 m0 kT/hbar^2)^(3/2) [x (1 + b x)]^(3/2):
       n = N_c(m0) G(eta, b),   N_c(m0) = 2 (m0 kT / (2 pi hbar^2))^(3/2),
       G(eta, b) = (1/Gamma(5/2)) Int_0^inf (-df/dx) K(x) dx = (1/Gamma(5/2)) Int_0^inf f(x - eta) K'(x) dx,
       K(x) = u^(3/2),  u = x (1 + b x),  K'(x) = (3/2) x^(1/2) (1 + b x)^(1/2) (1 + 2 b x).
   G(eta, 0) = F_1/2(eta) (normalized Fermi-Dirac integral): b = 0 is the parabolic closure. Expanding in
   b gives the asymptotic series G = F_1/2 + (15/4) b F_3/2 + (105/32) b^2 F_5/2 - (315/128) b^3 F_7/2 + ...,
   whose terms grow like (b eta)^m once eta >> 1: it DIVERGES for b eta >~ 1 (b = 0.03 at eta > 33), so
   it is not used.

2. EVALUATION (one implementation, a numpy evaluator and a DEVSIM expression from the same constants)
   G(eta, b) = F_AH(eta) + Delta(eta, b), F_AH the Aymerich-Humet F_1/2 the parabolic closure solves
   (physics_equilibrium.F12_aymerich_humet) and Delta = G - F_1/2 the EXACT-integral Kane increment, so
   b -> 0 gives the parabolic closure (Delta = 0 exactly at b = 0). Delta is evaluated in two pieces:
   * eta <= 8: generalized Gauss-Laguerre (weight x^(1/2) e^-x, N = 32, the 22 nodes x < 45 kept; the
     dropped ones contribute < 2e-15 of Delta at eta <= 8, below the round-off of G):
         Delta = (3/(2 Gamma(5/2))) Int x^(1/2) e^-x [e^eta / (1 + e^(eta - x))] g(b x) dx ,
         g(t) = (1 + t)^(1/2) (1 + 2 t) - 1 = t (1 + 2 t)/((1 + t)^(1/2) + 1) + 2 t   (cancellation-free).
     The bracket is analytic on the real axis (poles at eta +- i pi), so the rule converges fast while
     eta is moderate.
   * eta >= 6: Sommerfeld expansion of Int (-df/dx) K dx to the 6th derivative. K = u^(3/2) has the
     closed derivatives (u' = 1 + 2 b x, u'^2 = 1 + 4 b u, u'' = 2 b):
         K'' = (3/4) u^(-1/2) + 6 b u^(1/2),   K'''' = (9/16) u^(-5/2),
         K^(6) = (315/64) u^(-9/2) + (135/8) b u^(-7/2),
     so G ~ [u^(3/2) + (pi^2/6) K'' + (7 pi^4/360) K'''' + (31 pi^6/15120) K^(6)]/Gamma(5/2) at x = eta,
     and Delta is that minus its b = 0 value (the leading difference written cancellation-free). The
     series is asymptotic: useless at eta ~ 2 (4e-2 error for F_1/2), 1e-5 class from eta ~ 6 on.
   * 6 < eta < 8: Delta = (1 - w) Laguerre + w Sommerfeld, w = s^3 (10 - 15 s + 6 s^2), s = (eta - 6)/2
     (C2 blend). The Laguerre branch is evaluated at min(eta, 8) and the Sommerfeld one at max(eta, 6), so
     both are finite at every eta (DEVSIM traps overflow and a negative pow base even in an unselected
     branch); outside the zone the other branch has weight 0 exactly.
   MEASURED against scipy quadrature of G (eta in [-10, 60] step 0.25, b = 0.0103 / 0.03): the increment
   error |Delta - (G - F_1/2)_quad| / G_quad is <= 1.5e-6 / 3.5e-6 (worst at eta = 7.5, inside the blend).
   The closure's TOTAL error against the exact Kane density is therefore the Aymerich-Humet error of the
   parabolic part (<= 3.8e-3 at eta ~ -3, 1.4e-3 at eta ~ 15-20, F_1/2/G times smaller with b > 0) plus
   < 4e-6: the Kane closure is exactly as accurate as the parabolic closure it extends.

3. T -> 0 AND THE OPTICAL MASS
   For eta -> inf at fixed b eta = alpha E_F the Sommerfeld leading term gives n -> k_F^3/(3 pi^2) with
   hbar^2 k_F^2/(2 m0) = E_F (1 + alpha E_F): the Kane Fermi sphere. The Fermi-surface (Drude) mass of the
   same band, m_opt = hbar^2 k_F/(dE/dk) = m0 (1 + 2 alpha E_F), is materials.KaneOpticalMass; the band
   object hands it out (KaneBand.optical_mass), so optics and electrostatics read one (m0, alpha).

4. NOT THE SCHRODINGER-POISSON BULK LIMIT
   The sub-band model (schrodinger_poisson) is parabolic along z with an in-plane Kane DOS; its bulk limit
   is N_c [F_1/2 + 2 b F_3/2] (schrodinger_poisson.sp_bulk_density_m3), a DIFFERENT band from this
   isotropic one (first-order coefficient 2 b against 15/4 b; T -> 0 ratio (1 + 0.8 aE)/(1 + aE)^(3/2)).
   The two must not be equated.
"""

from __future__ import annotations

import math

import numpy as np

from dynameta.carriers import physics_equilibrium as PE
from dynameta.constants import KB, Q_E, T_REF

__all__ = ["KANE_BLEND_ETA", "kane_b", "kane_increment", "kane_increment_expr", "kane_closure",
           "kane_closure_slope", "invert_kane_closure", "KANE_INCREMENT_MODEL"]

# The DEVSIM node model holding Delta(eta, b) of a Kane region (and "<name>:Potential", its derivative).
KANE_INCREMENT_MODEL = "KaneDelta"

_G52 = math.gamma(2.5)
# Sommerfeld coefficients 2 (1 - 2^(1 - 2k)) zeta(2k), k = 1, 2, 3
_SOM1 = math.pi ** 2 / 6.0
_SOM2 = 7.0 * math.pi ** 4 / 360.0
_SOM3 = 31.0 * math.pi ** 6 / 15120.0
# the blend zone [lo, hi] of section 2
KANE_BLEND_ETA = (6.0, 8.0)
# Generalized Gauss-Laguerre nodes / weights, weight x^(1/2) e^-x, N = 32, nodes x < 45 (22 of 32): roots of
# L_32^(1/2) refined by Newton in 60-digit arithmetic (mpmath), w = Gamma(N + 3/2) x / (N! (N + 1)^2
# L_33^(1/2)(x)^2), exact for x^k (k <= 63) to 1e-60 before rounding. Literals so the DEVSIM model string does
# not depend on a library version (tests/test_kane_band.py checks them against scipy).
_LAG_X = (0.0753527434435432, 0.3015846258809085, 0.679218812319656, 1.2091346748887941, 1.8925791809476384,
          2.7311833356258357, 3.726984212738161, 4.882453313867251, 6.200532248147024, 7.684677044217206,
          9.33891281755023, 11.167901058389647, 13.177022531899675, 15.372479772098687, 17.76142452338247,
          20.352117419061983, 23.154129967825632, 26.178602997323242, 29.438581817741753, 32.94945776215069,
          36.72956060128426, 40.80097052984285)
_LAG_W = (0.038373959065755046, 0.12259449673289444, 0.18953755749416784, 0.19902396255871888,
          0.15769389419812385, 0.09870186920324343, 0.04995205783263851, 0.02070142268935196,
          0.0070735543848818245, 0.001999290401447072, 0.00046779254747998417, 9.051024052574552e-05,
          1.4443218531703535e-05, 1.8932119570990827e-06, 2.0274960694299234e-07, 1.761946366034585e-08,
          1.2322004417936328e-09, 6.865347068903945e-11, 3.0109295638938354e-12, 1.0245064341562693e-13,
          2.6578976435035708e-15, 5.1473615331283614e-17)
_LAG_D = tuple(math.exp(-x) for x in _LAG_X)        # e^-x_i: the Laguerre terms are c_i E/(1 + d_i E), E = e^eta


def kane_b(alpha_per_eV: float, T_K: float = T_REF) -> float:
    """b = alpha kT (dimensionless) for a nonparabolicity alpha [1/eV] at T [K]. At T_REF, kT/q is
    constants.V_T bit for bit (the equilibrium closure's V_t)."""
    return float(alpha_per_eV) * (KB * float(T_K) / Q_E)


def _check_b(b: float) -> float:
    b = float(b)
    if not (math.isfinite(b) and b >= 0.0):
        raise ValueError("Kane closure: b = alpha kT must be finite and >= 0, got {!r}".format(b))
    return b


def _laguerre_coeffs(b: float):
    """c_i = (3/2) w_i g(b x_i)/Gamma(5/2): the Laguerre weights of Delta (b-dependent, eta-independent)."""
    out = []
    for x, w in zip(_LAG_X, _LAG_W):
        t = b * x
        g = t * (1.0 + 2.0 * t) / (math.sqrt(1.0 + t) + 1.0) + 2.0 * t
        out.append(1.5 * w * g / _G52)
    return out


def kane_increment(eta, b: float):
    """Delta(eta, b) = G(eta, b) - F_1/2(eta) (module docstring, section 2); numpy, broadcasting in eta.
    Exactly 0 at b = 0."""
    b = _check_b(b)
    e = np.asarray(eta, dtype=np.float64)
    lo, hi = KANE_BLEND_ETA
    # Laguerre branch at min(eta, hi)
    eL = np.minimum(e, hi)
    E = np.exp(eL)
    lag = np.zeros_like(e)
    for c, d in zip(_laguerre_coeffs(b), _LAG_D):
        lag = lag + c * E / (1.0 + d * E)
    # Sommerfeld branch at max(eta, lo)
    eS = np.maximum(e, lo)
    y = b * eS
    u = eS * (1.0 + y)
    r = np.sqrt(1.0 + y)
    t1 = eS ** 1.5 * y * (3.0 + 3.0 * y + y * y) / ((1.0 + y) * r + 1.0)
    t2 = _SOM1 * (-0.75 * eS ** -0.5 * y / ((1.0 + y) + r) + 6.0 * b * u ** 0.5)
    t3 = _SOM2 * 0.5625 * (u ** -2.5 - eS ** -2.5)
    t4 = _SOM3 * (4.921875 * (u ** -4.5 - eS ** -4.5) + 16.875 * b * u ** -3.5)
    som = (t1 + t2 + t3 + t4) / _G52
    s = np.minimum(np.maximum((e - lo) / (hi - lo), 0.0), 1.0)
    w = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
    out = (1.0 - w) * lag + w * som
    return float(out) if np.ndim(out) == 0 else out


def kane_increment_expr(eta_expr: str, b: float) -> str:
    """The DEVSIM node-model expression of kane_increment(eta, b), eta given as a DEVSIM expression. Same
    constants and the same formula as kane_increment (they agree to round-off, gated in the tests). No
    sqrt() in the DEVSIM parser: pow(., 0.5)."""
    b = _check_b(b)
    e = "({})".format(eta_expr)
    lo, hi = KANE_BLEND_ETA
    E = "exp(min({}, {!r}))".format(e, hi)
    lag = " + ".join("{!r} * {E} / (1.0 + {!r} * {E})".format(c, d, E=E)
                     for c, d in zip(_laguerre_coeffs(b), _LAG_D))
    eS = "max({}, {!r})".format(e, lo)
    y = "({!r} * {})".format(b, eS)
    u = "({} * (1.0 + {}))".format(eS, y)
    r = "pow(1.0 + {}, 0.5)".format(y)
    t1 = "pow({eS}, 1.5) * {y} * (3.0 + 3.0 * {y} + {y} * {y}) / ((1.0 + {y}) * {r} + 1.0)".format(
        eS=eS, y=y, r=r)
    t2 = "{c!r} * (-0.75 * pow({eS}, -0.5) * {y} / ((1.0 + {y}) + {r}) + {b6!r} * pow({u}, 0.5))".format(
        c=_SOM1, eS=eS, y=y, r=r, b6=6.0 * b, u=u)
    t3 = "{c!r} * (pow({u}, -2.5) - pow({eS}, -2.5))".format(c=_SOM2 * 0.5625, u=u, eS=eS)
    t4 = "{c!r} * (4.921875 * (pow({u}, -4.5) - pow({eS}, -4.5)) + {b!r} * pow({u}, -3.5))".format(
        c=_SOM3, u=u, eS=eS, b=16.875 * b)
    som = "(({}) + ({}) + ({}) + ({})) / {!r}".format(t1, t2, t3, t4, _G52)
    s = "min(max(({} - {!r}) / {!r}, 0.0), 1.0)".format(e, lo, hi - lo)
    w = "({s} * {s} * {s} * (10.0 - 15.0 * {s} + 6.0 * {s} * {s}))".format(s=s)
    return "((1.0 - {w}) * ({lag}) + {w} * ({som}))".format(w=w, lag=lag, som=som)


def kane_closure(eta, b: float):
    """G(eta, b) = F_AH(eta) + Delta(eta, b): n / N_c(m0) of the Kane closure the electrostatics solves
    (scalar in, float out; array in, array out). b = 0 returns physics_equilibrium.F12_aymerich_humet(eta)
    exactly."""
    b = _check_b(b)
    e = np.asarray(eta, dtype=np.float64)
    par = np.vectorize(PE.F12_aymerich_humet, otypes=[np.float64])(e)
    if b == 0.0:
        out = par
    else:
        out = par + kane_increment(e, b)
    return float(out) if np.ndim(out) == 0 else out


def kane_closure_slope(eta, b: float, h: float = 1e-4):
    """dG/deta of the closure by a central difference (relative error ~1e-9 at h = 1e-4, the step
    screening_mesh.screening_length_m uses for the parabolic closure): the dn/dE_F of the screening
    length, n' = N_c G'(eta)/kT."""
    return (np.asarray(kane_closure(np.asarray(eta, float) + h, b))
            - np.asarray(kane_closure(np.asarray(eta, float) - h, b))) / (2.0 * h)


def invert_kane_closure(target: float, b: float, eta_min: float = -20.0, eta_max: float = 80.0,
                        tol: float = 1e-10) -> float:
    """eta with kane_closure(eta, b) = target (bisection to rtol `tol`; the bracket checks and messages of
    physics_equilibrium.invert_F12, one shared kernel). b = 0 is invert_F12 exactly."""
    b = _check_b(b)
    if b == 0.0:
        return PE.invert_F12(target, eta_min, eta_max, tol)
    return PE.invert_monotone(lambda eta: kane_closure(eta, b), target, eta_min, eta_max, tol,
                              name="invert_kane_closure", closure="G_Kane")
