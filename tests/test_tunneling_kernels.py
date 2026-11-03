"""D1d (v0.14.0): oracles for the gate-oxide tunneling KERNELS in reliability/leakage.py -- the Tsu-Esaki
WKB quadrature and the full Schuegraf-Hu closed form. Pure numpy/scipy; every reference is coded HERE,
independently of the code under test (constants re-typed from CODATA, the integral re-written for
scipy.integrate.quad, the FN coefficients re-derived from the Lenzlinger-Snow formulas).

Gates (all FAIL on the pre-D1d code, which has neither tsu_esaki_current nor schuegraf_hu_current):
  K1  Tsu-Esaki quadrature == an independent scipy.quad of the same integral (FN + DT, 0 K + 300 K).
  K2  FN reduction: T = 0, E_F >> E/c, deeply triangular barrier ->
        J_TE/J_FN = 1 - 1/b + 19/(9 b^2) - 20/(3 b^3) + O(b^-4),   b = B_FN/E
      (the second-order expansion of the WKB exponent about E_F, derived in the test docstring).
  K3  Crossover: Schuegraf-Hu == FN IDENTICALLY at V_ox >= phi_b; below it S-H = exponent-only form x
      (1 - sqrt(1-u))^-2; and J_TE/J_FN -> the K2 series as V_ox rises past phi_b.
  K4  300 K: J(300)/J(0) == the Murphy-Good factor (pi c kT/E)/sin(pi c kT/E) within the O(E/B)
      curvature bound.
  K5  Direct-tunneling limit: J_TE ~ V at small V_ox (ohmic supply), exceeds FN below the barrier.
"""
import math

import numpy as np
import pytest

# CODATA 2018 (re-typed, not imported, so the oracle shares nothing with the code under test)
Q = 1.602176634e-19
HBAR = 1.054571817e-34
H = 6.62607015e-34
M0 = 9.1093837015e-31
KB = 1.380649e-23
PHI, MOX = 3.1, 0.42


def _fn_ref(E):
    """Lenzlinger & Snow (1969): A = q^3 m0/(8 pi h m_ox phi), B = 8 pi sqrt(2 m_ox) phi^1.5/(3 q h)."""
    phi = PHI * Q
    A = Q ** 3 * M0 / (8 * math.pi * H * MOX * M0 * phi)
    B = 8 * math.pi * math.sqrt(2 * MOX * M0) * phi ** 1.5 / (3 * Q * H)
    return A * E ** 2 * math.exp(-B / E), B


def _tsu_esaki_ref(V, t, EF, T, ms=1.0):
    """Independent Tsu-Esaki/WKB-trapezoid integral via scipy.integrate.quad (eps = E_F - E_x)."""
    from scipy.integrate import quad
    qV, F, P, kT = Q * V, V / t, PHI * Q, KB * T

    def theta(e):
        U0 = P + e
        if U0 <= 0.0:
            return 0.0
        return 4 * math.sqrt(2 * MOX * M0) / (3 * HBAR * Q * F) * (U0 ** 1.5 - max(U0 - qV, 0.0) ** 1.5)

    def S(e):
        if T == 0:
            return min(max(e, 0.0), qV)
        return kT * (np.logaddexp(0.0, e / kT) - np.logaddexp(0.0, (e - qV) / kT))

    lo = -(P + 40 * kT) if T > 0 else 0.0
    pts = [p for p in (-P, 0.0, qV - P, qV) if lo < p < EF * Q]
    val = quad(lambda e: math.exp(-theta(e)) * S(e), lo, EF * Q, points=pts, limit=4000,
               epsabs=0.0, epsrel=1e-12)[0]
    return Q * ms * M0 / (2 * math.pi ** 2 * HBAR ** 3) * val


@pytest.mark.parametrize("V,t,EF,T", [
    (6.0, 5e-9, 10.0, 0.0),        # FN, metal supply, 0 K
    (6.0, 5e-9, 10.0, 300.0),      # FN, 300 K
    (3.5, 3e-9, 3.0, 0.0),         # just above the barrier: trapezoid for deep electrons
    (1.0, 2e-9, 0.1, 300.0),       # direct tunneling, semiconductor-like supply
    (0.2, 1.5e-9, 0.1, 300.0),     # deep DT, ultrathin oxide
    (2.0, 2e-9, 10.0, 300.0),      # DT, metal supply
])
def test_k1_tsu_esaki_matches_independent_quad(V, t, EF, T):
    from dynameta.reliability.leakage import tsu_esaki_current
    j = tsu_esaki_current(V, t, phi_b_eV=PHI, E_F_eV=EF, m_ox_over_m0=MOX, T_K=T)
    r = _tsu_esaki_ref(V, t, EF, T)
    assert r > 0.0
    assert j == pytest.approx(r, rel=1e-8)          # MEASURED <= 6e-11 (the 3.5 V trapezoid case)


@pytest.mark.parametrize("F", [4e8, 6e8, 8e8, 1.0e9, 1.2e9, 1.5e9])
def test_k2_fn_reduction_second_order_series(F):
    """theta(eps) = b' (phi + eps)^1.5 / F (triangular). With x = eps/phi, b = B/F and y = 1.5 b x:
    b[(1+x)^1.5 - 1 - 1.5x] = y^2/(6b) - y^3/(54 b^2) + y^4/(216 b^3) - ..., and expanding the exponential
    under Int y e^-y (.) dy (moments (k+1)!) gives J_TE/J_FN = 1 - 1/b + 19/(9 b^2) - 20/(3 b^3) + O(b^-4).
    MEASURED remainder (R - series) b^4 = +21..+26 over b = 16..60; the gate allows 40/b^4."""
    from dynameta.reliability.leakage import fowler_nordheim_current, tsu_esaki_current
    t = 20e-9                                          # V_ox = 8-30 V >> phi_b: triangular for all electrons
    j = tsu_esaki_current(F * t, t, phi_b_eV=PHI, E_F_eV=50.0, m_ox_over_m0=MOX, T_K=0.0)
    jfn, B = _fn_ref(F)
    # the code's FN is Lenzlinger-Snow; it uses the CODATA-ROUNDED hbar literal (6e-10 rel vs h/2pi),
    # which the exponent amplifies by b ~ 16-60 -> agreement to ~4e-8, not machine precision
    assert fowler_nordheim_current(F) == pytest.approx(jfn, rel=1e-7)
    b = B / F
    series = 1 - 1 / b + 19 / (9 * b ** 2) - 20 / (3 * b ** 3)
    assert abs(j / jfn - series) <= 40.0 / b ** 4


def test_k3_crossover_schuegraf_hu_fn_and_tsu_esaki():
    from dynameta.reliability.leakage import (direct_tunneling_current, fowler_nordheim_current,
                                              schuegraf_hu_current, tsu_esaki_current)
    t = 4e-9
    V_hi = np.array([3.1, 3.3, 4.0, 5.0, 6.0])
    # above the barrier the S-H prefactor is exactly 1: IDENTICAL to FN (not a tolerance)
    assert np.array_equal(schuegraf_hu_current(V_hi, t), fowler_nordheim_current(V_hi / t))
    # below it: S-H = exponent-only DT x (1 - sqrt(1-u))^-2  (the audit's "2-32x" factor)
    V_lo = np.array([1.0, 1.5, 2.0, 2.8])
    u = V_lo / PHI
    assert np.allclose(schuegraf_hu_current(V_lo, t),
                       direct_tunneling_current(V_lo, t) / (1 - np.sqrt(1 - u)) ** 2, rtol=1e-13)
    # Tsu-Esaki (0 K, metal supply) approaches the K2 series as the barrier turns triangular
    _, B = _fn_ref(1.0)
    dev, b4 = [], []
    for V in (3.3, 3.6, 4.0, 4.5, 5.0):
        F = V / t
        b = B / F
        ser = 1 - 1 / b + 19 / (9 * b ** 2) - 20 / (3 * b ** 3)
        r = tsu_esaki_current(V, t, phi_b_eV=PHI, E_F_eV=50.0, T_K=0.0) / _fn_ref(F)[0]
        dev.append(abs(r - ser))
        b4.append(40.0 / b ** 4)
    # MEASURED |R - series|: 4.8e-2 (3.3 V), 1.6e-3 (3.6 V), then the series' own O(b^-4) remainder
    # (0.9-1.6e-4) from 4.0 V on -- the deep electrons' trapezoid correction dies as exp(-(qV - phi)c/F)
    assert dev[0] > 0.04                                 # 0.2 V above phi_b: still trapezoidal (+5 %)
    assert dev[1] < dev[0] / 10.0                        # 0.5 V above: >10x closer
    assert all(d <= lim for d, lim in zip(dev[2:], b4[2:]))   # >= 0.9 V above: the triangular series
    # V_ox <= 0 -> exactly 0 (magnitude convention)
    assert schuegraf_hu_current(0.0, t) == 0.0
    assert tsu_esaki_current(-1.0, t, phi_b_eV=PHI, E_F_eV=1.0) == 0.0


@pytest.mark.parametrize("F", [6e8, 1.0e9, 2.0e9])
def test_k4_murphy_good_thermal_factor(F):
    """Linearised exponent (slope c/F at E_F, c = 2 sqrt(2 m_ox phi)/(q hbar)) + Fermi supply ->
    J(T)/J(0) = x/sin(x), x = pi c kT / F (Murphy & Good, Phys. Rev. 102, 1464 (1956)). The neglected
    curvature is O(F/B): the gate allows F/B (MEASURED residual 0.3-0.6 %, i.e. < F/B/4)."""
    from dynameta.reliability.leakage import tsu_esaki_current
    t = 20e-9
    c = 2 * math.sqrt(2 * MOX * M0 * PHI * Q) / (HBAR * Q * F)
    x = math.pi * c * KB * 300.0
    assert x < 2.0                                       # MG is valid for x < pi
    r = (tsu_esaki_current(F * t, t, phi_b_eV=PHI, E_F_eV=50.0, T_K=300.0)
         / tsu_esaki_current(F * t, t, phi_b_eV=PHI, E_F_eV=50.0, T_K=0.0))
    _, B = _fn_ref(F)
    assert abs(r / (x / math.sin(x)) - 1.0) <= F / B


def test_k5_direct_tunneling_small_bias_is_ohmic_and_exceeds_fn():
    from dynameta.reliability.leakage import (fowler_nordheim_current, schuegraf_hu_current,
                                              tsu_esaki_current)
    t = 2e-9
    g = [tsu_esaki_current(V, t, phi_b_eV=PHI, E_F_eV=0.1, T_K=300.0) / V for V in (1e-4, 2e-4, 4e-4)]
    assert max(g) / min(g) - 1 < 3e-3                   # J ~ V: supply ~ qV (MEASURED 1.1e-3)
    # the S-H closed form does NOT go to zero (its linearisation breaks down) -- documented limit
    assert schuegraf_hu_current(1e-4, t) > 1e3 * tsu_esaki_current(1e-4, t, phi_b_eV=PHI, E_F_eV=0.1)
    # direct tunneling through 2 nm at 1 V exceeds the FN extrapolation by orders of magnitude
    assert tsu_esaki_current(1.0, t, phi_b_eV=PHI, E_F_eV=0.1) > 1e6 * fowler_nordheim_current(1.0 / t)


def test_tsu_esaki_guards():
    from dynameta.reliability.leakage import tsu_esaki_current, wkb_trapezoid_exponent
    with pytest.raises(ValueError):
        tsu_esaki_current(1.0, 0.0, phi_b_eV=PHI, E_F_eV=1.0)
    with pytest.raises(ValueError):
        tsu_esaki_current(1.0, 2e-9, phi_b_eV=PHI, E_F_eV=0.0)
    with pytest.raises(ValueError):
        tsu_esaki_current(1.0, 2e-9, phi_b_eV=PHI, E_F_eV=1.0, T_K=-1.0)
    # the cancellation-free WKB form tends to the rectangular barrier as V -> 0
    th = wkb_trapezoid_exponent(0.0, 1e-9, 2e-9, phi_b_eV=PHI)
    rect = 2 * math.sqrt(2 * MOX * M0 * PHI * Q) * 2e-9 / HBAR
    assert th == pytest.approx(rect, rel=1e-8)
    arr = tsu_esaki_current(np.array([[0.5, -1.0], [0.0, 1.0]]), 2e-9, phi_b_eV=PHI, E_F_eV=0.1)
    assert arr.shape == (2, 2) and arr[0, 1] == 0.0 and arr[1, 0] == 0.0 and arr[1, 1] > arr[0, 0] > 0.0
