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


# =====================================================================================================
# v0.15 (plan item 4.1.1): the TWO-SIDED Tsu-Esaki kernel on a fixed energy grid + the stack WKB
# (dynameta.reliability.leakage.tsu_esaki_two_sided / wkb_layer_exponent). Every oracle below is coded
# HERE: the WKB exponent as a scipy quad of 2 kappa over x with brentq turning points, the supply
# integral as a scipy quad, the shot noise as a NESTED quad over (E_x, E_t) of the Landauer-Buttiker
# partition f_s (1 - f_g) + f_g (1 - f_s). On the pre-v0.15 tree every K6-K11 test fails (ImportError:
# no tsu_esaki_two_sided / wkb_layer_exponent / tunnel_energy_grid).
# =====================================================================================================
KT300 = KB * 300.0


def _quad_theta(E, faces_U, ts, masses):
    """2 Int kappa dx through a piecewise-LINEAR conduction edge: faces_U[j] = (U_a, U_b) [J] across
    layer j of thickness ts[j]; turning points by brentq, the integrable sqrt ends by quad."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    th = 0.0
    for (Ua, Ub), t, m in zip(faces_U, ts, masses):
        ga, gb = Ua - E, Ub - E
        if ga <= 0 and gb <= 0:
            continue

        def kappa(x, Ua=Ua, Ub=Ub, t=t, m=m):
            U = Ua + (Ub - Ua) * x / t
            return math.sqrt(2 * m * M0 * max(U - E, 0.0)) / HBAR
        lo, hi = 0.0, t
        if ga * gb < 0:                                   # one turning point inside the layer
            xt = brentq(lambda x: Ua + (Ub - Ua) * x / t - E, 0.0, t, xtol=1e-30, rtol=1e-15)
            lo, hi = (0.0, xt) if ga > 0 else (xt, t)
        th += 2.0 * quad(kappa, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return th


def _faces(d, lay):
    faces = [0.0] + list(d)
    U = [((off * Q + Q * faces[j]), (off * Q + Q * faces[j + 1])) for j, (off, t, m) in enumerate(lay)]
    return U, [t for _, t, _ in lay], [m for _, _, m in lay]


def _te2_ref(xs, xg, d, lay, ms=1.0):
    """scipy quad of q m kT/(2 pi^2 hbar^3) Int_0 T(eps) kT[ln(1+e^(xs-eps/kT)) - ln(1+e^(xg-eps/kT))]."""
    from scipy.integrate import quad
    kT = KT300
    U, ts, ms_ = _faces(d, lay)

    def f(eps):
        th = _quad_theta(eps, U, ts, ms_)
        return math.exp(-th) * kT * (np.logaddexp(0, xs - eps / kT) - np.logaddexp(0, xg - eps / kT))
    top = max(xs, xg, 0.0) * kT + 4.5 * Q
    pts = sorted({p for p in [max(xs, 0) * kT, max(xg, 0) * kT] + [u for pair in U for u in pair]
                  if 0.0 < p < top})
    val = quad(f, 0.0, top, points=pts or None, limit=2000, epsabs=0.0, epsrel=1e-12)[0]
    return Q * ms * M0 / (2 * math.pi ** 2 * HBAR ** 3) * val


# (x_s, x_g, d_V, layers=(offset_eV, t_m, m/m0)...): DT/FN, semiconductor (non-degenerate / degenerate)
# and gate-like supplies, E_Fs below and above E_c, single layers and stacks.
_K6_CASES = [
    (-5.0, -5.0 - 1.0 / 0.025852, (-1.0,), ((3.1, 2e-9, 0.42),)),            # DT, n-surface below E_c
    (2.0, 2.0 - 2.0 / 0.025852, (-2.0,), ((3.1, 2e-9, 0.42),)),             # DT, degenerate accumulation
    (1.0, 1.0 - 4.0 / 0.025852, (-4.0,), ((3.1, 4e-9, 0.42),)),             # FN (V_ox > phi_b)
    (-20.0, 60.0, (1.5,), ((3.1, 2e-9, 0.42),)),                            # the gate emits (reverse)
    (0.5, -30.0, (-0.6, -1.4), ((3.1, 0.8e-9, 0.42), (1.5, 2e-9, 0.18))),   # SiO2 / HfO2-like stack
    (0.5, -30.0, (-0.3, -0.8, -1.4), ((3.1, 0.6e-9, 0.42), (2.0, 1e-9, 0.3), (1.2, 1e-9, 0.2))),
    (-3.0, -3.0 - 0.3 / 0.025852, (-0.3,), ((3.1, 1.5e-9, 0.42),)),          # deep DT, small bias
    (3.868, 3.868 - 3.182 / 0.025852, (-3.182,), ((3.2, 3e-9, 0.42),)),     # trapezoid kink at the peak
]
_K6_KINK = (4, 7)       # cases whose turning-point kink (theta only C^1) sits inside the supply window


def _layers(spec):
    from dynameta.reliability.leakage import TunnelLayer
    return [TunnelLayer(o, t, m) for o, t, m in spec]


def _grid(xs, xg, step=0.125, extra_eV=4.5):
    from dynameta.reliability.leakage import tunnel_energy_grid
    return tunnel_energy_grid(KT300, max(xs, xg, 0.0) * KT300 + extra_eV * Q, step_kT=step)


@pytest.mark.parametrize("case", range(len(_K6_CASES)))
def test_k6_two_sided_matches_independent_quad(case):
    """K6: the fixed-grid two-sided kernel (default step kT/8) == an independent scipy quad of the same
    integral, the WKB exponent itself a quad of 2 kappa dx. Gate 1e-8 (K1's). MEASURED: <= 7e-15 on the
    smooth cases; on the two KINK cases (a turning point at the supply peak, where theta is only C^1)
    1.4e-10 (stack) and 3.8e-9 (trapezoid at V_ox = 3.182 V: the worst V of a 2.9-3.4 V sweep), falling
    as step^2.5 (kT/4: 2.1e-8, kT/16: 6.7e-10). Refute: the grid at half the panel width also passes;
    a 4x COARSER grid (kT/2) misses 1e-8 on the kink cases (MEASURED 2.5e-8 / > 1e-7) -- the gate sees
    the quadrature; on the reverse case the semiconductor's one-way part alone is < 1e-3 of J."""
    from dynameta.reliability.leakage import tsu_esaki_two_sided
    xs, xg, d, lay = _K6_CASES[case]
    r = tsu_esaki_two_sided(_grid(xs, xg), [xs], [xg], [list(d)], _layers(lay))
    ref = _te2_ref(xs, xg, d, lay)
    assert ref != 0.0 and r.tail[0] < 1e-30
    print("K6 case {}: rel err {:.3e}".format(case, r.J[0] / ref - 1))
    assert r.J[0] == pytest.approx(ref, rel=1e-8)
    r2 = tsu_esaki_two_sided(_grid(xs, xg, step=0.0625), [xs], [xg], [list(d)], _layers(lay))
    assert r2.J[0] == pytest.approx(ref, rel=1e-8)
    if case in _K6_KINK:
        r4 = tsu_esaki_two_sided(_grid(xs, xg, step=0.5), [xs], [xg], [list(d)], _layers(lay))
        assert abs(r4.J[0] / ref - 1) > 1e-8
    assert r.J_fwd[0] - r.J_bwd[0] == pytest.approx(r.J[0], rel=1e-12, abs=1e-12 * r.J_fwd[0])
    if xg > xs:                                          # the gate supply carries the current
        assert abs(r.J_fwd[0] / ref) < 1e-3


def test_k7_identity_with_the_one_sided_kernel():
    """K7 (consistency): with E_Fg = E_Fs - qV the two-sided kernel IS the shipped one-sided
    tsu_esaki_current (barrier phi_b above E_Fs -> offset phi_b + E_F over E_c). Budget 1e-13 on cases
    with no turning-point kink in the supply window (the two quadratures' own errors, K1 / K6; MEASURED
    <= 6e-15). NOTE (MEASURED, against the plan's premise): sweeping V over 1.0-1.4 V and 5-6 V (FN) the
    ADAPTIVE kernel showed no step above 7e-15 relative between neighbouring V -- the case for the fixed
    grid is the EXACT derivative of a fixed discrete sum (K9), not adaptive-grid noise."""
    from dynameta.reliability.leakage import TunnelLayer, tsu_esaki_current, tsu_esaki_two_sided
    EF = 0.1
    xs = EF * Q / KT300
    g = _grid(xs, 0.0)
    for V, t in ((1.0, 2e-9), (0.2, 1.5e-9), (2.0, 2e-9)):
        r = tsu_esaki_two_sided(g, [xs], [xs - Q * V / KT300], [[-V]], [TunnelLayer(3.1 + EF, t, 0.42)])
        old = tsu_esaki_current(V, t, phi_b_eV=3.1, E_F_eV=EF, T_K=300.0)
        assert r.J[0] == pytest.approx(old, rel=1e-13)


def test_k8_detailed_balance_is_exact_and_the_swap_is_odd():
    """K8: E_Fs == E_Fg -> J == 0.0 EXACTLY (the integrand vanishes pointwise), on random asymmetric
    stacks; swapping the two Fermi levels negates J bit for bit."""
    from dynameta.reliability.leakage import TunnelLayer, tsu_esaki_two_sided
    rng = np.random.default_rng(7)
    for _ in range(20):
        L = int(rng.integers(1, 4))
        lay = [TunnelLayer(float(rng.uniform(0.8, 3.5)), float(rng.uniform(0.4e-9, 2e-9)),
                           float(rng.uniform(0.1, 0.6))) for _ in range(L)]
        d = np.cumsum(rng.uniform(-1.0, 1.0, size=L))[None, :]
        x = float(rng.uniform(-30.0, 10.0))
        g = _grid(x, x)
        assert tsu_esaki_two_sided(g, [x], [x], d, lay).J[0] == 0.0
        y = x + float(rng.uniform(-5.0, 5.0))
        a = tsu_esaki_two_sided(g, [x], [y], d, lay).J[0]
        b = tsu_esaki_two_sided(g, [y], [x], d, lay).J[0]
        assert a == -b and a != 0.0


def _fd4(f, x, h):
    """4th-order central difference (one Richardson step) -- the K9 oracle."""
    d1 = (f(x + h) - f(x - h)) / (2 * h)
    d2 = (f(x + 2 * h) - f(x - 2 * h)) / (4 * h)
    return (4 * d1 - d2) / 3


@pytest.mark.parametrize("xs,xg,d,lay", _K6_CASES)
def test_k9_analytic_derivatives_are_exact_for_the_discrete_sum(xs, xg, d, lay):
    """K9: dJ/dx_s, dJ/dx_g and dJ/dd_k == 4th-order central differences of the SAME discrete sum (the
    grid is fixed, so the sum is a C^1 function of its inputs). Budget 1e-9 of the derivative scale:
    h^4 truncation (~1e-12 at h = 1e-3 in x / 2.5e-5 V in d) + round-off eps |J| / h (~1e-12). Where a
    grid node sits on a turning point (theta is C^1, its second derivative ~ (A)_+^(-1/2) is unbounded:
    the stencil error of THAT node falls only as sqrt(h)) the gate is a CONVERGENCE gate: the error at
    h/16 must fall to <= 1/2 of the error at h (sqrt(16) = 4 expected) -- a wrong analytic derivative
    would not converge. MEASURED: <= 1e-11 on the smooth cases; stack kink 6.4e-8 at h -> 2.2e-12 at h/16;
    trapezoid kink 6.3e-10."""
    from dynameta.reliability.leakage import tsu_esaki_two_sided
    g = _grid(xs, xg)
    layers = _layers(lay)

    def J(a, b, dv):
        return tsu_esaki_two_sided(g, [a], [b], [dv], layers).J[0]
    r = tsu_esaki_two_sided(g, [xs], [xg], [list(d)], layers)
    scale = max(abs(r.dJ_dxs[0]), abs(r.dJ_dxg[0]), float(np.max(np.abs(r.dJ_dd))) * 0.025852)
    checks = [(lambda a: J(a, xg, list(d)), xs, 1e-3, r.dJ_dxs[0], 1.0),
              (lambda b: J(xs, b, list(d)), xg, 1e-3, r.dJ_dxg[0], 1.0)]
    for k in range(len(lay)):
        def fk(v, k=k):
            dv = list(d)
            dv[k] = v
            return J(xs, xg, dv)
        checks.append((fk, d[k], 2.5e-5, r.dJ_dd[0, k], 0.025852))
    worst = 0.0
    for f, x0, h, an, unit in checks:
        e1 = abs(_fd4(f, x0, h) - an) * unit
        if e1 > 1e-9 * scale:                              # a kink inside the stencil: convergence form
            e2 = abs(_fd4(f, x0, h / 16.0) - an) * unit
            assert e2 <= 0.5 * e1, (e1 / scale, e2 / scale)
            e1 = e2
        worst = max(worst, e1 / scale)
    print("K9 max rel err {:.3e}".format(worst))
    assert worst <= 1e-7


def test_k10_stack_wkb_exponent():
    """K10: (a) one layer == the trapezoid closed form RE-TYPED here (<= 4 ulp: the same algebra,
    re-associated); (b) two identical layers == one of the summed thickness (1e-14); (c) three-layer
    stacks, incl. a dip below E and a turning point inside a layer, == quad of 2 kappa dx with brentq
    turning points (1e-10); (d) the derivatives are continuous across a turning-point switch (a kink
    would break Newton)."""
    from dynameta.reliability.leakage import wkb_layer_exponent, wkb_trapezoid_exponent
    m = MOX * M0
    for V, t in ((1.0, 2e-9), (4.0, 3e-9), (3.1, 2e-9)):
        e = np.linspace(-3.5 * Q, 0.5 * Q, 401)
        U0 = PHI * Q + e
        a, b = np.maximum(U0, 0), np.maximum(U0 - Q * V, 0)
        sa, sb = np.sqrt(a), np.sqrt(b)
        ref = np.where(b > 0, Q * V * (a + sa * sb + b) / np.where(b > 0, sa + sb, 1), a * sa) \
            * 4 * math.sqrt(2 * m) / (3 * HBAR * Q * V / t)
        th = wkb_trapezoid_exponent(e, V, t, phi_b_eV=PHI)
        assert np.all(np.abs(th - ref) <= 4 * np.spacing(np.maximum(np.abs(ref), 1e-300)))
    A = np.linspace(-0.5, 3.0, 57) * Q
    B = A - 1.3 * Q
    one = wkb_layer_exponent(A, B, 2e-9, 0.42)
    two = wkb_layer_exponent(A, A - 0.65 * Q, 1e-9, 0.42) + wkb_layer_exponent(A - 0.65 * Q, B, 1e-9, 0.42)
    assert np.allclose(two, one, rtol=1e-14, atol=1e-14)
    stacks = [
        [(3.1 * Q, 2.9 * Q), (1.6 * Q, 1.0 * Q), (2.0 * Q, 2.4 * Q)],     # a dip below E in the middle
        [(3.1 * Q, 2.1 * Q), (1.1 * Q, -0.2 * Q), (1.4 * Q, 0.9 * Q)],    # turning point inside layer 1
    ]
    for U in stacks:
        ts, ms_ = [0.7e-9, 1.3e-9, 0.9e-9], [0.42, 0.18, 0.3]
        for E in (0.05 * Q, 1.2 * Q, 1.5 * Q):
            th = sum(float(wkb_layer_exponent(Ua - E, Ub - E, t, mm)) for (Ua, Ub), t, mm in zip(U, ts, ms_))
            assert th == pytest.approx(_quad_theta(E, U, ts, ms_), rel=1e-10, abs=1e-12)
    for A0, B0 in ((2.0 * Q, 0.0), (1.0 * Q, 1.0 * Q)):   # B = 0 (trapezoid -> triangle), A = B (flat)
        h = 1e-12 * Q
        _, aL, bL = wkb_layer_exponent(A0, B0 - h, 2e-9, 0.42, derivatives=True)
        _, aR, bR = wkb_layer_exponent(A0, B0 + h, 2e-9, 0.42, derivatives=True)
        assert float(aL) == pytest.approx(float(aR), rel=1e-5)
        assert float(bL) == pytest.approx(float(bR), rel=1e-5)


def _noise_ref(xs, xg, d, lay):
    """S = 2 q^2 (m/(2 pi^2 hbar^3)) Int dE_x T Int_0^inf dE_t [f_s (1 - f_g) + f_g (1 - f_s)] by NESTED
    quad (T << 1: the T(1 - T) Landauer-Buttiker correction is neglected, as in the kernel)."""
    from scipy.integrate import quad
    kT = KT300
    U, ts, ms_ = _faces(d, lay)

    def part(et, eps):                                  # f_s (1 - f_g) + f_g (1 - f_s), in exp form
        zs, zg = (eps + et) / kT - xs, (eps + et) / kT - xg
        return (math.exp(-np.logaddexp(0, zs) - np.logaddexp(0, -zg))
                + math.exp(-np.logaddexp(0, zg) - np.logaddexp(0, -zs)))

    def inner(eps):
        top = (max(xs - eps / kT, xg - eps / kT, 0.0) + 60.0) * kT
        pts = [p for p in (xs * kT - eps, xg * kT - eps) if 0 < p < top]
        scale = max([part(0.0, eps)] + [part(p, eps) for p in pts]) * kT
        return quad(part, 0.0, top, args=(eps,), epsabs=1e-15 * scale, epsrel=1e-12, limit=400,
                    points=pts or None)[0]

    def f(eps):
        return math.exp(-_quad_theta(eps, U, ts, ms_)) * inner(eps)
    top = max(xs, xg, 0.0) * kT + 3.0 * Q
    pts = [p for p in (xs * kT, xg * kT) if 0 < p < top]
    val = quad(f, 0.0, top, epsabs=0.0, epsrel=1e-11, limit=400, points=pts or None)[0]
    return 2 * Q * Q * M0 / (2 * math.pi ** 2 * HBAR ** 3) * val


@pytest.mark.parametrize("xs,dlt", [(-5.0, 0.0), (-5.0, 0.7), (2.0, 0.0), (2.0, 3.0), (-3.0, 45.0),
                                    (4.0, -2.0), (-5.0, 3e-5)])
def test_k11_pair_noise_is_the_landauer_buttiker_partition(xs, dlt):
    """K11: the kernel's S = 2 q J coth(Delta/2) (closed form of the transverse partition integral, audit
    note 2.3) == a nested quad over (E_x, E_t) of T [f_s (1 - f_g) + f_g (1 - f_s)]: 1e-8 (the nested quad
    is converged to ~1e-11; MEASURED <= 1.1e-10, the small-Delta series case). Equilibrium: S == 4 kT dJ/dV with dJ/dV = (q/kT) dJ/dx_s, K9's derivative
    (1e-12: the same sums). Delta = 45: S == 2 q J (coth - 1 ~ 6e-20). The small-Delta branch (|Delta|
    < 1e-4) is the O(Delta^2) series (case Delta = 3e-5). Refute: the 'one-sided' shot sum
    2 q (J_fwd + J_bwd), which drops the Pauli term -2 N_sg, overcounts the DEGENERATE (x_s = 2)
    equilibrium by > 5 %, and S = 2 q |J| gives 0 there."""
    from dynameta.reliability.leakage import tsu_esaki_two_sided
    d, lay = (-1.0,), ((3.1, 1.5e-9, 0.42),)
    xg = xs - dlt
    r = tsu_esaki_two_sided(_grid(xs, xg), [xs], [xg], [list(d)], _layers(lay))
    ref = _noise_ref(xs, xg, d, lay)
    assert r.S[0] == pytest.approx(ref, rel=1e-8)
    if dlt == 0.0:
        assert r.J[0] == 0.0 and 2 * Q * abs(r.J[0]) == 0.0
        assert r.S[0] == pytest.approx(4 * KT300 * (Q / KT300) * r.dJ_dxs[0], rel=1e-12)
        one_sided = 2 * Q * (r.J_fwd[0] + r.J_bwd[0])
        print("K11 one-sided / exact at x_s = {}: {:.6f}".format(xs, one_sided / r.S[0]))
        if xs > 0:
            assert one_sided / r.S[0] - 1 > 0.05
    if dlt > 40:
        assert r.S[0] == pytest.approx(2 * Q * r.J[0], rel=1e-15)
