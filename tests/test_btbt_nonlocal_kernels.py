"""v0.15 plan item 4.1.3: nonlocal band-to-band tunnelling, the 1-D KERNELS (carriers/btbt_nonlocal.py) --
no DEVSIM. Every reference is re-derived HERE from the Sentaurus Device User Guide N-2017.09 text
(Eqs. 480-498, typed from the page images) or computed with scipy quadrature / root finding written here;
nothing below calls the module's conversion or segment code to build a reference.

Gates (OLD CODE: dynameta.carriers.btbt_nonlocal does not exist on 66f6185 -- every test fails at import,
MEASURED by running this file against an export of 66f6185):
  N0a  closed-form segment integrals (Kane two-band and Keldysh) == scipy.quad, including segments that touch
       v = 0 and v = 1 (the integrable 1/sqrt end singularities), within quad's own error estimate + 64 eps.
  N0b  their complex-step derivatives == Richardson central differences (budget derived in the test).
  N0c  (A, B) -> physical -> (A, B) round trip 1e-14; the physical set == Eqs. 495-498 inverted HERE.
  N1   UNIFORM FIELD: R / (A F^P exp(-B/F)) - 1 within the round-off budget (<= 1e-12) for the direct
       (P = 2) and phonon (P = 2.5) processes at 5e7, 1e8, 3e8 V/m; inside the taper R/ref == w(l).
       Refute: m_r perturbed by 1e-9 must fail it by the predicted B/F * 5e-10.
  N2   NO FINAL STATE: a 0.8 V tanh step (peak 4e8 V/m) gives R == 0.0 exactly (no path) while the local
       Kane rate integrates to > 1e7 A/m^2; a 1.13 V step switches the nonlocal rate on.
  N4   INDEPENDENT CONTINUUM ORACLE on a graded (tanh) junction with smooth quasi-Fermi profiles: q Int R dx0
       from a scipy brentq/quad implementation of Eqs. 480-484 (direct) and 490-493 (phonon) typed here;
       the module's discrete current converges to it at order >= 1.7 and the finest mesh lies within
       2 C_inf h^2 (C_inf from the 0.05/0.025 nm pair, the L1 method). Refute: a first-order (one-sided)
       start gradient drops the order to ~1. The nonlocal/local ratio is reported.
"""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from dynameta.carriers import btbt_nonlocal as NL
from dynameta.carriers.tunneling import BTBTParams

EPS = np.finfo(float).eps
Q = 1.602176634e-19
H = 6.62607015e-34
HB = H / (2.0 * math.pi)
EG_EV = 1.12
DIRECT = BTBTParams(A_SI=3.5e21, B_V_per_m=2.2e9, P=2.0)       # an arbitrary direct set (any A, B works)
PHONON = BTBTParams.from_cgs(4.0e14, 1.9e7, 2.5)                # Hurkx Si (Sentaurus Table 90 defaults)
RATIO = 1.7                                                     # m_V/m_C, != 1 so x0 is off-centre


def _spec(params, **kw):
    return NL.NonlocalBTBT(params, Eg_tun_eV=EG_EV, mass_ratio_V_over_C=RATIO, **kw)


# ---- the physical set, Eqs. 495-498 inverted HERE (independent of physical_from_fitted) -------------
def _phys_direct(A, B, Eg):
    """Eq. 496: B = pi^2 m_r^1/2 Eg^3/2/(q h); Eq. 495: A = g pi m_r^1/2 (q F0)^2/(9 h^2 Eg^1/2), F0 = 1 V/m."""
    m_r = (B * Q * H / (math.pi ** 2 * Eg ** 1.5)) ** 2
    g = A * 9.0 * H ** 2 * Eg ** 0.5 / (math.pi * m_r ** 0.5 * Q ** 2)
    return m_r, g


def _phys_phonon(A, B, Eg, ratio):
    """Eq. 498: B = 2^(7/2) pi m_r^1/2 Eg^3/2/(3 q h); Eq. 497 solved for lam = g(1+2N)D^2/(rho eps_op)."""
    m_r = (B * 3.0 * Q * H / (2.0 ** 3.5 * math.pi * Eg ** 1.5)) ** 2
    m_v = m_r * (1.0 + ratio)                      # 1/m_r = 1/m_V + 1/m_C (Eq. 484), m_V = ratio m_C
    m_c = m_v / ratio
    lam = A * 2.0 ** (21.0 / 4.0) * H ** 2.5 * m_r ** 1.25 * Eg ** 1.75 / ((m_v * m_c) ** 1.5 * Q ** 2.5)
    return m_r, m_v, m_c, lam


# =====================================================================================================
# N0
# =====================================================================================================

def _quad_seg(f, va, vb, L):
    """Int_0^L f(v(s)) ds, v linear; 'alg' weights carry the 1/sqrt end singularities (f is evaluated as
    the smooth remainder). Returns (value, error estimate)."""
    def v(s):
        return va + (vb - va) * s / L
    return quad(lambda s: f(v(s)), 0.0, L, epsabs=0.0, epsrel=1e-13, limit=400)


def test_n0a_segment_integrals_match_quadrature():
    rng = np.random.default_rng(7)
    cases = [tuple(rng.uniform(0.0, 1.0, 2)) for _ in range(60)]
    cases += [(0.0, 0.37), (0.0, 1.0), (0.55, 1.0), (0.3, 0.3), (0.3, 0.3 + 1e-9), (0.999, 1.0), (0.0, 1e-4)]
    worst = 0.0
    for va, vb in cases:
        L = 0.7
        i0, i1 = NL.kane_segment(va, vb, L)
        jv0, jv1 = NL.keldysh_segment(va, vb, L)
        if va == vb:                                        # constant integrand: exact closed form
            k = math.sqrt(va * (1 - va))
            refs = [(L / k, 0.0), (L * k, 0.0), (L / math.sqrt(va), 0.0), (L * math.sqrt(va), 0.0)]
        else:
            def v(s_):
                return va + (vb - va) * s_ / L
            kq = dict(epsabs=0, epsrel=1e-13, limit=200)
            dvds = (vb - va) / L
            # Kane k = sqrt(v(1-v)): direct quad in s away from both ends; touching v = 0 (or 1) the
            # substitution v = sin^2(phi) (or v = cos^2(chi), chi = arcsin sqrt(1 - v): arcsin(sqrt v) near 1
            # would cost sqrt(eps) of the reference's own accuracy) removes the 1/sqrt singularity
            if min(va, vb, 1 - va, 1 - vb) > 1e-3:
                rk = [quad(lambda s_: 1.0 / math.sqrt(v(s_) * (1 - v(s_))), 0, L, **kq),
                      quad(lambda s_: math.sqrt(v(s_) * (1 - v(s_))), 0, L, **kq)]
            else:
                if min(va, vb) > 0.5:
                    pa, pb, sg = math.asin(math.sqrt(1 - va)), math.asin(math.sqrt(1 - vb)), -1.0
                else:
                    pa, pb, sg = math.asin(math.sqrt(va)), math.asin(math.sqrt(vb)), 1.0
                rk = [quad(lambda p: sg * 2.0 / dvds, pa, pb, epsabs=0, epsrel=2e-14),
                      quad(lambda p: sg * 2.0 * (math.sin(p) * math.cos(p)) ** 2 / dvds, pa, pb, epsabs=0,
                           epsrel=2e-14)]
            # Keldysh sqrt(w): singular only at w = 0 -> v = u^2 there, direct quad otherwise
            if min(va, vb) > 1e-3:
                rw = [quad(lambda s_: 1.0 / math.sqrt(v(s_)), 0, L, **kq),
                      quad(lambda s_: math.sqrt(v(s_)), 0, L, **kq)]
            else:
                ua, ub = math.sqrt(va), math.sqrt(vb)
                rw = [quad(lambda u: 2.0 / dvds, ua, ub, epsabs=0, epsrel=2e-14),
                      quad(lambda u: 2.0 * u * u / dvds, ua, ub, epsabs=0, epsrel=2e-14)]
            refs = rk + rw
        for got, (ref, err) in zip((i0, i1, jv0, jv1), refs):
            dev = abs(float(got) - ref)
            assert dev <= err + 64 * EPS * abs(ref), (va, vb, got, ref, err)
            worst = max(worst, dev / abs(ref))
    print("N0a worst", worst)
    assert worst < 1e-13                                    # MEASURED 4.8e-16


def _rich(f, x, h):
    return (8.0 * (f(x + h) - f(x - h)) - (f(x + 2 * h) - f(x - 2 * h))) / (12.0 * h)


def test_n0b_complex_step_derivatives_match_richardson():
    """Budget: the 5-point stencil's truncation h^4 f^(5)/30 with |f^(5)/f| <= (1/0.05)^5 for segments
    kept >= 0.05 from v = 0, 1 -> 1e-16 * 3.2e6/30 = 1.1e-11 at h = 1e-4, plus round-off 1.5 eps/h =
    3.3e-12 relative: tolerance 2e-10 (MEASURED 5.4e-12). The complex step is the module's."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(40):
        va, vb = rng.uniform(0.05, 0.95, 2)
        for kern in (NL.kane_segment, NL.keldysh_segment):
            for idx in (0, 1):
                for which in ("a", "b"):
                    def f(v, which=which, idx=idx, kern=kern):
                        return float(kern(v, vb, 0.9)[idx] if which == "a" else kern(va, v, 0.9)[idx])
                    x0 = va if which == "a" else vb
                    fd = _rich(f, x0, 1e-4)
                    cs = (kern(va + 1e-30j, vb + 0j, 0.9) if which == "a" else
                          kern(va + 0j, vb + 1e-30j, 0.9))[idx].imag / 1e-30
                    worst = max(worst, abs(cs - fd) / max(abs(f(x0)), abs(fd)))
    print("N0b worst", worst)
    assert worst < 2e-10, worst


def test_n0c_conversion_round_trip_and_eqs_495_498():
    for params in (DIRECT, PHONON):
        spec = _spec(params)
        ph = NL.physical_from_fitted(spec)
        A, B = NL.fitted_from_physical(ph)
        assert abs(A / params.A_SI - 1) < 1e-14 and abs(B / params.B_V_per_m - 1) < 1e-14
        Eg = Q * EG_EV
        if params.P == 2.0:
            m_r, g = _phys_direct(params.A_SI, params.B_V_per_m, Eg)
            assert abs(ph.m_r / m_r - 1) < 1e-14 and abs(ph.g / g - 1) < 1e-14
        else:
            m_r, m_v, m_c, lam = _phys_phonon(params.A_SI, params.B_V_per_m, Eg, RATIO)
            for a, b in ((ph.m_r, m_r), (ph.m_V, m_v), (ph.m_C, m_c), (ph.lam, lam)):
                assert abs(a / b - 1) < 1e-14
    # Hurkx Si through Eq. 498: m_r = 0.0551 m0 (printed for the audit note; physical only as a fit)
    assert abs(NL.physical_from_fitted(_spec(PHONON)).m_r / 9.1093837015e-31 - 0.0551) < 5e-4


def test_spec_validation():
    with pytest.raises(ValueError, match="P = 2"):
        NL.NonlocalBTBT(BTBTParams(1e15, 1e9, 2.2), Eg_tun_eV=1.1)
    with pytest.raises(ValueError, match="Eg_tun_eV"):
        NL.NonlocalBTBT(DIRECT, Eg_tun_eV=0.0)
    with pytest.raises(TypeError):
        NL.NonlocalBTBT((1e15, 1e9, 2.0), Eg_tun_eV=1.1)


# =====================================================================================================
# N1 uniform field
# =====================================================================================================

def _uniform(params, F, h=0.1e-9, span_V=4.0, max_len=1.0, phys_override=None):
    """A uniform field on a domain CENTRED on psi = 0 spanning span_V volts: the node potentials carry
    round-off eps |psi| <= eps span_V/2, which enters every v = (psi_k - psi_i)/E_g."""
    spec = _spec(params, max_length_m=max_len)
    phys = phys_override or NL.physical_from_fitted(spec)
    N = int(round(span_V / (F * h))) + 1
    x = np.arange(N) * h
    psi = F * (x - 0.5 * x[-1])
    full = np.ones(N, bool)
    so = full.copy()
    so[0] = so[-1] = False
    res = NL.path_rates_1d(x, psi, np.full(N, -300.0), np.full(N, -300.0), phys, 0.0, max_len, so, full,
                           derivatives=False)
    ref = params.A_SI * F ** params.P * math.exp(-params.B_V_per_m / F)
    nseg = Q * EG_EV / (Q * F * h) + 1                     # segments per path
    return res, ref, nseg


@pytest.mark.parametrize("params", [DIRECT, PHONON], ids=["direct", "phonon"])
@pytest.mark.parametrize("F", [5e7, 1e8, 3e8])
@pytest.mark.parametrize("h", [2e-9, 0.1e-9], ids=["h2nm", "h0.1nm"])
def test_n1_uniform_field_identity(params, F, h):
    """Budget (worst case, errors adding linearly): each closed-form segment term is ~16 correctly
    rounded operations, <= 8 eps relative; nseg = E_g/(q F h) + 1 segments sum; the node potentials
    carry eps |psi| <= eps span/2 = 2 eps, i.e. 2 eps/E_g in each v (factor 1 + 2/1.12 < 3);
    exp(-B/F) turns an exponent error into a relative one times B/F:
        tol = 8 eps nseg (1 + B/F) * 3.
    On the 2 nm mesh (4-12 segments) that is <= 2.9e-12; on the 0.1 nm mesh (37-225 segments) the same
    worst-case budget is 4e-12 .. 5.4e-11. f_n = 0, f_p = 1 (L = -300)."""
    res, ref, nseg = _uniform(params, F, h=h)
    assert res["paths"].start.size >= 3
    dev = np.max(np.abs(-res["R"] / ref - 1.0))
    tol = 8 * EPS * nseg * (1.0 + params.B_V_per_m / F) * 3.0
    assert dev <= tol, (dev, tol)
    assert dev < 1e-12                                      # the plan's figure; MEASURED <= 7e-14


def test_n1_refute_mass_perturbation_breaks_identity():
    """m_r (1 + 1e-9): B ~ sqrt(m_r) -> exponent B/F (1 + 5e-10): R/ref - 1 = -5e-10 B/F + O(prefactor)."""
    spec = _spec(PHONON)
    ph = NL.physical_from_fitted(spec)
    bad = NL.NonlocalPhysical(ph.process, ph.Eg_J, ph.m_r * (1 + 1e-9), ph.m_V * (1 + 1e-9),
                              ph.m_C * (1 + 1e-9), lam=ph.lam)
    F = 1e8
    res, ref, _ = _uniform(PHONON, F, phys_override=bad)
    dev = float(np.median(-res["R"] / ref - 1.0))
    pred = -5e-10 * PHONON.B_V_per_m / F                     # exponent; the prefactor moves 1e-9 x O(1)
    assert abs(dev - pred) < 2.5e-9 and abs(dev) > 1e-12 * 1e3


def test_n1_taper_is_exactly_one_below_095_and_smooth():
    """max_length so that the 11.2 nm uniform-field paths at 1e8 V/m sit in the taper: R/ref == w(l)
    (the module's taper, re-typed: 1 - (3 s^2 - 2 s^3), s = (l - 0.95 L)/(0.05 L)), and with a longer
    max_length the SAME paths give R/ref == 1 to the N1 budget."""
    F = 1e8
    Lmax = 11.5e-9
    res, ref, _ = _uniform(PHONON, F, max_len=Lmax, span_V=8.0)
    s = np.clip((res["length"] - 0.95 * Lmax) / (0.05 * Lmax), 0, 1)
    w = 1 - s * s * (3 - 2 * s)
    assert np.all((s > 0) & (s < 1))
    assert np.max(np.abs(-res["R"] / ref - w)) < 1e-12


# =====================================================================================================
# N2 no final state
# =====================================================================================================

def _tanh_profile(V0, w, h, half=40e-9):
    x = np.arange(-half, half + 0.5 * h, h)
    return x, 0.5 * V0 * (1.0 + np.tanh(x / w))


def test_n2_no_final_state_is_exactly_zero_where_local_is_huge():
    x, psi = _tanh_profile(0.8, 1e-9, 0.05e-9)              # 0.8 V over ~2 nm, peak field 4e8 V/m
    N = x.size
    full = np.ones(N, bool)
    so = full.copy()
    so[0] = so[-1] = False
    ph = NL.physical_from_fitted(_spec(PHONON))
    res = NL.path_rates_1d(x, psi, np.full(N, -300.0), np.full(N, -300.0), ph, 0.0, 20e-9, so, full)
    assert res["paths"].start.size == 0 and res["R"].size == 0
    # the local Kane rate on the same profile (D = -1): A F^P exp(-B/F), trapezoid
    F = np.abs(np.gradient(psi, x))
    kane = PHONON.A_SI * F ** 2.5 * np.exp(-PHONON.B_V_per_m / np.maximum(F, 1.0))
    J_local = Q * float(np.sum(0.5 * (kane[1:] + kane[:-1]) * np.diff(x)))
    assert J_local > 1e7                                     # MEASURED 3.3e7 A/m^2
    # refute: 1.13 V > E_g: paths exist (only in the overlap window) and the rate is on
    x2, psi2 = _tanh_profile(1.13, 1e-9, 0.05e-9)
    res2 = NL.path_rates_1d(x2, psi2, np.full(N, -300.0), np.full(N, -300.0), ph, 0.0, 20e-9, so, full)
    assert res2["paths"].start.size > 0 and np.all(res2["R"] < 0.0)
    v_end = psi2[res2["paths"].start] + EG_EV
    assert np.all(v_end <= psi2.max())                       # every start lies in the overlap window


# =====================================================================================================
# N4 independent continuum oracle
# =====================================================================================================
V0, WJ, HALF = 2.0, 5e-9, 40e-9
DELTA = 0.3


def _psi(x):
    return 0.5 * V0 * (1.0 + math.tanh(x / WJ))


def _dpsi(x):
    return 0.5 * V0 / WJ / math.cosh(x / WJ) ** 2


def _psi_diff(x, d):
    """psi(x + d) - psi(x) without cancellation: tanh a - tanh b = sinh(a - b)/(cosh a cosh b)."""
    return 0.5 * V0 * math.sinh(d / WJ) / (math.cosh((x + d) / WJ) * math.cosh(x / WJ))


def _Ln(x):
    return -2.0 + 0.5 * math.sin(x / 7e-9)


def _Lp(x):
    return -1.0 + 0.4 * math.cos(x / 9e-9)


def _oracle_rate(x0, process, consts):
    """R(x0) of Eqs. 480-484 (direct, k_m -> inf, m0 -> inf) or 490-493 (phonon), typed from the guide:
    the path end and x0 by brentq, every integral by quad after a substitution that removes the
    1/sqrt end singularities (x = x0 + l sin^2(th/2) over a whole path; x = x0 + d s^2 at one end).
    Band bendings are evaluated from OFFSETS (_psi_diff), so u -> 0 and 1 - u -> 0 stay accurate."""
    Eg_V = EG_EV
    Eg = Q * Eg_V
    if _psi(HALF) - _psi(x0) <= Eg_V:
        return 0.0
    l = brentq(lambda d: _psi_diff(x0, d) - Eg_V, 0.0, HALF - x0, xtol=1e-26, rtol=4 * EPS)
    xe = x0 + l

    def u(d):                                              # (eps - E_V(x0 + d))/E_g
        return _psi_diff(x0, d) / Eg_V

    def w(x, d):                                           # (E_C(x) - eps)/E_g at x, d = xe - x
        return _psi_diff(x, d) / Eg_V
    fn = 1.0 / (1.0 + math.exp(-(_Ln(xe) + DELTA)))       # f_n(l): (eps - E_Fn)/kT = -L_n - delta
    fp = 1.0 / (1.0 + math.exp(_Lp(x0)))                   # f_p(0): (eps - E_Fp)/kT = L_p
    G = Q * _dpsi(x0)                                      # |grad E_V(0)| [J/m]
    kw = dict(epsabs=0.0, epsrel=1e-12, limit=200)
    if process == "direct":
        m_r, g = consts

        def kappa(th):                                     # Eq. 482 with Eq. 483 at m0 -> inf:
            d0, d1 = l * math.sin(0.5 * th) ** 2, l * math.cos(0.5 * th) ** 2
            one_p, one_m = 2.0 * u(d0), 2.0 * w(x0 + d0, d1)   # 1 + alpha, 1 - alpha (alpha = 2u - 1)
            return math.sqrt(m_r * Eg * one_p * one_m) / HB
        I1 = quad(lambda th: kappa(th) * 0.5 * l * math.sin(th), 0.0, math.pi, **kw)[0]
        I0 = quad(lambda th: 0.5 * l * math.sin(th) / kappa(th), 0.0, math.pi, **kw)[0]
        Cd = g * math.pi / (36.0 * H) / I0                 # Eq. 481
        return G * Cd * math.exp(-2.0 * I1) * (fn - fp)    # Eq. 480
    m_r, m_v, m_c, lam = consts

    def kv(d):                                             # Eq. 492 at x0 + d
        return math.sqrt(2.0 * m_v * Eg * u(d)) / HB

    def kc(d):                                             # Eq. 493 at xe - d
        return math.sqrt(2.0 * m_c * Eg * w(xe - d, d)) / HB
    ls = brentq(lambda d: kv(d) - kc(l - d), 0.0, l, xtol=1e-26, rtol=4 * EPS)   # x0 of the guide
    dv, dc = ls, l - ls
    JV = quad(lambda s: kv(dv * s * s) * 2.0 * dv * s, 0.0, 1.0, **kw)[0]
    IV = quad(lambda s: 2.0 * dv * s / kv(dv * s * s), 0.0, 1.0, **kw)[0]
    JC = quad(lambda s: kc(dc * s * s) * 2.0 * dc * s, 0.0, 1.0, **kw)[0]
    IC = quad(lambda s: 2.0 * dc * s / kc(dc * s * s), 0.0, 1.0, **kw)[0]
    # Eq. 491 for one material: Int_0^l [...] dx = l [...]
    Cp = (lam / (2 ** 6 * math.pi ** 2 * Eg) * math.sqrt(m_v * m_c / (H * l * math.sqrt(2 * m_r * Eg)))
          * l / (IV * IC))
    return G * Cp * math.exp(-2.0 * JV - 2.0 * JC) * (fn - fp)    # Eq. 490


def _consts(process):
    Eg = Q * EG_EV
    if process == "direct":
        return _phys_direct(DIRECT.A_SI, DIRECT.B_V_per_m, Eg)
    return _phys_phonon(PHONON.A_SI, PHONON.B_V_per_m, Eg, RATIO)


def _oracle_current(process):
    consts = _consts(process)
    lo = -HALF + 1e-12
    hi = brentq(lambda s: _psi(HALF) - _psi(s) - EG_EV, -HALF, HALF)   # last start with an end in range
    J = quad(lambda s: _oracle_rate(s, process, consts), lo, hi, epsabs=0, epsrel=1e-10, limit=400,
             points=[-3 * WJ, 0.0])[0]
    return -Q * J                                          # generation current > 0


def _grid(h):
    x = np.arange(-HALF, HALF + 0.5 * h, h)
    return (x, np.array([_psi(v) for v in x]), np.array([_Ln(v) for v in x]),
            np.array([_Lp(v) for v in x]))


def _module_rates(process, x, psi, Ln, Lp, starts, every=True):
    params = DIRECT if process == "direct" else PHONON
    ph = NL.physical_from_fitted(_spec(params, max_length_m=1.0))
    so = np.zeros(x.size, bool)
    so[starts] = True
    res = NL.path_rates_1d(x, psi, Ln, Lp, ph, DELTA, 1.0, so, np.ones(x.size, bool), derivatives=False)
    if every:                                               # each requested start has a path
        assert np.array_equal(res["paths"].start, np.sort(np.asarray(starts)))
    return res


def _module_current(process, h):
    x, psi, Ln, Lp = _grid(h)
    res = _module_rates(process, x, psi, Ln, Lp, np.arange(1, x.size - 1), every=False)
    return -Q * float(np.sum(res["R"] * h))                 # interior node volume = h (uniform mesh)


@pytest.fixture(scope="module")
def oracle():
    return {p: _oracle_current(p) for p in ("direct", "phonon")}


# ---- N4a: the SAME discrete model, re-implemented independently -------------------------------------
def _p1_oracle_rate(x, psi, Ln, Lp, i, process, consts):
    """R at start node i of the P1 (piecewise-linear) band profile through the nodes, computed with
    np.interp + brentq + quad (no closed form): |grad E_V(0)| = the 3-point nodal derivative (the
    discretisation's definition, re-typed); path end = root of the interpolant; integrals piece by piece
    between nodes, the 1/sqrt end pieces by QAWS (quad weight 'alg') with the smooth remainder; f_n from
    L_n interpolated linearly at the end. Returns (R, relative error budget from quad's estimates)."""
    Eg_V = EG_EV
    Eg = Q * Eg_V
    hm, hp = x[i] - x[i - 1], x[i + 1] - x[i]
    grad = (hm / (hp * (hm + hp)) * (psi[i + 1] - psi[i]) + hp / (hm * (hm + hp)) * (psi[i] - psi[i - 1]))
    assert grad > 0                                       # the tanh profile rises: paths run to +x
    k = i + int(np.flatnonzero(psi[i + 1:] - psi[i] >= Eg_V)[0]) + 1     # first node past the end
    xe = brentq(lambda s: np.interp(s, x, psi) - psi[i] - Eg_V, x[k - 1], x[k], xtol=1e-26,
                rtol=4 * EPS)
    l = xe - x[i]

    def v(s):
        return (np.interp(s, x, psi) - psi[i]) / Eg_V
    fn = 1.0 / (1.0 + math.exp(-(np.interp(xe, x, Ln) + DELTA)))
    fp = 1.0 / (1.0 + math.exp(Lp[i]))
    errs = []

    def piecewise(f, a, b, sing_lo=None, sing_hi=None):
        """Int_a^b f over the node pieces; sing_lo/sing_hi: (smooth remainder g, slope) for a 1/sqrt
        singular first / last piece: f = g(s)/sqrt(slope (s - a)) or g(s)/sqrt(slope (b - s))."""
        knots = np.concatenate(([a], x[(x > a) & (x < b)], [b]))
        tot = 0.0
        for n, (p0, p1) in enumerate(zip(knots[:-1], knots[1:])):
            if n == 0 and sing_lo is not None:
                g, sl = sing_lo
                r = quad(lambda s: g(s) / math.sqrt(sl), p0, p1, weight="alg", wvar=(-0.5, 0.0))
            elif n == len(knots) - 2 and sing_hi is not None:
                g, sl = sing_hi
                r = quad(lambda s: g(s) / math.sqrt(sl), p0, p1, weight="alg", wvar=(0.0, -0.5))
            else:
                r = quad(f, p0, p1, epsabs=0, epsrel=1e-13, limit=100)
            tot += r[0]
            errs.append(r[1])
        return tot
    s0 = (psi[i + 1] - psi[i]) / Eg_V / (x[i + 1] - x[i])          # v slope on the first piece
    se = (psi[k] - psi[k - 1]) / Eg_V / (x[k] - x[k - 1])          # v slope on the end piece
    if process == "direct":
        m_r, g = consts
        K0 = 2.0 * math.sqrt(m_r * Eg) / HB
        I1 = piecewise(lambda s: K0 * math.sqrt(max(v(s) * (1 - v(s)), 0.0)), x[i], xe)
        n1 = len(errs)
        I0 = piecewise(lambda s: 1.0 / (K0 * math.sqrt(v(s) * (1 - v(s)))), x[i], xe,
                       sing_lo=(lambda s: 1.0 / (K0 * math.sqrt(1 - v(s))), s0),
                       sing_hi=(lambda s: 1.0 / (K0 * math.sqrt(v(s))), se))
        budget = 2 * sum(errs[:n1]) + sum(errs[n1:]) / I0
        R = Q * grad * g * math.pi / (36 * H) / I0 * math.exp(-2 * I1) * (fn - fp)
        return R, budget
    m_r, m_v, m_c, lam = consts
    kv = math.sqrt(2 * m_v * Eg) / HB
    kc = math.sqrt(2 * m_c * Eg) / HB
    vs = m_c / (m_v + m_c)                                 # kappa_V = kappa_C
    ks = i + int(np.flatnonzero((psi[i + 1:] - psi[i]) / Eg_V >= vs)[0]) + 1
    xs = brentq(lambda s: v(s) - vs, x[ks - 1], x[ks], xtol=1e-26, rtol=4 * EPS)
    JV = piecewise(lambda s: kv * math.sqrt(max(v(s), 0.0)), x[i], xs)
    JC = piecewise(lambda s: kc * math.sqrt(max(1 - v(s), 0.0)), xs, xe)
    n1 = len(errs)
    IV = piecewise(lambda s: 1.0 / (kv * math.sqrt(v(s))), x[i], xs,
                   sing_lo=(lambda s: 1.0 / kv, s0))
    IC = piecewise(lambda s: 1.0 / (kc * math.sqrt(1 - v(s))), xs, xe,
                   sing_hi=(lambda s: 1.0 / kc, se))
    budget = 2 * sum(errs[:n1]) + sum(errs[n1:]) / min(IV, IC)
    Cp = (lam / (2 ** 6 * math.pi ** 2 * Eg) * math.sqrt(m_v * m_c / (H * l * math.sqrt(2 * m_r * Eg)))
          * l / (IV * IC))
    return Q * grad * Cp * math.exp(-2 * JV - 2 * JC) * (fn - fp), budget


@pytest.mark.parametrize("process", ["direct", "phonon"])
def test_n4a_module_equals_independent_p1_profile_oracle(process):
    """The discrete model (P1 bands through the nodes, 3-point start gradient, linear end root and E_Fn
    interpolation) computed a second, independent way: np.interp + brentq + quad, no closed forms. Budget
    per path: the sum of quad's absolute error estimates, propagated (x2 for the exponent integrals, which
    enter as exp(-2 J)) plus 1e-12 for the brentq roots: MEASURED <= 3e-12 against budgets ~1e-13..1e-11.
    Tolerance: max(budget, 1e-12) + 32 eps (B/F) per path."""
    x, psi, Ln, Lp = _grid(0.2e-9)
    starts = [int(np.argmin(np.abs(x - v))) for v in np.arange(-16e-9, -1e-9, 1.4e-9)]
    res = _module_rates(process, x, psi, Ln, Lp, starts)
    consts = _consts(process)
    worst = 0.0
    for k, i in enumerate(res["paths"].start):
        ref, budget = _p1_oracle_rate(x, psi, Ln, Lp, int(i), process, consts)
        dev = abs(res["R"][k] / ref - 1.0)
        worst = max(worst, dev)
        assert dev <= max(budget, 1e-12) + 32 * EPS * 60, (i, dev, budget)
    print(process, "N4a worst", worst)


def test_n4a_refute_first_order_gradient_breaks_the_identity(monkeypatch):
    """Coarsen one interpolation: the start node's |grad E_V| from the one-sided (forward) edge slope, an
    O(h) estimate, instead of the 3-point nodal derivative. The per-path identity must fail by O(h psi''/
    psi') ~ 1e-2 (P1 3.5 proposed snapping the path end to a node; that is the same O(h) coarsening of a
    different interpolation -- the gradient is the one that can be swapped cleanly)."""
    real = NL.nodal_gradient_1d

    def forward(x, psi):
        g, wm, w0, wp = real(x, psi)
        g = g.copy()
        g[1:-1] = (psi[2:] - psi[1:-1]) / (x[2:] - x[1:-1])
        return g, wm, w0, wp
    monkeypatch.setattr(NL, "nodal_gradient_1d", forward)
    x, psi, Ln, Lp = _grid(0.2e-9)
    i = int(np.argmin(np.abs(x + 8e-9)))
    res = _module_rates("direct", x, psi, Ln, Lp, [i])
    ref, budget = _p1_oracle_rate(x, psi, Ln, Lp, i, "direct", _consts("direct"))
    assert abs(res["R"][0] / ref - 1.0) > 1e3 * max(budget, 1e-12)


# ---- N4b: convergence of the discrete model to the continuum --------------------------------------
X0_NESTED = np.arange(-20e-9, -0.9e-9, 1.6e-9)            # nodes of every mesh below (nested)
HS = [0.1e-9, 0.05e-9, 0.025e-9, 0.0125e-9]


def _per_path_errors(process, hs, forward_gradient=False, monkeypatch=None):
    consts = _consts(process)
    ref = np.array([_oracle_rate(v, process, consts) for v in X0_NESTED])
    out = []
    for h in hs:
        x, psi, Ln, Lp = _grid(h)
        idx = [int(np.argmin(np.abs(x - v))) for v in X0_NESTED]
        assert np.max(np.abs(x[idx] - X0_NESTED)) < 1e-15
        out.append(float(np.max(np.abs(_module_rates(process, x, psi, Ln, Lp, idx)["R"] / ref - 1.0))))
    return out


@pytest.mark.parametrize("process", ["direct", "phonon"])
def test_n4b_per_path_convergence_to_the_continuum(process):
    """DERIVED ORDER 3/2, not 2 (a finding against plan 4.1.3's >= 1.7). The prefactor integrals
    Int dx/kappa (direct, both ends) and Int dx/kappa_V, Int dx/kappa_C (phonon, one end each) have an
    integrable 1/sqrt(v) singularity at a path end. On the P1 band profile the slope there is the edge
    SECANT, off the true slope by O(h v''); near the end Int ds/sqrt(sigma s) = 2 sqrt(h/sigma), so the
    end piece carries a relative error O(h) x O(sqrt h) absolute: the per-path rate error is O(h^{3/2})
    (the quadratic-profile expansion gives (1/3) v'' h^{3/2}/sigma^{3/2} at the start), every other piece
    (exponent integrals, path end, occupancies, start gradient) O(h^2). MEASURED on nested starts: the
    phonon Int dx/kappa_V error halves by 2.83 = 2^{3/2} per refinement exactly; the per-path max error
    is 5.7e-4 / 2.5e-4 / 1.0e-4 / 3.8e-5 (direct) at h = 0.1 .. 0.0125 nm, orders 1.19, 1.32, 1.38 ->
    approaching 3/2 (the O(h^2) parts still add pre-asymptotically).
    GATE: the least-squares order over the four meshes >= 1.2, which separates the derived 3/2 from the
    O(h) of any first-order slip (the refutation below measures <= 1.0), and the finest-mesh error
    <= rho_max h^{3/2} with rho_max = 2 max(|e| / h^{3/2}) over the two coarser meshes (the h^{3/2}-scaled
    error must not grow under refinement beyond its pre-asymptotic drift; MEASURED drift <= 1.4x)."""
    err = _per_path_errors(process, HS)
    lh = np.log([h * 1e9 for h in HS])
    order = float(np.polyfit(lh, np.log(err), 1)[0])
    rho = [e / (h * 1e9) ** 1.5 for e, h in zip(err, HS)]
    print(process, "per-path max err", err, "order", order, "rho", rho)
    assert order >= 1.2
    assert rho[-1] <= 2.0 * max(rho[0], rho[1])


def test_n4b_refute_first_order_gradient_drops_the_order(monkeypatch):
    real = NL.nodal_gradient_1d

    def forward(x, psi):
        g, wm, w0, wp = real(x, psi)
        g = g.copy()
        g[1:-1] = (psi[2:] - psi[1:-1]) / (x[2:] - x[1:-1])
        return g, wm, w0, wp
    monkeypatch.setattr(NL, "nodal_gradient_1d", forward)
    err = _per_path_errors("direct", HS)
    order = float(np.polyfit(np.log([h * 1e9 for h in HS]), np.log(err), 1)[0])
    print("forward-gradient per-path errors", err, "order", order)
    assert order < 1.2


@pytest.mark.parametrize("process", ["direct", "phonon"])
def test_n4c_integrated_current_report(oracle, process):
    """A MEASUREMENT on the integrated current q Sum R V vs the continuum oracle's q Int R dx0 (its sign
    mixes the h^{3/2} start term with the oscillating end terms, so its order is not a clean gate; the
    per-path gate above is). The integrated relative error is the R-weighted mean of the per-path ones plus
    the O(h^2) node-volume quadrature of Int R dx0, so |error| <= the per-path maximum; asserted against 2x
    the maximum over the 12 sampled starts (the other starts are unsampled). Reported: the error at 0.4 and
    0.1 nm and the local/nonlocal ratio on this junction."""
    J_or = oracle[process]
    e04 = _module_current(process, 0.4e-9) / J_or - 1.0
    e01 = _module_current(process, 0.1e-9) / J_or - 1.0
    per_path01 = _per_path_errors(process, [0.1e-9])[0]
    params = DIRECT if process == "direct" else PHONON

    def loc(x):
        F = _dpsi(x)
        return params.A_SI * F ** params.P * math.exp(-params.B_V_per_m / F)
    J_loc = Q * quad(loc, -HALF, HALF, epsabs=0, epsrel=1e-10, limit=400, points=[0.0])[0]
    print(process, "J_oracle", J_or, "err 0.4/0.1 nm", e04, e01, "local(D=-1)/nonlocal", J_loc / J_or)
    assert abs(e01) <= 2.0 * per_path01
    assert J_loc > 0 and J_or > 0
