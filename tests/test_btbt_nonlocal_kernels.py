"""v0.15 plan item 4.1.3: nonlocal band-to-band tunnelling, the 1-D KERNELS (carriers/btbt_nonlocal.py) --
no DEVSIM. Every reference is re-derived HERE from the Sentaurus Device User Guide N-2017.09 text
(Eqs. 480-498, typed from the page images) or computed with scipy quadrature / root finding written here;
nothing below calls the module's conversion or segment code to build a reference.

Gates (OLD CODE: dynameta.carriers.btbt_nonlocal does not exist on e7c469d -- every test fails at import,
MEASURED by running this file against an export of e7c469d):
  N0a  closed-form segment integrals (Kane two-band and Keldysh) == scipy.quad, including segments that touch
       v = 0 and v = 1 (the integrable 1/sqrt end singularities), within quad's own error estimate + 64 eps.
  N0b  their complex-step derivatives == Richardson central differences (budget derived in the test).
  N0c  (A, B) -> physical -> (A, B) round trip 1e-14; the physical set == Eqs. 495-498 inverted HERE.
  N1   UNIFORM FIELD: R / (A F^P exp(-B/F)) - 1 within the round-off budget (<= 1e-12) for the direct
       (P = 2) and phonon (P = 2.5) processes at 5e7, 1e8, 3e8 V/m; inside the taper R/ref == w(l).
       Refute: m_r perturbed by 1e-9 must fail it by the predicted B/F * 5e-10.
  N2   NO FINAL STATE: a 0.8 V tanh step (peak 4e8 V/m) gives R == 0.0 exactly (no path) while the local
       Kane rate integrates to > 1e7 A/m^2; a 1.13 V step switches the nonlocal rate on.
  N4a  the DISCRETE model computed a second, independent way (np.interp + brentq + quad, no closed forms)
       == the module per path, within quad's propagated error estimates (MEASURED ~1e-14).
  N4b  INDEPENDENT CONTINUUM ORACLE on a graded (tanh) junction with smooth quasi-Fermi profiles: R(x0)
       from a scipy brentq/quad implementation of Eqs. 480-484 (direct) and 490-493 (phonon) typed here;
       per-path error at nested start nodes converges at order >= 1.7 (MEASURED 2.0) and the 0.1 nm error
       <= 2 C_inf h^2 (C_inf from the finest pair, the L1 method). Refutations: a first-order start gradient
       (order 1.0) and the plain P1 prefactor integrals (order 1.3: the 1/sqrt end singularity, module doc
       section 3) both fail it.
  N4c  the integrated current q Sum R V converges to q Int R dx0 at order >= 1.7; the local/nonlocal ratio
       on this junction is reported.
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


def test_n0a_prefactor_moments_match_quadrature():
    """kane_moments / keldysh_moments (the slope-interpolated prefactor integrals, module doc section 3)
    against quad in v, the 1/sqrt end singularities by quad's 'alg' weight (QAWS) on the piece that
    touches them. Tolerance: quad's error estimate + 64 eps |ref| (+ 4 eps p1/dp |ref| for M1: the
    reference rounds v - p0 at v ~ p1). MEASURED worst 5.6e-8 relative: M1 of the 1e-9-wide piece at 0.3,
    inside that piece's reference-rounding allowance 2.7e-7; <= 4e-14 on every piece wider than 1e-3."""
    rng = np.random.default_rng(5)
    cases = [tuple(sorted(rng.uniform(0.0, 1.0, 2))) for _ in range(40)]
    cases += [(0.0, 0.37), (0.0, 1.0), (0.55, 1.0), (0.3, 0.3 + 1e-9), (0.999, 1.0), (0.0, 1e-4), (0.2, 0.9)]
    worst = 0.0
    for p0, p1 in cases:
        dp = p1 - p0
        kq = dict(epsabs=0, epsrel=1e-13, limit=200)

        def wq(f, kind):
            """Int_p0^p1 f(v) K(v) dv, K's singular factors as QAWS weights where they touch the piece."""
            a_ = -0.5 if (kind in ("kane", "V") and p0 == 0.0) else 0.0
            b_ = -0.5 if (kind in ("kane", "C") and p1 == 1.0) else 0.0

            def smooth(v):
                k = 1.0
                if kind in ("kane", "V") and a_ == 0.0:
                    k /= math.sqrt(v)
                if kind in ("kane", "C") and b_ == 0.0:
                    k /= math.sqrt(1.0 - v)
                return f(v) * k
            if a_ == 0.0 and b_ == 0.0:
                return quad(smooth, p0, p1, **kq)
            return quad(smooth, p0, p1, weight="alg", wvar=(a_, b_), **kq)
        for kind, got in (("kane", NL.kane_moments(p0, p1)), ("V", NL.keldysh_moments(p0, p1, True)),
                          ("C", NL.keldysh_moments(1 - p0, 1 - p1, False, dp=p1 - p0))):
            r0 = wq(lambda v: 1.0, kind)
            r1 = wq(lambda v: (v - p0) / dp, kind)
            for g_, (ref, err) in zip(got, (r0, r1)):
                dev = abs(float(g_) - ref)
                # + the reference's own rounding of (v - p0) at v ~ p1: eps p1/dp relative (M1 only)
                tol = err + 64 * EPS * abs(ref) + (4 * EPS * p1 / dp * abs(ref) if g_ is got[1] else 0.0)
                assert dev <= tol, (kind, p0, p1, g_, ref, err)
                worst = max(worst, dev / abs(ref))
    print("N0a moments worst", worst)


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
        for kern in (NL.kane_segment, NL.keldysh_segment, lambda a, b, L: NL.kane_moments(a, b),
                     lambda a, b, L: NL.keldysh_moments(a, b, True),
                     lambda a, b, L: NL.keldysh_moments(1 - a, 1 - b, False, dp=b - a)):
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


def test_n0c_one_kane_kernel(monkeypatch):
    """ONE implementation of Eqs. 495-496: fitted_from_physical's direct branch IS tunneling.kane_direct_params
    (breaking that function breaks it), and physical_from_fitted is its exact inverse: kane_direct_params(E_g,
    m_r, g) -> physical_from_fitted -> (m_r, g) to 1e-14 over random direct sets."""
    from dynameta.carriers import tunneling as TN
    rng = np.random.default_rng(13)
    M0 = 9.1093837015e-31
    for _ in range(25):
        eg, mr, g = rng.uniform(0.3, 3.5), rng.uniform(0.01, 0.5), rng.uniform(0.5, 4.0)
        kp = TN.kane_direct_params(eg, mr, g, source="test")
        ph = NL.physical_from_fitted(NL.NonlocalBTBT(kp, Eg_tun_eV=eg))
        assert abs(ph.m_r / (mr * M0) - 1) < 1e-14 and abs(ph.g / g - 1) < 1e-14
        A, B = NL.fitted_from_physical(ph)
        assert abs(A / kp.A_SI - 1) < 1e-14 and abs(B / kp.B_V_per_m - 1) < 1e-14

    def broken(*a, **k):
        raise RuntimeError("the one Kane kernel was called")
    monkeypatch.setattr(NL, "kane_direct_params", broken)
    with pytest.raises(RuntimeError, match="one Kane kernel"):
        NL.fitted_from_physical(NL.physical_from_fitted(_spec(DIRECT)))


def _material(eg, law=None):
    from dynameta.materials.band_temperature import BandTemperature
    from dynameta.materials.transport_model import TransportModel
    bt = None if law is None else BandTemperature(gap_law=law, affinity_fraction=0.5)
    return TransportModel(n_bg_m3=1e24, eps_static=11.7, dos_mass_kg_of_n_m3=lambda n: 1e-31, band_gap_eV=eg,
                          band_temperature=bt)


SI_LAW = (1.17, 4.73e-4, 636.0)          # Si Varshni triple (plan 5.3; P3's absolute Si constants) -- a test law


def test_gap_from_the_package_gap_kernel():
    """material= takes the gap from materials.band_temperature.band_data_at_T: at the set's quoted temperature
    it is band_gap_eV, and the physical set equals the explicit-Eg_tun_eV one bit for bit."""
    from dynameta.materials.band_temperature import band_data_at_T
    for params in (DIRECT, PHONON):
        a = NL.physical_from_fitted(_spec(params))
        b = NL.physical_from_fitted(NL.NonlocalBTBT(params, material=_material(EG_EV), mass_ratio_V_over_C=RATIO))
        assert a == b
        tm = _material(EG_EV, SI_LAW)
        ph = NL.physical_from_fitted(NL.NonlocalBTBT(params, material=tm, mass_ratio_V_over_C=RATIO), T_K=350.0)
        assert ph.Eg_set_J == Q * band_data_at_T(tm, 300.0).E_g_eV
        assert ph.Eg_J == Q * band_data_at_T(tm, 350.0).E_g_eV
    with pytest.raises(ValueError, match="exactly one"):
        NL.NonlocalBTBT(PHONON)
    with pytest.raises(ValueError, match="exactly one"):
        NL.NonlocalBTBT(PHONON, Eg_tun_eV=1.12, material=_material(1.12))


@pytest.mark.parametrize("params", [DIRECT, PHONON], ids=["direct", "phonon"])
def test_n1_uniform_field_at_another_temperature(params):
    """A set quoted at 300 K on a 350 K device whose material has a band-temperature model: the physical set
    (m_r, g or lam) is FIXED by the 300 K gap and the path sees E_g(350 K), so in a uniform field the model
    reproduces A' F^P exp(-B'/F) with, from Eqs. 495-498 at fixed masses (re-derived here):
      B' = B r^(3/2),  A' = A r^(-1/2) (direct, Eq. 495) or A r^(-7/4) (phonon, Eq. 497),  r = E_g(T)/E_g(T0).
    B' is the local model's Eq. 478 scaling (tunneling.btbt_at_temperature with the same band_gap_ratio) to
    round-off; A' is NOT (Eq. 478 keeps A fixed): MEASURED A'/A = r^(-1/2) = 1.006 / r^(-7/4) = 1.021 at
    350 K for this law -- the difference between the two temperature models, by construction. Budget: the
    N1 budget (8 eps nseg (1 + B/F) x 3) + 4 eps for the power-law references."""
    from dataclasses import replace
    from dynameta.carriers.tunneling import btbt_at_temperature
    from dynameta.materials.band_temperature import band_gap_ratio
    tm = _material(EG_EV, SI_LAW)
    r = band_gap_ratio(tm, 350.0, 300.0)
    spec = NL.NonlocalBTBT(params, material=tm, mass_ratio_V_over_C=RATIO, max_length_m=1.0)
    ph = NL.physical_from_fitted(spec, T_K=350.0)
    Bp = params.B_V_per_m * r ** 1.5
    Ap = params.A_SI * r ** (-0.5 if params.P == 2.0 else -1.75)
    assert abs(btbt_at_temperature(replace(params, scale_B_with_gap=True), 350.0, r).B_V_per_m / Bp - 1) < 4 * EPS
    A2, B2 = NL.fitted_from_physical(ph, at_path_gap=True)
    assert abs(A2 / Ap - 1) < 8 * EPS and abs(B2 / Bp - 1) < 8 * EPS
    F, h = 1e8, 0.1e-9
    N = int(round(4.0 / (F * h))) + 1
    x = np.arange(N) * h
    psi = F * (x - 0.5 * x[-1])
    full = np.ones(N, bool)
    so = full.copy()
    so[0] = so[-1] = False
    res = NL.path_rates_1d(x, psi, np.full(N, -300.0), np.full(N, -300.0), ph, 0.0, 1.0, so, full,
                           derivatives=False)
    nseg = ph.Eg_J / (Q * F * h) + 1
    dev = np.max(np.abs(-res["R"] / (Ap * F ** params.P * math.exp(-Bp / F)) - 1.0))
    print(params.P, "r", r, "A'/A", Ap / params.A_SI, "dev", dev)
    assert dev <= 8 * EPS * nseg * (1.0 + Bp / F) * 3.0 + 4 * EPS * (1 + Bp / F)


def test_spec_validation():
    with pytest.raises(ValueError, match="P = 2"):
        NL.NonlocalBTBT(BTBTParams(1e15, 1e9, 2.2), Eg_tun_eV=1.1)
    with pytest.raises(ValueError, match="Eg_tun_eV"):
        NL.NonlocalBTBT(DIRECT, Eg_tun_eV=0.0)
    with pytest.raises(ValueError, match="exactly one"):
        NL.NonlocalBTBT(DIRECT)
    with pytest.raises(TypeError):
        NL.NonlocalBTBT((1e15, 1e9, 2.0), Eg_tun_eV=1.1)


def test_walker_discards_and_counts():
    """The walker's discard categories (module doc section 3), on crafted 1-D profiles: a band that bends
    back before the end ("reentrant"), an end edge touching a node marked unusable (a contact: "contact"),
    a path longer than max_length ("too_long"); a usable path is kept with the exact linear end root."""
    ph = NL.physical_from_fitted(_spec(PHONON))
    h = 1e-9
    x = np.arange(12) * h
    full = np.ones(12, bool)
    # rises 0.3 V per node from node 2; node 4 dips: v at the path nodes is not increasing -> reentrant
    psi = np.array([0, 0, 0.0, 0.3, 0.2, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])
    so = np.zeros(12, bool)
    so[2] = True
    p = NL.trace_paths_1d(x, psi, EG_EV, 1.0, so, full, v_star=None)
    assert p.stats["reentrant"] == 1 and p.start.size == 0
    # a monotone ramp: the path from node 2 ends between nodes 5 and 6 (1.12 V over 0.3 V steps)
    psi = np.array([0, 0, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])
    p = NL.trace_paths_1d(x, psi, EG_EV, 1.0, so, full)
    assert p.start.tolist() == [2] and (p.a[0], p.b[0]) == (5, 6)
    t = p.seg_L[p.seg_end][0]
    assert abs(t - h) < 1e-24                                    # the end edge's full length is stored
    ok = full.copy()
    ok[6] = False                                               # node 6 is a contact
    p = NL.trace_paths_1d(x, psi, EG_EV, 1.0, so, ok)
    assert p.stats["contact"] == 1 and p.start.size == 0
    p = NL.trace_paths_1d(x, psi, EG_EV, 3.5e-9, so, full)      # the path is 3.73 nm long
    assert p.stats["too_long"] == 1 and p.start.size == 0
    res = NL.path_rates_1d(x, psi, np.full(12, -300.0), np.full(12, -300.0), ph, 0.0, 1.0, so, full)
    assert res["length"][0] == pytest.approx(3 * h + (EG_EV - 0.9) / 0.3 * h, rel=1e-14)


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
    """R at start node i of the DISCRETE model, computed with np.interp + brentq + quad (no closed form):
    the band is the P1 (piecewise-linear) interpolant through the nodes; |grad E_V(0)| and the nodal
    slopes are the 3-point nodal derivative (one-sided at the region ends) -- the discretisation's
    definitions, re-typed; the path end and the phonon x0 are roots of the interpolant; the EXPONENT
    integrals are quad over the P1 band, piece by piece; the PREFACTOR integrals Int dx/kappa are quad
    IN v of r(v) K(v) with r = ds/dv interpolated linearly in v between the path nodes' slopes (module
    doc section 3), the 1/sqrt end singularities by QAWS (quad weight 'alg'); f_n from L_n interpolated
    linearly at the end. Returns (R, relative error budget from quad's error estimates)."""
    Eg_V = EG_EV
    Eg = Q * Eg_V
    n = x.size

    def grad(k):
        if k == 0 or k == n - 1:
            j = 1 if k == 0 else n - 1
            return (psi[j] - psi[j - 1]) / (x[j] - x[j - 1])
        hm, hp = x[k] - x[k - 1], x[k + 1] - x[k]
        return (hm / (hp * (hm + hp)) * (psi[k + 1] - psi[k]) + hp / (hm * (hm + hp)) * (psi[k] - psi[k - 1]))
    g0 = grad(i)
    assert g0 > 0                                         # the tanh profile rises: paths run to +x
    k = i + int(np.flatnonzero(psi[i + 1:] - psi[i] >= Eg_V)[0]) + 1     # first node past the end
    xe = brentq(lambda s_: np.interp(s_, x, psi) - psi[i] - Eg_V, x[k - 1], x[k], xtol=1e-26,
                rtol=4 * EPS)
    l = xe - x[i]
    vn = (psi[i:k + 1] - psi[i]) / Eg_V                   # v at the path nodes
    rn = np.array([Eg_V / grad(m) for m in range(i, k + 1)])   # ds/dv at the path nodes

    def v(s_):
        return (np.interp(s_, x, psi) - psi[i]) / Eg_V

    def r(vv):
        return np.interp(vv, vn, rn)
    fn = 1.0 / (1.0 + math.exp(-(np.interp(xe, x, Ln) + DELTA)))
    fp = 1.0 / (1.0 + math.exp(Lp[i]))
    errs = []

    def in_s(f, a_, b_):
        knots = np.concatenate(([a_], x[(x > a_) & (x < b_)], [b_]))
        tot = 0.0
        for p0, p1 in zip(knots[:-1], knots[1:]):
            r_ = quad(f, p0, p1, epsabs=0, epsrel=1e-13, limit=100)
            tot += r_[0]
            errs.append(abs(r_[1]))
        return tot

    def in_v(K, lo, hi, sing_lo, sing_hi):
        """Int_lo^hi r(v) K(v) dv piece by piece between the node v's; K's 1/sqrt factor at lo (v^-1/2)
        and/or hi ((1 - v)^-1/2) as a QAWS weight on the touching piece (K then omits it)."""
        knots = np.concatenate(([lo], vn[(vn > lo) & (vn < hi)], [hi]))
        tot = 0.0
        for m, (p0, p1) in enumerate(zip(knots[:-1], knots[1:])):
            wl = -0.5 if (sing_lo and m == 0) else 0.0
            wh = -0.5 if (sing_hi and m == len(knots) - 2) else 0.0

            def f(vv, wl=wl, wh=wh):
                k_ = r(vv) * K(vv)
                if sing_lo and wl == 0.0:
                    k_ /= math.sqrt(vv - lo) if lo == 0.0 else 1.0
                if sing_hi and wh == 0.0:
                    k_ /= math.sqrt(1.0 - vv)
                return k_
            if wl or wh:
                r_ = quad(f, p0, p1, weight="alg", wvar=(wl, wh), epsabs=0, epsrel=1e-13, limit=100)
            else:
                r_ = quad(f, p0, p1, epsabs=0, epsrel=1e-13, limit=100)
            tot += r_[0]
            errs.append(abs(r_[1]))
        return tot
    if process == "direct":
        m_r, g = consts
        K0 = 2.0 * math.sqrt(m_r * Eg) / HB
        I1 = in_s(lambda s_: K0 * math.sqrt(max(v(s_) * (1 - v(s_)), 0.0)), x[i], xe)
        n1 = len(errs)
        # K = 1/(K0 sqrt(v (1 - v))); the sqrt(v) at 0 and sqrt(1 - v) at 1 are the QAWS weights
        I0 = in_v(lambda vv: 1.0 / K0, 0.0, 1.0, True, True)
        budget = 2 * sum(errs[:n1]) + sum(errs[n1:]) / I0
        R = Q * g0 * g * math.pi / (36 * H) / I0 * math.exp(-2 * I1) * (fn - fp)
        return R, budget
    m_r, m_v, m_c, lam = consts
    kv = math.sqrt(2 * m_v * Eg) / HB
    kc = math.sqrt(2 * m_c * Eg) / HB
    vs = m_c / (m_v + m_c)                                 # kappa_V = kappa_C
    ks = i + int(np.flatnonzero((psi[i + 1:] - psi[i]) / Eg_V >= vs)[0]) + 1
    xs = brentq(lambda s_: v(s_) - vs, x[ks - 1], x[ks], xtol=1e-26, rtol=4 * EPS)
    JV = in_s(lambda s_: kv * math.sqrt(max(v(s_), 0.0)), x[i], xs)
    JC = in_s(lambda s_: kc * math.sqrt(max(1 - v(s_), 0.0)), xs, xe)
    n1 = len(errs)
    IV = in_v(lambda vv: 1.0 / kv, 0.0, vs, True, False)
    IC = in_v(lambda vv: 1.0 / kc, vs, 1.0, False, True)
    budget = 2 * sum(errs[:n1]) + sum(errs[n1:]) / min(IV, IC)
    Cp = (lam / (2 ** 6 * math.pi ** 2 * Eg) * math.sqrt(m_v * m_c / (H * l * math.sqrt(2 * m_r * Eg)))
          * l / (IV * IC))
    return Q * g0 * Cp * math.exp(-2 * JV - 2 * JC) * (fn - fp), budget


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
    """Budget: O(h^2) discretisation (P1 exponent integrals, slope-interpolated prefactor integrals,
    second-order nodal gradient, linear end root and E_Fn interpolation): the least-squares order of the
    per-path maximum error over the nested starts (meshes 0.1 .. 0.0125 nm) >= 1.7, and the 0.1 nm error
    <= 2 C_inf h^2 with C_inf = error/h^2 of the finest mesh (the L1-addendum method). MEASURED: ratios
    3.9-4.1 per halving (order 2.0) for both processes.
    FINDING (module doc section 3): with the prefactor integrals on the P1 band too -- the plan's first
    design -- the order is 3/2 (MEASURED 1.30 / 1.37 over the same meshes): Int dx/kappa has an integrable
    1/sqrt singularity at a path end, where the P1 slope is the edge SECANT, off by O(h v''), so the end
    piece 2 sqrt(h/sigma) carries an O(h^{3/2}) error; the same secant jump makes the rate a sqrt-kinked
    function of psi whenever a path end crosses a node (MEASURED: a DEVSIM dI/dV off by 2.5e-4 between
    +-1 mV and +-0.1 mV differences). The refutation below re-creates that design."""
    err = _per_path_errors(process, HS)
    lh = np.log([h * 1e9 for h in HS])
    order = float(np.polyfit(lh, np.log(err), 1)[0])
    C_inf = err[-1] / (HS[-1] * 1e9) ** 2
    print(process, "per-path max err", err, "order", order)
    assert order >= 1.7
    assert err[0] <= 2.0 * C_inf * (HS[0] * 1e9) ** 2


def _p1_prefactor(monkeypatch):
    """Re-create the P1 prefactor: ds/dv = the edge secant on every segment (r_a = r_b = L/(vb - va))."""
    real = NL._segment_accumulators

    def p1(phys, va, vb, ra, rb, L, end, phase):
        sec = L / (vb - va)
        return real(phys, va, vb, sec, sec, L, end, phase)
    monkeypatch.setattr(NL, "_segment_accumulators", p1)


def _forward_gradient(monkeypatch):
    real = NL.nodal_gradient_1d

    def forward(x, psi):
        g, wm, w0, wp = real(x, psi)
        g = g.copy()
        g[1:-1] = (psi[2:] - psi[1:-1]) / (x[2:] - x[1:-1])
        return g, wm, w0, wp
    monkeypatch.setattr(NL, "nodal_gradient_1d", forward)


@pytest.mark.parametrize("variant", ["forward_gradient", "p1_prefactor"])
def test_n4b_refutations_drop_the_order(monkeypatch, variant):
    """Two coarsenings must fail N4b: the start gradient from the one-sided edge slope (O(h): MEASURED
    order 1.0) and the plain P1 prefactor (O(h^{3/2}): MEASURED 1.3). (P1 3.5 proposed snapping the path
    end to a node; both of these are the same kind of O(h^p < 2) coarsening, of interpolations that can
    be swapped cleanly.)"""
    (_forward_gradient if variant == "forward_gradient" else _p1_prefactor)(monkeypatch)
    err = _per_path_errors("direct", HS)
    order = float(np.polyfit(np.log([h * 1e9 for h in HS]), np.log(err), 1)[0])
    print(variant, "per-path errors", err, "order", order)
    assert order < 1.6


@pytest.mark.parametrize("process", ["direct", "phonon"])
def test_n4c_integrated_current_converges(oracle, process):
    """q Sum R V (every interior start, node volume h) against the continuum oracle's q Int R dx0. Budget:
    O(h^2) per path (N4b) plus the O(h^2) node-volume quadrature of the smooth Int R dx0: order between
    0.4/0.2/0.1 nm >= 1.7 and |error(0.1 nm)| <= 2 C_inf h^2, C_inf from the 0.05/0.025 nm pair. Reported:
    the local Kane rate (D = -1) integrated over the same junction, against the nonlocal oracle."""
    J_or = oracle[process]
    hs = [0.4e-9, 0.2e-9, 0.1e-9, 0.05e-9, 0.025e-9]
    err = [_module_current(process, h) / J_or - 1.0 for h in hs]
    orders = [math.log(abs(err[k] / err[k + 1]), 2) for k in range(len(hs) - 1)]
    C_inf = abs(err[-1]) / (hs[-1] * 1e9) ** 2
    params = DIRECT if process == "direct" else PHONON

    def loc(x):
        F = _dpsi(x)
        return params.A_SI * F ** params.P * math.exp(-params.B_V_per_m / F)
    J_loc = Q * quad(loc, -HALF, HALF, epsabs=0, epsrel=1e-10, limit=400, points=[0.0])[0]
    print(process, "J_oracle", J_or, "rel err", err, "orders", orders, "local(D=-1)/nonlocal", J_loc / J_or)
    assert orders[0] >= 1.7 and orders[1] >= 1.7
    assert abs(err[2]) <= 2.0 * C_inf * (hs[2] * 1e9) ** 2


# =====================================================================================================
# N0d f_n - f_p for FD-sized log activities (2026-11-08)
# =====================================================================================================

def test_n0d_occupancy_difference_is_overflow_free_and_accurate():
    """f_n - f_p = sigma(za) - sigma(zb) for |za|, |zb| up to 3000 (FD log activities ln u + Delta(u) reach
    > 1400 on wild Newton iterates: MEASURED on the 2-D corner device's first FD solve, where the product form
    sinh(d)/(2 cosh cosh) overflowed). Oracle: the same identity in mpmath at 50 digits on the SAME float inputs.
    Budget: 2 eps (1 + |za| + |zb|) relative (the function's own conditioning in its arguments: d ln f / dz <= 1,
    so an input rounding of eps |z| moves it by eps |z|, plus a few ulps). MEASURED <= 0.43 eps (1 + |za| + |zb|).
    Exactly 0 at za == zb; below the double range the result underflows. Prove-out: the product form raises
    an overflow RuntimeWarning on the large inputs; on the small ones the two forms are bitwise equal."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    rng = np.random.default_rng(1)
    za = np.concatenate([rng.uniform(-3000, 3000, 1500), rng.uniform(-40, 40, 500), [0.0, 999.0, 1001.0, -1500.0]])
    zb = np.concatenate([rng.uniform(-3000, 3000, 1500), rng.uniform(-40, 40, 500), [0.0, 1.0, -0.5, -1500.0]])
    zb[:200] = za[:200] + rng.uniform(-1e-6, 1e-6, 200)          # near equilibrium at large activities
    f = NL._occupancy_difference(za, zb)
    worst = 0.0
    for a, b, v in zip(za, zb, f):
        A, B = mp.mpf(float(a)), mp.mpf(float(b))
        ref = mp.sinh((A - B) / 2) / (2 * mp.cosh(A / 2) * mp.cosh(B / 2))
        if ref == 0:
            assert v == 0.0
        elif abs(ref) < 1e-300:
            assert abs(v) < 1e-290
        else:
            worst = max(worst, float(abs(mp.mpf(float(v)) - ref) / abs(ref)) / (1 + abs(a) + abs(b)))
    print("N0d worst relative error / (1 + |za| + |zb|):", worst / EPS, "eps")
    assert worst <= 2 * EPS
    small = (np.abs(za) + np.abs(zb)) <= NL.OCC_SCALED_ABOVE
    prod = np.sinh(0.5 * (za[small] - zb[small])) / (2.0 * np.cosh(0.5 * za[small]) * np.cosh(0.5 * zb[small]))
    assert np.array_equal(prod, f[small])
    with pytest.raises(RuntimeWarning, match="overflow"):
        with np.errstate(over="warn"):
            np.sinh(0.5 * (za - zb)) / (2.0 * np.cosh(0.5 * za) * np.cosh(0.5 * zb))
