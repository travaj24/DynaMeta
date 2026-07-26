"""Fast (pure numpy/scipy, no FEM) tests for the complex-omega pole finder (roadmap item 1.1).

Gates:
  1. real-axis evaluator == tmm_reference (R, T) to ~1e-10, 3-layer lossy stack, normal + 40 deg, s/p.
  2. Fabry-Perot slab poles vs the derived closed form (m = 3..6), Re rtol 1e-8, Im rtol 1e-6, Q.
  3. Q(reflectivity) trend: raising the slab index raises Q monotonically, quantitatively.
  4. lossless Q_rad == closed-form Q; loss makes Q_total < Q_rad and 1/Q_abs = 1/Q - 1/Q_rad;
     doubling the loss ~doubles 1/Q_abs.
  4b. q_budget VALIDATES the lossless-pass root by PROXIMITY (finding Q-4): an ENZ p-pol pole
     function fed without the eps-clearing escapes to a far-plane root -- sometimes correctly
     signed -- and must return NaN + a RuntimeWarning, never +inf; the ENZ-cleared form gives a
     finite Q_abs whose gamma_abs equals the Drude gamma to a few %.
  5. tracking the FP pole over L in [0.9, 1.1] um follows the closed form continuously.
  6. Berreman/ENZ pole of a thin Drude film near omega_p (p-pol, 50 deg), finite Q, thin-film trend.
  7. branch robustness: an evanescent-substrate pole converges and is stable under n_grid doubling.

Run: python -m pytest tests/test_resonance.py -q
"""
import math

import numpy as np
import pytest

from dynameta.constants import C_LIGHT
from dynameta.optics.tmm_reference import stack_rta
from dynameta.optics.resonance import (
    drude_eps, layered_smatrix_complex, smatrix_pole_func, k_par_from_angle,
    find_poles, newton_refine, pole_q, track_pole, q_budget, berreman_enz_pole,
)


# ------------------------------------------------------------------------------------------------
# Gate 1: real-axis evaluator reproduces tmm_reference
# ------------------------------------------------------------------------------------------------
def test_real_axis_matches_tmm_reference():
    lam = 1300e-9
    omega = 2.0 * math.pi * C_LIGHT / lam
    ns = [2.0 + 0.05j, 1.4 + 0.0j, 3.0 + 0.2j]         # 3-layer lossy stack (indices)
    ds = [120e-9, 80e-9, 60e-9]
    layers = [(complex(n) ** 2, d) for n, d in zip(ns, ds)]
    n_super, n_sub = 1.0, 1.5
    for theta_deg in (0.0, 40.0):
        for pol in ("s", "p"):
            R_t, T_t, _ = stack_rta(n_super, list(zip(ns, ds)), n_sub, lam,
                                    theta_deg=theta_deg, pol=pol)
            sm = layered_smatrix_complex(omega, layers, theta_rad=math.radians(theta_deg),
                                         pol=pol, n_super=n_super, n_sub=n_sub)
            assert sm.R == pytest.approx(R_t, rel=1e-10, abs=1e-12)
            assert sm.T == pytest.approx(T_t, rel=1e-10, abs=1e-12)
            # |r|^2 == R by construction (self-consistency of the amplitude vs power channels).
            assert abs(sm.r) ** 2 == pytest.approx(sm.R, rel=1e-12, abs=1e-15)


def test_normal_incidence_s_p_degenerate():
    lam = 1550e-9
    omega = 2.0 * math.pi * C_LIGHT / lam
    layers = [(complex(2.5) ** 2, 200e-9)]
    s = layered_smatrix_complex(omega, layers, pol="s", n_super=1.0, n_sub=1.0)
    p = layered_smatrix_complex(omega, layers, pol="p", n_super=1.0, n_sub=1.0)
    assert s.R == pytest.approx(p.R, rel=1e-12)
    assert s.T == pytest.approx(p.T, rel=1e-12)


# ------------------------------------------------------------------------------------------------
# Fabry-Perot closed form (symmetric slab in vacuum, s-pol, normal incidence)
#
# Round-trip pole condition (exp(-i w t), forward ~ exp(+i kz z)):
#   r(w) = r01 (1 - e^{2 i d}) / (1 - r01^2 e^{2 i d}),  d = n w L / c,  r01 = (1 - n)/(1 + n).
# Pole (denominator zero): e^{2 i d} = 1 / r01^2 = 1 / |r12|^2  (r12 = (n-1)/(n+1) = -r01).
#   => 2 i (n w L / c) = 2 ln(1/|r12|) + 2 pi i m
#   => w_m = (c / (n L)) (m pi - i ln(1/|r12|)),  Im(w_m) < 0 (decaying, exp(-i w t)).  QED.
# ------------------------------------------------------------------------------------------------
def _fp_pole(n, L, m):
    r12 = abs((n - 1.0) / (n + 1.0))
    base = C_LIGHT / (n * L)
    return base * (m * math.pi - 1j * math.log(1.0 / r12))


def _fp_Q(n, m):
    r12 = abs((n - 1.0) / (n + 1.0))
    return m * math.pi / (2.0 * math.log(1.0 / r12))


def test_fabry_perot_poles_closed_form():
    n, L = 2.2, 1.0e-6
    func = smatrix_pole_func([(complex(n) ** 2, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)
    base = C_LIGHT / (n * L)
    r12 = abs((n - 1.0) / (n + 1.0))
    im_line = -math.log(1.0 / r12) * base
    # One box bracketing m = 3..6 (all share the same Im line).
    center = complex(4.5 * math.pi * base, im_line)
    span = complex(2.1 * math.pi * base, 1.2 * abs(im_line))
    poles = find_poles(func, center, span, n_grid=60, refine_tol=1e-12)
    for m in (3, 4, 5, 6):
        want = _fp_pole(n, L, m)
        got = min(poles, key=lambda p, w=want: abs(p - w))
        assert got.real == pytest.approx(want.real, rel=1e-8)
        assert got.imag == pytest.approx(want.imag, rel=1e-6)
        assert pole_q(got) == pytest.approx(_fp_Q(n, m), rel=1e-6)
        # M11 really vanishes there.
        assert abs(func(got)) < 1e-6 * abs(func(want + base))


# ------------------------------------------------------------------------------------------------
# Gate 3: Q(reflectivity) trend
# ------------------------------------------------------------------------------------------------
def test_q_increases_with_index_quantitatively():
    L, m = 1.0e-6, 4
    Qs = []
    for n in (2.0, 2.5, 3.0, 3.5):
        func = smatrix_pole_func([(complex(n) ** 2, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)
        w0 = _fp_pole(n, L, m)
        pole = newton_refine(func, w0, tol=1e-12)
        Q = pole_q(pole)
        assert Q == pytest.approx(_fp_Q(n, m), rel=1e-6)      # quantitative vs closed form
        Qs.append(Q)
    assert all(Qs[i] < Qs[i + 1] for i in range(len(Qs) - 1))  # monotonic increase with n (=R)


# ------------------------------------------------------------------------------------------------
# Gate 4: radiative / absorptive Q split
# ------------------------------------------------------------------------------------------------
def test_q_budget_rad_abs_split():
    n, L, m = 2.2, 1.0e-6, 4
    kappa = 0.02                                              # base extinction added to the slab

    def make_func(loss_scale):
        eps = complex(n) ** 2 + 1j * loss_scale * kappa
        return smatrix_pole_func([(eps, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)

    w0 = _fp_pole(n, L, m)
    budget = q_budget(make_func, w0, refine_tol=1e-12, loss_scale=1.0)
    # Lossless pass recovers the closed-form radiative Q.
    assert budget["Q_rad"] == pytest.approx(_fp_Q(n, m), rel=1e-6)
    # Loss lowers the total Q below the radiative Q.
    assert budget["Q_total"] < budget["Q_rad"]
    assert budget["Q_abs"] > 0.0 and math.isfinite(budget["Q_abs"])
    assert budget["inv_Q_abs"] > 0.0

    # Doubling the loss ~doubles 1/Q_abs (absorption is linear in Im(eps) to leading order).
    budget2 = q_budget(make_func, budget["pole_total"], refine_tol=1e-12, loss_scale=2.0)
    assert budget2["inv_Q_abs"] / budget["inv_Q_abs"] == pytest.approx(2.0, rel=0.10)

    # Zero loss => Q_total == Q_rad (self-consistency of the two passes).
    budget0 = q_budget(make_func, w0, refine_tol=1e-12, loss_scale=0.0)
    assert budget0["Q_total"] == pytest.approx(budget0["Q_rad"], rel=1e-6)

    # finding Q-4: the CLEAN (non-ENZ) recipe must be accepted by the new validity gate, and the
    # measured proximity/residual must stay far inside it. Measured: shift_rel 2.089e-3 against a
    # 5-linewidth bound of ~0.80, |D(pole_rad)| 9.70e-18 of the off-pole reference.
    assert budget["pole_rad_ok"] and budget["warning"] == ""
    assert budget["pole_rad_shift_rel"] < 0.01
    assert budget["pole_rad_residual_rel"] < 1e-9


# ------------------------------------------------------------------------------------------------
# Gate 4b (finding Q-4): the lossless pass must be VALIDATED, and by PROXIMITY
# ------------------------------------------------------------------------------------------------
def _enz_budget_factories(eps_inf, wp, gamma, d, k_par):
    """(uncleared, ENZ-cleared) q_budget factories for a p-pol Drude film. The p-pol pole function
    carries a SPURIOUS simple pole at eps_film(omega) = 0; the cleared form D*eps_film is what
    berreman_enz_pole uses and what q_budget's docstring now requires."""
    def raw(ls):
        return smatrix_pole_func([(lambda w: drude_eps(w, eps_inf, wp, ls * gamma), d)],
                                 pol="p", n_super=1.0, n_sub=1.0, k_par_m=k_par)

    def cleared(ls):
        f = raw(ls)
        return lambda w: f(w) * complex(drude_eps(w, eps_inf, wp, ls * gamma))
    return raw, cleared


@pytest.mark.parametrize("eps_inf,wp,gamma,d_nm,theta_deg", [
    (2.0, 2.0e15, 1.0e14, 40.0, 45.0),
    (3.8, 2.5e15, 1.0e14, 40.0, 40.0),
    (4.0, 3.0e15, 5.0e13, 30.0, 60.0),
])
def test_q_budget_rejects_escaped_lossless_root(eps_inf, wp, gamma, d_nm, theta_deg):
    """REGRESSION (finding Q-4). `q_budget` performed NO validity check on the lossless-pass root.
    On the module's own flagship recipe -- a p-polarized pole function over an ENZ Drude film,
    fed WITHOUT the eps-clearing -- the loss_scale = 0 Newton pass escapes to a far-plane root and
    Q_abs was silently reported as +inf (with a negative implied gamma_abs).

    The escaped root is a GENUINE zero of D and is CORRECTLY SIGNED (Re > 0, Im < 0), so the
    obvious sign test is provably insufficient; only a PROXIMITY test catches it. Measured escape
    here: |pole_rad - pole_total| = 9.1 to 12.6 x |pole_total| (the ledger's independent
    reproduction measured a +2256.83% Re shift).

    Contract: NaN Q_rad/Q_abs plus a RuntimeWarning naming the ENZ-clearing recipe -- never +inf.
    """
    d = d_nm * 1e-9
    theta = math.radians(theta_deg)
    omega_p = wp / math.sqrt(eps_inf)
    k_par = k_par_from_angle(1.0, omega_p, theta)
    seed = berreman_enz_pole(eps_inf=eps_inf, wp=wp, gamma=gamma, thickness_m=d,
                             theta_rad=theta)["omega"]
    raw, cleared = _enz_budget_factories(eps_inf, wp, gamma, d, k_par)

    # --- uncleared: the escape must be CAUGHT ---
    with pytest.warns(RuntimeWarning, match="ENZ-CLEARED"):
        bad = q_budget(raw, seed, refine_tol=1e-12, loss_scale=1.0)
    assert bad["pole_rad_ok"] is False
    assert math.isnan(bad["Q_rad"]) and math.isnan(bad["Q_abs"]) and math.isnan(bad["inv_Q_abs"])
    assert not math.isinf(bad["Q_abs"])                        # NEVER +inf (the old signature)
    assert bad["pole_rad_shift_rel"] > 1.0                     # genuinely far away
    assert "ENZ-CLEARED" in bad["warning"]

    # --- ENZ-cleared: accepted, finite, and physically consistent ---
    good = q_budget(cleared, seed, refine_tol=1e-12, loss_scale=1.0)
    assert good["pole_rad_ok"] and good["warning"] == ""
    assert math.isfinite(good["Q_abs"]) and good["Q_abs"] > 0.0
    assert good["Q_rad"] > good["Q_total"]
    # ABSOLUTE-SCALE oracle: for a Drude film the absorptive decay rate IS the Drude gamma.
    # Measured: 9.918e13 / 9.985e13 / 4.889e13 against gamma = 1e14 / 1e14 / 5e13 (1-2%).
    gamma_abs = good["pole_total"].real / good["Q_abs"]
    assert gamma_abs == pytest.approx(gamma, rel=0.05)


def test_q_budget_sign_test_would_be_insufficient():
    """finding Q-4, the load-bearing correction: the escaped lossless root can be CORRECTLY
    SIGNED, so validating `Im < 0, Re > 0` would let it through. Pinned on the recipe where it
    happens (eps_inf = 2.0): pole_rad = 1.675e16 - 9.337e15j -- Re > 0, Im < 0, a genuine zero of
    D, and 12.6 |pole_total| away. (The sign of the escape is basin chaos, not a property of the
    recipe: at eps_inf = 4.0 the same construction escapes to Re < 0, Im > 0.)"""
    eps_inf, wp, gamma, d, theta = 2.0, 2.0e15, 1.0e14, 40e-9, math.radians(45.0)
    omega_p = wp / math.sqrt(eps_inf)
    k_par = k_par_from_angle(1.0, omega_p, theta)
    seed = berreman_enz_pole(eps_inf=eps_inf, wp=wp, gamma=gamma, thickness_m=d,
                             theta_rad=theta)["omega"]
    raw, _cleared = _enz_budget_factories(eps_inf, wp, gamma, d, k_par)
    with pytest.warns(RuntimeWarning):
        bad = q_budget(raw, seed, refine_tol=1e-12, loss_scale=1.0)
    p = bad["pole_rad"]
    assert p.real > 0.0 and p.imag < 0.0        # the sign test PASSES this escaped root
    assert bad["pole_rad_shift_rel"] > 10.0     # only the proximity test catches it
    assert bad["pole_rad_ok"] is False
    # and it really is a zero of D (not a Newton stall): |D| far below an off-pole reference
    D0 = raw(0.0)
    ref = abs(D0(complex(p.real, -0.5 * abs(p.real))))
    assert abs(D0(p)) < 1e-6 * ref


def test_newton_refine_require_convergence_flag():
    """finding Q-4: newton_refine returns the LAST iterate with no residual test -- for a func that
    never vanishes it returns a meaningless far-plane number. The opt-in convergence assertion
    must catch that; the default (off) must keep the legacy return-the-last-iterate behaviour."""
    never_zero = lambda z: 1.0 + 0.0j
    z = newton_refine(never_zero, 1.0e15 + 0.0j, tol=1e-11, maxiter=20)
    assert abs(z) > 1e14                                       # legacy: a meaningless iterate
    with pytest.raises(ValueError, match="no convergence"):
        newton_refine(never_zero, 1.0e15 + 0.0j, tol=1e-11, maxiter=20,
                      require_convergence=True)
    # a real root still converges with the flag on
    func = smatrix_pole_func([(complex(2.2) ** 2, 1.0e-6)], pol="s")
    root = newton_refine(func, _fp_pole(2.2, 1.0e-6, 4), tol=1e-12, require_convergence=True)
    assert abs(func(root)) < 1e-9 * abs(func(complex(root.real, -0.5 * root.real)))


# ------------------------------------------------------------------------------------------------
# Gate 5: parameter tracking
# ------------------------------------------------------------------------------------------------
def test_track_fp_pole_over_thickness():
    n, m = 2.2, 4
    Ls = np.linspace(0.9e-6, 1.1e-6, 21)

    def solver(L):
        return smatrix_pole_func([(complex(n) ** 2, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)

    poles = track_pole(solver, _fp_pole(n, Ls[0], m), Ls, refine_tol=1e-12)
    assert len(poles) == len(Ls)
    for L, got in zip(Ls, poles):
        want = _fp_pole(n, L, m)
        assert abs(got - want) <= 1e-6 * abs(want)           # follows the closed form continuously
    # 1/L scaling: Re(pole) * L is constant across the sweep.
    prod = [p.real * L for p, L in zip(poles, Ls)]
    assert max(prod) - min(prod) < 1e-6 * abs(prod[0])


# ------------------------------------------------------------------------------------------------
# Gate 6: Berreman / ENZ thin-film mode
# ------------------------------------------------------------------------------------------------
def _driven_absorptance_qfit(eps_inf, wp, gamma, d, theta_rad, n_pts=1200):
    """INDEPENDENT oracle for the Berreman/ENZ mode: scan REAL omega, compute the p-pol driven
    absorptance A(omega) = 1 - R - T of the film with the real-axis evaluator (itself pinned
    against tmm_reference in gate 1), and fit the resonance with analysis.lorentzian_fit (the
    driven-spectrum instrument). Returns (x0, Q). No pole/winding machinery is involved."""
    from dynameta.analysis import lorentzian_fit

    omega_p = wp / math.sqrt(eps_inf)
    film = [(lambda w: drude_eps(w, eps_inf, wp, gamma), d)]
    ws = np.linspace(0.85 * omega_p, 1.30 * omega_p, n_pts)
    A = np.empty(n_pts)
    for i, w in enumerate(ws):
        sm = layered_smatrix_complex(w, film, theta_rad=theta_rad, pol="p")
        A[i] = 1.0 - sm.R - sm.T
    fit = lorentzian_fit(ws, A)
    return fit.x0, fit.Q


@pytest.mark.parametrize("eps_inf,wp,gamma,d_nm,theta_deg", [
    (2.0, 2.0e15, 1.0e14, 40.0, 45.0),
    (3.8, 2.5e15, 1.0e13, 40.0, 40.0),
    (3.8, 2.5e15, 1.0e14, 40.0, 40.0),
    (4.0, 3.0e15, 5.0e13, 30.0, 60.0),
])
def test_berreman_enz_pole_eps_inf_gt_one_vs_driven_oracle(eps_inf, wp, gamma, d_nm, theta_deg):
    # REGRESSION (2026-07-19 adversarial verification): for eps_inf > 1 -- i.e. every REAL
    # TCO/ITO ENZ film (eps_inf ~ 3.7-4) -- the pre-fix finder silently returned spurious
    # far-plane zeros (Re/omega_p ~ 1e-9 or ~10, Q ~ 1e-9..1.4) because the genuine Berreman
    # zero sits next to the film's eps = 0 admittance pole of the p-pol pole function (argument
    # principle nets zeros - poles ~ 0) and the old hardcoded Newton seeds fell off to strays.
    # The fixed finder (ENZ-pole-cleared function + grid-minimum-seeded Newton backstop) must
    # agree with the independent driven-absorptance oracle. Measured agreement at the fix:
    # dRe <= 1.1e-3, dQ <= 9.3% (the largest on the broadest line -- the known pole-Q vs
    # driven-Q gap of low-finesse resonances).
    d = d_nm * 1e-9
    theta = math.radians(theta_deg)
    omega_p = wp / math.sqrt(eps_inf)
    res = berreman_enz_pole(eps_inf=eps_inf, wp=wp, gamma=gamma, thickness_m=d, theta_rad=theta)
    pole = res["omega"]
    assert pole.imag < 0.0                                    # decaying (exp(-i w t))
    assert 0.95 < pole.real / omega_p < 1.10                  # near omega_p (pre-fix: 1e-9 / ~10)
    x0, q_driven = _driven_absorptance_qfit(eps_inf, wp, gamma, d, theta)
    assert pole.real == pytest.approx(x0, rel=5e-3)           # centre vs the driven oracle
    assert res["Q"] == pytest.approx(q_driven, rel=0.15)      # Q vs the driven oracle


def test_find_poles_pole_on_subdivision_centre():
    # REGRESSION (2026-07-19 adversarial verification): a search box centred EXACTLY on a pole
    # (the natural user call) put that pole on the quad-tree dividing lines; both adjacent
    # children's windings were corrupted by the ~pi boundary phase step and the pole was
    # SILENTLY DROPPED (a box around FP m=5 returned only m=4 and m=6). Misses persisted for
    # centre offsets up to ~1e-4 of Re. The validated-split fix (children counts must be clean
    # integers summing to the parent count, else the split lines move to an irrational
    # fraction) must return all three poles at every offset.
    n, L = 3.5, 500e-9
    func = smatrix_pole_func([(complex(n) ** 2, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)
    p5 = _fp_pole(n, L, 5)
    span = complex(0.30 * p5.real, 0.6 * abs(p5.imag))
    for off in (0.0, 1e-9, 1e-6, 1e-4):
        centre = complex(p5.real * (1.0 + off), p5.imag)
        poles = find_poles(func, centre, span, n_grid=40)
        for m in (4, 5, 6):
            want = _fp_pole(n, L, m)
            got = min(poles, key=lambda p, w=want: abs(p - w))
            assert abs(got - want) < 1e-6 * abs(want), (
                "pole m={} missed with centre offset {}".format(m, off))


# ------------------------------------------------------------------------------------------------
# AUDIT Q-11: the ROOT-box winding diagnostic must reach the caller
# ------------------------------------------------------------------------------------------------
def test_root_box_untrustworthy_winding_is_reported_not_swallowed():
    """Audit Q-11. ``_winding_densified`` returns its own "is this count believable" signal (the
    residual max single-step |delta arg| after densification), and the quad-tree has always
    checked it on every CHILD box before accepting a split -- but ``find_poles`` DROPPED it on the
    root box, the one the caller asked about. A pole sitting ON the search-box boundary makes the
    raw winding come out 0.000000 with a residual step of ~1.6 rad, and ``count <= 0`` then
    returned an empty list: bit-for-bit the same answer as an honestly empty box, with nothing for
    the caller to test.

    The three outcomes must now be distinguishable, and the POLES must not move."""
    import warnings

    n, L, m = 3.5, 1.0e-6, 5
    func = smatrix_pole_func([(complex(n) ** 2, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)
    p5 = _fp_pole(n, L, m)
    span_re, span_im = 0.06 * p5.real, 1.5 * abs(p5.imag)

    # (a) CENTRED box -- the pole is found, and nothing is reported
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        centred = find_poles(func, complex(p5.real, p5.imag), complex(span_re, span_im),
                             n_grid=48)
    assert len(centred) == 1 and abs(centred[0] - p5) < 1e-6 * abs(p5)
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]

    # (b) the pole EXACTLY on the box's left edge -- empty list, and now a RuntimeWarning that
    #     says the count itself is not believable
    edge_centre = complex(p5.real + span_re, p5.imag)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        on_edge = find_poles(func, edge_centre, complex(span_re, span_im), n_grid=48)
    assert on_edge == []
    hits = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(hits) == 1, "the boundary pole was still swallowed silently"
    msg = str(hits[0].message)
    assert "untrustworthy" in msg and "does NOT mean 'no poles'" in msg

    # (c) an HONESTLY empty box -- empty list, and SILENCE (the diagnostic must discriminate)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        empty = find_poles(func, complex(p5.real * 1.5, p5.imag), complex(0.01 * p5.real, span_im),
                           n_grid=48)
    assert empty == []
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]

    # (d) on_untrusted policies: 'raise' is fatal, 'ignore' restores the old silence, and the
    #     RETURNED POLES ARE IDENTICAL in all three modes (this is a diagnostic, not a search
    #     change).
    with pytest.raises(RuntimeError, match="untrustworthy"):
        find_poles(func, edge_centre, complex(span_re, span_im), n_grid=48, on_untrusted="raise")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ignored = find_poles(func, edge_centre, complex(span_re, span_im), n_grid=48,
                             on_untrusted="ignore")
    assert ignored == [] and not [w for w in rec if issubclass(w.category, RuntimeWarning)]
    for mode in ("warn", "ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            same = find_poles(func, complex(p5.real, p5.imag), complex(span_re, span_im),
                              n_grid=48, on_untrusted=mode)
        assert same == centred                                   # bit-for-bit
    with pytest.raises(ValueError, match="on_untrusted"):
        find_poles(func, complex(p5.real, p5.imag), span_re, on_untrusted="shout")


def test_root_diagnostic_is_not_a_false_alarm_when_the_count_is_corroborated():
    """A contour can legitimately run at maxstep ~ pi (a near-cancellation on the boundary) while
    its winding is an exact integer AND the quad-tree then refines exactly that many roots. The
    two independent signals agree, so nothing is reported -- otherwise the diagnostic fires on
    ordinary multi-pole boxes and stops meaning anything. (Measured on the shipped
    nonlocal_tmm bulk-plasmon gate: maxstep 3.13 rad, winding exactly 2, 2 poles returned.)"""
    import warnings

    n, L = 3.5, 500e-9
    func = smatrix_pole_func([(complex(n) ** 2, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)
    p5 = _fp_pole(n, L, 5)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        poles = find_poles(func, complex(p5.real, p5.imag),
                           complex(0.30 * p5.real, 0.6 * abs(p5.imag)), n_grid=40)
    assert len(poles) == 3                                       # m = 4, 5, 6
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]

    # ... and this is the case the boundary RE-CHECK must not break: maxstep is large, but every
    # zero is well inside, so shrinking / growing the box by a hair leaves the count alone.
    from dynameta.optics.resonance import _boundary_is_clear, _winding_densified
    rect = (p5.real - 0.30 * p5.real, p5.real + 0.30 * p5.real,
            p5.imag - 0.6 * abs(p5.imag), p5.imag + 0.6 * abs(p5.imag))
    w_root, ms_root = _winding_densified(func, rect, 40)
    assert _boundary_is_clear(func, rect, 40, int(round(w_root))), (w_root, ms_root)


def _polyfunc(roots):
    """An analytic function with EXACTLY the given zeros -- find_poles takes any analytic callable,
    and a polynomial is the only way to place zeros to the bit."""
    rs = [complex(r) for r in roots]

    def f(z):
        out = 1.0 + 0j
        for r in rs:
            out *= (complex(z) - r)
        return out
    return f


@pytest.mark.parametrize("name,roots", [
    ("1 on the RIGHT edge", [1.0 + 0.0j]),
    ("1 on the TOP edge", [0.3 + 1.0j]),
    ("1 inside + 1 on the LEFT edge", [0.2 + 0.1j, -1.0 + 0.0j]),
    ("2 inside + 1 on the LEFT edge", [0.2 + 0.1j, -0.3 - 0.4j, -1.0 + 0.0j]),
    ("2 inside + 1 on the TOP edge", [0.2 + 0.1j, -0.3 - 0.4j, 0.15 + 1.0j]),
    ("3 inside + 1 on the RIGHT edge", [0.2 + 0.1j, -0.3 - 0.4j, 0.5 - 0.6j, 1.0 + 0.2j]),
    ("4 inside + 1 on the BOTTOM edge", [0.2 + 0.1j, -0.3 - 0.4j, 0.5 - 0.6j, -0.55 + 0.5j,
                                         0.1 - 1.0j]),
])
def test_boundary_straddling_zeros_are_reported_even_when_the_count_matches(name, roots):
    """AUDIT Q-11 residual: corroboration by COINCIDENCE. A zero on the search-box boundary makes
    the winding untrustworthy (a ~pi phase step no densification removes, or a half-integer count)
    -- but the quad-tree usually FINDS that zero too, so ``len(poles) == round(winding)`` held and
    the diagnostic went silent. Measured on 12 boundary geometries: 7 were silenced this way,
    these seven, including a single zero on the right edge (winding 1.000000, maxstep 3.14 rad,
    one pole returned, no warning).

    Matching counts is not corroboration when a zero sits ON the contour, because BOTH numbers
    come from the same ambiguity. The independent signal is whether the count survives a hair's
    shrink and a hair's growth of the box: only a zero within ``_BOX_RECHECK_REL`` of the boundary
    changes side between the two."""
    import warnings
    from dynameta.optics.resonance import _MAXSTEP_TRUST, _WINDING_INT_TOL, _winding_densified

    func = _polyfunc(roots)
    rect = (-1.0, 1.0, -1.0, 1.0)
    w, ms = _winding_densified(func, rect, 40)
    count = int(round(w))
    # precondition: this geometry IS the silenced kind -- untrustworthy winding, matching count
    assert ms > _MAXSTEP_TRUST or abs(w - count) > _WINDING_INT_TOL, (name, w, ms)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        poles = find_poles(func, 0.0 + 0.0j, 1.0 + 1.0j, n_grid=40)
    assert count > 0 and len(poles) == count, (name, count, len(poles))   # the coincidence
    flagged = [str(w_.message) for w_ in rec if issubclass(w_.category, RuntimeWarning)]
    assert flagged, name
    assert any("STRADDLES" in m for m in flagged), (name, flagged)

    # the policy knob works on this path too, and the poles are identical in all three modes
    with pytest.raises(RuntimeError, match="STRADDLES"):
        find_poles(func, 0.0 + 0.0j, 1.0 + 1.0j, n_grid=40, on_untrusted="raise")
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        quiet = find_poles(func, 0.0 + 0.0j, 1.0 + 1.0j, n_grid=40, on_untrusted="ignore")
    assert not [w_ for w_ in rec2 if issubclass(w_.category, RuntimeWarning)]
    assert quiet == poles


def test_a_trustworthy_winding_that_outcounts_the_search_is_reported():
    """AUDIT Q-11 residual, the OTHER side. The old check only ever looked at UNTRUSTWORTHY
    windings, so the opposite failure was silent: the argument principle counts N zeros with a
    perfectly believable contour, the quad-tree refines M < N of them, and the short list comes
    back with no diagnostic. Measured: a box with 8 simple zeros, winding exactly 8.000000 at
    maxstep 0.49 rad, returned 6 poles and said nothing."""
    import warnings
    from dynameta.optics.resonance import _MAXSTEP_TRUST, _winding_densified

    # the verifier's own 8-zero geometry, pinned so the gate does not drift with numpy's stream
    roots = [0.23192038375205482 + 0.030921803096026768j,
             -0.5132459423676295 + 0.19763929355103937j,
             -0.002985361700475808 + 0.6147498747936122j,
             -0.008932232461439149 + 0.11482212090586419j,
             0.000316665088244239 - 0.3250332535917761j,
             0.642015200552946 + 0.6016845734312937j,
             -0.21008764024884324 - 0.011584601542532824j,
             -0.3867203945104976 + 0.24612121090693617j]
    func = _polyfunc(roots)
    w, ms = _winding_densified(func, (-1.0, 1.0, -1.0, 1.0), 40)
    assert ms <= _MAXSTEP_TRUST and w == pytest.approx(8.0, abs=1e-6)   # a TRUSTED count of 8

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        poles = find_poles(func, 0.0 + 0.0j, 1.0 + 1.0j, n_grid=40)
    assert len(poles) == 6                                       # ... and the search fell short
    flagged = [str(x.message) for x in rec if issubclass(x.category, RuntimeWarning)]
    assert any("INCOMPLETE" in m and "argument principle counts 8" in m for m in flagged), flagged

    # a box whose count IS fully recovered stays silent (no blanket warning on multi-pole boxes)
    ok_roots = [0.2 + 0.1j, -0.3 - 0.4j, 0.5 - 0.6j]
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        got = find_poles(_polyfunc(ok_roots), 0.0 + 0.0j, 1.0 + 1.0j, n_grid=40)
    assert len(got) == 3
    assert not [x for x in rec2 if issubclass(x.category, RuntimeWarning)]


def test_on_untrusted_is_plumbed_through_every_pole_facing_entry_point():
    """AUDIT Q-11 residual (c). ``find_poles`` grew the diagnostic and the policy knob; the three
    public entry points a caller reaches poles through did not have it, so a user of
    ``berreman_enz_pole`` could neither escalate nor silence the report on a box they never see,
    and ``track_pole`` / ``q_budget`` had untrusted-result paths of their own with no policy at
    all. Same keyword, same vocabulary, same default everywhere."""
    import inspect
    import warnings
    from dynameta.optics import resonance as R

    for fn in (R.find_poles, R.track_pole, R.q_budget, R.berreman_enz_pole):
        p = inspect.signature(fn).parameters.get("on_untrusted")
        assert p is not None, fn.__name__
        assert p.default == "warn", fn.__name__
        with pytest.raises(ValueError, match="on_untrusted"):
            _call_with_untrusted(fn, "shout")

    # (a) berreman_enz_pole FORWARDS it -- and the RESULT is bit-identical in all three modes, on
    #     the shipped configuration. The Q-11 diagnostic is a diagnostic and nothing else.
    kw = dict(eps_inf=3.8, wp=2.0e15, gamma=5.0e13, thickness_m=15e-9,
              theta_rad=math.radians(60.0))
    a = berreman_enz_pole(**kw)
    b = berreman_enz_pole(on_untrusted="ignore", **kw)
    c = berreman_enz_pole(on_untrusted="raise", **kw)
    assert a["omega"] == b["omega"] == c["omega"] and a["Q"] == b["Q"] == c["Q"]
    assert a["poles"] == b["poles"] == c["poles"]
    # ... same for find_poles on the shipped Fabry-Perot boxes
    n, L = 3.5, 1.0e-6
    D = smatrix_pole_func([(complex(n) ** 2, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)
    for m in (3, 4, 5, 6):
        p = _fp_pole(n, L, m)
        ctr, spn = complex(p.real, p.imag), complex(0.06 * p.real, 1.5 * abs(p.imag))
        got = [find_poles(D, ctr, spn, n_grid=48, on_untrusted=mode)
               for mode in ("warn", "raise", "ignore")]
        assert got[0] == got[1] == got[2] and len(got[0]) == 1, m

    # (b) track_pole reports a jump it could not bisect away, and 'ignore' restores the silence.
    #     A pole function whose pole MOVES discontinuously with the parameter is the failure the
    #     jump_rel guard exists for; at max_subdiv = 0 the bisection cannot rescue it.
    def solver(p):
        return lambda z: complex(z) - complex(1.0 + p, -0.1)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        track = track_pole(solver, 1.0 - 0.1j, [0.0, 5.0], jump_rel=1e-3, max_subdiv=0)
    assert any("max_subdiv" in str(x.message)
               for x in rec if issubclass(x.category, RuntimeWarning))
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        quiet = track_pole(solver, 1.0 - 0.1j, [0.0, 5.0], jump_rel=1e-3, max_subdiv=0,
                           on_untrusted="ignore")
    assert not [x for x in rec2 if issubclass(x.category, RuntimeWarning)]
    assert quiet == track                       # the RESULT is identical in both modes
    with pytest.raises(RuntimeError, match="max_subdiv"):
        track_pole(solver, 1.0 - 0.1j, [0.0, 5.0], jump_rel=1e-3, max_subdiv=0,
                   on_untrusted="raise")
    # ... and a well-sampled track says nothing
    with warnings.catch_warnings(record=True) as rec3:
        warnings.simplefilter("always")
        track_pole(solver, 1.0 - 0.1j, list(np.linspace(0.0, 0.5, 21)))
    assert not [x for x in rec3 if issubclass(x.category, RuntimeWarning)]


def _call_with_untrusted(fn, mode):
    """Invoke each entry point far enough to hit its on_untrusted validation, no further."""
    if fn.__name__ == "find_poles":
        return fn(lambda z: complex(z), 0.0 + 0.0j, 1.0 + 1.0j, n_grid=8, on_untrusted=mode)
    if fn.__name__ == "track_pole":
        return fn(lambda p: (lambda z: complex(z)), 0.0 + 0.0j, [0.0], on_untrusted=mode)
    if fn.__name__ == "q_budget":
        return fn(lambda s: (lambda z: complex(z) - 1.0), 1.0 + 0.0j, on_untrusted=mode)
    return fn(eps_inf=3.8, wp=2.0e15, gamma=5.0e13, thickness_m=15e-9,
              theta_rad=math.radians(60.0), on_untrusted=mode)


# ------------------------------------------------------------------------------------------------
# AUDIT P-6: the scalar 2x2 rewrite is the module's ONE non-bit-identical change -- gate its drift
# ------------------------------------------------------------------------------------------------
def _stack_denominator_numpy_reference(omega, layers, pol, n_super, n_sub, k_par):
    """The PRE-P-6 array-numpy implementation of ``_stack_denominator``, verbatim: 2x2
    ``np.array`` builds and ``@`` products (which route through BLAS zgemm), ``np.sqrt(... + 0j)``
    and numpy complex division. Kept here as the reference the scalar rewrite is measured
    against."""
    from dynameta.optics.resonance import _admittance, _eval_eps

    k0 = omega / C_LIGHT
    kpar = complex(k_par)
    eps_super = complex(n_super) ** 2
    eps_sub = complex(n_sub) ** 2

    def kz_np(eps):
        return np.sqrt(eps * k0 * k0 - kpar * kpar + 0j)

    Y_super = _admittance(eps_super, kz_np(eps_super), pol)
    Y_sub = _admittance(eps_sub, kz_np(eps_sub), pol)
    Mc = np.eye(2, dtype=np.complex128)
    for e_spec, d in layers:
        eps = _eval_eps(e_spec, omega)
        kz = kz_np(eps)
        Y = _admittance(eps, kz, pol)
        phi = kz * float(d)
        c, s = np.cos(phi), np.sin(phi)
        Mc = Mc @ np.array([[c, -1j * s / Y], [-1j * Y * s, c]], dtype=np.complex128)
    return complex(Y_super * (Mc[0, 0] + Mc[0, 1] * Y_sub) + (Mc[1, 0] + Mc[1, 1] * Y_sub))


# SECONDARY sanity ceiling on the pole positions (see the mutation-teeth note in the test): the
# ACTIVE discriminator is the median relative drift of D(omega) itself, six decades tighter.
# Measured worst pole drift over every configuration below: 2.1e-16.
_P6_POLE_TOL = 1e-12


def test_p6_scalar_rewrite_drift_is_within_the_accepted_bound():
    """AUDIT P-6. ``_kz`` / ``_interface`` / ``_propagate`` / the ``layered_smatrix_complex``
    cascade / ``_stack_denominator`` were scalar physics written in array numpy; rewritten with
    plain complex scalars they are 2.7x-3.6x (``_stack_denominator``), 4.9x
    (``layered_smatrix_complex``) and 1.71x on ``find_poles`` end to end.

    This is the only change in the module that is NOT bit-identical -- numpy's 2x2 ``@`` routes
    through BLAS and accumulates differently from ``a*e + b*g``, and numpy's complex division
    differs from CPython's -- so the drift is GATED here rather than assumed, against the
    pre-P-6 implementation kept verbatim above. ``cmath.sqrt`` / ``cmath.exp`` are bit-identical
    to their numpy twins on the argument classes this module can reach, and ``np.cos`` /
    ``np.sin`` are NOT to ``cmath``'s, which is why the cosines stayed numpy -- see the module
    comment for the per-class measurement, including the two exceptions (denormal ``sqrt``
    arguments, ``exp`` arguments of magnitude >~ 400).

    WHAT HAS TEETH HERE, MEASURED (mutation test: copy the package, apply ONE algebraically-exact
    or physics-breaking rewrite of the 2x2 algebra, re-run these assertions):

      * ``median(rel drift of D)`` < 1e-15 in (a) is the ACTIVE discriminator and the reason this
        gate is worth running. It caught the control physics break (``-1j*s/Y`` -> ``-1j*s*Y`` in
        the characteristic matrix: median drift 1.8e13, i.e. a different function) AND a
        deliberate 1e-13 RELATIVE perturbation of D (median 4.9e-14) -- a change far too small to
        move a pole. A 1e-15 relative perturbation passes, which is the intended sensitivity: it
        is the accepted ulp band.
      * the POLE clauses in (b) (``< _P6_POLE_TOL`` = 1e-12, and ``worst < 1e-14``) never fire on
        their own. Under the 1e-13 perturbation the worst pole drift was 3.9e-15 -- inside BOTH
        bounds -- because Newton refinement to ``refine_tol = 1e-11`` cannot resolve a
        perturbation of D that small. They are kept as a SECONDARY sanity ceiling on the quantity
        the acceptance was actually stated on (and they do catch the control break, at 6.2e-2),
        not as the sharp instrument.
      * algebraically-EXACT reorderings pass BY DESIGN and that is not a hole: expanding
        ``_stack_denominator``, writing the p-pol admittance as ``1/(kz/eps)``, and
        ``omega * (1/C_LIGHT)`` all stayed inside the ulp band (median 1.6e-16 to 1.8e-16). This
        gate bounds the DRIFT of a rewrite; it is not a checksum of the source text."""
    from dynameta.optics.resonance import _stack_denominator

    configs = [
        ("FP slab", [(complex(3.5) ** 2, 1.0e-6)], "s", 1.0, 1.0, 0.0),
        ("3-layer lossy", [(complex(n) ** 2, d) for n, d in
                           zip((2.0 + 0.05j, 1.4 + 0.0j, 3.0 + 0.2j), (120e-9, 80e-9, 60e-9))],
         "p", 1.0, 1.0, k_par_from_angle(1.0, 2.0 * math.pi * C_LIGHT / 1300e-9,
                                         math.radians(40.0))),
        ("Drude film", [(lambda w: drude_eps(w, 3.8, 2.0e15, 1.0e14), 20e-9)],
         "p", 1.0, 1.0, k_par_from_angle(1.0, 2.0e15, math.radians(50.0))),
    ]

    # (a) D(omega) itself: 1-2 ulp over a wide swath of the complex plane
    rng = np.random.default_rng(0)
    for name, layers, pol, ns, nb, kp in configs:
        z = 1.2e15 * (1.0 + 0.3 * rng.standard_normal(1500)
                      + 0.3j * rng.standard_normal(1500))
        got = np.array([_stack_denominator(complex(w), layers, pol, ns, nb, kp) for w in z])
        ref = np.array([_stack_denominator_numpy_reference(complex(w), layers, pol, ns, nb, kp)
                        for w in z])
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
        # max is loose (it is dominated by the samples nearest a zero of D, where the relative
        # metric has no scale); the MEDIAN is the clause with teeth -- measured 1.6e-16, and a
        # rewrite that perturbs D by 1e-13 relative already reads 4.9e-14 here.
        assert np.max(rel) < 1e-12, (name, float(np.max(rel)))
        assert np.median(rel) < 1e-15, (name, float(np.median(rel)))

    # (b) POLE POSITIONS -- the quantity the acceptance bound is stated on. Every shipped gate
    #     configuration: Fabry-Perot m = 3..6 one at a time, the 3-pole box, and the Berreman/ENZ
    #     film through berreman_enz_pole's own eps-cleared function and default box.
    #     SECONDARY: see the mutation-teeth note in the docstring -- these clauses sit six decades
    #     above the median-D clause and never fire without it.
    worst = 0.0
    boxes = []
    for m in (3, 4, 5, 6):
        p = _fp_pole(3.5, 1.0e-6, m)
        boxes.append(([(complex(3.5) ** 2, 1.0e-6)], "s", 1.0, 1.0, 0.0, None,
                      complex(p.real, p.imag), complex(0.06 * p.real, 1.5 * abs(p.imag)), 48))
    p5 = _fp_pole(3.5, 500e-9, 5)
    boxes.append(([(complex(3.5) ** 2, 500e-9)], "s", 1.0, 1.0, 0.0, None,
                  complex(p5.real, p5.imag), complex(0.30 * p5.real, 0.6 * abs(p5.imag)), 40))
    for eps_inf, wp, gam, d_nm, th in ((1.0, 2.0e15, 1.0e14, 20.0, 50.0),
                                       (3.8, 2.0e15, 5.0e13, 15.0, 60.0),
                                       (4.0, 1.2e15, 2.0e13, 30.0, 45.0)):
        om_p = wp / math.sqrt(eps_inf)
        ef = (lambda w, a=eps_inf, b=wp, c=gam: drude_eps(w, a, b, c))
        boxes.append(([(ef, d_nm * 1e-9)], "p", 1.0, 1.0,
                      k_par_from_angle(1.0, om_p, math.radians(th)), ef,
                      complex(1.02 * om_p, -0.10 * om_p),
                      complex(0.14 * om_p, 0.099 * om_p), 48))

    for layers, pol, ns, nb, kp, clear, ctr, spn, ng in boxes:
        def make(impl, layers=layers, pol=pol, ns=ns, nb=nb, kp=kp, clear=clear):
            base = (lambda w: impl(complex(w), layers, pol, ns, nb, kp))
            if clear is None:
                return base
            return lambda w: base(w) * complex(clear(w))

        got = sorted(find_poles(make(_stack_denominator), ctr, spn, n_grid=ng,
                                on_untrusted="ignore"), key=lambda z: (z.real, z.imag))
        ref = sorted(find_poles(make(_stack_denominator_numpy_reference), ctr, spn, n_grid=ng,
                                on_untrusted="ignore"), key=lambda z: (z.real, z.imag))
        assert len(got) == len(ref) and got, (len(got), len(ref))
        for a, b in zip(got, ref):
            worst = max(worst, abs(a - b) / abs(b))
            assert abs(a - b) / abs(b) < _P6_POLE_TOL, (a, b)
            assert abs(pole_q(a) - pole_q(b)) / abs(pole_q(b)) < _P6_POLE_TOL
    assert worst < 1e-14, "measured 2.1e-16; a jump to {:g} means the algebra changed".format(worst)

    # (c) the OTHER half of the P-6 rewrite: layered_smatrix_complex's amplitude cascade. Its
    #     R / T are already pinned against the independent `tmm` oracle at 1e-10 by gate 1; this
    #     bounds the ulp-level drift of the complex amplitudes themselves.
    def m11_numpy_reference(omega, layers, pol, ns, nb, kp):
        from dynameta.optics.resonance import _admittance, _eval_eps
        k0 = complex(omega) / C_LIGHT
        kpar = complex(kp)

        def kzn(eps):
            return np.sqrt(eps * k0 * k0 - kpar * kpar + 0j)

        eps_l = [_eval_eps(e, complex(omega)) for e, _ in layers]
        kz_l = [kzn(e) for e in eps_l]
        Y = ([_admittance(complex(ns) ** 2, kzn(complex(ns) ** 2), pol)]
             + [_admittance(e, k, pol) for e, k in zip(eps_l, kz_l)]
             + [_admittance(complex(nb) ** 2, kzn(complex(nb) ** 2), pol)])

        def iface(a, b):
            rho = b / a
            return 0.5 * np.array([[1.0 + rho, 1.0 - rho], [1.0 - rho, 1.0 + rho]],
                                  dtype=np.complex128)

        M = iface(Y[0], Y[1])
        for j in range(len(layers)):
            e = np.exp(1j * kz_l[j] * float(layers[j][1]))
            M = M @ np.array([[1.0 / e, 0.0], [0.0, e]], dtype=np.complex128) \
                  @ iface(Y[j + 1], Y[j + 2])
        return complex(M[0, 0])

    for name, layers, pol, ns, nb, kp in configs:
        for w in (1.2e15, complex(1.35e15, -2.0e13), complex(9.0e14, -5.0e13)):
            got = layered_smatrix_complex(w, layers, pol=pol, n_super=ns, n_sub=nb,
                                          k_par_m=kp).M11
            want = m11_numpy_reference(w, layers, pol, ns, nb, kp)
            assert abs(got - want) / abs(want) < 1e-12, (name, w, got, want)


# ------------------------------------------------------------------------------------------------
# AUDIT Q-12: M11 is NOT a valid pole function -- the docstrings used to recommend it
# ------------------------------------------------------------------------------------------------
def test_m11_winding_miscounts_across_a_layer_branch_cut_but_the_denominator_does_not():
    """Audit Q-12. ``SMatrix.M11`` and ``find_poles`` used to recommend feeding ``M11`` to the
    pole finder, which ``_stack_denominator`` explicitly documents as WRONG: every Abeles entry
    (cos(kz d), sin(kz d)/Y, Y sin(kz d)) is EVEN in the layer kz, hence a function of
    kz^2 = eps k0^2 - k_par^2 -- polynomial in omega, no branch cut -- whereas M11 carries an
    explicit exp(+-i kz d) and inherits the layer's sqrt branch point.

    This pins the consequence rather than the prose. Put the LAYER light line (kz_layer = 0, at
    omega = k_par c / n) inside a small box. Both functions have the SAME zero in there -- they
    differ by nonvanishing factors, and both dip to ~5e-4 of their box-scale at the same point --
    but only the branch-cut-free one counts it."""
    import warnings

    from dynameta.optics.resonance import _winding_densified, layered_smatrix_complex

    n, L, m = 3.5, 1.0e-6, 5
    layers = [(complex(n) ** 2, L)]
    om = m * math.pi * C_LIGHT / (n * L)
    k_par = om * n / C_LIGHT                       # layer branch point sits exactly at omega = om

    D = smatrix_pole_func(layers, pol="s", k_par_m=k_par)

    def M11(w):
        return layered_smatrix_complex(w, layers, pol="s", k_par_m=k_par).M11

    h = 0.05
    rect = (om * (1 - h), om * (1 + h), -h * om, h * om)

    # ground truth: a zero IS inside, and it is the same zero for both functions
    re = np.linspace(rect[0], rect[1], 41)
    im = np.linspace(rect[2], rect[3], 41)
    for f in (D, M11):
        Z = np.array([[abs(f(complex(a, b))) for b in im] for a in re])
        assert Z.min() < 1e-3 * Z.max(), "no zero in the probe box"

    w_D, ms_D = _winding_densified(D, rect, 80)
    w_M, ms_M = _winding_densified(M11, rect, 80)
    assert w_D == pytest.approx(1.0, abs=1e-6)     # analytic: the argument principle works
    assert ms_D < 1.2                              # ... and its contour is well sampled
    assert w_M == pytest.approx(0.0, abs=1e-6)     # M11 MISCOUNTS the very same zero
    assert ms_M > 3.0                              # ... with the ~pi jump of the branch cut

    # end to end: the recommended function finds the pole silently; M11 finds nothing and the
    # Q-11 diagnostic is what tells you so.
    centre, span = complex(om, 0.0), complex(h * om, h * om)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = find_poles(D, centre, span, n_grid=80)
    assert len(got) == 1
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert find_poles(M11, centre, span, n_grid=80) == []
    assert [w for w in rec if issubclass(w.category, RuntimeWarning)]


def test_berreman_enz_pole_thin_drude_film():
    # ITO-like Drude film: eps_inf = 1, omega_p ~ 2e15 rad/s (ENZ in the near-IR), moderate loss.
    eps_inf, wp, gamma = 1.0, 2.0e15, 1.0e14
    omega_p = wp / math.sqrt(eps_inf)
    res = berreman_enz_pole(eps_inf=eps_inf, wp=wp, gamma=gamma, thickness_m=50e-9,
                            theta_rad=math.radians(50.0), n_super=1.0, n_sub=1.0)
    pole = res["omega"]
    assert pole.imag < 0.0                                    # decaying (exp(-i w t))
    assert abs(pole.real - omega_p) / omega_p < 0.05          # near omega_p (within a few %)
    assert 0.0 < res["Q"] < 1e6 and math.isfinite(res["Q"])  # finite Q

    # Thinning the film pushes the ENZ mode TOWARD omega_p (thin-film limit).
    res_thin = berreman_enz_pole(eps_inf=eps_inf, wp=wp, gamma=gamma, thickness_m=15e-9,
                                 theta_rad=math.radians(50.0), n_super=1.0, n_sub=1.0)
    assert abs(res_thin["omega"].real - omega_p) < abs(pole.real - omega_p)


# ------------------------------------------------------------------------------------------------
# Gate 7: branch robustness with an evanescent substrate (k_par > n_sub omega/c at the pole)
# ------------------------------------------------------------------------------------------------
def test_evanescent_substrate_pole_stable():
    # Dense superstrate (n=1.5), high-index slab (n=3), low-index substrate (n=1.0); at 45 deg the
    # in-plane wavevector k_par = 1.5 sin45 k0 ~ 1.06 k0 exceeds n_sub k0 = k0, so the substrate
    # channel is EVANESCENT (kz_sub purely imaginary, principal-branch decaying). The slab still
    # supports a leaky resonance; it must be found and be stable under n_grid doubling.
    n_super, n_slab, n_sub = 1.5, 3.0, 1.0
    L = 1.0e-6
    theta = math.radians(45.0)
    # Real carrier near a half-wave slab resonance; fix k_par there (QNM convention).
    omega_ref = math.pi * C_LIGHT / (n_slab * L)              # ~ first-order slab resonance scale
    k_par = k_par_from_angle(n_super, omega_ref, theta)
    # Confirm the substrate channel is evanescent at the REAL carrier: k_par > n_sub * omega_ref/c,
    # so kz_sub(omega_ref) is purely imaginary (a bound/decaying substrate tail, no radiation loss
    # into the substrate -- this exercises the principal (outgoing/decaying) sqrt branch).
    assert k_par > n_sub * omega_ref / C_LIGHT
    kz_sub_carrier = np.sqrt((complex(n_sub) ** 2) * (omega_ref / C_LIGHT) ** 2 - k_par ** 2 + 0j)
    assert abs(kz_sub_carrier.real) < 1e-9 * abs(kz_sub_carrier.imag)   # purely evanescent
    assert kz_sub_carrier.imag > 0.0                        # principal branch => decaying tail

    func = smatrix_pole_func([(complex(n_slab) ** 2, L)], pol="p",
                             n_super=n_super, n_sub=n_sub, k_par_m=k_par)
    center = complex(omega_ref, -0.15 * omega_ref)
    span = complex(0.4 * omega_ref, 0.25 * omega_ref)
    poles_a = find_poles(func, center, span, n_grid=40, refine_tol=1e-12)
    poles_b = find_poles(func, center, span, n_grid=80, refine_tol=1e-12)
    decaying_a = [p for p in poles_a if p.imag < 0.0 and p.real > 0.0]
    assert decaying_a, "no decaying pole found with an evanescent substrate"
    p0 = min(decaying_a, key=lambda p: abs(p.real - omega_ref))
    # Stability under n_grid doubling.
    p1 = min([p for p in poles_b if p.imag < 0.0 and p.real > 0.0],
             key=lambda p: abs(p - p0))
    assert abs(p1 - p0) <= 1e-8 * abs(p0)


# ------------------------------------------------------------------------------------------------
# Sanity: drude_eps sign convention (passive => Im(eps) > 0 for real omega)
# ------------------------------------------------------------------------------------------------
def test_drude_eps_passive_sign_and_enz():
    wp, gamma, eps_inf = 2.0e15, 1.0e14, 1.0
    w = 1.0e15
    eps = complex(drude_eps(w, eps_inf, wp, gamma))
    assert eps.imag > 0.0                                     # exp(-i w t) passive absorber
    # ENZ crossing of Re(eps) at omega ~ wp / sqrt(eps_inf).
    enz = wp / math.sqrt(eps_inf)
    assert complex(drude_eps(enz, eps_inf, wp, gamma)).real == pytest.approx(0.0, abs=5e-2 * eps_inf)


# ------------------------------------------------------------------------------------------------
# AUDIT V-3: the polarization vocabulary is EXACTLY {'s', 'p'} at every entry point
# ------------------------------------------------------------------------------------------------
_OFF_VOCAB = ["TE", "TM", "te", "tm", "S", "P", "x", "y", "", "sp"]


def test_smatrix_pole_func_validates_pol():
    """smatrix_pole_func -- the module's primary public pole finder, in __all__ -- used to accept
    ANY string and silently return the P-POL function ('TE', 'TM', 'x' all gave p-pol), while its
    sibling layered_smatrix_complex raised on the same input. Both now reject the same set, and the
    factory raises EAGERLY (at build time, not inside find_poles)."""
    layers = [(4.0 + 0.1j, 200e-9)]
    for bad in _OFF_VOCAB:
        with pytest.raises(ValueError, match="pol must be"):
            smatrix_pole_func(layers, pol=bad)                # eager: no closure is handed back
        with pytest.raises(ValueError, match="pol must be"):
            layered_smatrix_complex(1.0e15, layers, pol=bad)
    for good in ("s", "p"):                                   # the accepted set is unchanged
        assert callable(smatrix_pole_func(layers, pol=good))
        assert layered_smatrix_complex(1.0e15, layers, pol=good) is not None


def test_pol_fallthrough_was_answer_changing():
    """Discrimination: s and p pole functions are genuinely different at oblique incidence, so the
    old silent p-pol default was not a harmless alias -- a pol='TE' pole hunt returned plausible
    p-pol poles. (Also pins that the two accepted spellings still disagree, i.e. the guard did not
    accidentally collapse them.)"""
    lam = 1300e-9
    omega = 2.0 * math.pi * C_LIGHT / lam
    layers = [(4.0 + 0.1j, 200e-9)]
    k_par = 0.6 * omega / C_LIGHT                             # ~37 deg in vacuum
    ds = smatrix_pole_func(layers, pol="s", k_par_m=k_par)(omega)
    dp = smatrix_pole_func(layers, pol="p", k_par_m=k_par)(omega)
    assert abs(ds - dp) > 0.1 * max(abs(ds), abs(dp))


def test_admittance_twins_agree_on_off_vocabulary():
    """AUDIT V-3/V-8: the duplicated _admittance helpers had MIRROR-IMAGE silent fallbacks
    (resonance -> p-pol, nonlocal_tmm -> s-pol), so the two modules returned different physics for
    the same unrecognized string. Both must now raise -- the stricter behaviour."""
    from dynameta.optics.nonlocal_tmm import _admittance as adm_nl
    from dynameta.optics.resonance import _admittance as adm_res
    eps, kz = 4.0 + 0.1j, 1.2e7 + 0j
    for bad in _OFF_VOCAB:
        with pytest.raises(ValueError, match="pol must be"):
            adm_res(eps, kz, bad)
        with pytest.raises(ValueError, match="pol must be"):
            adm_nl(eps, kz, bad)
    for good in ("s", "p"):                                   # and they still agree where defined
        assert adm_res(eps, kz, good) == adm_nl(eps, kz, good)


# ------------------------------------------------------------------------------------------------
# Gate 4c (finding Q-15): the DISPERSIVE two-pass bias must be reported, not hidden
# ------------------------------------------------------------------------------------------------
def test_q_budget_reports_the_dispersive_omega0_shift():
    """audit Q-15: the Q_rad/Q_abs two-pass is evaluated at two DIFFERENT resonance frequencies
    whenever the loss knob is dispersive -- zeroing a Drude gamma also moves Re(eps) by
    wp^2 g^2 / (w^2 (w^2 + g^2)), so the lossless linewidth is measured at Re(pole_rad), not at
    Re(pole_total). The bias is small but real (a few % on gamma_abs at gamma/omega_0 ~ 0.1) and
    grows with gamma/omega_0, so q_budget must RETURN it as the split's own error bar."""
    eps_inf, wp, d, theta = 4.0, 2.0e15, 20e-9, np.radians(30.0)
    omega_p = wp / np.sqrt(eps_inf)
    k_par = k_par_from_angle(1.0, omega_p, theta)
    shifts = []
    for gamma in (1.0e13, 1.0e14):
        _raw, cleared = _enz_budget_factories(eps_inf, wp, gamma, d, k_par)
        seed = berreman_enz_pole(eps_inf=eps_inf, wp=wp, gamma=gamma, thickness_m=d,
                                 theta_rad=theta)["omega"]
        out = q_budget(cleared, seed, refine_tol=1e-12)
        assert out["pole_rad_ok"] and np.isfinite(out["omega0_shift_rel"])
        # it is a genuine measurement of the two passes, not a placeholder
        assert out["omega0_shift_rel"] == pytest.approx(
            (out["pole_rad"].real - out["pole_total"].real) / out["pole_total"].real, rel=1e-12)
        shifts.append(abs(out["omega0_shift_rel"]))
    # DISCRIMINATION: the bias grows with gamma/omega_0, which is the whole point of reporting it
    assert shifts[1] > shifts[0]
    assert shifts[1] < 0.05                     # still a small, usable split at gamma/w0 ~ 0.1
