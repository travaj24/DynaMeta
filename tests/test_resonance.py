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
