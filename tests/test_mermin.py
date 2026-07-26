"""Roadmap 2.3 -- Mermin / extended-Drude damping (frequency-dependent scattering).

Gates:
  * gamma(omega)->const == DrudeOptical byte-identical (1e-15), scalar and callable mass;
  * the analytic k->0 Mermin == plain Drude claim, verified NUMERICALLY against a finite-k
    hydrodynamic Lindhard chi (residual ~ (beta k / w)^2 -> 0), plus the local
    MerminDrudeOptical == ExtendedDrudeOptical identity and the finite-k deferral;
  * the ITO extended-Drude preset vs a plain Drude fixed at gamma_dc: absorption (Im eps)
    REDUCED in the near-IR/below-plasma window (direction pinned);
  * check_kk residual SMALL for causal models (Drude AND a 2-oscillator Lorentz -- two
    opposite extremes) and LARGE for acausal ones (a 2x and a 10x step in gamma, a
    sign-flipped Im), with the ranking held at the shipped grid AND at 4x it; plus the
    regression that the OLD "causal floor" was pure O(h) quadrature error, and the honest
    re-scoping of the ITO preset out of the causal class (finding Q-5);
  * plugs into tmm_reference for a 3-layer stack (R, T, A finite; energy budget holds);
  * exp(-i omega t) sign convention (Im eps > 0 where absorbing).

Independent oracle for the k->0 claim: a self-contained finite-k Mermin dielectric built from
the collisionless hydrodynamic Lindhard function, evaluated in this test only (the library never
computes finite-k Mermin -- that is roadmap 2.4). Mermin, Phys. Rev. B 1, 2362 (1970).
"""
import warnings

import numpy as np
import pytest

from dynameta.constants import Q_E, EPS0, M_E, HBAR, C_LIGHT
from dynameta.materials.optical_model import (
    DrudeOptical, ExtendedDrudeOptical, MerminDrudeOptical,
    gamma_ito_extended, check_kk,
)
from dynameta.materials.scattering import KaneOpticalMass

# ---- representative near-IR ITO ----
EPS_INF = 4.25
M_OPT = 0.35 * M_E
N_ITO = 5.0e26                       # m^-3
GAMMA_DC = 1.5e14                    # rad/s (DC / low-omega impurity-dominated damping)

WP2 = N_ITO * Q_E * Q_E / (EPS0 * M_OPT)
WP = np.sqrt(WP2)                    # ~2.13e15 rad/s


def _omega(lambda_m):
    return 2.0 * np.pi * C_LIGHT / lambda_m


# ===========================================================================
# Gate 1 -- gamma(omega)->const == DrudeOptical byte-identical (1e-15).
# ===========================================================================
def test_const_gamma_byte_identical_scalar_mass():
    g = 1.1e14
    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=g)
    ext_scalar = ExtendedDrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_omega=g)
    ext_callable = ExtendedDrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT,
                                        gamma_omega=lambda w: g)
    mermin = MerminDrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_omega=g)  # k=0
    for lam in (900e-9, 1300e-9, 1550e-9, 2000e-9):
        ref = complex(drude.eps(lam, n_m3=N_ITO))
        for model in (ext_scalar, ext_callable, mermin):
            e = complex(model.eps(lam, n_m3=N_ITO))
            assert e == ref or abs(e - ref) <= 1e-15 * abs(ref)


def test_const_gamma_byte_identical_callable_mass():
    # delegation must preserve a CALLABLE optical mass byte-identically too.
    g = 9.0e13
    mass = KaneOpticalMass(m0_kg=0.30 * M_E, alpha_eV=0.4)
    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=mass, gamma_rad_s=g)
    ext = ExtendedDrudeOptical(eps_inf=EPS_INF, m_opt_kg=mass, gamma_omega=g)
    n_arr = np.array([3.0e26, 5.0e26, 1.0e27])
    ref = np.asarray(drude.eps(1300e-9, n_m3=n_arr), dtype=np.complex128)
    got = np.asarray(ext.eps(1300e-9, n_m3=n_arr), dtype=np.complex128)
    assert np.allclose(got, ref, rtol=0.0, atol=0.0) or np.max(np.abs(got - ref)) <= 1e-15 * np.max(np.abs(ref))


# ===========================================================================
# Gate 2 -- the k->0 Mermin == plain Drude analysis, NUMERICALLY verified.
#
# Independent oracle: the collisionless hydrodynamic Lindhard function
#   eps_L(k, w) = 1 - wp^2 / (w^2 - beta^2 k^2),   eps_L(k,0) = 1 + k_TF^2/k^2,
# fed into the Mermin formula (Mermin 1970 Eq. 8). As k -> 0 the full local
# eps_inf + (eps_M - 1) must collapse to plain Drude, with residual ~ (beta k / w)^2.
# ===========================================================================
def _beta2():
    kF = (3.0 * np.pi ** 2 * N_ITO) ** (1.0 / 3.0)
    vF = HBAR * kF / M_OPT
    return 0.6 * vF ** 2                      # (3/5) v_F^2 (high-frequency)


def _eps_lindhard_hydro(k, w, beta2):
    return 1.0 - WP2 / (w * w - beta2 * k * k)


def _eps_mermin_local(k, w, g, beta2):
    """Full local eps = eps_inf + (eps_M - 1), Mermin Eq. 8 with hydrodynamic Lindhard."""
    chi_dyn = _eps_lindhard_hydro(k, w + 1j * g, beta2) - 1.0
    chi_stat = _eps_lindhard_hydro(k, 0.0, beta2) - 1.0
    eps_M = 1.0 + (1.0 + 1j * g / w) * chi_dyn / (1.0 + (1j * g / w) * chi_dyn / chi_stat)
    return EPS_INF + (eps_M - 1.0)


def test_mermin_k_to_zero_equals_drude():
    beta2 = _beta2()
    w = _omega(1550e-9)
    g = 1.0e14
    drude = complex(DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=g).eps(
        2.0 * np.pi * C_LIGHT / w, n_m3=N_ITO))
    ks = [1.0e8, 1.0e7, 1.0e6, 1.0e5]
    resid = [abs(_eps_mermin_local(k, w, g, beta2) - drude) for k in ks]
    # monotone quadratic convergence: each 10x smaller k -> ~100x smaller residual.
    for a, b in zip(resid[:-1], resid[1:]):
        assert b < a
        assert b < a * 0.02                 # ~1/100 (k^2 scaling), with margin
    assert resid[-1] < 1e-6                  # k=1e5: essentially plain Drude


def test_mermin_local_equals_extended_and_finite_k_deferred():
    # LOCAL (k=0) MerminDrudeOptical is EXACTLY ExtendedDrudeOptical (proof in module header).
    ext = ExtendedDrudeOptical(EPS_INF, M_OPT, gamma_ito_extended)
    merm = MerminDrudeOptical(EPS_INF, M_OPT, gamma_ito_extended, k_per_m=0.0)
    for lam in (1000e-9, 1550e-9):
        assert complex(merm.eps(lam, n_m3=N_ITO)) == complex(ext.eps(lam, n_m3=N_ITO))
    # finite-k number-conserving branch is DEFERRED (roadmap 2.4), not silently local.
    merm_k = MerminDrudeOptical(EPS_INF, M_OPT, gamma_ito_extended, k_per_m=1.0e8)
    with pytest.raises(NotImplementedError):
        merm_k.eps(1550e-9, n_m3=N_ITO)


# ===========================================================================
# Gate 3 -- ITO extended-Drude preset: absorption direction vs plain Drude @ gamma_dc.
# ===========================================================================
def test_ito_preset_reduces_below_plasma_absorption():
    plain = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAMMA_DC)
    ext = ExtendedDrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_omega=gamma_ito_extended)
    # near-IR / below-plasma window (device band): extended gamma < gamma_dc -> Im eps reduced.
    for lam in (1000e-9, 1300e-9, 1550e-9, 2000e-9):
        im_plain = complex(plain.eps(lam, n_m3=N_ITO)).imag
        im_ext = complex(ext.eps(lam, n_m3=N_ITO)).imag
        assert im_plain > 0.0 and im_ext > 0.0        # both passive
        assert im_ext < im_plain                       # DIRECTION: absorption reduced
    # pin the magnitude at telecom (1550 nm): ~0.59 of the plain-Drude absorption.
    r1550 = complex(ext.eps(1550e-9, n_m3=N_ITO)).imag / complex(plain.eps(1550e-9, n_m3=N_ITO)).imag
    assert 0.45 < r1550 < 0.75
    # and the extended damping there is below the DC value (the physical cause).
    assert ext.gamma_at(_omega(1550e-9)) < GAMMA_DC


# ===========================================================================
# Gate 4 -- check_kk discriminates causal (small) from acausal step (large).
# ===========================================================================
def _kk_grid():
    N, wmax = 8000, 80.0 * WP
    return np.linspace(wmax / N, wmax, N)


def _band():
    return (0.4 * WP, 5.0 * WP)


def _lorentz_band():
    return (0.3e15, 4.0e15)


class _Lorentz2:
    """A manifestly causal 2-oscillator Lorentz dielectric -- no 1/omega DC pole, so it is the
    OPPOSITE extreme from Drude as a causal reference. `flip_im` negates Im(eps), which breaks
    causality while leaving Re(eps) (and hence the normalization scale) untouched.

    OpticalModel duck-type: only eps(lambda_m, n_m3=...) is needed by check_kk.
    """

    OSCS = ((1.0e15, 6.0e14, 1.0e14), (2.2e15, 9.0e14, 2.0e14))   # (omega0, f, gamma) rad/s

    def __init__(self, eps_inf=2.0, flip_im=False):
        self.eps_inf, self.flip_im = eps_inf, flip_im

    def eps(self, lambda_m, *, n_m3=None):
        w = 2.0 * np.pi * C_LIGHT / float(lambda_m)
        e = complex(self.eps_inf)
        for w0, f, g in self.OSCS:
            e += f ** 2 / (w0 ** 2 - w ** 2 - 1j * g * w)        # exp(-i w t): Im(eps) > 0
        return complex(e.real, -e.imag) if self.flip_im else e


def _kk_models():
    """(name, model, n_m3, band, causal?) -- two causal references at opposite extremes (Drude
    with its DC pole, Lorentz without one) and three acausal probes of decreasing severity."""
    return [
        ("drude", DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=1.0e14),
         N_ITO, _band(), True),
        ("lorentz2", _Lorentz2(), None, _lorentz_band(), True),
        ("lorentz_flip_im", _Lorentz2(flip_im=True), None, _lorentz_band(), False),
        ("gamma_step_2x", ExtendedDrudeOptical(
            EPS_INF, M_OPT, lambda w: np.where(np.asarray(w) < 1.2e15, 1.5e14, 3.0e14)),
         N_ITO, _band(), False),
        ("gamma_step_10x", ExtendedDrudeOptical(
            EPS_INF, M_OPT, lambda w: np.where(np.asarray(w) < 1.2e15, 3.0e13, 3.0e14)),
         N_ITO, _band(), False),
    ]


def test_check_kk_causal_small_acausal_large():
    """RE-BASELINED for finding Q-5. The old absolute thresholds (0.035 / 0.30) were calibrated to
    a "causal floor" that was pure O(h) quadrature error -- the residual scaled EXACTLY with the
    grid step, and plain Drude (the DC pole) was the worst possible causal reference, which is why
    a 2x gamma discontinuity was invisible. With the endpoint-consistent Maclaurin variant the
    causal floor drops ~80x at this grid and then falls as O(h^2), so the thresholds move with it.

    Measured at the fix (N = 8000, rms_norm): drude 2.05e-4, lorentz2 3.10e-4 | 2x step 8.10e-3,
    10x step 1.82e-2, Im-flipped Lorentz 6.11e-1. Legacy (edge_correct=False) drude: 1.65e-2.
    """
    grid = _kk_grid()
    res = {name: check_kk(m, grid, n_m3=nn, metric_band=bd)
           for name, m, nn, bd, _c in _kk_models()}

    # causal models -- both extremes, DC-pole and no-DC-pole
    assert res["drude"]["rms_norm"] < 1.0e-3 and res["drude"]["max_norm"] < 1.0e-2
    assert res["lorentz2"]["rms_norm"] < 1.0e-3

    causal_worst = max(res["drude"]["rms_norm"], res["lorentz2"]["rms_norm"])
    # THE GATE: even a 2x discontinuity in gamma(omega) scores above EVERY causal model.
    assert res["gamma_step_2x"]["rms_norm"] > 3.0e-3
    assert res["gamma_step_2x"]["rms_norm"] > 5.0 * causal_worst
    assert res["gamma_step_10x"]["rms_norm"] > 8.0e-3
    assert res["gamma_step_10x"]["rms_norm"] > 10.0 * causal_worst
    # a sign-flipped Im is grossly acausal and must be nowhere near the causal band
    assert res["lorentz_flip_im"]["rms_norm"] > 0.1
    assert res["lorentz_flip_im"]["rms_norm"] > 100.0 * causal_worst
    # the localized spike at the jump is still a tell
    assert res["gamma_step_10x"]["max_norm"] > 5.0 * res["drude"]["max_norm"]


def test_check_kk_discrimination_survives_grid_REFINEMENT():
    """finding Q-5's core requirement, SCOPED (fix-verify W1 item 5): the ranking must not degrade
    when the grid is REFINED from a resolved one. A causal model's residual is quadrature-limited
    and FALLS with refinement; a genuine acausality is a property of the model and stays put.
    Checked at the shipped grid and at 4x.

    THIS IS NOT UNCONDITIONAL GRID-INDEPENDENCE and the docstring no longer claims it is. Below
    N ~ 6000 on this grid the ranking INVERTS, because the causal 2-oscillator Lorentz's narrowest
    line (gamma = 1e14) is no longer resolved: at N = 4000 (gamma/h = 2.3) it scores 7.79e-3,
    level with the acausal 2x gamma jump, and at N = 2000 (gamma/h = 1.2) it scores 4.56e-2, ABOVE
    the 10x jump. ``check_kk`` now reports ``feature_pts`` and warns below 5 grid points per line;
    the coarse end is pinned by test_check_kk_warns_when_the_sharpest_line_is_under_resolved.

    Measured rms_norm (N = 8000 -> 32000): drude 2.05e-4 -> 2.78e-5, lorentz2 3.10e-4 -> 1.92e-5,
    2x step 8.10e-3 -> 8.14e-3, 10x step 1.82e-2 -> 1.45e-2, flipped Im 6.11e-1 -> 6.01e-1. The
    2x-step-to-worst-causal margin therefore GROWS from 26x to 293x.
    """
    wmax = 80.0 * WP
    models = _kk_models()
    out = {}
    for N in (8000, 32000):
        g = np.linspace(wmax / N, wmax, N)
        for name, m, nn, bd, causal in models:
            if N > 8000 and name == "gamma_step_10x":
                continue                                   # 4 models at 4x is enough (runtime)
            out[(name, N)] = check_kk(m, g, n_m3=nn, metric_band=bd)["rms_norm"]

    for N in (8000, 32000):
        worst_causal = max(out[("drude", N)], out[("lorentz2", N)])
        assert out[("gamma_step_2x", N)] > 5.0 * worst_causal, (N, out)
        assert out[("lorentz_flip_im", N)] > 100.0 * worst_causal, (N, out)

    # causal residuals FALL with refinement (the O(h^2) floor); acausal ones do NOT.
    assert out[("drude", 32000)] < 0.5 * out[("drude", 8000)]
    assert out[("lorentz2", 32000)] < 0.5 * out[("lorentz2", 8000)]
    assert 0.5 < out[("gamma_step_2x", 32000)] / out[("gamma_step_2x", 8000)] < 2.0
    assert 0.5 < out[("lorentz_flip_im", 32000)] / out[("lorentz_flip_im", 8000)] < 2.0


def test_check_kk_warns_when_the_sharpest_line_is_under_resolved():
    """FIX-VERIFY W1 item 5. ``check_kk`` claimed its discrimination was "GRID-INDEPENDENT" and
    that refining "can never invert the ranking". The second half is true; the first is not. Below
    N ~ 6000 on the reference grid the CAUSAL 2-oscillator Lorentz is no longer resolved (its
    narrowest line is gamma = 1e14 and h = 80 wp / N), its residual is inflated by quadrature, and
    the ranking inverts:

        N = 2000 (gamma/h = 1.17): lorentz2 4.56e-2 > the acausal 10x gamma jump 4.66e-2 * 0.98
        N = 4000 (gamma/h = 2.35): lorentz2 7.79e-3, level with the acausal 2x jump 7.93e-3
        N = 8000 (gamma/h = 4.69): lorentz2 1.76e-4, 39x below the 2x jump  <- inside the envelope

    ``feature_pts`` measures that resolution model-free and a RuntimeWarning now fires below 5."""
    wmax = 80.0 * WP
    lor = _Lorentz2()
    step2 = ExtendedDrudeOptical(
        EPS_INF, M_OPT, lambda w: np.where(np.asarray(w) < 1.2e15, 1.5e14, 3.0e14))
    band = _band()
    got = {}
    for N in (4000, 8000):
        g = np.linspace(wmax / N, wmax, N)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            got[("lorentz2", N)] = check_kk(lor, g, metric_band=band, self_calib=False)
        got[("lorentz2_warn", N)] = [w for w in wl if issubclass(w.category, RuntimeWarning)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got[("step2", N)] = check_kk(step2, g, n_m3=N_ITO, metric_band=band, self_calib=False)

    # the COARSE grid: under-resolved, warned about, and the ranking is indeed destroyed
    assert got[("lorentz2", 4000)]["feature_pts"] < 5.0
    assert got[("lorentz2_warn", 4000)], "no under-resolution warning at N = 4000"
    assert "grid points" in str(got[("lorentz2_warn", 4000)][0].message)
    assert got[("step2", 4000)]["rms_norm"] < 2.0 * got[("lorentz2", 4000)]["rms_norm"]

    # the RESOLVED grid: no warning, and the acausal model separates by more than an order
    assert got[("lorentz2", 8000)]["feature_pts"] >= 5.0
    assert not got[("lorentz2_warn", 8000)], [str(w.message) for w in got[("lorentz2_warn", 8000)]]
    assert got[("step2", 8000)]["rms_norm"] > 10.0 * got[("lorentz2", 8000)]["rms_norm"]

    # feature_pts tracks gamma/h: refining 2x must widen the resolved line by ~2x
    assert (got[("lorentz2", 8000)]["feature_pts"]
            > 1.3 * got[("lorentz2", 4000)]["feature_pts"])


def test_check_kk_causal_floor_was_pure_quadrature_error():
    """REGRESSION (finding Q-5). The legacy first-order sum's normalized residual is EXACTLY
    proportional to the grid step -- `rms_norm_per_h` is 132.19 at both N = 8000 and N = 32000, a
    4-significant-figure match -- i.e. the old "causal floor" measured the Maclaurin step size and
    carried no causality information at all. The endpoint-consistent variant breaks that scaling
    (1.64 -> 0.89 over the same 4x refinement) and lowers the residual 80x at the shipped grid."""
    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=1.0e14)
    wmax, band = 80.0 * WP, _band()
    legacy, fixed = {}, {}
    for N in (8000, 32000):
        g = np.linspace(wmax / N, wmax, N)
        legacy[N] = check_kk(drude, g, n_m3=N_ITO, metric_band=band, edge_correct=False,
                             self_calib=False)
        fixed[N] = check_kk(drude, g, n_m3=N_ITO, metric_band=band, self_calib=False)
    # legacy: O(h) exactly -- rms_norm / h is grid-INVARIANT
    assert legacy[32000]["rms_norm_per_h"] == pytest.approx(legacy[8000]["rms_norm_per_h"],
                                                            rel=1e-3)
    # fixed: the same indicator FALLS, i.e. the residual is no longer O(h)-limited
    assert fixed[8000]["rms_norm_per_h"] < 0.1 * legacy[8000]["rms_norm_per_h"]
    assert fixed[32000]["rms_norm_per_h"] < 0.6 * fixed[8000]["rms_norm_per_h"]
    # and the shipped-grid residual itself drops by more than an order of magnitude
    assert fixed[8000]["rms_norm"] < 0.05 * legacy[8000]["rms_norm"]


def test_check_kk_ito_preset_is_not_in_the_causal_class():
    """HONEST SCOPING (finding Q-5). The old gate lumped the ITO extended-Drude preset in with the
    causal models because the O(h) floor hid everything below ~3.5e-2. It does not belong there:
    its residual is 1.007e-2 at N = 8000 and 1.005e-2 at N = 32000 -- GRID-INDEPENDENT, i.e. a
    genuine ~1% causality violation (the omega^1.5 impurity crossover is non-analytic at
    omega = 0), comparable to a 2x jump in gamma. Small enough to be usable, too large to call
    causal; the docstrings now say so."""
    wmax, band = 80.0 * WP, _band()
    ext = ExtendedDrudeOptical(EPS_INF, M_OPT, gamma_ito_extended)
    r = {}
    for N in (8000, 32000):
        g = np.linspace(wmax / N, wmax, N)
        r[N] = check_kk(ext, g, n_m3=N_ITO, metric_band=band, self_calib=False)["rms_norm"]
    assert 5.0e-3 < r[8000] < 3.0e-2
    assert r[32000] == pytest.approx(r[8000], rel=0.05)        # does NOT fall with refinement


def test_check_kk_auto_band_runs():
    # the auto metric band (Re-zero-crossing dispersive window) must produce finite metrics.
    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=1.0e14)
    k = check_kk(drude, _kk_grid(), n_m3=N_ITO)
    assert np.isfinite(k["rms_norm"]) and np.isfinite(k["max_norm"])
    assert k["rms_norm"] < 1.0e-3                              # re-baselined (finding Q-5)


def test_check_kk_rejects_nonuniform_and_nonpositive_grid():
    ext = ExtendedDrudeOptical(EPS_INF, M_OPT, gamma_ito_extended)
    with pytest.raises(ValueError):
        check_kk(ext, np.array([1.0e14, 2.0e14, 4.0e14, 8.0e14] * 4), n_m3=N_ITO)  # non-uniform
    with pytest.raises(ValueError):
        check_kk(ext, np.linspace(0.0, 80.0 * WP, 8000), n_m3=N_ITO)               # omega=0 pole


def test_gamma_table_no_silent_extrapolation():
    w_tab = np.linspace(0.5e15, 3.0e15, 32)
    g_tab = gamma_ito_extended(w_tab)
    ext = ExtendedDrudeOptical(EPS_INF, M_OPT, gamma_omega=(w_tab, g_tab))
    # in-range: matches the callable preset to within 32-point linear-interp error (~1e-4).
    lam = _in_range_lambda(1.5e15)
    assert abs(complex(ext.eps(lam, n_m3=N_ITO)) -
               complex(ExtendedDrudeOptical(EPS_INF, M_OPT, gamma_ito_extended).eps(lam, n_m3=N_ITO))
               ) < 1e-3 * abs(EPS_INF)
    # out-of-range omega -> raises (no silent extrapolation).
    with pytest.raises(ValueError):
        ext.eps(_in_range_lambda(5.0e15), n_m3=N_ITO)


def _in_range_lambda(omega):
    return 2.0 * np.pi * C_LIGHT / omega


# ===========================================================================
# Gate 5 -- plugs into tmm_reference (3-layer stack) + exp(-iwt) sign convention.
# ===========================================================================
def test_extended_drude_in_tmm_three_layer_stack():
    from dynameta.optics.tmm_reference import stack_rta, _passive_sqrt

    ext = ExtendedDrudeOptical(EPS_INF, M_OPT, gamma_ito_extended)
    lam = 1550e-9
    eps_ito = complex(ext.eps(lam, n_m3=N_ITO))
    # exp(-i omega t): a passive/absorbing ITO film has Im(eps) > 0.
    assert eps_ito.imag > 0.0
    n_ito = _passive_sqrt(eps_ito)
    assert n_ito.imag >= 0.0                 # decaying wave (passive branch)

    # air | ITO(50 nm) | glass -- the eps() model plugs straight into the TMM oracle.
    layers = [(n_ito, 50e-9)]
    R, T, A = stack_rta(1.0, layers, 1.5, lam, theta_deg=0.0, pol="s")
    assert 0.0 <= R <= 1.0 and 0.0 <= T <= 1.0 and 0.0 <= A <= 1.0
    assert A > 0.0                           # a lossy ITO film absorbs
    # AUDIT T-1: `R + T + A == 1` is an IDENTITY (stack_rta returns A := 1 - R - T) and gated
    # nothing -- it passes for a halved or sign-flipped T. This film is LOSSY, so gate the whole
    # triple against the INDEPENDENT Abeles TMM: that is what actually pins the extended-Drude
    # eps -> n -> R/T/A chain this test is about.
    from _rta_oracles import abeles_rta
    R_ref, T_ref, A_ref = abeles_rta(1.0, layers, 1.5, lam, theta_deg=0.0, pol="s")
    assert R == pytest.approx(R_ref, abs=1e-12)
    assert T == pytest.approx(T_ref, abs=1e-12)
    assert A == pytest.approx(A_ref, abs=1e-12)

    # sanity vs plain Drude @ gamma_dc: the extended film absorbs LESS (reduced Im eps).
    n_plain = _passive_sqrt(complex(
        DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAMMA_DC).eps(lam, n_m3=N_ITO)))
    _, _, A_plain = stack_rta(1.0, [(n_plain, 50e-9)], 1.5, lam, theta_deg=0.0, pol="s")
    assert A < A_plain
