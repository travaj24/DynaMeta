"""Roadmap 2.3 -- Mermin / extended-Drude damping (frequency-dependent scattering).

Gates:
  * gamma(omega)->const == DrudeOptical byte-identical (1e-15), scalar and callable mass;
  * the analytic k->0 Mermin == plain Drude claim, verified NUMERICALLY against a finite-k
    hydrodynamic Lindhard chi (residual ~ (beta k / w)^2 -> 0), plus the local
    MerminDrudeOptical == ExtendedDrudeOptical identity and the finite-k deferral;
  * the ITO extended-Drude preset vs a plain Drude fixed at gamma_dc: absorption (Im eps)
    REDUCED in the near-IR/below-plasma window (direction pinned);
  * check_kk residual SMALL for causal models (Drude, ITO preset) and LARGE for a
    deliberately acausal step gamma -- the diagnostic discriminates;
  * plugs into tmm_reference for a 3-layer stack (R, T, A finite; energy budget holds);
  * exp(-i omega t) sign convention (Im eps > 0 where absorbing).

Independent oracle for the k->0 claim: a self-contained finite-k Mermin dielectric built from
the collisionless hydrodynamic Lindhard function, evaluated in this test only (the library never
computes finite-k Mermin -- that is roadmap 2.4). Mermin, Phys. Rev. B 1, 2362 (1970).
"""
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


def test_check_kk_causal_small_acausal_large():
    grid, band = _kk_grid(), _band()

    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=1.0e14)
    ext = ExtendedDrudeOptical(EPS_INF, M_OPT, gamma_ito_extended)
    # deliberately ACAUSAL: a step in gamma(omega) -> discontinuous eps, cannot satisfy KK.
    step = ExtendedDrudeOptical(EPS_INF, M_OPT,
                                lambda w: np.where(np.asarray(w) < 1.2e15, 3.0e13, 3.0e14))

    kd = check_kk(drude, grid, n_m3=N_ITO, metric_band=band)
    ke = check_kk(ext, grid, n_m3=N_ITO, metric_band=band)
    ks = check_kk(step, grid, n_m3=N_ITO, metric_band=band)

    # causal models: small normalized residual.
    assert kd["rms_norm"] < 0.035 and kd["max_norm"] < 0.25
    assert ke["rms_norm"] < 0.035
    # acausal step: clearly larger, both in RMS and in the localized max spike at the jump.
    assert ks["rms_norm"] > 0.035
    assert ks["max_norm"] > 0.30
    # DISCRIMINATION: the step residual is several-fold above the causal floor.
    assert ks["rms_norm"] > 2.5 * kd["rms_norm"]
    assert ks["max_norm"] > 2.0 * kd["max_norm"]


def test_check_kk_auto_band_runs():
    # the auto metric band (Re-zero-crossing dispersive window) must produce finite metrics.
    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=1.0e14)
    k = check_kk(drude, _kk_grid(), n_m3=N_ITO)
    assert np.isfinite(k["rms_norm"]) and np.isfinite(k["max_norm"])
    assert k["rms_norm"] < 0.05


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
    R, T, A = stack_rta(1.0, [(n_ito, 50e-9)], 1.5, lam, theta_deg=0.0, pol="s")
    assert 0.0 <= R <= 1.0 and 0.0 <= T <= 1.0 and 0.0 <= A <= 1.0
    assert A > 0.0                           # a lossy ITO film absorbs
    assert abs(R + T + A - 1.0) < 1e-9       # energy budget holds (stack_rta also guards it)

    # sanity vs plain Drude @ gamma_dc: the extended film absorbs LESS (reduced Im eps).
    n_plain = _passive_sqrt(complex(
        DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAMMA_DC).eps(lam, n_m3=N_ITO)))
    _, _, A_plain = stack_rta(1.0, [(n_plain, 50e-9)], 1.5, lam, theta_deg=0.0, pol="s")
    assert A < A_plain
