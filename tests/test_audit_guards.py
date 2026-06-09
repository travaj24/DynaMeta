"""Coverage + guard tests added from the 2026-06-09 exhaustive audit (docs/audit/). These lock in
defensive guards (against pathological callable returns / unphysical inputs) AND confirm behaviors the
audit flagged as untested -- notably that SeparableXYLift PRESERVES SIGN for a pure-depletion profile
(the audit's CRIT-1 "sign-inversion bug" was a false positive; this test proves the code is correct).
Pure numpy (no devsim/ngsolve)."""
import numpy as np
import pytest

from dynameta.constants import M_E
from dynameta.materials.optical_model import DrudeOptical, TabulatedOptical
from dynameta.materials.scattering import MatthiessenGamma, ScatteringModel
from dynameta.geometry.unit_cell import UnitCell
from dynameta.geometry.stack import Feature
from dynameta.core.lift import SeparableXYLift


# ---- A. defensive guards (fail loudly instead of silent inf/NaN/gain) ----

def test_drude_nonpositive_mass_raises():
    bad = DrudeOptical(eps_inf=4.0, m_opt_kg=(lambda n: np.zeros_like(np.asarray(n, float))),
                       gamma_rad_s=1.0e14)
    with pytest.raises(ValueError):
        bad.eps(1300e-9, n_m3=1e27)
    ok = DrudeOptical(eps_inf=4.0, m_opt_kg=0.35 * M_E, gamma_rad_s=1.0e14)   # valid path unaffected
    assert np.isfinite(complex(ok.eps(1300e-9, n_m3=1e27)))


def test_matthiessen_nonpositive_temperature_raises():
    with pytest.raises(ValueError):
        MatthiessenGamma(gamma_const_rad_s=1.0e14, T_K=0.0)
    with pytest.raises(ValueError):
        MatthiessenGamma(gamma_const_rad_s=1.0e14, T_K=-5.0)
    assert MatthiessenGamma(gamma_const_rad_s=1.0e14, T_K=300.0)(1e27) > 0.0   # valid still works


def test_scattering_mobility_nonpositive_tau_raises():
    sm = ScatteringModel(one_over_tau=0.0, m_cond_kg=0.35 * M_E)
    with pytest.raises(ValueError):
        sm.mobility_of_n()(1e27)


def test_tabulated_out_of_range_raises():
    t = TabulatedOptical(lambda_m=np.array([1200e-9, 1400e-9]),
                         eps_complex=np.array([4.0 + 0.1j, 4.1 + 0.1j]))
    assert np.isfinite(complex(t.eps(1300e-9)))              # in-range OK
    with pytest.raises(ValueError):
        t.eps(1600e-9)                                       # out-of-range -> no silent extrapolation


def test_unit_cell_nonpositive_period_raises():
    with pytest.raises(ValueError):
        UnitCell(period_x_m=-1e-9, period_y_m=300e-9)
    with pytest.raises(ValueError):
        UnitCell(period_x_m=300e-9, period_y_m=0.0)


def test_feature_inverted_zrange_raises():
    with pytest.raises(ValueError):
        Feature(name="via", shape=None, material="metal", z_lo_m=100e-9, z_hi_m=50e-9)
    assert Feature(name="ok", shape=None, material="m", z_lo_m=0.0, z_hi_m=10e-9)   # valid


# ---- B. SeparableXYLift depletion: sign IS preserved (refutes audit CRIT-1) ----

def test_separable_xy_lift_depletion_preserves_sign():
    N_BG, P, Nx, Nv = 4.0e26, 300e-9, 21, 3
    x = np.linspace(0.0, P, Nx)
    prof = -1.0e26 * np.exp(-((x - 0.5 * P) / (0.15 * P)) ** 2)          # PURE depletion (negative)
    n_2d = N_BG + prof[:, None] * np.ones((1, Nv))                       # (Nx, Nv), single-sign deviation
    lift = SeparableXYLift(period_y_m=P, ny=64)
    n_3d, _x, _y, _v = lift.apply(n_2d, x, np.arange(Nv, dtype=float), n_bg=N_BG)
    assert np.all(n_3d <= N_BG * (1.0 + 1e-9))             # no spurious accumulation -> sign preserved
    assert float(np.min(n_3d)) < N_BG * (1.0 - 1e-6)       # real depletion is present


def test_weighted_objective_both_keys_raises():
    pytest.importorskip("jax")
    from dynameta.optics.inverse_design import weighted_objective
    loss = weighted_objective([{"value": (lambda p: p), "target": 1.0, "sense": "max"}])
    with pytest.raises(ValueError):
        loss(0.5)
