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


# ---- audit-v2 additions ----------------------------------------------------------------------

def test_drude_negative_gamma_raises_zero_allowed():
    # gamma < 0 = gain under exp(-iwt) -> reject; gamma = 0 (collisionless idealization) stays legal.
    bad = DrudeOptical(eps_inf=4.0, m_opt_kg=0.35 * M_E, gamma_rad_s=(lambda n: -1.0e14))
    with pytest.raises(ValueError):
        bad.eps(1300e-9, n_m3=1e27)
    lossless = DrudeOptical(eps_inf=4.0, m_opt_kg=0.35 * M_E, gamma_rad_s=0.0)
    e = complex(lossless.eps(1300e-9, n_m3=1e27))
    assert e.imag == 0.0 and np.isfinite(e.real)


def test_matthiessen_nonpositive_dc_ratio_raises():
    with pytest.raises(ValueError):
        MatthiessenGamma(gamma_const_rad_s=1.0e14, optical_dc_ratio=0.0)
    with pytest.raises(ValueError):
        MatthiessenGamma(gamma_const_rad_s=1.0e14, optical_dc_ratio=-1.0)


def test_sweepresults_wavelength_collision_raises():
    from types import SimpleNamespace
    from dynameta.results import SweepResults
    rows = [SimpleNamespace(bias_label="b", lambda_nm=w, result=SimpleNamespace(R=0.5))
            for w in (1300.0, 1300.0 + 5e-8)]            # distinct but within the 1e-6 nm key rounding
    with pytest.raises(ValueError):
        SweepResults.from_rows(rows)
    ok = SweepResults.from_rows([SimpleNamespace(bias_label="b", lambda_nm=w,
                                                 result=SimpleNamespace(R=0.5))
                                 for w in (1300.0, 1310.0)])
    assert ok.wavelengths_nm.size == 2                   # normal sweeps unaffected


def test_sweep_duplicate_bias_labels_raise():
    # audit C6-3: duplicate labels silently collapsed to the last point
    import pytest
    from dynameta.sweep import BiasPoint, Sweep
    with pytest.raises(ValueError, match="duplicate bias-point labels"):
        Sweep(bias_points=[BiasPoint({"g": 0.0}, "on"), BiasPoint({"g": 1.0}, "on")],
              wavelengths_nm=[1300.0])
    Sweep(bias_points=[BiasPoint({"g": 0.0}, "off"), BiasPoint({"g": 1.0}, "on")],
          wavelengths_nm=[1300.0])                             # distinct labels fine


def test_dispersion_check_catches_inband_feature():
    # audit C5-12: equal band-edge n with an in-band resonance must DISABLE the fast path
    # (the old two-edge check false-passed and froze a wrong band-centre index)
    import warnings
    import numpy as np
    import pytest
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical, TabulatedOptical
    from dynameta.optics.tmm_reference import end_media_indices
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("m", ConstantOptical(4.0 + 0j)))
    reg.add(Material("bump", TabulatedOptical(lambda_m=np.array([1.20e-6, 1.325e-6, 1.45e-6]),
                                              eps_complex=np.array([4.0 + 0j, 6.0 + 0j, 4.0 + 0j]))))
    d = Design(name="t", unit_cell=UnitCell.square(300e-9),
               stack=Stack(layers=[Layer("s", 100e-9, "m")],
                           superstrate_material="air", substrate_material="bump"),
               electrodes=[], materials=reg)
    lams = [1.20e-6, 1.325e-6, 1.45e-6]
    # edges are equal (the old check's blind spot)...
    ns_lo, nb_lo = end_media_indices(d, lams[0])
    ns_hi, nb_hi = end_media_indices(d, lams[-1])
    assert abs(nb_lo - nb_hi) < 1e-12
    # ...but the band-centre (freeze point) differs -- the NEW check must flag it
    _, nb_c = end_media_indices(d, 0.5 * (lams[0] + lams[-1]))
    assert abs(nb_c - nb_lo) > 0.1
