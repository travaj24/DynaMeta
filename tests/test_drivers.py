"""dynameta.drivers glue layer: every adapter vs the DIRECT underlying-model call (the
independent path), plus the None/raise contracts for missing upstream data. Pure numpy --
no DEVSIM/NGSolve."""
import math

import numpy as np
import pytest

from dynameta.drivers import (absorbed_fraction, contact_current_A,
                              contact_current_density_from_field,
                              em_mttf_from_carrier_field, lc_extra_fields, llg_extra_fields,
                              pcm_extra_fields, tddb_tbd_from_electrothermal,
                              tmm_absorption_by_layer_name)
from dynameta.reliability.em import EmParams, current_density_A_m2
from dynameta.reliability.tddb import TddbParams


class _FakeCF:
    def __init__(self, extras):
        self.extras = extras


class _FakeOpt:
    def __init__(self, A=None, A_independent=None, per_region_absorption=None):
        self.A = A
        self.A_independent = A_independent
        self.per_region_absorption = per_region_absorption


# ---- contact current -> EM ----

def test_contact_current_missing_key_is_none():
    assert contact_current_A(_FakeCF({}), "gate") is None
    assert contact_current_density_from_field(_FakeCF({}), "gate",
                                              width_m=1e-6, thickness_m=1e-7) is None


def test_contact_current_unknown_contact_raises():
    cf = _FakeCF({"contact_currents_A": {"top": 1e-6, "bot": -1e-6}})
    with pytest.raises(KeyError):
        contact_current_A(cf, "gate")


def test_em_mttf_matches_direct_call():
    I = 2.5e-3
    cf = _FakeCF({"contact_currents_A": {"top": -I, "bot": I}})
    w, t, T = 2e-6, 1e-7, 350.0
    params = EmParams.calibrated(J_A_m2=1.0e9, T_K=378.0, mttf_s=10.0 * 365 * 24 * 3600.0)
    J = current_density_A_m2(I, w, t)
    direct = params.mttf_s(J, T)
    via = em_mttf_from_carrier_field(cf, "top", width_m=w, thickness_m=t, T_K=T, params=params)
    assert via == pytest.approx(direct, rel=1e-12)
    assert em_mttf_from_carrier_field(_FakeCF({}), "top", width_m=w, thickness_m=t,
                                      T_K=T, params=params) is None


# ---- per-region absorption ----

def test_absorbed_fraction_total_prefers_independent():
    r = _FakeOpt(A=0.30, A_independent=0.31)
    assert absorbed_fraction(r) == pytest.approx(0.31)
    assert absorbed_fraction(_FakeOpt(A=0.30)) == pytest.approx(0.30)
    with pytest.raises(ValueError):
        absorbed_fraction(_FakeOpt())


def test_absorbed_fraction_region_and_drift_guard():
    r = _FakeOpt(A=0.3, per_region_absorption={"ito": 0.25, "oxide": 0.05})
    assert absorbed_fraction(r, "ito") == pytest.approx(0.25)
    with pytest.raises(KeyError):
        absorbed_fraction(r, "ITO_typo")
    with pytest.raises(ValueError):
        absorbed_fraction(_FakeOpt(A=0.3), "ito")


def test_tmm_rekey_uses_unique_layer_names_top_first():
    # keyed by LAYER name (unique by Stack contract; matches the FEM region labels and the
    # layer addressing of oxide_stress_from_electrothermal), NOT by material -- two layers
    # sharing a material must stay distinct, never silently summed.
    class _L:
        def __init__(self, name, mat):
            self.name = name
            self.background_material = mat

    class _Stack:
        layers = [_L("mirror", "metal"), _L("spacer_lo", "ito"), _L("spacer_hi", "ito")]

    class _Design:
        stack = _Stack()

    # slabs are top-first: slab_0 = spacer_hi, slab_1 = spacer_lo, slab_2 = mirror
    r = _FakeOpt(per_region_absorption={"slab_0": 0.10, "slab_1": 0.07, "slab_2": 0.02})
    by_name = tmm_absorption_by_layer_name(r, _Design())
    assert by_name == {"spacer_hi": pytest.approx(0.10), "spacer_lo": pytest.approx(0.07),
                       "mirror": pytest.approx(0.02)}
    with pytest.raises(ValueError):                        # FEM (name-keyed) map rejected
        tmm_absorption_by_layer_name(_FakeOpt(per_region_absorption={"ito": 0.1}), _Design())
    with pytest.raises(ValueError):                        # graded/sliced stack: no 1:1 map
        tmm_absorption_by_layer_name(
            _FakeOpt(per_region_absorption={"slab_0": 0.1, "slab_1": 0.1}), _Design())


# ---- electrothermal -> TDDB ----

def test_tddb_from_electrothermal_matches_direct():
    class _EL:
        def __init__(self, name):
            self.name = name

    class _ER:
        def mean_Ez_per_layer(self):
            return np.array([1e7, -4.2e8, 3e6])

    class _ET:
        layers = [_EL("metal"), _EL("hfo2"), _EL("ito")]
        E_result = _ER()
        T_per_layer = np.array([320.0, 335.0, 350.0])

    params = TddbParams.calibrated(E_ox_V_m=5e8, T_K=398.0, tbd_s=1.0e4)
    direct = params.tbd_s(4.2e8, 335.0)
    assert tddb_tbd_from_electrothermal(_ET(), "hfo2", params) == pytest.approx(direct, rel=1e-12)


# ---- LLG -> m_vector ----

def test_llg_extra_fields_settles_to_field_axis():
    from dynameta.carriers.llg import LLGMacrospin
    spin = LLGMacrospin(Ms_A_m=8.0e5, alpha=0.5)
    fields = llg_extra_fields(spin, lambda b: np.array([0.0, 0.0, float(b)]),
                              t_settle_s=5e-9, m0=[0.6, 0.0, 0.8])
    out = fields(1.0e5)                                    # H = 1e5 A/m along +z
    m = out["m_vector"]
    assert set(out) == {"m_vector"} and m.shape == (3,)
    assert m[2] == pytest.approx(1.0, abs=1e-3)            # relaxed onto the field axis
    assert np.linalg.norm(m) == pytest.approx(1.0, abs=1e-9)


def test_llg_extra_fields_guards():
    from dynameta.carriers.llg import LLGMacrospin
    with pytest.raises(ValueError):                        # undamped spin never settles
        llg_extra_fields(LLGMacrospin(Ms_A_m=8e5, alpha=0.0), lambda b: np.zeros(3),
                         t_settle_s=1e-9, m0=[0, 0, 1])
    spin = LLGMacrospin(Ms_A_m=8e5, alpha=0.01)
    fields = llg_extra_fields(spin, lambda b: np.array([1e5, 0.0, 0.0]),
                              t_settle_s=1e-12, m0=[0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError):                      # far too short to settle
        fields(0.0)


# ---- PCM -> crystalline_fraction ----

def test_pcm_extra_fields_matches_direct_integrate():
    from dynameta.carriers.switching import PCMSwitching
    sw = PCMSwitching(K0_per_s=1.0e13, E_a_J=2.0e-19, T_glass_K=450.0, T_melt_K=900.0)
    t = np.linspace(0.0, 1e-6, 400)
    T = np.full_like(t, 650.0)                             # anneal between Tg and Tm
    fields = pcm_extra_fields(sw, lambda b: (t, T))
    out = fields("b0")
    direct = float(sw.integrate(t, T, x0=0.0)[-1])
    assert set(out) == {"crystalline_fraction"}
    assert out["crystalline_fraction"] == pytest.approx(direct, rel=1e-12)
    assert 0.0 < out["crystalline_fraction"] <= 1.0
    # melt pulse at the end resets toward amorphous
    T_melt = T.copy(); T_melt[-40:] = 950.0
    out2 = pcm_extra_fields(sw, lambda b: (t, T_melt))("b0")
    assert out2["crystalline_fraction"] < 1e-6


# ---- LC director -> director_angle_rad ----

LC_KW = dict(K11=6.2e-12, K33=8.3e-12, eps_para=19.0, eps_perp=5.2, d_planar=2.0e-6)


def test_lc_extra_fields_below_threshold_is_planar():
    fields = lc_extra_fields(lambda b: float(b), reduce="midplane", **LC_KW)
    out = fields(0.0)                                      # V = 0 << V_th
    assert set(out) == {"director_angle_rad"}
    # plate-plane angle ~ pi/2 - theta_b ~ 0 (planar) below threshold
    assert abs(out["director_angle_rad"]) < math.radians(1.0)


def test_lc_extra_fields_above_threshold_tilts_and_matches_direct():
    from dynameta.carriers.lc_director import director_profile_bvp, director_to_extra_fields
    V = 3.0
    fields = lc_extra_fields(lambda b: V, reduce="midplane", **LC_KW)
    out = fields("bias")
    res = director_profile_bvp(V_app=V, **LC_KW)
    direct = director_to_extra_fields(res.theta_field_rad[res.theta_field_rad.size // 2])
    assert out["director_angle_rad"] == pytest.approx(direct["director_angle_rad"], rel=1e-9)
    assert out["director_angle_rad"] > math.radians(20.0)  # well-tilted toward homeotropic


def test_lc_extra_fields_profile_mode_shape():
    fields = lc_extra_fields(lambda b: 0.5, reduce="profile", nz=101, **LC_KW)
    th = fields(0)["director_angle_rad"]
    assert np.asarray(th).shape == (101,)
