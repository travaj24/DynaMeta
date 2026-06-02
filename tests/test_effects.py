"""Unit coverage for the EffectModel field-bundle seam (core.effects) -- the v0.3 keystone
that generalizes the scalar n->eps map to fields {n,E,T,...} -> scalar OR tensor eps. Pure numpy.
Run: python -m pytest tests/test_effects.py -q
"""
import numpy as np
import pytest

from dynameta.core.effects import OpticalModelEffect, ComposedEffect, as_tensor
from dynameta.materials.optical_model import ConstantOptical


def test_optical_model_effect_matches_scalar_model():
    om = ConstantOptical(complex(2.25, 0.3))
    eff = OpticalModelEffect(om)
    # density-independent: the field bundle is ignored, result == the bare OpticalModel
    assert eff.eps({}, 1300e-9) == pytest.approx(om.eps(1300e-9))
    assert eff.eps({"n": None}, 1300e-9) == pytest.approx(om.eps(1300e-9))


def test_optical_model_effect_forwards_density():
    # the adapter must forward fields['n'] to the model's n_m3 unchanged (the carrier path)
    class _FakeOptical:
        def eps(self, lambda_m, *, n_m3=None):
            return np.asarray(n_m3, dtype=complex) * 1e-27 + 1.0
    n = np.array([4e26, 5e26])
    eff = OpticalModelEffect(_FakeOptical())
    assert np.allclose(eff.eps({"n": n}, 1300e-9), _FakeOptical().eps(1300e-9, n_m3=n))


def test_as_tensor_promotes_scalar_to_isotropic():
    assert np.allclose(as_tensor(2.0 + 0j), 2.0 * np.eye(3))
    g = np.array([2.0, 3.0], dtype=complex)              # grid -> (...,3,3)
    tg = as_tensor(g)
    assert tg.shape == (2, 3, 3) and np.allclose(tg[1], 3.0 * np.eye(3))
    m = (np.arange(9).reshape(3, 3) + 1).astype(complex)  # already a tensor -> unchanged
    assert np.allclose(as_tensor(m), m)


def test_composed_effect_sums_background_plus_deltas_as_tensors():
    class _Const:                                         # a trivial EffectModel
        def __init__(self, v): self.v = v
        def eps(self, fields, lambda_m): return self.v
    comp = ComposedEffect(background=_Const(4.0 + 0j), deltas=[_Const(0.1 + 0j), _Const(0.05 + 0j)])
    out = comp.eps({}, 1300e-9)
    assert out.shape == (3, 3) and np.allclose(out, 4.15 * np.eye(3))


def test_eps_field_tensor_flags():
    from dynameta.core.eps_field import EpsField
    ax = np.array([0.0, 1.0])
    assert EpsField(scalar=4 + 0j).is_uniform and not EpsField(scalar=4 + 0j).is_tensor
    ut = EpsField(tensor=4.0 * np.eye(3, dtype=complex))
    assert ut.is_uniform and ut.is_tensor                                    # uniform tensor
    gs = EpsField(x_axis_u=ax, y_axis_u=ax, z_axis_u=ax, values_zyx=np.ones((2, 2, 2), complex))
    assert not gs.is_uniform and not gs.is_tensor                            # graded scalar
    gt = EpsField(x_axis_u=ax, y_axis_u=ax, z_axis_u=ax, values_zyx=np.ones((2, 2, 2, 3, 3), complex))
    assert not gt.is_uniform and gt.is_tensor                                # graded tensor
