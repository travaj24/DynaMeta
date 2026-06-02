"""Unit coverage for the EffectModel field-bundle seam (core.effects) -- the v0.3 keystone
that generalizes the scalar n->eps map to fields {n,E,T,...} -> scalar OR tensor eps. Pure numpy.
Run: python -m pytest tests/test_effects.py -q
"""
import numpy as np
import pytest

from dynameta.core.effects import (OpticalModelEffect, ComposedEffect, DeltaEffect, as_tensor,
                                   PockelsEffect, KerrEffect, FranzKeldyshEffect, ThermoOpticModel,
                                   MagnetoOpticModel)
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


def test_delta_effect_prevents_background_double_count():
    # The bundled effects return ABSOLUTE eps (own background + shift). Composing one DIRECTLY as a
    # delta double-counts the background; DeltaEffect (subtract zero-drive baseline) fixes it.
    no, ne, r13, r33, eps_bg, r = _linbo3()
    pk = PockelsEffect(eps_bg=eps_bg, r_voigt=r)
    base = OpticalModelEffect(ConstantOptical(complex(4.0, 0.0)))          # eps_bg = 4*I background
    Ez = 1.0e7
    # WRONG: composing the absolute-eps Pockels directly adds eps_bg on top of the 4.0 background
    naive = ComposedEffect(background=base, deltas=[pk]).eps({"E": np.zeros(3)}, 1300e-9)
    assert np.allclose(np.diag(naive), [4.0 + no ** 2, 4.0 + no ** 2, 4.0 + ne ** 2])  # bg twice
    # RIGHT: wrap in DeltaEffect with a zero-field baseline -> only the field-induced shift adds
    comp = ComposedEffect(background=base, deltas=[DeltaEffect(pk, {"E": np.zeros(3)})])
    assert np.allclose(comp.eps({"E": np.zeros(3)}, 1300e-9), 4.0 * np.eye(3))   # E=0 -> just bg
    shift = as_tensor(pk.eps({"E": [0, 0, Ez]}, 1300e-9)) - as_tensor(eps_bg)
    assert np.allclose(comp.eps({"E": np.array([0.0, 0.0, Ez])}, 1300e-9), 4.0 * np.eye(3) + shift)


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


# ---- field-effect electro-optic EffectModels (Phase 1) ----
def _linbo3():
    no, ne, r13, r33 = 2.21, 2.14, 9.6e-12, 30.9e-12
    eps_bg = np.diag([no ** 2, no ** 2, ne ** 2]).astype(complex)
    r = np.zeros((6, 3)); r[0, 2] = r13; r[1, 2] = r13; r[2, 2] = r33
    return no, ne, r13, r33, eps_bg, r


def test_pockels_reduces_to_background_and_shifts_index():
    no, ne, r13, r33, eps_bg, r = _linbo3()
    pk = PockelsEffect(eps_bg=eps_bg, r_voigt=r)
    assert np.allclose(pk.eps({"E": np.zeros(3)}, 1300e-9), eps_bg)          # E=0 -> background
    Ez = 1.0e7
    eps = pk.eps({"E": np.array([0.0, 0.0, Ez])}, 1300e-9)
    assert abs(eps[0, 1]) < 1e-12 and abs(eps[0, 2]) < 1e-12                 # stays diagonal
    nx, nz = np.sqrt(eps[0, 0].real), np.sqrt(eps[2, 2].real)
    assert nx == pytest.approx(no - 0.5 * no ** 3 * r13 * Ez, rel=1e-3)      # Pockels: n_o via r13
    assert nz == pytest.approx(ne - 0.5 * ne ** 3 * r33 * Ez, rel=1e-3)      # Pockels: n_e via r33
    pk0 = PockelsEffect(eps_bg=eps_bg, r_voigt=np.zeros((6, 3)))             # r=0 -> background
    assert np.allclose(pk0.eps({"E": np.array([0.0, 0.0, Ez])}, 1300e-9), eps_bg)


def test_kerr_reduces_and_is_quadratic_in_field():
    eps_bg = (2.0 ** 2) * np.eye(3, dtype=complex)
    kr = KerrEffect(eps_bg=eps_bg, s_kerr=1e-18)
    assert np.allclose(kr.eps({"E": np.zeros(3)}, 1300e-9), eps_bg)
    dB1 = 1.0 / kr.eps({"E": [0, 0, 1e8]}, 1300e-9)[0, 0] - 1.0 / eps_bg[0, 0]
    dB2 = 1.0 / kr.eps({"E": [0, 0, 2e8]}, 1300e-9)[0, 0] - 1.0 / eps_bg[0, 0]
    assert (dB2 / dB1).real == pytest.approx(4.0, rel=1e-6)                  # |E|^2 -> 2x field, 4x shift
    assert np.allclose(KerrEffect(eps_bg, 0.0).eps({"E": [0, 0, 1e8]}, 1300e-9), eps_bg)


def test_franz_keldysh_opens_field_absorption():
    fk = FranzKeldyshEffect(eps_bg=complex(12.0, 0.1), beta=1e-8)
    assert complex(fk.eps({"E": np.zeros(3)}, 1300e-9)) == pytest.approx(complex(12.0, 0.1))
    e = complex(fk.eps({"E": [0, 0, 1e6]}, 1300e-9))
    assert e.imag > 0.1 and e.real == pytest.approx(12.0)                    # field-on -> more loss


def test_thermo_optic_reduces_and_shifts_index():
    n0 = 3.48                                                                # Si
    m = ThermoOpticModel(eps_ref=complex(n0 ** 2, 0.0), dn_dT=1.8e-4, T_ref=300.0)
    assert complex(m.eps({"T": 300.0}, 1300e-9)) == pytest.approx(complex(n0 ** 2, 0.0))  # T=T_ref
    e = complex(m.eps({"T": 350.0}, 1300e-9))
    assert np.sqrt(e.real) == pytest.approx(n0 + 1.8e-4 * 50.0, rel=1e-9)    # n(T) = n0 + dn/dT*dT
    m0 = ThermoOpticModel(eps_ref=complex(n0 ** 2, 0.0), dn_dT=0.0)
    assert complex(m0.eps({"T": 500.0}, 1300e-9)) == pytest.approx(complex(n0 ** 2, 0.0))  # dn/dT=0
    with pytest.raises(ValueError):
        m.eps({}, 1300e-9)                                                   # T required


def test_magneto_optic_gyrotropic_tensor():
    eps_r, g = 2.25, 0.05
    mo = MagnetoOpticModel(eps_r=eps_r, g=g)
    T = np.asarray(mo.eps({}, 1550e-9))                                       # default magnetization=1
    assert T.shape == (3, 3)
    # gyrotropic structure: diagonal eps_r, off-diagonal +/- i g, Hermitian
    assert np.allclose(np.diag(T), eps_r)
    assert T[0, 1] == pytest.approx(1j * g) and T[1, 0] == pytest.approx(-1j * g)
    assert T[0, 2] == 0 and T[2, 0] == 0 and T[1, 2] == 0
    assert np.allclose(T, T.conj().T)                                        # Hermitian -> lossless
    # circular + axial eigenvalues {eps_r - g, eps_r, eps_r + g}
    assert np.allclose(np.sort(np.linalg.eigvals(T).real), [eps_r - g, eps_r, eps_r + g])
    # magnetization scales/flips g; 0 reduces to isotropic eps_r*I
    assert np.asarray(mo.eps({"magnetization": -1.0}, 1550e-9))[0, 1] == pytest.approx(-1j * g)
    assert np.allclose(np.asarray(mo.eps({"magnetization": 0.0}, 1550e-9)), eps_r * np.eye(3))
    assert np.allclose(np.asarray(MagnetoOpticModel(eps_r, 0.0).eps({}, 1550e-9)), eps_r * np.eye(3))
