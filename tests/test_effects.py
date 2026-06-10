"""Unit coverage for the EffectModel field-bundle seam (core.effects) -- the v0.3 keystone
that generalizes the scalar n->eps map to fields {n,E,T,...} -> scalar OR tensor eps. Pure numpy.
Run: python -m pytest tests/test_effects.py -q
"""
import numpy as np
import pytest

from dynameta.core.effects import (OpticalModelEffect, ComposedEffect, DeltaEffect, as_tensor,
                                   PockelsEffect, KerrEffect, FranzKeldyshEffect, ThermoOpticModel,
                                   MagnetoOpticModel, AnisotropicThermoOpticModel, IntersubbandEffect)
from dynameta.materials.optical_model import ConstantOptical, DrudeOptical
from dynameta.constants import M_E, HBAR
from dynameta.carriers.schrodinger_poisson import SubbandResult


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


def test_anisotropic_thermo_optic_diagonal_and_reduces_to_scalar():
    no, ne = 2.20, 2.30
    dno, dne = 1.0e-4, 3.0e-4
    m = AnisotropicThermoOpticModel(eps_ref_diag=(no ** 2, no ** 2, ne ** 2),
                                    dn_dT_diag=(dno, dno, dne), T_ref=300.0)
    T0 = np.asarray(m.eps({"T": 300.0}, 1550e-9))                    # at T_ref -> diag(eps_ref)
    assert T0.shape == (3, 3) and np.allclose(np.diag(T0), [no ** 2, no ** 2, ne ** 2])
    assert np.allclose(T0 - np.diag(np.diag(T0)), 0)                 # strictly diagonal
    dT = 50.0
    Tt = np.asarray(m.eps({"T": 300.0 + dT}, 1550e-9))
    assert np.sqrt(Tt[0, 0].real) == pytest.approx(no + dno * dT, rel=1e-9)   # ordinary axis
    assert np.sqrt(Tt[2, 2].real) == pytest.approx(ne + dne * dT, rel=1e-9)   # extraordinary axis
    # isotropic dn/dT reduces EXACTLY to the scalar ThermoOpticModel * I
    iso = AnisotropicThermoOpticModel((no ** 2,) * 3, (dno,) * 3)
    sca = ThermoOpticModel(eps_ref=complex(no ** 2, 0.0), dn_dT=dno)
    assert np.allclose(np.asarray(iso.eps({"T": 360.0}, 1550e-9)),
                       complex(sca.eps({"T": 360.0}, 1550e-9)) * np.eye(3))
    with pytest.raises(ValueError):
        m.eps({}, 1550e-9)                                          # T required


# ---- IntersubbandEffect (R7) ----------------------------------------------------------------

EPS_INF, M_OPT, GAM_INTRA, GAM_INTER = 4.25, 0.225 * M_E, 1.1e14, 1.0e13


def _one_band_subband():
    L = 3e-9
    z = np.linspace(0.0, L, 120)
    psi = np.sin(np.pi * z / L)[:, None]
    psi = psi / np.sqrt(np.sum(psi[:, 0] ** 2) * (z[1] - z[0]))
    return SubbandResult(energies_J=np.array([1e-21]), psi=psi, z_m=z,
                         sheet_density_m2=np.array([4.0e17]))


def test_intersubband_requires_subband_field():
    m = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER)
    with pytest.raises(ValueError):
        m.eps({}, 1300e-9)


def test_intersubband_single_band_reduces_to_drude():
    res = _one_band_subband()
    m = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER)
    eps_t = m.eps({"subband": res}, 1300e-9)
    n3d = float(res.sheet_density_m2.sum()) / (res.z_m[-1] - res.z_m[0])
    d = complex(DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAM_INTRA).eps(1300e-9, n_m3=n3d))
    assert eps_t.shape == (3, 3)
    assert np.allclose(eps_t, as_tensor(np.asarray(d)), atol=1e-13)   # isotropic == Drude*I
    assert abs(eps_t[0, 1]) == 0.0 and abs(eps_t[2, 2] - eps_t[0, 0]) < 1e-13


def test_intersubband_two_bands_anisotropic_and_passive():
    # synthetic two-band well: eps_zz != eps_xx and Im(eps_zz) > 0 (absorptive line)
    L = 2e-9
    z = np.linspace(0.0, L, 160)
    p1 = np.sin(np.pi * z / L); p2 = np.sin(2 * np.pi * z / L)
    h = z[1] - z[0]
    p1 = p1 / np.sqrt(np.sum(p1 ** 2) * h); p2 = p2 / np.sqrt(np.sum(p2 ** 2) * h)
    psi = np.stack([p1, p2], axis=1)
    E = np.array([0.0, 0.95 * 1.602e-19])                            # ~0.95 eV spacing
    res = SubbandResult(energies_J=E, psi=psi, z_m=z, sheet_density_m2=np.array([5e17, 1e17]))
    m = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER)
    lam12 = 2 * np.pi * 299792458.0 / ((E[1] - E[0]) / HBAR)
    eps_t = m.eps({"subband": res}, lam12)
    assert abs(eps_t[2, 2] - eps_t[0, 0]) > 1e-3                      # z carries the intersubband line
    assert eps_t[2, 2].imag > 0.0                                    # passive (exp(-iwt))
    assert eps_t[0, 0] == eps_t[1, 1]                                # in-plane isotropic Drude


# ---- BursteinMossEdge (R8) ------------------------------------------------------------------

def test_burstein_moss_reduces_and_off_switch():
    from dynameta.constants import Q_E
    eps_inf, m_opt, gamma = 4.25, 0.225 * M_E, 1.1e14
    n_ref = 4.0e26
    drude = DrudeOptical(eps_inf=eps_inf, m_opt_kg=m_opt, gamma_rad_s=gamma)
    bg = OpticalModelEffect(drude)
    from dynameta.core.effects import BursteinMossEdge
    edge = BursteinMossEdge(eps_inf=eps_inf, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5)
    comp = ComposedEffect(background=bg, deltas=[DeltaEffect(edge, baseline_fields={"n": n_ref})])
    # at n_ref the delta is identically zero -> bare Drude
    assert np.allclose(comp.eps({"n": n_ref}, 1300e-9),
                       as_tensor(np.asarray(complex(drude.eps(1300e-9, n_m3=n_ref)))), atol=1e-12)
    # enabled=False -> bare Drude at ALL n (true off-switch)
    off = ComposedEffect(background=bg, deltas=[DeltaEffect(
        BursteinMossEdge(eps_inf=eps_inf, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5,
                         enabled=False), baseline_fields={"n": n_ref})])
    for n in (2e26, 1e27):
        assert np.allclose(off.eps({"n": n}, 1300e-9),
                           as_tensor(np.asarray(complex(drude.eps(1300e-9, n_m3=n)))), atol=1e-14)


def test_burstein_moss_blueshift_and_passive():
    from dynameta.constants import Q_E
    from dynameta.core.effects import BursteinMossEdge
    edge = BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5)
    # Burstein-Moss: higher density -> larger optical gap (blueshift)
    assert edge.optical_gap_J(8e26) > edge.optical_gap_J(4e26) > edge.optical_gap_J(1e26)
    # above-gap Im(eps) >= 0 (passive, exp(-iwt)); probe at a photon energy above Eg_opt
    lam_above = 2 * np.pi * 299792458.0 * 1.0546e-34 / (4.5 * Q_E)   # ~4.5 eV photon (> gap)
    assert edge.eps({"n": 6e26}, lam_above).imag >= 0.0


def test_burstein_moss_requires_density():
    from dynameta.constants import Q_E
    from dynameta.core.effects import BursteinMossEdge
    edge = BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5)
    with pytest.raises(ValueError):
        edge.eps({}, 1300e-9)


# ---- VectorMagnetoOpticModel (R13) -----------------------------------------------------------

def test_vector_mo_reduces_to_z_axis_model():
    from dynameta.core.effects import VectorMagnetoOpticModel
    vm = VectorMagnetoOpticModel(eps_r=2.25, g_s=0.05)
    Tz = np.asarray(vm.eps({"m_vector": np.array([0.0, 0.0, 1.0])}, 1550e-9))
    T0 = np.asarray(MagnetoOpticModel(eps_r=2.25, g=0.05).eps({"magnetization": 1.0}, 1550e-9))
    assert np.array_equal(Tz, T0)                            # exact anchor to the validated z model


def test_vector_mo_x_axis_structure_and_guards():
    from dynameta.core.effects import VectorMagnetoOpticModel
    vm = VectorMagnetoOpticModel(eps_r=2.25, g_s=0.05)
    T = np.asarray(vm.eps({"m_vector": np.array([1.0, 0.0, 0.0])}, 1550e-9))
    assert T[1, 2] == pytest.approx(1j * 0.05) and T[2, 1] == pytest.approx(-1j * 0.05)
    assert T[0, 1] == 0.0 and np.allclose(np.diag(T), 2.25)  # x-gyration couples y<->z only
    assert np.allclose(T, T.conj().T)                        # Hermitian (lossless)
    with pytest.raises(ValueError):
        vm.eps({}, 1550e-9)                                  # m_vector required
    with pytest.raises(ValueError):
        vm.eps({"m_vector": np.array([1.0, 0.0])}, 1550e-9)  # trailing axis must be 3


# ---- LLG macrospin (R11) + the R13 seam ------------------------------------------------------

def test_llg_precession_and_seam_into_vector_mo():
    from dynameta.constants import MU0
    from dynameta.carriers.llg import LLGMacrospin, GAMMA_ELECTRON_RAD_ST
    from dynameta.core.effects import VectorMagnetoOpticModel
    H0 = 1.0e4
    llg = LLGMacrospin(Ms_A_m=1e5, alpha=0.0, H_applied_A_m=lambda t: np.array([0.0, 0.0, H0]))
    w = GAMMA_ELECTRON_RAD_ST * MU0 * H0
    t = np.linspace(0.0, 2 * np.pi / w, 101)               # one period
    r = llg.simulate(t, m0=[1.0, 0.0, 0.0])
    assert np.max(np.abs(np.linalg.norm(r.m_t, axis=1) - 1.0)) < 1e-12   # unit sphere conserved
    assert r.m_t[-1, 0] == pytest.approx(1.0, abs=1e-5)    # back to start after one period
    # the R13 seam: any m_t row drives the vector MO tensor directly
    T = np.asarray(VectorMagnetoOpticModel(eps_r=2.25, g_s=0.05).eps(
        {"m_vector": r.m_t[25]}, 1550e-9))
    assert T.shape == (3, 3) and np.allclose(T, T.conj().T)


def test_llg_guards_and_damped_alignment():
    from dynameta.carriers.llg import LLGMacrospin
    with pytest.raises(ValueError):
        LLGMacrospin(Ms_A_m=0.0)                           # Ms > 0
    with pytest.raises(ValueError):
        LLGMacrospin(Ms_A_m=1e5, alpha=-0.1)               # alpha >= 0
    llg = LLGMacrospin(Ms_A_m=1e5, alpha=0.5, H_applied_A_m=lambda t: np.array([0.0, 0.0, 1e4]))
    t = np.linspace(0.0, 50e-9, 201)
    r = llg.simulate(t, m0=[1.0, 0.0, 0.2])
    assert r.m_t[-1, 2] > 0.999                            # damping aligns m with H
    assert np.all(np.diff(r.energy_J_m3) <= 1e-12)         # Lyapunov
