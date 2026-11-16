"""Unit coverage for the EffectModel field-bundle seam (core.effects) -- the v0.3 keystone
that generalizes the scalar n->eps map to fields {n,E,T,...} -> scalar OR tensor eps. Pure numpy.
Run: python -m pytest tests/test_effects.py -q
"""
import functools

import numpy as np
import pytest

from dynameta.core.effects import (OpticalModelEffect, ComposedEffect, DeltaEffect, as_tensor,
                                   PockelsEffect, KerrEffect, FranzKeldyshEffect, ThermoOpticModel,
                                   MagnetoOpticModel, AnisotropicThermoOpticModel, IntersubbandEffect)
from dynameta.materials.optical_model import ConstantOptical, DrudeOptical
from dynameta.constants import M_E, HBAR
from dynameta.carriers.schrodinger_poisson import SubbandResult

# Tensor identity / structure checks (entries O(1-12), the expected zeros exactly 0): they hold to round-off
# (MEASURED <= 8.9e-16), so 1e-12 absolute (~1e3 ulp of the entries) with no relative term -- numpy's default
# atol 1e-8 would accept a 1e-8 leak into an off-diagonal entry that must vanish.
_EXACT = 1e-12


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
    assert np.allclose(as_tensor(2.0 + 0j), 2.0 * np.eye(3), rtol=0.0, atol=_EXACT)
    g = np.array([2.0, 3.0], dtype=complex)              # grid -> (...,3,3)
    tg = as_tensor(g)
    assert tg.shape == (2, 3, 3) and np.allclose(tg[1], 3.0 * np.eye(3), rtol=0.0, atol=_EXACT)
    m = (np.arange(9).reshape(3, 3) + 1).astype(complex)  # already a tensor -> unchanged
    assert np.allclose(as_tensor(m), m)


def test_composed_effect_sums_background_plus_deltas_as_tensors():
    class _Const:                                         # a trivial EffectModel
        def __init__(self, v): self.v = v
        def eps(self, fields, lambda_m): return self.v
    comp = ComposedEffect(background=_Const(4.0 + 0j), deltas=[_Const(0.1 + 0j), _Const(0.05 + 0j)])
    out = comp.eps({}, 1300e-9)
    assert out.shape == (3, 3) and np.allclose(out, 4.15 * np.eye(3), rtol=0.0, atol=_EXACT)


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
    assert np.allclose(comp.eps({"E": np.zeros(3)}, 1300e-9), 4.0 * np.eye(3), rtol=0.0, atol=_EXACT)   # E=0 -> just bg
    shift = as_tensor(pk.eps({"E": [0, 0, Ez]}, 1300e-9)) - as_tensor(eps_bg)
    assert np.allclose(comp.eps({"E": np.array([0.0, 0.0, Ez])}, 1300e-9), 4.0 * np.eye(3) + shift,
                       rtol=0.0, atol=_EXACT)


def test_r24_delta_effect_baseline_overrides_only_its_own_fields():
    """audit R-24: DeltaEffect's baseline REPLACED the bundle, so a model reading E and T evaluated
    its E-reference with T at its default and the 'E delta' carried the full T dependence. The
    baseline now overrides only its own keys."""
    class _ET:                                                 # eps = 2 + a E_z + b (T - 300)
        def eps(self, fields, lambda_m):
            E = np.asarray(fields.get("E", np.zeros(3)), dtype=float)
            T = float(fields.get("T", 300.0))
            return complex(2.0 + 1e-8 * E[..., 2] + 1e-3 * (T - 300.0))
    d = DeltaEffect(_ET(), {"E": np.zeros(3)})
    got = as_tensor(d.eps({"E": np.array([0.0, 0.0, 2e6]), "T": 400.0}, 1300e-9))
    assert np.allclose(got, 0.02 * np.eye(3), rtol=0.0, atol=1e-15)   # E shift only (pre-fix 0.12)


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
    assert np.allclose(pk.eps({"E": np.zeros(3)}, 1300e-9), eps_bg, rtol=0.0, atol=_EXACT)          # E=0 -> background
    Ez = 1.0e7
    eps = pk.eps({"E": np.array([0.0, 0.0, Ez])}, 1300e-9)
    assert abs(eps[0, 1]) < 1e-12 and abs(eps[0, 2]) < 1e-12                 # stays diagonal
    nx, nz = np.sqrt(eps[0, 0].real), np.sqrt(eps[2, 2].real)
    assert nx == pytest.approx(no - 0.5 * no ** 3 * r13 * Ez, rel=1e-3)      # Pockels: n_o via r13
    assert nz == pytest.approx(ne - 0.5 * ne ** 3 * r33 * Ez, rel=1e-3)      # Pockels: n_e via r33
    pk0 = PockelsEffect(eps_bg=eps_bg, r_voigt=np.zeros((6, 3)))             # r=0 -> background
    assert np.allclose(pk0.eps({"E": np.array([0.0, 0.0, Ez])}, 1300e-9), eps_bg, rtol=0.0, atol=_EXACT)


def test_kerr_reduces_and_is_quadratic_in_field():
    eps_bg = (2.0 ** 2) * np.eye(3, dtype=complex)
    kr = KerrEffect(eps_bg=eps_bg, s_kerr=1e-18)
    assert np.allclose(kr.eps({"E": np.zeros(3)}, 1300e-9), eps_bg, rtol=0.0, atol=_EXACT)
    dB1 = 1.0 / kr.eps({"E": [0, 0, 1e8]}, 1300e-9)[0, 0] - 1.0 / eps_bg[0, 0]
    dB2 = 1.0 / kr.eps({"E": [0, 0, 2e8]}, 1300e-9)[0, 0] - 1.0 / eps_bg[0, 0]
    assert (dB2 / dB1).real == pytest.approx(4.0, rel=1e-6)                  # |E|^2 -> 2x field, 4x shift
    assert np.allclose(KerrEffect(eps_bg, 0.0).eps({"E": [0, 0, 1e8]}, 1300e-9), eps_bg,
                       rtol=0.0, atol=_EXACT)


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
    assert np.allclose(T, T.conj().T, rtol=0.0, atol=_EXACT)                  # Hermitian -> lossless
    # circular + axial eigenvalues {eps_r - g, eps_r, eps_r + g}
    assert np.allclose(np.sort(np.linalg.eigvals(T).real), [eps_r - g, eps_r, eps_r + g])
    # magnetization scales/flips g; 0 reduces to isotropic eps_r*I
    assert np.asarray(mo.eps({"magnetization": -1.0}, 1550e-9))[0, 1] == pytest.approx(-1j * g)
    assert np.allclose(np.asarray(mo.eps({"magnetization": 0.0}, 1550e-9)), eps_r * np.eye(3),
                       rtol=0.0, atol=_EXACT)
    assert np.allclose(np.asarray(MagnetoOpticModel(eps_r, 0.0).eps({}, 1550e-9)), eps_r * np.eye(3),
                       rtol=0.0, atol=_EXACT)


def test_anisotropic_thermo_optic_diagonal_and_reduces_to_scalar():
    no, ne = 2.20, 2.30
    dno, dne = 1.0e-4, 3.0e-4
    m = AnisotropicThermoOpticModel(eps_ref_diag=(no ** 2, no ** 2, ne ** 2),
                                    dn_dT_diag=(dno, dno, dne), T_ref=300.0)
    T0 = np.asarray(m.eps({"T": 300.0}, 1550e-9))                    # at T_ref -> diag(eps_ref)
    assert T0.shape == (3, 3) and np.allclose(np.diag(T0), [no ** 2, no ** 2, ne ** 2])
    assert np.allclose(T0 - np.diag(np.diag(T0)), 0, rtol=0.0, atol=_EXACT)                 # strictly diagonal
    dT = 50.0
    Tt = np.asarray(m.eps({"T": 300.0 + dT}, 1550e-9))
    assert np.sqrt(Tt[0, 0].real) == pytest.approx(no + dno * dT, rel=1e-9)   # ordinary axis
    assert np.sqrt(Tt[2, 2].real) == pytest.approx(ne + dne * dT, rel=1e-9)   # extraordinary axis
    # isotropic dn/dT reduces EXACTLY to the scalar ThermoOpticModel * I
    iso = AnisotropicThermoOpticModel((no ** 2,) * 3, (dno,) * 3)
    sca = ThermoOpticModel(eps_ref=complex(no ** 2, 0.0), dn_dT=dno)
    assert np.allclose(np.asarray(iso.eps({"T": 360.0}, 1550e-9)),
                       complex(sca.eps({"T": 360.0}, 1550e-9)) * np.eye(3), rtol=0.0, atol=_EXACT)
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


def test_intersubband_single_band_reduces_to_eps_inf_along_z():
    """v0.16 (no Drude term along z): one sub-band has no intersubband transition, so the z response is the background
    alone (eps_zz == eps_inf: a confined electron cannot be accelerated along z) while the in-plane
    response is the Drude of its sheet density. v0.15.1 returned eps_zz == eps_xx (the Drude, the
    double count): eps_zz = 3.3568+0.0678i, |eps_zz - eps_inf| = 0.896 on this fixture. A one-state set is incomplete by
    construction, hence fsum_check='off'."""
    res = _one_band_subband()
    m = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off")
    eps_t = m.eps({"subband": res}, 1300e-9)
    n3d = float(res.sheet_density_m2.sum()) / (res.z_m[-1] - res.z_m[0])
    d = complex(DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAM_INTRA).eps(1300e-9, n_m3=n3d))
    assert eps_t.shape == (3, 3)
    assert abs(eps_t[0, 0] - d) < 1e-13 and eps_t[1, 1] == eps_t[0, 0]
    assert abs(eps_t[2, 2] - EPS_INF) < 1e-13                         # no z Drude term
    assert abs(eps_t[0, 1]) == 0.0 and m.last_fsum_ratio == 0.0


def test_intersubband_legacy_single_band_reduces_to_drude():
    """The deprecated z_response='drude_plus_subbands' keeps the v0.15.1 form (eps_zz = Drude + sum
    ISB, here the Drude alone) and says it is deprecated (removal in 0.17)."""
    from dynameta.warning_types import AdvisoryWarning
    res = _one_band_subband()
    m = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, z_response="drude_plus_subbands")
    with pytest.warns(AdvisoryWarning, match="deprecated"):
        eps_t = m.eps({"subband": res}, 1200e-9)
    n3d = float(res.sheet_density_m2.sum()) / (res.z_m[-1] - res.z_m[0])
    d = complex(DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAM_INTRA).eps(1200e-9, n_m3=n3d))
    assert np.allclose(eps_t, as_tensor(np.asarray(d)), rtol=0.0, atol=1e-13)  # isotropic == Drude*I


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
    m = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off")   # 2-state: incomplete
    lam12 = 2 * np.pi * 299792458.0 / ((E[1] - E[0]) / HBAR)
    eps_t = m.eps({"subband": res}, lam12)
    assert abs(eps_t[2, 2] - eps_t[0, 0]) > 1e-3                      # z carries the intersubband line
    assert abs(eps_t[2, 2] - EPS_INF) > 1e-3
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
    # AUDIT T-6: rtol=0.0 -- the delta is identically zero at n_ref, so this is an exactness check
    # and must not inherit numpy's 1e-5 relative default (|eps| ~ 4 -> 4e-5 absolute).
    assert np.allclose(comp.eps({"n": n_ref}, 1300e-9),
                       as_tensor(np.asarray(complex(drude.eps(1300e-9, n_m3=n_ref)))),
                       rtol=0.0, atol=1e-12)
    # enabled=False -> bare Drude at ALL n (true off-switch)
    off = ComposedEffect(background=bg, deltas=[DeltaEffect(
        BursteinMossEdge(eps_inf=eps_inf, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5,
                         enabled=False), baseline_fields={"n": n_ref})])
    for n in (2e26, 1e27):
        # AUDIT T-6: rtol=0.0 -- a true off-switch must reproduce the bare Drude tensor exactly.
        assert np.allclose(off.eps({"n": n}, 1300e-9),
                           as_tensor(np.asarray(complex(drude.eps(1300e-9, n_m3=n)))),
                           rtol=0.0, atol=1e-14)


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


# ---- audit R-1: the ISB pair filter must gate on the POPULATION DIFFERENCE -------------------

def _two_band_subband(ns, L=2e-9, nz=160):
    """Two hard-wall sub-bands with an analytic dipole: z_01 = -16 L/(9 pi^2)."""
    from dynameta.carriers.schrodinger_poisson import SubbandResult
    z = np.linspace(0.0, L, nz)
    h = z[1] - z[0]
    p1 = np.sin(np.pi * z / L)
    p2 = np.sin(2 * np.pi * z / L)
    p1 = p1 / np.sqrt(np.sum(p1 ** 2) * h)
    p2 = p2 / np.sqrt(np.sum(p2 ** 2) * h)
    E = np.array([0.0, 0.95 * 1.602176634e-19])
    return SubbandResult(energies_J=E, psi=np.stack([p1, p2], axis=1), z_m=z,
                         sheet_density_m2=np.asarray(ns, dtype=float))


def test_intersubband_full_ground_empty_excited_is_the_strongest_line():
    """audit R-1: the canonical absorber -- ground FULL, first excited EMPTY -- has the largest
    population difference and must produce the STRONGEST line. The shipped filter drew BOTH i and j
    from the occupancy-filtered set and deleted exactly this configuration (bare Drude, no warning).
    Anchored on the closed-form Lorentzian ON resonance: Im(chi_zz) = S_01/(eps0 w_12 gamma).
    v0.16: chi_zz is eps_zz - eps_inf (no Drude term along z); the 2-state well is incomplete,
    hence fsum_check='off'. The dipole is the Riemann sum about the slab centre; on this grid the states vanish at both
    end nodes, so it equals the trapezoid value the gate used before."""
    from dynameta.constants import HBAR, C_LIGHT, Q_E, EPS0
    from dynameta.core.numerics import trapz
    ns = 5.0e17
    res = _two_band_subband([ns, 0.0])                       # EMPTY excited sub-band
    m = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off")
    E, z, psi = res.energies_J, res.z_m, res.psi
    w12 = (E[1] - E[0]) / HBAR
    lam12 = 2.0 * np.pi * C_LIGHT / w12
    t = m.eps({"subband": res}, lam12)
    Leff = float(z[-1] - z[0])
    z01 = trapz(psi[:, 0] * z * psi[:, 1], z)
    S01 = (ns / Leff) * Q_E * Q_E * z01 * z01 * (2.0 * w12) / HBAR
    assert float((t[2, 2] - EPS_INF).imag) == pytest.approx(S01 / (EPS0 * w12 * GAM_INTER), rel=3e-2)
    # ... and it must be STRONGER than the same well with the upper band partly filled
    weaker = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off").eps(
        {"subband": _two_band_subband([ns, 0.4 * ns])}, lam12)
    assert (t[2, 2] - EPS_INF).imag > (weaker[2, 2] - EPS_INF).imag > 0.0


def test_intersubband_inverted_population_reaches_the_gain_warning():
    """audit R-1 (inverted case): n_s = [0, n] is the strongest possible inversion. The shipped
    filter removed the pair before the N_ij < 0 warning at the line-strength step could fire, so the
    module's only unphysical-occupancy diagnostic was unreachable in the configuration it exists for.
    v0.16: the clamped line leaves eps_zz == eps_inf (no z Drude term)."""
    from dynameta.constants import HBAR, C_LIGHT
    res = _two_band_subband([0.0, 5.0e17])
    lam12 = 2.0 * np.pi * C_LIGHT / ((res.energies_J[1] - res.energies_J[0]) / HBAR)
    m = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off")
    with pytest.warns(UserWarning, match="inverted population"):
        t = m.eps({"subband": res}, lam12)
    assert abs(t[2, 2] - EPS_INF) < 1e-12                    # clamped to 0: no gain, exp(-iwt)


def test_intersubband_equal_populations_give_no_line():
    from dynameta.constants import HBAR, C_LIGHT
    res = _two_band_subband([3.0e17, 3.0e17])                # N_ij = 0 -> nothing to radiate
    lam12 = 2.0 * np.pi * C_LIGHT / ((res.energies_J[1] - res.energies_J[0]) / HBAR)
    t = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off").eps(
        {"subband": res}, lam12)
    assert abs(t[2, 2] - EPS_INF) < 1e-13


# ---- audit R-3: Maclaurin KK endpoint half-weights (BOTH parities, BOTH grid lengths) ---------

_KK_E_LO, _KK_E_HI, _KK_SIG = 1.602176634e-19, 6.0 * 1.602176634e-19, 0.25 * 1.602176634e-19


def _kk_bump(center):
    return lambda E: np.exp(-0.5 * ((E - center) / _KK_SIG) ** 2)


def _kk_pv_reference(profile, e_eval):
    """PV reference by quadrature. Every evaluation energy used below sits where the profile is
    < 1e-30, so the pole contributes nothing and a split-range quad is exact."""
    from scipy.integrate import quad
    from dynameta.constants import HBAR, C_LIGHT
    d = 1e-6 * 1.602176634e-19
    lo = quad(lambda E: profile(E) / (E ** 2 - e_eval ** 2), _KK_E_LO, e_eval - d, limit=400)[0] \
        if e_eval - d > _KK_E_LO else 0.0
    hi = quad(lambda E: profile(E) / (E ** 2 - e_eval ** 2), e_eval + d, _KK_E_HI, limit=400)[0] \
        if e_eval + d < _KK_E_HI else 0.0
    return (HBAR * C_LIGHT / np.pi) * (lo + hi)


def _kk_order(profile, n0, idx=None, idx_from_end=None, levels=3):
    """Convergence order by 3x refinement. An ODD refinement factor preserves BOTH the evaluation
    index parity and the grid-length parity, and keeps the coarse evaluation ENERGY on every grid."""
    from dynameta.core.effects.base import kramers_kronig_dn
    errs = []
    for k in range(levels):
        n = 3 ** k * (n0 - 1) + 1
        E = np.linspace(_KK_E_LO, _KK_E_HI, n)
        dn = kramers_kronig_dn(E, profile(E))
        i = (n - idx_from_end) if idx_from_end is not None else 3 ** k * idx
        errs.append(abs(dn[i] - _kk_pv_reference(profile, E[i])))
    return min(np.log(errs[k] / errs[k + 1]) / np.log(3.0) for k in range(levels - 1))


@pytest.mark.parametrize("n0", [244, 245])                   # even N and odd N
def test_kramers_kronig_second_order_at_both_parities_and_both_grid_lengths(n0):
    """audit R-3. The alternate-point rule is composite-MIDPOINT on the branch summing interior
    nodes but composite-TRAPEZOID on the branch summing the panel NODES, where the vectorized form
    double-weighted the two grid endpoints -> O(h). For an ODD-length grid that hits one parity; for
    an EVEN-length grid (reachable via the public e_grid_J override) the endpoints land in DIFFERENT
    parity blocks and BOTH branches collapse, so halving only the odd branch is not enough. Measured
    orders with the fix reverted: 1.00 on the exposed branches; with it, 2.00 everywhere."""
    both = lambda E: _kk_bump(_KK_E_LO)(E) + _kk_bump(_KK_E_HI)(E)
    mid = (n0 - 1) // 2
    checks = [
        (both, dict(idx=mid)), (both, dict(idx=mid + 1)),            # interior, both parities
        (_kk_bump(_KK_E_HI), dict(idx=0)), (_kk_bump(_KK_E_HI), dict(idx=1)),
        (_kk_bump(_KK_E_LO), dict(idx_from_end=2)), (_kk_bump(_KK_E_LO), dict(idx_from_end=1)),
    ]
    for profile, where in checks:
        assert _kk_order(profile, n0, **where) >= 1.9, where


@pytest.mark.parametrize("n", [401, 402])                    # odd and even grid lengths
def test_kramers_kronig_batched_paths_agree_with_the_scalar_rule(n):
    """The R-3 endpoint weights are applied in THREE places (the scalar rule, the cached parity
    kernel, and the O(N) probe-row path). All three must stay in step -- BursteinMossEdge reaches
    the transform only through the third."""
    from dynameta.core.effects.base import kramers_kronig_dn, kramers_kronig_dn_rows
    from dynameta.constants import Q_E
    E = np.linspace(0.4 * Q_E, 5.0 * Q_E, n)
    rows = np.stack([np.exp(-0.5 * ((E - c * Q_E) / (0.4 * Q_E)) ** 2) for c in (1.0, 2.5, 4.6)])
    ref = np.stack([kramers_kronig_dn(E, r) for r in rows])
    assert np.allclose(kramers_kronig_dn_rows(E, rows), ref, rtol=1e-12, atol=0.0)
    e_eval = float(E[n // 3]) + 0.37 * float(E[1] - E[0])    # between two nodes
    got = kramers_kronig_dn_rows(E, rows, e_eval_J=e_eval)
    want = np.array([np.interp(e_eval, E, r) for r in ref])
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_kramers_kronig_matches_an_analytic_pair():
    """Closed-form KK pair: a(E') = A/(E'^2 + c^2)  ->  dn(E) = -hbar c A / (2 c (E^2 + c^2))
    (partial fractions plus P int_0^inf dE'/(E'^2 - E^2) = 0). The lower endpoint is HOT here, so
    this also exercises the R-3 endpoint weight."""
    from dynameta.core.effects.base import kramers_kronig_dn
    from dynameta.constants import HBAR, C_LIGHT, Q_E
    A, c = 1.0e6, 0.8 * Q_E
    E = np.linspace(1e-24, 60.0 * c, 6001)
    dn = kramers_kronig_dn(E, A / (E ** 2 + c ** 2))
    sel = np.array([200, 501, 1200, 3001])
    exact = -HBAR * C_LIGHT * A / (2.0 * c * (E[sel] ** 2 + c ** 2))
    assert np.max(np.abs(dn[sel] / exact - 1.0)) < 1e-2


# ---- audit R-2: BursteinMossEdge KK grid truncation + DeltaEffect grid pinning ----------------

def _bm_dn_quad(alpha_edge, Eg_J, e_ph_J, p=0.5):
    """EXACT dn including the tail to infinity, via E' = Eg/t (the probe is below the gap, so the
    KK integral has no pole in range). This is the oracle the audit prescribes -- a wider grid
    cannot see a truncation that is common to both halves of a difference."""
    from scipy.integrate import quad
    def f(t):
        ep = Eg_J / t
        return ((((ep - Eg_J) / Eg_J) ** p * (Eg_J / ep) ** 2 * ep / (ep ** 2 - e_ph_J ** 2))
                * (Eg_J / t ** 2))
    return alpha_edge / np.pi * quad(f, 1e-14, 1.0, limit=600)[0]


def test_burstein_moss_dn_matches_quad_to_infinity():
    """audit R-2: the shipped auto grid stopped 5 eV above the edge and was 44-50% LOW in absolute
    dn -- the cutoff-free Tauc tail leaves an E_hi^-1.5 remainder. The grid is now sized from that
    law and what remains is added back in closed form."""
    from dynameta.constants import Q_E
    from dynameta.core.effects import BursteinMossEdge
    from dynameta.core.effects.base import _photon_energy_J
    edge = BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5)
    n_re0 = np.sqrt(complex(4.25)).real
    for lam in (1300e-9, 1550e-9):
        for n in (1e26, 7e26, 1e27):
            got = float(np.sqrt(complex(edge.eps({"n": np.asarray(float(n))}, lam))).real - n_re0)
            ref = _bm_dn_quad(1.5, float(edge.optical_gap_J(np.asarray(float(n)))),
                              _photon_energy_J(lam))
            assert got == pytest.approx(ref, rel=2e-2)


def test_burstein_moss_delta_uses_one_pinned_grid():
    """audit R-2: the grid was rebuilt PER eps() CALL from the per-call Eg_opt(n), so DeltaEffect
    integrated its two halves on DIFFERENT grids and the common truncation did not cancel (the delta
    came out 14-16x too large). The grid is now a pure function of the configuration."""
    from dynameta.constants import Q_E
    from dynameta.core.effects import BursteinMossEdge
    from dynameta.core.effects.base import _photon_energy_J
    edge = BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5)
    n_re0, scale = np.sqrt(complex(4.25)).real, edge._kk_dn_scale()
    grid0 = edge._kk_grid()
    dn = lambda n, lam: float(np.sqrt(complex(edge.eps({"n": np.asarray(float(n))}, lam))).real - n_re0)
    for lam in (1300e-9, 1550e-9):
        e_ph = _photon_energy_J(lam)
        ref = dn(4e26, lam)
        q_ref = _bm_dn_quad(1.5, float(edge.optical_gap_J(np.asarray(4e26))), e_ph)
        for n in (1e26, 7e26, 1e27):
            d_model = dn(n, lam) - ref
            d_quad = _bm_dn_quad(1.5, float(edge.optical_gap_J(np.asarray(float(n)))), e_ph) - q_ref
            assert abs(d_model - d_quad) / scale < 1e-4
    assert edge._kk_grid() is grid0                           # same object across every n and lambda


def test_burstein_moss_tail_test_replaces_the_bracket_only_guard():
    """audit R-2: the old guard only checked that the grid BRACKETED the gap and the probe, so it
    accepted a grid ending 1 meV above Eg. A grid of the SHIPPED width must now be flagged."""
    from dynameta.constants import Q_E
    from dynameta.core.effects import BursteinMossEdge
    short = BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5,
                             e_grid_J=(1e-21, 9.0 * Q_E, 3001))
    with pytest.warns(RuntimeWarning, match="Kramers-Kronig grid ends"):
        short.eps({"n": np.asarray(4e26)}, 1300e-9)
    # p >= 2 is a genuinely divergent KK integral, not a grid choice
    with pytest.raises(ValueError, match="tauc_exponent"):
        BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5,
                         tauc_exponent=2.0).eps({"n": np.asarray(4e26)}, 1300e-9)


def test_burstein_moss_tauc_p0_is_a_clean_step_and_stays_passive():
    """FIX-VERIFY W1 kill 1. ``_eps2`` evaluated the Tauc shape factor as ``x ** p`` with
    ``x = max(E - Eg, 0)/Eg``.  IEEE-754 (and numpy) define ``0.0 ** 0.0 == 1.0``, so at
    ``tauc_exponent = 0`` -- the physical 2-D / constant-joint-DOS edge, and an ALLOWED exponent --
    the factor was 1 EVERYWHERE BELOW THE GAP and eps2 came out ``alpha_edge (Eg/E)^2``, diverging
    toward E -> 0 (measured: 1.5 at the gap, 6.0 at 0.5 Eg, 24.0 at 0.25 Eg).  That sub-gap
    absorption also entered the KK quadrature, whose grid starts at ~0, and the resulting dn drove
    ``n_re`` negative: ``eps(0.5 Eg)`` came back ``82.6 - 57.4j`` -- Im(eps) < 0, i.e. GAIN out of a
    passive interband edge.  The shape factor is now masked on ``x > 0``, so p = 0 is a clean step
    and every p > 0 is byte-identical to the unmasked form."""
    from dynameta.constants import Q_E
    from dynameta.core.effects import BursteinMossEdge
    from dynameta.core.effects.base import _photon_energy_J
    edge = BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5,
                            tauc_exponent=0.0)
    n = np.asarray(1e26)
    Eg = float(edge.optical_gap_J(n))

    # (a) the step: EXACTLY zero below (and at) the gap, alpha_edge (Eg/E)^2 above it.
    below = np.array([1e-30, 0.1, 0.25, 0.5, 0.9, 0.999, 1.0]) * Eg
    above = np.array([1.0001, 1.2, 2.0, 10.0]) * Eg
    assert np.all(np.asarray(edge._eps2(below, Eg)) == 0.0)
    assert np.allclose(np.asarray(edge._eps2(above, Eg)), 1.5 * (Eg / above) ** 2,
                       rtol=0.0, atol=0.0)

    # (b) passivity at the 0.5 Eg probe the pre-fix code turned into gain.
    e_half = complex(edge.eps({"n": n}, _photon_energy_J(1.0) / (0.5 * Eg)))
    assert e_half.imag == 0.0 and e_half.real > 0.0
    for frac in (0.1, 0.25, 0.5, 0.75, 0.95, 1.2, 2.0):
        assert complex(edge.eps({"n": n}, _photon_energy_J(1.0) / (frac * Eg))).imag >= 0.0

    # (c) continuity as p -> 0 (the pre-fix p = 0 branch was discontinuous in p by construction).
    n_re0 = np.sqrt(complex(4.25)).real
    lam = _photon_energy_J(1.0) / (0.5 * Eg)
    dn = {}
    for p in (0.0, 1e-9):
        ed = BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5,
                              tauc_exponent=p)
        dn[p] = float(np.sqrt(complex(ed.eps({"n": n}, lam))).real - n_re0)
    assert dn[1e-9] == pytest.approx(dn[0.0], rel=1e-8)
    # and it is the RIGHT number: the step edge's KK partner, to the O(h) the 10 meV grid allows
    assert dn[0.0] == pytest.approx(_bm_dn_quad(1.5, Eg, 0.5 * Eg, p=0.0), rel=2e-3)

    # (d) p > 0 is untouched, bit for bit.
    rng = np.random.default_rng(0)
    E = np.concatenate([np.linspace(1e-25, 5 * Eg, 20001), np.array([Eg]),
                        rng.uniform(1e-25, 5 * Eg, 5000)])
    for p in (0.25, 0.5, 1.0, 1.9):
        ed = BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5,
                              tauc_exponent=p)
        x = np.maximum(E - Eg, 0.0) / Eg
        assert np.array_equal(np.asarray(ed._eps2(E, Eg)), 1.5 * x ** p * (Eg / E) ** 2)


def test_burstein_moss_auto_kk_grid_is_size_capped():
    """FIX-VERIFY W1 kill 1 (second half).  The auto E_hi scales as ``tol^(-1/(2-p))``, so at the
    fixed 10 meV spacing the grid is 7.7e4 points at p = 0.5 but 1.56e9 -- a 12.4 GB float64 array,
    before the per-row temporaries -- at p = 1.5 (direct-forbidden Tauc).  ``_KK_ROW_CHUNK_BYTES``
    cannot help: a chunk is never smaller than ONE row of ``grid.size`` doubles.  The auto grid is
    now capped and raises with the ``e_grid_J`` escape hatch named; the override itself still
    works (only warning about the tail)."""
    from dynameta.constants import Q_E
    from dynameta.core.effects import BursteinMossEdge

    def mk(p, **kw):
        return BursteinMossEdge(eps_inf=4.25, Eg0_J=3.6 * Q_E, m_vc_kg=0.5 * M_E, alpha_edge=1.5,
                                tauc_exponent=p, **kw)

    for p, npts in ((0.5, 76849), (1.0, 1080001)):                 # under the cap, unchanged
        assert mk(p)._kk_grid().size == npts
    for p in (1.5, 1.9):
        with pytest.raises(ValueError, match="e_grid_J"):
            mk(p)._kk_grid()
    # the escape hatch: an explicit grid at p = 1.5 runs, warns about the tail, and stays passive
    over = mk(1.5, e_grid_J=(1e-21, 2000.0 * Q_E, 20001))
    with pytest.warns(RuntimeWarning, match="Kramers-Kronig grid ends"):
        val = complex(over.eps({"n": np.asarray(1e26)}, 1550e-9))
    assert np.isfinite(val.real) and val.imag >= 0.0


def test_photon_energy_uses_the_exact_planck_constant():
    """audit X-7 / R-17: h was re-derived as 2*pi*HBAR (6.1e-10 relative drift) upstream of the
    whole electro-absorption / KK stack."""
    from dynameta.constants import C_LIGHT, H_PLANCK
    from dynameta.core.effects.base import _photon_energy_J
    assert _photon_energy_J(1300e-9) == H_PLANCK * C_LIGHT / 1300e-9


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
    assert np.allclose(T, T.conj().T, rtol=0.0, atol=_EXACT)  # Hermitian (lossless)
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
    assert T.shape == (3, 3) and np.allclose(T, T.conj().T, rtol=0.0, atol=_EXACT)


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


# ---- v0.16: intersubband z response without the double count (no Drude term along z) and the dipole
# quadrature (Riemann sum about the slab centre) --

_ISB_M = 0.30 * M_E
_ISB_LAM = 1200e-9
_ISB_EPS_INF, _ISB_GAM, _ISB_NBG = 4.0, 1.2e14, 3e26           # the generic gated film


@functools.lru_cache(maxsize=None)
def _isb_slab(t_m, dz_m=0.025e-9, n_states=None):
    """Flat-band slab (n_bg 3e26 m^-3 at the model's own E_F, m 0.30 m_e, 300 K), all sub-bands kept
    (slab mode). n_states None -> min(z.size - 3, 600)."""
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, sp_bulk_fermi_level_J
    z = np.linspace(0.0, t_m, int(round(t_m / dz_m)) + 1)
    sp = SchrodingerPoisson1D(z, _ISB_M)
    ef = sp_bulk_fermi_level_J(_ISB_NBG, _ISB_M, 300.0)
    ns = min(z.size - 3, 600) if n_states is None else int(n_states)
    return sp.density(np.zeros_like(z), ef, n_states=ns, bound_tol=1e9)


def _isb_model(**kw):
    return IntersubbandEffect(eps_inf=_ISB_EPS_INF, m_opt_kg=_ISB_M, gamma_intra_rad_s=_ISB_GAM,
                              gamma_inter_rad_s=_ISB_GAM, **kw)


def _isb_v0151_reference(res, lambda_m, eps_inf, m_opt, g_intra, g_inter, floor=0.0):
    """FROZEN copy of the v0.15.1 IntersubbandEffect.eps body (electroabsorption.py:613-659 of v0.15.1),
    the oracle of the legacy-mode bit-identity gate (warnings stripped: the gate passes no inverted
    pair)."""
    from dynameta.constants import C_LIGHT, EPS0, HBAR, Q_E
    from dynameta.core.numerics import trapz
    z = np.asarray(res.z_m, dtype=np.float64)
    psi = np.asarray(res.psi, dtype=np.float64)
    E = np.asarray(res.energies_J, dtype=np.float64)
    ns = np.asarray(res.sheet_density_m2, dtype=np.float64)
    Leff = float(z[-1] - z[0])
    omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
    n3d = float(np.sum(ns)) / Leff
    wp2 = n3d * Q_E * Q_E / (EPS0 * float(m_opt))
    eps_intra = complex(eps_inf) - wp2 / (omega * omega + 1j * omega * float(g_intra))
    eps_zz = eps_intra
    gam = float(g_inter)
    for i in range(E.size):
        for j in range(i + 1, E.size):
            dns = float(ns[i] - ns[j])
            if abs(dns) <= floor:
                continue
            w_ij = (E[j] - E[i]) / HBAR
            z_ij = trapz(psi[:, i] * z * psi[:, j], z)
            N_ij = dns / Leff
            if N_ij < 0.0:
                N_ij = 0.0
            S_ij = N_ij * Q_E * Q_E * (z_ij * z_ij) * (2.0 * w_ij) / HBAR
            eps_zz += S_ij / (EPS0 * (w_ij * w_ij - omega * omega - 1j * omega * gam))
    out = np.zeros((3, 3), dtype=complex)
    out[0, 0] = eps_intra
    out[1, 1] = eps_intra
    out[2, 2] = eps_zz
    return out


def test_intersubband_fsum_of_a_complete_slab_is_one_and_equals_the_discrete_identity():
    """f-sum gate (no Drude term along z; Riemann dipole). 6 nm flat-band slab (0.025 nm grid): the f-sum
    ratio sum_{i<j} (n_i - n_j) f_ij / sum n_i must be within 1e-3 of 1 (MEASURED 0.99967 with 238 states;
    v0.15.1 had no diagnostic and its trapezoid dipole on the absolute z gives 1.1520 -- 93.1 after a 100 nm
    shift, measured on the v0.15.1 export at this tree's E_F).
    Independent oracle for the COMPLETE discrete set (all 239 states): for the finite-difference
    Hamiltonian the double commutator gives sum_j f_ij = kappa_i = h sum_k psi_i,k psi_i,k+1 exactly,
    so the ratio must equal sum n_i kappa_i / sum n_i to rounding (re-derived here from neighbour
    products only; no eigen-sum). The 1 - ratio = 3.3e-4 is that O((k h)^2) kinetic weight.
    """
    res = _isb_slab(6e-9)
    m = _isb_model()
    m.eps({"subband": res}, _ISB_LAM)
    assert abs(m.last_fsum_ratio - 1.0) < 1e-3, m.last_fsum_ratio
    full = _isb_slab(6e-9, n_states=239)                       # every interior state: complete
    mf = _isb_model()
    mf.eps({"subband": full}, _ISB_LAM)
    h = float(full.z_m[1] - full.z_m[0])
    kappa = h * np.sum(full.psi[:-1, :] * full.psi[1:, :], axis=0)
    ns = full.sheet_density_m2
    oracle = float(np.sum(ns * kappa) / np.sum(ns))
    assert abs(mf.last_fsum_ratio - oracle) < 1e-12, (mf.last_fsum_ratio, oracle)
    assert 0.999 < oracle < 1.0


def test_intersubband_eps_zz_is_invariant_to_the_z_origin():
    """Dipole-quadrature gate: a 100 nm shift of the z origin must not move eps_zz (budget 1e-12; MEASURED
    1.8e-13, the rounding of z - z_c at 100 nm). v0.15.1 moved it by 1.3e-3 (trapezoid on the absolute z,
    whose end-node weights break the Riemann orthonormality; measured on the v0.15.1 export).
    """
    from dynameta.carriers.schrodinger_poisson import SubbandResult
    res = _isb_slab(6e-9)
    shifted = SubbandResult(energies_J=res.energies_J, psi=res.psi, z_m=res.z_m + 100e-9,
                            sheet_density_m2=res.sheet_density_m2)
    a = _isb_model().eps({"subband": res}, _ISB_LAM)
    b = _isb_model().eps({"subband": shifted}, _ISB_LAM)
    assert abs(a[2, 2] - b[2, 2]) < 1e-12
    assert abs(a[0, 0] - b[0, 0]) < 1e-13                   # Leff = z[-1] - z[0] rounds at 100 nm
    old = abs(_isb_v0151_reference(res, _ISB_LAM, _ISB_EPS_INF, _ISB_M, _ISB_GAM, _ISB_GAM)[2, 2]
              - _isb_v0151_reference(shifted, _ISB_LAM, _ISB_EPS_INF, _ISB_M, _ISB_GAM, _ISB_GAM)[2, 2])
    assert old > 1e-3                                          # the defect the gate exists for


def test_intersubband_default_thickness_is_wall_to_wall_and_converges_at_second_order():
    """Leff without thickness_m. A SubbandResult of a Dirichlet slab lives on the INTERIOR nodes (the wall nodes,
    psi = 0, excluded), so the film is z[-1] - z[0] + 2h thick and that is the default Leff; a grid that includes
    its wall nodes (psi = 0 at both ends) gives z[-1] - z[0]. Checks, 6 nm flat-band slab, complete basis, 1200 nm:
      * default == thickness_m = 6 nm (the slab's own thickness) to 1e-13 (MEASURED 0, same arithmetic up to the
        rounding of z[-1] - z[0] + 2h);
      * the same states written on the grid WITH its wall nodes (zero rows added) give the same eps to 1e-13
        (MEASURED 0): the default reads the grid convention, not the array length;
      * eps_zz - eps_inf converges at second order in the spacing: successive differences at 101 / 201 / 401 /
        801 nodes shrink by 3.93 / 3.98 (bar [3.6, 4.4]: O(h^2) with an O(h^4) remainder). The pre-0.16 default
        z[-1] - z[0] (off by 2h / L, 1.0 % at 201 nodes) shrinks by 1.64 / 1.83 only (first order), and fails it."""
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, SubbandResult, sp_bulk_fermi_level_J
    ef = sp_bulk_fermi_level_J(_ISB_NBG, _ISB_M, 300.0)
    chi = []
    for nz in (101, 201, 401, 801):
        z = np.linspace(0.0, 6e-9, nz)
        res = SchrodingerPoisson1D(z, _ISB_M).density(np.zeros_like(z), ef, n_states=None, bound_tol=np.inf)
        e = _isb_model().eps({"subband": res}, _ISB_LAM)
        assert np.max(np.abs(e - _isb_model(thickness_m=6e-9).eps({"subband": res}, _ISB_LAM))) < 1e-13
        if nz == 201:
            walled = SubbandResult(energies_J=res.energies_J, z_m=z,
                                   psi=np.vstack([np.zeros((1, res.psi.shape[1])), res.psi,
                                                  np.zeros((1, res.psi.shape[1]))]),
                                   sheet_density_m2=res.sheet_density_m2)
            assert np.max(np.abs(_isb_model().eps({"subband": walled}, _ISB_LAM) - e)) < 1e-13
        chi.append(e[2, 2] - _ISB_EPS_INF)
    d = np.abs(np.diff(chi))
    ratios = d[:-1] / d[1:]
    assert np.all((3.6 <= ratios) & (ratios <= 4.4)), ratios


def test_intersubband_wide_slab_limit_eps_zz_tends_to_eps_xx_as_one_over_L():
    """Wide-slab gate: as the slab widens the transitions crowd into a continuum and eps_zz -> eps_xx (same
    mass for both here). abs(eps_zz - eps_xx) at 1200 nm for 10/20/40 nm must fall strictly, with the
    20 -> 40 nm ratio in [1.6, 2.4] (1/L: 2) and below 0.017 at 40 nm (1.3x the measured value).
    MEASURED 0.0509 / 0.0257 / 0.0133, ratio 1.94 (Leff the wall-to-wall slab thickness). v0.15.1 (Drude + sum ISB, measured on the v0.15.1
    export): 1.18 / 1.23 / 1.26 -- growing, the whole Drude term counted twice."""
    d = []
    for t in (10e-9, 20e-9, 40e-9):
        e = _isb_model().eps({"subband": _isb_slab(t)}, _ISB_LAM)
        d.append(abs(e[2, 2] - e[0, 0]))
    assert d[0] > d[1] > d[2], d
    assert 1.6 <= d[1] / d[2] <= 2.4, d
    assert d[2] < 0.017, d


def test_intersubband_legacy_mode_is_bit_identical_to_v0151():
    """Restore path: z_response='drude_plus_subbands' restores v0.15.1 bit for bit
    (np.array_equal against the frozen copy of the v0.15.1 body above) on the SP slab, a synthetic
    three-state well with an occupancy floor, and the plain two-band well; and it warns (deprecated)."""
    from dynameta.carriers.schrodinger_poisson import SubbandResult
    from dynameta.warning_types import AdvisoryWarning
    cases = [(_isb_slab(6e-9), _ISB_EPS_INF, _ISB_M, _ISB_GAM, _ISB_GAM, 0.0),
             (_two_band_subband([5e17, 1e17]), EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, 0.0)]
    three = _two_band_subband([5e17, 1e17])
    z = three.z_m
    p3 = np.sin(3 * np.pi * z / (z[-1] - z[0]))
    p3 = p3 / np.sqrt(np.sum(p3 ** 2) * (z[1] - z[0]))
    three = SubbandResult(energies_J=np.array([0.0, 0.95, 2.1]) * 1.602176634e-19,
                          psi=np.column_stack([three.psi, p3]), z_m=z,
                          sheet_density_m2=np.array([5e17, 1e17, 1e17]))
    cases.append((three, EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, 1e10))
    for res, ei, mo, gi, ge, floor in cases:
        for lam in (1200e-9, 1550e-9):
            mod = IntersubbandEffect(ei, mo, gi, ge, occ_floor_m2=floor, z_response="drude_plus_subbands")
            with pytest.warns(AdvisoryWarning, match="deprecated"):
                got = mod.eps({"subband": res}, lam)
            assert np.array_equal(got, _isb_v0151_reference(res, lam, ei, mo, gi, ge, floor))


def test_intersubband_free_population_is_an_isotropic_drude_and_legacy_refuses_it():
    """free_density_m3 (a population NOT in the sub-band set) adds the same Drude term to eps_xx and
    eps_zz (closed form re-derived here); the legacy form refuses it."""
    from dynameta.constants import C_LIGHT, EPS0, Q_E
    res = _two_band_subband([5e17, 0.0])
    base = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off").eps(
        {"subband": res}, 1200e-9)
    nf = 2e26
    got = IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, fsum_check="off",
                             free_density_m3=nf).eps({"subband": res}, 1200e-9)
    w = 2.0 * np.pi * C_LIGHT / 1200e-9
    chi = -(nf * Q_E ** 2 / (EPS0 * M_OPT)) / (w * w + 1j * w * GAM_INTRA)
    for k in range(3):
        assert abs((got[k, k] - base[k, k]) - chi) < 1e-12 * abs(chi)
    with pytest.raises(ValueError, match="free_density_m3"):
        IntersubbandEffect(EPS_INF, M_OPT, GAM_INTRA, GAM_INTER, free_density_m3=nf,
                           z_response="drude_plus_subbands").eps({"subband": res}, 1200e-9)


def test_intersubband_fsum_diagnostic_flags_a_truncated_set_and_the_mass_choice():
    """The f-sum check (AdvisoryWarning outside [0.9, 1.05]) fires for a truncated state set (the
    6 nm slab cut to its 3 lowest states: MEASURED ratio 0.372; keeping 6 states already gives 0.981 --
    the check catches gross truncation, not fine detail) and is silenced by fsum_check='off'; subband_mass_kg
    scales the ratio linearly (f_ij is proportional to it). A non-uniform grid raises (the Riemann
    quadrature needs one spacing).
    Behaviour change (an incomplete set is an error, not a loosened bar): a ratio below 0.5 now RAISES
    instead of warning, so the 3-state cut (0.372) is refused; the warning band [0.5, 0.9) is exercised with
    the complete set and a 0.8x subband_mass_kg (ratio 0.80)."""
    from dynameta.carriers.schrodinger_poisson import SubbandResult
    from dynameta.warning_types import AdvisoryWarning
    full = _isb_slab(6e-9)
    trunc = SubbandResult(energies_J=full.energies_J[:3], psi=full.psi[:, :3], z_m=full.z_m,
                          sheet_density_m2=full.sheet_density_m2[:3])
    m = _isb_model()
    with pytest.raises(ValueError, match="f-sum"):
        m.eps({"subband": trunc}, _ISB_LAM)
    assert m.last_fsum_ratio == pytest.approx(0.3719, abs=1e-3)
    with pytest.warns(AdvisoryWarning, match="f-sum"):
        mw = _isb_model(subband_mass_kg=0.8 * _ISB_M)
        mw.eps({"subband": full}, _ISB_LAM)
    assert 0.5 < mw.last_fsum_ratio < 0.9
    quiet = _isb_model(fsum_check="off")
    quiet.eps({"subband": trunc}, _ISB_LAM)                    # no warning (filterwarnings=error)
    a, b = _isb_model(), _isb_model(subband_mass_kg=1.02 * _ISB_M)
    a.eps({"subband": full}, _ISB_LAM)
    b.eps({"subband": full}, _ISB_LAM)
    assert b.last_fsum_ratio == pytest.approx(1.02 * a.last_fsum_ratio, rel=1e-12, abs=0)   # f_ij linear in m_sp
    z = full.z_m.copy()
    z[5] += 0.3 * (z[1] - z[0])
    bad = SubbandResult(energies_J=full.energies_J, psi=full.psi, z_m=z,
                        sheet_density_m2=full.sheet_density_m2)
    with pytest.raises(ValueError, match="uniform"):
        _isb_model().eps({"subband": bad}, _ISB_LAM)
    with pytest.raises(ValueError, match="z_response"):
        _isb_model(z_response="drude").eps({"subband": full}, _ISB_LAM)


def test_intersubband_default_refuses_an_incomplete_subband_set():
    """An incomplete sub-band set is refused. SchrodingerPoisson1D.density's default output keeps only the states
    needed for the density -- for a 6 nm flat-band film the occupied sub-band alone -- so the f-sum ratio is 0 and
    the default IntersubbandEffect returned eps_zz = eps_inf = 4.0+0j EXACTLY with only an AdvisoryWarning.
    Now a ratio below 0.5 raises; the complete set (the class docstring's example) passes silently with ratio
    0.9997 and an eps_zz far from eps_inf (MEASURED |eps_zz - eps_inf| = 1.16); fsum_check='off' keeps the
    deliberate model-study opt-out."""
    import warnings
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, sp_bulk_fermi_level_J
    m = _ISB_M
    z = np.linspace(0.0, 6e-9, 241)
    sp = SchrodingerPoisson1D(z, m, T_K=300.0)
    ef = sp_bulk_fermi_level_J(_ISB_NBG, m, 300.0)
    short = sp.density(np.zeros_like(z), ef)
    with pytest.raises(ValueError, match="f-sum ratio"):
        IntersubbandEffect(_ISB_EPS_INF, m, _ISB_GAM, _ISB_GAM).eps({"subband": short}, _ISB_LAM)
    off = IntersubbandEffect(_ISB_EPS_INF, m, _ISB_GAM, _ISB_GAM, fsum_check="off")
    assert off.eps({"subband": short}, _ISB_LAM)[2, 2] == _ISB_EPS_INF + 0j and off.last_fsum_ratio < 0.5
    full = sp.density(np.zeros_like(z), ef, n_states=None, bound_tol=np.inf)       # the docstring example
    mdl = IntersubbandEffect(_ISB_EPS_INF, m, _ISB_GAM, _ISB_GAM)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e = mdl.eps({"subband": full}, _ISB_LAM)
    assert 0.99 < mdl.last_fsum_ratio < 1.0 and abs(e[2, 2] - _ISB_EPS_INF) > 0.5
