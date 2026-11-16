"""Gates for materials.oxide_scattering and materials.uniaxial_optical (TCO physics kernels).

Every kernel is checked against an oracle written HERE (closed forms re-derived independently of the
module's code path) and every check is shown to be discriminating: a deliberately wrong parameter or a
mutant formula is asserted to MISS the same oracle.

Run: python -m pytest tests/test_oxide_scattering.py -q
"""
import math

import numpy as np
import pytest

from dynameta.constants import EPS0, HBAR, KB, M_E, Q_E
from dynameta.materials import (DrudeOptical, KaneBand, KaneOpticalMass, Material, MatthiessenGamma,
                                ScatteringModel, TransportModel)
from dynameta.materials.oxide_scattering import (IonizedImpurityScattering, MasettiMobility,
                                                 OpticalMobilityDamping, PolycrystallineScattering,
                                                 ResolvedMatthiessenGamma, SetoGrainBarrier, TabulatedMobility,
                                                 series_mobility)
from dynameta.materials.uniaxial_optical import UniaxialOptical

CM3 = 1.0e6
CM2VS = 1.0e-4
NS = np.array([1e20, 5e20, 1e21]) * CM3


def _bh_dingle_parabolic(n, eps_r, m, Z=1.0, Ni=None):
    """Independent oracle: degenerate Brooks-Herring-Dingle mobility of a PARABOLIC band in hbar form,
    mu = 24 pi^3 eps^2 hbar^3 n / (e^3 m^2 Z^2 N_i F), F = ln(1 + xi) - xi / (1 + xi), xi = (2 k_F / q_TF)^2
    with the Thomas-Fermi wavevector q_TF^2 = 3 n e^2 / (2 eps E_F) of the degenerate gas."""
    Ni = n / Z if Ni is None else Ni
    eps = eps_r * EPS0
    kF = (3.0 * math.pi ** 2 * n) ** (1.0 / 3.0)
    EF = HBAR ** 2 * kF ** 2 / (2.0 * m)
    qTF2 = 3.0 * n * Q_E ** 2 / (2.0 * eps * EF)
    xi = 4.0 * kF ** 2 / qTF2
    F = np.log1p(xi) - xi / (1.0 + xi)
    return 24.0 * math.pi ** 3 * eps ** 2 * HBAR ** 3 * n / (Q_E ** 3 * m ** 2 * Z ** 2 * Ni * F)


def test_ionized_impurity_parabolic_limit_is_brooks_herring_dingle():
    """alpha = 0: the Zawadzki-Pisarkiewicz kernel is the parabolic BH-Dingle result (oracle in hbar form,
    with the Thomas-Fermi screening parameter derived from E_F), to round-off; Z and N_i scale as 1/Z^2,
    1/N_i. Mutants -- hbar in place of h in the prefactor, the screening parameter without its factor 4 --
    miss the oracle by orders of magnitude / tens of percent."""
    m = 0.28 * M_E
    ii = IonizedImpurityScattering(eps_r=8.3, band=m)
    for n in NS:
        assert ii.mobility_m2Vs(n) == pytest.approx(_bh_dingle_parabolic(n, 8.3, m), rel=1e-12, abs=0.0)
        kF = (3.0 * math.pi ** 2 * n) ** (1.0 / 3.0)
        qTF2 = 3.0 * n * Q_E ** 2 / (2.0 * 8.3 * EPS0 * HBAR ** 2 * kF ** 2 / (2.0 * m))
        assert ii.screening_parameter(n) == pytest.approx(4.0 * kF ** 2 / qTF2, rel=1e-12, abs=0.0)
    n = NS[1]
    ii2 = IonizedImpurityScattering(eps_r=8.3, band=m, Z=2.0)              # N_i = n / 2 by default
    assert ii2.mobility_m2Vs(n) == pytest.approx(_bh_dingle_parabolic(n, 8.3, m, Z=2.0), rel=1e-12, abs=0.0)
    assert ii2.mobility_m2Vs(n) == pytest.approx(0.5 * ii.mobility_m2Vs(n), rel=1e-12, abs=0.0)   # 1/Z^2 x 1/N_i
    comp = IonizedImpurityScattering(eps_r=8.3, band=m, impurity_ratio=1.5)
    assert comp.mobility_m2Vs(n) == pytest.approx(ii.mobility_m2Vs(n) / 1.5, rel=1e-12, abs=0.0)
    # mutants of the oracle comparison
    mutant_hbar = ii.mobility_m2Vs(n) / (2.0 * math.pi) ** 3                # h -> hbar in h^3
    assert mutant_hbar != pytest.approx(_bh_dingle_parabolic(n, 8.3, m), rel=0.5, abs=0.0)
    xi = ii.screening_parameter(n)
    F_mut = np.log1p(xi / 4.0) - (xi / 4.0) / (1.0 + xi / 4.0)
    mut = ii.mobility_m2Vs(n) * ii.screening_function(n) / F_mut
    assert mut != pytest.approx(_bh_dingle_parabolic(n, 8.3, m), rel=0.1, abs=0.0)


def test_ionized_impurity_small_xi_limit_and_rate():
    """Weak-screening-length limit xi -> 0 (low density): F -> xi^2 / 2, so mu -> 48 pi^3 eps^2 hbar^3 /
    (e^3 m^2 xi^2) for N_i = n (density-independent prefactor times xi^-2 ~ n^-2/3). The rate is
    e / (m* mu). A wrong exponent (xi instead of xi^2) misses."""
    m = 0.3 * M_E
    ii = IonizedImpurityScattering(eps_r=8.0, band=m)
    n = 1e18                                                                # m^-3: xi ~ 1e-2
    xi = float(ii.screening_parameter(n))
    assert xi < 0.05
    eps = 8.0 * EPS0
    asym = 48.0 * math.pi ** 3 * eps ** 2 * HBAR ** 3 / (Q_E ** 3 * m ** 2 * xi ** 2)
    assert ii.mobility_m2Vs(n) == pytest.approx(asym, rel=2.0 * xi, abs=0.0)          # next order O(xi)
    assert ii.mobility_m2Vs(n) != pytest.approx(asym * xi, rel=0.5, abs=0.0)
    assert ii(n) == pytest.approx(Q_E / (m * ii.mobility_m2Vs(n)), rel=1e-14, abs=0.0)


def test_ionized_impurity_nonparabolic_independent_evaluation():
    """Nonparabolic branch against an INDEPENDENT evaluation of Ellmer & Mientus (2008) Eqs. (1)-(2) for ZnO
    (eps_r 8.3, m0* 0.28 m_e): 145 / 107 / 95 (parabolic), 130 / 79 / 61 (alpha 0.29) and 121 / 67 / 49
    (alpha 0.5) cm^2/(V s) at 1e20 / 5e20 / 1e21 cm^-3 (computed by hand from the printed equations,
    not by this module). Nonparabolicity lowers the mobility and the effect grows with density. A mutant
    that drops the nonparabolic correction of F (xi_np = 0 but m*(n) kept) misses the alpha = 0.5 set."""
    cases = {0.0: (145.0, 107.0, 95.0), 0.29: (130.0, 79.0, 61.0), 0.5: (121.0, 67.0, 49.0)}
    for alpha, ref in cases.items():
        ii = IonizedImpurityScattering(eps_r=8.3, band=KaneBand(0.28 * M_E, alpha))
        got = ii.mobility_m2Vs(NS) / CM2VS
        assert got == pytest.approx(np.array(ref), abs=1.0), alpha
    ii = IonizedImpurityScattering(eps_r=8.3, band=KaneBand(0.28 * M_E, 0.5))
    km = KaneOpticalMass(0.28 * M_E, 0.5)
    m = km(NS)
    xi = ii.screening_parameter(NS)
    F0 = np.log1p(xi) - xi / (1.0 + xi)
    mutant = 3.0 * (8.3 * EPS0) ** 2 * (2 * math.pi * HBAR) ** 3 / (m ** 2 * Q_E ** 3) / F0 / CM2VS
    assert not np.allclose(mutant, [121.0, 67.0, 49.0], atol=3.0)


def test_masetti_zno_fit_values_and_limits():
    """Ellmer & Mientus (2008) Table 2, ZnO: 54.3 / 50.3 / 34.6 / 18.3 cm^2/(V s) at 1e20 / 2e20 / 5e20 /
    1e21 cm^-3 (independent hand evaluation); n -> 0 gives mu_max, n -> inf gives mu_min - mu_1 = 5. A
    swapped n_ref2 (6e21) misses."""
    ma = MasettiMobility.zno_ellmer_mientus_2008()
    n = np.array([1e20, 2e20, 5e20, 1e21]) * CM3
    assert ma(n) / CM2VS == pytest.approx([54.3, 50.3, 34.6, 18.3], abs=0.1)
    assert ma(1e10) / CM2VS == pytest.approx(210.0, rel=1e-6, abs=0.0)
    assert ma(1e35) / CM2VS == pytest.approx(5.0, rel=1e-6, abs=0.0)
    bad = MasettiMobility(ma.mu_max_m2Vs, ma.mu_min_m2Vs, ma.mu_1_m2Vs, ma.n_ref1_m3, ma.alpha1, 6e21 * CM3, 2.0)
    assert bad(1e21 * CM3) / CM2VS != pytest.approx(18.3, abs=1.0)
    with pytest.raises(ValueError, match="mu_1"):
        MasettiMobility(0.02, 0.005, 0.006, 1e23, 1.0, 1e27, 2.0)


def test_seto_barrier_branches_continuity_and_limits():
    """Seto: E_b = e^2 Q^2 / (8 eps n) above n = Q/L and e^2 L^2 n / (8 eps) below; continuous at n = Q/L,
    where it peaks; Q = 0 gives mu_0 = e L / sqrt(2 pi m kT) (oracle here). Mutant: the partly-depleted
    branch with Q instead of Q^2 misses."""
    Q, L, er, m, T = 3e17, 30e-9, 8.3, 0.30 * M_E, 300.0
    gb = SetoGrainBarrier(trap_density_m2=Q, grain_size_m=L, eps_r=er, mass_kg=m, T_K=T)
    eps = er * EPS0
    n_hi = 5e26
    assert gb.barrier_height_J(n_hi) == pytest.approx(Q_E ** 2 * Q ** 2 / (8 * eps * n_hi), rel=1e-13, abs=0.0)
    assert gb.barrier_height_J(n_hi) / Q_E == pytest.approx(0.049, abs=0.002)        # ~0.05 eV at 5e20
    n_lo = 1e24
    assert gb.barrier_height_J(n_lo) == pytest.approx(Q_E ** 2 * L ** 2 * n_lo / (8 * eps), rel=1e-13, abs=0.0)
    nc = Q / L
    a, b = gb.barrier_height_J(nc * (1 - 1e-9)), gb.barrier_height_J(nc * (1 + 1e-9))
    assert a == pytest.approx(b, rel=1e-6, abs=0.0)
    grid = np.geomspace(nc / 10, nc * 10, 201)
    assert grid[int(np.argmax(gb.barrier_height_J(grid)))] == pytest.approx(nc, rel=0.03, abs=0.0)
    mu0 = Q_E * L / math.sqrt(2 * math.pi * m * KB * T)
    free = SetoGrainBarrier(trap_density_m2=0.0, grain_size_m=L, eps_r=er, mass_kg=m, T_K=T)
    assert free.mobility_m2Vs(n_hi) == pytest.approx(mu0, rel=1e-13, abs=0.0)
    expect = mu0 * math.exp(-gb.barrier_height_J(n_hi) / (KB * T))
    assert gb.mobility_m2Vs(n_hi) == pytest.approx(expect, rel=1e-13, abs=0.0)
    mutant = Q_E ** 2 * Q / (8 * eps * n_hi)
    assert gb.barrier_height_J(n_hi) != pytest.approx(mutant, rel=0.5, abs=0.0)
    with pytest.raises(ValueError, match="metres"):
        SetoGrainBarrier(trap_density_m2=Q, grain_size_m=30.0, eps_r=er, mass_kg=m)


def test_resolved_matthiessen_reduces_to_parent_and_sums_channels():
    """ionized_impurity=None: identical (bit for bit) to MatthiessenGamma with the same fields; with the
    resolved channel: optical_dc_ratio x (const + phonon + e/(m* mu_ii)). Setting both impurity channels
    is refused."""
    base = dict(gamma_const_rad_s=2e13, gamma_phonon_300K_rad_s=3e13, T_K=350.0, debye_T_K=600.0,
                optical_dc_ratio=0.8)
    n = np.array([1e26, 4e26, 9e26])
    assert np.array_equal(ResolvedMatthiessenGamma(**base)(n), MatthiessenGamma(**base)(n))
    ii = IonizedImpurityScattering(eps_r=8.7, band=KaneBand(0.24 * M_E, 0.33))
    g = ResolvedMatthiessenGamma(ionized_impurity=ii, **base)
    phon = MatthiessenGamma(**base)(n) / 0.8 - 2e13
    assert g(n) == pytest.approx(0.8 * (2e13 + phon + ii(n)), rel=1e-13, abs=0.0)
    with pytest.raises(ValueError, match="twice"):
        ResolvedMatthiessenGamma(ionized_impurity=ii, bh_prefactor_rad_s=1e13)


def test_polycrystalline_scattering_optical_vs_dc():
    """No grain boundary, no separate intra-grain mobility: PolycrystallineScattering IS ScatteringModel.
    With a Seto barrier the DC mobility is the series combination, while the optical damping (the intra-
    grain rate) is untouched -- the optical (intra-grain) vs Hall mobility split of polycrystalline
    films. Through Material(scattering=...)."""
    km = KaneOpticalMass(0.24 * M_E, 0.33)
    damp = OpticalMobilityDamping(mobility_m2Vs=38e-4, mass=km)
    n = np.array([2e26, 5e26, 1e27])
    plain = ScatteringModel(one_over_tau=damp, m_cond_kg=km)
    poly = PolycrystallineScattering(one_over_tau=damp, m_cond_kg=km)
    assert np.array_equal(poly.mobility_of_n()(n), plain.mobility_of_n()(n))
    assert poly.mobility_of_n()(n) == pytest.approx(38e-4, rel=1e-13, abs=0.0)              # e / (m gamma) = mu_opt
    gb = SetoGrainBarrier(trap_density_m2=3e17, grain_size_m=30e-9, eps_r=8.7, mass_kg=km)
    mig = MasettiMobility.zno_ellmer_mientus_2008()
    full = PolycrystallineScattering(one_over_tau=damp, m_cond_kg=km, intragrain_mobility=mig, grain_boundary=gb,
                                     hall_factor=1.1)
    expect = 1.1 / (1.0 / mig(n) + 1.0 / gb(n))
    assert full.mobility_of_n()(n) == pytest.approx(expect, rel=1e-13, abs=0.0)
    tr = TransportModel(n_bg_m3=5e26, eps_static=8.7, band=KaneBand(0.24 * M_E, 0.33))
    mat = Material("poly", DrudeOptical(eps_inf=3.6, m_opt_kg=km, gamma_rad_s=1.5e14), transport=tr, scattering=full)
    assert mat.optical.gamma_rad_s(n) == pytest.approx(damp(n), rel=0, abs=0.0)
    assert np.all(mat.transport.mobility_m2Vs_of_n_m3(n) < 38e-4)
    assert series_mobility(2.0, 2.0) == 1.0


def test_optical_mobility_damping_and_tabulated_mobility():
    """gamma = e / (m_opt(n) mu); a TabulatedMobility interpolates linearly in log10(n), holds the end values
    (or raises with extrapolate='raise'). Unit slips are refused."""
    km = KaneOpticalMass(0.10 * M_E, 0.735)
    tab = TabulatedMobility(n_m3=(1e25, 1e26, 1e27), mobility_m2Vs=(0.03, 0.04, 0.02))
    assert tab(np.sqrt(1e25 * 1e26)) == pytest.approx(0.035, rel=1e-13, abs=0.0)
    assert tab(1e24) == 0.03 and tab(1e28) == 0.02
    with pytest.raises(ValueError, match="outside"):
        TabulatedMobility(n_m3=(1e25, 1e26), mobility_m2Vs=(0.03, 0.04), extrapolate="raise")(1e27)
    d = OpticalMobilityDamping(mobility_m2Vs=tab, mass=km)
    n = np.array([3e25, 3e26])
    assert d(n) == pytest.approx(Q_E / (km(n) * tab(n)), rel=1e-14, abs=0.0)
    with pytest.raises(ValueError):
        OpticalMobilityDamping(mobility_m2Vs=-1.0, mass=km)
    with pytest.raises(ValueError):
        TabulatedMobility(n_m3=(1e26, 1e25), mobility_m2Vs=(0.03, 0.04))


def test_uniaxial_optical_tensor_and_assembler_path():
    """UniaxialOptical: diag(eps_o, eps_o, eps_e) with each component from its own model; the scalar eps()
    refuses; the permittivity assembler's EffectModel adapter (OpticalModelEffect) reaches the tensor
    through Material.field_response."""
    from dynameta.core.effects import OpticalModelEffect
    o = DrudeOptical(eps_inf=3.70, m_opt_kg=0.30 * M_E, gamma_rad_s=1.5e14)
    e = DrudeOptical(eps_inf=3.78, m_opt_kg=0.30 * M_E, gamma_rad_s=1.5e14)
    u = UniaxialOptical(ordinary=o, extraordinary=e)
    n = np.array([3e26, 5e26])
    t = u.eps_tensor(1.5e-6, n_m3=n)
    assert t.shape == (2, 3, 3)
    assert np.array_equal(t[:, 0, 0], o.eps(1.5e-6, n_m3=n)) and np.array_equal(t[:, 1, 1], t[:, 0, 0])
    assert np.array_equal(t[:, 2, 2], e.eps(1.5e-6, n_m3=n))
    assert t[:, 2, 2] - t[:, 0, 0] == pytest.approx(np.full(2, 0.08 + 0j), abs=1e-12)
    assert np.all(t[:, 0, 1] == 0)
    with pytest.raises(ValueError, match="UNIAXIAL"):
        u.eps(1.5e-6, n_m3=n)
    mat = Material("u", u)
    got = OpticalModelEffect(mat).eps({"n": n}, 1.5e-6)
    assert np.array_equal(got, t)
    with pytest.raises(TypeError):
        from dynameta.materials import BoundaryScattering
        UniaxialOptical(ordinary=DrudeOptical(3.7, 0.30 * M_E, 1.5e14, boundary=BoundaryScattering(thickness_m=8e-9)),
                        extraordinary=e)
