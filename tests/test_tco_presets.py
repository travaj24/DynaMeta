"""Gates for materials.tco_presets: AZO, GZO, In:CdO and ITO material presets (FTO: tests/test_enz_presets.py).

Each preset is compared with PUBLISHED reference points (source cited per test; full references in the
module docstring of dynameta.materials.tco_presets), within the stated literature spread, and every
comparison is shown to fail for a deliberately wrong parameter. Kramers-Kronig consistency is checked with
the existing check_kk diagnostic (falls under grid refinement for the causal presets; a sign-flipped
mutant does not).

Run: python -m pytest tests/test_tco_presets.py -q
"""
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from dynameta.constants import C_LIGHT, EPS0, M_E, Q_E
from dynameta.materials import check_kk
from dynameta.materials import tco_presets as TP
from dynameta.warning_types import AdvisoryWarning

CM3 = 1.0e6


def _trap(y, x):
    """Trapezoid rule (numpy-version independent)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _lam_enz(mat, n_m3, lo=0.5e-6, hi=8e-6):
    """Wavelength [m] of Re(eps) = 0 (the screened plasma / ENZ wavelength) of a Material at density n."""
    return brentq(lambda L: complex(mat.eps(L, n_m3=n_m3)).real, lo, hi, xtol=1e-15)


# ------------------------------------------------------------------------------------------------ build --

@pytest.mark.parametrize("factory, n", [(TP.azo, 5e26), (TP.gzo, 6e26), (TP.in_cdo, 2.8e26), (TP.ito, 3e26)])
def test_every_preset_builds_a_complete_material(factory, n):
    """Optical (Drude, Kane mass of the transport band, damping from the optical mobility), transport (Kane
    equilibrium electrostatics) and scattering (DC mobility) all present and consistent."""
    from dynameta.materials import DrudeOptical, KaneBand, KaneOpticalMass
    mat = factory(n)
    assert isinstance(mat.optical, DrudeOptical) and mat.is_semiconductor
    band = mat.transport.band
    assert isinstance(band, KaneBand) and mat.transport.physics == "equilibrium"
    assert mat.optical.m_opt_kg == KaneOpticalMass.from_band(band)        # one band for optics and charge
    g = mat.optical.gamma_rad_s(n)
    mu_opt = mat.optical.gamma_rad_s.mobility(n)
    assert g == pytest.approx(Q_E / (float(mat.optical.m_opt_kg(n)) * mu_opt), rel=1e-14, abs=0.0)
    assert mat.transport.mobility_m2Vs_of_n_m3(n) > 0.0
    assert mat.transport.chi_reference_T_K == 300.0
    e = complex(mat.eps(1.55e-6, n_m3=n))
    assert e.imag > 0.0 and np.isfinite(e.real)
    assert repr(mat.optical) == repr(factory(n).optical)                    # deterministic (cache key)
    kw = TP.dual_gate_film_kwargs(mat)
    assert kw == {"n_bg_m3": n, "m_eff_kg": band.m0_kg, "eps_static": mat.transport.eps_static,
                  "alpha_np_per_eV": band.alpha_per_eV}
    assert TP.PARAMETER_TABLES[factory.__name__]


def test_validity_ranges_refuse_with_clear_errors():
    """Outside the literature range: ValueError (strict) or AdvisoryWarning (strict=False); a density given
    in cm^-3 and a mobility given in cm^2/(V s) are recognised as unit slips."""
    with pytest.raises(ValueError, match="looks like a density in cm\\^-3"):  # not the range message
        TP.azo(5e20)
    with pytest.warns(AdvisoryWarning):                                     # below the range, not a slip
        TP.azo(5e24, strict=False)
    with pytest.raises(ValueError, match="outside the preset's literature range"):
        TP.gzo(3e27)
    with pytest.warns(AdvisoryWarning, match="extrapolating"):
        TP.gzo(3e27, strict=False)
    with pytest.raises(ValueError, match="default mobility table"):
        TP.in_cdo(1e27)                                                     # mass law ok, mobility table not
    TP.in_cdo(1e27, optical_mobility_m2Vs=100e-4)
    with pytest.raises(ValueError, match="implausible"):
        TP.ito(3e26, optical_mobility_m2Vs=30.0)
    with pytest.raises(ValueError, match="background"):
        TP.in_cdo(2e26, background="tabulated")


# ------------------------------------------------------------------------------------- reference points --

def test_gzo_crossover_wavelengths_mbe_films():
    """Ball et al., J. Phys. Photonics 5, 024001 (2023), Table 1 (MBE GZO): cross-over 1560 / 1700 / 1710 nm
    at 6.07 / 5.28 / 5.17e20 cm^-3. Preset with its default eps_inf 3.6: within 2 % (the +/-5 % spread of
    the doped-ZnO mass laws moves the wavelength by +/-2.5 %); with the films' own fitted eps_inf (3.5 /
    3.7 / 3.7): within 0.5 %. eps'' at the cross-over: Ball 0.31 / 0.41 / 0.31, preset within 0.12 (the
    spread between Ball's two films at 5.2-5.3e20). Mutants: a parabolic band (alpha = 0) puts the
    wavelength ~19 % short; an ALD-quality optical mobility (15 cm^2/(V s)) doubles eps''."""
    pts = [(6.07e20, 1560e-9, 3.5, 0.31), (5.28e20, 1700e-9, 3.7, 0.41), (5.17e20, 1710e-9, 3.7, 0.31)]
    for n_cm3, lam, einf, e2 in pts:
        n = n_cm3 * CM3
        assert _lam_enz(TP.gzo(n), n) == pytest.approx(lam, rel=0.02, abs=0.0)
        assert _lam_enz(TP.gzo(n, eps_inf=einf), n) == pytest.approx(lam, rel=0.005, abs=0.0)
        assert complex(TP.gzo(n, eps_inf=einf).eps(lam, n_m3=n)).imag == pytest.approx(e2, abs=0.12)
        assert _lam_enz(TP.gzo(n, alpha_per_eV=0.0), n) != pytest.approx(lam, rel=0.1, abs=0.0)
        bad = complex(TP.gzo(n, eps_inf=einf, optical_mobility_m2Vs=15e-4).eps(lam, n_m3=n)).imag
        assert bad != pytest.approx(e2, abs=0.25)


def test_azo_reference_points_pld_films():
    """Naik et al., PNAS 109, 8834 (2012): PLD AZO (~4.3e20 cm^-3, density estimated by the authors) metallic
    beyond 1.84 um -> preset ENZ within 2 %. Kinsey et al., Optica 2, 616 (2015): PLD AZO 9.8e20 cm^-3,
    eps(1.3 um) = 0.22 + 0.26i -> preset Re within 0.3 (= ~2.5 % in ENZ wavelength), Im within 15 %.
    Mutant: a heavier band edge, m0* = 0.40 m_e, misses both."""
    n = 4.3e20 * CM3
    assert _lam_enz(TP.azo(n), n) == pytest.approx(1.84e-6, rel=0.02, abs=0.0)
    assert _lam_enz(TP.azo(n, m0_ratio=0.40), n) != pytest.approx(1.84e-6, rel=0.08, abs=0.0)
    n = 9.8e20 * CM3
    e = complex(TP.azo(n).eps(1.3e-6, n_m3=n))
    assert e.real == pytest.approx(0.22, abs=0.3)
    assert e.imag == pytest.approx(0.26, rel=0.15, abs=0.0)
    eb = complex(TP.azo(n, m0_ratio=0.40).eps(1.3e-6, n_m3=n))
    assert eb.real != pytest.approx(0.22, abs=0.6)


def test_zno_optical_mass_law_against_enz_implied_masses():
    """The default ZnO band (0.24 m_e, 0.33 eV^-1; Romanyuk et al. 2016) against the masses implied by
    published ENZ points through m* = N e^2 / (eps0 eps_inf omega_ENZ^2) with the films' eps_inf (Ball et
    al. 2023): within 3 %; and against the competing peer-reviewed laws (Ruske 2009: 0.24 / 0.27; Kim 2008:
    0.247 / 0.33): within 7 % over 1e20-1e21 cm^-3 (the spread
    between those laws: 6.7 % at 1e21 cm^-3)."""
    from dynameta.materials import KaneOpticalMass
    km = TP.gzo(6e26).optical.m_opt_kg
    for n_cm3, lam, einf in [(6.07e20, 1560e-9, 3.5), (5.28e20, 1700e-9, 3.7), (5.17e20, 1710e-9, 3.7)]:
        n = n_cm3 * CM3
        w = 2.0 * math.pi * C_LIGHT / lam
        m_implied = n * Q_E ** 2 / (EPS0 * einf * w ** 2)
        assert float(km(n)) / M_E == pytest.approx(m_implied / M_E, rel=0.03, abs=0.0)
    ns = np.array([1e20, 5e20, 1e21]) * CM3
    for m0, a in ((0.24, 0.27), (0.247, 0.33)):
        assert km(ns) / M_E == pytest.approx(KaneOpticalMass(m0 * M_E, a)(ns) / M_E, rel=0.07, abs=0.0)


def test_in_cdo_enz_wavelengths_and_background():
    """In:CdO preset (HiPIMS-type: Nolen et al. 2020 band, Cleri et al. 2021 mobility, band-filling
    background) against:
      * Nolen et al., Phys. Rev. Materials 4, 025202 (2020): ENZ 3.87 / 2.72 / 1.92 um at 6.6e19 / 1.4e20 /
        4.1e20 cm^-3 (1e4 / their screened plasma wavenumbers) -> within 7 % (the 1.4e20 film sits 6 % off the
        paper's own mass law);
      * Yang et al., Nat. Photonics 11, 390 (2017): 2.10 um at 2.8e20 -> within 7 % (they used m* = 0.20 m_e,
        the preset's band gives 0.24: the published spread);
      * Liu et al., Phys. Rev. Applied 6, 064018 (2016), INDEPENDENT polycrystalline sputtered films:
        screened plasma wavelength 1.70 / 1.58 / 1.43 / 1.29 um at 5.4 / 6.6 / 8.9 / 11e20 -> within 2.5 %.
    Mutant: without the density-dependent background (constant eps_inf 5.5) the Liu points are 5-9 % long."""
    for n_cm3, lam, tol in [(6.6e19, 3.87e-6, 0.07), (1.4e20, 2.72e-6, 0.07), (4.1e20, 1.92e-6, 0.07),
                            (2.8e20, 2.10e-6, 0.07)]:
        n = n_cm3 * CM3
        assert _lam_enz(TP.in_cdo(n), n, lo=0.8e-6) == pytest.approx(lam, rel=tol, abs=0.0)
    for n_cm3, lam in [(5.4e20, 1.70e-6), (6.6e20, 1.58e-6), (8.9e20, 1.43e-6), (1.1e21, 1.29e-6)]:
        n = n_cm3 * CM3
        mat = TP.in_cdo(n, optical_mobility_m2Vs=100e-4)
        assert _lam_enz(mat, n, lo=0.8e-6) == pytest.approx(lam, rel=0.025, abs=0.0)
        flat = TP.in_cdo(n, optical_mobility_m2Vs=100e-4, background=None)
        assert _lam_enz(flat, n, lo=0.8e-6) != pytest.approx(lam, rel=0.04, abs=0.0)


def test_in_cdo_background_follows_measured_eps_inf_and_optical_gap():
    """The band-filling background: eps_b falls 5.5 -> 5.1 between 1.1e19 and 4.1e20 cm^-3 at 2 um (Nolen
    2020, the calibration of the edge amplitude -- a consistency check), monotonically; the model's
    absorption onset (renormalized gap + band filling) against the MEASURED optical gap of Cleri et al.
    (2021) Table I, 2.28 / 2.51 / 2.73 / 2.86 eV at 2.2e19 / 1.2e20 / 2.9e20 / 5.0e20 cm^-3 -- not used in
    the calibration: within 0.15 eV. Mutant: no gap renormalization misses by > 0.3 eV at 5e20."""
    import dataclasses
    bg = TP.in_cdo(2e26).optical.background
    eb = [float(bg.eps_background(n * CM3, 2e-6, 5.5)) for n in (1.1e19, 1e20, 4.1e20, 1e21)]
    assert eb[0] == pytest.approx(5.5, abs=1e-12) and eb[2] == pytest.approx(5.1, abs=0.005)
    assert np.all(np.diff(eb) < 0.0)                                        # same sign as the Drude response
    ref = [(2.2e19, 2.28), (1.2e20, 2.51), (2.9e20, 2.73), (5.0e20, 2.86)]
    nobgr = dataclasses.replace(bg, bgr_coeff_eV_m=0.0)
    for n_cm3, eg in ref:
        assert float(bg.optical_gap_J(n_cm3 * CM3)) / Q_E == pytest.approx(eg, abs=0.15)
    assert float(nobgr.optical_gap_J(5e20 * CM3)) / Q_E != pytest.approx(2.86, abs=0.3)


def test_in_cdo_mobility_and_low_loss():
    """Hall mobility of the default table at its own points (Cleri et al. 2021, Table I) and, at Nolen et
    al.'s densities, within 10 % of their 389 / 312 cm^2/(V s); CdO's ENZ loss Im(eps) at 2.8e20 cm^-3 is
    more than 2.5x below doped ZnO's and ITO's at their ENZ (0.14 vs 0.42 / 0.4: the high-mobility, low-loss
    character; the ~9x higher mobility is partly offset by CdO's larger eps_inf and lower ENZ frequency)."""
    mat = TP.in_cdo(2.9e26)
    assert mat.transport.mobility_m2Vs_of_n_m3(2.9e26) == pytest.approx(331e-4, rel=1e-12, abs=0.0)
    for n_cm3, mu in ((1.4e20, 389e-4), (4.1e20, 312e-4)):
        assert mat.transport.mobility_m2Vs_of_n_m3(n_cm3 * CM3) == pytest.approx(mu, rel=0.10, abs=0.0)
    def loss_at_enz(m, n, lo=0.5e-6):
        return complex(m.eps(_lam_enz(m, n, lo=lo), n_m3=n)).imag
    cdo = loss_at_enz(TP.in_cdo(2.8e26), 2.8e26, lo=0.8e-6)
    assert cdo < loss_at_enz(TP.azo(5e26), 5e26) / 2.5 and cdo < loss_at_enz(TP.ito(5e26), 5e26) / 2.5


def test_ito_mass_law_against_published_masses():
    """Markwitz et al., Phys. Rev. B 109, 115201 (2024): optical mass 0.33-0.44 m_e over 2.9e20-1.01e21 cm^-3
    from their (0.21, 0.52) law (their Eq. (5) gives 0.337 / 0.452) -> preset within 3 % of the stated range
    ends; the crystalline alternative (0.18, 0.5; Feneberg et al. 2016, '~0.4 m_e at 1e21') within 3 %.
    Mutant: alpha = 0 is 37 % light at 1e21."""
    km = TP.ito(3e26).optical.m_opt_kg
    assert float(km(2.9e26)) / M_E == pytest.approx(0.33, rel=0.03, abs=0.0)
    assert float(km(1.01e27)) / M_E == pytest.approx(0.44, rel=0.03, abs=0.0)
    alt = TP.ito(3e26, m0_ratio=0.18, alpha_per_eV=0.5).optical.m_opt_kg
    assert float(alt(1e27)) / M_E == pytest.approx(0.40, rel=0.03, abs=0.0)
    flat = TP.ito(3e26, alpha_per_eV=0.0).optical.m_opt_kg
    assert float(flat(1.01e27)) / M_E != pytest.approx(0.44, rel=0.2, abs=0.0)
    g = TP.ito(3e26).optical.gamma_rad_s(3e26)                             # typical sputtered ITO ~1.8e14 rad/s
    assert g == pytest.approx(1.8e14, rel=0.15, abs=0.0)


def test_zno_dc_mobility_and_grain_boundary_option():
    """DC transport mobility: the Masetti intra-grain law by default (Ellmer & Mientus 2008), lowered by an
    optional Seto grain barrier while the optical damping stays the intra-grain value (optical > Hall
    mobility, as measured in polycrystalline ALD AZO by Romanyuk et al. 2016, ratio 1.55-2.38)."""
    from dynameta.materials.oxide_scattering import MasettiMobility
    n = 5e26
    m0 = TP.azo(n)
    assert m0.transport.mobility_m2Vs_of_n_m3(n) == pytest.approx(MasettiMobility.zno_ellmer_mientus_2008()(n),
                                                                  rel=1e-13, abs=0.0)
    m1 = TP.azo(n, grain_boundary=TP.zno_seto_barrier())
    assert m1.transport.mobility_m2Vs_of_n_m3(n) < m0.transport.mobility_m2Vs_of_n_m3(n)
    assert m1.optical.gamma_rad_s(n) == m0.optical.gamma_rad_s(n)
    ratio = m1.optical.gamma_rad_s.mobility(n) / m1.transport.mobility_m2Vs_of_n_m3(n)
    assert 1.0 < ratio < 3.0
    # the helper's defaults in SI, against Seto's closed form written here: Q_t = 3e13 cm^-2 = 3e17 m^-2,
    # L = 30 nm, eps_r 8.7, the band's optical mass at n, T_REF; DC = Masetti in series with mu_gb
    from dynameta.constants import KB, T_REF
    m_n = float(m1.optical.m_opt_kg(n))
    e_b = Q_E ** 2 * 3e17 ** 2 / (8.0 * EPS0 * 8.7 * n)
    mu_gb = Q_E * 30e-9 / math.sqrt(2.0 * math.pi * m_n * KB * T_REF) * math.exp(-e_b / (KB * T_REF))
    mu_ig = MasettiMobility.zno_ellmer_mientus_2008()(n)
    assert m1.transport.mobility_m2Vs_of_n_m3(n) == pytest.approx(1.0 / (1.0 / mu_ig + 1.0 / mu_gb), rel=1e-12, abs=0.0)


def test_zno_uniaxial_option():
    """uniaxial=True: diag(eps_o, eps_o, eps_e) with eps_inf,par - eps_inf,perp = 0.08 (Ashkenov et al. 2003)
    and the same carriers; the ENZ wavelength of the extraordinary component is ~1 % longer -- below the
    film-to-film spread of eps_inf (2.9-3.9), as documented."""
    n = 6e26
    mat = TP.gzo(n, uniaxial=True)
    t = mat.optical.eps_tensor(1.6e-6, n_m3=np.array([n]))
    assert t[0, 2, 2] - t[0, 0, 0] == pytest.approx(0.08, abs=1e-12)
    assert mat.transport.mobility_m2Vs_of_n_m3(n) == pytest.approx(TP.gzo(n).transport.mobility_m2Vs_of_n_m3(n),
                                                                   rel=1e-14, abs=0.0)
    lo, hi = (brentq(lambda L, m=m: complex(m.eps(L, n_m3=n)).real, 0.5e-6, 8e-6, xtol=1e-15)
              for m in (mat.optical.ordinary, mat.optical.extraordinary))
    assert 0.005 < hi / lo - 1.0 < 0.02


def test_wurtzite_polarization_charges_signs_and_magnitude():
    """Metal-polar ZnO on non-polar oxides: sigma_top = P_sp = -0.057 C/m^2, sigma_bottom = +0.057 (Bernardini
    et al. 1997); O-polar flips; mixed polarity (0.5) cancels; compensation scales; a polar neighbour
    reduces the step. Magnitude vs a gate: 8 nm of eps-25 oxide at 1.5 V induces 0.041 C/m^2."""
    sb, st = TP.wurtzite_polarization_charges(metal_polar_fraction=1.0)
    assert (sb, st) == pytest.approx((0.057, -0.057), abs=1e-15)
    assert TP.wurtzite_polarization_charges(metal_polar_fraction=0.0) == pytest.approx((-0.057, 0.057), abs=1e-15)
    assert TP.wurtzite_polarization_charges(metal_polar_fraction=0.5) == pytest.approx((0.0, 0.0), abs=1e-15)
    assert TP.wurtzite_polarization_charges(metal_polar_fraction=1.0, compensated_fraction=0.75) == pytest.approx(
        (0.01425, -0.01425), abs=1e-15)
    sb, st = TP.wurtzite_polarization_charges(metal_polar_fraction=1.0, P_neighbours_C_m2=(0.0, -0.06))
    assert st == pytest.approx(0.003, abs=1e-15)
    gate = EPS0 * 25.0 / 8e-9 * 1.5
    assert 1.0 < 0.057 / gate < 2.0
    with pytest.raises(ValueError):
        TP.wurtzite_polarization_charges(metal_polar_fraction=1.2)


def test_polarization_charge_drives_a_dipole_in_the_gated_film():
    """Through the existing GateStack interface-charge hook of the dual-gate Schrodinger-Poisson solver: the
    metal-polar ZnO bound charges accumulate electrons at the bottom face and deplete the top, and (a dipole)
    leave the film's net sheet charge unchanged to within the dead-layer reshuffle (< 5 % of the moved
    charge); a mutant with both charges positive is a net charge and fails that."""
    from dynameta.carriers.sp_dual_gate import DualGateSchrodingerPoisson, GateStack
    mat = TP.azo(5e26)
    kw = TP.dual_gate_film_kwargs(mat)
    sb, st = TP.wurtzite_polarization_charges(metal_polar_fraction=1.0)

    def solve(qb, qt):
        s = DualGateSchrodingerPoisson(t_film_m=8e-9, bottom=GateStack(((5e-9, 20.0),), interface_charge_C_m2=qb),
                                       top=GateStack(((5e-9, 20.0),), interface_charge_C_m2=qt), nz=321, **kw)
        return s.solve_voltages(0.0, 0.0)
    ref, pol, bad = solve(0.0, 0.0), solve(sb, st), solve(abs(sb), abs(st))
    k = len(ref.z_m) // 4
    assert _trap(pol.n_m3[:k], ref.z_m[:k]) > _trap(ref.n_m3[:k], ref.z_m[:k])
    assert _trap(pol.n_m3[-k:], ref.z_m[-k:]) < _trap(ref.n_m3[-k:], ref.z_m[-k:])
    moved = abs(sb) / Q_E
    assert abs(pol.sheet_excess_total_m2 - ref.sheet_excess_total_m2) < 0.05 * moved
    assert abs(bad.sheet_excess_total_m2 - ref.sheet_excess_total_m2) > 0.5 * moved


# ----------------------------------------------------------------------------------------- causality --

@pytest.mark.parametrize("factory, n", [(TP.azo, 5e26), (TP.gzo, 6e26), (TP.in_cdo, 2.8e26), (TP.ito, 5e26)])
def test_presets_are_kramers_kronig_consistent(factory, n):
    """check_kk on the free-carrier model of each preset (the In:CdO band-filling background is real below
    the gap and KK-consistent by construction -- its own gates -- and is evaluated below the gap only, so the
    wide KK grid uses background=None): rms_norm < 3e-4 at N = 8000 on a grid to 20 omega_p (measured 7.2e-5 /
    7.0e-5 / 1.7e-4 / 7.1e-5 for azo / gzo / in_cdo / ito: the residual of the band truncated at 20 omega_p, so the
    bound is ~2x the largest) and it falls under refinement (causal; 1-66 % from N = 8000 to 16000); the
    sign-flipped mutant sits at ~0.3 and does not fall."""
    kw = {"background": None} if factory is TP.in_cdo else {}
    mat = factory(n, **kw)
    wp = math.sqrt(n * Q_E ** 2 / (EPS0 * float(mat.optical.m_opt_kg(n))))

    class _Flip:
        def eps(self, lam, n_m3=None):
            return np.conj(mat.optical.eps(lam, n_m3=n_m3))
    res = {}
    for N in (8000, 16000):
        w = np.linspace(20.0 * wp / N, 20.0 * wp, N)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res[N] = (check_kk(mat.optical, w, n_m3=n, self_calib=False)["rms_norm"],
                      check_kk(_Flip(), w, n_m3=n, self_calib=False)["rms_norm"])
    assert res[8000][0] < 3e-4 and res[16000][0] < res[8000][0]
    assert res[8000][1] > 0.1 and res[16000][1] > 0.5 * res[8000][1]


# ------------------------------------------------------------------------------------------- packaging --

def test_lazy_exports_resolve_and_keep_the_package_import_light():
    """dynameta.materials re-exports the presets and their models lazily: a fresh `import dynameta.materials`
    loads none of the preset / lattice / scattering modules, dir() lists every lazy name, each name resolves
    to its defining module and `from dynameta.materials import *` binds them. An unknown name still raises
    AttributeError."""
    import pathlib
    import subprocess
    import sys
    code = "\n".join([
        "import sys, dynameta.materials as M",
        "mods = ['tco_presets', 'enz_presets', 'lattice_optical', 'oxide_scattering', 'uniaxial_optical']",
        "loaded = [m for m in mods if 'dynameta.materials.' + m in sys.modules]",
        "assert not loaded, loaded",
        "d = set(dir(M))",
        "assert set(M._LAZY_NAMES) <= d and set(M.__all__) <= d, sorted(set(M._LAZY_NAMES) - d)",
        "for name, mod in M._LAZY_NAMES.items():",
        "    assert getattr(M, name).__module__ == 'dynameta.materials.' + mod, name",
        "ns = {}",
        "exec('from dynameta.materials import *', ns)",
        "assert all(n in ns for n in M._LAZY_NAMES)",
        "try:",
        "    M.definitely_not_a_name",
        "except AttributeError:",
        "    print('clean')"])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(pathlib.Path(__file__).resolve().parents[1]))
    assert out.returncode == 0 and "clean" in out.stdout, out.stderr


def test_built_parameters_match_the_documented_defaults():
    """The Material each factory builds carries exactly the defaults its docstring, PARAMETER_TABLES and
    docs/enz_presets.md state (eps_inf, band, optical mobility, static permittivity, gap, affinity): a default
    edited in one place only fails here. Mutants: ZnO eps_static 8.7 -> 87 and ITO eps_inf 4.0 -> 3.0 (neither
    moves any reference-point test outside its literature tolerance) fail."""
    expect = {  # factory, n: eps_inf, m0/m_e, alpha [1/eV], mu_opt [m^2/(V s)] or None (table), eps_static, Eg, chi
        "azo": (TP.azo, 5e26, (3.6, 0.24, 0.33, 38e-4, 8.7, 3.37, None)),
        "gzo": (TP.gzo, 5e26, (3.6, 0.24, 0.33, 38e-4, 8.7, 3.37, None)),
        "in_cdo": (TP.in_cdo, 2.8e26, (5.5, 0.10, 0.735, None, 21.9, 0.9, 5.8)),
        "ito": (TP.ito, 3e26, (4.0, 0.21, 0.52, 30e-4, 9.0, 2.8, None)),
        "fto": (TP.fto, 2e26, (3.9, 0.09, 0.8, 35e-4, 12.3, 3.56, None)),
    }
    for key, (factory, n, (ei, m0, a, mu, es, eg, chi)) in expect.items():
        mat = factory(n)
        band, tr = mat.transport.band, mat.transport
        got = (mat.optical.eps_inf, band.alpha_per_eV, tr.eps_static, tr.band_gap_eV, tr.chi_eV)
        assert got == (ei, a, es, eg, chi), key
        assert band.m0_kg == m0 * M_E, key
        if mu is not None:
            assert mat.optical.gamma_rad_s.mobility(n) == mu, key
        table = {row.quantity.split(" (")[0]: row.default for row in TP.PARAMETER_TABLES[key]}
        assert table["eps_inf"].startswith(repr(ei)), (key, table["eps_inf"])
        static = [v for q, v in table.items() if q.startswith("static permittivity")][0]
        assert static.startswith(repr(es)), (key, static)
