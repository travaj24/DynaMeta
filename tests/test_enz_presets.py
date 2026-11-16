"""Gates for materials.enz_presets (TiN, ZrN, n-InAs, n-InAsSb, SiC, AlN, SiO2) and the FTO preset of
materials.tco_presets. Each preset against published reference points (sources cited per test; full
references in the module docstrings), within the literature spread, with a mutant that fails.

Run: python -m pytest tests/test_enz_presets.py -q
"""
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from dynameta.constants import EPS0, HBAR, M_E, Q_E
from dynameta.materials import enz_presets as EP
from dynameta.materials import tco_presets as TP
from dynameta.materials.lattice_optical import coupled_mode_frequencies, rad_s_to_wavenumber, wavenumber_to_rad_s

CM3 = 1.0e6


def _zero(f, lo, hi):
    return brentq(lambda L: complex(f(L)).real, lo, hi, xtol=1e-16)


# --------------------------------------------------------------------------------------------- nitrides --

def test_tin_crossover_and_830nm_against_measured_films():
    """TiN default (Reddy et al. 2017, 200 nm film): crossover vs MEASURED near-stoichiometric films -- 479 nm
    at x = 0.99 (Judek et al. 2021, Table 1) -- within 2 %; eps(830 nm) inside the span of three measured films
    (Guler et al. 2015: real -11.7 .. -19.1, imaginary 3.7 .. 5.5). Mutant: the 500 C Naik film (E_p 5.95 eV,
    a sub-stoichiometric film) misses the x = 0.99 crossover window... and its eps(830) real part leaves the
    measured span; a 15 % lower plasma energy fails the crossover."""
    t = EP.tin()
    assert t.is_metal and not t.is_semiconductor
    lc = _zero(t.eps, 0.40e-6, 0.70e-6)
    assert lc == pytest.approx(479e-9, rel=0.02, abs=0.0)
    e = complex(t.eps(830e-9))
    assert -19.1 <= e.real <= -11.7 and 3.7 <= e.imag <= 5.5
    bad = EP.tin(parameter_set="naik2013_500C")
    assert not (-19.1 <= complex(bad.eps(830e-9)).real <= -11.7)
    from dynameta.materials.lattice_optical import LorentzOscillatorOptical
    o = t.optical
    low = LorentzOscillatorOptical(o.eps_inf, o.oscillators, (0.85 * o.drude[0], o.drude[1]))
    assert _zero(low.eps, 0.40e-6, 0.80e-6) != pytest.approx(479e-9, rel=0.05, abs=0.0)
    with pytest.raises(ValueError, match="validity range"):
        t.eps(3e-6)


def test_zrn_against_published_points():
    """ZrN (Naik et al. 2013, the single complete fit): crossover within 6 % of the authors' 'around 430 nm'
    (Naik et al. 2011); eps(830 nm) against the measured -21.15 + 7.64i (Guler et al. 2015): real part within
    10 %, imaginary within 25 % (a different film). Mutant: TiN's parameters miss the ZrN crossover."""
    z = EP.zrn()
    assert _zero(z.eps, 0.36e-6, 0.60e-6) == pytest.approx(430e-9, rel=0.06, abs=0.0)
    e = complex(z.eps(830e-9))
    assert e.real == pytest.approx(-21.15, rel=0.10, abs=0.0) and e.imag == pytest.approx(7.64, rel=0.25, abs=0.0)
    assert _zero(EP.tin().eps, 0.40e-6, 0.70e-6) != pytest.approx(430e-9, rel=0.06, abs=0.0)


def test_nitride_gate_tuning_is_negligible():
    """From the presets' own plasma energies (free-electron mass): n = eps0 m (E_p / hbar)^2 / e^2 ~ 3.6-4.7e22
    cm^-3; the Thomas-Fermi length sqrt(2 eps0 E_F / (3 n e^2)) (eps_r 1) is < 0.1 nm and an oxide-breakdown
    gate charge (10 MV/cm in SiO2, 2.2e13 cm^-2) changes a 10 nm film's electron count by < 1e-3."""
    for mat in (EP.tin(), EP.zrn()):
        wp = mat.optical.drude[0]
        n = EPS0 * M_E * wp ** 2 / Q_E ** 2
        assert 3e28 < n < 6e28
        EF = HBAR ** 2 * (3 * math.pi ** 2 * n) ** (2.0 / 3.0) / (2 * M_E)
        l_tf = math.sqrt(2 * EPS0 * EF / (3 * n * Q_E ** 2))
        assert l_tf < 1e-10
        sheet = EPS0 * 3.9 * 1e9 / Q_E
        assert sheet / (n * 10e-9) < 1e-3


# ----------------------------------------------------------------------------------------- InAs / InAsSb --

def test_inas_optical_mass_against_dyksik():
    """Two-band Kane band of the preset vs the optical masses stated by Dyksik et al. (2016): 0.036 / 0.064 m_e at
    1e18 / 1e19 cm^-3 -> within 5 %. Mutant: a parabolic band (alpha = 0) is 40 % light at 1e19."""
    m = EP.inas(1e25)
    km = m.optical.m_opt_kg
    assert float(km(1e24)) / M_E == pytest.approx(0.036, rel=0.05, abs=0.0)
    assert float(km(1e25)) / M_E == pytest.approx(0.064, rel=0.05, abs=0.0)
    from dynameta.materials import KaneOpticalMass
    flat = KaneOpticalMass(m.transport.band.m0_kg, 0.0)
    assert float(flat(1e25)) / M_E != pytest.approx(0.064, rel=0.2, abs=0.0)


def test_inas_screened_plasma_wavelengths_within_the_published_spread():
    """Published Si:InAs screened plasma wavelengths at nearly equal nominal density disagree by 7-25 % (the
    density calibration of each paper): 9.5 / 5.8 um at 1.25e19 / 7.35e19 cm^-3 (Wei et al. 2017, P), 10.1 /
    6.45 / 5.7 / 4.64 um at 1.0 / 3.25 / 4.4 / 7.5e19 cm^-3 (Desouky et al. 2017, values from Law et al. 2012, S).
    The preset lies within 10 % of the Desouky set and within 18 % of Wei's (whose 7.35e19 point sits 25 % above
    Desouky's 7.5e19 one). Mutant: alpha = 0 misses every point by > 50 %."""
    pts = [(1.0e19, 10.1e-6, 0.10), (3.25e19, 6.45e-6, 0.10), (4.4e19, 5.7e-6, 0.10), (7.5e19, 4.64e-6, 0.10),
           (1.25e19, 9.5e-6, 0.10), (7.35e19, 5.8e-6, 0.18)]
    for n_cm3, lam, tol in pts:
        n = n_cm3 * CM3
        m = EP.inas(n)
        assert _zero(lambda L: m.eps(L, n_m3=n), 3.5e-6, 40e-6) == pytest.approx(lam, rel=tol, abs=0.0)
    from dynameta.materials import DrudeOptical
    n = 3.25e19 * CM3
    flat = DrudeOptical(eps_inf=12.3, m_opt_kg=EP.inas(n).transport.band.m0_kg, gamma_rad_s=1.5e13)
    assert _zero(lambda L: flat.eps(L, n_m3=n), 1e-6, 40e-6) != pytest.approx(6.45e-6, rel=0.5, abs=0.0)


def test_inassb_taliercio_point():
    """InAs0.91Sb0.09:Si, 3.4e19 cm^-3 (Taliercio et al. 2015): m* = 0.1 m_e and screened plasma wavelength 6.25
    um -> preset within 4 % on both. Mutant: the InAs gap (0.354 eV) keeps the mass within 1 % -- the gap barely
    matters at this density, as the paper's comparison implies -- but eps_inf 12.3 (InAs) moves the wavelength
    by 2.5 %; a parabolic band misses by > 40 %."""
    n = 3.4e25
    m = EP.inassb(n)
    assert float(m.optical.m_opt_kg(n)) / M_E == pytest.approx(0.1, rel=0.04, abs=0.0)
    assert _zero(lambda L: m.eps(L, n_m3=n), 3e-6, 30e-6) == pytest.approx(6.25e-6, rel=0.04, abs=0.0)
    from dynameta.materials import DrudeOptical
    flat = DrudeOptical(eps_inf=11.7, m_opt_kg=m.transport.band.m0_kg, gamma_rad_s=1.5e13)
    assert _zero(lambda L: flat.eps(L, n_m3=n), 1e-6, 30e-6) != pytest.approx(6.25e-6, rel=0.4, abs=0.0)


def test_inas_plasmon_phonon_coupling_and_interband_guard():
    """The lattice background shifts the ENZ wavelength by 1-5 % at 1.5e18 cm^-3 (ENZ ~19 um, close to the LO
    phonon) and by < 0.5 % at >= 3e19 cm^-3; the interband guard refuses photon energies at or above Eg + E_F
    (and accepts just below); the LST static permittivity of the lattice is the transport eps_static."""
    def lam(n, lattice):
        m = EP.inas(n, lattice=lattice)
        return _zero(lambda L: m.eps(L, n_m3=n), 3.5e-6, 40e-6)
    for n, lo, hi in ((1.5e24, 0.01, 0.05), (3.25e25, 0.0, 0.005), (7.5e25, 0.0, 0.005)):
        shift = abs(lam(n, "default") / lam(n, None) - 1.0)
        assert lo < shift < hi, (n, shift)
    m = EP.inas(1e25)
    onset = float(m.optical.background.onset_J(1e25))
    lam_on = 2 * math.pi * HBAR * 299792458.0 / onset
    with pytest.raises(ValueError, match="interband"):
        m.eps(0.99 * lam_on, n_m3=1e25)
    m.eps(1.01 * lam_on, n_m3=1e25)
    assert m.transport.eps_static == pytest.approx(EP.INAS_LATTICE.static_permittivity(), rel=1e-14, abs=0.0)
    assert m.transport.eps_static == pytest.approx(12.3 * (238.5 / 217.4) ** 2, rel=1e-12, abs=0.0)


def test_inas_mobilities_and_range():
    """Optical damping reproduces Wei et al.'s fitted gamma at their two densities exactly (by construction of
    the table) and the DC mobility is the Hilsum-form law of Petrovic et al. (2024): 3e4 / (1 + (N/8e17)^0.75).
    Density range refused outside 1e18-1e20 cm^-3."""
    for n, g in ((1.25e25, 1.4e13), (7.35e25, 1.9e13)):
        assert EP.inas(n).optical.gamma_rad_s(n) == pytest.approx(g, rel=1e-12, abs=0.0)
    n = 2e25
    mu = EP.inas(n).transport.mobility_m2Vs_of_n_m3(n)
    assert mu == pytest.approx(3e4 / (1 + (2e19 / 8e17) ** 0.75) * 1e-4, rel=1e-12, abs=0.0)
    with pytest.raises(ValueError, match="outside"):
        EP.inas(5e26)


# ------------------------------------------------------------------------------------ polar crystals --

@pytest.mark.parametrize("poly, refs", [("3C", (10.71,)), ("4H", (10.75, 10.80)), ("6H", (10.73, 10.77))])
def test_sic_small_particle_resonance_wavelengths(poly, refs):
    """Mutschke et al. (1999), Sec. 2.5: the Re(eps) = -2 (small-sphere) resonance wavelengths 10.71 (3C), 10.75 /
    10.80 (4H, perp / par) and 10.73 / 10.77 um (6H) from their Table 1 parameters with damping 2 cm^-1 -> the
    preset within 0.015 um (the quoted values are rounded to 0.01 um; 4H perp reproduces as 10.737). Mutant:
    the ordinary and extraordinary modes swapped (or, for 3C, the 4H E par c mode) misses by > 0.015 um."""
    o = EP.sic(poly).optical
    comps = [o] if poly == "3C" else [o.ordinary, o.extraordinary]
    for c, ref in zip(comps, refs):
        lam2 = brentq(lambda L: complex(c.eps(L)).real + 2.0, 10.3e-6, 11.5e-6, xtol=1e-16)
        assert lam2 * 1e6 == pytest.approx(ref, abs=0.015)
    mutants = [EP.sic("4H").optical.extraordinary] if poly == "3C" else [comps[1], comps[0]]
    for c, ref in zip(mutants, refs):
        lam2 = brentq(lambda L: complex(c.eps(L)).real + 2.0, 10.3e-6, 11.5e-6, xtol=1e-16)
        assert lam2 * 1e6 != pytest.approx(ref, abs=0.015)


def test_sic_reststrahlen_enz_and_uniaxial_tensor():
    """4H-SiC: normal-incidence bulk reflectance > 0.97 over 810-900 and > 0.93 over 805-950 cm^-1 (inside the
    Reststrahlen band between TO 796.6 and LO 968.7 of the ordinary mode; it rolls off within ~20 cm^-1 of the
    LO), below 0.5 at 700 and 0.3 at 1200 cm^-1; ordinary ENZ within 0.5 % of 10.3 um (Caldwell et al. 2015,
    citing Tiwald 1999); ordinary and extraordinary ENZ differ (10.32 vs 10.37 um); the uniaxial tensor reaches
    the assembler adapter; eps_static_dc is the LST value along c."""
    from dynameta.core.effects import OpticalModelEffect
    s = EP.sic("4H")
    o = s.optical.ordinary
    def R(nu):
        nn = np.sqrt(complex(o.eps(0.01 / nu)))
        return abs((1 - nn) / (1 + nn)) ** 2
    assert all(R(nu) > 0.93 for nu in np.linspace(805.0, 950.0, 20))
    assert all(R(nu) > 0.97 for nu in np.linspace(810.0, 900.0, 10))
    assert R(700.0) < 0.5 and R(1200.0) < 0.3
    l_o = _zero(o.eps, 9.5e-6, 10.6e-6)
    l_e = _zero(s.optical.extraordinary.eps, 9.5e-6, 10.6e-6)
    assert l_o == pytest.approx(10.3e-6, rel=0.005, abs=0.0) and 3e-8 < l_e - l_o < 8e-8
    t = OpticalModelEffect(s).eps({"n": None}, 10.2e-6)
    assert t.shape == (3, 3) and t[2, 2] != t[0, 0]
    assert s.eps_static_dc == pytest.approx(6.78 * (964.2 / 783.6) ** 2, rel=1e-12, abs=0.0)


def test_doped_sic_coupled_mode_in_the_measured_band():
    """n-4H-SiC, E par c (the A1 LO mode, mass 0.29 m_e): the upper coupled LO-phonon-plasmon mode at 2.6e18 and
    4.3e18 cm^-3 falls in the 950-1050 cm^-1 band where Hergert, Elm and Klar (2023) observe and fit the LOPC
    Raman band for samples of those densities; it rises with density and tends to the bare LO at low density.
    The doped preset's Re(eps) zero (low damping) is that mode. Mutant: the perpendicular mass (0.42) on the
    E par c mode still lands in the band at 4.3e18, but with no free carriers the mode stays at the bare LO
    (964 cm^-1), outside the measured shift."""
    T, L = wavenumber_to_rad_s(783.6), wavenumber_to_rad_s(964.2)
    prev = rad_s_to_wavenumber(L)
    for n in (2.6e24, 4.3e24):
        wp = math.sqrt(n * Q_E ** 2 / (EPS0 * 0.29 * M_E))
        up = rad_s_to_wavenumber(coupled_mode_frequencies(6.78, T, L, wp)[1])
        assert 950.0 <= up <= 1050.0 and up > prev
        prev = up
    n = 4.3e24
    d = EP.sic("4H", n_m3=n, mobility_m2Vs=10.0, damping_cm=0.01).optical.extraordinary   # ~undamped limit
    z = brentq(lambda nu: complex(d.eps(0.01 / nu)).real, 970.0, 1100.0, xtol=1e-9)
    wp = math.sqrt(n * Q_E ** 2 / (EPS0 * 0.29 * M_E))
    assert z == pytest.approx(rad_s_to_wavenumber(coupled_mode_frequencies(6.78, T, L, wp)[1]), rel=2e-3, abs=0.0)
    with pytest.raises(ValueError, match="mobility"):
        EP.sic("4H", n_m3=n)
    with pytest.raises(ValueError, match="electron_mass_ratio"):
        EP.sic("6H", n_m3=n, mobility_m2Vs=1e-2)


def test_aln_enz_and_lst():
    """AlN: E perp c ENZ at 1 / 909.9 cm^-1 = 10.99 um, within 1 % of 10.9 um (Caldwell et al. 2015, citing Kazan
    2009: LO 916.1); E par c ENZ longer (A1(LO) 886.4); LST static permittivity along c 8.72. Mutant: the
    Caldwell LO 916.1 cm^-1 shifts the ENZ by 0.7 %."""
    a = EP.aln().optical
    lo = _zero(a.ordinary.eps, 10e-6, 12e-6)
    assert lo == pytest.approx(10.9e-6, rel=0.01, abs=0.0)
    assert lo == pytest.approx(0.01 / 909.9, rel=1e-3, abs=0.0)
    assert _zero(a.extraordinary.eps, 10e-6, 12e-6) > lo
    assert EP.aln().eps_static_dc == pytest.approx(4.13 * (886.4 / 610.0) ** 2, rel=1e-12, abs=0.0)


def test_sio2_bands_against_thermal_oxide():
    """Fused silica (Kitamura et al. 2007 fit): the TO-side zero of Re(eps) and the Im(eps) peak within 0.5 % of
    the thermal-oxide TO 1076 cm^-1 and the energy-loss peak within 0.5 % of its LO 1256 cm^-1 (Kirk 1988, via
    Kalem 2015, S); the ENZ (Re zero) at 1246 cm^-1. Mutant: dropping the strongest band (a = 7.81 at 1058
    cm^-1) moves the ENZ by > 5 %."""
    from dynameta.materials.lattice_optical import GaussianOscillatorOptical
    o = EP.sio2().optical
    nu = np.linspace(1000.0, 1400.0, 40001)
    e = o.eps_wavenumber(nu)
    zeros = nu[np.where(np.diff(np.sign(e.real)) != 0)[0]]
    assert len(zeros) == 2
    assert zeros[0] == pytest.approx(1076.0, rel=0.005, abs=0.0)
    assert nu[np.argmax(e.imag)] == pytest.approx(1076.0, rel=0.005, abs=0.0)
    assert nu[np.argmax(np.imag(-1.0 / e))] == pytest.approx(1256.0, rel=0.005, abs=0.0)
    assert zeros[1] == pytest.approx(1246.4, abs=0.5)
    osc = tuple(t for t in EP.SIO2_KITAMURA_2007[1] if t[1] != 1058.2)
    bad = GaussianOscillatorOptical(EP.SIO2_KITAMURA_2007[0], osc).eps_wavenumber(nu)
    zb = nu[np.where(np.diff(np.sign(bad.real)) != 0)[0]]
    assert len(zb) == 0 or abs(zb[-1] / 1246.4 - 1.0) > 0.05
    with pytest.raises(ValueError, match="validity range"):
        o.eps(3e-6)


# ------------------------------------------------------------------------------------------------ FTO --

def test_fto_against_rey2013_films_and_the_epitaxial_alternative():
    """FTO default (Rey et al. 2013 film law): screened plasma wavelengths of their films 1639 / 2203 / 2988 nm at
    4.3 / 2.2 / 0.9e20 cm^-3 (from their Table I) within 8 % (the law's scatter about its own data: 0.4 / 3 / 7.6
    %); the epitaxial alternative (m0 0.205 / 0.183, alpha 0.73 / 0.33 per axis, Feneberg et al. 2019)
    reproduces their measured masses within 8 %. Mutant: the epitaxial orientation-average law on the film data
    misses the 4.3e20 point by > 10 %."""
    for n_cm3, lam in ((4.3e20, 1639e-9), (2.2e20, 2203e-9), (9e19, 2988e-9)):
        n = n_cm3 * CM3
        f = TP.fto(n)
        assert _zero(lambda L: f.eps(L, n_m3=n), 1e-6, 5e-6) == pytest.approx(lam, rel=0.08, abs=0.0)
    from dynameta.materials import KaneOpticalMass
    ns = np.array([1.6e19, 2.6e19, 1.3e20, 2.6e20]) * CM3
    for (m0, a), ref in (((0.205, 0.73), (0.22, 0.26, 0.31, 0.39)), ((0.183, 0.33), (0.20, 0.20, 0.22, 0.26))):
        assert KaneOpticalMass(m0 * M_E, a)(ns) / M_E == pytest.approx(np.array(ref), rel=0.08, abs=0.0)
    n = 4.3e26
    epi = TP.fto(n, m0_ratio=0.197, alpha_per_eV=0.56)
    assert _zero(lambda L: epi.eps(L, n_m3=n), 1e-6, 5e-6) != pytest.approx(1639e-9, rel=0.10, abs=0.0)
    from dynameta.materials.oxide_scattering import MasettiMobility
    assert f.transport.mobility_m2Vs_of_n_m3(1e26) == pytest.approx(
        MasettiMobility.sno2_ellmer_mientus_2008()(1e26), rel=1e-13, abs=0.0)


def test_fixed_density_wrapper():
    """FixedDensityOptical evaluates the wrapped model at its own density whatever n_m3 is passed (n only sets
    the output shape)."""
    d = EP.sic("3C", n_m3=1e24, mobility_m2Vs=1e-2, electron_mass_ratio=0.30).optical
    v = complex(d.eps(11e-6))
    assert np.all(d.eps(11e-6, n_m3=np.array([1e20, 5e26])) == v)


def test_sic_doped_inputs_are_validated():
    """A non-positive mobility (it would give a negative damping, i.e. gain) and a scalar mass for a uniaxial
    polytype (it needs (perp, par)) are refused with clear messages."""
    with pytest.raises(ValueError, match="mobility_m2Vs must be finite and > 0"):
        EP.sic("4H", n_m3=3e24, mobility_m2Vs=-0.03)
    with pytest.raises(ValueError, match="perp, par"):
        EP.sic("4H", n_m3=3e24, mobility_m2Vs=0.03, electron_mass_ratio=0.3)


def test_every_preset_pickles_with_a_deterministic_repr():
    """Every preset Material survives a pickle round trip (process pools) with the same repr and the same
    permittivity and mobility, and its repr carries no memory address (the optical-solve cache keys on it):
    the scattering link's mobility callables are dataclasses, not closures."""
    import pickle
    mats = [(TP.azo(5e26), 1.55e-6, 5e26), (TP.gzo(6e26, uniaxial=True), None, 6e26), (TP.in_cdo(2.8e26), 2e-6, 2.8e26),
            (TP.ito(3e26), 1.55e-6, 3e26), (TP.fto(2e26), 2e-6, 2e26),
            (TP.azo(5e26, grain_boundary=TP.zno_seto_barrier()), 1.55e-6, 5e26), (EP.inas(1e25), 10e-6, 1e25),
            (EP.inassb(3e25), 8e-6, 3e25), (EP.tin(), 0.6e-6, None), (EP.zrn(), 0.6e-6, None),
            (EP.sic("4H"), None, None),
            (EP.sic("3C", n_m3=3e24, mobility_m2Vs=0.03, electron_mass_ratio=0.3), 11e-6, None),
            (EP.aln(), None, None), (EP.sio2(), 9e-6, None)]
    for mat, lam, n in mats:
        r = pickle.loads(pickle.dumps(mat))
        assert repr(r) == repr(mat) and " at 0x" not in repr(mat), mat.name
        if lam is not None:
            assert complex(r.eps(lam, n_m3=n)) == complex(mat.eps(lam, n_m3=n))
        if n is not None and mat.transport is not None:
            assert r.transport.mobility_m2Vs_of_n_m3(n) == mat.transport.mobility_m2Vs_of_n_m3(n)
