"""
Epsilon-near-zero material presets beyond the doped oxides: metallic nitrides (tin, zrn), heavily
n-doped III-V for the mid-infrared (inas, inassb) and phonon-polariton crystals and glass (sic, aln, sio2).
The doped oxides (AZO, GZO, In:CdO, ITO, FTO) are in materials.tco_presets; docs/enz_presets.md covers both.

TWO ENZ MECHANISMS. (1) Free carriers: Re(eps) = 0 where the electron plasma cancels the background
(doped oxides, nitrides, doped III-V). The density -- and so the ENZ point -- can be moved by a gate in the
semiconductors (n-InAs: large response, light mass), not in the nitrides (n ~ 4e22 cm^-3, Thomas-Fermi
screening length < 0.1 nm: a gate changes the film's electron count by < 1e-3). (2) Lattice vibrations:
Re(eps) = 0 at the longitudinal-optical phonon frequency of a polar crystal (SiC, AlN) or glass band (SiO2).
Not gate-tunable (no free carriers), sharp and low-loss in crystals; doping a polar crystal adds a free-carrier
term and splits the LO mode into coupled phonon-plasmon modes (plasmon-LO-phonon coupling).

SOURCES (status: P = primary / open-access full text read, A = abstract only, S = secondary, D = derived here).

  Nitrides
  [Red17]  H. Reddy et al., ACS Photonics 4, 1413 (2017), 10.1021/acsphotonics.7b00127 -- TiN Drude + 2 Lorentz,
           200 / 50 / 30 nm films, 23-900 C (P, arXiv:1702.03053 Eq. (1), SI Tables S1-S3)
  [Nai13]  G. V. Naik, V. M. Shalaev, A. Boltasseva, Adv. Mater. 25, 3264 (2013), 10.1002/adma.201205076 -- TiN
           and ZrN Drude + Lorentz, Table 2, valid 350-2000 nm (P, author copy)
  [Jud21]  J. Judek et al., Materials 14, 7095 (2021), 10.3390/ma14227095 -- TiN_x crossover 479 nm at x = 0.99,
           392.5-608 nm for x = 0.84-1.21 (P)
  [Gul15]  U. Guler et al., Faraday Discuss. 178, 71 (2015) -- measured eps(830 nm): TiN -11.73+5.47i,
           -15.57+3.68i, -19.08+4.11i; ZrN -21.15+7.64i (P)
  [Pat15]  P. Patsalas, N. Kalfagiannis, S. Kassavetis, Materials 8, 3128 (2015), 10.3390/ma8063128 --
           stoichiometry dependence, screened plasma energy 2.65 eV for x = 1 (P)
  [Lu17]   Y.-J. Lu et al., Nat. Commun. 8, 1631 (2017), 10.1038/s41467-017-01870-0 -- gating of a 7 nm, low-
           density TiN film; n = 5.9e20-4.1e22 cm^-3 in sputtered TiN (P)

  InAs, InAsSb
  [Dyk16]  M. Dyksik et al., J. Appl. Phys. 120, 043104 (2016), 10.1063/1.4958904 -- Si:InAs optical mass
           0.036 / 0.064 m_e at 1e18 / 1e19 cm^-3; Kane mass law with P^2 = 11.9 eV (P, accepted manuscript)
  [Wei17]  D. Wei, C. Harris, S. Law, Opt. Mater. Express 7, 2672 (2017), 10.1364/OME.7.002672 -- Si:InAs
           7.35e19 / 1.25e19 cm^-3: screened plasma wavelength 5.8 / 9.5 um, gamma 1.9e13 / 1.4e13 s^-1, eps_inf
           12.3 (P)
  [Sha22]  S. Shayegan et al., Sci. Adv. 8, eabm4308 (2022) -- n-InAs eps_inf 12.3 (ellipsometry fit) (P)
  [Liu17]  Liu et al., Phys. Rev. Applied 8, 034028 (2017) -- InAs TO / LO 217.4 / 238.5 cm^-1, 300 K (P)
  [Mil20]  D. Milovich et al., J. Photon. Energy 10, 025503 (2020) -- InAs Eg 0.354 eV; phonon damping 2.89e11
           rad/s (Adachi) (S)
  [Pet24]  M. Petrovic et al., Opt. Mater. Express 14, 2912 (2024) -- InAs/InAsSb Hall mobility law
           mu = 3e4 / (1 + (N / 8e17)^0.75) cm^2/(V s) (Baranov, for InAs; used for InAsSb too, D);
           InAs0.915Sb0.085 transparent above ~4.3 um (k.p calculation) (P / S)
  [Tal15]  T. Taliercio et al., Opt. Express 23, 29423 (2015), 10.1364/OE.23.029423 -- InAs0.91Sb0.09:Si
           3.4e19 cm^-3, m* = 0.1 m_e, screened plasma wavelength 6.25 um, eps_inf 11.7 (P)

  Polar crystals and glass
  [Mut99]  H. Mutschke et al., Astron. Astrophys. 345, 187 (1999) -- SiC 3C / 4H / 6H TO, LO, eps_inf
           (literature values collected in Table 1), damping 1-3 cm^-1 (2 cm^-1 used), Re(eps) = -2 resonance
           wavelengths (P)
  [Cal15]  J. D. Caldwell et al., Nanophotonics 4, 44 (2015), 10.1515/nanoph-2014-0003 -- 4H-SiC ENZ 10.3 um,
           AlN ENZ 10.9 um (S)
  [Sch17]  S. Schoeche et al., J. Appl. Phys. 121, 205701 (2017), 10.1063/1.4983765 -- AlN factorized fit:
           E1(TO) 666.4 / 3.1, E1(LO) 909.9 / 3.1, A1(LO) 886.4 / 9.1 cm^-1, eps_inf 4.08 / 4.13 (P, arXiv)
  [Iak03]  V. Iakovenko et al., arXiv:cond-mat/0301045 -- AlN Raman A1(TO) 610 cm^-1 (P, preprint)
  [Kit07]  R. Kitamura, L. Pilon, M. Jonasz, Appl. Opt. 46, 8118 (2007), 10.1364/AO.46.008118 -- fused silica
           8-Gaussian fit, eps_inf 2.1232, valid ~7-50 um (P)
  [Kirk88] C. T. Kirk, Phys. Rev. B 38, 1255 (1988) -- thermal SiO2 TO-LO pair 1076-1256 cm^-1 (S, via
           Kalem, arXiv:1512.08005)
  [HEK23]  Hergert, Elm, Klar, J. Raman Spectrosc. 54, 737 (2023), 10.1002/jrs.6531 -- n-4H-SiC coupled
           LO-phonon-plasmon band 950-1050 cm^-1 at ~2.6-4.3e18 cm^-3 (P)
  [Son95]  N. T. Son et al. (1995) -- 4H-SiC electron masses 0.42 (perp) / 0.29 (par) m_e (S, via Burin et al.,
           arXiv:2410.06798, and Lambrecht and Segall, arXiv:mtrl-th/9508004)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dynameta.constants import HBAR, M_E, Q_E
from dynameta.materials.lattice_optical import (GaussianOscillatorOptical, LorentzOscillatorOptical,
                                                PolarLatticeOptical, with_free_carriers)
from dynameta.materials.material import Material
from dynameta.materials.optical_model import DrudeOptical, OpticalModel
from dynameta.materials.oxide_scattering import OpticalMobilityDamping, PolycrystallineScattering, TabulatedMobility
from dynameta.materials.transport_model import KaneBand, TransportModel
from dynameta.materials.uniaxial_optical import UniaxialOptical

__all__ = ["tin", "zrn", "inas", "inassb", "sic", "aln", "sio2", "FixedDensityOptical", "InterbandOnsetGuard",
           "HilsumMobility", "kane_two_band", "INAS_LATTICE", "SIC_MODES_CM", "ALN_MODES_CM", "SIO2_KITAMURA_2007",
           "VALIDITY_RANGES"]

_CM3 = 1.0e6
_CM2VS = 1.0e-4


# =====================================================================================================
# small generic pieces
# =====================================================================================================

@dataclass(frozen=True)
class FixedDensityOptical(OpticalModel):
    """A density-dependent optical model evaluated at ONE fixed carrier density: for a uniformly doped layer
    that no carrier solver touches (a doped SiC substrate, a doped InAs ground plane). eps(lambda, n_m3=...)
    ignores the n_m3 passed in (it only sets the output shape)."""
    model: OpticalModel
    n_m3: float

    def __post_init__(self):
        n = float(self.n_m3)
        if not (math.isfinite(n) and n > 0.0):
            raise ValueError("FixedDensityOptical: n_m3 must be finite and > 0 (m^-3)")
        if getattr(self.model, "field_response", False):
            raise TypeError("FixedDensityOptical: a tensor-valued model cannot be wrapped")

    def eps(self, lambda_m: float, *, n_m3=None):
        v = complex(np.asarray(self.model.eps(lambda_m, n_m3=float(self.n_m3)))[()])
        return v if n_m3 is None else np.full(np.shape(n_m3), v, dtype=np.complex128)


@dataclass(frozen=True)
class InterbandOnsetGuard:
    """Background model that refuses photon energies at or above the interband absorption onset of a
    degenerate n-type semiconductor, Eg + E_F(n) (T = 0 Kane Fermi energy of the conduction band; band-gap
    renormalization neglected), where a free-carrier (Drude) model stops being valid -- the short-wavelength
    validity limit of a mid-IR preset. Below the onset it returns `inner`'s background (a lattice model, so the
    plasmon-LO-phonon coupling is kept) or the DrudeOptical's eps_inf when inner is None. Evaluated per node:
    a depleted node (E_F -> 0) has its onset at Eg."""
    gap_eV: float
    band: KaneBand
    inner: Optional[object] = None

    def __post_init__(self):
        if not (math.isfinite(self.gap_eV) and self.gap_eV > 0.0):
            raise ValueError("InterbandOnsetGuard: gap_eV must be finite and > 0")
        if not isinstance(self.band, KaneBand):
            raise TypeError("InterbandOnsetGuard: band must be a KaneBand")

    def onset_J(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), 0.0)
        return float(self.gap_eV) * Q_E + np.where(n > 0.0, self.band.optical_mass().fermi_energy_J(n), 0.0)

    def eps_background(self, n_m3, lambda_m, eps_inf):
        E = 2.0 * math.pi * HBAR * 299792458.0 / float(lambda_m)
        n = np.asarray(n_m3, dtype=np.float64)
        on = self.onset_J(n)
        if np.any(E >= on):
            raise ValueError("InterbandOnsetGuard: photon energy {:.4g} eV is at or above the interband onset "
                             "Eg + E_F = {:.4g} eV (lambda {:.4g} um); the free-carrier model is not valid there "
                             "(interband absorption)".format(E / Q_E, float(np.min(on)) / Q_E, float(lambda_m) * 1e6))
        if self.inner is None:
            return eps_inf if n.ndim == 0 else np.full(n.shape, float(eps_inf))
        return self.inner.eps_background(n_m3, lambda_m, eps_inf)


@dataclass(frozen=True)
class HilsumMobility:
    """Empirical mobility law mu(N) = mu_0 / (1 + (N / N_ref)^exponent) (C. Hilsum, Electron. Lett. 10, 259
    (1974) form). SI fields (m^2/(V s), m^-3). Callable of n."""
    mu0_m2Vs: float
    n_ref_m3: float
    exponent: float

    def __post_init__(self):
        for nm in ("mu0_m2Vs", "n_ref_m3", "exponent"):
            v = float(getattr(self, nm))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError("HilsumMobility: {} must be finite and > 0".format(nm))

    def __call__(self, n_m3):
        n = np.maximum(np.asarray(n_m3, dtype=np.float64), 0.0)
        out = self.mu0_m2Vs / (1.0 + (n / self.n_ref_m3) ** self.exponent)
        return float(out) if np.ndim(out) == 0 else out


def kane_two_band(gap_eV: float, kane_P2_eV: float) -> KaneBand:
    """The two-band Kane conduction band of a narrow-gap III-V: band-edge mass m0 / m_e = 1 / (1 + 4 P^2 /
    (3 Eg)) (with the free-electron term) and nonparabolicity alpha = (1 - m0 / m_e)^2 / Eg. P^2 in eV (the
    E(1 + alpha E) = hbar^2 k^2 / (2 m0) form of materials.KaneBand)."""
    eg, p2 = float(gap_eV), float(kane_P2_eV)
    if not (eg > 0.0 and p2 > 0.0):
        raise ValueError("kane_two_band: gap_eV and kane_P2_eV must be > 0")
    m0 = 1.0 / (1.0 + 4.0 * p2 / (3.0 * eg))
    return KaneBand(m0_kg=m0 * M_E, alpha_per_eV=(1.0 - m0) ** 2 / eg)


# =====================================================================================================
# metallic nitrides (passive plasmonic / ENZ layers)
# =====================================================================================================

_NITRIDE_RANGE = (0.35e-6, 2.0e-6)


def tin(*, name: str = "TiN", parameter_set: str = "reddy2017_200nm") -> Material:
    """Titanium nitride, a passive plasmonic / ENZ metal (Material(is_metal=True), no carrier physics).

    LorentzOscillatorOptical (Drude + Lorentz) from a published film fit, energies in eV:
      'reddy2017_200nm' (default) [Red17] 200 nm, 23 C, DC sputtered at 850 C on sapphire: eps_inf 4.74,
           E_p 7.01, Gamma_D 0.23; Lorentz (w_L^2 = 36.7 eV^2, E0 4.1, gamma 1.45) and (3.88 eV^2, 1.85, 1.33)
           -> d_eps = w_L^2 / E0^2. Crossover 481.5 nm (D).
      'naik2013_800C' [Nai13] eps_b 4.855, E_p 7.9308, gamma_p 0.1795; Lorentz f 3.2907, E0 4.2196, gamma 2.0341
           (crossover 474.7 nm, D).
      'naik2013_500C' [Nai13] eps_b 2.485, E_p 5.953, gamma_p 0.5142; f 2.0376, E0 3.9545, gamma 2.4852.
    STOICHIOMETRY / DEPOSITION dominate: the crossover runs 392-608 nm for TiN_x, x = 0.84-1.21 [Jud21]; good
    near-stoichiometric films sit at 464-495 nm; measured eps(830 nm) spans -11.7 .. -19.1 (real part)
    between films [Gul15]. Validity 350-2000 nm (the fit window; the fitted sets have spurious extra zeros of
    Re(eps) below ~300 nm). GATE TUNING is negligible: n ~ 3.5-4.7e22 cm^-3 (from E_p with the free-electron
    mass, D), Thomas-Fermi length ~0.06 nm, so even an oxide-breakdown gate charge (2e13 cm^-2) changes a
    10-30 nm film's electron count by ~2-5e-4 (D); the gating reported on TiN [Lu17] used deliberately
    low-density 7 nm films."""
    sets = {
        "reddy2017_200nm": (4.74, ((36.7 / 4.1 ** 2, 4.1, 1.45), (3.88 / 1.85 ** 2, 1.85, 1.33)), (7.01, 0.23)),
        "naik2013_800C": (4.855, ((3.2907, 4.2196, 2.0341),), (7.9308, 0.1795)),
        "naik2013_500C": (2.485, ((2.0376, 3.9545, 2.4852),), (5.953, 0.5142)),
    }
    if parameter_set not in sets:
        raise ValueError("tin: parameter_set must be one of {}, got {!r}".format(sorted(sets), parameter_set))
    ei, osc, dr = sets[parameter_set]
    opt = LorentzOscillatorOptical.from_ev(ei, osc, dr, wavelength_range_m=_NITRIDE_RANGE)
    return Material(name=name, optical=opt, is_metal=True)


def zrn(*, name: str = "ZrN") -> Material:
    """Zirconium nitride, passive plasmonic / ENZ metal. SINGLE complete published fit [Nai13] (reactive
    sputtered, 800 C, nitrogen-deficient ambient, inferred): eps_b 3.4656, E_p 8.018 eV, gamma_p 0.5192 eV;
    Lorentz f 2.4509, E0 5.48 eV, gamma 1.7369 eV. Crossover 408.7 nm (D); ~430 nm is quoted for ZrN films by
    Naik et al., Opt. Mater. Express 1, 1090 (2011) (S, not verified here).
    Drude plasma energies 6.6-8.0 eV and damping 0.29-0.62 eV are reported across films (Veszelei 1994, A;
    DFT 7.46 eV). Validity 350-2000 nm. Gate tuning negligible (as tin())."""
    opt = LorentzOscillatorOptical.from_ev(3.4656, ((2.4509, 5.48, 1.7369),), (8.018, 0.5192),
                                           wavelength_range_m=_NITRIDE_RANGE)
    return Material(name=name, optical=opt, is_metal=True)


# =====================================================================================================
# n-InAs, n-InAsSb (mid-IR, strongly nonparabolic)
# =====================================================================================================

_INAS_GAP = 0.354                 # eV, 300 K [Mil20] and two further sources (S)
_INAS_P2 = 11.9                   # eV, Kane P^2 used by [Dyk16] (Li et al. 1993) (S)
_INAS_EPS_INF = 12.3              # [Sha22] (P), [Wei17] (P); 11.7 (Palik, S) is the alternative
#: InAs lattice: TO / LO 217.4 / 238.5 cm^-1 [Liu17] (P), damping 1.5 cm^-1 (2.89e11 rad/s, Adachi via [Mil20], S)
INAS_LATTICE = PolarLatticeOptical.from_wavenumbers(_INAS_EPS_INF, ((217.4, 238.5, 1.53, 1.53),))
#: optical mobility of Si:InAs from [Wei17]'s damping with the preset band's optical mass (D)
_INAS_GAMMA_POINTS = ((1.25e19 * _CM3, 1.4e13), (7.35e19 * _CM3, 1.9e13))
_INAS_RANGE = (1.0e18 * _CM3, 1.0e20 * _CM3)
_INASSB_GAP = 0.288               # eV, InAs0.915Sb0.085, transparency edge 4.3 um ([Pet24], k.p, S)
_INASSB_EPS_INF = 11.7            # as used by [Tal15] (InAs value from Palik, S)
_INASSB_RANGE = (1.0e18 * _CM3, 1.0e20 * _CM3)
#: Hall mobility law of InAs and InAsSb ([Pet24] Eq. (1), Baranov)
INAS_HALL_MOBILITY = HilsumMobility(mu0_m2Vs=3.0e4 * _CM2VS, n_ref_m3=8.0e17 * _CM3, exponent=0.75)


def _iii_v(*, key, name, n_bg_m3, gap_eV, kane_P2_eV, eps_inf, lattice, optical_mobility_m2Vs, eps_static,
           chi_eV, strict):
    from dynameta.materials.tco_presets import _check_density_range
    n = _check_density_range(n_bg_m3, key, VALIDITY_RANGES[key], strict)
    band = kane_two_band(gap_eV, kane_P2_eV)
    mass = band.optical_mass()
    if optical_mobility_m2Vs is None:
        pts = tuple(sorted(_INAS_GAMMA_POINTS))
        optical_mobility_m2Vs = TabulatedMobility(
            n_m3=tuple(p[0] for p in pts), mobility_m2Vs=tuple(Q_E / (float(mass(p[0])) * p[1]) for p in pts),
            source="Wei, Harris, Law, Opt. Mater. Express 7, 2672 (2017): gamma 1.4e13 / 1.9e13 s^-1 at "
                   "1.25e19 / 7.35e19 cm^-3, with the preset band's optical mass")
    damping = OpticalMobilityDamping(mobility_m2Vs=optical_mobility_m2Vs, mass=mass)
    bg = InterbandOnsetGuard(gap_eV=gap_eV, band=band, inner=lattice)
    drude = DrudeOptical(eps_inf=float(eps_inf), m_opt_kg=mass, gamma_rad_s=damping, background=bg)
    scat = PolycrystallineScattering(one_over_tau=damping, m_cond_kg=mass, intragrain_mobility=INAS_HALL_MOBILITY)
    tr = TransportModel(n_bg_m3=n, eps_static=float(eps_static), band=band, band_gap_eV=float(gap_eV), chi_eV=chi_eV,
                        chi_reference_T_K=300.0, physics="equilibrium")
    return Material(name=name, optical=drude, transport=tr, scattering=scat)


def inas(n_bg_m3: float, *, name: str = "InAs", eps_inf: float = _INAS_EPS_INF, gap_eV: float = _INAS_GAP,
         kane_P2_eV: float = _INAS_P2, lattice="default", optical_mobility_m2Vs=None,
         eps_static: Optional[float] = None, chi_eV: Optional[float] = None, strict: bool = True) -> Material:
    """Heavily n-doped InAs ("designer metal") for the mid-IR, 1e18-1e20 cm^-3 (Si doping reaches ~8e19).

    Band: two-band Kane (kane_two_band) with Eg 0.354 eV and P^2 11.9 eV -> m0 = 0.0218 m_e, alpha = 2.70 eV^-1
    (D): the optical mass rises 0.037 -> 0.067 -> 0.125 m_e at 1e18 / 1e19 / 7.35e19 cm^-3, within 1.7 / 4.5 % of
    [Dyk16]'s 0.036 / 0.064 and up to 10 % above their full Kane formula at 1e20. Background: the lattice
    (TO/LO 217.4/238.5 cm^-1; lattice='default', or None, or a PolarLatticeOptical with the same eps_inf), so
    the plasmon-LO-phonon coupling is included (it moves the ENZ wavelength by ~2.4 % at 1.5e18 cm^-3 and < 0.5 %
    above 1e19); inside an InterbandOnsetGuard that refuses wavelengths at or below the interband onset Eg +
    E_F(n) (~1.7 um at 1e19, ~0.93 um at 1e20). Damping from [Wei17]'s fitted gamma (optical mobility, table in
    log n, held outside 1.25e19-7.35e19 cm^-3); the DC (Hall) mobility is the Hilsum-form law of [Pet24].
    eps_static: None = the LST value of the lattice (14.8; ~15.1-15.2 is the usual tabulated value, not
    verified here); chi_eV None (not verified).
    LITERATURE SPREAD: the published screened plasma wavelengths at equal nominal density differ by 7-25 %
    (5.8 vs 4.64 um near 7.4e19 cm^-3) -- mostly the density calibration (Hall, SIMS, optical) of each paper."""
    lat = INAS_LATTICE if (isinstance(lattice, str) and lattice == "default") else lattice
    if lat is not None and abs(float(lat.eps_inf) - float(eps_inf)) > 1e-12:
        lat = PolarLatticeOptical(eps_inf=float(eps_inf), modes=lat.modes)
    es = (lat.static_permittivity() if lat is not None else INAS_LATTICE.static_permittivity()) \
        if eps_static is None else eps_static
    return _iii_v(key="inas", name=name, n_bg_m3=n_bg_m3, gap_eV=gap_eV, kane_P2_eV=kane_P2_eV, eps_inf=eps_inf,
                  lattice=lat, optical_mobility_m2Vs=optical_mobility_m2Vs, eps_static=es, chi_eV=chi_eV,
                  strict=strict)


def inassb(n_bg_m3: float, *, name: str = "InAsSb", eps_inf: float = _INASSB_EPS_INF, gap_eV: float = _INASSB_GAP,
           kane_P2_eV: float = _INAS_P2, optical_mobility_m2Vs=None, eps_static: Optional[float] = None,
           chi_eV: Optional[float] = None, strict: bool = True) -> Material:
    """Heavily n-doped InAs(1-x)Sb(x), x ~ 0.09 (lattice-matched to GaSb), 1e18-1e20 cm^-3.

    The same two-band Kane machinery as inas() with the alloy gap (0.288 eV, k.p value of [Pet24]) and InAs's
    Kane P^2 (an assumption, D): optical mass 0.097 m_e at 3.4e19 cm^-3 against [Tal15]'s 0.1. eps_inf 11.7
    ([Tal15]). The alloy's phonons are not sourced, so no lattice term (its effect on the ENZ wavelength is
    < 0.5 % above ~3e19 cm^-3 in InAs); interband guard as inas(). Damping: the InAs table and DC mobility:
    the InAs Hilsum-form law (no InAsSb-specific data). eps_static: None = the InAs LST value (an
    approximation, the alloy value is not sourced)."""
    es = INAS_LATTICE.static_permittivity() if eps_static is None else eps_static
    return _iii_v(key="inassb", name=name, n_bg_m3=n_bg_m3, gap_eV=gap_eV, kane_P2_eV=kane_P2_eV, eps_inf=eps_inf,
                  lattice=None, optical_mobility_m2Vs=optical_mobility_m2Vs, eps_static=es, chi_eV=chi_eV,
                  strict=strict)


# =====================================================================================================
# phonon-polariton crystals and glass
# =====================================================================================================

#: SiC modes [Mut99] Table 1 (literature averages): polytype -> {axis: (eps_inf, nu_TO, nu_LO)} in cm^-1
SIC_MODES_CM = {
    "4H": {"perp": (6.56, 796.6, 968.7), "par": (6.78, 783.6, 964.2)},
    "6H": {"perp": (6.56, 797.0, 969.9), "par": (6.72, 788.1, 965.3)},
    "3C": {"iso": (6.49, 795.9, 972.3)},
}
_SIC_GAMMA_CM = 2.0               # [Mut99]: 1-3 cm^-1 for high-quality SiC; 2 used for their spectra
_SIC_MASS = {"4H": (0.42, 0.29)}  # (perp, par) m_e [Son95] (S)
#: AlN factorized modes: axis -> (eps_inf, ((nu_TO, nu_LO, g_TO, g_LO),)) cm^-1. perp: [Sch17]; par: A1(LO)
#: [Sch17], A1(TO) 610 [Iak03] (Raman), A1(TO) damping not measured -> g_TO = g_LO (D; passive window 4.3-9.1)
ALN_MODES_CM = {"perp": (4.08, ((666.4, 909.9, 3.1, 3.1),)), "par": (4.13, ((610.0, 886.4, 9.1, 9.1),))}
#: Fused silica, 8 Gaussian oscillators (a, nu0, fwhm) [cm^-1] and eps_inf [Kit07] Table 2 (fit to Popova et al.)
SIO2_KITAMURA_2007 = (2.1232, ((3.7998, 1089.7, 31.454), (0.46089, 1187.7, 100.46), (1.2520, 797.78, 91.601),
                              (7.8147, 1058.2, 63.153), (1.0313, 446.13, 275.111), (5.3757, 443.00, 45.220),
                              (6.3305, 465.80, 22.680), (1.2948, 1026.7, 232.14)))
_IR_RANGE_SIC = (5.0e-6, 50.0e-6)
_IR_RANGE_SIO2 = (7.0e-6, 50.0e-6)


def _lattice(eps_inf, modes_cm, rng):
    return PolarLatticeOptical.from_wavenumbers(eps_inf, modes_cm, wavelength_range_m=rng)


def sic(polytype: str = "4H", *, name: Optional[str] = None, damping_cm: float = _SIC_GAMMA_CM,
        n_m3: Optional[float] = None, electron_mass_ratio=None, mobility_m2Vs: Optional[float] = None) -> Material:
    """Silicon carbide, a phonon-polariton ENZ crystal (Reststrahlen band ~797-970 cm^-1, ENZ at the LO
    frequency, ~10.3 um). polytype '4H' / '6H' (uniaxial: UniaxialOptical with the E perp c and E par c modes,
    c axis along z) or '3C' (cubic, isotropic). Frequencies and eps_inf: [Mut99] Table 1 (literature
    values; other 4H entries there differ by ~0.05 in eps_inf); damping 2 cm^-1 for TO and LO ([Mut99]: 1-3
    cm^-1 for high-quality material). Validity 5-50 um.
    eps_static_dc: the LST static permittivity along z (c axis).

    DOPED SiC (n_m3 [m^-3] given): the free-carrier term on top of the lattice (plasmon-LO-phonon coupling)
    at that fixed density (FixedDensityOptical); electron_mass_ratio (perp, par) -- default (0.42, 0.29) for
    4H ([Son95], S), required for 6H/3C (a float for 3C) -- and mobility_m2Vs (required) set wp and gamma =
    e / (m mu)."""
    if polytype not in SIC_MODES_CM:
        raise ValueError("sic: polytype must be one of {}, got {!r}".format(sorted(SIC_MODES_CM), polytype))
    g = float(damping_cm)
    axes = SIC_MODES_CM[polytype]
    lat = {ax: _lattice(ei, ((t, l, g, g),), _IR_RANGE_SIC) for ax, (ei, t, l) in axes.items()}
    nm = name or "SiC-{}".format(polytype)
    if n_m3 is not None:
        if mobility_m2Vs is None:
            raise ValueError("sic: a doped SiC (n_m3 given) needs mobility_m2Vs (the free-carrier damping)")
        mu = float(mobility_m2Vs)
        if not (math.isfinite(mu) and mu > 0.0):
            raise ValueError("sic: mobility_m2Vs must be finite and > 0 (m^2/(V s)), got {!r}".format(mobility_m2Vs))
        masses = electron_mass_ratio if electron_mass_ratio is not None else _SIC_MASS.get(polytype)
        if masses is None:
            raise ValueError("sic: no sourced electron mass for {}-SiC; pass electron_mass_ratio".format(polytype))
        if polytype == "3C":
            masses = (float(masses), float(masses)) if np.ndim(masses) == 0 else tuple(masses)
        elif np.ndim(masses) != 1 or len(masses) != 2:
            raise ValueError("sic: electron_mass_ratio for {}-SiC is (perp, par), got {!r}".format(polytype, masses))
        doped = {}
        for ax, m in zip(("perp", "par"), masses):
            key = ax if ax in lat else "iso"
            mk = float(m) * M_E
            doped[ax] = FixedDensityOptical(with_free_carriers(lat[key], m_opt_kg=mk, gamma_rad_s=Q_E / (mk * mu)),
                                            float(n_m3))
        if polytype == "3C":
            return Material(name=nm, optical=doped["perp"])
        return Material(name=nm, optical=UniaxialOptical(ordinary=doped["perp"], extraordinary=doped["par"]))
    if polytype == "3C":
        return Material(name=nm, optical=lat["iso"], eps_static_dc=lat["iso"].static_permittivity())
    return Material(name=nm, optical=UniaxialOptical(ordinary=lat["perp"], extraordinary=lat["par"]),
                    eps_static_dc=lat["par"].static_permittivity())


def aln(*, name: str = "AlN") -> Material:
    """Wurtzite AlN (c axis along z), phonon-polariton ENZ crystal: E perp c [Sch17] E1(TO) 666.4, E1(LO) 909.9,
    damping 3.1 cm^-1, eps_inf 4.08; E par c: A1(LO) 886.4 (damping 9.1) and eps_inf 4.13 [Sch17], A1(TO) 610
    cm^-1 from Raman [Iak03] (mixed source; its damping is not measured: equal to the LO damping, the passive
    choice of the Lorentz limit). UniaxialOptical; ENZ 10.99 um (E perp c), 11.28 um (E par c). Validity 5-50
    um. eps_static_dc: the LST value along z (8.72 from this mixed set; [Sch17]'s own fit implies 9.11, i.e.
    an A1(TO) near 596 cm^-1 -- the static value is uncertain by ~4 %)."""
    lat = {ax: _lattice(ei, modes, _IR_RANGE_SIC) for ax, (ei, modes) in ALN_MODES_CM.items()}
    return Material(name=name, optical=UniaxialOptical(ordinary=lat["perp"], extraordinary=lat["par"]),
                    eps_static_dc=lat["par"].static_permittivity())


def sio2(*, name: str = "SiO2") -> Material:
    """Amorphous SiO2 (fused silica) in the infrared: the 8-Gaussian-oscillator fit of [Kit07] (Table 2), valid
    7-50 um. Re(eps) crosses zero at 1246 cm^-1 (8.02 um; the glass's LO-like ENZ point, D) and at 1077 cm^-1
    on the TO side; thermal oxide shows the TO-LO pair at 1076-1256 cm^-1 [Kirk88]. Sputtered, sub-
    stoichiometric oxide sits ~40 cm^-1 lower (Herguedas and Carretero 2023): fit your own film. Isotropic;
    eps_static_dc: the model's own zero-frequency limit."""
    ei, osc = SIO2_KITAMURA_2007
    opt = GaussianOscillatorOptical(ei, osc, wavelength_range_m=_IR_RANGE_SIO2)
    return Material(name=name, optical=opt, eps_static_dc=opt.static_permittivity())


VALIDITY_RANGES = {"inas": _INAS_RANGE, "inassb": _INASSB_RANGE}
