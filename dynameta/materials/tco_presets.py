"""
Transparent-conducting-oxide material presets for gate-tunable epsilon-near-zero (ENZ) films:
Al-doped ZnO (azo), Ga-doped ZnO (gzo), In-doped CdO (in_cdo), Sn-doped In2O3 (ito) and F-doped SnO2 (fto).

WHAT A PRESET IS. A factory function of the background carrier density n_bg that returns a complete
Material: a DrudeOptical (Kane nonparabolic optical mass, density-dependent damping, optionally a density-
dependent background), a TransportModel on the SAME Kane band (equilibrium electrostatics, static
permittivity, band gap, electron affinity) and a scattering model (optical damping and DC mobility). Every
number is a literature value cited below (authors, journal, year, DOI), with the spread between sources; a
number that is the library's own arithmetic on cited data is labelled DERIVED.

READ THIS FIRST -- PRESETS ARE STARTING POINTS. The free-carrier properties of these films depend strongly
on the deposition route (sputter, ALD, PLD, MBE, HiPIMS), temperature, thickness and anneal: the optical
mobility of AZO ranges 10-45 cm^2/(V s) between ALD and PLD films at the same density, and the fitted
eps_inf of doped ZnO ranges 2.9-3.9. A preset reproduces literature-typical films; for YOUR film, measure
n,k (spectroscopic ellipsometry) and the Hall density, fit (eps_inf, m_opt, gamma) with
materials.fit_drude_params, and pass the results back as overrides (eps_inf=, optical_mobility_m2Vs=, or
m0_ratio= / alpha_per_eV= for a mass law); docs/enz_presets.md shows the route.

CONVENTIONS. Kane band E (1 + alpha E) = hbar^2 k^2 / (2 m0): the optical (Drude) mass is the Fermi-surface
mass m_opt = m0 (1 + 2 alpha E_F) = m0 sqrt(1 + 2 alpha hbar^2 (3 pi^2 n)^(2/3) / m0) (KaneOpticalMass).
The "C" of the TCO literature in m* = m0* sqrt(1 + 2 C hbar^2 (3 pi^2 N)^(2/3) / m0*) is this alpha. One
exception, flagged where used: Nolen et al. (2020) print that formula but their fitted C = 1.47 eV^-1
reproduces their own data only WITHOUT the factor 2, i.e. alpha = 0.735 eV^-1 in the standard convention
(DERIVED: with the factor 2 their parameters give m* = 0.36 m_e at 4.1e20 cm^-3 against their measured
0.26-0.27). Damping: gamma = e / (m_opt mu_opt) with mu_opt the OPTICAL mobility (OpticalMobilityDamping).

SOURCES (status: P = read in the primary text (accepted manuscript or open-access version), A = abstract
only, S = secondary, quoted by another paper, D = derived here from cited inputs).

  ZnO host and doped ZnO films
  [Ash03]  N. Ashkenov et al., J. Appl. Phys. 93, 126 (2003), 10.1063/1.1526935 -- IR ellipsometry of bulk
           ZnO: eps_inf perp/par 3.70/3.78, eps_0 7.77/8.91 (S, via [PP11] Table I)
  [PP11]   J. Pellicer-Porres et al., Phys. Rev. B 84, 125202 (2011), 10.1103/PhysRevB.84.125202 --
           eps_inf 4.0(5)/3.8(3), eps_0 8.2(9)/8.7(6) perp/par (P)
  [Ser88]  B. E. Sernelius et al., Phys. Rev. B 37, 10244 (1988), 10.1103/PhysRevB.37.10244 -- eps_inf
           3.85, eps_0 8.0, m_h 0.59 m_e, Eg0 3.38 eV (P; values from Landolt-Boernstein)
  [Tsu08]  A. Tsukazaki et al., Phys. Rev. B 78, 233308 (2008), 10.1103/PhysRevB.78.233308 -- Eg 3.37 eV;
           MgZnO/ZnO polarization 2DEG 0.6-1.6e12 cm^-2 (P)
  [Rom16]  V. Romanyuk et al., Acta Phys. Pol. A 129, A-36 (2016), 10.12693/APhysPolA.129.A-36 -- ALD
           AZO: m0* = 0.24 m_e with C = 0.33 eV^-1; optical vs Hall mobility 1.55-2.38x (P)
  [Kim08]  W. M. Kim et al., J. Phys. D 41, 195409 (2008), 10.1088/0022-3727/41/19/195409 -- bare band
           mass 0.247 m_e, C = 0.33 eV^-1 (polaron 0.29 / 0.457) (A)
  [Rus09]  F. Ruske et al., Thin Solid Films 518, 1289 (2009), 10.1016/j.tsf.2009.03.218 -- sputtered AZO,
           m0* 0.24 (assumed), C = 0.27(2) eV^-1 (A)
  [Bal23]  A. Ball et al., J. Phys. Photonics 5, 024001 (2023), 10.1088/2515-7647/acbdd7 -- MBE GZO:
           6.07/5.28/5.17e20 cm^-3, cross-over 1560/1700/1710 nm, eps_inf 3.5/3.7/3.7, eps'' 0.31/0.41/0.31 (P)
  [Kin15]  N. Kinsey et al., Optica 2, 616 (2015), 10.1364/OPTICA.2.000616 -- PLD AZO 9.8e20 cm^-3,
           eps(1.3 um) = 0.22 + 0.26i (P, arXiv:1503.07832)
  [Nai12]  G. V. Naik et al., PNAS 109, 8834 (2012), 10.1073/pnas.1121517109 -- PLD AZO ~4.3e20 cm^-3,
           metallic beyond 1.84 um (P)
  [KimX13] J. Kim et al., Phys. Rev. X 3, 041037 (2013), 10.1103/PhysRevX.3.041037 -- PLD GZO Drude +
           Gaussians, hbar gamma 0.092 / 0.159 eV (P)
  [Kno15]  H. C. M. Knoops et al., J. Vac. Sci. Technol. A 33, 021509 (2015), 10.1116/1.4905086 -- PECVD
           AZO 7.5e20 cm^-3: Hall 30, FTIR 29, SE extended-Drude 28, SE plain-Drude 38 cm^2/(V s), all with
           m* = 0.4 m_e; the authors prefer the extended (frequency-dependent damping) Drude fit (P)
  [EM08]   K. Ellmer and R. Mientus, Thin Solid Films 516, 4620 (2008), 10.1016/j.tsf.2007.05.084 --
           ionized-impurity and Masetti mobility laws, Seto trap densities for ZnO (P, accepted manuscript)
  [Cor09]  S. Cornelius et al., Appl. Phys. Lett. 94, 042103 (2009), 10.1063/1.3074373 -- sputtered AZO
           6.0e20 cm^-3, Hall ~46 cm^2/(V s) (A)
  [Cla13]  B. Claflin, K. Leedy, D. Look, Opt. Eng. 52, 053801 (2013), 10.1117/1.OE.52.5.053801 -- PLD GZO
           8.05e20 cm^-3, Hall 32.1 cm^2/(V s) (P)
  [Gab14]  M. Gabas et al., APL Mater. 2, 012112 (2014), 10.1063/1.4863595 -- optical gap 3.63 eV at 5e20 (P)
  [Kle10]  A. Klein et al., Materials 3, 4892 (2010), 10.3390/ma3114892 -- AZO work function 3.1-4.5 eV,
           In2O3/ITO ionization potentials and gap (P)
  [Tek23]  M. Tekin et al., J. Appl. Phys. 134, 084503 (2023), 10.1063/5.0157726 -- sputtered ZnOx
           electron affinity 3.46 eV (XPS) / 3.72 eV (Fowler-Nordheim) (P)
  [BFV97]  F. Bernardini, V. Fiorentini, D. Vanderbilt, Phys. Rev. B 56, R10024 (1997),
           10.1103/PhysRevB.56.R10024 -- ZnO P_sp = -0.057 C/m^2, e33 0.89, e31 -0.51 C/m^2 (P)
  [GS05]   P. Gopal and N. A. Spaldin, arXiv:cond-mat/0507217 -- ZnO P_sp = -0.048 C/m^2 (P, preprint)

  CdO and In:CdO
  [Nol20]  K. Nolen et al., Phys. Rev. Materials 4, 025202 (2020), 10.1103/PhysRevMaterials.4.025202 --
           HiPIMS In:CdO 1.1e19-4.1e20: m0* 0.10, C 1.47 (convention: see above), Hall 426 / 389 / 312 at
           6.6e19 / 1.4e20 / 4.1e20 (P); eps_inf 5.5 -> 5.1 in the abstract, 5.5 -> 4.9 in the text (Fig. 2c);
           ENZ 3.87 / 2.72 / 1.92 um = 1e4 / (their screened plasma wavenumbers 2584 / 3680 / 5215 cm^-1) (D)
  [Liu16]  C. P. Liu et al., Phys. Rev. Applied 6, 064018 (2016), 10.1103/PhysRevApplied.6.064018 --
           sputtered (In,Ga):CdO: m0* 0.13, C 0.5; eps_inf ~5.5 -> 4.9 over 0-1e21; band-gap renormalization
           0.25-0.45 eV over 1e20-1.2e21 (Fig. 10); screened plasma wavelength 1.70 / 1.58 / 1.43 / 1.29 um at
           5.4 / 6.6 / 8.9 / 11e20; electron affinity 5.8 eV; eps_0 21.9 (P)
  [Men12]  R. J. Mendelsberg, Y. Zhu, A. Anders, J. Phys. D 45, 425302 (2012), 10.1088/0022-3727/45/42/425302
           -- arc In:CdO: m0* 0.16 +/- 0.05, C 0.5 +/- 0.2 (P)
  [Jef08]  P. H. Jefferson et al., Appl. Phys. Lett. 92, 022101 (2008), 10.1063/1.2833269 -- direct gap
           2.16 eV, band-edge mass 0.21 (k.p), non-dispersive valence band; eps_inf 5.3 (S) (P)
  [Cle21]  A. Cleri et al., Phys. Rev. Materials 5, 035202 (2021), 10.1103/PhysRevMaterials.5.035202 --
           HiPIMS In:CdO Hall mobility and optical gap vs density, Table I (P)
  [Yan17]  Y. Yang et al., Nat. Photonics 11, 390 (2017), 10.1038/nphoton.2017.64 -- In:CdO 2.8e20 cm^-3,
           ENZ 2.10 um, eps_inf 5.5, m* 0.20, mu 300 (P)
  [W15]    Wahila et al., Phys. Rev. B 91, 205307 (2015), 10.1103/PhysRevB.91.205307 -- uses eps_s = 21.9
           (S); quotes the DFT indirect gap 0.89 eV of Burbano et al. (JACS 2011) (S) (P)
  [Wan22]  H. Wang et al., APL Mater. 10, 121106 (2022), 10.1063/5.0116294 -- ion-gel gated CdO, sheet
           density 2e13 -> 2e14 cm^-2 (P)

  In2O3 and ITO
  [Mar24]  P. Markwitz et al., Phys. Rev. B 109, 115201 (2024), 10.1103/PhysRevB.109.115201 -- sputtered
           ITO, density set by nitrogen implantation (compensation): m0* 0.21, C 0.52 eV^-1 (Seebeck + Hall,
           2.9e20-1.01e21) (A)
  [Fen16]  M. Feneberg et al., Phys. Rev. B 93, 045203 (2016), 10.1103/PhysRevB.93.045203 -- m0* 0.18, C 0.5
           (S, via [Sto21]); eps_inf 4.08 (A)
  [Sto21]  M. Stokey et al., J. Appl. Phys. 129, 225102 (2021), 10.1063/5.0052848 -- eps_inf 4.05, eps_0
           10.55 (LST) for bulk In2O3 (P)
  [Hua16]  Y.-W. Huang et al., Nano Lett. 16, 5319 (2016), 10.1021/acs.nanolett.6b00555 -- sputtered ITO
           eps_inf 4.14-4.42, Hall 16-40 cm^2/(V s), 1.9-3.9e20 cm^-3 (P, SI Table S1)
  [Wal08]  A. Walsh et al., Phys. Rev. Lett. 100, 167402 (2008), 10.1103/PhysRevLett.100.167402 --
           fundamental gap <= 2.9 eV (P)
  [HG86]   I. Hamberg and C. G. Granqvist, J. Appl. Phys. 60, R123 (1986), 10.1063/1.337534 -- eps_0 8.9-9.5,
           eps_inf ~4 (S, via [Sto21])
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple


from dynameta.constants import M_E, T_REF
from dynameta.materials.material import Material
from dynameta.materials.optical_model import BandFillingBackground, DrudeOptical
from dynameta.materials.oxide_scattering import (MasettiMobility, OpticalMobilityDamping,
                                                 PolycrystallineScattering, SetoGrainBarrier, TabulatedMobility)
from dynameta.materials.transport_model import KaneBand, TransportModel
from dynameta.materials.uniaxial_optical import UniaxialOptical
from dynameta.warning_types import AdvisoryWarning

__all__ = ["azo", "gzo", "in_cdo", "ito", "fto", "TCOParameter", "PARAMETER_TABLES", "VALIDITY_RANGES_M3",
           "dual_gate_film_kwargs", "wurtzite_polarization_charges", "zno_seto_barrier",
           "CDO_HIPIMS_HALL_MOBILITY", "CDO_BAND_FILLING", "ZNO_P_SP_C_M2", "ZNO_E33_C_M2", "ZNO_E31_C_M2",
           "ZNO_GAP_EV", "ZNO_HOLE_MASS_RATIO"]

_CM3 = 1.0e6
_CM2VS = 1.0e-4


@dataclass(frozen=True)
class TCOParameter:
    """One row of a preset's parameter table: the default, the literature spread, the sources and status."""
    quantity: str
    default: str
    spread: str
    sources: str
    status: str


# =====================================================================================================
# ZnO host constants and doped-ZnO film parameters (shared by azo and gzo)
# =====================================================================================================

ZNO_GAP_EV = 3.37                 # [Tsu08] (FX 3.312 eV + 60 meV binding agrees); [Ser88] 3.38
ZNO_HOLE_MASS_RATIO = 0.59        # [Ser88] (Landolt-Boernstein); for the Burstein-Moss recoil term
ZNO_P_SP_C_M2 = -0.057            # [BFV97] Table II (Berry phase, LDA); [GS05] -0.048
ZNO_E33_C_M2 = 0.89               # [BFV97]; experimental 0.96-1.22 (S)
ZNO_E31_C_M2 = -0.51              # [BFV97]; experimental -0.51 .. -0.62 (S)

_ZNO_EPS_INF = 3.6                # doped-film single-background value; [Bal23] 3.5-3.7, host [Ash03] 3.70/3.78
_ZNO_EPS_INF_ANISOTROPY = 0.08    # eps_inf,par - eps_inf,perp of the host [Ash03] (S)
_ZNO_M0 = 0.24                    # [Rom16] (P), [Kim08] 0.247 (A), [Rus09] 0.24 (A)
_ZNO_ALPHA = 0.33                 # [Rom16] (P), [Kim08] (A); [Rus09] 0.27 (A)
_ZNO_MU_OPT = 38.0 * _CM2VS       # near-IR optical mobility of good films: [Bal23] 44 (D), [Kin15] 40 (D),
                                  # [KimX13] 34 (D), [Kno15] 38 (near-IR plain-Drude fit; 28 extended Drude)
_ZNO_EPS_STATIC = 8.7             # [PP11] eps_0,par 8.7 (P); [Ash03] 8.91 (S); isotropic 8.0 [Ser88]
_ZNO_RANGE = (1.0e20 * _CM3, 1.0e21 * _CM3)

_ZNO_TABLE = (
    TCOParameter("eps_inf (single background, films)", "3.6", "2.9-3.9 film fits ([Rom16] 3.88->2.92 over "
                 "0.9-4e20 IR; [Bal23] 3.5-3.7; [KimX13] effective 3.3-3.5); host 3.70/3.78 perp/par [Ash03], "
                 "3.85 [Ser88]", "[Bal23], [Ash03], [Ser88], [Rom16]", "P / S"),
    TCOParameter("band-edge mass m0*", "0.24 m_e", "0.24-0.28 (bare 0.23-0.28; polaron 0.29-0.32)",
                 "[Rom16], [Kim08], [Rus09]", "P / A"),
    TCOParameter("Kane nonparabolicity alpha (= C)", "0.33 eV^-1", "0.27-0.33 peer-reviewed; 0.56 [Bikowski "
                 "2014] (S, outlier); 0.142 [Fujiwara 2005] (S)", "[Rom16], [Kim08], [Rus09]", "P / A"),
    TCOParameter("optical mass m_opt(n)", "0.287 / 0.361 / 0.416 m_e at 1e20 / 5e20 / 1e21 cm^-3",
                 "+/-5 % between the (m0*, C) pairs above; ENZ-implied 0.34-0.42 at 4.3e20-9.8e20 (D)",
                 "derived from the row above", "D"),
    TCOParameter("near-IR optical mobility mu_opt", "38 cm^2/(V s)", "34-44 good PLD/MBE/PECVD films at "
                 "6e20-1e21 (PECVD [Kno15]: 38 plain-Drude, 28 extended-Drude); ALD 10-33 (FTIR) [Rom16]",
                 "[Kno15], [Bal23], [Kin15], [KimX13]", "P / D"),
    TCOParameter("DC intra-grain mobility", "Masetti fit of [EM08] Table 2: 54/35/18 cm^2/(V s) at "
                 "1e20/5e20/1e21", "best Hall ~46 at 6e20 [Cor09], 32.1 at 8e20 [Cla13]", "[EM08]", "P"),
    TCOParameter("grain-boundary trap density Q_t (optional Seto)", "none (single-crystal-like)",
                 "0.5-1.5e13 cm^-2 epitaxial, 3e13 polycrystalline sputtered, 1.8-4.2e13 ALD",
                 "[EM08] Tables 3-4, [Cor09], [Rom16]", "P"),
    TCOParameter("static permittivity eps_0", "8.7 (along c)", "7.77-8.2 perp, 8.7-8.91 par; 8.0 isotropic",
                 "[PP11], [Ash03], [Ser88]", "P / S"),
    TCOParameter("band gap", "3.37 eV", "3.37-3.40 (300 K)", "[Tsu08], [Ser88]", "P"),
    TCOParameter("electron affinity chi", "None (set yours)", "3.46-3.72 sputtered ZnOx [Tek23]; 3.6-4.5 from "
                 "AZO ionization potentials [Kle10] (D); ~4.2 single crystal (not verified)", "[Tek23], [Kle10]",
                 "P / D"),
    TCOParameter("spontaneous polarization P_sp", "-0.057 C/m^2 (not applied by default)", "-0.048 .. -0.057",
                 "[BFV97], [GS05]", "P"),
)

_AZO_TABLE = _ZNO_TABLE + (
    TCOParameter("density range", "1e20-1e21 cm^-3", "ALD 0.9-4e20 [Rom16]; sputter 4.3-6.9e20 [Cor09]; PLD "
                 "9.8e20 [Kin15]", "[Rom16], [Cor09], [Kin15]", "P"),
)
_GZO_TABLE = _ZNO_TABLE + (
    TCOParameter("density range", "1e20-1e21 cm^-3", "MBE 5.2-6.1e20 [Bal23]; PLD 8.05e20 (N_D 1.05e21) "
                 "[Cla13]; sputter ~5e20 [Gab14]", "[Bal23], [Cla13], [Gab14]", "P"),
)


# =====================================================================================================
# CdO / In:CdO
# =====================================================================================================

#: Hall mobility of HiPIMS In:CdO on r-sapphire, annealed ([Cle21] Table I): (n [cm^-3], mu [cm^2/(V s)]).
#: [Nol20] (HiPIMS on MgO) gives 426 / 389 / 312 at 6.6e19 / 1.4e20 / 4.1e20: +23 / +8 / -6 % from this table.
CDO_HIPIMS_HALL_MOBILITY = TabulatedMobility(
    n_m3=tuple(v * _CM3 for v in (2.2e19, 2.5e19, 1.2e20, 2.9e20, 3.5e20, 5.0e20)),
    mobility_m2Vs=tuple(v * _CM2VS for v in (298.0, 315.0, 365.0, 331.0, 334.0, 326.0)),
    extrapolate="hold",
    source="Cleri et al., Phys. Rev. Materials 5, 035202 (2021), Table I (HiPIMS In:CdO, Hall)")

_CDO_M0 = 0.10                    # [Nol20] as plotted (P)
_CDO_ALPHA = 0.735                # [Nol20] C = 1.47 without the factor 2 -> standard alpha (D, see module doc)
_CDO_EPS_INF = 5.5                # at the anchor density 1.1e19 cm^-3 ([Nol20] lowest-density film); [Yan17] 5.5
_CDO_N_REF = 1.1e19 * _CM3
_CDO_EPS_STATIC = 21.9            # Finkenrath and Uhle (1967) via [Liu16], [W15] (S)
_CDO_GAP_DIRECT = 2.16            # [Jef08] (P)
_CDO_GAP_INDIRECT = 0.9           # DFT 0.89 eV via [W15]; ~0.9 eV (XAS, Demchenko 2011, A)
_CDO_CHI = 5.8                    # [Liu16] (P); [Nol20] Fermi stabilization 4.9 eV below vacuum, ~1 eV above
                                  # the CBM (S) -> ~5.9 (D)
_CDO_BGR = 4.8e-10                # eV m: [Liu16] Fig. 10, 0.25 eV at 1e20 and 0.45 eV at 1.2e21 cm^-3 read off
                                  # the figure -> 5.4e-10 / 4.2e-10; mean (D)
_CDO_EDGE_AMPLITUDE = 7.81        # D: fitted so that eps_b(4.1e20) - eps_b(1.1e19) = -0.40 at 2 um ([Nol20]
                                  # abstract, 5.5 -> 5.1); it gives 4.87 at 1e21 ([Liu16] ~4.9). [Nol20]'s text
                                  # (5.5 -> 4.9 over its films) would need a larger amplitude
_CDO_RANGE = (1.0e19 * _CM3, 1.2e21 * _CM3)


def _cdo_band_filling(band: KaneBand, edge_amplitude: float = _CDO_EDGE_AMPLITUDE) -> BandFillingBackground:
    return BandFillingBackground(gap_eV=_CDO_GAP_DIRECT, edge_amplitude=float(edge_amplitude), n_ref_m3=_CDO_N_REF,
                                 conduction=band, valence_mass_kg=None, bgr_coeff_eV_m=_CDO_BGR)


#: The CdO density-dependent background of the in_cdo preset (Burstein-Moss band filling + gap
#: renormalization of the direct Gamma edge; [Jef08] gap and non-dispersive valence band, [Liu16]
#: renormalization, amplitude fitted to [Nol20]): eps_b(n) = 5.5 + chi(n) - chi(1.1e19 cm^-3).
CDO_BAND_FILLING = _cdo_band_filling(KaneBand(_CDO_M0 * M_E, _CDO_ALPHA))

_CDO_TABLE = (
    TCOParameter("eps_inf", "5.5 at 1.1e19 cm^-3, falling with density (band filling, below)",
                 "5.3-5.6 at low density; 5.1 (abstract) or 4.9 (text) at 4.1e20 [Nol20]; ~4.9 at 1e21 [Liu16]",
                 "[Nol20], [Liu16], [Yan17], [Jef08]", "P / S"),
    TCOParameter("band-edge mass m0*", "0.10 m_e", "0.10 [Nol20], 0.13 [Liu16], 0.16 +/- 0.05 [Men12]; "
                 "0.21 k.p [Jef08] (inconsistent with ENZ data)", "[Nol20], [Liu16], [Men12], [Jef08]", "P"),
    TCOParameter("Kane nonparabolicity alpha", "0.735 eV^-1", "0.5 +/- 0.2 [Liu16], [Men12]; 1/Eg = 0.46 "
                 "[W15]; 0.735 = [Nol20] converted (D)", "[Nol20], [Liu16], [Men12]", "P / D"),
    TCOParameter("optical mass m_opt(n)", "0.166 / 0.197 / 0.263 / 0.342 m_e at 0.66 / 1.4 / 4.1 / 10e20",
                 "[Liu16] law: 0.193 / 0.277 / 0.334 at 1 / 5 / 10e20 (within 3 % above 4e20)", "derived", "D"),
    TCOParameter("mobility (optical = Hall assumed)", "HiPIMS table [Cle21]: 298-365 cm^2/(V s), 2.2e19-5e20",
                 "[Nol20] 312-426; [Yan17] 300; sputtered polycrystalline ~65-130 [Liu16]; optical <= Hall by up "
                 "to ~20 % [Men12], [Liu16]", "[Cle21], [Nol20], [Yan17]", "P"),
    TCOParameter("background: direct gap / BGR / edge amplitude", "2.16 eV / 4.8e-10 eV m / 7.81",
                 "gap 2.16 +/- 0.02 [Jef08]; BGR 4.2-5.4e-10 [Liu16] Fig. 10 (D); amplitude fitted to [Nol20]'s "
                 "abstract value, consistent with [Liu16] at 1e21 (D)", "[Jef08], [Liu16], [Nol20]", "P / D"),
    TCOParameter("static permittivity eps_0", "21.9", "single source", "[Liu16], [W15] (Finkenrath 1967)", "S"),
    TCOParameter("band gap (fundamental, indirect)", "0.9 eV", "0.84-1.09 (Koffyberg 1976, not verified); 0.89 "
                 "DFT", "[W15]", "S"),
    TCOParameter("electron affinity chi", "5.8 eV", "5.8-5.9", "[Liu16]; [Nol20] (D)", "P / D"),
    TCOParameter("density range", "1e19-1.2e21 cm^-3 (default mobility table 2.2e19-5e20)",
                 "HiPIMS 1.1e19-5e20 [Nol20], [Cle21]; sputtered to 1.3e21 [Liu16]", "[Nol20], [Cle21], [Liu16]",
                 "P"),
)


# =====================================================================================================
# In2O3 / ITO
# =====================================================================================================

_ITO_EPS_INF = 4.0                # crystals 4.05-4.08 [Sto21], [Fen16]; sputtered films 3.9-4.4 [Hua16]
_ITO_M0 = 0.21                    # [Mar24] sputtered ITO (P)
_ITO_ALPHA = 0.52                 # [Mar24] (P); [Fen16] 0.5 (S)
_ITO_MU_OPT = 30.0 * _CM2VS       # [Hua16] Hall 16-40 cm^2/(V s) (P); gamma 1.7e14 rad/s at 3e20 cm^-3
_ITO_EPS_STATIC = 9.0             # [HG86] 8.9-9.5 (S); DFT 9.05 (S); [Sto21] LST 10.55 (P, bulk crystal)
_ITO_GAP = 2.8                    # [Kle10] 2.8 +/- 0.2 (P); [Wal08] <= 2.9 (P)
_ITO_RANGE = (1.0e20 * _CM3, 1.2e21 * _CM3)

_ITO_TABLE = (
    TCOParameter("eps_inf", "4.0", "4.05-4.08 bulk crystal; 3.9-4.42 sputtered films", "[Sto21], [Fen16], "
                 "[Hua16]", "P / A"),
    TCOParameter("band-edge mass m0* / alpha", "0.21 m_e / 0.52 eV^-1", "0.18 / 0.5 crystalline and epitaxial "
                 "[Fen16]; 0.263 / 0.42 (Liu 2014, S); 0.297 / 0.18 (Fujiwara 2005, S)", "[Mar24], [Fen16]",
                 "P / S"),
    TCOParameter("optical mass m_opt(n)", "0.280 / 0.380 / 0.451 m_e at 1e20 / 5e20 / 1e21", "+/-15 % at 5e20 "
                 "between the laws above", "derived", "D"),
    TCOParameter("optical mobility", "30 cm^2/(V s)", "16-40 sputtered films", "[Hua16]", "P"),
    TCOParameter("static permittivity eps_0", "9.0", "8.9-9.5 films (S); 10.55 bulk LST (P)", "[HG86], [Sto21]",
                 "S / P"),
    TCOParameter("band gap (fundamental)", "2.8 eV", "2.75-2.9; dipole-allowed onset ~3.7-3.8 eV",
                 "[Kle10], [Wal08]", "P"),
    TCOParameter("electron affinity chi", "None (set yours)", "~4.2-5.3 from ionization potentials 7.0-8.1 eV "
                 "minus the gap (D); 5 eV used as a fit parameter by [Hua16]", "[Kle10], [Hua16]", "D"),
    TCOParameter("density range", "1e20-1.2e21 cm^-3", "sputtered 1.9-3.9e20 [Hua16], 2.9e20-1.01e21 [Mar24]; "
                 "e-beam to 1.2e21 (Korneluk 2023)", "[Hua16], [Mar24]", "P"),
)

# =====================================================================================================
# SnO2:F (FTO)
# =====================================================================================================

_FTO_EPS_INF = 3.9                # [Rey13] 3.7-4.1 (P); [Fen19] 3.70 / 3.90 perp / par (S); VR96 ~4.1 (D)
_FTO_M0 = 0.09                    # [Rey13] Eq. (5), fit to polycrystalline FTO film masses (P)
_FTO_ALPHA = 0.8                  # [Rey13] (P); epitaxial SnO2 [Fen19]: 0.197 / 0.56 (orientation average)
_FTO_MU_OPT = 35.0 * _CM2VS       # [Rey13] Table I: 31-39 (mean ~34) cm^2/(V s) at 2.2-4.5e20 (D from tau, m*)
_FTO_EPS_STATIC = 12.3            # orientation average of 14 / 9 (van Daal 1968 via [Sch11], S); 11.5 [EM08] (S)
_FTO_GAP = 3.56                   # fundamental (dipole-forbidden) gap [Fen19] (P)
_FTO_RANGE = (1.5e19 * _CM3, 5.0e20 * _CM3)

_FTO_TABLE = (
    TCOParameter("eps_inf", "3.9", "3.7-4.2", "[Rey13], [Fen19], von Rottkay & Rubin 1996", "P / S / D"),
    TCOParameter("band-edge mass m0* / alpha", "0.09 m_e / 0.8 eV^-1 (polycrystalline films)", "epitaxial SnO2: "
                 "0.205 / 0.73 (perp), 0.183 / 0.33 (par), DOS average 0.197 / 0.56 [Fen19]; 0.094 / 1.035 "
                 "(Pisarkiewicz 1990, S); cyclotron band-edge mass 0.23-0.30 (S)", "[Rey13], [Fen19]", "P"),
    TCOParameter("optical mass m_opt(n)", "0.171 / 0.215 / 0.261 m_e at 0.9 / 2.2 / 4.3e20", "film data 0.16 / "
                 "0.23 / 0.24-0.29 [Rey13]; epitaxial law ~35 % heavier at 4.3e20", "derived", "D"),
    TCOParameter("optical mobility", "35 cm^2/(V s)", "31-39 at 2.2-4.5e20, ~50 at 9e19 [Rey13]; in-grain 37-53",
                 "[Rey13]", "P / D"),
    TCOParameter("DC intra-grain mobility", "Masetti SnO2 fit [EM08] Table 2", "epitaxial Hall 62 / 47 / 35 at "
                 "1.6e19 / 1.3e20 / 2.6e20 [Fen19]; polycrystalline ~20", "[EM08], [Fen19]", "P"),
    TCOParameter("static permittivity", "12.3 (orientation average)", "14 perp / 9 par; 11.5", "[Sch11], [EM08]",
                 "S"),
    TCOParameter("band gap", "3.56 eV (fundamental)", "dipole-allowed onset 4.29-4.36 eV", "[Fen19]", "P"),
    TCOParameter("electron affinity", "None (set yours)", "4.3-5.3 from ionization potentials 7.9-8.9 eV minus "
                 "the gap (D)", "[Kle10]", "D"),
    TCOParameter("density range", "1.5e19-5e20 cm^-3", "spray pyrolysis 2e19-4.5e20 [Rey13]; MBE 1.6e19-2.6e20 "
                 "[Fen19]", "[Rey13], [Fen19]", "P"),
)

#: Literature parameter table of each preset (rows: TCOParameter), for documentation and reporting.
PARAMETER_TABLES = {"azo": _AZO_TABLE, "gzo": _GZO_TABLE, "in_cdo": _CDO_TABLE, "ito": _ITO_TABLE,
                    "fto": _FTO_TABLE}

#: Construction-density validity range [m^-3] of each preset (the background density n_bg; the per-node
#: densities of a gated film may leave it -- every closure stays finite down to n = 0).
VALIDITY_RANGES_M3 = {"azo": _ZNO_RANGE, "gzo": _ZNO_RANGE, "in_cdo": _CDO_RANGE, "ito": _ITO_RANGE,
                      "fto": _FTO_RANGE}


# =====================================================================================================
# assembly
# =====================================================================================================

def _check_density(n_bg_m3, key: str, strict: bool) -> float:
    return _check_density_range(n_bg_m3, key, VALIDITY_RANGES_M3[key], strict)


def _check_density_range(n_bg_m3, key: str, rng, strict: bool) -> float:
    """n_bg [m^-3] checked: finite, > 0, not a cm^-3 slip, inside rng (ValueError, or AdvisoryWarning with
    strict=False)."""
    n = float(n_bg_m3)
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError("{}: n_bg_m3 must be finite and > 0 (m^-3), got {!r}".format(key, n_bg_m3))
    if 1e15 <= n <= 1e23:
        raise ValueError("{}: n_bg_m3 = {:.4g} looks like a density in cm^-3; densities are in m^-3 "
                         "(1e20 cm^-3 = 1e26 m^-3)".format(key, n))
    lo, hi = rng
    if not (lo <= n <= hi):
        msg = ("{}: n_bg = {:.4g} m^-3 ({:.4g} cm^-3) is outside the preset's literature range [{:.3g}, {:.3g}] "
               "cm^-3 (the films the parameters come from are listed in the preset's docstring)".format(
                   key, n, n / _CM3, lo / _CM3, hi / _CM3))
        if strict:
            raise ValueError(msg + "; pass strict=False to build it anyway (an extrapolation) or set the "
                             "parameters from your own film.")
        warnings.warn(msg + " -- extrapolating (strict=False).", AdvisoryWarning, stacklevel=3)
    return n


def _positive_or_none(v, name, where):
    if v is None:
        return None
    x = float(v)
    if not (math.isfinite(x) and x > 0.0):
        raise ValueError("{}: {} must be finite and > 0, got {!r}".format(where, name, v))
    return x


def _mobility_spec(v, name, where):
    if callable(v):
        return v
    x = float(v)
    if not (math.isfinite(x) and x > 0.0):
        raise ValueError("{}: {} must be finite and > 0 (m^2/(V s)) or a callable of n, got {!r}".format(
            where, name, v))
    if x > 1.0:
        raise ValueError("{}: {} = {!r} m^2/(V s) is implausible -- mobilities are in m^2/(V s) "
                         "(30 cm^2/(V s) = 30e-4 m^2/(V s))".format(where, name, v))
    return x


def _build(*, key: str, name: str, n_bg: float, eps_inf: float, m0_ratio: float, alpha: float, mu_opt,
           mu_dc_intragrain, grain_boundary, eps_static: float, band_gap_eV: Optional[float],
           chi_eV: Optional[float], background, eps_inf_extraordinary: Optional[float]) -> Material:
    where = key
    eps_inf = _positive_or_none(eps_inf, "eps_inf", where)
    m0_ratio = _positive_or_none(m0_ratio, "m0_ratio", where)
    a = float(alpha)
    if not (math.isfinite(a) and a >= 0.0):
        raise ValueError("{}: alpha_per_eV must be finite and >= 0, got {!r}".format(where, alpha))
    eps_static = _positive_or_none(eps_static, "eps_static", where)
    band = KaneBand(m0_kg=m0_ratio * M_E, alpha_per_eV=a)
    mass = band.optical_mass()
    damping = OpticalMobilityDamping(mobility_m2Vs=_mobility_spec(mu_opt, "optical_mobility_m2Vs", where), mass=mass)
    scattering = PolycrystallineScattering(
        one_over_tau=damping, m_cond_kg=mass,
        intragrain_mobility=None if mu_dc_intragrain is None else _mobility_spec(
            mu_dc_intragrain, "dc_mobility_m2Vs", where),
        grain_boundary=grain_boundary)
    drude = DrudeOptical(eps_inf=eps_inf, m_opt_kg=mass, gamma_rad_s=damping, background=background)
    transport = TransportModel(n_bg_m3=n_bg, eps_static=eps_static, band=band, band_gap_eV=band_gap_eV,
                               chi_eV=chi_eV, chi_reference_T_K=300.0, physics="equilibrium")
    if eps_inf_extraordinary is None:
        return Material(name=name, optical=drude, transport=transport, scattering=scattering)
    # uniaxial: Material.scattering needs a scalar DrudeOptical, so the scattering law is applied directly
    # (the same gamma(n) and mu(n) Material(scattering=...) would derive)
    ext = dataclasses.replace(drude, eps_inf=_positive_or_none(eps_inf_extraordinary, "eps_inf_extraordinary",
                                                                where))
    transport = dataclasses.replace(transport, mobility_m2Vs_of_n_m3=scattering.mobility_of_n())
    return Material(name=name, optical=UniaxialOptical(ordinary=drude, extraordinary=ext), transport=transport)


def _zno(key, n_bg_m3, *, name, eps_inf, m0_ratio, alpha_per_eV, optical_mobility_m2Vs, dc_mobility_m2Vs,
         grain_boundary, eps_static, band_gap_eV, chi_eV, background, uniaxial, strict):
    n = _check_density(n_bg_m3, key, strict)
    dc = MasettiMobility.zno_ellmer_mientus_2008() if dc_mobility_m2Vs == "masetti" else dc_mobility_m2Vs
    ext = (float(eps_inf) + _ZNO_EPS_INF_ANISOTROPY) if uniaxial else None
    return _build(key=key, name=name, n_bg=n, eps_inf=eps_inf, m0_ratio=m0_ratio, alpha=alpha_per_eV,
                  mu_opt=optical_mobility_m2Vs, mu_dc_intragrain=dc, grain_boundary=grain_boundary,
                  eps_static=eps_static, band_gap_eV=band_gap_eV, chi_eV=chi_eV, background=background,
                  eps_inf_extraordinary=ext)


_ZNO_DOC = """

    Parameters (SI; defaults and sources in PARAMETER_TABLES['{key}'] and the module docstring):
      n_bg_m3              : background (donor) electron density [m^-3]; literature range 1e20-1e21 cm^-3
                             (outside it: ValueError, or an AdvisoryWarning with strict=False).
      eps_inf              : background permittivity of the Drude term, 3.6 (film fits 2.9-3.9).
      m0_ratio, alpha_per_eV : Kane band 0.24 m_e, 0.33 eV^-1 ([Rom16]); one band for optics and electrostatics.
      optical_mobility_m2Vs: near-IR optical mobility [m^2/(V s)] or a callable of n; 38e-4 (good PLD/MBE/PECVD
                             films; 10e-4-33e-4 for ALD). The film-quality knob: gamma(n) = e / (m_opt(n) mu_opt).
      dc_mobility_m2Vs     : DC intra-grain mobility for transport: "masetti" (default; [EM08] fit of the best
                             single-crystal / epitaxial ZnO), a float or a callable of n.
      grain_boundary       : None (default) or a SetoGrainBarrier (zno_seto_barrier(...)) -- in series with the
                             intra-grain DC mobility; the optical damping is unaffected.
      eps_static           : static permittivity, 8.7 (along c: the gate field of a c-textured film); 8.0-8.3
                             isotropic.
      band_gap_eV          : 3.37. chi_eV: None (literature spread 3.5-4.5 eV; required by the builders only
                             for a heterostack -- set your value).
      background           : None (constant eps_inf, default) or a DrudeOptical background model (e.g. a
                             BandFillingBackground built with ZNO_GAP_EV and ZNO_HOLE_MASS_RATIO; no verified
                             renormalization coefficient for ZnO exists, so none is shipped).
      uniaxial             : False (isotropic, default) or True: a UniaxialOptical with eps_inf,par = eps_inf +
                             0.08 ([Ash03]; c axis = z). The 2 % anisotropy is well below the film-to-film spread
                             of eps_inf; the tensor is for studies that resolve it.
      strict               : refuse (True, default) or warn (False) for n_bg outside the range.
    Returns a Material (DrudeOptical + KaneBand equilibrium TransportModel + PolycrystallineScattering)."""


def azo(n_bg_m3: float, *, name: str = "AZO", eps_inf: float = _ZNO_EPS_INF, m0_ratio: float = _ZNO_M0,
        alpha_per_eV: float = _ZNO_ALPHA, optical_mobility_m2Vs=_ZNO_MU_OPT, dc_mobility_m2Vs="masetti",
        grain_boundary=None, eps_static: float = _ZNO_EPS_STATIC, band_gap_eV: Optional[float] = ZNO_GAP_EV,
        chi_eV: Optional[float] = None, background=None, uniaxial: bool = False, strict: bool = True) -> Material:
    """Al-doped ZnO film preset. The free carriers live in the ZnO host conduction band, so the band,
    damping and permittivities are the ZnO-host / doped-ZnO-film literature values shared with gzo(); the
    dopant sets the achievable density and activation (Al solubility ~3 at.%, [Cor09]), not the band within
    the resolution of the literature. Reference points (tests/test_tco_presets.py): PLD AZO metallic beyond
    1.84 um at ~4.3e20 cm^-3 [Nai12]; eps(1.3 um) = 0.22 + 0.26i at 9.8e20 cm^-3 [Kin15]."""
    return _zno("azo", n_bg_m3, name=name, eps_inf=eps_inf, m0_ratio=m0_ratio, alpha_per_eV=alpha_per_eV,
                optical_mobility_m2Vs=optical_mobility_m2Vs, dc_mobility_m2Vs=dc_mobility_m2Vs,
                grain_boundary=grain_boundary, eps_static=eps_static, band_gap_eV=band_gap_eV, chi_eV=chi_eV,
                background=background, uniaxial=uniaxial, strict=strict)


def gzo(n_bg_m3: float, *, name: str = "GZO", eps_inf: float = _ZNO_EPS_INF, m0_ratio: float = _ZNO_M0,
        alpha_per_eV: float = _ZNO_ALPHA, optical_mobility_m2Vs=_ZNO_MU_OPT, dc_mobility_m2Vs="masetti",
        grain_boundary=None, eps_static: float = _ZNO_EPS_STATIC, band_gap_eV: Optional[float] = ZNO_GAP_EV,
        chi_eV: Optional[float] = None, background=None, uniaxial: bool = False, strict: bool = True) -> Material:
    """Ga-doped ZnO film preset (same host parameters as azo(); Ga activates more efficiently: ~0.7 vs 0.04
    electrons per dopant at 1 at.% in sputtered films [Gab14], and PLD reaches N_D ~1e21 cm^-3 [Cla13]).
    Reference points: MBE GZO cross-over 1560 / 1700 / 1710 nm at 6.07 / 5.28 / 5.17e20 cm^-3 [Bal23]."""
    return _zno("gzo", n_bg_m3, name=name, eps_inf=eps_inf, m0_ratio=m0_ratio, alpha_per_eV=alpha_per_eV,
                optical_mobility_m2Vs=optical_mobility_m2Vs, dc_mobility_m2Vs=dc_mobility_m2Vs,
                grain_boundary=grain_boundary, eps_static=eps_static, band_gap_eV=band_gap_eV, chi_eV=chi_eV,
                background=background, uniaxial=uniaxial, strict=strict)


azo.__doc__ += _ZNO_DOC.format(key="azo")
gzo.__doc__ += _ZNO_DOC.format(key="gzo")


def in_cdo(n_bg_m3: float, *, name: str = "ICO", eps_inf: float = _CDO_EPS_INF, m0_ratio: float = _CDO_M0,
           alpha_per_eV: float = _CDO_ALPHA, optical_mobility_m2Vs=None, background="band_filling",
           edge_amplitude: float = _CDO_EDGE_AMPLITUDE, eps_static: float = _CDO_EPS_STATIC,
           band_gap_eV: Optional[float] = _CDO_GAP_INDIRECT, chi_eV: Optional[float] = _CDO_CHI,
           strict: bool = True) -> Material:
    """In-doped CdO (rock salt) film preset, HiPIMS-type epitaxial / large-grain films.

    CdO-specific physics carried by the preset: a light, strongly nonparabolic band (0.10 m_e, alpha 0.735
    eV^-1: the optical mass rises 2.7x from the band edge to 1e21 cm^-3), a high mobility (~300-400 cm^2/(V s),
    an order of magnitude above doped ZnO and ITO, hence a narrow, low-loss ENZ resonance) and a background
    that FALLS with density (5.5 -> 5.1 over 1.1e19-4.1e20 cm^-3 [Nol20]: band filling of the direct 2.16 eV
    edge), which moves the ENZ wavelength by ~4 % and adds to the gate response with the same sign as the
    Drude term. The default background (background="band_filling") is CDO_BAND_FILLING with the given
    edge_amplitude; it is evaluated below the renormalized direct gap only (1.78 eV at 5e20 cm^-3: wavelengths
    above ~0.7 um) and raises above it. background=None gives a constant eps_inf (also the route to the FDTD
    single-pole Drude, which cannot carry a density-dependent background); a callable / TabulatedBackground
    is passed through. The INDIRECT gap (~0.9 eV) is the fundamental gap (band_gap_eV) but its phonon-assisted
    absorption is weak and, with the conduction band filled to E_F ~ 0.3-1 eV, sits above the ENZ photon
    energies (an inference from [Men12], [Liu16], [Nol20]: no extra near-IR absorption beyond Drude).

    Parameters (SI; PARAMETER_TABLES['in_cdo']):
      n_bg_m3              : 1e19-1.2e21 cm^-3 (mass law); the default mobility table covers 2.2e19-5e20 cm^-3
                             and a density outside it requires optical_mobility_m2Vs.
      eps_inf              : the background at the anchor density 1.1e19 cm^-3 (5.5; constant if background=None).
      m0_ratio, alpha_per_eV : 0.10 m_e, 0.735 eV^-1 ([Nol20] in the standard convention; [Liu16] 0.13 / 0.5 is
                             the polycrystalline alternative and agrees within 3 % above 4e20 cm^-3).
      optical_mobility_m2Vs: None = CDO_HIPIMS_HALL_MOBILITY (optical = Hall assumed, as [Yan17] does; optical
                             can be up to ~20 % lower [Men12], [Liu16]); a float or callable otherwise
                             (polycrystalline sputtered films: ~65e-4-130e-4 [Liu16]). Also the DC mobility.
      edge_amplitude       : amplitude of the direct edge in the band-filling background (7.81: eps_inf 5.5 ->
                             5.1 over 1.1e19-4.1e20 cm^-3 as in [Nol20]'s abstract, and 4.87 at 1e21 as [Liu16]'s
                             ~4.9; [Nol20]'s text quotes 4.9 already at 4.1e20, which needs a larger amplitude).
      eps_static 21.9, band_gap_eV 0.9 (indirect, fundamental), chi_eV 5.8.
    Reference points (tests): ENZ 3.87 / 2.72 / 1.92 um at 6.6e19 / 1.4e20 / 4.1e20 cm^-3 ([Nol20]'s screened
    plasma frequencies, D); 2.10 um at
    2.8e20 [Yan17]; screened plasma wavelength 1.70 / 1.58 / 1.43 / 1.29 um at 5.4 / 6.6 / 8.9 / 11e20 [Liu16]
    (independent films: polycrystalline sputtered); optical gap vs density [Cle21]."""
    n = _check_density(n_bg_m3, "in_cdo", strict)
    band = KaneBand(m0_kg=float(m0_ratio) * M_E, alpha_per_eV=float(alpha_per_eV))
    if optical_mobility_m2Vs is None:
        lo, hi = CDO_HIPIMS_HALL_MOBILITY.density_range_m3
        if not (lo * (1.0 - 1e-12) <= n <= hi * (1.0 + 1e-12)):
            raise ValueError("in_cdo: n_bg = {:.4g} cm^-3 is outside the default mobility table [{:.3g}, {:.3g}] "
                             "cm^-3 ([Cle21]); pass optical_mobility_m2Vs for your film".format(
                                 n / _CM3, lo / _CM3, hi / _CM3))
        mu = CDO_HIPIMS_HALL_MOBILITY
    else:
        mu = optical_mobility_m2Vs
    if isinstance(background, str):
        if background != "band_filling":
            raise ValueError("in_cdo: background must be 'band_filling', None, or a background model; got {!r}"
                             .format(background))
        bg = _cdo_band_filling(band, edge_amplitude)
    else:
        bg = background
    return _build(key="in_cdo", name=name, n_bg=n, eps_inf=eps_inf, m0_ratio=m0_ratio, alpha=alpha_per_eV,
                  mu_opt=mu, mu_dc_intragrain=None, grain_boundary=None, eps_static=eps_static,
                  band_gap_eV=band_gap_eV, chi_eV=chi_eV, background=bg, eps_inf_extraordinary=None)


def ito(n_bg_m3: float, *, name: str = "ITO", eps_inf: float = _ITO_EPS_INF, m0_ratio: float = _ITO_M0,
        alpha_per_eV: float = _ITO_ALPHA, optical_mobility_m2Vs=_ITO_MU_OPT, dc_mobility_m2Vs=None,
        grain_boundary=None, eps_static: float = _ITO_EPS_STATIC, band_gap_eV: Optional[float] = _ITO_GAP,
        chi_eV: Optional[float] = None, background=None, strict: bool = True) -> Material:
    """Sn-doped In2O3 (ITO) film preset, sputtered films.

    Parameters (SI; PARAMETER_TABLES['ito']): n_bg_m3 1e20-1.2e21 cm^-3; eps_inf 4.0 (films 3.9-4.4); Kane band
    0.21 m_e / 0.52 eV^-1 ([Mar24], sputtered; pass m0_ratio=0.18, alpha_per_eV=0.5 for crystalline / epitaxial
    material [Fen16]); optical_mobility_m2Vs 30e-4 (films 16e-4-40e-4; gamma ~1.7e14 rad/s at 3e20 cm^-3);
    dc_mobility_m2Vs None (= the optical mobility) or a float / callable; grain_boundary None or a
    SetoGrainBarrier; eps_static 9.0 (films; 10.55 bulk crystal [Sto21]); band_gap_eV 2.8 (fundamental; the
    optical onset of degenerate films is ~3.7 eV); chi_eV None (spread ~1 eV; set yours); background None or a
    DrudeOptical background model. Reference points (tests): optical mass 0.33-0.44 m_e over 2.9e20-1.01e21
    cm^-3 [Mar24]; ~0.4 m_e at 1e21 cm^-3 [Fen16]."""
    n = _check_density(n_bg_m3, "ito", strict)
    return _build(key="ito", name=name, n_bg=n, eps_inf=eps_inf, m0_ratio=m0_ratio, alpha=alpha_per_eV,
                  mu_opt=optical_mobility_m2Vs, mu_dc_intragrain=dc_mobility_m2Vs, grain_boundary=grain_boundary,
                  eps_static=eps_static, band_gap_eV=band_gap_eV, chi_eV=chi_eV, background=background,
                  eps_inf_extraordinary=None)


def fto(n_bg_m3: float, *, name: str = "FTO", eps_inf: float = _FTO_EPS_INF, m0_ratio: float = _FTO_M0,
        alpha_per_eV: float = _FTO_ALPHA, optical_mobility_m2Vs=_FTO_MU_OPT, dc_mobility_m2Vs="masetti",
        grain_boundary=None, eps_static: float = _FTO_EPS_STATIC, band_gap_eV: Optional[float] = _FTO_GAP,
        chi_eV: Optional[float] = None, background=None, strict: bool = True) -> Material:
    """F-doped SnO2 (FTO) film preset, polycrystalline films (spray pyrolysis / CVD).

    Band: the film fit of [Rey13] (0.09 m_e, 0.8 eV^-1), which reproduces its polycrystalline films' optical
    masses (0.16-0.29 m_e over 0.9-4.5e20 cm^-3). THE TWO PUBLISHED LAWS DISAGREE BY ~30 %: epitaxial SnO2:Sb
    [Fen19] gives the anisotropic 0.205 / 0.73 (E perp c) and 0.183 / 0.33 (E par c), orientation average
    0.197 / 0.56 (0.36 m_e at 4.3e20 against the films' 0.24-0.28); pass m0_ratio=0.197, alpha_per_eV=0.56
    for epitaxial material. eps_inf 3.9; optical mobility 35 cm^2/(V s) ([Rey13]: optical ~ Hall at high
    density, optical above Hall below 1e20 from grain and twin boundaries -- add a SetoGrainBarrier for the
    DC mobility there); DC intra-grain mobility: the SnO2 Masetti fit of [EM08]. eps_static 12.3 (orientation
    average; 14 perp / 9 par), band gap 3.56 eV, chi None. Density range 1.5e19-5e20 cm^-3.
    Reference points (tests): [Rey13] screened plasma wavelengths 1639 / 2203 / 2988 nm at 4.3 / 2.2 / 0.9e20
    cm^-3 (D from their Table I); [Fen19] epitaxial masses with the alternative laws.
    Sources: [Rey13] G. Rey et al., J. Appl. Phys. 114, 183713 (2013), 10.1063/1.4829672 (P); [Fen19] M.
    Feneberg et al., APL Mater. 7, 022508 (2019), 10.1063/1.5054351 (P); [Sch11] A. Schleife et al., Phys. Rev.
    B 83, 035116 (2011) (S for the experimental values it quotes); [EM08] Ellmer and Mientus (2008) (P)."""
    n = _check_density(n_bg_m3, "fto", strict)
    dc = MasettiMobility.sno2_ellmer_mientus_2008() if dc_mobility_m2Vs == "masetti" else dc_mobility_m2Vs
    return _build(key="fto", name=name, n_bg=n, eps_inf=eps_inf, m0_ratio=m0_ratio, alpha=alpha_per_eV,
                  mu_opt=optical_mobility_m2Vs, mu_dc_intragrain=dc, grain_boundary=grain_boundary,
                  eps_static=eps_static, band_gap_eV=band_gap_eV, chi_eV=chi_eV, background=background,
                  eps_inf_extraordinary=None)


# =====================================================================================================
# helpers
# =====================================================================================================

def zno_seto_barrier(*, trap_density_cm2: float = 3.0e13, grain_size_m: float = 30e-9,
                     eps_static: float = _ZNO_EPS_STATIC, m0_ratio: float = _ZNO_M0,
                     alpha_per_eV: float = _ZNO_ALPHA, T_K: float = T_REF) -> SetoGrainBarrier:
    """A SetoGrainBarrier for a polycrystalline doped-ZnO film: trap density 3e13 cm^-2 (polycrystalline
    reactive-sputtered ZnO:Al, [EM08] Tables 3-4; 0.5-1.5e13 epitaxial, 1.6-1.75e13 [Cor09], 1.8-4.2e13 ALD
    [Rom16]), grain size 30 nm (sputtered 19-48 nm [Cor09]; ALD ~30 nm [Rom16]), the ZnO Kane band's optical
    mass at the local density. Pass it as azo(..., grain_boundary=zno_seto_barrier(...))."""
    band = KaneBand(m0_kg=float(m0_ratio) * M_E, alpha_per_eV=float(alpha_per_eV))
    return SetoGrainBarrier(trap_density_m2=float(trap_density_cm2) * 1.0e4, grain_size_m=float(grain_size_m),
                            eps_r=float(eps_static), mass_kg=band.optical_mass(), T_K=float(T_K))


def dual_gate_film_kwargs(material: Material) -> dict:
    """Keyword arguments of carriers.DualGateSchrodingerPoisson for a preset film: n_bg_m3, m_eff_kg (the
    band-edge mass; the solver's z dispersion is parabolic in it), eps_static and alpha_np_per_eV (its
    in-plane Kane nonparabolicity), all from the Material's KaneBand -- the same band as the optics. Add the
    geometry (t_film_m, bottom, top, nz, ...) yourself."""
    tr = getattr(material, "transport", None)
    band = getattr(tr, "band", None)
    if not isinstance(band, KaneBand):
        raise ValueError("dual_gate_film_kwargs: material {!r} has no KaneBand TransportModel (build it with a "
                         "preset factory or TransportModel(band=KaneBand(...)))".format(getattr(material, "name", "?")))
    return {"n_bg_m3": float(tr.n_bg_m3), "m_eff_kg": float(band.m0_kg), "eps_static": float(tr.eps_static),
            "alpha_np_per_eV": float(band.alpha_per_eV)}


def wurtzite_polarization_charges(*, metal_polar_fraction: float, P_sp_C_m2: float = ZNO_P_SP_C_M2,
                                  P_pz_C_m2: float = 0.0, compensated_fraction: float = 0.0,
                                  P_neighbours_C_m2: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """Bound polarization sheet charges (sigma_bottom, sigma_top) [C/m^2] at the two faces of a c-axis-
    textured wurtzite film (z = growth direction, bottom face at the substrate side), for
    GateStack(interface_charge_C_m2=...) of carriers.sp_dual_gate or LayeredDevsimBuilder(fixed_charge=...).

    P = P_sp + P_pz along [0001] (P_sp < 0 for ZnO: -0.057 C/m^2 [BFV97]; -0.048 [GS05]); a metal-polar
    (Zn-polar) grain has [0001] along +z. The bound charge at a face with outward normal n is
    sigma = (P_film - P_neighbour) . n, so a metal-polar film carries sigma_top = P - P_top,neighbour and
    sigma_bottom = -(P - P_bottom,neighbour): for ZnO on amorphous oxides (P_neighbour = 0) that is -0.057 C/m^2
    on top and +0.057 C/m^2 at the bottom -- electrons accumulate at the bottom face and deplete the top.

      metal_polar_fraction : fraction of grains with [0001] along +z (1 = metal-polar, 0 = O-polar, 0.5 =
                             mixed polarity: the bound charges cancel on average). Sputtered films on glass
                             can start O-polar and invert with thickness (Nulhakim and Makino 2016, abstract).
      P_pz_C_m2            : piezoelectric part (2 eps_xx (e31 - e33 c13 / c33) for biaxial strain; the
                             elastic constants are not shipped -- pass your value). |2 eps e31| ~ 5e-3 C/m^2 at
                             0.5 % strain, i.e. < 10 % of P_sp.
      compensated_fraction : fraction screened by fixed interface charge of the opposite sign (adsorbates,
                             defects). No measurement of this compensation for polycrystalline sputtered or ALD
                             films was found; 0 (default) gives the bare, upper-bound value.
      P_neighbours_C_m2    : (P below, P above) along [0001] of polar neighbours (e.g. a (Mg,Zn)O barrier).
    Magnitude: |P_sp| = 0.057 C/m^2 = 3.6e13 e/cm^2 is COMPARABLE to the gate-induced sheet charge (8 nm of
    eps 25 oxide at 1.5 V: 0.041 C/m^2), so an uncompensated polarization changes a gated accumulation layer at
    first order; the charges at the two faces are equal and opposite (a dipole), so the film's total
    electron count is unchanged."""
    f = float(metal_polar_fraction)
    c = float(compensated_fraction)
    if not (0.0 <= f <= 1.0):
        raise ValueError("wurtzite_polarization_charges: metal_polar_fraction must lie in [0, 1], got {!r}".format(
            metal_polar_fraction))
    if not (0.0 <= c <= 1.0):
        raise ValueError("wurtzite_polarization_charges: compensated_fraction must lie in [0, 1], got {!r}".format(
            compensated_fraction))
    P = float(P_sp_C_m2) + float(P_pz_C_m2)
    pb, pt = (float(v) for v in P_neighbours_C_m2)
    for v, nm in ((P, "P_sp + P_pz"), (pb, "P_neighbours[0]"), (pt, "P_neighbours[1]")):
        if not math.isfinite(v):
            raise ValueError("wurtzite_polarization_charges: {} must be finite".format(nm))
    s = (2.0 * f - 1.0) * (1.0 - c)           # net projection of [0001] on +z, times the unscreened fraction
    sigma_top = s * (P - pt)
    sigma_bottom = -s * (P - pb)
    return float(sigma_bottom), float(sigma_top)
