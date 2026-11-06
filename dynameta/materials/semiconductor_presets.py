"""
Cited, internally consistent silicon parameter BUNDLES for the bipolar drift-diffusion path (v0.15,
consolidated plan item 4.2.1; user decisions 26-29). Devsim-free.

WHY BUNDLES. A band-gap-narrowing (BGN) fit is extracted together with ONE intrinsic density n_i, ONE
pair of band densities of states N_c, N_v and ONE carrier statistics. Mixing sets moves the effective
intrinsic density n_ie^2 by ~2x without any error (old Slotboom or del Alamo with n_i = 9.65e9 cm^-3:
-2.3x, ~21 meV of gap). A bundle (SiliconPairing) carries all of them, and the paired BGN model refuses a
region that does not match (carriers.bipolar_options.check_bgn_pairing; BGNPairingError).

THE HOUSE PAIRING (decision 26): SI_PAIRING_YC2013_FD -- Yan & Cuevas, J. Appl. Phys. 114, 044508
(2013): a REAL narrowing fitted with Fermi-Dirac statistics,
    dEg = 0.042 meV [ln(N_D / 1e14 cm^-3)]^3                           (Eq. (3), Table I)
    n_i = 9.65e9 cm^-3 (Sec. III.B, Ref. 32), N_c = 2.86e19, N_v = 3.1e19 cm^-3 (Sec. I, Ref. 3), 300 K
    p0 n0 = n_i^2 (n0/N_c) exp(-F_1/2^-1(n0/N_c)) exp(dEg/kT)          (Eq. (1))
It needs carrier_statistics="fermi_dirac" (FD transport AND FD detailed balance), which the factory
below sets. SI_PAIRING_YC2013_BOLTZMANN is the same paper's APPARENT narrowing for Boltzmann statistics
(12.96 meV ln(N_D/1e17 cm^-3), carrier_statistics="boltzmann"); the two agree on n0 p0 within ~11 % over
1e18-5e19 cm^-3 (the paper's Fig. 3), and FD gives less above ~4e19.

LIMITS (stated, not hidden):
  * n-TYPE ONLY. The p-type constants are in Yan & Cuevas, J. Appl. Phys. 116, 194505 (2014), which has
    not been read; a p-type node with N_A > 1e14 cm^-3 under the house BGN is REFUSED
    (BGNFitPendingError), and silicon_transport(acceptor=True) with BGN on refuses at construction.
  * The fits used Klaassen's mobility model, which DynaMeta does not have: the mobilities are REQUIRED
    arguments with no default (a statistics/BGN bundle must not pretend to calibrate transport).
  * band_gap_eV is the STATISTICAL gap kT ln(N_c N_v / n_i^2) = 1.12973 eV, not the optical 1.124 eV:
    only the statistical gap is consistent with the bulk model, and the 2-D builder's heterojunction
    check (5 meV) refuses the 5.7 meV mismatch of the optical value.
  * 300 K only (decision 29): a paired set at another T refuses unless BipolarOptions(
    allow_unpaired_bgn=True) is set together with your own n_i(T), N_c(T), N_v(T).
  * chi_eV = 4.05 eV (the commonly tabulated Si affinity, e.g. Sze & Ng, Physics of Semiconductor
    Devices, 3rd ed. (2007) [secondary; not re-read in the primary]), quoted at 300 K (chi_reference_T_K).
  * The rational FD fit's Delta errs by up to +3.9 % in the moderately degenerate range (physics_
    bipolar_dd "ACCURACY OF Delta"): n0 p0 is 0.42 % low at 1e19 and 3.5 % low at 1e20 cm^-3 against
    Eq. (1) with exact F_1/2. Inside the paper's +/-0.003 meV (7 %) uncertainty on A, but not exact.

III-V RECOMBINATION PRESETS (v0.15, consolidated plan item 4.2.2, user decision 30; carriers.bipolar_options
RadiativeModel / AugerModel). A radiative coefficient is either INTERNAL (the material's recycling-free
B_int) or EFFECTIVE (the net loss of one structure, after photon recycling): B_eff = B_int (1 - P_r). For
GaAs they differ ~4x (a 2 um double heterostructure: 3.3e-10 vs 0.78e-10 cm^3/s), so every preset here is
LABELLED kind="internal" and its NAME ends in _INTERNAL. Give the layer's re-absorption probability with
`preset.with_photon_recycling(P_r)` (applied ONCE, at the B_rad write); an internal B with no factor builds
with one AdvisoryWarning per region (recycling ignored -> radiative loss overestimated).

  RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL  B_int = 3.3e-10 cm^3/s  [VERIFIED IN THE PRIMARY]
      Niemeyer et al., "Measurement of the non-radiative
      minority recombination lifetime and the effective radiative recombination coefficient in GaAs", AIP
      Adv. 9, 045034 (2019): abstract + Sec. VI ("(3.3 +/- 0.6)e-10 cm^3/s" for p-type doping below 1e18
      cm^-3; "(2.1 +/- 0.4)e-10" at 1e18, the degenerate roll-off, NOT modelled), Eq. (1) 1/tau_rad,eff =
      (1 - f) B_rad / (N + dn) = B_eff / (N + dn) with the recycling factor f = 0.76 +/- 0.04 and the
      directly measured B_eff = (0.78 +/- 0.07)e-10 cm^3/s, both for its 2 um Al0.7Ga0.3As/GaAs/Al0.7Ga0.3As
      DH (Sec. II.A). Read 2026-11-06 on the publisher page (the abstract record also lists 3.3 and 0.76).
      P_r = 0.76 belongs to THAT structure; it is not shipped as a default.
  AUGER_GAAS_STRAUSS_1993  C_n = C_p = 3.5e-30 cm^6/s  [SECONDARY: abstract record only; split ASSUMED]
      Strauss, Ruhle & Kohler, "Auger recombination in intrinsic GaAs", Appl. Phys. Lett. 62, 55 (1993):
      the abstract (via the scilit / OUCI index records) gives an EFFECTIVE Auger coefficient
      C_eff = (7 +/- 4)e-30 cm^6/s from time-resolved PL on INTRINSIC GaAs at high injection (n = p,
      significant above ~2.5e19 cm^-3) and B = (1.7 +/- 0.2)e-10 cm^3/s (not shipped: structure-dependent).
      The full text was NOT read. Reading C_eff as the ambipolar sum (U = C_eff n^3 at n = p, i.e.
      C_n + C_p = C_eff), it is split EQUALLY by assumption (decision 30). A secondary user, Sachenko,
      Kostylyov, Sokolovskyi & Shkrebtii ("Modeling key characteristics of high-efficiency gallium arsenide
      solar cells", arXiv preprint 2026, Eq. (18) and its Ref. [28] = Strauss 1993), instead puts
      C_e = C_h = 7e-30 EACH -- 2x our sum. The +/- 4 (57 %) covers neither reading decisively.
  RADIATIVE_INP_SEMYONOV_2010_INTERNAL  B_int = 1.2e-10 cm^3/s  [VERIFIED IN THE PRIMARY (arXiv text)]
      Semyonov, Subashiev, Chen & Luryi, "Radiation efficiency of heavily doped bulk n-InP semiconductor",
      arXiv:1003.6095 (2010), Eq. (6): B_R = 1.2e-10 (1 - 0.092 n/n0) cm^3/s, n0 = 1e18 cm^-3, room T. B_R
      is the recycling-REMOVED constant (B_R = B~ Phi / n, the measured rate times the effective recycling
      factor Phi; text before Eq. (6)), so INTERNAL. Fitted on n-InP wafers with n = 2-8e18 cm^-3: 1.2e-10 is
      the n -> 0 intercept (0.98e-10 at 2e18, 0.32e-10 at 8e18); the roll-off is documented, NOT modelled.
      InP Auger is NOT shipped (the paper's Auger constant and the tabulated values are inconsistent).
  DEFERRED (not shipped; decision 30): In0.53Ga0.47As (Ahrenkiel 1998 numbers seen by OCR only, recycling not
  removed), GaN (B spread over three decades; the measured Auger values are InGaN), Ge (unsourced tabulated
  values), and Varshni's 1967 GaAs B = 7.2e-10 (a calculation, ~2x the measurements; never a measured B).
  Band parameters (gap, masses, n_i) for these materials are NOT bundled: the presets are recombination
  coefficients only, to be combined with your own TransportModel (refs/vurgaftman2001 is the band source
  to use when III-V bundles are added).

Nothing here is applied automatically: TransportModel defaults are unchanged, MaterialRegistry never
registers presets, and a value change gets a NEW preset name (the old name keeps its values).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np

from dynameta.constants import HBAR, KB, Q_E

_CM3 = 1.0e6          # cm^-3 -> m^-3


@dataclass(frozen=True)
class SiliconPairing:
    """One self-consistent set: n_i, N_c, N_v [m^-3] at T_K, the carrier statistics the BGN fit was made
    with, and the BGN model(s). bgn_n is evaluated in N_D (n-type); bgn_p None means the p-type fit is
    PENDING (the region BGN refuses p-type nodes). mobility_note / sources are documentation."""
    name: str
    n_i_m3: float
    N_c_m3: float
    N_v_m3: float
    statistics: str
    T_K: float
    bgn_n: object                      # carriers.bipolar_options.BGNModel (typed loosely: devsim-free)
    bgn_p: Optional[object] = None
    mobility_note: str = ""
    sources: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.statistics not in ("fermi_dirac", "boltzmann"):
            raise ValueError("SiliconPairing.statistics must be 'fermi_dirac' or 'boltzmann'")
        for b in (self.bgn_n, self.bgn_p):
            if b is None:
                continue
            if not b.paired or float(b.paired_n_i_m3) != float(self.n_i_m3) \
                    or b.paired_statistics != self.statistics or float(b.paired_T_K) != float(self.T_K):
                raise ValueError("SiliconPairing {!r}: BGN set [{}] is not paired with this bundle "
                                 "(n_i, statistics, T must be the bundle's)".format(self.name, b.source))
            if self.statistics == "fermi_dirac" and (float(b.paired_N_c_m3) != float(self.N_c_m3)
                                                     or float(b.paired_N_v_m3) != float(self.N_v_m3)):
                raise ValueError("SiliconPairing {!r}: FD BGN set [{}] was fitted with other N_c/N_v"
                                 .format(self.name, b.source))

    @property
    def band_gap_eV(self) -> float:
        """The STATISTICAL gap kT ln(N_c N_v / n_i^2) [eV] at T_K -- the only gap consistent with the bulk
        model (the 2-D heterojunction check compares band_gap_eV with exactly this)."""
        return KB * float(self.T_K) / Q_E * math.log(self.N_c_m3 * self.N_v_m3 / self.n_i_m3 ** 2)

    def dos_mass_kg(self, N_m3: float) -> float:
        """The DOS effective mass [kg] that reproduces a band density N [m^-3] at T_K through
        physics_equilibrium.effective_dos_m3 (N = 2 (m kT / (2 pi hbar^2))^(3/2)), inverted exactly."""
        return (0.5 * float(N_m3)) ** (2.0 / 3.0) * 2.0 * math.pi * HBAR ** 2 / (KB * float(self.T_K))


def _yc2013_pairing(statistics: str) -> SiliconPairing:
    from dynameta.carriers.bipolar_options import (BGN_SI_YAN_CUEVAS_2013_BOLTZMANN,
                                                   BGN_SI_YAN_CUEVAS_2013_FD)
    fd = statistics == "fermi_dirac"
    return SiliconPairing(
        name="SI_PAIRING_YC2013_FD" if fd else "SI_PAIRING_YC2013_BOLTZMANN",
        n_i_m3=9.65e9 * _CM3, N_c_m3=2.86e19 * _CM3, N_v_m3=3.1e19 * _CM3, statistics=statistics,
        T_K=300.0, bgn_n=BGN_SI_YAN_CUEVAS_2013_FD if fd else BGN_SI_YAN_CUEVAS_2013_BOLTZMANN,
        bgn_p=None,
        mobility_note="fitted with Klaassen's unified mobility (Solid-State Electron. 35, 953 (1992)), "
                      "not implemented in DynaMeta: supply mobilities explicitly",
        sources=("Yan & Cuevas, J. Appl. Phys. 114, 044508 (2013): Eq. (1), Eq. (3), Table I",
                 "n_i: Misiakos & Tsamakis, J. Appl. Phys. 74, 3293 (1993) (the paper's Ref. 32)",
                 "N_c, N_v: Green, J. Appl. Phys. 67, 2944 (1990) (the paper's Ref. 3)",
                 "p-type: Yan & Cuevas, J. Appl. Phys. 116, 194505 (2014) -- PENDING (not read)"))


SI_PAIRING_YC2013_FD = _yc2013_pairing("fermi_dirac")
SI_PAIRING_YC2013_BOLTZMANN = _yc2013_pairing("boltzmann")
SI_PAIRINGS = (SI_PAIRING_YC2013_FD, SI_PAIRING_YC2013_BOLTZMANN)

SI_CHI_EV = 4.05            # Si electron affinity at 300 K, e.g. Sze & Ng (2007) [not re-read]
SI_EPS_STATIC = 11.7        # Si static permittivity, e.g. Sze & Ng (2007) [not re-read]


_AUGER = {"dziewior_schmid1977": "AUGER_SI_DZIEWIOR_SCHMID_1977", "altermatt1997": "AUGER_SI_ALTERMATT_1997"}


# =====================================================================================================
# III-V recombination presets (v0.15, plan 4.2.2, decision 30) -- see the module doc for the sources and
# which constants were verified in the primary text. Constants typed in the units the papers quote.
# =====================================================================================================

def _iii_v_presets():
    from dynameta.carriers.bipolar_options import AugerModel, RadiativeModel
    cm3s, cm6s = 1.0e-6, 1.0e-12               # cm^3/s -> m^3/s, cm^6/s -> m^6/s
    gaas_b = RadiativeModel(
        B_m3s=3.3e-10 * cm3s, kind="internal",
        source="INTERNAL (recycling-free) B_int, GaAs 300 K, p-type < 1e18 cm^-3: Niemeyer et al., AIP Adv. 9, "
               "045034 (2019); NOT the effective B of a device layer (2 um DH: B_eff 0.78e-10 = 4.2x smaller)",
        locator="abstract + Sec. VI: B_rad (3.3 +/- 0.6)e-10 cm^3/s below 1e18 cm^-3 ((2.1 +/- 0.4)e-10 at 1e18, "
                "not modelled); Eq. (1): B_eff = (1 - f) B_rad, f = 0.76 +/- 0.04 and B_eff (0.78 +/- 0.07)e-10 "
                "cm^3/s for its 2 um Al0.7Ga0.3As/GaAs DH",
        verification="primary")
    gaas_c = AugerModel(
        C_n_m6s=3.5e-30 * cm6s, C_p_m6s=3.5e-30 * cm6s,
        source="GaAs 300 K: Strauss, Ruhle & Kohler, Appl. Phys. Lett. 62, 55 (1993); ambipolar C_eff = "
               "(7 +/- 4)e-30 cm^6/s (high injection, intrinsic GaAs) split EQUALLY by assumption",
        locator="abstract only (index records): C_eff = (7 +/- 4)e-30 cm^6/s; full text not read; the "
                "sum reading C_n + C_p = C_eff is assumed (Sachenko et al. 2026 use 7e-30 for EACH)",
        verification="secondary")
    inp_b = RadiativeModel(
        B_m3s=1.2e-10 * cm3s, kind="internal",
        source="INTERNAL (recycling-removed) B_R, n-InP 300 K: Semyonov, Subashiev, Chen & Luryi, "
               "arXiv:1003.6095 (2010); n -> 0 intercept of a fit over n = 2-8e18 cm^-3",
        locator="Eq. (6): B_R = 1.2e-10 (1 - 0.092 n/n0) cm^3/s, n0 = 1e18 cm^-3 (roll-off not modelled); "
                "B_R = B~ Phi / n with Phi the effective recycling factor (text before Eq. (6), Table 1)",
        verification="primary")
    return gaas_b, gaas_c, inp_b


RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL, AUGER_GAAS_STRAUSS_1993, RADIATIVE_INP_SEMYONOV_2010_INTERNAL = \
    _iii_v_presets()
III_V_RECOMBINATION_PRESETS = (RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL, AUGER_GAAS_STRAUSS_1993,
                               RADIATIVE_INP_SEMYONOV_2010_INTERNAL)


def silicon_bipolar_options(pairing: SiliconPairing = SI_PAIRING_YC2013_FD, *, auger: Optional[str] = None,
                            radiative: bool = False, bgn: bool = True, ionization=None,
                            conduction_fraction: float = 0.5, T_K: Optional[float] = None,
                            allow_unpaired_bgn: bool = False):
    """A carriers.bipolar_options.BipolarOptions for one bundle: the pairing's n-type BGN (bgn=True;
    refuses p-type nodes while the p-type fit is pending), optional Auger ("dziewior_schmid1977" |
    "altermatt1997" | None) and radiative (Trupke 2003) recombination, an optional IonizationModel, and
    the BGN band-edge split gamma (conduction_fraction; 0.5 = symmetric, invisible on uniform regions).
    T_K != pairing.T_K is refused by the pairing check at setup unless allow_unpaired_bgn=True (then pass
    the region's own n_i(T), N_c(T), N_v(T)). Use with statistics = pairing.statistics."""
    import dataclasses
    from dynameta.carriers import bipolar_options as BO
    b = None
    if bgn:
        b = pairing.bgn_n
        if float(conduction_fraction) != float(b.conduction_fraction):
            b = dataclasses.replace(b, conduction_fraction=float(conduction_fraction))
    if auger is not None and auger not in _AUGER:
        raise ValueError("auger must be None or one of {}, got {!r}".format(sorted(_AUGER), auger))
    return BO.BipolarOptions(
        auger=getattr(BO, _AUGER[auger]) if auger is not None else None,
        radiative=BO.RADIATIVE_SI_TRUPKE_2003 if radiative else None,
        bgn=b, ionization=ionization, T_K=T_K, allow_unpaired_bgn=bool(allow_unpaired_bgn))


def _const(v: Union[float, Callable]) -> Callable:
    if callable(v):
        return v
    val = float(v)
    if not (math.isfinite(val) and val > 0.0):
        raise ValueError("a mobility must be finite and > 0 (m^2/(V s)), got {!r}".format(v))
    return lambda n: np.full_like(np.asarray(n, dtype=float), val)


def silicon_transport(*, n_bg_m3: float, mobility_n_m2Vs, mobility_p_m2Vs, tau_srh_s: float,
                      pairing: SiliconPairing = SI_PAIRING_YC2013_FD, acceptor: bool = False,
                      net_doping_expr: Optional[str] = None, options=None,
                      eps_static: float = SI_EPS_STATIC, chi_eV: float = SI_CHI_EV, **extra):
    """A bipolar_dd TransportModel for silicon from one bundle: n_i_m3 = pairing.n_i_m3; DOS masses that
    reproduce the pairing's N_c, N_v at T_K (dos_mass_kg); band_gap_eV = the STATISTICAL gap (1.12973 eV
    for the house pairing); carrier_statistics = pairing.statistics; chi_eV = 4.05 eV quoted at 300 K;
    bipolar_options = `options` or silicon_bipolar_options(pairing) (the paired BGN on).

    mobility_n_m2Vs / mobility_p_m2Vs (REQUIRED, no default): floats [m^2/(V s)] or callables of n, as
    TransportModel takes them -- the pairing's fit used Klaassen mobility, which is not available, so no
    mobility is implied. tau_srh_s: the SRH lifetime. Doping: n_bg_m3 with acceptor (uniform) or
    net_doping_expr (an in-region junction). With the house pairing a p-type region (acceptor=True) and
    the BGN on is refused here (BGNFitPendingError): the p-type fit is pending. `extra` passes any other
    TransportModel field (e.g. srh_traps)."""
    from dynameta.carriers.bipolar_options import BGNFitPendingError
    from dynameta.materials.transport_model import TransportModel
    opts = options if options is not None else silicon_bipolar_options(pairing)
    b = getattr(opts, "bgn", None)
    if acceptor and b is not None and b.outside_basis == "refuse" and b.doping_basis == "donors" \
            and float(n_bg_m3) > float(b.N_ref_m3):
        raise BGNFitPendingError(
            "silicon_transport: a p-type region (acceptor=True, N_A = {:.3g} cm^-3) with the {} bundle's "
            "BGN on: the p-type constants (Yan & Cuevas, J. Appl. Phys. 116, 194505 (2014)) have not been "
            "read. Pass options=silicon_bipolar_options(pairing, bgn=False) to run this region without "
            "narrowing knowingly.".format(float(n_bg_m3) / _CM3, pairing.name))
    m_n = pairing.dos_mass_kg(pairing.N_c_m3)
    m_p = pairing.dos_mass_kg(pairing.N_v_m3)
    return TransportModel(
        n_bg_m3=float(n_bg_m3), eps_static=float(eps_static),
        dos_mass_kg_of_n_m3=lambda n: np.full_like(np.asarray(n, dtype=float), m_n),
        band_gap_eV=pairing.band_gap_eV, chi_eV=float(chi_eV), physics="bipolar_dd",
        mobility_m2Vs_of_n_m3=_const(mobility_n_m2Vs), hole_mobility_m2Vs_of_n_m3=_const(mobility_p_m2Vs),
        tau_srh_s=float(tau_srh_s), n_i_m3=pairing.n_i_m3, acceptor=bool(acceptor),
        net_doping_expr=net_doping_expr, dos_mass_p_kg=m_p, bipolar_options=opts,
        chi_reference_T_K=float(pairing.T_K), carrier_statistics=pairing.statistics, **extra)


__all__ = ["SiliconPairing", "SI_PAIRING_YC2013_FD", "SI_PAIRING_YC2013_BOLTZMANN", "SI_PAIRINGS",
           "SI_CHI_EV", "SI_EPS_STATIC", "silicon_bipolar_options", "silicon_transport",
           "RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL", "AUGER_GAAS_STRAUSS_1993",
           "RADIATIVE_INP_SEMYONOV_2010_INTERNAL", "III_V_RECOMBINATION_PRESETS"]
