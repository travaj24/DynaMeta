"""Er:Yb co-doped fiber amplifier (EYDFA): the z-resolved coupled-power solve for a
phosphosilicate erbium/ytterbium sensitized amplifier, the sensitized counterpart of the
single-ion FiberAmplifier (steady_state.py). Ytterbium is the pump-absorbing SENSITIZER: it
has a ~30x larger 900-980 nm absorption cross-section than Er, soaks up the (typically
cladding-guided) 915/940/976 nm pump, and hands the excitation to Er by a near-resonant
dipole-dipole ENERGY TRANSFER Yb(2F5/2) + Er(4I15/2) -> Yb(2F7/2) + Er(4I11/2). The Er 4I11/2
pump band then relaxes fast (A_32 ~ 3e5-1e6 1/s in a high-phonon phosphosilicate host) to the
4I13/2 metastable level, from which the 1530-1565 nm C-band signal is amplified exactly as in a
plain EDFA. The point of the co-dope is CLADDING-PUMPABILITY at high power: Yb makes the weak,
narrow Er 980 nm line irrelevant, so a multimode diode pump can be absorbed in a few metres.

LEVELS. Yb: ground n_Yb1 (2F7/2), excited n_Yb2 (2F5/2), N_Yb = n_Yb1 + n_Yb2. Er: n1 (4I15/2
ground), n2 (4I13/2 metastable), n3 (4I11/2 pump band), N_Er = n1 + n2 + n3. fiber.n_t_m3 is
N_Er; n_yb_m3 is N_Yb.

FAST-4I11/2 LIMIT (standard for phosphosilicate; A_32 >> k_back n_Yb1). n3 is adiabatically
eliminated: A_32 n3 -> (transfer + direct Er pump) feeding n2, back-transfer -> 0, and N_Er ~
n1 + n2. The Er block collapses to the two-level EDFA algebra with an EXTRA optical-pump-like
term = the Yb->Er transfer, and the direct Er absorption (which in the two-level treatment also
promotes n1 -> n2, whether it lands on the 4I11/2 980 nm line or the 4I13/2 signal band) is the
usual R_a_Er. With f2 = n2/N_Er (Er metastable fraction) and b2 = n_Yb2/N_Yb (Yb inversion) the
z-local steady state is the coupled pair (per ion, 1/s):

    Er:  R_a_Er (1 - f2) - R_e_Er f2 - f2/tau_Er + k_tr b2 N_Yb (1 - f2) - C_up N_Er f2^2 = 0
    Yb:  R_a_Yb (1 - b2) - R_e_Yb b2 - b2/tau_Yb - k_tr b2 N_Er (1 - f2)               = 0

    R_{a/e}_ion = SUM_k Gamma_k sigma_{a/e}_ion,k P_k / (h nu_k A_dope)     (per-ion rates)

DERIVATION OF THE TRANSFER TERMS (resolves the dossier's flagged eta_tr N_Yb-vs-N_Er
ambiguity). The volumetric forward transfer rate is R_tr = k_tr n_Yb2 n1 (an excited Yb donor
meeting a GROUND Er acceptor). Per Yb ion (divide by N_Yb) this is k_tr n1 = k_tr N_Er (1 - f2):
a DRAIN on the Yb inversion set by the Er GROUND density, NOT by N_Yb -- so the transfer-out
term in the Yb equation is k_tr b2 N_Er (1 - f2). Per Er ground ion (divide the same R_tr by
N_Er) it is k_tr n_Yb2 = k_tr b2 N_Yb, an extra pump acting on the Er ground fraction (1 - f2) --
the transfer-in term in the Er equation. The physical weak-signal transfer efficiency is
therefore eta_tr = k_tr N_Er tau_Yb / (1 + k_tr N_Er tau_Yb) (N_Er, the acceptor density), which
is what transfer_efficiency() reduces to at low power -- see the DISCREPANCY NOTE below.

k_back optional refinement. Retaining back-transfer k_back n3 n_Yb1 in the (eliminated) n3
balance multiplies BOTH transfer terms by phi = A_32 / (A_32 + k_back (1 - b2) N_Yb) (the
fraction of 4I11/2 population that relaxes to 4I13/2 before back-transferring). For literature
k_back ~ 1e-24 and A_32 ~ 5e5 this is phi ~ 1 - 2e-4 (negligible), consistent with the
fast-limit assumption; it is included exactly (phi = 1 when k_back = 0, the default).

PROPAGATION. Every channel sees BOTH ions at its wavelength (a 976 pump interacts almost only
with Yb, a 1550 signal almost only with Er since sigma_Yb(1550) = 0 numerically):

    dP_k/dz = u_k { Gamma_k [ N_Er (sigma_e_Er,k f2 - sigma_a_Er,k (1 - f2))
                            + N_Yb (sigma_e_Yb,k b2 - sigma_a_Yb,k (1 - b2)) ] - l_k } P_k
              + [ASE] u_k Gamma_k m h nu_k dnu_k [ N_Er sigma_e_Er,k f2 + N_Yb sigma_e_Yb,k b2 ]

The spontaneous source is cross-section-weighted from both ions, so a C-band ASE bin is seeded
only by Er and a 1000-1100 nm (yb_ase) bin only by Yb, with no hand-assigned band boundary.

SOLVE. The two-point boundary-value problem is closed by the SAME relaxation scheme as
FiberAmplifier.solve (alternate forward 0->L / backward L->0 initial-value passes with the
other direction's z-profile frozen through the lean uniform-mesh interpolator, to convergence on
endpoints + interior). The z-local (f2, b2) are found by reducing the coupled pair to a single
bracketed scalar equation in f2 -- b2(f2) is a closed form (the Yb equation is LINEAR in b2 at
fixed f2) substituted into the Er residual H(f2), which satisfies H(0) >= 0 and H(1) < 0, so a
safeguarded Newton/bisection on [0, 1] is unconditionally robust. This sidesteps the naive
block fixed point, whose positive Yb<->Er feedback has spectral radius ~ k_tr^2 N_Er N_Yb
tau_Er tau_Yb >> 1 (the "stiff when k_tr N_Er >> 1/tau_Yb" regime) and diverges.

DISCREPANCY NOTE (bidirectional adversarial; dossier ">95% transfer" gate). With the measured
phosphosilicate numbers k_tr ~ 1.1e-22-2e-22 m^3/s, tau_Yb = 1.45 ms and N_Er = 2e25 m^-3, the
weak-signal transfer efficiency is k_tr N_Er tau_Yb ~ 4-6, i.e. eta_tr ~ 0.80-0.85, NOT >0.95.
The >95% figure requires either a larger k_tr N_Er product (higher Er loading or the fast-
transfer end of the k_tr range) or that most of the residual is recovered because a Yb photon
lost to spontaneous decay is largely reabsorbed by Yb and eventually transferred; that
reabsorption cascade is NOT in this z-local model. transfer_efficiency() reports the ACTUAL
model value (a rate-integral consistent with the low-power analytic form), and the model does
NOT force the >95% number.

TWO YTTERBIUM POPULATIONS (opt-in, `yb_coupled_fraction` f; DEFAULT 1.0 = the single-pool model
above, bit for bit). Dong, Matniyaz, Kalichevsky-Dong, Nilsson, Jeong, Opt. Express 28(11), 16244
(2020) Eqs. (1)-(12) split the ytterbium into a fraction f COUPLED to erbium (n5c, n6c) and a
fraction 1-f UNCOUPLED (n5nc, n6nc). Only the coupled pool transfers; BOTH pools are pumped,
decay with tau_Yb, and absorb and emit at every channel wavelength, and the propagation sees
their sum n6 = n6c + n6nc. That is the whole point: the uncoupled pool converts pump into Yb
fluorescence and 1-um ASE without ever reaching the erbium, which is what sets the measured
1-um ASE floor and why a single-population model over-predicts output at high pump. With
b2c = n6c/(f N_Yb) and b2nc = n6nc/((1-f) N_Yb) the z-local steady state becomes the TRIPLE

    Er:   R_a_Er (1 - f2) - R_e_Er f2 - f2/tau_Er + k_tr b2c (f N_Yb) (1 - f2)
          - C_up N_Er f2^2                                                          = 0
    Ybc:  R_a_Yb (1 - b2c) - R_e_Yb b2c - b2c/tau_Yb - k_tr b2c N_Er (1 - f2) phi
          - K2 b2c N_Er f2 - W_mig (1 - f) (b2c - b2nc)                             = 0
    Ybnc: R_a_Yb (1 - b2nc) - R_e_Yb b2nc - b2nc/tau_Yb + W_mig f (b2c - b2nc)      = 0

and the field sees the POPULATION-WEIGHTED mean inversion bbar = f b2c + (1 - f) b2nc, which is
what enters the propagation, the Yb parasitic-gain diagnostic and the photodarkening law.

SECONDARY TRANSFER K2 (`k_tr2_m3_s`, default 0). Yb(2F5/2) + Er(4I13/2) -> Yb(2F7/2) +
Er(4F9/2), and the 4F9/2 relaxes multiphonon-fast back to 4I13/2. The ERBIUM POPULATION IS
UNCHANGED by the round trip and the whole Yb quantum becomes heat, so K2 enters as a pure drain
-K2 b2c N_Er f2 on the COUPLED Yb inversion (the same near-neighbour geometry as k_tr) and as a
separate `k2_defect` term in energy_terms. Sefler, Mack, Valley, Rose, JOSA B 21(10), 1740 (2004)
Table 1 measure K2 = 1.5-4e-22 m^3/s on three double-clad Er:Yb fibers -- COMPARABLE to their own
K1 = 2-4e-22 -- and Laroche et al. 2006 report the secondary transfer as more efficient than
upconversion. Canat's 2006 thesis fits K_tr,2 = 1-2e-22 on his two fibers. It is the mechanism
that clamps the output once the erbium is inverted: exactly where the single-population model
over-predicts.

Yb-Yb MIGRATION (`yb_migration_rate_per_s` W_mig, default 0) and the ensemble-versus-pair
reconciliation. The exchange term is DERIVED, not posited. Migration hops an excitation from an
excited Yb to a ground-state Yb, so the volumetric coupled -> uncoupled rate is W' n6c n5nc and
the reverse is W' n6nc n5c. Their difference is

    J = W' [n6c n5nc - n6nc n5c] = W' f (1-f) N_Yb^2 (b2c - b2nc),

because the (1 - b) factors cancel identically -- a bimolecular exchange with detailed balance
collapses to a difference of EXCITATION FRACTIONS. Writing W_mig = W' N_Yb [1/s],

    d n6nc/dt |_mig = +J = -W_mig f (n6nc - n6c (1-f)/f),
    d b2nc/dt |_mig = +W_mig f (b2c - b2nc),   d b2c/dt |_mig = -W_mig (1-f) (b2c - b2nc),

which conserves f b2c + (1-f) b2nc exactly and has b2c = b2nc as its ONLY equilibrium (equal
excitation fractions, never equal densities). This is the physics behind the decay-measurement
versus device-fit split in the k_tr literature: Cheng et al., Materials 15(3), 996 (2022) measure
one bi-exponential Yb decay in Er/Yb/P glass that yields a FAST-PAIR rate 3.07e-21 m^3/s (the
close-coupled neighbour, the right number for the coupled pool of a two-population model) and an
ENSEMBLE-YIELD rate 1.2e-22 m^3/s (the right number for a single-population model) -- a factor of
26 from the same data. With W_mig > 0 the two pools exchange, so a large coupled-pool k_tr and a
small f reproduce an ensemble yield that a single pool would need a small k_tr to match.

CONCENTRATION QUENCHING AND PHOTODARKENING (`concentration=ConcentrationModel(...)`, default
None). Identical semantics to FiberAmplifier: pair-induced quenching in either convention removes
`concentration.dark_density(N_Er)` erbium ions from the active pool and adds their UNBLEACHABLE
absorption Gamma sigma_a_Er n_dark to every channel's background loss; C_up arrives through the
model (the ONE entry point -- passing both `upconversion_C_up` and a model warns and the model
wins, as in the single-ion class); and the Yb photodarkening equilibrium gray loss
pd_loss_per_m * bbar^pd_exponent is evaluated at the POPULATION-WEIGHTED Yb inversion of both
pools. Dark erbium is inert apart from its absorption -- it is not a transfer acceptor here, so a
Yb excitation transferred into a quenched pair is NOT modelled; state that when the pair fraction
is large.

DISTRIBUTED TEMPERATURE (`set_temperature_profile`). Per-z McCumber scaling of BOTH ions'
emission spectra about THEIR OWN zero lines: mcc_ion,k(z) = exp[(eps_ion - h nu_k)(1/kT(z) -
1/kT_ref)], eps_ion = RareEarthIon.eps_J. thermal.solve_with_thermal_feedback accepts this class
and drives Q(z) from `_heat_profile_W_per_m` -- the rate-balance dissipation (quantum defect +
transfer defect + K2 defect + both ions' fluorescence + background loss), not a numerical
gradient of the net flux.

TWO OPT-IN CO-DOPED TEMPERATURE EFFECTS ON TOP OF THAT (2026-09-15; both default OFF and both
exactly the identity at T = T_ref, so they change nothing without a profile). `rate_temperature=
RateTemperatureLaw(...)` gives k_tr, K2 and W_mig an Arrhenius temperature dependence -- Cheng
et al. 2022 measure the transfer rate 1/tau_1 - 1/tau_2 rising 8.5% over 300-480 K, which is
`RATE_ARRHENIUS_CHENG_2022`. `yb_stark_thermal=YbStarkThermal(...)` applies the Boltzmann
depopulation of the Stark level the 976 nm line uses (Canat 2006 / Dussardier 2005: 7% at 300 K,
13% at 400 K) to BOTH Yb cross-sections in a narrow band about the zero line. The second is
ORTHOGONAL to the McCumber factor and does not double count it: at the Yb zero line h nu =
eps_Yb, so mcc_Yb is identically 1 there at every temperature, and mcc never touches sigma_a at
all. Both raise the 1-um parasitic threshold of a hot fiber, which is the qualitative effect
Canat / Dussardier and Morasse 2007 report (Morasse: a core-temperature rise moved his measured
1-um onset from 14 W to 35 W of pump).

CALIBRATION AND FITTING (`eryb_fit.py`). `eryb_calibrate_pools` fits (f, k_tr, W_mig) to measured
Yb fluorescence decays and quenched Yb lifetimes; `eryb_fit_to_device` fits (f, k_tr) jointly to
a device's measured output power and 1-um ASE fraction with W_mig held. The recommended
phosphosilicate migration rate is `eryb_fit.W_MIG_PHOSPHOSILICATE_PER_S`.

References: Karasek, IEEE JQE 33(10):1699 (1997) [Er:Yb rate model, k_tr]; Laroche, Girard,
Sahu, Clarkson, Nilsson, JOSA B 23(2):195 (2006) [k_tr measured in phosphosilicate Er:Yb FIBER,
~6.4e-23 to 1.1e-22 m^3/s]; Hwang, Jiang, Luo, Watson, Sorbello, Peyghambarian, JOSA B 17(5):833
(2000) [k_tr ~ 1.1e-22 and >95% transfer measured in BULK phosphate at N_Er = 2-4e26 m^-3, where
k_tr N_Er tau_Yb = 32-64 -- the '>95%' and the fiber's ~0.80-0.85 (N_Er ~ 4e25) are the same k_tr
at different Er loadings, which resolves the DISCREPANCY NOTE above without a reabsorption
cascade]; Paschotta et al., IEEE JQE 33(7):1049 (1997) [Yb lifetime]; Giles & Desurvire, JLT
9(2):271 (1991) [coupled-power EDFA core]. (Citation audit 2026-09-13: the earlier 'Di Pasquale
& Federighi, JOSA B 23(3):195' attribution was wrong -- that paper is Laroche et al., issue 2.)
Two-population / secondary-transfer / migration references: Dong et al., Opt. Express 28(11),
16244 (2020) [two Yb populations, C63 = 2-3e-20 at f = 15-25%]; Sefler, Mack, Valley, Rose, JOSA B
21(10), 1740 (2004) [K2 secondary transfer 1.5-4e-22, f_NP = 15-17%]; Canat, PhD thesis SUPAERO
(2006) [K_tr,2 1-2e-22, f_np 5-15%]; Cheng et al., Materials 15(3), 996 (2022) [bi-exponential Yb
decay -> 3.07e-21 pair / 1.2e-22 ensemble]; Le Gouet et al., JLT 37(15), 3611 (2019) [Er:Yb Al:P
pair fraction 2k = 1.6%]. Full audit trail:
docs/audit/2026-09-14-eryb-two-population-physics.md. Pure
numpy/scipy; SI units; exp(-i omega t); ASCII-only. docs/fiber_amp_model_spec.md sec.1;
FORMULATION DOSSIER MODULE 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK, KB
from dynameta.core.numerics import trapz          # audit X-1: floor-safe (np.trapezoid needs numpy>=2.0)
from dynameta.optics.fiber_amp.rare_earth import ChannelSet
from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
from dynameta.optics.fiber_amp.waveguide import FiberSpec, cladding_pump_overlap, overlap_gamma
from dynameta.optics.fiber_amp.steady_state import (AseBand, ChannelPlan, Pump, Signal,
                                                    SteadyStateResult, _KEEP, _RELAX_LADDER,
                                                    _frozen_profile_interp,
                                                    _relaxation_residuals)

__all__ = ["ErYbAmplifier", "RateTemperatureLaw", "YbStarkThermal",
           "YB_STARK_976_CANAT_DUSSARDIER", "RATE_ARRHENIUS_CHENG_2022",
           "RATE_ARRHENIUS_30PCT_300_480K"]


@dataclass(frozen=True)
class RateTemperatureLaw:
    """ARRHENIUS temperature scaling of the three CO-DOPED rate coefficients (OPT-IN; the
    all-zero default is the exact identity and reproduces the isothermal model bit for bit).

        rate(T) = rate(T_ref) exp[-(Ea/kB) (1/T - 1/T_ref)]

    with Ea/kB in KELVIN -- one activation temperature per coefficient, exposed rather than
    buried, so a caller who disagrees with the calibration below can state their own. T_ref is
    NOT carried here: it is the T_ref of the amplifier's own `set_temperature_profile`, which is
    what makes a UNIFORM profile at T_ref an exact identity (gated) rather than a near one.

    WHAT IS MEASURED (Cheng et al., Materials 15(3), 996 (2022), Table 2, Er/Yb/P silica core
    glass, external heating 300 -> 480 K; the only temperature series on a co-doped
    phosphosilicate located by the 2026-09-14 literature validation):

        quantity                                300 K        480 K       change
        Yb decay FAST component tau_1           8.20 us      7.56 us     -7.8%
        Yb decay SLOW component tau_2           1333.76 us   1279.25 us  -4.1%
        Er 4I13/2 lifetime                      9.12 ms      8.14 ms     -10.7%
        sigma_a_Yb(974 nm)                      9.43e-25     5.86e-25    -37.9%
        sigma_a_Yb(940 nm)                      1.73e-25     1.73e-25    0 (stated insensitive)
        sigma_a_Yb(915 nm)                      1.73e-25     1.26e-25    -27.2%
        sigma_a_Yb(1018 nm)                     2.9e-26      6.0e-26     +107%

    The TRANSFER INDICATOR this model actually carries is the transfer rate itself, and the two
    decay components give it directly as R_tr = 1/tau_1 - 1/tau_2: 1.2120e5 -> 1.3149e5 1/s, i.e.
    +8.5% over 300 -> 480 K. `RATE_ARRHENIUS_CHENG_2022` is the Arrhenius law fitted to exactly
    that pair (Ea/kB = 65 K); it scales k_tr ONLY, because Cheng measure no separate indicator
    for the secondary transfer or for migration and assuming the same activation energy for them
    would be an invention. `RATE_ARRHENIUS_30PCT_300_480K` (Ea/kB = 210 K) is the law for the
    frequently-quoted "+30% transfer degradation over 300 -> 480 K" reading; NO measurement
    supporting that larger figure is present in the 2026-09-14 anchor collection, so it is
    shipped as a clearly-labelled alternative and NOT as the calibration.

    Sign convention: a POSITIVE Ea makes the rate RISE with temperature (the measured direction
    for the Yb -> Er transfer), because 1/T - 1/T_ref < 0 for T > T_ref.
    """

    k_tr_ea_over_k_K: float = 0.0
    k_tr2_ea_over_k_K: float = 0.0
    w_mig_ea_over_k_K: float = 0.0

    @property
    def is_identity(self) -> bool:
        """True when every activation temperature is exactly 0 -- the constructor then drops the
        law entirely, so an identity law is byte-identical to passing None."""
        return (float(self.k_tr_ea_over_k_K) == 0.0 and float(self.k_tr2_ea_over_k_K) == 0.0
                and float(self.w_mig_ea_over_k_K) == 0.0)

    @staticmethod
    def ea_over_k_from_ratio(ratio: float, T1_K: float, T2_K: float) -> float:
        """The activation temperature Ea/kB [K] for which rate(T2)/rate(T1) = `ratio`. The ONE
        place the two shipped constants below are derived, so neither is a transcribed number."""
        if not (ratio > 0.0 and T1_K > 0.0 and T2_K > 0.0 and T1_K != T2_K):
            raise ValueError("ea_over_k_from_ratio: need ratio > 0 and two distinct T > 0")
        return float(np.log(ratio) / (1.0 / float(T1_K) - 1.0 / float(T2_K)))

    def scale(self, T_K, T_ref_K):
        """(s_k_tr, s_k_tr2, s_w_mig) multipliers at temperature(s) T_K about T_ref_K. Exactly
        (1, 1, 1) at T == T_ref for any activation energy, and exactly (1, 1, 1) everywhere for
        the identity law."""
        dinv = 1.0 / np.asarray(T_K, float) - 1.0 / float(T_ref_K)
        return (np.exp(-float(self.k_tr_ea_over_k_K) * dinv),
                np.exp(-float(self.k_tr2_ea_over_k_K) * dinv),
                np.exp(-float(self.w_mig_ea_over_k_K) * dinv))


@dataclass(frozen=True)
class YbStarkThermal:
    """THERMAL DEPOPULATION of the Stark level the 976 nm ytterbium line uses (OPT-IN, default
    off). Both the 2F7/2 ground manifold and the 2F5/2 excited manifold are Stark-split; the
    976 nm line is the ZERO-PHONON line joining the LOWEST sub-level of each. As the fiber heats,
    population spreads into the higher sub-levels and the zero line loses oscillator strength in
    proportion to the Boltzmann occupancy of the level it starts and ends on.

        p0(T) = 1 / Z(T),   Z(T) = 1 + g exp(-Delta_E / (kB T)),
        factor(T) = p0(T) / p0(T_ref) = Z(T_ref) / Z(T)   <= 1 for T > T_ref

    -- a two-level effective manifold (one lumped excited Stark level of weight `degeneracy` at
    `delta_E_over_k_K` kelvin above the lowest), which is the simplest form that reproduces the
    measured pair. Canat (2006) / Dussardier (2005) give the 976 nm terminal level's thermal
    depopulation as 7% at 300 K and 13% at 400 K; `YB_STARK_976_CANAT_DUSSARDIER` is built from
    those two points by `from_upper_fractions`, so the ANCHORS, not a transcribed (Delta_E, g),
    are what the code carries. The resulting Delta_E/kB is 823 K = 572 cm^-1, the right magnitude
    for the Yb 2F5/2 Stark splitting.

    WHY IT IS NOT ALREADY IN THE McCUMBER SCALING, and why applying it does NOT double count.
    `set_temperature_profile` scales sigma_e by exp[(eps_ion - h nu)(1/kT - 1/kT_ref)]. At the
    ytterbium ZERO LINE h nu = eps_Yb exactly, so that factor is IDENTICALLY 1 at 976 nm at every
    temperature (gated), and it never touches sigma_a at any wavelength. The Stark depopulation
    is therefore orthogonal to it by construction: McCumber moves the RATIO sigma_e/sigma_a away
    from the zero line, this moves the SCALE of both AT the zero line.

    BOTH cross-sections are multiplied by the SAME factor, and that is forced rather than chosen:
    McCumber requires sigma_e = sigma_a at the zero line at every temperature, so any scale
    applied to one must be applied to the other or the relation is broken.

    BAND. The factor is applied only to channels within `band_half_width_m` of `band_center_m`
    (default 976 +/- 15 nm, which covers the 5.6 nm FWHM phosphosilicate pump line and excludes
    both 940 nm -- which Cheng measure to be temperature-INSENSITIVE -- and 1018 nm, whose
    cross-section RISES with temperature because it terminates on a thermally populated upper
    ground-manifold level). Outside the band nothing is scaled: this is a zero-line model, not a
    spectrum model, and pretending otherwise would get the 1-um band's sign wrong.

    SIZE CHECK. At 480 K this gives 0.89 of the 300 K cross-section, while Cheng MEASURE
    sigma_a_Yb(974 nm) falling to 0.62 -- so the Stark depopulation accounts for about a third of
    the measured loss and thermal line broadening (not modelled) for the rest. Read it as a LOWER
    BOUND on the pump-band softening.
    """

    delta_E_over_k_K: float
    degeneracy: float = 1.0
    band_center_m: float = 976e-9
    band_half_width_m: float = 15e-9

    def __post_init__(self):
        if not (self.delta_E_over_k_K > 0.0):
            raise ValueError("YbStarkThermal: delta_E_over_k_K must be > 0")
        if not (self.degeneracy > 0.0):
            raise ValueError("YbStarkThermal: degeneracy must be > 0")
        if not (self.band_center_m > 0.0 and self.band_half_width_m > 0.0):
            raise ValueError("YbStarkThermal: band_center_m and band_half_width_m must be > 0")

    @classmethod
    def from_upper_fractions(cls, T1_K: float, frac1: float, T2_K: float, frac2: float, **kw):
        """Build the two-level manifold from two MEASURED upper-Stark-level population fractions
        (1 - p0) at two temperatures -- an exact 2-point inversion, not a fit."""
        x1 = float(frac1) / (1.0 - float(frac1))
        x2 = float(frac2) / (1.0 - float(frac2))
        if not (x1 > 0.0 and x2 > 0.0):
            raise ValueError("from_upper_fractions: fractions must lie strictly in (0, 1)")
        dE = float(np.log(x2 / x1) / (1.0 / float(T1_K) - 1.0 / float(T2_K)))
        g = float(x1 * np.exp(dE / float(T1_K)))
        return cls(delta_E_over_k_K=dE, degeneracy=g, **kw)

    def upper_fraction(self, T_K):
        """1 - p0(T): the fraction of the manifold NOT in the level the zero line uses."""
        x = float(self.degeneracy) * np.exp(-float(self.delta_E_over_k_K) / np.asarray(T_K, float))
        return x / (1.0 + x)

    def factor(self, T_K, T_ref_K):
        """Z(T_ref)/Z(T): the multiplier on BOTH Yb cross-sections inside the band. Exactly 1.0
        at T == T_ref."""
        gg = float(self.degeneracy)
        dE = float(self.delta_E_over_k_K)
        z_ref = 1.0 + gg * np.exp(-dE / float(T_ref_K))
        z = 1.0 + gg * np.exp(-dE / np.asarray(T_K, float))
        return z_ref / z

    def band_mask(self, lambda_m):
        """Boolean (K,) mask of the channels the factor applies to."""
        lam = np.asarray(lambda_m, float)
        return np.abs(lam - float(self.band_center_m)) <= float(self.band_half_width_m)


# The 976 nm terminal-level thermal depopulation Canat (2006) / Dussardier (2005) quote: 7% at
# 300 K, 13% at 400 K -> Delta_E/kB = 823 K (572 cm^-1), degeneracy 1.17. A DOCUMENTED CONSTANT,
# not a default: ErYbAmplifier(yb_stark_thermal=...) is opt-in and defaults to None.
YB_STARK_976_CANAT_DUSSARDIER = YbStarkThermal.from_upper_fractions(300.0, 0.07, 400.0, 0.13)

# Arrhenius law fitted to Cheng 2022's OWN two decay components: the transfer rate
# R_tr = 1/tau_1 - 1/tau_2 rises from 1.2120e5 to 1.3149e5 1/s over 300 -> 480 K (+8.5%).
RATE_ARRHENIUS_CHENG_2022 = RateTemperatureLaw(
    k_tr_ea_over_k_K=RateTemperatureLaw.ea_over_k_from_ratio(
        (1.0 / 7.56e-6 - 1.0 / 1279.25e-6) / (1.0 / 8.20e-6 - 1.0 / 1333.76e-6), 300.0, 480.0))

# The larger "+30% over 300 -> 480 K" transfer-degradation figure, shipped as a labelled
# ALTERNATIVE: no measurement supporting it is present in the 2026-09-14 anchor collection.
RATE_ARRHENIUS_30PCT_300_480K = RateTemperatureLaw(
    k_tr_ea_over_k_K=RateTemperatureLaw.ea_over_k_from_ratio(1.30, 300.0, 480.0))


class ErYbAmplifier:
    """An Er:Yb co-doped fiber amplifier: an Er ion + a Yb sensitizer + a fiber + a channel plan
    (pumps, signals, an Er-band ASE band, and an optional Yb-band ASE band). Architecturally
    parallel to FiberAmplifier so the two can be swapped: it reuses the Pump / Signal / AseBand
    dataclasses and returns the SAME SteadyStateResult. solve() returns the z-profiles, the Er
    metastable fraction (nbar2_z = f2), the Yb inversion (meta['beta_yb_z'] = b2), the signal
    gain, and the transfer diagnostics.

    Parameters
    ----------
    er_ion, yb_ion : RareEarthIon
        The Er and Yb spectroscopy (cross-sections, lifetimes). Every channel is evaluated
        against BOTH ions at its wavelength.
    fiber : FiberSpec
        The doped fiber; fiber.n_t_m3 is the Er density N_Er. Cladding pumping uses
        fiber.clad_radius_m + Pump.cladding, exactly as in steady_state.
    pumps, signals : list
        steady_state.Pump / steady_state.Signal (reused verbatim).
    ase : AseBand, optional
        The Er C-band ASE band (both propagation directions). None -> ASE-free gain estimate.
    n_yb_m3 : float (keyword-only, required)
        The Yb density N_Yb.
    k_tr_m3_s : float
        Yb->Er forward energy-transfer coefficient k_tr [m^3/s] (phosphosilicate ~ 1.1e-22-5e-22;
        default 2e-22).
    k_back_m3_s : float
        Er->Yb back-transfer coefficient k_back [m^3/s] (default 0; see the phi refinement above).
    a32_per_s : float
        Er 4I11/2 -> 4I13/2 relaxation rate A_32 [1/s] (only enters through k_back; default 5e5).
    yb_ase : AseBand, optional
        A second ASE band (typically 1000-1100 nm) that tracks Yb-band / 1-um parasitic ASE.
    upconversion_C_up : float
        Er cooperative-upconversion coefficient C_up [m^3/s] (default 0; adds -C_up N_Er f2^2 to
        the Er balance, as in FiberAmplifier). OVERRIDDEN by `concentration.c_up_m3_s` when a
        ConcentrationModel is supplied -- the model is the ONE entry point, and passing a nonzero
        raw value alongside a model with a different one warns (the single-ion audit-B1 rule).
    yb_coupled_fraction : float
        f, the fraction of the ytterbium COUPLED to the erbium (default 1.0 = the single-pool
        model, bit-identical). f < 1 activates the Dong 2020 two-population model: (1 - f) N_Yb
        ions are pumped, decay and radiate at every channel but never transfer. Measured/fitted
        values: 0.83-0.85 (Sefler 2004, three double-clad fibers), 0.15-0.25 (Dong 2020, two
        high-power lasers), 0.85-0.95 (Canat 2006), 0.2-0.4 and 0.9 (the FORC "ETE" series).
    k_tr2_m3_s : float
        K2, the SECONDARY transfer coefficient [m^3/s] (default 0): Yb* + Er(4I13/2) ->
        Yb + Er(4F9/2 -> fast back to 4I13/2). Drains the coupled Yb inversion, leaves the Er
        population unchanged, and dumps the Yb quantum as heat. Sefler 2004: 1.5-4e-22.
    yb_migration_rate_per_s : float
        W_mig [1/s], the Yb-Yb migration exchange between the two pools (default 0). Detailed
        balance makes it a difference of EXCITATION FRACTIONS (module docstring); its only
        equilibrium is b2c = b2nc, and it conserves the total Yb excitation exactly.
    concentration : ConcentrationModel, optional
        Er pair-induced quenching (either convention), C_up, and the Yb photodarkening
        equilibrium gray loss -- the SAME object and the same semantics FiberAmplifier takes
        (concentration.py). None (default) -> the ideal model, bit-identical.
    rate_temperature : RateTemperatureLaw, optional
        Arrhenius temperature scaling of k_tr, K2 and W_mig (default None = isothermal
        coefficients, bit-identical). ACTS ONLY when an axial temperature profile is set: the
        law's T_ref is the profile's own, so a UNIFORM profile at T_ref is an exact identity.
        `RATE_ARRHENIUS_CHENG_2022` is the shipped calibration (+8.5% on k_tr over 300-480 K,
        from Cheng 2022's own two decay components).
    yb_stark_thermal : YbStarkThermal, optional
        Thermal depopulation of the Stark level the 976 nm Yb line uses -- a multiplier on BOTH
        Yb cross-sections inside a narrow band about the zero line (default None). Also acts only
        under a temperature profile. `YB_STARK_976_CANAT_DUSSARDIER` is the shipped constant
        (7% depopulated at 300 K, 13% at 400 K). It is ORTHOGONAL to the McCumber scaling, which
        is identically 1 at the Yb zero line and never touches sigma_a -- see the class.
    """

    def __init__(self, er_ion: RareEarthIon, yb_ion: RareEarthIon, fiber: FiberSpec,
                 pumps: List[Pump], signals: List[Signal], ase: Optional[AseBand] = None, *,
                 n_yb_m3: float, k_tr_m3_s: float = 2.0e-22, k_back_m3_s: float = 0.0,
                 a32_per_s: float = 5.0e5, yb_ase: Optional[AseBand] = None,
                 upconversion_C_up: float = 0.0, yb_coupled_fraction: float = 1.0,
                 k_tr2_m3_s: float = 0.0, yb_migration_rate_per_s: float = 0.0,
                 concentration=None, rate_temperature=None, yb_stark_thermal=None):
        if not (n_yb_m3 > 0.0):
            raise ValueError("ErYbAmplifier: n_yb_m3 (N_Yb) must be > 0")
        if not (a32_per_s > 0.0):
            raise ValueError("ErYbAmplifier: a32_per_s must be > 0")
        if not (0.0 <= float(yb_coupled_fraction) <= 1.0):
            raise ValueError("ErYbAmplifier: yb_coupled_fraction must be in [0, 1]; got %r"
                             % (yb_coupled_fraction,))
        if float(k_tr2_m3_s) < 0.0:
            raise ValueError("ErYbAmplifier: k_tr2_m3_s must be >= 0")
        if float(yb_migration_rate_per_s) < 0.0:
            raise ValueError("ErYbAmplifier: yb_migration_rate_per_s must be >= 0")
        self.er_ion, self.yb_ion, self.fiber = er_ion, yb_ion, fiber
        self.pumps, self.signals = list(pumps), list(signals)
        self.ase, self.yb_ase = ase, yb_ase
        self._n_er = float(fiber.n_t_m3)
        self._n_yb = float(n_yb_m3)
        self._k_tr = float(k_tr_m3_s)
        self._k_back = float(k_back_m3_s)
        self._a32 = float(a32_per_s)
        self._tau_er = float(er_ion.tau_s)
        self._tau_yb = float(yb_ion.tau_s)
        self._fc = float(yb_coupled_fraction)
        self._k_tr2 = float(k_tr2_m3_s)
        self._w_mig = float(yb_migration_rate_per_s)
        # THE single predicate that routes between the retained one-pool algebra and the
        # three-unknown one. Every new transfer mechanism is off exactly when this is False,
        # and every legacy code path below is then taken verbatim -- which is what makes the
        # byte-identity claim a structural property and not a numerical coincidence.
        self._two_pop = (self._fc != 1.0 or self._k_tr2 > 0.0 or self._w_mig > 0.0)
        self._Tz = None                     # optional axial T profile (set_temperature_profile)
        # An IDENTITY rate law collapses to the None path exactly as an identity
        # ConcentrationModel does, so `RateTemperatureLaw()` is byte-identical to omitting it.
        if rate_temperature is not None and getattr(rate_temperature, "is_identity", False):
            rate_temperature = None
        self.rate_temperature = rate_temperature
        self.yb_stark_thermal = yb_stark_thermal
        # THE predicate that decides whether a set temperature profile does anything BEYOND the
        # per-ion McCumber sigma_e scaling. False (the default) -> _mcc_matrices' 3rd and 4th
        # slots are None and every consumer takes its untouched pre-2026-09-15 branch.
        self._t_rates = (rate_temperature is not None or yb_stark_thermal is not None)
        # an all-default (identity) model collapses to the None path: truly byte-identical
        if concentration is not None and getattr(concentration, "is_identity", False):
            concentration = None
        self.concentration = concentration
        if concentration is not None:
            if float(upconversion_C_up) != 0.0 and \
                    float(upconversion_C_up) != float(concentration.c_up_m3_s):
                import warnings
                warnings.warn("ErYbAmplifier: upconversion_C_up=%.3g is OVERRIDDEN by "
                              "ConcentrationModel.c_up_m3_s=%.3g (the model wins; pass C_up "
                              "through the ConcentrationModel only -- the single-ion audit-B1 "
                              "rule, now shared by both amplifier classes)"
                              % (float(upconversion_C_up), concentration.c_up_m3_s),
                              stacklevel=2)
            self.upconversion_C_up = float(concentration.c_up_m3_s)
            # _n_er is the ACTIVE erbium everywhere below (the FiberAmplifier._n_active rule):
            # dark ions hold no gain and accept no transfer, they only absorb (see _coeffs).
            self._n_er_total = float(fiber.n_t_m3)
            self._n_er = float(concentration.active_density(fiber.n_t_m3))
            self._n_er_dark = float(concentration.dark_density(fiber.n_t_m3))
        else:
            self.upconversion_C_up = float(upconversion_C_up)
            self._n_er_total = self._n_er
            self._n_er_dark = 0.0

    # ---- type-preserving clone protocol ---------------------------------------------------
    def _clone(self, *, pumps: Optional[List[Pump]] = None,
               signals: Optional[List[Signal]] = None, ase=_KEEP, yb_ase=_KEEP) -> "ErYbAmplifier":
        """Clone through THIS class's own constructor, carrying EVERY opt-in: both ions'
        spectroscopy, the fiber, both ASE bands, the Yb density, the forward/back transfer
        coefficients, A_32, the Er upconversion coefficient, the two-population parameters
        (coupled fraction, secondary transfer, migration), the ConcentrationModel, the rate
        temperature law, the Yb Stark-thermal model and the axial temperature profile. The single place that lists what
        an ErYb clone must carry (the FiberAmplifier._clone analogue); the three public protocol
        methods below all route through it, so adding an opt-in to __init__ needs ONE edit here.
        PRIVATE -- callers outside the class use with_signals / with_pumps / without_ase."""
        new = ErYbAmplifier(self.er_ion, self.yb_ion, self.fiber,
                            list(self.pumps) if pumps is None else list(pumps),
                            list(self.signals) if signals is None else list(signals),
                            self.ase if ase is _KEEP else ase,
                            n_yb_m3=self._n_yb, k_tr_m3_s=self._k_tr,
                            k_back_m3_s=self._k_back, a32_per_s=self._a32,
                            yb_ase=self.yb_ase if yb_ase is _KEEP else yb_ase,
                            upconversion_C_up=self.upconversion_C_up,
                            yb_coupled_fraction=self._fc, k_tr2_m3_s=self._k_tr2,
                            yb_migration_rate_per_s=self._w_mig,
                            concentration=self.concentration,
                            rate_temperature=self.rate_temperature,
                            yb_stark_thermal=self.yb_stark_thermal)
        new._Tz = self._Tz                  # the axial T profile rides along (audit A-5)
        return new

    # ---- PUBLIC amplifier re-seed protocol -------------------------------------------------
    # The SAME three methods FiberAmplifier exposes, so AmplifierChain and metrics.* can rebuild
    # a stage without knowing which class they hold (audit A-3: the chain rebuilt every stage as
    # a FiberAmplifier and crashed here on the missing .ion; the A-3 follow-on found metrics.*
    # still reaching past the protocol into FiberAmplifier._clone, which does not exist here).
    def with_signals(self, signals: List[Signal]) -> "ErYbAmplifier":
        """Re-seed with a new signal list, preserving this class and every opt-in (see _clone)."""
        return self._clone(signals=signals)

    def with_pumps(self, pumps: List[Pump]) -> "ErYbAmplifier":
        """Re-seed with a new pump list, preserving this class and every opt-in (see _clone).
        Used by metrics.slope_efficiency to sweep the launched pump."""
        return self._clone(pumps=pumps)

    def without_ase(self) -> "ErYbAmplifier":
        """A copy with BOTH ASE bands dropped (the Er C-band and the Yb-band parasitic band) --
        the ASE-free configuration metrics.gain_spectrum probes the small-signal gain in."""
        return self._clone(ase=None, yb_ase=None)

    # ---- PUBLIC channel plan --------------------------------------------------------------
    def channel_plan(self) -> "ChannelPlan":
        """This amplifier's channel structure -- see steady_state.ChannelPlan (audit F-3).

        `channels` is None: every channel here carries BOTH ions' cross-sections, so there is no
        single ChannelSet to hand back. The per-ion spectroscopy is in `_plan()`'s dict under
        'sa_er'/'se_er'/'sa_yb'/'se_yb'. Everything geometric and structural -- wavelengths,
        directions, the ASE mask and its frequency bin widths, launched powers, overlap, background
        loss, mode counts -- is ion-independent and is reported here.

        Note this amplifier has TWO ASE bands (the Er signal band and the optional Yb band), so
        `indices('ase', 'fwd')` spans both; separate them by wavelength.
        """
        pl = self._plan()
        return ChannelPlan(lambda_m=np.asarray(pl["lam"], float).copy(),
                           direction=np.asarray(pl["u"], float).copy(),
                           is_ase=np.asarray(pl["is_ase"], bool).copy(),
                           dnu_hz=np.asarray(pl["dnu"], float).copy(),
                           kind=list(pl["kind"]),
                           launched_W=np.asarray(pl["bc"], float).copy(),
                           gamma=np.asarray(pl["gamma"], float).copy(),
                           loss_per_m=np.asarray(pl["loss"], float).copy(),
                           m_modes=np.where(np.asarray(pl["is_ase"], bool),
                                            np.asarray(pl["m"], float), 0.0),
                           channels=None)

    # ---- channel plan --------------------------------------------------------------------
    def _plan(self):
        """Assemble the channel arrays. Every channel carries BOTH ions' cross-sections; the
        overlap Gamma and background loss are ion-independent (geometry), and a cladding pump has
        its (core) overlap replaced by the doped-fraction cladding overlap A_dope/A_clad."""
        lam, u, is_ase, dnu, kind, bc, cladding, m = [], [], [], [], [], [], [], []
        for p in self.pumps:
            lam.append(p.lambda_m); u.append(+1.0 if p.direction == "fwd" else -1.0)
            is_ase.append(False); dnu.append(0.0); kind.append("pump")
            bc.append(p.power_W); cladding.append(p.cladding); m.append(2.0)
        for s in self.signals:
            lam.append(s.lambda_m); u.append(+1.0); is_ase.append(False); dnu.append(0.0)
            kind.append("signal"); bc.append(s.power_W); cladding.append(False); m.append(2.0)
        for band in (self.ase, self.yb_ase):
            if band is not None and band.n_bins > 0:
                edges = np.linspace(band.lambda_min_m, band.lambda_max_m, band.n_bins + 1)
                centres = 0.5 * (edges[:-1] + edges[1:])
                nu_edges = C_LIGHT / edges                       # bin width in FREQUENCY
                dnu_bins = np.abs(nu_edges[:-1] - nu_edges[1:])
                for direction in (+1.0, -1.0):
                    for cwl, dv in zip(centres, dnu_bins):
                        lam.append(float(cwl)); u.append(direction); is_ase.append(True)
                        dnu.append(float(dv)); kind.append("ase"); bc.append(0.0)
                        cladding.append(False); m.append(float(band.m_modes))
        lam = np.asarray(lam); u = np.asarray(u); is_ase = np.asarray(is_ase, bool)
        dnu = np.asarray(dnu); m = np.asarray(m)
        ch_er = ChannelSet.build(self.er_ion, self.fiber, lam, u, is_ase=is_ase, dnu_hz=dnu)
        ch_yb = ChannelSet.build(self.yb_ion, self.fiber, lam, u, is_ase=is_ase, dnu_hz=dnu)
        gamma = ch_er.gamma.copy()
        for k, cl in enumerate(cladding):
            if cl:
                gamma[k] = cladding_pump_overlap(self.fiber)
        return {"lam": lam, "u": u, "is_ase": is_ase, "dnu": dnu, "m": m, "kind": kind,
                "bc": np.asarray(bc), "gamma": gamma, "loss": ch_er.loss_per_m.copy(),
                "sa_er": ch_er.sigma_a, "se_er": ch_er.sigma_e,
                "sa_yb": ch_yb.sigma_a, "se_yb": ch_yb.sigma_e}

    def _coeffs(self, pl):
        """Hoist the per-channel overlap/cross-section/frequency products into a bundle once per
        solve (mirrors steady_state._coeffs). flux_* are the per-ion pumping/emission rates per
        unit power; g_* are the modal gain coefficients; s_pref_* the ASE spontaneous prefactors."""
        gamma, dnu, m = pl["gamma"], pl["dnu"], pl["m"]
        nu = C_LIGHT / pl["lam"]
        A = self.fiber.a_dope_m2
        inv_hnuA = 1.0 / (H_PLANCK * nu * A)
        N_Er, N_Yb = self._n_er, self._n_yb
        s_er = np.where(pl["is_ase"], gamma * N_Er * pl["se_er"] * m * H_PLANCK * nu * dnu, 0.0)
        s_yb = np.where(pl["is_ase"], gamma * N_Yb * pl["se_yb"] * m * H_PLANCK * nu * dnu, 0.0)
        c = {
            "flux_a_er": gamma * pl["sa_er"] * inv_hnuA,
            "flux_e_er": gamma * pl["se_er"] * inv_hnuA,
            "flux_a_yb": gamma * pl["sa_yb"] * inv_hnuA,
            "flux_e_yb": gamma * pl["se_yb"] * inv_hnuA,
            "g_e_er": gamma * N_Er * pl["se_er"], "g_a_er": gamma * N_Er * pl["sa_er"],
            "g_e_yb": gamma * N_Yb * pl["se_yb"], "g_a_yb": gamma * N_Yb * pl["sa_yb"],
            "loss": pl["loss"], "s_er": s_er, "s_yb": s_yb,
        }
        if self.concentration is not None:
            # Unbleachable pair-induced-quenching absorption: the dark erbium sits permanently
            # in 4I15/2, so it absorbs Gamma sigma_a_Er n_dark at EVERY channel and holds no
            # gain. Identical in form to FiberAmplifier._coeffs -- one convention, two classes.
            c["loss"] = c["loss"] + gamma * self._n_er_dark * pl["sa_er"]
        return c

    # ---- distributed temperature profile (per-ion McCumber z-scaling) ---------------------
    def set_temperature_profile(self, z_m, T_K, *, T_ref_K: float = 300.0):
        """Impose an axial temperature profile T(z) on the co-doped gain medium. EVERY
        sigma_e-proportional coefficient of BOTH ions (local emission gain, stimulated-emission
        rate, ASE spontaneous source) is scaled per-z by that ION's OWN McCumber factor

            mcc_ion,k(z) = exp[(eps_ion - h nu_k) (1/(kB T(z)) - 1/(kB T_ref))],

        eps_ion = `RareEarthIon.eps_J` -- a fitted `mccumber_eps_J` when the ion carries one and
        h c / zero_line_m otherwise. The two ions have DIFFERENT zero lines (Er 4I13/2 ~1530 nm,
        Yb 2F5/2 ~974 nm), so a single shared factor would be wrong by orders of magnitude in the
        pump band: each spectrum is scaled about its own line, which is the co-doped statement of
        FiberAmplifier.set_temperature_profile and reproduces spectroscopy.at_temperature applied
        to BOTH ions exactly for a uniform profile (gated). The two LIFETIMES are held (their
        T-dependence is second order at these Delta-T; spectroscopy docstring).

        WHAT ELSE THE PROFILE DRIVES, when the constructor was given it (2026-09-15). Both of
        these are OPT-IN and both are exactly the identity at T == T_ref, so a uniform profile at
        the reference temperature reproduces the isothermal solve bit for bit whether they are
        set or not:
          * `rate_temperature=RateTemperatureLaw(...)` -- Arrhenius scaling of k_tr, K2 and
            W_mig, per z. Cheng 2022 measure the Yb transfer component moving 8.20 -> 7.56 us
            over 300-480 K, i.e. the transfer RATE 1/tau_1 - 1/tau_2 rising 8.5%;
            `RATE_ARRHENIUS_CHENG_2022` is that law. k_back is NOT scaled (no measurement, and
            it is 0 by default).
          * `yb_stark_thermal=YbStarkThermal(...)` -- the Boltzmann depopulation of the Stark
            level the 976 nm line uses, a multiplier on BOTH Yb cross-sections inside a narrow
            band about the zero line. It is ORTHOGONAL to the McCumber factor above, which is
            identically 1 at the Yb zero line (h nu = eps_Yb there) and never touches sigma_a at
            any wavelength, so the two do not double count; `YB_STARK_976_CANAT_DUSSARDIER` is
            the shipped constant. sigma_a is otherwise still held.

        Cleared with clear_temperature_profile(). Used by thermal.solve_with_thermal_feedback,
        which drives Q(z) from this class's own rate balance (`_heat_profile_W_per_m`)."""
        z = np.asarray(z_m, float)
        T = np.asarray(T_K, float)
        if z.shape != T.shape or z.ndim != 1 or z.size < 2:
            raise ValueError("set_temperature_profile: z_m and T_K must be equal-length 1-D")
        if not np.all(T > 0.0):
            raise ValueError("set_temperature_profile: T_K must be > 0 everywhere")
        self._Tz = (z.copy(), T.copy(), float(T_ref_K))

    def clear_temperature_profile(self):
        self._Tz = None

    def _mcc_matrices(self, pl, z):
        """THE temperature bundle on the mesh z, or None when no profile is set (in which case
        every caller takes the untouched no-temperature branch). A 4-tuple:

            [0] (K, M) McCumber sigma_e scale for the ERBIUM spectrum
            [1] (K, M) McCumber sigma_e scale for the YTTERBIUM spectrum
            [2] (K, M) Yb Stark-band scale on BOTH Yb cross-sections, or None (YbStarkThermal)
            [3] (3, M) Arrhenius scale for (k_tr, K2, W_mig), or None (RateTemperatureLaw)

        Slots 2 and 3 are None unless the caller opted in, and every consumer branches on
        `is None` rather than multiplying by ones -- which is what keeps a profiled amplifier
        with neither opt-in byte-identical to the pre-2026-09-15 two-slot version."""
        if getattr(self, "_Tz", None) is None:
            return None
        zt, Tt, T_ref = self._Tz
        T = np.interp(np.asarray(z, float), zt, Tt)
        nu = C_LIGHT / pl["lam"]
        dinv = 1.0 / (KB * T) - 1.0 / (KB * T_ref)
        stark = rscale = None
        if getattr(self, "_t_rates", False):
            if self.yb_stark_thermal is not None:
                stark = np.ones((pl["lam"].size, T.size))
                stark[self.yb_stark_thermal.band_mask(pl["lam"]), :] = \
                    np.asarray(self.yb_stark_thermal.factor(T, T_ref), float)[None, :]
            if self.rate_temperature is not None:
                sc = self.rate_temperature.scale(T, T_ref)
                rscale = np.vstack([np.broadcast_to(np.asarray(x, float), T.shape) for x in sc])
        return (np.exp(np.outer(float(self.er_ion.eps_J) - H_PLANCK * nu, dinv)),
                np.exp(np.outer(float(self.yb_ion.eps_J) - H_PLANCK * nu, dinv)), stark, rscale)

    # ---- z-local coupled algebra ---------------------------------------------------------
    @staticmethod
    def _bracketed_newton(hdh):
        """Safeguarded Newton/bisection for a residual H(f2) that satisfies H(0) >= 0 and
        H(1) < 0, returning (f2, unpumped). `hdh(f)` returns (H, dH/df). THE single home of the
        root find: `_solve_fb` (one Yb pool) and `_solve_fbb` (two) differ only in the closed
        form of b2(f2) they substitute, never in how the scalar equation is solved."""
        if hdh(0.0)[0] <= 0.0:                            # no drive (unpumped, unseeded)
            return 0.0, True
        lo, hi, f = 0.0, 1.0, 0.5
        for _ in range(80):
            H, dH = hdh(f)
            if H > 0.0:
                lo = f
            else:
                hi = f
            f_new = f - H / dH if dH < 0.0 else 0.5 * (lo + hi)
            if not (lo < f_new < hi):
                f_new = 0.5 * (lo + hi)
            if abs(f_new - f) <= 1e-13 or (hi - lo) <= 1e-13:
                f = f_new
                break
            f = f_new
        return f, False

    def _solve_fb(self, Ra_Er: float, Re_Er: float, Ra_Yb: float, Re_Yb: float, rs=None
                  ) -> Tuple[float, float]:
        """Steady-state (f2, b2) at one z from the four per-ion rates. b2 is a closed form in f2
        (the Yb equation is linear in b2 for fixed f2); substituting it into the Er residual
        H(f2) leaves a single bracketed scalar equation on [0, 1] solved by a safeguarded
        Newton/bisection. H(0) >= 0 and H(1) < 0 (the transfer/absorption drive vanishes at full
        inversion, leaving only decay), so the root is bracketed and the solve cannot diverge.

        THIS IS THE ONE-Yb-POOL SPECIALIZATION (yb_coupled_fraction = 1, k_tr2 = 0, no
        migration), retained verbatim so the default model is bit-identical; `_solve_fbb` is the
        general three-unknown form and a gate holds the two to ~1e-14 of each other in the limit
        where both apply. Callers go through `_fbb`, which routes on `_two_pop`."""
        N_Er, N_Yb = self._n_er, self._n_yb
        k_tr, k_back, A32 = self._k_tr, self._k_back, self._a32
        if rs is not None:
            k_tr = k_tr * rs[0]                           # RateTemperatureLaw, this z only
        inv_tE, inv_tY = 1.0 / self._tau_er, 1.0 / self._tau_yb
        Cup = self.upconversion_C_up
        c = k_tr * N_Yb                                   # transfer-IN coefficient (x b2)
        s = k_tr * N_Er                                   # transfer-OUT (Yb drain) coeff (x (1-f2))
        Db0 = Ra_Yb + Re_Yb + inv_tY
        Da = Ra_Er + Re_Er + inv_tE

        def _b_phi(f):
            drain = s * (1.0 - f)
            if k_back <= 0.0:
                denom = Db0 + drain
                b = Ra_Yb / denom
                return b, 1.0, denom, drain
            b = Ra_Yb / (Db0 + drain)                     # phi = 1 seed
            phi = 1.0
            for _ in range(3):                            # inner fixed point (phi ~ 1, converges instantly)
                phi = A32 / (A32 + k_back * (1.0 - b) * N_Yb)
                b = Ra_Yb / (Db0 + drain * phi)
            return b, phi, Db0 + drain * phi, drain

        def _HdH(f):
            b, phi, denom, _ = _b_phi(f)
            dbdf = Ra_Yb * s * phi / (denom * denom)      # >0: less drain as f rises -> more Yb inversion
            G = c * phi * b
            H = Ra_Er * (1.0 - f) - Re_Er * f - f * inv_tE + G * (1.0 - f) - Cup * N_Er * f * f
            dH = -Da + c * phi * (dbdf * (1.0 - f) - b) - 2.0 * Cup * N_Er * f
            return H, dH, b

        f, _ = self._bracketed_newton(lambda x: _HdH(x)[:2])
        b, _, _, _ = _b_phi(f)
        return float(min(max(f, 0.0), 1.0)), float(min(max(b, 0.0), 1.0))

    def _solve_fbb(self, Ra_Er: float, Re_Er: float, Ra_Yb: float, Re_Yb: float, rs=None
                   ) -> Tuple[float, float, float]:
        """Steady-state (f2, b2c, b2nc) at one z with TWO ytterbium pools, the secondary transfer
        K2 and the migration exchange -- the three-unknown counterpart of `_solve_fb`, entered
        only when `_two_pop` is True.

        THE REDUCTION, and why it is still ONE bracketed scalar equation. At FIXED f2 the two Yb
        balances of the module docstring are LINEAR in (b2c, b2nc): the transfer drain, the K2
        drain and the migration exchange are all proportional to b2c or b2nc with f2-dependent
        coefficients only. Writing

            D   = R_a_Yb + R_e_Yb + 1/tau_Yb                               (both pools share it)
            Dc  = D + k_tr N_Er (1 - f2) phi + K2 N_Er f2                  (coupled extra drain)
            A   = [[Dc + W (1-f),   -W (1-f)],  [-W f,   D + W f]],   RHS = (R_a_Yb, R_a_Yb)

        the determinant collapses (the W^2 f(1-f) cross terms cancel exactly) to

            det = Dc D + W (f Dc + (1 - f) D)  > 0    always,
            b2c = R_a_Yb (D + W) / det,      b2nc = R_a_Yb (Dc + W) / det,

        a closed form in f2 with NO matrix solve and no possibility of a singular system. It has
        the right limits term by term: W -> 0 gives b2c = R_a/Dc and b2nc = R_a/D (an untouched
        plain-Yb pool); W -> inf gives b2c = b2nc = R_a/(f Dc + (1-f) D), the fully-mixed pool.
        Substituting b2c into the Er residual leaves the SAME scalar H(f2) on [0, 1] with
        H(0) >= 0 (the transfer and absorption drive) and H(1) < 0 (decay only), so
        `_bracketed_newton` is as unconditionally robust here as in the one-pool case. The Newton
        slope carries d b2c/d f2 = -b2c (D + W f) dDc/df2 / det with dDc/df2 = K2 N_Er - k_tr
        N_Er phi, which reduces to the one-pool `R_a s phi / Dc^2` when W = K2 = 0.

        phi is iterated with the same three-pass inner fixed point `_solve_fb` uses and is
        exactly 1.0 when k_back = 0 (the default). Note f2 = 0 exactly reproduces the Er-only
        limit and f = 0 reproduces a pure Yb-loss fiber: with no coupled pool c_in = 0, the Er
        equation loses its transfer drive, and b2nc = R_a_Yb / D is the textbook two-level Yb."""
        N_Er, N_Yb = self._n_er, self._n_yb
        fc = self._fc
        k_tr, k_back, A32 = self._k_tr, self._k_back, self._a32
        K2, Wm = self._k_tr2, self._w_mig
        if rs is not None:
            # RateTemperatureLaw at THIS z. k_back is deliberately NOT scaled: no temperature
            # measurement of the back transfer exists and it is exactly 0 by default anyway.
            k_tr, K2, Wm = k_tr * rs[0], K2 * rs[1], Wm * rs[2]
        inv_tE, inv_tY = 1.0 / self._tau_er, 1.0 / self._tau_yb
        Cup = self.upconversion_C_up
        c_in = k_tr * fc * N_Yb                       # transfer-IN coefficient (x b2c)
        s_out = k_tr * N_Er                           # coupled-Yb drain coefficient (x (1-f2))
        D = Ra_Yb + Re_Yb + inv_tY
        Da = Ra_Er + Re_Er + inv_tE

        def _pools(f):
            # phi = 1 seed, then the same three-pass inner fixed point _solve_fb's _b_phi runs,
            # in the SAME order: phi is updated from the current b2c FIRST and (Dc, det, b2c)
            # are then recomputed, so every value returned belongs to the same phi. (Updating
            # phi last would hand _HdH a phi that the Dc it also returns was not built with.)
            phi = 1.0
            Dc = D + s_out * (1.0 - f) + K2 * N_Er * f
            det = Dc * D + Wm * (fc * Dc + (1.0 - fc) * D)
            bc = Ra_Yb * (D + Wm) / det
            if k_back > 0.0:
                for _ in range(3):
                    phi = A32 / (A32 + k_back * fc * (1.0 - bc) * N_Yb)
                    Dc = D + s_out * (1.0 - f) * phi + K2 * N_Er * f
                    det = Dc * D + Wm * (fc * Dc + (1.0 - fc) * D)
                    bc = Ra_Yb * (D + Wm) / det
            return bc, Ra_Yb * (Dc + Wm) / det, phi, Dc, det

        def _HdH(f):
            bc, _bn, phi, _Dc, det = _pools(f)
            dDc = K2 * N_Er - s_out * phi
            dbdf = -bc * (D + Wm * fc) * dDc / det    # >0: less drain as f rises
            G = c_in * phi * bc
            H = Ra_Er * (1.0 - f) - Re_Er * f - f * inv_tE + G * (1.0 - f) - Cup * N_Er * f * f
            dH = -Da + c_in * phi * (dbdf * (1.0 - f) - bc) - 2.0 * Cup * N_Er * f
            return H, dH

        f, _ = self._bracketed_newton(_HdH)
        bc, bn, _phi, _Dc, _det = _pools(f)
        return (float(min(max(f, 0.0), 1.0)), float(min(max(bc, 0.0), 1.0)),
                float(min(max(bn, 0.0), 1.0)))

    def _fbb(self, Ra_Er, Re_Er, Ra_Yb, Re_Yb, rs=None):
        """(f2, b2c, b2nc, bbar) at one z, routed to the one- or two-pool algebra. bbar is the
        POPULATION-WEIGHTED Yb inversion f b2c + (1-f) b2nc -- the only Yb quantity the optical
        field, the parasitic-gain diagnostic and the photodarkening law ever see. b2nc is None
        for the one-pool model, where bbar IS b2 (the same object, so the caller is bitwise
        unaffected)."""
        if self._two_pop:
            f2, b2c, b2nc = self._solve_fbb(Ra_Er, Re_Er, Ra_Yb, Re_Yb, rs)
            return f2, b2c, b2nc, self._fc * b2c + (1.0 - self._fc) * b2nc
        f2, b2 = self._solve_fb(Ra_Er, Re_Er, Ra_Yb, Re_Yb, rs)
        return f2, b2, None, b2

    def _dP(self, c, u, P, mcc=None):
        """dP_k/dz [W/m] for every channel from the local power vector P (K,). `mcc` is the
        optional (mcc_er, mcc_yb) pair of (K,) per-channel McCumber sigma_e scale factors AT THIS
        z (set_temperature_profile); None -- the default and the only path a profile-free
        amplifier takes -- runs the untouched isothermal arithmetic."""
        P = np.maximum(P, 0.0)
        ga_yb, rs = c["g_a_yb"], None
        if mcc is None:
            Ra_Er = float(np.dot(c["flux_a_er"], P)); Re_Er = float(np.dot(c["flux_e_er"], P))
            Ra_Yb = float(np.dot(c["flux_a_yb"], P)); Re_Yb = float(np.dot(c["flux_e_yb"], P))
            ge_er, ge_yb, s_er, s_yb = c["g_e_er"], c["g_e_yb"], c["s_er"], c["s_yb"]
        else:
            m_er, m_yb, st, rs = mcc
            fa_yb = c["flux_a_yb"] if st is None else c["flux_a_yb"] * st
            fe_yb = c["flux_e_yb"] * m_yb if st is None else c["flux_e_yb"] * m_yb * st
            Ra_Er = float(np.dot(c["flux_a_er"], P))
            Re_Er = float(np.dot(c["flux_e_er"] * m_er, P))
            Ra_Yb = float(np.dot(fa_yb, P))
            Re_Yb = float(np.dot(fe_yb, P))
            ge_er, ge_yb = c["g_e_er"] * m_er, c["g_e_yb"] * m_yb
            s_er, s_yb = c["s_er"] * m_er, c["s_yb"] * m_yb
            if st is not None:
                ga_yb, ge_yb, s_yb = c["g_a_yb"] * st, ge_yb * st, s_yb * st
        f2, _b2c, _b2nc, b2 = self._fbb(Ra_Er, Re_Er, Ra_Yb, Re_Yb, rs)
        g = (ge_er * f2 - c["g_a_er"] * (1.0 - f2)
             + ge_yb * b2 - ga_yb * (1.0 - b2) - c["loss"])
        if self.concentration is not None:
            # Yb photodarkening equilibrium gray loss at the POPULATION-WEIGHTED Yb inversion --
            # both pools darken the glass, and an uncoupled pool sits at a HIGHER inversion than
            # a coupled one, so reading it off b2c alone would understate the loss.
            g = g - self.concentration.photodarkening_loss_per_m(b2)
        src = s_er * f2 + s_yb * b2
        return u * (g * P + src)

    def _fb_profile(self, c, P, mcc=None):
        """(f2(z), b2(z), b2nc(z)) at each z given the full power profile P (K, M). b2 is the
        COUPLED Yb inversion and b2nc the uncoupled one; b2nc is None for the one-pool model, in
        which case b2 is simply the Yb inversion. `mcc` is the optional ((K, M), (K, M)) McCumber
        scale pair on the SAME mesh."""
        M = P.shape[1]
        f2 = np.empty(M); b2 = np.empty(M)
        b2nc = np.empty(M) if self._two_pop else None
        st_m = None if mcc is None else mcc[2]
        rs_m = None if mcc is None else mcc[3]
        for j in range(M):
            Pj = np.maximum(P[:, j], 0.0)
            rs = None if rs_m is None else rs_m[:, j]
            if mcc is None:
                Ra_Er = float(np.dot(c["flux_a_er"], Pj))
                Re_Er = float(np.dot(c["flux_e_er"], Pj))
                Ra_Yb = float(np.dot(c["flux_a_yb"], Pj))
                Re_Yb = float(np.dot(c["flux_e_yb"], Pj))
            else:
                fa_yb = c["flux_a_yb"] if st_m is None else c["flux_a_yb"] * st_m[:, j]
                fe_yb = (c["flux_e_yb"] * mcc[1][:, j] if st_m is None
                         else c["flux_e_yb"] * mcc[1][:, j] * st_m[:, j])
                Ra_Er = float(np.dot(c["flux_a_er"], Pj))
                Re_Er = float(np.dot(c["flux_e_er"] * mcc[0][:, j], Pj))
                Ra_Yb = float(np.dot(fa_yb, Pj))
                Re_Yb = float(np.dot(fe_yb, Pj))
            if self._two_pop:
                f2[j], b2[j], b2nc[j] = self._solve_fbb(Ra_Er, Re_Er, Ra_Yb, Re_Yb, rs)
            else:
                f2[j], b2[j] = self._solve_fb(Ra_Er, Re_Er, Ra_Yb, Re_Yb, rs)
        return f2, b2, b2nc

    def _bbar(self, b2c, b2nc):
        """Population-weighted Yb inversion f b2c + (1-f) b2nc -- what the optical field sees
        (Dong 2020 n6 = n6c + n6nc). Returns b2c UNCHANGED (the same object) for the one-pool
        model, so every downstream expression is bitwise what it always was."""
        if b2nc is None:
            return b2c
        return self._fc * b2c + (1.0 - self._fc) * b2nc

    # ---- vectorized rate / RHS / Jacobian surface (shared with the transient march) --------
    # The steady solve needs only the z-local ROOT of the coupled pair (_solve_fb). A TIME MARCH
    # needs the pair's right-hand side and its Jacobian on the whole mesh at once, so the forms
    # below are the SINGLE HOME of that algebra: _solve_fb's residual H(f2) and _fb_rhs /
    # _fb_jacobian are the same two equations written once for a root find and once for an
    # integrator (dynamics.simulate_transient_eryb reads these rather than re-deriving them).

    def _rates_profile(self, c, P, mcc=None):
        """(R_a_Er, R_e_Er, R_a_Yb, R_e_Yb) [1/s] at every z from the power profile P (K, M) --
        the vectorized counterpart of the four np.dot calls _fb_profile makes per node. `mcc` is
        the optional ((K, M), (K, M)) McCumber sigma_e scale pair; it multiplies the two EMISSION
        rates only (sigma_a and the lifetimes are held)."""
        Pz = np.maximum(np.asarray(P, float), 0.0)
        if mcc is None:
            return (c["flux_a_er"] @ Pz, c["flux_e_er"] @ Pz,
                    c["flux_a_yb"] @ Pz, c["flux_e_yb"] @ Pz)
        m_er, m_yb, st = mcc[0], mcc[1], mcc[2]
        fa_yb = (c["flux_a_yb"] @ Pz if st is None
                 else np.sum(c["flux_a_yb"][:, None] * st * Pz, axis=0))
        fe_yb = (c["flux_e_yb"][:, None] * m_yb if st is None
                 else c["flux_e_yb"][:, None] * m_yb * st)
        return (c["flux_a_er"] @ Pz,
                np.sum(c["flux_e_er"][:, None] * m_er * Pz, axis=0),
                fa_yb,
                np.sum(fe_yb * Pz, axis=0))

    def _phi(self, b2):
        """Back-transfer branching factor phi = A_32 / (A_32 + k_back (1 - b2) N_Yb): the fraction
        of Er 4I11/2 population that relaxes to 4I13/2 before back-transferring (module
        docstring). The back-transfer acceptor is a GROUND-STATE COUPLED Yb, n5c = f N_Yb
        (1 - b2c), hence the f factor -- which is exactly 1.0 in the one-pool model, so
        multiplying by it changes no bit there. EXACTLY 1.0 overall when k_back = 0 (the
        default), so every expression carrying phi stays byte-identical to the
        no-back-transfer algebra."""
        b = np.asarray(b2, float)
        if self._k_back <= 0.0:
            return np.ones_like(b)
        return self._a32 / (self._a32 + self._k_back * self._fc * (1.0 - b) * self._n_yb)

    def _dphi_db(self, b2, phi):
        """d(phi)/d(b2c) = phi^2 k_back f N_Yb / A_32 (identically 0 when k_back = 0)."""
        b = np.asarray(b2, float)
        if self._k_back <= 0.0:
            return np.zeros_like(b)
        return phi * phi * self._k_back * self._fc * self._n_yb / self._a32

    def _fb_rhs3(self, Ra_Er, Re_Er, Ra_Yb, Re_Yb, f2, b2c, b2nc, rs=None):
        """(df2/dt, db2c/dt, db2nc/dt) [1/s]: the module docstring's THREE balances before the
        steady-state condition is imposed -- THE single home of the population algebra. The
        one-pool pair `_fb_rhs` is this function at b2nc = b2c (the migration term then vanishes
        identically, whatever W_mig is) with k_tr2 = 0 and f = 1, and `_solve_fbb` returns the
        (f2, b2c, b2nc) at which all three vanish, so a march built on this has the steady
        solve's own fixed point -- not a nearby one.

        EVERY new term is multiplied by a coefficient that is exactly 0 or exactly 1 in the
        legacy configuration (f = 1 makes `k_tr * f * N_Yb` the identical product `k_tr * N_Yb`;
        K2 = 0 and W_mig = 0 subtract exact zeros), which is why the default path is bit-exact
        rather than merely close."""
        fc = self._fc
        k_tr, k_tr2, w_mig = self._k_tr, self._k_tr2, self._w_mig
        if rs is not None:
            k_tr, k_tr2, w_mig = k_tr * rs[0], k_tr2 * rs[1], w_mig * rs[2]
        phi = self._phi(b2c)
        tr = phi * b2c * (1.0 - f2)           # transfer shape; x k_tr f N_Yb (in) / N_Er (out)
        mig = w_mig * (b2c - b2nc)            # detailed-balance exchange (module docstring)
        df = (Ra_Er * (1.0 - f2) - Re_Er * f2 - f2 / self._tau_er
              + k_tr * fc * self._n_yb * tr
              - self.upconversion_C_up * self._n_er * f2 * f2)
        dbc = (Ra_Yb * (1.0 - b2c) - Re_Yb * b2c - b2c / self._tau_yb
               - k_tr * self._n_er * tr
               - k_tr2 * self._n_er * f2 * b2c
               - (1.0 - fc) * mig)
        dbn = (Ra_Yb * (1.0 - b2nc) - Re_Yb * b2nc - b2nc / self._tau_yb
               + fc * mig)
        return df, dbc, dbn

    def _fb_rhs(self, Ra_Er, Re_Er, Ra_Yb, Re_Yb, f2, b2):
        """(df2/dt, db2/dt) [1/s] for a SINGLE Yb pool -- `_fb_rhs3` evaluated at b2nc = b2c, so
        the migration exchange is identically zero and only the Er and coupled-Yb balances
        survive. This is the pair the one-pool march and `energy_terms` use; with an uncoupled
        pool active the caller must use `_fb_rhs3`, which the three-reservoir march does."""
        df, dbc, _ = self._fb_rhs3(Ra_Er, Re_Er, Ra_Yb, Re_Yb, f2, b2, b2)
        return df, dbc

    def _fb_jacobian(self, Ra_Er, Re_Er, Ra_Yb, Re_Yb, f2, b2):
        """The EXACT 2x2 Jacobian (J11, J12, J21, J22) of _fb_rhs at (f2, b2).

        STRUCTURE (this is what licenses the exponential update in dynamics): J11 < 0 and J22 < 0
        (each is minus a sum of absorption, stimulated-emission, decay and transfer rates), while
        J12 >= 0 and J21 >= 0 (more Yb inversion pumps Er; more Er ground drains Yb). So
        tr(J) < 0; the cross terms CANCEL out of the determinant,

            det(J) = a d + a k_out (1 - f2) + k_in b2 d  > 0,
            a = R_a_Er + R_e_Er + 1/tau_Er + 2 C_up N_Er f2,   d = R_a_Yb + R_e_Yb + 1/tau_Yb,

        written here for k_back = 0; with k_back > 0 both cross terms pick up the SAME
        phi (phi + b2 dphi/db) factor and still cancel, leaving the same three positive terms with
        k_out (1 - f2) -> k_out (1 - f2)(phi + b2 dphi/db) and k_in b2 -> k_in phi b2. The
        discriminant (J11 - J22)^2 + 4 J12 J21 is likewise a sum of non-negative terms. Both
        eigenvalues are therefore REAL and NEGATIVE at every operating point: the linearised pair
        is a stable node, never a spiral, at any pump level and any transfer strength.

        This 2x2 is `_fb_jacobian3`'s leading block at b2nc = b2c; see there for the 3x3 the
        two-population march uses and for what survives of the argument above."""
        j = self._fb_jacobian3(Ra_Er, Re_Er, Ra_Yb, Re_Yb, f2, b2, b2)
        return j[0][0], j[0][1], j[1][0], j[1][1]

    def _fb_jacobian3(self, Ra_Er, Re_Er, Ra_Yb, Re_Yb, f2, b2c, b2nc, rs=None):
        """The EXACT 3x3 Jacobian of `_fb_rhs3` at (f2, b2c, b2nc), as a nested 3x3 list of
        arrays (row-major, y = (f2, b2c, b2nc)).

        STRUCTURE. J[0][2] = J[2][0] = 0: within a frozen-power step the erbium and the UNCOUPLED
        ytterbium never see each other -- they are connected only through the coupled pool. The
        diagonal is strictly negative (minus a sum of absorption, stimulated-emission, decay,
        transfer, K2 and migration rates). The migration block

            W [[-(1-f), (1-f)], [f, -f]]

        is a Markov generator with eigenvalues 0 and -W, and the similarity diag(1/sqrt(f),
        1/sqrt(1-f)) symmetrizes it, so it contributes REAL non-positive eigenvalues and cannot
        introduce a spiral by itself. With K2 = 0 the cross terms still cancel out of the
        determinant exactly as in the 2x2 case. With K2 > 0 the (b2c, f2) cross term picks up
        -K2 N_Er b2c and the cancellation is no longer exact, so no theorem is claimed for that
        corner: what IS asserted, and gated numerically over a sweep of operating points, is that
        every eigenvalue keeps a strictly negative real part, which is what the exponential
        Rosenbrock step in dynamics needs. phi_1 is evaluated on the matrix itself (scaling and
        squaring, no eigen-decomposition), so a complex pair would be handled correctly anyway."""
        fc = self._fc
        k_tr, k_tr2, w_mig = self._k_tr, self._k_tr2, self._w_mig
        if rs is not None:
            k_tr, k_tr2, w_mig = k_tr * rs[0], k_tr2 * rs[1], w_mig * rs[2]
        phi = self._phi(b2c)
        dphi = self._dphi_db(b2c, phi)
        k_in = k_tr * fc * self._n_yb               # transfer-IN coefficient (Er equation)
        k_out = k_tr * self._n_er                   # transfer-OUT coefficient (Yb equation)
        k2n = k_tr2 * self._n_er
        dtr_df = -phi * b2c
        dtr_db = (1.0 - f2) * (phi + b2c * dphi)
        zero = np.zeros_like(np.asarray(dtr_df, float))
        j00 = (-(Ra_Er + Re_Er + 1.0 / self._tau_er) + k_in * dtr_df
               - 2.0 * self.upconversion_C_up * self._n_er * f2)
        j01 = k_in * dtr_db
        j10 = -k_out * dtr_df - k2n * b2c
        j11 = (-(Ra_Yb + Re_Yb + 1.0 / self._tau_yb) - k_out * dtr_db - k2n * f2
               - w_mig * (1.0 - fc))
        j12 = w_mig * (1.0 - fc) + zero
        j21 = w_mig * fc + zero
        j22 = -(Ra_Yb + Re_Yb + 1.0 / self._tau_yb) - w_mig * fc
        return [[j00, j01, zero], [j10, j11, j12], [zero, j21, j22 + zero]]

    def _b2_quasi_equilibrium(self, Ra_Yb, Re_Yb, f2, b2):
        """The Yb inversion that balances the Yb equation at a FIXED Er state: the closed form
        b2 = R_a_Yb / (R_a_Yb + R_e_Yb + 1/tau_Yb + k_tr N_Er (1 - f2) phi) that _solve_fb
        substitutes into H(f2). phi is evaluated at the incoming b2 (exactly 1 unless k_back > 0).
        Used to SEED the Yb reservoir of a transient whose caller supplied only f2.

        With TWO pools this returns the PAIR (b2c, b2nc) from `_solve_fbb`'s closed form at the
        same fixed f2 -- the caller then has both reservoirs seeded consistently."""
        phi = self._phi(b2)
        denom = (Ra_Yb + Re_Yb + 1.0 / self._tau_yb
                 + self._k_tr * self._n_er * (1.0 - f2) * phi)
        if not self._two_pop:
            return Ra_Yb / denom
        D = Ra_Yb + Re_Yb + 1.0 / self._tau_yb
        Dc = denom + self._k_tr2 * self._n_er * f2
        det = Dc * D + self._w_mig * (self._fc * Dc + (1.0 - self._fc) * D)
        return (Ra_Yb * (D + self._w_mig) / det, Ra_Yb * (Dc + self._w_mig) / det)

    # ---- solve (relaxation, mirrors FiberAmplifier.solve) --------------------------------
    def solve(self, *, n_nodes: int = 201, max_iter: int = 200, tol: float = 1e-6,
              method: str = "LSODA", relax=1.0) -> SteadyStateResult:
        """Steady-state relaxation solve. `relax` is a number in (0, 1] -- DEFAULT 1.0, the plain
        undamped Gauss-Seidel iteration, unchanged -- or the string "auto", which walks
        steady_state._RELAX_LADDER (1.0, 0.5, 0.25) and returns the first attempt that CONVERGES,
        exactly as FiberAmplifier.solve does, reporting the ladder entries tried on
        `meta['relax_attempts']`.

        WHY "auto" IS ACCEPTED BUT NOT THE DEFAULT (2026-09-13). Accepting it is what makes this
        class substitutable in code written against FiberAmplifier -- a caller that passes the
        single-ion default through (the downstream `solve_closed(amp, relax="auto")` pattern) used
        to get a ValueError here. Making it the DEFAULT would be a behaviour flip for every
        existing co-doped caller, so it is not: `relax=1.0` still takes exactly one attempt and
        every previously-written call returns what it always did.

        See _solve_once for the method and for what under-relaxation is."""
        if isinstance(relax, str):
            if relax != "auto":
                raise ValueError("solve: relax must be a number in (0, 1] or \"auto\"; got %r"
                                 % (relax,))
            attempts, res = [], None
            for rx in _RELAX_LADDER:
                attempts.append(rx)
                res = self._solve_once(n_nodes=n_nodes, max_iter=max_iter, tol=tol, method=method,
                                       relax=rx)
                if res.meta.get("converged"):
                    break
            res.meta["relax_attempts"] = tuple(attempts)
            return res
        res = self._solve_once(n_nodes=n_nodes, max_iter=max_iter, tol=tol, method=method,
                               relax=relax)
        res.meta["relax_attempts"] = (float(relax),)
        return res

    def _solve_once(self, *, n_nodes: int = 201, max_iter: int = 200, tol: float = 1e-6,
                    method: str = "LSODA", relax: float = 1.0) -> SteadyStateResult:
        """ONE relaxation solve at a fixed `relax`; see FiberAmplifier._solve_once for the method.

        UNDER-RELAXATION (`relax` in (0, 1], audit F-14). The same alternating frozen-direction
        iteration as steady_state.solve, hence the same failure mode -- a counter-propagating pump
        can stop the Gauss-Seidel oscillation from decaying and leave the sweep on a spurious,
        essentially unpumped branch -- and this solver shipped without the remedy. `relax` blends
        each pass with the previous profile,
        `profile <- relax * (this pass) + (1 - relax) * (previous)`, which changes the PATH to the
        fixed point and not the fixed point itself. relax = 1.0 (the default) skips the blend
        entirely and is byte-identical to this method's pre-fix behaviour.

        `meta` carries `endpoint_residual` / `profile_residual` / `relax` / `min_power_W`, the same
        four diagnostics steady_state.solve reports."""
        from scipy.integrate import solve_ivp
        if not (0.0 < float(relax) <= 1.0):
            raise ValueError("solve: relax must be in (0, 1]; got %r" % (relax,))
        relax = float(relax)
        pl = self._plan()
        c = self._coeffs(pl)
        u, is_ase, kind, bc = pl["u"], pl["is_ase"], pl["kind"], pl["bc"]
        K = pl["lam"].size
        L = self.fiber.length_m
        z = np.linspace(0.0, L, n_nodes)
        fwd = np.where(u > 0)[0]
        bwd = np.where(u < 0)[0]

        P_bwd = (np.repeat(bc[bwd][:, None], n_nodes, axis=1) if bwd.size
                 else np.zeros((0, n_nodes)))
        P_fwd = np.repeat(bc[fwd][:, None], n_nodes, axis=1)

        def _assemble(Pf, Pb):
            P = np.empty(K)
            P[fwd] = Pf
            if bwd.size:
                P[bwd] = Pb
            return P

        # Lean frozen-profile interpolator on the uniform mesh (steady_state audit S6-2). Audit
        # X-3: IMPORTED, not re-copied -- the verbatim copy that used to live here carried the
        # endpoint-clamp / uniform-mesh contract twice in a module that already imports from
        # steady_state.
        def _make_interp(Y):
            return _frozen_profile_interp(Y, z, L, n_nodes)

        # Per-z McCumber sigma_e scaling of BOTH ions, frozen on the solver mesh and interpolated
        # into the IVP exactly as steady_state does for the single ion. None -- and therefore the
        # untouched isothermal RHS -- whenever no profile is set.
        mcc_mat = self._mcc_matrices(pl, z)
        mcc_er_of = _make_interp(mcc_mat[0]) if mcc_mat is not None else None
        mcc_yb_of = _make_interp(mcc_mat[1]) if mcc_mat is not None else None
        # Slots 2 and 3 (the Yb Stark band scale and the Arrhenius rate scale) ride the SAME
        # frozen-profile interpolator, so every temperature-dependent coefficient the RHS sees
        # is evaluated at the same interpolated z as the McCumber factors.
        stark_of = (_make_interp(mcc_mat[2])
                    if (mcc_mat is not None and mcc_mat[2] is not None) else None)
        rscale_of = (_make_interp(mcc_mat[3])
                     if (mcc_mat is not None and mcc_mat[3] is not None) else None)

        def _mcc_at(zz):
            if mcc_mat is None:
                return None
            return (mcc_er_of(zz), mcc_yb_of(zz),
                    None if stark_of is None else stark_of(zz),
                    None if rscale_of is None else rscale_of(zz))

        last_out = None
        last_prof = None
        converged = False
        end_resid = prof_resid = float("nan")     # final residuals, reported on meta (audit F-14)
        it = 0
        for it in range(max_iter):
            bwd_of = _make_interp(P_bwd) if bwd.size else None

            def rhs_f(zz, Pf):
                Pb = bwd_of(zz) if bwd.size else np.zeros(0)
                return self._dP(c, u, _assemble(Pf, Pb), _mcc_at(zz))[fwd]

            sf = solve_ivp(rhs_f, (0.0, L), bc[fwd], t_eval=z, method=method,
                           rtol=1e-7, atol=1e-15)
            # Under-relaxation (audit F-14), mirroring steady_state._solve_once: the relax == 1.0
            # branch is taken verbatim so the default path performs NO arithmetic on the arrays
            # and stays byte-identical -- a `1.0 * new + 0.0 * old` blend would not, because
            # 0.0 * inf is NaN and -0.0 + 0.0 flips a sign.
            P_fwd = sf.y if relax == 1.0 else relax * sf.y + (1.0 - relax) * P_fwd

            if bwd.size:
                fwd_of = _make_interp(P_fwd)

                def rhs_b(zz, Pb):
                    return self._dP(c, u, _assemble(fwd_of(zz), Pb), _mcc_at(zz))[bwd]

                sb = solve_ivp(rhs_b, (L, 0.0), bc[bwd], t_eval=z[::-1], method=method,
                               rtol=1e-7, atol=1e-15)
                new_bwd = sb.y[:, ::-1]
                P_bwd = new_bwd if relax == 1.0 else relax * new_bwd + (1.0 - relax) * P_bwd

            out = np.concatenate([P_fwd[:, -1], (P_bwd[:, 0] if bwd.size else [])])
            prof = np.concatenate([P_fwd, P_bwd], axis=0) if bwd.size else P_fwd.copy()
            if last_out is not None:
                # SAME convergence test as steady_state.solve -- literally the same function now
                # (audit F-13: this was a byte-identical COPY, so the endpoint-denominator defect
                # existed in triplicate).
                end_resid, prof_resid = _relaxation_residuals(out, last_out, prof, last_prof)
                if end_resid < tol and prof_resid < tol:
                    converged = True
                    break
            last_out = out
            last_prof = prof

        P = np.empty((K, n_nodes))
        P[fwd] = P_fwd
        if bwd.size:
            P[bwd] = P_bwd
        f2, b2, b2nc = self._fb_profile(c, P, mcc_mat)
        bbar = self._bbar(b2, b2nc)
        sig_idx = [i for i, kd in enumerate(kind) if kd == "signal"]
        gains_dB = np.array([10.0 * np.log10(P[i, -1] / bc[i]) for i in sig_idx])

        eta_tr = self._transfer_efficiency(c, P, f2, b2, z, b2nc, mcc_mat)
        yb_par_dB = self._yb_parasitic_gain_dB(P, f2, bbar, z)
        m_modes = (self.ase.m_modes if self.ase is not None
                   else (self.yb_ase.m_modes if self.yb_ase is not None else 2))
        meta = {"converged": converged, "iterations": it + 1,
                # audit F-14, lifted here from steady_state: the ACTUAL final residuals, the relax
                # value that produced them, and the most negative returned power (a fully absorbed
                # channel sitting in integrator noise -- physically zero, but it will surprise a
                # log or a sqrt).
                "endpoint_residual": end_resid, "profile_residual": prof_resid,
                "relax": float(relax), "min_power_W": float(np.min(P)),
                "dnu_hz": pl["dnu"].copy(), "gamma": pl["gamma"].copy(), "m_modes": m_modes,
                "sigma_a": pl["sa_er"].copy(), "sigma_e": pl["se_er"].copy(),
                "sigma_a_er": pl["sa_er"].copy(), "sigma_e_er": pl["se_er"].copy(),
                "sigma_a_yb": pl["sa_yb"].copy(), "sigma_e_yb": pl["se_yb"].copy(),
                "beta_yb_z": bbar, "eta_transfer": eta_tr,
                "yb_parasitic_gain_dB": yb_par_dB,
                "n_er_m3": self._n_er, "n_yb_m3": self._n_yb,
                "k_tr_m3_s": self._k_tr, "k_back_m3_s": self._k_back, "a32_per_s": self._a32,
                # ---- two-population / concentration / temperature state ----
                # beta_yb_z above is the POPULATION-WEIGHTED inversion the field sees; the two
                # pools are reported separately, and beta_yb_uncoupled_z is None for the
                # one-pool model (where beta_yb_coupled_z IS beta_yb_z, the same array).
                "beta_yb_coupled_z": b2, "beta_yb_uncoupled_z": b2nc,
                "yb_coupled_fraction": self._fc, "k_tr2_m3_s": self._k_tr2,
                "yb_migration_rate_per_s": self._w_mig,
                "n_er_active_m3": self._n_er, "n_er_dark_m3": self._n_er_dark,
                "upconversion_C_up": self.upconversion_C_up,
                # 'mcc' is the ERBIUM matrix, matching meta['sigma_a'] / ['sigma_e'] (which are
                # also the Er spectra) so noise.local_inversion_factor -- a C-band quantity --
                # keeps reading a (K, M) array of the right ion. The per-ion pair is explicit.
                "mcc": None if mcc_mat is None else mcc_mat[0].copy(),
                "mcc_er": None if mcc_mat is None else mcc_mat[0].copy(),
                "mcc_yb": None if mcc_mat is None else mcc_mat[1].copy()}
        return SteadyStateResult(z, P, pl["lam"], u, is_ase, kind, f2, gains_dB, meta=meta)

    # ---- diagnostics ---------------------------------------------------------------------
    def _transfer_efficiency(self, c, P, f2, b2, z, b2nc=None, mcc=None) -> float:
        """Fraction of Yb excited-state de-excitations that end as a USEFUL Yb->Er transfer,
        as a rate integral weighted by the excited-Yb density n_Yb2 = b2 N_Yb:

            eta_tr = INT k_tr n1 n_Yb2 dz / INT (k_tr n1 + 1/tau_Yb + R_e_Yb) n_Yb2 dz,

        where n1 = (1 - f2) N_Er is the Er ground (acceptor) density and R_e_Yb(z) is the local
        Yb stimulated-emission rate (a Yb photon re-emitted to the field instead of transferred).
        The denominator is the total Yb* loss rate: transfer + spontaneous decay + stimulated
        emission. At low power (R_e_Yb -> 0, f2 -> 0 so n1 -> N_Er) this collapses to the
        analytic k_tr N_Er tau_Yb / (1 + k_tr N_Er tau_Yb) -- see the module DISCREPANCY NOTE.

        WITH TWO POOLS this becomes the quantity FORC call the ENERGY TRANSFER EFFICIENCY (ETE):
        the UNCOUPLED pool's excitation appears in the denominator (it de-excites by fluorescence
        and stimulated emission and never transfers) and the secondary transfer K2 joins the
        non-useful channels of the coupled pool,

            eta = INT k_tr n1 n6c dz
                  / INT [(k_tr n1 + K2 n2 + 1/tau_Yb + R_e_Yb) n6c + (1/tau_Yb + R_e_Yb) n6nc] dz

        with n6c = f N_Yb b2c and n6nc = (1-f) N_Yb b2nc. At f = 1 and K2 = 0 the extra terms are
        identically zero and the expression is the one above."""
        N_Er, N_Yb = self._n_er, self._n_yb
        n1 = (1.0 - f2) * N_Er
        k_tr, k_tr2 = self._k_tr, self._k_tr2
        if mcc is None:
            Re_Yb = np.array([float(np.dot(c["flux_e_yb"], np.maximum(P[:, j], 0.0)))
                              for j in range(P.shape[1])])
        else:
            st = mcc[2]
            Re_Yb = np.array([float(np.dot(c["flux_e_yb"] * mcc[1][:, j]
                                           * (1.0 if st is None else st[:, j]),
                                           np.maximum(P[:, j], 0.0)))
                              for j in range(P.shape[1])])
            if mcc[3] is not None:
                # the transfer coefficients are z-dependent under a RateTemperatureLaw, so the
                # rate integral has to carry them INSIDE the integrand, not outside it
                k_tr, k_tr2 = k_tr * mcc[3][0], k_tr2 * mcc[3][1]
        if b2nc is None:
            nYb2 = b2 * N_Yb
            transfer = k_tr * n1 * nYb2
            total = (k_tr * n1 + 1.0 / self._tau_yb + Re_Yb) * nYb2
        else:
            n6c = self._fc * N_Yb * b2
            n6nc = (1.0 - self._fc) * N_Yb * b2nc
            idle = 1.0 / self._tau_yb + Re_Yb
            transfer = k_tr * n1 * n6c
            total = ((k_tr * n1 + k_tr2 * N_Er * f2 + idle) * n6c + idle * n6nc)
        num = float(trapz(transfer, z))
        den = float(trapz(total, z))
        return num / den if den > 0.0 else 0.0

    def transfer_efficiency(self, result: SteadyStateResult) -> float:
        """eta_tr for a solved result (returns the value cached in meta by solve())."""
        return float(result.meta["eta_transfer"])

    def _yb_parasitic_gain_dB(self, P, f2, b2, z) -> float:
        """Single-pass 1030 nm Yb parasitic gain [dB] from the POPULATION-WEIGHTED b2 profile (computed even with no
        yb_ase channels): G_dB = (10/ln10) INT Gamma(1030) [N_Yb (sigma_e_Yb b2 - sigma_a_Yb
        (1 - b2)) + N_Er (sigma_e_Er f2 - sigma_a_Er (1 - f2))] dz. The Er terms are ~0 at 1030
        nm. A large positive value flags 1-um parasitic-lasing risk (compare to the round-trip
        cavity loss -ln(R1 R2)/2; dossier design rule beta_Yb < ~0.05)."""
        lam = 1.030e-6
        gam = float(overlap_gamma(self.fiber, lam))
        se_yb = float(self.yb_ion.sigma_e.sigma(lam)); sa_yb = float(self.yb_ion.sigma_a.sigma(lam))
        se_er = float(self.er_ion.sigma_e.sigma(lam)); sa_er = float(self.er_ion.sigma_a.sigma(lam))
        g = gam * (self._n_yb * (se_yb * b2 - sa_yb * (1.0 - b2))
                   + self._n_er * (se_er * f2 - sa_er * (1.0 - f2)))
        ln_gain = float(trapz(g, z))
        return 10.0 / np.log(10.0) * ln_gain

    def yb_parasitic_gain_dB(self, result: SteadyStateResult) -> float:
        """1030 nm Yb parasitic gain [dB] for a solved result (cached in meta by solve())."""
        return float(result.meta["yb_parasitic_gain_dB"])

    # ---- energy bookkeeping (2026-09-13 closure work) --------------------------------------
    def _rate_balance_dissipation_W(self, result: SteadyStateResult) -> float:
        """Total dissipated optical power [W] from the LOCAL RATE BALANCE -- this class's
        counterpart of the FiberAmplifier `_dP_full_c` path, and the hook
        efficiency._dissipated_power_W looks for FIRST, so wall_plug_efficiency returns a finite
        `energy_balance_residual_W` for a co-doped result instead of NaN.

        The net forward flux is F(z) = sum_fwd P - sum_bwd P, so dF/dz = sum_k u_k dP_k/dz =
        sum_k (g_k P_k + s_k) -- THIS amplifier's own right-hand side (_dP: both ions' gain, both
        ions' spontaneous source, the z-local (f2, b2) re-solved from the returned powers by
        _solve_fb) evaluated ON the returned profile rather than accumulated along it. The heat is
        -INT dF/dz dz. Because the profile is re-fed through the same algebra the solve used, the
        residual `launched - exiting - dissipated` measures whether the returned P(z) actually
        satisfies the amplifier's ODEs, which a flux difference of the same two endpoints
        (thermal.total_heat_W) cannot -- see the efficiency module docstring.

        WHERE THE CO-DOPED LOSS CHANNELS SIT IN THIS NUMBER (all of them are INSIDE it; none is a
        separate additive term, and adding one would DOUBLE-COUNT):
          * Yb fluorescence. A Yb excitation lost to spontaneous decay leaves the guided field as
            an unreplaced 976 nm absorption -- it entered the balance when the pump photon was
            absorbed and never comes back out, except for the tiny guided fraction s_yb b2 that is
            an explicit ASE source and IS returned to the field.
          * Er fluorescence. Identically, via s_er f2 for the guided part.
          * The Yb -> Er transfer defect. The model promotes Er straight from 4I15/2 to 4I13/2 (the
            fast-4I11/2 limit adiabatically eliminates the intermediate level), so the 976 nm Yb
            quantum minus the ~1530 nm Er quantum -- the 4I11/2 -> 4I13/2 multiphonon relaxation,
            0.46 eV, ~36% of the pump photon -- never appears as light anywhere and is therefore
            ALREADY the difference between the pump power absorbed by Yb and the signal power
            emitted by Er. It is the dominant dissipation term of an EYDFA.
        `energy_terms` splits exactly this number into those named pieces; this function is the
        aggregate the efficiency layer wants.

        Returns NaN when the result did not come from this amplifier's channel plan."""
        pl = self._plan()
        P, z = result.power_W, result.z_m
        if P.shape[0] != pl["lam"].size or P.shape[1] != z.size:
            return float("nan")
        c = self._coeffs(pl)
        u = pl["u"]
        mcc = self._mcc_matrices(pl, z)
        if mcc is None:
            dF = np.array([float(np.sum(u * self._dP(c, u, P[:, j]))) for j in range(z.size)])
        else:
            dF = np.array([float(np.sum(u * self._dP(
                c, u, P[:, j], (mcc[0][:, j], mcc[1][:, j],
                                None if mcc[2] is None else mcc[2][:, j],
                                None if mcc[3] is None else mcc[3][:, j]))))
                for j in range(z.size)])
        return -float(trapz(dF, z))

    def _heat_profile_W_per_m(self, result) -> np.ndarray:
        """Local heat density Q(z) [W/m] from THIS amplifier's own rate balance, for
        thermal.solve_with_thermal_feedback -- the co-doped counterpart of
        thermal.heat_load_per_m, which the single-ion class still uses.

        WHY NOT THE FLUX GRADIENT. `heat_load_per_m` is -d/dz of the net forward flux evaluated
        by np.gradient on the returned profile: second-order accurate, but it inherits the mesh
        error of the solve AND it says nothing about WHICH mechanism deposited the heat. This
        returns `energy_terms(...)['dissipation']`, the analytic sum

            background loss + Er dissipation + Yb dissipation + transfer defect + K2 defect,

        which at the steady state equals q_optical identically (that identity is the closure
        gate). For an EYDFA the transfer defect is the DOMINANT term -- the 4I11/2 -> 4I13/2
        multiphonon step, 36% of every 976 nm pump photon that reaches the erbium -- and the Yb
        fluorescence of an uncoupled pool is the second, so a co-doped thermal loop driven by the
        Er quantum defect alone would understate the load badly."""
        z = np.asarray(result.z_m, float)
        f2 = np.asarray(result.nbar2_z, float)
        b2c = np.asarray(result.meta.get("beta_yb_coupled_z", result.meta["beta_yb_z"]), float)
        b2nc = result.meta.get("beta_yb_uncoupled_z")
        terms = self.energy_terms(result.power_W, f2, b2c, b2nc, z_m=z)
        return np.asarray(terms["dissipation"], float)

    def energy_terms(self, power_W, f2, b2, b2_uncoupled=None, *, z_m=None) -> dict:
        """Per-z energy bookkeeping [W/m] for one profile: powers (K, M), f2 (M,), b2 (M,).

        `b2` is the COUPLED Yb inversion and `b2_uncoupled` the uncoupled pool's; the latter is
        REQUIRED (and refused if missing) whenever an uncoupled pool exists, because guessing it
        would silently mis-state both the stored energy and the Yb fluorescence. For the one-pool
        model it must be None and `b2` is simply the Yb inversion. `z_m` is needed only when a
        temperature profile is set (it is what the per-z McCumber factors are interpolated onto);
        omitting it on a profiled amplifier is refused rather than silently run isothermally.

        DERIVATION. Write eps_Er and eps_Yb for the two stored excitation energies (the ions'
        McCumber zero-line quanta, RareEarthIon.eps_J: ~1.30e-19 J at 1530 nm for Er 4I13/2,
        ~2.04e-19 J at 975 nm for Yb 2F5/2) and let A = A_dope. Per unit length, with photon
        rates n_* = power/(h nu) summed over channels:

            U(z)     = A (eps_Er N_Er f2 + eps_Yb N_Yb b2)                     stored [J/m]
            Phi_tr   = A k_tr phi N_Er N_Yb b2 (1 - f2)                        transfers/(m s)
            q_opt(z) = -dF/dz = (absorbed + background loss)
                                - (stimulated emitted + guided spontaneous)

        Each event is charged at the level's own energy and the remainder is dissipated: an
        absorbed photon at h nu deposits eps_Er and sheds (h nu - eps_Er) to phonons; a
        stimulated-emission event removes eps_Er and hands h nu to the field (the SAME expression
        with the opposite sign -- anti-Stokes cooling when h nu > eps_Er); a spontaneous decay
        removes eps_Er of which only the guided part sum_k s_k reaches the field; upconversion
        removes eps_Er outright. Summing,

            D_Er   = sum_k (h nu_k - eps_Er)(n_a,k - n_e,k) + (eps_Er Phi_dec_Er - P_sp_Er)
                     + eps_Er Phi_up
            D_Yb   = sum_k (h nu_k - eps_Yb)(n_a,k - n_e,k) + (eps_Yb Phi_dec_Yb - P_sp_Yb)
            D_tr   = (eps_Yb - eps_Er) Phi_tr      <-- the 976 nm -> 1530 nm transfer defect
            D_loss = sum_k l_k P_k

        and the two sides close IDENTICALLY, which is the content of this method:

            q_opt(z) = dU/dt(z) + D_loss + D_Er + D_Yb + D_tr.

        The transfer defect enters with a PLUS sign on the dissipation side and is charged ONCE,
        to the transfer event -- not to the Yb absorption (which already paid only its own
        h nu - eps_Yb) and not to the Er emission. At the steady state dU/dt = 0 and the whole of
        q_opt is dissipation, which is why `_rate_balance_dissipation_W` needs no extra term.

        Returns a dict of (M,) arrays: 'q_optical', 'stored_J_per_m', 'd_stored_dt',
        'background_loss', 'er_dissipation', 'yb_dissipation', 'transfer_defect', 'k2_defect',
        'dissipation' (the five summed), 'transfer_rate_per_m', 'k2_rate_per_m', 'df2_dt',
        'db2_dt', plus 'db2nc_dt' and 'beta_yb_mean' when a second pool is present. 'k2_defect'
        and 'k2_rate_per_m' are identically zero when k_tr2 = 0. The identity above is an
        algebraic rearrangement of one rate balance, so a departure from it beyond round-off is an
        implementation defect, not physics -- which is what makes it usable as a gate on a
        TRANSIENT march, where dU/dt is no longer zero and the marched populations are an
        independent computation from the propagated powers.

        TWO-POPULATION AND SECONDARY-TRANSFER EXTENSION. With f < 1 every OPTICAL Yb term and the
        Yb fluorescence are evaluated at the population-weighted bbar = f b2c + (1-f) b2nc (the
        uncoupled pool absorbs, emits and fluoresces exactly like the coupled one), the stored
        energy is A eps_Yb N_Yb bbar, and its rate is A eps_Yb N_Yb (f db2c/dt + (1-f) db2nc/dt)
        -- in which the migration exchange cancels identically, which is the numerical signature
        that the exchange term conserves excitation. The transfer terms carry the COUPLED density
        f N_Yb, and the secondary transfer adds ONE new dissipation channel,

            Phi_K2 = A K2 N_Er f2 (f N_Yb b2c),      D_K2 = eps_Yb Phi_K2,

        charged at the FULL Yb quantum because the erbium ends the round trip in the same level
        it started in (4I13/2 -> 4F9/2 -> 4I13/2): nothing of that excitation reaches the field
        and nothing of it is stored. The identity becomes

            q_opt(z) = dU/dt(z) + D_loss + D_Er + D_Yb + D_tr + D_K2,

        and `dissipation` is the sum of those five. Pair-induced quenching needs no term of its
        own: dark erbium is already in `D_loss` through the unbleachable absorption `_coeffs`
        adds, and so is the photodarkening gray loss, both of which are pure heat."""
        pl = self._plan()
        c = self._coeffs(pl)
        P = np.maximum(np.asarray(power_W, float), 0.0)
        f2 = np.asarray(f2, float)
        b2 = np.asarray(b2, float)
        if P.ndim != 2 or f2.shape != (P.shape[1],) or b2.shape != (P.shape[1],):
            raise ValueError("energy_terms: power_W must be (K, M) with f2, b2 of shape (M,); "
                             "got %r, %r, %r" % (P.shape, f2.shape, b2.shape))
        if self._two_pop and b2_uncoupled is None:
            raise ValueError(
                "energy_terms: this amplifier has an UNCOUPLED ytterbium pool "
                "(yb_coupled_fraction=%g, k_tr2=%g, migration=%g), so the uncoupled inversion "
                "must be passed as the 4th argument -- solve() reports it on "
                "meta['beta_yb_uncoupled_z'] and the march on meta['beta_yb_uncoupled']. "
                "Defaulting it would mis-state both the stored energy and the Yb fluorescence."
                % (self._fc, self._k_tr2, self._w_mig))
        if b2_uncoupled is not None and not self._two_pop:
            raise ValueError("energy_terms: this amplifier has ONE ytterbium pool, so "
                             "b2_uncoupled must be None (got an array of shape %r)"
                             % (np.shape(b2_uncoupled),))
        b2nc = None if b2_uncoupled is None else np.asarray(b2_uncoupled, float)
        if b2nc is not None and b2nc.shape != (P.shape[1],):
            raise ValueError("energy_terms: b2_uncoupled must have shape (M,); got %r"
                             % (b2nc.shape,))
        mcc = None
        if getattr(self, "_Tz", None) is not None:
            if z_m is None:
                raise ValueError("energy_terms: a temperature profile is set, so z_m (the mesh "
                                 "the powers live on) is required to evaluate the per-z McCumber "
                                 "factors -- pass result.z_m")
            zz = np.asarray(z_m, float)
            if zz.shape != (P.shape[1],):
                raise ValueError("energy_terms: z_m must have shape (M,); got %r" % (zz.shape,))
            mcc = self._mcc_matrices(pl, zz)
        A = self.fiber.a_dope_m2
        N_Er, N_Yb = self._n_er, self._n_yb
        eps_er = float(self.er_ion.eps_J)
        eps_yb = float(self.yb_ion.eps_J)
        inv_h = (1.0 / (H_PLANCK * (C_LIGHT / pl["lam"])))[:, None]
        bbar = self._bbar(b2, b2nc)
        one_f, one_b = (1.0 - f2)[None, :], (1.0 - bbar)[None, :]
        st = None if mcc is None else mcc[2]
        ge_er = c["g_e_er"][:, None] if mcc is None else c["g_e_er"][:, None] * mcc[0]
        ge_yb = c["g_e_yb"][:, None] if mcc is None else c["g_e_yb"][:, None] * mcc[1]
        se_pref = c["s_er"][:, None] if mcc is None else c["s_er"][:, None] * mcc[0]
        sy_pref = c["s_yb"][:, None] if mcc is None else c["s_yb"][:, None] * mcc[1]
        ga_yb = c["g_a_yb"][:, None]
        if st is not None:
            ge_yb, sy_pref, ga_yb = ge_yb * st, sy_pref * st, ga_yb * st

        pw_a_er = c["g_a_er"][:, None] * one_f * P
        pw_e_er = ge_er * f2[None, :] * P
        pw_sp_er = se_pref * f2[None, :]
        pw_a_yb = ga_yb * one_b * P
        pw_e_yb = ge_yb * bbar[None, :] * P
        pw_sp_yb = sy_pref * bbar[None, :]
        loss_W = np.sum(c["loss"][:, None] * P, axis=0)

        p_a_er, p_e_er, p_sp_er = pw_a_er.sum(0), pw_e_er.sum(0), pw_sp_er.sum(0)
        p_a_yb, p_e_yb, p_sp_yb = pw_a_yb.sum(0), pw_e_yb.sum(0), pw_sp_yb.sum(0)
        n_a_er, n_e_er = (pw_a_er * inv_h).sum(0), (pw_e_er * inv_h).sum(0)
        n_a_yb, n_e_yb = (pw_a_yb * inv_h).sum(0), (pw_e_yb * inv_h).sum(0)

        q_opt = (p_a_er + p_a_yb + loss_W) - (p_e_er + p_e_yb + p_sp_er + p_sp_yb)

        rs_m = None if mcc is None else mcc[3]
        ra_er, re_er, ra_yb, re_yb = self._rates_profile(c, P, mcc)
        df, db, dbn = self._fb_rhs3(ra_er, re_er, ra_yb, re_yb, f2, b2,
                                    b2 if b2nc is None else b2nc, rs_m)
        dbbar = db if b2nc is None else self._fc * db + (1.0 - self._fc) * dbn
        phi = self._phi(b2)
        k_tr, k_tr2 = self._k_tr, self._k_tr2
        if rs_m is not None:
            k_tr, k_tr2 = k_tr * rs_m[0], k_tr2 * rs_m[1]
        n_yb_c = self._fc * N_Yb                     # the COUPLED Yb density (= N_Yb at f = 1)
        tr_rate = A * k_tr * phi * N_Er * n_yb_c * b2 * (1.0 - f2)
        k2_rate = A * k_tr2 * N_Er * f2 * n_yb_c * b2
        dec_er = A * N_Er * f2 / self._tau_er
        dec_yb = A * N_Yb * bbar / self._tau_yb
        up_er = A * N_Er * self.upconversion_C_up * N_Er * f2 * f2

        d_er = ((p_a_er - p_e_er) - eps_er * (n_a_er - n_e_er)
                + (eps_er * dec_er - p_sp_er) + eps_er * up_er)
        d_yb = ((p_a_yb - p_e_yb) - eps_yb * (n_a_yb - n_e_yb)
                + (eps_yb * dec_yb - p_sp_yb))
        d_tr = (eps_yb - eps_er) * tr_rate
        d_k2 = eps_yb * k2_rate
        out = {"q_optical": q_opt,
               "stored_J_per_m": A * (eps_er * N_Er * f2 + eps_yb * N_Yb * bbar),
               "d_stored_dt": A * (eps_er * N_Er * df + eps_yb * N_Yb * dbbar),
               "background_loss": loss_W, "er_dissipation": d_er, "yb_dissipation": d_yb,
               "transfer_defect": d_tr, "k2_defect": d_k2,
               "dissipation": loss_W + d_er + d_yb + d_tr + d_k2,
               "transfer_rate_per_m": tr_rate, "k2_rate_per_m": k2_rate,
               "df2_dt": df, "db2_dt": db}
        if b2nc is not None:
            out["db2nc_dt"] = dbn
            out["beta_yb_mean"] = bbar
        return out

    def stored_energy_J(self, f2, b2, z, b2_uncoupled=None) -> float:
        """Total excitation energy stored in the reservoirs [J]:
        INT A_dope (eps_Er N_Er f2(z) + eps_Yb N_Yb bbar(z)) dz, bbar = f b2c + (1-f) b2nc. This
        is the `stored` that a transient energy balance differentiates -- see energy_terms for
        eps_* and the identity. With two pools BOTH count: an uncoupled ytterbium ion stores the
        same 2.04e-19 J as a coupled one, it simply cannot hand it to the erbium. Pass the
        uncoupled inversion as `b2_uncoupled` (None, and `b2` is the single pool's inversion)."""
        A = self.fiber.a_dope_m2
        bb = b2 if b2_uncoupled is None else self._bbar(np.asarray(b2, float),
                                                        np.asarray(b2_uncoupled, float))
        u = A * (float(self.er_ion.eps_J) * self._n_er * np.asarray(f2, float)
                 + float(self.yb_ion.eps_J) * self._n_yb * np.asarray(bb, float))
        return float(trapz(u, np.asarray(z, float)))
