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

FAST-4I11/2 LIMIT (the DEFAULT; standard for phosphosilicate, A_32 >> k_back n_Yb1 -- set
`tau32_s` to carry n3 explicitly instead, see EXPLICIT Er 4I11/2 below). n3 is adiabatically
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

EXPLICIT Er 4I11/2 (opt-in, `tau32_s`; DEFAULT None = the adiabatic fast limit above, bit for
bit). Setting tau32_s carries the 4I11/2 population n3 as a THIRD erbium level instead of
eliminating it, which is what the measured 4I11/2 lifetimes ask for: 7 us (Sefler, Mack, Valley,
Rose, JOSA B 21(10), 1740 (2004)), "a few us" and 0.1-10 us (Canat, PhD thesis SUPAERO (2006)),
10 us (de Varona Ortega, Hannover thesis (2019) Table 5.1), against the 1 ns MODELLING CONVENTION
Dong et al., Opt. Express 28(11), 16244 (2020) Table 1 adopt precisely to make back-transfer
negligible. With f3 = n3/N_Er and N_Er = n1 + n2 + n3 the z-local balances become (per ion, 1/s)

    Er2: W12 f1 - W21 f2 - f2/tau_Er + f3/tau32 - c_up C_up N_Er f2^2                      = 0
    Er3: W13 f1 - W31 f3 - f3/tau32 + k_tr b2c (f N_Yb) f1 - k_back f3 (f N_Yb)(1 - b2c)
         + r_up C_up N_Er f2^2                                                             = 0
    Ybc: R_a_Yb (1 - b2c) - R_e_Yb b2c - b2c/tau_Yb - k_tr b2c N_Er f1 - K2 b2c N_Er f2
         + k_back N_Er f3 (1 - b2c) - W_mig (1 - f)(b2c - b2nc)                            = 0
    Ybnc: unchanged,          f1 = 1 - f2 - f3.

W12/W21 and W13/W31 are the SAME per-ion absorption/emission rate sums the adiabatic model calls
R_a_Er/R_e_Er, split over the channels by which Er level the photon terminates on: a channel is a
LEVEL-3 channel when h nu_k is closer to the 4I11/2 zero line (`er_4i11_2_zero_line_m`, default
977 nm, the pump-band peak) than to the 4I13/2 one (RareEarthIon.eps_J), i.e. below 1192 nm for
the shipped ions. W12 + W13 = R_a_Er and W21 + W31 = R_e_Er identically, so nothing is created or
lost by the split, and no channel an EYDFA carries sits within 140 nm of the crossover (pumps
900-1050 nm and a 1000-1100 nm Yb band are level-3; the 1480 nm in-band pump and the C band are
level-2, which is the physically right assignment).

Three consequences, all stated rather than buried. (i) The PROPAGATION sees Er absorption from
n1 ALONE -- the ground fraction becomes 1 - f2 - f3, not 1 - f2 -- and no C-band gain from n3
(the 4I11/2 -> 4I13/2 mid-infrared transition and 4I11/2 excited-state absorption are NOT
modelled); at a level-3 channel the Er emission and the guided spontaneous source are charged to
n3 rather than to n2, which is what makes the energy closure exact. For every Er spectrum this
repo ships that emission is ~5e-33 m^2 at 976 nm (eight orders below sigma_a there, because
sigma_e is McCumber-derived about the 1530 nm zero line), so W31 and the level-3 emission are
numerically zero; the machinery is there for a tabulated ion that carries a real 980 nm emission
band. (ii) The k_back BACK-TRANSFER is now EXPLICIT (k_back n3 n5c, returning the excitation to
the coupled Yb pool) and the adiabatic phi refinement is switched off -- phi is exactly 1 on this
path, because the branching it approximated is resolved. (iii) COOPERATIVE UPCONVERSION can be
routed through 4I11/2 as Dong 2020 Eq. (2) writes it (`upconversion_via_4i11_2=True`, default
False): -2 C_up n2^2 in the n2 balance and +C_up n2^2 in the n3 balance, i.e. (c_up, r_up) =
(2, 1) instead of (1, 0). Both routings collapse to the SAME -C_up N_Er f2^2 in the adiabatic
limit, which is this module's existing convention, so the default path is untouched either way.

THE REDUCTION IS STILL ONE BRACKETED SCALAR, through a quadratic. At fixed f2 the two ytterbium
balances stay linear in (b2c, b2nc) -- the back-transfer source k_back N_Er f3 (1 - b2c) is
linear in b2c -- so the closed form of the two-population case survives with D_c and the coupled
right-hand side picking up f3 terms, and BOTH are LINEAR IN f3. The 4I11/2 balance multiplied by
that determinant is therefore a QUADRATIC in f3 whose coefficients are closed forms, with
Q(0) >= 0 and Q(1 - f2) <= 0, so f3(f2) is the one root in [0, 1 - f2] by the stable quadratic
formula -- no inner iteration. Substituting it into the 4I13/2 residual leaves the same scalar
H(f2) on [0, 1] with H(0) >= 0 and H(1) < 0, solved by the same safeguarded Newton/bisection;
dH/df2 carries df3/df2 from implicit differentiation of the quadratic. See `_n3_closed_form`.

YTTERBIUM EMISSION SCALE (`yb_sigma_e_scale`, default 1.0 = unchanged, bit for bit). ONE scalar
multiplying the Yb emission cross-section at EVERY channel -- the pump line and the 1-um band
alike -- and nothing else: sigma_a_Yb is untouched. Morasse, Cliche, Babin et al. (2006) measured
every parameter of a CorActive Er:Yb fiber and still over-predicted the backward 1-um ASE by
1000x (19.5 mW against 0.079 mW measured) until the McCumber-DERIVED Yb emission cross-section
was scaled by 0.4, with the signal output insensitive to the change. The scale exists so that
correction is a stated model parameter rather than an edit to an ion. It is applied to sigma_e
ONLY, deliberately: the quantity that pins sigma_a is the fiber's MEASURED pump absorption in
dB/m, which is not in question, whereas the 1-um over-prediction is an emission statement. The
price is that the scaled spectra no longer satisfy McCumber -- the inversion clamp at the pump
line moves from sigma_a/(sigma_a + sigma_e) to sigma_a/(sigma_a + s sigma_e), which at s = 0.4 on
the Melkumov phosphosilicate table is 0.461 -> 0.639 -- and that is the reason the scale is a
calibration knob with a measured target behind it and not a default.
`eryb_fit_yb_sigma_e_scale(amp, measured_1um_ase_W, direction)` fits it to a measured 1-um ASE.

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
docs/audit/2026-09-14-eryb-two-population-physics.md.
Explicit-4I11/2 and Yb-emission-scale references: Dong et al., Opt. Express 28(11), 16244 (2020)
Eqs. (1)-(2) and Table 1 [the explicit n3 level scheme, tau_32 = 1 ns as a convention, C_up routed
through 4I11/2, C36 = C63]; Sefler et al., JOSA B 21(10), 1740 (2004) [tau_32 = 7 us];
de Varona Ortega, PhD thesis, Leibniz Universitaet Hannover (2019) Table 5.1 [explicit rho_3,
tau_32 = 10 us]; Canat, PhD thesis SUPAERO (2006) [4I11/2 "a few us", 0.1-10 us]; Morasse,
Cliche, Babin, IEEE/vendor CorActive cut-back (2006) [the 0.4x McCumber Yb sigma_e needed to
match a measured 0.079 mW backward 1-um ASE against a 19.5 mW prediction]. Full audit trail:
docs/audit/2026-09-15-eryb-4i11-2-and-yb-sigma-scale.md. Pure
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

__all__ = ["ErYbAmplifier", "YbSigmaEScaleFit", "eryb_fit_yb_sigma_e_scale"]


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
    tau32_s : float, optional
        tau_32 [s], the Er 4I11/2 -> 4I13/2 relaxation time. DEFAULT None = the adiabatic fast
        limit (the level is eliminated), bit-identical to every previous release. A float carries
        n3 as an explicit third erbium level (module docstring): N_Er = n1 + n2 + n3, the
        propagation absorbs from n1 only, the back-transfer k_back n3 n5c becomes explicit and
        REPLACES the phi refinement, and the march gains a fourth reservoir. Measured values:
        7 us (Sefler 2004), 10 us (de Varona 2019), 0.1-10 us (Canat 2006); 1 ns is Dong 2020's
        modelling convention for "adiabatic". The adiabatic default is the right model for a
        high-phosphorus host; set this when the 4I11/2 bottleneck or back-transfer is in question.
    upconversion_via_4i11_2 : bool
        Route cooperative upconversion through 4I11/2 as Dong 2020 Eq. (2) does -- -2 C_up n2^2
        out of 4I13/2 and +C_up n2^2 into 4I11/2 -- instead of the single -C_up N_Er f2^2 this
        module has always used (default False, unchanged). Both collapse to the same net loss in
        the adiabatic limit, so it is REFUSED unless tau32_s is set, rather than silently
        accepted as a no-op.
    er_4i11_2_zero_line_m : float
        The Er 4I11/2 zero line [m] (default 977e-9, the pump-band absorption peak). Used ONLY
        when tau32_s is set: it sets the level-2/level-3 channel split (the midpoint in PHOTON
        ENERGY between this line and the 4I13/2 one, 1192 nm for the shipped ions) and the
        4I11/2 stored energy, hence how the total transfer heat divides between the transfer
        defect and the 4I11/2 -> 4I13/2 relaxation. Their SUM does not depend on it.
    yb_sigma_e_scale : float
        ONE scalar multiplying the Yb EMISSION cross-section at every channel -- pump line and
        1-um band alike -- with sigma_a_Yb untouched (default 1.0, bit-identical). Morasse 2006
        needed 0.4 on a McCumber-derived Yb sigma_e to match a measured 1-um ASE 1000x below his
        prediction, with the signal output insensitive. See the module docstring for why the
        scale is on sigma_e only and what it costs (the spectra stop satisfying McCumber, so the
        pump-line inversion clamp moves), and `eryb_fit_yb_sigma_e_scale` to fit it to a measured
        1-um ASE.
    concentration : ConcentrationModel, optional
        Er pair-induced quenching (either convention), C_up, and the Yb photodarkening
        equilibrium gray loss -- the SAME object and the same semantics FiberAmplifier takes
        (concentration.py). None (default) -> the ideal model, bit-identical.
    """

    def __init__(self, er_ion: RareEarthIon, yb_ion: RareEarthIon, fiber: FiberSpec,
                 pumps: List[Pump], signals: List[Signal], ase: Optional[AseBand] = None, *,
                 n_yb_m3: float, k_tr_m3_s: float = 2.0e-22, k_back_m3_s: float = 0.0,
                 a32_per_s: float = 5.0e5, yb_ase: Optional[AseBand] = None,
                 upconversion_C_up: float = 0.0, yb_coupled_fraction: float = 1.0,
                 k_tr2_m3_s: float = 0.0, yb_migration_rate_per_s: float = 0.0,
                 tau32_s: Optional[float] = None, upconversion_via_4i11_2: bool = False,
                 er_4i11_2_zero_line_m: float = 977.0e-9, yb_sigma_e_scale: float = 1.0,
                 concentration=None):
        if not (n_yb_m3 > 0.0):
            raise ValueError("ErYbAmplifier: n_yb_m3 (N_Yb) must be > 0")
        if not (a32_per_s > 0.0):
            raise ValueError("ErYbAmplifier: a32_per_s must be > 0")
        if tau32_s is not None and not (float(tau32_s) > 0.0):
            raise ValueError("ErYbAmplifier: tau32_s must be > 0 (or None for the adiabatic "
                             "fast-4I11/2 limit); got %r" % (tau32_s,))
        if bool(upconversion_via_4i11_2) and tau32_s is None:
            raise ValueError(
                "ErYbAmplifier: upconversion_via_4i11_2=True needs an EXPLICIT 4I11/2 level "
                "(tau32_s=...). In the adiabatic limit the two routings are the same model -- "
                "Dong 2020's -2 C_up n2^2 out of 4I13/2 with +C_up n2^2 into 4I11/2 relaxes "
                "straight back and leaves the net -C_up N_Er f2^2 this module already uses -- so "
                "accepting the flag here would advertise a change that is not there.")
        if not (float(er_4i11_2_zero_line_m) > 0.0):
            raise ValueError("ErYbAmplifier: er_4i11_2_zero_line_m must be > 0")
        if not (float(yb_sigma_e_scale) > 0.0):
            raise ValueError("ErYbAmplifier: yb_sigma_e_scale must be > 0; got %r"
                             % (yb_sigma_e_scale,))
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
        # THE second routing predicate, exactly parallel to _two_pop: the explicit 4I11/2 level.
        # False -> every expression below takes the adiabatic path verbatim.
        self._tau32 = None if tau32_s is None else float(tau32_s)
        self._n3 = self._tau32 is not None
        self._cup_via_n3 = bool(upconversion_via_4i11_2)
        self._lam3 = float(er_4i11_2_zero_line_m)
        self._eps3 = H_PLANCK * C_LIGHT / self._lam3        # 4I11/2 stored quantum [J]
        self._yb_se_scale = float(yb_sigma_e_scale)
        self._Tz = None                     # optional axial T profile (set_temperature_profile)
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
               signals: Optional[List[Signal]] = None, ase=_KEEP, yb_ase=_KEEP,
               yb_sigma_e_scale=_KEEP) -> "ErYbAmplifier":
        """Clone through THIS class's own constructor, carrying EVERY opt-in: both ions'
        spectroscopy, the fiber, both ASE bands, the Yb density, the forward/back transfer
        coefficients, A_32, the Er upconversion coefficient, the two-population parameters
        (coupled fraction, secondary transfer, migration), the explicit-4I11/2 parameters
        (tau_32, the upconversion routing, the 4I11/2 zero line), the Yb emission scale, the
        ConcentrationModel and the axial temperature profile. `yb_sigma_e_scale` is the one
        parameter this method can OVERRIDE rather than only carry -- that is what
        `with_yb_sigma_e_scale` and the calibration fit re-seed. The single place that lists what
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
                            tau32_s=self._tau32,
                            upconversion_via_4i11_2=self._cup_via_n3,
                            er_4i11_2_zero_line_m=self._lam3,
                            yb_sigma_e_scale=(self._yb_se_scale
                                              if yb_sigma_e_scale is _KEEP
                                              else float(yb_sigma_e_scale)),
                            concentration=self.concentration)
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

    def with_yb_sigma_e_scale(self, scale: float) -> "ErYbAmplifier":
        """A copy whose Yb EMISSION cross-section is scaled by `scale` at every channel, every
        other opt-in preserved (see _clone). This is what `eryb_fit_yb_sigma_e_scale` re-seeds
        along its bracket, and the supported way to apply a calibration result to an amplifier
        that already exists."""
        return self._clone(yb_sigma_e_scale=float(scale))

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
        # THE one place the Yb emission scale is applied: every downstream coefficient (the modal
        # gain, the stimulated-emission rate, the ASE spontaneous prefactor, the McCumber
        # products, the parasitic diagnostic's own lookup) is built from this array, so one
        # multiplication scales the whole model consistently. The scale == 1.0 branch performs NO
        # arithmetic on the array, which is what makes the default bit-identical rather than
        # merely equal (the _solve_once relax == 1.0 idiom).
        se_yb = ch_yb.sigma_e
        if self._yb_se_scale != 1.0:
            se_yb = se_yb * self._yb_se_scale
        return {"lam": lam, "u": u, "is_ase": is_ase, "dnu": dnu, "m": m, "kind": kind,
                "bc": np.asarray(bc), "gamma": gamma, "loss": ch_er.loss_per_m.copy(),
                "sa_er": ch_er.sigma_a, "se_er": ch_er.sigma_e,
                "sa_yb": ch_yb.sigma_a, "se_yb": se_yb}

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
        if self._n3:
            # LEVEL-2 / LEVEL-3 channel split for the explicit 4I11/2 (module docstring). A
            # channel terminates on 4I11/2 when its photon energy is closer to THAT level's zero
            # line than to the 4I13/2 one -- an energy criterion evaluated from the two zero
            # lines, not a hand-assigned band edge. The masked copies are built ONCE per solve
            # here so the hot loops multiply instead of branching; they exist only on this path.
            w3 = np.asarray(H_PLANCK * nu > self._level_split_J(), float)
            w2 = 1.0 - w3
            c["w3"] = w3
            c["flux_a_er3"] = c["flux_a_er"] * w3       # W13, the direct 4I11/2 pump absorption
            c["flux_e_er3"] = c["flux_e_er"] * w3       # W31, its stimulated emission
            c["g_e_er2"] = c["g_e_er"] * w2             # modal gain charged to n2 / to n3
            c["g_e_er3"] = c["g_e_er"] * w3
            c["s_er2"] = s_er * w2                      # spontaneous source, likewise
            c["s_er3"] = s_er * w3
        return c

    def _level_split_J(self) -> float:
        """The photon energy [J] above which a channel is a LEVEL-3 (4I11/2) channel: the
        midpoint between the two Er zero-line quanta, eps_3 = h c / er_4i11_2_zero_line_m and
        eps_Er = RareEarthIon.eps_J. For the shipped ions that is 1.666e-19 J, i.e. 1192 nm --
        so a 900-1100 nm pump or Yb ASE bin is level-3 and a 1480 nm in-band pump or a C-band bin
        is level-2, with no channel within 140 nm of the boundary."""
        return 0.5 * (float(self.er_ion.eps_J) + self._eps3)

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
        to BOTH ions exactly for a uniform profile (gated). sigma_a and the two lifetimes are
        held (their T-dependence is second order at these Delta-T; spectroscopy docstring), and
        so are k_tr, K2 and W_mig -- their measured temperature dependence (Cheng 2022: the Yb
        transfer component moves 8.20 -> 7.56 us over 300-480 K, ~8%) is NOT modelled here.

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
        """((K, M), (K, M)) McCumber sigma_e scale factors for the Er and the Yb spectra on the
        mesh z, or None when no profile is set (in which case every caller takes the untouched
        no-temperature branch)."""
        if getattr(self, "_Tz", None) is None:
            return None
        zt, Tt, T_ref = self._Tz
        T = np.interp(np.asarray(z, float), zt, Tt)
        nu = C_LIGHT / pl["lam"]
        dinv = 1.0 / (KB * T) - 1.0 / (KB * T_ref)
        return (np.exp(np.outer(float(self.er_ion.eps_J) - H_PLANCK * nu, dinv)),
                np.exp(np.outer(float(self.yb_ion.eps_J) - H_PLANCK * nu, dinv)))

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

    def _solve_fb(self, Ra_Er: float, Re_Er: float, Ra_Yb: float, Re_Yb: float
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

    def _solve_fbb(self, Ra_Er: float, Re_Er: float, Ra_Yb: float, Re_Yb: float
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

    # ---- explicit 4I11/2: the closed forms and the z-local solve ---------------------------
    def _n3_coeffs(self):
        """The rate-independent coefficients of the explicit-4I11/2 balances, hoisted once:

            kin   = k_tr f N_Yb     transfer INTO 4I11/2, per unit b2c per unit ground fraction
            kout  = k_tr N_Er       the matching drain on the COUPLED Yb, per unit f1
            kb_er = k_back f N_Yb   back-transfer drain on n3, per unit (1 - b2c)
            kb_yb = k_back N_Er     the matching source on the coupled Yb, per unit f3
            K2n   = K2 N_Er,   cu = C_up N_Er,   A3 = 1/tau_32

        plus the upconversion routing pair (c_up, r_up) = (2, 1) when the upconversion is routed
        through 4I11/2 (Dong 2020 Eq. 2) and (1, 0) otherwise -- the two collapse to the same net
        -C_up N_Er f2^2 once n3 relaxes, which is why the adiabatic model needs only one of them.
        """
        return (self._k_tr * self._fc * self._n_yb, self._k_tr * self._n_er,
                self._k_back * self._fc * self._n_yb, self._k_back * self._n_er,
                self._k_tr2 * self._n_er, self.upconversion_C_up * self._n_er,
                1.0 / self._tau32,
                (2.0, 1.0) if self._cup_via_n3 else (1.0, 0.0))

    def _n3_closed_form(self, W13, W31, Ra_Yb, Re_Yb, f2, derivative=False):
        """(f3, b2c, b2nc, df3/df2) at a FIXED f2 -- the closed-form inner solve of the explicit
        4I11/2 model, with no iteration anywhere. Works elementwise, so it serves both the scalar
        z-local root find and the vectorized transient seed.

        WHY IT IS CLOSED. At fixed (f2, f3) the two ytterbium balances are still LINEAR in
        (b2c, b2nc): the new back-transfer source k_back N_Er f3 (1 - b2c) contributes a constant
        `kb_yb f3` to the coupled right-hand side and a drain `kb_yb f3` to its diagonal. So the
        two-population closed form survives with

            D   = R_a_Yb + R_e_Yb + 1/tau_Yb
            Dc  = D + kout (1 - f2 - f3) + K2n f2 + kb_yb f3   = P0 + P1 f3
            Sc  = R_a_Yb + kb_yb f3                            (the coupled RHS)
            det = (D + W f) Dc + W (1-f) D                     = E0 + E1 f3   > 0 always
            b2c = [Sc (D + W f) + W (1-f) R_a_Yb] / det        = (C0 + C1 f3) / det
            b2nc= [(Dc + W (1-f)) R_a_Yb + W f Sc] / det

        BOTH numerator and denominator are LINEAR in f3, so multiplying the 4I11/2 balance

            G(f3) = W13 f1 - (W31 + A3) f3 + r_up cu f2^2 + kin b2c f1 - kb_er (1 - b2c) f3

        by det (> 0 on the whole interval, so no root is created or destroyed) gives a QUADRATIC

            Q(f3) = a2 f3^2 + a1 f3 + a0,
            a0 = S E0 + kin g0 C0,   a1 = S E1 - T E0 + kin (g0 C1 - C0) - kb_er (E0 - C0),
            a2 = -T E1 - kin C1 - kb_er (E1 - C1),
            S  = W13 g0 + r_up cu f2^2,   T = W13 + W31 + A3,   g0 = 1 - f2.

        Q(0) = a0 >= 0 (a sum of non-negative products) and Q(g0) <= 0 whenever the level is
        actually pumped from below, so exactly ONE root lies in [0, g0] -- a quadratic cannot
        return to its starting sign inside an interval whose endpoints straddle zero. It is taken
        with the cancellation-free pair q = -(a1 + sign(a1) sqrt(disc))/2, roots q/a2 and a0/q,
        which also degenerates correctly to the linear root -a0/a1 as a2 -> 0 (a0/q -> -a0/a1) --
        so the stiff limit tau_32 -> 0, where a2 and a1 both blow up like A3, loses no digits.
        The result is clamped to [0, max(g0, 0)]: f2 + f3 <= 1 is the population constraint, and
        the clamp is what keeps H(1) < 0 for the OUTER bracket when a routed upconversion would
        otherwise feed n3 at f2 = 1.

        `derivative=True` also returns df3/df2 by implicit differentiation of Q (dQ/df2 divided
        by -dQ/df3, with da2/df2 = 0 since a2 carries no f2 at all); it is 0 where the root is
        clamped or dQ/df3 underflows, which only costs the outer solve a bisection step."""
        kin, kout, kb_er, kb_yb, K2n, cu, A3, (_c_up, r_up) = self._n3_coeffs()
        fc, Wm = self._fc, self._w_mig
        D = Ra_Yb + Re_Yb + 1.0 / self._tau_yb
        DW = D + Wm * fc
        g0 = 1.0 - f2
        P0 = D + kout * g0 + K2n * f2
        P1 = kb_yb - kout
        E0 = DW * P0 + Wm * (1.0 - fc) * D
        E1 = DW * P1
        C0 = Ra_Yb * (D + Wm)
        C1 = kb_yb * DW
        S = W13 * g0 + r_up * cu * f2 * f2
        T = W13 + W31 + A3
        a0 = S * E0 + kin * g0 * C0
        a1 = S * E1 - T * E0 + kin * (g0 * C1 - C0) - kb_er * (E0 - C0)
        a2 = -T * E1 - kin * C1 - kb_er * (E1 - C1)
        disc = np.maximum(a1 * a1 - 4.0 * a2 * a0, 0.0)
        sq = np.sqrt(disc)
        q = -0.5 * (a1 + np.where(a1 >= 0.0, 1.0, -1.0) * sq)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_a = np.where(a2 != 0.0, q / np.where(a2 != 0.0, a2, 1.0), np.inf)
            r_b = np.where(q != 0.0, a0 / np.where(q != 0.0, q, 1.0), 0.0)
        hi = np.maximum(g0, 0.0)
        pick = (r_a >= 0.0) & (r_a <= hi)
        f3 = np.where(pick, r_a, r_b)
        f3 = np.where(np.isfinite(f3), f3, 0.0)
        f3c = np.clip(f3, 0.0, hi)
        Dc = P0 + P1 * f3c
        det = E0 + E1 * f3c
        Sc = Ra_Yb + kb_yb * f3c
        b2c = (Sc * DW + Wm * (1.0 - fc) * Ra_Yb) / det
        b2nc = ((Dc + Wm * (1.0 - fc)) * Ra_Yb + Wm * fc * Sc) / det
        if not derivative:
            return f3c, b2c, b2nc, 0.0
        dS = -W13 + 2.0 * r_up * cu * f2
        dE0 = DW * (K2n - kout)
        da0 = dS * E0 + S * dE0 - kin * C0
        da1 = dS * E1 - T * dE0 - kin * C1 - kb_er * dE0
        dQ_df3 = 2.0 * a2 * f3c + a1
        dQ_df2 = da0 + da1 * f3c
        with np.errstate(divide="ignore", invalid="ignore"):
            df3 = np.where(dQ_df3 != 0.0, -dQ_df2 / np.where(dQ_df3 != 0.0, dQ_df3, 1.0), 0.0)
        df3 = np.where(np.isfinite(df3) & (f3 > 0.0) & (f3 < hi), df3, 0.0)
        return f3c, b2c, b2nc, df3

    def _solve_f3fb(self, W12, W21, W13, W31, Ra_Yb, Re_Yb):
        """Steady-state (f2, f3, b2c, b2nc) at one z with an EXPLICIT 4I11/2 -- the four-unknown
        counterpart of `_solve_fb` / `_solve_fbb`, entered only when `tau32_s` is set.

        The reduction is the same one, one level deeper: `_n3_closed_form` gives (f3, b2c, b2nc)
        in closed form at a fixed f2, so what is left is again ONE scalar residual on [0, 1],

            H(f2) = W12 (1 - f2 - f3) - W21 f2 - f2/tau_Er + f3/tau_32 - c_up C_up N_Er f2^2,

        with H(0) >= 0 (absorption and the relaxation feed, both non-negative) and H(1) < 0 (at
        f2 = 1 the clamp forces f3 = 0 and only -W21 - 1/tau_Er - c_up C_up N_Er is left), solved
        by the SAME `_bracketed_newton`. Its slope carries df3/df2 from the quadratic:

            dH/df2 = -W12 (1 + df3) - W21 - 1/tau_Er + df3/tau_32 - 2 c_up C_up N_Er f2."""
        _kin, _kout, _kb_er, _kb_yb, _K2n, cu, A3, (c_up, _r_up) = self._n3_coeffs()
        inv_tE = 1.0 / self._tau_er

        def _HdH(f):
            f3, _bc, _bn, df3 = self._n3_closed_form(W13, W31, Ra_Yb, Re_Yb, f, derivative=True)
            H = (W12 * (1.0 - f - f3) - W21 * f - f * inv_tE + A3 * f3
                 - c_up * cu * f * f)
            dH = (-W12 * (1.0 + df3) - W21 - inv_tE + A3 * df3 - 2.0 * c_up * cu * f)
            return float(H), float(dH)

        f, _ = self._bracketed_newton(_HdH)
        f3, bc, bn, _d = self._n3_closed_form(W13, W31, Ra_Yb, Re_Yb, f)
        f = float(min(max(f, 0.0), 1.0))
        return (f, float(min(max(f3, 0.0), 1.0 - f)), float(min(max(bc, 0.0), 1.0)),
                float(min(max(bn, 0.0), 1.0)))

    def _fbb_n3(self, W12, W21, W13, W31, Ra_Yb, Re_Yb):
        """(f2, f3, b2c, b2nc, bbar) at one z with the explicit 4I11/2. b2nc is None for the
        one-pool model, where bbar IS b2c (the same object), exactly as `_fbb` does it."""
        f2, f3, b2c, b2nc = self._solve_f3fb(W12, W21, W13, W31, Ra_Yb, Re_Yb)
        if self._two_pop:
            return f2, f3, b2c, b2nc, self._fc * b2c + (1.0 - self._fc) * b2nc
        return f2, f3, b2c, None, b2c

    def _fbb(self, Ra_Er, Re_Er, Ra_Yb, Re_Yb):
        """(f2, b2c, b2nc, bbar) at one z, routed to the one- or two-pool algebra. bbar is the
        POPULATION-WEIGHTED Yb inversion f b2c + (1-f) b2nc -- the only Yb quantity the optical
        field, the parasitic-gain diagnostic and the photodarkening law ever see. b2nc is None
        for the one-pool model, where bbar IS b2 (the same object, so the caller is bitwise
        unaffected)."""
        if self._two_pop:
            f2, b2c, b2nc = self._solve_fbb(Ra_Er, Re_Er, Ra_Yb, Re_Yb)
            return f2, b2c, b2nc, self._fc * b2c + (1.0 - self._fc) * b2nc
        f2, b2 = self._solve_fb(Ra_Er, Re_Er, Ra_Yb, Re_Yb)
        return f2, b2, None, b2

    def _dP(self, c, u, P, mcc=None):
        """dP_k/dz [W/m] for every channel from the local power vector P (K,). `mcc` is the
        optional (mcc_er, mcc_yb) pair of (K,) per-channel McCumber sigma_e scale factors AT THIS
        z (set_temperature_profile); None -- the default and the only path a profile-free
        amplifier takes -- runs the untouched isothermal arithmetic."""
        P = np.maximum(P, 0.0)
        if mcc is None:
            Ra_Er = float(np.dot(c["flux_a_er"], P)); Re_Er = float(np.dot(c["flux_e_er"], P))
            Ra_Yb = float(np.dot(c["flux_a_yb"], P)); Re_Yb = float(np.dot(c["flux_e_yb"], P))
            ge_er, ge_yb, s_er, s_yb = c["g_e_er"], c["g_e_yb"], c["s_er"], c["s_yb"]
        else:
            m_er, m_yb = mcc
            Ra_Er = float(np.dot(c["flux_a_er"], P))
            Re_Er = float(np.dot(c["flux_e_er"] * m_er, P))
            Ra_Yb = float(np.dot(c["flux_a_yb"], P))
            Re_Yb = float(np.dot(c["flux_e_yb"] * m_yb, P))
            ge_er, ge_yb = c["g_e_er"] * m_er, c["g_e_yb"] * m_yb
            s_er, s_yb = c["s_er"] * m_er, c["s_yb"] * m_yb
        if self._n3:
            # EXPLICIT 4I11/2. The Er block of the operator changes in exactly two places: the
            # ground fraction that absorbs is 1 - f2 - f3 (n1 alone), and the emission and the
            # spontaneous source at a LEVEL-3 channel are charged to n3 instead of n2. Everything
            # ytterbium is untouched. W12 = R_a_Er - W13 and W21 = R_e_Er - W31 by construction,
            # so the split conserves the totals the adiabatic path uses.
            W13 = float(np.dot(c["flux_a_er3"], P))
            W31 = (float(np.dot(c["flux_e_er3"], P)) if mcc is None
                   else float(np.dot(c["flux_e_er3"] * mcc[0], P)))
            f2, f3, _b2c, _b2nc, b2 = self._fbb_n3(Ra_Er - W13, Re_Er - W31, W13, W31,
                                                   Ra_Yb, Re_Yb)
            ge2 = c["g_e_er2"] if mcc is None else c["g_e_er2"] * mcc[0]
            ge3 = c["g_e_er3"] if mcc is None else c["g_e_er3"] * mcc[0]
            s2 = c["s_er2"] if mcc is None else c["s_er2"] * mcc[0]
            s3 = c["s_er3"] if mcc is None else c["s_er3"] * mcc[0]
            g = (ge2 * f2 + ge3 * f3 - c["g_a_er"] * (1.0 - f2 - f3)
                 + ge_yb * b2 - c["g_a_yb"] * (1.0 - b2) - c["loss"])
            if self.concentration is not None:
                g = g - self.concentration.photodarkening_loss_per_m(b2)
            return u * (g * P + (s2 * f2 + s3 * f3 + s_yb * b2))
        f2, _b2c, _b2nc, b2 = self._fbb(Ra_Er, Re_Er, Ra_Yb, Re_Yb)
        g = (ge_er * f2 - c["g_a_er"] * (1.0 - f2)
             + ge_yb * b2 - c["g_a_yb"] * (1.0 - b2) - c["loss"])
        if self.concentration is not None:
            # Yb photodarkening equilibrium gray loss at the POPULATION-WEIGHTED Yb inversion --
            # both pools darken the glass, and an uncoupled pool sits at a HIGHER inversion than
            # a coupled one, so reading it off b2c alone would understate the loss.
            g = g - self.concentration.photodarkening_loss_per_m(b2)
        src = s_er * f2 + s_yb * b2
        return u * (g * P + src)

    def _fb_profile(self, c, P, mcc=None):
        """(f2(z), b2(z), b2nc(z), f3(z)) at each z given the full power profile P (K, M). b2 is
        the COUPLED Yb inversion and b2nc the uncoupled one; b2nc is None for the one-pool model,
        in which case b2 is simply the Yb inversion, and f3 (the Er 4I11/2 fraction) is None
        unless `tau32_s` is set. `mcc` is the optional ((K, M), (K, M)) McCumber scale pair on the
        SAME mesh."""
        M = P.shape[1]
        f2 = np.empty(M); b2 = np.empty(M)
        b2nc = np.empty(M) if self._two_pop else None
        f3 = np.empty(M) if self._n3 else None
        for j in range(M):
            Pj = np.maximum(P[:, j], 0.0)
            if mcc is None:
                Ra_Er = float(np.dot(c["flux_a_er"], Pj))
                Re_Er = float(np.dot(c["flux_e_er"], Pj))
                Ra_Yb = float(np.dot(c["flux_a_yb"], Pj))
                Re_Yb = float(np.dot(c["flux_e_yb"], Pj))
            else:
                Ra_Er = float(np.dot(c["flux_a_er"], Pj))
                Re_Er = float(np.dot(c["flux_e_er"] * mcc[0][:, j], Pj))
                Ra_Yb = float(np.dot(c["flux_a_yb"], Pj))
                Re_Yb = float(np.dot(c["flux_e_yb"] * mcc[1][:, j], Pj))
            if self._n3:
                W13 = float(np.dot(c["flux_a_er3"], Pj))
                W31 = (float(np.dot(c["flux_e_er3"], Pj)) if mcc is None
                       else float(np.dot(c["flux_e_er3"] * mcc[0][:, j], Pj)))
                f2[j], f3[j], bc, bn = self._solve_f3fb(Ra_Er - W13, Re_Er - W31, W13, W31,
                                                        Ra_Yb, Re_Yb)
                b2[j] = bc
                if b2nc is not None:
                    b2nc[j] = bn
            elif self._two_pop:
                f2[j], b2[j], b2nc[j] = self._solve_fbb(Ra_Er, Re_Er, Ra_Yb, Re_Yb)
            else:
                f2[j], b2[j] = self._solve_fb(Ra_Er, Re_Er, Ra_Yb, Re_Yb)
        return f2, b2, b2nc, f3

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
        m_er, m_yb = mcc
        return (c["flux_a_er"] @ Pz,
                np.sum(c["flux_e_er"][:, None] * m_er * Pz, axis=0),
                c["flux_a_yb"] @ Pz,
                np.sum(c["flux_e_yb"][:, None] * m_yb * Pz, axis=0))

    def _rates_profile_n3(self, c, P, mcc=None):
        """(W12, W21, W13, W31, R_a_Yb, R_e_Yb) [1/s] at every z for the EXPLICIT-4I11/2 model:
        `_rates_profile`'s four numbers with the two erbium ones split over the level-2 and
        level-3 channels (`_coeffs`' masks). The split is exact by construction -- W12 + W13 is
        the SAME float sum R_a_Er the adiabatic path uses, computed as a difference so no channel
        can be double-counted or dropped."""
        Ra_Er, Re_Er, Ra_Yb, Re_Yb = self._rates_profile(c, P, mcc)
        Pz = np.maximum(np.asarray(P, float), 0.0)
        W13 = c["flux_a_er3"] @ Pz
        if mcc is None:
            W31 = c["flux_e_er3"] @ Pz
        else:
            W31 = np.sum(c["flux_e_er3"][:, None] * mcc[0] * Pz, axis=0)
        return Ra_Er - W13, Re_Er - W31, W13, W31, Ra_Yb, Re_Yb

    def _phi(self, b2):
        """Back-transfer branching factor phi = A_32 / (A_32 + k_back (1 - b2) N_Yb): the fraction
        of Er 4I11/2 population that relaxes to 4I13/2 before back-transferring (module
        docstring). The back-transfer acceptor is a GROUND-STATE COUPLED Yb, n5c = f N_Yb
        (1 - b2c), hence the f factor -- which is exactly 1.0 in the one-pool model, so
        multiplying by it changes no bit there. EXACTLY 1.0 overall when k_back = 0 (the
        default), so every expression carrying phi stays byte-identical to the
        no-back-transfer algebra.

        EXACTLY 1.0 ALSO when tau32_s is set: phi is the adiabatic APPROXIMATION to a branching
        that the explicit 4I11/2 level resolves, and applying both would charge the back-transfer
        twice. That routing is asserted rather than assumed -- the explicit path never calls the
        phi-carrying expressions -- and this guard makes it harmless if one ever does."""
        b = np.asarray(b2, float)
        if self._k_back <= 0.0 or self._n3:
            return np.ones_like(b)
        return self._a32 / (self._a32 + self._k_back * self._fc * (1.0 - b) * self._n_yb)

    def _dphi_db(self, b2, phi):
        """d(phi)/d(b2c) = phi^2 k_back f N_Yb / A_32 (identically 0 when k_back = 0, and
        likewise when tau32_s is set -- see `_phi`)."""
        b = np.asarray(b2, float)
        if self._k_back <= 0.0 or self._n3:
            return np.zeros_like(b)
        return phi * phi * self._k_back * self._fc * self._n_yb / self._a32

    def _fb_rhs3(self, Ra_Er, Re_Er, Ra_Yb, Re_Yb, f2, b2c, b2nc):
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
        phi = self._phi(b2c)
        tr = phi * b2c * (1.0 - f2)           # transfer shape; x k_tr f N_Yb (in) / N_Er (out)
        mig = self._w_mig * (b2c - b2nc)      # detailed-balance exchange (module docstring)
        df = (Ra_Er * (1.0 - f2) - Re_Er * f2 - f2 / self._tau_er
              + self._k_tr * fc * self._n_yb * tr
              - self.upconversion_C_up * self._n_er * f2 * f2)
        dbc = (Ra_Yb * (1.0 - b2c) - Re_Yb * b2c - b2c / self._tau_yb
               - self._k_tr * self._n_er * tr
               - self._k_tr2 * self._n_er * f2 * b2c
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

    def _fb_jacobian3(self, Ra_Er, Re_Er, Ra_Yb, Re_Yb, f2, b2c, b2nc):
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
        phi = self._phi(b2c)
        dphi = self._dphi_db(b2c, phi)
        k_in = self._k_tr * fc * self._n_yb         # transfer-IN coefficient (Er equation)
        k_out = self._k_tr * self._n_er             # transfer-OUT coefficient (Yb equation)
        k2n = self._k_tr2 * self._n_er
        dtr_df = -phi * b2c
        dtr_db = (1.0 - f2) * (phi + b2c * dphi)
        zero = np.zeros_like(np.asarray(dtr_df, float))
        j00 = (-(Ra_Er + Re_Er + 1.0 / self._tau_er) + k_in * dtr_df
               - 2.0 * self.upconversion_C_up * self._n_er * f2)
        j01 = k_in * dtr_db
        j10 = -k_out * dtr_df - k2n * b2c
        j11 = (-(Ra_Yb + Re_Yb + 1.0 / self._tau_yb) - k_out * dtr_db - k2n * f2
               - self._w_mig * (1.0 - fc))
        j12 = self._w_mig * (1.0 - fc) + zero
        j21 = self._w_mig * fc + zero
        j22 = -(Ra_Yb + Re_Yb + 1.0 / self._tau_yb) - self._w_mig * fc
        return [[j00, j01, zero], [j10, j11, j12], [zero, j21, j22 + zero]]

    def _fb_rhs_n3(self, W12, W21, W13, W31, Ra_Yb, Re_Yb, f2, f3, b2c, b2nc):
        """(df2/dt, df3/dt, db2c/dt, db2nc/dt) [1/s] with an EXPLICIT 4I11/2 -- the FOUR balances
        of the module docstring before the steady-state condition is imposed, and the single home
        of that algebra (the march and `energy_terms` both read it rather than re-deriving).

        `_solve_f3fb` returns the state at which all four vanish, so a march built on this has
        the steady solve's own fixed point. Passing b2nc = b2c makes the migration term vanish
        identically and leaves the first THREE components the one-Yb-pool system, which is how
        the march treats a `tau32_s`-only amplifier (the 4th is then not a reservoir at all)."""
        kin, kout, kb_er, kb_yb, K2n, cu, A3, (c_up, r_up) = self._n3_coeffs()
        fc = self._fc
        f1 = 1.0 - f2 - f3
        up = cu * f2 * f2
        mig = self._w_mig * (b2c - b2nc)
        df2 = W12 * f1 - W21 * f2 - f2 / self._tau_er + A3 * f3 - c_up * up
        df3 = (W13 * f1 - W31 * f3 - A3 * f3 + kin * b2c * f1
               - kb_er * (1.0 - b2c) * f3 + r_up * up)
        dbc = (Ra_Yb * (1.0 - b2c) - Re_Yb * b2c - b2c / self._tau_yb
               - kout * f1 * b2c - K2n * f2 * b2c + kb_yb * f3 * (1.0 - b2c)
               - (1.0 - fc) * mig)
        dbn = (Ra_Yb * (1.0 - b2nc) - Re_Yb * b2nc - b2nc / self._tau_yb + fc * mig)
        return df2, df3, dbc, dbn

    def _fb_jacobian_n3(self, W12, W21, W13, W31, Ra_Yb, Re_Yb, f2, f3, b2c, b2nc):
        """The EXACT 4x4 Jacobian of `_fb_rhs_n3` at (f2, f3, b2c, b2nc), as a nested 4x4 list of
        arrays (row-major). This is what licenses the exponential Rosenbrock step once the
        4I11/2 level is explicit, and it MUST be the exact one: 1/tau_32 is 1e5-1e9 1/s against
        1/tau_Er = 1e2, so the 4I11/2 row is the stiffest in the system by up to seven orders and
        a frozen or approximated Jacobian there would either ring or crawl.

        STRUCTURE. J[0][3] = J[1][3] = J[3][0] = J[3][1] = 0 -- within a frozen-power step the
        uncoupled ytterbium meets the erbium only through the coupled pool, exactly as in the 3x3
        case. The diagonal is strictly negative: each entry is minus a sum of absorption,
        stimulated-emission, decay, relaxation, transfer, back-transfer, K2 and migration rates.
        The off-diagonal signs are NOT all positive here and no determinant theorem is claimed --
        J[0][1] = A_32 - W12 is positive for any physical tau_32 (the relaxation feed dominates
        the level-2 absorption it displaces) while J[2][0] = (kout - K2n) b2c changes sign with
        K2 -- so what is asserted, and gated numerically over a sweep of operating points and
        tau_32 from 1 ns to 50 us, is the property the scheme actually needs: every eigenvalue
        keeps a strictly negative real part. phi_1 is evaluated on the matrix itself, so a
        complex pair would be integrated correctly anyway."""
        kin, kout, kb_er, kb_yb, K2n, cu, A3, (c_up, r_up) = self._n3_coeffs()
        fc, Wm = self._fc, self._w_mig
        f1 = 1.0 - f2 - f3
        zero = np.zeros_like(np.asarray(f2 + f3 + b2c, float))
        dY = -(Ra_Yb + Re_Yb + 1.0 / self._tau_yb)
        j00 = -W12 - W21 - 1.0 / self._tau_er - 2.0 * c_up * cu * f2 + zero
        j01 = -W12 + A3 + zero
        j10 = -W13 - kin * b2c + 2.0 * r_up * cu * f2
        j11 = -W13 - W31 - A3 - kin * b2c - kb_er * (1.0 - b2c)
        j12 = kin * f1 + kb_er * f3
        j20 = (kout - K2n) * b2c
        j21 = kout * b2c + kb_yb * (1.0 - b2c)
        j22 = dY - kout * f1 - K2n * f2 - kb_yb * f3 - Wm * (1.0 - fc)
        j23 = Wm * (1.0 - fc) + zero
        j32 = Wm * fc + zero
        j33 = dY - Wm * fc + zero
        return [[j00, j01, zero, zero], [j10, j11, j12, zero],
                [j20, j21, j22, j23], [zero, zero, j32, j33]]

    def _n3_quasi_equilibrium(self, W13, W31, Ra_Yb, Re_Yb, f2):
        """(f3, b2c, b2nc) at a FIXED f2 from the same closed form the steady solve uses --
        `_n3_closed_form` without its derivative. This seeds the 4I11/2 and ytterbium reservoirs
        of a transient whose caller supplied only f2: tau_32 is 1e2-1e5 times shorter than
        tau_Er, so those levels have no independently meaningful history at a given Er state and
        starting them at zero would inject a spurious relaxation the caller did not ask for --
        the same argument `_b2_quasi_equilibrium` makes for the ytterbium alone."""
        f3, b2c, b2nc, _d = self._n3_closed_form(W13, W31, Ra_Yb, Re_Yb, f2)
        return f3, b2c, b2nc

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

        def _mcc_at(zz):
            if mcc_mat is None:
                return None
            return (mcc_er_of(zz), mcc_yb_of(zz))

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
        f2, b2, b2nc, f3 = self._fb_profile(c, P, mcc_mat)
        bbar = self._bbar(b2, b2nc)
        sig_idx = [i for i, kd in enumerate(kind) if kd == "signal"]
        gains_dB = np.array([10.0 * np.log10(P[i, -1] / bc[i]) for i in sig_idx])

        eta_tr = self._transfer_efficiency(c, P, f2, b2, z, b2nc, mcc_mat)
        yb_par_dB = self._yb_parasitic_gain_dB(P, f2, bbar, z, f3)
        back_ratio = self._back_transfer_ratio(f2, f3, b2, z)
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
                # ---- explicit 4I11/2 and Yb emission scale ----
                # er_4i11_2_z is the Er 4I11/2 FRACTION f3 = n3/N_Er, None in the adiabatic
                # default; back_transfer_ratio is INT k_back n3 n5c dz / INT k_tr n1 n6c dz, the
                # share of the forward transfer that the back-transfer returns to the ytterbium
                # (also None when the level is eliminated, where the phi refinement carries it).
                "tau32_s": self._tau32, "er_4i11_2_z": f3,
                "back_transfer_ratio": back_ratio,
                "upconversion_via_4i11_2": self._cup_via_n3,
                "er_4i11_2_zero_line_m": self._lam3,
                "yb_sigma_e_scale": self._yb_se_scale,
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
        if mcc is None:
            Re_Yb = np.array([float(np.dot(c["flux_e_yb"], np.maximum(P[:, j], 0.0)))
                              for j in range(P.shape[1])])
        else:
            Re_Yb = np.array([float(np.dot(c["flux_e_yb"] * mcc[1][:, j],
                                           np.maximum(P[:, j], 0.0)))
                              for j in range(P.shape[1])])
        if b2nc is None:
            nYb2 = b2 * N_Yb
            transfer = self._k_tr * n1 * nYb2
            total = (self._k_tr * n1 + 1.0 / self._tau_yb + Re_Yb) * nYb2
        else:
            n6c = self._fc * N_Yb * b2
            n6nc = (1.0 - self._fc) * N_Yb * b2nc
            idle = 1.0 / self._tau_yb + Re_Yb
            transfer = self._k_tr * n1 * n6c
            total = ((self._k_tr * n1 + self._k_tr2 * N_Er * f2 + idle) * n6c + idle * n6nc)
        num = float(trapz(transfer, z))
        den = float(trapz(total, z))
        return num / den if den > 0.0 else 0.0

    def transfer_efficiency(self, result: SteadyStateResult) -> float:
        """eta_tr for a solved result (returns the value cached in meta by solve())."""
        return float(result.meta["eta_transfer"])

    def _back_transfer_ratio(self, f2, f3, b2c, z) -> Optional[float]:
        """INT k_back n3 n5c dz / INT k_tr n1 n6c dz: the share of the forward Yb->Er transfer
        that the EXPLICIT back-transfer hands straight back to the ytterbium, with n5c and n6c
        the coupled pool's ground and excited densities. None when the 4I11/2 level is
        adiabatically eliminated (there the branching is the phi refinement, which multiplies the
        transfer rather than returning excitation), and exactly 0.0 when k_back = 0.

        This is the number that says whether a long tau_32 matters: the back-transfer rate is
        k_back n3 n5c and n3 grows in PROPORTION to tau_32, so the ratio is monotone in tau_32 at
        fixed k_back, and it is the loss channel an explicit 4I11/2 exists to expose."""
        if f3 is None:
            return None
        if self._k_back <= 0.0:
            return 0.0
        n6c = self._fc * self._n_yb * np.asarray(b2c, float)
        n5c = self._fc * self._n_yb * (1.0 - np.asarray(b2c, float))
        n1 = self._n_er * (1.0 - np.asarray(f2, float) - np.asarray(f3, float))
        fwd = float(trapz(self._k_tr * n1 * n6c, z))
        bwd = float(trapz(self._k_back * self._n_er * np.asarray(f3, float) * n5c, z))
        return bwd / fwd if fwd > 0.0 else 0.0

    def _yb_parasitic_gain_dB(self, P, f2, b2, z, f3=None) -> float:
        """Single-pass 1030 nm Yb parasitic gain [dB] from the POPULATION-WEIGHTED b2 profile (computed even with no
        yb_ase channels): G_dB = (10/ln10) INT Gamma(1030) [N_Yb (s sigma_e_Yb b2 - sigma_a_Yb
        (1 - b2)) + N_Er (sigma_e_Er f2 - sigma_a_Er (1 - f2))] dz, with s the Yb emission scale
        (`yb_sigma_e_scale`, 1.0 by default) that the channel plan applies everywhere else -- this
        is the one place the spectrum is read outside `_plan`, so it is scaled here too or the
        diagnostic would contradict the ASE it is meant to predict. The Er terms are ~0 at 1030
        nm. A large positive value flags 1-um parasitic-lasing risk (compare to the round-trip
        cavity loss -ln(R1 R2)/2; dossier design rule beta_Yb < ~0.05).

        With an EXPLICIT 4I11/2 (f3 given) 1030 nm is a LEVEL-3 channel, so the erbium ground
        fraction is 1 - f2 - f3 and the (numerically dead, ~1e-39 m^2) Er emission there is
        charged to n3 -- the same bookkeeping the propagation uses, so the two agree."""
        lam = 1.030e-6
        gam = float(overlap_gamma(self.fiber, lam))
        se_yb = float(self.yb_ion.sigma_e.sigma(lam)) * self._yb_se_scale
        sa_yb = float(self.yb_ion.sigma_a.sigma(lam))
        se_er = float(self.er_ion.sigma_e.sigma(lam)); sa_er = float(self.er_ion.sigma_a.sigma(lam))
        if f3 is None:
            g_er = self._n_er * (se_er * f2 - sa_er * (1.0 - f2))
        else:
            g_er = self._n_er * (se_er * f3 - sa_er * (1.0 - f2 - f3))
        g = gam * (self._n_yb * (se_yb * b2 - sa_yb * (1.0 - b2)) + g_er)
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
            dF = np.array([float(np.sum(u * self._dP(c, u, P[:, j],
                                                     (mcc[0][:, j], mcc[1][:, j]))))
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
        terms = self.energy_terms(result.power_W, f2, b2c, b2nc, z_m=z,
                                  f3=result.meta.get("er_4i11_2_z"))
        return np.asarray(terms["dissipation"], float)

    def energy_terms(self, power_W, f2, b2, b2_uncoupled=None, *, z_m=None, f3=None) -> dict:
        """Per-z energy bookkeeping [W/m] for one profile: powers (K, M), f2 (M,), b2 (M,).

        `f3` is the Er 4I11/2 fraction (M,), REQUIRED (and refused if missing) whenever `tau32_s`
        is set and refused when it is not -- the same rule `b2_uncoupled` follows, and for the
        same reason: without it the stored energy, the ground fraction that absorbs and the split
        of the transfer heat between the transfer defect and the 4I11/2 relaxation would all be
        silently wrong. solve() reports it on `meta['er_4i11_2_z']` and the march on
        `meta['er_4i11_2']`.

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
        adds, and so is the photodarkening gray loss, both of which are pure heat.

        EXPLICIT 4I11/2 EXTENSION (`tau32_s` set, f3 passed). The erbium stores a third quantum,
        every erbium optical event is charged at the level it actually terminates on, and the
        transfer heat splits into the part the transfer itself pays and the part the relaxation
        pays:

            U      = A (eps_Er N_Er f2 + eps_3 N_Er f3 + eps_Yb N_Yb bbar)
            Phi_tr = A k_tr N_Er (1 - f2 - f3) (f N_Yb) b2c        (lands on 4I11/2 now)
            Phi_32 = A N_Er f3 / tau_32,     Phi_bk = A k_back N_Er f3 (f N_Yb)(1 - b2c)
            D_Er   = SUM_{k in lvl2} (h nu_k - eps_Er)(n_a,k - n_e,k)
                   + SUM_{k in lvl3} (h nu_k - eps_3 )(n_a,k - n_e,k)
                   + (eps_Er Phi_dec_Er - P_sp_lvl2) + (c_up eps_Er - r_up eps_3) Phi_up
            D_tr   = (eps_Yb - eps_3) Phi_tr
            D_32   = (eps_3 - eps_Er) Phi_32 - P_sp_lvl3        <- NEW: the multiphonon step
            D_bk   = (eps_3 - eps_Yb) Phi_bk                    <- NEW, and NEGATIVE

            q_opt(z) = dU/dt(z) + D_loss + D_Er + D_Yb + D_tr + D_K2 + D_32 + D_bk

        Three things to read off. `D_tr + D_32` is the adiabatic `(eps_Yb - eps_Er) Phi_tr` once
        every transferred ion relaxes immediately, so the TOTAL is unchanged in the adiabatic
        limit and only the naming moves -- which is the point of making the relaxation explicit.
        `D_bk` is negative because the back-transfer is UPHILL: eps_3 = 2.0332e-19 J (977 nm) sits
        BELOW eps_Yb = 2.0384-2.0421e-19 J (the shipped ytterbium zero lines, 974.5-972.75 nm),
        so 3.3-5.5 meV of phonon energy is ABSORBED per event. It is a real, tiny anti-Stokes
        term, not a sign error. And the guided spontaneous emission of a level-3 channel is
        credited against the relaxation energy (`- P_sp_lvl3`) exactly as the C-band ASE is
        credited against the 4I13/2 decay; the model carries no separate radiative
        branching ratio for 4I11/2, whose relaxation in a phosphosilicate host is multiphonon by
        ~1e3. The extra keys on this path are 'relaxation_32_defect', 'back_transfer_defect',
        'relaxation_32_rate_per_m', 'back_transfer_rate_per_m' and 'df3_dt'."""
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
        if self._n3 and f3 is None:
            raise ValueError(
                "energy_terms: this amplifier carries an EXPLICIT Er 4I11/2 level "
                "(tau32_s=%g s), so its population fraction f3 must be passed as f3=... -- "
                "solve() reports it on meta['er_4i11_2_z'] and the march on "
                "meta['er_4i11_2']. Defaulting it to zero would mis-state the stored energy, "
                "the erbium ground fraction and the split of the transfer heat."
                % (self._tau32,))
        if f3 is not None and not self._n3:
            raise ValueError("energy_terms: this amplifier eliminates the Er 4I11/2 level "
                             "adiabatically (tau32_s is None), so f3 must be None (got an array "
                             "of shape %r)" % (np.shape(f3),))
        if f3 is not None and np.shape(f3) != (P.shape[1],):
            raise ValueError("energy_terms: f3 must have shape (M,); got %r" % (np.shape(f3),))
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
        f3a = None if f3 is None else np.asarray(f3, float)
        one_f = ((1.0 - f2)[None, :] if f3a is None else (1.0 - f2 - f3a)[None, :])
        one_b = (1.0 - bbar)[None, :]
        ge_er = c["g_e_er"][:, None] if mcc is None else c["g_e_er"][:, None] * mcc[0]
        ge_yb = c["g_e_yb"][:, None] if mcc is None else c["g_e_yb"][:, None] * mcc[1]
        se_pref = c["s_er"][:, None] if mcc is None else c["s_er"][:, None] * mcc[0]
        sy_pref = c["s_yb"][:, None] if mcc is None else c["s_yb"][:, None] * mcc[1]

        pw_a_er = c["g_a_er"][:, None] * one_f * P
        pw_e_er = ge_er * f2[None, :] * P
        pw_sp_er = se_pref * f2[None, :]
        pw_a_yb = c["g_a_yb"][:, None] * one_b * P
        pw_e_yb = ge_yb * bbar[None, :] * P
        pw_sp_yb = sy_pref * bbar[None, :]
        loss_W = np.sum(c["loss"][:, None] * P, axis=0)

        p_a_er, p_e_er, p_sp_er = pw_a_er.sum(0), pw_e_er.sum(0), pw_sp_er.sum(0)
        p_a_yb, p_e_yb, p_sp_yb = pw_a_yb.sum(0), pw_e_yb.sum(0), pw_sp_yb.sum(0)
        n_a_er, n_e_er = (pw_a_er * inv_h).sum(0), (pw_e_er * inv_h).sum(0)
        n_a_yb, n_e_yb = (pw_a_yb * inv_h).sum(0), (pw_e_yb * inv_h).sum(0)

        dec_yb = A * N_Yb * bbar / self._tau_yb
        n_yb_c = self._fc * N_Yb                     # the COUPLED Yb density (= N_Yb at f = 1)
        k2_rate = A * self._k_tr2 * N_Er * f2 * n_yb_c * b2
        d_yb = ((p_a_yb - p_e_yb) - eps_yb * (n_a_yb - n_e_yb)
                + (eps_yb * dec_yb - p_sp_yb))
        d_k2 = eps_yb * k2_rate

        if f3a is not None:
            # ---- EXPLICIT 4I11/2 -------------------------------------------------------------
            # The erbium optical terms are re-split by LEVEL: everything absorbs from n1, a
            # level-2 channel emits from n2 and a level-3 channel from n3, and each event is
            # charged at ITS OWN level energy (eps_Er or eps_3). The transfer now lands on
            # 4I11/2, so its defect is (eps_Yb - eps_3) and the remaining (eps_3 - eps_Er) is
            # charged to the 4I11/2 -> 4I13/2 relaxation -- which is where it always was, inside
            # the adiabatic transfer defect. Their SUM is unchanged in the adiabatic limit and
            # that is the gate.
            # NOTE the block above computed pw_e_er / p_e_er / n_e_er charging EVERY channel's
            # emission to n2; on this path they are discarded and recomputed per level, which is
            # exactly what the split replaces. pw_a_er is reused as-is -- absorption is from n1
            # at every channel either way, and `one_f` already carries the 1 - f2 - f3 form.
            w3c = c["w3"][:, None]
            ge2 = c["g_e_er2"][:, None] if mcc is None else c["g_e_er2"][:, None] * mcc[0]
            ge3 = c["g_e_er3"][:, None] if mcc is None else c["g_e_er3"][:, None] * mcc[0]
            s2p = c["s_er2"][:, None] if mcc is None else c["s_er2"][:, None] * mcc[0]
            s3p = c["s_er3"][:, None] if mcc is None else c["s_er3"][:, None] * mcc[0]
            pw_a3 = pw_a_er * w3c
            pw_a2 = pw_a_er - pw_a3
            pw_e2 = ge2 * f2[None, :] * P
            pw_e3 = ge3 * f3a[None, :] * P
            pw_sp2 = s2p * f2[None, :]
            pw_sp3 = s3p * f3a[None, :]
            p_a2, p_a3 = pw_a2.sum(0), pw_a3.sum(0)
            p_e2, p_e3 = pw_e2.sum(0), pw_e3.sum(0)
            p_sp2, p_sp3 = pw_sp2.sum(0), pw_sp3.sum(0)
            n_a2, n_a3 = (pw_a2 * inv_h).sum(0), (pw_a3 * inv_h).sum(0)
            n_e2, n_e3 = (pw_e2 * inv_h).sum(0), (pw_e3 * inv_h).sum(0)
            p_a_er, p_e_er, p_sp_er = p_a2 + p_a3, p_e2 + p_e3, p_sp2 + p_sp3
            q_opt = (p_a_er + p_a_yb + loss_W) - (p_e_er + p_e_yb + p_sp_er + p_sp_yb)

            rates = self._rates_profile_n3(c, P, mcc)
            df, df3, db, dbn = self._fb_rhs_n3(rates[0], rates[1], rates[2], rates[3],
                                               rates[4], rates[5], f2, f3a, b2,
                                               b2 if b2nc is None else b2nc)
            dbbar = db if b2nc is None else self._fc * db + (1.0 - self._fc) * dbn
            _ki, _ko, _ke, _ky, _K2n, _cu, A3, (c_up, r_up) = self._n3_coeffs()
            tr_rate = A * self._k_tr * N_Er * (1.0 - f2 - f3a) * n_yb_c * b2
            back_rate = A * self._k_back * N_Er * f3a * n_yb_c * (1.0 - b2)
            relax_rate = A * N_Er * f3a * A3
            dec_er = A * N_Er * f2 / self._tau_er
            up_er = A * N_Er * self.upconversion_C_up * N_Er * f2 * f2
            d_er = ((p_a2 - p_e2) - eps_er * (n_a2 - n_e2)
                    + (p_a3 - p_e3) - self._eps3 * (n_a3 - n_e3)
                    + (eps_er * dec_er - p_sp2)
                    + (c_up * eps_er - r_up * self._eps3) * up_er)
            d_tr = (eps_yb - self._eps3) * tr_rate
            d_32 = (self._eps3 - eps_er) * relax_rate - p_sp3
            d_back = (self._eps3 - eps_yb) * back_rate
            out = {"q_optical": q_opt,
                   "stored_J_per_m": A * (eps_er * N_Er * f2 + self._eps3 * N_Er * f3a
                                          + eps_yb * N_Yb * bbar),
                   "d_stored_dt": A * (eps_er * N_Er * df + self._eps3 * N_Er * df3
                                       + eps_yb * N_Yb * dbbar),
                   "background_loss": loss_W, "er_dissipation": d_er, "yb_dissipation": d_yb,
                   "transfer_defect": d_tr, "k2_defect": d_k2,
                   "relaxation_32_defect": d_32, "back_transfer_defect": d_back,
                   "dissipation": loss_W + d_er + d_yb + d_tr + d_k2 + d_32 + d_back,
                   "transfer_rate_per_m": tr_rate, "k2_rate_per_m": k2_rate,
                   "relaxation_32_rate_per_m": relax_rate,
                   "back_transfer_rate_per_m": back_rate,
                   "df2_dt": df, "df3_dt": df3, "db2_dt": db}
            if b2nc is not None:
                out["db2nc_dt"] = dbn
                out["beta_yb_mean"] = bbar
            return out

        q_opt = (p_a_er + p_a_yb + loss_W) - (p_e_er + p_e_yb + p_sp_er + p_sp_yb)

        ra_er, re_er, ra_yb, re_yb = self._rates_profile(c, P, mcc)
        df, db, dbn = self._fb_rhs3(ra_er, re_er, ra_yb, re_yb, f2, b2,
                                    b2 if b2nc is None else b2nc)
        dbbar = db if b2nc is None else self._fc * db + (1.0 - self._fc) * dbn
        phi = self._phi(b2)
        tr_rate = A * self._k_tr * phi * N_Er * n_yb_c * b2 * (1.0 - f2)
        dec_er = A * N_Er * f2 / self._tau_er
        up_er = A * N_Er * self.upconversion_C_up * N_Er * f2 * f2

        d_er = ((p_a_er - p_e_er) - eps_er * (n_a_er - n_e_er)
                + (eps_er * dec_er - p_sp_er) + eps_er * up_er)
        d_tr = (eps_yb - eps_er) * tr_rate
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

    def stored_energy_J(self, f2, b2, z, b2_uncoupled=None, *, f3=None) -> float:
        """Total excitation energy stored in the reservoirs [J]:
        INT A_dope (eps_Er N_Er f2(z) + eps_Yb N_Yb bbar(z)) dz, bbar = f b2c + (1-f) b2nc. This
        is the `stored` that a transient energy balance differentiates -- see energy_terms for
        eps_* and the identity. With two pools BOTH count: an uncoupled ytterbium ion stores the
        same 2.04e-19 J as a coupled one, it simply cannot hand it to the erbium. Pass the
        uncoupled inversion as `b2_uncoupled` (None, and `b2` is the single pool's inversion).

        With an EXPLICIT 4I11/2 pass `f3` as well: that population stores eps_3 = 2.03e-19 J per
        ion (a 977 nm quantum, 1.57x the metastable one), so leaving it out would understate the
        stored energy by 1.57 f3/f2 of the erbium term. It is REQUIRED on that path rather than
        defaulted, the same rule energy_terms follows."""
        if self._n3 and f3 is None:
            raise ValueError("stored_energy_J: this amplifier carries an EXPLICIT Er 4I11/2 "
                             "level (tau32_s=%g s), so pass f3=result.meta['er_4i11_2_z'] -- "
                             "that population stores a 977 nm quantum each and dropping it "
                             "understates the stored energy." % (self._tau32,))
        if f3 is not None and not self._n3:
            raise ValueError("stored_energy_J: this amplifier eliminates the Er 4I11/2 level "
                             "adiabatically (tau32_s is None), so f3 must be None")
        A = self.fiber.a_dope_m2
        bb = b2 if b2_uncoupled is None else self._bbar(np.asarray(b2, float),
                                                        np.asarray(b2_uncoupled, float))
        u = A * (float(self.er_ion.eps_J) * self._n_er * np.asarray(f2, float)
                 + float(self.yb_ion.eps_J) * self._n_yb * np.asarray(bb, float))
        if f3 is not None:
            u = u + A * self._eps3 * self._n_er * np.asarray(f3, float)
        return float(trapz(u, np.asarray(z, float)))


# ============================ Yb emission-scale calibration ==================================
@dataclass
class YbSigmaEScaleFit:
    """The result of `eryb_fit_yb_sigma_e_scale`: the fitted Yb emission scale and what it costs.

    Fields: `scale` (the fitted multiplier on sigma_e_Yb), `ase_1um_W` (the 1-um ASE the fitted
    amplifier emits in the requested direction) against `target_1um_W` (the measurement),
    `ase_1um_unit_scale_W` (what the UNSCALED model predicted -- the over-prediction the fit
    exists to remove), `signal_out_W` and `signal_out_unit_scale_W` with their ratio
    `signal_change_dB` (the SENSITIVITY the fit has to be judged on: Morasse's point was that the
    signal barely moves, so a 1-um-only correction is admissible), the `bracket` actually used,
    the number of `solves` spent, and `converged` -- which is the AND of the root find converging
    and every solve inside it reporting convergence."""
    scale: float
    ase_1um_W: float
    target_1um_W: float
    ase_1um_unit_scale_W: float
    signal_out_W: float
    signal_out_unit_scale_W: float
    signal_change_dB: float
    bracket: Tuple[float, float]
    solves: int
    converged: bool
    direction: str

    def __str__(self) -> str:
        over = (self.ase_1um_unit_scale_W / self.target_1um_W
                if self.target_1um_W > 0.0 else float("nan"))
        return ("YbSigmaEScaleFit(scale=%.4g, 1um %.4g W -> target %.4g W (unscaled model "
                "%.4g W, %.1fx), signal %.4g -> %.4g W (%+.3f dB), %d solves, converged=%s)"
                % (self.scale, self.ase_1um_W, self.target_1um_W, self.ase_1um_unit_scale_W,
                   over, self.signal_out_unit_scale_W, self.signal_out_W,
                   self.signal_change_dB, self.solves, self.converged))


def _eryb_1um_and_signal(amp, scale, direction, n_nodes, solve_kwargs):
    """(1-um ASE output [W], signal output [W], converged) for `amp` at one Yb emission scale."""
    a = amp.with_yb_sigma_e_scale(scale)
    r = a.solve(n_nodes=n_nodes, **solve_kwargs)
    lo, hi = amp.yb_ase.lambda_min_m, amp.yb_ase.lambda_max_m
    want_u = 1.0 if direction == "fwd" else -1.0
    end = -1 if direction == "fwd" else 0
    p_ase = 0.0
    for i, kd in enumerate(r.kind):
        if kd == "ase" and lo <= r.lambda_m[i] <= hi and r.u[i] * want_u > 0.0:
            p_ase += float(r.power_W[i, end])
    sig = [i for i, kd in enumerate(r.kind) if kd == "signal"]
    p_sig = float(r.power_W[sig[0], -1]) if sig else float("nan")
    return p_ase, p_sig, bool(r.meta["converged"])


def eryb_fit_yb_sigma_e_scale(amp, measured_1um_ase_W: float, direction: str = "bwd", *,
                              n_nodes: int = 201, bracket: Tuple[float, float] = (0.2, 1.0),
                              scale_min: float = 1.0e-3, scale_max: float = 50.0,
                              rtol: float = 1.0e-3, max_solves: int = 40,
                              **solve_kwargs) -> YbSigmaEScaleFit:
    """Fit the ONE-parameter Yb emission scale `yb_sigma_e_scale` to a MEASURED 1-um ASE output.

    WHAT PROBLEM THIS SOLVES. Morasse et al. (2006) measured every parameter of a CorActive
    Er:Yb fiber -- core diameter, NA, both ion densities, both lifetimes, the cladding area, the
    background loss -- and the model still over-predicted the backward 1-um ASE by a factor 1000
    (19.5 mW against 0.079 mW measured), while the SIGNAL output was insensitive to the
    correction. He fixed it by scaling the McCumber-derived Yb emission cross-section by 0.4.
    That is a one-parameter calibration against one measurement, and this function is it: a
    bracketed root find on the 1-um ASE output at a fixed operating point, reporting the scale
    AND the signal change it causes -- because the second number is what says whether the
    correction is admissible. A scale that also moves the signal by dB is not calibrating the
    1-um band, it is refitting the amplifier.

    ARGUMENTS. `amp` must carry a `yb_ase` band -- the 1-um ASE is read from THAT band's
    wavelength range, not from a hardcoded window -- and its C-band `ase` must not overlap it.
    `direction` is "bwd" (the backward output at z = 0, which is what a cut-back measurement
    collects at the input end, and Morasse's case) or "fwd". `measured_1um_ase_W` is the total
    power in the band, in watts. `n_nodes` and any extra keywords go to `ErYbAmplifier.solve`
    (pass `relax=0.5` or `relax="auto"` for a long, deeply absorbed fiber).

    METHOD. The 1-um ASE is exponential in the Yb gain integral, so the root find runs on
    log(ASE) against log(scale): g(ln s) = ln ASE(s) - ln target, bracketed by geometric
    expansion of `bracket` (halving/doubling outward, to `scale_min`/`scale_max`) and closed by
    Brent's method to `rtol` in the scale. Every evaluation is a full relaxation solve, so the
    count is reported and capped; a typical fit is 8-14 solves. Monotonicity is NOT assumed --
    the bracket is established by an actual sign change, and if none exists inside
    [scale_min, scale_max] the function RAISES rather than returning the nearest endpoint, which
    is the honest outcome when a measurement and a model disagree by more than one emission scale
    can absorb.

    The fitted scale is a property of the ION MODEL as much as of the fiber: a McCumber-derived
    sigma_e_Yb and a measured table do not need the same correction, so report which ion pair the
    fit was run against. See the module docstring for what the scale does and does not touch."""
    from scipy.optimize import brentq
    if amp.yb_ase is None:
        raise ValueError("eryb_fit_yb_sigma_e_scale: the amplifier carries no yb_ase band, so it "
                         "emits no 1-um ASE to fit -- add yb_ase=AseBand(1.00e-6, 1.10e-6, n)")
    if direction not in ("fwd", "bwd"):
        raise ValueError("eryb_fit_yb_sigma_e_scale: direction must be 'fwd' or 'bwd'; got %r"
                         % (direction,))
    if not (float(measured_1um_ase_W) > 0.0):
        raise ValueError("eryb_fit_yb_sigma_e_scale: measured_1um_ase_W must be > 0; got %r"
                         % (measured_1um_ase_W,))
    if amp.ase is not None and not (amp.ase.lambda_max_m <= amp.yb_ase.lambda_min_m
                                    or amp.ase.lambda_min_m >= amp.yb_ase.lambda_max_m):
        raise ValueError("eryb_fit_yb_sigma_e_scale: the Er ASE band [%.4g, %.4g] m overlaps the "
                         "Yb band [%.4g, %.4g] m, so the 1-um output cannot be separated from "
                         "the C-band one by wavelength"
                         % (amp.ase.lambda_min_m, amp.ase.lambda_max_m,
                            amp.yb_ase.lambda_min_m, amp.yb_ase.lambda_max_m))
    lo, hi = float(bracket[0]), float(bracket[1])
    if not (0.0 < lo < hi):
        raise ValueError("eryb_fit_yb_sigma_e_scale: bracket must be (lo, hi) with 0 < lo < hi; "
                         "got %r" % (bracket,))
    target = float(measured_1um_ase_W)
    state = {"n": 0, "ok": True}
    cache = {}

    def _ase(s):
        key = float(s)
        if key not in cache:
            if state["n"] >= int(max_solves):
                raise RuntimeError(
                    "eryb_fit_yb_sigma_e_scale: exceeded max_solves=%d relaxation solves. Either "
                    "the 1-um ASE is not responding to the scale at this operating point (check "
                    "that the fiber produces 1-um ASE at scale 1.0 at all) or rtol=%g is tighter "
                    "than the solve's own convergence." % (max_solves, rtol))
            state["n"] += 1
            p_ase, p_sig, conv = _eryb_1um_and_signal(amp, key, direction, n_nodes, solve_kwargs)
            state["ok"] = state["ok"] and conv
            cache[key] = (p_ase, p_sig)
        return cache[key][0]

    def _g(ln_s):
        p = _ase(np.exp(ln_s))
        if not (p > 0.0):
            # a fully quenched 1-um band: log space cannot express it, and it is unambiguously
            # BELOW any positive target, so return a large negative residual instead of -inf
            return -700.0
        return float(np.log(p) - np.log(target))

    g_lo, g_hi = _g(np.log(lo)), _g(np.log(hi))
    while g_lo > 0.0 and lo > scale_min:                  # the whole bracket over-predicts
        hi, g_hi = lo, g_lo
        lo = max(lo * 0.5, scale_min)
        g_lo = _g(np.log(lo))
    while g_hi < 0.0 and hi < scale_max:                  # the whole bracket under-predicts
        lo, g_lo = hi, g_hi
        hi = min(hi * 2.0, scale_max)
        g_hi = _g(np.log(hi))
    if g_lo > 0.0 or g_hi < 0.0:
        raise RuntimeError(
            "eryb_fit_yb_sigma_e_scale: no Yb emission scale in [%g, %g] reproduces a 1-um ASE "
            "of %.4g W -- the model gives %.4g W at %g and %.4g W at %g. The measurement and the "
            "model disagree by more than one emission scale can absorb at this operating point; "
            "check the pump power, the fiber length and the band definition before widening "
            "scale_min/scale_max." % (scale_min, scale_max, target, _ase(lo), lo, _ase(hi), hi))
    ln_root = brentq(_g, np.log(lo), np.log(hi), xtol=float(np.log1p(rtol)), rtol=1e-12)
    s_fit = float(np.exp(ln_root))
    if s_fit not in cache:
        _ase(s_fit)
    p_ase, p_sig = cache[s_fit]
    if 1.0 not in cache:
        state["n"] += 1
        p1_ase, p1_sig, conv1 = _eryb_1um_and_signal(amp, 1.0, direction, n_nodes, solve_kwargs)
        state["ok"] = state["ok"] and conv1
        cache[1.0] = (p1_ase, p1_sig)
    p1_ase, p1_sig = cache[1.0]
    d_sig_dB = (10.0 * np.log10(p_sig / p1_sig) if (p_sig > 0.0 and p1_sig > 0.0)
                else float("nan"))
    return YbSigmaEScaleFit(scale=s_fit, ase_1um_W=float(p_ase), target_1um_W=target,
                            ase_1um_unit_scale_W=float(p1_ase), signal_out_W=float(p_sig),
                            signal_out_unit_scale_W=float(p1_sig),
                            signal_change_dB=float(d_sig_dB), bracket=(float(lo), float(hi)),
                            solves=int(state["n"]), converged=bool(state["ok"]),
                            direction=direction)
