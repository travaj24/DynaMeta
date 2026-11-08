"""
Opt-in bipolar drift-diffusion physics beyond SRH (audit R26 + R27, DynaMeta v0.14.0):
Auger + radiative recombination, apparent bandgap narrowing (BGN), incomplete dopant ionization,
an explicit lattice temperature, and a uniform external generation term.

EVERYTHING HERE IS OFF BY DEFAULT. `physics_bipolar_dd.setup_bipolar_region(..., options=None)` (the
default) never calls into this module, so every pre-existing bipolar result is byte-identical. A
`BipolarOptions` whose fields are all None is likewise inert (`active` is False).

ONE IMPLEMENTATION PER KERNEL. The fixed (doping-only) fields -- Delta E_g, the band-edge split, the
effective intrinsic density n_ie, the ionization reference densities, the charge-neutral equilibrium
(n0, p0) and the contact/seed potential -- are evaluated ONCE, in numpy, by `equilibrium_fields`, which
both the DEVSIM region setup (`install_option_fields` / `install_option_terms`) and the scalar
gate-frame helper `physics_bipolar_dd.ohmic_built_in_offset_V` call -- and, since the v0.14
integration, its charge-neutral root also carries the trapped charge of CHARGED physics_traps levels
(one contact rule for BGN, incomplete ionization and traps). The solution-dependent DEVSIM strings (the
recombination terms and the ionized-dopant charge) each come from one builder (`recombination_exprs`,
`ionized_net_doping_expr`); the SRH kernel is physics_traps.srh_rate_expr (shared with every trap
level and interface bin). Shared by the 1-D direct path, the 2-D LayeredDevsimBuilder and the 3-D
Stacked3DSpec builder (they all route through setup_bipolar_region).

COMPOSITION (v0.14 integration): Auger, radiative and the external G are ADDITIVE terms of
physics_bipolar_dd.add_recombination_term (RTerm_auger / RTerm_radiative / RTerm_gext, each with its
noise pair rate), the ionized-dopant charge re-bases physics_bipolar_dd.rebuild_poisson_charge; nothing
here re-defines a composite by name, so options compose with trap levels, interface traps and BTBT.

=====================================================================================================
1. RECOMBINATION (R26). Net rates U [m^-3 s^-1], positive = recombination, added to SRH:

   Auger      U_A   = (C_n n + C_p p) (n p - n_ie^2)
              Dziewior & Schmid, Appl. Phys. Lett. 31, 346 (1977): Si, 300 K,
              C_n = 2.8e-31 cm^6/s (eeh), C_p = 0.99e-31 cm^6/s (ehh), measured on heavily doped Si
              (N > 5e18 cm^-3) -- i.e. WITHOUT Coulomb enhancement, which matters below ~1e18.
   Coulomb-   C_n -> g_eeh(n0) C_n ,  C_p -> g_ehh(p0) C_p   (optional, `coulomb_enhanced=True`)
   enhanced   g_eeh(n0) = 1 + 13   [1 - tanh((n0/N_eeh)^0.66)],  N_eeh = 3.3e17 cm^-3
              g_ehh(p0) = 1 + 7.5  [1 - tanh((p0/N_ehh)^0.63)],  N_ehh = 7.0e17 cm^-3
              Altermatt, Schmidt, Heiser, Aberle, J. Appl. Phys. 82, 4938 (1997) (low-injection
              parameterization; n0/p0 are the EQUILIBRIUM densities, so g is a fixed field here).
   Radiative  U_rad = B (n p - n_ie^2)
              Trupke et al., J. Appl. Phys. 94, 4930 (2003): Si, 300 K, low injection,
              B = 9.5e-15 cm^3/s.
              v0.15: a B is INTERNAL (recycling-free) or EFFECTIVE (net of photon recycling); in a
              direct-gap layer they differ (GaAs 2 um DH: 4.2x). RadiativeModel.kind labels it and
              photon_recycling_factor P_r gives B = B_int (1 - P_r), applied once at the B_rad write.
              III-V presets (GaAs, InP; internal): materials.semiconductor_presets.

   Low-injection lifetimes (n-type, n0 >> p0, the oracles' closed forms):
              1/tau = 1/tau_p(SRH) + C_n n0^2 + B n0      (tau_A = 1/(C_n n0^2), tau_rad = 1/(B n0)).
   Detailed balance: every term vanishes at n p = n_ie^2, the SAME n_ie the contacts and the SRH term
   use (with BGN on, SRH's n_i^2 -> n_ie^2 and n1, p1 -> n1 n_ie/n_i, p1 n_ie/n_i so n1 p1 = n_ie^2;
   otherwise SRH would recombine AT EQUILIBRIUM inside a narrowed region).

   Temperature / injection dependence of C and B is NOT modelled (constant coefficients): the
   cited values are 300 K, and Dziewior-Schmid's own data are only weakly T-dependent. Altermatt 2005's
   injection-dependent B and Kerr-Cuevas/Richter ambipolar Auger are deliberately out of scope.

2. BANDGAP NARROWING (R27). Apparent (electrically extracted) narrowing as a function of a dopant
   density N chosen by the model's `doping_basis` -- the density the FIT was made in:

   Slotboom   dEg = E_ref [ ln(N/N_ref) + sqrt( ln^2(N/N_ref) + C ) ],   N = N_D + N_A ("total")
              Slotboom & de Graaff, Solid-State Electron. 19, 857 (1976) / IEEE TED 24, 1123 (1977):
                E_ref = 9 mV, N_ref = 1e17 cm^-3, C = 0.5   ("old Slotboom"; fitted on p-type
                n-p-n transistor BASES, i.e. in N_A -- Swirhun, del Alamo & Swanson's Stanford/JPL
                report (1986) sec. 3.2; applied here to N_D + N_A by the usual convention)
              Klaassen, Slotboom & de Graaff, Solid-State Electron. 35, 125 (1992) re-fit (unified n/p):
                E_ref = 6.92 mV, N_ref = 1.3e17 cm^-3, C = 0.5
   del Alamo  dEg = E_ref ln(N_D/N_ref) for N_D > N_ref, else 0,   N = N_D ("donors")
              del Alamo & Swanson, IEEE TED ED-34, 1580 (1987), Eq. (11) (also del Alamo, Swirhun &
              Swanson, IEDM 1985, p. 290, Eq. (7)); phosphorus-doped n-type Si ONLY:
                E_ref = 18.7 mV, N_ref = 7e17 cm^-3
              The fit has no basis on p-type or compensated material: it is evaluated in N_D only (a
              p-type node gets dEg from its donors, i.e. 0 when uncompensated), and
              `install_option_fields` emits one UnreliableResultWarning (the model is used outside
              its validity range) per region in which some node has N_A > N_ref (where the fit would
              be needed and is not valid). Use a Slotboom/Klaassen
              set for a heavily doped p-type region.
   power_log  dEg = A [ln(N/N_ref)]^b + C_off for N > N_ref, else C_off (0 for N <= 0)   (v0.15)
              Yan & Cuevas, J. Appl. Phys. 114, 044508 (2013), Eq. (3), Table I; n-type (Si:P), in N_D:
                REAL narrowing for Fermi-Dirac statistics:   A = 0.042 meV, N_ref = 1e14 cm^-3, b = 3
                APPARENT narrowing for Boltzmann statistics: A = 12.96 meV, N_ref = 1e17 cm^-3, b = 1
              (C_off = 0 for both). The clamp "C_off below N_ref" is OUR choice: with b = 1 the term is
              negative below N_ref and with b = 3 non-monotone there; the paper fits diffused regions
              with N_D >= ~4e18 only.
   The legacy sets (Slotboom, Klaassen, del Alamo) are APPARENT narrowings: they were extracted
   assuming Boltzmann statistics and a specific reference n_i, so (a) they must be paired with the n_i
   the user passes (state it with the result), and (b) on a DEGENERATE region with the FD g-factor on,
   the degeneracy is partly counted twice -- `install_option_fields` warns when that correction is
   non-negligible. Schenk's 1998 many-body model (carrier-density-dependent band edges) is NOT
   implemented (see the audit doc, sec. Limits).

   PAIRED SETS (v0.15, user decisions 26-29). A BGNModel that states `paired_n_i_m3` (and its
   `paired_statistics`, `paired_T_K`, for FD also `paired_N_c_m3`/`paired_N_v_m3`) is CHECKED against
   the region by ONE function, `check_bgn_pairing` (called by install_option_fields and by the 3-D gate
   frame physics_bipolar_dd.ohmic_built_in_offset_V): a region n_i (or N_c, N_v) off by more than 1e-3,
   another statistics, or |T - paired_T_K| > 0.5 K raises BGNPairingError, whose message gives the
   n_i mismatch as a gap equivalent kT ln((n_i,paired/n_i)^2) (1.84 meV for 1.0e10 against 9.65e9).
   BipolarOptions(allow_unpaired_bgn=True) turns the raise into one AdvisoryWarning per region -- use
   it only with your own n_i(T) (and N_c, N_v) at T != 300 K, or to reproduce a paper that paired
   differently. Legacy (unpaired) presets are never checked. The house Si pairing is Yan & Cuevas 2013
   FD: dynameta.materials.semiconductor_presets.SI_PAIRING_YC2013_FD (n-type only: its p-type
   constants, Yan & Cuevas, J. Appl. Phys. 116, 194505 (2014), have not been read, so a p-type node
   with N_A > N_ref is REFUSED with BGNFitPendingError instead of silently getting 0).

   STATISTICS (v0.15, decision 28; setup_bipolar_region(statistics=...), TransportModel.
   carrier_statistics). v0.15.0 (user decision, 2026-11-07): the DEFAULT (statistics None with the FD
   g-factor) is "fermi_dirac". Three choices -- which to use for which calculation:
     "legacy"        (was the default up to v0.14; now explicit, NOT recommended -- reproducing v0.14 numbers
                     only) BYTE-IDENTICAL to v0.14: FD g-factor transport + FD contact
                     potential (fd_enhancement=True), but every recombination term and the contacts pin
                     the BOLTZMANN product n p = n_ie^2. On a DEGENERATE region that contradicts the
                     transport's own equilibrium n p = n_ie^2 exp(-Delta(n/N_c) - Delta(p/N_v)) (zero
                     current <=> ln n + Delta - psi_n/V_t = const), so a junction next to a degenerate
                     layer carries a ZERO-BIAS current (MEASURED: -0.120 J0p at V = 0 on the R27-c long
                     diode with a 1e19 cm^-3 emitter; exp(-Delta(0.357)) - 1 = -0.122). The FORWARD
                     emitter J0 is the FD one (0.8789 of the Boltzmann J0, exp(-Delta) = 0.8781). Fine
                     for non-degenerate devices and for every v0.14 result; not for a paired FD set.
     "fermi_dirac"   (the DEFAULT since v0.15.0) FD transport AND FD detailed balance: n_ie becomes the
                     solution-dependent node model
                     n_ie = n_i exp(dEg/2kT) sqrt(gamma_n gamma_p), gamma = exp(-Delta(c/N)) (Delta in the
                     closed form of physics_heterojunction.setup_log_activity, the transport's own
                     integral of g), so SRH, Auger, radiative and (neutral) bulk trap rates vanish
                     exactly where the transport is in equilibrium, and the contacts pin the FD
                     charge-neutral root n0 p0 = n_ie0^2 gamma_n(n0) gamma_p(p0) -- Yan & Cuevas 2013
                     Eq. (1) with the code's Delta. REQUIRED by an FD-paired BGN set (the house Si
                     bundle). v0.15 completion (docs/audit/2026-11-06-v0.15-fermi-dirac-mode-complete.md):
                     the TRANSPORT takes the degeneracy as a quasi-field (g = 1, Delta1 - Delta0 in the
                     SG argument; physics_heterojunction.fd_quasi_field_expr), so the discrete zero-
                     current state IS this equilibrium -- zero-bias currents vanish to round-off and the
                     diffusion noise is exactly Nyquist (the legacy edge-averaged g leaves an O(h^2)
                     battery). Every SRH channel (built-in, TrapLevel, interface bin) uses the EXACT FD
                     emission densities gamma_n n1, gamma_p p1 (physics_traps.srh_reference_densities
                     fd=True); CHARGED levels enter the contacts' FD root at their exact FD occupancy;
                     local BTBT takes the FD Hurkx factor (tunneling.hurkx_d_expr n_i_p=); the nonlocal
                     BTBT and thermionic interfaces need nothing (they read the FD activities). All with
                     exact Jacobians. Its Delta is the
                     rational g fit's: within 1 % of the exact F_1/2 offset only for eta >~ 8
                     (physics_bipolar_dd "ACCURACY OF Delta"); at Si:P 1e19 / 1e20 cm^-3 (eta = -0.93 /
                     2.43) n0 p0 is 0.42 % / 3.5 % low against Eq. (1) with exact F_1/2 (MEASURED).
     "boltzmann"     fd_enhancement=False: Boltzmann transport and contacts. The right choice for an
                     APPARENT (Boltzmann-extracted) BGN set on a degenerate layer -- it removes the
                     double count the FD advisory reports -- and for the YC2013 Boltzmann pairing.

   REFERENCE n_i OF EACH SET (the pairing; these presets are LEGACY and UNPAIRED -- nothing checks the
   region's n_i against them). None of them was fitted with n_i = 1.0e10 cm^-3:
     old Slotboom   ~1.44e10 cm^-3 at 300 K [inferred from the source; the Stanford re-fit quotes
                    n_i(295 K) = 9.565e9]
     del Alamo      ~1.36-1.45e10 cm^-3 [inferred, TED 1987 Table II]
     Klaassen 1992  1.0e10 or 1.08e10 (Green 1990) -- not verified in the primary
     Yan & Cuevas 2013 (both sets)  9.65e9 cm^-3, N_c 2.86e19, N_v 3.1e19 cm^-3 [primary, Sec. I, III.B]
                                    -- PAIRED (checked, see below)
   The modern consensus n_i(300 K) is 9.65e9 cm^-3 (Altermatt et al., J. Appl. Phys. 93, 1598 (2003);
   Misiakos & Tsamakis 1993: 9.7 +/- 0.1e9), and it belongs with Fermi-Dirac statistics and a REAL
   (not apparent) narrowing. Pairing old Slotboom or del Alamo with 9.65e9 under-predicts n_ie^2 by
   ~2.3x (~21 meV of gap). Which n_i to pass for which calculation: the n_i the chosen set was fitted
   with when you want the set's own n_ie (e.g. an emitter J0 against that paper's data); the FD-paired
   Yan-Cuevas set (BGN_SI_YAN_CUEVAS_2013_FD with statistics="fermi_dirac", or the bundle
   materials.semiconductor_presets.SI_PAIRING_YC2013_FD) when you want the consensus n_i.

   EFFECTIVE INTRINSIC DENSITY + BAND-EDGE SPLIT (the classic error source; Lundstrom & Schuelke,
   IEEE TED 30, 1151 (1983); Selberherr, "Analysis and Simulation of Semiconductor Devices" (1984),
   sec. 4.x "heavy doping"). The gap shrinks by dEg; a fraction gamma (`conduction_fraction`) lowers
   E_c and the rest raises E_v:
              E_c = E_c0 - q psi - gamma dEg ,   E_v = E_v0 - q psi + (1 - gamma) dEg.
   With Boltzmann statistics and the potential referenced to the UNNARROWED intrinsic level:
              n = n_i exp((psi + D_c - phi_n)/V_t),   p = n_i exp((phi_p - psi + D_v)/V_t),
              D_c = gamma dEg/q,  D_v = (1 - gamma) dEg/q  [V]   =>   n p = n_ie^2 exp((phi_p-phi_n)/V_t)
              n_ie = n_i exp(dEg / (2 k T)).
   So electrons drift in the EFFECTIVE potential psi_n = psi + D_c and holes in psi_p = psi - D_v --
   the quasi-fields of Lundstrom-Schuelke. In the Scharfetter-Gummel edge flux that is
              vdiff_n = vdiff + (D_c@n0 - D_c@n1)/V_t ,   vdiff_p = vdiff - (D_v@n0 - D_v@n1)/V_t.
   gamma = 1/2 reproduces the "effective intrinsic density" form n = n_ie exp((psi - phi_n)/V_t)
   (the symmetric split; also the default of the ASYMMETRY parameter in commercial TCAD). On a UNIFORM
   region the split is invisible (a constant shift): only n_ie enters J0 = q n_ie^2 D_p/(N_D L_p). It
   matters only where the doping is GRADED, where it sets how the narrowing field divides between
   electrons and holes -- the gate `test_bgn_graded_equilibrium_is_current_free` pins exactly that.
   Ohmic contact: n0 p0 = n_ie^2 with neutrality, and the pinned potential is
              psi_eq = V_t [ln(n0/n_i) + Delta_FD] - D_c  (n-type)  =  -V_t [ln(p0/n_i) + Delta_FD] + D_v (p).

3. INCOMPLETE IONIZATION (R27). Ionized fractions, Boltzmann (non-degenerate) occupancy of a single
   dopant level E_D = E_c - dE_D (donors) and E_A = E_v + dE_A (acceptors), e.g. Sze & Ng, "Physics of
   Semiconductor Devices", 3rd ed. (2007), ch. 1 (impurity occupancy with degeneracy factors):
              N_D+ = N_D / (1 + g_D n/n1D),   n1D = N_c exp(-dE_D/kT),   g_D = 2
              N_A- = N_A / (1 + g_A p/p1A),   p1A = N_v exp(-dE_A/kT),   g_A = 4
   (n/N_c = exp((E_Fn - E_c)/kT): occupancy follows each carrier's OWN quasi-Fermi level.) Poisson then
   carries rho = q (p - n + N_D+ - N_A-), whose derivatives in n and p DEVSIM gets symbolically --
   the extra nonlinearity is rational, so Newton keeps its quadratic convergence.
   Optional Pearson-Bardeen lowering (Pearson & Bardeen, Phys. Rev. 75, 865 (1949)):
              dE_D(N_D) = dE_D0 - alpha_D N_D^(1/3)   (clamped at 0), likewise for acceptors.
   alpha defaults to 0 (off) -- no default value is shipped: no primary-verified alpha exists (the
   Silvaco ATLAS 2004 manual's 3.6e-8 / 3.037e-8 eV cm are a code default, not a measurement).
   UNITS: the field is alpha in eV*m (SI densities), the literature quotes eV*cm; alpha[eV m] =
   1e-2 alpha[eV cm]. Build from a quoted value with IonizationModel.pearson_bardeen(..., alpha_D_eVcm=)
   -- an unconverted eV*cm number would zero the ionization energy above ~3e12 cm^-3 and the clamp would
   hide it as full ionization, so the constructor REFUSES (IonizationUnitsError) any alpha that makes
   the energy vanish below 1e17 cm^-3 (a genuine Pearson-Bardeen alpha vanishes it at ~2-3e18).
   NOT modelled: the Mott transition / impurity-band conduction (Altermatt 2006's fraction b of
   "non-localized" dopants), Fermi-Dirac occupancy of the dopant level at degeneracy (there N_D+ -> N_D
   anyway), excited dopant states. Consequently the model FREEZES OUT WRONGLY above the Mott density
   (Si:P N_crit = 3.74e18 cm^-3): a 1e19 cm^-3 Si:P layer is computed 39 % ionized at 300 K, whereas
   metallic Si:P is fully ionized. `install_option_fields` emits one UnreliableResultWarning per region
   in which a node above the model's N_mott still has a non-zero ionization energy. Use the model for
   freeze-out below ~1e18 cm^-3 (cryogenic or moderately doped regions); above N_mott its ionized
   fraction is not a prediction (Altermatt et al. 2006's model is the planned replacement).

4. TEMPERATURE. `T_K` sets V_t = k T / q for the region (the default None keeps V_t = V_T at T_REF
   exactly). It does NOT rescale the material inputs: n_i, N_c (n_dos_m3), N_v (n_dos_p_m3), the
   mobilities and the lifetimes must be supplied AT T_K by the caller. The builders (2-D
   LayeredDevsimBuilder, 3-D Devsim3DEquilibrium) derive N_c, N_v from the DOS MASSES, so they evaluate
   them at T_K through ONE helper, `band_dos_at_T` (N proportional to T^(3/2)); n_i_m3 stays a direct
   input given at T_K. Not temperature-scaled (v0.14): the BTBT parameters (physics at T_REF; setup_btbt
   warns under T_K) and the electron affinity chi (a heterostack refuses unless the material states the
   temperature chi was quoted at -- LayeredDevsimBuilder(on_chi_temperature=...)).
   v0.15 (opt-in): a TransportModel.band_temperature model (materials.BandTemperature) evaluates E_g, chi
   and, with derive_n_i=True, n_i at T_K from their values at chi_reference_T_K (materials.band_data_at_T;
   anchored Varshni) -- then n_i_m3 is given at that reference T instead, and the chi guard is satisfied.
   v0.15 (opt-in, decision 20): a BTBT set with scale_B_with_gap=True takes B (E_g(T)/E_g(300))^(3/2) from
   that model (tunneling.btbt_for_material in the builders); otherwise setup_btbt still warns.

5. GENERATION. `generation_m3s` adds a uniform external generation G (region parameter `G_ext`,
   rampable with ds.set_parameter) to both continuity equations: U_net = U_SRH + U_A + U_rad - G.
   NOISE: by default noise_analysis counts G as a Poisson process -- photogeneration shot noise, GR
   source 2 G -- which is right when G stands for absorbed light. If G is a deterministic drive or a
   modelling device (a prescribed injection, a continuation aid, a noiseless pump, or source
   statistics handled elsewhere), call noise_analysis.noise_spectrum / terminal_noise with
   external_generation_shot=False: G then sets the operating point but adds no noise of its own.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from dynameta.constants import KB, Q_E
from dynameta.carriers.eq_registry import ds, node_with_derivs as _node_with_derivs, \
    edge_with_derivs as _edge_with_derivs
from dynameta.warning_types import AdvisoryWarning, UnreliableResultWarning

# cm -> m conversions, spelled once (the literature quotes CGS)
_CM3 = 1.0e6          # cm^-3 -> m^-3
_CM6S = 1.0e-12       # cm^6/s -> m^6/s
_CM3S = 1.0e-6        # cm^3/s -> m^3/s


# =====================================================================================================
# parameter dataclasses
# =====================================================================================================

@dataclass(frozen=True)
class AugerModel:
    """Band-to-band Auger U_A = (C_n n + C_p p)(n p - n_ie^2) [SI: C in m^6/s].

    coulomb_enhanced: Altermatt 1997 low-injection enhancement g_eeh(n0), g_ehh(p0) (fixed fields of
    the charge-neutral equilibrium densities). The tanh constants are the paper's; they are exposed
    only so a test can pin them, not as tuning knobs."""
    C_n_m6s: float
    C_p_m6s: float
    coulomb_enhanced: bool = False
    g_eeh_amp: float = 13.0
    g_ehh_amp: float = 7.5
    N_eeh_m3: float = 3.3e17 * _CM3
    N_ehh_m3: float = 7.0e17 * _CM3
    x_eeh: float = 0.66
    x_ehh: float = 0.63
    source: str = ""
    # v0.15 (plan 4.2.2): where the constants are in the source and how they were checked (the BGNModel
    # vocabulary: "primary" = read in the primary text, "secondary" = via another paper or an abstract
    # record only, "inferred", "unverified"). Documentation only; the defaults keep legacy presets as-is.
    locator: str = ""
    verification: str = "unverified"

    def __post_init__(self):
        for nm in ("C_n_m6s", "C_p_m6s"):
            v = float(getattr(self, nm))
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError("AugerModel.{} must be finite and >= 0, got {!r}".format(nm, v))
        if self.C_n_m6s == 0.0 and self.C_p_m6s == 0.0:
            raise ValueError("AugerModel with C_n = C_p = 0 is a no-op; pass auger=None instead")
        if self.verification not in _VERIFICATION:
            raise ValueError("AugerModel.verification must be one of {}".format(_VERIFICATION))


_RADIATIVE_KINDS = ("internal", "effective")
RADIATIVE_INTERNAL_NO_RECYCLING_MESSAGE = (
    "RadiativeModel [{source}] is an INTERNAL coefficient B_int (kind='internal') used as the net "
    "radiative loss with no photon_recycling_factor: photon recycling is ignored, which OVERESTIMATES the "
    "radiative loss of a thick direct-gap layer (GaAs 2 um double heterostructure: B_int / B_eff = 4.2x, "
    "Niemeyer et al. 2019). Pass RadiativeModel.with_photon_recycling(P_r) with the layer's re-absorption "
    "probability, or an effective (kind='effective') B measured on your structure. Region {region!r}.")


@dataclass(frozen=True)
class RadiativeModel:
    """Band-to-band radiative U_rad = B (n p - n_ie^2) [SI: B in m^3/s].

    INTERNAL vs EFFECTIVE B (v0.15, plan 4.2.2, decision 30). In a direct-gap layer a share P_r of the
    emitted photons is re-absorbed (photon recycling) and re-creates a pair, so the NET loss is
        B_eff = B_int (1 - P_r)          (Niemeyer et al., AIP Adv. 9, 045034 (2019), Eq. (1), f = P_r)
    and P_r depends on the layer thickness and cladding (GaAs, 2 um Al0.7Ga0.3As/GaAs DH: P_r = 0.76,
    B_int 3.3e-10 vs B_eff 0.78e-10 cm^3/s -- they differ 4.2x). DynaMeta has no re-absorption
    (photon-transport) model, so a coefficient must say which one it is:
      kind=None        (default) a legacy, unlabelled B, used as given (byte-identical to v0.14).
      kind="effective" B_m3s IS the net coefficient of your structure (already recycling-corrected): used
                       as given. A photon_recycling_factor with it is REFUSED (recycling subtracted twice).
      kind="internal"  B_m3s is the material's intrinsic (recycling-free) B_int. With
                       photon_recycling_factor = P_r in [0, 1) the kernel uses B_int (1 - P_r); without one
                       it uses B_int and the region setup emits ONE AdvisoryWarning per region
                       (RADIATIVE_INTERNAL_NO_RECYCLING_MESSAGE): the radiative loss is overestimated.
    WHICH TO USE: a thin layer or an escape-dominated geometry (P_r ~ 0) -> the internal preset as is (and
    accept the advisory) or with_photon_recycling(0.0); a thick direct-gap absorber / LED / laser layer ->
    with_photon_recycling(P_r) with P_r for YOUR thickness and cladding (from a photon-transport
    calculation or the literature for that structure); a B measured on your own structure -> kind=
    "effective".
    The factor is applied ONCE, at the B_rad parameter write (install_option_terms reads `net_B_m3s`);
    there is no second expression.
    NOISE CAVEAT: with recycling, an emitted-then-reabsorbed photon is a correlated pair event (the pair
    vanishes and reappears elsewhere). noise_analysis counts U_rad as independent Poisson pair processes at
    the NET rate B_eff (n p - n_ie^2), which is an approximation for a recycling layer (exact for P_r = 0).
    locator / verification: as on BGNModel (where in the source; "primary" = read in the primary text)."""
    B_m3s: float
    source: str = ""
    kind: Optional[str] = None
    photon_recycling_factor: Optional[float] = None
    locator: str = ""
    verification: str = "unverified"

    def __post_init__(self):
        v = float(self.B_m3s)
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError("RadiativeModel.B_m3s must be finite and > 0, got {!r}".format(v))
        if self.kind is not None and self.kind not in _RADIATIVE_KINDS:
            raise ValueError("RadiativeModel.kind must be None, 'internal' or 'effective', got {!r}"
                             .format(self.kind))
        if self.verification not in _VERIFICATION:
            raise ValueError("RadiativeModel.verification must be one of {}".format(_VERIFICATION))
        P = self.photon_recycling_factor
        if P is not None:
            if self.kind == "effective":
                raise ValueError(
                    "RadiativeModel: kind='effective' with photon_recycling_factor={!r} would subtract photon "
                    "recycling TWICE -- an effective B is already net of re-absorption. Pass the internal "
                    "B_int with kind='internal' and the factor, or the effective B alone.".format(P))
            if self.kind != "internal":
                raise ValueError(
                    "RadiativeModel: a photon_recycling_factor needs kind='internal' (the factor turns an "
                    "INTERNAL B_int into B_eff = B_int (1 - P_r)); this B is unlabelled (kind=None).")
            p = float(P)
            if not (math.isfinite(p) and 0.0 <= p < 1.0):
                raise ValueError("RadiativeModel.photon_recycling_factor must be in [0, 1), got {!r}"
                                 .format(P))

    @property
    def net_B_m3s(self) -> float:
        """The coefficient the kernel uses [m^3/s]: B_int (1 - P_r) for kind='internal' with a factor, else
        B_m3s unchanged (the SAME float, so a legacy model writes the same B_rad bits)."""
        if self.kind == "internal" and self.photon_recycling_factor is not None:
            return float(self.B_m3s) * (1.0 - float(self.photon_recycling_factor))
        return self.B_m3s

    def with_photon_recycling(self, P_r: float) -> "RadiativeModel":
        """This INTERNAL coefficient with the layer's photon-recycling (re-absorption) probability P_r
        (net B = B_int (1 - P_r)). Refused on an effective or unlabelled B (see the class doc)."""
        import dataclasses
        return dataclasses.replace(self, photon_recycling_factor=float(P_r))


_BGN_BASES = ("total", "donors", "acceptors")
_BGN_DEFAULT_BASIS = {"slotboom": "total", "del_alamo": "donors", "power_log": "total"}
_BGN_OUTSIDE = ("warn", "refuse")
# the accepted spellings of setup_bipolar_region(statistics=...); "fermi_dirac_legacy" is the resolved LABEL of
# "legacy" (accepted back, so a resolved label can be passed on unchanged)
_STATISTICS = ("boltzmann", "fermi_dirac", "legacy", "fermi_dirac_legacy")
_VERIFICATION = ("primary", "secondary", "inferred", "unverified")


class BGNPairingError(ValueError):
    """A PAIRED BGN set (BGNModel.paired_n_i_m3 set) used on a region whose n_i, N_c/N_v, carrier
    statistics or temperature differ from the ones the set was fitted with (check_bgn_pairing). Override
    with BipolarOptions(allow_unpaired_bgn=True) (one AdvisoryWarning instead)."""


class BGNFitPendingError(NotImplementedError):
    """A BGN set with outside_basis="refuse" met nodes of the doping type its fit does not cover (the
    house Si set is n-type only until the p-type constants of Yan & Cuevas, J. Appl. Phys. 116, 194505
    (2014) are read), where the narrowing is not negligible (other species > N_ref)."""


@dataclass(frozen=True)
class BGNModel:
    """Bandgap narrowing dEg(N), N chosen by `doping_basis`.

    kind: "slotboom" -> E_ref [ln(N/N_ref) + sqrt(ln^2(N/N_ref) + C)];
          "del_alamo" -> E_ref ln(N/N_ref) for N > N_ref, else 0;
          "power_log" -> A [ln(N/N_ref)]^b + offset for N > N_ref, else offset (0 for N <= 0), with
                         A = E_ref_eV, b = exponent_b, offset = offset_eV (Yan & Cuevas 2013 Eq. (3)).
    conduction_fraction (gamma): share of dEg that LOWERS E_c (the rest raises E_v); 0.5 = symmetric.
    doping_basis: the density the fit is evaluated in -- "total" (N_D + N_A), "donors" (N_D) or
    "acceptors" (N_A). None (default) = the kind's own basis: "total" for slotboom (Slotboom/Klaassen
    are used for both types) and power_log, "donors" for del_alamo (an n-type-only fit in N_D, TED 1987
    Eq. (11)). Under a one-species basis, nodes where the OTHER species exceeds N_ref have no basis in
    the fit; install_option_fields warns (UnreliableResultWarning) once per region (see
    `unsupported_nodes`).
    outside_basis: what to do with such nodes when they are of the OTHER doping type (p-type nodes of a
    donor-basis fit): "warn" (None, the default: the legacy behaviour above) or "refuse" (raise
    BGNFitPendingError: the house Yan-Cuevas sets, whose p-type constants are pending).
    PAIRING (all None/default = an unpaired legacy set, never checked): paired_n_i_m3 / paired_N_c_m3 /
    paired_N_v_m3 [m^-3], paired_statistics ("fermi_dirac" = a REAL narrowing fitted with FD statistics,
    "boltzmann" = an APPARENT one), paired_T_K: the inputs the fit was made with; check_bgn_pairing
    refuses a region that differs (module doc item 2). locator / verification: where the constants are
    in the source and how they were checked ("primary" = read in the primary text)."""
    kind: str
    E_ref_eV: float
    N_ref_m3: float
    C: float = 0.5
    conduction_fraction: float = 0.5
    source: str = ""
    doping_basis: Optional[str] = None
    exponent_b: float = 1.0
    offset_eV: float = 0.0
    outside_basis: Optional[str] = None
    paired_n_i_m3: Optional[float] = None
    paired_N_c_m3: Optional[float] = None
    paired_N_v_m3: Optional[float] = None
    paired_statistics: Optional[str] = None
    paired_T_K: float = 300.0
    locator: str = ""
    verification: str = "unverified"

    def __post_init__(self):
        if self.kind not in ("slotboom", "del_alamo", "power_log"):
            raise ValueError("BGNModel.kind must be 'slotboom', 'del_alamo' or 'power_log', got {!r}"
                             .format(self.kind))
        if not (self.E_ref_eV > 0.0 and self.N_ref_m3 > 0.0 and self.C >= 0.0):
            raise ValueError("BGNModel needs E_ref_eV > 0, N_ref_m3 > 0, C >= 0")
        if not (0.0 <= self.conduction_fraction <= 1.0):
            raise ValueError("BGNModel.conduction_fraction must lie in [0, 1]")
        if not (math.isfinite(self.exponent_b) and self.exponent_b > 0.0
                and math.isfinite(self.offset_eV) and self.offset_eV >= 0.0):
            raise ValueError("BGNModel needs a finite exponent_b > 0 and a finite offset_eV >= 0")
        if self.kind != "power_log" and (self.exponent_b != 1.0 or self.offset_eV != 0.0):
            raise ValueError("BGNModel.exponent_b / offset_eV apply to kind='power_log' only")
        if self.doping_basis is None:
            object.__setattr__(self, "doping_basis", _BGN_DEFAULT_BASIS[self.kind])
        elif self.doping_basis not in _BGN_BASES:
            raise ValueError("BGNModel.doping_basis must be one of {} (or None), got {!r}".format(
                _BGN_BASES, self.doping_basis))
        if self.outside_basis is None:
            object.__setattr__(self, "outside_basis", "warn")
        elif self.outside_basis not in _BGN_OUTSIDE:
            raise ValueError("BGNModel.outside_basis must be one of {} (or None), got {!r}".format(
                _BGN_OUTSIDE, self.outside_basis))
        if self.paired_statistics is not None and self.paired_statistics not in _STATISTICS:
            raise ValueError("BGNModel.paired_statistics must be one of {} (or None), got {!r}".format(
                _STATISTICS, self.paired_statistics))
        paired = (self.paired_n_i_m3, self.paired_statistics)
        if (paired[0] is None) != (paired[1] is None):
            raise ValueError("BGNModel: a paired set states BOTH paired_n_i_m3 and paired_statistics")
        for nm in ("paired_n_i_m3", "paired_N_c_m3", "paired_N_v_m3"):
            v = getattr(self, nm)
            if v is not None and not (math.isfinite(float(v)) and float(v) > 0.0):
                raise ValueError("BGNModel.{} must be finite and > 0, got {!r}".format(nm, v))
        if self.paired_statistics == "fermi_dirac" and (self.paired_N_c_m3 is None
                                                         or self.paired_N_v_m3 is None):
            raise ValueError("BGNModel: an FD-paired set states paired_N_c_m3 and paired_N_v_m3 (the "
                             "degeneracy n/N_c of its Eq. (1) depends on them)")
        if not (math.isfinite(self.paired_T_K) and self.paired_T_K > 0.0):
            raise ValueError("BGNModel.paired_T_K must be finite and > 0")
        if self.verification not in _VERIFICATION:
            raise ValueError("BGNModel.verification must be one of {}".format(_VERIFICATION))

    @property
    def paired(self) -> bool:
        """True for a set that states the n_i (statistics, T) it was fitted with -- check_bgn_pairing
        enforces those; False for the legacy unpaired sets (never checked)."""
        return self.paired_n_i_m3 is not None

    def refused_nodes(self, N_D_m3, N_A_m3):
        """Boolean mask of the nodes outside_basis="refuse" refuses: nodes of the OTHER doping type
        (N_A > N_D under "donors", N_D > N_A under "acceptors") whose other species exceeds N_ref, i.e.
        where the missing fit would give a non-negligible narrowing. All False otherwise."""
        shape = np.shape(np.asarray(N_D_m3))
        if self.outside_basis != "refuse" or self.doping_basis == "total":
            return np.zeros(shape, dtype=bool)
        nd = np.asarray(N_D_m3, dtype=np.float64)
        na = np.asarray(N_A_m3, dtype=np.float64)
        other, own = (na, nd) if self.doping_basis == "donors" else (nd, na)
        return (other > own) & (other > self.N_ref_m3)

    def basis_density_m3(self, N_D_m3, N_A_m3):
        """The density [m^-3] the fit is evaluated in, per node: N_D + N_A ("total"; the same
        arithmetic as before the basis existed), N_D ("donors") or N_A ("acceptors")."""
        if self.doping_basis == "donors":
            return np.asarray(N_D_m3, dtype=np.float64)
        if self.doping_basis == "acceptors":
            return np.asarray(N_A_m3, dtype=np.float64)
        return N_D_m3 + N_A_m3

    def unsupported_nodes(self, N_D_m3, N_A_m3):
        """Boolean mask of the nodes outside the fit's basis: under "donors", N_A > N_ref (p-type or
        compensated nodes heavy enough that narrowing would matter -- the fit was made on
        uncompensated n-type material only); under "acceptors", N_D > N_ref; never under "total"."""
        if self.doping_basis == "donors":
            return np.asarray(N_A_m3, dtype=np.float64) > self.N_ref_m3
        if self.doping_basis == "acceptors":
            return np.asarray(N_D_m3, dtype=np.float64) > self.N_ref_m3
        return np.zeros(np.shape(np.asarray(N_D_m3)), dtype=bool)

    def delta_Eg_eV(self, N_total_m3):
        """dEg [eV] for the basis density N [m^-3] (array-broadcasting; 0 for N <= 0). The caller
        passes `basis_density_m3(N_D, N_A)` -- equilibrium_fields does."""
        N = np.asarray(N_total_m3, dtype=np.float64)
        out = np.zeros_like(N)
        pos = N > 0.0
        x = np.log(np.where(pos, N, 1.0) / self.N_ref_m3)
        if self.kind == "slotboom":
            r = np.sqrt(x * x + self.C)
            # x + r loses everything to cancellation for x << 0 (light doping): use C/(r - x) there.
            val = np.where(x >= 0.0, x + r, self.C / np.where(x < 0.0, r - x, 1.0))
            out = np.where(pos, self.E_ref_eV * val, 0.0)
        elif self.kind == "power_log":
            # A x^b on x > 0 only (x <= 0 is clamped: negative for b = 1, non-monotone for b = 3); the
            # offset applies to every doped node. One log and one pow: round-off ~ b * 1 ulp of x.
            xb = np.power(np.where(x > 0.0, x, 0.0), self.exponent_b)
            out = np.where(pos, np.where(x > 0.0, self.E_ref_eV * xb, 0.0) + self.offset_eV, 0.0)
        else:
            out = np.where(pos & (x > 0.0), self.E_ref_eV * x, 0.0)
        return out


class IonizationUnitsError(ValueError):
    """A Pearson-Bardeen alpha that zeroes the ionization energy below 1e17 cm^-3: almost certainly a
    literature eV*cm value passed into the eV*m field (IonizationModel.alpha_*_eVm)."""


# Pearson-Bardeen units guard: the density at which dE0 - alpha N^(1/3) reaches 0 must not lie below
# this. A genuine alpha puts it at ~1.8-3e18 cm^-3 (ATLAS 2004 example values), 1.3 decades above; an
# unconverted eV*cm alpha puts it 6 decades lower (3.1e-8 with 45 meV: 3.1e12 cm^-3).
_PB_MIN_ZERO_M3 = 1.0e17 * _CM3
_PB_MIN_ZERO_CBRT = _PB_MIN_ZERO_M3 ** (1.0 / 3.0)     # [m^-1]
_EVCM_TO_EVM = 1.0e-2           # alpha [eV m] = 1e-2 alpha [eV cm]  (N^(1/3) [m^-1] = 100 N^(1/3) [cm^-1])

# Mott (metal-insulator) densities [m^-3] used by the freeze-out advisory, Si defaults:
#   donors    3.74e18 cm^-3, Si:P (Altermatt et al., J. Appl. Phys. 100, 113714 (2006) / SISPAD 2006;
#             del Alamo & Swanson, TED 1987: "about 3.7e18") -- the lowest documented donor value, so
#             As (Altermatt's N_b proxy 9e18) and Sb are advised conservatively early;
#   acceptors 4.5e18 cm^-3, Si:B -- Altermatt 2006's N_b used as the documented PROXY (the B Mott
#             density itself is unverified).
_SI_MOTT_D_M3 = 3.74e18 * _CM3
_SI_MOTT_A_M3 = 4.5e18 * _CM3


@dataclass(frozen=True)
class IonizationModel:
    """Incomplete ionization of a single donor and a single acceptor species (Boltzmann occupancy).

    dE_D_eV / dE_A_eV: ionization energies E_c - E_D and E_A - E_v [eV]; g_D / g_A: degeneracy factors.
    alpha_D_eVm / alpha_A_eVm: Pearson-Bardeen lowering dE(N) = dE0 - alpha N^(1/3) [eV m]; 0 = off.
      A literature alpha is quoted in eV*cm: build with `IonizationModel.pearson_bardeen(...,
      alpha_D_eVcm=...)`, which converts. Any alpha that makes dE vanish below 1e17 cm^-3 raises
      IonizationUnitsError (the eV*cm-in-the-eV*m-field trap; it would otherwise read as full
      ionization through the clamp).
    N_mott_D_m3 / N_mott_A_m3: Mott densities [m^-3] for the freeze-out WARNING only (an
      UnreliableResultWarning; the occupancy is unchanged): Si:P 3.74e18 and the Si:B proxy 4.5e18
      cm^-3 by default; set them for another dopant or material, or math.inf to silence it.

    Use this model for freeze-out in non-degenerate regions (cryogenic T, or N below ~1e18 cm^-3). It
    has no Mott transition, so above N_mott its ionized fraction is not a prediction (1e19 cm^-3 Si:P
    comes out 39 % ionized at 300 K; metallic Si:P is fully ionized)."""
    dE_D_eV: float
    dE_A_eV: float
    g_D: float = 2.0
    g_A: float = 4.0
    alpha_D_eVm: float = 0.0
    alpha_A_eVm: float = 0.0
    source: str = ""
    N_mott_D_m3: float = _SI_MOTT_D_M3
    N_mott_A_m3: float = _SI_MOTT_A_M3

    def __post_init__(self):
        if not (self.dE_D_eV >= 0.0 and self.dE_A_eV >= 0.0 and self.g_D > 0.0 and self.g_A > 0.0
                and self.alpha_D_eVm >= 0.0 and self.alpha_A_eVm >= 0.0
                and math.isfinite(self.alpha_D_eVm) and math.isfinite(self.alpha_A_eVm)):
            raise ValueError("IonizationModel needs dE >= 0, g > 0, finite alpha >= 0")
        if not (self.N_mott_D_m3 > 0.0 and self.N_mott_A_m3 > 0.0):
            raise ValueError("IonizationModel.N_mott_*_m3 must be > 0 (math.inf disables the advisory)")
        for sp, dE0, alpha in (("D", self.dE_D_eV, self.alpha_D_eVm), ("A", self.dE_A_eV, self.alpha_A_eVm)):
            # dE0 - alpha N^(1/3) = 0 at N = (dE0/alpha)^3; compared in N^(1/3) (no overflow at tiny alpha)
            if alpha > 0.0 and dE0 > 0.0 and dE0 / alpha < _PB_MIN_ZERO_CBRT:
                raise IonizationUnitsError(
                    "IonizationModel.alpha_{sp}_eVm = {a!r} zeroes the ionization energy at N = "
                    "{nz:.3g} cm^-3 (< 1e17 cm^-3): alpha looks like eV*cm; this field is eV*m = "
                    "1e-2 x eV*cm (alpha_{sp}_eVm = {conv!r}). Use IonizationModel.pearson_bardeen("
                    "..., alpha_{sp}_eVcm=...) to pass a literature value.".format(
                        sp=sp, a=alpha, nz=(dE0 / alpha) ** 3 / _CM3, conv=alpha * _EVCM_TO_EVM))

    @classmethod
    def pearson_bardeen(cls, dE_D_eV: float, dE_A_eV: float, *, alpha_D_eVcm: float = 0.0,
                        alpha_A_eVcm: float = 0.0, **kw) -> "IonizationModel":
        """Build from literature-unit Pearson-Bardeen coefficients: dE(N) = dE0 - alpha N^(1/3) with
        alpha in eV*cm and N in cm^-3 (how papers and TCAD manuals quote it), converted once to the
        eV*m field. Other IonizationModel fields pass through `kw`. The units guard still applies."""
        return cls(dE_D_eV=dE_D_eV, dE_A_eV=dE_A_eV, alpha_D_eVm=float(alpha_D_eVcm) * _EVCM_TO_EVM,
                   alpha_A_eVm=float(alpha_A_eVcm) * _EVCM_TO_EVM, **kw)

    def ionization_energies_eV(self, N_D_m3, N_A_m3):
        """(dE_D(N_D), dE_A(N_A)) [eV] with the Pearson-Bardeen lowering, clamped at 0 (per node)."""
        dED = np.maximum(self.dE_D_eV - self.alpha_D_eVm * np.cbrt(N_D_m3), 0.0)
        dEA = np.maximum(self.dE_A_eV - self.alpha_A_eVm * np.cbrt(N_A_m3), 0.0)
        return dED, dEA


@dataclass(frozen=True)
class BipolarOptions:
    """The opt-in bundle setup_bipolar_region(options=...) consumes. All fields default to None (off)."""
    auger: Optional[AugerModel] = None
    radiative: Optional[RadiativeModel] = None
    bgn: Optional[BGNModel] = None
    ionization: Optional[IonizationModel] = None
    T_K: Optional[float] = None
    # Uniform external generation G [m^-3 s^-1] (module doc item 5). Noise (user decision 2026-11-04):
    # noise_analysis treats it as POISSON photogeneration (GR source 2 G, shot noise 2 q I_ph when
    # every pair is collected) unless noise_spectrum / terminal_noise get external_generation_shot=
    # False -- use False when G is a deterministic drive / modelling device, not a photon stream.
    generation_m3s: Optional[float] = None
    # v0.15 (decision 27): a PAIRED bgn (BGNModel.paired_n_i_m3) whose n_i / N_c, N_v / statistics / T
    # differ from the region's raises BGNPairingError; True downgrades it to one AdvisoryWarning per
    # region. Use it with your OWN n_i(T), N_c(T), N_v(T) at T != 300 K (decision 29), or to reproduce a
    # paper that paired differently -- never to silence a mismatch you did not mean.
    allow_unpaired_bgn: bool = False

    def __post_init__(self):
        if self.T_K is not None and not (math.isfinite(float(self.T_K)) and float(self.T_K) > 0.0):
            raise ValueError("BipolarOptions.T_K must be finite and > 0, got {!r}".format(self.T_K))
        if self.generation_m3s is not None and not math.isfinite(float(self.generation_m3s)):
            raise ValueError("BipolarOptions.generation_m3s must be finite")

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.auger, self.radiative, self.bgn, self.ionization,
                                           self.T_K, self.generation_m3s))

    @property
    def alters_equilibrium(self) -> bool:
        """True when the charge-neutral equilibrium differs from the SRH-only model's (BGN or
        incomplete ionization): the contacts and the seed then pin the numpy fields instead of the
        closed-form CELEC/CHOLE strings."""
        return self.bgn is not None or self.ionization is not None

    def V_t(self) -> float:
        """Thermal voltage [V] at T_K (V_T at T_REF when T_K is None -- the SAME float)."""
        from dynameta.constants import V_T
        return V_T if self.T_K is None else KB * float(self.T_K) / Q_E


def band_dos_at_T(m_n_kg: float, m_p_kg: Optional[float], options: Optional[BipolarOptions]):
    """(N_c, N_v) [m^-3] of a bipolar region from its DOS masses at the region's temperature: options.T_K
    when set, else T_REF -- the ONE temperature source, the same one that sets the region's V_t.
    m_p_kg None -> N_v = N_c (the builders' long-standing default). THE single rule for both builders
    (2-D LayeredDevsimBuilder._setup_bipolar_region, 3-D Devsim3DEquilibrium body + gate frame; v0.14
    row 3.2-7: the 3-D path evaluated the masses at T_REF under T_K). The T_K=None call is the
    T_REF-default effective_dos_m3 call, bit-identical to what both builders computed before."""
    from dynameta.carriers.physics_equilibrium import effective_dos_m3
    T = getattr(options, "T_K", None)
    if T is None:
        Nc = effective_dos_m3(float(m_n_kg))
        Nv = effective_dos_m3(float(m_p_kg)) if m_p_kg else Nc
    else:
        Nc = effective_dos_m3(float(m_n_kg), float(T))
        Nv = effective_dos_m3(float(m_p_kg), float(T)) if m_p_kg else Nc
    return Nc, Nv


# =====================================================================================================
# literature presets (Si, 300 K). Opt-in ONLY: nothing applies them per material automatically.
# =====================================================================================================

AUGER_SI_DZIEWIOR_SCHMID_1977 = AugerModel(
    C_n_m6s=2.8e-31 * _CM6S, C_p_m6s=0.99e-31 * _CM6S,
    source="Dziewior & Schmid, Appl. Phys. Lett. 31, 346 (1977); Si 300 K")
AUGER_SI_ALTERMATT_1997 = AugerModel(
    C_n_m6s=2.8e-31 * _CM6S, C_p_m6s=0.99e-31 * _CM6S, coulomb_enhanced=True,
    source="Altermatt et al., J. Appl. Phys. 82, 4938 (1997); Coulomb-enhanced, Si 300 K")
RADIATIVE_SI_TRUPKE_2003 = RadiativeModel(
    B_m3s=9.5e-15 * _CM3S, source="Trupke et al., J. Appl. Phys. 94, 4930 (2003); Si 300 K")
# The three BGN sets are LEGACY and UNPAIRED: none was fitted with n_i = 1.0e10 cm^-3 (module doc,
# item 2: old Slotboom ~1.44e10, del Alamo ~1.36-1.45e10, Klaassen 1.0e10 or 1.08e10 unverified; the
# modern n_i = 9.65e9 belongs with FD statistics and a real BGN).
BGN_SI_SLOTBOOM_1977 = BGNModel(
    kind="slotboom", E_ref_eV=9.0e-3, N_ref_m3=1.0e17 * _CM3, C=0.5,
    source="Slotboom & de Graaff, Solid-State Electron. 19, 857 (1976); IEEE TED 24, 1123 (1977); "
           "fitted on p-type transistor bases (N_A), reference n_i ~1.44e10 cm^-3 [inferred]; unpaired")
BGN_SI_KLAASSEN_1992 = BGNModel(
    kind="slotboom", E_ref_eV=6.92e-3, N_ref_m3=1.3e17 * _CM3, C=0.5,
    source="Klaassen, Slotboom & de Graaff, Solid-State Electron. 35, 125 (1992) [constants via "
           "secondary sources]; reference n_i 1.0e10 or 1.08e10 cm^-3 [unverified]; unpaired")
BGN_SI_DEL_ALAMO_1985 = BGNModel(
    kind="del_alamo", E_ref_eV=18.7e-3, N_ref_m3=7.0e17 * _CM3, doping_basis="donors",
    source="del Alamo & Swanson, IEEE TED ED-34, 1580 (1987), Eq. (11); del Alamo, Swirhun & Swanson, "
           "IEDM 1985, p. 290, Eq. (7); n-type (Si:P) only, in N_D; reference n_i ~1.36-1.45e10 cm^-3 "
           "[inferred]; unpaired")
# Yan & Cuevas 2013 (v0.15, user decision 26 -- the HOUSE Si narrowing). Both sets were extracted from
# the SAME 24 phosphorus diffusions (J0 matched with Klaassen mobility, n_i = 9.65e9 cm^-3 from Misiakos &
# Tsamakis 1993, N_c = 2.86e19 / N_v = 3.1e19 cm^-3 at 300 K from Green 1990 -- Sec. I and III.B), so each is
# PAIRED with those inputs and its statistics. n-type ONLY (in N_D): the p-type companion, J. Appl. Phys.
# 116, 194505 (2014), has not been read, so p-type nodes above N_ref are refused (outside_basis).
# Use the FD set with statistics="fermi_dirac" (the house bundle), the Boltzmann set with "boltzmann".
_YC2013 = ("Yan & Cuevas, J. Appl. Phys. 114, 044508 (2013); n-type (Si:P) only, in N_D; fitted with "
           "n_i = 9.65e9, N_c = 2.86e19, N_v = 3.1e19 cm^-3 at 300 K and Klaassen mobility; p-type "
           "constants (JAP 116, 194505 (2014)) pending")
BGN_SI_YAN_CUEVAS_2013_FD = BGNModel(
    kind="power_log", E_ref_eV=0.042e-3, N_ref_m3=1.0e14 * _CM3, C=0.0, exponent_b=3.0, offset_eV=0.0,
    doping_basis="donors", outside_basis="refuse",
    paired_n_i_m3=9.65e9 * _CM3, paired_N_c_m3=2.86e19 * _CM3, paired_N_v_m3=3.1e19 * _CM3,
    paired_statistics="fermi_dirac", paired_T_K=300.0,
    locator="Eq. (3), Table I row 'Fermi-Dirac based BGN (this work)': A = 0.042 +/- 0.003 meV, "
            "N_ref = 1e14 cm^-3, b = 3, C = 0; inputs Sec. I (N_c, N_v; Ref. 3) and Sec. III.B (n_i; Ref. 32)",
    verification="primary", source=_YC2013 + "; REAL narrowing (Fermi-Dirac statistics)")
BGN_SI_YAN_CUEVAS_2013_BOLTZMANN = BGNModel(
    kind="power_log", E_ref_eV=12.96e-3, N_ref_m3=1.0e17 * _CM3, C=0.0, exponent_b=1.0, offset_eV=0.0,
    doping_basis="donors", outside_basis="refuse",
    paired_n_i_m3=9.65e9 * _CM3, paired_statistics="boltzmann", paired_T_K=300.0,
    locator="Eq. (3), Table I row 'Boltzmann based apparent BGN (this work)': A = 12.96 +/- 0.70 meV, "
            "N_ref = 1e17 cm^-3, b = 1, C = 0; n_i Sec. III.B (Ref. 32)",
    verification="primary", source=_YC2013 + "; APPARENT narrowing (Boltzmann statistics)")
# Ionization energies in Si [eV] (ground state 1s(A1) binding energies). Donors: P and As as tabulated in
# Sze & Ng, "Physics of Semiconductor Devices", 3rd ed. (2007), ch. 1 (45, 54 meV below E_c; the
# spectroscopic values of Ramdas & Rodriguez Table 4 are 45.59 / 53.76 meV, < 1 meV away, kept for the
# existing gates); Sb 42.74 meV from the spectroscopic table, Ramdas & Rodriguez, Rep. Prog. Phys. 44,
# 1297 (1981), Table 4 (Sze & Ng's 39 meV is an older, superseded value: 3.7 meV = 0.14 kT too shallow).
# Acceptor: B 45 meV above E_v (Sze & Ng).
SI_DONOR_IONIZATION_EV = {"P": 0.045, "As": 0.054, "Sb": 0.04274}
SI_ACCEPTOR_IONIZATION_EV = {"B": 0.045}
IONIZATION_SI_P_B = IonizationModel(dE_D_eV=0.045, dE_A_eV=0.045,
                                    source="Sze & Ng (2007): Si:P 45 meV, Si:B 45 meV; g_D=2, g_A=4")


# =====================================================================================================
# carrier statistics (v0.15, decision 28) and the ONE BGN pairing check (decisions 27, 29)
# =====================================================================================================

def resolve_statistics(statistics: Optional[str], fd_enhancement: bool) -> str:
    """The region's statistics LABEL from setup_bipolar_region's (statistics, fd_enhancement) -- module doc,
    STATISTICS:
      None (default, v0.15.0)  -> "fermi_dirac" with the FD g-factor (fd_enhancement=True, its default), or
                                  "boltzmann" with fd_enhancement=False (as before);
      "fermi_dirac"            -> "fermi_dirac" (needs fd_enhancement=True);
      "boltzmann"              -> "boltzmann" (needs fd_enhancement=False);
      "legacy"                 -> "fermi_dirac_legacy" (needs fd_enhancement=True): the v0.14 model, FD
                                  g-factor transport (edge-averaged g) with the Boltzmann product n p = n_ie^2
                                  in recombination and contacts -- for reproducing v0.14 numbers only.
    The label itself is accepted back (idempotent). A choice that contradicts fd_enhancement raises."""
    if statistics is None:
        return "fermi_dirac" if fd_enhancement else "boltzmann"
    if statistics not in _STATISTICS:
        raise ValueError("statistics must be None, 'fermi_dirac', 'boltzmann' or 'legacy', got {!r}".format(
            statistics))
    label = "fermi_dirac_legacy" if statistics == "legacy" else statistics
    if (label != "boltzmann") != bool(fd_enhancement):
        raise ValueError("statistics={!r} contradicts fd_enhancement={!r}: 'fermi_dirac' and 'legacy' need the "
                         "FD g-factor (fd_enhancement=True), 'boltzmann' needs fd_enhancement=False"
                         .format(statistics, fd_enhancement))
    return label


_PAIR_REL_TOL = 1.0e-3      # n_i / N_c / N_v are quoted to 3 significant figures; 1e-3 of n_i = 0.05 meV
_PAIR_T_TOL_K = 0.5         # decision 29: a fit is a 300 K statement


def check_bgn_pairing(bgn: Optional[BGNModel], *, n_i_m3: float, N_c_m3: float, N_v_m3: float,
                      statistics: str, T_K: Optional[float], allow: bool = False,
                      region: str = "") -> Optional[str]:
    """THE pairing check of a PAIRED BGN set (BGNModel.paired) against a region; equilibrium_fields calls
    it, so the region setup (install_option_fields) and the 3-D gate frame (ohmic_built_in_offset_V)
    share it. `statistics` is the resolve_statistics label, T_K the region temperature (None = T_REF).

    Refused (BGNPairingError) when (a) |n_i/n_i,paired - 1| > 1e-3 -- the message gives n_ie^2's error
    and its gap equivalent kT ln((n_i/n_i,paired)^2) at the paired T (1.84 meV for 1.0e10 against
    9.65e9); (a') for an FD-paired set the same on N_c and N_v (the degeneracy n/N_c of Eq. (1)); (b) the
    region statistics differ from the set's ("fermi_dirac_legacy" matches neither: it pins the Boltzmann
    product the FD set's Eq. (1) corrects, and it double-counts an apparent set's degeneracy); (c)
    |T - paired_T_K| > 0.5 K (no n_i(T) model: the fit is a 300 K statement). allow=True
    (BipolarOptions.allow_unpaired_bgn) RETURNS the message instead (the caller warns once); an unpaired
    or absent set returns None."""
    if bgn is None or not bgn.paired:
        return None
    from dynameta.constants import T_REF
    kT = KB * float(bgn.paired_T_K) / Q_E
    bad = []
    n_i = float(n_i_m3)
    ref = float(bgn.paired_n_i_m3)
    if abs(n_i / ref - 1.0) > _PAIR_REL_TOL:
        bad.append("n_i = {:.4g} cm^-3 but the set was fitted with {:.4g} cm^-3: n_ie^2 is off by {:+.1f} %, "
                   "a gap error of {:+.2f} meV".format(n_i / _CM3, ref / _CM3,
                                                     100.0 * ((n_i / ref) ** 2 - 1.0),
                                                     1e3 * kT * math.log((n_i / ref) ** 2)))
    if bgn.paired_statistics == "fermi_dirac":
        for nm, v, r in (("N_c", N_c_m3, bgn.paired_N_c_m3), ("N_v", N_v_m3, bgn.paired_N_v_m3)):
            if abs(float(v) / float(r) - 1.0) > _PAIR_REL_TOL:
                bad.append("{} = {:.4g} cm^-3 but the set was fitted with {:.4g} cm^-3 (the degeneracy "
                           "n/N_c of its Fermi-Dirac n_ie)".format(nm, float(v) / _CM3, float(r) / _CM3))
    if statistics != bgn.paired_statistics:
        why = {("fermi_dirac", "fermi_dirac_legacy"):
               "the legacy statistics pin the Boltzmann product n0 p0 = n_ie^2 in the contacts and in every "
               "recombination term, not the Fermi-Dirac n0 p0 = n_ie^2 exp(-Delta_n - Delta_p) the set was "
               "fitted with; pass statistics='fermi_dirac' (TransportModel.carrier_statistics)",
               ("fermi_dirac", "boltzmann"):
               "a REAL narrowing with Boltzmann statistics drops the degeneracy correction; pass "
               "statistics='fermi_dirac' (TransportModel.carrier_statistics)",
               ("boltzmann", "fermi_dirac_legacy"):
               "an APPARENT (Boltzmann-extracted) narrowing on the FD g-factor counts the degeneracy twice; "
               "pass statistics='boltzmann' (TransportModel.carrier_statistics)",
               ("boltzmann", "fermi_dirac"):
               "an APPARENT (Boltzmann-extracted) narrowing with FD statistics counts the degeneracy twice; "
               "pass statistics='boltzmann' (TransportModel.carrier_statistics)"}
        bad.append("the set is paired with {} statistics, the region runs {!r}: {}".format(
            bgn.paired_statistics, statistics, why[(bgn.paired_statistics, statistics)]))
    T = T_REF if T_K is None else float(T_K)
    if abs(T - float(bgn.paired_T_K)) > _PAIR_T_TOL_K:
        bad.append("the region runs at T = {:.6g} K but the fit is a {:.6g} K statement (no n_i(T) model "
                   "is shipped)".format(T, float(bgn.paired_T_K)))
    if not bad:
        return None
    msg = ("region {!r}: the PAIRED bandgap-narrowing set [{}] does not match the region: {}. Fix the "
           "region inputs, or set BipolarOptions(allow_unpaired_bgn=True) to proceed with your own n_i "
           "(and N_c, N_v) at this T.".format(region, bgn.source.split(";")[0], "; ".join(bad)))
    if allow:
        return msg
    raise BGNPairingError(msg)


def _refuse_outside_basis(bgn: BGNModel, N_D, N_A) -> None:
    """Raise BGNFitPendingError if an outside_basis="refuse" set meets nodes of the other doping type
    above N_ref (BGNModel.refused_nodes) -- the house Si sets on p-type nodes."""
    bad = bgn.refused_nodes(N_D, N_A)
    if np.any(bad):
        other = np.asarray(N_A if bgn.doping_basis == "donors" else N_D, dtype=np.float64)
        raise BGNFitPendingError(
            "the {} bandgap-narrowing set [{}] covers {} material only; {} node(s) are {} with up to {:.3g} "
            "cm^-3 > N_ref = {:.3g} cm^-3, where the narrowing is not negligible but its constants are not "
            "available (for Si: the p-type fit of Yan & Cuevas, J. Appl. Phys. 116, 194505 (2014), has not "
            "been read). Give the p-type region its own material with an explicit BGN choice, or wait for "
            "the 2014 constants.".format(
                bgn.kind, bgn.source.split(";")[0],
                "n-type" if bgn.doping_basis == "donors" else "p-type", int(np.count_nonzero(bad)),
                "p-type" if bgn.doping_basis == "donors" else "n-type",
                float(np.max(other[bad])) / _CM3, bgn.N_ref_m3 / _CM3))


# =====================================================================================================
# the ONE numpy kernel for every fixed (doping-only) field
# =====================================================================================================

_ROOT_ITERS = 200       # bisection steps in ln n; the bracket spans < 400 e-folds -> < 1e-58 after 200


def _neutral_root(N_D, N_A, n_ie, n1D, p1A, g_D, g_A, ionization: bool, traps=(), V_t=None,
                  N_c=None, N_v=None):
    """Charge-neutral equilibrium electron density n0 solving
         f(n) = n - n_ie^2/n - N_D+(n) + N_A-(n_ie^2/n) - Q_t(n)/q = 0
    (N_D+ = N_D, N_A- = N_A when ionization is False). `traps` = ((E_t - E_i [eV], N_t, kind), ...)
    are CHARGED bulk SRH levels with their equilibrium (Fermi) occupancy f_t = n/(n + n1),
    n1 = n_ie exp(E_t/kT) -- the steady-state SRH occupancy of physics_traps at n p = n_ie^2, the same
    n1 the rate uses: Q_t/q = -N_t f_t (acceptor-like), +N_t (1 - f_t) (donor-like). This is the
    charge-neutrality condition with trapped charge of Shockley & Read (1952) / Sze & Ng ch. 1, i.e.
    the brentq oracle of the D1c compensation gate. A band-referenced level (a 4-tuple, see
    physics_traps.equilibrium_trap_levels) takes its n1 from the band edge instead -- N_c exp(-E/kT), or
    n_ie^2/(N_v exp(-E/kT)) -- through physics_traps.equilibrium_n1, the same conversion the rate uses
    (N_c / N_v are then required). Every term of f is STRICTLY increasing in n
    (-Q_t/q too: +N_t n/(n + n1) and -N_t n1/(n + n1)), so the root is unique; bisection in ln n on
         [n_ie^2/S, S],   S = N_A + N_D + sum N_t + n_ie
    (f <= 0 at the lower end, >= 0 at the upper -- see the audit doc) converges unconditionally.
    Vectorized over arrays of equal shape. Returns (n0, p0 = n_ie^2/n0)."""
    N_D, N_A, n_ie, n1D, p1A = (np.asarray(a, dtype=np.float64) for a in (N_D, N_A, n_ie, n1D, p1A))
    ni2 = n_ie * n_ie
    if traps:
        hi = np.log(N_A + N_D + sum(float(t[1]) for t in traps) + n_ie)
        from dynameta.carriers.physics_traps import equilibrium_n1
        tl = [(float(t[1]), equilibrium_n1(t, n_ie, V_t, N_c, N_v), t[2]) for t in traps]
    else:
        hi = np.log(N_A + N_D + n_ie)
        tl = []
    lo = np.log(ni2) - hi

    if ionization:
        def f0(ln_n):
            n = np.exp(ln_n)
            p = ni2 / n
            return n - p - N_D / (1.0 + g_D * n / n1D) + N_A / (1.0 + g_A * p / p1A)
    else:
        def f0(ln_n):
            n = np.exp(ln_n)
            return n - ni2 / n - N_D + N_A
    if tl:
        def f(ln_n):                                         # + (-Q_t/q), increasing in n
            n = np.exp(ln_n)
            out = f0(ln_n)
            for N, n1, kind in tl:
                out = out + (N * n / (n + n1) if kind == "acceptor" else -N * n1 / (n + n1))
            return out
    else:
        f = f0                                               # the trap-free root, arithmetic unchanged
    for _ in range(_ROOT_ITERS):
        mid = 0.5 * (lo + hi)
        up = f(mid) > 0.0
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
        if np.all(hi - lo < 1e-15):
            break
    ln_n = 0.5 * (lo + hi)
    n0 = np.exp(ln_n)
    return n0, ni2 / n0


_ACT_NEWTON_ITERS = 100     # Newton steps for the activity inverse (quadratic: ~6-10 are used)


def fd_activity_inverse(t):
    """u >= 0 with L(u) = ln u + Delta(u) = t (vectorized): the inverse of the FD-enhanced transport's
    log-activity (Delta = physics_bipolar_dd.fd_potential_shift, the integral of the SAME rational g the
    edge models use). Newton in y = ln u on h(y) = y + Delta(e^y) - t, h'(y) = g(e^y) >= 1: h is convex
    and increasing, and h(t) = Delta(e^t) >= 0, so Newton started at y = t never overshoots and
    decreases monotonically to the root (stopped when the step is below 1e-15 in y)."""
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    from dynameta.carriers.einstein import g_einstein
    t = np.asarray(t, dtype=np.float64)
    y = t.copy()
    for _ in range(_ACT_NEWTON_ITERS):
        u = np.exp(y)
        step = (y + np.asarray(fd_potential_shift(u), dtype=np.float64) - t) / g_einstein(u)
        y = y - step
        if np.all(np.abs(step) < 1e-15):
            break
    return np.exp(y)


def _neutral_root_fd(N_D, N_A, n_ie0, n1D, p1A, g_D, g_A, ionization: bool, N_c: float, N_v: float,
                     traps=(), V_t=None):
    """Charge-neutral equilibrium of the FD detailed balance (statistics="fermi_dirac"): n0 solving
         f(n) = n - p(n) - N_D+(n) + N_A-(p(n)) - Q_t(n)/q = 0,
         L(p/N_v) = ln(n_ie0^2/(N_c N_v)) - L(n/N_c),   L(u) = ln u + Delta(u),
    i.e. eta_n + eta_p = -E_g,eff/kT in the transport's own activity: n0 p0 = n_ie0^2 exp(-Delta(n0/N_c)
    - Delta(p0/N_v)) -- Yan & Cuevas 2013 Eq. (1) with the code's Delta (n_ie0 = n_i exp(dEg/2kT)).
    `traps` (CHARGED bulk levels, as _neutral_root): the EXACT FD equilibrium occupancy of the level,
    f_t = 1/(1 + exp((E_t - E_F)/kT)) = n/(n + gamma_n(n) n1_B), gamma_n = exp(-Delta(n/N_c)) and n1_B the
    Boltzmann-frame n1 of the level (physics_traps.equilibrium_n1 with n_ie0) -- the steady-state
    occupancy physics_traps.srh_occupancy_expr(fd=True) takes at n p = n_ie^2, for any capture ratio.
    p(n) <= n_ie0^2/n is decreasing and n/gamma_n(n) increasing, so every term of f (the trap terms too)
    is strictly increasing and the root unique. Bracket (bisection in ln n): f >= 0 at S = N_A + N_D +
    sum N_t + n_ie0 (p <= n_ie0^2/S <= n_ie0); f <= 0 at n_ie0^2/S exp(-Delta(S/N_v) - 1), where p > S.
    Returns (n0, p0)."""
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    N_D, N_A, n_ie0, n1D, p1A = (np.asarray(a, dtype=np.float64) for a in (N_D, N_A, n_ie0, n1D, p1A))
    t = np.log(n_ie0 * n_ie0 / (N_c * N_v))
    S = N_A + N_D + n_ie0
    if traps:
        S = S + sum(float(tr[1]) for tr in traps)
        from dynameta.carriers.physics_traps import equilibrium_n1
        tl = [(float(tr[1]), equilibrium_n1(tr, n_ie0, V_t, N_c, N_v), tr[2]) for tr in traps]
    else:
        tl = []
    hi = np.log(S)
    lo = 2.0 * np.log(n_ie0) - hi - np.asarray(fd_potential_shift(S / N_v), dtype=np.float64) - 1.0

    def p_of(ln_n):
        u = np.exp(ln_n) / N_c
        return N_v * fd_activity_inverse(t - (np.log(u) + np.asarray(fd_potential_shift(u), np.float64)))

    def f(ln_n):
        n = np.exp(ln_n)
        p = p_of(ln_n)
        if ionization:
            out = n - p - N_D / (1.0 + g_D * n / n1D) + N_A / (1.0 + g_A * p / p1A)
        else:
            out = n - p - N_D + N_A
        if tl:
            g1 = np.exp(-np.asarray(fd_potential_shift(n / N_c), dtype=np.float64))   # gamma_n(n)
            for N, n1, kind in tl:                           # + (-Q_t/q), increasing in n
                out = out + (N * n / (n + g1 * n1) if kind == "acceptor" else -N * g1 * n1 / (n + g1 * n1))
        return out

    for _ in range(_ROOT_ITERS):
        mid = 0.5 * (lo + hi)
        up = f(mid) > 0.0
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
        if np.all(hi - lo < 1e-15):
            break
    ln_n = 0.5 * (lo + hi)
    return np.exp(ln_n), p_of(ln_n)


def equilibrium_fields(net_doping_m3, *, n_i_m3: float, N_c_m3: float, N_v_m3: float,
                       options: BipolarOptions, fd_enhancement: bool,
                       donors_m3=None, acceptors_m3=None, traps=(),
                       statistics: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Every fixed (doping-only) field the options need, per node. Pure numpy. THE single source shared
    by the DEVSIM setup and by ohmic_built_in_offset_V (the 3-D gate frame).

    `statistics` (v0.15; resolve_statistics with fd_enhancement): None keeps the v0.14 arithmetic
    BIT-identically (Boltzmann-product root n0 p0 = n_ie^2); "fermi_dirac" solves the FD detailed-balance
    root n0 p0 = n_ie0^2 exp(-Delta(n0/N_c) - Delta(p0/N_v)) (_neutral_root_fd, with the CHARGED trap
    levels at their exact FD occupancy n/(n + gamma_n n1)). The ONE BGN pairing check (check_bgn_pairing) and the outside-basis refusal
    (BGNFitPendingError) run here, so the region setup and the 3-D gate frame cannot disagree; with
    BipolarOptions.allow_unpaired_bgn the pairing message is returned as 'pairing_advisory' (None when
    the set matches or is unpaired) for install_option_fields to warn once.

    `traps`: CHARGED bulk trap levels ((E_t - E_i [eV], N_t [m^-3], kind), ...; physics_traps.
    equilibrium_trap_levels) whose equilibrium trapped charge enters the charge-neutral root (the ONE
    contact rule for BGN, incomplete ionization and trapped charge). () = no trap term (the arithmetic
    is then exactly the trap-free root's).

    Donors/Acceptors default to the net-doping split (N_D = max(N, 0), N_A = max(-N, 0)) -- a
    compensated region must pass them explicitly (or define 'Donors'/'Acceptors' node models).

    Returns arrays: V_t (scalar), dEg_eV, D_c_V, D_v_V, n_ie (the equilibrium effective intrinsic
    density sqrt(n0 p0); under "fermi_dirac" it carries the degeneracy factors), n_ie0 (n_i exp(dEg/2kT),
    the narrowing alone), N_D, N_A, n1D, p1A, n0, p0, fd_n, fd_p, psi_eq (the potential the ohmic contact
    pins, V), seed_shift (V_t units, see physics_bipolar_dd.equilibrium_seed_psi_expr), g_eeh, g_ehh;
    and pairing_advisory (str or None)."""
    N = np.asarray(net_doping_m3, dtype=np.float64)
    N_D = np.where(N > 0.0, N, 0.0) if donors_m3 is None else np.broadcast_to(
        np.asarray(donors_m3, dtype=np.float64), N.shape)
    N_A = np.where(N < 0.0, -N, 0.0) if acceptors_m3 is None else np.broadcast_to(
        np.asarray(acceptors_m3, dtype=np.float64), N.shape)
    Vt = options.V_t()
    n_i = float(n_i_m3)
    label = resolve_statistics(statistics, fd_enhancement)
    fdc = label == "fermi_dirac"
    # --- v0.15: the ONE pairing check + the outside-basis refusal (raise, or the advisory text) ---
    pairing_msg = check_bgn_pairing(options.bgn, n_i_m3=n_i, N_c_m3=float(N_c_m3), N_v_m3=float(N_v_m3),
                                    statistics=label, T_K=options.T_K,
                                    allow=bool(options.allow_unpaired_bgn))
    # --- BGN: dEg, band-edge split, n_ie ---
    if options.bgn is not None:
        _refuse_outside_basis(options.bgn, N_D, N_A)
        dEg = options.bgn.delta_Eg_eV(options.bgn.basis_density_m3(N_D, N_A))
        gam = float(options.bgn.conduction_fraction)
    else:
        dEg, gam = np.zeros_like(N), 0.5
    D_c = gam * dEg                                          # [V] (eV / e)
    D_v = dEg - D_c
    n_ie = n_i * np.exp(dEg / (2.0 * Vt))
    if not np.all(np.isfinite(n_ie * n_ie)) or not np.all(n_ie * n_ie > 0.0):
        raise ValueError("equilibrium_fields: n_ie^2 under/overflows float64 (n_i = {:.3e}, T = {}); "
                         "the region is outside the representable range".format(n_i, options.T_K))
    # --- ionization reference densities ---
    ion = options.ionization
    if ion is not None:
        dED, dEA = ion.ionization_energies_eV(N_D, N_A)
        n1D = float(N_c_m3) * np.exp(-dED / Vt)
        p1A = float(N_v_m3) * np.exp(-dEA / Vt)
        g_D, g_A = float(ion.g_D), float(ion.g_A)
    else:
        n1D = np.full_like(N, np.inf)
        p1A = np.full_like(N, np.inf)
        g_D = g_A = 1.0
    # --- charge-neutral equilibrium (unique doping combinations only: a profile has few) ---
    key = np.stack([N_D.ravel(), N_A.ravel(), n_ie.ravel(), np.broadcast_to(n1D, N.shape).ravel(),
                    np.broadcast_to(p1A, N.shape).ravel()], axis=1)
    uq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    if fdc:
        n0u, p0u = _neutral_root_fd(uq[:, 0], uq[:, 1], uq[:, 2], uq[:, 3], uq[:, 4], g_D, g_A,
                                    ion is not None, N_c=float(N_c_m3), N_v=float(N_v_m3),
                                    traps=tuple(traps or ()), V_t=Vt)
    else:
        n0u, p0u = _neutral_root(uq[:, 0], uq[:, 1], uq[:, 2], uq[:, 3], uq[:, 4], g_D, g_A,
                                 ion is not None, traps=tuple(traps or ()), V_t=Vt,
                                 N_c=float(N_c_m3), N_v=float(N_v_m3))
    n0 = n0u[inv].reshape(N.shape)
    p0 = p0u[inv].reshape(N.shape)
    # --- the potential the ohmic contact pins (Boltzmann frame + FD degeneracy + band-edge shift) ---
    if fd_enhancement:
        from dynameta.carriers.physics_bipolar_dd import fd_potential_shift   # devsim-importing module
        fd_n = np.asarray(fd_potential_shift(n0 / float(N_c_m3)), dtype=np.float64).reshape(N.shape)
        fd_p = np.asarray(fd_potential_shift(p0 / float(N_v_m3)), dtype=np.float64).reshape(N.shape)
    else:
        fd_n = np.zeros_like(N)
        fd_p = np.zeros_like(N)
    ntype = N > 0.0                                          # the SAME branch the contact ifelse takes
    psi_eq = np.where(ntype, Vt * (np.log(n0 / n_i) + fd_n) - D_c,
                      -Vt * (np.log(p0 / n_i) + fd_p) + D_v)
    sgn = np.where(ntype, 1.0, -1.0)
    seed_shift = sgn * (psi_eq / Vt - np.log(n0 / n_i))
    # --- Coulomb-enhanced Auger factors (fixed: functions of the equilibrium densities) ---
    a = options.auger
    if a is not None and a.coulomb_enhanced:
        g_eeh = 1.0 + a.g_eeh_amp * (1.0 - np.tanh(np.power(n0 / a.N_eeh_m3, a.x_eeh)))
        g_ehh = 1.0 + a.g_ehh_amp * (1.0 - np.tanh(np.power(p0 / a.N_ehh_m3, a.x_ehh)))
    else:
        g_eeh = np.ones_like(N)
        g_ehh = np.ones_like(N)
    n_ie_eq = n_ie * np.exp(-0.5 * (fd_n + fd_p)) if fdc else n_ie
    return dict(V_t=Vt, dEg_eV=dEg, D_c_V=D_c, D_v_V=D_v, n_ie=n_ie_eq, n_ie0=n_ie, N_D=N_D, N_A=N_A,
                n1D=np.broadcast_to(n1D, N.shape), p1A=np.broadcast_to(p1A, N.shape),
                n0=n0, p0=p0, fd_n=fd_n, fd_p=fd_p, psi_eq=psi_eq, seed_shift=seed_shift,
                g_eeh=g_eeh, g_ehh=g_ehh, pairing_advisory=pairing_msg)


# =====================================================================================================
# DEVSIM expression builders (one per kernel)
# =====================================================================================================

def _ni2(bgn: bool) -> str:
    return "n_ie^2" if bgn else "n_i^2"


def srh_expr(bgn: bool) -> str:
    """The region's base SRH channel with the effective intrinsic density -- a NAME for the ONE SRH
    kernel physics_traps.srh_rate_expr (v0.14 integration: the BGN substitution n_i^2 -> n_ie^2,
    n1, p1 -> n1 n_ie/n_i, p1 n_ie/n_i lives there, shared with every TrapLevel and interface-trap
    bin). bgn=False is the shipped string CHARACTER-FOR-CHARACTER; bgn=True the D1b string."""
    from dynameta.carriers.physics_traps import srh_rate_expr
    return srh_rate_expr(bgn=bool(bgn))


# Name of the external-generation term in the additive mechanism (noise_analysis keys its
# `external_generation_shot` switch on it).
GEXT_TERM = "gext"


def recombination_exprs(options: BipolarOptions, *, pair: bool = False,
                        nie: Optional[bool] = None) -> Dict[str, str]:
    """{node model name: expression} of the extra recombination terms (Auger, radiative). pair=True
    gives each process's PAIR rate G + R (n p - n_ie^2 -> n p + n_ie^2: the detailed-balance split the
    noise GR source 2 (G + R) uses) -- one builder for the rate and its pair rate. `nie`: reference the
    region's n_ie model (True) or n_i (False); None = whether the options carry a BGN (the v0.14 rule;
    statistics="fermi_dirac" passes True: its n_ie carries the degeneracy factors)."""
    bgn = options.bgn is not None if nie is None else bool(nie)
    sgn = "+" if pair else "-"
    out = {}
    if options.auger is not None:
        cn = "AugerCn*AugerGeeh" if options.auger.coulomb_enhanced else "AugerCn"
        cp = "AugerCp*AugerGehh" if options.auger.coulomb_enhanced else "AugerCp"
        out["UAuger"] = "({cn}*Electrons + {cp}*Holes)*(Electrons*Holes {s} {ni2})".format(
            cn=cn, cp=cp, s=sgn, ni2=_ni2(bgn))
    if options.radiative is not None:
        out["URad"] = "B_rad*(Electrons*Holes {} {})".format(sgn, _ni2(bgn))
    return out


# additive-term names of the R26 rates (physics_bipolar_dd.add_recombination_term)
_TERM_NAMES = {"UAuger": "auger", "URad": "radiative"}


def ionized_net_doping_expr() -> str:
    """N_D+ - N_A- with each species' occupancy set by its own carrier (Boltzmann reference densities)."""
    return ("IonDonors/(1 + IonGD*Electrons/IonN1D) - IonAcceptors/(1 + IonGA*Holes/IonP1A)")


# =====================================================================================================
# DEVSIM installation
# =====================================================================================================

def _set_field(device: str, region: str, name: str, values) -> None:
    ds.node_solution(device=device, region=region, name=name)
    ds.set_node_values(device=device, region=region, name=name,
                       values=[float(v) for v in np.asarray(values, dtype=np.float64).ravel()])


def _maybe_node_values(device: str, region: str, name: str):
    if name in ds.get_node_model_list(device=device, region=region):
        return np.asarray(ds.get_node_model_values(device=device, region=region, name=name),
                          dtype=np.float64)
    return None


# A degenerate region's FD built-in correction above this (in V_t units) makes apparent-BGN double
# counting non-negligible (~ this many V_t of the contact frame); warn above it.
_BGN_FD_WARN_VT = 0.05


def bgn_basis_advisory(region: str, bgn: BGNModel, N_D_m3, N_A_m3) -> Optional[str]:
    """The advisory text when some node of `region` lies outside the BGN fit's doping basis
    (BGNModel.unsupported_nodes: e.g. del Alamo -- an n-type-only fit in N_D -- on p-type or
    compensated nodes with N_A > N_ref), else None. Pure numpy; install_option_fields emits it as an
    UnreliableResultWarning (the fit is used outside its validity range; the result there may be
    wrong), one per region."""
    bad = bgn.unsupported_nodes(N_D_m3, N_A_m3)
    if not np.any(bad):
        return None
    fit, other_name, other_type = (("N_D", "N_A", "p-type") if bgn.doping_basis == "donors"
                                   else ("N_A", "N_D", "n-type"))
    other = np.asarray(N_A_m3 if bgn.doping_basis == "donors" else N_D_m3, dtype=np.float64)
    return ("install_bipolar_options: region {!r} uses the {} bandgap-narrowing fit, which is made in "
            "{} only, but {} of {} nodes carry {} up to {:.3g} cm^-3 > N_ref = {:.3g} cm^-3 "
            "({} or compensated material, where the fit has no basis). Those nodes get dEg from {} "
            "alone (0 on an uncompensated {} node). For a heavily doped {} region use a "
            "Slotboom/Klaassen set (doping_basis='total').".format(
                region, bgn.kind, fit, int(np.count_nonzero(bad)), int(bad.size),
                other_name, float(np.max(other[bad])) / _CM3, bgn.N_ref_m3 / _CM3, other_type, fit,
                other_type, other_type))


def mott_advisory(region: str, ion: IonizationModel, N_D_m3, N_A_m3, n0_m3, p0_m3, n1D_m3,
                  p1A_m3) -> Optional[str]:
    """The advisory text when some node of `region` is doped above the ionization model's Mott density
    (IonizationModel.N_mott_*_m3) AND still has a non-zero ionization energy -- the single-level
    Boltzmann model then freezes out where real (metallic) material is fully ionized -- else None.
    Quotes the worst equilibrium non-ionized fraction g n0/(n1 + g n0). Pure numpy; install_option_fields
    emits it as an UnreliableResultWarning (the ionized fraction there may be wrong), one per region."""
    N_D = np.asarray(N_D_m3, dtype=np.float64)
    N_A = np.asarray(N_A_m3, dtype=np.float64)
    dED, dEA = ion.ionization_energies_eV(N_D, N_A)
    parts = []
    for name, N, Nm, dE, frac in (
            ("N_D", N_D, ion.N_mott_D_m3, dED, ion.g_D * n0_m3 / (n1D_m3 + ion.g_D * n0_m3)),
            ("N_A", N_A, ion.N_mott_A_m3, dEA, ion.g_A * p0_m3 / (p1A_m3 + ion.g_A * p0_m3))):
        bad = (N > Nm) & (dE > 0.0)
        if np.any(bad):
            parts.append("{} up to {:.3g} cm^-3 > N_mott = {:.3g} cm^-3 on {} nodes, computed up to "
                         "{:.0f} % NON-ionized".format(name, float(np.max(N[bad])) / _CM3, Nm / _CM3,
                                                      int(np.count_nonzero(bad)),
                                                      100.0 * float(np.max(np.asarray(frac)[bad]))))
    if not parts:
        return None
    return ("install_bipolar_options: region {!r}: incomplete ionization above the Mott density ({}). "
            "The single-level Boltzmann model has no Mott transition, so it freezes out there although "
            "the metallic material is fully ionized; its ionized fraction is not a prediction above "
            "N_mott. Use it below ~1e18 cm^-3, or set N_mott_D_m3/N_mott_A_m3 for this dopant "
            "(math.inf silences this note).".format(region, "; ".join(parts)))


def _region_fields(device: str, region: str, options: Optional[BipolarOptions], *,
                   fd_enhancement: bool, traps=(), statistics: Optional[str] = None) -> Dict[str, np.ndarray]:
    """equilibrium_fields on a region's own parameters and doping (options None -> the inert bundle:
    only the charged trap levels change the trap-free equilibrium)."""
    o = options if options is not None else BipolarOptions()
    n_i = float(ds.get_parameter(device=device, region=region, name="n_i"))
    N_c = float(ds.get_parameter(device=device, region=region, name="N_dos"))
    N_v = float(ds.get_parameter(device=device, region=region, name="N_dos_p"))
    net = np.asarray(ds.get_node_model_values(device=device, region=region, name="NetDoping"),
                     dtype=np.float64)
    don = _maybe_node_values(device, region, "Donors")
    acc = _maybe_node_values(device, region, "Acceptors")
    if (don is None) != (acc is None):
        raise ValueError("install_bipolar_options: region {!r} defines only one of the 'Donors' / "
                         "'Acceptors' node models; define both (or neither: the NetDoping split)."
                         .format(region))
    return equilibrium_fields(net, n_i_m3=n_i, N_c_m3=N_c, N_v_m3=N_v, options=o,
                              fd_enhancement=fd_enhancement, donors_m3=don, acceptors_m3=acc,
                              traps=traps, statistics=statistics)


def install_fd_detailed_balance(device: str, region: str, base: str) -> str:
    """statistics="fermi_dirac": the solution-dependent effective intrinsic density
         n_ie = base * sqrt(gamma_n gamma_p),   gamma = (c/N) exp(-L(c/N)) = exp(-Delta(c/N)),
    with L = ln u + Delta(u) the transport's own log-activity in CLOSED form
    (physics_heterojunction.setup_log_activity, idempotent: a heterointerface on the region shares it)
    and `base` the fixed part ("NieBgn" = n_i exp(dEg/2kT) under BGN, else the parameter "n_i"). Every
    recombination string referencing n_ie (SRH, Auger, radiative, bulk trap levels, the noise pair
    rates) then vanishes where n p = n_ie^2 -- exactly the zero-current equilibrium of the FD-enhanced
    SG transport (d psi_n = g V_t d ln n => ln n + Delta - psi_n/V_t = const, likewise for holes).
    DEVSIM differentiates a model reference through '<model>:<var>', so the derivatives are given in
    closed form without cancellation: d ln gamma/dc = -(g - 1)/c = -(GA + GC s)/(N (1 + GB s + GD s^2)),
    s = (c/N)^(1/3) (the rational g fit, carriers.einstein; physics_heterojunction.fd_delta_derivative,
    shared with the FD-mode transport's quasi-field). Returns "n_ie"."""
    from dynameta.carriers.physics_heterojunction import fd_delta_derivative, setup_log_activity
    gam = []
    for var, dos, tag in (("Electrons", "N_dos", "N"), ("Holes", "N_dos_p", "P")):
        L = setup_log_activity(device, region, var, dos, fd=True)
        g_name = "FDGamma" + tag
        dl_name = "FDdLnGamma" + tag
        ds.node_model(device=device, region=region, name=g_name,
                      equation="({v}/{d})*exp(-{L})".format(v=var, d=dos, L=L))
        ds.node_model(device=device, region=region, name=dl_name,   # d ln gamma/dc = -d Delta/dc (one copy)
                      equation="-" + fd_delta_derivative(device, region, var, dos))
        ds.node_model(device=device, region=region, name="{}:{}".format(g_name, var),
                      equation="{}*{}".format(g_name, dl_name))
        gam.append((dl_name, var))
    ds.node_model(device=device, region=region, name="n_ie",
                  equation="{}*pow(FDGammaN*FDGammaP, 0.5)".format(base))
    for dl_name, var in gam:
        ds.node_model(device=device, region=region, name="n_ie:{}".format(var),
                      equation="0.5*n_ie*{}".format(dl_name))
    # v0.15 FD-mode completion: the per-band activities base*gamma_n, base*gamma_p (FDNiN * FDNiP = n_ie^2).
    # They are THE densities every occupancy-bearing kernel takes (physics_traps.srh_reference_densities
    # fd=True: n1 -> gamma_n n1, p1 -> gamma_p p1; tunneling.hurkx_d_expr: the Fermi functions at E_ie):
    # n/(n + FDNiN) IS 1/(1 + exp((E_ie - E_Fn)/kT)) with E_Fn the transport's own FD quasi-Fermi level.
    for tag, (dl_name, var) in zip("NP", gam):
        nm = "FDNi" + tag
        ds.node_model(device=device, region=region, name=nm, equation="{}*FDGamma{}".format(base, tag))
        ds.node_model(device=device, region=region, name="{}:{}".format(nm, var),
                      equation="{}*{}".format(nm, dl_name))
    return "n_ie"


def install_option_fields(device: str, region: str, options: BipolarOptions, *,
                          fd_enhancement: bool, statistics: Optional[str] = None) -> None:
    """Stage 1 of the options (setup_bipolar_region calls it after the currents, BEFORE the
    recombination models, when options.active or statistics="fermi_dirac"): V_t at T_K; the pairing
    check's advisory (allow_unpaired_bgn); under BGN the n_ie field and the band-edge quasi-fields
    re-defining vdiff_g / vdiff_gp (a TRANSPORT change, not an additive term); under statistics=
    "fermi_dirac" the solution-dependent n_ie (install_fd_detailed_balance); under incomplete ionization
    the reference fields, IonizedNetDoping and the Poisson composite rebuilt on it
    (physics_bipolar_dd.rebuild_poisson_charge, which keeps every QTerm_*)."""
    from dynameta.carriers.physics_bipolar_dd import rebuild_poisson_charge
    o = options
    fdc = resolve_statistics(statistics, fd_enhancement) == "fermi_dirac"
    if o.T_K is not None:
        ds.set_parameter(device=device, region=region, name="V_t", value=o.V_t())
    f = _region_fields(device, region, o, fd_enhancement=fd_enhancement, statistics=statistics)
    if f["pairing_advisory"] is not None:        # allow_unpaired_bgn: the refusal, downgraded, once
        warnings.warn("install_bipolar_options: " + f["pairing_advisory"], AdvisoryWarning, stacklevel=4)

    # --- BGN: n_ie field, band-edge quasi-fields in both SG currents ---
    if o.bgn is not None:
        if (fd_enhancement and o.bgn.paired_statistics != "fermi_dirac"
                and float(np.max(np.maximum(f["fd_n"], f["fd_p"]))) > _BGN_FD_WARN_VT):
            warnings.warn(
                "install_bipolar_options: region {!r} combines an APPARENT bandgap-narrowing model "
                "({}) with the Fermi-Dirac g-factor on a degenerate doping (FD built-in correction up "
                "to {:.3g} V_t). Apparent BGN was extracted with Boltzmann statistics, so part of the "
                "degeneracy is counted twice; use statistics='boltzmann' (fd_enhancement=False) with "
                "apparent BGN, or accept the overlap knowingly.".format(region, o.bgn.kind,
                                                float(np.max(np.maximum(f["fd_n"], f["fd_p"])))),
                AdvisoryWarning, stacklevel=4)
        msg = bgn_basis_advisory(region, o.bgn, f["N_D"], f["N_A"])
        if msg is not None:                      # the fit is used outside its validity range
            warnings.warn(msg, UnreliableResultWarning, stacklevel=4)
        _set_field(device, region, "NieBgn" if fdc else "n_ie", f["n_ie0"] if fdc else f["n_ie"])
        _set_field(device, region, "BgnDc", f["D_c_V"])
        _set_field(device, region, "BgnDv", f["D_v_V"])
        for nm in ("BgnDc", "BgnDv"):
            ds.edge_from_node_model(device=device, region=region, node_model=nm)
        qn = qp = ""
        if fdc:                                  # the FD quasi-field (physics_heterojunction.fd_quasi_field_expr)
            from dynameta.carriers.physics_heterojunction import fd_quasi_field_expr
            qn = " + " + fd_quasi_field_expr(device, region, "Electrons", "N_dos")
            qp = " - " + fd_quasi_field_expr(device, region, "Holes", "N_dos_p")
        _edge_with_derivs(device, region, "vdiff_g",
                          "(vdiff + (BgnDc@n0 - BgnDc@n1)/V_t) / g_enh" + qn, ("Potential", "Electrons"))
        _edge_with_derivs(device, region, "vdiff_gp",
                          "(vdiff - (BgnDv@n0 - BgnDv@n1)/V_t) / g_enh_p" + qp, ("Potential", "Holes"))
    if fdc:                                      # FD detailed balance: n_ie from the solution
        install_fd_detailed_balance(device, region, "NieBgn" if o.bgn is not None else "n_i")

    # --- incomplete ionization: Poisson charge with N_D+ - N_A- ---
    if o.ionization is not None:
        msg = mott_advisory(region, o.ionization, f["N_D"], f["N_A"], f["n0"], f["p0"], f["n1D"],
                            f["p1A"])
        if msg is not None:                      # the model is used outside its validity range
            warnings.warn(msg, UnreliableResultWarning, stacklevel=4)
        ds.set_parameter(device=device, region=region, name="IonGD", value=float(o.ionization.g_D))
        ds.set_parameter(device=device, region=region, name="IonGA", value=float(o.ionization.g_A))
        _set_field(device, region, "IonDonors", f["N_D"])
        _set_field(device, region, "IonAcceptors", f["N_A"])
        _set_field(device, region, "IonN1D", f["n1D"])
        _set_field(device, region, "IonP1A", f["p1A"])
        _node_with_derivs(device, region, "IonizedNetDoping", ionized_net_doping_expr(),
                          ("Electrons", "Holes"))
        rebuild_poisson_charge(device, region)


def install_option_terms(device: str, region: str, options: Optional[BipolarOptions], *,
                         fd_enhancement: bool, traps=(), statistics: Optional[str] = None) -> None:
    """Stage 2 (setup_bipolar_region, after USRH and the trap levels): Auger / radiative / external
    generation as ADDITIVE terms (physics_bipolar_dd.add_recombination_term, each with its noise pair
    rate; the diagnostics UAuger / URad keep their names), and -- when the equilibrium itself changes
    (BGN, incomplete ionization, or CHARGED trap levels `traps`) -- the charge-neutral fields the
    ohmic contacts and the seed pin (NeutralElectrons/Holes, EquilibriumPotential, SeedPsiShift).
    options may be None (charged traps alone). statistics="fermi_dirac" always installs those fields
    (the FD detailed-balance root) and points the recombination strings at the n_ie model."""
    from dynameta.carriers.physics_bipolar_dd import add_recombination_term
    o = options
    fdc = resolve_statistics(statistics, fd_enhancement) == "fermi_dirac"
    if o is not None:
        if o.auger is not None:
            ds.set_parameter(device=device, region=region, name="AugerCn", value=float(o.auger.C_n_m6s))
            ds.set_parameter(device=device, region=region, name="AugerCp", value=float(o.auger.C_p_m6s))
        if o.radiative is not None:
            # v0.15: the ONE place the photon-recycling factor is applied (RadiativeModel.net_B_m3s)
            if o.radiative.kind == "internal" and o.radiative.photon_recycling_factor is None:
                warnings.warn(RADIATIVE_INTERNAL_NO_RECYCLING_MESSAGE.format(
                    source=o.radiative.source or "unlabelled", region=region), AdvisoryWarning, stacklevel=3)
            ds.set_parameter(device=device, region=region, name="B_rad", value=float(o.radiative.net_B_m3s))
        f = None
        if o.auger is not None and o.auger.coulomb_enhanced:
            f = _region_fields(device, region, o, fd_enhancement=fd_enhancement, traps=traps,
                               statistics=statistics)
            _set_field(device, region, "AugerGeeh", f["g_eeh"])
            _set_field(device, region, "AugerGehh", f["g_ehh"])
        nie = True if fdc else None
        pairs = recombination_exprs(o, pair=True, nie=nie)
        for nm, ex in recombination_exprs(o, nie=nie).items():
            _node_with_derivs(device, region, nm, ex, ("Electrons", "Holes"))     # diagnostic name
            add_recombination_term(device, region, _TERM_NAMES[nm], nm, pair=pairs[nm],
                                   derivs={"Electrons": nm + ":Electrons", "Holes": nm + ":Holes"})
        if o.generation_m3s is not None:
            ds.set_parameter(device=device, region=region, name="G_ext", value=float(o.generation_m3s))
            # Poisson generation (photogeneration shot noise) unless the noise call says otherwise
            add_recombination_term(device, region, GEXT_TERM, "-G_ext", pair="G_ext")
    if (o is not None and o.alters_equilibrium) or traps or fdc:
        f = _region_fields(device, region, o, fd_enhancement=fd_enhancement, traps=traps,
                           statistics=statistics)
        _set_field(device, region, "NeutralElectrons", f["n0"])
        _set_field(device, region, "NeutralHoles", f["p0"])
        _set_field(device, region, "EquilibriumPotential", f["psi_eq"])
        _set_field(device, region, "SeedPsiShift", f["seed_shift"])


def options_contact_models(device: str, region: "str | None") -> bool:
    """True when `region` carries the option-equilibrium fields, i.e. its ohmic contacts must pin
    NeutralElectrons / NeutralHoles / EquilibriumPotential instead of the closed-form strings."""
    if region is None:
        return False
    return "EquilibriumPotential" in ds.get_node_model_list(device=device, region=region)


__all__ = ["AugerModel", "RadiativeModel", "BGNModel", "IonizationModel", "IonizationUnitsError",
           "BGNPairingError", "BGNFitPendingError", "BGN_SI_YAN_CUEVAS_2013_FD",
           "BGN_SI_YAN_CUEVAS_2013_BOLTZMANN", "resolve_statistics", "check_bgn_pairing",
           "fd_activity_inverse", "install_fd_detailed_balance",
           "BipolarOptions", "band_dos_at_T",
           "AUGER_SI_DZIEWIOR_SCHMID_1977", "AUGER_SI_ALTERMATT_1997", "RADIATIVE_SI_TRUPKE_2003",
           "BGN_SI_SLOTBOOM_1977", "BGN_SI_KLAASSEN_1992", "BGN_SI_DEL_ALAMO_1985",
           "SI_DONOR_IONIZATION_EV", "SI_ACCEPTOR_IONIZATION_EV", "IONIZATION_SI_P_B",
           "equilibrium_fields", "srh_expr", "recombination_exprs", "ionized_net_doping_expr",
           "install_option_fields", "install_option_terms", "options_contact_models", "GEXT_TERM",
           "bgn_basis_advisory", "mott_advisory"]
