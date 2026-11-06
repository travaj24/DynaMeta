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
   These are APPARENT narrowings: they were extracted assuming Boltzmann statistics and a specific
   reference n_i, so (a) they must be paired with the n_i the user passes (state it with the result),
   and (b) on a DEGENERATE region with the FD g-factor on, the degeneracy is partly counted twice --
   `install_option_fields` warns when that correction is non-negligible. Schenk's 1998 many-body
   model (carrier-density-dependent band edges) is NOT implemented (see the audit doc, sec. Limits).

   REFERENCE n_i OF EACH SET (the pairing; these presets are LEGACY and UNPAIRED -- nothing checks the
   region's n_i against them). None of them was fitted with n_i = 1.0e10 cm^-3:
     old Slotboom   ~1.44e10 cm^-3 at 300 K [inferred from the source; the Stanford re-fit quotes
                    n_i(295 K) = 9.565e9]
     del Alamo      ~1.36-1.45e10 cm^-3 [inferred, TED 1987 Table II]
     Klaassen 1992  1.0e10 or 1.08e10 (Green 1990) -- not verified in the primary
   The modern consensus n_i(300 K) is 9.65e9 cm^-3 (Altermatt et al., J. Appl. Phys. 93, 1598 (2003);
   Misiakos & Tsamakis 1993: 9.7 +/- 0.1e9), and it belongs with Fermi-Dirac statistics and a REAL
   (not apparent) narrowing. Pairing old Slotboom or del Alamo with 9.65e9 under-predicts n_ie^2 by
   ~2.3x (~21 meV of gap). Which n_i to pass for which calculation: the n_i the chosen set was fitted
   with when you want the set's own n_ie (e.g. an emitter J0 against that paper's data); a FD-paired
   modern set when you want the consensus n_i (not shipped yet).

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

    def __post_init__(self):
        for nm in ("C_n_m6s", "C_p_m6s"):
            v = float(getattr(self, nm))
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError("AugerModel.{} must be finite and >= 0, got {!r}".format(nm, v))
        if self.C_n_m6s == 0.0 and self.C_p_m6s == 0.0:
            raise ValueError("AugerModel with C_n = C_p = 0 is a no-op; pass auger=None instead")


@dataclass(frozen=True)
class RadiativeModel:
    """Band-to-band radiative U_rad = B (n p - n_ie^2) [SI: B in m^3/s]."""
    B_m3s: float
    source: str = ""

    def __post_init__(self):
        v = float(self.B_m3s)
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError("RadiativeModel.B_m3s must be finite and > 0, got {!r}".format(v))


_BGN_BASES = ("total", "donors", "acceptors")
_BGN_DEFAULT_BASIS = {"slotboom": "total", "del_alamo": "donors"}


@dataclass(frozen=True)
class BGNModel:
    """Apparent bandgap narrowing dEg(N), N chosen by `doping_basis`.

    kind: "slotboom" -> E_ref [ln(N/N_ref) + sqrt(ln^2(N/N_ref) + C)];
          "del_alamo" -> E_ref ln(N/N_ref) for N > N_ref, else 0.
    conduction_fraction (gamma): share of dEg that LOWERS E_c (the rest raises E_v); 0.5 = symmetric.
    doping_basis: the density the fit is evaluated in -- "total" (N_D + N_A), "donors" (N_D) or
    "acceptors" (N_A). None (default) = the kind's own basis: "total" for slotboom (Slotboom/Klaassen
    are used for both types), "donors" for del_alamo (an n-type-only fit in N_D, TED 1987 Eq. (11)).
    Under a one-species basis, nodes where the OTHER species exceeds N_ref have no basis in the fit;
    install_option_fields warns (UnreliableResultWarning) once per region (see `unsupported_nodes`)."""
    kind: str
    E_ref_eV: float
    N_ref_m3: float
    C: float = 0.5
    conduction_fraction: float = 0.5
    source: str = ""
    doping_basis: Optional[str] = None

    def __post_init__(self):
        if self.kind not in ("slotboom", "del_alamo"):
            raise ValueError("BGNModel.kind must be 'slotboom' or 'del_alamo', got {!r}".format(self.kind))
        if not (self.E_ref_eV > 0.0 and self.N_ref_m3 > 0.0 and self.C >= 0.0):
            raise ValueError("BGNModel needs E_ref_eV > 0, N_ref_m3 > 0, C >= 0")
        if not (0.0 <= self.conduction_fraction <= 1.0):
            raise ValueError("BGNModel.conduction_fraction must lie in [0, 1]")
        if self.doping_basis is None:
            object.__setattr__(self, "doping_basis", _BGN_DEFAULT_BASIS[self.kind])
        elif self.doping_basis not in _BGN_BASES:
            raise ValueError("BGNModel.doping_basis must be one of {} (or None), got {!r}".format(
                _BGN_BASES, self.doping_basis))

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


def equilibrium_fields(net_doping_m3, *, n_i_m3: float, N_c_m3: float, N_v_m3: float,
                       options: BipolarOptions, fd_enhancement: bool,
                       donors_m3=None, acceptors_m3=None, traps=()) -> Dict[str, np.ndarray]:
    """Every fixed (doping-only) field the options need, per node. Pure numpy. THE single source shared
    by the DEVSIM setup and by ohmic_built_in_offset_V (the 3-D gate frame).

    `traps`: CHARGED bulk trap levels ((E_t - E_i [eV], N_t [m^-3], kind), ...; physics_traps.
    equilibrium_trap_levels) whose equilibrium trapped charge enters the charge-neutral root (the ONE
    contact rule for BGN, incomplete ionization and trapped charge). () = no trap term (the arithmetic
    is then exactly the trap-free root's).

    Donors/Acceptors default to the net-doping split (N_D = max(N, 0), N_A = max(-N, 0)) -- a
    compensated region must pass them explicitly (or define 'Donors'/'Acceptors' node models).

    Returns arrays: V_t (scalar), dEg_eV, D_c_V, D_v_V, n_ie, N_D, N_A, n1D, p1A, n0, p0, fd_n, fd_p,
    psi_eq (the potential the ohmic contact pins, V), seed_shift (V_t units, see physics_bipolar_dd.
    equilibrium_seed_psi_expr), g_eeh, g_ehh."""
    N = np.asarray(net_doping_m3, dtype=np.float64)
    N_D = np.where(N > 0.0, N, 0.0) if donors_m3 is None else np.broadcast_to(
        np.asarray(donors_m3, dtype=np.float64), N.shape)
    N_A = np.where(N < 0.0, -N, 0.0) if acceptors_m3 is None else np.broadcast_to(
        np.asarray(acceptors_m3, dtype=np.float64), N.shape)
    Vt = options.V_t()
    n_i = float(n_i_m3)
    # --- BGN: dEg, band-edge split, n_ie ---
    if options.bgn is not None:
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
    return dict(V_t=Vt, dEg_eV=dEg, D_c_V=D_c, D_v_V=D_v, n_ie=n_ie, N_D=N_D, N_A=N_A,
                n1D=np.broadcast_to(n1D, N.shape), p1A=np.broadcast_to(p1A, N.shape),
                n0=n0, p0=p0, fd_n=fd_n, fd_p=fd_p, psi_eq=psi_eq, seed_shift=seed_shift,
                g_eeh=g_eeh, g_ehh=g_ehh)


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


def recombination_exprs(options: BipolarOptions, *, pair: bool = False) -> Dict[str, str]:
    """{node model name: expression} of the extra recombination terms (Auger, radiative). pair=True
    gives each process's PAIR rate G + R (n p - n_ie^2 -> n p + n_ie^2: the detailed-balance split the
    noise GR source 2 (G + R) uses) -- one builder for the rate and its pair rate."""
    bgn = options.bgn is not None
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
                   fd_enhancement: bool, traps=()) -> Dict[str, np.ndarray]:
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
                              traps=traps)


def install_option_fields(device: str, region: str, options: BipolarOptions, *,
                          fd_enhancement: bool) -> None:
    """Stage 1 of the options (setup_bipolar_region calls it after the currents, BEFORE the
    recombination models, only when options.active): V_t at T_K; under BGN the n_ie field and the
    band-edge quasi-fields re-defining vdiff_g / vdiff_gp (a TRANSPORT change, not an additive term);
    under incomplete ionization the reference fields, IonizedNetDoping and the Poisson composite
    rebuilt on it (physics_bipolar_dd.rebuild_poisson_charge, which keeps every QTerm_*)."""
    from dynameta.carriers.physics_bipolar_dd import rebuild_poisson_charge
    o = options
    if o.T_K is not None:
        ds.set_parameter(device=device, region=region, name="V_t", value=o.V_t())
    f = _region_fields(device, region, o, fd_enhancement=fd_enhancement)

    # --- BGN: n_ie field, band-edge quasi-fields in both SG currents ---
    if o.bgn is not None:
        if fd_enhancement and float(np.max(np.maximum(f["fd_n"], f["fd_p"]))) > _BGN_FD_WARN_VT:
            warnings.warn(
                "install_bipolar_options: region {!r} combines an APPARENT bandgap-narrowing model "
                "({}) with the Fermi-Dirac g-factor on a degenerate doping (FD built-in correction up "
                "to {:.3g} V_t). Apparent BGN was extracted with Boltzmann statistics, so part of the "
                "degeneracy is counted twice; use fd_enhancement=False with apparent BGN, or accept "
                "the overlap knowingly.".format(region, o.bgn.kind,
                                                float(np.max(np.maximum(f["fd_n"], f["fd_p"])))),
                AdvisoryWarning, stacklevel=4)
        msg = bgn_basis_advisory(region, o.bgn, f["N_D"], f["N_A"])
        if msg is not None:                      # the fit is used outside its validity range
            warnings.warn(msg, UnreliableResultWarning, stacklevel=4)
        _set_field(device, region, "n_ie", f["n_ie"])
        _set_field(device, region, "BgnDc", f["D_c_V"])
        _set_field(device, region, "BgnDv", f["D_v_V"])
        for nm in ("BgnDc", "BgnDv"):
            ds.edge_from_node_model(device=device, region=region, node_model=nm)
        _edge_with_derivs(device, region, "vdiff_g",
                          "(vdiff + (BgnDc@n0 - BgnDc@n1)/V_t) / g_enh", ("Potential", "Electrons"))
        _edge_with_derivs(device, region, "vdiff_gp",
                          "(vdiff - (BgnDv@n0 - BgnDv@n1)/V_t) / g_enh_p", ("Potential", "Holes"))

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
                         fd_enhancement: bool, traps=()) -> None:
    """Stage 2 (setup_bipolar_region, after USRH and the trap levels): Auger / radiative / external
    generation as ADDITIVE terms (physics_bipolar_dd.add_recombination_term, each with its noise pair
    rate; the diagnostics UAuger / URad keep their names), and -- when the equilibrium itself changes
    (BGN, incomplete ionization, or CHARGED trap levels `traps`) -- the charge-neutral fields the
    ohmic contacts and the seed pin (NeutralElectrons/Holes, EquilibriumPotential, SeedPsiShift).
    options may be None (charged traps alone)."""
    from dynameta.carriers.physics_bipolar_dd import add_recombination_term
    o = options
    if o is not None:
        if o.auger is not None:
            ds.set_parameter(device=device, region=region, name="AugerCn", value=float(o.auger.C_n_m6s))
            ds.set_parameter(device=device, region=region, name="AugerCp", value=float(o.auger.C_p_m6s))
        if o.radiative is not None:
            ds.set_parameter(device=device, region=region, name="B_rad", value=float(o.radiative.B_m3s))
        f = None
        if o.auger is not None and o.auger.coulomb_enhanced:
            f = _region_fields(device, region, o, fd_enhancement=fd_enhancement, traps=traps)
            _set_field(device, region, "AugerGeeh", f["g_eeh"])
            _set_field(device, region, "AugerGehh", f["g_ehh"])
        pairs = recombination_exprs(o, pair=True)
        for nm, ex in recombination_exprs(o).items():
            _node_with_derivs(device, region, nm, ex, ("Electrons", "Holes"))     # diagnostic name
            add_recombination_term(device, region, _TERM_NAMES[nm], nm, pair=pairs[nm],
                                   derivs={"Electrons": nm + ":Electrons", "Holes": nm + ":Holes"})
        if o.generation_m3s is not None:
            ds.set_parameter(device=device, region=region, name="G_ext", value=float(o.generation_m3s))
            # Poisson generation (photogeneration shot noise) unless the noise call says otherwise
            add_recombination_term(device, region, GEXT_TERM, "-G_ext", pair="G_ext")
    if (o is not None and o.alters_equilibrium) or traps:
        f = _region_fields(device, region, o, fd_enhancement=fd_enhancement, traps=traps)
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
           "BipolarOptions", "band_dos_at_T",
           "AUGER_SI_DZIEWIOR_SCHMID_1977", "AUGER_SI_ALTERMATT_1997", "RADIATIVE_SI_TRUPKE_2003",
           "BGN_SI_SLOTBOOM_1977", "BGN_SI_KLAASSEN_1992", "BGN_SI_DEL_ALAMO_1985",
           "SI_DONOR_IONIZATION_EV", "SI_ACCEPTOR_IONIZATION_EV", "IONIZATION_SI_P_B",
           "equilibrium_fields", "srh_expr", "recombination_exprs", "ionized_net_doping_expr",
           "install_option_fields", "install_option_terms", "options_contact_models", "GEXT_TERM",
           "bgn_basis_advisory", "mott_advisory"]
