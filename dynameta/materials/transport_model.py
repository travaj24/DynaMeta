"""
TransportModel: the DC / carrier-transport parameters consumed by Stage 1
(DEVSIM). Separate from OpticalModel (Stage 2/3) on purpose -- the static
(Poisson) permittivity and the DOS effective mass live here; the optical
eps_inf and optical mass live in the OpticalModel.

`physics` selects the Stage-1 formulation:
  "equilibrium"      -> single-variable nonlinear Poisson, Fermi-Dirac
                        electron density as a derived node_model (the proven
                        formulation; no currents). DEFAULT.
  "drift_diffusion"  -> full DD (electrons+holes solution variables, SG
                        currents, continuity, recombination, current BCs).
                        Phase 4; the equilibrium solve seeds it.

DD-only fields (mobility, recombination, traps) are optional and only read
when physics == "drift_diffusion".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Optional

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover - typing only
    from dynameta.carriers.bipolar_options import BipolarOptions


MassFn = Callable[[np.ndarray], np.ndarray]
MobilityFn = Callable[[np.ndarray], np.ndarray]
CarrierPhysics = Literal["equilibrium", "drift_diffusion", "bipolar_dd"]


_TRAPSPEC_REFUSAL = (
    "materials.TrapSpec(enabled=True) is not implemented: no solver path reads TransportModel.traps, so "
    "the trap would be silently ignored (it always was; docs/audit/2026-11-03-devsim-traps.md section 7). "
    "Use the DEVSIM trap physics instead (bipolar_dd only): bulk levels -> "
    "TransportModel(srh_traps=(physics_traps.TrapLevel(...), ...)) and srh_trap_energy_eV= (the E_t of the "
    "tau_srh_s channel), or Stacked3DSpec(srh_traps=..., srh_trap_energy_eV=...); interface traps -> "
    "physics_traps.InterfaceTraps via LayeredDevsimBuilder(interface_traps={...}), "
    "Stacked3DSpec(interface_traps=...) or Stacked3DSpec.from_design(..., interface_traps=...).")


@dataclass
class TrapSpec:
    """LEGACY placeholder for deep-level traps -- NOT IMPLEMENTED (user decision 2026-11-04).

    No solver path has ever read `TransportModel.traps`, so `enabled=True` used to be silently ignored.
    Setting `enabled` to a true value now raises NotImplementedError naming what does work
    (physics_traps.TrapLevel via TransportModel.srh_traps / srh_trap_energy_eV, physics_traps.
    InterfaceTraps via the builders' interface_traps=). The check is in __setattr__, so it covers the
    constructor AND a later `spec.enabled = True` -- the one place that cannot be bypassed short of
    object.__setattr__; a check in TransportModel.__post_init__ or in a builder would miss mutation
    after construction, or a consumer that forgets to check. enabled=False (the default) is unchanged
    and stays inert. Mapping it onto TrapLevel was not done: "depth" has no stated reference (E_c?) and
    E_i is not midgap when N_c != N_v (traps doc, section 7, item 1)."""
    enabled:             bool = False
    N_trap_acceptor_m3:  float = 1.0e24
    E_trap_depth_eV:     float = 0.30
    g_trap:              float = 1.0

    def __setattr__(self, name, value):
        if name == "enabled" and bool(value):
            raise NotImplementedError(_TRAPSPEC_REFUSAL)
        object.__setattr__(self, name, value)


@dataclass
class TransportModel:
    """DC carrier-transport parameters for one semiconductor material.

    Args:
      n_bg_m3              : background (donor) carrier density [m^-3]
      eps_static           : static relative permittivity for DEVSIM Poisson
      dos_mass_kg_of_n_m3  : DENSITY-OF-STATES effective mass [kg], callable of
                              n; used for the conduction-band Nc / Fermi level
      band_gap_eV, chi_eV  : band gap and electron affinity [eV], default None (v0.14, user
                              decision 2026-11-04; they were 3.6 / 4.5). LOAD-BEARING wherever two
                              semiconductor regions of DIFFERENT materials share a device (v0.13.0 3-D
                              equilibrium alignment, v0.14.0 DEVSIM heterointerfaces: Anderson rule
                              Delta E_c = chi_0 - chi_1): the builders REFUSE such a device while a
                              chi_eV is None, and a bipolar heterojunction refuses a None band_gap_eV
                              (it also checks the value against E_g = kT ln(N_c N_v / n_i^2)). Not read
                              for a single semiconductor or a same-material stack (zero offset by
                              identity), so those need neither. No other consumer reads them
                              (inventory 2026-11-04: docs/audit/2026-11-04-v0.14-user-decisions.md).
      carrier_statistics   : bipolar_dd only (v0.15, decision 28): None (default) = the legacy
                              statistics, BYTE-IDENTICAL (FD g-factor transport, Boltzmann-product
                              detailed balance); "fermi_dirac" = FD transport AND FD detailed balance
                              (degenerate layers; REQUIRED by an FD-paired BGN set such as the house Si
                              bundle materials.semiconductor_presets.silicon_transport); "boltzmann" =
                              no FD g-factor (an APPARENT, Boltzmann-extracted BGN set on a degenerate
                              layer -- removes the double count). Read by the 2-D LayeredDevsimBuilder
                              and Stacked3DSpec.from_design (carriers.bipolar_options module doc, item 2).
                              drift_diffusion (v0.15 FD-mode integration): "fermi_dirac" selects the
                              consistent FD quasi-field transport (physics_drift_diffusion.
                              setup_semiconductor_region_dd(statistics=)); "boltzmann" is refused.
      chi_reference_T_K    : the temperature [K] at which chi_eV (and band_gap_eV) were QUOTED. None =
                              unstated, read as T_REF (300 K). No chi(T) model exists in v0.14, so a
                              heterostack run at another temperature uses chi_eV unchanged; the 2-D
                              builder therefore REFUSES a heterostack whose run T differs from this
                              (LayeredDevsimBuilder(on_chi_temperature="refuse"), the default; "warn" /
                              "off" proceed). Set it equal to the device T (BipolarOptions.T_K) when
                              chi_eV is your value AT that T; that is the only thing it asserts. With a
                              band_temperature model it is the model's anchor T_ref instead (0.0 allowed).
      band_temperature     : v0.15 opt-in materials.BandTemperature (default None = off): band_gap_eV, chi_eV
                              (and, with derive_n_i, n_i_m3) are then values at chi_reference_T_K, evaluated
                              at the device T by materials.band_data_at_T (Varshni gap change anchored at
                              T_ref; chi moves by the REQUIRED affinity_fraction of it). A material carrying
                              one satisfies the on_chi_temperature check. See materials/band_temperature.py
                              for which anchor / affinity_fraction / derive_n_i to use when.
      physics              : "equilibrium" (default) | "drift_diffusion"
      mobility_m2Vs_of_n_m3      : electron mobility (DD only)
      hole_mobility_m2Vs_of_n_m3 : hole mobility (DD only)
      tau_srh_s            : SRH lifetime [s] (DD only); None -> no SRH
      traps                : LEGACY TrapSpec, read by no solver; enabled=True raises (TrapSpec doc)
      hooge_alpha_n, hooge_alpha_p : v0.15 (plan 4.2.4, user decision 33) the material's Hooge 1/f
                              parameter alpha_H for electrons / holes. None (default) = no Hooge source
                              for that carrier -- there are NO per-material defaults. Read ONLY by
                              LayeredDevsimBuilder.noise_spectrum(hooge="material"), which maps each
                              carrier region to its material's pair (noise_analysis.noise_spectrum's
                              hooge_alpha mapping); nothing else (DC, ssac, the other noise calls) reads
                              them. Finite and > 0 when set; electrons need a carrier physics
                              ('drift_diffusion' or 'bipolar_dd'), holes need 'bipolar_dd'. Values: Hooge
                              1994 (IEEE TED 41, 1926) puts alpha_H ~ 1e-4 for a perfect lattice and
                              1e-6 .. 1e-3 measured; the historical 2e-3 is a "poor samples" average,
                              not a material value (noise_analysis module doc, item 2, for the ranges).
    """
    n_bg_m3:                     float
    eps_static:                  float
    dos_mass_kg_of_n_m3:         MassFn
    band_gap_eV:                 Optional[float] = None
    chi_eV:                      Optional[float] = None
    physics:                     CarrierPhysics = "equilibrium"
    # Drift-diffusion only (Phase 4):
    mobility_m2Vs_of_n_m3:       Optional[MobilityFn] = None
    hole_mobility_m2Vs_of_n_m3:  Optional[MobilityFn] = None
    tau_srh_s:                   Optional[float] = None
    traps:                       Optional[TrapSpec] = None
    # Field-dependent mobility (R1): Caughey-Thomas velocity saturation on the DD current,
    # mu(E)=mu_low/(1+(mu_low|E|/v_sat)^ct_beta)^(1/ct_beta) with mu_low = the mobility callable above.
    # Off by default -> byte-identical constant-mobility solve; v_sat_ms required when on.
    field_dependent_mobility:    bool = False
    v_sat_ms:                    Optional[float] = None
    ct_beta:                     float = 2.0
    # Bipolar drift-diffusion only ("bipolar_dd"): full 3-variable (Potential, Electrons, Holes)
    # solve with SRH recombination -- for a real p-n device (the electron-only DD assumes a
    # degenerate unipolar semiconductor like ITO). n_i_m3 is the intrinsic carrier density;
    # tau_srh_s is the (shared) SRH lifetime; dos_mass_p_kg is the VALENCE-band DOS mass (-> N_v,
    # defaults to the conduction N_c). Doping sign: `acceptor` makes the region uniformly p-type
    # (NetDoping = -n_bg_m3); for an IN-REGION junction supply `net_doping_expr`, a DEVSIM
    # node-model expression (signed, m^-3) in the SI coordinates x/y, e.g.
    # "ifelse(x < 1.5e-7, -1e24, 1e24)" (the abrupt lateral p-n step).
    n_i_m3:                      Optional[float] = None
    acceptor:                    bool = False
    net_doping_expr:             Optional[str] = None
    dos_mass_p_kg:               Optional[float] = None
    # audit R26/R27 (v0.14.0): opt-in Auger / radiative recombination, bandgap narrowing, incomplete
    # ionization, T_K, uniform generation for the bipolar_dd path -- a
    # dynameta.carriers.bipolar_options.BipolarOptions. None (default) keeps the SRH-only physics
    # byte-identical. Read by the 2-D LayeredDevsimBuilder and the 3-D Stacked3DSpec.from_design.
    bipolar_options:             Optional["BipolarOptions"] = None
    # Band-to-band tunneling (D1d, bipolar_dd only): None (default) = OFF, byte-identical; else a
    # dynameta.carriers.tunneling.BTBTParams (e.g. SILICON_HURKX) -> the local Kane/Hurkx BTBT term
    # in both continuity equations, coupled in the Newton; or a TUPLE of them, one term per process,
    # summed (v0.15, e.g. tunneling.GERMANIUM_KAO). A set with scale_B_with_gap=True takes B at the
    # region T from band_temperature (tunneling.btbt_for_material). Typed loosely to keep this module
    # devsim-free (the params class lives with the DEVSIM physics).
    btbt:                        Optional[object] = None
    # Bipolar-only SRH trap physics (D1c, 2026-11-03; carriers.physics_traps). srh_trap_energy_eV is
    # E_t - E_i [eV] of the tau_srh_s channel (0.0 = the legacy midgap level, bit-identical);
    # srh_traps is a tuple of physics_traps.TrapLevel -- extra discrete levels (own E_t, lifetimes or
    # cross sections, density, acceptor/donor/neutral charge). Defaults leave the solve unchanged.
    srh_trap_energy_eV:          float = 0.0
    srh_traps:                   tuple = ()
    # Heterointerface thermionic emission (v0.14.0, carriers.physics_heterojunction): the effective
    # Richardson constants A* [A m^-2 K^-2] used when THIS material holds the barrier top of a
    # semiconductor/semiconductor interface. None -> 4 pi q m* k^2/h^3 from the DOS mass (electrons:
    # dos_mass_kg_of_n_m3(n_bg_m3); holes: dos_mass_p_kg, else the electron mass).
    richardson_n_Am2K2:          Optional[float] = None
    richardson_p_Am2K2:          Optional[float] = None
    # v0.14 row 3.2-10 (user decision 3, 2026-11-06): the temperature chi_eV was quoted at (see Args).
    # An assert-only tag; None = unstated (T_REF). Read by physics_heterojunction.check_affinity_temperature.
    chi_reference_T_K:           Optional[float] = None
    # v0.15 (decision 28): bipolar carrier statistics, see Args. None keeps every v0.14 result.
    carrier_statistics:          Optional[str] = None
    # v0.15 (plan 4.3.3): opt-in band-temperature model -- a materials.band_temperature.BandTemperature
    # (Varshni E_g(T) anchored at chi_reference_T_K, chi(T) by a required fraction of the gap change, optional
    # derived n_i(T)). None (default) = band_gap_eV / chi_eV / n_i_m3 used as given at every T (bit-identical).
    band_temperature:            Optional[object] = None
    # v0.15 (plan 4.2.4, decision 33): per-carrier Hooge alpha_H, no defaults (see Args). Read only by the
    # 2-D builder's noise_spectrum(hooge="material"); None keeps every result bit-identical.
    hooge_alpha_n:               Optional[float] = None
    hooge_alpha_p:               Optional[float] = None

    def __post_init__(self) -> None:
        if self.physics not in ("equilibrium", "drift_diffusion", "bipolar_dd"):
            raise ValueError("physics must be 'equilibrium', 'drift_diffusion' or 'bipolar_dd'")
        if self.physics in ("drift_diffusion", "bipolar_dd") and self.mobility_m2Vs_of_n_m3 is None:
            raise ValueError(
                "{} physics requires mobility_m2Vs_of_n_m3".format(self.physics))
        if self.field_dependent_mobility and self.v_sat_ms is None:
            raise ValueError("field_dependent_mobility=True requires v_sat_ms (saturation velocity, m/s)")
        if self.chi_reference_T_K is not None and not (
                np.isfinite(float(self.chi_reference_T_K)) and (float(self.chi_reference_T_K) > 0.0 or (
                    self.band_temperature is not None and float(self.chi_reference_T_K) == 0.0))):
            raise ValueError("chi_reference_T_K must be finite and > 0 (K) -- 0 K only with a band_temperature "
                             "model (data quoted at 0 K) -- got {!r}".format(self.chi_reference_T_K))
        if self.band_temperature is not None:
            from dynameta.materials.band_temperature import check_band_temperature
            check_band_temperature(self)
        for nm in ("richardson_n_Am2K2", "richardson_p_Am2K2"):
            v = getattr(self, nm)
            if v is not None and not (np.isfinite(float(v)) and float(v) > 0.0):
                raise ValueError("{} must be finite and > 0 (A m^-2 K^-2), got {!r}".format(nm, v))
        if self.physics == "bipolar_dd":
            missing = [nm for nm, v in (("hole_mobility_m2Vs_of_n_m3", self.hole_mobility_m2Vs_of_n_m3),
                                        ("n_i_m3", self.n_i_m3), ("tau_srh_s", self.tau_srh_s))
                       if v is None]
            if missing:
                raise ValueError("bipolar_dd physics requires {} (holes + recombination)".format(
                    ", ".join(missing)))
        if self.bipolar_options is not None:
            from dynameta.carriers.bipolar_options import BipolarOptions
            if not isinstance(self.bipolar_options, BipolarOptions):
                raise TypeError("TransportModel.bipolar_options must be a carriers.bipolar_options."
                                "BipolarOptions, got {!r}".format(type(self.bipolar_options).__name__))
            if self.physics != "bipolar_dd":
                raise ValueError("TransportModel.bipolar_options applies only to physics='bipolar_dd' "
                                 "(got {!r}); it would be silently ignored".format(self.physics))
        if self.carrier_statistics is not None:
            if self.carrier_statistics not in ("boltzmann", "fermi_dirac"):
                raise ValueError("carrier_statistics must be None, 'boltzmann' or 'fermi_dirac', got {!r}"
                                 .format(self.carrier_statistics))
            ok = (self.physics == "bipolar_dd"
                  or (self.physics == "drift_diffusion" and self.carrier_statistics == "fermi_dirac"))
            if not ok:
                raise ValueError("carrier_statistics={!r} applies only to physics='bipolar_dd' (and "
                                 "'fermi_dirac' to 'drift_diffusion': the consistent FD transport; the "
                                 "unipolar model has no Boltzmann mode) -- got physics={!r}; it would be "
                                 "silently ignored".format(self.carrier_statistics, self.physics))
        for nm, need in (("hooge_alpha_n", ("drift_diffusion", "bipolar_dd")),
                         ("hooge_alpha_p", ("bipolar_dd",))):
            v = getattr(self, nm)
            if v is None:
                continue
            if isinstance(v, bool) or not (np.isfinite(float(v)) and float(v) > 0.0):
                raise ValueError("{} must be finite and > 0 (dimensionless Hooge alpha_H), got {!r}"
                                 .format(nm, v))
            if self.physics not in need:
                raise ValueError("{} needs physics in {} (the carrier it fluctuates must be solved); got "
                                 "physics={!r}, so it would be silently ignored".format(nm, need, self.physics))
        if self.btbt is not None and self.physics != "bipolar_dd":
            raise ValueError("btbt (band-to-band tunneling generation) needs physics='bipolar_dd' "
                             "(it feeds the electron AND hole continuity equations); got physics={!r}"
                             .format(self.physics))
        if (self.srh_traps or self.srh_trap_energy_eV) and self.physics != "bipolar_dd":
            # the trap physics lives in the bipolar (n AND p) SRH model only; anywhere else it would
            # be silently dropped by the builders
            raise ValueError("srh_traps / srh_trap_energy_eV need physics='bipolar_dd' (got {!r})"
                             .format(self.physics))
