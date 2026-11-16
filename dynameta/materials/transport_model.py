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

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Optional

import numpy as np

from dynameta.warning_types import AdvisoryWarning

if TYPE_CHECKING:                                    # pragma: no cover - typing only
    from dynameta.carriers.bipolar_options import BipolarOptions


# v0.16: the density probe of TransportModel.dos_mass_at_bg and its relative-spread threshold
_DOS_PROBE_FACTORS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
_DOS_MASS_RTOL = 1e-2

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


@dataclass(frozen=True)
class KaneBand:
    """A Kane (nonparabolic) conduction band E (1 + alpha E) = hbar^2 k^2 / (2 m0) -- v0.16.

    ONE object describes the band; everything that needs it derives from it:
      * the equilibrium electrostatics (TransportModel(band=KaneBand(...)) in the 2-D and 3-D builders, or
        physics_equilibrium.setup_semiconductor_region(band=...)): n = N_c(m0) G(eta, alpha kT) with G the
        finite-temperature Kane density closure of carriers.kane_band (it reduces to the parabolic F_1/2
        closure at alpha = 0), and Phi_c0 calibrated on it (density_m3 / eta_of_density below);
      * the screening length of the default mesh grading (screening_mesh.screening_length_m);
      * the density-of-states mass reported to every consumer (dos_mass_kg; TransportModel sets its
        dos_mass_kg_of_n_m3 to it);
      * the optical (Drude) mass: optical_mass() is materials.KaneOpticalMass(m0, alpha), the Fermi-surface
        mass m0 (1 + 2 alpha E_F) of this band.

    m0_kg: band-edge (k = 0) effective mass [kg], finite and > 0. alpha_per_eV: nonparabolicity [1/eV],
    finite and >= 0 (0 = parabolic: every closure, model string and value is then the parabolic one with
    DOS mass m0). Temperatures default to T_REF, the temperature the equilibrium regions run at."""
    m0_kg: float
    alpha_per_eV: float

    def __post_init__(self) -> None:
        m, a = float(self.m0_kg), float(self.alpha_per_eV)
        if not (np.isfinite(m) and m > 0.0):
            raise ValueError("KaneBand.m0_kg must be finite and > 0 (kg), got {!r}".format(self.m0_kg))
        if not (np.isfinite(a) and a >= 0.0):
            raise ValueError("KaneBand.alpha_per_eV must be finite and >= 0 (1/eV), got {!r}".format(
                self.alpha_per_eV))

    def b(self, T_K: Optional[float] = None) -> float:
        """alpha kT (dimensionless) at T_K (default T_REF)."""
        from dynameta.carriers import kane_band as _KB
        return _KB.kane_b(self.alpha_per_eV, _T(T_K))

    def effective_dos_m3(self, T_K: Optional[float] = None) -> float:
        """N_c of the band-edge mass [m^-3] at T_K (default T_REF)."""
        from dynameta.carriers.physics_equilibrium import effective_dos_m3
        return float(effective_dos_m3(float(self.m0_kg), _T(T_K)))

    def density_m3(self, eta, T_K: Optional[float] = None):
        """n(eta) [m^-3] of the closure the electrostatics solves: N_c(m0) kane_band.kane_closure(eta, b)."""
        from dynameta.carriers import kane_band as _KB
        out = self.effective_dos_m3(T_K) * np.asarray(_KB.kane_closure(eta, self.b(T_K)))
        return float(out) if np.ndim(out) == 0 else out

    def eta_of_density(self, n_m3: float, T_K: Optional[float] = None) -> float:
        """The reduced Fermi level (E_F - E_c)/kT at density n_m3 (inverse of density_m3; rtol 1e-10). The
        bracket's upper end is max(80, gamma_F/kT + 20), gamma_F = hbar^2 (3 pi^2 n)^(2/3)/(2 m0) >= E_F (the
        T = 0 Kane E_F solves E_F (1 + alpha E_F) = gamma_F, and temperature lowers it), so a cold band works."""
        from dynameta.carriers import kane_band as _KB
        from dynameta.constants import HBAR, KB
        n = float(n_m3)
        gamma_F = HBAR ** 2 * (3.0 * np.pi ** 2 * n) ** (2.0 / 3.0) / (2.0 * float(self.m0_kg))
        top = max(80.0, gamma_F / (KB * _T(T_K)) + 20.0)
        return _KB.invert_kane_closure(n / self.effective_dos_m3(T_K), self.b(T_K), eta_max=top)

    def dos_mass_kg(self, n_m3, T_K: Optional[float] = None):
        """The density-of-states mass [kg] of this band at density n_m3: the mass m_d for which the
        parabolic closure at the same Fermi level holds the same density, N_c(m_d) F_1/2(eta) = n, i.e.
        m_d = m0 (G(eta, b)/F_1/2(eta))^(2/3) with eta = eta_of_density(n). T -> 0: m0 (1 + alpha E_F) (the
        counting mass n = (2 m_d E_F/hbar^2)^(3/2)/(3 pi^2)); alpha = 0: m0 exactly. Scalar or array in n."""
        from dynameta.carriers import kane_band as _KB
        from dynameta.carriers.physics_equilibrium import F12_aymerich_humet
        if float(self.alpha_per_eV) == 0.0:
            out = np.full(np.shape(n_m3), float(self.m0_kg))
            return float(out) if np.ndim(out) == 0 else out
        b = self.b(T_K)

        def one(n):
            eta = self.eta_of_density(n, T_K)
            return float(self.m0_kg) * (_KB.kane_closure(eta, b) / F12_aymerich_humet(eta)) ** (2.0 / 3.0)
        out = np.vectorize(one, otypes=[np.float64])(np.asarray(n_m3, dtype=np.float64))
        return float(out) if np.ndim(out) == 0 else out

    def optical_mass(self):
        """The Drude (Fermi-surface) mass of this band as a callable of n: materials.KaneOpticalMass(m0, alpha)
        (KaneOpticalMass.from_band), for DrudeOptical(m_opt_kg=band.optical_mass())."""
        from dynameta.materials.scattering import KaneOpticalMass
        return KaneOpticalMass.from_band(self)


def _T(T_K: Optional[float]) -> float:
    from dynameta.constants import T_REF
    t = T_REF if T_K is None else float(T_K)
    if not (np.isfinite(t) and t > 0.0):
        raise ValueError("temperature must be finite and > 0 (K), got {!r}".format(T_K))
    return t


def _is_band_dos(fn) -> bool:
    """True for a bound KaneBand.dos_mass_kg (the DOS-mass callable TransportModel derives from a band)."""
    return isinstance(getattr(fn, "__self__", None), KaneBand) and getattr(fn, "__func__", None) is KaneBand.dos_mass_kg


@dataclass
class TransportModel:
    """DC carrier-transport parameters for one semiconductor material.

    Args:
      n_bg_m3              : background (donor) carrier density [m^-3]
      eps_static           : static relative permittivity for DEVSIM Poisson
      dos_mass_kg_of_n_m3  : DENSITY-OF-STATES effective mass [kg], callable of
                              n; used for the conduction-band Nc / Fermi level. The parabolic
                              electrostatics evaluates it once, at n_bg (dos_mass_at_bg warns when it
                              varies). Omit it with band=KaneBand(...): it is then band.dos_mass_kg.
      band                 : v0.16 "parabolic" (default, v0.15.1 bit for bit) or
                              KaneBand(m0_kg, alpha_per_eV): the equilibrium electrostatics solves the
                              finite-temperature Kane density closure self-consistently (carriers.kane_band);
                              the screening length, Phi_c0, the reported DOS mass and band.optical_mass()
                              (for DrudeOptical) all derive from the one band. physics='equilibrium' only
                              (drift-diffusion / bipolar raise NotImplementedError).
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
      carrier_statistics   : (v0.15, decision 28; v0.15.0 default change, user decision 2026-11-07)
                              None (default) = "fermi_dirac" = FD transport (the consistent quasi-field)
                              AND FD detailed balance (exact zero-bias balance; REQUIRED by an FD-paired
                              BGN set such as the house Si bundle materials.semiconductor_presets.
                              silicon_transport); "legacy" = the v0.14 default, BYTE-IDENTICAL (FD
                              g-factor transport with the edge-averaged g, Boltzmann-product detailed
                              balance: a zero-bias current next to degenerate layers; NOT recommended,
                              reproducing v0.14 numbers only); "boltzmann" =
                              no FD g-factor (an APPARENT, Boltzmann-extracted BGN set on a degenerate
                              layer -- removes the double count). Read by the 2-D LayeredDevsimBuilder
                              and Stacked3DSpec.from_design (carriers.bipolar_options module doc, item 2).
                              drift_diffusion: None / "fermi_dirac" = the consistent FD quasi-field
                              transport, "legacy" = the v0.14 edge-averaged g
                              (physics_drift_diffusion.setup_semiconductor_region_dd(statistics=));
                              "boltzmann" is refused.
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
    dos_mass_kg_of_n_m3:         Optional[MassFn] = None     # required unless band is a KaneBand (Args)
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
    # v0.16: the conduction band of the electrostatics, "parabolic" (default; N_c from
    # dos_mass_kg_of_n_m3 at n_bg, bit-identical to v0.15.1) or a KaneBand (see Args).
    band:                        object = "parabolic"

    def __post_init__(self) -> None:
        if self.physics not in ("equilibrium", "drift_diffusion", "bipolar_dd"):
            raise ValueError("physics must be 'equilibrium', 'drift_diffusion' or 'bipolar_dd'")
        if isinstance(self.band, KaneBand):
            if self.physics != "equilibrium":
                raise NotImplementedError(
                    "TransportModel(band=KaneBand(...)) with physics={!r}: the Kane band is implemented for the "
                    "EQUILIBRIUM closure only. The drift-diffusion and bipolar transport (Scharfetter-Gummel "
                    "flux with the Fermi-Dirac enhancement, contact and heterointerface conditions, the "
                    "generalized Einstein relation) are written for a parabolic band; a Kane density there "
                    "would be inconsistent with its own current. Use physics='equilibrium', or a parabolic "
                    "band (dos_mass_kg_of_n_m3) for transport.".format(self.physics))
            if self.dos_mass_kg_of_n_m3 is not None and not _is_band_dos(self.dos_mass_kg_of_n_m3):
                raise ValueError("TransportModel: band=KaneBand(...) defines the density-of-states mass; omit "
                                 "dos_mass_kg_of_n_m3 (it becomes band.dos_mass_kg, the band's DOS mass)")
            self.dos_mass_kg_of_n_m3 = self.band.dos_mass_kg
        elif isinstance(self.band, str) and self.band == "parabolic":
            if self.dos_mass_kg_of_n_m3 is None:
                raise ValueError("TransportModel: dos_mass_kg_of_n_m3 is required for band='parabolic' (or give "
                                 "band=KaneBand(m0_kg, alpha_per_eV), which defines it)")
        else:
            raise TypeError("TransportModel.band must be 'parabolic' or a KaneBand, got {!r}".format(self.band))
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
            if self.carrier_statistics not in ("boltzmann", "fermi_dirac", "legacy"):
                raise ValueError("carrier_statistics must be None, 'fermi_dirac', 'boltzmann' or 'legacy', got {!r}"
                                 .format(self.carrier_statistics))
            ok = (self.physics == "bipolar_dd"
                  or (self.physics == "drift_diffusion" and self.carrier_statistics != "boltzmann"))
            if not ok:
                raise ValueError("carrier_statistics={!r} applies only to physics='bipolar_dd' (and "
                                 "'fermi_dirac' / 'legacy' to 'drift_diffusion': the unipolar model has no "
                                 "Boltzmann mode) -- got physics={!r}; it would be silently ignored".format(
                                     self.carrier_statistics, self.physics))
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

    def dos_mass_at_bg(self, where: str) -> float:
        """The DOS effective mass [kg] the carrier physics uses: dos_mass_kg_of_n_m3 evaluated at n_bg_m3,
        exactly float(dos_mass_kg_of_n_m3(n_bg_m3)) (the value every builder took before v0.16).

        v0.16 guard: the electrostatics has ONE N_c per region (a parabolic band with a constant DOS
        mass), so a density-dependent callable is reduced to its value at n_bg. The callable is probed at
        n = n_bg x {0.1, 0.2, 0.5, 1, 2, 5, 10} (one scalar call each, so scalar-only callables work); if it
        varies by more than 1 % of the n_bg value over that range, ONE AdvisoryWarning per TransportModel
        instance says so (`where` names the caller in it). A probe that raises or returns a non-finite value
        counts as a variation (the warning names it). Constants never warn.

        With band=KaneBand(...) (v0.16) the electrostatics solves that band's closure (density-dependent DOS
        by construction, consistent with the optical KaneOpticalMass of the same band), so nothing is
        reduced and nothing warns; the value returned is the band's DOS mass at n_bg (KaneBand.dos_mass_kg).
        The builders pass the band itself, not this number, to the closure."""
        if isinstance(self.band, KaneBand):
            return float(self.band.dos_mass_kg(self.n_bg_m3))
        m_bg = float(self.dos_mass_kg_of_n_m3(self.n_bg_m3))
        if not getattr(self, "_dos_mass_guard_done", False):
            object.__setattr__(self, "_dos_mass_guard_done", True)
            bad, vals = None, []
            for f in _DOS_PROBE_FACTORS:
                try:
                    v = float(self.dos_mass_kg_of_n_m3(f * self.n_bg_m3))
                except Exception as exc:                     # noqa: BLE001 - reported, not swallowed
                    bad = "raised {}: {}".format(type(exc).__name__, exc)
                    break
                if not np.isfinite(v):
                    bad = "returned {!r} at n = {:g} n_bg".format(v, f)
                    break
                vals.append(v)
            spread = (max(vals) - min(vals)) / abs(m_bg) if (bad is None and m_bg != 0.0) else float("inf")
            if bad is not None or spread > _DOS_MASS_RTOL:
                detail = bad if bad is not None else "varies by {:.3g} % of its n_bg value".format(100.0 * spread)
                warnings.warn(
                    "{}: TransportModel.dos_mass_kg_of_n_m3 is density-dependent over [0.1, 10] n_bg ({}), but "
                    "the electrostatics uses ONE DOS mass per region (a parabolic band with N_c from the mass "
                    "at n_bg = {:.4g} m^-3: {:.6g} kg). The charge distribution therefore does not follow the "
                    "density dependence (a nonparabolic band enters the optics only).".format(
                        where, detail, float(self.n_bg_m3), m_bg), AdvisoryWarning, stacklevel=2)
        return m_bg
