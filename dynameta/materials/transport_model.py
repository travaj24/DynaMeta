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
from typing import Callable, Literal, Optional

import numpy as np


MassFn = Callable[[np.ndarray], np.ndarray]
MobilityFn = Callable[[np.ndarray], np.ndarray]
CarrierPhysics = Literal["equilibrium", "drift_diffusion", "bipolar_dd"]


@dataclass
class TrapSpec:
    """Optional deep-level trap states in a semiconductor (Stage 1)."""
    enabled:             bool = False
    N_trap_acceptor_m3:  float = 1.0e24
    E_trap_depth_eV:     float = 0.30
    g_trap:              float = 1.0


@dataclass
class TransportModel:
    """DC carrier-transport parameters for one semiconductor material.

    Args:
      n_bg_m3              : background (donor) carrier density [m^-3]
      eps_static           : static relative permittivity for DEVSIM Poisson
      dos_mass_kg_of_n_m3  : DENSITY-OF-STATES effective mass [kg], callable of
                              n; used for the conduction-band Nc / Fermi level
      band_gap_eV, chi_eV  : informational band parameters
      physics              : "equilibrium" (default) | "drift_diffusion"
      mobility_m2Vs_of_n_m3      : electron mobility (DD only)
      hole_mobility_m2Vs_of_n_m3 : hole mobility (DD only)
      tau_srh_s            : SRH lifetime [s] (DD only); None -> no SRH
      traps                : optional trap states
    """
    n_bg_m3:                     float
    eps_static:                  float
    dos_mass_kg_of_n_m3:         MassFn
    band_gap_eV:                 float = 3.6
    chi_eV:                      float = 4.5
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

    def __post_init__(self) -> None:
        if self.physics not in ("equilibrium", "drift_diffusion", "bipolar_dd"):
            raise ValueError("physics must be 'equilibrium', 'drift_diffusion' or 'bipolar_dd'")
        if self.physics in ("drift_diffusion", "bipolar_dd") and self.mobility_m2Vs_of_n_m3 is None:
            raise ValueError(
                "{} physics requires mobility_m2Vs_of_n_m3".format(self.physics))
        if self.field_dependent_mobility and self.v_sat_ms is None:
            raise ValueError("field_dependent_mobility=True requires v_sat_ms (saturation velocity, m/s)")
        if self.physics == "bipolar_dd":
            missing = [nm for nm, v in (("hole_mobility_m2Vs_of_n_m3", self.hole_mobility_m2Vs_of_n_m3),
                                        ("n_i_m3", self.n_i_m3), ("tau_srh_s", self.tau_srh_s))
                       if v is None]
            if missing:
                raise ValueError("bipolar_dd physics requires {} (holes + recombination)".format(
                    ", ".join(missing)))
