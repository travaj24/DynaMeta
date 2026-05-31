"""
Material registry: optical eps(lambda) for FEM + Drude-active
semiconductor parameters for DC drift-diffusion + Drude.

A Material is the single source of truth for everything stage 1/2/3
need to know about a given solid in the device. Materials are
referenced by Layer.material in a Design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Material spec
# ---------------------------------------------------------------------------

# A function that maps wavelength [m] to complex eps. Allows callers to
# pass a constant scalar, a numpy interpolant, or any custom dispersion.
EpsAtLambdaFn = Callable[[float], complex]


@dataclass
class DrudeSpec:
    """Drude-active semiconductor parameters (e.g. ITO).

    The default formulation is the density-dependent model used by the
    Park 2021 metasurface project:

      eps(n, lambda) = eps_inf - omega_p^2 / (omega^2 + i*omega*gamma)
      omega_p^2(n)   = n * e^2 / (eps0 * m_eff(n))
      m_eff(n)       = Kane non-parabolic correction
      gamma(n)       = e / (m_eff(n) * mu(n))     where mu(n) is
                       Caughey-Thomas mobility

    Pass callables for `m_eff_kg_of_n_m3` and `gamma_rad_s_of_n_m3` to
    enable density-dependent values; pass scalars (cast to a 0-arg
    callable) for constant values.

    DOS mass vs optical mass: `m_eff_kg_of_n_m3` is the DENSITY-OF-STATES
    effective mass, used by Stage 1 for the conduction-band Nc (and thus
    Phi_c0 / Fermi level). In non-parabolic ITO the OPTICAL (conductivity)
    mass that enters the Drude omega_p^2 is DIFFERENT (lighter). Park 2021
    Fig S2 fits the optical mass ~0.225 m_e while the DOS mass is ~0.35 m_e.
    Set `m_eff_opt_kg_of_n_m3` to supply the optical mass to Stage 2 Drude;
    if left None, Stage 2 falls back to `m_eff_kg_of_n_m3` (legacy behaviour,
    which conflates the two and red-shifts the ENZ).
    """
    eps_inf:              float
    m_eff_kg_of_n_m3:     Callable[[np.ndarray], np.ndarray]
    gamma_rad_s_of_n_m3:  Callable[[np.ndarray], np.ndarray]
    # DC drift-diffusion parameters (consumed by Stage 1 only)
    n_bg_m3:              float        # background carrier density
    eps_static:           float        # static permittivity (relative)
    band_gap_eV:          float = 3.6  # ITO bandgap ~ 3.6 eV (default)
    chi_eV:               float = 4.5  # electron affinity (default ITO)
    # Optical (conductivity) effective mass for Stage 2 Drude omega_p^2.
    # None -> Stage 2 uses m_eff_kg_of_n_m3 (the DOS mass) as before.
    m_eff_opt_kg_of_n_m3: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def optical_mass_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """The effective-mass callable Stage 2 Drude should use: the
        dedicated optical mass if provided, else the DOS mass."""
        return (self.m_eff_opt_kg_of_n_m3 if self.m_eff_opt_kg_of_n_m3 is not None
                  else self.m_eff_kg_of_n_m3)


@dataclass
class TrapSpec:
    """Optional deep-acceptor trap states in a semiconductor (e.g. for
    explaining sub-bandgap losses in oxide semiconductors)."""
    enabled:             bool = False
    N_trap_acceptor_m3:  float = 1.0e24
    E_trap_depth_eV:     float = 0.30
    g_trap:              float = 1.0


@dataclass
class Material:
    """A solid material: optical dispersion + (optionally) Drude/DC
    semiconductor parameters.

    Args:
      name              : unique identifier referenced by Layer.material
      eps_at_lambda     : callable f(lambda_m) -> complex eps
                            (for FEM); if None, falls back to drude.eps_inf
                            (only for semis)
      drude             : non-None marks this material as a Drude-active
                            semiconductor; controlled by Stage 1+2
      traps             : optional trap states for Stage 1
      is_metal          : True for metals (no Drude treatment as semi;
                            optical eps from eps_at_lambda)
      pretty_name       : display name for plots / logs
    """
    name:                 str
    eps_at_lambda:        Optional[EpsAtLambdaFn] = None
    drude:                Optional[DrudeSpec]     = None
    traps:                Optional[TrapSpec]      = None
    is_metal:             bool                     = False
    pretty_name:          str                      = ""

    def __post_init__(self) -> None:
        if not self.pretty_name:
            self.pretty_name = self.name
        if self.eps_at_lambda is None and self.drude is None and not self.is_metal:
            raise ValueError(
                "Material '{}' has neither eps_at_lambda nor drude nor is_metal "
                "set -- can't be used in any stage.".format(self.name))

    @property
    def is_semiconductor(self) -> bool:
        return self.drude is not None

    def optical_eps(self, lambda_m: float) -> complex:
        """Get the wavelength-dependent optical eps for this material.

        For semiconductors this is a per-bias quantity computed in Stage 2.
        For metals/dielectrics it's just eps_at_lambda(lambda_m).
        """
        if self.eps_at_lambda is None:
            if self.drude is not None:
                # Background eps (n = n_bg) -- Stage 3 overrides this with
                # bias-dependent spatial eps for semiconductors.
                return complex(self.drude.eps_inf)
            raise RuntimeError(
                "Material '{}' has no optical eps lookup.".format(self.name))
        return complex(self.eps_at_lambda(lambda_m))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MaterialRegistry:
    """A registry that maps material names to Material instances.

    Designs reference materials by name; the registry is the single
    source of truth for what those names mean. A Design carries its
    own registry so different designs can use different material defs
    without colliding.
    """
    def __init__(self) -> None:
        self._materials: Dict[str, Material] = {}

    def add(self, material: Material) -> "MaterialRegistry":
        if material.name in self._materials:
            raise ValueError("Material '{}' already in registry".format(material.name))
        self._materials[material.name] = material
        return self

    def get(self, name: str) -> Material:
        if name not in self._materials:
            raise KeyError("Material '{}' not in registry. Known: {}".format(
                name, sorted(self._materials.keys())))
        return self._materials[name]

    def __contains__(self, name: str) -> bool:
        return name in self._materials

    def names(self) -> list[str]:
        return sorted(self._materials.keys())

    def __len__(self) -> int:
        return len(self._materials)
