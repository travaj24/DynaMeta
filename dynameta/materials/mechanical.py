"""Mechanical properties for thermo-mechanical reliability (driver D3). MechanicalProps started as a
reliability-LOCAL table in reliability/fatigue.py (REL6); promoting it onto the Material schema makes
{CTE, E, nu, sigma_crit} first-class material data the fatigue (REL6) and stress-migration (REL7)
post-processors can pull from the SAME registry the optical/transport models live in.
reliability.fatigue re-exports this class, so existing imports keep working. Material.mechanical
defaults to None -> byte-identical for every existing material. Pure data; SI units."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MechanicalProps:
    """Mechanical property set for one film/solid. Typical values: Cu E=110 GPa nu=0.34 CTE=16.5e-6;
    SiO2 E=70 GPa nu=0.17 CTE=0.5e-6 sigma_crit~0.5-1 GPa; ITO E=115 GPa nu=0.35 CTE~6e-6 (brittle,
    sigma_crit~0.3-1 GPa); Si substrate CTE=2.6e-6. sigma_crit_Pa = inf treats the film as
    ductile-only (no brittle-fracture branch)."""
    E_Pa: float
    nu: float
    cte_per_K: float
    sigma_crit_Pa: float = float("inf")

    def __post_init__(self):
        if not (self.E_Pa > 0.0):
            raise ValueError("MechanicalProps: E_Pa must be > 0")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError("MechanicalProps: Poisson ratio must be in (-1, 0.5)")
        if not (self.sigma_crit_Pa > 0.0):
            raise ValueError("MechanicalProps: sigma_crit_Pa must be > 0")
