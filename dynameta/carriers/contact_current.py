"""DEVSIM contact-current extractor (driver D1) -- the terminal current the reliability axis
(electromigration reliability.em consumes I_A externally) and any I-V/power analysis need.

ds.get_contact_current integrates the carrier flux through one contact for ONE continuity
equation; the total terminal current is the electron + hole sum over whichever continuity
contact equations exist AT THAT CONTACT (probed via ds.get_contact_equation_list -- a Poisson-
only gate on an oxide has none and is skipped, an ohmic unipolar contact has the electron one,
a bipolar ohmic contact has both). An equilibrium (Poisson-only) device therefore yields {} --
the byte-identical off-switch: no key is added to CarrierField.extras.

UNITS: the 2D layered mesh is built in METRES with an implicit out-of-plane depth, so the raw
contact current is per-unit-depth [A/m]; pass depth_m = unit-cell period_y to scale to amperes
(the linear-in-depth assumption of a periodic cell). The 3D gmsh mesh is emitted scaled to
metres (Mesh.ScalingFactor = 1e-9), so its contact current is already [A] -- pass depth_m=None.

Sign: DEVSIM reports the current flowing INTO the device through the contact, so the contacts
of a two-terminal resistive device satisfy sum(I) ~ 0 (charge conservation) -- the validation
gate every extraction is checked against.
"""

from __future__ import annotations

from typing import Dict, Optional

import devsim as ds

__all__ = ["extract_contact_currents", "CONTINUITY_EQUATIONS"]

CONTINUITY_EQUATIONS = ("ElectronContinuityEquation", "HoleContinuityEquation")


def extract_contact_currents(device: str, *, depth_m: Optional[float] = None) -> Dict[str, float]:
    """{contact_name: terminal current} for every contact of `device` that carries at least one
    carrier-continuity contact equation (electron + hole summed when both exist). depth_m scales
    a 2D per-unit-depth current [A/m] to [A] (pass the cell period_y_m); None = no scaling (3D
    meshes in metres are already [A]). Contacts with no continuity equation (Poisson-only gates,
    equilibrium devices) are skipped, so an equilibrium solve returns {}."""
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None for an already-in-amperes 3D mesh)")
    out: Dict[str, float] = {}
    for contact in ds.get_contact_list(device=device):
        eqs = set(ds.get_contact_equation_list(device=device, contact=contact))
        present = [eq for eq in CONTINUITY_EQUATIONS if eq in eqs]
        if not present:
            continue                                   # Potential-only contact: no terminal current
        current = 0.0
        for eq in present:
            current += float(ds.get_contact_current(device=device, contact=contact, equation=eq))
        out[str(contact)] = current * (float(depth_m) if depth_m is not None else 1.0)
    return out
