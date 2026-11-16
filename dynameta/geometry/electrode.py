"""
Electrode: a DEVSIM contact attached to a layer. Its lateral footprint is
either a CrossSection (e.g. the patch footprint), an edge selector (a thin
strip on a cell edge, for peripheral grounds), or "full" (the whole layer
face). The DEVSIM builder resolves the footprint to a contact bbox.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Literal, Optional, Union

from dynameta.geometry.cross_section import CrossSection


EdgeSelector = Literal["x_lo", "x_hi", "y_lo", "y_hi"]
ElectrodeRole = Literal["biased", "ground"]
Footprint = Union[CrossSection, str]   # CrossSection | EdgeSelector | "full"


@dataclass
class Electrode:
    """A contact attached to `layer`.

    work_function_eV : (v0.16) metal work function phi_m [eV] of a GATE electrode (non-edge footprint).
                       None (default) = the gate's flat-band voltage is 0 by construction (v0.15.1). When
                       set, the builders pin the gate at Potential = bias - (phi_m - W_s)/q with W_s = chi +
                       (E_c - E_F) of the semiconductor frame reference (physics_equilibrium.
                       work_function_offset_V; needs that material's TransportModel.chi_eV). Refused on an
                       edge electrode, which is an ohmic contact to the semiconductor.
    pin_potential    : (v0.16) edge electrodes only. On an EQUILIBRIUM semiconductor region an edge
                       electrode is the region's Fermi-level reference (no Dirichlet potential: in equilibrium
                       physics no current flows, so the electrode sets the Fermi level, not a potential); True
                       keeps the v0.15.1 contact: a thin contact slab at the cell boundary over the
                       layer's z range, in which DEVSIM captures only the two corner nodes of the edge column
                       (MEASURED: "contact ... with 2 nodes"). Those two nodes are pinned to the electrode's
                       potential and the rest of the column floats; on a gated film the corners are the film's
                       two interface nodes, i.e. n = n_bg is forced exactly where the accumulation layer forms.
                       A single-layer region with no neighbouring region captures no node at all and DEVSIM
                       drops the contact. Drift-diffusion and bipolar regions always get their ohmic contact.
    """
    name:             str
    layer:            str                 # attached layer name
    footprint:        Footprint           # CrossSection | "x_lo"/"x_hi"/"y_lo"/"y_hi" | "full"
    role:             ElectrodeRole = "biased"
    fixed_voltage_V:  float = 0.0         # used when role == "ground"
    work_function_eV: Optional[float] = None
    pin_potential:    bool = False

    def __post_init__(self) -> None:
        if isinstance(self.footprint, str):
            if self.footprint not in ("x_lo", "x_hi", "y_lo", "y_hi", "full"):
                raise ValueError(
                    "Electrode '{}' string footprint must be one of "
                    "x_lo/x_hi/y_lo/y_hi/full (got {!r})".format(
                        self.name, self.footprint))
        elif not isinstance(self.footprint, CrossSection):
            raise ValueError(
                "Electrode '{}' footprint must be a CrossSection or a "
                "selector string".format(self.name))
        if self.role not in ("biased", "ground"):
            raise ValueError("Electrode role must be 'biased' or 'ground'")
        if not isinstance(self.pin_potential, bool):
            raise TypeError("Electrode '{}' pin_potential must be a bool, got {!r}".format(
                self.name, self.pin_potential))
        if self.pin_potential and not self.is_edge:
            raise ValueError("Electrode '{}': pin_potential applies to edge electrodes (x_lo/x_hi/y_lo/y_hi) "
                             "only; a gate always pins its potential".format(self.name))
        if self.work_function_eV is not None:
            wf = self.work_function_eV
            if isinstance(wf, bool) or not isinstance(wf, numbers.Real) or not math.isfinite(float(wf)) \
                    or float(wf) <= 0.0:
                raise ValueError("Electrode '{}' work_function_eV must be a finite number > 0 (eV), got {!r}"
                                 .format(self.name, wf))
            if self.is_edge:
                raise ValueError("Electrode '{}': work_function_eV applies to gate electrodes; an edge "
                                 "electrode is an ohmic contact to the semiconductor (its Fermi level is the "
                                 "semiconductor's)".format(self.name))

    @property
    def is_edge(self) -> bool:
        return isinstance(self.footprint, str) and self.footprint != "full"
