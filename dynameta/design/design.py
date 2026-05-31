"""
Design: top-level dataclass binding geometry, materials, electrodes,
and mesh / optical specs into one immutable object.

A Design is what gets passed to run_full_pipeline(); everything else in
the library reads its information from a Design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dynameta.design.electrode import Electrode
from dynameta.design.layer import Layer
from dynameta.design.materials import Material, MaterialRegistry
from dynameta.design.mesh_spec import Mesh2DSpec, Mesh3DSpec
from dynameta.design.optical_spec import OpticalSpec


@dataclass
class Design:
    """Complete declarative specification of a metasurface device.

    A Design is fully validated at construction; if it builds, all
    consistency checks have passed.

    Args:
      name              : identifier used for output directory naming
      period_m          : unit-cell period in x and y (square unit cell only)
      patch_side_m      : square patch side length (applies to layers
                            with lateral_extent='patch_footprint')
      layers            : ordered bottom-to-top stack of Layer instances
      electrodes        : DEVSIM contacts (must reference Layer names)
      materials         : MaterialRegistry containing every material name
                            referenced by layers
      mesh_2d           : Stage 1 DEVSIM mesh spec
      mesh_3d           : Stage 3 NGSolve mesh spec
      optical           : Stage 3 incidence / polarization / solver
      pretty_name       : display name for plots / docs
    """
    name:           str
    period_m:       float
    patch_side_m:   float
    layers:         List[Layer]
    electrodes:     List[Electrode]
    materials:      MaterialRegistry
    mesh_2d:        Mesh2DSpec = field(default_factory=Mesh2DSpec)
    mesh_3d:        Mesh3DSpec = field(default_factory=Mesh3DSpec)
    optical:        OpticalSpec = field(default_factory=OpticalSpec)
    pretty_name:    str = ""

    def __post_init__(self) -> None:
        if not self.pretty_name:
            self.pretty_name = self.name
        if self.period_m <= 0 or self.patch_side_m <= 0:
            raise ValueError("period_m and patch_side_m must be positive")
        if self.patch_side_m > self.period_m:
            raise ValueError("patch_side_m ({}) > period_m ({})".format(
                self.patch_side_m, self.period_m))
        if not self.layers:
            raise ValueError("Design requires at least one Layer")
        # Layer name uniqueness
        layer_names = [L.name for L in self.layers]
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("Duplicate Layer names: {}".format(layer_names))
        # Every layer must reference a registered material
        for L in self.layers:
            if L.material not in self.materials:
                raise ValueError("Layer '{}' references unknown material '{}'"
                                  .format(L.name, L.material))
        # Electrode attached_layer must exist
        layer_set = set(layer_names)
        for E in self.electrodes:
            if E.attached_layer not in layer_set:
                raise ValueError("Electrode '{}' attached_layer '{}' not in "
                                  "layers".format(E.name, E.attached_layer))
        # Electrode name uniqueness
        elec_names = [E.name for E in self.electrodes]
        if len(set(elec_names)) != len(elec_names):
            raise ValueError("Duplicate Electrode names: {}".format(elec_names))

    # -----------------------------------------------------------------
    # Geometry helpers (used by the stage builders)
    # -----------------------------------------------------------------

    def layer_z_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Return {layer_name: (z_lo_m, z_hi_m)} for the stack, bottom-to-top.

        Note: layers with lateral_extent != full_period are still stacked
        in this list -- they sit ABOVE the previous layer's top in the
        order they appear in `self.layers`.
        """
        z = 0.0
        out: Dict[str, Tuple[float, float]] = {}
        for L in self.layers:
            out[L.name] = (z, z + L.thickness_m)
            z += L.thickness_m
        return out

    def find_layer(self, name: str) -> Layer:
        for L in self.layers:
            if L.name == name:
                return L
        raise KeyError("No layer named '{}' in design".format(name))

    def find_electrode(self, name: str) -> Electrode:
        for E in self.electrodes:
            if E.name == name:
                return E
        raise KeyError("No electrode named '{}' in design".format(name))

    def semiconductor_layers(self) -> List[Layer]:
        return [L for L in self.layers if L.role == "semiconductor"]

    def metal_layers(self) -> List[Layer]:
        return [L for L in self.layers if L.role == "metal"]

    def dielectric_layers(self) -> List[Layer]:
        return [L for L in self.layers if L.role == "dielectric"]
