"""
Design: the complete declarative device = UnitCell + Stack + electrodes +
materials + solver specs. Validated at construction. This is what the default
builders and the pipeline consume; it carries NO solver state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from dynameta.geometry.unit_cell import UnitCell
from dynameta.geometry.stack import Stack, Layer
from dynameta.geometry.electrode import Electrode
from dynameta.geometry.specs import Mesh2DSpec, Mesh3DSpec, OpticalSpec
from dynameta.materials.material import MaterialRegistry


Role = Literal["metal", "semiconductor", "dielectric"]
_SYM_ORDER = {"none": 0, "c2v": 1, "c4v": 2}


@dataclass
class Design:
    name:        str
    unit_cell:   UnitCell
    stack:       Stack
    electrodes:  List[Electrode]
    materials:   MaterialRegistry
    mesh_2d:     Mesh2DSpec = field(default_factory=Mesh2DSpec)
    mesh_3d:     Mesh3DSpec = field(default_factory=Mesh3DSpec)
    optical:     OpticalSpec = field(default_factory=OpticalSpec)
    pretty_name: str = ""

    def __post_init__(self) -> None:
        if not self.pretty_name:
            self.pretty_name = self.name
        # Every referenced material must be registered.
        used = {self.stack.superstrate_material, self.stack.substrate_material}
        for L in self.stack.layers:
            used.update(L.materials_used())
        for f in self.stack.features:
            used.add(f.material)
        for mat in used:
            if mat not in self.materials:
                raise ValueError("Material '{}' used in stack is not registered "
                                  "(known: {})".format(mat, self.materials.names()))
        # Electrodes reference existing layers; names unique.
        layer_names = {L.name for L in self.stack.layers}
        for E in self.electrodes:
            if E.layer not in layer_names:
                raise ValueError("Electrode '{}' references unknown layer '{}'"
                                  .format(E.name, E.layer))
        enames = [E.name for E in self.electrodes]
        if len(set(enames)) != len(enames):
            raise ValueError("Duplicate electrode names: {}".format(enames))

    # ---- geometry helpers ----
    def z_intervals(self) -> Dict[str, Tuple[float, float]]:
        return self.stack.z_intervals()

    def material_role(self, material_name: str) -> Role:
        m = self.materials.get(material_name)
        if m.is_metal:
            return "metal"
        if m.is_semiconductor:
            return "semiconductor"
        return "dielectric"

    def semiconductor_layers(self) -> List[Layer]:
        """Layers whose background or any inclusion is a semiconductor."""
        out = []
        for L in self.stack.layers:
            if any(self.material_role(m) == "semiconductor"
                    for m in L.materials_used()):
                out.append(L)
        return out

    def device_symmetry(self) -> str:
        """Coarse device point-group ('c4v' | 'c2v' | 'none'), the intersection
        of the lattice symmetry with every inclusion's centered symmetry. Used
        to gate the carrier-field lift (SeparableXYLift requires 'c4v')."""
        sym = self.unit_cell.lattice_symmetry()
        cx, cy = self.unit_cell.center_m
        for L in self.stack.layers:
            for inc in L.inclusions:
                scx, scy = inc.shape.center_m()
                if abs(scx - cx) > 1e-12 or abs(scy - cy) > 1e-12:
                    return "none"
                s = inc.shape.intrinsic_symmetry()
                if _SYM_ORDER[s] < _SYM_ORDER[sym]:
                    sym = s
        return sym
