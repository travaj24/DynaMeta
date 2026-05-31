"""
Solver-configuration specs carried by a Design: Stage-1 DEVSIM mesh, Stage-3
NGSolve mesh, and the optical/incidence configuration. Field names are generic
(no 'mirror'/'patch' coupling); the builders key mesh sizing off material role
+ stack position, not off layer names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class Mesh2DSpec:
    """Stage-1 DEVSIM (x, z) mesh refinement, all lengths in metres."""
    x_spacing_edge_m:          float = 30e-9    # near cell edges (peripheral grounds)
    x_spacing_feature_edge_m:  float = 1e-9     # near inclusion lateral edges
    x_spacing_feature_mid_m:   float = 2.5e-9   # inside an inclusion footprint
    coarse_ps_m:               float = 5e-10    # 0.5 nm layer bulk
    interface_ps_m:            float = 5e-11    # 0.05 nm at semiconductor interfaces
    interface_zone_m:          float = 8e-10    # 0.8 nm refinement zone each side
    # DC solve method: "newton" (coupled, default) | "gummel" (decoupled outer
    # iteration -- robust for stiff degenerate gated accumulation that diverges
    # under coupled Newton). Extensible in carriers/dc_solve.py.
    dc_method:                 str   = "newton"


@dataclass
class Mesh3DSpec:
    """Stage-3 NGSolve OCC mesh. Superstrate/substrate MATERIALS come from the
    Stack; these are buffer thicknesses + element sizes only."""
    pml_thk_m:               float = 400e-9
    superstrate_buffer_m:    float = 300e-9     # buffer above stack before top PML
    substrate_buffer_m:      float = 100e-9     # buffer below stack before bottom PML
    maxh_superstrate_m:      float = 120e-9
    maxh_substrate_m:        float = 80e-9
    maxh_pml_m:              float = 200e-9
    maxh_metal_m:            float = 30e-9
    # Thick-metal-film skin/bulk split (applied to the first/last metal film):
    metal_skin_thk_m:        float = 0.0         # 0 disables the split
    maxh_metal_skin_m:       Optional[float] = None
    maxh_metal_bulk_m:       float = 35e-9
    # In-inclusion vs background lateral refinement of cavity layers:
    maxh_inclusion_m:        float = 5e-9
    maxh_background_m:       float = 30e-9
    fem_order:               int = 2
    # Optional prismatic boundary layers grown into semiconductor interfaces:
    semi_prism_thk_m:        Optional[List[float]] = None


@dataclass
class OpticalSpec:
    """Incidence + polarization + linear solver + carrier-field lift control."""
    polarization:        Literal["x", "y"] = "x"
    incidence_angle_deg: float = 0.0
    incidence_side:      Literal["top", "bottom"] = "top"   # which semi-infinite medium
    outputs:             tuple = ("R", "T", "A")            # which to compute/report
    # Carrier-field lift (2D DEVSIM -> 3D eps): "auto" picks SeparableXYLift for a
    # square centered c4v device, else ExtrudeLift; explicit values force a choice
    # and are validated against the device symmetry.
    lift:                Literal["auto", "separable_xy", "extrude", "identity"] = "auto"
    ny_sym:              int = 256
    linear_solver:       Literal["umfpack", "bddc_cg", "bddc_gmres"] = "bddc_gmres"
    gmres_rtol:          float = 1e-6
    gmres_max_iter:      int = 800

    def __post_init__(self) -> None:
        if self.polarization not in ("x", "y"):
            raise ValueError("polarization must be 'x' or 'y'")
        # Oblique incidence (Bloch-Floquet) is implemented for s-pol (E along y,
        # perpendicular to the x-z plane of incidence); p-pol oblique is a follow-up.
        if abs(self.incidence_angle_deg) > 1e-6 and self.polarization != "y":
            raise NotImplementedError(
                "oblique incidence requires polarization='y' (s-pol); p-pol "
                "oblique is a documented follow-up (docs/roadmap_phase5_stretch.md).")
        if self.incidence_side not in ("top", "bottom"):
            raise ValueError("incidence_side must be 'top' or 'bottom'")
        if self.ny_sym < 4:
            raise ValueError("ny_sym must be >= 4")
