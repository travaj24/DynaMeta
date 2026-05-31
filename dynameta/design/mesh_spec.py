"""
Mesh refinement controls for Stage 1 (DEVSIM 2D) and Stage 3 (NGSolve 3D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Stage 1: 2D DEVSIM mesh
# ---------------------------------------------------------------------------

@dataclass
class Mesh2DSpec:
    """Spacing controls for the 2D DEVSIM Cartesian mesh.

    The y (vertical) discretization uses per-layer refinement: coarse
    spacing in the bulk of each layer with fine spacing in a small zone
    around each interface (semiconductor side).

    The x (lateral) discretization places mesh lines at:
      - x = 0      (left x-edge)
      - x = (P - L) / 2  (left patch edge; fine)
      - x = P / 2  (centre; coarse)
      - x = (P + L) / 2  (right patch edge; fine)
      - x = P      (right edge)

    All lengths in metres. Defaults match the Park 2021 metasurface.
    """
    # x-direction targets
    x_spacing_sym_m:         float = 30e-9        # at x = 0, x = P (far from patch)
    x_spacing_patch_edge_m:  float = 1e-9         # near patch edges
    x_spacing_patch_mid_m:   float = 2.5e-9       # under patch interior
    # y-direction (vertical)
    coarse_ps_m:             float = 5e-10        # 0.5 nm in layer bulks
    interface_ps_m:          float = 5e-11        # 0.05 nm at semiconductor interfaces
    interface_zone_m:        float = 8e-10        # 0.8 nm refinement zone each side
    # Patch metal layer (added as a separate region above adhesion)
    patch_thickness_m:       float = 50e-9        # Au patch thickness


# ---------------------------------------------------------------------------
# Stage 3: 3D NGSolve mesh
# ---------------------------------------------------------------------------

@dataclass
class Mesh3DSpec:
    """Spacing controls for the 3D NGSolve unit-cell mesh.

    The cavity (the layers between the mirror and the patch -- typically
    al2o3 + hfo2 + ITO + hfo2 + al2o3) is split into an "in-patch column"
    (fine mesh, ~2.5-5 nm) and an "outside-patch annulus" (coarser).

    Metal layers are skin/bulk split so only the skin gets fine mesh.

    All lengths in metres. Defaults match Park 2021 medium-fidelity.
    """
    pml_thk_m:                    float = 400e-9
    air_buffer_m:                 float = 300e-9
    substrate_thk_m:              float = 100e-9
    # Bulk-region maxh
    maxh_air_m:                   float = 120e-9
    maxh_pml_m:                   float = 200e-9
    maxh_substrate_m:             float = 80e-9
    maxh_metal_m:                 float = 30e-9
    # Cavity column
    maxh_cavity_in_patch_m:       float = 5e-9
    maxh_cavity_outside_m:        float = 30e-9
    # Mirror / patch skin+bulk split
    mirror_skin_thk_m:            float = 15e-9
    maxh_mirror_skin_m:           Optional[float] = None  # falls back to maxh_metal_m
    maxh_mirror_bulk_m:           float = 35e-9
    patch_skin_thk_m:             float = 25e-9
    maxh_patch_skin_m:            Optional[float] = None
    maxh_patch_bulk_m:            float = 15e-9
    # Polynomial order
    fem_order:                    int = 2
    # Prismatic boundary layers in the in-patch semiconductor (ITO): per-layer
    # thicknesses in METRES grown from BOTH ITO/oxide interfaces into the
    # in-patch column, e.g. [0.25e-9, 0.5e-9, 0.75e-9]. Structured prisms give
    # sub-nm z-resolution of the accumulation / ENZ layer (which sets the
    # bias-dependent response) WITHOUT the sliver tets that solid sub-layer
    # splitting produces -- so order-2 HCurl still assembles. None = plain tets.
    ito_prism_thk_m:              Optional[list] = None
