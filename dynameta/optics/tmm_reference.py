"""TMM reference: exact coherent transfer-matrix R/T/A for an UNSTRUCTURED layered
stack, via the `tmm` library. Two uses:
  * a fast PATH -- when a unit cell has no lateral structure (a plain slab stack), this
    is exact and ~instant, no FEM mesh/solve needed;
  * an ORACLE -- a per-bias / per-wavelength cross-check for the FEM solver on the
    laterally-uniform limit (the FEM must reproduce it).

This is solver-agnostic (only numpy + tmm). For LATERALLY-STRUCTURED cells (patches,
gratings) TMM does not apply -- use the FEM solver (or a future RCWA backend).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def stack_rta(n_super: complex, layers: Sequence[Tuple[complex, float]], n_sub: complex,
              lambda_m: float, *, theta_deg: float = 0.0, pol: str = "s") -> Tuple[float, float, float]:
    """Coherent-TMM (R, T, A) for super | layers | sub at wavelength lambda_m.

    Args:
      n_super, n_sub : semi-infinite incidence/exit refractive indices (sqrt(eps)).
      layers         : ordered [(n, thickness_m), ...] from the super side to the sub side.
      lambda_m       : vacuum wavelength (m).
      theta_deg      : incidence angle in the SUPERSTRATE (deg).
      pol            : 's' or 'p'.

    Returns (R, T, A) with A = 1 - R - T (TMM's exact absorbed fraction). T already
    carries the correct angle/index power factor (tmm handles it).
    """
    import tmm
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")
    n_list = [complex(n_super)] + [complex(n) for n, _ in layers] + [complex(n_sub)]
    # tmm wants d in the SAME unit as the wavelength; use nm for both. Ends are semi-infinite.
    lam_nm = float(lambda_m) * 1e9
    d_list = [np.inf] + [float(d) * 1e9 for _, d in layers] + [np.inf]
    res = tmm.coh_tmm(pol, n_list, d_list, math.radians(float(theta_deg)), lam_nm)
    R = float(res["R"])
    T = float(res["T"])
    return R, T, float(1.0 - R - T)


def design_layer_stack(design, lambda_m: float) -> Tuple[complex, List[Tuple[complex, float]], complex]:
    """Extract (n_super, [(n, thk_m), ...], n_sub) from a Design whose layers are all
    laterally UNIFORM (no inclusions) -- so TMM applies. Raises if any layer has an
    inclusion (then it is a metasurface, not a 1D stack; use the FEM solver). The
    per-layer index is sqrt(eps(material, lambda)) at zero bias (density-independent
    materials); for a carrier-modulated layer pass the biased eps yourself."""
    layers = []
    for L in design.stack.layers:
        if L.inclusions:
            raise ValueError(
                "design_layer_stack: layer '{}' has inclusions -- the cell is laterally "
                "structured and TMM does not apply; use the FEM solver.".format(L.name))
        eps = complex(design.materials.get(L.background_material).eps(lambda_m))
        layers.append((np.sqrt(eps), float(L.thickness_m)))
    n_super = np.sqrt(complex(design.materials.get(design.stack.superstrate_material).eps(lambda_m)))
    n_sub = np.sqrt(complex(design.materials.get(design.stack.substrate_material).eps(lambda_m)))
    return complex(n_super), layers, complex(n_sub)
