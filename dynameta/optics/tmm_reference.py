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

from dynameta.core.units import NM_PER_M

S = NM_PER_M   # m -> nm (tmm's d_list uses the same unit as the wavelength); single source


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
    lam_nm = float(lambda_m) * S
    d_list = [np.inf] + [float(d) * S for _, d in layers] + [np.inf]
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


# ---- LayeredStack consumer: the graded-TMM oracle + a LayeredStackSolver impl ----

def _pol_for(optical) -> str:
    """Map an OpticalSpec polarization to a tmm 's'/'p'. 'y' (E perp plane) -> s; 'p' -> p;
    'x' -> s (at normal incidence a layered stack is polarization-degenerate)."""
    p = getattr(optical, "polarization", "y")
    return "p" if p == "p" else "s"


def _coh_tmm_stack(stack, lambda_m, theta_deg, pol):
    """Run coherent TMM over a LayeredStack of SCALAR slabs; returns the tmm result dict
    (R, T, r, t, ...). Raises if any slab is laterally structured (then it is not a 1-D
    stack -- use the FEM solver or a future RCWA backend)."""
    import tmm
    if not stack.is_unstructured:
        raise ValueError("layered TMM requires an unstructured (all-scalar-slab) stack; "
                         "a laterally-structured slab needs the FEM solver / RCWA.")
    if abs(complex(stack.n_super).imag) > 1e-9:                # LTM-5: mirror the FEM OPT-1 contract
        raise ValueError("layered TMM: R/T/A and the energy budget A=1-R-T are defined only "
                         "for a LOSSLESS incidence medium (Im(n_super)=0); got n_super={}."
                         .format(stack.n_super))
    n_list = ([complex(stack.n_super)] + [np.sqrt(complex(s.eps)) for s in stack.slabs]
              + [complex(stack.n_sub)])
    d_list = [np.inf] + [float(s.thickness_m) * S for s in stack.slabs] + [np.inf]
    return tmm.coh_tmm(pol, n_list, d_list, math.radians(float(theta_deg)), float(lambda_m) * S)


def layered_rta(stack, lambda_m, *, theta_deg: float = 0.0, pol: str = "s"):
    """(R, T, A) for a LayeredStack of scalar slabs via coherent TMM -- the graded-TMM oracle
    (slice a graded eps(z) with core.layered.slice_profile, then call this)."""
    res = _coh_tmm_stack(stack, lambda_m, theta_deg, pol)
    R, T = float(res["R"]), float(res["T"])
    return R, T, float(1.0 - R - T)


class TmmLayeredSolver:
    """A LayeredStackSolver backed by `tmm` -- the FIRST concrete implementation of the
    layered seam (a future RCWA backend is the second). Exact for unstructured/graded stacks;
    raises on a laterally-structured slab. Maps the 0-order complex r/t to an OpticalResult."""

    def solve(self, stack, lambda_m, optical):
        from dynameta.core.interfaces import OpticalResult
        theta = float(getattr(optical, "incidence_angle_deg", 0.0) or 0.0)
        res = _coh_tmm_stack(stack, lambda_m, theta, _pol_for(optical))
        r, t = complex(res["r"]), complex(res["t"])
        R, T = float(res["R"]), float(res["T"])
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                             solve_time_s=0.0, t=t, T=T, A=float(1.0 - R - T))


def layered_stack_from_design(design, lambda_m, *, eps_by_region=None, n_slices=None):
    """Build a LayeredStack from a Design (+ optionally the bridge's eps_by_region for a graded
    carrier region). Uniform layers -> scalar slabs (material eps at lambda); a semiconductor
    layer whose eps_by_region entry is a gridded EpsField -> sliced via slice_eps_field. A layer
    with material inclusions raises (laterally structured -> FEM/RCWA, not this extractor).
    Slabs are ordered superstrate-side first (the Stack lists bottom->top, so reversed)."""
    from dynameta.core.layered import LayeredStack, LayeredSlab, slice_eps_field
    slabs_top_down = []
    for L in reversed(design.stack.layers):          # superstrate side first
        if L.inclusions:
            raise ValueError("layered_stack_from_design: layer '{}' has inclusions "
                             "(laterally structured); use the FEM solver or RCWA.".format(L.name))
        ef = (eps_by_region or {}).get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            slabs_top_down.extend(reversed(slice_eps_field(ef, 1.0 / S, n_slices=n_slices)))
        else:
            eps = complex(design.materials.get(L.background_material).eps(lambda_m))
            slabs_top_down.append(LayeredSlab(float(L.thickness_m), eps=eps))
    n_super = np.sqrt(complex(design.materials.get(design.stack.superstrate_material).eps(lambda_m)))
    n_sub = np.sqrt(complex(design.materials.get(design.stack.substrate_material).eps(lambda_m)))
    return LayeredStack(complex(n_super), complex(n_sub), slabs_top_down,
                        period_x_m=design.unit_cell.period_x_m,
                        period_y_m=design.unit_cell.period_y_m)


def make_layered_tmm_solver(*, n_slices=None):
    """Build an `optical_solver` callable for `run_pipeline(optical_solver=...)` that solves
    each (bias, wavelength) with the layered TMM instead of the FEM -- the FIRST concrete
    wiring of the pluggable seam (a future RCWA backend plugs in the same way). It assembles
    a LayeredStack from the Design + the bridge's `eps_by_region` (slicing a graded carrier
    region via slice_eps_field) and runs TmmLayeredSolver. Laterally-UNIFORM stacks only -- a
    layer with inclusions, or a laterally-structured eps_by_region entry, raises (use the FEM
    or RCWA).

    The returned callable has the exact seam signature
    ``fn(design, geo, eps_by_region, lam_m, n_super, n_sub) -> OpticalResult``; `geo`,
    `n_super`, `n_sub` are accepted for compatibility but unused (the LayeredStack re-derives
    the end media from the Design's super/substrate materials)."""
    solver = TmmLayeredSolver()

    def _solve(design, geo, eps_by_region, lam_m, n_super, n_sub):
        stack = layered_stack_from_design(design, lam_m, eps_by_region=eps_by_region,
                                          n_slices=n_slices)
        return solver.solve(stack, lam_m, design.optical)

    return _solve
