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


def _passive_sqrt(eps) -> complex:
    """n = sqrt(eps) on the PASSIVE branch (Im(n) >= 0, decaying). For a LOSSLESS medium with
    Re(eps) < 0 (an ideal metal / ENZ below plasma) the sign of the +/-0.0j imaginary part decides
    the branch: np.sqrt(-2+0.0j)=+1.41j (decaying, physical) but np.sqrt(-2-0.0j)=-1.41j (GAIN). A
    spurious -0.0j (from a fit/interp/ConstantOptical) would silently flip a metal into a gain
    medium, so when Im(eps) is negligible we force Im(n) >= 0. A GENUINELY lossy/gain eps (|Im| not
    negligible) is left to the principal branch and caught downstream by the energy-budget guard."""
    eps = complex(eps)
    n = np.sqrt(eps)
    if abs(eps.imag) <= 1e-12 * (abs(eps.real) + 1.0) and n.imag < 0.0:
        n = -n
    return complex(n)


def _require_theta(theta_deg) -> float:
    """Validate a real incidence angle in [0, 90) deg (>=90 is grazing/unphysical for a lossless
    superstrate; tmm's internal assert there is opaque). Returns the float angle."""
    t = float(theta_deg)
    if not (0.0 <= t < 90.0):
        raise ValueError("theta_deg must be in [0, 90) degrees; got {}.".format(theta_deg))
    return t


def _check_energy_budget(R: float, T: float, *, where: str, tol: float = 1e-6) -> float:
    """Return A = 1 - R - T, but RAISE if the energy budget is violated (R, T, or A < -tol). tmm's
    gain-guard only checks the two semi-infinite END media; an INTERIOR slab with Im(eps) < 0 (a
    sign/convention mistake = gain) silently yields T > 1, A < 0. Since this module is the FEM
    validation ORACLE, a wrong-but-plausible R/T must not pass silently."""
    A = 1.0 - R - T
    # NaN must NOT pass (nan < -tol is False); the oracle has to be at least as strict as the FEM
    # backstop in solver.py. A finite negative R/T/A flags an interior gain layer (Im(eps) < 0).
    if not (math.isfinite(R) and math.isfinite(T)) or R < -tol or T < -tol or A < -tol:
        raise ValueError(
            "{}: TMM R/T/A not finite or energy budget violated (R={}, T={}, A=1-R-T={}); a layer "
            "likely has Im(eps) < 0 (GAIN) -- check the eps sign convention (exp(-iwt) => Im(eps) "
            ">= 0 for a passive/absorbing medium).".format(where, R, T, A))
    return A


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
    carries the correct angle/index power factor (tmm handles it). RAISES if the energy
    budget is violated (an interior gain layer; see _check_energy_budget).

    NOTE: for a near-opaque layer (Im(delta) > 35) tmm clamps the layer and emits a one-time
    stdout notice (it modifies the layer to "let ~1 photon in 1e30 through"); the numerical
    effect on R/T is negligible (T ~ 1e-31).
    """
    import tmm
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")
    theta_deg = _require_theta(theta_deg)
    if abs(complex(n_super).imag) > 1e-9:                      # mirror _coh_tmm_stack's LTM-5 guard
        raise ValueError("stack_rta: R/T/A and the energy budget A=1-R-T are defined only for a "
                         "LOSSLESS incidence medium (Im(n_super)=0); got n_super={}.".format(n_super))
    n_list = [complex(n_super)] + [complex(n) for n, _ in layers] + [complex(n_sub)]
    # tmm wants d in the SAME unit as the wavelength; use nm for both. Ends are semi-infinite.
    lam_nm = float(lambda_m) * S
    d_list = [np.inf] + [float(d) * S for _, d in layers] + [np.inf]
    res = tmm.coh_tmm(pol, n_list, d_list, math.radians(theta_deg), lam_nm)
    R = float(res["R"])
    T = float(res["T"])
    return R, T, _check_energy_budget(R, T, where="stack_rta")


def end_media_indices(design, lambda_m: float) -> Tuple[complex, complex]:
    """(n_super, n_sub) = sqrt(eps) of the Design's semi-infinite superstrate/substrate media
    at lambda_m. One helper so the complex-sqrt branch (which matters for a lossy cladding) is
    chosen in exactly one place -- used by run_pipeline and both layered extractors."""
    n_super = _passive_sqrt(design.materials.get(design.stack.superstrate_material).eps(lambda_m))
    n_sub = _passive_sqrt(design.materials.get(design.stack.substrate_material).eps(lambda_m))
    return n_super, n_sub


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
        eps = design.materials.get(L.background_material).eps(lambda_m)
        layers.append((_passive_sqrt(eps), float(L.thickness_m)))
    n_super, n_sub = end_media_indices(design, lambda_m)
    return n_super, layers, n_sub


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
    theta_deg = _require_theta(theta_deg)
    if abs(complex(stack.n_super).imag) > 1e-9:                # LTM-5: mirror the FEM OPT-1 contract
        raise ValueError("layered TMM: R/T/A and the energy budget A=1-R-T are defined only "
                         "for a LOSSLESS incidence medium (Im(n_super)=0); got n_super={}."
                         .format(stack.n_super))
    n_list = ([complex(stack.n_super)] + [_passive_sqrt(s.eps) for s in stack.slabs]
              + [complex(stack.n_sub)])
    d_list = [np.inf] + [float(s.thickness_m) * S for s in stack.slabs] + [np.inf]
    return tmm.coh_tmm(pol, n_list, d_list, math.radians(theta_deg), float(lambda_m) * S)


def layered_rta(stack, lambda_m, *, theta_deg: float = 0.0, pol: str = "s"):
    """(R, T, A) for a LayeredStack of scalar slabs via coherent TMM -- the graded-TMM oracle
    (slice a graded eps(z) with core.layered.slice_profile, then call this)."""
    res = _coh_tmm_stack(stack, lambda_m, theta_deg, pol)
    R, T = float(res["R"]), float(res["T"])
    return R, T, _check_energy_budget(R, T, where="layered_rta")


def _per_layer_absorption(coh_tmm_res):
    """Per-slab absorbed fractions from a coh_tmm result (driver D2): tmm.absorp_in_each_layer
    returns [reflection-side semi-infinite, slab_0, ..., slab_{n-1}, transmission-side
    semi-infinite]; the two ends are stripped (for a LOSSLESS lossy-free incidence medium the
    first entry is R and the last is T -- they are NOT absorption). Keys 'slab_<i>' index
    stack.slabs top-first. sum(values) == A == 1 - R - T to library precision."""
    import tmm
    arr = np.asarray(tmm.absorp_in_each_layer(coh_tmm_res), dtype=np.float64)
    return {"slab_{}".format(i): float(arr[i + 1]) for i in range(arr.size - 2)}


def layered_per_layer_absorption(stack, lambda_m, *, theta_deg: float = 0.0, pol: str = "s"):
    """({'slab_<i>': absorbed fraction}, A_total) for a LayeredStack via coherent TMM -- the
    per-layer absorbed-power driver (D2) on the exact 1-D path; the dict sums to A_total."""
    res = _coh_tmm_stack(stack, lambda_m, theta_deg, pol)
    R, T = float(res["R"]), float(res["T"])
    return _per_layer_absorption(res), _check_energy_budget(R, T,
                                                            where="layered_per_layer_absorption")


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
        A = _check_energy_budget(R, T, where="TmmLayeredSolver")
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                             solve_time_s=0.0, t=t, T=T, A=A,
                             per_region_absorption=_per_layer_absorption(res))


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
    n_super, n_sub = end_media_indices(design, lambda_m)
    return LayeredStack(n_super, n_sub, slabs_top_down,
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
    ``fn(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult``; `geo`,
    `n_super`, `n_sub` are accepted for compatibility but unused (the LayeredStack re-derives
    the end media from the Design's super/substrate materials)."""
    solver = TmmLayeredSolver()

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        stack = layered_stack_from_design(design, lambda_m, eps_by_region=eps_by_region,
                                          n_slices=n_slices)
        return solver.solve(stack, lambda_m, design.optical)

    return _solve
