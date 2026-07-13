"""Lumenairy RCWA as a DynaMeta optical backend (roadmap v0.5 A1).

BRIDGE, not vendor: Lumenairy (>= 5.21, the single floor in _common.VERSION_FLOOR) is a REQUIRED dependency of dynameta but is
imported lazily inside the functions -- importing this module (and base dynameta) stays
fast and matplotlib-free; calling against a broken/outdated environment raises with an
install hint. Conventions are IDENTICAL on both sides (public exp(-i omega t), Im(eps) > 0 for
absorbers, metres, radians) -- verified in docs/lumenairy_rcwa_port_wishlist.md -- so no
sign/unit translation happens here, only geometry/result adaptation.

Pinned cross-library contracts (first pinned at lumenairy 5.14.2, re-verified on the 5.21 floor; file:line refs in the roadmap):
- RCWAStack(period, period_y=, n_superstrate=, n_substrate=, n_orders=, n_orders_y=):
  region media are complex refractive INDICES (or callables wl -> n); a stack is 1-D iff
  period_y/n_orders_y are omitted; the lattice is rectangular.
- add_layer(thickness, eps=|eps_cell=|eps_tensor_cell=|shapes=+eps_background=,
  formulation=): eps_cell is (Sx, Sy) with axis 0 = x; patterned cells need
  Sx >= 4 n_orders_x + 1 (and y alike); 'li'/'fff' formulation is isotropic-eps_cell-only.
- set_source(wavelength, theta=, phi=) in metres/radians; solve(retain_internal=,
  stabilize=) -> RCWAResult; efficiencies() rows are keyed INCIDENT E_x (row 0) /
  E_y (row 1) -- never relabel as TE/TM; jones_reflection()/jones_transmission() are the
  ZEROTH-order (2, 2) lab-basis Jones, columns = incident Ex/Ey.
- A stack holding any dispersive callable must be solved via solve_vs_wavelength; this
  bridge instead rebuilds a CONCRETE stack per wavelength (the pipeline's assemble_at
  closure is the dispersive source of truth, including carrier-modulated eps), which also
  fixes the band-centre end-media freeze of the generic sweep path: end media are
  re-derived per wavelength via end_media_indices.

Polarization mapping (DynaMeta OpticalSpec 'x'|'y'|'p' -> result row): 'x' -> row 0,
'y' -> row 1, 'p' -> row 0 (plane of incidence is x-z at phi = 0, so the in-plane TM field
is x-polarized at normal/oblique non-conical incidence). OpticalResult.R/T are the TOTAL
(co- + cross-polarized) order-summed efficiencies (also exposed as R_flux/T_flux -- for
RCWA they are the same Poynting quantities); r/t are the CO-polarized zeroth-order complex
amplitudes (phase-bearing, the modulator observable).

Inclusion handling: laterally structured layers are RASTERIZED onto a (Sx, Sy) cell with
the validated shared rasterizer (optics.rasterize, also the structured-FDTD painter:
background + priority-ordered inclusion overpainting).
Inclusion eps comes from the material registry at lambda; an eps_by_region override applies
to the layer BACKGROUND only (rasterizer contract). The Lumenairy analytic-shapes fast path
is a documented follow-on (requires the shape-frame pin + disjointness mapping).
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.core.layered import (LayeredStack, collapse_regions_to_layers,
                                   slice_eps_field)
from dynameta.optics.rasterize import cell_axes, layer_eps_cell
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices

__all__ = ["design_to_rcwa_stack", "make_lumenairy_rcwa_solver", "LumenairyStackSolver",
           "rcwa_result_to_optical_result"]

# Shared plumbing now lives in _common.py (audit 6.3: rcwa_backend had become the bridge's
# unofficial common module). Underscore aliases keep the historical import surface working.
from dynameta.optics.lumenairy_bridge._common import (angles_rad as _angles_rad,
                                                      guard_conical_ppol as _guard_conical_ppol,
                                                      guard_incidence_side as _guard_incidence_side,
                                                      p_basis_conversion as _p_basis_conversion,
                                                      pol_row as _pol_row,
                                                      require_lumenairy as _require_lumenairy)


def _min_samples(n_orders: int) -> int:
    return 4 * int(n_orders) + 1


def design_to_rcwa_stack(design, lambda_m: float, *, eps_by_region=None, n_orders: int = 11,
                         n_orders_y: Optional[int] = None, n_slices: Optional[int] = None,
                         cell_samples: Optional[int] = None, formulation: str = "laurent"):
    """Translate a DynaMeta Design (+ optionally the bridge's eps_by_region) into a CONCRETE
    Lumenairy RCWAStack at one wavelength. Returns (stack, layer_names_top_first) where the
    name list is aligned with the Lumenairy layers (graded fields contribute several slabs
    per design layer) -- the key for mapping per-layer absorption back to design layers.

    Layer dispatch, in order: a structured (gridded) EpsField -> sliced eps_cell /
    eps_tensor_cell slabs; a uniform tensor EpsField -> a tiled eps_tensor_cell; a uniform
    scalar EpsField -> eps; a graded laterally-uniform EpsField -> sliced uniform slabs;
    inclusions -> a rasterized eps_cell; else the material eps at lambda. Design layers are
    bottom -> top, the RCWA stack wants superstrate-side first -> reversed()."""
    lum = _require_lumenairy()
    n_super, n_sub = end_media_indices(design, lambda_m)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    # mesh-region-keyed dicts (the run_pipeline/FEM bridge output: 'ito_inpatch',
    # 'grating__incl0', ...) collapse to design-layer keys; identity when already layer-keyed
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})

    def _y_invariant(ef):
        v = ef.values_zyx
        return v is None or v.shape[1] == 1 or bool(
            np.allclose(v, v[:, :1], rtol=1e-12, atol=0.0))

    def _lamellar(L):
        """All inclusions are full-y rectangles (y-invariant grating lines) -> the layer is
        1-D-compatible and the stack can stay 1-D (a genuinely cheaper and
        better-conditioned solve than a y-degenerate 2-D one)."""
        for inc in L.inclusions:
            if getattr(inc.shape, "kind", "") != "rectangle":
                return False
            _, _, ylo, yhi = inc.shape.bbox_m()
            if ylo > 1e-12 * py or yhi < py * (1.0 - 1e-12):
                return False
        return True

    def _structured(L):
        ef = eps_by_region.get(L.name)
        grid = ef is not None and not getattr(ef, "is_uniform", True)
        utensor = ef is not None and getattr(ef, "is_tensor", False)
        return bool(L.inclusions) or grid or utensor

    def _needs_2d(L):
        ef = eps_by_region.get(L.name)
        grid_2d = (ef is not None and not getattr(ef, "is_uniform", True)
                   and ef.values_zyx is not None and not _y_invariant(ef))
        utensor = ef is not None and getattr(ef, "is_tensor", False)
        return (bool(L.inclusions) and not _lamellar(L)) or grid_2d or utensor

    structured = any(_structured(L) for L in design.stack.layers)
    is_2d = any(_needs_2d(L) for L in design.stack.layers)
    if is_2d:
        stack = lum.RCWAStack(px, period_y=py, n_superstrate=complex(n_super),
                              n_substrate=complex(n_sub), n_orders=int(n_orders),
                              n_orders_y=int(n_orders_y if n_orders_y is not None
                                             else n_orders))
        nx_o, ny_o = stack.n_orders_x, stack.n_orders_y
    else:
        stack = lum.RCWAStack(px, n_superstrate=complex(n_super),
                              n_substrate=complex(n_sub), n_orders=int(n_orders))
        nx_o, ny_o = stack.n_orders_x, 0

    s_min_x = _min_samples(nx_o)
    s_min_y = _min_samples(ny_o) if is_2d else 1
    sx = max(int(cell_samples or 0), s_min_x, 128 if structured else s_min_x)
    sy = max(int(cell_samples or 0), s_min_y, 128) if is_2d else 1

    names: List[str] = []
    for L in reversed(design.stack.layers):              # superstrate side first
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            # gridded EpsField (carrier/effect-graded; axes in nm solver units like the
            # TMM extractor) -> slice_eps_field returns slabs in ASCENDING z
            # (substrate-side first); this stack is superstrate-first, so reverse
            # (audit C5-1: unreversed slabs vertically flip the graded profile)
            for slab in reversed(slice_eps_field(ef, 1.0 / _S_NM, n_slices=n_slices)):
                if slab.eps is not None:
                    stack.add_layer(slab.thickness_m, eps=complex(slab.eps))
                elif slab.eps_cell is not None:
                    cell = np.asarray(slab.eps_cell, dtype=complex)
                    if not is_2d and cell.shape[1] > 1:
                        cell = cell[:, :1]                # y-invariant grid on a 1-D stack
                    stack.add_layer(slab.thickness_m,
                                    eps_cell=_meet_sampling(cell, s_min_x, s_min_y),
                                    formulation=formulation)
                else:
                    cell = np.asarray(slab.eps_tensor_cell, dtype=complex)
                    if not is_2d and cell.shape[1] > 1:
                        cell = cell[:, :1]
                    stack.add_layer(slab.thickness_m,
                                    eps_tensor_cell=_meet_sampling(cell, s_min_x, s_min_y))
                names.append(L.name)
            continue
        if ef is not None and getattr(ef, "is_tensor", False):
            cell = np.broadcast_to(np.asarray(ef.tensor, dtype=complex),
                                   (s_min_x, max(s_min_y, 1), 3, 3)).copy()
            stack.add_layer(L.thickness_m, eps_tensor_cell=cell)
            names.append(L.name)
            continue
        if L.inclusions:
            x, y = cell_axes(sx, sy, px, py)
            X, Y = np.meshgrid(x, y, indexing="ij")
            cell = layer_eps_cell(L, X, Y, lambda_m, design.materials, eps_by_region)
            stack.add_layer(L.thickness_m, eps_cell=np.asarray(cell, dtype=complex),
                            formulation=formulation)
            names.append(L.name)
            continue
        if ef is not None:                               # uniform scalar (effect-modulated)
            stack.add_layer(L.thickness_m, eps=complex(ef.scalar))
        else:
            stack.add_layer(L.thickness_m,
                            eps=complex(design.materials.get(L.background_material)
                                        .eps(lambda_m)))
        names.append(L.name)
    return stack, names


def _meet_sampling(cell, s_min_x: int, s_min_y: int):
    """Tile a too-coarse UNIFORM-pattern axis up to Lumenairy's sampling bound. Genuinely
    patterned axes that violate the bound raise downstream with Lumenairy's own message
    (refine the source grid or lower n_orders)."""
    cell = np.asarray(cell, dtype=complex)
    reps = [max(1, int(np.ceil(s_min_x / cell.shape[0]))),
            max(1, int(np.ceil(s_min_y / max(cell.shape[1], 1))))]
    if reps[0] > 1 or reps[1] > 1:
        uniform_x = bool(np.all(cell == cell[:1, ...]))
        uniform_y = bool(np.all(cell == cell[:, :1, ...]))
        reps[0] = reps[0] if uniform_x else 1
        reps[1] = reps[1] if uniform_y else 1
        tile = reps + [1] * (cell.ndim - 2)
        cell = np.tile(cell, tile)
    return cell


def rcwa_result_to_optical_result(res, row: int, *, t0: float,
                                  layer_names: Optional[List[str]] = None,
                                  absorption: bool = False,
                                  r_factor: complex = 1.0,
                                  t_factor: complex = 1.0) -> OpticalResult:
    """Map an RCWAResult to a DynaMeta OpticalResult for ONE incident polarization row.
    R/T = TOTAL order-summed efficiencies (== the Poynting R_flux/T_flux); r/t = co-polarized
    zeroth-order complex Jones, multiplied by the p-basis conversion factors when applicable
    (_p_basis_conversion); A = 1 - R - T (budget); with absorption=True, per-layer absorption
    (independent volumetric loss) fills per_region_absorption keyed by DESIGN layer name
    (slabs of a graded layer are summed) and its total fills A_independent."""
    orders, R2, T2 = res.efficiencies()
    R = float(np.sum(R2[row]))
    T = float(np.sum(T2[row]))
    jr = np.asarray(res.jones_reflection())
    jt = np.asarray(res.jones_transmission())
    r = complex(jr[row, row]) * complex(r_factor)
    t = complex(jt[row, row]) * complex(t_factor)
    A = 1.0 - R - T
    a_ind, pra = None, None
    if absorption and layer_names is not None:
        try:
            la = np.asarray(res.layer_absorption())
            if la.ndim == 2:                              # (2, n_layers) per incident pol
                la = la[row] if la.shape[0] == 2 else la[:, row]
            pra = {}
            for name, val in zip(layer_names, la):
                pra[name] = pra.get(name, 0.0) + float(val)
            a_ind = float(np.sum(la))
        except Exception as exc:                          # pragma: no cover - defensive
            warnings.warn("lumenairy bridge: per-layer absorption unavailable ({}); "
                          "A_independent left unset".format(exc), stacklevel=2)
    return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                         solve_time_s=time.perf_counter() - t0, t=t, T=T, A=A,
                         A_independent=a_ind, R_flux=R, T_flux=T,
                         per_region_absorption=pra)


def make_lumenairy_rcwa_solver(*, n_orders: int = 11, n_orders_y: Optional[int] = None,
                               n_slices: Optional[int] = None,
                               cell_samples: Optional[int] = None,
                               absorption: bool = False, stabilize: bool = False,
                               formulation: str = "laurent"):
    """Build an `optical_solver` for run_pipeline backed by Lumenairy RCWA, with the exact
    seam signature fn(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult
    PLUS the sweep-aware `solve_sweep` fast path (one result per wavelength, in order; end
    media re-derived PER WAVELENGTH, fixing the generic band-centre freeze). geo and the
    passed n_super/n_sub are accepted for seam compatibility but unused (the stack re-derives
    end media from the Design, the make_layered_tmm_solver precedent)."""

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        theta, phi = _angles_rad(design.optical)
        _guard_incidence_side(design.optical)
        _guard_conical_ppol(design.optical, phi)
        stack, names = design_to_rcwa_stack(design, lambda_m, eps_by_region=eps_by_region,
                                            n_orders=n_orders, n_orders_y=n_orders_y,
                                            n_slices=n_slices, cell_samples=cell_samples,
                                            formulation=formulation)
        stack.set_source(lambda_m, theta=theta, phi=phi)
        res = stack.solve(retain_internal=absorption, stabilize=stabilize)
        n_sup, n_sb = end_media_indices(design, lambda_m)
        rf, tf = _p_basis_conversion(getattr(design.optical, "polarization", "y"),
                                     theta, n_sup, n_sb)
        return rcwa_result_to_optical_result(res, _pol_row(design.optical), t0=t0,
                                             layer_names=names, absorption=absorption,
                                             r_factor=rf, t_factor=tf)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        return [_solve_at(design, assemble_at(lam), lam) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    return _solve


class LumenairyStackSolver:
    """LayeredStackSolver backed by Lumenairy RCWA -- the SECOND concrete implementation of
    the layered seam (TmmLayeredSolver is the first) and the first consumer of the
    structured LayeredSlab specs (eps_cell / eps_tensor_cell), which were stubbed in the
    data model awaiting exactly this backend. `shapes` slabs raise until the analytic-shape
    payload contract is pinned (documented follow-on)."""

    def __init__(self, *, n_orders: int = 11, n_orders_y: Optional[int] = None,
                 stabilize: bool = False):
        self.n_orders = int(n_orders)
        self.n_orders_y = n_orders_y
        self.stabilize = bool(stabilize)

    def solve(self, stack: LayeredStack, lambda_m: float, optical) -> OpticalResult:
        lum = _require_lumenairy()
        t0 = time.perf_counter()
        _guard_incidence_side(optical)
        _guard_conical_ppol(optical, _angles_rad(optical)[1])
        structured = not stack.is_unstructured
        # audit C5-8: the lambda-sized fallback period is only harmless when the stack is
        # laterally UNIFORM (0-order-only physics); a STRUCTURED stack solved at a fabricated
        # period gets wrong diffraction geometry silently (probe: R=0.061 vs 0.191 correct)
        if structured and (float(stack.period_x_m or 0.0) <= 0.0
                           or float(stack.period_y_m or 0.0) <= 0.0):
            raise ValueError(
                "LumenairyStackSolver: the stack holds structured (eps_cell/eps_tensor_cell) "
                "slabs but period_x_m/period_y_m are unset (LayeredStack defaults 0.0; the "
                "slicers do not populate them) -- set the real lattice periods on the "
                "LayeredStack (audit C5-8; the old code silently substituted px=py=lambda).")
        px = stack.period_x_m or float(lambda_m)          # period is irrelevant when uniform
        py = stack.period_y_m or float(lambda_m)
        if structured:
            rs = lum.RCWAStack(px, period_y=py, n_superstrate=complex(stack.n_super),
                               n_substrate=complex(stack.n_sub), n_orders=self.n_orders,
                               n_orders_y=int(self.n_orders_y if self.n_orders_y is not None
                                              else self.n_orders))
        else:
            rs = lum.RCWAStack(px, n_superstrate=complex(stack.n_super),
                               n_substrate=complex(stack.n_sub), n_orders=self.n_orders)
        s_min_x = _min_samples(rs.n_orders_x)
        s_min_y = _min_samples(rs.n_orders_y) if structured else 1
        for slab in stack.slabs:                          # already superstrate-side first
            if slab.eps is not None:
                rs.add_layer(slab.thickness_m, eps=complex(slab.eps))
            elif slab.eps_cell is not None:
                rs.add_layer(slab.thickness_m,
                             eps_cell=_meet_sampling(slab.eps_cell, s_min_x, s_min_y))
            elif slab.eps_tensor_cell is not None:
                rs.add_layer(slab.thickness_m,
                             eps_tensor_cell=_meet_sampling(slab.eps_tensor_cell,
                                                            s_min_x, s_min_y))
            else:
                raise NotImplementedError(
                    "LumenairyStackSolver: analytic-shape slabs are not wired yet (the "
                    "shapes payload contract is a documented follow-on); rasterize to "
                    "eps_cell instead")
        theta, phi = _angles_rad(optical)
        rs.set_source(lambda_m, theta=theta, phi=phi)
        res = rs.solve(stabilize=self.stabilize)
        rf, tf = _p_basis_conversion(getattr(optical, "polarization", "y"), theta,
                                     stack.n_super, stack.n_sub)
        return rcwa_result_to_optical_result(res, _pol_row(optical), t0=t0,
                                             r_factor=rf, t_factor=tf)
