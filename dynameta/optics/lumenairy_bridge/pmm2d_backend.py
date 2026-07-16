"""Lumenairy 2-D crossed-patterned PMM as a DynaMeta optical backend (audit 8.1-4).

Two engines behind one seam (``engine=``):
- ``'pure'`` (PMM2DStackPure, the default): the NO-FLOOR staggered modified-Legendre
  PMM -- every region is solved in one shared modal basis, interfaces are square modal
  matches, and the Rayleigh projection happens ONCE at the far field, so accuracy is
  ``n_orders``-INDEPENDENT and tracks only the modal degree ``n_modes``.  This is the
  2-D convergence REFEREE: on a metallic patch its wall-resolved basis pins the value
  RCWA converges toward as n_orders grows (gated in validation/lumenairy_pmm2d_bridge
  GATE B, the referee pattern of the 1-D PMM bridge).  Its price is the UNION-GRID
  constraint: all patterned layers share ONE square uniform (N, N) segmentation, so
  the bridge translates ANALYTIC Rectangle inclusions onto the smallest commensurate
  grid (layer_to_pure_cell) and refuses geometry that does not land on one.
- ``'hybrid'`` (PMM2DStackHybrid): the Fourier-projected 2-D PMM (per-layer exact
  spectral-element walls, no union-grid constraint, but the FMM ``n_orders`` floor).
  It accepts RASTERIZED cells (the shared optics.rasterize painter, byte-identical
  geometry to the RCWA bridge).
BOTH engines expose per-layer absorption (``layer_absorption``, wired via
``absorption=True`` at 1-D-PMM/RCWA parity -- the pure engine gained internal
retention in lumenairy 5.22, audit C3; before that only the hybrid engine had it).

Scope contract (v1, enforced loudly -- the honest-scope convention):
- 'pure': patterned layers must be axis-aligned Rectangle inclusions whose walls are
  COMMENSURATE with a uniform (N, N) grid, N <= pure_max_cells (raise -> use
  engine='hybrid' or the RCWA bridge); laterally-structured gridded EpsFields raise
  (rasters have no exact walls).  Per-layer absorption IS supported (lumenairy 5.22
  added internal retention to PMM2DStackPure -- absorption=True at hybrid parity).
- 'hybrid': patterned layers rasterize onto a (cell_samples, cell_samples) pixel grid
  (walls sit on pixel boundaries -- lumenairy's own cost guard rejects
  pathologically-many-wall rasters with its refine/route-elsewhere message);
  laterally-structured gridded EpsFields pass through as per-slab eps_cell.
- Both: tensor EpsFields raise (the hybrid CAN carry eps_tensor_cell but the tensor
  path is unvalidated through this bridge -- use the RCWA bridge (in-plane) or the
  Berreman / 1-D PMM bridges (OOP)); in-plane incidence only (conical azimuth raises,
  audit C4-2); TOP incidence only; no transmission Jones (OpticalResult.t is None,
  the 1-D PMM precedent -- T comes from the order-summed efficiencies).

Cross-library pins (verified against the installed 5.21 source, stack2d.py /
stack2d_pure.py): PMM2DStackPure(period_x, period_y, n_superstrate=, n_substrate=
[INDICES], n_modes=, n_orders=); add_layer(thickness, eps=|eps_cell=) where eps_cell
is a SQUARE (N, N) per-SEGMENT grid on uniform walls i*period/N (Granet staggered
basis, Basis1D xb = linspace); PMM2DStackHybrid(period_x, period_y, n_superstrate=,
n_substrate=, degree= [ODD], n_orders=, formulation=); add_layer(thickness,
eps=|eps_cell=|eps_tensor_cell=) where eps_cell is a PIXEL grid (walls extracted at
value changes).  Both: set_source(wavelength, theta=, phi=) [m/rad]; solve() ->
(orders(N, 2), R(2, N), T(2, N), jones_reflection(2, 2)) with rows keyed INCIDENT
E_x (0) / E_y (1) in the PUBLIC gauge -- the rcwa_jones_2d layout, so the pol-row
mapping and the lab-basis -> Byrnes p-pol r conversion are IDENTICAL to the RCWA
bridge (gated vs TMM in GATE A).  Dispersive callables force solve_vs_wavelength;
this bridge rebuilds CONCRETE stacks per wavelength (the bridge-wide policy).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.core.layered import collapse_regions_to_layers, slice_eps_field
from dynameta.optics.lumenairy_bridge._common import (angles_rad as _angles_rad,
                                                      guard_conical_ppol as _guard_conical_ppol,
                                                      guard_incidence_side as _guard_incidence_side,
                                                      p_basis_conversion as _p_basis_conversion,
                                                      pol_row as _pol_row,
                                                      require_lumenairy as _require_lumenairy)
from dynameta.optics.rasterize import cell_axes, layer_eps_cell
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices

__all__ = ["design_to_pmm2d_stack", "make_lumenairy_pmm2d_solver", "layer_to_pure_cell",
           "pure_union_grid_n"]

_REL_TOL = 1e-9
# commensurability slack for wall fractions (|f*N - round(f*N)|): generous vs float
# noise (~1e-13) yet far below any physical wall placement
_COMM_TOL = 1e-6
_ENGINES = ("pure", "hybrid")


def _check_engine(engine: str) -> str:
    if engine not in _ENGINES:
        raise ValueError("PMM2D bridge: engine must be one of {} (got {!r})".format(
            _ENGINES, engine))
    return engine


def _pure_wall_fractions(layer, period_x_m: float,
                         period_y_m: float) -> Tuple[List[float], List[float]]:
    """Wall positions of one patterned layer as PERIOD FRACTIONS (x list, y list).
    Every inclusion must be an axis-aligned Rectangle lying inside the cell --
    the pure engine's analytic vocabulary (exact walls, Eq. 26 of Granet 2023);
    anything else raises with a pointer (the honest scope)."""
    px, py = float(period_x_m), float(period_y_m)
    fx, fy = [], []
    for inc in layer.inclusions:
        if getattr(inc.shape, "kind", "") != "rectangle":
            raise ValueError(
                "PMM2D bridge (pure): layer {!r} inclusion kind {!r} is not an "
                "axis-aligned Rectangle -- the pure staggered basis needs exact "
                "rectangular walls; use engine='hybrid' (rasterized, pixel-snapped "
                "walls) or the RCWA bridge".format(
                    layer.name, getattr(inc.shape, "kind", "?")))
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        if xlo < -_REL_TOL * px or xhi > px * (1.0 + _REL_TOL) \
                or ylo < -_REL_TOL * py or yhi > py * (1.0 + _REL_TOL):
            raise ValueError(
                "PMM2D bridge (pure): layer {!r} Rectangle bbox x [{:.3e}, {:.3e}] "
                "y [{:.3e}, {:.3e}] leaves the cell [0, {:.3e}] x [0, {:.3e}] "
                "(wrap-around is a follow-on; shift the unit-cell origin)".format(
                    layer.name, xlo, xhi, ylo, yhi, px, py))
        fx += [max(xlo, 0.0) / px, min(xhi, px) / px]
        fy += [max(ylo, 0.0) / py, min(yhi, py) / py]
    return fx, fy


def pure_union_grid_n(design, *, max_cells: int = 32) -> Optional[int]:
    """The smallest uniform square grid size N such that EVERY patterned layer's
    Rectangle walls land on the segment boundaries i/N of BOTH axes -- the pure
    engine's union-grid (all patterned layers share ONE (N, N) segmentation, and
    lumenairy's staggered Basis1D uses UNIFORM segments only).  None when the
    design has no patterned layer; raises when no N <= max_cells fits (each extra
    segment costs modal DOF ~ (N*(n_modes-1))^2, so a large commensurate N is a
    cost problem, not just a validity one)."""
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    fracs: List[float] = []
    for L in design.stack.layers:
        if L.inclusions:
            fx, fy = _pure_wall_fractions(L, px, py)
            fracs += fx + fy
    if not fracs:
        return None
    for n in range(1, int(max_cells) + 1):
        if all(abs(f * n - round(f * n)) <= _COMM_TOL for f in fracs):
            return n
    raise ValueError(
        "PMM2D bridge (pure): no uniform (N, N) segmentation with N <= {} puts "
        "every Rectangle wall on a segment boundary (wall fractions {}); the pure "
        "staggered basis needs commensurate walls.  Raise pure_max_cells (cost "
        "grows ~ N^2 in modal DOF) or use engine='hybrid' / the RCWA bridge, "
        "which have no union-grid constraint.".format(
            int(max_cells), sorted({round(f, 6) for f in fracs})))


def layer_to_pure_cell(layer, design, lambda_m: float, period_x_m: float,
                       period_y_m: float, n_cells: int, *, bg_eps=None):
    """Paint one patterned layer onto the pure engine's (n_cells, n_cells)
    per-SEGMENT eps grid (segment i spans [i, i+1]*period/N): background eps
    (eps_by_region scalar override wins, the rasterizer contract) overpainted by
    each Rectangle in ASCENDING priority via segment-CENTER membership --
    unambiguous because pure_union_grid_n guaranteed the walls sit on segment
    boundaries, never inside a segment."""
    n = int(n_cells)
    px, py = float(period_x_m), float(period_y_m)
    _pure_wall_fractions(layer, px, py)          # re-validate shape kinds/bounds
    bg = complex(bg_eps if bg_eps is not None
                 else design.materials.get(layer.background_material).eps(lambda_m))
    cx = (np.arange(n) + 0.5) * (px / n)         # segment centres, axis 0 = x
    cy = (np.arange(n) + 0.5) * (py / n)
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    cell = np.full((n, n), bg, dtype=complex)
    for inc in sorted(layer.inclusions, key=lambda i: getattr(i, "priority", 0)):
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        mask = (X >= xlo) & (X <= xhi) & (Y >= ylo) & (Y <= yhi)
        cell[mask] = complex(design.materials.get(inc.material).eps(lambda_m))
    return cell


def design_to_pmm2d_stack(design, lambda_m: float, *, engine: str = "pure",
                          eps_by_region=None, n_modes: int = 8, degree: int = 11,
                          n_orders: Optional[int] = None, formulation: str = "li",
                          n_slices: Optional[int] = None,
                          cell_samples: Optional[int] = None,
                          pure_max_cells: int = 32):
    """Translate a DynaMeta Design into a CONCRETE Lumenairy PMM2DStackPure
    ('pure') or PMM2DStackHybrid ('hybrid') at one wavelength.  Returns
    (stack, layer_names_top_first), names aligned with the Lumenairy layers
    (graded fields contribute several slabs per design layer).

    Layer dispatch (design layers are bottom -> top; both 2-D PMM stacks are
    superstrate-side first -> reversed(), audit C5-1 -- and so is every
    slice_eps_field consumer: the slicer returns ASCENDING-z substrate-first
    slabs, gated by the asymmetric-profile GATE C):
    laterally-uniform gridded EpsField -> reversed uniform slabs (both engines);
    laterally-STRUCTURED gridded EpsField -> reversed eps_cell slabs (hybrid) /
    raise (pure: rasters have no exact walls); tensor EpsField -> raise (both,
    see the module docstring); inclusions -> layer_to_pure_cell on the union
    grid (pure) / the shared rasterizer (hybrid, byte-identical geometry to the
    RCWA bridge); uniform scalar -> eps."""
    lum = _require_lumenairy()
    _check_engine(engine)
    n_super, n_sub = end_media_indices(design, lambda_m)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    # mesh-region-keyed dicts (the run_pipeline/FEM bridge output) collapse to
    # design-layer keys; identity when already layer-keyed
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})
    if engine == "pure":
        stack = lum.PMM2DStackPure(px, py, n_superstrate=complex(n_super),
                                   n_substrate=complex(n_sub), n_modes=int(n_modes),
                                   **({} if n_orders is None
                                      else dict(n_orders=int(n_orders))))
        n_pure = pure_union_grid_n(design, max_cells=pure_max_cells)
    else:
        stack = lum.PMM2DStackHybrid(px, py, n_superstrate=complex(n_super),
                                     n_substrate=complex(n_sub), degree=int(degree),
                                     formulation=formulation,
                                     **({} if n_orders is None
                                        else dict(n_orders=int(n_orders))))
        n_pure = None
    sx = sy = int(cell_samples or 128)           # hybrid raster (pixel walls)

    names: List[str] = []
    for L in reversed(design.stack.layers):      # superstrate side first
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            if getattr(ef, "is_tensor", False):
                raise ValueError(
                    "PMM2D bridge: layer {!r} carries a gridded TENSOR EpsField; "
                    "the 2-D PMM bridge is scalar-only (v1) -- use the RCWA "
                    "bridge (eps_tensor_cell)".format(L.name))
            v = ef.values_zyx
            lat_uniform = bool(np.allclose(v, v[:, :1, :1], rtol=1e-12, atol=0.0))
            if not lat_uniform and engine == "pure":
                raise ValueError(
                    "PMM2D bridge (pure): layer {!r} carries a laterally-"
                    "structured gridded EpsField; the pure staggered basis "
                    "needs exact analytic walls, not a raster -- use "
                    "engine='hybrid' or the RCWA bridge".format(L.name))
            # slice_eps_field returns ascending-z (substrate-first) slabs; this
            # stack is superstrate-first, so reverse (audit C5-1)
            for slab in reversed(slice_eps_field(ef, 1.0 / _S_NM, n_slices=n_slices)):
                if slab.eps is not None:
                    stack.add_layer(slab.thickness_m, eps=complex(slab.eps))
                elif slab.eps_cell is not None:  # hybrid only (pure raised above)
                    stack.add_layer(slab.thickness_m,
                                    eps_cell=np.asarray(slab.eps_cell, dtype=complex))
                else:
                    raise ValueError(
                        "PMM2D bridge: layer {!r} sliced to a TENSOR slab; the "
                        "2-D PMM bridge is scalar-only (v1) -- use the RCWA "
                        "bridge".format(L.name))
                names.append(L.name)
            continue
        if ef is not None and getattr(ef, "is_tensor", False):
            raise ValueError(
                "PMM2D bridge: layer {!r} carries a uniform TENSOR EpsField; the "
                "tensor path is unvalidated through this bridge (v1; the hybrid "
                "engine is eps_tensor_cell-capable, a documented follow-on) -- "
                "use the RCWA bridge (in-plane) or the Berreman / 1-D PMM "
                "bridges (OOP)".format(L.name))
        if L.inclusions:
            if engine == "pure":
                cell = layer_to_pure_cell(L, design, lambda_m, px, py, n_pure,
                                          bg_eps=(ef.scalar if ef is not None
                                                  else None))
                stack.add_layer(L.thickness_m, eps_cell=cell)
            else:
                x, y = cell_axes(sx, sy, px, py)
                X, Y = np.meshgrid(x, y, indexing="ij")
                cell = layer_eps_cell(L, X, Y, lambda_m, design.materials,
                                      eps_by_region)
                stack.add_layer(L.thickness_m,
                                eps_cell=np.asarray(cell, dtype=complex))
            names.append(L.name)
            continue
        if ef is not None:                       # uniform scalar (effect-modulated)
            stack.add_layer(L.thickness_m, eps=complex(ef.scalar))
        else:
            stack.add_layer(L.thickness_m,
                            eps=complex(design.materials.get(L.background_material)
                                        .eps(lambda_m)))
        names.append(L.name)
    return stack, names


def make_lumenairy_pmm2d_solver(*, engine: str = "pure", n_modes: int = 8,
                                degree: int = 11, n_orders: Optional[int] = None,
                                formulation: str = "li",
                                n_slices: Optional[int] = None,
                                cell_samples: Optional[int] = None,
                                pure_max_cells: int = 32,
                                absorption: bool = False):
    """Build an `optical_solver` for run_pipeline backed by the Lumenairy 2-D
    crossed-patterned PMM (the exact seam signature + solve_sweep, mirroring the
    RCWA/1-D-PMM bridges incl. per-wavelength end media).  engine='pure' is the
    no-Fourier-floor referee (converge via n_modes; n_orders only needs to cover
    the propagating orders); engine='hybrid' is the Fourier-projected stack
    (converge via degree [ODD] + n_orders).  In-plane incidence only; TOP
    incidence only; OpticalResult.t is None (neither engine exposes a
    transmission Jones); R/T are total order-summed efficiencies; r is the
    zeroth-order co-polarized reflection Jones with the shared lab-basis ->
    Byrnes p-pol conversion (all gated in validation/lumenairy_pmm2d_bridge).

    absorption=True (BOTH engines): solves with retain_internal=True and fills
    per_region_absorption keyed by DESIGN layer name (slabs of a graded layer sum
    into their layer) + A_independent from PMM2DStack{Hybrid,Pure}.layer_absorption
    -- the internal z-flux difference per layer ((n_layers, 2) per incident pol),
    an independent volumetric measurement (lumenairy's own invariant closes it
    against 1 - sum R - sum T).  The pure engine gained internal retention in
    lumenairy 5.22 (audit C3); before that absorption raised for engine='pure'."""
    _check_engine(engine)

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        _guard_incidence_side(design.optical)
        theta, phi = _angles_rad(design.optical)
        _guard_conical_ppol(design.optical, phi)
        stack, names = design_to_pmm2d_stack(
            design, lambda_m, engine=engine, eps_by_region=eps_by_region,
            n_modes=n_modes, degree=degree, n_orders=n_orders,
            formulation=formulation, n_slices=n_slices, cell_samples=cell_samples,
            pure_max_cells=pure_max_cells)
        stack.set_source(lambda_m, theta=theta, phi=phi)
        # both PMM2DStackHybrid.solve and PMM2DStackPure.solve take the same
        # keyword surface (retain_internal retains the internals layer_absorption
        # integrates -- pure since lumenairy 5.22, audit C3)
        orders, R2, T2, jones = stack.solve(retain_internal=absorption)
        row = _pol_row(design.optical)
        R = float(np.sum(R2[row]))
        T = float(np.sum(T2[row]))
        n_sup, n_sb = end_media_indices(design, lambda_m)
        rf, _tf = _p_basis_conversion(getattr(design.optical, "polarization", "y"),
                                      theta, n_sup, n_sb)
        r = complex(np.asarray(jones)[row, row]) * complex(rf)
        a_ind, pra = None, None
        if absorption:
            try:
                la = np.asarray(stack.layer_absorption())[:, row]  # (n_layers, 2)
                pra = {}
                for name, val in zip(names, la):
                    pra[name] = pra.get(name, 0.0) + float(val)
                a_ind = float(np.sum(la))
            except Exception as exc:              # pragma: no cover - defensive
                import warnings
                warnings.warn("lumenairy PMM2D bridge: per-layer absorption "
                              "unavailable ({}); A_independent left unset".format(exc),
                              stacklevel=2)
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                             solve_time_s=time.perf_counter() - t0, t=None, T=T,
                             A=1.0 - R - T, A_independent=a_ind, R_flux=R, T_flux=T,
                             per_region_absorption=pra)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        return [_solve_at(design, assemble_at(lam), lam) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    return _solve
