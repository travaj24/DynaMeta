"""Lumenairy PMM as a DynaMeta optical backend (roadmap v0.5 A2).

PMM's roles in DynaMeta (complementing the RCWA backend, not replacing it):
- CONVERGENCE REFEREE: the 1-D PMM family is a subsectional spectral element with NO
  Fourier-factorization accuracy floor -- the referee for RCWA truncation settings on hard
  (metallic TM) 1-D cells. (The no-floor claim is 1-D-only; Lumenairy's 2-D PMM is itself
  Fourier-floored, so this bridge is deliberately 1-D.)
- TENSOR SPECIALIST: pmm_jones_1d / PMMStack carry FULL (3,3) tensors including
  out-of-plane (exz/eyz/ezx/ezy) couplings -- gyrotropic / slanted-LC cells that the RCWA
  in-plane tensor path cannot represent.

Scope contract (v1, enforced loudly): 1-D devices only -- every patterned layer's
inclusions must be Rectangles spanning the FULL y period (a lamellar grating), translated
into PMM segments [(width_fraction, eps)]. Rasterized (gridded structured) EpsFields raise
(PMM segments are analytic; use the RCWA bridge for grids). In-plane incidence only
(azimuth = 0; PMMStack has no conical source). PMM returns NO transmission Jones and no
per-order complex amplitudes: OpticalResult.t is None (T comes from the order-summed
efficiencies; r from the zeroth-order reflection Jones with the same lab-basis -> Byrnes
p-pol conversion as the RCWA bridge -- gated in validation/lumenairy_pmm_bridge.py).

Cross-library pins (lumenairy 5.14.2): PMMStack(period, n_substrate=, n_superstrate=
[INDICES], degree=, n_orders=); add_layer(thickness, segments=[(w_frac, EPS)] | eps=
[scalar or (3,3)], slant_angle=); set_source(wavelength, theta=); solve() ->
(orders, R_eff(2, M), T_eff(2, M), jones_reflection(2, 2)) with rows/columns keyed
INCIDENT E_x (0) / E_y (1) -- identical to the RCWA result layout. Dispersive callables
force solve_vs_wavelength; this bridge rebuilds CONCRETE stacks per wavelength instead
(same policy as the RCWA bridge)."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.core.layered import collapse_regions_to_layers, slice_eps_field
from dynameta.optics.lumenairy_bridge.rcwa_backend import (_angles_rad, _guard_incidence_side,
                                                           _p_basis_conversion, _pol_row,
                                                           _require_lumenairy)
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices

__all__ = ["design_to_pmm_stack", "make_lumenairy_pmm_solver", "layer_to_pmm_segments"]

_REL_TOL = 1e-9


def layer_to_pmm_segments(layer, design, lambda_m: float, period_x_m: float,
                          period_y_m: float, *, bg_eps=None) -> List[Tuple[float, complex]]:
    """Translate a lamellar (1-D) DynaMeta layer into PMM segments [(width_fraction, eps)].
    Every inclusion must be a Rectangle spanning the FULL y period (y-invariant grating
    lines), lying inside [0, period_x] and mutually non-overlapping in x; gaps are filled
    with the background eps. Anything else raises -- the honest 1-D scope."""
    px, py = float(period_x_m), float(period_y_m)
    bg = complex(bg_eps if bg_eps is not None
                 else design.materials.get(layer.background_material).eps(lambda_m))
    spans = []
    for inc in layer.inclusions:
        if getattr(inc.shape, "kind", "") != "rectangle":
            raise ValueError("PMM bridge: layer {!r} inclusion kind {!r} is not a lamellar "
                             "Rectangle (1-D scope); use the RCWA bridge".format(
                                 layer.name, getattr(inc.shape, "kind", "?")))
        xlo, xhi, ylo, yhi = inc.shape.bbox_m()
        if ylo > _REL_TOL * py or yhi < py * (1.0 - _REL_TOL):
            raise ValueError("PMM bridge: layer {!r} Rectangle does not span the full y "
                             "period (y-invariant grating lines required; got y in "
                             "[{:.3e}, {:.3e}] of {:.3e})".format(layer.name, ylo, yhi, py))
        if xlo < -_REL_TOL * px or xhi > px * (1.0 + _REL_TOL):
            raise ValueError("PMM bridge: layer {!r} Rectangle x-extent [{:.3e}, {:.3e}] "
                             "leaves the cell [0, {:.3e}] (wrap-around is a follow-on)"
                             .format(layer.name, xlo, xhi, px))
        spans.append((max(xlo, 0.0), min(xhi, px),
                      complex(design.materials.get(inc.material).eps(lambda_m))))
    spans.sort(key=lambda s: s[0])
    for (a_lo, a_hi, _), (b_lo, b_hi, _) in zip(spans, spans[1:]):
        if b_lo < a_hi - _REL_TOL * px:
            raise ValueError("PMM bridge: layer {!r} Rectangles overlap in x (segments "
                             "must partition the period)".format(layer.name))
    segs, x = [], 0.0
    for lo, hi, eps in spans:
        if lo > x + _REL_TOL * px:
            segs.append(((lo - x) / px, bg))
        segs.append(((hi - lo) / px, eps))
        x = hi
    if x < px * (1.0 - _REL_TOL):
        segs.append(((px - x) / px, bg))
    total = sum(w for w, _ in segs)
    return [(w / total, e) for w, e in segs]              # exact unit sum for PMM's check


def design_to_pmm_stack(design, lambda_m: float, *, eps_by_region=None,
                        degree: int = 16, n_orders: int = 21,
                        n_slices: Optional[int] = None):
    """Translate a 1-D-compatible DynaMeta Design into a Lumenairy PMMStack at one
    wavelength. Returns (stack, layer_names_top_first). Same layer dispatch as the RCWA
    bridge except: structured GRIDDED EpsFields raise (analytic segments only) and
    patterned layers go through layer_to_pmm_segments."""
    lum = _require_lumenairy()
    n_super, n_sub = end_media_indices(design, lambda_m)
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    # mesh-region-keyed dicts (the run_pipeline/FEM bridge output) collapse to design-layer
    # keys; identity when already layer-keyed
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})
    stack = lum.PMMStack(px, n_substrate=complex(n_sub), n_superstrate=complex(n_super),
                         degree=int(degree), n_orders=int(n_orders))
    names: List[str] = []
    for L in reversed(design.stack.layers):              # superstrate side first
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            v = ef.values_zyx
            if getattr(ef, "is_tensor", False) or not np.allclose(
                    v, v[:, :1, :1], rtol=1e-12, atol=0.0):
                raise ValueError("PMM bridge: layer {!r} carries a laterally-structured "
                                 "gridded EpsField; PMM segments are analytic -- use the "
                                 "RCWA bridge for rasterized cells".format(L.name))
            # slice_eps_field returns ascending-z (substrate-first) slabs; this stack
            # is superstrate-first, so reverse (audit C5-1)
            for slab in reversed(slice_eps_field(ef, 1.0 / _S_NM, n_slices=n_slices)):
                stack.add_layer(slab.thickness_m, eps=complex(slab.eps))
                names.append(L.name)
            continue
        if ef is not None and getattr(ef, "is_tensor", False):
            stack.add_layer(L.thickness_m, eps=np.asarray(ef.tensor, dtype=complex))
            names.append(L.name)
            continue
        if L.inclusions:
            stack.add_layer(L.thickness_m,
                            segments=layer_to_pmm_segments(
                                L, design, lambda_m, px, py,
                                bg_eps=(ef.scalar if ef is not None else None)))
            names.append(L.name)
            continue
        if ef is not None:
            stack.add_layer(L.thickness_m, eps=complex(ef.scalar))
        else:
            stack.add_layer(L.thickness_m,
                            eps=complex(design.materials.get(L.background_material)
                                        .eps(lambda_m)))
        names.append(L.name)
    return stack, names


def make_lumenairy_pmm_solver(*, degree: int = 16, n_orders: int = 21,
                              n_slices: Optional[int] = None,
                              stabilize=None):
    """Build an `optical_solver` for run_pipeline backed by Lumenairy PMM (the exact seam
    signature + solve_sweep, mirroring the RCWA bridge incl. per-wavelength end media).
    1-D lamellar devices only; in-plane incidence only (azimuth raises). OpticalResult.t
    is None (PMM exposes no transmission Jones); R/T are total order-summed efficiencies."""

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        _guard_incidence_side(design.optical)
        theta, phi = _angles_rad(design.optical)
        if abs(phi) > 1e-12:
            raise NotImplementedError("PMM bridge: conical incidence (azimuth != 0) is not "
                                      "supported by PMMStack; use the RCWA or FEM solver")
        stack, names = design_to_pmm_stack(design, lambda_m, eps_by_region=eps_by_region,
                                           degree=degree, n_orders=n_orders,
                                           n_slices=n_slices)
        stack.set_source(lambda_m, theta=theta)
        orders, R2, T2, jones = stack.solve(stabilize=stabilize)
        row = _pol_row(design.optical)
        R = float(np.sum(R2[row]))
        T = float(np.sum(T2[row]))
        n_sup, n_sb = end_media_indices(design, lambda_m)
        rf, _tf = _p_basis_conversion(getattr(design.optical, "polarization", "y"),
                                      theta, n_sup, n_sb)
        r = complex(np.asarray(jones)[row, row]) * complex(rf)
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                             solve_time_s=time.perf_counter() - t0, t=None, T=T,
                             A=1.0 - R - T, R_flux=R, T_flux=T)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        return [_solve_at(design, assemble_at(lam), lam) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    return _solve
