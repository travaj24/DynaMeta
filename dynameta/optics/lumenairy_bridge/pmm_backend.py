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

Cross-library pins (first pinned at lumenairy 5.14.2, re-verified on the 5.21 floor): PMMStack(period, n_substrate=, n_superstrate=
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
from dynameta.optics.lumenairy_bridge._common import (angles_rad as _angles_rad,
                                                      guard_incidence_side as _guard_incidence_side,
                                                      p_basis_conversion as _p_basis_conversion,
                                                      pol_row as _pol_row,
                                                      require_lumenairy as _require_lumenairy)
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices

__all__ = ["design_to_pmm_stack", "make_lumenairy_pmm_solver", "layer_to_pmm_segments"]

_REL_TOL = 1e-9
_PROP_TOL = 1e-8            # propagating-order selector (|Im kz| small, Re kz > tol), normalized by k0


def _pol_tangential_unit(pol: str, phi: float) -> np.ndarray:
    """Incident tangential (lab x, y) unit vector of the rotated s/p eigen-polarization the FEM
    solver and OpticalSpec use: 'y' = s-hat = (-sin phi, cos phi) (perpendicular to the plane of
    incidence); 'x'/'p' = the in-plane transverse direction (cos phi, sin phi). At phi = 0 these
    reduce to lab y and lab x, so the synthesis is continuous with the in-plane fast path."""
    if pol == "y":
        return np.array([-np.sin(phi), np.cos(phi)], dtype=float)
    return np.array([np.cos(phi), np.sin(phi)], dtype=float)          # 'x' or 'p'


def _conical_synthesis(stack, pol: str, theta: float, phi: float, n_super: complex):
    """Rotated s/p (R, T, r) for a conical (azimuth != 0) PMM solve, synthesized from the
    per-order complex amplitudes (audit 8.1-1 / consumer-gap B). lumenairy's per-order rows are
    keyed to incident LAB E_x / E_y; at phi != 0 the physical s/p eigen-polarization is a
    superposition of them, so the total efficiency has cross terms the per-order POWERS cannot
    provide. For incident tangential unit u the per-order response is u . (row0, row1); the
    order efficiency is (|Ex|^2 + |Ey|^2 + |Ez|^2) Re(kz/kz_inc) / |E_inc|^2 with the longitudinal
    Ez = -(kx Ex + ky Ey)/kz and |E_inc|^2 = 1 + |Ez_inc|^2 (a unit-tangential p wave carries its
    own Ez, so |E_inc,p|^2 = sec^2 theta -- the normalization the s vs p split hinges on). Summed
    over PROPAGATING orders in the (lossless) end medium: R (reflection port) and T (transmission
    port, kz_inc still the superstrate value). r = the zeroth-order co-polarized complex amplitude
    (magnitude sqrt of the zeroth-order co-pol reflectance, phase from the tangential projection
    onto the reflected eigen-direction) -- the phase-bearing modulator observable. All k's are
    normalized by k0 (per_order_amplitudes convention), so the arithmetic is scale-free."""
    u = _pol_tangential_unit(pol, phi)
    ns = float(np.real(n_super))
    kz_inc = ns * np.cos(theta)                                       # normalized incident kz
    # incident longitudinal for THIS polarization (transversality of the incident plane wave):
    # kx0 = ns sin th cos phi, ky0 = ns sin th sin phi (normalized)
    kx0 = ns * np.sin(theta) * np.cos(phi)
    ky0 = ns * np.sin(theta) * np.sin(phi)
    ez_inc = -(kx0 * u[0] + ky0 * u[1]) / kz_inc
    e_inc2 = float(u[0] ** 2 + u[1] ** 2 + abs(ez_inc) ** 2)          # 1 (s) or sec^2 theta (p)

    def _port_total(port):
        a = stack.per_order_amplitudes(port=port)
        kx, ky, kz = np.asarray(a["kx"]), np.asarray(a["ky"]), np.asarray(a["kz"])
        ex = u[0] * a["Ex"][0] + u[1] * a["Ex"][1]                   # response to u, per order
        ey = u[0] * a["Ey"][0] + u[1] * a["Ey"][1]
        prop = (np.abs(kz.imag) < _PROP_TOL) & (kz.real > _PROP_TOL)
        ez = np.zeros_like(ex)
        nz = np.abs(kz) > 1e-300
        ez[nz] = -(kx[nz] * ex[nz] + ky[nz] * ey[nz]) / kz[nz]
        w = kz.real / kz_inc
        eff = (np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2) * w
        tot = float(np.sum(eff[prop])) / e_inc2
        return tot, a, ex, ey, prop

    R, ar, exr, eyr, _pr = _port_total("reflection")
    T, at, ext, eyt, _pt = _port_total("transmission")
    # zeroth-order co-pol complex amplitude in the rotated frame: project the zeroth order's
    # FULL 3-D field (tangential + longitudinal Ez) onto the outgoing eigen-direction e_hat and
    # normalize by sqrt(|E_inc|^2), so |r|^2 is the zeroth-order co-pol reflectance and the phase
    # is the modulator observable. s: e_hat = (-sin phi, cos phi, 0) is purely tangential (Ez drops
    # -- reduces to the machine-exact tangential projection). p: e_hat is the outgoing p-hat, which
    # carries a z-component (cos th cos phi, cos th sin phi, +/- sin th_out) -- omitting Ez there
    # undercounts |r| (probed: 0.40 vs the true 0.46 = sqrt(R)). p_hat = (k_out x s_hat)/|k_out|.
    def _co_amp(a, ex, ey):
        z0 = np.where(np.asarray(a["orders"]) == 0)[0]
        if not z0.size:
            return complex(0.0)
        i0 = int(z0[0])
        kx0, ky0, kz0 = float(a["kx"][i0].real), float(a["ky"][i0].real), a["kz"][i0]
        exz, eyz = ex[i0], ey[i0]
        ez = -(kx0 * exz + ky0 * eyz) / kz0 if abs(kz0) > 1e-300 else 0.0
        kpar = np.hypot(kx0, ky0)
        if pol == "y" or kpar < 1e-12:                    # s-hat (tangential) or normal fallback
            eh = np.array([-np.sin(phi), np.cos(phi), 0.0])
        else:                                             # p-hat = (k_out x s_hat)/|k_out|
            shat = np.array([-ky0, kx0, 0.0]) / kpar
            kout = np.array([kx0, ky0, float(kz0.real)])
            eh = np.cross(kout, shat)
            eh = eh / np.linalg.norm(eh)
        return complex((eh[0] * exz + eh[1] * eyz + eh[2] * ez) / np.sqrt(e_inc2))

    r = _co_amp(ar, exr, eyr)
    t = _co_amp(at, ext, eyt)
    return R, T, r, t


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
                              stabilize=None, absorption: bool = False):
    """Build an `optical_solver` for run_pipeline backed by Lumenairy PMM (the exact seam
    signature + solve_sweep, mirroring the RCWA bridge incl. per-wavelength end media).
    1-D lamellar devices only; in-plane incidence only (azimuth raises). OpticalResult.t
    is None (PMM exposes no transmission Jones); R/T are total order-summed efficiencies.

    absorption=True (audit 8.1-3, PMM at RCWA-bridge parity): solves with
    retain_internal=True and fills per_region_absorption keyed by DESIGN layer name (slabs
    of a graded layer sum into their layer) + A_independent from PMMStack.layer_absorption
    -- the internal z-Poynting flux difference per layer, an independent volumetric
    measurement (lumenairy's own invariant closes it against 1 - sum R - sum T).

    CONICAL incidence (azimuth != 0) is now supported (audit 8.1-1 / consumer-gap B): the
    rotated s/p totals R/T and the co-pol amplitude r are SYNTHESIZED from the per-order
    complex amplitudes (lumenairy 5.22 per_order_amplitudes), since the native result rows
    are lab-basis s/p MIXTURES at phi != 0. OpticalResult.t (the zeroth-order transmission
    Jones, phase-bearing) is filled from PMMStack.jones_transmission on BOTH the in-plane
    and conical paths -- it was None before 5.22."""

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        _guard_incidence_side(design.optical)
        theta, phi = _angles_rad(design.optical)
        pol = getattr(design.optical, "polarization", "y")
        stack, names = design_to_pmm_stack(design, lambda_m, eps_by_region=eps_by_region,
                                           degree=degree, n_orders=n_orders,
                                           n_slices=n_slices)
        stack.set_source(lambda_m, theta=theta, phi=phi)
        orders, R2, T2, jones = stack.solve(stabilize=stabilize,
                                            retain_internal=absorption)
        row = _pol_row(design.optical)
        n_sup, n_sb = end_media_indices(design, lambda_m)
        rf, tf = _p_basis_conversion(pol, theta, n_sup, n_sb)
        if abs(phi) > 1e-12:
            # rotated s/p eigen-polarization: synthesize R/T + the co-pol zeroth-order r/t from
            # the per-order amplitudes (the native lab rows R2/T2/jones are s/p mixtures at
            # phi != 0). The p-basis conversion is already baked into the eigen-projection, so
            # rf/tf are NOT re-applied here.
            R, T, r, t = _conical_synthesis(stack, pol, theta, phi, n_sup)
        else:
            R = float(np.sum(R2[row]))
            T = float(np.sum(T2[row]))
            r = complex(np.asarray(jones)[row, row]) * complex(rf)
            t = complex(np.asarray(stack.jones_transmission())[row, row]) * complex(tf)
        a_ind, pra = None, None
        if absorption:
            try:
                la = np.asarray(stack.layer_absorption())     # (n_layers, 2) per incident pol
                la = la[:, row] if la.ndim == 2 and la.shape[1] == 2 else la[row]
                pra = {}
                for name, val in zip(names, la):
                    pra[name] = pra.get(name, 0.0) + float(val)
                a_ind = float(np.sum(la))
            except Exception as exc:                          # pragma: no cover - defensive
                import warnings
                warnings.warn("lumenairy PMM bridge: per-layer absorption unavailable ({}); "
                              "A_independent left unset".format(exc), stacklevel=2)
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                             solve_time_s=time.perf_counter() - t0, t=t, T=T,
                             A=1.0 - R - T, A_independent=a_ind, R_flux=R, T_flux=T,
                             per_region_absorption=pra)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        return [_solve_at(design, assemble_at(lam), lam) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    return _solve
