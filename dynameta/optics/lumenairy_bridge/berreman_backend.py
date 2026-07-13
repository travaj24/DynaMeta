"""Lumenairy Berreman 4x4 as a DynaMeta optical backend (roadmap v0.5 A4).

THE PLANAR ANISOTROPIC TIER. Berreman (1972) / Yeh (1979) solve a laterally-UNIFORM
multilayer whose layers may be fully anisotropic (uniaxial / biaxial / GYROTROPIC) by a
4x4 tangential-field cascade -- the exact generalization of the scalar transfer-matrix
coating model. This is the missing third member of the bridge's solver family:

  * RCWAStack / PMMStack : laterally PATTERNED (grating / metasurface) structures.
  * BerremanStack        : laterally UNIFORM anisotropic stacks (retarders, waveplates,
                           magneto-optic films, birefringent thermo-optic / EO slabs) --
                           microseconds per solve, ~100-1000x faster than routing the same
                           uniform tensor through RCWA's Fourier cascade or the 3-D FEM.

WHY this backend exists (the audit gap it closes): every DynaMeta UNIFORM anisotropic
device class -- LiquidCrystalModel retarders (core.effects.reconfigurable), MagnetoOptic /
VectorMagnetoOpticModel gyrotropic films (core.effects.magneto), AnisotropicThermoOpticModel
birefringent heaters (core.effects.thermo), Pockels / Kerr EO films (core.effects.electro)
-- previously solved ONLY through the 3-D vector FEM (slow, ~2% on the magneto Faraday
check, NON-differentiable) or the 1-D PMM modal eigensolve (Fourier overkill for an
unpatterned slab, in-plane only, no transmission Jones). Berreman is exact, conical-capable,
carries the full 2x2 Jones + per-layer absorption + internal field, AND is JAX-differentiable
end-to-end (the only differentiable Lumenairy solver) -- gradients through every layer eps
tensor (re+im), thickness, wavelength, angle, phi. Crucially it is the ONLY rigorous path
for the FULL z-coupled gyrotropic tensor (VectorMagnetoOpticModel) the RCWA in-plane tensor
path explicitly rejects (no exz/eyz/ezx/ezy).

BRIDGE, not vendor: lumenairy (>= 5.21 bridge floor; 5.14.4 added BerremanStack /
berreman_jones_1d) is a REQUIRED dependency but imported lazily; conventions are IDENTICAL
on both sides (public exp(-i omega t), Im(eps) > 0 for absorbers, metres, radians, RAW eps --
no conjugation), so this is geometry/result adaptation only, no sign/unit translation.

Scope contract (enforced loudly): laterally-UNIFORM stacks only. A layer with inclusions,
or a laterally-structured (patterned) eps_cell / eps_tensor_cell, RAISES with a pointer to
the RCWA/PMM backend -- this IS the dispatch boundary (Berreman is the planar tier; anything
patterned needs a transverse Fourier/nodal basis). Uniform tensor layers (a constant (3,3)
across the cell), uniform scalars, and graded laterally-uniform fields (z-sliced into uniform
slabs) are the supported inputs.

Cross-library pins (first pinned at lumenairy 5.14.4/5.14.5, re-verified on the 5.21 floor; file:line in elements/berreman.py):
- berreman_jones_1d(layers, n_substrate, n_superstrate, wavelength, *, angle=, phi=, theta=)
  -> (R, T, jones_r, jones_t). `layers` = [(eps, thickness_m), ...] SUPERSTRATE-side first
  (== LayeredStack.slabs order); eps scalar or (3,3) (public Im>0). R/T are (2,) TOTAL
  (co+cross) flux-normalized power per incident lab pol (E_x row 0 / E_y row 1); jones_r/t are
  the (2,2) lab-basis Jones (columns = incident Ex/Ey, rows = reflected/transmitted Ex/Ey) --
  the SAME layout as RCWAStack.jones_reflection (the docstring pins Berreman == rcwa_jones_1d
  to machine precision on a uniform-tensor slab), so the RCWA bridge's _pol_row / _p_basis_
  conversion / _angles_rad apply UNCHANGED.
- BerremanStack(n_substrate=, n_superstrate=).add_layer(thickness, eps=).set_source(wavelength,
  theta=, phi=).solve(retain_internal=) -> (R, T, jones_r) [NOTE: the class solve returns only
  jones_r, so this backend uses the FUNCTIONAL berreman_jones_1d for the far field to also get
  jones_t, and the class only when per-layer absorption / internal field is requested].
"""

from __future__ import annotations

import time
import warnings
from typing import List, Optional, Tuple

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.core.layered import (LayeredStack, collapse_regions_to_layers,
                                   slice_eps_field)
from dynameta.optics.lumenairy_bridge._common import (angles_rad as _angles_rad,
                                                      guard_conical_ppol as _guard_conical_ppol,
                                                      guard_incidence_side as _guard_incidence_side,
                                                      p_basis_conversion as _p_basis_conversion,
                                                      pol_row as _pol_row,
                                                      require_lumenairy as _require_lumenairy)
from dynameta.optics.tmm_reference import S as _S_NM
from dynameta.optics.tmm_reference import end_media_indices

__all__ = ["design_to_berreman_layers", "make_lumenairy_berreman_solver",
           "berreman_result_to_optical_result", "BerremanLayeredSolver"]


def _require_berreman():
    """lumenairy with the Berreman 4x4 surface. The single bridge floor (_common.VERSION_FLOOR,
    >= 5.21) already covers the 5.14.4 tier that added it; the symbol check stays as a cheap
    defensive guard against a partial/renamed install."""
    lum = _require_lumenairy()
    if not hasattr(lum, "berreman_jones_1d"):
        raise ImportError(
            "the Berreman backend needs lumenairy's BerremanStack / berreman_jones_1d surface, "
            "absent from this install ({}). pip install -U lumenairy".format(lum.__version__))
    return lum


def _uniform_tensor(cell, *, where: str):
    """A LAYERED (Sx, Sy, 3, 3) eps_tensor_cell that is constant across the cell -> its single
    (3, 3) tensor. Berreman is the PLANAR tier, so a tensor cell that actually varies in x/y is
    a patterned anisotropic layer and RAISES with a pointer to the RCWA backend (the dispatch
    boundary). A (1, 1, 3, 3) cell (the uniform output of slice_eps_field) passes trivially."""
    c = np.asarray(cell, dtype=complex)
    if c.ndim != 4 or c.shape[-2:] != (3, 3):
        raise ValueError("Berreman backend: {} eps_tensor_cell must be (Sx, Sy, 3, 3); got "
                         "shape {}.".format(where, c.shape))
    if not np.allclose(c, c[:1, :1], rtol=1e-12, atol=0.0):
        raise NotImplementedError(
            "Berreman backend: {} carries a LATERALLY-STRUCTURED (patterned) anisotropic "
            "eps_tensor_cell; Berreman is the PLANAR (laterally-uniform) tier -- use the RCWA "
            "backend (make_lumenairy_rcwa_solver) for a patterned anisotropic cell.".format(where))
    return c[0, 0]


def design_to_berreman_layers(design, lambda_m: float, *, eps_by_region=None,
                              n_slices: Optional[int] = None):
    """Translate a laterally-UNIFORM DynaMeta Design (+ optionally the bridge's eps_by_region)
    into Berreman `layers` at one wavelength. Returns (layers, n_super, n_sub, layer_names) with
    `layers` = [(eps, thickness_m), ...] ordered SUPERSTRATE-side first (the berreman_jones_1d
    contract) and `layer_names` aligned 1:1 (a graded layer contributes several uniform slabs,
    so names repeat) for mapping per-layer absorption back to design layers.

    Layer dispatch (same precedence as the RCWA bridge, but PATTERNED layers RAISE -- the planar
    scope boundary): a graded laterally-uniform EpsField -> z-sliced uniform slabs (scalar or a
    constant (3,3) per slab); a uniform tensor EpsField -> its (3,3); a uniform scalar EpsField
    -> its scalar; lateral inclusions or a structured grid -> NotImplementedError (use RCWA/PMM);
    else the background material eps at lambda."""
    _require_berreman()
    n_super, n_sub = end_media_indices(design, lambda_m)
    eps_by_region = collapse_regions_to_layers(design, eps_by_region or {})
    layers: List[Tuple[object, float]] = []
    names: List[str] = []
    for L in reversed(design.stack.layers):              # superstrate side first
        ef = eps_by_region.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            # slice_eps_field returns ascending-z (substrate-first) slabs; this list
            # is superstrate-first, so reverse (audit C5-1)
            for slab in reversed(slice_eps_field(ef, 1.0 / _S_NM, n_slices=n_slices)):
                if slab.eps is not None:
                    layers.append((complex(slab.eps), float(slab.thickness_m)))
                elif slab.eps_tensor_cell is not None:
                    layers.append((_uniform_tensor(slab.eps_tensor_cell,
                                                   where="layer {!r} slab".format(L.name)),
                                   float(slab.thickness_m)))
                else:                                    # patterned scalar grid (eps_cell)
                    raise NotImplementedError(
                        "Berreman backend: layer {!r} is a laterally-structured (patterned) "
                        "grid; Berreman is the PLANAR tier -- use the RCWA backend.".format(L.name))
                names.append(L.name)
            continue
        if ef is not None and getattr(ef, "is_tensor", False):
            layers.append((np.asarray(ef.tensor, dtype=complex), float(L.thickness_m)))
            names.append(L.name)
            continue
        if L.inclusions:
            raise NotImplementedError(
                "Berreman backend: layer {!r} has lateral inclusions (a patterned metasurface); "
                "Berreman is the PLANAR (laterally-uniform) tier -- use the RCWA or PMM "
                "backend.".format(L.name))
        if ef is not None:                               # uniform scalar (effect-modulated)
            layers.append((complex(ef.scalar), float(L.thickness_m)))
        else:
            layers.append((complex(design.materials.get(L.background_material).eps(lambda_m)),
                           float(L.thickness_m)))
        names.append(L.name)
    return layers, n_super, n_sub, names


def berreman_result_to_optical_result(R_arr, T_arr, jones_r, jones_t, row: int, *, t0: float,
                                      r_factor: complex = 1.0, t_factor: complex = 1.0,
                                      layer_absorption=None,
                                      layer_names: Optional[List[str]] = None) -> OpticalResult:
    """Map a berreman_jones_1d far field to a DynaMeta OpticalResult for ONE incident pol row.
    R/T = TOTAL (co+cross) flux-normalized power (== R_flux/T_flux); r/t = co-polarized complex
    Jones (Jr[row,row]/Jt[row,row]) times the p-basis conversion factors when applicable; A =
    1 - R - T. With layer_absorption (n_layers, 2) + layer_names, per_region_absorption is filled
    keyed by design layer (graded slabs summed) and A_independent = sum over layers."""
    R = float(R_arr[row])
    T = float(T_arr[row])
    r = complex(np.asarray(jones_r)[row, row]) * complex(r_factor)
    t = complex(np.asarray(jones_t)[row, row]) * complex(t_factor)
    A = 1.0 - R - T
    a_ind, pra = None, None
    if layer_absorption is not None and layer_names is not None:
        la = np.asarray(layer_absorption)
        col = la[:, row] if la.ndim == 2 else la
        pra = {}
        for name, val in zip(layer_names, col):
            pra[name] = pra.get(name, 0.0) + float(val)
        a_ind = float(np.sum(col))
    return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                         solve_time_s=time.perf_counter() - t0, t=t, T=T, A=A,
                         A_independent=a_ind, R_flux=R, T_flux=T,
                         per_region_absorption=pra)


def _berreman_layer_absorption(lum, layers, n_super, n_sub, lambda_m, theta, phi):
    """Per-layer absorbed fraction (n_layers, 2) via a BerremanStack(retain_internal=True). The
    functional entry has no internal-field hook, so the class is used here ONLY for absorption
    (the far field comes from berreman_jones_1d, which also returns the transmission Jones)."""
    st = lum.BerremanStack(n_substrate=complex(n_sub), n_superstrate=complex(n_super))
    for eps, thk in layers:
        st.add_layer(float(thk), eps=eps)
    st.set_source(float(lambda_m), theta=float(theta), phi=float(phi))
    st.solve(retain_internal=True)
    return np.asarray(st.layer_absorption())


def _rotate_layers_conical(layers, phi_rad: float):
    """Conical incidence via ROTATIONAL COVARIANCE (audit 8.2 step-4 Berreman leg): rotating
    the whole physical problem about z by -phi maps the conical source (k_par along
    (cos phi, sin phi), s-hat = (-sin phi, cos phi)) onto the validated IN-PLANE one (k_par
    along x, s-hat = y) and each layer permittivity onto Rz(-phi) eps Rz(-phi)^T. Berreman is
    the PLANAR tier -- no lateral structure breaks the symmetry -- so the mapped problem is
    EXACT, and the existing lab-row extraction + p-basis conversion apply verbatim ('y' IS
    the rotated s, 'x'/'p' the in-plane transverse). Scalars are rotation-invariant and pass
    through untouched (bit-identical); isotropic stacks therefore reproduce their in-plane
    result at any azimuth exactly (the azimuthal-invariance gate). The RCWA/PMM bridges keep
    their conical guard: a patterned lattice is NOT z-rotation-invariant, so this shortcut
    is wrong there (per-order Jones synthesis remains their documented follow-on)."""
    c, s = np.cos(-phi_rad), np.sin(-phi_rad)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = []
    for eps, thk in layers:
        e = np.asarray(eps)
        if e.ndim == 2:
            out.append((rz @ e @ rz.T, thk))
        else:
            out.append((eps, thk))
    return out


def make_lumenairy_berreman_solver(*, absorption: bool = False, n_slices: Optional[int] = None):
    """Build an `optical_solver` for run_pipeline backed by Lumenairy Berreman, with the exact
    seam signature fn(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult PLUS
    the sweep-aware solve_sweep fast path (one result per wavelength; end media re-derived PER
    wavelength, like the RCWA/PMM bridges). LATERALLY-UNIFORM anisotropic stacks only -- a
    patterned layer raises with a pointer to the RCWA backend. With absorption=True the per-layer
    absorbed-power map (A_independent + per_region_absorption) is filled from a retain_internal
    BerremanStack solve. geo and the passed n_super/n_sub are accepted for seam compatibility but
    unused (the layers re-derive end media from the Design)."""

    def _solve_at(design, eps_by_region, lambda_m):
        t0 = time.perf_counter()
        lum = _require_berreman()
        theta, phi = _angles_rad(design.optical)
        _guard_incidence_side(design.optical)
        layers, n_sup, n_sb, names = design_to_berreman_layers(
            design, lambda_m, eps_by_region=eps_by_region, n_slices=n_slices)
        # conical (azimuth != 0): solve the z-rotated EQUIVALENT in-plane problem so the
        # result is keyed to the rotated s/p eigen-polarizations, matching the FEM (the lab
        # rows of a native phi != 0 solve are s/p mixtures -- the audit C4-2 trap)
        if abs(phi) > 1e-12:
            layers = _rotate_layers_conical(layers, phi)
        R, T, Jr, Jt = lum.berreman_jones_1d(layers, complex(n_sb), complex(n_sup),
                                             float(lambda_m), angle=theta, phi=0.0)
        rf, tf = _p_basis_conversion(getattr(design.optical, "polarization", "y"),
                                     theta, n_sup, n_sb)
        la, a_names = None, None
        if absorption:
            try:
                la = _berreman_layer_absorption(lum, layers, n_sup, n_sb, lambda_m, theta, 0.0)
                a_names = names
            except Exception as exc:                      # pragma: no cover - defensive
                warnings.warn("Berreman bridge: per-layer absorption unavailable ({}); "
                              "A_independent left unset".format(exc), stacklevel=2)
        return berreman_result_to_optical_result(R, T, Jr, Jt, _pol_row(design.optical), t0=t0,
                                                  r_factor=rf, t_factor=tf, layer_absorption=la,
                                                  layer_names=a_names)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        return [_solve_at(design, assemble_at(lam), lam) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    return _solve


class BerremanLayeredSolver:
    """LayeredStackSolver backed by Lumenairy Berreman -- the THIRD concrete implementation of
    the layered seam (TmmLayeredSolver scalar, LumenairyStackSolver RCWA, this one the planar
    anisotropic tier). Consumes a LayeredStack of UNIFORM slabs: `eps` (scalar) and a constant
    `eps_tensor_cell` (a uniform (3,3)) are solved by Berreman; a patterned `eps_cell` / a
    laterally-varying tensor cell / `shapes` RAISE (use the RCWA backend). With absorption=True
    per-slab absorption fills per_region_absorption (keyed 'slab_<i>') + A_independent."""

    def __init__(self, *, absorption: bool = False):
        self.absorption = bool(absorption)

    def solve(self, stack: LayeredStack, lambda_m: float, optical) -> OpticalResult:
        lum = _require_berreman()
        t0 = time.perf_counter()
        _guard_incidence_side(optical)
        layers: List[Tuple[object, float]] = []
        for i, slab in enumerate(stack.slabs):            # already superstrate-side first
            if slab.eps is not None:
                layers.append((complex(slab.eps), float(slab.thickness_m)))
            elif slab.eps_tensor_cell is not None:
                layers.append((_uniform_tensor(slab.eps_tensor_cell,
                                               where="slab {}".format(i)),
                               float(slab.thickness_m)))
            elif slab.eps_cell is not None:
                raise NotImplementedError(
                    "BerremanLayeredSolver: slab {} carries a patterned eps_cell; Berreman is "
                    "the PLANAR tier -- use the RCWA backend (LumenairyStackSolver).".format(i))
            else:
                raise NotImplementedError(
                    "BerremanLayeredSolver: slab {} is an analytic-shape slab (planar tier does "
                    "not pattern); rasterize and use the RCWA backend.".format(i))
        theta, phi = _angles_rad(optical)
        if abs(phi) > 1e-12:                              # conical: rotated equivalent problem
            layers = _rotate_layers_conical(layers, phi)
        R, T, Jr, Jt = lum.berreman_jones_1d(layers, complex(stack.n_sub), complex(stack.n_super),
                                             float(lambda_m), angle=theta, phi=0.0)
        rf, tf = _p_basis_conversion(getattr(optical, "polarization", "y"), theta,
                                     stack.n_super, stack.n_sub)
        la, names = None, None
        if self.absorption:
            la = _berreman_layer_absorption(lum, layers, stack.n_super, stack.n_sub, lambda_m,
                                            theta, 0.0)
            names = ["slab_{}".format(i) for i in range(len(layers))]
        return berreman_result_to_optical_result(R, T, Jr, Jt, _pol_row(optical), t0=t0,
                                                  r_factor=rf, t_factor=tf, layer_absorption=la,
                                                  layer_names=names)
