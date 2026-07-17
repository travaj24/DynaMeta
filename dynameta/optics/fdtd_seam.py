"""FDTD OpticalSolver adapter -- wrap the time-domain FDTD (optics.fdtd_nd) as a drop-in
`optical_solver` for run_pipeline(optical_solver=...), the same pluggable seam the TMM/RCWA/FEM
backends use (core.interfaces.OpticalSolver; the TMM analogue is optics.tmm_reference.make_layered_tmm_solver).

The pipeline invokes the solver once per (bias, wavelength) with (design, geometry, eps_by_region,
lambda_m, n_super, n_sub) and expects an OpticalResult. This adapter maps the Design's through-stack to
FDTD layers (eps frozen at lambda_m), runs a narrow-band 2D/3D FDTD around lambda_m, and reads the
0-order R/T plus the complex (phase-de-embedded) reflection/transmission at lambda_m -- so a
laterally-uniform dielectric/absorbing stack reproduces the TMM/FEM R/T to the FDTD discretization.

WHY FDTD here: dispersion (Drude ADE) and the chi3/all-optical nonlinearity are native, and R_flux/T_flux
carry the full multi-order / cross-pol power. SCOPE:
  * laterally-STRUCTURED cells (pillars/holes/gratings) ARE supported for dim=3: each layer's inclusions
    are rasterized (CrossSection.contains_m) onto the (nx,ny,nz) eps grid -- this is where FDTD earns its
    keep (arbitrary geometry, vs TMM which is exact only for uniform stacks). The lateral grid carries
    REAL eps, so a LOSSY structured layer raises (lossy patterned -> FEM/RCWA);
  * a VACUUM superstrate/substrate (non-vacuum semi-infinite end media raise -> a later increment);
  * strong metals / ENZ are impractical (a Drude band-edge index blows the grid up) -> keep FEM/RCWA there,
    exactly the build-vs-buy verdict (FDTD = broadband / nonlinear / transient, not the ENZ accumulation layer).
ACCURACY: a laterally-uniform stack matches TMM to the FDTD discretization (~1e-4 near a reflection
minimum; ~1-2% for a single THIN resonant slab whose Fabry-Perot fringe shifts with the FDTD's numerical
dispersion -- a general single-slab FDTD effect, identical lossless/lossy, that tightens with resolution).
The lossy/absorbing path (one inverted Drude pole, exact eps at lambda) matches TMM to ~few 1e-3 when not
FP-dominated. Convention exp(-i omega t), SI; Im(eps) > 0 = loss.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.core.interfaces import OpticalResult
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import solve_fdtd_2d, solve_fdtd_3d
from dynameta.optics.rasterize import cell_axes, layer_bg_eps, layer_eps_cell

_VAC_TOL = 1.0e-3                                            # |n - 1| under this counts as vacuum end medium


def _eps_to_fdtd_layer(thickness_m, eps, lambda_m, loss_tol: float = 1.0e-6) -> FDTDLayer:
    """Map a single complex eps(lambda_m) to an FDTDLayer. A pure positive-real eps -> a non-dispersive
    dielectric. A lossy and/or negative-real eps (absorber / metal) -> ONE Drude pole inverted to
    reproduce eps EXACTLY at this omega, with eps_inf held >= 1 so the FDTD background stays stable:
        eps(w) = eps_inf - wp^2/(w^2 + i gamma w),  matched at w0 = 2*pi*c/lambda_m.
    Only this omega is read out, so the Drude's off-omega dispersion is irrelevant to the result."""
    eps = complex(eps)
    er = eps.real
    ei = max(0.0, eps.imag)                                 # passive (Im(eps) >= 0); clamp tiny negatives
    if ei <= loss_tol * (abs(er) + 1.0) and er > 0.0:
        return FDTDLayer(thickness_m=float(thickness_m), eps_inf=float(er))   # pure dielectric
    omega0 = 2.0 * math.pi * C_LIGHT / lambda_m
    if er < 1.0:                                            # absorber/metal: pin eps_inf = 1
        gamma = ei * omega0 / (1.0 - er) if (1.0 - er) > 0.0 else omega0
        eps_inf = 1.0
        wp2 = (1.0 - er) * (omega0 ** 2 + gamma ** 2)
    else:                                                  # high-index lossy: eps_inf = er + ei
        gamma = omega0
        eps_inf = er + ei
        wp2 = 2.0 * ei * omega0 ** 2
    return FDTDLayer(thickness_m=float(thickness_m), eps_inf=float(eps_inf),
                     drude_wp_rad_s=float(math.sqrt(max(wp2, 0.0))), drude_gamma_rad_s=float(gamma))


def effect_eps_to_fdtd_grid(eps_grid, lambda_m: float, loss_tol: float = 1.0e-6):
    """Per-cell time-domain eps hook (roadmap R4): vectorized twin of _eps_to_fdtd_layer that maps a
    COMPLEX eps grid (an EffectModel's per-cell linear eps at a fixed slow drive -- gate E, T, PCM
    fraction, ...) to (eps_inf, wp, gamma) grids for the FDTD lateral seam (solve_fdtd_2d's lateral_wp /
    lateral_gam, or solve_fdtd_3d_mo's lateral_tensor). Byte-identical to _eps_to_fdtd_layer cell-by-cell,
    so a uniform grid reduces to the validated single-layer inversion exactly. exp(-i w t), Im(eps) >= 0
    = loss; the same one-Drude-pole inversion is single-omega-exact (re-sample/re-fit per cell across a
    band for a broadband sweep)."""
    eps = np.asarray(eps_grid, dtype=np.complex128)
    er = eps.real
    ei = np.maximum(eps.imag, 0.0)                          # passive clamp (same as _eps_to_fdtd_layer)
    omega0 = 2.0 * math.pi * C_LIGHT / float(lambda_m)
    lossless = (ei <= loss_tol * (np.abs(er) + 1.0)) & (er > 0.0)
    absorber = (~lossless) & (er < 1.0)                     # eps_inf pinned to 1
    denom = 1.0 - er
    safe = np.where(denom > 0.0, denom, 1.0)
    gamma_abs = np.where(denom > 0.0, ei * omega0 / safe, omega0)
    wp2_abs = denom * (omega0 ** 2 + gamma_abs ** 2)
    gamma_hi = np.full_like(er, omega0)                     # high-index lossy: eps_inf = er + ei
    wp2_hi = 2.0 * ei * omega0 ** 2
    eps_inf = np.where(lossless, er, np.where(absorber, 1.0, er + ei))
    gamma = np.where(lossless, 0.0, np.where(absorber, gamma_abs, gamma_hi))
    wp2 = np.where(lossless, 0.0, np.where(absorber, wp2_abs, wp2_hi))
    wp = np.sqrt(np.maximum(wp2, 0.0))
    return eps_inf, wp, gamma


def _guard_optical_spec(design, *, structured: bool) -> None:
    """audit C5-7: every FDTD seam entry point silently IGNORED design.optical -- a Design
    with theta=30 deg (or azimuth, or bottom incidence) got the normal-incidence top-side
    answer with no warning (probe: R off 10% s-pol / 35% p-pol at 30 deg) while every
    sibling backend (FEM, TMM, lumenairy bridges) honors-or-raises the same fields. Raise
    for what this seam cannot represent, mirroring the sibling guard pattern. At NORMAL
    incidence polarization is provably irrelevant for a laterally-UNIFORM stack (any pol
    accepted); a STRUCTURED cell is solved with a y-polarized source, so only 'y' is
    accepted there."""
    opt = getattr(design, "optical", None)
    if opt is None:
        return
    theta = float(getattr(opt, "incidence_angle_deg", 0.0) or 0.0)
    phi = float(getattr(opt, "azimuth_deg", 0.0) or 0.0)
    side = getattr(opt, "incidence_side", "top") or "top"
    pol = getattr(opt, "polarization", "y") or "y"
    if abs(theta) > 1e-9 or abs(phi) > 1e-9:
        raise NotImplementedError(
            "FDTD seam: oblique/conical incidence (theta={:g} deg, azimuth={:g} deg) is not "
            "wired into the pipeline seam -- the normal-incidence answer would be silently "
            "wrong (audit C5-7); use solve_fdtd_2d_oblique directly, or the FEM/TMM solver."
            .format(theta, phi))
    if side != "top":
        raise NotImplementedError(
            "FDTD seam: incidence_side={!r} is not supported (TOP incidence only) -- swap the "
            "superstrate/substrate in the Design instead (audit C5-7).".format(side))
    if structured and pol != "y":
        raise NotImplementedError(
            "FDTD seam: a structured cell is solved with a y-polarized source; "
            "polarization={!r} would silently get the 'y' answer (audit C5-7) -- use the "
            "FEM/RCWA solver for other polarizations.".format(pol))


def design_to_fdtd_layers(design, lambda_m: float, *, eps_by_region: Optional[Dict] = None):
    """[FDTDLayer] for the through-stack in SUPERSTRATE-FIRST (incidence) order -- the order solve_fdtd_*
    places layers (the Stack lists bottom->top, so reversed). A uniform layer uses the bridge's
    eps_by_region scalar when present (the bias-modulated value), else the material eps(lambda_m). A
    GRADED (gridded, laterally-uniform) EpsField entry is sliced into thin uniform FDTDLayers, mirroring
    the TMM extractor (audit C5-2: it used to fall through to the NOMINAL material eps, silently zeroing
    the bias modulation -- or crashing on a DrudeOptical carrier region whose eps needs n_m3). A TENSOR
    entry raises (the scalar FDTD grid cannot carry anisotropy -- use the FEM); a layer with lateral
    inclusions raises (laterally structured -> FEM, or a future rasterizing FDTD adapter)."""
    from dynameta.core.layered import slice_eps_field
    from dynameta.optics.tmm_reference import S as _S_NM
    layers = []
    for L in reversed(design.stack.layers):                # incidence order: superstrate side first
        if getattr(L, "inclusions", None):
            raise NotImplementedError("design_to_fdtd_layers: layer '{}' has lateral inclusions; the FDTD "
                                      "seam Phase 0 handles laterally-uniform stacks only.".format(L.name))
        ef = (eps_by_region or {}).get(L.name)
        if ef is not None and getattr(ef, "is_tensor", False):
            raise NotImplementedError(
                "design_to_fdtd_layers: layer '{}' carries a TENSOR eps (anisotropic effect); the "
                "scalar FDTD grid cannot represent it -- use the FEM solver.".format(L.name))
        if ef is not None and not getattr(ef, "is_uniform", True):
            v = np.asarray(ef.values_zyx)
            if not np.allclose(v, v[:, :1, :1], rtol=1e-12, atol=0.0):
                raise NotImplementedError(
                    "design_to_fdtd_layers: layer '{}' carries a laterally-STRUCTURED gridded "
                    "EpsField; the uniform-stack FDTD seam cannot represent it -- use the FEM "
                    "solver or the RCWA bridge.".format(L.name))
            # slice_eps_field returns ascending-z (substrate-first) slabs; incidence order
            # is superstrate-first, so reversed (same contract as the TMM extractor and the
            # lumenairy bridges, audit C5-1); axes are nm solver units (the bridge convention)
            for slab in reversed(slice_eps_field(ef, 1.0 / _S_NM)):
                layers.append(_eps_to_fdtd_layer(float(slab.thickness_m), complex(slab.eps), lambda_m))
            continue
        if ef is not None and getattr(ef, "scalar", None) is not None:
            eps = complex(ef.scalar)
        else:
            eps = complex(design.materials.get(L.background_material).eps(lambda_m))
        layers.append(_eps_to_fdtd_layer(L.thickness_m, eps, lambda_m))
    return layers


def fit_drude_to_eps(lambdas_m, eps_values, *, eps_inf0=None, wp0=None, gamma0=None):
    """Least-squares fit of ONE FDTD Drude pole  eps(w) = eps_inf - wp^2/(w^2 + i*w*gamma)  (the FDTDLayer
    convention) to sampled complex eps(lambda). Returns (eps_inf, wp_rad_s, gamma_rad_s) that plug
    straight into an FDTDLayer so the broadband FDTD runs the material's real dispersion ACROSS the band,
    rather than freezing eps at the band centre. Exact for a lossless dielectric (-> wp~0, eps_inf=eps) or
    a genuine single-Drude metal; a least-squares approximation for multi-resonance media over a modest
    band. Convention: Im(eps) > 0 = loss (matches DrudeOptical / the FDTD ADE)."""
    from scipy.optimize import least_squares
    lam = np.asarray(lambdas_m, dtype=float).ravel()
    eps = np.asarray(eps_values, dtype=complex).ravel()
    w = 2.0 * math.pi * C_LIGHT / lam
    er, wmid = eps.real, float(np.median(w))
    if eps_inf0 is None:
        eps_inf0 = max(1.0, float(np.max(er)))             # high-frequency limit
    if wp0 is None:
        wp0 = math.sqrt(max(0.0, eps_inf0 - float(np.min(er)))) * wmid  # from the Re(eps) depression
        if wp0 < 1.0e9:
            wp0 = 0.05 * wmid
    if gamma0 is None:
        gamma0 = 0.1 * wmid

    def resid(p):
        einf, wp, g = p
        model = einf - wp ** 2 / (w ** 2 + 1j * w * g)
        d = model - eps
        return np.concatenate([d.real, d.imag])

    lo = [0.0, 0.0, 1.0e10]
    hi = [60.0, 60.0 * float(np.max(w)), 1.0e17]
    sol = least_squares(resid, [eps_inf0, wp0, gamma0], bounds=(lo, hi))
    einf, wp, g = sol.x
    return float(einf), float(wp), float(g)


def fit_drude_lorentz(lambdas_m, eps_values, *, with_drude=True):
    """Least-squares fit of ONE Drude + ONE Lorentz pole to sampled complex eps(lambda):
      eps(w) = eps_inf - wp^2/(w^2 + i w gd) + d_eps * w0^2/(w0^2 - w^2 - i w gl)
    (the FDTDLayer convention, exp(-i w t), Im(eps) > 0 = loss). The Lorentz pole captures a bound-electron
    / interband resonance the bare Drude cannot, so a metal-with-interband (Au-like) or a resonant
    dielectric is reproduced ACROSS the band, not just at one lambda. Returns an FDTDLayer-ready dict
    {eps_inf, drude_wp_rad_s, drude_gamma_rad_s, lorentz_w0_rad_s, lorentz_gamma_rad_s, lorentz_delta_eps}.
    with_drude=False fits a pure Lorentz (a lossy resonant dielectric)."""
    from scipy.optimize import least_squares
    lam = np.asarray(lambdas_m, dtype=float).ravel()
    eps = np.asarray(eps_values, dtype=complex).ravel()
    w = 2.0 * math.pi * C_LIGHT / lam
    wmid = float(np.median(w)); wmax = float(np.max(w)); wmin = float(np.min(w))

    def model(p):
        einf, wp, gd, w0, gl, de = p
        return einf - wp ** 2 / (w ** 2 + 1j * w * gd) + de * w0 ** 2 / (w0 ** 2 - w ** 2 - 1j * w * gl)

    def resid(p):
        d = model(p) - eps
        return np.concatenate([d.real, d.imag])

    lo = [0.0, 0.0, 1.0e10, 0.1 * wmid, 1.0e10, 0.0]
    hi = [60.0, (60.0 * wmax) if with_drude else 1.0e-3, 1.0e17, 5.0 * wmax, 1.0e17, 60.0]
    # the params span ~15 decades (eps_inf~1 vs w0~1e15); x_scale tells least_squares each param's
    # characteristic size so the trust region is well-conditioned (else it stalls at the start).
    x_scale = [1.0, wmax, wmax, wmax, wmax, 1.0]
    wp0 = (0.5 * wmid) if with_drude else 0.0
    # MULTI-START: a Lorentz resonance makes Re(eps) overshoot, so a single start (esp. eps_inf=max(Re))
    # lands in a local minimum. Scan w0 across the band and a few eps_inf guesses, keep the lowest cost.
    einf_guesses = [max(1.0, float(np.median(eps.real))), 1.0, max(1.0, float(eps.real[np.argmax(w)]))]
    best = None
    for w0g in np.linspace(wmin, wmax, 6):
        for eg in einf_guesses:
            p0 = [eg, wp0, 0.1 * wmid, float(w0g), 0.1 * wmid, 0.5]
            try:
                sol = least_squares(resid, p0, bounds=(lo, hi), x_scale=x_scale, max_nfev=4000)
            except Exception:                              # pragma: no cover - a degenerate start
                continue
            if best is None or sol.cost < best.cost:
                best = sol
    if best is None:                                        # every multi-start failed (degenerate input)
        raise RuntimeError("fit_drude_lorentz: all optimization starts failed -- the eps_values are likely "
                           "degenerate (NaN/inf/constant); check the input or use fit_drude_to_eps.")
    einf, wp, gd, w0, gl, de = best.x
    return dict(eps_inf=float(einf), drude_wp_rad_s=float(wp), drude_gamma_rad_s=float(gd),
                lorentz_w0_rad_s=float(w0), lorentz_gamma_rad_s=float(gl), lorentz_delta_eps=float(de))


def _design_to_fdtd_layers_dispersive(design, lambda_min_m, lambda_max_m, *, eps_band_by_region=None,
                                      n_fit=7):
    """Like design_to_fdtd_layers but DISPERSIVE: each uniform layer's eps(lambda) is sampled across the
    band (from a supplied per-region eps band, else the material as a lambda-function) and fitted to ONE
    FDTD Drude pole, so the broadband FDTD reproduces the material dispersion across the whole band (not
    just at the centre). eps_band_by_region: optional {layer_name -> complex array over the n_fit sample
    wavelengths} for bias-modulated / carrier regions whose eps is not a plain material lambda-function."""
    lams = np.linspace(lambda_min_m, lambda_max_m, int(n_fit))
    layers = []
    for L in reversed(design.stack.layers):                # incidence order: superstrate side first
        if getattr(L, "inclusions", None):
            raise NotImplementedError("dispersive sweep handles laterally-uniform stacks only; layer "
                                      "'{}' has inclusions.".format(L.name))
        band = (eps_band_by_region or {}).get(L.name)
        if band is not None:
            eps_band = np.asarray(band, dtype=complex).ravel()
        else:
            mat = design.materials.get(L.background_material)
            eps_band = np.array([complex(mat.eps(float(l))) for l in lams], dtype=complex)
        einf, wp, g = fit_drude_to_eps(lams, eps_band)
        layers.append(FDTDLayer(thickness_m=float(L.thickness_m), eps_inf=einf,
                                drude_wp_rad_s=wp, drude_gamma_rad_s=g))
    return layers


def eps_profile_from_carrier(n_m3_profile, lambda_m, drude_model):
    """Per-DEPTH complex eps from a carrier-density profile n(z) via a DrudeOptical model -- e.g. an ITO
    accumulation / ENZ profile from a DEVSIM n(z) solve. Returns the eps(z) array (same length as n(z)).
    This is the bridge that makes the gated-ITO optics QUANTITATIVE: the few-nm accumulation has a steep
    eps(z), so a single homogenized eps misses the ENZ; a graded profile (graded_fdtd_layers) captures it."""
    n = np.asarray(n_m3_profile, dtype=float).ravel()
    return np.array([complex(drude_model.eps(float(lambda_m), n_m3=float(ni))) for ni in n], dtype=complex)


def graded_fdtd_layers(thickness_m, eps_z, lambda_m, *, n_slices=None):
    """Slice a graded complex eps(z) profile into thin UNIFORM FDTDLayers (each via the one-Drude inversion
    _eps_to_fdtd_layer), so the FDTD reproduces the DEPTH profile (a quantitative graded ITO/ENZ layer, vs
    a single homogenized eps). eps_z[0] is the FRONT (incidence-side) of the layer; the returned layers are
    in that incidence order. With n_slices < len(eps_z) the profile is resampled to n_slices sublayers.
    The FDTD grid must resolve the slices (the documented ENZ caveat: a few-nm profile needs a fine dz)."""
    eps_z = np.asarray(eps_z, dtype=complex).ravel()
    if n_slices is not None and 0 < n_slices < len(eps_z):
        u = np.linspace(0.0, len(eps_z) - 1.0, int(n_slices))
        idx = np.arange(len(eps_z), dtype=float)
        eps_s = np.interp(u, idx, eps_z.real) + 1j * np.interp(u, idx, eps_z.imag)
    else:
        eps_s = eps_z
    d_sub = float(thickness_m) / len(eps_s)
    return [_eps_to_fdtd_layer(d_sub, e, lambda_m) for e in eps_s]


# The lateral rasterizer (cell_axes / layer_bg_eps / layer_eps_cell) was PROMOTED to the
# public optics.rasterize module (audit 2026-07-05 section 6.3: the lumenairy RCWA bridge
# shares it, and imported the underscore names across module boundaries). Re-exported here
# under the old private names for back-compat (same pattern as the MechanicalProps
# re-export in reliability.fatigue).
_cell_axes = cell_axes
_layer_bg_eps = layer_bg_eps
_layer_eps_cell = layer_eps_cell


def design_has_inclusions(design):
    """True if any layer is laterally STRUCTURED (has inclusions) -> needs the rasterized 3D path."""
    return any(getattr(L, "inclusions", None) for L in design.stack.layers)


def make_structured_lateral(design, lambda_m, *, eps_by_region=None, n_super=1.0, n_sub=1.0):
    """For a laterally-STRUCTURED design, return (layers_for_grid, lateral_fn): `layers_for_grid` are the
    superstrate-first FDTDLayers (eps_inf = the layer's MAX real eps, for grid sizing / z-placement; no
    Drude -> the lateral path is lossless), and `lateral_fn(nx,ny,nz,zc,pad,zs)` paints the real
    cross-section eps onto the (nx,ny,nz) eps_inf grid. The pads are filled with the END-MEDIA eps
    (n_super^2 / n_sub^2; default 1 = vacuum) so a STRUCTURED cell on a non-vacuum substrate (e.g.
    metasurface-on-glass) is correct -- solve_fdtd_3d's lateral_eps_inf OVERWRITES the whole grid, so the
    pads must be painted HERE (else they'd revert to vacuum and break the non-vacuum reference). Layers run
    from z=pad superstrate-first, matching solve_fdtd_3d. The lateral grid carries REAL eps only -> a lossy
    inclusion raises."""
    mats = design.materials
    layers_sf = list(reversed(design.stack.layers))         # incidence order: superstrate side first
    px, py = design.unit_cell.period_x_m, design.unit_cell.period_y_m
    ns2, nb2 = float(n_super) ** 2, float(n_sub) ** 2

    def _layer_max_eps(L):
        vals = [_layer_bg_eps(L, lambda_m, mats, eps_by_region)]
        vals += [complex(mats.get(inc.material).eps(lambda_m)) for inc in L.inclusions]
        return max(1.0, max(v.real for v in vals))
    layers_for_grid = [FDTDLayer(thickness_m=float(L.thickness_m), eps_inf=float(_layer_max_eps(L)))
                       for L in layers_sf]

    def lateral_fn(nx, ny, nz, zc, pad, z_struct):
        xs, ys = _cell_axes(nx, ny, px, py)
        X, Y = np.meshgrid(xs, ys, indexing="ij")           # (nx,ny)
        eps = np.ones((nx, ny, nz))
        eps[:, :, zc < pad] = ns2                           # superstrate semi-infinite pad (n_super^2)
        eps[:, :, zc >= pad + z_struct] = nb2               # substrate semi-infinite pad (n_sub^2)
        z = pad
        for L in layers_sf:
            zmask = (zc >= z) & (zc < z + L.thickness_m)
            cell = _layer_eps_cell(L, X, Y, lambda_m, mats, eps_by_region)
            if np.max(np.abs(cell.imag)) > 1e-6 * (np.max(np.abs(cell.real)) + 1.0):
                raise NotImplementedError(
                    "structured FDTD layer '{}' has lossy eps (Im != 0); the lateral grid carries real "
                    "eps_inf only -> use FEM/RCWA for lossy structured cells.".format(L.name))
            eps[:, :, zmask] = cell.real[:, :, None]
            z += L.thickness_m
        return eps

    return layers_for_grid, lateral_fn


def make_fdtd_optical_solver(*, dim: int = 2, resolution: int = 32, backend: str = "auto",
                             courant: float = 0.5, settle: float = 12.0, n_pad_wave: float = 4.0,
                             band_frac: float = 0.06):
    """Build an `optical_solver` callable wrapping the 2D (dim=2) or 3D (dim=3) FDTD, for
    run_pipeline(optical_solver=make_fdtd_optical_solver(...)). Each (bias, wavelength) call freezes the
    materials at lambda_m, runs a narrow-band FDTD around lambda_m, and returns the 0-order R/T/phase at
    lambda_m as an OpticalResult (with R_flux/T_flux = all-order flux). backend='auto' -> the fast numba
    kernel. A laterally-uniform stack gives the same result for dim=2 or 3, so 2 (faster) is the default;
    a laterally-STRUCTURED cell (layer inclusions) is rasterized onto the 3D grid and REQUIRES dim=3.

    Supports LOSSLESS semi-infinite end media (real n_super/n_sub, e.g. metasurface-on-glass); raises
    NotImplementedError for a LOSSY (complex) end medium, a structured cell with dim!=3, or a LOSSY
    structured layer (use the FEM/RCWA solver there). See the module docstring for the scope."""
    if dim not in (2, 3):
        raise ValueError("make_fdtd_optical_solver: dim must be 2 or 3")

    def _solve(design, geometry, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult:
        ns, nb = complex(n_super), complex(n_sub)
        if abs(ns.imag) > _VAC_TOL or abs(nb.imag) > _VAC_TOL:
            raise NotImplementedError(
                "FDTD seam supports LOSSLESS semi-infinite end media; got n_super={:.4g}, n_sub={:.4g} "
                "(absorbing incidence/exit medium -> use the FEM/TMM solver).".format(ns, nb))
        structured = design_has_inclusions(design)
        _guard_optical_spec(design, structured=structured)   # audit C5-7
        if structured and dim != 3:
            raise NotImplementedError(
                "a laterally-structured cell (layer inclusions) needs dim=3; got dim={}.".format(dim))
        lo, hi = lambda_m * (1.0 - band_frac), lambda_m * (1.0 + band_frac)
        kw = dict(lambda_min_m=lo, lambda_max_m=hi, resolution=resolution, courant=courant,
                  settle=settle, n_pad_wave=n_pad_wave, backend=backend,
                  n_super=ns.real, n_sub=nb.real)
        px, py = design.unit_cell.period_x_m, design.unit_cell.period_y_m
        t_start = time.time()
        if structured:
            layers, lateral_fn = make_structured_lateral(design, lambda_m, eps_by_region=eps_by_region,
                                                         n_super=ns.real, n_sub=nb.real)
            res = solve_fdtd_3d(layers, period_x_m=px, period_y_m=py, lateral_eps_inf=lateral_fn, **kw)
        elif dim == 2:
            res = solve_fdtd_2d(design_to_fdtd_layers(design, lambda_m, eps_by_region=eps_by_region),
                                period_x_m=px, **kw)
        else:
            res = solve_fdtd_3d(design_to_fdtd_layers(design, lambda_m, eps_by_region=eps_by_region),
                                period_x_m=px, period_y_m=py, **kw)
        solve_time_s = time.time() - t_start
        # Interpolate the spectrum to EXACTLY f=c/lambda (not the nearest FFT bin): for a dispersive Drude
        # layer the eps at an off-by-half-bin frequency differs from the target eps(lambda), which would
        # bias R/A; at exactly c/lambda the inverted Drude reproduces eps(lambda). freqs are increasing.
        ft = C_LIGHT / lambda_m
        f = res.freqs_Hz
        at = (lambda a: float(np.interp(ft, f, a)))
        cx = (lambda a: complex(np.interp(ft, f, a.real), np.interp(ft, f, a.imag)))
        R = at(res.R0); T = at(res.T0)
        r = cx(res.r0) if res.r0 is not None else complex(math.sqrt(max(R, 0.0)))
        t = cx(res.t0) if res.t0 is not None else None
        # absorption from the ALL-ORDER flux (R_flux/T_flux), NOT the 0-order specular R0/T0: for a
        # diffracting (structured) cell the diffracted power leaves through non-specular orders, so
        # 1 - R0 - T0 would count it as spurious absorption. Falls back to specular only if flux is
        # absent (uniform path: R_flux==R0, T_flux==T0, so this is byte-stable there).
        Rf = at(res.R_flux) if res.R_flux is not None else R
        Tf = at(res.T_flux) if res.T_flux is not None else T
        return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))), solve_time_s=solve_time_s,
                             t=t, T=T, A=float(1.0 - Rf - Tf), R_flux=Rf, T_flux=Tf)

    _solve.cache_fingerprint = "fdtd(dim={},res={},backend={},courant={},settle={},pad={},band={})".format(dim, resolution, backend, courant, settle, n_pad_wave, band_frac)
    return _solve


@dataclass
class FDTDSweepResult:
    """The full R/T spectrum of ONE broadband FDTD solve, on the well-excited band (sorted by wavelength).
    R/T are the 0-order specular; R_flux/T_flux the all-(kx,ky)-order flux; r/t the complex de-embedded
    0-order coefficients (phase). solve_time_s is the wall time of the single solve."""
    lambda_m: np.ndarray
    R: np.ndarray
    T: np.ndarray
    A: np.ndarray
    R_flux: np.ndarray
    T_flux: np.ndarray
    r: np.ndarray
    t: np.ndarray
    solve_time_s: float


def fdtd_sweep_spectrum(design, *, lambda_min_m, lambda_max_m, eps_by_region=None, dim=2,
                        resolution=32, backend="auto", courant=0.5, settle=12.0, n_pad_wave=4.0,
                        n_super=1.0 + 0j, n_sub=1.0 + 0j, dispersive=True, eps_band_by_region=None, n_fit=7):
    """ONE broadband FDTD over [lambda_min_m, lambda_max_m] -> the WHOLE R/T spectrum, vs the per-wavelength
    OpticalSolver seam (make_fdtd_optical_solver) which re-solves each wavelength. This is FDTD's native
    strength: one solve serves the whole sweep -- the fast path for a wavelength sweep at a fixed bias
    (N wavelengths in ~1 solve instead of N). Returns an FDTDSweepResult over the well-excited band.

    DISPERSION: with dispersive=True (default) each uniform layer's eps(lambda) is sampled across the band
    and fitted to ONE Drude pole the FDTD runs natively -> the spectrum is accurate across the band for a
    DISPERSIVE material (metal/ITO/Drude), not just a dielectric. Pass eps_band_by_region={name -> complex
    array over n_fit sample wavelengths} for bias-modulated carrier regions whose eps is not a plain
    material lambda-function (see run_fdtd_sweep). dispersive=False freezes eps at the band centre (exact
    only for a non-dispersive design). STRUCTURED cells stay frozen-at-centre (the lateral grid is real
    eps; a lossy/dispersive structured layer is out of scope -> FEM/RCWA). Scope: vacuum end media; uniform
    or structured (dim=3, vacuum end media only). Tip: request a band ~10-20% wider than your target so the
    tapered edges fall out. LOSSLESS non-vacuum end media (real n_super/n_sub) are supported for uniform
    stacks."""
    ns, nb = complex(n_super), complex(n_sub)
    if abs(ns.imag) > _VAC_TOL or abs(nb.imag) > _VAC_TOL:
        raise NotImplementedError("fdtd_sweep_spectrum supports LOSSLESS end media; got n_super={:.4g}, "
                                  "n_sub={:.4g} (absorbing -> FEM/TMM).".format(ns, nb))
    lam_c = 0.5 * (lambda_min_m + lambda_max_m)              # band centre (the eps freeze point if non-dispersive)
    px, py = design.unit_cell.period_x_m, design.unit_cell.period_y_m
    kw = dict(lambda_min_m=lambda_min_m, lambda_max_m=lambda_max_m, resolution=resolution,
              courant=courant, settle=settle, n_pad_wave=n_pad_wave, backend=backend,
              n_super=ns.real, n_sub=nb.real)
    structured = design_has_inclusions(design)
    _guard_optical_spec(design, structured=structured)       # audit C5-7
    if structured and dim != 3:
        raise NotImplementedError("a structured cell (inclusions) needs dim=3; got dim={}.".format(dim))
    if structured and (abs(ns.real - 1.0) > _VAC_TOL or abs(nb.real - 1.0) > _VAC_TOL):
        raise NotImplementedError("non-vacuum end media + a structured cell is not yet supported "
                                  "(the lateral rasterizer drops the end-media pads); use a uniform stack.")
    t_start = time.time()
    if structured:                                          # structured -> frozen-at-centre, real lateral eps
        layers, lateral_fn = make_structured_lateral(design, lam_c, eps_by_region=eps_by_region)
        res = solve_fdtd_3d(layers, period_x_m=px, period_y_m=py, lateral_eps_inf=lateral_fn, **kw)
    else:
        if dispersive:                                      # fit one Drude pole per layer across the band
            # audit C5-2: the dispersive band fit never reads eps_by_region (modulation must
            # arrive via eps_band_by_region); a graded/tensor entry here used to be silently
            # IGNORED -- refuse rather than solve the unmodulated stack
            for _name, _ef in (eps_by_region or {}).items():
                if _ef is not None and (getattr(_ef, "is_tensor", False)
                                        or not getattr(_ef, "is_uniform", True)):
                    raise NotImplementedError(
                        "fdtd_sweep_spectrum(dispersive=True): eps_by_region['{}'] is a {} "
                        "EpsField; the one-Drude-pole-per-layer band fit cannot carry it. Use "
                        "the per-wavelength FDTD solver (which slices graded layers), or "
                        "dispersive=False (frozen-at-centre slicing), or TMM/FEM.".format(
                            _name, "TENSOR" if getattr(_ef, "is_tensor", False)
                            else "graded (gridded)"))
            layers = _design_to_fdtd_layers_dispersive(design, lambda_min_m, lambda_max_m,
                                                        eps_band_by_region=eps_band_by_region, n_fit=n_fit)
        else:
            layers = design_to_fdtd_layers(design, lam_c, eps_by_region=eps_by_region)
        res = (solve_fdtd_2d(layers, period_x_m=px, **kw) if dim == 2
               else solve_fdtd_3d(layers, period_x_m=px, period_y_m=py, **kw))
    solve_time_s = time.time() - t_start
    f = res.freqs_Hz
    m = res.band & (f > 0)
    lam = C_LIGHT / f[m]
    order = np.argsort(lam)
    sel = (lambda a: np.asarray(a)[m][order])
    R, T = sel(res.R0), sel(res.T0)
    Rf = sel(res.R_flux) if res.R_flux is not None else R
    Tf = sel(res.T_flux) if res.T_flux is not None else T
    return FDTDSweepResult(lambda_m=lam[order], R=R, T=T, A=1.0 - Rf - Tf,   # all-order flux, not 0-order
                           R_flux=Rf, T_flux=Tf,
                           r=sel(res.r0), t=sel(res.t0), solve_time_s=solve_time_s)


def run_fdtd_sweep(design, lambdas_m, *, dim=2, resolution=32, backend="auto", eps_band_by_region=None,
                   band_pad=0.12, n_super=1.0 + 0j, n_sub=1.0 + 0j, **kw):
    """Sweep-aware FAST PATH: ONE broadband FDTD over the span of `lambdas_m`, then serve EACH wavelength
    by interpolation -> a list of OpticalResult (one per wavelength) -- the SAME per-(bias,wavelength)
    output run_pipeline's optical_solver produces, but from a SINGLE solve instead of N. This is the fix
    for the per-wavelength seam re-running the full settling tail at every wavelength (the audit's medium
    finding). dispersive=True (passed through, default) makes it accurate for dispersive layers; pass
    eps_band_by_region={name -> complex array over the fit wavelengths} for bias-modulated carrier regions.
    `band_pad` widens the solved band beyond the requested wavelengths so the pulse-tapered band edges fall
    OUTSIDE the served range. Use one call per bias in place of the per-wavelength FDTD seam loop."""
    lams = np.asarray(lambdas_m, dtype=float).ravel()
    lo, hi = float(lams.min()) * (1.0 - band_pad), float(lams.max()) * (1.0 + band_pad)
    sw = fdtd_sweep_spectrum(design, lambda_min_m=lo, lambda_max_m=hi, dim=dim, resolution=resolution,
                             backend=backend, eps_band_by_region=eps_band_by_region,
                             n_super=n_super, n_sub=n_sub, **kw)
    ip = (lambda a: np.interp(lams, sw.lambda_m, a))         # sw.lambda_m is increasing (well-excited band)
    R, T, Rf, Tf = ip(sw.R), ip(sw.T), ip(sw.R_flux), ip(sw.T_flux)
    rr = ip(sw.r.real) + 1j * ip(sw.r.imag)
    tt = ip(sw.t.real) + 1j * ip(sw.t.imag)
    return [OpticalResult(r=complex(rr[i]), R=float(R[i]), phase_deg=float(np.degrees(np.angle(rr[i]))),
                          solve_time_s=(float(sw.solve_time_s) if i == 0 else 0.0), t=complex(tt[i]),
                          T=float(T[i]), A=float(1.0 - Rf[i] - Tf[i]),       # all-order flux, not 0-order
                          R_flux=float(Rf[i]), T_flux=float(Tf[i]))
            for i in range(lams.size)]


class FDTDSweepOpticalSolver:
    """A SWEEP-AWARE optical_solver: ONE broadband FDTD per bias serves the whole wavelength sweep, instead
    of the per-wavelength make_fdtd_optical_solver re-settling a full pulse transient at every wavelength
    (10-100x fewer settling tails for an N-wavelength dispersive sweep). It is BOTH:
      * a normal per-(bias,wavelength) optical_solver (the __call__, a thin wrapper over make_fdtd_optical_
        solver) -- so it is a drop-in wherever an OpticalSolver is expected; and
      * sweep-aware: run_pipeline detects the `solve_sweep` method and, per bias, hands it an
        `assemble_eps_at(lambda_m)` callback + the wavelength list, so ONE broadband solve covers them all.

    The per-layer eps dispersion across the band is sampled from the bias's eps (the bias-modulated carrier
    region included) at the SAME n_fit wavelengths the dispersive layer-fitter uses, so the broadband result
    reproduces the per-wavelength one to the Drude-fit accuracy (exact for a Drude carrier region + non-
    dispersive dielectrics; a strongly-dispersive dielectric is the single-pole-fit limit). Uniform stacks
    only for the dispersive band; a structured cell still freezes at the band centre (one solve regardless)."""

    def __init__(self, *, dim: int = 2, resolution: int = 32, backend: str = "auto", courant: float = 0.5,
                 settle: float = 12.0, n_pad_wave: float = 4.0, n_fit: int = 7, band_pad: float = 0.08):
        self.dim = dim
        self.resolution = resolution
        self.backend = backend
        self.courant = courant
        self.settle = settle
        self.n_pad_wave = n_pad_wave
        self.n_fit = int(n_fit)
        self.band_pad = float(band_pad)
        # audit S5-2: answer-changing settings folded into the optical-cache identity
        self.cache_fingerprint = ("fdtd_sweep(dim={},res={},backend={},courant={},settle={},"
                                  "pad={},n_fit={},band_pad={})".format(
                                      dim, resolution, backend, courant, settle, n_pad_wave,
                                      n_fit, band_pad))
        self._per_wl = make_fdtd_optical_solver(dim=dim, resolution=resolution, backend=backend,
                                                courant=courant, settle=settle, n_pad_wave=n_pad_wave)

    def __call__(self, design, geometry, eps_by_region, lambda_m, n_super, n_sub):
        """Per-(bias,wavelength) fallback (a plain OpticalSolver), so this object also works in any context
        that calls the seam once per wavelength."""
        return self._per_wl(design, geometry, eps_by_region, lambda_m, n_super, n_sub)

    def solve_sweep(self, design, geometry, assemble_eps_at, lambdas_m, n_super, n_sub):
        """ONE broadband FDTD for the whole sweep at a fixed bias. `assemble_eps_at(lambda_m)` returns the
        per-layer eps_by_region dict for THIS bias at a given wavelength (run_pipeline's bridge closure);
        it is sampled at the n_fit band wavelengths to build the per-layer dispersion, then a single
        broadband FDTD (run_fdtd_sweep) is interpolated to each requested wavelength. Returns one
        OpticalResult per entry of lambdas_m (same order)."""
        lams = np.asarray(lambdas_m, dtype=float).ravel()
        lo, hi = float(lams.min()) * (1.0 - self.band_pad), float(lams.max()) * (1.0 + self.band_pad)
        fit_lams = np.linspace(lo, hi, self.n_fit)           # MUST match _design_to_fdtd_layers_dispersive
        samples = [assemble_eps_at(float(l)) for l in fit_lams]
        # build the per-layer eps band from each sample's uniform-scalar eps (bias-modulated carrier incl.);
        # a layer without a uniform scalar at every sample falls back to its material lambda-function.
        eps_band = {}
        for L in design.stack.layers:
            vals = []
            for s in samples:
                ef = (s or {}).get(L.name)
                if ef is not None and (getattr(ef, "is_tensor", False)
                                       or not getattr(ef, "is_uniform", True)):
                    # audit C5-2: a graded/tensor bias eps used to silently fall back to the
                    # NOMINAL material band (zeroing the modulation, or crashing on a
                    # DrudeOptical carrier region with a misleading n_m3 error)
                    raise NotImplementedError(
                        "FDTDSweepOpticalSolver.solve_sweep: layer '{}' carries a {} EpsField "
                        "for this bias; the broadband one-pole-per-layer path cannot carry it. "
                        "Use the per-wavelength FDTD solver (make_fdtd_optical_solver, which "
                        "slices graded layers) or the TMM sweep.".format(
                            L.name, "TENSOR" if getattr(ef, "is_tensor", False)
                            else "graded (gridded)"))
                if ef is not None and getattr(ef, "scalar", None) is not None:
                    vals.append(complex(ef.scalar))
                else:
                    vals = None
                    break
            if vals is not None:
                eps_band[L.name] = np.asarray(vals, dtype=complex)
        return run_fdtd_sweep(design, lams, dim=self.dim, resolution=self.resolution, backend=self.backend,
                              eps_band_by_region=(eps_band or None), band_pad=self.band_pad,
                              n_super=n_super, n_sub=n_sub, courant=self.courant, settle=self.settle,
                              n_pad_wave=self.n_pad_wave, n_fit=self.n_fit)


def make_fdtd_sweep_optical_solver(*, dim: int = 2, resolution: int = 32, backend: str = "auto",
                                   courant: float = 0.5, settle: float = 12.0, n_pad_wave: float = 4.0,
                                   n_fit: int = 7, band_pad: float = 0.08) -> "FDTDSweepOpticalSolver":
    """Build a SWEEP-AWARE FDTD optical_solver for run_pipeline(optical_solver=...): one broadband FDTD per
    bias serves the whole wavelength sweep (vs the per-wavelength make_fdtd_optical_solver re-settling each
    wavelength). See FDTDSweepOpticalSolver. Drop-in: run_pipeline auto-uses the fast sweep path."""
    return FDTDSweepOpticalSolver(dim=dim, resolution=resolution, backend=backend, courant=courant,
                                  settle=settle, n_pad_wave=n_pad_wave, n_fit=n_fit, band_pad=band_pad)
