"""2D solve front-ends: grid fill, coefficient builders, dispatch, R/T extraction.

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, T_REF
from dynameta.optics.fdtd_nd.spec import FDTDLayer, hot_carrier_guard
from dynameta.optics.fdtd_nd.backends import HAVE_NUMBA, have_jax, resolve_backend
from dynameta.optics.fdtd_nd.kernels2d_numba import _te2d_cuda, _te2d_numba
from dynameta.optics.fdtd_nd.results import FDTD2DObliqueResult, FDTD2DResult, _flux
from dynameta.optics.fdtd_nd.cpml import cpml_z
from dynameta.optics.fdtd_nd.kernels2d import run_2d_te
from dynameta.optics.fdtd_nd.kernels2d_jax import run_2d_te_jax
from dynameta.optics.fdtd_nd.oblique2d import _run_oblique



# --- homogeneous-superstrate REFERENCE-run cache (audit 6.2 perf) ------------------------------
# The pipeline seam invokes solve_fdtd_2d/_3d once per (bias, wavelength); the incident-reference
# (normalization) run depends ONLY on the grid, the source and the superstrate -- not on the
# structure -- so repeated seam solves at a fixed wavelength recompute an IDENTICAL reference
# (~2x total cost). Cache the last few probe tuples keyed on the EXACT inputs that determine the
# run (bytes/tuple key, exact reuse only -- a hit returns the same arrays, bit-identical by
# construction). Entries are marked read-only so an accidental downstream mutation fails loudly
# instead of silently corrupting later solves.
_REF_CACHE = {}
_REF_CACHE_MAX = 4                                           # FIFO entries (dict = insertion order)
_REF_CACHE_MAX_BYTES = 512 * 1024 * 1024                     # skip caching huge (production-3D) refs


def _ref_cache_call(key, fn):
    """Return fn() memoized on `key` (exact match only). Oversized results pass through uncached."""
    out = _REF_CACHE.get(key)
    if out is not None:
        return out
    out = fn()
    if sum(a.nbytes for a in out) <= _REF_CACHE_MAX_BYTES:
        for a in out:
            a.setflags(write=False)
        while len(_REF_CACHE) >= _REF_CACHE_MAX:
            _REF_CACHE.pop(next(iter(_REF_CACHE)))
        _REF_CACHE[key] = out
    return out


def _ring_time_s(layers) -> float:
    """Material-memory ring-down time (audit C3-6): the fixed 200*tau DFT window predates
    the Lorentz/gain ADEs -- a high-loaded-Q in-band pole rings past it, truncating the
    rfft with O(0.1) silent R0/T0 bias (probe: |dT0| = 0.102 vs the TMM oracle for a
    Q~600 line, no warning possible since the band mask checks excitation only). Returns
    the (2/Gamma) ln(1/1e-4) ~ 18.4/Gamma memory of the NARROWEST active Lorentz/gain
    pole (0.0 when no pole is active -> the legacy window, byte-identical)."""
    t_ring = 0.0
    for L in layers:
        if getattr(L, "lorentz_delta_eps", 0.0) != 0.0 and getattr(L, "lorentz_gamma_rad_s", 0.0) > 0.0:
            t_ring = max(t_ring, 18.4 / float(L.lorentz_gamma_rad_s))
        if getattr(L, "gain_dN_m3", 0.0) != 0.0 and getattr(L, "gain_dw_rad_s", 0.0) > 0.0:
            t_ring = max(t_ring, 18.4 / float(L.gain_dw_rad_s))
    return t_ring


# --- absorber thickness + probe placement vs the CPML (audit D-3, wave-3 redesign) --------------
# WHAT ACTUALLY CORRUPTS R/T. Measured 2026-07-26 on four fixtures (lossless eps=4 slab, backend
# 'numpy') with this guard bypassed, holding the GEOMETRY fixed and sweeping ONLY npml -- which
# isolates absorber effects from near-field contamination. The tables live in
# tests/test_fdtd_seam.py (test_d3_guard_verdict_matches_the_measured_fixtures and
# test_d3_npml_floor_is_a_hard_raise_below_the_measured_cliff); the headline rows are:
#
#  (1) TOO THIN AN ABSORBER -- the mode the pre-wave-3 guard did not test AT ALL. `npml` was never
#      validated, so npml <= 2 passed silently at (600 nm fixture) R0+T0 up to 1.8387 and (300 nm
#      fixture) R_flux+T_flux up to 2.9721 -- an 84-197 % energy-budget violation on a LOSSLESS
#      slab, WORSE than anything the guard did reject. npml=3 still costs 3.2-6.3 %, npml=4 lands
#      within 0.7-2.0 %, npml>=5 within 0.5 %. Hence a hard floor `_NPML_MIN` = 4 plus a warning
#      below `_NPML_WARN` = 6.
#
#  (2) A PROBE PLANE BURIED IN THE GRADED ABSORBER -- the binding placement constraint. cpml_z
#      grades sigma as (depth/npml)^3, so burial depth matters far more than "inside/outside":
#      600 nm fixture (k_pL=11, k_pR=38, nz=50), R_flux+T_flux vs probe depth d:
#          d = 0 (npml=11): 1.0000     d = 1 (npml=12): 0.9983     d = 2 (npml=13): 0.9836
#          d = 3 (npml=14): 0.9416     d = 4 (npml=15): 0.8670     d = 9 (npml=20): 0.3052
#      The predicted one-way attenuation `_pml_atten` tracks that within a factor ~1.5 (d=2:
#      predicted 1.7 % power deficit vs 1.6 % measured; d=3: 5.1 % vs 5.8 %; d=4: 10.3 % vs 13.3 %),
#      so the rule is an ATTENUATION budget, not a cell count -- one cell of burial is harmless at
#      npml=12 (0.13 %) and fatal at npml=4 (10.8 %).
#
#  (3) THE SOURCE IS NOT A CORRECTNESS PROBLEM -- and rejecting it was the wave-2 guard's mistake.
#      R0/T0/R_flux/T_flux are two-run DFT RATIOS against a vacuum reference injected through the
#      SAME absorber, and the reflected/transmitted wave inherits the same launch attenuation as
#      the incident wave that produced it, so the attenuation cancels: with the source buried 5-6
#      cells (npml=11-12) and the probes clear, both narrow-band fixtures return R0+T0 = 1.0000 and
#      R_flux+T_flux within 2e-4. Burial only costs SNR (a broadband 1.2-1.8 um fixture with the
#      source 5-8 cells deep scattered its band-edge bins by up to +-1.2 %), so it WARNS. The
#      terminal case -- the pulse never reaching the probe at all -- is caught by _check_band.
#
# Consequence for the callers: the old `k_src >= npml + 2` rule rejected npml=5..12 thin-pad
# configurations that measure PERFECT, including the library default npml=12 and
# validation/fdtd_oblique_jax.py GATE D (which passed by exactly one cell).
_NPML_MIN = 4                                                # hard floor on the CPML thickness (cells)
_NPML_WARN = 6                                               # below this the budget error is 0.5-2 %
_PROBE_ATTEN_MAX = 3.0e-3                                    # tolerated one-way amplitude loss at a probe


def _pml_atten(depth_cells, npml, m=3.0, R0=1.0e-6):
    """One-way AMPLITUDE attenuation between the CPML interface and a plane `depth_cells` cells
    inside the graded absorber (0.0 at or outside the interface).

    cpml_z grades sigma_j = sigma_max (j/npml)^m at depth j cells with
    sigma_max*dz = -(m+1) ln(R0) / (2 eta0 npml), and its docstring pins the one-way attenuation at
    exp(-n eta0 Int sigma dz) with sigma ~ 1/n matched to the end medium -- so n cancels and the
    exponent is a pure function of (depth, npml). Summing the Yee cells the wave crosses:

        a(d, npml) = 1 - exp(-[-(m+1) ln R0 / (2 npml)] * sum_{j=1..d} (j/npml)^m)
                   = 1 - exp(-27.63 * (d(d+1)/2)^2 / npml^4)          for m=3, R0=1e-6

    `m` / `R0` are cpml_z's own defaults: every solve_* front end calls cpml_z(nz, dz, dt, npml,
    n_super, n_sub) and never overrides them (pinned by test_d3_atten_model_uses_cpml_z_defaults).
    The flux extraction's H_x average also reads k-1, i.e. half a cell deeper than the E plane;
    that half cell is inside the factor-1.5 accuracy of this estimate and is absorbed by the
    `_PROBE_ATTEN_MAX` margin.
    """
    d = int(depth_cells)
    if d <= 0:
        return 0.0
    a = (-(m + 1.0) * np.log(R0) / (2.0 * npml)) * sum((j / float(npml)) ** m
                                                       for j in range(1, d + 1))
    return float(1.0 - np.exp(-a))


def _max_probe_depth(npml):
    """Deepest probe burial (cells) whose `_pml_atten` still fits the `_PROBE_ATTEN_MAX` budget:
    0 for npml <= 9 (one cell already costs 0.42-10.2 % there), 1 at npml=10..16, 2 at 17..23,
    3 at 24..30, ... -- the budget grows as npml^4 because sigma is graded (depth/npml)^3."""
    d = 0
    while _pml_atten(d + 1, npml) <= _PROBE_ATTEN_MAX:
        d += 1
    return d


def _check_probe_placement(entry_point, k_src, k_pL, k_pR, nz, npml, pad, dz,
                           n_pad_wave, resolution):
    """audit D-3: refuse an absorber too thin to absorb, or an R/T probe plane buried in it.

    Two hard rules and one warning (see the module-level block above for the measured evidence):

      * `npml >= _NPML_MIN` -- nothing else checked the absorber itself.
      * both probe planes must clear the graded absorber to within `_PROBE_ATTEN_MAX` of one-way
        amplitude attenuation. cpml_z puts the low-z grading on cells 0..npml-1 (depth
        npml - k, zero AT k = npml) and the high-z grading on cells nz-npml..nz-1 (depth
        k - (nz-1-npml)), so the burial depths are `npml - k_pL` and `k_pR - (nz-1-npml)`.
      * a source inside the absorber only WARNS: the two-run ratio cancels its launch attenuation.

    Every front end places k_src / k_pL / k_pR as FRACTIONS OF THE Z PAD (0.35 pad, 0.7 pad, and
    0.3 pad past the structure) while `npml` is a fixed CELL count, so a thin pad (small
    `n_pad_wave`, or a coarse `resolution`) slides them into the absorber. Both probes bind at the
    SAME pad depth -- k_pL - npml and (nz-1-npml) - k_pR are both 0.7*pad/dz - npml -- so one
    number, `0.7 * pad/dz >= npml - _max_probe_depth(npml)`, is the whole placement contract, and
    the message inverts it onto each knob.
    """
    npml = int(npml)
    if npml < _NPML_MIN:
        raise ValueError(
            "{}: npml={} is too thin a CPML -- the absorber itself, not the probe placement, then "
            "dominates the error. Measured on a LOSSLESS eps=4 slab with the probes fully clear: "
            "npml=1 gives R_flux+T_flux up to 2.97, npml=2 up to 1.46 (R0+T0 up to 1.84), npml=3 "
            "still 3.2-6.3 % off; npml>=4 lands within 2 % and npml>=5 within 0.5 % (audit D-3). "
            "Use npml >= {} (>= 8 recommended); the Roden-Gedney polynomial grading needs several "
            "cells before the discretized profile absorbs anything.".format(
                entry_point, npml, _NPML_MIN))
    if npml < _NPML_WARN:
        import warnings
        warnings.warn(
            "{}: npml={} is a thin CPML -- measured energy-budget error on a lossless slab is "
            "0.5-2.0 % at npml=4-5 versus <0.1 % at npml>=8 (audit D-3). Raise npml if you need "
            "better than ~1 %.".format(entry_point, npml), RuntimeWarning, stacklevel=3)

    d_lo = npml - k_pL                                       # cells the R probe sits inside the low-z PML
    d_hi = k_pR - (nz - 1 - npml)                            # cells the T probe sits inside the high-z PML
    a_lo, a_hi = _pml_atten(d_lo, npml), _pml_atten(d_hi, npml)
    bad = []
    if a_lo > _PROBE_ATTEN_MAX:
        bad.append("left (R) probe k_pL={} is {} cell(s) inside the low-z CPML (npml={}) -> "
                   "{:.2%} one-way amplitude attenuation".format(k_pL, d_lo, npml, a_lo))
    if a_hi > _PROBE_ATTEN_MAX:
        bad.append("right (T) probe k_pR={} is {} cell(s) inside the high-z CPML (nz={}, npml={}) "
                   "-> {:.2%} one-way amplitude attenuation".format(k_pR, d_hi, nz, npml, a_hi))
    if bad:
        # 0.7 * (pad/dz) is the single scale BOTH probe clearances reduce to (see the docstring).
        d_max = _max_probe_depth(npml)
        p_cells = pad / dz
        p_need = (npml - d_max) / 0.7
        scale = p_need / p_cells if p_cells > 0 else float("inf")
        npml_ok = 0
        for n in range(_NPML_MIN, npml):                     # largest npml this pad can still carry
            if n - _max_probe_depth(n) <= 0.7 * p_cells:
                npml_ok = n
        fix_npml = ("OR lowering npml to <= {} (currently {})".format(npml_ok, npml) if npml_ok
                    else "(npml cannot help: even the floor npml={} would not clear this pad)"
                         .format(_NPML_MIN))
        raise ValueError(
            "{}: {} -- the R/T probe planes are placed as fractions of the z pad and have fallen "
            "INSIDE the CPML, where the graded absorber damps the recorded field and R/T degrade "
            "SILENTLY (measured on a LOSSLESS slab with both probes at the same depth d: "
            "R_flux+T_flux = 0.9836 at d=2 / 0.87 % predicted attenuation, 0.8670 at d=4 / 5.3 %, "
            "0.3052 at d=9 / 29.5 %; audit D-3). The tolerance is {:.2%}, "
            "i.e. at npml={} a probe may be at most {} cell(s) deep. The pad is {:.1f} cells and "
            "needs >= {:.1f}: fix by raising n_pad_wave to >= {:.2f} (currently {:g}) at "
            "resolution={:g}, OR raising resolution to >= {:d} (currently {:g}) at "
            "n_pad_wave={:g}, {}.".format(
                entry_point, "; ".join(bad), _PROBE_ATTEN_MAX, npml, d_max, p_cells, p_need,
                n_pad_wave * scale, n_pad_wave, resolution,
                int(np.ceil(resolution * scale)), resolution, n_pad_wave, fix_npml))

    d_src = npml - k_src                                     # informational only: the ratio cancels it
    if d_src > 0:
        import warnings
        warnings.warn(
            "{}: the soft source k_src={} sits {} cell(s) inside the low-z CPML (npml={}), so the "
            "launch leaves the absorber {:.0%} down in amplitude. This is NOT rejected: R0/T0/"
            "R_flux/T_flux are two-run DFT ratios against a vacuum reference injected through the "
            "SAME absorber, so the launch attenuation cancels (measured: both narrow-band "
            "fixtures return R0+T0=1.0000 with the source 5-6 cells deep and the probes clear). "
            "What it does cost is SNR -- a broadband 1.2-1.8 um fixture with the source 5-8 cells "
            "deep scattered its band-edge bins by up to +-1.2 %. Raise n_pad_wave / resolution (or "
            "lower npml) if you need better than ~1 % there; audit D-3.".format(
                entry_point, k_src, d_src, npml, _pml_atten(d_src, npml)),
            RuntimeWarning, stacklevel=3)


def _check_band(entry_point, band, f_min, f_max):
    """audit D-3 (sub-mode): refuse an EMPTY well-excited band mask instead of returning silent zeros.

    With the source buried deep enough in the CPML the injected pulse never reaches the reference
    probe, so `np.abs(mL_inc) > 0.05 * max(...)` selects NOTHING (measured: band.sum() == 0 of 2617
    bins, no raise). Every downstream consumer then dies far from the cause with an opaque
    `ValueError: zero-size array to reduction operation minimum` on `result.R0[result.band].min()`.

    This is now the HARD backstop for a buried source: `_check_probe_placement` only warns about
    one (its launch attenuation cancels in the two-run ratio), so the terminal case -- no signal at
    all -- has to be caught here. Called from all six front ends including solve_fdtd_1d, whose Mur
    ABCs have no absorber; there an empty band means the source band and lambda_min/lambda_max
    disagree.
    """
    if not np.any(band):
        raise ValueError(
            "{}: the well-excited frequency band is EMPTY ({} rfft bins, none above the 5%-of-peak "
            "incident-amplitude threshold in [{:.4g}, {:.4g}] Hz) -- the source pulse never reached "
            "the reference probe. This is the deep-overlap mode of audit D-3: check that the source "
            "and probe planes clear the CPML (raise n_pad_wave / resolution, or lower npml), and "
            "that lambda_min_m/lambda_max_m bracket the source band.".format(
                entry_point, int(np.size(band)), f_min, f_max))


def _check_lateral_pads(entry_point, eps_grid, zc, pad, z_struct, n_super, n_sub):
    """audit D-6: `lateral_eps_inf` REPLACES the whole eps_inf grid, so the painter owns the pads.

    The 2-D comment used to say the pattern is "applied in the structure region"; it is not (and the
    3-D twin behaves the same way -- only fdtd_seam.make_structured_lateral, which paints the pads
    itself, documented the real contract). A caller who follows the old comment and returns
    `np.ones((nx, nz))` outside the structure LOSES the superstrate/substrate entirely, while

      * the incident reference run is still homogeneous n_super,
      * the CPML is still impedance-matched to n_super / n_sub per end,
      * T0 is still multiplied by the n_sub/n_super flux ratio,

    so R/T come back silently mis-normalized. Rather than let that pass, require the pads to carry
    the declared end media (exactly what make_structured_lateral does). The check is vacuous for the
    default vacuum end media only if the painter really did leave ones there -- which is the point:
    it also catches a painter that floods the WHOLE grid with the structure index."""
    eps = np.asarray(eps_grid, dtype=float)
    lo = zc < pad                                            # superstrate pad
    hi = zc >= pad + z_struct                                # substrate pad
    bad = []
    for mask, want, side, knob in ((lo, float(n_super) ** 2, "superstrate (low-z)", "n_super"),
                                   (hi, float(n_sub) ** 2, "substrate (high-z)", "n_sub")):
        if not np.any(mask):
            continue
        block = eps[:, mask]
        if not np.allclose(block, want, rtol=1e-9, atol=1e-12):
            bad.append("{} pad carries eps_inf in [{:.6g}, {:.6g}] but {}={:g} demands {:.6g}"
                       .format(side, float(block.min()), float(block.max()), knob,
                               float(n_super if knob == "n_super" else n_sub), want))
    if bad:
        raise ValueError(
            "{}: `lateral_eps_inf` REPLACES THE WHOLE (nx, nz) grid -- pads included -- so the "
            "painter must fill the z pads with the END-MEDIA permittivity, and this one did not: "
            "{}. The incident reference run, the per-end CPML match and T0's n_sub/n_super flux "
            "factor all still assume the declared end media, so R/T would come back silently "
            "mis-normalized (audit D-6). Fill zc < pad with n_super**2 and zc >= pad + z_struct "
            "with n_sub**2 inside your painter (optics.fdtd_seam.make_structured_lateral does this "
            "for you), or leave them at 1.0 for the default vacuum end media."
            .format(entry_point, "; ".join(bad)))


def _dispatch_2d_te(name, eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp=np,
                    lor=None, chi2=None, raman=None, gain=None, hot=None, hot_out=None):
    """Run ONE 2D-TE pass on the named backend and return the four probe x-lines as NumPy arrays, so the
    downstream FFT / R-T extraction stays backend-agnostic. 'numba' = the fused threaded CPU kernel;
    'jax' = the differentiable XLA scan; 'numpy'/'cupy' = the vectorized reference loop on the chosen
    array module (an explicit power-user `xp` is honored even for 'numpy', preserving the old xp=cupy API).
    `lor` = (C1,C2,C3) per-cell Lorentz ADE coefficients or None (no Lorentz pole). chi2/raman/gain
    (R15/R20) run on EVERY backend: the GPU kernels carry the same cell-local recurrences
    (numba-cuda in the cooperative kernel; cupy through the xp-parameterized reference loop),
    validated GPU==CPU in validation/fdtd_gpu_nonlinear.py. None keeps every backend
    byte-identical.

    `hot` (roadmap 2.1 per-cell hot-carrier two-temperature ADE) is carried by the NUMPY reference kernel
    ONLY -- the fused/GPU/differentiable kernels have no hot-carrier fast path yet (the Auger precedent), so
    a non-NumPy backend with hot != None raises a loud ValueError rather than silently ignoring the
    physics."""
    if hot is not None and name != "numpy":
        raise ValueError(
            "hot-carrier two-temperature dynamics run on the 'numpy' reference kernel only; backend "
            "'{}' has no hot-carrier fast path (extend the reference first, the Auger/nonlinear "
            "precedent). Use backend='numpy'.".format(name))
    (ke, be, ce), (kh, bh, ch) = cpml
    if name in ("numba", "numba-cuda"):
        has_lor = lor is not None
        z = np.zeros_like(eps_inf)
        C1, C2, C3 = (lor if has_lor else (z, z, z))
        chi2g = chi2 if chi2 is not None else z
        R1, R2, R3, chi3R = (raman if raman is not None else (z, z, z, z))
        G1, G2, G3 = (gain if gain is not None else (z, z, z))
        if name == "numba-cuda":
            return _te2d_cuda(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
                              nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor,
                              chi2g, chi2 is not None, R1, R2, R3, chi3R, raman is not None,
                              G1, G2, G3, gain is not None)
        return _te2d_numba(eps_inf, wp, gam, chi3, ke, be, ce, kh, bh, ch, dx, dz, dt,
                           nsteps, k_src, k_pL, k_pR, src, C1, C2, C3, has_lor,
                           chi2g, chi2 is not None, R1, R2, R3, chi3R, raman is not None,
                           G1, G2, G3, gain is not None)
    if name == "jax":
        out = run_2d_te_jax(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml,
                            lor, chi2=chi2, raman=raman, gain=gain)
        return tuple(np.asarray(v) for v in out)            # JAX arrays -> NumPy for the FFT/R-T stage
    if name == "cupy" and xp is np:
        import cupy as xp                                    # backend='cupy' auto-selects the device module
    a = tuple(xp.asarray(v) for v in (eps_inf, wp, gam, chi3))
    out = run_2d_te(*a, dx, dz, dt, nsteps, k_src, k_pL, k_pR, xp.asarray(src), cpml, xp, lor,
                    chi2=chi2, raman=raman, gain=gain, hot=hot, hot_out=hot_out)
    to_np = (lambda v: np.asarray(v.get()) if hasattr(v, "get") else np.asarray(v))
    return tuple(to_np(v) for v in out)


def solve_fdtd_2d(layers: List[FDTDLayer], *, period_x_m: float, nx: Optional[int] = None,
                  lateral_eps_inf: Optional[np.ndarray] = None,
                  lateral_wp: Optional[np.ndarray] = None, lateral_gam: Optional[np.ndarray] = None,
                  lambda_min_m: float, lambda_max_m: float, resolution: int = 40,
                  courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                  kerr: bool = False, source_amp: float = 1.0, npml: int = 12,
                  n_super: float = 1.0, n_sub: float = 1.0,
                  backend: str = "numpy", xp=np,
                  hot_out: Optional[dict] = None,
                  return_time_trace: bool = False) -> FDTD2DResult:
    """Broadband R(f)/T(f) of a periodic (period_x_m) 2D-TE unit cell at NORMAL incidence. `layers`
    is the through-stack (z) profile; supply `lateral_eps_inf` (a FULL (nx, nz) grid, or a callable
    building the (nx, nz) eps_inf -- shape doc corrected per audit 6.3) to make a laterally-structured
    grating, else the stack is laterally
    UNIFORM (and the result reduces to the 1D solver / TMM). `lateral_eps_inf` REPLACES THE WHOLE GRID,
    z PADS INCLUDED (same as solve_fdtd_3d), so the painter OWNS the super/substrate pads and must fill
    zc < pad with n_super**2 and zc >= pad + z_struct with n_sub**2 -- optics.fdtd_seam
    .make_structured_lateral does it for you, and a painter that does not is now REFUSED rather than
    silently mis-normalized (audit D-6). Returns both the 0-order (specular, x-mean)
    and the total-flux (all-diffraction-order) R/T.

    n_super / n_sub (default 1 = vacuum) are the lossless semi-infinite superstrate / substrate indices
    (metasurface-on-glass etc.): the z-pad regions are filled with n_super^2 / n_sub^2, the CPML is
    impedance-matched per end, and the incident reference is a homogeneous-superstrate run so R/T are
    correctly normalized (T carries the n_sub/n_super flux ratio). Reduces byte-identically to vacuum
    at n_super=n_sub=1.

    backend selects the compute kernel (see available_backends()): 'auto' (default-fastest CPU present),
    'numpy' (reference), 'numba' (fused threaded CPU -- fastest for unit cells), 'cupy' (NVIDIA GPU),
    'jax' (differentiable XLA), or the 'cpu'/'gpu' aliases. Every backend AGREES WITH THE NUMPY
    REFERENCE TO THE FLOAT64 ROUNDING FLOOR on R/T, NOT bit-for-bit (audit D-7): the numba/CUDA kernels
    factor the E-update constant differently (`e0dt*eps_inf` vs `EPS0*eps_eff/dt`) and carry
    fastmath=True, which licenses reassociation. What is actually GATED: max|dR|,max|dT| < 1e-9 on a
    non-dispersive lossless slab (validation/fdtd_2d_reduces.py GATE D) and < 1e-12 with the R15/R20
    chi2/Raman/gain nonlinearities active (validation/fdtd_nonlinear_backends.py GATES A/B/E,
    validation/fdtd_gpu_nonlinear.py on real hardware); 3-D backend parity is
    validation/fdtd_3d_reduces.py. xp is an advanced override for a custom array module."""
    if abs(complex(n_super).imag) > 1e-9 or abs(complex(n_sub).imag) > 1e-9:   # mirror the FEM guard
        raise NotImplementedError("solve_fdtd_2d: R/T and the energy budget are defined only for LOSSLESS "
                                  "end media (Im(n)=0); got n_super={}, n_sub={} (use the FEM/TMM solver "
                                  "for an absorbing incidence/exit medium).".format(n_super, n_sub))
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)      # sample the band (a Lorentz peak may be in-band)

    def _n_band_max(L):
        return max(abs(np.sqrt(L.eps_at(w))) for w in w_band)
    n_max = max(1.0, n_super, n_sub, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    if nx is None:
        nx = max(4, int(round(period_x_m / dz)))
    dx = period_x_m / nx
    # 2D CFL: dt <= courant / (c sqrt(1/dx^2 + 1/dz^2))
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1

    # z-profile, replicated over nx columns (laterally uniform unless lateral_eps_inf given)
    eps_inf = np.ones((nx, nz)); wp = np.zeros((nx, nz)); gam = np.zeros((nx, nz)); chi3 = np.zeros((nx, nz))
    lw0 = np.zeros((nx, nz)); lgam = np.zeros((nx, nz)); ldeps = np.zeros((nx, nz))  # Lorentz pole per cell
    chi2g = np.zeros((nx, nz))                                                       # R15 SHG chi2 [m/V]
    chi3R = np.zeros((nx, nz)); rw = np.zeros((nx, nz)); rgam = np.zeros((nx, nz))   # R15 Raman pole
    gw = np.zeros((nx, nz)); gdw = np.zeros((nx, nz)); gkdn = np.zeros((nx, nz))     # R20 gain line
    zc = (np.arange(nz) + 0.5) * dz
    # fill the semi-infinite super/substrate pads with the end-media permittivity (so the incident wave is
    # truly in n_super and the structure sees the n_sub backing); vacuum (n=1) leaves this as ones
    eps_inf[:, zc < pad] = n_super ** 2
    eps_inf[:, zc >= pad + z_struct] = n_sub ** 2
    # roadmap 2.1 hot-carrier per-cell scaffolding (opt-in). All None/-1 unless a layer carries a
    # HotCarrierParams; when every hot_carrier is None the bundle stays None so the run is byte-identical.
    _hot_layers = [getattr(L, "hot_carrier", None) for L in layers]
    _have_hot = any(h is not None for h in _hot_layers)
    if _have_hot:
        hc_mat_idx = np.full((nx, nz), -1, dtype=np.int64)   # per-cell material id, -1 = not hot
        hc_G = np.zeros((nx, nz)); hc_Tl = np.zeros((nx, nz))
        hc_alpha = np.zeros((nx, nz)); hc_Te0 = np.full((nx, nz), T_REF, dtype=float)
        hc_tables = []                                       # per-material (Te_grid,U_grid,wp_ratio,gam_ratio)
        hc_seen = {}                                         # id(HotCarrierParams) -> material index (dedupe)
        hc_n_update = None
        # roadmap 5.7 finite-lattice scaffolding: c_l = inf (Tl pinned) / g_sub = 0 everywhere until a
        # HotCarrierParams sets c_l_j_m3_k. _have_lat stays False if every layer is fixed-bath (c_l None),
        # so the bundle carries c_l=None and the kernel takes the byte-identical fixed-bath path.
        hc_c_l = np.full((nx, nz), np.inf); hc_g_sub = np.zeros((nx, nz))
        _have_lat = False
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, m] = L.eps_inf
        wp[:, m] = L.drude_wp_rad_s
        gam[:, m] = L.drude_gamma_rad_s
        lw0[:, m] = L.lorentz_w0_rad_s
        lgam[:, m] = L.lorentz_gamma_rad_s
        ldeps[:, m] = L.lorentz_delta_eps
        if kerr:
            chi3[:, m] = L.chi3_m2_V2
        chi2g[:, m] = L.chi2_m_V
        chi3R[:, m] = L.raman_chi3_m2_V2
        rw[:, m] = L.raman_w_rad_s
        rgam[:, m] = L.raman_gamma_rad_s
        gw[:, m] = L.gain_w_rad_s
        gdw[:, m] = L.gain_dw_rad_s
        gkdn[:, m] = L.gain_kappa_C2_kg * L.gain_dN_m3
        hc = getattr(L, "hot_carrier", None)
        if _have_hot and hc is not None:
            from dynameta.optics.hot_carrier import build_hot_carrier_tables  # lazy: keep fdtd_nd import-light
            mi = hc_seen.get(id(hc))
            if mi is None:
                mi = len(hc_tables)
                hc_seen[id(hc)] = mi
                hc_tables.append(build_hot_carrier_tables(hc))
            hc_mat_idx[:, m] = mi
            hc_G[:, m] = float(hc.ttm.G_e_l)
            hc_Tl[:, m] = float(hc.T_l_K)
            hc_alpha[:, m] = float(hc.ttm.alpha_abs)
            hc_Te0[:, m] = float(hc.T_e0_K)
            hc_n_update = int(hc.n_update) if hc_n_update is None else min(hc_n_update, int(hc.n_update))
            if hc.c_l_j_m3_k is not None:                     # roadmap 5.7 finite lattice on this layer
                _have_lat = True
                hc_c_l[:, m] = float(hc.c_l_j_m3_k)
                hc_g_sub[:, m] = float(hc.g_sub_w_m3_k)
        z += L.thickness_m
    if lateral_eps_inf is not None:
        # a laterally-structured grating: the pattern REPLACES THE WHOLE (nx, nz) eps_inf grid, PADS
        # INCLUDED -- not just the structure band (audit D-6: the comment here used to say "applied in
        # the structure region", which is what a caller reads before writing the painter, and only
        # fdtd_seam.make_structured_lateral knew the real contract). Same semantics as solve_fdtd_3d.
        # The painter therefore OWNS the n_super/n_sub pads and must fill them with n_super**2 /
        # n_sub**2 -- the two lines above that did so are discarded here -- while the incident
        # reference run, the per-end CPML match and T0's n_sub/n_super flux factor all still assume
        # the declared end media. `_check_lateral_pads` turns that from a silent mis-normalization
        # into a raise; make_structured_lateral already paints the pads for you.
        lat = lateral_eps_inf(nx, nz, zc, pad, z_struct) if callable(lateral_eps_inf) else np.asarray(lateral_eps_inf)
        eps_inf = np.asarray(lat, dtype=float)
        _check_lateral_pads("solve_fdtd_2d", eps_inf, zc, pad, z_struct, n_super, n_sub)
        # GRID-SIZING GUARD: dz was derived from `layers` (+ end media) BEFORE this override. If the
        # lateral pattern's peak index exceeds the sizing index, dz is too coarse and R/T are silently
        # under-resolved. Raise rather than mis-solve -- size `layers` eps_inf to the pattern's max index
        # (the make_structured_lateral seam already does this) so n_max/dz are derived correctly.
        _n_lat = float(np.sqrt(max(1.0, float(np.max(np.real(eps_inf))))))
        if _n_lat > n_max * (1.0 + 1e-9):
            raise NotImplementedError(
                "solve_fdtd_2d: lateral pattern peak index {:.3f} exceeds the grid-sizing index {:.3f} "
                "from `layers` (+ end media) -- dz is under-resolved by {:.0%}. Size the `layers` "
                "eps_inf to the lateral pattern max so n_max/dz are derived from it.".format(
                    _n_lat, n_max, _n_lat / n_max - 1.0))
    # PER-CELL LOSSY/graded eps (R4): a Drude (wp,gam) grid alongside eps_inf lets a slow drive (gate E,
    # T, PCM fraction) paint a graded ABSORBING eps the eps_inf-only lateral seam cannot carry. Each is a
    # callable(nx,nz,zc,pad,z_struct)->array or an (nx,nz) array (zero in the pads). Default None -> the
    # wp/gam grids stay zeros -> byte-identical to the dielectric path.
    if lateral_wp is not None:
        wp = np.asarray(lateral_wp(nx, nz, zc, pad, z_struct) if callable(lateral_wp) else lateral_wp,
                        dtype=float)
    if lateral_gam is not None:
        gam = np.asarray(lateral_gam(nx, nz, zc, pad, z_struct) if callable(lateral_gam) else lateral_gam,
                         dtype=float)

    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    _check_probe_placement("solve_fdtd_2d", k_src, k_pL, k_pR, nz, npml, pad, dz,
                           n_pad_wave, resolution)          # audit D-3

    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    t_ring = _ring_time_s(layers)                            # audit C3-6: pole memory
    if t_ring > 200 * tau:
        import warnings
        warnings.warn("FDTD window extended {:.1f}x for a narrow Lorentz/gain line "
                      "(material memory {:.2e} s > the 200*tau source window; audit "
                      "C3-6)".format(1.0 + t_ring / (200 * tau), t_ring),
                      RuntimeWarning, stacklevel=2)
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau + t_ring) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    # Lorentz ADE coefficients (central difference): PL^{n+1} = C1 PL^n + C2 PL^{n-1} + C3 E^n, where the
    # pole eps += d_eps w0^2/(w0^2 - w^2 - i gl w). lor is applied to the STRUCTURE run only (the reference
    # is the bare superstrate). With d_eps=0 everywhere lor=None -> the path is byte-identical to before.
    lor = None
    if np.any(ldeps != 0.0):
        if float(np.max(lw0)) * dt > 1.0:                    # audit S2-7
            raise ValueError("Lorentz pole under-resolved: lorentz_w0_rad_s*dt = {:.2f} > 1 -- "
                             "the central-difference ADE would diverge; refine the grid or move "
                             "the pole".format(float(np.max(lw0)) * dt))
        den = 1.0 + lgam * dt / 2.0
        C1 = (2.0 - lw0 ** 2 * dt ** 2) / den
        C2 = (lgam * dt / 2.0 - 1.0) / den
        C3 = (EPS0 * ldeps * lw0 ** 2 * dt ** 2) / den
        lor = (C1, C2, C3)

    # R15 chi2 / Raman flags + Raman vibrational-ADE coefficients (same central-difference template
    # as the Lorentz pole; Q'' + gam_R Q' + W_R^2 Q = W_R^2 E^2 -> Q^{n+1} = R1 Q + R2 Q^{n-1} + R3 E^2;
    # the polarization is P_R = eps0 chi3R E Q, formed in the kernel). All-zero grids -> None -> the
    # kernels take the pre-R15 code path byte-identically.
    chi2_arrs = chi2g if np.any(chi2g != 0.0) else None
    raman_arrs = None
    if np.any(chi3R != 0.0):
        if np.any((chi3R != 0.0) & (rw <= 0.0)):
            raise ValueError("Raman chi3 needs raman_w_rad_s > 0 on every Raman-active layer")
        if float(np.max(rw)) * dt > 1.0:                     # central-diff resonance stability margin
            raise ValueError("Raman resonance under-resolved: raman_w_rad_s*dt = {:.2f} > 1 -- raise "
                             "`resolution` (or lower the Raman frequency)".format(float(np.max(rw)) * dt))
        den_r = 1.0 + rgam * dt / 2.0
        raman_arrs = ((2.0 - rw ** 2 * dt ** 2) / den_r, (rgam * dt / 2.0 - 1.0) / den_r,
                      (rw ** 2 * dt ** 2) / den_r, chi3R)
    gain_arrs = None
    if np.any(gkdn != 0.0):
        if np.any((gkdn != 0.0) & ((gw <= 0.0) | (gdw <= 0.0))):
            raise ValueError("gain line needs gain_w_rad_s > 0 and gain_dw_rad_s > 0 on every "
                             "gain-active layer")
        if float(np.max(gw)) * dt > 1.0:                     # audit S2-7
            raise ValueError("gain line under-resolved: gain_w_rad_s*dt = {:.2f} > 1 -- the "
                             "central-difference ADE would diverge; refine the grid or move the "
                             "line".format(float(np.max(gw)) * dt))
        den_g = 1.0 + gdw * dt / 2.0
        gain_arrs = ((2.0 - gw ** 2 * dt ** 2) / den_g, (gdw * dt / 2.0 - 1.0) / den_g,
                     (-gkdn * dt ** 2) / den_g)
    cpml_struct = cpml_z(nz, dz, dt, npml, n_super, n_sub)   # PML matched to super (low z) + sub (high z)
    cpml_ref = cpml_z(nz, dz, dt, npml, n_super, n_super)    # homogeneous-superstrate reference -> super both ends
    name = resolve_backend(backend)                          # 'auto'/'cpu'/'gpu'/explicit -> concrete backend
    one = np.ones((nx, nz)); zero = np.zeros((nx, nz))

    def run(ei, w, g_, c3, cpml, lor=None, chi2=None, raman=None, gain=None, hot=None, hot_out=None):
        return _dispatch_2d_te(name, ei, w, g_, c3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, xp,
                               lor, chi2, raman, gain, hot=hot, hot_out=hot_out)

    # assemble the hot-carrier bundle (roadmap 2.1) for the STRUCTURE run only -- the incident reference is
    # the bare superstrate (no absorber, so no heating) and stays byte-identical / cache-shared. wp/gam here
    # already carry any lateral (wp,gam) override, so the kernel's cold anchors match the painted structure.
    hot_bundle = None
    if _have_hot:
        hot_bundle = {"mask": (hc_mat_idx >= 0).astype(float), "mat_idx": hc_mat_idx,
                      "tables": hc_tables, "G": hc_G, "Tl": hc_Tl, "alpha": hc_alpha,
                      "Te0": hc_Te0, "n_update": (hc_n_update if hc_n_update else 1),
                      "c_l": (hc_c_l if _have_lat else None),      # roadmap 5.7; None -> fixed-bath tier
                      "g_sub": (hc_g_sub if _have_lat else None)}

    # reference = homogeneous superstrate (no structure, no substrate) so the probe sees the pure incident
    # wave in n_super and the reflection subtraction is exact (same incident medium as the structure run).
    # Structure-independent, so it is memoized on its exact determinants (audit 6.2 perf); a custom xp
    # bypasses the cache (the key cannot pin an arbitrary array module).
    if xp is np:
        _key = ("2d", name, nx, nz, float(dx), float(dz), float(dt), int(nsteps), int(k_src),
                int(k_pL), int(k_pR), int(npml), complex(n_super), src.tobytes())
        eyL_i, hxL_i, eyR_i, hxR_i = _ref_cache_call(
            _key, lambda: run(n_super ** 2 * one, zero, zero, zero, cpml_ref))
    else:
        eyL_i, hxL_i, eyR_i, hxR_i = run(n_super ** 2 * one, zero, zero, zero, cpml_ref)
    eyL_t, hxL_t, eyR_t, hxR_t = run(eps_inf, wp, gam, chi3, cpml_struct, lor,
                                     chi2_arrs, raman_arrs, gain_arrs,
                                     hot=hot_bundle, hot_out=hot_out)  # structure run

    f = np.fft.rfftfreq(nsteps, dt)
    # ---- 0-order (specular) R/T from the x-MEAN field (== the 1D two-run method) ----
    mL_inc = np.fft.rfft(eyL_i.mean(axis=1)); mR_inc = np.fft.rfft(eyR_i.mean(axis=1))
    mRefl = np.fft.rfft((eyL_t - eyL_i).mean(axis=1)); mTrans = np.fft.rfft(eyR_t.mean(axis=1))
    k0 = 2.0 * np.pi * f / C_LIGHT
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(mRefl / mL_inc) ** 2
        # power transmittance carries the n_sub/n_super impedance (flux) ratio: the incident reference is
        # measured in n_super, the transmitted field in n_sub (Snell power continuity)
        T0 = np.abs(mTrans / mR_inc) ** 2 * (n_sub / n_super)
        # COMPLEX 0-order coeffs. np.fft.rfft yields exp(+i w t) phasors, but the library convention is
        # exp(-i w t), so conjugate to get the physical complex amplitudes; then de-embed the probe<->face
        # propagation phase. The superstrate phase velocity is c/n_super, so r0c (referenced to the front
        # face z=pad, probe at k_pL) carries n_super in k. t0c (audit C3-4): the incident reference
        # travels n_super the WHOLE way to the right probe while the transmitted leg is n_sub past the
        # back face, so the interface-referenced t carries n_sub*z_struct PLUS the (n_super-n_sub)
        # mismatch over the face->probe distance D = k_pR*dz - pad (the old bare exp(1j*k0*z_struct)
        # was vacuum-only: ~100 deg phase error on glass, frequency-dependent; |t| untouched).
        r0c = np.conj(mRefl / mL_inc) * np.exp(-2j * n_super * k0 * (pad - k_pL * dz))
        t0c = np.conj(mTrans / mR_inc) * np.exp(1j * k0 * (n_sub * z_struct
                                                           + (n_super - n_sub) * (k_pR * dz - pad)))
    # ---- TOTAL R/T from the Poynting flux (all diffraction orders) ----
    P_inc = _flux(eyL_i, hxL_i, dt)                          # dt: half-timestep H de-stagger (D-2)
    P_refl = _flux(eyL_t - eyL_i, hxL_t - hxL_i, dt)
    P_trans = _flux(eyR_t, hxR_t, dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_flux = np.abs(P_refl) / np.abs(P_inc)
        T_flux = np.abs(P_trans) / np.abs(P_inc)
    band = (f >= f_min) & (f <= f_max) & (np.abs(mL_inc) > 0.05 * np.max(np.abs(mL_inc)))
    _check_band("solve_fdtd_2d", band, f_min, f_max)         # audit D-3 sub-mode
    # OPT-IN (roadmap 3.1): expose the exit/entry-plane E_y + H_x x-lines already recorded above, as
    # copies. Purely additive -- R0/T0/R_flux/T_flux/band/r0/t0 are computed identically whether or not
    # the trace is attached, so return_time_trace=False (default) is byte-identical to the legacy path.
    # optics.harmonics reads the raw transmitted series to integrate the 2w/3w bands (the SHG/THG content
    # the ~0 incident reference makes invisible in the normalized R/T).
    time_trace = None
    if return_time_trace:
        time_trace = {
            "dt": dt,
            "t": tgrid.copy(),
            "transmitted": eyR_t.copy(), "transmitted_hx": hxR_t.copy(),
            "reflected": (eyL_t - eyL_i).copy(), "reflected_hx": (hxL_t - hxL_i).copy(),
            "incident_left": eyL_i.copy(), "incident_left_hx": hxL_i.copy(),
            "incident_right": eyR_i.copy(), "incident_right_hx": hxR_i.copy(),
            "period_x_m": float(period_x_m), "dx": float(dx), "nx": int(nx),
        }
    return FDTD2DResult(freqs_Hz=f, R0=R0, T0=T0, R_flux=R_flux, T_flux=T_flux, band=band, r0=r0c, t0=t0c,
                        time_trace=time_trace)




def solve_fdtd_2d_oblique(layers: List[FDTDLayer], *, period_x_m: float, angle_deg: float,
                          lambda_min_m: float, lambda_max_m: float, resolution: int = 40,
                          courant: float = 0.5, n_pad_wave: float = 6.0, settle: float = 12.0,
                          source_amp: float = 1.0, npml: int = 12, nx: int = 8,
                          backend: str = "numpy", pol: str = "s") -> FDTD2DObliqueResult:
    """Broadband reflectance/transmittance of a laterally-uniform stack at OBLIQUE incidence, via the
    complex-envelope Bloch method with a FIXED transverse wavevector k_par = (2 pi / lambda_c)
    sin(angle_deg) (angle_deg the physical angle at the band centre). pol='s' = TE (Ey,Hx,Hz); pol='p' =
    TM (Hy,Ex,Ez) -- the p-pol R/T come from the tangential-Ex up/down ratio. Because k_par is fixed, the
    physical angle varies with frequency: theta(f) = asin(k_par c/(2 pi f)); the result carries theta_deg(f)
    and the band mask excludes frequencies below the light line (k_par > w/c, evanescent). Vacuum ends.
    angle_deg=0 reduces to the normal-incidence solver. backend selects the TE kernel (numpy/numba); TM is
    the NumPy reference."""
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' (TE) or 'p' (TM); got {!r}".format(pol))
    if any(L.lorentz_delta_eps != 0.0 for L in layers):     # the oblique kernel carries Drude only
        raise NotImplementedError("solve_fdtd_2d_oblique supports Drude dispersion only (no Lorentz pole "
                                  "yet); use solve_fdtd_2d at normal incidence for a Lorentz material.")
    # audit C5-7: the oblique kernel also carries NO chi3/chi2/Raman/gain ADEs -- these terms
    # used to be silently DROPPED (an amplifying/SHG/Raman stack at 20 deg returned R0/T0
    # bit-identical to the passive layer), while the 1-D entry point raises for the same set
    _dropped = [t for t in ("chi3_m2_V2", "chi2_m_V", "raman_chi3_m2_V2", "gain_dN_m3")
                if any(getattr(L, t, 0.0) != 0.0 for L in layers)]
    if _dropped:
        raise NotImplementedError(
            "solve_fdtd_2d_oblique: the oblique kernel carries no {} terms -- they would be "
            "silently ignored (audit C5-7); use the normal-incidence solver or split the "
            "problem.".format("/".join(_dropped)))
    # audit D-1: hot_carrier cannot join `_dropped` (its sentinel is None, not 0.0, so the
    # `!= 0.0` test would fire on every passive layer); guarded by the shared spec helper.
    hot_carrier_guard("solve_fdtd_2d_oblique", layers)
    f_min, f_max = C_LIGHT / lambda_max_m, C_LIGHT / lambda_min_m
    f_c = 0.5 * (f_min + f_max)
    w_band = 2.0 * np.pi * np.linspace(f_min, f_max, 9)

    def _n_band_max(L):
        return max(abs(np.sqrt(L.eps_at(w))) for w in w_band)
    n_max = max(1.0, max(_n_band_max(L) for L in layers))
    dz = lambda_min_m / (resolution * n_max)
    dx = period_x_m / nx
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    kx = (2.0 * np.pi * f_c / C_LIGHT) * np.sin(np.radians(angle_deg))   # fixed transverse wavevector

    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1
    eps_inf = np.ones((nx, nz)); wp = np.zeros((nx, nz)); gam = np.zeros((nx, nz))
    zc = (np.arange(nz) + 0.5) * dz
    z = pad
    for L in layers:
        m = (zc >= z) & (zc < z + L.thickness_m)
        eps_inf[:, m] = L.eps_inf; wp[:, m] = L.drude_wp_rad_s; gam[:, m] = L.drude_gamma_rad_s
        z += L.thickness_m

    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    _check_probe_placement("solve_fdtd_2d_oblique", k_src, k_pL, k_pR, nz, npml, pad, dz,
                           n_pad_wave, resolution)          # audit D-3
    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    t_ring = _ring_time_s(layers)                            # audit C3-6: pole memory
    if t_ring > 200 * tau:
        import warnings
        warnings.warn("FDTD window extended {:.1f}x for a narrow Lorentz/gain line "
                      "(material memory {:.2e} s > the 200*tau source window; audit "
                      "C3-6)".format(1.0 + t_ring / (200 * tau), t_ring),
                      RuntimeWarning, stacklevel=2)
    nsteps = int(round((2.0 * t0 + (Lz / C_LIGHT) * 4.0 + 200 * tau + t_ring) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))

    cpml = cpml_z(nz, dz, dt, npml)
    one = np.ones((nx, nz)); zero = np.zeros((nx, nz))
    # 'numba' = the fused threaded complex-envelope kernel; 'auto'/'cpu' pick it when present; everything else
    # falls back to the vectorized NumPy reference (the oblique path is normal-incidence-free of jax/cupy).
    rb = resolve_backend(backend)
    # _run_oblique carries fused numba + differentiable jax kernels for BOTH s-pol (TE) and p-pol (TM);
    # pick the requested fast/diff backend when available, else the NumPy reference.
    if rb == "jax" and have_jax():
        name = "jax"                                         # differentiable oblique scan (s + p)
    elif rb == "numba" and HAVE_NUMBA:
        name = "numba"                                       # fused JIT oblique kernel (s + p)
    else:
        name = "numpy"
    eyL_i, hxL_i, eyR_i, hxR_i = _run_oblique(name, one, zero, zero, dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                              src, cpml, kx, pol)
    eyL_t, hxL_t, eyR_t, hxR_t = _run_oblique(name, eps_inf, wp, gam, dx, dz, dt, nsteps, k_src, k_pL, k_pR,
                                              src, cpml, kx, pol)
    # complex envelope -> full FFT; take the positive-frequency half (the forward exp(-iwt) response).
    # rfftfreq gives the monotonic positive-frequency axis matching fft(...)[:nf] within the band.
    nf = nsteps // 2 + 1
    f = np.fft.rfftfreq(nsteps, dt)
    mean = (lambda a: np.fft.fft(a.mean(axis=1))[:nf])
    inc_L = mean(eyL_i); inc_R = mean(eyR_i)
    refl = mean(eyL_t - eyL_i); trans = mean(eyR_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = np.abs(refl / inc_L) ** 2
        T0 = np.abs(trans / inc_R) ** 2
    sin_t = np.divide(kx * C_LIGHT, 2.0 * np.pi * np.maximum(f, 1e-30))   # sin theta(f) = k_par c / w
    theta = np.degrees(np.arcsin(np.clip(sin_t, -1.0, 1.0)))
    # audit C3-5: the old sin_t < 0.999 mask trusted points up to theta ~ 87 deg, where
    # the z-CPML's grazing round-trip echo reaches 0.1-0.5 FIELD (the absorber sees the
    # z-wavevector shrink as cos theta): the validation geometry re-run at 76 deg carried
    # band=True points with |R0 - TMM| = 0.39 and R0+T0-1 up to +0.38. Trust only
    # sin_t < 0.95 (theta < ~72 deg -- the measured error onset for the shipped npml=12);
    # warn when the mask removes otherwise-excited in-band points so the truncation is
    # visible rather than silent.
    _excited = (f >= f_min) & (f <= f_max) & (np.abs(inc_L) > 0.05 * np.max(np.abs(inc_L)))
    # audit D-3 sub-mode: gate the EXCITATION mask only -- the grazing (sin_t) cut below may
    # legitimately empty the trusted band at a large angle, and it already warns when it does.
    _check_band("solve_fdtd_2d_oblique", _excited, f_min, f_max)
    band = _excited & (sin_t < 0.95)
    _cut = _excited & (sin_t >= 0.95)
    if np.any(_cut):
        import warnings
        warnings.warn(
            "solve_fdtd_2d_oblique: {} excited in-band points at theta(f) >= 71.8 deg were "
            "EXCLUDED from the trusted band -- the grazing-incidence CPML echo corrupts R0/T0 "
            "there (audit C3-5); narrow the band, lower angle_deg, or strengthen npml."
            .format(int(np.sum(_cut))), RuntimeWarning, stacklevel=2)
    return FDTD2DObliqueResult(freqs_Hz=f, theta_deg=theta, R0=R0, T0=T0, band=band)


# =====================================================================================================
# 3D: full-vector Yee engine for a 2D-periodic (x AND y) unit cell at normal incidence.
# The 2D-TE engine above is the (d/dy = 0, {Ey,Hx,Hz}) reduction of this; this carries all six field
# components so a genuinely 2D-periodic structure (pillars/holes/crosses) couples into every order.
# =====================================================================================================
