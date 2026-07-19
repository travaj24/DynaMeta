"""
Spectral analysis helpers for reflection / transmission sweeps.

Generic post-processing utilities for the (wavelength, |r|^2) spectra that
Stage 3 produces -- not tied to any particular design.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.constants import Q_E as _Q_E   # elementary charge, C


from dynameta.core.numerics import trapz as _trapz   # shared (np.trapz removed in NumPy 2.x)


def gate_cv(carrier_fields: Sequence, region: str, *, voltage_key: str,
            density_field: str = ELECTRON_DENSITY):
    """DC gate charge Q(Vg) and differential capacitance C = dQ/dVg from a VOLTAGE SWEEP
    of CarrierFields (the per-bias fields run_pipeline / a Sweep already produces -- NO new
    solver). For an n-type accumulation gate the mirror gate charge per unit cell area is
    Q(Vg) = q * INT (n(z) - n_bg) dz (the accumulated excess electrons), laterally averaged;
    C(Vg) = dQ/dVg is the small-signal gate capacitance and is the first dynamic-adjacent
    figure of merit (with an access/sheet resistance it sets the RC modulation bandwidth).

    Args:
      carrier_fields : iterable of CarrierField at different gate biases (same device).
      region         : the semiconductor region key.
      voltage_key    : the electrode-name key into CarrierField.voltages for the gate bias.

    Returns (Vg, Q, Vmid, C): Vg (V, sorted) and Q (C/m^2) per bias; Vmid (V) and
    C (F/m^2) on the midpoints (central differences need >= 2 biases).

    Note: Q is the EXCESS sheet charge over the flat doping n_bg. For a
    Schrodinger-Poisson carrier the hard-wall dead layer makes n < n_bg near the body
    contact at every bias, so the absolute Q(0) is slightly negative and only the
    DIFFERENTIAL C = dQ/dVg (in which that constant offset cancels) is physically
    meaningful for SP fields; a classical/DD carrier gives Q(0) ~ 0.
    """
    Vg: List[float] = []
    Q: List[float] = []
    for cf in carrier_fields:
        reg = cf.regions[region]
        if reg.grid_fields is None or density_field not in reg.grid_fields:
            raise ValueError("region '{}' has no gridded '{}'".format(region, density_field))
        n = np.asarray(reg.grid_fields[density_field], dtype=np.float64)
        n_bg = float(cf.n_bg_by_region[region])
        if n.ndim == 3:                                   # (Nx, Ny, Nz) -> n(z)
            xc = np.asarray(reg.grid_axes_m["x"], dtype=np.float64)
            yc = np.asarray(reg.grid_axes_m["y"], dtype=np.float64)
            zc = np.asarray(reg.grid_axes_m["z"], dtype=np.float64)
            if n.shape != (xc.size, yc.size, zc.size):    # AN-4: reject a transposed/mis-laid grid
                raise ValueError("gate_cv: 3D density shape {} != (len(x),len(y),len(z))={}; "
                                 "the grid is transposed or mis-laid".format(
                                     n.shape, (xc.size, yc.size, zc.size)))
            prof = n.mean(axis=(0, 1))
        elif n.ndim == 2:                                 # (Nx, Nv); 2D vertical axis = 'y'
            xc = np.asarray(reg.grid_axes_m["x"], dtype=np.float64)
            zc = np.asarray(reg.grid_axes_m["y"], dtype=np.float64)
            if n.shape != (xc.size, zc.size):             # AN-4: reject a transposed/mis-laid grid
                raise ValueError("gate_cv: 2D density shape {} != (len(x),len(y))={}; the grid "
                                 "is transposed or mis-laid".format(n.shape, (xc.size, zc.size)))
            prof = n.mean(axis=0)
        else:
            raise ValueError("density grid must be 2D or 3D")
        Q.append(_Q_E * _trapz(prof - n_bg, zc))          # C/m^2 (excess electron sheet charge)
        Vg.append(float(cf.voltages[voltage_key]))
    Vg_a = np.asarray(Vg, dtype=np.float64)
    Q_a = np.asarray(Q, dtype=np.float64)
    order = np.argsort(Vg_a)
    Vg_a, Q_a = Vg_a[order], Q_a[order]
    if Vg_a.size < 2:
        return Vg_a, Q_a, np.array([]), np.array([])
    if np.any(np.diff(Vg_a) == 0.0):                  # coincident biases -> dQ/dVg = 0/0 = NaN
        dup = np.unique(Vg_a[:-1][np.diff(Vg_a) == 0.0]).tolist()
        raise ValueError("gate_cv: duplicate gate-bias point(s) {} V -> dQ/dVg undefined; "
                         "supply one CarrierField per distinct voltage".format(dup))
    Vmid = 0.5 * (Vg_a[1:] + Vg_a[:-1])
    C = np.diff(Q_a) / np.diff(Vg_a)
    return Vg_a, Q_a, Vmid, C


def sheet_resistance_ohm_sq(n_m3, mobility_m2Vs, thickness_m):
    """Sheet resistance [Ohm/sq] of a conductive layer: rho_s = 1/(q n mu t). A non-conducting layer
    (zero carriers / mobility / thickness) returns inf rather than dividing by zero (so an undoped or
    frozen-carrier region does not propagate a silent inf/-inf into the RC-bandwidth / FOM downstream)."""
    sigma = _Q_E * np.asarray(n_m3, dtype=np.float64) * float(mobility_m2Vs)   # S/m
    denom = sigma * float(thickness_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0.0, 1.0 / np.where(denom > 0.0, denom, 1.0), np.inf)


def lumped_rc_bandwidth(C_F_per_m2, sheet_resistance_ohm_sq, *,
                        path_length_m, pad_width_m, cell_area_m2):
    """Intrinsic single-cell electrical RC 3-dB bandwidth of a gated modulator cell
    (ported from the Metasurface_Modulator Stage-4 lumped model).

    The gate charges through the in-plane ACCESS resistance
    R_access = rho_sheet * (path_length / pad_width) into the per-cell capacitance
    C_cell = C_per_area * cell_area, so f_3dB = 1 / (2 pi R_access C_cell). Pair it with
    gate_cv()'s differential capacitance (which may be an array over bias) for f_3dB(Vg).
    The access path geometry is a MODELING CHOICE -- sweep a few plausible (path, pad).

    Returns (R_access_ohm, C_cell_F, f_3dB_Hz), each broadcasting over C_F_per_m2.
    """
    C_per = np.asarray(C_F_per_m2, dtype=np.float64)
    if np.any(C_per <= 0.0):
        warnings.warn("lumped_rc_bandwidth: non-positive capacitance encountered "
                      "(depletion branch?); f_3dB is NaN there. Pass the accumulation-"
                      "branch (C>0) capacitance for a physical bandwidth.", stacklevel=2)
    R = float(sheet_resistance_ohm_sq) * float(path_length_m) / float(pad_width_m)
    C_cell = C_per * float(cell_area_m2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f3db = 1.0 / (2.0 * np.pi * R * C_cell)
    if np.ndim(f3db) == 0:                            # preserve scalar-in -> scalar-out
        f3db = float(f3db) if C_cell > 0.0 else float("nan")
    else:
        f3db = np.where(C_cell > 0.0, f3db, np.nan)
    return R, C_cell, f3db


def switching_energy_per_area(C_F_per_m2, voltage_swing_V):
    """Dynamic switching energy per unit area of a capacitive gate: E = 0.5 C V^2 [J/m^2]
    (C the gate areal capacitance, V the full drive swing). x cell_area -> J/event."""
    return 0.5 * np.asarray(C_F_per_m2, dtype=np.float64) * float(voltage_swing_V) ** 2


def modulator_figure_of_merit(*, optical_contrast, contrast_lambda_nm, gate_C_per_area_F_m2,
                              voltage_swing_V, sheet_resistance_ohm_sq, path_length_m, pad_width_m,
                              cell_area_m2):
    """Assemble a gated electro-optic modulator's DEVICE SPEC SHEET by fusing its two validated halves:
    the OPTICAL modulation contrast (e.g. peak |R_on - R_off| between bias states, from the FDTD sweep)
    and the ELECTRICAL gate (areal capacitance + the in-plane access geometry). Returns the RC switching
    bandwidth f_3dB = 1/(2 pi R_access C_cell), the per-event switching energy E = 0.5 C V^2 * cell_area,
    and a contrast-per-energy figure of merit (optical modulation delivered per fJ switched). All SI in,
    convenience units (GHz, fJ) in the returned dict.

    Returns a dict: optical_contrast, contrast_lambda_nm, f_3dB_GHz, switching_energy_fJ, gate_C_fF,
    R_access_ohm, contrast_per_fJ."""
    R, C_cell, f3db = lumped_rc_bandwidth(gate_C_per_area_F_m2, sheet_resistance_ohm_sq,
                                          path_length_m=path_length_m, pad_width_m=pad_width_m,
                                          cell_area_m2=cell_area_m2)
    E_event = float(switching_energy_per_area(gate_C_per_area_F_m2, voltage_swing_V)) * float(cell_area_m2)
    fom = float(optical_contrast) / (E_event * 1e15) if E_event > 0 else float("nan")  # contrast / fJ
    return {
        "optical_contrast":      float(optical_contrast),
        "contrast_lambda_nm":    float(contrast_lambda_nm),
        "f_3dB_GHz":             float(f3db) * 1e-9,
        "switching_energy_fJ":   E_event * 1e15,
        "gate_C_fF":             float(C_cell) * 1e15,
        "R_access_ohm":          float(R),
        "contrast_per_fJ":       fom,
    }


def resonance_dip(wavelengths_nm: Sequence[float],
                  spectrum: Sequence[float]) -> Tuple[float, float]:
    """Locate a resonance dip (minimum) in a spectrum with sub-grid accuracy.

    Fits a parabola to the three points around the discrete minimum and
    returns the interpolated (wavelength, value) of the vertex. This recovers
    the dip position to better than the wavelength sampling step -- important
    when comparing small bias-induced resonance shifts that are a fraction of
    the scan spacing.

    Args:
      wavelengths_nm : 1D array of wavelengths (need not be sorted)
      spectrum       : 1D array of the quantity to minimize (e.g. |r|^2),
                       same length as wavelengths_nm

    Returns:
      (dip_wavelength_nm, dip_value). Falls back to the discrete minimum if
      there are < 3 points or the minimum is at an array edge (no parabola).

    Example:
      lam = [1250, 1300, 1350, 1400]; R = [0.30, 0.05, 0.18, 0.40]
      dip_nm, dip_val = resonance_dip(lam, R)   # ~1297 nm, ~0.04
    """
    lam = np.asarray(wavelengths_nm, dtype=np.float64).ravel()
    y = np.asarray(spectrum, dtype=np.float64).ravel()
    if lam.shape != y.shape:
        raise ValueError("wavelengths_nm and spectrum must have equal length")
    finite = np.isfinite(lam) & np.isfinite(y)
    if not finite.any():
        raise ValueError("spectrum has no finite values to locate a dip")
    if not finite.all():
        # a NaN/inf sample (e.g. a failed solve point) must not hijack argmin or the
        # parabolic fit; locate the dip among the finite samples only.
        lam, y = lam[finite], y[finite]
    order = np.argsort(lam)
    lam, y = lam[order], y[order]

    i = int(np.argmin(y))
    if len(lam) < 3 or i == 0 or i == len(lam) - 1:
        return float(lam[i]), float(y[i])

    xs = lam[i - 1:i + 2]
    ys = y[i - 1:i + 2]
    # Exact parabola through the 3 (possibly unequally-spaced) points: y = a x^2
    # + b x + c. np.polyfit on exactly 3 points is the exact interpolant; the
    # vertex is -b/2a. (The old symmetric-step formula was exact only on a
    # uniform grid -- it biased the dip by up to a full step on a non-uniform one.)
    a, b, c = np.polyfit(xs, ys, 2)
    if a <= 1e-30:                       # not an upward parabola -> no interior min
        return float(lam[i]), float(y[i])
    lam_dip = -b / (2.0 * a)
    if not (xs[0] <= lam_dip <= xs[-1]):  # vertex outside the bracket -> fall back
        return float(lam[i]), float(y[i])
    val_dip = a * lam_dip ** 2 + b * lam_dip + c
    return float(lam_dip), float(val_dip)


def resonance_shift(wavelengths_nm: Sequence[float],
                    spectrum_ref: Sequence[float],
                    spectrum_test: Sequence[float]) -> float:
    """Spectral shift (nm) of a resonance dip between two spectra on the same
    wavelength grid: dip(test) - dip(ref). Positive = redshift.
    """
    ref_nm, _ = resonance_dip(wavelengths_nm, spectrum_ref)
    test_nm, _ = resonance_dip(wavelengths_nm, spectrum_test)
    return float(test_nm - ref_nm)


# ------------------------------------------------------------------------------
# Fano / Lorentzian lineshape fitting + quasi-BIC asymmetry scaling
#
# A symmetry-broken (quasi-)BIC driven spectrum is a Fano lineshape
#     T(x) = a_bg + b_bg * (q + eps_r)^2 / (1 + eps_r^2),   eps_r = 2 (x - x0) / gamma
# with q the Fano asymmetry parameter (Fano, Phys. Rev. 124:1866 (1961)); gamma is the
# FWHM of the underlying Lorentzian in x-units and Q = |x0| / gamma. As the coupling
# symmetry is restored the radiative width collapses and the mode Q diverges as the
# canonical quasi-BIC power law Q ~ delta^-2 in the asymmetry parameter delta
# (Koshelev et al., PRL 121:193903 (2018)); quasi_bic_scaling() extracts that exponent.
#
# q-PARAMETERIZATION (stability). Fitting the raw q is ill-conditioned in both limits
# (q -> +-inf is a Lorentzian PEAK where b_bg and q^2 become degenerate; q -> 0 is a
# symmetric DIP where the dispersive weight vanishes). We instead fit the algebraically
# equivalent SYMMETRIC + DISPERSIVE Lorentzian decomposition (the "Fano phase" form)
#     T(x) = C0 + C_L / (1 + eps_r^2) + C_D * eps_r / (1 + eps_r^2)
# which is LINEAR in (C0, C_L, C_D) once (x0, gamma) are fixed:
#     C0 = a_bg + b_bg,   C_L = b_bg (q^2 - 1),   C_D = 2 b_bg q.
# Only (x0, gamma) are nonlinear, so the fit is a 2-parameter variable-projection
# (VARPRO) least squares with an exact inner linear solve -- well conditioned across the
# whole q range. We then RECOVER and REPORT the physical (a_bg, b_bg, q):
#     b_bg = [ -C_L + sqrt(C_L^2 + C_D^2) ] / 2   (the b_bg >= 0 branch, resonant gain),
#     q    = C_D / (2 b_bg),   a_bg = C0 - b_bg.
# The peak limit q -> +-inf shows up as b_bg -> 0 (reported q = +-inf, x0/gamma still
# exact); the dip limit q -> 0 shows up as C_D -> 0 (reported q = 0). Both fit stably.
# ------------------------------------------------------------------------------

_FANO_X_KINDS = ("freq", "wavelength", "energy")


@dataclass(frozen=True)
class FanoFit:
    """Result of fano_fit(). omega0 and gamma_fwhm are in the SAME units as the input x
    (Hz for x_kind='freq', nm for 'wavelength', ...). q is the Fano asymmetry parameter
    (Fano, Phys. Rev. 124:1866 (1961)); a_bg / b_bg the constant background and resonant
    amplitude of T = a_bg + b_bg (q + eps_r)^2 / (1 + eps_r^2); Q = |omega0| / gamma_fwhm;
    residual_rms the RMS of (fit - data). q = +-inf flags the Lorentzian-peak limit."""
    omega0: float
    gamma_fwhm: float
    q: float
    a_bg: float
    b_bg: float
    Q: float
    residual_rms: float


@dataclass(frozen=True)
class LorentzianFit:
    """Result of lorentzian_fit() -- the symmetric (q -> +-inf) sibling of FanoFit,
    T = baseline + amplitude / (1 + (2 (x - x0) / fwhm)^2). amplitude > 0 is a peak,
    < 0 a dip; fwhm is in x-units, Q = |x0| / fwhm, residual_rms the RMS residual."""
    x0: float
    fwhm: float
    Q: float
    amplitude: float
    baseline: float
    residual_rms: float


def _fano_columns(u: np.ndarray, x0u: float, gu: float, ncols: int) -> np.ndarray:
    """Design matrix columns [1, L, (D)] at normalized (x0u, gu): L = 1/(1+eps^2),
    D = eps/(1+eps^2), eps = 2 (u - x0u)/gu. ncols=2 -> symmetric (Lorentzian) only,
    ncols=3 -> add the dispersive column (full Fano)."""
    eps = 2.0 * (u - x0u) / gu
    denom = 1.0 + eps * eps
    cols = [np.ones_like(u), 1.0 / denom]
    if ncols == 3:
        cols.append(eps / denom)
    return np.stack(cols, axis=1)


def _fano_inner_solve(u, y, x0u, gu, ncols):
    """Exact linear least-squares of the amplitude coeffs at fixed (x0u, gu); returns
    (coeffs, model). This is the VARPRO inner solve."""
    A = _fano_columns(u, x0u, gu, ncols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, A @ coef


def _fano_seed_candidates(u, y):
    """(x0u, gu) seed candidates in normalized coords. Primary seed follows the spec's
    extremum-pair strategy: a Fano has an adjacent min/max whose SPACING seeds gamma and
    whose ORDER seeds sign(q) (handled implicitly by the dispersive column). A light box
    smooth de-sensitizes the extrema to noise. Several gamma scales are added so the
    2-D VARPRO never stalls in a spurious minimum (the q -> 0 symmetric dip has no true
    max, and large |q| pushes the antiresonance far into one wing)."""
    n = u.size
    w = max(3, n // 40)
    if w % 2 == 0:
        w += 1
    ker = np.ones(w) / w
    ys = np.convolve(y, ker, mode="same") if n > w else np.asarray(y, dtype=np.float64)
    imin = int(np.argmin(ys))
    imax = int(np.argmax(ys))
    span = float(u.max() - u.min()) or 1.0
    x0_pair = 0.5 * (u[imin] + u[imax])
    g_pair = abs(u[imax] - u[imin])
    if not np.isfinite(g_pair) or g_pair < 1e-3 * span:
        g_pair = 0.3 * span
    x0_cands = [x0_pair, float(u[imin]), float(u[imax]), 0.5 * (float(u.min()) + float(u.max()))]
    g_cands = [g_pair, 0.5 * g_pair, 2.0 * g_pair, 0.15 * span, 0.5 * span]
    seeds = []
    for x0c in x0_cands:
        for gc in g_cands:
            gc = min(max(gc, 1e-3 * span), 6.0 * span)
            seeds.append((float(x0c), float(gc)))
    return seeds


def _fano_varpro(x, y, ncols):
    """Core VARPRO fit shared by fano_fit / lorentzian_fit. Fits T(x) with ncols amplitude
    columns; only (x0, gamma) are nonlinear. Returns (x0, gamma, coeffs, resid_rms) in the
    ORIGINAL x-units. Robust to arbitrary x-scale (Hz ~ 1e14 or nm ~ 1e3) via internal
    normalization, and to seed choice via a small multi-start."""
    from scipy.optimize import least_squares

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("x and spectrum must have equal length")
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5:
        raise ValueError("need >= 5 finite (x, spectrum) samples to fit a Fano lineshape")
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]

    xc = 0.5 * (float(x.min()) + float(x.max()))
    xs = 0.5 * (float(x.max()) - float(x.min()))
    if xs == 0.0:
        raise ValueError("x samples are all equal; no spectral axis to fit")
    u = (x - xc) / xs                                   # normalized to ~[-1, 1]

    span = float(u.max() - u.min())
    lo = np.array([u.min() - 0.2 * span, 1e-4 * span])
    hi = np.array([u.max() + 0.2 * span, 8.0 * span])

    def resid(p):
        x0u, gu = p
        _, model = _fano_inner_solve(u, y, x0u, gu, ncols)
        return model - y

    best = None
    for x0s, gs in _fano_seed_candidates(u, y):
        p0 = np.array([min(max(x0s, lo[0]), hi[0]), min(max(gs, lo[1]), hi[1])])
        try:
            sol = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                                ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=4000)
        except Exception:                               # pragma: no cover - degenerate seed
            continue
        cost = float(sol.cost)
        if best is None or cost < best[0]:
            best = (cost, sol.x)
    if best is None:                                    # pragma: no cover
        raise RuntimeError("fano/lorentzian fit failed from every seed")

    x0u, gu = best[1]
    coef, model = _fano_inner_solve(u, y, x0u, gu, ncols)
    resid_rms = float(np.sqrt(np.mean((model - y) ** 2)))
    x0 = xc + x0u * xs
    gamma = gu * xs
    return x0, gamma, coef, resid_rms


def fano_fit(x_hz_or_nm: Sequence[float], spectrum: Sequence[float], *,
             x_kind: str = "freq") -> FanoFit:
    """Robust least-squares fit of a Fano lineshape to a driven spectrum T(x):

        T(x) = a_bg + b_bg * (q + eps_r)^2 / (1 + eps_r^2),   eps_r = 2 (x - x0) / gamma

    (Fano, Phys. Rev. 124:1866 (1961)). This is the driven-spectrum signature of a
    quasi-BIC / coupled-resonance feature; q sets the asymmetry, gamma the linewidth.

    Args:
      x_hz_or_nm : 1-D spectral axis (need not be sorted / uniform). Frequency (Hz),
                   wavelength (nm), or energy -- see x_kind. Returned omega0 / gamma_fwhm
                   are in THESE units.
      spectrum   : 1-D T(x) (e.g. transmittance or reflectance), same length.
      x_kind     : 'freq' | 'wavelength' | 'energy' (informational; Q = |omega0|/gamma_fwhm
                   is a units-ratio and is the resonance Q for a frequency/energy axis and
                   the first-order lambda0/Delta_lambda for a wavelength axis).

    Returns a FanoFit(omega0, gamma_fwhm, q, a_bg, b_bg, Q, residual_rms).

    Method / failure modes: the fit is done in the numerically stable symmetric+dispersive
    ("Fano phase") parameterization and (a_bg, b_bg, q) are RECOVERED afterward (see the
    module comment above). Both edge regimes fit WITHOUT divergence:
      * q -> +-inf  (Lorentzian PEAK limit): b_bg -> 0, so q is degenerate with the
        background scale; the returned q is +-inf (or a large magnitude) while omega0 /
        gamma_fwhm / Q remain well-determined. Use lorentzian_fit() for this regime.
      * q -> 0      (symmetric DIP limit): the dispersive weight C_D -> 0 and q -> 0
        cleanly; the shape is a pure Lorentzian dip on the a_bg background.
    """
    if x_kind not in _FANO_X_KINDS:
        raise ValueError("x_kind must be one of {}".format(_FANO_X_KINDS))
    x0, gamma, coef, resid_rms = _fano_varpro(x_hz_or_nm, spectrum, ncols=3)
    C0, C_L, C_D = float(coef[0]), float(coef[1]), float(coef[2])

    # Recover the physical (a_bg, b_bg, q) on the b_bg >= 0 branch. Two algebraically equal
    # forms of b_bg = [-C_L + sqrt(C_L^2 + C_D^2)] / 2 avoid catastrophic cancellation:
    # the rationalized form D^2 / (2(R + C_L)) is stable for C_L > 0 (large |q|), the direct
    # form for C_L <= 0 (small |q|).
    Rq = float(np.hypot(C_L, C_D))
    if C_L > 0.0:
        b_bg = (C_D * C_D) / (2.0 * (Rq + C_L)) if (Rq + C_L) > 0.0 else 0.0
    else:
        b_bg = 0.5 * (-C_L + Rq)
    b_scale = abs(C0) + Rq + 1e-300
    if b_bg <= 1e-12 * b_scale:                         # Lorentzian-peak limit: q degenerate
        q = float(np.inf) if C_D >= 0.0 else float(-np.inf)
        a_bg = C0 - b_bg
    else:
        q = C_D / (2.0 * b_bg)
        a_bg = C0 - b_bg
    Q = abs(x0) / gamma if gamma > 0 else float("inf")
    return FanoFit(omega0=float(x0), gamma_fwhm=float(gamma), q=float(q),
                   a_bg=float(a_bg), b_bg=float(b_bg), Q=float(Q),
                   residual_rms=float(resid_rms))


def lorentzian_fit(x: Sequence[float], spectrum: Sequence[float]) -> LorentzianFit:
    """Fit a symmetric Lorentzian T(x) = baseline + amplitude / (1 + (2 (x - x0)/fwhm)^2) --
    the q -> +-inf (symmetric peak) / q -> 0 (symmetric dip) limit of fano_fit(), sharing
    the same VARPRO machinery (it just drops the dispersive column). amplitude > 0 is a
    peak, amplitude < 0 a dip. Returns a LorentzianFit(x0, fwhm, Q, amplitude, baseline,
    residual_rms); Q = |x0| / fwhm. On a genuinely symmetric line this recovers the same
    (x0, fwhm) as fano_fit to fit-tolerance."""
    x0, gamma, coef, resid_rms = _fano_varpro(x, spectrum, ncols=2)
    baseline, amplitude = float(coef[0]), float(coef[1])
    Q = abs(x0) / gamma if gamma > 0 else float("inf")
    return LorentzianFit(x0=float(x0), fwhm=float(gamma), Q=float(Q),
                         amplitude=float(amplitude), baseline=float(baseline),
                         residual_rms=float(resid_rms))


def quasi_bic_scaling(delta_values: Sequence[float],
                      q_factors: Sequence[float]) -> Tuple[float, float, float]:
    """Log-log fit of the radiative quality factor Q vs the symmetry-breaking asymmetry
    parameter delta for a quasi-BIC: Q = prefactor * delta^exponent. The canonical
    symmetry-protected quasi-BIC obeys exponent = -2 (Q ~ 1/delta^2, Koshelev et al.,
    PRL 121:193903 (2018)); a departure or a poor r2 flags a Q that has saturated (e.g. an
    absorption-limited Q_abs floor contaminating the radiative scaling).

    Args:
      delta_values : asymmetry parameters (> 0), one per structure.
      q_factors    : the corresponding fitted quality factors (> 0).

    Returns (exponent, prefactor, r2): exponent = slope of log Q vs log delta;
    prefactor = exp(intercept) so Q ~= prefactor * delta**exponent; r2 = coefficient of
    determination of the log-log fit (1.0 = perfect power law; a drop flags saturation /
    contamination). Needs >= 2 valid points.
    """
    delta = np.asarray(delta_values, dtype=np.float64).ravel()
    Q = np.asarray(q_factors, dtype=np.float64).ravel()
    if delta.shape != Q.shape:
        raise ValueError("delta_values and q_factors must have equal length")
    good = np.isfinite(delta) & np.isfinite(Q) & (delta > 0.0) & (Q > 0.0)
    if good.sum() < 2:
        raise ValueError("need >= 2 points with delta > 0 and Q > 0 for a log-log fit")
    ld = np.log(delta[good])
    lq = np.log(Q[good])
    exponent, intercept = np.polyfit(ld, lq, 1)
    fit = exponent * ld + intercept
    ss_res = float(np.sum((lq - fit) ** 2))
    ss_tot = float(np.sum((lq - lq.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return float(exponent), float(np.exp(intercept)), float(r2)
