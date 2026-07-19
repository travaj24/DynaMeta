"""Harmonic diagnostics for chi2/chi3 FDTD runs (roadmap 3.1: SH/TH spectral extraction).

The nonlinear FDTD kernels (fdtd_nd.kernels2d.run_2d_te / solve_fdtd_2d) already MARCH the
second- and third-harmonic content of a chi2/chi3 slab -- an instantaneous chi2 term radiates
2w (P = eps0 chi2 E^2) and the instantaneous Kerr term radiates 3w (the kernel uses
eps_eff = eps_inf + 3 chi3 E^2, i.e. D_NL = 3 eps0 chi3 E^3; the full real field cubed carries
a cos(3wt) component). This module POST-PROCESSES the recorded exit-plane time series into
band powers at w / 2w / 3w, the SHG/THG conversion efficiencies, the P_2w ~ P_w^2 /
P_3w ~ P_w^3 log-log slopes, and an undepleted-pump validity check. It re-reads the
already-validated physics -- it does not re-derive it (the coupled-wave oracle lives in
validation/fdtd_chi2_shg_raman.py).

Conventions: SI units; exp(-i omega t) (so the +z transmitted flux S_z = -Re(E_y H_x*) is
positive for a forward wave, matching fdtd_nd.results._flux); pure numpy. Reference for the
undepleted coupled-wave SHG closed form: R. W. Boyd, "Nonlinear Optics", 3rd ed., ch. 2
(sec. 2.2-2.3): for perfect phase matching (Delta_k = 0) the second-harmonic field grows as
A_2(z) ~ i (2 d_eff omega^2 / k_2 c^2) A_1^2 z, so the SH INTENSITY I_2 ~ (d_eff L)^2 I_1^2 --
i.e. P_2w scales as the SQUARE of the pump power P_w (slope 2), and the third harmonic as the
CUBE (slope 3).

BANDWIDTH / WINDOWING CONVENTION (documented so the band integrals are reproducible):
  * The source pulse has finite bandwidth, so each harmonic occupies a BAND, not a line. The
    n-th harmonic is the n-fold self-convolution of the pump spectrum (E^n in time), so it is
    centered on n*f0 and is ~n times WIDER than the pump. We therefore integrate the n-th
    harmonic over a CONSTANT FRACTIONAL band [n*f0*(1 - bw), n*f0*(1 + bw)]: the absolute width
    2*n*f0*bw grows linearly with n, matching the n-fold broadening, while `bw` (default 0.15)
    stays fixed. bw < 1/(2*n_max + 1) guarantees adjacent harmonic bands do NOT overlap (the
    binding 2w<->3w gap needs bw < 0.2); the default 0.15 is safe through the third harmonic
    and matches the SHG window proven in validation/fdtd_chi2_shg_raman.py ([1.85, 2.15]*f0).
  * NO apodization window is applied (rectangular): the modulated-Gaussian source returns to
    ~0 at both ends of the record and the run is many pulse-widths long, so the trace is already
    effectively tapered; a window would bias the bin-for-bin comparison against the analytic
    closed form. The pump / 2w / 3w lobes are separated by ~f0 >> their widths, so inter-band
    spectral leakage sits far below the double-precision numerical floor (verified by the
    zero-nonlinearity floor gate: harmonic bands > 60 dB below the fundamental).
  * "Power" is the band-summed spectral density. When the magnetic probe H_x is available the
    density is the per-frequency time-averaged Poynting flux S_z = -Re(E_y H_x*) (a true power,
    correct even for a dispersive exit medium); when only E is supplied it is |E(f)|^2 (energy
    spectral density). Efficiency RATIOS (P_2w/P_w) are identical either way in a non-dispersive
    exit medium, where the impedance is frequency-independent and cancels.

ORDER RESOLUTION: for a periodic 2D unit cell the exit probe is an x-line (nsteps, nx). A spatial
DFT over x resolves the diffraction orders m = 0, +-1, ...; the module returns per-order band
powers (with 1/nx Parseval normalization so the order sum equals the x-summed flux). A laterally
uniform chi2/chi3 slab radiates only the specular m = 0 order; a grating spreads the harmonics
across orders.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------------------
# input normalization
# --------------------------------------------------------------------------------------------------
def _extract(obj, dt, field):
    """Return (e, h, dt) from any accepted input.

    Accepts: an FDTD result carrying a `.time_trace` dict (from solve_fdtd_1d/2d with
    return_time_trace=True); a bare time_trace dict; a 2-tuple (trace, dt); a 3-tuple
    (E_trace, H_trace, dt); or a bare ndarray (dt then required as a kwarg). `field` selects which
    boundary series of a time_trace dict to analyze ('transmitted' default; 'reflected';
    'incident_right'; ...). Its co-located H_x probe, if present, is read from `field + "_hx"`.
    """
    if hasattr(obj, "time_trace") and not isinstance(obj, dict):
        tt = obj.time_trace
        if tt is None:
            raise ValueError(
                "harmonic diagnostics: the FDTD result carries no time_trace -- re-run with "
                "solve_fdtd_2d(..., return_time_trace=True) (or solve_fdtd_1d) so the exit-plane "
                "series is exposed.")
        obj = tt
    if isinstance(obj, dict):
        if "dt" not in obj:
            raise ValueError("harmonic diagnostics: trace dict needs a 'dt' entry")
        dt = float(obj["dt"])
        key = field if field in obj else None
        if key is None:
            for k in ("transmitted", "ey", "e"):     # generic fallbacks
                if k in obj:
                    key = k
                    break
        if key is None:
            raise ValueError("harmonic diagnostics: trace dict has no {!r} (or transmitted/ey/e) "
                             "series".format(field))
        e = np.asarray(obj[key], dtype=float)
        h = obj.get(key + "_hx", obj.get("hx", obj.get("h")))
        return e, (None if h is None else np.asarray(h, dtype=float)), dt
    if isinstance(obj, (tuple, list)):
        if len(obj) == 2 and np.isscalar(obj[1]):
            return np.asarray(obj[0], dtype=float), None, float(obj[1])
        if len(obj) == 3:
            return (np.asarray(obj[0], dtype=float), np.asarray(obj[1], dtype=float), float(obj[2]))
        raise ValueError("harmonic diagnostics: tuple input must be (trace, dt) or (E, H, dt)")
    if dt is None:
        raise ValueError("harmonic diagnostics: pass dt=... for a bare-array trace")
    return np.asarray(obj, dtype=float), None, float(dt)


def _order_spectra(e, h, dt):
    """(freqs, S, orders, power_type): the one-sided per-order spectral power density.

    e (and optional h) are (nsteps,) [single order] or (nsteps, nx) [periodic x-line]. A real
    rfft in time and a full fft over x give the (freq, order) amplitudes. With h: the +z Poynting
    density S = -Re(E_k conj(H_k))/nx (sums over orders to the x-summed flux, Parseval). Without h:
    |E_k|^2/nx (energy spectral density). `orders` are the signed diffraction indices 0, +-1, ...
    """
    e = np.asarray(e, dtype=float)
    if e.ndim == 1:
        e = e[:, None]
    nsteps, nx = e.shape
    f = np.fft.rfftfreq(nsteps, dt)
    Ek = np.fft.fft(np.fft.rfft(e, axis=0), axis=1)        # (nfreq, nx): time->freq, x->order
    if h is not None:
        h = np.asarray(h, dtype=float)
        if h.ndim == 1:
            h = h[:, None]
        Hk = np.fft.fft(np.fft.rfft(h, axis=0), axis=1)
        S = -np.real(Ek * np.conj(Hk)) / nx
        ptype = "poynting_flux"
    else:
        S = (np.abs(Ek) ** 2) / nx
        ptype = "e2_density"
    orders = np.fft.fftfreq(nx, d=1.0 / nx).round().astype(int)   # [0, 1, ..., -2, -1]
    return f, S, orders, ptype


def _band(f, S, f0, n, bw):
    """(P_total, P_by_order, (lo, hi), truncated): the band power of harmonic n."""
    lo, hi = n * f0 * (1.0 - bw), n * f0 * (1.0 + bw)
    df = f[1] - f[0] if f.size > 1 else 0.0
    truncated = bool(hi > f[-1] + 0.5 * df)
    m = (f >= lo) & (f <= hi)
    P_by_order = S[m].sum(axis=0)                          # (norder,)
    return float(P_by_order.sum()), P_by_order, (float(lo), float(hi)), truncated


# --------------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------------
def harmonic_spectrum(result_or_trace, fundamental_hz, *, dt=None, field="transmitted",
                      bandwidth_frac=0.15, max_order=3):
    """Band power at the fundamental, 2w and 3w from a recorded FDTD exit-plane time series.

    result_or_trace : an FDTD result (return_time_trace=True), a time_trace dict, a (trace, dt) or
        (E, H, dt) tuple, or a bare ndarray (then pass dt=...). See _extract for the full contract.
    fundamental_hz  : the pump carrier frequency f0 [Hz].
    field           : which boundary series to analyze in a time_trace dict (default 'transmitted').
    bandwidth_frac  : the constant-fractional half-bandwidth `bw`; harmonic n is integrated over
        [n*f0*(1-bw), n*f0*(1+bw)] (see the module docstring for the convention). Default 0.15.
    max_order       : highest harmonic to report (default 3 -> w, 2w, 3w).

    Returns a dict with (all powers in the units of the chosen density -- Poynting flux if the
    trace carries H_x, else |E|^2):
      fundamental_hz, freqs_Hz, power_spectrum (order-summed density vs f), orders,
      power {n: P_total}, power_by_order {n: (norder,) array}, bands_hz {n: (lo, hi)},
      band_truncated {n: bool}, power_type, bandwidth_frac, nyquist_hz,
      and the convenience aliases P_w / P_2w / P_3w.
    """
    f0 = float(fundamental_hz)
    if f0 <= 0.0:
        raise ValueError("fundamental_hz must be > 0")
    # adjacent harmonic bands [n*f0*(1-bw), n*f0*(1+bw)] and [(n+1)..] are disjoint iff
    # bw < 1/(2n+1); the tightest pair among harmonics 1..max_order is n = max_order-1, so the
    # overlap-free bound is bw < 1/(2*max_order-1) (= 0.2 for w/2w/3w).
    _bw_max = 1.0 / (2 * max_order - 1) if max_order > 1 else 1.0
    if not (0.0 < bandwidth_frac < _bw_max):
        raise ValueError("bandwidth_frac must satisfy 0 < bw < 1/(2*max_order-1) = {:.4f} so the "
                         "harmonic bands do not overlap; got {}".format(_bw_max, bandwidth_frac))
    e, h, dt = _extract(result_or_trace, dt, field)
    f, S, orders, ptype = _order_spectra(e, h, dt)
    power, power_by_order, bands, trunc = {}, {}, {}, {}
    for n in range(1, max_order + 1):
        Ptot, Pord, band_hz, tr = _band(f, S, f0, n, bandwidth_frac)
        power[n], power_by_order[n], bands[n], trunc[n] = Ptot, Pord, band_hz, tr
    out = {
        "fundamental_hz": f0,
        "freqs_Hz": f,
        "power_spectrum": S.sum(axis=1),
        "orders": orders,
        "power": power,
        "power_by_order": power_by_order,
        "bands_hz": bands,
        "band_truncated": trunc,
        "power_type": ptype,
        "bandwidth_frac": float(bandwidth_frac),
        "nyquist_hz": float(f[-1]),
    }
    out["P_w"] = power.get(1)
    out["P_2w"] = power.get(2)
    out["P_3w"] = power.get(3)
    return out


def conversion_efficiency(result_or_trace, fundamental_hz, *, incident=None,
                          normalization="incident", dt=None, field="transmitted",
                          bandwidth_frac=0.15):
    """SHG / THG conversion efficiencies eta_shg = P_2w / P_w, eta_thg = P_3w / P_w.

    P_2w, P_3w are the transmitted second-/third-harmonic band powers. P_w is the pump power used
    for normalization:
      normalization='incident' (default): the INCIDENT fundamental power -- taken from `incident`
        (an array / trace / result passed explicitly) if given, else from the result's own
        'incident_right' time_trace series if present. This is the standard SHG/THG efficiency
        (radiated harmonic per incident pump).
      normalization='transmitted': the TRANSMITTED fundamental band power from the same series.
    The dict records which normalization actually applied (falls back to 'transmitted' with the
    flag set if no incident reference is available).
    """
    hs = harmonic_spectrum(result_or_trace, fundamental_hz, dt=dt, field=field,
                           bandwidth_frac=bandwidth_frac)
    P_w_trans, P_2w, P_3w = hs["P_w"], hs["P_2w"], hs["P_3w"]
    used = "transmitted"
    P_w = P_w_trans
    if normalization == "incident":
        inc_src = None
        if incident is not None:
            inc_src = incident
        else:
            tt = getattr(result_or_trace, "time_trace", None)
            if tt is None and isinstance(result_or_trace, dict):
                tt = result_or_trace
            if tt is not None and "incident_right" in tt:
                inc_src = tt
        if inc_src is not None:
            hs_inc = harmonic_spectrum(inc_src, fundamental_hz, dt=dt, field="incident_right",
                                       bandwidth_frac=bandwidth_frac)
            P_w = hs_inc["P_w"]
            used = "incident"
    elif normalization != "transmitted":
        raise ValueError("normalization must be 'incident' or 'transmitted'")
    return {
        "eta_shg": (P_2w / P_w) if P_w else float("inf"),
        "eta_thg": (P_3w / P_w) if P_w else float("inf"),
        "P_w": float(P_w),
        "P_w_transmitted": float(P_w_trans),
        "P_2w": float(P_2w),
        "P_3w": float(P_3w),
        "normalization": used,
        "power_type": hs["power_type"],
        "harmonics": hs,
    }


def power_slope(pump_powers, harmonic_powers):
    """Log-log slope + r^2 of a harmonic-vs-pump power sweep.

    Fits log(harmonic_power) = slope * log(pump_power) + intercept by least squares. For a
    perturbative chi2 slab P_2w ~ P_w^2 (slope 2); for chi3 THG P_3w ~ P_w^3 (slope 3). Requires
    >= 2 strictly-positive samples. Returns {slope, intercept, r2, n}.
    """
    x = np.asarray(pump_powers, dtype=float)
    y = np.asarray(harmonic_powers, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("power_slope: pump_powers and harmonic_powers must be 1-D and same length")
    if x.size < 2:
        raise ValueError("power_slope: need >= 2 samples")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("power_slope: a log-log slope needs strictly positive powers")
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ly, rcond=None)
    yhat = slope * lx + intercept
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n": int(x.size)}


def undepleted_validity(result_or_trace, fundamental_hz, *, incident=None, dt=None,
                        field="transmitted", bandwidth_frac=0.15, threshold=0.1):
    """Undepleted-pump validity diagnostic: is the harmonic conversion small enough that the
    perturbative (undepleted-pump) closed forms apply?

    Reports:
      eta_shg, eta_thg, eta_total = (P_2w + P_3w) / P_w  -- the converted fraction (the pump
        drives the harmonics, so eta_total is the leading-order pump-depletion estimate; Boyd
        ch. 2: undepleted pump requires eta << 1).
      depletion_estimate  = eta_total (energy drained INTO the harmonics, a lower bound).
      depletion_measured  = (P_w_incident - P_w_transmitted) / P_w_incident when an incident
        reference is available (the DIRECT pump loss to conversion + reflection); None otherwise.
      undepleted (bool)   = eta_total < threshold (default 0.1).
    """
    ce = conversion_efficiency(result_or_trace, fundamental_hz, incident=incident,
                               normalization="incident", dt=dt, field=field,
                               bandwidth_frac=bandwidth_frac)
    P_w = ce["P_w"]
    eta_total = (ce["P_2w"] + ce["P_3w"]) / P_w if P_w else float("inf")
    depletion_measured = None
    if ce["normalization"] == "incident" and ce["P_w"] > 0.0:
        depletion_measured = float((ce["P_w"] - ce["P_w_transmitted"]) / ce["P_w"])
    return {
        "eta_shg": ce["eta_shg"],
        "eta_thg": ce["eta_thg"],
        "eta_total": float(eta_total),
        "depletion_estimate": float(eta_total),
        "depletion_measured": depletion_measured,
        "undepleted": bool(eta_total < threshold),
        "threshold": float(threshold),
        "normalization": ce["normalization"],
    }
