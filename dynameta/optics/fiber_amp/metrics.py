"""Amplifier performance metrics on top of the steady-state solve (docs sec.4): gain saturation
(the compression curve and the 3 dB saturation output power), power-conversion and slope
efficiency against the quantum-defect (Stokes) ceiling lambda_pump/lambda_signal, and the
small-signal gain spectrum with its flatness / tilt across a signal band.

Each metric re-runs the amplifier with the pump or signal rescaled (Pump/Signal are frozen, so
this uses dataclasses.replace to clone), which is why they take a FiberAmplifier rather than a
single solved result.

AMPLIFIER CONTRACT (audit A-3 follow-on). Every rebuild goes through the PUBLIC, type-preserving
re-seed protocol, never through a class-private clone:
    with_signals(signals) -> a copy with that signal list
    with_pumps(pumps)     -> a copy with that pump list
    without_ase()         -> a copy with the ASE band(s) dropped
FiberAmplifier and ErYbAmplifier both implement all three, so every metric here works on either
class. A third-party amplifier MUST implement them too: this module reached for the
FiberAmplifier-only `_clone` before, which made gain_compression_curve / saturation_output_power /
slope_efficiency / gain_spectrum raise a bare `AttributeError: 'ErYbAmplifier' object has no
attribute '_clone'` (only power_conversion_efficiency, which never clones, worked). A missing
protocol method now raises a TypeError NAMING the method instead. Pure numpy/scipy; SI units.
docs/fiber_amp_model_spec.md sec.4.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from dynameta.optics.fiber_amp.steady_state import (FiberAmplifier, Signal, SteadyStateResult)

__all__ = ["CompressionCurve", "SlopeEfficiency", "GainSpectrum",
           "gain_compression_curve", "saturation_output_power", "power_conversion_efficiency",
           "slope_efficiency", "gain_spectrum", "gain_flatness", "stokes_limit"]


# ---- amplifier cloning helpers -------------------------------------------------------------
_PROTOCOL = ("with_signals(signals), with_pumps(pumps) and without_ase() -- the public, "
             "type-preserving amplifier re-seed protocol (FiberAmplifier and ErYbAmplifier both "
             "implement all three)")


def _rebuild(amp, method: str, *args):
    """Run ONE step of the public amplifier re-seed protocol, or raise a TypeError NAMING the
    method that is missing. Every clone in this module goes through here: reaching straight for
    the FiberAmplifier-private `_clone` is what made four of the five metrics die with a bare
    `AttributeError: 'ErYbAmplifier' object has no attribute '_clone'` (audit A-3 follow-on).

    The clone must carry every opt-in -- the ConcentrationModel (audit S3-1: dropping it made
    every metric run the ideal model, silently losing PIQ dark absorption and photodarkening and
    mis-scaling upconversion) and the axial temperature profile (audit A-5) -- which is why the
    protocol is implemented by the amplifier CLASS rather than re-rolled here."""
    fn = getattr(amp, method, None)
    if not callable(fn):
        raise TypeError(
            "{} does not implement {}(), required by dynameta.optics.fiber_amp.metrics. An "
            "amplifier passed to these metrics must implement {}; each must return a copy of the "
            "SAME class carrying every opt-in (ASE band, concentration model, upconversion "
            "coefficient, axial temperature profile).".format(type(amp).__name__, method,
                                                              _PROTOCOL))
    return fn(*args)


def _with(amp: FiberAmplifier, *, pumps=None, signals=None) -> FiberAmplifier:
    """Clone the amplifier with pumps and/or signals swapped, through the public protocol only
    (see _rebuild). Neither argument -> the amplifier is returned unchanged."""
    out = amp
    if pumps is not None:
        out = _rebuild(out, "with_pumps", list(pumps))
    if signals is not None:
        out = _rebuild(out, "with_signals", list(signals))
    return out


def _set_total_pump(amp: FiberAmplifier, P_total_W: float) -> FiberAmplifier:
    cur = sum(p.power_W for p in amp.pumps)
    if cur <= 0.0:
        raise ValueError("_set_total_pump: the amplifier has zero total launched pump power; "
                         "cannot rescale to {} W (audit S3-37: the old behaviour silently "
                         "collapsed every pump to zero)".format(P_total_W))
    sc = P_total_W / cur
    return _with(amp, pumps=[replace(p, power_W=p.power_W * sc) for p in amp.pumps])


def _set_signal(amp: FiberAmplifier, P_W: float, index: int = 0) -> FiberAmplifier:
    sigs = list(amp.signals)
    sigs[index] = replace(sigs[index], power_W=P_W)
    return _with(amp, signals=sigs)


def _launched_pump_W(amp: FiberAmplifier) -> float:
    return float(sum(p.power_W for p in amp.pumps))


def _signal_out_in(result: SteadyStateResult):
    """(total signal output power at z=L, total signal input power at z=0)."""
    idx = [i for i, k in enumerate(result.kind) if k == "signal"]
    return (float(np.sum(result.power_W[idx, -1])), float(np.sum(result.power_W[idx, 0])))


# ---- gain saturation -----------------------------------------------------------------------
@dataclass
class CompressionCurve:
    p_in_W: np.ndarray
    p_out_W: np.ndarray
    gain_dB: np.ndarray
    small_signal_gain_dB: float
    p_sat_out_W: float             # 3 dB saturation OUTPUT power (NaN if not reached)
    p_sat_in_W: float
    compression_dB: float


def gain_compression_curve(amp: FiberAmplifier, p_in_W: Sequence[float], *, signal_index: int = 0,
                           compression_dB: float = 3.0) -> CompressionCurve:
    """Sweep the input signal power (channel signal_index) and record output power and gain. The
    small-signal gain is taken at the smallest input; the saturation output power is where the
    gain has dropped by compression_dB (default 3 dB), found by interpolation on the gain curve."""
    p_in = np.atleast_1d(np.asarray(p_in_W, float))
    p_out = np.empty_like(p_in)
    gdB = np.empty_like(p_in)
    for j, pin in enumerate(p_in):
        r = _set_signal(amp, float(pin), signal_index).solve()
        po = float(r.power_W[[i for i, k in enumerate(r.kind) if k == "signal"][signal_index], -1])
        p_out[j] = po
        gdB[j] = 10.0 * np.log10(po / pin)
    g0 = float(gdB[0])
    target = g0 - compression_dB
    p_sat_out, p_sat_in = np.nan, np.nan
    below = np.where(gdB <= target)[0]
    if below.size and below[0] > 0:
        k = below[0]
        # linear interpolation in log(P_in) on the gain curve
        x0, x1 = np.log(p_in[k - 1]), np.log(p_in[k])
        y0, y1 = gdB[k - 1], gdB[k]
        xs = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
        p_sat_in = float(np.exp(xs))
        p_sat_out = float(np.interp(xs, [x0, x1], [np.log(p_out[k - 1]), np.log(p_out[k])]))
        p_sat_out = float(np.exp(p_sat_out))
    return CompressionCurve(p_in, p_out, gdB, g0, p_sat_out, p_sat_in, compression_dB)


def saturation_output_power(amp: FiberAmplifier, *, signal_index: int = 0,
                            compression_dB: float = 3.0, p_in_min_W: float = 1e-7,
                            p_in_max_W: float = 5e-2, n: int = 25) -> float:
    """The compression_dB (default 3 dB) saturation OUTPUT power [W]: the output signal power at
    which the gain has compressed by compression_dB below its small-signal value. Convenience
    wrapper over gain_compression_curve on a log-spaced input sweep."""
    p_in = np.geomspace(p_in_min_W, p_in_max_W, n)
    return gain_compression_curve(amp, p_in, signal_index=signal_index,
                                  compression_dB=compression_dB).p_sat_out_W


# ---- efficiency ----------------------------------------------------------------------------
def stokes_limit(pump_lambda_m: float, signal_lambda_m: float) -> float:
    """Quantum-defect (Stokes) efficiency ceiling lambda_pump/lambda_signal: the maximum fraction
    of pump POWER convertible to signal power (each pump photon yields at most one signal photon
    of lower energy). 980/1560 ~ 0.628 for Er; 976/1030 ~ 0.947 for Yb."""
    return float(pump_lambda_m / signal_lambda_m)


def power_conversion_efficiency(amp: FiberAmplifier, result: SteadyStateResult) -> float:
    """PCE = (signal power added) / (pump power launched) = (P_sig_out - P_sig_in) / P_pump_in."""
    p_out, p_in = _signal_out_in(result)
    pp = _launched_pump_W(amp)
    return (p_out - p_in) / pp if pp > 0.0 else np.nan


@dataclass
class SlopeEfficiency:
    pump_W: np.ndarray
    signal_out_W: np.ndarray
    slope: float                   # dP_signal_out / dP_pump_launched (above threshold)
    threshold_pump_W: float        # pump at which signal_out = signal_in (net transparency)
    stokes_limit: float


def slope_efficiency(amp: FiberAmplifier, pump_W: Sequence[float], *, signal_index: int = 0,
                     saturating_signal_W: Optional[float] = None) -> SlopeEfficiency:
    """Sweep the launched pump at a FIXED, gain-saturating input signal and fit the slope
    dP_signal_out/dP_pump above the transparency threshold. The saturating signal ensures the
    extracted power goes into the signal rather than ASE, so the slope tends toward the Stokes
    ceiling for an efficient amplifier. Uses signal channel signal_index; its pump/signal
    wavelengths set the Stokes limit."""
    pump = np.atleast_1d(np.asarray(pump_W, float))
    if saturating_signal_W is not None:
        amp = _set_signal(amp, float(saturating_signal_W), signal_index)
    p_sig_in = amp.signals[signal_index].power_W
    s_out = np.empty_like(pump)
    for j, pp in enumerate(pump):
        r = _set_total_pump(amp, float(pp)).solve()
        s_out[j] = r.power_W[[i for i, k in enumerate(r.kind) if k == "signal"][signal_index], -1]
    # threshold: first pump where signal_out exceeds signal_in
    above = np.where(s_out > p_sig_in)[0]
    thr = float(pump[above[0]]) if above.size else np.nan
    # slope over the above-threshold points (>= 2 needed)
    sel = s_out > p_sig_in
    if np.count_nonzero(sel) >= 2:
        slope = float(np.polyfit(pump[sel], s_out[sel], 1)[0])
    else:
        slope = np.nan
    lam_p = amp.pumps[0].lambda_m
    lam_s = amp.signals[signal_index].lambda_m
    return SlopeEfficiency(pump, s_out, slope, thr, stokes_limit(lam_p, lam_s))


# ---- gain spectrum / flatness --------------------------------------------------------------
@dataclass
class GainSpectrum:
    lambda_m: np.ndarray
    gain_dB: np.ndarray
    peak_lambda_m: float
    flatness_dB: float             # max - min over the band
    tilt_dB_per_nm: float          # linear slope of gain vs wavelength


def gain_spectrum(amp: FiberAmplifier, probe_lambda_m: Sequence[float], *,
                  probe_power_W: float = 1e-7, with_ase: bool = False) -> GainSpectrum:
    """Small-signal gain spectrum: a weak probe of power probe_power_W is placed, one wavelength
    at a time, on the amplifier (all its own signals removed so the probe does not saturate the
    medium) and the gain recorded. ASE is off by default (small-signal gain is ASE-independent
    and the sweep is much faster); set with_ase=True to include ASE loading."""
    lams = np.atleast_1d(np.asarray(probe_lambda_m, float))
    base = _with(amp, signals=[])
    if not with_ase:
        # THROUGH THE PROTOCOL (audit A-3 follow-on): this used to hand-roll a FiberAmplifier(
        # base.ion, ...) rebuild -- a fourth, divergent copy of the "what a clone must carry"
        # list that read the FiberAmplifier-only `.ion`, dropped the RamanStokes coupling, and
        # had to re-attach `_Tz` by hand. without_ase() drops the ASE band(s) and carries
        # everything else, including the axial temperature profile (audit A-5).
        base = _rebuild(base, "without_ase")
    gdB = np.empty_like(lams)
    for j, lam in enumerate(lams):
        r = _with(base, signals=[Signal(probe_power_W, float(lam))]).solve()
        si = [i for i, k in enumerate(r.kind) if k == "signal"][0]
        gdB[j] = 10.0 * np.log10(r.power_W[si, -1] / r.power_W[si, 0])
    peak = float(lams[int(np.argmax(gdB))])
    flat = float(np.max(gdB) - np.min(gdB))
    tilt = float(np.polyfit(lams * 1e9, gdB, 1)[0]) if lams.size >= 2 else np.nan
    return GainSpectrum(lams, gdB, peak, flat, tilt)


def gain_flatness(result: SteadyStateResult) -> float:
    """Peak-to-peak gain spread [dB] across the signal channels of a single solve (the WDM gain
    flatness). Requires >= 2 signal channels."""
    idx = [i for i, k in enumerate(result.kind) if k == "signal"]
    g = result.signal_gain_dB
    return float(np.max(g) - np.min(g)) if len(idx) >= 2 else 0.0
