"""Chirped-pulse amplification (CPA) chain and its quality metrics (docs sec.11). A CPA system
lowers the peak power inside the gain fiber by STRETCHING the seed with a large chromatic
dispersion, amplifying the long low-peak-power pulse (keeping the accumulated nonlinear phase --
the B-integral -- small), then RECOMPRESSING with the opposite dispersion:

    seed -> stretcher(+GDD) -> amplifier(gain-GNLSE) -> compressor(-GDD) -> compressed pulse.

The stretcher/compressor are lumped dispersive elements: pure spectral phase
exp[i(gdd/2 omega^2 + tod/6 omega^3)]. The recompressed pulse quality is set by (a) residual
uncompensated dispersion, (b) the B-integral (nonlinear phase a linear compressor cannot undo),
and (c) gain narrowing (Phase 13) shrinking the spectrum and thus lengthening the transform
limit. The Strehl ratio (compressed peak / transform-limited peak) quantifies it. Pure
numpy/scipy; SI units. Ref: Strickland & Mourou; Agrawal. docs/fiber_amp_model_spec.md sec.11.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynameta.optics.fiber_amp.pulse import Pulse, propagate_gnlse, SaturableGain

__all__ = ["apply_spectral_phase", "transform_limited", "strehl_ratio", "CPAResult", "cpa_chain"]


def apply_spectral_phase(pulse: Pulse, gdd_s2: float = 0.0, tod_s3: float = 0.0) -> Pulse:
    """Apply a lumped spectral phase phi(omega) = gdd/2 omega^2 + tod/6 omega^3 (a stretcher or
    compressor / grating pair): Afield -> IFFT[exp(i phi) FFT[Afield]]. gdd = group-delay
    dispersion [s^2], tod = third-order dispersion [s^3]."""
    w = pulse.omega_rad_s()
    phi = gdd_s2 / 2.0 * w ** 2 + tod_s3 / 6.0 * w ** 3
    A = np.fft.ifft(np.exp(1j * phi) * np.fft.fft(pulse.field))
    return Pulse(pulse.t_s.copy(), A, pulse.lambda0_m)


def transform_limited(pulse: Pulse) -> Pulse:
    """The transform-limited pulse of the same power spectrum -- zero spectral phase, the shortest
    / highest-peak pulse the spectrum supports (centred at t=0). Its peak power is the ceiling the
    compressor is trying to reach."""
    mag = np.abs(np.fft.fft(pulse.field))
    A = np.fft.fftshift(np.fft.ifft(mag))     # zero-phase -> real, symmetric, centred
    return Pulse(pulse.t_s.copy(), A.astype(np.complex128), pulse.lambda0_m)


def strehl_ratio(pulse: Pulse) -> float:
    """Compression quality: peak power of the pulse / peak power of its transform limit, in [0,1].
    1.0 = fully compressed (flat spectral phase); < 1 = residual dispersion or nonlinear phase."""
    tl = transform_limited(pulse)
    return float(pulse.peak_power_W / tl.peak_power_W)


@dataclass
class CPAResult:
    seed: Pulse
    stretched: Pulse
    amplified: Pulse
    compressed: Pulse
    b_integral_rad: float
    energy_gain_dB: float
    strehl: float
    compressed_fwhm_s: float
    transform_limited_fwhm_s: float
    stretch_factor: float                  # stretched FWHM / seed FWHM
    meta: dict = field(default_factory=dict)


def cpa_chain(seed: Pulse, *, stretch_gdd_s2: float, amp_length_m: float,
              beta2_s2_m: float = 0.0, beta3_s3_m: float = 0.0, gamma_W_m: float = 0.0,
              gain_per_m: float = 0.0, saturable_gain: Optional[SaturableGain] = None,
              compress_gdd_s2: Optional[float] = None, compress_tod_s3: float = 0.0,
              n_steps: int = 400) -> CPAResult:
    """Run seed -> stretch(+GDD) -> amplify(gain-GNLSE) -> compress through the amplifier. The
    amplifier is length amp_length_m with beta2/beta3 dispersion, Kerr gamma, and either a flat
    gain_per_m or a SaturableGain. If compress_gdd_s2 is None the compressor is set to undo the
    stretcher plus the amplifier's linear GDD: -(stretch_gdd + beta2 * amp_length). Returns every
    stage, the B-integral, and the recompressed Strehl / duration."""
    stretched = apply_spectral_phase(seed, stretch_gdd_s2)
    amp = propagate_gnlse(stretched, amp_length_m, beta2_s2_m=beta2_s2_m, beta3_s3_m=beta3_s3_m,
                          gamma_W_m=gamma_W_m, gain_per_m=gain_per_m,
                          saturable_gain=saturable_gain, n_steps=n_steps)
    if compress_gdd_s2 is None:
        compress_gdd_s2 = -(stretch_gdd_s2 + beta2_s2_m * amp_length_m)
    compressed = apply_spectral_phase(amp.output, compress_gdd_s2, compress_tod_s3)
    tl = transform_limited(compressed)
    return CPAResult(
        seed=seed, stretched=stretched, amplified=amp.output, compressed=compressed,
        b_integral_rad=amp.b_integral_rad,
        energy_gain_dB=float(10.0 * np.log10(amp.output.energy_J / seed.energy_J)),
        strehl=strehl_ratio(compressed), compressed_fwhm_s=compressed.fwhm_s(),
        transform_limited_fwhm_s=tl.fwhm_s(),
        stretch_factor=float(stretched.fwhm_s() / seed.fwhm_s()) if seed.fwhm_s() > 0 else np.nan,
        meta={"compress_gdd_s2": float(compress_gdd_s2)})
