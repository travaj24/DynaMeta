"""Pulse propagation for chirped-pulse amplification (CPA): the generalized nonlinear
Schrodinger equation (GNLSE) envelope model that carries a short optical pulse through a
dispersive, Kerr-nonlinear, gain fiber (docs sec.11). This is the Phase-12 CORE -- dispersion +
Kerr + a linear (unsaturated, flat) gain -- solved by the symmetric split-step Fourier method;
Phase 13 adds the saturable, spectrally-shaped gain, Phase 14 the stretcher/compressor chain.

Envelope A(z, t) [sqrt(W)], |A|^2 = instantaneous power [W], in the retarded frame t = t - z/v_g:
    dA/dz = D_hat A + i gamma |A|^2 A,
    D_hat(omega) = i (beta2/2 omega^2 + beta3/6 omega^3) + (g - alpha)/2,
with beta_k the dispersion [s^k/m], gamma = n2 omega0/(c A_eff) the Kerr coefficient [1/W/m],
alpha the loss [1/m], g the (flat, here) gain [1/m]. Sign convention pinned by the fundamental
soliton (beta2 < 0, gamma > 0 -> a shape-preserving bright soliton). Ref: Agrawal, "Nonlinear
Fiber Optics". Pure numpy; SI units. docs/fiber_amp_model_spec.md sec.11.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = ["Pulse", "gaussian_pulse", "sech_pulse", "dispersion_length", "nonlinear_length",
           "soliton_order", "propagate_gnlse", "SaturableGain"]


@dataclass
class Pulse:
    """A complex optical-field envelope on a uniform time grid. t_s [s] (length N, uniform);
    field [sqrt(W)] complex (|field|^2 = power [W]); lambda0_m the carrier wavelength."""
    t_s: np.ndarray
    field: np.ndarray
    lambda0_m: float

    @property
    def dt_s(self) -> float:
        return float(self.t_s[1] - self.t_s[0])

    @property
    def power_W(self) -> np.ndarray:
        return np.abs(self.field) ** 2

    @property
    def energy_J(self) -> float:
        return float(np.sum(self.power_W) * self.dt_s)

    @property
    def peak_power_W(self) -> float:
        return float(np.max(self.power_W))

    def fwhm_s(self) -> float:
        """Full width at half maximum of the intensity envelope [s] (linear interpolation)."""
        return _fwhm(self.t_s, self.power_W)

    def omega_rad_s(self) -> np.ndarray:
        """Angular-frequency grid (baseband, rad/s) matching numpy FFT ordering."""
        return 2.0 * np.pi * np.fft.fftfreq(self.t_s.size, self.dt_s)

    def spectrum(self):
        """(omega_shifted [rad/s], spectral power density [arb]) sorted by frequency."""
        w = np.fft.fftshift(self.omega_rad_s())
        S = np.abs(np.fft.fftshift(np.fft.fft(self.field))) ** 2
        return w, S

    def spectral_fwhm_rad_s(self) -> float:
        w, S = self.spectrum()
        return _fwhm(w, S)

    def copy(self) -> "Pulse":
        return Pulse(self.t_s.copy(), self.field.copy(), self.lambda0_m)


def _fwhm(x, y) -> float:
    y = np.asarray(y, float)
    ypk = y.max()
    if ypk <= 0.0:
        return 0.0
    half = 0.5 * ypk
    above = np.where(y >= half)[0]
    if above.size < 2:
        return 0.0
    i0, i1 = above[0], above[-1]
    # linear-interpolate the crossings for sub-sample accuracy
    xl = x[i0] if i0 == 0 else np.interp(half, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]])
    xr = x[i1] if i1 == y.size - 1 else np.interp(half, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]])
    return float(abs(xr - xl))


# ---- pulse builders ------------------------------------------------------------------------

def gaussian_pulse(t_s, *, t0_s: float, peak_power_W: float = None, energy_J: float = None,
                   lambda0_m: float = 1.03e-6, chirp: float = 0.0) -> Pulse:
    """Transform-limited (or linearly-chirped) Gaussian pulse A = sqrt(P0) exp[-(1+i C)/2 (t/t0)^2].
    t0_s is the 1/e half-width (FWHM = 2 sqrt(ln2) t0). Give peak_power_W or energy_J (for a
    Gaussian, energy = P0 t0 sqrt(pi)). chirp C is the dimensionless linear chirp parameter."""
    t = np.asarray(t_s, float)
    if peak_power_W is None:
        if energy_J is None:
            raise ValueError("gaussian_pulse: give peak_power_W or energy_J")
        peak_power_W = energy_J / (t0_s * np.sqrt(np.pi))
    env = np.sqrt(peak_power_W) * np.exp(-(1.0 + 1j * chirp) / 2.0 * (t / t0_s) ** 2)
    return Pulse(t, env.astype(np.complex128), lambda0_m)


def sech_pulse(t_s, *, t0_s: float, peak_power_W: float = None, energy_J: float = None,
               lambda0_m: float = 1.03e-6) -> Pulse:
    """Hyperbolic-secant pulse A = sqrt(P0) sech(t/t0) (the soliton shape). For a sech,
    energy = 2 P0 t0 and FWHM = 1.7627 t0."""
    t = np.asarray(t_s, float)
    if peak_power_W is None:
        if energy_J is None:
            raise ValueError("sech_pulse: give peak_power_W or energy_J")
        peak_power_W = energy_J / (2.0 * t0_s)
    env = np.sqrt(peak_power_W) / np.cosh(t / t0_s)
    return Pulse(t, env.astype(np.complex128), lambda0_m)


# ---- characteristic lengths ----------------------------------------------------------------

def dispersion_length(t0_s: float, beta2_s2_m: float) -> float:
    """L_D = t0^2 / |beta2| [m]: the length over which dispersion appreciably broadens a pulse."""
    return float(t0_s ** 2 / abs(beta2_s2_m))


def nonlinear_length(peak_power_W: float, gamma_W_m: float) -> float:
    """L_NL = 1 / (gamma P0) [m]: the length over which SPM imprints ~1 rad of nonlinear phase."""
    return float(1.0 / (gamma_W_m * peak_power_W))


def soliton_order(t0_s: float, peak_power_W: float, beta2_s2_m: float, gamma_W_m: float) -> float:
    """Soliton order N = sqrt(L_D / L_NL) = sqrt(gamma P0 t0^2 / |beta2|). N=1 -> fundamental."""
    return float(np.sqrt(dispersion_length(t0_s, beta2_s2_m)
                         / nonlinear_length(peak_power_W, gamma_W_m)))


# ---- the GNLSE split-step propagator -------------------------------------------------------

@dataclass(frozen=True)
class SaturableGain:
    """A saturable, spectrally-shaped fiber-amplifier gain for the GNLSE (Phase 13). The local
    power-gain coefficient is
        g(omega, E) = g_small_per_m * 1/(1 + E/e_sat_J) * shape(omega),
    with E the pulse energy at the current z (homogeneous saturation) and shape a normalized band
    of half-width gain_bandwidth_rad_s about center_omega_rad_s: 'parabolic' 1-x^2 (the analytic
    gain-narrowing model), 'lorentzian' 1/(1+x^2), or 'gaussian' exp(-x^2). A short pulse's
    spectrum is progressively NARROWED as its wings see less gain than the centre -- the effect
    that bounds the recompressed CPA pulse duration. Couple to the CW model via g_small from the
    inversion and e_sat_J = dynamics.saturation_energy."""
    g_small_per_m: float
    e_sat_J: float
    gain_bandwidth_rad_s: float
    shape: str = "parabolic"
    center_omega_rad_s: float = 0.0

    def g_omega(self, energy_J: float, omega) -> np.ndarray:
        sat = 1.0 / (1.0 + energy_J / self.e_sat_J)
        x = (np.asarray(omega, float) - self.center_omega_rad_s) / self.gain_bandwidth_rad_s
        if self.shape == "parabolic":
            shp = 1.0 - x ** 2
        elif self.shape == "lorentzian":
            shp = 1.0 / (1.0 + x ** 2)
        elif self.shape == "gaussian":
            shp = np.exp(-x ** 2)
        else:
            raise ValueError("SaturableGain.shape must be parabolic|lorentzian|gaussian")
        return self.g_small_per_m * sat * shp


@dataclass
class PropagationResult:
    output: Pulse
    z_m: np.ndarray = None                 # stored slice positions (if store_slices)
    field_zt: np.ndarray = None            # (n_slice, N) complex field along z
    b_integral_rad: float = 0.0            # accumulated peak nonlinear phase gamma INT P_pk dz
    meta: dict = field(default_factory=dict)


def propagate_gnlse(pulse: Pulse, length_m: float, *, beta2_s2_m: float = 0.0,
                    beta3_s3_m: float = 0.0, gamma_W_m: float = 0.0, loss_per_m: float = 0.0,
                    gain_per_m: float = 0.0, gain_omega: Optional[Callable] = None,
                    saturable_gain: Optional["SaturableGain"] = None,
                    n_steps: int = 400, store_slices: int = 0) -> PropagationResult:
    """Propagate a Pulse through length_m of fiber by the symmetric split-step Fourier method:
    exp(h/2 D) exp(h N) exp(h/2 D) per step, with D the linear (dispersion + gain/loss) operator
    in frequency and N = i gamma |A|^2 the Kerr operator in time. Gain options: a flat gain_per_m,
    a fixed spectral gain_omega(omega)->g [1/m], or a SaturableGain (Phase 13) whose spectral gain
    is recomputed each step from the pulse energy at that z (energy grows -> gain saturates). With
    saturable_gain=None the linear operator is precomputed once. Returns the output pulse, the
    accumulated B-integral, and optionally store_slices field snapshots along z."""
    A = pulse.field.astype(np.complex128).copy()
    w = pulse.omega_rad_s()
    dt = pulse.dt_s
    D_fixed = 1j * (beta2_s2_m / 2.0 * w ** 2 + beta3_s3_m / 6.0 * w ** 3) - 0.5 * loss_per_m
    if gain_omega is not None:
        D_fixed = D_fixed + 0.5 * np.asarray(gain_omega(w), np.complex128)
    elif gain_per_m:
        D_fixed = D_fixed + 0.5 * gain_per_m
    h = length_m / n_steps
    half_static = None if saturable_gain is not None else np.exp(D_fixed * (h / 2.0))
    b_int = 0.0

    slices, zs = [], []
    store_every = max(1, n_steps // store_slices) if store_slices else 0

    for k in range(n_steps):
        if saturable_gain is not None:
            E = float(np.sum(np.abs(A) ** 2) * dt)               # pulse energy at this z
            half = np.exp((D_fixed + 0.5 * saturable_gain.g_omega(E, w)) * (h / 2.0))
        else:
            half = half_static
        A = np.fft.ifft(half * np.fft.fft(A))
        p = np.abs(A) ** 2
        b_int += gamma_W_m * float(p.max()) * h
        A = A * np.exp(1j * gamma_W_m * p * h)
        A = np.fft.ifft(half * np.fft.fft(A))
        if store_every and (k % store_every == 0):
            slices.append(A.copy())
            zs.append((k + 1) * h)

    out = Pulse(pulse.t_s.copy(), A, pulse.lambda0_m)
    return PropagationResult(out, np.asarray(zs) if slices else None,
                             np.asarray(slices) if slices else None, float(b_int),
                             meta={"n_steps": n_steps, "h_m": h})
