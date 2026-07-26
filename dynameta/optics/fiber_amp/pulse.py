"""Pulse propagation for chirped-pulse amplification (CPA): the generalized nonlinear
Schrodinger equation (GNLSE) envelope model that carries a short optical pulse through a
dispersive, Kerr-nonlinear, gain fiber (docs sec.11). This is the Phase-12 CORE -- dispersion +
Kerr + a linear (unsaturated, flat) gain -- solved by the symmetric split-step Fourier method;
Phase 13 adds the saturable, spectrally-shaped gain (with an OPT-IN Frantz-Nodvik temporal law
for the short-pulse regime -- SaturableGain.frantz_nodvik, audit A-10), Phase 14 the
stretcher/compressor chain.

Envelope A(z, t) [sqrt(W)], |A|^2 = instantaneous power [W], in the retarded frame t = t - z/v_g:
    dA/dz = D_hat A + i gamma |A|^2 A,
    D_hat(omega) = i (beta2/2 omega^2 + beta3/6 omega^3) + (g - alpha)/2,
with beta_k the dispersion [s^k/m], gamma = n2 omega0/(c A_eff) the Kerr coefficient [1/W/m],
alpha the loss [1/m], g the (flat, here) gain [1/m]. Sign convention pinned by the fundamental
soliton (beta2 < 0, gamma > 0 -> a shape-preserving bright soliton). Ref: Agrawal, "Nonlinear
Fiber Optics". Pure numpy; SI units. docs/fiber_amp_model_spec.md sec.11.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.optics.fiber_amp.dynamics import frantz_nodvik_instantaneous_gain

__all__ = ["Pulse", "gaussian_pulse", "sech_pulse", "dispersion_length", "nonlinear_length",
           "soliton_order", "propagate_gnlse", "SaturableGain", "raman_response_freq"]


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
        """Angular-frequency grid (baseband, rad/s) matching numpy FFT ordering.

        SIGN (audit V-6/A-9 -- this was undocumented while the OPERATORS in this same module
        documented it): the envelope is propagated in the numpy exp(+i w t) basis, so this
        axis is the NEGATIVE of the physical detuning under the repo-wide exp(-i omega t)
        convention. physical_detuning = -omega_rad_s(). Any ASYMMETRIC spectrum plotted
        against this axis without the flip is MIRRORED about the carrier -- including the
        Raman red shift, which appears at POSITIVE w here. The same flip applies to the
        `beta3` and CPA operators (which state it) and to SaturableGain.center_omega_rad_s /
        gain_omega (which now state it too).
        """
        return 2.0 * np.pi * np.fft.fftfreq(self.t_s.size, self.dt_s)

    def spectrum(self):
        """(omega_shifted [rad/s], spectral power density [arb]) sorted by frequency.

        The axis is the NUMPY frequency variable, i.e. MINUS the physical detuning -- see
        omega_rad_s (audit V-6). Negate it before plotting against physical wavelength."""
        w = np.fft.fftshift(self.omega_rad_s())
        S = np.abs(np.fft.fftshift(np.fft.fft(self.field))) ** 2
        return w, S

    def spectral_fwhm_rad_s(self) -> float:
        """Spectral FWHM [rad/s] on the numpy frequency axis. A WIDTH is sign-symmetric, so
        the V-6 flip does not affect this number -- it affects where the band SITS."""
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


# ---- delayed Raman response ----------------------------------------------------------------

def raman_response_freq(n: int, dt_s: float, model: str = "blow_wood"):
    """(f_R, H(w)) for the silica delayed Raman response on the numpy FFT grid of an n-point,
    dt-spaced time window: the nonlinear response is R(t) = (1-f_R) delta(t) + f_R h_R(t) and
    H(w) is the FFT-convention transform of the causal h_R sampled on [0, n*dt), normalized to
    H(0) = 1 (h_R integrates to one). Models:
      'blow_wood'  : h_R = (tau1^2+tau2^2)/(tau1 tau2^2) exp(-t/tau2) sin(t/tau1), f_R = 0.18,
                     tau1 = 12.2 fs, tau2 = 32 fs (Blow & Wood, IEEE JQE 25:2665, 1989);
      'lin_agrawal': adds the anisotropic boson peak h_b = (2 tau_b - t)/tau_b^2 exp(-t/tau_b),
                     f_R = 0.245, (f_a, f_b, f_c) = (0.75, 0.21, 0.04), tau_b = 96 fs (Lin &
                     Agrawal, Opt. Lett. 31:3086, 2006). f_R is PAIRED to the model -- 0.18
                     belongs to Blow-Wood only, 0.245 to Lin-Agrawal only (dossier warning).
    The frequency-domain convolution (h_R conv |A|^2) = ifft(H * fft(|A|^2)) with THIS pairing
    delays the nonlinear phase behind the intensity (causal lag), which is what red-shifts a
    soliton: to first order I_eff ~ I - T_R dI/dT with T_R = f_R int t h_R dt ~ 3 fs, whose mean
    chirp <d phi/dT> = +gamma T_R int(I')^2/E > 0 in the numpy exp(+i w t) basis = a LOWER
    physical frequency (repo convention exp(-i omega t): physical detuning = -w_numpy)."""
    t = np.arange(int(n)) * float(dt_s)
    tau1, tau2 = 12.2e-15, 32e-15
    ha = (tau1 ** 2 + tau2 ** 2) / (tau1 * tau2 ** 2) * np.exp(-t / tau2) * np.sin(t / tau1)
    if model == "blow_wood":
        f_R, h = 0.18, ha
    elif model == "lin_agrawal":
        tau_b = 96e-15
        hb = (2.0 * tau_b - t) / tau_b ** 2 * np.exp(-t / tau_b)
        f_R, h = 0.245, (0.75 + 0.04) * ha + 0.21 * hb
    else:
        raise ValueError("raman model must be 'blow_wood' or 'lin_agrawal'")
    _guard_raman_sampling(float(dt_s), model)
    H = np.fft.fft(h) * float(dt_s)
    return f_R, H / H[0].real                       # exact H(0)=1 (unit-area response)


# Longest kernel time constant per model: tau2 = 32 fs (Blow-Wood), tau_b = 96 fs (Lin-Agrawal).
# The response must be sampled several times within it or the discrete kernel is not the response
# at all -- see _guard_raman_sampling (audit A-11).
_RAMAN_TAU_MAX_S = {"blow_wood": 32e-15, "lin_agrawal": 96e-15}
_RAMAN_MIN_SAMPLES_PER_TAU = 4.0


def _guard_raman_sampling(dt_s: float, model: str) -> None:
    """audit A-11: `raman_response_freq` had NO time-sampling guard. For dt >~ tau the sampled
    h_R is one or two points, and the H/H[0] normalization then forces H(w) ~ 1 across the whole
    grid -- i.e. the DELAYED Raman response silently degrades to an instantaneous (delta)
    nonlinearity with NO soliton self-frequency shift at all, and nothing anywhere reports it.
    A stretched-pulse CPA run (a nanosecond window sampled at picoseconds) lands squarely in that
    regime. Warn rather than raise: a coarse-dt run is still a legitimate Kerr-only simulation --
    it just is not a Raman one, and the user must know which they got."""
    tau = _RAMAN_TAU_MAX_S.get(model)
    if tau is None or not (dt_s > 0.0):
        return
    n_per_tau = tau / dt_s
    if n_per_tau < _RAMAN_MIN_SAMPLES_PER_TAU:
        warnings.warn(
            "raman_response_freq({!r}): dt = {:.3g} s samples the {:.3g} s Raman kernel only "
            "{:.2g} times (need >~ {:g}). The discrete kernel aliases and the H(0) = 1 "
            "normalization then flattens H(w) to ~1, so the DELAYED Raman response silently "
            "becomes an instantaneous one -- no soliton self-frequency shift, no Raman-induced "
            "red shift, with a plausible-looking result. Refine dt (more time points over the "
            "same window) or drop the Raman term and call the run Kerr-only.".format(
                model, dt_s, tau, n_per_tau, _RAMAN_MIN_SAMPLES_PER_TAU),
            RuntimeWarning, stacklevel=3)


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
    inversion and e_sat_J = dynamics.saturation_energy.

    FREQUENCY CONVENTION (audit A-9): `center_omega_rad_s` and the `omega` argument of
    `g_omega` are on the NUMPY frequency grid (Pulse.omega_rad_s), which is MINUS the physical
    detuning under the repo-wide exp(-i omega t) convention -- the same flip the beta3 and CPA
    operators in this module document. A gain band centred at a physical detuning +Delta0 is
    therefore requested as center_omega_rad_s = -Delta0. The band is symmetric about its
    centre, so `gain_bandwidth_rad_s` is unaffected.

    TEMPORAL LAW (audit A-10). The default is the CW homogeneous law above: `E` is the pulse's
    TOTAL energy at that z, so the saturation is ONE SCALAR multiplying the whole time window
    and the propagator can never produce leading-edge/trailing-edge asymmetry. That is a
    gain-narrowing model, not a pulse-extraction model -- correct for a pulse long against the
    upper-state lifetime, but for the short-pulse (CPA) regime it mis-states the extracted energy
    by up to ~27% in the intermediate saturation regime and reshapes nothing.

      frantz_nodvik=True  (OPT-IN; default OFF -> byte-identical to the CW behaviour)
          replaces the CW factor with the Frantz-Nodvik instantaneous law
          G(t) = G0/(G0 - (G0-1) exp(-U_in(t)/E_sat)) applied in the TIME domain each half-step
          (dynamics.frantz_nodvik_instantaneous_gain -- the repo's single home for the law),
          with G0 = exp(g_small_per_m * dz) the sub-step small-signal gain. The map is exactly
          composable in u = expm1(U/E_sat), so slicing the fiber does not change the answer, and
          the total extracted energy reproduces dynamics.frantz_nodvik_output_energy.
      recovery_time_s     the upper-state lifetime that re-pumps the gain BETWEEN parts of the
          pulse; inf (default) is the no-recovery FN regime (pulse short vs lifetime). Finite
          tau integrates the full gain rate equation dg/dt = (g0-g)/tau - g P/E_sat, whose
          memoryless limit for a pulse LONG against tau is the CW saturation law
          g = g_small/(1 + P tau/E_sat): the CW law's `E` is then the energy delivered within
          one recovery time (P tau), which for a flat-top pulse of duration tau is exactly the
          `E` the default law uses. Ignored when frantz_nodvik is False.

          ORDER COST (audit W5-2 -- read this before choosing a finite tau). The tau = inf branch
          is the EXACT slab map, so the propagator keeps its symmetric-split O(h^2). A FINITE tau
          switches to the thin-slab exponential-Euler form (_fn_slab_gain), which is O(h) -- and
          because it sits inside the step, it drops the WHOLE propagator to first order, not just
          the gain factor. Measured on a 200 fs Gaussian with beta2 = 1 ps^2/m and gamma =
          3e-4 /W/m (self-convergence of the output field in n_steps, 16 -> 256): observed order
          2.04 at tau = inf, 1.06 at tau = 1 ns. Budget ~ (2nd-order n_steps)^2 steps for the same
          field accuracy, or state the first-order error bar.

          NO-RECOVERY SPELLING: use np.inf, not a huge finite number. tau = 1e30 s takes the
          finite-tau branch and therefore inherits its O(h) error even though the physics is
          identical to tau = inf: measured 0.39% energy disagreement at n_steps = 100 on a
          flat-top pulse (7.9798e-6 J vs 7.9487e-6 J), shrinking only as 1/n_steps. `inf` takes
          the exact branch and is n_steps-independent.

          SIGN: a NEGATIVE g_small_per_m (a saturable ABSORBER) is legal on all three branches
          -- CW, tau = inf and finite tau (audit W5-3 fixed the last of those, which used to
          return an all-NaN field with only a RuntimeWarning).

    Spectral shaping composes with the FN mode as a RELATIVE band filter (see propagate_gnlse):
    the band centre takes the full FN gain and the wings are attenuated by the same
    wing/centre ratio the CW law would apply at the FN-realized (saturated) gain, so a flat band
    (shape(omega) == 1 everywhere) leaves the FN result untouched bit for bit. That filter is an
    APPROXIMATION for a non-flat band and it OVER-saturates -- see propagate_gnlse for the
    mechanism and the measured bounds (audit W5-5)."""
    g_small_per_m: float
    e_sat_J: float
    gain_bandwidth_rad_s: float
    shape: str = "parabolic"
    center_omega_rad_s: float = 0.0
    frantz_nodvik: bool = False
    recovery_time_s: float = np.inf

    def shape_omega(self, omega) -> np.ndarray:
        """The normalized band shape alone, in [0, 1] and == 1 at center_omega_rad_s. Split out
        of g_omega so the FN mode can reuse the SAME band definition (audit A-10)."""
        x = (np.asarray(omega, float) - self.center_omega_rad_s) / self.gain_bandwidth_rad_s
        if self.shape == "parabolic":
            # clamped at zero outside the band: the raw 1 - x^2 diverges to -inf in the wings,
            # applying unbounded parasitic LOSS where the physical gain simply goes to zero
            # (audit S3-39). Near the band centre (where the gain-narrowing law lives) the
            # clamp is inactive, so the analytic 1/Omega^2 narrowing gates are unaffected.
            return np.maximum(1.0 - x ** 2, 0.0)
        if self.shape == "lorentzian":
            return 1.0 / (1.0 + x ** 2)
        if self.shape == "gaussian":
            return np.exp(-x ** 2)
        raise ValueError("SaturableGain.shape must be parabolic|lorentzian|gaussian")

    def g_omega(self, energy_J: float, omega) -> np.ndarray:
        sat = 1.0 / (1.0 + energy_J / self.e_sat_J)
        shp = self.shape_omega(omega)
        return self.g_small_per_m * sat * shp


def _gain_with_recovery(t_s, p_W, g0_per_m: float, e_sat_J: float, tau_s: float) -> np.ndarray:
    """Local gain coefficient g(t) [1/m] of a saturable gain WITH re-pumping, from the standard
    rate equation (Siegman ch.10; the finite-lifetime generalization of Frantz-Nodvik)

        dg/dt = (g0 - g)/tau - g P(t)/E_sat,        g(t -> -inf) = g0,

    solved EXACTLY on the pulse's own (possibly non-uniform) time grid for a midpoint-constant
    decay rate a(t) = 1/tau + P(t)/E_sat. Limits: tau -> inf gives g = g0 exp(-U(t)/E_sat), the
    no-recovery law (the propagator uses the exact FN slab form there instead); a pulse LONG
    against tau drives g to the memoryless steady state g0/(1 + P tau/E_sat) -- the CW saturation
    law, with P tau the energy delivered within one recovery time (audit A-10 gate (b)).

    NUMERICS. The integrating-factor solution is
        g_i = exp(-A_i) [g0 + SUM_{j<i} c_j exp(A_{j+1})],  A_i = INT_0^{t_i} a dt,
    whose exp(A) factors overflow for any realistic window (A ~ E_pulse/E_sat + T/tau). The sum
    is therefore accumulated in LOG space with np.logaddexp.accumulate, which is exact to
    rounding for every window length and needs no python loop over the time grid.

    SIGN (audit W5-3). The rate equation is LINEAR in g and its only source term is proportional
    to g0, while the decay rate a(t) = 1/tau + P/E_sat does not involve g0 at all. Hence
    g(t) = g0 * h(t) with h the unit-g0 solution (h(0) = 1) -- so a NEGATIVE g0, i.e. a saturable
    ABSORBER (legal and finite on both the CW and the tau = inf branches), is handled by running
    the log-space algebra on |g0| and restoring the sign at the end. Taking log(g0) directly, as
    this routine used to, made EVERY sample NaN with nothing but a RuntimeWarning, and the NaN
    then propagated silently through the whole field. For g0 >= 0 the restored sign is exactly
    +1.0 and the arithmetic is bit-identical to before; g0 = 0 gives g == 0 (log(0) = -inf ->
    exp(-inf) = 0), which is the correct no-gain limit."""
    t = np.asarray(t_s, float)
    p = np.asarray(p_W, float)
    if t.size < 2:
        return np.full(p.shape, float(g0_per_m))
    g0 = float(g0_per_m)
    sign = -1.0 if g0 < 0.0 else 1.0                    # exactly +-1 -> g0 > 0 stays unchanged
    mag = abs(g0)
    a = 1.0 / tau_s + p / e_sat_J                       # local decay rate [1/s], >= 0
    am = 0.5 * (a[1:] + a[:-1])                         # midpoint rate over each interval
    dA = am * np.diff(t)
    A = np.concatenate(([0.0], np.cumsum(dA)))
    # am is > 0 for every finite tau; it can only vanish in the tau = inf limit over an interval
    # of exactly zero power, where the source term c_j = gss (1 - exp(-dA)) is 0/0 * 0. Clamping
    # the DIVISOR (not the numerator) leaves every am > 0 entry bit-identical and sends those
    # intervals to the correct 0 via -expm1(-0) = 0, so the documented tau -> inf limit
    # g = g0 exp(-U/E_sat) is reproduced instead of NaN.
    gss = (mag / tau_s) / np.where(am > 0.0, am, 1.0)    # |steady state| over each interval
    with np.errstate(divide="ignore"):                  # c_j = 0 (tau = inf) -> log(0) = -inf
        terms = np.concatenate(([np.log(mag)], np.log(gss * (-np.expm1(-dA))) + A[1:]))
    g = sign * np.exp(np.logaddexp.accumulate(terms) - A)
    g[0] = g0                                           # exact IC (the log round-trip costs 1 ulp)
    return g


def _fn_slab_gain(sg: "SaturableGain", t_s, p_W, dz_m: float) -> np.ndarray:
    """Instantaneous POWER-gain factor G(t) of a slab of length dz_m traversed by a pulse of
    instantaneous power p_W(t), for the opt-in Frantz-Nodvik mode (audit A-10). G >= 1 for a gain
    (g_small_per_m > 0); G <= 1 for a saturable ABSORBER (g_small_per_m < 0, legal on every
    branch -- audit W5-3), which bleaches TOWARD unity as the pulse depletes it.

    recovery_time_s = inf: the EXACT Frantz-Nodvik slab solution (dynamics.
    frantz_nodvik_instantaneous_gain with G0 = exp(g_small dz)) -- exact for any dz because the
    FN map composes in u = expm1(U/E_sat). Finite recovery_time_s: the rate equation is solved in
    t at the slab ENTRANCE and the slab applies exp(g(t) dz), the thin-slab (exponential-Euler in
    z) form. That form is FIRST order in dz, and it converges to the FN branch at first order --
    NOT at "the split-step's own O(h^2)", which is what this docstring used to claim (audit W5-1;
    the two statements contradicted each other and the measurement settles it: refining n_steps
    from 25 to 1600 on a flat-top pulse shrinks the tau = 1e6 s vs tau = inf energy gap 1.59e-2 ->
    2.43e-4, an observed order of 1.00, and the shipped gate
    test_fn_recovery_time_recovers_the_cw_saturation_law only ever asserted the honest
    err[1] < 0.35 err[0], i.e. first order with margin). Halving h halves this error; it does not
    quarter it. See SaturableGain.recovery_time_s for what that costs the WHOLE propagator."""
    if not (sg.recovery_time_s > 0.0):
        raise ValueError("SaturableGain.recovery_time_s must be > 0 (inf = the no-recovery "
                         "Frantz-Nodvik limit); got {!r}".format(sg.recovery_time_s))
    if not np.isfinite(sg.recovery_time_s):
        return frantz_nodvik_instantaneous_gain(t_s, p_W, np.exp(sg.g_small_per_m * dz_m),
                                                sg.e_sat_J)
    return np.exp(_gain_with_recovery(t_s, p_W, sg.g_small_per_m, sg.e_sat_J,
                                      sg.recovery_time_s) * dz_m)


# ---- FN window-wrap guard (audit W5-4) -----------------------------------------------------
#
# The Frantz-Nodvik factor needs U(t) = INT_-inf^t P dt', and the code evaluates it as a running
# integral from the LEFT EDGE of the FFT window (dynamics._cumtrapz). That IS U(t) -- but only if
# the pulse is fully CONTAINED in the window, so that "the left edge" and "t = -inf" carry the
# same zero. Everything else in the propagator is periodic (the split-step's linear operator is
# an FFT), so a pulse that straddles the periodic boundary is physically the same pulse; the FN
# factor is the one piece that is NOT roll-equivariant, and it silently treats the wrapped
# leading part as arriving LAST -- i.e. as fully saturated instead of unsaturated.
_FN_EDGE_SAMPLE_DIVISOR = 64                 # samples averaged at each edge: max(1, N // 64)
_FN_EDGE_POWER_TOL = 1.0e-3                  # edge/peak power ratio above which the guard fires


def _fn_edge_power_ratio(p_W) -> float:
    """mean(power over the first and last N//64 samples) / peak power -- the wrap diagnostic."""
    p = np.asarray(p_W, float)
    if p.size == 0:
        return 0.0
    pk = float(p.max())
    if not (pk > 0.0):
        return 0.0
    m = max(1, p.size // _FN_EDGE_SAMPLE_DIVISOR)
    return float(0.5 * (float(p[:m].mean()) + float(p[-m:].mean())) / pk)


def _warn_fn_wrap(ratio: float) -> None:
    warnings.warn(
        "propagate_gnlse (Frantz-Nodvik mode): the pulse is not contained in the time window -- "
        "the power averaged over the first/last {:g}th of the samples is {:.3g} of the peak "
        "(guard fires above {:g}). The FN gain factor integrates U(t) = INT P dt' from the LEFT "
        "EDGE of the periodic FFT window, so a pulse that WRAPS across that boundary has its "
        "leading part treated as the TRAILING part -- fully saturated instead of unsaturated. "
        "Unlike every other operator here (all spectral, hence exactly roll-equivariant) this "
        "one is not, and the error is silent and large: a half-window cyclic shift of a "
        "contained pulse changes the extracted energy by 43.8% (audit W5-4). Re-centre the pulse "
        "in t (np.roll / np.fft.fftshift) or widen the window until the edges are dark.".format(
            _FN_EDGE_SAMPLE_DIVISOR, ratio, _FN_EDGE_POWER_TOL),
        RuntimeWarning, stacklevel=3)


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
                    raman: Optional[str] = None, self_steepening: bool = False,
                    n_steps: int = 400, store_slices: int = 0) -> PropagationResult:
    """Propagate a Pulse through length_m of fiber by the symmetric split-step Fourier method:
    exp(h/2 D) exp(h N) exp(h/2 D) per step, with D the linear (dispersion + gain/loss) operator
    in frequency and N the nonlinear operator in time. Gain options: a flat gain_per_m, a fixed
    spectral gain_omega(omega)->g [1/m], or a SaturableGain (Phase 13) whose spectral gain is
    recomputed each step from the pulse energy at that z. With saturable_gain=None the linear
    operator is precomputed once.

    FRANTZ-NODVIK TEMPORAL RESHAPING (audit A-10; SaturableGain(frantz_nodvik=True), OPT-IN --
    with it OFF this function is byte-identical to the Phase-13 behaviour). The CW saturation
    factor 1/(1+E/e_sat) is one scalar per z, so it cannot reshape a pulse. With the opt-in the
    step becomes the symmetric five-factor split

        exp(h/2 D) . FN(h/2) . exp(h N) . FN(h/2) . exp(h/2 D),

    where D no longer carries the gain and FN(dz) multiplies the field by sqrt(G(t)) with G(t)
    the Frantz-Nodvik instantaneous gain of a dz slab (dynamics.frantz_nodvik_instantaneous_gain,
    the repo's single home for the law) evaluated on the pulse's own RETARDED time axis. The
    leading edge sees the full exp(g_small dz) and the trailing edge sees ~1, so the pulse
    steepens toward earlier t and the extracted energy follows frantz_nodvik_output_energy
    instead of the CW law (which under-extracts by up to ~27% mid-regime). A finite
    SaturableGain.recovery_time_s restores the CW limit for pulses long against the lifetime --
    at the cost of the propagator's ORDER: the tau = inf FN map is exact in dz, so the split stays
    O(h^2) (measured 2.04), while a FINITE tau swaps in the thin-slab exponential-Euler gain and
    drops the WHOLE step to O(h) (measured 1.06 on the same dispersion + Kerr case; audit W5-2).
    Spell "no recovery" as np.inf, not 1e30: a huge finite tau takes the first-order branch and
    disagrees with the exact one by 0.39% at n_steps = 100.

    WINDOW CONTAINMENT (audit W5-4). The FN factor is the one operator here that is NOT
    roll-equivariant: it builds U(t) = INT P dt' from the LEFT EDGE of the periodic FFT window, so
    it presumes the pulse is fully contained in that window. A pulse straddling the boundary has
    its leading part treated as trailing (saturated instead of unsaturated) -- a 43.8% energy
    error on a half-window cyclic shift, silently. FN mode therefore checks the edge power every
    half-step and warns once (RuntimeWarning) if it exceeds 1e-3 of the peak; keep the pulse
    centred and the window wide enough that its edges stay dark.

    A non-flat gain band composes with FN as a RELATIVE filter: after the time-domain FN factor,
    the field is multiplied by exp[0.5 g_realized (shape(omega) - 1) dz] with g_realized =
    ln(E_out/E_in)/dz the gain coefficient the FN step ACTUALLY realized. That reproduces the
    wing/centre ratio of the CW law at the same saturation (so gain narrowing stops as the
    amplifier saturates) and is exactly the identity for a flat band.

    ... and it is an APPROXIMATION whenever the band is NOT flat (audit W5-5). The exact
    homogeneously-broadened short-pulse system for a cross-section sigma(w) = sigma0 shape(w) is
        dP_i/dz = s_i n P_i,   dn/dt = -(n/E_sat) SUM_j s_j P_j,   n(t -> -inf) = g_small,
    i.e. ONE shared inversion depleted by the SHAPE-WEIGHTED total power. Two consequences:
      * ln G_i = s_i INT n dz, so the RATIO of log-gains between two spectral components is
        exactly s_i/s_j -- which the relative filter reproduces EXACTLY
        (g_real dz + g_real (s-1) dz = s g_real dz). The gain-narrowing physics is right.
      * the saturation variable here is the UNWEIGHTED total power, at the CENTRE small-signal
        gain, because the FN factor is applied before the filter. Wing energy therefore depletes
        the inversion as if it sat at the band centre, so the model OVER-saturates and
        UNDER-predicts the extracted energy whenever any pulse energy sits where shape < 1.
    Measured against a two-line RK4 oracle of the system above (parabolic band, Omega =
    3e13 rad/s): exact (to 1e-4) with all energy at the band centre; -0.69% at the shipped CPA
    example's exposure (energy-weighted mean shape 0.975, E_in = 0.5 E_sat -- pinned in
    tests/test_fiber_amp.py); -0.91% at the same band with E_in = 3 E_sat; -13.4% in a designed
    worst case (half the energy at shape = 0.305, E_in = 3 E_sat). Use a flat band, or the CW
    path, if you need the wing energetics to better than that.

    Nonlinear completion terms (both OFF by default -> the pure-Kerr N = i gamma |A|^2 A path,
    byte-identical to the Phase-12 propagator):
      * raman = 'blow_wood' | 'lin_agrawal': the DELAYED Raman response (raman_response_freq),
        N = i gamma A [(1-f_R)|A|^2 + f_R h_R conv |A|^2]. The convolution is a TIME-domain
        operation, so it needs no numpy-vs-physical sign gymnastics (unlike the spectral beta3
        term); its causal lag is what produces the soliton self-frequency red shift at Gordon's
        rate d nu/dz = -(4/(15 pi)) |beta2| T_R / T0^4 (Gordon, Opt. Lett. 11:662, 1986).
      * self_steepening=True: the optical-shock prefactor, N -> i gamma (1 + (i/omega0) d/dT)
        [A I_eff]. d/dT is implemented spectrally as multiplication by (+i w) -- the derivative
        rule OF THE NUMPY exp(+i w t) BASIS, which keeps the time-domain operator physical with
        no explicit sign flip. The shock steepens the TRAILING edge (peak drifts to later T) --
        gated in tests. With the shock on, the nonlinear step integrates by RK4 (the operator is
        no longer a pure phase); without it, the exact unitary phase step is kept.
    Conservation contract (tests): pure Kerr conserves energy AND photon number; Raman-only
    conserves energy exactly (real phase) while red-shifting the spectrum; Raman + shock is the
    photon-number-conserving, energy-decreasing form (Blow-Wood).

    Returns the output pulse, the accumulated B-integral, and optionally store_slices
    field snapshots along z."""
    A = pulse.field.astype(np.complex128).copy()
    w = pulse.omega_rad_s()
    dt = pulse.dt_s
    # numpy's ifft reconstructs A(t) = sum A_hat(w) exp(+i w t), so a physical envelope component
    # exp(-i Delta t) (repo convention exp(-i omega t), Agrawal beta_k) sits at numpy frequency
    # w = -Delta. Even orders are immune; ODD orders must flip sign in numpy-w variables so that
    # a literature beta3 > 0 produces the physical trailing-edge oscillations (audit S3-11: the
    # old +beta3/6 w^3 mirrored the TOD asymmetry for external inputs).
    D_fixed = 1j * (beta2_s2_m / 2.0 * w ** 2 - beta3_s3_m / 6.0 * w ** 3) - 0.5 * loss_per_m
    if gain_omega is not None:
        D_fixed = D_fixed + 0.5 * np.asarray(gain_omega(w), np.complex128)
    elif gain_per_m:
        D_fixed = D_fixed + 0.5 * gain_per_m
    h = length_m / n_steps
    # audit A-10: in FN mode the gain leaves the (frequency-domain) linear operator entirely and
    # becomes a time-domain factor, so the linear half-step is static again.
    fn_active = saturable_gain is not None and bool(saturable_gain.frantz_nodvik)
    cw_saturable = saturable_gain is not None and not fn_active
    half_static = None if cw_saturable else np.exp(D_fixed * (h / 2.0))
    b_int = 0.0
    if fn_active:
        fn_shape = saturable_gain.shape_omega(w)
        fn_flat_band = bool(np.all(fn_shape == 1.0))
        fn_dz = h / 2.0
        fn_t = pulse.t_s
        fn_wrap_warned = False

        def _fn_half(A_now):
            """FN gain over half a z-step: sqrt(G(t)) in time, then the relative band filter."""
            nonlocal fn_wrap_warned
            P_now = np.abs(A_now) ** 2
            if not fn_wrap_warned:                      # W5-4: U(t) starts at the window edge
                ratio = _fn_edge_power_ratio(P_now)
                if ratio > _FN_EDGE_POWER_TOL:
                    fn_wrap_warned = True               # once per propagate_gnlse call
                    _warn_fn_wrap(ratio)
            G_t = _fn_slab_gain(saturable_gain, fn_t, P_now, fn_dz)
            A_next = A_now * np.sqrt(G_t)
            if fn_flat_band:
                return A_next
            e_in = float(np.sum(P_now) * dt)
            e_out = float(np.sum(P_now * G_t) * dt)
            g_real = np.log(e_out / e_in) / fn_dz if (e_in > 0.0 and e_out > 0.0) else 0.0
            return np.fft.ifft(np.exp(0.5 * g_real * (fn_shape - 1.0) * fn_dz)
                               * np.fft.fft(A_next))

    raman_active = raman is not None
    if raman_active:
        f_R, H_R = raman_response_freq(A.size, dt, raman)
    omega0 = 2.0 * np.pi * C_LIGHT / pulse.lambda0_m

    def _I_eff(p_now):
        """(1-f_R)|A|^2 + f_R (h_R conv |A|^2); plain |A|^2 when Raman is off."""
        if not raman_active:
            return p_now
        conv = np.real(np.fft.ifft(H_R * np.fft.fft(p_now)))
        return (1.0 - f_R) * p_now + f_R * conv

    def _N_op(A_now):
        """Full nonlinear operator i gamma (1 + (i/omega0) d/dT)[A I_eff] for the RK4 path."""
        Q = A_now * _I_eff(np.abs(A_now) ** 2)
        dQ = np.fft.ifft((1j * w) * np.fft.fft(Q))
        return 1j * gamma_W_m * (Q + (1j / omega0) * dQ)

    slices, zs = [], []
    store_every = max(1, n_steps // store_slices) if store_slices else 0

    for k in range(n_steps):
        if cw_saturable:
            E = float(np.sum(np.abs(A) ** 2) * dt)               # pulse energy at this z
            half = np.exp((D_fixed + 0.5 * saturable_gain.g_omega(E, w)) * (h / 2.0))
        else:
            half = half_static
        A = np.fft.ifft(half * np.fft.fft(A))
        if fn_active:
            A = _fn_half(A)
        p = np.abs(A) ** 2
        b_int += gamma_W_m * float(p.max()) * h
        if self_steepening:
            k1 = _N_op(A)                                        # RK4: N is not a pure phase
            k2 = _N_op(A + 0.5 * h * k1)
            k3 = _N_op(A + 0.5 * h * k2)
            k4 = _N_op(A + h * k3)
            A = A + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif raman_active:
            A = A * np.exp(1j * gamma_W_m * _I_eff(p) * h)       # exact unitary (real I_eff)
        else:
            A = A * np.exp(1j * gamma_W_m * p * h)               # legacy pure-Kerr path
        if fn_active:
            A = _fn_half(A)
        A = np.fft.ifft(half * np.fft.fft(A))
        if store_every and (k % store_every == 0):
            slices.append(A.copy())
            zs.append((k + 1) * h)

    out = Pulse(pulse.t_s.copy(), A, pulse.lambda0_m)
    meta = {"n_steps": n_steps, "h_m": h}
    if fn_active:                                                # audit A-10 provenance
        meta["frantz_nodvik"] = True
        meta["fn_small_signal_gain"] = float(np.exp(saturable_gain.g_small_per_m * length_m))
        meta["fn_recovery_time_s"] = float(saturable_gain.recovery_time_s)
    return PropagationResult(out, np.asarray(zs) if slices else None,
                             np.asarray(slices) if slices else None, float(b_int), meta=meta)
