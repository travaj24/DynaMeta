"""Transverse 2-D (x-z) gain-coupled beam-propagation model for a broad-area QD-SOA -- the lateral
axis the 1-D longitudinal engine lumps into the confinement factor Gamma and modal area A_mode. A
split-step Fresnel BPM with a saturable complex gain and lateral carrier diffusion, it resolves the
physics the 1-D model cannot: diffraction, gain guiding, transverse spatial hole burning, and
alpha-driven self-focusing / FILAMENTATION (a key beam-quality limit of high-power SOAs).

SCOPE (honest): a CW STEADY 2-D (x-z) model -- there is NO z-t time axis (no transient / pulse). The
carrier is ADIABATICALLY eliminated into the saturable gain g0/(1+S/Isat), with lateral diffusion
folded in as the steady ambipolar Green's function (below) -- NOT a time-dependent 2-D carrier
drift-diffusion solve. It is a STANDALONE phenomenological saturable-gain BPM: it does NOT couple the
QDGainModel rate equations into 2-D (the filamentation / beam-quality axis, not the full QD physics in
2-D). Paraxial (small-angle) Fresnel; PERIODIC (FFT) lateral boundary -- not a real ridge / buried-
heterostructure waveguide (no guided transverse mode).

Field (paraxial, exp(-i omega t), propagation along z, lateral x):

    dA/dz = (i / 2k) d2A/dx2  +  0.5 (Gamma g(x) (1 - i alpha) - alpha_i) A,   k = 2 pi n0 / lambda.

The first term is DIFFRACTION; g(x) is the local saturable gain; alpha the linewidth-enhancement
factor (the gain -> index coupling that drives filamentation); alpha_i the background loss.

Gain + lateral carrier diffusion:

    g(x) = g0 / (1 + S_eff(x) / Isat),

with S_eff the carrier response to the local intensity |A(x)|^2 -- the steady ambipolar-diffusion
Green's function applied in Fourier as S_eff_k = |A|^2_k / (1 + (L_diff k_x)^2), L_diff = sqrt(D tau)
the carrier diffusion length. Lateral diffusion SMOOTHS the spatial hole burning over L_diff, which
SUPPRESSES filamentation (washes out the carrier/index ripple the alpha coupling would self-focus).

Marched by the symmetric (Strang) split D(dz/2) . N(dz) . D(dz/2) per z-step: D the EXACT unitary
spectral diffraction phase (FFT in x), N the saturable-gain multiply with a MIDPOINT-corrected gain
(re-evaluated on the half-advanced field) so the nonlinear leg is 2nd-order too -- the whole split then
converges as O(dz^2) (verified ratio 4.0). Reduces to the 1-D saturable-gain ODE for a laterally-
uniform beam (only the k_x = 0 component, which diffraction leaves untouched). Pure numpy; SI; ASCII.
(Broad-area-SOA filamentation; Marciante-Agrawal / Hess -- cited as background, NOT validated against
the analytic modulational-instability gain spectrum; the gates test the filament-band growth + the
diffusion-suppression direction.)
"""
from __future__ import annotations

import numpy as np


class TransverseBPM:
    """2-D (x-z) paraxial gain-coupled BPM. Lx_m x nx is the transverse grid (periodic FFT); lambda0_m
    the vacuum wavelength, n0 the background index. Saturable gain g0/(1+S/Isat) with linewidth
    enhancement alpha_lef and lateral carrier diffusion length L_diff_m; alpha_i_per_m the background
    loss; gamma_confinement multiplies the gain (1.0 if g0 is already modal). g0=0 -> a passive
    diffractor. CW STEADY 2-D model (no time axis), adiabatic saturable-gain carrier (not a 2-D DD
    solve), periodic (FFT) lateral boundary (not a guided-mode waveguide) -- see the module docstring."""

    def __init__(self, Lx_m, nx, lambda0_m, n0, *, g0_per_m=0.0, gamma_confinement=1.0,
                 alpha_i_per_m=0.0, Isat_W=np.inf, alpha_lef=0.0, L_diff_m=0.0,
                 qd_gain_table=None):
        if nx < 4 or Lx_m <= 0.0:
            raise ValueError("TransverseBPM: need nx >= 4 and Lx_m > 0")
        if g0_per_m < 0.0 or alpha_i_per_m < 0.0 or Isat_W <= 0.0 or L_diff_m < 0.0:
            raise ValueError("TransverseBPM: g0, alpha_i >= 0; Isat > 0; L_diff >= 0")
        self.nx = int(nx)
        self.Lx = float(Lx_m)
        self.dx = self.Lx / self.nx
        self.x = (np.arange(self.nx) - self.nx // 2) * self.dx
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.k = 2.0 * np.pi * float(n0) / float(lambda0_m)        # medium wavenumber
        self.k0 = 2.0 * np.pi / float(lambda0_m)                   # vacuum wavenumber (index lens)
        self.n0 = float(n0)
        self.g0 = float(g0_per_m)
        self.gam = float(gamma_confinement)
        self.alpha_i = float(alpha_i_per_m)
        self.Isat = float(Isat_W)
        self.alpha = float(alpha_lef)
        self.L_diff = float(L_diff_m)
        self._diff_lp = 1.0 / (1.0 + (self.L_diff * self.kx) ** 2)  # carrier-diffusion low-pass
        # QD-COUPLED saturable gain: a precomputed (P_grid, g_grid) table g_QD(P) [material gain 1/m vs
        # local guided power, from QDGainModel.saturation_curve / qd_gain_table] that REPLACES the
        # phenomenological g0/(1+S/Isat) -- so the 2-D field saturates the REAL group-resolved QD gain
        # (with its WL/ES reservoir and inhomogeneous broadening), the filamentation feedback the
        # standalone toy lacked. None -> the phenomenological form (byte-identical).
        if qd_gain_table is None:
            self._qd_P = None
            self._qd_g = None
        else:
            # the table REPLACES the phenomenological g0/(1+S/Isat) entirely -- passing both is a
            # contradiction (the g0/Isat pair would be silently ignored), so reject it loudly.
            if g0_per_m > 0.0 or np.isfinite(Isat_W):
                raise ValueError("TransverseBPM: qd_gain_table REPLACES the phenomenological saturable "
                                 "gain -- do not also pass g0_per_m > 0 or a finite Isat_W (they would "
                                 "be ignored); set only one gain model. (alpha_lef/alpha_i/L_diff/"
                                 "gamma_confinement still apply with a table.)")
            self._qd_P = np.asarray(qd_gain_table[0], dtype=np.float64)
            self._qd_g = np.asarray(qd_gain_table[1], dtype=np.float64)
            if (self._qd_P.ndim != 1 or self._qd_P.size < 2 or self._qd_g.shape != self._qd_P.shape
                    or np.any(np.diff(self._qd_P) <= 0.0)):
                raise ValueError("TransverseBPM: qd_gain_table = (P_grid, g_grid), P_grid strictly "
                                 "increasing 1-D >= 2 points, g_grid the same shape")

    def _diffract(self, A, dz):
        """Exact unitary paraxial diffraction over dz: A_k *= exp(-i k_x^2 dz / 2k)."""
        return np.fft.ifft(np.fft.fft(A) * np.exp(-1j * self.kx ** 2 * dz / (2.0 * self.k)))

    def carrier_gain(self, A):
        """Local saturable MATERIAL gain g(x) [1/m] given the field A(x). With a qd_gain_table, g(x) is
        the REAL QD saturable gain g_QD interpolated at the local diffusion-smoothed power S_eff(x)
        (clamped to the table range) -- the group-resolved QD saturation. Otherwise the phenomenological
        g0/(1 + S_eff/Isat). S_eff = |A|^2 lateral-diffusion-smoothed (L_diff = 0 -> S_eff = |A|^2)."""
        S = np.abs(A) ** 2
        if self.L_diff > 0.0:
            S = np.fft.ifft(np.fft.fft(S) * self._diff_lp).real
        if self._qd_P is not None:                            # QD-coupled saturable gain (interp table)
            return np.interp(S, self._qd_P, self._qd_g)       # flat-clamped outside the table range
        return self.g0 / (1.0 + S / self.Isat)

    def _coef(self, A):
        """Per-length amplitude coefficient 0.5 (Gamma g(A)(1 - i alpha) - alpha_i)."""
        return 0.5 * (self.gam * self.carrier_gain(A) * (1.0 - 1j * self.alpha) - self.alpha_i)

    def _gain(self, A, dz):
        """Saturable-gain amplitude sub-step over dz, MIDPOINT-corrected to 2nd order: the gain is
        re-evaluated on the half-advanced field so the nonlinear leg matches the Strang split's
        2nd-order accuracy (frozen-g exponential-Euler would cap the whole method at 1st order)."""
        A_half = A * np.exp(0.5 * self._coef(A) * dz)        # predictor: half-step at the start gain
        arg = self._coef(A_half) * dz                        # full step at the MIDPOINT gain
        arg = np.clip(arg.real, None, 100.0) + 1j * arg.imag  # overflow guard (unsaturated ceiling)
        return A * np.exp(arg)

    def propagate(self, A_in_x, Lz_m, nz, *, return_profile=False, T_profile_x=None,
                  dndt_per_K=0.0):
        """March the input transverse field A_in_x (nx,) over length Lz_m in nz steps. Returns a dict:
        x [m], A_out (nx,), I_out=|A_out|^2; with return_profile, also I_xz (nz+1, nx) the intensity
        at each z-plane and g_out the final-plane gain g(x).

        THERMAL LENSING (2-D thermo-optic): pass a transverse temperature profile T_profile_x (nx,) and
        dndt_per_K to impose the carrier/heat index lens delta_n(x) = dndt (T(x) - mean T), a per-step
        real phase exp(i k0 delta_n dz) (k0 the VACUUM wavenumber; the x-mean is an irrelevant global
        phase). A hot-centred T(x) with dndt > 0 -> higher on-axis index -> converging THERMAL LENS
        (self-focusing); dndt < 0 -> defocusing; a linear T(x) ramp steers the beam (a prism). None /
        dndt = 0 -> byte-identical (no lens). Phase-only, so it is energy-conserving."""
        A = np.asarray(A_in_x, dtype=np.complex128).copy()
        if A.shape != (self.nx,):
            raise ValueError("propagate: A_in_x must have shape (nx,)")
        dz = float(Lz_m) / int(nz)
        lens = None                                          # per-step thermal-lens phase rate [rad/m]
        if T_profile_x is not None and dndt_per_K != 0.0:
            Tx = np.asarray(T_profile_x, dtype=np.float64)
            if Tx.shape != (self.nx,):
                raise ValueError("propagate: T_profile_x must have shape (nx,)")
            lens = self.k0 * float(dndt_per_K) * (Tx - Tx.mean())
        prof = np.empty((int(nz) + 1, self.nx)) if return_profile else None
        if return_profile:
            prof[0] = np.abs(A) ** 2
        for n in range(int(nz)):
            A = self._diffract(A, 0.5 * dz)
            A = self._gain(A, dz)
            if lens is not None:                             # thermo-optic index lens (real phase)
                A = A * np.exp(1j * lens * dz)
            A = self._diffract(A, 0.5 * dz)
            if return_profile:
                prof[n + 1] = np.abs(A) ** 2
        out = {"x": self.x, "A_out": A, "I_out": np.abs(A) ** 2}
        if return_profile:
            out["I_xz"] = prof
            out["g_out"] = self.carrier_gain(A)
        return out

    def rms_width(self, I_x):
        """Intensity RMS transverse width sqrt(<x^2> - <x>^2) [m] of a profile I(x)."""
        I = np.asarray(I_x, dtype=np.float64)
        tot = I.sum()
        xbar = np.sum(self.x * I) / tot
        return float(np.sqrt(np.sum((self.x - xbar) ** 2 * I) / tot))


def qd_gain_table(model, drive, nu_Hz, P_grid_W):
    """Precompute the QD saturable MATERIAL gain g(P) [1/m] vs local guided power P at fixed injection
    `drive`, to drive a TransverseBPM(qd_gain_table=(P_grid, g_grid)) with the REAL group-resolved QD
    gain saturation (WL/ES reservoir + inhomogeneous broadening) in place of the phenomenological
    g0/(1 + S/Isat). g(P) is the static gain-vs-power compression curve QDGainModel.saturation_curve
    returns (the steady-state line-resolved modal gain at nu_s at each power). Returns (P_grid_W,
    g_grid); the BPM interpolates g at the local diffusion-smoothed power and flat-clamps outside the
    grid, so include a low first power (the small-signal-gain plateau) and a high last power (deep
    saturation) bracketing the operating intensities. Pure-numpy table build; SI; ASCII."""
    P = np.asarray(P_grid_W, dtype=np.float64)
    if P.ndim != 1 or P.size < 2 or np.any(np.diff(P) <= 0.0) or np.any(P <= 0.0):
        raise ValueError("qd_gain_table: P_grid_W must be strictly increasing positive 1-D >= 2 points")
    g, _S = model.saturation_curve(drive, P, nu_s_Hz=float(nu_Hz))
    return P, np.asarray(g, dtype=np.float64)
