"""Traveling-wave QD-SOA dynamics (roadmap SOA Phase 2): the z-resolved, time-domain engine
where the NONLINEAR gain dynamics actually live -- dynamic gain saturation, gain recovery,
pattern effects, cross-gain modulation, and (with the group-resolved QD model) spectral-hole
dynamics. This is the "core of the simulation" the spec flags.

Method of characteristics on a grid aligned to the group velocity: with the time step tied to
the slice transit time, dt = dz/v_g, the forward optical power advances exactly one slice per
step (no numerical dispersion in the advection), while each slice's carrier state integrates
its rate equations over dt driven by the local guided power (operator splitting):

    P(z=0, t) = P_in(t);   P_{k+1}^{n+1} = P_k^n exp((Gamma g_k - alpha_i) dz)
    carrier_k^{n+1} = step(carrier_k^n, P_mid_k^n, dt)        [P_mid = slice-average power]

The slab gain model is pluggable (duck-typed) so the SAME engine drives both the deep
group-resolved QD model (optics.soa.qd_gain.QDGainModel) and a simple two-level saturable
gain (TwoLevelSaturableGain below), the latter being the analytic Agrawal-Olsson oracle for
verifying the propagation + saturation numerics. A slab model must provide:

    .v_g                                  group velocity [m/s]
    .gamma_confinement                    modal confinement (1.0 if gain is already modal)
    .init_slices(n_slices, drive)         -> opaque per-slice carrier state
    .gain_per_m_slices(state, nu)         -> material gain g [1/m] per slice (Nz,)
    .step_slices(state, P_local_W, dt, nu, drive) -> advanced state

Pure numpy/scipy; SI units; exp(-i omega t) (gain -> Im(chi) < 0). Incoherent intensity
signal: power propagation (no phase), correct for the intensity-encoded OVMM gain leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import HBAR

H_PLANCK = 2.0 * np.pi * HBAR

__all__ = ["TravelingWaveSOA", "TwoLevelSaturableGain", "agrawal_olsson_output"]


class TravelingWaveSOA:
    """z-resolved time-domain SOA. `model` is any slab gain model implementing the protocol in
    the module docstring (QDGainModel for the deep physics; TwoLevelSaturableGain for the
    analytic oracle). The internal time step is dt = dz/v_g (the slice transit time)."""

    def __init__(self, model, length_m: float, n_slices: int, *,
                 alpha_i_per_m: float = 0.0, nu_s_Hz: float = None):
        if not (length_m > 0.0 and n_slices >= 1):
            raise ValueError("TravelingWaveSOA: length_m > 0 and n_slices >= 1 required")
        if alpha_i_per_m < 0.0:
            raise ValueError("TravelingWaveSOA: alpha_i_per_m must be >= 0")
        self.model = model
        self.L = float(length_m)
        self.nz = int(n_slices)
        self.dz = self.L / self.nz
        self.dt = self.dz / model.v_g                         # transit time of one slice
        self.alpha_i = float(alpha_i_per_m)
        self.nu_s = float(nu_s_Hz) if nu_s_Hz is not None else float(model.nu0_Hz) \
            if hasattr(model, "nu0_Hz") else float(getattr(model, "nu_s_Hz", 0.0))
        if not self.nu_s > 0.0:
            raise ValueError("TravelingWaveSOA: a signal frequency nu_s_Hz is required")

    def amplify(self, P_in, drive, *, nu_s_Hz: float = None, state0=None,
                return_traces: bool = False):
        """Amplify the input power waveform P_in (1-D array sampled at this engine's dt) at
        injection `drive`. Returns a dict: t [s], P_in, P_out [W], gain_dB, and (if
        return_traces) the line-centre material gain per slice over time `g_zt` (nt, nz) and
        the carrier state. The amplifier starts in the unsaturated steady state at `drive`."""
        nu = float(nu_s_Hz) if nu_s_Hz is not None else self.nu_s
        P_in = np.asarray(P_in, dtype=np.float64)
        if P_in.ndim != 1 or P_in.size < 2:
            raise ValueError("amplify: P_in must be a 1-D waveform with >= 2 samples")
        nt = P_in.size
        state = self.model.init_slices(self.nz, drive) if state0 is None else state0
        gam = self.model.gamma_confinement
        # the device field starts empty (Pnode = 0); the first nz steps fill it (a startup
        # transient one transit time long -- t = nz*dt -- before P_out is meaningful)
        Pnode = np.zeros(self.nz + 1)
        P_out = np.empty(nt)
        g_zt = np.empty((nt, self.nz)) if return_traces else None
        for n in range(nt):
            g = self.model.gain_per_m_slices(state, nu)       # (nz,) material gain
            amp = np.exp((gam * g - self.alpha_i) * self.dz)  # per-slice power amplification
            # method-of-characteristics shift: field at node k moves to node k+1, amplified
            new = np.empty_like(Pnode)
            new[0] = P_in[n]
            new[1:] = Pnode[:-1] * amp
            P_mid = 0.5 * (Pnode[:-1] + Pnode[1:])            # slice-average power this step
            state = self.model.step_slices(state, P_mid, self.dt, nu, drive)
            Pnode = new
            P_out[n] = Pnode[-1]
            if return_traces:
                g_zt[n] = g
        t = np.arange(nt) * self.dt
        # NOTE: gain_dB is the instantaneous P_out/P_in ratio; during transients P_out[n] is
        # the response to P_in[n - nz] (one transit earlier), so the ratio mixes wavefronts
        # and is physically meaningful only at CW steady state. Use the g_zt material-gain
        # trace (return_traces=True) for the medium's instantaneous gain.
        with np.errstate(divide="ignore", invalid="ignore"):
            gain_dB = 10.0 * np.log10(np.where(P_in > 0.0, P_out / P_in, np.nan))
        out = {"t": t, "P_in": P_in, "P_out": P_out, "gain_dB": gain_dB, "dt": self.dt,
               "state": state}
        if return_traces:
            out["g_zt"] = g_zt
        return out

    def amplify_coherent(self, A_in, drive, *, nu_s_Hz: float = None, alpha_lef: float = None,
                         state0=None):
        """Coherent multi-tone amplification: propagate the COMPLEX field envelope A(z, t) so
        cross-gain modulation and four-wave mixing emerge. The carrier-induced index couples
        through the linewidth enhancement factor alpha (model.alpha_lef unless overridden):
        the field amplification per slice is exp(0.5*(Gamma g (1 - i alpha) - alpha_i) dz)
        and the carriers are driven by the local power |A|^2. Two input tones beating at a
        detuning within the carrier cutoff pulsate the gain/index and scatter into FWM
        sidebands; far-detuned tones simply cross-saturate (XGM). Returns a dict with the
        complex A_out(t), |A_out|^2 power, t, dt. Reduces to amplify() (power) when alpha = 0
        and the input is a single real tone.

        Convention exp(-i omega t): a complex baseband tone exp(-i 2 pi f t) sits at envelope
        frequency f; the FFT of A_out exposes the FWM products at 2 f1 - f2."""
        nu = float(nu_s_Hz) if nu_s_Hz is not None else self.nu_s
        alpha = float(alpha_lef) if alpha_lef is not None else float(
            getattr(self.model, "alpha_lef", 0.0))
        A_in = np.asarray(A_in, dtype=np.complex128)
        if A_in.ndim != 1 or A_in.size < 2:
            raise ValueError("amplify_coherent: A_in must be a 1-D complex waveform >= 2 samples")
        nt = A_in.size
        state = self.model.init_slices(self.nz, drive) if state0 is None else state0
        gam = self.model.gamma_confinement
        Anode = np.zeros(self.nz + 1, dtype=np.complex128)
        A_out = np.empty(nt, dtype=np.complex128)
        for n in range(nt):
            g = self.model.gain_per_m_slices(state, nu)
            amp = np.exp(0.5 * (gam * g * (1.0 - 1j * alpha) - self.alpha_i) * self.dz)
            new = np.empty_like(Anode)
            new[0] = A_in[n]
            new[1:] = Anode[:-1] * amp
            P_mid = 0.5 * (np.abs(Anode[:-1]) ** 2 + np.abs(Anode[1:]) ** 2)
            state = self.model.step_slices(state, P_mid, self.dt, nu, drive)
            Anode = new
            A_out[n] = Anode[-1]
        t = np.arange(nt) * self.dt
        return {"t": t, "A_in": A_in, "A_out": A_out, "P_in": np.abs(A_in) ** 2,
                "P_out": np.abs(A_out) ** 2, "dt": self.dt, "state": state}

    def steady_gain_dB(self, P_cw_W: float, drive, *, nu_s_Hz: float = None,
                       settle_lifetimes: float = 30.0, tol_dB: float = 1e-3) -> float:
        """Single-pass CW gain [dB] at input power P_cw, at STEADY STATE. The settle time is
        set by the CARRIER relaxation (the spontaneous lifetime), NOT the transit time -- a
        saturated SOA relaxes on the carrier lifetime, which is many transit times. Settles
        for settle_lifetimes * the model's relaxation time (>= 4 transits), then asserts the
        gain has stopped changing to tol_dB over the last tenth of the window (RuntimeWarning
        if still drifting)."""
        tau = float(getattr(self.model, "relaxation_time_s", 4.0 * self.dt))
        nt = max(int(settle_lifetimes * tau / self.dt), 4 * self.nz, 200)
        P_in = np.full(nt, float(P_cw_W))
        g = self.amplify(P_in, drive, nu_s_Hz=nu_s_Hz)["gain_dB"]
        tail = g[max(nt - nt // 10, 1):]
        if np.nanmax(tail) - np.nanmin(tail) > tol_dB:
            import warnings
            warnings.warn("steady_gain_dB: gain still drifting by {:.2e} dB over the last "
                          "tenth of the settle window -- raise settle_lifetimes".format(
                              float(np.nanmax(tail) - np.nanmin(tail))), RuntimeWarning,
                          stacklevel=2)
        return float(g[-1])


@dataclass
class TwoLevelSaturableGain:
    """Textbook two-level saturable gain -- the slab model whose distributed dynamics reduce
    to the Agrawal-Olsson lumped result, used to verify the traveling-wave engine. The local
    MODAL gain g(z, t) obeys the standard SOA gain-saturation equation

        dg/dt = (g0 - g)/tau_c - g P/E_sat

    (g0 unsaturated modal gain [1/m]; tau_c carrier lifetime; E_sat saturation energy [J];
    P local guided power [W]). gamma_confinement = 1 (g is already modal). State is the
    per-slice gain array."""
    g0_per_m: float
    tau_c_s: float
    E_sat_J: float
    v_g_m_s: float = 8.5e7
    nu0_Hz: float = 1.934e14
    alpha_lef: float = 0.0            # linewidth enhancement factor (0 -> pure amplitude gain)

    def __post_init__(self):
        if not (self.tau_c_s > 0.0 and self.E_sat_J > 0.0 and self.v_g_m_s > 0.0):
            raise ValueError("TwoLevelSaturableGain: tau_c, E_sat, v_g must be > 0")

    @property
    def v_g(self):
        return self.v_g_m_s

    @property
    def relaxation_time_s(self):
        return self.tau_c_s

    @property
    def gamma_confinement(self):
        return 1.0

    def init_slices(self, n_slices, drive):
        return np.full(int(n_slices), self.g0_per_m)

    def gain_per_m_slices(self, state, nu_Hz):
        return state

    def step_slices(self, state, P_local_W, dt_s, nu_s_Hz, drive):
        g = state

        def f(gg):
            return (self.g0_per_m - gg) / self.tau_c_s - gg * np.asarray(P_local_W) / self.E_sat_J

        k1 = f(g)
        k2 = f(g + 0.5 * dt_s * k1)
        k3 = f(g + 0.5 * dt_s * k2)
        k4 = f(g + dt_s * k3)
        return g + dt_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def agrawal_olsson_output(t_s, P_in_W, g0_per_m, L_m, tau_c_s, E_sat_J):
    """Analytic Agrawal-Olsson reference for nonlinear pulse amplification in a saturable-gain
    SOA (Agrawal & Olsson, IEEE JQE 25:2297, 1989). Integrates the lumped log-gain

        dh/dt = (h0 - h)/tau_c - (P_in(t)/E_sat)(exp(h) - 1),   h0 = g0 L,

    and returns P_out(t) = P_in(t) exp(h(t)). This is the LUMPED limit the distributed
    traveling-wave engine must reproduce as n_slices grows. Pure quadrature (explicit RK4 on
    the supplied uniform t grid)."""
    t = np.asarray(t_s, dtype=np.float64)
    P = np.asarray(P_in_W, dtype=np.float64)
    if t.ndim != 1 or t.size != P.size or t.size < 2:
        raise ValueError("agrawal_olsson_output: t_s and P_in_W must be equal-length 1-D")
    dt = float(t[1] - t[0])
    h0 = g0_per_m * L_m
    h = np.empty(t.size)
    hh = h0
    for i in range(t.size):
        h[i] = hh
        Pi = 0.5 * (P[i] + P[min(i + 1, P.size - 1)])         # midpoint power for the step

        def f(x):
            return (h0 - x) / tau_c_s - (Pi / E_sat_J) * (np.exp(x) - 1.0)

        k1 = f(hh)
        k2 = f(hh + 0.5 * dt * k1)
        k3 = f(hh + 0.5 * dt * k2)
        k4 = f(hh + dt * k3)
        hh = hh + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return P * np.exp(h)
