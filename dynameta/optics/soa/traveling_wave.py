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

__all__ = ["TravelingWaveSOA", "TwoLevelSaturableGain", "UltrafastCompression",
           "agrawal_olsson_output"]


@dataclass
class UltrafastCompression:
    """Sub-picosecond nonlinear gain compression beyond the carrier-density dynamics: spectral
    hole burning (SHB) and carrier heating (CH). Each suppresses the local gain by a depth
    h_X that relaxes to eps_X * S_conf (the local confined photon density) with its own time
    constant, so the gain seen by the field is g_eff = g (1 - h_SHB - h_CH):

        dh_X/dt = (-h_X + eps_X S_conf) / tau_X .

    SHB is the fastest (intradot carrier-carrier scattering, ~50-100 fs); CH is the electron-
    phonon relaxation (~0.5-1 ps) and SHARES its time constant with the carrier-heating
    (two-temperature) relaxation that carriers.carrier_heating.TwoTempParams models -- folded
    here phenomenologically into the gain (it does NOT remove energy from the carrier
    reservoir; the carriers still see the real photons via the rate equations). eps_X [m^3]
    sets the compression depth; eps_X = 0 (default) disables the channel and the engine
    reduces EXACTLY to the carrier-density-only result. `floor` caps the suppression so g_eff
    stays physical under extreme drive. NB the SHB channel is only well-resolved when the
    time step dt << tau_shb (use fine slices / a short section for sub-100 fs SHB studies)."""
    eps_shb_m3: float = 0.0
    tau_shb_s: float = 1.0e-13
    eps_ch_m3: float = 0.0
    tau_ch_s: float = 7.0e-13
    floor: float = 0.05

    @property
    def active(self) -> bool:
        return self.eps_shb_m3 > 0.0 or self.eps_ch_m3 > 0.0


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

    # ---- optional ultrafast (SHB + CH) gain-compression layer ----
    def _uf_init(self, ultrafast):
        if ultrafast is None or not ultrafast.active:
            return None
        return {"uf": ultrafast, "h_shb": np.zeros(self.nz), "h_ch": np.zeros(self.nz),
                "d_shb": float(np.exp(-self.dt / ultrafast.tau_shb_s)),
                "d_ch": float(np.exp(-self.dt / ultrafast.tau_ch_s))}

    def _uf_suppress(self, uf, g):
        if uf is None:
            return g
        return g * np.clip(1.0 - uf["h_shb"] - uf["h_ch"], uf["uf"].floor, 1.0)

    def _uf_relax(self, uf, P_mid_W, nu):
        if uf is None:
            return
        S = self.model.photon_density(P_mid_W, nu)            # local confined photon density
        uf["h_shb"] = uf["h_shb"] * uf["d_shb"] + uf["uf"].eps_shb_m3 * S * (1.0 - uf["d_shb"])
        uf["h_ch"] = uf["h_ch"] * uf["d_ch"] + uf["uf"].eps_ch_m3 * S * (1.0 - uf["d_ch"])

    def amplify(self, P_in, drive, *, nu_s_Hz: float = None, state0=None,
                return_traces: bool = False, ultrafast=None):
        """Amplify the input power waveform P_in (1-D array sampled at this engine's dt) at
        injection `drive`. `ultrafast` (an UltrafastCompression) adds the sub-ps SHB + carrier-
        heating gain compression on top of the carrier-density dynamics (None -> off, the
        carrier-density-only result). Returns a dict: t [s], P_in, P_out [W], gain_dB, and (if
        return_traces) the line-centre material gain per slice over time `g_zt` (nt, nz) and
        the carrier state. The amplifier starts in the unsaturated steady state at `drive`."""
        nu = float(nu_s_Hz) if nu_s_Hz is not None else self.nu_s
        P_in = np.asarray(P_in, dtype=np.float64)
        if P_in.ndim != 1 or P_in.size < 2:
            raise ValueError("amplify: P_in must be a 1-D waveform with >= 2 samples")
        nt = P_in.size
        state = self.model.init_slices(self.nz, drive) if state0 is None else state0
        gam = self.model.gamma_confinement
        uf = self._uf_init(ultrafast)                         # ultrafast-compression state
        # the device field starts empty (Pnode = 0); the first nz steps fill it (a startup
        # transient one transit time long -- t = nz*dt -- before P_out is meaningful)
        Pnode = np.zeros(self.nz + 1)
        P_out = np.empty(nt)
        g_zt = np.empty((nt, self.nz)) if return_traces else None
        h_uf = np.zeros(nt) if (return_traces and uf is not None) else None
        for n in range(nt):
            g = self.model.gain_per_m_slices(state, nu)       # (nz,) material gain
            g = self._uf_suppress(uf, g)                      # SHB/CH gain compression (if on)
            amp = np.exp((gam * g - self.alpha_i) * self.dz)  # per-slice power amplification
            # method-of-characteristics shift: field at node k moves to node k+1, amplified
            new = np.empty_like(Pnode)
            new[0] = P_in[n]
            new[1:] = Pnode[:-1] * amp
            P_mid = 0.5 * (Pnode[:-1] + Pnode[1:])            # slice-average power this step
            self._uf_relax(uf, P_mid, nu)                     # drive the compression by S_conf
            state = self.model.step_slices(state, P_mid, self.dt, nu, drive)
            Pnode = new
            P_out[n] = Pnode[-1]
            if return_traces:
                g_zt[n] = g
                if uf is not None:
                    h_uf[n] = float(np.mean(uf["h_shb"] + uf["h_ch"]))
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
            if h_uf is not None:
                out["h_uf"] = h_uf                            # SHB+CH compression depth vs t
        return out

    # ---- optional spectral-dispersion (Maxwell-Bloch line-filter) layer ----
    def _line_filter_init(self, nu):
        """Set up the per-group complex-Lorentzian polarization filter that gives each spectral
        component of the envelope its OWN complex gain Gamma_field(nu_s + f) (the line shape AND
        its Kramers-Kronig dispersive partner), so gain dispersion across the signal band is
        resolved -- the up/down FWM asymmetry and pulse reshaping the flat-gain engine misses.

        One pole per inhomogeneous group: lam_j = -2 pi hw + 1j 2 pi (nu_s - nu_j) (hw = HWHM of
        the homogeneous line). Re(lam_j) = -2 pi hw < 0 always, so the polarization is integrated
        by the EXACT exponential step (unconditionally stable, no new CFL):
            p_j^{n+1} = E_j p_j^n + kappa_j A_ref coef_j,  E_j = exp(lam_j dt), coef_j = (E_j-1)/lam_j.
        The gain the field accumulates crossing a slice is the polarization integrated over the
        transit, NOT its start-of-step value -- using the start value is a zero-order-hold that
        leaks the (large) real gain into the (small) dispersive imaginary part as an O(dt)
        half-sample-delay error. So amplify_coherent uses the transit-AVERAGED polarization
            <p_j> = (1/dt) integral_0^dt p_j(t) dt = E_avg_j p_j + kappa_j A_ref coef_avg_j,
            E_avg_j = (E_j - 1)/(lam_j dt),  coef_avg_j = (E_j - 1)/(lam_j^2 dt) - 1/lam_j,
        whose sum sum_j <p_j> equals 2 Gamma_field(nu_s + f) A_ref to O(dt^2) for every offset f
        (Re and Im, verified ~2e-4 at nz=50 out to 200 GHz; no FFT, no tone comb). The field then
        gains the flat carrier part multiplicatively and the dispersive DEVIATION
        (sum_j <p_j> - g(nu_s) A_ref) ADDITIVELY -- the polarization SOURCES field, so the line
        radiates correctly into a field null (where dividing sum<p>/A would blow up). Requires a
        spectral gain model (QDGainModel)."""
        m = self.model
        if not (hasattr(m, "line_kappa_slices") and hasattr(m, "nu_j")
                and hasattr(m, "p") and hasattr(m.p, "fwhm_hom_Hz")):
            raise ValueError("amplify_coherent(line_filter=True) needs a spectral gain model with "
                             "nu_j, p.fwhm_hom_Hz and line_kappa_slices (e.g. QDGainModel); the {} "
                             "model has no homogeneous line to disperse".format(type(m).__name__))
        hw = 0.5 * float(m.p.fwhm_hom_Hz)
        nu_j = np.asarray(m.nu_j, dtype=np.float64)
        lam = -2.0 * np.pi * hw + 1j * 2.0 * np.pi * (float(nu) - nu_j)    # (ng,) complex pole
        E = np.exp(lam * self.dt)
        coef = (E - 1.0) / lam                                # exact-exp source weight (Re(lam)<0)
        E_avg = (E - 1.0) / (lam * self.dt)                   # transit-averaged decay
        coef_avg = (E - 1.0) / (lam * lam * self.dt) - 1.0 / lam   # transit-averaged source weight
        return {"hw": hw, "E": E, "coef": coef, "E_avg": E_avg, "coef_avg": coef_avg,
                "pol": np.zeros((self.nz, nu_j.size), dtype=np.complex128)}

    # ---- optional background group-velocity-dispersion (GVD) layer ----
    def _gvd_phase(self, beta2_s2_per_m: float, n_samples: int, length_m: float):
        """GVD spectral phase exp(+0.5j beta2 omega^2 length) over a propagation distance length_m on
        the full retarded-time waveform (n_samples sampled at dt). Background (waveguide + material)
        group-velocity
        dispersion adds
            dA/dz = -(i beta2 / 2) d2A/dT2     (retarded time T = t - z/v_g, exp(-i omega t)),
        so a baseband tone exp(-i 2 pi f t) (at optical nu_s + f) picks up the residual dispersive
        phase exp(+i (beta2/2)(2 pi f)^2 z) -- beta(nu_s+f) with the carrier (beta0) and the group
        delay (beta1, already carried EXACTLY by the dt = dz/v_g advection) removed; beta2 is the
        curvature d2 beta/d omega^2.

        GVD is applied as a symmetric (Strang) split. With gvd_segments = 1 it is a single
        DEVICE-scale split -- D(L/2) before the nonlinear streaming marcher and D(L/2) after -- so the
        carriers see the half-dispersed field and the output carries the full L of dispersion; EXACT
        in the linear / passive limit (per-tone phase, broadening, chirp -- where D and the marcher
        commute) but an UNCONTROLLED approximation of the distributed dispersion-gain coupling when
        beta2 AND a saturating gain are both active. With gvd_segments = S > 1 the device is split
        into S sub-sections [D(L/2S) . N(L/S) . D(L/S) . ... . N(L/S) . D(L/2S)], interleaving
        dispersion and gain S times -- a CONTROLLED 2nd-order (Strang) refinement whose splitting
        error falls as O(1/S^2) to the true distributed coupling (qd_soa_gvd_distributed.py measures
        the 2nd-order rate). D operates on the WHOLE waveform (the true retarded-time signal): FFT
        over n_samples, multiply each angular-frequency bin omega = 2 pi fftfreq(n_samples, dt) by
        exp(+0.5j beta2 omega^2 length), iFFT. The spectral phase is UNITARY (|.| = 1) so D is
        loss-free and unconditionally stable; beta2 is EVEN in omega so the FFT sign convention is
        immaterial (a future odd beta3 term would not be). This is the BROADBAND background index,
        kept distinct from the resonant gain-line dispersion of the line filter. NB the spatial node
        array could NOT be dispersed per-step: it is a snapshot along z, not a fixed retarded-time
        window, so dispersing the shifting window injects a boundary discontinuity and leaks energy --
        the device-/segment-scale split on the full waveform avoids this entirely."""
        omega = 2.0 * np.pi * np.fft.fftfreq(int(n_samples), d=self.dt)   # baseband ang. freq [rad/s]
        return np.exp(0.5j * float(beta2_s2_per_m) * omega * omega * float(length_m))

    def amplify_coherent(self, A_in, drive, *, nu_s_Hz: float = None, alpha_lef: float = None,
                         state0=None, ultrafast=None, line_filter: bool = False,
                         beta2_s2_per_m: float = None, gvd_segments: int = 1,
                         langevin: bool = False, seed=None):
        """Coherent multi-tone amplification: propagate the COMPLEX field envelope A(z, t) so
        cross-gain modulation and four-wave mixing emerge. The carrier-induced index couples
        through the linewidth enhancement factor alpha (model.alpha_lef unless overridden):
        the field amplification per slice is exp(0.5*(Gamma g (1 - i alpha) - alpha_i) dz)
        and the carriers are driven by the local power |A|^2. Two input tones beating at a
        detuning within the carrier cutoff pulsate the gain/index and scatter into FWM
        sidebands; far-detuned tones simply cross-saturate (XGM). Returns a dict with the
        complex A_out(t), |A_out|^2 power, t, dt. Reduces to amplify() (power) when alpha = 0
        and the input is a single real tone.

        line_filter (default False): when True, replaces the single carrier-frequency gain
        g(nu_s) with the per-group complex-Lorentzian SPECTRAL gain Gamma_field(nu_s + f) (the
        Maxwell-Bloch polarization ADE of _line_filter_init / model.line_kappa_slices), so each
        tone sees its own gain AND the resonant Kramers-Kronig dispersive phase -- gain
        dispersion across the band, enlarged up/down FWM asymmetry, group delay. False is the
        DEFAULT and that branch is byte-identical to the flat-gain engine (no model state touched
        before the branch), so all existing callers are unaffected. Requires a spectral gain
        model (QDGainModel); raises for the two-level oracle model. The broadband alpha index is
        kept distinct from the resonant line dispersion -- alpha multiplies ONLY the real gain.

        beta2_s2_per_m (default None -> model.beta2_s2_per_m or 0): background group-velocity
        dispersion d2 beta/d omega^2 [s^2/m]. When nonzero each tone at nu_s + f accumulates the
        broadband dispersive phase exp(+i (beta2/2)(2 pi f)^2 L), applied as a symmetric Strang split
        with the exact unitary spectral operator (see _gvd_phase) -- pulse broadening/chirp on top of
        the gain. 0 (the default) leaves the field branch byte-identical. This is the NON-resonant
        waveguide index, distinct from the resonant gain-line dispersion (line_filter).

        gvd_segments (default 1): number of dispersion-gain interleaving sub-sections. 1 = a single
        device-scale split D(L/2).marcher.D(L/2) -- EXACT in the linear / passive limit, but an
        uncontrolled approximation of the distributed coupling when beta2 and a saturating gain are
        both active. S > 1 splits the device into S streaming sub-marchers with dispersion between
        them, a CONTROLLED 2nd-order (Strang) refinement converging as O(1/S^2) to the true
        distributed coupling (must divide n_slices). Ignored when beta2 = 0.

        Convention exp(-i omega t): a complex baseband tone exp(-i 2 pi f t) sits at optical
        frequency nu_s + f; the FFT of A_out exposes the FWM products at 2 f1 - f2."""
        nu = float(nu_s_Hz) if nu_s_Hz is not None else self.nu_s
        # alpha: an explicit alpha_lef arg pins a constant scalar; otherwise the model supplies it,
        # per-slice and carrier-density-dependent via alpha_lef_slices(state) if available (the
        # constant-alpha models / slope=0 return a scalar -> the engine stays byte-identical)
        if alpha_lef is not None:
            alpha = float(alpha_lef)
            alpha_dyn = None
        else:
            alpha = float(getattr(self.model, "alpha_lef", 0.0))
            alpha_dyn = getattr(self.model, "alpha_lef_slices", None)
        beta2 = float(beta2_s2_per_m) if beta2_s2_per_m is not None else float(
            getattr(self.model, "beta2_s2_per_m", 0.0))
        A_in = np.asarray(A_in, dtype=np.complex128)
        if A_in.ndim != 1 or A_in.size < 2:
            raise ValueError("amplify_coherent: A_in must be a 1-D complex waveform >= 2 samples")
        if beta2 != 0.0 and int(gvd_segments) > 1:
            return self._amplify_coherent_segmented(
                A_in, drive, int(gvd_segments), beta2, nu_s_Hz=nu, alpha_lef=alpha_lef,
                ultrafast=ultrafast, line_filter=line_filter)
        nt = A_in.size
        state = self.model.init_slices(self.nz, drive) if state0 is None else state0
        gam = self.model.gamma_confinement
        uf = self._uf_init(ultrafast)
        lf = self._line_filter_init(nu) if line_filter else None
        # Langevin spontaneous-emission noise (opt-in). Each slice each step adds a complex Gaussian
        # field increment of variance Gamma g_sp(z) h nu v_g (real + imag each half) -- the
        # fluctuation-dissipation source whose downstream-amplified accumulation reproduces the
        # analytic ASE PSD n_sp h nu (G-1) EXACTLY (the geometric slice sum cancels to that), and
        # whose phase diffusion gives the Henry (1 + alpha^2) linewidth. Reproducible via seed; OFF
        # (default) makes no RNG calls -> the deterministic engine is byte-identical.
        lang = None
        if langevin:
            lang = {"rng": np.random.default_rng(seed),
                    "npref": 0.5 * gam * H_PLANCK * nu * self.model.v_g}   # = var/2 per unit g_sp
        # GVD as a symmetric (Strang) split at the device scale: D(L/2) . N(L) . D(L/2). The
        # dispersion operator D is the EXACT unitary spectral phase on the full waveform (the
        # retarded-time signal); the streaming nonlinear marcher N is left untouched. Half the
        # dispersion pre-chirps the field the carriers see, half post-chirps the output -- EXACT in
        # the linear (CW / passive) limit; an uncontrolled (single step, no z-refinement) approx of
        # the distributed dispersion-gain coupling when both are active. (A per-step dispersion of
        # the spatial node array is invalid: that array is a snapshot along z, not a fixed
        # retarded-time window, so dispersing the shifting window leaks energy.)
        A_in_orig = A_in
        if beta2 != 0.0:
            half = self._gvd_phase(beta2, nt, 0.5 * self.L)
            A_in = np.fft.ifft(np.fft.fft(A_in) * half)
        Anode = np.zeros(self.nz + 1, dtype=np.complex128)
        A_out = np.empty(nt, dtype=np.complex128)
        for n in range(nt):
            new = np.empty_like(Anode)
            new[0] = A_in[n]
            a_eff = alpha if alpha_dyn is None else alpha_dyn(state)   # per-slice density-dependent
            if lf is None:
                g = self._uf_suppress(uf, self.model.gain_per_m_slices(state, nu))
                amp = np.exp(0.5 * (gam * g * (1.0 - 1j * a_eff) - self.alpha_i) * self.dz)
                new[1:] = Anode[:-1] * amp
            else:
                A_ref = Anode[:-1]
                kappa = self.model.line_kappa_slices(state, nu, lf["hw"])   # (nz, ng) live carriers
                src = kappa * A_ref[:, None]                        # (nz, ng) polarization source
                avg_p = lf["E_avg"] * lf["pol"] + src * lf["coef_avg"]      # transit-averaged pol
                sum_p = np.sum(avg_p, axis=1)                       # (nz,); CW: 2 Gamma_field(nu_s+f) A_ref
                g_un = self.model.gain_per_m_slices(state, nu)      # carrier real gain (no division)
                g_flat = self._uf_suppress(uf, g_un)
                amp = np.exp(0.5 * (gam * g_flat * (1.0 - 1j * a_eff) - self.alpha_i) * self.dz)
                # flat carrier gain (multiplicative, == OFF amp) + ADDITIVE dispersive correction:
                # the polarization sum_p minus its flat-gain equivalent g_un*A_ref is the resonant
                # line deviation (zero at the carrier). Additive so the line radiates field into a
                # null (sum_p != 0 there) -- no divide-by-field, stable for modulated/nulling
                # waveforms; first order in the small per-slice deviation (~1e-3).
                new[1:] = amp * (A_ref + 0.5 * gam * (sum_p - g_un * A_ref) * self.dz)
                lf["pol"] = lf["E"] * lf["pol"] + src * lf["coef"]
            if lang is not None:                               # Langevin spontaneous-emission source
                gsp = self.model.emission_gain_per_m_slices(state, nu)   # (nz,) >= 0
                sig = np.sqrt(lang["npref"] * gsp)             # std of real (= imag) part per slice
                new[1:] = new[1:] + sig * (lang["rng"].standard_normal(self.nz)
                                           + 1j * lang["rng"].standard_normal(self.nz))
            P_mid = 0.5 * (np.abs(Anode[:-1]) ** 2 + np.abs(Anode[1:]) ** 2)
            self._uf_relax(uf, P_mid, nu)
            state = self.model.step_slices(state, P_mid, self.dt, nu, drive)
            Anode = new
            A_out[n] = Anode[-1]
        if beta2 != 0.0:
            A_out = np.fft.ifft(np.fft.fft(A_out) * half)     # second half-dispersion (post-march)
        t = np.arange(nt) * self.dt
        return {"t": t, "A_in": A_in_orig, "A_out": A_out, "P_in": np.abs(A_in_orig) ** 2,
                "P_out": np.abs(A_out) ** 2, "dt": self.dt, "state": state}

    def _amplify_coherent_segmented(self, A_in, drive, segments, beta2, *, nu_s_Hz, alpha_lef,
                                    ultrafast, line_filter):
        """Distributed GVD: the symmetric Strang split refined to S sub-sections that interleave
        dispersion and gain S times instead of lumping all the dispersion at the two device ends:

            D(L/2S) . N(L/S) . D(L/S) . N(L/S) . ... . N(L/S) . D(L/2S)   (S blocks of N).

        Each N(L/S) is a full streaming sub-marcher over the length L/S (its own slices + fresh
        carriers, so segment k's gain saturates on the field that has already crossed segments
        0..k-1 -- this is the real distributed coupling); D is the exact unitary spectral phase on
        the full waveform (no internal beta2 in the sub-marchers). The dt = dz/v_g is preserved
        (dz = (L/S)/(nz/S) = L/nz), so the dispersion grid is unchanged. The splitting error falls as
        O(1/S^2) (2nd-order Strang) to the true distributed limit, EXACT for any S in the linear /
        passive limit (D commutes with the gain-free delay). Total dispersion = L/2S + (S-1)L/S +
        L/2S = L. Returns the same dict as amplify_coherent (state is the LAST segment's carriers).

        CAVEAT (S-dependent startup): each sub-marcher re-pays its own nz/S-sample device-fill
        transient, so unlike the monolithic engine's single clean nz-sample fill the LEADING ~nz
        samples of A_out are an S-DEPENDENT superposition of S separate startup ramps plus
        segment-boundary zeros. Only the steady tail A_out[nz:] (after the full nz-sample transit) is
        physical -- window past the first nz samples, exactly as for the monolithic amplify()/
        amplify_coherent() (whose gain_dB is likewise meaningful only at CW steady state)."""
        if int(segments) < 1 or self.nz % int(segments) != 0:
            raise ValueError("amplify_coherent(gvd_segments=S): S must be >= 1 and divide n_slices "
                             "({} % {} != 0)".format(self.nz, int(segments)))
        S = int(segments)
        nt = A_in.size
        h = self.L / S
        sub = TravelingWaveSOA(self.model, h, self.nz // S, alpha_i_per_m=self.alpha_i,
                               nu_s_Hz=self.nu_s)
        D_full = self._gvd_phase(beta2, nt, h)                # interior segment-to-segment D(L/S)
        D_half = self._gvd_phase(beta2, nt, 0.5 * h)          # the two end half-steps D(L/2S)
        field = np.fft.ifft(np.fft.fft(A_in) * D_half)
        state = None
        for k in range(S):
            r = sub.amplify_coherent(field, drive, nu_s_Hz=nu_s_Hz, alpha_lef=alpha_lef,
                                     ultrafast=ultrafast, line_filter=line_filter,
                                     beta2_s2_per_m=0.0)
            field, state = r["A_out"], r["state"]
            field = np.fft.ifft(np.fft.fft(field) * (D_half if k == S - 1 else D_full))
        t = np.arange(nt) * self.dt
        return {"t": t, "A_in": A_in, "A_out": field, "P_in": np.abs(A_in) ** 2,
                "P_out": np.abs(field) ** 2, "dt": self.dt, "state": state}

    def amplify_coherent_dualpol(self, A_te_in, A_tm_in, drive, *, nu_s_Hz=None, alpha_lef=None,
                                 pdg_ratio: float = 1.0, state0=None, ultrafast=None):
        """Polarization-dependent gain (PDG): co-propagate the TE and TM complex envelopes through
        ONE shared carrier reservoir. The TM modal gain is pdg_ratio x the TE modal gain --
        pdg_ratio folds the TE/TM modal-confinement ratio and the QD material gain anisotropy into a
        single number (flat self-assembled QDs favour TE, so pdg_ratio < 1). Both polarizations
        deplete the SAME dots: the confined density that saturates the carriers is the modal-weighted
        total |A_TE|^2 + pdg_ratio |A_TM|^2, so a strong signal in one polarization CROSS-SATURATES
        the gain seen by the other (the physics a single-pol run cannot show). pdg_ratio = 1 makes
        the two pols gain-degenerate and each reduces EXACTLY to amplify_coherent (flat-gain branch).

        Flat-gain path (no line filter / GVD -- those are single-pol features); the alpha(rho)
        density dependence applies identically to both pols. Returns a dict: A_te_out / A_tm_out
        (complex), P_te_out / P_tm_out, t, dt, state. PDG(dB) = 10 log10(G_TE / G_TM) on the steady
        tails (small-signal -> (1 - pdg_ratio) Gamma g L * 10/ln10)."""
        nu = float(nu_s_Hz) if nu_s_Hz is not None else self.nu_s
        if alpha_lef is not None:
            alpha, alpha_dyn = float(alpha_lef), None
        else:
            alpha = float(getattr(self.model, "alpha_lef", 0.0))
            alpha_dyn = getattr(self.model, "alpha_lef_slices", None)
        r = float(pdg_ratio)
        te_in = np.asarray(A_te_in, dtype=np.complex128)
        tm_in = np.asarray(A_tm_in, dtype=np.complex128)
        if te_in.shape != tm_in.shape or te_in.ndim != 1 or te_in.size < 2:
            raise ValueError("amplify_coherent_dualpol: A_te_in, A_tm_in must be equal-length 1-D "
                             ">= 2 samples")
        nt = te_in.size
        state = self.model.init_slices(self.nz, drive) if state0 is None else state0
        gam = self.model.gamma_confinement
        uf = self._uf_init(ultrafast)
        te = np.zeros(self.nz + 1, dtype=np.complex128)
        tm = np.zeros(self.nz + 1, dtype=np.complex128)
        te_out = np.empty(nt, dtype=np.complex128)
        tm_out = np.empty(nt, dtype=np.complex128)
        for n in range(nt):
            a_eff = alpha if alpha_dyn is None else alpha_dyn(state)
            g = self._uf_suppress(uf, self.model.gain_per_m_slices(state, nu))
            phase = 1.0 - 1j * a_eff
            amp_te = np.exp(0.5 * (gam * g * phase - self.alpha_i) * self.dz)
            amp_tm = np.exp(0.5 * (r * gam * g * phase - self.alpha_i) * self.dz)
            nte = np.empty_like(te)
            nte[0] = te_in[n]
            nte[1:] = te[:-1] * amp_te
            ntm = np.empty_like(tm)
            ntm[0] = tm_in[n]
            ntm[1:] = tm[:-1] * amp_tm
            # modal-weighted total power saturates the shared reservoir (TM contributes r x its power)
            P_mid = 0.5 * ((np.abs(te[:-1]) ** 2 + np.abs(te[1:]) ** 2)
                           + r * (np.abs(tm[:-1]) ** 2 + np.abs(tm[1:]) ** 2))
            self._uf_relax(uf, P_mid, nu)
            state = self.model.step_slices(state, P_mid, self.dt, nu, drive)
            te, tm = nte, ntm
            te_out[n] = te[-1]
            tm_out[n] = tm[-1]
        t = np.arange(nt) * self.dt
        return {"t": t, "A_te_out": te_out, "A_tm_out": tm_out, "P_te_out": np.abs(te_out) ** 2,
                "P_tm_out": np.abs(tm_out) ** 2, "dt": self.dt, "state": state}

    def amplify_fabry_perot(self, A_in, drive, *, R1: float, R2: float, nu_s_Hz=None,
                            alpha_lef=None, roundtrip_phase: float = 0.0, state0=None,
                            ultrafast=None, langevin: bool = False, seed=None):
        """Fabry-Perot SOA: counter-propagating FORWARD (F) and BACKWARD (B) complex envelopes
        coupled by the facet POWER reflectivities R1, R2 (field reflectivities sqrt(R)), both
        saturating the SHARED carrier reservoir (the dots see |F|^2 + |B|^2). The cavity replaces the
        single-pass Saitoh-Mukai ripple METRIC with the real round-trip feedback:

            F advances +z, gains exp(0.5(Gamma g(1-i alpha) - alpha_i) dz) per slice;
            B advances -z with the same per-slice gain;
            facet 1 (node 0):  F_in = t1 A_in + r1 e^{i phi/2} B(0)   (t1=sqrt(1-R1), r1=sqrt(R1));
            facet 2 (node nz): B_in = r2 e^{i phi/2} F(nz);   transmitted output = t2 F(nz).

        roundtrip_phase phi is the cavity DETUNING 2 beta(nu_s) L mod 2pi carried by the envelope
        (removed from the baseband carrier) -- sweeping phi traces the FP gain ripple; the resonant
        (phi=0) gain follows the Airy denominator (1 - sqrt(R1 R2) G)^-2 and SATURATES under the
        external-seed feedback as sqrt(R1 R2) G -> 1 (the built-up intracavity field depletes the
        carriers). R1=R2=0 reduces EXACTLY to the single-pass amplify_coherent forward field (B stays
        0). SCOPE: no spontaneous-emission / ASE seed is modelled (F=B=0 at t=0; only the coherent
        A_in drives the cavity), so true lasing FROM NOISE is out of scope -- above sqrt(R1 R2) G = 1
        the device is non-physical without a seed and the field is bounded only by the injected power;
        the gain saturation here tracks the injected seed, it is not a self-consistent threshold pin.
        Flat-gain path (no line filter / GVD); alpha(rho) applies. Returns A_out (transmitted), the
        intracavity |F|^2/|B|^2 at the last step, t, dt, state."""
        nu = float(nu_s_Hz) if nu_s_Hz is not None else self.nu_s
        if alpha_lef is not None:
            alpha, alpha_dyn = float(alpha_lef), None
        else:
            alpha = float(getattr(self.model, "alpha_lef", 0.0))
            alpha_dyn = getattr(self.model, "alpha_lef_slices", None)
        if not (0.0 <= R1 < 1.0 and 0.0 <= R2 < 1.0):
            raise ValueError("amplify_fabry_perot: need 0 <= R1, R2 < 1")
        r1, r2 = np.sqrt(R1), np.sqrt(R2)
        t1, t2 = np.sqrt(1.0 - R1), np.sqrt(1.0 - R2)
        ph = np.exp(0.5j * float(roundtrip_phase))
        A_in = np.asarray(A_in, dtype=np.complex128)
        if A_in.ndim != 1 or A_in.size < 2:
            raise ValueError("amplify_fabry_perot: A_in must be a 1-D complex waveform >= 2 samples")
        nt = A_in.size
        state = self.model.init_slices(self.nz, drive) if state0 is None else state0
        gam = self.model.gamma_confinement
        uf = self._uf_init(ultrafast)
        # Langevin spontaneous emission into BOTH counter-propagating modes (seeds lasing from noise;
        # near threshold the gain-clamped amplitude-phase coupling gives the Henry (1+alpha^2)
        # linewidth). OFF -> no RNG, byte-identical.
        lang = {"rng": np.random.default_rng(seed),
                "npref": 0.5 * gam * H_PLANCK * nu * self.model.v_g} if langevin else None
        F = np.zeros(self.nz + 1, dtype=np.complex128)
        B = np.zeros(self.nz + 1, dtype=np.complex128)
        A_out = np.empty(nt, dtype=np.complex128)
        for n in range(nt):
            a_eff = alpha if alpha_dyn is None else alpha_dyn(state)
            g = self._uf_suppress(uf, self.model.gain_per_m_slices(state, nu))
            amp = np.exp(0.5 * (gam * g * (1.0 - 1j * a_eff) - self.alpha_i) * self.dz)
            nF = np.empty_like(F)
            nB = np.empty_like(B)
            nF[1:] = F[:-1] * amp                              # forward advection + gain
            nF[0] = t1 * A_in[n] + r1 * ph * B[0]              # facet 1: input + reflected backward
            nB[:-1] = B[1:] * amp                              # backward advection + gain
            nB[self.nz] = r2 * ph * F[self.nz]                 # facet 2: reflected forward
            if lang is not None:                               # spontaneous source into both modes
                gsp = self.model.emission_gain_per_m_slices(state, nu)
                sg = np.sqrt(lang["npref"] * gsp)
                rng = lang["rng"]
                nF[1:] = nF[1:] + sg * (rng.standard_normal(self.nz) + 1j * rng.standard_normal(self.nz))
                nB[:-1] = nB[:-1] + sg * (rng.standard_normal(self.nz) + 1j * rng.standard_normal(self.nz))
            P_mid = (0.5 * (np.abs(F[:-1]) ** 2 + np.abs(F[1:]) ** 2)
                     + 0.5 * (np.abs(B[:-1]) ** 2 + np.abs(B[1:]) ** 2))   # both pump the reservoir
            self._uf_relax(uf, P_mid, nu)
            state = self.model.step_slices(state, P_mid, self.dt, nu, drive)
            F, B = nF, nB
            A_out[n] = t2 * F[self.nz]                          # transmitted through facet 2
        t = np.arange(nt) * self.dt
        return {"t": t, "A_in": A_in, "A_out": A_out, "P_out": np.abs(A_out) ** 2,
                "P_fwd_intracav": np.abs(F) ** 2, "P_bwd_intracav": np.abs(B) ** 2,
                "dt": self.dt, "state": state}

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
