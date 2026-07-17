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

from dynameta.constants import H_PLANCK


__all__ = ["TravelingWaveSOA", "TwoLevelSaturableGain", "UltrafastCompression", "NonlinearLoss",
           "agrawal_olsson_output"]


def _exact_emit_factor(x):
    """Exact-emit variance factor (expm1(x)/x, ->1 as x->0) for Langevin noise born inside a
    slice of net power-gain exponent x = a_net*dz (audit C4-4): injecting the bare O(dz)
    source AFTER the slice amplification accumulates to n_sp h nu (G-1) * [a dz/(e^{a dz}-1)]
    -- a systematic ~lnG/(2 nz) ASE deficit (4.3% at 30 dB gain, nz=80). Scaling the
    per-slice variance by this factor makes the geometric slice sum telescope EXACTLY to
    n_sp h nu (G-1), mirroring ase_output_psd's exact per-slice emit."""
    x = np.asarray(x, dtype=np.float64)
    safe = np.where(np.abs(x) < 1e-9, 1.0, x)
    return np.where(np.abs(x) < 1e-9, 1.0 + 0.5 * x, np.expm1(safe) / safe)


@dataclass
class NonlinearLoss:
    """Intensity- and carrier-DEPENDENT internal loss beyond the fixed alpha_i: two-photon
    absorption (TPA) and free-carrier absorption (FCA). The per-slice loss added to the marcher's
    modal coefficient is

        alpha_nl(z) = sigma_fca_m2 * N_w(z)            [FCA, carrier-dependent]
                    + (beta_tpa_m_per_W / A_eff) * P(z) [TPA, intensity-dependent]

    so the total per-slice power coefficient becomes Gamma g - alpha_i - alpha_nl.

      - FCA: sigma_fca_m2 is the EFFECTIVE MODAL free-carrier cross-section [m^2] -- it multiplies the
        wetting-layer reservoir density N_w = state[0]. This makes the internal loss grow with pumping
        and relax as the signal depletes N_w -- the fixed alpha_i becomes dynamic. Reduces to no extra
        loss when sigma_fca_m2 = 0. TWO conventions are folded into sigma_fca, so a literature MATERIAL
        cross-section must be pre-scaled before use:
          * Confinement: it is the MODAL cross-section (absorbs the confinement factor), i.e.
            sigma_fca_modal = Gamma * sigma_fca_material (Gamma = model.gamma_confinement, ~0.06 here).
            The marcher Gamma-weights the gain explicitly (Gamma g) but NOT the FCA -- the Gamma is
            hidden inside sigma_fca, so a raw material sigma would under-loss by ~Gamma.
          * Single-reservoir proxy: N_w is the WETTING-LAYER density ONLY (in the e/h-split layout the
            ELECTRON WL N_w_e alone; the hole WL N_w_h and the confined ES/GS carriers are NOT summed
            in). sigma_fca is therefore an effective LUMPED cross-section calibrated to that one
            density, not a first-principles per-species sigma.
      - TPA: beta_tpa_m_per_W is the TPA coefficient beta [m/W]; A_eff_m2 the effective nonlinear
        modal area [m^2] (<= 0 -> the model's A_mode_m2). The power loss is dP/dz|_TPA = -(beta/A_eff)
        P^2, i.e. an intensity-dependent coefficient beta P / A_eff. Reduces to zero when beta = 0.

    SCOPE: both are pure ABSORPTION (the real part). The companion REACTIVE terms -- the free-carrier
    plasma index (Drude dn ~ -N) and the Kerr / TPA-carrier self-phase modulation -- are a separate
    refinement (a phase term in the coherent marcher), as is the second-order TPA-GENERATED-carrier
    FCA (beta-generated carriers feeding sigma_fca); not included here. Default all-zero -> the
    marcher is byte-identical."""
    beta_tpa_m_per_W: float = 0.0
    sigma_fca_m2: float = 0.0
    A_eff_m2: float = 0.0

    @property
    def active(self) -> bool:
        return self.beta_tpa_m_per_W > 0.0 or self.sigma_fca_m2 > 0.0

    def __post_init__(self):
        if self.beta_tpa_m_per_W < 0.0 or self.sigma_fca_m2 < 0.0:
            raise ValueError("NonlinearLoss: beta_tpa_m_per_W and sigma_fca_m2 must be >= 0")


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

    def _nl_setup(self, nl_loss):
        """Resolve the NonlinearLoss config (None/inactive -> None, so the marcher stays byte-
        identical). Returns (sigma_fca, beta_over_Aeff) or None."""
        if nl_loss is None or not nl_loss.active:
            return None
        A_eff = nl_loss.A_eff_m2 if nl_loss.A_eff_m2 > 0.0 else float(self.model.p.A_mode_m2)
        return {"sig": float(nl_loss.sigma_fca_m2), "b_over_A": float(nl_loss.beta_tpa_m_per_W) / A_eff}

    def _nl_alpha(self, nl, state, P_slice):
        """Per-slice nonlinear loss coefficient alpha_nl = sigma_fca N_w + (beta/A_eff) P_slice [1/m]
        (FCA carrier-dependent + TPA intensity-dependent). P_slice is the local modal power [W]."""
        return nl["sig"] * self.model.wl_density_slices(state) + nl["b_over_A"] * P_slice

    def amplify(self, P_in, drive, *, nu_s_Hz: float = None, state0=None,
                return_traces: bool = False, ultrafast=None, transport_tau_s: float = 0.0,
                nl_loss=None, rc_tau_s: float = 0.0):
        """Amplify the input power waveform P_in (1-D array sampled at this engine's dt) at
        injection `drive`. `ultrafast` (an UltrafastCompression) adds the sub-ps SHB + carrier-
        heating gain compression on top of the carrier-density dynamics (None -> off, the
        carrier-density-only result). Returns a dict: t [s], P_in, P_out [W], gain_dB, and (if
        return_traces) the line-centre material gain per slice over time `g_zt` (nt, nz) and
        the carrier state. The amplifier starts in the unsaturated steady state at `drive`.

        transport_tau_s (default 0): the SCH (separate-confinement-heterostructure) carrier-TRANSPORT
        time -- the reduced drift-diffusion stage the lumped current injection omits. The injected
        current first fills an SCH reservoir N_sch that feeds the wetting layer with time tau_t
        (dN_sch/dt = I/(qV) - N_sch/tau_t; the WL sees the transport rate N_sch/tau_t), adding a pole
        that SLOWS the gain recovery / limits the modulation bandwidth. 0 -> instant transport, the
        lumped-injection result byte-identical; steady state is unchanged for any tau_t (N_sch ->
        I tau_t/(qV) -> the same WL feed). For the FULL spatially-resolved transport / current
        crowding, pass `drive` as a per-slice injection PROFILE I(z) from a DEVSIM drift-diffusion
        solve (init_slices + rhs_fields carry it).

        nl_loss (default None): a NonlinearLoss adding two-photon absorption (intensity-dependent,
        beta P/A_eff) and carrier-dependent free-carrier absorption (sigma_fca N_w) to the internal
        loss, so alpha_i becomes dynamic. None -> byte-identical.

        rc_tau_s (default 0): the electrical-parasitic RC time (pad/bond + junction, tau_RC = R C). The
        drive current is first low-passed dI_rc/dt = (I_drive - I_rc)/tau_RC (BACKWARD Euler, I_rc =
        (I_rc + (dt/tau)I)/(1+dt/tau), unconditionally stable for any dt) -- a first-order pole at
        f_RC = 1/(2 pi tau_RC) that limits the direct-current-modulation bandwidth BEFORE the SCH
        transport / injection stage (so RC and transport_tau cascade). 0 -> no RC; and for a CONSTANT
        drive I_rc stays at the steady value, so the result is byte-identical (RC acts only on a
        time-varying drive). I_rc inits at the steady drive (no startup transient)."""
        nu = float(nu_s_Hz) if nu_s_Hz is not None else self.nu_s
        P_in = np.asarray(P_in, dtype=np.float64)
        if P_in.ndim != 1 or P_in.size < 2:
            raise ValueError("amplify: P_in must be a 1-D waveform with >= 2 samples")
        nt = P_in.size
        # drive may be: a scalar I; a (nz,) spatial injection PROFILE I(z) (DEVSIM DD / current
        # crowding); or a (nt,) TIME-varying current I(t) (direct current modulation, nt != nz). The
        # carriers init at I(t=0) for the time form. Audit S3-4: when a 1-D drive has EXACTLY
        # nt == nz samples the two readings collide; the old code silently took the SPATIAL
        # reading and discarded the temporal modulation -- now it refuses the ambiguous case.
        dr = np.asarray(drive, dtype=np.float64)
        if dr.ndim == 1 and dr.size == nt and nt == self.nz:
            raise ValueError(
                "amplify: ambiguous 1-D drive -- its length ({}) equals BOTH the waveform sample "
                "count nt and the slice count nz, so it could be a time-varying current I(t) or a "
                "spatial injection profile I(z). Resample the waveform (nt != nz) or pass a "
                "scalar/explicitly-shaped drive.".format(dr.size))
        time_drive = bool(dr.ndim == 1 and dr.size == nt and nt != self.nz)
        I0 = dr[0] if time_drive else drive
        state = self.model.init_slices(self.nz, I0) if state0 is None else state0
        gam = self.model.gamma_confinement
        uf = self._uf_init(ultrafast)                         # ultrafast-compression state
        nl = self._nl_setup(nl_loss)                          # TPA + dynamic FCA (None -> byte-id)
        # SCH carrier-transport reservoir (opt-in): N_sch filled by I, feeding the WL with tau_t.
        tau_t = float(transport_tau_s)
        N_sch = np.asarray(I0, dtype=np.float64) * tau_t / self.model._qVa if tau_t > 0.0 else None
        rc_tau = float(rc_tau_s)                              # electrical-parasitic RC (opt-in)
        I_rc = np.asarray(I0, dtype=np.float64) if rc_tau > 0.0 else None   # inits at the steady drive
        # the device field starts empty (Pnode = 0); the first nz steps fill it (a startup
        # transient one transit time long -- t = nz*dt -- before P_out is meaningful)
        Pnode = np.zeros(self.nz + 1)
        P_out = np.empty(nt)
        g_zt = np.empty((nt, self.nz)) if return_traces else None
        h_uf = np.zeros(nt) if (return_traces and uf is not None) else None
        for n in range(nt):
            g = self.model.gain_per_m_slices(state, nu)       # (nz,) material gain
            g = self._uf_suppress(uf, g)                      # SHB/CH gain compression (if on)
            a_nl = 0.0 if nl is None else self._nl_alpha(nl, state, Pnode[:-1])  # TPA+FCA loss
            amp = np.exp((gam * g - self.alpha_i - a_nl) * self.dz)  # per-slice power amplification
            # method-of-characteristics shift: field at node k moves to node k+1, amplified
            new = np.empty_like(Pnode)
            new[0] = P_in[n]
            new[1:] = Pnode[:-1] * amp
            P_mid = 0.5 * (Pnode[:-1] + Pnode[1:])            # slice-average power this step
            self._uf_relax(uf, P_mid, nu)                     # drive the compression by S_conf
            I_now = dr[n] if time_drive else drive            # instantaneous injection this step
            if I_rc is not None:                              # electrical-parasitic RC low-pass
                a_rc = self.dt / rc_tau                       # BACKWARD Euler (unconditionally stable)
                I_rc = (I_rc + a_rc * np.asarray(I_now)) / (1.0 + a_rc)
                I_now = I_rc
            if N_sch is None:
                I_eff = I_now                                 # instant transport (lumped injection)
            else:                                             # SCH-transport-limited WL feed
                I_eff = N_sch / tau_t * self.model._qVa
                N_sch = N_sch + self.dt * (np.asarray(I_now) / self.model._qVa - N_sch / tau_t)
            state = self.model.step_slices(state, P_mid, self.dt, nu, I_eff)
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

    def saturation_curve(self, drive, P_in_W, *, nu_s_Hz: float = None, settle_transits: int = 60):
        """ABSOLUTE-power saturation characterization -- the observables an experimentalist reads off
        a measured gain-compression curve (which the density-only QDGainModel.saturation_curve does NOT
        give: it returns a confined photon DENSITY, not P_out in mW). For each CW input power in
        P_in_W [W] this runs the traveling-wave marcher to CW STEADY STATE and records the absolute
        output power and gain. Returns a dict:
          P_in_W, P_out_W [W];  P_in_dBm, P_out_dBm [dBm = 10 log10(P/1 mW)];
          gain_dB  : 10 log10(P_out/P_in) at CW steady state, per input;
          G0_dB    : the unsaturated (small-signal) gain = gain at the LOWEST input power;
          Pin_sat3dB_dBm  : the INPUT saturation power -- input where gain = G0 - 3 dB (interpolated;
                            nan if the sweep never reaches 3 dB compression);
          Psat_out_dBm    : the OUTPUT saturation power at that point (the usual datasheet 'P_sat,out').
        The curve follows the canonical saturable-amplifier law G = G0 exp(-(G - 1) P_in / P_sat); this
        method PRODUCES the absolute observable, but its MAGNITUDE rides on the (uncalibrated) generic
        material parameters -- fit P_sat to a measured device to turn it into a device prediction.
        settle_transits: CW samples per point = settle_transits * nz (long enough to reach steady
        state; the first nz samples are the device-fill transient and are excluded by reading P_out
        at the final, steady, sample)."""
        Pin = np.atleast_1d(np.asarray(P_in_W, dtype=np.float64)).copy()
        if Pin.ndim != 1 or Pin.size < 2 or np.any(Pin <= 0.0):
            raise ValueError("saturation_curve: P_in_W must be a 1-D array of >= 2 positive powers")
        order = np.argsort(Pin)                                # ascending input for monotone interp
        Pin = Pin[order]
        nt = max(8, int(settle_transits) * self.nz)
        Pout = np.empty(Pin.size)
        for i, P in enumerate(Pin):
            r = self.amplify(np.full(nt, float(P)), drive, nu_s_Hz=nu_s_Hz)
            Pout[i] = float(r["P_out"][-1])                    # CW steady-state output power [W]
        gain_dB = 10.0 * np.log10(Pout / Pin)
        G0_dB = float(gain_dB[0])                              # smallest input -> unsaturated gain
        to_dBm = lambda P: 10.0 * np.log10(np.asarray(P) / 1.0e-3)
        # -3 dB INPUT saturation power: gain_dB is monotone-decreasing in Pin (compression); find
        # where it first drops to G0 - 3 by linear interpolation in log10(Pin).
        target = G0_dB - 3.0
        Pin_sat3dB_dBm = np.nan
        Psat_out_dBm = np.nan
        if np.nanmin(gain_dB) <= target:
            logPin = np.log10(Pin)
            # gain_dB decreasing -> reverse so the x (gain) is increasing for np.interp
            log_pin_sat = float(np.interp(target, gain_dB[::-1], logPin[::-1]))
            Pin_sat_W = 10.0 ** log_pin_sat
            Pin_sat3dB_dBm = float(to_dBm(Pin_sat_W))
            Psat_out_dBm = float(Pin_sat3dB_dBm + target)      # P_out = P_in * G(=G0-3dB) in dB
        return {"P_in_W": Pin, "P_out_W": Pout, "P_in_dBm": to_dBm(Pin), "P_out_dBm": to_dBm(Pout),
                "gain_dB": gain_dB, "G0_dB": G0_dB, "Pin_sat3dB_dBm": Pin_sat3dB_dBm,
                "Psat_out_dBm": Psat_out_dBm}

    def psat_vs_detuning(self, drive, P_in_W, nu_s_list_Hz, *, settle_transits: int = 60):
        """Output saturation power P_sat,out [dBm] vs signal frequency -- the wavelength dependence of
        gain saturation an experimentalist maps by detuning the probe. Returns a dict: nu_s_Hz (the
        grid), G0_dB(nu) (unsaturated gain per detuning), Psat_out_dBm(nu). Off the gain peak the
        unsaturated gain falls and the saturation power shifts (the canonical blue-shift of P_sat with
        the gain roll-off), which the single-tone-at-nu0 saturation gates never exercised. Runs
        saturation_curve once per frequency."""
        nus = np.atleast_1d(np.asarray(nu_s_list_Hz, dtype=np.float64))
        G0 = np.empty(nus.size)
        Psat = np.empty(nus.size)
        for k, nu in enumerate(nus):
            sc = self.saturation_curve(drive, P_in_W, nu_s_Hz=float(nu), settle_transits=settle_transits)
            G0[k] = sc["G0_dB"]
            Psat[k] = sc["Psat_out_dBm"]
        return {"nu_s_Hz": nus, "G0_dB": G0, "Psat_out_dBm": Psat}

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
                         langevin: bool = False, seed=None, nl_loss=None, eta_in: float = 1.0):
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

        nl_loss (default None): a NonlinearLoss adding two-photon absorption (beta |A|^2/A_eff) and
        carrier-dependent free-carrier absorption (sigma_fca N_w) to the internal loss. None -> byte-
        identical. Not supported together with distributed GVD (gvd_segments>1 AND beta2!=0; raises);
        with beta2=0 the gvd_segments setting is a no-op and nl_loss runs normally.

        eta_in (default 1.0): the INPUT coupling efficiency (fiber-to-chip), 0 < eta_in <= 1. The input
        field is attenuated by sqrt(eta_in) at the input facet (a coupling LOSS of -10 log10(eta_in) dB)
        while the ASE is still generated internally at full gain -- so the FIBER-TO-FIBER noise figure
        the marcher reports is the internal-gain NF degraded by 1/eta_in (the standard input-loss-adds-
        to-NF result), the value a datasheet quotes, rather than the internal-gain-only NF. The output
        dict reports the FIBER-referred input (A_in / P_in are what you passed, before the facet loss),
        so gain and NF are fiber-to-fiber. eta_in = 1.0 (default) is byte-identical (no scaling). Acts
        only on the main (non-segmented) path. (The post-processing ase_noise.noise_figure already
        carries the same 1/eta_in factor; this wires it into the time-domain Langevin engine too.)

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
        if nl_loss is not None and nl_loss.active and beta2 != 0.0 and int(gvd_segments) > 1:
            raise NotImplementedError("amplify_coherent: nl_loss (TPA/FCA) with distributed GVD "
                                      "(gvd_segments>1) is not supported; use gvd_segments=1")
        if beta2 != 0.0 and int(gvd_segments) > 1:
            return self._amplify_coherent_segmented(
                A_in, drive, int(gvd_segments), beta2, nu_s_Hz=nu, alpha_lef=alpha_lef,
                ultrafast=ultrafast, line_filter=line_filter)
        nt = A_in.size
        eta = float(eta_in)                                   # input coupling efficiency (NF penalty)
        if not (0.0 < eta <= 1.0):
            raise ValueError("amplify_coherent: eta_in (input coupling efficiency) must be in (0, 1]")
        A_fiber = A_in                                        # fiber-referred input (reported as P_in)
        if eta != 1.0:
            A_in = A_in * np.sqrt(eta)                        # input-facet coupling LOSS into the device
        state = self.model.init_slices(self.nz, drive) if state0 is None else state0
        gam = self.model.gamma_confinement
        uf = self._uf_init(ultrafast)
        nl = self._nl_setup(nl_loss)                          # TPA + dynamic FCA (None -> byte-id)
        lf = self._line_filter_init(nu) if line_filter else None
        # Langevin spontaneous-emission noise (opt-in). Each slice each step adds a complex Gaussian
        # field increment of variance Gamma g_sp(z) h nu v_g * _exact_emit_factor(a_net dz)
        # (real + imag each half) -- the fluctuation-dissipation source whose downstream-amplified
        # accumulation telescopes EXACTLY to the analytic ASE PSD n_sp h nu (G-1) (audit C4-4:
        # the bare O(dz) source, injected after the slice gain, carried a ~lnG/(2 nz) deficit),
        # and whose phase diffusion gives the Henry (1 + alpha^2) linewidth. Reproducible via
        # seed; OFF (default) makes no RNG calls -> the deterministic engine is byte-identical.
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
        A_in_orig = A_fiber                                   # report FIBER-referred input (P_in/A_in)
        if beta2 != 0.0:
            half = self._gvd_phase(beta2, nt, 0.5 * self.L)
            A_in = np.fft.ifft(np.fft.fft(A_in) * half)
        Anode = np.zeros(self.nz + 1, dtype=np.complex128)
        A_out = np.empty(nt, dtype=np.complex128)
        for n in range(nt):
            new = np.empty_like(Anode)
            new[0] = A_in[n]
            a_eff = alpha if alpha_dyn is None else alpha_dyn(state)   # per-slice density-dependent
            a_nl = 0.0 if nl is None else self._nl_alpha(nl, state, np.abs(Anode[:-1]) ** 2)  # TPA+FCA
            if lf is None:
                g = self._uf_suppress(uf, self.model.gain_per_m_slices(state, nu))
                g_used = g                                     # per-slice net gain for the C4-4 emit factor
                amp = np.exp(0.5 * (gam * g * (1.0 - 1j * a_eff) - self.alpha_i - a_nl) * self.dz)
                new[1:] = Anode[:-1] * amp
            else:
                A_ref = Anode[:-1]
                kappa = self.model.line_kappa_slices(state, nu, lf["hw"])   # (nz, ng) live carriers
                src = kappa * A_ref[:, None]                        # (nz, ng) polarization source
                avg_p = lf["E_avg"] * lf["pol"] + src * lf["coef_avg"]      # transit-averaged pol
                sum_p = np.sum(avg_p, axis=1)                       # (nz,); CW: 2 Gamma_field(nu_s+f) A_ref
                g_un = self.model.gain_per_m_slices(state, nu)      # carrier real gain (no division)
                g_flat = self._uf_suppress(uf, g_un)
                g_used = g_flat                                # per-slice net gain for the C4-4 emit factor
                amp = np.exp(0.5 * (gam * g_flat * (1.0 - 1j * a_eff) - self.alpha_i - a_nl) * self.dz)
                # flat carrier gain (multiplicative, == OFF amp) + ADDITIVE dispersive correction:
                # the polarization sum_p minus its flat-gain equivalent is the resonant line
                # deviation (zero at the carrier). audit C4-6: the subtraction must be the
                # GS-BAND-ONLY gain -- the polarization poles carry ONLY the GS band, so
                # subtracting the full GS+ES g_un silently CANCELLED the entire ES band
                # (probe: ON gain -0.004 dB vs OFF 3.79 dB near the ES centre); the ES band
                # rides the flat multiplicative amp. Identical when sigma_pk_ES = 0.
                # Additive so the line radiates field into a null (sum_p != 0 there) -- no
                # divide-by-field, stable for modulated/nulling waveforms; first order in
                # the small per-slice deviation (~1e-3).
                g_gs_un = self.model.gain_per_m_slices_gs(state, nu)
                new[1:] = amp * (A_ref + 0.5 * gam * (sum_p - g_gs_un * A_ref) * self.dz)
                lf["pol"] = lf["E"] * lf["pol"] + src * lf["coef"]
            if lang is not None:                               # Langevin spontaneous-emission source
                gsp = self.model.emission_gain_per_m_slices(state, nu)   # (nz,) >= 0
                emit = _exact_emit_factor((gam * g_used - self.alpha_i - a_nl) * self.dz)
                sig = np.sqrt(lang["npref"] * gsp * emit)      # std of real (= imag) part per slice (C4-4)
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
                                 pdg_ratio: float = 1.0, tm_peak_shift_Hz: float = 0.0,
                                 state0=None, ultrafast=None):
        """Polarization-dependent gain (PDG): co-propagate the TE and TM complex envelopes through
        ONE shared carrier reservoir. The TM modal gain is pdg_ratio x the TE modal gain --
        pdg_ratio folds the TE/TM modal-confinement ratio and the QD material gain anisotropy into a
        single number (flat self-assembled QDs favour TE, so pdg_ratio < 1). Both polarizations
        deplete the SAME dots: the confined density that saturates the carriers is the modal-weighted
        total |A_TE|^2 + pdg_ratio |A_TM|^2, so a strong signal in one polarization CROSS-SATURATES
        the gain seen by the other (the physics a single-pol run cannot show). pdg_ratio = 1 (and
        tm_peak_shift = 0) makes the two pols gain-degenerate and each reduces EXACTLY to
        amplify_coherent (flat-gain branch).

        tm_peak_shift_Hz (default 0): VECTORIAL PDG -- the TM material-gain spectrum is the TE spectrum
        rigidly shifted by tm_peak_shift_Hz (strain / heavy-vs-light-hole splitting of the TM band), so
        the TM gain is evaluated at nu - tm_peak_shift while TE is at nu. This makes the PDG
        FREQUENCY-DEPENDENT -- PDG(nu) = 10 log10(G_TE/G_TM) varies across the band and reverses sign
        across the split -- instead of the flat (frequency-independent) scalar-ratio PDG. 0 -> the TM
        band coincides with TE -> the scalar-pdg_ratio behaviour (byte-identical).

        SCOPE: this resolves the TE/TM GAIN-SPECTRUM split (frequency-dependent small-signal PDG). The
        carrier SATURATION is LUMPED -- both pols deplete the SHARED reservoir at the TE/reservoir
        frequency nu via P_mid (the carrier step is taken at nu, not nu_tm), so the cross-saturation is
        a carrier-DENSITY effect; group-resolved TM-band SPECTRAL-HOLE burning at nu_tm is NOT resolved
        (it would require adding the TM stimulated term at nu_tm into the rate equations).

        Flat-gain path (no line filter / GVD -- those are single-pol features); the alpha(rho)
        density dependence applies identically to both pols. Returns a dict: A_te_out / A_tm_out
        (complex), P_te_out / P_tm_out, t, dt, state. PDG(dB) = 10 log10(G_TE / G_TM) on the steady
        tails (small-signal -> (1 - pdg_ratio) Gamma g L * 10/ln10 at tm_peak_shift = 0)."""
        nu = float(nu_s_Hz) if nu_s_Hz is not None else self.nu_s
        if alpha_lef is not None:
            alpha, alpha_dyn = float(alpha_lef), None
        else:
            alpha = float(getattr(self.model, "alpha_lef", 0.0))
            alpha_dyn = getattr(self.model, "alpha_lef_slices", None)
        r = float(pdg_ratio)
        nu_tm = nu - float(tm_peak_shift_Hz)                  # TM band evaluated at the shifted freq
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
            g_te = self._uf_suppress(uf, self.model.gain_per_m_slices(state, nu))
            g_tm = (g_te if tm_peak_shift_Hz == 0.0
                    else self._uf_suppress(uf, self.model.gain_per_m_slices(state, nu_tm)))
            phase = 1.0 - 1j * a_eff
            amp_te = np.exp(0.5 * (gam * g_te * phase - self.alpha_i) * self.dz)
            amp_tm = np.exp(0.5 * (r * gam * g_tm * phase - self.alpha_i) * self.dz)
            nte = np.empty_like(te)
            nte[0] = te_in[n]
            nte[1:] = te[:-1] * amp_te
            ntm = np.empty_like(tm)
            ntm[0] = tm_in[n]
            ntm[1:] = tm[:-1] * amp_tm
            # modal-weighted total power saturates the shared reservoir (TM contributes r x its power)
            P_te_mid = 0.5 * (np.abs(te[:-1]) ** 2 + np.abs(te[1:]) ** 2)
            P_tm_mid = 0.5 * r * (np.abs(tm[:-1]) ** 2 + np.abs(tm[1:]) ** 2)
            P_mid = P_te_mid + P_tm_mid
            self._uf_relax(uf, P_mid, nu)
            if tm_peak_shift_Hz == 0.0:
                state = self.model.step_slices(state, P_mid, self.dt, nu, drive)
            else:
                # audit C4-7: with a shifted TM band the TM field amplifies at nu_tm but
                # used to DEPLETE carriers at the TE frequency nu -- wrong groups by
                # L(nu-nu_j)/L(nu_tm-nu_j) and total depletion mis-scaled by ~g(nu)/g(nu_tm),
                # so TM gain compression was largely lost (probe: 97% of the compression
                # missed, photon number not conserved). Deplete each polarization at ITS
                # OWN frequency via the WDM per-channel-lineshape step (which raises for
                # eh_split, inheriting the correct guard).
                state = self.model.step_slices_wdm(state, [P_te_mid, P_tm_mid],
                                                   [nu, nu_tm], self.dt, drive)
            te, tm = nte, ntm
            te_out[n] = te[-1]
            tm_out[n] = tm[-1]
        t = np.arange(nt) * self.dt
        return {"t": t, "A_te_out": te_out, "A_tm_out": tm_out, "P_te_out": np.abs(te_out) ** 2,
                "P_tm_out": np.abs(tm_out) ** 2, "dt": self.dt, "state": state}

    def amplify_wdm(self, channels, drive, *, alpha_lef=None, state0=None):
        """WDM multi-channel coherent amplification: co-propagate SEVERAL signals at DISTINCT optical
        frequencies through ONE shared QD reservoir, each saturating the carriers via its OWN
        homogeneous lineshape -- WAVELENGTH-RESOLVED cross-gain saturation. `channels` is a list of
        (nu_k_Hz, A_k_in), A_k_in a complex envelope (nt,) (all equal length). Per z-slice each channel
        is amplified by exp(0.5 (Gamma g(state, nu_k)(1 - i alpha) - alpha_i) dz) with its OWN material
        gain g(nu_k), and the carriers advance by model.step_slices_wdm with EVERY channel's mid-slice
        power. So a strong channel bleaches the QD groups RESONANT WITH IT, and a probe far away
        (separation >> homogeneous linewidth but within the inhomogeneous band) sees REDUCED cross-gain
        saturation -- the inhomogeneous-broadening / spectral-hole-burning low-crosstalk advantage of a
        QD-SOA that the single-scalar-at-nu_s marcher (carrier back-reaction lumped to one frequency)
        cannot show. Returns {'channels': [{'nu_Hz','A_out','P_out'} per input], 't', 'dt', 'state'}.

        Reduces to amplify_coherent (flat-gain branch) for a single channel (to ~machine precision; the
        carrier stim is formed as L(nu) S vs S then L, a different float association). Excitonic models
        only (step_slices_wdm raises for eh_split). Flat gain + alpha index only -- no line filter / GVD
        / Langevin (those are single-pol single-band features); the leading nz output samples are the
        device-fill transient (window past them, as for amplify_coherent)."""
        if len(channels) < 1:
            raise ValueError("amplify_wdm: need >= 1 channel")
        nus = [float(nu) for nu, _ in channels]
        fin = [np.asarray(A, dtype=np.complex128) for _, A in channels]
        nt = fin[0].size
        if any(A.ndim != 1 or A.size != nt for A in fin) or nt < 2:
            raise ValueError("amplify_wdm: all channel envelopes must be equal-length 1-D >= 2 samples")
        if alpha_lef is not None:
            alpha, alpha_dyn = float(alpha_lef), None
        else:
            alpha = float(getattr(self.model, "alpha_lef", 0.0))
            alpha_dyn = getattr(self.model, "alpha_lef_slices", None)
        state = self.model.init_slices(self.nz, drive) if state0 is None else state0
        gam = self.model.gamma_confinement
        nch = len(channels)
        nodes = [np.zeros(self.nz + 1, dtype=np.complex128) for _ in range(nch)]
        outs = [np.empty(nt, dtype=np.complex128) for _ in range(nch)]
        for n in range(nt):
            a_eff = alpha if alpha_dyn is None else alpha_dyn(state)
            new_nodes = []
            P_mid = []
            for c in range(nch):
                g = self.model.gain_per_m_slices(state, nus[c])
                amp = np.exp(0.5 * (gam * g * (1.0 - 1j * a_eff) - self.alpha_i) * self.dz)
                nd = nodes[c]
                new = np.empty_like(nd)
                new[0] = fin[c][n]
                new[1:] = nd[:-1] * amp
                P_mid.append(0.5 * (np.abs(nd[:-1]) ** 2 + np.abs(nd[1:]) ** 2))
                new_nodes.append(new)
            state = self.model.step_slices_wdm(state, P_mid, nus, self.dt, drive)
            nodes = new_nodes
            for c in range(nch):
                outs[c][n] = nodes[c][-1]
        t = np.arange(nt) * self.dt
        chans = [{"nu_Hz": nus[c], "A_out": outs[c], "P_out": np.abs(outs[c]) ** 2}
                 for c in range(nch)]
        return {"channels": chans, "t": t, "dt": self.dt, "state": state}

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
                emit = _exact_emit_factor((gam * g - self.alpha_i) * self.dz)
                sg = np.sqrt(lang["npref"] * gsp * emit)       # exact-emit variance (audit C4-4)
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
