"""Calibrate the QD-SOA model to a measured datasheet -- the step that turns the generic-parameter gain
core into a DEVICE-matched parameter set. The calibration is DATASHEET-AGNOSTIC: any device is described
by a DeviceTargets (the universal spec set every SOA datasheet provides -- peak wavelength, small-signal
gain, NET -3 dB optical bandwidth, saturation output power, NF bound, drive, chip length) and fitted by
calibrate_device(targets). The Innolume BOA1310060CC600MXXXX (1310 nm QD booster) is ONE preset among
several; calibrate_innolume_boa1310 remains as its thin wrapper.

WHAT THIS FITS (the STATIC / CW load-bearing axes the datasheet constrains): peak wavelength (read-off),
gain bandwidth (fwhm_inhom -- SEE THE C4-8 CAVEAT BELOW), small-signal gain magnitude (sigma_pk, the effective free factor of the
degenerate product Gamma*N_q*mu_GS*sigma_pk*L -- N_q is FIXED at a standard QD value, recorded as the
convention), the absolute saturation output power P_sat (A_mode), and the GS/ES band split (dE_ES_GS,
enabling the two-band ASE). The fit + the validation use only STEADY-STATE physics (the gain core's
saturation_curve gives the local saturated gain g_QD(P); the device output is the z-integral
dP/dz = (Gamma g_QD(P) - alpha_i) P) -- the time-domain marcher OVERFLOWS at this device's 35 dB
single-pass gain, and the steady-state path is exact for the CW datasheet numbers anyway.

WHAT THIS DOES NOT CALIBRATE (no datasheet data -- left at flagged defaults, UNCALIBRATED): the linewidth-
enhancement factor alpha_lef (no chirp/FWM data), the carrier kinetic times tau_cap/tau_esc/tau_ES_GS
(no pump-probe / modulation-bandwidth), RIN / linewidth, NF(lambda)/NF(G), TPA/FCA, and the thermal
slopes. Pin those with a pump-probe gain-recovery trace, an FWM / chirp-asymmetry measurement, an RF-RIN
measurement, a spectral-NF measurement, and gain-vs-temperature data respectively.

BANDWIDTH (audit C4-8, RESOLVED): the datasheet's '-3 dB gain bandwidth' is a NET amplifier-gain
observable. The fit now honors that: after the peak-gain stage, a NET-BANDWIDTH CO-FIT jointly
tunes the ES-band strength ratio (sigma_pk_ES / sigma_pk -- the GS+ES two-band overlap is what
physically flattens a QD amplifier's gain top) and the inhomogeneous width until the FULL GS+ES
net -3 dB bandwidth at the operating gain matches the datasheet number, while re-pinning the
peak gain each step (exact in one step: at S=0 the populations are cross-section-independent, so
the net spectrum scales linearly in sigma_pk at fixed ES ratio). The report's 'bandwidth_nm' is
now the NET -3 dB bandwidth (the datasheet semantic); the fitted material half-max width is
reported separately as 'material_fwhm_nm' (it comes out ~2-3x WIDER than the net band -- the
high-gain narrowing now runs the honest direction). Off-peak spectral predictions (WDM channels
near the band edges, XGM/ASE/OSNR spectra) therefore see the datasheet band, not a 3.6x-narrowed
one. Set fit_net_bandwidth=False to recover the legacy material-width interpretation.

SI; ASCII; exp(-i omega t).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK, Q_E   # single-source CODATA (was re-declared here)
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams


# Innolume BOA1310060CC600MXXXX datasheet (99-S01-273-01), 25 C, 2000 mA operating point.
INNOLUME_BOA1310_TARGETS = {
    "peak_nm": 1310.0,            # gain peak wavelength
    "G0_dB": 35.0,               # small-signal (Pin=-20 dBm) chip gain, typ
    "bandwidth_nm": 60.0,        # -3 dB gain bandwidth, typ
    "Psat_out_dBm": 23.2,        # -3 dB saturation output power @ 2000 mA (20.3/24.7 at 1/3 A)
    "NF_dB_max": 5.0,            # noise figure @ Pin=-20 dBm, excluding input coupling
    "ase_es_nm": 1210.0,         # ES/WL ASE band (blue of the GS signal band)
    "drive_A": 2.0, "L_m": 8.0e-3, "T_K": 298.0, "facet_R": 1.0e-5,
}


@dataclass(frozen=True)
class DeviceTargets:
    """A datasheet-agnostic SOA calibration target set: the universal spec set every SOA
    datasheet provides, at its quoted operating point (CW, case temperature T_K). All optical
    specs are CHIP/net-gain observables (fiber-to-fiber sheets: subtract the coupling first).
    peak_nm = gain-peak wavelength; G0_dB = small-signal gain at the peak; net_bw_3dB_nm = the
    -3 dB OPTICAL bandwidth of the NET small-signal gain (the datasheet semantic); Psat_out_dBm
    = -3 dB-compression saturation OUTPUT power; drive_A / L_m the operating current and chip
    length; NF_dB_max an upper NF bound (constrains alpha_i only loosely); facet_R the residual
    facet reflectivity (ripple); ase_es_nm an optional second (blue) ASE band centre -- for a QD
    device the ES band, which pins the GS/ES split dE and enables the two-band model (None ->
    the ES split stays at the QDGainParams default and the ES band is still available as a
    bandwidth-fit knob only if fit_net_bandwidth requests it)."""
    name: str
    peak_nm: float
    G0_dB: float
    net_bw_3dB_nm: float
    Psat_out_dBm: float
    drive_A: float
    L_m: float
    T_K: float = 298.0
    NF_dB_max: Optional[float] = None
    facet_R: float = 1.0e-5
    ase_es_nm: Optional[float] = None


# The Innolume sheet as a DeviceTargets (the dict above is the raw-sheet record, kept for
# back-compat); further presets are appended below (SOA_PRESETS).
INNOLUME_BOA1310 = DeviceTargets(
    name="Innolume BOA1310060", peak_nm=1310.0, G0_dB=35.0, net_bw_3dB_nm=60.0,
    Psat_out_dBm=23.2, drive_A=2.0, L_m=8.0e-3, T_K=298.0, NF_dB_max=5.0,
    facet_R=1.0e-5, ase_es_nm=1210.0)


@dataclass(frozen=True)
class CalibratedDevice:
    """A device-matched QD-SOA: the fitted gain params + the device-level constants (length, internal
    loss, operating drive, signal frequency) that the datasheet pins but QDGainParams does not hold."""
    params: QDGainParams
    length_m: float
    alpha_i_per_m: float
    drive_A: float
    nu0_Hz: float
    name: str
    report: dict


def _net_gain_spectrum(model, drive, nu, alpha_i, L):
    """Unsaturated NET chip gain [dB] over the length L at frequencies nu (S_conf -> 0)."""
    y = model.steady_state(drive, S_conf_m3=0.0)
    g = model.material_gain_per_m(model.rho_GS(y), np.atleast_1d(nu))
    return (10.0 / np.log(10.0)) * (model.gamma_confinement * g - alpha_i) * L


def _g0_dB(model, drive, nu0, alpha_i, L):
    return float(_net_gain_spectrum(model, drive, np.array([nu0]), alpha_i, L)[0])


def _bandwidth_nm(model, drive, nu0, alpha_i, L):
    """-3 dB bandwidth [nm] of the MATERIAL gain coefficient g(nu) (half-max of g) -- the INTRINSIC
    gain width (~ the inhomogeneous FWHM), gain-level-independent and consistent with the datasheet's
    visible ~120 nm gain-spectrum span. NB the NET amplifier-gain -3 dB at the 35 dB small-signal peak
    is much NARROWER (high-gain spectral narrowing makes a 3 dB drop a tiny fractional g drop), so
    matching the datasheet 60 nm there would force an unphysically wide distribution -- the intrinsic
    interpretation is the physical one."""
    nu = nu0 + np.linspace(-30e12, 30e12, 1201)
    y = model.steady_state(drive, S_conf_m3=0.0)
    g = model.material_gain_per_m(model.rho_GS(y), nu)
    above = nu[g >= 0.5 * g.max()]
    if above.size < 2:
        return 0.0
    lam = C_LIGHT / nu0
    return float((above.max() - above.min()) * lam * lam / C_LIGHT * 1.0e9)


def device_saturation_curve(model, drive, nu, alpha_i, L, P_in_W, nz=2000):
    """Absolute CW saturation curve P_out(P_in) [W] by STEADY-STATE z-integration (robust at high gain
    where the time-domain marcher overflows). g_QD(P) = the gain core's local saturated material gain at
    local power P (QDGainModel.saturation_curve, steady state at each photon density); the device output
    integrates dP/dz = (Gamma g_QD(P) - alpha_i) P over L by RK4. Returns (P_in_W, P_out_W)."""
    P_in = np.atleast_1d(np.asarray(P_in_W, dtype=np.float64))
    # local saturated gain on a wide power grid bracketing P_in .. P_out
    P_grid = np.logspace(np.log10(P_in.min()) - 1.0, np.log10(P_in.max() * 1.0e4) + 1.0, 400)
    g_loc, _S = model.saturation_curve(drive, P_grid, nu_s_Hz=nu)
    gam = model.gamma_confinement
    gQD = lambda P: np.interp(P, P_grid, g_loc)
    dz = L / int(nz)
    # RK4 vectorized ACROSS the P_in array (audit 6.2 perf): every input power takes the same
    # step count/size, so the former serial per-P_in python loop is just the same elementwise
    # float ops in vector lanes (np.interp is elementwise too) -- BIT-identical per P_in
    # (probe-verified), ~11x on this leg.
    P = P_in.astype(np.float64, copy=True)
    for _ in range(int(nz)):
        k1 = (gam * gQD(P) - alpha_i) * P
        k2 = (gam * gQD(P + 0.5 * dz * k1) - alpha_i) * (P + 0.5 * dz * k1)
        k3 = (gam * gQD(P + 0.5 * dz * k2) - alpha_i) * (P + 0.5 * dz * k2)
        k4 = (gam * gQD(P + dz * k3) - alpha_i) * (P + dz * k3)
        P = P + dz / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return P_in, P


def _psat_out_dBm(model, drive, nu, alpha_i, L):
    """Output -3 dB saturation power [dBm] from the steady-state device saturation curve."""
    P_in = np.logspace(-9.0, 0.0, 40)                       # 1 nW .. 1 W input
    P_in, P_out = device_saturation_curve(model, drive, nu, alpha_i, L, P_in)
    G_dB = 10.0 * np.log10(P_out / P_in)
    G0 = G_dB[0]
    target = G0 - 3.0
    if np.nanmin(G_dB) > target:
        return np.nan, G0
    # output power at the -3 dB compression point (interp on the monotone-decreasing gain)
    logPout = np.log10(P_out)
    log_pout_sat = float(np.interp(target, G_dB[::-1], logPout[::-1]))
    return float(10.0 * np.log10((10.0 ** log_pout_sat) / 1.0e-3)), float(G0)


def _net_gain_spectrum_full(model, drive, nu, alpha_i, L):
    """Unsaturated NET chip gain [dB] over L at nu, INCLUDING the ES band (total_material_gain).
    _net_gain_spectrum is the GS-only sibling kept for the legacy fit path."""
    y = model.steady_state(drive, S_conf_m3=0.0)
    g = model.total_material_gain(model.rho_ES(y), model.rho_GS(y), np.atleast_1d(nu))
    return (10.0 / np.log(10.0)) * (model.gamma_confinement * g - alpha_i) * L


def _resolved_groups(n_min: int, fwhm_inhom_hz: float, *, fwhm_hom_hz: float = 1.0e12,
                     span_sigma: float = 3.0) -> int:
    """Smallest ODD group count that keeps the inhomogeneous-comb spacing <= HALF the
    homogeneous FWHM -- the smooth-band resolution condition. ADVERSARIAL-VERIFIER FIX: the
    co-fit originally widened fwhm_inhom ~6.5x at a FIXED n_groups=41, leaving the ensemble a
    comb of RESOLVED spikes (group spacing 2.08 THz >> 1 THz homogeneous FWHM); the 'bandwidth'
    was then the max-min envelope of 6 disjoint comb fingers and the peak gain a sampling spike
    at nu0 (converged physics with the same params: 18.7 dB / 80.9 nm smooth vs the reported
    35 dB / 59 nm). Auto-scaling the group count with the fitted width makes every fit
    evaluation spectrally converged."""
    spacing_target = 0.5 * fwhm_hom_hz
    sigma = fwhm_inhom_hz / 2.35482
    need = int(np.ceil(2.0 * span_sigma * sigma / spacing_target)) + 1
    n = max(int(n_min), need)
    return n if n % 2 == 1 else n + 1


def _contiguous_band_nm(nu_g, net_dB, i_ref: int, drop_dB: float = 3.0) -> float:
    """CONTIGUOUS -drop_dB bandwidth [nm] around sample i_ref: expand left/right while the
    spectrum stays within drop_dB of net_dB[i_ref]. A threshold-set max-min measure would count
    disjoint ripple fingers as 'band' (the comb artifact _resolved_groups fixes); the contiguous
    measure is the physical passband and can never be fooled by ripple."""
    thr = float(net_dB[i_ref]) - drop_dB
    lo = i_ref
    while lo - 1 >= 0 and net_dB[lo - 1] >= thr:
        lo -= 1
    hi = i_ref
    while hi + 1 < net_dB.size and net_dB[hi + 1] >= thr:
        hi += 1
    if hi <= lo:
        return 0.0
    return float((C_LIGHT / nu_g[lo] - C_LIGHT / nu_g[hi]) * 1.0e9)


def _net_bw_and_peak(nu_g, net_dB):
    """(CONTIGUOUS -3 dB net bandwidth [nm] around the peak, peak wavelength [nm], peak net
    gain [dB]) from a net spectrum."""
    i_pk = int(np.argmax(net_dB))
    pk = float(net_dB[i_pk])
    lam_pk = float(C_LIGHT / nu_g[i_pk] * 1.0e9)
    return _contiguous_band_nm(nu_g, net_dB, i_pk), lam_pk, pk


def calibrate_device(targets, *, N_q_m3=5.0e22, alpha_i_per_m=300.0, n_groups=41,
                     fit_net_bandwidth=True, es_ratio_max=1.5, verbose=False):
    """Fit the QD-SOA model to ANY datasheet described by a DeviceTargets (module header).
    N_q_m3 is FIXED at a standard QD value (the degenerate product Gamma*N_q*mu_GS*sigma_pk is
    broken by holding N_q, Gamma, mu_GS and fitting sigma_pk -- the EFFECTIVE-PRODUCT
    convention); alpha_i_per_m is only loosely bounded by the NF spec.

    fit_net_bandwidth=True (default): the NET-BANDWIDTH CO-FIT. Key structural fact it exploits:
    at S_conf=0 the level populations are independent of the optical cross-sections, so for a
    given inhomogeneous width ONE steady_state yields GS/ES basis spectra and the net gain is
    exactly linear in sigma_pk at fixed ES ratio r = sigma_pk_ES/sigma_pk -- re-pinning the peak
    to G0_dB is a one-step exact update, and the (r, fwhm_inhom) search for the net -3 dB
    bandwidth is pure arithmetic on the basis spectra (no solver calls in the bisection).
    Knob roles: r extends/flattens the BLUE side via the ES band (the physical GS+ES two-band
    overlap of real QD amplifiers); fwhm_inhom fills the GS-ES valley and sets the overall
    envelope. A peak-location guard keeps the fitted net peak within ~12 nm of the datasheet
    peak (the ES band must broaden, not hijack, the peak).

    fit_net_bandwidth=False: the LEGACY material-width interpretation (pre-co-fit behavior:
    datasheet bandwidth mapped onto the material half-max width; net band comes out ~3-4x
    narrower than the sheet -- audit C4-8). Returns a CalibratedDevice either way."""
    t = targets if isinstance(targets, DeviceTargets) else DeviceTargets(**targets)
    nu0 = C_LIGHT / (t.peak_nm * 1.0e-9)
    L, drive = t.L_m, t.drive_A
    use_es = t.ase_es_nm is not None
    dE = (H_PLANCK * (C_LIGHT / (t.ase_es_nm * 1.0e-9) - nu0) / Q_E if use_es else 0.060)
    off = (10.0 / np.log(10.0)) * alpha_i_per_m * L          # net_dB = (Gamma g L)_dB - off
    lam0 = C_LIGHT / nu0
    SIG_REF = 1.0e-18                                        # reference cross-section for bases

    def mk(fwhm_inh, sig, A_mode, sig_es):
        # n_groups auto-scales with the width so every evaluation is spectrally RESOLVED
        # (_resolved_groups; the comb-artifact fix). n_groups is the FLOOR, not the count.
        p = QDGainParams(n_groups=_resolved_groups(n_groups, fwhm_inh), nu0_Hz=nu0,
                         fwhm_inhom_Hz=fwhm_inh, sigma_pk_m2=sig,
                         A_mode_m2=A_mode, T_K=t.T_K, sigma_pk_ES_m2=sig_es, dE_ES_GS_eV=dE,
                         N_q_m3=N_q_m3)
        return QDGainModel(p.with_detailed_balance_taus())

    A_mode = 0.4e-12
    if not fit_net_bandwidth:
        # ---- LEGACY staged fit (material-width interpretation, audit C4-8 behavior) ----
        fwhm, sig, sig_es = 13.0e12, 2.0e-18, 0.0
        for _ in range(6):
            m = mk(fwhm, sig, A_mode, sig_es)
            G0 = _g0_dB(m, drive, nu0, alpha_i_per_m, L)
            sig = sig * (t.G0_dB + off) / (G0 + off)
            m = mk(fwhm, sig, A_mode, sig_es)
            bw = _bandwidth_nm(m, drive, nu0, alpha_i_per_m, L)
            fwhm = fwhm * (t.net_bw_3dB_nm / bw) ** 0.7
        r_es = 0.3 if use_es else 0.0
    else:
        # ---- NET-BANDWIDTH CO-FIT ----
        bw_hz = t.net_bw_3dB_nm * 1.0e-9 * C_LIGHT / lam0 ** 2      # target width in frequency
        dE_hz = dE * Q_E / H_PLANCK
        fwhm = max(bw_hz, 0.8 * dE_hz) if use_es else 1.3 * bw_hz   # seed: bridge the GS-ES gap
        sig = 2.0e-18
        nu_g = nu0 + np.linspace(-60e12, 60e12, 2401)      # covers GS band + ES band + wings

        def bases(fwhm_inh):
            """One steady_state -> (gamma, gGS(nu), gES(nu)) per SIG_REF of cross-section."""
            m = mk(fwhm_inh, SIG_REF, A_mode, SIG_REF)       # r_es = 1 reference
            y = m.steady_state(drive, S_conf_m3=0.0)
            gGS = m.material_gain_per_m(m.rho_GS(y), nu_g)
            gALL = m.total_material_gain(m.rho_ES(y), m.rho_GS(y), nu_g)
            return m.gamma_confinement, gGS, gALL - gGS

        i0 = int(np.argmin(np.abs(nu_g - nu0)))              # the GS/datasheet peak sample

        def measure(gam, gGS, gES, r):
            """Pin the gain AT nu0 (exact one step: net(nu0)+off is strictly proportional to
            sigma), then measure (sig, bw_nm, excess). The -3 dB band is measured RELATIVE TO
            net(nu0) -- the datasheet's peak -- and excess = net_max - net(nu0) [dB] is the
            ES-hijack indicator. With excess capped at ~0 the -3 dB threshold is FIXED at
            net(nu0)-3, so adding ES gain anywhere only ADDS band -> bw is strictly monotone
            in r on [0, r_cap] (a nonzero excess allowance breaks that: the ES lobe raises the
            global peak and with it the -3 dB level, which can NARROW the measured band)."""
            base = gam * (gGS + r * gES)                     # per SIG_REF, before loss
            net0_ref = (10.0 / np.log(10.0)) * (base[i0] - alpha_i_per_m) * L
            # net0(sig) + off = (10/ln10) * gam*(...)[i0] * (sig/SIG_REF) * L  -> exact pin:
            sig_p = SIG_REF * (t.G0_dB + off) / max(net0_ref + off, 1e-30)
            net = (10.0 / np.log(10.0)) * (base * (sig_p / SIG_REF) - alpha_i_per_m) * L
            bw = _contiguous_band_nm(nu_g, net, i0)          # contiguous: ripple-proof measure
            return sig_p, bw, float(net.max() - net[i0])

        # The ES lobe must BROADEN the band, not hijack the peak (the failure mode of a naive
        # bandwidth bisection: past a critical r the global max jumps to the ES wavelength and
        # the measured -3 dB band collapses around the WRONG lobe). So per fwhm: (1) find
        # r_cap = largest r with excess <= 0.3 dB (monotone in r -> bisection), (2) tune r in
        # [0, r_cap] for the bandwidth (monotone on that interval), (3) walk fwhm outer-loop.
        r_es, bw_got = 0.0, 0.0
        for rnd in range(10):
            gam, gGS, gES = bases(fwhm)
            sig, bw0, _ = measure(gam, gGS, gES, 0.0)
            if not use_es:
                r_cap = 0.0
            else:
                _, _, exc_hi = measure(gam, gGS, gES, es_ratio_max)
                if exc_hi <= 0.05:
                    r_cap = es_ratio_max
                else:
                    lo, hi = 0.0, es_ratio_max                # bisect the hijack boundary
                    for _ in range(20):
                        mid = 0.5 * (lo + hi)
                        _, _, exc = measure(gam, gGS, gES, mid)
                        if exc <= 0.05:
                            lo = mid
                        else:
                            hi = mid
                    r_cap = lo
            _, bw_cap, _ = measure(gam, gGS, gES, r_cap)
            if bw0 >= t.net_bw_3dB_nm + 1.5:                  # GS alone too wide: narrow ensemble
                r_es, bw_got = 0.0, bw0
                fwhm = float(np.clip(fwhm * (t.net_bw_3dB_nm / bw0) ** 0.7, 2e12, 40e12))
                continue
            if bw_cap < t.net_bw_3dB_nm - 1.5:                # even max shoulder short: widen
                r_es, bw_got = r_cap, bw_cap
                fwhm = float(np.clip(fwhm * (t.net_bw_3dB_nm / max(bw_cap, 1.0)) ** 0.5,
                                     2e12, 40e12))
                continue
            lo, hi = 0.0, r_cap                               # bw monotone on [0, r_cap]
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                _, bw_m, _ = measure(gam, gGS, gES, mid)
                if bw_m < t.net_bw_3dB_nm:
                    lo = mid
                else:
                    hi = mid
            r_es = 0.5 * (lo + hi)
            sig, bw_got, _ = measure(gam, gGS, gES, r_es)
            if abs(bw_got - t.net_bw_3dB_nm) <= 1.5:
                break
            fwhm = float(np.clip(fwhm * (t.net_bw_3dB_nm / max(bw_got, 1.0)) ** 0.5,
                                 2e12, 40e12))
        # final exact re-pin at the settled (fwhm, r_es)
        gam, gGS, gES = bases(fwhm)
        sig, bw_got, _ = measure(gam, gGS, gES, r_es)

    # ---- A_mode -> saturation output power ----
    # ONE saturated-gain scan serves EVERY A_mode iteration: the model's local gain depends on
    # the confined PHOTON DENSITY S = P/(A_mode v_g h nu), so the g(S) relation measured at a
    # reference A_mode maps to any other A_mode by pure rescaling of the power axis -- the
    # 400-solve saturation scan (the fit's dominant cost, ~8x here) runs once. The final report
    # still calls the unshortcut _psat_out_dBm as an independent check of this identity.
    m_ref = mk(fwhm, sig, A_mode, r_es * sig)
    P_grid = np.logspace(-10.0, 1.0, 400)
    g_loc, _S = m_ref.saturation_curve(drive, P_grid, nu_s_Hz=nu0)
    S_grid = P_grid / (A_mode * m_ref.v_g * H_PLANCK * nu0)      # photon-density axis
    gam_c = m_ref.gamma_confinement

    def _psat_from_scan(A_now):
        conv = A_now * m_ref.v_g * H_PLANCK * nu0                # P = S * conv
        P_in = np.logspace(-9.0, 0.0, 40)
        gQD = lambda P: np.interp(P / conv, S_grid, g_loc)       # noqa: E731
        nz = 2000
        dz = L / nz
        P = P_in.astype(np.float64, copy=True)
        for _ in range(nz):
            k1 = (gam_c * gQD(P) - alpha_i_per_m) * P
            k2 = (gam_c * gQD(P + 0.5 * dz * k1) - alpha_i_per_m) * (P + 0.5 * dz * k1)
            k3 = (gam_c * gQD(P + 0.5 * dz * k2) - alpha_i_per_m) * (P + 0.5 * dz * k2)
            k4 = (gam_c * gQD(P + dz * k3) - alpha_i_per_m) * (P + dz * k3)
            P = P + dz / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        G_dB = 10.0 * np.log10(P / P_in)
        target = G_dB[0] - 3.0
        if np.nanmin(G_dB) > target:
            return np.nan
        logPout = np.log10(P)
        return float(10.0 * np.log10(10.0 ** np.interp(target, G_dB[::-1], logPout[::-1])
                                     / 1.0e-3))

    for _ in range(8):
        ps = _psat_from_scan(A_mode)
        if not np.isfinite(ps):
            A_mode *= 0.5
            continue
        A_mode = A_mode * 10.0 ** ((t.Psat_out_dBm - ps) / 10.0)

    # ---- final report ----
    m = mk(fwhm, sig, A_mode, r_es * sig)
    ps, _ = _psat_out_dBm(m, drive, nu0, alpha_i_per_m, L)
    mat_bw = _bandwidth_nm(m, drive, nu0, alpha_i_per_m, L)
    nu_rep = nu0 + np.linspace(-45e12, 45e12, 1801)
    net_dB = _net_gain_spectrum_full(m, drive, nu_rep, alpha_i_per_m, L)
    net_bw_nm, lam_pk_nm, G0_net = _net_bw_and_peak(nu_rep, net_dB)
    # 'bandwidth_nm' now carries the NET -3 dB bandwidth (the datasheet semantic) on the co-fit
    # path; on the legacy path it remains the material width (the pre-co-fit alias).
    report = {"G0_dB": G0_net, "bandwidth_nm": net_bw_nm if fit_net_bandwidth else mat_bw,
              "material_fwhm_nm": mat_bw, "net_3dB_bw_nm": net_bw_nm,
              "peak_lambda_nm": lam_pk_nm, "Psat_out_dBm": ps, "sigma_pk_m2": sig,
              "es_ratio": r_es, "sigma_pk_ES_m2": r_es * sig, "fwhm_inhom_Hz": fwhm,
              "A_mode_m2": A_mode, "dE_ES_GS_eV": dE, "N_q_m3": N_q_m3,
              "alpha_i_per_m": alpha_i_per_m, "fit_net_bandwidth": bool(fit_net_bandwidth)}
    if verbose:
        for k, v in report.items():
            print("  {:16s} {}".format(k, v))
    return CalibratedDevice(params=m.p, length_m=L, alpha_i_per_m=alpha_i_per_m, drive_A=drive,
                            nu0_Hz=nu0, name="{} (static/CW fit)".format(t.name), report=report)


def calibrate_innolume_boa1310(N_q_m3=5.0e22, alpha_i_per_m=300.0, n_groups=41, verbose=False,
                               fit_net_bandwidth=True):
    """The Innolume BOA1310060 preset through the generic calibrate_device (module header). With
    the default fit_net_bandwidth=True the device's NET -3 dB bandwidth matches the datasheet's
    60 nm (the C4-8 resolution); pass False for the legacy material-width interpretation."""
    return calibrate_device(INNOLUME_BOA1310, N_q_m3=N_q_m3, alpha_i_per_m=alpha_i_per_m,
                            n_groups=n_groups, fit_net_bandwidth=fit_net_bandwidth,
                            verbose=verbose)


# Datasheet-agnostic preset registry. The Thorlabs BOA1004P (QW 1550 PM booster) targets are
# recorded for the QW gain core (optics.soa.qw_gain) -- fitting a QW device with the QD ensemble
# core is possible but physically mislabeled; the entry documents the sheet either way.
SOA_PRESETS = {
    "innolume_boa1310": INNOLUME_BOA1310,
    "thorlabs_boa1004p": DeviceTargets(
        name="Thorlabs BOA1004P (QW)", peak_nm=1550.0, G0_dB=27.0, net_bw_3dB_nm=85.0,
        Psat_out_dBm=15.0, drive_A=0.6, L_m=1.5e-3, T_K=298.0, NF_dB_max=7.5,
        facet_R=1.0e-5, ase_es_nm=None),
}


@dataclass(frozen=True)
class InferredDynamic:
    """A DYNAMIC parameter INFERRED from the static/CW calibration -- NOT measured. value/unit, the
    confidence (HIGH/MEDIUM/LOW with the reason), and the method/physics used. These are physically-
    motivated ESTIMATES to seed the dynamic phases (18-33) instead of the generic defaults; every one
    must be flagged inferred-not-measured and refined by the named dynamic measurement before any
    dynamic/coherent prediction is trusted."""
    value: float
    unit: str
    confidence: str
    method: str


def infer_dynamics_from_cw(device, *, slow_div_deg=6.0, fast_div_deg=27.0):
    """Infer DYNAMIC parameters from the calibrated static/CW set (device = a CalibratedDevice). Returns a
    dict of InferredDynamic. EACH IS AN ESTIMATE, NOT A MEASUREMENT -- the CW datasheet does not contain
    dynamic data; these exploit physical LINKS between CW observables and dynamic quantities, and carry an
    explicit per-parameter confidence. Inputs beyond the device: the datasheet far-field beam divergence
    (slow 6 deg / fast 27 deg FWHM) used to pin the mode area.

    The chain rests on the textbook SOA relation P_sat = h nu A_eff / (Gamma a tau_eff): the CW P_sat (and
    the divergence-derived A_eff and the model differential gain a) jointly pin tau_eff. The Phase-34 fit
    is left UNTOUCHED (it reproduces every CW number via an EFFECTIVE A_mode that absorbed the a*tau
    degeneracy); these inferred values are the PHYSICAL interpretation of that same saturation."""
    import numpy as _np
    m = QDGainModel(device.params)
    nu0, drive = device.nu0_Hz, device.drive_A
    lam = C_LIGHT / nu0
    out = {}

    # --- (1) effective mode area A_eff from the far-field divergence -------------------------------------
    # Gaussian far-field: a 1/e^2-intensity near-field radius w gives a 1/e^2 half-angle theta = lam/(pi w).
    # The datasheet quotes the FWHM full-angle; for a Gaussian, FWHM_full = 1.18 * theta_(1/e2,half), so
    # w = lam / (pi * theta_FWHM_full / 1.18). A_eff (1/e^2 intensity) = pi * w_slow * w_fast.
    # CONFIDENCE: MEDIUM. The geometric mode size from divergence is well-defined (HIGH), but mapping it to
    # the model's effective SATURATION area carries a ~2x convention factor (effective-area vs 1/e^2 area)
    # and assumes a single Gaussian transverse mode (a real ridge mode is only approximately Gaussian).
    th_s = _np.deg2rad(slow_div_deg) / 1.18
    th_f = _np.deg2rad(fast_div_deg) / 1.18
    w_s, w_f = lam / (_np.pi * th_s), lam / (_np.pi * th_f)
    A_eff = float(_np.pi * w_s * w_f)
    out["A_eff_m2"] = InferredDynamic(
        A_eff, "m^2", "MEDIUM (geometric mode size HIGH; ~2x effective-area convention + Gaussian-mode "
        "assumption)", "A_eff = pi*w_s*w_f, w = lambda/(pi*theta_FWHM/1.18) from the 6deg/27deg far-field")

    # --- (2) GS differential gain dg/dN -- DIAGNOSTIC: it reveals GAIN CLAMPING -------------------------
    # Finite-difference the GS material gain vs a small injection step at the operating point; N is the
    # total confined+WL carrier density. At 2 A the GS is ~fully inverted (2 rho_GS - 1 ~ 0.998), so the
    # extra carriers go into the WL/ES RESERVOIR, not the clamped GS -> dg/dN ~ 0. This is NOT a usable
    # differential gain (it would blow the textbook P_sat = h nu A/(Gamma a tau) up); it is the SIGNATURE
    # that the QD gain is CLAMPED and the saturation is RESERVOIR-limited, which is why tau_eff below uses
    # the stimulated CROSS-SECTION form (sigma_pk) instead of dg/dN. CONFIDENCE: HIGH as a clamping
    # diagnostic (dg/dN ~ 0 at the operating point); NOT to be used as a differential-gain value.
    y1 = m.steady_state(drive, S_conf_m3=0.0)
    y2 = m.steady_state(drive * 1.02, S_conf_m3=0.0)
    g1 = float(m.material_gain_per_m(m.rho_GS(y1), nu0))
    g2 = float(m.material_gain_per_m(m.rho_GS(y2), nu0))
    N1, N2 = m.total_carrier_density(y1), m.total_carrier_density(y2)
    a_diff = (g2 - g1) / (N2 - N1) if N2 != N1 else float("nan")
    out["dg_dN_diagnostic_m2"] = InferredDynamic(
        float(a_diff), "m^2 (dg/dN, ~0 = CLAMPED)", "HIGH as a clamping diagnostic (GS inverted ~0.998 at "
        "2 A -> dg/dN ~ 0, reservoir-limited); NOT a usable differential gain",
        "finite-difference dg/dN at the operating point (clamped)")

    # --- (3) effective gain-recovery / saturation time tau_eff (cross-section form) --------------------
    # For a CLAMPED QD gain the saturation is set by the stimulated CROSS-SECTION, not dg/dN: a dot's
    # stimulated rate is sigma_pk * v_g * S_conf, the confined photon density is S_conf = Gamma P /
    # (A_eff v_g h nu), and saturation occurs when that rate balances the recovery 1/tau_eff. Setting
    # sigma_pk * Gamma * P_sat / (A_eff h nu) = 1/tau_eff gives
    #     tau_eff = A_eff h nu / (Gamma sigma_pk P_sat),
    # using the PHYSICAL A_eff (1), the FITTED sigma_pk, and the calibrated P_sat. For this device this is
    # ~100 ps -- the QD reservoir-refill gain-recovery time that governs pattern effects / XGM, NOT the slow
    # ns carrier lifetime and NOT the sub-ps SHB/carrier-heating (both are separate timescales absent from
    # a CW P_sat). CONFIDENCE: MEDIUM for the order of magnitude -- it inherits the A_eff ~2x convention and
    # the Gamma/sigma_pk degeneracy from the gain fit, so trust the ~10^2 ps SCALE, not the digits; pin it
    # with a pump-probe gain-recovery trace.
    Psat_W = 10.0 ** (device.report["Psat_out_dBm"] / 10.0) * 1.0e-3
    gam, sig = m.gamma_confinement, float(device.params.sigma_pk_m2)
    tau_eff = (H_PLANCK * nu0 * A_eff) / (gam * sig * Psat_W) if (sig and Psat_W) else float("nan")
    out["tau_eff_s"] = InferredDynamic(
        float(tau_eff), "s", "MEDIUM (order of magnitude ~100 ps; inherits A_eff ~2x + the Gamma/sigma_pk "
        "degeneracy; the fast reservoir-refill recovery, NOT the slow lifetime or the sub-ps SHB/CH)",
        "tau_eff = A_eff h nu / (Gamma sigma_pk P_sat), cross-section saturation form (QD gain is clamped)")

    # --- (4) small-signal modulation / gain-recovery 3 dB frequency (order of magnitude) ---------------
    # The CW-saturation time sets the low-frequency recovery: f_3dB ~ 1/(2 pi tau_eff). CONFIDENCE: LOW --
    # this is only the SLOW envelope; the true high-speed response is dominated by the ps reservoir dynamics
    # (unmeasured), so the real modulation bandwidth is HIGHER than this estimate.
    f3 = 1.0 / (2.0 * _np.pi * tau_eff) if tau_eff and _np.isfinite(tau_eff) else float("nan")
    out["f_3dB_slow_Hz"] = InferredDynamic(
        float(f3), "Hz", "LOW (slow-envelope only; the ps reservoir dynamics raise the true bandwidth)",
        "f ~ 1/(2 pi tau_eff)")

    # --- (5) linewidth enhancement factor alpha (NOT reliably inferable) --------------------------------
    # alpha = -dn'/dn'' is the carrier derivative of the KK-paired index/gain. Inferring it from the CW gain
    # would need the gain ASYMMETRY (the carrier-induced index slope), but the fitted GS gain is a SYMMETRIC
    # Gaussian comb -> its Kramers-Kronig index change is ANTISYMMETRIC and crosses ZERO at the 1310 nm peak
    # -> the KK estimate of alpha at the operating wavelength is ~0, a trivial (useless) lower bound. The
    # real alpha (1-3 for QD near the GS) comes from the asymmetric WL/ES background the symmetric fit omits.
    # CONFIDENCE: LOW / effectively NOT inferable from this calibration. alpha stays at the flagged default
    # (alpha_lef=2.0); pin it with an FWM up/down-asymmetry or AM/PM-chirp measurement.
    out["alpha_lef"] = InferredDynamic(
        float(device.params.alpha_lef), "-", "LOW / NOT inferable (KK of the symmetric fitted gain gives "
        "~0 at the peak; the real alpha needs the gain asymmetry / a measurement)",
        "KK of the CW gain is antisymmetric -> ~0 at peak; default 2.0 retained as a placeholder")

    # --- NOTE on the kinetic rate RATIOS (HIGH, already applied) -----------------------------------------
    # The forward/backward kinetic-rate RATIOS (tau_esc/tau_cap and tau_GS_ES/tau_ES_GS) are pinned by
    # DETAILED BALANCE given the GS/ES energy separation dE_ES_GS = 0.078 eV, which WAS calibrated from the
    # 1210/1310 nm two-band ASE split and is ALREADY APPLIED in the fitted params (.with_detailed_balance_
    # taus()). CONFIDENCE: HIGH for the ratios (exact detailed balance); the ABSOLUTE kinetic times remain
    # uncalibrated (only their ratios + the aggregate tau_eff above are constrained).
    out["dE_ES_GS_eV"] = InferredDynamic(
        float(device.params.dE_ES_GS_eV), "eV", "HIGH for the detailed-balance rate RATIOS it fixes (from "
        "the CW two-band ASE); absolute kinetic times still uncalibrated",
        "dE from the 1210/1310 nm ASE split -> detailed-balance forward/backward tau ratios")
    return out
