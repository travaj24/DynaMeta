"""Intensity-modulated / direct-detection (IM-DD) link metrics for an amplified signal: PAM-N
levels, decision thresholds, symbol- and bit-error rate, Q factor and the largest constellation a
given operating point supports (audit 2026-08-04 F-6).

WHY THIS IS HERE. detection.py already gives the three optical noise variances at ONE optical
power. A modulated link needs them PER LEVEL, decision thresholds between the levels, and an error
rate -- none of which existed anywhere in the package, although
docs/DynaMeta_QD_SOA_extension_spec.md asks for "a true symbol-stream SNDR/EVM" as future work.
There is no new physics in this module: every variance comes from optics.amp_noise, the single
Monte-Carlo-pinned home for the beat-noise algebra, and everything else is the standard Gaussian
receiver analysis.

THE ONE THING THAT IS EASY TO GET WRONG, and the reason this is a module rather than three lines at
a call site: the per-level variances must be EVALUATED per level, not scaled. Shot noise and
signal-spontaneous beat noise are linear in the level power; spontaneous-spontaneous beat noise and
the receiver's thermal noise are level-INDEPENDENT. So rescaling detection_noise's `var_total` by
the level power is wrong for two of the four terms. pam_levels_W -> level_statistics does it
correctly by calling the variance algebra once per level.

Because the variances differ between levels, the maximum-likelihood decision threshold is NOT the
midpoint: it solves (x-mu_k)^2/2 s_k^2 + ln s_k = (x-mu_j)^2/2 s_j^2 + ln s_j, a quadratic whose
root between the two means is the threshold. ml_thresholds does that exactly and degenerates to the
midpoint when the variances are equal.

SCOPE. Gaussian semi-analytic error rates -- exact for this noise model, and the standard tool for
error rates far below what Monte-Carlo can reach (a Q of 50 corresponds to a SER around 1e-500, so
sampling is not an option). NOT modelled: ISI, receiver filter shaping beyond a noise-equivalent
bandwidth, equalization, source RIN, clock recovery, or FEC coding gain. Those are link-design
concerns above this layer.

References: Agrawal, "Fiber-Optic Communication Systems" 4th ed. ch. 4-5 (IM/DD receiver, Q, PAM);
Desurvire, "Erbium-Doped Fiber Amplifiers" ch. 2 (beat-noise variances); ITU-T G.975.1 and
OIF-400ZR for the pre-FEC error-rate thresholds used by max_pam_order.
Pure numpy/scipy; SI units. ASCII only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK, KB, Q_E, T_REF

__all__ = ["PamFormat", "LevelStatistics", "LinkPerformance", "pam_levels_W", "gray_map",
           "ml_thresholds", "pam_ser", "ser_to_ber", "q_to_ser", "level_statistics",
           "link_performance", "max_pam_order", "FEC_THRESHOLDS"]

# Pre-FEC error-rate thresholds a link is usually designed to. 'hard' is the classic
# hard-decision-FEC operating point; 'ofec'/'sd' are soft-decision thresholds (OIF-400ZR uses
# ~2e-2). These are DESIGN CONVENTIONS, quoted so max_pam_order's default is not a bare magic
# number -- not physics.
FEC_THRESHOLDS: Dict[str, float] = {"none": 1e-12, "hard": 1e-3, "ofec": 2.0e-2, "sd": 2.0e-2}


def _qfunc(x):
    """Gaussian tail Q(x) = 0.5 erfc(x/sqrt(2)); stable far into the tail."""
    from scipy.special import erfc
    return 0.5 * erfc(np.asarray(x, dtype=np.float64) / math.sqrt(2.0))


def q_to_ser(q: float) -> float:
    """Binary (OOK) symbol error rate from a Q factor: SER = Q_func(q)."""
    return float(_qfunc(float(q)))


@dataclass(frozen=True)
class PamFormat:
    """A PAM-N intensity-modulation format.

    order              N, a power of two >= 2 (2 = OOK/NRZ) -> log2(N) bits per symbol
    baud_hz            symbol rate
    mean_power_W       MEAN optical power of the constellation (levels are set so this is exact)
    extinction_ratio   P_max / P_min, linear. A finite ER is the physical case; the lowest level of
                       a real transmitter is never zero.
    gray_coded         Gray mapping, so adjacent levels differ in ONE bit and a symbol error costs
                       one bit error -- the assumption behind ser_to_ber.
    """
    order: int = 4
    baud_hz: float = 10.0e9
    mean_power_W: float = 1.0e-3
    extinction_ratio: float = 12.0
    gray_coded: bool = True

    def __post_init__(self):
        if self.order < 2 or (self.order & (self.order - 1)) != 0:
            raise ValueError("PamFormat: order must be a power of two >= 2 (got %r)" % (self.order,))
        if not (self.extinction_ratio > 1.0):
            raise ValueError("PamFormat: extinction_ratio must be > 1 (got %r)"
                             % (self.extinction_ratio,))
        if not (self.baud_hz > 0.0 and self.mean_power_W > 0.0):
            raise ValueError("PamFormat: baud_hz and mean_power_W must be > 0")

    @property
    def bits_per_symbol(self) -> int:
        return int(round(math.log2(self.order)))

    @property
    def bitrate_bit_s(self) -> float:
        return float(self.baud_hz) * self.bits_per_symbol


def pam_levels_W(fmt: PamFormat) -> np.ndarray:
    """The N optical power levels [W], equally spaced in POWER (the IM/DD convention), with mean
    exactly fmt.mean_power_W and P_max/P_min = fmt.extinction_ratio:

        P_min = 2 P_mean / (1 + ER),   P_max = 2 P_mean ER / (1 + ER)

    so that (P_min + P_max)/2 = P_mean, and for equally spaced levels that IS the mean.
    """
    er = float(fmt.extinction_ratio)
    p_min = 2.0 * fmt.mean_power_W / (1.0 + er)
    p_max = 2.0 * fmt.mean_power_W * er / (1.0 + er)
    lv = np.linspace(p_min, p_max, int(fmt.order))
    return lv * (fmt.mean_power_W / float(np.mean(lv)))     # exact mean against round-off


def gray_map(order: int) -> np.ndarray:
    """g[k] = the Gray-coded bit word carried by level k, so adjacent LEVELS differ in one bit."""
    k = np.arange(int(order))
    return k ^ (k >> 1)


def ml_thresholds(mu, sigma) -> np.ndarray:
    """Maximum-likelihood decision thresholds between adjacent Gaussian levels of UNEQUAL variance.

    For equiprobable symbols the threshold between levels k and j solves

        (x - mu_k)^2 / (2 s_k^2) + ln s_k = (x - mu_j)^2 / (2 s_j^2) + ln s_j

    -- a quadratic; the root lying between the two means is taken. Degenerates exactly to the
    midpoint when s_k == s_j. Returns NaN for any pair with a non-finite or non-positive sigma
    rather than raising, so one bad operating point in a sweep cannot abort the sweep.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sg = np.asarray(sigma, dtype=np.float64)
    if mu.shape != sg.shape or mu.ndim != 1 or mu.size < 2:
        raise ValueError("ml_thresholds: mu and sigma must be matching 1-D arrays of length >= 2")
    out = np.empty(mu.size - 1)
    for k in range(mu.size - 1):
        m1, m2, s1, s2 = mu[k], mu[k + 1], sg[k], sg[k + 1]
        if not (np.isfinite(m1) and np.isfinite(m2) and np.isfinite(s1) and np.isfinite(s2)) \
                or s1 <= 0.0 or s2 <= 0.0:
            out[k] = np.nan
            continue
        if abs(s1 - s2) <= 1e-15 * max(s1, s2):
            out[k] = 0.5 * (m1 + m2)
            continue
        a = 1.0 / (2.0 * s1 ** 2) - 1.0 / (2.0 * s2 ** 2)
        b = -m1 / s1 ** 2 + m2 / s2 ** 2
        c = m1 ** 2 / (2.0 * s1 ** 2) - m2 ** 2 / (2.0 * s2 ** 2) + math.log(s1 / s2)
        roots = np.roots([a, b, c])
        roots = roots[np.isreal(roots)].real
        inside = roots[(roots > min(m1, m2)) & (roots < max(m1, m2))]
        out[k] = float(inside[0]) if inside.size else 0.5 * (m1 + m2)
    return out


def pam_ser(mu, sigma, thresholds=None) -> float:
    """Symbol error rate of an equiprobable PAM constellation with per-level Gaussian noise.

    Exact for the Gaussian model: an interior level can err across either of its two thresholds, an
    outer level across one. With EQUAL variances and equal spacing d this reduces analytically to
    2 (N-1)/N * Q(d / 2 sigma), which is gated.

    Pass `thresholds` to score a receiver whose thresholds are NOT matched to the current levels --
    e.g. frozen at a different operating point, which is how a burst-mode gain excursion is
    measured. Returns NaN on non-finite input.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sg = np.asarray(sigma, dtype=np.float64)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sg)) and np.all(sg > 0.0)):
        return float("nan")
    th = ml_thresholds(mu, sg) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    if th.size != mu.size - 1 or not np.all(np.isfinite(th)):
        return float("nan")
    n = mu.size
    tot = 0.0
    for k in range(n):
        p = 0.0
        if k > 0:
            p += float(_qfunc((mu[k] - th[k - 1]) / sg[k]))      # falls below its lower threshold
        if k < n - 1:
            p += float(_qfunc((th[k] - mu[k]) / sg[k]))          # rises above its upper threshold
        tot += min(p, 1.0)
    return float(tot / n)


def ser_to_ber(ser: float, order: int, gray_coded: bool = True) -> float:
    """Bit error rate from a symbol error rate. With Gray coding the overwhelmingly likely symbol
    error is to an ADJACENT level, which differs in exactly one bit, so BER = SER / log2(N). Without
    Gray coding a symbol error scrambles bits at random and BER -> SER/2 * N/(N-1) is the usual
    estimate; the Gray form is the one that matches practice."""
    bits = int(round(math.log2(int(order))))
    if gray_coded:
        return float(ser) / bits
    return float(ser) * 0.5 * int(order) / (int(order) - 1)


@dataclass
class LevelStatistics:
    """Per-level received statistics of a PAM constellation."""
    levels_W: np.ndarray            # optical power per level at the detector [W]
    current_A: np.ndarray           # mean photocurrent per level [A]
    sigma_A: np.ndarray             # total current-noise sigma per level [A]
    terms_A2: Dict[str, np.ndarray] = field(default_factory=dict)   # per-level variance breakdown
    thresholds_A: Optional[np.ndarray] = None

    @property
    def worst_q(self) -> float:
        """Smallest adjacent-level Q = (I_{k+1} - I_k) / (sigma_{k+1} + sigma_k)."""
        mu, sg = self.current_A, self.sigma_A
        return float(np.min((mu[1:] - mu[:-1]) / (sg[1:] + sg[:-1])))


def level_statistics(fmt: PamFormat, *, rho_sp_W_Hz: float, responsivity_A_W: float,
                     optical_bw_Hz: float, electrical_bw_Hz: Optional[float] = None,
                     m_pol: int = 2, loss_dB: float = 0.0,
                     dark_current_A: float = 0.0,
                     tia_current_noise_A_rtHz: float = 0.0,
                     load_ohm: Optional[float] = None,
                     temperature_K: float = T_REF) -> LevelStatistics:
    """Per-level photocurrent and noise for a PAM constellation at a direct-detection receiver.

    rho_sp_W_Hz    the SINGLE-POLARIZATION ASE power spectral density at the signal wavelength,
                   i.e. noise.analyze_noise(...).fwd_ase.psd_1pol interpolated to the carrier (or
                   the same quantity from chain.ChainResult.rho_out_1pol_W_Hz). NOT the total ASE
                   power in a band.
    loss_dB        passive loss between the amplifier output and this detector (span, filter, tap,
                   a 1:N splitter). Applied to the signal AND to rho_sp, since passive loss cannot
                   separate them -- so OSNR is loss-invariant and only the electrical terms move.
    electrical_bw_Hz   noise-equivalent receiver bandwidth; None -> 0.75 * baud, the usual matched
                   choice for PAM.

    The three optical variances come from optics.amp_noise.beat_noise_variances, called ONCE PER
    LEVEL (see the module docstring on why they cannot be scaled), plus the receiver's own thermal
    term which is level-independent.
    """
    from dynameta.optics.amp_noise import beat_noise_variances
    # A NEGATIVE ASE spectral density is not a degenerate case, it is invalid input -- and it would
    # otherwise propagate as a silent NaN through sigma, the thresholds and the SER. NaN input is
    # allowed through (a caller sweeping operating points may legitimately hand one over from a
    # failed solve, and the downstream SER already reports NaN honestly).
    if float(rho_sp_W_Hz) < 0.0:
        raise ValueError("level_statistics: rho_sp_W_Hz must be >= 0 (got %r); it is a power "
                         "spectral density" % (rho_sp_W_Hz,))
    b_e = float(electrical_bw_Hz) if electrical_bw_Hz else 0.75 * float(fmt.baud_hz)
    atten = 10.0 ** (-float(loss_dB) / 10.0)
    lv = pam_levels_W(fmt) * atten
    rho = float(rho_sp_W_Hz) * atten
    R = float(responsivity_A_W)

    n = lv.size
    terms = {k: np.empty(n) for k in ("shot", "sig_spont", "spont_spont", "thermal")}
    var_th = float(tia_current_noise_A_rtHz) ** 2 * b_e
    if load_ohm:
        var_th += 4.0 * KB * float(temperature_K) * b_e / float(load_ohm)
    for k, p in enumerate(lv):
        v = beat_noise_variances(float(p), rho, responsivity_A_W=R, electrical_bw_Hz=b_e,
                                 optical_bw_Hz=float(optical_bw_Hz), m_pol=int(m_pol),
                                 I_dark_A=float(dark_current_A))
        terms["shot"][k] = v["shot"]
        terms["sig_spont"][k] = v["sig_spont"]
        terms["spont_spont"][k] = v["spont_spont"]
        terms["thermal"][k] = var_th
    sigma = np.sqrt(sum(terms.values()))
    mu = R * lv
    return LevelStatistics(levels_W=lv, current_A=mu, sigma_A=sigma, terms_A2=terms,
                           thresholds_A=ml_thresholds(mu, sigma))


@dataclass
class LinkPerformance:
    """IM-DD link performance at one operating point."""
    fmt: PamFormat
    levels: LevelStatistics
    ser: float
    ber: float
    q_factor: float
    osnr_dB: float                  # in ref_bw_nm, measured BEFORE the passive loss (loss-invariant)
    snr_elec_dB: float              # on the full PAM swing
    enob: float                     # (SNR_dB - 1.76) / 6.02
    p_rx_mean_dBm: float
    dominant_noise: str
    meta: Dict[str, float] = field(default_factory=dict)


def link_performance(fmt: PamFormat, *, rho_sp_W_Hz: float, signal_lambda_m: float,
                     quantum_efficiency: float = 1.0, responsivity_A_W: Optional[float] = None,
                     optical_bw_Hz: float, electrical_bw_Hz: Optional[float] = None,
                     m_pol: int = 2, loss_dB: float = 0.0, ref_bw_nm: float = 0.1,
                     **rx) -> LinkPerformance:
    """End-to-end IM-DD metrics for a PAM-N signal carrying rho_sp of ASE.

    Extra keyword arguments (dark_current_A, tia_current_noise_A_rtHz, load_ohm, temperature_K) are
    forwarded to level_statistics. `responsivity_A_W` overrides the quantum-efficiency form
    R = eta q lambda / (h c).
    """
    nu_s = C_LIGHT / float(signal_lambda_m)
    R = (float(responsivity_A_W) if responsivity_A_W is not None
         else float(quantum_efficiency) * Q_E / (H_PLANCK * nu_s))
    st = level_statistics(fmt, rho_sp_W_Hz=rho_sp_W_Hz, responsivity_A_W=R,
                          optical_bw_Hz=optical_bw_Hz, electrical_bw_Hz=electrical_bw_Hz,
                          m_pol=m_pol, loss_dB=loss_dB, **rx)
    ser = pam_ser(st.current_A, st.sigma_A, st.thresholds_A)
    ber = ser_to_ber(ser, fmt.order, fmt.gray_coded) if np.isfinite(ser) else float("nan")

    # OSNR is measured at the amplifier output in ref_bw_nm; passive loss attenuates signal and ASE
    # identically, so it is loss-invariant and must be computed BEFORE the loss.
    dnu_ref = float(ref_bw_nm) * 1e-9 * C_LIGHT / float(signal_lambda_m) ** 2
    p_ase_ref = int(m_pol) * float(rho_sp_W_Hz) * dnu_ref
    osnr = (10.0 * math.log10(fmt.mean_power_W / p_ase_ref) if p_ase_ref > 0.0 else float("inf"))

    swing = float(st.current_A[-1] - st.current_A[0])
    sigma_rms = float(np.sqrt(np.mean(st.sigma_A ** 2)))
    snr_dB = 20.0 * math.log10(swing / sigma_rms) if sigma_rms > 0.0 else float("inf")
    p_rx = float(np.mean(st.levels_W))
    dom = max(st.terms_A2, key=lambda k: float(st.terms_A2[k][-1]))
    return LinkPerformance(
        fmt=fmt, levels=st, ser=ser, ber=ber, q_factor=st.worst_q, osnr_dB=osnr,
        snr_elec_dB=snr_dB, enob=(snr_dB - 1.76) / 6.02,
        p_rx_mean_dBm=10.0 * math.log10(max(p_rx, 1e-300) * 1e3),
        dominant_noise=dom.replace("_", "-"),
        meta={"responsivity_A_W": R, "electrical_bw_Hz": (float(electrical_bw_Hz)
                                                          if electrical_bw_Hz
                                                          else 0.75 * fmt.baud_hz),
              "loss_dB": float(loss_dB), "rho_sp_W_Hz": float(rho_sp_W_Hz),
              "ref_bw_nm": float(ref_bw_nm)})


def max_pam_order(fmt: PamFormat, *, target_ser: float = FEC_THRESHOLDS["ofec"],
                  orders: Sequence[int] = (2, 4, 8, 16, 32, 64, 128), **kw) -> int:
    """The largest PAM order in `orders` that meets target_ser at this operating point, holding the
    MEAN power fixed (so the amplifier's operating point is identical across orders and only the
    receiver's job gets harder). Returns 0 if even the smallest order fails.

    Extra keywords are forwarded to link_performance.
    """
    from dataclasses import replace
    best = 0
    for o in orders:
        r = link_performance(replace(fmt, order=int(o)), **kw)
        if np.isfinite(r.ser) and r.ser <= float(target_ser):
            best = int(o)
    return best
