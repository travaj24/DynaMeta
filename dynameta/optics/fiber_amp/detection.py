"""Photodetected noise of the amplified output (docs sec.11): the beat-noise spectra a receiver
sees when the amplifier's signal + ASE fall on a square-law photodiode, and the resulting
electrical SNR and beat-noise noise figure. Builds on the ASE power spectral density from noise.py.

When the field (signal at power P_s, ASE with per-polarization PSD rho_sp) is detected, the
photocurrent i = R|E|^2 carries (Agrawal; Desurvire):
  * shot noise           sigma_shot^2  = 2 e (I_sig + I_ase) B_e
  * signal-spontaneous   sigma_sigsp^2 = 4 R^2 P_s rho_sp B_e      (signal beats co-pol ASE)
  * spontaneous-spont.   sigma_spsp^2  = 2 R^2 rho_sp^2 m (2 B_o - B_e) B_e   (ASE beats itself)
with R the responsivity, B_o / B_e the optical / electrical bandwidths, m the ASE polarizations.
The electrical SNR is I_sig^2 / sigma_total^2, and the beat-noise NOISE FIGURE
NF = SNR_in(shot-limited) / SNR_out reduces, in the high-gain signal-spontaneous-dominated limit,
to the optical noise figure (2 n_sp (G-1) + 1)/G -- the cross-check that ties this module to
noise.py. Pure numpy; SI units. docs/fiber_amp_model_spec.md sec.11.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK, Q_E
from dynameta.optics.fiber_amp.noise import analyze_noise
from dynameta.optics.fiber_amp.steady_state import SteadyStateResult

__all__ = ["BeatNoiseResult", "detection_noise"]

_DB = lambda x: 10.0 * np.log10(np.maximum(x, 1e-300))    # noqa: E731


@dataclass
class BeatNoiseResult:
    signal_lambda_m: float
    responsivity_A_W: float
    optical_bw_Hz: float
    electrical_bw_Hz: float
    gain_dB: float
    i_signal_A: float               # mean signal photocurrent
    i_ase_A: float                  # mean ASE photocurrent
    var_shot: float                 # A^2
    var_sig_sp: float               # A^2 (signal-spontaneous beat)
    var_sp_sp: float                # A^2 (spontaneous-spontaneous beat)
    var_total: float
    snr_elec_dB: float
    nf_beat_dB: float               # beat-noise-derived optical NF
    added_rin_per_Hz: float         # excess intensity noise the amplifier adds (ASE beats)
    meta: dict = field(default_factory=dict)

    @property
    def dominant_term(self) -> str:
        pairs = {"shot": self.var_shot, "sig-sp": self.var_sig_sp, "sp-sp": self.var_sp_sp}
        return max(pairs, key=pairs.get)


def detection_noise(result: SteadyStateResult, signal_lambda_m: float, *,
                    optical_bw_Hz: float, electrical_bw_Hz: float,
                    quantum_efficiency: float = 1.0, responsivity_A_W: float = None,
                    m_modes: int = None) -> BeatNoiseResult:
    """Beat-noise analysis of the amplified signal at a photodetector. optical_bw_Hz is the
    filter bandwidth in front of the diode; electrical_bw_Hz the receiver bandwidth. Detector is
    R = responsivity_A_W, or eta e/(h nu) from quantum_efficiency if responsivity is not given.
    Returns the shot / signal-spontaneous / spontaneous-spontaneous variances, the electrical
    SNR, the beat-noise NF (-> optical NF in the signal-spont-dominated limit), and the excess
    RIN the amplifier adds."""
    from dynameta.optics.fiber_amp.noise import _meta_m_modes
    m_modes = _meta_m_modes(result, m_modes)     # default: the value the solve used (audit S3-31)
    nr = analyze_noise(result, signal_lambda_m, m_modes=m_modes)
    G = nr.gain_lin
    nu_s = C_LIGHT / signal_lambda_m
    R = responsivity_A_W if responsivity_A_W is not None else quantum_efficiency * Q_E / (H_PLANCK
                                                                                          * nu_s)
    P_sig_out = float(nr.meta["P_signal_out_W"])
    si = [i for i, k in enumerate(result.kind) if k == "signal"]
    i0 = min(si, key=lambda k: abs(result.lambda_m[k] - signal_lambda_m)) if si else 0
    P_sig_in = float(result.power_W[i0, 0])
    # per-pol forward-ASE PSD at the signal, reusing the spectrum analyze_noise already extracted
    fwd = nr.fwd_ase
    if fwd.lambda_m.size == 0:
        rho_sp = 0.0
    elif fwd.lambda_m.size == 1:
        rho_sp = float(fwd.psd_1pol[0])
    else:
        rho_sp = float(np.interp(signal_lambda_m, fwd.lambda_m, fwd.psd_1pol))

    B_o, B_e = float(optical_bw_Hz), float(electrical_bw_Hz)
    # the beat-noise algebra lives ONCE in optics.amp_noise (post-audit unification of the
    # S3-2/C4-3 duplicate pair); this module supplies the fiber-side inputs and packaging
    from dynameta.optics.amp_noise import beat_noise_variances
    v = beat_noise_variances(P_sig_out, rho_sp, responsivity_A_W=R, electrical_bw_Hz=B_e,
                             optical_bw_Hz=B_o, m_pol=m_modes)
    I_sig, I_ase = v["I_sig"], v["I_ase"]
    var_shot, var_sig_sp, var_sp_sp = v["shot"], v["sig_spont"], v["spont_spont"]
    var_total = v["total"]

    snr_out = I_sig ** 2 / var_total if var_total > 0.0 else np.inf
    # NF = SNR_in/SNR_out with SNR_in at an IDEAL (eta=1) shot-noise-limited input detector: the
    # amplifier NF is a property of the amplifier and must be eta-independent (audit S3-10: the
    # old eta factor reported sub-quantum-limit NF for real detectors).
    snr_in = P_sig_in / (2.0 * H_PLANCK * nu_s * B_e)
    nf_beat = snr_in / snr_out if snr_out > 0.0 else np.inf
    added_rin = (var_sig_sp + var_sp_sp) / (I_sig ** 2 * B_e) if I_sig > 0.0 else np.inf

    return BeatNoiseResult(
        float(signal_lambda_m), float(R), B_o, B_e, float(nr.gain_dB),
        float(I_sig), float(I_ase), float(var_shot), float(var_sig_sp), float(var_sp_sp),
        float(var_total), float(_DB(snr_out)), float(_DB(nf_beat)), float(added_rin),
        meta={"rho_sp_W_per_Hz": float(rho_sp), "gain_lin": float(G),
              "P_signal_out_W": P_sig_out, "P_signal_in_W": P_sig_in})
