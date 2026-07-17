"""Amplified spontaneous emission (ASE) spectrum and optical noise figure of the fiber
amplifier, read off a solved steady state (docs sec.4). The steady_state solve already
propagates m=2-polarization ASE bins with the spontaneous source m h nu d-nu; this module turns
those output bins into the measurable quantities:

  * the forward / backward output ASE spectrum P_ASE(lambda) and its per-polarization power
    spectral density rho(lambda) = P_bin / (m d-nu);
  * the spontaneous-emission (population-inversion) factor n_sp(lambda) = rho / (h nu (G-1)),
    which is bounded BELOW by 1 for any phase-insensitive amplifier (the quantum limit) and
    reaches 1 only at full inversion;
  * the optical noise figure NF(lambda_s) = 2 n_sp (G-1)/G + 1/G = 2 rho(lambda_s)/(h nu_s G)
    + 1/G, whose high-gain / full-inversion limit is the 3 dB quantum floor;
  * OSNR in a reference bandwidth (default 0.1 nm).

The n_sp / NF forms follow Desurvire (EDFA book) and the standard amplifier-noise result
NF = (2 n_sp (G-1) + 1)/G. Pure numpy; SI units. docs/fiber_amp_model_spec.md sec.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp.steady_state import SteadyStateResult

__all__ = ["AseSpectrum", "NoiseResult", "output_ase_spectrum", "noise_figure",
           "local_inversion_factor", "analyze_noise"]

_DB = lambda x: 10.0 * np.log10(np.maximum(x, 1e-300))    # noqa: E731


def _meta_m_modes(result: SteadyStateResult, m_modes) -> int:
    """Resolve the ASE polarization-mode count: an explicit argument wins, else the value the
    SOLVE actually used (result.meta['m_modes'], audit S3-31 -- a non-default AseBand.m_modes
    previously corrupted PSD/n_sp/NF/OSNR because the noise layer independently assumed 2)."""
    if m_modes is not None:
        return int(m_modes)
    return int(result.meta.get("m_modes", 2))


def local_inversion_factor(result: SteadyStateResult, lambda_m: float) -> np.ndarray:
    """Medium spontaneous-emission factor along z at wavelength lambda_m,
        n_sp(z) = sigma_e nbar2 / (sigma_e nbar2 - sigma_a (1 - nbar2) - sigma_esa nbar2),
    using the cross-sections the solve cached (result.meta). The denominator is the NET
    stimulated coefficient the solver actually used, including opt-in excited-state absorption
    (audit S3-33: omitting the ESA term made the reported n_sp inconsistent with the gain under
    ESA; with sigma_esa = 0 this is the classic two-level form, >= 1 by construction). This is
    the honest medium n_sp, as opposed to the ASE-PSD-derived effective factor which carries
    discretization noise. NaN where the cross-sections are unavailable."""
    sa = result.meta.get("sigma_a")
    se = result.meta.get("sigma_e")
    if sa is None or se is None:
        return np.full(result.z_m.shape, np.nan)
    i = min(range(result.lambda_m.size), key=lambda k: abs(result.lambda_m[k] - lambda_m))
    sig_a, sig_e = float(sa[i]), float(se[i])
    esa = result.meta.get("sigma_esa")
    sig_esa = float(esa[i]) if esa is not None else 0.0
    n2 = result.nbar2_z
    num = sig_e * n2
    den = sig_e * n2 - sig_a * (1.0 - n2) - sig_esa * n2
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


@dataclass
class AseSpectrum:
    """Output ASE spectrum in one propagation direction.
      lambda_m   (Nb,) bin centre wavelengths [m]
      power_W    (Nb,) ASE power per bin (both polarizations) at the output end [W]
      psd_1pol   (Nb,) per-polarization power spectral density rho = P_bin/(m d-nu) [W/Hz]
      n_sp       (Nb,) inversion factor rho / (h nu (G(lambda)-1)) if gains supplied, else NaN
      direction  'fwd' (measured at z=L) or 'bwd' (measured at z=0)."""
    lambda_m: np.ndarray
    power_W: np.ndarray
    psd_1pol: np.ndarray
    n_sp: np.ndarray
    direction: str

    @property
    def total_power_W(self) -> float:
        return float(np.sum(self.power_W))


@dataclass
class NoiseResult:
    signal_lambda_m: float
    gain_dB: float
    gain_lin: float
    n_sp: float                      # effective factor from the output ASE PSD (the measurement)
    n_sp_local_min: float            # min local two-level factor along z (>= 1 by construction)
    n_sp_local_in: float             # local factor at the signal-input end
    nf_dB: float                     # optical noise figure
    nf_lin: float
    osnr_dB: float                   # signal / forward-ASE in ref_bw
    ref_bw_nm: float
    fwd_ase: AseSpectrum
    bwd_ase: AseSpectrum
    meta: dict = field(default_factory=dict)


def _signal_gain(result: SteadyStateResult, lambda_m: float):
    """(G_linear, index) for the signal channel nearest lambda_m; falls back to any channel."""
    sig = [i for i, k in enumerate(result.kind) if k == "signal"]
    pool = sig if sig else list(range(result.lambda_m.size))
    i = min(pool, key=lambda k: abs(result.lambda_m[k] - lambda_m))
    G = float(result.power_W[i, -1] / result.power_W[i, 0])
    return G, i


def _bin_dnu_hz(result: SteadyStateResult, idx: np.ndarray) -> np.ndarray:
    """Frequency bin width d-nu [Hz] for the ASE channels at indices idx, recovered from the
    ChannelSet the solve carried (stored on result.meta if present) or from neighbour spacing."""
    dnu = result.meta.get("dnu_hz")
    if dnu is not None:
        return np.asarray(dnu)[idx]
    nu = C_LIGHT / result.lambda_m[idx]
    order = np.argsort(nu)
    dv = np.gradient(nu[order])
    out = np.empty_like(nu)
    out[order] = np.abs(dv)
    return out


def output_ase_spectrum(result: SteadyStateResult, direction: str = "fwd", *,
                        m_modes: Optional[int] = None,
                        signal_lambda_m: Optional[float] = None) -> AseSpectrum:
    """Extract the output ASE spectrum for one direction. Forward ASE is read at z=L, backward
    at z=0. m_modes defaults to the value the SOLVE used (result.meta). n_sp per bin uses the
    local signal gain G(lambda) interpolated from the amplifier's signal channel when available
    (else the gain at signal_lambda_m); n_sp is left NaN if no gain reference exists."""
    m_modes = _meta_m_modes(result, m_modes)
    u = result.u
    mask = result.is_ase & (u > 0 if direction == "fwd" else u < 0)
    idx = np.where(mask)[0]
    if idx.size == 0:
        z = np.zeros(0)
        return AseSpectrum(z, z, z, z, direction)
    lam = result.lambda_m[idx]
    order = np.argsort(lam)
    idx = idx[order]
    lam = result.lambda_m[idx]
    end = -1 if direction == "fwd" else 0
    P = result.power_W[idx, end]
    nu = C_LIGHT / lam
    dnu = _bin_dnu_hz(result, idx)
    psd = P / (m_modes * np.maximum(dnu, 1e-300))          # per-polarization PSD [W/Hz]

    # gain reference at each ASE wavelength (nearest signal channel), for n_sp
    sig = [i for i, k in enumerate(result.kind) if k == "signal"]
    if sig:
        sl = result.lambda_m[sig]
        sG = np.array([result.power_W[i, -1] / result.power_W[i, 0] for i in sig])
        oo = np.argsort(sl)
        G_at = np.interp(lam, sl[oo], sG[oo]) if len(sig) > 1 else np.full(lam.shape, sG[0])
    elif signal_lambda_m is not None:
        G_at = np.full(lam.shape, _signal_gain(result, signal_lambda_m)[0])
    else:
        G_at = np.full(lam.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_sp = psd / (H_PLANCK * nu * (G_at - 1.0))
    return AseSpectrum(lam, P, psd, n_sp, direction)


def noise_figure(result: SteadyStateResult, signal_lambda_m: float, *,
                 m_modes: Optional[int] = None, _fwd: Optional[AseSpectrum] = None):
    """Optical noise figure at the signal wavelength (docs sec.4):
        NF = 2 rho_1pol(nu_s) / (h nu_s G) + 1/G,
    with G the signal gain and rho_1pol the per-polarization forward-ASE PSD interpolated to the
    signal frequency. m_modes defaults to the solve's own value (result.meta). Returns
    (NF_linear, G_linear, n_sp_effective). Equivalent to (2 n_sp (G-1) + 1)/G with
    n_sp = rho_1pol / (h nu_s (G-1)). _fwd lets a caller that already extracted the forward
    spectrum pass it in (audit S6-14: analyze_noise used to extract it twice)."""
    G, _ = _signal_gain(result, signal_lambda_m)
    fwd = _fwd if _fwd is not None else output_ase_spectrum(
        result, "fwd", m_modes=m_modes, signal_lambda_m=signal_lambda_m)
    nu_s = C_LIGHT / signal_lambda_m
    if fwd.lambda_m.size == 0:
        rho_s = 0.0
    elif fwd.lambda_m.size == 1:
        rho_s = float(fwd.psd_1pol[0])
    else:
        rho_s = float(np.interp(signal_lambda_m, fwd.lambda_m, fwd.psd_1pol))
    nf = 2.0 * rho_s / (H_PLANCK * nu_s * G) + 1.0 / G
    n_sp_eff = rho_s / (H_PLANCK * nu_s * (G - 1.0)) if G > 1.0 else np.nan
    return nf, G, n_sp_eff


def analyze_noise(result: SteadyStateResult, signal_lambda_m: float, *, ref_bw_nm: float = 0.1,
                  m_modes: Optional[int] = None) -> NoiseResult:
    """Full noise analysis at signal_lambda_m: gain, NF, effective n_sp, forward/backward ASE
    spectra, and OSNR in a ref_bw_nm optical bandwidth (default 0.1 nm, the standard OSNR grid).
    m_modes defaults to the value the solve used (result.meta). OSNR = signal output power /
    forward-ASE power within ref_bw around the signal."""
    m_modes = _meta_m_modes(result, m_modes)
    fwd = output_ase_spectrum(result, "fwd", m_modes=m_modes, signal_lambda_m=signal_lambda_m)
    bwd = output_ase_spectrum(result, "bwd", m_modes=m_modes, signal_lambda_m=signal_lambda_m)
    nf, G, n_sp = noise_figure(result, signal_lambda_m, m_modes=m_modes, _fwd=fwd)

    # signal output power
    _, si = _signal_gain(result, signal_lambda_m)
    P_sig_out = float(result.power_W[si, -1])

    # forward-ASE power in the reference bandwidth around the signal: integrate PSD*m over d-nu
    nu_s = C_LIGHT / signal_lambda_m
    dnu_ref = C_LIGHT / signal_lambda_m ** 2 * (ref_bw_nm * 1e-9)      # |d-nu| for d-lambda
    if fwd.lambda_m.size:
        rho_s = float(np.interp(signal_lambda_m, fwd.lambda_m, fwd.psd_1pol)) \
            if fwd.lambda_m.size > 1 else float(fwd.psd_1pol[0])
    else:
        rho_s = 0.0
    P_ase_ref = m_modes * rho_s * dnu_ref
    osnr = P_sig_out / P_ase_ref if P_ase_ref > 0.0 else np.inf

    nsp_loc = local_inversion_factor(result, signal_lambda_m)
    nsp_loc_pos = nsp_loc[np.isfinite(nsp_loc) & (nsp_loc > 0.0)]
    nsp_min = float(np.min(nsp_loc_pos)) if nsp_loc_pos.size else np.nan
    nsp_in = float(nsp_loc[0]) if nsp_loc.size else np.nan

    return NoiseResult(
        signal_lambda_m=float(signal_lambda_m), gain_dB=float(_DB(G)), gain_lin=float(G),
        n_sp=float(n_sp), n_sp_local_min=nsp_min, n_sp_local_in=nsp_in,
        nf_dB=float(_DB(nf)), nf_lin=float(nf),
        osnr_dB=float(_DB(osnr)), ref_bw_nm=float(ref_bw_nm), fwd_ase=fwd, bwd_ase=bwd,
        meta={"nu_s_Hz": float(nu_s), "P_signal_out_W": P_sig_out,
              "P_ase_ref_W": float(P_ase_ref)})
