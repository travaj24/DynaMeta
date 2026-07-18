"""Multi-stage amplifier chains: compose fiber-amplifier stages with passive inter-stage
elements (isolators, filters, gain-flattening filters, splices) and track the signal power,
in-band ASE PSD, net gain, and cascaded noise figure through the chain.

Noise bookkeeping is PSD-BASED, which makes the classic cascade rules emerge rather than being
assumed: each amplifier stage multiplies the incoming per-polarization in-band ASE PSD by its
own gain and adds its own generated PSD (rho_out = rho_in G + rho_gen); a passive element
multiplies signal and in-band PSD by its transmission. The chain NF then follows from the
end-to-end PSD, NF = 2 rho_end/(h nu G_tot) + 1/G_tot (optics.amp_noise.nf_from_psd), and
reproduces Friis NF_tot = NF1 + (NF2-1)/G1 + ... and the attenuator rules (pre-amp loss adds
dB-for-dB; post-amp loss is nearly free) -- both gated in tests.

Physical scope notes:
  * In-band ASE cannot be filtered away from the signal, so an optical filter's in-band PSD
    follows the SIGNAL transmission; the ase_transmission knob applies to the OUT-of-band total
    ASE power ledger only (saturation bookkeeping for downstream stages).
  * Inter-stage ASE power is NOT injected into the next stage's saturation solve (each stage is
    solved with the signal alone); the per-stage records carry the total ASE power so the user
    can check it stays small vs the stage saturation power. Refs: Desurvire ch. 2; Becker,
    Olsson & Simpson, "EDFAs: Fundamentals and Technology" ch. 8.

Pure numpy; SI units."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.optics.amp_noise import nf_from_psd

__all__ = ["PassiveElement", "StageRecord", "ChainResult", "AmplifierChain"]

_DB = lambda x: 10.0 * np.log10(np.maximum(x, 1e-300))    # noqa: E731


@dataclass(frozen=True)
class PassiveElement:
    """A passive inter-stage element. loss_dB = insertion loss at the signal wavelength
    (isolator ~0.3-0.6 dB, filter/GFF 0.5-3 dB, splice ~0.05 dB); ase_transmission = linear
    transmission applied to the OUT-of-band total-ASE ledger (None -> same as the signal;
    0.0 -> an ideal out-of-band ASE stripper). In-band PSD always follows the signal loss."""
    name: str
    loss_dB: float
    ase_transmission: Optional[float] = None

    def __post_init__(self):
        if self.loss_dB < 0.0:
            raise ValueError("PassiveElement.loss_dB must be >= 0 (it is a LOSS)")


@dataclass
class StageRecord:
    name: str
    kind: str                       # 'amp' | 'passive'
    P_in_W: float
    P_out_W: float
    gain_dB: float                  # this element's signal gain (negative for passives)
    nf_stage_dB: float              # amplifier's own NF at its input (nan for passives)
    rho_out_1pol_W_Hz: float        # in-band per-pol ASE PSD after this element
    p_ase_total_W: float            # out-of-band total-ASE ledger after this element
    meta: dict = field(default_factory=dict)


@dataclass
class ChainResult:
    signal_lambda_m: float
    stages: List[StageRecord]
    P_out_W: float
    gain_total_dB: float
    nf_total_dB: float              # end-to-end, PSD-based (input-referred)
    rho_out_1pol_W_Hz: float
    osnr_dB: float                  # in ref_bw_nm
    ref_bw_nm: float


class AmplifierChain:
    """An ordered chain of amplifier stages (FiberAmplifier / ErYbAmplifier -- anything with
    .signals and .solve() returning a SteadyStateResult) and PassiveElements. solve() walks the
    chain: each amplifier is re-solved with ITS actual input signal power (metrics-style clone
    via the element's own signal list with the power swapped), so inter-stage losses and
    saturation interact correctly."""

    def __init__(self, elements: List, *, signal_index: int = 0):
        if not elements:
            raise ValueError("AmplifierChain: empty chain")
        self.elements = list(elements)
        self.signal_index = int(signal_index)

    def solve(self, P_in_W: float, signal_lambda_m: float, *, ref_bw_nm: float = 0.1,
              **solve_kw) -> ChainResult:
        from dynameta.optics.fiber_amp.metrics import _set_signal
        from dynameta.optics.fiber_amp.noise import analyze_noise

        nu_s = C_LIGHT / signal_lambda_m
        P = float(P_in_W)
        P_in0 = P
        rho = 0.0                                   # in-band per-pol PSD [W/Hz]
        p_ase = 0.0                                 # out-of-band total ASE ledger [W]
        records: List[StageRecord] = []
        for el in self.elements:
            if isinstance(el, PassiveElement):
                t = 10.0 ** (-el.loss_dB / 10.0)
                t_ase = t if el.ase_transmission is None else float(el.ase_transmission)
                P_new = P * t
                rho = rho * t                       # in-band follows the signal
                p_ase = p_ase * t_ase
                records.append(StageRecord(el.name, "passive", P, P_new, _DB(t), float("nan"),
                                           rho, p_ase))
                P = P_new
            else:
                amp = _set_signal(el, P, self.signal_index) if hasattr(el, "signals") else el
                res = amp.solve(**solve_kw)
                nr = analyze_noise(res, signal_lambda_m)
                G = float(nr.gain_lin)
                P_new = P * G
                # per-pol PSD the stage generated at the signal wavelength (analyze_noise
                # reports the m*rho*dnu_ref reference-band power; invert its own definition)
                rho_gen = (float(nr.meta["P_ase_ref_W"])
                           / (_ref_dnu(signal_lambda_m, ref_bw_nm) * _meta_m(res)))
                p_ase_gen = _total_fwd_ase_W(res)
                rho = rho * G + rho_gen
                p_ase = p_ase * G + p_ase_gen
                name = getattr(el, "name", None) or type(el).__name__
                records.append(StageRecord(name, "amp", P, P_new, _DB(G),
                                           float(nr.nf_dB), rho, p_ase,
                                           meta={"converged": res.meta.get("converged")}))
                P = P_new
        G_tot = P / P_in0
        nf_tot = nf_from_psd(G_tot, rho, nu_s)
        dnu_ref = _ref_dnu(signal_lambda_m, ref_bw_nm)
        m_out = 2
        p_ase_ref = m_out * rho * dnu_ref
        osnr = P / p_ase_ref if p_ase_ref > 0.0 else np.inf
        return ChainResult(float(signal_lambda_m), records, float(P), float(_DB(G_tot)),
                           float(_DB(nf_tot)), float(rho), float(_DB(osnr)), float(ref_bw_nm))


def _ref_dnu(lambda_m, ref_bw_nm):
    return C_LIGHT / lambda_m ** 2 * (ref_bw_nm * 1e-9)


def _meta_m(res):
    return int(res.meta.get("m_modes", 2))


def _total_fwd_ase_W(res):
    idx = [i for i in range(len(res.kind)) if res.kind[i] == "ase" and res.u[i] > 0]
    return float(np.sum(res.power_W[idx, -1])) if idx else 0.0
