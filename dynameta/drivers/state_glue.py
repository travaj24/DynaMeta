"""State glue: per-bias material-STATE solvers -> run_pipeline extra_fields closures.

run_pipeline's extra_fields seam takes fn(bias_point) -> dict merged into the EffectModel field
bundle, with the contract that the KEY SET is identical at every bias. Each factory here wraps
one validated state solver (LLG macrospin, PCM crystallization kinetics, LC director BVP) into
such a closure, producing exactly the key its partner EffectModel reads:

    llg_extra_fields  -> {'m_vector': (3,)}            for VectorMagnetoOpticModel
    pcm_extra_fields  -> {'crystalline_fraction': f}   for PCMModel
    lc_extra_fields   -> {'director_angle_rad': th}    for LiquidCrystalModel

Angle conventions are bridged INSIDE the factories (the LC BVP returns FIELD-AXIS tilt; the
optics reads the PLATE-PLANE angle -- director_to_extra_fields does the pi/2 flip exactly once).
Pure numpy/scipy: importable without DEVSIM/NGSolve.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np

from dynameta.carriers.lc_director import director_profile_bvp, director_to_extra_fields
from dynameta.carriers.llg import LLGMacrospin
from dynameta.carriers.switching import PCMSwitching

__all__ = ["llg_extra_fields", "pcm_extra_fields", "lc_extra_fields"]


def llg_extra_fields(macrospin: LLGMacrospin,
                     H_of_bias: Callable[[object], np.ndarray], *,
                     t_settle_s: float, m0: np.ndarray,
                     n_steps: int = 200) -> Callable[[object], dict]:
    """extra_fields closure: per bias, relax the macrospin under the STATIC field
    H_of_bias(bias_point) [A/m, shape (3,)] for t_settle_s and emit the final unit
    magnetization as {'m_vector': (3,)} for VectorMagnetoOpticModel.

    t_settle_s must cover the Gilbert relaxation (~1/(alpha*omega_p)); the factory raises if
    the trajectory has not settled (|m(t_end) - m(t_end/2)| > 1e-3) rather than silently
    emitting a still-precessing snapshot. Requires alpha > 0 (an undamped spin never settles)."""
    if not (macrospin.alpha > 0.0):
        raise ValueError("llg_extra_fields: macrospin.alpha must be > 0 to reach a settled state")
    if not (t_settle_s > 0.0 and n_steps >= 5):
        raise ValueError("llg_extra_fields: t_settle_s > 0 and n_steps >= 5 required")
    m0 = np.asarray(m0, dtype=np.float64)

    def _fields(bias_point) -> dict:
        H = np.asarray(H_of_bias(bias_point), dtype=np.float64).reshape(3)
        spin = dataclasses.replace(macrospin, H_applied_A_m=lambda t, H=H: H)
        t_eval = np.linspace(0.0, float(t_settle_s), int(n_steps))
        res = spin.simulate(t_eval, m0)
        m_end, m_mid = res.m_t[-1], res.m_t[res.m_t.shape[0] // 2]
        if float(np.max(np.abs(m_end - m_mid))) > 1e-3:
            raise RuntimeError("llg_extra_fields: trajectory not settled by t_settle_s={:.3e} s "
                               "(|dm|={:.2e}); increase t_settle_s or alpha".format(
                                   t_settle_s, float(np.max(np.abs(m_end - m_mid)))))
        return {"m_vector": m_end.copy()}

    return _fields


def pcm_extra_fields(switching: PCMSwitching,
                     pulse_of_bias: Callable[[object], tuple], *,
                     x0: float = 0.0) -> Callable[[object], dict]:
    """extra_fields closure: per bias, integrate the JMAK crystallization kinetics through the
    thermal pulse pulse_of_bias(bias_point) -> (t_s, T_K) [equal-length 1D arrays] starting
    from fraction x0, and emit {'crystalline_fraction': x_final} for PCMModel. Melt-quench
    (T >= T_melt) resets to amorphous inside the integrator; the emitted value is x(t_end)."""
    if not (0.0 <= x0 <= 1.0):
        raise ValueError("pcm_extra_fields: x0 must be in [0, 1]")

    def _fields(bias_point) -> dict:
        t_s, T_K = pulse_of_bias(bias_point)
        x = switching.integrate(np.asarray(t_s, dtype=np.float64),
                                np.asarray(T_K, dtype=np.float64), x0=x0)
        return {"crystalline_fraction": float(x[-1])}

    return _fields


def lc_extra_fields(V_of_bias: Callable[[object], float], *,
                    reduce: str = "midplane",
                    **bvp_kwargs) -> Callable[[object], dict]:
    """extra_fields closure: per bias, solve the two-constant static director BVP at
    V_app = V_of_bias(bias_point) and emit {'director_angle_rad': ...} for LiquidCrystalModel.

    bvp_kwargs are passed to director_profile_bvp (K11, K33, eps_para, eps_perp, d_planar or
    geo, anchoring, flexo, ... -- all keyword-only there). The BVP returns FIELD-AXIS tilt; the
    plate-plane flip the optics expects happens here exactly once (director_to_extra_fields).
    reduce: 'midplane' (cell-centre angle, the natural scalar for a thin LC film), 'mean'
    (profile average), or 'profile' (the full (nz,) gridded angle -- the caller is then
    responsible for aligning it to the optics grid). Raises on a non-converged/untilted-branch
    BVP (result.success False) instead of silently railing n_eff."""
    if reduce not in ("midplane", "mean", "profile"):
        raise ValueError("lc_extra_fields: reduce must be 'midplane', 'mean', or 'profile'")

    def _fields(bias_point) -> dict:
        res = director_profile_bvp(V_app=float(V_of_bias(bias_point)), **bvp_kwargs)
        if not res.success:
            raise RuntimeError("lc_extra_fields: director BVP failed at bias {!r}: {}".format(
                bias_point, res.message or "not converged"))
        theta_field = np.asarray(res.theta_field_rad, dtype=np.float64)
        if reduce == "midplane":
            theta_field = theta_field[theta_field.size // 2]
        elif reduce == "mean":
            theta_field = float(np.mean(theta_field))
        return director_to_extra_fields(theta_field)

    return _fields
