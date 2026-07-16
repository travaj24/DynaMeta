"""Time-domain amplifier dynamics (docs sec.8): the slow metastable-population transient
nbar2(z, t) that drives gain recovery and add/drop cross-gain modulation, plus the fast-pulse
Frantz-Nodvik saturable-gain energy extraction.

TWO TIMESCALES. The upper-state lifetime (~1-10 ms) is enormous next to the fiber transit time
(~ns), so the optical powers are quasi-static: at each instant they satisfy the steady
propagation for the current inversion, while nbar2(z, t) evolves slowly. simulate_transient
exploits this -- each step (i) propagates the powers through the frozen gain g(z; nbar2) by an
exact integrating-factor sweep (forward channels 0->L, backward L->0, no inner relaxation since
the gain is fixed), then (ii) advances nbar2(z) with an exponential integrator on the local
two-level balance, which is unconditionally stable for any step. Add/drop is driven by making
the input powers functions of time.

FAST PULSES. When the pulse is short against the lifetime (no pumping/relaxation during it), gain
saturation is analytic (Frantz & Nodvik, JAP 34:2346 1963): E_sat = h nu A/(Gamma(sigma_a+
sigma_e)), and the output pulse P_out(t) = P_in(t) G0 / (G0 - (G0-1) exp(-U_in(t)/E_sat)) with
G0 the small-signal gain and U_in the running input energy; the leading edge sees full G0, the
trailing edge sees a saturated gain of 1.

Pure numpy/scipy; SI units. docs/fiber_amp_model_spec.md sec.8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp.steady_state import FiberAmplifier

__all__ = ["TransientResult", "simulate_transient", "saturation_energy",
           "frantz_nodvik_output_energy", "frantz_nodvik_gain", "frantz_nodvik_pulse"]


# ============================ transient nbar2(z, t) dynamics ============================

@dataclass
class TransientResult:
    t_s: np.ndarray                 # (Nt,)
    z_m: np.ndarray                 # (Nz,)
    nbar2_zt: np.ndarray            # (Nt, Nz)
    signal_out_W: np.ndarray        # (Nt, n_signal)
    pump_out_W: np.ndarray          # (Nt, n_pump)
    signal_gain_dB: np.ndarray      # (Nt, n_signal)
    kind: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _cumtrapz(y, x):
    """Cumulative trapezoid with a leading 0 (same length as y)."""
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _propagate_fixed(z, g, s, bc, u):
    """Powers P (K, Nz) for FIXED per-channel gain g(z) and source s(z) (both (K, Nz)) with
    boundary powers bc (K,) and directions u (K,). Exact integrating-factor solution of
    dP/dz = u (g P + s): forward channels seeded at z=0, backward at z=L. No iteration -- the
    gain does not depend on P here (that coupling lives in the slow nbar2 update)."""
    K = g.shape[0]
    P = np.empty_like(g)
    for k in range(K):
        if u[k] > 0.0:
            G = _cumtrapz(g[k], z)
            P[k] = np.exp(G) * (bc[k] + _cumtrapz(s[k] * np.exp(-G), z))
        else:
            zr = z[::-1]
            zeta = z[-1] - zr
            G = _cumtrapz(g[k][::-1], zeta)
            Pr = np.exp(G) * (bc[k] + _cumtrapz(s[k][::-1] * np.exp(-G), zeta))
            P[k] = Pr[::-1]
    return P


def simulate_transient(amp: FiberAmplifier, t_grid, *,
                       signal_drive: Optional[Callable] = None,
                       pump_drive: Optional[Callable] = None,
                       n_nodes: int = 81, nbar2_0=None) -> TransientResult:
    """March the amplifier's inversion nbar2(z, t) over t_grid. signal_drive(t) / pump_drive(t),
    if given, return the input-power vector (length = number of signals / pumps) at time t --
    step functions of them produce add/drop transients; default (None) holds the configured
    input powers. Powers are quasi-static each step; nbar2 advances by an exponential integrator.
    Initialised from the steady state at the first drive unless nbar2_0 is supplied."""
    ch, bc0, u, is_ase, kind = amp._plan()
    L = amp.fiber.length_m
    z = np.linspace(0.0, L, n_nodes)
    A = amp.fiber.a_dope_m2
    na = amp._n_active

    sig_idx = [i for i, k in enumerate(kind) if k == "signal"]
    pmp_idx = [i for i, k in enumerate(kind) if k == "pump"]
    inv_tau = 1.0 / ch.tau_s
    hnu = H_PLANCK * (C_LIGHT / ch.lambda_m)
    m = amp.ase.m_modes if amp.ase else 2

    def boundary(t):
        bc = bc0.copy()
        if signal_drive is not None:
            for j, i in enumerate(sig_idx):
                bc[i] = signal_drive(t)[j]
        if pump_drive is not None:
            for j, i in enumerate(pmp_idx):
                bc[i] = pump_drive(t)[j]
        return bc

    # per-z gain/source from an inversion profile
    sig_a = ch.sigma_a[:, None]
    sig_e = ch.sigma_e[:, None]
    gam = ch.gamma[:, None]
    loss = ch.loss_per_m[:, None]
    ase_col = is_ase[:, None]
    src_pref = ase_col * (gam * na * sig_e * m * (hnu * ch.dnu_hz)[:, None])

    def g_s(n2z):
        n2 = n2z[None, :]
        g = gam * na * (sig_e * n2 - sig_a * (1.0 - n2)) - loss
        if amp.concentration is not None:
            g = g - gam * amp._n_dark * sig_a
            g = g - amp.concentration.photodarkening_loss_per_m(n2z)[None, :]
        s = src_pref * n2
        return g, s

    def rates(P):
        flux = ch.gamma[:, None] * P / (hnu[:, None] * A)       # (K, Nz)
        R_a = np.sum(ch.sigma_a[:, None] * flux, axis=0)         # (Nz,)
        R_e = np.sum(ch.sigma_e[:, None] * flux, axis=0)
        return R_a, R_e

    # initial inversion: steady state at the first drive (interp to z), unless supplied
    t0 = float(t_grid[0])
    if nbar2_0 is not None:
        n2 = np.broadcast_to(np.asarray(nbar2_0, float), z.shape).astype(float).copy()
    else:
        amp0 = _amp_with_boundary(amp, boundary(t0), sig_idx, pmp_idx, kind)
        r0 = amp0.solve(n_nodes=n_nodes)
        n2 = np.interp(z, r0.z_m, r0.nbar2_z)

    t_grid = np.asarray(t_grid, float)
    Nt = t_grid.size
    n2_zt = np.empty((Nt, z.size))
    sig_out = np.empty((Nt, len(sig_idx)))
    pmp_out = np.empty((Nt, len(pmp_idx)))
    gain_dB = np.empty((Nt, len(sig_idx)))

    for it in range(Nt):
        t = float(t_grid[it])
        bc = boundary(t)
        g, s = g_s(n2)
        P = _propagate_fixed(z, g, s, bc, u)
        n2_zt[it] = n2
        for j, i in enumerate(sig_idx):
            sig_out[it, j] = P[i, -1]
            gain_dB[it, j] = 10.0 * np.log10(P[i, -1] / max(bc[i], 1e-300))
        for j, i in enumerate(pmp_idx):
            pmp_out[it, j] = P[i, -1] if u[i] > 0 else P[i, 0]
        if it == Nt - 1:
            break
        # advance nbar2 over dt with an exponential integrator on the local linear balance
        dt = float(t_grid[it + 1] - t)
        R_a, R_e = rates(P)
        B = R_a + R_e + inv_tau
        n2_ss = R_a / B
        n2 = n2_ss + (n2 - n2_ss) * np.exp(-B * dt)
        if amp.concentration is not None and amp.concentration.c_up_m3_s > 0.0:
            n2 = n2 - amp.concentration.c_up_m3_s * na * n2 * n2 * dt   # explicit upconv correction
        n2 = np.clip(n2, 0.0, 1.0)

    return TransientResult(t_grid, z, n2_zt, sig_out, pmp_out, gain_dB, list(kind),
                           meta={"n_signal": len(sig_idx), "n_pump": len(pmp_idx)})


def _amp_with_boundary(amp, bc, sig_idx, pmp_idx, kind):
    """Clone amp with pump/signal input powers set from a boundary vector bc (ASE seeds are 0)."""
    from dataclasses import replace
    pumps = list(amp.pumps)
    signals = list(amp.signals)
    for j, i in enumerate(pmp_idx):
        pumps[j] = replace(pumps[j], power_W=float(bc[i]))
    for j, i in enumerate(sig_idx):
        signals[j] = replace(signals[j], power_W=float(bc[i]))
    return FiberAmplifier(amp.ion, amp.fiber, pumps, signals, amp.ase,
                          concentration=amp.concentration)


# ============================ Frantz-Nodvik fast-pulse extraction ============================

def saturation_energy(ion, fiber, lambda_m: float) -> float:
    """Frantz-Nodvik saturation energy E_sat = h nu A_dope / (Gamma (sigma_a + sigma_e)) [J] at
    wavelength lambda_m. The (sigma_a+sigma_e) sum (rather than sigma_e alone) accounts for the
    quasi-three-level ground-state reabsorption that also saturates."""
    from dynameta.optics.fiber_amp.waveguide import overlap_gamma
    nu = C_LIGHT / lambda_m
    gam = float(overlap_gamma(fiber, lambda_m))
    sa = float(ion.sigma_a.sigma(lambda_m))
    se = float(ion.sigma_e.sigma(lambda_m))
    return float(H_PLANCK * nu * fiber.a_dope_m2 / (gam * (sa + se)))


def frantz_nodvik_output_energy(e_in_J, small_signal_gain: float, e_sat_J: float):
    """Extracted pulse energy (Frantz-Nodvik): E_out = E_sat ln{1 + [exp(E_in/E_sat) - 1] G0}.
    G0 = exp(g0 L) is the linear small-signal gain. Limits: E_in << E_sat -> G0 E_in (linear);
    E_in >> E_sat -> E_in + E_sat ln G0 (all stored energy E_sat ln G0 extracted)."""
    ein = np.asarray(e_in_J, float)
    return e_sat_J * np.log1p((np.expm1(ein / e_sat_J)) * small_signal_gain)


def frantz_nodvik_gain(e_in_J, small_signal_gain: float, e_sat_J: float):
    """Saturated energy gain E_out/E_in for the Frantz-Nodvik pulse."""
    return frantz_nodvik_output_energy(e_in_J, small_signal_gain, e_sat_J) / np.asarray(e_in_J,
                                                                                        float)


def frantz_nodvik_pulse(t_s, p_in_W, small_signal_gain: float, e_sat_J: float):
    """Output temporal power P_out(t) for an input pulse P_in(t) through a saturable gain
    (Frantz-Nodvik):
        P_out(t) = P_in(t) G0 / (G0 - (G0-1) exp(-U_in(t)/E_sat)),  U_in(t) = INT_-inf^t P_in dt'.
    Returns P_out (same shape as p_in_W). The leading edge is amplified by G0, the trailing edge
    by ~1 as the stored energy is depleted; integral(P_out) matches frantz_nodvik_output_energy."""
    t = np.asarray(t_s, float)
    pin = np.asarray(p_in_W, float)
    U = _cumtrapz(pin, t)
    G0 = float(small_signal_gain)
    return pin * G0 / (G0 - (G0 - 1.0) * np.exp(-U / e_sat_J))
