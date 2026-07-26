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
trailing edge sees a saturated gain of 1. The per-sample factor of that law
(frantz_nodvik_instantaneous_gain) is the SINGLE HOME of the algebra: pulse.propagate_gnlse's
opt-in SaturableGain(frantz_nodvik=True) mode reads it rather than re-deriving it (audit A-10).

SAME OBJECT, SAME PHYSICS. The march reads the amplifier's own opt-ins rather than a subset of
them: the normalised upconversion coefficient (either constructor spelling -- audit A-1) enters
the semi-implicit nbar2 update, and an axial temperature profile left set by
thermal.solve_with_thermal_feedback scales every sigma_e-proportional coefficient by the same
per-z McCumber factor the steady solve uses (audit A-5). The long-time limit of the transient is
therefore amp.solve()'s fixed point for the SAME amp -- SUBJECT TO the ASE-runaway condition
below, which is what the gates assert.

SCOPE LIMIT -- ASE RUNAWAY (audit A-7). The step (i) propagation freezes the gain, so the ASE
generated inside a step does NOT deplete the inversion that made it. At the march's fixed point
that is exactly right (propagating the CONVERGED g(z) reproduces the steady solve), but the
fixed-point ITERATION is only stable while the ASE stays a perturbation: the ASE grows like
exp(INT g dz), so once that exponent is large a small nbar2 error is amplified without bound and
the march settles somewhere else -- or nowhere. Two measured examples on this repo's own
amplifiers: a 400 W cladding-pumped Yb amplifier driven at a 0.1 mW signal (frozen-step ASE
5e6 W against 400 W launched) lands 54 dB below its own solve(); the SAME amplifier at a 220 K
uniform profile (McCumber > 1 below T_ref, i.e. a COLD profile -- reachable through the audit-A-5
temperature path) lands 31 dB below. Both were silent. simulate_transient now MEASURES the
condition every step and, when it trips, sets meta['quasi_static_valid'] = False (with the two
measured margins) and raises a RuntimeWarning naming the limitation. Sub-stepping does NOT help
-- _propagate_fixed is the EXACT solution of the frozen-gain ODE, so a finer z or t grid returns
the same over-amplified ASE; only a genuinely ASE-coupled transient (a saturating gain inside the
step) would, which is out of scope here. Use amp.solve() for the steady operating point in that
regime.

Pure numpy/scipy; SI units. docs/fiber_amp_model_spec.md sec.8.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp.steady_state import FiberAmplifier

__all__ = ["TransientResult", "simulate_transient", "saturation_energy",
           "frantz_nodvik_output_energy", "frantz_nodvik_gain", "frantz_nodvik_pulse",
           "frantz_nodvik_instantaneous_gain"]


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


def _cumtrapz2(Y, x):
    """Row-wise cumulative trapezoid along axis 1 with a leading 0 column. For independent rows
    this is exactly the per-row scalar _cumtrapz (same operands, same order), so batching the
    channels through it is bit-identical to the old per-channel loop (audit S6-1: verified
    maxdiff 0.0 at K=6/42/82; 5.9x end-to-end on an ASE-resolved transient)."""
    out = np.zeros_like(Y)
    out[:, 1:] = np.cumsum(0.5 * (Y[:, 1:] + Y[:, :-1]) * np.diff(x), axis=1)
    return out


def _propagate_fixed(z, g, s, bc, u):
    """Powers P (K, Nz) for FIXED per-channel gain g(z) and source s(z) (both (K, Nz)) with
    boundary powers bc (K,) and directions u (K,). Exact integrating-factor solution of
    dP/dz = u (g P + s): forward channels seeded at z=0, backward at z=L. No iteration -- the
    gain does not depend on P here (that coupling lives in the slow nbar2 update). Vectorized
    over the channel axis (each channel is an independent row problem)."""
    P = np.empty_like(g)
    fwd = u > 0.0
    if np.any(fwd):
        G = _cumtrapz2(g[fwd], z)
        P[fwd] = np.exp(G) * (bc[fwd][:, None] + _cumtrapz2(s[fwd] * np.exp(-G), z))
    bwd = ~fwd
    if np.any(bwd):
        zeta = z[-1] - z[::-1]
        G = _cumtrapz2(g[bwd][:, ::-1], zeta)
        Pr = np.exp(G) * (bc[bwd][:, None] + _cumtrapz2(s[bwd][:, ::-1] * np.exp(-G), zeta))
        P[bwd] = Pr[:, ::-1]
    return P


# ---- quasi-static validity monitor (audit A-7) ---------------------------------------------
# Thresholds derived from a fine sweep of this repo's own amplifiers (Yb 400 W cladding-pumped
# booster, signal 5 W -> 1 uW and uniform T profiles 800 K -> 150 K; C-band EDFA, pump 0.05-1 W,
# L 4-30 m, signal 1e-5/1e-7 W), scoring each run by |G_transient(t_end) - G_solve()|:
#
#   frozen-step ASE power / total LAUNCHED power    valid runs <= 0.60   broken runs >= 2.2e4
#   frozen-step INT g dz over the ASE channels      valid runs <= 13.9   broken runs >= 28.5
#
# Both separate by >4 decades / >14 e-folds, so the limits sit deep inside the gap. The ASE
# criterion is referenced to the LAUNCHED power, not to the signal: a perfectly healthy C-band
# EDFA at a 10 uW input legitimately runs 30-3000x more in-band ASE than signal (measured
# disagreement 0.001 dB), so an ASE/signal threshold would fire on the library's most common
# amplifier. Against the launched power the criterion is energy conservation -- the frozen step
# cannot emit more than was put in -- which no valid run comes within a factor of 1.6 of.
_ASE_TO_LAUNCHED_LIMIT = 1.0
_GAIN_INTEGRAL_LIMIT = 20.0


def _no_raman(amp):
    """The transient fixed-inversion propagator assumes per-channel LINEAR gain; the SRS
    Stokes exchange is bilinear in the powers, so a raman-coupled amplifier must refuse rather
    than silently drop the coupling (the S3-1 clone-drop failure mode, applied to transients)."""
    if getattr(amp, "raman", None) is not None:
        raise NotImplementedError("simulate_transient does not support RamanStokes coupling; "
                                  "solve the steady state with raman or run the transient "
                                  "without it")


def simulate_transient(amp: FiberAmplifier, t_grid, *,
                       signal_drive: Optional[Callable] = None,
                       pump_drive: Optional[Callable] = None,
                       n_nodes: int = 81, nbar2_0=None) -> TransientResult:
    """March the amplifier's inversion nbar2(z, t) over t_grid. signal_drive(t) / pump_drive(t),
    if given, return the input-power vector (length = number of signals / pumps) at time t --
    step functions of them produce add/drop transients; default (None) holds the configured
    input powers. Powers are quasi-static each step; nbar2 advances by an exponential integrator.
    Initialised from the steady state at the first drive unless nbar2_0 is supplied.

    VALIDITY (audit A-7): the frozen-inversion step is only trustworthy while the ASE stays a
    perturbation. The march measures that every step and reports it on the result --
    meta['quasi_static_valid'] (bool), meta['max_ase_to_launched'], meta['max_ase_gain_integral'],
    meta['validity_limits'], meta['validity_warning'] -- and raises a RuntimeWarning when the flag
    goes False; the arrays are still returned (unchanged) but must not be trusted. See the module
    docstring for the two measured failure cases and why sub-stepping cannot fix them."""
    _no_raman(amp)
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
    sig_esa = ch.sigma_esa[:, None]
    gam = ch.gamma[:, None]
    loss = ch.loss_per_m[:, None]
    ase_col = is_ase[:, None]
    src_pref = ase_col * (gam * na * sig_e * m * (hnu * ch.dnu_hz)[:, None])
    # Optional axial temperature profile (set_temperature_profile / thermal.solve_with_thermal_
    # feedback): the SAME (K, Nz) McCumber sigma_e scaling the steady solve applies, on every
    # sigma_e-proportional coefficient -- the local emission gain, the ASE spontaneous source and
    # the stimulated-emission rate (audit A-5: the transient ran the COLD model on an amplifier
    # whose profile was explicitly SET, so its long-time limit disagreed with amp.solve()).
    # mcc is None (no profile) -> every expression below is the original one, byte-identically.
    mcc = amp._mcc_matrix(ch, z)
    sig_e_z = sig_e if mcc is None else sig_e * mcc
    if mcc is not None:
        src_pref = src_pref * mcc
    flux_e_pref = ch.sigma_e[:, None] if mcc is None else ch.sigma_e[:, None] * mcc

    def g_s(n2z):
        n2 = n2z[None, :]
        g = gam * na * (sig_e_z * n2 - sig_a * (1.0 - n2) - sig_esa * n2) - loss
        if amp.concentration is not None:
            g = g - gam * amp._n_dark * sig_a
            g = g - amp.concentration.photodarkening_loss_per_m(n2z)[None, :]
        s = src_pref * n2
        return g, s

    def rates(P):
        flux = ch.gamma[:, None] * P / (hnu[:, None] * A)       # (K, Nz)
        R_a = np.sum(ch.sigma_a[:, None] * flux, axis=0)         # (Nz,)
        R_e = np.sum(flux_e_pref * flux, axis=0)                 # sigma_e (x McCumber, if profiled)
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

    # quasi-static validity monitor (audit A-7): the frozen-gain step over-amplifies ASE once the
    # ASE stops being a perturbation, and the march then converges somewhere other than
    # amp.solve()'s fixed point. Measure the two documented margins every step; nothing here
    # touches P or n2, so every returned array is byte-identical to the unmonitored march.
    dz = np.diff(z)
    ase_fwd = np.where(is_ase & (u > 0.0))[0]
    ase_bwd = np.where(is_ase & (u < 0.0))[0]
    ase_any = np.where(is_ase)[0]
    worst_ase_ratio = 0.0
    worst_gain_integral = 0.0
    nonfinite = False

    for it in range(Nt):
        t = float(t_grid[it])
        bc = boundary(t)
        g, s = g_s(n2)
        P = _propagate_fixed(z, g, s, bc, u)
        # --- validity monitor (read-only) ---
        if not np.all(np.isfinite(P)):
            nonfinite = True
        if ase_any.size:
            p_ase = (float(np.sum(P[ase_fwd, -1])) if ase_fwd.size else 0.0) \
                + (float(np.sum(P[ase_bwd, 0])) if ase_bwd.size else 0.0)
            p_launched = float(np.sum(np.maximum(bc, 0.0)))
            if not np.isfinite(p_ase):
                nonfinite = True
            elif p_launched > 0.0:
                worst_ase_ratio = max(worst_ase_ratio, p_ase / p_launched)
            gi = np.sum(0.5 * (g[ase_any, 1:] + g[ase_any, :-1]) * dz, axis=1)
            gi_max = float(np.max(gi))
            if np.isfinite(gi_max):
                worst_gain_integral = max(worst_gain_integral, gi_max)
            else:
                nonfinite = True
        n2_zt[it] = n2
        for j, i in enumerate(sig_idx):
            sig_out[it, j] = P[i, -1]
            gain_dB[it, j] = 10.0 * np.log10(P[i, -1] / max(bc[i], 1e-300))
        for j, i in enumerate(pmp_idx):
            pmp_out[it, j] = P[i, -1] if u[i] > 0 else P[i, 0]
        if it == Nt - 1:
            break
        # advance nbar2 over dt with an exponential integrator on the local balance. The
        # cooperative-upconversion loss C n_a n2^2 is folded in SEMI-IMPLICITLY by linearizing
        # about the current n2 (rate C n_a n2_current per unit n2), so the update stays
        # unconditionally stable for any dt and its fixed point satisfies the exact quadratic
        # balance R_a(1-n2) = n2/tau + R_e n2 + C n_a n2^2 (audit S3-38: the old explicit-Euler
        # bolt-on biased the converged inversion by O(dt) and broke the stability claim).
        dt = float(t_grid[it + 1] - t)
        R_a, R_e = rates(P)
        B = R_a + R_e + inv_tau
        # Gate on the NORMALISED coefficient, not on the ConcentrationModel: FiberAmplifier
        # folds BOTH documented spellings (concentration.c_up_m3_s and the raw
        # upconversion_C_up=) into self.upconversion_C_up, and the steady-state _nbar2_c reads
        # only that attribute. Gating on the object dropped the raw-C_up opt-in entirely --
        # the transient came out bit-equal to the C_up = 0 ideal and disagreed with its own
        # steady-state solve by 4.5 dB at the repo's own test value 3e-23 (audit A-1).
        if amp.upconversion_C_up > 0.0:
            B = B + amp.upconversion_C_up * na * n2
        n2_ss = R_a / B
        n2 = n2_ss + (n2 - n2_ss) * np.exp(-B * dt)
        n2 = np.clip(n2, 0.0, 1.0)

    reasons = []
    if nonfinite:
        reasons.append("non-finite channel powers appeared during the march")
    if worst_ase_ratio > _ASE_TO_LAUNCHED_LIMIT:
        reasons.append("frozen-step ASE power reached {:.3g}x the LAUNCHED optical power (limit "
                       "{:g}x)".format(worst_ase_ratio, _ASE_TO_LAUNCHED_LIMIT))
    if worst_gain_integral > _GAIN_INTEGRAL_LIMIT:
        reasons.append("frozen-step ASE gain integral reached INT g dz = {:.4g} (limit {:g}, i.e."
                       " a single-pass ASE gain of e^{:g})".format(
                           worst_gain_integral, _GAIN_INTEGRAL_LIMIT, _GAIN_INTEGRAL_LIMIT))
    warn_msg = None
    if reasons:
        warn_msg = (
            "simulate_transient: the quasi-static (frozen-inversion) step is OUT OF ITS VALID "
            "REGIME -- " + "; ".join(reasons) + ". The step propagates exp(INT g dz) at a FIXED "
            "inversion, so ASE generated inside a step does not deplete the gain that made it; "
            "once the ASE stops being a perturbation the march converges somewhere other than "
            "amp.solve()'s fixed point (measured: 54 dB low on a 400 W Yb booster at a 0.1 mW "
            "signal, 31 dB low on the same amplifier at a 220 K profile). These results are NOT "
            "trustworthy: use amp.solve() for the steady operating point, raise the signal power, "
            "or narrow/disable the ASE band. Sub-stepping does not help -- the frozen-gain "
            "propagation is already exact. See TransientResult.meta['quasi_static_valid'] "
            "(audit A-7).")
        warnings.warn(warn_msg, RuntimeWarning, stacklevel=2)
    return TransientResult(t_grid, z, n2_zt, sig_out, pmp_out, gain_dB, list(kind),
                           meta={"n_signal": len(sig_idx), "n_pump": len(pmp_idx),
                                 # audit A-7 validity flag: False => the frozen-inversion step
                                 # left its regime and the long-time limit is NOT amp.solve()
                                 "quasi_static_valid": not reasons,
                                 "max_ase_to_launched": float(worst_ase_ratio),
                                 "max_ase_gain_integral": float(worst_gain_integral),
                                 "nonfinite_powers": bool(nonfinite),
                                 "validity_limits": {
                                     "ase_to_launched": _ASE_TO_LAUNCHED_LIMIT,
                                     "ase_gain_integral": _GAIN_INTEGRAL_LIMIT},
                                 "validity_warning": warn_msg})


def _amp_with_boundary(amp, bc, sig_idx, pmp_idx, kind):
    """Clone amp with pump/signal input powers set from a boundary vector bc (ASE seeds are 0).
    Routed through FiberAmplifier._clone so the raw upconversion_C_up (audit A-1) and the axial
    temperature profile (audit A-5) travel with the clone -- this rebuild used to forward the
    ConcentrationModel alone, so the steady state that SEEDS the transient was a different
    amplifier from the one being marched."""
    from dataclasses import replace
    pumps = list(amp.pumps)
    signals = list(amp.signals)
    for j, i in enumerate(pmp_idx):
        pumps[j] = replace(pumps[j], power_W=float(bc[i]))
    for j, i in enumerate(sig_idx):
        signals[j] = replace(signals[j], power_W=float(bc[i]))
    return amp._clone(pumps=pumps, signals=signals)


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


def _fn_denominator(t_s, p_in_W, G0: float, e_sat_J: float):
    """G0 - (G0-1) exp(-U_in(t)/E_sat), the shared Frantz-Nodvik denominator, with the running
    input energy U_in(t) = INT_-inf^t P_in dt' on the pulse's own time grid. SINGLE HOME for the
    FN instantaneous-saturation algebra: frantz_nodvik_pulse and frantz_nodvik_instantaneous_gain
    are the two spellings of the same law and both read it from here (so does the GNLSE
    propagator's opt-in FN mode, pulse.propagate_gnlse -- audit A-10)."""
    t = np.asarray(t_s, float)
    pin = np.asarray(p_in_W, float)
    U = _cumtrapz(pin, t)
    return G0 - (G0 - 1.0) * np.exp(-U / e_sat_J)


def frantz_nodvik_instantaneous_gain(t_s, p_in_W, small_signal_gain: float, e_sat_J: float):
    """Instantaneous POWER gain G(t) = P_out(t)/P_in(t) of a saturable gain traversed by a pulse
    short against the upper-state lifetime (Frantz & Nodvik, JAP 34:2346 1963):

        G(t) = G0 / (G0 - (G0-1) exp(-U_in(t)/E_sat)),   U_in(t) = INT_-inf^t P_in dt'.

    G0 = exp(g0 L) is the small-signal (unsaturated) gain of the slab. The leading edge (U -> 0)
    sees the full G0; the trailing edge of a pulse with E_in >> E_sat sees G -> 1, which is what
    STEEPENS the pulse -- the temporal reshaping a CW saturation law cannot produce. For a GAIN
    (G0 > 1) G(t) is monotonically decreasing in t and always in [1, G0]. G0 < 1 is the saturable
    ABSORBER (a negative g0; legal here and on both pulse.SaturableGain branches -- audit W5-3):
    the same formula runs, monotonically INCREASING in t and confined to [G0, 1] as the absorber
    bleaches toward transparency.

    COMPOSITION (why this is exact for a split-step propagator): the map is linear in the
    variable u = expm1(U/E_sat), u_out = G0 u_in, so applying it over L1 then L2 with
    G0 = exp(g0 L1), exp(g0 L2) is EXACTLY the single application over L1+L2. A slab can
    therefore be sliced arbitrarily without changing the answer.

    Frame: t is the RETARDED time of the pulse (the frame propagate_gnlse works in); the law
    assumes no pumping and no relaxation during the pulse (see frantz_nodvik_pulse for the
    energy form, and pulse.SaturableGain(recovery_time_s=...) for the finite-lifetime
    generalization that restores the CW limit)."""
    G0 = float(small_signal_gain)
    return G0 / _fn_denominator(t_s, p_in_W, G0, e_sat_J)


def frantz_nodvik_pulse(t_s, p_in_W, small_signal_gain: float, e_sat_J: float):
    """Output temporal power P_out(t) for an input pulse P_in(t) through a saturable gain
    (Frantz-Nodvik):
        P_out(t) = P_in(t) G0 / (G0 - (G0-1) exp(-U_in(t)/E_sat)),  U_in(t) = INT_-inf^t P_in dt'.
    Returns P_out (same shape as p_in_W). The leading edge is amplified by G0, the trailing edge
    by ~1 as the stored energy is depleted; integral(P_out) matches frantz_nodvik_output_energy.
    The per-sample gain factor alone is frantz_nodvik_instantaneous_gain."""
    pin = np.asarray(p_in_W, float)
    G0 = float(small_signal_gain)
    return pin * G0 / _fn_denominator(t_s, pin, G0, e_sat_J)
