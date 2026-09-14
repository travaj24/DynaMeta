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
step) would. Use amp.solve() for the steady operating point in that regime -- or, since
2026-09-15, `ase_mode`.

LIFTING THE LIMIT -- ase_mode (2026-09-15). `simulate_transient(..., ase_mode=...)` selects how
each step treats the ASE feedback, on BOTH amplifier classes:

  "quasi_static"    THE DEFAULT and bit-for-bit the march described above.
  "self_consistent" Each step solves  P = propagate(gain(advance(P))),  advance(P) = the EXISTING
                    population update taken with the rates read off P, to the STEADY SOLVER'S OWN
                    residuals and tolerance (march_ase.solve_self_consistent_step). The gain
                    therefore DEPLETES as the in-step ASE grows, which is the negative feedback
                    the frozen step is missing; the update is implicit, the scheme is stable at
                    any step, and its fixed point is still exactly amp.solve()'s.
  "auto"            Quasi-static until the audit-A-7 monitor PREDICTS a violation -- the two
                    documented margins at the step's own populations, plus the gain integral
                    projected at the populations the explicit update would reach -- then
                    self-consistent for exactly the steps that need it. meta['ase_mode_steps'],
                    ['ase_switch_steps'] and ['ase_switch_times'] log every switch.

The REPORTED powers, the resolved-ASE arrays and frame_as_steady are unchanged in meaning in
every mode: they remain the instantaneous frozen-population propagation at the populations the
same frame reports, which is the exact quasi-static pair at that instant. Only the population
UPDATE reads the self-consistent powers. See march_ase's module docstring for the derivation and
for why a literal "frozen-population BVP" is a no-op.

CO-DOPED (Er:Yb). simulate_transient DISPATCHES an eryb.ErYbAmplifier to
simulate_transient_eryb, which marches TWO coupled z-local reservoirs -- the Er metastable
fraction f2 and the Yb inversion b2 -- with the same quasi-static power step and an
EXPONENTIAL ROSENBROCK update (ETD1 with the exact 2x2 Jacobian) that reduces term for term
to the scalar exponential integrator above when the transfer coupling vanishes. The return
type is unchanged: nbar2_zt is f2 and meta['beta_yb'] carries b2. See that function and the
block comment above it for the integrator's four properties and its first-order-on-the-path
price, and docs/audit/2026-09-13-eryb-transient-closure.md for the measured numbers.

Pure numpy/scipy; SI units. docs/fiber_amp_model_spec.md sec.8 (co-doped: sec.14).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp import march_ase
from dynameta.optics.fiber_amp.steady_state import _relaxation_residuals
# ChannelPlan / SteadyStateResult are annotation targets on TransientResult (F-2/F-3): the
# quoted forward references still need the names bound at module scope for ruff F821 and for
# anyone resolving the annotations at runtime (the CI lint caught them unbound).
from dynameta.optics.fiber_amp.steady_state import (ChannelPlan, FiberAmplifier,
                                                    SteadyStateResult)

__all__ = ["TransientResult", "simulate_transient", "simulate_transient_eryb",
           "saturation_energy", "amplifier_saturation_energy",
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
    # ---- resolved ASE, and the channel plan it is indexed by (audit 2026-08-04 F-2) ----------
    # The march already computes the FULL frozen-inversion power matrix for every channel at every
    # step (_propagate_fixed below); it used to keep only the signal and pump endpoints and discard
    # every ASE channel, so time-resolved ASE / OSNR / noise figure was unreachable -- and
    # noise.analyze_noise accepts only a SteadyStateResult, so there was no bridge at all. Both are
    # free: the numbers were already in memory.
    ase_fwd_W: Optional[np.ndarray] = None      # (Nt, n_bins) forward ASE per bin at z = L
    ase_bwd_W: Optional[np.ndarray] = None      # (Nt, n_bins) backward ASE per bin at z = 0
    ase_lambda_m: Optional[np.ndarray] = None   # (n_bins,) bin centres, sorted by wavelength
    ase_dnu_hz: Optional[np.ndarray] = None     # (n_bins,) bin widths in FREQUENCY [Hz]
    plan: Optional["ChannelPlan"] = None        # the channel structure every array is indexed by
    power_zt: Optional[np.ndarray] = None       # (Nt, K, Nz), only if store_profiles=True

    def ase_psd_1pol_W_Hz(self, direction: str = "fwd") -> np.ndarray:
        """(Nt, n_bins) single-POLARIZATION ASE power spectral density [W/Hz].

        P_bin = rho * m_modes * dnu_bin over m_modes polarization modes, so the per-polarization
        density is P_bin / (m_modes dnu_bin) -- the same convention noise.output_ase_spectrum uses,
        so the two agree bin for bin.
        """
        P = self.ase_fwd_W if direction == "fwd" else self.ase_bwd_W
        if P is None or self.ase_dnu_hz is None:
            raise ValueError("no resolved ASE on this result (the amplifier had no AseBand)")
        m = float(self.meta.get("m_modes", 2))
        return P / (m * np.maximum(self.ase_dnu_hz, 1e-300)[None, :])

    def frame_as_steady(self, index: int) -> "SteadyStateResult":
        """The `index`-th time step packaged as a SteadyStateResult, so the whole existing noise
        layer (noise.analyze_noise / output_ase_spectrum / noise_figure, detection.detection_noise,
        thermal.heat_load_per_m) applies to a transient frame with no new code.

        Requires simulate_transient(..., store_profiles=True), since it needs the full z-resolved
        channel powers rather than just the endpoints.

        WHAT THIS IS AND IS NOT. It is the QUASI-STATIC reading of that instant, exact in the same
        sense the march itself is: the transit time is ~1e-4 of the inversion response time, so the
        powers really are the frozen-inversion solution for nbar2(z, t). It is NOT a claim that the
        frame is a steady state OF THE DRIVE -- the inversion is still moving, and at a burst edge
        it is moving fast. `meta['quasi_static_valid']` is propagated onto the returned result
        (along with `meta['transient_frame']`) so the audit-A-7 flag cannot be lost by going through
        this method.
        """
        if self.power_zt is None or self.plan is None:
            raise ValueError("frame_as_steady needs the z-resolved profiles; re-run "
                             "simulate_transient(..., store_profiles=True)")
        from dynameta.optics.fiber_amp.steady_state import SteadyStateResult as _SSR
        it = int(index)
        P = np.asarray(self.power_zt[it], float)
        pl = self.plan
        sig = pl.indices("signal")
        # The gain denominator is THIS FRAME's launch, P[i, 0], not plan.launched_W -- which is the
        # amplifier's CONFIGURED input and is exactly what signal_drive() overrides. Reading the
        # configured value made the reported gain wrong by the drive ratio itself (measured +3.0105
        # dB under a 2x drive, i.e. 10 log10(2)), and that error propagated into
        # metrics.gain_flatness computed over frames. Signals are forward channels, so z = 0 is
        # their input end -- the same convention signal_gain_dB uses on the march itself.
        gains = np.array([10.0 * np.log10(P[i, -1] / max(float(P[i, 0]), 1e-300)) for i in sig])
        ch = pl.channels
        meta = {"converged": True, "iterations": 0,
                "dnu_hz": pl.dnu_hz.copy(),
                "gamma": pl.gamma.copy(),
                "m_modes": int(self.meta.get("m_modes", 2)),
                "mcc": self.meta.get("mcc"),
                # provenance: this did not come from a relaxation solve
                "transient_frame": it, "t_s": float(self.t_s[it]),
                "quasi_static_valid": bool(self.meta.get("quasi_static_valid", True))}
        # ... and, in the SELF-CONSISTENT modes only, which march produced the frame and whether
        # THAT march was inside its regime -- the audit-A-7 "the flag cannot be lost through the
        # frame" rule, applied to the flag that matters there. Added conditionally because the
        # frame's meta KEY SET is itself pinned as unchanged behaviour
        # (test_fiber_eryb_transient.py::test_single_ion_march_and_efficiency_are_unchanged_from_
        # main asserts it exactly), and in the default mode march_valid IS quasi_static_valid, so
        # there is nothing to carry that is not already there.
        _mode = self.meta.get("ase_mode", "quasi_static")
        if _mode != "quasi_static":
            meta["ase_mode"] = _mode
            meta["march_valid"] = bool(self.meta.get("march_valid", True))
        if ch is not None:
            meta.update({"sigma_a": ch.sigma_a.copy(), "sigma_e": ch.sigma_e.copy(),
                         "sigma_esa": ch.sigma_esa.copy()})
        else:
            # CO-DOPED frame. ChannelPlan.channels is None there by design (every channel carries
            # TWO ions' cross-sections, so there is no single ChannelSet), and without this the
            # frame would reach the noise/thermal layer with no cross-sections at all. Take them
            # -- and THIS frame's Yb inversion -- from the march's own meta, which is where
            # simulate_transient_eryb puts them, so the frame is self-consistent rather than
            # silently spectroscopy-free.
            for key in ("sigma_a", "sigma_e", "sigma_a_er", "sigma_e_er", "sigma_a_yb",
                        "sigma_e_yb", "n_er_m3", "n_yb_m3", "k_tr_m3_s"):
                if key in self.meta:
                    meta[key] = self.meta[key]
            beta = self.meta.get("beta_yb")
            if beta is not None:
                meta["beta_yb_z"] = np.asarray(beta[it], float).copy()
        return _SSR(self.z_m.copy(), P, pl.lambda_m.copy(), pl.direction.copy(),
                    pl.is_ase.copy(), list(pl.kind), np.asarray(self.nbar2_zt[it], float).copy(),
                    gains, meta=meta)


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

# The same two numbers as a mapping, for march_ase.auto_switch. meta['validity_limits'] keeps
# building its own dict literal so a caller cannot mutate the module's copy through a result.
_VALIDITY_LIMITS = {"ase_to_launched": _ASE_TO_LAUNCHED_LIMIT,
                    "ase_gain_integral": _GAIN_INTEGRAL_LIMIT}

# Ceiling on the opt-in (Nt, K, Nz) float64 profile matrix of simulate_transient(store_profiles=
# True). 2 GiB is well above any legitimate use of frame_as_steady (the audit's own 80-step,
# 42-channel, 121-node frame is 3.3 MB) and well below the point where the allocation stops being
# an exception and becomes an OOM kill.
_STORE_PROFILES_MAX_BYTES = 2 * 1024 ** 3


def _no_raman(amp):
    """The transient fixed-inversion propagator assumes per-channel LINEAR gain; the SRS
    Stokes exchange is bilinear in the powers, so a raman-coupled amplifier must refuse rather
    than silently drop the coupling (the S3-1 clone-drop failure mode, applied to transients)."""
    if getattr(amp, "raman", None) is not None:
        raise NotImplementedError("simulate_transient does not support RamanStokes coupling; "
                                  "solve the steady state with raman or run the transient "
                                  "without it")


def _ase_share(P, ase_fwd, ase_bwd, p_launched):
    """Total ASE leaving the fiber as a fraction of the LAUNCHED power -- the audit-A-7 margin,
    written so it can be evaluated on an ARBITRARY power profile. The in-loop monitor keeps its
    own inlined copy (touching it would put arithmetic on the default march's path); this one
    exists for "auto", which needs the same quantity at the PROJECTED populations."""
    p_ase = (float(np.sum(P[ase_fwd, -1])) if ase_fwd.size else 0.0) \
        + (float(np.sum(P[ase_bwd, 0])) if ase_bwd.size else 0.0)
    if not np.isfinite(p_ase):
        return float("inf")
    return p_ase / p_launched if p_launched > 0.0 else 0.0


def _ase_mode_meta(mode, ctrl, step_mode, switch_steps, t_grid, step_resid, want_resid):
    """The ase_mode block of TransientResult.meta -- one home, both marches. Every entry is None
    or 0 in the default mode, so a caller can read them unconditionally and see at a glance that
    nothing self-consistent happened."""
    sw = np.asarray(switch_steps, int)
    return {
        "ase_mode": mode,
        # (Nt,) int8: 0 = this step was taken quasi-statically, 1 = self-consistently
        "ase_mode_steps": step_mode,
        "ase_switch_steps": None if ctrl is None else sw,
        "ase_switch_times": None if ctrl is None else np.asarray(t_grid, float)[sw],
        "n_self_consistent_steps": 0 if ctrl is None else int(ctrl.n_steps),
        "ase_probe_propagations": 0 if ctrl is None else int(ctrl.n_probes),
        "ase_inner_iterations_total": 0 if ctrl is None else int(ctrl.n_iter_total),
        "ase_inner_iterations_max": 0 if ctrl is None else int(ctrl.max_iter_used),
        "ase_inner_relax": (None if ctrl is None
                            else float(ctrl.relax_ladder[min(int(ctrl.rung),
                                                             len(ctrl.relax_ladder) - 1)])),
        "ase_inner_tol": None if ctrl is None else float(ctrl.tol),
        "ase_inner_nonconverged_steps": 0 if ctrl is None else int(ctrl.n_nonconverged),
        "max_self_consistency_residual": (None if ctrl is None
                                          else float(ctrl.max_profile_residual)),
        "max_step_power_residual": float(step_resid) if want_resid else None,
    }


def _self_consistent_reasons(ctrl, nonfinite):
    """Why a SELF-CONSISTENT / AUTO march is not trustworthy. The audit-A-7 quasi-static margins
    are deliberately NOT among them: exceeding them is the condition this mode exists to handle,
    and it is reported (meta['quasi_static_valid'] and the two margins) without being an error.
    What IS an error is an inner solve that did not reach its tolerance, or powers that went
    non-finite anyway."""
    reasons = []
    if nonfinite:
        reasons.append("non-finite channel powers appeared during the march")
    if ctrl is not None and ctrl.n_nonconverged:
        reasons.append("the self-consistent step did not reach its tolerance on %d of %d steps "
                       "(worst interior residual %.3g against tol %g, damping down to relax = "
                       "%g)" % (ctrl.n_nonconverged, ctrl.n_steps, ctrl.max_profile_residual,
                                ctrl.tol,
                                ctrl.relax_ladder[min(int(ctrl.rung),
                                                      len(ctrl.relax_ladder) - 1)]))
    return reasons


def simulate_transient(amp: FiberAmplifier, t_grid, *,
                       signal_drive: Optional[Callable] = None,
                       pump_drive: Optional[Callable] = None,
                       n_nodes: int = 81, nbar2_0=None,
                       store_profiles: bool = False,
                       ase_mode: str = "quasi_static",
                       ase_tol: float = 1e-6, ase_max_iter: int = 120,
                       ase_step_residual: bool = False) -> TransientResult:
    """March the amplifier's inversion nbar2(z, t) over t_grid. signal_drive(t) / pump_drive(t),
    if given, return the input-power vector (length = number of signals / pumps) at time t --
    step functions of them produce add/drop transients; default (None) holds the configured
    input powers. Powers are quasi-static each step; nbar2 advances by an exponential integrator.
    Initialised from the steady state at the first drive unless nbar2_0 is supplied.

    CO-DOPED AMPLIFIERS (2026-09-13). An eryb.ErYbAmplifier is DISPATCHED to
    simulate_transient_eryb, which marches the two coupled reservoirs (Er f2, Yb b2) and returns
    the same TransientResult type with nbar2_zt = f2 and meta['beta_yb'] = b2. The call form is
    identical, so downstream code that swaps the amplifier class needs no branch; nbar2_0 there
    additionally accepts a (f2, b2) tuple. See that function for the integrator and its limits.

    VALIDITY (audit A-7): the frozen-inversion step is only trustworthy while the ASE stays a
    perturbation. The march measures that every step and reports it on the result --
    meta['quasi_static_valid'] (bool), meta['max_ase_to_launched'], meta['max_ase_gain_integral'],
    meta['validity_limits'], meta['validity_warning'] -- and raises a RuntimeWarning when the flag
    goes False; the arrays are still returned (unchanged) but must not be trusted. See the module
    docstring for the two measured failure cases and why sub-stepping cannot fix them.

    ase_mode (2026-09-15) LIFTS that limit rather than only reporting it:

      "quasi_static"    THE DEFAULT; bit-for-bit the march above, and the only mode that can
                        report meta['quasi_static_valid'] = False as a reason not to trust the
                        result.
      "self_consistent" Every step solves the ASE-coupled power problem -- P propagating through
                        the gain of the populations that step ENDS at, the populations advanced
                        by the existing exponential integrator with those powers -- to the steady
                        solver's own residuals and tolerance (`ase_tol`, default 1e-6, capped at
                        `ase_max_iter` inner iterations, warm-started from the previous step).
                        Stable at any step; fixed point still exactly amp.solve()'s.
      "auto"            Quasi-static until the A-7 monitor PREDICTS a violation (the two margins
                        at the current populations, plus the ASE gain integral projected at the
                        populations the explicit update would reach, each against half its
                        documented limit), then self-consistent for exactly those steps.

    In the two new modes the result additionally carries meta['ase_mode'], ['march_valid'] (the
    flag to test in those modes -- False only if powers went non-finite or an inner solve did not
    converge), ['ase_mode_steps'] (Nt int8: 0 quasi-static, 1 self-consistent), ['ase_switch_
    steps'], ['ase_switch_times'], ['n_self_consistent_steps'], ['ase_probe_propagations'] (the
    extra propagations "auto" spent measuring a projected ASE level), ['ase_inner_iterations_
    total'], ['ase_inner_iterations_max'], ['ase_inner_relax'], ['ase_inner_nonconverged_steps']
    and ['max_self_consistency_residual']. frame_as_steady carries ['march_valid'] and
    ['ase_mode'] onto the frame alongside the existing ['quasi_static_valid'].

    ase_step_residual (default False, any mode) additionally measures the STEP-BY-STEP closure:
    after each update, re-propagate at the populations the step ended at and report the worst
    relative disagreement with the powers the step actually used, on
    meta['max_step_power_residual']. It costs one extra propagation per step, which is why it is
    opt-in; it is the number that separates a step whose own inversion sustains its powers from
    one whose does not.

    RESOLVED ASE (audit F-2): the result now carries the time-resolved ASE spectrum the march was
    already computing -- ase_fwd_W / ase_bwd_W (Nt, n_bins), ase_lambda_m, ase_dnu_hz, and
    ase_psd_1pol_W_Hz() -- plus the ChannelPlan every array is indexed by. store_profiles=True
    additionally keeps the whole (Nt, K, Nz) power matrix, which enables frame_as_steady(index) and
    therefore lets noise.analyze_noise / detection.detection_noise run on a transient frame. The
    memory cost is Nt*K*Nz floats, so it is opt-in -- and a request above 2 GiB
    (_STORE_PROFILES_MAX_BYTES) is REFUSED with the shape, because past that point the allocation
    stops raising and starts taking the process down with it."""
    _no_raman(amp)
    ase_mode = march_ase.check_ase_mode(ase_mode)
    # DISPATCH (2026-09-13). The body below is the SINGLE-ION march: it reads _n_active, the
    # McCumber matrix and the ConcentrationModel, and advances ONE scalar reservoir per node. A
    # CO-DOPED amplifier has two coupled ion populations and no single nbar2, so it is handed to
    # simulate_transient_eryb -- same signature, same TransientResult (nbar2_zt = the Er
    # metastable fraction, meta['beta_yb'] = the Yb inversion). Every OTHER class still refuses by
    # name rather than letting the _plan() shape mismatch surface as "too many values to unpack
    # (expected 5)", which tells the caller nothing.
    if not isinstance(amp, FiberAmplifier):
        from dynameta.optics.fiber_amp.eryb import ErYbAmplifier
        if isinstance(amp, ErYbAmplifier):
            return simulate_transient_eryb(amp, t_grid, signal_drive=signal_drive,
                                           pump_drive=pump_drive, n_nodes=n_nodes,
                                           nbar2_0=nbar2_0, store_profiles=store_profiles,
                                           ase_mode=ase_mode, ase_tol=ase_tol,
                                           ase_max_iter=ase_max_iter,
                                           ase_step_residual=ase_step_residual)
        raise TypeError("simulate_transient supports FiberAmplifier and ErYbAmplifier only, not "
                        "%s: the single-ion march reads the single-ion inversion state "
                        "(_n_active, the McCumber matrix, the ConcentrationModel) and the "
                        "co-doped march reads the two-reservoir rate pair; neither fits this "
                        "class. Use %s.solve() for its steady operating point."
                        % (type(amp).__name__, type(amp).__name__))
    ch, bc0, u, is_ase, kind = amp._plan()
    L = amp.fiber.length_m
    z = np.linspace(0.0, L, n_nodes)
    A = amp.fiber.a_dope_m2
    na = amp._n_active
    if store_profiles:
        # Refuse BEFORE the initial steady solve, and before np.empty commits the pages. The
        # (Nt, K, Nz) matrix scales as the product of three innocuous-looking arguments: an
        # ASE-resolved amplifier at 2000 time steps and n_nodes = 201 is already 6-12 GiB, which
        # on this class of box is an out-of-memory kill of the whole process rather than an
        # exception. Say the shape and the size instead.
        _need = 8 * int(np.size(t_grid)) * int(ch.lambda_m.size) * int(n_nodes)
        if _need > _STORE_PROFILES_MAX_BYTES:
            raise ValueError(
                "simulate_transient(store_profiles=True) would allocate %.2f GiB for the "
                "(Nt, K, Nz) = (%d, %d, %d) power matrix, above the %.2f GiB guard. Shorten "
                "t_grid, drop n_nodes, narrow the AseBand (K counts BOTH ASE directions), or "
                "leave store_profiles=False -- ase_fwd_W / ase_bwd_W are kept either way, and "
                "frame_as_steady is the only thing that needs the full matrix."
                % (_need / 1024.0 ** 3, np.size(t_grid), ch.lambda_m.size, n_nodes,
                   _STORE_PROFILES_MAX_BYTES / 1024.0 ** 3))

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

    def advance(n2z, P, dt):
        """nbar2 after dt, from the powers P -- the exponential integrator on the local balance.
        THE single home of the update: the quasi-static march calls it once per step with the
        frozen-inversion powers, and the self-consistent step calls it inside its iteration with
        the powers being solved for. Same arithmetic, same order, either way.

        The cooperative-upconversion loss C n_a n2^2 is folded in SEMI-IMPLICITLY by linearizing
        about the current n2 (rate C n_a n2_current per unit n2), so the update stays
        unconditionally stable for any dt and its fixed point satisfies the exact quadratic
        balance R_a(1-n2) = n2/tau + R_e n2 + C n_a n2^2 (audit S3-38: the old explicit-Euler
        bolt-on biased the converged inversion by O(dt) and broke the stability claim).

        Gate on the NORMALISED coefficient, not on the ConcentrationModel: FiberAmplifier folds
        BOTH documented spellings (concentration.c_up_m3_s and the raw upconversion_C_up=) into
        self.upconversion_C_up, and the steady-state _nbar2_c reads only that attribute. Gating
        on the object dropped the raw-C_up opt-in entirely -- the transient came out bit-equal to
        the C_up = 0 ideal and disagreed with its own steady-state solve by 4.5 dB at the repo's
        own test value 3e-23 (audit A-1)."""
        R_a, R_e = rates(P)
        B = R_a + R_e + inv_tau
        if amp.upconversion_C_up > 0.0:
            B = B + amp.upconversion_C_up * na * n2z
        n2_ss = R_a / B
        return np.clip(n2_ss + (n2z - n2_ss) * np.exp(-B * dt), 0.0, 1.0)

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

    # ---- self-consistent ASE stepping (2026-09-15) -----------------------------------------
    # ctrl is None for ase_mode == "quasi_static", which is what makes that path the untouched
    # one: every branch below tests it, so the default march performs no extra arithmetic and
    # returns byte-identical arrays.
    fwd_idx = np.where(u > 0.0)[0]
    bwd_idx = np.where(u < 0.0)[0]
    ctrl = None
    step_mode = None
    switch_steps = []
    worst_step_resid = 0.0
    if ase_mode != "quasi_static":
        ctrl = march_ase.SelfConsistentControl(tol=float(ase_tol), max_iter=int(ase_max_iter))
        step_mode = np.zeros(Nt, np.int8)

    # ---- resolved-ASE capture (audit F-2) --------------------------------------------------
    # The forward and backward ASE bins are the SAME spectral grid generated twice by _plan, so one
    # wavelength/bin-width vector describes both. Sort by wavelength once and index with it, so the
    # reported spectrum is monotone in lambda like noise.output_ase_spectrum's.
    if ase_fwd.size:
        ase_order = np.argsort(ch.lambda_m[ase_fwd])
        ase_fwd_idx = ase_fwd[ase_order]
        ase_bwd_idx = ase_bwd[np.argsort(ch.lambda_m[ase_bwd])] if ase_bwd.size else ase_bwd
        ase_lam = ch.lambda_m[ase_fwd_idx].copy()
        ase_dnu = ch.dnu_hz[ase_fwd_idx].copy()
        ase_f_zt = np.empty((Nt, ase_fwd_idx.size))
        ase_b_zt = np.empty((Nt, ase_bwd_idx.size)) if ase_bwd_idx.size else None
    else:
        ase_fwd_idx = ase_bwd_idx = np.empty(0, int)
        ase_lam = ase_dnu = ase_f_zt = ase_b_zt = None
    prof_zt = np.empty((Nt, ch.lambda_m.size, z.size)) if store_profiles else None

    for it in range(Nt):
        t = float(t_grid[it])
        bc = boundary(t)
        g, s = g_s(n2)
        P = _propagate_fixed(z, g, s, bc, u)
        # --- validity monitor (read-only) ---
        # step_* are THIS step's margins (the running worst_* are unchanged); "auto" reads them
        # to decide whether this step needs the self-consistent solver.
        step_ase_ratio = 0.0
        step_gain_integral = 0.0
        step_launched = 0.0
        step_nonfinite = not np.all(np.isfinite(P))
        if step_nonfinite:
            nonfinite = True
        if ase_any.size:
            p_ase = (float(np.sum(P[ase_fwd, -1])) if ase_fwd.size else 0.0) \
                + (float(np.sum(P[ase_bwd, 0])) if ase_bwd.size else 0.0)
            p_launched = float(np.sum(np.maximum(bc, 0.0)))
            step_launched = p_launched
            if not np.isfinite(p_ase):
                nonfinite = True
                step_nonfinite = True
            elif p_launched > 0.0:
                step_ase_ratio = p_ase / p_launched
                worst_ase_ratio = max(worst_ase_ratio, step_ase_ratio)
            gi = np.sum(0.5 * (g[ase_any, 1:] + g[ase_any, :-1]) * dz, axis=1)
            gi_max = float(np.max(gi))
            if np.isfinite(gi_max):
                step_gain_integral = gi_max
                worst_gain_integral = max(worst_gain_integral, gi_max)
            else:
                nonfinite = True
                step_nonfinite = True
        n2_zt[it] = n2
        # audit F-2: keep the resolved ASE the frozen-inversion step just computed (forward at
        # z = L, backward at z = 0), and optionally the whole profile matrix. Read-only w.r.t. the
        # march -- nothing below this touches P or n2 -- so every previously returned array stays
        # byte-identical.
        if ase_f_zt is not None:
            ase_f_zt[it] = P[ase_fwd_idx, -1]
        if ase_b_zt is not None:
            ase_b_zt[it] = P[ase_bwd_idx, 0]
        if prof_zt is not None:
            prof_zt[it] = P
        for j, i in enumerate(sig_idx):
            sig_out[it, j] = P[i, -1]
            gain_dB[it, j] = 10.0 * np.log10(P[i, -1] / max(bc[i], 1e-300))
        for j, i in enumerate(pmp_idx):
            pmp_out[it, j] = P[i, -1] if u[i] > 0 else P[i, 0]
        if it == Nt - 1:
            break
        # advance nbar2 over dt with an exponential integrator on the local balance -- `advance`
        # above is the single home of that update; see its docstring for the semi-implicit
        # upconversion term (audit S3-38) and the normalised-coefficient gate (audit A-1).
        dt = float(t_grid[it + 1] - t)
        P_used = P
        if ctrl is None:
            n2 = advance(n2, P, dt)
        else:
            # The EXPLICIT update is taken first in every mode: "self_consistent" discards it,
            # "auto" uses the populations it reaches to PROJECT this step's ASE gain integral,
            # which is the predictive half of the switch criterion (a cold start looks perfectly
            # healthy right up to the step that inverts the fiber).
            n2_exp = None
            take_sc = ase_mode == "self_consistent"
            if not take_sc:
                n2_exp = advance(n2, P, dt)
                gi_proj = 0.0
                probed = None
                if ase_any.size:
                    g_proj, s_proj = g_s(n2_exp)
                    gi_proj = float(np.max(np.sum(
                        0.5 * (g_proj[ase_any, 1:] + g_proj[ase_any, :-1]) * dz, axis=1)))
                    if march_ase.needs_ase_probe(step_gain_integral, gi_proj):
                        ctrl.n_probes += 1
                        probed = _ase_share(_propagate_fixed(z, g_proj, s_proj, bc, u),
                                            ase_fwd, ase_bwd, step_launched)
                if march_ase.auto_switch(step_ase_ratio, step_gain_integral, gi_proj,
                                         step_nonfinite, _VALIDITY_LIMITS, ctrl.switch_fraction,
                                         ctrl.step_error, probed):
                    ctrl.arm()
                    take_sc = True
                else:
                    take_sc = ctrl.sticky()
            if take_sc:
                P_used, rep = march_ase.solve_self_consistent_step(
                    lambda gg, ss: _propagate_fixed(z, gg, ss, bc, u),
                    lambda Pq: advance(n2, Pq, dt), g_s, fwd_idx, bwd_idx, ctrl, P)
                ctrl.record(rep)
                step_mode[it] = 1
                switch_steps.append(it)
                n2 = advance(n2, P_used, dt)
            else:
                n2 = n2_exp
                # keep the warm start FRESH: a switch that happens ten steps from now should
                # start from this step's profile, not from the last switched step's. Free (an
                # assignment), and it only ever costs inner iterations, never correctness.
                ctrl.seed = P
        if ase_step_residual:
            # STEP-BY-STEP CLOSURE (opt-in, any mode): do the powers this step used survive at
            # the inversion the step ENDED at? Re-propagate there and take the steady solver's
            # own profile residual against them. A self-consistent step answers "yes to the
            # inner tolerance" by construction; a quasi-static step in the runaway regime does
            # not, and that difference is the honest measure of what the mode buys.
            g_chk, s_chk = g_s(n2)
            P_chk = _propagate_fixed(z, g_chk, s_chk, bc, u)
            o_chk, pr_chk = march_ase.residual_views(P_chk, fwd_idx, bwd_idx)
            o_use, pr_use = march_ase.residual_views(P_used, fwd_idx, bwd_idx)
            _e_r, _p_r = _relaxation_residuals(o_chk, o_use, pr_chk, pr_use)
            worst_step_resid = (max(worst_step_resid, _p_r) if np.isfinite(_p_r)
                                else float("inf"))

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
    # The ONLY mode that treats the audit-A-7 margins as a reason to distrust the result is the
    # quasi-static one; the other two step through that regime on purpose. march_valid is the
    # flag to test in those modes (and is identical to quasi_static_valid in the default one).
    if ase_mode == "quasi_static":
        march_valid = not reasons
    else:
        sc_reasons = _self_consistent_reasons(ctrl, nonfinite)
        march_valid = not sc_reasons
        if sc_reasons:
            warnings.warn(
                "simulate_transient(ase_mode=%r): the self-consistent ASE step did not close -- "
                % ase_mode + "; ".join(sc_reasons) + ". Raise ase_max_iter, loosen ase_tol, or "
                "shorten the time step; meta['march_valid'] is False and the returned arrays "
                "must not be trusted.", RuntimeWarning, stacklevel=2)
    if reasons and ase_mode == "quasi_static":
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
    meta = _ase_mode_meta(ase_mode, ctrl, step_mode, switch_steps, t_grid, worst_step_resid,
                          ase_step_residual)
    meta.update({"n_signal": len(sig_idx), "n_pump": len(pmp_idx),
                 # audit A-7 validity flag: False => the frozen-inversion step left its regime
                 # and the long-time limit is NOT amp.solve()
                 "quasi_static_valid": not reasons,
                 # the flag for THIS march: identical to quasi_static_valid in the default mode,
                 # and the one to test in the other two
                 "march_valid": bool(march_valid),
                 "max_ase_to_launched": float(worst_ase_ratio),
                 "max_ase_gain_integral": float(worst_gain_integral),
                 "nonfinite_powers": bool(nonfinite),
                 "validity_limits": {"ase_to_launched": _ASE_TO_LAUNCHED_LIMIT,
                                     "ase_gain_integral": _GAIN_INTEGRAL_LIMIT},
                 "validity_warning": warn_msg,
                 # audit F-2/A-5: the mode count and the per-z McCumber matrix the march
                 # actually used, so a frame handed to the noise layer is self-consistent rather
                 # than mixing a T_ref sigma_e with a hot nbar2 (the audit-A-6 trap, in
                 # transient form).
                 "m_modes": int(m), "mcc": None if mcc is None else mcc.copy()})
    return TransientResult(t_grid, z, n2_zt, sig_out, pmp_out, gain_dB, list(kind), meta=meta,
                           ase_fwd_W=ase_f_zt, ase_bwd_W=ase_b_zt, ase_lambda_m=ase_lam,
                           ase_dnu_hz=ase_dnu, plan=amp.channel_plan(), power_zt=prof_zt)


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


# ================= co-doped (Er:Yb) transient: two coupled z-local reservoirs =================
#
# WHY THIS NEEDS ITS OWN MARCH. The single-ion march advances ONE scalar per node by an
# exponential integrator on a linear balance, n2' = R_a - B n2, whose exact solution over a step
# is n2_ss + (n2 - n2_ss) e^{-B dt}. The co-doped amplifier has TWO reservoirs per node, f2 (Er
# 4I13/2) and b2 (Yb 2F5/2), coupled BOTH ways by the energy transfer: Yb inversion pumps Er
# (+k_tr N_Yb b2 (1 - f2) in the Er equation) and Er GROUND drains Yb (-k_tr N_Er b2 (1 - f2) in
# the Yb equation). Nothing decouples them and the two timescales are far apart -- for this
# repo's phosphosilicate numbers (k_tr 2e-22, N_Er 2e25, tau_Er 10 ms, tau_Yb 1.45 ms) the
# transfer drain is k_tr N_Er = 4.0e3 /s against 1/tau_Yb = 6.9e2 /s and 1/tau_Er = 1.0e2 /s, and
# a saturating 976 nm pump adds R_a_Yb ~ 1e5 /s on top. So the Yb reservoir is 1e3 x faster than
# the Er one it feeds: dt resolving Yb would need ~1e-6 s where the Er transient the caller cares
# about runs for 1e-2 s, i.e. 1e4 steps of pure overhead.
#
# INTEGRATOR: EXPONENTIAL ROSENBROCK (ETD1 with the EXACT local Jacobian).
#
#     y' = F(y),  y = (f2, b2);   y_{n+1} = y_n + [dt phi_1(dt J)] F(y_n),
#     J = dF/dy at y_n,  phi_1(X) = (e^X - I) X^{-1} = SUM_{k>=0} X^k/(k+1)!
#
# Four properties, each of which a cheaper scheme gives up:
#   1. It REDUCES EXACTLY to the existing single-ion update when the coupling vanishes. With
#      k_tr = 0 the Jacobian is diagonal, J11 = -(R_a + R_e + 1/tau) = -B, and the formula
#      collapses to f2_ss + (f2 - f2_ss) e^{-B dt} term for term. The Er-only limit of this march
#      is therefore the same physics as simulate_transient, not merely a close one (gate (a)).
#   2. The AMPLIFIER'S OWN STEADY STATE IS AN EXACT FIXED POINT. F(y) = 0 gives y_{n+1} = y_n for
#      ANY dt, because the increment carries F(y_n) as a factor. That is what lets a constant-drive
#      march started from amp.solve() sit still (gate (b)); an implicit Euler on a LINEARISED pair
#      would only have it as a fixed point if the linearisation were re-centred every step.
#   3. It is UNCONDITIONALLY STABLE, and that is the property a SPLIT scheme does not have.
#      eryb._fb_jacobian shows tr(J) < 0, det(J) > 0 and a non-negative discriminant at every
#      operating point, so both eigenvalues are real and negative; e^{dt J} is a contraction for
#      every dt > 0, and as dt -> inf the step becomes y_n - J^{-1} F(y_n), a Newton step onto the
#      steady state. Populations cannot ring or blow up no matter how the caller spaces t_grid
#      (gate (e) sweeps dt over 1e-8 .. 1e-4 s). A semi-implicit split (freeze b2, advance f2, then
#      swap) SHARES the fixed point -- both sub-steps vanish only at a root of the pair -- but not
#      this: as dt grows each sub-step runs to its own local root, so the step DEGENERATES INTO the
#      naive block Gauss-Seidel iteration whose positive Yb<->Er feedback has spectral radius
#      ~ k_tr^2 N_Er N_Yb tau_Er tau_Yb >> 1 and diverges (eryb module docstring). A split march
#      is therefore stability-limited to dt well inside 1/(k_tr N_Er) ~ 250 us, which is the regime
#      the whole two-timescale design exists to leave.
#   4. It needs NO nonlinear solve. A safeguarded implicit Euler with a Newton iteration per node
#      is the other admissible choice and is equally stable, but it costs an inner loop per node
#      per step with a convergence test that can fail, and it is only first-order accurate on the
#      path -- the same order this is -- so it buys nothing here.
# The price is that ETD1 is first-order in the PATH (the Jacobian is frozen across the step), so
# the transient trajectory carries an O(dt) error even though its endpoints and stability do not.
# That is the same order as the single-ion march and is measured in the audit note.
#
# phi_1 OF A 2x2, ROBUSTLY. The eigen-decomposition route (f(J) = alpha I + beta J with beta a
# divided difference of the two eigenvalues) loses all its digits when the eigenvalues coalesce,
# which happens routinely here -- the Er and Yb blocks cross as the pump rises. So phi_1 is
# evaluated by SCALING AND SQUARING instead, on the matrix itself and with no branch: scale X by
# 2^-m to ||X|| <= 1/2, Taylor both e^X and phi_1(X) there (order 18: the truncation is
# 0.5^19/20! ~ 8e-25), then square up with the exact doubling identities
#     e^{2X} = (e^X)^2,     phi_1(2X) = (1/2) phi_1(X) (e^X + I),
# which follow from e^{2X} - I = (e^X - I)(e^X + I). Everything is vectorized over z.


_PHI1_TAYLOR_ORDER = 18          # 0.5^19/20! ~ 8e-25 truncation at the scaled norm below
_PHI1_SCALE_TARGET = 0.5
_PHI1_MAX_SQUARINGS = 64         # || dt J || up to ~1e19 before this binds; a guard, not a limit


def _mmn(A, B):
    """n x n matrix product of two nested n x n lists of (N,) arrays (row-major). The inner sum
    is accumulated in ASCENDING k with no reordering, so at n = 2 it evaluates exactly
    `a11 b11 + a12 b21` etc. -- the arithmetic the 2x2 spelling has always performed, term for
    term. (An np.matmul over a (N, n, n) stack would be the obvious alternative and is NOT used:
    its reduction order is a BLAS property, so it would make the kernel's output build-dependent
    and turn the existing pinned march values into a false gate.)"""
    n = len(A)
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            acc = A[i][0] * B[0][j]
            for k in range(1, n):
                acc = acc + A[i][k] * B[k][j]
            row.append(acc)
        out.append(row)
    return out


def _i_plus_sXYn(X, Y, s):
    """I + s X Y for nested n x n lists of (N,) arrays (the Horner step of both Taylor series)."""
    Pm = _mmn(X, Y)
    n = len(X)
    return [[(1.0 + s * Pm[i][j]) if i == j else (s * Pm[i][j]) for j in range(n)]
            for i in range(n)]


def _phi1_dt_nxn(J, dt):
    """dt * phi_1(dt J) for a BATCH of n x n Jacobians given as a nested n x n list of (N,)
    arrays; returns the same nested-list shape.

    phi_1(X) = (e^X - I) X^{-1} = SUM_{k>=0} X^k/(k+1)!, so the returned M gives the exponential
    Rosenbrock step y <- y + M F(y). Scaling-and-squaring (see the block comment above): no
    eigenvalue branch, no matrix inverse, exact in the limits M -> dt I (dt -> 0) and
    M -> -J^{-1} (dt -> inf, all eigenvalues in the left half-plane). The size n is a parameter
    rather than a second copy of the algorithm: n = 2 is the single-Yb-pool pair and n = 3 the
    two-population triple (f2, b2c, b2nc)."""
    n = len(J)
    x = [[dt * J[i][j] for j in range(n)] for i in range(n)]
    rows = []
    for i in range(n):
        acc = np.abs(x[i][0])
        for j in range(1, n):
            acc = acc + np.abs(x[i][j])
        rows.append(acc)
    rowmax = rows[0]
    for r in rows[1:]:
        rowmax = np.maximum(rowmax, r)
    nrm = float(np.max(rowmax))
    m = 0
    if np.isfinite(nrm) and nrm > _PHI1_SCALE_TARGET:
        m = min(int(np.ceil(np.log2(nrm / _PHI1_SCALE_TARGET))), _PHI1_MAX_SQUARINGS)
        sc = 0.5 ** m
        x = [[x[i][j] * sc for j in range(n)] for i in range(n)]
    one = np.ones_like(x[0][0])
    zero = np.zeros_like(x[0][0])
    E = [[one if i == j else zero for j in range(n)] for i in range(n)]      # -> e^X
    S = [[one if i == j else zero for j in range(n)] for i in range(n)]      # -> phi_1(X)
    for k in range(_PHI1_TAYLOR_ORDER, 0, -1):
        E = _i_plus_sXYn(x, E, 1.0 / k)
        S = _i_plus_sXYn(x, S, 1.0 / (k + 1.0))
    for _ in range(m):
        EpI = [[E[i][j] + 1.0 if i == j else E[i][j] for j in range(n)] for i in range(n)]
        SE = _mmn(S, EpI)
        S = [[0.5 * SE[i][j] for j in range(n)] for i in range(n)]
        E = _mmn(E, E)
    return [[dt * S[i][j] for j in range(n)] for i in range(n)]


def _phi1_dt_2x2(j11, j12, j21, j22, dt):
    """dt * phi_1(dt J) for a BATCH of 2x2 Jacobians given as four (N,) arrays, returned as a
    4-tuple (m11, m12, m21, m22). The 2x2 spelling of `_phi1_dt_nxn`, kept because the 2x2 is the
    shape the single-Yb-pool march and its pinned gate values speak in."""
    M = _phi1_dt_nxn([[j11, j12], [j21, j22]], dt)
    return (M[0][0], M[0][1], M[1][0], M[1][1])


def _split_eryb_seed(nbar2_0):
    """(f2_seed, b2_seed, b2nc_seed) from the nbar2_0 argument. A TUPLE of length 2 is the
    (f2, b2) pair and of length 3 the two-population triple (f2, b2_coupled, b2_uncoupled);
    anything else (scalar, list, ndarray) is f2 alone and leaves the ytterbium to be seeded from
    its quasi-equilibrium. The tuple/list distinction is deliberate and documented rather than
    sniffed from shapes -- a 2-node mesh would otherwise make `[0.3, 0.4]` ambiguous."""
    if isinstance(nbar2_0, tuple):
        if len(nbar2_0) == 2:
            return nbar2_0[0], nbar2_0[1], None
        if len(nbar2_0) == 3:
            return nbar2_0[0], nbar2_0[1], nbar2_0[2]
        raise ValueError("simulate_transient(nbar2_0=...): a tuple seed must be the pair "
                         "(f2, b2) -- or the triple (f2, b2_coupled, b2_uncoupled) for a "
                         "two-population co-doped amplifier; got length %d" % len(nbar2_0))
    return nbar2_0, None, None


def simulate_transient_eryb(amp, t_grid, *,
                            signal_drive: Optional[Callable] = None,
                            pump_drive: Optional[Callable] = None,
                            n_nodes: int = 81, nbar2_0=None,
                            store_profiles: bool = False,
                            ase_mode: str = "quasi_static",
                            ase_tol: float = 1e-6, ase_max_iter: int = 120,
                            ase_step_residual: bool = False) -> TransientResult:
    """March an eryb.ErYbAmplifier's TWO coupled reservoirs f2(z, t) (Er 4I13/2) and b2(z, t) (Yb
    2F5/2) over t_grid. Same call signature and same return type as simulate_transient, which
    dispatches here, so a caller holding either amplifier class writes the same line.

    ARGUMENTS. signal_drive(t) / pump_drive(t) return the input-power vector at time t exactly as
    for the single-ion march. nbar2_0 additionally accepts a TUPLE (f2_0, b2_0) -- each entry a
    scalar or a length-n_nodes array -- to set BOTH reservoirs. A bare scalar/array still means f2
    alone, and b2 is then seeded from its own quasi-equilibrium at that f2 and the first drive (the
    closed form eryb._b2_quasi_equilibrium, iterated against the frozen-population propagation):
    tau_Yb is ~7x shorter than tau_Er and the Yb reservoir has no independently meaningful history
    at a given Er state, so starting it at 0 would inject a spurious millisecond of Yb charging
    that the caller did not ask for. nbar2_0 = None seeds both from amp.solve() at the first drive.

    THREE RESERVOIRS. When the amplifier carries an UNCOUPLED ytterbium pool (yb_coupled_fraction
    < 1, a secondary transfer k_tr2 > 0 or a migration rate > 0 -- eryb._two_pop), the state is
    the TRIPLE y = (f2, b2c, b2nc) and the very same exponential Rosenbrock step is taken with the
    exact 3x3 Jacobian eryb._fb_jacobian3, through the size-parametrized kernel _phi1_dt_nxn. The
    two-reservoir step is that kernel at n = 2, not a separate scheme. nbar2_0 then also accepts a
    TRIPLE (f2_0, b2c_0, b2nc_0); a PAIR seeds b2nc from the same quasi-equilibrium closed form as
    b2c, and None seeds all three from amp.solve().

    RETURNS a TransientResult with nbar2_zt = f2(t, z) and the Yb inversion history on
    meta['beta_yb'] (Nt, Nz) -- the POPULATION-WEIGHTED inversion f b2c + (1-f) b2nc, which is
    what the optical field sees and what the one-pool model has always reported -- alongside the
    usual resolved ASE arrays, per-signal gain, channel plan and (opt-in) full profile matrix.
    With two pools meta['beta_yb_coupled'] and meta['beta_yb_uncoupled'] carry the pools
    separately (both None-free only in that case). frame_as_steady(i) works on the result and
    carries f2, b2 and both ions' cross-sections onto the frame.

    VALIDITY. The same audit-A-7 quasi-static monitor as the single-ion march (frozen-step ASE
    power against launched power, and the ASE gain integral), reported identically on
    meta['quasi_static_valid'] / ['max_ase_to_launched'] / ['max_ase_gain_integral']. Two
    co-doped-specific diagnostics are added, both REPORTED rather than gated because the
    integrator is stable at any of them: meta['max_dt_times_rate'], the largest ||dt J||_inf the
    march saw (>> 1 means the Yb reservoir was slaved to the Er state within a step rather than
    resolved -- correct for the endpoints, first-order on the path), and
    meta['max_population_overshoot'], the largest excursion outside [0, 1] the clip had to undo.

    ase_mode / ase_tol / ase_max_iter / ase_step_residual (2026-09-15) are exactly the
    simulate_transient options, with the SAME meanings and the same meta keys; the
    self-consistent step wraps the coupled pair's own exponential Rosenbrock update rather than
    the single-ion exponential integrator, and is otherwise the same iteration. See that
    function's docstring, and march_ase's, for what each mode does."""
    _no_raman(amp)
    ase_mode = march_ase.check_ase_mode(ase_mode)
    if getattr(amp, "_Tz", None) is not None:
        raise NotImplementedError(
            "simulate_transient: this co-doped amplifier carries an axial temperature profile "
            "(set_temperature_profile / solve_with_thermal_feedback), which the march does not "
            "yet apply -- the frozen-population step would silently propagate the COLD "
            "cross-sections and disagree with amp.solve() by dB. Call "
            "amp.clear_temperature_profile() to march the isothermal amplifier, or use "
            "amp.solve() / thermal.solve_with_thermal_feedback for the hot steady state.")
    two_pop = bool(getattr(amp, "_two_pop", False))
    fc = float(getattr(amp, "_fc", 1.0))
    pl = amp._plan()
    lam, u, is_ase, kind = pl["lam"], pl["u"], np.asarray(pl["is_ase"], bool), list(pl["kind"])
    bc0 = np.asarray(pl["bc"], float)
    K = int(lam.size)
    L = amp.fiber.length_m
    z = np.linspace(0.0, L, n_nodes)
    t_grid = np.asarray(t_grid, float)
    Nt = int(t_grid.size)
    if store_profiles:
        _need = 8 * Nt * K * int(n_nodes)
        if _need > _STORE_PROFILES_MAX_BYTES:
            raise ValueError(
                "simulate_transient(store_profiles=True) would allocate %.2f GiB for the "
                "(Nt, K, Nz) = (%d, %d, %d) power matrix, above the %.2f GiB guard. Shorten "
                "t_grid, drop n_nodes, narrow the AseBand (K counts BOTH ASE directions and BOTH "
                "bands), or leave store_profiles=False -- ase_fwd_W / ase_bwd_W are kept either "
                "way, and frame_as_steady is the only thing that needs the full matrix."
                % (_need / 1024.0 ** 3, Nt, K, n_nodes, _STORE_PROFILES_MAX_BYTES / 1024.0 ** 3))

    c = amp._coeffs(pl)
    sig_idx = [i for i, k in enumerate(kind) if k == "signal"]
    pmp_idx = [i for i, k in enumerate(kind) if k == "pump"]
    g_e_er, g_a_er = c["g_e_er"][:, None], c["g_a_er"][:, None]
    g_e_yb, g_a_yb = c["g_e_yb"][:, None], c["g_a_yb"][:, None]
    loss_col = c["loss"][:, None]
    s_er_col, s_yb_col = c["s_er"][:, None], c["s_yb"][:, None]
    m_modes = (amp.ase.m_modes if amp.ase is not None
               else (amp.yb_ase.m_modes if amp.yb_ase is not None else 2))

    def boundary(t):
        bc = bc0.copy()
        if signal_drive is not None:
            for j, i in enumerate(sig_idx):
                bc[i] = signal_drive(t)[j]
        if pump_drive is not None:
            for j, i in enumerate(pmp_idx):
                bc[i] = pump_drive(t)[j]
        return bc

    def g_s(f2, b2):
        """Frozen-population gain g (K, Nz) and spontaneous source s (K, Nz) -- term for term the
        bracket of eryb._dP, so the march and the steady solve propagate the SAME operator."""
        f, b = f2[None, :], b2[None, :]
        g = (g_e_er * f - g_a_er * (1.0 - f) + g_e_yb * b - g_a_yb * (1.0 - b) - loss_col)
        return g, s_er_col * f + s_yb_col * b

    # ---- seed the reservoirs ---------------------------------------------------------------
    t0 = float(t_grid[0])
    bc_0 = boundary(t0)
    b2_seed_shift = 0.0
    b2n = None
    if nbar2_0 is None:
        amp0 = _amp_with_boundary(amp, bc_0, sig_idx, pmp_idx, kind)
        r0 = amp0.solve(n_nodes=n_nodes)
        f2 = np.interp(z, r0.z_m, np.asarray(r0.nbar2_z, float))
        if two_pop:
            b2 = np.interp(z, r0.z_m, np.asarray(r0.meta["beta_yb_coupled_z"], float))
            b2n = np.interp(z, r0.z_m, np.asarray(r0.meta["beta_yb_uncoupled_z"], float))
        else:
            b2 = np.interp(z, r0.z_m, np.asarray(r0.meta["beta_yb_z"], float))
    else:
        f_seed, b_seed, bn_seed = _split_eryb_seed(nbar2_0)
        if bn_seed is not None and not two_pop:
            raise ValueError("simulate_transient(nbar2_0=...): a TRIPLE seed was given but this "
                             "amplifier has ONE ytterbium pool (yb_coupled_fraction = 1, "
                             "k_tr2 = 0, no migration), so the seed must be the pair (f2, b2)")
        f2 = np.broadcast_to(np.asarray(f_seed, float), z.shape).astype(float).copy()
        if two_pop:
            b2n = (np.broadcast_to(np.asarray(bn_seed, float), z.shape).astype(float).copy()
                   if bn_seed is not None else None)
        if b_seed is not None and (b2n is not None or not two_pop):
            b2 = np.broadcast_to(np.asarray(b_seed, float), z.shape).astype(float).copy()
        else:
            # Quasi-equilibrium seed, iterated against the frozen-population propagation. With
            # two pools the closed form returns BOTH inversions at the same fixed f2, so the
            # uncoupled reservoir is seeded consistently rather than at zero -- exactly the
            # argument that put the coupled one at its quasi-equilibrium.
            if b_seed is not None:
                b2 = np.broadcast_to(np.asarray(b_seed, float), z.shape).astype(float).copy()
            else:
                b2 = np.zeros_like(f2)
            if two_pop and b2n is None:
                b2n = np.zeros_like(f2)
            for _ in range(24):
                g, s = g_s(f2, b2 if b2n is None else fc * b2 + (1.0 - fc) * b2n)
                P0 = _propagate_fixed(z, g, s, bc_0, u)
                _ra_e, _re_e, ra_y, re_y = amp._rates_profile(c, P0)
                qe = amp._b2_quasi_equilibrium(ra_y, re_y, f2, b2)
                if two_pop:
                    b_new, bn_new = np.clip(qe[0], 0.0, 1.0), np.clip(qe[1], 0.0, 1.0)
                    b2_seed_shift = float(max(np.max(np.abs(b_new - b2)),
                                              np.max(np.abs(bn_new - b2n))))
                    b2n = 0.5 * (b2n + bn_new)
                else:
                    b_new = np.clip(qe, 0.0, 1.0)
                    b2_seed_shift = float(np.max(np.abs(b_new - b2)))
                b2 = 0.5 * (b2 + b_new)              # under-relaxed: the fixed point is on a
                if b2_seed_shift < 1e-12:            # pump the Yb itself depletes
                    break
    f2 = np.clip(f2, 0.0, 1.0)
    b2 = np.clip(b2, 0.0, 1.0)
    if b2n is not None:
        b2n = np.clip(b2n, 0.0, 1.0)

    # ---- outputs ---------------------------------------------------------------------------
    f2_zt = np.empty((Nt, z.size))
    b2_zt = np.empty((Nt, z.size))
    b2c_zt = np.empty((Nt, z.size)) if two_pop else None
    b2n_zt = np.empty((Nt, z.size)) if two_pop else None
    sig_out = np.empty((Nt, len(sig_idx)))
    pmp_out = np.empty((Nt, len(pmp_idx)))
    gain_dB = np.empty((Nt, len(sig_idx)))

    dz = np.diff(z)
    ase_fwd = np.where(is_ase & (u > 0.0))[0]
    ase_bwd = np.where(is_ase & (u < 0.0))[0]
    ase_any = np.where(is_ase)[0]
    worst_ase_ratio = 0.0
    worst_gain_integral = 0.0
    worst_dt_rate = 0.0
    worst_overshoot = 0.0
    nonfinite = False

    step_diag = {"rate": 0.0, "over": None}

    def advance(y, P, dt):
        """The coupled reservoirs after dt, from the powers P -- the exponential Rosenbrock step
        of the block comment above, and THE single home of it. The quasi-static march calls it
        once per step with the frozen-population powers; the self-consistent step calls it inside
        its iteration with the powers being solved for.

        y is the (f2, b2) pair or the (f2, b2c, b2nc) triple; the returned tuple is clipped to
        [0, 1] exactly as the march has always clipped it. The two reported diagnostics
        (||dt J||_inf and the clip overshoot, the latter None when the step went non-finite) are
        left in `step_diag` for `commit_diag` to fold into the running maxima -- so an inner
        iterate that is later discarded cannot pollute them."""
        rt = amp._rates_profile(c, P)
        n_st = len(y)
        if n_st == 3:
            rhs = amp._fb_rhs3(rt[0], rt[1], rt[2], rt[3], y[0], y[1], y[2])
            J = amp._fb_jacobian3(rt[0], rt[1], rt[2], rt[3], y[0], y[1], y[2])
        else:
            rhs = amp._fb_rhs(rt[0], rt[1], rt[2], rt[3], y[0], y[1])
            j11, j12, j21, j22 = amp._fb_jacobian(rt[0], rt[1], rt[2], rt[3], y[0], y[1])
            J = [[j11, j12], [j21, j22]]
        rowsum = np.abs(J[0][0])
        for _j in range(1, n_st):
            rowsum = rowsum + np.abs(J[0][_j])
        for _i in range(1, n_st):
            acc = np.abs(J[_i][0])
            for _j in range(1, n_st):
                acc = acc + np.abs(J[_i][_j])
            rowsum = np.maximum(rowsum, acc)
        step_diag["rate"] = float(np.max(rowsum)) * abs(dt)
        M = _phi1_dt_nxn(J, dt)
        y_new = []
        for _i in range(n_st):
            # accumulate STARTING FROM y[i], left to right -- the association the two-reservoir
            # step has always used (y + m0 r0) + m1 r1. Summing the increment first and adding it
            # to y last is algebraically the same and numerically is not: it moves the pinned
            # one-pool march by ~1 ULP per node, which would falsify the byte-identity gate.
            acc_y = y[_i] + M[_i][0] * rhs[0]
            for _j in range(1, n_st):
                acc_y = acc_y + M[_i][_j] * rhs[_j]
            y_new.append(acc_y)
        if all(np.all(np.isfinite(v)) for v in y_new):
            over = np.maximum(-y_new[0], y_new[0] - 1.0)
            for v in y_new[1:]:
                over = np.maximum(over, np.maximum(-v, v - 1.0))
            step_diag["over"] = float(np.max(over))
        else:
            step_diag["over"] = None
        return tuple(np.clip(v, 0.0, 1.0) for v in y_new)

    def commit_diag():
        """Fold the ACCEPTED step's diagnostics into the running maxima, in the order the march
        has always folded them (the rate first, then the overshoot / the non-finite latch)."""
        nonlocal worst_dt_rate, worst_overshoot, nonfinite
        worst_dt_rate = max(worst_dt_rate, step_diag["rate"])
        if step_diag["over"] is None:
            nonfinite = True
        else:
            worst_overshoot = max(worst_overshoot, step_diag["over"])

    def gs_y(y):
        """(g, s) from a population tuple: the population-weighted Yb inversion the optical field
        sees is what g_s takes, and for ONE pool that is b2 itself (the same object, untouched)."""
        return g_s(y[0], y[1] if len(y) == 2 else fc * y[1] + (1.0 - fc) * y[2])

    # ---- self-consistent ASE stepping (2026-09-15); see simulate_transient for the modes ----
    fwd_idx = np.where(u > 0.0)[0]
    bwd_idx = np.where(u < 0.0)[0]
    ctrl = None
    step_mode = None
    switch_steps = []
    worst_step_resid = 0.0
    if ase_mode != "quasi_static":
        ctrl = march_ase.SelfConsistentControl(tol=float(ase_tol), max_iter=int(ase_max_iter))
        step_mode = np.zeros(Nt, np.int8)

    if ase_fwd.size:
        ase_fwd_idx = ase_fwd[np.argsort(lam[ase_fwd])]
        ase_bwd_idx = ase_bwd[np.argsort(lam[ase_bwd])] if ase_bwd.size else ase_bwd
        ase_lam = lam[ase_fwd_idx].copy()
        ase_dnu = np.asarray(pl["dnu"], float)[ase_fwd_idx].copy()
        ase_f_zt = np.empty((Nt, ase_fwd_idx.size))
        ase_b_zt = np.empty((Nt, ase_bwd_idx.size)) if ase_bwd_idx.size else None
    else:
        ase_fwd_idx = ase_bwd_idx = np.empty(0, int)
        ase_lam = ase_dnu = ase_f_zt = ase_b_zt = None
    prof_zt = np.empty((Nt, K, z.size)) if store_profiles else None

    for it in range(Nt):
        t = float(t_grid[it])
        bc = boundary(t)
        bb = b2 if b2n is None else fc * b2 + (1.0 - fc) * b2n
        g, s = g_s(f2, bb)
        P = _propagate_fixed(z, g, s, bc, u)
        # step_* are THIS step's margins (the running worst_* are unchanged); "auto" reads them.
        step_ase_ratio = 0.0
        step_gain_integral = 0.0
        step_launched = 0.0
        step_nonfinite = not np.all(np.isfinite(P))
        if step_nonfinite:
            nonfinite = True
        if ase_any.size:
            p_ase = (float(np.sum(P[ase_fwd, -1])) if ase_fwd.size else 0.0) \
                + (float(np.sum(P[ase_bwd, 0])) if ase_bwd.size else 0.0)
            p_launched = float(np.sum(np.maximum(bc, 0.0)))
            step_launched = p_launched
            if not np.isfinite(p_ase):
                nonfinite = True
                step_nonfinite = True
            elif p_launched > 0.0:
                step_ase_ratio = p_ase / p_launched
                worst_ase_ratio = max(worst_ase_ratio, step_ase_ratio)
            gi = np.sum(0.5 * (g[ase_any, 1:] + g[ase_any, :-1]) * dz, axis=1)
            gi_max = float(np.max(gi))
            if np.isfinite(gi_max):
                step_gain_integral = gi_max
                worst_gain_integral = max(worst_gain_integral, gi_max)
            else:
                nonfinite = True
                step_nonfinite = True
        f2_zt[it] = f2
        b2_zt[it] = bb
        if two_pop:
            b2c_zt[it] = b2
            b2n_zt[it] = b2n
        if ase_f_zt is not None:
            ase_f_zt[it] = P[ase_fwd_idx, -1]
        if ase_b_zt is not None:
            ase_b_zt[it] = P[ase_bwd_idx, 0]
        if prof_zt is not None:
            prof_zt[it] = P
        for j, i in enumerate(sig_idx):
            sig_out[it, j] = P[i, -1]
            gain_dB[it, j] = 10.0 * np.log10(P[i, -1] / max(bc[i], 1e-300))
        for j, i in enumerate(pmp_idx):
            pmp_out[it, j] = P[i, -1] if u[i] > 0 else P[i, 0]
        if it == Nt - 1:
            break

        # ---- advance the coupled pair (exponential Rosenbrock; see the block comment) -------
        dt = float(t_grid[it + 1] - t)
        y = (f2, b2, b2n) if two_pop else (f2, b2)
        P_used = P
        if ctrl is None:
            y_new = advance(y, P, dt)
            commit_diag()
        else:
            # The EXPLICIT update is taken first in every mode: "self_consistent" discards it,
            # "auto" uses the populations it reaches to PROJECT this step's ASE gain integral,
            # which is the predictive half of the switch criterion (a cold start looks perfectly
            # healthy right up to the step that inverts the fiber).
            y_exp = None
            take_sc = ase_mode == "self_consistent"
            if not take_sc:
                y_exp = advance(y, P, dt)
                gi_proj = 0.0
                probed = None
                if ase_any.size:
                    g_proj, s_proj = gs_y(y_exp)
                    gi_proj = float(np.max(np.sum(
                        0.5 * (g_proj[ase_any, 1:] + g_proj[ase_any, :-1]) * dz, axis=1)))
                    if march_ase.needs_ase_probe(step_gain_integral, gi_proj):
                        ctrl.n_probes += 1
                        probed = _ase_share(_propagate_fixed(z, g_proj, s_proj, bc, u),
                                            ase_fwd, ase_bwd, step_launched)
                if march_ase.auto_switch(step_ase_ratio, step_gain_integral, gi_proj,
                                         step_nonfinite, _VALIDITY_LIMITS, ctrl.switch_fraction,
                                         ctrl.step_error, probed):
                    ctrl.arm()
                    take_sc = True
                else:
                    take_sc = ctrl.sticky()
            if take_sc:
                P_used, rep = march_ase.solve_self_consistent_step(
                    lambda gg, ss: _propagate_fixed(z, gg, ss, bc, u),
                    lambda Pq: advance(y, Pq, dt), gs_y, fwd_idx, bwd_idx, ctrl, P)
                ctrl.record(rep)
                step_mode[it] = 1
                switch_steps.append(it)
                y_new = advance(y, P_used, dt)
            else:
                y_new = y_exp
                ctrl.seed = P            # keep the warm start fresh (see simulate_transient)
            commit_diag()
        f2 = y_new[0]
        b2 = y_new[1]
        if two_pop:
            b2n = y_new[2]
        if ase_step_residual:
            # STEP-BY-STEP CLOSURE (opt-in, any mode): do the powers this step used survive at
            # the populations the step ENDED at? See simulate_transient for the argument.
            g_chk, s_chk = gs_y(y_new)
            P_chk = _propagate_fixed(z, g_chk, s_chk, bc, u)
            o_chk, pr_chk = march_ase.residual_views(P_chk, fwd_idx, bwd_idx)
            o_use, pr_use = march_ase.residual_views(P_used, fwd_idx, bwd_idx)
            _e_r, _p_r = _relaxation_residuals(o_chk, o_use, pr_chk, pr_use)
            worst_step_resid = (max(worst_step_resid, _p_r) if np.isfinite(_p_r)
                                else float("inf"))

    reasons = []
    if nonfinite:
        reasons.append("non-finite channel powers or populations appeared during the march")
    if worst_ase_ratio > _ASE_TO_LAUNCHED_LIMIT:
        reasons.append("frozen-step ASE power reached {:.3g}x the LAUNCHED optical power (limit "
                       "{:g}x)".format(worst_ase_ratio, _ASE_TO_LAUNCHED_LIMIT))
    if worst_gain_integral > _GAIN_INTEGRAL_LIMIT:
        reasons.append("frozen-step ASE gain integral reached INT g dz = {:.4g} (limit {:g}, i.e."
                       " a single-pass ASE gain of e^{:g})".format(
                           worst_gain_integral, _GAIN_INTEGRAL_LIMIT, _GAIN_INTEGRAL_LIMIT))
    warn_msg = None
    # Only the quasi-static mode treats the audit-A-7 margins as a reason to distrust the result;
    # the other two step through that regime on purpose (see simulate_transient).
    if ase_mode == "quasi_static":
        march_valid = not reasons
    else:
        sc_reasons = _self_consistent_reasons(ctrl, nonfinite)
        march_valid = not sc_reasons
        if sc_reasons:
            warnings.warn(
                "simulate_transient(ase_mode=%r): the self-consistent ASE step did not close on "
                "this co-doped amplifier -- " % ase_mode + "; ".join(sc_reasons) + ". Raise "
                "ase_max_iter, loosen ase_tol, or shorten the time step; meta['march_valid'] is "
                "False and the returned arrays must not be trusted.", RuntimeWarning,
                stacklevel=2)
    if reasons and ase_mode == "quasi_static":
        warn_msg = (
            "simulate_transient: the quasi-static (frozen-population) step is OUT OF ITS VALID "
            "REGIME for this co-doped amplifier -- " + "; ".join(reasons) + ". The step propagates "
            "exp(INT g dz) at FIXED (f2, b2), so ASE generated inside a step does not deplete the "
            "inversion that made it; once the ASE stops being a perturbation the march converges "
            "somewhere other than amp.solve()'s fixed point. These results are NOT trustworthy: "
            "use amp.solve() for the steady operating point, raise the signal power, or "
            "narrow/disable the ASE bands. Sub-stepping does not help -- the frozen-population "
            "propagation is already exact. See TransientResult.meta['quasi_static_valid'] "
            "(audit A-7, co-doped extension).")
        warnings.warn(warn_msg, RuntimeWarning, stacklevel=2)

    meta = _ase_mode_meta(ase_mode, ctrl, step_mode, switch_steps, t_grid, worst_step_resid,
                          ase_step_residual)
    meta.update({
            "n_signal": len(sig_idx), "n_pump": len(pmp_idx),
            "quasi_static_valid": not reasons,
            "march_valid": bool(march_valid),
            "max_ase_to_launched": float(worst_ase_ratio),
            "max_ase_gain_integral": float(worst_gain_integral),
            "nonfinite_powers": bool(nonfinite),
            "validity_limits": {"ase_to_launched": _ASE_TO_LAUNCHED_LIMIT,
                                "ase_gain_integral": _GAIN_INTEGRAL_LIMIT},
            "validity_warning": warn_msg,
            "m_modes": int(m_modes), "mcc": None,
            # ---- co-doped state and provenance ----
            "beta_yb": b2_zt,                       # (Nt, Nz) the Yb inversion history
            # the two pools separately (None unless an uncoupled pool exists); beta_yb above is
            # their population-weighted mean, i.e. what the optical field sees
            "beta_yb_coupled": b2c_zt, "beta_yb_uncoupled": b2n_zt,
            "yb_coupled_fraction": fc, "k_tr2_m3_s": amp._k_tr2,
            "yb_migration_rate_per_s": amp._w_mig,
            "beta_yb_seed_residual": float(b2_seed_shift),
            "max_dt_times_rate": float(worst_dt_rate),
            "max_population_overshoot": float(worst_overshoot),
            "integrator": "exponential-rosenbrock-2x2",
            "n_er_m3": amp._n_er, "n_yb_m3": amp._n_yb,
            "k_tr_m3_s": amp._k_tr, "k_back_m3_s": amp._k_back, "a32_per_s": amp._a32,
            # the per-ion cross-sections frame_as_steady needs (ChannelPlan.channels is None for a
            # co-doped plan -- there is no single ChannelSet when every channel carries two ions)
            "sigma_a": pl["sa_er"].copy(), "sigma_e": pl["se_er"].copy(),
            "sigma_a_er": pl["sa_er"].copy(), "sigma_e_er": pl["se_er"].copy(),
            "sigma_a_yb": pl["sa_yb"].copy(), "sigma_e_yb": pl["se_yb"].copy()})
    return TransientResult(t_grid, z, f2_zt, sig_out, pmp_out, gain_dB, list(kind), meta=meta,
                           ase_fwd_W=ase_f_zt, ase_bwd_W=ase_b_zt, ase_lambda_m=ase_lam,
                           ase_dnu_hz=ase_dnu, plan=amp.channel_plan(), power_zt=prof_zt)


def amplifier_saturation_energy(amp, lambda_m: float) -> float:
    """Frantz-Nodvik saturation energy [J] of whichever amplifier object is handed in, at
    lambda_m. The single-ion classes carry their ion as `.ion`; the co-doped ErYbAmplifier carries
    two, and the one that saturates at a C-band signal is the ERBIUM ion (`.er_ion`) -- Yb has no
    1550 nm cross-section to speak of, so E_sat there is the plain Er number on the same fiber.
    A thin adapter over saturation_energy so a study that swaps amplifier classes does not have to
    branch on the attribute name."""
    ion = getattr(amp, "ion", None)
    if ion is None:
        ion = getattr(amp, "er_ion", None)
    if ion is None:
        raise TypeError("amplifier_saturation_energy: %s carries neither .ion nor .er_ion"
                        % type(amp).__name__)
    return saturation_energy(ion, amp.fiber, lambda_m)


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
