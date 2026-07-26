"""Coupled carrier <-> optics TRANSIENT: map a time-dependent carrier-density trajectory n(z, t) -- the
gate-driven ITO accumulation as it charges/discharges -- through the free-carrier Drude eps(n) and a
layered optical solve to get the modulator's optical response R(t) / T(t). This closes the dynamic loop the
steady-state run_pipeline does not: a voltage pulse drives n(t) [carrier RC / drift-diffusion dynamics] ->
eps(t) [free-carrier Drude / ENZ] -> R(t) [optical transient], so you get the validated turn-on/turn-off
WAVEFORM and response time of the device, not just the DC OFF/ON contrast.

The carrier trajectory is supplied as a callback `n_of_t(t) -> n(z)` (an (Nz,) ITO depth profile in m^-3,
or a scalar for a homogeneous film), so it composes with EITHER an analytic large-signal model
(rc_accumulation, below -- the gate-capacitor RC charging) OR a real DEVSIM drift-diffusion transient
(extract the ITO n(z) at each recorded step and feed it in). The optical map is a LayeredStack built per
time step (build_stack), solved by coherent TMM (fast -- the modulator is a layered stack), so the whole
R(t) waveform is cheap once n(t) is known. Convention exp(-i w t), SI; Im(eps) > 0 = loss.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.optics.tmm_reference import layered_rta


def rc_accumulation(times_s, n_off: float, n_on: float, tau_s: float):
    """The gate-capacitor large-signal carrier-accumulation model: a HOMOGENEOUS ITO density that charges
    n(t) = n_off + (n_on - n_off)(1 - exp(-t/tau)) toward the ON level with the access-RC time constant tau
    (an exponential turn-on; for turn-off swap n_off<->n_on). Returns n(t) (same length as times_s).

    ``times_s`` must be >= 0 (audit R-13): the closed form is the step response of a circuit
    switched at t = 0, so a negative t returns n BELOW n_off (for t = -tau, 172% of the way past
    n_off in the WRONG direction) -- a plausible-looking curve with no physical meaning."""
    t = np.asarray(times_s, dtype=float)
    if np.any(t < 0.0):
        raise ValueError(
            "rc_accumulation: times_s must be >= 0 -- this is the t >= 0 step response of a gate "
            "switched at t = 0; a negative time extrapolates the exponential backwards and "
            "returns a density below n_off with no physical meaning.")
    return float(n_off) + (float(n_on) - float(n_off)) * (1.0 - np.exp(-t / float(tau_s)))


def enz_reflector_stack(eps_ito, lambda_m, *, t_ito_m: float = 10e-9, eps_oxide: float = 9.0,
                        t_oxide_m: float = 120e-9, eps_mirror: complex = -120.0 + 3.0j):
    """A default reflective gated-ITO modulator stack for the transient map: air | ITO(eps_ito) | oxide |
    mirror. eps_ito is the (graded or scalar) ITO permittivity at this instant. If eps_ito is an array it is
    split into equal ITO sublayers (the depth-resolved ENZ profile); a scalar is one homogeneous film.

    DEPTH-ORDERING CONTRACT (audit S2-3): eps_ito[0] is placed adjacent to AIR and eps_ito[-1]
    adjacent to the OXIDE (gate) -- i.e. the array runs AIR-SIDE-FIRST, top-down. A gated
    accumulation layer therefore belongs at eps_ito[-1]. This is the OPPOSITE of the repo's
    substrate-first depth-array convention (core.layered.slice_profile), so a DEVSIM-extracted
    ascending-z profile must be REVERSED before feeding it here; a flipped profile changes R(t)
    while R+T+A still closes to 1 (invisible-flip class)."""
    eps_ito = np.atleast_1d(np.asarray(eps_ito, dtype=complex))
    nsl = eps_ito.size
    slabs = [LayeredSlab(t_ito_m / nsl, eps=complex(e)) for e in eps_ito]
    slabs.append(LayeredSlab(t_oxide_m, eps=complex(eps_oxide)))
    return LayeredStack(1.0 + 0j, np.sqrt(complex(eps_mirror)), slabs)


def optical_transient_response(times_s, n_of_t: Callable, lambda_m: float, *, drude_model=None,
                               drude_of_t: Optional[Callable] = None,
                               build_stack: Optional[Callable] = None,
                               adiabatic_rt_factor: float = 2.0,
                               orientation: str = "air_first"):
    """R(t)/T(t) of the modulator as the carriers evolve. `n_of_t(t)` returns the ITO carrier density n(z)
    [m^-3] (depth array or scalar) at time t; the free-carrier Drude maps it to eps(z) at lambda_m;
    `build_stack(eps_ito, lambda_m)` (default enz_reflector_stack) assembles the LayeredStack, solved by
    coherent TMM. Returns (t_s, R, T, eps_front) -- the optical turn-on/off waveform plus the front-ITO
    permittivity trajectory (its ENZ crossing). Depth arrays are AIR-SIDE-FIRST (see
    enz_reflector_stack; reverse a substrate-first DEVSIM profile). NOTE on T (audit S2-16): with
    the DEFAULT lossy-mirror stack, T is the power flux absorbed INTO the semi-infinite mirror
    (~3%), not a device transmittance (true transmission through the opaque mirror is ~0); T is a
    genuine transmittance only for a caller-supplied lossless-substrate build_stack.

    Supply EXACTLY ONE of `drude_model` (a fixed DrudeOptical, the original behavior -- byte-identical) or
    `drude_of_t(t) -> DrudeOptical` (a per-instant Drude, e.g. carrier-heating Te(t)-dependent m*/Gamma from
    carriers.carrier_heating). The drude_of_t hook lets the SAME loop carry a time-varying material response
    without widening DrudeOptical.eps; drude_of_t=None preserves the fixed-drude_model path exactly.

    DEPTH ORIENTATION -- STATE IT (audit V-9). `enz_reflector_stack` documents that eps_ito runs
    AIR-SIDE-FIRST (eps_ito[0] adjacent to air, eps_ito[-1] adjacent to the gate oxide), which is
    the INVERSE of the repo's substrate-first depth convention (`core.layered.slice_profile`) --
    and a flipped profile changes R(t) while R + T + A still closes to 1, so nothing downstream
    can see the mistake. The hazard was fully diagnosed in that docstring and then not enforced:
    this function accepted any 1-D array with no orientation argument at all. Declare it:
    `orientation='air_first'` (the default, the incumbent behaviour, byte-identical) or
    `orientation='substrate_first'`, which REVERSES each n(z) sample before the eps map -- pass
    that for an ascending-z DEVSIM profile instead of reversing it by hand and hoping.

    QUASI-STATIC (ADIABATIC) VALIDITY -- READ THIS (audit R-13). Each sample is an INDEPENDENT
    steady-state TMM solve of the instantaneous eps(z): the stack is assumed to have fully
    re-established its standing-wave field before the material moves again. That holds only while
    the drive is slow against the stack's own optical round trip
    ``t_rt ~ 2 sum_j Re(n_j) d_j / c`` (a few fs for a ~100 nm modulator). Below it the real
    device filters and delays the response and this function does NOT -- it returns a smooth,
    plausible, WRONG R(t) with no error, which is why the condition is CHECKED rather than merely
    stated: a sample spacing under ``adiabatic_rt_factor`` round trips emits a
    ``RuntimeWarning``.

    The default ``adiabatic_rt_factor = 2.0`` flags only the regime where the approximation is
    definitely broken (the field cannot re-establish between samples at all), not the grey band
    above it -- deliberately, to keep the warning meaningful. Calibration: the shipped
    ``validation/carrier_heating_enz.py`` path samples at ~3.0 round trips, i.e. inside the grey
    band and silent at the default; the auditor's failing case (a 1 fs drive on a ~3 fs round
    trip, 0.33) warns loudly. Raise the factor (5-10) if you want the grey band flagged too, or
    set ``adiabatic_rt_factor=0`` to silence the check entirely. For a genuinely sub-round-trip
    drive use the FDTD path (optics.fdtd with a time-varying eps hook), which marches the field.
    """
    if (drude_model is None) == (drude_of_t is None):
        raise ValueError("supply exactly one of drude_model (fixed) or drude_of_t (per-instant)")
    if orientation not in ("air_first", "substrate_first"):          # audit V-9
        raise ValueError(
            "optical_transient_response: orientation must be 'air_first' (n(z)[0] adjacent to "
            "AIR -- enz_reflector_stack's contract, the default) or 'substrate_first' (an "
            "ascending-z DEVSIM profile, reversed here); got {!r}. A flipped profile changes "
            "R(t) while R + T + A still closes to 1, so it cannot be caught downstream.".format(
                orientation))
    flip = orientation == "substrate_first"
    build_stack = build_stack or enz_reflector_stack
    t = np.asarray(times_s, dtype=float)
    R = np.empty(t.size); T = np.empty(t.size); eps_front = np.empty(t.size, dtype=complex)
    for i, ti in enumerate(t):
        dm = drude_of_t(float(ti)) if drude_of_t is not None else drude_model
        nz = np.atleast_1d(np.asarray(n_of_t(float(ti)), dtype=float))
        if flip:
            nz = nz[::-1]                                   # substrate-first -> air-first (V-9)
        eps_ito = np.atleast_1d(np.asarray(dm.eps(lambda_m, n_m3=nz), dtype=complex))  # vectorized
        eps_front[i] = eps_ito[0]
        stack = build_stack(eps_ito, lambda_m)
        if i == 0 and float(adiabatic_rt_factor) > 0.0 and t.size > 1:
            _warn_if_not_adiabatic(stack, t, float(adiabatic_rt_factor))
        R[i], T[i], _A = layered_rta(stack, lambda_m)
    return t, R, T, eps_front


def _warn_if_not_adiabatic(stack, t, factor: float) -> None:
    """audit R-13: compare the drive's SAMPLE SPACING against the stack's optical round trip
    2 sum_j Re(n_j) d_j / c. The quasi-static map has no memory, so once the two are comparable
    the returned R(t) is a sequence of unrelated steady states, not a transient."""
    try:
        t_rt = 0.0
        for sl in getattr(stack, "slabs", []) or []:
            eps = getattr(sl, "eps", None)
            if eps is None:                                # patterned slab: no scalar index
                return
            t_rt += float(np.real(np.sqrt(complex(eps)))) * float(sl.thickness_m)
        t_rt = 2.0 * t_rt / C_LIGHT
    except Exception:                                      # pragma: no cover - defensive
        return
    if not (t_rt > 0.0):
        return
    dt_min = float(np.min(np.diff(t))) if t.size > 1 else float("inf")
    if dt_min < factor * t_rt:
        warnings.warn(
            "optical_transient_response: the sample spacing {:.3g} s is under {:g}x the stack's "
            "optical round trip {:.3g} s, so the QUASI-STATIC (adiabatic) approximation this "
            "function is built on no longer holds -- each sample is an independent steady-state "
            "TMM solve and the returned R(t) will look smooth and plausible while being wrong. "
            "Use the FDTD path (a time-varying eps hook) for sub-round-trip drives, or pass "
            "adiabatic_rt_factor=0 to silence this.".format(dt_min, factor, t_rt),
            RuntimeWarning, stacklevel=3)
