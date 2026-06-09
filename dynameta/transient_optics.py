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

from typing import Callable, Optional

import numpy as np

from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.optics.tmm_reference import layered_rta


def rc_accumulation(times_s, n_off: float, n_on: float, tau_s: float):
    """The gate-capacitor large-signal carrier-accumulation model: a HOMOGENEOUS ITO density that charges
    n(t) = n_off + (n_on - n_off)(1 - exp(-t/tau)) toward the ON level with the access-RC time constant tau
    (an exponential turn-on; for turn-off swap n_off<->n_on). Returns n(t) (same length as times_s)."""
    t = np.asarray(times_s, dtype=float)
    return float(n_off) + (float(n_on) - float(n_off)) * (1.0 - np.exp(-t / float(tau_s)))


def enz_reflector_stack(eps_ito, lambda_m, *, t_ito_m: float = 10e-9, eps_oxide: float = 9.0,
                        t_oxide_m: float = 120e-9, eps_mirror: complex = -120.0 + 3.0j):
    """A default reflective gated-ITO modulator stack for the transient map: air | ITO(eps_ito) | oxide |
    mirror. eps_ito is the (graded or scalar) ITO permittivity at this instant. If eps_ito is an array it is
    split into equal ITO sublayers (the depth-resolved ENZ profile); a scalar is one homogeneous film."""
    eps_ito = np.atleast_1d(np.asarray(eps_ito, dtype=complex))
    nsl = eps_ito.size
    slabs = [LayeredSlab(t_ito_m / nsl, eps=complex(e)) for e in eps_ito]
    slabs.append(LayeredSlab(t_oxide_m, eps=complex(eps_oxide)))
    return LayeredStack(1.0 + 0j, np.sqrt(complex(eps_mirror)), slabs)


def optical_transient_response(times_s, n_of_t: Callable, lambda_m: float, *, drude_model=None,
                               drude_of_t: Optional[Callable] = None,
                               build_stack: Optional[Callable] = None):
    """R(t)/T(t) of the modulator as the carriers evolve. `n_of_t(t)` returns the ITO carrier density n(z)
    [m^-3] (depth array or scalar) at time t; the free-carrier Drude maps it to eps(z) at lambda_m;
    `build_stack(eps_ito, lambda_m)` (default enz_reflector_stack) assembles the LayeredStack, solved by
    coherent TMM. Returns (t_s, R, T, eps_front) -- the optical turn-on/off waveform plus the front-ITO
    permittivity trajectory (its ENZ crossing).

    Supply EXACTLY ONE of `drude_model` (a fixed DrudeOptical, the original behavior -- byte-identical) or
    `drude_of_t(t) -> DrudeOptical` (a per-instant Drude, e.g. carrier-heating Te(t)-dependent m*/Gamma from
    carriers.carrier_heating). The drude_of_t hook lets the SAME loop carry a time-varying material response
    without widening DrudeOptical.eps; drude_of_t=None preserves the fixed-drude_model path exactly."""
    if (drude_model is None) == (drude_of_t is None):
        raise ValueError("supply exactly one of drude_model (fixed) or drude_of_t (per-instant)")
    build_stack = build_stack or enz_reflector_stack
    t = np.asarray(times_s, dtype=float)
    R = np.empty(t.size); T = np.empty(t.size); eps_front = np.empty(t.size, dtype=complex)
    for i, ti in enumerate(t):
        dm = drude_of_t(float(ti)) if drude_of_t is not None else drude_model
        nz = np.atleast_1d(np.asarray(n_of_t(float(ti)), dtype=float))
        eps_ito = np.atleast_1d(np.asarray(dm.eps(lambda_m, n_m3=nz), dtype=complex))  # vectorized
        eps_front[i] = eps_ito[0]
        stack = build_stack(eps_ito, lambda_m)
        R[i], T[i], _A = layered_rta(stack, lambda_m)
    return t, R, T, eps_front
