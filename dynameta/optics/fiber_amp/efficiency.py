"""Electrical (wall-plug) efficiency of a rare-earth fiber amplifier -- the ELECTRICAL-domain layer
above metrics.py's optical-optical accounting (audit 2026-08-04 F-7).

Everything else in this package stops at optical-optical: power_conversion_efficiency (added signal
per launched pump), slope_efficiency (dP_sig/dP_pump), stokes_limit (the quantum-defect ceiling) and
thermal.total_heat_W (the dissipation). None of them knows what the pump cost to make. For anyone
sizing a real amplifier that is the number that matters, and it is not a small correction: a 45%-
efficient pump diode throws away more of the input than the quantum defect does.

WHAT THIS MODULE IS. A bookkeeping layer, not new physics. It takes a solved amplifier plus an
electrical characterization of its pump sources and reports the whole chain

    electrical in -> pump diode (WPE) -> fibre coupling -> absorbed in the core
                  -> quantum defect (Stokes ceiling) -> extracted into the signal

so that the BINDING loss is visible rather than buried in one aggregate number. Every optical
quantity is read from the solve; nothing is re-derived.

THREE DEFINITIONAL CHOICES, made explicitly because each is a place a budget can be quietly
flattered:

  1. The numerator is the signal power ADDED, P_sig_out - P_sig_in, not P_sig_out. This matches
     metrics.power_conversion_efficiency and stops a strong input signal from being counted as
     amplifier output. `gross` on the result carries the other convention if you want it.
  2. Residual (unabsorbed) pump stays in the denominator. It was paid for electrically. An
     absorbed-pump efficiency is a useful diagnostic (`eta_pump_absorption`) but is not a wall-plug
     number.
  3. Duty cycle is the caller's to supply. A CW-pumped amplifier serving a bursty load pays for the
     pump continuously while delivering signal only during bursts, so the honest average depends on
     traffic this module cannot see. duty_cycle=1.0 (the default) is the CW case.

CROSS-CHECK, and WHY IT IS NOT total_heat_W. Everything launched must leave as light or as heat.
Written with `thermal.total_heat_W` as the heat that statement is VACUOUS: total_heat_W is defined
as F(0) - F(L), which expands to exactly (launched - exiting) over the same two z-planes, so the
residual is X - X and stays at machine zero on a corrupted solve (measured: perturbing the signal
output endpoint by 1% left the residual at 5.6e-17 W). Integrating the local heat density instead
does not help -- `np.gradient` followed by a trapezoid telescopes back to F(0) - F(L) exactly.

So `energy_balance_residual_W` closes the balance against the DISSIPATION COMPUTED FROM THE LOCAL
RATE BALANCE: absorbed minus stimulated-emitted minus spontaneously-seeded power, evaluated from
the solve's own inversion profile and cached cross-sections and integrated over z. That is a
different computation from the endpoint flux difference -- it asks whether the returned P(z) is
actually a solution of the amplifier's own ODEs -- so it CAN fail, and it does: the same 1% endpoint
perturbation moves it to -1.7e-3 W, a factor 1476. On a healthy solve what is left is trapezoid
discretization, which falls as O(dz^2): measured 1.5e-5 / 3.9e-6 / 6.0e-7 / 1.5e-7 of the launched
power at n_nodes = 81 / 161 / 401 / 801 on the reference EDFA.

Pure numpy; SI units (W, m, s, J). ASCII only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from dynameta.core.numerics import trapz          # audit X-1: floor-safe (np.trapezoid needs >=2.0)
from dynameta.optics.fiber_amp.metrics import (effective_pump_lambda_m, power_conversion_efficiency,
                                               stokes_limit)
from dynameta.optics.fiber_amp.steady_state import FiberAmplifier, SteadyStateResult
from dynameta.optics.fiber_amp.thermal import total_heat_W

__all__ = ["PumpSource", "WallPlugBudget", "wall_plug_efficiency", "energy_per_bit_J"]

# Relative closure the power balance must reach before wall_plug_efficiency stops complaining.
# The residual is trapezoid discretization on the rate integral (see the module docstring), so it
# is set by n_nodes: 1.5e-5 of the launched power at n_nodes = 81, the coarsest mesh anything in
# this repo solves on. 1e-3 leaves ~65x headroom there and still catches the 5.7e-3 a 1%-corrupted
# endpoint produces -- which the previous 2e-2 threshold did not.
_BALANCE_NOTE_FRAC = 1e-3


def _dissipated_power_W(amp: FiberAmplifier, result: SteadyStateResult) -> float:
    """Total dissipated optical power [W] from the LOCAL RATE BALANCE, independent of the endpoint
    flux difference `thermal.total_heat_W` reports (module docstring on why that distinction is the
    whole point of this function).

    The net forward flux is F(z) = sum_fwd P - sum_bwd P, so dF/dz = sum_k (g_k P_k + s_k) with
    g_k the local net gain coefficient and s_k the spontaneous source -- the amplifier's OWN
    right-hand side, evaluated on the RETURNED profile rather than accumulated along it. The heat
    is -INT dF/dz dz.

    Returns NaN for an amplifier that does not expose the mean-field RHS (the co-doped and
    radially-resolved classes carry their own, with different signatures), so the caller reports an
    unavailable closure rather than a wrong one."""
    if not hasattr(amp, "_dP_full_c"):
        return float("nan")
    ch, _bc, u, _is_ase, _kind = amp._plan()
    P, z = result.power_W, result.z_m
    if P.shape[0] != ch.lambda_m.size or P.shape[1] != z.size:
        return float("nan")             # the result did not come from this amplifier's channels
    amp._tau_s = ch.tau_s               # the back-compat single-call forms set this the same way
    c = amp._coeffs(ch)
    mcc = result.meta.get("mcc")
    dF = np.array([float(np.sum(u * amp._dP_full_c(c, u, P[:, j],
                                                   None if mcc is None else mcc[:, j])))
                   for j in range(z.size)])
    return -float(trapz(dF, z))


@dataclass(frozen=True)
class PumpSource:
    """Electrical characterization of ONE pump source.

    wallplug_efficiency  optical W out of the diode facet per electrical W in. Representative
                         values: 0.45-0.55 for 9xx nm telecom/multimode diodes, lower (~0.30) for
                         a 1480 nm InGaAsP module -- which is why the pump-wavelength choice is not
                         settled by the quantum defect alone.
    coupling_efficiency  everything between the diode facet and the doped core (combiner/WDM
                         insertion loss, splices). The amplifier's Pump.power_W is what is
                         LAUNCHED, so the diode must emit power_W / coupling_efficiency.
    """
    wallplug_efficiency: float
    coupling_efficiency: float = 1.0

    def __post_init__(self):
        if not (0.0 < self.wallplug_efficiency <= 1.0):
            raise ValueError("PumpSource: wallplug_efficiency must be in (0, 1] (got {!r})"
                             .format(self.wallplug_efficiency))
        if not (0.0 < self.coupling_efficiency <= 1.0):
            raise ValueError("PumpSource: coupling_efficiency must be in (0, 1] (got {!r})"
                             .format(self.coupling_efficiency))

    def electrical_W(self, launched_W: float) -> float:
        """Electrical drive needed to LAUNCH launched_W into the fiber."""
        return float(launched_W) / (self.coupling_efficiency * self.wallplug_efficiency)


@dataclass
class WallPlugBudget:
    """Electrical-to-useful-optical accounting. Powers [W], efficiencies dimensionless."""
    p_signal_in_W: float
    p_signal_out_W: float
    p_signal_added_W: float
    p_pump_launched_W: float
    p_pump_residual_W: float
    p_pump_absorbed_W: float
    p_electrical_pumps_W: float
    p_electrical_aux_W: float
    p_electrical_total_W: float
    heat_W: float
    # launched - exiting - (dissipation from the LOCAL RATE BALANCE). NOT closed against
    # thermal.total_heat_W, which would make it identically zero; see the module docstring.
    energy_balance_residual_W: float
    eta_optical_pce: float          # added signal / launched pump  (== metrics.power_conversion_*)
    eta_stokes_limit: float         # quantum-defect ceiling at the photon-weighted mean pump
    eta_wallplug: float             # added signal / total electrical
    eta_wallplug_gross: float       # OUTPUT signal / total electrical
    duty_cycle: float
    chain: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["wall-plug budget: eta = %.4f (%.2f%%), gross %.4f"
                 % (self.eta_wallplug, 100.0 * self.eta_wallplug, self.eta_wallplug_gross),
                 "  optical PCE %.4f of a %.4f Stokes ceiling"
                 % (self.eta_optical_pce, self.eta_stokes_limit),
                 "  electrical: pumps %.4g W + aux %.4g W = %.4g W"
                 % (self.p_electrical_pumps_W, self.p_electrical_aux_W, self.p_electrical_total_W),
                 "  chain: " + ", ".join("%s %.4f" % (k, v) for k, v in self.chain.items())]
        lines += ["  ! " + n for n in self.notes]
        return "\n".join(lines)


def wall_plug_efficiency(amp: FiberAmplifier, result: SteadyStateResult,
                         sources: Sequence[PumpSource], *,
                         aux_electrical_W: float = 0.0,
                         tec_fraction_of_pump_heat: float = 0.0,
                         duty_cycle: float = 1.0,
                         signal_index: Optional[int] = None) -> WallPlugBudget:
    """Wall-plug efficiency of a solved amplifier (audit F-7).

    sources    one PumpSource per amp.pumps, in the same order (or a single PumpSource to apply to
               all of them).
    aux_electrical_W    pump controller, monitor photodiodes, MCU -- electrical power that is paid
               regardless of pump level. This is not a rounding term: because it is FIXED, it moves
               the optimal pump power. Measured on a 300 mW-pumped C-band EDFA, a 0.5 W controller
               shifts the wall-plug-optimal launched pump from 0.20 W to 0.80 W.
    tec_fraction_of_pump_heat   TEC electrical power as a fraction of the electrical power the
               diodes dissipate as heat. 0 = uncooled.
    duty_cycle in (0, 1]: the fraction of time useful signal is delivered while the pump runs
               continuously. Scales the delivered optical power, not the electrical cost. See the
               module docstring -- this is the caller's call, not something the solve knows.

    Returns a WallPlugBudget whose `chain` gives each stage's efficiency so the binding loss is
    visible, and whose `energy_balance_residual_W` is an independent closure on the solve: the
    launched power minus the exiting light minus the dissipation recomputed from the amplifier's
    own rate equations (NOT from thermal.total_heat_W, against which the balance would be an
    identity -- module docstring).
    """
    pumps = list(amp.pumps)
    if not pumps:
        raise ValueError("wall_plug_efficiency: the amplifier has no pumps")
    src = list(sources) if isinstance(sources, (list, tuple)) else [sources]
    if len(src) == 1 and len(pumps) > 1:
        src = src * len(pumps)
    if len(src) != len(pumps):
        raise ValueError("wall_plug_efficiency: got %d PumpSource(s) for %d pump(s); pass one per "
                         "pump (in order) or a single source to apply to all"
                         % (len(src), len(pumps)))
    if not (0.0 < duty_cycle <= 1.0):
        raise ValueError("wall_plug_efficiency: duty_cycle must be in (0, 1] (got %r)"
                         % (duty_cycle,))
    if tec_fraction_of_pump_heat < 0.0:
        raise ValueError("wall_plug_efficiency: tec_fraction_of_pump_heat must be >= 0")

    kinds = list(result.kind)
    sig_all = [i for i, k in enumerate(kinds) if k == "signal"]
    if not sig_all:
        raise ValueError("wall_plug_efficiency: the result carries no signal channel")
    sig = sig_all if signal_index is None else [sig_all[int(signal_index)]]
    p_sig_out = float(np.sum(result.power_W[sig, -1]))
    p_sig_in = float(np.sum(result.power_W[sig, 0]))

    # Residual pump measured at each pump's OWN output end (forward at z=L, backward at z=0).
    pmp = [i for i, k in enumerate(kinds) if k == "pump"]
    p_pump_launched = float(sum(float(p.power_W) for p in pumps))
    p_pump_residual = float(sum(result.power_W[i, -1] if result.u[i] > 0 else result.power_W[i, 0]
                                for i in pmp))
    p_pump_residual = max(p_pump_residual, 0.0)     # a fully absorbed pump lands in integrator noise
    p_pump_absorbed = p_pump_launched - p_pump_residual

    pump_elec = float(sum(s.electrical_W(p.power_W) for s, p in zip(src, pumps)))
    pump_heat = max(pump_elec - p_pump_launched, 0.0)
    aux = float(aux_electrical_W) + float(tec_fraction_of_pump_heat) * pump_heat
    total_elec = pump_elec + aux

    added = (p_sig_out - p_sig_in) * duty_cycle
    gross = p_sig_out * duty_cycle
    lam_p = effective_pump_lambda_m(pumps)
    lam_s = float(result.lambda_m[sig[0]])
    stokes = stokes_limit(lam_p, lam_s)

    try:
        heat = float(total_heat_W(result))
    except Exception:                                       # pragma: no cover - defensive
        heat = float("nan")
    # Independent closure: every watt launched must exit as light or become heat -- with the heat
    # taken from the LOCAL RATE BALANCE, not from total_heat_W, which would make the statement an
    # identity (module docstring). launched_total counts EVERY signal, whatever signal_index
    # selected for the efficiency numerator: this is an energy balance over the whole fiber.
    launched_total = p_pump_launched + float(np.sum(result.power_W[sig_all, 0]))
    exiting = float(np.sum([result.power_W[i, -1] if result.u[i] > 0 else result.power_W[i, 0]
                            for i in range(result.power_W.shape[0])]))
    heat_rate = _dissipated_power_W(amp, result)
    residual = launched_total - exiting - heat_rate

    eta_abs = p_pump_absorbed / p_pump_launched if p_pump_launched > 0.0 else float("nan")
    chain = {
        "eta_diode": float(np.mean([s.wallplug_efficiency for s in src])),
        "eta_coupling": float(np.mean([s.coupling_efficiency for s in src])),
        "eta_pump_absorption": eta_abs,
        "eta_stokes": stokes,
        # what fraction of the quantum-defect-limited absorbed pump actually became signal, i.e.
        # how much of the available inversion the signal won against ASE and spontaneous decay
        "eta_extraction": ((p_sig_out - p_sig_in) / (p_pump_absorbed * stokes)
                           if p_pump_absorbed * stokes > 0.0 else float("nan")),
        "eta_duty": float(duty_cycle),
    }
    notes: List[str] = []
    if total_elec <= 0.0:
        # A NaN with no explanation is the failure mode this module's own audit finding complained
        # about; say why rather than just returning it.
        notes.append("total electrical power is zero (every pump is at 0 W), so every efficiency "
                     "below is NaN by definition rather than by failure")
    if not bool(result.meta.get("converged", True)):
        notes.append("the solve did NOT report convergence; check meta['endpoint_residual'] -- for "
                     "a counter-pumped amplifier a non-converged solve can be wrong by tens of dB "
                     "(try solve(relax=0.2))")
    if chain["eta_extraction"] > 1.0 + 1e-9:
        notes.append("extraction exceeds the quantum-defect ceiling (%.4f) -- the solve is not "
                     "physical, or duty_cycle/signal_index is mismatched"
                     % chain["eta_extraction"])
    if not np.isfinite(residual):
        notes.append("the independent optical power balance is unavailable for this amplifier "
                     "class (it needs the mean-field right-hand side), so "
                     "energy_balance_residual_W is NaN rather than zero")
    elif abs(residual) > _BALANCE_NOTE_FRAC * max(launched_total, 1e-30):
        notes.append("optical power balance closes to only %.3g W of %.3g W launched -- the "
                     "returned P(z) does not satisfy the amplifier's own rate equations to the "
                     "mesh's discretization error; check meta['converged'] and raise n_nodes"
                     % (residual, launched_total))

    return WallPlugBudget(
        p_signal_in_W=p_sig_in, p_signal_out_W=p_sig_out, p_signal_added_W=p_sig_out - p_sig_in,
        p_pump_launched_W=p_pump_launched, p_pump_residual_W=p_pump_residual,
        p_pump_absorbed_W=p_pump_absorbed, p_electrical_pumps_W=pump_elec,
        p_electrical_aux_W=aux, p_electrical_total_W=total_elec, heat_W=heat,
        energy_balance_residual_W=float(residual),
        eta_optical_pce=float(power_conversion_efficiency(amp, result)),
        eta_stokes_limit=stokes,
        eta_wallplug=added / total_elec if total_elec > 0.0 else float("nan"),
        eta_wallplug_gross=gross / total_elec if total_elec > 0.0 else float("nan"),
        duty_cycle=float(duty_cycle), chain=chain, notes=notes)


def energy_per_bit_J(budget: WallPlugBudget, bitrate_bit_s: float, *, ports: int = 1) -> float:
    """Electrical energy per delivered bit [J/bit] = P_electrical_total / (bitrate * duty).

    ports > 1 divides by the port count, which is the right figure for a BROADCAST fan-out (one
    amplifier feeding N receivers with the same data): the delivered bit rate is independent of N,
    so J/bit is the honest number for the LINK while J/bit/port is the honest number for the
    ARCHITECTURE -- it is what N separately amplified links have to beat.
    """
    if not (bitrate_bit_s > 0.0):
        raise ValueError("energy_per_bit_J: bitrate_bit_s must be > 0")
    if int(ports) < 1:
        raise ValueError("energy_per_bit_J: ports must be >= 1")
    delivered = float(bitrate_bit_s) * budget.duty_cycle * int(ports)
    return budget.p_electrical_total_W / delivered
