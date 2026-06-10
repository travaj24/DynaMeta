"""REL5: optical damage -- laser-induced damage threshold (LIDT) scaling + absorbed-power thermal
runaway. A free-space ENZ modulator absorbs hardest exactly where it works (the ENZ peak), so the
maximum usable optical intensity is a reliability limit, and the Im(eps(T)) feedback (damping rises
with T through the Matthiessen phonon channel) makes the CW limit a genuine BIFURCATION the steady
balance cannot see past.

Pulsed (thermal-diffusion regime, ~ns..ms pulses):
    F_th(tau) = F_ref * sqrt(tau / tau_ref)                (Stuart 1996 fluence scaling; the fs-ps
                                                            regime departs from sqrt -- out of scope)

CW lumped runaway (0-D thermal node):
    C_th dT/dt = absorbed(T) * I * S  -  (T - T_sink) / R_th
with absorbed(T) the stack's absorbed FRACTION (a caller-supplied callable -- e.g. rebuilt per-T from
the ACTUAL layered TMM with the ITO Drude damping at MatthiessenGamma(T_K=T)), I the irradiance
[W/m^2], S the illuminated area [m^2], R_th [K/W], C_th [J/K]. The steady state solves
absorbed(T) I S = (T - T_sink)/R_th; when the absorption-vs-T slope outruns the heat-loss slope the
balance loses its root -> THERMAL RUNAWAY. cw_critical_intensity() bisects for that onset.

DRIVER NOTE: a per-region absorbed-power MAP from the optics result is the roadmap-flagged follow-on
driver; this lumped model needs only the total absorbed fraction (already an OpticalResult output)
or any caller-supplied absorbed(T). Pure numpy/scipy; oracles in validation/reliability_lidt.py
(linear-absorption runaway has the EXACT closed form I_crit = 1/(a1 S R_th)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def lidt_fluence_J_m2(tau_pulse_s, *, F_ref_J_m2: float, tau_ref_s: float):
    """Damage-threshold fluence at pulse length tau: F_th = F_ref sqrt(tau/tau_ref) (thermal-diffusion
    regime). Broadcasts over tau arrays."""
    if not (F_ref_J_m2 > 0.0 and tau_ref_s > 0.0):
        raise ValueError("LIDT: F_ref and tau_ref must be > 0")
    tau = np.asarray(tau_pulse_s, dtype=np.float64)
    if np.any(tau <= 0.0):
        raise ValueError("LIDT: tau_pulse_s must be > 0")
    return F_ref_J_m2 * np.sqrt(tau / tau_ref_s)


@dataclass(frozen=True)
class ThermalNode:
    """The 0-D thermal lumping of the illuminated stack: R_th [K/W] to the sink, C_th [J/K] heat
    capacity, S [m^2] illuminated area, T_sink [K]."""
    R_th_K_W: float
    C_th_J_K: float
    area_m2: float
    T_sink_K: float = 300.0

    def __post_init__(self):
        if not (self.R_th_K_W > 0.0 and self.C_th_J_K > 0.0 and self.area_m2 > 0.0
                and self.T_sink_K > 0.0):
            raise ValueError("ThermalNode: R_th, C_th, area, T_sink must all be > 0")


def cw_steady_temperature_K(absorbed_of_T, I_W_m2: float, node: ThermalNode, *,
                            T_max_K: float = 2000.0) -> float:
    """The STABLE steady temperature under CW irradiance I, or raises RuntimeError on thermal
    runaway (no balance root with negative net-power slope below T_max). absorbed_of_T(T) -> the
    absorbed fraction in [0, 1]. I = 0 -> T_sink exactly."""
    if I_W_m2 < 0.0:
        raise ValueError("LIDT: I_W_m2 must be >= 0")
    if I_W_m2 == 0.0:
        return float(node.T_sink_K)                        # byte-identical no-drive limit
    from scipy.optimize import brentq
    net = lambda T: float(absorbed_of_T(T)) * I_W_m2 * node.area_m2 \
        - (T - node.T_sink_K) / node.R_th_K_W              # > 0 heats, < 0 cools
    Ts = np.linspace(node.T_sink_K, T_max_K, 1201)
    vals = np.array([net(T) for T in Ts])
    # the FIRST sign change + -> - above T_sink is the stable root (net falls through zero)
    for i in range(len(Ts) - 1):
        if vals[i] > 0.0 and vals[i + 1] <= 0.0:
            return float(brentq(net, Ts[i], Ts[i + 1], rtol=1e-12))
    if vals[0] <= 0.0:                                     # absorbed so little it never heats a step
        return float(node.T_sink_K)
    raise RuntimeError("LIDT: thermal RUNAWAY -- absorbed power exceeds heat loss everywhere up to "
                       "T_max = {:.0f} K at I = {:.3e} W/m^2 (no stable steady state).".format(
                           T_max_K, I_W_m2))


def cw_critical_intensity_W_m2(absorbed_of_T, node: ThermalNode, *, I_lo: float = 1.0,
                               I_hi: float = 1.0e14, T_max_K: float = 2000.0,
                               rel_tol: float = 1e-6) -> float:
    """The CW runaway threshold: the largest irradiance with a stable steady state (bisection on the
    existence of the balance root). Raises if even I_lo runs away or I_hi is still stable."""
    def stable(I):
        try:
            cw_steady_temperature_K(absorbed_of_T, I, node, T_max_K=T_max_K)
            return True
        except RuntimeError:
            return False
    if not stable(I_lo):
        raise ValueError("LIDT: runaway already at I_lo = {:.3e} W/m^2".format(I_lo))
    if stable(I_hi):
        raise ValueError("LIDT: still stable at I_hi = {:.3e} W/m^2 -- no runaway in range "
                         "(raise I_hi or the absorption has no positive T-feedback)".format(I_hi))
    lo, hi = I_lo, I_hi
    while (hi - lo) / hi > rel_tol:
        mid = np.sqrt(lo * hi)                             # log bisection
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def cw_transient_K(times_s, absorbed_of_T, I_W_m2: float, node: ThermalNode, *,
                   T0_K: float = None):
    """Integrate the lumped node C_th dT/dt = absorbed(T) I S - (T - T_sink)/R_th. Returns T(t).
    I = 0 with T0 = T_sink -> T(t) == T_sink (no drive)."""
    from scipy.integrate import solve_ivp
    t = np.asarray(times_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("LIDT: times_s must be 1D strictly increasing with >= 2 samples")
    T0 = float(node.T_sink_K if T0_K is None else T0_K)
    rhs = lambda tt, y: [(float(absorbed_of_T(y[0])) * I_W_m2 * node.area_m2
                          - (y[0] - node.T_sink_K) / node.R_th_K_W) / node.C_th_J_K]
    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), [T0], t_eval=t, method="LSODA",
                    rtol=1e-9, atol=1e-9, max_step=float(np.min(np.diff(t))) * 50.0)
    if not sol.success:
        raise RuntimeError("LIDT transient: ODE integration failed ({})".format(sol.message))
    return sol.y[0]


def stack_absorbed_of_T(build_stack_at_T, lambda_m: float):
    """Helper: absorbed_of_T(T) from the ACTUAL coherent TMM -- build_stack_at_T(T) returns the
    LayeredStack at temperature T (e.g. with the ITO Drude damping at MatthiessenGamma(T_K=T) via
    dataclasses.replace), and the absorbed fraction is A = 1 - R - T from layered_rta."""
    from dynameta.optics.tmm_reference import layered_rta

    def absorbed(T_K: float) -> float:
        R, Tr, A = layered_rta(build_stack_at_T(float(T_K)), lambda_m)
        return float(A)
    return absorbed
