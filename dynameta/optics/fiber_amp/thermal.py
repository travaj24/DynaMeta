"""Cladding-pumped operation and the quantum-defect thermal load (docs sec.7). Double-clad
pumping itself is already in the propagation model (Pump.cladding uses the overlap
Gamma_p = A_core/A_clad from waveguide.cladding_pump_overlap); this module adds the heat side:

  * the local heat density Q(z) [W/m] deposited in the core, from the rigorous optical-power
    balance Q = -d/dz (net forward optical flux) -- the power that leaves the optical fields
    (quantum defect + background loss + reabsorbed ASE) becomes heat, while spontaneous light
    that escapes does not;
  * the quantum-defect fraction 1 - lambda_pump/lambda_signal, the floor on the heat fraction
    (5% for Yb 976->1030, 37% for Er 980->1560) -- the reason Yb double-clad fibers scale to
    kilowatts;
  * the steady-state RADIAL temperature profile of a uniformly core-heated fiber cooled by
    convection at the outer surface (Brown & Hoffman, IEEE JQE 37:207, 2001), and the
    centre-to-coolant temperature rise.

Pure numpy; SI units. docs/fiber_amp_model_spec.md sec.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.optics.fiber_amp.steady_state import SteadyStateResult

__all__ = ["ThermalModel", "quantum_defect_fraction", "net_forward_flux", "heat_load_per_m",
           "total_heat_W", "radial_temperature_rise", "peak_temperature_rise"]


def quantum_defect_fraction(pump_lambda_m: float, signal_lambda_m: float) -> float:
    """Stokes / quantum-defect heat fraction 1 - lambda_pump/lambda_signal: the minimum fraction
    of each absorbed pump photon's energy that must be dissipated as heat when it is converted to
    a longer-wavelength signal photon (0.052 for Yb 976->1030, 0.372 for Er 980->1560)."""
    return float(1.0 - pump_lambda_m / signal_lambda_m)


def net_forward_flux(result: SteadyStateResult) -> np.ndarray:
    """Net optical power crossing each z-plane toward +z: sum of forward-channel powers minus
    backward-channel powers (W). Its downhill gradient is the local heat deposition."""
    u = result.u
    return (np.sum(result.power_W[u > 0], axis=0) - np.sum(result.power_W[u < 0], axis=0))


def heat_load_per_m(result: SteadyStateResult) -> np.ndarray:
    """Local heat density Q(z) [W/m] = -d/dz(net forward optical flux). Positive where the fiber
    dissipates (pump absorption region); the integral is the total heat (total_heat_W). Captures
    quantum-defect + background-loss + reabsorbed-ASE heating; escaping spontaneous light is not
    counted as heat. Slightly conservative where tracked ASE bins miss out-of-band spontaneous
    emission."""
    return -np.gradient(net_forward_flux(result), result.z_m)


def total_heat_W(result: SteadyStateResult) -> float:
    """Total dissipated power [W] = flux in at both ends minus flux out = F(0) - F(L)."""
    F = net_forward_flux(result)
    return float(F[0] - F[-1])


@dataclass(frozen=True)
class ThermalModel:
    """Steady-state radial-conduction model of the fiber cross-section (Brown-Hoffman). All heat
    is generated uniformly in the core (radius a); it conducts out through core and (inner)
    cladding, then leaves by convection at the outer radius b. core_k / clad_k = thermal
    conductivities [W/m/K] (fused silica ~1.38); h_conv = convective coefficient at the outer
    surface [W/m^2/K]; T_coolant_K = ambient/coolant temperature."""
    core_k_W_mK: float = 1.38
    clad_k_W_mK: float = 1.38
    h_conv_W_m2K: float = 1000.0
    T_coolant_K: float = 300.0

    def __post_init__(self):
        for nm, v in (("core_k_W_mK", self.core_k_W_mK), ("clad_k_W_mK", self.clad_k_W_mK),
                      ("h_conv_W_m2K", self.h_conv_W_m2K)):
            if not (v > 0.0):
                raise ValueError("ThermalModel: {} must be > 0".format(nm))


def peak_temperature_rise(Q_per_m: float, a_core_m: float, b_outer_m: float,
                          model: ThermalModel) -> float:
    """Centre-to-coolant temperature rise [K] for heat Q_per_m [W/m] generated uniformly in the
    core (Brown-Hoffman):
        dT = Q/(4 pi k_core) + Q/(2 pi k_clad) ln(b/a) + Q/(2 pi b h).
    The three terms are core conduction, cladding conduction, and surface convection."""
    q = float(Q_per_m)
    dT_core = q / (4.0 * np.pi * model.core_k_W_mK)
    dT_clad = q / (2.0 * np.pi * model.clad_k_W_mK) * np.log(b_outer_m / a_core_m)
    dT_conv = q / (2.0 * np.pi * b_outer_m * model.h_conv_W_m2K)
    return float(dT_core + dT_clad + dT_conv)


def radial_temperature_rise(Q_per_m: float, a_core_m: float, b_outer_m: float,
                            model: ThermalModel, r_m=None, n: int = 200):
    """Radial temperature-rise profile T(r) - T_coolant [K] for a uniformly core-heated fiber.
        core (r<=a): dT(r) = dT_center - (Q/(4 pi k_core)) (r/a)^2
        clad (a<r<=b): dT(r) = (Q/(2 pi k_clad)) ln(b/r) + Q/(2 pi b h)
    Returns (r [m], dT [K]). Monotonically decreasing from the centre; matches the outer-surface
    convective drop Q/(2 pi b h) at r=b."""
    q = float(Q_per_m)
    r = np.linspace(0.0, b_outer_m, n) if r_m is None else np.atleast_1d(np.asarray(r_m, float))
    dT_center = peak_temperature_rise(q, a_core_m, b_outer_m, model)
    dT = np.empty_like(r)
    core = r <= a_core_m
    dT[core] = dT_center - q / (4.0 * np.pi * model.core_k_W_mK) * (r[core] / a_core_m) ** 2
    cl = ~core
    dT[cl] = (q / (2.0 * np.pi * model.clad_k_W_mK) * np.log(b_outer_m / np.maximum(r[cl], 1e-30))
              + q / (2.0 * np.pi * b_outer_m * model.h_conv_W_m2K))
    return r, dT
