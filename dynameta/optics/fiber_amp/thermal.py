"""Cladding-pumped operation and the quantum-defect thermal load (docs sec.7). Double-clad
pumping itself is already in the propagation model (Pump.cladding uses the overlap
Gamma_p = A_core/A_clad from waveguide.cladding_pump_overlap); this module adds the heat side:

  * the local heat density Q(z) [W/m] deposited in the core, from the rigorous optical-power
    balance Q = -d/dz (net forward optical flux) -- the power that leaves the optical fields
    (quantum defect + background loss + reabsorbed ASE) becomes heat. TRACKED (in-band) ASE that
    exits the fiber is correctly NOT counted as heat; spontaneous emission OUTSIDE the tracked
    band is invisible to the flux balance, so its pump-photon energy IS counted as heat -- an
    overestimate bounded by the untracked spontaneous fraction (audit S3-32);
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
           "total_heat_W", "radial_temperature_rise", "peak_temperature_rise",
           "thermal_lens_focal_power_per_m", "thermal_guiding_onset_Q_per_m",
           "thermo_optic_phase_rad", "solve_with_thermal_feedback"]

DN_DT_SILICA = 1.2e-5                                  # dn/dT of fused silica [1/K] (1.1-1.3e-5)


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
    quantum-defect + background-loss + reabsorbed-ASE heating. Tracked (in-band) ASE leaving the
    fiber is correctly excluded from the heat; spontaneous emission outside the tracked band is
    invisible to this balance and is therefore counted AS heat -- a conservative overestimate
    bounded by the untracked spontaneous fraction (audit S3-32). Widen the AseBand to tighten it."""
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


def thermal_lens_focal_power_per_m(Q_per_m, a_core_m: float, *, dndt: float = DN_DT_SILICA,
                                   k_core_W_mK: float = 1.38, n_core: float = 1.45):
    """Thermal-lens dioptric power PER UNIT LENGTH [1/m^2] of a uniformly core-heated fiber.
    The parabolic core temperature (radial_temperature_rise) makes a graded-index lens
    n(r) = n_c - (1/2) b r^2 with b = 2 delta_n0/a^2 and delta_n0 = (dn/dT) Q/(4 pi k), so
        D' = 1/(f L) = b/n_c = (dn/dT) Q / (2 pi n_c k a^2).
    NOTE the factor 2 vs the naive (dn/dT)Q/(4 pi k a^2): delta_n0 carries 1/(4 pi k) but the
    parabola CURVATURE doubles it (dossier Module 4 correction). Vectorizes over Q (accepts the
    heat_load_per_m profile). Divide-by-n_c = the in-fiber ray-matrix convention."""
    Q = np.asarray(Q_per_m, dtype=np.float64)
    out = dndt * Q / (2.0 * np.pi * n_core * k_core_W_mK * a_core_m ** 2)
    return out if out.ndim else float(out)


def thermal_guiding_onset_Q_per_m(na: float, *, dndt: float = DN_DT_SILICA,
                                  k_core_W_mK: float = 1.38, n_core: float = 1.45) -> float:
    """Heat density [W/m] at which the thermal index bump delta_n0 = (dn/dT)Q/(4 pi k) reaches
    the guiding step Delta_n = NA^2/(2 n_c) -- the onset of thermally-dominated guiding (mode
    shrinkage, the precursor regime of transverse mode instability). ~1.8 kW/m at NA = 0.06;
    falls to ~0.4-0.8 kW/m for NA 0.03-0.04 LMA fibers."""
    dn_step = float(na) ** 2 / (2.0 * n_core)
    return float(dn_step * 4.0 * np.pi * k_core_W_mK / dndt)


def thermo_optic_phase_rad(P_heat_W: float, lambda_m: float, *, dndt: float = DN_DT_SILICA,
                           k_core_W_mK: float = 1.38) -> float:
    """Accumulated centre-to-edge thermo-optic phase [rad] over the whole fiber:
        Delta_phi = k0 integral delta_n0 dz = (dn/dT) P_heat / (2 lambda k),
    with P_heat = integral Q dz = total_heat_W. ~28 rad for 10 W of heat at 1.55 um -- the
    thermal analogue of the B-integral (wavefront bulge unless guiding dominates)."""
    return float(dndt * float(P_heat_W) / (2.0 * float(lambda_m) * k_core_W_mK))


def solve_with_thermal_feedback(amp, model: ThermalModel, b_outer_m: float, *,
                                T_ref_K: float = 300.0, max_iter: int = 12,
                                tol_K: float = 0.2, relax: float = 0.7, **solve_kw):
    """SELF-CONSISTENT distributed-temperature solve: iterate
        solve -> Q(z) = heat_load_per_m -> T(z) = T_coolant + Q(z) * (core + clad + convection
        resistances, peak_temperature_rise coefficients) -> set_temperature_profile (per-z
        McCumber sigma_e scaling) -> re-solve
    with under-relaxation until max|Delta T| < tol_K. Closes the loop the audit left open: the
    heat load softens the signal-band emission cross-sections exactly where the fiber runs
    hottest. SENSITIVITY CAVEAT (adversarial-verifier finding): the pure-McCumber rescaling
    used here gives d ln sigma_e/dT = -(eps - h nu)/(k T^2) ~ -0.9 to -1.4 %/K at Yb
    1030-1064 nm, an UPPER BOUND ~3-5x the measured NET slopes (~-0.1 to -0.3 %/K, Newell
    Opt. Commun. 273:256 (2007) / Brilliant-Lagonik OL 26:1669 (2001)) because real sigma_a
    co-broadening partially compensates and is held fixed here -- so this loop somewhat
    OVERSTATES the thermal gain softening (conservative for derating). The ion temperature is
    the CORE CENTRE temperature (the dopant sits there). Returns (result, T_z_K, info) with info = {'iterations',
    'converged_T', 'max_dT_K', 'Q_per_m'}; the amplifier's profile is left SET (clear with
    amp.clear_temperature_profile()). b_outer_m = outer (coating/glass) radius for the radial
    conduction stack; model supplies conductivities/convection/coolant."""
    coef = (1.0 / (4.0 * np.pi * model.core_k_W_mK)
            + np.log(b_outer_m / amp.fiber.core_radius_m) / (2.0 * np.pi * model.clad_k_W_mK)
            + 1.0 / (2.0 * np.pi * b_outer_m * model.h_conv_W_m2K))   # K per (W/m)
    T_z = None
    res = None
    info = {"iterations": 0, "converged_T": False, "max_dT_K": np.inf, "Q_per_m": None}
    for it in range(max_iter):
        res = amp.solve(**solve_kw)
        Q = np.maximum(heat_load_per_m(res), 0.0)
        T_new = model.T_coolant_K + coef * Q
        if T_z is None:
            # the UNPROFILED first solve represents a uniform T_ref (the ion's reference
            # spectra), so the first convergence test must compare against T_ref -- NOT the
            # coolant (adversarial-verifier finding: with T_coolant != T_ref the old
            # coolant-referenced check could declare the T_ref solve converged at a coolant
            # the fiber never saw, up to several dB wrong in cryo/hot-coolant regimes)
            T_z = T_new
            dT = float(np.max(np.abs(T_new - T_ref_K)))
        else:
            dT = float(np.max(np.abs(T_new - T_z)))
            T_z = T_z + relax * (T_new - T_z)
        info.update(iterations=it + 1, max_dT_K=dT, Q_per_m=Q)
        if dT < tol_K:
            info["converged_T"] = True
            break
        amp.set_temperature_profile(res.z_m, T_z, T_ref_K=T_ref_K)
    return res, T_z, info


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
