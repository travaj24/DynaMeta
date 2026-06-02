"""
Steady-state thermal driver for thermo-optic / electro-thermal modulators (roadmap Phase 2a).
It PRODUCES a per-layer temperature field T that the caller places in a field bundle (the
`{"T": ...}` dict a ThermoOpticModel reads) to obtain a temperature-shifted eps. (The bridge does
not yet auto-assemble T from a driver -- assemble_eps currently builds only the carrier field 'n';
wiring drivers for {E, T} into the bridge is a tracked seam. The driver + ThermoOpticModel are
validated end-to-end at the FEM level in validation/thermo_optic_modulator.py.) For a LAYERED
stack with a heat flux conducted to a sink (the
thermo-optic analog of the electrostatics series-capacitor driver), 1D steady conduction gives a
series-THERMAL-RESISTANCE temperature profile:

    q [W/m^2] is continuous (no volumetric source between the flux plane and the sink); across
    layer i the temperature drop is dT_i = q * R_i with R_i = d_i / k_i (thermal resistance per
    area). The MEAN temperature of layer i is  T_i = T_sink + q * ( sum_{j below i} R_j + 0.5 R_i ).

Exact for a 1D layered geometry with the sink at one face and a uniform flux through the stack (a
resistive heater dumping power-per-area q into the far face, conducted to a substrate sink). A
full heat-equation FEM (volumetric Joule source, transient, lateral spreading) is a later
extension. Pure numpy; no devsim/ngsolve.
"""

from __future__ import annotations

import numpy as np


def steady_layered_temperature(k_thermal, thickness_m, flux_W_m2, T_sink_K: float = 300.0):
    """Per-layer MEAN temperature [K] for a layered stack under a steady heat flux conducted to a
    sink. Layers are ordered from the SINK side outward (index 0 adjacent to the sink); the flux
    `flux_W_m2` flows toward the sink. Returns one mean temperature per layer (input order)."""
    k = np.asarray(k_thermal, dtype=np.float64)
    d = np.asarray(thickness_m, dtype=np.float64)
    if k.ndim != 1 or k.shape != d.shape:
        raise ValueError("k_thermal and thickness_m must be 1D arrays of equal length")
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(d))):
        raise ValueError("k_thermal and thickness_m must be finite")
    if np.any(k <= 0.0) or np.any(d <= 0.0):
        raise ValueError("k_thermal and thickness_m must be strictly positive")
    R = d / k                                              # per-layer thermal resistance per area
    R_below = np.concatenate([[0.0], np.cumsum(R)[:-1]])   # resistance from sink to layer i's base
    return float(T_sink_K) + float(flux_W_m2) * (R_below + 0.5 * R)


def uniform_temperature_rise(power_per_area_W_m2, k_thermal, thickness_m,
                              T_sink_K: float = 300.0) -> float:
    """Lumped temperature of an active region: T = T_sink + q * R_total, R_total = sum(d_i/k_i) to
    the sink (one number; the simplest thermo-optic drive). `power_per_area_W_m2` is the heater
    flux conducted to the sink (>= 0 in the intended regime; a negative value would drive T below
    the sink, which is unphysical for a resistive heater -- the caller must supply physical T to a
    ThermoOpticModel)."""
    k = np.asarray(k_thermal, dtype=np.float64)
    d = np.asarray(thickness_m, dtype=np.float64)
    if k.ndim != 1 or k.shape != d.shape:
        raise ValueError("k_thermal and thickness_m must be 1D arrays of equal length")
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(d))):
        raise ValueError("k_thermal and thickness_m must be finite")
    if np.any(k <= 0.0) or np.any(d <= 0.0):
        raise ValueError("k_thermal and thickness_m must be strictly positive")
    return float(T_sink_K) + float(power_per_area_W_m2) * float(np.sum(d / k))
