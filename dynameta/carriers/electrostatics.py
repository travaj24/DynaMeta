"""
Electrostatic driving-field solver for field-effect (Pockels / Kerr / Franz-Keldysh) modulators
-- the Phase-1 driver (roadmap 1a). It PRODUCES the applied static E-field that the caller places
in a field bundle (the `{"E": ...}` dict a field-dependent EffectModel -- PockelsEffect, ... --
reads) to obtain a tensor eps. (The bridge does not yet auto-assemble E from a driver; that wiring
is a tracked seam. The driver + EffectModels are validated end-to-end at the FEM level in
validation/pockels_phase_modulator.py.)

For a LAYERED dielectric stack with a voltage applied across it (no mobile carriers, no free
charge), the normal displacement D is continuous between layers, so the per-layer static field is
the exact series-capacitor result:

    D = eps0 * eps_i * E_i = const   ->   V = sum_i E_i d_i = (D/eps0) * sum_i (d_i / eps_i)
    E_i = V / ( eps_i * sum_j (d_j / eps_j) )      [V/m]

This is EXACT for a 1D parallel-plate layered geometry (the thin-film EO phase arm the Vpi*L
oracle assumes). A laterally non-uniform geometry would need a Laplace/Poisson FEM solve (a later
extension). Pure numpy; no devsim/ngsolve.

Sign: with the applied voltage on the TOP electrode (larger z) relative to a grounded bottom,
phi decreases with depth, so E = -grad(phi) points along -z; the returned z-component is therefore
NEGATIVE for a positive applied voltage. (Only the field MAGNITUDE enters |Vpi|; the sign sets the
sign of the index change.)
"""

from __future__ import annotations

import numpy as np


def layered_static_field_z(eps_static, thickness_m, applied_V: float) -> np.ndarray:
    """Per-layer static field E_z [V/m] for a layered dielectric stack under `applied_V` across
    the whole stack (top electrode at +V, bottom grounded). Series-capacitor result; returns one
    E_z per layer, in the same order as the inputs (signed: negative for applied_V > 0)."""
    eps = np.asarray(eps_static, dtype=np.float64)
    d = np.asarray(thickness_m, dtype=np.float64)
    if eps.ndim != 1 or eps.shape != d.shape:
        raise ValueError("eps_static and thickness_m must be 1D arrays of equal length")
    if np.any(eps <= 0.0) or np.any(d <= 0.0):
        raise ValueError("eps_static and thickness_m must be strictly positive")
    series = float(np.sum(d / eps))                  # sum_j d_j/eps_j
    D_over_eps0 = float(applied_V) / series          # = D / eps0  [V/m]
    return -(D_over_eps0 / eps)                       # E_i = -(D/eps0)/eps_i, along -z for V>0


def parallel_plate_field_z(eps_static: float, gap_m: float, applied_V: float) -> float:
    """Single-layer (parallel-plate) static field E_z [V/m] = -V/gap. The simplest EO geometry:
    one dielectric of thickness `gap_m` fully between the electrodes (eps cancels for a single
    layer; kept in the signature for symmetry with the layered case)."""
    if not (gap_m > 0.0):
        raise ValueError("gap_m must be > 0")
    return -float(applied_V) / float(gap_m)
