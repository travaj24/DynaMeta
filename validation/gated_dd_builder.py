"""
Validate the builder-wired full-edge gated drift-diffusion on the REAL Park metasurface: the
LayeredDevsimBuilder now auto-carves a thin edge-metal strip at each DD-semiconductor edge ground
(making it a region-region interface -> full-line node capture), so the GATED DD metasurface
converges -- the thing that previously did not (carriers/devsim_layered.py). And it reduces to the
equilibrium Fermi-Dirac accumulation in the zero-current limit, the rigorous oracle.

This closes roadmap item 1 (builder wiring) and exercises item 2's chain (the converged DD
metasurface -> gate C-V -> intrinsic RC f_3dB via the existing analysis helpers). The
frequency-RESOLVED gate C(omega) via ssac on the gate is the small remaining follow-on (it needs the
unipolar-DD charge time-node model + a circuit-driven gate); the quasi-static C here is what the
equilibrium path already provides, now reproduced by the full DD solve.

GATE A: the gated DD Park metasurface CONVERGES at every gate bias (0, +1, +2 V) -- previously it
        did not (weak 2-node ITO edge ground).
GATE B: the DD ITO accumulation profile (x-averaged over the cell) matches the equilibrium
        Fermi-Dirac profile to < 3% (the oxide keeps eta within the FD g-factor's validity, as in
        the 1D validation/gated_dd).
CHAIN : gate_cv on the DD CarrierFields -> C(Vg) > 0 -> lumped_rc_bandwidth -> a finite f_3dB.

Run: python -m validation.gated_dd_builder
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from validation._reference_device import build_reference_modulator
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
from dynameta.carriers import eq_registry as _R
from dynameta.sweep import BiasPoint
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.analysis import gate_cv, sheet_resistance_ohm_sq, lumped_rc_bandwidth

GATE_BIASES = [0.0, 1.0, 2.0]
PROFILE_RTOL = 3.0e-2
MU, T_ITO, PERIOD = 30e-4, 5e-9, 370e-9          # ITO mobility, thickness, cell (for the RC chain)


def _sweep(physics):
    """Solve the Park metasurface (physics='equilibrium'|'drift_diffusion') at each gate bias;
    return {vg: CarrierField} and a per-bias convergence flag. Each builder is torn down after."""
    d = build_reference_modulator(physics)
    b = LayeredDevsimBuilder(d, mesh_name=physics + "_m", device_name=physics + "_d")
    fields, conv = {}, {}
    for vg in GATE_BIASES:
        try:
            fields[vg] = b.solve(BiasPoint({"top_contact": vg}, "g{:+.0f}".format(vg)))
            conv[vg] = True
        except Exception as e:                       # noqa: BLE001
            conv[vg] = False
            print("[t]   {} vg={:+.1f} FAILED: {}".format(physics, vg, str(e)[:90]), flush=True)
            break
    try:
        ds.delete_device(device=b.device); ds.delete_mesh(mesh=b.mesh_name); _R.clear(b.device)
    except Exception:
        pass
    return fields, conv


def _ito_zprofile(cf):
    """x-averaged ITO electron z-profile from a CarrierField (robust to the ~1nm edge-metal seam)."""
    g = np.asarray(cf.regions["ito"].grid_fields[ELECTRON_DENSITY], dtype=np.float64)  # (nx, nz)
    return g.mean(axis=0)


def main():
    print("[t] === Builder-wired full-edge gated DD on the Park metasurface ===", flush=True)
    eq, eq_conv = _sweep("equilibrium")
    dd, dd_conv = _sweep("drift_diffusion")

    dd_all = bool(len(dd_conv) == len(GATE_BIASES) and all(dd_conv.values()))
    profile_ok = True
    for vg in GATE_BIASES:
        if vg not in dd or vg not in eq:
            profile_ok = False; continue
        pe, pd = _ito_zprofile(eq[vg]), _ito_zprofile(dd[vg])
        rel = float(np.max(np.abs(pd - pe) / np.maximum(pe, 1e20)))
        profile_ok = profile_ok and (rel < PROFILE_RTOL)
        print("[t]   Vg={:+.1f}V  DD n_max={:.3e}  EQ n_max={:.3e}  profile rel-diff={:.2e}".format(
            vg, pd.max(), pe.max(), rel), flush=True)

    # CHAIN: gate C-V from the DD fields -> intrinsic RC f_3dB (item 2 quasi-static path)
    chain_ok = False
    if dd_all:
        Vg, Q, Vmid, C = gate_cv(list(dd.values()), "ito", voltage_key="top_contact")
        rho_s = sheet_resistance_ohm_sq(float(dd[GATE_BIASES[0]].n_bg_by_region["ito"]), MU, T_ITO)
        R, C_cell, f3db = lumped_rc_bandwidth(C, rho_s, path_length_m=5e-6, pad_width_m=1e-6,
                                              cell_area_m2=PERIOD ** 2)
        chain_ok = bool(np.all(C > 0) and np.all(np.isfinite(f3db)) and np.all(f3db > 0))
        print("[t]   gate C-V (DD): C(Vg) = {} mF/m^2 ; f_3dB = {} GHz".format(
            np.round(C * 1e3, 2).tolist(),
            np.round(np.atleast_1d(f3db) * 1e-9, 2).tolist()), flush=True)

    gate_a = dd_all
    gate_b = profile_ok
    overall = gate_a and gate_b and chain_ok
    print("[t]", flush=True)
    print("[t] GATE A (gated DD metasurface converges at every gate bias): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (DD accumulation profile == equilibrium < {:.0%}): {}".format(
        PROFILE_RTOL, "PASS" if gate_b else "FAIL"), flush=True)
    print("[t] CHAIN (DD gate C-V -> C>0 -> finite f_3dB): {}".format(
        "PASS" if chain_ok else "FAIL"), flush=True)
    print("[t] *** BUILDER-WIRED GATED DD (Park metasurface): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
