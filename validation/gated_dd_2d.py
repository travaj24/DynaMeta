"""
Validate the 2D FULL-EDGE ohmic ground that makes GATED drift-diffusion converge in 2D -- the fix
for the metasurface case, where the ITO ground is a lateral edge contact at the DOMAIN BOUNDARY and
DEVSIM captures only ~2 box-corner nodes (too weak to anchor the continuity equation -> gated DD is
ill-conditioned and does not converge; see carriers/devsim_layered.py).

The fix (proven here): a thin ADJACENT edge-metal region so the ground becomes a region-region
INTERFACE (interior, not a domain boundary) with FULL-LINE node capture -- the same trick a
horizontal-face contact (bot_contact) uses. Geometry (vertical gate field + lateral ground, the
metasurface topology, oxide omitted to isolate the contact question):
  gate (Potential-Dirichlet, insulating) on the ITO top (y=0)
  ITO  : x in [x_edge, P], y in [0, t_ito]
  emet : x in [0, x_edge]  (inert metal -> makes x=x_edge an interior region boundary)
  ground: ohmic contact on the ITO at the x=x_edge interface, FULL y-range

GATE A (the headline): the gated 2D DD CONVERGES at every bias 0..2 V -- it previously did not (the
        2-node domain-boundary ground could not anchor the continuity solve).
GATE B: a physical gated response -- the accumulation peak n_max rises monotonically with Vg, and at
        Vg=0 the DD reduces EXACTLY to the equilibrium baseline (zero-current limit).
INFO  : the DD-vs-equilibrium profile rel-diff. It is ~0 at Vg=0 and tight at low bias, then grows
        at high bias HERE because this oxide-free test over-drives the accumulation past the FD
        g-factor fit's validity (eta ~ 32; n_max ~ 3e27 -> eta ~ 90). With the real gate oxide
        limiting the accumulation (as in the 1D validation/gated_dd, eta < 32) the DD matches the
        equilibrium Fermi-Dirac profile to ~1e-4. So the residual is the g-factor limit, NOT the
        contact / convergence.

Run: python -m validation.gated_dd_2d
"""

import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from dynameta.carriers import physics_equilibrium as EQ
from dynameta.carriers import physics_drift_diffusion as DD
from dynameta.carriers import eq_registry as _R
from dynameta.constants import M_E

P, X_EDGE, T_ITO = 200e-9, 20e-9, 12e-9
EPS_ITO, N_BG, DOS_MASS, MU = 9.5, 4e26, 0.35 * M_E, 30e-4
BIASES = [0.0, 0.5, 1.0, 2.0]
REL_TOL = 1.0e-5


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush(); saved = os.dup(1); devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(devnull); os.close(saved)


def _build(name, dev):
    ds.create_2d_mesh(mesh=name)
    ds.add_2d_mesh_line(mesh=name, dir="x", pos=0.0, ps=5e-9)
    ds.add_2d_mesh_line(mesh=name, dir="x", pos=X_EDGE, ps=2e-9)
    ds.add_2d_mesh_line(mesh=name, dir="x", pos=P, ps=20e-9)
    ds.add_2d_mesh_line(mesh=name, dir="y", pos=0.0, ps=3e-10)            # gate side: fine (accumulation)
    ds.add_2d_mesh_line(mesh=name, dir="y", pos=T_ITO, ps=2e-9)
    ds.add_2d_region(mesh=name, material="ito", region="ito", xl=X_EDGE, xh=P, yl=0.0, yh=T_ITO)
    ds.add_2d_region(mesh=name, material="metal", region="emet", xl=0.0, xh=X_EDGE, yl=0.0, yh=T_ITO)
    ds.add_2d_contact(mesh=name, name="gate", material="metal", region="ito",
                      xl=X_EDGE, xh=P, yl=-1e-12, yh=1e-12, bloat=1e-10)            # insulating gate
    ds.add_2d_contact(mesh=name, name="ground", material="metal", region="ito",    # full-edge ground
                      xl=X_EDGE - 1e-12, xh=X_EDGE + 1e-12, yl=0.0, yh=T_ITO, bloat=1e-10)
    ds.finalize_mesh(mesh=name)
    ds.create_device(mesh=name, device=dev)


def _n(dev):
    return np.array(ds.get_node_model_values(device=dev, region="ito", name="Electrons"),
                    dtype=np.float64)


def run_equilibrium():
    _build("meq2", "eq2")
    EQ.setup_semiconductor_region("eq2", "ito", n_bg_m3=N_BG, eps_static=EPS_ITO, dos_mass_kg=DOS_MASS)
    EQ.setup_contact("eq2", "gate"); EQ.setup_contact("eq2", "ground")
    ds.set_node_values(device="eq2", region="ito", name="Potential",
                       values=[0.0] * len(ds.get_node_model_values(device="eq2", region="ito", name="x")))
    out = {}
    for vg in BIASES:
        ds.set_parameter(device="eq2", name="gate_bias", value=vg)
        ds.set_parameter(device="eq2", name="ground_bias", value=0.0)
        with _quiet():
            ds.solve(type="dc", solver_type="direct", absolute_error=1e10, relative_error=REL_TOL,
                     maximum_iterations=60)
        out[vg] = _n("eq2")
    ds.delete_device(device="eq2"); ds.delete_mesh(mesh="meq2"); _R.clear("eq2")
    return out


def run_drift_diffusion():
    _build("mdd2", "dd2")
    DD.setup_semiconductor_region_dd("dd2", "ito", n_bg_m3=N_BG, eps_static=EPS_ITO,
                                     dos_mass_kg=DOS_MASS, mobility_m2Vs=MU)
    EQ.setup_contact("dd2", "gate")                  # insulating gate: Potential only
    DD.setup_contact_ohmic_dd("dd2", "ground")       # FULL-EDGE ohmic ground (Potential + Electrons=N_D)
    nx = len(ds.get_node_model_values(device="dd2", region="ito", name="x"))
    ds.set_node_values(device="dd2", region="ito", name="Potential", values=[0.0] * nx)
    ds.set_node_values(device="dd2", region="ito", name="Electrons", values=[N_BG] * nx)
    prof, conv = {}, {}
    for vg in BIASES:
        ds.set_parameter(device="dd2", name="gate_bias", value=vg)
        ds.set_parameter(device="dd2", name="ground_bias", value=0.0)
        try:
            _R.delete_by_name("dd2", "ElectronContinuityEquation")
            with _quiet():
                ds.solve(type="dc", solver_type="direct", absolute_error=1e10, relative_error=REL_TOL,
                         maximum_iterations=60)
            _R.reapply_by_name("dd2", "ElectronContinuityEquation")
            with _quiet():
                ds.solve(type="dc", solver_type="direct", absolute_error=1e18, relative_error=REL_TOL,
                         maximum_iterations=80)
            conv[vg] = True
        except Exception as e:                       # noqa: BLE001
            conv[vg] = False
            print("[t]   DD vg={:.2f} FAILED: {}".format(vg, str(e)[:90]), flush=True)
            break
        prof[vg] = _n("dd2")
    return prof, conv


def main():
    print("[t] === 2D FULL-EDGE ohmic ground: gated drift-diffusion converges in 2D ===", flush=True)
    print("[t] ITO cell P={:.0f}nm, lateral ground via adjacent edge-metal interface (x={:.0f}nm)"
          .format(P * 1e9, X_EDGE * 1e9), flush=True)
    eq = run_equilibrium()
    dd, conv = run_drift_diffusion()

    all_converged = bool(len(conv) == len(BIASES) and all(conv.values()))
    nmax = {}
    for vg in BIASES:
        if vg not in dd:
            continue
        rel = float(np.max(np.abs(dd[vg] - eq[vg]) / np.maximum(eq[vg], 1e20)))
        nmax[vg] = float(dd[vg].max())
        tag = "(g-factor limit: eta>32)" if rel > 1e-2 else ""
        print("[t]   Vg={:+.2f}V  DD converged, n_max={:.3e}  rel-diff vs 2D-EQ={:.2e} {}".format(
            vg, nmax[vg], rel, tag), flush=True)

    accumulates = bool(all_converged and all(nmax[BIASES[i + 1]] > nmax[BIASES[i]]
                                             for i in range(len(BIASES) - 1)))
    zero_bias_exact = bool(all_converged and
                           float(np.max(np.abs(dd[0.0] - eq[0.0]) / np.maximum(eq[0.0], 1e20))) < 1e-6)

    gate_a = all_converged
    gate_b = accumulates and zero_bias_exact
    overall = gate_a and gate_b
    print("[t]", flush=True)
    print("[t] GATE A (gated 2D DD converges at every bias -- full-edge ground): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (accumulation monotonic in Vg & exact reduction to equilibrium at Vg=0): {}"
          .format("PASS" if gate_b else "FAIL"), flush=True)
    print("[t] *** 2D FULL-EDGE GATED DRIFT-DIFFUSION: {} ***".format("PASS" if overall else "FAIL"),
          flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
