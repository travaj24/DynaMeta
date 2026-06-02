"""
Validate small-signal AC (DEVSIM ssac) on a DRIFT-DIFFUSION device: the junction capacitance of a
reverse-biased p-n diode vs the analytic abrupt-junction depletion capacitance C_j = eps/W(V).

This is the frequency-RESOLVED companion to the equilibrium-Poisson ssac (validation/ac_capacitance):
it runs ssac on a full bipolar drift-diffusion device (transport, not just electrostatics), proving
the carrier-charge capacitance the transient charge time-node models (NCharge/PCharge in
physics_bipolar_dd) unlock. A diode is the right vehicle -- it has a DC current path, so it
converges (unlike the gated capacitor, whose DD solve does not -- that needs the separate gated-DD
convergence work for a frequency-resolved MODULATOR bandwidth).

Device: 1D Si p-n diode (2 um, abrupt junction), Na=Nd=1e24 m^-3, n_i=1e16. The n-contact is a
grounded ohmic contact; the p-contact is CIRCUIT-DRIVEN (setup_contact_ohmic_bipolar_circuit) so the
AC voltage source V1 excites the terminal. After a staged DC solve + a reverse-bias ramp,
ac_analysis.ssac_admittance extracts C(f) = -Im(I_V1)/omega.

GATE A: ssac junction C matches the analytic depletion C_j = eps0 eps_r / W(V) within 10% at each
        reverse bias, and TRACKS the C ~ 1/sqrt(Vbi - V) bias dependence (C falls as |V| rises).
GATE B: C is frequency-flat across the probe band (a depletion capacitance is ~constant in f).

Run: python -m validation.ac_diode
"""

import contextlib
import math
import os
import sys

import numpy as np
import devsim as ds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers import physics_bipolar_dd as BP
from dynameta.carriers import eq_registry as _R
from dynameta.carriers import ac_analysis as AC
from dynameta.carriers.physics_equilibrium import V_T, EPS0, Q_E

LEN_M, X_JUNC = 2.0e-6, 1.0e-6
NA_M3 = ND_M3 = 1.0e24
N_I_M3, N_DOS_M3 = 1.0e16, 2.8e25
MU_N, MU_P, TAU_S, EPS_R = 0.135, 0.048, 1.0e-7, 11.7
DEVICE, MESH, REGION = "acdiode", "acdiode_mesh", "bulk"

REVERSE_BIASES = [-0.5, -1.0, -2.0]       # p-contact reverse (depletion widens, C falls)
FREQS_HZ = [1.0e3, 1.0e6, 1.0e9]
C_RTOL = 0.10                              # ssac C vs analytic abrupt-junction depletion C_j
FLAT_RTOL = 1.0e-3                         # C(f) spread across the probe band

VBI = V_T * math.log(NA_M3 * ND_M3 / N_I_M3 ** 2)


def depletion_cap(v_applied: float) -> float:
    """Analytic abrupt-junction depletion capacitance per area [F/m^2] at applied bias v_applied
    (reverse < 0): C_j = eps0 eps_r / W, W = sqrt(2 eps0 eps_r (Vbi - V)/q (1/Na + 1/Nd))."""
    W = math.sqrt(2.0 * EPS_R * EPS0 * (VBI - v_applied) / Q_E * (1.0 / NA_M3 + 1.0 / ND_M3))
    return EPS_R * EPS0 / W


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush(); saved = os.dup(1); devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(devnull); os.close(saved)


def _solve(ae=1e18, re=1e-6, mi=80):
    with _quiet():
        ds.solve(type="dc", solver_type="direct", absolute_error=ae, relative_error=re,
                 maximum_iterations=mi)


def build():
    ds.create_1d_mesh(mesh=MESH)
    ds.add_1d_mesh_line(mesh=MESH, pos=0.0, ps=2e-9, tag="p")
    ds.add_1d_mesh_line(mesh=MESH, pos=X_JUNC, ps=5e-10)
    ds.add_1d_mesh_line(mesh=MESH, pos=LEN_M, ps=2e-9, tag="n")
    ds.add_1d_contact(mesh=MESH, name="p", tag="p", material="metal")
    ds.add_1d_contact(mesh=MESH, name="n", tag="n", material="metal")
    ds.add_1d_region(mesh=MESH, material="Si", region=REGION, tag1="p", tag2="n")
    ds.finalize_mesh(mesh=MESH)
    ds.create_device(mesh=MESH, device=DEVICE)
    ds.node_model(device=DEVICE, region=REGION, name="NetDoping",
                  equation="ifelse(x < {}, {}, {})".format(X_JUNC, -NA_M3, ND_M3))
    BP.setup_bipolar_region(DEVICE, REGION, eps_static=EPS_R, n_dos_m3=N_DOS_M3, n_i_m3=N_I_M3,
                            mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=TAU_S, tau_p_s=TAU_S)
    BP.setup_equilibrium_seed_models(DEVICE, REGION)
    BP.setup_contact_ohmic_bipolar(DEVICE, "n")                       # grounded ohmic
    BP.setup_contact_ohmic_bipolar_circuit(DEVICE, "p", node_name="vp", source_name="V1")


def staged_equilibrium_solve():
    """Seed equilibrium, potential-only pre-solve (freeze carriers), then coupled Newton at vp=0."""
    n0 = "ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(ce=BP.CELEC, ch=BP.CHOLE)
    ds.node_model(device=DEVICE, region=REGION, name="_seed_psi",
                  equation="V_t*log({}/n_i)".format(n0))
    for nm, src in (("Potential", "_seed_psi"), ("Electrons", "IntrinsicElectrons"),
                    ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=DEVICE, region=REGION, name=nm,
                           values=ds.get_node_model_values(device=DEVICE, region=REGION, name=src))
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(DEVICE, ceq)
    with _quiet():
        ds.solve(type="dc", solver_type="direct", absolute_error=1e10, relative_error=1e-10,
                 maximum_iterations=100)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(DEVICE, ceq)
    for nm, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=DEVICE, region=REGION, name=nm,
                           values=ds.get_node_model_values(device=DEVICE, region=REGION, name=src))
    _solve()


def ramp_to(v_target: float, step: float = 0.05):
    v = 0.0
    while abs(v - v_target) > 1e-9:
        v = max(v_target, v - step) if v_target < 0 else min(v_target, v + step)
        ds.circuit_alter(name="V1", value=v)            # change the source DC value (not re-create)
        _solve()
    return v


def main():
    print("[t] === AC ssac on a drift-diffusion p-n diode: junction C vs depletion C_j ===",
          flush=True)
    print("[t] Na=Nd={:.0e} m^-3, n_i={:.0e}, Vbi={:.4f} V; p-contact circuit-driven (source V1)"
          .format(NA_M3, N_I_M3, VBI), flush=True)
    build()
    staged_equilibrium_solve()
    print("[t] equilibrium (vp=0) solved; ramping reverse + ssac", flush=True)

    gate_a = True
    gate_b = True
    c_by_bias = []
    for vb in REVERSE_BIASES:
        ramp_to(vb)
        freqs, C, G = AC.ssac_admittance(FREQS_HZ, source_name="V1")
        c_mean = float(np.mean(C))
        c_by_bias.append(c_mean)
        c_anal = depletion_cap(vb)
        ratio = c_mean / c_anal
        spread = float((C.max() - C.min()) / c_mean) if c_mean != 0 else float("inf")
        c_match = bool(abs(ratio - 1.0) < C_RTOL)
        c_flat = bool(spread < FLAT_RTOL)
        gate_a = gate_a and c_match
        gate_b = gate_b and c_flat
        print("[t] V={:+.2f}V  ssac C={:.4e}  depletion C_j={:.4e} F/m^2  ratio={:.3f}  "
              "freq-flat(spread={:.1e})={}".format(vb, c_mean, c_anal, ratio, spread, c_flat),
              flush=True)

    # C must FALL as reverse bias rises (depletion widens): C(-0.5) > C(-1) > C(-2).
    c_decreasing = bool(c_by_bias[0] > c_by_bias[1] > c_by_bias[2])
    gate_a = gate_a and c_decreasing

    overall = gate_a and gate_b
    print("[t]", flush=True)
    print("[t] GATE A (ssac C == depletion C_j within {:.0%}, C falls with reverse bias): {}".format(
        C_RTOL, "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (C frequency-flat): {}".format("PASS" if gate_b else "FAIL"), flush=True)
    print("[t] *** AC ssac DRIFT-DIFFUSION DIODE: {} ***".format("PASS" if overall else "FAIL"),
          flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
