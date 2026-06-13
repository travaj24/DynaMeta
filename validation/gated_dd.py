"""
Validate GATED drift-diffusion convergence: a 1D metal/oxide/ITO/ohmic MOS-cap (a gate across an
oxide -- NO DC current path) solved with the full electron-continuity drift-diffusion physics, and
its reduction to the equilibrium Fermi-Dirac carrier profile in the zero-current limit.

This closes the long-standing "gated cap does NOT converge with DD" limitation. The root cause was
that the semiconductor's ohmic ground was a WEAK 2-node contact (in the 2D metasurface mesh), which
cannot anchor the continuity equation across the gate-field region. In 1D the ITO ohmic contact is
a FULL boundary, so the continuity solve is well-posed -- and the recipe that makes it converge is:
  (1) a full-boundary (here, the whole 1D end) ohmic contact pinning Electrons = N_D,
  (2) a relaxed RELATIVE tolerance (the trivial-bias Potential ~ 0 makes ||du||/||u|| floor at the
      numerical-precision ~1e-6, so a 1e-10 rel target never converges -- abs is the real gate),
  (3) a staged solve: a potential-only Poisson pre-solve (continuity frozen) then coupled Newton.
The 2D metasurface uses the SAME ohmic contact promoted to a FULL EDGE -- shipped 8106849
(validation/gated_dd_2d.py) and builder-wired for the reference metasurface in 77f92ab
(validation/gated_dd_builder.py); this 1D case is the physics + recipe proof.

Physics check (the rigorous oracle): with no DC current the DD steady state MUST reduce to the
local equilibrium Fermi-Dirac density n(z) = N_c F_1/2((Potential - Phi_c0)/V_t). So the DD and the
equilibrium-Poisson carrier profiles -- and the gate charge Q(Vg) = q INT (n - n_bg) dz (hence the
C-V) -- must agree at every bias.

GATE A: the gated DD CONVERGES at every bias 0..2 V (the headline -- it previously did not).
GATE B: the DD carrier profile matches the equilibrium profile to < 1% rel, and the DD gate charge
        Q(Vg) matches the equilibrium Q(Vg) to < 5% and rises monotonically (accumulation).

Run: python -m validation.gated_dd
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
from dynameta.core.numerics import trapz as _trapz
from dynameta.constants import M_E, Q_E

T_OX, T_ITO = 10e-9, 12e-9
EPS_OX, EPS_ITO = 9.0, 9.5
N_BG = 4e26
DOS_MASS = 0.35 * M_E
MU = 30e-4
BIASES = [0.0, 0.5, 1.0, 1.5, 2.0]
REL_TOL = 1.0e-5          # relaxed: the trivial-bias rel error floors ~1e-6 (abs is the real gate)
PROFILE_RTOL = 1.0e-2     # DD vs equilibrium carrier profile
Q_RTOL = 5.0e-2           # DD vs equilibrium gate charge Q(Vg)


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush(); saved = os.dup(1); devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(devnull); os.close(saved)


def _build_mesh(name):
    ds.create_1d_mesh(mesh=name)
    ds.add_1d_mesh_line(mesh=name, pos=0.0, ps=1e-9, tag="gate")
    ds.add_1d_mesh_line(mesh=name, pos=T_OX, ps=2e-10, tag="mid")        # fine at oxide/ITO interface
    ds.add_1d_mesh_line(mesh=name, pos=T_OX + T_ITO, ps=1e-9, tag="back")
    ds.add_1d_contact(mesh=name, name="gate", tag="gate", material="metal")
    ds.add_1d_contact(mesh=name, name="back", tag="back", material="metal")
    ds.add_1d_region(mesh=name, material="oxide", region="ox", tag1="gate", tag2="mid")
    ds.add_1d_region(mesh=name, material="ito", region="ito", tag1="mid", tag2="back")
    ds.add_1d_interface(mesh=name, name="ox_ito", tag="mid")
    ds.finalize_mesh(mesh=name)


def _ito_xn(dev):
    x = np.array(ds.get_node_model_values(device=dev, region="ito", name="x"), dtype=np.float64)
    n = np.array(ds.get_node_model_values(device=dev, region="ito", name="Electrons"), dtype=np.float64)
    return x, n


def _gate_charge(x, n):
    """Accumulated electron sheet charge Q(Vg) = q INT (n - n_bg) dz over the ITO [C/m^2]."""
    return Q_E * float(_trapz(n - N_BG, x))


def _solve(rel=REL_TOL, ab=1e10, mi=60):
    with _quiet():
        ds.solve(type="dc", solver_type="direct", absolute_error=ab, relative_error=rel,
                 maximum_iterations=mi)


def run_equilibrium():
    _build_mesh("meq"); ds.create_device(mesh="meq", device="eq")
    EQ.setup_dielectric_region("eq", "ox", eps_static=EPS_OX)
    EQ.setup_semiconductor_region("eq", "ito", n_bg_m3=N_BG, eps_static=EPS_ITO, dos_mass_kg=DOS_MASS)
    EQ.setup_interface("eq", "ox_ito")
    EQ.setup_contact("eq", "gate"); EQ.setup_contact("eq", "back")
    for reg in ("ox", "ito"):
        nx = len(ds.get_node_model_values(device="eq", region=reg, name="x"))
        ds.set_node_values(device="eq", region=reg, name="Potential", values=[0.0] * nx)
    prof, Q = {}, {}
    for vg in BIASES:
        ds.set_parameter(device="eq", name="gate_bias", value=vg)
        ds.set_parameter(device="eq", name="back_bias", value=0.0)
        _solve()
        x, n = _ito_xn("eq"); prof[vg] = (x, n); Q[vg] = _gate_charge(x, n)
    ds.delete_device(device="eq"); ds.delete_mesh(mesh="meq"); _R.clear("eq")
    return prof, Q


def run_drift_diffusion():
    _build_mesh("mdd"); ds.create_device(mesh="mdd", device="dd")
    EQ.setup_dielectric_region("dd", "ox", eps_static=EPS_OX)
    DD.setup_semiconductor_region_dd("dd", "ito", n_bg_m3=N_BG, eps_static=EPS_ITO,
                                     dos_mass_kg=DOS_MASS, mobility_m2Vs=MU)
    EQ.setup_interface("dd", "ox_ito")
    EQ.setup_contact("dd", "gate")                   # gate on oxide: Potential Dirichlet only
    DD.setup_contact_ohmic_dd("dd", "back")          # ITO ohmic: Potential + Electrons=N_D (full 1D boundary)
    for reg in ("ox", "ito"):
        nx = len(ds.get_node_model_values(device="dd", region=reg, name="x"))
        ds.set_node_values(device="dd", region=reg, name="Potential", values=[0.0] * nx)
    ds.set_node_values(device="dd", region="ito", name="Electrons",
                       values=[N_BG] * len(ds.get_node_model_values(device="dd", region="ito", name="x")))
    prof, Q, converged = {}, {}, {}
    for vg in BIASES:
        ds.set_parameter(device="dd", name="gate_bias", value=vg)
        ds.set_parameter(device="dd", name="back_bias", value=0.0)
        try:
            _R.delete_by_name("dd", "ElectronContinuityEquation")    # (1) Poisson-only pre-solve
            _solve()
            _R.reapply_by_name("dd", "ElectronContinuityEquation")
            _solve(ab=1e18)                                          # (2) coupled Newton (density-scaled abs)
            converged[vg] = True
        except Exception as e:                                       # noqa: BLE001
            converged[vg] = False
            print("[t]   DD vg={:.2f} FAILED: {}".format(vg, str(e)[:90]), flush=True)
            break
        x, n = _ito_xn("dd"); prof[vg] = (x, n); Q[vg] = _gate_charge(x, n)
    return prof, Q, converged


def main():
    print("[t] === GATED drift-diffusion: 1D MOS-cap converges + reduces to equilibrium ===",
          flush=True)
    print("[t] gate | oxide({:.0f}nm) | ITO({:.0f}nm, n_bg={:.0e}) | ohmic-back ; no DC path".format(
        T_OX * 1e9, T_ITO * 1e9, N_BG), flush=True)
    eq_prof, eq_Q = run_equilibrium()
    dd_prof, dd_Q, conv = run_drift_diffusion()

    all_converged = bool(len(conv) == len(BIASES) and all(conv.values()))
    profile_ok = True
    q_ok = True
    for vg in BIASES:
        if vg not in dd_prof:
            profile_ok = False; q_ok = False; continue
        xe, ne = eq_prof[vg]; xd, nd = dd_prof[vg]
        rel = float(np.max(np.abs(nd - ne) / np.maximum(ne, 1e20)))
        qrel = abs(dd_Q[vg] - eq_Q[vg]) / max(abs(eq_Q[vg]), 1e-30) if vg != 0.0 else 0.0
        profile_ok = profile_ok and (rel < PROFILE_RTOL)
        q_ok = q_ok and (qrel < Q_RTOL)
        print("[t]   Vg={:+.2f}V  DD n(ox-iface)={:.4e}  prof rel-diff={:.2e}  Q_dd={:+.3e} "
              "Q_eq={:+.3e} C/m^2 (rel {:.2e})".format(vg, nd[0], rel, dd_Q[vg], eq_Q[vg], qrel),
              flush=True)
    # accumulation: gate charge rises monotonically with Vg (both eq and dd)
    q_monotonic = bool(all(dd_Q[BIASES[i + 1]] > dd_Q[BIASES[i]] for i in range(len(BIASES) - 1))
                       if all_converged else False)

    gate_a = all_converged
    gate_b = profile_ok and q_ok and q_monotonic
    overall = gate_a and gate_b
    print("[t]", flush=True)
    print("[t] GATE A (gated DD converges at every bias 0..2 V): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (DD profile==equilibrium <{:.0%}, Q matches <{:.0%} & monotonic): {}".format(
        PROFILE_RTOL, Q_RTOL, "PASS" if gate_b else "FAIL"), flush=True)
    print("[t] *** GATED DRIFT-DIFFUSION (1D): {} ***".format("PASS" if overall else "FAIL"),
          flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
