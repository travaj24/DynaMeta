"""
Validate the small-signal AC (DEVSIM ssac) capacitance extraction
(dynameta.carriers.ac_analysis) against the EXACT parallel-plate result C = eps0 * eps_r / d.

Device: a 1-D parallel-plate dielectric capacitor built with dynameta's OWN equilibrium Poisson
physics (physics_equilibrium.setup_dielectric_region) -- the bottom plate is an ordinary grounded
contact (setup_contact), the top plate is the AC-driven contact (ac_analysis.setup_circuit_contact,
the new helper). A pure-Poisson (charge-free) dielectric has the geometric capacitance and NO
transport, so its ssac admittance is Y = i*omega*C with C frequency-INDEPENDENT and G = 0 -- an
exact oracle for the extraction. Two geometries confirm C scales as eps0*eps_r/d (not a single-point
coincidence).

This validates the EXTRACTION (the new capability): the equilibrium device gives the quasi-static
geometric/depletion C (the same physical quantity as analysis.gate_cv's dQ/dV, obtained here by a
rigorous small-signal linearization instead of a finite voltage difference). A frequency-RESOLVED
modulator bandwidth -- the carrier-dynamics roll-off where C(f) actually falls -- needs the
drift-diffusion transport physics (carriers.physics_drift_diffusion); ssac on a DD device is the
follow-on that exercises that. Here C is flat by construction.

GATE A: ssac C == eps0*eps_r/d at every probe frequency, on two geometries.
GATE B: C frequency-flat (pure-Poisson cap) and loss tangent G/(omega C) ~ 0 (lossless).
CHAIN : feed the validated areal C into analysis.lumped_rc_bandwidth -> a finite, positive f_3dB.

Run: python -m validation.ac_capacitance
"""

import contextlib
import os
import sys

import numpy as np
import devsim as ds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers import physics_equilibrium as EQ
from dynameta.carriers import eq_registry as _R
from dynameta.carriers import ac_analysis as AC
from dynameta.analysis import lumped_rc_bandwidth
from dynameta.constants import EPS0

DEVICE = "cap"
MESH = "cap_mesh"
REGION = "diel"

# (dielectric thickness [m], relative permittivity) -- two geometries so C = eps0*eps_r/d is
# confirmed by SCALING, not a single coincidental match.
GEOMETRIES = [(100e-9, 4.0), (60e-9, 9.0)]
FREQS_HZ = [1.0e3, 1.0e6, 1.0e9, 1.0e12]      # 9 decades -> exposes any (spurious) freq dependence

C_RTOL = 5.0e-3        # ssac C vs analytic eps0*eps_r/d (linear-potential cap is ~exact on any mesh)
FLAT_RTOL = 1.0e-6     # C(f) spread across frequency (pure-Poisson cap is exactly flat)
LOSS_TAN_MAX = 1.0e-6  # G/(omega C): lossless dielectric -> ~0

# representative access network for the f_3dB chain demo (ITO-like sheet R + a medium geometry)
RHO_SHEET, PATH_M, PAD_M = 800.0, 5e-6, 1e-6


@contextlib.contextmanager
def _quiet():
    """Suppress DEVSIM's per-iteration stdout (C-level fd redirect) so only [t] lines print."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def build_cap(d_m: float, eps_r: float) -> None:
    """1-D parallel-plate cap on dynameta's own equilibrium Poisson region: bottom plate grounded
    (setup_contact), top plate AC-driven (ac_analysis.setup_circuit_contact)."""
    # Clear prior state WITHOUT reset_devsim(): reset would also drop the UMFPACK ("custom")
    # direct-solver callback (the only solver in this build), breaking the next solve. Delete the
    # device/mesh and the circuit individually instead -- the UMFPACK registration then survives
    # across geometries, and delete_circuit() clears V1/vac so they can be re-added cleanly.
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for mh in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=mh)
    try:
        ds.delete_circuit()                            # clears V1/vac node + stale ssac solutions
    except Exception:
        pass
    _R.clear(DEVICE)

    ds.create_1d_mesh(mesh=MESH)
    ds.add_1d_mesh_line(mesh=MESH, pos=0.0, ps=d_m / 100.0, tag="bot")
    ds.add_1d_mesh_line(mesh=MESH, pos=d_m, ps=d_m / 100.0, tag="top")
    ds.add_1d_contact(mesh=MESH, name="bot", tag="bot", material="metal")
    ds.add_1d_contact(mesh=MESH, name="top", tag="top", material="metal")
    ds.add_1d_region(mesh=MESH, material="oxide", region=REGION, tag1="bot", tag2="top")
    ds.finalize_mesh(mesh=MESH)
    ds.create_device(mesh=MESH, device=DEVICE)

    EQ.setup_dielectric_region(DEVICE, REGION, eps_static=eps_r)   # charge-free Poisson (Laplace)
    EQ.setup_contact(DEVICE, "bot")                                # grounded plate (bot_bias = 0)
    AC.setup_circuit_contact(DEVICE, "top")                        # AC-driven plate (source V1)


def main():
    print("[t] === AC small-signal capacitance (DEVSIM ssac) vs exact parallel-plate ===",
          flush=True)
    print("[t] extraction: C = -Im(I_V1)/(omega) , G = -Re(I_V1) , unit AC drive (per area, 1-D)",
          flush=True)

    gate_a = True
    gate_b = True
    first_C_areal = None
    for d_m, eps_r in GEOMETRIES:
        C_analytic = eps_r * EPS0 / d_m                            # F/m^2
        build_cap(d_m, eps_r)
        with _quiet():
            ds.solve(type="dc", absolute_error=1e-10, relative_error=1e-10,
                     maximum_iterations=30)
            freqs, C, G = AC.ssac_admittance(FREQS_HZ)

        if first_C_areal is None:
            first_C_areal = float(np.mean(C))

        c_match = bool(np.all(np.abs(C / C_analytic - 1.0) < C_RTOL))
        c_spread = float((C.max() - C.min()) / np.mean(C)) if np.mean(C) != 0.0 else float("inf")
        c_flat = bool(c_spread < FLAT_RTOL)
        omega = 2.0 * np.pi * np.asarray(FREQS_HZ)
        loss_tan = np.abs(G) / (omega * np.abs(C))                 # G/(omega C) = tan(delta)
        lossless = bool(np.all(loss_tan < LOSS_TAN_MAX))

        gate_a = gate_a and c_match
        gate_b = gate_b and c_flat and lossless

        print("[t]", flush=True)
        print("[t] geometry d={:.0f} nm eps_r={:.1f}:  C_analytic = eps0*eps_r/d = {:.6e} F/m^2"
              .format(d_m * 1e9, eps_r, C_analytic), flush=True)
        for f, c, g, lt in zip(FREQS_HZ, C, G, loss_tan):
            print("[t]   f={:9.1e} Hz  C={:.6e} F/m^2  ratio={:.5f}  G={:+.2e} S/m^2  "
                  "tan(d)={:.1e}".format(f, c, c / C_analytic, g, lt), flush=True)
        print("[t]   C matches analytic (<{:.1e})={}  freq-flat (spread {:.1e}<{:.0e})={}  "
              "lossless={}".format(C_RTOL, c_match, c_spread, FLAT_RTOL, c_flat, lossless),
              flush=True)

    # ---- CHAIN: validated areal C -> intrinsic RC f_3dB (reuses the Stage-4 lumped model) ----
    cell_area = (370e-9) ** 2                                      # a representative unit-cell area
    R, C_cell, f3db = lumped_rc_bandwidth(first_C_areal, RHO_SHEET, path_length_m=PATH_M,
                                          pad_width_m=PAD_M, cell_area_m2=cell_area)
    chain_ok = bool(np.isfinite(f3db) and f3db > 0.0 and C_cell > 0.0)
    print("[t]", flush=True)
    print("[t] CHAIN ssac C -> f_3dB: C={:.3e} F/m^2  R_access={:.0f} Ohm  C_cell={:.3f} fF  "
          "f_3dB={:.2f} GHz".format(first_C_areal, R, C_cell * 1e15, f3db * 1e-9), flush=True)

    overall = gate_a and gate_b and chain_ok
    print("[t]", flush=True)
    print("[t] GATE A (ssac C == eps0*eps_r/d, both geometries): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (C frequency-flat & lossless): {}".format(
        "PASS" if gate_b else "FAIL"), flush=True)
    print("[t] CHAIN (C -> finite positive f_3dB): {}".format(
        "PASS" if chain_ok else "FAIL"), flush=True)
    print("[t] *** AC ssac CAPACITANCE: {} ***".format("PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
