"""Validate BIPOLAR drift-diffusion (holes + SRH) WIRED INTO THE 2D LayeredDevsimBuilder. The 1D
bipolar diode (validation/bipolar_diode.py) proved the 3-variable physics; this proves the builder
path: a Design with a bipolar_dd semiconductor layer carrying a lateral p-n junction
(net_doping_expr = -Na for x<P/2, +Nd for x>=P/2) and two edge contacts (anode on the p-side x_lo,
cathode on the n-side x_hi) solves through LayeredDevsimBuilder.solve() (its staged coupled-Newton
bipolar path) and rectifies.

GATE A: the 2D bipolar solve CONVERGES at forward, zero, and reverse bias.
GATE B: RECTIFICATION -- the forward terminal current is >> the reverse (opposite sign), and the
        forward current grows with bias (a real diode J-V, on the 2D mesh).

Run: python -m validation.bipolar_diode_2d
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from dynameta.materials import Material, MaterialRegistry, ConstantOptical, TransportModel, M_E
from dynameta.geometry import UnitCell, Stack, Layer, Electrode, Design
from dynameta.geometry.specs import Mesh2DSpec
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
from dynameta.carriers import eq_registry as _R
from dynameta.sweep import BiasPoint

P = 300e-9                     # cell period; graded junction at P/2
T_SI = 40e-9                   # Si layer thickness (thin -> fewer 2D nodes, fast)
NA = ND = 1.0e24               # 1e18 cm^-3
N_I = 1.0e16                   # Si-like intrinsic density
MU_N, MU_P = 0.135, 0.048      # Si mobilities (m^2/Vs)
TAU = 1.0e-7                   # SRH lifetime
V_FWD, V_REV = 0.4, -0.4       # forward, reverse anode bias (p-side)
BIASES = [V_FWD, 0.0, V_REV]


def _const(v):
    return lambda n: np.full_like(np.asarray(n, dtype=float), v)


def build_diode_design() -> Design:
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    # an inert metal -> the builder carves a thin edge-metal strip at each contact edge so the
    # anode/cathode are full-edge (region-interface) contacts, strong enough to anchor the 2D
    # bipolar solve (a bare domain-boundary edge contact captures only ~2 corner nodes).
    reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    xj = P / 2.0
    # GRADED junction (tanh over ~5 nm) rather than an abrupt step: NetDoping = Nd*tanh((x-xj)/w) for
    # Na==Nd. The smooth space-charge transition is far easier for the 2D coupled Newton than an
    # abrupt step at finite mesh resolution (the standard p-n convergence aid).
    w_grade = 5e-9
    nd_expr = "{nd:.8e}*tanh((x - {xj:.8e})/{w:.8e})".format(nd=ND, xj=xj, w=w_grade)
    reg.add(Material("Si_pn", ConstantOptical(12.0 + 0j),
                     transport=TransportModel(
                         n_bg_m3=ND, eps_static=11.7, dos_mass_kg_of_n_m3=_const(1.08 * M_E),
                         band_gap_eV=1.12, chi_eV=4.05, physics="bipolar_dd",
                         mobility_m2Vs_of_n_m3=_const(MU_N),
                         hole_mobility_m2Vs_of_n_m3=_const(MU_P),
                         tau_srh_s=TAU, n_i_m3=N_I, net_doping_expr=nd_expr,
                         dos_mass_p_kg=0.81 * M_E)))
    stack = Stack(layers=[Layer("si", T_SI, "Si_pn")], superstrate_material="air",
                  substrate_material="air")
    electrodes = [
        Electrode("anode", "si", "x_lo", role="biased"),                       # p-side (x<P/2)
        Electrode("cathode", "si", "x_hi", role="ground", fixed_voltage_V=0.0),  # n-side (x>P/2)
    ]
    m2 = Mesh2DSpec(x_spacing_feature_mid_m=2.0e-9)        # moderate mesh at the graded junction (x=P/2)
    return Design(name="pn_diode_2d", unit_cell=UnitCell.square(P), stack=stack,
                  electrodes=electrodes, materials=reg, mesh_2d=m2)


def _terminal_current(device, contact):
    jn = ds.get_contact_current(device=device, contact=contact, equation="ElectronContinuityEquation")
    jp = ds.get_contact_current(device=device, contact=contact, equation="HoleContinuityEquation")
    return float(jn) + float(jp)


def main():
    print("[t] === Bipolar p-n diode through the 2D LayeredDevsimBuilder ===", flush=True)
    d = build_diode_design()
    b = LayeredDevsimBuilder(d, mesh_name="pn2d_m", device_name="pn2d_d")
    currents, conv = {}, {}
    for v in BIASES:
        try:
            b.solve(BiasPoint({"anode": v}, "v{:+.2f}".format(v)))
            currents[v] = _terminal_current(b.device, "anode")
            conv[v] = True
            print("[t]   anode V={:+.2f}: converged, I_anode={:+.3e} A/m".format(v, currents[v]), flush=True)
        except Exception as e:                                                  # noqa: BLE001
            conv[v] = False
            print("[t]   anode V={:+.2f}: FAILED: {}".format(v, str(e)[:110]), flush=True)
    try:
        ds.delete_device(device=b.device); ds.delete_mesh(mesh=b.mesh_name); _R.clear(b.device)
    except Exception:
        pass

    gate_a = bool(len(conv) == len(BIASES) and all(conv.values()))
    gate_b = False
    if gate_a:
        i_fwd, i_rev = currents[V_FWD], currents[V_REV]
        opp_sign = (i_fwd * i_rev) < 0.0                       # rectifying: forward/reverse opposite sign
        ratio = abs(i_fwd) / max(abs(i_rev), 1e-30)
        gate_b = bool(opp_sign and ratio > 1.0e3 and abs(i_fwd) > abs(currents[0.0]))
        print("[t]   rectification: |I_fwd|/|I_rev| = {:.2e} (opposite sign={})".format(ratio, opp_sign),
              flush=True)

    print("[t] GATE A (2D bipolar solve converges fwd/zero/reverse): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (rectifies: |I_fwd| >> |I_rev|, opposite sign): {}".format(
        "PASS" if gate_b else "FAIL"), flush=True)
    ok = gate_a and gate_b
    print("[t] *** BIPOLAR DD IN THE 2D BUILDER: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
