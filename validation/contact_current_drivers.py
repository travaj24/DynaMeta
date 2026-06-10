"""D1 contact-current extractor oracle (terminal currents into CarrierField.extras).

GATE A (Ohm's law, closed form): a UNIFORM n-type Si bar (the bipolar 2D-builder diode design
        with the junction removed) biased at 10 mV is a resistor -- the extracted terminal
        current must equal I = sigma A V / L with sigma = q Nd mu_n, A = t_layer * period_y,
        L the span between the two full-edge contact interfaces (P - 2 w_strip).
GATE B (charge conservation + depth linearity): sum of the extracted contact currents ~ 0
        (DEVSIM reports INTO-device currents); doubling depth_m EXACTLY doubles every current.
GATE C (bipolar seam): the forward-biased p-n diode of validation/bipolar_diode_2d carries
        contact_currents_A in CarrierField.extras; each entry equals the manual
        (electron + hole) get_contact_current sum * period_y EXACTLY, and conserves charge.
GATE D (equilibrium off-switch): the same bar with physics='equilibrium' (Poisson-only) has NO
        contact_currents_A key -- continuity contact equations do not exist, the extractor
        returns {}, and extras stays empty (byte-identical).

Run: python -m validation.contact_current_drivers
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from dynameta.constants import Q_E
from dynameta.materials import Material, MaterialRegistry, ConstantOptical, TransportModel, M_E
from dynameta.geometry import UnitCell, Stack, Layer, Electrode, Design
from dynameta.geometry.specs import Mesh2DSpec
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder, _EDGE_METAL_W_M
from dynameta.carriers.contact_current import extract_contact_currents
from dynameta.carriers import eq_registry as _R
from dynameta.sweep import BiasPoint

P = 300e-9
T_SI = 40e-9
ND = 1.0e24
N_I = 1.0e16
MU_N, MU_P = 0.135, 0.048
V_BAR = 0.01                                       # ohmic drive for the uniform bar


def _const(v):
    return lambda n: np.full_like(np.asarray(n, dtype=float), v)


def _bar_design(physics: str) -> Design:
    """Uniform n-type Si bar with edge contacts -- a resistor (bipolar physics) or a Poisson-only
    equilibrium device, depending on `physics`."""
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    kw = dict(n_bg_m3=ND, eps_static=11.7, dos_mass_kg_of_n_m3=_const(1.08 * M_E),
              band_gap_eV=1.12, chi_eV=4.05, physics=physics,
              mobility_m2Vs_of_n_m3=_const(MU_N))
    if physics == "bipolar_dd":
        kw.update(hole_mobility_m2Vs_of_n_m3=_const(MU_P), tau_srh_s=1.0e-7, n_i_m3=N_I,
                  net_doping_expr="{:.8e}".format(ND), dos_mass_p_kg=0.81 * M_E)
    reg.add(Material("Si_bar", ConstantOptical(12.0 + 0j), transport=TransportModel(**kw)))
    stack = Stack(layers=[Layer("si", T_SI, "Si_bar")], superstrate_material="air",
                  substrate_material="air")
    electrodes = [Electrode("anode", "si", "x_lo", role="biased"),
                  Electrode("cathode", "si", "x_hi", role="ground", fixed_voltage_V=0.0)]
    return Design(name="bar_" + physics, unit_cell=UnitCell.square(P), stack=stack,
                  electrodes=electrodes, materials=reg, mesh_2d=Mesh2DSpec())


def _teardown(b):
    try:
        ds.delete_device(device=b.device); ds.delete_mesh(mesh=b.mesh_name); _R.clear(b.device)
    except Exception:
        pass


def main():
    print("[cc] === D1 contact-current extractor ===", flush=True)
    ok = True

    # ---- GATE A + B: uniform resistor bar (bipolar physics, no junction) ----
    d = _bar_design("bipolar_dd")
    b = LayeredDevsimBuilder(d, mesh_name="ccbar_m", device_name="ccbar_d")
    cf = b.solve(BiasPoint({"anode": V_BAR}, "vbar"))
    cc = cf.extras.get("contact_currents_A")
    cc2 = extract_contact_currents(b.device, depth_m=2.0 * d.unit_cell.period_y_m)
    _teardown(b)
    if not cc or set(cc) != {"anode", "cathode"}:
        print("[cc] FAIL: bar extras missing contacts (got {})".format(cc), flush=True)
        return False
    sigma = Q_E * ND * MU_N                                   # p0 = n_i^2/Nd ~ 1e8 -> negligible
    L = P - 2.0 * _EDGE_METAL_W_M                             # span between contact interfaces
    I_ohm = sigma * (T_SI * d.unit_cell.period_y_m) * V_BAR / L
    relA = abs(abs(cc["anode"]) - I_ohm) / I_ohm
    g_a = bool(relA < 2e-2)
    ok = ok and g_a
    print("[cc] GATE A: extracted |I| = {:.4e} A vs Ohm sigma*A*V/L = {:.4e} A (rel {:.1e}) "
          "-> {}".format(abs(cc["anode"]), I_ohm, relA, "PASS" if g_a else "FAIL"), flush=True)

    cons = abs(cc["anode"] + cc["cathode"]) / max(abs(cc["anode"]), 1e-300)
    ratio = cc2["anode"] / cc["anode"]
    g_b = bool(cons < 1e-8 and ratio == 2.0)
    ok = ok and g_b
    print("[cc] GATE B: charge conservation |I_a + I_c|/|I_a| = {:.1e}; 2x depth ratio = {!r} "
          "-> {}".format(cons, ratio, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: bipolar p-n diode seam (reuse the validated 2D-builder diode) ----
    from validation.bipolar_diode_2d import build_diode_design
    dd = build_diode_design()
    b2 = LayeredDevsimBuilder(dd, mesh_name="ccpn_m", device_name="ccpn_d")
    cf2 = b2.solve(BiasPoint({"anode": 0.4}, "fwd"))
    cc_d = cf2.extras.get("contact_currents_A")
    manual = {}
    for c in ("anode", "cathode"):
        jn = ds.get_contact_current(device=b2.device, contact=c,
                                    equation="ElectronContinuityEquation")
        jp = ds.get_contact_current(device=b2.device, contact=c,
                                    equation="HoleContinuityEquation")
        manual[c] = (float(jn) + float(jp)) * dd.unit_cell.period_y_m
    _teardown(b2)
    match = max(abs(cc_d[c] - manual[c]) for c in manual) if cc_d else float("inf")
    cons_d = abs(cc_d["anode"] + cc_d["cathode"]) / max(abs(cc_d["anode"]), 1e-300) if cc_d else 1.0
    # Conservation bound 1e-3, NOT machine: DEVSIM's ohmic contact replaces continuity with
    # Dirichlet at the contact nodes, so get_contact_current omits the SRH recombination inside
    # the contact nodes' own control volumes -- a real ~2e-4 imbalance for this forward-biased
    # diode that does NOT shrink with rel_tol (probed 1e-5 vs 1e-8: identical). The sharp
    # conservation test is the recombination-free bar (GATE B, 1e-12).
    g_c = bool(cc_d is not None and match == 0.0 and cons_d < 1e-3 and abs(cc_d["anode"]) > 0.0)
    ok = ok and g_c
    print("[cc] GATE C: diode extras == manual e+h sum * period_y (|d| = {:.1e}); conservation "
          "{:.1e}; I_fwd = {:+.3e} A -> {}".format(
              match, cons_d, cc_d["anode"] if cc_d else float("nan"),
              "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: equilibrium off-switch ----
    de = _bar_design("equilibrium")
    b3 = LayeredDevsimBuilder(de, mesh_name="cceq_m", device_name="cceq_d")
    cf3 = b3.solve(BiasPoint({"anode": 0.0}, "eq"))
    raw = extract_contact_currents(b3.device, depth_m=de.unit_cell.period_y_m)
    _teardown(b3)
    g_d = bool("contact_currents_A" not in cf3.extras and raw == {})
    ok = ok and g_d
    print("[cc] GATE D: equilibrium device -> extractor {{}} and NO extras key (extras = {}) "
          "-> {}".format(dict(cf3.extras), "PASS" if g_d else "FAIL"), flush=True)

    print("[cc] *** D1 CONTACT-CURRENT EXTRACTOR: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
