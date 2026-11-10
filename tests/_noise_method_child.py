"""Child-process driver for tests/test_devsim_noise_method.py (LayeredDevsimBuilder.noise_spectrum, v0.15
plan 4.2.5). DEVSIM circuits are process-global and cannot be removed in-process (tests/_bipolar_calllog.py),
so every scenario that makes a gate circuit-driven runs here, one scenario per process, and prints one line
"NOISE_METHOD_RESULT <json>". Floats travel as JSON numbers (repr: exact round trip); arrays as sha256
digests of their float64 bytes where bitwise identity is the gate.

    python tests/_noise_method_child.py <scenario>

Scenarios: plain_1V, hand_ref, wrap_ref, fresh1_ref, seq_ref, seq_xp (the reference gated-ITO modulator,
validation/_reference_device; seq_xp on an extended-precision builder, nightly)
and hand_te, wrap_te (a gated bipolar n-GaAs / N-Al0.3Ga0.7As heterostack with a thermionic interface).
"hand_*" is the documented hand recipe -- solve, set_ssac_gate, circuit_alter, a library DC re-settle
(dc_solve.solve_dc with the builder's branch and tolerances), noise_analysis.noise_spectrum -- written out
here call by call; "wrap_*" the one-call method.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.devsim_env import import_devsim    # noqa: E402

ds = import_devsim()

from dynameta.carriers import noise_analysis as NA           # noqa: E402
from dynameta.carriers.ac_analysis import ssac_admittance    # noqa: E402
from dynameta.carriers.dc_solve import solve_dc              # noqa: E402
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder, OhmicContactNoiseError   # noqa: E402
from dynameta.sweep import BiasPoint                         # noqa: E402

FF_REF = [1e9, 1e11, 1e13, 1e15]             # validation/devsim_noise.py FREQS_G (gate G)
FF_TE = [1e9, 1e11]
GATE = "top_contact"
TE_ALPHAS = {"gaas": (1e-4, 3e-5), "algaas": (2e-5, None)}    # Hooge 1994 lattice value and two others


def digest(sn) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(sn.S, dtype=np.float64).tobytes())
    for d in (sn.by_mechanism, sn.by_region):
        for k in sorted(d):
            h.update(k.encode())
            h.update(np.asarray(d[k], dtype=np.float64).tobytes())
    for d in (sn.by_region_mechanism, sn.by_interface):
        for k in sorted(d):
            for m in sorted(d[k]):
                h.update((k + ":" + m).encode())
                h.update(np.asarray(d[k][m], dtype=np.float64).tobytes())
    return h.hexdigest()


def state(dev):
    """Every equation variable of every region + the circuit dcop values (bitwise)."""
    out = {}
    for r in ds.get_region_list(device=dev):
        for eq in ds.get_equation_list(device=dev, region=r):
            var = ds.get_equation_command(device=dev, region=r, name=eq)["variable_name"]
            out["{}:{}".format(r, var)] = [float(v) for v in ds.get_node_model_values(device=dev, region=r, name=var)]
    if "dcop" in ds.get_circuit_solution_list():
        for nm in ds.get_circuit_node_list():
            out["circuit:" + nm] = [float(ds.get_circuit_node_value(node=nm, solution="dcop"))]
    return out


def hand(b, ff, *, bipolar):
    """The hand recipe after b.solve(...): circuit-drive the gate, set its source, re-settle, noise."""
    b.set_ssac_gate(GATE, source_name="V1")
    ds.circuit_alter(name="V1", value=0.0)
    if bipolar:
        solve_dc(b.device, method="newton", abs_tol=None, rel_tol=1e-5, max_iter=60)
    else:
        solve_dc(b.device, method=b.design.mesh_2d.dc_method, abs_tol=None, rel_tol=1e-5, max_iter=60,
                 semiconductor_regions=sorted(b._dd_regions))
    sn = NA.noise_spectrum(b.device, ff, output_node="V1.I")
    f, C, G = ssac_admittance(ff, source_name="V1")
    return {"digest": digest(sn), "S": [float(v) for v in sn.S], "G": [float(v) for v in G],
            "mechanisms": sorted(sn.by_mechanism)}


def te_design():
    """Gated bipolar heterostack: Al-Nd mirror (bottom gate) / 7 nm HfO2 / 20 nm n-GaAs (grounded on both
    x edges) / 20 nm N-Al0.3Ga0.7As / 7 nm HfO2 / Au patch (top gate), 60 nm cell. Boltzmann statistics
    (carrier_statistics='boltzmann': the diffusion source is Nyquist-exact), A* x 1e-3 so the thermionic
    interface carries most of the gate's dissipation at 1 GHz."""
    from validation import _reference_device as RD
    from validation.devsim_heterojunction import ALGAAS, GAAS, richardson, transport
    from dynameta.geometry import Design, Electrode, Inclusion, Layer, Stack, UnitCell, centered_square
    from dynameta.materials import ConstantOptical, Material
    reg = RD.build_materials("drift_diffusion")
    for nm, mat, (an, ap) in (("GaAs_n", GAAS, TE_ALPHAS["gaas"]), ("AlGaAs_n", ALGAAS, TE_ALPHAS["algaas"])):
        tm = dataclasses.replace(transport(mat, n_bg=1e23, physics="bipolar_dd"), carrier_statistics="boltzmann",
                                 richardson_n_Am2K2=1e-3 * richardson(0.092),
                                 richardson_p_Am2K2=1e-3 * richardson(0.55), hooge_alpha_n=an, hooge_alpha_p=ap)
        reg.add(Material(nm, ConstantOptical(12.0 + 0j), transport=tm))
    cell = UnitCell.square(60e-9)
    layers = [Layer("mirror", 70e-9, "Al-Nd"), Layer("lower_hfo2", 7e-9, "HfO2"), Layer("gaas", 20e-9, "GaAs_n"),
              Layer("algaas", 20e-9, "AlGaAs_n"), Layer("upper_hfo2", 7e-9, "HfO2"),
              Layer("patch", 50e-9, "air", inclusions=[Inclusion(centered_square(cell, 30e-9), "Au")])]
    el = [Electrode("bot_contact", "mirror", "full", role="biased"),
          Electrode(GATE, "patch", centered_square(cell, 30e-9), role="biased"),
          Electrode("gnd_left", "gaas", "x_lo", role="ground", fixed_voltage_V=0.0),
          Electrode("gnd_right", "gaas", "x_hi", role="ground", fixed_voltage_V=0.0)]
    return Design(name="te_gate", unit_cell=cell, stack=Stack(layers=layers, superstrate_material="air",
                                                               substrate_material="Si"),
                  electrodes=el, materials=reg)


def ref_builder(tag="nm"):
    from validation._reference_device import build_reference_modulator
    return LayeredDevsimBuilder(build_reference_modulator("drift_diffusion"), mesh_name=tag + "_m",
                                device_name=tag + "_d")


def run(scenario):
    res = {}
    if scenario == "plain_1V":                    # the W5 reference: a fresh builder solved at 1 V, no circuit
        f = ref_builder()
        f.solve(BiasPoint({GATE: 1.0}, "g1"))
        res["fresh_1V"] = state(f.device)
    elif scenario == "hand_ref":
        b = ref_builder()
        b.solve(BiasPoint({GATE: 0.0}, "g0"))
        res.update(hand(b, FF_REF, bipolar=False))
        res["period"] = float(b.design.unit_cell.period_y_m)
    elif scenario == "wrap_ref":
        b = ref_builder()
        b.solve(BiasPoint({GATE: 0.0}, "g0"))
        before = state(b.device)
        w0 = b.noise_spectrum(FF_REF, gate=GATE, vg=0.0, per_cell=False)
        after = state(b.device)
        res["restored"] = {k: v for k, v in after.items() if not k.startswith("circuit:")} == before
        res["circuit_after"] = {k: v for k, v in after.items() if k.startswith("circuit:")}
        res["w0"] = {"digest": digest(w0), "S": [float(v) for v in w0.S], "units": w0.units,
                     "depth": w0.depth_m, "mechanisms": sorted(w0.by_mechanism)}
        wc = b.noise_spectrum(FF_REF, gate=GATE, maps=True)          # vg defaults to the last bias, per cell
        res["wc"] = {"S": [float(v) for v in wc.S], "units": wc.units, "depth": wc.depth_m, "vg": wc.vg_V,
                     "voltages": wc.voltages,
                     "map_sum": [float(v) for v in sum(wc.maps[r][m].sum(axis=1) for r in wc.maps
                                                       for m in wc.maps[r] if m not in ("x", "y", "z"))],
                     "breakdown_exact": all(np.array_equal(wc.by_mechanism[m], w0.by_mechanism[m] * wc.depth_m)
                                            for m in w0.by_mechanism)
                     and all(np.array_equal(wc.by_region[r], w0.by_region[r] * wc.depth_m) for r in w0.by_region)}
        w1 = b.noise_spectrum(FF_REF, gate=GATE, vg=1.0, per_cell=False)
        res["w1"] = {"digest": digest(w1), "S": [float(v) for v in w1.S]}
        w2 = b.noise_spectrum(FF_REF, gate=GATE, vg=0.0, per_cell=False)
        res["w2"] = {"digest": digest(w2)}
        hm = b.noise_spectrum(FF_REF, gate=GATE, vg=0.0, per_cell=False, hooge="material")
        res["hooge_material_none"] = {"digest": digest(hm), "mechanisms": sorted(hm.by_mechanism)}
        tn = b.terminal_noise(FF_REF, gate=GATE)
        res["tn"] = {"S_I": [float(v) for v in tn.S_I], "ReY": [float(v) for v in tn.Y.real],
                     "units": tn.units, "S_V_ok": bool(np.array_equal(tn.S_V, tn.S_I / np.abs(tn.Y) ** 2))}
        # refusals (W6), none of which may touch the device
        try:
            b.noise_spectrum(FF_REF, gate="ito_gnd_left")
            res["ohmic"] = "returned"
        except OhmicContactNoiseError as e:
            res["ohmic"] = "OhmicContactNoiseError:" + str(e)
        try:
            b.noise_spectrum(FF_REF, gate=GATE, source_name="V2")
            res["second_source"] = "returned"
        except ValueError as e:
            res["second_source"] = "ValueError:" + str(e)
        try:
            b.set_ssac_gate("bot_contact", source_name="V1")
            res["name_clash"] = "returned"
        except ValueError as e:
            res["name_clash"] = "ValueError:" + str(e)
        res["bot_contact_eqs"] = list(ds.get_contact_equation_list(device=b.device, contact="bot_contact"))
        try:
            b.noise_spectrum(FF_REF, gate=GATE, hooge="materials")
            res["bad_hooge"] = "returned"
        except ValueError as e:
            res["bad_hooge"] = "ValueError:" + str(e)
        res["after_refusals"] = {k: v for k, v in state(b.device).items() if not k.startswith("circuit:")} == before
        cf = b.solve(BiasPoint({GATE: 1.0}, "g1"))                   # W5: solve() after the method
        res["solve_1V"] = state(b.device)
        res["solve_1V_voltages"] = dict(cf.voltages)
    elif scenario == "fresh1_ref":
        b = ref_builder()
        b.solve(BiasPoint({GATE: 0.0}, "g0"))
        w1 = b.noise_spectrum(FF_REF, gate=GATE, vg=1.0, per_cell=False)
        res["w1"] = {"digest": digest(w1), "S": [float(v) for v in w1.S]}
    elif scenario == "seq_ref":
        # W4 on a NON-trivial base state: the bottom gate held at +0.5 V, so the top gate's 0 V state is not
        # the exactly representable flat band (psi = 0, n = n_bg) that any Newton lands on bit for bit
        b = ref_builder()
        b.solve(BiasPoint({GATE: 0.0, "bot_contact": 0.5}, "b05"))
        res["seq"] = [digest(b.noise_spectrum(FF_REF, gate=GATE, vg=v, per_cell=False)) for v in (0.0, 1.0, 0.0)]
        # adversarial: the same sequence WITHOUT the state restore (what a method that leaves the device at vg
        # would do): settle at 0, ramp to +1 and back in the method's 0.25 V steps, noise each time
        seq = []
        for path in ((0.0,), (0.25, 0.5, 0.75, 1.0), (0.75, 0.5, 0.25, 0.0)):
            for v in path:
                b._drive(GATE, v)
                b._resettle(1e-5, None, 60)
            seq.append(NA.noise_spectrum(b.device, FF_REF, output_node="V1.I"))
        res["no_restore"] = {"digests": [digest(x) for x in seq],
                             "S_third_over_first": [float(v) for v in seq[2].S / seq[0].S]}
    elif scenario == "seq_xp":
        # P2 W7 (v0.15 follow-up; nightly): seq_ref's 0 / +1 / 0 V sequence on an EXTENDED-PRECISION builder.
        # The restore writes the double-rounded state back, so the bitwise repeat is not implied by seq_ref.
        import warnings
        from dynameta.carriers import extended_precision as XP
        from validation._reference_device import build_reference_modulator
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            b = LayeredDevsimBuilder(build_reference_modulator("drift_diffusion"), mesh_name="nx_m",
                                     device_name="nx_d", extended_precision=True)
            cf = b.solve(BiasPoint({GATE: 0.0, "bot_contact": 0.5}, "b05"))
            n_build = len(rec)
            res["seq"] = [digest(b.noise_spectrum(FF_REF, gate=GATE, vg=v, per_cell=False)) for v in (0.0, 1.0, 0.0)]
        res["extended"] = [cf.extras.get("extended_precision"), bool(XP.is_enabled(b.device))]
        res["warnings_build"] = [(w.category.__name__, str(w.message)[:120]) for w in rec[:n_build]]
        res["warnings_calls"] = [(w.category.__name__, str(w.message)[:120]) for w in rec[n_build:]]
    elif scenario == "hand_te":
        b = LayeredDevsimBuilder(te_design(), mesh_name="nm_m", device_name="nm_d")
        b.solve(BiasPoint({}, "eq"))
        res.update(hand(b, FF_TE, bipolar=True))
        res["ifaces"] = b.heterointerfaces()
    elif scenario == "wrap_te":
        b = LayeredDevsimBuilder(te_design(), mesh_name="nm_m", device_name="nm_d")
        b.solve(BiasPoint({}, "eq"))
        w = b.noise_spectrum(FF_TE, gate=GATE, per_cell=False)
        res["w"] = {"digest": digest(w), "mechanisms": sorted(w.by_mechanism),
                    "by_interface": {i: {m: [float(x) for x in v] for m, v in d.items()}
                                     for i, d in w.by_interface.items()},
                    "S": [float(v) for v in w.S]}
        tn = b.terminal_noise(FF_TE, gate=GATE)
        res["tn"] = {"S_I": [float(v) for v in tn.S_I], "ReY": [float(v) for v in tn.Y.real]}
        lit = b.terminal_noise(FF_TE, gate=GATE, mechanisms=("diffusion_n", "diffusion_p", "gr"))
        res["tn_plan_literal_default"] = {"S_I": [float(v) for v in lit.S_I], "ReY": [float(v) for v in lit.Y.real]}
        hm = b.noise_spectrum(FF_TE, gate=GATE, per_cell=False, hooge="material")
        hx = b.noise_spectrum(FF_TE, gate=GATE, per_cell=False, hooge=dict(TE_ALPHAS))
        res["hooge"] = {"material": digest(hm), "explicit": digest(hx),
                        "keys": sorted(m for m in hm.by_mechanism if m.startswith("hooge")),
                        "algaas_p_zero": bool(np.all(hm.by_region_mechanism["algaas"]["hooge_p"] == 0.0)),
                        "gaas_n": [float(v) for v in hm.by_region_mechanism["gaas"]["hooge_n"]]}
    else:
        raise SystemExit("unknown scenario " + scenario)
    return res


if __name__ == "__main__":
    out = run(sys.argv[1])
    sys.stdout.flush()
    print("NOISE_METHOD_RESULT " + json.dumps(out))
