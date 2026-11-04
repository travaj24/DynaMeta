"""Unit coverage for the LayeredDevsimBuilder full-edge-ground region planning (the gated-DD fix):
a drift-diffusion semiconductor layer with edge GROUND electrodes gets a thin adjacent edge-metal
strip carved at each grounded edge (so the ground is a region-region interface -> full-line node
capture -> gated DD converges), while the equilibrium path is left untouched. This tests the PURE
region-spec planning (_region_specs); the full build+solve convergence lives in
validation/gated_dd_builder.py. Importing the builder needs DEVSIM, so skip when absent (CI).
Run: python -m pytest tests/test_gated_builder.py -q
"""
import numpy as np
import pytest


def _design(physics):
    from dynameta.materials import (Material, MaterialRegistry, ConstantOptical,
                                    TransportModel, M_E)
    from dynameta.geometry import UnitCell, Stack, Layer, Electrode, Design
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("Au", ConstantOptical(-100 + 8j), is_metal=True))
    mob = (None if physics == "equilibrium"
           else (lambda n: np.full_like(np.asarray(n, float), 30e-4)))
    reg.add(Material("semi", optical=ConstantOptical(4.0 + 0j),
                     transport=TransportModel(n_bg_m3=4e26, eps_static=9.5,
                                              dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E,
                                              physics=physics, mobility_m2Vs_of_n_m3=mob)))
    cell = UnitCell.square(200e-9)
    layers = [Layer("metalpad", 20e-9, "Au"), Layer("ito", 12e-9, "semi")]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    electrodes = [Electrode("gl", "ito", "x_lo", role="ground"),
                  Electrode("gr", "ito", "x_hi", role="ground")]
    return Design(name="g", unit_cell=cell, stack=stack, electrodes=electrodes, materials=reg)


def test_dd_edge_grounds_carve_edge_metal_strips():
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder, _EDGE_METAL_W_M
    b = LayeredDevsimBuilder(_design("drift_diffusion"))
    w, P = _EDGE_METAL_W_M, 200e-9
    # both grounded edges (x_lo and x_hi) get an inert edge-metal strip
    egnd = {s.name: s for s in b._specs if "egnd" in s.name}
    assert sorted(egnd) == ["ito_egnd_hi", "ito_egnd_lo"]
    assert all(s.role == "metal" for s in egnd.values())
    assert abs(egnd["ito_egnd_lo"].x_lo) < 1e-15 and abs(egnd["ito_egnd_lo"].x_hi - w) < 1e-15
    assert abs(egnd["ito_egnd_hi"].x_lo - (P - w)) < 1e-15 and abs(egnd["ito_egnd_hi"].x_hi - P) < 1e-15
    # the semiconductor is narrowed to [w, P-w] (the ground sits at the interfaces)
    ito = next(s for s in b._specs if s.name == "ito")
    assert abs(ito.x_lo - w) < 1e-15 and abs(ito.x_hi - (P - w)) < 1e-15
    assert b._dd_full_edge_grounds() == {"ito": {"x_lo", "x_hi"}}


def test_equilibrium_edge_ground_not_carved():
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    b = LayeredDevsimBuilder(_design("equilibrium"))
    assert not any("egnd" in s.name for s in b._specs)          # equilibrium path untouched
    ito = next(s for s in b._specs if s.name == "ito")
    assert ito.x_lo == 0.0 and abs(ito.x_hi - 200e-9) < 1e-15
    assert b._dd_full_edge_grounds() == {}


def test_eq_registry_forget_is_targeted():
    """forget(name, loc) drops ONLY the matching recorded entry (and forget(name) drops all by
    name) without touching the live equation -- the bookkeeping half of repointing a contact."""
    R = pytest.importorskip("dynameta.carriers.eq_registry")
    dev = "utest_forget_dev"
    R._REG[dev] = [
        {"scope": "region", "loc": "ito", "name": "PotentialEquation", "kwargs": {}},
        {"scope": "contact", "loc": "gate", "name": "PotentialEquation", "kwargs": {}},
        {"scope": "contact", "loc": "gnd", "name": "PotentialEquation", "kwargs": {}},
        {"scope": "contact", "loc": "gate", "name": "ElectronContinuityEquation", "kwargs": {}},
    ]
    try:
        R.forget(dev, "PotentialEquation", loc="gate")          # only the gate's Potential record
        present = {(e["scope"], e["loc"], e["name"]) for e in R._REG[dev]}
        assert ("contact", "gate", "PotentialEquation") not in present
        assert ("region", "ito", "PotentialEquation") in present        # region kept
        assert ("contact", "gnd", "PotentialEquation") in present       # other contact kept
        assert ("contact", "gate", "ElectronContinuityEquation") in present  # other name kept
        R.forget(dev, "PotentialEquation")                      # no loc -> drop all by name
        names = {(e["loc"], e["name"]) for e in R._REG[dev]}
        assert names == {("gate", "ElectronContinuityEquation")}
        R.forget("no_such_device", "PotentialEquation")         # missing device is a no-op
    finally:
        R.clear(dev)


def test_set_ssac_gate_requires_built_device():
    """set_ssac_gate raises (does not silently no-op) if called before the device is built."""
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    b = LayeredDevsimBuilder(_design("drift_diffusion"), mesh_name="ut_ssac_m",
                             device_name="ut_ssac_d")
    assert not b._built
    with pytest.raises(RuntimeError):
        b.set_ssac_gate("gl")


def test_2d_builder_refuses_lateral_isolation_cases():
    # audit C2-3: non-ambient background WITH inclusions (and laterally touching
    # inclusions) get NO lateral interface in the 2D builder -- the inclusion would be
    # electrostatically isolated silently; must raise (the 3D builder wires these)
    import pytest
    pytest.importorskip("devsim")              # devsim_layered imports devsim at module load
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hfo2", ConstantOptical(4.0 + 0j)))
    reg.add(Material("ito", ConstantOptical(3.9 + 0j),
                     transport=TransportModel(n_bg_m3=4e26, eps_static=9.5,
                                              dos_mass_kg_of_n_m3=lambda n: 3.2e-31)))
    P = 370e-9
    bar = Inclusion(shape=Rectangle(P / 2, P / 4, 0.4 * P, P), material="ito")
    d = Design(name="t", unit_cell=UnitCell.square(P),
               stack=Stack(layers=[Layer("mix", 10e-9, "hfo2", inclusions=[bar])],
                           superstrate_material="air", substrate_material="air"),
               electrodes=[], materials=reg)
    with pytest.raises(NotImplementedError, match="lateral"):
        LayeredDevsimBuilder(d)
    # two x-touching inclusions in an AMBIENT layer likewise raise
    b1 = Inclusion(shape=Rectangle(P / 4, P / 4, 0.25 * P, P), material="ito")
    b2 = Inclusion(shape=Rectangle(P / 2, P / 4, 0.25 * P, P), material="ito")
    d2 = Design(name="t2", unit_cell=UnitCell.square(P),
                stack=Stack(layers=[Layer("pair", 10e-9, "air", inclusions=[b1, b2])],
                            superstrate_material="air", substrate_material="air"),
                electrodes=[], materials=reg)
    with pytest.raises(NotImplementedError, match="DECOUPLED"):
        LayeredDevsimBuilder(d2)


def test_bipolar3d_expr_requires_body_doping():
    # audit C2-1: net_doping_expr makes the (acceptor, n_bg_m3) scalars dead as doping --
    # phi_bi derived from them silently mis-references the gate (+0.714 V at Vg=0 in the
    # audit probe); the spec-level contract now requires the signed body-side doping
    pytest.importorskip("devsim")              # devsim_3d imports devsim at module load
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    spec = Stacked3DSpec(physics="bipolar_dd", n_i_m3=1e17, net_doping_expr="-1.0e23")
    assert spec.body_net_doping_m3 is None                     # guard fires at solve()


# ---- v0.14.0 blocker (2026-11-04): solve() after set_ssac_gate ---------------------------------------
# MEASURED on 6595bde before the fix (reference modulator, top_contact): solve(1 V) -> set_ssac_gate ->
# circuit_alter(V1, 1 V) + DC solve -> builder.solve(2 V) returned the 1 V state -- every node Potential
# equal to it to 1.1e-16 V, and 1.0 V (upper_al2o3) / 0.80 V (upper_hfo2) / 0.12 V (ito) off a fresh
# builder solved at 2 V. The ramp wrote only the `top_contact_bias` parameter, which the circuit-driven
# Dirichlet no longer reads. Fixed: solve() drives the electrode's circuit source (LayeredDevsimBuilder.
# _drive). After the fix the two states agree to <= 6e-16 (relative, Electrons; absolute V, Potential)
# for both physics, so the 1e-12 bound below is round-off with margin, not a fitted tolerance. Run in a
# CHILD process: the DEVSIM circuit is process-global, and a leftover V1/vac would leak into every
# later devsim test on the same worker.
_SSAC_CHILD = r'''
import json, sys
import numpy as np
from dynameta.carriers.devsim_env import import_devsim
ds = import_devsim()
from validation._reference_device import build_reference_modulator
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
from dynameta.carriers import ac_analysis as AC, eq_registry as R, devsim_3d as D3
from dynameta.sweep import BiasPoint
physics, GATE = sys.argv[1], "top_contact"

def state(dev):
    out = {}
    for r in ds.get_region_list(device=dev):
        for v in ("Potential", "Electrons"):
            try:
                out[r + ":" + v] = np.array(ds.get_node_model_values(device=dev, region=r, name=v))
            except Exception:
                pass                      # dielectric regions carry no Electrons
    return out

def wipe():
    for d in list(ds.get_device_list()):
        ds.delete_device(device=d)
        R.clear(d)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    try:
        ds.delete_circuit()
    except Exception:
        pass

res = {}
f = LayeredDevsimBuilder(build_reference_modulator(physics), mesh_name="f_m", device_name="f_d")
f.solve(BiasPoint({GATE: 2.0}, "g2"))
ref = state("f_d")
wipe()
b = LayeredDevsimBuilder(build_reference_modulator(physics), mesh_name="a_m", device_name="a_d")
b.solve(BiasPoint({GATE: 1.0}, "g1"))
b.set_ssac_gate(GATE, source_name="V1")
ds.circuit_alter(name="V1", value=1.0)
ds.solve(type="dc", solver_type="direct", absolute_error=1e18, relative_error=1e-5, maximum_iterations=60)
cf = b.solve(BiasPoint({GATE: 2.0}, "g2"))
got = state("a_d")
res["vac"] = float(ds.get_circuit_node_value(node="vac", solution="dcop"))
res["dev"] = {k: float(np.max(np.abs(got[k] - ref[k])) /
                        (np.max(np.abs(ref[k])) if k.endswith("Electrons") else 1.0)) for k in ref}
res["voltages"] = dict(cf.voltages)
wipe()

# a contact repointed circuit-driven OUTSIDE set_ssac_gate: solve() cannot know its source -> refuse
r = LayeredDevsimBuilder(build_reference_modulator(physics), mesh_name="r_m", device_name="r_d")
r.solve(BiasPoint({GATE: 1.0}, "g1"))
ds.delete_contact_equation(device="r_d", contact=GATE, name="PotentialEquation")
R.forget("r_d", "PotentialEquation", loc=GATE)
AC.setup_circuit_contact("r_d", GATE, source_name="V1")
try:
    r.solve(BiasPoint({GATE: 2.0}, "g2"))
    res["raw2d"] = "returned"
except RuntimeError as e:
    res["raw2d"] = str(e)
wipe()

# the 3-D builder has no set_ssac_gate: a circuit-driven gate must refuse, not go stale
import validation.devsim_3d_mesh as V3
s3 = D3.Devsim3DEquilibrium(V3._spec(physics="equilibrium"), device_name="c3d", mesh_name="c3m")
s3.solve(BiasPoint({"gate": 0.5}, "g"))
ds.delete_contact_equation(device="c3d", contact="gate", name="PotentialEquation")
R.forget("c3d", "PotentialEquation", loc="gate")
AC.setup_circuit_contact("c3d", "gate", source_name="V1")
try:
    s3.solve(BiasPoint({"gate": 1.0}, "g"))
    res["raw3d"] = "returned"
except RuntimeError as e:
    res["raw3d"] = str(e)
wipe()
print("SSAC_RESULT " + json.dumps(res))
'''


@pytest.mark.parametrize("physics", ["equilibrium", "drift_diffusion"])
def test_solve_after_set_ssac_gate_drives_the_circuit_source(physics):
    """builder.solve(new bias) after set_ssac_gate equals a fresh builder solved at that bias (to
    round-off), and the circuit node follows; a contact made circuit-driven behind the builder's back
    (2-D raw setup_circuit_contact, or ANY circuit-driven contact on the 3-D builder, which has no
    set_ssac_gate) is refused instead of silently returning the previous operating point.
    On 6595bde: dev = 1.0 V, vac = 1.0 V (the stale 1 V state), and both raw solves 'returned'."""
    pytest.importorskip("devsim")
    import json
    import os
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [repo] + [p for p in sys.path if isinstance(p, str) and p]))
    p = subprocess.run([sys.executable, "-W", "error", "-c", _SSAC_CHILD, physics], cwd=repo, env=env,
                       capture_output=True, text=True, timeout=900)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("SSAC_RESULT ")]
    assert p.returncode == 0 and lines, (p.returncode, p.stdout[-1500:], p.stderr[-3000:])
    res = json.loads(lines[-1][len("SSAC_RESULT "):])
    assert abs(res["vac"] - 2.0) < 1e-12, res
    assert res["voltages"] == {"top_contact": 2.0}
    assert len(res["dev"]) >= 6 and max(res["dev"].values()) < 1e-12, res["dev"]
    assert "not repointed with set_ssac_gate" in res["raw2d"], res["raw2d"]
    assert "circuit-driven" in res["raw3d"] and "stale" in res["raw3d"], res["raw3d"]
