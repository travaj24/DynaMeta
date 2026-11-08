"""
D2c 3-D MESH GEOMETRY (v0.14.0) -- devsim_3d's box-method volumes and couplings are exact, so integrated
charge, sheet charge and the gate charge close on the 3-D builder. docs/audit/2026-11-03-devsim-3d-mesh.md.

THE DEFECT (pre-D2c): DEVSIM 2.11's 3-D ElementEdgeCouple sums the MAGNITUDES of each tetrahedron's
circumcentric dual pieces (src/GeomModels/TetrahedronEdgeCouple.cc); ElementNodeVolume = couple * L/6
feeds EdgeNodeVolume and NodeVolume. On the gmsh mesh's non-well-centred tets (42-57 %) the volumes and
couplings over-count. Stacked3DSpec.mesher = "tensor" (default since D2c) meshes a graded tensor grid
split into right-angled Kuhn tets, on which every piece is >= 0.

  GATE M1  geometry: sum NodeVolume, 2 sum ElementNodeVolume and sum EdgeCouple*EdgeLength/3 equal the
           exact region volume (< 1e-12) on five builder topologies (full gate, patch gate, multi-
           dielectric, a buried-oxide inclusion, a semiconductor inclusion under a patch); no EdgeCouple < 0.
  GATE M2  carrier-free sheet charge: both layers dielectric (Si 11.7 / SiO2 3.9, 30 / 8 nm), both contacts
           at 0 V, a sheet sigma on the interface (physics_traps.interface_node_area weights): the interface
           potential == sigma/(eps_si/d_si + eps_ox/d_ox) at EVERY interface node (< 1e-9).
  GATE M3  Gauss: the bipolar n-Si/SiO2 cap at Vg = 0.5 V (the D1c G2 fixture, trap-free). The
           NodeVolume-integrated semiconductor charge + the body contact charge balances the GEOMETRIC gate
           charge eps_ox (psi_gate - psi_s)/t_ox L^2 (exact for the charge-free planar oxide), and DEVSIM's
           own gate contact charge equals it (< 1e-6 each).
  GATE M4  lateral invariance of that laterally uniform cap: max - min psi_s over the interface < 1e-9 V.
  GATE M5  the D1c trap gates (validation/traps_oracles.py gate G): G2 3-D D_it shift == 1-D (5 %, the 3-D
           z-resolution) and G3 == the 1-D solve on the 3-D z-lines (1e-6).
  GATE M6  mesher='gmsh' (kept, user decision 2026-11-06): with exact_geometry=False it carries DEVSIM's
           defect (sum NodeVolume > 1.1 x) and warns; by default (exact_geometry=True) the builder's automatic
           correction makes sum NodeVolume = L^2 t_si to 1e-12 (devsim_geometry).
  GATE M7  (--base PATH) M1-M5 FAIL on a pre-D2c checkout (run in a subprocess on that tree).

Run:  python -m validation.devsim_3d_mesh [--base <pre-D2c checkout, e.g. a worktree at e46bd25>]
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TREE = HERE
if "--tree" in sys.argv:                       # measure-only subprocess on another checkout (gate M7)
    _TREE = os.path.abspath(sys.argv[sys.argv.index("--tree") + 1])
sys.path.insert(0, _TREE)

Q, EPS0 = 1.602176634e-19, 8.8541878128e-12
L, TSI, TOX = 12e-9, 30e-9, 8e-9


def _spec(**kw):
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    from dynameta.constants import M_E
    base = dict(semi_material="Si", oxide_material="SiO2", lateral_m=L, semi_thk_m=TSI, oxide_thk_m=TOX,
                n_bg_m3=1e23, eps_semi=11.7, eps_oxide=3.9, dos_mass_kg=1.08 * M_E, mobility_m2Vs=0.135,
                grid_n=(4, 4, 33), mesh_min_nm=0.5)
    base.update(kw)
    names = {f.name for f in dataclasses.fields(Stacked3DSpec)}
    for k in ("mesher", "exact_geometry"):     # a pre-D2c tree: its only mesher is gmsh, uncorrected
        if k in base and k not in names:
            base.pop(k)
    return Stacked3DSpec(**base)


def _clear():
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers import eq_registry as _R
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
        _R.clear(dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)


def _v(dev, r, n):
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    return np.asarray(ds.get_node_model_values(device=dev, region=r, name=n), dtype=float)


def geometry(mesher="tensor"):
    """M1: max over regions/topologies of the three volume identities, the min EdgeCouple, build times."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Inclusion3D
    hfo2 = [("HfO2", 4e-9, 18.0)]
    cases = {"full gate": {}, "patch 0.5": dict(gate_patch_frac=0.5), "multi-dielectric": dict(extra_dielectrics=hfo2),
             "buried-oxide inclusion": dict(extra_dielectrics=hfo2, inclusions=[Inclusion3D(
                 name="pill", material="HfO2", eps=18.0, in_layer="oxide", x_frac=0.5, y_frac=0.5)]),
             "semi inclusion + patch": dict(gate_patch_frac=0.6, inclusions=[Inclusion3D(
                 name="sx", material="SiO2", eps=3.9, in_layer="semi", x_frac=0.3, y_frac=0.5)])}
    out = {}
    tmp = tempfile.mkdtemp(prefix="d2c_geo_")
    for name, kw in cases.items():
        spec = _spec(physics="drift_diffusion", mesher=mesher, **kw)
        sol = Devsim3DEquilibrium(spec, device_name="m1", mesh_name="m1m", msh_path=os.path.join(tmp, "m1.msh"))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol.build_device()
        tb = time.time() - t0
        exact = {r: L * L * (zh - zl) * 1e-9 for (r, _m, _ro, zl, zh, _e) in spec.layer_stack_nm()}
        for inc in spec.inclusions:
            zl, zh = [(a, b) for (r, _m, _ro, a, b, _e) in spec.layer_stack_nm() if r == inc.in_layer][0]
            exact[inc.name] = inc.x_frac * inc.y_frac * L * L * (zh - zl) * 1e-9
            exact[inc.in_layer] -= exact[inc.name]
        worst, minec, nn = 0.0, np.inf, 0
        for r in ds.get_region_list(device="m1"):
            nv = _v("m1", r, "NodeVolume")
            ec = np.asarray(ds.get_edge_model_values(device="m1", region=r, name="EdgeCouple"))
            el = np.asarray(ds.get_edge_model_values(device="m1", region=r, name="EdgeLength"))
            env = np.asarray(ds.get_element_model_values(device="m1", region=r, name="ElementNodeVolume"))
            for val in (nv.sum(), 2.0 * env.sum(), (ec * el).sum() / 3.0):
                worst = max(worst, abs(val / exact[r] - 1.0))
            minec = min(minec, float(ec.min()))
            nn += nv.size
        out[name] = {"worst_rel": worst, "min_EdgeCouple": minec, "nodes": nn, "build_s": tb}
        sol.teardown()
    return out


def sheet_charge(mesher="tensor", sigma=1e-3):
    """M2: carrier-free 3-D sheet charge on the semi/oxide interface vs the series-capacitor closed form."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_traps as PT
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium
    spec = _spec(physics="drift_diffusion", mesher=mesher)
    tmp = tempfile.mkdtemp(prefix="d2c_sheet_")
    sol = Devsim3DEquilibrium(spec, device_name="m2", mesh_name="m2m", msh_path=os.path.join(tmp, "m2.msh"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol._build_mesh()
    _clear()
    ds.create_gmsh_mesh(mesh="m2m", file=sol.msh_path)
    for (name, mat, _r, _e) in sol._mesh_regions:
        ds.add_gmsh_region(mesh="m2m", gmsh_name=name, region=name, material=mat)
    ds.add_gmsh_contact(mesh="m2m", gmsh_name="gate", region=sol._top_diel, name="gate", material="metal")
    ds.add_gmsh_contact(mesh="m2m", gmsh_name="body", region="semi", name="body", material="metal")
    for (ifn, r0, r1) in sol._iface_pairs:
        ds.add_gmsh_interface(mesh="m2m", gmsh_name=ifn, region0=r0, region1=r1, name=ifn)
    ds.finalize_mesh(mesh="m2m")
    ds.create_device(mesh="m2m", device="m2")
    PE.setup_dielectric_region("m2", "semi", 11.7)
    PE.setup_dielectric_region("m2", "oxide", 3.9)
    PE.setup_interface("m2", "if_1")
    PE.setup_contact("m2", "gate")
    PE.setup_contact("m2", "body")
    area = PT.interface_node_area("m2", "semi", "if_1")
    nv = _v("m2", "semi", "NodeVolume")
    ds.node_solution(device="m2", region="semi", name="SheetW")
    ds.set_node_values(device="m2", region="semi", name="SheetW",
                       values=[float(a / v) if a > 0 else 0.0 for a, v in zip(area, nv)])
    # node charge model = -rho (the PE convention: q (n - N_D)); a +sigma sheet raises the potential
    ds.node_model(device="m2", region="semi", name="SheetQ", equation="-{!r} * SheetW".format(float(sigma)))
    ds.node_model(device="m2", region="semi", name="SheetQ:Potential", equation="0")
    _R.record_region_equation("m2", "semi", name="PotentialEquation", variable_name="Potential",
                              node_model="SheetQ", edge_model="PotentialEdgeFlux", variable_update="default")
    ds.solve(type="dc", absolute_error=1e-12, relative_error=1e-10, maximum_iterations=10)   # linear: 2 steps
    z, pot = _v("m2", "semi", "z"), _v("m2", "semi", "Potential")
    top = z >= z.max() - 1e-15
    v_exact = sigma / (11.7 * EPS0 / TSI + 3.9 * EPS0 / TOX)
    rel = pot[top] / v_exact - 1.0
    _clear()
    return {"V_exact": v_exact, "V_mean": float(pot[top].mean()), "worst_rel": float(np.max(np.abs(rel))),
            "mean_ratio": float(pot[top].mean() / v_exact), "total_area_rel": float(area.sum() / L ** 2 - 1)}


def gauss(mesher="tensor", vg=0.5):
    """M3/M4: bipolar cap; integrated semiconductor charge vs the geometric gate charge; lateral spread."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium
    from dynameta.sweep import BiasPoint
    spec = _spec(physics="bipolar_dd", n_i_m3=1e17, mobility_p_m2Vs=0.048, tau_srh_s=1e-7, mesher=mesher)
    tmp = tempfile.mkdtemp(prefix="d2c_gauss_")
    sol = Devsim3DEquilibrium(spec, device_name="m3", mesh_name="m3m", msh_path=os.path.join(tmp, "m3.msh"))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol.build_device()
        tb = time.time() - t0
        sol.solve(BiasPoint({"gate": vg}, "g"))
    ts = time.time() - t0 - tb
    z, pot = _v("m3", "semi", "z"), _v("m3", "semi", "Potential")
    rho = Q * (_v("m3", "semi", "Holes") - _v("m3", "semi", "Electrons") + _v("m3", "semi", "NetDoping"))
    q_semi = float((rho * _v("m3", "semi", "NodeVolume")).sum())
    zo, poto = _v("m3", "oxide", "z"), _v("m3", "oxide", "Potential")
    top = z >= z.max() - 1e-15
    psi_s, psi_g = float(pot[top].mean()), float(poto[zo >= zo.max() - 1e-15].mean())
    q_gate_geo = 3.9 * EPS0 * (psi_g - psi_s) / TOX * L ** 2
    q_gate_dev = float(ds.get_contact_charge(device="m3", contact="gate", equation="PotentialEquation"))
    q_body = float(ds.get_contact_charge(device="m3", contact="body", equation="PotentialEquation"))
    nn = z.size + zo.size
    sol.teardown()
    return {"Q_semi_C_m2": q_semi / L ** 2, "Q_gate_geo_C_m2": q_gate_geo / L ** 2, "Q_body_C_m2": q_body / L ** 2,
            "Q_gate_devsim_C_m2": q_gate_dev / L ** 2,
            "gauss_rel": abs(q_semi + q_gate_geo + q_body) / abs(q_gate_geo),
            "gate_charge_rel": abs(q_gate_dev / q_gate_geo - 1.0), "psi_s_spread_V": float(np.ptp(pot[top])),
            "nodes": nn, "build_s": tb, "solve_s": ts}


def trap_gate():
    """M5: the D1c gate G (G2 against the fine 1-D cap; G3 against the 1-D cap on the 3-D z-lines)."""
    from validation import traps_oracles as TO
    TO._load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s3, s1, dg = TO.gate_g()
    out = {"shift3d": s3, "shift1d": s1, "ratio": s3 / s1}
    if dg.get("same_z") is not None:
        a1z, b1z = dg["same_z"]
        out["rel_same_z"] = abs(s3 / (b1z - a1z) - 1.0)
        out["spread"] = max(dg["psi_s_spread"])
    return out


def measure_all():
    return {"M1": geometry(), "M2": sheet_charge(), "M3": gauss(), "M5": trap_gate()}


def verdicts(m):
    """(label, ok, detail) for M1-M5 from a measurement dict (shared by this tree and gate M7)."""
    g = m["M1"]
    w = max(v["worst_rel"] for v in g.values())
    mn = min(v["min_EdgeCouple"] for v in g.values())
    s, a, t = m["M2"], m["M3"], m["M5"]
    out = [("M1 volumes exact", w < 1e-12 and mn >= 0.0, "worst |sum/V - 1| {:.1e} over {} topologies, min "
            "EdgeCouple {:.1e}".format(w, len(g), mn)),
           ("M2 carrier-free sheet == sigma/(C1 + C2)", s["worst_rel"] < 1e-9,
            "V {:.6e} vs {:.6e} V, mean ratio {:.6f}, worst node {:.1e}".format(s["V_mean"], s["V_exact"],
                                                                                s["mean_ratio"], s["worst_rel"])),
           ("M3 Gauss: Q_semi + Q_body + Q_gate(geom) = 0", a["gauss_rel"] < 1e-6 and a["gate_charge_rel"] < 1e-6,
            "Q_semi {:.6e}, Q_body {:.6e}, Q_gate(geom) {:.6e} C/m^2 -> residual {:.1e}; DEVSIM gate charge "
            "{:.6e} (rel {:.1e})".format(a["Q_semi_C_m2"], a["Q_body_C_m2"], a["Q_gate_geo_C_m2"], a["gauss_rel"],
                                         a["Q_gate_devsim_C_m2"], a["gate_charge_rel"])),
           ("M4 lateral invariance", a["psi_s_spread_V"] < 1e-9, "psi_s spread {:.1e} V".format(a["psi_s_spread_V"])),
           ("M5 D1c G2: 3-D D_it shift == 1-D (5 %)", abs(t["ratio"] - 1.0) < 0.05,
            "{:+.6f} vs {:+.6f} V, ratio {:.4f}".format(t["shift3d"], t["shift1d"], t["ratio"]))]
    if "rel_same_z" in t:
        out.append(("M5 D1c G3: == 1-D on the 3-D z-lines", t["rel_same_z"] < 1e-6 and t["spread"] < 1e-9,
                    "rel {:.1e}, spread {:.1e} V".format(t["rel_same_z"], t["spread"])))
    else:
        out.append(("M5 D1c G3: == 1-D on the 3-D z-lines", False, "no tensor z-lines on this tree"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="a pre-D2c checkout (gate M7)")
    ap.add_argument("--tree", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.tree:                                  # measure-only child (gate M7): one JSON line
        print("D2C_JSON " + json.dumps(measure_all()), flush=True)
        return 0
    import importlib.util
    from validation._harness import Harness
    H = Harness("m3d", "D2c 3-D MESH: exact box-method geometry on the devsim_3d builder")
    H.require(importlib.util.find_spec("devsim") is not None, "devsim not installed")
    m = measure_all()
    for name, d in m["M1"].items():
        print("[m3d]   M1 {:24s} worst {:.1e}  min EC {:.1e}  {:6d} nodes  build {:.2f} s".format(
            name, d["worst_rel"], d["min_EdgeCouple"], d["nodes"], d["build_s"]), flush=True)
    print("[m3d]   M3 cap: {} nodes, build {:.2f} s, solve {:.2f} s".format(
        m["M3"]["nodes"], m["M3"]["build_s"], m["M3"]["solve_s"]), flush=True)
    for (label, ok, det) in verdicts(m):
        H.gate(label, ok, det)

    def legacy(exact):
        from dynameta.carriers.devsim_3d import Devsim3DEquilibrium
        from dynameta.warning_types import UnreliableResultWarning
        spec = _spec(physics="drift_diffusion", mesher="gmsh", exact_geometry=exact)
        sol = Devsim3DEquilibrium(spec, device_name="m6", mesh_name="m6m",
                                  msh_path=os.path.join(tempfile.mkdtemp(prefix="d2c_leg_"), "m6.msh"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sol.build_device()
        r = _v("m6", "semi", "NodeVolume").sum() / (L * L * TSI)
        msgs = [str(x.message) for x in w if issubclass(x.category, UnreliableResultWarning)]
        sol.teardown()
        if exact:
            ok = abs(r - 1.0) < 1e-12 and not any("exact_geometry=False" in m for m in msgs)
        else:
            ok = r > 1.1 and any("exact_geometry=False" in m for m in msgs)
        return ok, "sum NodeVolume / V(semi) = {:.15f}, UnreliableResultWarnings {}".format(r, len(msgs))
    H.run("M6a mesher='gmsh', exact_geometry=False keeps DEVSIM's defect, warns", lambda: legacy(False))
    H.run("M6b mesher='gmsh' default: geometry corrected (sum NodeVolume = V to 1e-12)", lambda: legacy(True))

    if a.base:
        def base_fails():
            env = dict(os.environ, PYTHONPATH=os.path.abspath(a.base))
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--tree", os.path.abspath(a.base)],
                               cwd=os.path.abspath(a.base), env=env, capture_output=True, text=True, timeout=3600)
            line = [x for x in r.stdout.splitlines() if x.startswith("D2C_JSON ")]
            if not line:
                return False, "child rc {}: {}".format(r.returncode, (r.stderr or r.stdout)[-300:])
            vb = verdicts(json.loads(line[0][9:]))
            for (label, ok, det) in vb:
                print("[m3d]   base {}: {} -> {}".format(label, det, "pass" if ok else "FAIL"), flush=True)
            return all(not ok for (_l, ok, _d) in vb), "{}/{} gates FAIL on {}".format(
                sum(not ok for (_l, ok, _d) in vb), len(vb), a.base)
        H.run("M7 every gate FAILS on the pre-D2c tree", base_fails)
    else:
        H.skip("M7 pre-D2c tree fails", "pass --base <checkout>")
    return H.summary("D2c 3-D MESH")


if __name__ == "__main__":
    raise SystemExit(main())
