"""Child-process scenarios for tests/test_devsim_gate_leakage.py (DynaMeta v0.16, gate leakage / lateral Fermi field).

WHY A CHILD. The gate-leakage sink is a custom term (carriers/custom_terms.py): its DEVSIM dispatcher can never be
unregistered, so every scenario that installs one runs in a fresh interpreter and reports plain JSON (floats
round-trip exactly through repr).

    python tests/_gate_leakage_child.py <scenario> <out.json> [json options]

Devices: the generic gated 6 nm degenerate film of tests/_wiring_fixture.py (6 nm oxB eps 20 next to the film + 2 nm
oxA eps 9 on both faces, metal gates "bot" and "top"), with an edge electrode on the film, at the cell length and lateral spacing a scenario
asks for. This module only builds, solves and reads arrays; the tests derive every oracle.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

ABS_TOL = 1e-14                       # volts (a Potential / Fermi-potential device: absolute test)
MU = 3.0e-3                           # m^2/Vs film mobility of the scenarios


def _ds():
    from dynameta.carriers.devsim_env import import_devsim
    return import_devsim()


def _design(P=40e-9, dx=4e-9, edges=None, mats=None, **kw):
    import _wiring_fixture as W
    from dynameta.geometry import UnitCell
    d = W.film_design(edges=[W.edge()] if edges is None else edges, mats=mats,
                      x_lines=dict(x_spacing_edge_m=dx, x_spacing_feature_edge_m=dx, x_spacing_feature_mid_m=dx), **kw)
    return dataclasses.replace(d, unit_cell=UnitCell(period_x_m=P, period_y_m=40e-9))


def barriers(hb=2.5, hm=0.15, ab=3.4, am=0.3):
    """Offsets above the film conduction edge [eV] and tunnelling masses of the two dielectrics (hb, hm: oxB next to
    the film; ab, am: oxA), both faces."""
    from dynameta.carriers.tunneling import DielectricBarrier
    h, a = DielectricBarrier(hb, "TCO", hm), DielectricBarrier(ab, "TCO", am)
    return {"up_oxb": h, "lo_oxb": h, "up_oxa": a, "lo_oxa": a}


def te_leak(scale=1.0, **kw):
    from dynameta.carriers.gate_leakage import GateLeakage
    return GateLeakage(model="tsu_esaki", barriers=barriers(**kw), gate_fermi_energy_eV=10.0, mobility_m2Vs=MU,
                       energy_max_eV=8.0, scale=scale)


def _solve(design, voltages, leak, tag, **bkw):
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    b = LayeredDevsimBuilder(design, mesh_name="m_" + tag, device_name="d_" + tag, gate_leakage=leak, **bkw)
    b.solve(BiasPoint(voltages, tag), abs_tol=ABS_TOL)
    return b


def _arrays(b):
    ds = _ds()
    out = {}
    for r in ds.get_region_list(device=b.device):
        models = ds.get_node_model_list(device=b.device, region=r)
        for nm in ("x", "y", "Potential", "Electrons", "EF_ref_V"):
            if nm in models:
                out["{}|{}".format(r, nm)] = [float(v) for v in ds.get_node_model_values(device=b.device, region=r,
                                                                                          name=nm)]
    return out


def _report(b):
    rep = b.gate_leakage_report(per_cell=False)["film"]
    out = {k: rep[k] for k in ("balance", "balance_rel", "max_drop_V", "ef_span_V", "vertical_spread_V", "P_joule",
                               "P_tunnel", "P_terminals")}
    out["gates"] = {g: {k: v for k, v in d.items() if k != "interface"} for g, d in rep["gates"].items()}
    out["contacts"] = rep["contacts"]
    out["x_m"] = [float(v) for v in rep["x_m"]]
    out["phi_F_V"] = [float(v) for v in rep["phi_F_V"]]
    return out


def _sheet(b, region="film"):
    """q mu Int n dz on the middle node column (trapezoid over the native z nodes)."""
    from dynameta.core.numerics import trapz
    ds = _ds()
    X = np.asarray(ds.get_node_model_values(device=b.device, region=region, name="x"))
    Y = np.asarray(ds.get_node_model_values(device=b.device, region=region, name="y"))
    n = np.asarray(ds.get_node_model_values(device=b.device, region=region, name="Electrons"))
    xs = np.unique(X)
    sel = X == xs[len(xs) // 2]
    o = np.argsort(Y[sel])
    return 1.602176634e-19 * MU * trapz(n[sel][o], Y[sel][o])


# =====================================================================================================
def sc_zero(opts):
    """Zero leakage (scale 0) against the plain equilibrium device; plus the composed options."""
    vt = opts.get("vt", 2.0)
    out = {}
    kind = opts.get("kind", "plain")
    bkw, mats, dkw = {}, None, {}
    if kind == "composed":
        import _wiring_fixture as W
        from dynameta.carriers.physics_traps import FilmInterfaceTraps
        from dynameta.materials import M_E, Material, TransportModel
        from dynameta.materials.transport_model import KaneBand
        from dynameta.materials import MaterialRegistry
        base = W.materials()
        old = base.get("TCO")
        mats = MaterialRegistry()
        for nm in base.names():
            if nm != "TCO":
                mats.add(base.get(nm))
        mats.add(Material("TCO", optical=old.optical,
                          transport=TransportModel(W.N_BG, W.EPS_S, physics="equilibrium", chi_eV=W.CHI_EV,
                                                   band=KaneBand(m0_kg=0.3 * M_E, alpha_per_eV=0.4))))
        bkw = dict(fixed_charge={("film", "up_oxb"): -1.0e-3}, bulk_charge={"lo_oxb": 1.0e5},
                   interface_traps={("film", "up_oxb"): FilmInterfaceTraps(dit_eV_m2=1e16, e_lo_eV=-0.6, e_hi_eV=0.6,
                                                                            neutral_level_eV=0.0)})
        dkw = dict(wf_top=4.6)
    a = _solve(_design(mats=mats, **dkw), {"top": vt, "bot": 0.0}, None, "a", **bkw)
    A = _arrays(a)
    a.teardown()
    z = _solve(_design(mats=mats, **dkw), {"top": vt, "bot": 0.0}, te_leak(scale=0.0), "z", **bkw)
    Z = _arrays(z)
    out["report_zero"] = _report(z)
    z.teardown()
    out["eq"] = A
    out["zero"] = Z
    # the same device with the leakage on (composition runs, balances)
    o = _solve(_design(mats=mats, **dkw), {"top": vt, "bot": 0.0}, te_leak(), "o", **bkw)
    out["report_on"] = _report(o)
    o.teardown()
    return out


def sc_parabola(opts):
    """Uniform J0 into a uniform sheet (both gates at 0 V: the film is at flat band, n = n_bg)."""
    from dynameta.carriers.gate_leakage import GateLeakage
    import _wiring_fixture as W
    P, dx, J0 = opts["P"], opts["dx"], opts["J0"]
    out = {}
    const = GateLeakage(model="table", mobility_m2Vs=MU, gates=("top",),
                        table=lambda V: (np.full(np.shape(V), J0), np.zeros(np.shape(V))))
    for tag, edges in (("one", [W.edge()]), ("two", [W.edge(), W.edge("gnd2", side="x_hi")])):
        b = _solve(_design(P, dx, edges=edges), {"top": 0.0, "bot": 0.0}, const, tag)
        ds = _ds()
        r = _report(b)
        r["bottom_x"], r["bottom_phi"] = _face_row(ds, b, "lo")
        r["sheet"] = _sheet(b)
        out[tag] = r
        b.teardown()
    # two contacts at different voltages, no leakage: the linear ohmic profile (the equilibrium edge reference
    # refuses this bias)
    zero = GateLeakage(model="table", mobility_m2Vs=MU, table=lambda V: (np.zeros(np.shape(V)), np.zeros(np.shape(V))))
    dv = opts.get("dv", 1e-4)
    b = _solve(_design(P, dx, edges=[W.edge(), W.edge("gnd2", side="x_hi", role="biased")]),
               {"top": 0.0, "bot": 0.0, "gnd2": dv}, zero, "lin")
    r = _report(b)
    r["sheet"] = _sheet(b)
    out["linear"] = r
    b.teardown()
    return out


def _face_row(ds, b, which):
    X = np.asarray(ds.get_node_model_values(device=b.device, region="film", name="x"))
    Y = np.asarray(ds.get_node_model_values(device=b.device, region="film", name="y"))
    F = np.asarray(ds.get_node_model_values(device=b.device, region="film", name="EF_ref_V"))
    y0 = Y.min() if which == "lo" else Y.max()
    sel = np.abs(Y - y0) < 1e-15
    o = np.argsort(X[sel])
    return [float(v) for v in X[sel][o]], [float(v) for v in F[sel][o]]


def _table(opts):
    from dynameta.carriers.gate_leakage import LeakageTable
    V = np.linspace(-3.0, 3.0, 61)
    J = opts["J1"] * np.sinh(V / opts["V1"])
    return LeakageTable(tuple(V), tuple(J))


def sc_nonlinear(opts):
    """A strongly nonlinear table J(V) on the top gate at +vt; the sheet conductance at local Fermi potentials phi_k
    from narrow-cell solves (contact at phi_k); the long film; optionally a lateral mesh series."""
    from dynameta.carriers.gate_leakage import GateLeakage
    import _wiring_fixture as W
    vt = opts["vt"]
    tab = _table(opts)
    leak = GateLeakage(model="table", table=tab, mobility_m2Vs=MU, gates=("top",))
    out = {"table_V": list(tab.V_V), "table_J": list(tab.J_A_m2)}
    sig = []
    phis = opts["phis"]
    gnd = W.edge(role="biased")
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    nb = LayeredDevsimBuilder(_design(40e-9, 4e-9, edges=[gnd]), mesh_name="m_s", device_name="d_s", gate_leakage=leak)
    for p in phis:
        nb.solve(BiasPoint({"top": vt, "bot": 0.0, "gnd": p}, "s"), abs_tol=ABS_TOL)
        sig.append(_sheet(nb))
    nb.teardown()
    out["phis"], out["sigma"] = list(phis), sig
    runs = []
    for dx in opts["dxs"]:
        b = _solve(_design(opts["P"], dx), {"top": vt, "bot": 0.0}, leak, "n{}".format(len(runs)))
        r = _report(b)
        r["dx"] = dx
        r["top_x"], r["top_phi"] = _face_row(_ds(), b, "hi")
        runs.append(r)
        b.teardown()
    out["runs"] = runs
    return out


def sc_kernel(opts):
    """The term's own state at a solved Tsu-Esaki device: the inputs of the kernel and its J at every node."""
    from dynameta.carriers import gate_leakage as GL
    ds = _ds()
    vt, vb = opts.get("vt", 3.0), opts.get("vb", -2.0)
    b = _solve(_design(), {"top": vt, "bot": vb}, te_leak(**opts.get("bar", {})), "k")
    out = {"report": _report(b), "hist": _newton_history(b, vt)}
    for t in GL.gate_leakage_terms(b.device):
        st = t.state(b.device)
        out[t.gate] = {"J": st["J"].tolist(), "d": st["d"].tolist(), "x_s": st["x_s"].tolist(),
                       "x_g": st["x_g"].tolist(), "layers": [[l.offset_eV, l.thickness_m, l.m_over_m0] for l in t.layers],
                       "m_supply": t.m_supply, "kT_J": t.grid.kT_J, "V_t": float(ds.get_parameter(
                           device=b.device, region="film", name="V_t")), "area": t.area.tolist()}
    b.teardown()
    return out


def sc_newton(opts):
    """Full ramp to (vt, vb) with the physical model (adaptive continuation), then one Newton from a 2 % perturbed
    top bias with DEVSIM's per-iteration update norms."""
    vt, vb = opts["vt"], opts["vb"]
    b = _solve(_design(), {"top": vt, "bot": vb}, te_leak(**opts.get("bar", {})), "nw")
    out = {"hist": _newton_history(b, vt), "report": _report(b)}
    b.teardown()
    return out


def _newton_history(b, vt):
    """Max update norm per Newton iteration of one solve from a 2 % perturbed top bias back to vt."""
    ds = _ds()
    b._set_drive("top", vt * 0.98)
    ds.solve(type="dc", absolute_error=ABS_TOL, relative_error=1e-5, maximum_iterations=40)
    b._set_drive("top", vt)
    info = ds.solve(type="dc", absolute_error=ABS_TOL, relative_error=1e-13, maximum_iterations=40, info=True)
    return [max(e["absolute_error"] for r in it["devices"][0]["regions"] for e in r["equations"])
            for it in info["iterations"]]


def sc_field_table(opts):
    """A J(E) table (LeakageTable.from_field, t = 8 nm, the top dielectric's thickness) on the top gate at +vt, optionally with a top-gate work function:
    the term's state at every interface node, the raw film / top-dielectric potentials (the test derives the
    dielectric drop itself), the report, and one Newton from a 2 % perturbed top bias."""
    from dynameta.carriers import gate_leakage as GL
    vt = opts.get("vt", 2.0)
    E = np.linspace(-6e8, 6e8, 61)
    tab = GL.LeakageTable.from_field(tuple(E), tuple(opts["J1"] * np.sinh(E / opts["E1"])), opts["t"])
    leak = GL.GateLeakage(model="table", table=tab, mobility_m2Vs=MU, gates=("top",))
    b = _solve(_design(wf_top=opts.get("wf_top")), {"top": vt, "bot": 0.0}, leak, "ft")
    t = [x for x in GL.gate_leakage_terms(b.device) if x.gate == "top"][0]
    st = t.state(b.device)
    out = {"J": st["J"].tolist(), "V_table": st["V_table"].tolist(), "V_local": st["V_local"].tolist(),
           "is_contact": t.is_contact.tolist(), "arrays": _arrays(b), "report": _report(b),
           "x_iface": np.asarray(_ds().get_node_model_values(device=b.device, region="film", name="x"))[t.iface].tolist(),
           "table_V": list(tab.V_V), "table_J": list(tab.J_A_m2)}
    out["hist"] = _newton_history(b, vt)
    b.teardown()
    return out


def sc_refuse_1d(opts):
    """setup_fermi_field_region on a 1-D device -> NotImplementedError text."""
    from dynameta.carriers import gate_leakage as GL
    from dynameta.carriers import physics_equilibrium as PE
    ds = _ds()
    ds.create_1d_mesh(mesh="m1")
    ds.add_1d_mesh_line(mesh="m1", pos=0.0, ps=1e-9, tag="a")
    ds.add_1d_mesh_line(mesh="m1", pos=5e-9, ps=1e-9, tag="b")
    ds.add_1d_region(mesh="m1", material="TCO", region="film", tag1="a", tag2="b")
    ds.add_1d_contact(mesh="m1", name="c0", tag="a", material="metal")
    ds.add_1d_contact(mesh="m1", name="c1", tag="b", material="metal")
    ds.finalize_mesh(mesh="m1")
    ds.create_device(mesh="m1", device="d1")
    PE.setup_semiconductor_region("d1", "film", n_bg_m3=3e26, eps_static=9.0, dos_mass_kg=0.30 * 9.1093837015e-31,
                                  fermi_ref_param=PE.FERMI_REF_PARAM, fermi_ref_node=True)
    try:
        GL.setup_fermi_field_region("d1", "film", mobility_m2Vs=MU)
    except NotImplementedError as e:
        return {"error": str(e)}
    return {"error": None}


SCENARIOS = {"zero": sc_zero, "parabola": sc_parabola, "nonlinear": sc_nonlinear, "kernel": sc_kernel,
             "newton": sc_newton, "refuse_1d": sc_refuse_1d, "field_table": sc_field_table}


def run_child(scenario, **opts):
    """Run `scenario` in a fresh interpreter on this sys.path; returns its JSON result."""
    import subprocess
    import tempfile
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), scenario, out, json.dumps(opts)],
                           env=env, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError("child {} failed:\n{}".format(scenario, r.stderr[-6000:]))
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.remove(out)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error")                       # warnings are errors, as in the suite
    _opts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    _res = SCENARIOS[sys.argv[1]](_opts)
    with open(sys.argv[2], "w", encoding="utf-8") as _f:
        json.dump(_res, _f)
