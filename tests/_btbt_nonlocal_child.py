"""Child-process scenarios for tests/test_btbt_nonlocal.py (DynaMeta v0.15 plan item 4.1.3, nonlocal BTBT).

WHY A CHILD. The nonlocal term is a custom_terms term; DEVSIM's custom-equation registration is process-
global and permanent (tests/_custom_terms_child.py). Every scenario that adds the term runs here, in a fresh
interpreter on the caller's sys.path, and reports plain JSON.

    python tests/_btbt_nonlocal_child.py <scenario> <out.json> [options-json]

Devices:
  zener   the 1-D abrupt p+n+ Zener diode of tests/test_devsim_tunneling.py (Na = Nd = 5e24 m^-3, 0.2 nm at
          the junction, n_i = 1e16, Boltzmann), two plain ohmic contacts pc (bias pc_bias) / nc.
  fdt     validation.devsim_noise's FDT diode (n_i = 1e19, 1e22 m^-3, um scale; source V1 on 'hi'), where
          ssac and noise are trustworthy (the D2b set-up; on the 5e24 Zener DEVSIM's ssac Re Y has an
          absolute floor of ~1 S/m^2 with or without any BTBT term -- MEASURED -0.56 S/m^2 against a DC
          conductance of 6.6e-6 S/m^2 -- so ssac / noise gates use this device) with an artificial BTBT
          set strong enough to dominate the zero-bias conductance (B = 1e7 V/m, A = 1e17 for P = 2.5 and
          1.5e20 for P = 2: the same rate at the ~2e6 V/m junction field) and a small
          tunnelling gap (Eg_tun = 0.25 eV < the 0.357 V built-in) so that paths exist at equilibrium.
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

Q = 1.602176634e-19
ZN, ZNI, ZXJ, ZL = 5.0e24, 1.0e16, 100e-9, 200e-9
FDT_A = {2.5: 1.0e17, 2.0: 1.5e20}               # A_SI per P: the same rate at the ~2e6 V/m junction field
FDT_B = 1.0e7                                     # B [V/m]
FDT_EG, FDT_LMAX = 0.25, 3e-7


def _ds():
    from dynameta.carriers.devsim_env import import_devsim
    return import_devsim()


def _lst(a):
    return [float(v) for v in np.ravel(a)]


def _quiet():
    import validation.devsim_noise as V
    return V.quiet()


# =====================================================================================================
# devices
# =====================================================================================================

def build_zener(dev="z", *, fd=False, local=None, h_junction=0.2e-9):
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    mesh = dev + "_m"
    _R.clear(dev)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="pc")
        ds.add_1d_mesh_line(mesh=mesh, pos=ZXJ, ps=h_junction)
        ds.add_1d_mesh_line(mesh=mesh, pos=ZL, ps=2e-9, tag="nc")
        ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
        ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping",
                      equation="ifelse(x < {}, {}, {})".format(ZXJ, -ZN, ZN))
        BP.setup_bipolar_region(dev, "b", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=ZNI,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-3, tau_p_s=1e-3,
                                fd_enhancement=fd, btbt=local)
        BP.setup_equilibrium_seed_models(dev, "b")
        for c in ("pc", "nc"):
            BP.setup_contact_ohmic_bipolar(dev, c)
        ds.node_model(device=dev, region="b", name="_seed",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "b")))
        for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="b", name=v,
                               values=ds.get_node_model_values(device=dev, region="b", name=m))
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.delete_by_name(dev, ceq)
        ds.solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=100)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.reapply_by_name(dev, ceq)
    return dev


def solve(ae=1e12, re_=1e-10, it=100):
    """DC Newton; returns the iteration count (raises on non-convergence)."""
    ds = _ds()
    with _quiet():
        info = ds.solve(type="dc", absolute_error=ae, relative_error=re_, maximum_iterations=it, info=True)
    if not info["converged"]:
        raise RuntimeError("DC solve did not converge")
    return len(info["iterations"])


def terminal_zener(dev):
    ds = _ds()
    return -sum(ds.get_contact_current(device=dev, contact="pc", equation=e)
                for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))


def build_fdt(P=2.5):
    import validation.devsim_noise as V
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import BTBTParams
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8)
    term = NL.setup_btbt_nonlocal("fdt", "bulk", NL.NonlocalBTBT(BTBTParams(FDT_A[P], FDT_B, P), Eg_tun_eV=FDT_EG,
                                                                  max_length_m=FDT_LMAX))
    V._solve(1e8, 1e-5)
    return term


def _same_node(term):
    """The local-model shortcut: both carriers of each event at the START node (a same-node pair)."""
    from dynameta.carriers.noise_analysis import SourceRow, TwoPointSource
    real = term.noise_sources

    def same(device):
        s = real(device)[0]
        r0 = s.rows[0]
        return [TwoPointSource(s.mechanism, s.region, s.rate,
                               (r0, SourceRow(r0.region, "ElectronContinuityEquation", -np.ones(r0.coeff.size),
                                              r0.nodes)), s.map_nodes)]
    term.noise_sources = same


def g_rich(dv, ae=1e8):
    """dI/dV by Richardson of +-dv and +-2dv central differences of solved DC currents."""
    import validation.devsim_noise as V
    v0 = getattr(V.set_bias, "_v", {}).get("V1", 0.0)
    g1 = V.dc_conductance(v0, ae, dv=dv)
    g2 = V.dc_conductance(v0, ae, dv=2 * dv)
    return (4.0 * g1 - g2) / 3.0, g1


# =====================================================================================================
# scenarios
# =====================================================================================================

def sc_defaults(opts):
    """Importing the module registers nothing and pulls in neither custom_terms nor devsim."""
    import importlib
    mods_before = set(sys.modules)
    importlib.import_module("dynameta.carriers.btbt_nonlocal")
    new = set(sys.modules) - mods_before
    from dynameta.carriers import custom_terms as CT
    return {"devsim_loaded": any(m == "devsim" or m.startswith("devsim.") or m.startswith("devsim_py")
                                 for m in new),
            "custom_terms_loaded_by_import": "dynameta.carriers.custom_terms" in new,
            "registered": CT.is_registered()}


def sc_fdt(opts):
    """Equilibrium detailed balance, FDT through the two-point source (and the same-node prove-out),
    ssac vs DC dI/dV at 0 V and at reverse bias."""
    import validation.devsim_noise as V
    from dynameta.carriers import btbt_nonlocal as NL
    term = build_fdt(opts.get("P", 2.5))
    out = {"stats": dict(term.last_stats)}
    st, res = term.evaluate("fdt")
    one_way = 0.5 * res["R_pre"] * res["pair"]                  # at equilibrium f_p(1-f_n) = f_n(1-f_p)
    out["db_ratio"] = float(np.max(np.abs(res["R"]) / one_way))
    out["one_way_max"] = float(np.max(one_way))
    ff = [1.0, 1e5, 1e7, 1e9]
    Y = V.ssac_Y(ff)
    sn = V.noise(ff)
    out["ReY"] = _lst(Y.real)
    out["S"] = _lst(sn.S)
    out["share_btbt"] = float(sn.by_mechanism["btbt_nl"][0] / sn.S[0])
    out["G0"], out["G0_1"] = g_rich(1e-4)
    out["G0_2"] = g_rich(2e-4)[0]
    if opts.get("same_node"):
        _same_node(term)
        out["S_same"] = _lst(V.noise(ff).S)
    # reverse bias: ssac vs DC dI/dV, conservation of the terminal current
    V.set_bias(-opts.get("vr", 0.2), 1e8)
    V._solve(1e8, 1e-8)
    out["Yr"] = _lst(V.ssac_Y([1.0]).real)
    out["Gr"], out["Gr_1"] = g_rich(1e-4)
    out["Gr_2"] = g_rich(2e-4)[0]
    g = NL.btbt_nonlocal_generation("fdt", "bulk")
    out["Ip_r"], out["In_r"], out["Isrc_r"] = g["I_p"], g["I_n"], V.source_current()
    return out


def _rows_share(dev, region, term):
    """The custom term's own residual share (rhs_on - rhs_off) summed over the hole and electron rows."""
    from dynameta.carriers import custom_terms as CT
    import _custom_terms_child as CC
    real = term.assemble
    term.active = True
    term.assemble = lambda d, w, t: real(d, w, t) if term.active else ([], [])
    term.active = False
    off = CC.matrix()[3]
    term.active = True
    on = CC.matrix()[3]
    ds = _ds()
    n = len(ds.get_node_model_values(device=dev, region=region, name="x"))
    contacts = CT._contact_node_set(dev, region)
    inner = np.array([k for k in range(n) if k not in contacts])
    rp = CT.node_rows(dev, region, "HoleContinuityEquation", inner)
    rn = CT.node_rows(dev, region, "ElectronContinuityEquation", inner)
    d = on - off
    term.assemble = real
    return float(np.sum(d[rp])), float(np.sum(d[rn])), float(np.sum(np.abs(d[rp])))


def sc_zener(opts):
    """Reverse sweep 0 -> 1.2 V of the Zener diode, nonlocal (Si Hurkx = Sentaurus' default nonlocal set,
    Eg_tun 1.12 eV) and local twins: currents, conservation, iteration counts, stats."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import SILICON_HURKX, btbt_generation_current
    ds = _ds()
    out = {"nl": [], "loc": []}
    vrs = [float(v) for v in opts.get("vr", [0.3, 0.6, 0.9, 1.2])]
    dev = build_zener("znl")
    term = NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12))
    its0 = solve()
    out["eq"] = {"its": its0, "stats": dict(term.last_stats),
                 "I_p": NL.btbt_nonlocal_generation(dev, "b")["I_p"]}
    for vr in vrs:
        ds.set_parameter(device=dev, name="pc_bias", value=-vr)
        its = solve()
        g = NL.btbt_nonlocal_generation(dev, "b")
        hs, es, habs = _rows_share(dev, "b", term)
        out["nl"].append({"vr": vr, "its": its, "I_p": g["I_p"], "I_n": g["I_n"], "J": terminal_zener(dev),
                          "stats": g["stats"], "hole_share": hs, "elec_share": es, "abs_share": habs})
    ds.delete_device(device=dev)
    dev = build_zener("zloc", local=SILICON_HURKX)
    out["loc_eq_its"] = solve()
    for vr in vrs:
        ds.set_parameter(device=dev, name="pc_bias", value=-vr)
        its = solve()
        out["loc"].append({"vr": vr, "its": its, "I": btbt_generation_current(dev, "b"),
                           "J": terminal_zener(dev)})
    return out


def sc_flip(opts):
    """Refutation of the conservation gate: the electron entries with the wrong sign."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import SILICON_HURKX
    ds = _ds()
    dev = build_zener("zflip")
    term = NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12))
    solve()
    ds.set_parameter(device=dev, name="pc_bias", value=-0.9)
    solve()
    real = term.assemble
    eq_rows = None

    def flipped(d, w, t):
        rcv, rv = real(d, w, t)
        from dynameta.carriers import custom_terms as CT
        en = set(int(r) for r in CT.node_rows(d, "b", "ElectronContinuityEquation",
                                              np.arange(1, len(ds.get_node_model_values(
                                                  device=d, region="b", name="x")) - 1)))
        rv = [(-v if (k % 2 == 1 and int(rv[k - 1]) in en) else v) for k, v in enumerate(rv)]
        return rcv, rv
    del eq_rows
    term.assemble = flipped
    hs, es, habs = _rows_share(dev, "b", term)
    return {"hole_share": hs, "elec_share": es, "abs_share": habs}


def fd_columns(dev, region, var, nodes, rows, h, term):
    """The custom term's OWN Jacobian share J_on - J_off at `rows` x (var at `nodes`) against central
    differences (steps h and 2h) of its own residual share rhs_on - rhs_off (get_matrix_and_rhs returns
    rhs = -F, so FD = -J). Toggling term.active at the same state isolates the term. rhs_scale: the
    round-off scale of the rows, max_r sum_k |J_on[r, k]| |x_k| over the solution variables. (The
    method of tests/_custom_terms_child.fd_columns, restricted to Potential / Electrons / Holes: that
    helper scans every node model with get_equation_numbers, which trips a DEVSIM circuit assertion on a
    device without a circuit.)"""
    import _custom_terms_child as CC
    import scipy.sparse as sp
    from dynameta.carriers import custom_terms as CT
    ds = _ds()

    def both():
        term.active = False
        off = CC.matrix()
        term.active = True
        on = CC.matrix()
        return on, off
    on, off = both()
    Jon = sp.csc_matrix((on[0], on[1], on[2]))
    Jc = Jon - sp.csc_matrix((off[0], off[1], off[2]))
    xs = np.zeros(Jon.shape[1])
    for v in ("Potential", "Electrons", "Holes"):
        c = np.asarray(ds.get_equation_numbers(device=dev, region=region, variable=v))
        ok = c >= 0
        xs[c[ok]] = np.asarray(ds.get_node_model_values(device=dev, region=region, name=v))[ok]
    R = float(np.max(np.abs(Jon[rows, :]) @ np.abs(xs)))
    cols = CT.node_columns(dev, region, var, nodes)
    out = []
    for node, col in zip(nodes, cols):
        x0 = list(ds.get_node_model_values(device=dev, region=region, name=var))[node]
        hh = h(x0)
        rec = {"node": int(node), "var": var, "h": hh, "J": _lst(Jc[rows, int(col)].toarray()),
               "rhs_scale": R}
        for tag, step in (("fd", hh), ("fd2", 2.0 * hh)):
            share = []
            for sgn in (+1.0, -1.0):
                ds.set_node_value(device=dev, region=region, name=var, index=int(node), value=x0 + sgn * step)
                on, off = both()
                share.append(on[3][rows] - off[3][rows])
            rec[tag] = _lst((share[0] - share[1]) / (2.0 * step))
        ds.set_node_value(device=dev, region=region, name=var, index=int(node), value=x0)
        out.append(rec)
    return out


def sc_jacobian(opts):
    """The term's share of DEVSIM's assembled Jacobian against central differences (steps h and 2h) of
    DEVSIM's own residual, at a reverse-biased Zener (the tests/_custom_terms_child.fd_columns method)."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers import custom_terms as CT
    from dynameta.carriers.tunneling import BTBTParams, SILICON_HURKX
    ds = _ds()
    params = SILICON_HURKX if opts.get("P", 2.5) == 2.5 else BTBTParams(3.5e21, 2.2e9, 2.0)
    dev = build_zener("zj")
    term = NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(params, Eg_tun_eV=opts.get("eg", 1.12),
                                                            mass_ratio_V_over_C=1.7))
    solve()
    if opts.get("vr", 0.9):
        ds.set_parameter(device=dev, name="pc_bias", value=-opts.get("vr", 0.9))
        solve()
    if opts.get("drop_dt"):                                   # refutation: no implicit end-point derivative
        real_rates = NL.path_rates_1d

        def no_dt(*a, **k):
            r = real_rates(*a, **k)
            if "jac" in r:
                r["jac"]["dt"] = 0.0 * r["jac"]["dt"]
            return r
        NL.path_rates_1d = no_dt
    real = term.assemble
    term.active = True
    term.assemble = lambda d, w, t: real(d, w, t) if term.active else ([], [])
    st, res = term.evaluate(dev)
    p, perm = res["paths"], st["perm"]
    k = int(np.argmax(np.abs(res["R"])))
    i, a, b = int(p.start[k]), int(p.a[k]), int(p.b[k])
    d = int(p.direction[k])
    mid = (i + a) // 2
    reg = "b"
    rows = np.concatenate([CT.node_rows(dev, reg, "HoleContinuityEquation", perm[[i]]),
                           CT.node_rows(dev, reg, "ElectronContinuityEquation", perm[[a, b]])])
    psi_nodes = sorted({i - 1, i, i + 1, mid, a, b, b + d})
    recs = fd_columns(dev, reg, "Potential", list(perm[psi_nodes]), rows, lambda x: 1e-5, term)
    recs += fd_columns(dev, reg, "Electrons", list(perm[sorted({a, b})]), rows, lambda x: 1e-5 * x, term)
    recs += fd_columns(dev, reg, "Holes", list(perm[[i]]), rows, lambda x: 1e-5 * x, term)
    labels = {int(perm[i]): "start", int(perm[i - 1]): "start-1", int(perm[i + 1]): "start+1",
              int(perm[mid]): "mid", int(perm[a]): "a", int(perm[b]): "b", int(perm[b + d]): "beyond"}
    for r in recs:
        r["label"] = labels.get(r["node"], "?")
    return {"recs": recs, "path": [i, a, b, d], "t": float(res["t"][k])}


def sc_equilibrium(opts):
    """|R| / one-way rate at V = 0 on the Zener with a tunnelling gap below the built-in potential, so paths
    exist at equilibrium; Boltzmann (fd=False) and the FD g-factor transport (fd=True)."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import SILICON_HURKX
    out = {}
    for fd in opts.get("fd", [False, True]):
        for h in opts.get("h", [0.2e-9]):
            dev = build_zener("ze{}{}".format(int(fd), int(round(h * 1e12))), fd=fd, h_junction=h)
            term = NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=0.9))
            solve()
            if opts.get("delta_shift"):                    # refutation: the f_n frame offset off by kT x shift
                real_state = term.state

                def shifted(device, real_state=real_state):
                    st_ = real_state(device)
                    st_["delta"] += opts["delta_shift"]
                    return st_
                term.state = shifted
            st, res = term.evaluate(dev)
            one_way = 0.5 * res["R_pre"] * res["pair"]
            out["{}_{}".format("fd" if fd else "boltz", h)] = {
                "ratio": float(np.max(np.abs(res["R"]) / one_way)), "paths": int(res["R"].size),
                "one_way_max": float(np.max(one_way))}
            _ds().delete_device(device=dev)
    return out


SCENARIOS = {"defaults": sc_defaults, "fdt": sc_fdt, "zener": sc_zener, "flip": sc_flip,
             "jacobian": sc_jacobian, "equilibrium": sc_equilibrium}


def run_child(scenario, **opts):
    """Run `scenario` in a fresh interpreter on this sys.path; returns its JSON result."""
    import subprocess
    import tempfile
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), scenario, out, json.dumps(opts)],
                           env=env, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError("child {} failed:\n{}".format(scenario, r.stderr[-4000:]))
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
