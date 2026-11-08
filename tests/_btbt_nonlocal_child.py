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
import time

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


# The frozen-carrier POISSON presolve of build_zener / build_2d (2026-11-08, PR #36 CI). Its seed is locally
# neutral, so the presolve is a Laplace solve between the two contacts: psi is linear in x, and on the
# symmetric devices here (+-N, the junction or the slant centre at the middle) a whole column of nodes
# solves to psi = 0 (+- 9e-17 V of round-off; the slanted diode: the 25 nodes at x = 0.6 um). DEVSIM's
# relative test |dpsi| / (|psi| + 1e-10) is then an ABSOLUTE test |dpsi| < relative_error * 1e-10 there --
# 1e-18 V at the old relative_error = 1e-8, 1/100 ulp of the 0.18 V contact potential and BELOW the
# round-off of the converged update at those nodes (MEASURED 1.3-3.5e-8 relative = 1.3-3.5e-18 V). The old
# presolve exited only when the last bits fell under it: 7 / 8 iterations with WSL's OpenBLAS Haswell kernel /
# on Windows, "Convergence failure!" after 100 with the Sandybridge / Nehalem kernels (OPENBLAS_CORETYPE)
# and on the CI runner; the lateral Zener's took 94 of its 100 there (now 6 and 11 on every kernel). The criterion is therefore the
# ABSOLUTE potential update -- for a Poisson-only region DEVSIM's abs is max |dpsi| [V] -- at 1e-12 V: 7e4 x
# the measured 1.5e-17 V floor of max |dpsi|, and tighter than the old test anywhere psi is not ~0 (there
# 1e-8 x 0.18 V = 1.8e-9 V). DEVSIM ANDs the relative test, so it is parked at 1e-3 (3e4 x its 3.5e-8
# floor); the Laplace solve converges quadratically through both in the same step (MEASURED: abs 2.5e-2 ->
# 7e-16 V). The presolve is only the coupled Newton's starting point; that Newton keeps its own criteria.
PRESOLVE_ABS_V, PRESOLVE_REL = 1e-12, 1e-3


def _poisson_presolve(dev):
    """Frozen-carrier Poisson solve of `dev` (its continuity equations dropped, then re-applied)."""
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(dev, ceq)
    ds.solve(type="dc", absolute_error=PRESOLVE_ABS_V, relative_error=PRESOLVE_REL, maximum_iterations=100)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(dev, ceq)


# =====================================================================================================
# devices
# =====================================================================================================

def build_zener(dev="z", *, fd=False, local=None, h_junction=0.2e-9, xj=ZXJ, length=ZL, statistics=None):
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    mesh = dev + "_m"
    _R.clear(dev)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="pc")
        ds.add_1d_mesh_line(mesh=mesh, pos=xj, ps=h_junction)
        ds.add_1d_mesh_line(mesh=mesh, pos=length, ps=2e-9, tag="nc")
        ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
        ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping",
                      equation="ifelse(x < {}, {}, {})".format(xj, -ZN, ZN))
        BP.setup_bipolar_region(dev, "b", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=ZNI,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-3, tau_p_s=1e-3,
                                fd_enhancement=fd, btbt=local,
                                **({"statistics": statistics} if (fd and statistics) else {}))
        BP.setup_equilibrium_seed_models(dev, "b")
        for c in ("pc", "nc"):
            BP.setup_contact_ohmic_bipolar(dev, c)
        ds.node_model(device=dev, region="b", name="_seed",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "b")))
        for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="b", name=v,
                               values=ds.get_node_model_values(device=dev, region="b", name=m))
        _poisson_presolve(dev)
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
    """Importing the module (on top of tunneling, which it imports -- and which since the v0.15 oxide-tunnelling
    merge imports custom_terms itself, inertly) adds only itself, loads no devsim and registers nothing."""
    import importlib
    importlib.import_module("dynameta.carriers.tunneling")
    mid = set(sys.modules)
    importlib.import_module("dynameta.carriers.btbt_nonlocal")
    new = sorted(m for m in set(sys.modules) - mid if m.startswith("dynameta"))
    from dynameta.carriers import custom_terms as CT
    return {"devsim_loaded": any(m == "devsim" or m.startswith("devsim.") or m.startswith("devsim_py")
                                 for m in sys.modules),
            "new_dynameta_modules": new, "registered": CT.is_registered()}


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
    hj = float(opts.get("h", 0.2e-9))
    geo = {"xj": float(opts.get("xj", ZXJ)), "length": float(opts.get("length", ZL))}
    geo.update(fd=bool(opts.get("fd", False)), statistics=opts.get("statistics"))
    dev = build_zener("znl", h_junction=hj, **geo)
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
    dev = build_zener("zloc", local=SILICON_HURKX, h_junction=hj, **geo)
    out["loc_eq_its"] = solve()
    for vr in vrs:
        ds.set_parameter(device=dev, name="pc_bias", value=-vr)
        its = solve()
        out["loc"].append({"vr": vr, "its": its, "I": btbt_generation_current(dev, "b", check_resolution=bool(opts.get("check", True))),
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


def sc_jacobian(opts):
    """The term's share of DEVSIM's assembled Jacobian against central differences (steps h and 2h) of
    DEVSIM's own residual, at a reverse-biased Zener (tests/_custom_terms_child.fd_columns, the shared helper)."""
    import _custom_terms_child as CC
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
            if "jac_t" in r:                                 # the end-point (t) derivative the assembly reads
                r["jac_t"] = 0.0 * r["jac_t"]
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
    recs = CC.fd_columns(dev, reg, "Potential", list(perm[psi_nodes]), rows, lambda x: 1e-5, term)
    recs += CC.fd_columns(dev, reg, "Electrons", list(perm[sorted({a, b})]), rows, lambda x: 1e-5 * x, term)
    recs += CC.fd_columns(dev, reg, "Holes", list(perm[[i]]), rows, lambda x: 1e-5 * x, term)
    labels = {int(perm[i]): "start", int(perm[i - 1]): "start-1", int(perm[i + 1]): "start+1",
              int(perm[mid]): "mid", int(perm[a]): "a", int(perm[b]): "b", int(perm[b + d]): "beyond"}
    for r in recs:
        r["label"] = labels.get(r["node"], "?")
    return {"recs": recs, "path": [i, a, b, d], "t": float(res["t"][k])}


def sc_equilibrium(opts):
    """|R| / one-way rate at V = 0 on the Zener with a tunnelling gap below the built-in potential, so paths
    exist at equilibrium; Boltzmann (fd=False) and the FD g-factor transport (fd=True; opts["statistics"]:
    the FD devices' statistics, unset = the default, 'fermi_dirac' since v0.15.0)."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import SILICON_HURKX
    out = {}
    for fd in opts.get("fd", [False, True]):
        for h in opts.get("h", [0.2e-9]):
            dev = build_zener("ze{}{}".format(int(fd), int(round(h * 1e12))), fd=fd, h_junction=h,
                              statistics=opts.get("statistics"))
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


def sc_warn_T(opts):
    """A region at T != T_REF gets ONE AdvisoryWarning naming the temperature; T_REF gets none."""
    import warnings
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.bipolar_options import BipolarOptions
    from dynameta.carriers.tunneling import SILICON_HURKX
    ds = _ds()
    out = {}
    for tag, opt in (("T350", BipolarOptions(T_K=350.0)), ("Tref", None)):
        dev = "zw" + tag
        mesh = dev + "_m"
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="a")
        ds.add_1d_mesh_line(mesh=mesh, pos=5e-8, ps=1e-9, tag="b")
        ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
        ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="a", tag2="b")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="s", name="NetDoping", equation="ifelse(x < 2.5e-8, -1e24, 1e24)")
        with _quiet():
            BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                    mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-6, tau_p_s=1e-6,
                                    fd_enhancement=False, options=opt)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NL.setup_btbt_nonlocal(dev, "s", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12))
        out[tag] = [(type(x.message).__name__, str(x.message)) for x in w]
    return out


def sc_reverse_refusal(opts):
    """tunneling.setup_btbt on a region that carries a live nonlocal term: BTBTDoubleCountError, and the
    region is unchanged (no BTBT_U model); on the OTHER region of no term (a second device) it installs."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import BTBTDoubleCountError, SILICON_HURKX, setup_btbt
    ds = _ds()
    dev = build_zener("zr")
    NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12))
    out = {}
    try:
        setup_btbt(dev, "b", SILICON_HURKX)
        out["raised"] = None
    except BTBTDoubleCountError as e:
        out["raised"] = str(e)
    out["models_after"] = sorted(m for m in ds.get_edge_model_list(device=dev, region="b") if m.startswith("BTBT"))
    dev2 = build_zener("zr2")
    setup_btbt(dev2, "b", SILICON_HURKX)                  # no nonlocal term there: installs
    out["other_models"] = sorted(m for m in ds.get_edge_model_list(device=dev2, region="b")
                                 if m.startswith("BTBT_U") and ":" not in m)
    return out


def sc_gap_material(opts):
    """material= (the package gap kernel) vs the equivalent explicit Eg_tun_eV at T_REF: the solved Zener and
    the term's rates are bit-identical; at T_K = 350 K with a band-temperature model the path gap follows
    band_data_at_T and no advisory fires."""
    import warnings
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import SILICON_HURKX
    from dynameta.materials.band_temperature import BandTemperature, band_data_at_T
    from dynameta.materials.transport_model import TransportModel
    ds = _ds()
    base = dict(n_bg_m3=1e24, eps_static=11.7, dos_mass_kg_of_n_m3=lambda n: 0.26 * 9.1093837015e-31,
                band_gap_eV=float(opts["eg"]))
    tm = TransportModel(**base)
    out = {}
    for tag in ("explicit", "material"):
        dev = build_zener("zg" + tag)
        if tag == "explicit":
            spec = NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=float(tm.band_gap_eV))
        else:
            spec = NL.NonlocalBTBT(SILICON_HURKX, material=tm)
        term = NL.setup_btbt_nonlocal(dev, "b", spec)
        ds.set_parameter(device=dev, name="pc_bias", value=-0.9)
        solve()
        st, res = term.evaluate(dev)
        out[tag] = {"R_hex": [float(v).hex() for v in res["R"]],
                    "psi_hex": [float(v).hex() for v in st["psi"]], "Eg_path": term.phys.Eg_J / Q}
        ds.delete_device(device=dev)
    bt = BandTemperature(gap_law=tuple(opts["law"]), affinity_fraction=0.5)
    tmT = TransportModel(**dict(base, band_temperature=bt))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        term = _zener_at_T(NL.NonlocalBTBT(SILICON_HURKX, material=tmT), 350.0)
    out["T350"] = {"Eg_path": term.phys.Eg_J / Q, "Eg_set": term.phys.Eg_set_J / Q,
                   "kernel_path": band_data_at_T(tmT, 350.0).E_g_eV,
                   "kernel_set": band_data_at_T(tmT, 300.0).E_g_eV,
                   "warnings": [str(x.message) for x in w]}
    return out


def _zener_at_T(spec, T_K):
    """A Zener-like 1-D region at T_K (BipolarOptions.T_K) carrying the nonlocal term; returns the term."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.bipolar_options import BipolarOptions
    ds = _ds()
    dev = "zT"
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="a")
    ds.add_1d_mesh_line(mesh=mesh, pos=5e-8, ps=1e-9, tag="b")
    ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="a", tag2="b")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="s", name="NetDoping", equation="ifelse(x < 2.5e-8, -1e24, 1e24)")
    with _quiet():
        BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-6, tau_p_s=1e-6,
                                fd_enhancement=False, options=BipolarOptions(T_K=T_K))
    return NL.setup_btbt_nonlocal(dev, "s", spec)



# =====================================================================================================
# 2-D devices and scenarios (v0.15 2-D nonlocal BTBT)
# =====================================================================================================

def build_2d(dev, xlines, ylines, doping, *, n_i, tau, mu=(0.135, 0.048), circuit=False, local=None,
             contacts=None, presolve=True, ae=1e12, fd=False, statistics=None, n_dos=2.8e25):
    """A 2-D bipolar device on a DEVSIM tensor mesh (right triangles: well centred). xlines / ylines:
    [(pos, ps)]; doping: a NetDoping expression in x, y. contacts: [(name, xl, xh, yl, yh)] boxes on the
    boundary (default: 'lo' at x = xmin, 'hi' at x = xmax). 'lo' is ohmic (bias parameter lo_bias), 'hi'
    ohmic or (circuit=True) circuit-driven by source V1 on node vhi. fd=False: Boltzmann; fd=True: the FD g-factor
    transport with `statistics` (unset = the package default, 'fermi_dirac' since v0.15.0; 'legacy' = v0.14);
    n_dos: N_c = N_v [m^-3]."""
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    mesh = dev + "_m"
    _R.clear(dev)
    x0, x1 = xlines[0][0], xlines[-1][0]
    y0, y1 = ylines[0][0], ylines[-1][0]
    if contacts is None:
        contacts = [("lo", x0, x0, y0, y1), ("hi", x1, x1, y0, y1)]
    px = xlines[0][1]                                   # DEVSIM 2-D contacts sit on mesh OUTSIDE the region
    py = ylines[0][1]                                   # they touch: inert pad regions on all four sides
    with _quiet():
        ds.create_2d_mesh(mesh=mesh)
        for pos, ps in [(x0 - px, px)] + list(xlines) + [(x1 + px, px)]:
            ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=pos, ps=ps)
        for pos, ps in [(y0 - py, py)] + list(ylines) + [(y1 + py, py)]:
            ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=pos, ps=ps)
        ds.add_2d_region(mesh=mesh, material="Si", region="b", xl=x0, xh=x1, yl=y0, yh=y1)
        for nm, b in (("pl", (x0 - px, x0, y0, y1)), ("pr", (x1, x1 + px, y0, y1)),
                      ("pb", (x0, x1, y0 - py, y0)), ("pt", (x0, x1, y1, y1 + py))):
            ds.add_2d_region(mesh=mesh, material="Si", region=nm, xl=b[0], xh=b[1], yl=b[2], yh=b[3])
        for name, xl, xh, yl, yh in contacts:
            ds.add_2d_contact(mesh=mesh, name=name, material="metal", region="b", xl=xl, xh=xh, yl=yl, yh=yh,
                              bloat=1e-12)
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping", equation=doping)
        BP.setup_bipolar_region(dev, "b", eps_static=11.7, n_dos_m3=n_dos, n_i_m3=n_i,
                                mobility_n_m2Vs=mu[0], mobility_p_m2Vs=mu[1], tau_n_s=tau, tau_p_s=tau,
                                fd_enhancement=fd, btbt=local,
                                **({"statistics": statistics} if (fd and statistics) else {}))
        BP.setup_equilibrium_seed_models(dev, "b")
        for name, *_ in contacts:
            if circuit and name == "hi":
                BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1")
            else:
                BP.setup_contact_ohmic_bipolar(dev, name)
        ds.node_model(device=dev, region="b", name="_seed",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "b")))
        for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="b", name=v,
                               values=ds.get_node_model_values(device=dev, region="b", name=m))
        if presolve:
            _poisson_presolve(dev)
    return dev


_CIRCUIT = {"made": False}


def _has_circuit():
    return _CIRCUIT["made"]


def _xy(dev, region="b"):
    ds = _ds()
    return np.stack([np.asarray(ds.get_node_model_values(device=dev, region=region, name=c)) for c in "xy"], 1)


ZENER2D_X = [(0.0, 2e-9), (ZXJ, 0.2e-9), (ZL, 2e-9)]
ZENER2D_Y = [(0.0, 1e-9), (3e-9, 1e-9)]


def sc_lat2d(opts):
    """Laterally uniform 2-D Zener (4 rows) vs the 1-D model: (a) on the 2-D SOLUTION, each 2-D path == the
    1-D model on its own row's values; (b) q Int G per width vs the 1-D Zener device at the same bias, with the
    rows' psi spread and the 2-D-vs-1-D psi difference reported; (c) conservation, terminal current;
    (d) Newton iterations vs a local-model 2-D twin."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers.tunneling import SILICON_HURKX, btbt_generation_current
    ds = _ds()
    vr = float(opts.get("vr", 0.9))
    out = {}
    dop = "ifelse(x < {}, {}, {})".format(ZXJ, -ZN, ZN)
    fdk = {"fd": bool(opts.get("fd", False)), "statistics": opts.get("statistics")}
    dev = build_2d("z2", ZENER2D_X, ZENER2D_Y, dop, n_i=ZNI, tau=1e-3, **fdk)
    term = NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12))
    out["its_eq"] = solve()
    ds.set_parameter(device=dev, name="lo_bias", value=-vr)
    out["its"] = solve()
    st, res = term.evaluate(dev)
    ps = res["pathset"]
    xy = _xy(dev)
    rows = np.unique(xy[:, 1])
    devs = []
    for yv in rows:
        sel = np.flatnonzero(xy[:, 1] == yv)
        sel = sel[np.argsort(xy[sel, 0])]
        xr = xy[sel, 0]
        so = np.ones(sel.size, bool)
        so[0] = so[-1] = False
        r1 = NL.path_rates_1d(xr, st["psi"][sel], st["Ln"][sel], st["Lp"][sel], term.phys, st["delta"],
                              term.spec.max_length_m, so, np.ones(sel.size, bool), derivatives=False)
        m1 = dict(zip(sel[r1["paths"].start].tolist(), r1["R"]))
        for k, i in enumerate(ps.start):
            if xy[i, 1] == yv:
                devs.append(abs(res["R"][k] / m1[int(i)] - 1.0))
        out.setdefault("n1d", 0)
        out["n1d"] += len(m1)
    out["n2d"] = int(ps.start.size)
    out["per_path_dev"] = float(max(devs))
    g = NL.btbt_nonlocal_generation(dev, "b")
    W = float(rows[-1] - rows[0])
    out["I2d_per_W"] = g["I_p"] / W
    out["In2d_per_W"] = g["I_n"] / W
    out["J2d_per_W"] = -sum(ds.get_contact_current(device=dev, contact="lo", equation=e)
                            for e in ("ElectronContinuityEquation", "HoleContinuityEquation")) / W
    hs, es, habs = _rows_share(dev, "b", term)
    out.update(hole_share=hs, elec_share=es, abs_share=habs, paths=int(ps.start.size), W=W)
    # rows' psi spread (y-invariance of the 2-D solution)
    psi = st["psi"]
    spread = 0.0
    for yv in rows[1:]:
        a = np.flatnonzero(xy[:, 1] == rows[0])
        b = np.flatnonzero(xy[:, 1] == yv)
        spread = max(spread, float(np.max(np.abs(psi[a[np.argsort(xy[a, 0])]] - psi[b[np.argsort(xy[b, 0])]]))))
    out["row_spread_V"] = spread
    xrow = np.sort(xy[xy[:, 1] == rows[0], 0])
    psirow = psi[np.flatnonzero(xy[:, 1] == rows[0])][np.argsort(xy[xy[:, 1] == rows[0], 0])]
    ds.delete_device(device=dev)
    # the 1-D Zener at the same bias
    d1 = build_zener("z1", **fdk)
    t1 = NL.setup_btbt_nonlocal(d1, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12))
    solve()
    ds.set_parameter(device=d1, name="pc_bias", value=-vr)
    solve()
    out["I1d"] = NL.btbt_nonlocal_generation(d1, "b")["I_p"]
    x1 = np.asarray(ds.get_node_model_values(device=d1, region="b", name="x"))
    p1 = np.asarray(ds.get_node_model_values(device=d1, region="b", name="Potential"))
    o = np.argsort(x1)
    out["same_x_nodes"] = bool(x1.size == xrow.size and np.array_equal(x1[o], xrow))
    out["psi_2d_vs_1d_V"] = float(np.max(np.abs(p1[o] - psirow))) if out["same_x_nodes"] else None
    del t1
    ds.delete_device(device=d1)
    # local twin in 2-D: Newton iterations
    dl = build_2d("z2l", ZENER2D_X, ZENER2D_Y, dop, n_i=ZNI, tau=1e-3, local=SILICON_HURKX, **fdk)
    out["its_eq_local"] = solve()
    ds.set_parameter(device=dl, name="lo_bias", value=-vr)
    out["its_local"] = solve()
    out["I2d_local_per_W"] = btbt_generation_current(dl, "b") / W
    return out


SLANT_N, SLANT_NI = 1e22, 1e19
SLANT_LX, SLANT_LY, SLANT_XJ = 1.2e-6, 0.24e-6, 0.6e-6
SLANT_A, SLANT_B, SLANT_EG, SLANT_LMAX = 1e19, 1e7, 0.25, 3e-7
SLANTFD_NDOS = 2e21                 # N_c = N_v of the DEGENERATE (FD) slanted diode: N / N_c = 5, eta ~ 3.3


def build_slant(theta_deg=30.0, circuit=True, fd=False, statistics=None, n_dos=2.8e25):
    """A 2-D FDT-type diode (n_i = 1e19, 1e22 m^-3: ssac and noise are trustworthy, tests/_btbt_nonlocal_child
    header) whose junction is SLANTED by theta against the tensor mesh: paths cross triangles obliquely and
    turn with the field (the dynamic direction)."""
    ds = _ds()
    tan = math.tan(math.radians(theta_deg))
    xl = [(0.0, 40e-9), (SLANT_XJ - 0.2e-6, 8e-9), (SLANT_XJ + 0.2e-6, 8e-9), (SLANT_LX, 40e-9)]
    yl = [(0.0, 10e-9), (SLANT_LY, 10e-9)]
    dop = "ifelse(x - (y - {yc})*{t} < {xj}, {n}, {mn})".format(yc=SLANT_LY / 2, t=repr(tan), xj=SLANT_XJ,
                                                                n=SLANT_N, mn=-SLANT_N)
    dev = build_2d("sl", xl, yl, dop, n_i=SLANT_NI, tau=1e-6, circuit=circuit, ae=1e8, fd=fd,
                   statistics=statistics, n_dos=n_dos)
    if circuit:
        _CIRCUIT["made"] = True
    del ds
    return dev


def _drop_rotation(NL):
    """Refutation: the dynamic-direction Jacobian WITHOUT the rotation terms (the frozen-geometry part)."""
    real = NL.path_rates_2d

    def wrapped(*a, **k):
        r = real(*a, **k)
        if "jac_theta" in r:
            import scipy.sparse as sp
            r["jac_R"] = (r["jac_R"] - sp.diags(r["dR_dtheta"]) @ r["jac_theta"]).tocsr()
            r.pop("jac_theta")
        return r
    NL.path_rates_2d = wrapped


def sc_slant(opts):
    """Slanted FDT diode (2-D): detailed balance at 0 V, FDT with the two-point source (and the same-node
    prove-out), ssac vs DC dI/dV at 0 V and at -0.2 V, and the term's Jacobian vs central differences of
    DEVSIM's residual at -0.2 V (dynamic direction: the rotation terms included). opts: part ("eq": the 0 V
    block only; "rev_ssac" / "rev_jac" / "rev": the -0.2 V ssac and / or Jacobian only), fd / statistics /
    n_dos (the FD transport, its statistics, N_c = N_v), A (the Kane A), carriers (Jacobian columns of the
    carrier variables too), boltz_dL (perturbation: the occupancies' carrier derivatives with g = 1),
    drop_rotation (refutation), same_node (prove-out)."""
    import validation.devsim_noise as V
    import _custom_terms_child as CC
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers import custom_terms as CT
    from dynameta.carriers.tunneling import BTBTParams
    ds = _ds()
    t0 = time.time()
    dev = build_slant(opts.get("theta", 30.0), fd=bool(opts.get("fd", False)), statistics=opts.get("statistics"),
                      n_dos=float(opts.get("n_dos", 2.8e25)))
    term = NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(BTBTParams(float(opts.get("A", SLANT_A)), SLANT_B, 2.5),
                                                              Eg_tun_eV=SLANT_EG, max_length_m=SLANT_LMAX))
    if opts.get("boltz_dL"):            # perturbation: the occupancies' carrier derivatives with g = 1 (Boltzmann)
        real_state = term.state

        def boltz_state(device, real_state=real_state):
            st_ = real_state(device)
            for k, v in (("dLn", "Electrons"), ("dLp", "Holes")):
                st_[k] = 1.0 / np.asarray(ds.get_node_model_values(device=device, region="b", name=v))[st_["perm"]]
            return st_
        term.state = boltz_state
    V._solve(1e8, 1e-8)
    out = {"build_s": time.time() - t0, "stats": dict(term.last_stats),
           "nodes": len(ds.get_node_model_values(device=dev, region="b", name="x"))}
    st, res = term.evaluate(dev)
    one_way = 0.5 * res["R_pre"] * res["pair"]
    out["db_ratio"] = float(np.max(np.abs(res["R"]) / one_way))
    out["eta_max"] = float(max(np.max(st["Ln"]), np.max(st["Lp"])))      # the degeneracy reached (L = eta)
    out["J0"] = float(sum(ds.get_contact_current(device=dev, contact="lo", equation=e)
                          for e in ("ElectronContinuityEquation", "HoleContinuityEquation")))
    ff = [1.0, 1e5, 1e7]
    if opts.get("part", "").startswith("rev"):                   # skip the equilibrium noise block
        Y = sn = None
    else:
        t1 = time.time()
        Y = V.ssac_Y(ff)
        sn = V.noise(ff)
        out["noise_s"] = time.time() - t1
    if sn is not None:
        out["ReY"], out["S"] = _lst(Y.real), _lst(sn.S)
        out["share_btbt"] = float(sn.by_mechanism["btbt_nl"][0] / sn.S[0])
        out["G0"], out["G0_1"] = g_rich(1e-4)
        out["G0_2"] = g_rich(2e-4)[0]
    if opts.get("same_node"):
        real = term.noise_sources
        _same_node(term)
        out["S_same"] = _lst(V.noise(ff).S)
        term.noise_sources = real
    if opts.get("part") == "eq":
        out["total_s"] = time.time() - t0
        return out
    V.set_bias(-0.2, 1e8, step=0.05)
    V._solve(1e8, 1e-8)
    g = NL.btbt_nonlocal_generation(dev, "b")
    out["Ip_r"], out["In_r"] = g["I_p"], g["I_n"]
    if opts.get("part") != "rev_jac":                          # "rev_ssac" (or the whole run)
        out["Yr"] = _lst(V.ssac_Y([1.0]).real)
        out["Gr"], out["Gr_1"] = g_rich(1e-4)
        out["Gr_2"] = g_rich(2e-4)[0]
        if opts.get("part") == "rev_ssac":
            out["total_s"] = time.time() - t0
            return out
    # Jacobian vs FD of DEVSIM's residual at -0.2 V
    if opts.get("drop_rotation"):
        _drop_rotation(NL)
    realA = term.assemble
    term.active = True
    term.assemble = lambda d, w, t: realA(d, w, t) if term.active else ([], [])
    st, res = term.evaluate(dev, derivatives=True)
    ps = res["pathset"]
    k = int(np.argmax(np.abs(res["R"])))
    i = int(ps.start[k])
    pts = np.flatnonzero(ps.pt_path == k)
    mid = int(ps.pt_n1[pts[len(pts) // 2]])
    patch = st["gops"][0][i].indices.tolist()                     # the start's gradient patch: theta(psi)
    e_nodes = sorted(set(res["e_nodes"][k].tolist()))
    patch = [j for j in patch if j not in (i, mid)]
    cols = sorted(set([i, mid] + patch[:2] + e_nodes))
    rows = np.concatenate([CT.node_rows(dev, "b", "HoleContinuityEquation", [i]),
                           CT.node_rows(dev, "b", "ElectronContinuityEquation", e_nodes)])
    out["recs"] = CC.fd_columns(dev, "b", "Potential", cols, rows, lambda x: 1e-6, term)
    if opts.get("carriers"):            # the occupancies' columns: holes at the start, electrons at the end triangle
        out["recs_n"] = CC.fd_columns(dev, "b", "Electrons", e_nodes, rows, lambda x: 1e-6 * abs(x), term)
        out["recs_p"] = CC.fd_columns(dev, "b", "Holes", [i], rows, lambda x: 1e-6 * abs(x), term)
    out["dR_dtheta"] = float(res["dR_dtheta"][k]) if "dR_dtheta" in res else None
    out["R_k"] = float(res["R"][k])
    out["total_s"] = time.time() - t0
    return out


CORNER_L, CORNER_C = 120e-9, 60e-9


def sc_corner(opts):
    """MEASUREMENT: an n+ quadrant (x > c, y > c) in a p+ square (5e24 / 5e24 m^-3, Si Hurkx in both models):
    planar junction arms along x = c (y > c) and y = c (x > c) meeting at a corner (convex for n, concave for
    p). Per bias: the nonlocal device's hole / electron generation per node, the self-consistent local
    device's total (DEVSIM's own integral), and on EACH device's solution the local Kane-Hurkx rate at the
    nodes (A |grad psi|^P exp(-B/|grad psi|) (-D) NodeVolume, nodal gradient = the 2-D recovery) for the
    spatial comparison."""
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers import btbt_nonlocal_2d as B2
    from dynameta.carriers.tunneling import SILICON_HURKX, btbt_generation_current
    ds = _ds()
    h = float(opts.get("h", 0.5e-9))                     # spacing AT the junction lines x = c, y = c
    vrs = [float(v) for v in opts.get("vr", [0.9, 1.5])]
    L, c = CORNER_L, CORNER_C
    xl = [(0.0, 2e-9), (c, h), (L, 2e-9)]                 # graded tensor mesh, finest on the junction lines
    if opts.get("geom", "corner") == "corner":
        dop = "ifelse((x > {c}) && (y > {c}), {n}, {mn})".format(c=c, n=ZN, mn=-ZN)
        contacts = [("lo", 0.0, L, 0.0, 0.0), ("hi", c + 20e-9, L, L, L)]
    else:                            # the planar reference: the same square, junction along y = c
        dop = "ifelse(y > {c}, {n}, {mn})".format(c=c, n=ZN, mn=-ZN)
        contacts = [("lo", 0.0, L, 0.0, 0.0), ("hi", 0.0, L, L, L)]
    out = {"h": h, "rows": [], "geom": opts.get("geom", "corner")}
    A, B, P = SILICON_HURKX.A_SI, SILICON_HURKX.B_V_per_m, SILICON_HURKX.P
    for kind in ("nonlocal", "local"):
        t0 = time.time()
        dev = build_2d("cn" + kind[0] + out["geom"][0], xl, xl, dop, n_i=ZNI, tau=1e-3, contacts=contacts,
                       fd=bool(opts.get("fd", False)), statistics=opts.get("statistics"),
                       local=SILICON_HURKX if kind == "local" else None)
        its = []
        if kind == "nonlocal" and opts.get("fd"):
            # FD: the equilibrium from the seed WITHOUT the term first (the bare FD device needs ~290 Newton
            # steps from the seed at h = 0.5 nm, and with the term installed it does not converge in 400;
            # MEASURED 2026-11-08), then the term (1 more step). Boltzmann: as recorded (term first, 45 steps).
            its.append(solve(it=400))
        term = (NL.setup_btbt_nonlocal(dev, "b", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12))
                if kind == "nonlocal" else None)
        its.append(solve(it=400))
        xy = _xy(dev)
        tri = np.asarray(ds.get_element_node_list(device=dev, region="b"), np.int64)
        gops = B2.gradient_ops_2d(B2.build_mesh2d(xy, tri))
        vol = np.asarray(ds.get_node_model_values(device=dev, region="b", name="NodeVolume"))
        done = 0.0
        for vr in vrs:
            n = max(1, int(math.ceil((vr - done) / 0.15)))
            for j in range(1, n + 1):
                ds.set_parameter(device=dev, name="lo_bias", value=-(done + (vr - done) * j / n))
                its.append(solve())
            done = vr
            psi = np.asarray(ds.get_node_model_values(device=dev, region="b", name="Potential"))
            nn = np.asarray(ds.get_node_model_values(device=dev, region="b", name="Electrons"))
            pp = np.asarray(ds.get_node_model_values(device=dev, region="b", name="Holes"))
            F = np.hypot(gops[0] @ psi, gops[1] @ psi)
            D = (nn * pp - ZNI ** 2) / ((nn + ZNI) * (pp + ZNI))
            with np.errstate(under="ignore"):
                Gloc = A * np.exp(P * np.log(np.maximum(F, 1.0)) - B / np.maximum(F, 1.0)) * (-D) * vol
            row = {"kind": kind, "vr": vr, "Gloc_post": _lst(Gloc)}
            if kind == "nonlocal":
                g = NL.btbt_nonlocal_generation(dev, "b")
                row.update(Gp=_lst(g["G_p"]), Gn=_lst(g["G_n"]), I=g["I_p"], stats=g["stats"])
            else:
                row["I"] = btbt_generation_current(dev, "b")
            out["rows"].append(row)
        out[kind + "_its"] = its
        out[kind + "_s"] = time.time() - t0
        out["vol"] = _lst(vol)
        out["x"], out["y"] = _lst(xy[:, 0]), _lst(xy[:, 1])
        del term
        ds.delete_device(device=dev)
    return out


SCENARIOS = {"defaults": sc_defaults, "fdt": sc_fdt, "zener": sc_zener, "flip": sc_flip,
             "jacobian": sc_jacobian, "equilibrium": sc_equilibrium, "warn_T": sc_warn_T,
             "reverse_refusal": sc_reverse_refusal, "gap_material": sc_gap_material,
             "lat2d": sc_lat2d, "slant": sc_slant, "corner": sc_corner}


def run_child(scenario, *, timeout=900, **opts):
    """Run `scenario` in a fresh interpreter on this sys.path; returns its JSON result (timeout [s]: the
    child's wall-clock limit)."""
    import subprocess
    import tempfile
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), scenario, out, json.dumps(opts)],
                           env=env, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError("child {} failed:\n{}".format(scenario, r.stderr[-4000:]))
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.remove(out)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error")                       # warnings are errors, as in the suite
    # the suite's DC well-posedness hook (tests/conftest.py, plan 4.4.2) on this child's raw DC solves too
    if sys.argv[1] != "defaults" and not os.environ.get("NL_NO_HOOK"):   # defaults measures import side effects
        import conftest
        _ds()
        conftest._install_dc_well_posedness_hook()
    _opts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    _res = SCENARIOS[sys.argv[1]](_opts)
    with open(sys.argv[2], "w", encoding="utf-8") as _f:
        json.dump(_res, _f)
