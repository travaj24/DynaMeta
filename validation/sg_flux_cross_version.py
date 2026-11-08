"""
THE SG FLUX (dynameta.carriers.sg_flux, 2026-11-04): accuracy of every flux family, and the
cross-version guard of the defaults-off bipolar solve (D1d's solution-array identity, which was a
sha256 comparison until the flux changed on purpose). docs/audit/2026-11-04-sg-flux-two-bernoulli.md.

  GATE A  every flux family (unipolar DD electrons, bipolar DD electrons / holes, unipolar DG
          electrons, bipolar DG electrons / holes) on one edge vs a 50-digit mpmath flux and
          Jacobian at |v| up to 40: < 1e-13 (tests/test_devsim_sg_flux.py's oracle, run here too).
  GATE A' the SAME edges evaluated with the pre-2026-11-04 kahan3 strings (B(-v) spelled B(v) + v)
          FAIL A: the old flux, in this tree, at the v DEVSIM computed.
  GATE B  CROSS-VERSION: D1d's defaults-off fixtures -- validation.bipolar_diode (FD on) at +0.3 /
          -0.5 V and the validation.bipolar_diode_2d design through LayeredDevsimBuilder at
          +0.3 / -0.4 V -- solved on a reference checkout (--ref, default the main checkout next to
          this worktree, kahan3 everywhere) and on this tree; every field identical to round-off
          (TOL_B) and the terminal currents to the round-off of their contact edges' terms.
  GATE C  Newton convergence unchanged: per-solve iteration counts on the GATE B paths equal to
          the reference's within 1, and the totals within 5 %.
  REPORT  max |v| per edge (each carrier's Bernoulli argument) on the GATE B states.

Run:  python -m validation.sg_flux_cross_version [--ref <checkout>]
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from validation._harness import Harness                                 # noqa: E402

DEFAULT_REF = os.path.join(os.path.dirname(os.path.dirname(HERE)), "DynaMeta")
EPS = 2.220446049250313e-16

# GATE B tolerances, from the MEASURED move (main 5cbd998 = 6595bde's flux -> this tree, 2026-11-04,
# scipy OpenBLAS; the numbers are printed by the gate): fields (Potential relative to its max |.|,
# densities entry-wise) and terminal currents |dI| <= TOL_B |I| + 4 eps T, T an upper bound of each
# drift / diffusion term of the region's edges, q V_t / min(h) max(mu_n n + mu_p p) (a majority
# contact current is their eps T round-off in EITHER flux spelling -- tests/_sgflux_reference.py).
TOL_B = 1e-11
TOL_I_TERM = 4.0 * EPS

_FIXTURE = textwrap.dedent('''
    import json, sys
    import numpy as np
    try:
        from dynameta.carriers.devsim_env import import_devsim
        ds = import_devsim()
    except ImportError:                                  # main has no devsim_env
        import devsim as ds
    ITER = []
    _solve = ds.solve
    def solve(*a, **kw):                                 # record Newton iterations, same semantics
        kw["info"] = True
        r = _solve(*a, **kw)
        ITER.append(len(r["iterations"]))
        if not r["converged"]:
            raise ds.error("Convergence failure!")
        return r
    ds.solve = solve
    OUT = {}
    def nv(dev, reg, name):
        return [float(x) for x in ds.get_node_model_values(device=dev, region=reg, name=name)]
    def maxv(dev):
        m = {}
        for reg in ds.get_region_list(device=dev):
            em = set(ds.get_edge_model_list(device=dev, region=reg))
            for v in ("vdiff_g", "vdiff_gp", "vdiff_g_dg", "vdiff_gp_dg"):
                if v in em:
                    a = np.abs(np.asarray(ds.get_edge_model_values(device=dev, region=reg, name=v)))
                    m[v] = max(m.get(v, 0.0), float(a.max()) if a.size else 0.0)
        return m
    def term(dev, reg):
        """UPPER BOUND of each drift / diffusion term of any edge of the region:
        q V_t / min(h) * max(mu_n n + mu_p p)."""
        par = lambda nm: float(ds.get_parameter(device=dev, region=reg, name=nm))
        h = float(np.min(ds.get_edge_model_values(device=dev, region=reg, name="EdgeLength")))
        n = np.asarray(nv(dev, reg, "Electrons")); p = np.asarray(nv(dev, reg, "Holes"))
        return float(par("ElectronCharge") * par("V_t") / h * np.max(par("mu_n") * n + par("mu_p") * p))
    # ---- 1-D: validation.bipolar_diode, FD on ----
    import validation.bipolar_diode as BD
    BD.build_diode(True)
    BD.staged_equilibrium_solve()
    for v in (0.3, -0.5):
        k0 = len(ITER)
        BD.ramp_to(v)
        key = "d1_{:+.1f}".format(v)
        for f in ("Potential", "Electrons", "Holes"):
            OUT[key + ":" + f] = nv(BD.DEVICE, BD.REGION, f)
        jt, jn, jp = BD.terminal_current("p_contact")
        OUT[key + ":I"] = [float(jn), float(jp)]
        OUT[key + ":T"] = [term(BD.DEVICE, BD.REGION)] * 2
        OUT[key + ":maxv"] = maxv(BD.DEVICE)
        OUT[key + ":iters"] = ITER[k0:]
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    # ---- 2-D: validation.bipolar_diode_2d through LayeredDevsimBuilder ----
    import validation.bipolar_diode_2d as B2
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    b = LayeredDevsimBuilder(B2.build_diode_design(), mesh_name="sgx_m", device_name="sgx_d")
    for v in (0.3, -0.4):
        k0 = len(ITER)
        cf = b.solve(BiasPoint({"anode": v}, "v"))
        key = "d2_{:+.1f}".format(v)
        for r in sorted(cf.regions):
            for f, a in sorted(cf.regions[r].node_fields.items()):
                OUT["{}:{}/{}".format(key, r, f)] = [float(x) for x in np.ravel(a)]
        I = [float(ds.get_contact_current(device=b.device, contact="anode", equation=e))
             for e in ("ElectronContinuityEquation", "HoleContinuityEquation")]
        OUT[key + ":I"] = I
        semis = [r for r in ds.get_region_list(device=b.device)
                 if "Holes" in ds.get_node_model_list(device=b.device, region=r)]
        for r in semis:                                  # the CarrierField carries no hole density
            OUT["{}:{}/Holes".format(key, r)] = nv(b.device, r, "Holes")
        OUT[key + ":T"] = [max(term(b.device, r) for r in semis)] * 2
        OUT[key + ":maxv"] = maxv(b.device)
        OUT[key + ":iters"] = ITER[k0:]
    with open(sys.argv[1], "w") as f:
        json.dump(OUT, f)
''')


def run_fixture(tree):
    """(rc, results dict, stderr tail) of _FIXTURE on `tree` (cwd + PYTHONPATH = tree)."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ, PYTHONPATH=tree)
        r = subprocess.run([sys.executable, "-c", _FIXTURE, path], cwd=tree, env=env,
                           capture_output=True, text=True, timeout=3000)
        out = {}
        if r.returncode == 0:
            with open(path, encoding="utf-8") as f:
                out = json.load(f)
        return r.returncode, out, r.stderr[-800:]
    finally:
        os.remove(path)


def compare(old, new):
    """[(key, metric, ok)]: Potential-like fields relative to max |old|, positive densities entry-
    wise, currents |dI| / (TOL_B |I| + 4 eps T) against 1."""
    rows = []
    for key in sorted(k for k in old if k.rsplit(":", 1)[1] not in ("T", "maxv", "iters")):
        a, b = np.asarray(old[key], dtype=float), np.asarray(new.get(key, []), dtype=float)
        if a.shape != b.shape:
            rows.append((key, math.inf, False))
            continue
        d = np.abs(b - a)
        if key.endswith(":I"):
            t = np.asarray(old[key[:-1] + "T"], dtype=float)
            m = float(np.max(d / (TOL_B * np.abs(a) + TOL_I_TERM * t)))
            rows.append((key, m, m <= 1.0))
        elif a.size and np.all(a > 0):
            m = float(np.max(d / a))
            rows.append((key, m, m <= TOL_B))
        else:
            s = float(np.max(np.abs(a))) if a.size else 0.0
            m = float(np.max(d) / s) if s > 0 else float(np.max(d, initial=0.0))
            rows.append((key, m, m <= TOL_B))
    return rows


# the pre-2026-11-04 kahan3 brackets (B(-v) spelled B(v) + v), for GATE A' only
_K3 = {"electrons": "kahan3({c}@n1*{b}, {c}@n1*{v}, -{c}@n0*{b})",
       "holes": "kahan3({c}@n1*{b}, -{c}@n0*{b}, -{c}@n0*{v})"}


def gate_a_old(T):
    """(worst value error, worst Jacobian error) per family of the kahan3 strings on GATE A's edges
    (sg_flux.sg_bracket swapped for the old bracket while the edges are built)."""
    from dynameta.carriers import sg_flux as SG
    out = {}
    real = SG.sg_bracket

    def k3(carrier, density, vdiff, bern, bern_m):
        return _K3[carrier].format(c=density, v=vdiff, b=bern)

    SG.sg_bracket = k3
    try:
        for fam in T._FAMILIES:
            out[fam] = T.flux_accuracy(fam, vs=(-40.0, -33.0, -27.6, -21.0, 21.0, 27.6, 33.0, 40.0))
    finally:
        SG.sg_bracket = real
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    a = ap.parse_args()
    H = Harness("sgx", "SG flux: every family accurate; defaults-off solve == reference to round-off")
    H.require(importlib.util.find_spec("devsim") is not None, "devsim not installed")
    H.require(importlib.util.find_spec("mpmath") is not None, "mpmath not installed")
    spec = importlib.util.spec_from_file_location("_sgf_tests", os.path.join(HERE, "tests",
                                                                           "test_devsim_sg_flux.py"))
    T = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.join(HERE, "tests"))
    spec.loader.exec_module(T)

    print("[sgx] --- A: every flux family vs mpmath, |v| <= 40", flush=True)
    worst = {}
    for fam in T._FAMILIES:
        wv, wd = T.flux_accuracy(fam)
        worst[fam] = (wv, wd)
        print("[sgx]   {:22s} value {:.1e}  Jacobian {:.1e}".format(fam, wv, wd), flush=True)
    H.gate("A every family accurate", all(max(w) < 1e-13 for w in worst.values()),
           "worst {:.1e} < 1e-13".format(max(max(w) for w in worst.values())))
    old = gate_a_old(T)
    for fam, (wv, wd) in old.items():
        print("[sgx]   kahan3 {:15s} value {:.1e}  Jacobian {:.1e}".format(fam, wv, wd), flush=True)
    H.gate("A' the kahan3 strings fail A", all(w[0] > 1e-3 and w[1] > 1e-3 for w in old.values()),
           "every family's old value AND Jacobian error > 1e-3 at |v| <= 40")

    if not os.path.isdir(os.path.join(a.ref, "dynameta")):
        H.skip("B/C cross-version", "no reference checkout at {} (--ref)".format(a.ref))
        raise SystemExit(H.summary())
    print("[sgx] --- B: defaults-off fixtures, {} vs this tree".format(a.ref), flush=True)
    rc0, r0, e0 = run_fixture(a.ref)
    rc1, r1, e1 = run_fixture(HERE)
    rows = compare(r0, r1) if rc0 == rc1 == 0 else []
    for key, m, ok in rows:
        print("[sgx]   {:44s} {:.2e}{}".format(key, m, "" if ok else "  <-- over"), flush=True)
    H.gate("B identical to round-off", rc0 == rc1 == 0 and len(rows) == 16 and all(r[2] for r in rows),
           "rc {} / {}, {} arrays, worst field {:.1e} (TOL_B {:.0e}), worst current {:.2f} of tol".format(
               rc0, rc1, len(rows), max((r[1] for r in rows if not r[0].endswith(":I")), default=math.inf),
               TOL_B, max((r[1] for r in rows if r[0].endswith(":I")), default=math.inf))
           + ("" if rc0 == rc1 == 0 else " | " + e0 + " | " + e1))
    if rc0 == rc1 == 0:
        its = [(k[:-6], r0[k], r1[k]) for k in sorted(r0) if k.endswith(":iters")]
        per = all(len(o) == len(n) and all(abs(x - y) <= 1 for x, y in zip(o, n)) for _k, o, n in its)
        tot0, tot1 = sum(sum(o) for _k, o, _n in its), sum(sum(n) for _k, _o, n in its)
        for k, o, n in its:
            print("[sgx]   iterations {:8s} ref {}  this {}".format(k, o, n), flush=True)
        H.gate("C Newton iterations unchanged", per and abs(tot1 - tot0) <= 0.05 * tot0,
               "per solve within 1, total {} vs {}".format(tot1, tot0))
        for k in sorted(r1):
            if k.endswith(":maxv"):
                print("[sgx]   REPORT max |v| {:8s} {}".format(k[:-5], "  ".join(
                    "{} {:.2f}".format(m, x) for m, x in sorted(r1[k].items()))), flush=True)
    raise SystemExit(H.summary())


if __name__ == "__main__":
    main()
