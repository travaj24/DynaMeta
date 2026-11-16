"""Equilibrium in-Newton density gradient (carriers.physics_equilibrium_dg, v0.16) against the dual-gate
Schrodinger-Poisson solver (carriers.sp_dual_gate.DualGateSchrodingerPoisson).

Device (validation/_eqdg_fixture.py): a degenerate film, parabolic band, gated on both faces through an oxide
stack ("std": 6 nm eps 20 + 2 nm eps 9 from the film outward; "thin": 2 nm eps 25), film eps_s 9.0, metal gates
"bot" / "top", built through
LayeredDevsimBuilder(density_gradient={"film": EquilibriumDensityGradient(...)}) in a one-cell-wide 2-D cell
(laterally uniform), on the builder's DEFAULT mesh. Oracle: the same stack in DualGateSchrodingerPoisson with
h <= 0.0125 nm. Metrics (z from the film's lower face):
  dsheet : (sheet excess of DG - of SP) / S               -- the electrostatics (gate charge)
  L1     : int |n_DG - n_SP| dz / S                      -- the profile (SP's Friedel oscillations included)
           S = max(n_bg t, the SP carrier sheet): the film's carrier sheet, or its background sheet when depleted
  dz50   : distance from each face to where n first reaches half its maximum within 1.5 nm of that face, DG - SP
           [nm] -- the dead layer

GATE A  the shipped default gamma (EQUILIBRIUM_DG_GAMMA) over the matrix (default: a reduced matrix; --full a
        t 3/6/10/20/40 nm x n_bg 1e26/3e26/1e27 m^-3 x m 0.2/0.30 m_e matrix on the "std" stack, 8 bias pairs,
        plus the thin high-k stack): every point inside the validity range (check_equilibrium_dg_validity) within
        the stated budget, and every point converges. (The default gamma is calibrated on the --full grid:
        carriers/physics_equilibrium_dg.py, EQUILIBRIUM_DG_GAMMA.)
GATE B  the classical model on the same points is far worse (the correction is doing the work): max |dsheet|
        classical >= 4x the DG one.
GATE C  (--full) the gamma scan behind the default: the per-point sheet-matching gamma* and the gamma
        minimizing the worst |dsheet|.
Run: python -m validation.equilibrium_density_gradient [--full]
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation._harness import Harness            # noqa: E402

H = Harness("eqdg", "equilibrium density gradient vs dual-gate Schrodinger-Poisson")

import importlib.util                                # noqa: E402
HAVE_DEVSIM = importlib.util.find_spec("devsim") is not None     # (devsim's ONE import home is devsim_env)

B_STD = [(0.0, -3.0), (0.0, -2.0), (0.0, 0.0), (0.0, 2.0), (0.0, 3.0), (0.0, 5.0), (3.0, 3.0), (-3.0, -3.0)]
B_THIN = [(0.0, -1.5), (0.0, -0.5), (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]

REDUCED = [(6e-9, 3e26, 0.30, "std", B_STD), (10e-9, 1e26, 0.2, "std", B_STD), (3e-9, 1e27, 0.30, "thin", B_THIN),
           (20e-9, 1e27, 0.30, "std", B_STD[::2])]


def full_matrix():
    cases = []
    for t in (3e-9, 6e-9, 10e-9, 20e-9, 40e-9):
        for nbg in (1e26, 3e26, 1e27):
            for m in (0.2, 0.30):
                cases.append((t, nbg, m, "std", B_STD))
    for t in (3e-9, 6e-9, 20e-9):
        for nbg in (1e26, 1e27):
            cases.append((t, nbg, 0.30, "thin", B_THIN))
    return cases


def dz50(z, n, window=1.5e-9):
    """(lower, upper) dead-layer widths: the distance from each face to the first node where n reaches half of its
    maximum within `window` of that face."""
    lo_m = z <= z[0] + window
    up_m = z >= z[-1] - window
    lo = float(z[int(np.argmax(n >= 0.5 * float(np.max(n[lo_m]))))] - z[0])
    k = len(z) - 1 - int(np.argmax(n[::-1] >= 0.5 * float(np.max(n[up_m]))))
    return lo, float(z[-1] - z[k])


def run_case(t, nbg, m_rel, ox, biases, gammas, tag="v"):
    """{gamma: [row per bias]} for gamma in gammas (0 = classical), rows with dsheet, l1, dz50 errors."""
    from validation import _eqdg_fixture as FX
    from dynameta.constants import M_E
    from dynameta.carriers.density_gradient import EquilibriumDensityGradient
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    m = m_rel * M_E
    sp = FX.oracle(t, nbg, m, ox)
    sols = {vb: sp.solve_voltages(*vb) for vb in biases}
    out = {}
    for g in gammas:
        kw = {} if g == 0.0 else {"density_gradient": {"film": EquilibriumDensityGradient(gamma=g, m_eff_kg=m)}}
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            b = LayeredDevsimBuilder(FX.design(t, nbg, m, ox), device_name=tag + "d", mesh_name=tag + "m", **kw)
            rows = []
            try:
                for vb in biases:
                    t0 = time.time()
                    try:
                        b.solve(BiasPoint({"bot": vb[0], "top": vb[1]}, "p"))
                    except Exception as e:                       # noqa: BLE001 (a failure is a reported row)
                        rows.append({"vb": vb, "error": str(e)[:120]})
                        continue
                    z, n = FX.column(b.device)
                    r = FX.compare(z, n, sols[vb], nbg, t)
                    lo, up = dz50(z, n)
                    los, ups = dz50(sols[vb].z_m, sols[vb].n_m3)
                    r.update(vb=vb, dz50_lo=lo - los, dz50_up=up - ups, dt=time.time() - t0)
                    rows.append(r)
            finally:
                b.teardown()
        unreliable = any("outside the calibrated range" in str(w.message) for w in rec)
        for r in rows:
            r["out_of_range"] = unreliable
        out[g] = rows
    return out


def main():
    full = "--full" in sys.argv
    H.require(HAVE_DEVSIM, "devsim not installed")
    from dynameta.carriers import physics_equilibrium_dg as EQDG
    g0 = EQDG.EQUILIBRIUM_DG_GAMMA
    cases = full_matrix() if full else REDUCED
    gammas = [0.0, g0] + ([1.0, 1.5, 1.7, 1.9, 2.1, 2.4] if full else [])
    print("[eqdg] default gamma {}; {} cases; budget inside the validity range: |dsheet| <= {:.1%}, L1 <= {:.2f}"
          .format(g0, len(cases), EQDG.BUDGET_DSHEET, EQDG.BUDGET_L1), flush=True)
    worst = {"dg": 0.0, "cl": 0.0, "l1": 0.0, "fail": 0, "dz": 0.0}
    scan = []
    for (t, nbg, mr, ox, biases) in cases:
        res = run_case(t, nbg, mr, ox, biases, sorted(set(gammas)))
        for vb_i, vb in enumerate(biases):
            rd, rc = res[g0][vb_i], res[0.0][vb_i]
            if "error" in rd:
                worst["fail"] += 1
                print("[eqdg]   t={:4.1f} nm n={:.0e} m={:.2f} {:4s} V={} DG FAILED: {}".format(
                    t * 1e9, nbg, mr, ox, vb, rd["error"]), flush=True)
                continue
            inside = not rd["out_of_range"]
            print("[eqdg]   t={:4.1f} nm n={:.0e} m={:.2f} {:4s} V=({:+.1f},{:+.1f}) dsheet DG {:+.3%} classical {:+.3%}"
                  "  L1 DG {:.3f} classical {:.3f}  dz50 lo/up {:+.3f}/{:+.3f} nm{}".format(
                      t * 1e9, nbg, mr, ox, vb[0], vb[1], rd["dsheet"], rc["dsheet"], rd["l1"], rc["l1"],
                      rd["dz50_lo"] * 1e9, rd["dz50_up"] * 1e9, "" if inside else "  (outside range: warned)"),
                  flush=True)
            if inside:
                worst["dg"] = max(worst["dg"], abs(rd["dsheet"]))
                worst["cl"] = max(worst["cl"], abs(rc["dsheet"]))
                worst["l1"] = max(worst["l1"], rd["l1"])
                worst["dz"] = max(worst["dz"], abs(rd["dz50_lo"]), abs(rd["dz50_up"]))
        if full:
            for g, rows in res.items():
                for vb, r in zip(biases, rows):
                    scan.append((t, nbg, mr, ox, vb, g, r))
    H.gate("A", worst["fail"] == 0 and worst["dg"] <= EQDG.BUDGET_DSHEET and worst["l1"] <= EQDG.BUDGET_L1,
           "default gamma {}: worst |dsheet| {:.3%} (budget {:.1%}), worst L1 {:.3f} (budget {:.2f}), worst |dz50| "
           "{:.3f} nm, {} failed solves".format(g0, worst["dg"], EQDG.BUDGET_DSHEET, worst["l1"], EQDG.BUDGET_L1,
                                                worst["dz"] * 1e9, worst["fail"]))
    H.gate("B", worst["cl"] >= 4.0 * worst["dg"],
           "classical worst |dsheet| {:.3%} vs DG {:.3%} (ratio {:.1f})".format(
               worst["cl"], worst["dg"], worst["cl"] / max(worst["dg"], 1e-30)))
    if full:
        gs = sorted(set(s[5] for s in scan if s[5] > 0.0))
        line = []
        for g in gs:
            d = [abs(s[6]["dsheet"]) for s in scan if s[5] == g and "error" not in s[6] and not s[6]["out_of_range"]]
            nf = sum(1 for s in scan if s[5] == g and "error" in s[6])
            line.append((max(d) if d else np.inf, g, nf))
            print("[eqdg]   gamma {:.2f}: worst |dsheet| in range {:.3%}, median {:.3%}, failures {}".format(
                g, max(d) if d else np.nan, float(np.median(d)) if d else np.nan, nf), flush=True)
        best = min(line)
        H.gate("C", abs(best[1] - g0) <= 0.2 + 1e-12,
               "gamma minimizing the worst in-range |dsheet| on the scan grid: {:.2f} ({:.3%}); default {}".format(
                   best[1], best[0], g0))
    else:
        H.skip("C", "the gamma scan runs with --full (about 15 min on the dev box)")
    return H.summary()


if __name__ == "__main__":
    raise SystemExit(main())
