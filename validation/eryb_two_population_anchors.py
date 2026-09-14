"""Er:Yb TWO-POPULATION model against the measured device anchors -- a VALIDATION SCRIPT, not a
test: it reports residuals, it does not assert them, and nothing here is tuned to the anchors
beyond the literature ranges the 2026-09-14 audit note tabulates.

WHAT IT ANSWERS. The 2026-09-14 validation record (scratchpad eryb/validation/
VALIDATION_SUMMARY.md) closed with a specific failure of the SINGLE-population model: at the
device-validated effective coefficient k_tr = 1e-21 m^3/s it reproduces the small-core
cladding-pumped stages and the vendor PCE, but OVER-PREDICTS the large-core and > 20 W points by
1.4-2x, and it has no mechanism at all for the FORC observation that the 1-um ASE fraction is set
by the fiber's energy-transfer efficiency (ETE 0.2-0.4 -> 21%-0.5% of the output at 1 um; ETE 0.9
-> 0.05%). Both gaps are the SAME missing physics: ytterbium that is pumped but not coupled to
the erbium. This script re-runs the anchors with it.

THREE COLUMNS, so the two changes can be told apart:
  (A) single population, k_tr = 1e-21, the validation record's own ion pair (aluminosilicate Er
      re-labelled + parametric phosphosilicate Yb). This is the BEFORE column and it reproduces
      the numbers in VALIDATION_SUMMARY.md section 2.
  (B) two populations, (f, k_fast, K2) = (0.9, 3.0e-21, 2.0e-22), SAME ions. f = 0.9 is the
      high-ETE commercial-fiber end of the measured range (FORC put Nufern LMA-EYDF-25P/300 at
      ~0.9; Sefler 2004 measured 0.83-0.85); k_fast = 3.07e-21 is Cheng et al. 2022's fast-pair
      rate from a bi-exponential Yb decay; K2 = 2e-22 is inside Sefler's 1.5-4e-22.
  (C) the same two-population parameters with the MEASURED phosphosilicate spectroscopy that
      shipped alongside (erbium("phosphosilicate") + ytterbium_melkumov("phosphosilicate")),
      which isolates the spectroscopy change from the population change.
  (D) the CONTROL, and the column to read for "does the pool help": two populations at f = 0.9
      with k_tr chosen so the PRODUCT f * k_tr equals column A's 1e-21, i.e. the same effective
      transfer strength a single-population fit sees. Without it B and A differ in TWO things at
      once (B's f * k_fast is 2.7e-21, 2.7x A's), and the pool's own contribution cannot be read
      off. D is not a fit: f * k_tr is exactly A's device-validated effective coefficient and f
      is the measured high-ETE value, so nothing is tuned to the anchors here either.

ANCHORS: Exail (vendor PCE), Bai 2015 (two cladding-pumped stages), Wei 2020, the Hannover
core-pumped 1018 nm series (de Varona Ortega 2019 Ch. 6), and the Morasse 2006 cut-back. Their
fiber parameters are transcribed from the anchor JSONs written by the literature agents; the
record LOADER is reused from that folder's `run_anchors.py` (densities back-computed from the
catalogue dB/m with the study's own cross-sections and Marcuse overlap) rather than re-derived.

Run:  python -m validation.eryb_two_population_anchors [--anchors DIR] [--nodes 401]
The anchor folder defaults to $DYNAMETA_ERYB_ANCHORS and then to the 2026-09-14 scratchpad path;
when it is absent the script says so and exits 0 (it is a reporting tool, not a gate).
"""
import importlib.util
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fiber_amp import (                                        # noqa: E402
    erbium, ytterbium, ytterbium_melkumov, FiberSpec, Pump, Signal, AseBand,
    PumpSource, wall_plug_efficiency,
)
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier                       # noqa: E402

_DEFAULT_ANCHORS = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "Temp", "claude", "C--Users-Tesla", "3f69d565-aa3a-47f5-ae6a-4e270f7ee8db",
    "scratchpad", "eryb", "validation")

ARGS = sys.argv[1:]
ANCHOR_DIR = (ARGS[ARGS.index("--anchors") + 1] if "--anchors" in ARGS
              else os.environ.get("DYNAMETA_ERYB_ANCHORS", _DEFAULT_ANCHORS))
NODES = int(ARGS[ARGS.index("--nodes") + 1]) if "--nodes" in ARGS else 401

# The parameter sets. NOTHING below is fitted here; every value is sourced in the audit note.
K_SINGLE = 1.0e-21              # device-validated effective single-population coefficient
F_COUPLED = 0.9                 # high-ETE commercial fiber (FORC; Sefler 0.83-0.85)
K_FAST = 3.0e-21                # Cheng 2022 fast-pair rate, 3.07e-21
K2 = 2.0e-22                    # Sefler 2004 secondary transfer, 1.5-4.0e-22
C_UP = 1.1e-24                  # Hwang 2000 bulk phosphate (the conservative end)


def _load_loader():
    """Import the anchor folder's own run_anchors.py -- the record loader, the density
    back-computation and the vendor product lookup are reused verbatim rather than re-derived."""
    path = os.path.join(ANCHOR_DIR, "run_anchors.py")
    if not os.path.isfile(path):
        return None
    sys.path.insert(0, ANCHOR_DIR)
    spec = importlib.util.spec_from_file_location("_eryb_run_anchors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the five anchor families, in run_anchors' own record schema ----------------------------
# Every fiber number is transcribed from the anchor JSONs / datasheets the literature agents
# recorded; the "measured" field is what the source states.
def _rec(label, *, core_um, na, L, clad_um, pump_nm, P, direction, geometry,
         er_abs, er_nm, pump_abs, pump_abs_nm, pump_abs_clad, sig_nm, p_in, measured, bg=50.0):
    fiber = dict(core_diameter_um=core_um, na=na, length_m=L, clad_diameter_um=clad_um,
                 er_absorption_dB_per_m=er_abs, er_absorption_wavelength_nm=er_nm,
                 background_loss_dB_per_km=bg)
    if pump_abs_clad:
        fiber["clad_pump_absorption_dB_per_m"] = pump_abs
        fiber["clad_pump_absorption_wavelength_nm"] = pump_abs_nm
    else:
        fiber["yb_or_pump_absorption_dB_per_m"] = pump_abs
        fiber["yb_absorption_wavelength_nm"] = pump_abs_nm
    return dict(source={"authors": label, "year": ""}, fiber=fiber,
                pump=dict(wavelength_nm=pump_nm, geometry=geometry, launched_power_W=P,
                          direction=direction),
                signal=dict(wavelength_nm=sig_nm, input_W=p_in),
                measured=measured)


def anchors():
    out = []
    # (1) Exail IXF-2CF-EY-O-6-130-LNF-L2, 915 nm backward, Pin 0.5 W, 6 m; datasheet PCE > 0.40
    for P in (5.0, 10.0, 20.0):
        out.append(_rec("Exail L2 915bwd %gW" % P, core_um=6.0, na=0.20, L=6.0, clad_um=125.0,
                        pump_nm=915.0, P=P, direction="counter", geometry="cladding",
                        er_abs=40.0, er_nm=1536.0, pump_abs=1.05, pump_abs_nm=915.0,
                        pump_abs_clad=True, sig_nm=1550.0, p_in=0.5,
                        measured={"pce": 0.40, "kind": "datasheet PCE (> 0.40)"}))
    # (2) Exail AG-EY-O-5-105-125, 976 nm forward, Pin 10 mW, 4.7 m (datasheet curve)
    for P in (1.0, 2.0, 4.0):
        out.append(_rec("Exail AG-EY-O-5 976fwd %gW" % P, core_um=5.0, na=0.19, L=4.7,
                        clad_um=105.0, pump_nm=976.0, P=P, direction="co", geometry="cladding",
                        er_abs=75.0, er_nm=1536.0, pump_abs=3.7, pump_abs_nm=976.0,
                        pump_abs_clad=True, sig_nm=1550.0, p_in=0.010,
                        measured={"pce": 0.37, "kind": "datasheet curve (~0.37)"}))
    # (3) Bai 2015, Nufern PM-EYDF-12/130, two stages
    out.append(_rec("Bai2015 stage1 12/130", core_um=12.0, na=0.20, L=3.0, clad_um=130.0,
                    pump_nm=976.0, P=7.5, direction="co", geometry="cladding",
                    er_abs=30.0, er_nm=1535.0, pump_abs=5.4, pump_abs_nm=976.0,
                    pump_abs_clad=True, sig_nm=1550.0, p_in=0.030,
                    measured={"p_out_W": 2.6, "gain_dB": 19.4, "kind": "measured"}))
    out.append(_rec("Bai2015 stage2 12/130", core_um=12.0, na=0.20, L=2.0, clad_um=130.0,
                    pump_nm=976.0, P=25.0, direction="co", geometry="cladding",
                    er_abs=30.0, er_nm=1535.0, pump_abs=5.4, pump_abs_nm=976.0,
                    pump_abs_clad=True, sig_nm=1550.0, p_in=2.6,
                    measured={"p_out_W": 10.0, "kind": "measured"}))
    # (4) Wei 2020, CorActive DCF-EY-10/128-PM
    out.append(_rec("Wei2020 10/128", core_um=10.0, na=0.20, L=3.8, clad_um=128.0,
                    pump_nm=976.0, P=21.0, direction="co", geometry="cladding",
                    er_abs=32.0, er_nm=1535.0, pump_abs=4.8, pump_abs_nm=976.0,
                    pump_abs_clad=True, sig_nm=1550.0, p_in=2.1,
                    measured={"p_out_W": 8.15, "kind": "measured"}))
    # (5) Hannover (de Varona Ortega 2019 Ch. 6), CORE-pumped at 1018 nm, seed 621 mW, 2.5 m
    for part, core_um, na, er_abs, pump_abs, clad_um in (
            ("SM-EYDF-6/125-HE", 6.0, 0.18, 45.0, 0.75, 125.0),
            ("DCF-EY-6/128", 5.6, 0.20, 50.0, 0.90, 128.0)):
        for P in (5.0, 20.0):
            out.append(_rec("Hannover %s %gW" % (part, P), core_um=core_um, na=na, L=2.5,
                            clad_um=clad_um, pump_nm=1018.0, P=P, direction="co",
                            geometry="core", er_abs=er_abs, er_nm=1535.0, pump_abs=pump_abs,
                            pump_abs_nm=915.0, pump_abs_clad=True, sig_nm=1556.0, p_in=0.621,
                            measured={"kind": "roll-off with a steep 1-um rise; "
                                              "~48% slope then clamp"}))
    # (6) Morasse 2006 cut-back, CorActive Hpa-Ey-10-01 -- every parameter measured
    for L, p_out in ((1.3, 0.59), (2.75, 0.67), (4.2, 0.56), (5.75, 0.46), (7.8, 0.215),
                     (9.7, 0.115)):
        out.append(dict(source={"authors": "Morasse2006 cutback %.2f m" % L, "year": ""},
                        fiber=dict(core_diameter_um=10.0, na=0.186, length_m=L,
                                   clad_diameter_um=131.2, n_er_m3=4.7e25, n_yb_m3=8.5e26,
                                   background_loss_dB_per_km=595.0),
                        pump=dict(wavelength_nm=915.0, geometry="cladding",
                                  launched_power_W=4.31, direction="co"),
                        signal=dict(wavelength_nm=1556.0, input_W=7.3e-3),
                        measured={"p_out_W": p_out, "kind": "measured cut-back"}))
    return out


def build(rec, mod, er_ion, yb_ion, **kw):
    """Build an ErYbAmplifier from one record, reusing run_anchors' parsing and density
    back-computation. Returns (amp, note) or (None, reason)."""
    g = mod.g
    fib = dict(rec.get("fiber") or {})
    prod = mod.lookup_product(fib.get("part"))
    if prod:
        for key, pkey in (("na", "core_na"), ("core_diameter_um", "core_diameter_um"),
                          ("clad_diameter_um", "inner_clad_um")):
            if fib.get(key) is None:
                fib[key] = prod.get(pkey)
    pump, sig = rec.get("pump") or {}, rec.get("signal") or {}
    a = g(fib, "core_radius_um") or 0.5 * g(fib, "core_diameter_um", default=0.0)
    na, L = g(fib, "na"), g(fib, "length_m")
    lam_p, P = g(pump, "wavelength_nm", default=976.0), g(pump, "launched_power_W")
    clad = str(g(pump, "geometry", default="core")).lower().startswith("clad")
    clad_r = 0.5e-6 * g(fib, "clad_diameter_um", "cladding_diameter_um", default=125.0)
    lam_s, p_in = g(sig, "wavelength_nm", default=1550.0), g(sig, "input_W")
    if None in (a, na, L, P, p_in) or a <= 0.0:
        return None, "not runnable"
    a_m = a * 1e-6
    note = []
    n_er, n_yb = g(fib, "n_er_m3"), g(fib, "n_yb_m3")
    if n_er is None:
        er_abs = g(fib, "er_absorption_dB_per_m")
        er_lam = g(fib, "er_absorption_wavelength_nm", default=1535.0)
        if er_abs is None:
            return None, "no Er density or absorption"
        n_er = mod.density_from_absorption(er_abs, er_lam,
                                           float(er_ion.sigma_a.sigma(er_lam * 1e-9)), a_m, na)
        note.append("N_Er from %.0f dB/m @%.0f" % (er_abs, er_lam))
    if n_yb is None:
        yb_abs = g(fib, "yb_or_pump_absorption_dB_per_m", "clad_pump_absorption_dB_per_m")
        yb_lam = g(fib, "yb_absorption_wavelength_nm", "clad_pump_absorption_wavelength_nm",
                   default=976.0)
        yb_clad = g(fib, "clad_pump_absorption_dB_per_m") is not None
        if yb_abs is None:
            return None, "no Yb density or pump absorption"
        s_yb = float(yb_ion.sigma_a.sigma(yb_lam * 1e-9))
        n_yb = mod.density_from_absorption(yb_abs, yb_lam, s_yb, a_m, na, clad=yb_clad,
                                           clad_r=clad_r)
        if not yb_clad:
            n_yb -= n_er * float(er_ion.sigma_a.sigma(yb_lam * 1e-9)) / s_yb
        note.append("N_Yb from %.2f dB/m @%.0f (%s)" % (yb_abs, yb_lam,
                                                        "clad" if yb_clad else "core"))
    spec = FiberSpec(a_m, na, n_er, L, clad_radius_m=(clad_r if clad else None),
                     background_loss_per_m=g(fib, "background_loss_dB_per_km", default=30.0)
                     * 1e-3 * math.log(10.0) / 10.0)
    direction = str(g(pump, "direction", default="co")).lower()
    d = "bwd" if direction.startswith("counter") else "fwd"
    pumps = [Pump(P, lam_p * 1e-9, d, cladding=clad)]
    amp = ErYbAmplifier(er_ion, yb_ion, spec, pumps, [Signal(p_in, lam_s * 1e-9)],
                        AseBand(1520e-9, 1570e-9, 24), yb_ase=AseBand(1000e-9, 1100e-9, 12),
                        k_back_m3_s=0.0, a32_per_s=3e5, upconversion_C_up=C_UP,
                        n_yb_m3=n_yb, **kw)
    return amp, "; ".join(note)


def run(amp, p_in, p_pump):
    r = amp.solve(n_nodes=NODES, relax=0.5)
    i_s = [i for i, k in enumerate(r.kind) if k == "signal"][0]
    yb_bins = [i for i, k in enumerate(r.kind) if k == "ase" and r.lambda_m[i] < 1.2e-6]
    p_yb = float(sum(r.power_W[i, -1 if r.u[i] > 0 else 0] for i in yb_bins))
    p_out = float(r.power_W[i_s, -1])
    b = wall_plug_efficiency(amp, r, PumpSource(wallplug_efficiency=1.0))
    return dict(p_out=p_out, gain_dB=float(r.signal_gain_dB[0]),
                pce=(p_out - p_in) / p_pump, eta_tr=float(r.meta["eta_transfer"]),
                p_yb=p_yb, yb_frac=p_yb / max(p_out + p_yb, 1e-30),
                par_dB=float(r.meta["yb_parasitic_gain_dB"]),
                conv=bool(r.meta["converged"]),
                resid=float(b.energy_balance_residual_W) / p_pump)


def main():
    mod = _load_loader()
    if mod is None:
        print("anchor folder not found: %s" % ANCHOR_DIR)
        print("pass --anchors DIR or set DYNAMETA_ERYB_ANCHORS; this script only REPORTS, so")
        print("its absence is not a failure.")
        return 0
    er_legacy, yb_legacy = erbium(), ytterbium("phosphosilicate")
    er_phs, yb_phs = erbium("phosphosilicate"), ytterbium_melkumov("phosphosilicate")
    cols = (
        ("A single k=1e-21", dict(k_tr_m3_s=K_SINGLE, yb_coupled_fraction=1.0, k_tr2_m3_s=0.0),
         er_legacy, yb_legacy),
        ("B two-pop f=0.9", dict(k_tr_m3_s=K_FAST, yb_coupled_fraction=F_COUPLED,
                                 k_tr2_m3_s=K2), er_legacy, yb_legacy),
        ("C two-pop + PhS", dict(k_tr_m3_s=K_FAST, yb_coupled_fraction=F_COUPLED,
                                 k_tr2_m3_s=K2), er_phs, yb_phs),
        ("D two-pop f*k=A", dict(k_tr_m3_s=K_SINGLE / F_COUPLED,
                                 yb_coupled_fraction=F_COUPLED, k_tr2_m3_s=K2),
         er_legacy, yb_legacy),
    )
    print("anchors: %s | nodes %d" % (ANCHOR_DIR, NODES))
    print("A = single population k_tr %.2g | B = f %.2g, k_fast %.2g, K2 %.2g, same ions"
          " | C = B with the measured phosphosilicate spectroscopy"
          " | D = f %.2g at the SAME effective f*k_tr as A (k_tr %.3g), the control"
          % (K_SINGLE, F_COUPLED, K_FAST, K2, F_COUPLED, K_SINGLE / F_COUPLED))
    print("%-34s %-26s %s" % ("anchor", "measured", "  ".join("%-34s" % c[0] for c in cols)))
    ratios = {c[0]: [] for c in cols}
    for rec in anchors():
        label = rec["source"]["authors"]
        meas = rec["measured"]
        mtxt = (("out %.3g W" % meas["p_out_W"]) if "p_out_W" in meas else
                ("PCE %.2f" % meas["pce"]) if "pce" in meas else meas["kind"][:24])
        cells = []
        for _name, kw, er_ion, yb_ion in cols:
            amp, note = build(rec, mod, er_ion, yb_ion, **kw)
            if amp is None:
                cells.append("%-34s" % note)
                continue
            try:
                o = run(amp, rec["signal"]["input_W"], rec["pump"]["launched_power_W"])
            except Exception as exc:                                       # noqa: BLE001
                cells.append("%-34s" % ("ERROR " + type(exc).__name__))
                continue
            if "p_out_W" in meas and meas["p_out_W"] > 0.0:
                ratios[_name].append(o["p_out"] / meas["p_out_W"])
            cells.append("%-34s" % ("out %6.3f W PCE %.3f 1um %5.1f%%%s"
                                    % (o["p_out"], o["pce"], 100.0 * o["yb_frac"],
                                       "" if o["conv"] else "!")))
        print("%-34s %-26s %s" % (label[:34], mtxt, "  ".join(cells)))
    print()
    print("predicted/measured output power, over the anchors that state one (n = %d):"
          % len(ratios[cols[0][0]]))
    for name, _kw, _e, _y in cols:
        v = np.asarray(ratios[name], float)
        if v.size:
            print("  %-18s mean %.3f  median %.3f  worst %.3f  (a value of 1.0 is exact)"
                  % (name, float(np.mean(v)), float(np.median(v)), float(np.max(v))))

    # ---- the FORC ETE claim: the 1-um ASE fraction must become ETE-dependent ----------------
    print()
    print("FORC ETE claim: 1-um ASE fraction vs the coupled fraction at FIXED f*k_fast")
    print("(measured: ETE 0.2-0.4 -> 21%-0.5% of the output at 1 um; ETE 0.9 -> 0.05%)")
    base = anchors()[4]                      # the Exail AG-EY-O-5 976 nm point, 2 W
    print("%-8s %-10s %-12s %-12s %-10s %-10s" % ("f", "eta_tr", "out (W)", "1-um (W)",
                                                  "1-um frac", "par (dB)"))
    for f in (0.2, 0.3, 0.4, 0.6, 0.9, 1.0):
        amp, _n = build(base, mod, er_legacy, yb_legacy, k_tr_m3_s=(F_COUPLED * K_FAST) / f,
                        yb_coupled_fraction=f, k_tr2_m3_s=K2)
        o = run(amp, base["signal"]["input_W"], base["pump"]["launched_power_W"])
        print("%-8.2f %-10.3f %-12.4g %-12.4g %-10.3f %-10.1f"
              % (f, o["eta_tr"], o["p_out"], o["p_yb"], 100.0 * o["yb_frac"], o["par_dB"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
