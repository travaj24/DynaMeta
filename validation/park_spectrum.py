"""Tier-2 confirmation: re-tabulate the Park reflection spectrum with the CORRECTED
gate-oxide DC permittivity (HfO2 18 / Al2O3 9), which raised the +2V accumulation
1.09->1.35. Compare +2V vs -2V near the cavity resonance to confirm the stronger
accumulation deepens the modulation (the eps_static_dc fix is already Stage-1
validated; this is the optical confirmation). Run: python -m validation.park_spectrum
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.park_2021 import build_park_design
from dynameta.sweep import Sweep, BiasPoint
from dynameta.pipeline import run_pipeline

design = build_park_design()           # equilibrium ITO, eps_static_dc=18/9 in effect
sweep = Sweep(
    bias_points=[BiasPoint({"top_contact": +2.0}, "patch+2V"),
                  BiasPoint({"top_contact": -2.0}, "patch-2V")],
    wavelengths_nm=[1200.0, 1250.0, 1300.0])
rows = run_pipeline(design, sweep, verbose=True)
print("[t] --- Park spectrum (corrected oxide DC eps 18/9) ---", flush=True)
by_wl = {}
for r in rows:
    res = r.result
    print("[t] {:9s} lam={:.0f}nm  R={:.4f}".format(r.bias_label, r.lambda_nm, res.R), flush=True)
    by_wl.setdefault(r.lambda_nm, {})[r.bias_label] = res.R
for wl in sorted(by_wl):
    d = by_wl[wl]
    if "patch+2V" in d and "patch-2V" in d:
        print("[t] lam={:.0f}nm  |dR(+2V vs -2V)| = {:.4f}".format(
            wl, abs(d["patch+2V"] - d["patch-2V"])), flush=True)
print("[t] *** PARK SPECTRUM RE-TABULATION DONE ***", flush=True)
