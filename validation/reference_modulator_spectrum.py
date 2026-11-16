"""Diagnostic (no gate): tabulate the reflection spectrum of the generic gated-film metasurface cell
(validation/_reference_device.py) at a positive and a negative gate bias near its gap-mode resonance,
and print |dR| between them -- the optical end of the full carrier -> Drude -> 3-D FEM pipeline.
Run: python -m validation.reference_modulator_spectrum
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation._reference_device import build_reference_modulator
from dynameta.sweep import Sweep, BiasPoint
from dynameta.pipeline import run_pipeline

design = build_reference_modulator()           # equilibrium film; the oxides' eps_static_dc set the gate C
sweep = Sweep(
    bias_points=[BiasPoint({"top_contact": +3.0}, "patch+3V"),
                  BiasPoint({"top_contact": -3.0}, "patch-3V")],
    wavelengths_nm=[1150.0, 1200.0, 1250.0])
rows = run_pipeline(design, sweep, verbose=True)
print("[t] --- reference-cell spectrum ---", flush=True)
by_wl = {}
for r in rows:
    res = r.result
    print("[t] {:9s} lam={:.0f}nm  R={:.4f}".format(r.bias_label, r.lambda_nm, res.R), flush=True)
    by_wl.setdefault(r.lambda_nm, {})[r.bias_label] = res.R
for wl in sorted(by_wl):
    d = by_wl[wl]
    if "patch+3V" in d and "patch-3V" in d:
        print("[t] lam={:.0f}nm  |dR(+3V vs -3V)| = {:.4f}".format(
            wl, abs(d["patch+3V"] - d["patch-3V"])), flush=True)
print("[t] *** REFERENCE-CELL SPECTRUM TABULATION DONE ***", flush=True)
