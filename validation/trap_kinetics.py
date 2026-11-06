"""
KINETIC TRAPS (v0.15, plan 4.3.2 K1-K4) -- the nightly print-out of every kinetic-trap gate with its numbers.

The gates and their oracles / derived tolerances live in tests/test_trap_kinetics.py (K1, K2, the DC form,
well-posedness) and tests/test_trap_noise.py (K3); this script loads those modules (their builders and
closed forms, one copy of each) and PRINTS the measured quantities next to each tolerance, so a drift shows
as a number, not just a red test:

  K1-2  DC identity, kinetic vs quasi-static (5 fixtures)          K1-4  DLTS emission vs the TR error law
  K1-5  transient Kirchhoff and the integrated contact charge     K1-6  ssac Kirchhoff
  K2-1  N&G single-level G_p/omega per point vs its budget        K2-3  the 44-bin continuum vs N&G (18)/(20)
  K2-5  the +-0.1 mV gate step vs the Fig. 6 circuit              K2-6  2-D extrusion and Stacked3DSpec == 1-D
  K3-1  FDT, charged kinetic level, 1 Hz - 1 GHz                  K3-2  the load-bearing perturbations
  K3-4  the trapping Lorentzian of a resistor                      K3-5  MOS FDT with kinetic bins
Docs: docs/audit/2026-11-06-v0.15-kinetic-traps.md. Run: python -m validation.run_all --tier full trap_kinetics
(~3 min). Exit 1 on any failed gate, 42 without devsim.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

from validation._harness import Harness                                  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location("_tk_" + name, os.path.join(HERE, "tests", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    H = Harness("tk", "KINETIC TRAPS: occupancy dynamics, conductance method, trap g-r noise")
    try:
        import devsim  # noqa: F401
    except Exception:
        H.require(False, "devsim not installed")
    warnings.simplefilter("ignore")                       # the decision-39 advisories are expected here
    TK, TN = _load("test_trap_kinetics"), _load("test_trap_noise")
    from _pytest.monkeypatch import MonkeyPatch

    def k12():
        worst = 0.0
        for case in sorted(TK.DC_CASES):
            TK.V.fresh()
            qs, kn = TK._dc_state(case, False), TK._dc_state(case, True)
            for k in ("Electrons", "Holes", "TrapOccupancy0"):
                worst = max(worst, float(np.max(np.abs(qs[k] - kn[k]) / np.abs(qs[k]))))
        return worst < 1e-10, "max rel n/p/f difference {:.2e} (tol 1e-10)".format(worst)
    H.run("K1-2 DC identity (5 fixtures)", k12)

    def k14():
        TK.V.fresh()
        dev = TK._dl_device("vdl", (TK._dl_level(True),))
        k, tau, finf, f0 = TK._dl_oracle(dev)
        out = []
        for i, hf in enumerate((1 / 25, 1 / 50)):
            if i:
                TK._dl_bias(TK.DL_VR, 0.0)
            t, f, i1, i2, kind = TK._dl_transient(dev, k, tau, hf)
            off = (f[kind == 0][-1] - f0) / (f0 - finf)
            post = kind == 2
            ora = finf + (f0 - finf) * np.exp(-t[post] / tau)
            err = (f[post] - ora) / (ora - finf) - off
            etr = TK._tr_error(t[post] - t[kind == 1][-1], hf, tau)
            out.append((hf, off, err[-1], etr[-1], float(np.max(np.abs(i1 + i2)) / np.max(np.abs(i1)))))
        TK.V.fresh()
        ok = all(abs(e - et) <= 0.02 * abs(et) + 1e-6 for _, _, e, et, _ in out) and abs(out[0][2] / out[1][2] / 4 - 1) < 0.2
        return ok, "; ".join("h=tau/{:.0f}: off {:.2e}, err(5tau) {:.4e} vs TR {:.4e}, KCL {:.1e}".format(
            1 / hf, o, e, et, kc) for hf, o, e, et, kc in out) + "; h^2 ratio {:.3f}".format(out[0][2] / out[1][2])
    H.run("K1-4/K1-5 DLTS emission + transient Kirchhoff", k14)

    for label, fn in (("K1-5 integrated contact charge", TK.test_transient_integrated_charge_through_the_n_contact),
                      ("K1-6 ssac Kirchhoff", TK.test_ssac_kirchhoff_kinetic_vs_quasi_static),
                      ("K2-5 gate step = N&G Fig. 6", TK.test_gate_step_transient_is_the_ng_circuit),
                      ("K2-6 2-D extrusion == 1-D", TK.test_conductance_method_2d_wiring),
                      ("K2-6 Stacked3DSpec == 1-D", TK.test_conductance_method_3d_stacked_spec_equals_1d),
                      ("K2-3 continuum (N&G 18-20)", TK.test_conductance_method_continuum),
                      ("well-posedness: trap rows pinned", TK.test_kinetic_trap_rows_are_pinned_candidates_of_the_well_posedness_check),
                      ("K3-3 neutral level = GR Lorentzian", TN.test_neutral_kinetic_level_reproduces_the_ambipolar_gr_lorentzian),
                      ("K3-4 trapping Lorentzian", TN.test_trapping_noise_lorentzian_of_a_resistor),
                      ("K3-6 native adjoint on trap rows", TN.test_native_noise_field_on_trap_rows_is_the_adjoint)):
        def thunk(fn=fn):
            TK.V.fresh()
            fn()
            TK.V.fresh()
            return True, "passed (numbers in the test docstring)"
        H.run(label, thunk)

    def k21():
        TK.V.fresh()
        psib = TK._mos_ac("vmos", TK._it_single())
        TK._mos_vg(psib, 0.0, TK.MOS_VG)
        psi_s, ps, ns = TK._mos_surface("vmos")
        cs, tau, f0, mino = TK._ng_single_level(ps, ns)
        r_h = TK._hole_path_resistance("vmos")
        w, y = TK._y_sc(TK.WT / tau / (2 * math.pi))
        TK.V.fresh()
        rel = y.real / w / (cs * TK.WT / (1 + TK.WT ** 2)) - 1
        bud = np.array([TK._conductance_budget(ww, psi_s, ps, cs, tau, mino, r_h) for ww in w])
        return bool(np.all(np.abs(rel) <= 10 * bud)), "omega tau {}: rel {} / budget {}".format(
            list(TK.WT), ["{:.1e}".format(v) for v in rel], ["{:.1e}".format(v) for v in bud])
    H.run("K2-1 N&G single level", k21)

    def k31():
        TN.V.fresh()
        r, sn, _ = TN._fdt_diode((TN._charged(),))
        TN.V.fresh()
        return float(np.max(np.abs(r))) < 1e-4, "max |S/4kT Re Y - 1| = {:.2e} (1 Hz - 1 GHz; tol 1e-4)".format(
            float(np.max(np.abs(r))))
    H.run("K3-1 FDT, charged kinetic level", k31)

    def k32():
        out = []
        for how in ("drop_hole", "flip", "double"):
            mp = MonkeyPatch()
            try:
                TN.V.fresh()
                out.append((how, float(np.max(np.abs(TN._perturbed(mp, how))))))
            finally:
                mp.undo()
        return all(v > 1e-2 for _, v in out), ", ".join("{} {:.2e}".format(h, v) for h, v in out)
    H.run("K3-2 load-bearing (each must break K3-1 by > 1e-2)", k32)

    for which in ("single", "continuum"):
        def k35(which=which):
            TN.V.fresh()
            TN.test_mos_fdt_with_kinetic_interface_bins(which)
            return True, "S_I = 4kT Re Y_m to 1e-4 (omega tau 1e-2 .. 1e2)"
        H.run("K3-5 MOS FDT ({})".format(which), k35)
    return H.summary("KINETIC TRAPS")


if __name__ == "__main__":
    raise SystemExit(main())
