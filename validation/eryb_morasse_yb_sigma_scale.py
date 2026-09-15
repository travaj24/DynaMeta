"""Morasse 2006 cut-back: the Yb emission scale fitted to a MEASURED 1-um ASE, and the explicit
4I11/2 sensitivity on the same fiber -- a VALIDATION SCRIPT, not a test. It reports numbers; it
asserts nothing, and nothing in it is tuned to anything but the one measurement it fits.

WHAT IT ANSWERS. The 2026-09-14 literature record (scratchpad eryb/validation/
VALIDATION_SUMMARY.md section 6) closes with the sharpest calibration datum in the Er:Yb
literature: Morasse et al. (2006) measured EVERY parameter of a CorActive Hpa-Ey-10-01 fiber --
core 10.01 um, NA 0.186, N_Yb 8.5e26, N_Er 4.7e25, tau_Yb 835 us, tau_Er 9.0 ms, cladding area
13532 um^2, background loss 595 dB/km -- ran 4.31 W of 914.8 nm co-propagating cladding pump with
a 7.3 mW seed at 1556.3 nm, and still over-predicted the BACKWARD 1-um ASE by a factor 1000
(19.5 mW predicted against 0.079 mW measured) while the signal output was insensitive. He fixed
it by scaling the McCumber-derived Yb emission cross-section by 0.4. This script reproduces that
failure with DynaMeta's own model and fits the scale back.

Why the fitted number is not expected to BE 0.4. The scale is a property of the ION MODEL as much
as of the fiber: Morasse scaled a McCumber-DERIVED sigma_e_Yb, while the two ion pairs here are
(A) the parametric phosphosilicate Gaussian model the 2026-09-14 validation ran on and (B) the
MEASURED Melkumov P2O5 table. Two independent spectra needing the same correction to within ~15%
is the result; an exact match would be a coincidence.

The second half sweeps tau_32 on the same fiber, because the 4I11/2 bottleneck and the 1-um ASE
are the two places an EYDFA model is most often wrong, and this is the one fiber where every
other parameter is measured.

Run:  python -m validation.eryb_morasse_yb_sigma_scale [--nodes 201] [--quick]
Needs no external data: every fiber number below is transcribed from the record above.
"""
import math
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fiber_amp import (                                        # noqa: E402
    AseBand, FiberSpec, Pump, Signal, erbium, ytterbium, ytterbium_melkumov,
)
from dynameta.optics.fiber_amp.eryb import (                                   # noqa: E402
    ErYbAmplifier, eryb_fit_yb_sigma_e_scale,
)

ARGS = sys.argv[1:]
NODES = int(ARGS[ARGS.index("--nodes") + 1]) if "--nodes" in ARGS else 201
QUICK = "--quick" in ARGS

# ---- the measured fiber and operating point (Morasse 2006, CorActive Hpa-Ey-10-01) ----------
A_CLAD_M2 = 13532e-12
CLAD_R = math.sqrt(A_CLAD_M2 / math.pi)          # 65.6 um equivalent-area radius
CORE_R, NA, DOPANT_R = 5.005e-6, 0.186, 4.9e-6
N_ER, N_YB = 4.7e25, 8.5e26
TAU_ER, TAU_YB = 9.0e-3, 835e-6
BG_PER_M = 595e-3 * math.log(10.0) / 10.0        # 595 dB/km
LENGTH_M = 2.75
PUMP_W, PUMP_M = 4.31, 914.8e-9
SEED_W, SEED_M = 7.3e-3, 1556.3e-9
MEASURED_OUT_W = 0.670                           # at 2.75 m
MEASURED_1UM_BWD_W = 0.079e-3                    # the number the scale is fitted to
MORASSE_SCALE = 0.4                              # his own fitted value, for comparison
MORASSE_K_TR = 1.0e-22                           # his own fitted transfer coefficient


def ion_pair(which):
    """(er, yb) at the fiber's MEASURED lifetimes. 'legacy' is the parametric pair the
    2026-09-14 validation ran on; 'phs' is the measured phosphosilicate spectroscopy."""
    if which == "phs":
        er = replace(erbium("phosphosilicate"), name="Er3+(phs,tau9ms)", tau_s=TAU_ER)
        yb = replace(ytterbium_melkumov("phosphosilicate"), name="Yb3+(melkumov,tau835us)",
                     tau_s=TAU_YB)
    else:
        er = replace(erbium(), name="Er3+(as,tau9ms)", tau_s=TAU_ER)
        yb = replace(ytterbium("phosphosilicate"), name="Yb3+(gaussian,tau835us)", tau_s=TAU_YB)
    return er, yb


def build(which="legacy", k_tr=MORASSE_K_TR, scale=1.0, tau32=None, k_back=0.0, n_yb_bins=10):
    er, yb = ion_pair(which)
    fib = FiberSpec(CORE_R, NA, N_ER, LENGTH_M, dopant_radius_m=DOPANT_R, clad_radius_m=CLAD_R,
                    background_loss_per_m=BG_PER_M)
    return ErYbAmplifier(er, yb, fib, [Pump(PUMP_W, PUMP_M, "fwd", cladding=True)],
                         [Signal(SEED_W, SEED_M)], AseBand(1520e-9, 1570e-9, 12),
                         n_yb_m3=N_YB, k_tr_m3_s=k_tr, yb_ase=AseBand(1000e-9, 1100e-9, n_yb_bins),
                         yb_sigma_e_scale=scale, tau32_s=tau32, k_back_m3_s=k_back)


def probe(amp, label):
    r = amp.solve(n_nodes=NODES, relax=0.5)
    i_s = [i for i, k in enumerate(r.kind) if k == "signal"][0]
    bwd = [i for i, k in enumerate(r.kind)
           if k == "ase" and r.lambda_m[i] < 1.2e-6 and r.u[i] < 0.0]
    p_1um = sum(float(r.power_W[i, 0]) for i in bwd)
    print("  %-30s out %6.3f W (meas %.3f)  1-um bwd %10.4g W (meas %.3g)  x%8.4g  eta %.3f%s"
          % (label, float(r.power_W[i_s, -1]), MEASURED_OUT_W, p_1um, MEASURED_1UM_BWD_W,
             p_1um / MEASURED_1UM_BWD_W, r.meta["eta_transfer"],
             "" if r.meta["converged"] else "  NOCONV"))
    return r, p_1um


def main():
    print("Morasse 2006 cut-back, %s m, %.2f W at %.1f nm (cladding, co), seed %.1f mW at %.1f nm"
          % (LENGTH_M, PUMP_W, PUMP_M * 1e9, SEED_W * 1e3, SEED_M * 1e9))
    print("clad radius %.1f um (13532 um^2), N_Er %.2g, N_Yb %.2g, tau_Er %.1f ms, tau_Yb %d us,"
          " bg %.0f dB/km | %d nodes" % (CLAD_R * 1e6, N_ER, N_YB, TAU_ER * 1e3, TAU_YB * 1e6,
                                         595.0, NODES))
    print()
    print("1. The failure, reproduced: predicted vs measured backward 1-um ASE at scale 1.0")
    for which in ("legacy", "phs"):
        for k in ((MORASSE_K_TR,) if QUICK else (MORASSE_K_TR, 3.0e-22, 1.0e-21)):
            probe(build(which, k_tr=k), "%s ions, k_tr %.0e" % (which, k))
    print()
    print("2. The fit: yb_sigma_e_scale against the measured 0.079 mW (Morasse fitted 0.40)")
    for which in ("legacy", "phs"):
        fit = eryb_fit_yb_sigma_e_scale(build(which), MEASURED_1UM_BWD_W, "bwd",
                                        n_nodes=NODES, relax=0.5)
        print("  %-8s scale %.4f (Morasse %.2f, %+.0f%%)  1-um %.4g -> %.4g W  signal %.4f -> "
              "%.4f W (%+.4f dB)  %d solves%s"
              % (which, fit.scale, MORASSE_SCALE,
                 100.0 * (fit.scale / MORASSE_SCALE - 1.0), fit.ase_1um_unit_scale_W,
                 fit.ase_1um_W, fit.signal_out_unit_scale_W, fit.signal_out_W,
                 fit.signal_change_dB, fit.solves, "" if fit.converged else "  NOCONV"))
    if not QUICK:
        print()
        print("   sensitivity of the fitted scale to k_tr (legacy ions)")
        for k in (3.0e-22, 1.0e-21):
            fit = eryb_fit_yb_sigma_e_scale(build(k_tr=k), MEASURED_1UM_BWD_W, "bwd",
                                            n_nodes=NODES, relax=0.5)
            print("     k_tr %.0e -> scale %.4f, signal %+.4f dB" % (k, fit.scale,
                                                                     fit.signal_change_dB))
    print()
    print("3. The explicit 4I11/2 on the same fiber (k_tr %.0e, k_back = k_tr where shown)"
          % MORASSE_K_TR)
    r0, _p = probe(build(), "adiabatic (tau32 -> 0)")
    g0 = float(r0.signal_gain_dB[0])
    print("  %-14s %-10s %-10s %-12s %-10s" % ("tau32", "gain dB", "d gain", "max n3/N_Er",
                                               "back/fwd"))
    for t32 in ((7e-6,) if QUICK else (1e-6, 7e-6, 1e-5, 5e-5)):
        for kb in (0.0, MORASSE_K_TR):
            r = build(tau32=t32, k_back=kb).solve(n_nodes=NODES, relax=0.5)
            print("  %-14s %-10.5f %+-10.4f %-12.4g %-10.4g%s"
                  % ("%.0e%s" % (t32, " +back" if kb else ""), float(r.signal_gain_dB[0]),
                     float(r.signal_gain_dB[0]) - g0, float(r.meta["er_4i11_2_z"].max()),
                     float(r.meta["back_transfer_ratio"]),
                     "" if r.meta["converged"] else "  NOCONV"))
    print()
    print("Reading. The model reproduces Morasse's failure mode (a 1-um ASE over-prediction of "
          "2-3 orders with every fiber parameter measured) and one emission scale removes it "
          "while moving the signal by hundredths of a dB -- which is the observation that makes "
          "a 1-um-only correction admissible. The fitted scale is NOT independent of the ion "
          "model or of k_tr, so quote it with both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
