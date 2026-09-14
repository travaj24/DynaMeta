"""Er:Yb MIGRATION CALIBRATION, TEMPERATURE LAWS and PER-FIBER (f, k_tr) FITS -- a VALIDATION
SCRIPT, not a test: it reports numbers, it does not assert them.

WHAT IT ANSWERS, in four sections.

  1. THE MIGRATION RATE. `eryb_fit.yb_two_pool_from_decay` inverts Cheng et al. 2022's printed
     bi-exponential ytterbium decay plus its Er-free control EXACTLY to (f, k_tr, W_mig), and
     `eryb_calibrate_pools` combines that with Jeong 2007's stated decay pattern and Laroche
     2006's five quenched lifetimes. The 2026-09-14 note's limit 1 was that W_mig had no source;
     this section is the source.

  2. THE FORC ETE RESIDUALS, before and after. The two-population build reproduced the ORDERING
     of the FORC 1-um ASE fractions against the energy-transfer efficiency but over-predicted the
     low-ETE end by ~3x and under-predicted the high-ETE end by ~10x. This re-runs that sweep
     with the calibrated W_mig, and with W_mig pushed far above it, and reports what changes.

  3. PER-FIBER (f, k_tr). `eryb_fit_to_device` fitted to each of the four FORC fibers and to Bai
     2015 stage 1, with W_mig held at the calibrated value -- the per-fiber-class calibration,
     with the covariance that says which of the two parameters the data actually pinned.

  4. THE TEMPERATURE LAWS. What `RateTemperatureLaw` and `YbStarkThermal` do to a hot fiber: the
     uniform-profile identity, the measured Cheng/Canat anchors the constants reproduce, and the
     1-um parasitic threshold against temperature.

SELF-CONTAINED: every fiber number is transcribed into this file from the 2026-09-14 anchor
collection and the vendor datasheets, so this runs with no scratchpad folder.

Run:  python -m validation.eryb_migration_thermal_fit [--nodes 161] [--max-nfev 30]
                                                      [--sections 1,2,3,4]
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fiber_amp import (                                        # noqa: E402
    AseBand, ErYbAmplifier, FiberSpec, Pump, Signal, erbium, overlap_gamma, ytterbium,
)
from dynameta.optics.fiber_amp.eryb import (                                   # noqa: E402
    RATE_ARRHENIUS_30PCT_300_480K, RATE_ARRHENIUS_CHENG_2022, RateTemperatureLaw,
    YB_STARK_976_CANAT_DUSSARDIER,
)
from dynameta.optics.fiber_amp.eryb_fit import (                               # noqa: E402
    CHENG_2022_DECAY, JEONG_2007_DECAY, LAROCHE_2006_LIFETIMES,
    W_MIG_PHOSPHOSILICATE_CEILING_PER_S, W_MIG_PHOSPHOSILICATE_PER_S,
    W_MIG_PHOSPHOSILICATE_RANGE_PER_S, DeviceTarget, device_observables, eryb_calibrate_pools,
    eryb_fit_to_device, yb_two_pool_decay, yb_two_pool_from_decay,
)

ARGS = sys.argv[1:]


def _arg(flag, default, cast=int):
    return cast(ARGS[ARGS.index(flag) + 1]) if flag in ARGS else default


NODES = _arg("--nodes", 161)
MAX_NFEV = _arg("--max-nfev", 30)
SECTIONS = set(_arg("--sections", "1,2,3,4", str).split(","))
C_UP = 1.1e-24                    # Hwang 2000 bulk phosphate, the conservative end
CATIONS_PER_M3 = 2.2e28           # silica cation density; the anchor folder's own convention


# ---------------------------------------------------------------------------------------------
def density_from_absorption(alpha_dB_per_m, lam_nm, sigma_a_m2, a_m, na, clad=False,
                            clad_r=None):
    """n = alpha[1/m] / (Gamma sigma_a). A CORE absorption uses the Marcuse overlap; a CLADDING
    absorption uses Gamma = A_core/A_clad. Transcribed from the 2026-09-14 anchor folder's
    run_anchors.py so the two scripts back-compute densities identically."""
    alpha = alpha_dB_per_m * math.log(10) / 10.0
    gam = ((a_m / clad_r) ** 2 if clad
           else float(overlap_gamma(FiberSpec(a_m, na, 1e25, 1.0), lam_nm * 1e-9)))
    return alpha / (gam * sigma_a_m2)


def build_amp(*, core_um, na, length_m, clad_um, n_er_m3, n_yb_m3, pump_W, pump_nm=976.0,
              pump_dir="fwd", seed_W, seed_nm=1555.0, bg_dB_per_km=50.0, f, k_tr,
              w_mig=0.0, k_tr2=2.0e-22, er_ion=None, yb_ion=None, n_er_ase=16, n_yb_ase=8,
              **eryb_kw):
    """One cladding-pumped Er:Yb amplifier at a stated (f, k_tr, W_mig). Extra keywords
    (`rate_temperature`, `yb_stark_thermal`) pass straight through to the constructor."""
    a_m = 0.5e-6 * core_um
    spec = FiberSpec(a_m, na, n_er_m3, length_m, clad_radius_m=0.5e-6 * clad_um,
                     background_loss_per_m=bg_dB_per_km * 1e-3 * math.log(10.0) / 10.0)
    return ErYbAmplifier(er_ion or erbium(), yb_ion or ytterbium("phosphosilicate"), spec,
                         [Pump(pump_W, pump_nm * 1e-9, pump_dir, cladding=True)],
                         [Signal(seed_W, seed_nm * 1e-9)], AseBand(1520e-9, 1570e-9, n_er_ase),
                         yb_ase=AseBand(1000e-9, 1100e-9, n_yb_ase), n_yb_m3=n_yb_m3,
                         k_tr_m3_s=k_tr, k_back_m3_s=0.0, a32_per_s=3e5,
                         upconversion_C_up=C_UP, yb_coupled_fraction=f, k_tr2_m3_s=k_tr2,
                         yb_migration_rate_per_s=w_mig, **eryb_kw)


# ---- the four FORC fibers and Bai 2015 stage 1 ----------------------------------------------
# Lipatov et al., Fibers 9(3), 15 (2021) and the 2026-09-14 validation record. 976 nm
# co-propagating cladding pumping, 25 W launched, 0.6 W of 1555 nm seed throughout.
#
# ROWS 3 AND 4 are printed in full: ASL-116 (fluorophosphosilicate, 20/125, NA 0.065, 4.5 m,
# Er 0.04 at%, Yb 0.11 at%, ETE 0.38, 0.5% of the output at 1030 nm, PCE 0.19) and Nufern
# LMA-EYDF-25P/300-HE (25/300, NA 0.090, 1.5 m, Er 0.05 at%, Yb 0.88 at%, ETE 0.90, 0.05%,
# PCE 0.262).
#
# ROWS 1 AND 2 (ETE 0.20 -> 21% and ETE 0.25 -> 2%) are the same 20/125 NA 0.065 geometry at 8 m
# and 5 m with a PCE of 0.194. Their compositions are NOT printed, so two STATED ASSUMPTIONS
# stand in: they carry ASL-116's erbium, and their ytterbium is scaled as 1/L_optimal from
# ASL-116's (a cladding-pumped amplifier's optimum sits near a fixed total pump absorption
# alpha_clad L, so the optimal length is inversely proportional to N_Yb at fixed geometry). That
# makes the Yb loading 0.062 / 0.099 / 0.11 at% across the three, i.e. the ETE RISING as the
# ytterbium falls -- which is Melkumov's own reading of what sets the ETE, and is a consequence
# of the assumption rather than an input to it.
#
# DENSITY CONVENTION for the FORC rows: at% of cations x 2.2e28 m^-3, the anchor folder's own
# silica value. Against the Nufern datasheet (85 dB/m core at 1535 nm, 4.4 dB/m cladding at
# 976 nm as Bai 2015 measured on the PM version of the same fiber) this convention runs about 3x
# LOW in N_Er and 1.6x HIGH in N_Yb, so the fitted k_tr carries that absolute scale uncertainty
# while the fitted RATE k_tr N_Er does not. Both are reported.
def _at_pct(x):
    return 1e-2 * x * CATIONS_PER_M3


FORC_FIBERS = [
    dict(label="FORC ETE 0.20 (20/125, 8.0 m)", core_um=20.0, na=0.065, clad_um=125.0,
         length_m=8.0, n_er_m3=_at_pct(0.04), n_yb_m3=_at_pct(0.11 * 4.5 / 8.0),
         ete=0.20, one_um=0.21, pce=0.194, assumed=True),
    dict(label="FORC ETE 0.25 (20/125, 5.0 m)", core_um=20.0, na=0.065, clad_um=125.0,
         length_m=5.0, n_er_m3=_at_pct(0.04), n_yb_m3=_at_pct(0.11 * 4.5 / 5.0),
         ete=0.25, one_um=0.02, pce=0.194, assumed=True),
    dict(label="FORC ASL-116 ETE 0.38 (20/125, 4.5 m)", core_um=20.0, na=0.065, clad_um=125.0,
         length_m=4.5, n_er_m3=_at_pct(0.04), n_yb_m3=_at_pct(0.11),
         ete=0.38, one_um=0.005, pce=0.192, assumed=False),
    dict(label="Nufern LMA-EYDF-25P/300-HE ETE 0.90 (1.5 m)", core_um=25.0, na=0.090,
         clad_um=300.0, length_m=1.5, n_er_m3=_at_pct(0.05), n_yb_m3=_at_pct(0.88),
         ete=0.90, one_um=0.0005, pce=0.262, assumed=False),
]
FORC_PUMP_W, FORC_SEED_W = 25.0, 0.6


def forc_amp(rec, f, k_tr, w_mig, **kw):
    return build_amp(core_um=rec["core_um"], na=rec["na"], length_m=rec["length_m"],
                     clad_um=rec["clad_um"], n_er_m3=rec["n_er_m3"], n_yb_m3=rec["n_yb_m3"],
                     pump_W=FORC_PUMP_W, seed_W=FORC_SEED_W, f=f, k_tr=k_tr, w_mig=w_mig, **kw)


def bai_amp(f, k_tr, w_mig, **kw):
    """Bai et al. 2015 stage 1: Nufern PM-EYDF-12/130, 3.0 m, 7.5 W of 976 nm co-propagating,
    30 mW of 1550 nm seed -> 2.6 W out at 19.4 dB with 'no obvious 1-um ASE'. Densities
    back-computed from the catalogue absorptions (30 dB/m core at 1535 nm, 5.4 dB/m cladding at
    976 nm), the convention the 2026-09-14 anchor run uses."""
    er, yb = erbium(), ytterbium("phosphosilicate")
    a_m = 6.0e-6
    n_er = density_from_absorption(30.0, 1535.0, float(er.sigma_a.sigma(1535e-9)), a_m, 0.20)
    n_yb = density_from_absorption(5.4, 976.0, float(yb.sigma_a.sigma(976e-9)), a_m, 0.20,
                                   clad=True, clad_r=65e-6)
    return build_amp(core_um=12.0, na=0.20, length_m=3.0, clad_um=130.0, n_er_m3=n_er,
                     n_yb_m3=n_yb, pump_W=7.5, seed_W=0.030, seed_nm=1550.0, f=f, k_tr=k_tr,
                     w_mig=w_mig, **kw)


# =============================================================================================
def section1():
    print("=" * 100)
    print("1. THE MIGRATION RATE: an exact inversion, then the joint fit")
    print("=" * 100)
    a = CHENG_2022_DECAY
    inv = yb_two_pool_from_decay(tau_fast_s=a.tau_fast_s, tau_slow_s=a.tau_slow_s,
                                 amp_fast=a.amp_fast, tau_yb_s=a.tau_yb_s, n_er_m3=a.n_er_m3)
    print("Cheng 2022 (tau_1 %.2f us, tau_2 %.2f us, A1 %.3f, Er-free control %.2f us,"
          " N_Er %.3g m^-3)" % (a.tau_fast_s * 1e6, a.tau_slow_s * 1e6, a.amp_fast,
                                a.tau_yb_s * 1e6, a.n_er_m3))
    print("  EXACT inversion:  f = %.4f   k_tr = %.4g m^3/s   W_mig = %.1f 1/s"
          % (inv["coupled_fraction"], inv["k_tr_m3_s"], inv["migration_per_s"]))
    m = yb_two_pool_decay(tau_yb_s=a.tau_yb_s, n_er_m3=a.n_er_m3, k_tr_m3_s=inv["k_tr_m3_s"],
                          coupled_fraction=inv["coupled_fraction"],
                          migration_per_s=inv["migration_per_s"])
    eta = m.transfer_yield
    k_ens = eta / (1.0 - eta) / (a.n_er_m3 * a.tau_yb_s)
    print("  round trip:       tau_1 %.3f us  tau_2 %.2f us  A1 %.4f  yield %.5f"
          % (m.tau_fast_s * 1e6, m.tau_slow_s * 1e6, m.amp_fast, eta))
    print("  THE RECONCILIATION: this ONE parameter set reproduces both coefficients the anchor")
    print("  collection extracts from this one measurement -- the fast-pair %.3g m^3/s AND the"
          % inv["k_tr_m3_s"])
    print("  ensemble-yield %.3g m^3/s (the model's own yield %.4f inverted through the one-pool"
          % (k_ens, eta))
    print("  weak-signal formula). That factor of %.0f is what W_mig was introduced to carry."
          % (inv["k_tr_m3_s"] / k_ens))
    print()
    print("  sensitivity (+/- 5% on each measured input, one at a time):")
    for key in ("tau_fast_s", "tau_slow_s", "amp_fast", "tau_yb_s"):
        row = []
        for rel in (-0.05, 0.05):
            kw = dict(tau_fast_s=a.tau_fast_s, tau_slow_s=a.tau_slow_s, amp_fast=a.amp_fast,
                      tau_yb_s=a.tau_yb_s, n_er_m3=a.n_er_m3)
            kw[key] = kw[key] * (1.0 + rel)
            row.append(yb_two_pool_from_decay(**kw)["migration_per_s"])
        print("    %-12s -> W_mig %6.1f / %6.1f 1/s" % (key, row[0], row[1]))
    print("  RECOMMENDED W_mig = %.3g 1/s, range %.3g - %.3g 1/s (measurement), ceiling %.2g 1/s"
          % (W_MIG_PHOSPHOSILICATE_PER_S, *W_MIG_PHOSPHOSILICATE_RANGE_PER_S,
             W_MIG_PHOSPHOSILICATE_CEILING_PER_S))
    print()

    legs = (("Cheng 2022 only", dict(decays=[CHENG_2022_DECAY])),
            ("Jeong 2007 only", dict(decays=[JEONG_2007_DECAY])),
            ("both decays", dict(decays=[CHENG_2022_DECAY, JEONG_2007_DECAY])),
            ("Laroche 2006 only", dict(lifetimes=list(LAROCHE_2006_LIFETIMES))),
            ("decays + Laroche", dict(decays=[CHENG_2022_DECAY, JEONG_2007_DECAY],
                                      lifetimes=list(LAROCHE_2006_LIFETIMES))))
    print("%-20s %8s %12s %12s %12s %10s  %s" % ("fit leg", "f", "k_tr", "W_mig", "f*k_tr",
                                                 "cost", "W profile interval (x2 cost)"))
    cals = {}
    for name, kw in legs:
        c = eryb_calibrate_pools(w_profile_points=25, **kw)
        cals[name] = c
        print("%-20s %8.4f %12.4g %12.4g %12.4g %10.4g  [%.3g, %.3g]"
              % (name, c.coupled_fraction, c.k_tr_m3_s, c.migration_per_s,
                 c.effective_f_k_m3_s, c.cost, *c.w_interval_per_s))
    print()
    print("Laroche 2006, five phosphosilicate fibers, at the Laroche-only fit:")
    c = cals["Laroche 2006 only"]
    print("  %-38s %10s %10s %8s" % ("fiber", "model", "measured", "ratio"))
    worst = 0.0
    for k, v in c.lifetime_report.items():
        worst = max(worst, abs(v["ratio"] - 1.0))
        print("  %-38s %8.1f us %8.1f us %8.3f" % (k[:38], v["tau_model_s"] * 1e6,
                                                   v["tau_measured_s"] * 1e6, v["ratio"]))
    print("  worst deviation %.1f%%" % (100.0 * worst))
    print()
    print("  What each Laroche fiber implies ON ITS OWN, at the calibrated W_mig and Cheng's")
    print("  fast-pair k_tr -- the coupled fraction that reproduces its measured lifetime:")
    k_fast = inv["k_tr_m3_s"]
    for anc in LAROCHE_2006_LIFETIMES:
        lo, hi, f_imp = 1e-4, 0.999, float("nan")
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            tau = yb_two_pool_decay(tau_yb_s=anc.tau_yb_s, n_er_m3=anc.n_er_m3,
                                    k_tr_m3_s=k_fast, coupled_fraction=mid,
                                    migration_per_s=W_MIG_PHOSPHOSILICATE_PER_S).mean_lifetime_s
            if tau > anc.tau_measured_s:
                lo = mid
            else:
                hi = mid
            f_imp = mid
        print("    %-38s N_Er %.3g  Yb:Er %4.0f  ->  f = %.3f"
              % (anc.label[:38], anc.n_er_m3, anc.n_yb_m3 / anc.n_er_m3, f_imp))
    print()
    print("READING. The two decay anchors and the five lifetimes are NOT describing the same")
    print("regime, and the profile says so rather than averaging over it. The decays pin W_mig")
    print("near 2e2 1/s and fail above ~1e5. Laroche's five lifetimes are reproduced within 25%")
    print("ONLY in the fully MIXED limit (W_mig >> k_tr N_Er ~ 1e5 1/s), where the two pools")
    print("share one excitation and the model collapses to a single population at f k_tr = 7.5e-23")
    print("-- exactly the 6.4-8.6e-23 Laroche themselves report. So that dataset constrains the")
    print("PRODUCT f k_tr and says nothing about W_mig; the decays constrain W_mig. Reporting a")
    print("single joint fit over both (the last row above) would hide that.")
    print()


# =============================================================================================
def section2():
    print("=" * 100)
    print("2. THE FORC ETE CLAIM: the 1-um ASE fraction against the coupled fraction")
    print("=" * 100)
    print("Exail AG-EY-O-5-105-125 at 2 W of 976 nm (the 2026-09-14 note's own section 4.2")
    print("fixture), f*k_fast held at 2.7e-21 m^3/s while f is swept -- so only the SPLIT moves.")
    er, yb = erbium(), ytterbium("phosphosilicate")
    a_m = 2.5e-6
    n_er = density_from_absorption(75.0, 1536.0, float(er.sigma_a.sigma(1536e-9)), a_m, 0.19)
    n_yb = density_from_absorption(3.7, 976.0, float(yb.sigma_a.sigma(976e-9)), a_m, 0.19,
                                   clad=True, clad_r=52.5e-6)
    fk = 0.9 * 3.0e-21
    ws = (0.0, W_MIG_PHOSPHOSILICATE_PER_S, W_MIG_PHOSPHOSILICATE_CEILING_PER_S, 1.0e6)
    print()
    print("%-6s %s" % ("f", "  ".join("%-26s" % ("W_mig = %.3g 1/s" % w) for w in ws)))
    print("%-6s %s" % ("", "  ".join("%-26s" % "eta_tr  sig(W)   1-um frac" for _w in ws)))
    table = {}
    for f in (0.20, 0.25, 0.30, 0.38, 0.40, 0.60, 0.90, 1.00):
        cells = []
        for w in ws:
            amp = build_amp(core_um=5.0, na=0.19, length_m=4.7, clad_um=105.0, n_er_m3=n_er,
                            n_yb_m3=n_yb, pump_W=2.0, seed_W=0.010, seed_nm=1550.0,
                            f=f, k_tr=fk / f, w_mig=w, n_er_ase=24, n_yb_ase=12)
            o = device_observables(amp, n_nodes=NODES, relax=0.5)
            table[(f, w)] = o
            cells.append("%-26s" % ("%6.3f %7.4f %11.4g"
                                    % (o["eta_transfer"], o["p_out_W"], o["one_um_fraction"])))
        print("%-6.2f %s" % (f, "  ".join(cells)))
    print()
    print("RESIDUALS against the four measured FORC (ETE, 1-um fraction) pairs, reading the")
    print("model at f = ETE (the two-population model's coupled fraction IS the quantity FORC")
    print("measure as the energy-transfer efficiency):")
    print("%-8s %-12s %s" % ("ETE", "measured",
                             "  ".join("%-22s" % ("W = %.3g" % w) for w in ws)))
    for ete, meas in ((0.20, 0.21), (0.25, 0.02), (0.38, 0.005), (0.90, 0.0005)):
        cells = []
        for w in ws:
            o = table[(ete, w)]
            cells.append("%-22s" % ("%9.4g  (x%.3g)"
                                    % (o["one_um_fraction"], o["one_um_fraction"] / meas)))
        print("%-8.2f %-12.4g %s" % (ete, meas, "  ".join(cells)))
    print()
    print("RMS of |log10(model/measured)| over the four rows -- one number per column:")
    cells = []
    for w in ws:
        r = [abs(np.log10(table[(e, w)]["one_um_fraction"] / m))
             for e, m in ((0.20, 0.21), (0.25, 0.02), (0.38, 0.005), (0.90, 0.0005))]
        cells.append("W = %-8.3g %.3f (a typical factor of %.1f)"
                     % (w, float(np.sqrt(np.mean(np.square(r)))),
                        10.0 ** float(np.sqrt(np.mean(np.square(r))))))
    for c in cells:
        print("  " + c)
    print()
    print("READING, and it is not the one the brief expected. The CALIBRATED W_mig (2.2e2 1/s)")
    print("changes the FORC residuals by a few percent and not at all in the aggregate: the")
    print("exchange it adds is three orders below the coupled pool's own transfer rate, so it")
    print("cannot move a 20-30 dB single-pass 1-um gain. A migration rate at the TOP of what the")
    print("decay anchors admit (1e4 1/s, 45x the calibrated value and the ceiling Jeong's bands")
    print("allow) DOES help, and substantially -- the ETE 0.25 row moves from 32x to 1.4x and the")
    print("ETE 0.38 row from 87x to 0.13x -- but it over-corrects the high-ETE row and it is not")
    print("the value the measurements give. Push further (1e6) and the two pools merge, the 1-um")
    print("fraction collapses to a single-population number, and the ETE dependence the model")
    print("exists to express is gone. So this table does NOT select W_mig. What it is dominated by")
    print("is the 2026-09-14 note's limit 6 -- a factor-2 uncertainty on sigma_e_Yb compounded by")
    print("the exponential sensitivity of an ASE fraction to that gain. Section 3 fits (f, k_tr)")
    print("per fiber to BOTH observables instead of reading f off the ETE, which is the right")
    print("response to a scale error in one of them.")
    print()


# =============================================================================================
def _fit_one(label, build, target, f0, k0, w_mig):
    # relax = 1.0 (the undamped Gauss-Seidel default) converges these fibers in 3-9 iterations
    # and takes 0.1-0.7 s per solve; relax = 0.5 needs 26-29 and 4-6x the time for the SAME fixed
    # point. Every residual evaluation here is one of those solves, so the choice is the
    # difference between a minute and an hour -- and it is checked, not assumed: `converged` is
    # reported on every fit below.
    fit = eryb_fit_to_device(build, target, migration_per_s=w_mig, f0=f0, k_tr0_m3_s=k0,
                             observe=lambda amp: device_observables(amp, n_nodes=NODES,
                                                                    relax=1.0),
                             max_nfev=MAX_NFEV)
    print("  %-44s f %.4f  k_tr %.4g  f*k_tr %.4g  k_tr*N_Er %.4g 1/s"
          % (label[:44], fit.coupled_fraction, fit.k_tr_m3_s, fit.effective_f_k_m3_s,
             fit.k_tr_m3_s * getattr(build, "n_er_m3", float("nan"))))
    print("      model: out %.4g W  1-um %.4g  gain %.3g dB  eta_tr %.4g   (%d solves, %s)"
          % (fit.model["p_out_W"], fit.model["one_um_fraction"], fit.model["gain_dB"],
             fit.model["eta_transfer"], fit.n_solves,
             "converged" if fit.converged else "NOT converged"))
    print("      residuals %s | corr(logit f, ln k) %+0.3f | sigma[ln f*k] %.3f | "
          "sigma[ln f/k] %.3f | cond %.3g"
          % ({k: round(v, 3) for k, v in fit.residuals.items()}, fit.correlation,
             fit.sigma_ln_f_k, fit.sigma_ln_ratio, fit.condition_number))
    return fit


def section3():
    print("=" * 100)
    print("3. PER-FIBER (f, k_tr), fitted to output power AND 1-um fraction, W_mig held at %.3g"
          % W_MIG_PHOSPHOSILICATE_PER_S)
    print("=" * 100)
    fits = {}
    for rec in FORC_FIBERS:
        p_out = rec["pce"] * FORC_PUMP_W + FORC_SEED_W
        tgt = DeviceTarget(p_out_W=p_out, one_um_fraction=rec["one_um"])

        def build(f, k, _rec=rec):
            return forc_amp(_rec, f, k, W_MIG_PHOSPHOSILICATE_PER_S, n_er_ase=8, n_yb_ase=6)

        build.n_er_m3 = rec["n_er_m3"]
        print("  target: out %.3f W (PCE %.3f of %.0f W + seed), 1-um fraction %.4g, ETE %.2f%s"
              % (p_out, rec["pce"], FORC_PUMP_W, rec["one_um"], rec["ete"],
                 "  [composition ASSUMED -- see the header]" if rec["assumed"] else ""))
        fits[rec["label"]] = _fit_one(rec["label"], build, tgt, f0=max(rec["ete"], 0.2),
                                      k0=3.0e-21, w_mig=W_MIG_PHOSPHOSILICATE_PER_S)
        print()
    tgt = DeviceTarget(p_out_W=2.6, gain_dB=19.4, one_um_fraction=0.01,
                       one_um_is_upper_bound=True)
    # 'no obvious 1-um ASE' is scored as an upper bound of 1% of the output, which is roughly the
    # level at which a 1-um shoulder becomes obvious on an OSA trace next to a 19 dB signal.
    print("  target: Bai 2015 stage 1 -- 2.6 W out, 19.4 dB, 'no obvious 1-um ASE' (scored as an")
    print("  UPPER BOUND of 1%, not as a measurement)")
    def bai_build(f, k):
        return bai_amp(f, k, W_MIG_PHOSPHOSILICATE_PER_S, n_er_ase=8, n_yb_ase=6)

    bai_build.n_er_m3 = density_from_absorption(
        30.0, 1535.0, float(erbium().sigma_a.sigma(1535e-9)), 6.0e-6, 0.20)   # 1.79e25 m^-3
    fits["Bai 2015 stage 1"] = _fit_one("Bai 2015 stage 1 (PM-EYDF-12/130)", bai_build, tgt,
                                        f0=0.6, k0=2.0e-21,
                                        w_mig=W_MIG_PHOSPHOSILICATE_PER_S)
    print()
    print("READING. `sigma[ln f*k]` against `sigma[ln f/k]` is the whole point: the PRODUCT is")
    print("what the output power fixes and the SPLIT is what the 1-um fraction fixes, and a")
    print("correlation near -1 with a large sigma[ln f/k] means only the product was pinned.")
    print()


# =============================================================================================
def section4():
    print("=" * 100)
    print("4. TEMPERATURE: what the two laws do")
    print("=" * 100)
    st = YB_STARK_976_CANAT_DUSSARDIER
    print("YbStarkThermal from Canat/Dussardier two points (7 pct at 300 K, 13 pct at 400 K):")
    print("  Delta_E/kB = %.1f K = %.0f cm^-1, degeneracy %.3f"
          % (st.delta_E_over_k_K, st.delta_E_over_k_K / 1.4388, st.degeneracy))
    print("  %-8s %-14s %-14s" % ("T (K)", "upper fraction", "sigma scale"))
    for T in (300.0, 350.0, 400.0, 450.0, 480.0, 500.0):
        print("  %-8.0f %-14.4f %-14.4f" % (T, float(st.upper_fraction(T)),
                                            float(st.factor(T, 300.0))))
    print("  Cheng MEASURE sigma_a_Yb(974 nm) at 0.62 of its 300 K value at 480 K; the Stark")
    print("  depopulation alone gives %.3f, so it accounts for about a third of the measured"
          % float(st.factor(480.0, 300.0)))
    print("  loss and thermal line broadening (not modelled) for the rest.")
    print()
    print("RateTemperatureLaw:")
    print("  Cheng 2022 transfer rate 1/tau_1 - 1/tau_2: %.4g -> %.4g 1/s over 300 -> 480 K"
          % (1.0 / 8.20e-6 - 1.0 / 1333.76e-6, 1.0 / 7.56e-6 - 1.0 / 1279.25e-6))
    for nm, law in (("RATE_ARRHENIUS_CHENG_2022", RATE_ARRHENIUS_CHENG_2022),
                    ("RATE_ARRHENIUS_30PCT_300_480K", RATE_ARRHENIUS_30PCT_300_480K)):
        s480 = float(np.asarray(law.scale(480.0, 300.0)[0]))
        print("  %-30s Ea/kB %7.2f K  -> k_tr(480 K)/k_tr(300 K) = %.4f"
              % (nm, law.k_tr_ea_over_k_K, s480))
    print()
    print("A HOT FIBER (6 um core, 1 W-class cladding pump, f = 0.5): the 1-um ASE fraction")
    print("against temperature and pump, with both laws on. The 1-um THRESHOLD is the pump")
    print("power at which the 1-um fraction reaches 1 percent.")
    er, yb = erbium(), ytterbium("phosphosilicate")
    z = np.linspace(0.0, 3.0, 41)
    pumps = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    print("  %-8s %-8s %s" % ("T (K)", "laws", "  ".join("%-11s" % ("%g W" % p)
                                                         for p in pumps)))
    thresholds = {}
    for T in (300.0, 400.0, 500.0):
        for tag, kw in (("off", {}), ("on", dict(rate_temperature=RATE_ARRHENIUS_CHENG_2022,
                                                 yb_stark_thermal=st))):
            fr = []
            for p in pumps:
                amp = build_amp(core_um=6.0, na=0.20, length_m=3.0, clad_um=125.0,
                                n_er_m3=4.0e25, n_yb_m3=4.0e26, pump_W=p, seed_W=0.020,
                                seed_nm=1550.0, f=0.5, k_tr=3.0e-21, w_mig=0.0,
                                er_ion=er, yb_ion=yb, n_er_ase=8, n_yb_ase=8, **kw)
                amp.set_temperature_profile(z, np.full_like(z, T), T_ref_K=300.0)
                fr.append(device_observables(amp, n_nodes=121, relax=0.5)["one_um_fraction"])
            fr = np.asarray(fr)
            above = np.where(fr >= 0.01)[0]
            if above.size and above[0] > 0:
                i = above[0]
                x0, x1 = np.log(fr[i - 1]), np.log(fr[i])
                thr = pumps[i - 1] + (pumps[i] - pumps[i - 1]) * (np.log(0.01) - x0) / (x1 - x0)
            else:
                thr = pumps[0] if above.size else float("inf")
            thresholds[(T, tag)] = thr
            print("  %-8.0f %-8s %s   threshold %.2f W"
                  % (T, tag, "  ".join("%-11.4g" % v for v in fr), thr))
    print()
    print("  The threshold rises monotonically with temperature (Canat / Dussardier's")
    print("  qualitative result; Morasse 2007 measured a core-temperature rise moving his own")
    print("  1-um onset from 14 W to 35 W of pump), and turning the two laws ON raises it")
    print("  further at every temperature.")
    print()


def main():
    print("DynaMeta Er:Yb migration / thermal / fit validation -- nodes %d, max_nfev %d"
          % (NODES, MAX_NFEV))
    print()
    if "1" in SECTIONS:
        section1()
    if "2" in SECTIONS:
        section2()
    if "3" in SECTIONS:
        section3()
    if "4" in SECTIONS:
        section4()
    return 0


if __name__ == "__main__":
    sys.exit(main())
