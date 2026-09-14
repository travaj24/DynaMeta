"""Er:Yb CALIBRATION: turning measured ytterbium fluorescence decays, quenched Yb lifetimes and
device operating points into the three parameters the two-population model of `eryb.py` actually
needs -- the coupled fraction f, the coupled-pool transfer coefficient k_tr, and the Yb-Yb
migration rate W_mig.

WHY THIS MODULE EXISTS. The 2026-09-14 two-population build shipped `yb_migration_rate_per_s`
with the honest admission that NO measurement of it had been located (audit note limit 1). It is
the one parameter that reconciles the two mutually inconsistent transfer coefficients the
literature extracts from the SAME data -- a FAST-PAIR rate near 3e-21 m^3/s from a decay curve's
fast component, and an ENSEMBLE-YIELD rate near 1.2e-22 m^3/s from the same curve's integral --
so leaving it unmeasured left the model unable to carry both. This module closes that: the free
two-pool decay has an exact closed form, and INVERTING it turns a published bi-exponential Yb
decay plus an Er-free control lifetime into (f, k_tr, W_mig) with no fitting at all.

THE FREE DECAY, EXACTLY. Switch the pump off, hold the erbium in its ground state (f2 -> 0, the
low-excitation condition every one of these measurements is taken under) and `eryb._fb_rhs3`
collapses to a LINEAR 2x2 system in the two pools' excitation fractions:

    d b2c /dt = -(a + (1-f) W) b2c + (1-f) W b2nc,     a = 1/tau_Yb + k_tr N_Er phi
    d b2nc/dt = +f W b2c - (d + f W) b2nc,             d = 1/tau_Yb

The measured fluorescence is proportional to the TOTAL excited ytterbium, i.e. to the
population-weighted I(t) = f b2c + (1-f) b2nc, and a pulse excites both pools to the SAME
excitation fraction (both absorb the pump per ion identically), so b2c(0) = b2nc(0) = 1. Then

    lam_{1,2} = [tr -/+ sqrt(tr^2 - 4 det)]/2,   tr = a + d + W,
                det = a d + W (f a + (1-f) d),   disc = (A - D)^2 + 4 f (1-f) W^2 >= 0
    I(t)      = A1 exp(-lam1 t) + A2 exp(-lam2 t),  A1 + A2 = 1,  A1 lam1 + A2 lam2 = f a + (1-f) d
    INT b2c   = (d + W)/det,   INT b2nc = (a + W)/det,   yield = f (a - d) (d + W)/det

-- the discriminant is a sum of squares, so the decay is ALWAYS a genuine bi-exponential and
never a damped oscillation, and the two integrals are the same closed forms `eryb._solve_fbb`
already uses for the steady state (there with the pump on; the algebra is one object, written
once for a root find and once for a decay).

THE INVERSION, AND WHERE W_mig COMES FROM. Three measured numbers -- the fast time constant, the
slow time constant and the fast AMPLITUDE -- plus the intrinsic lifetime tau_Yb from an Er-free
control determine the three parameters uniquely:

    m1 = A1 lam1 + A2 lam2          (the initial slope of the measured decay)
    W  = (lam1 - d)(lam2 - d)/(m1 - d),   R = k_tr N_Er = lam1 + lam2 - 2 d - W,   f = (m1 - d)/R

The middle expression is the whole point. W_mig is read off the SLOW component's excess rate over
the Er-free control, lam2 - d: a genuinely uncoupled ytterbium pool would decay at exactly the
intrinsic rate, and every 1/s by which it decays faster is excitation that MIGRATED into the
coupled pool and was transferred. Cheng et al. 2022 measure exactly that excess and remark on it
("the slow component is 26% shorter than the Er-free control") without naming the mechanism; it
is the measurement of W_mig the 2026-09-14 note could not find. See
`W_MIG_PHOSPHOSILICATE_PER_S` for the recommended value and its uncertainty.

WHAT IS HERE.
  * `yb_two_pool_decay` / `YbDecayModes` -- the forward closed form and its observables.
  * `yb_two_pool_from_decay` -- the exact 3-number inversion above.
  * `YbDecayAnchor` / `YbLifetimeAnchor` and the shipped literature anchors
    (`CHENG_2022_DECAY`, `JEONG_2007_DECAY`, `LAROCHE_2006_LIFETIMES`), so the gates and the
    audit report do not depend on a scratchpad folder.
  * `eryb_calibrate_pools` -- the constrained joint fit of (f, k_tr, W_mig) to any mix of decay
    and lifetime anchors, with a PROFILE in W_mig (the 1-D scan that gives the recommended value
    an interval rather than a point).
  * `eryb_fit_to_device` / `DeviceTarget` / `DeviceFit` -- the per-fiber-class calibration: fit
    (f, k_tr) jointly to a device's measured output power AND its measured 1-um ASE fraction with
    W_mig held, reporting the covariance and the degeneracy direction, because fitting either
    observable alone is exactly the under-determination the 2026-09-14 note's limit 8 warns about.

Units SI throughout; ASCII-only; pure numpy/scipy. References: Cheng et al., Materials 15(3), 996
(2022); Jeong et al., IEEE JSTQE 13(3), 573 (2007); Laroche et al., JOSA B 23(2), 195 (2006);
Dong et al., Opt. Express 28(11), 16244 (2020); Sefler et al., JOSA B 21(10), 1740 (2004).
Full audit trail: docs/audit/2026-09-15-eryb-migration-thermal-fit.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["YbDecayModes", "yb_two_pool_decay", "yb_two_pool_from_decay",
           "YbDecayAnchor", "YbLifetimeAnchor", "PoolCalibration", "eryb_calibrate_pools",
           "DeviceTarget", "DeviceFit", "device_observables", "eryb_fit_to_device",
           "CHENG_2022_DECAY", "JEONG_2007_DECAY", "LAROCHE_2006_LIFETIMES",
           "W_MIG_PHOSPHOSILICATE_PER_S", "W_MIG_PHOSPHOSILICATE_RANGE_PER_S",
           "W_MIG_PHOSPHOSILICATE_CEILING_PER_S",
           "F_COUPLED_LITERATURE_BOUNDS", "K_TR_FAST_LITERATURE_BOUNDS_M3_S"]


# ---------------------------------------------------------------------------------------------
# Literature ranges the fits are boxed to. NOTHING outside these is reachable by a fit here, which
# is what stops a two-observable fit from wandering to a mathematically better but unphysical
# corner (the 2026-09-14 note's limit 8).
# ---------------------------------------------------------------------------------------------
# COUPLED FRACTION f. Sefler 2004 measure a non-participating fraction of 15-17% on three
# double-clad fibers (f = 0.83-0.85); Canat 2006 fits f_np = 5-15% (f = 0.85-0.95); Jeong 2007
# measure an isolated fraction of 2-5% (f = 0.95-0.98); Cheng 2022's fast-component amplitude is
# 0.864; Dong 2020 fit f = 0.15-0.25 on two high-power lasers; FORC measure an "ETE" of 0.20-0.38
# on their fluorophosphosilicate fibers and ~0.90 on Nufern LMA-EYDF-25P/300-HE. The box spans
# the whole measured spread because f is a FIBER property, not a host constant.
F_COUPLED_LITERATURE_BOUNDS = (0.10, 0.99)

# COUPLED-POOL (fast-pair) k_tr. Cheng 2022's fast component gives 3.07e-21; Dong 2020's C63 is
# 2-3e-20 at f = 0.15-0.25; Sefler 2004's K1 is 2-4e-22; Canat 2006 fits 4e-22. The box is the
# PAIR-rate range and deliberately excludes the ensemble-yield values (6.4e-23 to 1.2e-22), which
# are the right numbers for a ONE-pool model and the wrong ones for the coupled pool.
K_TR_FAST_LITERATURE_BOUNDS_M3_S = (2.0e-22, 3.0e-20)


@dataclass(frozen=True)
class YbDecayModes:
    """The exact bi-exponential free decay of the two ytterbium pools, normalised to I(0) = 1.

    `rate_fast_per_s` >= `rate_slow_per_s` > 0 and `amp_fast` + `amp_slow` = 1. `amp_fast` is NOT
    the coupled fraction f: migration moves amplitude between the two components, and the gap
    between them is exactly the information that separates W_mig from f (see the module
    docstring). `transfer_yield` is the fraction of the initial excitation that leaves as a
    Yb -> Er transfer -- the quantity a fluorescence-yield measurement reports, and the ENSEMBLE
    observable a one-pool model has to match.
    """

    rate_fast_per_s: float
    rate_slow_per_s: float
    amp_fast: float
    amp_slow: float
    transfer_yield: float
    coupled_fraction: float
    k_tr_m3_s: float
    migration_per_s: float
    tau_yb_s: float

    @property
    def tau_fast_s(self) -> float:
        return 1.0 / self.rate_fast_per_s

    @property
    def tau_slow_s(self) -> float:
        return 1.0 / self.rate_slow_per_s

    @property
    def mean_lifetime_s(self) -> float:
        """INT I(t) dt = A1/lam1 + A2/lam2 -- the AMPLITUDE-WEIGHTED mean lifetime, which is what
        a single number quoted as "the measured fluorescence lifetime" of a quenched sample means
        and what its fluorescence quantum yield is proportional to. It is also the observable that
        makes the comparison exact in the one-pool limit: at f = 1 the decay is the single
        exponential 1/(1/tau_Yb + k_tr N_Er), which is precisely the form Laroche et al. invert
        to extract their transfer coefficient."""
        return self.amp_fast / self.rate_fast_per_s + self.amp_slow / self.rate_slow_per_s

    def intensity(self, t_s):
        """I(t), normalised to 1 at t = 0."""
        t = np.asarray(t_s, float)
        return (self.amp_fast * np.exp(-self.rate_fast_per_s * t)
                + self.amp_slow * np.exp(-self.rate_slow_per_s * t))

    def fraction_relaxed_by(self, t_s):
        """1 - I(t): the fraction of the initially excited ytterbium that has relaxed by t."""
        return 1.0 - self.intensity(t_s)


def yb_two_pool_decay(*, tau_yb_s: float, n_er_m3: float, k_tr_m3_s: float,
                      coupled_fraction: float, migration_per_s: float = 0.0,
                      er_inversion: float = 0.0, k_tr2_m3_s: float = 0.0,
                      phi: float = 1.0) -> YbDecayModes:
    """The free (pump-off) two-pool ytterbium decay in closed form -- the analytic solution of
    `eryb.ErYbAmplifier._fb_rhs3` at zero optical rates and a frozen erbium state, gated against
    that right-hand side rather than re-derived from it.

    `er_inversion` is the erbium metastable fraction f2 held during the measurement (0 for the
    low-excitation decays every anchor here is taken under); it enters through the (1 - f2)
    ground-state-acceptor factor on the primary transfer and through the f2 factor on the
    secondary transfer K2, exactly as in the march. `phi` is the back-transfer branching factor
    (1.0 unless k_back > 0).
    """
    f = float(coupled_fraction)
    if not (0.0 <= f <= 1.0):
        raise ValueError("yb_two_pool_decay: coupled_fraction must be in [0, 1]; got %r" % (f,))
    if not (tau_yb_s > 0.0):
        raise ValueError("yb_two_pool_decay: tau_yb_s must be > 0")
    if float(migration_per_s) < 0.0:
        raise ValueError("yb_two_pool_decay: migration_per_s must be >= 0")
    d = 1.0 / float(tau_yb_s)
    f2 = float(er_inversion)
    w = float(migration_per_s)
    r_tr = float(phi) * float(k_tr_m3_s) * float(n_er_m3) * (1.0 - f2)
    a = d + r_tr + float(k_tr2_m3_s) * float(n_er_m3) * f2
    cap_a = a + (1.0 - f) * w
    cap_d = d + f * w
    tr = cap_a + cap_d
    det = a * d + w * (f * a + (1.0 - f) * d)
    disc = (cap_a - cap_d) ** 2 + 4.0 * f * (1.0 - f) * w * w     # a sum of squares: never < 0
    root = float(np.sqrt(max(disc, 0.0)))
    lam1 = 0.5 * (tr + root)
    lam2 = 0.5 * (tr - root)
    m1 = f * a + (1.0 - f) * d                                    # -dI/dt at t = 0
    if root <= 1e-12 * max(tr, 1.0):
        # DEGENERATE (W = 0 with a == d, or f exactly 0 or 1 with no transfer contrast): the two
        # modes coincide and the decay is a single exponential. Putting all the amplitude on the
        # "fast" slot keeps every observable continuous through the degeneracy.
        lam1 = lam2 = 0.5 * tr
        a1, a2 = 1.0, 0.0
    else:
        a1 = (m1 - lam2) / (lam1 - lam2)
        a2 = 1.0 - a1
    yield_tr = (f * r_tr * (d + w) / det) if det > 0.0 else 0.0
    return YbDecayModes(rate_fast_per_s=float(lam1), rate_slow_per_s=float(lam2),
                        amp_fast=float(a1), amp_slow=float(a2),
                        transfer_yield=float(yield_tr), coupled_fraction=f,
                        k_tr_m3_s=float(k_tr_m3_s), migration_per_s=w,
                        tau_yb_s=float(tau_yb_s))


def yb_two_pool_from_decay(*, tau_fast_s: float, tau_slow_s: float, amp_fast: float,
                           tau_yb_s: float, n_er_m3: float) -> Dict[str, float]:
    """EXACT inversion: a measured bi-exponential Yb decay (fast/slow time constants and the fast
    component's amplitude) plus the intrinsic lifetime from an Er-free control, to
    (coupled_fraction, k_tr_m3_s, migration_per_s). No fitting, no iteration -- the three
    equations of the module docstring are triangular.

    THE MIGRATION RATE IS THE SLOW COMPONENT'S EXCESS over the Er-free control, divided by the
    coupled fraction: `W = (lam1 - d)(lam2 - d)/(m1 - d)` with `m1 - d = f k_tr N_Er`. A slow
    component AT the control lifetime means W = 0 exactly; a slow component FASTER than the
    control is excitation leaking out of the uncoupled pool, and there is no other way for it to
    leave. Refused (rather than silently clipped) when the slow component is SLOWER than the
    control, which would need a negative W and means the control and the sample are not the same
    host -- state that instead of fitting round it.

    Returns a dict with the three parameters and the diagnostics `initial_slope_per_s`,
    `transfer_rate_per_s` (= k_tr N_Er) and `transfer_yield`.
    """
    lam1, lam2 = 1.0 / float(tau_fast_s), 1.0 / float(tau_slow_s)
    if lam1 < lam2:
        lam1, lam2 = lam2, lam1
        amp_fast = 1.0 - float(amp_fast)
    d = 1.0 / float(tau_yb_s)
    a1 = float(amp_fast)
    m1 = a1 * lam1 + (1.0 - a1) * lam2
    if not (m1 > d):
        raise ValueError(
            "yb_two_pool_from_decay: the measured initial slope %.4g 1/s is not FASTER than the "
            "Er-free control rate %.4g 1/s, so there is no transfer to attribute. Check that "
            "tau_yb_s is the control (Er-free) lifetime and not the sample's own slow component."
            % (m1, d))
    if lam2 < d:
        raise ValueError(
            "yb_two_pool_from_decay: the SLOW component (%.4g us) is longer than the Er-free "
            "control (%.4g us). A two-pool model with migration can only make the uncoupled pool "
            "decay FASTER than the control, never slower, so this pair needs a negative "
            "migration rate and is refused. The usual cause is a control measured on a different "
            "host or Yb loading." % (1e6 / lam2, 1e6 / d))
    w = (lam1 - d) * (lam2 - d) / (m1 - d)
    r_tr = (lam1 + lam2 - 2.0 * d) - w
    if not (r_tr > 0.0):
        raise ValueError("yb_two_pool_from_decay: the inversion gives a non-positive transfer "
                         "rate (%.4g 1/s); the three measured numbers are inconsistent with the "
                         "two-pool model." % (r_tr,))
    f = (m1 - d) / r_tr
    det = lam1 * lam2
    return {"coupled_fraction": float(f), "k_tr_m3_s": float(r_tr / float(n_er_m3)),
            "migration_per_s": float(w), "initial_slope_per_s": float(m1),
            "transfer_rate_per_s": float(r_tr),
            "transfer_yield": float(f * r_tr * (d + w) / det)}


# ---------------------------------------------------------------------------------------------
# Measurement records
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class YbDecayAnchor:
    """One measured ytterbium fluorescence decay. Every constraint is OPTIONAL and every one is
    either a POINT (a printed number) or a BAND (a range the paper states in words); a band
    contributes zero residual anywhere inside it, which is the only honest way to score "50-70%
    of the ions relax within 10 us".

    `tau_yb_s` is the INTRINSIC lifetime for this host -- an Er-free control sample where the
    paper provides one, and the literature value for the host otherwise. It is not fitted: the
    whole inversion rests on the difference between the sample's slow component and it.
    """

    label: str
    n_er_m3: float
    tau_yb_s: float
    tau_fast_s: Optional[float] = None
    tau_slow_s: Optional[float] = None
    amp_fast: Optional[float] = None
    transfer_yield: Optional[float] = None
    tau_fast_bounds: Optional[Tuple[float, float]] = None
    tau_slow_bounds: Optional[Tuple[float, float]] = None
    amp_fast_bounds: Optional[Tuple[float, float]] = None
    relaxed_by: Tuple[Tuple[float, float, float], ...] = ()     # (t_s, frac_lo, frac_hi)
    excited_at: Tuple[Tuple[float, float, float], ...] = ()     # (t_s, I_lo, I_hi)
    n_er_is_assumed: bool = False
    source: str = ""
    note: str = ""


@dataclass(frozen=True)
class YbLifetimeAnchor:
    """One measured QUENCHED ytterbium lifetime at a stated Er (and Yb) density. The model
    counterpart is `YbDecayModes.mean_lifetime_s` -- see there for why the amplitude-weighted
    mean is the right comparison and not the slow component."""

    label: str
    n_er_m3: float
    tau_measured_s: float
    tau_yb_s: float
    n_yb_m3: Optional[float] = None
    source: str = ""
    note: str = ""


# ---- the shipped literature anchors ----------------------------------------------------------
# Cheng et al., Materials 15(3), 996 (2022), Table 2, Er/Yb/P silica core glass (Er2O3 0.09 /
# Yb2O3 0.86 / P2O5 12.45 mol%). THE decisive record: it is the only one that prints a
# bi-exponential AND an Er-free control from the same melt, which is exactly the three-plus-one
# numbers the inversion needs. N_Er = 2 x 0.0009 x 2.2e28 formula units/m^3 (the silica value;
# Cheng do not print N_Er, and at 12.45 mol% P2O5 this overstates it by perhaps 10-20%, which
# scales k_tr down by the same factor and leaves f and W_mig untouched -- they depend only on
# RATES).
CHENG_2022_DECAY = YbDecayAnchor(
    label="Cheng 2022 Er/Yb/P core glass",
    n_er_m3=3.96e25, tau_yb_s=1793.46e-6,
    tau_fast_s=8.20e-6, tau_slow_s=1333.76e-6, amp_fast=0.864, transfer_yield=0.8949,
    n_er_is_assumed=True,
    source="Materials 15(3), 996 (2022), Table 2; DOI 10.3390/ma15030996",
    note="tau_Yb is the Er-FREE control YPS (1793.46 us), NOT the sample's own slow component. "
         "The 26% gap between the two is the migration signal.")

# Jeong et al., IEEE JSTQE 13(3), 573 (2007), Fig. 4 and the surrounding text, measured on the
# SPI/Southampton 30/600 fiber that made 297 W. The paper prints the decay in words rather than
# as a table, so every constraint here is a BAND. N_Er is NOT printed and is ASSUMED to be the
# study's own 4.0e25 m^-3; that assumption sets k_tr and NOTHING else (f and W_mig come from
# rates and amplitudes, which are density-free).
JEONG_2007_DECAY = YbDecayAnchor(
    label="Jeong 2007 SPI 30/600 (297 W fiber)",
    n_er_m3=4.0e25, tau_yb_s=1.30e-3,
    tau_fast_bounds=(10.0e-6, 15.5e-6),
    amp_fast_bounds=(0.93, 0.98),
    relaxed_by=((10.0e-6, 0.50, 0.70),),
    excited_at=((100.0e-6, 0.010, 0.030),),
    n_er_is_assumed=True,
    source="IEEE J. Sel. Top. Quantum Electron. 13(3), 573 (2007); DOI 10.1109/JSTQE.2007.897178",
    note="'50-70% of the ions relax within 10 us', 'about 95% with a 10-15 us time constant', "
         "'about 2% with >= 100 us', isolated (Er-uncoupled) Yb fraction 2-5%. The 1/e time of "
         "the fast component is printed as 15.5 us (21.4 uJ trace), which sets the upper end of "
         "tau_fast_bounds.")

# Laroche et al., JOSA B 23(2), 195 (2006), Table 1: five phosphosilicate Er/Yb double-clad
# fibers spanning an order of magnitude in Er density. tau_Yb0 = 1.31 ms is recovered from the
# table's own rows (the five agree to 3.3%) and is independently corroborated by Nilsson's patent
# US6445494B1.
LAROCHE_2006_LIFETIMES = tuple(
    YbLifetimeAnchor(label="Laroche 2006 fiber %d (Yb:Er %g)" % (i, ratio),
                     n_er_m3=n_er, n_yb_m3=n_yb, tau_measured_s=tau * 1e-6, tau_yb_s=1.31e-3,
                     source="JOSA B 23(2), 195 (2006), Table 1; DOI 10.1364/JOSAB.23.000195",
                     note="the paper's own CALCULATED column reads %g us here" % calc)
    for i, (n_er, n_yb, ratio, tau, calc) in enumerate(
        ((1.03e25, 3.6e26, 35, 702.0, 744.0),
         (2.05e25, 4.1e26, 20, 406.0, 477.0),
         (2.72e25, 4.9e26, 18, 396.0, 352.0),
         (4.50e25, 7.2e26, 16, 215.0, 172.0),
         (6.30e25, 6.3e26, 10, 206.0, 137.0)), start=1))


# ---- the recommended migration rate ----------------------------------------------------------
# W_mig for phosphosilicate Er:Yb at N_Yb ~ 2-7e26 m^-3. NOT a fit: `yb_two_pool_from_decay`
# applied to Cheng et al. 2022's printed bi-exponential (tau_1 = 8.20 us, tau_2 = 1333.76 us,
# A1 = 0.864) together with the Er-free control from the SAME melt (tau_Yb0 = 1793.46 us) inverts
# EXACTLY to
#
#     f = 0.864,   k_tr = 3.06e-21 m^3/s,   W_mig = 2.22e2 1/s
#
# and that single parameter set then reproduces BOTH numbers the anchor folder extracts from that
# one measurement: the fast-pair coefficient 3.07e-21 AND the ensemble-yield coefficient
# 1.20e-22 (the model's own transfer yield at these parameters is 0.8949, the measured value to
# four figures). Reconciling those two is the entire reason `yb_migration_rate_per_s` exists, and
# this is the measurement of it that the 2026-09-14 note's limit 1 said did not exist.
#
# UNCERTAINTY. The value is set by the SLOW component's excess rate over the control,
# lam2 - d = 192 1/s, a difference of two ~600-750 1/s rates, so it is more robust than that
# framing suggests: a +/- 5% perturbation of EITHER lifetime (a bi-exponential fit's realistic
# component uncertainty) moves it only over 1.5e2 - 3.0e2 1/s, and +/- 5% on tau_fast or on the
# amplitude moves it by under 6%. The RANGE below is that measurement interval. A wider
# HOST-to-host ceiling of ~1e4 1/s comes from the second anchor: Jeong 2007's stated decay
# pattern (50-70% relaxed by 10 us, ~2% still excited at 100 us, 2-5% isolated ytterbium) is
# satisfiable for any W_mig from ~2 to ~1e4 1/s and fails above 1e5, so it CONFIRMS the Cheng
# value without sharpening it.
#
# A DOCUMENTED CONSTANT, NOT A DEFAULT: ErYbAmplifier(yb_migration_rate_per_s=...) still defaults
# to 0 and nothing in the package reads this value unless a caller passes it.
#
# WHAT IT DOES AND DOES NOT DO (audit note 2026-09-15 section 3). 2.2e2 1/s is comparable to the
# ytterbium's own 1/tau_Yb ~ 6-8e2 1/s and three orders BELOW the coupled pool's transfer rate
# k_tr N_Er ~ 1e5 1/s, so it reconciles the two transfer coefficients (which is what it was
# derived from) and shifts a device's 1-um ASE fraction by tens of percent -- it does NOT close
# the order-of-magnitude gap between the modelled and measured 1-um fractions of the FORC ETE
# series. Closing THAT needs the factor-2 sigma_e_Yb uncertainty of the 2026-09-14 note's limit
# 6, not more migration: reaching the measured low-ETE 1-um fraction takes W_mig > 1e5 1/s, which
# the decay anchors exclude by two orders of magnitude.
W_MIG_PHOSPHOSILICATE_PER_S = 2.2e2
W_MIG_PHOSPHOSILICATE_RANGE_PER_S = (1.5e2, 3.0e2)
# The looser ceiling Jeong 2007's band-only decay pattern allows, for a caller who wants to bound
# the sensitivity of a design to migration rather than to adopt the Cheng value.
W_MIG_PHOSPHOSILICATE_CEILING_PER_S = 1.0e4


# ---------------------------------------------------------------------------------------------
# The joint pool calibration
# ---------------------------------------------------------------------------------------------
def _band_residual(value: float, lo: float, hi: float, log: bool = True) -> float:
    """Zero inside [lo, hi], and the (log) distance to the nearer edge outside it. The scoring a
    BAND deserves: a paper that says "50 to 70 percent" is not measuring 60 percent."""
    v = float(value)
    if log:
        if v <= 0.0:
            return 50.0
        if v < lo:
            return float(np.log(v / lo))
        if v > hi:
            return float(np.log(v / hi))
        return 0.0
    if v < lo:
        return v - lo
    if v > hi:
        return v - hi
    return 0.0


@dataclass
class PoolCalibration:
    """The result of `eryb_calibrate_pools`."""

    coupled_fraction: float
    k_tr_m3_s: float
    migration_per_s: float
    cost: float
    converged: bool
    message: str
    decay_report: Dict[str, dict] = field(default_factory=dict)
    lifetime_report: Dict[str, dict] = field(default_factory=dict)
    w_profile: Tuple[Tuple[float, float], ...] = ()
    w_interval_per_s: Tuple[float, float] = (float("nan"), float("nan"))
    exact_inversions: Dict[str, dict] = field(default_factory=dict)

    @property
    def effective_f_k_m3_s(self) -> float:
        """f k_tr -- the product a ONE-pool model sees, and the combination every ensemble
        measurement actually constrains."""
        return self.coupled_fraction * self.k_tr_m3_s

    def __str__(self) -> str:
        return ("PoolCalibration(f=%.4f, k_tr=%.4g m^3/s, W_mig=%.4g 1/s, f*k_tr=%.4g, "
                "cost=%.4g, %s)" % (self.coupled_fraction, self.k_tr_m3_s, self.migration_per_s,
                                    self.effective_f_k_m3_s, self.cost,
                                    "converged" if self.converged else "NOT converged"))


def _decay_residuals(anchor: YbDecayAnchor, f: float, k: float, w: float) -> Tuple[list, dict]:
    m = yb_two_pool_decay(tau_yb_s=anchor.tau_yb_s, n_er_m3=anchor.n_er_m3, k_tr_m3_s=k,
                          coupled_fraction=f, migration_per_s=w)
    res, rep = [], {"tau_fast_s": m.tau_fast_s, "tau_slow_s": m.tau_slow_s,
                    "amp_fast": m.amp_fast, "transfer_yield": m.transfer_yield,
                    "mean_lifetime_s": m.mean_lifetime_s}
    if anchor.tau_fast_s is not None:
        res.append(float(np.log(m.tau_fast_s / anchor.tau_fast_s)))
    if anchor.tau_slow_s is not None:
        res.append(float(np.log(m.tau_slow_s / anchor.tau_slow_s)))
    if anchor.amp_fast is not None:
        res.append(float(m.amp_fast - anchor.amp_fast))
    if anchor.transfer_yield is not None:
        res.append(float(m.transfer_yield - anchor.transfer_yield))
    if anchor.tau_fast_bounds is not None:
        res.append(_band_residual(m.tau_fast_s, *anchor.tau_fast_bounds))
    if anchor.tau_slow_bounds is not None:
        res.append(_band_residual(m.tau_slow_s, *anchor.tau_slow_bounds))
    if anchor.amp_fast_bounds is not None:
        res.append(_band_residual(m.amp_fast, *anchor.amp_fast_bounds, log=False))
    for t, lo, hi in anchor.relaxed_by:
        v = float(m.fraction_relaxed_by(t))
        rep["relaxed_by_%.3gus" % (t * 1e6)] = v
        res.append(_band_residual(v, lo, hi, log=False))
    for t, lo, hi in anchor.excited_at:
        v = float(m.intensity(t))
        rep["excited_at_%.3gus" % (t * 1e6)] = v
        res.append(_band_residual(v, lo, hi, log=False))
    return res, rep


def _lifetime_residual(anchor: YbLifetimeAnchor, f: float, k: float, w: float) -> Tuple[float, dict]:
    m = yb_two_pool_decay(tau_yb_s=anchor.tau_yb_s, n_er_m3=anchor.n_er_m3, k_tr_m3_s=k,
                          coupled_fraction=f, migration_per_s=w)
    tau = m.mean_lifetime_s
    return (float(np.log(tau / anchor.tau_measured_s)),
            {"tau_model_s": tau, "tau_measured_s": anchor.tau_measured_s,
             "ratio": tau / anchor.tau_measured_s,
             "tau_fast_s": m.tau_fast_s, "tau_slow_s": m.tau_slow_s, "amp_fast": m.amp_fast})


def eryb_calibrate_pools(*, decays: Sequence[YbDecayAnchor] = (),
                         lifetimes: Sequence[YbLifetimeAnchor] = (),
                         f_bounds: Tuple[float, float] = F_COUPLED_LITERATURE_BOUNDS,
                         k_bounds: Tuple[float, float] = K_TR_FAST_LITERATURE_BOUNDS_M3_S,
                         w_bounds: Tuple[float, float] = (1.0, 1.0e8),
                         decay_weight: float = 1.0, lifetime_weight: float = 1.0,
                         x0: Optional[Tuple[float, float, float]] = None,
                         w_profile_points: int = 25,
                         profile_cost_factor: float = 2.0) -> PoolCalibration:
    """Fit (coupled_fraction f, k_tr, W_mig) to measured ytterbium decays and quenched lifetimes.

    f and k_tr are BOX-CONSTRAINED to the literature ranges (`F_COUPLED_LITERATURE_BOUNDS`,
    `K_TR_FAST_LITERATURE_BOUNDS_M3_S`); W_mig is free within `w_bounds`, which is the asymmetry
    the brief this was built for asks for and also the honest one -- f and k_tr have a dozen
    published determinations between them and W_mig has none.

    The residual vector is dimensionless throughout: log ratios for time constants and lifetimes,
    plain differences for amplitudes and fractions (which are already O(1)), and zero inside a
    stated band. `decay_weight` / `lifetime_weight` scale the two families; setting one to 0 fits
    the other alone, which is how the audit note tells the two datasets apart.

    Also returns, without fitting anything:
      * `exact_inversions` -- `yb_two_pool_from_decay` applied to every decay anchor that prints a
        full (tau_fast, tau_slow, amp_fast) triple. These are EXACT and are the values the
        recommended `W_MIG_PHOSPHOSILICATE_PER_S` rests on; the fit is the way to combine them
        with band-only anchors, not a replacement for them.
      * `w_profile` -- (W_mig, cost) with (f, k_tr) re-minimised at each W on a log grid, and
        `w_interval_per_s`, the range over which the profile stays within `profile_cost_factor`
        of its minimum. A flat profile means the data do not constrain W_mig, and saying so is the
        result.
    """
    from scipy.optimize import least_squares

    decays, lifetimes = list(decays), list(lifetimes)
    if not decays and not lifetimes:
        raise ValueError("eryb_calibrate_pools: give at least one decay or lifetime anchor")

    def residuals(v, w_fixed=None):
        f = 1.0 / (1.0 + np.exp(-v[0]))
        k = float(np.exp(v[1]))
        w = float(np.exp(v[2])) if w_fixed is None else float(w_fixed)
        out = []
        for a in decays:
            r, _rep = _decay_residuals(a, f, k, w)
            out.extend([decay_weight * x for x in r])
        for a in lifetimes:
            r, _rep = _lifetime_residual(a, f, k, w)
            out.append(lifetime_weight * r)
        return np.asarray(out, float)

    def logit(p):
        return float(np.log(p / (1.0 - p)))

    lo = [logit(max(f_bounds[0], 1e-6)), float(np.log(k_bounds[0])), float(np.log(w_bounds[0]))]
    hi = [logit(min(f_bounds[1], 1.0 - 1e-6)), float(np.log(k_bounds[1])),
          float(np.log(w_bounds[1]))]
    if x0 is None:
        x0 = (float(np.sqrt(f_bounds[0] * f_bounds[1])),
              float(np.sqrt(k_bounds[0] * k_bounds[1])),
              float(np.sqrt(w_bounds[0] * w_bounds[1])))
    v0 = [logit(min(max(x0[0], f_bounds[0] + 1e-9), f_bounds[1] - 1e-9)),
          float(np.log(x0[1])), float(np.log(x0[2]))]
    v0 = [float(min(max(v0[i], lo[i] + 1e-9), hi[i] - 1e-9)) for i in range(3)]
    sol = least_squares(residuals, v0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    f_hat = 1.0 / (1.0 + np.exp(-sol.x[0]))
    k_hat = float(np.exp(sol.x[1]))
    w_hat = float(np.exp(sol.x[2]))

    dec_rep, life_rep = {}, {}
    for a in decays:
        r, rep = _decay_residuals(a, f_hat, k_hat, w_hat)
        rep["residuals"] = list(r)
        dec_rep[a.label] = rep
    for a in lifetimes:
        r, rep = _lifetime_residual(a, f_hat, k_hat, w_hat)
        rep["residual"] = r
        life_rep[a.label] = rep

    exact = {}
    for a in decays:
        if a.tau_fast_s and a.tau_slow_s and a.amp_fast is not None:
            try:
                exact[a.label] = yb_two_pool_from_decay(
                    tau_fast_s=a.tau_fast_s, tau_slow_s=a.tau_slow_s, amp_fast=a.amp_fast,
                    tau_yb_s=a.tau_yb_s, n_er_m3=a.n_er_m3)
            except ValueError as exc:                                          # noqa: PERF203
                exact[a.label] = {"error": str(exc)}

    # ---- the W_mig profile -------------------------------------------------------------------
    prof = []
    if w_profile_points > 1:
        for w in np.geomspace(w_bounds[0], w_bounds[1], int(w_profile_points)):
            s2 = least_squares(lambda v, _w=w: residuals([v[0], v[1], 0.0], w_fixed=_w),
                               [sol.x[0], sol.x[1]], bounds=(lo[:2], hi[:2]),
                               xtol=1e-12, ftol=1e-12, gtol=1e-12)
            prof.append((float(w), float(2.0 * s2.cost)))
    w_int = (float("nan"), float("nan"))
    if prof:
        costs = np.array([c for _w, c in prof])
        ws = np.array([w for w, _c in prof])
        ok = costs <= max(costs.min(), 1e-30) * float(profile_cost_factor) + 1e-30
        if ok.any():
            w_int = (float(ws[ok].min()), float(ws[ok].max()))
    return PoolCalibration(coupled_fraction=float(f_hat), k_tr_m3_s=k_hat, migration_per_s=w_hat,
                           cost=float(2.0 * sol.cost), converged=bool(sol.success),
                           message=str(sol.message), decay_report=dec_rep,
                           lifetime_report=life_rep, w_profile=tuple(prof),
                           w_interval_per_s=w_int, exact_inversions=exact)


# ---------------------------------------------------------------------------------------------
# The per-device (f, k_tr) fit
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceTarget:
    """What a device record measures, and how well. `sigma_ln_*` are the LOG-scale one-sigma
    uncertainties the residuals are divided by; their defaults encode the 2026-09-14 audit note's
    own limits -- 5% on an output power a paper prints, and a FACTOR OF TWO on any 1-um ASE
    magnitude (limit 6: Morasse 2006 needed 0.4x the McCumber sigma_e_Yb to match his measured
    1-um ASE, and an ASE fraction is exponentially sensitive to a 20-30 dB single-pass gain).

    `one_um_is_upper_bound` scores a "no obvious 1-um ASE" statement correctly: zero residual
    anywhere below the bound, a log penalty above it. Papers state that far more often than they
    state a number, and treating the bound as a measurement would fit to a fiction.
    """

    p_out_W: Optional[float] = None
    one_um_fraction: Optional[float] = None
    one_um_is_upper_bound: bool = False
    gain_dB: Optional[float] = None
    sigma_ln_power: float = 0.05
    sigma_ln_one_um: float = 0.693
    sigma_gain_dB: float = 0.5


@dataclass
class DeviceFit:
    """The result of `eryb_fit_to_device`.

    `correlation` is the (logit f, ln k_tr) correlation coefficient: a value near -1 says the fit
    has only determined the PRODUCT f k_tr, which is exactly the degeneracy a single observable
    leaves and the reason this routine insists on two. `sigma_ln_f_k` is the uncertainty on
    ln(f k_tr), the well-determined combination, and `sigma_ln_ratio` the uncertainty along the
    degenerate direction -- the two together are the honest error bar.
    """

    coupled_fraction: float
    k_tr_m3_s: float
    cost: float
    converged: bool
    message: str
    n_solves: int
    residuals: Dict[str, float] = field(default_factory=dict)
    model: Dict[str, float] = field(default_factory=dict)
    covariance: Optional[np.ndarray] = None
    correlation: float = float("nan")
    sigma_ln_f_k: float = float("nan")
    sigma_ln_ratio: float = float("nan")
    condition_number: float = float("nan")
    degeneracy_direction: Optional[np.ndarray] = None
    migration_per_s: float = 0.0

    @property
    def effective_f_k_m3_s(self) -> float:
        return self.coupled_fraction * self.k_tr_m3_s

    def __str__(self) -> str:
        return ("DeviceFit(f=%.4f, k_tr=%.4g, f*k_tr=%.4g, corr=%+.3f, "
                "sigma[ln f*k]=%.3f, sigma[ln f/k]=%.3f, cost=%.4g)"
                % (self.coupled_fraction, self.k_tr_m3_s, self.effective_f_k_m3_s,
                   self.correlation, self.sigma_ln_f_k, self.sigma_ln_ratio, self.cost))


def device_observables(amp, *, n_nodes: int = 201, relax=0.5,
                       one_um_max_m: float = 1.2e-6, **solve_kw) -> Dict[str, float]:
    """Solve `amp` and return the observables a device record states: signal output power, signal
    gain, the 1-um ASE power leaving the fiber (both directions, every ASE bin below
    `one_um_max_m`), the 1-um fraction of the total light out, and eta_transfer.

    The 1-um FRACTION is defined as P_1um / (P_signal + P_1um), which is the quantity FORC and the
    high-power papers quote as "the share of ASE near 1030 nm relative to the output signal
    power" to within the difference between a share of the total and a ratio to the signal -- at
    the sub-percent levels those papers report the two coincide to better than the factor-2
    uncertainty on either."""
    r = amp.solve(n_nodes=n_nodes, relax=relax, **solve_kw)
    sig = [i for i, k in enumerate(r.kind) if k == "signal"]
    yb = [i for i, k in enumerate(r.kind) if k == "ase" and r.lambda_m[i] < one_um_max_m]
    p_sig = float(sum(r.power_W[i, -1] for i in sig))
    p_yb = float(sum(r.power_W[i, -1 if r.u[i] > 0 else 0] for i in yb))
    return {"p_out_W": p_sig, "p_one_um_W": p_yb,
            "one_um_fraction": p_yb / max(p_sig + p_yb, 1e-300),
            "gain_dB": float(r.signal_gain_dB[0]) if len(sig) else float("nan"),
            "eta_transfer": float(r.meta["eta_transfer"]),
            "yb_parasitic_gain_dB": float(r.meta["yb_parasitic_gain_dB"]),
            "converged": float(bool(r.meta["converged"]))}


def eryb_fit_to_device(build: Callable, target: DeviceTarget, *,
                       migration_per_s: float = 0.0,
                       f0: float = 0.5, k_tr0_m3_s: float = 1.0e-21,
                       f_bounds: Tuple[float, float] = F_COUPLED_LITERATURE_BOUNDS,
                       k_bounds: Tuple[float, float] = K_TR_FAST_LITERATURE_BOUNDS_M3_S,
                       observe: Optional[Callable] = None,
                       max_nfev: int = 40, diff_step: float = 0.08) -> DeviceFit:
    """Fit the coupled fraction f and the coupled-pool k_tr of ONE device to its measured output
    power AND its measured 1-um ASE fraction, with the migration rate HELD at `migration_per_s`
    (the calibrated `W_MIG_PHOSPHOSILICATE_PER_S`, normally).

    WHY BOTH OBSERVABLES, ALWAYS. The 2026-09-14 note's limit 8: Dong's own paper fits two
    mutually inconsistent one-pool coefficients (1.1e-21 from the Yb parasitic threshold, 2.63e-21
    from the maximum output) to the SAME laser. Output power is sensitive almost entirely to the
    PRODUCT f k_tr -- the effective transfer strength -- while the 1-um fraction is sensitive to
    the SPLIT, because it is the uncoupled pool that radiates at 1 um. Fit one and the other is
    unconstrained; fit both and the covariance this routine returns says how well each is pinned.

    `build(f, k_tr)` must return a configured `ErYbAmplifier` (the caller owns the fiber, the
    densities, the pump geometry and the ASE bands -- this routine owns only the two parameters).
    `observe(amp)` defaults to `device_observables`; pass a partial with your own `n_nodes` /
    `relax` to control the cost, since every residual evaluation is a full relaxation solve.

    The optimisation runs in (logit f, ln k_tr) so both parameters are scale-free and the box
    constraints are exact. `diff_step` is the RELATIVE finite-difference step in that space; the
    default 0.08 is deliberately coarse because the residual is a numerical solve whose own
    convergence noise would otherwise dominate a tighter step.
    """
    from scipy.optimize import least_squares

    obs = observe if observe is not None else device_observables
    n_calls = [0]

    def logit(p):
        return float(np.log(p / (1.0 - p)))

    def unpack(v):
        return 1.0 / (1.0 + np.exp(-float(v[0]))), float(np.exp(float(v[1])))

    def model_at(v):
        f, k = unpack(v)
        n_calls[0] += 1
        return obs(build(f, k))

    def resid_from(m):
        out = {}
        if target.p_out_W is not None:
            out["ln_power"] = (np.log(max(m["p_out_W"], 1e-300) / target.p_out_W)
                               / target.sigma_ln_power)
        if target.gain_dB is not None:
            out["gain_dB"] = (m["gain_dB"] - target.gain_dB) / target.sigma_gain_dB
        if target.one_um_fraction is not None:
            v = np.log(max(m["one_um_fraction"], 1e-300) / target.one_um_fraction)
            if target.one_um_is_upper_bound:
                v = max(v, 0.0)
            out["ln_one_um"] = v / target.sigma_ln_one_um
        if not out:
            raise ValueError("eryb_fit_to_device: the DeviceTarget states no measurement")
        return out

    keys = list(resid_from(obs(build(float(f0), float(k_tr0_m3_s)))).keys())

    def residuals(v):
        m = model_at(v)
        r = resid_from(m)
        return np.asarray([r[k] for k in keys], float)

    lo = [logit(max(f_bounds[0], 1e-6)), float(np.log(k_bounds[0]))]
    hi = [logit(min(f_bounds[1], 1.0 - 1e-6)), float(np.log(k_bounds[1]))]
    v0 = [float(min(max(logit(float(f0)), lo[0] + 1e-9), hi[0] - 1e-9)),
          float(min(max(np.log(float(k_tr0_m3_s)), lo[1] + 1e-9), hi[1] - 1e-9))]
    sol = least_squares(residuals, v0, bounds=(lo, hi), diff_step=diff_step,
                        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=int(max_nfev))
    f_hat, k_hat = unpack(sol.x)
    m_hat = model_at(sol.x)
    r_hat = resid_from(m_hat)

    cov = corr = s_fk = s_ratio = cond = None
    direction = None
    try:
        # The covariance is built from the EIGEN-DECOMPOSITION of J^T J rather than from
        # np.linalg.inv, and a direction the data does not constrain at all is reported as an
        # INFINITE uncertainty rather than inverted. Inverting a numerically singular J^T J
        # (which is exactly what happens when one residual is an inactive one-sided bound, so its
        # row of the Jacobian is identically zero) returns garbage that can come back as a
        # spuriously SMALL sigma -- an error bar that says the opposite of the truth.
        jtj = sol.jac.T @ sol.jac
        evals, evecs = np.linalg.eigh(jtj)
        evals = np.maximum(evals, 0.0)
        top = float(evals.max()) if evals.size else 0.0
        cond = float(top / evals.min()) if (evals.size and evals.min() > 0.0) else float("inf")
        direction = np.asarray(evecs[:, 0], float)          # the LEAST constrained direction
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_evals = np.where(evals > 1e-12 * max(top, 1e-300), 1.0 / evals, np.inf)
        cov = evecs @ np.diag(inv_evals) @ evecs.T
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        # ln(f k) has gradient (1 - f, 1) in (logit f, ln k); the orthogonal (degenerate) ratio
        # direction ln(f/k) has gradient (1 - f, -1).
        g_prod = np.array([1.0 - f_hat, 1.0])
        g_ratio = np.array([1.0 - f_hat, -1.0])

        def _sigma(g):
            proj = (evecs.T @ g) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(proj > 0.0, proj * inv_evals, 0.0)
            return float(np.sqrt(np.sum(terms)))

        s_fk, s_ratio = _sigma(g_prod), _sigma(g_ratio)
    except np.linalg.LinAlgError:
        pass
    return DeviceFit(coupled_fraction=float(f_hat), k_tr_m3_s=float(k_hat),
                     cost=float(2.0 * sol.cost), converged=bool(sol.success),
                     message=str(sol.message), n_solves=int(n_calls[0]),
                     residuals={k: float(v) for k, v in r_hat.items()},
                     model={k: float(v) for k, v in m_hat.items()},
                     covariance=cov,
                     correlation=float("nan") if corr is None else corr,
                     sigma_ln_f_k=float("nan") if s_fk is None else s_fk,
                     sigma_ln_ratio=float("nan") if s_ratio is None else s_ratio,
                     condition_number=float("nan") if cond is None else cond,
                     degeneracy_direction=direction,
                     migration_per_s=float(migration_per_s))
