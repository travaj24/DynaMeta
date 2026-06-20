"""QD-SOA carrier-density-dependent linewidth factor alpha(rho) + polarization-dependent gain (PDG)
vs analytic oracles. Two opt-in realism upgrades replacing scalar/single-pol reductions:

  (1) alpha(rho): the linewidth enhancement factor rises with inversion as the gain clamps
      (dg/dN falls, the carrier-induced index dn/dN persists) -- QDGainParams.alpha_lef_density_slope
      sets d(alpha)/d(rho_GS); the coherent marcher applies it PER SLICE from the local carrier
      state. (The FREQUENCY dependence of the carrier-induced index is the resonant Kramers-Kronig
      line filter, already shipped.)
  (2) PDG: amplify_coherent_dualpol co-propagates TE + TM envelopes through ONE shared carrier
      reservoir with a TM/TE modal-gain ratio pdg_ratio, so the two pols cross-saturate.

GATE A (alpha reduction): slope=0 -> amplify_coherent byte-identical to a constant alpha; the
        alpha_lef_slices helper returns the scalar alpha_lef.
GATE B (alpha formula): alpha_lef_slices == alpha_lef + (slope/2)(2 rho_GS - 1), exact at a state.
GATE C (alpha is the chirp/gain ratio): the DEFINING relation -- a CW tone's imposed phase over its
        log-amplitude gain, -arg(A_out)/ln|A_out/A_in|, equals the gain-weighted z-average of
        alpha(rho(z)) from the steady state (= alpha_lef when slope=0; shifted by the density slope
        otherwise). This is the physical meaning of the linewidth factor and ties the imposed index
        to the local alpha.
GATE D (PDG reduction): pdg_ratio=1 with only TE excited -> A_te_out byte-identical to single-pol
        amplify_coherent (the pols are gain-degenerate and uncoupled when the other is dark).
GATE E (PDG ratio): small-signal 10 log10(G_TE/G_TM) == (1 - pdg_ratio) Gamma g L * 10/ln10.
GATE F (PDG cross-saturation): a STRONG TE signal depletes the shared reservoir, dropping the gain a
        weak TM probe sees (vs the TM-alone gain) -- the shared-carrier physics single-pol misses.

Run: python -m validation.qd_soa_alpha_pdg
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams
from dynameta.optics.soa.traveling_wave import TravelingWaveSOA


def main():
    print("[ap] === QD-SOA alpha(rho) density dependence + polarization-dependent gain ===",
          flush=True)
    ok = True

    # ---- alpha(rho): GATES A-C ----
    m0 = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    L, nz = 0.6e-3, 60
    soa0 = TravelingWaveSOA(m0, L, nz, nu_s_Hz=m0.p.nu0_Hz)
    nt = 2400
    eps = 1.0e-4
    A = np.full(nt, eps) + 0j

    # GATE A: slope=0 byte-identical to constant alpha + scalar helper
    base = soa0.amplify_coherent(A, 40e-3)["A_out"]                 # model alpha=2.0, slope=0
    same = soa0.amplify_coherent(A, 40e-3, alpha_lef=2.0)["A_out"]  # explicit constant
    sc = m0.alpha_lef_slices(m0.init_slices(nz, 40e-3))
    g_a = bool(np.array_equal(base, same) and np.ndim(sc) == 0 and abs(sc - 2.0) < 1e-15)
    ok = ok and g_a
    print("[ap] GATE A: slope=0 == constant-alpha engine (byte-identical) + scalar helper -> {}".format(
        "PASS" if g_a else "FAIL"), flush=True)

    # GATE B: alpha_lef_slices formula exact
    slope = 4.0
    md = QDGainModel(QDGainParams(n_groups=21, alpha_lef_density_slope=slope)
                     .with_detailed_balance_taus())
    st = md.init_slices(nz, 40e-3)
    al = md.alpha_lef_slices(st)
    inv = md._gs_inversion(st)
    inv_mean = np.tensordot(inv, md.w_j, axes=([-1], [0])) / np.sum(md.w_j)
    pred = md.p.alpha_lef + 0.5 * slope * inv_mean
    relB = float(np.max(np.abs(al - pred)))
    g_b = bool(relB < 1e-13 and np.ndim(al) == 1)
    ok = ok and g_b
    print("[ap] GATE B: alpha_lef_slices == alpha_lef + (slope/2)(2 rho-1) (max abs {:.1e}, alpha "
          "{:.3f}..{:.3f}) -> {}".format(relB, float(np.min(al)), float(np.max(al)),
                                         "PASS" if g_b else "FAIL"), flush=True)

    # GATE C: -arg(A_out)/ln|gain| == the GAIN-WEIGHTED z-average of alpha(rho(z)). Unsaturated CW
    # (z-uniform) ties the imposed index to the local alpha; a strongly-saturating CW (where the
    # power grows along z, so g(z)/alpha(z) develop a real z-gradient) DISCRIMINATES the gain-
    # weighting from a plain z-average -- the gate then fails if the weighting were replaced by a mean.
    def alpha_eff_and_avgs(mm, eng, Acw, amp_in):
        r = eng.amplify_coherent(Acw, 40e-3)
        ao = r["A_out"][Acw.size - 1]
        meas = -np.angle(ao) / np.log(np.abs(ao) / amp_in)        # -dphi / d ln|A| = alpha_eff
        g_sl = mm.gain_per_m_slices(r["state"], mm.p.nu0_Hz)
        a_sl = mm.alpha_lef_slices(r["state"])
        a_sl = np.full(nz, a_sl) if np.ndim(a_sl) == 0 else a_sl
        return meas, float(np.sum(g_sl * a_sl) / np.sum(g_sl)), float(np.mean(a_sl))

    relC = 0.0
    for sl in (0.0, 4.0):
        mm = QDGainModel(QDGainParams(n_groups=21, alpha_lef_density_slope=sl)
                         .with_detailed_balance_taus())
        meas, gw, _ = alpha_eff_and_avgs(mm, TravelingWaveSOA(mm, L, nz, nu_s_Hz=mm.p.nu0_Hz), A, eps)
        relC = max(relC, abs(meas - gw) / abs(gw))                # unsaturated: matches to machine
    # saturating discrimination: gain-weighted beats a plain z-average, and a real gradient exists
    msat = QDGainModel(QDGainParams(n_groups=21, alpha_lef_density_slope=4.0)
                       .with_detailed_balance_taus())
    Asat = np.full(6000, np.sqrt(0.25)) + 0j                      # P = 0.25 W >> P_sat -> z-gradient
    m_s, gw_s, zavg_s = alpha_eff_and_avgs(msat, TravelingWaveSOA(msat, L, nz, nu_s_Hz=msat.p.nu0_Hz),
                                           Asat, np.sqrt(0.25))
    discrim = bool(abs(m_s - gw_s) < 0.3 * abs(m_s - zavg_s) and (gw_s - zavg_s) > 1e-2)
    g_c = bool(relC < 1e-3 and discrim)
    ok = ok and g_c
    print("[ap] GATE C: -arg/ln|gain| == gain-weighted <alpha> (unsat rel {:.1e}); under saturation "
          "|meas-gw|={:.1e} << |meas-zavg|={:.1e} (gradient {:.2e}) -> {}".format(
              relC, abs(m_s - gw_s), abs(m_s - zavg_s), gw_s - zavg_s, "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- PDG: GATES D-F (a higher-gain device so the cross-saturation is unambiguous) ----
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    Lp, nzp, Idrv = 3.0e-3, 150, 120e-3
    soa = TravelingWaveSOA(m, Lp, nzp, nu_s_Hz=m.p.nu0_Hz)
    ntp = 4000
    Aw = np.full(ntp, eps) + 0j
    zero = np.zeros(ntp) + 0j
    kp = ntp - 1

    # GATE D: pdg_ratio=1, TE-only == single-pol amplify_coherent
    single = soa.amplify_coherent(Aw, Idrv, alpha_lef=0.0)["A_out"]
    dz = soa.amplify_coherent_dualpol(Aw, zero, Idrv, alpha_lef=0.0, pdg_ratio=1.0)
    g_d = bool(np.max(np.abs(dz["A_te_out"] - single)) < 1e-15)
    ok = ok and g_d
    print("[ap] GATE D: PDG TE-only (r=1) == single-pol amplify_coherent (max|d| {:.1e}) -> {}".format(
        float(np.max(np.abs(dz["A_te_out"] - single))), "PASS" if g_d else "FAIL"), flush=True)

    # GATE E: small-signal PDG == (1-r) Gamma g L * 10/ln10
    rr = 0.6
    d2 = soa.amplify_coherent_dualpol(Aw, Aw, Idrv, alpha_lef=0.0, pdg_ratio=rr)
    Gte = np.abs(d2["A_te_out"][kp]) ** 2 / eps ** 2
    Gtm = np.abs(d2["A_tm_out"][kp]) ** 2 / eps ** 2
    pdg_meas = 10.0 * np.log10(Gte / Gtm)
    y = m.steady_state(Idrv)
    g = float(m.material_gain_per_m(m.rho_GS(y), m.p.nu0_Hz))
    pdg_pred = (1.0 - rr) * m.p.Gamma * g * Lp * 10.0 / np.log(10.0)
    relE = abs(pdg_meas - pdg_pred) / abs(pdg_pred)
    g_e = bool(relE < 5e-2)
    ok = ok and g_e
    print("[ap] GATE E: small-signal PDG meas {:.3f} dB == (1-r)Gamma g L pred {:.3f} dB (rel {:.1e}) "
          "-> {}".format(pdg_meas, pdg_pred, relE, "PASS" if g_e else "FAIL"), flush=True)

    # GATE F: cross-saturation -- strong TE drops the weak-TM gain
    strong = np.full(ntp, np.sqrt(0.10)) + 0j
    tm_alone = soa.amplify_coherent_dualpol(zero, Aw, Idrv, alpha_lef=0.0,
                                            pdg_ratio=rr)["A_tm_out"][kp]
    tm_te = soa.amplify_coherent_dualpol(strong, Aw, Idrv, alpha_lef=0.0,
                                         pdg_ratio=rr)["A_tm_out"][kp]
    g_alone = 20.0 * np.log10(abs(tm_alone) / eps)
    g_sat = 20.0 * np.log10(abs(tm_te) / eps)
    drop = g_alone - g_sat
    g_f = bool(drop > 0.5)
    ok = ok and g_f
    print("[ap] GATE F: TM gain {:.2f} dB alone -> {:.2f} dB with strong TE (cross-sat drop {:.2f} "
          "dB > 0.5) -> {}".format(g_alone, g_sat, drop, "PASS" if g_f else "FAIL"), flush=True)

    print("[ap] *** QD-SOA ALPHA(RHO) + PDG: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
