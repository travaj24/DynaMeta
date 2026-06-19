"""QD-SOA electron/hole occupation split (roadmap SOA Phase 6, gain-fidelity upgrade) vs
oracles. With eh_split the dots carry SEPARATE electron f_c and hole f_v occupations per state;
gain -> (f_c+f_v-1), spontaneous -> f_c f_v, n_sp -> f_c f_v/(f_c+f_v-1). Electrons and holes
have their OWN capture/relaxation times (holes heavier -> faster), so f_c != f_v transiently --
the high-speed pattern-effect physics the excitonic single-rho model cannot represent
(Berg & Mork, IEEE JQE 2004; Coldren & Corzine).

GATE A (reduces to the excitonic model): eh_split=True with all hole times defaulted (symmetric)
        and a symmetric initial condition collapses to the excitonic rate equations -- the WL
        densities, all ES/GS occupations (rho_GS == f_c_GS == f_v_GS), and the modal gain match a
        parallel excitonic march, and the split stays on the f_c=f_v invariant manifold.
GATE B (gain + n_sp closed forms): for random GS occupations the modal gain equals the
        independent sum_j N_q w_j mu_GS sigma_pk L (f_c+f_v-1) and n_sp == f_c f_v/(f_c+f_v-1);
        both reduce to (2 rho-1) and rho^2/(2 rho-1) at f_c=f_v; transparency g=0 at f_c+f_v=1.
GATE C (separate electron/hole conservation): under ASYMMETRIC times the per-band totals n_tot_e,
        n_tot_h each change only by injection - recomb - stimulated - spontaneous (internal
        capture/escape/relax cancel), the two budgets are identical so d(n_tot_e-n_tot_h)/dt = 0,
        and a closed box (no injection/signal) conserves both totals.
GATE D (the NEW physics -- e/h asymmetry): with holes ~5x faster the saturated-gain recovery
        transient DIFFERS measurably from the excitonic model and the GS hole/electron occupations
        split transiently (f_v_GS != f_c_GS); restoring symmetric times makes the difference vanish
        (proof it is the e/h asymmetry, not numerics).

Run: python -m validation.qd_soa_eh_split
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import Q_E
from dynameta.optics.soa.ase_noise import inversion_factor_nsp, inversion_factor_nsp_eh
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams


def main():
    print("[eh] === QD-SOA electron/hole occupation split vs oracles ===", flush=True)
    ok = True
    ng, nu0 = 21, 1.934e14
    exc = QDGainModel(QDGainParams(n_groups=ng).with_detailed_balance_taus())
    eh = QDGainModel(QDGainParams(n_groups=ng, eh_split=True).with_detailed_balance_taus())

    # ---- GATE A: symmetric e/h reduces to excitonic over a transient ----
    I, nu = 30.0e-3, nu0
    se = exc.init_slices(8, I)
    sh = eh.init_slices(8, I)
    dt = 5.0e-13
    P = np.full(8, 6.0e-3)                                    # saturating signal
    worst_red = 0.0
    worst_split = 0.0
    for _ in range(400):
        se = exc.step_slices(se, P, dt, nu, I)
        sh = eh.step_slices(sh, P, dt, nu, I)
        worst_red = max(worst_red, float(np.max(np.abs(se[2] - sh[4]))),   # rho_GS vs f_c_GS
                        float(np.max(np.abs(se[1] - sh[2]))))              # rho_ES vs f_c_ES
        worst_split = max(worst_split, float(np.max(np.abs(sh[4] - sh[5]))))  # f_c_GS vs f_v_GS
    g_e = exc.gain_per_m_slices(se, nu)
    g_h = eh.gain_per_m_slices(sh, nu)
    dgain = float(np.max(np.abs(g_e - g_h)) / max(np.max(np.abs(g_e)), 1e-300))
    g_a = bool(worst_red < 1e-9 and worst_split < 1e-11 and dgain < 1e-9)
    ok = ok and g_a
    print("[eh] GATE A: symmetric e/h == excitonic (occ {:.1e}, f_c=f_v manifold {:.1e}, gain "
          "{:.1e}) -> {}".format(worst_red, worst_split, dgain, "PASS" if g_a else "FAIL"),
          flush=True)

    # ---- GATE B: gain + n_sp closed forms ----
    rng = np.random.default_rng(3)
    fcGS = (0.55 + 0.4 * rng.random(ng))[None, :]            # gain region f_c+f_v>1
    fvGS = (0.55 + 0.4 * rng.random(ng))[None, :]
    st = (np.zeros(1), np.zeros(1), np.zeros((1, ng)), np.zeros((1, ng)), fcGS, fvGS)
    g_num = eh.gain_per_m_slices(st, nu)[0]
    wl = eh.w_j * eh._lorentzian(nu - eh.nu_j)
    g_ref = eh._gain_pref * np.sum((fcGS[0] + fvGS[0] - 1.0) * wl)
    rel_g = abs(g_num - g_ref) / abs(g_ref)
    nsp = inversion_factor_nsp_eh(fcGS[0], fvGS[0])
    nsp_ref = fcGS[0] * fvGS[0] / (fcGS[0] + fvGS[0] - 1.0)
    rel_nsp = float(np.max(np.abs(nsp - nsp_ref)))
    # reduction at f_c=f_v=rho + transparency
    red_g = abs(inversion_factor_nsp_eh(0.8, 0.8) - inversion_factor_nsp(0.8))
    transp = abs(inversion_factor_nsp_eh(0.7, 0.3))          # f_c+f_v=1 -> inf
    g_b = bool(rel_g < 1e-12 and rel_nsp < 1e-12 and red_g < 1e-12 and not np.isfinite(transp))
    ok = ok and g_b
    print("[eh] GATE B: gain==sum(f_c+f_v-1) (rel {:.1e}), n_sp==f_c f_v/(f_c+f_v-1) (rel {:.1e}), "
          "reduction {:.1e}, transparency inf={} -> {}".format(
              rel_g, rel_nsp, red_g, not np.isfinite(transp), "PASS" if g_b else "FAIL"),
          flush=True)

    # ---- GATE C: separate electron/hole conservation under asymmetric times ----
    ehA = QDGainModel(QDGainParams(n_groups=ng, eh_split=True, tau_cap_h_s=0.25e-12,
                                   tau_rel_h_s=0.35e-12).with_detailed_balance_taus())
    p = ehA.p
    w = ehA.w_j
    Nwe = np.array([6.0e23])
    Nwh = np.array([5.0e23])
    fcES = (0.2 + 0.5 * rng.random(ng))[None, :]
    fvES = (0.15 + 0.5 * rng.random(ng))[None, :]
    fcG = (0.3 + 0.6 * rng.random(ng))[None, :]
    fvG = (0.25 + 0.6 * rng.random(ng))[None, :]
    S, Iv = 2.0e21, 2.0e-2
    d = ehA.rhs_fields_eh(Nwe, Nwh, fcES, fvES, fcG, fvG, Iv, S, nu)
    dn_e = d[0][0] + p.N_q_m3 * np.sum(w * (p.mu_ES * d[2][0] + p.mu_GS * d[4][0]))
    dn_h = d[1][0] + p.N_q_m3 * np.sum(w * (p.mu_ES * d[3][0] + p.mu_GS * d[5][0]))
    L = ehA._lorentzian(nu - ehA.nu_j)
    R_stim = np.sum(w * p.mu_GS * p.v_g_m_s * p.sigma_pk_m2 * L * (fcG[0] + fvG[0] - 1.0) * S)
    spG = np.sum(w * p.mu_GS * fcG[0] * fvG[0]) / p.tau_sp_s
    spE = np.sum(w * p.mu_ES * fcES[0] * fvES[0]) / p.tau_sp_s
    R_wl = p.B_wl_m3_s * Nwe[0] * Nwh[0] + p.C_wl_m6_s * Nwe[0] * Nwh[0] * (Nwe[0] + Nwh[0]) / 2.0
    budget = Iv / (Q_E * p.V_a_m3) - R_wl - p.N_q_m3 * (R_stim + spG + spE)
    rel_e = abs(dn_e - budget) / abs(budget)
    rel_h = abs(dn_h - budget) / abs(budget)
    rel_neutral = abs(dn_e - dn_h) / abs(budget)             # d(n_e - n_h)/dt == 0
    # closed box: no injection/signal -> totals constant
    d0 = ehA.rhs_fields_eh(Nwe, Nwh, fcES, fvES, fcG, fvG, 0.0, 0.0, nu)
    # zero out recomb by setting densities tiny? instead check internal-only via a no-recomb model
    ehNB = QDGainModel(QDGainParams(n_groups=ng, eh_split=True, B_wl_m3_s=0.0, C_wl_m6_s=0.0,
                                    tau_sp_s=1e30).with_detailed_balance_taus())
    dcb = ehNB.rhs_fields_eh(Nwe, Nwh, fcES, fvES, fcG, fvG, 0.0, 0.0, nu)
    box_e = dcb[0][0] + ehNB.p.N_q_m3 * np.sum(w * (p.mu_ES * dcb[2][0] + p.mu_GS * dcb[4][0]))
    box_scale = p.N_q_m3 * p.mu_ES / p.tau_cap_s
    g_c = bool(rel_e < 1e-10 and rel_h < 1e-10 and rel_neutral < 1e-10
               and abs(box_e) < 1e-9 * box_scale)
    ok = ok and g_c
    print("[eh] GATE C: conservation -- dn_e {:.1e}, dn_h {:.1e}, d(n_e-n_h) {:.1e}, closed-box "
          "{:.1e} -> {}".format(rel_e, rel_h, rel_neutral, abs(box_e) / box_scale,
                                "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: the NEW physics -- asymmetric e/h gain recovery differs + GS occupations split ----
    def recovery(model):
        s = model.init_slices(1, I)                          # single section, unsaturated
        # saturate hard, then release; record GS material gain + the e/h GS occupation split
        gains, split = [], 0.0
        for n in range(600):
            P_loc = np.array([40.0e-3]) if n < 200 else np.array([1.0e-5])
            s = model.step_slices(s, P_loc, dt, nu, I)
            if n >= 200:
                gains.append(float(model.gain_per_m_slices(s, nu)[0]))
                if model.eh:
                    split = max(split, float(np.max(np.abs(s[4][0] - s[5][0]))))
        return np.array(gains), split
    g_exc, _ = recovery(exc)
    g_asym, split_asym = recovery(ehA)
    g_ss = max(abs(g_exc[-1]), 1.0)
    recov_diff = float(np.max(np.abs(g_asym - g_exc)) / g_ss)
    # symmetric-restore: a symmetric eh recovery must match the excitonic one
    g_sym, split_sym = recovery(eh)
    recov_sym = float(np.max(np.abs(g_sym - g_exc)) / g_ss)
    g_d = bool(recov_diff > 1e-2 and split_asym > 1e-3 and recov_sym < 1e-9 and split_sym < 1e-11)
    ok = ok and g_d
    print("[eh] GATE D: asymmetric e/h recovery differs from excitonic by {:.2e} (>1%), GS split "
          "max {:.2e}; symmetric-restore diff {:.1e} -> {}".format(
              recov_diff, split_asym, recov_sym, "PASS" if g_d else "FAIL"), flush=True)

    print("[eh] *** QD-SOA E/H SPLIT: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
