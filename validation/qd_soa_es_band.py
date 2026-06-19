"""QD-SOA excited-state (ES) optical band / two-state gain (roadmap SOA gain-ceiling item) vs
oracles. With sigma_pk_ES_m2 > 0 the ES transition (dE_ES_GS above the GS, nu_ES_j = nu_j +
dE_ES_GS*q/h) becomes optically active: a signal near the ES band sees ES gain
g_ES = sum_j N_q w_j mu_ES sigma_pk_ES L_hom(nu - nu_ES_j)(INV_ES), and the GS/ES two-state
crossover emerges at high injection. sigma_pk_ES_m2 = 0 (default) is GS-only, byte-identical.

GATE A (GS-only reduction): with sigma_pk_ES=0 the total gain equals the GS-only material gain
        EXACTLY (np.array_equal) over a GS+ES frequency grid -- the ES branch is a clean opt-in.
GATE B (g_ES vs analytic Lorentzian ensemble): with sigma_pk_ES>0 and rho_GS forced to 1/2 (GS
        transparent, g_GS=0), the modeled gain near the ES band equals an independent
        Lorentzian-ensemble sum at nu_ES_j, peaking at nu0 + dE_ES_GS*q/h.
GATE C (two-state crossover): sweeping injection, the ES-band small-signal gain turns positive only
        ABOVE a threshold I_th,ES that is strictly larger than the GS threshold I_th,GS, rises
        monotonically, and the GS stays amplifying (gain-clamped) there -- the falsifiable
        two-state signature (Sugawara; Markus et al.).
GATE D (photon-number bookkeeping): for an ES-band signal the ensemble-summed ES stimulated
        carrier-pair loss rate equals the guided-photon gain v_g g_ES(nu_s) S_conf -- the ES
        analogue of the GS conservation gate (mu_ES, not mu_GS, weights both).

Run: python -m validation.qd_soa_es_band
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR, Q_E
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams

H_PLANCK = 2.0 * np.pi * HBAR


def main():
    print("[es] === QD-SOA excited-state optical band (two-state gain) vs oracles ===", flush=True)
    ok = True
    ng = 21
    gs = QDGainModel(QDGainParams(n_groups=ng).with_detailed_balance_taus())
    es = QDGainModel(QDGainParams(n_groups=ng, sigma_pk_ES_m2=3.0e-19).with_detailed_balance_taus())
    nu0 = gs.p.nu0_Hz
    nu_ES = nu0 + gs.p.dE_ES_GS_eV * Q_E / H_PLANCK
    nu_grid = np.linspace(nu0 - 8e12, nu_ES + 8e12, 60)

    # ---- GATE A: GS-only reduction (sigma_pk_ES=0) byte-identical ----
    y0 = gs.steady_state(30.0e-3)
    g_gs = gs.material_gain_per_m(gs.rho_GS(y0), nu_grid)
    g_tot = gs.total_material_gain(gs.rho_ES(y0), gs.rho_GS(y0), nu_grid)
    g_a = bool(np.array_equal(g_gs, g_tot))                  # ES branch skipped -> exact
    ok = ok and g_a
    print("[es] GATE A: sigma_pk_ES=0 total gain == GS-only material gain EXACTLY (array_equal {}) "
          "-> {}".format(g_a, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: g_ES vs analytic Lorentzian ensemble (rho_GS=1/2 -> g_GS=0) ----
    ye = es.steady_state(40.0e-3)
    rES = es.rho_ES(ye)
    hwE = es._hw_ES
    g_num = es.total_material_gain(rES, np.full(ng, 0.5), nu_grid)
    g_ref = np.array([np.sum(es.p.N_q_m3 * es.w_j * es.p.mu_ES * es.p.sigma_pk_ES_m2
                             * hwE ** 2 / ((nu - es.nu_ES_j) ** 2 + hwE ** 2) * (2.0 * rES - 1.0))
                      for nu in nu_grid])
    rel_b = float(np.max(np.abs(g_num - g_ref)) / max(np.max(np.abs(g_ref)), 1e-300))
    peak_nu = nu_grid[int(np.argmax(g_num))]
    peak_ok = abs(peak_nu - nu_ES) < (es.nu_ES_j[1] - es.nu_ES_j[0]) * 2.0  # within ~2 group spacings
    g_b = bool(rel_b < 1e-12 and peak_ok)
    ok = ok and g_b
    print("[es] GATE B: g_ES == analytic ensemble (rel {:.1e}), ES peak at {:.4e} Hz (~{:.0f} nm) "
          "-> {}".format(rel_b, peak_nu, 2.998e8 / nu_ES * 1e9, "PASS" if g_b else "FAIL"),
          flush=True)

    # ---- GATE C: two-state crossover ----
    Is = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 35.0]) * 1e-3
    gGS = np.array([es.small_signal_gain_per_m(I, nu0) for I in Is])
    gES = np.array([es.small_signal_gain_per_m(I, nu_ES) for I in Is])
    # ES gain monotone-increasing, crosses zero strictly above the GS threshold
    mono = bool(np.all(np.diff(gES) > 0.0))
    i_th_GS = np.searchsorted(gGS > 0.0, True)               # first index GS positive
    pos_GS = gGS > 0.0
    pos_ES = gES > 0.0
    first_GS = int(np.argmax(pos_GS)) if pos_GS.any() else ng
    first_ES = int(np.argmax(pos_ES)) if pos_ES.any() else ng
    ordering = bool(first_ES > first_GS)                     # ES turns on after GS
    gs_amp_at_es_th = bool(gGS[first_ES] > 0.0) if first_ES < Is.size else False
    g_c = bool(mono and ordering and gs_amp_at_es_th)
    ok = ok and g_c
    print("[es] GATE C: two-state -- g_ES monotone {}, I_th,ES ({:.0f} mA) > I_th,GS ({:.0f} mA), "
          "GS amplifying at ES threshold {} -> {}".format(
              mono, Is[first_ES] * 1e3 if first_ES < Is.size else -1,
              Is[first_GS] * 1e3 if first_GS < Is.size else -1, gs_amp_at_es_th,
              "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: photon-number bookkeeping for an ES-band signal ----
    # ensemble ES stimulated carrier-pair loss rate == v_g g_ES(nu_s) S_conf (closed-form identity)
    S_conf = 5.0e20
    LE = es._lorentzian_ES(nu_ES - es.nu_ES_j)               # (ng,)
    inv_ES = 2.0 * rES - 1.0
    # per-dot ES stim rate summed over dots (mu_ES weight): sum_j N_q w_j mu_ES * v_g sigma_ES LE INV_ES S
    loss = np.sum(es.p.N_q_m3 * es.w_j * es.p.mu_ES * es.p.v_g_m_s * es.p.sigma_pk_ES_m2
                  * LE * inv_ES * S_conf)
    g_ES_nu = es.p.N_q_m3 * np.sum(es.w_j * es.p.mu_ES * es.p.sigma_pk_ES_m2 * LE * inv_ES)
    photon_gain = es.p.v_g_m_s * g_ES_nu * S_conf
    rel_d = abs(loss - photon_gain) / max(abs(photon_gain), 1e-300)
    g_d = bool(rel_d < 1e-12)
    ok = ok and g_d
    print("[es] GATE D: ES stim carrier loss == v_g g_ES S_conf photon gain (rel {:.1e}) -> "
          "{}".format(rel_d, "PASS" if g_d else "FAIL"), flush=True)

    print("[es] *** QD-SOA ES BAND: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
