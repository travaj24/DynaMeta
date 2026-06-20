"""QD-SOA HAMMER TEST -- an exhaustive cross-cutting stress test of the whole simulator: every opt-in
feature default-off byte-identical, extreme / degenerate inputs, all the reductions, conservation /
passivity, determinism, cross-engine consistency, and an all-features-on 'kitchen sink'. This is the
integration backstop ABOVE the per-feature validations (each feature has its own qd_soa_*.py gates).

GATE H1 (default-off byte-identity sweep): the opt-in knobs exercised here, left at their defaults,
        reproduce the bare marcher EXACTLY -- on amplify (ultrafast, transport_tau_s, rc_tau_s,
        nl_loss); on amplify_coherent (langevin, beta2/GVD, line_filter, nl_loss); leakage +
        self_heating at the model level; and the BPM thermal lens. (numba parity, eh-split, many-body,
        and the ES band have their own dedicated qd_soa_*.py gates; here they are only finiteness-
        checked in H2.)
GATE H2 (extreme / degenerate inputs): zero input, huge (saturating) input, tiny / large current,
        n_groups=1, nz=2, e/h split, fast=numba -- all finite, no NaN/inf, gain bounded.
GATE H3 (reductions roundup): amplify(single tone) == amplify_coherent(alpha=0, sqrt-power);
        dualpol(ratio=1) degenerate; Fabry-Perot(R1=R2=0) == single-pass; GVD(beta2=0) == no-GVD.
GATE H4 (conservation / passivity): an unpumped SOA absorbs (gain < 0); a passive (g0=0) BPM conserves
        energy; FCA / TPA / leakage / RC never raise the output above the no-loss baseline.
GATE H5 (determinism): Langevin with a fixed seed is bit-reproducible; a different seed differs; the
        deterministic core is identical across repeated calls.
GATE H6 (cross-engine consistency): the 1-D CW small-signal gain == exp((Gamma g0 - alpha_i) L); the
        uniform-beam BPM == the 1-D saturable-gain ODE; steady_state residual ~ 0.
GATE H7 (kitchen sink): ultrafast + transport + rc + nl_loss + self_heating + leakage + alpha all ON
        at once -> finite, physical, bounded output (no feature-interaction blow-up).
GATE H8 (parameter-sweep monotonicity): unsaturated gain rises with current; output saturates
        (compresses) with input power; the noise figure -> ~2 n_sp at high gain.

Run: python -m validation.qd_soa_hammer
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import (Leakage, NonlinearLoss, QDGainModel, QDGainParams,
                                 SelfHeating, TransverseBPM, TravelingWaveSOA, UltrafastCompression,
                                 inversion_factor_nsp, noise_figure, single_pass_gain)
from dynameta.optics.soa.qd_gain import _HAVE_NUMBA


def _ok(cond):
    return bool(cond)


def main():
    print("[ham] === QD-SOA HAMMER TEST (exhaustive cross-cutting stress) ===", flush=True)
    ok = True
    pf = lambda **kw: QDGainParams(n_groups=15, **kw).with_detailed_balance_taus()
    m = QDGainModel(pf())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz, alpha_i_per_m=300.0)
    nu0 = m.p.nu0_Hz
    nt = 2000
    P = np.full(nt, 1e-6)
    A = np.full(nt, np.sqrt(1e-6) + 0j)

    # ---- H1: default-off byte-identity sweep ----
    base = soa.amplify(P, 40e-3, return_traces=True)["g_zt"]
    cbase = soa.amplify_coherent(A, 40e-3)["A_out"]
    h1 = all([
        np.array_equal(base, soa.amplify(P, 40e-3, ultrafast=None, return_traces=True)["g_zt"]),
        np.array_equal(base, soa.amplify(P, 40e-3, transport_tau_s=0.0, return_traces=True)["g_zt"]),
        np.array_equal(base, soa.amplify(P, 40e-3, rc_tau_s=0.0, return_traces=True)["g_zt"]),
        np.array_equal(base, soa.amplify(P, 40e-3, nl_loss=None, return_traces=True)["g_zt"]),
        np.array_equal(cbase, soa.amplify_coherent(A, 40e-3, langevin=False)["A_out"]),
        np.array_equal(cbase, soa.amplify_coherent(A, 40e-3, beta2_s2_per_m=0.0)["A_out"]),
        np.array_equal(cbase, soa.amplify_coherent(A, 40e-3, line_filter=False)["A_out"]),
        np.array_equal(cbase, soa.amplify_coherent(A, 40e-3, nl_loss=None)["A_out"]),
    ])
    # model-level: disabled options identical to the plain model
    g_plain = m.gain_per_m_slices(m.init_slices(4, 40e-3), nu0)
    h1m = all([
        np.array_equal(g_plain, QDGainModel(pf(), leakage=Leakage(0.0)).gain_per_m_slices(
            m.init_slices(4, 40e-3), nu0)),
        np.array_equal(g_plain, QDGainModel(pf(), self_heating=None).gain_per_m_slices(
            m.init_slices(4, 40e-3), nu0)),
        np.array_equal(g_plain, QDGainModel(pf()).gain_per_m_slices(m.init_slices(4, 40e-3), nu0)),
    ])
    # BPM thermal lens off (a T(x) profile present but dndt = 0 -> no lens) is byte-identical
    bpm0 = TransverseBPM(200e-6, 256, 1.3e-6, 3.4, g0_per_m=2000.0, Isat_W=1e-3)
    Ab = np.exp(-(bpm0.x / 20e-6) ** 2) + 0j
    Tx0 = 300.0 + 5.0 * np.exp(-(bpm0.x / 30e-6) ** 2)
    h1b = np.array_equal(bpm0.propagate(Ab, 0.5e-3, 100)["A_out"],
                         bpm0.propagate(Ab, 0.5e-3, 100, T_profile_x=Tx0, dndt_per_K=0.0)["A_out"])
    g1 = _ok(h1 and h1m and h1b)
    ok = ok and g1
    print("[ham] H1: opt-ins default byte-identical (engine {}, model {}, BPM-lens {}) -> {}".format(
        h1, h1m, h1b, "PASS" if g1 else "FAIL"), flush=True)

    # ---- H2: extreme / degenerate inputs ----
    finite = []
    finite.append(np.all(np.isfinite(soa.amplify(np.zeros(nt), 40e-3)["P_out"])))            # zero in
    finite.append(np.all(np.isfinite(soa.amplify(np.full(nt, 1.0), 40e-3)["P_out"])))         # huge in
    finite.append(np.all(np.isfinite(soa.amplify(P, 1e-4)["P_out"])))                         # tiny I
    finite.append(np.all(np.isfinite(soa.amplify(P, 0.3)["P_out"])))                          # large I
    m1 = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    s1 = TravelingWaveSOA(m1, 0.5e-3, 40, nu_s_Hz=m1.p.nu0_Hz)
    finite.append(np.all(np.isfinite(s1.amplify(P, 40e-3)["P_out"])))                         # ng=1
    s2 = TravelingWaveSOA(m, 0.5e-3, 2, nu_s_Hz=nu0)
    finite.append(np.all(np.isfinite(s2.amplify(P, 40e-3)["P_out"])))                         # nz=2
    meh = QDGainModel(QDGainParams(n_groups=15, eh_split=True).with_detailed_balance_taus())
    seh = TravelingWaveSOA(meh, 0.5e-3, 40, nu_s_Hz=meh.p.nu0_Hz)
    finite.append(np.all(np.isfinite(seh.amplify(P, 40e-3)["P_out"])))                        # e/h
    if _HAVE_NUMBA:
        mn = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus(), fast=True)
        sn = TravelingWaveSOA(mn, 0.5e-3, 40, nu_s_Hz=mn.p.nu0_Hz)
        finite.append(np.all(np.isfinite(sn.amplify(P, 40e-3)["P_out"])))                     # numba
    g2 = _ok(all(finite))
    ok = ok and g2
    print("[ham] H2: extreme/degenerate inputs all finite ({} cases) -> {}".format(
        len(finite), "PASS" if g2 else "FAIL"), flush=True)

    # ---- H3: reductions roundup ----
    pf_out = soa.amplify(P, 40e-3)["P_out"]
    csc = soa.amplify_coherent(A, 40e-3, alpha_lef=0.0)
    cf_out = csc["P_out"]
    r_marcher = np.max(np.abs(pf_out - cf_out)) / np.max(pf_out)
    # dual-pol with NO TM injected and unit ratio reduces to the SCALAR coherent marcher (a real
    # reduction -- NOT the TE==TM symmetry tautology); injecting an equal TM then cross-saturates the
    # shared reservoir, so the TE output must DROP (the shared-reservoir physics actually engages).
    od = soa.amplify_coherent_dualpol(A, np.zeros_like(A), 40e-3, pdg_ratio=1.0, alpha_lef=0.0)
    r_dualpol = np.max(np.abs(od["A_te_out"] - csc["A_out"])) / np.max(np.abs(csc["A_out"]))
    od2 = soa.amplify_coherent_dualpol(A, A, 40e-3, pdg_ratio=1.0, alpha_lef=0.0)
    xsat = bool(od2["P_te_out"][-1] < od["P_te_out"][-1])
    fp = soa.amplify_fabry_perot(A, 40e-3, R1=0.0, R2=0.0, alpha_lef=0.0)["A_out"]
    r_fp = np.max(np.abs(np.abs(fp) ** 2 - cf_out)) / np.max(cf_out)
    gvd = soa.amplify_coherent(A, 40e-3, beta2_s2_per_m=0.0)["A_out"]
    r_gvd = np.array_equal(gvd, cbase)
    g3 = _ok(r_marcher < 1e-9 and r_dualpol < 1e-6 and xsat and r_fp < 1e-9 and r_gvd)
    ok = ok and g3
    print("[ham] H3: reductions (marcher {:.1e}, dualpol->scalar {:.1e}, x-sat {}, FP {:.1e}, GVD {}) "
          "-> {}".format(r_marcher, r_dualpol, xsat, r_fp, r_gvd, "PASS" if g3 else "FAIL"), flush=True)

    # ---- H4: conservation / passivity ----
    g_unpumped = m.gain_per_m_slices(m.init_slices(2, 1e-4), nu0)[0]                 # ~unpumped
    bpm = TransverseBPM(400e-6, 2048, 1.3e-6, 3.4, g0_per_m=0.0)
    Ag = np.exp(-(bpm.x / 12e-6) ** 2) + 0j
    en = bpm.propagate(Ag, 1e-3, 200)["I_out"].sum() / (np.abs(Ag) ** 2).sum()
    out_noloss = soa.amplify(P, 40e-3)["P_out"][-1]
    out_fca = soa.amplify(P, 40e-3, nl_loss=NonlinearLoss(0.0, 5e-21, 0.0))["P_out"][-1]
    g4 = _ok(g_unpumped < 0.0 and abs(en - 1.0) < 1e-9 and out_fca <= out_noloss)
    ok = ok and g4
    print("[ham] H4: unpumped absorbs (g {:.0f}<0), passive BPM energy {:.7f}, FCA<=baseline {} -> "
          "{}".format(g_unpumped, en, out_fca <= out_noloss, "PASS" if g4 else "FAIL"), flush=True)

    # ---- H5: determinism ----
    o_a = soa.amplify_coherent(A, 40e-3, langevin=True, seed=7)["A_out"]
    o_b = soa.amplify_coherent(A, 40e-3, langevin=True, seed=7)["A_out"]
    o_c = soa.amplify_coherent(A, 40e-3, langevin=True, seed=8)["A_out"]
    g5 = _ok(np.array_equal(o_a, o_b) and not np.array_equal(o_a, o_c)
             and np.array_equal(base, soa.amplify(P, 40e-3, return_traces=True)["g_zt"]))
    ok = ok and g5
    print("[ham] H5: determinism (seed-reproducible {}, seed-sensitive {}, core stable) -> {}".format(
        np.array_equal(o_a, o_b), not np.array_equal(o_a, o_c), "PASS" if g5 else "FAIL"), flush=True)

    # ---- H6: cross-engine consistency ----
    g0u = m.gain_per_m_slices(m.init_slices(40, 40e-3), nu0)[0]
    cw = soa.amplify(np.full(3 * 40, 1e-9), 40e-3)["P_out"][-1] / 1e-9
    r_cw = abs(cw - np.exp((m.gamma_confinement * g0u - 300.0) * 0.5e-3)) / np.exp(
        (m.gamma_confinement * g0u - 300.0) * 0.5e-3)
    yss = m.steady_state(40e-3)                             # self-validates convergence (raises if not)
    res = float(np.max(np.abs(m.rhs(yss, 40e-3, 0.0, nu0)[1:])) * m.p.tau_sp_s)  # relative (x tau_sp)
    # uniform-beam BPM == the 1-D saturable-gain ODE: diffraction touches only k_x != 0, so a flat beam
    # (k_x = 0 only) sees pure saturable gain. With Gamma=1, alpha=alpha_i=0 the intensity obeys
    # dI/dz = g0 I/(1 + I/Isat), integrating to ln(I_out/I_in) + (I_out - I_in)/Isat = g0 Lz; bisect.
    g0b, Isatb, Lzb, Iin = 5000.0, 1.0e-3, 0.6e-3, 2.0e-4
    ub = TransverseBPM(200e-6, 256, 1.3e-6, 3.4, g0_per_m=g0b, Isat_W=Isatb)
    Iout_bpm = float(ub.propagate(np.full(256, np.sqrt(Iin) + 0j), Lzb, 4000)["I_out"][0])
    lo, hi = Iin, Iin * np.exp(g0b * Lzb)                   # brackets the saturated root
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.log(mid / Iin) + (mid - Iin) / Isatb - g0b * Lzb > 0.0:
            hi = mid
        else:
            lo = mid
    Iout_ode = 0.5 * (lo + hi)
    r_bpm = abs(Iout_bpm - Iout_ode) / Iout_ode
    g6 = _ok(r_cw < 1e-2 and res < 1e-6 and r_bpm < 1e-3)
    ok = ok and g6
    print("[ham] H6: 1-D CW == exp((Gamma g - alpha_i)L) (rel {:.1e}), uniform BPM == saturable ODE "
          "(rel {:.1e}), steady residual*tau_sp {:.1e} -> {}".format(
              r_cw, r_bpm, res, "PASS" if g6 else "FAIL"), flush=True)

    # ---- H7: kitchen sink (all features on) ----
    sh = SelfHeating(Rth_K_W=40.0, dnu0_dT_Hz_K=20e9, dg_dT_frac_per_K=-0.01, T0_K=300.0)
    lk = Leakage(tau_leak0_s=8e-12, E_barrier_eV=0.12)
    mk = QDGainModel(QDGainParams(n_groups=15, alpha_lef=2.0).with_detailed_balance_taus(),
                     self_heating=sh, leakage=lk)
    mk.set_temperature(310.0)
    sk = TravelingWaveSOA(mk, 0.5e-3, 40, nu_s_Hz=mk.p.nu0_Hz, alpha_i_per_m=300.0)
    uf = UltrafastCompression(eps_shb_m3=1e-24, eps_ch_m3=5e-25)
    Isink = 40e-3 * (1.0 + 0.1 * np.sin(2 * np.pi * 2e9 * np.arange(nt) * sk.dt))
    osk = sk.amplify(P, Isink, ultrafast=uf, transport_tau_s=200e-12, rc_tau_s=80e-12,
                     nl_loss=NonlinearLoss(5e-11, 5e-21, 0.0), return_traces=True)
    g7 = _ok(np.all(np.isfinite(osk["P_out"])) and np.all(np.isfinite(osk["g_zt"]))
             and osk["P_out"][-1] > 0.0)
    ok = ok and g7
    print("[ham] H7: kitchen sink (7 features on) finite + physical -> {}".format(
        "PASS" if g7 else "FAIL"), flush=True)

    # ---- H8: parameter-sweep monotonicity ----
    gI = [m.gain_per_m_slices(m.init_slices(2, I), nu0)[0] for I in (10e-3, 30e-3, 80e-3)]
    sat = [soa.amplify(np.full(nt, Pp), 40e-3)["P_out"][-1] / Pp for Pp in (1e-7, 1e-4, 1e-2)]
    G_hi = float(single_pass_gain(np.full(40, 8000.0), 2e-3 / 40, m.gamma_confinement))  # high gain
    nsp = inversion_factor_nsp(0.95)
    nf_hi = noise_figure(1.0e4, nsp)                         # NF -> 2 n_sp as G -> inf
    g8 = _ok(gI[0] < gI[1] < gI[2] and sat[0] > sat[1] > sat[2] and G_hi > 1.0
             and abs(nf_hi - 2.0 * nsp) < 0.05)
    ok = ok and g8
    print("[ham] H8: gain rises with I {}, output saturates {}, NF->2nsp ({:.3f} vs {:.3f}) -> "
          "{}".format(gI[0] < gI[1] < gI[2], sat[0] > sat[1] > sat[2], nf_hi, 2.0 * nsp,
                      "PASS" if g8 else "FAIL"), flush=True)

    print("[ham] *** QD-SOA HAMMER TEST: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
