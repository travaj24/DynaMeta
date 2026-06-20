"""QD-SOA two-photon absorption (TPA) + dynamic free-carrier absorption (FCA) vs analytic oracles.
NonlinearLoss adds, per slice, alpha_nl = sigma_fca N_w + (beta/A_eff) P to the marcher loss, so the
internal loss becomes intensity- AND carrier-dependent rather than the fixed alpha_i.

GATE A (byte-identical default): nl_loss=None and nl_loss=NonlinearLoss(0,0,0) reproduce the baseline
        amplify / amplify_coherent EXACTLY (opt-in, default-off).
GATE B (FCA exact reduction): weak (unsaturated) CW signal -> the FCA loss is a constant coefficient
        sigma_fca N_w, so P_out(FCA)/P_out(off) == exp(-sigma_fca N_w L) with N_w the unsaturated
        steady-state WL density (the marcher is exact for a constant coefficient).
GATE C (TPA passive Bernoulli): at TRANSPARENCY (g=0, so the strong signal cannot deplete carriers)
        the field obeys dP/dz = -(alpha_i + (beta/A_eff) P) P, whose closed form is the Bernoulli
        P(L) = a P0 e^{aL}/(a + b P0(e^{aL}-1)), a=-alpha_i, b=beta/A_eff -- the marcher converges to
        it at 1st order (error halves as nz doubles). CAVEAT: TPA is oracle-checked only at
        transparency (carriers frozen, where the closed form is valid); its interplay with a SATURATING
        gain runs through the same marcher path but is not separately oracle-checked here, and TPA-
        generated carriers are out of scope (see NonlinearLoss SCOPE).
GATE D (TPA self-limiting): TPA is intensity-dependent, so at transparency the transmission
        T = P_out/P_in DECREASES monotonically with input power (the TPA-limited regime) and sits below
        the linear (TPA-off) transmission.
GATE E (dynamic FCA + wiring consistency + passivity): the FCA loss GROWS with pump current (more pump
        -> higher N_w -> more loss: the loss is dynamic, not fixed); a WIRING-consistency check that the
        coherent marcher's 0.5*-on-amplitude path (single real tone, alpha=0, fed sqrt(P)) reproduces
        the power-on-intensity path (same formula in two representations -- it pins the 0.5 factor and
        the |A|^2 argument, not an independent oracle); and no nl_loss ever increases the output
        (passivity), all finite.

Run: python -m validation.qd_soa_nonlinear_loss
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import (NonlinearLoss, QDGainModel, QDGainParams, TravelingWaveSOA)


def _transparency_current(m, nu):
    """Bisect the injection current for g(nu) = 0 (transparency)."""
    def g_of_I(I):
        return m.gain_per_m_slices(m.init_slices(2, I), nu)[0]
    lo, hi = 1e-4, 40e-3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g_of_I(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    print("[nl] === QD-SOA TPA + dynamic FCA vs oracles ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    nu = m.p.nu0_Hz
    gam = m.gamma_confinement
    A_eff = m.p.A_mode_m2
    L, nz, I = 1.0e-3, 200, 40e-3
    soa = TravelingWaveSOA(m, L, nz, nu_s_Hz=nu, alpha_i_per_m=300.0)
    nt = 3 * nz

    # ---- GATE A: byte-identical default ----
    Pin = np.full(nt, 1e-7)
    base = soa.amplify(Pin, I)["P_out"]
    a_none = np.array_equal(base, soa.amplify(Pin, I, nl_loss=None)["P_out"])
    a_zero = np.array_equal(base, soa.amplify(Pin, I, nl_loss=NonlinearLoss(0.0, 0.0, 0.0))["P_out"])
    Ac = np.full(nt, 1e-7 + 0j)
    cbase = soa.amplify_coherent(Ac, I, alpha_lef=0.0)["A_out"]
    a_coh = np.array_equal(cbase, soa.amplify_coherent(Ac, I, alpha_lef=0.0, nl_loss=None)["A_out"])
    g_a = bool(a_none and a_zero and a_coh)
    ok = ok and g_a
    print("[nl] GATE A: nl_loss None/zeros byte-identical (amplify {}/{}, coherent {}) -> {}".format(
        a_none, a_zero, a_coh, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: FCA exact reduction ----
    st0 = m.init_slices(nz, I)
    Nw = m.wl_density_slices(st0)[0]
    sig = 5.0e-21
    fca = soa.amplify(Pin, I, nl_loss=NonlinearLoss(0.0, sig, 0.0))["P_out"][-1]
    ratio = fca / base[-1]
    expect = np.exp(-sig * Nw * L)
    relB = abs(ratio - expect) / expect
    g_b = bool(relB < 1e-6)
    ok = ok and g_b
    print("[nl] GATE B: FCA ratio == exp(-sigma Nw L) (meas {:.4e} vs {:.4e}, rel {:.1e}) -> {}".format(
        ratio, expect, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: TPA passive Bernoulli (transparency) ----
    I_tr = _transparency_current(m, nu)
    alpha_i = 100.0
    beta = 8.0e-11
    b = beta / A_eff
    a = -alpha_i
    P0 = 0.5
    P_an = a * P0 * np.exp(a * L) / (a + b * P0 * (np.exp(a * L) - 1.0))
    rels = []
    for nzc in (100, 200, 400):
        s = TravelingWaveSOA(m, L, nzc, nu_s_Hz=nu, alpha_i_per_m=alpha_i)
        out = s.amplify(np.full(3 * nzc, P0), I_tr, nl_loss=NonlinearLoss(beta, 0.0, A_eff),
                        return_traces=True)
        rels.append(abs(out["P_out"][-1] - P_an) / P_an)
    gmax = float(np.max(np.abs(out["g_zt"][-1])))           # transparency held over the run
    rate = rels[1] / rels[2]                                # ~2 for 1st-order convergence
    g_c = bool(rels[1] < 1e-3 and gmax < 1.0 and 1.7 < rate < 2.3)
    ok = ok and g_c
    print("[nl] GATE C: TPA == Bernoulli (rel nz=200 {:.2e}, conv rate {:.2f}, g~0 {:.1e}) -> {}".format(
        rels[1], rate, gmax, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: TPA self-limiting (intensity-dependent) ----
    sd = TravelingWaveSOA(m, L, nz, nu_s_Hz=nu, alpha_i_per_m=alpha_i)
    Ts = []
    for Pp in (1e-3, 0.1, 0.5, 2.0):
        po = sd.amplify(np.full(nt, Pp), I_tr, nl_loss=NonlinearLoss(beta, 0.0, A_eff))["P_out"][-1]
        Ts.append(po / Pp)
    T_lin = sd.amplify(np.full(nt, 2.0), I_tr)["P_out"][-1] / 2.0   # TPA off, same power
    mono = bool(np.all(np.diff(Ts) < 0))                   # transmission falls as power rises
    g_d = bool(mono and Ts[-1] < T_lin)
    ok = ok and g_d
    print("[nl] GATE D: TPA self-limiting (T {:.3f}->{:.3f} falling {}, < linear T {:.3f}) -> {}".format(
        Ts[0], Ts[-1], mono, T_lin, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: dynamic FCA grows with pump + cross-marcher + passivity ----
    def fca_loss_dB(Ipump):
        on = soa.amplify(Pin, Ipump, nl_loss=NonlinearLoss(0.0, sig, 0.0))["P_out"][-1]
        off = soa.amplify(Pin, Ipump)["P_out"][-1]
        return -10.0 * np.log10(on / off)                  # extra loss from FCA [dB]
    loss_lo, loss_hi = fca_loss_dB(10e-3), fca_loss_dB(60e-3)
    dynamic = bool(loss_hi > loss_lo > 0.0)                # more pump -> higher N_w -> more FCA
    # coherent (single real tone, alpha=0) with FCA == power marcher with FCA. NB amplify takes POWER
    # P_in; amplify_coherent takes the field AMPLITUDE A_in, so feed sqrt(P) for a matched operating
    # point (|A_in|^2 == P_in); then |A_out|^2 must equal P_out (rel ~ exp rounding).
    Acf = np.full(nt, np.sqrt(1e-7) + 0j)
    pf = soa.amplify(Pin, I, nl_loss=NonlinearLoss(0.0, sig, 0.0))["P_out"]
    cf = soa.amplify_coherent(Acf, I, alpha_lef=0.0, nl_loss=NonlinearLoss(0.0, sig, 0.0))["P_out"]
    cross = float(np.max(np.abs(pf - cf)) / np.max(pf))    # relative
    # passivity: TPA or FCA never increases the output
    passive = bool(fca <= base[-1] and Ts[-1] <= T_lin and np.all(np.isfinite(pf)))
    g_e = bool(dynamic and cross < 1e-9 and passive)
    ok = ok and g_e
    print("[nl] GATE E: FCA grows with pump ({:.3f}->{:.3f} dB), coh-wiring==power ({:.1e}), passive "
          "{} -> {}".format(loss_lo, loss_hi, cross, passive, "PASS" if g_e else "FAIL"), flush=True)

    print("[nl] *** QD-SOA NONLINEAR LOSS (TPA + FCA): {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
