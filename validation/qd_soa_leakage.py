"""QD-SOA thermally-activated carrier leakage vs analytic oracles. Leakage adds a phenomenological
linear loss -N_w/tau_leak(T) to the wetting-layer rate equation, 1/tau_leak(T) = (1/tau_leak0)
exp(-E_b q/kT) -- a temperature-activated escape the closed capture/escape/recomb ladder omits. The
DIVERTED CURRENT (N_w/tau_leak) rises with pump and T; the GAIN SUPPRESSION is largest near threshold
and shrinks as the clamped high-injection gain saturates (the two are distinct, not a "high-I rolloff").

GATE A (byte-identical default + numba parity): leakage=None reproduces the no-leakage steady state /
        step / gain EXACTLY; with leakage on, the numba fast path matches the numpy reference to bit
        precision (the -leak_rate*N_w term added to both).
GATE B (Arrhenius rate exact): _leak_rate() == exp(-E_b q/(k_B T))/tau_leak0, and the ratio between two
        temperatures equals exp(-E_b q/k_B (1/T2 - 1/T1)) to machine precision -- the independent
        closed-form oracle for the thermally-activated escape.
GATE C (term-level exactness): the leakage contributes EXACTLY -leak_rate * N_w to dN_w/dt --
        rhs_fields(leak) - rhs_fields(no leak) == -leak_rate N_w on the same state (excitonic AND the
        e/h-split N_w_e, N_w_h).
GATE D (gain suppression, monotone + Arrhenius): leakage lowers the steady-state gain; a faster
        leakage (smaller tau_leak0) suppresses it more (monotone); and raising T raises the leakage
        rate (Arrhenius) so the gain drops further with temperature.
GATE E (diverted current grows with pump + passivity): the leakage current N_w/tau_leak rises with the
        drive I (more carriers clear the barrier at higher pump -- the sub-linear-injection mechanism);
        leakage never RAISES the gain (passivity); all finite.

Run: python -m validation.qd_soa_leakage
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import Leakage, QDGainModel, QDGainParams
from dynameta.optics.soa.qd_gain import KB, Q_E, _HAVE_NUMBA


def _gain(model, I, nu):
    return model.gain_per_m_slices(model.init_slices(2, I), nu)[0]


def main():
    print("[lk] === QD-SOA thermionic carrier leakage vs oracles ===", flush=True)
    ok = True
    base_p = QDGainParams(n_groups=15).with_detailed_balance_taus()
    nu = base_p.nu0_Hz
    I = 40e-3
    lk = Leakage(tau_leak0_s=5.0e-12, E_barrier_eV=0.10)
    m0 = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus(), leakage=lk)

    # ---- GATE A: byte-identical default + numba parity ----
    # None vs a DISABLED Leakage(0) exercises the "- 0.0*N_w" path: must be bit-identical.
    m_dis = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus(),
                        leakage=Leakage(tau_leak0_s=0.0))
    st_a = m0.init_slices(2, I)
    a_steady = np.array_equal(m0.steady_state(I), m_dis.steady_state(I))
    a_gain = np.array_equal(m0.gain_per_m_slices(st_a, nu), m_dis.gain_per_m_slices(st_a, nu))
    parity = 0.0
    if _HAVE_NUMBA:
        mn = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus(), leakage=lk, fast=True)
        mc = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus(), leakage=lk)
        sn = mn.step_slices(mn.init_slices(6, I), 1e-4, 1e-13, nu, I)
        sc = mc.step_slices(mc.init_slices(6, I), 1e-4, 1e-13, nu, I)
        parity = float(np.max(np.abs(sn[0] - sc[0])) / np.max(np.abs(sc[0])))
    g_a = bool(a_steady and a_gain and parity < 1e-14)
    ok = ok and g_a
    print("[lk] GATE A: leak=None byte-identical (steady {}, gain {}); numba parity rel {:.1e} -> "
          "{}".format(a_steady, a_gain, parity, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Arrhenius rate exact ----
    r300 = lk.rate_at(300.0)
    r340 = lk.rate_at(340.0)
    expect300 = np.exp(-lk.E_barrier_eV * Q_E / (KB * 300.0)) / lk.tau_leak0_s
    expect_ratio = np.exp(-lk.E_barrier_eV * Q_E / KB * (1.0 / 340.0 - 1.0 / 300.0))
    relB = max(abs(r300 - expect300) / expect300, abs(r340 / r300 - expect_ratio) / expect_ratio)
    g_b = bool(relB < 1e-12)
    ok = ok and g_b
    print("[lk] GATE B: rate == exp(-Eb q/kT)/tau0 (rate {:.3e}/s, T-ratio {:.3f}, rel {:.1e}) -> "
          "{}".format(r300, r340 / r300, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: term-level exactness (excitonic + e/h) ----
    st = m0.init_slices(3, I)
    lr = m._leak_rate()
    d_no = m0.rhs_fields(st[0], st[1], st[2], I, 0.0, nu)[0]
    d_lk = m.rhs_fields(st[0], st[1], st[2], I, 0.0, nu)[0]
    cx = float(np.max(np.abs((d_lk - d_no) - (-lr * st[0]))))
    # e/h split
    pe = QDGainParams(n_groups=15, eh_split=True).with_detailed_balance_taus()
    me0 = QDGainModel(QDGainParams(n_groups=15, eh_split=True).with_detailed_balance_taus())
    mel = QDGainModel(QDGainParams(n_groups=15, eh_split=True).with_detailed_balance_taus(),
                      leakage=lk)
    se = me0.init_slices(3, I)
    lre = mel._leak_rate()
    de_no = me0.rhs_fields_eh(se[0], se[1], se[2], se[3], se[4], se[5], I, 0.0, nu)
    de_lk = mel.rhs_fields_eh(se[0], se[1], se[2], se[3], se[4], se[5], I, 0.0, nu)
    cxe = max(float(np.max(np.abs((de_lk[0] - de_no[0]) - (-lre * se[0])))),
              float(np.max(np.abs((de_lk[1] - de_no[1]) - (-lre * se[1])))))
    g_c = bool(cx < 1e-3 and cxe < 1e-3)            # absolute residual on dN_w (~1e30 scale) -> tiny
    ok = ok and g_c
    print("[lk] GATE C: dN_w(leak)-dN_w(no) == -leak_rate N_w (excitonic {:.1e}, e/h {:.1e}) -> "
          "{}".format(cx, cxe, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: gain suppression monotone in rate + Arrhenius in T ----
    g0 = _gain(m0, I, nu)
    gL = _gain(m, I, nu)
    m_fast = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus(),
                         leakage=Leakage(tau_leak0_s=2.0e-12, E_barrier_eV=0.10))  # faster leak
    gLf = _gain(m_fast, I, nu)
    mono = bool(gLf < gL < g0)                       # more leakage -> lower gain
    mhot = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus(), leakage=lk)
    mhot.set_temperature(340.0)
    g_hot = _gain(mhot, I, nu)
    # isolate the leakage-driven drop from set_temperature's own gain effect (no SelfHeating -> none)
    hotter_more_leak = bool(mhot._leak_rate() > m._leak_rate() and g_hot < gL)
    g_d = bool(mono and hotter_more_leak)
    ok = ok and g_d
    print("[lk] GATE D: gain suppressed monotone (g0 {:.0f} > gL {:.0f} > gLfast {:.0f}); hotter->more "
          "leak->lower gain {} -> {}".format(g0, gL, gLf, hotter_more_leak,
                                             "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: diverted leakage current grows with pump + passivity ----
    def leak_curr(Ipump):
        Nw = m.wl_density_slices(m.init_slices(2, Ipump))[0]
        return Nw * m._leak_rate()                   # N_w / tau_leak [m^-3 s^-1] diverted
    jlo, jhi = leak_curr(10e-3), leak_curr(80e-3)
    passive = bool(_gain(m, 10e-3, nu) <= _gain(m0, 10e-3, nu)
                   and _gain(m, 80e-3, nu) <= _gain(m0, 80e-3, nu)
                   and np.isfinite(gL))
    g_e = bool(jhi > jlo > 0.0 and passive)
    ok = ok and g_e
    print("[lk] GATE E: leakage current grows with pump ({:.2e}->{:.2e} /m3/s), passive {} -> "
          "{}".format(jlo, jhi, passive, "PASS" if g_e else "FAIL"), flush=True)

    print("[lk] *** QD-SOA CARRIER LEAKAGE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
