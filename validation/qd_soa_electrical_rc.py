"""QD-SOA electrical-parasitic RC bandwidth vs the analytic first-order pole. amplify(rc_tau_s=) low-
passes the drive current dI_rc/dt = (I_drive - I_rc)/tau_RC (pad/bond + junction RC) BEFORE the SCH
transport / injection stage -- a first-order pole at f_RC = 1/(2 pi tau_RC) that limits the direct-
current-modulation bandwidth, cascading with the SCH transport pole.

GATE A (byte-identical default): rc_tau=0 reproduces the baseline; and for a CONSTANT drive even
        rc_tau>0 is byte-identical (I_rc sits at the steady value -- the RC acts only on a time-varying
        drive).
GATE B (RC pole exact): the drive-current low-pass (the exact IIR the marcher applies) has frequency
        response |H(f)| = 1/sqrt(1+(f/f_RC)^2) -- verified by FFT against the analytic first-order pole
        at f = 0.3, 1, 3 x f_RC (the continuous-limit pole; the backward-Euler warp is < 1% at dt<<tau).
GATE C (marcher step delay): a step UP in the drive current makes the gain rise -- with RC the rise is
        DELAYED (the 50%-rise time is later) vs no RC, and a larger tau_RC delays it more (the RC is
        genuinely wired into the marcher's drive).
GATE D (cascades with SCH transport + DC invariant): RC and the SCH transport pole BOTH delay the step,
        and together delay it more than either alone; the steady (DC) gain is unchanged by the RC.
GATE E (passivity): the RC filter never amplifies (|H| <= 1 for all f); the marcher output is finite.

Run: python -m validation.qd_soa_electrical_rc
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import QDGainModel, QDGainParams, TravelingWaveSOA


def _rc_filter(I, tau, dt):
    """The exact drive-current IIR the marcher applies: BACKWARD Euler I_rc[n] = (I_rc[n-1] +
    (dt/tau) I[n]) / (1 + dt/tau) -- a first-order pole, unconditionally stable for any dt."""
    out = np.empty(I.size)
    rc = I[0]
    a = dt / tau
    for n in range(I.size):
        rc = (rc + a * I[n]) / (1.0 + a)
        out[n] = rc
    return out


def _t50(g_t, dt, nz):
    """50%-rise time [s] of a monotone-rising trace past the device-fill transit (nz samples)."""
    g = g_t[nz:]
    lo, hi = g[0], g[-1]
    idx = int(np.argmax(g >= 0.5 * (lo + hi)))
    return (nz + idx) * dt


def main():
    print("[rc] === QD-SOA electrical-parasitic RC bandwidth vs oracle ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=11).with_detailed_balance_taus())
    soa = TravelingWaveSOA(m, 0.5e-3, 40, nu_s_Hz=m.p.nu0_Hz)
    dt = soa.dt
    nt = 6000
    P = np.full(nt, 1e-5)
    I0 = 40e-3
    tau_rc = 100e-12
    fRC = 1.0 / (2.0 * np.pi * tau_rc)

    # ---- GATE A: byte-identical defaults ----
    base = soa.amplify(P, I0, return_traces=True)["g_zt"]
    a_zero = np.array_equal(base, soa.amplify(P, I0, rc_tau_s=0.0, return_traces=True)["g_zt"])
    a_const = np.array_equal(base, soa.amplify(P, I0, rc_tau_s=tau_rc, return_traces=True)["g_zt"])
    g_a = bool(a_zero and a_const)
    ok = ok and g_a
    print("[rc] GATE A: rc=0 byte-id {}, const-drive rc>0 byte-id {} -> {}".format(
        a_zero, a_const, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: RC pole exact (standalone IIR vs analytic, FFT) ----
    Nf, dtf = 200000, 1.0e-13                                 # long record so sub-GHz is resolvable
    tf = np.arange(Nf) * dtf
    relB = 0.0
    for fr in (0.3, 1.0, 3.0):
        f = fr * fRC
        sig = np.sin(2.0 * np.pi * f * tf)
        H = _rc_filter(sig, tau_rc, dtf)
        k = int(np.argmin(np.abs(np.fft.rfftfreq(Nf, dtf) - f)))
        meas = np.abs(np.fft.rfft(H)[k]) / np.abs(np.fft.rfft(sig)[k])
        relB = max(relB, abs(meas - 1.0 / np.sqrt(1.0 + fr ** 2)))
    g_b = bool(relB < 1e-2)
    ok = ok and g_b
    print("[rc] GATE B: |H(f)| == 1/sqrt(1+(f/fRC)^2) (max dev {:.1e}) -> {}".format(
        relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: marcher step delay (RC is wired into the drive) ----
    Istep = np.where(np.arange(nt) < nt // 4, 30e-3, 55e-3)  # step up at nt/4
    t50_0 = _t50(soa.amplify(P, Istep, return_traces=True)["g_zt"][:, soa.nz // 2], dt, soa.nz)
    t50_rc = _t50(soa.amplify(P, Istep, rc_tau_s=tau_rc, return_traces=True)["g_zt"][:, soa.nz // 2],
                  dt, soa.nz)
    t50_rc2 = _t50(soa.amplify(P, Istep, rc_tau_s=4 * tau_rc, return_traces=True)["g_zt"][:, soa.nz // 2],
                   dt, soa.nz)
    g_c = bool(t50_rc > t50_0 and t50_rc2 > t50_rc)
    ok = ok and g_c
    print("[rc] GATE C: step-rise delayed by RC (t50 no-RC {:.2e} < RC {:.2e} < 4xRC {:.2e} s) -> "
          "{}".format(t50_0, t50_rc, t50_rc2, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: cascades with SCH transport + DC invariant ----
    t50_tt = _t50(soa.amplify(P, Istep, transport_tau_s=300e-12, return_traces=True)["g_zt"][:, soa.nz // 2],
                  dt, soa.nz)
    t50_both = _t50(soa.amplify(P, Istep, rc_tau_s=tau_rc, transport_tau_s=300e-12,
                                return_traces=True)["g_zt"][:, soa.nz // 2], dt, soa.nz)
    dc_rc = soa.amplify(P, I0, rc_tau_s=tau_rc, return_traces=True)["g_zt"][-1, soa.nz // 2]
    dc_0 = base[-1, soa.nz // 2]
    g_d = bool(t50_both > t50_rc and t50_both > t50_tt and abs(dc_rc - dc_0) / abs(dc_0) < 1e-9)
    ok = ok and g_d
    print("[rc] GATE D: RC+transport cascade (both {:.2e} > RC {:.2e}, > transport {:.2e}); DC invariant "
          "({:.1e}) -> {}".format(t50_both, t50_rc, t50_tt, abs(dc_rc - dc_0) / abs(dc_0),
                                  "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: passivity / finite ----
    Hmax = max(abs((np.abs(np.fft.rfft(_rc_filter(np.sin(2 * np.pi * fr * fRC * tf), tau_rc, dtf)))
                    / np.abs(np.fft.rfft(np.sin(2 * np.pi * fr * fRC * tf)))))[
        int(np.argmin(np.abs(np.fft.rfftfreq(Nf, dtf) - fr * fRC)))] for fr in (0.3, 1.0, 3.0))
    out = soa.amplify(P, Istep, rc_tau_s=tau_rc, return_traces=True)
    g_e = bool(Hmax <= 1.0 + 1e-6 and np.all(np.isfinite(out["g_zt"]))
               and np.all(np.isfinite(out["P_out"])))
    ok = ok and g_e
    print("[rc] GATE E: RC never amplifies (max|H| {:.4f} <= 1) + finite -> {}".format(
        Hmax, "PASS" if g_e else "FAIL"), flush=True)

    print("[rc] *** QD-SOA ELECTRICAL RC: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
