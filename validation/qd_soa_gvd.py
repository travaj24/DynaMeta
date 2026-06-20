"""QD-SOA background group-velocity dispersion (GVD) vs analytic oracles. amplify_coherent gains a
beta2_s2_per_m argument that applies the broadband (waveguide) dispersion d2 beta/d omega^2 as a
SYMMETRIC device-scale split D(L/2) . marcher . D(L/2), where D is the EXACT unitary spectral phase
exp(+0.5j beta2 omega^2 (L/2)) on the full retarded-time waveform. Each tone at nu_s + f then carries
exp(+i (beta2/2)(2 pi f)^2 L). This is the NON-resonant waveguide index, distinct from the resonant
gain-line dispersion of the Maxwell-Bloch line filter. The device-scale split is EXACT in the linear
(passive / CW) limit; when beta2 and gain are both active it is an uncontrolled (single step, no
z-refinement) approximation of the distributed coupling -- the leading dispersive phase is captured,
the z-resolved running FWM phase-matching is not (these gates verify the linear limit, where the
oracles are analytic; a per-step dispersion of the spatial node array is invalid -- that array is a
snapshot along z, not a fixed retarded-time window, so dispersing the shifting window leaks energy).

GATE A (reduction): beta2 = 0 -> amplify_coherent is byte-identical to the no-dispersion engine.
GATE B (operator assembly + per-tone phase): gain-free, the split == a single full-L spectral
        dispersion of the marched field (machine exact), and a windowed CW tone picks up exactly
        phi = (beta2/2)(2 pi f)^2 L for both signs of beta2.
GATE C (Gaussian broadening, NLSE oracle): a transform-limited Gaussian (T0) broadens to
        T1 = T0 sqrt(1 + (L/L_D)^2), L_D = T0^2/|beta2| (Agrawal, Nonlinear Fiber Optics ch.3) --
        the independent textbook reference, measured by the second moment of |A_out|^2.
GATE D (chirp SIGN + magnitude): the dispersed Gaussian acquires a quadratic temporal phase
        phi(T) = C T^2, C = -(beta2 L)/(2(T0^4 + (beta2 L)^2)); the SIGN flips with sgn(beta2) (an
        independent physical sign oracle the per-tone phase magnitude alone cannot pin).
GATE E (unitarity): pure dispersion (no gain) conserves the pulse energy integral |A|^2 dt -- the
        spectral phase has |.| = 1, so the split is loss-free and unconditionally stable.

Run: python -m validation.qd_soa_gvd
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.traveling_wave import TravelingWaveSOA, TwoLevelSaturableGain


def _passive(v_g=8.5e7, nu0=1.934e14):
    """A gain-free slab (g0 = 0): amp = 1, pure advection + dispersion -- the clean GVD test bed."""
    return TwoLevelSaturableGain(g0_per_m=0.0, tau_c_s=1.0e-9, E_sat_J=1.0e-12, v_g_m_s=v_g,
                                 nu0_Hz=nu0, alpha_lef=0.0)


def _rms(t, w):
    """Second-moment RMS width of a weighted distribution (weights w >= 0)."""
    W = np.sum(w)
    m1 = np.sum(t * w) / W
    m2 = np.sum(t * t * w) / W
    return float(np.sqrt(max(m2 - m1 * m1, 0.0)))


def main():
    print("[gv] === QD-SOA background group-velocity dispersion vs analytic oracles ===", flush=True)
    ok = True
    L, nz = 1.0e-3, 256
    eng = TravelingWaveSOA(_passive(), L, nz)
    dt, W = eng.dt, nz * eng.dt                              # node-window (transit) duration
    nt = 4 * nz
    tgrid = np.arange(nt) * dt
    tc = (nt // 2) * dt

    # ---- GATE A: reduction (beta2 = 0 byte-identical) ----
    A_probe = np.exp(-((tgrid - tc) ** 2) / (2.0 * (W / 8.0) ** 2)).astype(np.complex128)
    base = eng.amplify_coherent(A_probe, 0.0)["A_out"]
    z0 = eng.amplify_coherent(A_probe, 0.0, beta2_s2_per_m=0.0)["A_out"]
    g_a = bool(np.array_equal(base, z0))
    ok = ok and g_a
    print("[gv] GATE A: beta2=0 == no-dispersion engine (byte-identical {}) -> {}".format(
        g_a, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: operator assembly (machine) + per-tone analytic phase (both signs) ----
    beta2B = 5.0e-22
    # (B1) the split must assemble to exactly D(L) of the gain-free marched field
    marched = eng.amplify_coherent(A_probe, 0.0)["A_out"]    # gain-free -> pure delay of A_probe
    om = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
    refB = np.fft.ifft(np.fft.fft(marched) * np.exp(0.5j * beta2B * om * om * L))
    gotB = eng.amplify_coherent(A_probe, 0.0, beta2_s2_per_m=beta2B)["A_out"]
    relB1 = float(np.max(np.abs(gotB - refB)) / np.max(np.abs(refB)))
    # (B2) a broad-windowed CW tone picks up phi = (beta2/2)(2 pi f)^2 L (measured at the carrier
    # FFT bin, so the symmetric envelope contributes no phase; the nz-step group delay subtracted)
    Tenv = W / 3.0
    worst_b2 = 0.0
    for beta2 in (beta2B, -beta2B):
        for kbin in (4, 9):
            f = kbin / (nt * dt)                            # exact full-window FFT bin
            env = np.exp(-((tgrid - tc) ** 2) / (2.0 * Tenv * Tenv))
            A = (env * np.exp(-1j * 2.0 * np.pi * f * (tgrid - tc))).astype(np.complex128)
            out = eng.amplify_coherent(A, 0.0, beta2_s2_per_m=beta2)["A_out"]
            jb = (nt - kbin) % nt                           # exp(-i 2 pi f t) sits at bin -kbin
            IN, OUT = np.fft.fft(A), np.fft.fft(out)
            phi = np.angle(OUT[jb] * np.conj(IN[jb])) - 2.0 * np.pi * f * nz * dt   # remove delay
            phi_ref = 0.5 * beta2 * (2.0 * np.pi * f) ** 2 * L
            worst_b2 = max(worst_b2, abs(np.angle(np.exp(1j * (phi - phi_ref)))))
    g_b = bool(relB1 < 1e-10 and worst_b2 < 5e-3)
    ok = ok and g_b
    print("[gv] GATE B: split == D(L) of marched field (rel {:.1e}); CW tone phase == (beta2/2)"
          "(2 pi f)^2 L both signs (max |dphi| {:.1e} rad) -> {}".format(
              relB1, worst_b2, "PASS" if g_b else "FAIL"), flush=True)

    # ---- Gaussian pulse setup (gain-free), used by C/D/E ----
    T0 = W / 16.0                                           # << window -> fits with no FFT wraparound
    beta2 = 5.4e-22                                         # L_D = T0^2/beta2 ~ L (L/L_D ~ 1)
    L_D = T0 * T0 / abs(beta2)
    A_g = np.exp(-((tgrid - tc) ** 2) / (2.0 * T0 * T0)).astype(np.complex128)
    rms_in = _rms(tgrid, np.abs(A_g) ** 2)
    out_p = eng.amplify_coherent(A_g, 0.0, beta2_s2_per_m=beta2)["A_out"]
    out_n = eng.amplify_coherent(A_g, 0.0, beta2_s2_per_m=-beta2)["A_out"]

    # ---- GATE C: RMS broadening == NLSE law (independent oracle), sign-symmetric ----
    ratio_law = float(np.sqrt(1.0 + (L / L_D) ** 2))
    relC = 0.0
    for out in (out_p, out_n):
        ratio = _rms(tgrid, np.abs(out) ** 2) / rms_in
        relC = max(relC, abs(ratio - ratio_law) / ratio_law)
    g_c = bool(relC < 2e-2)
    ok = ok and g_c
    print("[gv] GATE C: Gaussian RMS broadens x{:.3f} == T0 sqrt(1+(L/L_D)^2) (L/L_D={:.2f}, "
          "max rel {:.1e}) -> {}".format(ratio_law, L / L_D, relC, "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: quadratic chirp sign + magnitude (independent sign oracle) ----
    C_ref = -(beta2 * L) / (2.0 * (T0 ** 4 + (beta2 * L) ** 2))
    relD, sign_ok = 0.0, True
    for out, b in ((out_p, beta2), (out_n, -beta2)):
        p = np.abs(out) ** 2
        mask = p > 0.05 * p.max()                          # fit the phase only where the pulse lives
        t0 = tgrid[int(np.argmax(p))]
        tt = tgrid[mask] - t0
        ph = np.unwrap(np.angle(out[mask]))
        c2 = np.polyfit(tt, ph, 2)[0]                      # quadratic coefficient C
        Cref_b = -(b * L) / (2.0 * (T0 ** 4 + (b * L) ** 2))
        sign_ok = sign_ok and (np.sign(c2) == np.sign(Cref_b))
        relD = max(relD, abs(c2 - Cref_b) / abs(Cref_b))
    g_d = bool(sign_ok and relD < 8e-2)
    ok = ok and g_d
    print("[gv] GATE D: chirp phi=C T^2, C sign flips with beta2 (sign_ok {}); |C| == analytic "
          "(C_ref {:.2e}, max rel {:.1e}) -> {}".format(sign_ok, C_ref, relD,
                                                        "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: unitarity (pure dispersion conserves pulse energy) ----
    E_in = float(np.sum(np.abs(A_g) ** 2))
    relE = max(abs(float(np.sum(np.abs(out_p) ** 2)) - E_in) / E_in,
               abs(float(np.sum(np.abs(out_n) ** 2)) - E_in) / E_in)
    g_e = bool(relE < 1e-6)
    ok = ok and g_e
    print("[gv] GATE E: gain-free dispersion conserves sum|A|^2 (unitary, max rel {:.1e}) -> "
          "{}".format(relE, "PASS" if g_e else "FAIL"), flush=True)

    print("[gv] *** QD-SOA GVD: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
