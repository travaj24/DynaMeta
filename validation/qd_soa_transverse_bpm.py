"""QD-SOA transverse 2-D (x-z) gain-coupled BPM vs analytic oracles. TransverseBPM resolves the
lateral axis the 1-D engine lumps into Gamma/A_mode: split-step Fresnel diffraction + saturable
complex gain + lateral carrier diffusion -> diffraction, gain guiding, transverse spatial hole
burning, and alpha-driven self-focusing / FILAMENTATION.

GATE A (reduce to 1-D): a laterally-UNIFORM beam (only k_x=0, untouched by diffraction) reduces to the
        1-D saturable-gain ODE dI/dz = (Gamma g0/(1+I/Isat) - alpha_i) I; small-signal it is the exact
        exp((Gamma g0 - alpha_i) L); the profile stays flat.
GATE B (pure diffraction): with the gain off a Gaussian beam spreads by the exact paraxial law
        w(z) = w0 sqrt(1+(z/zR)^2), zR = pi n0 w0^2/lambda (RMS width), and the energy is conserved
        (the diffraction operator is unitary).
GATE C (self-focusing direction): a bright Gaussian under gain self-lenses through alpha -- the output
        is NARROWER than the alpha=0 (gain-spatial-hole-burning) baseline for alpha>0 (converging,
        filamentation-prone) and BROADER for alpha<0 (defocusing).
GATE D (filamentation + diffusion suppression): a broad flat-top beam with a small noise seed breaks
        into FILAMENTS -- the filament-scale (high transverse spatial frequency) power grows strongly
        with alpha; lateral carrier diffusion (L_diff>0) SUPPRESSES it (washes out the carrier ripple).
GATE E (carrier-diffusion SHB smoothing + passivity): a bright narrow spot burns a gain hole whose
        contrast SHRINKS monotonically as L_diff grows (diffusion smooths the spatial hole burning);
        all profiles finite.

Run: python -m validation.qd_soa_transverse_bpm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.integrate import solve_ivp

from dynameta.optics.soa.transverse_bpm import TransverseBPM

LAM, N0 = 1.3e-6, 3.4


def main():
    print("[bpm] === QD-SOA transverse 2-D gain-coupled BPM vs oracles ===", flush=True)
    ok = True

    # ---- GATE A: uniform beam reduces to the 1-D saturable-gain ODE ----
    g0, Isat, ai, Lz = 2000.0, 5.0e-3, 300.0, 0.5e-3
    b = TransverseBPM(100e-6, 256, LAM, N0, g0_per_m=g0, alpha_i_per_m=ai, Isat_W=Isat)
    I0 = 1.0e-3
    o = b.propagate(np.full(256, np.sqrt(I0) + 0j), Lz, 400)
    sol = solve_ivp(lambda z, I: (g0 / (1.0 + I / Isat) - ai) * I, (0.0, Lz), [I0],
                    rtol=1e-10, atol=1e-16, t_eval=[Lz])
    relA = abs(o["I_out"][128] - sol.y[0, -1]) / sol.y[0, -1]
    flat = (o["I_out"].max() - o["I_out"].min()) / o["I_out"].mean()
    # small-signal exact
    bss = TransverseBPM(100e-6, 256, LAM, N0, g0_per_m=g0, alpha_i_per_m=ai, Isat_W=1e9)
    oss = bss.propagate(np.full(256, np.sqrt(1e-9) + 0j), Lz, 400)
    rel_ss = abs(oss["I_out"][128] / 1e-9 - np.exp((g0 - ai) * Lz)) / np.exp((g0 - ai) * Lz)
    g_a = bool(relA < 1e-3 and flat < 1e-12 and rel_ss < 1e-3)
    ok = ok and g_a
    print("[bpm] GATE A: uniform == 1-D ODE (rel {:.1e}), flat {:.0e}, small-signal exact (rel {:.1e}) "
          "-> {}".format(relA, flat, rel_ss, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: pure diffraction Gaussian width + energy ----
    bd = TransverseBPM(400e-6, 4096, LAM, N0, g0_per_m=0.0)
    w0 = 12e-6
    A = np.exp(-(bd.x / w0) ** 2) + 0j
    zR = np.pi * N0 * w0 ** 2 / LAM
    od = bd.propagate(A, zR, 200)
    relB = abs(bd.rms_width(od["I_out"]) - (w0 / 2.0) * np.sqrt(2.0)) / ((w0 / 2.0) * np.sqrt(2.0))
    energy = od["I_out"].sum() / (np.abs(A) ** 2).sum()
    g_b = bool(relB < 1e-3 and abs(energy - 1.0) < 1e-9)
    ok = ok and g_b
    print("[bpm] GATE B: diffraction == Gaussian w(z) at zR (rel {:.1e}), energy {:.7f} -> {}".format(
        relB, energy, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: self-focusing direction (alpha lens) ----
    def out_width(alpha):
        bc = TransverseBPM(300e-6, 2048, LAM, N0, g0_per_m=1500.0, Isat_W=1e-3, alpha_lef=alpha)
        Ac = np.sqrt(3e-3) * np.exp(-(bc.x / 15e-6) ** 2) + 0j
        return bc.rms_width(bc.propagate(Ac, 0.4e-3, 400)["I_out"])
    wm, w0c, wp = out_width(-3.0), out_width(0.0), out_width(3.0)
    g_c = bool(wp < w0c < wm)                          # alpha>0 focuses, alpha<0 defocuses
    ok = ok and g_c
    print("[bpm] GATE C: self-focusing (w(a=-3) {:.3e} > w(0) {:.3e} > w(a=+3) {:.3e}) -> {}".format(
        wm, w0c, wp, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: filamentation + diffusion suppression ----
    # Grid-invariant metric: the out/in filament-band AMPLIFICATION (the white-noise seed PSD ~ nx
    # cancels in the ratio), NOT the raw output band power.
    def filament_amp(alpha, Ld):
        nx = 1024
        bf = TransverseBPM(200e-6, nx, LAM, N0, g0_per_m=1000.0, Isat_W=2e-3, alpha_lef=alpha,
                           L_diff_m=Ld)
        rng = np.random.default_rng(7)
        ft = 1.0 / (1.0 + (bf.x / 60e-6) ** 10)       # 120 um flat-top
        Af = np.sqrt(1e-3) * ft * (1.0 + 0.01 * rng.standard_normal(nx)) + 0j
        kx = np.abs(2.0 * np.pi * np.fft.fftfreq(nx, d=bf.dx))
        band = (kx > 2 * np.pi / 40e-6) & (kx < 2 * np.pi / 5e-6)   # 5..40 um filament scales
        bp = lambda I: (np.abs(np.fft.fft(I - I.mean())) ** 2)[band].sum()
        return bp(bf.propagate(Af, 1.5e-3, 900)["I_out"]) / bp(np.abs(Af) ** 2)
    a0, a4 = filament_amp(0.0, 0.0), filament_amp(4.0, 0.0)
    a4d = filament_amp(4.0, 6e-6)
    g_d = bool(a4 > 3.0 * a0 and a4d < a4)
    ok = ok and g_d
    print("[bpm] GATE D: filament-band amplification grows ({:.2f}x, a=0 {:.2f}->a=4 {:.2f}); L_diff "
          "suppresses ({:.2f}<{:.2f}) -> {}".format(a4 / a0, a0, a4, a4d, a4,
                                                    "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: carrier-diffusion SHB smoothing + passivity ----
    def gain_contrast(Ld):
        be = TransverseBPM(120e-6, 512, LAM, N0, g0_per_m=1500.0, Isat_W=1e-3, L_diff_m=Ld)
        g = be.carrier_gain(np.sqrt(2e-3) * np.exp(-(be.x / 8e-6) ** 2) + 0j)
        return (g.max() - g.min()) / g.mean(), g
    c0, _ = gain_contrast(0.0)
    c3, _ = gain_contrast(3e-6)
    c8, g8 = gain_contrast(8e-6)
    g_e = bool(c8 < c3 < c0 and np.all(np.isfinite(g8)))   # monotone smoothing
    ok = ok and g_e
    print("[bpm] GATE E: lateral diffusion smooths SHB (gain contrast {:.3f} -> {:.3f} -> {:.3f} for "
          "L_diff 0/3/8 um) -> {}".format(c0, c3, c8, "PASS" if g_e else "FAIL"), flush=True)

    # ---- GATE F: 2-D thermal lensing (Phase 24) ----
    bl = TransverseBPM(400e-6, 4096, LAM, N0, g0_per_m=0.0)
    w0l = 20e-6
    Al = np.exp(-(bl.x / w0l) ** 2) + 0j
    # dndt=0 byte-identical
    f_byte = np.array_equal(bl.propagate(Al, 0.3e-3, 200)["A_out"],
                            bl.propagate(Al, 0.3e-3, 200, T_profile_x=np.ones(4096),
                                         dndt_per_K=0.0)["A_out"])
    # linear T ramp -> EXACT beam steering <kx> == k0 dndt b L (Fourier shift)
    dndt, bslope, Lzl = 3e-4, 1e5, 0.3e-3
    oL = bl.propagate(Al, Lzl, 200, T_profile_x=bslope * bl.x, dndt_per_K=dndt)
    F = np.abs(np.fft.fft(oL["A_out"])) ** 2
    kxc = np.sum(bl.kx * F) / np.sum(F)
    relF = abs(kxc - bl.k0 * dndt * bslope * Lzl) / abs(bl.k0 * dndt * bslope * Lzl)
    en = oL["I_out"].sum() / (np.abs(Al) ** 2).sum()
    # quadratic hot-centre -> focusing (dndt>0 narrower, dndt<0 broader)
    Tq = -0.5 * 5e12 * bl.x ** 2
    wn = bl.rms_width(bl.propagate(Al, 0.6e-3, 300)["I_out"])
    wf = bl.rms_width(bl.propagate(Al, 0.6e-3, 300, T_profile_x=Tq, dndt_per_K=2e-3)["I_out"])
    wd = bl.rms_width(bl.propagate(Al, 0.6e-3, 300, T_profile_x=Tq, dndt_per_K=-2e-3)["I_out"])
    g_f = bool(f_byte and relF < 1e-9 and abs(en - 1.0) < 1e-9 and wf < wn < wd)
    ok = ok and g_f
    print("[bpm] GATE F: thermal lens -- dndt=0 byte-id {}; linear-ramp steer exact (rel {:.1e}, energy "
          "{:.7f}); hot-centre focuses (wf {:.2e} < wn {:.2e} < wd {:.2e}) -> {}".format(
              f_byte, relF, en, wf, wn, wd, "PASS" if g_f else "FAIL"), flush=True)

    # ---- GATE G: Strang-split is 2nd-order in dz (Richardson self-convergence) ----
    # The module docstring claims O(dz^2) ("verified ratio 4.0") but no gate proved it. Use a beam that
    # exercises BOTH legs at once -- a Gaussian with strong saturable gain (Isat finite) AND diffraction
    # AND alpha-coupling -- so the test is sensitive to the NONLINEAR-leg midpoint correction (a frozen-g
    # 1st-order leg would cap the whole method at O(dz)). Richardson: with the SAME physical Lz solved at
    # nz, 2nz, 4nz steps, the successive-difference norm ratio ||A(nz)-A(2nz)|| / ||A(2nz)-A(4nz)|| -> 2^p.
    # 2nd order (real) -> 4; 1st order (frozen-gain, no midpoint) -> 2. Asserting ratio > 3 EXCLUDES the
    # 1st-order method (the discriminator), and the upper bound catches a regime/roundoff pathology.
    bg = TransverseBPM(300e-6, 512, LAM, N0, g0_per_m=1500.0, Isat_W=1.0e-3, alpha_lef=3.0,
                       alpha_i_per_m=200.0)
    Ag = np.sqrt(3.0e-3) * np.exp(-(bg.x / 15e-6) ** 2) + 0j
    Lzg = 0.4e-3
    A1 = bg.propagate(Ag, Lzg, 40)["A_out"]
    A2 = bg.propagate(Ag, Lzg, 80)["A_out"]
    A4 = bg.propagate(Ag, Lzg, 160)["A_out"]
    d1 = float(np.linalg.norm(A1 - A2))
    d2 = float(np.linalg.norm(A2 - A4))
    ratio = d1 / d2 if d2 > 0.0 else np.inf
    g_g = bool(d2 < d1 and 3.0 < ratio < 5.0)              # ~4 (2nd order), excludes ~2 (1st order)
    ok = ok and g_g
    print("[bpm] GATE G: Strang-split Richardson ratio ||dA(40-80)||/||dA(80-160)|| = {:.2f} (2nd-order "
          "-> 4, 1st-order -> 2; d1 {:.2e} > d2 {:.2e}) -> {}".format(
              ratio, d1, d2, "PASS" if g_g else "FAIL"), flush=True)

    print("[bpm] *** QD-SOA TRANSVERSE BPM: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
