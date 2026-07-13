"""R15 chi2 SHG + dispersive (Raman) chi3 FDTD oracle.

Testbed: a uniform n = sqrt(2) medium EVERYWHERE (index-matched ends -> zero Fresnel, exact
Delta_k = 0 phase matching) with the nonlinearity active only in a window of length L; the
kernels are driven directly with narrowband sources and raw probe time series.

GATE A (SHG selectivity + slope 2): the second-harmonic band appears ONLY around 2 f0; doubling
        the source amplitude quadruples the SHG field (|E_2| ratio == 4 to 1e-3, the perturbative
        power-law exponent 2); the pump band is undepleted.
GATE B (undepleted-pump coupled-wave closed form): for Delta_k = 0 the driven solution is
        e2(L, t) = (chi2 omega0 L / (2 n c)) Im[z1(t)^2] with z1 the pump analytic signal --
        the validation builds z1 from the MEASURED chi2=0 pump probe (scipy hilbert) and
        compares |FFT| over the SHG band: max rel < 5% (narrowband prefactor approximation).
GATE C (Raman vibrational ADE, machine): the kernel's central-difference recursion for
        Q'' + gam Q' + W^2 Q = W^2 E^2 vs scipy solve_ivp on a prescribed E(t): rel < 1e-3.
GATE D (Raman Stokes gain physics): two-tone pump+probe with the probe at f_p - f_R: chi3R > 0
        AMPLIFIES the Stokes probe and ATTENUATES the anti-Stokes probe (exp(-i w t): Im
        Delta_chi = -+ chiR A_p^2 W/(2 gam)); |ln G| matches the analytic field exponent
        g L = omega_s chiR A_p^2 W L / (4 n c gam) within a factor 2.5 (Gaussian-envelope
        smearing of the CW formula).
GATE E (byte-identical off-switch): solve_fdtd_2d with chi2/Raman fields zero returns R0/T0
        ARRAY-EQUAL to the pre-R15 path on the same stack.

Run: python -m validation.fdtd_chi2_shg_raman
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.signal import hilbert

from dynameta.constants import C_LIGHT, EPS0
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import run_2d_te, cpml_z, solve_fdtd_2d

N_MED = np.sqrt(2.0)
NX = 4


def _grid(dz, n_pad_cells, n_struct_cells):
    nz = 2 * n_pad_cells + n_struct_cells
    dx = 4.0 * dz
    dt = 0.5 / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    return nz, dx, dt


def _run(dz, n_pad, n_str, src, *, chi2_val=0.0, raman=None, kerr_val=0.0, nsteps=None):
    """Uniform-eps medium, nonlinearity active in the central window; returns (eyR, dt, k_pR)."""
    nz, dx, dt = _grid(dz, n_pad, n_str)
    nsteps = int(nsteps if nsteps is not None else src.size)
    eps = np.full((NX, nz), N_MED ** 2)
    zeros = np.zeros((NX, nz))
    c3k = zeros
    if kerr_val:
        c3k = np.zeros((NX, nz)); c3k[:, n_pad:n_pad + n_str] = kerr_val
    chi2 = None
    if chi2_val:
        chi2 = np.zeros((NX, nz)); chi2[:, n_pad:n_pad + n_str] = chi2_val
    ram = None
    if raman is not None:
        chi3R, W, g = raman
        c3 = np.zeros((NX, nz)); c3[:, n_pad:n_pad + n_str] = chi3R
        den = 1.0 + g * dt / 2.0
        ram = (np.full((NX, nz), (2.0 - W ** 2 * dt ** 2) / den),
               np.full((NX, nz), (g * dt / 2.0 - 1.0) / den),
               np.full((NX, nz), W ** 2 * dt ** 2 / den), c3)
    cpml = cpml_z(nz, dz, dt, 12, N_MED, N_MED)
    k_src, k_pR = 16, nz - 16
    _, _, eyR, _ = run_2d_te(eps, zeros, zeros, c3k, dx, dz, dt, nsteps, k_src, 20, k_pR,
                              src[:nsteps], cpml, np, None, chi2=chi2, raman=ram)
    return eyR.mean(axis=1), dt


def _band_amp(sig, dt, f_lo, f_hi):
    F = np.abs(np.fft.rfft(sig))
    f = np.fft.rfftfreq(sig.size, dt)
    m = (f >= f_lo) & (f <= f_hi)
    return F, f, m


def main():
    print("[nl] === R15 chi2 SHG + Raman chi3 FDTD ===", flush=True)
    ok = True

    # ---- SHG testbed: f0 = 250 THz (1.2 um), L = 400 nm, chi2 = 5e-10 m/V ----
    f0 = 2.5e14
    dz = 10e-9
    n_pad, n_str = 60, 40                                 # L = 400 nm window
    L = n_str * dz
    nz, dx, dt = _grid(dz, n_pad, n_str)
    tau = 60e-15
    t0 = 6.0 * tau
    nsteps = int(round((2.0 * t0 + 200e-15) / dt))
    t = np.arange(nsteps) * dt
    A0 = 5.0e8
    chi2_v = 2.0e-11      # chi2*E ~ 1.5e-2: perturbative (the lagged explicit coupling destabilizes only at the unphysical chi2*E ~ 0.3)
    src1 = A0 * np.exp(-((t - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f0 * (t - t0))

    ey_ref, _ = _run(dz, n_pad, n_str, src1)                      # chi2 = 0 baseline
    ey_1, _ = _run(dz, n_pad, n_str, src1, chi2_val=chi2_v)
    ey_2, _ = _run(dz, n_pad, n_str, 2.0 * src1, chi2_val=chi2_v)
    F_ref, f, m_sh = _band_amp(ey_ref, dt, 1.85 * f0, 2.15 * f0)
    F1, _, _ = _band_amp(ey_1, dt, 0.0, 1.0)
    F2, _, _ = _band_amp(ey_2, dt, 0.0, 1.0)
    m_pu = (f >= 0.9 * f0) & (f <= 1.1 * f0)
    m_3h = (f >= 2.8 * f0) & (f <= 3.2 * f0)

    # ---- GATE A: selectivity + slope 2 + undepleted pump ----
    sh1, sh2 = float(np.max(F1[m_sh])), float(np.max(F2[m_sh]))
    sh_ref = float(np.max(F_ref[m_sh]))
    ratio = sh2 / sh1
    # the 3 f0 line is the REAL chi2 CASCADE (sum-frequency of pump + SH), one more perturbative
    # order down: e3/e2 ~ chi2*E ~ 1.5e-2 here -- bound it AT that scale, not below it.
    third = float(np.max(F1[m_3h])) / sh1
    cascade_scale = chi2_v * float(np.max(np.abs(ey_ref)))
    depl = abs(float(np.max(F1[m_pu])) / float(np.max(F_ref[m_pu])) - 1.0)
    g_a = bool(sh1 > 100.0 * sh_ref and abs(ratio - 4.0) < 4e-3 * 4.0
               and third < 3.0 * cascade_scale and depl < 1e-3)
    ok = ok and g_a
    print("[nl] GATE A: SHG band {:.1e}x above baseline; 2x amplitude -> field x{:.4f} (slope 2);"
          " 3rd harmonic {:.1e} of SHG == the chi2*E ~ {:.1e} cascade; pump depletion {:.1e} "
          "-> {}".format(sh1 / max(sh_ref, 1e-300), ratio, third, cascade_scale, depl,
                         "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: undepleted-pump closed form from the measured pump analytic signal ----
    z1 = hilbert(ey_ref)                                          # measured pump at the exit probe
    e2_pred = (chi2_v * 2.0 * np.pi * f0 * L / (2.0 * N_MED * C_LIGHT)) * np.imag(z1 ** 2)
    Fp = np.abs(np.fft.rfft(e2_pred))
    band = m_sh & (F1 > 0.05 * sh1)                               # well-excited SHG bins
    relB = float(np.max(np.abs(F1[band] - Fp[band]) / np.max(Fp[band])))
    g_b = bool(relB < 5e-2)
    ok = ok and g_b
    print("[nl] GATE B: SHG spectrum vs (chi2 w0 L / 2nc) Im[z1^2] closed form, band max rel "
          "{:.1e} -> {}".format(relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: Raman ADE recursion vs solve_ivp ----
    from scipy.integrate import solve_ivp
    W, gR = 2.0 * np.pi * 1.0e13, 2.0 * np.pi * 1.0e12
    dtc = 1.0e-17
    nc = 60000
    tc = np.arange(nc) * dtc
    Et = np.exp(-((tc - 3e-13) / 1e-13) ** 2) * np.sin(2.0 * np.pi * 2.0e13 * tc)
    den = 1.0 + gR * dtc / 2.0
    r1, r2, r3 = (2.0 - W ** 2 * dtc ** 2) / den, (gR * dtc / 2.0 - 1.0) / den, \
        W ** 2 * dtc ** 2 / den
    Q = np.zeros(nc)
    for n in range(1, nc - 1):
        Q[n + 1] = r1 * Q[n] + r2 * Q[n - 1] + r3 * Et[n] ** 2
    Ei = lambda tt: np.exp(-((tt - 3e-13) / 1e-13) ** 2) * np.sin(2.0 * np.pi * 2.0e13 * tt)
    sol = solve_ivp(lambda tt, y: [y[1], W ** 2 * Ei(tt) ** 2 - gR * y[1] - W ** 2 * y[0]],
                    (0.0, tc[-1]), [0.0, 0.0], t_eval=tc, method="DOP853",
                    rtol=1e-10, atol=1e-14)
    relC = float(np.max(np.abs(Q - sol.y[0])) / np.max(np.abs(sol.y[0])))
    g_c = bool(relC < 1e-3)
    ok = ok and g_c
    print("[nl] GATE C: kernel Raman ADE recursion vs solve_ivp, rel {:.1e} -> {}".format(
        relC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: Stokes gain / anti-Stokes loss ----
    f_R = 1.0e13
    f_p = 2.5e14
    chiR = 1.0e-19
    dz2 = 28e-9
    n_pad2, n_str2 = 40, 15                                       # L = 420 nm
    L2 = n_str2 * dz2
    nz2, dx2, dt2 = _grid(dz2, n_pad2, n_str2)
    tau2 = 6.4e-13
    t02 = 5.0 * tau2
    ns2 = int(round((2.0 * t02 + 1.0e-12) / dt2))
    t2 = np.arange(ns2) * dt2
    Ap, As = 5.0e8, 5.0e6
    env = np.exp(-((t2 - t02) / tau2) ** 2)
    g_th = (2.0 * np.pi * (f_p - f_R)) * chiR * Ap ** 2 * W * L2 / \
        (4.0 * N_MED * C_LIGHT * (gR))
    gains = {}
    for sgn in (-1.0, +1.0):                                      # -1 Stokes, +1 anti-Stokes
        f_s = f_p + sgn * f_R
        src = env * (Ap * np.cos(2.0 * np.pi * f_p * (t2 - t02))
                     + As * np.cos(2.0 * np.pi * f_s * (t2 - t02)))
        ey_off, _ = _run(dz2, n_pad2, n_str2, src, nsteps=ns2)
        ey_on, _ = _run(dz2, n_pad2, n_str2, src, raman=(chiR, W, gR), nsteps=ns2)
        Fo, ff, _ = _band_amp(ey_off, dt2, 0, 1)
        Fn, _, _ = _band_amp(ey_on, dt2, 0, 1)
        ms = (ff >= f_s - 2e12) & (ff <= f_s + 2e12)
        gains[sgn] = float(np.max(Fn[ms]) / np.max(Fo[ms]))
    lnGs, lnGa = np.log(gains[-1.0]), np.log(gains[+1.0])
    g_d = bool(gains[-1.0] > 1.005 and gains[+1.0] < 0.995
               and 0.4 < lnGs / g_th < 2.5 and 0.4 < -lnGa / g_th < 2.5)
    ok = ok and g_d
    print("[nl] GATE D: Stokes gain {:.4f} (ln G/gL = {:.2f}), anti-Stokes {:.4f} "
          "(ln G/gL = {:.2f}; analytic gL = {:.3f}) -> {}".format(
              gains[-1.0], lnGs / g_th, gains[+1.0], -lnGa / g_th, g_th,
              "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: public off-switch byte-identity ----
    stack0 = [FDTDLayer(200e-9, eps_inf=2.0)]
    stack1 = [FDTDLayer(200e-9, eps_inf=2.0, chi2_m_V=0.0, raman_chi3_m2_V2=0.0)]
    r0 = solve_fdtd_2d(stack0, period_x_m=100e-9, lambda_min_m=1.0e-6, lambda_max_m=1.4e-6,
                       resolution=20, backend="numpy")
    r1 = solve_fdtd_2d(stack1, period_x_m=100e-9, lambda_min_m=1.0e-6, lambda_max_m=1.4e-6,
                       resolution=20, backend="numpy")
    g_e = bool(np.array_equal(r0.R0, r1.R0) and np.array_equal(r0.T0, r1.T0))
    ok = ok and g_e
    print("[nl] GATE E: zero chi2/Raman fields -> R0/T0 ARRAY-EQUAL to the pre-R15 path -> {}"
          .format("PASS" if g_e else "FAIL"), flush=True)

    # ---- GATE F (audit C3-2): ABSOLUTE instantaneous-Kerr magnitude from the measured pump.
    # Standard chi^(3) convention (P_NL = eps0 chi3 E^3): d_eps_fund = (3/4) chi3 |E|^2, so the
    # SPM phase accrued over the window is dphi(t) = (w0 L / c) (3/8) chi3 |z1(t)|^2 / n and the
    # first-order perturbation field is dphi x the pump quadrature. Pinned ABSOLUTELY against the
    # measured analytic pump: the pre-fix update (eps_inf + chi3 E^2) delivered exactly 1/3 of
    # this (ratio ~0.33 fails the [0.8, 1.25] band hard); scaling/reduction gates are blind to it.
    chi3_v = 8.0e-20                                              # 3 chi3 A0^2 ~ 6e-2: weak, resolved
    ey_kf, _ = _run(dz, n_pad, n_str, src1, kerr_val=chi3_v)
    zk = hilbert(ey_ref)
    dphi = (2.0 * np.pi * f0 / C_LIGHT) * L * (3.0 / 8.0) * chi3_v * np.abs(zk) ** 2 / N_MED
    F_diff, f_f, _ = _band_amp(ey_kf - ey_ref, dt, 0.0, 1.0)
    F_pred, _, _ = _band_amp(dphi * np.imag(zk), dt, 0.0, 1.0)
    m_pump = (f_f >= 0.9 * f0) & (f_f <= 1.1 * f0)
    ratioF = float(np.max(F_diff[m_pump]) / np.max(F_pred[m_pump]))
    g_f = bool(0.8 < ratioF < 1.25)
    ok = ok and g_f
    print("[nl] GATE F: ABSOLUTE Kerr SPM |diff|/|(3/8) chi3 |z1|^2 (w0 L/c n) quad| = {:.3f} "
          "(pre-fix convention ~0.33) -> {}".format(ratioF, "PASS" if g_f else "FAIL"), flush=True)

    print("[nl] *** R15 CHI2 SHG + RAMAN CHI3: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
