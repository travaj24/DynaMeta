"""R20 four-level gain-medium oracle (clamped-inversion gain ADE + exact population dynamics).

Testbed (field gates): the R15 index-matched uniform n = sqrt(2) medium with the gain line
active in a window of length L; narrowband probe; raw kernel probes.

GATE A (small-signal gain closed form): the measured field amplification at line center equals
        exp(g0 L / 2) with g0 = kappa dN/(n c eps0 dw) -- tol 5%; doubling dN doubles ln(G)
        (small-signal linearity).
GATE B (Lorentzian lineshape): the ln-gain at detuning +-dw/2 from line center is HALF the
        line-center value (the half-width of Im chi) -- tol 10%.
GATE C (passive equivalence, EXACT): dN < 0 with kappa|dN| = eps0 delta_eps w_a^2 produces
        probe traces ARRAY-EQUAL to the SHIPPED passive Lorentz pole (identical recursion
        coefficients by construction -- absorption and gain are one sign flip).
GATE D (four-level populations, exact propagator): evolve() vs scipy solve_ivp (tight rtol)
        agrees to < 1e-8; sum(N) == N_total to machine at every time; steady state matches the
        chain-balance closed form and inversion_ss = W_p N0 (tau_21 - tau_10).
GATE E (off-switch): gain fields zero -> public solve_fdtd_2d R0/T0 ARRAY-EQUAL; guards raise.

Run: python -m validation.fdtd_gain_medium
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT, EPS0, M_E, Q_E
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import run_2d_te, cpml_z, solve_fdtd_2d
from dynameta.optics.laser_gain import FourLevelSystem, small_signal_gain_per_m

N_MED = np.sqrt(2.0)
NX = 4
KAPPA = Q_E ** 2 / M_E                     # classical electron coupling [C^2/kg]


def _grid(dz, n_pad, n_str):
    nz = 2 * n_pad + n_str
    dx = 4.0 * dz
    dt = 0.5 / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    return nz, dx, dt


def _run(dz, n_pad, n_str, src, *, gain=None, lor=None):
    nz, dx, dt = _grid(dz, n_pad, n_str)
    eps = np.full((NX, nz), N_MED ** 2)
    zeros = np.zeros((NX, nz))

    def _coeffs(w, dw, src_coeff):
        den = 1.0 + dw * dt / 2.0
        g1 = np.full((NX, nz), (2.0 - w ** 2 * dt ** 2) / den)
        g2 = np.full((NX, nz), (dw * dt / 2.0 - 1.0) / den)
        g3 = np.zeros((NX, nz)); g3[:, n_pad:n_pad + n_str] = src_coeff * dt ** 2 / den
        return g1, g2, g3

    gn = ln = None
    if gain is not None:
        w, dw, kdn = gain
        gn = _coeffs(w, dw, -kdn)
    if lor is not None:
        w, dw, deps = lor
        ln = _coeffs(w, dw, EPS0 * deps * w ** 2)
    cpml = cpml_z(nz, dz, dt, 12, N_MED, N_MED)
    _, _, eyR, _ = run_2d_te(eps, zeros, zeros, zeros, dx, dz, dt, src.size, 16, 20, nz - 16,
                              src, cpml, np, ln, gain=gn)
    return eyR.mean(axis=1), dt


def _amp_at(sig, dt, f):
    F = np.abs(np.fft.rfft(sig))
    fr = np.fft.rfftfreq(sig.size, dt)
    return float(F[np.argmin(np.abs(fr - f))])


def main():
    print("[gm] === R20 four-level gain medium ===", flush=True)
    ok = True
    f0 = 2.5e14
    w_a = 2.0 * np.pi * f0
    dw = 2.0 * np.pi * 2.0e13                                 # broad line (probe fits inside)
    dz = 10e-9
    n_pad, n_str = 60, 40
    L = n_str * dz
    nz, dx, dt = _grid(dz, n_pad, n_str)
    tau = 120e-15
    t0 = 6.0 * tau
    nsteps = int(round((2.0 * t0 + 200e-15) / dt))
    t = np.arange(nsteps) * dt
    src = 1.0e6 * np.exp(-((t - t0) / tau) ** 2) * np.cos(w_a * (t - t0))

    dN = 6.0e24
    g0 = small_signal_gain_per_m(KAPPA, dN, N_MED, dw)        # ~2.4e5 1/m -> g0 L/2 ~ 0.05
    ey_ref, _ = _run(dz, n_pad, n_str, src)
    ey_g, _ = _run(dz, n_pad, n_str, src, gain=(w_a, dw, KAPPA * dN))
    ey_g2, _ = _run(dz, n_pad, n_str, src, gain=(w_a, dw, KAPPA * 2.0 * dN))

    # ---- GATE A: line-center gain + linearity in dN ----
    lnG = np.log(_amp_at(ey_g, dt, f0) / _amp_at(ey_ref, dt, f0))
    lnG2 = np.log(_amp_at(ey_g2, dt, f0) / _amp_at(ey_ref, dt, f0))
    relA = abs(lnG - g0 * L / 2.0) / (g0 * L / 2.0)
    lin = abs(lnG2 / lnG - 2.0)
    g_a = bool(relA < 5e-2 and lin < 5e-2)
    ok = ok and g_a
    print("[gm] GATE A: ln(G) = {:.4f} vs g0 L/2 = {:.4f} (rel {:.1e}); 2x dN -> ln(G) x{:.3f} "
          "-> {}".format(lnG, g0 * L / 2.0, relA, lnG2 / lnG, "PASS" if g_a else "FAIL"),
          flush=True)

    # ---- GATE B: Lorentzian half-width ----
    worstB = 0.0
    for sgn in (-1.0, +1.0):
        f_d = f0 + sgn * dw / (4.0 * np.pi)                   # w_a +- dw/2
        lnGd = np.log(_amp_at(ey_g, dt, f_d) / _amp_at(ey_ref, dt, f_d))
        worstB = max(worstB, abs(lnGd / lnG - 0.5))
    g_b = bool(worstB < 0.10)
    ok = ok and g_b
    print("[gm] GATE B: ln-gain at w_a +- dw/2 == half the line-center value (worst |d| = "
          "{:.1e}) -> {}".format(worstB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: dN < 0 ARRAY-EQUAL to the passive Lorentz pole ----
    deps_eq = KAPPA * dN / (EPS0 * w_a ** 2)
    ey_neg, _ = _run(dz, n_pad, n_str, src, gain=(w_a, dw, -KAPPA * dN))
    ey_lor, _ = _run(dz, n_pad, n_str, src, lor=(w_a, dw, deps_eq))
    g_c = bool(np.array_equal(ey_neg, ey_lor))
    ok = ok and g_c
    print("[gm] GATE C: dN < 0 gain line ARRAY-EQUAL to the passive Lorentz pole with "
          "delta_eps = kappa|dN|/(eps0 w_a^2) -> {}".format("PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: four-level populations ----
    from scipy.integrate import solve_ivp
    sysm = FourLevelSystem(tau_32_s=4e-7, tau_21_s=2.3e-4, tau_10_s=1e-8, W_p_per_s=50.0,
                           N_total_m3=1e25)
    tt = np.linspace(0.0, 2.0e-3, 41)
    Nev = sysm.evolve(tt)
    sol = solve_ivp(lambda _t, y: sysm.rate_matrix() @ y, (0.0, tt[-1]),
                    [1e25, 0.0, 0.0, 0.0], t_eval=tt, method="LSODA", rtol=1e-11, atol=1e3)
    relD = float(np.max(np.abs(Nev - sol.y.T)) / 1e25)
    cons = float(np.max(np.abs(Nev.sum(axis=1) - 1e25)) / 1e25)   # expm roundoff scale (~1e-12)
    ss = sysm.steady_state()
    # the slowest relaxation is the pump (1/W_p = 20 ms): evolve to 0.5 s (e^-25) for the
    # steady-state comparison -- a single exact expm step, not a long integration
    N_long = sysm.evolve(np.array([0.5]))[0]
    dn_ss = sysm.inversion_ss_m3()
    dn_cf = sysm.W_p_per_s * ss[0] * (2.3e-4 - 1e-8)
    g_d = bool(relD < 1e-8 and cons < 1e-11
               and float(np.max(np.abs(N_long - ss)) / 1e25) < 1e-9
               and abs(dn_ss - dn_cf) / dn_cf < 1e-12)
    ok = ok and g_d
    print("[gm] GATE D: expm propagator vs solve_ivp rel {:.1e}; sum(N) conserved {:.1e}; "
          "steady state + inversion W_p N0 (tau21 - tau10) exact -> {}".format(
              relD, cons, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: public off-switch + guards ----
    r0 = solve_fdtd_2d([FDTDLayer(150e-9, eps_inf=2.0)], period_x_m=100e-9, lambda_min_m=1.0e-6,
                       lambda_max_m=1.4e-6, resolution=16, backend="numpy")
    r1 = solve_fdtd_2d([FDTDLayer(150e-9, eps_inf=2.0, gain_dN_m3=0.0)], period_x_m=100e-9,
                       lambda_min_m=1.0e-6, lambda_max_m=1.4e-6, resolution=16, backend="numpy")
    guards = False
    try:
        solve_fdtd_2d([FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=KAPPA, gain_dN_m3=1e24)],
                      period_x_m=100e-9, lambda_min_m=1.0e-6, lambda_max_m=1.4e-6,
                      resolution=16, backend="numpy")        # no w_a/dw
    except ValueError:
        guards = True
    g_e = bool(np.array_equal(r0.R0, r1.R0) and np.array_equal(r0.T0, r1.T0) and guards)
    ok = ok and g_e
    print("[gm] GATE E: zero gain fields -> R0/T0 ARRAY-EQUAL; missing w_a/dw raises -> {}"
          .format("PASS" if g_e else "FAIL"), flush=True)

    print("[gm] *** R20 FOUR-LEVEL GAIN MEDIUM: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
