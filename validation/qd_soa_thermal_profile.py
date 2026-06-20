"""QD-SOA spatially-resolved (z) thermal profile vs analytic oracles -- the reduced 1-D heat-
conduction upgrade to the lumped Rth/Cth self-heating, plus the per-slice gain coupling and the
thermal-FEM seam. thermal_profile_steady_1d solves the 1-D fin equation kappa A T'' - (T-T0)/Rth' =
-q(z) for T(z); QDGainModel.gain_per_m_thermal applies the per-slice red-shift + gain scale.

GATE A (lumped reduction): kappa A -> 0 (insulated ends) -> every slice relaxes to its local lumped
        value T(z) = T0 + q(z) Rth' (the lumped self-heating, now per slice).
GATE B (dome vs analytic): sunk facets (T=T0) + uniform q + finite conduction -> the DOME
        T(z) = T0 + q Rth'[1 - cosh((z-L/2)/Lc)/cosh(L/2Lc)], Lc = sqrt(kappa A Rth') -- the numerical
        solve matches the analytic cosh.
GATE C (non-uniform dissipation -> non-uniform T): a ramped q(z) gives a monotone T(z) following it.
GATE D (per-slice gain coupling): gain_per_m_thermal(T0) == gain_per_m_slices (T0 identity); a
        uniform-hot T(z) == the lumped set_temperature(T) gain (the second reduction); a hot DOME
        T(z) red-shifts + scales the gain locally so the hot region has LOWER gain.
GATE E (external-field sampler seam, STAND-IN lambda): sample_T_along_axis point-samples a STAND-IN
        T(x,y,z) (not a live FEM) onto the SOA axis -> the gain interface. The round-trip is exact by
        construction; this checks plumbing + gain passivity (finite, shape), not a live FEM solve.

Run: python -m validation.qd_soa_thermal_profile
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams, SelfHeating
from dynameta.optics.soa.thermal import (dome_analytic, sample_T_along_axis,
                                         thermal_profile_steady_1d, thermal_profile_transient_1d)


def main():
    print("[th] === QD-SOA spatially-resolved thermal profile vs oracles ===", flush=True)
    ok = True
    nz, L, T0 = 60, 1.0e-3, 300.0
    Rp, kA = 5.0e-4, 2.0e-5

    # ---- GATE A: lumped reduction (kappaA=0, insulated) ----
    dz = L / nz
    q = np.full(nz, 2.0e4)
    Tl = thermal_profile_steady_1d(q, dz, 0.0, Rp, T0, ends="insulated")
    relA = float(np.max(np.abs(Tl - (T0 + q * Rp))))
    # 2nd-order Neumann: uniform q + insulated -> uniform q Rth' for ANY conduction (flat-profile exact)
    Tflat = thermal_profile_steady_1d(q, dz, kA, Rp, T0, ends="insulated")
    relAc = float(np.max(np.abs(Tflat - (T0 + q * Rp))))
    g_a = bool(relA < 1e-9 and relAc < 1e-9)
    ok = ok and g_a
    print("[th] GATE A: kappaA=0 -> T = T0 + q Rth' lumped per slice (max|d| {:.1e} K, dT {:.2f}); "
          "uniform-q insulated flat for any kappaA ({:.1e}) -> {}".format(
              relA, Tl[nz // 2] - T0, relAc, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: dome vs analytic cosh (node grid so Dirichlet sits at the true facets) ----
    dzn = L / (nz - 1)
    zn = np.arange(nz) * dzn
    Tn = thermal_profile_steady_1d(q, dzn, kA, Rp, T0, ends="sunk")
    Tan = dome_analytic(q[0], L, kA, Rp, T0, zn)
    relB = float(np.max(np.abs(Tn - Tan)) / np.max(np.abs(Tan - T0)))
    g_b = bool(relB < 1e-2)
    ok = ok and g_b
    print("[th] GATE B: sunk-facet DOME == analytic cosh (peak dT {:.2f} K, max rel {:.1e}) -> "
          "{}".format(Tn.max() - T0, relB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: non-uniform dissipation -> monotone T(z) ----
    qr = np.linspace(1.0e4, 4.0e4, nz)
    Tr = thermal_profile_steady_1d(qr, dz, kA, Rp, T0, ends="insulated")
    g_c = bool(np.all(np.diff(Tr) > 0) and (Tr.max() - Tr.min()) > 1.0)
    ok = ok and g_c
    print("[th] GATE C: ramped q(z) -> monotone T(z) (dT span {:.1f} K, monotone {}) -> {}".format(
        Tr.max() - Tr.min(), bool(np.all(np.diff(Tr) > 0)), "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: per-slice gain coupling (T0 identity, uniform-hot reduction, dome) ----
    sh = SelfHeating(Rth_K_W=50.0, dnu0_dT_Hz_K=20e9, dg_dT_frac_per_K=-0.01, T0_K=300.0)
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus(), self_heating=sh)
    st = m.init_slices(nz, 40e-3)
    nu0 = m.p.nu0_Hz
    relD = float(np.max(np.abs(m.gain_per_m_thermal(st, nu0, np.full(nz, T0))
                                - m.gain_per_m_slices(st, nu0))))     # T0 identity (clean model)
    Thot = 330.0
    g_thermal_hot = m.gain_per_m_thermal(st, nu0, np.full(nz, Thot))  # uniform-hot, COLD-comb path
    m.set_temperature(Thot)                                           # mutate to the lumped single-T
    relD2 = float(np.max(np.abs(g_thermal_hot - m.gain_per_m_slices(st, nu0))))  # uniform == lumped
    m.set_temperature(T0)                                             # restore the clean model
    Tdome = thermal_profile_steady_1d(np.full(nz, 3.0e4), dzn, kA, Rp, T0, ends="sunk")  # node grid
    g_th = m.gain_per_m_thermal(st, nu0, Tdome)
    hot_lower = bool(g_th[nz // 2] < g_th[0])               # hot middle -> lower gain
    g_d = bool(relD < 1e-9 and relD2 < 1e-9 and hot_lower)
    ok = ok and g_d
    print("[th] GATE D: gain_per_m_thermal(T0)==gain_per_m_slices ({:.1e}); uniform-hot==set_temperature"
          " ({:.1e}); dome hot-mid {:.0f} < end {:.0f} /m -> {}".format(
              relD, relD2, g_th[nz // 2], g_th[0], "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: FEM seam (sample an external 3-D T-field onto the SOA axis) + passivity ----
    # T_at(x,y,z) stands in for a thermal-FEM ThermalResult.T_at; sample it along z (the SOA axis).
    def T_at(x, y, z):
        return T0 + 12.0 * np.sin(np.pi * z / L)               # an external dome along the stripe
    z_centers = (np.arange(nz) + 0.5) * dz
    T_fem = sample_T_along_axis(T_at, z_centers, axis="z")
    direct = np.array([T_at(0.0, 0.0, zc) for zc in z_centers])
    g_fem = m.gain_per_m_thermal(st, nu0, T_fem)
    g_e = bool(np.max(np.abs(T_fem - direct)) < 1e-12 and np.all(np.isfinite(g_fem))
               and g_fem.shape == (nz,))
    ok = ok and g_e
    print("[th] GATE E: sample_T_along_axis(STAND-IN T(x,y,z) lambda, not a live FEM) -> gain "
          "interface (round-trip {:.1e} exact-by-construction, finite {}) -> {}".format(
              np.max(np.abs(T_fem - direct)), bool(np.all(np.isfinite(g_fem))),
              "PASS" if g_e else "FAIL"), flush=True)

    # ---- GATE F: ES-band thermal coupling (Phase 22) ----
    me = QDGainModel(QDGainParams(n_groups=15, sigma_pk_ES_m2=1e-19).with_detailed_balance_taus(),
                     self_heating=sh)
    ste = me.init_slices(nz, 40e-3)
    relF0 = float(np.max(np.abs(me.gain_per_m_thermal(ste, nu0, np.full(nz, T0))
                                - me.gain_per_m_slices(ste, nu0))))      # T0 identity incl ES
    g_hot = me.gain_per_m_thermal(ste, nu0, np.full(nz, 330.0))
    me.set_temperature(330.0)
    relFh = float(np.max(np.abs(g_hot - me.gain_per_m_slices(ste, nu0))))  # uniform-hot == set_temp
    me.set_temperature(T0)
    nuES = me.nu_ES_j[len(me.nu_ES_j) // 2]                              # ES band contributes (>0)
    es_present = bool(me.gain_per_m_thermal(ste, nuES, np.full(nz, T0))[0] != 0.0)
    g_f = bool(relF0 < 1e-9 and relFh < 1e-9 and es_present)
    ok = ok and g_f
    print("[th] GATE F: ES-band thermal: T0 identity ({:.1e}) + uniform-hot==set_temperature ({:.1e}) "
          "+ ES band present -> {}".format(relF0, relFh, "PASS" if g_f else "FAIL"), flush=True)

    # ---- GATE G: transient 1-D thermal (Phase 23) ----
    Cline, q_u = 1.0e-3, np.full(nz, 2.0e4)
    tau = Cline * Rp
    Ttr = thermal_profile_transient_1d(q_u, dz, kA, Rp, T0, Cline, tau / 50, 4000, ends="sunk")
    relG_steady = float(np.max(np.abs(Ttr - thermal_profile_steady_1d(q_u, dz, kA, Rp, T0,
                                                                      ends="sunk"))))
    H = thermal_profile_transient_1d(q_u, dz, 0.0, Rp, T0, Cline, tau / 200, 400, ends="insulated",
                                     return_history=True)
    tg = np.arange(H.shape[0]) * (tau / 200)
    Tan = T0 + q_u[0] * Rp * (1.0 - np.exp(-tg / tau))                   # lumped RC charge-up
    relG_rc = float(np.max(np.abs(H[:, nz // 2] - Tan)) / (q_u[0] * Rp))
    g_g = bool(relG_steady < 1e-6 and relG_rc < 1e-2)
    ok = ok and g_g
    print("[th] GATE G: transient t->inf == steady ({:.1e} K); lumped RC == T0+qRth(1-e^-t/tau) (rel "
          "{:.1e}) -> {}".format(relG_steady, relG_rc, "PASS" if g_g else "FAIL"), flush=True)

    print("[th] *** QD-SOA THERMAL PROFILE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
