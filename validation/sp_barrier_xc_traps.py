"""Validate the v0.16 options of the dual-gate Schrodinger-Poisson solver (carriers/sp_dual_gate.py) and
report their size on the 6 nm generic gated film.

Options: finite oxide barriers (wavefunction penetration into the oxide, BenDaniel-Duke mass step, the
oxide's electron charge in Poisson, Gauss matching plane at the end of the penetration region); LDA
exchange-correlation in the ground state (exchange, exchange + Perdew-Zunger 1981 correlation, effective
atomic units of the film); interface traps (physics_traps.FilmInterfaceTraps, occupancy at the
electrostatic potential of the interface plane).

Gates (independent references; the full set with brute-force whole-stack oracles is
tests/test_sp_barrier_xc_traps.py):
  A  analytic finite well with a mass step: the discrete BenDaniel-Duke interface operator converges at
     second order (error ratio per halving in [3.7, 4.3]);
  B  infinite-barrier limit: the barrier solution converges to the hard-wall solver, as 1/V_b once
     kappa h >> 1 (ratio per decade in [8, 12]);
  C  Gauss's law at the matching planes (re-derived) and the exact discrete charge balance (2e-9 C/m^2);
  D  exchange-correlation leaves the flat-band bulk density at n_bg (30 nm film, centre within 2e-3);
  E  small-signal gate efficiency with traps = C_ox / (C_ox + C_s + q^2 D_it) (1e-6 relative);
  F  options off = the existing code path, bit for bit.
Tables (generic film: 6 nm, n_bg 3e26 m^-3, m* 0.30 m_e, eps_s 9.0; 8 nm oxides of eps 20 on both faces;
481 film nodes, 300 K):
  1  finite barriers 1.5 / 2.5 / 3.5 eV, oxide mass 0.2 / 0.3 / 0.4 m_e (automatic penetration);
  2  exchange-correlation (x, LDA; static and high-frequency screening);
  3  interface traps D_it 1e16 .. 1e18 m^-2 eV^-1 and the D_it that reduces the free sheet-charge swing
     (-4 V .. +4 V) to a given fraction of its trap-free value, tabulated for SWING_FRACTIONS (0.9 to 0.5).

Run: python -m validation.sp_barrier_xc_traps            (about 2-4 min on a loaded box)
"""
import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.physics_traps import FilmInterfaceTraps  # noqa: E402
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D  # noqa: E402
from dynameta.carriers.sp_dual_gate import DualGateSchrodingerPoisson, GateStack, OxideBarrier  # noqa: E402
from dynameta.constants import EPS0, HBAR, KB, M_E, Q_E  # noqa: E402
from dynameta.core.numerics import trapz  # noqa: E402

M = 0.30 * M_E          # the generic gated film
NBG = 3e26
EPS_S = 9.0
T_FILM = 6e-9
T_OX, EPS_OX = 8e-9, 20.0
NZ = 481
FAILS = []


def gate(name, ok, detail):
    print("  [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


def solver(barrier=None, traps=None, xc=None, nz=NZ, t=T_FILM, **kw):
    g = GateStack((T_OX, EPS_OX), barrier=barrier, traps=traps)
    return DualGateSchrodingerPoisson(t_film_m=t, bottom=g, top=g, n_bg_m3=NBG, m_eff_kg=M, eps_static=EPS_S,
                                      nz=nz, xc=xc, **kw)


def solve_flagged(s, V):
    """Solve and return (solution, flag): 'T' when the solver reports the tunnelling regime, 'S' when the
    tail is not converged at the far wall, '' otherwise."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sol = s.solve_voltages(*V)
    msgs = " ".join(str(r.message) for r in rec)
    return sol, ("T" if "tunnelling" in msgs else ("S" if "penetration_m for" in msgs else ""))


# -------------------------------------------------------------------------------------------------
# gates
# -------------------------------------------------------------------------------------------------
def gate_A():
    from scipy.optimize import brentq
    L, V0, mw, mb = 5e-9, 1.0 * Q_E, M, 0.2 * M_E

    def g(E, par):
        k = math.sqrt(2.0 * mw * E) / HBAR
        kap = math.sqrt(2.0 * mb * (V0 - E)) / HBAR
        x = 0.5 * k * L
        return (k * math.sin(x) / mw - kap * math.cos(x) / mb if par == 0
                else -k * math.cos(x) / mw - kap * math.sin(x) / mb)
    Es = np.linspace(1e-6 * V0, V0 * (1 - 1e-9), 4001)
    ex = []
    for par in (0, 1):
        v = [g(E, par) for E in Es]
        ex += [brentq(g, Es[i], Es[i + 1], args=(par,), xtol=1e-40, rtol=1e-15)
               for i in range(len(Es) - 1) if v[i] * v[i + 1] < 0.0]
    ex = np.array(sorted(ex)[:3])
    errs = []
    for nzf in (101, 201, 401):
        h = L / (nzf - 1)
        nb = int(round(6e-9 / h))
        z = h * np.arange(-nb, nzf + nb)
        inv = np.full(z.size - 1, 1.0 / mw)
        inv[:nb] = inv[nb + nzf - 1:] = 1.0 / mb
        U = np.zeros(z.size)
        U[:nb] = U[nb + nzf:] = V0
        U[nb] = U[nb + nzf - 1] = 0.5 * V0
        E, _, _ = SchrodingerPoisson1D(z, mw).solve_schrodinger(U, inv_mass_cell_per_kg=inv, n_states=3)
        errs.append(np.abs(E - ex))
    errs = np.array(errs)
    r = errs[:-1] / errs[1:]
    gate("A finite well (BenDaniel-Duke mass step) second order", bool(np.all((r > 3.7) & (r < 4.3))),
         "levels {} meV, error at h=0.0125 nm {} ueV, ratios {}".format(
             np.round(ex / Q_E * 1e3, 4).tolist(), np.round(errs[-1] / Q_E * 1e6, 3).tolist(),
             np.round(r.ravel(), 3).tolist()))


def gate_B():
    bot = GateStack(((1e-9, 3.9), (6e-9, 22.0)), work_function_offset_V=0.3, interface_charge_C_m2=-5e-3)
    top = GateStack((8e-9, 20.0), work_function_offset_V=-0.2, interface_charge_C_m2=2e-3)
    mk = lambda b: DualGateSchrodingerPoisson(                                          # noqa: E731
        t_film_m=T_FILM, nz=241, bottom=GateStack(bot.layers, 0.3, -5e-3, b), top=GateStack(top.layers, -0.2, 2e-3, b),
        n_bg_m3=NBG, m_eff_kg=M, eps_static=EPS_S)
    ref = mk(None).solve_voltages(1.0, -1.0)
    rows = []
    for Vb in (10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        sol, _ = solve_flagged(mk(OxideBarrier(Vb, M, penetration_m=0.5e-9)), (1.0, -1.0))
        rows.append((Vb, max(abs(sol.phi_bottom_V - ref.phi_bottom_V), abs(sol.phi_top_V - ref.phi_top_V)),
                     np.max(np.abs(sol.n_m3 - ref.n_m3)) / NBG))
    for Vb, d, dn in rows:
        print("      V_b = {:8.0e} eV: max |d phi_face| = {:.3e} V, max |dn| = {:.3e} n_bg".format(Vb, d, dn))
    r_lo = rows[0][1] / rows[1][1]
    r_hi = [rows[k][1] / rows[k + 1][1] for k in (3, 4)]
    gate("B infinite-barrier limit -> hard wall", 2.8 < r_lo < 3.8 and all(8 < x < 12 for x in r_hi)
         and rows[-1][1] < 1e-5, "ratio per decade {:.2f} (10->100 eV, V_b^-1/2 regime), {} (1e4->1e6 eV, "
         "1/V_b regime)".format(r_lo, np.round(r_hi, 2).tolist()))


def gate_C():
    b = OxideBarrier((3.0, 1.6), (0.4 * M_E, 0.2 * M_E), penetration_m=2e-9)
    tr = FilmInterfaceTraps(2e17, -0.6, 1.2, 0.7, levels=((0.9, 3e16),))
    bot = GateStack(((1e-9, 3.9), (6e-9, 22.0)), 0.3, -5e-3, b, tr)
    top = GateStack((8e-9, 20.0), -0.2, 2e-3, OxideBarrier(2.5, 0.3 * M_E, penetration_m=2e-9), tr)
    s = DualGateSchrodingerPoisson(t_film_m=T_FILM, nz=241, bottom=bot, top=top, n_bg_m3=NBG, m_eff_kg=M, eps_static=EPS_S,
                                   xc="lda")
    Cb, Ct = 22.0 * EPS0 / 5e-9, 20.0 * EPS0 / 6e-9
    worst = [0.0, 0.0]
    for V in ((0.0, 0.0), (1.5, -1.0), (-2.0, 1.0)):
        sol = s.solve_voltages(*V)
        z, phi, n = sol.z_ext_m, sol.phi_ext_V, sol.n_ext_m3
        h = z[1] - z[0]
        Rb = Cb * (phi[0] - (V[0] - 0.3)) - 22.0 * EPS0 * (phi[1] - phi[0]) / h
        Rt = 20.0 * EPS0 * (phi[-1] - phi[-2]) / h + Ct * (phi[-1] - (V[1] + 0.2))
        Nd = np.where((z > -1e-15) & (z < T_FILM + 1e-15), NBG, 0.0)
        Nd[np.argmin(np.abs(z))] *= 0.5
        Nd[np.argmin(np.abs(z - T_FILM))] *= 0.5
        tot = (Cb * (V[0] - 0.3 - phi[0]) + Ct * (V[1] + 0.2 - phi[-1]) - 3e-3 + sol.trap_charge_bottom_C_m2
               + sol.trap_charge_top_C_m2 + Q_E * h * float(np.sum(Nd - n)))
        worst = [max(worst[0], abs(Rb), abs(Rt)), max(worst[1], abs(tot))]
    gate("C matching-plane Gauss law and charge balance (barrier + traps + LDA)",
         worst[0] < 1e-9 and worst[1] < 2e-9, "max |R| = {:.2e}, balance {:.2e} C/m^2".format(*worst))


def gate_D():
    s0 = solver(t=30e-9, nz=601)
    s1 = solver(t=30e-9, nz=601, xc="lda")
    a, b = s0.solve_voltages(0.0, 0.0), s1.solve_voltages(0.0, 0.0)
    gate("D xc keeps the flat-band bulk at n_bg", abs(b.n_m3[300] / NBG - 1) < 2e-3 and abs(b.phi_V[300]) < 1e-3,
         "centre n/n_bg - 1 = {:+.2e} (no xc {:+.2e}), centre phi {:+.2e} V".format(
             b.n_m3[300] / NBG - 1, a.n_m3[300] / NBG - 1, b.phi_V[300]))


def gate_E():
    tol = dict(gauss_tol_C_m2=1e-12, inner_tol_V=1e-12)
    s0 = solver(**tol)
    p = [s0.solve_voltages(0.0, 1.0 + k * 1e-3) for k in (-1, 0, 1)]
    eff0 = (p[2].phi_top_V - p[0].phi_top_V) / 2e-3
    Ct = EPS_OX * EPS0 / T_OX
    Cs = Ct * (1 - eff0) / eff0
    e0 = (s0.E_F_J + Q_E * p[1].phi_top_V) / Q_E
    worst = 0.0
    for D in (1e16, 1e17, 1e18):
        g0 = GateStack((T_OX, EPS_OX))
        g1 = GateStack((T_OX, EPS_OX), traps=FilmInterfaceTraps(D, e0 - 1.0, e0 + 1.0, e0))
        s = DualGateSchrodingerPoisson(t_film_m=T_FILM, bottom=g0, top=g1, n_bg_m3=NBG, m_eff_kg=M, eps_static=EPS_S, nz=NZ,
                                       **tol)
        q = [s.solve_voltages(0.0, 1.0 + k * 1e-3) for k in (-1, 0, 1)]
        eff = (q[2].phi_top_V - q[0].phi_top_V) / 2e-3
        worst = max(worst, abs(eff / (Ct / (Ct + Cs + Q_E * D)) - 1.0))
    gate("E small-signal efficiency C_ox/(C_ox + C_s + q^2 D_it)", worst < 1e-6,
         "max relative error {:.1e} (C_ox {:.4e}, C_s {:.4e} F/m^2 at (0, +1) V)".format(worst, Ct, Cs))


def gate_F():
    a = solver(nz=241).solve_voltages(1.0, -2.0)
    b = solver(OxideBarrier(2.5, 0.3 * M_E, penetration_m=0.0), FilmInterfaceTraps(), "none",
               nz=241).solve_voltages(1.0, -2.0)
    gate("F options off = existing path, bit for bit", np.array_equal(a.phi_V, b.phi_V)
         and np.array_equal(a.n_m3, b.n_m3), "np.array_equal on phi and n")


# -------------------------------------------------------------------------------------------------
# tables
# -------------------------------------------------------------------------------------------------
def _centroid(sol, sol0, side):
    """Distance from the interface plane (positive into the film) of the charge induced relative to 0 V,
    over the half of the domain on that side (penetration region included)."""
    z = sol.z_ext_m if sol.z_ext_m is not None else sol.z_m
    n = sol.n_ext_m3 if sol.n_ext_m3 is not None else sol.n_m3
    n0 = sol0.n_ext_m3 if sol0.n_ext_m3 is not None else sol0.n_m3
    dn = n - n0
    if side == 0:
        w, d = z <= 0.5 * T_FILM, z
    else:
        w, d = z >= 0.5 * T_FILM, T_FILM - z
    return trapz(d[w] * dn[w], z[w]) / trapz(dn[w], z[w])


def _peak_depth(sol):
    i = int(np.argmax(sol.n_m3))
    return min(sol.z_m[i], T_FILM - sol.z_m[i])


def _edge_depth(sol):
    """Depth below the TOP interface at which the density first reaches n_bg / 2 (depletion edge),
    linear interpolation between nodes."""
    z, n = T_FILM - sol.z_m[::-1], sol.n_m3[::-1]
    k = int(np.flatnonzero(n >= 0.5 * NBG)[0])
    if k == 0:
        return 0.0
    return float(z[k - 1] + (0.5 * NBG - n[k - 1]) * (z[k] - z[k - 1]) / (n[k] - n[k - 1]))


def table_1():
    print("\nTABLE 1 -- finite barriers (both oxides 8 nm / eps 20, automatic penetration 'pen'). One block per "
          "bias: 'acc1' = gates (+1, +1) V, 'acc' = (+2, +2) V, 'dep' = (-2, -2) V (symmetric: both interfaces "
          "alike, top interface listed), 'asym' = (+2, -2) V (bottom accumulated, top depleted; both "
          "interfaces listed). Deltas are barrier minus hard wall at the same bias. dzc: shift of the "
          "centroid of the gate-induced charge (relative to the same model at 0 V), measured from the "
          "interface plane into the film [nm] (negative = toward / into the oxide); dC: change of the "
          "gate capacitance dQ_t/dV_t (bottom gate fixed); dQ: change of the gate charge at fixed V (= the "
          "gate-induced sheet charge); ox: electrons in the penetration region / electrons in the film; "
          "n(0): density at the interface plane / n_bg (0 for the hard wall); dpk: change of the depth of "
          "the density maximum from the nearest interface [nm] (dep: change of the depletion edge, the depth "
          "at which n reaches n_bg / 2 -- the depleted profile has no sharp maximum). Rows '1.5/0.3 @2nm' and "
          "'@3nm' repeat 1.5 eV / 0.3 m_e with a penetration fixed at 2 / 3 nm (flag S: tail not decayed at "
          "the wall). Flag T = tunnelling regime (the oxide band falls below E_F + 5 kT inside the "
          "penetration region: the closed one-Fermi-level model is not valid there -- numbers shown for "
          "completeness only).", flush=True)
    hard = solver()
    biases = {"acc1": (1.0, 1.0), "acc": (2.0, 2.0), "dep": (-2.0, -2.0), "asym": (2.0, -2.0)}
    ref = {k: hard.solve_voltages(*V) for k, V in biases.items()}
    ref0 = hard.solve_voltages(0.0, 0.0)

    def cap(s, V):
        a, _ = solve_flagged(s, (V[0], V[1] - 0.01))
        b, _ = solve_flagged(s, (V[0], V[1] + 0.01))
        return (b.gate_charge_top_C_m2 - a.gate_charge_top_C_m2) / 0.02
    Cref = {k: cap(hard, biases[k]) for k in ("acc1", "acc", "dep")}
    for k in ("acc1", "acc", "dep"):
        print("  hard wall {}: Q_g {:.5e} C/m^2, C_t {:.5e} F/m^2, induced-charge centroid {:.4f} nm, density "
              "maximum at {:.4f} nm, n_bg/2 edge at {:.4f} nm".format(
                  k, ref[k].gate_charge_top_C_m2, Cref[k], _centroid(ref[k], ref0, 1) * 1e9,
                  _peak_depth(ref[k]) * 1e9, _edge_depth(ref[k]) * 1e9))
    print("  hard wall 0 V: Q_g {:.5e} C/m^2, face potential {:.4f} V".format(ref0.gate_charge_top_C_m2,
                                                                            ref0.phi_top_V))
    print("  hard wall asym: induced-charge centroid bottom {:.4f} / top {:.4f} nm".format(
        _centroid(ref["asym"], ref0, 0) * 1e9, _centroid(ref["asym"], ref0, 1) * 1e9))
    rows = {}
    cases = [(Vb, mo, None) for Vb in (1.5, 2.5, 3.5) for mo in (0.2, 0.3, 0.4)]
    cases[3:3] = [(1.5, 0.3, 2e-9), (1.5, 0.3, 3e-9)]
    for Vb, mo, pen in cases:
        if True:
            s = solver(OxideBarrier(Vb, mo * M_E, penetration_m=pen))
            s0, _ = solve_flagged(s, (0.0, 0.0))
            r = {"pen": s.penetration_m[0] * 1e9, "q0": s0.gate_charge_top_C_m2, "phi0": s0.phi_top_V}
            for k, V in biases.items():
                sol, f = solve_flagged(s, V)
                ne = -sol.film_charge_C_m2 / Q_E + NBG * T_FILM
                d = dict(flag=f or "-", ox=(-sol.oxide_electron_charge_top_C_m2 / Q_E) / ne,
                         oxb=(-sol.oxide_electron_charge_bottom_C_m2 / Q_E) / ne,
                         dzc=(_centroid(sol, s0, 1) - _centroid(ref[k], ref0, 1)) * 1e9,
                         dzcb=(_centroid(sol, s0, 0) - _centroid(ref[k], ref0, 0)) * 1e9,
                         dQ=sol.gate_charge_top_C_m2 / ref[k].gate_charge_top_C_m2 - 1.0,
                         dQb=sol.gate_charge_bottom_C_m2 / ref[k].gate_charge_bottom_C_m2 - 1.0,
                         n0=sol.n_m3[-1] / NBG, n0b=sol.n_m3[0] / NBG,
                         dpk=((_edge_depth(sol) - _edge_depth(ref[k])) if k == "dep"
                              else (_peak_depth(sol) - _peak_depth(ref[k]))) * 1e9)
                if k != "asym":
                    d["dC"] = cap(s, V) / Cref[k] - 1.0
                r[k] = d
            rows[(Vb, mo, pen)] = r
    for k in ("acc1", "acc", "dep"):
        print("  [{}] gates {} V".format(k, biases[k]))
        print("    {:>9} {:>6} {:>8} {:>8} {:>8} {:>9} {:>7} {:>8} {:>3}".format(
            "V_b/m_ox", "pen", "dzc", "dC", "dQ", "ox", "n(0)", "dpk", "fl"))
        for (Vb, mo, pen), r in rows.items():
            d = r[k]
            print("    {:>9} {:6.2f} {:+8.4f} {:+8.2%} {:+8.2%} {:9.2e} {:7.3f} {:+8.4f} {:>3}".format(
                _lab(Vb, mo, pen), r["pen"], d["dzc"], d["dC"], d["dQ"], d["ox"], d["n0"], d["dpk"], d["flag"]), flush=True)
    print("  [asym] gates (+2, -2) V: bottom interface accumulated, top depleted")
    print("    {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>3}".format(
        "V_b/m_ox", "dzc bot", "dQ bot", "ox bot", "dzc top", "dQ top", "ox top", "fl"))
    for (Vb, mo, pen), r in rows.items():
        d = r["asym"]
        print("    {:>9} {:+9.4f} {:+9.2%} {:9.2e} {:+9.4f} {:+9.2%} {:9.2e} {:>3}".format(
            _lab(Vb, mo, pen), d["dzcb"], d["dQb"], d["oxb"], d["dzc"], d["dQ"], d["ox"], d["flag"]), flush=True)
    print("  zero bias (barrier): Q_g(0 V) / face potential -- the dead-layer charge the barrier removes:")
    for (Vb, mo, pen), r in rows.items():
        print("    {:>9}: Q_g(0 V) {:.4e} C/m^2 (hard {:.4e}), phi_face(0 V) {:.4f} V (hard {:.4f})".format(
            _lab(Vb, mo, pen), r["q0"], ref0.gate_charge_top_C_m2, r["phi0"], ref0.phi_top_V))
    return rows


def _lab(Vb, mo, pen):
    return "{:.1f}/{:.1f}".format(Vb, mo) + ("" if pen is None else " @{:.0f}nm".format(pen * 1e9))


def table_2():
    print("\nTABLE 2 -- exchange-correlation in the ground state (hard walls; and with a 2.5 eV / 0.3 m_e "
          "barrier). E_i - E_F of the four lowest sub-bands at 0 V and (+2, +2) V [meV]; peak n / n_bg and its "
          "depth at +2 V; gate charge Q_t [mC/m^2] versus tied gate voltage. eps_xc = 9.0 (static, default) "
          "or 4.0 (high-frequency screening). Flags per bias (T = tunnelling regime, invalid). Q_t changes are "
          "relative to 'none' (xc rows) and to 'bar' (bar+lda).", flush=True)
    cases = [("none", None, None, None), ("x", "x", None, None), ("lda", "lda", None, None),
             ("lda eps4.0", "lda", 4.0, None), ("bar", None, None, OxideBarrier(2.5, 0.3 * M_E)),
             ("bar+lda", "lda", None, OxideBarrier(2.5, 0.3 * M_E))]
    Vs = (-4.0, -2.0, 0.0, 2.0, 4.0)
    base = None
    for name, xc, ex, b in cases:
        s = solver(b, None, xc, xc_eps_r=ex)
        a0, _ = solve_flagged(s, (0.0, 0.0))
        a2, _ = solve_flagged(s, (2.0, 2.0))
        Es0 = (a0.subbands.energies_J[:4] - s.E_F_J) / Q_E * 1e3
        Es2 = (a2.subbands.energies_J[:4] - s.E_F_J) / Q_E * 1e3
        Q, fl = [], []
        for V in Vs:
            sol, f = solve_flagged(s, (V, V))
            Q.append(sol.gate_charge_top_C_m2 * 1e3)
            fl.append(f or "-")
        if base is None or name == "bar":
            base = Q
        print("  {:<11} E(0V) {}  E(+2V) {}  peak {:.4f} n_bg at {:.4f} nm | Q_t {} | vs base {} {}".format(
            name, np.round(Es0, 1).tolist(), np.round(Es2, 1).tolist(), a2.n_m3.max() / NBG,
            _peak_depth(a2) * 1e9, np.round(Q, 4).tolist(),
            ["{:+.2%}".format(q / q0 - 1) for q, q0 in zip(Q, base)], "".join(fl)), flush=True)


SWING_FRACTIONS = (0.9, 0.8, 0.7, 0.6, 0.5)      # swing fractions of the D_it table (TABLE 3)


def table_3():
    print("\nTABLE 3 -- interface traps on both faces (hard walls): uniform D_it over E_0 +- 1 eV, neutral "
          "level E_0 = E_F - E_c of the flat-band bulk, split acceptor/donor. Config 'top': top "
          "gate swept, bottom at 0 V; 'tied': both gates at V. eff_ss = d phi_s,top / dV at 0 V (small "
          "signal); eff_ls = (phi_s(+4) - phi_s(-4)) / 8 V; swing = free-electron sheet excess N(+4 V) - "
          "N(-4 V) relative to the trap-free swing; Q_it(+4) the trapped charge at +4 V [mC/m^2]. 'tied full' = "
          "tied gates with a 3.5 eV / 0.3 m_e barrier on both faces and LDA xc (swing = film + oxide electrons).",
          flush=True)
    s_ref = solver()
    E0 = s_ref.E_F_J / Q_E
    print("  E_0 = E_F - E_c = {:.4f} eV".format(E0), flush=True)

    def run(D, cfg):
        tr = None if D == 0.0 else FilmInterfaceTraps(D, E0 - 1.0, E0 + 1.0, E0)
        full = cfg.endswith("full")
        s = solver(OxideBarrier(3.5, 0.3 * M_E) if full else None, tr, "lda" if full else None)
        V = (lambda v: (0.0, v)) if cfg == "top" else (lambda v: (v, v))
        a, b = s.solve_voltages(*V(-0.01)), s.solve_voltages(*V(0.01))
        lo, hi = s.solve_voltages(*V(-4.0)), s.solve_voltages(*V(4.0))
        return dict(eff_ss=(b.phi_top_V - a.phi_top_V) / 0.02, eff_ls=(hi.phi_top_V - lo.phi_top_V) / 8.0,
                    swing=hi.sheet_excess_total_m2 - lo.sheet_excess_total_m2, qit=hi.trap_charge_top_C_m2 * 1e3)
    Ds = (0.0, 1e16, 2e16, 5e16, 1e17, 2e17, 5e17, 1e18)
    res = {}
    for cfg in ("top", "tied", "tied full"):
        res[cfg] = {D: run(D, cfg) for D in Ds}
        s0 = res[cfg][0.0]["swing"]
        print("  config {}: trap-free swing {:.4e} m^-2".format(cfg, s0))
        print("    {:>8} {:>8} {:>8} {:>8} {:>9}".format("D_it", "eff_ss", "eff_ls", "swing", "Q_it(+4)"))
        for D in Ds:
            r = res[cfg][D]
            print("    {:8.1e} {:8.4f} {:8.4f} {:8.4f} {:9.3f}".format(D, r["eff_ss"], r["eff_ls"], r["swing"] / s0,
                                                                       r["qit"]), flush=True)
        # D_it for each swing fraction: log-linear interpolation, then a check solve
        lr = np.array([res[cfg][D]["swing"] / s0 for D in Ds[1:]])
        lD = np.log(np.array(Ds[1:]))
        for target in SWING_FRACTIONS:
            k = int(np.flatnonzero(lr < target)[0])
            Dt = float(np.exp(np.interp(target, [lr[k], lr[k - 1]], [lD[k], lD[k - 1]])))
            chk = run(Dt, cfg)["swing"] / s0
            print("    swing {:.2f} of trap-free at D_it ~ {:.3e} m^-2 eV^-1 ({:.2e} cm^-2 eV^-1; check: {:.4f})".format(
                target, Dt, Dt * 1e-4, chk), flush=True)
    return res


def main():
    t0 = time.perf_counter()
    print("[t] gates", flush=True)
    for g in (gate_A, gate_B, gate_C, gate_D, gate_E, gate_F):
        g()
    table_1()
    table_2()
    table_3()
    print("\n[t] {:.0f} s".format(time.perf_counter() - t0))
    if FAILS:
        print("FAIL: {}".format(", ".join(FAILS)))
        sys.exit(1)
    print("PASS (6 gates)")


if __name__ == "__main__":
    main()
