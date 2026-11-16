"""Validate the dual-gate Schrodinger-Poisson carrier solver (carriers.sp_dual_gate.DualGateSchrodingerPoisson): a thin
degenerate film gated through oxide stacks on BOTH faces, its Fermi level the grounded reference, each face potential
fixed by Gauss's law at that face (the gates mirror tests/test_sp_dual_gate.py, which also holds the brute-force
meshed-oxide oracle).

Film (the generic gated film): 6 nm, n_bg 3e26 m^-3, m* 0.30 m_e, eps_s 9.0, 300 K, nz 481. Asymmetric stacks: bottom
4 nm eps 3.9 + 6 nm eps 22, work-function offset +0.3 V, interface charge -5e-3 C/m^2; top 8 nm eps 20, offset -0.2 V,
charge +2e-3 C/m^2.

  A  Gauss's law at both faces, RE-DERIVED here from the returned profile: half-cell flux of the end node (budget
     1e-9 C/m^2, the solver's stop) and an independent one-sided second-order difference (2e-6 C/m^2, O(h^2));
     charge balance Q_gate_b + Q_gate_t + sigma_b + sigma_t + q int (N_d - n) dz = 0 (2e-9 C/m^2); six bias pairs
     of both signs (-4 .. +3 V), hard walls (n = 0) at both faces.
  B  symmetry: symmetric stacks and biases give a mirror-symmetric profile; exchanging the two gates (stacks, offsets,
     charges, voltages) mirrors it, z -> t - z (1e-9 V, 1e-9 n_bg, outer stop tightened to 1e-11 C/m^2 as in the test:
     the default 1e-9 C/m^2 stop leaves ~1e-9 of path dependence); the asymmetric profile is > 0.1 V from its own
     mirror image (the check is not vacuous).
  C  grid convergence at (+2, -1) V, nz 241 / 481 / 961 / 1921: successive differences of both face potentials and of
     the film charge shrink by 3..5 (second order); finest face-potential difference < 2e-5 V.
  D  single-gate limit, 20 nm film: a far gate behind 10 um of eps 1 and no far gate agree (1e-6 V), and the top face
     reproduces the single-gate solver (SchrodingerPoissonCarrier, face-field map, Neumann body, same top oxide) at
     +-2 V: face potential within 5e-5 V, density over the top 5 nm within 3e-4 n_bg.
  E  signs: raising one gate raises its face potential and the film's electron sheet (both gates, both faces).

Run: python -m validation.sp_dual_gate     (pure numpy/scipy; no DEVSIM, no NGSolve)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dynameta.carriers.sp_dual_gate import DualGateSchrodingerPoisson, GateStack   # noqa: E402
from dynameta.constants import EPS0, M_E, Q_E                                          # noqa: E402
from validation._harness import Harness                                                # noqa: E402

MSTAR = 0.30 * M_E          # the generic gated film
NBG = 3e26
EPS_S = 9.0
BIASES = ((0.0, 0.0), (2.0, -1.0), (-3.0, 2.0), (-4.0, -4.0), (3.0, 3.0), (1.5, 0.0))


def asym_gates():
    bot = GateStack(((4e-9, 3.9), (6e-9, 22.0)), work_function_offset_V=0.3, interface_charge_C_m2=-5e-3)
    top = GateStack(((8e-9, 20.0),), work_function_offset_V=-0.2, interface_charge_C_m2=2e-3)
    return bot, top


def solver(bot, top, **kw):
    kw.setdefault("t_film_m", 6e-9)
    kw.setdefault("nz", 481)
    return DualGateSchrodingerPoisson(bottom=bot, top=top, n_bg_m3=NBG, m_eff_kg=MSTAR, eps_static=EPS_S, T_K=300.0,
                                      **kw)


def face_laws(bot, top, sol):
    """(R_b, R_t) with the half-cell flux and with the one-sided second-order flux [C/m^2], written out here."""
    phi, n, z = sol.phi_V, sol.n_m3, sol.z_m
    h = z[1] - z[0]
    ee = EPS_S * EPS0
    Fb = ee * (phi[1] - phi[0]) / h + Q_E * 0.5 * h * (NBG - n[0])
    Ft = ee * (phi[-1] - phi[-2]) / h - Q_E * 0.5 * h * (NBG - n[-1])
    Fb1 = ee * (-3.0 * phi[0] + 4.0 * phi[1] - phi[2]) / (2.0 * h)
    Ft1 = ee * (3.0 * phi[-1] - 4.0 * phi[-2] + phi[-3]) / (2.0 * h)
    Cb = 1.0 / sum(t / (EPS0 * e) for t, e in bot.layers)
    Ct = 1.0 / sum(t / (EPS0 * e) for t, e in top.layers)
    out = []
    for F_b, F_t in ((Fb, Ft), (Fb1, Ft1)):
        Rb = Cb * (sol.phi_bottom_V - (sol.V_bottom - bot.work_function_offset_V)) - F_b - bot.interface_charge_C_m2
        Rt = F_t + Ct * (sol.phi_top_V - (sol.V_top - top.work_function_offset_V)) - top.interface_charge_C_m2
        out.append((Rb, Rt))
    return out, Cb, Ct


def gate_A(H):
    bot, top = asym_gates()
    s = solver(bot, top)
    w_half = w_one = w_bal = 0.0
    walls = True
    print("[sp_dual_gate] bias (Vb, Vt) | phi_b, phi_t [V] | gate charges b, t [C/m^2] | sheet excess [m^-2]")
    for V in BIASES:
        sol = s.solve_voltages(*V)
        ((Rb, Rt), (Rb1, Rt1)), Cb, Ct = face_laws(bot, top, sol)
        w_half = max(w_half, abs(Rb), abs(Rt))
        w_one = max(w_one, abs(Rb1), abs(Rt1))
        Qgb = Cb * (V[0] - bot.work_function_offset_V - sol.phi_bottom_V)
        Qgt = Ct * (V[1] - top.work_function_offset_V - sol.phi_top_V)
        y = NBG - sol.n_m3
        Qf = Q_E * 0.5 * float(np.sum((y[1:] + y[:-1]) * np.diff(sol.z_m)))       # trapezoid written out
        w_bal = max(w_bal, abs(Qgb + Qgt + bot.interface_charge_C_m2 + top.interface_charge_C_m2 + Qf))
        walls = walls and sol.n_m3[0] == 0.0 and sol.n_m3[-1] == 0.0
        print("[sp_dual_gate]   ({:+.1f}, {:+.1f}) | {:+.6f}, {:+.6f} | {:+.4e}, {:+.4e} | {:+.4e}".format(
            V[0], V[1], sol.phi_bottom_V, sol.phi_top_V, Qgb, Qgt, sol.sheet_excess_m2), flush=True)
    H.gate("A1 Gauss at both faces, half-cell flux re-derived (1e-9 C/m^2)", w_half < 1e-9, "worst {:.2e}".format(w_half))
    H.gate("A2 Gauss, independent one-sided flux (2e-6 C/m^2, O(h^2))", w_one < 2e-6, "worst {:.2e}".format(w_one))
    H.gate("A3 charge balance (2e-9 C/m^2) and hard walls", w_bal < 2e-9 and walls,
           "worst {:.2e}, walls {}".format(w_bal, walls))


def gate_B(H):
    g = GateStack((8e-9, 20.0))
    tight = dict(gauss_tol_C_m2=1e-11)
    s = solver(g, g, **tight)
    w_sym = 0.0
    for V in ((0.0, 0.0), (2.0, 2.0), (-3.0, -3.0)):
        sol = s.solve_voltages(*V)
        w_sym = max(w_sym, np.max(np.abs(sol.phi_V - sol.phi_V[::-1])),
                    np.max(np.abs(sol.n_m3 - sol.n_m3[::-1])) / NBG)
    bot, top = asym_gates()
    s1, s2 = solver(bot, top, **tight), solver(top, bot, **tight)
    w_ex, far = 0.0, np.inf
    for V in ((2.0, -1.0), (-3.0, 2.0)):
        a, b = s1.solve_voltages(*V), s2.solve_voltages(V[1], V[0])
        w_ex = max(w_ex, np.max(np.abs(a.phi_V - b.phi_V[::-1])), np.max(np.abs(a.n_m3 - b.n_m3[::-1])) / NBG,
                   abs(a.gate_charge_bottom_C_m2 - b.gate_charge_top_C_m2) * 10.0)
        far = min(far, np.max(np.abs(a.phi_V - a.phi_V[::-1])))
    H.gate("B1 symmetric stacks and biases: mirror-symmetric (1e-9)", w_sym < 1e-9, "worst {:.2e}".format(w_sym))
    H.gate("B2 gate exchange mirrors the profile (1e-9), not vacuous (> 0.1 V)", w_ex < 1e-9 and far > 0.1,
           "worst {:.2e}; asymmetry {:.3f} V".format(w_ex, far))


def gate_C(H):
    bot, top = asym_gates()
    vals = []
    for nz in (241, 481, 961, 1921):
        sol = solver(bot, top, nz=nz).solve_voltages(2.0, -1.0)
        vals.append((sol.phi_bottom_V, sol.phi_top_V, sol.film_charge_C_m2))
        print("[sp_dual_gate]   nz {:5d}: phi_b {:.8f} V, phi_t {:.8f} V, film charge {:.9e} C/m^2".format(
            nz, *vals[-1]), flush=True)
    d = np.abs(np.diff(np.array(vals), axis=0))
    r = d[:-1] / d[1:]
    ok = bool(np.all((r > 3.0) & (r < 5.0))) and float(np.max(d[2, :2])) < 2e-5
    H.gate("C grid convergence second order (ratios in [3, 5]; finest < 2e-5 V)", ok,
           "ratios {}".format(np.round(r, 3).tolist()))


def gate_D(H):
    from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
    from dynameta.sweep import BiasPoint
    t, nz = 20e-9, 801
    c = SchrodingerPoissonCarrier(semi_thk_m=t, n_bg_m3=NBG, m_eff_kg=MSTAR, eps_static=EPS_S, nz=nz, n_lateral=2,
                                  oxide_thk_m=8e-9, eps_oxide=20.0, neumann_body=True, n_states=200)
    thick = solver(GateStack((1e-5, 1.0)), GateStack((8e-9, 20.0)), t_film_m=t, nz=nz)
    open_ = solver(None, GateStack((8e-9, 20.0)), t_film_m=t, nz=nz)
    top5 = slice(nz - 201, None)
    w_lim = w_phi = w_n = 0.0
    for vg in (2.0, -2.0):
        cf = c.solve(BiasPoint({"gate": vg, "body": 0.0}, "g"))
        psi = cf.extras["surface_potential_V"]
        n1 = cf.regions["semi"].grid_fields["electron_density_m3"][0, 0]
        a, b = thick.solve_voltages(0.0, vg), open_.solve_voltages(0.0, vg)
        w_lim = max(w_lim, abs(a.phi_top_V - b.phi_top_V))
        w_phi = max(w_phi, abs(a.phi_top_V - psi))
        w_n = max(w_n, float(np.max(np.abs(a.n_m3[top5] - n1[top5]))) / NBG)
    H.gate("D1 far gate behind 10 um of eps 1 = no far gate (1e-6 V)", w_lim < 1e-6, "{:.2e} V".format(w_lim))
    H.gate("D2 top face = single-gate solver (5e-5 V, 3e-4 n_bg over the top 5 nm)", w_phi < 5e-5 and w_n < 3e-4,
           "{:.2e} V, {:.2e} n_bg".format(w_phi, w_n))


def gate_E(H):
    bot, top = asym_gates()
    s = solver(bot, top)
    base = s.solve_voltages(0.0, 0.0)
    up_t = s.solve_voltages(0.0, 0.5)
    up_b = s.solve_voltages(0.5, 0.0)
    ok = (up_t.phi_top_V > base.phi_top_V and up_b.phi_bottom_V > base.phi_bottom_V
          and up_t.sheet_excess_m2 > base.sheet_excess_m2 and up_b.sheet_excess_m2 > base.sheet_excess_m2
          and up_t.n_m3[-10] > base.n_m3[-10] and up_b.n_m3[9] > base.n_m3[9])
    H.gate("E raising a gate raises its face potential, the near-face density and the sheet", ok,
           "dphi_t {:+.4f} V, dphi_b {:+.4f} V".format(up_t.phi_top_V - base.phi_top_V,
                                                     up_b.phi_bottom_V - base.phi_bottom_V))


def main():
    t0 = time.perf_counter()
    H = Harness("sp_dual_gate", "dual-gate Schrodinger-Poisson: face laws, symmetry, convergence, single-gate limit")
    for fn in (gate_A, gate_B, gate_C, gate_D, gate_E):
        try:                                   # an exception is a failed gate, never a crash (validation._harness)
            fn(H)
        except Exception as exc:               # noqa: BLE001
            H.gate(fn.__name__, False, "EXCEPTION {}: {}".format(type(exc).__name__, str(exc)[:200]))
    print("[sp_dual_gate] {:.0f} s".format(time.perf_counter() - t0), flush=True)
    return H.summary()


if __name__ == "__main__":
    raise SystemExit(main())
