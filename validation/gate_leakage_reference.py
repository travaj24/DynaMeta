"""Gate leakage and the lateral Fermi-level drop of a gated 6 nm degenerate film (carriers.gate_leakage): the parameter
study of docs/audit/2026-11-14-v0.16-gate-leakage-lateral-drop.md (section 5), on the generic gated-film cell.

    python -m validation.gate_leakage_reference            # central barrier set + the 2-D / 1-D cross-check (~10 min)
    python -m validation.gate_leakage_reference --full     # all three barrier sets and the two-gate cases

Device (tests/_wiring_fixture.py): film n = 3e26 m^-3, DOS mass 0.30 m0, eps 9.0, between 6 nm oxB (eps 20) + 2 nm
oxA (eps 9) on each face (oxB next to the film), metal gates whose work function equals the film's (flat band at
0 V); top gate up to +-3 V, bottom gate up to +-5 V; mobility 30 cm^2/Vs (40 in the sheet-resistance table).

Method. (1) Narrow-cell DEVSIM solves (10 nm cell: laterally uniform, the drop < 1 mV even at the largest currents) of
the film with the physical (Tsu-Esaki/WKB) leakage, gates at their nominal bias and the edge contact swept through
phi = 0 .. V_g: the net gate current density J(phi) into the film and the sheet conductance sigma_s(phi) = q mu Int n
dz at the local Fermi potential phi (shift invariance: a contact at phi is the film at local Fermi potential phi).
(2) The reduced sheet equation (gate_leakage.solve_lateral_1d) with those tables gives the Fermi-level profile for
any contact distance L from 100 nm to 1 mm (a 2-D mesh of a millimetre film is not possible), the maximum drop, the
effective gate action (mean (V_g - phi_F) / V_g), the L at which the drop reaches 10 % / 50 % of V_g, and, for
L = 10 / 100 / 1000 um, the leakage needed to reduce the gate action to a given fraction, tabulated for the fractions
in ACTION_FRACTIONS (0.9 to 0.5). (3) GATE (exit code): the 2-D
DEVSIM solve of a 0.5 um film at the bottom-gate +5 V case agrees with the reduced model on the maximum drop and the
gate action to 2 %.

Barrier sets (eV above the FILM FERMI LEVEL, tunnelling masses in m0): see BARRIER_SETS, three illustrative sets
(low / central / high) for the two generic dielectrics. The conduction-band offset handed to the kernel is that
barrier + (E_F - E_c) of the film at n_bg.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

Q = 1.602176634e-19
M_E = 9.1093837015e-31
N_BG, T_FILM = 3.0e26, 6e-9
MU = 30e-4
BARRIER_SETS = {          # (oxB barrier, oxB mass, oxA barrier, oxA mass)
    "low": (1.5, 0.11, 2.5, 0.28),
    "central": (2.0, 0.15, 2.9, 0.30),
    "high": (2.5, 0.40, 3.3, 0.45),
}
SINGLE = (("top", 3.0), ("top", -3.0), ("bot", 5.0), ("bot", -5.0))
BOTH = (((3.0, 5.0)), ((-3.0, -5.0)))
LENGTHS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
ACTION_FRACTIONS = (0.9, 0.8, 0.7, 0.6, 0.5)     # gate-action fractions of the leakage-needed table


def ef_above_ec_eV() -> float:
    from dynameta.carriers import physics_equilibrium as PE
    nc = PE.effective_dos_m3(0.30 * M_E)
    return PE.invert_F12(N_BG / nc) * PE.V_T


def leak(set_name, scale=1.0):
    from dynameta.carriers.gate_leakage import GateLeakage
    from dynameta.carriers.tunneling import DielectricBarrier
    hb, hm, ab, am = BARRIER_SETS[set_name]
    de = ef_above_ec_eV()
    h, a = DielectricBarrier(hb + de, "TCO", hm), DielectricBarrier(ab + de, "TCO", am)
    return GateLeakage(model="tsu_esaki", barriers={"up_oxb": h, "lo_oxb": h, "up_oxa": a, "lo_oxa": a},
                       gate_fermi_energy_eV=10.0, mobility_m2Vs=MU, energy_max_eV=8.0, scale=scale)


def design(P, dx, role="biased"):
    import _wiring_fixture as W
    from dynameta.geometry import UnitCell
    d = W.film_design(edges=[W.edge(role=role)], x_lines=dict(x_spacing_edge_m=dx, x_spacing_feature_edge_m=dx,
                                                                x_spacing_feature_mid_m=dx))
    return dataclasses.replace(d, unit_cell=UnitCell(period_x_m=P, period_y_m=40e-9))


def sheet(ds, device):
    from dynameta.core.numerics import trapz
    X = np.asarray(ds.get_node_model_values(device=device, region="film", name="x"))
    Y = np.asarray(ds.get_node_model_values(device=device, region="film", name="y"))
    n = np.asarray(ds.get_node_model_values(device=device, region="film", name="Electrons"))
    xs = np.unique(X)
    sel = X == xs[len(xs) // 2]
    o = np.argsort(Y[sel])
    return Q * MU * trapz(n[sel][o], Y[sel][o])


def tables(set_name, vt, vb, n_phi=13, tag="t"):
    """Narrow-cell sweep of the contact voltage phi from 0 to the gate voltage(s): J_net(phi), sigma_s(phi), per-gate
    J at phi = 0 and the oxide fields there."""
    from dynameta.carriers import gate_leakage as GL
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    ds = import_devsim()
    b = LayeredDevsimBuilder(design(10e-9, 5e-9), mesh_name="m_" + tag, device_name="d_" + tag,
                             gate_leakage=leak(set_name))
    vmax = max(abs(vt), abs(vb))
    sgn = 1.0 if (vt if abs(vt) >= abs(vb) else vb) > 0 else -1.0
    phis = sgn * vmax * np.linspace(0.0, 1.0, n_phi) ** 2     # denser near phi = 0, where the drops are
    out = {"phi": [], "J": [], "sigma": [], "gates0": None, "fields0": None}
    for k, p in enumerate(phis):
        try:
            b.solve(BiasPoint({"top": vt, "bot": vb, "gnd": float(p)}, "p"), abs_tol=1e-14)
        except Exception:                                     # noqa: BLE001 (keep the points collected so far)
            if k < 3:
                b.teardown()
                raise
            out["truncated_at"] = float(p)
            break
        rep = GL.gate_leakage_report(b.device, "film", contacts=["gnd"])
        phi_loc = float(np.mean(rep["phi_F_V"]))
        out["phi"].append(phi_loc)
        out["J"].append(sum(g["J_mean"] for g in rep["gates"].values()))
        out["sigma"].append(sheet(ds, b.device))
        if k == 0:
            out["gates0"] = {g: v["J_mean"] for g, v in rep["gates"].items()}
            fl = {}
            for t in GL.gate_leakage_terms(b.device):
                d = t.state(b.device)["d"][0]
                fl[t.gate] = (abs(d[0]) / t.layers[0].thickness_m, abs(d[1] - d[0]) / t.layers[1].thickness_m)
            out["fields0"] = fl
    if "truncated_at" not in out:
        b.teardown()
    else:
        from dynameta.carriers.devsim_env import import_devsim as _imp
        for dv in list(_imp().get_device_list()):           # the failed device state is unusable
            _imp().delete_device(device=dv)
        for m in list(_imp().get_mesh_list()):
            _imp().delete_mesh(mesh=m)
    o = np.argsort(out["phi"])
    for k in ("phi", "J", "sigma"):
        out[k] = np.asarray(out[k])[o]
    return out


def reduced(tab, L, scale=1.0, n_nodes=801):
    from scipy.interpolate import PchipInterpolator
    from dynameta.carriers.gate_leakage import solve_lateral_1d
    phi, J, sig = tab["phi"], tab["J"], tab["sigma"]
    J0 = max(1e-30, 1e-9 * float(np.max(np.abs(J))))
    yj = PchipInterpolator(phi, np.arcsinh(J / J0))
    dyj = yj.derivative()
    ps = PchipInterpolator(phi, sig)
    dps = ps.derivative()
    lo, hi = float(phi[0]), float(phi[-1])

    def cl(p):
        return np.clip(p, lo, hi)

    def Jf(p):
        y = yj(cl(p))
        return scale * J0 * np.sinh(y), scale * J0 * np.cosh(y) * dyj(cl(p))

    def sf(p):
        return ps(cl(p)), dps(cl(p))

    return solve_lateral_1d(L, sf, Jf, n_nodes=n_nodes)


def gate_action(r, vg):
    from dynameta.core.numerics import trapz
    x, phi = r["x_m"], r["phi_V"]
    return 1.0 - trapz(phi, x) / x[-1] / vg


def find_L(tab, vg, frac):
    """L at which the max drop reaches frac * |V_g| (bisection in log L over 1e-8 .. 1 m); inf if never."""
    def drop(L):
        try:
            return reduced(tab, L)["max_drop_V"] / abs(vg)
        except Exception:                                     # noqa: BLE001 (beyond the solver's reach)
            return float("nan")
    a, b = 1e-8, 1.0
    if not drop(b) >= frac:
        return float("inf")
    for _ in range(40):
        m = math.sqrt(a * b)
        d = drop(m)
        if d >= frac:
            b = m
        else:
            a = m
        if b / a < 1.01:
            break
    return math.sqrt(a * b)


def find_scale(tab, vg, L, target):
    """Leakage scale s with gate action = target at length L (bisection in log s); returns (s, J_nominal)."""
    def act(s):
        return gate_action(reduced(tab, L, scale=s), vg)
    a, b = 1e-12, 1e30
    for _ in range(200):
        m = math.sqrt(a * b)
        try:
            g = act(m)
        except Exception:                                     # noqa: BLE001
            b = m
            continue
        if g > target:
            a = m
        else:
            b = m
        if b / a < 1.02:
            break
    s = math.sqrt(a * b)
    if s > 1e29:                                              # the ceiling: no leakage level reaches the target
        return float("inf"), float("inf")
    return s, s * float(tab["J"][np.argmin(np.abs(tab["phi"]))])


def cross_check(set_name, tab, L=0.5e-6, dx=25e-9):
    """2-D DEVSIM film of length L (bottom gate +5 V) against the reduced model on the same tables."""
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    b = LayeredDevsimBuilder(design(L, dx, role="ground"), mesh_name="m_x", device_name="d_x",
                             gate_leakage=leak(set_name))
    t0 = time.time()
    b.solve(BiasPoint({"top": 0.0, "bot": 5.0}, "x"), abs_tol=1e-14)
    rep = b.gate_leakage_report(per_cell=False)["film"]
    b.teardown()
    r = reduced(tab, L)
    return {"drop2d": rep["max_drop_V"], "action2d": rep["gates"]["bot"]["gate_action"], "drop1d": r["max_drop_V"],
            "action1d": gate_action(r, 5.0), "balance_rel": rep["balance_rel"], "seconds": time.time() - t0,
            "vspread": rep["vertical_spread_V"]}


def main(argv):
    full = "--full" in argv
    print("[gate_leakage_reference] E_F - E_c of the film at n_bg = {:.4f} eV".format(ef_above_ec_eV()))
    for mu in (30e-4, 40e-4):
        s = Q * N_BG * mu * T_FILM
        print("  mobility {:.0f} cm^2/Vs: sigma_s = {:.4g} S, R_s = {:.1f} ohm/sq".format(mu * 1e4, s, 1.0 / s))
    sets = list(BARRIER_SETS) if full else ["central"]
    ok = True
    results = {}
    for sn in sets:
        print("\n=== barrier set {!r}: oxB {} eV / {} m0, oxA {} eV / {} m0 (above the film E_F) ===".format(
            sn, *BARRIER_SETS[sn]))
        cases = [((v if g == "top" else 0.0), (v if g == "bot" else 0.0), g, v) for g, v in SINGLE]
        if full:
            cases += [(vt, vb, "both", vt) for vt, vb in BOTH]
        for vt, vb, g, vg in cases:
            t0 = time.time()
            try:
                tab = tables(sn, vt, vb, tag="{}{}".format(sn[0], len(results)))
            except Exception as e:                            # noqa: BLE001
                print("  top {:+.1f} V, bottom {:+.1f} V: NOT CONVERGED in the narrow-cell sweep ({}: {})".format(
                    vt, vb, type(e).__name__, str(e).splitlines()[0][:120]))
                results[(sn, vt, vb)] = None
                continue
            results[(sn, vt, vb)] = tab
            Jn = tab["J"][np.argmin(np.abs(tab["phi"]))]
            fl = tab["fields0"]
            print("  top {:+.1f} V, bottom {:+.1f} V: J_net(phi=0) = {:+.3e} A/m^2 ({:+.3e} A/cm^2); per gate {}; "
                  "E(oxB, oxA) [MV/cm] {}; sigma_s {:.4g}..{:.4g} S  ({:.0f} s)".format(
                      vt, vb, Jn, Jn * 1e-4, {k: "{:+.3e}".format(v) for k, v in tab["gates0"].items()},
                      {k: "({:.2f}, {:.2f})".format(a * 1e-8, b_ * 1e-8) for k, (a, b_) in fl.items()},
                      float(np.min(tab["sigma"])), float(np.max(tab["sigma"])), time.time() - t0))
            vmain = vt if abs(vt) >= abs(vb) else vb
            if "truncated_at" in tab:
                print("    (contact sweep stopped by a non-converged solve at phi = {:.3g} V; tables up to {:.3g} V; a "
                      "result marked * leaves that range, where J and sigma are held at their last values -- J falls "
                      "toward V_g, so * marks an UPPER bound on the drop)".format(tab["truncated_at"],
                                                                               float(np.max(np.abs(tab["phi"])))))
            row = []
            for L in LENGTHS:
                r = reduced(tab, L)
                star = "*" if r["max_drop_V"] > float(np.max(np.abs(tab["phi"]))) + 1e-12 else ""
                row.append("L={:.0e}: drop {:.3e} V{}, action {:.4f}".format(L, r["max_drop_V"], star,
                                                                           gate_action(r, vmain)))
            print("    " + "; ".join(row))
            print("    L(drop = 10 % V_g) = {:.3e} m, L(50 %) = {:.3e} m".format(
                find_L(tab, vmain, 0.1), find_L(tab, vmain, 0.5)))
            for L in (1e-5, 1e-4, 1e-3):
                parts = []
                for eta in ACTION_FRACTIONS:
                    s, jn = find_scale(tab, vmain, L, eta)
                    sig = float(tab["sigma"][np.argmin(np.abs(tab["phi"]))])
                    j_uni = 3.0 * sig * abs(vmain) * (1.0 - eta) / L ** 2
                    need = ("unreachable" if not math.isfinite(s) else
                            "{:.2e} A/cm^2 (x{:.1e} the model)".format(abs(jn) * 1e-4, s))
                    parts.append("action {:.2f}: J needed {}; uniform-J estimate {:.2e} A/cm^2".format(
                        eta, need, j_uni * 1e-4))
                print("    L={:.0e} m: ".format(L) + "; ".join(parts))
    print("\n=== GATE: 2-D DEVSIM (0.5 um film, bottom +5 V, central set) vs the reduced 1-D model ===")
    tab = results.get(("central", 0.0, 5.0))
    if tab is None:
        print("  the central bottom +5 V tables did not converge -> FAIL")
        return False
    cx = cross_check("central", tab)
    print("  2-D: drop {:.5f} V, gate action {:.5f}; 1-D: drop {:.5f} V, action {:.5f}; balance {:.1e}; vertical "
          "spread {:.2e} V ({:.0f} s)".format(cx["drop2d"], cx["action2d"], cx["drop1d"], cx["action1d"],
                                             cx["balance_rel"], cx["vspread"], cx["seconds"]))
    if cx["drop1d"] > float(np.max(np.abs(tab["phi"]))):
        print("  the 1-D profile leaves the sampled table range -> FAIL")
        return False
    rel = abs(cx["drop2d"] / cx["drop1d"] - 1.0)
    rel_a = abs((1.0 - cx["action2d"]) / (1.0 - cx["action1d"]) - 1.0)
    ok = ok and rel < 0.02 and rel_a < 0.02 and abs(cx["balance_rel"]) < 1e-10
    print("  relative difference: drop {:.2e}, (1 - action) {:.2e} -> {}".format(rel, rel_a, "PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main(sys.argv[1:]) else 1)
