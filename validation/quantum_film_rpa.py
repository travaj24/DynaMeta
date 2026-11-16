"""
Self-consistent (RPA) response of a quantum-confined degenerate film vs the local, effective and
hydrodynamic models (dynameta.optics.quantum_film; v0.16).

Fixture (the generic gated-film cell): 6 nm film, background density 3e26 m^-3, m* 0.30 m_e (z and
in-plane), eps_inf 4.0 (optical background), eps_static 9.0 (electrostatics), damping 1.2e14 rad/s, 300 K;
gated through 6 nm high-k oxide (static eps 20) on both faces (carriers.DualGateSchrodingerPoisson,
481 nodes). Optical stack: air | 6 nm oxide (n 2.00) | film | 6 nm oxide (n 2.00) | glass (n 1.45),
p-polarization at 60 deg, 1100 / 1200 / 1450 nm; the top gate faces the superstrate. The gate electrodes
themselves are outside the optical stack (assumed beyond it / transparent).

Models compared (all fed the SAME self-consistent ground state):
  RPA      -- the exact planar element of the self-consistent sub-band response (quantum_film);
  RPA+xc   -- the same with adiabatic LDA exchange-correlation (sensitivity);
  RPA-eq   -- the thin-film equivalent uniaxial layer of the RPA (first order in L);
  (i)   local graded Drude: isotropic eps(n(z)) in one sub-layer per grid cell;
  (ii)  Part 1 effective layer: uniform uniaxial layer, eps_xx = Drude of N_s / L, eps_zz =
        IntersubbandEffect(z_response='subbands', thickness_m=L) on the complete sub-band set;
  (iii) hydrodynamic: nonlocal_tmm graded run on n(z) ('high_freq' beta, hard walls);
  (legacy) the v0.15.1 eps_zz = Drude + sum ISB uniform layer, for context only.

GATES (PASS/FAIL):
  A  cross-route: modes vs real-space alpha_perp on the 0 V film at 1200 nm, 1e-8 relative.
  B  f-sum: sum of RPA mode strengths = (e^2/(eps0 eps_inf m)) sum n_i kappa_i, 1e-10 relative.
  C  energy: A_RPA = 1 - R - T >= 0 at every bias and wavelength; gamma = 0 gives |A| < 1e-12.
  D  equivalent layer: |r_RPA - r_RPA-eq| < 1e-4 at every point.
  E  hydrodynamic agreement where valid: 6 nm flat band near the screened plasma frequency, nonlocal
     corrections agree to 10 %; reported (not gated) where invalid.
  F  wide-film limit: |eps_zz,eq - eps_D| falls with L over 10 / 20 / 40 / 80 nm (flat band, 1200 nm),
     below 0.014 at 80 nm (1.28x the measured value, the ratio of the bar to the measurement it was set
     from).

Run: python -m validation.quantum_film_rpa
"""

import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers import DualGateSchrodingerPoisson, GateStack
from dynameta.carriers.schrodinger_poisson import SubbandResult
from dynameta.constants import C_LIGHT, EPS0, M_E, Q_E
from dynameta.core.effects.electroabsorption import IntersubbandEffect
from dynameta.optics import nonlocal_tmm as nt
from dynameta.optics.quantum_film import QuantumFilm, QuantumFilmResponse, uniaxial_layer_matrix, film_stack_rt

M = 0.30 * M_E
EPS_INF = 4.0
GAM = 1.2e14
N_BG = 3e26
T_FILM = 6e-9
N_OX = 2.00
N_SUB = 1.45
THETA = math.radians(60.0)
LAMS = (1100e-9, 1200e-9, 1450e-9)
BIASES = ((0.0, 0.0), (2.0, 2.0), (-2.0, -2.0), (0.0, 3.0), (0.0, -3.0))
ABOVE = [nt.DielectricLayer(N_OX ** 2, 6e-9)]
BELOW = [nt.DielectricLayer(N_OX ** 2, 6e-9)]
F_BAR = 0.014        # 1.28x the measured 80 nm value
KW = dict(n_sub=N_SUB, theta_rad=THETA)


def _superstrate_first(film):
    """(z, n) ordered from the superstrate-side face (the top, z = t) downward."""
    z, n = film.z_m, film.density_m3.copy()
    return (z[-1] - z)[::-1], n[::-1]


def _local_layers(film, w):
    zs, ns = _superstrate_first(film)
    nav = 0.5 * (ns[:-1] + ns[1:])
    out = []
    for nk, dk in zip(nav, np.diff(zs)):
        wp2 = nk * Q_E ** 2 / (EPS0 * M)
        out.append(nt.DielectricLayer(EPS_INF - wp2 / (w * (w + 1j * GAM)), float(dk)))
    return out


def _hydro_layers(film):
    zs, ns = _superstrate_first(film)
    ns = ns.copy()
    ns[ns <= 0.0] = 1e-9 * ns.max()                       # the two hard-wall face nodes
    return nt.hydro_layers_from_profile(zs, ns, m_eff_kg=M, eps_inf=EPS_INF, gamma=GAM)


def _part1(film, lam, legacy=False):
    res = SubbandResult(energies_J=film.energies_J, psi=film.psi, z_m=film.z_m[1:-1],
                        sheet_density_m2=film.sheet_density_m2)
    kw = dict(thickness_m=film.thickness_m)
    if legacy:
        kw["z_response"] = "drude_plus_subbands"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                   # the legacy deprecation advisory
        e = IntersubbandEffect(EPS_INF, M, GAM, GAM, **kw).eps({"subband": res}, lam)
    return complex(e[0, 0]), complex(e[2, 2])


def _uniaxial_rt(w, exx, ezz, L):
    return film_stack_rt(w, lambda ww, kp: uniaxial_layer_matrix(exx, ezz, L, complex(ww) / C_LIGHT, kp),
                         ABOVE, BELOW, **KW)


def _phat(res):
    """nonlocal_tmm.stack_rt's RTA with r in the p-hat convention of film_stack_rt (r -> -r)."""
    return res._replace(r=-res.r)


def main():
    print("[qf] === self-consistent (RPA) response of a quantum-confined film ===", flush=True)
    ok = True
    t0 = time.time()
    s = DualGateSchrodingerPoisson(t_film_m=T_FILM, bottom=GateStack((6e-9, 20.0)), top=GateStack((6e-9, 20.0)),
                                   n_bg_m3=N_BG, m_eff_kg=M, eps_static=9.0, nz=481)
    films = {}
    for vb, vt in BIASES:
        films[(vb, vt)] = QuantumFilm.from_dual_gate(s, s.solve_voltages(vb, vt))
    print("[qf] dual-gate solves + complete sub-band sets: {:.1f} s".format(time.time() - t0), flush=True)

    # ---- ground state per bias ----
    print("[qf] bias (Vb, Vt) | N_s [1e18 m^-2] (N_s / n_bg L) | n_max / n_bg | occupied sub-bands | pairs")
    for key, f in films.items():
        r = QuantumFilmResponse(f, EPS_INF, GAM)
        print("[qf]   ({:+.0f}, {:+.0f}) V | {:.4f} ({:.4f}) | {:.3f} | {} | {}".format(
            key[0], key[1], f.sheet_density_total_m2 / 1e18, f.sheet_density_total_m2 / (N_BG * T_FILM),
            f.density_m3.max() / N_BG, r.occ.size, r.n_pairs), flush=True)

    # ---- GATE A, B ----
    f0 = films[(0.0, 0.0)]
    w12 = 2.0 * math.pi * C_LIGHT / 1200e-9
    rm = QuantumFilmResponse(f0, EPS_INF, GAM, route="modes")
    rr = QuantumFilmResponse(f0, EPS_INF, GAM, route="real_space")
    am, ar = rm.alpha_perp(w12), rr.alpha_perp(w12)
    dA = abs(am - ar) / abs(am)
    gA = bool(dA < 1e-8)
    rm.modes()
    fs = rm.fsum()
    dB = abs(fs["modes"] - fs["identity"]) / fs["identity"]
    gB = bool(dB < 1e-10)
    ok = ok and gA and gB
    print("[qf] GATE A (cross-route alpha_perp, 0 V, 1200 nm): {:.6e} vs {:.6e} m, rel {:.1e} -> {}".format(
        am, ar, dA, "PASS" if gA else "FAIL"), flush=True)
    print("[qf] GATE B (f-sum): sum F_m / identity - 1 = {:.1e}; bare TRK ratio {:.5f} -> {}".format(
        dB, fs["ratio_bare"], "PASS" if gB else "FAIL"), flush=True)

    # ---- the comparison table (item 4) ----
    print("[qf] --- complex reflection r and absorbed fraction A (60 deg, p) ---", flush=True)
    gC, gD = True, True
    rows = []
    for key, f in films.items():
        resp = QuantumFilmResponse(f, EPS_INF, GAM)
        resp_x = QuantumFilmResponse(f, EPS_INF, GAM, xc="lda")
        hyd = _hydro_layers(f)
        for lam in LAMS:
            w = 2.0 * math.pi * C_LIGHT / lam
            out = {}
            out["RPA"] = resp.stack_rt(w, ABOVE, BELOW, **KW)
            out["RPA+xc"] = resp_x.stack_rt(w, ABOVE, BELOW, **KW)
            out["RPA-eq"] = resp.stack_rt(w, ABOVE, BELOW, model="equivalent", **KW)
            # nonlocal_tmm.stack_rt returns the tangential-E r (the negative of the p-hat r of the film routes)
            out["(i) local graded"] = _phat(nt.stack_rt(w, ABOVE + _local_layers(f, w) + BELOW, **KW))
            exx1, ezz1 = _part1(f, lam)
            out["(ii) Part 1 eps_zz"] = _uniaxial_rt(w, exx1, ezz1, f.thickness_m)
            out["(iii) hydrodynamic"] = _phat(nt.stack_rt(w, ABOVE + hyd + BELOW, **KW))
            exl, ezl = _part1(f, lam, legacy=True)
            out["(legacy v0.15.1)"] = _uniaxial_rt(w, exl, ezl, f.thickness_m)
            exx, ezz, _ = resp.effective_layer(w)
            u = max(x for x in nt.hydro_validity(w, hyd[len(hyd) // 4: 3 * len(hyd) // 4]).u)
            gC = gC and out["RPA"].A >= 0.0
            gD = gD and abs(out["RPA"].r - out["RPA-eq"].r) < 1e-4
            print("[qf] ({:+.0f}, {:+.0f}) V, {:.0f} nm: RPA eps_xx,eq = {:.4f}, eps_zz,eq = {:.4f}; Part 1 eps_zz = "
                  "{:.4f}; hydro centre-half u_max = {:.2f}".format(key[0], key[1], lam * 1e9, exx, ezz, ezz1, u),
                  flush=True)
            for name, v in out.items():
                print("[qf]     {:20s} r = {:+.5f}{:+.5f}i  |r| = {:.5f}  A = {:.5f}  |r - r_RPA| = {:.2e}  "
                      "A - A_RPA = {:+.5f}".format(name, v.r.real, v.r.imag, abs(v.r), v.A,
                                                  abs(v.r - out["RPA"].r), v.A - out["RPA"].A), flush=True)
            rows.append((key, lam, out))
    r0 = QuantumFilmResponse(f0, EPS_INF, 0.0)
    a0 = max(abs(r0.stack_rt(2.0 * math.pi * C_LIGHT / lam, ABOVE, BELOW, **KW).A) for lam in LAMS)
    gC = gC and a0 < 1e-12
    ok = ok and gC and gD
    print("[qf] GATE C (energy: A_RPA >= 0 everywhere; lossless max|A| = {:.1e}) -> {}".format(
        a0, "PASS" if gC else "FAIL"), flush=True)
    print("[qf] GATE D (equivalent layer within 1e-4 of the exact element everywhere; max {:.2e}) -> {}".format(
        max(abs(o["RPA"].r - o["RPA-eq"].r) for _, _, o in rows), "PASS" if gD else "FAIL"), flush=True)

    # ---- GATE E: hydrodynamics where valid / invalid ----
    ff = QuantumFilm.flat_band(T_FILM, N_BG, M, nz=121)
    rf = QuantumFilmResponse(ff, EPS_INF, GAM)
    hf = _hydro_layers(ff)
    wpt = math.sqrt(N_BG * Q_E ** 2 / (EPS0 * M) / EPS_INF)
    gE = True
    for x in (0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5):
        w = x * wpt
        rr_ = rf.stack_rt(w, ABOVE, BELOW, **KW).r
        rl = -nt.stack_rt(w, ABOVE + _local_layers(ff, w) + BELOW, **KW).r       # tangential-E -> p-hat
        rh = -nt.stack_rt(w, ABOVE + hf + BELOW, **KW).r
        u = nt.hydro_validity(w, [hf[len(hf) // 2]]).u[0]
        rel = abs(rr_ - rh) / abs(rr_ - rl)
        if u <= 0.6:
            gE = gE and rel < 0.10
        print("[qf]   flat 6 nm, omega/w~p = {:.2f} ({:.0f} nm), centre u = {:.2f}: |r_RPA - r_local| = {:.3e}, "
              "|r_hyd - r_local| = {:.3e}, |r_RPA - r_hyd| / |r_RPA - r_local| = {:.3f}{}".format(
                  x, 2 * math.pi * C_LIGHT / w * 1e9, u, abs(rr_ - rl), abs(rh - rl), rel,
                  "" if u <= 0.6 else "  (hydro invalid)"), flush=True)
    ok = ok and gE
    print("[qf] GATE E (hydrodynamics agrees to 10 % of the nonlocal correction where u <= 0.6) -> {}".format(
        "PASS" if gE else "FAIL"), flush=True)

    # ---- GATE F: wide-film limit ----
    wp2 = N_BG * Q_E ** 2 / (EPS0 * M)
    eD = EPS_INF - wp2 / (w12 * (w12 + 1j * GAM))
    d = []
    for t, nz in ((10e-9, 201), (20e-9, 401), (40e-9, 801), (80e-9, 1601)):
        tt = time.time()
        rw = QuantumFilmResponse(QuantumFilm.flat_band(t, N_BG, M, nz=nz), EPS_INF, GAM, route="real_space")
        exx, ezz, _ = rw.effective_layer(w12)
        d.append(abs(ezz - eD))
        print("[qf]   flat {:.0f} nm: eps_xx,eq = {:.4f}, eps_zz,eq = {:.4f} (local eps_zz {:.4f}), eps_D = {:.4f}, "
              "|eps_zz,eq - eps_D| = {:.4f} ({:.1f} s)".format(t * 1e9, exx, ezz, rw.local_eps_zz(w12), eD, d[-1],
                                                               time.time() - tt), flush=True)
    gF = bool(all(a > b for a, b in zip(d, d[1:])) and d[-1] < F_BAR)
    ok = ok and gF
    print("[qf] GATE F (wide-film limit, ratios {}) -> {}".format(
        ", ".join("{:.2f}".format(a / b) for a, b in zip(d, d[1:])), "PASS" if gF else "FAIL"), flush=True)

    print("[qf] *** QUANTUM FILM RPA: {} ({:.0f} s) ***".format("PASS" if ok else "FAIL", time.time() - t0),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
