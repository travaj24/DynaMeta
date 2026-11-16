"""Quantum-film response inside a structured cell: the equivalent uniaxial layer vs the exact
per-harmonic film, the small-in-plane-wavevector approximation, and the film-model comparison on a
metal-strip device across bias and wavelength.

Device (the generic strip cell): metal strips (eps -60+4i), 150 nm wide, period 400 nm, 40 nm tall, on
2 nm oxide A (n 1.70) / 6 nm oxide B (n 2.00) / 6 nm film / 6 nm oxide B / 2 nm oxide A over an 80 nm
mirror metal (eps -120+20i) on glass (n 1.45); air above; TM (E across the strips), normal incidence.
Metal permittivities are held constant over the band (a fixture simplification).
Film: 3e26 m^-3, m* 0.30 m_e, eps_static 9.0, gated through 2 nm (static eps 9) + 6 nm (static eps 20)
on both faces (carriers.DualGateSchrodingerPoisson, 481 nodes; laterally uniform, as that solver is:
the strips are not treated as a laterally patterned gate). Optical response: eps_inf 4.0, damping
1.2e14 rad/s, RPA (optics.quantum_film).

Film models: (i) the classical LOCAL graded Drude film eps(n(z)) of the same density profile (one slab
per grid interval); (ii) the equivalent uniaxial layer of the RPA response (what
optics.quantum_film_pipeline hands the structured solvers); (iii) the exact RPA film per lateral harmonic
(optics.quantum_film_cell: q -> 0 electronic response, no thin-layer reduction). The hydrodynamic film
(optics.fem_2d_tm_hydro) is NOT in the sweep: on such a profile its dilute wall layers set the conductor
row height (h_z ~ 1/max|k_L|, set by the most dilute layer), which makes a solve very large -- and the
profile is outside the model's validity (u ~ 1) anyway.

Gates
  A  planar stack, exact film per harmonic vs the exact planar element (film_stack_rt), 0 and 35 deg:
     |dr| < 1e-10, energy residual < 1e-12
  B  strip cell with a uniform uniaxial film: cell solver vs optics.fem_2d_tm (order 7, 12 levels; order 6
     / 10 levels is itself 2.3e-7 off the converged film absorption of this cell, above the 1e-7 bar):
     |dr| < 5e-6, |dA_film| < 1e-7
  C  exactly treated film harmonics 40 vs 80: |dr| < 1e-7 at every bias
  D  the exact-vs-equivalent difference is mesh independent (order 4 vs 5): < 2e-6
  E  energy closes in every cell solve: |1 - R - T - sum A| < 1e-10
  F  the pipeline's FEM route (run_quantum_film_pipeline + make_fem_2d_tm_optical_solver) equals the
     cell solver with the equivalent layer to the FEM's discretization (< 1e-4)

SPLIT BY WAVELENGTH BAND. Unsplit, this script takes ~1350 s on the 2-thread dev box (~85 % of it the
model-comparison sweep: 7 biases x 61 wavelengths x 3 film models, one
cell factorization per wavelength) against run_all's 2700 s per-script cap. Every wavelength of the sweep
is independent (its own cell factorization), so the sweep is split into three bands with the SAME
wavelengths, biases, models and solver settings; gate E (same 1e-10 tolerance) is checked in each
part over every cell solve that part makes, so the union covers every cell solve of the unsplit script:
  quantum_film_strip           gates A, B, C, D, F + tables 1 and 2 at 1200 nm + sweep 1050-1130 nm;
                               gate E over the table-1 solves and its sweep band
  quantum_film_strip_sweep_mid sweep 1135-1240 nm, gate E
  quantum_film_strip_sweep_long sweep 1245-1350 nm, gate E
Each part prints its band's per-model max |R - R_iii| (the full-band maximum is the largest of the
three). The full-band dip table (dip / R_min / FWHM / shift per model and bias) needs all 61
wavelengths in one process; it is the opt-in report validation/quantum_film_strip_dip_table.py
(excluded from run_all, reason and time recorded there).
Measured wall time (dev box, 2 threads, lightly loaded, 2026-11-15): quantum_film_strip
508 s (~145 s gates and tables, ~360 s sweep), quantum_film_strip_sweep_mid 484 s, quantum_film_strip_sweep_long
482 s; the slowest part is under 1/5 of the cap.

Run: python -m validation.quantum_film_strip
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation._harness import Harness  # noqa: E402

LAM0 = 1200e-9
P, W = 400e-9, 150e-9
EA, EB = 1.70 ** 2, 2.00 ** 2                  # oxide A (2 nm), oxide B (6 nm)
E_STRIP, E_MIRROR, N_SUB = -60 + 4j, -120 + 20j, 1.45
M_OPT = 0.30
EPS_INF, GAM = 4.0, 1.2e14
N_BG, T_FILM = 3e26, 6e-9
BIASES_T1 = ((0.0, 0.0), (2.0, 2.0), (-2.0, -2.0), (0.0, 3.0), (0.0, -3.0))
BIASES_T3 = ((-2.0, -2.0), (-1.0, -1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.0, 3.0), (0.0, -3.0))
LAMS_T3 = np.arange(1050e-9, 1350e-9 + 1e-12, 5e-9)
MF = 40
if os.environ.get("DYNAMETA_QFS_QUICK"):          # smoke run of the script's own code paths (not a gate run)
    BIASES_T3 = ((0.0, 0.0), (2.0, 2.0))
    LAMS_T3 = np.array([1190e-9, 1200e-9, 1210e-9])
# the three sweep bands [nm], inclusive; their union is LAMS_T3 (17 + 22 + 22 wavelengths). The long
# wavelengths cost more per cell (thicker buffers / PML), hence the shorter first band, which also
# carries the 1200 nm gates.
BANDS_NM = {"short": (1050.0, 1130.0), "mid": (1135.0, 1240.0), "long": (1245.0, 1350.0)}
MODELS = ("(i) local graded", "(ii) equivalent RPA", "(iii) exact RPA")


def band_lams(name):
    """The LAMS_T3 entries inside band `name` -- the very same float values as the full grid."""
    lo, hi = BANDS_NM[name]
    nm = LAMS_T3 * 1e9
    return LAMS_T3[(nm > lo - 1e-6) & (nm < hi + 1e-6)]


def _cell_layers():
    from dynameta.optics.fem_2d_tm import TMBlock, TMLayer
    return [TMLayer("oxb_in", 3e-9, EB), TMLayer("oxa", 2e-9, EA),
            TMLayer("grating", 40e-9, 1.0, (TMBlock(P / 2 - W / 2, P / 2 + W / 2, E_STRIP, "metal"),))]


def _lower(film):
    from dynameta.optics.quantum_film_cell import UniformLayer
    return [UniformLayer("mirror", 80e-9, E_MIRROR), UniformLayer("lo_oxa", 2e-9, EA), UniformLayer("lo_oxb", 6e-9, EB),
            film, UniformLayer("oxb_rest", 3e-9, EB)]


def _lindhard_film(resp):
    """Estimate of the leading in-plane finite-q correction: the q -> 0 element with each sub-band's
    Drude weight multiplied by the 2-D Lindhard high-frequency factor 1 + (3/4) (q v_F,i / omega)^2
    (v_F,i the sub-band's Fermi velocity). The out-of-plane (intersubband) dispersion is not included."""
    from dynameta.constants import EPS0, KB
    from dynameta.optics.quantum_film import film_transfer_matrix
    from dynameta.optics.quantum_film_cell import FilmLayer
    f = resp.film
    m = float(f.m_eff_kg)
    ekin = np.maximum(float(f.fermi_level_J) - resp.E, KB * float(f.T_K))
    vF = np.sqrt(2.0 * ekin / m)
    D = resp.drude_weights
    cache = {}

    def fn(omega, q):
        R = cache.get(float(omega))
        if R is None:
            cache.clear()
            R = resp.response_kernel(omega)
            cache[float(omega)] = R
        w = complex(omega)
        fac = 1.0 + 0.75 * (abs(q) * vF / abs(w)) ** 2
        dw = np.zeros(resp.N)
        dw[1:-1] = (resp.psi ** 2) @ (D * fac)
        exx = EPS_INF + 1j * (1j * dw / (w + 1j * resp.gamma_inplane)) / (w * EPS0)
        return film_transfer_matrix(omega, q, resp.h, exx[::-1], EPS_INF, R[::-1, ::-1])
    return FilmLayer("film", float(f.thickness_m), fn, True, "lindhard")


def _dip(lams, R):
    """(lambda_dip [nm], R_min, FWHM [nm] at half depth below max R in the window, NaN if a side does
    not cross inside the window)."""
    lams = np.asarray(lams) * 1e9
    R = np.asarray(R)
    i = int(np.argmin(R))
    if 0 < i < R.size - 1:
        x, y = lams[i - 1:i + 2], R[i - 1:i + 2]
        a, b, c = np.polyfit(x, y, 2)
        lmin = -b / (2 * a)
        rmin = c - b * b / (4 * a)
    else:
        lmin, rmin = lams[i], R[i]
    half = 0.5 * (rmin + R.max())
    left = right = float("nan")
    for j in range(i, 0, -1):
        if R[j - 1] >= half > R[j] or R[j - 1] > half >= R[j]:
            left = lams[j - 1] + (half - R[j - 1]) * (lams[j] - lams[j - 1]) / (R[j] - R[j - 1])
            break
    for j in range(i, R.size - 1):
        if R[j + 1] >= half > R[j] or R[j + 1] > half >= R[j]:
            right = lams[j] + (half - R[j]) * (lams[j + 1] - lams[j]) / (R[j + 1] - R[j])
            break
    return float(lmin), float(rmin), float(right - left)


def start(tag, title):
    """Harness + capability check + the dual-gate ground states and RPA responses for every bias
    (the same bias set, in the same order, in every part: the dual-gate solver warm-starts from
    the previous bias)."""
    h = Harness(tag, title)
    try:
        import ngsolve  # noqa: F401
    except ImportError:
        h.require(False, "ngsolve not installed")
    from dynameta.carriers import DualGateSchrodingerPoisson, GateStack
    from dynameta.constants import M_E
    from dynameta.materials import DrudeOptical
    from dynameta.optics.quantum_film import QuantumFilm, QuantumFilmResponse

    ox = GateStack(((2e-9, 9.0), (6e-9, 20.0)))
    sp = DualGateSchrodingerPoisson(t_film_m=T_FILM, bottom=ox, top=ox, n_bg_m3=N_BG, m_eff_kg=M_OPT * M_E,
                                    eps_static=9.0, nz=481, semi_material="FILM")
    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT * M_E, gamma_rad_s=GAM)
    films, resps = {}, {}
    t0 = time.time()
    for V in sorted(set(BIASES_T1) | set(BIASES_T3)):
        films[V] = QuantumFilm.from_dual_gate(sp, sp.solve_voltages(*V))
        resps[V] = QuantumFilmResponse(films[V], EPS_INF, GAM)
        resps[V].modes()
    print("[{}] dual-gate solves + responses (modes) for {} biases: {:.1f} s".format(
        tag, len(films), time.time() - t0), flush=True)
    for V, f in films.items():
        print("[{}]   ({:+.0f}, {:+.0f}) V: N_s / (n_bg L) = {:.4f}, n_max / n_bg = {:.3f}".format(
            tag, V[0], V[1], f.sheet_density_total_m2 / (N_BG * T_FILM), f.density_m3.max() / N_BG), flush=True)
    return h, sp, drude, films, resps


def sweep(tag, lams, films, resps, drude, t_start):
    """The model comparison at the given wavelengths: per wavelength one order-4 / 100-harmonic cell,
    every bias of BIASES_T3, the three film models. Returns (R, Af, worst energy residual, times)."""
    from dynameta.optics.quantum_film_cell import CellMeshSpec, FilmCell, FilmLayer
    R = {(mo, V): [] for mo in MODELS for V in BIASES_T3}
    Af = {(mo, V): [] for mo in MODELS for V in BIASES_T3}
    times = {mo: 0.0 for mo in MODELS}
    t_cells = 0.0
    worst_e = 0.0
    for lam in lams:
        t1 = time.time()
        cell = FilmCell(P, _cell_layers(), lambda_m=lam, order=4, n_harmonics=100, mesh=CellMeshSpec(corner_levels=6))
        t_cells += time.time() - t1
        for V in BIASES_T3:
            resp, f = resps[V], films[V]
            eq = FilmLayer.equivalent(resp)
            for mo, film, kw in ((MODELS[0], FilmLayer.local_graded(f.z_m, f.density_m3, drude), {}),
                                 (MODELS[1], eq, {}),
                                 (MODELS[2], FilmLayer.exact(resp), dict(film_harmonics=MF, film_fallback=eq))):
                t1 = time.time()
                res = cell.solve(_lower(film), N_SUB ** 2, **kw)
                times[mo] += time.time() - t1
                worst_e = max(worst_e, abs(res.energy_residual))
                R[(mo, V)].append(res.R)
                Af[(mo, V)].append(res.absorption["film"])
        print("[{}]   {:.0f} nm done ({:.0f} s elapsed)".format(tag, lam * 1e9, time.time() - t_start), flush=True)
    times["cell setups"] = t_cells
    return R, Af, worst_e, times


def print_differences(tag, R, what):
    print("[{}] differences from (iii) {}: max |R - R_iii| per model and bias".format(tag, what), flush=True)
    for mo in (MODELS[0], MODELS[1]):
        row = ["({:+.0f},{:+.0f}) {:.1e}".format(V[0], V[1], float(np.max(np.abs(np.array(R[(mo, V)])
                                                                                  - np.array(R[(MODELS[2], V)])))))
               for V in BIASES_T3]
        print("[{}]   {:20s} {}".format(tag, mo, "  ".join(row)), flush=True)


def print_times(tag, lams, times, t_start):
    print("[{}] time: cell setups {:.0f} s ({} wavelengths); solves: {}; total {:.0f} s".format(
        tag, times["cell setups"], len(lams), ", ".join("{} {:.0f} s".format(k, times[k]) for k in MODELS),
        time.time() - t_start), flush=True)


def run_band(tag, band):
    """One sweep band as a stand-alone gated part (quantum_film_strip_sweep_mid / _long)."""
    lams = band_lams(band)
    h, _sp, drude, films, resps = start(
        tag, "quantum film in a strip cell: model-comparison sweep, {:.0f}-{:.0f} nm".format(*BANDS_NM[band]))
    t_start = time.time()
    print("[{}] --- model comparison: {} biases x {} wavelengths ({:.0f}-{:.0f} nm) ---".format(
        tag, len(BIASES_T3), lams.size, lams[0] * 1e9, lams[-1] * 1e9), flush=True)
    R, _Af, worst_e, times = sweep(tag, lams, films, resps, drude, t_start)
    h.gate("E", worst_e < 1e-10, "max |1 - R - T - sum A| over every cell solve of the {:.0f}-{:.0f} nm band: "
           "{:.1e}".format(lams[0] * 1e9, lams[-1] * 1e9, worst_e))
    print_differences(tag, R, "over {:.0f}-{:.0f} nm".format(lams[0] * 1e9, lams[-1] * 1e9))
    print_times(tag, lams, times, t_start)
    return h.summary()


def main():
    from dynameta.constants import C_LIGHT, HBAR, M_E
    from dynameta.optics import nonlocal_tmm as nt
    from dynameta.optics.fem_2d_tm import TMBlock, TMLayer, TMMeshSpec, solve_fem_2d_tm
    from dynameta.optics.quantum_film import film_stack_rt
    from dynameta.optics.quantum_film_cell import CellMeshSpec, FilmCell, FilmLayer, UniformLayer, film_spectrum

    t_start = time.time()
    h, sp, drude, films, resps = start("qfs", "quantum film in a strip cell: equivalent layer vs exact per-harmonic film")

    # ---- A: planar, exact film per harmonic vs the planar element ----
    def gate_a():
        resp = resps[(0.0, 3.0)]
        w = 2.0 * math.pi * C_LIGHT / LAM0
        above = [nt.DielectricLayer(1.0, 30e-9), nt.DielectricLayer(EA, 2e-9), nt.DielectricLayer(EB, 6e-9)]
        below = [nt.DielectricLayer(EB, 6e-9), nt.DielectricLayer(EA, 2e-9), nt.DielectricLayer(E_MIRROR, 80e-9)]
        worst, wres = 0.0, 0.0
        for th, pol, sgn in ((0.0, "x", -1.0), (35.0, "p", 1.0)):      # film_stack_rt r is the p-hat r_H
            cl = [TMLayer("oxb_in", 3e-9, EB), TMLayer("oxa", 2e-9, EA), TMLayer("gap", 30e-9, 1.0)]
            cell = FilmCell(P, cl, lambda_m=LAM0, theta_deg=th, polarization=pol, order=4, n_harmonics=8)
            res = cell.solve(_lower(FilmLayer.exact(resp)), N_SUB ** 2)
            ref = film_stack_rt(w, lambda ww, kp: resp.transfer_matrix(ww, kp), above, below, n_super=1.0,
                                n_sub=N_SUB, theta_rad=math.radians(th))
            worst = max(worst, abs(res.r - sgn * ref.r))
            wres = max(wres, abs(res.energy_residual))
        return worst < 1e-10 and wres < 1e-12, "max |dr| = {:.1e}, max energy residual {:.1e}".format(worst, wres)
    h.run("A", gate_a)

    # ---- B: strip with a uniform uniaxial film vs fem_2d_tm ----
    w0 = 2.0 * math.pi * C_LIGHT / LAM0
    exx0, ezz0, _ = resps[(0.0, 0.0)].effective_layer(w0)
    t0 = time.time()
    cell5 = FilmCell(P, _cell_layers(), lambda_m=LAM0, order=5, n_harmonics=150, mesh=CellMeshSpec(corner_levels=8))
    t_c5 = time.time() - t0
    t0 = time.time()
    cell4 = FilmCell(P, _cell_layers(), lambda_m=LAM0, order=4, n_harmonics=100, mesh=CellMeshSpec(corner_levels=6))
    t_c4 = time.time() - t0
    print("[qfs] cell setup: order 5 / 150 harmonics {:.1f} s ({} DOFs); order 4 / 100 harmonics {:.1f} s ({} DOFs)".format(
        t_c5, cell5.ndof, t_c4, cell4.ndof), flush=True)

    def gate_b():
        full = [TMLayer("mirror", 80e-9, E_MIRROR), TMLayer("lo_oxa", 2e-9, EA), TMLayer("lo_oxb", 6e-9, EB),
                TMLayer("film", T_FILM, (exx0, ezz0)), TMLayer("up_oxb", 6e-9, EB), TMLayer("up_oxa", 2e-9, EA),
                TMLayer("grating", 40e-9, 1.0, (TMBlock(P / 2 - W / 2, P / 2 + W / 2, E_STRIP, "metal"),))]
        t1 = time.time()
        ref = solve_fem_2d_tm(P, full, lambda_m=LAM0, n_super=1.0, n_sub=N_SUB, order=7,
                              mesh_spec=TMMeshSpec(corner_levels=12), num_threads=2)
        t2 = time.time()
        res = cell5.solve(_lower(UniformLayer("film", T_FILM, (exx0, ezz0))), N_SUB ** 2)
        dr = abs(res.r - ref.r)
        dA = abs(res.absorption["film"] - ref.per_region_absorption["film"])
        return dr < 5e-6 and dA < 1e-7, ("cell {:.7f}{:+.7f}i vs fem_2d_tm {:.7f}{:+.7f}i: |dr| = {:.1e}, "
                                         "|dA_film| = {:.1e} (fem_2d_tm {:.1f} s, cell solve {:.2f} s)".format(
                                             res.r.real, res.r.imag, ref.r.real, ref.r.imag, dr, dA, t2 - t1,
                                             res.solve_s))
    h.run("B", gate_b)

    # ---- Table 1 + gates C, D (and E's table-1 part): exact vs equivalent at LAM0 ----
    print("[qfs] --- {:.0f} nm: equivalent layer (ii) vs exact film (iii) in the strip cell (and in the planar "
          "stack) ---".format(LAM0 * 1e9), flush=True)
    print("[qfs] bias | r (ii) | r (iii) | |dr| | A_film (ii) | A_film (iii) | dA | dA/A | |dr| planar | m=0-only "
          "|dr| | exact-film time", flush=True)
    worst_c = worst_d = worst_e = 0.0
    spectra = {}
    cl_pl = [TMLayer("oxb_in", 3e-9, EB), TMLayer("oxa", 2e-9, EA), TMLayer("gap", 40e-9, 1.0)]
    cellp = FilmCell(P, cl_pl, lambda_m=LAM0, order=4, n_harmonics=4)
    for V in BIASES_T1:
        resp = resps[V]
        eq, ex = FilmLayer.equivalent(resp), FilmLayer.exact(resp)
        diffs = []
        for cell in (cell4, cell5):
            a = cell.solve(_lower(eq), N_SUB ** 2)
            t1 = time.time()
            b80 = cell.solve(_lower(ex), N_SUB ** 2, film_harmonics=80, film_fallback=eq)
            t_ex = time.time() - t1
            b40 = cell.solve(_lower(ex), N_SUB ** 2, film_harmonics=MF, film_fallback=eq)
            worst_c = max(worst_c, abs(b40.r - b80.r))
            worst_e = max(worst_e, abs(a.energy_residual), abs(b80.energy_residual))
            diffs.append(b80.r - a.r)
        worst_d = max(worst_d, abs(diffs[0] - diffs[1]))
        b0 = cell5.solve(_lower(ex), N_SUB ** 2, film_harmonics=0, film_fallback=eq)
        pa = cellp.solve(_lower(eq), N_SUB ** 2)
        pb = cellp.solve(_lower(ex), N_SUB ** 2, film_harmonics=0, film_fallback=eq)
        spectra[V] = (b80, resp)
        Aa, Ab = a.absorption["film"], b80.absorption["film"]
        print("[qfs] ({:+.0f}, {:+.0f}) | {:.5f}{:+.5f}i | {:.5f}{:+.5f}i | {:.2e} | {:.5f} | {:.5f} | {:+.2e} | {:+.2%} | "
              "{:.1e} | {:.2e} | {:.1f} s".format(
                  V[0], V[1], a.r.real, a.r.imag, b80.r.real, b80.r.imag, abs(b80.r - a.r), Aa, Ab, Ab - Aa,
                  (Ab - Aa) / Ab, abs(pb.r - pa.r), abs(b0.r - a.r), t_ex), flush=True)
    h.gate("C", worst_c < 1e-7, "film harmonics 40 vs 80: max |dr| = {:.1e}".format(worst_c))
    h.gate("D", worst_d < 2e-6, "exact - equivalent, order 4 vs 5 cell: max change {:.1e}".format(worst_d))

    # ---- Table 2: the lateral spectrum of the field in the film; q -> 0 assessment ----
    print("[qfs] --- lateral wavevector content of the film field (exact film, {:.0f} nm) ---".format(LAM0 * 1e9),
          flush=True)
    print("[qfs] bias | k_F [1/nm] | omega/v_F [1/nm] | |q| holding 50/90/99% of A_film [1/nm] | 99% of |D_z|^2 "
          "| <(q/k_F)^2>_A | <(q v_F/omega)^2>_A | in-plane Lindhard: |dr|, dA", flush=True)
    for V in BIASES_T1:
        res, resp = spectra[V]
        s_ = film_spectrum(res)
        q, A, Dz = np.abs(s_["q_m"]), s_["A_m"], s_["Dz2"]
        o = np.argsort(q)
        cumA = np.cumsum(A[o]) / np.sum(A)
        cumD = np.cumsum(Dz[o]) / np.sum(Dz)
        nmax = float(np.max(resp.film.density_m3))
        kF = (3.0 * math.pi ** 2 * nmax) ** (1.0 / 3.0)
        vF = HBAR * kF / (M_OPT * M_E)
        qa = [q[o][min(np.searchsorted(cumA, x), q.size - 1)] for x in (0.5, 0.9, 0.99)]
        qd = q[o][min(np.searchsorted(cumD, 0.99), q.size - 1)]
        wA = np.abs(A) / np.sum(np.abs(A))
        m_kf = float(np.sum(wA * (q / kF) ** 2))
        m_vf = float(np.sum(wA * (q * vF / w0) ** 2))
        lf = cell5.solve(_lower(_lindhard_film(resp)), N_SUB ** 2, film_harmonics=MF,
                         film_fallback=FilmLayer.equivalent(resp))
        print("[qfs] ({:+.0f}, {:+.0f}) | {:.3f} | {:.3f} | {:.4f} / {:.4f} / {:.4f} | {:.4f} | {:.1e} | {:.1e} | "
              "{:.1e}, {:+.1e}".format(V[0], V[1], kF * 1e-9, w0 / vF * 1e-9, qa[0] * 1e-9, qa[1] * 1e-9,
                                       qa[2] * 1e-9, qd * 1e-9, m_kf, m_vf, abs(lf.r - res.r),
                                       lf.absorption["film"] - res.absorption["film"]), flush=True)

    # ---- F: the pipeline's FEM route vs the cell solver (equivalent layer) ----
    def gate_f():
        from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
        from dynameta.geometry.cross_section import Rectangle
        from dynameta.geometry.specs import OpticalSpec
        from dynameta.materials import ConstantOptical, Material, MaterialRegistry
        from dynameta.optics.fem_2d_tm import make_fem_2d_tm_optical_solver
        from dynameta.optics.quantum_film_pipeline import QuantumFilmOptical, run_quantum_film_pipeline
        from dynameta.sweep import BiasPoint, Sweep
        reg = MaterialRegistry()
        for nm, e in (("air", 1.0), ("glass", N_SUB ** 2), ("oxA", EA), ("oxB", EB), ("metal", E_STRIP),
                      ("mirror_metal", E_MIRROR)):
            reg.add(Material(nm, ConstantOptical(complex(e))))
        reg.add(Material("FILM", QuantumFilmOptical(eps_inf=EPS_INF, gamma_rad_s=GAM)))
        layers = [Layer(n, t, m) for n, t, m in (("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"),
                                                 ("lo_oxb", 6e-9, "oxB"), ("film", T_FILM, "FILM"),
                                                 ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA"))]
        layers.append(Layer("grating", 40e-9, "air", inclusions=[Inclusion(Rectangle(P / 2, P / 2, W, P), "metal")]))
        d = Design(name="strip", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                   stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                   electrodes=[], materials=reg, optical=OpticalSpec(polarization="x"))
        bps = [BiasPoint({"bottom": v[0], "top": v[1]}, "b{}".format(k)) for k, v in enumerate(BIASES_T1)]
        t1 = time.time()
        rows, _c = run_quantum_film_pipeline(d, Sweep(bias_points=bps, wavelengths_nm=[LAM0 * 1e9]), sp,
                                             film_layer="film", optical_solver=make_fem_2d_tm_optical_solver(num_threads=2),
                                             verbose=False)
        t_p = time.time() - t1
        worst = 0.0
        for V, row in zip(BIASES_T1, rows):
            ref = cell5.solve(_lower(FilmLayer.equivalent(resps[V])), N_SUB ** 2)
            worst = max(worst, abs(row.result.r - ref.r))
        return worst < 1e-4, "pipeline (fem_2d_tm order 4) vs cell, 5 biases: max |dr| = {:.1e} ({:.1f} s for 5 points)".format(
            worst, t_p)
    h.run("F", gate_f)

    # ---- the model-comparison sweep, short-wavelength band (the other two bands are their own parts) ----
    lams = band_lams("short")
    print("[qfs] --- model comparison: {} biases x {} wavelengths ({:.0f}-{:.0f} nm; bands {:.0f}-{:.0f} and "
          "{:.0f}-{:.0f} nm in quantum_film_strip_sweep_mid / _long) ---".format(
              len(BIASES_T3), lams.size, lams[0] * 1e9 if lams.size else float("nan"),
              lams[-1] * 1e9 if lams.size else float("nan"), *BANDS_NM["mid"], *BANDS_NM["long"]), flush=True)
    import gc
    cell4 = cell5 = cellp = spectra = None                # release the factorizations (~1 GB each)
    gc.collect()
    R, _Af, worst_sweep, times = sweep("qfs", lams, films, resps, drude, t_start)
    h.gate("E", max(worst_e, worst_sweep) < 1e-10,
           "max |1 - R - T - sum A| over every cell solve ({:.0f} nm tables: {:.1e}; sweep {:.0f}-{:.0f} nm: "
           "{:.1e})".format(LAM0 * 1e9, worst_e, *BANDS_NM["short"], worst_sweep))
    if lams.size:
        print_differences("qfs", R, "over {:.0f}-{:.0f} nm".format(lams[0] * 1e9, lams[-1] * 1e9))
    print_times("qfs", lams, times, t_start)
    return h.summary()


if __name__ == "__main__":
    raise SystemExit(main())
