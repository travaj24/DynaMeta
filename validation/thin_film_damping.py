"""
Thin-film free-carrier damping and density-dependent background permittivity (v0.16):
materials.scattering.BoundaryScattering / fuchs_sondheimer_factor and materials.optical_model
DrudeOptical(boundary=, background=), BandFillingBackground.

Part 1 -- size-effect damping.
  * The in-plane Fuchs-Sondheimer factor (exact, complex kappa at optical frequency) against scipy
    quadrature of its defining integral, and its thick-film limit (an additive rate 3 (1 - p) v_F / (8 d)).
  * CALIBRATION of the out-of-plane Kreibig-type rate A v_F / L against the self-consistent quantum
    response (optics.quantum_film, RPA), which contains the Landau-type damping of the confined carriers
    through the sub-band spectrum. Fixture = validation/quantum_film_rpa.py's (the generic gated-film cell):
    6 nm film, n_bg 3e26 m^-3, m* 0.30 m_e, eps_inf 4.0, eps_static 9.0, bulk rate 1.2e14 rad/s, 300 K, dual
    gate through 6 nm oxide (static eps 20), biases (Vb, Vt) = (0,0), (+2,+2), (-2,-2), (0,+3), (0,-3);
    optical stack air | 6 nm n 2.00 | film | 6 nm n 2.00 | glass n 1.45, p-pol, 60 deg. Both models see the
    SAME ground state n(z).
    Three measures of the out-of-plane linewidth / loss, each solved for A (film length 2d and the local
    heuristic length):
      FWHM  of the main peak of Im alpha_perp(omega) (the out-of-plane loss function of the film);
      EQW   equivalent width 2 int Im alpha_perp domega / (pi max Im alpha_perp) (counts fragmentation);
      A1200 the p-pol absorbed fraction at 1200 nm (the in-plane response is the same local Drude in both).
    Plus flat-band 10 / 20 nm films (A1200 only) for the thickness dependence.
Part 2 -- background permittivity eps_b(n): the closed-form band-filling model against the Kramers-Kronig
  quadrature of its eps2; anchoring; comparison with published In2O3 material data (reported); the shift of
  the density at which Re(eps) = 0 at 1200 nm for the generic film.

GATES (PASS/FAIL): A FS kernel vs quadrature (1e-10); B FS optical thick-film limit (1e-6); C uniaxial cell
cascade == nonlocal_tmm.stack_rt for an isotropic film (1e-12; the calibration's local solver); D local
model at A = 0 through the library rate kernel == the plain bulk Drude (exact); E band-filling chi vs KK
quadrature (1e-9); F background anchored (exact) and decreasing with density; G calibration ran on every
bias (the A table itself is REPORTED, not gated -- it is the finding).

MEASURED (Windows, BLAS 2 threads, 70 s): all gates PASS. Calibration: the RPA loss peak FWHM is 1.20-1.30e14
rad/s (= the bulk rate; 2.02e13 at a 2e13 bulk rate) while the local film at A = 0 already has 1.44-2.43e14,
so no A >= 0 matches any linewidth; the 1200 nm absorption of the RPA (0.0018-0.0065) is 1.2-2.0x the local
film's at A = 0 and is reached within A <= 3 only at (-2, -2) V (A = 2.17 film length / 0.51 local length);
flat band 10 / 20 nm: 1.72 / 1.35 (film). See docs/audit/2026-11-14-v0.16-thin-film-damping-and-background.md.

Run: python -m validation.thin_film_damping
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers import DualGateSchrodingerPoisson, GateStack  # noqa: E402
from dynameta.constants import C_LIGHT, EPS0, HBAR, M_E, Q_E  # noqa: E402
from dynameta.core.numerics import trapz  # noqa: E402
from dynameta.materials import (BandFillingBackground, BoundaryScattering, DrudeOptical,  # noqa: E402
                                KaneOpticalMass, fs_effective_rate, fuchs_sondheimer_factor)
from dynameta.optics import nonlocal_tmm as nt  # noqa: E402
from dynameta.optics.nonlocal_tmm import fermi_velocity  # noqa: E402
from dynameta.optics.quantum_film import (QuantumFilm, QuantumFilmResponse, film_stack_rt,  # noqa: E402
                                          uniaxial_layer_matrix)

M = 0.30 * M_E
EPS_INF = 4.0
GAM = 1.2e14
D_FILM = 6e-9
N_BG = 3e26
LAM = 1200e-9
W12 = 2.0 * math.pi * C_LIGHT / LAM
ABOVE = [nt.DielectricLayer(2.00 ** 2, 6e-9)]
BELOW = [nt.DielectricLayer(2.00 ** 2, 6e-9)]
KW = dict(n_sub=1.45, theta_rad=math.radians(60.0))
BIASES = ((0.0, 0.0), (2.0, 2.0), (-2.0, -2.0), (0.0, 3.0), (0.0, -3.0))
W = np.linspace(0.2e15, 3.5e15, 1651)                     # loss-function grid [rad/s], step 2e12
A_SCAN = np.linspace(0.0, 3.0, 61)


def _p(msg):
    print("[tfd] " + msg, flush=True)


# ------------------------------------------------------------------------------------------------
# Part 1 helpers
# ------------------------------------------------------------------------------------------------
def _fs_quad(kappa, p):
    from scipy.integrate import quad
    k = complex(kappa)
    if k.imag == 0.0:
        a = k.real
        f = lambda t: (t ** -3 - t ** -5) * (1 - math.exp(-a * t)) / (1 - p * math.exp(-a * t))  # noqa: E731
        return 1.0 - 1.5 * (1 - p) / a * quad(f, 1.0, np.inf, limit=500, epsabs=1e-15, epsrel=1e-13)[0]
    a, b = k.real, -k.imag
    f = lambda t: (t ** -3 - t ** -5) * math.exp(-a * t)  # noqa: E731
    re = quad(f, 1.0, np.inf, weight="cos", wvar=b)[0]
    im = quad(f, 1.0, np.inf, weight="sin", wvar=b)[0]
    return 1.0 - 1.5 / k * (0.25 - (re + 1j * im))


def _perp_rate_profile(film, A, mode):
    """Gamma_zz(z) = GAM + A v_F(z) / L(z) from the LIBRARY kernel (BoundaryScattering.perp_rate)."""
    n = film.density_m3
    if A == 0.0:
        return np.full(n.size, GAM)
    b = BoundaryScattering(in_plane=False, perp_coefficient=float(A), perp_length=mode)
    vF = fermi_velocity(np.maximum(n, 1e10), 1.0) / M
    return GAM + b.perp_rate(n, vF, film.thickness_m, film.z_m)


def _alpha_local(film, gz):
    """Out-of-plane sheet response length of the local film, int (1 - eps_inf / eps_zz(z)) dz, on W."""
    n, z = film.density_m3, film.z_m
    wp2 = n * Q_E ** 2 / (EPS0 * M)
    ezz = EPS_INF - wp2[None, :] / (W[:, None] * (W[:, None] + 1j * gz[None, :]))
    y = 1.0 - EPS_INF / ezz
    return np.array([trapz(r.real, z) + 1j * trapz(r.imag, z) for r in y])


def _widths(im):
    i = int(np.argmax(im))
    h = 0.5 * im[i]
    lo = i
    while lo > 0 and im[lo] > h:
        lo -= 1
    hi = i
    while hi < im.size - 1 and im[hi] > h:
        hi += 1
    wl = np.interp(h, [im[lo], im[lo + 1]], [W[lo], W[lo + 1]])
    wh = np.interp(h, [im[hi], im[hi - 1]], [W[hi], W[hi - 1]])
    return W[i], wh - wl, 2.0 * trapz(im, W) / (math.pi * im[i])


def _cascade(cells, w, kp):
    """Uniaxial cell cascade, superstrate side first: cells = [(eps_xx, eps_zz, dz), ...]."""
    T = np.eye(2, dtype=complex)
    for exx, ezz, dz in cells:
        T = uniaxial_layer_matrix(exx, ezz, dz, complex(w) / C_LIGHT, kp) @ T
    return T


def _local_cells(film, gz, w):
    """Cells of the local film (top face toward the superstrate): eps_xx = bulk Drude (as the RPA's
    in-plane response), eps_zz with the rate profile gz; cell values are the node means (xx) and the
    harmonic node means (zz, series)."""
    n, z = film.density_m3[::-1], film.z_m[::-1]
    g = gz[::-1]
    wp2 = n * Q_E ** 2 / (EPS0 * M)
    exx = EPS_INF - wp2 / (w * (w + 1j * GAM))
    ezz = EPS_INF - wp2 / (w * (w + 1j * g))
    return [(0.5 * (exx[j] + exx[j + 1]), 2.0 / (1.0 / ezz[j] + 1.0 / ezz[j + 1]), abs(z[j] - z[j + 1]))
            for j in range(z.size - 1)]


def _A1200_local(film, gz):
    cells = _local_cells(film, gz, W12)
    return film_stack_rt(W12, lambda ww, kp: _cascade(cells, ww, kp), ABOVE, BELOW, **KW).A


def _solve_A(metric, target):
    """Smallest A in A_SCAN's range with metric(A) = target (linear interpolation then bisection);
    None when the scan never reaches the target."""
    vals = np.array([metric(a) for a in A_SCAN])
    s = np.sign(vals - target)
    idx = np.flatnonzero(s[:-1] * s[1:] <= 0)
    if idx.size == 0:
        return None, vals
    i = int(idx[0])
    lo, hi = A_SCAN[i], A_SCAN[i + 1]
    flo = vals[i] - target
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        fm = metric(mid) - target
        if (fm <= 0) == (flo <= 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi), vals


def _fmt_A(a):
    return "  none " if a is None else "{:6.3f}".format(a)


# ------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    ok = True
    _p("=== thin-film damping and background permittivity ===")

    # ---- GATE A: FS kernel vs quadrature ----
    pts = [(0.1, 0.0), (0.901, 0.0), (10.0, 0.0), (1.0, 0.5), (0.3, 0.9), (0.901 - 11.8j, 0.0),
           (0.05 - 3.0j, 0.0), (1.0 - 70.0j, 0.0)]
    errA = max(abs(complex(fuchs_sondheimer_factor(k, p)) - _fs_quad(k, p)) for k, p in pts)
    gA = errA < 1e-10
    _p("GATE A (Fuchs-Sondheimer F vs quadrature, {} points incl. complex kappa): max |dF| = {:.1e} -> {}"
       .format(len(pts), errA, "PASS" if gA else "FAIL"))

    # ---- GATE B: optical thick-film limit ----
    vF0 = fermi_velocity(N_BG, M)
    errB = 0.0
    for d, p in ((2e-6, 0.0), (5e-6, 0.4)):
        ge = fs_effective_rate(W12, GAM, vF0, d, p)
        want = GAM + 3.0 * (1.0 - p) * vF0 / (8.0 * d)
        errB = max(errB, abs(ge - want) / want)
    gB = errB < 1e-6
    _p("GATE B (FS at 1200 nm, d = 2 / 5 um: gamma_eff = gamma + 3 (1-p) v_F / (8 d)): rel {:.1e} -> {}"
       .format(errB, "PASS" if gB else "FAIL"))

    # ---- reference case ----
    kF = (3.0 * math.pi ** 2 * N_BG) ** (1.0 / 3.0)
    l_mfp = vF0 / GAM
    k_dc = D_FILM / l_mfp
    k_opt = D_FILM * (GAM - 1j * W12) / vF0
    F_dc = complex(fuchs_sondheimer_factor(k_dc)).real
    F_opt = complex(fuchs_sondheimer_factor(k_opt))
    g_dc = fs_effective_rate(0.0, GAM, vF0, D_FILM).real
    g_opt = fs_effective_rate(W12, GAM, vF0, D_FILM)
    _p("--- reference case: d = {:.0f} nm, n = {:.0e} m^-3, m* = {:.2f} m_e, bulk rate {:.1e} rad/s ---".format(
        D_FILM * 1e9, N_BG, M / M_E, GAM))
    _p("  v_F = {:.4e} m/s, k_F d = {:.2f}, bulk mean free path l = {:.3f} nm".format(vF0, kF * D_FILM, l_mfp * 1e9))
    _p("  d.c.: kappa = d/l = {:.4f}, F = {:.4f}, in-plane rate gamma/F = {:.4e} rad/s (+{:.4e})".format(
        k_dc, F_dc, g_dc, g_dc - GAM))
    _p("  1200 nm: kappa = {:.4f}{:+.4f}i, F = {:.6f}{:+.6f}i, gamma_eff = {:.4e}{:+.3e}i rad/s (Re +{:.4e}; "
       "thick-film formula 3 v_F/(8d) = {:.4e})".format(k_opt.real, k_opt.imag, F_opt.real, F_opt.imag,
                                                      g_opt.real, g_opt.imag, g_opt.real - GAM,
                                                      3 * vF0 / (8 * D_FILM)))
    _p("  out of plane (film length 2d): A v_F / (2d) = A x {:.4e} rad/s".format(vF0 / (2 * D_FILM)))
    zf = np.linspace(0.0, D_FILM, 201)
    nf = np.full(zf.size, N_BG)
    absorbed = {}
    for name, bnd in (("bulk Drude", None),
                      ("+ in-plane FS (p = 0)", BoundaryScattering(thickness_m=D_FILM)),
                      ("+ in-plane FS (p = 0.5)", BoundaryScattering(specularity=0.5, thickness_m=D_FILM)),
                      ("+ out-of-plane A = 0.75", BoundaryScattering(in_plane=False, perp_coefficient=0.75,
                                                                     thickness_m=D_FILM)),
                      ("+ both (p = 0, A = 0.75)", BoundaryScattering(perp_coefficient=0.75, thickness_m=D_FILM))):
        d = DrudeOptical(EPS_INF, M, GAM, boundary=bnd)
        T = d.eps_tensor(LAM, n_m3=nf, z_axis_m=zf)
        exx, ezz = complex(T[0, 0, 0]), complex(T[0, 2, 2])
        r = film_stack_rt(W12, lambda ww, kp, a=exx, b=ezz: uniaxial_layer_matrix(a, b, D_FILM, complex(ww) / C_LIGHT,
                                                                                   kp), ABOVE, BELOW, **KW)
        absorbed[name] = r.A
        _p("  uniform film, p-pol 60 deg, 1200 nm, {:26s}: eps_xx = {:.4f}{:+.4f}i, eps_zz = {:.4f}{:+.4f}i, "
           "A = {:.5f} ({:+.1f} %)".format(name, exx.real, exx.imag, ezz.real, ezz.imag, r.A,
                                           100 * (r.A / absorbed["bulk Drude"] - 1)))

    # ---- calibration fixture ----
    s = DualGateSchrodingerPoisson(t_film_m=D_FILM, bottom=GateStack((6e-9, 20.0)), top=GateStack((6e-9, 20.0)),
                                   n_bg_m3=N_BG, m_eff_kg=M, eps_static=9.0, nz=481)
    films = {k: QuantumFilm.from_dual_gate(s, s.solve_voltages(*k)) for k in BIASES}

    # ---- GATE C: the local solver of the calibration (uniaxial cascade) vs nonlocal_tmm ----
    f0 = films[(0.0, 0.0)]
    cells = _local_cells(f0, np.full(f0.z_m.size, GAM), W12)
    iso = [(c[0], c[0], c[2]) for c in cells]
    r_c = film_stack_rt(W12, lambda ww, kp: _cascade(iso, ww, kp), ABOVE, BELOW, **KW).r
    # nonlocal_tmm.stack_rt returns the tangential-E r, the negative of film_stack_rt's p-hat r
    r_n = -nt.stack_rt(W12, ABOVE + [nt.DielectricLayer(c[0], c[2]) for c in iso] + BELOW, **KW).r
    gC = abs(r_c - r_n) < 1e-12
    _p("GATE C (uniaxial cell cascade == nonlocal_tmm.stack_rt, isotropic local film): |dr| = {:.1e} -> {}"
       .format(abs(r_c - r_n), "PASS" if gC else "FAIL"))

    # ---- GATE D: A = 0 through the library kernel is the plain bulk rate ----
    gz1 = GAM + BoundaryScattering(in_plane=False, perp_coefficient=0.0, perp_length="local").perp_rate(
        f0.density_m3, fermi_velocity(np.maximum(f0.density_m3, 1e10), 1.0) / M, f0.thickness_m, f0.z_m)
    gD = bool(np.all(gz1 == GAM))
    _p("GATE D (A = 0 leaves the bulk rate exactly) -> {}".format("PASS" if gD else "FAIL"))

    # ---- calibration table ----
    _p("--- CALIBRATION of A (Gamma_zz = {:.1e} + A v_F / L) against the RPA, {:.0f} nm dual-gate film ---".format(
        GAM, D_FILM * 1e9))
    _p("  bias      | RPA: peak [1e15]  FWHM [1e14]  EQW [1e14]  A1200   | local A=0: FWHM  EQW  A1200 |"
       " A from FWHM / EQW / A1200  (film | local)")
    table = []
    for key, f in films.items():
        resp = QuantumFilmResponse(f, EPS_INF, GAM)
        ar = np.array([resp.alpha_perp(w) for w in W])
        pk, fw, eq = _widths(ar.imag)
        a_rpa = resp.stack_rt(W12, ABOVE, BELOW, **KW).A
        g0 = _perp_rate_profile(f, 0.0, "film")
        _, fw0, eq0 = _widths(_alpha_local(f, g0).imag)
        a0 = _A1200_local(f, g0)
        row = {"bias": key, "rpa": (pk, fw, eq, a_rpa), "loc0": (fw0, eq0, a0)}
        for mode in ("film", "local"):
            fwA, _ = _solve_A(lambda A, f=f, mode=mode: _widths(_alpha_local(f, _perp_rate_profile(f, A, mode)).imag)[1],
                              fw)
            eqA, _ = _solve_A(lambda A, f=f, mode=mode: _widths(_alpha_local(f, _perp_rate_profile(f, A, mode)).imag)[2],
                              eq)
            abA, _ = _solve_A(lambda A, f=f, mode=mode: _A1200_local(f, _perp_rate_profile(f, A, mode)), a_rpa)
            row[mode] = (fwA, eqA, abA)
        table.append(row)
        _p("  ({:+.0f},{:+.0f}) V  |      {:.4f}       {:.3f}       {:.3f}     {:.5f} |   {:.3f}  {:.3f}  {:.5f} |"
           " {} {} {} | {} {} {}".format(key[0], key[1], pk / 1e15, fw / 1e14, eq / 1e14, a_rpa, fw0 / 1e14,
                                         eq0 / 1e14, a0, *[_fmt_A(x) for x in row["film"]],
                                         *[_fmt_A(x) for x in row["local"]]))
    # the RPA's own width at a smaller bulk rate: the intrinsic (Landau-type) part of its main mode
    resp_s = QuantumFilmResponse(f0, EPS_INF, 2e13)
    _, fw_s, eq_s = _widths(np.array([resp_s.alpha_perp(w) for w in W]).imag)
    _p("  RPA (0,0) V with bulk rate 2e13: FWHM {:.3e} (grid step 2e12), EQW {:.3e} rad/s -> the main mode's "
       "width is the bulk rate; the confined spectrum adds < ~1e13 rad/s".format(fw_s, eq_s))
    for mode in ("film", "local"):
        a13 = [r[mode][2] for r in table if r[mode][2] is not None]
        if a13:
            _p("  A1200 ({}): {} of 5 biases solvable, A = {:.3f} .. {:.3f} (mean {:.3f}, rel spread {:.0f} %)".format(
                mode, len(a13), min(a13), max(a13), np.mean(a13), 100 * np.std(a13) / np.mean(a13)))
        nfw = sum(r[mode][0] is not None for r in table)
        _p("  FWHM ({}): solvable for {} of 5 biases (the local film at A = 0 is already WIDER than the RPA "
           "peak at every bias)".format(mode, nfw))
    # thickness dependence, flat band
    for t, nz in ((10e-9, 401), (20e-9, 801)):
        ff = QuantumFilm.flat_band(t, N_BG, M, nz=nz)
        a_rpa = QuantumFilmResponse(ff, EPS_INF, GAM).stack_rt(W12, ABOVE, BELOW, **KW).A
        res = []
        for mode in ("film", "local"):
            a, _ = _solve_A(lambda A, f=ff, mode=mode: _A1200_local(f, _perp_rate_profile(f, A, mode)), a_rpa)
            res.append(_fmt_A(a))
        _p("  flat band {:2.0f} nm: A_RPA(1200) = {:.5f}, local A=0: {:.5f}; A from A1200 film | local = {} | {}"
           .format(t * 1e9, a_rpa, _A1200_local(ff, np.full(ff.z_m.size, GAM)), *res))
    gG = len(table) == len(BIASES)

    # ---- Part 2 ----
    _p("--- background permittivity eps_b(n) ---")
    from scipy.integrate import quad
    bg = BandFillingBackground(3.75, 2.5, 1e26, KaneOpticalMass(m0_kg=0.18 * M_E, alpha_eV=0.5),
                               valence_mass_kg=0.6 * M_E, bgr_coeff_eV_m=2e-10)
    errE = 0.0
    for n in (1e25, 3e26, 1.2e27):
        for lam in (2000e-9, 1200e-9, 600e-9):
            E = 2 * math.pi * HBAR * C_LIGHT / lam / Q_E
            Eg = float(bg.renormalized_gap_J(n)) / Q_E
            Elo = float(bg.optical_gap_J(n)) / Q_E
            f = lambda e: e * 2.5 * math.sqrt(max(e - Eg, 0) / 3.75) * (3.75 / e) ** 2 / (e * e - E * E)  # noqa: E731
            want = (2 / math.pi) * (quad(f, Elo, 50 * Eg, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
                                    + quad(f, 50 * Eg, np.inf, limit=200, epsabs=1e-14)[0])
            errE = max(errE, abs(float(bg.interband_chi(n, lam)) - want) / abs(want))
    gE = errE < 1e-9
    _p("GATE E (band-filling chi vs Kramers-Kronig quadrature of its eps2, Kane band): rel {:.1e} -> {}".format(
        errE, "PASS" if gE else "FAIL"))
    nn = np.geomspace(1e24, 2e27, 9)
    e_b = bg.eps_background(nn, LAM, EPS_INF)
    gF = bool(bg.eps_background(1e26, LAM, EPS_INF) == EPS_INF) and bool(np.all(np.diff(
        BandFillingBackground(3.75, 2.5, 1e26, KaneOpticalMass(m0_kg=0.18 * M_E, alpha_eV=0.5)
                              ).eps_background(nn, LAM, EPS_INF)) < 0))
    _p("GATE F (anchored at n_ref exactly; pure band filling lowers eps_b monotonically) -> {}".format(
        "PASS" if gF else "FAIL"))
    _p("  (example, Kane 0.18 m_e / 0.5 eV^-1, m_v 0.6 m_e, C_bgr 2e-10 eV m, A 2.5, n_ref 1e26) eps_b(1200 nm): "
       + ", ".join("{:.0e}: {:.3f}".format(a, b) for a, b in zip(nn, e_b)))

    # literature comparison: published In2O3 material data
    kane = KaneOpticalMass(m0_kg=0.18 * M_E, alpha_eV=0.5)       # Feneberg et al. PRB 93, 045203 (2016)
    nlit = np.array([1e25, 1e26, 3e26, 6e26, 1e27])
    pure = BandFillingBackground(3.75, 1.0, 1e23, kane)
    shift_pure = (pure.optical_gap_J(nlit) / Q_E - 3.75)
    c_fit = (float(pure.band_filling_shift_J(1e27)) / Q_E - 0.8) / 1e9
    fit = BandFillingBackground(3.75, 1.0, 1e23, kane, bgr_coeff_eV_m=c_fit)
    shift_fit = (fit.optical_gap_J(nlit) / Q_E - 3.75)
    _p("  optical-gap widening (Kane band of Feneberg 2016, flat valence band), n [m^-3] -> eV:")
    _p("    band filling only : " + ", ".join("{:.0e}: {:.3f}".format(a, b) for a, b in zip(nlit, shift_pure)))
    _p("    + renormalization C_bgr = {:.3e} eV m (ONE-parameter fit to a 0.8 eV widening at 1e27): ".format(c_fit)
       + ", ".join("{:.0e}: {:.3f}".format(a, b) for a, b in zip(nlit, shift_fit)))
    _p("    published: widening up to ~0.8 eV for Sn-doped In2O3 films up to ~1e27 m^-3 from an onset of 3.75 eV "
       "(Hamberg et al., PRB 30, 3240 (1984)); band filling alone overshoots, the renormalization is required")
    for lam in (1200e-9, 5000e-9):
        drops = []
        for A in (1.0, 2.5, 4.0):
            m_ = BandFillingBackground(3.75, A, 1e23, kane, bgr_coeff_eV_m=c_fit)
            drops.append(float(m_.eps_background(1e27, lam, 0.0)))
        _p("  eps_b(1e27) - eps_b(undoped) at {:.0f} nm for edge amplitude A = 1 / 2.5 / 4: {:+.3f} / {:+.3f} / "
           "{:+.3f}".format(lam * 1e9, *drops))
    _p("    published: undoped In2O3 eps_inf 4.05-4.08 (Stokey et al. 2021 / Feneberg et al. 2016); degenerate "
       "Sn-doped In2O3 films 3.4-4.0 (Gerfin and Gratzel, J. Appl. Phys. 79, 1722 (1996)), with +-0.5 film-to-film "
       "scatter tied to "
       "microstructure (a 2023 deposition-condition study, PMC10343510: 4.31-5.42, not monotonic in N)")

    # zero crossing of Re(eps) at 1200 nm for the generic film
    from scipy.optimize import brentq
    base = DrudeOptical(EPS_INF, M, GAM)
    nc0 = brentq(lambda n: base.eps(LAM, n_m3=n).real, 1e26, 1e28, xtol=1e20)
    _p("  Re(eps) = 0 at 1200 nm, generic film (m* {:.2f} m_e parabolic, gamma {:.1e}, eps_inf {:.1f} at n_ref {:.0e}):"
       " constant background n_c = {:.4e} m^-3".format(M / M_E, GAM, EPS_INF, N_BG, nc0))
    for A in (1.0, 2.5, 4.0):
        for C in (0.0, c_fit):
            bgm = BandFillingBackground(3.75, A, N_BG, M, bgr_coeff_eV_m=C)
            nc = brentq(lambda n: DrudeOptical(EPS_INF, M, GAM, background=bgm).eps(LAM, n_m3=n).real,
                        1e26, 1e28, xtol=1e20)
            _p("    A = {:.1f}, C_bgr = {:.2e}: n_c = {:.4e} ({:+.2f} %), eps_b(n_c) = {:.4f}".format(
                A, C, nc, 100 * (nc / nc0 - 1), float(bgm.eps_background(nc, LAM, EPS_INF))))

    ok = gA and gB and gC and gD and gE and gF and gG
    _p("total {:.1f} s -> {}".format(time.time() - t0, "PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
