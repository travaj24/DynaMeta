"""Solver-free gates for optics/quantum_film_cell.py: everything downstream of the FEM factorization.

FilmCell's constructor meshes and factorizes the patterned region with NGSolve (solver leg only), but
the half the module docstring calls "any stack below costs only its admittances and one dense
(2M+1)^2 solve" -- the per-harmonic stack admittance, the Woodbury bottom-port update, the reflection
reference-plane shift, the per-harmonic Poynting bookkeeping and the film spectrum -- is pure numpy.
These gates drive that half with the EXACT port data of the simplest FEM region there is: a
homogeneous superstrate-filled slab, whose port responses are closed form (derived in
_analytic_region below), so the whole pipeline must reproduce an ordinary planar transfer-matrix
result:

  * isotropic stacks: the `tmm` package (r, T and the per-layer absorption);
  * uniaxial stacks at oblique incidence: an H_y transfer matrix written in this file (_h_tmm);
  * the exact quantum-film element: QuantumFilmResponse.stack_rt (the planar route of
    optics.quantum_film), which shares no code with the bottom-port / flux bookkeeping here.

Tolerances: every quantity is a closed-form expression evaluated in double precision, so agreement is
at round-off (~1e-13); 1e-10 leaves margin for the conditioning of the 80 nm metal layer (|cos kz d|
~ e^5) and is still ~6 orders below any physical effect.
"""
import cmath
import math

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, M_E
from dynameta.materials.optical_model import DrudeOptical
from dynameta.optics.quantum_film import QuantumFilm, QuantumFilmResponse, uniaxial_layer_matrix
from dynameta.optics.quantum_film_cell import (CellMeshSpec, FilmCell, FilmCellResult, FilmLayer,
                                               UniformLayer, _graded_points, film_spectrum,
                                               stack_admittance)

LAM = 1200e-9
OMEGA = 2.0 * math.pi * C_LIGHT / LAM
K0 = OMEGA / C_LIGHT                                   # 1/m
P_M = 400e-9
EA, EB = 1.70 ** 2, 2.00 ** 2                          # the generic oxides oxA, oxB
EMIR = -120 + 20j                                      # the mirror metal
ESUB = 1.45 ** 2
EXX, EZZ = 2.7503 + 0.0955j, 2.5838 + 0.4766j          # a uniaxial film (the 0 V equivalent layer, 1200 nm)
EPS_INF, GAM, N_BG, M_OPT = 4.0, 1.2e14, 3e26, 0.30 * M_E
TOL = 1e-10


def _sq(z):
    """Passive square root (Im >= 0; +real on the real axis)."""
    w = cmath.sqrt(complex(z))
    return -w if (w.imag < 0 or (w.imag == 0 and w.real < 0)) else w


def _rel(A, B):
    """Largest entry-wise relative difference (the matrix entries span ~14 decades in SI units of
    the (E_x, H_y / (omega eps0)) basis, so an absolute norm would hide the small ones)."""
    A, B = np.asarray(A), np.asarray(B)
    return float(np.max(np.abs(A - B) / np.maximum(np.abs(B), 1e-300)))


def _h_tmm(layers_bottom_first, eps_sub, q, k0=K0):
    """INDEPENDENT oracle: (H, G = (1/eps_xx) dH/dz) at the top face of a uniaxial stack carrying the
    substrate's down-going wave exp(-i kz z) (z up). layers: [(exx, ezz, d)], bottom first. In a layer
    kz^2 = exx (k0^2 - q^2/ezz) and (H, G) is carried up by [[cos, sin/g], [-g sin, cos]], g = kz/exx."""
    ks = _sq(eps_sub * k0 * k0 - q * q)
    v = np.array([1.0, -1j * ks / eps_sub], dtype=complex)
    for exx, ezz, d in layers_bottom_first:
        kz = _sq(exx * (k0 * k0 - q * q / ezz))
        g = kz / exx
        c, s = cmath.cos(kz * d), cmath.sin(kz * d)
        v = np.array([[c, s / g], [-g * s, c]]) @ v
    return v


def _rh_planar(layers_bottom_first, eps_sub, q, eps_s=1.0, k0=K0):
    """H_y reflection ratio at the top face of the stack for incidence from a superstrate eps_s."""
    H, G = _h_tmm(layers_bottom_first, eps_sub, q, k0)
    ks = _sq(eps_s * k0 * k0 - q * q)
    # H = a (e^{-ikz} + r e^{ikz}) at z = 0: G/H = (-i ks/eps_s)(1 - r)/(1 + r)
    y = G / H * eps_s / (-1j * ks)
    return (1 - y) / (1 + y)


def _analytic_region(lambda_m, period_m, M, *, theta_deg=0.0, polarization="p", buffer_nm=40.0,
                     struct_nm=12.0):
    """A FilmCell whose FEM region is a homogeneous slab of the superstrate (eps_s = 1) of thickness
    buffer_nm (bottom port line z = 0, top port line z = buffer_nm), the 'patterned layers' being the
    lowest struct_nm of it. Its port data are closed form per harmonic m (k_m = sqrt(eps_s k0^2 -
    q_m^2), Im >= 0; the weak form of the module docstring with L = 1/eps_s):

      * X = A1^-1 B_b (unit bottom source, top radiation condition): (1/eps_s) U'(0) = -1 and U an
        up-going wave, U = (i eps_s / k_m) exp(i k_m z): Cb^T X = diag(U(0)), Ct^T X = diag(U(top));
      * y = A1^-1 f (unit incident H at the top line, natural (Neumann) bottom): U = exp(-i k (z - d))
        + exp(2 i k d) exp(i k (z - d)): Cb^T y = 2 exp(i k d) delta_m0, Ct^T y = (1 + exp(2 i k d))
        delta_m0.
    Laterally uniform, so the harmonics decouple and the cell must reproduce planar physics."""
    cell = object.__new__(FilmCell)
    P = period_m * 1e9
    k0 = 2.0 * math.pi / (lambda_m * 1e9)
    eps_s = 1.0 + 0j
    kx = k0 * math.sin(math.radians(theta_deg))
    ms = np.arange(-M, M + 1)
    q = kx + 2.0 * math.pi * ms / P
    km = np.array([cmath.sqrt(eps_s * k0 * k0 - qq * qq) for qq in q])
    d = float(buffer_nm)
    i0 = int(np.flatnonzero(ms == 0)[0])
    Gb = 1j * eps_s / km
    cby = np.zeros(ms.size, complex)
    cty = np.zeros(ms.size, complex)
    cby[i0] = 2.0 * cmath.exp(1j * km[i0] * d)
    cty[i0] = 1.0 + cmath.exp(2j * km[i0] * d)
    cell.__dict__.update(
        P=P, lambda_m=lambda_m, omega=2.0 * math.pi * C_LIGHT / lambda_m, k0=k0, eps_s=eps_s, kx=kx,
        theta_deg=theta_deg, polarization=polarization, M=M, ms=ms, q_nm=q, k_m=km, i0=i0, a0=1.0 + 0j,
        z_struct_top=float(struct_nm), z_top=d, ndof=0, CbX=np.diag(Gb),
        CtX=np.diag(Gb * np.exp(1j * km * d)), Cby=cby, Cty=cty)
    return cell


def _lower():
    return [UniformLayer("mirror", 80e-9, EMIR), UniformLayer("lo", 6e-9, EB),
            UniformLayer("oxa", 2e-9, EA + 0.05j)]


# ------------------------------------------------------------------------------------------------
# the laterally uniform layers
# ------------------------------------------------------------------------------------------------

def test_uniform_layer_matrix_is_the_closed_form_and_unimodular():
    """UniformLayer.matrix is the p-pol characteristic matrix [[cos, i kz^2 sin/(kz exx)], [i exx
    sin/kz, cos]] (nonlocal_tmm basis, superstrate face -> substrate face); it is unimodular (det = 1:
    reciprocity, which stack_admittance relies on) and even in k_par."""
    exx, ezz, d = EXX, EZZ, 6e-9
    lay = UniformLayer("f", d, (exx, ezz))
    assert lay.components() == (exx, ezz)
    assert UniformLayer("i", d, 2.0).components() == (2.0, 2.0)
    q = 0.8 * K0
    kz = _sq(exx * (K0 ** 2 - q ** 2 / ezz))
    ref = np.array([[cmath.cos(kz * d), 1j * kz * cmath.sin(kz * d) / exx],
                    [1j * exx * cmath.sin(kz * d) / kz, cmath.cos(kz * d)]])
    T = lay.matrix(OMEGA, q)
    assert _rel(T, ref) < 1e-12
    assert abs(np.linalg.det(T) - 1.0) < 1e-12
    assert np.max(np.abs(T - lay.matrix(OMEGA, -q))) == 0.0
    # decay exponent and the half-space admittance of the decaying (passive) wave
    assert abs(lay.decay(OMEGA, q) - abs(kz.imag) * d) < 1e-15
    assert abs(lay.halfspace_admittance(OMEGA, q) - exx / kz) < 1e-12 * abs(exx / kz)
    metal = UniformLayer("m", 1e-6, EMIR)
    kzm = _sq(EMIR * K0 ** 2)
    assert metal.halfspace_admittance(OMEGA, 0.0).imag * (EMIR / kzm).imag > 0
    assert abs(metal.halfspace_admittance(OMEGA, 0.0) - EMIR / kzm) < 1e-12 * abs(EMIR / kzm)


def test_stack_admittance_matches_an_independent_h_field_transfer_matrix():
    """Y = [(1/eps_xx) d_z H / H] at the top face of the lower stack (z up) equals G/H of an H_y
    transfer matrix built independently from the substrate's outgoing wave, for propagating and
    evanescent in-plane wavevectors; with no layers it is the substrate's -i kz/eps."""
    for q in (0.0, 0.6 * K0, 3.0 * K0):
        Y0, mats0, st0 = stack_admittance([], OMEGA, q, ESUB)
        assert mats0 == [] and st0 == 0
        assert abs(Y0 - (-1j * _sq(ESUB * K0 ** 2 - q * q) / ESUB)) < 1e-12 * abs(Y0)
        lower = _lower() + [UniformLayer("ani", 4e-9, (2.2 + 0.1j, 2.3 + 0.4j))]
        Y, mats, start = stack_admittance(lower, OMEGA, q, ESUB)
        tup = [(*L.components(), L.thickness_m) for L in lower]
        H, G = _h_tmm(tup, ESUB, q)
        assert start == 0 and all(m is not None for m in mats)
        assert abs(Y - G / H) < TOL * abs(G / H)


def test_stack_admittance_takes_an_opaque_layer_as_a_half_space():
    """A layer whose evanescent attenuation exceeds exp(-35) is replaced by a half-space: everything
    below it is dropped (matrices None, start index past it) and Y equals that of the same stack
    sitting directly on the opaque material to exp(-70) (here: round-off)."""
    thick = UniformLayer("mirror", 1e-6, EMIR)                # Im kz d ~ 58 > 35
    lower = [UniformLayer("glass", 50e-9, 2.1), thick, UniformLayer("lo", 6e-9, EB)]
    Y, mats, start = stack_admittance(lower, OMEGA, 0.4 * K0, ESUB)
    assert start == 2 and mats[0] is None and mats[1] is None and mats[2] is not None
    H, G = _h_tmm([(EB, EB, 6e-9)], EMIR, 0.4 * K0)
    assert abs(Y - G / H) < TOL * abs(G / H)


def test_film_layer_adapters():
    """FilmLayer.local_graded of a UNIFORM density is the single slab of the Drude permittivity at that
    density (a cascade of identical slabs is one slab); turning the profile over with top_is_last
    reverses the cascade; FilmLayer.equivalent is the uniaxial slab of QuantumFilmResponse.
    effective_layer; FilmLayer.exact is the response's transfer matrix and is even in q."""
    opt = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=M_OPT, gamma_rad_s=GAM)
    z = np.linspace(0.0, 6e-9, 13)
    flat = FilmLayer.local_graded(z, np.full(z.size, N_BG), opt)
    e = complex(opt.eps(LAM, n_m3=N_BG))
    q = 0.7 * K0
    ref = UniformLayer("u", 6e-9, e).matrix(OMEGA, q)
    assert flat.label == "local" and abs(flat.thickness_m - 6e-9) < 1e-24
    assert _rel(flat.matrix(OMEGA, q), ref) < 1e-12
    # graded profile: top_is_last=True with z ascending == top_is_last=False with the profile
    # listed from the superstrate face (reversed)
    n = N_BG + 1.2e27 * np.exp(-(6e-9 - z) / 1e-9)
    a = FilmLayer.local_graded(z, n, opt, top_is_last=True).matrix(OMEGA, q)
    b = FilmLayer.local_graded(z, n[::-1], opt, top_is_last=False).matrix(OMEGA, q)
    c = FilmLayer.local_graded(z, n, opt, top_is_last=False).matrix(OMEGA, q)
    assert _rel(a, b) < 1e-12
    assert _rel(a, c) > 1e-4                         # the profile IS asymmetric
    # the slab cascade by hand: interval means, superstrate face first; each matrix maps its
    # superstrate face to its substrate face, so deeper slabs multiply from the left
    zz, nn = (z[-1] - z)[::-1], n[::-1]
    T = np.eye(2, dtype=complex)
    for k in range(z.size - 1):
        ek = complex(opt.eps(LAM, n_m3=0.5 * (nn[k] + nn[k + 1])))
        T = uniaxial_layer_matrix(ek, ek, zz[k + 1] - zz[k], K0, q) @ T
    assert _rel(a, T) < 1e-12

    resp = QuantumFilmResponse(QuantumFilm.flat_band(6e-9, N_BG, M_OPT, nz=97), EPS_INF, GAM)
    eq = FilmLayer.equivalent(resp, "film")
    exx, ezz, L = resp.effective_layer(OMEGA)
    assert eq.label == "equivalent" and eq.thickness_m == L
    assert _rel(eq.matrix(OMEGA, q), UniformLayer("e", L, (exx, ezz)).matrix(OMEGA, q)) < 1e-14
    ex = FilmLayer.exact(resp, "film")
    Tq = ex.matrix(OMEGA, q)
    assert ex.label == "exact" and ex.symmetric_in_q
    assert _rel(Tq, resp.transfer_matrix(OMEGA, q)) < 1e-14
    assert _rel(ex.matrix(OMEGA, -q), Tq) < 1e-12
    assert abs(np.linalg.det(Tq) - 1.0) < 1e-9                  # reciprocal film element


# ------------------------------------------------------------------------------------------------
# the bottom port + bookkeeping of FilmCell.solve, on the closed-form region
# ------------------------------------------------------------------------------------------------

def _tmm_isotropic(lower, theta_deg, eps_sub=ESUB):
    tmm = pytest.importorskip("tmm")
    top_first = list(reversed(lower))
    n_list = [1.0] + [_sq(L.components()[0]) for L in top_first] + [_sq(eps_sub)]
    d_list = [np.inf] + [L.thickness_m * 1e9 for L in top_first] + [np.inf]
    data = tmm.coh_tmm("p", n_list, d_list, math.radians(theta_deg), LAM * 1e9)
    return data, tmm.absorp_in_each_layer(data)


@pytest.mark.parametrize("theta", [0.0, 35.0])
def test_closed_form_region_reproduces_the_tmm_library(theta):
    """FilmCell.solve on the homogeneous region over an isotropic metal/oxide stack: r (p-hat) at the
    top of the patterned layers is the tmm r_p shifted by exp(2 i kz s) (s = the patterned thickness
    the region keeps above the stack), R and T are tmm's, every layer's absorption is tmm's per-layer
    absorption, and R + T + sum A = 1."""
    cell = _analytic_region(LAM, P_M, 3, theta_deg=theta, polarization="p")
    res = cell.solve(_lower(), ESUB, absorption=False)
    data, absorb = _tmm_isotropic(_lower(), theta)
    kz0 = cell.k_m[cell.i0]
    assert abs(res.r - data["r"] * cmath.exp(2j * kz0 * cell.z_struct_top)) < TOL
    assert res.r == res.r_H
    assert abs(res.R - data["R"]) < TOL and abs(res.T - data["T"]) < TOL
    top_first = list(reversed(_lower()))
    for k, L in enumerate(top_first):
        assert abs(res.absorption[L.name] - absorb[k + 1]) < TOL, L.name
    assert abs(1.0 - res.R - res.T - sum(res.absorption.values())) < TOL
    assert math.isnan(res.energy_residual)                      # absorption=False: no FEM integral
    assert res.harmonics == 3 and isinstance(res, FilmCellResult)


def test_normal_incidence_x_polarization_is_the_lab_ex_ratio():
    """'x' at normal incidence reports r = -r_H, the E_x reflection: at normal incidence that is the
    s-polarized E reflection of the tmm library (r_s = -r_p there)."""
    tmm = pytest.importorskip("tmm")
    cell = _analytic_region(LAM, P_M, 2, polarization="x", struct_nm=0.0)
    res = cell.solve(_lower(), ESUB, absorption=False)
    top_first = list(reversed(_lower()))
    n_list = [1.0] + [_sq(L.components()[0]) for L in top_first] + [_sq(ESUB)]
    d_list = [np.inf] + [L.thickness_m * 1e9 for L in top_first] + [np.inf]
    rs = tmm.coh_tmm("s", n_list, d_list, 0.0, LAM * 1e9)["r"]
    assert abs(res.r - rs) < TOL and abs(res.r + res.r_H) < 1e-15


def test_oblique_uniaxial_stack_and_opaque_mirror():
    """Uniaxial layers at 50 deg over an opaque (1 um) metal: r_H is the independent H_y transfer
    matrix's (with the metal as the substrate), T = 0, and the half-space layer absorbs exactly what
    is neither reflected nor absorbed above it."""
    ani = (EXX, EZZ)
    lower = [UniformLayer("mirror", 1e-6, EMIR), UniformLayer("lo", 6e-9, EB),
             UniformLayer("film", 6e-9, ani), UniformLayer("up", 6e-9, EB)]
    cell = _analytic_region(LAM, P_M, 2, theta_deg=50.0, struct_nm=0.0)
    res = cell.solve(lower, ESUB, absorption=False)
    q = K0 * math.sin(math.radians(50.0))
    ref = _rh_planar([(EB, EB, 6e-9), (*ani, 6e-9), (EB, EB, 6e-9)], EMIR, q)
    assert abs(res.r_H - ref) < TOL
    assert res.T == 0.0
    others = res.absorption["lo"] + res.absorption["film"] + res.absorption["up"]
    assert abs(res.absorption["mirror"] - (1.0 - res.R - others)) < TOL
    assert res.absorption["film"] > 1e-3 and abs(res.absorption["lo"]) < 1e-12   # lossless oxide


def test_exact_quantum_film_element_matches_the_planar_film_route():
    """The exact film element (FilmLayer.exact) in the bottom stack: r and the film absorption equal
    QuantumFilmResponse.stack_rt (planar route; its r is the p-hat r_H) at 60 deg, the per-harmonic film data
    carry the whole film absorption in the m = 0 harmonic (the only one excited by a laterally uniform
    region), and film_spectrum normalizes |q_m H_m|^2 to that harmonic."""
    from dynameta.optics import nonlocal_tmm as nt
    resp = QuantumFilmResponse(QuantumFilm.flat_band(6e-9, N_BG, M_OPT, nz=97), EPS_INF, GAM)
    film = FilmLayer.exact(resp, "film")
    lower = [UniformLayer("lo", 6e-9, EB), film, UniformLayer("up", 6e-9, EB)]
    cell = _analytic_region(LAM, P_M, 2, theta_deg=60.0, struct_nm=0.0)
    res = cell.solve(lower, ESUB, absorption=False)
    pl = resp.stack_rt(OMEGA, [nt.DielectricLayer(EB, 6e-9)], [nt.DielectricLayer(EB, 6e-9)],
                       n_sub=1.45, theta_rad=math.radians(60.0))
    assert abs(res.r_H - pl.r) < TOL
    assert abs(res.absorption["film"] - pl.A) < TOL
    assert abs(res.T - pl.T) < TOL
    fd = res.film
    assert abs(np.sum(fd["A_m"]) - res.absorption["film"]) < 1e-15
    assert np.all(fd["A_m"][np.arange(fd["m"].size) != cell.i0] == 0.0)
    assert np.allclose(fd["q_m"], cell.q_nm * 1e9, rtol=0, atol=0)
    sp = film_spectrum(res)
    assert abs(sp["Dz2"][cell.i0] - 1.0) < 1e-15 and abs(np.sum(sp["Dz2"]) - 1.0) < 1e-15
    # at normal incidence q_0 = 0: no normal displacement anywhere, returned unnormalized (zeros)
    res0 = _analytic_region(LAM, P_M, 2, struct_nm=0.0).solve(lower, ESUB, absorption=False)
    assert np.all(film_spectrum(res0)["Dz2"] == 0.0)


def test_film_fallback_beyond_film_harmonics_and_q_symmetric_memo():
    """admittances(): harmonics |m| <= film_harmonics see the film layer, the others the fallback; at
    normal incidence Y(-m) = Y(m) (every layer even in q) and the memo serves the mirror harmonic;
    off normal the two differ."""
    resp = QuantumFilmResponse(QuantumFilm.flat_band(6e-9, N_BG, M_OPT, nz=97), EPS_INF, GAM)
    ex, eq = FilmLayer.exact(resp, "film"), FilmLayer.equivalent(resp, "film")
    lower = [UniformLayer("lo", 6e-9, EB), ex, UniformLayer("up", 6e-9, EB)]
    cell = _analytic_region(LAM, P_M, 3, struct_nm=0.0)
    Y, mats, starts = cell.admittances(lower, ESUB, film_harmonics=1, film_fallback=eq)
    for k, m in enumerate(cell.ms):
        use = lower if abs(m) <= 1 else [lower[0], eq, lower[2]]
        y_ref = stack_admittance(use, OMEGA, complex(cell.q_nm[k]) * 1e9, ESUB)[0] / 1e9
        assert abs(Y[k] - y_ref) < 1e-12 * abs(y_ref), m
    assert np.all(Y == Y[::-1])
    assert starts == [0] * cell.ms.size
    obl = _analytic_region(LAM, P_M, 2, theta_deg=20.0, struct_nm=0.0)
    Yo = obl.admittances(lower, ESUB)[0]
    assert abs(Yo[0] - Yo[-1]) > 1e-6 * abs(Yo[0])
    # the fallback really is different physics for the high harmonics (else the gate is vacuous)
    Yfull = cell.admittances(lower, ESUB)[0]
    assert np.max(np.abs(Yfull - Y)) > 1e-9 * np.max(np.abs(Y))


def test_film_name_selection_and_explicit_film():
    """solve(film=None) reports the FIRST FilmLayer's data; film='lo' reports that layer instead."""
    resp = QuantumFilmResponse(QuantumFilm.flat_band(6e-9, N_BG, M_OPT, nz=97), EPS_INF, GAM)
    lower = [UniformLayer("lo", 6e-9, EB + 0.02j), FilmLayer.equivalent(resp, "film"),
             UniformLayer("up", 6e-9, EB)]
    cell = _analytic_region(LAM, P_M, 1, theta_deg=30.0, struct_nm=0.0)
    a = cell.solve(lower, ESUB, absorption=False)
    b = cell.solve(lower, ESUB, film="lo", absorption=False)
    assert abs(np.sum(a.film["A_m"]) - a.absorption["film"]) < 1e-15
    assert abs(np.sum(b.film["A_m"]) - b.absorption["lo"]) < 1e-15
    assert abs(b.absorption["lo"]) > 1e-4


# ------------------------------------------------------------------------------------------------
# mesh-line planning
# ------------------------------------------------------------------------------------------------

def test_graded_points_keep_breaks_bound_the_spacing_and_grade_geometrically():
    """_graded_points: every break is a point, the spacing never exceeds h, and toward each grade
    point the distances h r^k (k = 1..levels) are present on both sides (inside the span)."""
    breaks, h, r, lv = [0.0, 40.0, 120.0, 400.0], 20.0, 0.3, 4
    pts = _graded_points(breaks, h, [40.0, 120.0, 0.0], lv, r)
    assert np.all(np.diff(pts) > 0)
    assert pts[0] == 0.0 and pts[-1] == 400.0
    for b in breaks:
        assert np.min(np.abs(pts - b)) < 1e-12
    assert np.max(np.diff(pts)) <= h + 1e-12
    for g in (40.0, 120.0):
        for k in range(1, lv + 1):
            for p in (g - h * r ** k, g + h * r ** k):
                assert np.min(np.abs(pts - p)) < 1e-9
    assert np.min(pts[pts > 0]) == pytest.approx(h * r ** lv, abs=1e-12)  # only the inner side of 0
    spec = CellMeshSpec()
    assert (spec.h_m, spec.corner_levels, spec.corner_ratio, spec.buffer_top_m, spec.min_rows) == \
        (None, 6, 0.3, 30e-9, 1)


def test_film_cell_result_spectrum_uses_both_faces():
    """film_spectrum averages |q H|^2 over the film's two faces before normalizing."""
    f = {"q_m": np.array([-2.0, 0.0, 3.0]), "H_top": np.array([1.0, 5.0, 1j]),
         "H_bottom": np.array([0.0, 1.0, 2.0]), "A_m": np.array([0.1, 0.2, 0.3])}
    res = FilmCellResult(r=0j, r_H=0j, R=0.0, T=0.0, absorption={}, energy_residual=0.0, film=f)
    dz = 0.5 * np.array([4.0 + 0.0, 0.0, 9.0 + 36.0])
    sp = film_spectrum(res)
    assert np.allclose(sp["Dz2"], dz / dz.sum(), rtol=1e-15, atol=0)
    assert np.array_equal(sp["A_m"], f["A_m"]) and np.array_equal(sp["q_m"], f["q_m"])
