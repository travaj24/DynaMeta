"""Gates for the v0.16 default z mesh grading of LayeredDevsimBuilder (Mesh2DSpec.semiconductor_refinement =
"screening": refinement to the screening length of each unipolar semiconductor layer; "legacy" restores the
v0.15.1 lines exactly).

WHY. The v0.15.1 lines (interface 0.05 nm, coarse 0.5 nm, zone 0.8 nm) leave 19 nodes across a 6 nm gated
degenerate film and 47 across a 20 nm one, with the SAME lines near each interface, so the accumulation and
depletion layers (screening length 0.78 nm) are under-resolved whatever the thickness: interface density
-1.9e-2 at +5 V and +1.1e-1 at -3 V against the converged answer, for 6 nm and 20 nm alike (the generic gated-film
cell of tests/_gated_film_fixture.py).

ORACLE. An independent 1-D solve written here: the nonlinear Poisson equation through the same oxide /
film / oxide stack (box method on a 1819-17499-node graded grid, numpy/scipy Newton), with the closure the
DEVSIM equilibrium region uses (n = N_c F_1/2((psi - Phi_c0)/V_t), Aymerich-Humet F_1/2 re-typed below,
n(0) = n_bg) and the gates as Dirichlet potentials on the outer oxide faces. Its own discretization error is
below 1e-5 (MEASURED: k = 1, 2, 4 refinements converge at second order, top-face density k = 1 / 2 vs 4:
-2.8e-6 / -5.6e-7; the k = 2 grid is used). Its bottom-face density agrees with every DEVSIM mesh to <= 7e-7, a
check that the two solve the same model.

ERROR BUDGET. Default mesh vs oracle, MEASURED (DEVSIM 2.11.2, Windows/scipy
OpenBLAS):
  6 nm : n(top) -9.57e-5 at +5 V, +3.72e-4 at -3 V; excess sheet +1.8e-5 / +9.7e-6 (204 nodes across the film)
  20 nm: n(top) -9.17e-5 at +5 V, +3.48e-4 at -3 V; excess sheet +1.7e-5 / +9.1e-6 (298 nodes; 19 / 47 legacy)
Halving the spacing factors divides the error by ~4 (second order). Gates: 3e-4 (+5 V) and 2e-3 (-3 V), 3x / 5x
the measured error (1e-3 / 1e-2 accuracy targets are met with room). v0.15.1 default (= "legacy"):
-1.88e-2 / +1.12e-1 (6 nm), -1.88e-2 / +1.12e-1 (20 nm) -- every n(top) gate below FAILS on v0.15.1.

A DEVSIM PROPERTY FOUND HERE: DEVSIM's own line grading drops the generated points within ~1e-10 mesh units
before a mesh line (0.1 nm in this builder's metres), so a film's upper face kept a ~0.1 nm cell next to its
face cell under ANY requested spacing (fine legacy-rule meshes then converge to a fixed offset below the oracle
at strong accumulation). The screening grading therefore places every node of a refined zone as an explicit mesh
line.
"""
import numpy as np
import pytest

import _gated_film_fixture as G

devsim = pytest.importorskip("devsim")

from dynameta.carriers import devsim_layered as DL                    # noqa: E402
from dynameta.core.numerics import trapz                               # noqa: E402
from dynameta.geometry.specs import Mesh2DSpec                         # noqa: E402
from dynameta.sweep import BiasPoint                                   # noqa: E402

EPS0, Q, KB, HBAR, M_E, T = 8.8541878128e-12, 1.602176634e-19, 1.380649e-23, 1.054571817e-34, \
    9.1093837015e-31, 300.0
VT = KB * T / Q
L_S_FILM = 7.76688887527296e-10           # screening length of the fixture film (gate A below)


# ---------------------------------------------------------------------------------------------------
# the independent 1-D oracle
# ---------------------------------------------------------------------------------------------------

def _F12(eta):
    eta = np.asarray(eta, float)
    a = eta ** 4 + 50.0 + 33.6 * eta * (1.0 - 0.68 * np.exp(-0.17 * (eta + 1.0) ** 2))
    return 1.0 / ((3.0 * np.sqrt(np.pi) / 4.0) / a ** 0.375 + np.exp(-eta))


def _inv_F12(x):
    lo, hi = -20.0, 80.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _F12(mid) < x else (lo, mid)
    return 0.5 * (lo + hi)


def _graded(a, b, h0, ratio, hmax):
    L = b - a
    s, h = [0.0], h0
    while s[-1] < L / 2:
        s.append(s[-1] + h)
        h = min(h * ratio, hmax)
    s = np.array(s)
    s = s[s < L / 2]
    return a + np.unique(np.r_[s, L / 2, L - s[::-1]])


def oracle_1d(t_film, v_top, v_bot=0.0, k=2):
    """(z from the film's lower face, n) of the film: 1-D box-method nonlinear Poisson, Dirichlet gates on
    the outer oxide faces (oxA 2 nm eps 9 / oxB 6 nm eps 20 on both sides, as in the fixture)."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    layers = [(2e-9, 9.0, False), (6e-9, 20.0, False), (t_film, 9.0, True), (6e-9, 20.0, False),
              (2e-9, 9.0, False)]
    Nc = 2.0 * (0.30 * M_E * KB * T / (2.0 * np.pi * HBAR ** 2)) ** 1.5
    phi_c0 = -_inv_F12(G.N_BG / Nc) * VT
    z, eps_c, semi_c, z0 = [np.array([0.0])], [], [], 0.0
    for t, e, s in layers:
        pts = (_graded(z0, z0 + t, 1e-13 / k, 1 + 0.02 / k, 5e-12 / k) if s
               else np.linspace(z0, z0 + t, max(2, int(np.ceil(t / 5e-11))) + 1))
        z.append(pts[1:])
        eps_c += [e * EPS0] * (len(pts) - 1)
        semi_c += [s] * (len(pts) - 1)
        z0 += t
    z = np.concatenate(z)
    h = np.diff(z)
    semi_c = np.array(semi_c, float)
    g = np.array(eps_c) / h
    vol = np.zeros(z.size)
    vol[:-1] += 0.5 * h * semi_c
    vol[1:] += 0.5 * h * semi_c
    psi = np.linspace(v_bot, v_top, z.size)
    for _ in range(300):
        eta = (psi - phi_c0) / VT
        n = Nc * _F12(eta)
        dn = Nc * (_F12(eta + 1e-6) - _F12(eta - 1e-6)) / 2e-6 / VT
        r = np.zeros(z.size)
        fl = g * (psi[1:] - psi[:-1])
        r[:-1] += fl
        r[1:] -= fl
        r -= Q * vol * (n - G.N_BG)
        main = -np.r_[g, 0.0] - np.r_[0.0, g] - Q * vol * dn
        J = diags([main, g, g], [0, 1, -1], format="lil")
        J[0, :] = 0
        J[0, 0] = 1.0
        J[-1, :] = 0
        J[-1, -1] = 1.0
        r[0], r[-1] = psi[0] - v_bot, psi[-1] - v_top
        d = spsolve(J.tocsr(), -r)
        step = np.max(np.abs(d))
        psi += d * min(1.0, 0.1 / step) if step > 0 else d
        if step < 1e-13:
            break
    else:                                                              # pragma: no cover
        raise RuntimeError("oracle Newton did not converge")
    i = np.where(vol > 0)[0]
    return z[i] - z[i[0]], Nc * _F12((psi[i] - phi_c0) / VT)


# ---------------------------------------------------------------------------------------------------
# A: the screening length
# ---------------------------------------------------------------------------------------------------

def test_screening_length_against_exact_fermi_integrals():
    """ORACLE: L_s^2 = eps eps0 kT / (q^2 N_c F_{-1/2}(eta)) with F_{1/2}, F_{-1/2} the EXACT normalized
    Fermi-Dirac integrals (scipy quadrature here) and eta from F_{1/2}(eta) = n_bg / N_c. The library uses
    the derivative of the Aymerich-Humet fit its closure solves (fit error ~1e-3 at eta ~ 21): 3e-3 bar.
    Limits: Debye length when non-degenerate (1e-4; 1e-12 below the non-degenerate cut and for Boltzmann
    statistics), and a bipolar acceptor material uses its hole band."""
    from types import SimpleNamespace
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from dynameta.materials import TransportModel

    def fd(j, eta):
        f = (lambda e: e ** j / (1.0 + np.exp(min(e - eta, 700.0))))
        val = quad(f, 0.0, max(eta, 0.0) + 60.0, limit=400, points=[max(eta, 0.0)] if eta > 0 else None)[0]
        from math import gamma
        return val / gamma(j + 1.0)

    def Nc(m):
        return 2.0 * (m * KB * T / (2.0 * np.pi * HBAR ** 2)) ** 1.5

    tr = G.materials().get("TCO").transport
    L = DL.screening_length_m(tr)
    N = Nc(0.30 * M_E)
    eta = brentq(lambda e: fd(0.5, e) - G.N_BG / N, -5.0, 60.0, xtol=1e-12)
    L_exact = np.sqrt(9.0 * EPS0 * VT / (Q * N * fd(-0.5, eta)))
    assert abs(L / L_exact - 1.0) < 3e-3
    assert L == pytest.approx(L_S_FILM, rel=1e-12, abs=0)          # recorded value (used below)
    # non-degenerate: the Debye length
    for n in (1e21, 1e18):
        t2 = TransportModel(n_bg_m3=n, eps_static=11.7, dos_mass_kg_of_n_m3=lambda x: 0.26 * M_E)
        LD = np.sqrt(11.7 * EPS0 * VT / (Q * n))
        assert abs(DL.screening_length_m(t2) / LD - 1.0) < (1e-4 if n > 1e19 else 1e-12)
    # bipolar, acceptor, Boltzmann statistics: exactly Debye (and the electron mass is not read)
    bp = SimpleNamespace(physics="bipolar_dd", acceptor=True, net_doping_expr=None, dos_mass_p_kg=0.8 * M_E,
                         carrier_statistics="boltzmann", n_bg_m3=1e24, eps_static=12.9,
                         dos_mass_kg_of_n_m3=lambda x: float("nan"))
    assert DL.screening_length_m(bp) == pytest.approx(np.sqrt(12.9 * EPS0 * VT / (Q * 1e24)), rel=1e-12, abs=0)


# ---------------------------------------------------------------------------------------------------
# B: the default mesh against the oracle
# ---------------------------------------------------------------------------------------------------

def _devsim_top(t_film, vt, **mesh):
    b = DL.LayeredDevsimBuilder(G.film_design(t_film, **mesh))
    try:
        cf = b.solve(BiasPoint({"top": vt, "bot": 0.0}, "b"), grid_n_x=8)
        z, n = G.column(b.device, 20e-9)
    finally:
        b.teardown()
    return z - z[0], n, cf.extras["interface_resolution"]["film"]


def _film_lines(t_film, **mesh):
    """(film z interval, the builder's z mesh-line positions inside the film relative to its lower face, the
    refined-zone depth zs of the grading rule) for the fixture film with Mesh2DSpec(**mesh)."""
    d = G.film_design(t_film, **mesh)
    ys, _ = DL.LayeredDevsimBuilder(d)._y_mesh_lines(d.mesh_2d)
    zlo, zhi = d.z_intervals()["film"]
    pos = np.array([c[0] for c in ys])
    pos = pos[(pos > zlo - 1e-15) & (pos < zhi + 1e-15)] - zlo
    zs = min(d.mesh_2d.screening_zone_factor * L_S_FILM, 0.45 * t_film)
    return zhi - zlo, pos, zs


def _assert_nodes_follow_the_rule(z, t_film, legacy):
    """The DEVSIM node column z (from the film's lower face) against the builder's mesh lines: every line is a
    node (to 1e-17 m, round-off of positions ~1e-7 m). Default rule: inside each refined zone (depth zs from a
    face) the nodes ARE the explicit lines, no more (DEVSIM inserted nothing, dropped nothing), so the zone node
    count is the grading rule's; legacy rule: no gap finer than interface_ps_m = 0.05 nm (it cannot resolve a
    0.78 nm screening layer, whose default face gap is L_s / 40 = 0.019 nm)."""
    t, pos, zs = _film_lines(t_film, **({"semiconductor_refinement": "legacy"} if legacy else {}))
    assert abs(z[-1] - t) < 1e-17
    assert all(np.min(np.abs(z - p)) < 1e-17 for p in pos), "a mesh line is not a node"
    if legacy:
        assert np.min(np.diff(z)) >= Mesh2DSpec().interface_ps_m * (1.0 - 1e-6)
        return
    lower, upper = (lambda v: v <= zs + 1e-17), (lambda v: v >= t - zs - 1e-17)
    for in_zone in (lower, upper):
        assert np.count_nonzero(in_zone(z)) == np.count_nonzero(in_zone(pos)), "DEVSIM changed the zone lines"
    assert np.min(np.diff(z)) <= L_S_FILM / 40 * (1.0 + 1e-2)


@pytest.mark.parametrize("t_film", [6e-9, 20e-9])
def test_default_grading_matches_independent_1d_solve(t_film):
    """Gate (see module docstring): default Mesh2DSpec, interface density within 3e-4 of the oracle at
    +5 V and 2e-3 at -3 V, excess sheet density within 1e-4; the legacy lines miss by -1.9e-2 / +1.1e-1
    (the v0.15.1 default -- the gate fails there): asserted as |err_legacy| > 10 x the gate's tolerance (3e-3 /
    2e-2), i.e. the legacy mesh is an order of magnitude outside what the default meets, which holds with room
    for any legitimate change of the legacy path (MEASURED 6.3x / 5.6x above that bar). The node columns follow
    the mesh rule (_assert_nodes_follow_the_rule). extras['interface_resolution'] (max |ln(n_k+1/n_k)| along z)
    drops with the refinement."""
    for vt, tol in ((5.0, 3e-4), (-3.0, 2e-3)):
        zo, no = oracle_1d(t_film, vt)
        z, n, ires = _devsim_top(t_film, vt)
        zl, nl, ires_l = _devsim_top(t_film, vt, semiconductor_refinement="legacy")
        _assert_nodes_follow_the_rule(z, t_film, legacy=False)
        _assert_nodes_follow_the_rule(zl, t_film, legacy=True)
        err = n[-1] / no[-1] - 1.0
        err_l = nl[-1] / no[-1] - 1.0
        assert abs(err) < tol, (vt, err)
        assert abs(err_l) > 10.0 * tol, (vt, err_l)                   # v0.15.1: -1.88e-2 / +1.12e-1
        assert abs(n[0] / no[0] - 1.0) < 1e-5                         # lower face (same model)
        sheet, sheet_o = trapz(n - G.N_BG, z), trapz(no - G.N_BG, zo)
        assert abs(sheet / sheet_o - 1.0) < 1e-4
        assert 2.0 * ires < ires_l, (ires, ires_l)          # MEASURED 0.022 / 0.150 vs 0.350 / 0.707


def test_screening_grading_converges_at_second_order():
    """Halving the face and zone spacing factors divides the +5 V interface error (vs the oracle) by ~4
    (MEASURED 6 nm: -9.57e-5 -> -2.34e-5): the refinement is a convergent discretization, not a tuned point."""
    zo, no = oracle_1d(6e-9, 5.0)
    e1 = _devsim_top(6e-9, 5.0)[1][-1] / no[-1] - 1.0
    e2 = _devsim_top(6e-9, 5.0, screening_ps_factor=1 / 40, screening_interface_factor=1 / 80)[1][-1] / no[-1] - 1.0
    assert abs(e2) < 0.4 * abs(e1)


# ---------------------------------------------------------------------------------------------------
# C: "legacy" = the v0.15.1 mesh lines, exactly
# ---------------------------------------------------------------------------------------------------

# add_2d_mesh_line calls (dir, pos, ns, ps) of the v0.15.1 builder (origin/main ebdfe5c, captured by loading
# that commit's devsim_layered.py) on the fixture with a 6 nm and a 20 nm film
_V0151_Y = {
    6e-9: [(0.0, 5e-11, 5e-11), (8e-10, 5e-10, 5e-10), (7.920000000000001e-08, 5e-10, 5e-10), (4e-08, 5e-10, 5e-10),
           (8e-08, 5e-11, 5e-11), (8.08e-08, 5e-10, 5e-10), (8.120000000000001e-08, 5e-10, 5e-10),
           (8.2e-08, 5e-11, 5e-11), (8.28e-08, 5e-10, 5e-10), (8.720000000000001e-08, 5e-10, 5e-10),
           (8.500000000000001e-08, 5e-10, 5e-10), (8.800000000000001e-08, 5e-11, 5e-11), (8.88e-08, 5e-10, 5e-10),
           (9.320000000000001e-08, 5e-10, 5e-10), (9.100000000000001e-08, 5e-10, 5e-10),
           (9.400000000000001e-08, 5e-11, 5e-11), (9.48e-08, 5e-10, 5e-10), (9.920000000000001e-08, 5e-10, 5e-10),
           (9.700000000000001e-08, 5e-10, 5e-10), (1.0000000000000001e-07, 5e-11, 5e-11), (1.008e-07, 5e-10, 5e-10),
           (1.0120000000000002e-07, 5e-10, 5e-10), (1.0200000000000001e-07, 5e-11, 5e-11),
           (1.0280000000000001e-07, 5e-10, 5e-10), (1.4120000000000002e-07, 5e-10, 5e-10), (1.22e-07, 5e-10, 5e-10),
           (1.4200000000000003e-07, 5e-11, 5e-11)],
    20e-9: [(0.0, 5e-11, 5e-11), (8e-10, 5e-10, 5e-10), (7.920000000000001e-08, 5e-10, 5e-10), (4e-08, 5e-10, 5e-10),
            (8e-08, 5e-11, 5e-11), (8.08e-08, 5e-10, 5e-10), (8.120000000000001e-08, 5e-10, 5e-10),
            (8.2e-08, 5e-11, 5e-11), (8.28e-08, 5e-10, 5e-10), (8.720000000000001e-08, 5e-10, 5e-10),
            (8.500000000000001e-08, 5e-10, 5e-10), (8.800000000000001e-08, 5e-11, 5e-11), (8.88e-08, 5e-10, 5e-10),
            (1.0720000000000002e-07, 5e-10, 5e-10), (9.8e-08, 5e-10, 5e-10), (1.0800000000000001e-07, 5e-11, 5e-11),
            (1.0880000000000001e-07, 5e-10, 5e-10), (1.1320000000000002e-07, 5e-10, 5e-10),
            (1.1100000000000001e-07, 5e-10, 5e-10), (1.1400000000000001e-07, 5e-11, 5e-11),
            (1.1480000000000001e-07, 5e-10, 5e-10), (1.1520000000000003e-07, 5e-10, 5e-10),
            (1.1600000000000002e-07, 5e-11, 5e-11), (1.1680000000000001e-07, 5e-10, 5e-10),
            (1.5520000000000001e-07, 5e-10, 5e-10), (1.3600000000000003e-07, 5e-10, 5e-10),
            (1.5600000000000002e-07, 5e-11, 5e-11)],
}
_V0151_X = [(0.0, 4e-09, 4e-09), (2e-08, 4e-09, 4e-09), (4e-08, 4e-09, 4e-09)]


def _mesh_calls(monkeypatch, design):
    calls = []
    real = DL.ds.add_2d_mesh_line

    def spy(**kw):
        calls.append((kw["dir"], kw["pos"], kw["ns"], kw["ps"]))
        return real(**kw)

    monkeypatch.setattr(DL.ds, "add_2d_mesh_line", spy)
    b = DL.LayeredDevsimBuilder(design)
    try:
        b.build_device()
    finally:
        b.teardown()
        monkeypatch.undo()
    return calls


@pytest.mark.parametrize("t_film", [6e-9, 20e-9])
def test_legacy_reproduces_the_v0151_mesh_line_calls(monkeypatch, t_film):
    """Same-tree A/B for the restore keyword: semiconductor_refinement="legacy" issues exactly the v0.15.1
    add_2d_mesh_line calls, in the same order and to the last bit; the default issues different ones."""
    legacy = _mesh_calls(monkeypatch, G.film_design(t_film, semiconductor_refinement="legacy"))
    assert legacy == [("x",) + c for c in _V0151_X] + [("y",) + c for c in _V0151_Y[t_film]]
    default = _mesh_calls(monkeypatch, G.film_design(t_film))
    assert [c for c in default if c[0] == "x"] == [("x",) + c for c in _V0151_X]
    assert [c for c in default if c[0] == "y"] != [("y",) + c for c in _V0151_Y[t_film]]


# ---------------------------------------------------------------------------------------------------
# D: structure of the screening lines, and the layers it leaves alone (pure: no DEVSIM build)
# ---------------------------------------------------------------------------------------------------

def test_screening_lines_structure_on_the_film():
    """The grading rule (screening_mesh._graded_offsets(h0, h1, zs), h0 = L_s / 40 at the face, h1 = L_s / 20 at the
    zone edge zs = min(4 L_s, 0.45 t)): steps h0 r^j with the constant ratio r = (zs - h0) / (zs - h1) (to 1e-9,
    round-off of the positions), the count k rounded to an integer and the whole sequence rescaled by f to land on
    zs exactly. Rounding k moves the sum by at most half a step at the far end, which bounds the rescale:
    |f - 1| <= h1 / (2 (zs - h1)) (0.73 % on the 6 nm film, 0.63 % on the 20 nm one; MEASURED f - 1 = +0.50 % /
    -0.17 %), so the face gap is h0 (1 +- that) and the largest gap h1 (1 + that). Each line's ns/ps is above its
    gaps (DEVSIM inserts nothing between them); every legacy line of the dielectric and metal layers is kept with its
    legacy spacing (face lines finer)."""
    for t_film in (6e-9, 20e-9):
        d = G.film_design(t_film)
        b = DL.LayeredDevsimBuilder(d)
        ys, _ = b._y_mesh_lines(d.mesh_2d)
        yl, _ = b._y_mesh_lines(Mesh2DSpec(**{**d.mesh_2d.__dict__, "semiconductor_refinement": "legacy"}))
        pos = np.array([c[0] for c in ys])
        assert np.all(np.diff(pos) > 0)                                 # sorted, no duplicates
        zlo, zhi = d.z_intervals()["film"]
        m = d.mesh_2d
        h0, h1 = m.screening_interface_factor * L_S_FILM, m.screening_ps_factor * L_S_FILM
        zs = min(m.screening_zone_factor * L_S_FILM, 0.45 * t_film)
        r = (zs - h0) / (zs - h1)
        bound = h1 / (2.0 * (zs - h1))
        for face, sgn in ((zlo, 1.0), (zhi, -1.0)):
            inner = np.sort(np.abs(pos[(sgn * (pos - face) >= -1e-18) & (sgn * (pos - face) <= zs + 1e-18)] - face))
            gaps = np.diff(inner)
            assert abs(gaps[0] / h0 - 1.0) <= bound, (t_film, gaps[0] / h0 - 1.0, bound)
            assert inner[-1] == pytest.approx(zs, rel=1e-12, abs=0)
            assert np.max(np.abs(gaps[1:] / gaps[:-1] / r - 1.0)) < 1e-9
            assert gaps.max() <= h1 * (1.0 + bound)
        by = {round(c[0] / 1e-15): c for c in ys}
        for p, ns, ps in yl:
            if zlo - 1e-15 < p < zhi + 1e-15:
                continue                                                # inside the film: replaced
            c = by[round(p / 1e-15)]
            assert (c[1], c[2]) == (ns, ps)                             # outside the film: legacy lines
        for face in (zlo, zhi):                                         # faces: finer than legacy
            c = by[round(face / 1e-15)]
            assert min(c[1], c[2]) <= L_S_FILM / 40 * (1 + 1e-9)


def test_layers_needing_no_refinement_keep_the_legacy_lines():
    """A lightly doped unipolar semiconductor (L_s ~ 40 nm, so the default spacings already resolve it) and a
    bipolar_dd stack (n_bg is a doping; the Debye length sets neither an inversion nor a junction depletion
    width) get the legacy lines exactly."""
    from dynameta.geometry.specs import Mesh2DSpec as M2
    from validation.devsim_heterojunction import ALGAAS, GAAS, stack_design, transport
    d = G.film_design(6e-9, n_bg=1e22)
    b = DL.LayeredDevsimBuilder(d)
    assert b.layer_screening_lengths()["film"] > 20e-9
    leg = M2(**{**d.mesh_2d.__dict__, "semiconductor_refinement": "legacy"})
    assert b._y_mesh_lines(d.mesh_2d)[0] == b._y_mesh_lines(leg)[0]
    bip = stack_design([("pgaas", "GaAs_p", 400e-9, transport(GAAS, n_bg=1e23, physics="bipolar_dd", acceptor=True)),
                        ("nalgaas", "AlGaAs_n", 400e-9, transport(ALGAAS, n_bg=1e23, physics="bipolar_dd"))])
    bb = DL.LayeredDevsimBuilder(bip)
    assert bb.layer_screening_lengths() == {}
    leg = M2(**{**bip.mesh_2d.__dict__, "semiconductor_refinement": "legacy"})
    assert bb._y_mesh_lines(bip.mesh_2d)[0] == bb._y_mesh_lines(leg)[0]


def test_mesh2dspec_validates_the_new_fields():
    with pytest.raises(ValueError, match="semiconductor_refinement"):
        Mesh2DSpec(semiconductor_refinement="fine")
    for name in ("screening_zone_factor", "screening_ps_factor", "screening_interface_factor"):
        with pytest.raises(ValueError, match=name):
            Mesh2DSpec(**{name: 0.0})
    m = Mesh2DSpec()
    assert (m.semiconductor_refinement, m.screening_zone_factor, m.screening_ps_factor,
            m.screening_interface_factor) == ("screening", 4.0, 1 / 20, 1 / 40)


def test_screening_length_has_one_implementation_and_needs_no_devsim():
    """The screening length and the grading helpers live in the devsim-free carriers.screening_mesh (the fast
    legs test them); devsim_layered re-exports the SAME objects, so there is one implementation."""
    from dynameta.carriers import screening_mesh as SM
    for name in ("screening_length_m", "_graded_offsets", "_needs_screening", "_interface_resolution",
                 "_NONDEGENERATE_X"):
        assert getattr(DL, name) is getattr(SM, name), name
    with open(SM.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert "import devsim" not in src and "import_devsim" not in src


def test_patterned_gate_footprint_edges_are_mesh_lines():
    """A patterned (Rectangle) gate on a FULL metal layer: its contact captures the nodes inside the footprint,
    so the footprint edges must be x mesh lines. Before the fix the x lines came from inclusion edges only, and
    a 16 nm gate (x 11 .. 27 nm) over the fixture's 4 nm x mesh captured x 12 .. 24 nm (12 nm, off-centre).
    The full-footprint electrodes keep the v0.15.1 x lines (test_legacy_reproduces_the_v0151_mesh_line_calls)."""
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.geometry import Electrode
    from dynameta.geometry.cross_section import Rectangle
    ds = import_devsim()
    d = G.film_design(semiconductor_refinement="legacy")
    d.electrodes = [Electrode("bot", "mirror", "full"),
                    Electrode("top", "gate", Rectangle(cx_m=19e-9, cy_m=20e-9, width_m=16e-9, height_m=40e-9))]
    b = DL.LayeredDevsimBuilder(d, device_name="fpl", mesh_name="fpl_m")
    try:
        b.build_device()
        reg = b._contact_region["top"]
        X = np.array(ds.get_node_model_values(device="fpl", region=reg, name="x"))
        idx = sorted({int(i) for e in ds.get_element_node_list(device="fpl", region=reg, contact="top") for i in e})
        assert X[idx].min() == pytest.approx(11e-9, abs=1e-15) and X[idx].max() == pytest.approx(27e-9, abs=1e-15)
    finally:
        b.teardown()
