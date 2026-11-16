"""A quantum film on finite oxide barriers no longer counts the penetration regions twice.

QuantumFilm.from_dual_gate builds the response element on the Schrodinger domain, which reaches into the oxides
when the gate stacks carry an OxideBarrier. If an optical output (transfer_matrix, effective_layer, stack_rt,
the cell solver's film layers, the pipeline coupling) described that EXTENDED element while the caller passes
the designed oxides, the penetration regions would be counted twice. Every output describes the
NOMINAL film (QuantumFilm class docstring): the element is folded back onto the nominal faces by removing the
background it puts in the oxide, and the equivalent layer has the nominal thickness, so every route takes the
designed stack and the total optical thickness is the designed one by construction.

Oracles:
  * impenetrable walls: an element extended 2 nm into a +1e8 eV "oxide" holds exactly the nominal film's
    electrons, so the designed stack must give the nominal film's r (MEASURED 2.8e-10 rpa / 1.5e-10
    equivalent); counting the extension on top of the designed oxides fails it;
  * depth independence on the dual-gate film: the depth the element reaches into the oxide (2 nm default vs
    the solver's whole 3.85 nm) must not change r once the tail is resolved (MEASURED <= 2.7e-8, against a
    bias modulation of 1.6e-3);
  * the pipeline coupling accepts the barrier film (its state check compares the carrier span, 6 nm, with the
    NOMINAL thickness, not the element's) and hands the optics the nominal-thickness equivalent layer.
"""
import math
import warnings

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, M_E, Q_E
from dynameta.optics.nonlocal_tmm import DielectricLayer
from dynameta.optics.quantum_film import QuantumFilm, QuantumFilmResponse

W1200 = 2.0 * math.pi * C_LIGHT / 1200e-9
KW = dict(n_super=1.0, n_sub=1.45, theta_rad=math.radians(60.0))
ABOVE = [DielectricLayer(2.00 ** 2 + 0j, 6e-9), DielectricLayer(1.70 ** 2 + 0j, 2e-9)]   # superstrate side first
BELOW = [DielectricLayer(1.70 ** 2 + 0j, 2e-9), DielectricLayer(2.00 ** 2 + 0j, 6e-9)]
EPS_INF, GAM, T_FILM, N_BG, M_FILM = 4.0, 1.2e14, 6e-9, 3e26, 0.30 * M_E


def test_extended_element_behind_impenetrable_walls_equals_the_nominal_film():
    from dynameta.carriers.schrodinger_poisson import sp_bulk_fermi_level_J
    t, n = T_FILM, 241
    h = t / (n - 1)
    m = M_FILM
    ef = sp_bulk_fermi_level_J(N_BG, m, 300.0)
    zn = np.linspace(0.0, t, n)
    un = -0.1 * Q_E * np.sin(np.pi * zn / t) ** 2                         # band bending (asymmetric-free)
    ne = int(round(2e-9 / h))
    ze = np.linspace(-ne * h, t + ne * h, n + 2 * ne)
    ue = np.interp(ze, zn, un)
    ue[(ze <= 1e-3 * h) | (ze >= t - 1e-3 * h)] = 1e8 * Q_E               # walls at (and beyond) the faces
    nominal = QuantumFilm(zn, un, ef, m)
    ext = QuantumFilm(ze, ue, ef, m, nominal_m=(0.0, t))
    # lengths of ~1e-9 m: abs=0, else pytest's default abs 1e-12 would be a 1e-3 relative tolerance
    assert ext.extension_m == pytest.approx((2e-9, 2e-9), rel=1e-9, abs=0)
    assert ext.nominal_thickness_m == pytest.approx(t, rel=1e-12, abs=0)
    assert abs(ext.sheet_density_total_m2 / nominal.sheet_density_total_m2 - 1.0) < 1e-7
    rn = QuantumFilmResponse(nominal, EPS_INF, GAM, route="real_space")
    rx = QuantumFilmResponse(ext, EPS_INF, GAM, route="real_space")
    for model, tol in (("rpa", 2e-9), ("equivalent", 1e-9)):
        a = rn.stack_rt(W1200, ABOVE, BELOW, model=model, **KW).r
        b = rx.stack_rt(W1200, ABOVE, BELOW, model=model, **KW).r
        assert abs(a - b) < tol, (model, abs(a - b))
    assert rx.effective_layer(W1200)[2] == pytest.approx(t, rel=1e-12, abs=0)
    # the same film seen from the other face (sign-sensitive orientation of the fold)
    a = rn.stack_rt(W1200, ABOVE, BELOW, superstrate_face="bottom", **KW).r
    b = rx.stack_rt(W1200, ABOVE, BELOW, superstrate_face="bottom", **KW).r
    assert abs(a - b) < 2e-9


@pytest.fixture(scope="module")
def barrier_solver():
    from dynameta.carriers.sp_dual_gate import DualGateSchrodingerPoisson, GateStack, OxideBarrier
    ox = ((2e-9, 9.0), (6e-9, 20.0))
    b = OxideBarrier(3.0, 0.4 * M_E)
    return DualGateSchrodingerPoisson(t_film_m=T_FILM, bottom=GateStack(ox, barrier=b), top=GateStack(ox, barrier=b),
                                      n_bg_m3=N_BG, m_eff_kg=M_FILM, eps_static=9.0, nz=241)


def test_depth_into_the_oxide_does_not_change_the_optics(barrier_solver):
    s = barrier_solver
    assert min(s.penetration_m) > 3e-9                                    # the solver's automatic domain
    r = {}
    for vt in (0.0, 2.0):
        sol = s.solve_voltages(0.0, vt)
        default = QuantumFilm.from_dual_gate(s, sol)
        assert default.extension_m == pytest.approx((2e-9, 2e-9), abs=1e-15)
        full = QuantumFilm.from_dual_gate(s, sol, optical_penetration_m=1.0)
        assert full.extension_m == pytest.approx(s.penetration_m, abs=1e-15)
        r[vt] = [QuantumFilmResponse(f, EPS_INF, GAM).stack_rt(W1200, ABOVE, BELOW, **KW).r for f in (default, full)]
        assert abs(r[vt][0] - r[vt][1]) < 1e-7, (vt, abs(r[vt][0] - r[vt][1]))
    assert abs(r[2.0][0] - r[0.0][0]) > 5e-4                               # the bias modulation it resolves


def test_a_cut_through_the_tail_is_flagged(barrier_solver):
    from dynameta.warning_types import UnreliableResultWarning
    sol = barrier_solver.solve_voltages(0.0, 2.0)
    with pytest.warns(UnreliableResultWarning, match="optical element is cut"):
        QuantumFilm.from_dual_gate(barrier_solver, sol, optical_penetration_m=0.5e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        QuantumFilm.from_dual_gate(barrier_solver, sol)                   # the default stays silent


def test_pipeline_coupling_takes_the_barrier_film_on_the_nominal_layer(barrier_solver):
    from dynameta.materials.optical_model import STACK_AXIS_FIELD
    from dynameta.optics.quantum_film_cell import FilmLayer
    from dynameta.optics.quantum_film_pipeline import QUANTUM_FILM_FIELD, QuantumFilmCoupling, QuantumFilmOptical
    from dynameta.sweep import BiasPoint
    c = QuantumFilmCoupling(barrier_solver)
    bp = BiasPoint({"bottom": 0.0, "top": 2.0}, "b")
    cf = c.solve(bp)
    reg = cf.regions[c.region_name]
    n = reg.grid_fields["electron_density_m3"]
    z = reg.grid_axes_m["z"]
    opt = QuantumFilmOptical(eps_inf=EPS_INF, gamma_rad_s=GAM)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tens = opt.eps_fields({QUANTUM_FILM_FIELD: c.state(bp), "n": n, STACK_AXIS_FIELD: z}, 1200e-9)
    resp = c.response(bp, opt)
    exx, ezz, L = resp.effective_layer(W1200)
    assert L == pytest.approx(T_FILM, rel=1e-12, abs=0)
    assert np.allclose(np.diag(tens), [exx, exx, ezz], rtol=1e-12, atol=0.0)
    assert FilmLayer.exact(resp).thickness_m == pytest.approx(T_FILM, rel=1e-12, abs=0)
    assert FilmLayer.equivalent(resp).thickness_m == pytest.approx(T_FILM, rel=1e-12, abs=0)
