"""Gates for the pipeline wiring of the quantum-film response (optics/quantum_film_pipeline.py, v0.16).

Fixture (the generic gated-film cell): a 6 nm film, 3e26 m^-3, m* 0.30 m_e, eps_static 9.0, gated through
2 nm (static eps 9) + 6 nm (static eps 20) oxide on both faces (carriers.DualGateSchrodingerPoisson, 201
nodes for speed), in the planar stack air | 2 nm oxA (n 1.70) | 6 nm oxB (n 2.00) | film | 6 nm oxB | 2 nm
oxA | 80 nm mirror metal (eps -120+20i) | glass (n 1.45), solved by the Berreman bridge (a planar route
that takes a tensor eps). Response: eps_inf 4.0, damping 1.2e14 rad/s.

On v0.15.1 every gate fails at `import dynameta.optics.quantum_film_pipeline` (ModuleNotFoundError); the route table's PMM row additionally relies on nothing new (a UNIFORM tensor
layer was already accepted there).

Oracles:
  * by hand: the documented few lines (solve_voltages -> QuantumFilm.from_dual_gate ->
    QuantumFilmResponse -> effective_layer -> EpsField(tensor) -> the same optical solver) must give the
    pipeline's numbers BIT FOR BIT -- the wiring adds nothing;
  * the exact planar element (QuantumFilmResponse.stack_rt): the pipeline's equivalent layer must sit
    within its first-order error (MEASURED 3.4e-7 .. 3.4e-5 in r);
  * the classical path (a DrudeOptical film through the same entry point) must equal run_pipeline
    without the extra field (the quantum wiring is invisible to every other material).
"""
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("lumenairy")

from dynameta.carriers import DualGateSchrodingerPoisson, GateStack                  # noqa: E402
from dynameta.constants import C_LIGHT, M_E                                          # noqa: E402
from dynameta.core.eps_field import EpsField                                         # noqa: E402
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell              # noqa: E402
from dynameta.geometry.cross_section import Rectangle                                # noqa: E402
from dynameta.geometry.specs import OpticalSpec                                      # noqa: E402
from dynameta.materials import ConstantOptical, DrudeOptical, Material, MaterialRegistry  # noqa: E402
from dynameta.optics.quantum_film import QuantumFilm, QuantumFilmResponse            # noqa: E402
from dynameta.optics.quantum_film_pipeline import (                                  # noqa: E402
    QUANTUM_FILM_FIELD, QuantumFilmCoupling, QuantumFilmOptical, QuantumFilmState,
    film_layer_optics, run_quantum_film_pipeline)
from dynameta.sweep import BiasPoint, Sweep                                          # noqa: E402

P = 400e-9
NZ = 201
EPS_INF, GAM = 4.0, 1.2e14


def _opt(**kw):
    """The fixture's response options with `kw` changed."""
    args = dict(eps_inf=EPS_INF, gamma_rad_s=GAM)
    args.update(kw)
    return QuantumFilmOptical(**args)


OPT = _opt()


def _registry(film_optical):
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0), ("glass", 1.45 ** 2), ("oxA", 1.70 ** 2), ("oxB", 2.00 ** 2),
                  ("metal", -60 + 4j), ("mirror_metal", -120 + 20j)):
        reg.add(Material(nm, ConstantOptical(complex(e))))
    reg.add(Material("FILM", film_optical))
    return reg


def _design(film_optical=OPT, strips=False, pol="p", theta=60.0, film_t=6e-9):
    layers = [Layer(n, t, m) for n, t, m in (("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"),
                                             ("lo_oxb", 6e-9, "oxB"), ("film", film_t, "FILM"),
                                             ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA"))]
    if strips:
        layers.append(Layer("grating", 40e-9, "air",
                            inclusions=[Inclusion(Rectangle(P / 2, P / 2, 150e-9, P), "metal")]))
    return Design(name="qf", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=_registry(film_optical),
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _solver(**kw):
    ox = GateStack(((2e-9, 9.0), (6e-9, 20.0)))
    args = dict(t_film_m=6e-9, bottom=ox, top=ox, n_bg_m3=3e26, m_eff_kg=0.30 * M_E, eps_static=9.0,
                nz=NZ, semi_material="FILM")
    args.update(kw)
    return DualGateSchrodingerPoisson(**args)


def _berreman():
    from dynameta.optics.lumenairy_bridge import make_lumenairy_berreman_solver
    return make_lumenairy_berreman_solver()


BIASES = [BiasPoint({"bottom": 0.0, "top": 0.0}, "fb"), BiasPoint({"bottom": 2.0, "top": 2.0}, "acc"),
          BiasPoint({"bottom": 0.0, "top": -3.0}, "dep1")]
LAMS = [1150.0, 1250.0]


@pytest.fixture(scope="module")
def pipeline_run():
    s = _solver()
    design = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows, coupling = run_quantum_film_pipeline(design, Sweep(bias_points=BIASES, wavelengths_nm=LAMS),
                                                   s, film_layer="film", optical_solver=_berreman(),
                                                   verbose=False)
    return s, design, rows, coupling


def test_pipeline_equals_the_documented_hand_recipe_bit_for_bit(pipeline_run):
    """Per bias and wavelength the pipeline row equals the by-hand recipe exactly (same solver, same
    layer): the wiring adds no arithmetic of its own."""
    _s, design, rows, _c = pipeline_run
    s2 = _solver()                                     # a fresh solver: no shared cache
    solve = _berreman()
    from dynameta.optics.tmm_reference import end_media_indices
    for row in rows:
        bp = next(b for b in BIASES if b.label == row.bias_label)
        sol = s2.solve_voltages(bp.voltages["bottom"], bp.voltages["top"])
        resp = QuantumFilmResponse(QuantumFilm.from_dual_gate(s2, sol), EPS_INF, GAM)
        lam = row.lambda_nm * 1e-9
        exx, ezz, _L = resp.effective_layer(2.0 * math.pi * C_LIGHT / lam)
        ebr = {"film": EpsField(tensor=np.diag([exx, exx, ezz]))}
        ns, nb = end_media_indices(design, lam)
        res = solve(design, None, ebr, lam, ns, nb)
        assert res.r == row.result.r and res.R == row.result.R


def test_bias_sweep_reuses_the_carrier_cache_and_builds_each_response_once(pipeline_run):
    """3 biases x 2 wavelengths: 3 dual-gate solves (the coupling's state() is served by the solver's
    own cache), 3 ground states, ONE response per bias (the collective modes), 2 layer evaluations
    per bias (only the frequency-dependent part is recomputed)."""
    s, _d, rows, c = pipeline_run
    assert len(rows) == 6
    assert s.stats["biases_solved"] == 3
    assert s.stats["cache_hits"] >= 3                  # the state build reused each converged bias
    assert c.stats["states_built"] == 3
    for bp in BIASES:
        st = c.state(bp)
        assert st.stats["responses_built"] == 1
        assert st.stats["layer_evaluations"] == 2
    # a repeat of the same sweep builds nothing new
    built = c.stats["states_built"]
    for bp in BIASES:
        for lam in LAMS:
            c.effective_layer(bp, OPT, lam * 1e-9)
    assert c.stats["states_built"] == built
    assert all(c.state(bp).stats["layer_evaluations"] == 2 for bp in BIASES)


def test_exact_response_is_reachable_and_the_layer_is_first_order_accurate(pipeline_run):
    """coupling.response(bias, options) is the exact response object; in this planar stack the
    pipeline's equivalent layer differs from the exact film element by its first-order error
    (MEASURED 3.4e-7 .. 3.4e-5 in r over the 3 biases x 2 wavelengths; budget 1e-4). Sign-sensitive: the
    equivalent layer with its two components swapped misses by > 1e-3 (MEASURED >= 1.7e-3).
    """
    from dynameta.optics import nonlocal_tmm as nt
    from dynameta.optics.quantum_film import film_stack_rt, uniaxial_layer_matrix
    _s, _d, _rows, c = pipeline_run
    above = [nt.DielectricLayer(1.70 ** 2, 2e-9), nt.DielectricLayer(2.00 ** 2, 6e-9)]
    below = [nt.DielectricLayer(2.00 ** 2, 6e-9), nt.DielectricLayer(1.70 ** 2, 2e-9),
             nt.DielectricLayer(-120 + 20j, 80e-9)]
    kw = dict(n_super=1.0, n_sub=1.45, theta_rad=math.radians(60.0))
    for bp in BIASES:
        resp = c.response(bp, OPT)
        assert isinstance(resp, QuantumFilmResponse)
        for lam in LAMS:
            w = 2.0 * math.pi * C_LIGHT / (lam * 1e-9)
            exx, ezz, L = c.effective_layer(bp, OPT, lam * 1e-9)
            ex = resp.stack_rt(w, above, below, **kw).r
            eq = film_stack_rt(w, lambda ww, kp: uniaxial_layer_matrix(exx, ezz, L, ww / C_LIGHT, kp),
                               above, below, **kw).r
            sw = film_stack_rt(w, lambda ww, kp: uniaxial_layer_matrix(ezz, exx, L, ww / C_LIGHT, kp),
                               above, below, **kw).r
            assert abs(ex - eq) < 1e-4
            assert abs(ex - sw) > 1e-3


def test_cache_fingerprints_carry_every_response_option():
    """The optical-solver cache keys a material it cannot evaluate by its repr: every option change
    (eps_inf, both damping rates, xc, eps_xc, route) gives a new key, equal options the same key; and
    the state keeps one response per option set."""
    from dynameta.cache import _materials_fingerprint
    variants = [OPT, _opt(eps_inf=4.2), _opt(gamma_rad_s=2e14),
                _opt(gamma_inplane_rad_s=5e13), _opt(xc="x"),
                _opt(xc="lda"), _opt(xc="lda", eps_xc=5.0),
                _opt(route="real_space")]
    keys = {_materials_fingerprint(_design(v), 1.2e-6) for v in variants}
    assert len(keys) == len(variants)
    # two separately built option sets with equal values give the same key
    assert _materials_fingerprint(_design(_opt()), 1.2e-6) == \
        _materials_fingerprint(_design(OPT), 1.2e-6)
    c = QuantumFilmCoupling(_solver(nz=121))
    st = c.state((0.0, 0.0))
    layers = [st.effective_layer(v, 1.2e-6) for v in variants[:6]]
    assert st.stats["responses_built"] == 6
    assert len({(round(a.real, 12), round(a.imag, 12), round(b.real, 12), round(b.imag, 12))
                for a, b, _ in layers}) == 6                     # every physical option moves the layer


def test_refusals_and_guards():
    """eps() without a ground state, a missing / wrong-type state, a state from another bias, a film
    layer of the wrong thickness, the reserved extra_fields key, invalid options."""
    from dynameta.core.bridge import assemble_eps
    with pytest.raises(ValueError, match="ground state"):
        OPT.eps(1.2e-6)
    with pytest.raises(ValueError, match="carries no"):
        OPT.eps_fields({"n": np.ones(3)}, 1.2e-6)
    with pytest.raises(TypeError, match="QuantumFilmState"):
        OPT.eps_fields({QUANTUM_FILM_FIELD: object()}, 1.2e-6)
    for bad in (dict(xc="rpa"), dict(route="fast"), dict(eps_inf=-1.0), dict(gamma_rad_s=-1.0)):
        with pytest.raises(ValueError):
            QuantumFilmOptical(**{**dict(eps_inf=EPS_INF, gamma_rad_s=GAM), **bad})
    # eps_inf and gamma_rad_s are required (no material-specific defaults)
    with pytest.raises(TypeError, match="eps_inf"):
        QuantumFilmOptical()
    s = _solver(nz=121)
    c = QuantumFilmCoupling(s)
    design = _design()
    opt = film_layer_optics(design, "film", c)
    cf = c.solve(BIASES[0])
    from dynameta.core.lift import IdentityLift
    from dynameta.core.n_to_eps import MaterialEpsMap
    args = (cf, opt.alignment(), MaterialEpsMap(design.materials), IdentityLift(), 1.2e-6)
    ef = assemble_eps(*args, extra_fields=c.extra_fields(BIASES[0]))["film"]
    assert ef.is_uniform and ef.is_tensor
    with pytest.raises(ValueError, match="does not match"):
        assemble_eps(*args, extra_fields=c.extra_fields(BIASES[1]))
    with pytest.raises(ValueError, match="carries no"):
        assemble_eps(*args)
    with pytest.raises(ValueError, match="thick"):
        film_layer_optics(_design(film_t=5e-9), "film", c)
    with pytest.raises(ValueError, match="no layer"):
        film_layer_optics(design, "nope", c)
    with pytest.raises(ValueError, match="reserved"):
        run_quantum_film_pipeline(design, Sweep(bias_points=BIASES[:1], wavelengths_nm=[1200.0]), c,
                                  film_layer="film", optical_solver=_berreman(), verbose=False,
                                  extra_fields={QUANTUM_FILM_FIELD: 1})
    with pytest.raises(TypeError, match="DualGateSchrodingerPoisson"):
        QuantumFilmCoupling(object())
    st = c.state(BIASES[0])
    assert isinstance(st, QuantumFilmState) and "QuantumFilmState" in repr(st)


def test_classical_film_through_the_same_entry_point_is_unchanged():
    """A DrudeOptical film (the classical graded film) through run_quantum_film_pipeline equals plain
    run_pipeline with the same carrier and builder and no extra field, bit for bit: the quantum
    field is invisible to every other material."""
    from dynameta.pipeline import run_pipeline
    drude = DrudeOptical(eps_inf=EPS_INF, m_opt_kg=0.30 * M_E, gamma_rad_s=GAM)
    design = _design(drude)
    sw = Sweep(bias_points=BIASES[:2], wavelengths_nm=[1200.0])
    s = _solver(nz=121)
    rows, _c = run_quantum_film_pipeline(design, sw, s, film_layer="film", optical_solver=_berreman(),
                                         verbose=False)
    s2 = _solver(nz=121)
    rows2 = run_pipeline(design, sw, carrier_solver=s2, optical_builder=film_layer_optics(design, "film", s2),
                         optical_solver=_berreman(), verbose=False)
    assert [r.result.r for r in rows] == [r.result.r for r in rows2]


def test_route_table():
    """What each optical route does with the equivalent layer (a uniform uniaxial tensor): the scalar
    layered TMM refuses with its tensor message; Berreman (planar), RCWA and PMM solve; Berreman
    refuses a grating with its planar-tier message. (The 2-D TM FEM is gated in
    tests/test_quantum_film_cell.py, which needs ngsolve.) The three planar solves agree to 1e-9."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver, make_lumenairy_rcwa_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    s = _solver(nz=121)
    c = QuantumFilmCoupling(s)
    sw = Sweep(bias_points=BIASES[:1], wavelengths_nm=[1200.0])
    with pytest.raises(ValueError, match="TENSOR"):
        run_quantum_film_pipeline(_design(), sw, c, film_layer="film", optical_solver=make_layered_tmm_solver(),
                                  verbose=False)
    r = {}
    for nm, mk in (("berreman", _berreman), ("rcwa", make_lumenairy_rcwa_solver),
                   ("pmm", make_lumenairy_pmm_solver)):
        rows, _ = run_quantum_film_pipeline(_design(pol="x", theta=0.0), sw, c, film_layer="film",
                                            optical_solver=mk(), verbose=False)
        r[nm] = rows[0].result.r
    assert abs(r["berreman"] - r["rcwa"]) < 1e-9 and abs(r["berreman"] - r["pmm"]) < 1e-9
    with pytest.raises(NotImplementedError, match="PLANAR"):
        run_quantum_film_pipeline(_design(strips=True, pol="x", theta=0.0), sw, c, film_layer="film",
                                  optical_solver=_berreman(), verbose=False)


def test_exact_element_at_evanescent_in_plane_wavevectors():
    """The per-harmonic reference (optics.quantum_film_cell) uses film_transfer_matrix at large q
    (evanescent harmonics, q L up to ~3). Independent check of its q handling there: with a LOCAL
    kernel R = diag(1 - eps_inf / eps_zz) on a uniform film it must equal the closed-form uniaxial
    layer matrix (the staggered grid is second order: budget 2e-5 relative at 961 nodes, MEASURED 5.2e-6),
    and with the RPA kernel the matrix must stay reciprocal (det T = 1) for every q."""
    from dynameta.optics.quantum_film import film_transfer_matrix, uniaxial_layer_matrix
    L, N = 6e-9, 961                                      # node spacing 6.25 pm
    h = L / (N - 1)
    w = 2.0 * math.pi * C_LIGHT / 1.2e-6
    k0 = w / C_LIGHT
    exx, ezz, einf = 2.2 + 0.1j, 1.7 + 0.4j, EPS_INF
    R = np.diag(np.full(N - 1, 1.0 - einf / ezz))
    for q in (0.0, 1e8, 3e8, 6e8):
        T = film_transfer_matrix(w, q, h, np.full(N, exx), einf, R)
        U = uniaxial_layer_matrix(exx, ezz, L, k0, q)
        assert np.max(np.abs(T - U)) / np.max(np.abs(U)) < 2e-5, q
    resp = QuantumFilmResponse(QuantumFilm.flat_band(6e-9, 3e26, 0.30 * M_E, nz=145), EPS_INF, GAM)
    for q in (0.0, 3e8, 1e9):
        T = resp.transfer_matrix(w, q)
        assert abs(np.linalg.det(T) - 1.0) < 1e-8
