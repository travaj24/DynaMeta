"""Gates for optics/quantum_film_cell.py (v0.16): the per-harmonic exact treatment of a laterally uniform
film under a lamellar grating, and the 2-D TM FEM route of the quantum-film pipeline.

New module: on v0.15.1 every gate fails at `import dynameta.optics.quantum_film_cell`.

Oracles (independent of the module under test):
  * planar stacks: optics.quantum_film.film_stack_rt (the exact planar film element composed with
    nonlocal_tmm layers) -- shares only the film's transfer matrix with the cell solver, not its FEM,
    ports, Woodbury update or extraction;
  * the metal-strip cell with a uniform uniaxial film: optics.fem_2d_tm (PML truncation, its own mesh,
    probe-line extraction, volume absorption) -- shares no solver code;
  * energy: R + T + every absorption = 1 (FEM-region volume integrals + per-harmonic Poynting fluxes of
    the stack below), an identity the discretization does not enforce by construction.
"""
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.constants import C_LIGHT, M_E                                          # noqa: E402
from dynameta.optics import nonlocal_tmm as nt                                       # noqa: E402
from dynameta.optics.fem_2d_tm import TMBlock, TMLayer, TMMeshSpec, solve_fem_2d_tm  # noqa: E402
from dynameta.optics.quantum_film import (QuantumFilm, QuantumFilmResponse,          # noqa: E402
                                          film_stack_rt, uniaxial_layer_matrix)
from dynameta.optics.quantum_film_cell import (CellMeshSpec, FilmCell, FilmLayer,    # noqa: E402
                                               UniformLayer, film_spectrum, solve_film_cell)

# the generic gated-film strip cell
LAM = 1200e-9
P, W = 400e-9, 150e-9
EA, EB = 1.70 ** 2, 2.00 ** 2                          # oxA (2 nm), oxB (6 nm)
E_STRIP, E_MIRROR, N_SUB = -60 + 4j, -120 + 20j, 1.45
# the 0 V equivalent layer of the fixture film (481 nodes, 1200 nm)
EXX, EZZ = 2.7503 + 0.0955j, 2.5838 + 0.4766j
STRIP = TMBlock(P / 2 - W / 2, P / 2 + W / 2, E_STRIP, "metal")
EPS_INF, GAM = 4.0, 1.2e14


def _below():
    return [UniformLayer("mirror", 80e-9, E_MIRROR), UniformLayer("lo_oxa", 2e-9, EA),
            UniformLayer("lo_oxb", 6e-9, EB)]


def _cell_layers(strip=True):
    return [TMLayer("oxb_in", 3e-9, EB), TMLayer("oxa", 2e-9, EA),
            TMLayer("grating", 40e-9, 1.0, (STRIP,) if strip else ())]


def _solver(**kw):
    from dynameta.carriers import DualGateSchrodingerPoisson, GateStack
    ox = GateStack(((2e-9, 9.0), (6e-9, 20.0)))
    args = dict(t_film_m=6e-9, bottom=ox, top=ox, n_bg_m3=3e26, m_eff_kg=0.30 * M_E, eps_static=9.0, nz=201)
    args.update(kw)
    return DualGateSchrodingerPoisson(**args)


@pytest.fixture(scope="module")
def response():
    s = _solver()
    return QuantumFilmResponse(QuantumFilm.from_dual_gate(s, s.solve_voltages(0.0, 3.0)), EPS_INF, GAM)


def test_planar_stack_reproduces_the_exact_film_element(response):
    """No grating: the cell solver with the EXACT film per harmonic equals film_stack_rt at normal
    ('x') and 35 deg ('p') incidence, for the exact and the equivalent film. MEASURED |dr| <= 2.7e-14,
    energy residual <= 2.5e-15 (budget 1e-10 / 1e-12). Sign conventions: film_stack_rt's r is the p-hat r_H (as optics.fem_2d_tm), so
    'x' (normal incidence, r = E_x ratio = -r_H) r = -r_film_stack_rt and 'p' r = r_film_stack_rt; a wrong
    port sign or admittance branch fails this by O(1)."""
    w = 2.0 * math.pi * C_LIGHT / LAM
    above = [nt.DielectricLayer(1.0, 40e-9), nt.DielectricLayer(EA, 2e-9), nt.DielectricLayer(EB, 6e-9)]
    below = [nt.DielectricLayer(EB, 6e-9), nt.DielectricLayer(EA, 2e-9), nt.DielectricLayer(E_MIRROR, 80e-9)]
    exx, ezz, L = response.effective_layer(w)
    for theta, pol, sgn in ((0.0, "x", -1.0), (35.0, "p", 1.0)):
        cell = FilmCell(P, _cell_layers(strip=False), lambda_m=LAM, theta_deg=theta, polarization=pol,
                        order=4, n_harmonics=8)
        for film, fm in ((FilmLayer.exact(response), lambda ww, kp: response.transfer_matrix(ww, kp)),
                         (FilmLayer.equivalent(response),
                          lambda ww, kp: uniaxial_layer_matrix(exx, ezz, L, ww / C_LIGHT, kp))):
            res = cell.solve(_below() + [film, UniformLayer("oxb_rest", 3e-9, EB)], N_SUB ** 2)
            ref = film_stack_rt(w, fm, above, below, n_super=1.0, n_sub=N_SUB, theta_rad=math.radians(theta))
            assert abs(res.r - sgn * ref.r) < 1e-10, (theta, film.label, abs(res.r - sgn * ref.r))
            assert abs(res.R - ref.R) < 1e-10 and abs(res.T - ref.T) < 1e-10
            assert abs(res.energy_residual) < 1e-12


def test_strip_cell_with_a_uniaxial_film_matches_the_2d_fem():
    """The metal-strip cell with the film as a uniform uniaxial layer: cell solver (order 4, 100
    harmonics, 8 grading levels) vs optics.fem_2d_tm (order 6, 10 levels; its own convergence ladder
    puts it within 1.0e-6 in r and 2.3e-7 in A_film of order 8 / 14 levels). MEASURED |dr| = 6.2e-7,
    |dA_film| = 1.1e-7, |dA_metal| = 1.1e-7; budget 2e-5 / 1e-6. Energy residual 1.2e-12 (budget 1e-10).
    The generic cell converges more slowly in the absorption at the strip corners: at the
    old 6 levels the cell is 1.8e-6 off the converged A_film (budget 1e-6), so the cell mesh was refined
    to 8 levels instead of widening the budget."""
    full = _full_stack()
    ref = solve_fem_2d_tm(P, full, lambda_m=LAM, n_super=1.0, n_sub=N_SUB, order=6,
                          mesh_spec=TMMeshSpec(corner_levels=10), num_threads=2)
    res = solve_film_cell(P, _cell_layers(), _below() + [UniformLayer("film", 6e-9, (EXX, EZZ)),
                                                          UniformLayer("oxb_rest", 3e-9, EB)],
                          lambda_m=LAM, eps_sub=N_SUB ** 2, order=4, n_harmonics=100,
                          mesh=CellMeshSpec(corner_levels=8))
    assert abs(res.r - ref.r) < 2e-5, abs(res.r - ref.r)
    assert abs(res.absorption["film"] - ref.per_region_absorption["film"]) < 1e-6
    assert abs(res.absorption["metal"] - ref.per_region_absorption["metal"]) < 1e-6
    assert abs(res.energy_residual) < 1e-10


def _full_stack():
    """The whole strip cell with the film as the uniform uniaxial layer (EXX, EZZ), for optics.fem_2d_tm."""
    return [TMLayer("mirror", 80e-9, E_MIRROR), TMLayer("lo_oxa", 2e-9, EA), TMLayer("lo_oxb", 6e-9, EB),
            TMLayer("film", 6e-9, (EXX, EZZ)), TMLayer("up_oxb", 6e-9, EB), TMLayer("up_oxa", 2e-9, EA),
            TMLayer("grating", 40e-9, 1.0, (STRIP,))]


def test_fem_2d_route_of_the_pipeline_takes_the_equivalent_layer():
    """run_quantum_film_pipeline with the 2-D TM FEM seam (make_fem_2d_tm_optical_solver): the film
    layer arrives as a uniform (3, 3) tensor EpsField and the Design adapter solves it -- identical to
    solving the same layers by hand with the film as (eps_xx,eq, eps_zz,eq)."""
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
    opt = QuantumFilmOptical(eps_inf=EPS_INF, gamma_rad_s=GAM)
    reg.add(Material("FILM", opt))
    spec = (("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"), ("lo_oxb", 6e-9, "oxB"),
            ("film", 6e-9, "FILM"), ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA"))
    layers = [Layer(n, t, m) for n, t, m in spec]
    layers.append(Layer("grating", 40e-9, "air", inclusions=[Inclusion(Rectangle(P / 2, P / 2, W, P), "metal")]))
    d = Design(name="qf", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
               stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=reg, optical=OpticalSpec(polarization="x"))
    s = _solver(nz=121, semi_material="FILM")
    bp = BiasPoint({"bottom": 0.0, "top": 2.0}, "b")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows, c = run_quantum_film_pipeline(d, Sweep(bias_points=[bp], wavelengths_nm=[1200.0]), s,
                                            film_layer="film", optical_solver=make_fem_2d_tm_optical_solver(num_threads=2),
                                            verbose=False)
    # the wavelength exactly as the pipeline row carries it: 1200.0 nm * 1e-9 is one ulp above 1200e-9, and
    # that one-ulp change alone moves this FEM's r by 3.7e-11 (MEASURED; above the 1e-12 identity bar)
    lam = rows[0].lambda_nm * 1e-9
    exx, ezz, _ = c.effective_layer(bp, opt, lam)
    full = [TMLayer(n, t, complex(reg.get(m).eps(lam)) if m != "FILM" else (exx, ezz)) for n, t, m in spec]
    full.append(TMLayer("grating", 40e-9, 1.0 + 0j, (TMBlock(P / 2 - W / 2, P / 2 + W / 2, E_STRIP, "grating__incl0"),)))
    ref = solve_fem_2d_tm(P, full, lambda_m=lam, n_super=1.0, n_sub=N_SUB, num_threads=2)
    assert abs(rows[0].result.r - ref.r) < 1e-12


def test_refusals():
    with pytest.raises(ValueError, match="polarization='p'"):
        FilmCell(P, _cell_layers(), lambda_m=LAM, theta_deg=10.0, polarization="x")
    with pytest.raises(ValueError, match="lossless"):
        FilmCell(P, _cell_layers(), lambda_m=LAM, n_super=1.0 + 0.1j)
    with pytest.raises(ValueError, match="polarization"):
        FilmCell(P, _cell_layers(), lambda_m=LAM, polarization="y")


@pytest.mark.slow
def test_strip_cell_tight_agreement_and_harmonic_convergence(response):
    """Order 5 / 150 harmonics / 8 levels vs fem_2d_tm order 7 / 12 levels: MEASURED |dr| 8.6e-8,
    |dA_film| 3.3e-8 (budget 5e-6 / 1e-7); 150 vs 250 harmonics move r by 8.2e-11 (budget 1e-8). With
    the EXACT film (one-sided gating, (0, +3) V): exactly treated film harmonics 40 vs 80 move r by
    < 1e-7 (MEASURED 3.6e-9 at 201 nodes); the exact-vs-equivalent difference is mesh independent
    (order 4 vs 5: < 2e-6; MEASURED 5.3e-8) and well above both (MEASURED 3.05e-3 at 201 nodes).
    The generic cell's reference had to move to order
    7 / 12 levels: order 6 / 10 is itself 2.3e-7 off the converged A_film (order 8 / 14), above the 1e-7
    budget."""
    full = _full_stack()
    ref = solve_fem_2d_tm(P, full, lambda_m=LAM, n_super=1.0, n_sub=N_SUB, order=7,
                          mesh_spec=TMMeshSpec(corner_levels=12), num_threads=2)
    uni = _below() + [UniformLayer("film", 6e-9, (EXX, EZZ)), UniformLayer("oxb_rest", 3e-9, EB)]
    c5 = FilmCell(P, _cell_layers(), lambda_m=LAM, order=5, n_harmonics=150, mesh=CellMeshSpec(corner_levels=8))
    r5 = c5.solve(uni, N_SUB ** 2)
    assert abs(r5.r - ref.r) < 5e-6
    assert abs(r5.absorption["film"] - ref.per_region_absorption["film"]) < 1e-7
    r250 = solve_film_cell(P, _cell_layers(), uni, lambda_m=LAM, eps_sub=N_SUB ** 2, order=5, n_harmonics=250,
                           mesh=CellMeshSpec(corner_levels=8))
    assert abs(r250.r - r5.r) < 1e-8
    c4 = FilmCell(P, _cell_layers(), lambda_m=LAM, order=4, n_harmonics=100, mesh=CellMeshSpec(corner_levels=6))
    ex, eq = FilmLayer.exact(response), FilmLayer.equivalent(response)
    diffs = []
    for cell in (c4, c5):
        a = cell.solve(_below() + [eq, UniformLayer("oxb_rest", 3e-9, EB)], N_SUB ** 2)
        b40 = cell.solve(_below() + [ex, UniformLayer("oxb_rest", 3e-9, EB)], N_SUB ** 2,
                         film_harmonics=40, film_fallback=eq)
        b80 = cell.solve(_below() + [ex, UniformLayer("oxb_rest", 3e-9, EB)], N_SUB ** 2,
                         film_harmonics=80, film_fallback=eq)
        assert abs(b40.r - b80.r) < 1e-7
        assert abs(b80.energy_residual) < 1e-10
        diffs.append(b80.r - a.r)
        sp = film_spectrum(b80)
        assert abs(np.sum(sp["A_m"]) - b80.absorption["film"]) < 1e-12
    assert abs(diffs[0] - diffs[1]) < 2e-6
    assert abs(diffs[1]) > 100 * abs(diffs[0] - diffs[1])


def test_p_reflection_sign_agrees_with_solve_film_cell_and_fem_2d_tm():
    """ONE sign of r_p (the solver part; the solver-free part is tests/test_quantum_film.py::
    test_p_reflection_sign_of_film_stack_rt_against_nonlocal_tmm_and_tmm). The same planar p-pol stack (air |
    eps 4.0, 20 nm | uniaxial equivalent film of the 6 nm flat-band response | oxide 1.7^2, 20 nm | glass 1.45)
    at 50 deg: QuantumFilmResponse.stack_rt (new in v0.16) == solve_film_cell ('p') to 1e-13 (MEASURED 4.8e-15:
    two transfer-matrix codes, round-off) and == fem_2d_tm ('p', the bridges' p-hat convention r = r_H) to 1e-6
    (its default-mesh discretization, MEASURED 2.1e-10). abs(r) = 0.094, so a flipped sign misses each equality
    by 0.19 (stack_rt returning the tangential-E r gave abs(stack_rt - solve_film_cell) = 0.188)."""
    lam, th = 1200e-9, 50.0
    w = 2.0 * math.pi * C_LIGHT / lam
    q = QuantumFilmResponse(QuantumFilm.flat_band(6e-9, 3e26, 0.30 * M_E, nz=201), EPS_INF, GAM)
    exx, ezz, L = q.effective_layer(w)
    cap, ox = nt.DielectricLayer(4.0, 20e-9), nt.DielectricLayer(1.7 ** 2, 20e-9)
    a = q.stack_rt(w, [cap], [ox], model="equivalent", n_super=1.0, n_sub=1.45, theta_rad=math.radians(th))
    c = solve_film_cell(300e-9, [TMLayer("cap", 20e-9, 4.0)], [UniformLayer("ox", 20e-9, 1.7 ** 2),
                                                               FilmLayer.equivalent(q)],
                        lambda_m=lam, eps_sub=1.45 ** 2, theta_deg=th, polarization="p", n_harmonics=8)
    assert abs(a.r - c.r) < 1e-13, abs(a.r - c.r)
    f = solve_fem_2d_tm(300e-9, [TMLayer("ox", 20e-9, 1.7 ** 2), TMLayer("film", L, (exx, ezz)),
                                 TMLayer("cap", 20e-9, 4.0)],
                        lambda_m=lam, theta_deg=th, n_super=1.0, n_sub=1.45, polarization="p", num_threads=2)
    assert abs(a.r - f.r) < 1e-6, abs(a.r - f.r)
    assert abs(a.r) > 0.05                                           # the sign is resolvable here
