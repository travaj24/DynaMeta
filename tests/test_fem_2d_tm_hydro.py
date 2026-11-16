"""Gates for the nonlocal (hydrodynamic) conductor layer of the 2-D TM FEM
(optics/fem_2d_tm_hydro.py through optics/fem_2d_tm.py).

The module is new in v0.16: on v0.15.1 every gate fails at `import dynameta.optics.fem_2d_tm_hydro`
(ModuleNotFoundError), and fem_2d_tm refuses hydrodynamic input outright.

Oracles (independent of the FEM code path unless labelled a consistency check):
  * optics.nonlocal_tmm: the planar transfer-matrix route -- a banded wave-amplitude solve over a
    staircase of uniform sub-layers (graded films; Richardson-extrapolated in the sub-layer count,
    second order) and the closed-form single-layer ABC matrix (uniform films). It shares only the
    material kernel with the FEM (eps_transverse, beta_from_density, hydro_psi_per_charge), no
    discretization, no basis, no boundary treatment;
  * the local FEM (fem_2d_tm with a GriddedEps2D eps_T) for the beta -> 0 limit;
  * exact symmetries (lateral translation, reciprocity) -- consistency checks;
  * the strip cell's own order-7 / 10-level solution -- a self-convergence (consistency) check.

Fixture (generic gated film): air | 6 nm oxB (n 2.00) | 6 nm graded film | 6 nm oxB | glass
(n 1.45), p-pol, 60 deg, 1200 nm; film eps_inf 4.0, m 0.30 m_e, gamma 1.2e14 1/s, 'high_freq' beta,
n0(z) = 3e26 + 1.2e27 exp(-z / 1 nm) on 11 samples (z from the top face; Re eps_T runs from +2.7 to -2.4
across the film). Sign convention: fem_2d_tm 'p' returns r_H, nonlocal_tmm
the tangential-E ratio, so r_fem = -r_planar (checked on a local film too: measured sum 6e-11).

Error budget for the planar gates (tolerance 1e-8 in r): FEM order 4 on the sample-aligned
conductor mesh 1e-10 (order ladder below), PML / cladding discretization ~1e-9 (the local solver's
own planar gates), Richardson-extrapolated staircase ~1e-11.
"""
import cmath
import dataclasses
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.constants import C_LIGHT                                      # noqa: E402
from dynameta.materials import M_E                                          # noqa: E402
from dynameta.optics import nonlocal_tmm as nt                              # noqa: E402
from dynameta.optics.fem_2d_tm import (                                     # noqa: E402
    GriddedEps2D, TMBlock, TMLayer, TMMeshSpec, solve_fem_2d_tm)
from dynameta.optics.fem_2d_tm_hydro import (                               # noqa: E402
    HydroConductor2D, hydro_conductor_from_profile, hydro_z_convergence)
from dynameta.warning_types import UnreliableResultWarning                 # noqa: E402

LAM = 1200e-9
THREADS = 2
HF = 2.00 ** 2                                        # the oxB cladding of the film
KW = dict(m_eff_kg=0.30 * M_E, eps_inf=4.0, gamma=1.2e14)
T_F = 6e-9                                            # film thickness
ZP = np.linspace(0.0, T_F, 11)                       # from the TOP face (planar convention)
NP = 3e26 + 1.2e27 * np.exp(-ZP / 1e-9)


def _omega(lam):
    return 2 * math.pi * C_LIGHT / lam


def _planar(film_top_first, lam=LAM, theta=60.0, top=None, bot=None):
    """-r of nonlocal_tmm for oxB | film | oxB (or given claddings), i.e. fem_2d_tm's r_H."""
    top = [nt.DielectricLayer(HF, 6e-9)] if top is None else top
    bot = [nt.DielectricLayer(HF, 6e-9)] if bot is None else bot
    res = nt.stack_rt(_omega(lam), top + list(film_top_first) + bot, pol="p", n_super=1.0,
                      n_sub=1.45, theta_rad=math.radians(theta))
    return -res.r, res


def _planar_graded(lam=LAM, theta=60.0, D=0.0, beta_scale=1.0, top=None, bot=None, kw=None):
    """Richardson extrapolation of the second-order staircase (N = 4096, 8192)."""
    kw = KW if kw is None else kw
    rs = []
    for n_layers in (4096, 8192):
        run = nt.hydro_layers_from_profile(ZP, NP, n_layers=n_layers, D=D, **kw)
        if beta_scale != 1.0:
            run = [dataclasses.replace(L, beta=L.beta * beta_scale) for L in run]
        rs.append(_planar(run, lam, theta, top, bot)[0])
    return rs[1] + (rs[1] - rs[0]) / 3.0


def _film(cond):
    return [TMLayer("lo", 6e-9, HF), TMLayer("f", T_F, cond), TMLayer("up", 6e-9, HF)]


def _solve(layers, lam=LAM, theta=60.0, P=100e-9, order=4, **kw):
    pol = "x" if theta == 0 else "p"
    return solve_fem_2d_tm(P, layers, lambda_m=lam, theta_deg=theta, n_super=1.0, n_sub=1.45,
                           polarization=pol, order=order, return_solution=True,
                           num_threads=THREADS, **kw)


def _cond(**kw):
    opts = dict(KW)
    opts.update(check_validity=False)
    opts.update(kw)
    return hydro_conductor_from_profile(ZP, NP, **opts)


# ------------------------------------------------------------------------------------------------
# planar limits vs the planar route
# ------------------------------------------------------------------------------------------------

def test_planar_graded_film_matches_nonlocal_tmm_at_oblique_incidence():
    """Laterally uniform cell, 60 deg: the graded film vs nonlocal_tmm's run (Richardson in the
    sub-layer count). Measured |dr| = 1.4e-10 (order 4), 2.7e-10 (order 5); order 3 7.3e-8. The
    nonlocal effect itself is 2.3e-3 in r (vs the local eps_T film), so a gate at 1e-8 resolves it
    to 5 digits."""
    r_ref = _planar_graded()
    sol = _solve(_film(_cond()))
    assert abs(sol.result.r - r_ref) < 1e-8
    assert abs(sol.energy_residual) < 1e-8
    loc = _solve(_film(_cond(beta_scale=0.0))).result.r
    assert abs(loc - r_ref) > 2e-3                     # the nonlocal effect is resolved


def test_profile_constructor_drives_both_routes_identically():
    """hydro_conductor_from_profile takes hydro_layers_from_profile's arguments; its planar_layers()
    IS hydro_layers_from_profile's run (dataclass equality), and n_layers is refused (the 2-D
    route uses the continuous profile)."""
    c = hydro_conductor_from_profile(ZP, NP, **KW)
    mine, ref = c.planar_layers(), nt.hydro_layers_from_profile(ZP, NP, **KW)
    assert len(mine) == len(ref)
    for a_, b_ in zip(mine, ref):                     # identical up to z round-off (1e-25 m)
        assert abs(a_.thickness_m - b_.thickness_m) < 1e-22
        assert dataclasses.replace(a_, thickness_m=b_.thickness_m) == b_
    assert abs(c.thickness_m - T_F) < 1e-24
    assert np.array_equal(c.n0_m3, NP[::-1])          # stored from the layer bottom
    with pytest.raises(ValueError, match="n_layers"):
        hydro_conductor_from_profile(ZP, NP, n_layers=8, **KW)


@pytest.mark.parametrize("case", ["film_evanescent", "na_propagating"])
def test_uniform_film_matches_the_closed_form_single_layer(case):
    """A uniform film vs nonlocal_tmm's closed-form single-layer ABC matrix (a run of one layer).
    Degenerate-semiconductor-like 8e26 m^-3 at 1200 nm (evanescent longitudinal wave, |k_L| d ~ 4.4):
    measured 1.3e-10 (order 4). Na-like (2.65e28, m_e, eps_inf 1) at 1.2 wp, 45 deg (PROPAGATING
    longitudinal wave, k_L d = 44, across the 6 nm film): measured 6.3e-12 (order 5)."""
    if case == "film_evanescent":
        n0, m, ei, lam, th, order = 8e26, 0.30 * M_E, 4.0, LAM, 60.0, 4
    else:
        n0, m, ei, lam, th, order = 2.65e28, M_E, 1.0, None, 45.0, 5
    lay = nt.hydro_layers_from_profile(np.array([0.0, T_F]), np.array([n0, n0]), m_eff_kg=m,
                                       eps_inf=ei, gamma=1.2e14)[0]
    if lam is None:
        lam = 2 * math.pi * C_LIGHT / (1.2 * lay.wp)
    single = nt.HydroLayer(lay.eps_inf, lay.wp, lay.gamma, lay.beta, T_F)
    r_ref = _planar([single], lam, th)[0]
    cond = HydroConductor2D(np.array([n0, n0]), np.array([0.0, T_F]), m, ei, 1.2e14,
                            check_validity=False)
    sol = _solve(_film(cond), lam, th, order=order)
    assert abs(sol.result.r - r_ref) < 1e-8
    assert abs(sol.energy_residual) < 1e-6


def _dissipation_split(D):
    """(total film absorption, Ohmic part, diffusion part) of the graded film at 60 deg, order 4, all in
    the solver's normalisation. With X the free current and q the conductor potential of the
    hydrodynamic extension, omega eps0 E = i X / Delta + i grad q and div X = -kappa^2 q (J.n = 0), so
    Re(conj(X) . omega eps0 E) integrates to  Re(i / Delta) |X|^2  (the Ohmic gamma loss:
    omega gamma / wp^2 |X|^2)  +  Re(i conj(kappa^2)) |q|^2  (the GNOR diffusion loss: kappa^2 is real
    for D = 0, where the pressure term is reactive). Both are integrated here separately from the
    solver's total (1/2) Re(conj(J) . E)."""
    import types
    import ngsolve as ng
    from dynameta.optics.fem_2d_tm import build_tm_cell, solve_tm_cell
    from dynameta.optics.fem_2d_tm_hydro import hydro_extensions_for_cell
    cell = build_tm_cell(100e-9, _film(_cond(D=D)), lambda_m=LAM, theta_deg=60.0, n_super=1.0, n_sub=1.45)
    sol = solve_tm_cell(cell, polarization="p", order=4, num_threads=THREADS, keep_field=True)
    ext = hydro_extensions_for_cell(cell)[0]
    k0 = 2 * math.pi / (LAM * 1e9)
    ctx = types.SimpleNamespace(ng=ng, cell=cell, k0=k0, kx=k0 * math.sin(math.radians(60.0)))
    a1, kappa2 = ext._coefficients(ctx)
    X, q = sol.gfu.components[1], sol.gfu.components[2]
    inv_delta = a1 - 1.0 / float(ext.spec.eps_inf)
    on = cell.mesh.Materials("f")
    p_ohm = complex(ng.Integrate(1j * inv_delta * (ng.Conj(X[0]) * X[0] + ng.Conj(X[1]) * X[1]), cell.mesh,
                                 definedon=on, order=10)).real / ext._norm(ctx)
    p_dif = complex(ng.Integrate(1j * ng.Conj(kappa2) * ng.Conj(q) * q, cell.mesh, definedon=on,
                                 order=10)).real / ext._norm(ctx)
    return sol.result.per_region_absorption["f"], p_ohm, p_dif


def test_gnor_graded_film_matches_planar_and_closes_energy():
    """GNOR diffusion D = 1e-3 m^2/s: measured |dr| = 1.1e-10 vs the planar run (order 4); the
    independent absorption (1/2) Re(conj(J).E) closes R + T + A = 1 to 8.1e-10. The diffusion loss
    channel is positive and accounted for: the film absorption splits into an Ohmic (gamma) part and a
    diffusion part integrated separately (_dissipation_split); measured with D: total 0.0070455 = Ohmic
    0.0038959 + diffusion 0.0031496 (sum to 2e-17); without D the diffusion part is 7e-20 and the
    Ohmic part is the whole 0.0122893. The film absorption itself can move EITHER way with D --
    GNOR shifts and broadens the longitudinal resonance (here it damps the current, so the Ohmic loss
    falls by more than the diffusion loss adds: 0.00705 vs 0.01229, -43 %) -- so that gate is
    |A_D / A_0 - 1| > 0.1."""
    r_ref = _planar_graded(D=1e-3)
    sol = _solve(_film(_cond(D=1e-3)))
    sol0 = _solve(_film(_cond()))
    assert abs(sol.result.r - r_ref) < 1e-8
    assert abs(sol.energy_residual) < 1e-8
    assert abs(sol.result.per_region_absorption["f"] / sol0.result.per_region_absorption["f"] - 1.0) > 0.1
    tot, ohm, dif = _dissipation_split(1e-3)
    assert abs(tot - sol.result.per_region_absorption["f"]) < 1e-12
    assert dif > 1e-3 and ohm > 0.0                       # the diffusion channel: positive (3.1e-3)
    assert abs(tot - ohm - dif) < 1e-8                    # and the whole remainder (closure tolerance)
    tot0, ohm0, dif0 = _dissipation_split(0.0)
    assert abs(dif0) < 1e-12 and abs(tot0 - ohm0) < 1e-8  # D = 0: the pressure term dissipates nothing

@pytest.mark.parametrize("lam", [1200e-9, 2500e-9])
def test_lossless_graded_film_has_no_spurious_gain(lam):
    """gamma = 0, D = 0: the conductor absorbs nothing. At 1200 nm Re eps_T crosses zero inside the
    film (+2.7 to -2.4), at 2500 nm it is metallic throughout (-1.5 .. -23.3). Measured |A_film| <= 1.8e-17,
    1 - R - T = 7.7e-10 / 1.9e-10 (the claddings' discretization, as for the local solver), |dr| vs the planar run
    6.0e-10 / 5.5e-11."""
    kw = dict(KW, gamma=0.0)
    r_ref = _planar_graded(lam=lam, kw=kw)
    sol = _solve(_film(_cond(gamma=0.0)), lam)
    res = sol.result
    assert abs(res.per_region_absorption["f"]) < 1e-14
    assert abs(1.0 - res.R - res.T) < 1e-8
    assert abs(res.r - r_ref) < 1e-8


def test_local_limit_beta_to_zero():
    """beta_scale = 0 (D = 0) is solved by the local solver with eps_T(z): bit-identical to the
    plain local solve of the same eps_T samples. Small beta: |r - r_local| is LINEAR in beta
    (measured 1.921e-4 / 1.912e-5 at beta_scale 0.1 / 0.01, all-metallic profile at 2500 nm) and
    equals the planar run at the same beta to 3.3e-11."""
    lam = 2500e-9
    c0 = _cond(beta_scale=0.0)
    s0 = _solve(_film(c0), lam)
    et = np.array([nt.eps_transverse(_omega(lam), L) for L in c0._point_layers(c0._n2d()[:, 0])])
    s_loc = _solve(_film(GriddedEps2D(et, c0._z_nm() * 1e-9)), lam)
    assert s0.result.r == s_loc.result.r and s0.ndof == s_loc.ndof
    d = []
    for bs in (0.1, 0.01):
        sol = _solve(_film(_cond(beta_scale=bs)), lam)
        d.append(abs(sol.result.r - s0.result.r))
        assert abs(sol.result.r - _planar_graded(lam=lam, beta_scale=bs)) < 1e-8
    assert abs(d[0] / d[1] - 10.0) < 0.1


# ------------------------------------------------------------------------------------------------
# structured cells
# ------------------------------------------------------------------------------------------------

# the generic strip cell (tests/_strip_cell_fixture) with the graded hydrodynamic film
P_A, W_A = 400e-9, 150e-9
R_STRIP_CONVERGED = -0.12019454097 - 0.45828657365j   # order 7, 10 corner levels


def _strip_layers(film):
    return [TMLayer("mirror", 80e-9, -120 + 20j), TMLayer("lo_oxa", 2e-9, 1.70 ** 2),
            TMLayer("lo_oxb", 6e-9, HF), TMLayer("film", T_F, film),
            TMLayer("up_oxb", 6e-9, HF), TMLayer("up_oxa", 2e-9, 1.70 ** 2),
            TMLayer("grating", 40e-9, 1.0 + 0j,
                    (TMBlock(P_A / 2 - W_A / 2, P_A / 2 + W_A / 2, -60 + 4j, "strip"),))]


def test_strip_grating_over_graded_film():
    """Metal strips over the graded film (accumulation at the gate side), TM, normal incidence,
    order 4 / 6 corner levels. CONSISTENCY gates (no independent structured oracle at this level; see
    the validation script): |r - r(order 7, 10 levels)| measured 6.5e-6; energy closes with the
    hydrodynamic absorption (R + T + A - 1 = 1.6e-9). The nonlocal correction against the local solve
    of the same profile: |r_nl - r_loc| = 0.150, and the nonlocal film absorbs less than the local one
    (0.570 vs 0.657): A_loc - A_nl = 0.086 measured, gated at > 0.05 directly (a pressure term weakened
    to beta_scale 0.3 gives 0.021); the band on A_nl pins its value (a solve that drops the free-carrier
    current altogether leaves a near-lossless film)."""
    sol = solve_fem_2d_tm(P_A, _strip_layers(_cond()), lambda_m=LAM, n_super=1.0, n_sub=1.45,
                          order=4, return_solution=True, num_threads=THREADS)
    loc = solve_fem_2d_tm(P_A, _strip_layers(_cond(beta_scale=0.0)), lambda_m=LAM, n_super=1.0,
                          n_sub=1.45, order=4, return_solution=True, num_threads=THREADS)
    res = sol.result
    assert abs(res.r - R_STRIP_CONVERGED) < 1e-4
    assert abs(sol.energy_residual) < 1e-7
    assert abs(res.R + res.T + res.A_independent - 1.0) < 1e-7
    assert abs(res.r - loc.result.r) > 0.1
    assert 0.52 < res.per_region_absorption["film"] < 0.62
    assert loc.result.per_region_absorption["film"] - res.per_region_absorption["film"] > 0.05


def test_structured_mesh_with_planar_physics_matches_planar():
    """A 20 nm metal sheet containing a BLOCK of the same metal (structured layer, block edges with
    geometric corner grading, planar physics) over the graded film, 80 nm mirror metal below, 30 deg:
    vs the planar route with the same layers. Measured |dr| at orders 3 / 4 / 5 / 6: block 5.9e-6 / 1.6e-8 /
    1.3e-9 / 1.2e-10, the flat sheet without the block 5.9e-6 / 1.6e-8 / 1.3e-9 / 1.0e-10 -- discretization of
    this stack, falling with the order, identical with and without the block; gated at order 5 (1e-8) and
    at order 4 (5e-8). Exercises the hydrodynamic layer inside a corner-graded structured mesh against an
    independent planar solver.
    corner_mode="all": the default rule does not grade the edges of a block of the sheet's own metal (no
    permittivity jump across the edge: a flat metal / dielectric interface, no corner singularity), so the
    grading this gate exercises is requested explicitly."""
    from dynameta.optics.fem_2d_tm import TMMeshSpec
    em = -60 + 4j
    lays = [TMLayer("mirror", 80e-9, -120 + 20j), TMLayer("lo", 6e-9, HF),
            TMLayer("f", T_F, _cond()), TMLayer("up", 6e-9, HF),
            TMLayer("sheet", 20e-9, em, (TMBlock(100e-9, 250e-9, em, "m_a"),))]
    # the 1e-8 bar at order 5; order 4 sits at 1.6e-8 on this stack (the ladder in the docstring)
    sol = _solve(lays, LAM, 30.0, P=P_A, order=5, mesh_spec=TMMeshSpec(corner_mode="all"))
    sol4 = _solve(lays, LAM, 30.0, P=P_A, order=4, mesh_spec=TMMeshSpec(corner_mode="all"))
    assert len(sol.cell.corners_nm) == 4
    r_ref = _planar_graded(theta=30.0, top=[nt.DielectricLayer(em, 20e-9), nt.DielectricLayer(HF, 6e-9)],
                           bot=[nt.DielectricLayer(HF, 6e-9), nt.DielectricLayer(-120 + 20j, 80e-9)])
    assert abs(sol.result.r - r_ref) < 1e-8
    assert abs(sol4.result.r - r_ref) < 5e-8                        # the default order stays exercised


def test_laterally_structured_density():
    """n0(x, z): identical columns reproduce the z-only route (measured 0, bit for bit; identical columns
    now collapse to the z-only path); an x-modulated density obeys lateral translation by a
    sample-aligned shift (4.2e-15) and reciprocity r(theta) = r(-theta) (6.7e-15) -- consistency
    checks; the modulation itself moves r by 9.7e-4."""
    P = 200e-9
    cz = _cond()
    x = np.linspace(0.0, P, 16, endpoint=False)
    nxz = np.repeat(cz._n2d(), x.size, axis=1)
    r_z = _solve(_film(cz), theta=30.0, P=P).result.r
    cxz = HydroConductor2D(nxz, cz.z_m, cz.m_eff_kg, 4.0, 1.2e14, x_m=x, check_validity=False)
    assert abs(_solve(_film(cxz), theta=30.0, P=P).result.r - r_z) < 1e-11
    mod = 1 + 0.3 * np.cos(2 * np.pi * x / P) + 0.1 * np.sin(4 * np.pi * x / P)

    def c_mod(m):
        return HydroConductor2D(cz._n2d() * m[None, :], cz.z_m, cz.m_eff_kg, 4.0, 1.2e14, x_m=x,
                                check_validity=False)
    s1 = _solve(_film(c_mod(mod)), theta=30.0, P=P)
    r1 = s1.result.r
    assert abs(s1.energy_residual) < 1e-7
    assert abs(r1 - r_z) > 1e-4
    assert abs(_solve(_film(c_mod(np.roll(mod, 3))), theta=30.0, P=P).result.r - r1) < 1e-11
    assert abs(_solve(_film(c_mod(mod)), theta=-30.0, P=P).result.r - r1) < 1e-11


# ------------------------------------------------------------------------------------------------
# diagnostics, mesh control, refusals
# ------------------------------------------------------------------------------------------------

def test_validity_is_reported_and_warned_not_blocked():
    """The fixture film at 1200 nm has u = |k_L| v_F / omega = 1.06 > 0.5: an
    UnreliableResultWarning names it, the result is still returned, and the report (the planar
    kernel's hydro_validity of the same profile) is in the diagnostics. check_validity=False
    silences the warning, not the report. u is the ratio of the longitudinal wave's k_L v_F to omega;
    the 'high_freq' hydrodynamic expansion is trusted for u <= 0.5 (hydro_validity's u_limit), so this
    film sits OUTSIDE the model's validity on purpose: the warning is what
    this test checks. u = 1 is no threshold of anything (the FEM and the planar route solve the same
    model, so their agreement does not depend on u); the band (1.0, 1.1) only pins the value."""
    with pytest.warns(UnreliableResultWarning, match="validity"):
        sol = _solve(_film(_cond(check_validity=True)))
    d = sol.diagnostics["hydro"]["f"]
    rep = nt.hydro_validity(_omega(LAM), nt.hydro_layers_from_profile(ZP, NP, **KW))
    assert d["validity"].u == rep.u and d["validity"].message == rep.message and not rep.ok
    assert abs(d["u_max"] - max(rep.u)) < 1e-12 and 1.0 < d["u_max"] < 1.1
    assert d["rows"] >= 10 and d["ndof_current"] > 0 and d["ndof_potential"] > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sol2 = _solve(_film(_cond()))
    assert sol2.diagnostics["hydro"]["f"]["validity"].u == rep.u


def test_conductor_z_resolution_is_independent_and_converges():
    """h_z_m sets the conductor's own rows without touching the optical mesh outside it; the
    helper tabulates the z-ladder with the profile samples NOT aligned to mesh lines: |r - r_planar|
    measured 2.4e-6 / 6.5e-7 / 4.0e-7 / 5.1e-8 at 9 / 17 / 34 / 67 rows (the density's kinks inside elements limit the rate; with the samples
    aligned, the default, 10 rows give 1.4e-10 (13 rows))."""
    c = _cond(align_samples=False)
    rows = hydro_z_convergence(100e-9, _film(c), h_factors=(1.5, 0.75, 0.375, 0.1875), orders=(4,),
                               lambda_m=LAM, theta_deg=60.0, n_sub=1.45, polarization="p",
                               num_threads=THREADS)
    nrows = [r["diagnostics"]["hydro"]["f"]["rows"] for r in rows]
    assert nrows == sorted(nrows) and nrows[-1] >= 4 * nrows[0] - 4
    r_ref = _planar_graded()
    err = [abs(r["r"] - r_ref) for r in rows]
    assert err[-1] < 1e-7 and err[-1] < err[0]
    assert rows[-1]["d_last"] == 0.0 and rows[1]["d_prev"] is not None
    # optical mesh outside the conductor unchanged by the conductor's z control
    s1 = _solve(_film(_cond(h_z_m=0.5e-9)))
    s2 = _solve(_film(_cond(h_z_m=0.125e-9)))
    out1 = [z for z in s1.cell.z_lines_nm if not (6.0 < z < 12.0)]
    out2 = [z for z in s2.cell.z_lines_nm if not (6.0 < z < 12.0)]
    assert np.array_equal(out1, out2)
    assert len(s2.cell.z_lines_nm) > len(s1.cell.z_lines_nm)


def test_refusals():
    c = _cond()
    with pytest.raises(NotImplementedError, match="block"):
        _solve([TMLayer("a", 10e-9, 2.0, (TMBlock(0.0, 50e-9, c),)), TMLayer("b", 5e-9, 2.0)])
    with pytest.raises(NotImplementedError, match="blocks inside"):
        _solve([TMLayer("f", T_F, c, (TMBlock(0.0, 50e-9, -10 + 1j),))])
    with pytest.raises(ValueError, match="adjacent"):
        _solve([TMLayer("f1", T_F, c), TMLayer("f2", T_F, c)])
    with pytest.raises(ValueError, match="thick"):
        _solve([TMLayer("f", 7e-9, c)])
    with pytest.raises(TypeError, match="real"):
        HydroConductor2D(NP, ZP, 0.30 * M_E, 4.0 + 0.1j, 1.2e14)
    with pytest.raises(ValueError, match="> 0"):
        HydroConductor2D(np.array([1e26, 0.0]), np.array([0.0, 1e-9]), 0.30 * M_E, 4.0, 1.2e14)
    with pytest.raises(ValueError, match="x_m"):
        HydroConductor2D(np.ones((3, 4)) * 1e26, np.array([0.0, 1e-9, 2e-9]), 0.30 * M_E, 4.0, 1.2e14)
    from dynameta.optics.hydro_fem import HydroParams
    with pytest.raises(NotImplementedError, match="HydroConductor2D"):
        _solve([TMLayer("m", 10e-9, HydroParams(eps_inf=1.0, wp=1.4e16, gamma=1e14, beta=1e6))])


def test_extension_space_tuple_contract_keeps_single_space_extensions():
    """The multi-space extension contract is additive: a single-space extension still receives
    single trial/test/solution objects (the local-current extension of test_fem_2d_tm passes
    unchanged); here a two-space hydrodynamic extension is visible in the DOF count."""
    s_nl = _solve(_film(_cond()))
    s_loc = _solve(_film(_cond(beta_scale=0.0)))
    d = s_nl.diagnostics["hydro"]["f"]
    assert s_nl.ndof > s_loc.ndof + d["ndof_potential"]
    assert s_loc.diagnostics == {}


# ------------------------------------------------------------------------------------------------
# slow ladders (nightly: pytest -m slow)
# ------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_strip_order_and_corner_ladder_converges():
    """The strip cell of test_strip_grating_over_graded_film over (order, corner levels) = (3, 6),
    (4, 6), (5, 8), (6, 10) against (7, 10): measured |r - r_finest| = 1.1e-4 / 6.5e-6 / 9.2e-8 / 6.9e-8 -> monotone; energy
    closes to < 1e-6 on every row (4.3e-7 at order 3, <= 7.6e-9 above)."""
    d = []
    for order, lev in ((3, 6), (4, 6), (5, 8), (6, 10)):
        sol = solve_fem_2d_tm(P_A, _strip_layers(_cond()), lambda_m=LAM, n_super=1.0,
                              n_sub=1.45, order=order, mesh_spec=TMMeshSpec(corner_levels=lev),
                              return_solution=True, num_threads=THREADS)
        d.append(abs(sol.result.r - R_STRIP_CONVERGED))
        assert abs(sol.energy_residual) < 1e-6
    assert all(a > b for a, b in zip(d, d[1:])), d
    assert d[-1] < 1e-6


@pytest.mark.slow
def test_strip_conductor_z_ladder():
    """Refining only the conductor rows (h_z = 1, 0.5, 0.25, 0.125 nm, order 4) moves the strip r
    by <= 2.3e-8 (bar 5e-8): at this fixture the conductor is resolved by its sample-aligned rows and
    the remaining error is lateral (corner) -- the order ladder above."""
    rs = []
    for hz in (1.0e-9, 0.5e-9, 0.25e-9, 0.125e-9):
        sol = solve_fem_2d_tm(P_A, _strip_layers(_cond(h_z_m=hz)), lambda_m=LAM, n_super=1.0,
                              n_sub=1.45, order=4, return_solution=True, num_threads=THREADS)
        rs.append(sol.result.r)
    assert max(abs(r - rs[-1]) for r in rs) < 5e-8


@pytest.mark.slow
def test_local_limit_small_beta_against_planar():
    """beta_scale = 1e-3 (the conductor rows follow 1/|k_L|): |r - r_local| = 1.911e-6 (linear
    continuation of 1.921e-4 at 0.1) and the planar run at the same beta agrees to 2.0e-9."""
    lam = 2500e-9
    r0 = _solve(_film(_cond(beta_scale=0.0)), lam).result.r
    sol = _solve(_film(_cond(beta_scale=1e-3)), lam)
    assert abs(sol.result.r - _planar_graded(lam=lam, beta_scale=1e-3)) < 2e-8
