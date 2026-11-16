"""Gates for the 2-D periodic TM FEM (optics/fem_2d_tm.py).

The solver is new in v0.16 (there is nothing on v0.15.1 to fail: `import dynameta.optics.fem_2d_tm`
raises ModuleNotFoundError there), so each gate's "fails on the old tree" statement is that import.
Every gate below compares against an INDEPENDENT oracle written in this file or taken from a
different method:

  * planar stacks: a TM transfer matrix for uniaxial layers re-derived here (_tm_tmm), and the
    `tmm` library for the r/t sign and reference-plane conventions;
  * the generic metal-strip cell (tests/_strip_cell_fixture): a converged polynomial-modal-method
    (PMM) reference (lumenairy PMMStack, epr 8 / degree 40, R_REF_PMM below, good to ~1e-4) -- a
    modal method that shares no code with the FEM;
  * graded films: the transfer matrix of the same piecewise-linear profile, finely sliced;
  * laterally structured grids: lumenairy RCWA on the same smooth profile (slow) and exact
    symmetry statements (translation of the profile by a lattice-incommensurate shift);
  * the extension hook: a local Drude current carried as an explicit unknown must reproduce
    the local permittivity to round-off (two discretely equivalent formulations).

Error budget for the planar gates (tolerance 1e-7 in r, 1e-6 in t): the PML is sized for a
round-trip attenuation of 1e-10 and order-4 elements at lambda/8 resolve the claddings to
~1e-9; measured |dr| <= 3e-9, |dt| <= 4e-7 (the transmitted amplitude carries the bottom PML and
the substrate probe lines).
"""
import cmath
import math

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.optics.fem_2d_tm import (                                    # noqa: E402
    GriddedEps2D, TMBlock, TMLayer, TMMeshSpec, build_tm_cell, design_to_tm_layers,
    make_fem_2d_tm_optical_solver, solve_design_2d_tm, solve_fem_2d_tm, solve_tm_cell,
    tm_convergence_study)

LAM = 1200e-9
THREADS = 2


def _sq(z):
    w = cmath.sqrt(complex(z))
    return -w if (w.imag < 0 or (w.imag == 0 and w.real < 0)) else w


def _tm_tmm(layers, lam, theta_deg, n1, n2):
    """Independent TM transfer matrix (H_y, uniaxial layers). layers: [(exx, ezz, d_m)] BOTTOM
    first. Returns (r_H, t_H): the H_y reflection at the top interface and the H_y transmitted
    amplitude at the bottom interface, per unit incident H_y at the top interface.
    In a layer kz^2 = exx (k0^2 - kx^2/ezz); (H, G = (1/exx) dH/dz) is carried up by
    [[cos, sin/q], [-q sin, cos]], q = kz/exx."""
    k0 = 2 * math.pi / lam
    e1, e2 = complex(n1) ** 2, complex(n2) ** 2
    kx = complex(n1).real * k0 * math.sin(math.radians(theta_deg))
    q1 = _sq(e1 * k0 ** 2 - kx ** 2) / e1
    q2 = _sq(e2 * k0 ** 2 - kx ** 2) / e2
    v = np.array([1.0, -1j * q2], dtype=complex)
    for exx, ezz, d in layers:
        kz = _sq(exx * (k0 ** 2 - kx ** 2 / ezz))
        q = kz / exx
        c, s = cmath.cos(kz * d), cmath.sin(kz * d)
        v = np.array([[c, s / q], [-q * s, c]]) @ v
    t = 2j * q1 / (1j * q1 * v[0] - v[1])
    return t * v[0] - 1.0, t


def _lay_tuples(layers):
    out = []
    for L in layers:
        e = L.eps
        if isinstance(e, tuple):
            out.append((complex(e[0]), complex(e[1]), L.thickness_m))
        else:
            out.append((complex(e), complex(e), L.thickness_m))
    return out


def _solve(layers, theta=0.0, n1=1.0, n2=1.5, P=300e-9, **kw):
    pol = "x" if theta == 0 else "p"
    return solve_fem_2d_tm(P, layers, lambda_m=LAM, theta_deg=theta, n_super=n1, n_sub=n2,
                           polarization=pol, return_solution=True, num_threads=THREADS, **kw)


THIN_LOSSY = [TMLayer("a", 120e-9, 2.0 + 0j), TMLayer("f", 3e-9, -5 + 4j), TMLayer("b", 80e-9, 4.0)]


# ------------------------------------------------------------------------------------------------
# planar stacks vs the analytic transfer matrix
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.0, 35.0])
def test_planar_thin_lossy_film_matches_transfer_matrix(theta):
    """A 3 nm lossy metal-like film between dielectrics, normal and 35 deg (p): measured
    |dr| 9.7e-10 / 6.9e-10, |dt| 2.5e-7 / 3.8e-7."""
    sol = _solve(THIN_LOSSY, theta)
    rH, tH = _tm_tmm(_lay_tuples(THIN_LOSSY), LAM, theta, 1.0, 1.5)
    res = sol.result
    rH_fem = -res.r if theta == 0 else res.r
    assert abs(rH_fem - rH) < 1e-7
    assert abs(sol.t_orders[0] - tH) < 1e-6
    assert abs(sol.energy_residual) < 1e-6
    assert sol.pml_reflection < 1e-7


def test_oblique_uniaxial_film_is_sign_sensitive_to_the_tensor_orientation():
    """Uniaxial layers at 30 deg: the x-derivative must be weighted by 1/ezz and the z-derivative by
    1/exx. Swapping the two components moves r by far more than the tolerance, so this gate
    distinguishes the correct operator from its transpose."""
    lays = [TMLayer("a", 50e-9, (2.0 + 0.1j, 3.5 + 0.2j)), TMLayer("b", 1e-9, (0.5 + 1j, 0.05 + 0.3j)),
            TMLayer("c", 60e-9, 2.25)]
    sol = _solve(lays, 30.0, n2=1.45)
    rH, _ = _tm_tmm(_lay_tuples(lays), LAM, 30.0, 1.0, 1.45)
    assert abs(sol.result.r - rH) < 1e-7
    swapped = [(ezz, exx, d) for exx, ezz, d in _lay_tuples(lays)]
    rH_sw, _ = _tm_tmm(swapped, LAM, 30.0, 1.0, 1.45)
    assert abs(rH_sw - rH) > 1e-3                      # the wrong operator is far outside tol


def test_reference_plane_and_sign_conventions_match_the_tmm_library():
    """r is referenced to the TOP interface, t to the bottom one, in the p-hat convention for 'p'
    (r_p, t_p of the `tmm` library, as the RCWA/PMM bridges return them) and the lab-E_x one for
    'x' at normal incidence (= r_s = -r_p there)."""
    tmm = pytest.importorskip("tmm")
    n_top_first = [1.0, 2.0, _sq(-5 + 4j), math.sqrt(2.0), 1.5]       # THIN_LOSSY, reversed
    d_top_first = [np.inf, 80.0, 3.0, 120.0, np.inf]
    for theta, pol in ((0.0, "x"), (0.0, "p"), (35.0, "p")):
        res = solve_fem_2d_tm(300e-9, THIN_LOSSY, lambda_m=LAM, theta_deg=theta, n_super=1.0,
                              n_sub=1.5, polarization=pol, num_threads=THREADS)
        ref = tmm.coh_tmm("p", n_top_first, d_top_first, math.radians(theta), LAM * 1e9)
        r_ref = -ref["r"] if pol == "x" else ref["r"]
        assert abs(res.r - r_ref) < 1e-7, (theta, pol)
        assert abs(res.t - ref["t"]) < 1e-6, (theta, pol)
        assert abs(res.R - ref["R"]) < 1e-7 and abs(res.T - ref["T"]) < 1e-6


def test_metal_mirror_and_opaque_metal_substrate():
    """An 80 nm lossy metal mirror on glass (T ~ 2e-5 survives) and a semi-infinite metal
    substrate (opaque: t/T None, its absorption is part of the budget) at 20 deg."""
    efilm = 2.7159 + 0.0982j          # a lossy Drude-film permittivity (the strip fixture's film at 1200 nm)
    mirror = [TMLayer("m", 80e-9, -120 + 20j), TMLayer("o", 6e-9, 4.0), TMLayer("i", 6e-9, efilm),
              TMLayer("u", 6e-9, 4.0)]
    sol = _solve(mirror, 0.0, n2=1.45)
    rH, tH = _tm_tmm(_lay_tuples(mirror), LAM, 0.0, 1.0, 1.45)
    assert abs(-sol.result.r - rH) < 1e-7 and abs(sol.t_orders[0] - tH) < 1e-7      # |t| ~ 5e-3
    assert abs(sol.energy_residual) < 1e-7
    n_m = _sq(-120 + 20j)
    top = [TMLayer("o", 30e-9, 4.0), TMLayer("i", 6e-9, efilm)]
    sol = _solve(top, 20.0, n2=n_m)
    rH, _ = _tm_tmm(_lay_tuples(top), LAM, 20.0, 1.0, n_m)
    res = sol.result
    assert sol.cell.opaque_substrate and res.t is None and res.T is None
    assert abs(res.r - rH) < 1e-7
    assert "substrate" in res.per_region_absorption
    assert abs(res.R + res.A_independent - 1.0) < 1e-7


# ------------------------------------------------------------------------------------------------
# graded (gridded) permittivity
# ------------------------------------------------------------------------------------------------

def _graded_profile():
    z = np.concatenate([np.linspace(0.0, 14e-9, 8), np.linspace(14e-9, 20e-9, 13)[1:]])  # non-uniform
    eps = (2.4 - 3.0 * (z / 20e-9) ** 2) + 1j * (0.05 + 0.6 * (z / 20e-9))
    return z, eps


def _sliced(z, eps, n=4000):
    zc = (np.arange(n) + 0.5) * (z[-1] / n)
    e = np.interp(zc, z, eps.real) + 1j * np.interp(zc, z, eps.imag)
    return [(complex(v), complex(v), z[-1] / n) for v in e]


@pytest.mark.parametrize("flip", [False, True])
def test_z_graded_film_matches_the_transfer_matrix_of_the_same_profile(flip):
    """A 20 nm film whose eps varies piecewise-linearly on NON-uniform z samples (through
    Re eps < 0), oblique p-pol, both orientations. Oracle: the transfer matrix of the same
    piecewise-linear profile cut into 4000 slabs (slicing error ~1e-9). The two orientations
    differ by > 1e-2 in r, so a z flip in the coefficient would fail. Measured |dr| = 1.4e-8 /
    7.4e-9 (slicing 4000 vs 8000: 4e-10)."""
    z, eps = _graded_profile()
    if flip:
        eps = eps[::-1]
    film = TMLayer("film", 20e-9, GriddedEps2D(values=eps, z_m=z))
    lays = [TMLayer("sp", 40e-9, 2.1), film, TMLayer("cap", 10e-9, 3.0)]
    sol = _solve(lays, 25.0, n2=1.45)
    ref = [(2.1 + 0j, 2.1 + 0j, 40e-9)] + _sliced(z, eps) + [(3.0 + 0j, 3.0 + 0j, 10e-9)]
    rH, tH = _tm_tmm(ref, LAM, 25.0, 1.0, 1.45)
    assert abs(sol.result.r - rH) < 2e-7
    assert abs(sol.energy_residual) < 1e-6
    ref_other = [(2.1 + 0j, 2.1 + 0j, 40e-9)] + _sliced(z, eps[::-1]) + [(3.0 + 0j, 3.0 + 0j, 10e-9)]
    assert abs(_tm_tmm(ref_other, LAM, 25.0, 1.0, 1.45)[0] - rH) > 1e-2


def test_laterally_structured_grid_with_identical_columns_matches_the_planar_profile():
    """The (x, z) VoxelCoefficient path (laterally structured branch) fed identical columns must
    reproduce the planar graded profile: an orientation check of the 2-D grid (z must not be
    read as x) against the independent transfer matrix. Measured |dr| = 7.4e-10.
    Exactly identical columns now collapse to the native-z path (core.lateral), so
    the columns carry a 1e-9 relative lateral ripple to keep this test on the structured voxel path it is
    about (the ripple moves r by ~1e-9, far below the 2e-7 bar)."""
    z = np.linspace(0.0, 20e-9, 11)
    eps_z = (2.4 - 3.0 * (z / 20e-9) ** 2) + 1j * (0.05 + 0.6 * (z / 20e-9))
    x = np.linspace(0.0, 300e-9, 8, endpoint=False)
    vals = np.repeat(eps_z[:, None], x.size, axis=1) * (1.0 + 1e-9 * np.cos(2 * np.pi * x / 300e-9))[None, :]
    lays = [TMLayer("sp", 40e-9, 2.1), TMLayer("film", 20e-9, GriddedEps2D(vals, z, x))]
    sol = _solve(lays, 25.0, n2=1.45)
    rH, _ = _tm_tmm([(2.1 + 0j, 2.1 + 0j, 40e-9)] + _sliced(z, eps_z), LAM, 25.0, 1.0, 1.45)
    assert abs(sol.result.r - rH) < 2e-7


def test_laterally_structured_grid_is_translation_invariant():
    """Shifting a band-limited lateral profile by an arbitrary dx (here 0.37 P, not a multiple of the
    sample pitch) and re-sampling it is a translation of the structure: the zeroth-order r cannot
    change at normal incidence. For node-aligned linear interpolation of a band-limited profile the
    discrete problem inherits the invariance (every momentum-conserving product of Fourier
    components carries a unit phase), so it holds to round-off -- measured |dr| = 1.4e-15. A
    clamped (non-periodic) edge or a dropped last cell [x_last, P] breaks it at O(1e-3)."""
    P = 300e-9
    x = np.linspace(0.0, P, 60, endpoint=False)
    z = np.array([0.0, 30e-9])

    def layer(shift):
        e = 2.25 + 1.5 * np.cos(2 * np.pi * (x - shift) / P) + 0.3j * (1 + np.sin(2 * np.pi * (x - shift) / P))
        return [TMLayer("sp", 20e-9, 2.1), TMLayer("g", 30e-9, GriddedEps2D(np.vstack([e, e]), z, x))]

    r0 = _solve(layer(0.0), 0.0, P=P).result.r
    r1 = _solve(layer(0.37 * P), 0.0, P=P).result.r
    assert abs(r1 - r0) < 1e-10
    assert abs(r0 - _solve([TMLayer("sp", 20e-9, 2.1), TMLayer("g", 30e-9, 2.25 + 0.3j)], 0.0, P=P).result.r) > 1e-3


def test_gridded_z_extent_must_match_the_layer():
    z = np.linspace(0.0, 20.0, 5)                       # nm passed as metres: wrong unit
    with pytest.raises(ValueError, match="span"):
        build_tm_cell(300e-9, [TMLayer("f", 20e-9, GriddedEps2D(np.full(5, 2.0 + 0j), z))],
                      lambda_m=LAM)


# ------------------------------------------------------------------------------------------------
# the metal-strip cell vs the modal (PMM) reference
# ------------------------------------------------------------------------------------------------

P_A, W_A, H_A = 400e-9, 150e-9, 40e-9
R_REF_PMM = 0.3352997 + 0.0171262j       # lumenairy 5.50.2 PMMStack, epr 8 / degree 40 (119 s)


def _eps_film():
    from dynameta.materials import M_E, DrudeOptical
    return complex(DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14).eps(LAM, n_m3=3.0e26))


def _strip_layers(film=None):
    film = [TMLayer("film", 6e-9, _eps_film())] if film is None else film
    lays = [TMLayer("mirror", 80e-9, -120 + 20j), TMLayer("lo_oxa", 2e-9, 1.70 ** 2),
            TMLayer("lo_oxb", 6e-9, 2.00 ** 2)] + film + [
        TMLayer("up_oxb", 6e-9, 2.00 ** 2), TMLayer("up_oxa", 2e-9, 1.70 ** 2),
        TMLayer("grating", H_A, 1.0 + 0j, (TMBlock(P_A / 2 - W_A / 2, P_A / 2 + W_A / 2, -60 + 4j, "strip"),))]
    return lays


def _strip(order=4, spec=None, film=None):
    return solve_fem_2d_tm(P_A, _strip_layers(film), lambda_m=LAM, n_super=1.0, n_sub=1.45,
                           order=order, mesh_spec=spec, return_solution=True, num_threads=THREADS)


def _pmm_strip(lum, film, theta_deg=0.0):
    """lumenairy PMMStack (epr 8, degree 32) of the strip cell; film = [(thickness_m, eps)] bottom first."""
    g = (1 - W_A / P_A) / 2
    st = lum.PMMStack(P_A, n_substrate=1.45, n_superstrate=1.0, degree=32, elements_per_region=8)
    st.add_layer(H_A, segments=[(g, 1.0 + 0j), (W_A / P_A, -60 + 4j), (g, 1.0 + 0j)])
    for t, e in [(2e-9, 1.70 ** 2), (6e-9, 2.00 ** 2)] + list(film[::-1]) + [
            (6e-9, 2.00 ** 2), (2e-9, 1.70 ** 2), (80e-9, -120 + 20j)]:
        st.add_layer(t, eps=complex(e))
    st.set_source(LAM, theta=math.radians(theta_deg), phi=0.0)
    _o, _R, _T, J = st.solve()
    return complex(np.asarray(J)[0, 0])


def test_metal_strip_grating_matches_the_converged_modal_reference():
    """The generic strip cell: metal strips (eps -60+4i, 150 nm of 400 nm, 40 nm tall) on a thin
    oxide/film/oxide spacer over a lossy 80 nm mirror metal (eps -120+20i), TM, normal incidence,
    1200 nm. Default mesh (order 4, 6 geometric levels at the metal corners): measured
    |r - r_ref| = 4.3e-5 (r = 0.335282+0.017165i); the FEM's own converged value
    (tests/_strip_cell_fixture.R_FEM_CONVERGED, orders 8-9 and three gradings) is 0.335270+0.017192i,
    7.2e-5 from r_ref -- inside r_ref's own ~1e-4 uncertainty. Budget: 1e-4 (r_ref) + 1e-4 (order-4
    discretization at the corners)."""
    sol = _strip()
    res = sol.result
    assert abs(res.r - R_REF_PMM) < 2e-4
    assert abs(sol.energy_residual) < 1e-6                         # R + T + A = 1
    assert abs(res.R + res.T + res.A_independent - 1.0) < 1e-6
    assert set(res.per_region_absorption) >= {"mirror", "film", "strip"}
    assert res.per_region_absorption["mirror"] > res.per_region_absorption["film"] > 0.0


def test_design_adapter_reproduces_the_direct_layer_call():
    """The same strip cell as a dynameta Design (Rectangle inclusion spanning the y period, film
    eps from eps_by_region) through solve_design_2d_tm: identical layers -> identical r."""
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0), ("glass", 1.45 ** 2), ("oxA", 1.70 ** 2), ("oxB", 2.00 ** 2),
                  ("metal", -60 + 4j), ("mirror_metal", -120 + 20j), ("TCO", 4.0)):
        reg.add(Material(nm, ConstantOptical(complex(e))))
    film = [("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"), ("lo_oxb", 6e-9, "oxB"),
            ("film", 6e-9, "TCO"), ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA")]
    layers = [Layer(n, t, m) for n, t, m in film]
    layers.append(Layer("grating", H_A, "air",
                        inclusions=[Inclusion(Rectangle(P_A / 2, P_A / 2, W_A, P_A), "metal")]))
    d = Design(name="strip", unit_cell=UnitCell(period_x_m=P_A, period_y_m=P_A),
               stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=reg, optical=OpticalSpec(polarization="x"))
    ebr = {"film": EpsField(scalar=_eps_film())}
    P, lays, ns, nb = design_to_tm_layers(d, ebr, LAM)
    assert P == P_A and abs(nb - 1.45) < 1e-12 and [L.name for L in lays][-1] == "grating"
    rd = solve_design_2d_tm(d, ebr, LAM, num_threads=THREADS)
    rl = _strip().result
    assert abs(rd.r - rl.r) < 1e-12
    seam = make_fem_2d_tm_optical_solver(num_threads=THREADS)        # run_pipeline seam signature
    assert "fem_2d_tm" in seam.cache_fingerprint
    assert abs(seam(d, None, ebr, LAM, ns, nb).r - rl.r) < 1e-12


@pytest.mark.slow
def test_metal_strip_mesh_convergence():
    """Mesh convergence of the strip cell in order and corner grading (validation/
    fem_2d_tm_strip.py prints the full table). Measured |r - r_finest| (finest: order 7, 10
    levels): order 3 / 4 / 5 / 6 at 6 levels = 4.3e-4 / 2.7e-5 / 3.9e-6 / 1.3e-6 -> monotone; the finest is 7.0e-5 from
    r_ref (inside r_ref's ~1e-4)."""
    rows = tm_convergence_study(lambda o: _strip(order=o, spec=TMMeshSpec(corner_levels=6 if o < 7 else 10)),
                                [3, 4, 5, 6, 7])
    d = [row["d_last"] for row in rows[:-1]]
    assert all(a > b for a, b in zip(d, d[1:])), d
    assert d[-1] < 1e-5
    assert abs(rows[-1]["r"] - R_REF_PMM) < 1e-4


@pytest.mark.slow
def test_graded_film_under_strips_matches_pmm_with_the_same_slabs():
    """The strip cell with the 6 nm film replaced by 6 graded slabs, solved by the FEM (order 5)
    and by lumenairy's PMM with the same slabs (elements-per-region 8, degree 32; ~100 s):
    agreement to 1e-3. Measured: FEM 0.075338-0.607879i (order 7 / 10 levels moves it by
    1.4e-6), PMM 0.075316-0.607915i, |dr| = 4.3e-5 -- the PMM corner convergence floor at that setting
    (1.2e-4 on the uniform film)."""
    lum = pytest.importorskip("lumenairy")
    from dynameta.materials import M_E, DrudeOptical
    drude = DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)
    # a gate-side accumulation profile: density rising toward the top of the film (bottom first)
    slabs = [complex(drude.eps(LAM, n_m3=n)) for n in (2.7e26, 3.0e26, 3.5e26, 4.2e26, 5.6e26, 7.5e26)]
    film = [TMLayer("film{}".format(k), 1e-9, e) for k, e in enumerate(slabs)]
    r_fem = _strip(order=5, film=film).result.r
    assert abs(r_fem - _pmm_strip(lum, [(1e-9, e) for e in slabs])) < 1e-3


@pytest.mark.slow
def test_oblique_strip_cell_matches_pmm():
    """The strip cell at 20 deg, p-pol (Bloch-periodic FEM vs PMM): FEM order 5 r =
    -0.331327-0.016271i (order 6: -0.331326-0.016274i); PMM epr 8 / degree 32 (~60 s) r_p = -0.331373-0.016169i, 1.1e-4 away
    (epr 6 / degree 24 is 4.1e-4 away). Bar 5e-4 = the PMM floor at this setting plus margin; a
    wrong Bloch phase sign or p-basis sign is O(0.1)."""
    lum = pytest.importorskip("lumenairy")
    r_fem = solve_fem_2d_tm(P_A, _strip_layers(), lambda_m=LAM, theta_deg=20.0, n_super=1.0,
                            n_sub=1.45, polarization="p", order=5, num_threads=THREADS).r
    r_p = -_pmm_strip(lum, [(6e-9, _eps_film())], theta_deg=20.0)   # lab-basis Jones r_xx -> p-hat r_p
    assert abs(r_fem - r_p) < 5e-4


def _rcwa_r_p(lum, P, eps_of_x, theta_deg, n_orders=81, S_=2048):
    """lumenairy RCWA zeroth-order r in the p-hat convention for [spacer 20 nm | graded layer 30 nm]
    on glass; eps_of_x evaluated on lumenairy's node lattice x_j = j P / S."""
    st = lum.RCWAStack(P, n_superstrate=1.0, n_substrate=1.5, n_orders=n_orders)
    # formulation named: lumenairy 5.51's default ('auto') would resolve this 1-D cell to 'li', so the
    # reference would depend on the installed release; 'laurent' is what it was measured with
    st.add_layer(30e-9, eps_cell=eps_of_x(np.arange(S_) * P / S_).reshape(-1, 1),
                 formulation="laurent")
    st.add_layer(20e-9, eps=2.1 + 0j)
    st.set_source(LAM, theta=math.radians(theta_deg), phi=0.0)
    return -complex(np.asarray(st.solve().jones_reflection())[0, 0])   # lab r_xx -> p-hat


def test_smooth_lateral_grid_matches_rcwa():
    """A laterally structured, x-asymmetric smooth grating (cos + sin eps, 64 samples, linear between
    them) in a 30 nm layer at 15 deg (p): FEM vs lumenairy RCWA on the SAME piecewise-linear profile
    (81 orders; a continuous profile converges in the Laurent rule). Measured at normal incidence
    |dr| = 1.2e-8 against the interpolant and 1.3e-5 against the exact cosine (the 64-sample
    interpolation itself). Reciprocity makes the zeroth order blind to an x-mirror of the profile
    (r_0(theta) of the mirror = r_0(-theta) of the original = r_0(theta)), so the mirrored profile
    must give the same r in both solvers -- checked as an exact-symmetry gate on the FEM."""
    lum = pytest.importorskip("lumenairy")
    P = 300e-9
    x = np.linspace(0.0, P, 64, endpoint=False)
    e = 2.25 + 1.5 * np.cos(2 * np.pi * x / P) + 0.3j * (1 + np.sin(2 * np.pi * x / P))
    lays = [TMLayer("sp", 20e-9, 2.1), TMLayer("g", 30e-9, GriddedEps2D(np.vstack([e, e]), [0.0, 30e-9], x))]
    r_fem = _solve(lays, 15.0, P=P, order=5).result.r
    xe, ee = np.concatenate([x, [P]]), np.concatenate([e, e[:1]])

    def interp(xs):
        return np.interp(xs, xe, ee.real) + 1j * np.interp(xs, xe, ee.imag)

    r_rc = _rcwa_r_p(lum, P, interp, 15.0)
    assert abs(r_fem - r_rc) < 1e-6
    em = interp((P - x) % P)
    lays_m = [TMLayer("sp", 20e-9, 2.1), TMLayer("g", 30e-9, GriddedEps2D(np.vstack([em, em]), [0.0, 30e-9], x))]
    assert abs(_solve(lays_m, 15.0, P=P, order=5).result.r - r_fem) < 1e-6


# ------------------------------------------------------------------------------------------------
# extension hook (the seam a nonlocal subdomain field will use)
# ------------------------------------------------------------------------------------------------

class _LocalDrudeCurrent:
    """A local Drude polarisation current J_p carried as an explicit unknown on one region:
    E = eps_b^-1 (J grad H - J_p) / (-i omega eps0) and [wp2/eps_b - (1 + i g)] J_p = (wp2/eps_b) J grad H
    (frequencies in units of omega). Eliminating J_p gives eps = eps_b - wp2 / (1 + i g) exactly, so
    the coupled solve must equal the local-eps solve to round-off."""

    def __init__(self, region, eps_b, wp2, g):
        self.region, self.eps_b, self.c, self.om = region, eps_b, wp2 / eps_b, 1.0 + 1j * g

    def space(self, ctx):
        mesh = ctx.cell.mesh
        return ctx.ng.VectorL2(mesh, order=ctx.order, complex=True,
                               definedon=mesh.Materials(self.region))

    def add_terms(self, ctx, a, f, h, hv, x, xv):
        ng, dxr = ctx.ng, ctx.dx(self.region)
        gv, gh = ng.grad(hv), ng.grad(h)
        jgh, jgb = ctx.inplane_curl(gh), ctx.inplane_curl(ctx.grad_H_bg)
        a += ((1.0 / self.eps_b) * (x[0] * gv[1] - x[1] * gv[0])) * dxr
        a += ((self.c - self.om) * (x[0] * xv[0] + x[1] * xv[1])
              - self.c * (jgh[0] * xv[0] + jgh[1] * xv[1])) * dxr
        f += (self.c * (jgb[0] * xv[0] + jgb[1] * xv[1])) * dxr

    def absorption(self, ctx, gfx, g_tot):
        ng = ctx.ng
        jg = ctx.inplane_curl(g_tot)
        E = ((jg[0] - gfx[0]) / self.eps_b, (jg[1] - gfx[1]) / self.eps_b)
        q = ng.Conj(gfx[0]) * E[0] + ng.Conj(gfx[1]) * E[1]
        val = ng.Integrate((q - ng.Conj(q)) / 2j, ctx.cell.mesh,
                           definedon=ctx.cell.mesh.Materials(self.region), order=2 * ctx.order + 2)
        e1 = complex(ctx.cell.n_super) ** 2
        kz0 = cmath.sqrt(e1 * ctx.k0 ** 2 - ctx.kx ** 2)
        return {self.region: -complex(val).real / (ctx.cell.period_nm * (kz0 / e1).real)}


def test_extension_hook_reproduces_the_local_permittivity():
    """Strip of a Drude conductor (eps_b = 1, wp^2/omega^2 = 60, gamma/omega = 0.08) on a spacer over
    glass at 20 deg: the explicit-current extension equals the local-eps solve on the same mesh
    (measured |dr| = 6.8e-13, |dA| = 6e-15) and closes the energy budget through its own absorption
    hook (residual 3.6e-10)."""
    eps_b, wp2, g = 1.0, 60.0, 0.08
    eps_d = eps_b - wp2 / (1 + 1j * g)

    def lays(eps_strip):
        return [TMLayer("sp", 30e-9, 2.1),
                TMLayer("gr", 40e-9, 1.0, (TMBlock(100e-9, 220e-9, eps_strip, "cond"),))]

    loc = solve_fem_2d_tm(320e-9, lays(eps_d), lambda_m=LAM, theta_deg=20.0, n_sub=1.5,
                          polarization="p", return_solution=True, num_threads=THREADS)
    # one explicit mesh for both runs (the automatic sizing reads eps, which differs between them)
    spec = TMMeshSpec(corner_mode="all", maxh_m=20e-9, metal_h_per_skin=1e9)
    loc2 = solve_fem_2d_tm(320e-9, lays(eps_d), lambda_m=LAM, theta_deg=20.0, n_sub=1.5,
                           polarization="p", mesh_spec=spec, return_solution=True, num_threads=THREADS)
    ext = solve_fem_2d_tm(320e-9, lays(eps_b), lambda_m=LAM, theta_deg=20.0, n_sub=1.5,
                          polarization="p", mesh_spec=spec, return_solution=True, num_threads=THREADS,
                          extensions=[_LocalDrudeCurrent("cond", eps_b, wp2, g)])
    assert loc2.n_elements == ext.n_elements
    assert abs(ext.result.r - loc2.result.r) < 1e-10
    assert abs(ext.result.per_region_absorption["cond"] - loc2.result.per_region_absorption["cond"]) < 1e-9
    assert abs(ext.energy_residual) < 1e-6
    assert abs(loc.result.r - loc2.result.r) < 1e-4                    # mesh-level agreement only


# ------------------------------------------------------------------------------------------------
# refusals
# ------------------------------------------------------------------------------------------------

def test_hydrodynamic_input_is_refused_naming_the_gap():
    from dynameta.optics.hydro_fem import HydroParams
    hp = HydroParams(eps_inf=1.0, wp=1.4e16, gamma=1e14, beta=1e6)
    with pytest.raises(NotImplementedError, match="nonlocal"):
        solve_fem_2d_tm(300e-9, [TMLayer("m", 10e-9, hp)], lambda_m=LAM)
    with pytest.raises(NotImplementedError, match="hydrodynamic"):
        solve_fem_2d_tm(300e-9, [TMLayer("a", 10e-9, 2.0, (TMBlock(0.0, 1e-7, hp),))], lambda_m=LAM)


def test_unsupported_incidence_is_refused():
    lay = [TMLayer("a", 10e-9, 2.0)]
    with pytest.raises(NotImplementedError, match="polarization='p'"):
        solve_fem_2d_tm(300e-9, lay, lambda_m=LAM, theta_deg=10.0, polarization="x")
    with pytest.raises(NotImplementedError, match="TE"):
        solve_fem_2d_tm(300e-9, lay, lambda_m=LAM, polarization="y")
    with pytest.raises(NotImplementedError, match="LOSSLESS"):
        solve_fem_2d_tm(300e-9, lay, lambda_m=LAM, n_super=1.0 + 0.1j)
    with pytest.raises(ValueError, match="overlap"):
        build_tm_cell(300e-9, [TMLayer("a", 10e-9, 2.0, (TMBlock(0, 1e-7, 4.0), TMBlock(5e-8, 2e-7, 4.0)))],
                      lambda_m=LAM)


def test_block_wrapping_across_the_cell_edge_is_a_translation():
    """A strip crossing x = 0 is the same periodic structure as the centred one shifted by a
    lattice-incommensurate dx; the zeroth-order r (normal incidence) cannot change. The wrapped
    block becomes two mesh pieces of one region, so its corners and absorption must survive.
    Measured |dr| = 3.1e-8 (the two meshes differ)."""
    def lays(lo):
        return [TMLayer("sp", 20e-9, 2.1),
                TMLayer("gr", 40e-9, 1.0, (TMBlock(lo, lo + 120e-9, -40 + 3j, "s"),))]
    a = _solve(lays(90e-9), 0.0, P=300e-9)
    b = _solve(lays(-50e-9), 0.0, P=300e-9)
    assert len(b.cell.corners_nm) == len(a.cell.corners_nm)
    assert abs(a.result.r - b.result.r) < 1e-6
    assert abs(a.result.per_region_absorption["s"] - b.result.per_region_absorption["s"]) < 1e-6


def test_diffracting_lossless_grating_closes_energy_over_all_orders():
    """Period 1.5 um at 1.2 um, 5 deg: three reflected and three transmitted orders propagate. For a
    lossless cell R + T = 1 holds only for the ORDER-SUMMED efficiencies (the specular pair carries
    0.24), so this gates the per-order projection, its kz weighting and the order bookkeeping.
    Measured |R + T - 1| = 2.0e-9. (The buffer rule this relies on near a cutoff is measured in
    fem_2d_tm._auto_buffer: capping it moved a 1.28 um cell's R + T by 1.1e-2.)"""
    lays = [TMLayer("sp", 100e-9, 2.1),
            TMLayer("g", 200e-9, 1.0, (TMBlock(0.0, 660e-9, 3.4 ** 2, "rib"),))]
    sol = solve_fem_2d_tm(1500e-9, lays, lambda_m=LAM, theta_deg=5.0, n_sub=1.45,
                          polarization="p", return_solution=True, num_threads=THREADS)
    res = sol.result
    assert len(sol.R_orders) == 3 and len(sol.T_orders) == 3
    assert abs(res.R + res.T - 1.0) < 1e-6
    assert abs(res.A_independent) < 1e-12
    assert sol.R_orders[0] + sol.T_orders[0] < 0.5


# ------------------------------------------------------------------------------------------------
# grating-coupled total internal reflection: order 0 evanescent in the substrate, diffracted orders not
# ------------------------------------------------------------------------------------------------

# n_super 1.5 over an air substrate at 60 deg (kx = 1.299 k0 > k0: order 0 is past total internal
# reflection), P = lambda / 0.9: substrate orders -1 and -2 propagate. A 250 nm grating, eps 4 blocks
# (0.4 P) in eps 2.25. PMM reference (lumenairy PMMStack, epr 8 / degree 40; degree 30 / 50 / 60 and epr
# 12 agree to <= 1.1e-9 in r and <= 1e-10 in T), p-hat convention:
_TIR_LAM = 1200e-9
_TIR_P = _TIR_LAM / 0.9
_TIR_R_PMM = 0.750451329 + 0.577934555j
_TIR_T_PMM = {-1: 0.026461008, -2: 0.001769045}            # T = 0.028230053, R = 0.971769947
# MEASURED convergence of the FEM default (order 4, default TMMeshSpec: the eps 4 / 2.25 corners have
# contrast 1.78 < corner_contrast, ungraded) against order 7 / 'all' / 8 levels (itself 5e-10 from the
# PMM reference): |dr| = 6.0e-5 / 2.1e-5 / 9.8e-6 / 5.2e-6 at orders 3 / 4 / 5 / 6 (algebraic: the
# ungraded corners), |dT| = 7.6e-7 and |dT_-1| = 8.5e-7 at order 4; with the corners graded ('all')
# order 4 is at 6.4e-9. Bars: 5e-5 in r and 2e-6 in each T (about 2.4x the measured order-4 error).
_TIR_BAR_R, _TIR_BAR_T = 5e-5, 2e-6


def _tir_solve(P=_TIR_P, n_sub=1.0, **kw):
    W = 0.4 * P
    lays = [TMLayer("g", 250e-9, 2.25 + 0j, (TMBlock(P / 2 - W / 2, P / 2 + W / 2, 4.0 + 0j, "blk"),))]
    return solve_fem_2d_tm(P, lays, lambda_m=_TIR_LAM, theta_deg=60.0, n_super=1.5, n_sub=n_sub,
                           polarization="p", return_solution=True, num_threads=THREADS, **kw)


def test_grating_coupled_tir_keeps_the_diffracted_transmission():
    """Order 0 is evanescent in the substrate (decay length 230 nm < lambda), but orders -1 / -2
    propagate into it: the substrate is NOT opaque. r, T and each transmitted order match the PMM
    reference, R + T = 1 closes, and the lossless structure reports no absorption. Judging the
    substrate from order 0 alone called it opaque: T None, the transmitted 0.0282 reported as
    absorption (A = 1 - R), r off by 7.2e-4 (measured on the pre-fix tree)."""
    sol = _tir_solve()
    res = sol.result
    assert not sol.cell.opaque_substrate
    assert abs(res.r - _TIR_R_PMM) < _TIR_BAR_R
    assert sorted(sol.T_orders) == [-2, -1, 0] and sol.T_orders[0] == 0.0       # order 0 carries no flux
    for m, t_ref in _TIR_T_PMM.items():
        assert abs(sol.T_orders[m] - t_ref) < _TIR_BAR_T, m
    assert abs(res.T - sum(_TIR_T_PMM.values())) < _TIR_BAR_T
    # energy: R + T = 1 to the solver's lossless closure (measured residual 3.4e-8; bar as the other
    # lossless gates), and no absorption anywhere (A_independent is an integral of Im eps = 0: round-off)
    assert abs(sol.energy_residual) < 1e-6
    assert abs(res.A_independent) < 1e-12
    assert abs(res.A) < 1e-6
    assert "substrate" not in res.per_region_absorption


def test_grating_coupled_tir_into_a_weakly_lossy_substrate():
    """The same cell over n_sub = 1 + 1e-3j: order 0 still decays within lambda, so the order-0 rule called
    this substrate opaque too; it is transparent, and T is the full interface flux (_lossy_substrate_flux).
    PMM (epr 8 / degree 40; degree 20 / 30 agree to 5e-9 in r, 4e-10 per order): r = 0.748191549 +
    0.576204178i, T_-1 = 0.026478690, T_-2 = 0.001765966. The FEM default's measured errors against its
    own order-7 / 'all' value equal the lossless case's (2.1e-5 in r, 8.5e-7 in T_-1), hence the same bars.
    PMM's order-summed T is 4.5e-7 below the FEM's (PMM's R + T = 1 - 4.5e-7 for this lossless grating:
    the evanescent substrate orders' interface flux in a lossy substrate is outside its sum), so the
    closure, not PMM's total, gates the FEM's T."""
    sol = _tir_solve(n_sub=1.0 + 1e-3j)
    res = sol.result
    assert not sol.cell.opaque_substrate
    assert abs(res.r - (0.748191549 + 0.576204178j)) < _TIR_BAR_R
    assert abs(sol.T_orders[-1] - 0.026478690) < _TIR_BAR_T
    assert abs(sol.T_orders[-2] - 0.001765966) < _TIR_BAR_T
    assert abs(sol.energy_residual) < 1e-6 and abs(res.A_independent) < 1e-12     # measured 3.4e-8


def test_total_internal_reflection_without_diffracted_orders_stays_opaque():
    """The same grating at P = lambda / 3: every substrate order is evanescent and the slowest (order 0)
    decays within lambda, so the substrate is opaque (t / T None) and the lossless cell reflects
    everything: R = 1 to the closure of the solver."""
    sol = _tir_solve(P=_TIR_LAM / 3.0)
    res = sol.result
    assert sol.cell.opaque_substrate and res.T is None and res.t is None
    assert abs(res.R - 1.0) < 1e-6
    assert abs(res.A_independent) < 1e-12


# ------------------------------------------------------------------------------------------------
# automatic cladding buffer: evanescent orders of a LOSSY cladding (fem_2d_tm._auto_buffer)
# ------------------------------------------------------------------------------------------------

# a 1.28 um period rib grating (528 nm wide, 160 nm tall, eps 11.56, in air) at 1.04 um, 10 deg, p:
# lambda / P = 0.8125 puts substrate order -2 just past its cut-off (decay length 2.64 um in a
# lossless n 1.45 substrate), the case _auto_buffer's docstring measures
_BUF_LAM, _BUF_P, _BUF_TH = 1040e-9, 1280e-9, 10.0
_BUF_LAYERS = [TMLayer("g", 160e-9, 1.0, (TMBlock(0.0, 528e-9, 3.4 ** 2, "rib"),))]


def _buf_args(n_sub):
    """_auto_buffer's arguments for the substrate of the buffer case (nm units, as build_tm_cell)."""
    lam_nm, p_nm = _BUF_LAM * 1e9, _BUF_P * 1e9
    k0 = 2.0 * math.pi / lam_nm
    n_sub = complex(n_sub)
    return n_sub ** 2, k0, k0 * math.sin(math.radians(_BUF_TH)), p_nm, lam_nm, max(abs(n_sub.real), 1e-3)


def _old_auto_buffer(eps, k0, kx, P, lam0_nm, n):
    """The rule before the lossy-cladding fix (an order is evanescent when |Re kz| < 1e-12 k0), kept
    here as the lossless oracle."""
    from dynameta.optics.resonance import _kz
    n = max(n, 1e-6)
    buf, kap = 0.5 * lam0_nm / n, None
    for m in range(-200, 201):
        kz = _kz(eps, k0, kx + 2 * math.pi * m / P)
        if abs(kz.real) < 1e-12 * k0 and kz.imag > 0:
            kap = kz.imag if kap is None else min(kap, kz.imag)
    if kap is not None:
        buf = max(buf, min(8.0 / kap, 30.0 * lam0_nm / n))
    return buf


def _buf_solve(n_sub, buffer_m=None):
    # corner_mode 'metal' (the rib's dielectric corners ungraded): these gates compare buffers at one
    # mesh rule, and on the tensor-product mesh every graded x line runs through the whole 20-40 um
    # substrate buffer of this near-cut-off case -- grading the rib corners raises the element count
    # 1.8x (19754 -> 35090 at the automatic buffer) and the 2x-buffer solve past 11 GB
    return solve_fem_2d_tm(_BUF_P, _BUF_LAYERS, lambda_m=_BUF_LAM, theta_deg=_BUF_TH, n_super=1.0,
                           n_sub=n_sub, polarization="p", order=4, return_solution=True,
                           num_threads=THREADS,
                           mesh_spec=TMMeshSpec(corner_mode="metal", substrate_buffer_m=buffer_m))


def test_lossy_substrate_transmission_counts_every_order():
    """A LOSSY transparent substrate under the diffracting grating above (three propagating substrate
    orders, -1 / 0 / +1, and the near-cut-off order -2): T is the power entering the substrate at the
    interface over ALL orders, so R + T + A closes to the solver's lossless tolerance. Measured
    residual 2.7e-8 at n_sub = 1.45 + 1e-3j (lossless: 2.4e-8). Before, T summed order 0 only: residual
    0.46 and an UnreliableResultWarning (an error under this suite's warning filter). The per-order
    list still names the orders above the lossless cut-off, and in a lossy substrate its sum is below T
    (0.8447 against 0.8544: the rest is absorbed near the interface by the evanescent orders).
    Lossless limit: T is continuous -- |T - T_lossless| = 1.2e-11 at Im(n) = 1e-12, linear in the loss
    (1.2e-8 at 1e-9) -- and the lossless T is the per-order sum, unchanged."""
    lossless = _buf_solve(1.45)
    assert sorted(lossless.T_orders) == [-1, 0, 1]
    assert lossless.result.T == sum(lossless.T_orders.values())
    lossy = _buf_solve(1.45 + 1e-3j)
    assert sorted(lossy.T_orders) == [-1, 0, 1]
    assert abs(lossy.energy_residual) < 1e-6, lossy.energy_residual
    assert abs(lossy.result.R + lossy.result.T + lossy.result.A_independent - 1.0) < 1e-6
    assert sum(lossy.T_orders.values()) < lossy.result.T - 1e-3
    tiny = _buf_solve(1.45 + 1e-12j)
    assert abs(tiny.result.T - lossless.result.T) < 1e-9
    assert abs(tiny.energy_residual - lossless.energy_residual) < 1e-9


def test_auto_buffer_lossless_cladding_is_unchanged(monkeypatch):
    """Lossless claddings: the new classification (Re(kz^2) <= 0) selects exactly the orders the old
    one (|Re kz| < 1e-12 k0) did, so the buffer is bit-identical over a range of periods, angles and
    indices (incl. the near-cut-off case and a non-diffracting one), and so is r of the grating case
    with the OLD rule patched back in (same mesh, same solve)."""
    import warnings
    from dynameta.optics import fem_2d_tm as F
    from dynameta.warning_types import UnreliableResultWarning
    for n_c in (1.0, 1.45, 2.3):
        for p_nm in (240.0, 720.0, 1280.0, 2160.0):
            for th in (0.0, 10.0, 37.0):
                k0 = 2.0 * math.pi / 1040.0
                args = (complex(n_c) ** 2, k0, k0 * math.sin(math.radians(th)), p_nm, 1040.0, n_c)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UnreliableResultWarning)   # a capped near-cut-off case
                    assert F._auto_buffer(*args) == _old_auto_buffer(*args), (n_c, p_nm, th)
    new = _buf_solve(1.45)
    monkeypatch.setattr(F, "_auto_buffer", _old_auto_buffer)
    old = _buf_solve(1.45)
    assert new.cell.z_intervals_nm == old.cell.z_intervals_nm
    assert new.result.r == old.result.r                               # measured: bit for bit


def test_auto_buffer_weak_loss_is_continuous():
    """n_sub = 1.45 + 1e-6j: the buffer is within 1e-3 relative of the lossless one (measured 6.9e-8:
    21143.82 nm against 21143.82 nm); the old rule gave lambda/(2n) = 359 nm there."""
    from dynameta.optics import fem_2d_tm as F
    b0 = F._auto_buffer(*_buf_args(1.45))
    b1 = F._auto_buffer(*_buf_args(1.45 + 1e-6j))
    assert abs(b1 - b0) <= 1e-3 * b0, (b0, b1)
    assert _old_auto_buffer(*_buf_args(1.45 + 1e-6j)) == pytest.approx(0.5 * 1040.0 / 1.45, rel=1e-12)


def test_auto_buffer_lossy_cladding_reaches_the_converged_r():
    """n_sub = 1.45 + 1e-3j (decay length of the near-cut-off order ~2.5 um): r with the automatic buffer
    (19.96 um) against a 2x longer buffer: measured 2.8e-8 (4x: 4.2e-8); budget 1e-6 (the lossless
    case's own 2x / 4x buffer spread is 7.1e-10, its PML / discretization floor ~1e-8). The OLD rule's
    359 nm buffer misses the 4x reference by 7.3e-3 (an order that the PML does not attenuate
    reflects off the outer wall)."""
    from dynameta.optics import fem_2d_tm as F
    n_sub = 1.45 + 1e-3j
    b_new = F._auto_buffer(*_buf_args(n_sub))
    auto = _buf_solve(n_sub)
    assert auto.cell.z_intervals_nm["substrate"][1] - auto.cell.z_intervals_nm["substrate"][0] == \
        pytest.approx(b_new, rel=1e-12)
    ref = _buf_solve(n_sub, buffer_m=2.0 * b_new * 1e-9).result.r
    assert abs(auto.result.r - ref) < 1e-6, abs(auto.result.r - ref)
    import warnings
    from dynameta.warning_types import UnreliableResultWarning
    with warnings.catch_warnings():
        # the too-short buffer is the point here: its reflected evanescent order also breaks the energy
        # balance (measured residual -1.3e-2), which the solver reports
        warnings.simplefilter("ignore", UnreliableResultWarning)
        old = _buf_solve(n_sub, buffer_m=_old_auto_buffer(*_buf_args(n_sub)) * 1e-9).result.r
    assert abs(old - ref) > 1e-3, abs(old - ref)                    # discrimination


def test_auto_buffer_strongly_lossy_cladding_stays_short():
    """n_sub = 1.45 + 0.1j: every near-cut-off order now decays within ~0.4 um, and the buffer is
    8 decay lengths of the slowest one (1/Im kz with the full complex kz, an independent sum here):
    3.39 um, a sixth of the lossless 21.1 um. r agrees with a 2x longer buffer (measured 1.1e-11)."""
    from dynameta.optics import fem_2d_tm as F
    from dynameta.optics.resonance import _kz
    eps, k0, kx, p_nm, lam_nm, n = _buf_args(1.45 + 0.1j)
    kap = min(_kz(eps, k0, kx + 2 * math.pi * m / p_nm).imag for m in range(-60, 61)
              if eps.real * k0 ** 2 - (kx + 2 * math.pi * m / p_nm) ** 2 <= 0)
    b = F._auto_buffer(eps, k0, kx, p_nm, lam_nm, n)
    assert b == pytest.approx(max(0.5 * lam_nm / n, 8.0 / kap), rel=1e-12)
    assert b < 0.2 * F._auto_buffer(*_buf_args(1.45))
    r1 = _buf_solve(1.45 + 0.1j).result.r
    r2 = _buf_solve(1.45 + 0.1j, buffer_m=2.0 * b * 1e-9).result.r
    assert abs(r1 - r2) < 1e-6


def test_cell_build_and_solve_split_is_reusable():
    cell = build_tm_cell(300e-9, THIN_LOSSY, lambda_m=LAM, n_sub=1.5)
    s1 = solve_tm_cell(cell, order=4, num_threads=THREADS)
    s2 = solve_tm_cell(cell, polarization="p", order=4, num_threads=THREADS)
    assert abs(s1.result.r + s2.result.r) < 1e-14                     # 'x' = -'p' at normal incidence
    assert s1.ndof == s2.ndof and s1.cell is cell
