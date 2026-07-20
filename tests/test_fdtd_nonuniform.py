"""Gates for roadmap 5.1 -- the NONUNIFORM-z 1-D FDTD grid (nm gaps / accumulation layers resolved
in wavelength-scale domains without a globally fine uniform mesh). Convention exp(-i omega t), SI.

Formulation under test: the standard nonuniform Yee scheme (Taflove & Hagness, Computational
Electrodynamics, ch. 3; Monk-Suli nonuniform-grid analysis). E (Ex) lives on the PRIMAL nonuniform
grid (E-node j at z_edges[j]); H (Hy) lives on the DUAL grid (primal-cell centers). Each spatial
derivative uses its LOCAL spacing -- the PRIMAL spacing dz_primal[j] = z_edges[j+1]-z_edges[j] for the
H update, and the DUAL spacing dz_dual[j] = 0.5(dz_primal[j-1]+dz_primal[j]) for the E update. A
single global dt is bounded by the SMALLEST cell (Courant S = c dt / min(dz) <= 1). 2nd-order accurate
on smoothly graded meshes; drops to 1st order at abrupt spacing jumps (hence the geometric grading in
make_graded_z, ratio <= ~1.15/cell).

Dual-grid coefficient statement (the one load-bearing fact): the Drude ADE (aJ, bJ) and Kerr eps_eff
coefficients are dz-INDEPENDENT and carry over unchanged from the uniform kernel; ONLY the two
spatial-derivative denominators become per-cell arrays (primal drives H, dual drives the E curl). That
is why the ADE composes on the graded mesh with no re-derivation (gate 6).

The gates:
  1 UNIFORM-LIMIT BYTE-IDENTITY: z_edges = the uniform grid's own edges reproduces the default path
    bit-for-bit (tobytes incl. NaN payloads); the pre-existing FDTD gates still pass.
  2 ACCURACY: a lossy (Drude) film's R/T on a graded mesh (coarse far-field + fine film) matches the
    uniform-FINE reference to < 0.1% at ~8x fewer cells.
  3 CONVERGENCE ORDER: R error vs max-dz on smoothly graded meshes fits slope ~2 (>= 1.7).
  4 nm-GAP: a 3-nm high-index gap between two thick dielectrics, graded solve vs a brute-force
    uniform-at-gap-resolution reference, < 0.5% in R at ~10x fewer cells.
  5 STABILITY: a 50k-step run at the documented Courant bound stays bounded; dt above the bound is
    caught by a guard (raise) rather than exploding.
  6 DRUDE ADE on a graded mesh: the Drude slab absorption gate (mirroring the infra test) passes.
"""

import numpy as np
import pytest

from dynameta.constants import C_LIGHT
from dynameta.optics.fdtd import (solve_fdtd_1d, FDTDLayer, make_graded_z, uniform_z_edges,
                                  _run_nu, _refined_full_edges, _grid_metrics)


# ============================================================================================
# helpers
# ============================================================================================

def _graded_cellcount(layers, lmin, lmax, res, npw, refine):
    dz, pad, z_struct, Lz, nz, n_max, f_min, f_max = _grid_metrics(layers, lmin, lmax, res, npw)
    return _refined_full_edges(layers, dz, pad, z_struct, refine, 8).size


def _band_max_absdiff(r_test, r_ref, field):
    """max |field_test - field_ref| over the reference's trustworthy band (test interpolated on)."""
    b = r_ref.band
    ft = np.interp(r_ref.freqs_Hz, r_test.freqs_Hz, getattr(r_test, field))
    return float(np.max(np.abs(ft[b] - getattr(r_ref, field)[b])))


# ============================================================================================
# GATE 1 -- UNIFORM-LIMIT BYTE-IDENTITY
# ============================================================================================

def _byte_cases():
    return [
        ([FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0)],
         dict(lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30)),
        ([FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0, drude_wp_rad_s=6.0e14, drude_gamma_rad_s=5.0e13)],
         dict(lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30)),
        ([FDTDLayer(thickness_m=0.20e-6, eps_inf=2.25), FDTDLayer(thickness_m=0.10e-6, eps_inf=6.0)],
         dict(lambda_min_m=1.0e-6, lambda_max_m=1.5e-6, resolution=24)),
    ]


def test_gate1_uniform_edges_byte_identical():
    for layers, kw in _byte_cases():
        r_base = solve_fdtd_1d(layers, **kw)                          # default (legacy scalar) path
        ze = uniform_z_edges(layers, lambda_min_m=kw["lambda_min_m"], lambda_max_m=kw["lambda_max_m"],
                             resolution=kw["resolution"])
        r_nu = solve_fdtd_1d(layers, **kw, z_edges=ze)                # the same grid via z_edges=
        # .tobytes() is the literal bit-for-bit identity (same NaN payloads at the DFT-zero bins)
        assert r_nu.R.tobytes() == r_base.R.tobytes()
        assert r_nu.T.tobytes() == r_base.T.tobytes()
        assert r_nu.freqs_Hz.tobytes() == r_base.freqs_Hz.tobytes()
        assert np.array_equal(r_nu.band, r_base.band)
        assert np.array_equal(r_nu.R, r_base.R, equal_nan=True)
        assert np.array_equal(r_nu.T, r_base.T, equal_nan=True)


def test_gate1_first_spacing_is_exact_dz():
    # the uniform-limit detection hinges on the FIRST np.diff of uniform_z_edges being EXACTLY dz
    layers = [FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0)]
    dz, pad, z_struct, Lz, nz, n_max, f_min, f_max = _grid_metrics(layers, 1.2e-6, 1.45e-6, 30, 6.0)
    ze = uniform_z_edges(layers, lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30)
    assert ze.size == nz
    assert float(np.diff(ze)[0]) == dz            # bit-exact (arange(nz)[1]*dz - 0)


def test_gate1_existing_fdtd_infra_gates_still_pass():
    # the additive nonuniform path must not perturb the pre-existing coverage gates
    from test_audit_2026_07_17_infra import (test_fdtd_1d_dielectric_slab_vs_airy,
                                             test_fdtd_1d_drude_slab_absorbs)
    test_fdtd_1d_dielectric_slab_vs_airy()
    test_fdtd_1d_drude_slab_absorbs()


def test_gate1_time_trace_on_nonuniform_grid():
    # return_time_trace (the minimum-supported combination) works on the nonuniform grid
    layers = [FDTDLayer(thickness_m=0.10e-6, eps_inf=4.0)]
    r = solve_fdtd_1d(layers, lambda_min_m=1.2e-6, lambda_max_m=1.6e-6, resolution=20, n_pad_wave=1.5,
                      refine={0: 4}, return_time_trace=True)
    tt = r.time_trace
    assert tt is not None
    for key in ("dt", "t", "reflected", "transmitted", "incident_left", "incident_right"):
        assert key in tt
    assert tt["reflected"].shape == tt["t"].shape and tt["dt"] > 0.0


# ============================================================================================
# GATE 2 -- ACCURACY: graded (coarse far-field + fine film) vs uniform-FINE, < 0.1% at fewer cells
# ============================================================================================

def test_gate2_lossy_film_graded_matches_uniform_fine():
    lmin, lmax, npw = 1.0e-6, 1.5e-6, 1.5
    res_base, factor = 30, 10
    film = [FDTDLayer(thickness_m=0.05e-6, eps_inf=2.0, drude_wp_rad_s=2.5e15, drude_gamma_rad_s=1.5e14)]
    r_graded = solve_fdtd_1d(film, lambda_min_m=lmin, lambda_max_m=lmax, resolution=res_base,
                             n_pad_wave=npw, refine={0: factor})
    r_fine = solve_fdtd_1d(film, lambda_min_m=lmin, lambda_max_m=lmax, resolution=res_base * factor,
                           n_pad_wave=npw)
    dR = _band_max_absdiff(r_graded, r_fine, "R")
    dT = _band_max_absdiff(r_graded, r_fine, "T")
    n_graded = _graded_cellcount(film, lmin, lmax, res_base, npw, {0: factor})
    _, _, _, _, n_fine, _, _, _ = _grid_metrics(film, lmin, lmax, res_base * factor, npw)
    # the film really is lossy over the band (a meaningful, not trivial, R/T)
    A = 1.0 - r_fine.R[r_fine.band] - r_fine.T[r_fine.band]
    assert A.min() > 0.02
    assert dR < 1e-3, dR                                  # < 0.1% in R
    assert dT < 1e-3, dT                                  # < 0.1% in T
    assert n_fine / n_graded >= 5.0, (n_graded, n_fine)   # ~8x fewer cells (report: see final message)


# ============================================================================================
# GATE 3 -- CONVERGENCE ORDER ~2 on smoothly graded meshes
# ============================================================================================

def test_gate3_convergence_order_two():
    slab = [FDTDLayer(thickness_m=0.5e-6, eps_inf=4.0)]       # n=2 dielectric slab (faces on nodes)
    lmin, lmax, npw = 1.2e-6, 1.6e-6, 1.0
    res_list = [16, 22, 32, 45]
    # reference: a much finer graded mesh (SAME staggering -> a clean same-scheme fine reference)
    r_ref = solve_fdtd_1d(slab, lambda_min_m=lmin, lambda_max_m=lmax, resolution=200, n_pad_wave=npw,
                          refine={0: 2})
    errs, maxdz = [], []
    for rr in res_list:
        r = solve_fdtd_1d(slab, lambda_min_m=lmin, lambda_max_m=lmax, resolution=rr, n_pad_wave=npw,
                          refine={0: 2})
        errs.append(_band_max_absdiff(r, r_ref, "R"))
        dz, _, _, _, _, _, _, _ = _grid_metrics(slab, lmin, lmax, rr, npw)
        maxdz.append(dz)                                     # coarsest cell == max-dz on this mesh
    errs, maxdz = np.array(errs), np.array(maxdz)
    # res_list is INCREASING (finer meshes, smaller max-dz), so the error must shrink monotonically
    assert np.all(np.diff(errs) < 0), errs                 # error shrinks as the mesh refines
    slope = float(np.polyfit(np.log(maxdz), np.log(errs), 1)[0])
    assert slope >= 1.7, slope                              # ~2nd order (measured ~2.5)


# ============================================================================================
# GATE 4 -- THE nm-GAP CASE: 3-nm high-index gap, graded vs uniform-at-gap-resolution, < 0.5% in R
# ============================================================================================

def test_gate4_three_nm_gap():
    lmin, lmax, npw = 1.0e-6, 1.6e-6, 1.0
    gap = 3e-9
    dz_gap = 1e-9                                           # 3 cells across the gap
    stack = [FDTDLayer(thickness_m=0.25e-6, eps_inf=2.25),
             FDTDLayer(thickness_m=gap, eps_inf=12.0),      # high-index gap (n ~ 3.46)
             FDTDLayer(thickness_m=0.25e-6, eps_inf=2.25)]
    _, _, _, _, _, n_max, _, _ = _grid_metrics(stack, lmin, lmax, 24, npw)
    res_gap_uniform = lmin / (dz_gap * n_max)               # uniform resolution giving dz ~ dz_gap
    r_graded = solve_fdtd_1d(stack, lambda_min_m=lmin, lambda_max_m=lmax, resolution=24, n_pad_wave=npw,
                             refine={1: dz_gap})            # target dz [m] in the gap layer
    r_ref = solve_fdtd_1d(stack, lambda_min_m=lmin, lambda_max_m=lmax, resolution=res_gap_uniform,
                          n_pad_wave=npw)                    # brute-force uniform at the gap resolution
    dR = _band_max_absdiff(r_graded, r_ref, "R")
    n_graded = _graded_cellcount(stack, lmin, lmax, 24, npw, {1: dz_gap})
    _, _, _, _, n_ref, _, _, _ = _grid_metrics(stack, lmin, lmax, res_gap_uniform, npw)
    assert r_ref.R[r_ref.band].mean() > 1e-3               # the gap actually reflects (nontrivial R)
    assert dR < 5e-3, dR                                    # < 0.5% in R
    assert n_ref / n_graded >= 8.0, (n_graded, n_ref)       # ~10x fewer cells


# ============================================================================================
# GATE 5 -- STABILITY at the Courant bound (50k steps) + over-bound guard
# ============================================================================================

def test_gate5_50k_steps_bounded_at_courant_bound():
    # build a graded grid (vacuum pads + refined dielectric slab) and march 50k steps at S = 1
    slab = [FDTDLayer(thickness_m=0.4e-6, eps_inf=4.0)]
    lmin, lmax, npw, res = 1.2e-6, 1.6e-6, 1.0, 30
    dz0, pad, z_struct, Lz, nz, n_max, f_min, f_max = _grid_metrics(slab, lmin, lmax, res, npw)
    z_edges = _refined_full_edges(slab, dz0, pad, z_struct, {0: 4}, 8)
    dz_primal = np.diff(z_edges)
    nE = z_edges.size
    eps_inf = np.ones(nE); wp = np.zeros(nE); gam = np.zeros(nE); chi3 = np.zeros(nE)
    m = (z_edges >= pad) & (z_edges < pad + 0.4e-6)
    eps_inf[m] = 4.0
    dt = 1.0 * float(dz_primal.min()) / C_LIGHT            # S = 1 exactly (the documented bound)
    nsteps = 50000
    f_c = 0.5 * (f_min + f_max)
    tau = 1.0 / (np.pi * (f_max - f_min))
    tgrid = np.arange(nsteps) * dt
    src = np.exp(-((tgrid - 6 * tau) / tau) ** 2) * np.cos(2 * np.pi * f_c * (tgrid - 6 * tau))
    i_src = max(2, int(np.searchsorted(z_edges, 0.35 * pad)))
    i_pL = int(np.searchsorted(z_edges, 0.7 * pad))
    i_pR = min(int(np.searchsorted(z_edges, pad + z_struct + 0.3 * pad)), nE - 1)
    eL, eR = _run_nu(eps_inf, wp, gam, chi3, dz_primal, dt, nsteps, i_src, i_pL, i_pR, src)
    assert np.all(np.isfinite(eL)) and np.all(np.isfinite(eR))
    assert np.max(np.abs(eL)) < 1e3 and np.max(np.abs(eR)) < 1e3     # bounded, not exploding
    # the pulse is absorbed/radiated away -> the tail decays (no spurious growth at S = 1)
    assert np.max(np.abs(eL[-1000:])) < 1e-3


def test_gate5_over_bound_courant_raises():
    slab = [FDTDLayer(thickness_m=0.4e-6, eps_inf=4.0)]
    # dt 5% above the min-cell Courant bound must be CAUGHT (raise), not silently explode
    with pytest.raises(ValueError, match="Courant"):
        solve_fdtd_1d(slab, lambda_min_m=1.2e-6, lambda_max_m=1.6e-6, resolution=30, n_pad_wave=1.0,
                      refine={0: 4}, courant=1.05)
    # a courant <= 1 is accepted (no raise)
    solve_fdtd_1d(slab, lambda_min_m=1.2e-6, lambda_max_m=1.6e-6, resolution=24, n_pad_wave=1.0,
                  refine={0: 3}, courant=0.99)


# ============================================================================================
# GATE 6 -- DRUDE ADE on a graded mesh (mirror the infra Drude-slab absorption gate)
# ============================================================================================

def test_gate6_drude_slab_absorbs_on_graded_mesh():
    # same Drude params as tests/test_audit_2026_07_17_infra.test_fdtd_1d_drude_slab_absorbs,
    # but the slab is refined on a graded mesh (coarse vacuum far-field + fine Drude film)
    lay = [FDTDLayer(thickness_m=0.2e-6, eps_inf=2.0, drude_wp_rad_s=1.2e15, drude_gamma_rad_s=8e13)]
    r = solve_fdtd_1d(lay, lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30, n_pad_wave=1.5,
                      refine={0: 5})
    b = r.band
    A = 1.0 - r.R[b] - r.T[b]
    assert np.all(A > 0.02)                                 # lossy Drude absorbs across the band
    assert np.all((r.R[b] >= 0) & (r.T[b] >= 0) & (A <= 1.0))


# ============================================================================================
# make_graded_z construction + unsupported-combination guards
# ============================================================================================

def test_make_graded_z_geometry():
    layers = [FDTDLayer(thickness_m=1.0e-6, eps_inf=2.25),
              FDTDLayer(thickness_m=0.03e-6, eps_inf=4.0),
              FDTDLayer(thickness_m=1.0e-6, eps_inf=2.25)]
    dz0 = 40e-9
    ze = make_graded_z(layers, resolution=25, refine={1: 8}, dz_base_m=dz0)
    d = np.diff(ze)
    assert np.all(d > 0)                                   # strictly increasing
    assert ze[0] == 0.0 and abs(ze[-1] - 2.03e-6) < dz0    # spans the structure
    assert d.min() <= dz0 / 6.0                            # the refined layer really got fine cells
    assert d.max() <= dz0 * 1.001                          # never coarser than the baseline
    # geometric transitions: the ratio is <= ~1.15 in the smooth interior (interface snap cells aside)
    ratios = d[1:] / d[:-1]
    assert np.percentile(ratios, 95) <= 1.16
    # each interior layer boundary lands ON a node (material interfaces resolved)
    for b in (1.0e-6, 1.03e-6):
        assert np.min(np.abs(ze - b)) < 1e-12


def test_refine_and_z_edges_mutually_exclusive():
    layers = [FDTDLayer(thickness_m=0.2e-6, eps_inf=4.0)]
    ze = uniform_z_edges(layers, lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=20)
    with pytest.raises(ValueError, match="either"):
        solve_fdtd_1d(layers, lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=20,
                      z_edges=ze, refine={0: 4})


def test_nonuniform_rejects_time_varying_hooks():
    layers = [FDTDLayer(thickness_m=0.2e-6, eps_inf=4.0)]
    with pytest.raises(NotImplementedError, match="time-varying"):
        solve_fdtd_1d(layers, lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=20, n_pad_wave=1.5,
                      refine={0: 4}, eps_inf_of_t=lambda t: 4.0)


def test_nonuniform_rejects_bichromatic():
    layers = [FDTDLayer(thickness_m=0.2e-6, eps_inf=4.0)]
    with pytest.raises(NotImplementedError, match="bichromatic"):
        solve_fdtd_1d(layers, lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=20, n_pad_wave=1.5,
                      refine={0: 4}, second_source={"lambda0_m": 1.3e-6})


def test_z_edges_must_be_increasing():
    layers = [FDTDLayer(thickness_m=0.2e-6, eps_inf=4.0)]
    bad = np.array([0.0, 1e-7, 0.5e-7, 2e-7, 3e-7])        # not monotone
    with pytest.raises(ValueError, match="increasing"):
        solve_fdtd_1d(layers, lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=20, z_edges=bad)
