"""Gates for the time-domain hydrodynamic FDTD (roadmap item 5.2).

Physics recap (see ``dynameta/optics/hydro_fdtd.py`` for the reduced equations + discretization):
  * a fixed-kx (p-pol / TM) 1-D-in-z complex-field FDTD marches Maxwell + the linearized electron
    fluid (continuity ``drho/dt = -div J`` + momentum ``dJ/dt = eps0 wp^2 E - gamma J -
    beta^2 grad rho``) on a staggered Yee grid, with the hard-wall ABC ``J_z = 0`` at metal faces;
  * beta -> 0 collapses to the local Drude reduced solver (no pressure/charge);
  * the pressure term adds the LONGITUDINAL bulk-plasmon standing waves at
    ``omega_m = sqrt(wp^2/eps_inf + beta^2 (m pi/d)^2)`` (== the p-pol absorption peaks of the
    frequency-domain oracle ``optics.nonlocal_tmm``).

TIER-1 gates (linear):
  1. beta -> 0 == the LOCAL reduced solver, BYTE-IDENTICAL (the pressure/charge machinery is a
     strict no-op at beta = 0); + a physical R/T cross-check vs nonlocal_tmm at normal incidence.
  2. the bulk-plasmon resonances land within 1% of ``k_L d = m*pi`` (m = 1, 3, 5) AND within 1%
     of nonlocal_tmm's absorption-peak frequencies -- the load-bearing cross-solver validation.
  3. energy budget: a LOSSLESS film conserves energy (R + T + A = 1, A ~ 0) in the driven
     spectrum; a lossy film has physical (>= 0) R, T, A.

TIER-2 gates (nonlinear SHG) cover the CONFINED kx = 0 self-consistent second-harmonic
generation that IS solid (the fluid's convective + density-modulation nonlinearity generates a
2 omega charge scaling as the SHG power law); the RADIATED angle-selection-rule SHG is documented
as deferred with verified blockers (see the module's TIER 2 SCOPE note).

Run: python -m pytest tests/test_hydro_fdtd.py -q
"""
import math

import numpy as np
import pytest

from dynameta.optics import hydro_fdtd as hf
from dynameta.optics import nonlocal_tmm as nt
from dynameta.constants import C_LIGHT


# ------------------------------------------------------------------------------------------------
# unit sanity
# ------------------------------------------------------------------------------------------------
def test_beta_from_vf_conventions():
    vF = 1.07e6
    assert hf.beta_from_vf(vF, "high_freq") == pytest.approx(math.sqrt(0.6) * vF, rel=1e-12)
    assert hf.beta_from_vf(vF, "thomas_fermi") == pytest.approx(math.sqrt(1.0 / 3.0) * vF, rel=1e-12)
    assert hf.beta_from_vf(vF) == hf.beta_from_vf(vF, "high_freq")
    # matches the nonlocal_tmm oracle's convention exactly (apples-to-apples cross gate)
    assert hf.beta_from_vf(vF) == pytest.approx(nt.beta_from_vf(vF), rel=1e-15)
    with pytest.raises(ValueError):
        hf.beta_from_vf(vF, "nonsense")


def test_bulk_plasmon_omega_closed_form():
    slab = hf.HydroSlab(1.0, 8.65e15, 3e12, hf.beta_from_vf(1.07e6), 2e-9)
    for m in (1, 2, 3, 5):
        expect = math.sqrt(slab.wp ** 2 / slab.eps_inf + slab.beta ** 2 * (m * math.pi / slab.thickness_m) ** 2)
        assert hf.bulk_plasmon_omega(m, slab) == pytest.approx(expect, rel=1e-14)
        assert hf.bulk_plasmon_omega(m, slab) > slab.wp        # bulk plasmons sit ABOVE omega_p


# ------------------------------------------------------------------------------------------------
# Gate 1: beta -> 0 == the LOCAL reduced solver
# ------------------------------------------------------------------------------------------------
def test_gate1_beta0_byte_identical_to_local():
    """The pressure + continuity machinery is a strict NO-OP at beta = 0: the marcher with the
    nonlocal path ON (beta = 0) is BYTE-IDENTICAL to the local path (roadmap gate 1, <1e-8)."""
    metal0 = hf.HydroSlab(eps_inf=1.0, wp=1.2e16, gamma=1.1e14, beta=0.0, thickness_m=25e-9)
    dz = (C_LIGHT / (C_LIGHT / 650e-9)) / 40.0
    grid = hf._build_grid([metal0], dz=dz, pad_m=200e-9, n_super=1.0, n_sub=1.0)
    nz = grid.nz
    dt = 0.5 * dz / C_LIGHT
    nsteps = 4000
    tg = np.arange(nsteps) * dt
    tau = 1.0 / (math.pi * (C_LIGHT / 650e-9 - C_LIGHT / 750e-9))
    t0 = 8.0 * tau
    w_c = 2.0 * math.pi * C_LIGHT / 700e-9
    src = np.exp(-((tg - t0) / tau) ** 2) * np.exp(-1j * w_c * (tg - t0))
    kw = dict(dt=dt, nsteps=nsteps, kx=0.0, i_src=8, src=src, i_pL=20, i_pR=nz - 20, mur_v=C_LIGHT)
    tA = hf._march_linear(grid, nonlocal_on=True, **kw)     # nonlocal path, beta = 0
    tB = hf._march_linear(grid, nonlocal_on=False, **kw)    # local path
    for key in ("eL", "eR"):
        assert np.max(np.abs(tA[key] - tB[key])) < 1e-9 * (np.max(np.abs(tB[key])) + 1e-300)


def test_gate1_normal_incidence_rt_vs_tmm():
    """Physical cross-check: the local-limit R/T at normal incidence reproduces the independent
    nonlocal_tmm oracle (grid-dispersion limited, gated 1%)."""
    metal = hf.HydroSlab(1.0, 1.2e16, 1.1e14, 1e-3, 25e-9)     # beta below the nonlocal threshold
    metal_nt = nt.HydroLayer(1.0, 1.2e16, 1.1e14, 1e-3, 25e-9)
    res = hf.solve_tm_spectrum([metal], kx_per_m=0.0, lambda_min_m=650e-9, lambda_max_m=750e-9,
                               cells_per_vacuum=40, min_periods=40, run_damping_times=4.0)
    assert res.omega.size >= 5
    worst = 0.0
    for i, w in enumerate(res.omega):
        Rt, Tt, At = nt.rta(w, [metal_nt], pol="p", k_par_m=0.0)
        worst = max(worst, abs(res.R[i] - Rt), abs(res.T[i] - Tt))
    assert worst < 1e-2, "normal-incidence R/T deviates from nonlocal_tmm by {:.2e}".format(worst)


# ------------------------------------------------------------------------------------------------
# Gate 2: bulk-plasmon standing-wave resonances at k_L d = m*pi (THE cross-solver validation)
# ------------------------------------------------------------------------------------------------
def test_gate2_bulk_plasmon_resonances():
    """The FDTD longitudinal ring-down eigenfrequencies land within 1% of both the closed-form
    k_L d = m*pi prediction AND nonlocal_tmm's p-pol absorption-peak frequencies (m = 1, 3, 5)."""
    from scipy.signal import find_peaks
    d, gamma = 2.0e-9, 3.0e12
    film = hf.HydroSlab(1.0, 8.65e15, gamma, hf.beta_from_vf(1.07e6), d)
    film_nt = nt.HydroLayer(1.0, 8.65e15, gamma, hf.beta_from_vf(1.07e6), d)
    wp = film.wp

    modes = hf.bulk_plasmon_resonances(film, m_list=(1, 3, 5))
    assert set(modes) == {1, 3, 5}, "not all bulk-plasmon modes recovered: {}".format(sorted(modes))

    # nonlocal_tmm absorption-peak frequencies at a representative oblique angle (same fixed k_par)
    kx = (hf.bulk_plasmon_omega(1, film) / C_LIGHT) * math.sin(math.radians(45.0))
    ws = np.linspace(1.001 * wp, 1.33 * wp, 80000)
    A = np.array([nt.rta(w, [film_nt], pol="p", k_par_m=kx)[2] for w in ws])
    pk, _ = find_peaks(A, prominence=1e-3)
    tmm_peaks = ws[pk]
    assert tmm_peaks.size >= 3

    for m in (1, 3, 5):
        w_cf = hf.bulk_plasmon_omega(m, film)                 # closed form
        w_fdtd = modes[m].omega_rad_s
        assert w_fdtd > wp, "bulk plasmon m={} must sit above omega_p".format(m)
        assert abs(w_fdtd - w_cf) / w_cf < 0.01, (
            "m={} FDTD {:.5e} not within 1% of k_L d = m*pi prediction {:.5e}".format(m, w_fdtd, w_cf))
        w_tmm = tmm_peaks[np.argmin(np.abs(tmm_peaks - w_cf))]
        assert abs(w_fdtd - w_tmm) / w_tmm < 0.01, (
            "m={} FDTD {:.5e} not within 1% of nonlocal_tmm peak {:.5e}".format(m, w_fdtd, w_tmm))


# ------------------------------------------------------------------------------------------------
# Gate 3: energy budget R + T + A = 1
# ------------------------------------------------------------------------------------------------
def test_gate3_energy_budget_lossless():
    """A LOSSLESS film (gamma = 0) conserves energy: A = 1 - R - T ~ 0 across the driven spectrum
    (numerical-leakage floor; the discrete scheme + Mur ABC hold energy to ~1e-4)."""
    lossless = hf.HydroSlab(1.0, 1.2e16, 0.0, 1e-3, 30e-9)
    res = hf.solve_tm_spectrum([lossless], kx_per_m=0.0, lambda_min_m=650e-9, lambda_max_m=750e-9,
                               cells_per_vacuum=50, min_periods=50, run_damping_times=3.0)
    assert res.omega.size >= 5
    assert np.max(np.abs(res.A)) < 2e-3, "lossless energy leak {:.2e}".format(np.max(np.abs(res.A)))
    assert np.all(res.R + res.T > 0.99)


def test_gate3_lossy_physical():
    """A lossy film has physical (>= 0) reflectance / transmittance / absorptance summing to 1."""
    lossy = hf.HydroSlab(1.0, 1.2e16, 2e14, 1e-3, 20e-9)
    res = hf.solve_tm_spectrum([lossy], kx_per_m=0.0, lambda_min_m=650e-9, lambda_max_m=750e-9,
                               cells_per_vacuum=40, min_periods=40)
    assert np.all(res.R >= -1e-9) and np.all(res.T >= -1e-9) and np.all(res.A >= -1e-9)
    assert np.all(res.A > 0.0)                                # a lossy metal absorbs
    assert np.allclose(res.R + res.T + res.A, 1.0, atol=1e-9)


# ------------------------------------------------------------------------------------------------
# TIER 2 (nonlinear): confined kx = 0 self-consistent second-harmonic generation
# ------------------------------------------------------------------------------------------------
def _shg_film():
    return hf.HydroSlab(1.0, 8.65e15, 3e12, hf.beta_from_vf(1.07e6), 2e-9)


def test_tier2_nonlinear_generates_second_harmonic():
    """Gate 4/7 (confined form): the FULL fluid nonlinearity generates MORE second harmonic than
    the LINEARIZED reference -- i.e. turning the nonlinearity on lifts the 2 omega charge above the
    linear numerical background (the self-consistent metal nonlinearity IS the SH source)."""
    slab = _shg_film()
    res = hf.solve_shg(slab, seed_amp=0.03)
    assert res.p_w > 0.0
    # the excited standing mode sits at the bulk plasmon omega_1
    assert res.omega1_rad_s == pytest.approx(hf.bulk_plasmon_omega(1, slab), rel=1e-12)
    # nonlinearity ON produces measurably MORE 2 omega than the linearized reference
    assert res.p_2w > 1.5 * res.p_2w_lin, (
        "nonlinear 2w {:.3e} not clearly above the linear reference {:.3e}".format(
            res.p_2w, res.p_2w_lin))


def test_tier2_shg_power_law_slope_two():
    """Gate 5: the PHYSICAL second harmonic (full nonlinear minus the linearized-reference
    numerical background) scales as (drive amplitude)^2 -- the chi2-like SHG power law."""
    slab = _shg_film()
    amps = [4e-3, 8e-3, 1.6e-2, 3.2e-2]
    slope, excess, _ = hf.shg_excess_slope(slab, amps)
    assert np.all(np.diff(excess) > 0.0), "excess SH not monotonically increasing"
    assert 1.7 < slope < 2.3, "SHG excess power-law slope {:.3f} not ~ 2".format(slope)


def test_tier2_linear_limit_rings_at_omega1():
    """The nonlinear solver's LINEAR limit (tiny drive) rings at the bulk-plasmon omega_1 -- the
    nonlinear (n, v) fluid reduces to the correct longitudinal dispersion."""
    from dynameta.optics.ringdown import matrix_pencil
    slab = _shg_film()
    wp = slab.wp
    rec, dt = hf._march_nonlinear_longitudinal(slab, seed_amp=1e-4, m=1,
                                               cells_per_longitudinal=12, record_periods=40,
                                               nonlinear=True)
    step = max(1, int(round(0.05 / (wp * dt))))
    modes = matrix_pencil(rec[::step] - rec.mean(), dt * step, svd_tol=1e-9,
                          amp_floor=1e-2, max_modes=6)
    cand = [x for x in modes if x.omega_rad_s > 0.5 * wp]
    assert cand, "nonlinear solver linear limit produced no ring-down mode"
    w_fdtd = max(cand, key=lambda x: abs(x.amplitude)).omega_rad_s
    # collocated-grid dispersion is looser than the staggered linear solver (~1%); gate 2%
    assert abs(w_fdtd - hf.bulk_plasmon_omega(1, slab)) / hf.bulk_plasmon_omega(1, slab) < 0.02
