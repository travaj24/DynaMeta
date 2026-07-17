"""Coverage gates added by the 2026-07-17 audit remediation (Wave 3): the load-bearing paths
that previously had no CI-reachable test. Pure numpy/scipy (devsim-free by construction --
that IS one of the gates)."""

import subprocess
import sys

import numpy as np
import pytest


# ---- S1-2: the Fermi-integral fit + inverter, devsim-free ---------------------------------

def test_f12_aymerich_humet_monotone_and_accurate():
    from dynameta.carriers.physics_equilibrium import F12_aymerich_humet, invert_F12
    etas = np.linspace(-20.0, 80.0, 8001)
    vals = np.array([F12_aymerich_humet(e) for e in etas])
    assert np.all(np.diff(vals) > 0.0)          # strictly increasing incl. the eta>=20 regime
    # accuracy vs direct quadrature of the NORMALIZED Fermi-Dirac integral
    # F_1/2(eta) = (2/sqrt(pi)) int_0^inf sqrt(x)/(1+exp(x-eta)) dx  (F -> e^eta nondegenerate)
    from scipy.integrate import quad
    pref = 2.0 / np.sqrt(np.pi)
    for eta in (-10.0, 0.0, 5.0, 20.0, 40.0, 70.0):
        raw, _ = quad(lambda x: np.sqrt(x) / (1.0 + np.exp(np.clip(x - eta, -700, 700))),
                      0.0, max(200.0, eta + 200.0), limit=400)
        exact = pref * raw
        assert abs(F12_aymerich_humet(eta) - exact) / exact < 8e-3, eta
    # the inverter lands on the right root
    for eta_true in (-5.0, 10.0, 25.0, 60.0):
        assert abs(invert_F12(F12_aymerich_humet(eta_true)) - eta_true) < 1e-6


def test_f12_importable_without_devsim():
    # audit S1-2: the pure-math Fermi machinery must import with devsim BLOCKED
    code = ("import sys; sys.modules['devsim'] = None\n"
            "import importlib\n"
            "import dynameta.carriers.physics_equilibrium as pe\n"
            "assert pe.F12_aymerich_humet(25.0) > 0\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


# ---- S1-3: the solver-free base-import contract -------------------------------------------

def test_base_imports_pull_no_heavy_solvers():
    # runs on EVERY leg (the point is the base import stays clean even when solvers ARE installed)
    code = ("import sys\n"
            "import dynameta\n"
            "import dynameta.carriers\n"
            "import dynameta.drivers\n"
            "import dynameta.optics\n"
            "for mod in ('devsim', 'ngsolve', 'netgen', 'gmsh'):\n"
            "    assert mod not in sys.modules, mod + ' leaked into the base import'\n"
            "print('clean')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "clean" in out.stdout, out.stderr


# ---- S4-4: resample_to_grid (the carrier->grid bridge gridder) ----------------------------

def test_resample_to_grid_linear_and_nan_fill():
    from dynameta.core.resample import resample_to_grid
    rng = np.random.default_rng(7)
    # scattered nodes over ~[0,1]^2 with a linear field; the grid spans the node bbox, whose
    # corners are (with random nodes) OUTSIDE the convex hull -> the NaN nearest-fill path runs
    pts = rng.uniform(0.0, 1.0, size=(400, 2))
    vals = 2.0 * pts[:, 0] - 3.0 * pts[:, 1] + 0.5
    out = resample_to_grid(pts, {"f": vals}, (21, 19))
    grid = out["f"]
    xg, yg = out["axis_0"], out["axis_1"]
    assert grid.shape == (21, 19)
    assert np.all(np.isfinite(grid))                  # convex-hull NaNs were nearest-filled
    Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
    interior = ((Xg > xg[0] + 0.15) & (Xg < xg[-1] - 0.15)
                & (Yg > yg[0] + 0.15) & (Yg < yg[-1] - 0.15))
    exact = 2.0 * Xg - 3.0 * Yg + 0.5
    assert np.max(np.abs(grid[interior] - exact[interior])) < 5e-2
    # the NaN nearest-fill index alignment (the audited ravel-order contract): a bbox corner
    # outside the hull must take a value from a NEARBY node, never a scrambled far one
    assert abs(grid[0, 0] - exact[0, 0]) < 0.6
    assert abs(grid[-1, -1] - exact[-1, -1]) < 0.6
    assert abs(grid[0, -1] - exact[0, -1]) < 0.6
    assert abs(grid[-1, 0] - exact[-1, 0]) < 0.6


# ---- S2-4: solve_fdtd_1d fast gates -------------------------------------------------------

def _c():
    from dynameta.constants import C_LIGHT
    return C_LIGHT


def test_fdtd_1d_dielectric_slab_vs_airy():
    # audit S2-4: solve_fdtd_1d had zero pytest coverage (low-res port of validation GATE A)
    from dynameta.optics.fdtd import solve_fdtd_1d, FDTDLayer
    n_slab, d = 2.0, 0.30e-6
    res = solve_fdtd_1d([FDTDLayer(thickness_m=d, eps_inf=n_slab ** 2)],
                        lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30)
    lam_arr = _c() / res.freqs_Hz
    band = res.band
    r12 = (1 - n_slab) / (1 + n_slab)
    delta = 2.0 * np.pi * n_slab * d / lam_arr
    r_airy = r12 * (1 - np.exp(2j * delta)) / (1 - r12 ** 2 * np.exp(2j * delta))
    R_airy = np.abs(r_airy) ** 2
    assert np.max(np.abs(res.R[band] - R_airy[band])) < 0.02
    assert np.max(np.abs(res.R[band] + res.T[band] - 1.0)) < 0.02   # lossless energy closure


def test_fdtd_1d_drude_slab_absorbs():
    from dynameta.optics.fdtd import solve_fdtd_1d, FDTDLayer
    res = solve_fdtd_1d([FDTDLayer(thickness_m=0.2e-6, eps_inf=2.0,
                                   drude_wp_rad_s=1.2e15, drude_gamma_rad_s=8e13)],
                        lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30)
    band = res.band
    A = 1.0 - res.R[band] - res.T[band]
    assert np.all(A > 0.02)                            # lossy Drude absorbs across the band
    assert np.all((res.R[band] >= 0) & (res.T[band] >= 0) & (A <= 1.0))
