"""Fast unit tests for the per-cell time-domain eps hook (optics/fdtd_seam.effect_eps_to_fdtd_grid,
roadmap R4): it must be byte-identical, cell-by-cell, to the scalar _eps_to_fdtd_layer Drude inversion."""
import numpy as np

from dynameta.optics.fdtd_seam import effect_eps_to_fdtd_grid, _eps_to_fdtd_layer

LAM = 1300e-9


def test_grid_matches_scalar_layer_per_cell():
    # cover all three regimes: lossless dielectric, absorber (er<1), high-index lossy (er>=1).
    samples = [4.0 + 0.0j, 6.25 + 0.0j,            # lossless
               -3.0 + 2.0j, 0.2 + 1.5j,            # absorber/metal (er < 1)
               9.0 + 0.8j, 2.25 + 0.05j]           # high-index lossy
    grid = np.array(samples, dtype=np.complex128)
    eps_inf, wp, gam = effect_eps_to_fdtd_grid(grid, LAM)
    for i, e in enumerate(samples):
        L = _eps_to_fdtd_layer(1e-9, e, LAM)
        assert float(eps_inf[i]) == L.eps_inf
        assert float(wp[i]) == L.drude_wp_rad_s
        assert float(gam[i]) == L.drude_gamma_rad_s


def test_lossless_grid_is_pure_dielectric():
    grid = np.array([2.0, 4.0, 6.25], dtype=np.complex128)
    eps_inf, wp, gam = effect_eps_to_fdtd_grid(grid, LAM)
    assert np.allclose(eps_inf, [2.0, 4.0, 6.25])
    assert np.all(wp == 0.0) and np.all(gam == 0.0)


def test_negative_imag_clamped_passive():
    # a tiny gain (Im<0) must clamp to passive (no negative wp^2 / no gain).
    eps_inf, wp, gam = effect_eps_to_fdtd_grid(np.array([4.0 - 1e-3j]), LAM)
    assert np.all(np.isfinite(wp)) and np.all(wp >= 0.0)
    assert float(eps_inf[0]) == 4.0                            # treated as lossless (Im clamped to 0)


def test_shape_preserved_2d():
    g = (4.0 + 0.1j) * np.ones((5, 7))
    eps_inf, wp, gam = effect_eps_to_fdtd_grid(g, LAM)
    assert eps_inf.shape == (5, 7) and wp.shape == (5, 7) and gam.shape == (5, 7)
