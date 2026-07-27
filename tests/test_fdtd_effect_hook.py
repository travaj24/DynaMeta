"""Fast unit tests for the per-cell time-domain eps hook (optics/fdtd_seam.effect_eps_to_fdtd_grid,
roadmap R4): it must be byte-identical, cell-by-cell, to the scalar _eps_to_fdtd_layer Drude inversion."""
import numpy as np
import pytest

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


def test_negative_imag_raises_like_the_scalar_twin():
    """AUDIT V-2: Im(eps) < 0 is GAIN under exp(-i omega t) and must RAISE, not clamp. The clamp
    this replaces was unbounded (eps = -180 - 30j was realized as a strictly real, gamma = 0,
    zero-absorption layer), and the byte-identity contract with the scalar twin covers the guard
    too -- both spellings must reject the same cell."""
    for bad in (4.0 - 1e-3j, -180.0 - 30.0j):
        with pytest.raises(ValueError, match="exp\\(-i omega t\\)"):
            effect_eps_to_fdtd_grid(np.array([bad]), LAM)
        with pytest.raises(ValueError, match="exp\\(-i omega t\\)"):
            _eps_to_fdtd_layer(1e-9, bad, LAM)
    # ONE bad cell anywhere in a valid grid is enough (no partial/silent pass-through)
    with pytest.raises(ValueError):
        effect_eps_to_fdtd_grid(np.array([[4.0 + 0.1j, 9.0 + 0j], [2.0 + 0j, -3.0 - 2.0j]]), LAM)


def test_minus_zero_imag_is_not_a_violation():
    """-0.0 is +0.0 numerically (a lossless layer written with a negative zero, e.g. from a
    conjugate or an interpolator); it must still be accepted and normalized, so valid inputs stay
    byte-identical to the pre-guard behaviour."""
    eps_inf, wp, gam = effect_eps_to_fdtd_grid(np.array([complex(4.0, -0.0)]), LAM)
    assert float(eps_inf[0]) == 4.0 and float(wp[0]) == 0.0 and float(gam[0]) == 0.0
    L = _eps_to_fdtd_layer(1e-9, complex(4.0, -0.0), LAM)
    assert L.eps_inf == 4.0 and L.drude_wp_rad_s == 0.0 and L.drude_gamma_rad_s == 0.0


def test_shape_preserved_2d():
    g = (4.0 + 0.1j) * np.ones((5, 7))
    eps_inf, wp, gam = effect_eps_to_fdtd_grid(g, LAM)
    assert eps_inf.shape == (5, 7) and wp.shape == (5, 7) and gam.shape == (5, 7)
