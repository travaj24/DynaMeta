"""Tests for the R19 density-gradient post-hoc quantum correction. The Schrodinger-Poisson
dead-layer oracle lives in validation/density_gradient_dead_layer.py."""
import numpy as np
import pytest

from dynameta.constants import HBAR, M_E, Q_E as Q
from dynameta.carriers.density_gradient import (dg_correct_density_1d, dg_length_m,
                                                quantum_potential_V)

MSTAR = 0.35 * M_E


def test_quantum_potential_gaussian_closed_form():
    """audit C-3/N3: the `inner = |z| < 8e-9` mask this test used to carry is GONE -- the closed
    form is now met at EVERY node including the four the old np.gradient^2 composition got wrong.
    Measured on this fixture: max relative error 3.1e-5 over the full array (it was 4.96e-1, i.e.
    496x over the 1e-3 budget, at the end nodes before the fix)."""
    s = 3e-9
    z = np.linspace(-12e-9, 12e-9, 1601)
    n = 1e26 * np.exp(-z ** 2 / (2.0 * s ** 2))
    lam = quantum_potential_V(z, n, MSTAR)
    b = HBAR ** 2 / (6.0 * MSTAR * Q)
    cf = b * (z ** 2 / (4.0 * s ** 4) - 1.0 / (2.0 * s ** 2))
    assert np.max(np.abs(lam - cf)) < 1e-3 * np.max(np.abs(cf))       # NO mask


def test_quantum_potential_is_second_order_at_the_four_end_nodes():
    """audit C-3 / N3 (= 2026-07-17 ledger S1-6), the load-bearing gate.

    `np.gradient(np.gradient(u, z), z)` is non-convergent at FOUR nodes: edge_order=1 makes the
    inner pass one-sided-first-order at 0 and N-1, and the outer centred pass then mixes that
    biased slope into 1 and N-2. The limits are exactly (1/2) f'' at 0/N-1 and (3/4) f'' at
    1/N-2 -- measured on the Gaussian fixture above as ratios 0.5036 and 0.7518 -- and they do
    NOT improve with h. So refinement alone is the discriminator: this gate measures the OBSERVED
    order p = log2(e(h)/e(h/2)) at each of the four nodes (plus one interior node as a control)
    and requires p ~ 2 everywhere. Under the old composition p would be ~0 at all four."""
    k = 3.0e8                                                     # ~5 oscillations over 20 nm
    zmax = 20e-9

    def n_of(z):                                                  # u = sqrt(n) = exp(sin(k z))
        return np.exp(2.0 * np.sin(k * z))

    def lam_exact(z):                                             # b (u''/u) = b (s'' + s'^2)
        return b_ref * (-k ** 2 * np.sin(k * z) + (k * np.cos(k * z)) ** 2)

    b_ref = HBAR ** 2 / (6.0 * MSTAR * Q)
    errs = {}
    for nn in (201, 401, 801, 1601):
        z = np.linspace(0.0, zmax, nn)
        got = quantum_potential_V(z, n_of(z), MSTAR)
        ex = lam_exact(z)
        scale = np.max(np.abs(ex))
        errs[nn] = {tag: abs(got[i] - ex[i]) / scale
                    for tag, i in (("0", 0), ("1", 1), ("mid", nn // 2),
                                   ("N-2", nn - 2), ("N-1", nn - 1))}
    grids = sorted(errs)
    for tag in ("0", "1", "mid", "N-2", "N-1"):
        for coarse, fine in zip(grids[:-1], grids[1:]):
            p = np.log2(errs[coarse][tag] / errs[fine][tag])
            assert p > 1.7, "node {}: observed order {:.2f} between n={} and n={} (want ~2)".format(
                tag, p, coarse, fine)
    # and the absolute error at the two hardest nodes is small on the finest grid
    assert errs[1601]["0"] < 1e-4 and errs[1601]["N-1"] < 1e-4


def test_quantum_potential_matches_a_quadratic_exactly_at_every_node():
    """The stencils' exactness contract: interior is the 3-point non-uniform second difference
    (exact for quadratics) and the ends are 4-point one-sided (exact for cubics). On u = 1 + a z
    + c z^2 the discrete u'' must equal 2c to round-off AT EVERY NODE -- including on a
    NON-UNIFORM mesh, which is the case the end weights are solved for."""
    for z in (np.linspace(0.0, 12e-9, 61),
              np.sort(np.concatenate([np.linspace(0.0, 4e-9, 25),      # deliberately graded
                                      np.linspace(4.2e-9, 12e-9, 20)]))):
        a, c = 3.0e7, 5.0e15
        u = 1.0 + a * z + c * z ** 2
        lam = quantum_potential_V(z, u ** 2, MSTAR)
        b = HBAR ** 2 / (6.0 * MSTAR * Q)
        assert np.allclose(lam, b * 2.0 * c / u, rtol=1e-8)


def test_quantum_potential_rejects_a_non_monotonic_grid():
    z = np.array([0.0, 1e-9, 3e-9, 2e-9, 4e-9, 5e-9])             # not sorted
    with pytest.raises(ValueError, match="monotonic"):
        quantum_potential_V(z, np.full_like(z, 1e26), MSTAR)


def test_c8_degeneracy_factor_shortens_l_q_and_defaults_to_the_boltzmann_limit():
    """audit C-8: the BVP closes with BOLTZMANN statistics for a material this module declares
    degenerate, and the in-Newton twin (physics_density_gradient) uses the FD g-form. The
    FD-consistent length is sqrt(g) shorter -- pinned here against the auditor's exact numbers --
    so the ~1.2 nm the header quotes is a CALIBRATION that the fitted `gamma` absorbs, not a
    derivation. degeneracy_g=1 is the byte-identical off-switch."""
    from dynameta.carriers.einstein import g_einstein
    lq_b = dg_length_m(MSTAR)
    assert lq_b == pytest.approx(1.184731e-9, rel=1e-6)               # the Boltzmann value
    assert dg_length_m(MSTAR, degeneracy_g=1.0) == lq_b               # explicit off-switch
    assert dg_length_m(MSTAR, degeneracy_g=13.4) == pytest.approx(3.236438e-10, rel=1e-6)
    for g in (6.8, 13.4):                                             # eta ~ 10 and ~ 20
        assert lq_b / dg_length_m(MSTAR, degeneracy_g=g) == pytest.approx(np.sqrt(g), rel=1e-12)
    assert 2.6 < lq_b / dg_length_m(MSTAR, degeneracy_g=6.8) < 3.7    # the reported 2.6-3.7x band
    # the factor is the repo's own g(n/N_c), not a magic constant
    assert dg_length_m(MSTAR, degeneracy_g=float(g_einstein(30.0))) < 0.5 * lq_b
    with pytest.raises(ValueError):
        dg_length_m(MSTAR, degeneracy_g=0.0)


def test_dg_off_switch_and_dead_layer():
    z = np.linspace(0.0, 15e-9, 901)
    n = np.full_like(z, 2e26)
    assert np.array_equal(dg_correct_density_1d(z, n, MSTAR, gamma=0.0), n)
    n_dg = dg_correct_density_1d(z, n, MSTAR)
    lq = dg_length_m(MSTAR)
    assert n_dg[0] < 1e-3 * 2e26                          # hard wall
    assert abs(n_dg[-1] / 2e26 - 1.0) < 1e-4              # bulk recovered
    assert 0.8e-9 < lq < 1.8e-9                           # the ~1 nm ITO dead-layer scale


def test_dg_guards():
    z = np.linspace(0.0, 10e-9, 301)
    with pytest.raises(ValueError):
        dg_correct_density_1d(z, np.zeros_like(z), MSTAR)
    with pytest.raises(ValueError):
        dg_correct_density_1d(z, np.full_like(z, 1e26), MSTAR, hard_wall="middle")
    with pytest.raises(ValueError):
        quantum_potential_V(z, np.full(300, 1e26), MSTAR)


# ---- R19 follow-on: in-Newton DG-DD (module-level guards; the 4-variable Newton oracle is
# validation/dg_dd_in_newton.py) -----------------------------------------------------------------

def test_dg_b_coefficient_and_guards():
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_density_gradient import dg_b_coefficient, set_dg_gamma
    b = dg_b_coefficient(MSTAR, 1.0)
    assert b == pytest.approx(HBAR ** 2 / (6.0 * MSTAR * Q), rel=1e-14)
    assert dg_b_coefficient(MSTAR, 0.0) == 0.0
    with pytest.raises(ValueError):
        dg_b_coefficient(-1.0)
    with pytest.raises(ValueError):
        set_dg_gamma("d", "r", 1.5)                      # frac outside [0, 1]


def test_dg_contact_guard():
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_density_gradient import setup_contact_dg
    with pytest.raises(ValueError):
        setup_contact_dg("dev", "c", 0.0)
