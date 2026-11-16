"""optics.fem_2d_tm uses the ONE out-of-plane wavenumber kernel (resonance._kz: one implementation per kernel, see the
duplicated-kernel census tests/test_nonlocal_tmm.py::test_q19_one_kz_admittance_kpar, which
failed while the FEM carried its own copy).

The shared kernel is the PRINCIPAL square root; the removed copy flipped it onto Im >= 0. For a
cladding with Im(eps) >= +0 the two are the same function bit for bit (the principal root of a
number with Im >= 0 already has both parts >= 0) -- measured: R, T, r, t, A_independent of six
cells (strip at normal / 20 deg, a diffracting lossless grating with evanescent cladding orders, a
lossy and an opaque metal substrate, a 1.5 - 0j substrate index) bit-identical under the two
kernels in one process. Every FEM call passes a cladding eps = n**2 of a passive medium. A
NEGATIVE-zero imaginary part would be the one passive trap (the principal root of -a - 0j is
-i sqrt(a), a growing evanescent order), but CPython forms n**2 by multiplying up from 1+0j,
which turns Im = -0.0 into +0.0; the second gate pins that."""
import cmath

import pytest

pytest.importorskip("ngsolve")

from dynameta.optics import fem_2d_tm, resonance  # noqa: E402
from dynameta.optics.fem_2d_tm import TMBlock, TMLayer, solve_fem_2d_tm  # noqa: E402


def test_fem_uses_the_shared_kz_kernel():
    assert fem_2d_tm._kz is resonance._kz


def test_negative_zero_substrate_index_is_the_positive_zero_one():
    """A diffracting lossless grating at 10 deg (substrate orders evanescent): n_sub = 1.5 - 0j and
    n_sub = 1.5 give the same solve bit for bit."""
    lays = [TMLayer("g", 200e-9, 1.0, (TMBlock(0.0, 660e-9, 3.4 ** 2, "rib"),))]

    def solve(n_sub):
        r = solve_fem_2d_tm(1.6e-6, lays, lambda_m=1.2e-6, theta_deg=10.0, n_super=1.0,
                            n_sub=n_sub, polarization="p", order=3, num_threads=2)
        return r.r, r.R, r.t, r.T
    assert solve(1.5 - 0j) == solve(1.5)
    k0 = 2e-3
    q_trap = complex(2.25 * k0 * k0 - 5e-3 * 5e-3, -0.0)   # Im = -0.0 takes the growing root ...
    assert cmath.sqrt(q_trap).imag < 0
    assert resonance._kz(complex(1.5 - 0j) ** 2, k0, 5e-3).imag > 0      # but n**2 has Im = +0
