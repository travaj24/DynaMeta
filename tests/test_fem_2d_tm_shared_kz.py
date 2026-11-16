"""optics.fem_2d_tm uses the ONE out-of-plane wavenumber kernel (resonance._kz: one implementation per kernel, see the
duplicated-kernel census tests/test_nonlocal_tmm.py::test_q19_one_kz_admittance_kpar, which
failed while the FEM carried its own copy).

The shared kernel is the PRINCIPAL square root with an exactly-zero imaginary part of its argument
read as +0.0 (resonance._outgoing_sqrt); the removed copy flipped the root onto Im >= 0. For a
cladding with Im(eps) >= 0 the two give the same root: the principal root of a number with
Im >= +0 already has both parts >= 0, and the zero rule makes a lossless cladding's Im = -0.0
behave as +0.0. Measured with the two kernels in one process: R, T, r, t, A_independent of six
cells (strip at normal / 20 deg, a
diffracting lossless grating with evanescent cladding orders, a lossy and an opaque metal
substrate, a 1.5 - 0j substrate index) bit-identical under the two kernels in one process. Every
FEM call passes a cladding eps = n**2 of a passive medium.

The signed zero is the one passive trap: the bare principal root of -a - 0j is -i sqrt(a), a
GROWING evanescent order. n**2 keeps Im = -0.0 for a lossless METAL written as n = (-0.0, +y) --
the passive index tmm_reference._passive_sqrt returns for eps = -y^2 - 0j -- or (0, -y), so the
kernel itself has to apply the rule; for a dielectric (Re n > 0) CPython's n**2 already returns
Im = +0.0. The gates below construct both signed zeros explicitly, so they mean the same on every
Python version."""
import cmath
import math

import pytest

pytest.importorskip("ngsolve")

from dynameta.optics import fem_2d_tm, resonance  # noqa: E402
from dynameta.optics.fem_2d_tm import TMBlock, TMLayer, solve_fem_2d_tm  # noqa: E402


def _neg_zero(x: float) -> bool:
    return x == 0.0 and math.copysign(1.0, x) < 0.0


def test_fem_uses_the_shared_kz_kernel():
    assert fem_2d_tm._kz is resonance._kz


def test_negative_zero_substrate_index_is_the_positive_zero_one():
    """A diffracting lossless grating at 10 deg (substrate orders evanescent): n_sub = 1.5 with a
    -0.0 imaginary part and n_sub = 1.5 give the same solve bit for bit."""
    lays = [TMLayer("g", 200e-9, 1.0, (TMBlock(0.0, 660e-9, 3.4 ** 2, "rib"),))]

    def solve(n_sub):
        r = solve_fem_2d_tm(1.6e-6, lays, lambda_m=1.2e-6, theta_deg=10.0, n_super=1.0,
                            n_sub=n_sub, polarization="p", order=3, num_threads=2)
        return r.r, r.R, r.t, r.T
    assert solve(complex(1.5, -0.0)) == solve(1.5)


def test_lossless_metal_substrate_does_not_depend_on_the_sign_of_zero():
    """A lossy grating on a lossless metal substrate (eps = -20): the index spelled (-0.0, +sqrt 20)
    -- Im n > 0, passive, but n**2 = -20 - 0j -- gives the solve of (+0.0, +sqrt 20) bit for bit.
    Before the zero rule every substrate order took the growing root and the solve overflowed
    (OverflowError: math range error)."""
    lays = [TMLayer("g", 120e-9, complex(2.0, 0.2), (TMBlock(0.0, 300e-9, complex(4.0, 0.1), "rib"),))]
    n_pos, n_neg = complex(0.0, math.sqrt(20.0)), complex(-0.0, math.sqrt(20.0))
    assert _neg_zero((n_neg ** 2).imag)                        # the trap reaches the cladding eps

    def solve(n_sub):
        r = solve_fem_2d_tm(800e-9, lays, lambda_m=1.0e-6, theta_deg=10.0, n_super=1.0,
                            n_sub=n_sub, polarization="p", order=3, num_threads=2)
        return r.r, r.R
    assert solve(n_neg) == solve(n_pos)


def test_kernel_reads_a_negative_zero_cladding_eps_as_positive():
    k0, kp = 2e-3, 5e-3
    e_metal = complex(-0.0, math.sqrt(20.0)) ** 2                # -20 - 0j: Im = -0.0 survives n**2
    q_trap = e_metal * k0 * k0 - kp * kp
    assert q_trap.real < 0 and _neg_zero(q_trap.imag)
    assert cmath.sqrt(q_trap).imag < 0                          # the bare root: the growing order
    assert resonance._kz(e_metal, k0, kp).imag > 0              # the kernel: the decaying one
    e_diel = complex(2.25, -0.0)                                 # the argument's -0.0 built directly
    q = e_diel * complex(k0, -0.0) * complex(k0, -0.0) - kp * kp
    assert q.real < 0 and _neg_zero(q.imag)
    assert resonance._kz(e_diel, complex(k0, -0.0), kp) == cmath.sqrt(complex(q.real, 0.0))
