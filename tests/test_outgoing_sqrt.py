"""The square-root branch of a LOSSLESS medium: an exactly-zero imaginary part of the argument is
read as +0.0, so a negative real argument gives the decaying root +i sqrt(a), never the growing
-i sqrt(a), whatever the sign of the zero (optics.resonance._outgoing_sqrt, the one rule behind
_kz and every end-medium root that does not flip onto Im >= 0 itself).

IEEE 754 keeps the sign of a zero, and the principal root of complex(-a, -0.0) is -i sqrt(a). A
lossless medium carries Im = -0.0 easily: np.conj of a real permittivity; complex(x, -0.0); a
negative real permittivity times a positive real (-20 * (+0.0) = -0.0), so a lossless metal
written as eps = -20 - 0j keeps the -0.0 through eps k0^2 on every interpreter; the square of the
passive index tmm_reference._passive_sqrt returns for that eps, n = (-0.0, +sqrt(20)); and, from
Python 3.14 on, mixed complex-real arithmetic (complex(2.25, -0.0) * 2e-3 has Im -0.0 on 3.14 and
+0.0 on 3.13).

Every signed zero here is built EXPLICITLY on the square-root argument path (and each test checks
that it really reaches the argument), so the tests mean the same on every Python version. The one
exception is marked: the totally-reflected fundamental of rudnick_stern_flat_shg, whose -0.0 only
exists through 3.14's mixed arithmetic; on older interpreters that assertion holds before and
after the fix."""
import cmath
import math
import struct

import numpy as np
import pytest

from dynameta.constants import C_LIGHT
from dynameta.optics import resonance as rz

LAM = 1.0e-6
W = 2.0 * math.pi * C_LIGHT / LAM
K0 = W / C_LIGHT
EPS_M = -20.0                                   # a lossless metal (Re eps < 0, Im eps = 0)
FILM = [(complex(2.0, 0.5), 60e-9)]             # a lossy film on that metal


def _bits(z) -> bytes:
    z = complex(z)
    return struct.pack("<dd", z.real, z.imag)


def _is_neg_zero(x: float) -> bool:
    return x == 0.0 and math.copysign(1.0, x) < 0.0


def _metal_indices():
    """(n with +0.0, n with -0.0 real part): the two spellings of the passive index of eps = -20.
    The second is what the library's own tmm_reference._passive_sqrt returns for eps = -20 - 0j;
    its square carries Im = -0.0 (checked), the value the end-medium kz is built from."""
    from dynameta.optics.tmm_reference import _passive_sqrt
    n_pos = complex(0.0, math.sqrt(-EPS_M))
    n_neg = _passive_sqrt(complex(EPS_M, -0.0))
    assert _bits(n_neg) == _bits(complex(-0.0, math.sqrt(-EPS_M)))
    assert _is_neg_zero((n_neg ** 2).imag) and not _is_neg_zero((n_pos ** 2).imag)
    return n_pos, n_neg


# ------------------------------------------------------------------------------------------------
# the kernel
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("eps, k0, kp", [
    (complex(EPS_M, -0.0), 1.2e7, 3.0e6),                       # float k0, float k_par
    (complex(EPS_M, -0.0), complex(1.2e7, 0.0), 3.0e6),         # complex k0 with +0.0
    (complex(EPS_M, -0.0), complex(1.2e7, 0.0), complex(3.0e6, 0.0)),
    (complex(2.25, -0.0), complex(2e-3, -0.0), 5e-3),           # dielectric, evanescent order
    (np.complex128(complex(EPS_M, -0.0)), np.float64(1.2e7), np.float64(3.0e6)),
])
def test_negative_zero_argument_takes_the_decaying_root(eps, k0, kp):
    arg = eps * k0 * k0 - kp * kp
    assert arg.real < 0.0 and _is_neg_zero(arg.imag)            # the trap reaches the argument
    assert cmath.sqrt(arg).imag < 0.0                           # bare principal root: growing
    kz = rz._kz(eps, k0, kp)
    assert _bits(kz) == _bits(cmath.sqrt(complex(arg.real, 0.0)))
    assert kz.imag > 0.0 and kz.real == 0.0


def test_outgoing_sqrt_reads_a_zero_imaginary_part_as_positive():
    assert _bits(rz._outgoing_sqrt(complex(-4.0, -0.0))) == _bits(complex(0.0, 2.0))
    assert _bits(rz._outgoing_sqrt(complex(-4.0, 0.0))) == _bits(complex(0.0, 2.0))
    assert _bits(rz._outgoing_sqrt(complex(4.0, -0.0))) == _bits(complex(2.0, 0.0))
    assert _bits(rz._outgoing_sqrt(complex(-0.0, -0.0))) == _bits(complex(0.0, 0.0))
    assert _bits(rz._outgoing_sqrt(-4.0)) == _bits(complex(0.0, 2.0))      # a real float argument
    # a genuine gain argument (Im < 0) keeps the principal branch: Re >= 0, Im < 0
    g = rz._outgoing_sqrt(complex(-4.0, -1e-3))
    assert g.real > 0.0 and g.imag < 0.0
    assert _bits(g) == _bits(cmath.sqrt(complex(-4.0, -1e-3)))


def _argument_classes(rng):
    """(eps, k0, k_par) triples over the classes the kernel sees, plus classes it should not."""
    out = []
    n = 400
    k0r = rng.uniform(1e6, 2e7, n)
    for i in range(n):
        k0 = float(k0r[i])
        kp_prop = float(rng.uniform(0.0, 0.9)) * k0
        kp_evan = float(rng.uniform(1.1, 4.0)) * k0
        re = float(rng.uniform(-60.0, 60.0))
        out += [
            (complex(re, float(rng.uniform(1e-6, 10.0))), k0, kp_prop),            # lossy
            (complex(re, float(rng.uniform(1e-6, 10.0))), k0, kp_evan),
            (complex(abs(re) + 1.0, 0.0), k0, kp_prop),                           # lossless, +0.0
            (complex(abs(re) + 1.0, 0.0), k0, kp_evan),
            (complex(-abs(re) - 1.0, 0.0), k0, kp_prop),
            (complex(re, -float(rng.uniform(1e-6, 10.0))), k0, kp_evan),          # gain
            (complex(re, float(rng.choice([1e-300, -1e-300, 5e-324, -5e-324, 1e-200]))), k0,
             kp_evan),                                                            # tiny Im
            (complex(re, float(rng.uniform(-1.0, 1.0))) * 1e284, k0, kp_evan),    # |arg| ~ 1e300
            (complex(re, float(rng.uniform(0.0, 1.0))),
             complex(k0, float(rng.uniform(-1e5, 1e5))),
             complex(kp_evan, float(rng.uniform(-1e4, 1e4)))),                    # complex k0/k_par
            (np.complex128(complex(re, float(rng.uniform(-1.0, 1.0)))), np.float64(k0),
             np.float64(kp_evan)),                                                # numpy scalars
            # signed-zero inputs: the argument may or may not carry -0.0 (interpreter-dependent
            # for a positive eps with a float k0); both outcomes are checked below
            (complex(re, -0.0), k0, kp_evan),
            (complex(abs(re) + 1.0, -0.0), complex(k0, -0.0), kp_evan),
        ]
    return out


def test_kz_is_bit_identical_to_the_principal_root_except_for_a_negative_zero():
    """_kz == cmath.sqrt(eps k0^2 - k_par^2) bit for bit on every argument whose imaginary part is
    not an exact -0.0 (lossy, lossless with +0.0, gain, tiny / denormal Im, magnitudes near 1e300,
    complex k0 / k_par, numpy scalars); on an exact -0.0 it equals the root of the +0.0 argument."""
    rng = np.random.default_rng(20261116)
    same = flipped = 0
    for eps, k0, kp in _argument_classes(rng):
        arg = eps * k0 * k0 - kp * kp
        new = rz._kz(eps, k0, kp)
        if _is_neg_zero(arg.imag):
            assert _bits(new) == _bits(cmath.sqrt(complex(arg.real, 0.0))), (eps, k0, kp)
            if arg.real < 0.0:
                assert new.imag > 0.0
            flipped += 1
        else:
            assert _bits(new) == _bits(cmath.sqrt(arg)), (eps, k0, kp)
            same += 1
    assert same >= 4000 and flipped >= 400, (same, flipped)       # both classes really exercised


# ------------------------------------------------------------------------------------------------
# public entry points: a lossless metal half-space written with either zero
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("theta_deg", [0.0, 40.0])
def test_layered_smatrix_on_a_lossless_metal_does_not_depend_on_the_sign_of_zero(pol, theta_deg):
    """Before the rule the -0.0 spelling took the growing substrate wave: R 0.990 instead of 0.854
    (s, normal incidence). The +0.0 answer is the physical one: it is the lossless limit of the
    passive metal eps = -20 + 1e-9 i. |dR| is linear in that Im(eps) (measured max over these four
    cases 2.54e-11 at 1e-9, 2.54e-12 at 1e-10); bar 1e-9, while the growing root misses by 0.14."""
    n_pos, n_neg = _metal_indices()
    n_lossy = cmath.sqrt(complex(EPS_M, 1e-9))
    th = math.radians(theta_deg)
    a = rz.layered_smatrix_complex(W, FILM, theta_rad=th, pol=pol, n_super=1.0, n_sub=n_pos)
    b = rz.layered_smatrix_complex(W, FILM, theta_rad=th, pol=pol, n_super=1.0, n_sub=n_neg)
    c = rz.layered_smatrix_complex(W, FILM, theta_rad=th, pol=pol, n_super=1.0, n_sub=n_lossy)
    assert _bits(a.r) == _bits(b.r) and _bits(a.t) == _bits(b.t)
    assert a.R == b.R and np.array_equal(a.S, b.S)
    assert abs(a.R - c.R) < 1e-9, abs(a.R - c.R)
    # the pole function on the real axis, same medium
    kp = K0 * math.sin(th)
    d_pos = rz.smatrix_pole_func(FILM, pol=pol, n_super=1.0, n_sub=n_pos, k_par_m=kp)(W)
    d_neg = rz.smatrix_pole_func(FILM, pol=pol, n_super=1.0, n_sub=n_neg, k_par_m=kp)(W)
    assert _bits(d_pos) == _bits(d_neg)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_nonlocal_and_film_stacks_on_a_lossless_metal(pol):
    from dynameta.optics import nonlocal_tmm as nt
    from dynameta.optics.quantum_film import film_stack_rt
    n_pos, n_neg = _metal_indices()
    lay = [nt.DielectricLayer(complex(2.0, 0.5), 60e-9)]
    th = math.radians(30.0)
    a = nt.stack_rt(W, lay, pol=pol, n_super=1.0, n_sub=n_pos, theta_rad=th)
    b = nt.stack_rt(W, lay, pol=pol, n_super=1.0, n_sub=n_neg, theta_rad=th)
    assert _bits(a.r) == _bits(b.r) and a.R == b.R and a.A == b.A
    if pol == "p":
        eye = np.eye(2, dtype=complex)                           # a film of zero thickness
        fa = film_stack_rt(W, eye, lay, (), n_super=1.0, n_sub=n_pos, theta_rad=th)
        fb = film_stack_rt(W, eye, lay, (), n_super=1.0, n_sub=n_neg, theta_rad=th)
        assert _bits(fa.r) == _bits(fb.r) and fa.R == fb.R


def test_film_cell_substrate_admittance_on_a_lossless_metal():
    """quantum_film_cell.stack_admittance takes the substrate PERMITTIVITY; -20 - 0j used to give
    the opposite-sign admittance of the growing wave."""
    from dynameta.optics.quantum_film_cell import stack_admittance
    q = 1.3 * K0
    ya = stack_admittance([], W, q, complex(EPS_M, 0.0))[0]
    yb = stack_admittance([], W, q, complex(EPS_M, -0.0))[0]
    assert _bits(ya) == _bits(yb)


def test_flat_shg_oracle_on_a_lossless_metal():
    """rudnick_stern_flat_shg with eps_w = -20 - 0j gives exactly the eps_w = -20 + 0j result (the
    metal's index was the gain-signed -i sqrt(20), which conjugated the phase of E_perp_in and A)."""
    from dynameta.optics.shg_fem import rudnick_stern_flat_shg
    eps_2w, chi = complex(-5.0, 0.3), 1e-20
    a = rudnick_stern_flat_shg(LAM, 45.0, complex(EPS_M, 0.0), eps_2w, chi)
    b = rudnick_stern_flat_shg(LAM, 45.0, complex(EPS_M, -0.0), eps_2w, chi)
    for key in ("E_perp_in", "A", "beta1", "beta2"):
        assert _bits(a[key]) == _bits(b[key]), key
    assert a["S_up"] == b["S_up"]


def test_flat_shg_oracle_under_total_reflection_of_the_fundamental():
    """A real eps_w below sin^2(theta) totally reflects the fundamental, and the transmitted
    evanescent field must DECAY into the medium: E_perp_in is the limit from the passive side
    eps_w + 1e-12 i (|d| linear in that Im: measured 1.57e-12 at 1e-12, 1.57e-13 at 1e-13; bar
    1e-10) and not its conjugate (|d| = 2.575, what the growing root gave). The -0.0 here arises
    only through Python >= 3.14 mixed arithmetic (1.0 - complex keeps the sign of a zero), so on
    older interpreters this holds before the fix as well."""
    from dynameta.optics.shg_fem import rudnick_stern_flat_shg
    eps_2w, chi = complex(-5.0, 0.3), 1e-20
    th = 53.13
    assert 0.09 < math.sin(math.radians(th)) ** 2               # below the critical permittivity
    e_real = rudnick_stern_flat_shg(LAM, th, 0.09, eps_2w, chi)["E_perp_in"]
    e_lossy = rudnick_stern_flat_shg(LAM, th, complex(0.09, 1e-12), eps_2w, chi)["E_perp_in"]
    assert abs(e_real - e_lossy) < 1e-10, abs(e_real - e_lossy)
    assert abs(e_real - e_lossy.conjugate()) > 1.0


def test_enz_reflector_mirror_index_is_passive_for_a_lossless_metal():
    """The mirror index of enz_reflector_stack: eps_mirror = -20 - 0j used to give n = -i sqrt(20),
    which the lumenairy bridges refuse as a gain half-space (Im n < 0). A lossy mirror's index is
    unchanged bit for bit."""
    from dynameta.transient_optics import enz_reflector_stack
    a = enz_reflector_stack(complex(1.0, 0.3), LAM, eps_mirror=complex(EPS_M, 0.0))
    b = enz_reflector_stack(complex(1.0, 0.3), LAM, eps_mirror=complex(EPS_M, -0.0))
    assert _bits(a.n_sub) == _bits(b.n_sub) == _bits(cmath.sqrt(complex(EPS_M, 0.0)))
    assert b.n_sub.imag > 0.0
    lossy = enz_reflector_stack(complex(1.0, 0.3), LAM)              # default -120 + 3i mirror
    assert _bits(lossy.n_sub) == _bits(np.sqrt(complex(-120.0, 3.0)))


def test_region_index_advice_of_the_lumenairy_translation_is_passive_for_a_lossless_metal():
    """optical_model_to_lumenairy_eps documents how to lift its permittivity callable to a lumenairy
    REGION index: np.sqrt(fn(wl) + 0j). For a lossless metal whose eps carries Im = -0.0 that is the
    decaying (passive) index +i sqrt(20), bit-identical to _outgoing_sqrt and accepted by the bridges'
    gain guard; the bare np.sqrt(fn(wl)) gives the gain-signed -i sqrt(20), which the guard refuses
    (the control). A lossy eps gets the same index from both."""
    from dynameta.materials import ConstantOptical
    from dynameta.optics.lumenairy_bridge._common import refuse_gain_half_space
    from dynameta.optics.lumenairy_bridge.translate import optical_model_to_lumenairy_eps
    fn = optical_model_to_lumenairy_eps(ConstantOptical(complex(EPS_M, -0.0)))
    assert math.copysign(1.0, complex(fn(LAM)).imag) < 0.0       # the -0.0 reaches the callable's output
    n_rule, n_bare = complex(np.sqrt(fn(LAM) + 0j)), complex(np.sqrt(fn(LAM)))
    assert _bits(n_rule) == _bits(rz._outgoing_sqrt(fn(LAM)))
    assert n_rule.imag > 0.0 and n_bare.imag < 0.0
    refuse_gain_half_space("advice", 1.0 + 0j, n_rule)            # accepted
    with pytest.raises(ValueError, match="gain"):
        refuse_gain_half_space("bare sqrt", 1.0 + 0j, n_bare)
    lossy = optical_model_to_lumenairy_eps(ConstantOptical(complex(EPS_M, 0.4)))
    assert _bits(np.sqrt(lossy(LAM) + 0j)) == _bits(np.sqrt(lossy(LAM)))
