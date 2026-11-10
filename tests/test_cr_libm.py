"""tests/_cr_libm.py, the canonical (correctly rounded) transcendentals the call-log digests are recorded with
(tests/_calllog.py; docs/audit/2026-11-10-v0.15.1-digest-svd-fragility.md). Pure numpy + stdlib decimal."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cr_libm as C  # noqa: E402


def _decimal(key, args):
    """The definition: the decimal path alone (nan where it defers to the platform)."""
    return C._scalar_value(key, args, float("nan"))


def test_known_hard_case_is_correctly_rounded():
    """exp(-7.384693350077327): the Windows UCRT returns ...7206 (0.510 ulp from the exact value), glibc and
    the correctly rounded double ...7205 (MEASURED against mpmath at 200 bits, 2026-11-08 audit note). Both
    the decimal path and the vectorized fast path (here it defers to decimal or decides alike) give ...7205."""
    x = -7.384693350077327
    with C.correctly_rounded():
        assert math.exp(x) == 0.0006206809661957205
        assert float(np.exp(np.array([x]))[0]) == 0.0006206809661957205
        assert float(np.exp(x)) == 0.0006206809661957205


@pytest.mark.parametrize("key", ["exp", "log", "pow"])
def test_fast_path_equals_the_decimal_definition(key):
    """The double-double fast path only DECIDES early: every value it decides equals the decimal path's.
    Budget: exact equality (both are the correctly rounded double). MEASURED 2026-11-10: 0 mismatches on
    440017 random elements (scratch validation; this gate keeps 3000 per function) and on all 1388023
    elements of the bipolar default call log. REFUTE (MEASURED 2026-11-10, Windows): ln 2 without its low word
    (_LN2 = (hi, 0)) fails exp, log and pow (e.g. a 1-ulp-wrong value decided for exp(174.0)); the expm1 series cut
    at t^3 fails exp and pow; a decision bound of 2^-20 decides nothing and fails the coverage line.
    """
    rng = np.random.default_rng(20261110)
    n = 1000
    if key == "exp":
        cases = [(rng.uniform(-690.0, 690.0, n),), (rng.uniform(-1.0, 1.0, n) * 10.0 ** rng.uniform(-20, 0, n),),
                 (rng.uniform(-30.0, 30.0, n),)]
    elif key == "log":
        cases = [(10.0 ** rng.uniform(-300, 300, n),), (1.0 + rng.uniform(-1, 1, n) * 10.0 ** rng.uniform(-15, -1, n),),
                 (rng.uniform(0.0, 200.0, n),)]
    else:
        x = rng.uniform(0.0, 136.0, n)
        cases = [(x, np.full(n, 1.0 / 3.0)), (x, np.full(n, 4.0 / 3.0)),
                 (10.0 ** rng.uniform(-30, 30, n), rng.uniform(-5.0, 5.0, n))]
    decided = 0
    for args in cases:
        vals, dec = C._fast_values(key, list(args))
        for i in np.flatnonzero(dec):
            assert vals[i] == _decimal(key, tuple(float(a[i]) for a in args)), (key, [a[i] for a in args])
        decided += int(dec.sum())
    assert decided >= 0.99 * 3 * n                  # it decides (almost) everything: MEASURED >= 0.997


def test_tanh_and_refusals():
    """tanh (added for the DG seed) is correctly rounded through decimal: odd, tanh(0) = 0, saturates to 1.0
    exactly, and agrees with the mathematically equal expm1 form to the last bit on these arguments
    (MEASURED against mpmath at 300 bits on the DG fixture's 15 arguments: 0 misses; numpy's tanh 3). A
    transcendental with no correctly rounded version raises inside the context (fail closed)."""
    with C.correctly_rounded():
        assert float(np.tanh(np.array([0.0]))[0]) == 0.0
        assert math.tanh(41.0) == 1.0 and math.tanh(-41.0) == -1.0
        for x in (0.8440734755535655, 5.822158378615143, 6.989685097425106):    # numpy's 3 misses
            assert math.tanh(-x) == -math.tanh(x)
        assert math.tanh(0.8440734755535655) == 0.6879606208955744
        assert math.tanh(5.822158378615143) == 0.9999824626619203
        assert math.tanh(6.989685097425106) == 0.9999983022791165
        with pytest.raises(RuntimeError, match="no correctly rounded version"):
            np.sin(np.array([1.0]))
    assert np.sin(1.0) == math.sin(1.0)                  # restored on exit
