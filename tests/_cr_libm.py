"""Correctly rounded transcendentals for the call-log recorders: the CANONICAL form of a recorded log.

WHY. A call-log gate (tests/_fdext_calllog.py) hashes repr() of every float the physics modules hand to
DEVSIM. Most of those floats are IEEE-754 basic operations (+ - * / sqrt) on typed inputs: correctly
rounded, so the same bits on every platform. The rest pass through a C-library transcendental (math.exp,
np.exp, np.log, np.power, ...), whose last bit IEEE-754 does not fix: each libm may round a hard case
either way. MEASURED 2026-11-08: exp(-7.384693350077327) -- interface-trap bin 14's n1 / n_i -- is
0.0006206809661957206 from the Windows UCRT (0.510 ulp from the exact value) and 0.0006206809661957205
from glibc (0.490 ulp: the correctly rounded double); the 501-call legacy log differed in that one literal
(12 occurrences in 9 calls) and its digest differed (docs/audit/2026-11-08-v0.15-linux-ci-exactness.md).

WHAT. `correctly_rounded()` is a context manager that, while active, replaces the transcendental entry
points of the `math` module and of numpy by CORRECTLY ROUNDED versions computed with the stdlib `decimal`
module at 60 significant digits (199 bits). decimal's exp, ln and log10 are correctly rounded at the
context precision (Python docs, decimal.Decimal.exp / ln); the hardest-to-round binary64 arguments of exp
and log need fewer than 160 bits (Lefevre & Muller, "Worst cases for correct rounding of the elementary
functions in double precision", ARITH 15, 2001), so float() of the 60-digit value is the correctly rounded
double for EVERY argument. power goes through decimal's power at 60 digits ("almost always correctly
rounded" at that precision, i.e. to ~1e-60 relative -- far inside the binary64 rounding decision).
A log recorded inside the context is then a platform-independent function of the code: IEEE basic
operations are correctly rounded and the transcendentals are too. It still sees EVERY change of the code's
arithmetic (a re-association moves a literal by an ulp and changes the digest), which a tolerance on the
literals could not.

FAIL CLOSED. The transcendentals it does not implement (trigonometric, hyperbolic, erf, gamma, logaddexp,
...) raise inside the context: a recorded path that starts using one gets a clear error here instead of a
silently platform-dependent digest. Arguments outside the implemented domain (non-finite, x <= 0 for log /
pow, numpy's where= / complex) fall back to the platform function, whose result IEEE fixes there (inf,
nan, 0, exact powers) or raise.

COST. Each element goes through decimal (~35 us on the dev box): the fdext legacy log (~2.5e5 elements,
inferred from the time) records in 24 s instead of 15 s, but the FD-statistics path evaluates ~2e6
np.power elements (its FD tables) and takes 466 s -- record a path canonically only where the digest
crosses platforms; two logs compared inside one process (one libm) need no canonical form.

NOT COVERED (cannot be intercepted): Python's float ** operator and numpy's ndarray ** operator call the
C pow() directly, and a function bound before the context (from math import exp) keeps the platform
version. tests/test_fd_mode_complete.py lists the ones the fdext fixture reaches (MEASURED equal on
Windows UCRT and glibc 2.39: the canonical logs are bitwise identical).
"""
from __future__ import annotations

import contextlib
import decimal
import math

import numpy as np

PRECISION = 60
_CTX = decimal.Context(prec=PRECISION, rounding=decimal.ROUND_HALF_EVEN,
                       Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN, traps=[])

# entry points replaced by a correctly rounded version / made to raise inside the context
_MATH_CR = ("exp", "expm1", "log", "log2", "log10", "log1p", "pow")
_MATH_REFUSED = ("exp2", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
                 "asinh", "acosh", "atanh", "erf", "erfc", "gamma", "lgamma", "cbrt")
_NP_CR = ("exp", "expm1", "log", "log2", "log10", "log1p", "power")
_NP_REFUSED = ("exp2", "float_power", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh",
               "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "logaddexp", "logaddexp2", "cbrt")


def _d(x: float) -> decimal.Decimal:
    return decimal.Decimal(float(x))                      # exact: every double is a finite decimal


def _guard(fn):
    """Correctly rounded value of fn on finite input with a finite, nonzero platform result; else None."""
    def cr(*a):
        try:
            v = float(fn(*a))
        except (decimal.InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
            return None
        return v if math.isfinite(v) else None
    return cr


def _ln1p(x):
    one_plus = decimal.Context(prec=PRECISION + max(0, -_d(x).adjusted())).add(1, _d(x))   # 1 + x exactly
    return _CTX.ln(one_plus)


def _em1(x):
    ctx = decimal.Context(prec=PRECISION + max(0, -_d(x).adjusted()), Emax=decimal.MAX_EMAX,
                          Emin=decimal.MIN_EMIN)
    return ctx.subtract(ctx.exp(_d(x)), 1)


_SCALAR = {
    "exp": _guard(lambda x: _CTX.exp(_d(x))),
    "expm1": _guard(_em1),
    "log": _guard(lambda x: _CTX.ln(_d(x))),
    "log2": _guard(lambda x: _CTX.divide(_CTX.ln(_d(x)), _CTX.ln(decimal.Decimal(2)))),
    "log10": _guard(lambda x: _CTX.log10(_d(x))),
    "log1p": _guard(_ln1p),
    "pow": _guard(lambda x, y: _CTX.power(_d(x), _d(y))),
}
_DOMAIN = {"exp": lambda x: True, "expm1": lambda x: True, "log": lambda x: x > 0.0, "log2": lambda x: x > 0.0,
           "log10": lambda x: x > 0.0, "log1p": lambda x: x > -1.0}


_LOGB = _guard(lambda a, b: _CTX.divide(_CTX.ln(_d(a)), _CTX.ln(_d(b))))


def _scalar_value(name, args, platform_value):
    """The correctly rounded value where it is defined and finite, else the platform's."""
    if not all(math.isfinite(a) for a in args):
        return platform_value
    if name == "pow":
        if not args[0] > 0.0:
            return platform_value
    elif not _DOMAIN[name](args[0]):
        return platform_value
    v = _SCALAR[name](*args)
    return platform_value if v is None else v


def _math_wrapper(name, orig):
    if name == "log":
        def log(x, *base):
            r = orig(x, *base)                                # the platform's errors / specials first
            if not base:
                return _scalar_value("log", (float(x),), r)
            a, b = float(x), float(base[0])
            if not (math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0 and b != 1.0):
                return r
            v = _LOGB(a, b)
            return r if v is None else v
        return log

    def f(*args):
        r = orig(*args)
        return _scalar_value(name, tuple(float(a) for a in args), r)
    f.__name__ = name
    return f


def _np_wrapper(name, orig):
    key = "pow" if name == "power" else name

    def f(*args, **kw):
        if "where" in kw or set(kw) - {"out", "dtype"}:
            raise RuntimeError("_cr_libm: np.{} called with {}: not canonicalized".format(name, sorted(kw)))
        r = orig(*args, **kw)
        ra = np.asarray(r)
        if ra.dtype != np.float64:
            if ra.dtype.kind in "iub":                       # integer power: exact, platform-independent
                return r
            raise RuntimeError("_cr_libm: np.{} on dtype {}: not canonicalized".format(name, ra.dtype))
        ins = [np.broadcast_to(np.asarray(a, dtype=np.float64), ra.shape) for a in args[:2 if key == "pow" else 1]]
        flat = ra.reshape(-1)
        vals = np.array([_scalar_value(key, tuple(float(a.flat[i]) for a in ins), float(flat[i]))
                         for i in range(flat.size)], dtype=np.float64)
        if isinstance(r, np.ndarray):
            r[...] = vals.reshape(ra.shape)                    # in place: covers out=
            return r
        return np.float64(vals[0])
    f.__name__ = name
    return f


def _refuse(where, name):
    def f(*a, **k):
        raise RuntimeError("_cr_libm: {}.{} has no correctly rounded version here -- add one to "
                           "tests/_cr_libm.py before recording a path that calls it".format(where, name))
    f.__name__ = name
    return f


@contextlib.contextmanager
def correctly_rounded():
    """Replace math / numpy transcendentals by correctly rounded ones (and refuse the rest) while active."""
    saved = []
    try:
        for mod, where, cr_names, refused, wrap in ((math, "math", _MATH_CR, _MATH_REFUSED, _math_wrapper),
                                                    (np, "np", _NP_CR, _NP_REFUSED, _np_wrapper)):
            for nm in cr_names + refused:
                if not hasattr(mod, nm):
                    continue
                orig = getattr(mod, nm)
                saved.append((mod, nm, orig))
                setattr(mod, nm, wrap(nm, orig) if nm in cr_names else _refuse(where, nm))
        yield
    finally:
        for mod, nm, orig in reversed(saved):
            setattr(mod, nm, orig)
