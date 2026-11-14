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
module at 60 significant digits (199 bits): exp, expm1, log, log2, log10, log1p, pow / power and (added
2026-11-10 for the DG seed of tests/_dg_calllog.py) tanh. decimal's exp, ln and log10 are correctly rounded at the
context precision (Python docs, decimal.Decimal.exp / ln); the hardest-to-round binary64 arguments of exp
and log need fewer than 160 bits (Lefevre & Muller, "Worst cases for correct rounding of the elementary
functions in double precision", ARITH 15, 2001), so float() of the 60-digit value is the correctly rounded
double for EVERY argument. power goes through decimal's power at 60 digits ("almost always correctly
rounded" at that precision, i.e. to ~1e-60 relative -- far inside the binary64 rounding decision); tanh is
s (e^(2a) - 1)/(e^(2a) + 1) at 60 digits relative to the argument (same ~1e-58 accuracy: correctly rounded
unless the exact value lies that close to a rounding midpoint; MEASURED 2026-11-10: the 15 tanh arguments of
the DG fixture all agree with mpmath at 300 bits, where numpy's own tanh misses 3 by 0.57-0.68 ulp).
A log recorded inside the context is then a platform-independent function of the code: IEEE basic
operations are correctly rounded and the transcendentals are too. It still sees EVERY change of the code's
arithmetic (a re-association moves a literal by an ulp and changes the digest), which a tolerance on the
literals could not.

FAIL CLOSED. The transcendentals it does not implement (trigonometric, hyperbolic, erf, gamma, logaddexp,
...) raise inside the context: a recorded path that starts using one gets a clear error here instead of a
silently platform-dependent digest. Arguments outside the implemented domain (non-finite, x <= 0 for log /
pow, numpy's where= / complex) fall back to the platform function, whose result IEEE fixes there (inf,
nan, 0, exact powers) or raise.

COST. Through decimal an element costs 35-145 us on the dev box: the FD-statistics paths evaluate ~1.4e6
np.power elements (the Gauss-Legendre rule of physics_bipolar_dd.fd_potential_shift inside the FD
equilibrium root) and took 466 s (fdext) / 254 s (bipolar). Since 2026-11-10 numpy exp / log / power
arrays first go through a VECTORIZED double-double evaluation (the fast path below, Ziv's strategy: it only
decides early, decimal stays the definition): MEASURED on the bipolar default log, 1388006 of 1388023
elements decided there, 0 differences from decimal, the 17 others through decimal; that log records in
19 s instead of 1.4 s raw. Two logs compared inside one machine (one libm) still need no canonical form.

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
_MATH_CR = ("exp", "expm1", "log", "log2", "log10", "log1p", "pow", "tanh")
_MATH_REFUSED = ("exp2", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh",
                 "asinh", "acosh", "atanh", "erf", "erfc", "gamma", "lgamma", "cbrt")
_NP_CR = ("exp", "expm1", "log", "log2", "log10", "log1p", "power", "tanh")
_NP_REFUSED = ("exp2", "float_power", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh",
               "cosh", "arcsinh", "arccosh", "arctanh", "logaddexp", "logaddexp2", "cbrt")


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


def _tanh(x):
    """tanh x = s (e^(2a) - 1) / (e^(2a) + 1), a = |x|, s = sign x, at PRECISION digits RELATIVE to x (the working
    precision grows with -log10 a, as in _em1, so e^(2a) - 1 keeps its digits for tiny a). Added 2026-11-10: the
    unipolar DG seed (physics_density_gradient._seed_dg, the tests/_dg_calllog.py fixture) calls np.tanh."""
    d = _d(x)
    if d == 0:
        return d
    a = abs(d)
    if a > 40:                                    # 1 - tanh a < 2 e^-80 < 2^-115: the double is exactly 1
        return decimal.Decimal(1).copy_sign(d)
    ctx = decimal.Context(prec=PRECISION + max(0, -a.adjusted()), Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)
    em = ctx.subtract(ctx.exp(ctx.multiply(2, a)), 1)
    return _CTX.divide(em, ctx.add(em, 2)).copy_sign(d)


_SCALAR = {
    "exp": _guard(lambda x: _CTX.exp(_d(x))),
    "expm1": _guard(_em1),
    "log": _guard(lambda x: _CTX.ln(_d(x))),
    "log2": _guard(lambda x: _CTX.divide(_CTX.ln(_d(x)), _CTX.ln(decimal.Decimal(2)))),
    "log10": _guard(lambda x: _CTX.log10(_d(x))),
    "log1p": _guard(_ln1p),
    "pow": _guard(lambda x, y: _CTX.power(_d(x), _d(y))),
    "tanh": _guard(_tanh),
}
_DOMAIN = {"exp": lambda x: True, "expm1": lambda x: True, "tanh": lambda x: True, "log": lambda x: x > 0.0, "log2": lambda x: x > 0.0,
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


# ---- vectorized fast path for numpy exp / log / power (Ziv's strategy) -----------------------------------
# Each element is evaluated in double-double (dd: an unevaluated sum hi + lo of two doubles, ~106 bits) with
# IEEE basic operations only, so the dd value is the same on every platform. Its error is bounded a priori
# (below); where the whole interval value +- bound rounds to ONE double, that double IS the correctly rounded
# result (the decimal path would return it too). Elements whose interval straddles a rounding midpoint, and
# arguments outside the fast domain, go through the decimal path above. The fast path only DECIDES early: the
# definition of the canonical value is unchanged.
# Error budget (relative, a priori; dd basic operations are accurate to ~2^-104 each):
#   log_dd(x): x = m 2^e, m in [sqrt(1/2), sqrt(2)); ln m = 2 atanh(f), f = (m-1)/(m+1), |f| <= 0.1716, by a
#     dd series truncated at f^43 (remainder < 0.0295^22 < 2^-111 relative); e ln2 with a 107-bit ln2. No
#     cancellation (|ln m| <= 0.347 < ln2 <= |e ln2| when e != 0): relative error < 40 ops x 2^-104 < 2^-98.
#   exp_dd(z), |z| <= 690: z - k ln2 (|k| <= 996, absolute error < 2^-96), scaled by 2^-10, Taylor expm1 to
#     t^11 (|t| <= 3.4e-4: remainder < 2^-150), squared back 10 times (relative error doubles per squaring:
#     < 2^10 x 8 x 2^-104 = 2^-91): relative error < 2^-90. Results below 2^-960 are not decided here (the
#     low word's subnormal rounding would cost up to 2^-1075 / 2^-960 = 2^-115).
#   pow_dd(x, y) = exp_dd(y log_dd(x)): the argument's relative error 2^-97 times |z| <= 690 adds < 2^-87.
# The decision uses BOUND_REL = 2^-72 x (1 + |z|): >= 2^15 times every budget above (the margin covers an
# error in this analysis, not a known term). It sends ~2^-18 of uniformly placed results to decimal.
# MEASURED 2026-11-10 against the decimal path (tests/test_cr_libm.py and the audit note
# docs/audit/2026-11-10-v0.15.1-digest-svd-fragility.md): 0 mismatches.
_BOUND_LOG2 = -72
_FAST = ("exp", "log", "pow")
FAST_PATH = True                                            # False: every element through decimal (the check)
_SPLITTER = 134217729.0                                     # 2^27 + 1 (Veltkamp)


def _dd_const(dec):
    hi = float(dec)
    return hi, float(_CTX.subtract(dec, decimal.Decimal(hi)))


_LN2 = _dd_const(_CTX.ln(decimal.Decimal(2)))
_INV_FACT = [_dd_const(_CTX.divide(1, math.factorial(k))) for k in range(12)]        # 1/k!, k = 0..11
_INV_ODD = [_dd_const(_CTX.divide(1, 2 * k + 1)) for k in range(22)]                 # 1/(2k+1), k = 0..21


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _fast_two_sum(a, b):                                    # requires |a| >= |b| (or a == 0)
    s = a + b
    return s, b - (s - a)


def _two_prod(a, b):
    p = a * b
    t = _SPLITTER * a
    ah = t - (t - a)
    t = _SPLITTER * b
    bh = t - (t - b)
    al, bl = a - ah, b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dd_add(ah, al, bh, bl):
    s, e = _two_sum(ah, bh)
    t, f = _two_sum(al, bl)
    s, e = _fast_two_sum(s, e + t)
    return _fast_two_sum(s, e + f)


def _dd_mul(ah, al, bh, bl):
    p, e = _two_prod(ah, bh)
    return _fast_two_sum(p, e + (ah * bl + al * bh))


def _dd_div(ah, al, bh, bl):
    q1 = ah / bh
    ph, pl = _dd_mul(q1, 0.0, bh, bl)
    rh, rl = _dd_add(ah, al, -ph, -pl)
    q2 = rh / bh
    ph, pl = _dd_mul(q2, 0.0, bh, bl)
    rh, rl = _dd_add(rh, rl, -ph, -pl)
    q1, q2 = _fast_two_sum(q1, q2)
    return _dd_add(q1, q2, rh / bh, 0.0)


def _log_dd(x):
    """ln x in dd for normal positive x (error budget above)."""
    m, e = np.frexp(x)                                       # x = m 2^e exactly, m in [0.5, 1)
    low = m < 0.7071067811865476
    m = np.where(low, 2.0 * m, m)
    e = np.where(low, e - 1, e).astype(np.float64)
    nh, nl = m - 1.0, np.zeros_like(m)                       # exact (Sterbenz)
    dh, dl = _two_sum(m, 1.0)
    fh, fl = _dd_div(nh, nl, dh, dl)
    f2h, f2l = _dd_mul(fh, fl, fh, fl)
    sh, sl = _INV_ODD[-1][0] + 0.0 * m, np.zeros_like(m)
    for c in reversed(_INV_ODD[:-1]):                        # Horner in f^2
        sh, sl = _dd_mul(sh, sl, f2h, f2l)
        sh, sl = _dd_add(sh, sl, c[0], c[1])
    sh, sl = _dd_mul(sh, sl, fh, fl)
    lh, ll = 2.0 * sh, 2.0 * sl                              # ln m = 2 atanh(f)
    kh, kl = _two_prod(e, _LN2[0])
    kl = kl + e * _LN2[1]
    return _dd_add(kh, kl, lh, ll)


def _exp_dd(zh, zl):
    """exp(zh + zl) in dd for |zh| <= 690 (error budget above)."""
    k = np.rint(zh / _LN2[0])
    ph, pl = _two_prod(k, _LN2[0])
    pl = pl + k * _LN2[1]
    rh, rl = _dd_add(zh, zl, -ph, -pl)
    th, tl = rh * 2.0 ** -10, rl * 2.0 ** -10                 # exact scaling
    sh, sl = _INV_FACT[-1][0] + 0.0 * th, np.zeros_like(th)
    for c in reversed(_INV_FACT[1:-1]):                      # expm1(t) = t (1 + t/2! + ... ) by Horner
        sh, sl = _dd_mul(sh, sl, th, tl)
        sh, sl = _dd_add(sh, sl, c[0], c[1])
    eh, el = _dd_mul(sh, sl, th, tl)
    for _ in range(10):                                      # (1 + e)^2 - 1 = 2e + e^2
        qh, ql = _dd_mul(eh, el, eh, el)
        eh, el = _dd_add(2.0 * eh, 2.0 * el, qh, ql)
    oh, ol = _dd_add(1.0, 0.0, eh, el)
    ki = k.astype(np.int64)
    return np.ldexp(oh, ki), np.ldexp(ol, ki)


def _fast_values(key, ins):
    """(values, decided) for float64 arrays: the correctly rounded value where the dd evaluation decides it."""
    x = ins[0].reshape(-1)
    with np.errstate(all="ignore"):
        ok = np.isfinite(x) & (x >= 2.0 ** -1000) & (x <= 2.0 ** 1000)
        if key == "exp":
            ok = np.isfinite(x) & (np.abs(x) <= 690.0)
        xs = np.where(ok, x, 1.0)
        if key == "exp":
            zh, zl = xs, np.zeros_like(xs)
        else:
            lh, ll = _log_dd(xs)
            if key == "log":
                zh, zl = lh, ll
            else:
                y = ins[1].reshape(-1)
                ok &= np.isfinite(y)
                ys = np.where(ok, y, 0.0)
                zh, zl = _two_prod(ys, lh)
                zh, zl = _fast_two_sum(zh, zl + ys * ll)
                ok &= np.abs(zh) <= 690.0
                zh, zl = np.where(ok, zh, 0.0), np.where(ok, zl, 0.0)
        if key == "log":
            vh, vl, mag = zh, zl, np.ones_like(zh)
        else:
            vh, vl = _exp_dd(zh, zl)
            mag = 1.0 + np.abs(zh)
        c = vh + vl                                          # the double nearest the dd value
        ok &= np.isfinite(c) & (np.abs(c) >= 2.0 ** -960) & (np.abs(c) <= 2.0 ** 1000)
        cs = np.where(ok, c, 1.0)
        d = (vh - cs) + vl                                   # dd value - c (vh - c exact: Sterbenz)
        bound = np.abs(cs) * mag * 2.0 ** _BOUND_LOG2
        up = 0.5 * (np.nextafter(cs, np.inf) - cs)           # distance to the upper / lower midpoint
        dn = 0.5 * (cs - np.nextafter(cs, -np.inf))
        decided = ok & (d + bound < up) & (d - bound > -dn)
    return c, decided


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
        if key in _FAST and FAST_PATH:
            vals, decided = _fast_values(key, ins)
            vals = vals.copy()
            todo = np.flatnonzero(~decided)
        else:
            vals, todo = np.empty(flat.size), range(flat.size)
        for i in todo:
            vals[i] = _scalar_value(key, tuple(float(a.flat[i]) for a in ins), float(flat[i]))
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
