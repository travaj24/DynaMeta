"""The ONE implementation behind the DEVSIM call-log recorders (tests/_bipolar_calllog.py, _btbt_calllog.py,
_dg_calllog.py, _fdext_calllog.py): the interceptor that logs every definition call, the canonical-recording
context, the digest, and the number-masked skeleton.

A call log is the ordered list of DEVSIM model-, equation-, parameter- and solution-defining calls a fixture
makes, each as [name, sorted (keyword, repr(value)) pairs]; a recorder may append its own [name, repr(value)]
rows. Its sha256 (digest) pins a definition character for character.

CANONICAL FORM (2026-11-08 for the fdext log, 2026-11-10 for the other three). repr() of a float that came out
of a C-library transcendental (exp, log, pow, ...) is not fixed by IEEE-754: each libm may round a hard case
either way, so a raw digest can differ between Windows (UCRT) and Linux (glibc) without any change of the
code (MEASURED for the fdext log: docs/audit/2026-11-08-v0.15-linux-ci-exactness.md). recording(True) records
inside tests/_cr_libm.correctly_rounded(), where math / numpy exp, log, log10, log2, expm1, log1p and power
return the correctly rounded double: the log is then a platform-independent function of the code, and every
IEEE operation is still recorded bit for bit (a 1-ulp change of the code still changes the digest).
recording(False) uses the platform's libm (the raw log; two logs of one machine compare equal raw).
docs/audit/2026-11-10-v0.15.1-digest-svd-fragility.md has the measurements of the four pinned logs.

These helpers import nothing from dynameta, so a recorder (with this file and tests/_cr_libm.py copied next to
it) runs unchanged on an export of an older tree -- that is how each digest is re-recorded.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re

import _cr_libm


def recording(canonical: bool):
    """The context a log is recorded in: correctly rounded transcendentals (canonical) or the platform's."""
    return _cr_libm.correctly_rounded() if canonical else contextlib.nullcontext()


@contextlib.contextmanager
def intercepting(ds, names, log, value_repr=repr):
    """While active, every ds.<name> call (name in `names`) first appends [name, sorted (key, value_repr(v))
    pairs of its keywords] to `log`, then runs. The originals are restored on exit."""
    saved = {}

    def _wrap(name, fn):
        def inner(*a, **kw):
            log.append([name, sorted((k, value_repr(v)) for k, v in kw.items())])
            return fn(*a, **kw)
        return inner

    for nm in names:
        saved[nm] = getattr(ds, nm)
        setattr(ds, nm, _wrap(nm, saved[nm]))
    try:
        yield log
    finally:
        for nm, fn in saved.items():
            setattr(ds, nm, fn)


def digest(log) -> str:
    """sha256 of the JSON dump of the log (tuples and lists dump identically)."""
    return hashlib.sha256(json.dumps(log, sort_keys=False).encode("utf-8")).hexdigest()


_NUMBER = re.compile(r"(?<![A-Za-z_0-9.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def skeleton(log):
    """The log with every numeric literal replaced by '#': what the libm choice cannot change."""
    out = []
    for name, kw in log:
        if isinstance(kw, str):                               # a recorder's own [name, repr(value)] row
            out.append([name, _NUMBER.sub("#", kw)])
        else:
            out.append([name, [[k, _NUMBER.sub("#", v)] for k, v in kw]])
    return out
