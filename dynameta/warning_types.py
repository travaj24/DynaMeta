"""DynaMeta's warning categories (audit V-17, 2026-11-02).

Top-level (like `dynameta.constants`) and stdlib-only ON PURPOSE: materials/core/optics modules
import it at module scope, and placing it under `dynameta.core` would run core/__init__ (which
imports materials) from inside materials -- the circular import the first draft hit.

Before this module the library's ~100 `warnings.warn` sites used bare `UserWarning` (55) and
`RuntimeWarning` (40) with no consistent meaning, plus two module-local classes
(`optics.solver.FEMDiagnosticWarning`, `optics.soa.metrics.TransferFitWarning`). A caller could
not answer the one question a warning category exists for -- "is the number I got back still
trustworthy?" -- without parsing message text, and could not silence DynaMeta's advisories without
also silencing NumPy's.

The hierarchy is deliberately SMALL; a category exists only where it changes what the caller
should do:

    DynaMetaWarning(UserWarning)                     every warning DynaMeta itself raises
    +-- UnreliableResultWarning(.., RuntimeWarning)  the RETURNED VALUE may be wrong: a solve or fit
    |                                                did not converge, a grid/window does not
    |                                                resolve the answer, a model left its validity
    |                                                range, or a value was clamped to stay physical
    +-- AdvisoryWarning                              the result is fine; a choice was made for the
    |                                                caller (fallback, ignored option, regime note)
    +-- DynaMetaDeprecationWarning(.., DeprecationWarning)  a renamed spelling still accepted

COMPATIBILITY RULE (why the multiple inheritance): every site that adopts a category here keeps
its OLD category as a base class, so an existing `pytest.warns(RuntimeWarning)`,
`filterwarnings('ignore', category=UserWarning)` or `except RuntimeWarning` keeps matching. That is
why `UnreliableResultWarning` is BOTH a UserWarning (via DynaMetaWarning) and a RuntimeWarning --
the sites that adopted it used both -- while `AdvisoryWarning` is adopted only at sites that were
plain UserWarning. Messages were NOT reworded.

Usage:

    warnings.filterwarnings("error", category=UnreliableResultWarning)  # fail fast in a sweep
    warnings.filterwarnings("ignore", category=AdvisoryWarning)         # quiet the notes
    warnings.filterwarnings("ignore", category=DynaMetaWarning)         # every DynaMeta warning

The module-local categories that predate this file are rebased onto it without being renamed:
`FEMDiagnosticWarning` (solve_fem's advisory/quality flags -- both kinds, kept as one class
because callers already filter it) derives from DynaMetaWarning, and `TransferFitWarning` from
UnreliableResultWarning. Stdlib-only, so it is importable from every tier.
"""

from __future__ import annotations

__all__ = ["DynaMetaWarning", "UnreliableResultWarning", "AdvisoryWarning",
           "DynaMetaDeprecationWarning"]


class DynaMetaWarning(UserWarning):
    """Base class of every warning category DynaMeta raises (audit V-17)."""


class UnreliableResultWarning(DynaMetaWarning, RuntimeWarning):
    """The returned value may be WRONG: non-convergence, an unresolved grid/window, a model used
    outside its validity range, or a value clamped to keep it physical. Escalate it to an error
    (`filterwarnings('error', category=UnreliableResultWarning)`) to fail fast in a sweep."""


class AdvisoryWarning(DynaMetaWarning):
    """The result is trustworthy; DynaMeta made a choice on the caller's behalf (a fallback, an
    ignored or overridden option, a cheaper equivalent formulation is available)."""


class DynaMetaDeprecationWarning(DynaMetaWarning, DeprecationWarning):
    """A deprecated spelling (renamed keyword, aliased function) that still works."""
