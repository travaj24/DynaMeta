"""AUDIT T-9: the declared public API, gated as a surface rather than one name at a time.

171 of 867 `__all__` names (19.7%) had no test caller at all; 125 appeared in no test, validation
*or* example. Writing 171 contract tests is not the fix (and the high-value ones got real gates
-- see tests/test_audit_2026_07_25_infra.py and the T-9 block at the end of
tests/test_lumenairy_bridge.py). What this file buys, cheaply and for EVERY name, is the failure
mode an uncalled entry point actually has: it stops existing. A name deleted from a module but
left in `__all__`, a re-export whose source module was renamed, a class whose default
construction was broken by a field added without a default, a module that no longer imports on a
bare `pip install dynameta` -- each of those ships silently today and is caught here.

Deliberately NOT a coverage substitute: it proves import health and construction, never physics.

Run: python -m pytest tests/test_api_surface.py -q
"""
import importlib
import inspect
import pkgutil
import sys

import pytest

import dynameta

# Third-party packages a module is allowed to need AT IMPORT TIME. Everything here is an optional
# extra (pyproject [project.optional-dependencies]) -- a leg without it may skip that module, but
# a missing package OUTSIDE this list is a broken dependency declaration, not a skip.
_OPTIONAL = {"devsim", "ngsolve", "gmsh", "h5py", "zarr", "matplotlib", "numba", "jax", "jaxlib",
             "cupy", "tmm", "jarvis", "mp_api", "refractiveindex", "netgen", "scipy", "lumenairy"}


def _classify_import_error(name, exc, skipped):
    """A missing OPTIONAL extra is a skip; anything else is a broken package."""
    missing = (getattr(exc, "name", None) or "").split(".")[0]
    if isinstance(exc, ImportError) and missing in _OPTIONAL:
        skipped.append((name, missing))
        return
    raise AssertionError(
        "{} failed to import for a NON-optional reason: {!r}".format(name, exc)) from exc


def _walk_modules():
    """Import every module in the installed dynameta package. Returns (modules, skipped) where
    `skipped` is [(module, missing_optional_package)]. `onerror` is supplied on purpose:
    pkgutil.walk_packages SWALLOWS ImportErrors raised while importing a package to traverse it,
    which would silently drop that package's whole subtree from this gate."""
    mods, skipped = [], []

    def _onerror(name):
        _classify_import_error(name, sys.exc_info()[1], skipped)

    for mi in pkgutil.walk_packages(dynameta.__path__, prefix="dynameta.", onerror=_onerror):
        try:
            mods.append(importlib.import_module(mi.name))
        except ImportError as exc:
            _classify_import_error(mi.name, exc, skipped)
    return mods, skipped


def _public_names(mods):
    for m in mods:
        for name in getattr(m, "__all__", ()) or ():
            yield m, name


@pytest.fixture(scope="module")
def surface():
    mods, skipped = _walk_modules()
    return mods, skipped, list(_public_names(mods))


def test_every_module_imports_and_the_package_is_not_empty(surface):
    mods, skipped, names = surface
    assert len(mods) > 100, "walk_packages found only {} modules".format(len(mods))
    assert len(names) > 700, "only {} public names found".format(len(names))
    # a skip is allowed only for a declared optional extra (the assertion is inside _walk_modules;
    # this pins the reason list so a silently-skipped module cannot hide here)
    assert all(pkg in _OPTIONAL for _mod, pkg in skipped), skipped


def test_every_all_entry_exists_and_is_declared_once(surface):
    """The cheapest real failure of an uncalled entry point: `__all__` names something the module
    no longer defines. `from module import *` raises AttributeError at import time for these, so
    this is an import-health gate for every public name, covered or not."""
    mods, _skipped, _names = surface
    missing, duped = [], []
    for m in mods:
        all_ = list(getattr(m, "__all__", ()) or ())
        for name in all_:
            assert isinstance(name, str), "{}.__all__ contains a non-string".format(m.__name__)
            if not hasattr(m, name):
                missing.append("{}.{}".format(m.__name__, name))
        if len(set(all_)) != len(all_):
            duped.append("{}: {}".format(m.__name__,
                                         sorted({n for n in all_ if all_.count(n) > 1})))
    assert not missing, "__all__ names with no attribute: {}".format(missing)
    assert not duped, "duplicate __all__ entries: {}".format(duped)


def test_every_public_callable_has_an_introspectable_signature(surface):
    """A public function whose signature cannot be read is either a broken decorator/wrapper or a
    re-export that lost its target. Builtins re-exported on purpose (e.g. numba's `prange`, which
    falls back to `range`) are excluded -- only objects DEFINED in dynameta are gated -- as are
    exception types, which inherit BaseException's unreadable C-level signature."""
    _mods, _skipped, names = surface
    bad = []
    for m, name in names:
        obj = getattr(m, name)
        if not callable(obj) or not str(getattr(obj, "__module__", "")).startswith("dynameta"):
            continue
        if inspect.isclass(obj) and issubclass(obj, BaseException):
            continue
        try:
            inspect.signature(obj)
        except (ValueError, TypeError) as exc:
            bad.append("{}.{}: {}".format(m.__name__, name, exc))
    assert not bad, "public callables with no readable signature: {}".format(bad)


def test_zero_argument_constructible_public_classes_construct_and_repr(surface):
    """Every public class of dynameta's own whose constructor needs no arguments is instantiated
    and repr'd. This is what catches a required field added to an all-default dataclass, a
    default that references a deleted symbol, or a `__repr__` that raises -- on the ~50 classes
    nothing else instantiates. typing.Protocols are skipped (they cannot be instantiated by
    design)."""
    _mods, _skipped, names = surface
    made, failures = 0, []
    for m, name in names:
        obj = getattr(m, name)
        if not inspect.isclass(obj) or not str(getattr(obj, "__module__", "")).startswith("dynameta"):
            continue
        if getattr(obj, "_is_protocol", False):
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue
        if not all(p.default is not inspect.Parameter.empty
                   or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                   for p in sig.parameters.values()):
            continue                                        # needs arguments: out of scope here
        try:
            text = repr(obj())
            assert text, "{}.{} repr'd to an empty string".format(m.__name__, name)
            made += 1
        except Exception as exc:                            # noqa: BLE001 -- report, do not mask
            failures.append("{}.{}: {}: {}".format(m.__name__, name, type(exc).__name__, exc))
    assert not failures, "zero-arg public classes that failed to construct/repr: {}".format(failures)
    assert made >= 30, "only {} zero-arg-constructible public classes exercised".format(made)
