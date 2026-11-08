"""Stage 1 (DEVSIM carriers): default layered builder + physics modules.

The default builder (`LayeredDevsimBuilder`) and the DEVSIM physics modules need the
DEVSIM solver. They are imported LAZILY (PEP 562 `__getattr__`) so that importing this
package does NOT pull in `devsim` -- the pure-numpy/scipy submodules (e.g.
`carriers.schrodinger_poisson`, `carriers.sp_carrier`) stay importable, and CI-testable,
without the heavy solver stack installed. `from dynameta.carriers import
LayeredDevsimBuilder` still works (the import happens on first attribute access).

STAR IMPORT IS EAGER BY DEFINITION (audit V-13, 2026-11-02). `from dynameta.carriers import *`
resolves EVERY name in `__all__`, and `__all__` names the DEVSIM builder and the DEVSIM physics
submodules -- so the star form imports devsim (measured: devsim + scipy loaded, against scipy
only for `import dynameta.carriers`), and on an install without devsim it raises that
ImportError. The laziness above is a property of `import dynameta.carriers` and of NAMED imports
(`from dynameta.carriers import schrodinger_poisson`), not of `*`. That is deliberate rather than
fixed: `__all__` is the declared public surface (tests/test_api_surface.py walks it), and
trimming the solver names out of it to make `*` cheap would un-declare the package's headline
builder. Use named imports on a devsim-free install.
"""

__all__ = ["LayeredDevsimBuilder", "physics_equilibrium", "ac_analysis", "noise_analysis", "transient",
           "lc_director", "lc_director_2d", "lc_dynamics", "carrier_heating", "llg"]


_SUBMODULES = None                       # names of this package's submodules (filled on first use)


def _submodules():
    global _SUBMODULES
    if _SUBMODULES is None:
        import pkgutil
        _SUBMODULES = frozenset(mi.name for mi in pkgutil.iter_modules(__path__))
    return _SUBMODULES


def __getattr__(name):
    # The re-exported CLASS and (audit V-20, 2026-11-02) every SUBMODULE resolve lazily, through
    # importlib.import_module ONLY -- it binds the submodule onto the package and returns it without
    # re-entering this function (the recursion an end-to-end run_pipeline test once caught came from
    # a `from dynameta.carriers import ...` here). V-20: `__all__` named 8 submodules that were not
    # attributes until something imported them, so `dynameta.carriers.llg` raised AttributeError
    # although __all__ and dir() advertised it. A submodule whose optional dependency (devsim) is
    # missing raises that ImportError here -- the honest failure, exactly what importing it does.
    import importlib
    if name == "LayeredDevsimBuilder":
        return importlib.import_module("dynameta.carriers.devsim_layered").LayeredDevsimBuilder
    if not name.startswith("__") and name in _submodules():
        return importlib.import_module("{}.{}".format(__name__, name))
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    # audit X-21: this returned sorted(__all__), which HID exactly the pure-numpy submodules the
    # docstring advertises as the CI-testable path (schrodinger_poisson, sp_carrier, einstein, ...).
    # Every name listed here resolves through __getattr__ (V-20).
    return sorted(set(__all__) | set(globals()) | _submodules())
