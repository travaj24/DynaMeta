"""LazyModule: a module-level stand-in for a heavy optional import, loaded at FIRST ATTRIBUTE ACCESS.

Why (user decision 2026-11-04, v0.14 task 8): the FEM carrier modules (carriers.electrostatics_fem,
electrothermal, fem_mesh, thermal_fem.*) did `import ngsolve as ng` at module scope, and ngsolve loads
its own OpenBLAS (ngsolve_openblas, MAX_THREADS=128), which COMMITS ~4 GB of private memory per process
at load (docs/audit/2026-11-03-devsim-2.11-core.md, open question 4). `import dynameta.carriers` never
imported them (lazy package __getattr__, gated by tests/test_audit_2026_07_17_infra.py), but anything
that resolved one of those submodules -- tests/test_audit_2026_07_25_carriers_hygiene.py's child resolves
every advertised name of the package -- paid the 4 GB before a single FEM call. MEASURED (box A, this
date): a process that resolves every dynameta.carriers submodule held 6163 MB private commit with the
eager imports. With `ng = LazyModule("ngsolve")` the module imports without ngsolve and the first
`ng.<attr>` performs the real import.

Contract:
  * `ng.Mesh`, `ng.GridFunction(...)`, `with ng.TaskManager():` ... behave exactly as with the real
    module (attribute access is forwarded; the real module object is cached after the first load).
  * A missing package raises the SAME ModuleNotFoundError ("No module named 'ngsolve'") as the eager
    import did -- at first use instead of at import of the dynameta module.
  * Annotations: every adopting module has `from __future__ import annotations`, so `ng.Mesh` in a
    signature is a string and costs nothing until typing.get_type_hints evaluates it (which imports).
Stdlib-only (importable from every tier, like dynameta.warning_types).
"""

from __future__ import annotations

import importlib

__all__ = ["LazyModule"]


class LazyModule:
    """Proxy for `import <name>`; the import happens on the first attribute access."""

    __slots__ = ("_lazy_name", "_lazy_mod")

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_mod", None)

    def _lazy_load(self):
        mod = object.__getattribute__(self, "_lazy_mod")
        if mod is None:
            mod = importlib.import_module(object.__getattribute__(self, "_lazy_name"))
            object.__setattr__(self, "_lazy_mod", mod)
        return mod

    @property
    def is_loaded(self) -> bool:
        """True once the real module has been imported through this proxy."""
        return object.__getattribute__(self, "_lazy_mod") is not None

    def __getattr__(self, attr):                 # only called when normal lookup fails: forward it
        return getattr(self._lazy_load(), attr)

    def __setattr__(self, attr, value):          # e.g. a test monkeypatching ng.<attr>
        setattr(self._lazy_load(), attr, value)

    def __delattr__(self, attr):
        delattr(self._lazy_load(), attr)

    def __dir__(self):
        return dir(self._lazy_load())

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_lazy_name")
        return "<LazyModule {!r} ({})>".format(name, "loaded" if self.is_loaded else "not loaded")
