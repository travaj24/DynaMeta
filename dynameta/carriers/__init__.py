"""Stage 1 (DEVSIM carriers): default layered builder + physics modules.

The default builder (`LayeredDevsimBuilder`) and the DEVSIM physics modules need the
DEVSIM solver. They are imported LAZILY (PEP 562 `__getattr__`) so that importing this
package does NOT pull in `devsim` -- the pure-numpy/scipy submodules (e.g.
`carriers.schrodinger_poisson`, `carriers.sp_carrier`) stay importable, and CI-testable,
without the heavy solver stack installed. `from dynameta.carriers import
LayeredDevsimBuilder` still works (the import happens on first attribute access).
"""

__all__ = ["LayeredDevsimBuilder", "physics_equilibrium", "ac_analysis"]


def __getattr__(name):
    # Only the re-exported CLASS needs lazy handling. SUBMODULES (e.g.
    # physics_equilibrium) must NOT be handled here -- `from dynameta.carriers import
    # physics_equilibrium` is resolved by the normal submodule-import machinery once
    # this raises AttributeError; intercepting it would recurse into __getattr__
    # (the bug an end-to-end run_pipeline test caught). Use importlib so the class
    # lookup never re-enters this function either.
    if name == "LayeredDevsimBuilder":
        import importlib
        return importlib.import_module("dynameta.carriers.devsim_layered").LayeredDevsimBuilder
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(__all__)
