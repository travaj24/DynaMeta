"""Stage 1 (DEVSIM carriers): default layered builder + physics modules.

The default builder (`LayeredDevsimBuilder`) and the DEVSIM physics modules need the
DEVSIM solver. They are imported LAZILY (PEP 562 `__getattr__`) so that importing this
package does NOT pull in `devsim` -- the pure-numpy/scipy submodules (e.g.
`carriers.schrodinger_poisson`, `carriers.sp_carrier`) stay importable, and CI-testable,
without the heavy solver stack installed. `from dynameta.carriers import
LayeredDevsimBuilder` still works (the import happens on first attribute access).
"""

__all__ = ["LayeredDevsimBuilder", "physics_equilibrium"]


def __getattr__(name):
    if name == "LayeredDevsimBuilder":
        from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
        return LayeredDevsimBuilder
    if name == "physics_equilibrium":
        from dynameta.carriers import physics_equilibrium
        return physics_equilibrium
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(__all__)
