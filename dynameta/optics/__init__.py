"""Stage 3 (NGSolve optics): default layered builder, eps assembler, FEM solver.

The heavy ngsolve/netgen-dependent names are re-exported LAZILY (PEP 562) so that
`import dynameta.optics` -- and the pure `dynameta.optics.tmm_reference` submodule --
work WITHOUT ngsolve installed (e.g. the numpy/scipy-only CI test environment). Each
name imports its backing module only when first accessed.
"""

import importlib

_LAZY = {
    "LayeredOpticalBuilder": "dynameta.optics.ngsolve_layered",
    "OpticalGeometry":       "dynameta.optics.ngsolve_layered",
    "assemble_eps_cf":       "dynameta.optics.eps_assembler",
    "solve_fem":             "dynameta.optics.solver",
}
__all__ = list(_LAZY)


def __getattr__(name):
    # Only the re-exported names are handled here; submodules (tmm_reference, solver, ...)
    # fall through to normal submodule import, so this never recurses.
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
