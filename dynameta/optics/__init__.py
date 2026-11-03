"""Stage 3 (NGSolve optics): default layered builder, eps assembler, FEM solver.

The heavy ngsolve/netgen-dependent names are re-exported LAZILY (PEP 562) so that
`import dynameta.optics` -- and the pure `dynameta.optics.tmm_reference` submodule --
work WITHOUT ngsolve installed (e.g. the numpy/scipy-only CI test environment). Each
name imports its backing module only when first accessed.

STAR IMPORT IS EAGER BY DEFINITION (audit V-13, 2026-11-02): every name in `__all__` below is an
NGSolve name, so `from dynameta.optics import *` resolves all of them and imports netgen/ngsolve
(measured: ngsolve + netgen loaded, against scipy only for `import dynameta.optics`); without the
[solvers] extra it raises that ImportError. The lazy guarantee covers `import dynameta.optics`
and NAMED submodule imports (`from dynameta.optics import tmm_reference`), not `*`.
"""

import importlib

_LAZY = {
    "LayeredOpticalBuilder": "dynameta.optics.ngsolve_layered",
    "OpticalGeometry":       "dynameta.optics.ngsolve_layered",
    "assemble_eps_cf":       "dynameta.optics.eps_assembler",
    "solve_fem":             "dynameta.optics.solver",
    "solve_fem_pols":        "dynameta.optics.solver",
    "FEMFactorCache":        "dynameta.optics.solver",
    "pardiso_available":     "dynameta.optics.solver",
}
__all__ = list(_LAZY)


def __getattr__(name):
    # Only the re-exported names are handled here; submodules (tmm_reference, solver, ...)
    # fall through to normal submodule import, so this never recurses.
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
