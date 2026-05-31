"""Stage 3 (NGSolve optics): default layered builder, eps assembler, FEM solver."""

from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder, OpticalGeometry
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem

__all__ = ["LayeredOpticalBuilder", "OpticalGeometry", "assemble_eps_cf", "solve_fem"]
