"""Stage 3: 3D NGSolve FEM optical scattering with spatially-varying
bias-dependent eps."""

from dynameta.stage3_optical.ngsolve_build  import build_unit_cell_3d, Cell3DGeometry
from dynameta.stage3_optical.eps_loader     import build_eps_cf_at_bias_lambda
from dynameta.stage3_optical.solver         import solve_fem, FEMResult
from dynameta.stage3_optical.runner         import run_stage3, SweepRow

__all__ = ["build_unit_cell_3d", "Cell3DGeometry",
              "build_eps_cf_at_bias_lambda", "solve_fem", "FEMResult",
              "run_stage3", "SweepRow"]
