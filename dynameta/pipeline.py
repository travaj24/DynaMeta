"""
Top-level orchestration on the new bridge architecture:

  carrier solver (DEVSIM) --solve each bias--> CarrierField
        |                                           |
        |                                  core.bridge.assemble_eps
        v                                           v
  optical geometry builder (NGSolve) <--GeometryAlignment--  EpsField per region
        |                                           |
        +-------------- optical solver -------------+ --> OpticalResult per (bias, lambda)

The pipeline is written against the core Protocols, so a caller can swap in a
bring-your-own CarrierSolver / OpticalGeometryBuilder / OpticalSolver. The
defaults are the layered DEVSIM/NGSolve builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from dynameta.core.bridge import assemble_eps
from dynameta.core.lift import choose_lift
from dynameta.core.n_to_eps import MaterialEpsMap
from dynameta.core.interfaces import OpticalResult
from dynameta.geometry.design import Design
from dynameta.sweep import Sweep
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.eps_assembler import assemble_eps_cf
from dynameta.optics.solver import solve_fem


@dataclass
class SweepRow:
    bias_label:   str
    lambda_nm:    float
    result:       OpticalResult


def _fem_optical_solver(design, geo, eps_by_region, lam_m, n_super, n_sub) -> OpticalResult:
    """Default optical solve: assemble the NGSolve VoxelCoefficient + run the FEM."""
    eps_cf = assemble_eps_cf(geo, eps_by_region)
    return solve_fem(geo, lam_m, eps_cf, design.optical,
                      order=design.mesh_3d.fem_order, n_super=n_super, n_sub=n_sub)


def run_pipeline(design: Design, sweep: Sweep, *,
                   verbose: bool = True,
                   carrier_solver=None,
                   optical_builder=None,
                   optical_solver=None) -> List[SweepRow]:
    """Run the full Design + Sweep through carriers -> bridge -> optics.

    carrier_solver / optical_builder may be supplied to override the defaults
    (bring-your-own); each must satisfy the corresponding core Protocol.

    optical_solver: an optional callable
    ``fn(design, geo, eps_by_region, lam_m, n_super, n_sub) -> OpticalResult`` that replaces
    the default per-(bias, wavelength) FEM solve. This is the seam for an alternative optical
    backend (e.g. a layered TMM/RCWA solver via
    ``dynameta.optics.tmm_reference.layered_stack_from_design`` + a LayeredStackSolver); the
    default is the NGSolve FEM (`_fem_optical_solver`).
    """
    carrier = carrier_solver or LayeredDevsimBuilder(design)
    # 1) carriers: solve every bias, collect CarrierFields, then free DEVSIM
    fields = {}
    for bp in sweep.bias_points:
        if verbose:
            print("[carriers] bias '{}' ...".format(bp.label), flush=True)
        fields[bp.label] = carrier.solve(bp)
    if hasattr(carrier, "teardown"):
        carrier.teardown()

    # 2) optical geometry: build mesh once + the alignment contract
    optical = optical_builder or LayeredOpticalBuilder(design)
    geo = optical.build()
    align = optical.alignment()
    mesh_regions = optical.mesh_regions()
    align.validate_coverage(mesh_regions)
    if verbose:
        print("[optics] mesh ne={} nv={}; {} spatial + {} fixed regions".format(
            geo.mesh.ne, geo.mesh.nv, len(align.region_alignments),
            len(align.fixed_eps_regions)), flush=True)

    solve_optics = optical_solver or _fem_optical_solver
    n_to_eps = MaterialEpsMap(design.materials)
    lift = choose_lift(design.device_symmetry(), design.optical.lift,
                        period_y_m=design.unit_cell.period_y_m, ny=design.optical.ny_sym)
    import cmath
    sup_mat = design.stack.superstrate_material
    sub_mat = design.stack.substrate_material

    # 3) bridge + solve per (bias, wavelength)
    rows: List[SweepRow] = []
    for bp in sweep.bias_points:
        cf = fields[bp.label]
        for lam_nm in sweep.wavelengths_nm:
            lam_m = float(lam_nm) * 1e-9
            n_super = cmath.sqrt(complex(design.materials.get(sup_mat).eps(lam_m)))
            n_sub = cmath.sqrt(complex(design.materials.get(sub_mat).eps(lam_m)))
            eps_by_region = assemble_eps(cf, align, n_to_eps, lift, lam_m,
                                          mesh_regions=mesh_regions)
            res = solve_optics(design, geo, eps_by_region, lam_m, n_super, n_sub)
            rows.append(SweepRow(bp.label, float(lam_nm), res))
            if verbose:
                tstr = ("T={:.4f} A={:+.4f}".format(res.T, res.A)
                         if res.T is not None else "T=n/a")
                print("[optics]   {} lam={:.0f}nm  R={:.4f}  {}  phase={:+.1f}  ({:.1f}s)".format(
                    bp.label, lam_nm, res.R, tstr, res.phase_deg, res.solve_time_s), flush=True)
    return rows
