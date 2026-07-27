"""dynameta.optics.lumenairy_bridge: Lumenairy RCWA/PMM as DynaMeta optical backends.

Lumenairy is a REQUIRED dependency of dynameta (core since v0.5) but is imported lazily:
this package imports without touching it (keeping base `import dynameta` fast and
matplotlib-free); the backends raise with an install hint if the environment lacks it.
The SIGN conventions are identical on both sides (exp(-i omega t), Im(eps) > 0, metres,
radians), so no permittivity is ever conjugated across the seam.

IT IS STILL A TRANSLATION LAYER (audit V-7 -- this header used to claim otherwise while the
code translated at least nine quantities). What the bridge converts, and where:

  * layer ORDER (DynaMeta Stack is bottom-first; lumenairy stacks are superstrate-first)
  * half-spaces as INDICES vs layer specs as PERMITTIVITIES (translate.py; the *_design API)
  * incidence ANGLES degrees -> radians, and azimuth (_common.angles_rad)
  * incidence SIDE (guard_incidence_side: top only -- bottom incidence is refused, not mapped)
  * polarization LABEL -> lumenairy lab row index (_common.pol_row: x/y/p -> 0/1/0)
  * p-pol AMPLITUDE sign and the cos_i/cos_t scale, mapping lumenairy's lab-basis Jones onto
    DynaMeta's incumbent Byrnes-tmm p-hat basis (_common.p_basis_conversion)
  * CONICAL incidence (azimuth != 0), which needs per-order Jones synthesis rather than a row
    pick (_common.conical_synthesis; refused for the row-pick backends by guard_conical_ppol)
  * per-layer RECORDS on the reverse path (translate.py)
  * eps CELL rasterization from Design inclusions (optics.rasterize)

A convention bug in any of those is a wrong number with a plausible R + T, which is why each
has its own gate under validation/lumenairy_*.py.
See docs/roadmap_v0.5_integration_photonics.md.
"""

from dynameta.optics.lumenairy_bridge.berreman_backend import (BerremanLayeredSolver,
                                                              berreman_result_to_optical_result,
                                                              design_to_berreman_layers,
                                                              make_lumenairy_berreman_solver)
from dynameta.optics.lumenairy_bridge.berreman_design import berreman_jones, berreman_RT
from dynameta.optics.lumenairy_bridge.rcwa_design import (drude_eps_jax, pmm_stack_jones,
                                                          pmm_stack_RT, rcwa_grating_RT,
                                                          rcwa_stack_jones, rcwa_stack_RT)
from dynameta.optics.lumenairy_bridge.bor_backend import (BorLayer, BorResult, BorStackSpec,
                                                          bor_result_to_optical_result,
                                                          make_lumenairy_bor_solver, solve_bor)
from dynameta.optics.lumenairy_bridge.emt_screen import (bruggeman_eps,
                                                        homogenize_lamellar_layers,
                                                        make_lumenairy_emt_screen_solver,
                                                        maxwell_garnett_eps,
                                                        rytov_tensor_for_layer)
from dynameta.optics.lumenairy_bridge.pmm2d_backend import (design_to_pmm2d_stack,
                                                            layer_to_pure_cell,
                                                            make_lumenairy_pmm2d_solver,
                                                            pure_union_grid_n)
from dynameta.optics.lumenairy_bridge.pmm_backend import (design_to_pmm_stack,
                                                          layer_to_pmm_segments,
                                                          make_lumenairy_pmm_solver)
from dynameta.optics.lumenairy_bridge.rcwa_backend import (LumenairyStackSolver,
                                                           design_to_rcwa_stack,
                                                           make_lumenairy_rcwa_solver,
                                                           rcwa_result_to_optical_result)
from dynameta.optics.lumenairy_bridge.translate import (CallableOptical,
                                                        lumenairy_eps_to_optical_model,
                                                        optical_model_to_lumenairy_eps,
                                                        rcwa_stack_to_design)

__all__ = ["LumenairyStackSolver", "design_to_rcwa_stack", "make_lumenairy_rcwa_solver",
           "rcwa_result_to_optical_result", "CallableOptical",
           "lumenairy_eps_to_optical_model", "optical_model_to_lumenairy_eps",
           "rcwa_stack_to_design", "design_to_pmm_stack", "layer_to_pmm_segments",
           "make_lumenairy_pmm_solver",
           "design_to_pmm2d_stack", "layer_to_pure_cell", "make_lumenairy_pmm2d_solver",
           "pure_union_grid_n",
           "BerremanLayeredSolver", "berreman_result_to_optical_result",
           "design_to_berreman_layers", "make_lumenairy_berreman_solver",
           "rytov_tensor_for_layer", "homogenize_lamellar_layers",
           "make_lumenairy_emt_screen_solver", "maxwell_garnett_eps", "bruggeman_eps",
           "berreman_RT", "berreman_jones",
           "rcwa_grating_RT", "rcwa_stack_RT", "rcwa_stack_jones",
           "pmm_stack_RT", "pmm_stack_jones", "drude_eps_jax",
           "BorLayer", "BorResult", "BorStackSpec", "solve_bor", "bor_result_to_optical_result",
           "make_lumenairy_bor_solver"]
