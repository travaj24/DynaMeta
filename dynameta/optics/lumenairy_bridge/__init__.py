"""dynameta.optics.lumenairy_bridge: Lumenairy RCWA/PMM as DynaMeta optical backends.

Lumenairy is a REQUIRED dependency of dynameta (core since v0.5) but is imported lazily:
this package imports without touching it (keeping base `import dynameta` fast and
matplotlib-free); the backends raise with an install hint if the environment lacks it.
Conventions are
identical on both sides (exp(-i omega t), Im(eps) > 0, metres, radians), so the bridge is a
geometry/result adapter, not a translation layer. See docs/roadmap_v0.5_integration_photonics.md.
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
