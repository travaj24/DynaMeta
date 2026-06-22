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
           "make_lumenairy_pmm_solver"]
