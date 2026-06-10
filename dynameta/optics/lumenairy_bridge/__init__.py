"""dynameta.optics.lumenairy_bridge: Lumenairy RCWA/PMM as DynaMeta optical backends.

Lumenairy is an OPTIONAL dependency (`pip install dynameta[lumenairy]`); this package
imports without it -- the backends raise with an install hint when called. Conventions are
identical on both sides (exp(-i omega t), Im(eps) > 0, metres, radians), so the bridge is a
geometry/result adapter, not a translation layer. See docs/roadmap_v0.5_integration_photonics.md.
"""

from dynameta.optics.lumenairy_bridge.rcwa_backend import (LumenairyStackSolver,
                                                           design_to_rcwa_stack,
                                                           make_lumenairy_rcwa_solver,
                                                           rcwa_result_to_optical_result)

__all__ = ["LumenairyStackSolver", "design_to_rcwa_stack", "make_lumenairy_rcwa_solver",
           "rcwa_result_to_optical_result"]
