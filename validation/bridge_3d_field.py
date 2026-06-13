"""Verify the bridge consumes a NATIVE 3D CarrierField end-to-end: devsim_3d ->
CarrierField(ndim=3) -> assemble_eps (3D branch, IdentityLift) -> EpsField, and the
eps reflects the gate accumulation (Re(eps) lower at the gate-side z, toward ENZ).
This exercises the ndim=3 bridge path (assemble_eps now branches on a 3D grid).
Run: python -m validation.bridge_3d_field
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
from dynameta.sweep import BiasPoint
from dynameta.materials import Material, MaterialRegistry, DrudeOptical, M_E
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.core import NM, MaterialEpsMap, assemble_eps, choose_lift
from dynameta.core.lift import IdentityLift

spec = Stacked3DSpec()            # ITO semi + HfO2 oxide, 12nm lateral
solver = Devsim3DEquilibrium(spec)
cf = solver.solve(BiasPoint({"gate": 1.0, "body": 0.0}, "gate+1V"))

# n->eps map: ITO Drude (matches the validated reference ITO optical model)
reg = MaterialRegistry()
reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14)))
n_to_eps = MaterialEpsMap(reg)

# minimal alignment: the "semi" mesh region <- carrier "semi", z = through-stack
align = GeometryAlignment(
    unit_scale=NM,
    region_alignments=[RegionAlignment("semi", "semi",
                        (0.0, spec.lateral_m, 0.0, spec.lateral_m, 0.0, spec.semi_thk_m),
                        stack_axis="z")],
    fixed_eps_regions={})

eps_by_region = assemble_eps(cf, align, n_to_eps, IdentityLift(), 1300e-9,
                              mesh_regions=["semi"])
ef = eps_by_region["semi"]
v = ef.values_zyx                 # (Nz, Ny, Nx) complex
re_bot = float(np.real(v[0]).mean())     # z=0 body side (n_bg)
re_top = float(np.real(v[-1]).mean())    # gate-side interface (accumulation)
print("[t] EpsField from 3D carrier: values_zyx shape={}".format(v.shape), flush=True)
print("[t] Re(eps) body-side z={:+.3f}  gate-side z={:+.3f}".format(re_bot, re_top), flush=True)
ok = v.ndim == 3 and re_top < re_bot     # accumulation lowers Re(eps) toward ENZ
print("[t] *** bridge ndim=3 path: {} (accumulation lowers Re(eps) at the gate) ***".format(
    "OK" if ok else "FAIL"), flush=True)
solver.teardown()
raise SystemExit(0 if ok else 1)
