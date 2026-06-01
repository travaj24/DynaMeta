"""Verify carriers/devsim_3d.py: the 3D equilibrium solve emits a bridge-consumable
CarrierField(ndim=3) with the correct gate accumulation. Run: python -m validation.carriers_3d_field
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
from dynameta.sweep import BiasPoint
from dynameta.core.carrier_field import ELECTRON_DENSITY

N_BG = 4e26
solver = Devsim3DEquilibrium(Stacked3DSpec(n_bg_m3=N_BG))
cf = solver.solve(BiasPoint({"gate": 1.0, "body": 0.0}, "gate+1V"))
reg = cf.regions["semi"]
n = reg.grid_fields[ELECTRON_DENSITY]
print("[t] CarrierField ndim={}  grid_axes={}  n grid shape={}".format(
    cf.ndim, sorted(reg.grid_axes_m), n.shape), flush=True)
print("[t] gate+1V: n_top/n_bg={:.3f} (accum)  n_bot/n_bg={:.3f}".format(
    n[:, :, -1].mean() / N_BG, n[:, :, 0].mean() / N_BG), flush=True)
ok = cf.ndim == 3 and n.ndim == 3 and n[:, :, -1].mean() > n[:, :, 0].mean()
print("[t] *** devsim_3d -> CarrierField(ndim=3): {} ***".format("OK" if ok else "FAIL"), flush=True)
solver.teardown()
