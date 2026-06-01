"""Regression guard for audit GEO-1 (HIGH): a CLOCKWISE-wound freeform Polygon inclusion
must build the SAME geometry as the counter-clockwise winding. Before the fix, a CW ring
extruded to a negative-volume OCC face whose cell-intersection captured the COMPLEMENT of the
footprint, silently SWAPPING the inclusion and background materials (and sometimes leaving no
background region at all). _polygon_prism now normalizes the winding to CCW via the signed
shoelace area.

Builds a centered square Polygon inclusion both windings (build only, no solve -> fast),
integrates the per-region volumes, and checks: the inclusion volume equals the analytic
square prism (a^2 * t) for BOTH windings, the background region is non-empty for both, and the
two windings agree. Run: python -m validation.inclusion_winding
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ngsolve as ng
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design
from dynameta.geometry.cross_section import Polygon
from dynameta.geometry.specs import Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder

P, THK, A = 400.0, 200.0, 160.0                 # cell / thickness / square side (nm)
LO, HI = (P - A) / 2 * 1e-9, (P + A) / 2 * 1e-9
CCW = [(LO, LO), (HI, LO), (HI, HI), (LO, HI)]  # counter-clockwise (positive shoelace)
V_INCL_NM3 = A * A * THK                        # analytic inclusion prism volume (nm^3)
TOL = 0.03                                      # 3% (meshed-volume integration)


def _region_volume(geo, substr):
    names = [m for m in geo.mesh.GetMaterials() if substr(m)]
    if not names:
        return 0.0
    reg = geo.mesh.Materials("|".join(re.escape(m) for m in names))
    return float(ng.Integrate(ng.CoefficientFunction(1.0), geo.mesh, definedon=reg))


def _build_volumes(points, label):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(complex(2.45 ** 2, 0.0))))
    stack = Stack(layers=[Layer("L", THK * 1e-9, "air",
                                inclusions=[Inclusion(Polygon(points), "hi")])],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=700e-9, superstrate_buffer_m=1200e-9, substrate_buffer_m=1200e-9,
                    maxh_superstrate_m=60e-9, maxh_substrate_m=60e-9,
                    maxh_background_m=30e-9, maxh_inclusion_m=30e-9)
    d = Design(name=label, unit_cell=UnitCell.square(P * 1e-9), stack=stack,
               electrodes=[], materials=reg, mesh_3d=m3)
    geo = LayeredOpticalBuilder(d).build()
    v_incl = _region_volume(geo, lambda m: "__incl" in m)
    v_bg = _region_volume(geo, lambda m: m == "L" or "__bg" in m)
    print("[t] {:4s}: incl_vol={:.3e} nm^3 (analytic {:.3e})  bg_vol={:.3e} nm^3".format(
        label, v_incl, V_INCL_NM3, v_bg), flush=True)
    return v_incl, v_bg


def main():
    print("[t] POLYGON WINDING (GEO-1): centered {:.0f}nm square in {:.0f}nm cell, t={:.0f}nm".format(
        A, P, THK), flush=True)
    vi_ccw, vb_ccw = _build_volumes(CCW, "ccw")
    vi_cw, vb_cw = _build_volumes(CCW[::-1], "cw")        # clockwise (reversed)
    incl_ok = (abs(vi_ccw - V_INCL_NM3) / V_INCL_NM3 < TOL
               and abs(vi_cw - V_INCL_NM3) / V_INCL_NM3 < TOL)
    bg_ok = vb_ccw > 0.0 and vb_cw > 0.0
    agree = abs(vi_ccw - vi_cw) / V_INCL_NM3 < TOL and abs(vb_ccw - vb_cw) / max(vb_ccw, 1.0) < TOL
    ok = incl_ok and bg_ok and agree
    print("[t]   inclusion-vol==analytic (both)={}  background non-empty (both)={}  "
          "CW==CCW={}".format(incl_ok, bg_ok, agree), flush=True)
    print("[t] *** POLYGON WINDING NORMALIZATION (GEO-1): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
