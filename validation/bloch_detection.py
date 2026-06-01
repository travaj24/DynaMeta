"""Fast (build-only, no FEM solve) regression for the Floquet-Bloch periodic-phase
machinery the oblique/conical solver rests on (audit conical-F2): the per-identification
direction detection (_detect_bloch_dirs) and the phase-list assignment (_bloch_phase_list).

netgen interleaves the x- and y-face identifications (x,y,x,y,... one pair per z-layer),
so a wrong mapping silently puts phase=1 on the x-faces -> the solver returns the
normal-incidence field at every angle. This checks, WITHOUT a heavy solve, that:
  (1) detection recovers exactly n_px 'x' + n_py 'y' entries (the function also asserts this);
  (2) at ky=0 the y-identifications carry phase 1 and the x ones carry exp(i kx Px);
  (3) at ky!=0 (conical) the y-identifications carry a NON-unit exp(i ky Py).
Run: python -m validation.bloch_detection
"""
import sys, os, cmath
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.materials import Material, MaterialRegistry, ConstantOptical
from dynameta.geometry import UnitCell, Stack, Layer, Design
from dynameta.geometry.specs import Mesh3DSpec
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
from dynameta.optics.solver import _detect_bloch_dirs, _bloch_phase_list


def _geo():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("a", ConstantOptical(4.0 + 0j)))
    reg.add(Material("b", ConstantOptical(6.0 + 0j)))
    cell = UnitCell.square(220e-9)
    # two device layers -> several z-layers -> interleaved x/y idnrs to disentangle
    stack = Stack(layers=[Layer("la", 120e-9, "a"), Layer("lb", 120e-9, "b")],
                   superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=600e-9, substrate_buffer_m=600e-9,
                     maxh_superstrate_m=80e-9, maxh_substrate_m=80e-9, maxh_background_m=60e-9)
    d = Design(name="bloch", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)
    return LayeredOpticalBuilder(d).build()


def main():
    geo = _geo()
    ok = True
    print("[t] built 2-layer cell: n_px={} n_py={}".format(geo.n_px, geo.n_py), flush=True)
    if not (geo.n_px > 0 and geo.n_py > 0):
        print("[t] FAIL: no periodic identifications", flush=True); return False

    dirs = _detect_bloch_dirs(geo)                       # raises if the x/y counts disagree
    cx, cy = dirs.count("x"), dirs.count("y")
    det_ok = (len(dirs) == geo.n_px + geo.n_py and cx == geo.n_px and cy == geo.n_py)
    print("[t] detection: {} entries, x={} (n_px={}), y={} (n_py={}) -> {}".format(
        len(dirs), cx, geo.n_px, cy, geo.n_py, "OK" if det_ok else "MISMATCH"), flush=True)
    ok = ok and det_ok

    kx, ky = 0.01, 0.013                                  # nm^-1
    Px, Py = geo.period_x_nm, geo.period_y_nm
    # ky=0 (x-z-plane oblique): y-faces -> phase 1, x-faces -> exp(i kx Px)
    ph0 = _bloch_phase_list(geo, kx, 0.0)
    y_unit = all(abs(p - 1.0) < 1e-9 for p, d in zip(ph0, dirs) if d == "y")
    x_phase = all(abs(p - cmath.exp(1j * kx * Px)) < 1e-9 for p, d in zip(ph0, dirs) if d == "x")
    print("[t] ky=0: y-idnrs phase==1: {}   x-idnrs phase==exp(i kx Px): {}".format(y_unit, x_phase),
          flush=True)
    ok = ok and y_unit and x_phase
    # ky!=0 (conical): y-faces -> exp(i ky Py) != 1 (the observable transverse phase)
    ph1 = _bloch_phase_list(geo, kx, ky)
    y_conical = all(abs(p - cmath.exp(1j * ky * Py)) < 1e-9 and abs(p - 1.0) > 1e-6
                     for p, d in zip(ph1, dirs) if d == "y")
    print("[t] ky!=0: y-idnrs phase==exp(i ky Py)!=1: {}".format(y_conical), flush=True)
    ok = ok and y_conical

    print("[t] *** BLOCH PHASE DETECTION/ASSIGNMENT: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
