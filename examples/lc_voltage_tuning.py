"""END-TO-END: drive voltage -> LC director BVP -> effective index -> spectrum, THROUGH run_pipeline.

The chain this example wires (each piece is oracle-validated; the pipeline integration is new):

    V --drivers.lc_extra_fields / two-constant director BVP--> director_angle_rad profile
        (the field-axis -> plate-plane convention flip happens INSIDE the glue, exactly once)
        --custom scalar EffectModel (e-wave index of the tilted axis, OPL-averaged)--> eps(V)
        --run_pipeline + make_layered_tmm_solver--> R(V) of an LC-filled etalon

The custom EffectModel is the documented BYO pattern: normal-incidence x-pol sees the
extraordinary index n(theta') with theta' the angle between k (the z/field axis) and the optic
axis -- the library's n_local_from_theta(theta_field, model='extra_k_axis').

GATE A: below the Freedericksz threshold the director stays planar (plate-plane angle ~ 0)
        and n_eff ~ n_e; far above threshold n_eff falls toward n_o, monotonically in V.
GATE B (chain equality): at EVERY bias the pipeline R equals a hand-rolled
        BVP -> n_local -> TMM solve (the independent path) to 1e-12.
GATE C: the V-swing moves the etalon reflectance by more than 5 percentage points.

numpy/scipy + tmm only (no DEVSIM/NGSolve). Run: python -m examples.lc_voltage_tuning
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.lc_director import (director_profile_bvp, freedericksz_threshold_V,
                                           n_local_from_theta)
from dynameta.core import NM, EffectEpsMap
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.core.carrier_field import CarrierField, CarrierRegion, ELECTRON_DENSITY
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.drivers import lc_extra_fields
from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.optics.tmm_reference import TmmLayeredSolver, make_layered_tmm_solver
from dynameta.pipeline import run_pipeline
from dynameta.sweep import BiasPoint, Sweep

PERIOD = 400e-9
D_LC = 2.0e-6
LAM_NM = 1550.0
N_O, N_E = 1.52, 1.74                            # 5CB-like
LC_KW = dict(K11=6.2e-12, K33=8.3e-12, eps_para=19.0, eps_perp=5.2, d_planar=D_LC, nz=201)


@dataclass
class LCEffectiveIndexModel:
    """BYO EffectModel: plate-plane director profile -> OPL-averaged e-wave index -> scalar eps.
    Reads fields['director_angle_rad'] (a (nz,) profile from lc_extra_fields(reduce='profile')
    on a UNIFORM z grid, so the OPL average is the plain mean of the local index)."""
    n_o: float
    n_e: float

    def eps(self, fields, lambda_m):
        theta_plate = np.asarray(fields.get("director_angle_rad", 0.0), dtype=np.float64)
        theta_field = np.pi / 2.0 - theta_plate              # back to the field-axis convention
        n_loc = n_local_from_theta(theta_field, self.n_o, self.n_e, model="extra_k_axis")
        return complex(float(np.mean(n_loc)) ** 2)


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.45 ** 2))))
    reg.add(Material("lc", ConstantOptical(complex(N_E ** 2))))
    return Design(name="lc_tune", unit_cell=UnitCell.square(PERIOD),
                  stack=Stack(layers=[Layer("lc", D_LC, "lc")],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0,
                                      lift="identity"))


class _StubCarrier:
    def solve(self, bp):
        ax = np.linspace(0.0, PERIOD, 3)
        z = np.linspace(0.0, D_LC, 3)
        reg = CarrierRegion(name="lc", role="dielectric", material="lc",
                            nodes_m=np.zeros((1, 3)), node_fields={},
                            grid_axes_m={"x": ax, "y": ax, "z": z},
                            grid_fields={ELECTRON_DENSITY: np.zeros((3, 3, 3))})
        return CarrierField(bias_label=bp.label, voltages=dict(bp.voltages), ndim=3,
                            temperature_K=300.0, regions={"lc": reg},
                            n_bg_by_region={"lc": 0.0}, unit_cell_m=(PERIOD, PERIOD))


class _StubGeo:
    class mesh:
        ne = 0
        nv = 0


class _StubBuilder:
    def build(self):
        return _StubGeo()

    def alignment(self):
        return GeometryAlignment(unit_scale=NM, region_alignments=[
            RegionAlignment("lc", "lc", (0.0, PERIOD, 0.0, PERIOD, 0.0, D_LC),
                            stack_axis="z")], fixed_eps_regions={})

    def mesh_regions(self):
        return ["lc"]


def main():
    print("[lct] === V -> director BVP -> n_eff -> TMM R, via run_pipeline ===", flush=True)
    ok = True
    design = _design()
    model = LCEffectiveIndexModel(N_O, N_E)
    fields_of_bias = lc_extra_fields(lambda bp: float(bp.voltages["lc"]),
                                     reduce="profile", **LC_KW)
    V_th = freedericksz_threshold_V(LC_KW["K11"], LC_KW["eps_para"] - LC_KW["eps_perp"])
    volts = [0.0, 0.5 * V_th, 2.0, 4.0, 8.0]
    sweep = Sweep(bias_points=[BiasPoint(label="V{:g}".format(v), voltages={"lc": v})
                               for v in volts], wavelengths_nm=[LAM_NM])
    rows = run_pipeline(design, sweep, verbose=False, carrier_solver=_StubCarrier(),
                        optical_builder=_StubBuilder(),
                        optical_solver=make_layered_tmm_solver(),
                        n_to_eps=EffectEpsMap(design.materials, effects={"lc": model}),
                        extra_fields=fields_of_bias)

    lam = LAM_NM * 1e-9
    n_eff, R = [], [row.result.R for row in rows]
    for bp in sweep.bias_points:
        n_eff.append(float(np.sqrt(model.eps(fields_of_bias(bp), lam).real)))
    for v, n, r in zip(volts, n_eff, R):
        print("[lct]   V = {:6.3f} V -> n_eff = {:.5f}, R = {:.4f}".format(v, n, r), flush=True)
    print("[lct]   (Freedericksz V_th = {:.3f} V)".format(V_th), flush=True)

    g_a = bool(abs(n_eff[0] - N_E) < 5e-3 and abs(n_eff[1] - N_E) < 5e-3
               and all(n_eff[i + 1] < n_eff[i] + 1e-12 for i in range(1, len(n_eff) - 1))
               and N_O - 1e-3 < n_eff[-1] < n_eff[1]
               and (n_eff[1] - n_eff[-1]) > 0.5 * (N_E - N_O))
    ok = ok and g_a
    print("[lct] GATE A: planar (n_e) below V_th, monotone fall toward n_o above -> {}".format(
        "PASS" if g_a else "FAIL"), flush=True)

    # GATE B: hand-rolled independent path at every bias
    worst = 0.0
    for v, r_pipe in zip(volts, R):
        res = director_profile_bvp(V_app=v, **LC_KW)
        n_loc = n_local_from_theta(res.theta_field_rad, N_O, N_E, model="extra_k_axis")
        stk = LayeredStack(1.0 + 0j, 1.45 + 0j,
                           [LayeredSlab(D_LC, eps=complex(float(np.mean(n_loc)) ** 2))],
                           period_x_m=PERIOD, period_y_m=PERIOD)
        r_direct = TmmLayeredSolver().solve(stk, lam, design.optical).R
        worst = max(worst, abs(r_pipe - r_direct))
    g_b = bool(worst < 1e-12)
    ok = ok and g_b
    print("[lct] GATE B: pipeline R == direct BVP->n_local->TMM chain at every bias "
          "(worst |dR| = {:.2e}) -> {}".format(worst, "PASS" if g_b else "FAIL"), flush=True)

    g_c = bool(max(R) - min(R) > 0.05)
    ok = ok and g_c
    print("[lct] GATE C: etalon reflectance swing dR = {:.3f} -> {}".format(
        max(R) - min(R), "PASS" if g_c else "FAIL"), flush=True)

    print("[lct] *** LC VOLTAGE-TUNING WORKFLOW: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
