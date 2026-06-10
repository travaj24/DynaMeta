"""END-TO-END: thermal anneal pulse -> PCM crystalline fraction -> reflectance, THROUGH run_pipeline.

The chain this example wires (each piece is oracle-validated; the pipeline integration is new):

    anneal pulse (t, T) --drivers.pcm_extra_fields / PCMSwitching JMAK--> crystalline_fraction
        --EffectEpsMap + PCMModel (Bruggeman blend)--> eps(f)
        --run_pipeline + make_layered_tmm_solver--> R(anneal time)

Bias points parameterize the anneal duration; the stub carrier solver stands in for DEVSIM
(the PCM film carries no free-carrier physics), exactly the documented pluggable-seam pattern.

GATE A: fraction endpoints -- no anneal -> x = 0; long anneal -> x > 0.999.
GATE B (chain equality): at EVERY bias the pipeline R equals a hand-rolled
        PCMModel.eps(x) -> TMM stack solve (the independent path) to 1e-12.
GATE C: amorphous <-> crystalline reflectance contrast exceeds 10 percentage points.

numpy + tmm only (no DEVSIM/NGSolve). Run: python -m examples.pcm_reflectance_switching
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.switching import PCMSwitching
from dynameta.core import NM, EffectEpsMap
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.core.carrier_field import CarrierField, CarrierRegion, ELECTRON_DENSITY
from dynameta.core.effects import PCMModel
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.drivers import pcm_extra_fields
from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.optics.tmm_reference import TmmLayeredSolver, make_layered_tmm_solver
from dynameta.pipeline import run_pipeline
from dynameta.sweep import BiasPoint, Sweep

PERIOD = 400e-9
T_PCM = 50e-9
LAM_NM = 1550.0
EPS_A = complex(16.0, 1.0)            # amorphous GST-like
EPS_C = complex(36.0, 12.0)           # crystalline GST-like
EPS_SUB = complex(-100.0, 10.0)       # metal back-reflector substrate
# K(650 K) = K0 exp(-E_a/kT) ~ 5e7 1/s -> JMAK x: ~0.02 at 5 ns, ~0.6 at 20 ns, ~1 at 2 us
SW = PCMSwitching(K0_per_s=1.0e13, E_a_J=1.1e-19, T_glass_K=450.0, T_melt_K=900.0)
T_ANNEAL_K = 650.0


def _design():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("gst", ConstantOptical(EPS_A)))
    reg.add(Material("mirror", ConstantOptical(EPS_SUB), is_metal=True))
    return Design(name="pcm_switch", unit_cell=UnitCell.square(PERIOD),
                  stack=Stack(layers=[Layer("pcm", T_PCM, "gst")],
                              superstrate_material="air", substrate_material="mirror"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=0.0,
                                      lift="identity"))


class _StubCarrier:
    """The PCM film has no free-carrier physics -- a uniform placeholder CarrierField keeps the
    orchestrator's carrier seam satisfied (the documented pluggable pattern)."""

    def solve(self, bp):
        ax = np.linspace(0.0, PERIOD, 3)
        z = np.linspace(0.0, T_PCM, 3)
        reg = CarrierRegion(name="pcm", role="dielectric", material="gst",
                            nodes_m=np.zeros((1, 3)), node_fields={},
                            grid_axes_m={"x": ax, "y": ax, "z": z},
                            grid_fields={ELECTRON_DENSITY: np.zeros((3, 3, 3))})
        return CarrierField(bias_label=bp.label, voltages=dict(bp.voltages), ndim=3,
                            temperature_K=300.0, regions={"pcm": reg},
                            n_bg_by_region={"pcm": 0.0}, unit_cell_m=(PERIOD, PERIOD))


class _StubGeo:
    class mesh:
        ne = 0
        nv = 0


class _StubBuilder:
    def build(self):
        return _StubGeo()

    def alignment(self):
        return GeometryAlignment(unit_scale=NM, region_alignments=[
            RegionAlignment("pcm", "pcm", (0.0, PERIOD, 0.0, PERIOD, 0.0, T_PCM),
                            stack_axis="z")], fixed_eps_regions={})

    def mesh_regions(self):
        return ["pcm"]


def _pulse_of_bias(bp):
    """Anneal at T_ANNEAL_K for bp.voltages['anneal_ns'] nanoseconds (0 -> stays amorphous)."""
    t_ns = float(bp.voltages["anneal_ns"])
    t_end = max(t_ns, 1.0) * 1e-9
    t = np.linspace(0.0, t_end, 600)
    T = np.where(t <= t_ns * 1e-9, T_ANNEAL_K, 300.0) if t_ns > 0 else np.full_like(t, 300.0)
    return t, T


def main():
    print("[pcm] === anneal pulse -> JMAK fraction -> PCMModel -> TMM R, via run_pipeline ===",
          flush=True)
    ok = True
    design = _design()
    model = PCMModel(eps_amorphous=EPS_A, eps_crystalline=EPS_C)
    fields_of_bias = pcm_extra_fields(SW, _pulse_of_bias)

    anneal_ns = [0.0, 5.0, 20.0, 2000.0]
    sweep = Sweep(bias_points=[BiasPoint(label="t{:g}ns".format(a),
                                         voltages={"anneal_ns": a}) for a in anneal_ns],
                  wavelengths_nm=[LAM_NM])
    rows = run_pipeline(design, sweep, verbose=False, carrier_solver=_StubCarrier(),
                        optical_builder=_StubBuilder(),
                        optical_solver=make_layered_tmm_solver(),
                        n_to_eps=EffectEpsMap(design.materials, effects={"gst": model}),
                        extra_fields=fields_of_bias)

    fracs = [fields_of_bias(bp)["crystalline_fraction"] for bp in sweep.bias_points]
    R = [row.result.R for row in rows]
    for a, f, r in zip(anneal_ns, fracs, R):
        print("[pcm]   anneal {:>6g} ns -> x = {:.4f}, R = {:.4f}".format(a, f, r), flush=True)

    g_a = bool(fracs[0] == 0.0 and fracs[-1] > 0.999)
    ok = ok and g_a
    print("[pcm] GATE A: fraction endpoints (x(0) = {:.3g}, x(long) = {:.4f}) -> {}".format(
        fracs[0], fracs[-1], "PASS" if g_a else "FAIL"), flush=True)

    # GATE B: hand-rolled independent path -- PCMModel eps at the SAME fraction -> direct TMM
    lam = LAM_NM * 1e-9
    worst = 0.0
    for f, r_pipe in zip(fracs, R):
        eps_f = complex(model.eps({"crystalline_fraction": f}, lam))
        stk = LayeredStack(1.0 + 0j, complex(np.sqrt(EPS_SUB)),
                           [LayeredSlab(T_PCM, eps=eps_f)],
                           period_x_m=PERIOD, period_y_m=PERIOD)
        r_direct = TmmLayeredSolver().solve(stk, lam, design.optical).R
        worst = max(worst, abs(r_pipe - r_direct))
    g_b = bool(worst < 1e-12)
    ok = ok and g_b
    print("[pcm] GATE B: pipeline R == direct PCMModel->TMM chain at every bias "
          "(worst |dR| = {:.2e}) -> {}".format(worst, "PASS" if g_b else "FAIL"), flush=True)

    g_c = bool(abs(R[-1] - R[0]) > 0.10)
    ok = ok and g_c
    print("[pcm] GATE C: amorphous <-> crystalline contrast |dR| = {:.3f} -> {}".format(
        abs(R[-1] - R[0]), "PASS" if g_c else "FAIL"), flush=True)

    print("[pcm] *** PCM SWITCHING WORKFLOW: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
