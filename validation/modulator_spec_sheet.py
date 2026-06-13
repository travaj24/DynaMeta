"""END-TO-END modulator SPEC SHEET: fuse the two validated halves of the gated ITO modulator into one
device characterization -- the project's culmination.

  ELECTRICAL (DEVSIM 2D drift-diffusion + ssac): the reference metasurface gate response -> areal capacitance
    C, the access-RC switching bandwidth f_3dB, and the per-event switching energy E = 0.5 C V^2.
  OPTICAL (FDTD dispersive broadband sweep): the gated ITO (carrier density background -> accumulation,
    free-carrier Drude eps shifting toward ENZ) -> the reflection modulation contrast dR(lambda).

analysis.modulator_figure_of_merit combines them into the spec (contrast, f_3dB, energy, contrast-per-fJ).
Each half is independently validated (resolved_bandwidth_metasurface.py ; fdtd_modulator_study.py); this
ties them into the device-level figure of merit a designer reads.

GATE: the spec assembles -- real optical contrast (peak |dR| > 0.05), a finite positive RC bandwidth, a
finite switching energy, and a finite figure of merit. (Both physics halves are validated against their
own oracles -- TMM and quasi-static dQ/dV -- in their dedicated scripts; this script checks the fused
device spec is well-formed and physical, not the per-half physics again.)

Run: python -m validation.modulator_spec_sheet
"""
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from dynameta.analysis import modulator_figure_of_merit, sheet_resistance_ohm_sq
from dynameta.carriers import ac_analysis as AC
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
from dynameta.optics.fdtd_seam import run_fdtd_sweep
from dynameta.optics.tmm_reference import layered_rta, layered_stack_from_design
from dynameta.sweep import BiasPoint
from validation._reference_device import build_reference_modulator

from validation.fdtd_modulator_study import _design as _optical_design

VG, GATE = 1.0, "top_contact"
MU, T_ITO, PERIOD = 30e-4, 5e-9, 370e-9
N_OFF, N_ON = 4.0e20, 1.2e21        # cm^-3 design-target ITO densities for the ENZ optical modulation


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush(); saved = os.dup(1); devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(devnull); os.close(saved)


def _electrical_gate_C_per_area():
    """DEVSIM 2D DD + ssac on the reference metasurface -> the gate areal capacitance [F/m^2]."""
    d = build_reference_modulator("drift_diffusion")
    b = LayeredDevsimBuilder(d, mesh_name="spec_m", device_name="spec_d")
    with _quiet():
        b.solve(BiasPoint({GATE: VG}, "g{:+.0f}".format(VG)))
        b.set_ssac_gate(GATE, source_name="V1")
        ds.circuit_alter(name="V1", value=float(VG))
        ds.solve(type="dc", solver_type="direct", absolute_error=1e18, relative_error=1e-5,
                 maximum_iterations=60)
        _, C, _ = AC.ssac_admittance([1.0e6], source_name="V1")     # F/m (2D per unit y)
    return float(np.mean(C)) / PERIOD                                # F/m^2 (divide out the cell x-extent)


def _optical_contrast():
    """FDTD dispersive sweep of the gated ITO at two carrier densities -> peak reflection modulation."""
    targets = np.linspace(1250e-9, 1650e-9, 9)
    off = run_fdtd_sweep(_optical_design(N_OFF), targets, dim=2, resolution=30, band_pad=0.12)
    on = run_fdtd_sweep(_optical_design(N_ON), targets, dim=2, resolution=30, band_pad=0.12)
    dmod = np.array([on[i].R - off[i].R for i in range(targets.size)])
    i_pk = int(np.argmax(np.abs(dmod)))
    return float(np.abs(dmod[i_pk])), float(targets[i_pk] * 1e9)


def main():
    print("[ms] === END-TO-END modulator spec sheet (DEVSIM bandwidth + FDTD contrast) ===", flush=True)

    print("[ms] [electrical] DEVSIM 2D DD + ssac on the reference metasurface ...", flush=True)
    C_area = _electrical_gate_C_per_area()
    rho_s = sheet_resistance_ohm_sq(N_OFF * 1e6, MU, T_ITO)

    print("[ms] [optical] FDTD dispersive sweep, ITO bias {:.0e}->{:.0e} cm^-3 ...".format(N_OFF, N_ON),
          flush=True)
    contrast, lam_pk = _optical_contrast()

    spec = modulator_figure_of_merit(
        optical_contrast=contrast, contrast_lambda_nm=lam_pk, gate_C_per_area_F_m2=C_area,
        voltage_swing_V=2.0 * VG, sheet_resistance_ohm_sq=rho_s, path_length_m=5e-6, pad_width_m=1e-6,
        cell_area_m2=PERIOD ** 2)

    print("[ms]", flush=True)
    print("[ms] ===== ITO METASURFACE MODULATOR SPEC =====", flush=True)
    print("[ms]   optical contrast |dR|   : {:.3f}  @ {:.0f} nm".format(
        spec["optical_contrast"], spec["contrast_lambda_nm"]), flush=True)
    print("[ms]   gate capacitance        : {:.3f} fF/cell  ({:.3e} F/m^2)".format(spec["gate_C_fF"], C_area),
          flush=True)
    print("[ms]   switching bandwidth     : {:.1f} GHz (access-RC)".format(spec["f_3dB_GHz"]), flush=True)
    print("[ms]   switching energy        : {:.3f} fJ/event".format(spec["switching_energy_fJ"]), flush=True)
    print("[ms]   figure of merit         : {:.3f} contrast / fJ".format(spec["contrast_per_fJ"]), flush=True)
    print("[ms]   (ENZ caveat: a few-nm accumulation is FDTD-underresolved; tight ENZ -> FEM/RCWA)", flush=True)

    gate = bool(spec["optical_contrast"] > 0.05 and spec["f_3dB_GHz"] > 0.0 and
                np.isfinite(spec["switching_energy_fJ"]) and np.isfinite(spec["contrast_per_fJ"]) and
                spec["switching_energy_fJ"] > 0.0)
    print("[ms]", flush=True)
    print("[ms] *** MODULATOR SPEC SHEET (electrical bandwidth + optical contrast fused): {} ***".format(
        "PASS" if gate else "FAIL"), flush=True)
    return gate


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
