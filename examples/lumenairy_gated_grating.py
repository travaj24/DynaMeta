"""
Example: gated ITO ENZ modulator under a gold LAMELLAR grating, solved through the
Lumenairy RCWA bridge -- the v0.5 integration flagship.

The device is reference-2021-style gate cavity (Al-Nd mirror / oxide-ITO-oxide / Au lines)
with the nanopatch swapped for y-invariant grating lines, so the SAME Design exercises:

  1. real DEVSIM equilibrium carriers (ITO accumulation vs gate bias),
  2. the graded n(z) -> Drude eps(z) -> sliced-EpsField chain into the RCWA bridge,
  3. the lamellar 1-D RCWA fast path (full-y Rectangle lines, TM polarization),
  4. per-layer absorption attribution (absorption=True -> WHERE the loss goes vs bias),
  5. the PMM bridge as a cross-method referee on the same geometry (ungated: the biased
     accumulation is laterally structured under the lines -- RCWA territory; the PMM
     analytic-segment scope raises on rasterized lateral structure by contract).

Requires lumenairy (a core dynameta dependency) + devsim (`dynameta[solvers]`); exits 0
with a SKIP banner when either is missing from the environment.

Run:
    python -m examples.lumenairy_gated_grating            # 2 biases x 4 wavelengths
    python -m examples.lumenairy_gated_grating --quick    # 1 bias, 1 wavelength
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynameta.core.eps_field import EpsField
from dynameta.geometry import Design, Electrode, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import (M_E, ConstantOptical, DrudeOptical, Material,
                                MaterialRegistry, TransportModel)
from dynameta.pipeline import run_pipeline
from dynameta.sweep import BiasPoint, Sweep

Q_E = 1.602176634e-19
PERIOD = 600e-9
DUTY = 0.5

# ITO Drude (representative near-IR ITO data fit) -- background ENZ near 1630 nm; accumulation at
# positive gate bias pushes the local ENZ shorter inside the 5 nm channel.
ITO_N_BG = 4.0e20 * 1e6
ITO_M_OPT = 0.225 * M_E
ITO_GAMMA = 1.1e14
ITO_M_LOW = 0.27 * M_E
ITO_KANE_ALPHA = 0.5


def ito_dos_mass(n_m3):
    n = np.maximum(np.asarray(n_m3, dtype=np.float64), 1e10)
    HBAR = 1.054571817e-34
    KF = (3.0 * np.pi ** 2 * n) ** (1.0 / 3.0)
    E_F = HBAR ** 2 * KF ** 2 / (2.0 * ITO_M_LOW)
    return ITO_M_LOW * np.sqrt(1.0 + 2.0 * ITO_KANE_ALPHA * E_F / Q_E)


def build_materials() -> MaterialRegistry:
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("Si", ConstantOptical(12.0 + 0j)))
    reg.add(Material("Al2O3", ConstantOptical(2.756 + 0j), eps_static_dc=9.0))
    reg.add(Material("HfO2", ConstantOptical(4.0 + 0j), eps_static_dc=18.0))
    reg.add(Material("Al-Nd", ConstantOptical(-180 + 30j), is_metal=True))
    reg.add(Material("Au", ConstantOptical(-100 + 8j), is_metal=True))
    reg.add(Material("ITO",
                     optical=DrudeOptical(eps_inf=4.25, m_opt_kg=ITO_M_OPT,
                                          gamma_rad_s=ITO_GAMMA),
                     transport=TransportModel(n_bg_m3=ITO_N_BG, eps_static=9.5,
                                              dos_mass_kg_of_n_m3=ito_dos_mass,
                                              band_gap_eV=3.6, chi_eV=4.5,
                                              physics="equilibrium"),
                     pretty_name="ITO (indium tin oxide)"))
    return reg


def build_design() -> Design:
    cell = UnitCell.square(PERIOD)
    # y-invariant Au lines: a Rectangle spanning the FULL y period -- the geometry both
    # bridges special-case (RCWA: lamellar 1-D fast path; PMM: analytic segments).
    lines = Rectangle(PERIOD / 2.0, PERIOD / 2.0, DUTY * PERIOD, PERIOD)
    layers = [
        Layer("mirror", 70e-9, "Al-Nd"),
        Layer("lower_al2o3", 1e-9, "Al2O3"),
        Layer("lower_hfo2", 7e-9, "HfO2"),
        Layer("ito", 5e-9, "ITO"),
        Layer("upper_hfo2", 7e-9, "HfO2"),
        Layer("upper_al2o3", 1e-9, "Al2O3"),
        Layer("grating", 50e-9, "air", inclusions=[Inclusion(lines, "Au")]),
    ]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="Si")
    electrodes = [
        Electrode("bot_contact", "mirror", "full", role="biased"),
        Electrode("top_contact", "grating", lines, role="biased"),
        Electrode("ito_gnd_left", "ito", "x_lo", role="ground", fixed_voltage_V=0.0),
        Electrode("ito_gnd_right", "ito", "x_hi", role="ground", fixed_voltage_V=0.0),
    ]
    # E_x across the lines = TM, the strongly-modulated (and numerically hard) case.
    return Design(name="lumenairy_gated_grating", unit_cell=cell, stack=stack,
                  electrodes=electrodes, materials=build_materials(),
                  optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0),
                  pretty_name="Gated ITO ENZ modulator under a Au lamellar grating")


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    missing = [m for m in ("lumenairy", "devsim") if importlib.util.find_spec(m) is None]
    if missing:
        print("[lgg] *** SKIP: {} not installed -- example not run ***".format(
            ", ".join(missing)), flush=True)
        return 0
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm_solver,
                                                  make_lumenairy_rcwa_solver)

    print("[lgg] === gated ITO grating: DEVSIM carriers -> Lumenairy RCWA bridge ===",
          flush=True)
    ok = True
    design = build_design()
    biases = [BiasPoint({"top_contact": +2.0, "bot_contact": +2.0}, "gate+2V"),
              BiasPoint({"top_contact": -2.0, "bot_contact": -2.0}, "gate-2V")]
    lams_nm = [1400.0] if args.quick else [1400.0, 1500.0, 1600.0, 1700.0]
    if args.quick:
        biases = biases[:1]
    sweep = Sweep(bias_points=biases, wavelengths_nm=lams_nm)

    # n_slices fixed so the RCWA path and the PMM referee see the IDENTICAL staircase of
    # the graded ITO eps(z); absorption=True turns on per-layer attribution.
    rcwa = make_lumenairy_rcwa_solver(n_orders=24, n_slices=12, absorption=True)
    rows = run_pipeline(design, sweep, verbose=True, optical_solver=rcwa)

    # per-layer absorption: where does the pump go as the gate accumulates the channel?
    for row in rows:
        pra = row.result.per_region_absorption or {}
        top = sorted(pra.items(), key=lambda kv: -abs(kv[1]))[:4]
        print("[lgg]   {} lam={:.0f}nm  A={:.4f}  ({})".format(
            row.bias_label, row.lambda_nm, row.result.A,
            ", ".join("{} {:.4f}".format(k, v) for k, v in top)), flush=True)

    # GATE A: attribution closes on the energy budget at every point
    worst_close = max(abs(sum((r.result.per_region_absorption or {}).values())
                          - r.result.A) for r in rows)
    g_a = bool(worst_close < 1e-6)
    ok = ok and g_a
    print("[lgg] GATE A: per-layer absorption closes on A = 1 - R - T at every "
          "(bias, wavelength) (worst |d| = {:.1e}) -> {}".format(
              worst_close, "PASS" if g_a else "FAIL"), flush=True)

    # GATE B: PMM referee on the same GEOMETRY, ungated (ITO at its background Drude
    # density). The biased accumulation is laterally structured (gated only under the
    # lines) -- RCWA territory; the PMM bridge's analytic-segment scope raises on
    # rasterized lateral structure, so the cross-method check runs the flat-band device.
    lam0 = lams_nm[0] * 1e-9
    ef0 = {"ito": EpsField(scalar=complex(
        design.materials.get("ITO").eps(lam0, n_m3=ITO_N_BG)))}
    r_rcwa = make_lumenairy_rcwa_solver(n_orders=24)(design, None, ef0, lam0,
                                                     1.0 + 0j, np.sqrt(12.0) + 0j)
    r_pmm = make_lumenairy_pmm_solver(degree=14, n_orders=15)(design, None, ef0, lam0,
                                                              1.0 + 0j, np.sqrt(12.0) + 0j)
    d_ref = abs(r_rcwa.R - r_pmm.R)
    g_b = bool(d_ref < 5e-3)
    ok = ok and g_b
    print("[lgg] GATE B: RCWA bridge vs PMM referee on the ungated device "
          "(R {:.4f} vs {:.4f}, |d| = {:.1e}) -> {}".format(
              r_rcwa.R, r_pmm.R, d_ref, "PASS" if g_b else "FAIL"), flush=True)

    # GATE C: the gate actually modulates -- R differs between +2V and -2V somewhere
    if not args.quick:
        by_bias = {}
        for r in rows:
            by_bias.setdefault(r.bias_label, {})[r.lambda_nm] = r.result.R
        dR = max(abs(by_bias["gate+2V"][w] - by_bias["gate-2V"][w]) for w in lams_nm)
        g_c = bool(dR > 1e-4)
        ok = ok and g_c
        print("[lgg] GATE C: gate modulation max |R(+2V) - R(-2V)| = {:.2e} -> {}".format(
            dR, "PASS" if g_c else "FAIL"), flush=True)

    print("[lgg] *** LUMENAIRY GATED GRATING: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
