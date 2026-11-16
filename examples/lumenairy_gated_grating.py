"""
Example: a gated ENZ film under a metal LAMELLAR grating, solved through the
Lumenairy RCWA bridge -- the v0.5 integration flagship.

The device is a gate-cavity design (metal mirror / oxide-film-oxide / metal lines)
with the nanopatch swapped for y-invariant grating lines, so the SAME Design exercises:

  1. real DEVSIM equilibrium carriers (film accumulation vs gate bias),
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

# The generic transparent-conductor film (tests/_strip_cell_fixture.py; background ENZ near 2.1 um);
# accumulation at positive gate bias pushes the local ENZ shorter inside the 6 nm film.
FILM_N_BG = 3.0e26
FILM_M_OPT = 0.30 * M_E
FILM_GAMMA = 1.2e14
FILM_M_LOW = 0.25 * M_E
FILM_KANE_ALPHA = 0.4
N_GLASS = 1.45


def film_dos_mass(n_m3):
    n = np.maximum(np.asarray(n_m3, dtype=np.float64), 1e10)
    HBAR = 1.054571817e-34
    KF = (3.0 * np.pi ** 2 * n) ** (1.0 / 3.0)
    E_F = HBAR ** 2 * KF ** 2 / (2.0 * FILM_M_LOW)
    return FILM_M_LOW * np.sqrt(1.0 + 2.0 * FILM_KANE_ALPHA * E_F / Q_E)


def build_materials() -> MaterialRegistry:
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(N_GLASS ** 2 + 0j)))
    reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j), eps_static_dc=9.0))
    reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j), eps_static_dc=20.0))
    reg.add(Material("mirror_metal", ConstantOptical(-120 + 20j), is_metal=True))
    reg.add(Material("metal", ConstantOptical(-60 + 4j), is_metal=True))
    reg.add(Material("TCO",
                     optical=DrudeOptical(eps_inf=4.0, m_opt_kg=FILM_M_OPT,
                                          gamma_rad_s=FILM_GAMMA),
                     transport=TransportModel(n_bg_m3=FILM_N_BG, eps_static=9.0,
                                              dos_mass_kg_of_n_m3=film_dos_mass,
                                              band_gap_eV=3.5, chi_eV=4.5,
                                              physics="equilibrium"),
                     pretty_name="TCO (generic transparent-conductor film)"))
    return reg


def build_design() -> Design:
    cell = UnitCell.square(PERIOD)
    # y-invariant metal lines: a Rectangle spanning the FULL y period -- the geometry both
    # bridges special-case (RCWA: lamellar 1-D fast path; PMM: analytic segments).
    lines = Rectangle(PERIOD / 2.0, PERIOD / 2.0, DUTY * PERIOD, PERIOD)
    layers = [
        Layer("mirror", 80e-9, "mirror_metal"),
        Layer("lo_oxa", 2e-9, "oxA"),
        Layer("lo_oxb", 6e-9, "oxB"),
        Layer("film", 6e-9, "TCO"),
        Layer("up_oxb", 6e-9, "oxB"),
        Layer("up_oxa", 2e-9, "oxA"),
        Layer("grating", 40e-9, "air", inclusions=[Inclusion(lines, "metal")]),
    ]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="glass")
    electrodes = [
        Electrode("bot_contact", "mirror", "full", role="biased"),
        Electrode("top_contact", "grating", lines, role="biased"),
        Electrode("film_gnd_left", "film", "x_lo", role="ground", fixed_voltage_V=0.0),
        Electrode("film_gnd_right", "film", "x_hi", role="ground", fixed_voltage_V=0.0),
    ]
    # E_x across the lines = TM, the strongly-modulated (and numerically hard) case.
    return Design(name="lumenairy_gated_grating", unit_cell=cell, stack=stack,
                  electrodes=electrodes, materials=build_materials(),
                  optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0),
                  pretty_name="Gated ENZ film under a metal lamellar grating")


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

    print("[lgg] === gated-film grating: DEVSIM carriers -> Lumenairy RCWA bridge ===",
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
    # the graded film eps(z); absorption=True turns on per-layer attribution.
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

    # GATE B: PMM referee on the same GEOMETRY, ungated (film at its background Drude
    # density). The biased accumulation is laterally structured (gated only under the
    # lines) -- RCWA territory; the PMM bridge's analytic-segment scope raises on
    # rasterized lateral structure, so the cross-method check runs the flat-band device.
    lam0 = lams_nm[0] * 1e-9
    ef0 = {"film": EpsField(scalar=complex(
        design.materials.get("TCO").eps(lam0, n_m3=FILM_N_BG)))}
    # The RCWA default resolves this TM metal grating to Li's rule. MEASURED against the 2-D TM FEM
    # (optics.fem_2d_tm, R 0.9449520 at 1400 nm, identical to 7 digits from order 5 to 7 and with every
    # corner graded), |dR|:
    #   n_orders            24       48       96       192
    #   'li' (default)    7.1e-4   5.6e-4   4.8e-4   2.3e-4
    #   Laurent           1.8e-3   1.2e-3   1.6e-4   5.2e-4    (not monotone)
    # and the PMM referee: degree 14 / 24 / 32 / 48: 1.5e-4 / 4.9e-5 / 2.6e-5 / 1.1e-5. With n_orders 96 and
    # degree 32 both routes sit within 5e-4 of the FEM, so their difference is below 1e-3; the bar is 2e-3.
    r_rcwa = make_lumenairy_rcwa_solver(n_orders=96)(design, None, ef0, lam0,
                                                     1.0 + 0j, N_GLASS + 0j)
    r_pmm = make_lumenairy_pmm_solver(degree=32, n_orders=15)(design, None, ef0, lam0,
                                                              1.0 + 0j, N_GLASS + 0j)
    d_ref = abs(r_rcwa.R - r_pmm.R)
    g_b = bool(d_ref < 2e-3)
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
