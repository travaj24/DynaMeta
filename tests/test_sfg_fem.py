"""Gates for the nondegenerate two-step FEM sum/difference-frequency solver (roadmap item 4.3):
optics/shg_fem.sfg_two_step + rudnick_stern_flat_sfg. ngsolve-gated (the pattern of
tests/test_shg_fem.py, whose degenerate SHG two-step this generalizes).

Physics + conventions (see optics/shg_fem.py, "NONDEGENERATE THREE-WAVE MIXING" block):
  Boyd D-factor: P_z(omega3) = eps0 chi_zzz D E_z(omega1) E_z(omega2), D = 2 for the nondegenerate
  SFG/DFG process (the two field orderings (1,2)+(2,1)) vs the degenerate SHG convention
  P_z(2w) = eps0 chi_zzz E_z(w)^2 (D = 1). The /4 in the degeneracy identity below is the
  amplitude-2 -> power-4 map, not a switch of D.
  DFG conjugation (exp(-i omega t)): the -omega2 field component is conj(E(+omega2)), so DFG uses
  P_z ~ E1 conj(E2) and K_par3 = k_par1 - k_par2.
  Citations: Rudnick & Stern, PRB 4, 4274 (1971); Heinz/Sipe interface BCs (Sipe, JOSA B 4, 481
  (1987); Sipe/So/Fukui/Stegeman, PRB 21, 4389 (1980)); Boyd, "Nonlinear Optics" ch. 2.

GATES:
  1. DEGENERACY IDENTITY (load-bearing): sfg_two_step(w, w)/4 == shg_two_step(w) for the radiated
     power, pinned to 1e-10 relative (they share every code path but the D-factor). Plus the
     analytic cross-check rudnick_stern_flat_sfg(w,w)/4 == rudnick_stern_flat_shg (machine eps).
  2. NONDEGENERATE flat surface: sfg_two_step's p_up_3w vs rudnick_stern_flat_sfg * cell area at 2
     angle pairs, < 10% (oblique-PML-limited; measured ~0.4-0.6%).
  3. BILINEAR SLOPES: P(omega3) linear in each fundamental intensity (slope 1.00 +/- 0.02 each;
     exact by construction, validates normalization).
  4. DFG MOMENTUM: the radiated field carries the DIFFERENCE wavevector k_par1 - k_par2 (extracted
     from the sourced-solve field phase), matched to 5% and clearly distinct from the SFG sum.
  5. SELECTION RULE: s-pol fundamentals give zero a-term mixing for BOTH processes.
"""

import math

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.constants import EPS0, C_LIGHT                                       # noqa: E402
from dynameta.optics.shg_fem import (rudnick_stern_flat_sfg,                       # noqa: E402
                                      rudnick_stern_flat_shg, sfg_field_transverse_kx)

Z0 = 1.0 / (EPS0 * C_LIGHT)

LAM = 1000e-9                     # a reference fundamental wavelength (m)
EPS_W = complex(-40.0, 2.5)      # Drude metal (gold-like near 1 um); ConstantOptical => same at 2w
CHI = 1.0e-20                    # representative normal surface susceptibility (m^2/V)
W1 = 2.0 * math.pi * C_LIGHT / 1000e-9    # color 1 (rad/s)
W2 = 2.0 * math.pi * C_LIGHT / 1200e-9    # color 2 (rad/s); W1 > W2 so DFG omega3 > 0


def _flat_gold_design(period_m, theta_deg):
    """A flat Drude half-space cell (vacuum cap + gold slab, gold substrate), matching the shipped
    SHG quantitative gate's coarse mesh (tests/test_shg_fem.test_flat_shg_fem_quantitative)."""
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Design
    from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec

    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("gold", ConstantOptical(EPS_W)))
    stack = Stack(layers=[Layer("metalL", 400e-9, "gold"), Layer("capL", 300e-9, "air")],
                  superstrate_material="air", substrate_material="gold")
    m3 = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=800e-9, substrate_buffer_m=250e-9,
                    maxh_superstrate_m=80e-9, maxh_substrate_m=90e-9, maxh_metal_m=25e-9,
                    maxh_background_m=80e-9, maxh_pml_m=170e-9)
    opt = OpticalSpec(polarization="p", incidence_angle_deg=theta_deg, linear_solver="umfpack")
    return Design(name="sfg_flat", unit_cell=UnitCell.square(period_m), stack=stack,
                  electrodes=[], materials=reg, mesh_3d=m3, optical=opt)


# --------------------------------------------------------------------------------------------
# GATE 1 -- the DEGENERACY IDENTITY (load-bearing)
# --------------------------------------------------------------------------------------------
def test_analytic_degeneracy_identity():
    """Analytic (fast) half of the identity: the nondegenerate oracle at equal frequencies/angles,
    divided by 4, reproduces the degenerate SHG oracle EXACTLY (D = 2 -> amplitude x2 -> power x4).
    This pins the permutation-factor bookkeeping independent of the FEM."""
    w = 2.0 * math.pi * C_LIGHT / LAM
    for th in (20.0, 35.0, 50.0):
        sfg = rudnick_stern_flat_sfg(w, w, th, th, EPS_W, EPS_W, EPS_W, CHI, process="sfg")["S_up"]
        shg = rudnick_stern_flat_shg(LAM, th, EPS_W, EPS_W, CHI)["S_up"]
        assert abs(sfg / 4.0 - shg) <= 1e-12 * shg, (th, sfg / 4.0, shg)


def test_fem_degeneracy_identity():
    """THE load-bearing gate: the FULL FEM two-step at equal colors, sfg_two_step(w, w)/4, equals
    shg_two_step(w) for the radiated power to 1e-10 relative. sfg_two_step is the D = 2 nondegenerate
    path; feeding it equal fields doubles the sheet AMPLITUDE and quadruples the radiated POWER, and
    every other code path (metal-field sampling, sheet vacuum field, sourced solve, port extraction)
    is byte-shared with shg_two_step -- so the ratio must be nearly exact."""
    from dynameta.optics.shg_fem import shg_two_step, sfg_two_step

    th = 25.0
    w = 2.0 * math.pi * C_LIGHT / LAM
    d = _flat_gold_design(220e-9, th)
    shg = shg_two_step(d, lambda_fund_m=LAM, chi_zzz=CHI)
    sfg = sfg_two_step(d, omega1_rad_s=w, omega2_rad_s=w, chi_zzz=CHI,
                       theta1_deg=th, theta2_deg=th, process="sfg")
    assert sfg["D_factor"] == 2.0
    # the per-color normal fields must equal the SHG one exactly (analytic Sipe closed form)
    assert abs(sfg["E_perp_in_1"] - shg["E_perp_in"]) < 1e-12 * abs(shg["E_perp_in"])
    assert abs(sfg["E_perp_in_2"] - shg["E_perp_in"]) < 1e-12 * abs(shg["E_perp_in"])
    rel = abs(sfg["p_up_3w"] / 4.0 - shg["p_up_2w"]) / shg["p_up_2w"]
    assert rel < 1e-10, "sfg/4={:.6e} vs shg={:.6e} (rel {:.2e})".format(
        sfg["p_up_3w"] / 4.0, shg["p_up_2w"], rel)


# --------------------------------------------------------------------------------------------
# GATE 2 -- nondegenerate flat surface vs the generalized analytic oracle
# --------------------------------------------------------------------------------------------
def test_nondegenerate_flat_vs_oracle():
    """SFG of two distinct colors on a flat Drude mirror: sfg_two_step's p_up_3w matches the
    generalized analytic oracle rudnick_stern_flat_sfg * cell area to < 10% at two angle pairs
    (measured ~0.4-0.6%; residual is the documented fixed-alpha z-PML oblique approximation, now at
    THREE wavelengths). ConstantOptical => eps equal at both colors and at omega3."""
    from dynameta.optics.shg_fem import sfg_two_step

    period = 220e-9
    for (t1, t2) in ((20.0, 30.0), (30.0, 40.0)):
        d = _flat_gold_design(period, t1)
        out = sfg_two_step(d, omega1_rad_s=W1, omega2_rad_s=W2, chi_zzz=CHI,
                           theta1_deg=t1, theta2_deg=t2, process="sfg")
        orc = rudnick_stern_flat_sfg(W1, W2, t1, t2, EPS_W, EPS_W, EPS_W, CHI, process="sfg")
        p_an = orc["S_up"] * period ** 2
        assert p_an > 0.0
        rel = abs(out["p_up_3w"] - p_an) / p_an
        assert rel < 0.10, "pair ({},{}): FEM p_up_3w={:.4e} vs analytic {:.4e} (rel {:.3f})".format(
            t1, t2, out["p_up_3w"], p_an, rel)
        # the FEM emission angle must equal the oracle's momentum-matched theta3
        assert abs(out["theta3_deg"] - orc["theta3_deg"]) < 1e-6


# --------------------------------------------------------------------------------------------
# GATE 3 -- bilinear pump slopes (P(omega3) linear in EACH fundamental intensity)
# --------------------------------------------------------------------------------------------
def test_bilinear_slopes():
    """SFG radiated power is BILINEAR: linear in intensity I1 at fixed I2, and linear in I2 at fixed
    I1 (slope 1.00 +/- 0.02 each). Exact by construction (P_z ~ E1 E2 -> S_up ~ |E1|^2 |E2|^2 ~
    I1 I2); validates the two-color normalization pipeline. Oracle-level (prefactor-independent)."""
    t1, t2 = 20.0, 30.0
    amps = [0.5, 1.0, 2.0, 4.0, 8.0]
    # slope in I1 at fixed I2
    S1 = [rudnick_stern_flat_sfg(W1, W2, t1, t2, EPS_W, EPS_W, EPS_W, CHI, E0_1=a)["S_up"]
          for a in amps]
    I1 = [0.5 * EPS0 * C_LIGHT * a ** 2 * math.cos(math.radians(t1)) for a in amps]
    slope1 = np.polyfit(np.log(I1), np.log(S1), 1)[0]
    assert abs(slope1 - 1.0) < 0.02, slope1
    # slope in I2 at fixed I1
    S2 = [rudnick_stern_flat_sfg(W1, W2, t1, t2, EPS_W, EPS_W, EPS_W, CHI, E0_2=a)["S_up"]
          for a in amps]
    I2 = [0.5 * EPS0 * C_LIGHT * a ** 2 * math.cos(math.radians(t2)) for a in amps]
    slope2 = np.polyfit(np.log(I2), np.log(S2), 1)[0]
    assert abs(slope2 - 1.0) < 0.02, slope2


# --------------------------------------------------------------------------------------------
# GATE 4 -- DFG conjugation + difference-wavevector momentum
# --------------------------------------------------------------------------------------------
def test_dfg_conjugation_convention():
    """DFG (exp(-i omega t)): the omega2 field enters CONJUGATED (the -omega2 component is
    conj(E(+omega2))), so the omega2 normal field used by DFG is the complex conjugate of the SFG
    one, and omega3 = omega1 - omega2. Documents + pins the conjugation convention (analytic)."""
    t1, t2 = 30.0, 30.0
    sfg = rudnick_stern_flat_sfg(W1, W2, t1, t2, EPS_W, EPS_W, EPS_W, CHI, process="sfg")
    dfg = rudnick_stern_flat_sfg(W1, W2, t1, t2, EPS_W, EPS_W, EPS_W, CHI, process="dfg")
    assert abs(sfg["E_perp_in_2"].imag) > 1e-6 * abs(sfg["E_perp_in_2"])   # lossy metal -> complex
    assert abs(dfg["E_perp_in_2"] - sfg["E_perp_in_2"].conjugate()) < 1e-15 * abs(sfg["E_perp_in_2"])
    assert abs(dfg["omega3"] - (W1 - W2)) < 1e-3
    assert abs(sfg["omega3"] - (W1 + W2)) < 1e-3


def test_dfg_difference_momentum():
    """DFG momentum: the sourced-solve field radiates with the in-plane wavevector K_par1 - K_par2
    (extracted from the total-field x-phase in the superstrate), matched to 5% and clearly distinct
    from the SFG sum K_par1 + K_par2. Equal fundamental angles keep the difference emission
    propagating (theta3 = theta). Idler at ~3.5 um from 1.0/1.4 um pumps."""
    from dynameta.optics.shg_fem import sfg_two_step

    w2_dfg = 2.0 * math.pi * C_LIGHT / 1400e-9
    th = 45.0
    period = 500e-9
    d = _flat_gold_design(period, th)
    out = sfg_two_step(d, omega1_rad_s=W1, omega2_rad_s=w2_dfg, chi_zzz=CHI,
                       theta1_deg=th, theta2_deg=th, process="dfg")
    S = 1e9
    kpar1 = (2.0 * math.pi / (1000e-9 * S)) * math.sin(math.radians(th))   # nm^-1
    kpar2 = (2.0 * math.pi / (1400e-9 * S)) * math.sin(math.radians(th))
    k_diff = kpar1 - kpar2
    k_sum = kpar1 + kpar2
    # the code's bookkeeping equals the difference wavevector
    assert abs(out["K_par3_per_nm"] - k_diff) < 1e-9 * abs(k_diff)
    # and the RADIATED field carries it (independent field-phase extraction)
    kx_meas = sfg_field_transverse_kx(out)
    assert abs(kx_meas - k_diff) < 0.05 * abs(k_diff), (kx_meas, k_diff)
    # clearly the DIFFERENCE, not the sum
    assert abs(kx_meas - k_diff) < 0.2 * abs(kx_meas - k_sum)


# --------------------------------------------------------------------------------------------
# GATE 5 -- s-pol selection rule (no normal E -> no a-term mixing), BOTH processes
# --------------------------------------------------------------------------------------------
def test_spol_selection_rule_both_processes():
    """s-pol fundamentals have no normal E, so the a-term (chi_zzz) mixing vanishes IDENTICALLY for
    both SFG and DFG; a single s-pol beam is enough (P_z ~ E1 E2). Exact-zero selection rule."""
    t1, t2 = 20.0, 30.0
    for proc in ("sfg", "dfg"):
        both_s = rudnick_stern_flat_sfg(W1, W2, t1, t2, EPS_W, EPS_W, EPS_W, CHI,
                                        process=proc, polarization="s")
        p = rudnick_stern_flat_sfg(W1, W2, t1, t2, EPS_W, EPS_W, EPS_W, CHI, process=proc)
        assert both_s["S_up"] == 0.0
        assert abs(both_s["E_perp_in_1"]) == 0.0 and abs(both_s["E_perp_in_2"]) == 0.0
        assert p["S_up"] > 1e6 * max(both_s["S_up"], 1e-300)   # p-pol strongly nonzero
