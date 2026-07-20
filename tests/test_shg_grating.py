"""Gates for the STRUCTURED-surface (grating) surface-SHG driver (roadmap item 5.3):
optics/shg_fem.shg_structured_two_step. ngsolve-gated (the pattern of tests/test_shg_fem.py, whose
FLAT shg_two_step this generalizes to a corrugated metal top boundary).

Physics + the honest FEM finding (see the "STRUCTURED-SURFACE" block in optics/shg_fem.py):
  The flat driver samples the fundamental normal field ANALYTICALLY and radiates the sheet as a
  single analytic plane wave. A structured surface has neither luxury. Direct measurement (ngsolve
  6.2.2604) shows the roadmap-suggested surface-current route CANNOT source the Rudnick-Stern NORMAL
  sheet in an HCurl discretization -- a boundary form (nhat . v.Trace()) is IDENTICALLY ZERO and the
  full normal trace (nhat . v) is REFUSED -- and a thin volume-current band needs sub-element
  resolution the coarse ngsolve-gated meshes cannot give (15-200% noise; per-solid maxh is silently
  ignored in this build, only a global maxh refines, which explodes the cell). So the driver uses the
  COARSE-MESH-ROBUST scattered-field route shg_two_step already uses:
    (a) fundamental FEM solve on the structured cell (reconstructed via solve_fem_sourced so the
        field is exposed; solve_fem returns only R/T);
    (b) extract E_perp along the surface by point-sampling E_z in the DIELECTRIC at small standoffs
        and fitting the two-wave field to the boundary (the CONTINUOUS normal-D route; E_inside =
        D_perp / eps_metal). Measured extraction noise vs the Sipe closed form on a flat mirror:
        ~0.2% at 20-35 deg;
    (c) assemble the Rudnick-Stern normal sheet P_perp(x) = eps0 chi E_perp^2 along the surface and
        Fourier-decompose it (including the leading surface-height phase) into diffraction orders c_m;
    (d) radiate each order as an analytic normal-dipole-sheet plane wave and (radiate=True) scatter it
        off the structured metal at 2w via solve_fem_sourced, extracting SH per order.

GATES:
  1. FLAT LIMIT (load-bearing): a geometrically flat cell reproduces shg_two_step's radiated power to
     < 2% (only m = 0 survives; E0 collapses to the single analytic sheet plane wave). This validates
     the trace-extraction + sheet-assembly against the analytic-sampling path.
  2. SLOPE 2: the SH power is quadratic in the sheet strength (chi), slope 2.00 -- P ~ |eps0 chi
     E_perp^2|^2 ~ chi^2 (equivalently ~ I_w^2 at fixed geometry); validates the |sheet|^2 -> power
     normalization pipeline through the sourced solve.
  3. SHALLOW GRATING: a shallow lamellar corrugation (metal tooth, depth h << period, lambda). The
     surface-sheet nonspecular Fourier amplitude |c_{+-1}| grows LINEARLY in h (perturbation theory)
     -> the m = +-1 SHEET SH power ~ h^2 (log-log slope 2, gated within 25%); the m = 0 (specular)
     sheet stays within ~10% of flat for the shallowest h. The gate uses the SHEET-order power (the
     clean physical-optics surface-SHG trend); the full FEM sourced +-1 power ADDITIONALLY carries
     linear SH re-diffraction off the structured metal (a real but separate effect, non-perturbative
     for a strong 500-nm gold scatterer), so it is exercised end-to-end but not magnitude-gated here.
  4. s-POL SELECTION RULE: on a structured surface an s-pol fundamental generates a residual normal
     field only at tilted facets (h-suppressed), so rather than exact zero we gate the SUPPRESSION of
     s-pol vs p-pol SH; on a flat cell the residual is pure extraction noise and s/p << 1.
  5. DADAP (documented fallback -- not delivered): the small-cylinder multipole check (Dadap et al.,
     PRL 83, 4045 (1999)) needs the a-term sheet on a strongly-curved surface whose normal is RADIAL
     (in-plane) -- the driver here builds the a-term from the z-component of the normal field, exact
     for near-planar corrugations (gate 3) but blind to the radial-normal SH that dominates an
     isolated cylinder. A faithful Dadap check needs a conformal sheet source, which the HCurl
     normal-sheet limitation above precludes on coarse meshes. Per the roadmap's honest-fallback
     allowance the grating gates (1-4) stand alone. See the module docstring for the full derivation.

Citations: Rudnick & Stern, PRB 4, 4274 (1971); Heinz / Sipe interface BCs (Sipe, JOSA B 4, 481
(1987); Sipe/So/Fukui/Stegeman, PRB 21, 4389 (1980)); Dadap, Shan, Eisenthal, Heinz, PRL 83, 4045
(1999).
"""

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.constants import EPS0, C_LIGHT                                       # noqa: E402
from dynameta.optics.shg_fem import shg_two_step, shg_structured_two_step          # noqa: E402

Z0 = 1.0 / (EPS0 * C_LIGHT)
LAM = 1000e-9
EPS_W = complex(-40.0, 2.5)      # Drude metal (gold-like near 1 um); ConstantOptical => same at 2w
CHI = 1.0e-20                    # representative normal surface susceptibility (m^2/V)


def _flat_gold_design(period_m, theta_deg, pol="p"):
    """A flat Drude half-space cell (vacuum cap + gold slab, gold substrate) -- the SAME coarse mesh
    as the shipped flat SHG gate (tests/test_shg_fem.test_flat_shg_fem_quantitative). Called through
    shg_structured_two_step on a FLAT geometry it must reproduce shg_two_step (gate 1)."""
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
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=theta_deg, linear_solver="umfpack")
    return Design(name="shg_flat", unit_cell=UnitCell.square(period_m), stack=stack,
                  electrodes=[], materials=reg, mesh_3d=m3, optical=opt)


def _grating_gold_design(period_m, theta_deg, h_m):
    """A shallow lamellar grating: a thick gold base + a thin (thickness h) top layer carrying a
    centered gold tooth (interior Rectangle inclusion) in an air background, so the metal/air surface
    is corrugated by depth h. h = 0 returns the flat two-layer cell. period_m is chosen so the m = +-1
    SH diffraction orders PROPAGATE while the fundamental stays 0-order (checked in the gate)."""
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Design, Inclusion, Rectangle
    from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec

    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("gold", ConstantOptical(EPS_W)))
    if h_m <= 0.0:
        layers = [Layer("metalL", 300e-9, "gold"), Layer("capL", 250e-9, "air")]
    else:
        tooth = Inclusion(shape=Rectangle(cx_m=period_m / 2, cy_m=period_m / 2,
                                          width_m=0.5 * period_m, height_m=0.7 * period_m),
                          material="gold")
        layers = [Layer("metalL", 300e-9, "gold"),
                  Layer("toothL", h_m, "air", inclusions=[tooth]),
                  Layer("capL", 250e-9, "air")]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="gold")
    m3 = Mesh3DSpec(pml_thk_m=450e-9, superstrate_buffer_m=650e-9, substrate_buffer_m=200e-9,
                    maxh_superstrate_m=95e-9, maxh_substrate_m=95e-9, maxh_metal_m=45e-9,
                    maxh_background_m=70e-9, maxh_pml_m=180e-9)
    opt = OpticalSpec(polarization="p", incidence_angle_deg=theta_deg, linear_solver="umfpack")
    return Design(name="shg_grat", unit_cell=UnitCell.square(period_m), stack=stack,
                  electrodes=[], materials=reg, mesh_3d=m3, optical=opt)


# --------------------------------------------------------------------------------------------
# GATE 1 -- FLAT LIMIT (load-bearing): reproduce shg_two_step to < 2%
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [20.0, 35.0])
def test_structured_flat_limit_matches_shg_two_step(theta_deg):
    """A geometrically FLAT 'structured' cell run through shg_structured_two_step reproduces the flat
    shg_two_step radiated power to < 2% (measured ~0.8% at 20 deg, ~1.2% at 35 deg). This validates
    the FEM trace-extraction of E_perp AND the multi-order sheet assembly against the analytic-
    sampling path: on a flat surface only the m = 0 order survives, and E0 collapses to shg_two_step's
    single analytic sheet plane wave. The extracted E_perp must also match the analytic Sipe field."""
    d = _flat_gold_design(220e-9, theta_deg)
    ref = shg_two_step(d, lambda_fund_m=LAM, chi_zzz=CHI)
    out = shg_structured_two_step(d, lambda_fund_m=LAM, chi_zzz=CHI, n_orders=2,
                                  metal_region="metalL")
    rel = abs(out["p_up_2w"] - ref["p_up_2w"]) / ref["p_up_2w"]
    assert rel < 0.02, "theta={}: structured p_up_2w={:.4e} vs shg_two_step {:.4e} (rel {:.3f})".format(
        theta_deg, out["p_up_2w"], ref["p_up_2w"], rel)
    # only the specular order is populated on a flat surface (nonspecular is extraction noise)
    assert out["p_up_specular"] > 100.0 * out["p_nonspecular"]
    # the FEM-extracted normal field matches the analytic Sipe closed form to the extraction noise
    rel_e = abs(abs(out["E_perp_in"]) - abs(ref["E_perp_in"])) / abs(ref["E_perp_in"])
    assert rel_e < 0.02, "E_perp extraction noise {:.3f} (theta={})".format(rel_e, theta_deg)


# --------------------------------------------------------------------------------------------
# GATE 2 -- SLOPE 2 (SH power quadratic in the sheet strength)
# --------------------------------------------------------------------------------------------
def test_structured_slope_two_in_chi():
    """SH radiated power is quadratic in the surface-susceptibility chi (equivalently in the
    fundamental intensity at fixed geometry): P ~ |eps0 chi E_perp^2|^2 ~ chi^2, slope 2.00 +/- 0.02.
    Runs the FULL structured pipeline (extraction -> sheet -> sourced solve -> radiated power) at three
    chi on the flat cell, validating the |sheet|^2 -> power normalization through the sourced solve."""
    d = _flat_gold_design(220e-9, 30.0)
    chis = [0.5e-20, 1.0e-20, 2.0e-20]
    ps = [shg_structured_two_step(d, lambda_fund_m=LAM, chi_zzz=c, n_orders=1,
                                  metal_region="metalL")["p_up_2w"] for c in chis]
    slope = np.polyfit(np.log(chis), np.log(ps), 1)[0]
    assert abs(slope - 2.0) < 0.02, "chi-slope {:.4f} (P={})".format(slope, ps)


# --------------------------------------------------------------------------------------------
# GATE 3 -- SHALLOW GRATING: nonspecular sheet SH ~ h^2, specular ~ flat for shallow h
# --------------------------------------------------------------------------------------------
def test_shallow_grating_perturbation_trend():
    """A shallow lamellar corrugation (metal tooth, depth h): the surface-sheet nonspecular Fourier
    amplitude |c_{+-1}| grows LINEARLY in h (perturbation theory), so the m = +-1 SHEET SH power ~ h^2
    -- gate the log-log power slope within 25% of 2.0. The m = 0 (specular) sheet stays within ~10% of
    the flat value for the shallowest h. Uses radiate=False (fundamental solve + surface-sheet
    extraction only, skipping the expensive 2w sourced solve) for the clean physical-optics trend."""
    period = 700e-9
    theta = 12.0
    flat = shg_structured_two_step(_grating_gold_design(period, theta, 0.0), lambda_fund_m=LAM,
                                   chi_zzz=CHI, n_orders=2, nx=32, ny=4, metal_region="metalL",
                                   radiate=False)
    c0_flat = abs(flat["sheet"]["c0"])
    hs = [2e-9, 4e-9, 7e-9]
    c0s, cm1s = [], []
    for h in hs:
        out = shg_structured_two_step(_grating_gold_design(period, theta, h), lambda_fund_m=LAM,
                                      chi_zzz=CHI, n_orders=2, nx=32, ny=4,
                                      metal_region="toothL__incl0", radiate=False)
        orders = out["sheet"]["orders"]
        c0s.append(abs(out["sheet"]["c0"]))
        cm1s.append(abs(orders.get(1, 0j)) + abs(orders.get(-1, 0j)))
    # specular sheet within 10% of flat for the shallowest h
    rel_spec = abs(c0s[0] - c0_flat) / c0_flat
    assert rel_spec < 0.10, "specular sheet drift {:.3f} at h={:.0f}nm".format(rel_spec, hs[0] * 1e9)
    # nonspecular amplitude grows monotonically and ~ linearly in h
    assert cm1s[0] < cm1s[1] < cm1s[2], cm1s
    ampl_slope = np.polyfit(np.log(hs), np.log(cm1s), 1)[0]
    power_slope = 2.0 * ampl_slope                       # power ~ |amplitude|^2
    assert abs(power_slope - 2.0) < 0.5, (
        "|c_+-1| amplitude slope {:.3f} -> nonspecular power slope {:.3f} (expect ~2.0 within 25%); "
        "|c_+-1|={}".format(ampl_slope, power_slope, cm1s))


def test_shallow_grating_full_pipeline_runs_and_diffracts():
    """The FULL structured pipeline (radiate=True) runs end-to-end on the grating: the 2w sourced
    solve converges and returns finite radiated SH partitioned over PROPAGATING diffraction orders
    including m = +-1 (the grating opens nonspecular SH channels absent on the flat surface). The
    magnitude of the FEM +-1 is NOT gated here -- it additionally carries linear SH re-diffraction off
    the structured gold at 500 nm (see gate 3 for the clean surface-SHG trend)."""
    out = shg_structured_two_step(_grating_gold_design(700e-9, 12.0, 8e-9), lambda_fund_m=LAM,
                                  chi_zzz=CHI, n_orders=2, nx=32, ny=4, metal_region="toothL__incl0")
    assert np.isfinite(out["p_up_2w"]) and out["p_up_2w"] > 0.0
    # the specular AND at least one first order propagate and radiate
    assert 0 in out["p_up_by_order"]
    assert any(m in out["p_up_by_order"] for m in (1, -1)), out["p_up_by_order"]
    assert all(np.isfinite(p) and p >= 0.0 for p in out["p_up_by_order"].values())


# --------------------------------------------------------------------------------------------
# GATE 4 -- s-pol selection rule (suppression vs p-pol, not exact zero on a structured surface)
# --------------------------------------------------------------------------------------------
def test_spol_suppressed_vs_ppol():
    """The a-term (chi_zzz) sheet is driven by the NORMAL fundamental field. On a flat surface an
    s-pol fundamental has no normal E, so its structured-driver SH is pure extraction noise and is
    suppressed vs p-pol by many orders of magnitude (measured s/p ~ 1e-10 in power). We gate the
    SUPPRESSION ratio rather than exact zero (a structured surface would generate an h-suppressed
    residual at tilted facets)."""
    dp = _flat_gold_design(220e-9, 30.0, pol="p")
    dy = _flat_gold_design(220e-9, 30.0, pol="y")        # s-pol (E along y)
    pp = shg_structured_two_step(dp, lambda_fund_m=LAM, chi_zzz=CHI, n_orders=1, metal_region="metalL")
    ss = shg_structured_two_step(dy, lambda_fund_m=LAM, chi_zzz=CHI, n_orders=1, metal_region="metalL")
    assert pp["p_up_2w"] > 0.0
    ratio = ss["p_up_2w"] / pp["p_up_2w"]
    assert ratio < 1e-4, "s-pol/p-pol SH ratio {:.3e} (not suppressed)".format(ratio)
    # the driving normal field itself is strongly suppressed for s-pol
    assert abs(ss["E_perp_in"]) < 1e-2 * abs(pp["E_perp_in"])
