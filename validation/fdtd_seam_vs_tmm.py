"""FDTD OpticalSolver seam vs the TMM oracle. optics.fdtd_seam.make_fdtd_optical_solver wraps the 2D/3D
FDTD as the same pluggable `optical_solver` run_pipeline drives (core.interfaces.OpticalSolver). On a
laterally-uniform stack the FDTD must reproduce the exact coherent-TMM R/T/A -- the de-risking gate that
the seam is wired correctly (Design -> FDTD layers -> R/T/A/phase), mirroring graded_tmm_vs_fem.py.

GATE 1 (lossless dielectric stack): a 3-layer dielectric in air -- the seam's 0-order R/T == TMM to the
        FDTD discretization, and R+T = 1.
GATE 2 (lossy slab -> the Drude-inversion path): a single absorbing slab (complex eps) -- the seam (which
        inverts a Drude pole to hit eps(lambda) exactly) reproduces the TMM R/T/A; an absorbing layer
        cannot fake the split, so this checks both the loss handling and the energy budget A = 1-R-T.
GATE 3 (phase): the de-embedded front-face reflection phase agrees with TMM. 3a: the two shipped
        geometries above, per-geometry (sub-cell staircase limited, +-k0*dz/2 per interface).
        3b (2026-11-02, audit S2-14 siblings): the REGISTRATION bias itself -- the mean phase error
        over 8 midpoint-uniform sub-cell pad offsets (the staircase sawtooth cancels exactly, the
        mean is the offset of the structure against the probe frame). It used to be absorbed by a
        15-deg tolerance ("the half-cell limit") that hid a systematic -k0*dz bias: every sibling of
        fdtd_mo registered materials at cell centres (measured here before the fix: see the bars).

Run: python -m validation.fdtd_seam_vs_tmm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.optics.fdtd_seam import make_fdtd_optical_solver
from dynameta.optics.tmm_reference import (TmmLayeredSolver, end_media_indices,
                                           layered_rta, layered_stack_from_design)

LAM = 1300e-9


def _design(layer_specs, name):
    """layer_specs: list of (eps_complex, thickness_m), in Stack (bottom->top) order; air super/substrate."""
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    layers = []
    for k, (eps, th) in enumerate(layer_specs):
        reg.add(Material("m%d" % k, ConstantOptical(complex(eps))))
        layers.append(Layer("s%d" % k, float(th), "m%d" % k))
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    return Design(name=name, unit_cell=UnitCell.square(220e-9), stack=stack, electrodes=[], materials=reg)


def _seam_vs_tmm(d, solver):
    nsup, nsub = end_media_indices(d, LAM)
    res = solver(d, None, {}, LAM, nsup, nsub)               # the FDTD seam (OpticalSolver call)
    stack = layered_stack_from_design(d, LAM)
    R_t, T_t, A_t = layered_rta(stack, LAM)                  # TMM oracle (R, T, A)
    tmm = TmmLayeredSolver().solve(stack, LAM, OpticalSpec(polarization="y", incidence_angle_deg=0.0))
    T_f = res.T if res.T is not None else 0.0
    dR, dT = abs(res.R - R_t), abs(T_f - T_t)
    dphase = abs(((res.phase_deg - tmm.phase_deg + 180.0) % 360.0) - 180.0)
    return res, (R_t, T_t, A_t), dR, dT, dphase


def _grid_dz(d, res, band_frac):
    """The seam's own dz for design d at LAM (solve_fdtd_2d's z_grid on the seam's layer list)."""
    from dynameta.constants import C_LIGHT
    from dynameta.optics.fdtd_nd.grid import layer_band_index_max
    from dynameta.optics.fdtd_seam import design_to_fdtd_layers
    lo, hi = LAM * (1.0 - band_frac), LAM * (1.0 + band_frac)
    layers = design_to_fdtd_layers(d, LAM)
    w_band = 2.0 * np.pi * np.linspace(C_LIGHT / hi, C_LIGHT / lo, 9)
    n_max = layer_band_index_max(layers, w_band, (1.0, 1.0))
    return lo / (res * n_max), hi


def _k0dz_deg(d, res, band_frac):
    dz, _hi = _grid_dz(d, res, band_frac)
    return float(np.degrees(2.0 * np.pi / LAM * dz))


def _offset_mean_dphase(d, res, band_frac, n_off=8, base=4.0):
    """Mean SIGNED dphase [deg] over n_off midpoint-uniform sub-cell offsets of the front face."""
    dz, hi = _grid_dz(d, res, band_frac)
    m = round(base * hi / dz)
    out = []
    for k in range(n_off):
        npw = (m + (k + 0.5) / n_off) * dz / hi           # pad = npw * lambda_max -> pad/dz = m + u_k
        solver = make_fdtd_optical_solver(dim=2, resolution=res, band_frac=band_frac, n_pad_wave=npw)
        res_k, _rta, _dR, _dT, _abs = _seam_vs_tmm(d, solver)
        stack = layered_stack_from_design(d, LAM)
        tmm = TmmLayeredSolver().solve(stack, LAM, OpticalSpec(polarization="y", incidence_angle_deg=0.0))
        out.append(((res_k.phase_deg - tmm.phase_deg + 180.0) % 360.0) - 180.0)
    return float(np.mean(out))


def main():
    print("[fs] === FDTD OpticalSolver seam vs TMM oracle (laterally-uniform stacks) ===", flush=True)
    solver = make_fdtd_optical_solver(dim=2, resolution=28, band_frac=0.10)

    # GATE 1: lossless 3-layer dielectric stack in air
    d1 = _design([(2.2 ** 2, 250e-9), (1.5 ** 2, 180e-9), (2.2 ** 2, 250e-9)], "diel")
    r1, (R1, T1, A1), dR1, dT1, dph1 = _seam_vs_tmm(d1, solver)
    en1 = abs(r1.R + (r1.T or 0.0) - 1.0)
    gate1 = bool(dR1 < 6e-3 and dT1 < 6e-3 and en1 < 6e-3)
    print("[fs] 1 dielectric: seam R={:.4f} T={:.4f} | TMM R={:.4f} T={:.4f} | |dR|={:.2e} |dT|={:.2e} "
          "R+T-1={:.2e} -> {}".format(r1.R, r1.T or 0.0, R1, T1, dR1, dT1, en1, "PASS" if gate1 else "FAIL"),
          flush=True)

    # GATE 2: lossy -> the Drude-inversion path, on a THICK strongly-absorbing slab. (Why thick: the
    # back reflection is absorbed before returning, so R is the front-Fresnel value = Fabry-Perot-
    # INSENSITIVE; a thin resonant slab's R wobbles ~1-2% with the FDTD's numerical dispersion shifting
    # the FP fringe -- a GENERAL single-slab FDTD effect, identical for a LOSSLESS slab, not a loss bug.)
    # The seam inverts one Drude pole to hit eps(lambda) exactly; A ~ 0.8 cannot be faked (lossless trap).
    d2 = _design([(3.24 + 1.0j, 700e-9)], "absorber")
    r2, (R2, T2, A2), dR2, dT2, dph2 = _seam_vs_tmm(d2, solver)
    dA2 = abs((r2.A or 0.0) - A2)
    gate2 = bool(dR2 < 8e-3 and dT2 < 8e-3 and dA2 < 8e-3 and A2 > 0.5)
    print("[fs] 2 absorber (Drude inversion): seam R={:.4f} T={:.4f} A={:.4f} | TMM R={:.4f} T={:.4f} "
          "A={:.4f} | |dR|={:.2e} |dT|={:.2e} |dA|={:.2e} -> {}".format(
              r2.R, r2.T or 0.0, r2.A or 0.0, R2, T2, A2, dR2, dT2, dA2, "PASS" if gate2 else "FAIL"),
          flush=True)

    # GATE 3a: reflection phase (de-embedded to the front face) vs TMM on the two shipped geometries.
    # audit S2-14 siblings (2026-11-02): this was `< 15 deg` ("the half-cell limit"), which hid the
    # -k0*dz registration bias of every non-MO engine (k0*dz = 5.26 / 6.24 deg on these two stacks).
    # A SINGLE geometry cannot see that bias cleanly: it also carries its sub-cell staircase (each
    # interface within +-dz/2, a +-k0*dz/2-scale phase per face), which can add to or cancel it.
    # MEASURED 2026-11-02 (dev box): before the fix dielectric 6.15 deg, lossy 4.61 deg; after 5.00 /
    # 4.61 deg (the absorber's two faces both have frac(z/dz) < 1/2, where the node and cell-centre
    # rules pick the SAME cells -- bit-identical result). So 3a is the per-geometry SANITY bar
    # (sign / exp(-iwt) convention / de-embedding): 1.5 k0*dz, tighter than the old 15 deg (~3 k0dz)
    # but not a registration gate. The registration gate is 3b.
    k0dz1, k0dz2 = _k0dz_deg(d1, 28, 0.10), _k0dz_deg(d2, 28, 0.10)
    gate3a = bool(dph1 < 1.5 * k0dz1 and dph2 < 1.5 * k0dz2)
    print("[fs] 3a phase: |dphase| dielectric={:.2f}deg lossy={:.2f}deg (front-face ref; k0dz {:.2f}/{:.2f}"
          " deg) -> {}".format(dph1, dph2, k0dz1, k0dz2, "PASS" if gate3a else "FAIL"), flush=True)

    # GATE 3b: the registration bias, isolated -- mean signed dphase over 8 MIDPOINT-uniform sub-cell
    # pad offsets (pad/dz = m + (k+1/2)/8), dielectric stack. The node registration leaves only the
    # O((k dz)^2) Yee dispersion; the old cell-centre rule sat at ~ -k0*dz. MEASURED 2026-11-02:
    # before -5.195 deg (-0.99 k0dz) / after +0.066 deg (+0.013 k0dz) -- see
    # docs/audit/2026-11-02-fdtd-registration.md. Bar 0.25 k0*dz = 1.315 deg (fails the old code 4x).
    mean3b = _offset_mean_dphase(d1, 28, 0.10)
    gate3b = bool(abs(mean3b) < 0.25 * k0dz1)
    print("[fs] 3b registration bias: mean dphase over 8 sub-cell offsets = {:+.3f} deg (bar 0.25 k0dz = "
          "{:.3f} deg; pre-fix ~ -k0dz) -> {}".format(mean3b, 0.25 * k0dz1, "PASS" if gate3b else "FAIL"),
          flush=True)
    gate3 = gate3a and gate3b

    overall = gate1 and gate2 and gate3
    print("[fs] *** FDTD SEAM vs TMM (dielectric R/T; lossy R/T/A via Drude inversion; phase): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
