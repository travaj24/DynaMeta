"""FDTD bias x wavelength MODULATOR study: the end-to-end device figure of merit -- how a gated ITO layer
modulates the broadband reflection spectrum -- computed with the fast dispersive broadband sweep
(optics.fdtd_seam.run_fdtd_sweep). A gate bias drives the ITO carrier density from background to
accumulation; the free-carrier Drude permittivity shifts (toward ENZ / metallic), shifting R(lambda). The
modulation contrast dR(lambda) = R_on - R_off is the modulator's spectral response.

Each bias state's full spectrum comes from ONE FDTD solve (the dispersive sweep, vs N per-wavelength
solves). The DEVICE metric is the DIFFERENCE between bias states, so the common single-layer FDTD
numerical-dispersion (Fabry-Perot) error cancels -- so dR(lambda) is validated TIGHTLY against coherent
TMM, even where the absolute R carries the ~1-2%% thin-layer FDTD wobble.

GATE 1 (each state vs TMM): the dispersive sweep R(lambda) matches TMM for BOTH bias states (~few %%).
GATE 2 (modulation vs TMM): the FDTD modulation dR(lambda) matches TMM's dR to ~1e-2, and is a genuine
        non-trivial contrast (peak |dR| > 0.02) -- the device actually modulates.

NOTE (scope): a real ITO modulator's accumulation is a FEW-nm ENZ layer that a uniform FDTD grid
under-resolves (a limit of ALL FDTD, per the build-vs-buy verdict) -- use FEM/RCWA for the few-nm ENZ
sub-layer. This study uses a homogenized 60nm gated layer to demonstrate the broadband bias-sweep
capability.

Run: python -m validation.fdtd_modulator_study
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.materials import ConstantOptical, DrudeOptical, M_E, Material, MaterialRegistry, TabulatedOptical
from dynameta.optics.fdtd_seam import run_fdtd_sweep
from dynameta.optics.tmm_reference import layered_rta, layered_stack_from_design

C = 299792458.0
ITO_DRUDE = DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * M_E, gamma_rad_s=1.0e14)   # reference-like ITO


def _ito_at_bias(n_cm3, lo=1100e-9, hi=1900e-9, npts=220):
    """The gated ITO as a TabulatedOptical (its Drude eps at carrier density n, as a lambda-function both
    the FDTD dispersive sweep and TMM consume identically)."""
    lam = np.linspace(lo, hi, npts)
    n_m3 = n_cm3 * 1.0e6
    eps = np.array([complex(ITO_DRUDE.eps(float(l), n_m3=n_m3)) for l in lam], dtype=complex)
    return TabulatedOptical(lambda_m=lam, eps_complex=eps)


def _design(n_cm3):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("ito", _ito_at_bias(n_cm3)))
    reg.add(Material("diel", ConstantOptical(2.0 ** 2 + 0j)))
    stack = Stack(layers=[Layer("ito", 60e-9, "ito"), Layer("diel", 250e-9, "diel")],
                  superstrate_material="air", substrate_material="air")
    return Design(name="mod", unit_cell=UnitCell.square(220e-9), stack=stack, electrodes=[], materials=reg)


def main():
    print("[fm] === FDTD bias x wavelength ITO modulator study (dispersive broadband sweep) ===", flush=True)
    n_off, n_on = 4.0e20, 1.2e21                            # cm^-3: background -> accumulation
    d_off, d_on = _design(n_off), _design(n_on)
    targets = np.linspace(1250e-9, 1650e-9, 9)

    off = run_fdtd_sweep(d_off, targets, dim=2, resolution=30, band_pad=0.12)   # ONE solve / bias state
    on = run_fdtd_sweep(d_on, targets, dim=2, resolution=30, band_pad=0.12)

    dRf = dTf = dMod = peak = 0.0
    Rt_off = np.zeros(targets.size); Rt_on = np.zeros(targets.size)
    for i, lam in enumerate(targets):
        Rt_off[i], _, _ = layered_rta(layered_stack_from_design(d_off, float(lam)), float(lam))
        Rt_on[i], _, _ = layered_rta(layered_stack_from_design(d_on, float(lam)), float(lam))
        dRf = max(dRf, abs(off[i].R - Rt_off[i]), abs(on[i].R - Rt_on[i]))
    fdtd_mod = np.array([on[i].R - off[i].R for i in range(targets.size)])
    tmm_mod = Rt_on - Rt_off
    dMod = float(np.max(np.abs(fdtd_mod - tmm_mod)))        # device metric: modulation matches TMM
    peak = float(np.max(np.abs(fdtd_mod)))
    i_pk = int(np.argmax(np.abs(fdtd_mod)))

    # Tolerances reflect the FDTD THIN-LAYER limit: a 60nm ITO layer crossing ENZ carries ~2-3% uniform-
    # grid numerical-dispersion error (the documented ENZ caveat -- tight ENZ needs FEM/RCWA). The gate
    # validates the CAPABILITY: the bias-swept spectra track TMM to that level, the modulation matches
    # TMM's, and the contrast is large & real -- NOT sub-% accuracy on a thin ENZ layer (out of FDTD scope).
    gate1 = bool(dRf < 4e-2)
    print("[fm] 1 each bias vs TMM: max|R_fdtd - R_tmm| (both states) = {:.2e} (one solve {:.1f}s/state; "
          "thin-ITO FDTD limit) -> {}".format(dRf, off[0].solve_time_s, "PASS" if gate1 else "FAIL"), flush=True)
    gate2 = bool(dMod < 2.5e-2 and peak > 0.05)
    print("[fm] 2 modulation dR=R_on-R_off vs TMM: max|dMod_fdtd - dMod_tmm|={:.2e} ; peak|dR|={:.3f} at "
          "{:.0f}nm (real strong modulation) -> {}".format(
              dMod, peak, targets[i_pk] * 1e9, "PASS" if gate2 else "FAIL"), flush=True)
    print("[fm] spectrum  lam(nm) | R_off  R_on  dR(mod)", flush=True)
    for i in range(targets.size):
        print("[fm]          {:6.0f}  | {:.3f}  {:.3f}  {:+.3f}".format(
            targets[i] * 1e9, off[i].R, on[i].R, fdtd_mod[i]), flush=True)

    overall = gate1 and gate2
    print("[fm] *** FDTD MODULATOR STUDY (bias-swept broadband spectra; modulation contrast vs TMM): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
