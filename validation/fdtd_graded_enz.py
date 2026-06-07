"""ENZ GRADED-eps coupling: make the gated-ITO optics QUANTITATIVE by feeding the carrier-density DEPTH
profile n(z) (the accumulation layer) into a GRADED eps(z) the FDTD resolves, instead of a single
homogenized eps. A real accumulation is a steep near-interface profile that crosses ENZ; a single eps
misses it. optics.fdtd_seam.eps_profile_from_carrier (n(z) -> eps(z) via the ITO Drude) +
graded_fdtd_layers (slice eps(z) into thin FDTD sublayers via the one-Drude inversion) close that gap.

GATE 1 (graded machinery vs TMM): a graded ITO accumulation profile -> the FDTD (sliced) R == coherent
        TMM on the SAME sliced profile -- the depth grading is carried correctly.
GATE 2 (the profile matters): the graded R differs CLEARLY from a single homogenized (depth-averaged)
        eps -- i.e. the near-interface ENZ shaping changes the optics, so the graded coupling is not
        cosmetic.

SCOPE: validated on a resolvable (thick) graded layer + an absorbing profile (Fabry-Perot-suppressed).
A REAL few-nm ITO accumulation needs a correspondingly fine FDTD dz (the documented ENZ caveat); this
validates the graded-eps MACHINERY that, at fine enough resolution, makes the ITO modulator quantitative.

Run: python -m validation.fdtd_graded_enz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.materials import DrudeOptical, M_E
from dynameta.optics.fdtd_seam import _eps_to_fdtd_layer, eps_profile_from_carrier, graded_fdtd_layers
from dynameta.optics.fdtd_nd import solve_fdtd_2d
from dynameta.optics.tmm_reference import layered_rta

C = 299792458.0
LAM = 1500e-9
ITO_DRUDE = DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * M_E, gamma_rad_s=1.0e14)
ITO_D = 400e-9          # thick enough that the sublayers are FDTD-resolvable (the few-nm real layer = ENZ caveat)
NSLICE = 10


def _accumulation_eps(n_bg_cm3=4.0e20, dn_cm3=3.5e20, lam_D=100e-9, npts=80):
    """Graded eps(z) of an accumulation layer n(z) = n_bg + dn*exp(-z/lam_D) (front = gate interface)."""
    z = np.linspace(0.0, ITO_D, npts)
    n_cm3 = n_bg_cm3 + dn_cm3 * np.exp(-z / lam_D)
    eps_full = eps_profile_from_carrier(n_cm3 * 1e6, LAM, ITO_DRUDE)
    u = np.linspace(0.0, npts - 1.0, NSLICE)                # resample to NSLICE sublayers
    idx = np.arange(npts, dtype=float)
    return np.interp(u, idx, eps_full.real) + 1j * np.interp(u, idx, eps_full.imag)


def _fdtd_R(layers, resolution=36):
    r = solve_fdtd_2d(layers, period_x_m=300e-9, lambda_min_m=LAM * 0.9, lambda_max_m=LAM * 1.1,
                      resolution=resolution, backend="auto")
    # interpolate to EXACTLY c/lambda: each inverted-Drude sublayer hits its target eps only at c/lambda,
    # so a nearest-FFT-bin read biases R (the Move-1 lesson); freqs are increasing.
    return float(np.interp(C / LAM, r.freqs_Hz, r.R0))


def main():
    print("[ge] === ENZ graded-eps coupling: graded ITO accumulation vs TMM, and vs homogenized ===", flush=True)
    eps_s = _accumulation_eps()
    d_sub = ITO_D / NSLICE
    print("[ge] graded ITO: {} slices over {:.0f}nm ; eps front={:.3f} -> back={:.3f} (ENZ-crossing)".format(
        NSLICE, ITO_D * 1e9, eps_s[0], eps_s[-1]), flush=True)

    # FDTD with the graded (sliced) profile vs TMM on the SAME slices
    R_fdtd = _fdtd_R(graded_fdtd_layers(ITO_D, eps_s, LAM))
    stack = LayeredStack(1.0 + 0j, 1.0 + 0j, [LayeredSlab(d_sub, eps=complex(e)) for e in eps_s])
    R_tmm, _, _ = layered_rta(stack, LAM)
    dGT = abs(R_fdtd - R_tmm)
    gate1 = bool(dGT < 2e-2)                                 # graded machinery == TMM to the FDTD ENZ limit
    print("[ge] 1 graded FDTD vs TMM: R_fdtd={:.4f} R_tmm={:.4f} |d|={:.2e} -> {}".format(
        R_fdtd, R_tmm, dGT, "PASS" if gate1 else "FAIL"), flush=True)

    # homogenized (depth-averaged) single eps -- should give a CLEARLY different R
    eps_h = complex(np.mean(eps_s))
    R_homog = _fdtd_R([_eps_to_fdtd_layer(ITO_D, eps_h, LAM)])
    dGH = abs(R_fdtd - R_homog)
    gate2 = bool(dGH > 0.02)
    print("[ge] 2 profile matters: R_graded={:.4f} vs R_homogenized={:.4f} |d|={:.3f} (>0.02) -> {}".format(
        R_fdtd, R_homog, dGH, "PASS" if gate2 else "FAIL"), flush=True)

    overall = gate1 and gate2
    print("[ge] *** ENZ GRADED-eps COUPLING (carrier n(z) -> graded eps(z); FDTD==TMM; profile matters): "
          "{} ***".format("PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
