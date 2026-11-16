"""Measurement averaging on the strip-on-mirror dip, PART 2 of 4: strip-width scatter at std 2 and
5 nm -- gates D, E1 and F.

Split out of validation/measurement_strip_broadening.py (2026-11-14) with the cell, the solver
settings, the quadrature rules and the tolerances UNCHANGED; see that file's docstring for the cell,
the measurement models, the full gate list and why the script was split (unsplit it took 1977-2487 s
and timed out at run_all's 2700 s cap). Requires ngsolve (2-D TM FEM; exit 42 without it).

Cases: the plane wave and the width scatter (normal, Gauss-Hermite 9 nodes) at std 2 and 5 nm,
full collection <|r|^2> and specular |<r>|^2 from the same solves.

Gates
  D   |<r>|^2 <= <|r|^2> (diffuse >= 0) and |r_coh|^2 <= the power signal, every wavelength, over
      the spectra computed here (plane wave, std 2 nm, std 5 nm)
  E1  with full collection the FWHM and R_min grow strictly from the plane wave to std 2 nm to std
      5 nm, and the specular dip is never shallower than the full-collection one at 2 and 5 nm (the
      first two links of the original gate E; the 5 -> 10 nm link is E2 in
      measurement_strip_scatter_wide)
  F   independent check of the scatter broadening: Voigt width from the plane-wave FWHM and the dip
      sensitivity d(lambda)/dw (two extra plane-wave spectra at w -+ 5 nm), within 5 % at std 2 and
      5 nm

684 plane-wave solves (36 + 2 x 8 x 36 + 72; the centre width node is the plane-wave solve, reused).
Measured wall time: 501 s (dev box, 2 threads, lightly loaded, 2026-11-14; ~715 s at the per-solve
cost of the loaded run that timed out unsplit).

Run: python -m validation.measurement_strip_scatter
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.measurement_strip_broadening import LAMS, W, Study, dip_metrics  # noqa: E402


def main():
    s = Study("mss", "measurement averaging on the strip-on-mirror dip: width scatter 2 / 5 nm")
    h, model = s.h, s.model
    s.plane_wave()
    for sd in (2e-9, 5e-9):
        s.width_case(sd, convergence=False)
    s.gate_d()
    s.gate_e("E1", ("plane wave", "width std 2 nm, all", "width std 5 nm, all"),
             ("width std 2 nm, all", "width std 5 nm, all"))
    byl = s.byl()

    def gate_f():
        # independent prediction: dip sensitivity d(lambda)/dw by central difference of two
        # plane-wave spectra, Gaussian centre spread FWHM_G = 2.3548 std |dlambda/dw|, combined with
        # the plane-wave FWHM_L by the standard empirical Voigt-width formula (0.02 % accurate). Valid while
        # the dip shifts linearly over the spread (2 and 5 nm); 5 % tolerance for the change of depth
        # and width with w that the Voigt picture ignores.
        xa = dip_metrics(LAMS, np.abs(model(LAMS, width_m=W - 5e-9)) ** 2)[0]
        xb = dip_metrics(LAMS, np.abs(model(LAMS, width_m=W + 5e-9)) ** 2)[0]
        slope = (xb - xa) / 10.0                                          # nm per nm
        fL = byl["plane wave"][3]
        msgs, ok = [], True
        for sd in (2.0, 5.0):
            fG = 2.3548 * sd * abs(slope)
            fV = 0.5346 * fL + math.sqrt(0.2166 * fL * fL + fG * fG)
            got = byl["width std {:.0f} nm, all".format(sd)][3]
            ok = ok and abs(got / fV - 1.0) < 0.05
            msgs.append("std {:.0f} nm: FWHM {:.1f} vs Voigt {:.1f} ({:+.1f} %)".format(
                sd, got, fV, 100 * (got / fV - 1.0)))
        return ok, "dlambda/dw = {:.2f}; ".format(slope) + "; ".join(msgs)
    h.run("F", gate_f)
    return s.finish()


if __name__ == "__main__":
    raise SystemExit(main())
