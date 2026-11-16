"""Measurement averaging on the strip-on-mirror dip, PART 4 of 4: convergence of the width-scatter
quadrature at std 10 nm -- gates C and D.

Split out of validation/measurement_strip_broadening.py (2026-11-14) with the cell, the solver
settings, the quadrature rules and the tolerances UNCHANGED; see that file's docstring for the cell,
the measurement models, the full gate list and why the script was split. Requires ngsolve (2-D TM
FEM; exit 42 without it).

Case: the width scatter at std 10 nm, exactly the unsplit script's setup -- Gauss-Hermite 11 nodes
with MeasurementSetup(convergence_nodes=13, convergence_tol=2e-3) -- plus the plane wave (the shift
reference, and the centre node of both rules).

Gates
  C  scatter quadrature converged: std 10 nm with 11 vs 13 Hermite nodes agree to 2e-3 in R at
     every wavelength (MEASURED 3.2e-4; the 10 nm spread rivals the linewidth, hence 11 nodes, not 9)
  D  |<r>|^2 <= <|r|^2> (diffuse >= 0) and |r_coh|^2 <= the power signal, every wavelength, over
     the spectra computed here (plane wave, std 10 nm)

828 plane-wave solves (36 + 10 x 36 + 12 x 36; the shared centre node is the plane-wave solve).
Measured wall time: 638 s (dev box, 2 threads, lightly loaded, 2026-11-14; ~870 s at the per-solve
cost of the loaded run that timed out unsplit -- the slowest of the four parts).

Run: python -m validation.measurement_strip_scatter_conv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.measurement_strip_broadening import Study  # noqa: E402


def main():
    s = Study("msc", "measurement averaging on the strip-on-mirror dip: 10 nm scatter quadrature")
    s.plane_wave()
    res = s.width_case(10e-9, convergence=True)
    conv_delta = res.convergence_delta
    s.h.gate("C", conv_delta is not None and conv_delta < 2e-3,
             "max |R(11 nodes) - R(13 nodes)| at std 10 nm = {:.2e}".format(
                 conv_delta if conv_delta is not None else float("nan")))
    s.gate_d()
    return s.finish()


if __name__ == "__main__":
    raise SystemExit(main())
