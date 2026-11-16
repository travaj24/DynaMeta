"""Measurement averaging on the strip-on-mirror dip, PART 3 of 4: the wide width scatter (std 10 nm)
against std 5 nm -- gates D and E2.

Split out of validation/measurement_strip_broadening.py (2026-11-14) with the cell, the solver
settings, the quadrature rules and the tolerances UNCHANGED; see that file's docstring for the cell,
the measurement models, the full gate list and why the script was split. Requires ngsolve (2-D TM
FEM; exit 42 without it).

Cases: the plane wave (the shift reference, and the centre node of both width rules), the width
scatter at std 5 nm (Gauss-Hermite 9 nodes) and at std 10 nm (11 nodes), full collection and
specular from the same solves. The 10 nm rule is the primary rule of the unsplit script; its
13-node refinement (gate C) is in measurement_strip_scatter_conv and is not needed here, because
the refinement is a separate evaluation that does not change the primary result.

Gates
  D   |<r>|^2 <= <|r|^2> (diffuse >= 0) and |r_coh|^2 <= the power signal, every wavelength, over
      the spectra computed here (plane wave, std 5 nm, std 10 nm)
  E2  with full collection the FWHM and R_min grow strictly from std 5 nm to std 10 nm, and the
      specular dip is never shallower than the full-collection one at 10 nm (the last link of the
      original gate E; E1 in measurement_strip_scatter holds the plane wave -> 2 -> 5 nm links)

684 plane-wave solves (36 + 8 x 36 + 10 x 36; the centre width node is the plane-wave solve).
Measured wall time: 499 s (dev box, 2 threads, lightly loaded, 2026-11-14; ~715 s at the per-solve
cost of the loaded run that timed out unsplit).

Run: python -m validation.measurement_strip_scatter_wide
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.measurement_strip_broadening import Study  # noqa: E402


def main():
    s = Study("msw", "measurement averaging on the strip-on-mirror dip: width scatter 5 -> 10 nm")
    s.plane_wave()
    for sd in (5e-9, 10e-9):
        s.width_case(sd, convergence=False)
    s.gate_d()
    s.gate_e("E2", ("width std 5 nm, all", "width std 10 nm, all"), ("width std 10 nm, all",))
    return s.finish()


if __name__ == "__main__":
    raise SystemExit(main())
