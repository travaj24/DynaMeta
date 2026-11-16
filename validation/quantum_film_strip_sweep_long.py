"""Quantum film in the metal-strip cell: the film-model comparison sweep, 1245-1350 nm band -- gate E.

Split out of validation/quantum_film_strip.py (2026-11-14) UNCHANGED: same device, same dual-gate
ground states (same biases, same solve order), same order-4 / 100-harmonic cell per wavelength, same
three film models ((i) local graded Drude, (ii) equivalent RPA layer, (iii) exact RPA film with 40
exactly treated harmonics), same wavelength values (the 1245-1350 nm entries of the 1050-1350 nm,
5 nm grid). See that file's docstring for the device and why the sweep is split by band (unsplit,
the full sweep in one process takes ~1350 s against run_all's 2700 s per-script cap). Requires ngsolve (exit 42
without it).

Gate
  E  energy closes in every cell solve of this band: |1 - R - T - sum A| < 1e-10 (the tolerance of
     the unsplit gate; with quantum_film_strip and quantum_film_strip_sweep_mid it covers every cell
     solve of the unsplit script)

Prints the band's max |R - R_iii| per model and bias. Measured wall time: 482 s (dev box, 2
threads, lightly loaded, 2026-11-15).

Run: python -m validation.quantum_film_strip_sweep_long
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.quantum_film_strip import run_band  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_band("qfl", "long"))
