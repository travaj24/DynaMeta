"""Quantum film in the metal-strip cell: the FULL-BAND film-model comparison table (opt-in REPORT,
no gate; excluded from validation/run_all -- see SKIP there).

The descriptive half of the sweep that validation/quantum_film_strip.py ran before 2026-11-14: per
film model ((i) local graded Drude, (ii) equivalent RPA layer, (iii) exact RPA film) and bias, the
dip position, R_min, FWHM (half depth below the window maximum), shift against (0, 0) V and A_film
at the dip, over 1050-1350 nm in 5 nm steps, plus the full-band max |R - R_iii|. Dip and FWHM need
all 61 wavelengths in one process, so this cannot be split by band the way the gated sweep is.

Same device, ground states, cells, models and wavelength values as quantum_film_strip (it calls the
same functions). Its only gate-like quantity, energy closure in every cell solve, is GATED in run_all
by the three band parts (quantum_film_strip, quantum_film_strip_sweep_mid, quantum_film_strip_sweep_long;
tolerance 1e-10); here it is printed for information only.

Cost: the whole 61-wavelength sweep in one process -- 1350 s on the lightly loaded 2-thread dev box
(2026-11-15) -- about half of run_all's 2700 s
per-script cap for a table that asserts nothing, which is why it is opt-in (its code path alone, with
DYNAMETA_QFS_QUICK=1: 34 s). Requires ngsolve (exit 42 without it).

Run: python -m validation.quantum_film_strip_dip_table
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.quantum_film_strip import (BIASES_T3, LAMS_T3, MODELS, _dip, print_differences,  # noqa: E402
                                           print_times, start, sweep)


def main():
    tag = "qft"
    _h, _sp, drude, films, resps = start(tag, "quantum film in a strip cell: full-band model-comparison table "
                                              "(report, no gate)")
    t_start = time.time()
    print("[{}] --- model comparison: {} biases x {} wavelengths ({:.0f}-{:.0f} nm) ---".format(
        tag, len(BIASES_T3), LAMS_T3.size, LAMS_T3[0] * 1e9, LAMS_T3[-1] * 1e9), flush=True)
    R, Af, worst_e, times = sweep(tag, LAMS_T3, films, resps, drude, t_start)
    print("[{}] (info, gated in the band parts) max |1 - R - T - sum A| over every cell solve: {:.1e}".format(
        tag, worst_e), flush=True)
    print("[{}] model | bias (Vb, Vt) | dip [nm] | R_min | FWHM [nm] | shift vs (0, 0) [nm] | A_film at the dip".format(
        tag), flush=True)
    for mo in MODELS:
        ref = _dip(LAMS_T3, R[(mo, (0.0, 0.0))])[0]
        for V in BIASES_T3:
            lmin, rmin, fw = _dip(LAMS_T3, R[(mo, V)])
            i = int(np.argmin(R[(mo, V)]))
            print("[{}] {:20s} | ({:+.0f}, {:+.0f}) | {:.1f} | {:.4f} | {:.1f} | {:+.1f} | {:.3f}".format(
                tag, mo, V[0], V[1], lmin, rmin, fw, lmin - ref, Af[(mo, V)][i]), flush=True)
    print_differences(tag, R, "over the band")
    print_times(tag, LAMS_T3, times, t_start)
    print("[{}] report complete (no gate: exit 0)".format(tag), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
