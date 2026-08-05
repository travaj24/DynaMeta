"""Shared available-RAM capability guard for the heavy 3-D FEM validation oracles.

WHY THIS EXISTS (nightly runs 30348065879 / 30441185460 / 30898049092, 2026-07-28..08-04):
the heavy FEM oracles were measured on the dev workstation (128 GB) and mesh multi-GB 3-D
vectorial cells. On a 16 GB GitHub-hosted runner they do not fail -- they get the runner's
OOM "shutdown signal", which CANCELS the job and every step queued behind it: three nightlies
in a row died ~2.5 min into graded_tmm_vs_fem's mesh and the full validation tier (c2) never
executed at all. A step that cannot fit costs the coverage queued after it.

THE CONTRACT: each heavy oracle calls require_available_ram_gb(<threshold>) FIRST. If the
machine's AVAILABLE physical memory is below the threshold, the script prints the reason and
exits 42 -- the run_all capability-skip code, which is INFORMATIONAL outside the smoke tier
(these oracles are not in the smoke set), so the nightly reports a visible skip and KEEPS
GOING. On any capable machine the guard is a no-op. Thresholds are MEASURED peak process-tree
working sets (see each call site) x ~1.3 headroom, not guesses.

If psutil is unavailable the guard is inert (the oracle just runs): psutil is a [dev] extra,
installed on every CI leg that runs validation; a bare local install without it simply keeps
the old behavior. `_`-prefixed file: excluded from every run_all tier by convention.
"""
import sys


def require_available_ram_gb(needed_gb: float, script: str, measured_note: str) -> None:
    """Exit 42 (capability skip) when available RAM < needed_gb; no-op without psutil."""
    try:
        import psutil
    except ImportError:
        return
    avail_gb = psutil.virtual_memory().available / 2 ** 30
    if avail_gb >= needed_gb:
        return
    print("[{}] CAPABILITY SKIP: needs ~{:.0f} GB available RAM ({}); this machine has "
          "{:.1f} GB available. On a 16 GB hosted runner this oracle previously drew the "
          "OOM shutdown signal and cancelled the whole nightly job behind it -- skipping "
          "keeps the rest of the tier running. Run it on the dev workstation.".format(
              script, needed_gb, measured_note, avail_gb), flush=True)
    sys.exit(42)
