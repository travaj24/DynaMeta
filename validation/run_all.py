"""Run every PASS/FAIL validation as a subprocess and gate on its EXIT CODE (audit
cross-cutting F1/F3). Each validation now `raise SystemExit(0 if ok else 1)`, so this is a
single machine-checkable command for the solver-backed physics that the fast CI suite
(`pytest tests/`, which covers the solver-free bridge spine + Schrodinger + data model)
cannot run -- the heavy NGSolve/DEVSIM stack is not installed in CI.

  python -m validation.run_all                 # run all gated validations
  python -m validation.run_all oblique sp      # only scripts whose name matches a token

Exits non-zero if ANY selected validation fails or errors. Pure-diagnostic scripts (no
PASS/FAIL verdict) are skipped. Intended for a local run or a nightly job with the
[solvers] extra installed; budget tens of minutes for the full set.
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# pure diagnostics (no PASS/FAIL gate) + this runner
SKIP = {"run_all", "oblique_field_dump", "oblique_phase_diag", "oblique_sign_pml_diag",
        "park_spectrum"}
PER_SCRIPT_TIMEOUT_S = 1800


def _gated_scripts():
    out = []
    for fn in sorted(os.listdir(HERE)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        name = fn[:-3]
        if name in SKIP:
            continue
        src = open(os.path.join(HERE, fn), encoding="utf-8").read()
        if re.search(r"SystemExit|sys\.exit", src):     # only exit-code-gated validations
            out.append(name)
    return out


def main(argv):
    tokens = argv[1:]
    scripts = [s for s in _gated_scripts() if not tokens or any(t in s for t in tokens)]
    print("[run_all] {} gated validations to run\n".format(len(scripts)), flush=True)
    results = []
    for name in scripts:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-u", "-m", "validation." + name],
                                cwd=os.path.dirname(HERE), timeout=PER_SCRIPT_TIMEOUT_S)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        dt = time.time() - t0
        tag = "PASS" if rc == 0 else ("TIMEOUT" if rc == 124 else "FAIL(rc={})".format(rc))
        results.append((name, rc, dt))
        print("[run_all] {:40s} {:8s} ({:5.0f}s)".format(name, tag, dt), flush=True)
    failed = [r for r in results if r[1] != 0]
    print("\n[run_all] {}/{} passed; {} failed/errored".format(
        len(results) - len(failed), len(results), len(failed)), flush=True)
    if failed:
        print("[run_all] FAILURES: " + ", ".join(n for n, _, _ in failed), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
