"""Shared gate harness for the DynaMeta validation suite (structural pass, 2026-08-31).

WHY THIS EXISTS. Every validation script hand-rolls the same three things:

    ok = True
    ...
    g_a = bool(measured < tol)
    ok = ok and g_a                                   # (1) fold the gate into the verdict
    print("[xx] GATE A: ... -> {}".format("PASS" if g_a else "FAIL"), flush=True)
    ...
    raise SystemExit(0 if main() else 1)              # (2) map the verdict to an exit code

That repetition has three failure modes, and the 2026-08-30 nightly -- the first run that ever
executed the full validation tier on CI -- hit ALL THREE in one night, across five scripts:

  * A GATE THAT RAISES IS NOT A FAILING GATE, IT IS A CRASH. All five failures were an exception
    escaping main(): a library guard raising NotImplementedError, a missing optional package, a
    backend resolver raising RuntimeError. run_all reports rc=1, which in the summary is
    indistinguishable from a physics defect -- so triage started from the wrong hypothesis every
    time. Harness.run() turns any exception into a reported FAIL carrying the exception text,
    which is the honest reading: the gate did not pass, and you can see why without opening the
    traceback. The gates AFTER it still run, so one broken gate stops hiding the rest.
  * FORGETTING THE 'ok = ok and g_x' LINE SILENTLY DROPS A GATE from the verdict. Nothing catches
    it -- the script still prints PASS/FAIL for that gate and still exits 0. Here the harness
    owns the verdict, so a recorded gate is a counted gate.
  * "NOT EXERCISABLE HERE" HAD NO SPELLING. numba absent, grcwa absent, no CUDA GPU: each script
    invented its own handling, and the ones that invented none crashed. skip() and require()
    make the capability-absent case first class and route it to run_all's exit code 42, which is
    counted separately and never green-washed into PASS (audit T-3).

MODELLED ON lumenairy's validation/_harness.py -- same run/check/summary shape, deliberately, so
the two suites read alike -- with one addition DynaMeta needs and lumenairy does not: lumenairy's
runner has a pass/fail contract, while DynaMeta's run_all has 0 PASS / 1 FAIL / 42 SKIP /
124 TIMEOUT / 125 OVER-BUDGET, so SKIP has to be a real outcome rather than a print.

OPT-IN AND ADDITIVE. The ~200 existing scripts are untouched and keep working exactly as they
are; this is for new gates and for scripts already being edited. The printed format is
deliberately byte-compatible with the existing house style, so a converted script looks the same
in the nightly log and no downstream grep has to change.

    H = Harness("n3", "3D chi2/Raman/gain FDTD")
    H.gate("A", dA < 1e-12, "3D exit probe == 2D, rel {:.1e}".format(dA))
    H.run("B", lambda: (measured < tol, "detail {:.2e}".format(measured)))
    H.skip("C", "numba absent; the numba CI job covers it")
    raise SystemExit(H.summary())

Run: python -m validation._harness      (self-check; gate C must FAIL, not crash)
"""
import sys
import traceback

SKIP_RC = 42                      # run_all's capability-absent code (audit C6-6 / T-3)


class Harness:
    """Collects gate outcomes, prints them in the house format, owns the exit code.

    Never aborts on the first failure: a validation script's value is the WHOLE picture, and a
    run that stops at gate A tells you nothing about gates C through G."""

    def __init__(self, tag, title=None):
        self.tag = tag
        self.title = title
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []
        if title:
            print("[{}] === {} ===".format(tag, title), flush=True)

    # ---- recording -------------------------------------------------------------------------
    def _record(self, label, state, detail):
        self.results.append((label, state, detail))
        print("[{}] GATE {}{} -> {}".format(self.tag, label,
                                            ": " + detail if detail else "", state), flush=True)

    def gate(self, label, ok, detail=""):
        """Record an inline boolean gate -- the direct replacement for the hand-rolled
        assign / and-into-ok / print triple."""
        ok = bool(ok)
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self._record(label, "PASS" if ok else "FAIL", detail)
        return ok

    def run(self, label, fn, detail=""):
        """Run a thunked gate. fn returns bool or (bool, detail). AN EXCEPTION IS A FAIL, not a
        crash -- the whole reason this harness exists (see the module docstring)."""
        try:
            result = fn()
            if isinstance(result, tuple):
                ok, got = bool(result[0]), str(result[1])
            else:
                ok, got = bool(result), detail
        except Exception as exc:                      # noqa: BLE001 -- report it, never mask it
            tb = traceback.format_exc(limit=2).strip().splitlines()
            self.failed += 1
            self._record(label, "FAIL", "EXCEPTION {}: {} | {}".format(
                type(exc).__name__, str(exc)[:200].replace("\n", " "),
                tb[-1][:160] if tb else ""))
            return False
        return self.gate(label, ok, got)

    def skip(self, label, reason):
        """Record a gate that CANNOT be exercised here (capability absent). Counted separately:
        never a PASS (it proved nothing) and never a FAIL (nothing is broken). If EVERY gate in
        the script skips, summary() returns 42, so run_all reports a capability skip rather than
        a green run that executed nothing -- the audit T-3 failure mode."""
        self.skipped += 1
        self._record(label, "SKIP", reason)

    def require(self, ok, reason):
        """Hard capability precondition, checked before any gate runs (a missing optional import,
        no GPU). Exits 42 immediately with the reason -- the spelling the three grcwa importers
        and the CUDA oracle each had to invent separately."""
        if not ok:
            print("[{}] SKIP: {} (exit 42; run_all counts it separately, audit C6-6)".format(
                self.tag, reason), flush=True)
            raise SystemExit(SKIP_RC)

    # ---- verdict ---------------------------------------------------------------------------
    def summary(self, name=None):
        """Print the house banner and return the exit code run_all expects: 1 if anything
        failed, 42 if nothing ran but something skipped, else 0."""
        total = self.passed + self.failed + self.skipped
        if self.failed:
            state = "FAIL"
        elif self.passed == 0 and self.skipped:
            state = "SKIP"
        else:
            state = "PASS"
        print("[{}] *** {}: {} ({}/{} passed{}) ***".format(
            self.tag, name or self.title or self.tag, state, self.passed, total,
            ", {} skipped".format(self.skipped) if self.skipped else ""), flush=True)
        if self.failed:
            return 1
        if self.passed == 0 and self.skipped:
            return SKIP_RC
        return 0


if __name__ == "__main__":                            # self-check: gate C must FAIL, not crash
    H = Harness("hz", "harness self-check")
    H.gate("A", True, "an inline pass")
    H.run("B", lambda: (True, "a thunked pass"))
    H.run("C", lambda: 1 / 0)
    H.skip("D", "a capability that is absent here")
    rc = H.summary()
    print("self-check exit code:", rc, "(expected 1 -- gate C raised and was recorded FAIL)")
    sys.exit(0 if rc == 1 else 1)
