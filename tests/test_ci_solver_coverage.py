"""Drift gate: every solver-gated test file runs on a CI leg that installs the solver stack.

WHY (v0.14.0 release blocker, 2026-11-04). The four `test` legs, `floor`, `smoke` and `numba` install
neither devsim nor ngsolve/netgen/gmsh, so a test file that importorskips them SKIPS there -- green,
and testing nothing. Only the jobs that `pip install` the `[solvers]` extra can run it, and those name
their test files EXPLICITLY (the whole suite does not fit their caps). An unlisted file therefore
silently runs nowhere: the audit-T-5 failure mode, which recurred for v0.14.0 -- twelve devsim test
files written for the release (wave-7 carriers, bipolar DG, recombination/BGN/ionization, 3-D mesh,
heterojunction, noise, phase composition, tunneling, extended precision, traps, trap energy
reference) were on no solvers leg, so not one of their gates had ever run on CI. Same class as
tests/test_validation_runner.py's shard-matrix gate: nothing turns red, coverage just shrinks.

WHAT COUNTS AS SOLVER-GATED: a module that importorskips / find_specs devsim, ngsolve, netgen, gmsh or
pyngcore, or reaches DEVSIM through dynameta.carriers.devsim_env.import_devsim, plus _INDIRECT.
MEASURED 2026-11-04 (Windows, py3.14): the test files with ANY skip mechanism (73; a file with none
cannot skip) were run `-m "not slow"` with those packages hidden by a sitecustomize setting
sys.modules[pkg] = None (so child processes too) -- all but test_shg_grating.py (DEV_ONLY) and
test_transient_dt_floor.py (below). Exactly 37 files reported a skip naming one of the packages: the
36 this rule matches plus test_polarization_vocabulary.py, now in _INDIRECT; the rule matched no file
that skipped nothing. (The one failure, test_v13_plain_import_is_solver_free..., is the hiding
itself: a None entry in sys.modules reads as "imported".) tests/test_transient_dt_floor.py is NOT
solver-gated: it stands a devsim shim in for the import and runs on every leg (under the hiding, the
real devsim then loads in later modules of the same worker, so it was left out of that run). A new skip MECHANISM this rule cannot see (e.g.
importorskip of a dynameta module that imports a solver at load) belongs in _INDIRECT, with the reason.

THE RULE: each such file is named in a PR-reachable job that installs `[solvers]` (`solvers`,
`solvers-devsim`), or is a documented nightly-only module (NIGHTLY_ONLY: too slow for the PR tier,
named in the `nightly` job), or is documented dev-workstation-only (DEV_ONLY: does not fit any hosted
runner; named in ci.yml's top-of-file not-covered list).

Run: python -m pytest tests/test_ci_solver_coverage.py -q
"""
import glob
import io
import os
import re
import tokenize

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CI_YML = os.path.join(REPO, ".github", "workflows", "ci.yml")

_SOLVER_PAT = re.compile(
    r"""importorskip\(\s*["'](?:devsim|ngsolve|netgen|gmsh|pyngcore)(?:\.[\w.]+)?["']"""
    r"""|find_spec\(\s*["'](?:devsim|ngsolve|netgen|gmsh|pyngcore)["']"""
    r"""|\bimport_devsim\(""")

# solver-gated through a mechanism _SOLVER_PAT cannot see: {file: why}
_INDIRECT = {
    "test_polarization_vocabulary.py": "probes turn an ImportError naming an _OPTIONAL extra "
                                       "(ngsolve/devsim/...) into pytest.skip -- 2 skips with them hidden",
}

# too slow for the PR tier, run by the `nightly` job, step (b1) -- see the `solvers` job's comment
NIGHTLY_ONLY = {"test_fem_drivers.py", "test_graded_tensor_fem.py"}
# does not fit any GitHub-hosted runner (memory-killed it twice) -- ci.yml top-of-file not-covered list
DEV_ONLY = {"test_shg_grating.py"}

_SCHEDULE_ONLY = "if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"


def _code_without_comments(path):
    """The module source with COMMENT tokens dropped (strings kept: child-process code lives in
    them). A comment that merely MENTIONS an importorskip of devsim -- test_transient_dt_floor explains
    why it does NOT use one -- is not a skip."""
    with open(path, encoding="utf-8") as fh:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(fh.read()).readline)
                if t.type != tokenize.COMMENT]
    return tokenize.untokenize(toks)


def _solver_gated_files():
    out = set(_INDIRECT)
    for path in sorted(glob.glob(os.path.join(REPO, "tests", "test_*.py"))):
        if _SOLVER_PAT.search(_code_without_comments(path)):
            out.add(os.path.basename(path))
    return out


def _jobs():
    """{job id: (installs [solvers], PR-reachable, test files its commands name)} from ci.yml.
    Comments are stripped first: they name files too (e.g. the not-covered list), and a file a job
    only TALKS about is exactly the silent gap this gate exists for."""
    with open(CI_YML, encoding="utf-8") as fh:
        src = fh.read()
    body = src[src.index("\njobs:\n"):]
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^  ([A-Za-z0-9_-]+):\s*$", body, re.M)]
    out = {}
    for k, (pos, name) in enumerate(starts):
        block = body[pos:starts[k + 1][0] if k + 1 < len(starts) else len(body)]
        code = "\n".join(re.sub(r"(^|\s)#.*$", "", ln) for ln in block.splitlines())
        installs = bool(re.search(r"pip install[^\n]*\[[^\]\n]*\bsolvers\b[^\]\n]*\]", code))
        pr = _SCHEDULE_ONLY not in code
        files = set(re.findall(r"tests/(test_\w+\.py)", code))
        out[name] = (installs, pr, files)
    return out


def test_the_detector_and_the_parser_are_not_vacuous():
    """Pin the fixtures both halves rest on, so a refactor that blinds either one fails loudly
    instead of passing every file through."""
    gated = _solver_gated_files()
    for must in ("test_devsim_smoke.py", "test_devsim_noise.py", "test_fem_drivers.py",
                 "test_ngsolve_ellipse_primitive.py", "test_equilibrium_simplify_cache.py"):
        assert must in gated, must
    assert "test_transient_dt_floor.py" not in gated        # shims devsim: runs on every leg
    assert len(gated) >= 35, len(gated)
    jobs = _jobs()
    assert jobs["solvers"][0] and jobs["solvers"][1] and len(jobs["solvers"][2]) >= 20
    assert jobs["nightly"][0] and not jobs["nightly"][1]
    assert not jobs["test"][0] and not jobs["floor"][0]     # the legs that SKIP these files


def test_every_solver_gated_test_file_runs_on_a_solver_leg():
    """THE gate: a solver-gated file named by no PR solver job (and not a documented nightly-only or
    dev-only module) runs nowhere on CI. Add it to the `solvers`/`solvers-devsim` file list, with its
    measured runtime, in the same commit that adds the file."""
    jobs = _jobs()
    pr = set().union(*(f for (inst, is_pr, f) in jobs.values() if inst and is_pr))
    nightly = set().union(*(f for (inst, is_pr, f) in jobs.values() if inst and not is_pr))
    missing = sorted(f for f in _solver_gated_files() - DEV_ONLY
                     if f not in pr and not (f in NIGHTLY_ONLY and f in nightly))
    assert not missing, ("solver-gated test files on NO solver CI leg (they skip everywhere else): "
                         "{}".format(missing))


def test_documented_exceptions_are_real():
    """NIGHTLY_ONLY files are actually run by a schedule-only solver job; DEV_ONLY files are named in
    ci.yml's not-covered list and run by NO job (if one becomes runnable, move it off DEV_ONLY)."""
    jobs = _jobs()
    nightly = set().union(*(f for (inst, is_pr, f) in jobs.values() if inst and not is_pr))
    assert NIGHTLY_ONLY <= nightly, NIGHTLY_ONLY - nightly
    named = set().union(*(f for (_i, _p, f) in jobs.values()))
    assert not (DEV_ONLY & named), DEV_ONLY & named
    with open(CI_YML, encoding="utf-8") as fh:
        head = fh.read().split("\non:", 1)[0]
    for f in DEV_ONLY:
        assert "tests/" + f in head, f


@pytest.mark.parametrize("job", ["solvers", "solvers-devsim", "nightly"])
def test_every_file_a_solver_job_names_exists(job):
    """A typo in a job's file list is a pytest usage error at best and a silently shorter list at
    worst (a renamed file); pin that every named file exists."""
    jobs = _jobs()
    assert job in jobs, sorted(jobs)
    for f in jobs[job][2]:
        assert os.path.isfile(os.path.join(REPO, "tests", f)), (job, f)
