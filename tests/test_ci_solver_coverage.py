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

THE PYTEST COMMAND LINES THEMSELVES (2026-11-06). The file-name pattern above reads `tests/test_x.py`
out of the text, so it cannot see what the SHELL will do with the line. Twice on 2026-11-04/05 a
`\\n` was typed where a line continuation belonged (`tests/test_a.py \\n    tests/test_b.py`): it is
still valid YAML, the pattern still finds both files, and bash turns the unquoted `\\n` into a stray
positional `n` -- pytest then errors on "file or directory not found: n" (or, under a shell that
keeps the backslash, on `\\n`). `test_every_pytest_run_command_is_well_formed` shell-splits every
pytest `run:` command after joining backslash-newline continuations exactly as bash does, and
requires every positional argument to be an existing path under tests/ and no argument to carry a
backslash.

Run: python -m pytest tests/test_ci_solver_coverage.py -q
"""
import glob
import io
import os
import re
import shlex
import textwrap
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


def _ci_text():
    with open(CI_YML, encoding="utf-8") as fh:
        return fh.read()


def _jobs(src=None):
    """{job id: (installs [solvers], PR-reachable, test files its commands name)} from ci.yml (or the
    workflow text `src`). Comments are stripped first: they name files too (e.g. the not-covered
    list), and a file a job only TALKS about is exactly the silent gap this gate exists for."""
    if src is None:
        src = _ci_text()
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


# ------------------------------------------------------------------ the pytest command lines themselves
_RUN_KEY = re.compile(r"^(?P<ind> *)(?:- +)?run:[ \t]*(?P<rest>.*)$")
# pytest options that take their value as the NEXT argument (a `--opt=value` spelling needs no entry).
# An option missing here makes its value read as a positional path, so this gate fails loudly rather
# than passing a stray argument: add the option, do not loosen the path rule.
_PYTEST_VALUE_OPTS = frozenset({"-m", "-k", "-n", "-p", "-c", "-o", "-W", "--cov", "--cov-report",
                                "--cov-fail-under", "--durations", "--rootdir", "--deselect", "--ignore",
                                "--maxfail", "--junitxml", "--basetemp", "--dist"})
_SHELL_OPERATORS = frozenset({";", "&&", "||", "|", "&"})


def _run_scripts(src):
    """[(line number, script)] for every `run:` of a workflow text: a block scalar (`run: |`) is its
    more-indented lines, dedented (YAML keeps a typed `\\n` in it as the two characters); a one-line
    plain scalar is its text. Folded scalars (`>`) are returned with a None script: the gate refuses
    them rather than guess how YAML joins their lines."""
    lines = src.splitlines()
    out, k = [], 0
    while k < len(lines):
        m = _RUN_KEY.match(lines[k])
        if not m:
            k += 1
            continue
        ind, rest = len(m.group("ind")), m.group("rest").strip()
        if rest[:1] in ("|", ">"):
            j = k + 1
            while j < len(lines) and (not lines[j].strip()
                                      or len(lines[j]) - len(lines[j].lstrip(" ")) > ind):
                j += 1
            body = "\n".join(lines[k + 1:j])
            out.append((k + 1, None if rest[0] == ">" else textwrap.dedent(body)))
            k = j
        else:
            out.append((k + 1, rest))
            k += 1
    return out


def _pytest_run_problems(src):
    """(number of pytest invocations, [problem]) over every `run:` of the workflow text `src`.

    Each script is read as bash reads it: backslash-newline continuations are removed first, then each
    line is split POSIX-style into commands at ; && || | &. A command is a pytest run when it starts
    with `pytest` or `python -m pytest`. Problems: (a) a backslash left anywhere on that joined line
    (a continuation is the only backslash a pytest line in this file has a use for, and it is gone by
    now -- what remains is a typed `\\n` or a `\\ ` not at the end of the line); (b) a positional
    argument (not an option, not the value of a _PYTEST_VALUE_OPTS option) that is not an existing
    path under tests/ (a `path::node` id is checked by its path); (c) a line shlex cannot split, or a
    folded scalar."""
    n_cmd, problems = 0, []
    for lineno, script in _run_scripts(src):
        if script is None:
            problems.append("ci.yml:{}: folded `run: >` scalar (not checked; use `run: |`)".format(lineno))
            continue
        for line in script.replace("\\\n", "").splitlines():
            if "pytest" not in line:
                continue
            lex = shlex.shlex(line, posix=True, punctuation_chars=True)
            lex.whitespace_split = True
            try:
                toks = list(lex)
            except ValueError as exc:
                problems.append("ci.yml:{}: cannot shell-split ({}): {}".format(lineno, exc, line.strip()))
                continue
            cmds, cur = [], []
            for t in toks + [";"]:
                if t in _SHELL_OPERATORS:
                    if cur:
                        cmds.append(cur)
                    cur = []
                else:
                    cur.append(t)
            for cmd in cmds:
                if cmd[:1] == ["pytest"]:
                    args = cmd[1:]
                elif len(cmd) >= 3 and cmd[0] in ("python", "python3") and cmd[1:3] == ["-m", "pytest"]:
                    args = cmd[3:]
                else:
                    continue
                n_cmd += 1
                where = "ci.yml:{} (`run:` key line)".format(lineno)
                if "\\" in line:
                    problems.append("{}: literal backslash in a pytest command: {}".format(where, line.strip()))
                skip_value = False
                for a in args:
                    if skip_value:
                        skip_value = False
                        continue
                    if a.startswith("-"):
                        skip_value = "=" not in a and a in _PYTEST_VALUE_OPTS
                        continue
                    path = a.split("::", 1)[0]
                    if not (path.startswith("tests/") and os.path.exists(os.path.join(REPO, path))):
                        problems.append("{}: positional argument {!r} is not an existing path under "
                                        "tests/".format(where, a))
    return n_cmd, problems


def test_every_pytest_run_command_is_well_formed():
    """Every pytest invocation in ci.yml, split as bash will split it, names only existing paths under
    tests/ and carries no stray backslash (module docstring: the typed-`\\n` defect)."""
    n_cmd, problems = _pytest_run_problems(_ci_text())
    # non-vacuous: 9 pytest invocations on 2026-11-06 (test x2, floor, numba, solvers x2,
    # solvers-devsim, nightly (a) and (b1))
    assert n_cmd >= 9, n_cmd
    assert not problems, "\n".join(problems)


def test_the_command_gate_catches_a_typed_backslash_n():
    """Shown failing on the real defect: ci.yml with ONE continuation of the `solvers-devsim` file list
    replaced by a typed `\\n` (the next file joins the line), as happened twice on 2026-11-04/05. It is
    still valid YAML, and the file-name parser the drift gates use sees exactly the same file set -- that
    is the gap. The command gate must name both symptoms: the backslash and bash's stray `n`."""
    real = _ci_text()
    bad, n_sub = re.subn(r"(tests/test_devsim_heterojunction\.py) \\\n( +)", r"\1 \\n\2", real, count=1)
    assert n_sub == 1                                    # the fixture line exists in the real file
    assert _jobs(bad)["solvers-devsim"][2] == _jobs(real)["solvers-devsim"][2]   # invisible to it
    n_cmd, problems = _pytest_run_problems(bad)
    assert n_cmd == _pytest_run_problems(real)[0]
    assert any("literal backslash" in p for p in problems), problems
    assert any("positional argument 'n'" in p for p in problems), problems
    # adversarial: a misspelled path, a stray word and a `path::node` id of a missing file are each caught;
    # an option value (-k "pardiso or auto") and a real node id are not
    for arg, caught in (("tests/test_devsim_noize.py", True), ("tests", True), ("x", True),
                        ("tests/test_nope.py::test_a", True),
                        ("tests/test_ci_solver_coverage.py::test_documented_exceptions_are_real", False)):
        text = ("jobs:\n  j:\n    steps:\n      - run: |\n          pytest -q -k \"a or b\" \\\n"
                "            {}\n".format(arg))
        n, probs = _pytest_run_problems(text)
        assert n == 1 and bool(probs) is caught, (arg, probs)
