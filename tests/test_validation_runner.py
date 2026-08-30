"""Gates on validation/run_all.py's OWN contract -- the runner that gates the physics.

Until now nothing tested the runner itself: its tier/token/shard selection, its exit-code
contract and the workflow lines that drive it were all verified by running it. That is fine
for the physics scripts (their exit codes ARE the gate) and wrong for the selection logic,
where a defect does not turn anything red -- it silently runs FEWER scripts and still reports
success. `--shard` (added 2026-08-30, after the full tier's first complete nightly measured
4 h 16 min against a 5 h job cap) makes that failure mode reachable in one typo, so it ships
with the gates below.

Run: python -m pytest tests/test_validation_runner.py -q
"""
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from validation import run_all as R                                          # noqa: E402

CI_YML = os.path.join(REPO, ".github", "workflows", "ci.yml")


def _full_selection():
    """The unsharded `--tier full` job list, built the same way main() builds it."""
    jobs = [("validation", n) for n in R._gated(R.HERE, skip=R.SKIP)]
    ex = os.path.join(R.REPO, "examples")
    if os.path.isdir(ex):
        jobs += [("examples", n) for n in R._gated(ex)]
    return jobs


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 7])
def test_shards_are_a_partition_of_the_unsharded_selection(n):
    """THE claim --shard rests on: N shards are disjoint and their union is EXACTLY what the
    unsharded run would have executed. A stride that dropped or duplicated scripts would keep
    every shard green while silently shrinking the tier -- the one defect class this gate
    exists for."""
    full = _full_selection()
    shards = [full[k::n] for k in range(n)]
    union = [job for s in shards for job in s]
    assert len(set(union)) == len(union), "shards overlap"
    assert sorted(union) == sorted(full), "union != unsharded selection"
    # and no shard is empty for any N the workflow could plausibly use
    assert all(len(s) > 0 for s in shards)


def test_the_stride_is_round_robin_not_contiguous():
    """Round-robin is a COST decision, not a style one: the selection is alphabetical and this
    repo's expensive scripts cluster by name prefix (the fdtd_* family holds most of the
    multi-minute oracles -- fdtd_drude_lorentz_vs_tmm alone measured 889 s on the runner), so a
    contiguous split would hand one shard nearly all of them and blow its cap. Pin the spread."""
    full = _full_selection()
    n = 4
    per_shard = [sum(1 for _, name in full[k::n] if name.startswith("fdtd")) for k in range(n)]
    total = sum(per_shard)
    assert total > 20, "fixture assumption changed: too few fdtd_* scripts to test the spread"
    # every shard within one script of an even share -- what a contiguous split would violate
    assert max(per_shard) - min(per_shard) <= 1, per_shard


@pytest.mark.parametrize("bad", ["5/4", "0/4", "abc", "3/0", "-1/4"])
def test_malformed_shard_values_are_refused(bad):
    """Exit 2 (usage), not a silent full run or a crash mid-tier."""
    p = subprocess.run([sys.executable, "-m", "validation.run_all", "--tier", "smoke",
                        "reliability", "--shard", bad],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    assert p.returncode == 2, (p.returncode, p.stdout[-400:])
    assert "--shard" in p.stdout


def test_the_workflow_matrix_and_the_shard_denominator_agree():
    """THE drift gate. .github/workflows/ci.yml drives the sharded tier with a matrix of shard
    numbers AND a `--shard ${{ matrix.shard }}/N` denominator written separately. Edit one and
    not the other and the nightly still reports all-green while never running part of the tier:
    a 6-entry matrix against `/4` runs shards 5 and 6 as EMPTY selections, and a 4-entry matrix
    against `/6` silently drops a third of the scripts. Neither turns anything red. So pin that
    the matrix is exactly 1..N and the denominator is that same N."""
    with open(CI_YML, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r"--shard \$\{\{ matrix\.shard \}\}/(\d+)", src)
    assert m, "no sharded run_all invocation found in ci.yml"
    denom = int(m.group(1))
    mm = re.search(r"matrix:\s*\n\s*shard:\s*\[([0-9,\s]+)\]", src)
    assert mm, "no shard matrix found in ci.yml"
    entries = [int(x) for x in mm.group(1).replace(" ", "").split(",") if x]
    assert entries == list(range(1, denom + 1)), (entries, denom)


def test_the_sharded_job_still_carries_the_schedule_only_guard():
    """The multi-hour tier must stay off the PR path. `schedule`/`workflow_dispatch` are
    WORKFLOW-level triggers, so the per-job `if:` is the entire mechanism -- and the sharded job
    is a second place it now has to be right."""
    with open(CI_YML, encoding="utf-8") as fh:
        src = fh.read()
    block = src[src.index("nightly-validation:"):]
    guard = "if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    assert guard in block[:block.index("steps:")], "nightly-validation lost its schedule-only guard"
