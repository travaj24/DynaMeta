.PHONY: test lint validate help

help:
	@echo "make test      - the WHOLE tests/ tree, @slow included; modules needing the"
	@echo "                 [solvers] extra (ngsolve/devsim/gmsh), jax, numba or zarr run"
	@echo "                 where installed and skip otherwise -- so with solvers installed"
	@echo "                 this is the full suite, not a fast one. CI's test legs add"
	@echo "                 -m \"not slow\": make test ARGS='-m \"not slow\"'"
	@echo "make lint      - ruff over the whole tree (the same check the CI lint leg runs)"
	@echo "make validate  - heavy solver-backed validations, gated by exit code"
	@echo "                 (needs the [solvers] extra: ngsolve/devsim/gmsh; tens of minutes)"

# audit S5-19 (wave-7, 2026-11-02): the help text above used to call this a "fast local suite
# (numpy/scipy only ...)"; it runs every module under tests/, and the solver-backed ones are
# skipped only when their extra is ABSENT. ARGS is forwarded (default empty = unchanged).
# NOT the whole of CI -- audit R-9: this comment used to
# claim "this is what CI runs". CI runs six legs: four `test` legs across the python matrix
# (each `-m "not slow"`, with coverage on 3.12), a declared-floor pin leg, a numba/Windows leg,
# a ruff `lint` leg and a solvers leg (see .github/workflows/ci.yml). This target is the closest
# local approximation of the `test` legs; add `-m "not slow"` to match them exactly, and run
# `make lint` for the lint leg.
test:
	python -m pytest tests/ -q $(ARGS)

# audit T-4/R-5: ruff is pinned in [dev] and enforced by its own CI leg, but had no make target,
# so the local loop never ran it and 179 violations accumulated. The enforced surface is pinned
# in pyproject ([tool.ruff.lint] select = F, E9), not here.
lint:
	ruff check .

# Full physics gate -- each validation exits non-zero on failure; run_all aggregates.
# Pass a filter, e.g.  make validate ARGS="oblique sp"
validate:
	python -m validation.run_all $(ARGS)
