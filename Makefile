.PHONY: test validate help

help:
	@echo "make test      - fast CI suite (numpy/scipy only: data model, dielectric DB,"
	@echo "                 Schrodinger-Poisson, solver-free bridge spine)"
	@echo "make validate  - heavy solver-backed validations, gated by exit code"
	@echo "                 (needs the [solvers] extra: ngsolve/devsim/gmsh; tens of minutes)"

# Fast gate -- this is what CI runs (no ngsolve/devsim required).
test:
	python -m pytest tests/ -q

# Full physics gate -- each validation exits non-zero on failure; run_all aggregates.
# Pass a filter, e.g.  make validate ARGS="oblique sp"
validate:
	python -m validation.run_all $(ARGS)
