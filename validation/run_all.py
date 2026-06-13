"""Run every PASS/FAIL validation as a subprocess and gate on its EXIT CODE (audit
cross-cutting F1/F3). Each validation `raise SystemExit(0 if ok else 1)`, so this is a
single machine-checkable command for the solver-backed physics that the fast CI suite
(`pytest tests/`, which covers the solver-free bridge spine + Schrodinger + data model)
cannot run -- the heavy NGSolve/DEVSIM stack is not installed in CI.

  python -m validation.run_all                       # all gated validations (legacy default)
  python -m validation.run_all --tier smoke          # fast solver-free subset (~minutes; CI-able)
  python -m validation.run_all --tier full           # everything + the examples/ workflows
  python -m validation.run_all oblique sp            # only scripts whose name matches a token
  python -m validation.run_all --tier smoke lc       # tier and tokens compose (AND)

Tiers:
  smoke -- the SMOKE set below: pure numpy/scipy scripts measured/verified < ~30 s each, no
           DEVSIM/NGSolve and no multi-minute FDTD time loops. A NEW fast solver-free
           validation must be ADDED to the set explicitly (opt-in keeps the tier honest:
           import-grepping misclassifies lazy-import and long-numpy cases).
  full  -- every gated validation PLUS the exit-gated examples/ workflows.

Exits non-zero if ANY selected script fails or errors. Pure-diagnostic scripts (no PASS/FAIL
verdict) are skipped. Budget: smoke ~ a few minutes; full ~ hours with the [solvers] extra.
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# pure diagnostics (no PASS/FAIL gate) + this runner
SKIP = {"run_all", "oblique_field_dump", "oblique_phase_diag", "oblique_sign_pml_diag",
        "reference_modulator_spectrum"}
PER_SCRIPT_TIMEOUT_S = 1800

# The fast solver-free tier (opt-in; see module docstring). Verified 2026-06-10: each is pure
# numpy/scipy at runtime (lazy-import traps like results_io_demo included deliberately;
# runtime-heavy-but-import-light traps like magneto_optic_faraday excluded deliberately).
SMOKE = {
    "backend_autodiff", "bandwidth_cv", "burstein_moss_blueshift", "carrier_heating_enz",
    "density_gradient_dead_layer", "drude_matthiessen_kane", "fdtd_layered",
    "intersubband_eps_zz", "lc_chiral_twist", "lc_cyl_flexo_optics", "lc_director_2d",
    "lc_director_dynamics", "lc_dynamics_anchoring_backflow", "lc_two_constant_bvp",
    "lc_weak_anchoring_temperature", "llg_macrospin", "modulator_design_space",
    "pcm_nucleation_growth", "qcse_density_screening", "qcse_electroabsorption",
    "qcse_elliott_mqw", "qcse_voigt_lineshape", "reconfigurable_modulators",
    "reliability_bti", "reliability_corrosion", "reliability_dedoping", "reliability_em",
    "reliability_fatigue", "reliability_hci", "reliability_leakage", "reliability_lidt",
    "reliability_mttf", "reliability_stressmig", "reliability_tddb", "results_io_demo",
    "scattering_link", "schrodinger_poisson", "sp_carrier", "sp_carrier_nonparabolic",
    "sp_neumann_body", "sp_nonparabolic", "sp_oxide_cap", "sp_per_column",
    "sp_self_consistent_nonparabolic", "switching_drivers", "transient_optics_response",
    "vector_mo_tensor",
}


def _gated(directory, skip=()):
    out = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        name = fn[:-3]
        if name in skip:
            continue
        src = open(os.path.join(directory, fn), encoding="utf-8").read()
        if re.search(r"SystemExit|sys\.exit", src):     # only exit-code-gated scripts
            out.append(name)
    return out


def main(argv):
    args = argv[1:]
    tier = None
    if "--tier" in args:
        i = args.index("--tier")
        try:
            tier = args[i + 1]
        except IndexError:
            print("[run_all] --tier needs a value: smoke | full", flush=True)
            return 2
        if tier not in ("smoke", "full"):
            print("[run_all] unknown tier {!r} (smoke | full)".format(tier), flush=True)
            return 2
        args = args[:i] + args[i + 2:]
    tokens = args

    jobs = [("validation", n) for n in _gated(HERE, skip=SKIP)]
    n_gated = len(jobs)
    if tier == "smoke":
        jobs = [(p, n) for p, n in jobs if n in SMOKE]
        missing = SMOKE - {n for _, n in jobs}
        if missing:                                     # tier-set drift is loud, not silent
            print("[run_all] WARNING: SMOKE names with no matching gated script: {}".format(
                ", ".join(sorted(missing))), flush=True)
    elif tier == "full":
        ex_dir = os.path.join(REPO, "examples")
        if os.path.isdir(ex_dir):
            jobs += [("examples", n) for n in _gated(ex_dir)]
    if tokens:
        jobs = [(p, n) for p, n in jobs if any(t in n for t in tokens)]

    print("[run_all] {} of {} gated validations selected (tier={}{}){}\n".format(
        len([j for j in jobs if j[0] == "validation"]), n_gated, tier or "all",
        ", tokens=" + ",".join(tokens) if tokens else "",
        "; + {} examples".format(len([j for j in jobs if j[0] == "examples"]))
        if any(p == "examples" for p, _ in jobs) else ""), flush=True)

    results = []
    for pkg, name in jobs:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-u", "-m", pkg + "." + name],
                               cwd=REPO, timeout=PER_SCRIPT_TIMEOUT_S)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
        dt = time.time() - t0
        tag = "PASS" if rc == 0 else ("TIMEOUT" if rc == 124 else "FAIL(rc={})".format(rc))
        results.append((pkg + "." + name, rc, dt))
        print("[run_all] {:48s} {:8s} ({:5.0f}s)".format(pkg + "." + name, tag, dt), flush=True)
    failed = [r for r in results if r[1] != 0]
    print("\n[run_all] {}/{} passed; {} failed/errored".format(
        len(results) - len(failed), len(results), len(failed)), flush=True)
    if failed:
        print("[run_all] FAILURES: " + ", ".join(n for n, _, _ in failed), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
