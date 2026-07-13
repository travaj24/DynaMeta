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
    "optical_cache", "scattering_link", "schrodinger_poisson", "sp_carrier",
    "sp_carrier_nonparabolic", "sp_neumann_body", "sp_nonparabolic", "sp_open_body",
    "sp_oxide_cap", "sp_per_column", "sp_self_consistent_nonparabolic", "switching_drivers",
    "transient_optics_response", "vector_mo_tensor",
    # the solver-free QD-SOA family (audit 1.2/6.3: flagged by the reverse-drift warning
    # since it shipped; pure numpy -- the newest never-CI'd subsystem now smoke-gated)
    "qd_soa_alpha_pdg", "qd_soa_ase_zresolved", "qd_soa_bidir_ase",
    "qd_soa_calibration_innolume", "qd_soa_eh_split", "qd_soa_electrical_rc",
    "qd_soa_enob_budget", "qd_soa_es_band", "qd_soa_fabry_perot",
    "qd_soa_filament_qd", "qd_soa_fwm_xgm", "qd_soa_gain_core",
    "qd_soa_gvd", "qd_soa_gvd_distributed", "qd_soa_hammer",
    "qd_soa_inferred_dynamics", "qd_soa_langevin", "qd_soa_leakage",
    "qd_soa_many_body", "qd_soa_maxwell_bloch", "qd_soa_noise_metrics",
    "qd_soa_nonlinear_loss", "qd_soa_nonmarkovian", "qd_soa_numba_parity",
    "qd_soa_rin_linewidth", "qd_soa_saturation_power", "qd_soa_sbe",
    "qd_soa_spectral_dispersion", "qd_soa_thermal_profile", "qd_soa_transport",
    "qd_soa_transverse_bpm", "qd_soa_traveling_wave", "qd_soa_ultrafast",
    "qd_soa_vectorial_pdg", "qd_soa_wdm",
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
        # audit 6.3: the old behavior INCLUDED only scripts matching this regex, silently
        # classifying a raise/assert-gated validation as a diagnostic and NEVER RUNNING it.
        # Any non-zero exit gates equally well, so membership is now everything-except-SKIP
        # and the regex only powers a visible drift note.
        if not re.search(r"SystemExit|sys\.exit", src):
            print("[run_all] NOTE: {} has no explicit exit-gate text -- it runs anyway "
                  "(non-zero exit = FAIL); verify it actually gates".format(name), flush=True)
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
    all_gated = {n for _, n in jobs}                    # full gated set, BEFORE the SMOKE filter
    if tier == "smoke":
        jobs = [(p, n) for p, n in jobs if n in SMOKE]
        missing = SMOKE - {n for _, n in jobs}
        if missing:                                     # tier-set drift (missing direction)
            print("[run_all] WARNING: SMOKE names with no matching gated script: {}".format(
                ", ".join(sorted(missing))), flush=True)
        # reverse-direction drift: a gated script NOT in SMOKE that imports NO heavy solver is a
        # SOLVER-FREE validation the smoke tier silently excludes (e.g. the qd_soa_* family). Surface
        # it loudly so a new subsystem is curated into SMOKE deliberately, not dropped by omission.
        # A script is "heavy" if its SOURCE references a heavy solver DIRECTLY (import ngsolve/devsim/
        # gmsh) OR TRANSITIVELY through a DynaMeta wrapper that pulls one: optics.solver / ngsolve_layered
        # -> ngsolve; optics.fdtd* -> multi-minute time loops; carriers.devsim* / thermal_fem /
        # electrostatics_fem -> devsim/ngsolve; optics.inverse_design / topology_opt -> jax-FDTD;
        # dynameta.pipeline -> devsim+ngsolve. The earlier literal import-only scan misclassified ~80
        # FEM/FDTD/DEVSIM scripts as solver-free (they import their solver through these wrappers, not
        # via a top-level `import ngsolve`) -- pass-2 re-audit P2. Substrings (not import-form) so a
        # deferred/in-function import (e.g. behind a find_spec skip-guard) still counts as heavy.
        _HEAVY = (
            "ngsolve", "devsim", "import gmsh", "from gmsh",           # direct heavy deps
            "optics.solver", "optics.fdtd", "optics.inverse_design",  # FEM / FDTD / jax-FDTD wrappers
            "optics.topology_opt", "carriers.thermal_fem", "carriers.electrostatics_fem",
            "carriers.devsim", "dynameta.pipeline", "LayeredOpticalBuilder", "make_fem_optical_solver",
        )
        extra = []
        for n in sorted(all_gated - SMOKE):
            src = open(os.path.join(HERE, n + ".py"), encoding="utf-8").read()
            if not any(h in src for h in _HEAVY):
                extra.append(n)
        if extra:
            # "no heavy path" = no NGSolve/DEVSIM/FDTD/jax-FDTD import; a few entries still pull
            # lumenairy/jax (the bridge validations) -- fast but not pure-numpy, so verify the runtime
            # before adding them to the pure-numpy smoke tier.
            print("[run_all] WARNING: {} gated validation(s) with NO NGSolve/DEVSIM/FDTD heavy path NOT "
                  "in SMOKE (curate into the smoke tier or confirm intentional; verify runtime -- a few "
                  "pull lumenairy/jax): {}".format(len(extra), ", ".join(extra)), flush=True)
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
        # audit C6-6: rc == 42 is the SKIP convention (required capability absent --
        # CUDA/cupy/jax/ngsolve/devsim/lumenairy not installed), counted separately so a
        # never-executed physics gate cannot read as a green PASS in the summary
        tag = ("PASS" if rc == 0 else "SKIP" if rc == 42
               else ("TIMEOUT" if rc == 124 else "FAIL(rc={})".format(rc)))
        results.append((pkg + "." + name, rc, dt))
        print("[run_all] {:48s} {:8s} ({:5.0f}s)".format(pkg + "." + name, tag, dt), flush=True)
    skipped = [r for r in results if r[1] == 42]
    failed = [r for r in results if r[1] not in (0, 42)]
    print("\n[run_all] {}/{} passed; {} skipped (capability absent); {} failed/errored".format(
        len(results) - len(failed) - len(skipped), len(results), len(skipped), len(failed)),
        flush=True)
    if skipped:
        print("[run_all] SKIPPED: " + ", ".join(n for n, _, _ in skipped), flush=True)
    if failed:
        print("[run_all] FAILURES: " + ", ".join(n for n, _, _ in failed), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
