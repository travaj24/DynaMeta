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
  python -m validation.run_all --tier smoke --allow-skip qd_soa_numba_parity   # declared exception

Tiers:
  smoke -- the SMOKE set below: solver-free scripts MEASURED at < 60 s each, no DEVSIM/NGSolve and
           no multi-minute FDTD time loops (most are pure numpy/scipy and finish in seconds; the
           lumenairy-bridge gates pull lumenairy/jax, both of which CI already installs). A NEW
           fast solver-free validation must be ADDED to the set explicitly (opt-in keeps the tier
           honest: import-grepping misclassifies lazy-import and long-numpy cases). A solver-free
           script that MEASURES too slow goes in SMOKE_EXCLUDED with its time, so the reverse-drift
           warning below names only genuinely uncurated scripts (audit X-26).
  full  -- every gated validation PLUS the exit-gated examples/ workflows.

EXIT-CODE CONTRACT (audit T-3 -- the tier used to read green while executing nothing):
  0   PASS.
  42  SKIP: the script declared a required capability absent (numba/CUDA/jax/ngsolve/devsim).
      Counted SEPARATELY from PASS in the summary, never green-washed into it.
      * In the SMOKE tier a 42 is a FAILURE by default: smoke is the CI-able tier, so a
        capability it needs is a capability CI was supposed to install. Naming the script in
        --allow-skip turns that back into an informational skip -- and forces the exception
        to be visible in the workflow file (e.g. numba, which the dedicated `numba` CI leg
        runs for real instead).
      * In the other tiers a 42 stays informational: those tiers legitimately run without
        the [solvers] extra.
  124 TIMEOUT (PER_SCRIPT_TIMEOUT_S).
  ANY OTHER non-zero (including a code this runner does not know) is a FAILURE. There is no
  "unrecognized, therefore harmless" branch.

Exits non-zero if ANY selected script fails, errors, times out, or (smoke, undeclared) skips.
Pure-diagnostic scripts (no PASS/FAIL verdict) are excluded. Budget: smoke ~ a few minutes;
full ~ hours with the [solvers] extra.
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
# The `_` prefix excludes a file from EVERY tier (see _gated), and it carries TWO opposite
# meanings: a legitimately SHARED FIXTURE (imported by other validations, never run on its own)
# and a PARKED WIP (a real gate suite that nothing executes). Declare the fixtures here so the
# second kind is NAMED on every run instead of vanishing silently -- audit X-6, where a parked
# `_dg_hard_wall_wip.py` was the only caller of a shipped `__all__` export, so that export had no
# executable gate anywhere and nobody could see it.
FIXTURES = {"_reference_device"}
PER_SCRIPT_TIMEOUT_S = 1800
SKIP_RC = 42                      # the capability-absent convention (audit C6-6 / T-3)
TIMEOUT_RC = 124

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
    "sp_carrier_nonparabolic", "sp_neumann_body", "sp_nonparabolic",
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
    # the rare-earth fiber amplifier (EDFA/YDFA) end-to-end gates -- pure numpy/scipy.
    # fiber_radial_inversion = the radially-resolved-inversion / TSHB gates. MEASURED 2026-08-04,
    # this box, cold subprocess: 29.2 / 29.3 / 40.9 / 54 / 56.8 / 58.9 s wall. That spread is NOT
    # the gates -- the script prints its own total, which was 0.4-2.0 s in every one of those runs
    # -- it is `import dynameta.optics.fiber_amp`, measured ALONE at 14 s (quiet box) to 40 s
    # (loaded box) in the same session. fiber_amp_physics, already in this tier and paying the
    # same import, measured 38.6 / 42.0 / 45.6 / 47.6 / 56.6 s across the same runs, so the new
    # script is the CHEAPER of the two on its own work and adds no new dependency. Under the 60 s
    # bar on the merits; if the tier ever needs to shed time, the import is the thing to attack,
    # not either script.
    "fiber_amp_physics", "fiber_radial_inversion",
    # audit X-19/X-26: the solver-free validations the reverse-drift warning below had been naming
    # since they shipped. `lumenairy` is a CORE dependency CI already installs, so the bridge gates
    # -- the only executable coverage of the 11-module optics/lumenairy_bridge subsystem -- cost
    # nothing extra to run. Wall-clock MEASURED 2026-07-26 (this box, cold start, per script):
    #   bic_capstone 13.9 s | inverse_design_oracle 3.6 s | lumenairy_berreman_bridge 3.0 s
    #   lumenairy_emt_screen 2.1 s | lumenairy_translate 3.1 s | lumenairy_rcwa_jax 35.8 s
    #   lumenairy_pmm_bridge 47.8 s | nonlocal_hydro_material 0.4 s | resonance_pole_finder 2.5 s
    # = ~112 s added to the tier. The three that measured OVER the 60 s bar are in SMOKE_EXCLUDED
    # below with their times, so the reverse-drift warning stays signal rather than noise.
    "bic_capstone", "inverse_design_oracle", "lumenairy_berreman_bridge", "lumenairy_emt_screen",
    "lumenairy_pmm_bridge", "lumenairy_rcwa_jax", "lumenairy_translate",
    # audit X-19: cheap smoke wrappers for two optics modules that had NO validation script at all,
    # so their only oracle was the test written beside the implementation in the same commit
    "nonlocal_hydro_material", "resonance_pole_finder",
}

# Solver-free validations DELIBERATELY outside SMOKE, with the reason (audit X-26: "add them to
# SMOKE after measuring runtime, or record an explicit exclusion reason next to each so the
# reverse-drift warning stops being noise"). The reverse-drift check below subtracts this set, so
# anything it still names is genuinely uncurated. Re-measure before moving one INTO the tier.
SMOKE_EXCLUDED = {
    # The scalar gain-BPM field solver. Unlike the other two fiber scripts, whose run_all time is
    # almost entirely `import dynameta.optics.fiber_amp`, this one does real work of its own -- and
    # that work is CPU-bound FFT marching, so unlike an import it does not amortize against a busy
    # box. MEASURED 2026-08-04, same box, same session, gate body / run_all: 16.1-17.6 s / 18-23 s
    # with the box quiet (fiber_radial_inversion was 0.6 s / 3 s in the same runs), and
    # 61.7-86.4 s / 108-145 s with one competing full-CPU workload. The dev box's own reading that
    # day was 66 s in run_all. Quiet it would fit the tier; loaded it is 2x over, and the smoke
    # tier is the one CI runs on shared hardware -- so it stays out until the FFT budget or the
    # 14-40 s fiber_amp import is attacked.
    "fiber_gain_bpm":            "gate body 16-18 s quiet / 62-86 s on a loaded box, 18-145 s in "
                                 "run_all, measured 2026-08-04 -- over the ~60 s smoke bar "
                                 "whenever the box is busy",
    "lumenairy_bor_bridge":      "62 s measured 2026-07-26 -- over the ~60 s per-script smoke bar",
    "lumenairy_berreman_jax":    "145 s measured 2026-07-26 (jax jit warm-up + sweep)",
    "lumenairy_pmm2d_bridge":    "268 s measured 2026-07-26 (2-D PMM cascade; the slowest gate)",
}


def _gated(directory, skip=()):
    out = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".py"):
            continue
        if fn.startswith("_"):
            # audit X-6: a `_`-prefixed file runs in NO tier. That is correct for a shared fixture
            # and wrong for a parked gate suite, and the prefix cannot tell them apart -- so name
            # every non-fixture one loudly rather than dropping it in silence.
            if fn[:-3] not in FIXTURES:
                print("[run_all] NOTE: {}/{} is `_`-prefixed, so it is EXCLUDED FROM EVERY TIER. "
                      "If it is a gate suite, rename it (drop the underscore) or gate it from "
                      "tests/; if it is a shared fixture, add it to FIXTURES.".format(
                          os.path.basename(directory), fn), flush=True)
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
    # audit T-3: declared exceptions to the smoke tier's strict-skip rule. Repeatable and/or
    # comma-separated; a name here must be a script the CALLER runs elsewhere for real (the
    # workflow file is where that promise is visible).
    allow_skip = set()
    while "--allow-skip" in args:
        i = args.index("--allow-skip")
        try:
            value = args[i + 1]
        except IndexError:
            print("[run_all] --allow-skip needs a value: NAME[,NAME...]", flush=True)
            return 2
        allow_skip.update(n for n in value.replace(",", " ").split() if n)
        args = args[:i] + args[i + 2:]
    tokens = args
    # smoke is the CI-able tier, so "capability absent" there means CI did not install
    # something it was supposed to -- a red, not a shrug (audit T-3).
    strict_skip = tier == "smoke"

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
            "carriers.devsim", "carriers.physics_equilibrium",   # physics_equilibrium imports devsim
            "dynameta.pipeline", "LayeredOpticalBuilder", "make_fem_optical_solver",
        )
        extra = []
        for n in sorted(all_gated - SMOKE - set(SMOKE_EXCLUDED)):
            src = open(os.path.join(HERE, n + ".py"), encoding="utf-8").read()
            if not any(h in src for h in _HEAVY):
                extra.append(n)
        if extra:
            # "no heavy path" = no NGSolve/DEVSIM/FDTD/jax-FDTD import; a few entries still pull
            # lumenairy/jax (the bridge validations) -- fast but not pure-numpy, so verify the runtime
            # before adding them to the pure-numpy smoke tier. A script that has been MEASURED and
            # judged too slow belongs in SMOKE_EXCLUDED with its time, not here (audit X-26).
            print("[run_all] WARNING: {} gated validation(s) with NO NGSolve/DEVSIM/FDTD heavy path NOT "
                  "in SMOKE and NOT in SMOKE_EXCLUDED (curate into the smoke tier, or record a measured "
                  "exclusion reason in SMOKE_EXCLUDED): {}".format(len(extra), ", ".join(extra)),
                  flush=True)
        # keep SMOKE_EXCLUDED honest in both directions: a name that no longer exists, or that has
        # since been curated INTO SMOKE, is stale bookkeeping the next reader would trust.
        stale_ex = [n for n in sorted(SMOKE_EXCLUDED) if n not in all_gated or n in SMOKE]
        if stale_ex:
            print("[run_all] WARNING: SMOKE_EXCLUDED name(s) that are no longer excludable gated "
                  "scripts (deleted, or now in SMOKE): {}".format(", ".join(stale_ex)), flush=True)
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
            rc = TIMEOUT_RC
        dt = time.time() - t0
        # audit C6-6: rc == SKIP_RC is the SKIP convention (required capability absent --
        # CUDA/cupy/jax/ngsolve/devsim/lumenairy not installed), counted separately so a
        # never-executed physics gate cannot read as a green PASS in the summary. audit T-3:
        # in the smoke tier that skip is itself a FAILURE unless --allow-skip declares it,
        # because smoke is the tier CI is supposed to be able to run in full.
        undeclared_skip = rc == SKIP_RC and strict_skip and name not in allow_skip
        if rc == 0:
            tag = "PASS"
        elif rc == SKIP_RC:
            tag = "SKIP!" if undeclared_skip else "SKIP"
        elif rc == TIMEOUT_RC:
            tag = "TIMEOUT"
        else:
            tag = "FAIL(rc={})".format(rc)          # incl. any rc this runner does not know
        results.append((pkg + "." + name, rc, dt, undeclared_skip))
        print("[run_all] {:48s} {:8s} ({:5.0f}s)".format(pkg + "." + name, tag, dt), flush=True)
    bad_skips = [r for r in results if r[3]]
    skipped = [r for r in results if r[1] == SKIP_RC and not r[3]]
    failed = [r for r in results if r[1] not in (0, SKIP_RC)]
    print("\n[run_all] {}/{} passed; {} skipped (capability absent, declared); {} failed/errored;"
          " {} skipped-but-required".format(
              len(results) - len(failed) - len(skipped) - len(bad_skips), len(results),
              len(skipped), len(failed), len(bad_skips)), flush=True)
    if skipped:
        print("[run_all] SKIPPED (declared): " + ", ".join(n for n, _, _, _ in skipped), flush=True)
    if failed:
        print("[run_all] FAILURES: " + ", ".join(n for n, _, _, _ in failed), flush=True)
    if bad_skips:
        print("[run_all] REQUIRED-BUT-SKIPPED (smoke tier: install the missing capability, or "
              "pass --allow-skip NAME once something else runs it for real): "
              + ", ".join(n for n, _, _, _ in bad_skips), flush=True)
    # keep the exception list from rotting: an --allow-skip name that did NOT skip is either a
    # typo or a capability that is now installed, and either way the caller should drop it.
    selected = {n.split(".", 1)[-1] for n, _, _, _ in results}
    stale = (allow_skip & selected) - {n.split(".", 1)[-1] for n, rc, _, _ in results
                                       if rc == SKIP_RC}
    if stale:
        print("[run_all] NOTE: --allow-skip name(s) that ran instead of skipping: {} (the "
              "capability is present -- drop them from the caller)".format(
                  ", ".join(sorted(stale))), flush=True)
    return 0 if not (failed or bad_skips) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
