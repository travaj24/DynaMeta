"""Gates for the opt-in EXTENDED-PRECISION DEVSIM solves (dynameta.carriers.extended_precision,
DynaMeta v0.14.0, workstream D2b).

EVERY ORACLE IS INDEPENDENT OF THE CODE UNDER TEST: the Shockley ideal-diode current (finite neutral
widths) plus the Sah-Noyce-Shockley depletion-SRH integral on the depletion-approximation potential,
evaluated with scipy.constants + scipy.integrate.quad in validation/extended_precision.py -- never a
number DEVSIM or DynaMeta computed. Each accuracy gate is paired with its DOUBLE-precision twin, which
must FAIL the same oracle (so the gate is load-bearing):

  A  GaN-like p-n junction (n_i = 1e-4 m^-3), forward 1.0 / 1.5 V: extended J/oracle - 1 = +9.0e-4 /
     +7.4e-4 and the edge currents conserved to 2.1e-5 / 1.0e-9; double gives -1.3e-8 / +2.8e-8 A/m^2
     (oracle 8.2e-18 / 1.5e-13: 1.6e9x / 1.9e5x off, edge spread 1e4 / 5e3 x |J|).
  B  p+n junction (N_A 1e25 / N_D 1e22, Si-like n_i 1e16) at zero bias, absolute tolerance 1e8:
     double cannot converge (the p+ continuity update floors at ~1e9); extended converges and
     dI/dV(+-0.1 mV) = G0 to -1.2e-6. The ssac Re Y(1 Hz) is NOT fixed (strict xfail below).
  C  default off: no extended_* parameter is ever written on the default path (the cross-commit
     sha256 fingerprints are in docs/audit/2026-11-03-devsim-extended-precision.md).
  D  no leakage: default -> extended -> default in one process; the default fingerprints match
     byte for byte and extended_solver is off afterwards, also after a convergence failure.

Fixtures are tiny 1-D devices (317 / 443 / 734 nodes) plus a 2-D build and a stubbed 3-D build; the
file runs in ~75 s at -n 2 (box A). devsim self-skips; the 3-D switch test also needs gmsh. On the
pre-D2b code every test errors at collection (no dynameta.carriers.extended_precision).
"""
import functools

import pytest

devsim = pytest.importorskip("devsim")

import validation.extended_precision as E                     # noqa: E402
from dynameta.carriers import extended_precision as XP        # noqa: E402
from dynameta.warning_types import AdvisoryWarning            # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    E.fresh()
    yield
    E.fresh()


def test_devsim_build_has_extended_precision():
    """The pip wheel (2.11.2) reports info['extended_precision'] = True; every gate below needs it."""
    assert XP.available()


# =====================================================================================================
# GATE A: wide-bandgap forward current
# =====================================================================================================

@functools.lru_cache(maxsize=None)
def _gan(extended):
    return E.gate_a_wide_bandgap(extended)


def test_gate_a_wide_bandgap_extended_matches_oracle():
    """GaN-like diode: the forward current against the SNS + Shockley oracle (measured +9.0e-4 at
    1.0 V, +7.4e-4 at 1.5 V on the 1 nm junction mesh; the residual is the depletion approximation
    and the mesh) and conserved along the device (edge spread 2.1e-5, 1.0e-9 of |J|)."""
    for V, (J, Jo, spread, _t) in _gan(True).items():
        assert abs(J / Jo - 1.0) < 3e-3, (V, J, Jo)
        assert spread < 1e-3, (V, spread)


def test_gate_a_wide_bandgap_double_fails_the_same_oracle():
    """The double twin of gate A: the current is round-off of the ~1e8 A/m^2 majority drift and
    diffusion terms -- measured -1.3e-8 and +2.8e-8 A/m^2 against 8.2e-18 and 1.5e-13, with an edge
    spread 1e4x and 5e3x the mean. Required: wrong by > 100x AND not conserved."""
    for V, (J, Jo, spread, _t) in _gan(False).items():
        assert abs(J / Jo - 1.0) > 100.0, (V, J, Jo)
        assert spread > 1.0, (V, spread)


# =====================================================================================================
# GATE B: heavily doped p+n junction
# =====================================================================================================

@functools.lru_cache(maxsize=None)
def _ppn(extended):
    return E.gate_b_heavy_doping(extended, ae=1e8)


def test_gate_b_heavy_doping_extended_conductance():
    """Extended converges at the absolute tolerance 1e8 and the DC small-signal conductance equals
    the oracle G0 = 2.5823e-5 S/m^2 (depletion SRH 2.52e-5 + diffusion 6.0e-7): measured -1.2e-6."""
    b = _ppn(True)
    assert b["converged"]
    assert abs(b["G_dc"] / b["G_oracle"] - 1.0) < 1e-4, b


def test_gate_b_heavy_doping_double_cannot_converge():
    """The double twin: the p+ (1e25 m^-3) continuity update floors at ~1e25 * eps ~ 1e9 > 1e8, so the
    equilibrium Newton raises Convergence failure. (Loosened to 1e13 double DOES converge and gives
    the same G_dc to 1e-12 -- the DC conductance itself is not precision-limited; recorded in the
    audit note.)"""
    assert not _ppn(False)["converged"]


@pytest.mark.xfail(strict=True, reason="DEVSIM factorizes and solves the ssac system in DOUBLE even "
                   "with extended_solver (float128 preconditioners cast to double): Re Y(1 Hz) = "
                   "+5.995e-5 vs G0 = 2.582e-5 on this diode (driven at the n end; -0.0727 driven "
                   "at the p+ end). A strict xfail, so a DEVSIM that fixes it flags this note.")
def test_gate_b_ssac_reaches_oracle_in_extended():
    b = _ppn(True)
    assert abs(b["ReY_1Hz"] / b["G_oracle"] - 1.0) < 0.1, b


# =====================================================================================================
# GATES C / D: default off, no leakage
# =====================================================================================================

def test_gate_c_default_path_never_writes_extended_parameters(monkeypatch):
    """Every DynaMeta solve path (solve_dc, ssac_admittance, noise_spectrum, transient_step) on a
    default device: no extended_* parameter is ever written, so DEVSIM runs exactly the double code it
    ran before D2b. (Cross-commit sha256 of the same run: identical on all 7 fingerprints, audit note.)"""
    writes = []
    real = devsim.set_parameter

    def spy(*a, **kw):
        writes.append(kw.get("name"))
        return real(*a, **kw)

    monkeypatch.setattr(devsim, "set_parameter", spy)
    fp = E.fingerprint_1d(False)
    assert set(fp) == {"dc0", "dc_fwd", "ssac", "noise", "transient"}
    assert writes and not [w for w in writes if str(w).startswith("extended")]


def test_gate_d_no_leakage_between_devices():
    """default -> extended -> default in one process. The extended run really differs (all five
    fingerprints move) and the second default run is byte-identical to the first; the global
    extended_solver is off afterwards."""
    h1, h2, h3, flag = E.gate_d_leakage()
    assert h1 == h3
    assert all(h2[k] != h1[k] for k in h1), (h1, h2)
    assert not flag


# =====================================================================================================
# small-signal warning (coordinator decision 2026-11-03)
# =====================================================================================================

def test_small_signal_warning_fires_for_ssac_and_noise_only():
    """On an opted-in device the 'ac' and 'noise' solves warn (one per frequency under 'always': two
    each), while the DC build, the DC ramp and the transient do not."""
    E.fingerprint_1d(True)
    log = list(E.LAST_SMALL_SIGNAL_WARNINGS)
    assert log.count("ssac") == 2 and log.count("noise") == 2, log
    assert not {"dc0", "dc_fwd", "transient"} & set(log), log


def test_small_signal_warning_never_fires_on_default_devices():
    """The same five paths on a default device: no small-signal warning at all."""
    E.fingerprint_1d(False)
    assert E.LAST_SMALL_SIGNAL_WARNINGS == []


def test_small_signal_warning_text_and_once_per_call_site():
    """The message names the double factorization and both remedies; under the DEFAULT warnings
    filter a 3-frequency ssac sweep (one call site) shows it once, attributed to ac_analysis."""
    import warnings
    from dynameta.carriers.ac_analysis import ssac_admittance
    lines, dop = E._si_lines()
    E.diode_1d("ws", lines, dop, extended=True, circuit=True, n_i=E.SI["n_i"], tau=E.SI["tau"],
               eps_r=E.SI["eps_r"], mu_n=E.SI["mu_n"], mu_p=E.SI["mu_p"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        with E.quiet():
            ssac_admittance([1.0, 1e3, 1e6])
    hits = [w for w in caught if issubclass(w.category, AdvisoryWarning)]
    assert len(hits) == 1, [str(w.message) for w in caught]
    for word in ("DOUBLE", "DC dI/dV", "LIGHTLY doped"):
        assert word in str(hits[0].message)
    assert hits[0].filename.endswith("ac_analysis.py")


def test_gate_b_extended_ssac_warns_and_dc_does_not():
    """Gate B's extended run: the ssac warned, the DC conductance (+-0.1 mV solves) did not."""
    assert _ppn(True)["small_signal_warnings"] == ["ssac"]


def test_gate_d_solver_flag_restored_after_failure():
    """A convergence failure inside the extended solve must not leave extended_solver on."""
    E.gan_diode(True)
    devsim.set_parameter(device="gan", name="lo_bias", value=2.5)       # a 2.5 V jump in 1 iteration
    with pytest.raises(devsim.error):
        with E.quiet():
            XP.solve(type="dc", solver_type="direct", absolute_error=1e-30, relative_error=1e-30,
                     maximum_iterations=1)
    assert not E.solver_flag()


def test_extended_solver_context_restores_previous_value():
    """Re-entrant: an outer True survives an inner block; the final state is the caller's."""
    devsim.set_parameter(name="extended_solver", value=True)
    try:
        with XP.extended_solver():
            assert devsim.get_parameter(name="extended_solver")
        assert devsim.get_parameter(name="extended_solver")
    finally:
        devsim.set_parameter(name="extended_solver", value=False)
    with XP.extended_solver():
        assert devsim.get_parameter(name="extended_solver")
    assert not devsim.get_parameter(name="extended_solver")


def test_opt_in_dies_with_the_device():
    """The opt-in is a DEVICE parameter: delete_device clears it, so a later default device of the
    SAME name is not extended (DEVSIM 2.1: parameters set on a device are cleared with it)."""
    lines, dop = E._si_lines()
    kw = dict(n_i=E.SI["n_i"], tau=E.SI["tau"], eps_r=E.SI["eps_r"], mu_n=E.SI["mu_n"],
              mu_p=E.SI["mu_p"])
    E.diode_1d("same", lines, dop, extended=True, **kw)
    assert XP.is_enabled("same") and XP.any_enabled()
    E.diode_1d("same", lines, dop, extended=False, **kw)       # fresh() deletes, then rebuilds
    assert not XP.is_enabled("same") and not XP.any_enabled()


def test_mixed_live_devices_warn():
    """DEVSIM solves every live device in one Newton: an opted-in device next to a default one
    moves the default one's answer, so XP.solve says so."""
    lines, dop = E._si_lines()
    E.diode_1d("mx", lines, dop, extended=True, n_i=E.SI["n_i"], tau=E.SI["tau"],
               eps_r=E.SI["eps_r"], mu_n=E.SI["mu_n"], mu_p=E.SI["mu_p"])
    devsim.create_device(mesh="mx_m", device="mx_default")     # no physics: nothing to solve there
    with pytest.warns(AdvisoryWarning, match="mx_default"):
        with E.quiet():
            XP.solve(type="dc", solver_type="direct", absolute_error=1e8, relative_error=1e-10,
                     maximum_iterations=40)


def test_enable_refuses_late_or_unknown_devices():
    lines, dop = E._si_lines()
    E.diode_1d("late", lines, dop, extended=False, n_i=E.SI["n_i"], tau=E.SI["tau"],
               eps_r=E.SI["eps_r"], mu_n=E.SI["mu_n"], mu_p=E.SI["mu_p"])
    with pytest.raises(RuntimeError, match="after physics was installed"):
        XP.enable("late")
    with pytest.raises(ValueError, match="no device"):
        XP.enable("no_such_device")


# =====================================================================================================
# builder switches
# =====================================================================================================

def test_layered_builder_switch():
    """LayeredDevsimBuilder(extended_precision=True) opts the device in at build (default False)."""
    from validation._reference_device import build_reference_modulator
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    d = build_reference_modulator("drift_diffusion")
    assert LayeredDevsimBuilder(d, mesh_name="lb_m", device_name="lb_d").extended_precision is False
    b = LayeredDevsimBuilder(d, mesh_name="lb_m", device_name="lb_d", extended_precision=True)
    with E.quiet():
        b.build_device()
    try:
        assert XP.is_enabled("lb_d")
        assert all(devsim.get_parameter(device="lb_d", name=p) for p in XP.MODEL_PARAMETERS)
    finally:
        b.teardown()
    assert not XP.any_enabled()


def test_stacked3d_spec_switch(monkeypatch, tmp_path):
    """Stacked3DSpec.extended_precision (default False) is carried by from_design and opts the 3-D
    device in at build_device (region physics stubbed: the FD-closure setup of a real 3-D region costs
    minutes of DEVSIM symbolic differentiation; the stub creates the Potential solution it would)."""
    pytest.importorskip("gmsh")
    from validation._reference_device import build_reference_modulator
    from dynameta.carriers import devsim_3d as D3
    d = build_reference_modulator("equilibrium")
    assert D3.Stacked3DSpec.from_design(d).extended_precision is False
    spec = D3.Stacked3DSpec.from_design(d, extended_precision=True)
    assert spec.extended_precision is True

    seen = {}

    def _stub(device, region, *, n_bg_m3, eps_static, dos_mass_kg):
        seen["enabled_before_physics"] = XP.is_enabled(device)
        D3.PE.setup_dielectric_region(device, region, eps_static)

    monkeypatch.setattr(D3.PE, "setup_semiconductor_region", _stub)
    solver = D3.Devsim3DEquilibrium(spec, device_name="x3d", mesh_name="x3m",
                                    msh_path=str(tmp_path / "x3.msh"))
    try:
        with E.quiet():
            solver.build_device()
        assert seen["enabled_before_physics"] is True
    finally:
        from dynameta.carriers import eq_registry as _R
        _R.clear("x3d")
        solver.teardown()
    assert not XP.any_enabled()


def test_no_extended_parameter_left_after_module_runs():
    """Belt and braces: after this file's devices are gone, nothing is opted in and the global
    solver flag is off (None = never set in this process is fine too)."""
    assert not XP.any_enabled()
    assert not E.solver_flag()
