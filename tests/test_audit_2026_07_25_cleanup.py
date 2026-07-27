"""Gates for the 2026-07-25 v0.9.0 audit's wave-5 cleanup findings (open-items campaign).

  X-8(b) `transient_optics.enz_reflector_stack` accepted a `lambda_m` it completely ignored
         (AST-verified): it returned a byte-identical stack at 1.31 um and 1.55 um while its
         signature promised a wavelength-dependent build. `lambda_m` is now READ -- it is
         validated on every call, and `eps_oxide`/`eps_mirror` accept a dispersion callable
         `f(lambda_m) -> complex` that is evaluated at it. The scalar defaults still produce a
         wavelength-independent stack, which is now a DOCUMENTED AND GATED property rather than
         an ignored argument. (X-8(a), `soa/qw_gain.noise_figure_db`'s ignored `nu_Hz`, was
         closed by the amplifier batch and is gated in tests/test_soa_qw.py.)

  X-11   `carriers/thermal_fem`'s cosmetic "promotion": three helpers shipped under two public
         names each and the two spellings had split by caller (18 in-package call sites on the
         underscored aliases vs 3 on the public names), with the package facade re-exporting
         both. Each name is now SINGLE-HOMED on the public spelling -- every in-package call site
         uses it -- with the underscored spellings surviving only as thin deprecated aliases
         bound to the SAME object, so the facade keeps both importable.

  T-13   the warnings-policy follow-ups this wave picked up: five blanket
         `warnings.simplefilter("ignore")` blocks in tests/ (which locally disable the repo's
         `filterwarnings = ["error"]` allow-list for whatever runs inside), and pyproject's
         TEMPORARY message-scoped exemption for solve_fem_sourced's lossy-substrate `p_down`
         advisory, which is deleted now that the warn site carries FEMDiagnosticWarning.

Run: python -m pytest tests/test_audit_2026_07_25_cleanup.py -q
"""
import ast
import pathlib
import warnings

import numpy as np
import pytest

from dynameta.materials import DrudeOptical, M_E
from dynameta.optics.tmm_reference import layered_rta
from dynameta.transient_optics import enz_reflector_stack, optical_transient_response

_REPO = pathlib.Path(__file__).resolve().parent.parent


# ==============================================================================================
# X-8(b): enz_reflector_stack's lambda_m
# ==============================================================================================

def test_x8_lambda_m_is_not_an_ignored_argument():
    """The AST check the auditor ran, re-run as a gate: `lambda_m` must appear in the body of
    `enz_reflector_stack`, not merely in its signature and docstring."""
    src = (_REPO / "dynameta" / "transient_optics.py").read_text(encoding="cp1252")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "enz_reflector_stack")
    assert "lambda_m" in [a.arg for a in fn.args.args]
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) else fn.body       # drop the docstring
    used = {n.id for b in body for n in ast.walk(b) if isinstance(n, ast.Name)}
    assert "lambda_m" in used, "lambda_m is accepted and never read (audit X-8)"


def test_x8_scalar_defaults_are_wavelength_independent_by_construction():
    """The incumbent behaviour is PRESERVED and now documented: with scalar eps_oxide/eps_mirror
    the stack is identical at 1.31 and 1.55 um. That is the property the auditor measured; it is
    fine, as long as it is a stated contract rather than a silently dropped argument."""
    eps = np.array([-2.0 + 0.5j, 1.0 + 0.3j, 2.5 + 0.2j])
    a = enz_reflector_stack(eps, 1.31e-6)
    b = enz_reflector_stack(eps, 1.55e-6)
    assert a.n_sub == b.n_sub
    assert [s.eps for s in a.slabs] == [s.eps for s in b.slabs]
    assert [s.thickness_m for s in a.slabs] == [s.thickness_m for s in b.slabs]


def test_x8_dispersion_callables_are_evaluated_at_lambda_m():
    """... and when a dispersion is supplied, lambda_m DRIVES it: the callable sees exactly the
    wavelength passed in, and the two builds now differ."""
    seen = []

    def oxide(lam):
        seen.append(lam)
        return 9.0 + 0.4 * (lam / 1.55e-6)

    def mirror(lam):
        seen.append(lam)
        return -120.0 * (lam / 1.55e-6) ** 2 + 3.0j

    eps = np.array([-1.0 + 0.4j, 2.0 + 0.2j])
    a = enz_reflector_stack(eps, 1.31e-6, eps_oxide=oxide, eps_mirror=mirror)
    b = enz_reflector_stack(eps, 1.55e-6, eps_oxide=oxide, eps_mirror=mirror)
    assert seen == [1.31e-6, 1.31e-6, 1.55e-6, 1.55e-6]
    assert a.slabs[-1].eps != b.slabs[-1].eps                 # oxide moved
    assert a.n_sub != b.n_sub                                 # mirror moved
    # the ITO sublayers are the caller's eps_ito and are untouched by the dispersion hook
    assert [s.eps for s in a.slabs[:-1]] == [complex(e) for e in eps]
    # and it is optically visible, not just structurally different
    assert abs(layered_rta(a, 1.31e-6)[0] - layered_rta(b, 1.55e-6)[0]) > 1e-6


def test_x8_constant_callable_reproduces_the_scalar_path_exactly():
    eps = np.array([-1.0 + 0.4j, 2.0 + 0.2j])
    ref = enz_reflector_stack(eps, 1.55e-6)
    got = enz_reflector_stack(eps, 1.55e-6, eps_oxide=lambda _l: 9.0,
                              eps_mirror=lambda _l: -120.0 + 3.0j)
    assert got.n_sub == ref.n_sub
    assert [s.eps for s in got.slabs] == [s.eps for s in ref.slabs]


def test_x8_lambda_m_is_validated():
    eps = np.array([1.0 + 0.0j])
    for bad in (0.0, -1.55e-6, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lambda_m must be a finite wavelength"):
            enz_reflector_stack(eps, bad)
    with pytest.raises(ValueError, match="non-finite permittivity"):
        enz_reflector_stack(eps, 1.55e-6, eps_mirror=lambda _l: complex("nan"))


def test_x8_transient_loop_is_byte_identical_on_the_scalar_default():
    """The default build_stack path through optical_transient_response is unchanged: honouring
    lambda_m must not perturb the shipped waveform."""
    dm = DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * M_E, gamma_rad_s=1.0e14)
    t = np.linspace(0.0, 3e-11, 12)
    _t, R, T, eps_front = optical_transient_response(t, lambda _ti: 8e26, 1.55e-6, drude_model=dm)
    ref = layered_rta(enz_reflector_stack(complex(dm.eps(1.55e-6, n_m3=8e26)), 1.55e-6), 1.55e-6)
    assert R[0] == ref[0] and T[0] == ref[1]                  # exact, not approx
    assert np.all(R == R[0]) and np.all(T == T[0])
    assert np.all(eps_front == eps_front[0])


# ==============================================================================================
# X-11: thermal_fem's double-named helpers
# ==============================================================================================

_X11_DOUBLED = ("mean_T_per_layer", "add_load_terms", "build_thermal_forms")
_X11_MODULES = ("common.py", "steady.py", "transient.py", "twotemp.py", "kirchhoff.py",
                "__init__.py")


def test_x11_no_in_package_call_site_uses_a_private_alias():
    """Source-level (no NGSolve needed, so this runs on every CI leg): the underscored spellings
    may appear ONLY where they are BOUND -- the one-line aliases in common.py and the facade's
    re-export/__all__ in __init__.py -- never as a call site or an import in a solver module."""
    pkg = _REPO / "dynameta" / "carriers" / "thermal_fem"
    offenders = []
    for name in ("steady.py", "transient.py", "twotemp.py", "kirchhoff.py"):
        src = (pkg / name).read_text(encoding="cp1252")
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            for doubled in _X11_DOUBLED:
                if "_" + doubled in code:
                    offenders.append("{}:{}: {}".format(name, i, line.strip()))
    assert not offenders, "private-alias call sites resurfaced (audit X-11):\n" + "\n".join(offenders)
    # common.py BINDS each alias exactly once (one thin `_x = x` line) and calls none of them
    common = (pkg / "common.py").read_text(encoding="cp1252")
    tree = ast.parse(common)
    bound = [t.id for n in tree.body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name)]
    for doubled in _X11_DOUBLED:
        assert bound.count("_" + doubled) == 1, doubled
        calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Name) and c.func.id == "_" + doubled]
        assert not calls, doubled
    # ... and common.py no longer re-declares the mesh scale that fem_mesh already owns
    assert "_S" not in bound


def test_x11_both_spellings_are_the_same_object_on_the_facade():
    pytest.importorskip("ngsolve")
    import dynameta.carriers.thermal_fem as tf
    for doubled in _X11_DOUBLED:
        pub = getattr(tf, doubled)
        priv = getattr(tf, "_" + doubled)
        assert priv is pub, doubled                            # one home, two names
        assert doubled in tf.__all__ and "_" + doubled in tf.__all__
    assert tf._S is tf.MESH_SCALE


def test_x11_the_migrated_solvers_still_solve():
    """The migration is a rename, so the physics must be untouched: the steady FEM solve still
    reduces to the analytic series-resistance profile."""
    pytest.importorskip("ngsolve")
    from dynameta.carriers.thermal import steady_layered_temperature
    from dynameta.carriers.thermal_fem import ThermalLayer, solve_thermal_fem
    lay = [ThermalLayer("si", 200e-9, 140.0), ThermalLayer("ox", 100e-9, 1.4)]
    flux, T_sink = 1.0e7, 300.0
    res = solve_thermal_fem(lay, period_x_m=80e-9, period_y_m=80e-9, flux_W_m2=flux,
                            T_sink_K=T_sink, maxh_m=40e-9)
    ref = steady_layered_temperature([L.k_thermal for L in lay], [L.thickness_m for L in lay],
                                     flux_W_m2=flux, T_sink_K=T_sink)
    got = res.mean_T_per_layer()
    assert len(got) == len(lay)
    assert np.allclose(got, ref, rtol=2e-2), (got, ref)


# ==============================================================================================
# Warning-policy hygiene (test-infra handoff): the blanket suppressions this wave replaced
# ==============================================================================================

def test_no_blanket_warning_suppression_in_the_handed_off_files():
    """tests/test_fdtd_mixing.py (4 sites) and tests/test_fem_drivers.py (1) carried
    `warnings.simplefilter("ignore")`, which blindfolds the repo's `filterwarnings = ["error"]`
    policy for whatever runs inside. Two were narrowed to a message-scoped filter; three were
    MEASURED to suppress nothing at all and deleted."""
    for name in ("test_fdtd_mixing.py", "test_fem_drivers.py"):
        src = (_REPO / "tests" / name).read_text(encoding="cp1252")
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            assert 'simplefilter("ignore")' not in code, "{}:{} {}".format(name, i, line.strip())


def test_p_down_advisory_is_categorised_and_its_pyproject_exemption_is_gone():
    """pyproject carried a TEMPORARY message-scoped exemption for solve_fem_sourced's
    lossy-substrate `p_down is undefined` advisory, with its own FOLLOW-UP note. The advisory now
    carries FEMDiagnosticWarning like the solver's other regime advisories (MEASURED on the
    test_shg_fem sourced solve: one FEMDiagnosticWarning, p_down=None), so conftest's conditional
    `ignore::FEMDiagnosticWarning` covers it and the special case is deleted. Source-level so it
    runs on the CI legs without the [solvers] extra."""
    proj = (_REPO / "pyproject.toml").read_text(encoding="cp1252")
    for i, line in enumerate(proj.splitlines(), 1):
        code = line.split("#", 1)[0]
        assert "p_down is undefined" not in code, "exemption resurfaced at pyproject.toml:%d" % i
    src = (_REPO / "dynameta" / "optics" / "solver.py").read_text(encoding="cp1252")
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "warn"):
            continue
        if "p_down is undefined" not in ast.dump(node.args[0]):
            continue
        hits.append(node)
        assert len(node.args) >= 2 and isinstance(node.args[1], ast.Name), \
            "the p_down advisory must name a warning category"
        assert node.args[1].id == "FEMDiagnosticWarning", node.args[1].id
    assert len(hits) == 1, "expected exactly one p_down warn site, found %d" % len(hits)


def test_message_scoped_filter_still_lets_an_unrelated_warning_through():
    """The point of the narrowing, proved directly: the surviving filter suppresses the
    broadband-pump advisory and NOTHING else -- an unrelated warning raised inside the same
    `catch_warnings` block still reaches the session's `error` policy."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="mixing_spectrum: BROADBAND pump")
        warnings.warn("mixing_spectrum: BROADBAND pump(s) -- measured pump spectral std ...",
                      UserWarning)                             # suppressed
        with pytest.raises(UserWarning):
            warnings.warn("something else entirely", UserWarning)     # propagates -> error
