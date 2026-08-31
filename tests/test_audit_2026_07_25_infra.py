"""Gates for the 2026-07-25 v0.9.0 exhaustive audit's test-infrastructure findings.

  T-9  the never-called public surfaces this file adopts: `fiber_amp/concentration.py`'s two
       literature factories and `fiber_amp/calibration.py`'s `ion_from_cross_sections` +
       `EDFA_CBAND_TARGETS` (the bridge entry points -- `make_lumenairy_bor_solver`,
       `rcwa_stack_jones`, `pmm_stack_jones` -- are gated in tests/test_lumenairy_bridge.py
       next to the rest of the bridge; the blanket import/repr smoke lives in
       tests/test_api_surface.py).
  T-11 `numba_env`'s verdict taxonomy: a livelock and a hard crash force the workqueue fallback
       and are cached; a probe that merely FAILED (import error, any other exception, a launch
       failure, a silent clean exit) does neither.
  T-15 the non-oblique 2-D/3-D JAX FDTD kernels, which no test reached: the jax extra is
       installed on two CI legs and `solve_fdtd_2d/3d(..., backend="jax")` had no caller in
       `tests/`, so numpy-vs-jax equivalence was gated only by validations CI excludes.

Run: python -m pytest tests/test_audit_2026_07_25_infra.py -q
"""
import importlib.util
import json
import math
import subprocess

import numpy as np
import pytest

HAVE_JAX = importlib.util.find_spec("jax") is not None


# ==============================================================================================
# T-11: numba_env's negative verdict was indiscriminate (every child failure -> "the threading
# layer wedged", workqueue forced for the session AND all children, cached 24 h).
# ==============================================================================================

def _fake_completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["python", "-c", "..."], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


_TB = "Traceback (most recent call last):\n  File \"<string>\", line 5, in <module>\n"


@pytest.mark.parametrize("returncode,stdout,stderr,expect", [
    (0, "LAYER_OK:tbb\n", "", "default-ok:tbb"),
    (0, "quiet\n", "", "probe-error:no-verdict"),          # clean exit, no marker
    (1, "", _TB + "ModuleNotFoundError: No module named 'scipy'\n", "probe-error:import"),
    (1, "", _TB + "ValueError: guard tripped\n", "probe-error:exception"),
    (3221225477, "", "", "workqueue-fallback:crash"),      # 0xC0000005, Windows
    (-11, "", "std::terminate\n", "workqueue-fallback:crash"),   # SIGSEGV, POSIX
])
def test_t11_classify_separates_the_four_collapsed_outcomes(returncode, stdout, stderr, expect):
    """AUDIT T-11: the old code mapped every non-'LAYER_OK' outcome onto one verdict. Only a
    child that died WITHOUT a Python traceback is evidence that the threading layer crashed;
    a Python-level failure in the probe body says nothing about the layer."""
    from dynameta.numba_env import _classify
    verdict, why = _classify(returncode, stdout, stderr)
    assert verdict == expect
    if verdict != "default-ok:tbb":
        assert why, "a non-success verdict must carry a diagnostic"


def _isolated_env(monkeypatch, tmp_path, run_result):
    """Point numba_env at a throwaway cache, unset any inherited NUMBA_THREADING_LAYER (else the
    call short-circuits to 'explicit'), and stub the probe child with `run_result` (a
    CompletedProcess or an exception INSTANCE to raise)."""
    from dynameta import numba_env
    cache = tmp_path / "probe.json"
    monkeypatch.delenv("NUMBA_THREADING_LAYER", raising=False)
    monkeypatch.setattr(numba_env, "_cache_path", lambda: str(cache))

    def _run(*_a, **_kw):
        if isinstance(run_result, BaseException):
            raise run_result
        return run_result

    monkeypatch.setattr(numba_env.subprocess, "run", _run)
    return numba_env, cache


@pytest.mark.parametrize("result,expect", [
    (_fake_completed(1, "", _TB + "ImportError: cannot import name 'solve_fdtd_2d'\n"),
     "probe-error:import"),
    (_fake_completed(1, "", _TB + "TypeError: unexpected keyword 'chi2_m_V'\n"),
     "probe-error:exception"),
    (_fake_completed(0, "nothing to see\n", ""), "probe-error:no-verdict"),
    (FileNotFoundError("sys.executable is gone"), "probe-error:launch"),
])
def test_t11_probe_error_leaves_the_layer_alone_and_is_not_cached(monkeypatch, tmp_path,
                                                                  result, expect):
    """The behavioural half of T-11: a probe that could not be EVALUATED must not export
    NUMBA_THREADING_LAYER=workqueue for this process and all its children, and must not persist
    that degraded mode for 24 h (a source fix has to be re-probed on the next session)."""
    pytest.importorskip("numba")
    ne, cache = _isolated_env(monkeypatch, tmp_path, result)
    verdict = ne.ensure_working_threading_layer(windows_only=False, verbose=False)
    assert verdict == expect
    import os
    assert "NUMBA_THREADING_LAYER" not in os.environ, "an inconclusive probe serialised the kernels"
    assert not cache.exists(), "an inconclusive probe was cached"


@pytest.mark.parametrize("result,expect", [
    (subprocess.TimeoutExpired(cmd="python", timeout=1.0), "workqueue-fallback:timeout"),
    (_fake_completed(-1073741819, "", ""), "workqueue-fallback:crash"),
])
def test_t11_livelock_and_crash_still_force_workqueue_and_are_cached(monkeypatch, tmp_path,
                                                                     result, expect):
    """The two REAL failure modes keep the old behaviour (that is the point of the module), but
    now say which one fired."""
    pytest.importorskip("numba")
    ne, cache = _isolated_env(monkeypatch, tmp_path, result)
    verdict = ne.ensure_working_threading_layer(windows_only=False, verbose=False)
    assert verdict == expect
    import os
    assert os.environ.get("NUMBA_THREADING_LAYER") == "workqueue"
    assert json.loads(cache.read_text())["verdict"] == expect


def test_t11_success_is_cached_and_the_key_tracks_numba_and_tbb_versions(monkeypatch, tmp_path):
    """AUDIT T-11: the verdict was keyed on (numba version, tbb PRESENT), so a tbb upgrade could
    not clear a stale negative. The key now carries both VERSIONS (plus the probe source and the
    dynameta version), and a key mismatch re-probes rather than replaying the cached verdict."""
    pytest.importorskip("numba")
    ne, cache = _isolated_env(monkeypatch, tmp_path,
                              _fake_completed(0, "LAYER_OK:omp\n", ""))
    monkeypatch.setattr(ne, "_tbb_version", lambda: "2021.1.0")
    assert ne.ensure_working_threading_layer(windows_only=False, verbose=False) == "default-ok:omp"
    key_old = json.loads(cache.read_text())["key"]
    assert "2021.1.0" in key_old

    # a tbb UPGRADE must invalidate the cached verdict: the probe runs again (here it now reports
    # a crash) instead of the stale 'default-ok' being replayed
    monkeypatch.setattr(ne, "_tbb_version", lambda: "2022.2.0")
    monkeypatch.setattr(ne.subprocess, "run",
                        lambda *_a, **_kw: _fake_completed(3221225477, "", ""))
    assert ne.ensure_working_threading_layer(windows_only=False,
                                             verbose=False) == "workqueue-fallback:crash"
    assert json.loads(cache.read_text())["key"] != key_old


def test_t11_explicit_layer_still_wins_without_probing(monkeypatch, tmp_path):
    """Cost control unchanged: an explicit NUMBA_THREADING_LAYER short-circuits the probe."""
    pytest.importorskip("numba")
    ne, _cache = _isolated_env(monkeypatch, tmp_path, RuntimeError("the probe must not run"))
    monkeypatch.setenv("NUMBA_THREADING_LAYER", "tbb")
    assert ne.ensure_working_threading_layer(windows_only=False, verbose=False) == "explicit"


# ==============================================================================================
# T-9: fiber_amp/concentration.py + calibration.py -- public surfaces with no caller anywhere
# (no test, no validation, no example). Fast paths only: no amplifier solve runs here.
# ==============================================================================================

def test_t9_erbium_upconversion_levels_are_ordered_and_only_heavy_quenches():
    from dynameta.optics.fiber_amp import erbium_upconversion
    light, moderate, heavy = (erbium_upconversion(k) for k in ("light", "moderate", "heavy"))
    assert light.c_up_m3_s < moderate.c_up_m3_s < heavy.c_up_m3_s      # C_up rises with doping
    assert moderate.c_up_m3_s == erbium_upconversion().c_up_m3_s       # 'moderate' is the default
    assert light.pair_fraction == moderate.pair_fraction == 0.0        # pairs only at heavy doping
    assert 0.0 < heavy.pair_fraction < 0.1
    # every level is a REAL model (a no-op would be silently identical to concentration=None)
    for m in (light, moderate, heavy):
        assert not m.is_identity
        assert 1e-25 < m.c_up_m3_s < 1e-22                             # docs sec.6 order of magnitude
    # the Delevaque dark-pair split is a partition of the ion density
    n_t = 4.0e25
    assert heavy.active_density(n_t) + heavy.dark_density(n_t) == pytest.approx(n_t, rel=1e-15)
    assert heavy.dark_density(n_t) == pytest.approx(0.03 * n_t, rel=1e-12)


def test_t9_ytterbium_photodarkening_follows_the_equilibrium_law():
    # 2026-08-31 literature audit: the equilibrium loss is LINEAR in inversion (Jetschke 2007;
    # Jauregui AOP 2020); the legacy exponent 7 is the Koponen RATE law and under-predicts
    # equilibrium by 10^2-10^3 at amplifier inversions. The factory now takes a measured
    # equilibrium anchor (dB/m at the 0.46 benchmark inversion) instead of a raw 1/m scale.
    from dynameta.optics.fiber_amp import ytterbium_photodarkening
    m = ytterbium_photodarkening()
    assert m.pd_exponent == 1.0                                        # equilibrium default
    assert m.c_up_m3_s == 0.0 and m.pair_fraction == 0.0               # photodarkening ONLY
    assert not m.is_identity
    ln10_10 = math.log(10.0) / 10.0
    # the anchor round-trips: at nbar2_ref the loss equals the measured dB/m anchor
    assert float(m.photodarkening_loss_per_m(0.46)) == pytest.approx(0.58 * ln10_10, rel=1e-12)
    n = np.array([0.0, 0.25, 0.5, 1.0])
    loss = m.photodarkening_loss_per_m(n)
    assert loss.shape == n.shape                                       # shape-preserving (S3-38)
    assert np.allclose(loss, m.pd_loss_per_m * n, rtol=1e-13, atol=0.0)
    # linear law: dropping 0.46 -> 0.115 suppresses by exactly 4x, NOT by 4^7 ~ 1.6e4
    assert (float(m.photodarkening_loss_per_m(0.46))
            / float(m.photodarkening_loss_per_m(0.115))) == pytest.approx(4.0, rel=1e-12)
    # the exponent knob still exists for rate-law studies, anchored at the same benchmark
    steep = ytterbium_photodarkening(alpha_eq_dB_per_m=0.58, pd_exponent=7.0)
    assert float(steep.photodarkening_loss_per_m(0.46)) == pytest.approx(0.58 * ln10_10, rel=1e-12)
    assert float(steep.photodarkening_loss_per_m(0.115)) < 1e-3 * float(
        m.photodarkening_loss_per_m(0.115))


def test_t9_ion_from_cross_sections_round_trips_the_table_and_pins_mccumber():
    """`ion_from_cross_sections` is the measured-data entry point (`giles_calibrated_fiber` goes
    through it); it had no direct caller. The tables must interpolate to themselves at the nodes,
    the scalar metadata must reach the ion, and McCumber must reduce to sigma_a at the zero line
    -- the identity that proves zero_line_m was wired, not just stored."""
    from dynameta.optics.fiber_amp import ion_from_cross_sections
    lam = np.array([1.50e-6, 1.53e-6, 1.55e-6, 1.60e-6])
    sa = np.array([2.0e-25, 6.0e-25, 4.0e-25, 1.0e-25])
    se = np.array([1.0e-25, 5.5e-25, 5.0e-25, 2.0e-25])
    ion = ion_from_cross_sections("probe-Er", lam, sa, se, tau_s=10.0e-3, zero_line_m=1.53e-6)
    assert (ion.name, ion.host) == ("probe-Er", "measured")
    assert ion.tau_s == 10.0e-3 and ion.zero_line_m == 1.53e-6
    assert np.allclose(ion.sigma_a.sigma(lam), sa, rtol=1e-14, atol=0.0)   # AUDIT T-6
    assert np.allclose(ion.sigma_e.sigma(lam), se, rtol=1e-14, atol=0.0)
    assert float(ion.sigma_a.sigma(1.54e-6)) == pytest.approx(float(np.interp(1.54e-6, lam, sa)),
                                                              rel=1e-14)
    # McCumber: sigma_e(lam0) == sigma_a(lam0) at the zero-phonon line, for any temperature
    for T in (250.0, 300.0, 350.0):
        assert float(ion.sigma_e_mccumber(1.53e-6, T)) == pytest.approx(float(sa[1]), rel=1e-12)


def test_t9_edfa_cband_targets_is_the_calibration_report_default(monkeypatch):
    """`EDFA_CBAND_TARGETS` had no reader outside its own module. Gate the datasheet values AND
    the wiring: `calibration_report(amp)` with no targets must use it (and apply gain_tol_dB /
    the NF ceiling to it). The amplifier and the noise analysis are stubbed -- this is the
    fast path, the physics is gated by the fiber_amp suite."""
    from dynameta.optics.fiber_amp import calibration as cal
    from dynameta.optics.fiber_amp import noise as noise_mod
    t = cal.EDFA_CBAND_TARGETS
    assert set(t) == {"pump_nm", "signal_nm", "pump_power_mW", "signal_in_dBm",
                      "small_signal_gain_dB", "nf_dB_max"}
    assert t["pump_nm"] == 980.0 and t["signal_nm"] == 1550.0        # C-band, 980 pump
    assert t["nf_dB_max"] >= 3.0                                     # above the quantum limit

    class _StubAmp:
        def solve(self):
            return "result-sentinel"

    class _StubNoise:
        def __init__(self, gain_dB, nf_dB):
            self.gain_dB, self.nf_dB = gain_dB, nf_dB

    seen = {}

    def _fake_analyze(res, lam_s):
        seen["res"], seen["lam_s"] = res, lam_s
        return _StubNoise(seen["gain"], seen["nf"])

    monkeypatch.setattr(noise_mod, "analyze_noise", _fake_analyze)

    seen["gain"], seen["nf"] = t["small_signal_gain_dB"] - 1.0, t["nf_dB_max"] - 0.5
    rep = cal.calibration_report(_StubAmp())
    assert seen["res"] == "result-sentinel"
    assert seen["lam_s"] == pytest.approx(t["signal_nm"] * 1e-9, rel=1e-15)   # nm -> m
    assert rep.targets == t and rep.targets is not t                 # defensive copy
    assert rep.gain_ok and rep.nf_ok

    # ... and the tolerances actually discriminate
    seen["gain"], seen["nf"] = t["small_signal_gain_dB"] - 4.0, t["nf_dB_max"] + 0.1
    bad = cal.calibration_report(_StubAmp())
    assert not bad.gain_ok and not bad.nf_ok
    assert cal.calibration_report(_StubAmp(), gain_tol_dB=5.0).gain_ok      # tol is honoured


# ==============================================================================================
# T-15: numpy-vs-jax parity for the NON-oblique 2-D/3-D FDTD kernels (kernels2d_jax/kernels3d_jax)
# ==============================================================================================

needs_jax = pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
_BAND = dict(lambda_min_m=1.0e-6, lambda_max_m=1.4e-6, resolution=8, n_pad_wave=3.0)


@needs_jax
def test_t15_solve_fdtd_2d_jax_matches_numpy():
    """AUDIT T-15: the only four `backend="jax"` call sites in tests/ were the OBLIQUE solvers,
    so `kernels2d_jax` had no test coverage at all. MEASURED here (jax 0.11, x64 on):
    max|dR0| = 6.1e-16, max|dT0| = 1.1e-15 on R0 ~ 0.38 / T0 ~ 0.76 -- i.e. the two kernels agree
    to a few float64 ulp. Gated at 1e-12 absolute with rtol=0.0 (AUDIT T-6: an atol-only
    np.allclose would have been a 1e-5 RELATIVE gate)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_nd import solve_fdtd_2d
    layers = [FDTDLayer(thickness_m=200e-9, eps_inf=4.0)]
    kw = dict(period_x_m=600e-9, nx=4, **_BAND)
    a = solve_fdtd_2d(layers, backend="numpy", **kw)
    b = solve_fdtd_2d(layers, backend="jax", **kw)
    m = a.band
    assert np.max(np.abs(a.R0[m])) > 0.1 and np.max(np.abs(a.T0[m])) > 0.1   # non-trivial fixture
    assert np.allclose(np.asarray(b.R0)[m], a.R0[m], rtol=0.0, atol=1e-12)
    assert np.allclose(np.asarray(b.T0)[m], a.T0[m], rtol=0.0, atol=1e-12)


@needs_jax
def test_t15_solve_fdtd_3d_jax_matches_numpy():
    """The 3-D half of T-15 (`kernels3d_jax`). MEASURED: max|dR0| = 1.1e-15, max|dT0| = 1.3e-15
    on R0 ~ 0.41; gated at 1e-12 absolute, rtol=0.0. Kept small on purpose (nx=ny=3, npml=8) --
    ~6 s for the pair."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_nd import solve_fdtd_3d
    layers = [FDTDLayer(thickness_m=200e-9, eps_inf=4.0)]
    kw = dict(period_x_m=120e-9, period_y_m=120e-9, nx=3, ny=3, npml=8,
              lambda_min_m=1.0e-6, lambda_max_m=1.4e-6, resolution=6, n_pad_wave=2.0)
    a = solve_fdtd_3d(layers, backend="numpy", **kw)
    b = solve_fdtd_3d(layers, backend="jax", **kw)
    m = a.band
    assert np.max(np.abs(a.R0[m])) > 0.1 and np.max(np.abs(a.T0[m])) > 0.1
    assert np.allclose(np.asarray(b.R0)[m], a.R0[m], rtol=0.0, atol=1e-12)
    assert np.allclose(np.asarray(b.T0)[m], a.T0[m], rtol=0.0, atol=1e-12)
