"""Smoke + reduces-to-analytic coverage for the NGSolve FEM field drivers (carriers.thermal_fem,
carriers.electrostatics_fem) -- previously exercised only by validation/ (no pytest). Skipped when
ngsolve is absent (the numpy-only CI leg); the rigorous oracles live in validation/thermal_fem.py +
validation/electrostatics_fem.py."""
import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.carriers.thermal_fem import (ThermalLayer, solve_thermal_fem,
                                           solve_thermal_transient_fem)
from dynameta.carriers.electrostatics_fem import ElectrostaticLayer, solve_electrostatics_fem


def test_thermal_fem_single_layer_series_resistance():
    # one conducting layer, bottom sink + top inflow flux, no source: T is linear with
    # dT = flux*L/k across the layer, so the volume-mean rise is flux*L/(2k).
    L, k, flux, Tsink = 100e-9, 1.5, 1.0e8, 300.0
    res = solve_thermal_fem([ThermalLayer("slab", L, k)], period_x_m=60e-9, period_y_m=60e-9,
                            flux_W_m2=flux, T_sink_K=Tsink, maxh_m=30e-9, order=2)
    mean_rise = float(res.mean_T_per_layer()[0]) - Tsink
    assert abs(mean_rise - flux * L / (2.0 * k)) / (flux * L / (2.0 * k)) < 0.05   # ~3.33 K


def test_thermal_transient_reaches_steady():
    # coarse single slab, uniform Joule, uniform IC=T_sink: a few large backward-Euler steps must
    # converge to the steady uniform-Joule profile T_mean = T_sink + Q L^2/(3k) within ~5%.
    L, k, Q, Tsink = 100e-9, 10.0, 5.0e15, 300.0
    slab = [ThermalLayer("slab", L, k, rho_kg_m3=7140.0, Cp_J_kgK=340.0)]   # ITO-like rho*Cp
    tr = solve_thermal_transient_fem(slab, period_x_m=60e-9, period_y_m=60e-9, t_end_s=2e-7,
                                     dt_s=1e-8, flux_W_m2=0.0, T_sink_K=Tsink, joule_W_m3=Q,
                                     maxh_m=30e-9, order=2)
    t_steady = Tsink + Q * L ** 2 / (3.0 * k)
    mean_final = float(tr.mean_T_per_layer()[0])
    assert abs(mean_final - t_steady) / abs(t_steady - Tsink) < 0.05
    # monotone rise (theta=1, no overshoot) + sink pinned at the final field
    assert float(np.min(np.diff(tr.mean_T_per_layer_t[:, 0]))) > -1e-6
    assert abs(tr.T_at(30e-9, 30e-9, 0.0) - Tsink) < 1e-6


def test_thermal_transient_requires_rho_cp():
    # rho/Cp default 0 -> the transient cannot run (would be a singular mass matrix); must raise.
    slab = [ThermalLayer("slab", 100e-9, 10.0)]                              # no rho/Cp
    with pytest.raises(ValueError):
        solve_thermal_transient_fem(slab, period_x_m=60e-9, period_y_m=60e-9, t_end_s=1e-7,
                                    dt_s=1e-8, flux_W_m2=1e8)


def test_electrostatics_fem_series_capacitor_field():
    # two dielectric layers in series: the displacement D = eps0 eps_r E_z is continuous, so
    # eps1*E1 == eps2*E2, and the field drops sum to the applied voltage.
    e1, e2, t = 4.0, 16.0, 50e-9
    V = 1.0
    res = solve_electrostatics_fem([ElectrostaticLayer("d1", t, e1), ElectrostaticLayer("d2", t, e2)],
                                   V, period_x_m=60e-9, period_y_m=60e-9, maxh_m=30e-9, order=2)
    Ez = res.mean_Ez_per_layer()
    D1, D2 = e1 * Ez[0], e2 * Ez[1]
    assert abs(D1 - D2) / abs(D1) < 0.03                              # D continuous across the interface
    assert abs(abs(Ez[0]) * t + abs(Ez[1]) * t - V) / V < 0.03        # series fields sum to applied V


def test_split_gate_tddb_stress_statistic():
    # audit C4-9: a +/-V split gate has signed layer-mean Ez ~ 0 (the pre-fix TDDB
    # adapter reported ~zero stress -> exponentially overstated t_BD, silently); the
    # sign-robust mean|Ez| statistic must see the ~3 MV/cm field
    ng = pytest.importorskip("ngsolve")
    from types import SimpleNamespace
    from dynameta.carriers.electrostatics_fem import (ElectrostaticLayer,
                                                      solve_electrostatics_fem)
    from dynameta.carriers.fem_mesh import _S
    from dynameta.reliability.tddb import oxide_stress_from_electrothermal
    lay = [ElectrostaticLayer("hfo2", 10e-9, 18.0)]
    P = 300e-9
    vcf = ng.IfPos(ng.x / (_S * P) - 0.5, 3.0, -3.0)
    # AUDIT T-13 follow-up: this solve carried a blanket simplefilter("ignore") with no stated
    # cause. MEASURED under simplefilter("always"): solve_electrostatics_fem with a split-gate
    # top_voltage_cf emits NO warning, so the suppression was vestigial and blindfolded the only
    # FEM electrostatics call in this file. Removed; warnings here now hit the `error` policy.
    r = solve_electrostatics_fem(lay, 3.0, period_x_m=P, period_y_m=P, top_voltage_cf=vcf)
    am = r.mean_absEz_per_layer()[0]
    assert am > 1e8                                           # ~3 MV/cm seen
    assert abs(r.mean_Ez_per_layer()[0]) < 0.05 * am          # the signed mean cancels
    et = SimpleNamespace(layers=lay, E_result=r, T_per_layer=[300.0])
    ez, T = oxide_stress_from_electrothermal(et, "hfo2")
    assert ez == pytest.approx(am)                            # adapter uses the C4-9 stat


def test_c9_kt_transient_steady_limit_refuses_the_constant_k_closed_form():
    """audit C-9: `ThermalTransientResult.steady_limit_T` evaluates the series-resistance closed
    form on the CONSTANT `L.k_thermal`. The k(T) transient assembles its stiffness from per-element
    k_of_T and reads `L.k_thermal` only for the `> 0` validation, so on that path the property used
    to hand back the t->infinity limit of a DIFFERENT material -- a plausible-looking array with no
    signal that it was wrong. Here k_thermal=10 is the placeholder and k(T)=1 is what the run uses:
    the legacy answer understates the steady rise 10x. It must now return None."""
    from dynameta.carriers.thermal import steady_layered_temperature
    from dynameta.carriers.thermal_fem import solve_thermal_transient_kt_fem
    L, flux, Tsink = 100e-9, 1.0e8, 300.0
    lay = [ThermalLayer("slab", L, 10.0, rho_kg_m3=7140.0, Cp_J_kgK=340.0)]   # k_thermal = PLACEHOLDER
    tr = solve_thermal_transient_kt_fem(lay, lambda T: 1.0 + 0.0 * np.asarray(T),
                                        period_x_m=60e-9, period_y_m=60e-9, t_end_s=2e-7,
                                        dt_s=2e-8, flux_W_m2=flux, T_sink_K=Tsink,
                                        maxh_m=30e-9, order=2)
    assert tr.steady_limit_T is None                                   # the C-9 fix
    # ... and the value it used to return really was the wrong material's: the run converges to the
    # k(T)=1 series limit (305 K), while the pinned-L.k_thermal closed form says 300.5 K -- the
    # steady RISE understated by exactly k_thermal/k(T) = 10x, silently.
    legacy = float(steady_layered_temperature([lay[0].k_thermal], [L], flux, Tsink)[0])
    truth = float(steady_layered_temperature([1.0], [L], flux, Tsink)[0])   # the k(T) the run used
    mean_final = float(tr.mean_T_per_layer()[0])
    assert abs(mean_final - truth) < 0.05 * (truth - Tsink)
    assert (truth - Tsink) / (legacy - Tsink) == pytest.approx(10.0, rel=1e-9)
    assert abs(mean_final - legacy) > 0.8 * (truth - Tsink)
    # an INSULATED bottom has no steady limit at all (pure Neumann: the box stores the energy)
    tr_ins = solve_thermal_transient_kt_fem(lay, lambda T: 1.0 + 0.0 * np.asarray(T),
                                            period_x_m=60e-9, period_y_m=60e-9, t_end_s=4e-8,
                                            dt_s=2e-8, flux_W_m2=flux, T_sink_K=Tsink,
                                            maxh_m=30e-9, order=2, bottom_bc="insulated")
    assert tr_ins.steady_limit_T is None


def test_c9_constant_k_transient_still_reports_its_steady_limit():
    """The C-9 fields default to (None, 'sink'), so the CONSTANT-k solver -- whose stiffness really
    is built from L.k_thermal with a Dirichlet sink -- keeps returning the closed form unchanged."""
    from dynameta.carriers.thermal import steady_layered_temperature
    L, k, flux, Tsink = 100e-9, 10.0, 1.0e8, 300.0
    lay = [ThermalLayer("slab", L, k, rho_kg_m3=7140.0, Cp_J_kgK=340.0)]
    tr = solve_thermal_transient_fem(lay, period_x_m=60e-9, period_y_m=60e-9, t_end_s=2e-7,
                                     dt_s=1e-8, flux_W_m2=flux, T_sink_K=Tsink,
                                     maxh_m=30e-9, order=2)
    want = steady_layered_temperature([k], [L], flux, Tsink)
    assert tr.steady_limit_T is not None
    assert np.allclose(tr.steady_limit_T, want, rtol=0.0, atol=0.0)     # byte-identical closed form
    assert abs(float(tr.mean_T_per_layer()[0]) - float(want[0])) < 0.05 * (float(want[0]) - Tsink)
    # a Joule source still suppresses it (the pre-existing rule)
    tr_q = solve_thermal_transient_fem(lay, period_x_m=60e-9, period_y_m=60e-9, t_end_s=4e-8,
                                       dt_s=2e-8, flux_W_m2=0.0, T_sink_K=Tsink,
                                       joule_W_m3=1.0e15, maxh_m=30e-9, order=2)
    assert tr_q.steady_limit_T is None


def _const_flux(_t):                      # module level so the result stays PICKLABLE (see below)
    return 1.0e8


def _const_joule(_t):
    return 1.0e16


def test_c9_time_dependent_load_hooks_are_recorded_and_suppress_the_steady_limit():
    """audit C-9 TWIN (wave-5 residual). `flux_of_t` / `joule_of_t` were accepted and driven the
    march, but the RESULT recorded only the static `flux_W_m2` / `joule_W_m3` arguments -- so
    steady_limit_T evaluated the closed form on a load the run never applied. Measured before the
    fix: flux_of_t = 1e8 with the default flux_W_m2 = 0.0 claimed a 300.0 K limit ('no rise') while
    the run settles at 300.5 K, and joule_of_t = 1e16 was invisible to the Joule guard, which
    returned 300.5 K against an actual 303.8 K. Same honesty rule C-9 established: return None."""
    import pickle
    from dynameta.carriers.thermal import steady_layered_temperature
    L, k, Tsink = 100e-9, 10.0, 300.0
    lay = [ThermalLayer("slab", L, k, rho_kg_m3=7140.0, Cp_J_kgK=340.0)]
    kw = dict(period_x_m=60e-9, period_y_m=60e-9, t_end_s=2e-7, dt_s=1e-8, T_sink_K=Tsink,
              maxh_m=30e-9, order=2)

    tr_f = solve_thermal_transient_fem(lay, flux_W_m2=0.0, flux_of_t=_const_flux, **kw)
    assert tr_f.flux_of_t is _const_flux and tr_f.joule_of_t is None
    assert tr_f.steady_limit_T is None                       # was [300.0] -- "no rise"
    reached = float(tr_f.mean_T_per_layer()[0])
    assert reached == pytest.approx(
        float(steady_layered_temperature([k], [L], 1.0e8, Tsink)[0]), abs=0.05)
    assert reached - Tsink > 0.4                             # the rise the stale answer denied

    tr_j = solve_thermal_transient_fem(lay, flux_W_m2=1.0e8, joule_of_t=_const_joule, **kw)
    assert tr_j.joule_of_t is _const_joule
    assert tr_j.steady_limit_T is None                       # was [300.5] -- the Joule term unseen
    assert float(tr_j.mean_T_per_layer()[0]) - Tsink > 3.0   # measured 303.8 K

    # the CONSTANT-load path leaves both fields None, so its result is unchanged ...
    tr_c = solve_thermal_transient_fem(lay, flux_W_m2=1.0e8, **kw)
    assert (tr_c.flux_of_t, tr_c.joule_of_t) == (None, None)
    assert np.allclose(tr_c.steady_limit_T, steady_layered_temperature([k], [L], 1.0e8, Tsink),
                       rtol=0.0, atol=0.0)
    # ... and still pickles. A result carrying MODULE-LEVEL hooks pickles too; a LAMBDA is what does
    # not (documented on ThermalTransientResult rather than worked around by storing a bool).
    pickle.loads(pickle.dumps(tr_c))
    pickle.loads(pickle.dumps(tr_f))
    tr_lam = solve_thermal_transient_fem(lay, flux_W_m2=0.0, flux_of_t=lambda t: 1.0e8, **kw)
    assert tr_lam.steady_limit_T is None
    with pytest.raises(Exception):
        pickle.dumps(tr_lam)
    # the documented escape hatch keeps the honesty AND the picklability
    import dataclasses
    tr_repr = dataclasses.replace(tr_lam, flux_of_t=repr(tr_lam.flux_of_t))
    assert tr_repr.steady_limit_T is None
    pickle.loads(pickle.dumps(tr_repr))
