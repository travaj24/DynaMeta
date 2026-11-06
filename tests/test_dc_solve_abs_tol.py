"""Gates for the v0.15 density-scaled default absolute tolerance (plan 4.4.1, decision 58):
dc_solve.default_abs_tol = max(1e10, 1024 u S), solve_dc(abs_tol=None).

ORACLES. The convergence and accuracy gates compare the DOUBLE-precision Newton under the new default
against the SAME device solved by DEVSIM's extended (128-bit residual) Newton and then tightened to
relative 1e-25 -- an independent arithmetic, not the code under test (the extended run keeps the
historical 1e10, which is attainable there). The aggregation S is re-derived here from the raw node
values, not read back from solution_size. Fixtures: tests/_abs_tol_fixtures.py.

FAILS ON THE OLD DEFAULT (MEASURED 2026-11-06, scipy OpenBLAS, DEVSIM 2.11.2, this box):
  * ITO gate (unipolar DD, n_bg = 4e26): abs_tol = 1e10 EXHAUSTS 100 iterations on the first gate step
    (0.25 V) -- 1e10 is below ONE ulp of 4e26 (8.9e10), so only an exactly-zero update at every node
    could satisfy it; the measured floor is 0.5-1.06 ulp of S. New default: 4/4/4/4 iterations
    (extended: 5/4/4/4).
  * GATE C classical accumulation (n 1e24, wall +0.15 V in 5 steps, peak 8.2e25): old 6/6/11/7/9
    iterations (up to +5 over the extended run's 6/6/6/6/6, waiting for a lucky update); new 6/6/6/6/6.
    (Round-off decides the old count, so it is recorded here, not asserted: D2d measured 10-18 extra
    iterations and one exhaustion on Elmer's OpenBLAS.)
  * R27-a slab (1e25): 3 = 3 iterations either way (its floor happens to sit under 1e10).
Runtime 65 s serial on the Windows dev box (every test < 13 s; DEVSIM model setup and the two
extended-precision MOS ramps dominate).
"""
import numpy as np
import pytest

devsim = pytest.importorskip("devsim")

import _abs_tol_fixtures as F                                             # noqa: E402

U53 = 2.0 ** -53
THRESHOLD = 1.0e10 / (1024 * U53)                 # 8.796e22 m^-3: below it the floor 1e10 binds


def _S_oracle(dev):
    """DEVSIM's aggregation, re-derived: max over regions of sum over equations of max |x|."""
    best = 0.0
    for r in devsim.get_region_list(device=dev):
        tot = 0.0
        for eq in devsim.get_equation_list(device=dev, region=r):
            var = devsim.get_equation_command(device=dev, region=r, name=eq)["variable_name"]
            tot += max(abs(v) for v in devsim.get_node_model_values(device=dev, region=r, name=var))
        best = max(best, tot)
    return best


def _run(build, dev, rel, *, extended=False, tighten=False, **kw):
    from dynameta.carriers import dc_solve as DC
    steps, regs = build(dev, extended=extended, **kw)
    try:
        log = F.run_steps(steps, abs_tol=DC.default_abs_tol, rel_tol=rel)   # every live device
        if tighten:
            with F.quiet():
                info = F.XP.solve(type="dc", solver_type="direct", absolute_error=1e10,
                                  relative_error=1e-25, maximum_iterations=100, info=True)
            assert info["converged"]
        return log, F.snapshot(dev, regs), DC.default_abs_tol()
    finally:
        F.drop(dev)


def test_default_matches_the_formula_and_the_threshold():
    """default_abs_tol == max(1e10, 1024 * 2^-53 * S) with S re-derived from the node values, on a
    light (1e22: exactly 1e10, the SAME float) and a degenerate (4e26) device; the threshold is
    8.8e22 m^-3; an extended-precision device keeps exactly 1e10 at S = 8e25 (u = 2^-113)."""
    from dynameta.carriers import dc_solve as DC
    assert THRESHOLD == pytest.approx(8.796e22, rel=1e-3)
    steps, _ = F.r27a_slab("abs_light", nd=1e22)
    try:
        s = _S_oracle("abs_light")
        assert s < THRESHOLD and DC.default_abs_tol("abs_light") == 1e10
        assert DC.solution_size("abs_light") == s
    finally:
        F.drop("abs_light")
    F.ito_gate("abs_ito")
    try:
        s = _S_oracle("abs_ito")
        assert s == pytest.approx(4e26, rel=1e-12)
        assert DC.default_abs_tol("abs_ito") == max(1e10, 1024 * U53 * s)
    finally:
        F.drop("abs_ito")
    F.r27a_slab("abs_ext", extended=True, nd=8e25)
    try:
        assert F.XP.is_enabled("abs_ext") and _S_oracle("abs_ext") >= 8e25
        assert DC.default_abs_tol("abs_ext") == 1e10       # G5-5: 1024 * 2^-113 * 8e25 ~ 8e-6
    finally:
        F.drop("abs_ext")


@pytest.mark.parametrize("name", ["ito_gate", "mos_accumulation", "r27a_slab"])
def test_degenerate_devices_converge_like_the_extended_newton(name):
    """GATE G5-1 + G5-2. Every continuation step converges under the new default within 2 iterations
    of the extended run's count (measured: equal, or 1 FEWER on the ITO's first step), with zero
    exhaustion; and every node of every variable is within rel_tol |x*| + abs_tol of the extended
    reference x* (tightened to relative 1e-25). The MINORITY is checked on its own: the GATE C bulk
    holes, 1e8 m^-3 next to the 8.2e25 electron peak, must match x* to rel_tol RELATIVE (measured
    1.2e-15; the abs_tol of ~5e12 m^-3 would allow anything there -- it is the relative criterion,
    a max over nodes, that protects them). Refutation: test_gate_resolves_a_one_microvolt_change."""
    build, rel = F.FIXTURES[name]
    log_d, x_d, abs_new = _run(build, "abs_d_" + name, rel)
    log_x, x_ref, _ = _run(build, "abs_x_" + name, rel, extended=True, tighten=True)
    assert all(ok for _, ok, _ in log_d), log_d
    assert len(log_d) == len(log_x)
    for (it_d, _, _), (it_x, _, _) in zip(log_d, log_x):
        assert it_d <= it_x + 2, (log_d, log_x)
    for k, xs in x_ref.items():
        assert np.all(np.abs(x_d[k] - xs) <= rel * np.abs(xs) + abs_new), k
    if name == "mos_accumulation":
        h = x_ref[("s", "Holes")]
        assert h.max() == pytest.approx(1e8, rel=1e-6) and x_ref[("s", "Electrons")].max() > 8e25
        assert np.max(np.abs(x_d[("s", "Holes")] / h - 1.0)) < rel


def test_old_default_cannot_converge_the_ito_gate():
    """The failure the new default removes (shown on the old default): with abs_tol = 1e10 the ITO
    gate's first 0.25 V step exhausts 100 iterations. Not round-off luck: 1e10 m^-3 is 0.11 ulp of the
    4e26 background (ulp 8.9e10), so it demands an exactly-zero update at every node."""
    steps, _ = F.ito_gate("abs_old")
    try:
        log = F.run_steps(steps, abs_tol=1e10, rel_tol=1e-5)
    finally:
        F.drop("abs_old")
    assert 1e10 < np.spacing(4e26)
    assert log[0][1] and not log[1][1] and log[1][0] == 100


def test_gate_resolves_a_one_microvolt_change():
    """Refutation of G5-2: the per-node bound rel_tol |x*| + abs_tol is tight enough to see a 1 uV
    change of the wall potential (dn/n ~ 1e-6 / V_t = 4e-5 >> rel_tol = 1e-7): a reference solved at
    +0.15 V + 1 uV must FAIL the bound."""
    _, x_d, abs_new = _run(F.mos_accumulation, "abs_p0", 1e-7)
    _, x_p, _ = _run(F.mos_accumulation, "abs_p1", 1e-7, delta=0.150001)
    e = x_p[("s", "Electrons")]
    assert abs_new > 1e12                                  # the degenerate default is in force
    assert np.any(np.abs(x_d[("s", "Electrons")] - e) > 1e-7 * np.abs(e) + abs_new)


def test_fourteenfold_density_growth_within_one_step_still_converges():
    """G5-4 refutation of the K = 1024 margin: the default is resolved on the PRE-solve state, so S can
    grow during the step. The GATE C ramp in 2 steps of 75 mV grows the peak 1e24 -> 1.4e25 (14x) in
    the first step: it still converges, within 2 iterations of the extended run (measured 7 / 7 vs
    extended)."""
    log_d, _, _ = _run(F.mos_accumulation, "abs_g_d", 1e-7, steps=2)
    log_x, _, _ = _run(F.mos_accumulation, "abs_g_x", 1e-7, extended=True, steps=2)
    assert all(ok for _, ok, _ in log_d)
    for (it_d, _, _), (it_x, _, _) in zip(log_d, log_x):
        assert it_d <= it_x + 2, (log_d, log_x)


def test_round_off_floor_is_inside_the_margin():
    """G5-4: the K = 1024 derivation. Run every GATE C / ITO step with abs_tol = 0 (unattainable) and
    record DEVSIM's device absolute error on every iteration AFTER the relative test first passed: the
    round-off floor, in ulp of the pre-solve S. K u = 512 ulp must exceed 10x its max (measured max
    11.0 ulp GATE C -- the pre-solve S, so the in-step growth is included -- and 1.06 ulp ITO; median
    1.5 / 0.5)."""
    from dynameta.carriers import dc_solve as DC
    worst = 0.0
    for name in ("mos_accumulation", "ito_gate"):
        build, rel = F.FIXTURES[name]
        dev = "abs_k_" + name
        steps, _ = build(dev)
        try:
            for step in steps:
                step()
                S = _S_oracle(dev)
                with F.quiet():
                    info = F.XP.solve(type="dc", solver_type="direct", absolute_error=0.0,
                                      relative_error=rel, maximum_iterations=25, info=True)
                assert not info["converged"]
                a = np.array([it["devices"][0]["absolute_error"] for it in info["iterations"]])
                r = np.array([it["devices"][0]["relative_error"] for it in info["iterations"]])
                first = int(np.flatnonzero(r < rel)[0])
                worst = max(worst, float(np.max(a[first + 1:])) / (2.0 ** -52 * S))
                F.run_steps([lambda: None], abs_tol=DC.default_abs_tol, rel_tol=rel)
        finally:
            F.drop(dev)
    assert 10.0 * worst <= DC.ABS_TOL_ULPS / 2, worst


def test_light_doping_and_equilibrium_are_byte_identical():
    """G5-3 / G5-6: below the threshold the default IS the old 1e10 -- solve_dc(abs_tol=None) and the
    explicit abs_tol=1e10 give bitwise-equal arrays on a 1e22 slab (S = 1e22) and on a Poisson-only
    equilibrium device (abs in volts, S ~ 1 V). Just above (1e23: default 1.14e10) the iteration count
    and the answer agree with the explicit 1e10 run to rel_tol.
    The default sizes EVERY live device (DEVSIM's Newton is joint), so the explicit reference is the
    1e10 the old code used only when nothing heavier is live in this process (`ref`); with a heavier
    device left by another module's fixture, 1e10 would stall that device's Newton and the honest
    reference is the same default float (test_default_sizes_every_live_device)."""
    from dynameta.carriers import dc_solve as DC
    from dynameta.carriers import physics_equilibrium as EQ
    others = DC.default_abs_tol()                       # nothing of this test is live yet
    ref = 1e10 if others == 1e10 else others

    def slab(dev, nd, abs_tol):
        steps, regs = F.r27a_slab(dev, nd=nd)
        try:
            with F.quiet():
                DC.solve_dc(dev, abs_tol=abs_tol, rel_tol=1e-10, max_iter=80)
            return F.snapshot(dev, regs)
        finally:
            F.drop(dev)

    a, b = slab("abs_b1", 1e22, None), slab("abs_b2", 1e22, ref)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    hi_a, hi_b = slab("abs_b3", 1e23, None), slab("abs_b4", 1e23, max(ref, 1e10))
    assert all(np.max(np.abs(hi_a[k] - hi_b[k]) / np.abs(hi_b[k])) <= 1e-10 for k in hi_a)

    def eq(dev, abs_tol):
        F._mesh_1d(dev, [(0.0, 1e-9, "a"), (50e-9, 1e-9, "b")], [("r", "Si", "a", "b")])
        try:
            with F.quiet():
                EQ.setup_semiconductor_region(dev, "r", n_bg_m3=4e26, eps_static=9.5,
                                              dos_mass_kg=0.35 * F.M_E)
                EQ.setup_contact(dev, "a")
                EQ.setup_contact(dev, "b")
                devsim.set_parameter(device=dev, name="a_bias", value=0.5)
                assert DC.default_abs_tol(dev) == 1e10        # S = |psi| in volts
                # rel 1e-5 (the default): the relative test floors at ~1e-6 where psi crosses 0
                DC.solve_dc(dev, abs_tol=abs_tol, rel_tol=1e-5, max_iter=80)
            return F.snapshot(dev, ("r",), ("Potential",))
        finally:
            F.drop(dev)

    e1, e2 = eq("abs_e1", None), eq("abs_e2", ref)
    assert np.array_equal(e1[("r", "Potential")], e2[("r", "Potential")])


# =====================================================================================================
# every live device (the v0.15 integration fix, 2026-11-06)
# =====================================================================================================

def test_default_sizes_every_live_device():
    """GATE: DEVSIM's ds.solve runs ONE Newton over every live device and demands abs < absolute_error
    on EACH, so the default must be attainable on the heaviest. A converged ITO gate (n_bg 4e26, at
    1 V) is left live -- as a module-scoped fixture of another test file is under xdist -- and a light
    1e22 slab is then solved by solve_dc(abs_tol=None): it must converge, to its solo answer within
    rel_tol (measured: bitwise), and default_abs_tol() must be the max over devices of 1024 u_d S_d.
    FAILS ON 94e61fc (measured): default_abs_tol(device) sized the light slab alone (exactly 1e10,
    0.11 ulp of the live 4e26 background), so the joint Newton ran its 60 iterations and raised -- the
    same mechanism as the order-dependent test_w1_layered_oxide_tunneling_extras_and_default_off
    failure (the leftover there: test_devsim_tunneling's module-scoped 9 V MOS, floor 2.0e11 against
    the builder device's 1.1e11). Refutation: the explicit abs_tol=1e10 still cannot converge it."""
    import _wellposed_fixtures as W
    from dynameta.carriers import dc_solve as DC
    names = ("Potential", "Electrons", "Holes")
    W.slab("abs_solo", nd=1e22, contact="ohmic", hi_contact="ohmic")
    try:
        with F.quiet():
            DC.solve_dc("abs_solo", rel_tol=1e-10, max_iter=60)
        solo = F.snapshot("abs_solo", ("bulk",), names)
    finally:
        W.drop("abs_solo")
    steps, _ = F.ito_gate("abs_left")
    try:
        assert all(ok for _, ok, _ in F.run_steps(steps, abs_tol=DC.default_abs_tol, rel_tol=1e-5))
        W.slab("abs_with", nd=1e22, contact="ohmic", hi_contact="ohmic")
        s_ito = _S_oracle("abs_left")
        assert DC.default_abs_tol("abs_with") == 1e10
        assert DC.default_abs_tol() == max(1e10, 1024 * U53 * s_ito) > 1e13
        with F.quiet():
            DC.solve_dc("abs_with", rel_tol=1e-10, max_iter=60)
        joint = F.snapshot("abs_with", ("bulk",), names)
        # refutation: the light device's own 1e10 cannot converge the joint Newton, and the failure
        # message names the device that holds it back
        with pytest.raises(devsim.error, match="abs_left/ito/ElectronContinuityEquation"):
            with F.quiet():
                DC.solve_dc("abs_with", abs_tol=1e10, rel_tol=1e-10, max_iter=60)
    finally:
        W.drop("abs_with")
        F.drop("abs_left")
    for k, x in solo.items():
        assert np.all(np.abs(joint[k] - x) <= 1e-10 * np.abs(x)), k


def test_mixed_precision_devices_are_sized_by_the_double_one():
    """GATE: u is per device. A live extended-precision device (residual in 128-bit) must not shrink
    the tolerance of a DOUBLE device solved with it (its residual stays double inside the extended
    Newton): with an extended slab (8e25) live, default_abs_tol() = 1024 * 2^-53 * S of the double
    ITO gate, and solve_dc converges the ITO ramp. FAILS ON 94e61fc (measured): XP.any_enabled() made
    u = 2^-113 for everything, so the default was exactly 1e10 (0.11 ulp of 4e26) and the first gate
    step exhausted its iterations."""
    from dynameta.carriers import dc_solve as DC
    from dynameta.warning_types import AdvisoryWarning
    steps, _ = F.ito_gate("abs_mx_dbl")             # its Poisson pre-solve runs while it is alone
    try:
        F.r27a_slab("abs_mx_ext", extended=True, nd=8e25)
        try:
            assert F.XP.is_enabled("abs_mx_ext") and not F.XP.is_enabled("abs_mx_dbl")
            assert DC.default_abs_tol() == max(1e10, 1024 * U53 * _S_oracle("abs_mx_dbl"))
            with pytest.warns(AdvisoryWarning, match="opted in"):
                for step in steps[:2]:
                    step()
                    with F.quiet():
                        DC.solve_dc("abs_mx_dbl", rel_tol=1e-5, max_iter=40)
        finally:
            F.drop("abs_mx_ext")
    finally:
        F.drop("abs_mx_dbl")
