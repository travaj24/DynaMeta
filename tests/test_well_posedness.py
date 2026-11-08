"""Gates for the v0.15 DC well-posedness check (plan 4.4.2; decisions 55-57):
dynameta.carriers.well_posedness + its hooks (dc_solve.solve_dc, tests/conftest.py).

ORACLES. The null modes are re-derived here, not read back from the module: for a LEFT mode the test
builds w = 1 on the candidate rows itself from devsim.get_equation_numbers and the returned row
permutation and measures |w^T J| against the column norms of the raw matrix; for a RIGHT mode J v with
v = 1 on the Potential columns. Well-posedness itself is shown dynamically (the test_slab_is_well_posed
recipe of the R26 note): a well-posed device whose body is shifted re-converges to the reference; an
ill-posed one does not (or hits the zero pivot).

FAILS ON THE OLD CODE (9b6937f, measured 2026-11-06): no check existed -- the R26 floating slab solved
"successfully" through solve_dc and raw devsim.solve alike (its answer set by round-off: the old
fixture's +20 mV body shift re-converges in place, or UMFPACK hits the exact zero pivot, depending on
the BLAS build: docs/audit/2026-11-03-devsim-recombination-bgn-ionization.md), and a stalled Newton
failure said only "Convergence failure!".
Runtime ~40 s serial on the Windows dev box.
"""
import numpy as np
import pytest

devsim = pytest.importorskip("devsim")

import _wellposed_fixtures as W                                           # noqa: E402

EPS = np.finfo(float).eps


def _raw_left_residual(dev, eqs):
    """Oracle: max_j |sum_i w_i J_ij| / max_j sum_i |w_i| |J_ij| with w = 1 on the rows of `eqs` that
    DEVSIM kept in place (its permutation 'row' == identity)."""
    import scipy.sparse as sp
    m = devsim.get_matrix_and_rhs(format="csc")
    s = m["static"]
    n = len(s["ap"]) - 1
    J = sp.csc_matrix((np.asarray(s["av"]), np.asarray(s["ai"]), np.asarray(s["ap"])), shape=(n, n))
    row = np.asarray(m["permutation"]["row"])
    w = np.zeros(n)
    for eq in eqs:
        k = np.asarray(devsim.get_equation_numbers(device=dev, region="bulk", equation=eq))
        w[k[row[k] == k]] = 1.0
    return float(np.max(np.abs(J.T @ w)) / np.max(abs(J).T @ np.abs(w)))


def _body_shift_reconverges(dev, region="bulk", dpsi=0.02, contact_node=0):
    """Shift psi at every node but the contact's by +dpsi, re-solve (raw devsim.solve), return the max
    |psi - psi_ref| -- 0 for a well-posed device, the shift itself (or a devsim.error) for a floating one."""
    ref = W._v(dev, region, "Potential")
    psi = ref.copy()
    keep = np.zeros(psi.size, bool)
    keep[contact_node] = True
    psi[~keep] += dpsi
    devsim.set_node_values(device=dev, region=region, name="Potential", values=list(psi))
    with W.quiet():
        devsim.solve(type="dc", solver_type="direct", absolute_error=1e11, relative_error=1e-10,
                     maximum_iterations=80)
    return float(np.max(np.abs(W._v(dev, region, "Potential") - ref)))


# =====================================================================================================
# G4-1 the R26 floating body
# =====================================================================================================

@pytest.mark.parametrize("nd", [1e25, -1e25])
def test_r26_floating_slab_is_refused(nd):
    """GATE G4-1. The pre-2026-11-03 R26 slab (Potential-only contact): IllPosedDCError from
    check_dc_well_posed, from dc_solve.solve_dc AND from a raw devsim.solve (the conftest hook), naming
    the region, the carrier and 'conserved'; the mode's equilibrated residual r <= 1e-15 (measured
    6.6e-18 n / 4.6e-18 p; threshold 64 eps = 1.4e-14). Independent oracle: w = 1 on every kept electron
    AND hole row gives |w^T J| <= 64 eps of the column norms on the raw matrix (measured ~1e-16). On the
    old code all three solved without complaint."""
    from dynameta.carriers import dc_solve as DC
    from dynameta.carriers import well_posedness as WP
    dev = "wp_g41_{}".format("n" if nd > 0 else "p")
    try:
        W.slab(dev, nd=nd, contact="potential")
        rep = WP.check_dc_well_posed(dev, raise_error=False)
        assert len(rep.confirmed) == 1 and rep.confirmed[0].r <= 1e-15
        assert _raw_left_residual(dev, ("ElectronContinuityEquation", "HoleContinuityEquation")) < 64 * EPS
        with pytest.raises(WP.IllPosedDCError) as e1:
            WP.check_dc_well_posed(dev)
        msg = str(e1.value)
        assert "'bulk'" in msg and "conserved" in msg and "allow_floating" in msg
        assert ("Electron" in msg) if nd > 0 else ("Hole" in msg)
        with pytest.raises(WP.IllPosedDCError):
            DC.solve_dc(dev, abs_tol=1e11, rel_tol=1e-10)
        with pytest.raises(WP.IllPosedDCError):                      # the conftest hook
            devsim.solve(type="dc", solver_type="direct", absolute_error=1e11, relative_error=1e-10,
                         maximum_iterations=80)
    finally:
        W.drop(dev)


@pytest.mark.floating_ok("demonstrates the old, ill-posed R26 fixture")
def test_r26_floating_slab_answer_is_not_determined_on_the_old_path():
    """The failure the check prevents, shown on the raw (unchecked) path: the floating slab's body
    shifted by +20 mV either re-converges IN PLACE (the shift survives) or UMFPACK meets the exact zero
    pivot -- which of the two depends on the BLAS build (R26 note). Either way the DC answer is not
    determined. Refutation: test_majority_pinned_slab_is_well_posed (the same shift comes back to 0)."""
    dev = "wp_old"
    try:
        W.slab(dev, nd=1e25, contact="potential")
        with W.quiet():
            devsim.solve(type="dc", solver_type="direct", absolute_error=1e11, relative_error=1e-10,
                         maximum_iterations=80)
        try:
            moved = _body_shift_reconverges(dev)
        except devsim.error as e:
            assert "numeric" in str(e).lower() or "singular" in str(e).lower() or \
                "convergence" in str(e).lower()
            return
        assert moved > 1e-6
    finally:
        W.drop(dev)


@pytest.mark.parametrize("nd", [1e25, -1e25])
def test_majority_pinned_slab_is_well_posed(nd):
    """Refutation of G4-1 (the 2026-11-03 fix): psi + the MAJORITY density pinned at 'lo'. The static
    pass still proposes the free MINORITY block, the matrix refutes it -- r = 4.4e-8 (n) / 1.5e-8 (p),
    the SRH coupling h^2 / (D tau), 6 orders above the threshold -- and nothing raises; the +20 mV body
    shift comes back to the reference (< 1e-9 V)."""
    from dynameta.carriers import well_posedness as WP
    dev = "wp_fix_{}".format("n" if nd > 0 else "p")
    try:
        W.slab(dev, nd=nd, contact="majority")
        rep = WP.check_dc_well_posed(dev)
        assert not rep.confirmed and len(rep.refuted) == 1 and rep.refuted[0].r > 1e3 * WP.TOL_SINGULAR
        minority = "HoleContinuityEquation" if nd > 0 else "ElectronContinuityEquation"
        assert [b.equation for b in rep.refuted[0].weights] == [minority]
        with W.quiet():
            devsim.solve(type="dc", solver_type="direct", absolute_error=1e11, relative_error=1e-10,
                         maximum_iterations=80)
        assert _body_shift_reconverges(dev) < 1e-9
    finally:
        W.drop(dev)


# =====================================================================================================
# G4-2 / G4-3: rank deficiency 2, the right null vector, the equilibrium refutation
# =====================================================================================================

def test_recombination_free_slab_has_rank_deficiency_two():
    """GATE G4-2. tau = 1e30 (no Auger / radiative), Potential-only contact: each species is conserved
    separately -> TWO confirmed modes, one per carrier. Refutation: pin the electrons (the minority of
    this p slab) -- the holes alone still raise."""
    from dynameta.carriers import well_posedness as WP
    try:
        W.slab("wp_u0", nd=1e25, contact="potential", tau=1e30)
        rep = WP.check_dc_well_posed("wp_u0", raise_error=False)
        assert len(rep.confirmed) == 2
        assert sorted(b.equation for m in rep.confirmed for b in m.weights) == \
            ["ElectronContinuityEquation", "HoleContinuityEquation"]
    finally:
        W.drop("wp_u0")
    try:
        W.slab("wp_u0e", nd=-1e25, contact="electrons", tau=1e30)
        with pytest.raises(WP.IllPosedDCError, match="hole number"):
            WP.check_dc_well_posed("wp_u0e")
    finally:
        W.drop("wp_u0e")


def test_pure_neumann_potential_is_refused_but_equilibrium_is_not():
    """GATE G4-3. A bipolar DD slab whose 'lo' contact pins both carriers but NOT the potential: J v = 0
    for v = 1 on the Potential columns (oracle: computed here on the raw matrix, <= 64 eps of the row
    norms) -> IllPosedDCError naming 'pure-Neumann'. Refutation: an EQUILIBRIUM (n = n(psi)) region with
    no contact equation at all is NOT refused -- n(psi) fixes the absolute level (refuted, r = O(1))."""
    import scipy.sparse as sp
    from dynameta.carriers import physics_equilibrium as EQ
    from dynameta.carriers import well_posedness as WP
    try:
        W.slab("wp_neu", nd=1e25, contact="carriers")
        m = devsim.get_matrix_and_rhs(format="csc")["static"]
        n = len(m["ap"]) - 1
        J = sp.csc_matrix((np.asarray(m["av"]), np.asarray(m["ai"]), np.asarray(m["ap"])), shape=(n, n))
        v = np.zeros(n)
        v[np.asarray(devsim.get_equation_numbers(device="wp_neu", region="bulk", variable="Potential"))] = 1
        assert np.max(np.abs(J @ v)) <= 64 * EPS * np.max(abs(J) @ v)
        with pytest.raises(WP.IllPosedDCError, match="pure-Neumann"):
            WP.check_dc_well_posed("wp_neu")
    finally:
        W.drop("wp_neu")
    try:
        W._mesh_1d("wp_eq", [(0.0, 1e-9, "a"), (50e-9, 1e-9, "b")], [("bulk", "Si", "a", "b")])
        with W.quiet():
            EQ.setup_semiconductor_region("wp_eq", "bulk", n_bg_m3=1e25, eps_static=11.7,
                                          dos_mass_kg=0.26 * 9.1093837015e-31)
        rep = WP.check_dc_well_posed("wp_eq")
        assert [b.equation for g in rep.candidates for b in g] == ["PotentialEquation"]
        assert not rep.confirmed and rep.refuted and rep.refuted[0].r > 1e-3
    finally:
        W.drop("wp_eq")


# =====================================================================================================
# G4-5 the floating CONTACT is well-posed; acknowledgement; cache
# =====================================================================================================

_FLOAT_CHILD = r"""
import json, sys
sys.path.insert(0, "tests")
import numpy as np
import _wellposed_fixtures as W
from dynameta.carriers import physics_bipolar_dd as BP
from dynameta.carriers import well_posedness as WP
ds = W.ds
dev = "wp_float"
W.slab(dev, nd=1e23, contact="ohmic", length=1e-6)
with W.quiet():
    BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="wp_vf", source_name="I_wpf",
                                           drive="current")
rep = WP.check_dc_well_posed(dev)

def solve():
    with W.quiet():
        WP.ensure_dc_well_posed()
        ds.solve(type="dc", solver_type="direct", absolute_error=1e11, relative_error=1e-10,
                 maximum_iterations=80)

solve()
ref = W._v(dev, "bulk", "Potential")
psi = ref.copy(); psi[1:] += 0.02
ds.set_node_values(device=dev, region="bulk", name="Potential", values=list(psi))
solve()
print("FLOAT_RESULT " + json.dumps({"candidates": len(rep.candidates), "well_posed": rep.well_posed,
      "shift": float(np.max(np.abs(W._v(dev, "bulk", "Potential") - ref)))}))
"""


def test_floating_contact_through_a_zero_current_node_is_well_posed():
    """GATE G4-5. An n slab: 'lo' ohmic and grounded, 'hi' an ohmic contact tied to a circuit node
    driven by a 0 A current source (open circuit -- the SOI-body / open-circuit-voltage recipe). Its
    carrier rows are pinned relative to the node's UNKNOWN potential, so there is no candidate and no
    raise, and the +20 mV body shift comes back (< 1e-9 V; measured 0): one DC solution. Run in a CHILD
    process: DEVSIM circuits cannot be deleted in-process, and a left-over current-source node with no
    device behind it makes every later DC solve of the session singular (its KCL row is empty) -- a
    circuit-level floating node this device-level check does not see (audit note)."""
    import json
    import os
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [repo] + [p for p in sys.path if isinstance(p, str) and p]))
    p = subprocess.run([sys.executable, "-W", "error", "-c", _FLOAT_CHILD], cwd=repo, env=env,
                       capture_output=True, text=True, timeout=600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("FLOAT_RESULT ")]
    assert p.returncode == 0 and lines, (p.returncode, p.stdout[-1500:], p.stderr[-3000:])
    res = json.loads(lines[-1][len("FLOAT_RESULT "):])
    assert res["candidates"] == 0 and res["well_posed"], res
    assert res["shift"] < 1e-9, res


def test_allow_floating_acknowledges_and_the_cache_is_invalidated():
    """allow_floating turns the R26 mode into report.acknowledged (no raise) and requires a reason; the
    hook's verdict is cached (a second ensure_dc_well_posed assembles nothing) and eq_registry.clear
    invalidates it."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import well_posedness as WP
    dev = "wp_ack"
    calls = []
    real = WP._assemble
    try:
        W.slab(dev, nd=1e25, contact="potential")
        with pytest.raises(ValueError, match="reason"):
            WP.allow_floating(dev, "bulk", reason=" ")
        WP.allow_floating(dev, "bulk", reason="isolated well, solved transiently")
        rep = WP.check_dc_well_posed(dev)
        assert rep.acknowledged and not rep.confirmed
        WP._assemble = lambda: calls.append(1) or real()
        WP.ensure_dc_well_posed(dev)
        WP.ensure_dc_well_posed(dev)
        assert len(calls) == 1
        _R.clear(dev)                                       # drops the acknowledgement + bumps generation
        with pytest.raises(WP.IllPosedDCError):
            WP.ensure_dc_well_posed(dev)
        assert len(calls) == 2
    finally:
        WP._assemble = real
        W.drop(dev)


# =====================================================================================================
# G4-6 near-singular diagnosis on a stalled Newton
# =====================================================================================================

def test_mos_inversion_stall_carries_the_near_singular_diagnosis():
    """GATE G4-6 (plan P4 1.1's MOS: p 3e23, classical DD, double precision, 20 mV steps through
    solve_dc with the new default abs_tol). Up to 0.78 V every step converges -- no message; at 0.30 V
    the equilibrated sigma_min/sigma_max is 2.3e-11 (not near-singular). At 0.80 V Newton STALLS (40
    iterations, the electron relative update wandering 1e-6..1e-2 -- measured) and the re-raised
    devsim.error carries 'near-singular DC Jacobian', sigma_min/sigma_max <= 1e-15 (measured 4.4e-16)
    and 'mode in Electrons'. Refutations: a failure that is NOT a stall -- a 0.78 -> 1.2 V jump that
    ends far from a solution -- carries no diagnosis (the dense SVD is not paid on every routine
    failure); and the same diagnosis at 0.30 V does not say near-singular.
    REVISED 2026-11-07 (v0.15.0 default statistics 'fermi_dirac', the FD contact root + quasi-field): the stall
    moved ONE step earlier, to 0.78 V (MEASURED; the last converged step is 0.76 V). The gate walks the 20 mV
    ladder to the FIRST stall and requires it in [0.70, 0.90] V -- the double-precision floor of this
    inversion layer, not a statistics effect -- then checks the same diagnosis there and the jump refutation
    from the last converged step.
    REVISED 2026-11-08 (Linux CI; docs/audit/2026-11-08-v0.15-linux-ci-exactness.md). Both the stall's walk and
    the reported ratio are round-off at this floor, so neither may be pinned to one platform's value:
    (i) WHERE the stalled walk ends is chance (the update along the near-null mode is round-off x 1/sigma_min):
    Windows stalls at 0.78 V wandering 3e-10..4e-7; Linux (OpenBLAS 0.3.26) converges 0.78 V, stalls at 0.80 V,
    reaches 4.7e-5 at iteration 13, escapes to 1e3 and ends at 6.5e-2 -- the old last-iterate stall rule gave
    no diagnosis there. dc_solve._stalled now asks whether ANY iterate came within 1e-2 (MEASURED best 2.7e-10
    / 4.7e-5). The jump refutation runs 10 iterations (MEASURED >= 1e3 on both platforms: unambiguously far;
    over 40 its best was 1.9e-2 / 4.5e-2, within 2-5x of the 1e-2 rule). (ii) The true equilibrated ratio is
    below double resolution (MEASURED 1.6e-17 / 1.9e-17 by gesdd without vectors, 1.6e-17 / 3.7e-17 by inverse
    iteration on a sparse LU), so the dense SVD returns ITS round-off: 4.4e-16 (2 eps) on Windows, 3.6e-15
    (16 eps) in WSL and 1.4e-15 on the CI runner (another process history) for the same fixture. A priori: LAPACK's SVD is backward stable, the computed
    sigma are the exact ones of J_eq + E with ||E||_2 <= p(n) u ||J_eq||_2 (LAPACK Users' Guide 3rd ed. 4.9;
    p 'a modestly growing function', taken linear: p(n) = 2n), and by Weyl sigma_min(computed)/sigma_max
    <= ratio_true + p(n) u = n eps (1.8e-13 for the 804 unknowns): that is the gate (was 1e-15, MEASURED
    0.002 / 0.02 of the new bound). It still sits 5.6x below NEAR_SINGULAR (1e-12) and 130x below the 0.30 V
    ratio; picking s[-2] (an off-by-one: 6.0e-7) or a well-conditioned block fails it."""
    from dynameta.carriers import dc_solve as DC
    from dynameta.carriers import well_posedness as WP
    dev = "wp_mos"
    try:
        bend, _ = W.mos_inversion(dev)
        with W.quiet():
            DC.solve_dc(dev, rel_tol=1e-7, max_iter=100)
        e, v_stall = None, None
        for k in range(1, 46):
            bend(0.02 * k)
            try:
                with W.quiet():
                    DC.solve_dc(dev, rel_tol=1e-7, max_iter=40)
            except devsim.error as err:
                e, v_stall = err, 0.02 * k
                break
            if k == 15:
                d30 = WP.diagnose_singular(dev)
        assert e is not None and 0.70 <= v_stall <= 0.90, v_stall
        n_dof = WP._assemble()[0].shape[0]                 # the unknowns of the diagnosed Jacobian (804)
        bend(1.2)                                           # DEVSIM restored the last converged state
        with pytest.raises(devsim.error) as e_jump:
            with W.quiet():
                DC.solve_dc(dev, rel_tol=1e-7, max_iter=10)
    finally:
        W.drop(dev)
    assert str(e_jump.value).startswith("Convergence failure!") and "sigma_min" not in str(e_jump.value)
    msg = str(e)
    assert msg.startswith("Convergence failure!") and "near-singular DC Jacobian" in msg
    assert "mode in Electrons" in msg
    assert "physics_bipolar_dd.setup_quasi_equilibrium_carrier" in msg     # the remedy is named (v0.15 4.4.3)
    ratio = float(msg.split("sigma_min/sigma_max = ")[1].split(")")[0])
    assert ratio <= n_dof * EPS, (ratio, n_dof)          # the SVD's backward-error floor (docstring (ii))
    assert "near-singular" not in d30 and "mode in Electrons" in d30


# =====================================================================================================
# defaults-off: the check and solve_dc's info=True path change no result
# =====================================================================================================

@pytest.mark.devsim_isolated("bitwise comparison of separate DEVSIM solves: a device left live by another test is iterated by the same joint Newton and moves the answer by ulps")
def test_check_and_the_info_path_leave_results_bit_identical():
    """Same-tree A/B (plan section 2 item 5 (i)). An ohmic n slab and the majority-pinned R26 slab (whose
    minority candidate makes the check ASSEMBLE the matrix): (a) check_dc_well_posed changes no node
    value (bitwise, before vs after); (b) the solved state through solve_dc -- the check + DEVSIM's
    info=True Newton -- is bitwise equal to the same solve through the unwrapped devsim.solve with no
    check. (ii) the call-log digests (tests/_bipolar_calllog.py) and (iii) the cross-version numeric
    gates are run in the audit note: the check defines no model and no equation."""
    from dynameta.carriers import dc_solve as DC
    from dynameta.carriers import well_posedness as WP
    raw = getattr(devsim.solve, "__wrapped__", devsim.solve)
    names = ("Potential", "Electrons", "Holes")
    for kind in ("ohmic", "majority"):
        out = []
        for path in ("solve_dc", "raw"):
            dev = "wp_ab_{}_{}".format(kind, path)
            try:
                W.slab(dev, nd=1e25, contact=kind, hi_contact="ohmic" if kind == "ohmic" else None)
                devsim.set_node_values(device=dev, region="bulk", name="Holes",
                                       values=list(3.0 * W._v(dev, "bulk", "Holes")))
                before = {v: W._v(dev, "bulk", v) for v in names}
                WP.check_dc_well_posed(dev)
                assert all(np.array_equal(before[v], W._v(dev, "bulk", v)) for v in names)
                with W.quiet():
                    if path == "solve_dc":
                        DC.solve_dc(dev, abs_tol=1e11, rel_tol=1e-10, max_iter=80)
                    else:
                        raw(type="dc", solver_type="direct", absolute_error=1e11, relative_error=1e-10,
                            maximum_iterations=80)
                out.append({v: W._v(dev, "bulk", v) for v in names})
            finally:
                W.drop(dev)
        assert all(np.array_equal(out[0][v], out[1][v]) for v in names), kind
