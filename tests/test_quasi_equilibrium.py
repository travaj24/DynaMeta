"""v0.15 item 4.4.3 (decisions 48/49): the QUASI-EQUILIBRIUM carrier closure for strong inversion
(physics_bipolar_dd.setup_quasi_equilibrium_carrier). The MOS harness and the Schrodinger-Poisson
inversion oracle live in validation/bipolar_density_gradient.py (GATE G: the full gamma / doping /
precision matrix and the SP comparison, nightly). Plan: docs/plans/2026-11-04-devsim-v0.15/
P4_dg_ii_numerics.md sec. 1.6; audit note docs/audit/2026-11-06-v0.15-dg-wall-strong-inversion.md.

On 6f9044d (before the closure) every gate here fails: setup_quasi_equilibrium_carrier does not
exist, and the full model it replaces cannot reach the strong-inversion states at all (classical DD:
Newton fails at 0.78-0.80 V; bipolar DG: 0.70 V -- P4 1.1 and the 4.4.2 stall diagnosis)."""
import os
import subprocess
import sys

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validation.bipolar_density_gradient as V  # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP  # noqa: E402
from dynameta.constants import EPS0, Q_E, V_T  # noqa: E402

NA = 3.0e23


def _qs_boltzmann(delta, n_a=NA):
    """|Q_s| [C/m^2] of the classical Boltzmann MOS, Poisson's first integral (Kingston-Neustadter /
    Sze ch. 4 F-function, re-derived: E_s^2 = (2 q p0 V_t/eps)[(e^-x + x - 1) + (n0/p0)(e^x - x - 1)],
    x = delta/V_t, Q_s = eps E_s by Gauss) with the DD's own neutral body p0 = CHOLE, n0 = n_i^2/p0."""
    eps = V.EPS * EPS0
    p0 = 1.0 + 0.5 * (n_a + np.sqrt(n_a ** 2 + 4.0 * V.NI_SI ** 2))
    n0 = V.NI_SI ** 2 / p0
    x = delta / V_T
    f = (np.exp(-x) + x - 1.0) + (n0 / p0) * (np.exp(x) - x - 1.0)
    return float(np.sqrt(2.0 * Q_E * p0 * V_T * eps * f))


def _probe_rows(dev):
    """FD-activity value and Jacobian of the closure row, evaluated on the live device by writing a
    wide range of electron densities u = n/N_c in [1e-6, 136] (the g fit's validated range)."""
    reg = "s"
    nn = V._get(dev, reg, "x").size
    nc = float(ds.get_parameter(device=dev, region=reg, name="N_dos"))
    n = nc * np.geomspace(1e-6, 136.0, nn)
    ds.set_node_values(device=dev, region=reg, name="Electrons", values=list(n))
    ds.set_node_values(device=dev, region=reg, name="Potential", values=[0.0] * nn)
    imref = V._get(dev, reg, "QE_n_imref")
    row = V._get(dev, reg, "QE_n")
    jac = V._get(dev, reg, "QE_n:Electrons")
    fd = []
    for s in (+1.0, -1.0):
        ds.set_node_values(device=dev, region=reg, name="Electrons", values=list(n * (1.0 + s * 1e-6)))
        fd.append(V._get(dev, reg, "QE_n"))
    return {"n": n, "nc": nc, "imref": imref, "row": row, "jac": jac,
            "jac_fd": (fd[0] - fd[1]) / (2e-6 * n), "jpsi": V._get(dev, reg, "QE_n:Potential")}


def test_closure_row_fd_activity_and_jacobian():
    """G1-9 + the Jacobian unit gate. The FD activity of the row is the heterojunction module's closed
    form (setup_log_activity, reused -- plan 5.6); against the INDEPENDENT numpy Gauss-Legendre
    quadrature physics_bipolar_dd.fd_potential_shift the quasi-Fermi level -V_t[ln(n/n_i) + Delta(u)]
    (psi = 0) agrees to <= 1e-12 relative of V_t ln(n/n_i) + V_t Delta, u in [1e-6, 136] (budget:
    the closed form's measured 2.7e-12 ABSOLUTE at u = 1e5, i.e. far below 1e-12 relative here).
    Jacobian: dR/dn = V_t g(u)/n against central finite differences (step 1e-6 n) to 1e-7 relative
    (truncation 1e-12, round-off ~V_t eps/1e-6 n); dR/dpsi = -1 exactly."""
    out = V.mos_qe(NA, (0.0,), fd=True, tag="row", probe=_probe_rows)[0]["probe"]
    n, u = out["n"], out["n"] / out["nc"]
    ref = -V_T * (np.log(n / V.NI_SI) + BP.fd_potential_shift(u))
    assert np.max(np.abs(out["imref"] - ref) / np.abs(ref)) <= 1e-12
    assert np.max(np.abs(out["jac"] / out["jac_fd"] - 1.0)) <= 1e-7
    assert np.all(out["jpsi"] == -1.0)


def _sigma(dev):
    """(equilibrated sigma_min, sigma_max, sigma_min after the quasi-Fermi change of variables) of
    DEVSIM's DC Jacobian (the 4.4.2 tooling: well_posedness._assemble / _ruiz)."""
    from dynameta.carriers import well_posedness as WP
    J, _, _ = WP._assemble()
    A = J.toarray()

    def eq_svd(M):
        _, _, Me, _ = WP._ruiz(__import__("scipy.sparse", fromlist=["csc_matrix"]).csc_matrix(M))
        s = np.linalg.svd(Me.toarray(), compute_uv=False)
        return s[-1], s[0]

    smin, smax = eq_svd(A)
    out = {"smin": smin, "smax": smax}
    eqs = set(ds.get_equation_list(device=dev, region="s"))
    if "ElectronContinuityEquation" in eqs and ds.get_equation_command(
            device=dev, region="s", name="ElectronContinuityEquation").get("edge_model"):
        num = (lambda v: np.asarray(ds.get_equation_numbers(device=dev, region="s", variable=v),
                                    dtype=int))
        ip, i_n, ih = num("Potential"), num("Electrons"), num("Holes")
        n, p = V._get(dev, "s", "Electrons"), V._get(dev, "s", "Holes")
        T = np.eye(A.shape[0])                     # (psi, n, p) -> (psi, phi_n, phi_p), chain rule
        T[i_n, ip], T[i_n, i_n] = n / V_T, -n / V_T
        T[ih, ip], T[ih, ih] = -p / V_T, p / V_T
        out["smin_qfl"] = eq_svd(A @ T)[0]
    return out


def test_closure_removes_the_near_null_mode():
    """G1-0, the diagnosis turned into a gate (P4 1.1): classical 1-D MOS (3e23 p-Si, the P4 fixture:
    150 nm, 268 nodes, 0.05 nm at the wall) at 0.70 V of band bending, equilibrated (Ruiz) dense SVD of DEVSIM's DC
    Jacobian. MEASURED: full sigma_min 2.6e-16 (sigma_max 3.9; P4: 2.8e-16), the minority-electron
    near-null mode; electrons closed: 5.1e-8 -- closed / full = 1.9e8 >= 1e8 (the plan gate). The
    remaining weak mode (diagnose_singular: "mode in Holes, x in [15, 87] nm") is the HOLE block of the
    depletion region; with BOTH carriers closed (the MOS capacitor's equilibrium) sigma_min = 1.46e-4,
    cond 1.5e4 <= 1e6 (P4's 1.45e-4 / 1.4e4, which is therefore the both-closed system). Pinned as NO
    HELP: the quasi-Fermi change of variables raises the full model's sigma_min by < 10x (MEASURED 1.7x;
    P4 0.65x) -- a future "Slotboom" proposal must beat this. FAILS on the old code (only the full
    model exists)."""
    kw = dict(mesh_nb="p4", length=150e-9, probe=_sigma)        # the P4 fixture: 268 nodes, 804 dof
    full = V.mos_qe(NA, (0.70,), closure=False, step=0.02, tag="svdf", **kw)[0]["probe"]
    closed = V.mos_qe(NA, (0.70,), tag="svdc", **kw)[0]["probe"]
    both = V.mos_qe(NA, (0.70,), tag="svdb", close=("electrons", "holes"), **kw)[0]["probe"]
    assert closed["smin"] / full["smin"] >= 1e8, (closed, full)
    assert both["smax"] / both["smin"] <= 1e6, both
    assert full["smin_qfl"] / full["smin"] < 10.0


@pytest.fixture(scope="module")
def qs_meshes():
    """The classical closure on three NESTED meshes (N = 3000 / 6000 / 12000 segments, beta = ln 1e3:
    first spacing 6.9e-4 / 3.5e-4 / 1.7e-4 nm) at the plan's band bendings."""
    deltas = (0.2, 0.6, 0.9, 1.1, 1.3)
    beta = float(np.log(1e3))
    return deltas, [V.mos_qe(NA, deltas, mesh_nb=(nseg, beta), tag="qs%d" % nseg)
                    for nseg in (3000, 6000, 12000)]


def test_classical_closure_reproduces_the_analytic_surface_charge(qs_meshes):
    """G1-1: the classical (Boltzmann) closure's semiconductor charge sum q (p - n - N_A) NodeVolume
    against the analytic F-function Q_s (_qs_boltzmann) at 0.2-1.3 V: observed order >= 1.8 (the box
    method is O(h^2 / l^2), l the inversion-layer thickness) and the Richardson limit within 1e-5 of
    the analytic value. MEASURED: order 2.00 at every bias; Richardson 1e-14..2e-13 (<= 1.1 V) and
    1.8e-5 at 1.3 V on 1500/3000/6000 -- hence the finer meshes here (at 1.3 V the Boltzmann layer is
    l = V_t/E_s = 2e-12 m thick: n_s = 2.3e30 m^-3, a numerics regime, not physics). On the old code
    the full model does not converge past 0.78-0.80 V, so 0.9-1.3 V cannot be computed at all."""
    deltas, runs = qs_meshes
    for i, d in enumerate(deltas):
        q = [abs(r[i]["Q"]) for r in runs]
        e = [abs(x / _qs_boltzmann(d) - 1.0) for x in q]
        orders = [np.log2(e[k] / e[k + 1]) for k in range(2)]
        rich = q[2] + (q[2] - q[1]) / 3.0
        assert min(orders) >= 1.8, (d, e, orders)
        assert abs(rich / _qs_boltzmann(d) - 1.0) <= 1e-5, (d, rich)


def test_charge_bookkeeping_is_gauss_exact(qs_meshes):
    """G1-1's refutation half: the charge the gate measures cannot hide a bookkeeping error -- DEVSIM's
    two contact charges (the PotentialEdgeFlux of the wall and body contacts) balance the interior
    charge sum, sum_interior rho V + Q_wall + Q_sub = 0, to the round-off of the sum: <= 4 N eps
    sum |rho V| (N nodes; MEASURED <= 1.8e-12 |Q| = 1.3 N eps at N = 6000). (The wall node's own
    half-cell charge rho_0 V_0 is what makes the FULL sum O(h^2)-accurate for Q_s, unlike the edge flux
    alone, which is E at h/2: O(h).) Perturbation that must break G1-1: permittivity x (1 + 1e-4) in the
    oracle moves Q_s by 5e-5, five times the Richardson budget."""
    deltas, runs = qs_meshes
    for r in runs:
        for rec in r:
            nn = rec["x"].size
            bal = rec["Q_interior"] + rec["Q_gauss"]["wall"] + rec["Q_gauss"]["sub"]
            assert abs(bal) <= 4.0 * nn * 2.22e-16 * abs(rec["Q"]), (rec["delta"], bal)
    i = deltas.index(0.6)
    q = [abs(r[i]["Q"]) for r in runs]
    rich = q[2] + (q[2] - q[1]) / 3.0
    assert abs(rich / (_qs_boltzmann(0.6) * np.sqrt(1.0 + 1e-4)) - 1.0) > 1e-5


@pytest.mark.slow                    # ~90 s (11 extended-precision MOS ramps); nightly
def test_closure_equals_the_full_model_where_both_converge():
    """G1-2 (and G1-2b): the zero-current SG relation n1/n0 = exp(-v) of the full model IS the closure's
    ratio on every edge (Boltzmann), so where the full model converges both are the SAME discrete
    solution -- <= 1e-10 relative in n, p (and Lambda_n, to its max) at every node. Extended Newton,
    REL 1e-10 (both). Classical: 0.1, 0.4, 0.7 V; DG (gamma = 1 on the electrons of the bipolar
    region, insulator wall, installed at each bias after a classical ramp -- the D2d GATE E order;
    the FULL DG model's own psi ramp fails, "Matrix solve failed" on the way to 0.6 V, MEASURED, with or
    without the hole DG): 0.1, 0.35, 0.6 V. The full model with tau = 1e-7 and 1e-5 s: equilibrium does not depend on
    tau (the closure has none). The plan's tau = 1e3 s is not usable, and with tau = 1e-5 s only the
    first two biases: the full classical model then FAILS to converge on the ramp to 0.7 V even in
    extended precision (MEASURED, 40 iterations) -- the weaker the SRH coupling, the weaker the
    inversion layer's tie to the body, i.e. the near-null mode this closure removes."""
    worst = {}
    cases = ([("cl", d, {}) for d in (0.1, 0.4, 0.7)]
             + [("dg", d, {"dg_gamma": 1.0, "dg_at_target": True}) for d in (0.1, 0.35, 0.6)])
    for tag, d, kw in cases:
        closed = V.mos_qe(NA, (d,), ext=True, rel=1e-10, step=0.025, tag=tag + "c", **kw)
        for tau in (1e-7, 1e-5):
            if tau > 1e-6 and d > 0.45:                       # the weakly coupled full model: <= 0.4 V
                continue
            full = V.mos_qe(NA, (d,), ext=True, rel=1e-10, step=0.025, closure=False, tau=tau,
                            tag="{}f{:g}".format(tag, tau), **kw)
            for a, b in zip(closed, full):
                m = max(float(np.max(np.abs(a[k] / b[k] - 1.0))) for k in ("n", "p"))
                if "lam_n" in a:
                    m = max(m, float(np.max(np.abs(a["lam_n"] - b["lam_n"])) /
                                     np.max(np.abs(b["lam_n"]))))
                worst[(tag, tau, a["delta"])] = m
    assert max(worst.values()) <= 1e-10, worst


@pytest.mark.parametrize("ext", [False, True], ids=["double", "extended"])
def test_dg_closure_reaches_strong_inversion_in_few_newton_iterations(ext):
    """G1-3 (one configuration here; the gamma x doping matrix is validation GATE G): bipolar DG,
    gamma_n = gamma_p = 1, 3e23 p-Si, insulator wall, BOTH carriers closed to the body contact (a MOS
    capacitor's DC state is equilibrium for holes too; with only the electrons closed the hole DG row
    fails at 0.5 V on a psi ramp -- MEASURED, the hole density at the depleted surface spans 34
    decades), psi_s - psi_b from 0 to 1.3 V in 50 mV steps: every step converges in <= 12 Newton
    iterations, in double (REL 1e-5, the solve_dc default; REL 1e-7 is below the Lambda tail's
    round-off, D2d sec. 1.6) and extended (REL 1e-7). MEASURED: max 10 / 10. Detailed balance (G1-6):
    both quasi-Fermi levels flat to 1e-12 V. On the old code: the full DG fails at 0.70 V."""
    deltas = tuple(np.round(np.arange(0.05, 1.3001, 0.05), 10))
    r = V.mos_qe(NA, deltas, dg_gamma=1.0, dg_holes=True, close=("electrons", "holes"), ext=ext,
                 rel=1e-7 if ext else 1e-5, tag="g13" + ("x" if ext else "d"),
                 probe=lambda dev: (np.ptp(V._get(dev, "s", "QE_n_imref")),
                                    np.ptp(V._get(dev, "s", "QE_p_imref"))))
    its = [max(rec["its"]) for rec in r]
    assert max(its) <= 12, its
    assert max(max(rec["probe"]) for rec in r) <= 1e-12
    n_inv = V.mos_sheet(r[-1])[0]
    assert n_inv > 1e16                                  # strong inversion reached (1.25e17 m^-2)


def test_refusals_and_round_trip():
    """The closure is not a transport model: (a) a pn diode (two ohmic contacts pinning electrons)
    refuses it unless allow_imref_split=True, and with it the +0.4 V forward current is wrong by more
    than 100 % (MEASURED 18x) (the refutation of 'a silent transport substitute', plan G1-8); (b) setup_bipolar_dg
    after the closure refuses (it would bolt the DG flux onto the algebraic row); (c) a second closure
    of the same carrier and a non-ohmic reference contact refuse; (d) delete_quasi_equilibrium_carrier
    restores the recorded continuity equation call for call and the full model re-solves to the same
    state."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_density_gradient as DG
    dev, mesh = "qe_diode", "qe_diode_m"
    try:
        reg = V._diode(dev, mesh)
        V._ramp_bias(dev, 0.4)
        i_full = V._terminal(dev, "anode")
        with pytest.raises(ValueError, match="allow_imref_split"):
            BP.setup_quasi_equilibrium_carrier(dev, reg, "electrons", contact="cathode")
        before = _R.recorded(dev, "ElectronContinuityEquation", scope="region", loc=reg)
        BP.setup_quasi_equilibrium_carrier(dev, reg, "electrons", contact="cathode",
                                           allow_imref_split=True)
        with pytest.raises(ValueError, match="already closed"):
            BP.setup_quasi_equilibrium_carrier(dev, reg, "electrons", contact="cathode",
                                               allow_imref_split=True)
        with pytest.raises(ValueError, match="BEFORE the closure"):
            DG.setup_bipolar_dg(dev, reg, m_eff_n_kg=V.MN)
        assert BP.quasi_equilibrium_carriers(dev, reg) == ("electrons",)
        V._solve()
        i_qe = V._terminal(dev, "anode")
        assert abs(i_qe / i_full - 1.0) > 1.0, (i_qe, i_full)    # MEASURED 18x (307 vs 16.7 A/m^2)
        BP.delete_quasi_equilibrium_carrier(dev, reg, "electrons")
        assert _R.recorded(dev, "ElectronContinuityEquation", scope="region", loc=reg) == before
        assert BP.quasi_equilibrium_carriers(dev, reg) == ()
        V._solve()
        assert V._terminal(dev, "anode") == pytest.approx(i_full, rel=1e-6)
        with pytest.raises(ValueError, match="not closed"):
            BP.delete_quasi_equilibrium_carrier(dev, reg, "electrons")
        with pytest.raises(ValueError, match="'electrons' or 'holes'"):
            BP.setup_quasi_equilibrium_carrier(dev, reg, "excitons", contact="cathode")
    finally:
        V._drop(dev, mesh)


_SSAC_CHILD = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import validation.bipolar_density_gradient as V
from dynameta.carriers import ac_analysis as AC
from dynameta.carriers import noise_analysis as NZ
ds = V.ds
out = {}
def probe(dev):
    # re-drive the wall from a circuit node at the same DC potential; ssac at 1 Hz; DC dQ/dV by
    # +-1 mV central differences of the wall contact charge; then noise must refuse
    from dynameta.carriers import eq_registry as R
    v0 = float(ds.get_parameter(device=dev, name="wall_bias"))
    ds.delete_contact_equation(device=dev, contact="wall", name="PotentialEquation")
    R.forget(dev, "PotentialEquation", loc="wall", scope="contact")
    AC.setup_circuit_contact(dev, "wall", source_name="Vw", node_name="vw")
    def dc(v):
        ds.circuit_alter(name="Vw", value=v)
        V._solve()
        return float(ds.get_contact_charge(device=dev, contact="wall", equation="PotentialEquation"))
    q0 = dc(v0)
    _, C, G = AC.ssac_admittance([1.0], source_name="Vw")
    d1 = (dc(v0 + 1e-3) - dc(v0 - 1e-3)) / 2e-3
    d2 = (dc(v0 + 5e-4) - dc(v0 - 5e-4)) / 1e-3
    dc(v0)
    out["C_ssac"], out["G_ssac"] = float(C[0]), float(G[0])
    out["C_dc"], out["C_dc1"] = (4.0 * d2 - d1) / 3.0, d1
    try:
        NZ.noise_spectrum(dev, [1.0], output_node="Vw.I")
        out["noise"] = "accepted"
    except NotImplementedError as e:
        out["noise"] = str(e)
    return q0
V.mos_qe(3e23, (1.0,), ext=True, rel=1e-10, tag="ssac", probe=probe)
json.dump(out, open(sys.argv[2], "w"))
'''


def test_ssac_is_the_low_frequency_cv_and_noise_refuses(tmp_path):
    """ssac IS allowed on a closed device (the row has no time term: the minority carrier follows
    instantaneously -- the low-frequency / quasi-static C-V, plan G1-5's cross-method gate): at 1.0 V of
    band bending (strong inversion, classical, extended Newton, REL 1e-10) the ssac capacitance at 1 Hz
    equals the DC dQ/dV of the wall charge to 1e-6 relative. dQ/dV: Richardson of the +-1 mV and
    +-0.5 mV central differences (the 1 mV difference alone carries (dV/2V_t)^2/6 = 6.2e-5 of curvature
    for Q ~ exp(delta/2V_t), and is asserted to MISS -- the refutation; after Richardson O(dV^4) ~ 1e-9,
    solve noise ~1e-10). noise_analysis REFUSES the closed region (NotImplementedError: its equation has
    no Scharfetter-Gummel edge model). Runs in a child process (DEVSIM circuits cannot be deleted
    in-process)."""
    out = tmp_path / "ssac.json"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root)
    subprocess.run([sys.executable, "-c", _SSAC_CHILD, root, str(out)], check=True, env=env,
                   capture_output=True, timeout=600)
    import json
    r = json.loads(out.read_text())
    assert r["C_ssac"] == pytest.approx(r["C_dc"], rel=1e-6), r
    assert abs(r["C_ssac"] / r["C_dc1"] - 1.0) > 1e-5, r     # the bare 1 mV difference must miss
    assert "noise sources are defined only" in r["noise"], r
