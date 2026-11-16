"""The in-Newton density gradient for EQUILIBRIUM regions (carriers.physics_equilibrium_dg, v0.16): the
2-D LayeredDevsimBuilder(density_gradient=...) and Stacked3DSpec(density_gradient=...) paths.

Oracle: the dual-gate Schrodinger-Poisson solver (carriers.sp_dual_gate, an independent method: sub-band
eigenstates, hard walls, Gauss's law at both faces). Fixture: validation/_eqdg_fixture.py (here a 6 nm film, n_bg
3e26 m^-3, m = 0.30 m_e, eps_s 9.0, between 6 nm eps 20 + 2 nm eps 9 gate stacks). Every gate below is new in
v0.16: before it, LayeredDevsimBuilder has no `density_gradient` argument (TypeError) and
physics_equilibrium_dg does not exist."""
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("devsim")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation import _eqdg_fixture as FX                    # noqa: E402

from dynameta.constants import EPS0, M_E, Q_E                # noqa: E402
from dynameta.core.numerics import trapz                     # noqa: E402
from dynameta.carriers.density_gradient import EquilibriumDensityGradient   # noqa: E402
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder           # noqa: E402
from dynameta.carriers import physics_equilibrium_dg as EQDG               # noqa: E402
from dynameta.carriers.devsim_env import import_devsim       # noqa: E402
from dynameta.sweep import BiasPoint                          # noqa: E402

ds = import_devsim()
M = 0.30 * M_E
N_BG = 3e26
TF = 6e-9


def _builder(t=TF, n_bg=N_BG, m=M, ox="std", dg=True, gamma=None, name="eqdg", **kw):
    dkw = kw.pop("design_kw", {})
    spec = {"density_gradient": {"film": EquilibriumDensityGradient(gamma=gamma, m_eff_kg=m)}} if dg else {}
    spec.update(kw)
    return LayeredDevsimBuilder(FX.design(t, n_bg, m, ox, **dkw), device_name=name, mesh_name=name + "_m", **spec)


def _solve_column(b, vb, vt, **solve_kw):
    b.solve(BiasPoint({"bot": vb, "top": vt}, "p"), **solve_kw)
    return FX.column(b.device)


# ---- formulation gates ---------------------------------------------------------------------------------------

def test_gamma_zero_and_no_request_are_the_classical_device_bit_for_bit():
    """Bit-identical default: density_gradient=None makes no new DEVSIM call (the pre-edge-reference
    call-log digests in tests/test_layered_edge_reference.py gate that); an explicit EquilibriumDensityGradient(gamma=0)
    installs nothing either, so its solved arrays equal the classical device's exactly."""
    out = []
    for kw in ({"dg": False}, {"dg": True, "gamma": 0.0}):
        b = _builder(name="g0_{}".format(len(out)), **kw)
        try:
            z, n = _solve_column(b, 0.0, 2.0)
            assert "QSqrtN" not in ds.get_node_model_list(device=b.device, region="film")
            out.append((z, n, FX.column(b.device, name="Potential")[1]))
        finally:
            b.teardown()
    assert np.array_equal(out[0][1], out[1][1]) and np.array_equal(out[0][2], out[1][2])


def test_gauss_law_gate_charges_balance_the_film_charge():
    """Consistency check (box-method Gauss law): the two gate charges reported by DEVSIM (get_contact_charge, the
    contacts' edge_charge_model) plus the film's charge q sum_i V_i (N_D - n_i) vanish, per unit area, at
    accumulation, flat band and depletion. MEASURED <= 2.3e-14 of the gate charge."""
    b = _builder(name="gauss")
    try:
        W = b.design.unit_cell.period_x_m
        for vb, vt in ((0.0, 3.0), (0.0, 0.0), (0.0, -3.0), (2.0, 2.0)):
            b.solve(BiasPoint({"bot": vb, "top": vt}, "p"))
            vol = np.asarray(ds.get_node_model_values(device=b.device, region="film", name="NodeVolume"))
            n = np.asarray(ds.get_node_model_values(device=b.device, region="film", name="Electrons"))
            q_film = Q_E * float(np.sum(vol * (N_BG - n))) / W
            q_g = [ds.get_contact_charge(device=b.device, contact=c, equation="PotentialEquation") / W
                   for c in ("bot", "top")]
            scale = max(abs(q_g[0]), abs(q_g[1]), 1e-3)
            assert abs(q_g[0] + q_g[1] + q_film) <= 1e-9 * scale, (vb, vt, q_g, q_film)
    finally:
        b.teardown()


def _newton_history(device, *, max_it=8):
    info = ds.solve(type="dc", solver_type="direct", absolute_error=1e-30, relative_error=1e-30,
                    maximum_iterations=max_it, info=True)
    hist = []
    for it in info["iterations"]:
        e = {}
        for dv in it["devices"]:
            for rg in dv["regions"]:
                if rg["name"] == "film":
                    for eq in rg["equations"]:
                        e[eq["name"]] = eq["absolute_error"]
        hist.append(e)
    return hist


def test_newton_converges_quadratically_with_the_analytic_jacobian():
    """From a converged DG state at +2 V, step the top gate to +2.1 V and record the film's Newton updates: the
    potential update must fall quadratically (each step <= 10 x the square of the previous one, in volts, down
    to the 1e-14 V round-off floor), reaching < 1e-13 V within 6 iterations. CONTROL: with Poisson's Lambda
    derivative zeroed (PotentialNodeCharge:QLambda = 0, a wrong Jacobian) the same step converges only linearly
    (it is still above 1e-10 V after the same iterations). MEASURED: 1.8e-2 -> 2.8e-5 -> 1.4e-10 -> 5.6e-17 V (3
    iterations); control 1.6e-2 -> 1.5e-3 -> 1.8e-4 -> 2.2e-5 V."""
    b = _builder(name="quad")
    try:
        b.solve(BiasPoint({"bot": 0.0, "top": 2.0}, "p"))
        state = {v: list(ds.get_node_model_values(device=b.device, region="film", name=v))
                 for v in ("Potential", "QSqrtN", "QLambda")}
        ds.set_parameter(device=b.device, name="top_bias", value=2.1)
        h = [e["PotentialEquation"] for e in _newton_history(b.device)]
        k_end = next(i for i, e in enumerate(h) if e < 1e-13)
        assert k_end <= 6, h
        for a, c in zip(h[:k_end], h[1:k_end + 1]):
            if a > 1e-7:
                assert c <= 10.0 * a * a / 1e0 or c < 1e-13, h      # quadratic (volts)
        # control: a wrong Jacobian (the Lambda coupling of the closure dropped) converges linearly
        ds.set_parameter(device=b.device, name="top_bias", value=2.0)
        for v, vals in state.items():
            ds.set_node_values(device=b.device, region="film", name=v, values=vals)
        ds.node_model(device=b.device, region="film", name="PotentialNodeCharge:QLambda", equation="0")
        ds.set_parameter(device=b.device, name="top_bias", value=2.1)
        hc = [e["PotentialEquation"] for e in _newton_history(b.device, max_it=k_end + 1)]
        print("analytic", h, "control", hc)
        assert hc[k_end] > 1e-10, (h, hc)
    finally:
        b.teardown()


def test_refusals():
    """Kane band, drift-diffusion region, unknown region, gamma < 0: clear errors before any solve."""
    from dynameta.materials.transport_model import KaneBand
    with pytest.raises(ValueError, match="gamma"):
        EquilibriumDensityGradient(gamma=-1.0)
    with pytest.raises(ValueError, match="not a semiconductor region"):
        LayeredDevsimBuilder(FX.design(TF, N_BG, M), device_name="rf0", mesh_name="rf0_m",
                             density_gradient={"up0": EquilibriumDensityGradient()}).build_device()
    with pytest.raises(ValueError, match="EquilibriumDensityGradient"):
        LayeredDevsimBuilder(FX.design(TF, N_BG, M), density_gradient={"film": 1.7})
    d = FX.design(TF, N_BG, M, band=KaneBand(m0_kg=0.25 * M_E, alpha_per_eV=0.4))
    b = LayeredDevsimBuilder(d, device_name="rf1", mesh_name="rf1_m",
                             density_gradient={"film": EquilibriumDensityGradient(m_eff_kg=M)})
    try:
        with pytest.raises(NotImplementedError, match="Kane band"):
            b.build_device()
    finally:
        b.teardown()


def test_abs_tol_in_volts_is_refused_on_a_dg_device_before_any_newton():
    """solve(abs_tol=...) is documented in volts for a Potential-only device, but DEVSIM applies the one
    absolute error to every equation, including the sqrt(n) rows of the density gradient (sqrt(3e26) = 1.7e13
    m^-3/2: one unit round-off is ~2e-3). abs_tol=1e-14 used to run all 60 Newton iterations and fail
    (QSqrtNEquation abs ~7e-3 at relative ~3e-16, MEASURED on a gated degenerate film); it is now a ValueError naming the cause, raised before the
    seed solve (no Newton is attempted). The default abs_tol=None still converges on the same device."""
    b = _builder(name="eqdg_abs")
    try:
        with pytest.raises(ValueError, match="QSqrtNEquation"):
            b.solve(BiasPoint({"bot": 0.0, "top": 1.0}, "p"), abs_tol=1e-14)
        assert getattr(b, "_last_converged_bias", None) is None          # refused before any solve
        b.solve(BiasPoint({"bot": 0.0, "top": 1.0}, "p"))                  # the default converges
        assert b._last_converged_bias is not None
    finally:
        b.teardown()


def test_pinned_edge_electrode_on_a_dg_film_is_refused():
    """An Electrode(pin_potential=True) edge on the film is a Potential Dirichlet contact on the two corner nodes,
    which are also dielectric-wall nodes; its rows overwrote the dead-layer wall there. OLD CODE (MEASURED on a
    gated degenerate film at +2 V, 40 nm cell): it built and converged silently with
    n(face) = n_bg at the corner column (instead of 1e-12 n_bg) and that column's sheet excess ~6x below the cell
    centre's. Now refused in the constructor; the unpinned (Fermi-reference) edge still builds."""
    from dynameta.geometry import Electrode
    dg = {"film": EquilibriumDensityGradient(m_eff_kg=M)}
    for pin in (True, False):
        els = [Electrode("bot", "mirror", "full"), Electrode("top", "gate", "full"),
               Electrode("gnd", "film", "x_lo", role="ground", pin_potential=pin)]
        d = FX.design(TF, N_BG, M, "std", cell=40e-9, x_spacing=4e-9, electrodes=els)
        if pin:
            with pytest.raises(NotImplementedError, match="pin_potential=True edge electrode 'gnd'"):
                LayeredDevsimBuilder(d, density_gradient=dg)
        else:
            LayeredDevsimBuilder(d, density_gradient=dg)          # constructor only, no DEVSIM call


def test_classical_limit_small_gamma_converges_to_the_classical_solve():
    """gamma -> 0: the DG profile approaches the classical one everywhere but at the wall nodes themselves (which
    hold the floor density for any gamma > 0: an O(h) half-cell the mesh refinement removes). Measured on the
    default mesh at +2 V: max |n_DG - n_cl| / n_bg off the wall nodes falls with gamma (1e-2 -> 1e-4 -> 1e-6:
    MEASURED 1.33 / 2.235e-2 / 2.215e-2 -- below gamma 1e-4 the node next to the wall, inside the O(h) half cell,
    sets the maximum), and the sheet difference at gamma = 1e-6 halves (within 30 %) when the face spacing halves
    (MEASURED 1.148e15 -> 5.68e14 m^-2, ratio 0.494)."""
    ref = {}
    for refine in (1.0, 0.5):
        mesh = {"screening_interface_factor": refine / 40.0, "screening_ps_factor": refine / 20.0}
        b = _builder(dg=False, name="cl{}".format(refine), design_kw={"mesh": mesh})
        try:
            ref[refine] = _solve_column(b, 0.0, 2.0)
        finally:
            b.teardown()
    errs, dsheet = [], {}
    for g in (1e-2, 1e-4, 1e-6):
        b = _builder(gamma=g, name="g{}".format(g))
        try:
            z, n = _solve_column(b, 0.0, 2.0)
        finally:
            b.teardown()
        assert np.allclose(z, ref[1.0][0], rtol=0, atol=1e-18)
        errs.append(float(np.max(np.abs(n - ref[1.0][1])[1:-1])) / N_BG)
        if g == 1e-6:
            dsheet[1.0] = float(trapz(ref[1.0][1] - n, z))
    assert errs[0] > errs[1] > errs[2], errs
    mesh = {"screening_interface_factor": 0.5 / 40.0, "screening_ps_factor": 0.5 / 20.0}
    b = _builder(gamma=1e-6, name="g6f", design_kw={"mesh": mesh})
    try:
        z, n = _solve_column(b, 0.0, 2.0)
    finally:
        b.teardown()
    dsheet[0.5] = float(trapz(ref[0.5][1] - n, z))
    assert 0.35 < dsheet[0.5] / dsheet[1.0] < 0.65, dsheet


def test_grid_convergence_of_the_dg_profile():
    """The default screening-length mesh resolves the dead layer: refining every graded spacing by 2 and 4 (face
    spacing 0.0194 / 0.0097 / 0.0050 nm, the last at the builder's 5e-12 m floor) moves the sheet excess and the
    excess centroid (distance from the gate face) by amounts falling by more than 2 per refinement, and the
    default mesh's sheet is within 2e-3 of n_bg t of the finest. MEASURED at +3 V (6 nm, 3e26, gamma 1.8): sheet
    2.12852247 / 2.12849852 / 2.12849285e17 m^-2 (differences 2.4e12, 5.7e11: second order; default vs finest
    1.7e-6 of n_bg t)."""
    vals = []
    for f in (1.0, 0.5, 0.25):
        mesh = {"screening_interface_factor": f / 40.0, "screening_ps_factor": f / 20.0}
        b = _builder(name="gc{}".format(f), design_kw={"mesh": mesh})
        try:
            z, n = _solve_column(b, 0.0, 3.0)
        finally:
            b.teardown()
        exc = float(trapz(n - N_BG, z))
        vals.append((exc, float(trapz((z[-1] - z) * (n - N_BG), z)) / exc))     # sheet, excess centroid
    d1 = [abs(vals[0][k] - vals[1][k]) for k in (0, 1)]
    d2 = [abs(vals[1][k] - vals[2][k]) for k in (0, 1)]
    for a, c in zip(d1, d2):
        assert c < 0.5 * a + 1e-300, (vals, d1, d2)
    assert abs(vals[0][0] - vals[2][0]) / (N_BG * TF) < 2e-3, vals


# ---- integration: charges, Fermi reference, 2-D ----------------------------------------------------------------

def test_fixed_charge_and_fermi_reference_act_on_the_dg_film_like_on_the_classical_one():
    """(a) A sheet charge Q at film/up0 with the top gate shifted by -Q/C_top gives the uncharged DG solution
    (Gauss: the oxide carries the same field) -- n to 1e-9 relative. (b) An edge Fermi-level reference at
    +0.3 V equals the device without it with both gates shifted by -0.3 V: n to 1e-9, potentials shifted by
    exactly 0.3 V (to 1e-9 V). Both are the classical device's identities; the DG rows must not break them."""
    from dynameta.geometry import Electrode
    c_top = 1.0 / sum(t / (EPS0 * e) for t, e in FX.STACKS["std"])
    q = 2e-3
    b0 = _builder(name="fq0")
    b1 = _builder(name="fq1", fixed_charge={("film", "up0"): q})
    try:
        _, n0 = _solve_column(b0, 0.0, 2.0)
        _, n1 = _solve_column(b1, 0.0, 2.0 - q / c_top)
        assert np.max(np.abs(n1 / n0 - 1.0)) < 1e-9
    finally:
        b0.teardown()
        b1.teardown()
    els = [Electrode("bot", "mirror", "full"), Electrode("top", "gate", "full"),
           Electrode("gnd", "film", "x_lo", role="ground", fixed_voltage_V=0.3)]
    b2 = _builder(name="fr", design_kw={"electrodes": els})
    b3 = _builder(name="fr0")
    try:
        _, n2 = _solve_column(b2, 0.0, 2.0)
        assert b2.fermi_references() == {"gnd": ["film"]}
        p2 = FX.column(b2.device, name="Potential")[1]
        _, n3 = _solve_column(b3, -0.3, 1.7)
        p3 = FX.column(b3.device, name="Potential")[1]
        assert np.max(np.abs(n2 / n3 - 1.0)) < 1e-9
        assert np.max(np.abs(p2 - p3 - 0.3)) < 1e-9
    finally:
        b2.teardown()
        b3.teardown()


def test_laterally_uniform_2d_equals_the_one_column_result():
    """A 16 nm cell with x lines every 4 nm (5 node columns) under full gates: every column equals the one-cell
    device's column (same z lines) to 1e-10 relative -- no lateral artefact from the wall mask or the seed."""
    b1 = _builder(name="lu1")
    b4 = _builder(name="lu4", design_kw={"cell": 16e-9, "x_spacing": 4e-9})
    try:
        z1, n1 = _solve_column(b1, 0.0, 3.0)
        b4.solve(BiasPoint({"bot": 0.0, "top": 3.0}, "p"))
        X = np.unique(np.round(np.asarray(ds.get_node_model_values(device=b4.device, region="film", name="x")), 15))
        assert X.size == 5
        for x in X:
            z4, n4 = FX.column(b4.device, x=x)
            assert np.allclose(z4, z1, rtol=0, atol=1e-18)
            assert np.max(np.abs(n4 / n1 - 1.0)) < 1e-10
    finally:
        b1.teardown()
        b4.teardown()


def _patch_device(name, x_spacing, dg=True):
    from dynameta.geometry import Electrode
    from dynameta.geometry.cross_section import Rectangle
    els = [Electrode("bot", "mirror", "full"),
           Electrode("top", "gate", Rectangle(cx_m=20e-9, cy_m=20e-9, width_m=16e-9, height_m=40e-9))]
    return _builder(name=name, dg=dg, design_kw={"cell": 40e-9, "x_spacing": x_spacing, "electrodes": els})


def _sheets(device):
    X = np.unique(np.round(np.asarray(ds.get_node_model_values(device=device, region="film", name="x")), 15))
    out = []
    for x in X:
        z, n = FX.column(device, x=x)
        out.append(float(trapz(n - N_BG, z)))
    return X, np.asarray(out)


def test_laterally_structured_gate_is_symmetric_converged_and_keeps_the_dead_layer():
    """A 16 nm top-gate patch centred in a 40 nm cell at +3 V (bottom gate full, 0 V), x lines every 2 nm and
    1 nm. Physically: accumulation under the patch with the dead layer at the top face (the face node at the
    floor, the peak >= 0.25 nm inside; MEASURED 0.775 nm), the column sheet excess largest at the patch centre and
    falling monotonically to the cell edge (fringing), the profile mirror-symmetric about the patch centre (1e-9
    relative; MEASURED 1.2e-15), and the DG induced sheet below the classical one (the dead layer's series
    capacitance: MEASURED -23.9 % at the centre). x-mesh: the gate-edge corner makes the 2-D electrostatics
    converge at FIRST order in x for the classical and the DG device alike (MEASURED centre sheet, x spacing
    2 / 1 / 0.5 nm: classical 2.5260 / 2.5051 / 2.4953e17, DG 1.9221 / 1.9031 / 1.8942e17 m^-2, differences
    halving); the gate: the DG device's relative change 2 -> 1 nm is within 1.5x the classical device's
    (MEASURED 9.9e-3 vs 8.3e-3), i.e. the density gradient adds no lateral mesh error of its own. The 0.5 nm
    mesh is the @slow companion below."""
    res = {}
    for dg in (False, True):
        for xs in (2e-9, 1e-9):
            b = _patch_device("pt{}{}".format(int(dg), int(xs * 1e10)), xs, dg=dg)
            try:
                b.solve(BiasPoint({"bot": 0.0, "top": 3.0}, "p"))
                X, S = _sheets(b.device)
                zc, nc = FX.column(b.device, x=20e-9)
                res[(dg, xs)] = (X, S, zc, nc)
            finally:
                b.teardown()
    X, S, zc, nc = res[(True, 2e-9)]
    k = int(np.argmin(abs(X - 20e-9)))
    assert np.max(np.abs(S - S[::-1])) < 1e-9 * np.max(np.abs(S))
    assert np.all(np.diff(S[k:]) < 0.0) and S[k] > 0.0
    assert nc[-1] < 1e-9 * N_BG and zc[-1] - zc[int(np.argmax(nc))] > 0.25e-9
    centre = {key: v[1][int(np.argmin(abs(v[0] - 20e-9)))] for key, v in res.items()}
    assert 0.7 < centre[(True, 2e-9)] / centre[(False, 2e-9)] < 0.95
    rel = {dg: abs(centre[(dg, 1e-9)] / centre[(dg, 2e-9)] - 1.0) for dg in (False, True)}
    assert rel[True] < 1.5 * rel[False] and rel[True] < 1e-2, rel


@pytest.mark.slow
def test_laterally_structured_gate_x_mesh_convergence_is_first_order_like_the_classical_device():
    """@slow (~70 s): the 2 / 1 / 0.5 nm x meshes of the patch device above; the DG centre-column sheet
    differences fall by a factor 1.7-2.3 per halving (first order, the gate-edge corner), as the classical
    device's do (MEASURED 2.11 / 2.13 for classical / DG)."""
    for dg in (False, True):
        c = []
        for xs in (2e-9, 1e-9, 0.5e-9):
            b = _patch_device("ps{}{}".format(int(dg), int(xs * 1e11)), xs, dg=dg)
            try:
                b.solve(BiasPoint({"bot": 0.0, "top": 3.0}, "p"))
                X, S = _sheets(b.device)
                c.append(S[int(np.argmin(abs(X - 20e-9)))])
            finally:
                b.teardown()
        ratio = (c[0] - c[1]) / (c[1] - c[2])
        assert 1.7 < ratio < 2.3, (dg, c)


# ---- the frozen-potential first integral: an implementation-independent oracle of the DG rows ---------------
#
# With the potential FROZEN flat (psi = 0; PotentialEquation deleted after the build) the equilibrium DG of a film
# between an insulating wall (z = 0) and an ohmic reservoir reduces to b u'' = Lambda u with the closure
# u^2 = N_c F(eta_b + Lambda/V_t). With t = u/u_b: Lambda(t)/V_t = F^-1(t^2 F(eta_b)) - eta_b, and b u'' = Lambda u
# integrates once to (u')^2 = (2 V_t u_b^2/b) G(t), G(t) = int_t^1 (-Lambda/V_t) tau dtau, so the dark-space
# deficit is int (n_b - n) dz = n_b L int_0^1 (1 - t^2)/sqrt(2 G(t)) dt with L = sqrt(b/V_t). F is the region
# closure (Aymerich-Humet); the quadrature is written here, independent of the DEVSIM rows (closure model, its
# Lambda derivative, the Lambda equation, the wall penalty, the reservoir rows).

def _first_integral_deficit(eta_b):
    from scipy.integrate import quad
    from dynameta.carriers.physics_equilibrium import F12_aymerich_humet, invert_F12
    f_b = F12_aymerich_humet(eta_b)

    def lam(tau):                                       # -Lambda/V_t >= 0 for tau <= 1
        return -(invert_F12(tau * tau * f_b, -200.0, 200.0, 1e-14) - eta_b) if tau < 1.0 else 0.0

    def big_g(t):
        return quad(lambda tau: lam(tau) * tau, t, 1.0, epsabs=0.0, epsrel=1e-11, limit=200)[0]

    return quad(lambda t: (1.0 - t * t) / np.sqrt(2.0 * big_g(t)), 0.0, 1.0, epsabs=0.0, epsrel=1e-9,
                limit=200)[0]


def _frozen_wall_bar(tag, *, n0, h0, gamma=EQDG.EQUILIBRIUM_DG_GAMMA, length=30e-9):
    """1-D: oxide [0, 1 nm] | film [1 nm, 1 nm + length]; contacts 'gate' (oxide end, Potential 0) and 'back'
    (film end: the DG reservoir rows, Potential 0). Film nodes x_k = L (e^{beta k/N} - 1)/(e^beta - 1) from the
    wall (spacing ratio 40, nested when h0 halves). Returns (x from the wall, n, L = sqrt(b/V_t), eta_b)."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_equilibrium as PE
    mesh, dev = "fwm_" + tag, "fwd_" + tag
    t_ox = 1e-9
    beta = np.log(40.0)
    nseg = int(round(length * beta / (h0 * np.expm1(beta))))
    xs = t_ox + length * np.expm1(beta * np.arange(nseg + 1) / nseg) / np.expm1(beta)
    try:
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=0.1e-9, tag="gate")
        gaps = np.diff(xs)
        for k, xk in enumerate(xs):
            kw = {"tag": "i"} if k == 0 else ({"tag": "back"} if k == nseg else {})
            ds.add_1d_mesh_line(mesh=mesh, pos=float(xk), ps=1.01 * float(gaps[min(k, nseg - 1)]), **kw)
        ds.add_1d_contact(mesh=mesh, name="gate", tag="gate", material="metal")
        ds.add_1d_contact(mesh=mesh, name="back", tag="back", material="metal")
        ds.add_1d_region(mesh=mesh, material="ox", region="ox", tag1="gate", tag2="i")
        ds.add_1d_region(mesh=mesh, material="semi", region="film", tag1="i", tag2="back")
        ds.add_1d_interface(mesh=mesh, name="ox_film", tag="i")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        PE.setup_dielectric_region(dev, "ox", 9.0)
        PE.setup_semiconductor_region(dev, "film", n_bg_m3=n0, eps_static=FX.EPS_S, dos_mass_kg=M)
        PE.setup_interface(dev, "ox_film")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # the validity advisory does not apply to a bar
            EQDG.setup_equilibrium_dg(dev, "film", m_eff_kg=M, gamma=gamma)
        PE.setup_contact(dev, "gate")
        PE.setup_contact(dev, "back")
        _R.delete_by_name(dev, "PotentialEquation")     # freeze psi = 0 (flat band)

        def _solve():
            # DEVSIM restores the previous state when a solve does not converge, so the test must converge: its
            # relative test on the bulk Lambda (~0 there) floors near 1e-6 (|dLambda| ~ 1e-16 V), so 1e-5 is
            # met at an update of ~1e-6 V; the second call adds Newton steps from there (quadratic: ~1e-12 V).
            for _ in range(2):
                info = ds.solve(type="dc", absolute_error=1e30, relative_error=1e-5, maximum_iterations=60,
                                info=True)
                assert info["converged"], tag

        EQDG.set_equilibrium_dg_fraction(dev, "film", 0.05)
        EQDG.seed_equilibrium_dg(dev, "film")
        EQDG.adaptive_continuation(lambda f: EQDG.set_equilibrium_dg_fraction(dev, "film", f), 0.05, 1.0, _solve,
                                   step=0.25)
        x = np.asarray(ds.get_node_model_values(device=dev, region="film", name="x"))
        n = np.asarray(ds.get_node_model_values(device=dev, region="film", name="Electrons"))
        o = np.argsort(x)
        b = float(ds.get_parameter(device=dev, region="film", name="b_dg_full"))
        v_t = float(ds.get_parameter(device=dev, region="film", name="V_t"))
        eta_b = -float(ds.get_parameter(device=dev, region="film", name="Phi_c0")) / v_t
        return x[o] - t_ox, n[o], float(np.sqrt(b / v_t)), eta_b
    finally:
        _R.clear(dev)
        if dev in ds.get_device_list():
            ds.delete_device(device=dev)
        if mesh in ds.get_mesh_list():
            ds.delete_mesh(mesh=mesh)


@pytest.mark.parametrize("n0", [3e26, 1e27])
def test_frozen_potential_dark_space_converges_to_the_first_integral(n0):
    """The DG rows against the first integral above (frozen flat potential, wall + reservoir), at the default
    gamma: nested meshes h0 = 0.02 / 0.01 / 0.005 nm converge at observed order >= 1.8 and the Richardson value of
    the last two lies within 1e-5 of the integral. Sign-sensitive: a wrong sign of Lambda (or of the closure
    shift) would ACCUMULATE at the wall instead (negative deficit). MEASURED 2026-11-15 (m = 0.30 m_e): 3e26 m^-3
    (eta_b 21.05): integral 0.3572160 n_b L, errors 6.4e-5 / 1.5e-5 / 3.1e-6 (orders 2.07 / 2.31), Richardson
    -9.6e-7; 1e27 (eta_b 47.01): 0.2396546, errors 3.4e-5 / 7.8e-6 / 1.2e-6 (2.13 / 2.67), Richardson -9.6e-7."""
    exact = None
    d = []
    for k, h in enumerate((0.02e-9, 0.01e-9, 0.005e-9)):
        x, n, L, eta_b = _frozen_wall_bar("{}_{}".format(int(n0 / 1e26), k), n0=n0, h0=h)
        if exact is None:
            exact = _first_integral_deficit(eta_b)
        nb = n[-1]
        d.append(float(trapz(nb - n, x)) / (nb * L))
    e = [abs(v / exact - 1.0) for v in d]
    orders = [np.log2(e[k] / e[k + 1]) for k in range(2)]
    assert exact > 0.0 and min(orders) >= 1.8, (exact, d, e)
    rich = d[2] + (d[2] - d[1]) / 3.0
    assert abs(rich / exact - 1.0) <= 1e-5, (exact, rich)


def test_thomas_fermi_wall_limit_gives_the_a_priori_gamma():
    """Pure math (the EQUILIBRIUM_DG_GAMMA comment): the T -> 0 Fermi-Dirac DG dark space at a hard wall is
    sqrt(gamma/3) I_TF / k_F with I_TF = int_0^1 (1 - w^2)/sqrt(3 w^(10/3)/5 - w^2 + 2/5) dw = 1.64378, so the free-gas
    Friedel deficit 3 pi/(8 k_F) fixes gamma_TF = 1.541; the finite-T first integral above tends to that limit
    (relative difference 2.5e-3 at eta 21.0, 3.0e-4 at eta 47.0, falling ~ 1/eta^2). The calibrated default lies
    within 20 % of gamma_TF."""
    from scipy.integrate import quad
    from dynameta.constants import HBAR
    from dynameta.carriers.physics_equilibrium import V_T, effective_dos_m3, invert_F12
    i_tf = quad(lambda w: (1 - w * w) / np.sqrt(0.6 * w ** (10.0 / 3.0) - w * w + 0.4), 0.0, 1.0, limit=200,
                epsabs=1e-13, epsrel=1e-12)[0]
    assert i_tf == pytest.approx(1.6437820, abs=1e-6)
    g_tf = 3.0 * (3.0 * np.pi / (8.0 * i_tf)) ** 2
    assert g_tf == pytest.approx(1.5410, abs=1e-4)
    assert 1.0 < EQDG.EQUILIBRIUM_DG_GAMMA / g_tf < 1.2
    rel = []
    for n0 in (3e26, 1e27):
        eta_b = invert_F12(n0 / effective_dos_m3(M))
        e_f = HBAR ** 2 * (3.0 * np.pi ** 2 * n0) ** (2.0 / 3.0) / (2.0 * M) / Q_E
        rel.append(abs(_first_integral_deficit(eta_b) / (i_tf * np.sqrt(V_T / e_f)) - 1.0))
    assert rel[0] < 3e-3 and rel[1] < 0.3 * rel[0], rel


# ---- against the dual-gate Schrodinger-Poisson oracle (the calibration's reference) -----------------------------

def test_default_gamma_matches_dual_gate_schrodinger_poisson_on_a_6nm_film():
    """6 nm, 3e26 m^-3, 0.30 m_e, 6 nm eps 20 + 2 nm eps 9 on both faces (inside the validity range), default gamma,
    default mesh, vs DualGateSchrodingerPoisson (h = 0.0125 nm): sheet-excess error <= 0.3 % of the carrier sheet
    and profile L1 <= 0.10 at depletion / flat band / accumulation, each >= 10x / 3x better than the classical
    model; the accumulation peak off the gate face by the SP's distance within 0.15 nm. MEASURED 2026-11-15:
    dsheet +0.063 / +0.019 / -0.020 %, L1 0.0541 / 0.0618 / 0.0661 at (0,-3) / (0,0) / (0,+3) V; classical +1.87 /
    +2.68 / +3.04 % and 0.1626 / 0.2390 / 0.2878 (the L1 ratio at (0,-3) V is 3.005, just above the 3x bar); peak
    0.775 nm off the gate face vs SP 0.850 nm."""
    sp = FX.oracle(TF, N_BG, M)
    out = {}
    for dg in (True, False):
        b = _builder(dg=dg, name="sp{}".format(int(dg)))
        try:
            for vb in ((0.0, -3.0), (0.0, 0.0), (0.0, 3.0)):
                z, n = _solve_column(b, *vb)
                out[(dg, vb)] = FX.compare(z, n, sp.solve_voltages(*vb), N_BG, TF)
        finally:
            b.teardown()
    for vb in ((0.0, -3.0), (0.0, 0.0), (0.0, 3.0)):
        d, c = out[(True, vb)], out[(False, vb)]
        assert abs(d["dsheet"]) <= 3e-3 and d["l1"] <= 0.10, (vb, d)
        assert abs(c["dsheet"]) >= 10.0 * abs(d["dsheet"]) and c["l1"] >= 3.0 * d["l1"], (vb, d, c)
    d = out[(True, (0.0, 3.0))]
    assert abs((TF - d["zpeak"]) - (TF - d["zpeak_sp"])) <= 0.15e-9, d
    assert TF - d["zpeak"] > 0.3e-9                     # the peak is pulled off the gate face


def test_validity_guard_warns_for_a_film_with_fewer_than_two_subbands():
    """check_equilibrium_dg_validity at build: a 3 nm film at 1e26 m^-3 (t k_F/pi = 1.37 < 2) with the default gamma
    warns UnreliableResultWarning; the generic 6 nm 3e26 film (3.96, eta 21.05) does not (filterwarnings = error)."""
    from dynameta.warning_types import UnreliableResultWarning
    b = _builder(t=3e-9, n_bg=1e26, name="vg0")
    try:
        with pytest.warns(UnreliableResultWarning, match="outside the calibrated range"):
            b.build_device()
        reg = EQDG.equilibrium_dg_regime(b.device, "film")
        # t_eff = 2 x the largest node-to-wall distance: the centre node sits within one cell of the mid-plane
        assert reg["subbands"] == pytest.approx(1.37, abs=0.03) and reg["t_eff_m"] == pytest.approx(3e-9, rel=0.02)
    finally:
        b.teardown()
    b = _builder(name="vg1")
    try:
        b.build_device()
        reg = EQDG.equilibrium_dg_regime(b.device, "film")
        assert reg["subbands"] == pytest.approx(3.96, abs=0.05) and reg["eta_bg"] == pytest.approx(21.05, abs=0.01)
    finally:
        b.teardown()


def test_validity_guard_warns_for_a_user_gamma_too():
    """The guard ran only when gamma == EQUILIBRIUM_DG_GAMMA, so a user gamma (1.7 here) on the same out-of-range
    3 nm / 1e26 m^-3 film built silently (OLD CODE: no warning -> this test fails at pytest.warns). The range is the
    model's: outside it the per-point matching gamma spreads over 1.36 .. 2.40 (the calibration notes in
    docs/audit/2026-11-14-v0.16-equilibrium-density-gradient.md), so no constant gamma is validated there. A user gamma inside the range stays silent
    (filterwarnings = error)."""
    from dynameta.warning_types import UnreliableResultWarning
    b = _builder(t=3e-9, n_bg=1e26, gamma=1.7, name="vgu0")
    try:
        with pytest.warns(UnreliableResultWarning, match=r"outside the calibrated range.*gamma = 1\.7 is not the default"):
            b.build_device()
    finally:
        b.teardown()
    b = _builder(gamma=1.7, name="vgu1")
    try:
        b.build_device()
    finally:
        b.teardown()


# ---- 3-D builder --------------------------------------------------------------------------------------------

# the 3-D builder's semiconductor and oxide set to the film of this file (the spec has no material defaults)
_FILM3D = dict(n_bg_m3=N_BG, dos_mass_kg=M, eps_semi=FX.EPS_S, eps_oxide=20.0, oxide_thk_m=8e-9)


def _run_3d(dg, vg, **spec_kw):
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    spec = Stacked3DSpec(lateral_m=4e-9, mesh_min_nm=0.02, mesh_max_nm=0.5,
                         density_gradient=EquilibriumDensityGradient(m_eff_kg=M) if dg else None, **_FILM3D, **spec_kw)
    s = Devsim3DEquilibrium(spec, device_name="d3_{}".format(int(dg)), mesh_name="m3_{}".format(int(dg)))
    try:
        s.solve(BiasPoint({"gate": vg, "body": 0.0}, "p"))
        g = {k: np.asarray(ds.get_node_model_values(device=s.device, region="semi", name=k))
             for k in ("x", "y", "z", "Electrons")}
    finally:
        s.teardown()
    return g


def test_3d_builder_density_gradient_matches_the_2d_builder_and_is_laterally_uniform():
    """Stacked3DSpec(density_gradient=...) (12 nm film over an ohmic body, 8 nm eps 20 oxide, full gate; z mesh
    0.02 nm at the oxide face) at +2 V: every node column equals the corner column to 1e-12 (MEASURED 2.7e-15), the
    oxide face carries the wall (n at the floor) and the body face the reservoir (n = n_bg's classical value), and
    the sheet excess equals the 2-D builder's on the same topology (validation/_eqdg_fixture.body_design: its own
    z mesh) within 2e-3 relative (MEASURED 4.2e-5) -- two meshes, the same model."""
    g = _run_3d(True, 2.0)
    k0 = (abs(g["x"]) < 1e-13) & (abs(g["y"]) < 1e-13)
    o = np.argsort(g["z"][k0])
    z3, n3 = g["z"][k0][o], g["Electrons"][k0][o]
    cols = {}
    for xi, yi, zi, ni in zip(g["x"], g["y"], g["z"], g["Electrons"]):
        cols.setdefault((round(xi, 13), round(yi, 13)), []).append((zi, ni))
    for c in cols.values():
        c.sort()
        assert np.max(np.abs(np.array([v for _, v in c]) / n3 - 1.0)) < 1e-12
    assert n3[-1] < 1e-9 * N_BG and n3[0] == pytest.approx(N_BG, rel=1e-6)
    b = LayeredDevsimBuilder(FX.body_design(12e-9, N_BG, M, t_ox=_FILM3D["oxide_thk_m"], eps_ox=_FILM3D["eps_oxide"]),
                             device_name="b2", mesh_name="b2_m",
                             density_gradient={"film": EquilibriumDensityGradient(m_eff_kg=M)})
    try:
        z2, n2 = _solve_column(b, 0.0, 2.0)
    finally:
        b.teardown()
    s3, s2 = float(trapz(n3 - N_BG, z3)), float(trapz(n2 - N_BG, z2))
    assert abs(s3 / s2 - 1.0) < 2e-3, (s3, s2)


def test_3d_builder_density_gradient_on_the_default_mesh_resolves_the_dead_layer():
    """Stacked3DSpec(density_gradient=...) on the DEFAULT mesh (mesh_min_nm 0.5) at +2 V, 12 nm film. OLD CODE
    (MEASURED on a gated degenerate film): 9 z nodes in the film and a sheet excess
    1.8 % below the fine mesh's (mesh_min_nm 0.02), silently. Now the wall plane is meshed at the 2-D screening face
    spacing (Stacked3DSpec.wall_mesh_min_nm, L_s/40 = 0.0194 nm here): MEASURED 1.85292e17 against 1.85118e17 on the
    fine mesh (+9.4e-4); gate 1e-3. A mesh_min_nm
    already finer than L_s/20 is kept as given (the fine-mesh reference here is therefore unchanged), and without the
    density gradient the axes are the original ones (the per-plane size reduces to the old rule bit for bit)."""
    from dynameta.carriers import devsim_3d as D3
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    dg = EquilibriumDensityGradient(m_eff_kg=M)
    assert Stacked3DSpec(**_FILM3D).wall_mesh_min_nm() == 0.5
    assert Stacked3DSpec(density_gradient=dg, mesh_min_nm=0.02, **_FILM3D).wall_mesh_min_nm() == 0.02
    hw = Stacked3DSpec(density_gradient=dg, **_FILM3D).wall_mesh_min_nm()
    assert hw == pytest.approx(0.0194, rel=0.02)                 # L_s / 40 of the film
    old = D3._graded_axis([0.0, 12.0, 20.0], [12.0], 0.5, 3.0)
    assert np.array_equal(old, D3._graded_axis([0.0, 12.0, 20.0], [12.0], 0.5, 3.0, refine_h_min=[0.5]))
    s = Devsim3DEquilibrium(Stacked3DSpec(lateral_m=4e-9, density_gradient=dg, **_FILM3D), device_name="d3dm",
                            mesh_name="m3dm")
    try:
        s.solve(BiasPoint({"gate": 2.0, "body": 0.0}, "p"))
        g = {k: np.asarray(ds.get_node_model_values(device=s.device, region="semi", name=k))
             for k in ("x", "y", "z", "Electrons")}
    finally:
        s.teardown()
    k0 = (abs(g["x"]) < 1e-13) & (abs(g["y"]) < 1e-13)
    o = np.argsort(g["z"][k0])
    z, n = g["z"][k0][o], g["Electrons"][k0][o]
    assert 0.9 * hw * 1e-9 < np.min(np.diff(z)) <= hw * 1e-9 * (1.0 + 1e-6)     # the wall cell
    f = _run_3d(True, 2.0)                                                     # mesh_min_nm 0.02 / mesh_max_nm 0.5
    kf = (abs(f["x"]) < 1e-13) & (abs(f["y"]) < 1e-13)
    of = np.argsort(f["z"][kf])
    s_def = float(trapz(n - N_BG, z))
    s_fine = float(trapz(f["Electrons"][kf][of] - N_BG, f["z"][kf][of]))
    assert abs(s_def / s_fine - 1.0) < 1e-3, (s_def, s_fine)


def test_3d_builder_refusals():
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    from dynameta.materials.transport_model import KaneBand
    g = EquilibriumDensityGradient()
    with pytest.raises(NotImplementedError, match="equilibrium"):
        Stacked3DSpec(physics="drift_diffusion", density_gradient=g, **_FILM3D)
    with pytest.raises(NotImplementedError, match="extra semiconductor"):
        Stacked3DSpec(density_gradient=g, extra_semiconductors=[("xs", "TCO", 2e-9, 9.0, 1e26)], **_FILM3D)
    with pytest.raises(NotImplementedError, match="Kane"):
        Stacked3DSpec(density_gradient=g, bands={"semi": KaneBand(m0_kg=0.25 * M_E, alpha_per_eV=0.4)},
                      **dict(_FILM3D, dos_mass_kg=KaneBand(m0_kg=0.25 * M_E, alpha_per_eV=0.4).dos_mass_kg(N_BG)))
    with pytest.raises(ValueError, match="EquilibriumDensityGradient"):
        Stacked3DSpec(density_gradient=1.8, **_FILM3D)


def test_film_interface_traps_on_a_dg_film_keep_gauss_law_and_their_classical_occupancy():
    """physics_traps.FilmInterfaceTraps (D_it 5e16 eV^-1 m^-2 over E - E_c in [-0.6, 0.6] eV, neutral at E_c, at
    film/up0) on a DG film at +2 V: Gauss's law with the trapped charge (gate charges + film charge + q sum_i V_i
    QTerm_fit) closes to 1e-9 of the gate charge, and the traps hold charge (|Q_it| > 1e-3 C/m^2): their occupancy
    follows the classical eta of the face potential (a localized state sees E_F - E_c, not the carriers' Lambda),
    so with the DG on, the trap charge differs from the classical device's only through the face potential."""
    from dynameta.carriers.physics_traps import FilmInterfaceTraps
    traps = {("film", "up0"): FilmInterfaceTraps(dit_eV_m2=5e16, e_lo_eV=-0.6, e_hi_eV=0.6, neutral_level_eV=0.0)}
    b = _builder(name="it", interface_traps=traps)
    try:
        b.solve(BiasPoint({"bot": 0.0, "top": 2.0}, "p"))
        W = b.design.unit_cell.period_x_m
        get = (lambda nm: np.asarray(ds.get_node_model_values(device=b.device, region="film", name=nm)))
        vol, n = get("NodeVolume"), get("Electrons")
        terms = [m for m in ds.get_node_model_list(device=b.device, region="film")
                 if m.startswith("QTerm_") and ":" not in m]
        assert terms
        q_it = Q_E * sum(float(np.sum(vol * get(t))) for t in terms) / W
        q_film = Q_E * float(np.sum(vol * (N_BG - n))) / W
        q_g = [ds.get_contact_charge(device=b.device, contact=c, equation="PotentialEquation") / W
               for c in ("bot", "top")]
        assert abs(q_it) > 1e-3
        assert abs(q_g[0] + q_g[1] + q_film + q_it) <= 1e-9 * max(abs(q_g[0]), abs(q_g[1])), (q_g, q_film, q_it)
    finally:
        b.teardown()
