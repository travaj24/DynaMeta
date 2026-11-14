"""v0.15.1 -- the local BTBT resolution check in 2-D and 3-D (carriers.tunneling.btbt_quadrature_error on the
ELEMENT assembly, and the 1 % UnreliableResultWarning btbt_generation_current now emits by default in every
dimension). Audit note: docs/audit/2026-11-08-v0.15-btbt-resolution-2d.md (derivation, budgets, measurements);
the 1-D estimator and its gates: tests/test_devsim_btbt_resolution.py, docs/audit/2026-11-06-v0.15-followups.md.

Gates
  Q1  FROZEN laterally uniform states on a 2-D tensor mesh (right triangles) and a 3-D Kuhn mesh whose element
      fields sample the R1 continuum fields at the element circumcentres (= the column midpoints; what the FV
      solve does on a mesh-aligned junction): the 2-D / 3-D estimate equals the true quadrature error (scipy
      quad, the independent oracle of R1) to 1e-10 absolute, and so the 1-D estimate. Budget: R1's (quad 1e-13,
      round-off <= ~200 terms x 1e-16 x B/E) plus the piece rule: the collapsed 16^2 (2-D) / 8^3 (3-D) Gauss rule
      is exact to degree 30 / 13 and ln f varies by <= 0.6 over a piece, remainder < 0.6^14/14! = 1e-14; the
      2-D/3-D sums have 6-24x more terms than 1-D (round-off <= 5e-12). 1e-10 leaves >= 20x.
      ADVERSARIAL (each must miss the oracle by >= 1e-5): the element field sampled at the CENTROID instead of
      the circumcentre (the two triangles of a cell then sample x0 + h/3 and x0 + 2h/3, not the midpoint), and
      NO side classification (every sample counted level with the element: the centred / two-sided stencil
      everywhere, so the cusp of the abrupt junction is fitted across).
  Q2  SOLVED laterally uniform 2-D Zener (4 rows) and its 1-D twin on the same x nodes (h = 0.5, 0.25 nm; 0.9, 1.5 V):
      on the solved field with D uniform the 2-D estimate equals the 1-D one (Q1 budget); predicted vs measured
      (against the 1-D Richardson reference) inside the band [1, 2] for both; warns at 0.5 nm naming the region
      and an element on the junction column, silent at 0.25 nm; A/B: the check moves no value (==) and no later
      solve (bitwise).
  Q3  3-D (Kuhn bar) solved Zener: in the band, equal to the 1-D estimate within 5 %, warns.
  Q4  the CORNER of the nonlocal study on its uniform 2 nm mesh (the "+36 %, no warning" mesh): warns, in the
      band against a stored mesh-converged reference; the centroid perturbation leaves the band.
  Q5  ORIENTATION: the Q2 Zener on its mesh rotated by 30 / 61 deg gives the same estimate (1e-8).
  Q6  LIMITATION (characterisation): an abrupt junction crossing an axis-aligned mesh at 30 deg is NOT covered.
  @slow: the slant reference re-measured; the corner's 2 / 1 nm currents re-measured with the band and the order
      (its 58k-node 0.5 nm reference mesh stays a standalone measurement: CORNER_I); the slant's fine meshes silent.
  Measured numbers: the audit note, section 5.
FAILS ON THE OLD CODE (098553f): btbt_quadrature_error returns None in 2-D / 3-D (TypeError on rep["error"]);
btbt_generation_current emits no warning in 2-D (Q2's pytest.warns fails).
"""
import math
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

import test_devsim_btbt_presets as P                                       # noqa: E402  (_wipe)
import test_devsim_btbt_resolution as R1                                   # noqa: E402  (the R1 fields, oracle)
from dynameta.carriers import physics_bipolar_dd as BP                       # noqa: E402
from dynameta.carriers import tunneling as TN                                # noqa: E402
from dynameta.warning_types import UnreliableResultWarning                   # noqa: E402

KINDS = ["uniform", "long", "parabola"]


# --------------------------------------------------------------------------------------------------------------
# Q1: frozen laterally uniform states (2-D tensor, 3-D Kuhn)
# --------------------------------------------------------------------------------------------------------------

def _kuhn_mesh(mesh, gx, gy, gz):
    """A Kuhn (path) tetrahedral mesh of the tensor grid gx x gy x gz, in memory (6 tets per cell)."""
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(coords.shape[0]).reshape(X.shape)
    el = []
    for i in range(gx.size - 1):
        for j in range(gy.size - 1):
            for k in range(gz.size - 1):
                for pm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                    v = [0, 0, 0]
                    path = [idx[i, j, k]]
                    for ax in pm:
                        v[ax] += 1
                        path.append(idx[i + v[0], j + v[1], k + v[2]])
                    el += [3, 0] + [int(t) for t in path]
    ds.create_gmsh_mesh(mesh=mesh, coordinates=[float(c) for c in coords.ravel()], physical_names=["r"],
                        elements=el)
    ds.add_gmsh_region(mesh=mesh, gmsh_name="r", region="r", material="Si")


def _frozen_nd(dev, xs, kind, dim):
    """The R1 frozen state (R1._frozen) on a 2-D tensor mesh (rows y = 0..3 nm) or a 3-D Kuhn mesh (ONE cell in y
    and z: every sample at one transverse level, the rank-deficient stencil case), the node potentials built so
    that every element's x field equals the continuum field at its column midpoint (the circumcentre), n = p =
    1e6 m^-3 (D = -1 + 2e-10 on every piece)."""
    P._wipe()
    mesh = dev + "_m"
    if dim == 2:
        ds.create_2d_mesh(mesh=mesh)
        hx = np.diff(xs)
        for i, x in enumerate(xs):                   # explicit lines, ps = the gap: no subdivision
            ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=float(x), ps=float(hx[min(i, hx.size - 1)]))
        for y in (0.0, 1e-9, 2e-9, 3e-9):
            ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=y, ps=1e-9)
        ds.add_2d_region(mesh=mesh, material="Si", region="r")
    else:
        _kuhn_mesh(mesh, xs, np.array([0.0, 1e-9]), np.array([0.0, 1.5e-9]))
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="r", name="NetDoping", equation="1e22")
    with R1.P._quiet():
        BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16, mobility_n_m2Vs=0.135,
                                mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=False,
                                btbt=R1.PRM)
    x = np.asarray(ds.get_node_model_values(device=dev, region="r", name="x"))
    ux = np.unique(x)
    assert ux.size == xs.size and np.max(np.abs(ux - xs)) < 1e-21           # the mesh has exactly these lines
    s_mid = 0.5 * (xs[1:] + xs[:-1])
    phi1 = np.concatenate([[0.0], -np.cumsum(np.diff(xs) * R1._field(kind, s_mid))])
    phi = phi1[np.searchsorted(xs, x - 1e-21)]
    for v, val in (("Potential", phi), ("Electrons", np.full(x.size, 1e6)), ("Holes", np.full(x.size, 1e6))):
        ds.set_node_values(device=dev, region="r", name=v, values=[float(a) for a in val])


def _q1(kind, dim):
    xs = R1._nodes(kind)
    try:
        x1 = R1._frozen("tq_1" + kind, xs, kind)
        e1 = TN.btbt_quadrature_error("tq_1" + kind, "r")["error"]
        _frozen_nd("tq_{}{}".format(dim, kind), np.sort(x1), kind, dim)
        rep = TN.btbt_quadrature_error("tq_{}{}".format(dim, kind), "r")
    finally:
        P._wipe()
    eps, _s, _h = R1._oracle(x1, kind)
    return rep, e1, eps


# 3-D "long" left out: the same check as 2-D on a 9.4k-tet mesh (+6 s)
@pytest.mark.parametrize("dim,kind", [(2, k) for k in KINDS] + [(3, "uniform"), (3, "parabola")])
def test_q1_laterally_uniform_estimate_equals_the_oracle_and_the_1d_estimate(dim, kind):
    rep, e1, eps = _q1(kind, dim)
    assert abs(rep["error"] - eps) <= 1e-10, (dim, kind, rep["error"], eps)
    assert abs(rep["error"] - e1) <= 1e-10, (dim, kind, rep["error"], e1)
    # MEASURED (estimate - 1-D estimate): 2-D uniform +3e-16, long -4e-15, parabola +2e-16; 3-D uniform -8e-15,
    # parabola +1.0e-12 (the 8^3 rule; still 100x inside the budget)
    if kind == "long":                                   # the worst element is a cell of the 0.128 nm peak column
        assert rep["worst_h_m"] == pytest.approx(0.128e-9, rel=1e-9)
        assert abs(rep["worst_centre_m"][0]) < 1e-15


@pytest.mark.parametrize("perturb", ["centroid", "no_sides"])
@pytest.mark.parametrize("kind", ["uniform", "long"])
def test_q1_adversarial_crude_samples_or_stencils_miss_the_oracle(kind, perturb, monkeypatch):
    if perturb == "centroid":
        monkeypatch.setattr(TN, "_element_samples", lambda X, cT, nhat: X.mean(axis=1))
    else:
        monkeypatch.setattr(TN, "_vertex_sides", lambda pv, tol: np.zeros(pv.shape, dtype=int))
    rep, _e1, eps = _q1(kind, 2)
    assert abs(rep["error"] - eps) >= 1e-5, (kind, perturb, rep["error"], eps)


# --------------------------------------------------------------------------------------------------------------
# Q2: SOLVED laterally uniform Zener (2-D, 4 rows) and its 1-D twin on the same x nodes
# --------------------------------------------------------------------------------------------------------------
# The corner study's junction (tests/_btbt_nonlocal_child.py: Na = Nd = 5e24 m^-3, n_i = 1e16, Si Hurkx, abrupt at
# 60 nm of 120 nm), UNIFORM explicit x lines at h (no remainder edge), plain ohmic contacts at x = 0 (p, biased) and
# x = 120 nm.
ZL, ZC = 120e-9, 60e-9
VRS = (0.9, 1.5)


def _seed_and_presolve(dev):
    from dynameta.carriers import eq_registry as _R
    BP.setup_equilibrium_seed_models(dev, "b")
    ds.node_model(device=dev, region="b", name="_seed",
                  equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "b")))
    for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region="b", name=v, values=ds.get_node_model_values(device=dev, region="b",
                                                                                           name=m))
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(dev, ceq)
    P._solve(1e10, 1e-8)                                      # Poisson-only pre-solve
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(dev, ceq)
    P._solve()


def _region_args():
    from _btbt_nonlocal_child import ZN, ZNI
    return ZN, dict(eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=ZNI, mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048,
                    tau_n_s=1e-3, tau_p_s=1e-3, fd_enhancement=False, btbt=TN.SILICON_HURKX)


def _zener_1d(dev, h):
    N, kw = _region_args()
    mesh = dev + "_m"
    xs = h * np.arange(int(round(ZL / h)) + 1)
    with P._quiet():
        ds.create_1d_mesh(mesh=mesh)
        for k, x in enumerate(xs):
            ds.add_1d_mesh_line(mesh=mesh, pos=float(x), ps=1.0,
                                **({"tag": "pc"} if k == 0 else {"tag": "nc"} if k == xs.size - 1 else {}))
        ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
        ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping", equation="ifelse(x < {}, {}, {})".format(ZC, -N, N))
        BP.setup_bipolar_region(dev, "b", **kw)
        for c in ("pc", "nc"):
            BP.setup_contact_ohmic_bipolar(dev, c)
        _seed_and_presolve(dev)
    return "pc_bias"


def _zener_2d(dev, h):
    """The 2-D twin: the child's build_2d (DEVSIM tensor mesh, inert pads, contacts 'lo' at x = 0 / 'hi' at L)."""
    import _btbt_nonlocal_child as C
    N, kw = _region_args()
    xl = [(k * h, h) for k in range(int(round(ZL / h)) + 1)]
    with P._quiet():
        C.build_2d(dev, xl, [(0.0, 1e-9), (3e-9, 1e-9)], "ifelse(x < {}, {}, {})".format(ZC, -N, N),
                   n_i=kw["n_i_m3"], tau=1e-3, local=TN.SILICON_HURKX)
        P._solve()
    return "lo_bias"


def _ramp_to(dev, name, vr, steps=6):
    v0 = ds.get_parameter(device=dev, name=name)
    for v in np.linspace(v0, -vr, steps + 1)[1:]:
        ds.set_parameter(device=dev, name=name, value=float(v))
        P._solve()


def _current(dev, W=1.0):
    return TN.btbt_generation_current(dev, "b", check_resolution=False) / W


@pytest.fixture(scope="module")
def lateral():
    """Per h (0.5 / 0.25 nm) and bias: the 2-D and 1-D estimates and currents [A/m^2], the 2-D warnings, and the
    estimates of the SOLVED 1-D field copied to both meshes with D uniform (carriers 1e6 m^-3 after the solve);
    plus the 1-D mesh-converged reference (Richardson of uniform 0.125 / 0.0625 nm; second order MEASURED: 1-D
    errors 1.83e-3 / 4.57e-4 at those h, ratio 4.00)."""
    out = {"ref": {}}
    try:
        for hf in (0.125e-9, 0.0625e-9):
            P._wipe()
            b = _zener_1d("tq_ref", hf)
            for vr in VRS:
                _ramp_to("tq_ref", b, vr)
                out["ref"].setdefault(vr, []).append(_current("tq_ref"))
        for h in (0.5e-9, 0.25e-9):
            P._wipe()
            b1, b2 = _zener_1d("tq_l1", h), _zener_2d("tq_l2", h)
            for vr in VRS:
                _ramp_to("tq_l1", b1, vr)
                _ramp_to("tq_l2", b2, vr)
                row = {"e1": TN.btbt_quadrature_error("tq_l1", "b"), "e2": TN.btbt_quadrature_error("tq_l2", "b"),
                       "I1": _current("tq_l1"), "I2": _current("tq_l2", 3e-9)}
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    row["I2_checked"] = TN.btbt_generation_current("tq_l2", "b") / 3e-9
                row["warn"] = [str(x.message) for x in w if issubclass(x.category, UnreliableResultWarning)]
                row["other_warn"] = [str(x.message) for x in w if not issubclass(x.category, UnreliableResultWarning)]
                out[(h, vr)] = row
            # the solved 1-D field on both meshes, D uniform (carriers 1e6 m^-3)
            x2 = np.asarray(ds.get_node_model_values(device="tq_l2", region="b", name="x"))
            x1 = np.asarray(ds.get_node_model_values(device="tq_l1", region="b", name="x"))
            psi1 = np.asarray(ds.get_node_model_values(device="tq_l1", region="b", name="Potential"))
            o = np.argsort(x1)
            psi2 = psi1[o][np.searchsorted(x1[o], x2 - 1e-21)]
            for dv, vals in (("tq_l1", psi1), ("tq_l2", psi2)):
                ds.set_node_values(device=dv, region="b", name="Potential", values=[float(v) for v in vals])
                for c in ("Electrons", "Holes"):
                    ds.set_node_values(device=dv, region="b", name=c, values=[1e6] * len(vals))
            out[(h, "uniformD")] = (TN.btbt_quadrature_error("tq_l1", "b")["error"],
                                    TN.btbt_quadrature_error("tq_l2", "b")["error"])
    finally:
        P._wipe()
    for vr in VRS:
        a, b = out["ref"][vr]
        out["ref"][vr] = (4.0 * b - a) / 3.0
    return out


@pytest.mark.parametrize("h", [0.5e-9, 0.25e-9])
def test_q2_solved_field_estimate_equals_the_1d_estimate(lateral, h):
    """On the SOLVED 1.5 V field of the 1-D twin, copied to the 2-D nodes, with D uniform: equal within the Q1
    budget. On the two SOLVED states the estimates differ only through D -- the 2-D scheme weights each half cell
    by its node's D, the 1-D one by the edge mean (MEASURED 2-D / 1-D - 1 = -4.9e-3 / +4.4e-4 at 0.9 / 1.5 V,
    h = 0.5 nm) -- so that comparison is reported (audit note), not gated."""
    e1, e2 = lateral[(h, "uniformD")]
    assert abs(e2 - e1) <= 1e-10, (h, e1, e2)
    assert e1 > 0.0


@pytest.mark.parametrize("vr", VRS)
def test_q2_predicted_vs_measured_and_the_warning(lateral, vr):
    """Predicted (the estimate) vs MEASURED (I(h) / I_ref - 1, I_ref the 1-D Richardson reference: the laterally
    uniform 2-D continuum IS the 1-D one) for the 2-D device and its 1-D twin. BAND 1 <= predicted / measured <= 2
    (audit note, 'The band'). The 2-D device warns at h = 0.5 nm (estimate +3.1 % / +1.9 %), naming the region and
    an element on the junction column, and is SILENT at h = 0.25 nm (+0.78 % / +0.47 %: the well-resolved case);
    the value is the same with or without the check (==)."""
    ref = lateral["ref"][vr]
    for h in (0.5e-9, 0.25e-9):
        row = lateral[(h, vr)]
        assert row["I2_checked"] == row["I2"]                      # the check never changes the value
        assert not row["other_warn"], row["other_warn"]
        for dim, e, I in ((1, row["e1"], row["I1"]), (2, row["e2"], row["I2"])):
            meas = I / ref - 1.0
            assert 1.0 <= e["error"] / meas <= 2.0, (h, vr, dim, e["error"], meas)
        if h == 0.5e-9:
            assert row["e2"]["error"] > TN.BTBT_QUADRATURE_TOL and len(row["warn"]) == 1, row["warn"]
            msg = row["warn"][0]
            assert "region 'b' of 'tq_l2'" in msg and "{:+.2%}".format(row["e2"]["error"]) in msg, msg
            assert abs(row["e2"]["worst_centre_m"][0] - ZC) <= 1.0e-9, row["e2"]["worst_centre_m"]
        else:
            assert 0.0 < row["e2"]["error"] < TN.BTBT_QUADRATURE_TOL and not row["warn"], row["warn"]


def test_q2_ab_the_check_does_not_move_any_value():
    """A/B (values must not move): the same 2-D device built twice from scratch (same name, same sequence) and
    solved to 0.9 V; run A calls the check there (with its one-time layout read: temporary element models, deleted
    again), run B does not; both then solve on to 1.5 V. The currents with and without the check are identical (==)
    and the two runs' solutions are bitwise equal node by node. (Two DIFFERENT devices built in one process differ
    by ~2e-15 with or without any check -- MEASURED, DEVSIM's own -- so the comparison is run against run.)"""
    try:
        res = {}
        for run, check in (("A", True), ("B", False)):
            P._wipe()
            dev = "tq_ab"
            b = _zener_2d(dev, 0.5e-9)
            _ramp_to(dev, b, 0.9)
            if check:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UnreliableResultWarning)
                    i_on = TN.btbt_generation_current(dev, "b")
                assert i_on == TN.btbt_generation_current(dev, "b", check_resolution=False)
            _ramp_to(dev, b, 1.5)
            res[run] = [np.asarray(ds.get_node_model_values(device=dev, region="b", name=v))
                        for v in ("Potential", "Electrons", "Holes")]
            res[run].append(np.array([TN.btbt_generation_current(dev, "b", check_resolution=False)]))
        for a, b in zip(res["A"], res["B"]):
            assert np.array_equal(a, b)
    finally:
        P._wipe()


def _zener_3d(dev, h):
    """The 3-D twin: a 120 x 1 x 1 nm Kuhn bar (devsim_3d.kuhn_bar_mesh, cells of side h), contacts on the x ends."""
    from dynameta.carriers import devsim_3d as D3
    N, kw = _region_args()
    mesh = dev + "_m"
    with P._quiet():
        D3.kuhn_bar_mesh(mesh, (ZL * 1e9, 1.0, 1.0), h * 1e9, region="b", contacts=("pc", "nc"))
        ds.add_gmsh_region(mesh=mesh, gmsh_name="b", region="b", material="Si")
        for c in ("pc", "nc"):
            ds.add_gmsh_contact(mesh=mesh, gmsh_name=c, region="b", name=c, material="metal")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping", equation="ifelse(x < {}, {}, {})".format(ZC, -N, N))
        BP.setup_bipolar_region(dev, "b", **kw)
        for c in ("pc", "nc"):
            BP.setup_contact_ohmic_bipolar(dev, c)
        _seed_and_presolve(dev)
    return "pc_bias"


def test_q3_3d_solved_zener_warns_like_the_1d_twin(lateral):
    """3-D (the estimator generalises unchanged: Q1 3-D is exact): the SOLVED laterally uniform Kuhn-bar Zener at
    h = 0.5 nm and 0.9 V. Its current per area equals the 1-D twin's (the y/z-invariant 3-D continuum is the 1-D
    one); its estimate lands in the Q2 band against the 1-D reference; it warns, naming the region and an element
    on the junction column, and the value is unchanged by the check."""
    try:
        P._wipe()
        b = _zener_3d("tq_z3", 0.5e-9)
        _ramp_to("tq_z3", b, 0.9)
        rep = TN.btbt_quadrature_error("tq_z3", "b")
        I = _current("tq_z3", 1e-18)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            I_chk = TN.btbt_generation_current("tq_z3", "b") / 1e-18
    finally:
        P._wipe()
    assert I_chk == I
    meas = I / lateral["ref"][0.9] - 1.0
    assert 1.0 <= rep["error"] / meas <= 2.0, (rep["error"], meas)
    assert abs(rep["error"] / lateral[(0.5e-9, 0.9)]["e1"]["error"] - 1.0) < 0.05
    unrel = [str(x.message) for x in w if issubclass(x.category, UnreliableResultWarning)]
    assert len(unrel) == 1 and "region 'b' of 'tq_z3'" in unrel[0], unrel
    assert abs(rep["worst_centre_m"][0] - ZC) <= 1.0e-9, rep["worst_centre_m"]


# --------------------------------------------------------------------------------------------------------------
# Q4 / Q5: the 2-D devices of the nonlocal study -- the junction CORNER and a SLANTED junction
# --------------------------------------------------------------------------------------------------------------
# CORNER: tests/_btbt_nonlocal_child.py sc_corner's local device (n+ quadrant x, y > 60 nm in a p+ 120 nm square,
# 5e24 m^-3, Si Hurkx, contacts 'lo' along y = 0 and 'hi' along y = L for x > 80 nm) on UNIFORM tensor meshes; the
# 2 nm one is the mesh of the nonlocal note's "+36 % with no warning" (docs/audit/2026-11-07-v0.15-nonlocal-btbt-2d.md
# section 7 item 2). SLANT: the same junction doping slanted by 30 deg in a 120 x 40 nm slab (contacts at x = 0 / L),
# h uniform in y and within x in [24, 96] nm (2 nm outside, beyond the depletion edge), with the abrupt doping
# AVERAGED over each node's control volume (the trapezoid CDF of n.x over the h x h cell): sampled at the nodes, an
# abrupt junction that is not on mesh lines becomes a staircase whose charge error is FIRST order in h and negative
# (MEASURED, audit note section 6), which no quadrature estimate covers; the CV average removes it.
SLANT_LY, SLANT_DEG = 40e-9, 30.0


def _slant_cv_doping(h):
    from _btbt_nonlocal_child import ZN
    th = math.radians(SLANT_DEG)
    nx, ny = math.cos(th), -math.sin(th)                   # junction normal: x cos - (y - yc) sin = c cos
    b1, b2 = sorted([abs(nx) * h / 2, abs(ny) * h / 2], reverse=True)
    t = "(-(({nx})*x + ({ny})*(y - {yc}) - {s0}))".format(nx=repr(nx), ny=repr(ny), yc=SLANT_LY / 2, s0=repr(ZC * nx))
    F = ("ifelse({t} <= -{a}, 0, ifelse({t} <= -{d}, ({t} + {a})^2/(8*{b1}*{b2}), ifelse({t} <= {d}, "
         "({t} + {b1})/(2*{b1}), ifelse({t} <= {a}, 1 - ({a} - {t})^2/(8*{b1}*{b2}), 1))))").format(
        t=t, a=repr(b1 + b2), d=repr(b1 - b2), b1=repr(b1), b2=repr(b2))
    return "{} + {}*(1 - {})".format(-ZN, 2 * ZN, F)        # -N + 2N * (n-side fraction of the cell)


def _device_2d(geom, h, dev):
    """Build and pre-solve the corner (uniform h) or the CV-averaged slant (module comment); returns the width
    [m] the current is divided by (1 for the corner: per depth)."""
    import _btbt_nonlocal_child as C
    N, kw = _region_args()
    if geom == "corner":
        xl = [(k * h, h) for k in range(int(round(ZL / h)) + 1)]
        # the nodes ON the junction lines are p, as in the child (whose lines sit at exactly 60 nm): k h is
        # 6.000000000000001e-08 here, so the threshold carries a sub-mesh offset (else that row flips to n and the
        # n quadrant grows by one cell: MEASURED +5.5 % in the current)
        dop = "ifelse((x > {c}) && (y > {c}), {n}, {mn})".format(c=ZC + 1e-13, n=N, mn=-N)
        contacts = [("lo", 0.0, ZL, 0.0, 0.0), ("hi", ZC + 20e-9, ZL, ZL, ZL)]
        yl, W = xl, 1.0
    else:
        win, far = 36e-9, 2e-9
        pos = ([k * far for k in range(int(round((ZC - win) / far)))]
               + [ZC - win + k * h for k in range(int(round(2 * win / h)) + 1)]
               + [ZC + win + k * far for k in range(1, int(round((ZC - win) / far)) + 1)])
        gaps = np.diff(pos)
        xl = [(float(p), float(g)) for p, g in zip(pos, np.append(gaps, gaps[-1]))]
        yl = [(k * h, h) for k in range(int(round(SLANT_LY / h)) + 1)]
        dop = _slant_cv_doping(h)
        contacts = [("lo", 0.0, 0.0, 0.0, SLANT_LY), ("hi", ZL, ZL, 0.0, SLANT_LY)]
        W = SLANT_LY
    from dynameta.carriers import eq_registry as _R
    with P._quiet():
        # the child's Poisson pre-solve allows 100 Newton steps; the 1 nm CV-averaged slant needs 192 (MEASURED)
        C.build_2d(dev, xl, yl, dop, n_i=kw["n_i_m3"], tau=1e-3, contacts=contacts, local=TN.SILICON_HURKX,
                   presolve=False)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.delete_by_name(dev, ceq)
        ds.solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=400)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.reapply_by_name(dev, ceq)
        # the equilibrium start at relative 1e-8: at 1e-10 the 1 nm slant's zero-bias Newton stalls on its round-off
        # floor (MEASURED, no convergence in 400 steps); every biased solve below runs at 1e-10
        ds.solve(type="dc", absolute_error=1e12, relative_error=1e-8, maximum_iterations=400)
    return W


def _at_bias(geom, h, vr, dev):
    """(estimate report, current [A/m per depth, or A/m^2 for the slant]) of the solved device at reverse vr,
    ramped in steps <= 0.15 V (the corner child's)."""
    W = _device_2d(geom, h, dev)
    n = max(1, int(math.ceil(vr / 0.15)))
    for j in range(1, n + 1):
        ds.set_parameter(device=dev, name="lo_bias", value=-vr * j / n)
        P._solve()
    return TN.btbt_quadrature_error(dev, "b"), _current(dev, W), W



# MEASURED mesh-converged references at 0.9 / 1.5 V reverse (2026-11-08, this box), each the Richardson
# extrapolation (4 I(h/2) - I(h)) / 3 of the two finest meshes; the @slow test_q4_q6_references_reproduce re-measures
# the slant one and the corner's 2 / 1 nm currents (CORNER_I). CORNER per depth [A/m], uniform 1.0 / 0.5 nm
# (14641 / 58081 nodes); its errors at 2 / 1 nm (+40.3 / +10.6 % at 0.9 V) fall by 3.80 per halving, second order. SLANT (CV-averaged doping) per width [A/m^2], h = 0.5 / 0.25 nm
# (13689 / 50391 nodes).
CORNER_REF = {0.9: 1.716591e-06, 1.5: 2.174167e-05}
SLANT_REF = {0.9: 4.874140e-07 / SLANT_LY, 1.5: 6.602365e-06 / SLANT_LY}


def _ratio_case(geom, h, vr, dev, ref, perturb=None):
    """(estimate report, measured error I / ref - 1, the estimate under the named perturbation or None, the
    current read WITH the check, the warnings it raised) on the solved device."""
    try:
        P._wipe()
        rep, I, W = _at_bias(geom, h, vr, dev)
        bad = None
        if perturb is not None:
            mp = pytest.MonkeyPatch()
            try:
                if perturb == "centroid":
                    mp.setattr(TN, "_element_samples", lambda X, cT, nhat: X.mean(axis=1))
                else:
                    mp.setattr(TN, "_vertex_sides", lambda pv, tol: np.zeros(pv.shape, dtype=int))
                bad = TN.btbt_quadrature_error(dev, "b")["error"]
            finally:
                mp.undo()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            I_chk = TN.btbt_generation_current(dev, "b") / W
    finally:
        P._wipe()
    return rep, I / ref - 1.0, bad, I_chk, w


def test_q4_corner_2nm_the_unflagged_36_percent_mesh_now_warns_in_band():
    """The nonlocal note's uniform 2 nm corner mesh (3721 nodes) at 0.9 V reverse: the local current is +40.3 %
    off the mesh-converged reference (MEASURED; the note's +36 % was against its 0.5 nm graded mesh, itself
    +2.9 % off), and is now FLAGGED: predicted +48.4 %, predicted / measured = 1.20, inside the band [1, 2]
    (audit note, "The band"). The worst element is on the vertical junction arm where it meets the top boundary
    next to the contact, where the field is highest. Perturbation (named, must leave the band): the element field
    sampled at the CENTROID -- MEASURED predicted / measured 0.70. The value is unchanged by the check."""
    rep, meas, bad, I_chk, w = _ratio_case("corner", 2e-9, 0.9, "tq_c2", CORNER_REF[0.9], perturb="centroid")
    assert I_chk / CORNER_REF[0.9] - 1.0 == meas                                    # the check moves nothing
    assert 1.0 <= rep["error"] / meas <= 2.0, (rep["error"], meas)
    assert not 1.0 <= bad / meas <= 2.0, (bad, meas)
    unrel = [str(x.message) for x in w if issubclass(x.category, UnreliableResultWarning)]
    assert len(unrel) == 1 and "region 'b' of 'tq_c2'" in unrel[0], unrel
    assert abs(rep["worst_centre_m"][0] - ZC) <= 2e-9 and rep["worst_centre_m"][1] > ZL - 2e-9, rep


def _rotated_zener(dev, h, deg, rows=4, dv=1e-9):
    """The Q2 lateral Zener on a tensor mesh ROTATED by deg (in-memory gmsh mesh of the same right triangles,
    contacts on the rotated ends): junction and mesh lines aligned, nothing axis-aligned."""
    from dynameta.carriers import eq_registry as _R
    th = math.radians(deg)
    nu = int(round(ZL / h))
    U, V = np.meshgrid(np.arange(nu + 1) * h, np.arange(rows) * dv, indexing="ij")
    u, v = U.ravel(), V.ravel()
    X = np.column_stack([u * math.cos(th) - v * math.sin(th), u * math.sin(th) + v * math.cos(th), 0.0 * u])
    idx = np.arange(u.size).reshape(nu + 1, rows)
    el = []
    for i in range(nu):
        for j in range(rows - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            el += [2, 0, int(a), int(b), int(d), 2, 0, int(b), int(c), int(d)]
    for i, ph in ((0, 1), (nu, 2)):
        for j in range(rows - 1):
            el += [1, ph, int(idx[i, j]), int(idx[i, j + 1])]
    N, kw = _region_args()
    mesh = dev + "_m"
    with P._quiet():
        ds.create_gmsh_mesh(mesh=mesh, coordinates=[float(c) for c in X.ravel()], physical_names=["b", "pc", "nc"],
                            elements=el)
        ds.add_gmsh_region(mesh=mesh, gmsh_name="b", region="b", material="Si")
        for c in ("pc", "nc"):
            ds.add_gmsh_contact(mesh=mesh, gmsh_name=c, region="b", name=c, material="metal")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        s = "(x*{} + y*{})".format(repr(math.cos(th)), repr(math.sin(th)))
        ds.node_model(device=dev, region="b", name="NetDoping",
                      equation="ifelse({} < {}, {}, {})".format(s, ZC - 0.25 * h, -N, N))
        BP.setup_bipolar_region(dev, "b", **kw)
        for c in ("pc", "nc"):
            BP.setup_contact_ohmic_bipolar(dev, c)
        BP.setup_equilibrium_seed_models(dev, "b")
        ds.node_model(device=dev, region="b", name="_seed",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "b")))
        for vv, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="b", name=vv,
                               values=ds.get_node_model_values(device=dev, region="b", name=m))
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.delete_by_name(dev, ceq)
        ds.solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=400)   # 133 steps at 30 deg
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.reapply_by_name(dev, ceq)
        ds.solve(type="dc", absolute_error=1e12, relative_error=1e-8, maximum_iterations=400)
    return "pc_bias"


def test_q5_estimate_is_independent_of_the_mesh_orientation():
    """A SLANTED junction on a mesh ALIGNED with it: the Q2 lateral Zener on its tensor mesh rotated by 30 and
    61 deg (nothing axis-aligned: field directions, element frames, samples, pieces all oblique). The physics is
    the unrotated device's, so the current and the estimate must be the unrotated ones. Budget: the solves converge
    to a relative update of 1e-10 per variable, which the field factor amplifies by B/E + P ~ 12 -> 1.2e-9 in the
    current and the estimate; 1e-8 relative (MEASURED current 6e-15 / 4e-14, estimate 4e-13 / 2e-13). Any
    coordinate-dependent choice (an axis-aligned frame, level counts along x / y) would break it far above that."""
    res = {}
    try:
        for deg in (0.0, 30.0, 61.0):
            P._wipe()
            b = _rotated_zener("tq_rot", 0.5e-9, deg)
            _ramp_to("tq_rot", b, 0.9)
            res[deg] = (TN.btbt_quadrature_error("tq_rot", "b")["error"], _current("tq_rot"))
    finally:
        P._wipe()
    e0, i0 = res[0.0]
    assert e0 > TN.BTBT_QUADRATURE_TOL
    for deg in (30.0, 61.0):
        assert abs(res[deg][0] / e0 - 1.0) <= 1e-8, (deg, res[deg], e0)
        assert abs(res[deg][1] / i0 - 1.0) <= 1e-8, (deg, res[deg], i0)


def test_q6_limitation_abrupt_junction_across_the_mesh_is_not_covered():
    """CHARACTERISATION of what the check does NOT cover (audit note section 6): the abrupt junction slanted by
    30 deg across an axis-aligned tensor mesh (CV-averaged doping), h = 1 nm, 0.9 V. The mesh error there is the
    representation of the abrupt doping on the oblique mesh, not the quadrature: MEASURED -4.65 % against the
    mesh-converged reference while the quadrature estimate is +0.54 %, so NO warning fires. This pins the
    limitation: a future check that covers it makes this test fail (on the warning count) and must revisit it."""
    rep, meas, _bad, I_chk, w = _ratio_case("slant", 1e-9, 0.9, "tq_s1", SLANT_REF[0.9])
    assert 0.0 < rep["error"] < TN.BTBT_QUADRATURE_TOL, rep["error"]
    assert meas < -3.0 * TN.BTBT_QUADRATURE_TOL, meas
    assert not [x for x in w if issubclass(x.category, UnreliableResultWarning)]
    assert I_chk / SLANT_REF[0.9] - 1.0 == meas


def _at_bias_child(geom, h, vr):
    """_at_bias in a FRESH interpreter (one per mesh): (estimate, current), so no large mesh shares a worker's
    DEVSIM memory with the meshes before it (the measurements ran one mesh per process too)."""
    import json
    import os
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    code = ("import json, sys; sys.path.insert(0, {here!r}); import test_devsim_btbt_resolution_2d as Q; "
            "r, I, W = Q._at_bias({g!r}, {h!r}, {v!r}, 'tq_refc'); print('RESULT ' + json.dumps([r['error'], I]))"
            ).format(here=here, g=geom, h=h, v=vr)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert out.returncode == 0 and lines, out.stderr[-3000:]
    return tuple(json.loads(lines[-1][7:]))


# the corner currents per depth at 0.9 V [A/m] behind CORNER_REF (MEASURED, standalone processes): uniform 2 / 1 / 0.5 nm
CORNER_I = {2e-9: 2.408494791649435e-06, 1e-9: 1.8989596990983665e-06, 0.5e-9: 1.7621830395995812e-06}


@pytest.mark.slow
@pytest.mark.parametrize("geom", ["corner", "slant"])
def test_q4_q6_references_reproduce_and_the_band_holds_at_every_mesh(geom):
    """@slow (MEASURED 99 s corner / ~16 min slant; one fresh interpreter per mesh, _at_bias_child).
    CORNER: re-measures the 2 and 1 nm currents behind the stored reference (to 1e-6 relative: the Newton tolerance
    amplified by B/E is ~1e-9; the rest covers platform BLAS differences), checks the band [1, 2] at both against
    CORNER_REF (MEASURED 1.20 / 1.18) and the second order of the measured error (2 -> 1 nm: MEASURED 3.80, inside
    [3.5, 4.5]; k h = 2.7 at 2 nm is pre-asymptotic). The 58081-node 0.5 nm mesh of the reference is NOT re-run
    here: its factorisation fails intermittently inside the suite ("umfpack numeric failed", MEASURED 2 of 4 runs,
    in-process and in a child; inferred: DEVSIM's 32-bit UMFPACK near its memory limit), while it converged in both
    standalone runs that measured it (CORNER_I, audit note section 9).
    SLANT: re-measures the reference (Richardson 0.5 / 0.25 nm, 50391 nodes; to 1e-6) and checks that both meshes
    are silent and within 1 % (estimates +0.22 / +0.07 %, measured -0.26 / -0.07 %)."""
    hs = (2e-9, 1e-9) if geom == "corner" else (0.5e-9, 0.25e-9)
    rows = {h: _at_bias_child(geom, h, 0.9) for h in hs}
    if geom == "corner":
        for h in hs:
            assert abs(rows[h][1] / CORNER_I[h] - 1.0) <= 1e-6, (h, rows[h], CORNER_I[h])
            assert 1.0 <= rows[h][0] / (rows[h][1] / CORNER_REF[0.9] - 1.0) <= 2.0, (h, rows[h])
        order = (rows[2e-9][1] / CORNER_REF[0.9] - 1.0) / (rows[1e-9][1] / CORNER_REF[0.9] - 1.0)
        assert 3.5 <= order <= 4.5, order
    else:
        ref = (4.0 * rows[hs[-1]][1] - rows[hs[-2]][1]) / 3.0
        assert abs(ref / SLANT_REF[0.9] - 1.0) <= 1e-6, (ref, SLANT_REF[0.9])
        for h in hs:
            assert 0.0 < rows[h][0] < TN.BTBT_QUADRATURE_TOL, (h, rows[h])
            assert abs(rows[h][1] / ref - 1.0) < TN.BTBT_QUADRATURE_TOL, (h, rows[h])
