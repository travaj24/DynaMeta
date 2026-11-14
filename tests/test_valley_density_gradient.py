"""v0.15.1: the VALLEY-RESOLVED electron density gradient (opt-in;
physics_density_gradient.setup_bipolar_dg(electron_valleys=...), density_gradient.SI100_ELECTRON_VALLEYS).
Each valley set carries its own quantum potential (its own quantization mass and share of N_c); all sets share
one electron quasi-Fermi level, so there is ONE continuity row for the total density, whose Scharfetter-Gummel
flux uses the effective band-edge shift Lambda_eff (physics_density_gradient, the valley block). Holes are
unchanged. The MOS harness and the Schrodinger-Poisson oracle are validation/bipolar_density_gradient.py
(mos_qe(valleys=...), sp_inversion(valleys=sp_valleys(...))); the full doping x bias matrix, the mesh
convergence and the gamma calibration are validation/valley_density_gradient.py. Audit note:
docs/audit/2026-11-10-v0.15.1-valley-resolved-dg.md.

On aed0aa9 (before this option) every physics gate here fails: electron_valleys does not exist (TypeError), and
the isotropic model it replaces misses the anisotropic SP by -14 % at the gated point (asserted below). The
defaults-off gate passes there by construction (that is its point: the digest was recorded on aed0aa9)."""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validation.bipolar_density_gradient as V  # noqa: E402
from dynameta.carriers import physics_density_gradient as DG  # noqa: E402
from dynameta.carriers.density_gradient import (SI100_ELECTRON_VALLEYS, ValleySet,  # noqa: E402
                                                check_valley_table)

NA = 3.0e23
GW = V.gamma_wall()
SI = SI100_ELECTRON_VALLEYS
TWIN = (ValleySet("a", V.MN, 0.5), ValleySet("b", V.MN, 0.5))

# sha256 of tests/_bipolar_dg_calllog.record_call_log() -- the BIPOLAR DG definition without electron_valleys
# (both carriers, quantum SRH, ohmic + insulator-wall rows, wall seed, gamma ramp, the quasi-equilibrium closure
# on both carriers; statistics 'fermi_dirac' and 'boltzmann'). RECORDED 2026-11-10 on an export of aed0aa9 (the
# base of this option) and on this tree: 638 calls, equal (Windows; canonical recording, tests/_calllog.py).
_BIPOLAR_DG_CALLLOG_DIGEST = "92cb247b4052422fdfa7ce4cacabb307407be0229ed7565306f1a6b96bd2d710"
_BIPOLAR_DG_CALLLOG_CALLS = 638


# ---- defaults off --------------------------------------------------------------------------------------------

def test_default_bipolar_dg_definition_is_byte_identical():
    """Without electron_valleys every DEVSIM definition call of the bipolar DG + wall + closure path is the
    aed0aa9 one, in order, character for character (the generalized kernel's tie=None / declare=True, the
    closure's valley hook returning None, the set_dg_gamma / seed / contact / wall branches that only read
    parameter lists). The unipolar DG and bipolar DD logs are gated in tests/test_bipolar_density_gradient.py
    (unchanged). STILL BITES (MEASURED): writing the kernel's default tie as '{u}^2 - {c}' changes the digest."""
    import _bipolar_dg_calllog as CL
    log = CL.record_call_log()
    assert (len(log), CL.digest(log)) == (_BIPOLAR_DG_CALLLOG_CALLS, _BIPOLAR_DG_CALLLOG_DIGEST)


def test_valley_table_and_api_guards():
    """The table is data (ValleySet rows) and is validated; the API refuses ambiguous or unsupported input
    BEFORE installing anything."""
    assert check_valley_table(SI) == SI
    assert sum(v.dos_share for v in SI) == 1.0
    assert [v.m_z_kg / V.M_E for v in SI] == pytest.approx([0.916, 0.19], rel=1e-15)
    for bad in ((), (ValleySet("a", V.MN, 0.5),), (ValleySet("a", -1.0, 1.0),), (ValleySet("a", V.MN, 0.0),),
                (ValleySet("a", V.MN, 0.6), ValleySet("b", V.MN, 0.6))):
        with pytest.raises(ValueError):
            check_valley_table(bad)
    reg = V._bar("vg_guard", "vg_guard_m", 40e-9)
    try:
        with pytest.raises(ValueError, match="OR electron_valleys"):
            DG.setup_bipolar_dg("vg_guard", reg, m_eff_n_kg=V.MN, electron_valleys=SI)
        assert DG.electron_valley_sets("vg_guard", reg) == 0
        assert "QSqrtN" not in ds.get_node_model_list(device="vg_guard", region=reg)
    finally:
        V._drop("vg_guard", "vg_guard_m")


def test_statistics_legacy_is_refused():
    """statistics='legacy' (the v0.14 averaged-g transport) has no activity equilibrium for the sets to share:
    refused before anything is installed."""
    from dynameta.carriers import physics_bipolar_dd as BP
    dev, mesh = "vg_lg", "vg_lg_m"
    V._drop(dev, mesh)
    try:
        with V._quiet():
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="a")
            ds.add_1d_mesh_line(mesh=mesh, pos=20e-9, ps=1e-9, tag="b")
            ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
            ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="a", tag2="b")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            ds.node_model(device=dev, region="s", name="NetDoping", equation="-1e23")
            BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=3e25, n_i_m3=1e16, mobility_n_m2Vs=0.1,
                                    mobility_p_m2Vs=0.04, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=True,
                                    n_dos_p_m3=1.8e25, statistics="legacy")
        with pytest.raises(ValueError, match="statistics='legacy'"):
            DG.setup_bipolar_dg(dev, "s", electron_valleys=SI)
        assert "QSqrtN_v0" not in ds.get_node_model_list(device=dev, region="s")
    finally:
        V._drop(dev, mesh)


def test_one_set_table_is_the_isotropic_model_bitwise():
    """A one-set table (share 1, m_z = the isotropic mass) IS today's model: it installs the isotropic rows call
    for call, so the solved MOS (FD, GAMMA_WALL, closure, 0.6 and 1.2 V) is bit-identical (np.array_equal) in
    n, p, psi and Lambda_n."""
    a = V.mos_qe(NA, (0.6, 1.2), fd=True, dg_gamma=GW, tag="one_iso")
    b = V.mos_qe(NA, (0.6, 1.2), fd=True, dg_gamma=GW, tag="one_set", valleys=(ValleySet("iso", V.MN, 1.0),))
    for ra, rb in zip(a, b):
        for k in ("n", "p", "psi", "lam_n"):
            assert np.array_equal(ra[k], rb[k]), k
        assert "n_v" not in rb


@pytest.mark.parametrize("fd", [False, True], ids=["boltzmann", "fermi_dirac"])
def test_two_identical_sets_reduce_to_the_isotropic_model(fd):
    """The valley MACHINERY (sum and intervalley rows, per-set walls and seeds, Lambda_eff, the reference-set
    closure) with two identical sets (m_z = 0.26 m0, shares 1/2) is the isotropic model: each set carries half
    the density with the same Lambda. MEASURED (closure, GAMMA_WALL, 0.6 / 1.2 V): n, p <= 4.9e-15 relative,
    psi <= 1.1e-16 V, Lambda_n vs Lambda_eff <= 1.9e-16 V. Gate 1e-12: the two are different sequences of
    double-precision operations converged to REL 1e-5 from the same seed, so only round-off separates them
    (~10 eps per row x the Newton's last correction). Refutation: one set's mass x (1 + 1e-4) must move n by
    more than 1e-9 (a real, if tiny, model change -- MEASURED 8.8e-5 Boltzmann, 8.4e-5 FD)."""
    a = V.mos_qe(NA, (0.6, 1.2), fd=fd, dg_gamma=GW, tag="tw_iso")
    b = V.mos_qe(NA, (0.6, 1.2), fd=fd, dg_gamma=GW, tag="tw_two", valleys=TWIN)
    bad = V.mos_qe(NA, (1.2,), fd=fd, dg_gamma=GW, tag="tw_bad",
                   valleys=(ValleySet("a", V.MN, 0.5), ValleySet("b", V.MN * (1.0 + 1e-4), 0.5)))
    for ra, rb in zip(a, b):
        for k in ("n", "p"):
            assert np.max(np.abs(rb[k] / ra[k] - 1.0)) <= 1e-12, k
        assert np.max(np.abs(rb["psi"] - ra["psi"])) <= 1e-12
        assert np.max(np.abs(rb["lam_n"] - ra["lam_n"])) <= 1e-12
        assert np.max(np.abs(rb["n_v"][0] / rb["n_v"][1] - 1.0)) <= 1e-12
    moved = float(np.max(np.abs(bad[0]["n"] / a[1]["n"] - 1.0)))
    assert moved > 1e-9, moved


# ---- the exact Jacobian ----------------------------------------------------------------------------------------

def _jacobian_probe(dev, reg="s"):
    """max over sampled columns (every variable, ~25 nodes each) and ALL rows of |J_rc - dF_r/dx_c| / max_c' |J_rc'|
    (row-scaled), DEVSIM's assembled DC Jacobian J against central finite differences of its residual F = -rhs
    (step 1e-6 |x|, Lambda and psi at least 1e-9 V)."""
    import scipy.sparse as sp

    def system():
        s = ds.get_matrix_and_rhs(format="csc")["static"]
        n = len(s["ap"]) - 1
        return (sp.csc_matrix((np.asarray(s["av"]), np.asarray(s["ai"]), np.asarray(s["ap"])), shape=(n, n)),
                np.asarray(s["rhs"]))

    J, _ = system()
    rowscale = np.asarray(abs(J).max(axis=1).todense()).ravel()
    rowscale[rowscale == 0] = 1.0
    worst = {}
    sols = [v for v in ds.get_node_model_list(device=dev, region=reg)
            if v in ("Potential", "Electrons", "Holes") or re.fullmatch(r"Q(SqrtN|Lambda)_v\d", v)]
    for v in sols:
        cols = np.asarray(ds.get_equation_numbers(device=dev, region=reg, variable=v), dtype=int)
        x0 = np.asarray(ds.get_node_model_values(device=dev, region=reg, name=v), dtype=float)
        for i in range(0, x0.size, max(1, x0.size // 25)):
            h = 1e-6 * max(abs(x0[i]), 1e-3 if (v == "Potential" or v.startswith("QLambda")) else 0.0)
            if h == 0.0:
                continue
            res = []
            for sgn in (1.0, -1.0):
                x = x0.copy()
                x[i] += sgn * h
                ds.set_node_values(device=dev, region=reg, name=v, values=list(x))
                res.append(system()[1])
            ds.set_node_values(device=dev, region=reg, name=v, values=list(x0))
            fd = -(res[0] - res[1]) / (2.0 * h)
            col = np.asarray(J[:, cols[i]].todense()).ravel()
            worst[v] = max(worst.get(v, 0.0), float(np.max(np.abs(col - fd) / rowscale)))
    return worst


_JAC_TOL = 1e-7


@pytest.mark.parametrize("case", ["closure", "transport"])
def test_jacobian_is_exact(case, monkeypatch):
    """DEVSIM's DC Jacobian of the valley-resolved system (Si(100) sets, FD, GAMMA_WALL, 60 nm MOS) against
    central finite differences, every variable, row-scaled: the sum / intervalley rows, the per-set kernels,
    the wall and ohmic rows (contact node models differentiated by DEVSIM), Lambda_eff's derivative chain
    (node models <model>:<var> that diff() looks up by name, carried to the edges as QLambda@n0:<var>@n0) in
    the flux and the quantum SRH, and the closure row on the reference set. closure: 1.0 V, electrons closed;
    transport: 0.5 V, the full model (no closure; DG installed at the target). MEASURED <= 3.0e-9 (closure) /
    2.1e-9 (transport). Budget 1e-7: central-difference truncation ~1e-12 and round-off ~eps / 1e-6 = 2e-10
    times the row's term-to-Jacobian ratio (<= ~100 here). Refutation (transport): the flux's derivative chain
    through Lambda_eff w.r.t. the second set's u (QLambda@n0:QSqrtN_v1@n0, QLambda@n1:QSqrtN_v1@n1) set to 0
    must fail it."""
    kw = (dict(deltas=(1.0,), close=("electrons",)) if case == "closure"
          else dict(deltas=(0.5,), closure=False, dg_at_target=True))
    deltas = kw.pop("deltas")
    common = dict(fd=True, dg_gamma=GW, valleys=SI, mesh_nb=V.qe_mesh(0.5e-9, 60e-9, ratio=10.0), length=60e-9)
    w = V.mos_qe(NA, deltas, tag="jac" + case, probe=_jacobian_probe, **common, **kw)[0]["probe"]
    assert max(w.values()) <= _JAC_TOL, w
    if case == "transport":
        orig = DG._install_valley_electrons

        def broken(device, region, valleys, gamma):
            out = orig(device, region, valleys, gamma)
            for nd in ("n0", "n1"):
                ds.edge_model(device=device, region=region, name="QLambda@{0}:QSqrtN_v1@{0}".format(nd),
                              equation="0")
            return out
        monkeypatch.setattr(DG, "_install_valley_electrons", broken)
        w2 = V.mos_qe(NA, deltas, tag="jacbad", probe=_jacobian_probe, **common, **kw)[0]["probe"]
        assert max(w2.values()) > _JAC_TOL, w2


# ---- Phase A: the closure path (MOS capacitor) against the anisotropic Schrodinger-Poisson oracle ---------------

def test_si100_valleys_against_the_anisotropic_sp_oracle():
    """3e23 p-Si, Fermi-Dirac, GAMMA_WALL, electrons closed, psi_s - psi_b = 1.2 V, against sp_inversion with the
    SAME Si(100) valleys made DOS-consistent (sp_valleys: g_v / 1.3716, so both sides have the DD's N_c -- the
    2026-11-08 correction). Gates: the v0.15 GATE G tolerances registered in advance for the DG-vs-SP
    comparison (N_inv 10 %, centroid 15 %: SP_TOL_NINV / SP_TOL_CENTROID); the 2-fold share within 0.1 of the
    SP's. MEASURED: N_inv +3.71 %, centroid -2.42 %, 2-fold share 0.692 vs SP 0.750. The isotropic model (the
    old code's only option) against the same oracle: N_inv -13.8 %, centroid +17.8 % -- it MISSES both registered
    tolerances (asserted: the gate fails on aed0aa9's model)."""
    sp = V.sp_inversion(NA, 1.2, valleys=V.sp_valleys(SI))
    sb = [float(np.sum(x[1])) for x in sp["subbands"]]
    r = V.mos_qe(NA, (1.2,), fd=True, dg_gamma=GW, valleys=SI, tag="spv")[0]
    n_inv, cen = V.mos_sheet(r)
    assert abs(n_inv / sp["N_inv"] - 1.0) <= V.SP_TOL_NINV, n_inv / sp["N_inv"] - 1.0
    assert abs(cen / sp["centroid"] - 1.0) <= V.SP_TOL_CENTROID, cen / sp["centroid"] - 1.0
    assert abs(V.valley_shares(r)[0] - sb[0] / sum(sb)) <= 0.1
    iso = V.mos_qe(NA, (1.2,), fd=True, dg_gamma=GW, tag="spi")[0]
    n_i, c_i = V.mos_sheet(iso)
    assert abs(n_i / sp["N_inv"] - 1.0) > V.SP_TOL_NINV and abs(c_i / sp["centroid"] - 1.0) > V.SP_TOL_CENTROID


def test_mesh_convergence_of_the_inversion_charge():
    """Nested meshes (first spacing 0.1 / 0.05 / 0.025 nm, ratio 40, 300 nm): observed order of N_inv and the
    centroid >= 1.7 (the scheme is second order; G1-7's gate for the isotropic DG), Si(100) sets, FD,
    GAMMA_WALL, 1.2 V. And the Richardson estimate of the DEFAULT mesh's (0.05 nm) discretization error,
    (4/3)|q(0.05) - q(0.025)| / q, must be <= 1e-3: 1/25 of the smallest DG-vs-SP difference of the validation
    matrix (+2.86 %), so no reported model error is a mesh artefact. MEASURED: orders 1.956 (N_inv) / 1.963
    (centroid); default-mesh error 2.2e-4 / 2.4e-4."""
    runs = [V.mos_qe(NA, (1.2,), fd=True, dg_gamma=GW, valleys=SI, mesh_nb=V.qe_mesh(h0), tag="mc%d" % i)[0]
            for i, h0 in enumerate((0.1e-9, 0.05e-9, 0.025e-9))]
    for j in (0, 1):
        q = [V.mos_sheet(r)[j] for r in runs]
        order = np.log2(abs(q[1] - q[0]) / abs(q[2] - q[1]))
        assert order >= 1.7, (j, q, order)
        assert 4.0 / 3.0 * abs(q[1] - q[2]) / abs(q[2]) <= 1e-3, (j, q)


def test_two_d_laterally_uniform_valley_mos_equals_one_d():
    """The 2-D tensor-mesh MOS (mos_qe(lateral=...), graded lateral lines 0 / 0.3 / 1.3 / 4.3 nm) with the
    Si(100) sets equals the 1-D device on every lateral line, node for node, at 0.6 and 1.2 V (FD, GAMMA_WALL,
    closure): <= 1e-10 relative in n, p, every set's density, Lambda_eff (to its max) and the charge per width
    (G1-7's tolerance: only round-off is left, the right-triangle box method reduces exactly to 1-D for a
    laterally uniform field). MEASURED 5.4e-15. Refutation: a 2-D body doped 1e-6 higher breaks it."""
    y = (0.0, 0.3e-9, 1.3e-9, 4.3e-9)
    kw = dict(fd=True, dg_gamma=GW, valleys=SI, mesh_nb=V.qe_mesh(0.1e-9), step=0.1)
    one = V.mos_qe(NA, (0.6, 1.2), tag="v2d1", **kw)
    two = V.mos_qe(NA, (0.6, 1.2), tag="v2d2", lateral=y, **kw)
    worst = 0.0
    for a, b in zip(one, two):
        assert len(b["cols"]) == len(y)
        for col in b["cols"].values():
            assert np.array_equal(col["x"], a["x"])
            for k in ("n", "p"):
                worst = max(worst, float(np.max(np.abs(col[k] / a[k] - 1.0))))
            worst = max(worst, float(np.max(np.abs(col["lam_n"] - a["lam_n"])) / np.max(np.abs(a["lam_n"]))))
        worst = max(worst, abs(b["Q"] / a["Q"] - 1.0))
    assert worst <= 1e-10, worst
    bad = V.mos_qe(NA * (1.0 + 1e-6), (0.6,), tag="v2dbad", lateral=y, **kw)[0]
    assert max(float(np.max(np.abs(c["p"] / one[0]["p"] - 1.0))) for c in bad["cols"].values()) > 1e-10


# ---- Phase C: the opt-in calibrated gamma ---------------------------------------------------------------------

def test_calibrated_gamma_states_its_measured_residuals():
    """GAMMA_SI100_SP_CALIBRATED is a FIT (density_gradient: to DynaMeta's anisotropic SP oracle, valley-resolved
    Si(100) DG, FD, closure, 1e23 .. 1e24 m^-3, 1.00 .. 1.30 V). Its stated residuals hold on the 3e23 m^-3 row of
    the fit matrix (seven band bendings): N_inv within -1.35 .. +0.95 %, centroid within -2.80 .. +1.55 % (the
    statement rounded outward to 0.05 point; the full matrix is validation GATE VC). MEASURED at 3e23: see the
    assertion messages. Refutation: GAMMA_WALL (the default wall value) on the same points leaves that band
    (N_inv +3.0 .. +7.0 %). Opt-in: the defaults (gamma 1, GAMMA_WALL) are unchanged."""
    from dynameta.carriers.density_gradient import GAMMA_SI100_SP_CALIBRATED
    assert GAMMA_SI100_SP_CALIBRATED == 1.88 and DG.GAMMA_SI100_SP_CALIBRATED == 1.88
    deltas = tuple(np.round(np.arange(1.0, 1.3001, 0.05), 10))
    sp = [V.sp_inversion(NA, d, valleys=V.sp_valleys(SI)) for d in deltas]
    for g, inside in ((GAMMA_SI100_SP_CALIBRATED, True), (GW, False)):
        r = V.mos_qe(NA, deltas, fd=True, dg_gamma=g, valleys=SI, tag="cal")
        en = [V.mos_sheet(a)[0] / s["N_inv"] - 1.0 for a, s in zip(r, sp)]
        ec = [V.mos_sheet(a)[1] / s["centroid"] - 1.0 for a, s in zip(r, sp)]
        ok = -0.0135 <= min(en) and max(en) <= 0.0095 and -0.0280 <= min(ec) and max(ec) <= 0.0155
        assert ok == inside, (g, en, ec)


# ---- Phase B: full transport (the single continuity row with the Lambda_eff flux) ------------------------------

def _qfl_ptp(dev):
    """Peak-to-peak of the electron quasi-Fermi level of the full model, written on the reference set exactly as
    the closure row does: psi + Lambda_0 - V_t (A(n_0/N_0) + ln(N_c/n_i)) [V]."""
    g = V._get
    nd = float(ds.get_parameter(device=dev, region="s", name="N_dos"))
    phi = (g(dev, "s", "Potential") + g(dev, "s", "QLambda_v0")
           - V.V_T * (g(dev, "s", "HJ_L_VDGn_v0") + np.log(nd / V.NI_SI)))
    return float(np.ptp(phi))


@pytest.mark.parametrize("fd,delta", [(False, 0.6),
                                      pytest.param(True, 0.35, marks=pytest.mark.slow)],   # FD: ~50 s, nightly
                         ids=["boltzmann_0.6V", "fermi_dirac_0.35V"])
def test_full_transport_equals_the_closure_at_equilibrium(fd, delta, monkeypatch):
    """(B-ii/iii) A MOS capacitor's DC state is equilibrium, so the FULL model (one continuity row, the
    Lambda_eff Scharfetter-Gummel flux, SRH) and the closure must be the same discrete solution where both
    converge (the full DG model is installed at the target bias, the D2d GATE E order): <= 1e-10 relative in
    n, p, every set's density and Lambda_eff (to its max), extended Newton REL 1e-10 (G1-2's gate). And the full
    model's electron quasi-Fermi level is flat: <= 1e-12 V. MEASURED (Boltzmann 0.1 / 0.35 / 0.6 V and FD 0.1 /
    0.35 / 0.6 V): <= 1.1e-15 and 2.8e-16 V. Refutation: the flux built on Lambda_0 instead of Lambda_eff (the
    activity correction dropped) -- MEASURED 49-57 % apart, quasi-Fermi level 24-35 mV non-flat."""
    kw = dict(fd=fd, dg_gamma=GW, valleys=SI, dg_at_target=True, ext=True, rel=1e-10, step=0.025)
    closed = V.mos_qe(NA, (delta,), tag="eqc", **kw)[0]
    full = V.mos_qe(NA, (delta,), tag="eqf", closure=False, probe=_qfl_ptp, **kw)[0]
    m = max(float(np.max(np.abs(closed[k] / full[k] - 1.0))) for k in ("n", "p"))
    m = max(m, max(float(np.max(np.abs(a / b - 1.0))) for a, b in zip(closed["n_v"], full["n_v"])))
    m = max(m, float(np.max(np.abs(closed["lam_n"] - full["lam_n"])) / np.max(np.abs(full["lam_n"]))))
    assert m <= 1e-10, m
    assert full["probe"] <= 1e-12, full["probe"]
    orig = DG._install_valley_electrons

    def lambda0_flux(device, region, valleys, gamma):
        out = orig(device, region, valleys, gamma)
        ds.node_model(device=device, region=region, name="QLambda", equation="QLambda_v0")
        for nm in ds.get_node_model_list(device=device, region=region):
            if nm.startswith("QLambda:") and nm != "QLambda:QLambda_v0":
                ds.node_model(device=device, region=region, name=nm, equation="0")
        return out
    monkeypatch.setattr(DG, "_install_valley_electrons", lambda0_flux)
    bad = V.mos_qe(NA, (delta,), tag="eqb", closure=False, probe=_qfl_ptp, **kw)[0]
    assert float(np.max(np.abs(closed["n"] / bad["n"] - 1.0))) > 1e-2
    assert bad["probe"] > 1e-3


@pytest.mark.slow                    # ~70 s: five extended-precision diode ramps
def test_diode_transport():
    """(B-iii/iv) The valley flux CARRYING CURRENT: the 1-D Si p-n diode of GATE F (1e23 / 1e22 m^-3, no wall,
    Boltzmann, extended Newton) at 0, +0.4 and -0.5 V.
      * two identical sets == the isotropic DG: MEASURED bit-identical currents and densities (the extended
        Newton converges both to the correctly rounded double); gate 1e-12.
      * the Si(100) sets: terminal currents conserve, |I_a + I_c| / |I_a| <= 1e-8 (GATE F's gate; MEASURED
        <= 2.1e-16); the DG changes the current by a small FIRST-ORDER amount, linear in gamma: change(gamma =
        1) / change(gamma = 1/2) = 2 +- 0.02 (MEASURED 1.996 forward, 2.005 reverse); and within 2 % of the
        isotropic DG's change (smooth potential, Lambda ~ 0.02 V_t: the sets' masses matter only at second
        order; MEASURED 0.4 % forward, 0.5 % reverse).
      * equilibrium (0 V): every edge current <= 1e-20 of q mu n_max V_t / h (MEASURED 2e-29).
    Refutation: the Lambda_0-flux perturbation of the MOS gate gives a NONZERO equilibrium diode current."""
    def probe(dev):
        jn = np.asarray(ds.get_edge_model_values(device=dev, region="s", name="ElectronCurrentDG"))
        n = V._get(dev, "s", "Electrons")
        return float(np.max(np.abs(jn))) / float(V.Q_E * 0.135 * np.max(n) * V.V_T / 10e-9)

    bias = (0.0, 0.4, -0.5)
    cl = V.diode(gamma_fracs=(None,), biases=bias)[None]
    iso = V.diode(gamma_fracs=(1.0,), biases=bias, probe=probe)[1.0]
    twin = V.diode(gamma_fracs=(1.0,), biases=bias, valleys=TWIN, probe=probe)[1.0]
    si = V.diode(gamma_fracs=(1.0, 0.5), biases=bias, valleys=SI, probe=probe)
    for b in bias[1:]:
        key = "I" if b > 0 else "I_rec"
        assert twin[b][key] == pytest.approx(iso[b][key], rel=1e-12)
        for q in ("Electrons", "Holes"):
            assert np.max(np.abs(twin[b]["arrays"][q] / iso[b]["arrays"][q] - 1.0)) <= 1e-12
        ch = {fr: si[fr][b][key] / cl[b][key] - 1.0 for fr in (1.0, 0.5)}
        assert si[1.0][b]["I_sum"] <= 1e-8
        assert ch[1.0] / ch[0.5] == pytest.approx(2.0, abs=0.02)
        assert ch[1.0] == pytest.approx(iso[b][key] / cl[b][key] - 1.0, rel=0.02)
    assert si[1.0][0.0]["probe"] <= 1e-20


_SSAC_CHILD = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import validation.bipolar_density_gradient as V
from dynameta.carriers import ac_analysis as AC
from dynameta.carriers import eq_registry as R
from dynameta.carriers.density_gradient import SI100_ELECTRON_VALLEYS as SI
ds = V.ds
out = {}
def probe(dev):
    v0 = float(ds.get_parameter(device=dev, name="wall_bias"))
    ds.delete_contact_equation(device=dev, contact="wall", name="PotentialEquation")
    R.forget(dev, "PotentialEquation", loc="wall", scope="contact")
    AC.setup_circuit_contact(dev, "wall", source_name="Vw", node_name="vw")
    def dc(v):
        ds.circuit_alter(name="Vw", value=v)
        V._solve()
        return float(ds.get_contact_charge(device=dev, contact="wall", equation="PotentialEquation"))
    dc(v0)
    _, C, G = AC.ssac_admittance([1.0], source_name="Vw")
    d1 = (dc(v0 + 1e-3) - dc(v0 - 1e-3)) / 2e-3
    d2 = (dc(v0 + 5e-4) - dc(v0 - 5e-4)) / 1e-3
    out["C_ssac"], out["C_dc"], out["C_dc1"] = float(C[0]), (4.0 * d2 - d1) / 3.0, d1
    return 0
V.mos_qe(3e23, (1.0,), fd=True, dg_gamma=V.gamma_wall(), valleys=SI, ext=True, rel=1e-10, step=0.1,
         tag="vssac", probe=probe)
json.dump(out, open(sys.argv[2], "w"))
'''


def test_ssac_of_the_valley_closure_is_the_low_frequency_cv(tmp_path):
    """ssac on the valley-resolved closure (Si(100) sets, FD, GAMMA_WALL, 1.0 V, extended Newton REL 1e-10): the
    1 Hz capacitance equals the DC dQ/dV of the wall charge (Richardson of +-1 / +-0.5 mV) to 1e-6, G1-5''s gate
    and budget; the bare 1 mV difference must miss by > 1e-5 (its O(dV^2) curvature). The small-signal matrix is
    the DC Jacobian above plus the time terms, so this also checks that no valley row is missing from it. Child
    process (DEVSIM circuits cannot be deleted in-process). MEASURED: C_ssac / C_dc - 1 = -8.3e-10; the bare
    1 mV difference -5.8e-5."""
    out = tmp_path / "ssac.json"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", _SSAC_CHILD, root, str(out)], check=True,
                   env=dict(os.environ, PYTHONPATH=root), capture_output=True, timeout=900)
    import json
    r = json.loads(out.read_text())
    assert r["C_ssac"] == pytest.approx(r["C_dc"], rel=1e-6), r
    assert abs(r["C_ssac"] / r["C_dc1"] - 1.0) > 1e-5, r
