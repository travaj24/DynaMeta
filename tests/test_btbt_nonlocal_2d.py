"""2-D nonlocal band-to-band tunnelling in the DEVSIM Newton (carriers/btbt_nonlocal.py + btbt_nonlocal_2d.py).
Every case that adds the term runs in a CHILD process (tests/_btbt_nonlocal_child.py: DEVSIM's custom-equation
registration and circuits are permanent); the children run the suite's DC well-posedness hook too. numba is
REQUIRED for 2-D (user decision 23).

OLD CODE (22b1ba7): setup_btbt_nonlocal refuses every 2-D device (NotImplementedError, decision 21) -- each child
below fails at its setup call (MEASURED by running this file against an export of 22b1ba7).

  L2  LATERALLY UNIFORM 2-D Zener (4 rows, DEVSIM tensor mesh) at 0.9 V reverse: (a) every 2-D path == the 1-D
      model on its own row of the 2-D solution; (b) q Int G per width == the 1-D Zener DEVICE (its own solve);
      (c) the hole / electron residual shares cancel, the terminal current == q Int G within the contact-current
      floor; (d) the Newton needs no more iterations than the local model's 2-D twin (+3).
  S2  SLANTED (30 deg) FDT-type diode: the junction crosses the tensor mesh, paths cross triangles obliquely and
      turn with the field (DYNAMIC direction, exact Jacobian): detailed balance at 0 V; S = 4kT Re Y to 1e-4
      with the nonlocal BTBT >= 50 % of S (both carriers at the start -- the local shortcut -- must fail); ssac
      Re Y(1 Hz) == the DC dI/dV at 0 V and at -0.2 V; the term's share of DEVSIM's Jacobian == central
      differences of DEVSIM's residual at -0.2 V, the start's gradient-patch columns included (they carry the
      rotation of the path). Refutation: the Jacobian without the rotation terms fails it.
  F2  the FD transport under the DEFAULT statistics ('fermi_dirac', v0.15.0) on a DEGENERATE slanted diode (the
      S2 device with N_c = N_v = 2e21 m^-3: N / N_c = 5, eta up to 3.3): zero current at equilibrium (paths and
      terminal) to round-off, FDT with the two-point source (same-node prove-out), ssac == DC dI/dV at 0 V and
      -0.2 V, the Jacobian vs DEVSIM's residual INCLUDING the carrier columns (the FD occupancy derivatives
      g/n, g/p). Shown failing (@slow): statistics='legacy' (Boltzmann detailed balance on FD transport: the
      v0.14 zero-bias battery) breaks equilibrium and FDT; Boltzmann occupancy derivatives (g = 1) break the
      carrier columns and the reverse-bias ssac. L2 runs under the FD default too (fixture param).
  D2  refusals: no numba -> ImportError before anything is registered.
  CORNER (@slow, ~15 min) a MEASUREMENT of the local model's 2-D error on an n+ quadrant corner, with the
      independent check that the far arms reproduce the 1-D planar Zener's local / nonlocal ratio.
"""
import numpy as np
import pytest

ds = pytest.importorskip("devsim")
pytest.importorskip("numba")

from dynameta.carriers.devsim_env import import_devsim  # noqa: E402

import_devsim()
import _btbt_nonlocal_child as C  # noqa: E402

EPS = np.finfo(float).eps
Q = 1.602176634e-19
KT4 = 4.0 * 1.380649e-23 * 300.0
VT = 1.380649e-23 * 300.0 / Q
EG = 1.12


@pytest.fixture(scope="module", params=[False, True], ids=["boltzmann", "fd_default"])
def lat2d(request):
    """Boltzmann, and the FD transport under the DEFAULT statistics ('fermi_dirac' since v0.15.0; the 1-D twin
    and the local twin built the same way): the L2 gates hold for both. MEASURED (FD): per path 6.2e-13,
    I/W vs 1-D 7e-15, J - I 1.7e-3 A/m^2 against the 9.9e-3 floor, iterations 17 == 17."""
    return C.run_child("lat2d", vr=0.9, fd=request.param)


def test_l2a_each_2d_path_equals_the_1d_model_on_its_row(lat2d):
    """Budget (the kernel gate U2): 1e-13 x (1 + B/F) x 4 with B/F <= 20 at the 0.9 V junction field
    (1.9e9 / ~1e8 V/m) = 8.4e-12. MEASURED 5.6e-13 over 192 paths."""
    r = lat2d
    assert r["n2d"] == r["n1d"] == r["paths"] > 100
    assert r["per_path_dev"] <= 1e-13 * (1 + 20.0) * 4


def test_l2b_current_per_width_equals_the_1d_device(lat2d):
    """The 2-D and 1-D devices are solved separately. Budget: the per-path identity above (8.4e-12) plus the
    rate's sensitivity to the potential, d ln R/d psi <= (B/F)/E_g = 18 /V, times the measured difference
    of the two solutions' potentials (max over the rows' spread and 2-D vs 1-D: ~1e-16 V). MEASURED 8e-15."""
    r = lat2d
    assert r["same_x_nodes"]
    dpsi = max(r["row_spread_V"], r["psi_2d_vs_1d_V"])
    dev = abs(r["I2d_per_W"] / r["I1d"] - 1.0)
    print("2-D/W vs 1-D", dev, "dpsi", dpsi)
    assert dev <= 1e-13 * 21 * 4 + 18.0 * dpsi


def test_l2c_conservation_and_newton(lat2d):
    """Shares cancel to 4 N eps; terminal current vs q Int G within 32 ulps of the 1.4e12 A/m^2 majority SG term
    (the Z3 budget of tests/test_devsim_tunneling.py; per unit width) + 1e-4 I; iterations <= local + 3."""
    r = lat2d
    assert abs(r["hole_share"] + r["elec_share"]) <= 4 * r["paths"] * EPS * r["abs_share"]
    assert abs(r["In2d_per_W"] - r["I2d_per_W"]) <= 4 * r["paths"] * EPS * r["I2d_per_W"]
    floor = 32.0 * EPS * Q * 0.135 * VT * C.ZN / 2e-9
    assert abs(r["J2d_per_W"] - r["I2d_per_W"]) <= floor + 1e-4 * r["I2d_per_W"]
    assert r["its"] <= r["its_local"] + 3 and r["its_eq"] <= r["its_eq_local"] + 3


@pytest.fixture(scope="module")
def slant_eq():
    return C.run_child("slant", part="eq", same_node=True)


def test_s2_detailed_balance_and_fdt(slant_eq):
    """Detailed balance: 64 eps x 60 (the 1-D N3 budget). FDT: 1e-4 (the D2 gate-B tolerance). MEASURED 5.3e-15
    and <= 6.4e-9 with the nonlocal BTBT 55 % of S at 1 Hz."""
    r = slant_eq
    S, Y = np.array(r["S"]), np.array(r["ReY"])
    dev = S / (KT4 * Y) - 1.0
    print("slant: paths", r["stats"], "share", r["share_btbt"], "FDT", dev, "db", r["db_ratio"])
    assert r["stats"]["paths"] > 300
    assert r["db_ratio"] <= 64 * EPS * 60
    assert r["share_btbt"] >= 0.5
    assert np.max(np.abs(dev)) <= 1e-4


def test_s2_proveout_same_node_source_fails_fdt(slant_eq):
    """MEASURED -3.0e-2: the separated placement (hole at the start, electron spread over the end triangle) is
    what the FDT needs in 2-D as in 1-D."""
    r = slant_eq
    dev = np.array(r["S_same"]) / (KT4 * np.array(r["ReY"])) - 1.0
    assert abs(dev[0]) > 1e-4


def _ssac_tol(r, tag):
    G, G1, G2 = r["G" + tag], r["G" + tag + "_1"], r["G" + tag + "_2"]
    return abs(G1 / G - 1.0) + 2.0 * abs(G2 / G - 1.0) + 2e-6


def test_s2_ssac_equals_dc_conductance_at_0v(slant_eq):
    """Budget as the 1-D N8: the Richardson correction's size + 2x the spread to the dv = 2e-4 V reference +
    2e-6. MEASURED 1.8e-8."""
    r = slant_eq
    assert abs(r["ReY"][0] / r["G0"] - 1.0) <= _ssac_tol(r, "0")


@pytest.fixture(scope="module")
def slant_rev():
    return C.run_child("slant", part="rev_ssac")


@pytest.fixture(scope="module")
def slant_jac():
    return C.run_child("slant", part="rev_jac")


def test_s2_ssac_equals_dc_conductance_at_reverse_bias(slant_rev):
    """MEASURED 1.8e-5 against its budget 7.4e-5 (the reference's own spread across step sizes, 3e-5, is the
    Lipschitz kinks of I(V) when path ends cross mesh edges -- module doc, section 3)."""
    r = slant_rev
    assert r["Ip_r"] > 0 and abs(r["In_r"] - r["Ip_r"]) <= 1e-13 * r["Ip_r"]
    assert abs(r["Yr"][0] / r["Gr"] - 1.0) <= _ssac_tol(r, "r")


def _jac_ratio(recs):
    """max |J - Richardson FD| / (1e-6 row scale + 2 |fd - fd2|) over the tested columns."""
    J = np.array([q["J"] for q in recs])
    fd = -np.array([q["fd"] for q in recs])
    fd2 = -np.array([q["fd2"] for q in recs])
    rich = (4.0 * fd - fd2) / 3.0
    scale = np.max(np.abs(J), axis=0)[None, :]
    return float(np.max(np.abs(J - rich) / (1e-6 * scale + 2.0 * np.abs(fd - fd2)))), \
        float(np.max(np.abs(J - rich) / scale))


def test_s2_jacobian_matches_fd_of_devsim_residual(slant_jac):
    """The strongest path's hole row and end-triangle electron rows, columns: the start, a mid-path node, the
    end triangle and the start's gradient patch (theta(psi)). Budget: 1e-6 of the row scale (the J1 pattern)
    + 2 |fd - fd2|. MEASURED 1.4e-10 relative; dR/dtheta / R = 1.2 % per radian at that path."""
    r = slant_jac
    ratio, rel = _jac_ratio(r["recs"])
    print("slant Jacobian: ratio", ratio, "rel", rel, "dR/dtheta", r["dR_dtheta"], "R", r["R_k"])
    assert r["dR_dtheta"] is not None and abs(r["dR_dtheta"]) > 1e-3 * abs(r["R_k"])
    assert ratio <= 1.0


@pytest.mark.slow
def test_s2_refute_jacobian_without_rotation_terms():
    """~50 s (a third slanted-diode child). The frozen-geometry Jacobian of a DYNAMIC-direction term misses the
    rotation of the paths: it must fail the gate above."""
    r = C.run_child("slant", part="rev_jac", drop_rotation=True)
    ratio, rel = _jac_ratio(r["recs"])
    print("without rotation terms: ratio", ratio, "rel", rel)
    assert ratio > 10.0


# =====================================================================================================
# F2: the FD default statistics on a DEGENERATE slanted diode
# =====================================================================================================

def _contact_floor():
    """Round-off floor of the slanted diode's terminal current [A/m]: 32 ulps of its majority SG flux
    q mu V_t N / h (h = the 40 nm contact spacing) over the 0.24 um contact (the Z3 pattern)."""
    return 32.0 * EPS * Q * 0.135 * VT * C.SLANT_N / 40e-9 * C.SLANT_LY


@pytest.fixture(scope="module")
def slantfd_eq():
    return C.run_child("slant", part="eq", fd=True, n_dos=C.SLANTFD_NDOS, same_node=True)


@pytest.fixture(scope="module")
def slantfd_rev():
    return C.run_child("slant", part="rev", fd=True, n_dos=C.SLANTFD_NDOS, carriers=True)


def test_f2_fd_degenerate_equilibrium_and_fdt(slantfd_eq):
    """Degenerate (max L = eta >= 2). Paths: 64 eps x 60 (the 1-D N3 budget). Terminal current at 0 V: the
    contact round-off floor (2.4e-13 A/m). FDT: 1e-4, BTBT >= 50 % of S. MEASURED (FD default): eta 3.3, 760
    paths, 7.1e-15, J(0) 2.3e-15 A/m, FDT <= 9.9e-11 with BTBT 90 % of S. Legacy: 5.2e-3, -1.9e-4 A/m,
    2.7e-3 (test_f2_refute_legacy_statistics)."""
    r = slantfd_eq
    S, Y = np.array(r["S"]), np.array(r["ReY"])
    dev = S / (KT4 * Y) - 1.0
    print("FD slant: eta", r["eta_max"], r["stats"], "db", r["db_ratio"], "J0", r["J0"], "share", r["share_btbt"],
          "FDT", dev)
    assert r["eta_max"] >= 2.0 and r["stats"]["paths"] > 300
    assert r["db_ratio"] <= 64 * EPS * 60
    assert abs(r["J0"]) <= _contact_floor()
    assert r["share_btbt"] >= 0.5
    assert np.max(np.abs(dev)) <= 1e-4


def test_f2_fd_proveout_same_node_source_fails_fdt(slantfd_eq):
    """MEASURED -2.1e-2 (legacy: -1.8e-2): the separated placement is what the FDT needs."""
    dev = np.array(slantfd_eq["S_same"]) / (KT4 * np.array(slantfd_eq["ReY"])) - 1.0
    assert abs(dev[0]) > 1e-4


def test_f2_fd_ssac_equals_dc_conductance_at_0v(slantfd_eq):
    """The N8 budget. MEASURED 9.4e-9 (budget 2.3e-6)."""
    r = slantfd_eq
    assert abs(r["ReY"][0] / r["G0"] - 1.0) <= _ssac_tol(r, "0")


def test_f2_fd_ssac_equals_dc_conductance_at_reverse_bias(slantfd_rev):
    """MEASURED 1.4e-6 (budget 9.7e-6); with Boltzmann occupancy derivatives -3.5e-3 (@slow refutation)."""
    r = slantfd_rev
    assert r["Ip_r"] > 0 and abs(r["In_r"] - r["Ip_r"]) <= 1e-13 * r["Ip_r"]
    assert abs(r["Yr"][0] / r["Gr"] - 1.0) <= _ssac_tol(r, "r")


def test_f2_fd_jacobian_matches_fd_of_devsim_residual_incl_carriers(slantfd_rev):
    """The S2 columns (potential: start, mid-path, gradient patch, end triangle) plus the carrier columns the
    FD occupancies enter through: holes at the start, electrons at the end-triangle nodes (relative step
    1e-6). Budget as S2. MEASURED ratio 9.2e-5 / 1.1e-3 / 3.0e-3 (potential / electrons / holes), relative
    9e-11 / 1.1e-9 / 3.0e-9; Boltzmann derivatives: 5.4e4 / 6.3e4 (@slow refutation)."""
    r = slantfd_rev
    for key in ("recs", "recs_n", "recs_p"):
        ratio, rel = _jac_ratio(r[key])
        print("FD slant Jacobian", key, "ratio", ratio, "rel", rel)
        assert ratio <= 1.0
    assert abs(r["dR_dtheta"]) > 1e-3 * abs(r["R_k"])


@pytest.mark.slow
def test_f2_refute_legacy_statistics():
    """~45 s. statistics='legacy' (v0.14: FD transport, Boltzmann detailed balance) on the same degenerate
    diode: the zero-bias state is not an equilibrium of the FD transport. MEASURED: paths 5.2e-3, terminal
    -1.9e-4 A/m (8e8 x the floor), FDT 2.7e-3 -- each fails the gate above."""
    r = C.run_child("slant", part="eq", fd=True, statistics="legacy", n_dos=C.SLANTFD_NDOS)
    dev = np.array(r["S"]) / (KT4 * np.array(r["ReY"])) - 1.0
    print("legacy: db", r["db_ratio"], "J0", r["J0"], "FDT", dev)
    assert r["db_ratio"] > 1e3 * 64 * EPS * 60
    assert abs(r["J0"]) > 1e3 * _contact_floor()
    assert np.max(np.abs(dev)) > 1e-3


@pytest.mark.slow
def test_f2_refute_boltzmann_occupancy_derivatives():
    """~90 s. The term's carrier derivatives with g = 1 (dL/dn = 1/n, the Boltzmann form) on the degenerate
    diode: the carrier columns and the reverse-bias ssac fail, the potential columns still pass (control).
    MEASURED: ratios 5.4e4 (electrons) / 6.3e4 (holes), relative 5.4e-2 / 6.3e-2; ssac -3.5e-3."""
    r = C.run_child("slant", part="rev", fd=True, n_dos=C.SLANTFD_NDOS, carriers=True, boltz_dL=True)
    assert _jac_ratio(r["recs"])[0] <= 1.0
    assert _jac_ratio(r["recs_n"])[0] > 10.0 and _jac_ratio(r["recs_p"])[0] > 10.0
    assert abs(r["Yr"][0] / r["Gr"] - 1.0) > 10.0 * _ssac_tol(r, "r")


def test_d2_numba_missing_refuses_before_registration(monkeypatch):
    import dynameta.numba_env as NE
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers import btbt_nonlocal_2d as B2
    from dynameta.carriers import custom_terms as CT
    from dynameta.carriers.tunneling import SILICON_HURKX
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    try:
        ds.create_2d_mesh(mesh="nm2_m")
        for pos in (0.0, 2e-8):
            ds.add_2d_mesh_line(mesh="nm2_m", dir="x", pos=pos, ps=5e-9)
            ds.add_2d_mesh_line(mesh="nm2_m", dir="y", pos=pos, ps=5e-9)
        ds.add_2d_region(mesh="nm2_m", material="Si", region="r")
        ds.finalize_mesh(mesh="nm2_m")
        ds.create_device(mesh="nm2_m", device="nm2")
        _R.clear("nm2")
        ds.node_model(device="nm2", region="r", name="NetDoping", equation="ifelse(x < 1e-8, -1e24, 1e24)")
        BP.setup_bipolar_region("nm2", "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16, mobility_n_m2Vs=0.1,
                                mobility_p_m2Vs=0.04, tau_n_s=1e-6, tau_p_s=1e-6, fd_enhancement=False)
        monkeypatch.setattr(NE, "_NUMBA_OK", False)
        monkeypatch.setattr(B2, "have_numba", NE.have_numba)
        with pytest.raises(ImportError, match="decision 23"):
            NL.setup_btbt_nonlocal("nm2", "r", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=EG))
        assert not CT.is_registered()
    finally:
        for dv in list(ds.get_device_list()):
            ds.delete_device(device=dv)
        for m in list(ds.get_mesh_list()):
            ds.delete_mesh(mesh=m)
        _R.clear("nm2")


# =====================================================================================================
# CORNER: a measurement (no closed form exists for a 2-D corner junction)
# =====================================================================================================

def _bands(r):
    """Node masks of the corner device: within 12 nm of the corner; and, on the two junction arms (15 nm
    either side of x = c / y = c), the windows [c + 5, c + 25) nm (next to the corner), [c + 25, c + 35) nm
    and [c + 35 nm, L) (the FAR window, up to the homogeneous-Neumann far side), both arms together. Also the
    vertical / horizontal arm masks from c + 25 nm on, for the mirror check."""
    x, y = np.array(r["x"]), np.array(r["y"])
    c, L = C.CORNER_C, C.CORNER_L
    hb, vb = np.abs(y - c) < 15e-9, np.abs(x - c) < 15e-9

    def win(a, b):
        return (hb & (x >= c + a) & (x < c + b)) | (vb & (y >= c + a) & (y < c + b))

    return {"corner": np.hypot(x - c, y - c) < 12e-9, "flank": win(5e-9, 25e-9), "mid": win(25e-9, 35e-9),
            "far": win(35e-9, 1.0), "arm_v": vb & (y >= c + 25e-9), "arm_h": hb & (x >= c + 25e-9)}


@pytest.mark.slow
@pytest.mark.parametrize("fd", [False, True], ids=["boltzmann", "fd_default"])
def test_corner_local_vs_nonlocal_measurement(fd):
    """MEASUREMENT, ~15 min (890 s on the loaded dev box: two graded-mesh 2-D devices, 12769 nodes, x two
    models, 18 Newton solves each). An n+ quadrant in a p+ square (5e24 / 5e24 m^-3, Si Hurkx A, B in both
    models, 0.5 nm at the junction lines): the local / nonlocal generation ratio next to the corner and on
    the arms, at 0.9 / 1.5 / 2.5 V. Results in docs/audit/2026-11-07-v0.15-nonlocal-btbt-2d.md.

    Asserted: what must hold whatever the physics (both models converge, nonlocal electron total == hole
    total, the two arms -- mirror images -- agree to 1 %, every current positive), and the INDEPENDENT CHECK:
    far from the corner the arms are planar junctions, so their local / nonlocal ratio is the 1-D planar
    Zener's (same doping, contact distances 60 nm, same h; its own DEVSIM devices). Budget: the corner's
    influence left in the far window, bounded by the change of the ratio from the [c+25, c+35) window to the
    far one (it decays away from the corner), plus 0.5 % for the 1-D ratio's own mesh error at 0.5 nm (it
    moves by <= 0.6 % from 0.5 to 0.1 nm, measured). MEASURED: -0.9 / -0.3 / +1.8 % against budgets
    3.7 / 4.1 / 9.9 %. The flank window [c+5, c+25) fails the same budget (+9.3 / +7.7 / +20 %).
    fd_default (the FD transport, default statistics, both devices and the 1-D reference; ~13 min, the FD
    nonlocal device's equilibrium from the seed takes 186 Newton steps): MEASURED -1.0 / -0.3 / +1.8 % against
    3.7 / 4.1 / 9.9 %; every ratio within 0.7 % of the Boltzmann run (audit note section 10)."""
    r = C.run_child("corner", h=0.5e-9, vr=[0.9, 1.5, 2.5], fd=fd, timeout=3600)
    p = C.run_child("zener", h=0.5e-9, vr=[0.9, 1.5, 2.5], xj=C.CORNER_C, length=C.CORNER_L, check=False, fd=fd)
    m = _bands(r)
    nl = [q for q in r["rows"] if q["kind"] == "nonlocal"]
    lo = [q for q in r["rows"] if q["kind"] == "local"]
    for a, b, pn, pl in zip(nl, lo, p["nl"], p["loc"]):
        assert a["vr"] == b["vr"] == pn["vr"] == pl["vr"]
        Gp, Gl = np.array(a["Gp"]), np.array(b["Gloc_post"])
        assert a["I"] > 0 and b["I"] > 0
        assert abs(np.sum(a["Gn"]) / np.sum(a["Gp"]) - 1.0) < 1e-12
        assert abs(Gp[m["arm_v"]].sum() / Gp[m["arm_h"]].sum() - 1.0) < 0.01
        ratio = {k: Gl[v].sum() / Gp[v].sum() for k, v in m.items()}
        planar = pl["I"] / pn["I_p"]
        budget = abs(ratio["mid"] - ratio["far"]) / planar + 0.005
        print("Vr", a["vr"], "device local/nonlocal %.3f" % (b["I"] / a["I"]), "planar 1-D %.3f" % planar,
              " ".join("%s %.3f" % kv for kv in ratio.items()), "budget %.3f" % budget)
        assert abs(ratio["far"] / planar - 1.0) <= budget
