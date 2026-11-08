"""v0.15 plan item 4.1.3: nonlocal band-to-band tunnelling in the DEVSIM Newton, 1-D (carriers/btbt_nonlocal.py,
a custom_terms term). The kernel gates (closed forms, uniform-field identity, continuum oracle) are in
tests/test_btbt_nonlocal_kernels.py; this file gates the coupling. Every case that adds the term runs in a
CHILD process (tests/_btbt_nonlocal_child.py): DEVSIM's custom-equation registration is permanent.

OLD CODE (66f6185): dynameta.carriers.btbt_nonlocal does not exist -- every child fails at its import and the
in-process refusal tests fail at theirs (MEASURED by running this file against an export of 66f6185).
The physical contrast is gated too: at V_r = 0.1 V the nonlocal Zener current is exactly 0 (no final state:
every path is too long) while the local model gives 2.6e-2 A/m^2 (N7).

  D0  defaults off: importing the module adds only itself (beyond tunneling), loads no devsim and registers
      nothing; every refusal fires before any DEVSIM state changes (no registration); the local model refuses a
      region with a live nonlocal term (BTBTDoubleCountError); material= takes the gap from band_data_at_T.
  N3  detailed balance at V = 0 (Zener with E_g,tun = 0.9 eV < the 1.04 V built-in, so paths exist): |R|/one-way
      <= 64 eps x 60 (Boltzmann). Refute: the f_n frame offset delta shifted by 1e-3 kT breaks it. The legacy
      FD g-factor transport (fd_enhancement=True) is REPORTED: its zero-bias state is not a strict equilibrium
      (physics_bipolar_dd.setup_bipolar_region, statistics=None) -- MEASURED 2.7e-4 / 1.6e-4 at 0.2 / 0.1 nm.
  N5  conservation on the Zener sweep: the term's hole-row and electron-row residual shares cancel to round-off;
      the terminal current == q Int G within the SI contact-current floor (the Z3 budget); I_n == I_p; the DC
      Newton needs no more iterations than the local model's (+3 allowed). Refute: a flipped electron sign.
  N6  Jacobian: the term's share of DEVSIM's assembled matrix == Richardson central differences of DEVSIM's own
      residual, including a MID-PATH node (no local model has an entry there) and the node beyond the path end
      (the end slope of the prefactor), both processes; the carrier columns at equilibrium. Refute: the implicit
      end-point derivative dropped.
  N7  local vs nonlocal on the Zener (a MEASUREMENT; no direction is gated -- P1 3.1): within the published-
      model spread (factor 2) at high field (V_r = 2.5, 3.0 V); the ratio at low bias is reported.
  N8  ssac: Re Y(1 Hz) == DC dI/dV (Richardson of solved currents) at 0 V and at reverse bias, on the FDT diode
      (on the 5e24 Zener DEVSIM's ssac has an absolute floor of ~1 S/m^2 with or without any BTBT term).
  N9  noise: S = 4 kT Re Y (1 Hz - 1 GHz) at equilibrium with BTBT >= 50 % of S, through the two-point source
      (hole at the path start, electron at the end); placing both carriers at the start (the local shortcut)
      must FAIL the 1e-4 FDT tolerance.
Runtime: ~50 s serial (MEASURED on the Windows dev box; the FDT diode children are ~10 s each).
"""
import math

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

from dynameta.carriers.devsim_env import import_devsim  # noqa: E402

import_devsim()
import _btbt_nonlocal_child as C  # noqa: E402

EPS = np.finfo(float).eps
Q = 1.602176634e-19
KT4 = 4.0 * 1.380649e-23 * 300.0
ZMU_N, VT = 0.135, 1.380649e-23 * 300.0 / Q


# =====================================================================================================
# D0 defaults off
# =====================================================================================================

def test_d0_import_is_inert():
    r = C.run_child("defaults")
    assert r == {"devsim_loaded": False, "new_dynameta_modules": ["dynameta.carriers.btbt_nonlocal"],
                 "registered": False}


def _wipe():
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)


def test_d0_refusals_fire_before_any_registration():
    from dynameta.carriers import btbt_nonlocal as NL
    from dynameta.carriers import custom_terms as CT
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.bipolar_options import BGN_SI_SLOTBOOM_1977, BipolarOptions
    from dynameta.carriers.tunneling import BTBTDoubleCountError, SILICON_HURKX
    spec = NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12)
    _wipe()
    try:
        # 2-D (since the 2-D walker): a region without the continuity equations is refused
        ds.create_2d_mesh(mesh="nl2d_m")
        for pos in (0.0, 1e-8):
            ds.add_2d_mesh_line(mesh="nl2d_m", dir="x", pos=pos, ps=5e-9)
            ds.add_2d_mesh_line(mesh="nl2d_m", dir="y", pos=pos, ps=5e-9)
        ds.add_2d_region(mesh="nl2d_m", material="Si", region="r")
        ds.finalize_mesh(mesh="nl2d_m")
        ds.create_device(mesh="nl2d_m", device="nl2d")
        with pytest.raises(ValueError, match="no ElectronContinuityEquation"):
            NL.setup_btbt_nonlocal("nl2d", "r", spec)
        with pytest.raises(TypeError):
            NL.setup_btbt_nonlocal("nl2d", "r", SILICON_HURKX)

        def oned(dev, **kw):
            _R.clear(dev)
            ds.create_1d_mesh(mesh=dev + "_m")
            ds.add_1d_mesh_line(mesh=dev + "_m", pos=0.0, ps=1e-9, tag="a")
            ds.add_1d_mesh_line(mesh=dev + "_m", pos=5e-8, ps=1e-9, tag="b")
            ds.add_1d_contact(mesh=dev + "_m", name="a", tag="a", material="metal")
            ds.add_1d_contact(mesh=dev + "_m", name="b", tag="b", material="metal")
            ds.add_1d_region(mesh=dev + "_m", material="Si", region="s", tag1="a", tag2="b")
            ds.finalize_mesh(mesh=dev + "_m")
            ds.create_device(mesh=dev + "_m", device=dev)
            if kw.get("bipolar", True):
                ds.node_model(device=dev, region="s", name="NetDoping", equation="ifelse(x < 2.5e-8, -1e24, 1e24)")
                BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                        mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-6,
                                        tau_p_s=1e-6, fd_enhancement=False, btbt=kw.get("btbt"),
                                        options=kw.get("options"))
            return dev
        with pytest.raises(ValueError, match="no ElectronContinuityEquation"):
            NL.setup_btbt_nonlocal(oned("nl_plain", bipolar=False), "s", spec)
        with pytest.raises(ValueError, match="frozen direction is a 2-D option"):
            NL.setup_btbt_nonlocal("nl_plain", "s", NL.NonlocalBTBT(SILICON_HURKX, Eg_tun_eV=1.12,
                                                                    direction=(1.0, 0.0)))
        with pytest.raises(BTBTDoubleCountError, match="LOCAL BTBT"):
            NL.setup_btbt_nonlocal(oned("nl_loc", btbt=SILICON_HURKX), "s", spec)
        with pytest.raises(NotImplementedError, match="bandgap narrowing"):
            NL.setup_btbt_nonlocal(oned("nl_bgn", options=BipolarOptions(bgn=BGN_SI_SLOTBOOM_1977)), "s", spec)
        assert not CT.is_registered()                     # nothing reached add_custom_term
        assert "HJ_L_Electrons" not in ds.get_node_model_list(device="nl_loc", region="s")
    finally:
        _wipe()
        for d in ("nl_plain", "nl_loc", "nl_bgn"):
            _R.clear(d)


def test_d0_temperature_advisory():
    r = C.run_child("warn_T")
    assert len(r["T350"]) == 1 and r["T350"][0][0] == "AdvisoryWarning" and "no temperature law" in r["T350"][0][1]
    assert r["Tref"] == []


def test_d0_local_after_nonlocal_is_refused():
    """The reverse refusal (tunneling.setup_btbt): a region with a live nonlocal term refuses the local model
    with BTBTDoubleCountError and gets no BTBT_U model; a region without one installs normally. OLD CODE
    (merge base c6aa0a4 + this branch's nonlocal module, tunneling.py unpatched): setup_btbt installs the
    second model silently -- MEASURED: raised None."""
    r = C.run_child("reverse_refusal")
    assert r["raised"] is not None and "NONLOCAL" in r["raised"]
    assert not any(m.startswith("BTBT_U") for m in r["models_after"])
    assert r["other_models"] == ["BTBT_U"]


def test_d0_gap_from_material_kernel():
    """material= (band_data_at_T) at T_REF gives the explicit-gap device bit for bit (solution and rates); at
    350 K with a band-temperature model the term's path gap and conversion gap ARE band_data_at_T's values
    and no temperature advisory fires (the explicit-gap device at 350 K warns: test_d0_temperature_advisory)."""
    r = C.run_child("gap_material", eg=1.12, law=[1.17, 4.73e-4, 636.0])
    assert r["explicit"]["R_hex"] == r["material"]["R_hex"] and r["explicit"]["psi_hex"] == r["material"]["psi_hex"]
    t = r["T350"]
    assert t["Eg_path"] == t["kernel_path"] and t["Eg_set"] == t["kernel_set"] and t["warnings"] == []
    assert t["Eg_path"] < t["Eg_set"]


# =====================================================================================================
# N3 detailed balance
# =====================================================================================================

@pytest.fixture(scope="module")
def equilibrium():
    return C.run_child("equilibrium", h=[0.2e-9, 0.1e-9])


def test_n3_detailed_balance_boltzmann(equilibrium):
    """f_n(l) - f_p(0) is evaluated as sinh((a-b)/2)/(2 cosh cosh) from the two occupancy arguments, each a
    sum of O(1)..O(60) log activities and psi/V_t: round-off <= 64 eps x 60 = 8.5e-13 relative to the
    one-way rate (MEASURED 1.1e-14 / 7.1e-15 at 0.2 / 0.1 nm, ~20 paths each, one-way up to 6e19 m^-3 s^-1)."""
    for h in ("2e-10", "1e-10"):
        r = equilibrium["boltz_" + h]
        assert r["paths"] >= 15 and r["one_way_max"] > 1e19
        assert r["ratio"] <= 64 * EPS * 60, r


def test_n3_fd_transport_reported(equilibrium):
    """CONSISTENCY CHECK, not a gate: under the legacy FD g-factor transport the zero-bias state carries the
    documented FD/Boltzmann inconsistency (setup_bipolar_region, statistics=None), so E_F is not flat to
    round-off. Asserted only that the imbalance is small and shrinks under junction refinement."""
    a, b = equilibrium["fd_2e-10"]["ratio"], equilibrium["fd_1e-10"]["ratio"]
    print("FD transport |R|/one-way at 0.2 / 0.1 nm:", a, b)
    assert b < a < 1e-3


def test_n3_refute_frame_offset_breaks_detailed_balance():
    """delta (the statistics-frame offset of f_n, ln(N_c N_v/n_i^2) - E_g,tun/kT) shifted by 1e-3: the ratio
    becomes ~1e-3 (MEASURED below), 1e9 x the tolerance."""
    r = C.run_child("equilibrium", h=[0.2e-9], fd=[False], delta_shift=1e-3)["boltz_2e-10"]
    print("shifted delta ratio", r["ratio"])
    assert r["ratio"] > 1e-4


# =====================================================================================================
# N5 conservation + Newton, N7 local vs nonlocal
# =====================================================================================================
VRS = [0.1, 0.3, 0.6, 0.9, 1.2, 2.0, 2.5, 3.0]


@pytest.fixture(scope="module")
def zener():
    return C.run_child("zener", vr=VRS)


def test_n5_conservation_and_newton(zener):
    """(a) The term's hole-row share and electron-row share of DEVSIM's residual are equal and opposite: the
    same q R V_i per path enters both, so they cancel to the round-off of summing ~N_paths terms:
    |sum_h + sum_e| <= 4 N eps sum|h|. (b) The terminal current == q Int G (SRH at tau = 1 ms and the minority
    diffusion are < 1e-7 A/m^2): the SI contact current is a difference of majority SG terms of
    q mu_n V_t N/h = 1.4e12 A/m^2 (2 nm contact spacing), so the floor is 32 ulps of that + 1e-4 I (the Z3
    budget of tests/test_devsim_tunneling.py). (c) I_n == I_p (one amount per path, split (1-t), t).
    (d) The coupled Newton converges in no more iterations than the local model + 3 (plan N5; MEASURED equal
    at every bias: 7-13)."""
    floor = 32.0 * EPS * Q * ZMU_N * VT * C.ZN / 2e-9
    for row, loc in zip(zener["nl"], zener["loc"]):
        n = max(row["stats"]["paths"], 1)
        assert abs(row["hole_share"] + row["elec_share"]) <= 4 * n * EPS * row["abs_share"] + 1e-300, row
        assert abs(row["hole_share"] - row["I_p"]) <= 4 * n * EPS * abs(row["I_p"]) + 1e-300
        assert abs(row["I_n"] - row["I_p"]) <= 4 * n * EPS * abs(row["I_p"]) + 1e-300
        assert abs(row["J"] - row["I_p"]) <= floor + 1e-4 * abs(row["I_p"]), row
        assert row["its"] <= loc["its"] + 3
    assert zener["eq"]["I_p"] == 0.0 and zener["eq"]["stats"]["paths"] == 0   # built-in 1.04 V < E_g


def test_n5_refute_flipped_sign_is_not_conserved():
    r = C.run_child("flip")
    assert abs(r["hole_share"] + r["elec_share"]) > 1.9 * r["abs_share"]


def test_n7_local_vs_nonlocal_zener(zener):
    """A MEASUREMENT (P1 3.1 premise check: nonlocal is not generally lower). Same Hurkx (A, B) in both;
    the nonlocal model takes it as Sentaurus' default nonlocal set. MEASURED local/nonlocal: V_r = 0.1 V:
    nonlocal exactly 0 (no path is short enough: no final state within reach) vs 2.6e-2 A/m^2 local;
    0.3 V: 1134; 0.6: 49; 0.9: 12.8; 1.2: 6.1; 2.0: 2.2; 2.5: 1.45; 3.0: 0.96. Asserted: within the
    published-model spread (factor 2, plan N7) at V_r >= 2.5 V, and nonlocal == 0 at 0.1 V."""
    ratios = {r["vr"]: (l_["I"] / r["I_p"] if r["I_p"] else math.inf) for r, l_ in zip(zener["nl"], zener["loc"])}
    print("local / nonlocal:", ratios)
    assert ratios[0.1] == math.inf and zener["loc"][0]["I"] > 1e-2
    for vr in (2.5, 3.0):
        assert 0.5 <= ratios[vr] <= 2.0


# =====================================================================================================
# N6 Jacobian
# =====================================================================================================

def _jac_errors(recs, var_filter, per_row):
    """max over columns of |J - Richardson FD| / tol, tol = 1e-6 scale + 2 |fd - fd2| (the FD's own
    truncation estimate). scale: the largest |J| of the ROW over the tested columns (per_row) or of the
    column. Returns (worst ratio, worst relative error)."""
    sel = [q for q in recs if q["var"] in var_filter]
    J = np.array([q["J"] for q in sel])                         # (ncol, nrow)
    fd = -np.array([q["fd"] for q in sel])
    fd2 = -np.array([q["fd2"] for q in sel])
    rich = (4.0 * fd - fd2) / 3.0
    scale = (np.max(np.abs(J), axis=0)[None, :] if per_row else np.max(np.abs(J), axis=1)[:, None])
    tol = 1e-6 * scale + 2.0 * np.abs(fd - fd2)
    return float(np.max(np.abs(J - rich) / tol)), float(np.max(np.abs(J - rich) / scale))


@pytest.mark.parametrize("P", [2.5, 2.0])
def test_n6_jacobian_matches_fd_of_devsim_residual(P):
    """Reverse-biased Zener (0.9 V), the strongest path: Potential columns at start-1, start, start+1, a
    mid-path node, a, b and the node beyond b (the end slope of the slope-interpolated prefactor) against
    central differences of DEVSIM's residual (steps 1e-5 V and 2e-5 V, Richardson). Budget 1e-6 of the row
    scale (the J1 pattern) + 2 |fd - fd2|. MEASURED <= 1e-7 relative. The mid-path entry exists (a local
    model has none) and matches."""
    r = C.run_child("jacobian", P=P, vr=0.9)
    worst, rel = _jac_errors(r["recs"], ("Potential",), per_row=True)
    print("P", P, "Potential columns: worst/tol", worst, "worst rel", rel)
    assert worst <= 1.0
    mid = next(q for q in r["recs"] if q["label"] == "mid")
    rowscale = max(max(abs(v) for v in q["J"]) for q in r["recs"] if q["var"] == "Potential")
    assert max(abs(v) for v in mid["J"]) > 1e-3 * rowscale


def test_n6_carrier_columns_at_equilibrium():
    """The occupancy derivatives (n at the end edge, p at the start) are below the FD round-off floor in
    reverse bias (f_n ~ 0, f_p ~ 1); at equilibrium (E_g,tun = 0.9 eV, paths at V = 0) they are live.
    Column-scaled budget 1e-6 + 2 |fd - fd2| (MEASURED <= 6e-6 relative, at the FD noise)."""
    r = C.run_child("jacobian", P=2.5, eg=0.9, vr=0)
    worst, rel = _jac_errors(r["recs"], ("Electrons", "Holes"), per_row=False)
    print("carrier columns: worst/tol", worst, "worst rel", rel)
    assert worst <= 1.0
    worst_p, _ = _jac_errors(r["recs"], ("Potential",), per_row=True)
    assert worst_p <= 1.0


def test_n6_refute_dropped_end_point_derivative():
    r = C.run_child("jacobian", P=2.5, vr=0.9, drop_dt=True)
    worst, rel = _jac_errors(r["recs"], ("Potential",), per_row=True)
    print("dropped dt: worst/tol", worst, "worst rel", rel)
    assert worst > 1e3


# =====================================================================================================
# N8 ssac, N9 noise
# =====================================================================================================

@pytest.fixture(scope="module", params=[2.5, 2.0], ids=["phonon", "direct"])
def fdt(request):
    return request.param, C.run_child("fdt", P=request.param, same_node=True, vr=0.2)


def test_n8_ssac_equals_dc_conductance(fdt):
    """Re Y(1 Hz) against dI/dV from solved DC currents. Reference: Richardson of the +-dv and +-2dv central
    differences at dv = 1e-4 V (G); its own error is estimated from the same reference at dv = 2e-4 V
    (G'): the size of the Richardson correction |G_central/G - 1| plus 2 |G'/G - 1| (path ends crossing
    nodes inside the stencil make I(V) only C^1 with Lipschitz kinks -- module doc section 3 -- which
    Richardson does not remove), plus 2e-6 (the agreement of the same check on this device WITHOUT the
    term, MEASURED 1.4e-6). MEASURED 6.6e-7 at 0 V and 4.8e-6 at -0.2 V (phonon)."""
    P, r = fdt
    for tag, Y in (("0", r["ReY"][0]), ("r", r["Yr"][0])):
        G, G1, G2 = r["G" + tag], r["G" + tag + "_1"], r["G" + tag + "_2"]
        dev = abs(Y / G - 1.0)
        tol = abs(G1 / G - 1.0) + 2.0 * abs(G2 / G - 1.0) + 2e-6
        print("P", P, tag, "ReY/G - 1", Y / G - 1.0, "tol", tol, "G'/G - 1", G2 / G - 1.0)
        assert dev <= tol


def test_n9_noise_fdt_through_the_two_point_source(fdt):
    """S = 4 kT Re Y to 1e-4 (the D2 gate-B tolerance) at 1 Hz .. 1 GHz with the nonlocal BTBT >= 50 % of S at
    1 Hz, and detailed balance on this device. MEASURED <= 7.2e-7 (phonon; direct below)."""
    P, r = fdt
    S, Y = np.array(r["S"]), np.array(r["ReY"])
    dev = S / (KT4 * Y) - 1.0
    print("P", P, "S/4kTReY - 1", dev, "btbt share", r["share_btbt"], "detailed balance", r["db_ratio"])
    assert r["share_btbt"] >= 0.5
    assert np.max(np.abs(dev)) <= 1e-4
    assert r["db_ratio"] <= 64 * EPS * 60


def test_n9_proveout_same_node_source_fails_fdt(fdt):
    """Both carriers of each event at the path START (the local model's same-node pair): MEASURED -2.5e-4 at
    1 Hz (phonon), > 1e-4: the separated placement is what the FDT needs."""
    P, r = fdt
    dev = np.array(r["S_same"]) / (KT4 * np.array(r["ReY"])) - 1.0
    print("P", P, "same-node S/4kTReY - 1", dev)
    assert abs(dev[0]) > 1e-4


def test_n5_reverse_bias_carriers_balance_on_the_fdt_diode(fdt):
    """At -0.2 V the BTBT current dominates the FDT diode's reverse current: the electron and hole
    generation integrals are equal to round-off (one amount per path, split (1 - t), t at the end)."""
    P, r = fdt
    assert r["Ip_r"] > 0.0
    assert abs(r["In_r"] - r["Ip_r"]) <= 1e-13 * r["Ip_r"]
