"""D2d (v0.14.0): BIPOLAR density-gradient -- electrons AND holes each carry a quantum potential
(physics_density_gradient.setup_bipolar_dg / setup_contact_bipolar_dg / setup_dg_insulator_wall /
seed_bipolar_dg). The physics oracles are the functions of validation/bipolar_density_gradient.py
(run it for the full report); every one is independent of the code under test. Formulation and the
measured numbers: docs/audit/2026-11-03-devsim-bipolar-density-gradient.md.

On the integration base e46bd25 every physics gate here FAILS (there is no hole quantum potential and
no setup_bipolar_dg); the two defaults-off digest gates PASSED there by construction (they pinned that
the pre-D2d definitions were unchanged) until the 2026-11-04 two-Bernoulli flux re-recorded them (the
cross-version numeric half: tests/test_devsim_sg_flux.py)."""
import os
import sys

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validation.bipolar_density_gradient as V  # noqa: E402

# sha256 of tests/_dg_calllog.record_call_log() -- the UNIPOLAR in-Newton DG definition, character-for-
# character. RE-RECORDED 2026-11-04 (139 calls) for the two-Bernoulli SG flux (sg_flux; user decision):
# the e46bd25 / 6595bde value 6bcbe925...febad4 (127 calls) is the kahan3 flux. The call logs differ
# ONLY in the flux models (measured: the ElectronCurrent / ElectronCurrentDG strings and derivatives,
# Bern_g_dg's clamp, the new Bern_g_m / Bern_g_dg_m and their derivatives; +12 calls). What the old
# digest pinned numerically is kept by tests/test_devsim_sg_flux.py's cross-version gate (the fixture
# SOLVED vs 6595bde, to the flux change's measured round-off).
_DG_CALLLOG_DIGEST = "bc52c7927c657f04917d1c2d792ba5cfe8a7365ae10fb4f7352cc73537ff480a"
_DG_CALLLOG_CALLS = 139
# sha256 of tests/_bipolar_calllog.record_call_log() -- the bipolar DD definition with no options.
# RE-RECORDED 2026-11-04 (237 calls) for the two-Bernoulli SG flux: the v0.13.0 (main 5cbd998) /
# e46bd25 / 6595bde value ac63536e...b0c6aa (217 calls) is the kahan3 flux; the logs differ only in
# ElectronCurrent / HoleCurrent and their derivatives plus Bern_g_m / Bern_gp_m (+20 calls).
# Numerically pinned against 6595bde by tests/test_devsim_sg_flux.py.
_BIPOLAR_CALLLOG_DIGEST = "3542731548315655dff28454e48cca4a2dcf69389e22881dad88a879bb205f80"
_BIPOLAR_CALLLOG_CALLS = 237


# ---- (e) defaults off: byte-identical ---------------------------------------------------------

def test_unipolar_dg_definition_is_byte_identical_to_the_recorded_digest():
    """The shared kernel (_install_dg_variables), the generalized contact repoint and the hole-aware
    set_dg_gamma leave the UNIPOLAR DG's every DEVSIM definition call -- expression strings,
    parameter values, equation bindings, in order -- exactly as recorded (e46bd25 up to the
    2026-11-04 flux re-record; see _DG_CALLLOG_DIGEST)."""
    import _dg_calllog as CL
    log = CL.record_call_log()
    assert len(log) == _DG_CALLLOG_CALLS
    assert CL.digest(log) == _DG_CALLLOG_DIGEST


def test_bipolar_dd_without_dg_is_byte_identical():
    """No bipolar DG requested -> the bipolar DD definition is the recorded one (v0.13 / e46bd25 up to
    the 2026-11-04 flux re-record; the D2d code only runs from setup_bipolar_dg; the SRH kernel's new
    `scale` keyword defaults to None)."""
    import _bipolar_calllog as CL
    log = CL.record_call_log()
    assert len(log) == _BIPOLAR_CALLLOG_CALLS
    assert CL.digest(log) == _BIPOLAR_CALLLOG_DIGEST


def test_srh_kernel_scale_default_is_the_legacy_string():
    from dynameta.carriers.physics_traps import srh_rate_expr, srh_reference_densities
    assert srh_rate_expr() == ("(Electrons*Holes - n_i^2)/(taup*(Electrons + n1) + "
                               "taun*(Holes + p1))")
    assert srh_reference_densities() == ("n1", "p1", "n_i^2")
    assert srh_reference_densities(bgn=True) == ("n1*n_ie/n_i", "p1*n_ie/n_i", "n_ie^2")
    # the DG substitution composes ON TOP of BGN: n1 p1 = n_i'^2 s^2 is kept
    assert srh_reference_densities(bgn=True, scale="S") == ("n1*n_ie/n_i*S", "p1*n_ie/n_i*S",
                                                            "n_ie^2*S^2")


# ---- guards ---------------------------------------------------------------------------------------

def test_guards():
    from dynameta.carriers import physics_density_gradient as DG
    with pytest.raises(ValueError, match="m_eff_n_kg and/or m_eff_p_kg"):
        DG.setup_bipolar_dg("d", "r")
    with pytest.raises(ValueError, match="u_floor_rel"):
        DG.setup_dg_insulator_wall("d", "w", u_floor_rel=0.0)


def test_unipolar_region_is_refused_and_wall_needs_a_solved_device():
    """setup_bipolar_dg refuses a region with no Holes (the unipolar DG is setup_dg_quantum_correction);
    setup_dg_insulator_wall refuses a wall whose carrier density is not yet positive (its floor is
    relative to the classical solution)."""
    from dynameta.constants import M_E
    from dynameta.carriers import physics_density_gradient as DG
    from dynameta.carriers.physics_drift_diffusion import setup_semiconductor_region_dd
    dev, mesh = "bdg_guard", "bdg_guard_m"
    try:
        V._drop(dev, mesh)
        with V._quiet():
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="wall")
            ds.add_1d_mesh_line(mesh=mesh, pos=10e-9, ps=1e-9, tag="sub")
            ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
            ds.add_1d_contact(mesh=mesh, name="sub", tag="sub", material="metal")
            ds.add_1d_region(mesh=mesh, material="ITO", region="u", tag1="wall", tag2="sub")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            setup_semiconductor_region_dd(dev, "u", n_bg_m3=1e26, eps_static=9.5,
                                          dos_mass_kg=0.35 * M_E, mobility_m2Vs=0.004)
        with pytest.raises(ValueError, match="not a bipolar region"):
            DG.setup_bipolar_dg(dev, "u", m_eff_n_kg=0.35 * M_E)
    finally:
        V._drop(dev, mesh)
    dev, mesh = "bdg_guard2", "bdg_guard2_m"
    try:
        reg = V._bar(dev, mesh, 20e-9, quantum_mesh=False)
        ds.set_node_values(device=dev, region=reg, name="Electrons",
                           values=[0.0] * V._get(dev, reg, "x").size)
        with V._quiet():
            DG.setup_bipolar_dg(dev, reg, m_eff_n_kg=V.MN)
        with pytest.raises(ValueError, match="not positive"):
            DG.setup_dg_insulator_wall(dev, "wall")
    finally:
        V._drop(dev, mesh)


def test_contacts_report_the_dg_flux_for_both_carriers_incl_circuit():
    """audit S1-5 for both carriers: after setup_contact_bipolar_dg every ohmic contact's continuity
    equation reports its terminal current from ElectronCurrentDG / HoleCurrentDG, and a circuit contact
    keeps its circuit_node. Also: the region equations keep their generation composites."""
    from dynameta.carriers import eq_registry as R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_density_gradient as DG
    dev, mesh = "bdg_cc", "bdg_cc_m"
    try:
        V._drop(dev, mesh)
        with V._quiet():
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=5e-9, tag="a")
            ds.add_1d_mesh_line(mesh=mesh, pos=100e-9, ps=5e-9, tag="c")
            ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
            ds.add_1d_contact(mesh=mesh, name="c", tag="c", material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="a", tag2="c")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            ds.node_model(device=dev, region="s", name="NetDoping",
                          equation="ifelse(x < 5e-8, -1e23, 1e22)")
            BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                    mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-7,
                                    tau_p_s=1e-7, fd_enhancement=False)
            BP.setup_contact_ohmic_bipolar(dev, "a")
            BP.setup_contact_ohmic_bipolar_circuit(dev, "c", node_name="ccn", source_name="Vcc")
            assert DG.setup_bipolar_dg(dev, "s", m_eff_n_kg=V.MN,
                                       m_eff_p_kg=V.MP) == ("electrons", "holes")
            for c in ("a", "c"):
                DG.setup_contact_bipolar_dg(dev, c)
        assert DG.bipolar_dg_carriers(dev, "s") == ("electrons", "holes")
        for c in ("a", "c"):
            for eq, cur in (("ElectronContinuityEquation", "ElectronCurrentDG"),
                            ("HoleContinuityEquation", "HoleCurrentDG")):
                rec = R.recorded(dev, eq, scope="contact", loc=c)
                assert len(rec) == 1 and rec[0]["kwargs"]["edge_current_model"] == cur
                assert rec[0]["kwargs"].get("circuit_node") == ("ccn" if c == "c" else None)
        for eq, cur, gen in (("ElectronContinuityEquation", "ElectronCurrentDG", "ElectronGeneration"),
                             ("HoleContinuityEquation", "HoleCurrentDG", "HoleGeneration")):
            kw = R.recorded(dev, eq, scope="region", loc="s")[0]["kwargs"]
            assert kw["edge_model"] == cur and kw["node_model"] == gen
            assert ds.get_equation_command(device=dev, region="s", name=eq)["edge_model"] == cur
        # the quantum SRH correction is ONE additive term with its pair rate (noise inventory)
        assert BP.recombination_terms(dev, "s")["node"][DG.DG_SRH_TERM] == ("RTerm_dgsrh",
                                                                            "PTerm_dgsrh")
        assert "ElectronGeneration:QLambdaP" in ds.get_node_model_list(device=dev, region="s")
    finally:
        V._drop(dev, mesh)


def test_bgn_quasi_fields_compose_with_the_quantum_potentials():
    """The DG fluxes are built ON TOP of the region's own quasi-field: with BGN on, vdiff_g carries the
    band-edge D_c and vdiff_g_dg = vdiff_g + dLambda_n/(V_t g) adds the quantum shift, and the quantum
    SRH reference composes n_ie^2 with the quantum factor."""
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_density_gradient as DG
    from dynameta.carriers.bipolar_options import BGN_SI_SLOTBOOM_1977, BipolarOptions
    dev, mesh = "bdg_bgn", "bdg_bgn_m"
    try:
        V._drop(dev, mesh)
        with V._quiet():
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=5e-9, tag="a")
            ds.add_1d_mesh_line(mesh=mesh, pos=100e-9, ps=5e-9, tag="c")
            ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
            ds.add_1d_contact(mesh=mesh, name="c", tag="c", material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="a", tag2="c")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            ds.node_model(device=dev, region="s", name="NetDoping", equation="1e25")
            BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                    mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-7,
                                    tau_p_s=1e-7, fd_enhancement=False,
                                    options=BipolarOptions(bgn=BGN_SI_SLOTBOOM_1977))
            DG.setup_bipolar_dg(dev, "s", m_eff_n_kg=V.MN, m_eff_p_kg=V.MP)
        nodes = ds.get_node_model_list(device=dev, region="s")
        assert "BgnDc" in nodes and "n_ie" in nodes and "RTerm_dgsrh" in nodes
        # at Lambda = 0 (the seed) the quantum SRH correction vanishes exactly and the DG flux
        # argument equals the BGN quasi-field one
        with V._quiet():
            ds.set_node_values(device=dev, region="s", name="Electrons",
                               values=[1e25] * V._get(dev, "s", "x").size)
            ds.set_node_values(device=dev, region="s", name="Holes",
                               values=[1e7] * V._get(dev, "s", "x").size)
        r = np.asarray(ds.get_node_model_values(device=dev, region="s", name="RTerm_dgsrh"))
        assert np.all(r == 0.0)
        a = np.asarray(ds.get_edge_model_values(device=dev, region="s", name="vdiff_g_dg"))
        b = np.asarray(ds.get_edge_model_values(device=dev, region="s", name="vdiff_g"))
        assert np.array_equal(a, b)
    finally:
        V._drop(dev, mesh)


# ---- (a) classical limit --------------------------------------------------------------------------

def test_classical_limit_b0_reproduces_bipolar_dd():
    """GATE A: the bipolar DG ACTIVE at b = 0 reproduces the classical bipolar DD solution of a p-n
    diode (fields and the terminal current) to solver tolerance; Lambda_n = Lambda_p = 0. Measured:
    0 (bit-identical double readouts) at +0.4 / +0.2 / -0.5 V in the extended Newton the validation
    runs since 2026-11-03. (The earlier 1.2e-12 in double held only because the DG flux then repeated
    the classical kahan3 arithmetic bit for bit: the bipolar DG's two-Bernoulli flux differs from it at
    the forward current's ~1e-8 double round-off floor and measured 2.2e-9 here -- validation _diode.)"""
    d = V.diode(gamma_fracs=(None, 0.0), biases=(0.4,))
    cl, dg = d[None][0.4], d[0.0][0.4]
    for k in ("Potential", "Electrons", "Holes"):
        a, b = cl["arrays"][k], dg["arrays"][k]
        assert np.max(np.abs(a - b) / np.abs(a)) < 1e-9, k
    assert abs(dg["I"] / cl["I"] - 1.0) < 1e-9
    assert dg["max_lam_Vt"] < 1e-12


@pytest.mark.parametrize("carrier", ["electrons", "holes"])
def test_dg_edge_flux_is_accurate_at_large_drift(carrier):
    """The bipolar DG Scharfetter-Gummel flux keeps full precision at |v| ~ 20-40, where an insulator-
    wall edge sits by construction (c@wall = 1e-12 c_cl -> |v| ~ 27.6). One 1 nm edge, the drift set
    through the quantum potential on a flat psi; each Bernoulli term isolated by zeroing the other
    node's density. Oracle: the flux bracket c@n1 B(-v) - c@n0 B(v) (electrons) / c@n1 B(v) -
    c@n0 B(-v) (holes) with a 40-digit mpmath B, at the v DEVSIM computed. Gate 1e-13. The kahan3
    form (B(-v) spelled B(v) + v) FAILS it: measured 1.9e-7 at v = -21, 6.5e-5 at -27.6, 100 % at
    -40, so the gate fails on it for both carriers; the two-Bernoulli form measured 3.0e-16
    (electrons) / 3.1e-16 (holes) on both OpenBLAS builds (a4f00d9's plain bracket), and the ONE SG
    kernel's sign-selected bracket (dynameta.carriers.sg_flux, 2026-11-04) 3.3e-16 / 2.2e-16. Every
    other flux family: tests/test_devsim_sg_flux.py."""
    mp = pytest.importorskip("mpmath")
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_density_gradient as DG
    dev, mesh = "bdg_flux", "bdg_flux_m"
    el = carrier == "electrons"
    lam, c, cur, vd = (("QLambda", "Electrons", "ElectronCurrentDG", "vdiff_g_dg") if el else
                       ("QLambdaP", "Holes", "HoleCurrentDG", "vdiff_gp_dg"))
    mu, sgn = ("mu_n", 1.0) if el else ("mu_p", -1.0)
    worst = 0.0
    try:
        V._drop(dev, mesh)
        with V._quiet():
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="a")
            ds.add_1d_mesh_line(mesh=mesh, pos=1e-9, ps=1e-9, tag="c")
            ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
            ds.add_1d_contact(mesh=mesh, name="c", tag="c", material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="a", tag2="c")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            ds.node_model(device=dev, region="s", name="NetDoping", equation="1e22")
            BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                    mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-7,
                                    tau_p_s=1e-7, fd_enhancement=False)
            DG.setup_bipolar_dg(dev, "s", m_eff_n_kg=V.MN, m_eff_p_kg=V.MP)
            other = "Holes" if el else "Electrons"
            for nm, val in (("Potential", 0.0), (other, 1e10), ("QSqrtN", 1.0), ("QSqrtP", 1.0),
                            ("QLambda", 0.0), ("QLambdaP", 0.0)):
                ds.set_node_values(device=dev, region="s", name=nm, values=[val, val])
        vt = float(ds.get_parameter(device=dev, region="s", name="V_t"))
        pref = sgn * float(ds.get_parameter(device=dev, region="s", name="ElectronCharge")) * float(
            ds.get_parameter(device=dev, region="s", name=mu)) * vt * float(
            ds.get_edge_model_values(device=dev, region="s", name="EdgeInverseLength")[0])

        def B(x):                                     # 40 digits, scoped (no global mp.dps write)
            with mp.workdps(40):
                return +(x / mp.expm1(x)) if x != 0 else mp.mpf(1)

        for v in (-40.0, -27.6, -21.0, -5.0, 0.3, 5.0, 21.0, 27.6, 40.0):
            # electrons: v = (Lambda@n0 - Lambda@n1)/V_t ; holes: v = -(Lambda@n0 - Lambda@n1)/V_t
            ds.set_node_values(device=dev, region="s", name=lam,
                               values=[(v if el else -v) * vt, 0.0])
            for c0, c1 in ((0.0, 1e20), (1e20, 0.0)):
                ds.set_node_values(device=dev, region="s", name=c, values=[c0, c1])
                vv = mp.mpf(float(ds.get_edge_model_values(device=dev, region="s", name=vd)[0]))
                got = float(ds.get_edge_model_values(device=dev, region="s", name=cur)[0]) / pref
                ref = (c1 * B(-vv) - c0 * B(vv)) if el else (c1 * B(vv) - c0 * B(-vv))
                worst = max(worst, float(abs(got / ref - 1)))
    finally:
        V._drop(dev, mesh)
    assert worst < 1e-13


# ---- (b) analytic dark space ----------------------------------------------------------------------

def test_flat_wall_dark_space_matches_the_closed_form_for_both_carriers():
    """GATE B: frozen flat potential, impenetrable wall: the DG deficit is I c L_q with
    I = int_0^1 (1-t^2)/sqrt(t^2 ln t^2 - t^2 + 1) dt = 1.187207, for EACH carrier with its OWN L_q
    (1.375 / 1.001 nm), and the whole profile follows z(t) = L_q int_0^t ds/sqrt(F). Measured: rel
    -3.2e-6 (n) / -2.6e-5 (p), profile |dz|/L_q 1.4e-4 / 2.3e-4. The wall's u-floor only fixes the wall
    node's own density: 1e-6 -> 1e-4 moves the deficit by 8.3e-5."""
    ii = V.dg_wall_constant()
    assert ii == pytest.approx(1.187207, abs=2e-6)
    fw = V.flat_wall()
    assert fw["electrons"][2] == pytest.approx(1.3746e-9, rel=1e-4)
    assert fw["holes"][2] == pytest.approx(1.0013e-9, rel=1e-4)
    for key, (dfc, zerr, _lq) in fw.items():
        assert abs(dfc / ii - 1.0) < 1e-4, key
        assert zerr < 1e-3, key
    fw4 = V.flat_wall(u_floor_rel=1e-4)
    for key in fw:
        assert abs(fw4[key][0] / fw[key][0] - 1.0) < 2e-4, key


# ---- (b) MOS vs Schrodinger-Poisson: electrons AND holes (the p-channel is the new physics) --------

@pytest.mark.parametrize("carrier", ["electrons", "holes"])
def test_mos_accumulation_vs_schrodinger_poisson(carrier):
    """GATE C: strong accumulation (N = 1e24 m^-3, |psi_s - psi_b| = 0.15 V, FD-consistent DD) vs the
    self-consistent Schrodinger-Poisson (holes by the exact mirror). Classical DD: sheet +177/+125 %,
    centroid -64/-56 %, peak AT the wall. DG at gamma = 1 (Ancona-Iafrate; no fit) closes >= 70 % of
    that gap (measured residual 0.12-0.14 sheet, 0.27 centroid); DG at gamma_wall = 1.672 -- fixed a
    priori by the exact free-gas wall (GATE B), NOT fitted to the MOS -- lands within 8 % (measured
    +5.6/-5.1 % electrons, +4.0/-3.7 % holes), peak at 0.84 / 0.82 x the SP depth."""
    gw = V.gamma_wall()
    assert gw == pytest.approx(1.6717, abs=1e-3)
    r = V.mos_vs_sp(carrier, gammas=(1.0, gw))
    sp, cl = r["SP"], r["classical"]
    assert r["meta"]["converged"]
    assert cl["peak"] == 0.0 and abs(cl["sheet"] / sp["sheet"] - 1.0) > 1.0
    dg1, dgw = r[("DG", 1.0)], r[("DG", gw)]
    for k in ("sheet", "centroid"):
        assert abs(dg1[k] - sp[k]) <= 0.30 * abs(cl[k] - sp[k]), k
        assert abs(dgw[k] / sp[k] - 1.0) <= 0.08, k
    assert 0.5 <= dg1["peak"] / sp["peak"] <= 1.5
    assert 0.75 <= dgw["peak"] / sp["peak"] <= 1.25


# ---- (b') fixed point -----------------------------------------------------------------------------

@pytest.mark.parametrize("carrier", ["electrons", "holes"])
def test_dg_density_is_the_dg_bvp_on_its_own_potential(carrier):
    """GATE D: the converged coupled DG density equals the independent post-hoc DG BVP (scipy
    solve_bvp, carriers.density_gradient) on the DG solution's own potential. Measured 2.7e-4 (n),
    4.0e-4 (p) of the peak."""
    assert V.fixed_point(carrier) < 2e-3


# ---- (c) equilibrium ------------------------------------------------------------------------------

def test_inversion_equilibrium_flat_qfls_and_np():
    """GATE E: inverted MOS, both carriers quantum at the wall: the modified quasi-Fermi levels are flat
    (measured 3.3e-16 / 5.6e-17 V on scipy's OpenBLAS, 4.3e-15 / 5.6e-17 V on Elmer's), n p = n_i^2
    exp((Ln + Lp)/V_t) at every node (7.1e-15 / 1.6e-13) and n_i^2 in the neutral bulk (4.2e-7, the
    1e-6 solve tolerance). quantum_srh=False breaks detailed balance: phi drifts by 6 mV off the wall
    node.
    TIGHTENED 2026-11-03 from 1e-8 / 1e-7 to 1e-11 / 1e-10: the 1.7e-10 V and 6.4e-9 measured until
    then were not a solve floor but the kahan3 flux's B(v) + v cancellation on the electron wall edge
    (|v| = 27.6; the kernel is now dynameta.carriers.sg_flux), so the new bounds fail the old flux by 16x /
    64x in double and keep >= 600x margin (in the extended Newton the validation now runs, the old
    flux's loss would shrink to ~1e-22: test_dg_edge_flux_is_accurate_at_large_drift pins the flux
    itself)."""
    eq = V.inversion_equilibrium(quantum_srh=True)
    assert eq["max_phi_n"] < 1e-11 and eq["max_phi_p"] < 1e-11
    assert eq["np_quantum"] < 1e-10 and eq["np_bulk"] < 1e-6
    assert eq["classical_peak_nm"] == 0.0 and eq["inversion_peak_nm"] > 0.5
    eq0 = V.inversion_equilibrium(quantum_srh=False)
    assert eq0["max_phi_interior"] > 1e-4


# ---- (d) transport --------------------------------------------------------------------------------

def test_diode_transport_is_a_small_first_order_change():
    """GATE F: in a 1e23/1e22 Si diode the DG changes the forward current by -8.3e-5 at +0.4 V
    (max|Lambda| 0.02 V_t), linearly in gamma (ratio 1.997), and the terminal currents still conserve."""
    d = V.diode(gamma_fracs=(None, 0.5, 1.0), biases=(0.4,))
    cl = d[None][0.4]
    r1 = d[1.0][0.4]["I"] / cl["I"] - 1.0
    rh = d[0.5][0.4]["I"] / cl["I"] - 1.0
    assert 0.0 < abs(r1) < 1e-2 and abs(r1) <= 2.0 * d[1.0][0.4]["max_lam_Vt"]
    assert 1.8 < r1 / rh < 2.2
    assert d[1.0][0.4]["I_sum"] < max(1e-6, 10.0 * cl["I_sum"])
