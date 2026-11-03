"""Gates for small-signal NOISE by the impedance-field method (dynameta.carriers.noise_analysis,
DynaMeta v0.14.0, workstream D2a).

EVERY ORACLE IS INDEPENDENT OF THE CODE UNDER TEST: Johnson-Nyquist with the SOLVED DC conductance
(+-1 mV central difference of DEVSIM terminal currents), the fluctuation-dissipation theorem with the
ssac admittance (ac_analysis, a different linear solve), the ideal-diode law 2qI coth(qV/2kT) with
the solved DC current, the closed-form ambipolar GR Lorentzian, Hooge's alpha I^2/(N f), and an
independent scipy solve of the adjoint system built from ds.get_matrix_and_rhs. Constants in the
oracles come from scipy.constants (validation/devsim_noise.py), not dynameta.constants.

EVERY GATE FAILS ON THE PRE-D2a CODE (feat/devsim-2.11-v0.14 at d3a7012): there is no
dynameta.carriers.noise_analysis (ImportError), no drive= keyword on the circuit contacts and no
physics_drift_diffusion.setup_contact_ohmic_dd_circuit. So that a stub returning "4kT Re Y" could not
pass either, gate E (diffusion vs shot) requires S = 2qI = (1/2) 4kT G_d on a forward diode AND
S = 4kT G != 2qI on a biased resistor, and gate C requires the full coth transition.

Fixtures are tiny 1-D/2-D/3-D devices built by validation/devsim_noise.py (one implementation of each
builder); the whole file runs in ~20 s. devsim self-skips; the 3-D case also needs gmsh.
"""
import math

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
# NOT importorskip: on code without noise_analysis the fixture module's ImportError must FAIL every
# gate (importorskip would turn it into a silent skip).
try:
    import validation.devsim_noise as V
    _IMPORT_ERROR = None
except ImportError as _e:                                     # pragma: no cover - old code only
    V, _IMPORT_ERROR = None, _e


@pytest.fixture(autouse=True)
def _clean():
    if _IMPORT_ERROR is not None:
        pytest.fail("noise_analysis unavailable: {!r}".format(_IMPORT_ERROR))
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()


# =====================================================================================================
# GATE A: Johnson-Nyquist
# =====================================================================================================

def _johnson(build, ae):
    build()
    G = V.dc_conductance(0.0, ae)
    return V.noise(V.FREQS_A).S / (4 * V.KT * G), G


def test_johnson_bipolar_1d():
    """GATE A1. n-type Si bar 1e22 m^-3, 1 um, NON-uniform mesh (100/27 nm), equilibrium: S_I = 4kT G
    at 1 Hz, 1 MHz, 1 THz. Measured |S/4kTG - 1| = 8.9e-16 (the edge sum is the box scheme's discrete
    Dirichlet energy, so the identity is exact in the discrete sense, not just to O(h^2)); G_dc equals
    sigma A/L to 1.4e-12."""
    r, G = _johnson(lambda: V.bipolar_1d(
        "res1d", [(0.0, 1e-7, "lo"), (3.3e-7, 2.7e-8, None), (1e-6, 1e-7, "hi")], "1e22",
        n_i=1e16, tau=1e-7), 1e8)
    assert np.max(np.abs(r - 1)) < 1e-9
    assert G == pytest.approx(V.QE * V.MU_N * 1e22 / 1e-6, rel=1e-9)


def test_johnson_unipolar_1d_fermi_dirac():
    """GATE A2. Unipolar DD ITO bar (n = 5e26, eta ~ 10, FD g-factor ON) through the new
    setup_contact_ohmic_dd_circuit: S = 4kT G to 1.1e-16 (uniform n -> g cancels at equilibrium)."""
    r, _ = _johnson(V.unipolar_1d_resistor, 5e14)
    assert np.max(np.abs(r - 1)) < 1e-9


def test_johnson_bipolar_2d():
    """GATE A3. 2-D bipolar bar 1 x 0.4 um (non-uniform x and y lines, triangles): S = 4kT G per unit
    depth to 8.9e-16; G_dc = sigma W/L to 1e-9."""
    r, G = _johnson(V.bipolar_2d_resistor, 1e9)
    assert np.max(np.abs(r - 1)) < 1e-9
    assert G == pytest.approx(V.QE * V.MU_N * 1e23 * 0.4, rel=1e-6)


def test_johnson_unipolar_3d():
    """GATE A4. 3-D unipolar ITO bar on an unstructured gmsh tetra mesh: S = 4kT G to 1.1e-16 (G is
    the DEVSIM DC conductance of THAT mesh -- the bar's own Ohm's-law error vs sigma A/L is mesh-limited,
    validation/carriers_3d_resistor.py -- and the noise identity holds on the same mesh exactly)."""
    pytest.importorskip("gmsh")
    r, _ = _johnson(V.unipolar_3d_resistor, 4e14)
    assert np.max(np.abs(r - 1)) < 1e-9


# =====================================================================================================
# GATE B: fluctuation-dissipation, frequency-dependent
# =====================================================================================================

def test_fdt_long_diode_current_and_open_circuit_voltage():
    """GATE B. Long Ge-like p-n diode (n_i 1e19, N 1e22, tau 1 us, 4 L each side) at ZERO bias, 1 Hz -
    10 GHz: Re Z falls 0.148 -> 2.9e-6 Ohm m^2 (diffusion admittance sqrt(1 + i w tau), then the
    depletion C shunted by the bulk R). S_I = 4kT Re Y (voltage-driven contact, Y from ssac) and the
    OPEN-CIRCUIT S_V = 4kT Re(1/Y) from a SEPARATE device whose contact is current-driven
    (drive='current', output node = the contact voltage: a different adjoint system). Measured max
    |ratio - 1| = 4.5e-6 (S_I) and 1.0e-5 (S_V) -- the ssac / DC round-off, not the noise."""
    rI, rV, ReZ = V.gate_b_fdt()
    assert ReZ.max() / ReZ.min() > 1e4                         # genuinely frequency-dependent
    assert np.max(np.abs(rI - 1)) < 1e-4
    assert np.max(np.abs(rV - 1)) < 1e-4


def test_fdt_two_regions_carrier_continuous_interface():
    """GATE B3. The FDT diode split at the junction into two regions joined by a carrier-CONTINUOUS
    interface (Potential/Electrons/Holes type='continuous': DEVSIM permutes one copy's interface rows
    into the other's). S_I = 4kT Re Y (measured <= 5.8e-6) and S equals the single-region device's
    (measured <= 7.5e-6), so both copies of the interface node carry the merged-row Green's function
    and both regions' sources are counted."""
    r2, ratio, regs = V.gate_b_two_region()
    assert regs == ["r_n", "r_p"]
    assert np.max(np.abs(r2)) < 1e-4
    assert np.max(np.abs(ratio - 1)) < 1e-4


def test_fdt_with_auger_radiative_bgn_options(monkeypatch):
    """GATE B2. The same zero-bias diode with BipolarOptions Auger + radiative (coefficients raised to
    C = 1e-38 m^6/s, B = 1e-16 m^3/s so that C n^2 ~ B n ~ 1/tau_SRH ~ 1e6 /s) and Slotboom BGN (n_ie in
    every term): S_I = 4kT Re Y holds (measured 2.3e-6 at 1 Hz, 1.5e-8 at 10 MHz), which requires the
    noise G + R of EVERY mechanism to be its own detailed-balance pair rate c (np + n_ie^2); GR carries
    54 % of S at 1 Hz. ON is load-bearing twice: the options move Re Y(1 Hz) from 6.78 to 11.41 S/m^2,
    and with the Auger/radiative pair rates DROPPED from the noise (their PTerm models filtered out
    of the noise's term inventory) the FDT breaks: S/(4kT Re Y) = 0.692 (gate: off by > 10 %)."""
    from dynameta.carriers import bipolar_options as BO
    opts = BO.BipolarOptions(auger=BO.AugerModel(C_n_m6s=1e-38, C_p_m6s=1e-38),
                             radiative=BO.RadiativeModel(B_m3s=1e-16), bgn=BO.BGN_SI_SLOTBOOM_1977)
    ff = [1.0, 1e5, 1e7, 1e9]
    r_on, sn, Y_on = V.gate_b_fdt(options=opts, open_circuit=False, freqs=ff)
    assert np.max(np.abs(r_on - 1)) < 1e-4
    assert sn.by_mechanism["gr"][0] / sn.S[0] > 0.3
    r_off, _, Y_off = V.gate_b_fdt(options=None, open_circuit=False, freqs=ff)
    assert np.max(np.abs(r_off - 1)) < 1e-4
    assert Y_on.real[0] / Y_off.real[0] > 1.5
    from dynameta.carriers import noise_analysis as NA
    real = NA._pair_term_models

    def _drop(dev, reg, shot):                                  # noise-side only (device built)
        out = real(dev, reg, shot)
        out["node"] = [m for m in out["node"] if m not in ("PTerm_auger", "PTerm_radiative")]
        return out
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, options=opts)
    monkeypatch.setattr(NA, "_pair_term_models", _drop)
    s_drop = V.noise([1.0]).S[0]
    monkeypatch.setattr(NA, "_pair_term_models", real)
    assert abs(s_drop / (4 * V.KT * Y_on.real[0]) - 1) > 0.1


# =====================================================================================================
# GATE C + E: ideal diode shot noise, and diffusion vs shot
# =====================================================================================================

@pytest.fixture(scope="module")
def shot_rows():
    if _IMPORT_ERROR is not None:
        pytest.fail("noise_analysis unavailable: {!r}".format(_IMPORT_ERROR))
    V.fresh()
    V.set_bias._v = {}
    try:
        return V.gate_c_shot()
    finally:
        V.fresh()


def test_ideal_diode_coth_law_and_full_shot_noise(shot_rows):
    """GATE C. Long Si p+n diode (N_A 1e25 2 um, N_D 1e22 5 L_p, tau 20 us), f = 1 Hz, V = 0.05..0.55 V:
    S_I = 2 q I coth(qV/2kT) (= 2q(I + 2 I_s), which is 4kT G_0 at V -> 0 and full shot 2qI at forward
    bias), with I the solved DC current. Measured |ratio - 1| <= 2.0e-4 (worst at 0.55 V, injection
    p_n/N_D = 1.7e-3) and S/2qI = 1.00000 for V >= 0.4. The coth holds through the SNS-dominated low
    bias too: the depletion GR's G/R ratio is exp(-qV/kT) at every point. Long-diode physics: the GR
    source carries 28 % of S at 0.5 V (diffusion share 0.721) -- diffusion noise alone is NOT shot."""
    rows, _ = shot_rows
    for r in rows:
        assert r["coth"] == pytest.approx(1.0, abs=3e-3), r
        if r["V"] >= 0.4:
            assert r["shot"] == pytest.approx(1.0, abs=2e-3), r
            assert 0.6 < r["diff_share"] < 0.8, r          # GR in the neutral region is load-bearing
    assert rows[0]["shot"] > 1.3                           # 0.05 V: coth(0.97) = 1.34, not shot


def test_van_der_ziel_long_diode_high_frequency(shot_rows):
    """GATE C2. van der Ziel's long-diode law S_I(f) = 4kT Re Y(f) - 2qI at 0.5 V, f = 1 Hz .. 1 MHz
    (omega tau up to 126; S_I/2qI rises 1 -> 13.4). Measured ratio 1.0092 .. 1.0012: the residual is the
    diode's own non-ideality (G_d V_t / I = 0.995 at 0.5 V), which the ideal law ignores."""
    _, hf = shot_rows
    assert hf["shot"][-1] > 10.0
    assert np.max(np.abs(hf["vdz"] - 1)) < 1.5e-2


def test_diffusion_vs_shot_discriminant(shot_rows):
    """GATE E (the diffusion-vs-shot discriminant). A BIASED uniform resistor (10 um, 0.1 V) keeps
    S = 4kT G (measured 1 + 6e-6) -- 2qI would be 1.93x LARGER (S/2qI = 0.517) -- while the forward
    diode has S = 2qI = (1/2) 4kT G_d (measured S/(4kT G_d) = 0.5043 / 0.5021 / 0.5028 at 0.45 / 0.5 /
    0.55 V). A code returning 4kT Re Y fails the diode by 2x; one returning 2qI fails the resistor."""
    rows, _ = shot_rows
    nyq, shot = V.gate_e_resistor_bias()
    assert nyq == pytest.approx(1.0, abs=1e-4)
    assert shot == pytest.approx(2 * V.KT / (V.QE * 0.1), rel=1e-3)    # = 0.517: not shot noise
    for r in rows:
        if r["V"] >= 0.45:
            assert r["nyq"] == pytest.approx(0.5, abs=1e-2), r


# =====================================================================================================
# GATE D: GR Lorentzian
# =====================================================================================================

def test_gr_lorentzian_with_contact_shorting():
    """GATE D1. Uniform n-type slab (N_D 1e22, n_i 1e21 -> p0 1e20, SRH tau 10 ns -> tau_eff 1.196 tau),
    50 ambipolar diffusion lengths, 0.05 V: the GR part of S_I equals the closed-form ambipolar Lorentzian
    (q(mu_n+mu_p)E)^2 2(G+R)/L^2 int |g|^2, g = tau~[1 - cosh(x/lam)/cosh(L/2lam)], at f/f_c = 1e-3 .. 100
    (f_c = 1/2 pi tau_eff). Measured |ratio - 1| <= 3e-5. The plain Lorentzian is 6 % high here (the
    contacts short the fluctuation within ~L_D): that is what D2 checks on a 400 L_D slab."""
    x, r_exact, _, shape = V.gate_d_gr(50.0)
    assert np.max(np.abs(r_exact - 1)) < 1e-3
    assert shape[-1] < 1e-3                                # rolled off by 1e4 at 100 f_c


def test_gr_plain_lorentzian_plateau_and_corner():
    """GATE D2. 400 L_D slab: the plain Lorentzian 2(G+R) tau^2 (q(mu_n+mu_p)E)^2 / (L(1 + w^2 tau^2)) --
    plateau within 0.75 % (measured 0.9925; the residual is the 2 L_D/L contact shorting) and the corner
    S(f_c)/S(0) = 0.5010 (Lorentzian: 0.5) -- and the exact form to 1e-5."""
    x, r_exact, r_plain, shape = V.gate_d_gr(400.0)
    assert np.max(np.abs(r_exact - 1)) < 1e-3
    assert r_plain[0] == pytest.approx(1.0, abs=1.5e-2)
    assert shape[3] == pytest.approx(0.5, abs=1e-2)
    assert shape[1] == pytest.approx(1 / (1 + 0.1 ** 2), abs=1e-2)


# =====================================================================================================
# GATE F: Hooge (opt-in)
# =====================================================================================================

def test_hooge_one_over_f():
    """GATE F. alpha_H = 2e-3, biased 10 um resistor: S_H = alpha I^2/(N f) at 1 Hz / 1 kHz / 1 MHz,
    N = n L per unit area. Measured ratio 0.9999924 (the SG edge density n_eff = n(1 + v^2/12))."""
    r = V.gate_f_hooge()
    assert np.max(np.abs(r - 1)) < 1e-4


# =====================================================================================================
# GATE G: 2-D layered builder
# =====================================================================================================

def test_fdt_layered_metasurface_2d():
    """GATE G. The 2-D LAYERED reference metasurface (LayeredDevsimBuilder: degenerate ITO with the FD
    g-factor, HfO2/Al2O3 dielectrics, 9 regions, edge-metal grounds) with the gate made circuit-driven
    by set_ssac_gate: S_I = 4kT Re Y from 1 GHz to 1 PHz (Re Y spans 7 decades). Measured
    |S/4kTReY - 1| <= 1.3e-11 at Vg = 0 (flat band: uniform n, g cancels) and <= 3.7e-5 at Vg = +2 V
    (degenerate accumulation: the FD g-factor SG edge flux is Nyquist-consistent only to O(h^2) -- the
    documented approximation of the degenerate diffusion source)."""
    r0, ReY, byr = V.gate_g_layered(0.0)
    assert np.max(np.abs(r0 - 1)) < 1e-9
    assert ReY.max() / ReY.min() > 1e5 and set(byr) == {"ito"}
    r2, _, _ = V.gate_g_layered(2.0)
    assert np.max(np.abs(r2 - 1)) < 1e-3


# =====================================================================================================
# native solve == adjoint; state and bookkeeping
# =====================================================================================================

def test_native_noise_solve_is_the_adjoint():
    """DEVSIM's solve(type='noise') is (J_static + i w J_dynamic)^T y = e_out: compared with an
    independent (equilibrated) scipy spsolve on ds.get_matrix_and_rhs at 1 GHz, interior nodes, all
    three equations. Measured max abs diff 1.6e-15 (electron field ~1), 2.2e-15 (minority-hole field)
    and 7.6e-6 on the Potential field (~6e9, i.e. 1e-15 relative).
    The circuit-contact value the module substitutes is the adjoint's circuit-node entry (= 1)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    V.bipolar_1d("adj", [(0.0, 1e-7, "lo"), (3.3e-7, 2.7e-8, None), (1e-6, 1e-7, "hi")], "1e22",
                 n_i=1e16, tau=1e-7)
    f = 1e9
    with V.quiet():
        devsim.solve(type="noise", frequency=f, output_node="V1.I")
    mr = devsim.get_matrix_and_rhs(format="csc")
    n = len(mr["static"]["ap"]) - 1

    def mk(d):
        return sp.csc_matrix((np.asarray(d["av"]), np.asarray(d["ai"]), np.asarray(d["ap"])),
                             shape=(n, n))
    A = (mk(mr["static"]) + 1j * 2 * math.pi * f * mk(mr["dynamic"])).T.tocsc()
    e = np.zeros(n, complex)
    e[devsim.get_circuit_equation_number(node="V1.I")] = 1.0
    # row + column max-norm equilibration: WITHOUT it scipy's spsolve loses the minority-hole rows
    # (p = 1e10 next to n = 1e22) -- measured 2.1e-4 off on the hole field, while the equilibrated
    # solve and DEVSIM (UMFPACK scales internally) agree to 2.2e-15.
    r = 1.0 / abs(A).max(axis=1).toarray().ravel()
    As = sp.diags(r) @ A
    c = 1.0 / abs(As).max(axis=0).toarray().ravel()
    y = c * spla.spsolve((As @ sp.diags(c)).tocsc(), r * e)
    for eq in ("ElectronContinuityEquation", "HoleContinuityEquation", "PotentialEquation"):
        rows = np.asarray(devsim.get_equation_numbers(device="adj", region="bulk", equation=eq))
        nat = (np.asarray(devsim.get_node_model_values(device="adj", region="bulk",
                                                       name="V1.I_{}_real".format(eq)))
               + 1j * np.asarray(devsim.get_node_model_values(device="adj", region="bulk",
                                                              name="V1.I_{}_imag".format(eq))))
        scale = max(1.0, float(np.max(np.abs(nat))))
        assert np.max(np.abs(y[rows][1:-1] - nat[1:-1])) / scale < 1e-12, eq
        assert nat[0] == 0 and nat[-1] == 0                  # DEVSIM zeroes contact nodes
    assert abs(y[devsim.get_circuit_equation_number(node="vhi")] - 1.0) < 1e-12


def test_dc_state_and_models_untouched_and_maps_sum():
    """noise_spectrum only READS: the DC solution (Potential/Electrons/Holes) is bit-identical after it,
    no node/edge model is left behind, and the per-node maps / per-region / per-mechanism splits each
    sum to S."""
    from dynameta.carriers import noise_analysis as NA
    V.bipolar_1d("st", [(0.0, 1e-8, "lo"), (5e-7, 1e-9, None), (1e-6, 1e-8, "hi")],
                 "ifelse(x < 5e-7, -1e22, 1e22)", n_i=1e16, tau=1e-7)
    snap = {nm: list(devsim.get_node_model_values(device="st", region="bulk", name=nm))
            for nm in ("Potential", "Electrons", "Holes")}
    nodes0 = set(devsim.get_node_model_list(device="st", region="bulk"))
    edges0 = set(devsim.get_edge_model_list(device="st", region="bulk"))
    with V.quiet():
        sn = NA.noise_spectrum("st", [1e3, 1e9], maps=True)
    for nm, vals in snap.items():
        assert list(devsim.get_node_model_values(device="st", region="bulk", name=nm)) == vals
    assert set(devsim.get_node_model_list(device="st", region="bulk")) == nodes0
    assert set(devsim.get_edge_model_list(device="st", region="bulk")) == edges0
    assert sn.kind == "current" and set(sn.by_mechanism) == {"diffusion_n", "diffusion_p", "gr"}
    tot = sum(sn.by_mechanism.values())
    np.testing.assert_allclose(tot, sn.S, rtol=1e-14)
    np.testing.assert_allclose(sn.by_region["bulk"], sn.S, rtol=1e-14)
    mapsum = sum(sn.maps["bulk"][m].sum(axis=1) for m in sn.by_mechanism)
    np.testing.assert_allclose(mapsum, sn.S, rtol=1e-12)
    assert sn.temperature_K["bulk"] == pytest.approx(300.0, rel=1e-12)


def test_terminal_noise_s_v_is_norton_thevenin():
    """terminal_noise: S_V = S_I / |Y|^2 with Y = G + i w C from ssac at the same point."""
    from dynameta.carriers import noise_analysis as NA
    V.bipolar_1d("tn", [(0.0, 1e-8, "lo"), (5e-7, 1e-9, None), (1e-6, 1e-8, "hi")],
                 "ifelse(x < 5e-7, -1e22, 1e22)", n_i=1e16, tau=1e-7)
    with V.quiet():
        tn = NA.terminal_noise("tn", [1e6, 1e10])
    np.testing.assert_allclose(tn.S_V, tn.S_I / np.abs(tn.Y) ** 2, rtol=1e-14)
    np.testing.assert_allclose(tn.S_I, tn.current.S, rtol=0)


def test_refusals():
    """Bad inputs raise instead of returning a number."""
    from dynameta.carriers import noise_analysis as NA
    from dynameta.carriers import physics_bipolar_dd as BP
    V.bipolar_1d("rf", [(0.0, 1e-7, "lo"), (1e-6, 1e-7, "hi")], "1e22", n_i=1e16, tau=1e-7)
    with pytest.raises(ValueError, match="frequencies"):
        NA.noise_spectrum("rf", [0.0])
    with pytest.raises(ValueError, match="mechanism"):
        NA.noise_spectrum("rf", [1.0], mechanisms=("shot",))
    with pytest.raises(ValueError, match="circuit node"):
        NA.noise_spectrum("rf", [1.0], output_node="nope")
    with pytest.raises(ValueError, match="hooge_alpha"):
        NA.noise_spectrum("rf", [1.0], hooge_alpha=-1.0)
    with pytest.raises(ValueError, match="unknown region"):
        NA.noise_spectrum("rf", [1.0], regions=["nowhere"])
    with pytest.raises(ValueError, match="drive"):
        BP.setup_contact_ohmic_bipolar_circuit("rf", "lo", drive="charge")
    with pytest.raises(ValueError, match="starting with 'I'"):
        BP.setup_contact_ohmic_bipolar_circuit("rf", "lo", drive="current", source_name="V9")


def test_bernoulli_kernel():
    """B(x) = x/(e^x - 1): B(0) = 1, B(-x) = B(x) + x, and the series branch is continuous."""
    from dynameta.carriers.noise_analysis import bernoulli
    x = np.array([-30.0, -1.0, -1e-9, 0.0, 1e-9, 1e-6, 1.0, 30.0])
    b = bernoulli(x)
    assert b[3] == 1.0
    np.testing.assert_allclose(bernoulli(-x), b + x, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(b[[0, 1, 6, 7]], x[[0, 1, 6, 7]] / np.expm1(x[[0, 1, 6, 7]]),
                               rtol=1e-15)
