"""Trap generation-recombination noise of KINETIC traps (DynaMeta v0.15, plan item 4.3.2 step K3; P3 section 1.5
K3 rows). Each kinetic level / interface bin is two Poissonian processes, electron exchange (capture +
emission, event rate P_n = [n (1-f) + n1' f]/tau_n0) and hole exchange (P_p = [p f + p1' (1-f)]/tau_p0),
each a two-point event between its carrier row and the trap row (noise_analysis single-carrier sources):

    S = q^2 V [2 P_n |y_t - y_n|^2 + 2 P_p |y_p - y_t|^2]          (mechanisms "trap_n", "trap_p")

ORACLES (independent of the code under test):
  * the FLUCTUATION-DISSIPATION THEOREM S_I = 4kT Re Y on equilibrium devices (Re Y from DEVSIM's ssac, a
    different solve from the noise sum), 1e-4 as the D2a gate B -- for a CHARGED bulk level on the gate-B
    long diode (the plan's xfail flips) and for kinetic interface bins on the conductance-method MOS;
  * the closed-form ambipolar GR Lorentzian (gate D1) for a nearly-empty-of-charge neutral kinetic level;
  * an equilibrated scipy adjoint of ds.get_matrix_and_rhs for DEVSIM's native field on the trap rows.
Load-bearing perturbations: drop the hole process, flip the trap-row sign, double-count a fast-trap pair,
count the contact-node events, each must break the FDT.
"""
import math

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
V = pytest.importorskip("validation.devsim_noise")

from dynameta.carriers import noise_analysis as NA            # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP        # noqa: E402
from dynameta.carriers import physics_traps as PT             # noqa: E402
from dynameta.warning_types import AdvisoryWarning            # noqa: E402

ds = V.ds
KT = 1.380649e-23 * 300.0                                     # re-typed (CODATA 2018)
FREQS = list(np.logspace(0, 9, 10))                           # 1 Hz .. 1 GHz


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()


def _charged(dynamic=True, **kw):
    """test_devsim_phase_composition's CHARGED_TRAP (an acceptor, N_t = 20 % of the gate-B doping)."""
    base = dict(energy_eV=0.15, density_m3=2e21, kind="acceptor", tau_n0_s=3e-7, tau_p0_s=5e-7)
    base.update(kw)
    return PT.TrapLevel(dynamic=dynamic, **base)


def _fdt_diode(traps, freqs=FREQS, mechanisms=None):
    """The gate-B long diode (validation/devsim_noise.py) with `traps`: S_I / 4kT Re Y - 1 per frequency,
    the spectrum and Y."""
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, traps=traps,
                 srh_trap_energy_eV=-0.1)
    kw = {} if mechanisms is None else {"mechanisms": mechanisms}
    sn = V.noise(freqs, **kw)
    Y = V.ssac_Y(freqs)
    return sn.S / (4 * KT * Y.real) - 1.0, sn, Y


# =====================================================================================================
# K3-1 / K3-2: the FDT with a charged kinetic level, and what breaks it
# =====================================================================================================

def test_fdt_with_a_charged_kinetic_level_all_frequencies():
    """K3-1 (the plan's xfail flips). The gate-B diode with the CHARGED acceptor level made KINETIC: S_I =
    4kT Re Y from 1 Hz to 1 GHz to 1e-4 (the D2a gate-B tolerance; ssac-limited). MEASURED max 1.6e-6.
    The quasi-static level misses by 9.9e-2 at 1 MHz (and raises the decision-39 advisory); the kinetic
    sources are a real share of S (trap_n + trap_p = 35 % at 100 kHz). The trap rows sit at the circuit
    contact's node (the trap row is kept there): the refusal is restricted to equations whose contact row
    is replaced (test_contact_row_refusal_both_sides)."""
    r, sn, _ = _fdt_diode((_charged(),))
    assert np.max(np.abs(r)) < 1e-4, r
    share = (sn.by_mechanism["trap_n"] + sn.by_mechanism["trap_p"]) / sn.S
    assert share[5] > 0.2
    with pytest.warns(AdvisoryWarning, match="QUASI-STATIC CHARGED traps"):
        rq, _, _ = _fdt_diode((_charged(dynamic=False),), freqs=[1e6])
    assert abs(rq[0]) > 1e-2


def _perturbed(monkeypatch, how):
    """Re-run K3-1 with ONE deliberate change to the trap sources (noise side only, the device is the same)."""
    real = NA._single_carrier_sources

    def drop_hole(device, region, eqs, mechanisms):
        return [s for s in real(device, region, eqs, mechanisms) if s.mechanism != "trap_p"]

    def flip_trap_row(device, region, eqs, mechanisms):
        out = []
        for s in real(device, region, eqs, mechanisms):
            rows = tuple(NA.SourceRow(r.region, r.equation, -r.coeff if r.equation.startswith("TrapEquation")
                                      else r.coeff, r.nodes) for r in s.rows)
            out.append(NA.TwoPointSource(s.mechanism, s.region, s.rate, rows, s.map_nodes))
        return out

    if how in ("drop_hole", "flip"):
        monkeypatch.setattr(NA, "_single_carrier_sources", drop_hole if how == "drop_hole" else flip_trap_row)
        r, _, _ = _fdt_diode((_charged(),), freqs=[1.0, 1e5, 1e7])
        return r
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, traps=(_charged(),),
                 srh_trap_energy_eV=-0.1)
    if how == "double":            # the fast-trap pair of the SAME level next to its kinetic events
        args = ("trap0_n1", "trap0_p1", "trap0_taun", "trap0_taup")
        BP.add_recombination_term("fdt", "bulk", "dbl", "0", pair=PT.srh_rate_expr(*args, pair=True),
                                  derivs={"Electrons": "0", "Holes": "0"})
    sn = V.noise([1.0, 1e5, 1e7])
    Y = V.ssac_Y([1.0, 1e5, 1e7])
    return sn.S / (4 * KT * Y.real) - 1.0


@pytest.mark.parametrize("how", ["drop_hole", "flip", "double"])
def test_fdt_load_bearing(monkeypatch, how):
    """K3-2. Each perturbation must break the K3-1 FDT by > 1e-2 at some frequency (the plan's figure):
    (a) drop the hole-exchange source; (b) flip the trap-row coefficient (a wrong relative sign); (c) ALSO
    install the fast-trap pair source of the same level (double counting). MEASURED at 1 Hz / 100 kHz /
    10 MHz: (a) -0.19 / -0.18 / -2.1e-4, (b) +26 / +25 / +7.8, (c) +0.33 / +0.30 / +6.2e-4."""
    r = _perturbed(monkeypatch, how)
    assert np.max(np.abs(r)) > 1e-2, (how, r)


def test_fdt_breaks_with_the_contact_node_events(monkeypatch):
    """The contact-node rule (noise_analysis._single_carrier_sources, v0.15 K3): a kinetic level's events at
    an ohmic contact node are invisible to the DC / ssac system and carry no source. Counting them (the
    4.0.2 behaviour) misses the FDT by the two contact nodes' share: MEASURED +2.3e-3 at 1 Hz -- 23x the
    K3-1 tolerance (the per-node map at the 'hi' node: trap_n 8.2e-4 + trap_p 1.5e-3 of S)."""
    real = NA._single_carrier_sources

    def keep_contacts(device, region, eqs, mechanisms):
        saved = ds.get_contact_list
        try:
            ds.get_contact_list = lambda **kw: ()           # the contact-node zeroing sees no contact
            return real(device, region, eqs, mechanisms)
        finally:
            ds.get_contact_list = saved
    monkeypatch.setattr(NA, "_single_carrier_sources", keep_contacts)
    r, _, _ = _fdt_diode((_charged(),), freqs=[1.0, 1e5])
    assert np.max(np.abs(r)) > 10 * 1e-4, r


# =====================================================================================================
# the circuit-contact-node refusal, both sides
# =====================================================================================================

def test_contact_row_refusal_both_sides():
    """noise_spectrum refuses an event row on a circuit-contact node ONLY for an equation whose row that
    contact replaces (it carries a contact equation of that name) and that has no fix-up. (i) A kinetic
    trap row at the circuit contact's node -- kept by DEVSIM, no contact equation -- is accepted (K3-1 runs
    on it; asserted here by the inventory and a solve). (ii) A single-carrier term whose partner row is
    PotentialEquation (replaced by the circuit contact, no fix-up) is still refused, naming the row."""
    V.bipolar_1d("ref", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, traps=(_charged(),),
                 srh_trap_energy_eV=-0.1)
    hi = NA._contact_nodes("ref", "bulk", "hi")
    assert hi.size and "TrapEquation_trap0" not in ds.get_contact_equation_list(device="ref", contact="hi")
    sn = V.noise([1e3])
    assert sn.by_mechanism["trap_n"][0] > 0 and sn.by_mechanism["trap_p"][0] > 0
    BP.add_recombination_term("ref", "bulk", "badpartner", "0", carrier="electron", pair="1e10",
                              derivs={"Electrons": "0", "Holes": "0"},
                              partner=("PotentialEquation", 1.0), mechanism="trap_n")
    with pytest.raises(NotImplementedError, match="PotentialEquation.*replaces"):
        V.noise([1e3])


# =====================================================================================================
# K3-5: the conductance-method MOS (kinetic interface bins) obeys the FDT
# =====================================================================================================
@pytest.mark.parametrize("which", ["single", "continuum"])
def test_mos_fdt_with_kinetic_interface_bins(which):
    """K3-5. The K2 MOS (tests/test_trap_kinetics.py: p-Si 1e17 cm^-3 / 10 nm SiO2, circuit gate, V_G =
    0.45 V, an insulating gate: the device is in EQUILIBRIUM) with a kinetic single level or the uniform
    1e12 cm^-2 eV^-1 continuum (44 kinetic bins): S_I(gate) = 4kT Re Y_m at omega tau = 1e-2 .. 1e2 (tau of
    the single level), 1e-4 (D2a). MEASURED <= 4.2e-13 (single) / 2.8e-13 (continuum); the trap-free MOS
    1.8e-13; the trap sources carry 100 % of S at the peak (trap_p; the minority trap_n 1e-10). The
    QUASI-STATIC level misses by +12.7 at every frequency: its S keeps the trap-free diffusion noise while
    its Re Y is only the series-resistance loss (asserted > 1, with the advisory)."""
    import test_trap_kinetics as TK
    from dynameta.carriers import ac_analysis as AC
    spec = (TK._it_single() if which == "single" else
            PT.InterfaceTraps(dit_eV_m2=TK.DIT, band_gap_eV=TK.EG, sigma_n_m2=TK.IT_SN, sigma_p_m2=TK.IT_SP,
                              v_th_n_ms=TK.IT_VTH, v_th_p_ms=TK.IT_VTH, dynamic=True))

    def fdt(it, warn=False):
        V.fresh()
        psib = TK._mos_ac("k35", it)
        TK._mos_vg(psib, 0.0, TK.MOS_VG)
        _psi, ps, ns = TK._mos_surface("k35")
        tau = TK._ng_single_level(ps, ns)[1]
        ff = np.array([1e-2, 0.1, 1.0, 10.0, 100.0]) / tau / (2 * math.pi)
        with pytest.warns(AdvisoryWarning) if warn else _nothing():
            with V.quiet():
                sn = NA.noise_spectrum("k35", ff, output_node="V1.I")
                _f, _C, G = AC.ssac_admittance(ff)
        return sn.S / (4 * KT * G) - 1.0, sn
    r, sn = fdt(spec)
    assert np.max(np.abs(r)) < 1e-4, r
    assert sn.by_mechanism["trap_p"][2] / sn.S[2] > 0.5
    if which == "single":
        qs = PT.InterfaceTraps(distribution="discrete", levels=((TK.IT_E, TK.IT_N),), charge="acceptor",
                               sigma_n_m2=TK.IT_SN, sigma_p_m2=TK.IT_SP, v_th_n_ms=TK.IT_VTH,
                               v_th_p_ms=TK.IT_VTH)
        rq, _ = fdt(qs, warn=True)
        assert np.min(np.abs(rq)) > 1.0, rq


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# =====================================================================================================
# K3-6: DEVSIM's native adjoint on the trap rows
# =====================================================================================================
def test_native_noise_field_on_trap_rows_is_the_adjoint():
    """K3-6. The native noise solve's field on the kinetic trap rows (<out>_TrapEquation_trap0_real/imag) and
    on the electron / hole rows equals an independent, row+column-equilibrated scipy solve of (J_static +
    i w J_dynamic)^T y = e_out built from ds.get_matrix_and_rhs (the tests/test_devsim_noise.py recipe) at
    1 MHz on the gate-B diode with the charged kinetic level, EVERY node of the trap row (contact nodes
    included: DEVSIM keeps the trap row there -- unlike the continuity rows, whose contact entries it
    zeroes). TOLERANCE: both are backward-stable solves of the same equilibrated system, so they agree to
    kappa_1(A_eq) eps of each field's scale (kappa_1 by scipy's onenormest; the D2a gate's fixed 1e-12 is
    not derivable here: this junction's equilibrated kappa_1 is 2e11 -> 4.4e-5). MEASURED 2.9e-10 (trap row),
    2.2e-11 / 3.3e-11 (electron / hole)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    V.bipolar_1d("adj", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, traps=(_charged(),),
                 srh_trap_energy_eV=-0.1)
    f = 1e6
    with V.quiet():
        devsim.solve(type="noise", frequency=f, output_node="V1.I")
    mr = devsim.get_matrix_and_rhs(format="csc")
    n = len(mr["static"]["ap"]) - 1

    def mk(d):
        return sp.csc_matrix((np.asarray(d["av"]), np.asarray(d["ai"]), np.asarray(d["ap"])), shape=(n, n))
    A = (mk(mr["static"]) + 1j * 2 * math.pi * f * mk(mr["dynamic"])).T.tocsc()
    e = np.zeros(n, complex)
    e[devsim.get_circuit_equation_number(node="V1.I")] = 1.0
    r = 1.0 / abs(A).max(axis=1).toarray().ravel()
    As = sp.diags(r) @ A
    c = 1.0 / abs(As).max(axis=0).toarray().ravel()
    Ae = (As @ sp.diags(c)).tocsc()
    y = c * spla.spsolve(Ae, r * e)
    lu = spla.splu(Ae)
    inv = spla.LinearOperator(Ae.shape, matvec=lu.solve, rmatvec=lambda x: lu.solve(x, trans="H"),
                              dtype=complex)
    tol = spla.onenormest(Ae) * spla.onenormest(inv) * np.finfo(float).eps
    assert tol < 1e-3                                           # a meaningful bound
    for eq in ("TrapEquation_trap0", "ElectronContinuityEquation", "HoleContinuityEquation"):
        rows = np.asarray(devsim.get_equation_numbers(device="adj", region="bulk", equation=eq))
        nat = (np.asarray(devsim.get_node_model_values(device="adj", region="bulk", name="V1.I_{}_real".format(eq)))
               + 1j * np.asarray(devsim.get_node_model_values(device="adj", region="bulk",
                                                              name="V1.I_{}_imag".format(eq))))
        scale = max(float(np.max(np.abs(nat))), 1e-300)
        sel = slice(None) if eq.startswith("Trap") else slice(1, -1)
        assert np.max(np.abs(y[rows][sel] - nat[sel])) / scale < tol, eq
        if eq.startswith("Trap"):
            assert abs(nat[-1]) / scale < tol and abs(nat[0]) / scale < tol   # decoupled at the contact nodes


# =====================================================================================================
# K3-3: a nearly charge-free neutral kinetic level IS the fast-trap GR (gate D1's Lorentzian)
# =====================================================================================================
def test_neutral_kinetic_level_reproduces_the_ambipolar_gr_lorentzian():
    """K3-3 (~1 s). Gate D1's slab (validation/devsim_noise.py: n-type 1e22, n_i 1e21 -> p0 1e20,
    50 ambipolar diffusion lengths, 0.05 V) with the built-in SRH channel switched off (1e3 s) and the SAME
    midgap level (tau_n0 = tau_p0 = 10 ns) as a NEUTRAL KINETIC TrapLevel of N_t = 1e13 m^-3: its g-r part
    (trap_n + trap_p) must equal the closed-form ambipolar Lorentzian with contact shorting (gr_analytic --
    the D1 oracle) at f/f_c = 1e-3 .. 100. Budget: gate D1's 1e-3 tolerance (D1 MEASURED 3e-5) plus the
    kinetic corrections, which vanish as N_t -> 0: the stored charge O(N_t/p0) = 1e-7, and FINDING the
    level's own exchange time tau_t = N_t tau_n0/(n0 + n1 + p0 + p1): the trap row's adjoint lags by
    omega tau_t while |y_p - y_n| rolls off, an excess growing as (N_t omega)^2 -- MEASURED +0.223 / 2.2e-3
    / < 1e-5 at 100 f_c for N_t = 1e15 / 1e14 / 1e13 (exactly N_t^2), +1.8e-3 at 30 f_c for 1e15. Extrapolated
    to 1e13 at 100 f_c: 2.2e-5. Gated at 1e-3 + 10 N_t/p0. MEASURED (1e13): |gr/oracle - 1| <= 1.6e-5."""
    n0, p0, Da, tau, GR2 = V.gr_constants()
    LD = math.sqrt(Da * tau)
    L = 50.0 * LD
    nt = 1e13
    lv = PT.TrapLevel(energy_eV=0.0, density_m3=nt, kind="neutral", tau_n0_s=V.GR_TAU, tau_p0_s=V.GR_TAU,
                      dynamic=True)
    V.bipolar_1d("grk", [(0.0, LD / 200, "lo"), (L / 2, LD / 10, None), (L, LD / 200, "hi")],
                 "{:.10e}".format(V.GR_ND), n_i=V.GR_NI, tau=1e3, presolve=False, ae=1e8, traps=(lv,))
    V.set_bias._v = {}
    V.set_bias(V.GR_V, 1e8)
    fc = 1.0 / (2 * math.pi * tau)
    ff = fc * np.array([1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
    sn = V.noise(ff)
    gr = sn.by_mechanism["trap_n"] + sn.by_mechanism["trap_p"]
    exact, _plain = V.gr_analytic(ff, L, V.GR_V / L)
    assert np.max(np.abs(gr / exact - 1.0)) < 1e-3 + 10.0 * nt / p0, gr / exact - 1.0


# =====================================================================================================
# K3-4: the single-level trapping Lorentzian of a biased resistor
# =====================================================================================================
def test_trapping_noise_lorentzian_of_a_resistor():
    """K3-4. A 20 um n-type resistor (N_D 1e22, n_i 1e16: holes absent), an ACCEPTOR kinetic level near E_F
    (E_t - E_i = 0.36 eV, N_t 1e21, c_n = c_p = 5e-17 m^3/s), bulk SRH off, 0.05 V. The test DERIVES the
    oracle (the plan's: Burgess g-r theorem with the number constraint; van Vliet & Fassett not accessed):
    neutrality gives dn = -N_t df, so 1/tau = c_n (n0 + n1 + N_t (1 - f0)), the electron-number variance per
    volume is [1/n0 + 1/(N_t f0 (1 - f0))]^-1, and S_I = 4 I^2 var tau/(n0^2 L (1 + omega^2 tau^2)) (n0, f0
    at the centre from the DC state). GATES: (a) trap_n + trap_p against it at omega tau = 0.01 .. 10, within
    the contact shorting -- the trapped-charge fluctuation is screened by the contacts over ~ the Debye length
    L_D' = sqrt(eps V_t/(q (n0 + N_t f0 (1-f0)))) at each end: budget 4 L_D'/L (2 L_D' per end) = 8.2e-3;
    (b) the shape S(f_c)/S(0) = 1/2 and S/S(0) = 1/(1 + omega^2 tau^2) to 1.5e-2 (the D2 analogue). MEASURED
    -6.2e-3 (flat in omega: the shorting is frequency-independent; it scales as 1/L -- -2.5e-2 / -6.2e-3 /
    -1.6e-3 at L = 5 / 20 / 80 um, a shorting length 6.2e-8 m = 1.5 L_D' per end); the corner 0.50003.
    OLD CODE (quasi-static level): the trapping corner does not exist -- its 'gr' spectrum is flat through
    f_c (S(f_c)/S(0) > 0.9, asserted, with the advisory)."""
    nd, ni, nt, et, cn = 1e22, 1e16, 1e21, 0.36, 5e-17
    L = 20e-6

    def build(dyn):
        lv = PT.TrapLevel(energy_eV=et, density_m3=nt, kind="acceptor", tau_n0_s=1 / (cn * nt),
                          tau_p0_s=1 / (cn * nt), dynamic=dyn)
        V.bipolar_1d("gk", [(0.0, 5e-9, "lo"), (L / 2, L / 200, None), (L, 5e-9, "hi")], "%.6e" % nd, n_i=ni,
                     tau=1e3, presolve=False, ae=1e8, traps=(lv,))
        V.set_bias._v = {}
        V.set_bias(0.05, 1e8)

    build(True)
    x = np.asarray(ds.get_node_model_values(device="gk", region="bulk", name="x"))
    ic = int(np.argmin(np.abs(x - L / 2)))
    n0 = float(ds.get_node_model_values(device="gk", region="bulk", name="Electrons")[ic])
    f0 = float(ds.get_node_model_values(device="gk", region="bulk", name="TrapOccupancy0")[ic])
    vt = KT / 1.602176634e-19
    n1 = ni * math.exp(et / vt)
    tau = 1.0 / (cn * (n0 + n1 + nt * (1.0 - f0)))
    var = 1.0 / (1.0 / n0 + 1.0 / (nt * f0 * (1.0 - f0)))
    current = V.source_current()
    wt = np.array([1e-2, 0.3, 1.0, 3.0, 10.0])
    ff = wt / tau / (2 * math.pi)
    sn = V.noise(ff)
    gr = sn.by_mechanism["trap_n"] + sn.by_mechanism["trap_p"]
    ora = 4.0 * current ** 2 * var * tau / (n0 ** 2 * L * (1.0 + wt ** 2))
    q = 1.602176634e-19
    ld = math.sqrt(11.7 * 8.8541878128e-12 * vt / (q * (n0 + nt * f0 * (1.0 - f0))))
    assert np.all(np.abs(gr / ora - 1.0) <= 4.0 * ld / L), (gr / ora - 1.0, 4 * ld / L)
    assert gr[2] / gr[0] == pytest.approx(0.5 * (1 + 1e-4), abs=1.5e-2)
    assert np.all(np.abs(gr / gr[0] * (1.0 + wt ** 2) / (1.0 + wt[0] ** 2) - 1.0) < 1.5e-2)
    V.fresh()
    with pytest.warns(AdvisoryWarning):
        build(False)
        sq = V.noise(ff[[0, 2]]).by_mechanism["gr"]
    assert sq[1] / sq[0] > 0.9, sq


# =====================================================================================================
# the one-call method: LayeredDevsimBuilder.noise_spectrum / terminal_noise with kinetic traps
# =====================================================================================================
def test_layered_builder_noise_with_kinetic_traps():
    """LayeredDevsimBuilder.noise_spectrum / terminal_noise (plan 4.2.5) on a gated bipolar n-GaAs builder with a
    kinetic bulk acceptor (TransportModel.srh_traps) AND kinetic interface bins (interface_traps=), run in a
    child process (DEVSIM circuits are process-global: tests/_trap_noise_child.py): the builder seeds and
    DC-solves the kinetic levels (solve_dc's DC form), noise_analysis.DEFAULT_MECHANISMS includes trap_n /
    trap_p and the method inherits them (by_mechanism), and the FDT holds through the method per cell, S_I =
    4kT Re Y to 1e-4 at 1 MHz / 100 MHz / 10 GHz (the D2a tolerance; the reference-device gate W2 reaches
    1e-9 without traps). MEASURED: audit note. The trap sources dominate S here (> 99.9 %)."""
    import json
    import os
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([repo] + [p for p in sys.path if isinstance(p, str) and p]))
    r = subprocess.run([sys.executable, "-W", "error", os.path.join(repo, "tests", "_trap_noise_child.py"),
                        "kinetic_builder"], cwd=repo, env=env, capture_output=True, text=True, timeout=900)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("TRAP_NOISE_RESULT ")]
    assert r.returncode == 0 and lines, (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    res = json.loads(lines[-1][len("TRAP_NOISE_RESULT "):])
    assert "TrapW_trap0" in res["kinetic_vars"] and sum(v.startswith("ItW") for v in res["kinetic_vars"]) == 8
    assert res["default_has_trap"] and {"trap_n", "trap_p"} <= set(res["mechanisms"])
    assert min(res["trap_share"]) > 0.5
    fdt = np.array(res["S_I"]) / (4.0 * KT * np.array(res["ReY"])) - 1.0
    assert np.max(np.abs(fdt)) < 1e-4, fdt
