"""v0.14 DEVSIM-phase integration gates: the D1b / D1c / D1d / D2a physics COMPOSE.

After the integration (docs/audit/2026-11-03-devsim-phase-integration.md) every opt-in generation /
recombination / charge term of a bipolar region enters through ONE additive mechanism
(physics_bipolar_dd.add_recombination_term / add_charge_term), every SRH channel through ONE kernel
(physics_traps.srh_rate_expr, with the bandgap-narrowing substitution), and the noise GR source reads
the same term inventory. These gates turn on bandgap narrowing + Auger + radiative + a NON-midgap
charged trap level + BTBT TOGETHER and check:

  (1) structure: every term appears EXACTLY ONCE in the assembled composites, the composites are the
      ones the continuity / Poisson equations reference, and a later term SUMS (never overwrites);
  (2) numbers: on a solved reverse-biased junction, the assembled node generation equals the sum of
      the individually computed terms re-evaluated in numpy from the SOLVED n, p, n_ie (independent
      closed forms, constants from scipy), and so does the edge-assembled BTBT; the integrated total
      equals the sum of the per-term integrals; the trapped charge in Poisson uses the n_ie-scaled
      occupancy;
  (3) contacts: with a CHARGED trap the ohmic contacts pin the trap-aware charge-neutral root
      (scipy brentq on a hand-written neutrality residual);
  (4) noise: S_I = 4kT Re Y (fluctuation-dissipation) still holds on the zero-bias long diode with a
      non-midgap TrapLevel and with BTBT on (D = 0 at equilibrium, but its pair rate and its linear
      response are not), each load-bearing; and a term without a pair rate makes the noise REFUSE.
  (5) 2-D element assembly: the element-assembled BTBT pair rate reaches each node with the residual's
      own ElementNodeVolume weights (uniform state: P_i = pair * NodeVolume_i).
"""
import math

import numpy as np
import pytest

ds = pytest.importorskip("devsim")
sc = pytest.importorskip("scipy.constants")
from scipy.optimize import brentq                             # noqa: E402

from dynameta.carriers import eq_registry as _R               # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP        # noqa: E402
from dynameta.carriers import bipolar_options as BO           # noqa: E402
from dynameta.carriers import physics_traps as PT             # noqa: E402
from dynameta.carriers.tunneling import BTBTParams, SILICON_HURKX   # noqa: E402

KT_Q = sc.k * 300.0 / sc.e            # oracle thermal voltage (scipy constants, T_REF = 300 K)
NI = 1.0e16
ND = 5.0e24
L, XJ = 200e-9, 100e-9
TRAP = PT.TrapLevel(energy_eV=0.2, density_m3=1.0e21, kind="acceptor", tau_n0_s=2e-7, tau_p0_s=1e-7)
OPTS = BO.BipolarOptions(auger=BO.AUGER_SI_DZIEWIOR_SCHMID_1977,
                         radiative=BO.RADIATIVE_SI_TRUPKE_2003, bgn=BO.BGN_SI_SLOTBOOM_1977)
TAU = 1e-6


class _Log:
    """Records the LAST definition of every node/edge model (name -> equation string)."""

    def __init__(self):
        self.node, self.edge, self.count = {}, {}, {}

    def __enter__(self):
        self._saved = (ds.node_model, ds.edge_model)
        n0, e0 = self._saved

        def nm(**kw):
            self.node[kw["name"]] = kw["equation"]
            self.count[kw["name"]] = self.count.get(kw["name"], 0) + 1
            return n0(**kw)

        def em(**kw):
            self.edge[kw["name"]] = kw["equation"]
            return e0(**kw)
        ds.node_model, ds.edge_model = nm, em
        return self

    def __exit__(self, *a):
        ds.node_model, ds.edge_model = self._saved


def _wipe():
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    for dv in ("cmp", "cmp2d"):
        _R.clear(dv)


def _solve(ae=1e12, re=1e-10):
    ds.solve(type="dc", absolute_error=ae, relative_error=re, maximum_iterations=100)


def _v(dev, name, region="b"):
    return np.asarray(ds.get_node_model_values(device=dev, region=region, name=name))


def _e(dev, name, region="b"):
    return np.asarray(ds.get_edge_model_values(device=dev, region=region, name=name))


def _build(dev="cmp", *, options=OPTS, traps=(TRAP,), btbt=SILICON_HURKX, log=None):
    _wipe()
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="pc")
    ds.add_1d_mesh_line(mesh=mesh, pos=XJ, ps=0.2e-9)
    ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=2e-9, tag="nc")
    ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
    ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="b", name="NetDoping",
                  equation="ifelse(x < {}, {}, {})".format(XJ, -ND, ND))
    kw = dict(eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=NI, mobility_n_m2Vs=0.135,
              mobility_p_m2Vs=0.048, tau_n_s=TAU, tau_p_s=TAU, fd_enhancement=False,
              options=options, traps=traps, btbt=btbt)
    if log is not None:
        with log:
            BP.setup_bipolar_region(dev, "b", **kw)
    else:
        BP.setup_bipolar_region(dev, "b", **kw)
    BP.setup_equilibrium_seed_models(dev, "b")
    for c in ("pc", "nc"):
        BP.setup_contact_ohmic_bipolar(dev, c)
    ds.node_model(device=dev, region="b", name="_seed",
                  equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "b")))
    for var, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"),
                   ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region="b", name=var, values=_v(dev, m).tolist())
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(dev, ceq)
    _solve(1e10, 1e-8)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(dev, ceq)
    _solve()
    return dev


@pytest.fixture(scope="module")
def reverse():
    """The all-on junction, recorded at setup, solved at equilibrium and at 0.5 V reverse."""
    log = _Log()
    dev = _build(log=log)
    eq = {k: _v(dev, k) for k in ("Electrons", "Holes", "Potential")}
    for vb in (0.1, 0.2, 0.3, 0.4, 0.5):
        ds.set_parameter(device=dev, name="nc_bias", value=vb)
        _solve()
    yield dev, log, eq
    _wipe()


def test_each_term_exactly_once(reverse):
    dev, log, _ = reverse
    gen = log.node["ElectronGeneration"]
    for t in ("USRH", "RTerm_auger", "RTerm_radiative", "RTerm_trap0"):
        assert gen.count(t) == 1, (t, gen)
    assert gen.count("RTerm_") == 3 and "UAuger" not in gen and "URad" not in gen
    assert log.node["HoleGeneration"] == gen.replace("-ElectronCharge", "+ElectronCharge")
    # the ONE SRH kernel with the BGN substitution, for the base channel AND the trap level
    assert log.node["USRH"] == PT.srh_rate_expr(bgn=True)
    assert log.node["RTerm_trap0"] == PT.srh_rate_expr("trap0_n1", "trap0_p1", "trap0_taun",
                                                       "trap0_taup", bgn=True)
    assert log.count["USRH"] == 1                              # defined once, never re-defined by name
    # Poisson: the trapped charge once, on NetDoping (no ionization here)
    pnc = log.node["PotentialNodeCharge"]
    assert pnc.count("QTerm_bulk_traps") == 1 and "NetDoping" in pnc
    # BTBT: one edge term in the edge composite, which is what the equations assemble
    assert log.edge["ElectronGenerationEdge"].count("RTermEdge_btbt") == 1
    for eq, comp in (("ElectronContinuityEquation", ("ElectronGeneration", "ElectronGenerationEdge")),
                     ("HoleContinuityEquation", ("HoleGeneration", "HoleGenerationEdge"))):
        kw = _R.recorded(dev, eq, scope="region", loc="b")[0]["kwargs"]
        assert (kw["node_model"], kw["edge_volume_model"]) == comp
    inv = BP.recombination_terms(dev, "b")
    assert sorted(inv["node"]) == ["auger", "radiative", "trap0"] and list(inv["edge"]) == ["btbt"]
    assert all(pm is not None for _, pm in list(inv["node"].values()) + list(inv["edge"].values()))
    # BTBT's Hurkx D uses n_ie (the D1d merge note)
    assert "n_ie@n0" in log.edge["BTBT_U"] and "n_i@" not in log.edge["BTBT_U"]


def _oracle_terms(dev):
    """Every recombination term re-evaluated in numpy from the SOLVED state (independent of the
    DEVSIM strings): base SRH, the E_t = 0.2 eV trap level, Auger, radiative (node, m^-3 s^-1) and
    BTBT (edge)."""
    n, p, nie = _v(dev, "Electrons"), _v(dev, "Holes"), _v(dev, "n_ie")
    s = nie / NI
    np_ = n * p - nie * nie
    u_srh = np_ / (TAU * (n + NI * s) + TAU * (p + NI * s))
    e_t = TRAP.energy_eV / KT_Q
    n1, p1 = NI * math.exp(e_t) * s, NI * math.exp(-e_t) * s
    u_trap = np_ / (TRAP.tau_p0_s * (n + n1) + TRAP.tau_n0_s * (p + p1))
    a = OPTS.auger
    u_aug = (a.C_n_m6s * n + a.C_p_m6s * p) * np_
    u_rad = OPTS.radiative.B_m3s * np_
    # BTBT (Hurkx 1992 Si, the edge field, D = mean of the two end nodes, n_ie)
    E = np.sqrt(_e(dev, "ElectricField") ** 2 + 1.0)
    i0 = _e(dev, "node_index@n0").astype(int) if "node_index@n0" in ds.get_edge_model_list(
        device=dev, region="b") else None
    if i0 is None:
        ds.edge_from_node_model(device=dev, region="b", node_model="node_index")
        i0 = _e(dev, "node_index@n0").astype(int)
    i1 = _e(dev, "node_index@n1").astype(int)
    D = (n * p - nie ** 2) / ((n + nie) * (p + nie))
    A, B, P = 4.0e14 * 1e6 * 100.0 ** -2.5, 1.9e7 * 100.0, 2.5
    u_btbt = A * E ** P * np.exp(-B / E) * 0.5 * (D[i0] + D[i1])
    return dict(srh=u_srh, trap=u_trap, auger=u_aug, rad=u_rad), u_btbt, (i0, i1)


def test_total_generation_is_the_sum_of_the_terms(reverse):
    dev, _, _ = reverse
    node, u_btbt, _ = _oracle_terms(dev)
    q = sc.e
    total = node["srh"] + node["trap"] + node["auger"] + node["rad"]
    got = -_v(dev, "ElectronGeneration") / q
    scale = np.max(np.abs(total))
    assert np.max(np.abs(got - total)) / scale < 1e-12
    assert np.array_equal(_v(dev, "HoleGeneration"), -_v(dev, "ElectronGeneration"))
    # each term individually against its own installed model
    for k, m in (("srh", "USRH"), ("trap", "RTerm_trap0"), ("auger", "RTerm_auger"),
                 ("rad", "RTerm_radiative")):
        sk = max(np.max(np.abs(node[k])), 1e-300)
        assert np.max(np.abs(_v(dev, m) - node[k])) / sk < 1e-12, k
        assert np.max(np.abs(node[k])) > 0.0, k                 # every term is live in this fixture
    # the edge composite == BTBT alone (one edge term), and BTBT is a real GENERATION here
    ge = -_e(dev, "ElectronGenerationEdge") / q
    assert np.max(np.abs(ge - u_btbt)) / np.max(np.abs(u_btbt)) < 1e-12
    assert np.min(u_btbt) < 0.0
    # integrated: q Int G over the assembled residual == sum of the per-term integrals
    V, ENV = _v(dev, "NodeVolume"), _e(dev, "EdgeNodeVolume")
    tot = float(np.sum(got * V) + np.sum(ge * 2.0 * ENV))
    parts = [float(np.sum(_v(dev, m) * V)) for m in ("USRH", "RTerm_trap0", "RTerm_auger",
                                                      "RTerm_radiative")]
    parts.append(float(np.sum(_e(dev, "BTBT_U") * 2.0 * ENV)))
    assert abs(tot - sum(parts)) <= 1e-12 * sum(abs(x) for x in parts)
    assert abs(parts[-1]) > 0.1 * abs(tot)                      # BTBT carries a real share


def test_trapped_charge_uses_the_narrowed_occupancy(reverse):
    dev, _, _ = reverse
    n, p, nie = _v(dev, "Electrons"), _v(dev, "Holes"), _v(dev, "n_ie")
    s = nie / NI
    e_t = TRAP.energy_eV / KT_Q
    n1, p1 = NI * math.exp(e_t) * s, NI * math.exp(-e_t) * s
    f = (TRAP.tau_p0_s * n + TRAP.tau_n0_s * p1) / (TRAP.tau_p0_s * (n + n1) + TRAP.tau_n0_s * (p + p1))
    q_t = -TRAP.density_m3 * f
    pnc = -sc.e * (p - n + _v(dev, "NetDoping") + q_t)
    got = _v(dev, "PotentialNodeCharge")
    assert np.max(np.abs(got - pnc)) <= 1e-12 * np.max(np.abs(pnc))


def test_charged_trap_contacts_pin_the_trap_aware_root(reverse):
    """With a charged level the ohmic contacts pin the charge-neutral root INCLUDING the trapped
    charge (one contact rule with BGN / ionization). Oracle: brentq on p - n + N - N_t f(E_F) = 0
    with n p = n_ie^2 and the Fermi occupancy f = n/(n + n1), n1 = n_ie exp(E_t/kT)."""
    dev, _, _ = reverse
    x = _v(dev, "x")
    nie = _v(dev, "n_ie")
    nn = _v(dev, "NeutralElectrons")
    for k in (0, x.size - 1):
        N = -ND if x[k] < XJ else ND
        ni_e = nie[k]
        n1 = ni_e * math.exp(TRAP.energy_eV / KT_Q)

        def g(ln_n):
            n = math.exp(ln_n)
            return ni_e * ni_e / n - n + N - TRAP.density_m3 * n / (n + n1)
        ref = math.exp(brentq(g, math.log(ni_e * ni_e / 1e27), math.log(1e27), xtol=1e-14, rtol=1e-15))
        assert nn[k] == pytest.approx(ref, rel=1e-11)
    # the trap moves the n-side contact density (a load-bearing check of the rule): ~1e21 of 5e24
    k = x.size - 1
    assert abs(nn[k] - _trap_free_root(ND, nie[k])) / _trap_free_root(ND, nie[k]) > 1e-4


def _trap_free_root(N, nie):
    return 0.5 * (N + math.sqrt(N * N + 4 * nie * nie)) if N > 0 else \
        nie * nie / (0.5 * (-N + math.sqrt(N * N + 4 * nie * nie)))


def test_terms_sum_never_overwrite_and_foreign_slot_refused(reverse):
    dev, _, _ = reverse
    before = _v(dev, "ElectronGeneration").copy()
    BP.add_recombination_term(dev, "b", "zz_extra", "1e20", pair="0")
    after = _v(dev, "ElectronGeneration")
    assert np.allclose(after, before - sc.e * 1e20, rtol=1e-12, atol=0)
    BP.add_recombination_term(dev, "b", "zz_extra", "0", pair="0")      # re-adding REPLACES
    assert np.allclose(_v(dev, "ElectronGeneration"), before, rtol=1e-12, atol=0)
    ge0 = _e(dev, "ElectronGenerationEdge").copy()
    BP.add_recombination_term(dev, "b", "zz_edge", "1e20", assembly="edge", pair="0")
    assert np.allclose(_e(dev, "ElectronGenerationEdge"), ge0 - sc.e * 1e20, rtol=1e-12, atol=0)
    ds.edge_model(device=dev, region="b", name="Foreign", equation="0")
    kw = _R.recorded(dev, "ElectronContinuityEquation", scope="region", loc="b")[0]["kwargs"]
    assert kw["edge_volume_model"] == "ElectronGenerationEdge"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _R.amend_region_equation(dev, "b", "ElectronContinuityEquation", edge_volume_model="Foreign")
    BP.add_recombination_term(dev, "b", "zz_nopair", "0")               # no pair rate given ...
    from dynameta.carriers import noise_analysis as NA
    with pytest.raises(RuntimeError, match="without a pair"):            # ... the noise refuses it
        NA._pair_term_models(dev, "b", True)
    for nm in ("zz_extra", "zz_nopair"):
        BP.add_recombination_term(dev, "b", nm, "0", pair="0")
    BP.add_recombination_term(dev, "b", "zz_edge", "0", assembly="edge", pair="0")


# =====================================================================================================
# (4) noise: FDT with a non-midgap trap level, and with BTBT at equilibrium
# =====================================================================================================

@pytest.fixture
def noise_mod():
    V = pytest.importorskip("validation.devsim_noise")
    V.fresh()
    V.set_bias._v = {}
    yield V
    V.fresh()


FDT_FREQS = [1.0, 1e5, 1e7, 1e9]


def _drop_pair(NA, kind, names):
    real = NA._pair_term_models

    def drop(dev, reg, shot):
        out = real(dev, reg, shot)
        out[kind] = [m for m in out[kind] if m not in names] if names else []
        return out
    return real, drop


def test_noise_fdt_with_a_non_midgap_trap_level(noise_mod):
    """The FDT diode of validation/devsim_noise.py (Ge-like n_i 1e19, tau 1 us) with a recombination-
    only level at E_t - E_i = +0.15 eV (tau 0.3/0.5 us) and the built-in channel moved to -0.1 eV:
    S_I = 4kT Re Y to 1e-4 at 1 Hz - 1 GHz (the D2a gate; MEASURED <= 2.2e-6). Load-bearing: the trap
    moves Re Y(1 Hz) by > 20 % (MEASURED 1.49x), and with the trap's pair rate dropped from the noise
    the FDT breaks."""
    V = noise_mod
    trap = PT.TrapLevel(energy_eV=0.15, kind="neutral", tau_n0_s=3e-7, tau_p0_s=5e-7)
    kw = dict(traps=(trap,), srh_trap_energy_eV=-0.1)
    r_on, sn, Y_on = V.gate_b_fdt(open_circuit=False, freqs=FDT_FREQS, **kw)
    assert np.max(np.abs(r_on - 1)) < 1e-4, r_on
    _, _, Y_off = V.gate_b_fdt(open_circuit=False, freqs=FDT_FREQS)
    assert abs(Y_on.real[0] / Y_off.real[0] - 1) > 0.2
    from dynameta.carriers import noise_analysis as NA
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, **kw)
    real, drop = _drop_pair(NA, "node", ("PTerm_trap0",))
    try:
        NA._pair_term_models = drop
        s_drop = V.noise([1.0]).S[0]
    finally:
        NA._pair_term_models = real
    assert abs(s_drop / (4 * V.KT * Y_on.real[0]) - 1) > 0.05


CHARGED_TRAP = PT.TrapLevel(energy_eV=0.15, density_m3=2e21, kind="acceptor", tau_n0_s=3e-7,
                            tau_p0_s=5e-7)


def test_noise_fdt_with_a_charged_trap_level_quasi_static(noise_mod):
    """The same diode with an ACCEPTOR level (N_t 2e21 = 20 % of the doping; trap-aware contacts):
    in the quasi-static regime (1 Hz) S_I = 4kT Re Y to 1e-4 (MEASURED 1.4e-6), load-bearing (Re Y
    1.57x). Above it see the strict xfail below."""
    V = noise_mod
    r_on, sn, Y_on = V.gate_b_fdt(open_circuit=False, freqs=[1.0], traps=(CHARGED_TRAP,),
                                  srh_trap_energy_eV=-0.1)
    assert abs(r_on[0] - 1) < 1e-4, r_on
    _, _, Y_off = V.gate_b_fdt(open_circuit=False, freqs=[1.0])
    assert abs(Y_on.real[0] / Y_off.real[0] - 1) > 0.2


@pytest.mark.xfail(strict=True, reason="D1c limit: CHARGED traps are quasi-static (no trap-occupancy "
                   "variable), so the ssac linear response is not charge-conserving above the "
                   "quasi-static regime and the FDT cannot hold there (MEASURED S/4kT Re Y - 1 = "
                   "5.6e-3 at 100 kHz, 6.6e-2 at 10 MHz). Turns green when trap kinetics land.")
def test_noise_fdt_with_a_charged_trap_level_high_frequency(noise_mod):
    V = noise_mod
    r_on, _, _ = V.gate_b_fdt(open_circuit=False, freqs=[1e5, 1e7], traps=(CHARGED_TRAP,),
                              srh_trap_energy_eV=-0.1)
    assert np.max(np.abs(r_on - 1)) < 1e-4, r_on


# BTBT params chosen so the 0.36 V built-in field (~2e6 V/m) of the FDT diode tunnels at a rate
# comparable to its depletion SRH (the Si set gives exp(-B/E) ~ e^-800 there: nothing to test).
# A = 1e12 is an INTEGRAL float: before the v0.14 integration fix of tunneling.btbt_rate_expr it was
# printed "1000000000000" and DEVSIM clamped it to 2147483647 (this gate then saw Re Y move 8e-4).
FDT_BTBT = BTBTParams(A_SI=1.0e12, B_V_per_m=1.0e7, P=2.5)


def test_noise_fdt_with_btbt_at_equilibrium(noise_mod):
    """Zero bias, BTBT on: the net BTBT rate is ZERO (Hurkx D = 0 at n p = n_i^2), but its pair rate
    G + R and its linear response are not. S_I = 4kT Re Y must still hold (the 1-D edge assembly takes
    D as the mean of the two end nodes, so the discrete identity is exact only to O(h^2) inside the
    junction: gate 1e-4 as D2a's; MEASURED <= 2.3e-6). Load-bearing: BTBT moves Re Y(1 Hz) by > 20 %
    (MEASURED 1.39x), and with the BTBT pair rate dropped from the noise the FDT breaks."""
    V = noise_mod
    r_on, sn, Y_on = V.gate_b_fdt(open_circuit=False, freqs=FDT_FREQS, btbt=FDT_BTBT)
    import devsim as _ds
    dev = _ds.get_device_list()[0]
    u = np.asarray(_ds.get_edge_model_values(device=dev, region="bulk", name="BTBT_U"))
    assert np.max(np.abs(u)) < 1e-4 * np.max(np.abs(np.asarray(_ds.get_edge_model_values(
        device=dev, region="bulk", name="PTermEdge_btbt"))))      # net rate ~ 0 at equilibrium
    assert np.max(np.abs(r_on - 1)) < 1e-4, r_on
    _, _, Y_off = V.gate_b_fdt(open_circuit=False, freqs=FDT_FREQS)
    assert abs(Y_on.real[0] / Y_off.real[0] - 1) > 0.2, (Y_on.real[0], Y_off.real[0])
    from dynameta.carriers import noise_analysis as NA
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, btbt=FDT_BTBT)
    real, drop = _drop_pair(NA, "edge", ())
    try:
        NA._pair_term_models = drop
        s_drop = V.noise([1.0]).S[0]
    finally:
        NA._pair_term_models = real
    assert abs(s_drop / (4 * V.KT * Y_on.real[0]) - 1) > 0.05


# =====================================================================================================
# (5) 2-D element assembly of the BTBT pair rate
# =====================================================================================================

def test_element_pair_rate_reaches_nodes_with_the_residual_weights():
    """2-D triangles, uniform field and uniform n, p: the element-assembled BTBT pair rate must reach
    each node as pair * NodeVolume_i (DEVSIM's sum of ElementNodeVolume over a node's element edges is
    its box volume -- the D1d quadrature identity), through the noise's element index map."""
    from dynameta.carriers import noise_analysis as NA
    _wipe()
    dev, mesh = "cmp2d", "cmp2d_m"
    ds.create_2d_mesh(mesh=mesh)
    for x in (0.0, 4e-9, 8e-9):
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=x, ps=1e-9)
    for y in (0.0, 4e-9, 8e-9):
        ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=y, ps=1e-9)
    ds.add_2d_region(mesh=mesh, material="Si", region="b")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="b", name="NetDoping", equation="1e24")
    BP.setup_bipolar_region(dev, "b", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=NI,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-3,
                            tau_p_s=1e-3, fd_enhancement=False, btbt=SILICON_HURKX)
    x = _v(dev, "x")
    ds.set_node_values(device=dev, region="b", name="Potential", values=(-1.5e8 * x).tolist())
    for var, val in (("Electrons", 3e18), ("Holes", 2e17)):
        ds.set_node_values(device=dev, region="b", name=var, values=[val] * x.size)
    nn = x.size
    i0 = np.zeros(1, dtype=int)
    P = NA._gr_pair_per_node(dev, "b", True, nn, i0[:0], i0[:0])
    n, p = 3e18, 2e17
    A, B, Pw = SILICON_HURKX.A_SI, SILICON_HURKX.B_V_per_m, SILICON_HURKX.P
    E = math.sqrt(1.5e8 ** 2 + 1.0)
    pair_btbt = A * E ** Pw * math.exp(-B / E) * (n * p + NI * NI) / ((n + NI) * (p + NI))
    pair_srh = (n * p + NI * NI) / (1e-3 * (n + NI) + 1e-3 * (p + NI))
    want = (pair_btbt + pair_srh) * _v(dev, "NodeVolume")
    assert np.max(np.abs(P - want) / want) < 1e-12
    _wipe()


# =====================================================================================================
# (6) D1a heterojunctions x D1b BGN: one contact rule, BGN-aware interface, noise on a heterointerface
# =====================================================================================================

# a TEST BGN set (not a GaAs model): Slotboom form, 14.1 mV at 1e17 cm^-3, with DIFFERENT band-edge
# splits on the two sides so the split itself is load-bearing
_BGN_N = dict(kind="slotboom", E_ref_eV=0.02, N_ref_m3=1e23, C=0.5)


@pytest.fixture(scope="module", params=["auto", "continuous"])
def hetero_bgn(request):
    """N-Al0.3Ga0.7As (gamma 0.3) / p-GaAs (gamma 0.7), 1e17 cm^-3 each, BGN on both sides, through the
    2-D LayeredDevsimBuilder (D1a's gate-B device), solved at zero bias, with the 'auto' (= thermionic
    here) and the 'continuous' interface model. MEASURED with the BGN interface terms removed: the
    effective-offset gate misses by 5.66 meV on both models (0.24310 vs 0.24876 eV), while the
    flat-E_F gate moves only to 2.05e-4 eV -- the battery drops where the minority density is below
    the 1 m^-3 floor of that gate, so the offset gate is the load-bearing one."""
    import dataclasses
    import warnings
    from validation.devsim_heterojunction import (ALGAAS, GAAS, column, stack_design, transport)
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    N = 1e23
    tp = dataclasses.replace(transport(GAAS, n_bg=N, physics="bipolar_dd", acceptor=True),
                             bipolar_options=BO.BipolarOptions(bgn=BO.BGNModel(**_BGN_N,
                                                                               conduction_fraction=0.7)))
    tn = dataclasses.replace(transport(ALGAAS, n_bg=N, physics="bipolar_dd"),
                             bipolar_options=BO.BipolarOptions(bgn=BO.BGNModel(**_BGN_N,
                                                                               conduction_fraction=0.3)))
    d = stack_design([("pgaas", "GaAs_p", 400e-9, tp), ("nalgaas", "AlGaAs_n", 400e-9, tn)])
    _wipe()
    b = LayeredDevsimBuilder(d, mesh_name="cmp_hjm", device_name="cmp_hjd",
                             heterointerface=request.param)
    out = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.solve(BiasPoint(label="v0", voltages={"anode": 0.0}), rel_tol=1e-8)
        out["frames"] = b.band_frames()
        out["ifaces"] = b.heterointerfaces()
        for r in ("pgaas", "nalgaas"):
            for nm in ("Potential", "Electrons", "Holes", "BgnDc", "BgnDv", "n_ie"):
                out[(r, nm)] = column(b, r, nm)
    finally:
        b.teardown()
        _wipe()
    out["mats"] = {"pgaas": GAAS, "nalgaas": ALGAAS}
    out["model"] = request.param
    out["dEg"] = BO.BGNModel(**_BGN_N).delta_Eg_eV(N + 0.0)
    return out


def test_heterojunction_with_bgn_built_in_voltage_matches_band_diagram(hetero_bgn):
    """q V_bi = W_p - W_n with the BGN band-edge shifts in the work functions: W_n = chi_A + gamma_A dEg
    - kT eta(N_D/N_c,A), W_p = chi_G + E_g,G - (1 - gamma_G) dEg + kT eta(N_A/N_v,G) (direct FD
    quadrature, the D1a oracle). Tolerance 2e-4 V (D1a's gate B: the g fit vs exact FD). Load-bearing:
    BGN moves V_bi by (gamma_A + 1 - gamma_G) dEg = 8.5 mV."""
    from validation.devsim_heterojunction import ALGAAS, GAAS, fd_eta, nc_m3
    from dynameta.constants import V_T
    h = hetero_bgn
    assert h["ifaces"] == {"pgaas__nalgaas": "thermionic" if h["model"] == "auto" else "continuous"}
    dEg = float(h["dEg"])
    assert dEg == pytest.approx(0.02 * math.sqrt(0.5), rel=1e-12)
    eta_n = fd_eta(1e23 / nc_m3(ALGAAS["me"]))
    eta_p = fd_eta(1e23 / nc_m3(GAAS["mh"]))
    w_n = ALGAAS["chi"] + 0.3 * dEg - eta_n * V_T
    w_p = GAAS["chi"] + GAAS["eg"] - (1 - 0.7) * dEg + eta_p * V_T
    _, psip = h[("pgaas", "Potential")]
    _, psin = h[("nalgaas", "Potential")]
    vbi_sim = psin[-1] - psip[0]
    assert vbi_sim == pytest.approx(w_p - w_n, abs=2e-4)
    vbi_no_bgn = (GAAS["chi"] + GAAS["eg"] + eta_p * V_T) - (ALGAAS["chi"] - eta_n * V_T)
    assert abs(vbi_sim - vbi_no_bgn) > 5e-3


def test_heterojunction_with_bgn_flat_fermi_level(hetero_bgn):
    """Zero bias: E_Fn and E_Fp flat at 0 through both regions and across the interface, measured from
    the SOLVED densities against the EFFECTIVE edges E_c,eff = q(Phi_c - psi) - q D_c, E_v,eff =
    E_c,eff... (exact FD quadrature, D1a's tolerance 2e-4 eV). The interface reads E_F against the
    effective edges (physics_heterojunction._bgn_interface_arguments); without it the junction carries a
    D_c,A - D_c,G battery."""
    from validation.devsim_heterojunction import fd_eta, nc_m3
    from dynameta.constants import V_T
    h = hetero_bgn
    for r in ("pgaas", "nalgaas"):
        mat = h["mats"][r]
        _, psi = h[(r, "Potential")]
        _, n = h[(r, "Electrons")]
        _, p = h[(r, "Holes")]
        _, dc = h[(r, "BgnDc")]
        _, dv = h[(r, "BgnDv")]
        ec = h["frames"][r][0] - psi - dc
        ev = h["frames"][r][0] - psi - mat["eg"] + dv
        checked = 0
        for k in range(len(psi)):
            if n[k] > 1.0:
                assert abs(ec[k] + V_T * fd_eta(n[k] / nc_m3(mat["me"]))) < 2e-4, (r, k, "n")
                checked += 1
            if p[k] > 1.0:
                assert abs(ev[k] - V_T * fd_eta(p[k] / nc_m3(mat["mh"]))) < 2e-4, (r, k, "p")
        assert checked > 10


def test_heterojunction_with_bgn_effective_offsets_at_the_interface(hetero_bgn):
    """The interface relation itself: on the two node copies of the interface, E_c,eff - E_F and
    E_v,eff - E_F read from the SOLVED densities (exact FD quadrature, E_F = 0) must differ by the
    Anderson offsets PLUS the BGN band-edge shifts, dE_c,eff = dE_c - D_c,A + D_c,G and
    dE_v,eff = dE_v + D_v,G - D_v,A (D1a's gate at 1e-6 eV). This is what the BGN-aware interface
    arguments enforce; reading E_F against the unnarrowed edges misses by D_c,G - D_c,A = 5.7 meV."""
    from validation.devsim_heterojunction import ALGAAS, DEC, DEV, GAAS, fd_eta, nc_m3
    from dynameta.constants import V_T
    h = hetero_bgn
    _, n_g = h[("pgaas", "Electrons")]
    _, p_g = h[("pgaas", "Holes")]
    _, n_a = h[("nalgaas", "Electrons")]
    _, p_a = h[("nalgaas", "Holes")]
    dc_g, dc_a = h[("pgaas", "BgnDc")][1][-1], h[("nalgaas", "BgnDc")][1][0]
    dv_g, dv_a = h[("pgaas", "BgnDv")][1][-1], h[("nalgaas", "BgnDv")][1][0]
    ec_g = -V_T * fd_eta(n_g[-1] / nc_m3(GAAS["me"]))
    ec_a = -V_T * fd_eta(n_a[0] / nc_m3(ALGAAS["me"]))
    ev_g = V_T * fd_eta(p_g[-1] / nc_m3(GAAS["mh"]))
    ev_a = V_T * fd_eta(p_a[0] / nc_m3(ALGAAS["mh"]))
    assert abs(dc_g - dc_a) > 5e-3                                # the split differs across the junction
    assert ec_a - ec_g == pytest.approx(DEC - dc_a + dc_g, abs=1e-6)
    assert ev_g - ev_a == pytest.approx(DEV + dv_g - dv_a, abs=1e-6)


def test_noise_fdt_on_a_continuous_qfl_heterointerface(noise_mod):
    """The FDT diode as a heterojunction (n_i 1e19 | 3e18, affinity step 0.05 eV, frame shift
    s(r_p) = 18.9 mV through the shared contact rule) with D1a's CONTINUOUS quasi-Fermi-level interface:
    S_I = 4kT Re Y to 1e-4 over 1 Hz - 1 GHz (MEASURED <= 5.7e-6). Load-bearing: Re Y(1 Hz) is 3.06
    against the homojunction's 6.78 S/m^2."""
    V = noise_mod
    dev, fr = V.hetero_two_region("continuous")
    assert fr["r_p"][1] != 0.0
    sn = V.noise(FDT_FREQS)
    Y = V.ssac_Y(FDT_FREQS)
    r = sn.S / (4 * V.KT * Y.real) - 1
    assert np.max(np.abs(r)) < 1e-4, r
    assert abs(Y.real[0] / 6.78 - 1) > 0.3


@pytest.mark.xfail(strict=True, reason="LIMIT (documented): the thermionic-emission interface flux is a "
                   "DEVSIM fluxterm with no noise source -- its shot noise 2q(J_0->1 + J_1->0) is not in "
                   "noise_analysis' source set, so wherever the interface resistance matters the FDT "
                   "misses (MEASURED S/4kT Re Y - 1 = -0.69 at 1 Hz with A* x 1e-6). Turns green when an "
                   "interface TE source is added.")
def test_noise_fdt_on_a_thermionic_heterointerface(noise_mod):
    V = noise_mod
    V.hetero_two_region("thermionic", astar_scale=1e-6)
    sn = V.noise(FDT_FREQS[:2])
    Y = V.ssac_Y(FDT_FREQS[:2])
    assert np.max(np.abs(sn.S / (4 * V.KT * Y.real) - 1)) < 1e-4
