"""Per-region / per-carrier Hooge alpha (v0.15 plan 4.2.4, user decision 33): noise_spectrum(hooge_alpha=
{region: alpha | (alpha_n, alpha_p)}) and TransportModel.hooge_alpha_n / hooge_alpha_p.

THE ORACLE (re-derived here, independent of noise_analysis). Hooge's law is a conductivity fluctuation
delta j = xi J, <xi(r) xi(r')> = alpha(r) / (n f) delta(r - r'), so the short-circuit terminal current noise
is S = Int alpha (J . grad y)^2 / (n f) dV, y the adjoint field of the terminal current. For a uniform bar of
length L, cross-section A (per unit depth in 2-D, per unit area in 1-D) between two ohmic contacts,
y = x / L and J = I / A, so a region r of length L_r contributes
    S_r f = alpha_r (I/A)^2 L^-2 A L_r / n = I^2 (L_r / L)^2 alpha_r / N_r,      N_r = n A L_r,
the resistance-fluctuation sum S_I / I^2 = Sum_r (R_r / R)^2 S_{R_r} / R_r^2 of resistors in series. For
one region it is the familiar alpha I^2 / (N f). Per carrier, with both carriers uniform, each carrier's
current I_c = I sigma_c / sigma carries its own alpha_c and N_{c,r} = c A L_r.

FINDING against the plan's stated gate: "S f = I^2 Sum_r alpha_r / N_r" omits the (L_r / L)^2 = (R_r / R)^2
weights. On the series fixture below it is off by a factor 2.41 (asserted). The weighted form is the
physics, and it is what these gates use.

THE FIXTURES are exactly uniform, so the closed form is valid on the DISCRETE device (asserted first: n
uniform to 1e-12, I = sigma V A / L to 1e-9):
  * a 2-D unipolar DD bar (ITO-like, n = 4e26, the gate of tests/test_devsim_noise_hooge.py) on DEVSIM's
    rectilinear mesh, cut at x = L_a into two REGIONS joined by a carrier-continuous interface, the same
    material on both sides -- the regions differ only in their alpha, so the mapping is the only thing
    tested (a doping or mobility step would add a Debye-length space-charge layer, ~L_D / L = 4e-3 at
    1e22, far outside the budget);
  * a 1-D INTRINSIC bipolar bar (n = p = n_i = 1e22, no doping, two regions): both carriers conduct
    (I_n / I = mu_n / (mu_n + mu_p) = 0.738), so a per-carrier (alpha_n, alpha_p) is visible in both.
DERIVED BUDGETS. (a) The TOTAL with one alpha everywhere (the gate F3 budget, validation/devsim_noise.py):
the SG edge density of noise_analysis is n_eff = n (v/2) coth(v/2) <= n (1 + v^2/12) for uniform n and a
linear potential (v = the edge's SG argument), every P1 couple of a right triangle is >= 0 (no sign-mixed
amplification), so |S f N / (alpha I^2) - 1| <= max v^2/12 + 4 |I / I_ohm - 1| + 1e-11.
(b) THE SPLIT between regions (and so any total with UNEQUAL alphas) is first order in the departure of the
adjoint field from y = x / L, which the total only sees at second order (Sum of Delta y is fixed by the two
contacts). Under bias the DD contact layer shifts the adjoint by O(V): MEASURED at 1 mV, DEVSIM's own y
next to the grounded contact sits 6.3e-9 above x / L (3.2e-7 of an edge's Delta y), moving +5.4e-8 of the
Hooge noise into the contact-side region and -2.4e-8 out of the other, total unchanged to 1e-17. With
eta_T = (dy/dx)_T L - 1 per element (read here from DEVSIM's native adjoint solve, its contact rows filled
with 0 / the circuit value as the module doc states -- the field noise_analysis consumes, not its source
assembly, which is what is under test), S_T = S_T^closed (1 + eta_T)^2 (1 + O(v^2/12)), so each region's
ratio is within (1 + max|eta|)^2 - 1 + the (a) terms. Both are evaluated in each test; the plan's 1e-4
(gate F's budget) is asserted as the outer bound.

OLD CODE (aacfe46, MEASURED): every mapping gate fails at the call -- float(hooge_alpha) raises TypeError
("float() argument must be a string or a real number, not 'dict'"), asserted by name nowhere here because
the new code does not raise; the audit note records the run. TransportModel had no hooge_alpha_n /
hooge_alpha_p (TypeError on the keyword).
"""
import hashlib
import math

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
import validation.devsim_noise as V                                         # noqa: E402
from dynameta.carriers import eq_registry as _R                             # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP                      # noqa: E402
from dynameta.carriers import physics_drift_diffusion as DD                 # noqa: E402
from dynameta.carriers.devsim_env import import_devsim                      # noqa: E402
from dynameta.carriers.physics_equilibrium import M_E                       # noqa: E402

ds = import_devsim()

QE = V.QE                                   # scipy.constants (oracle side)
FF = np.array([1.0, 1.0e3])
V_BIAS = 1e-3
# 2-D series bar
L_A, L_B, WIDTH = 120e-9, 280e-9, 60e-9     # L_a != L_b: a label swap is visible
N_BG, MU = 4e26, 0.004
ALPHA_A, ALPHA_B = 1.0e-4, 1.0e-3           # Hooge 1994's perfect-lattice value and a "measured" one
# 1-D intrinsic bipolar bar. Its neutral (ambipolar) perturbations are NOT screened, so the bias bends the
# adjoint field over the WHOLE bar at O(V / V_T) (MEASURED: 3.6 % at 1 mV): bias it at 0.1 uV
NI, L1, L2, V_BIAS_1D = 1e22, 3e-6, 7e-6, 1e-7


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()


def _continuous_interface(dev, iface, variables):
    for var, eq in variables:
        im = "cont_{}".format(var)
        ds.interface_model(device=dev, interface=iface, name=im, equation="{v}@r0 - {v}@r1".format(v=var))
        ds.interface_model(device=dev, interface=iface, name="{}:{}@r0".format(im, var), equation="1")
        ds.interface_model(device=dev, interface=iface, name="{}:{}@r1".format(im, var), equation="-1")
        _R.record_interface_equation(dev, iface, name=eq, interface_model=im, type="continuous")


def _series_bar_2d():
    """Two-region uniform DD bar, biased at V_BIAS; returns (I, fixture checks, max v)."""
    dev, mesh, L = "hm2", "hm2_m", L_A + L_B
    pad = 10e-9
    ds.create_2d_mesh(mesh=mesh)
    for pos, ps in ((-pad, pad), (0.0, 8e-9), (L_A, 8e-9), (L, 8e-9), (L + pad, pad)):
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=pos, ps=ps)
    ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=0.0, ps=10e-9)
    ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=WIDTH, ps=10e-9)
    ds.add_2d_region(mesh=mesh, material="ITO", region="ra")
    ds.add_2d_region(mesh=mesh, material="ITO", region="rb", xl=L_A, xh=L)
    ds.add_2d_region(mesh=mesh, material="metal", region="pad_lo", xl=-pad, xh=0.0)
    ds.add_2d_region(mesh=mesh, material="metal", region="pad_hi", xl=L, xh=L + pad)
    ds.add_2d_interface(mesh=mesh, name="jab", region0="ra", region1="rb")
    ds.add_2d_contact(mesh=mesh, name="lo", material="metal", region="ra", xl=0.0, xh=0.0, bloat=1e-10)
    ds.add_2d_contact(mesh=mesh, name="hi", material="metal", region="rb", xl=L, xh=L, bloat=1e-10)
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    with V.quiet():
        for r in ("ra", "rb"):
            DD.setup_semiconductor_region_dd(dev, r, n_bg_m3=N_BG, eps_static=9.5, dos_mass_kg=0.35 * M_E,
                                             mobility_m2Vs=MU)
        _continuous_interface(dev, "jab", (("Potential", "PotentialEquation"),
                                           ("Electrons", "ElectronContinuityEquation")))
        DD.setup_contact_ohmic_dd(dev, "lo")
        DD.setup_contact_ohmic_dd_circuit(dev, "hi", node_name="vhi", source_name="V1")
    ae = max(1e10, N_BG * 1e-12)
    V._solve(ae, 1e-9)
    V.set_bias(V_BIAS, ae)
    I = V.source_current()
    n = np.concatenate([ds.get_node_model_values(device=dev, region=r, name="Electrons") for r in ("ra", "rb")])
    vol = sum(float(np.sum(ds.get_node_model_values(device=dev, region=r, name="NodeVolume")))
              for r in ("ra", "rb"))
    v = np.concatenate([ds.get_edge_model_values(device=dev, region=r, name="vdiff_g") for r in ("ra", "rb")])
    chk = {"area": vol / (L * WIDTH) - 1.0, "n_spread": float(n.max() / n.min() - 1.0),
           "ohm": float(I / (QE * MU * N_BG * V_BIAS * WIDTH / L) - 1.0)}
    return I, chk, float(np.max(np.abs(v)))


def _series_oracle(I, alphas):
    """S_r f per region: I^2 (L_r/L)^2 alpha_r / N_r, N_r = n A L_r (module doc)."""
    L = L_A + L_B
    return {r: I * I * (Lr / L) ** 2 * a / (N_BG * WIDTH * Lr)
            for r, Lr, a in (("ra", L_A, alphas["ra"]), ("rb", L_B, alphas["rb"]))}


def _digest(sn):
    h = hashlib.sha256()
    h.update(np.asarray(sn.S, dtype=np.float64).tobytes())
    for r in sorted(sn.by_region_mechanism):
        for m in sorted(sn.by_region_mechanism[r]):
            h.update((r + m).encode())
            h.update(np.asarray(sn.by_region_mechanism[r][m], dtype=np.float64).tobytes())
    return h.hexdigest()


def _edge_ends(dev, region):
    ds.edge_from_node_model(device=dev, region=region, node_model="node_index")
    e = np.stack([np.asarray(ds.get_edge_model_values(device=dev, region=region, name="node_index@n{}".format(k)),
                             dtype=np.int64) for k in (0, 1)], axis=1)
    for k in (0, 1):
        ds.delete_edge_model(device=dev, region=region, name="node_index@n{}".format(k))
    return e


def _eta_max(dev, regions, eqs, L):
    """Budget (b): max |(dy/dx) L - 1| over the x-directed edges, y = DEVSIM's native adjoint of V1.I at 1 Hz
    with the contact nodes set to 0 (grounded 'lo') and to the circuit value of 'vhi' (circuit contact 'hi')."""
    from dynameta.carriers import extended_precision as XP
    with V.quiet():
        XP.solve(type="noise", frequency=1.0, output_node="V1.I")
    y_hi = float(ds.get_circuit_node_value(node="vhi", solution="noise_V1.I_real"))
    worst = 0.0
    for r in regions:
        x = np.asarray(ds.get_node_model_values(device=dev, region=r, name="x"))
        e = _edge_ends(dev, r)
        dx = x[e[:, 1]] - x[e[:, 0]]
        sel = np.abs(dx) > 1e-15
        for eq in eqs:
            y = np.asarray(ds.get_node_model_values(device=dev, region=r, name="V1.I_{}_real".format(eq)))
            for c, val in (("lo", 0.0), ("hi", y_hi)):
                try:
                    el = ds.get_element_node_list(device=dev, region=r, contact=c)
                except Exception:                                   # contact not on this region
                    continue
                y[np.unique(np.asarray(el, dtype=np.int64).ravel())] = val
            eta = (y[e[sel, 1]] - y[e[sel, 0]]) / dx[sel] * L - 1.0
            worst = max(worst, float(np.max(np.abs(eta))))
        for nm in list(ds.get_node_model_list(device=dev, region=r)):
            if nm.startswith("V1.I_"):
                ds.delete_node_model(device=dev, region=r, name=nm)
    return worst


def test_two_material_series_bar_per_region_alpha():
    """GATE H2 (2-D). alpha_a = 1e-4, alpha_b = 1e-3 on the two regions of a uniform series bar: each
    region's Hooge noise and the total match S_r f = I^2 (L_r/L)^2 alpha_r / N_r within the derived budget
    (module doc). Also (a): with ONE alpha in both regions the total meets the second-order budget.
    Adversarial: the swapped mapping (alpha_a <-> alpha_b) misses the original oracle by -49 %; the plan's
    unweighted form Sum alpha_r / N_r misses by a factor 2.41."""
    I, chk, vmax = _series_bar_2d()
    assert abs(chk["area"]) < 1e-12 and abs(chk["n_spread"]) < 1e-12, chk
    assert abs(chk["ohm"]) < 1e-9, chk
    tot_budget = vmax * vmax / 12.0 + 4.0 * abs(chk["ohm"]) + 1e-11          # (a)
    eta = _eta_max("hm2", ("ra", "rb"), ("ElectronContinuityEquation",), L_A + L_B)
    budget = (1.0 + eta) ** 2 - 1.0 + tot_budget                              # (b)
    assert budget < 1e-4                                       # the plan's gate-F bound is the outer one
    # (a) one alpha in both regions: the total is the single-material closed form, second order only
    one = V.noise(FF, hooge_alpha={"ra": ALPHA_B, "rb": ALPHA_B})
    r1 = one.by_mechanism["hooge_n"] * FF / (ALPHA_B * I * I / (N_BG * WIDTH * (L_A + L_B)))
    assert np.max(np.abs(r1 - 1.0)) < tot_budget, (r1 - 1.0, tot_budget)
    alphas = {"ra": ALPHA_A, "rb": ALPHA_B}
    sn = V.noise(FF, hooge_alpha=alphas)
    orc = _series_oracle(I, alphas)
    for r in ("ra", "rb"):
        ratio = sn.by_region_mechanism[r]["hooge_n"] * FF / orc[r]
        assert np.max(np.abs(ratio - 1.0)) < budget, (r, ratio - 1.0, budget)
    tot = sn.by_mechanism["hooge_n"] * FF / (orc["ra"] + orc["rb"])
    assert np.max(np.abs(tot - 1.0)) < budget, (tot - 1.0, budget)
    # adversarial 1: swap the labels -- must miss the oracle by far more than the budget
    sw = V.noise(FF, hooge_alpha={"ra": ALPHA_B, "rb": ALPHA_A})
    miss = sw.by_mechanism["hooge_n"] * FF / (orc["ra"] + orc["rb"]) - 1.0
    assert np.min(np.abs(miss)) > 0.4, miss                   # closed form 148/292 - 1 = -0.493
    # adversarial 2 (the plan's literal oracle, a finding): no (L_r/L)^2 weights -> off by 2.41x
    literal = I * I * sum(a / (N_BG * WIDTH * Lr) for a, Lr in ((ALPHA_A, L_A), (ALPHA_B, L_B)))
    assert abs(literal / (orc["ra"] + orc["rb"]) - 2.41) < 0.01


def test_region_missing_from_the_mapping_contributes_exactly_zero():
    """A region absent from the mapping has NO Hooge source: its by_region_mechanism entry is exactly 0.0,
    and the region that is mapped gets bit for bit the value it gets in the full mapping (the element sums
    are per region). An empty mapping is no Hooge at all (no hooge_* key)."""
    _series_bar_2d()
    full = V.noise(FF, hooge_alpha={"ra": ALPHA_A, "rb": ALPHA_B})
    part = V.noise(FF, hooge_alpha={"ra": ALPHA_A})
    assert np.all(part.by_region_mechanism["rb"]["hooge_n"] == 0.0)
    assert np.array_equal(part.by_region_mechanism["ra"]["hooge_n"], full.by_region_mechanism["ra"]["hooge_n"])
    assert np.array_equal(part.by_mechanism["hooge_n"], full.by_region_mechanism["ra"]["hooge_n"])
    none = V.noise(FF)
    empty = V.noise(FF, hooge_alpha={})
    assert not any(k.startswith("hooge") for k in empty.by_mechanism)
    assert _digest(empty) == _digest(none)


def test_float_path_is_bit_identical_to_the_mapping_of_the_same_float():
    """GATE H3 (same tree). hooge_alpha=2e-3 (the v0.14 float path, whose arithmetic math.sqrt(float(a)) is
    unchanged) equals {region: 2e-3} and {region: (2e-3, 2e-3)} bit for bit (sha256 of S and every
    by_region_mechanism array). The cross-version half -- the float path against aacfe46 in separate
    processes -- is in the audit note (MEASURED identical)."""
    _series_bar_2d()
    a = 2e-3
    d_float = _digest(V.noise(FF, hooge_alpha=a))
    assert _digest(V.noise(FF, hooge_alpha={"ra": a, "rb": a})) == d_float
    assert _digest(V.noise(FF, hooge_alpha={"ra": (a, a), "rb": (a, None)})) == d_float  # unipolar: no holes


def _intrinsic_bar_1d():
    """1-D intrinsic bipolar bar (two regions, n = p = n_i), biased at V_BIAS; returns (I, checks, max v)."""
    dev, mesh = "hm1", "hm1_m"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=L1 / 20, tag="lo")
    ds.add_1d_mesh_line(mesh=mesh, pos=L1, ps=L1 / 20, tag="mid")
    ds.add_1d_mesh_line(mesh=mesh, pos=L1 + L2, ps=L2 / 30, tag="hi")
    ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="r1", tag1="lo", tag2="mid")
    ds.add_1d_region(mesh=mesh, material="Si", region="r2", tag1="mid", tag2="hi")
    ds.add_1d_interface(mesh=mesh, name="j12", tag="mid")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    for r in ("r1", "r2"):
        ds.node_model(device=dev, region=r, name="NetDoping", equation="0.0")
        with V.quiet():
            BP.setup_bipolar_region(dev, r, eps_static=11.7, n_dos_m3=V.N_DOS, n_i_m3=NI, mobility_n_m2Vs=V.MU_N,
                                    mobility_p_m2Vs=V.MU_P, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=False)
            BP.setup_equilibrium_seed_models(dev, r)
            ds.node_model(device=dev, region=r, name="_seed_psi", equation=BP.equilibrium_seed_psi_expr(None))
    _continuous_interface(dev, "j12", (("Potential", "PotentialEquation"),
                                       ("Electrons", "ElectronContinuityEquation"),
                                       ("Holes", "HoleContinuityEquation")))
    with V.quiet():
        BP.setup_contact_ohmic_bipolar(dev, "lo")
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1")
    for r in ("r1", "r2"):
        V._seed(dev, ("Potential", "Electrons", "Holes"), region=r)
    V._solve(1e8)
    V.set_bias(V_BIAS_1D, 1e8)
    I = V.source_current()
    dens = np.concatenate([ds.get_node_model_values(device=dev, region=r, name=c)
                           for r in ("r1", "r2") for c in ("Electrons", "Holes")])
    v = np.concatenate([ds.get_edge_model_values(device=dev, region=r, name=m)
                        for r in ("r1", "r2") for m in ("vdiff_g", "vdiff_gp")])
    chk = {"spread": float(np.max(np.abs(dens / NI - 1.0))),
           "ohm": float(I / (QE * (V.MU_N + V.MU_P) * NI * V_BIAS_1D / (L1 + L2)) - 1.0)}
    return I, chk, float(np.max(np.abs(v)))


def test_per_carrier_alphas_on_an_intrinsic_bipolar_bar():
    """GATE H2 per carrier (1-D). Mapping {r1: (alpha_n 1e-4, alpha_p 3e-5), r2: (2e-5, None)}: each region's
    hooge_n / hooge_p matches I_c^2 (L_r/L)^2 alpha_c,r / N_c,r with I_n = I mu_n/(mu_n + mu_p), I_p = I
    mu_p/(mu_n + mu_p), N = n_i L_r (per unit area), within the derived budget; r2's hooge_p is exactly 0.
    Adversarial: swapping the carriers of r1's pair misses hooge_n by -70 %."""
    I, chk, vmax = _intrinsic_bar_1d()
    assert chk["spread"] < 1e-12 and abs(chk["ohm"]) < 1e-9, chk
    eta = _eta_max("hm1", ("r1", "r2"), ("ElectronContinuityEquation", "HoleContinuityEquation"), L1 + L2)
    budget = (1.0 + eta) ** 2 - 1.0 + vmax * vmax / 12.0 + 4.0 * abs(chk["ohm"]) + 1e-11
    assert budget < 1e-4
    L = L1 + L2
    In, Ip = I * V.MU_N / (V.MU_N + V.MU_P), I * V.MU_P / (V.MU_N + V.MU_P)
    amap = {"r1": (1e-4, 3e-5), "r2": (2e-5, None)}
    sn = V.noise(FF, hooge_alpha=amap)
    for r, Lr in (("r1", L1), ("r2", L2)):
        for k, (m, Ic) in enumerate((("hooge_n", In), ("hooge_p", Ip))):
            a = amap[r][k]
            got = sn.by_region_mechanism[r][m]
            if a is None:
                assert np.all(got == 0.0), (r, m, got)
                continue
            ratio = got * FF / (Ic * Ic * (Lr / L) ** 2 * a / (NI * Lr))
            assert np.max(np.abs(ratio - 1.0)) < budget, (r, m, ratio - 1.0, budget)
    sw = V.noise(FF, hooge_alpha={"r1": (3e-5, 1e-4), "r2": (2e-5, None)})
    miss = sw.by_region_mechanism["r1"]["hooge_n"] / sn.by_region_mechanism["r1"]["hooge_n"] - 1.0
    assert np.all(np.abs(miss + 0.7) < 1e-6), miss


def test_mapping_refusals():
    """A non-positive / non-finite alpha, a wrong-length pair, an unknown region and a region without a
    continuity equation (the inert contact pad) all raise ValueError naming the entry."""
    _series_bar_2d()
    for bad, match in (({"ra": 0.0}, "ra"), ({"ra": -1e-4}, "ra"), ({"ra": float("nan")}, "ra"),
                       ({"ra": (1e-4, math.inf)}, "ra"), ({"ra": (1e-4,)}, "pair"),
                       ({"nope": 1e-4}, "unknown region"), ({"pad_lo": 1e-4}, "continuity"), (0.0, "> 0")):
        with pytest.raises(ValueError, match=match):
            V.noise(FF, hooge_alpha=bad)


def test_transport_model_hooge_fields_have_no_default_and_validate():
    """TransportModel.hooge_alpha_n / _p default to None (no per-material default, decision 33); a set value
    must be finite and > 0; electrons need a carrier physics and holes bipolar_dd (otherwise the value would
    be silently ignored)."""
    import dataclasses

    from dynameta.materials import TransportModel
    dflt = {f.name: f.default for f in dataclasses.fields(TransportModel)}
    assert dflt["hooge_alpha_n"] is None and dflt["hooge_alpha_p"] is None

    def tm(**kw):
        base = dict(n_bg_m3=1e24, eps_static=10.0, dos_mass_kg_of_n_m3=lambda n: 0.3 * M_E + 0 * n,
                    physics="drift_diffusion", mobility_m2Vs_of_n_m3=lambda n: 0.01 + 0 * n)
        base.update(kw)
        return TransportModel(**base)
    assert tm(hooge_alpha_n=1e-4).hooge_alpha_n == 1e-4
    for bad in (0.0, -1e-4, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="hooge_alpha_n"):
            tm(hooge_alpha_n=bad)
    with pytest.raises(ValueError, match="hooge_alpha_p"):
        tm(hooge_alpha_p=1e-4)                                   # unipolar DD: no holes
    with pytest.raises(ValueError, match="hooge_alpha_n"):
        tm(physics="equilibrium", mobility_m2Vs_of_n_m3=None, hooge_alpha_n=1e-4)
