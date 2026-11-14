"""Gates for the general TWO-POINT (event) noise source of dynameta.carriers.noise_analysis (DynaMeta v0.15,
plan item 4.0.1; P1 section 0.4, gate C0(d); user decisions 7 and 9).

S = q^2 2P |Sum_r c_r y_r|^2 per event (noise_analysis module doc, item 3). Today's GR source -- SRH, every
PTerm_* term, the edge-assembled BTBT -- now goes through this one kernel.

  (1) BIT-IDENTITY (0 ulp, the plan's tolerance when the summation order is unchanged -- it is: the same
      per-node P, the same factor order q^2 |.|^2 (2P), c = +-1.0 products exact, y_p + (-y_n) == y_p - y_n
      in IEEE arithmetic). On the D2a FDT diode with BTBT AND a non-midgap trap level, at equilibrium and
      at 0.2 V forward, the per-node GR map and the GR total equal the pre-v0.15 expression
      q^2 |y_p - y_n|^2 (2P), re-typed below from 0e88d31 and evaluated on DEVSIM's own adjoint fields.
      This is a consistency check of the re-expression (it shares _gr_pair_per_node, unchanged); the
      CROSS-VERSION measurement (0e88d31 vs this tree in separate processes: S, every by_mechanism entry
      and every per-node map float.hex-identical on three diodes x two biases) is in the audit note.
  (2) The INDEPENDENT oracle is the fluctuation-dissipation theorem S_I = 4 kT Re Y (ssac) on the same
      diode, budget 1e-4 (the D2a gate B budget; MEASURED <= 2.3e-6 there).
  (3) Kernel: a synthetic event with three rows of arbitrary coefficients and a circuit row against the
      closed form evaluated here with scipy constants.
  ADVERSARIAL: the GR rows with the WRONG relative sign (+1, +1: |y_p + y_n|) break both the bit-identity
  and the FDT.
OLD CODE (0e88d31, MEASURED by running this file against it): the adversarial test, (3) and the registry
test FAIL (no TwoPointSource / two_point_psd / _gr_source / TWO_POINT_MECHANISMS: AttributeError). (1) is
an IDENTITY gate (it passes on the old code by construction -- what it must show is that the new path did
not move; its sensitivity is the adversarial test) and (2) is the oracle the re-expression must keep.
Runtime ~15 s.
"""
import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
V = pytest.importorskip("validation.devsim_noise")
from scipy.constants import e as QE, k as KB                  # noqa: E402

from dynameta.carriers import noise_analysis as NA            # noqa: E402
from dynameta.carriers import physics_traps as PT             # noqa: E402
from dynameta.carriers.tunneling import BTBTParams            # noqa: E402

FDT_BTBT = BTBTParams(A_SI=1.0e12, B_V_per_m=1.0e7, P=2.5)    # tests/test_devsim_phase_composition.py
TRAP = PT.TrapLevel(energy_eV=0.15, kind="neutral", tau_n0_s=3e-7, tau_p0_s=5e-7)
FF = [1.0, 1e5, 1e7, 1e9]
EQ_N, EQ_P = "ElectronContinuityEquation", "HoleContinuityEquation"


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()


def _build():
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, btbt=FDT_BTBT,
                 traps=(TRAP,), srh_trap_energy_eV=-0.1)
    return "fdt"


def _legacy_gr(dev, f):
    """The pre-v0.15 GR map, RE-TYPED from 0e88d31 noise_analysis.noise_spectrum:
        s_i = Q_E * Q_E * np.abs(ys[_EQ_P] - ys[_EQ_N]) ** 2 * d.gr,   d.gr = 2.0 * P
    with ys DEVSIM's native noise fields plus the circuit-contact fix-up (module doc, item 4), at one
    frequency."""
    out = "V1.I"
    n_nodes = len(V.ds.get_node_model_values(device=dev, region="bulk", name="x"))
    P = NA._gr_pair_per_node(dev, "bulk", True, n_nodes, *NA._edge_nodes(dev, "bulk"))
    with V.quiet():
        ds = V.ds
        ds.solve(type="noise", frequency=float(f), output_node=out)
    yc = complex(float(ds.get_circuit_node_value(node="vhi", solution="noise_V1.I_real")),
                 float(ds.get_circuit_node_value(node="vhi", solution="noise_V1.I_imag")))
    hi = NA._contact_nodes(dev, "bulk", "hi")
    ys = {}
    for eq in (EQ_N, EQ_P):
        y = (np.asarray(ds.get_node_model_values(device=dev, region="bulk", name=out + "_" + eq + "_real"))
             + 1j * np.asarray(ds.get_node_model_values(device=dev, region="bulk",
                                                         name=out + "_" + eq + "_imag")))
        y[hi] = yc
        ys[eq] = y
    NA._cleanup_noise_models(dev, ["bulk"], out)
    gr = 2.0 * P
    return NA.Q_E * NA.Q_E * np.abs(ys[EQ_P] - ys[EQ_N]) ** 2 * gr


@pytest.mark.parametrize("bias", [0.0, 0.2])
def test_gr_through_the_two_point_kernel_is_bit_identical(bias):
    """GATE (1): 0 ulp, per node and in total, at every frequency, BTBT + trap level on (GR = 61 % of S at
    1 Hz at equilibrium, MEASURED)."""
    dev = _build()
    if bias:
        V.set_bias(bias, 1e8)
    sn = V.noise(FF, maps=True)
    for k, f in enumerate(FF):
        legacy = _legacy_gr(dev, f)
        new = sn.maps["bulk"]["gr"][k]
        assert np.array_equal(new, legacy), np.max(np.abs(new - legacy) / np.abs(legacy).max())
        assert sn.by_mechanism["gr"][k] == float(np.sum(legacy))
    assert sn.by_mechanism["gr"][0] / sn.S[0] > 0.05              # GR is a real share of S


def test_wrong_relative_sign_breaks_bit_identity_and_the_fdt(monkeypatch):
    """ADVERSARIAL: GR rows (+1, +1) instead of (+1, -1). The map no longer equals the legacy one and the
    FDT misses by far more than its 1e-4 budget (MEASURED S/(4kT Re Y) - 1 = +6.53 at 1 Hz, against
    -1.3e-7 with the right sign; GR is 61 % of S there)."""
    dev = _build()
    Y = V.ssac_Y([1.0])
    real = NA._gr_source

    def wrong(region, P):
        s = real(region, P)
        return NA.TwoPointSource(s.mechanism, s.region, s.rate,
                                 (s.rows[0], NA.SourceRow(region, EQ_N, np.ones(P.size), s.rows[1].nodes)),
                                 s.map_nodes)
    monkeypatch.setattr(NA, "_gr_source", wrong)
    sn = V.noise([1.0], maps=True)
    assert not np.array_equal(sn.maps["bulk"]["gr"][0], _legacy_gr(dev, 1.0))
    T = V.ds.get_parameter(device=dev, region="bulk", name="V_t") * QE / KB
    assert abs(sn.S[0] / (4 * KB * T * Y.real[0]) - 1) > 0.1


def test_fdt_holds_through_the_two_point_path():
    """GATE (2), the independent oracle: S_I = 4 kT Re Y(f), 1 Hz - 1 GHz, BTBT + trap level on, budget
    1e-4 (D2a gate B); the trap and BTBT pair rates go through the new kernel."""
    dev = _build()
    sn = V.noise(FF)
    Y = V.ssac_Y(FF)
    T = V.ds.get_parameter(device=dev, region="bulk", name="V_t") * QE / KB
    assert np.max(np.abs(sn.S / (4 * KB * T * Y.real) - 1)) < 1e-4
    assert set(sn.by_mechanism) == {"diffusion_n", "diffusion_p", "gr"}   # no v0.15 key appears


def test_kernel_closed_form_with_a_circuit_row():
    """GATE (3): three node rows with coefficients (-1, 0.25, 0.75) (a hole at a start node, an electron
    spread over an end element -- the nonlocal-BTBT shape) and a circuit row, against
    q^2 2P |Sum c y|^2 written out here (scipy e). Budget: a handful of roundings, 16 eps relative."""
    rng = np.random.default_rng(7)
    n = 5
    y = {r: rng.normal(size=n) + 1j * rng.normal(size=n) for r in ("a", "b", "c")}
    yc = 0.3 - 0.7j
    P = rng.uniform(1e3, 1e9, size=n)
    rows = (NA.SourceRow("reg", "a", -np.ones(n), np.arange(n)),
            NA.SourceRow("reg", "b", np.full(n, 0.25), np.arange(n)),
            NA.SourceRow("reg", "c", np.full(n, 0.75), np.arange(n)),
            NA.SourceRow(None, "gate", np.full(n, 1.0)))
    src = NA.TwoPointSource("btbt_nl", "reg", P, rows)

    def field(row):
        return yc if row.region is None else y[row.equation][row.nodes]
    got = NA.two_point_psd(src, field)
    ref = QE ** 2 * 2 * P * np.abs(-y["a"] + 0.25 * y["b"] + 0.75 * y["c"] + yc) ** 2
    assert np.max(np.abs(got / ref - 1)) < 16 * np.finfo(float).eps
    # GR special case: (+1, -1) at the same node is |y_p - y_n|^2
    gr = NA._gr_source("reg", P)
    ys = {EQ_P: y["a"], EQ_N: y["b"]}
    got = NA.two_point_psd(gr, lambda row: ys[row.equation][row.nodes])
    assert np.array_equal(got, NA.Q_E * NA.Q_E * np.abs(y["a"] - y["b"]) ** 2 * (2.0 * P))


def test_mechanism_registry():
    """User decision 9: separate keys, all known, all default-on (Hooge stays opt-in); a caller wrapping
    noise_spectrum inherits DEFAULT_MECHANISMS."""
    assert NA.TWO_POINT_MECHANISMS == ("te_n", "te_p", "trap_n", "trap_p", "tunnel_ox", "btbt_nl")
    assert set(NA.TWO_POINT_MECHANISMS) <= set(NA.MECHANISMS)
    assert set(NA.TWO_POINT_MECHANISMS) <= set(NA.DEFAULT_MECHANISMS)
    assert not {"hooge_n", "hooge_p"} & set(NA.DEFAULT_MECHANISMS)
    import inspect
    assert inspect.signature(NA.noise_spectrum).parameters["mechanisms"].default == NA.DEFAULT_MECHANISMS
