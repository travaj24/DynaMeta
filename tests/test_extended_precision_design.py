"""Gates for EXTENDED PRECISION SELECTED FROM THE DESIGN (v0.15, consolidated plan item 4.2.6, P2 item 7:
E1-E6 + the adversarial leak checks; user decision 37): Design.carriers = geometry.specs.CarrierSpec,
carriers.extended_precision.resolve_design_flag, LayeredDevsimBuilder / Stacked3DSpec.from_design
(extended_precision=None follows the design) and run_pipeline (reaches it through the default builder).
Audit note: docs/audit/2026-11-06-v0.15-extended-precision-design-iii-v.md.

DEVICES: a tiny lateral Si p-n diode on the 2-D builder (145 nodes; < 2 s per extended solve) for the
precedence / equivalence / pipeline / leak gates, and (nightly, @slow, ~2 min) the GaN-like lateral diode
of E4 (1131 nodes) whose ORACLE -- the Shockley + Sah-Noyce-Shockley current of an abrupt p-n junction --
is written out in THIS file with scipy.constants / scipy.integrate.quad (re-derived, not imported from
validation/extended_precision.py or from DynaMeta).

FAILS ON OLD CODE (bd925ac): geometry exports no CarrierSpec (ImportError at collection), Design has no
`carriers` field (TypeError), and LayeredDevsimBuilder(design) ignores any design setting (E3's spy sees 0
enable calls, E4's double-precision current is 3e4x off the oracle). Runtime of the fast tests: ~25 s
serial (box A, loaded).
"""
import dataclasses
import hashlib
import math
import re
import warnings

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
from scipy.constants import e as QE, k as KB, epsilon_0 as EPS0          # noqa: E402  (oracle consts)
from scipy.integrate import quad                                         # noqa: E402
from scipy.optimize import brentq                                        # noqa: E402

from dynameta.carriers import devsim_layered as DL                      # noqa: E402
from dynameta.carriers import extended_precision as XP                  # noqa: E402
from dynameta.geometry import CarrierSpec, Design, Electrode, Layer, Stack, UnitCell   # noqa: E402
from dynameta.geometry.specs import Mesh2DSpec                          # noqa: E402
from dynameta.materials import ConstantOptical, M_E, Material, MaterialRegistry, TransportModel  # noqa: E402
from dynameta.sweep import BiasPoint                                    # noqa: E402
from dynameta.warning_types import AdvisoryWarning                      # noqa: E402

VT = KB * 300.0 / QE
OVERRIDE = "overrides Design.carriers"
MIXED = re.escape("are default (double) while")


def _const(v):
    return lambda n: np.full_like(np.asarray(n, dtype=float), v)


# --- the two lateral diodes (p side x < P/2, abrupt; anode = x_lo on the p side, cathode grounded) -----
SI = dict(P=300e-9, T=10e-9, NA=1e24, ND=1e24, n_i=1e16, mu_n=0.135, mu_p=0.048, tau=1e-7, eps_r=11.7,
          Eg=1.12, mid=5e-9)
# GaN-like: the 1-D gate A device of the D2b audit (n_i 1e-4 m^-3 = 1e-10 cm^-3), as a 2-D lateral diode.
# Junction mesh 0.5 nm: the 1-D gate's nominal 1 nm gives +4.9e-3 on this 2-D mesh (measured; 0.5 nm:
# +8.4e-4, 0.25 nm: +4.6e-4 at 1.0 V). CAUSE (v0.15 follow-up, test_e4_mesh_cause_*): not the 2-D physics but
# DEVSIM's line grading, which leaves a 0.77 nm remainder edge ending at the junction line, right on the
# ~2.6 nm-wide SRH recombination peak; a clean uniform 1 nm junction band gives -6.4e-4.
GAN = dict(P=2e-6, T=10e-9, NA=1e23, ND=1e23, n_i=1e-4, mu_n=0.10, mu_p=0.002, tau=1e-6, eps_r=9.7,
           Eg=3.4, mid=0.5e-9)
EDGE_METAL_W = 1.5e-9          # devsim_layered._EDGE_METAL_W_M: the contacts sit at the strip interfaces


def _diode(g, flag=None, name="xpd"):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    dop = "ifelse(x < {:.10e}, {:.10e}, {:.10e})".format(g["P"] / 2, -g["NA"], g["ND"])
    reg.add(Material("semi", ConstantOptical(12.0 + 0j), transport=TransportModel(
        n_bg_m3=g["ND"], eps_static=g["eps_r"], dos_mass_kg_of_n_m3=_const(1.08 * M_E), band_gap_eV=g["Eg"],
        chi_eV=4.0, physics="bipolar_dd", mobility_m2Vs_of_n_m3=_const(g["mu_n"]),
        hole_mobility_m2Vs_of_n_m3=_const(g["mu_p"]), tau_srh_s=g["tau"], n_i_m3=g["n_i"],
        net_doping_expr=dop, dos_mass_p_kg=0.81 * M_E, carrier_statistics="boltzmann")))
    m2 = Mesh2DSpec(x_spacing_edge_m=20e-9, x_spacing_feature_edge_m=20e-9, x_spacing_feature_mid_m=g["mid"],
                    coarse_ps_m=5e-9, interface_ps_m=5e-9, interface_zone_m=0.0)
    return Design(name=name, unit_cell=UnitCell.square(g["P"]),
                  stack=Stack(layers=[Layer("pn", g["T"], "semi")], superstrate_material="air",
                              substrate_material="air"),
                  electrodes=[Electrode("anode", "pn", "x_lo", role="biased"),
                              Electrode("cathode", "pn", "x_hi", role="ground", fixed_voltage_V=0.0)],
                  materials=reg, mesh_2d=m2, carriers=CarrierSpec(extended_precision=flag))


def _anode_current(dev):
    """Terminal current into the anode [A/m, per unit depth]."""
    return sum(float(devsim.get_contact_current(device=dev, contact="anode", equation=e))
               for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))


def _sha(cf, dev):
    h = hashlib.sha256()
    for r in sorted(cf.regions):
        for nm, a in sorted(cf.regions[r].node_fields.items()):
            h.update(nm.encode()); h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(np.float64(_anode_current(dev)).tobytes())
    return h.hexdigest()


def _clean_process():
    from dynameta.carriers import eq_registry as _R
    for d in list(devsim.get_device_list()):
        _R.clear(d)
        devsim.delete_device(device=d)
    for m in list(devsim.get_mesh_list()):
        devsim.delete_mesh(mesh=m)


def _solver_flag():
    try:
        return devsim.get_parameter(name="extended_solver")
    except devsim.error:
        return None


@pytest.fixture(autouse=True)
def _clean():
    _clean_process()
    yield
    _clean_process()


def _solve_si(b, v=0.3):
    return b.solve(BiasPoint({"anode": v}, "v"))


# =====================================================================================================
# CarrierSpec / precedence (no solve)
# =====================================================================================================

def test_carrier_spec_is_appended_unstated_and_validated():
    """Design.carriers defaults to CarrierSpec(extended_precision=None) (unstated = off) and is the LAST
    field (positional Design(...) calls are unchanged); anything but True/False/None is refused."""
    d = _diode(SI)
    assert dataclasses.fields(Design)[-1].name == "carriers"
    assert Design(*(getattr(d, f.name) for f in dataclasses.fields(Design)[:-1])).carriers == CarrierSpec()
    assert CarrierSpec().extended_precision is None
    with pytest.raises(TypeError, match="True, False or None"):
        CarrierSpec(extended_precision=1)
    with pytest.raises(TypeError, match="must be a CarrierSpec"):
        dataclasses.replace(d, carriers={"extended_precision": True})


@pytest.mark.parametrize("stated,explicit,expect,warns", [
    (None, None, False, 0), (None, True, True, 0), (None, False, False, 0),
    (True, None, True, 0), (False, None, False, 0), (True, True, True, 0), (False, False, False, 0),
    (True, False, False, 1), (False, True, True, 1)])
def test_e5_precedence_explicit_wins_one_advisory_on_contradiction(stated, explicit, expect, warns):
    """GATE E5. The explicit builder argument wins; exactly ONE AdvisoryWarning when it contradicts a
    STATED design value, none otherwise (an unstated design keeps the v0.14 call silent). Same resolver
    on the 2-D builder and the 3-D Stacked3DSpec.from_design. Adversarial: dropping the advisory, or
    letting the design win, fails the (True, False) / (False, True) rows."""
    from dynameta.carriers import devsim_3d as D3
    from validation._reference_device import build_reference_modulator
    kw = {} if explicit is None else {"extended_precision": explicit}
    for build in (lambda: DL.LayeredDevsimBuilder(_diode(SI, stated), **kw).extended_precision,
                  lambda: D3.Stacked3DSpec.from_design(dataclasses.replace(
                      build_reference_modulator("equilibrium"),
                      carriers=CarrierSpec(extended_precision=stated)), **kw).extended_precision):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got = build()
        hits = [w for w in rec if issubclass(w.category, AdvisoryWarning) and OVERRIDE in str(w.message)]
        assert got is expect
        assert len(hits) == warns, [str(w.message)[:90] for w in rec]
        if warns:
            assert "extended_precision={!r}".format(explicit) in str(hits[0].message)


# =====================================================================================================
# E1 / E2: equivalence and defaults off (tiny Si diode)
# =====================================================================================================

@pytest.mark.devsim_isolated("bitwise comparison of separate DEVSIM solves: a device left live by another test is iterated by the same joint Newton and moves the answer by ulps")
def test_e2_design_flag_equals_explicit_argument_bitwise_and_off_is_default(monkeypatch):
    """GATE E2 + E1 (same tree). sha256 of every node field + the anode current at +0.3 V:
    design-flag (CarrierSpec(True)) == explicit LayeredDevsimBuilder(design, extended_precision=True), bit
    for bit; the default design == CarrierSpec(False) == explicit False, bit for bit, and on that path no
    extended_* parameter is ever written (spy on devsim.set_parameter) and CarrierField.extras has no
    'extended_precision' key. Load-bearing: the extended answer differs from the double one (anode current
    7.333490e-10 vs 7.333647e-10 A/m, measured)."""
    shas, keys = {}, {}
    for tag, design, kw in (("flag", _diode(SI, True), {}), ("explicit", _diode(SI), {"extended_precision": True}),
                            ("default", _diode(SI), {}), ("stated_off", _diode(SI, False), {}),
                            ("explicit_off", _diode(SI), {"extended_precision": False})):
        writes = []
        real = devsim.set_parameter

        def spy(*a, **k):
            writes.append(str(k.get("name")))
            return real(*a, **k)
        monkeypatch.setattr(devsim, "set_parameter", spy)
        b = DL.LayeredDevsimBuilder(design, mesh_name="e2m", device_name="e2d", **kw)
        cf = _solve_si(b)
        shas[tag], keys[tag] = _sha(cf, "e2d"), cf.extras.get("extended_precision")
        monkeypatch.setattr(devsim, "set_parameter", real)
        b.teardown()
        ext_writes = [w for w in writes if w.startswith("extended")]
        assert bool(ext_writes) is (tag in ("flag", "explicit")), (tag, ext_writes)
    assert shas["flag"] == shas["explicit"]
    assert shas["default"] == shas["stated_off"] == shas["explicit_off"]
    assert shas["flag"] != shas["default"]
    assert keys == {"flag": True, "explicit": True, "default": None, "stated_off": None, "explicit_off": None}
    assert not XP.any_enabled() and not _solver_flag()


# =====================================================================================================
# E3: run_pipeline reaches it, enables exactly once; adversarial: nothing leaks into the next run
# =====================================================================================================

class _StubGeo:
    class mesh:
        ne = 0
        nv = 0


class _StubOptics:
    """No ngsolve: alignment() / mesh_regions() for the semiconductor layer 'pn' only."""
    def __init__(self, g):
        self.g = g

    def build(self):
        return _StubGeo()

    def alignment(self):
        from dynameta.core import NM
        from dynameta.core.alignment import GeometryAlignment, RegionAlignment
        P, T = self.g["P"], self.g["T"]
        return GeometryAlignment(unit_scale=NM, region_alignments=[
            RegionAlignment("pn", "pn", (0.0, P, 0.0, P, 0.0, T), stack_axis="y")], fixed_eps_regions={})

    def mesh_regions(self):
        return ["pn"]


def _optics(design, geo, eps_by_region, lambda_m, n_super, n_sub):
    from dynameta.core.interfaces import OpticalResult
    return OpticalResult(r=0j, R=0.0, phase_deg=0.0, solve_time_s=0.0)


@pytest.mark.devsim_isolated("bitwise comparison of separate DEVSIM solves: a device left live by another test is iterated by the same joint Newton and moves the answer by ulps")
def test_e3_run_pipeline_enables_once_and_leaks_nothing(monkeypatch):
    """GATE E3 + adversarial. run_pipeline(carrier_solver=None) on a design with CarrierSpec(True) over TWO
    bias points calls XP.enable exactly ONCE (spy wrapping the real call: one device build serves the
    sweep); a default design never calls it. After each bias solve (spy on LayeredDevsimBuilder.solve)
    the global extended_solver is off, and after the run no device is opted in. A default run AFTER the
    extended one is bitwise equal (sha256 of node fields + anode current) to a default run BEFORE it.
    On the old code the flag cannot be set (TypeError) and, with the builder ignoring the design, the
    spy would see 0 calls."""
    from dynameta.pipeline import run_pipeline
    from dynameta.sweep import Sweep
    enables, after, shas = [], [], []
    real_enable, real_solve = XP.enable, DL.LayeredDevsimBuilder.solve

    def spy_enable(device):
        enables.append(device)
        return real_enable(device)

    def spy_solve(self, bias, **kw):
        cf = real_solve(self, bias, **kw)
        after.append(_solver_flag())
        shas.append((self.design.name, bias.label, _sha(cf, self.device)))
        return cf

    monkeypatch.setattr(XP, "enable", spy_enable)
    monkeypatch.setattr(DL.LayeredDevsimBuilder, "solve", spy_solve)
    sweep = Sweep(bias_points=[BiasPoint({"anode": 0.0}, "v0"), BiasPoint({"anode": 0.3}, "v3")],
                  wavelengths_nm=[1550.0])
    counts = []
    for name, flag in (("before", None), ("ext", True), ("after", None)):
        n0 = len(enables)
        run_pipeline(_diode(SI, flag, name=name), sweep, verbose=False, optical_builder=_StubOptics(SI),
                     optical_solver=_optics)
        counts.append(len(enables) - n0)
        assert not XP.any_enabled() and not devsim.get_device_list()
    assert counts == [0, 1, 0]
    assert all(not f for f in after)
    by = {(d, lab): s for d, lab, s in shas}
    assert by[("before", "v3")] == by[("after", "v3")] and by[("before", "v0")] == by[("after", "v0")]
    assert by[("ext", "v3")] != by[("before", "v3")]


# =====================================================================================================
# process-global DEVSIM: two designs that disagree, alive in one process
# =====================================================================================================

def test_two_designs_that_disagree_in_one_process_warn_and_teardown_is_global():
    """DEVSIM's extended_solver is global and every live device is solved in ONE Newton. A default design's
    builder solved while an extended design's device is alive is solved by the extended Newton -- every
    such solve raises the mixed-devices AdvisoryWarning naming both (documented in CarrierSpec); and
    LayeredDevsimBuilder.teardown() deletes EVERY device in the process (the other builder's too)."""
    a = DL.LayeredDevsimBuilder(_diode(SI, True, name="a"), mesh_name="am", device_name="ad")
    _solve_si(a, 0.0)
    b = DL.LayeredDevsimBuilder(_diode(SI, name="b"), mesh_name="bm", device_name="bd")
    with pytest.warns(AdvisoryWarning, match=MIXED) as rec:
        _solve_si(b, 0.0)
    assert any("'bd'" in str(w.message) and "'ad'" in str(w.message) for w in rec)
    b.teardown()
    assert not devsim.get_device_list() and not XP.any_enabled()


# =====================================================================================================
# E6: the 3-D builder reads the design flag (build-time only; a 3-D extended solve is minutes long)
# =====================================================================================================

def test_e6_stacked3d_from_design_follows_the_design():
    """GATE E6 (spec half; the build-time enable of a True spec is gated by test_extended_precision.py::
    test_stacked3d_spec_switch). from_design(design with CarrierSpec(True)) gives extended_precision True,
    the default design False, CarrierSpec(False) False."""
    from dynameta.carriers import devsim_3d as D3
    from validation._reference_device import build_reference_modulator
    d = build_reference_modulator("equilibrium")
    for flag, expect in ((True, True), (False, False), (None, False)):
        spec = D3.Stacked3DSpec.from_design(dataclasses.replace(d, carriers=CarrierSpec(extended_precision=flag)))
        assert spec.extended_precision is expect


# =====================================================================================================
# E4 (nightly): GaN-like 2-D diode from the design flag vs Sah-Noyce-Shockley
# =====================================================================================================

def _sns_oracle_A_per_m2(V, g):
    """Abrupt p-n junction current density [A/m^2] (re-derived here; scipy only): Shockley diffusion with
    finite neutral widths W' = W - x_dep, J0 = q n_i^2 [D_n/(N_A L_n) coth(W_p'/L_n) + D_p/(N_D L_p)
    coth(W_n'/L_p)], plus the Sah-Noyce-Shockley depletion-region SRH current q int U dx with
    U = n_i^2 (e^{V/V_T} - 1) / (tau (n + p + 2 n_i)) (n1 = p1 = n_i, flat quasi-Fermi levels) on the
    depletion-approximation potential (parabolic, 2 V_T Debye correction)."""
    NA, ND, ni, tau = g["NA"], g["ND"], g["n_i"], g["tau"]
    eps = g["eps_r"] * EPS0
    W = g["P"] / 2 - EDGE_METAL_W                                  # junction to each contact
    Vbi = VT * math.log(NA * ND / ni ** 2)
    xd = math.sqrt(2 * eps * (Vbi - V - 2 * VT) * (NA + ND) / (QE * NA * ND))
    xp, xn = xd * ND / (NA + ND), xd * NA / (NA + ND)
    Dn, Dp = g["mu_n"] * VT, g["mu_p"] * VT
    Ln, Lp = math.sqrt(Dn * tau), math.sqrt(Dp * tau)
    em1 = math.expm1(V / VT)
    j_diff = QE * ni ** 2 * em1 * (Dn / (NA * Ln) / math.tanh((W - xp) / Ln) + Dp / (ND * Lp) / math.tanh((W - xn) / Lp))
    psi_p, psi_n = V - VT * math.log(NA / ni), VT * math.log(ND / ni)

    def psi(x):
        return psi_p + VT + QE * NA / (2 * eps) * (x + xp) ** 2 if x <= 0 else psi_n - VT - QE * ND / (2 * eps) * (xn - x) ** 2

    def U(x):
        u = psi(x)
        return ni ** 2 * em1 / (tau * (ni * math.exp(u / VT) + ni * math.exp((V - u) / VT) + 2 * ni))
    xm = brentq(lambda x: psi(x) - V / 2, -xp, xn)
    srh = quad(U, -xp, xm, limit=500, epsabs=0, epsrel=1e-12)[0] + quad(U, xm, xn, limit=500, epsabs=0, epsrel=1e-12)[0]
    return j_diff + QE * srh


def _gan_1d_miss(xs, V=1.0):
    """The GaN-like diode as a 1-D extended-precision device on the node set xs (contacts at xs[0], xs[-1],
    junction at P/2), ramped to V: (anode current / oracle) - 1, the oracle of this file (W = P/2 - 1.5 nm)."""
    import validation.extended_precision as VX
    g = GAN
    lines = [(float(x), 1.0, "lo" if k == 0 else "hi" if k == xs.size - 1 else None) for k, x in enumerate(xs)]
    dop = "ifelse(x < {:.10e}, {:.10e}, {:.10e})".format(g["P"] / 2, -g["NA"], g["ND"])
    dev = VX.diode_1d("e4c1", lines, dop, n_i=g["n_i"], tau=g["tau"], eps_r=g["eps_r"], mu_n=g["mu_n"],
                      mu_p=g["mu_p"], extended=True, ae=1e8, re=1e-12, presolve=False)
    try:
        for k in range(1, 11):
            devsim.set_parameter(device=dev, name="lo_bias", value=V * k / 10)
            VX._solve(dev, 1e8, 1e-12, 60)
        J = float(VX.edge_current(dev).mean())
    finally:
        VX.fresh()
    return J / _sns_oracle_A_per_m2(V, g) - 1.0


def _junction_band(h, h_odd, band=20e-9, grow=1.02, h_max=20e-9):
    """Explicit nodes: uniform h over [P/2 - band, P/2 + band] with the junction a node, EXCEPT the one edge
    ending at the junction from the p side, of length h_odd; graded (x grow, capped) out to the contacts."""
    xj, lo, hi = GAN["P"] / 2, EDGE_METAL_W, GAN["P"] - EDGE_METAL_W
    m = int(round(band / h))
    right, left = [xj + h * k for k in range(m + 1)], [xj - h_odd - h * k for k in range(m)]

    def out(x0, sgn, stop):
        o, s_ = [], h
        while True:
            s_ = min(s_ * grow, h_max)
            nx = (o[-1] if o else x0) + sgn * s_
            if (nx <= stop + 0.5 * s_) if sgn < 0 else (nx >= stop - 0.5 * s_):
                return o + [stop]
            o.append(nx)
    return np.array(sorted(out(left[-1], -1.0, lo) + left + right + out(right[-1], 1.0, hi)))


def test_e4_mesh_cause_is_the_remainder_edge_on_the_recombination_peak():
    """WHY E4 needs 0.5 nm where the 1-D gate meets 3e-3 at 1 nm (v0.15 follow-up, audit note
    docs/audit/2026-11-06-v0.15-followups.md item 3). A characterisation gate: no code changed, so it holds on
    the old code too. At 1.0 V the SRH current is the whole current (J_srh/J = 1 - 3.6e-13) and its integrand,
    U ~ 1/(cosh((psi - V/2)/V_t) + 1), is a peak of width V_t/E = 1.29 nm at the junction (E = 2.0e7 V/m).
      (a) DEVSIM's line grading (20 nm at the contact line -> 1 nm at the mid line) leaves a REMAINDER edge
          ending at the junction: 0.7716 nm on the builder's x lines (the 1-D gate's lines leave 0.0099 nm).
      (b) The 1-D device on the builder's OWN x-node set misses by > 3e-3 (MEASURED +4.944e-3, the 2-D
          builder's +4.9e-3; at 0.5 nm +8.394e-4 vs the 2-D +8.39e-4): it is the node set, not the 2-D
          element physics or the 2-D mesher dropping lines.
      (c) A clean uniform 1 nm band through the junction meets the 3e-3 budget (MEASURED -6.39e-4; 0.5 nm
          -1.64e-4, 0.25 nm -5.2e-5: second order -- so the audit's "residual ~4e-4 is the oracle's depletion
          approximation" was the mesh too).
      (d) ADVERSARIAL: that clean band with ONE 0.5 nm edge ending at the junction misses by > 3e-3 (MEASURED
          +4.62e-3; 0.77 nm +3.49e-3, 0.2 nm +2.47e-3, 0.01 nm +1.3e-4): the remainder edge alone.
    Budget 3e-3: the D2b gate-A tolerance E4 inherits."""
    b = DL.LayeredDevsimBuilder(_diode(dict(GAN, mid=1e-9), True, name="gan"), mesh_name="e4cm",
                                device_name="e4cd")
    try:
        b.build_device()
        xs = np.unique(np.round(np.asarray(devsim.get_node_model_values(device="e4cd", region="pn", name="x")),
                                15))
    finally:
        b.teardown()
    i = int(np.argmin(np.abs(xs - GAN["P"] / 2)))
    assert xs[i] == pytest.approx(GAN["P"] / 2, abs=1e-15)
    assert abs((xs[i] - xs[i - 1]) - 0.7716e-9) < 1e-13 and abs((xs[i + 1] - xs[i]) - 1e-9) < 1e-13     # (a)
    assert _gan_1d_miss(xs) > 3e-3                                                                        # (b)
    assert abs(_gan_1d_miss(_junction_band(1e-9, 1e-9))) <= 3e-3                                          # (c)
    assert _gan_1d_miss(_junction_band(1e-9, 0.5e-9)) > 3e-3                                              # (d)


@pytest.mark.slow
def test_e4_gan_like_diode_from_the_design_flag_matches_sns_and_double_does_not():
    """GATE E4 (nightly, ~2 min on box A: the extended ramps to 1.0 and 1.5 V are ~35 + ~47 s, the double
    twin ~15 s). The GaN-like lateral 2-D diode (n_i 1e-4 m^-3, N_A = N_D = 1e23 m^-3, 1131 nodes) built
    through LayeredDevsimBuilder from Design.carriers = CarrierSpec(True): the anode current per unit depth
    must equal the oracle J x layer thickness to 3e-3 at +1.0 and +1.5 V (the D2b gate-A budget; MEASURED
    +8.4e-4 / +6.9e-4). The SAME design with the flag unstated (double) must be off by more than a factor
    1e3 (or have the wrong sign): MEASURED 2.9e-38 vs 8.2e-26 A/m at 1.0 V (12 decades low -- the double
    current is round-off) and 4.4e-17 vs 1.5e-21 A/m at 1.5 V (3e4x high) -- so the flag, not the mesh,
    carries the answer."""
    oracle = {V: _sns_oracle_A_per_m2(V, GAN) * GAN["T"] for V in (1.0, 1.5)}
    got = {}
    for flag in (True, None):
        b = DL.LayeredDevsimBuilder(_diode(GAN, flag, name="gan"), mesh_name="e4m", device_name="e4d")
        try:
            for V in (1.0, 1.5):
                cf = b.solve(BiasPoint({"anode": V}, "v"), rel_tol=1e-12, abs_tol=1e8)
                got[(flag, V)] = _anode_current("e4d")
                assert cf.extras.get("extended_precision") is (True if flag else None)
        finally:
            b.teardown()
    for V in (1.0, 1.5):
        assert got[(True, V)] / oracle[V] - 1.0 == pytest.approx(0.0, abs=3e-3), (V, got[(True, V)], oracle[V])
        r = got[(None, V)] / oracle[V]                   # "off by > 1e3": a FACTOR (or the wrong sign)
        assert r <= 0.0 or abs(math.log10(r)) > 3.0, (V, got[(None, V)], oracle[V])
