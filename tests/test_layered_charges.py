"""v0.16 carrier wiring through LayeredDevsimBuilder's public inputs:
fixed sheet charge (fixed_charge=), dielectric bulk charge (bulk_charge=), gate work functions
(Electrode.work_function_eV), equilibrium-film interface traps (interface_traps={...: FilmInterfaceTraps}),
and the DOS-mass guard (TransportModel.dos_mass_at_bg) at the builder's call sites.

Device: tests/_wiring_fixture.py -- the generic gated-film cell, a laterally uniform 6 nm degenerate film (n_bg 3e26
m^-3) between two oxide-insulated metal gates; the top dielectric is 6 nm oxB (eps 20) + 2 nm oxA (eps 9), series
capacitance C_top = eps0 / (6 nm/20 + 2 nm/9) = 1.695e-2 F/m^2. Solved with abs_tol = 1e-14 V.

ORACLES (closed forms re-derived here; z up, towards the top gate). The top dielectric holds no charge of its
own, so in it D is uniform and the potential piecewise linear -- the box method reproduces that exactly. Gauss
at the film's top face: D_ox - D_film = Q (a sheet Q there) and D_ox = C_top (psi_s - V_top), so a solution
with the sheet at V_top - Q/C_top has the same film profile as the one without it at V_top (gate shift
-Q/C_top). A uniform rho in the oxB layer (thickness t1, eps1; then t2, eps2 to the gate) raises D by rho (z - z_f)
inside it, so the gate shift is -rho t1 (t1/(2 eps1) + t2/eps2)/eps0 (the charge acts at its centroid). A
gate of work function phi_m pins Potential = V - (phi_m - W_s)/q, W_s = chi + (E_c - E_F) at psi = 0: at
V_fb = (phi_m - W_s)/q on both gates the film is exactly flat. Interface traps of density D_it add
C_it = q D_it (f(e_lo) - f(e_hi)) at the film face: d psi_s/dV_top = C_top/(C_top + C_s + C_it), C_s the
film's own response measured without traps on the same mesh.

On the library before this change every gate here fails: LayeredDevsimBuilder has no
fixed_charge / bulk_charge (TypeError), Electrode.work_function_eV is ignored (the flat-band and bias-shift
gates fail by the full offset, ~1 V), the builder reads the DOS mass without the guard (no warning), and
InterfaceTraps(fixed_charge_C_m2=-1.6e-3) on the film interface was refused before the equilibrium charge
primitives existed (ValueError). FilmInterfaceTraps reach the builder since then; the efficiency gate is new here.
"""
import math
import warnings

import numpy as np
import pytest
from scipy import integrate, optimize

ds = pytest.importorskip("devsim")
from dynameta.carriers.devsim_env import import_devsim     # noqa: E402

ds = import_devsim()

from dynameta.carriers import physics_equilibrium as PE                 # noqa: E402
from dynameta.carriers import physics_traps as PT                       # noqa: E402
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder        # noqa: E402
from dynameta.constants import EPS0, KB, M_E, Q_E, T_REF, V_T            # noqa: E402
from dynameta.sweep import BiasPoint                                     # noqa: E402
from dynameta.warning_types import AdvisoryWarning                       # noqa: E402

import _wiring_fixture as W                                              # noqa: E402

ABS_TOL = 1e-14          # volts


@pytest.fixture(autouse=True)
def _clean():
    yield
    from dynameta.carriers import eq_registry as _R
    for dv in list(ds.get_device_list()):
        _R.clear(dv)
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)


_COUNT = [0]


def _builder(design, **kw):
    _COUNT[0] += 1
    k = _COUNT[0]
    return LayeredDevsimBuilder(design, mesh_name="m{}".format(k), device_name="d{}".format(k), **kw)


def _solve(design, voltages, **kw):
    """Solve, snapshot, tear down (W.Snapshot: DEVSIM solves every live device jointly)."""
    b = _builder(design, **kw)
    b.solve(BiasPoint(voltages, "b"), abs_tol=ABS_TOL)
    s = W.Snapshot(ds, b)
    b.teardown()
    return s


def _film(s, name="Electrons"):
    return s.cols("film", name)


def _rel(a, b):
    return max(float(np.max(np.abs(a[x] / b[x] - 1.0))) for x in b)


def _abs(a, b, shift=0.0):
    return max(float(np.max(np.abs(a[x] - b[x] - shift))) for x in b)


def _uniform(cols):
    xs = sorted(cols)
    ref = cols[xs[len(xs) // 2]]
    return max(float(np.max(np.abs(cols[x] / ref - 1.0))) for x in xs)


# ------------------------------------------------------------------------------------------------
# fixed sheet charge
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q, key", [(2.0e-3, ("film", "up_oxb")), (-2.0e-3, ("up_oxb", "film")),
                                    (2.0e-3, "film__up_oxb")])
def test_sheet_charge_shifts_the_top_gate_by_minus_q_over_c(q, key):
    """Sheet Q at the film/top-oxide interface: the film profile at V_top - Q/C_top equals the charge-free
    one at V_top (MEASURED 6.7e-16 relative in n, 1.1e-16 V; bar 1e-10 / 1e-12 V), laterally uniform, and
    the sign is physical (at equal V_top, Q = +-2e-3 C/m^2 moves n at the top face by +2.9 % / -2.9 %). Key forms: either tuple order
    or the interface name."""
    vt = 2.0
    b0 = _solve(W.film_design(), {"top": vt, "bot": 0.0})
    n0, p0 = _film(b0), _film(b0, "Potential")
    b1 = _solve(W.film_design(), {"top": vt - q / W.C_TOP, "bot": 0.0}, fixed_charge={key: q})
    n1, p1 = _film(b1), _film(b1, "Potential")
    assert _rel(n1, n0) < 1e-10 and _abs(p1, p0) < 1e-12
    assert _uniform(n1) < 1e-12
    b2 = _solve(W.film_design(), {"top": vt, "bot": 0.0}, fixed_charge={key: q})
    assert np.sign(b2.top(name="Electrons") - b0.top(name="Electrons")) == np.sign(q)


def test_fixed_charge_only_interface_traps_build_and_match_fixed_charge():
    """The call InterfaceTraps(charge='none', recombination=False, fixed_charge_C_m2=-1.6e-3) on
    the equilibrium film's interface -- builds (it is routed to the sheet charge, on the film side) and gives
    the fixed_charge= solution (installed on the oxide side) to the solver tolerance (MEASURED 6.7e-16)."""
    q = -1.6e-3
    spec = PT.InterfaceTraps(charge="none", recombination=False, fixed_charge_C_m2=q)
    b1 = _solve(W.film_design(), {"top": 1.0, "bot": 0.0}, interface_traps={("film", "up_oxb"): spec})
    b2 = _solve(W.film_design(), {"top": 1.0, "bot": 0.0}, fixed_charge={("film", "up_oxb"): q})
    assert _rel(_film(b1), _film(b2)) < 1e-10


def test_bulk_oxide_charge_shifts_the_top_gate_by_the_centroid_formula():
    """Uniform rho in the 6 nm oxB next to the film (then 2 nm oxA to the gate): the gate shift is
    -rho t1 (t1/(2 eps1) + t2/eps2)/eps0 = -0.1009 V at rho = 4e5 C/m^3 (MEASURED match 6.7e-16 in n,
    5.6e-17 V)."""
    rho = 4.0e5
    (t1, e1), (t2, e2) = W.TOP_STACK
    dv = -rho * t1 * (t1 / (2.0 * e1) + t2 / e2) / EPS0
    b0 = _solve(W.film_design(), {"top": 1.5, "bot": 0.0})
    b1 = _solve(W.film_design(), {"top": 1.5 + dv, "bot": 0.0}, bulk_charge={"up_oxb": rho})
    assert _rel(_film(b1), _film(b0)) < 1e-10
    assert _abs(_film(b1, "Potential"), _film(b0, "Potential")) < 1e-12


def test_charge_inputs_are_checked():
    with pytest.raises(TypeError, match="mapping"):
        _builder(W.film_design(), fixed_charge=[("film", "up_oxb", 1e-3)])
    with pytest.raises(ValueError, match="finite"):
        _builder(W.film_design(), fixed_charge={("film", "up_oxb"): float("nan")})
    with pytest.raises(TypeError, match="2-tuple"):
        _builder(W.film_design(), bulk_charge={("a", "b", "c"): 1.0})
    with pytest.raises(ValueError, match="matches no interface"):
        _builder(W.film_design(), fixed_charge={("up_oxa", "gate"): 1e-3}).build_device()
    with pytest.raises(ValueError, match="DIELECTRIC regions only"):
        _builder(W.film_design(), bulk_charge={"film": 1e5}).build_device()
    with pytest.raises(ValueError, match="not a region"):
        _builder(W.film_design(), bulk_charge={"nope": 1e5}).build_device()


# ------------------------------------------------------------------------------------------------
# gate work function
# ------------------------------------------------------------------------------------------------
def _exact_ec_minus_ef_eV():
    """E_c - E_F [eV] of the neutral film from an independent F_1/2 (scipy quadrature)."""
    nc = 2.0 * (W.M_DOS_REL * M_E * KB * T_REF / (2.0 * math.pi * (1.054571817e-34) ** 2)) ** 1.5

    def f12(eta):
        v, _ = integrate.quad(lambda e: math.sqrt(e) / (1.0 + math.exp(e - eta)), 0.0, max(eta, 0.0) + 60.0,
                              limit=400, points=[max(eta, 0.0)])
        return 2.0 / math.sqrt(math.pi) * v
    eta = optimize.brentq(lambda e: f12(e) - W.N_BG / nc, -10.0, 60.0, xtol=1e-14)
    return -eta * KB * T_REF / Q_E


def test_flat_band_voltage_equals_the_work_function_difference():
    """Both gates of work function phi_m = 4.2 eV, biased at V_fb = phi_m - W_s (W_s = chi + q Phi_c of the
    film, from band_frames()): the film is flat -- n = n_bg and psi = 0 at every node (MEASURED 1.1e-11 /
    2.7e-11 V: the floor of Phi_c0's own calibration, invert_F12 rtol 1e-10). V_fb = +0.2442 V here. Phi_c agrees
    with an independent Fermi-Dirac integral to the Aymerich-Humet F_1/2 accuracy the closure uses (MEASURED
    0.49 meV at eta = 21.0; bar 1 mV). Sign: at 0 V a metal of higher work function
    than W_s depletes the film face, a lower one accumulates it."""
    phi_m = 4.2
    d = W.film_design(wf_top=phi_m, wf_bot=phi_m)
    b = _builder(d)
    b.build_device()
    phi_c = b.band_frames()["film"][0]
    b.teardown()
    assert abs(phi_c - _exact_ec_minus_ef_eV()) < 1e-3
    w_s = W.CHI_EV + phi_c
    v_fb = phi_m - w_s
    s = _solve(d, {"top": v_fb, "bot": v_fb})
    assert s.wf_offset == {"top": -(phi_m - w_s), "bot": -(phi_m - w_s)}
    n, p = _film(s), _film(s, "Potential")
    assert max(float(np.max(np.abs(n[x] / W.N_BG - 1.0))) for x in n) < 1e-9
    assert max(float(np.max(np.abs(p[x]))) for x in p) < 1e-10
    for phi, sgn in ((w_s + 0.3, -1.0), (w_s - 0.3, +1.0)):
        bb = _solve(W.film_design(wf_top=phi), {"top": 0.0, "bot": 0.0})
        assert np.sign(bb.top(name="Electrons") - W.N_BG) == sgn


@pytest.mark.devsim_isolated("bitwise comparison of two DEVSIM solves")
def test_work_function_equals_a_bias_shift_bit_for_bit():
    """A top gate of work function phi_m at bias V gives node arrays bit-identical to the same device without
    a work function at V - (phi_m - W_s)/q: the offset is added to the bias value and the ramp runs in the
    applied potential (seed at 0, the same steps)."""
    phi_m, v = 5.1, 1.0
    b1 = _solve(W.film_design(wf_top=phi_m), {"top": v, "bot": 0.0})
    off = b1.wf_offset["top"]
    assert off == -(phi_m - (W.CHI_EV + b1.frames["film"][0]))
    b0 = _solve(W.film_design(), {"top": v + off, "bot": 0.0})
    assert sorted(b1.raw) == sorted(b0.raw) and len(b1.raw) >= 14
    for key in b1.raw:
        assert np.array_equal(b1.raw[key], b0.raw[key]), key


def test_work_function_refusals():
    """No chi_eV on the frame reference -> ValueError naming chi_eV; a work function on a metal layer whose
    contact lands on the semiconductor (a Schottky contact) -> NotImplementedError."""
    with pytest.raises(ValueError, match="chi_eV"):
        _builder(W.film_design(wf_top=4.5, mats=W.materials(chi_eV=None))).build_device()
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    layers = [Layer("mirror", 80e-9, "mirror_metal"), Layer("lo_oxb", 6e-9, "oxB"), Layer("film", 6e-9, "TCO"),
              Layer("gate", 40e-9, "metal")]
    d = Design(name="schottky", unit_cell=UnitCell.square(40e-9),
               stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
               electrodes=[Electrode("bot", "mirror", "full"), Electrode("top", "gate", "full",
                                                                         work_function_eV=4.8)],
               materials=W.materials(), mesh_2d=Mesh2DSpec(**W.X_LINES))
    with pytest.raises(NotImplementedError, match="Schottky"):
        _builder(d).build_device()


# ------------------------------------------------------------------------------------------------
# equilibrium-film interface traps through the builder
# ------------------------------------------------------------------------------------------------
D_IT = 5.0e16            # eV^-1 m^-2 (5e12 cm^-2 eV^-1)
HALF_BAND = 0.6          # eV either side of the neutral level


def _traps():
    """Split band centred on the film's Fermi level at flat band (E_F - E_c = -Phi_c0), so at 0 V the
    trapped charge vanishes and the operating point is the trap-free one."""
    nc = PE.effective_dos_m3(W.M_DOS_REL * M_E)
    e0 = PE.invert_F12(W.N_BG / nc) * V_T                   # = -Phi_c0, the builder's own closure value
    return PT.FilmInterfaceTraps(dit_eV_m2=D_IT, e_lo_eV=e0 - HALF_BAND, e_hi_eV=e0 + HALF_BAND,
                                 neutral_level_eV=e0, charge="split")


def _efficiency(design, dv=1e-4, **kw):
    b = _builder(design, **kw)
    psi = []
    for v in (dv, -dv):
        b.solve(BiasPoint({"top": v, "bot": 0.0}, "s"), abs_tol=ABS_TOL)
        s = W.Snapshot(ds, b)
        psi.append(s.top())
        assert _uniform(_film(s)) < 1e-12
    b.teardown()
    return (psi[0] - psi[1]) / (2.0 * dv)


def test_film_traps_small_signal_gate_efficiency():
    """d psi_s/dV_top at flat band equals C_top/(C_top + C_s + C_it) with C_s from the trap-free efficiency
    on the same mesh and C_it = q D_it (f(e_lo) - f(e_hi)) (MEASURED agreement 3.3e-12 at dV = 1e-4 V, 2.1e-10
    at 1e-3 V -- the O(dV^2) term; bar 1e-8, which also covers the 1e-14 V solve on a 1.3e-5 V swing).
    C_top 1.695e-2, C_s 1.026e-1, C_it 8.01e-3 F/m^2. The traps lower the efficiency by 6.3 % (0.1418 -> 0.1329)."""
    eff0 = _efficiency(W.film_design())
    c_s = W.C_TOP * (1.0 / eff0 - 1.0)
    f_lo = 1.0 / (1.0 + math.exp(-HALF_BAND / V_T))
    c_it = Q_E * D_IT * (f_lo - (1.0 - f_lo))
    want = W.C_TOP / (W.C_TOP + c_s + c_it)
    got = _efficiency(W.film_design(), interface_traps={("film", "up_oxb"): _traps()})
    assert abs(got / want - 1.0) < 1e-8
    assert got < 0.97 * eff0


def test_film_traps_follow_the_edge_reference():
    """Traps occupy by the film's own Fermi level, so with an edge reference at 0.3 V the traps device at
    (V_top, V_bot) equals the unreferenced traps device at gates shifted by -0.3 V (MEASURED 7.8e-16 in n,
    5.6e-17 V)."""
    v = 0.3
    traps = {("film", "up_oxb"): _traps()}
    b1 = _solve(W.film_design(edges=[W.edge(v=v)]), {"top": 1.0, "bot": 0.0}, interface_traps=traps)
    b0 = _solve(W.film_design(), {"top": 1.0 - v, "bot": -v}, interface_traps=traps)
    assert _rel(_film(b1), _film(b0)) < 1e-12
    assert _abs(_film(b1, "Potential"), _film(b0, "Potential"), v) < 1e-12


# ------------------------------------------------------------------------------------------------
# the DOS-mass-at-n_bg guard at the builder's call sites
# ------------------------------------------------------------------------------------------------
def test_builder_dos_mass_guard():
    """A density-dependent DOS mass warns ONCE (AdvisoryWarning naming the builder region) when the builder
    reads it; the region's N_c is the value at n_bg either way (unchanged float). A constant mass is silent."""
    def varying(n):
        return W.M_DOS_REL * M_E * (1.0 + 0.1 * np.log(np.asarray(n, float) / W.N_BG))
    d = W.film_design(mats=W.materials(dos_mass_fn=varying))
    b = _builder(d)
    with pytest.warns(AdvisoryWarning, match="LayeredDevsimBuilder region 'film'") as rec:
        b.build_device()
    assert sum("density-dependent" in str(w.message) for w in rec) == 1
    nc = float(ds.get_parameter(device=b.device, region="film", name="N_c"))
    assert nc == PE.effective_dos_m3(float(varying(W.N_BG)), T_REF)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _builder(W.film_design()).build_device()
