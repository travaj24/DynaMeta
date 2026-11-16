"""v0.16 carrier physics primitives: Fermi-level reference, sheet / bulk fixed charge and gate work function,
the DOS-mass-at-n_bg guard and interface traps for EQUILIBRIUM films.

Every DEVSIM gate runs on the 1-D MOS of tests/_equilibrium_mos.py (gate | 10 nm oxide | 200 nm
equilibrium semiconductor, zero-field far end), built directly from the physics_equilibrium primitives.

On v0.15.1 (2e1af87) every gate below fails: physics_equilibrium has no install_sheet_charge /
install_bulk_charge / work_function_offset_V / fermi_ref_param, physics_traps no FilmInterfaceTraps,
TransportModel no dos_mass_at_bg, Electrode no work_function_eV (AttributeError / TypeError), and
install_interface_traps(InterfaceTraps(fixed_charge_C_m2=...)) on a non-bipolar interface raises
ValueError. The "unchanged" gates pin v0.15.1 values: the model-string digest below was
recorded on 2e1af87.
"""
import math
import warnings

import numpy as np
import pytest
from scipy import integrate

from dynameta.carriers import physics_equilibrium as PE
from dynameta.carriers import physics_traps as PT
from dynameta.constants import EPS0, M_E, Q_E, T_REF, V_T

import _equilibrium_mos as MOS


# ------------------------------------------------------------------------------------------------
# devsim-free
# ------------------------------------------------------------------------------------------------
def test_work_function_offset_closed_form_and_sign():
    """-(phi_m - W_s), W_s = chi + (E_c - E_F). A larger work function gives a more negative gate
    potential (acts like a negative bias)."""
    assert PE.work_function_offset_V(4.8, 4.5, 0.1) == -(4.8 - (4.5 + 0.1))
    assert PE.work_function_offset_V(5.2, 4.5, -0.3) < PE.work_function_offset_V(4.6, 4.5, -0.3)
    assert PE.work_function_offset_V(4.2, 4.0, 0.2) == 0.0                    # W_s = phi_m: flat band
    with pytest.raises(ValueError, match="chi_eV"):
        PE.work_function_offset_V(4.8, None, 0.1)
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError):
            PE.work_function_offset_V(bad, 4.5, 0.1)
    with pytest.raises(ValueError):
        PE.work_function_offset_V(4.8, 4.5, float("nan"))


def test_electrode_work_function_and_pin_potential_fields():
    from dynameta.geometry import Electrode
    e = Electrode("g", "metal", "full")
    assert e.work_function_eV is None and e.pin_potential is False           # defaults: v0.15.1
    assert Electrode("g", "metal", "full", work_function_eV=4.8).work_function_eV == 4.8
    assert Electrode("gnd", "film", "x_lo", role="ground", pin_potential=True).pin_potential
    for bad in (float("nan"), float("inf"), 0.0, -4.0, True, "4.8"):
        with pytest.raises(ValueError):
            Electrode("g", "metal", "full", work_function_eV=bad)
    with pytest.raises(ValueError, match="edge electrodes"):
        Electrode("g", "metal", "full", pin_potential=True)
    with pytest.raises(ValueError, match="ohmic"):
        Electrode("gnd", "film", "x_lo", role="ground", work_function_eV=4.8)
    with pytest.raises(TypeError):
        Electrode("gnd", "film", "x_lo", role="ground", pin_potential=1)


def _tm(fn, n_bg=3e26):
    from dynameta.materials import TransportModel
    return TransportModel(n_bg_m3=n_bg, eps_static=9.0, dos_mass_kg_of_n_m3=fn)


def test_dos_mass_at_bg_warns_for_a_density_dependent_mass_once_per_instance():
    """DOS-mass-at-n_bg guard: m0 (1 + 0.1 ln(n/n_bg)) varies by 46 % over [0.1, 10] n_bg -> one AdvisoryWarning per
    TransportModel instance; the returned value is exactly the callable at n_bg."""
    from dynameta.warning_types import AdvisoryWarning
    fn = lambda n: 0.30 * M_E * (1.0 + 0.1 * np.log(n / 3e26))                # noqa: E731
    tr = _tm(fn)
    with pytest.warns(AdvisoryWarning, match="builder X: .*density-dependent"):
        m = tr.dos_mass_at_bg("builder X")
    assert m == float(fn(3e26))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tr.dos_mass_at_bg("builder X") == m                           # once per instance
    with pytest.warns(AdvisoryWarning):
        _tm(fn).dos_mass_at_bg("again")                                      # a new instance warns again
    with pytest.warns(AdvisoryWarning, match="raised"):
        _tm(lambda n: 0.3 * M_E if n == 3e26 else 1 / 0).dos_mass_at_bg("y")  # noqa: E731


def test_dos_mass_at_bg_is_silent_and_bit_identical_for_constants():
    """Every constant spelling used across tests/ (scalar lambdas, np.full_like over an array-cast n,
    numpy scalars) is silent, and the value is float(callable(n_bg)) bit for bit."""
    spellings = [lambda n: 0.30 * M_E,
                 lambda n: np.full_like(np.asarray(n, float), 0.30 * M_E),
                 lambda n: np.float64(0.22 * M_E),
                 lambda n: 0.30 * M_E + 0.0 * n,
                 lambda n: 0.30 * M_E * (1.0 + 1e-4 * np.log(n / 3e26))]     # 0.046 % spread: below 1 %
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for fn in spellings:
            tr = _tm(fn)
            assert tr.dos_mass_at_bg("t") == float(fn(tr.n_bg_m3))


def test_fermi_reference_derivative_is_precomputed():
    """setup_semiconductor_region(fermi_ref_param=FERMI_REF_PARAM) hits the precomputed DEVSIM simplify()
    (a live simplify takes ~300 s); the cached string is the v0.15.1 one with the shifted argument."""
    eta = "(Potential - {} - Phi_c0) / V_t".format(PE.FERMI_REF_PARAM)
    key = "simplify(diff({}, Potential))".format("N_c * " + PE.F12_aymerich_humet_expr(eta))
    assert key in PE._SIMPLIFIED
    base = PE._SIMPLIFIED["simplify(diff({}, Potential))".format(
        "N_c * " + PE.F12_aymerich_humet_expr("(Potential - Phi_c0) / V_t"))]
    assert PE._SIMPLIFIED[key] == base.replace("(-Phi_c0 + Potential)",
                                               "(-{} - Phi_c0 + Potential)".format(PE.FERMI_REF_PARAM))
    for bad in ("Phi_c0", "Potential", "1x", "a-b", ""):
        with pytest.raises(ValueError):
            PE._check_param_name(bad, "t")


def test_film_traps_spec_validation():
    with pytest.raises(ValueError):
        PT.FilmInterfaceTraps(dit_eV_m2=1e16, e_lo_eV=0.1, e_hi_eV=-0.1, neutral_level_eV=0.0)
    with pytest.raises(ValueError):
        PT.FilmInterfaceTraps(dit_eV_m2=1e16, e_lo_eV=-0.5, e_hi_eV=0.5)              # split needs E_0
    with pytest.raises(ValueError):
        PT.FilmInterfaceTraps(dit_eV_m2=-1.0)
    with pytest.raises(ValueError):
        PT.FilmInterfaceTraps(charge="none")
    with pytest.raises(ValueError):
        PT.FilmInterfaceTraps(levels=((0.1, -1.0),), charge="acceptor")
    s = PT.FilmInterfaceTraps(dit_eV_m2=1e16, e_lo_eV=-0.5, e_hi_eV=0.5, neutral_level_eV=0.2)
    assert s.bands() == [(-0.5, 0.2, "donor"), (0.2, 0.5, "acceptor")]
    assert PT.FilmInterfaceTraps(dit_eV_m2=1e16, e_lo_eV=-0.5, e_hi_eV=0.5, neutral_level_eV=0.9).bands() \
        == [(-0.5, 0.5, "donor")]
    assert PT.FilmInterfaceTraps().is_empty()
    assert PT.FilmInterfaceTraps(levels=((0.1, 0.0),), charge="donor").is_empty()


def _fd(E, ef):
    return 1.0 / (1.0 + math.exp((E - ef) / V_T))


def test_film_traps_closed_form_equals_fermi_dirac_quadrature():
    """The closed-form sheet charge (softplus) equals the Fermi-Dirac integral of the trap states done by
    scipy quadrature, for a split band (CNL inside, E_F below / at / above it) and discrete levels; the
    small-signal C_it = -dQ/dpsi equals q^2 D_it [f(e_lo) - f(e_hi)] (sign-sensitive)."""
    D, lo, hi, e0 = 2e16, -0.45, 0.35, -0.05
    lv = ((0.1, 3e15), (-0.3, 1e15))
    s = PT.FilmInterfaceTraps(dit_eV_m2=D, e_lo_eV=lo, e_hi_eV=hi, neutral_level_eV=e0, levels=lv)
    for ef in (-0.6, -0.2, -0.05, 0.0, 0.12, 0.5):                     # E_F - E_c [eV]
        def fdq(g, a, b):
            pts = [ef] if a < ef < b else None
            return integrate.quad(g, a, b, epsabs=1e-14, epsrel=1e-11, limit=400, points=pts)[0]

        don = fdq(lambda E: 1.0 - _fd(E, ef), lo, e0)
        acc = fdq(lambda E: _fd(E, ef), e0, hi)
        q_ref = Q_E * (D * don - D * acc - 3e15 * _fd(0.1, ef) + 1e15 * (1.0 - _fd(-0.3, ef)))
        q = float(s.sheet_charge_C_m2(ef / V_T))
        # absolute floor 1e-13 q D: the oracle quadrature's absolute accuracy on the band charge (q passes through
        # zero across the scan; MEASURED |q - q_ref| <= 2.2e-19 C/m^2)
        assert q == pytest.approx(q_ref, rel=1e-9, abs=1e-13 * Q_E * D)
    band = PT.FilmInterfaceTraps(dit_eV_m2=D, e_lo_eV=lo, e_hi_eV=hi, neutral_level_eV=e0)
    for ef in (-0.3, 0.0, 0.2):
        h = 1e-6
        c_it = -(band.sheet_charge_C_m2((ef + h) / V_T) - band.sheet_charge_C_m2((ef - h) / V_T)) / (2 * h)
        assert c_it == pytest.approx(Q_E * D * (_fd(lo, ef) - _fd(hi, ef)), rel=1e-7)
        assert c_it > 0.0


# ------------------------------------------------------------------------------------------------
# DEVSIM harness
# ------------------------------------------------------------------------------------------------
ds = pytest.importorskip("devsim")
from dynameta.carriers.devsim_env import import_devsim     # noqa: E402

ds = import_devsim()
DEV = "eqchg"
# v0.15.1 (2e1af87) digest of the fixture's model-defining calls (tests/_equilibrium_mos.LOGGED, floats
# masked) -- MOS.build_mos with no new option. Recorded on the unfixed tree.
V0151_DIGEST = "19b423384a05358374ec32efe3dad140c7a848724be9e5d2505440d533bef166"


LOG_DEV = "base"                     # the device name is part of the log; the pin was recorded with it


def _log_digest(**kw):
    from _calllog import digest, intercepting
    log = []
    with intercepting(ds, MOS.LOGGED, log, value_repr=MOS.float_free_repr):
        MOS.build_mos(ds, LOG_DEV, **kw)
    return digest(log), log


def _semi(name):
    return MOS.vals(ds, DEV, "semi", name)


def _walk(v, steps=6):
    MOS.solve(ds)
    MOS.set_gate(ds, DEV, v, steps=steps)


@pytest.fixture(autouse=True)
def _clean():
    yield
    MOS.clear(ds, DEV)


def test_model_strings_unchanged_without_new_options():
    """Bit-identical defaults: a device that requests no new option makes exactly the v0.15.1 definition
    calls (digest recorded on 2e1af87; this gate passes on v0.15.1 by construction)."""
    d, _ = _log_digest()
    assert d == V0151_DIGEST


def test_fermi_reference_changes_only_the_closure_strings():
    """With fermi_ref_param the log differs from v0.15.1 exactly in the electron closure and its
    derivative (eta -> (Potential - EF_ref_V - Phi_c0)/V_t); every other call is identical. None is the
    default."""
    assert PE.setup_semiconductor_region.__kwdefaults__["fermi_ref_param"] is None
    _, base = _log_digest()
    _, ref = _log_digest(fermi_ref_param=PE.FERMI_REF_PARAM)
    assert len(base) == len(ref)
    diff = [(a, b) for a, b in zip(base, ref) if a != b]
    assert len(diff) == 2
    for a, b in diff:
        assert dict(a[1])["name"] in ("'Electrons'", "'Electrons:Potential'")
        assert "EF_ref_V" in dict(b[1])["equation"] and "EF_ref_V" not in dict(a[1])["equation"]


def test_fermi_reference_zero_matches_and_v_shifts_every_potential():
    """Fermi-level reference primitive. EF_ref_V = 0 gives the default device's solution to round-off: NOT bit for bit,
    because DEVSIM canonicalises the closure's expression tree and the longer eta subtree reorders a few
    products (MEASURED: n differs by 1 ulp, 2.4e-16 relative, at equal potentials; the solved potentials by
    ~1e-18 V). Each solve stops when its Newton update falls below abs_err = 1e-14 V with a quadratic tail, so each
    is within abs_err of the discrete solution and the two agree to 2 abs_err (the bar); a reference offset by
    delta shifts every potential by delta (second part of this test), so delta > 2e-14 V fails it. Devices that
    do not request the reference keep the v0.15.1 strings (test above).
    EF_ref_V = V with the gate at Vg + V gives the Vg solution with every potential shifted by V and the
    same electron density (to the solver tolerance): a Fermi-level reference, no Dirichlet anywhere."""
    vg, vref = 0.4, 0.3
    MOS.build_mos(ds, DEV)
    _walk(vg)
    p0, n0 = _semi("Potential"), _semi("Electrons")
    MOS.build_mos(ds, DEV, fermi_ref_param=PE.FERMI_REF_PARAM)
    _walk(vg)
    assert np.max(np.abs(_semi("Potential") - p0)) < 2.0 * 1e-14                # 2 x the solves' abs_err
    assert np.max(np.abs(_semi("Electrons") / n0 - 1.0)) < 1e-13
    for v in (vref, -vref):
        MOS.build_mos(ds, DEV, fermi_ref_param=PE.FERMI_REF_PARAM)
        ds.set_parameter(device=DEV, region="semi", name=PE.FERMI_REF_PARAM, value=v)
        ds.set_node_values(device=DEV, region="semi", name="Potential", values=[v] * len(p0))
        nox = len(MOS.vals(ds, DEV, "ox", "x"))
        ds.set_node_values(device=DEV, region="ox", name="Potential", values=[v] * nox)
        ds.set_parameter(device=DEV, name="gate_bias", value=v)
        _walk(vg + v)
        assert np.max(np.abs(_semi("Potential") - v - p0)) < 1e-12
        assert np.max(np.abs(_semi("Electrons") / n0 - 1.0)) < 1e-10
        assert abs(MOS.psi_s(ds, DEV) - v - p0[0]) < 1e-12


def _profile(vg, *, sheet=None, sheet_side="semi", bulk=None, steps=6):
    MOS.build_mos(ds, DEV)
    if sheet is not None:
        name = PE.install_sheet_charge(DEV, "ox_semi", sheet_side, sheet)
        assert name.startswith("QTerm_sheet")
    if bulk is not None:
        PE.install_bulk_charge(DEV, "ox", bulk)
    _walk(vg, steps)
    return _semi("Potential"), _semi("Electrons")


@pytest.mark.parametrize("side", ["semi", "ox"])
@pytest.mark.parametrize("qsign", [+1.0, -1.0])
def test_sheet_charge_shifts_the_gate_by_minus_q_over_cox(side, qsign):
    """Sheet charge. Gauss at the oxide face, C_ox (V_G - psi_s) = -(Q_s + Q): at fixed semiconductor charge a sheet Q
    moves V_G by exactly -Q/C_ox. Device with Q at V0 - Q/C_ox vs device without at V0: the semiconductor
    solutions agree to 1e-9 of the 0.29 V shift (the oxide is charge-free, so the discrete potential is
    exactly linear there). Either side of the interface carries it identically. MEASURED 2.4e-17 of the
    shift; with the sign of Q flipped the potentials differ by 0.047 / 0.081 V."""
    v0, Q = 0.3, qsign * 1.0e-3                                      # Q/C_ox = 0.290 V
    pa, na = _profile(v0)
    pb, nb = _profile(v0 - Q / MOS.C_OX, sheet=Q, sheet_side=side, steps=10)
    assert np.max(np.abs(pb - pa)) <= 1e-9 * abs(Q / MOS.C_OX)
    assert np.max(np.abs(nb / na - 1.0)) < 1e-8


@pytest.mark.parametrize("rsign", [+1.0, -1.0])
def test_bulk_oxide_charge_shifts_the_gate_by_the_centroid_formula(rsign):
    """Bulk oxide charge. Uniform rho in the oxide: dV_G = -rho t_ox^2 / (2 eps_ox) (charge rho t_ox at its centroid t_ox/2
    from the gate). The box-method discretisation of a quadratic potential is exact on any mesh, so the
    profiles agree to 1e-9 of the 0.29 V shift (MEASURED 1.2e-17)."""
    v0, rho = 0.3, rsign * 2.0e5                                      # shift 0.290 V
    shift = -rho * MOS.T_OX ** 2 / (2.0 * MOS.EPS_OX * EPS0)
    pa, na = _profile(v0)
    pb, nb = _profile(v0 + shift, bulk=rho, steps=10)
    assert np.max(np.abs(pb - pa)) <= 1e-9 * abs(shift)
    assert np.max(np.abs(nb / na - 1.0)) < 1e-8


def _dd_profile(vg, *, sheet=None):
    """The fixture with a unipolar drift-diffusion semiconductor (Electrons a solution variable, ohmic body)."""
    from dynameta.carriers import physics_drift_diffusion as DD
    MOS.clear(ds, DEV)
    MOS.build_mesh(ds, DEV)
    with MOS.quiet():
        PE.setup_dielectric_region(DEV, "ox", MOS.EPS_OX)
        DD.setup_semiconductor_region_dd(DEV, "semi", n_bg_m3=MOS.N_BG, eps_static=MOS.EPS_S,
                                         dos_mass_kg=MOS.M_REL * M_E, mobility_m2Vs=0.01)
        PE.setup_interface(DEV, "ox_semi")
        PE.setup_contact(DEV, "gate")
        DD.setup_contact_ohmic_dd(DEV, "body")
    if sheet is not None:
        PE.install_sheet_charge(DEV, "ox_semi", "semi", sheet)
    MOS.solve(ds, abs_err=1e30, rel_err=1e-8)
    for k in range(1, 11):
        ds.set_parameter(device=DEV, name="gate_bias", value=vg * k / 10.0)
        MOS.solve(ds, abs_err=1e30, rel_err=1e-8)
    return _semi("Potential"), _semi("Electrons")


def test_sheet_charge_on_a_drift_diffusion_region():
    """The same -Q/C_ox gate shift on a unipolar drift-diffusion semiconductor (its Poisson composite has
    the Electrons derivative): profiles agree to 1e-9 of the shift (MEASURED 7.2e-17; n to 1.2e-15); with
    the sign of Q flipped they differ by 0.047 V."""
    v0, Q = 0.3, 1.0e-3
    pa, na = _dd_profile(v0)
    pb, nb = _dd_profile(v0 - Q / MOS.C_OX, sheet=Q)
    assert np.max(np.abs(pb - pa)) <= 1e-9 * abs(Q / MOS.C_OX)
    assert np.max(np.abs(nb / na - 1.0)) < 1e-9


def test_sheet_charge_between_two_grounded_dielectrics():
    """Dielectric/dielectric: sigma at the interface of eps1/d1 | eps2/d2 with both ends grounded raises
    the interface to sigma / (eps1 eps0/d1 + eps2 eps0/d2) (Gauss; the potential is piecewise linear, so
    the box method is exact). Sign-sensitive; also checks that a dielectric's PotentialEquation gains the
    node model (it had none)."""
    from dynameta.carriers import eq_registry as _R
    MOS.clear(ds, DEV)
    m = DEV + "_m"
    ds.create_1d_mesh(mesh=m)
    for pos, ps, tag in ((0.0, 0.2e-9, "a"), (6e-9, 0.1e-9, "i"), (16e-9, 0.3e-9, "b")):
        ds.add_1d_mesh_line(mesh=m, pos=pos, ps=ps, tag=tag)
    ds.add_1d_contact(mesh=m, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=m, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=m, material="d1", region="r0", tag1="a", tag2="i")
    ds.add_1d_region(mesh=m, material="d2", region="r1", tag1="i", tag2="b")
    ds.add_1d_interface(mesh=m, name="if", tag="i")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=DEV)
    with MOS.quiet():
        PE.setup_dielectric_region(DEV, "r0", 3.9)
        PE.setup_dielectric_region(DEV, "r1", 22.0)
        PE.setup_interface(DEV, "if")
        PE.setup_contact(DEV, "a")
        PE.setup_contact(DEV, "b")
    assert not _R.recorded(DEV, "PotentialEquation", scope="region", loc="r1")[0]["kwargs"].get("node_model")
    sigma = -2.5e-3
    PE.install_sheet_charge(DEV, "if", "r1", sigma)
    assert _R.recorded(DEV, "PotentialEquation", scope="region", loc="r1")[0]["kwargs"]["node_model"] \
        == "PotentialNodeCharge"
    MOS.solve(ds)
    v_if = MOS.vals(ds, DEV, "r1", "Potential")[0]
    expect = sigma / (3.9 * EPS0 / 6e-9 + 22.0 * EPS0 / 10e-9)
    assert v_if == pytest.approx(expect, rel=1e-12, abs=0)


def test_charge_installers_refuse_bad_input():
    MOS.build_mos(ds, DEV)
    with pytest.raises(ValueError):
        PE.install_sheet_charge(DEV, "ox_semi", "semi", float("nan"))
    with pytest.raises(ValueError):
        PE.install_sheet_charge(DEV, "nope", "semi", 1e-3)
    with pytest.raises(ValueError):
        PE.install_sheet_charge(DEV, "ox_semi", "elsewhere", 1e-3)
    with pytest.raises(ValueError):
        PE.install_bulk_charge(DEV, "ox", float("inf"))


@pytest.mark.devsim_isolated("bitwise comparison of two DEVSIM solves")
def test_work_function_equals_a_bias_shift_and_sets_the_flat_band():
    """Gate work function. A work-function difference is the same device biased by -(phi_m - W_s)/q: applied as bias +
    work_function_offset_V(...) the node values are bit-identical to biasing by V - (phi_m - W_s); written
    into the contact expression instead (psi_offset_V) they agree to the solver tolerance. Physics: at
    V_G = V_FB = (phi_m - W_s)/q the device is flat (psi = 0, n = n_bg everywhere), and at V_G = 0 a metal
    with phi_m > W_s depletes the surface."""
    MOS.build_mos(ds, DEV)
    phi_c0 = float(ds.get_parameter(device=DEV, region="semi", name="Phi_c0"))   # E_c - E_F at psi = 0
    chi, phi_m, vg = 4.3, 4.5, 0.2
    off = PE.work_function_offset_V(phi_m, chi, phi_c0)
    MOS.solve(ds)
    MOS.set_gate(ds, DEV, vg + off, steps=6)
    pa, na = _semi("Potential"), _semi("Electrons")
    MOS.build_mos(ds, DEV)
    MOS.solve(ds)
    MOS.set_gate(ds, DEV, vg - (phi_m - (chi + phi_c0)), steps=6)
    assert np.array_equal(_semi("Potential"), pa) and np.array_equal(_semi("Electrons"), na)
    # the offset written into the contact string: same physics, rounding differs
    MOS.clear(ds, DEV)
    MOS.build_mesh(ds, DEV)
    with MOS.quiet():
        PE.setup_dielectric_region(DEV, "ox", MOS.EPS_OX)
        PE.setup_semiconductor_region(DEV, "semi", n_bg_m3=MOS.N_BG, eps_static=MOS.EPS_S,
                                      dos_mass_kg=MOS.M_REL * M_E)
        PE.setup_interface(DEV, "ox_semi")
        PE.setup_contact(DEV, "gate", psi_offset_V=off)
    MOS.solve(ds)
    MOS.set_gate(ds, DEV, vg, steps=6)
    assert np.max(np.abs(_semi("Potential") - pa)) < 1e-12
    # flat band at V_FB, depletion at 0 V
    MOS.build_mos(ds, DEV)
    MOS.solve(ds)
    MOS.set_gate(ds, DEV, (phi_m - (chi + phi_c0)) + off, steps=6)
    # flat = the neutral bulk potential, which is 2.4e-12 V, not 0: Phi_c0 is calibrated by invert_F12 to
    # rtol 1e-10 in n (v0.15.1 behaviour), i.e. ~V_t 1e-10 in potential
    assert np.max(np.abs(_semi("Potential"))) < 1e-11
    assert np.max(np.abs(_semi("Electrons") / MOS.N_BG - 1.0)) < 1e-9
    MOS.set_gate(ds, DEV, 0.0 + off, steps=6)
    assert MOS.psi_s(ds, DEV) < -0.02 and _semi("Electrons")[0] < 0.5 * MOS.N_BG


def test_interface_traps_fixed_charge_only_routes_to_the_sheet_charge():
    """The call InterfaceTraps(charge='none', recombination=False, fixed_charge_C_m2=-1.6e-3) on
    a non-bipolar interface -- now installs the fixed charge as a sheet charge (it raised ValueError on
    v0.15.1) and solves like install_sheet_charge; trap LEVELS there still raise, naming
    FilmInterfaceTraps."""
    Q = -1.6e-3
    _, nb = _profile(0.3 - Q / MOS.C_OX, sheet=Q, steps=10)
    MOS.build_mos(ds, DEV)
    out = PT.install_interface_traps(DEV, "ox_semi", PT.InterfaceTraps(
        charge="none", recombination=False, fixed_charge_C_m2=Q))
    assert out["region"] == "semi" and out["insulator"] == "ox"
    assert out["sheet_charge_model"] == "QTerm_sheet0"
    _walk(0.3 - Q / MOS.C_OX, 10)
    assert np.max(np.abs(_semi("Electrons") / nb - 1.0)) < 1e-12
    MOS.build_mos(ds, DEV)
    with pytest.raises(ValueError, match="FilmInterfaceTraps"):
        PT.install_interface_traps(DEV, "ox_semi", PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.0,
                                                                     recombination=False))


# ------------------------------------------------------------------------------------------------
# equilibrium-film interface traps
# ------------------------------------------------------------------------------------------------
def _closure():
    """The fixture's closure n(psi) = N_c F_1/2((psi - Phi_c0)/V_t) (the material law under test is
    the trap charge, not the F_1/2 fit, so the oracle uses the same fit)."""
    nc = PE.effective_dos_m3(MOS.M_REL * M_E, T_REF)
    eta_bg = PE.invert_F12(MOS.N_BG / nc)
    phi_c0 = -eta_bg * V_T
    return nc, eta_bg, phi_c0, (lambda psi: nc * PE.F12_aymerich_humet((psi - phi_c0) / V_T))


def _qs_continuum(psi_s):
    """Semi-infinite semiconductor charge per area at surface potential psi_s, the first integral of
    Poisson (independent of DEVSIM): Q_s = -sign(psi_s) sqrt(2 eps_s q int_0^psi_s (n - n_bg) dpsi)."""
    _nc, _e, _p, n = _closure()
    if psi_s == 0.0:
        return 0.0
    w = integrate.quad(lambda p: n(p) - MOS.N_BG, 0.0, psi_s, epsabs=0, epsrel=1e-13, limit=200)[0]
    return -math.copysign(1.0, psi_s) * math.sqrt(2.0 * MOS.EPS_S * EPS0 * Q_E * w)


def _spec(D):
    _nc, eta_bg, _p, _n = _closure()
    e_f = eta_bg * V_T                                                 # E_F - E_c at flat band [eV]
    return PT.FilmInterfaceTraps(dit_eV_m2=D, e_lo_eV=e_f - 0.6, e_hi_eV=e_f + 0.6, neutral_level_eV=e_f)


def _film(D, *, vg=0.0, steps=6):
    MOS.build_mos(ds, DEV)
    out = PT.install_interface_traps(DEV, "ox_semi", _spec(D))
    _walk(vg, steps)
    return out


def _efficiency(dv=1e-5):
    v = float(ds.get_parameter(device=DEV, name="gate_bias"))
    MOS.set_gate(ds, DEV, v + dv)
    up = MOS.psi_s(ds, DEV)
    MOS.set_gate(ds, DEV, v - dv)
    dn = MOS.psi_s(ds, DEV)
    MOS.set_gate(ds, DEV, v)
    return (up - dn) / (2.0 * dv)


DIT = 1.0e16                                                           # 1e12 cm^-2 eV^-1: C_it/C_ox = 0.46


def test_film_traps_small_signal_gate_efficiency():
    """Interface traps on an equilibrium film. At flat band (CNL at E_F, so Q_it = 0 and psi_s = 0 with or without traps) the
    gate-to-surface-potential efficiency is C_ox/(C_ox + C_s + q^2 D_it):
      * vs the D_it = 0 device of the SAME mesh (C_s from its own efficiency, C_ox (1/eff0 - 1)): 1e-9
        (central difference: MEASURED 3.5e-9 at dV = 1e-4 V, 1.5e-12 at 1e-5 V, -3.6e-11 at 1e-6 V);
      * vs the continuum oracle (C_s = d Q_s/d psi_s of the first-integral Q_s, scipy quadrature):
        mesh-limited, 5e-5 (MEASURED -8.7e-6 without, -8.2e-6 with traps).
    eff0 = 0.13608, eff = 0.12800: the trap-free efficiency misses the trapped one by 5.9 %."""
    c_it = Q_E * DIT
    _film(0.0)
    assert abs(MOS.psi_s(ds, DEV)) < 1e-11            # the neutral bulk potential (2.4e-12 V, see above)
    eff0 = _efficiency()
    _film(DIT)
    assert abs(MOS.psi_s(ds, DEV)) < 1e-11
    eff = _efficiency()
    c_s_mesh = MOS.C_OX * (1.0 / eff0 - 1.0)
    assert eff == pytest.approx(MOS.C_OX / (MOS.C_OX + c_s_mesh + c_it), rel=1e-9)
    h = 1e-4
    c_s = -(_qs_continuum(h) - _qs_continuum(-h)) / (2.0 * h)
    assert eff0 == pytest.approx(MOS.C_OX / (MOS.C_OX + c_s), rel=5e-5)
    assert eff == pytest.approx(MOS.C_OX / (MOS.C_OX + c_s + c_it), rel=5e-5)
    assert abs(eff / eff0 - 1.0) > 0.03


@pytest.mark.parametrize("vg", [-0.6, -0.25, 0.25, 0.6])
def test_film_traps_large_signal_gauss_law(vg):
    """Gauss at the oxide face with the trapped charge: the trapped device at V_G and the trap-free device
    at V_G + Q_it(psi_s)/C_ox (Q_it from the closed form in numpy at the trapped device's psi_s) have the
    same surface potential to 1e-12 V (MEASURED <= 7e-18 V) -- depletion and accumulation; Q_it/C_ox is
    +0.052 .. -0.027 V here, and with its sign flipped psi_s misses by 2.7e-3 .. 3.1e-2 V. Against the
    continuum first integral the trapped (V_G, psi_s) lies on V_G = psi_s - (Q_s + Q_it)/C_ox to 5e-5 V
    (mesh-limited, MEASURED <= 8.2e-6 V)."""
    _nc, _e, phi_c0, _n = _closure()
    out = _film(DIT, vg=vg, steps=12)
    assert out["model"] == "QTerm_fit0" and out["region"] == "semi"
    ps = MOS.psi_s(ds, DEV)
    q_it = float(_spec(DIT).sheet_charge_C_m2((ps - phi_c0) / V_T))
    assert abs(q_it / MOS.C_OX) > 1e-3                                # the traps matter here
    fit = MOS.vals(ds, DEV, "semi", "FitCharge0")[0]
    assert fit == pytest.approx(q_it, rel=1e-12, abs=1e-18)
    MOS.build_mos(ds, DEV)
    _walk(vg + q_it / MOS.C_OX, 12)
    assert abs(MOS.psi_s(ds, DEV) - ps) < 1e-12
    va = ps - (_qs_continuum(ps) + q_it) / MOS.C_OX
    assert abs(va - vg) < 5e-5


@pytest.mark.devsim_isolated("bitwise comparison of two DEVSIM solves")
def test_film_traps_zero_density_is_bit_identical():
    """D_it = 0 (and no levels) installs nothing: the model log is v0.15.1's and the solution is the
    no-trap one bit for bit."""
    MOS.build_mos(ds, DEV)
    _walk(0.45)
    p0, n0 = _semi("Potential"), _semi("Electrons")
    from _calllog import digest, intercepting
    log = []
    with intercepting(ds, MOS.LOGGED, log, value_repr=MOS.float_free_repr):
        MOS.build_mos(ds, LOG_DEV)
        out = PT.install_interface_traps(LOG_DEV, "ox_semi", [_spec(0.0), PT.FilmInterfaceTraps(
            levels=((0.1, 0.0),), charge="acceptor")])
    assert digest(log) == V0151_DIGEST and "model" not in out
    MOS.clear(ds, LOG_DEV)
    MOS.build_mos(ds, DEV)
    PT.install_interface_traps(DEV, "ox_semi", _spec(0.0))
    _walk(0.45)
    assert np.array_equal(_semi("Potential"), p0) and np.array_equal(_semi("Electrons"), n0)


def test_film_traps_derivative_is_exact():
    """The supplied QTerm_fit0:Potential equals the central difference of QTerm_fit0 (relative 1e-7 at
    h = 1e-6 V) over E_F - E_c from the band's lower edge to above its top, for a split band plus two
    discrete levels -- the Jacobian entry that keeps Newton quadratic."""
    _nc, eta_bg, phi_c0, _n = _closure()
    e_f = eta_bg * V_T
    spec = PT.FilmInterfaceTraps(dit_eV_m2=DIT, e_lo_eV=e_f - 0.3, e_hi_eV=e_f + 0.3, neutral_level_eV=e_f,
                                 levels=((e_f + 0.1, 4e15), (e_f - 0.15, 2e15)))
    MOS.build_mos(ds, DEV)
    PT.install_film_interface_traps(DEV, "ox_semi", spec)
    n = len(_semi("x"))
    on = _semi("FitArea0") > 0.0
    assert on.sum() == 1
    h = 1e-6

    def at(p):
        ds.set_node_values(device=DEV, region="semi", name="Potential", values=[p] * n)
        return _semi("QTerm_fit0")[on][0], _semi("QTerm_fit0:Potential")

    for delta in np.linspace(-0.5, 0.5, 41):                          # E_F - E_c - e_f [eV]
        p = phi_c0 + e_f + float(delta)
        up, dn = at(p + h)[0], at(p - h)[0]
        val, d = at(p)
        assert d[on][0] == pytest.approx((up - dn) / (2.0 * h), rel=1e-7)
        assert d[on][0] < 0.0 and np.all(d[~on] == 0.0)          # more filled: less positive charge


def test_film_traps_refuse_non_equilibrium_sides():
    MOS.build_mos(ds, DEV)
    with pytest.raises(ValueError, match="not an equilibrium"):
        PT.install_film_interface_traps(DEV, "ox_semi", _spec(DIT), region="ox")
    with pytest.raises(TypeError):
        PT.install_interface_traps(DEV, "ox_semi", [_spec(DIT), PT.InterfaceTraps(fixed_charge_C_m2=1e-3)])
