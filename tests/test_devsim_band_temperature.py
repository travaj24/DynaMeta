"""v0.15 plan 4.3.3 -- the band-temperature model on the DEVSIM builders (materials.band_temperature wired into
LayeredDevsimBuilder and Stacked3DSpec.from_design). Devsim-free gates: tests/test_band_temperature.py.
Design + measured numbers: docs/audit/2026-11-06-v0.15-band-temperature.md.

INDEPENDENT ORACLES (written here or in validation/devsim_heterojunction.py, never imported from the code
under test): the Varshni law and the Vurgaftman 2001 Table I / II / XII constants re-typed (Eq. (2.13),
(4.1)); N_c(T), N_v(T) from the masses; Fermi-Dirac statistics by direct quadrature (fd_eta, and F_3/2
here); the abrupt-heterojunction band diagram (q V_bi = W_p - W_n; the isotype barrier from the first
integral of Poisson's equation with exact FD on both sides); the Richardson law.

Materials: the D1a gate materials (GaAs chi 4.07 / E_g 1.424 eV; Al0.3Ga0.7As chi 3.827 / E_g 1.798 eV at
300 K) with band-temperature models anchored at 300 K (GaAs Table I law; Al0.3Ga0.7As by Eq. (4.1) over
GaAs and AlAs with C = -0.127 + 1.310 x) and n_i derived from its 300 K value.

FAILS ON THE OLD CODE (cc0bdf0): TransportModel has no band_temperature field and materials.band_temperature
does not exist -- every test here errors (MEASURED, audit note). The size of what the old code could do at
best (the v0.14 route: gaps and n_i supplied at T, chi tagged at T and used unchanged) is computed
oracle-side in each gate and asserted to exceed 5x the gate's tolerance.

DEVSIM keeps global state: every device is built, measured and torn down inside a module fixture with a
unique device / mesh name; the gates assert on plain numbers.
"""
import dataclasses
import json
import math
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

from validation.devsim_heterojunction import (                               # noqa: E402
    ALGAAS, GAAS, column, current_density, fd_eta, nc_m3, n_i_m3, richardson, stack_design,
    thermal_voltage, transport)
from dynameta.constants import EPS0, M_E, Q_E, T_REF                          # noqa: E402

# ---- the re-typed Vurgaftman 2001 tables (the oracle's own copy) -------------------------------------
GAAS_T1 = (1.519, 0.5405e-3, 204.0)          # Table I
ALAS_T2 = (3.099, 0.885e-3, 530.0)           # Table II (Gamma)
X_AL = 0.3


def varshni(T, e0, a, b):
    return e0 - a * T * T / (T + b)


def v_gaas(T):
    return varshni(T, *GAAS_T1)


def v_algaas(T, x=X_AL):
    return (1 - x) * varshni(T, *GAAS_T1) + x * varshni(T, *ALAS_T2) - x * (1 - x) * (-0.127 + 1.310 * x)


def band_oracle(mat, v_law, gamma, T, T_ref=T_REF):
    """(E_g(T), chi(T)) of a material quoted at T_ref: anchored Varshni shift, chi by gamma of it."""
    d = v_law(T) - v_law(T_ref)
    return mat["eg"] + d, mat["chi"] - gamma * d


def _law(mat):
    from dynameta.materials.band_temperature import algaas_gap_law
    return "GaAs" if mat is GAAS else algaas_gap_law(X_AL)


def tm_band(mat, gamma, T, *, n_bg, acceptor=False, derive=True, T_ref=None):
    """A validation transport (n_i at T_ref, gaps / chi as quoted) + BipolarOptions(T_K=T) + a band model."""
    from dynameta.carriers.bipolar_options import BipolarOptions
    from dynameta.materials.band_temperature import BandTemperature
    t = transport(mat, n_bg=n_bg, physics="bipolar_dd", acceptor=acceptor,
                  n_i_T_K=(T_REF if T_ref is None else T_ref))
    return dataclasses.replace(t, bipolar_options=BipolarOptions(T_K=float(T)), chi_reference_T_K=T_ref,
                               band_temperature=BandTemperature(_law(mat), gamma, derive_n_i=derive))


def _builder(design, tag, **kw):
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    return LayeredDevsimBuilder(design, mesh_name="btm_" + tag, device_name="btd_" + tag, **kw)


def _bias(v):
    from dynameta.sweep import BiasPoint
    return BiasPoint(label="v{:+.4f}".format(v), voltages={"anode": float(v)})


def _trap(y, x):
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))


# =====================================================================================================
# H2-b + defaults off (i): same-tree A/B -- a model AT its reference temperature changes nothing
# =====================================================================================================
_RECORDED = ("node_model", "edge_model", "contact_node_model", "equation", "contact_equation", "set_parameter",
             "node_solution", "edge_from_node_model", "interface_model", "interface_equation",
             "contact_edge_model", "interface_normal_model")


def _recorded_solve(design, tag, biases):
    """Build + solve `design`, recording every DEVSIM definition call (the _bipolar_calllog recorder's rule:
    name + kwargs repr, device / mesh name normalised) and returning (log, {region: arrays}, frames, J)."""
    log, saved = [], {}
    dev, mesh = "btd_" + tag, "btm_" + tag

    def _wrap(name, fn):
        def inner(*a, **kw):
            log.append([name, sorted((k, repr(v).replace(dev, "DEV").replace(mesh, "MESH"))
                                     for k, v in kw.items())])
            return fn(*a, **kw)
        return inner
    for nm in _RECORDED:
        if hasattr(ds, nm):
            saved[nm] = getattr(ds, nm)
            setattr(ds, nm, _wrap(nm, saved[nm]))
    b = _builder(design, tag)
    out = {}
    try:
        for v in biases:
            b.solve(_bias(v), rel_tol=1e-8)
            out[v] = {(r, q): np.asarray(ds.get_node_model_values(device=b.device, region=r, name=q)).copy()
                      for r in ("pgaas", "nalgaas") for q in ("Potential", "Electrons", "Holes")}
            out[(v, "J")] = current_density(b, None, "cathode")
        out["params"] = {(r, p): float(ds.get_parameter(device=b.device, region=r, name=p))
                         for r in ("pgaas", "nalgaas") for p in ("n_i", "N_dos", "N_dos_p", "V_t")}
        out["frames"] = b.band_frames()
        out["ifaces"] = b.heterointerfaces()
    finally:
        for nm, fn in saved.items():
            setattr(ds, nm, fn)
        b.teardown()
    return log, out


@pytest.fixture(scope="module")
def ab_pairs():
    """(model off, model on at its T_ref) for an N-AlGaAs / p-GaAs (200 nm each, 1e17 cm^-3): at 300 K with no
    T_K, and at 250 K where the v0.14 device is the TAGGED one (chi_reference_T_K = 250, n_i at 250 K) and
    the model is anchored at 250 K (gamma_chi = 1, derive_n_i on)."""
    from dynameta.materials.band_temperature import BandTemperature
    res = {}
    for T in (None, 250.0):
        off_l, on_l = [], []
        for ln, nm, mat, acc in (("pgaas", "GaAs_p", GAAS, True), ("nalgaas", "AlGaAs_n", ALGAAS, False)):
            t_off = transport(mat, n_bg=1e23, physics="bipolar_dd", acceptor=acc, T_K=T)
            t_on = dataclasses.replace(t_off, band_temperature=BandTemperature(_law(mat), 1.0, derive_n_i=True))
            off_l.append((ln, nm, 200e-9, t_off))
            on_l.append((ln, nm, 200e-9, t_on))
        tag = "ab{}".format("300" if T is None else "{:g}".format(T))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res[(T, "off")] = _recorded_solve(stack_design(off_l, name=tag), tag + "off", (0.0, 0.3))
            res[(T, "on")] = _recorded_solve(stack_design(on_l, name=tag), tag + "on", (0.0, 0.3))
        res[(T, "tr")] = [t for _ln, _nm, _th, t in on_l]
    return res


@pytest.mark.parametrize("T", [None, 250.0])
@pytest.mark.devsim_isolated("bitwise comparison of separate DEVSIM solves: a device left live by another test is iterated by the same joint Newton and moves the answer by ulps")
def test_band_model_at_its_reference_T_is_bit_identical_to_the_model_off_device(ab_pairs, T):
    """GATE H2-b and defaults-off check (i). A heterostack whose materials carry band-temperature models,
    run AT the models' reference temperature, is the model-off device bit for bit: the ordered log of every
    DEVSIM definition call (model / equation / parameter / interface strings and values; 285 calls), the
    Anderson frames, the region parameters (n_i derived by the model == the input n_i), every node of
    Potential / Electrons / Holes at 0 and 0.3 V, and the terminal current. At 250 K the model-off device is
    the v0.14 TAGGED one (chi_reference_T_K = 250), so this also shows that a model satisfies the chi(T) guard
    without moving anything (warnings are errors in the fixture). Adversarial (the identity is not vacuous):
    the same model-on materials 1 K away from T_ref give a different chi, E_g and n_i (the model IS live in
    these devices; the frame-level shift is gated in test_guard_and_band_gap_check_read_the_band_model)."""
    log_off, off = ab_pairs[(T, "off")]
    log_on, on = ab_pairs[(T, "on")]
    assert len(log_off) > 250                                         # MEASURED 285 calls
    assert json.dumps(log_on) == json.dumps(log_off)
    assert on["frames"] == off["frames"] and on["ifaces"] == off["ifaces"] == {"pgaas__nalgaas": "thermionic"}
    assert on["params"] == off["params"]
    for v in (0.0, 0.3):
        for k, arr in off[v].items():
            assert np.array_equal(on[v][k], arr), (v, k)
        assert on[(v, "J")] == off[(v, "J")]
    from dynameta.materials.band_temperature import band_data_at_T
    T0 = T_REF if T is None else T
    for t in ab_pairs[(T, "tr")]:
        at, away = band_data_at_T(t, T0), band_data_at_T(t, T0 + 1.0)
        assert at == (t.band_gap_eV, t.chi_eV, t.n_i_m3)
        assert away.chi_eV != t.chi_eV and away.E_g_eV != t.band_gap_eV and away.n_i_m3 != t.n_i_m3


# =====================================================================================================
# the v0.14 chi(T) guard and the band-gap check, with models (build only)
# =====================================================================================================
def _build(layers, tag, **kw):
    b = _builder(stack_design(layers, name="bg_" + tag), tag, **kw)
    try:
        b.build_device()
        return b.band_frames()
    finally:
        b.teardown()


def test_guard_and_band_gap_check_read_the_band_model():
    """At 250 K, no chi tags, the default on_chi_temperature="refuse":
    * both materials with models -> builds SILENTLY, and the frame offset Phi_c(AlGaAs) - Phi_c(GaAs) is the
      oracle's chi_GaAs(250) - chi_AlGaAs(250) (gamma = 1/2) to 1e-12 V -- 0.44 meV away from the v0.14 offset
      DEC = chi_GaAs - chi_AlGaAs (old code: the API does not exist; with tags it would use DEC);
    * GaAs with a model, AlGaAs without -> AffinityTemperatureError naming AlGaAs only (v0.14 rule kept);
    * models but derive_n_i=False with the 300 K n_i -> the band-gap check refuses (kT ln(N_c N_v/n_i^2)
      disagrees with E_g(250 K)) and names derive_n_i."""
    from dynameta.carriers.physics_heterojunction import AffinityTemperatureError
    T, g = 250.0, 0.5
    lay = [("pgaas", "GaAs_p", 200e-9, tm_band(GAAS, g, T, n_bg=1e23, acceptor=True)),
           ("nalgaas", "AlGaAs_n", 200e-9, tm_band(ALGAAS, g, T, n_bg=1e23))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fr = _build(lay, "both")
    chi_g = band_oracle(GAAS, v_gaas, g, T)[1]
    chi_a = band_oracle(ALGAAS, v_algaas, g, T)[1]
    off = fr["nalgaas"][0] - fr["pgaas"][0]
    assert off == pytest.approx(chi_g - chi_a, abs=1e-12)
    assert abs(off - (GAAS["chi"] - ALGAAS["chi"])) > 4e-4
    untagged = transport(ALGAAS, n_bg=1e23, physics="bipolar_dd", T_K=T)
    untagged = dataclasses.replace(untagged, chi_reference_T_K=None)
    with pytest.raises(AffinityTemperatureError) as ei:
        _build([lay[0], ("nalgaas", "AlGaAs_n", 200e-9, untagged)], "mixed")
    msg = str(ei.value)
    assert "AlGaAs_n (chi_eV quoted at 300 K" in msg and "GaAs_p (chi_eV" not in msg
    nod = [(ln, nm, th, dataclasses.replace(t, band_temperature=dataclasses.replace(t.band_temperature,
                                                                                    derive_n_i=False)))
           for ln, nm, th, t in lay]
    with pytest.raises(ValueError, match="derive_n_i=True"):
        _build(nod, "noderive")


# =====================================================================================================
# H2-c / H2-d: V_bi(T) and Delta E_c(T) of N-AlGaAs / p-GaAs against an independent band diagram
# =====================================================================================================
NA = ND = 1.0e23
TOL_VBI = 2e-4          # V: the D1a T-b budget (FD-SG g fit vs exact FD; MEASURED 2.5-3.9e-5 there)
TOL_DEC = 1e-6          # eV: the D1a gate B budget
VBI_CASES = ((200.0, 1.0), (400.0, 1.0), (400.0, 0.5))


@pytest.fixture(scope="module")
def pn_band():
    out = {}
    for T, g in VBI_CASES:
        tag = "pn{:g}_{:g}".format(T, g * 10)
        d = stack_design([("pgaas", "GaAs_p", 400e-9, tm_band(GAAS, g, T, n_bg=NA, acceptor=True)),
                          ("nalgaas", "AlGaAs_n", 400e-9, tm_band(ALGAAS, g, T, n_bg=ND))], name=tag)
        b = _builder(d, tag)
        try:
            b.solve(_bias(0.0), rel_tol=1e-8)
            out[(T, g)] = {(r, q): column(b, r, q) for r in ("pgaas", "nalgaas")
                           for q in ("Potential", "Electrons", "Holes")}
        finally:
            b.teardown()
    return out


def _vbi_oracle(T, g):
    vt = thermal_voltage(T)
    eg_g, chi_g = band_oracle(GAAS, v_gaas, g, T)
    _eg_a, chi_a = band_oracle(ALGAAS, v_algaas, g, T)
    eta_n = fd_eta(ND / nc_m3(ALGAAS["me"], T))
    eta_p = fd_eta(NA / nc_m3(GAAS["mh"], T))
    return (chi_g + eg_g + eta_p * vt) - (chi_a - eta_n * vt), chi_g - chi_a


@pytest.mark.slow       # ~25 s on the Windows dev box (three 800 nm bipolar heterostacks)
@pytest.mark.parametrize("T,g", VBI_CASES)
def test_built_in_voltage_and_conduction_offset_at_T_follow_the_band_model(pn_band, T, g):
    """GATE H2-c / H2-d. q V_bi(T) = W_p - W_n with W_p = chi_GaAs(T) + E_g,GaAs(T) + kT eta_p and W_n =
    chi_AlGaAs(T) - kT eta_n (E_g, chi from the oracle's own anchored Varshni and gamma_chi, N_c(T), N_v(T) from
    the masses, eta by exact FD quadrature): to 2e-4 V. The conduction spike read from the SOLVED densities on
    the two interface node copies (E_c - E_F = -kT eta(n/N_c(T))) equals chi_GaAs(T) - chi_AlGaAs(T) to 1e-6 eV.
    MEASURED: V_bi +4.97e-5 / +1.98e-5 / +1.98e-5 V (the g fit's Delta at the bulk u = 0.26 / 0.093 of the
    AlGaAs, 0.021 / 0.0073 of the GaAs); Delta E_c
    to 8e-16 / 3.3e-10 / 3.3e-10 eV.
    LOAD-BEARING (P3 fallback, derived not tuned): the best the v0.14 code can do (gaps and n_i at T, chi used
    unchanged) misses V_bi by gamma x the gap-change mismatch, which must exceed 5x the tolerance. At P3's
    250 / 350 K it is 0.88 / 1.27 mV for gamma = 1 (4.4x / 6.4x) and half that for gamma = 1/2 -- too thin --
    so the gate runs at 200 K (1.34 mV, 6.7x) and 400 K (2.89 / 1.45 mV, 14x / 7.2x) instead of moving to an
    AlAs-rich alloy (past the Gamma-X crossover at x = 0.38-0.39 the Gamma gap is no longer the edge).
    Adversarial: the oracle written with the other convention (gamma 1 <-> 1/2) must miss both gates."""
    vt = thermal_voltage(T)
    r = pn_band[(T, g)]
    _, psip = r[("pgaas", "Potential")]
    _, psin = r[("nalgaas", "Potential")]
    vbi, dec = _vbi_oracle(T, g)
    assert psin[-1] - psip[0] == pytest.approx(vbi, abs=TOL_VBI)
    _, n_p = r[("pgaas", "Electrons")]
    _, n_n = r[("nalgaas", "Electrons")]
    ec_g = -vt * fd_eta(n_p[-1] / nc_m3(GAAS["me"], T))
    ec_a = -vt * fd_eta(n_n[0] / nc_m3(ALGAAS["me"], T))
    assert ec_a - ec_g == pytest.approx(dec, abs=TOL_DEC)
    # load-bearing: the v0.14 route (chi unchanged = gamma 0) misses V_bi by > 5x the tolerance
    vbi_old = vbi + g * ((v_algaas(T) - v_algaas(T_REF)) - (v_gaas(T) - v_gaas(T_REF)))
    assert abs(vbi_old - vbi) > 5 * TOL_VBI
    # adversarial: the other convention's oracle fails both gates
    vbi_x, dec_x = _vbi_oracle(T, 1.5 - g)
    assert abs((psin[-1] - psip[0]) - vbi_x) > TOL_VBI
    assert abs((ec_a - ec_g) - dec_x) > 100 * TOL_DEC


# =====================================================================================================
# H2-e: isotype thermionic-emission conductance with an INDEPENDENTLY computed barrier
# =====================================================================================================
N_TE = 1.0e22           # see the docstring: non-degenerate on both sides, where the FD-SG g fit is accurate
V_LR = 5e-4
TOL_G = 2e-3            # the D1a gate C budget
TE_CASES = ((400.0, 1.0), (400.0, 0.5))
# the FD-SG rational g fit's Delta(u) error vs exact Fermi-Dirac, MEASURED 2026-11-06 against scipy quadrature
# (audit note): <= 6.9e-4 kT for u <= 0.1 (1.9e-3 kT at u = 0.2, 2.5e-2 kT at u = 2)
_FIT_DELTA_ERR_KT_U01 = 6.9e-4


def _f32(eta):
    """Normalised F_3/2 (dF_3/2/d eta = F_1/2): (1/Gamma(5/2)) int e^(3/2)/(1 + exp(e - eta)) de, quadrature."""
    from scipy import integrate
    v, _ = integrate.quad(lambda e: e ** 1.5 / (1.0 + math.exp(min(e - eta, 700.0))), 0.0, max(eta, 0.0) + 60.0,
                          limit=200, epsabs=0.0, epsrel=1e-13)
    return v / math.gamma(2.5)


def _f12(eta):
    from scipy import integrate
    v, _ = integrate.quad(lambda e: math.sqrt(e) / (1.0 + math.exp(min(e - eta, 700.0))), 0.0, max(eta, 0.0) + 60.0,
                          limit=200, epsabs=0.0, epsrel=1e-13)
    return 2.0 / math.sqrt(math.pi) * v


def isotype_barrier(T, chi_g, chi_a):
    """The equilibrium band diagram of n-GaAs / N-AlGaAs (both N_TE, semi-infinite, holes negligible), exact
    FD: on each side the first integral of Poisson's equation gives the interface D-field,
        D^2 = 2 eps q [N_c kT (F_3/2(eta + u/V_t) - F_3/2(eta)) - N u],    u = psi(0) - psi(bulk),
    continuity of D (no interface charge) and of psi with u_G - u_A = V = [(chi_G - kT eta_G) - (chi_A -
    kT eta_A)]/q (the work-function difference) fixes the split (brentq). Returns (phi_B, u_peak) with phi_B =
    E_c,AlGaAs(0) - E_F = -kT eta_A - q u_A (the barrier the AlGaAs side holds) and u_peak = n/N_c at the
    GaAs interface (the most degenerate point of the device)."""
    from scipy import optimize
    vt = thermal_voltage(T)
    ncg, nca = nc_m3(GAAS["me"], T), nc_m3(ALGAAS["me"], T)
    eg_, ea_ = fd_eta(N_TE / ncg), fd_eta(N_TE / nca)
    vbi = (chi_g - vt * eg_) - (chi_a - vt * ea_)
    f0g, f0a = _f32(eg_), _f32(ea_)

    def side(nc, eta, f0, u):
        return nc * vt * (_f32(eta + u / vt) - f0) - N_TE * u
    ug = optimize.brentq(lambda u: GAAS["eps"] * side(ncg, eg_, f0g, u) - ALGAAS["eps"] * side(nca, ea_, f0a, u - vbi),
                         0.0, vbi, xtol=1e-15, rtol=4 * np.finfo(float).eps)
    return -vt * ea_ - (ug - vbi), _f12(eg_ + ug / vt)


@pytest.fixture(scope="module")
def isotype_band():
    out = {}
    for T, g in TE_CASES:
        tag = "te{:g}_{:g}".format(T, g * 10)
        d = stack_design([("ngaas", "GaAs_n", 400e-9, tm_band(GAAS, g, T, n_bg=N_TE)),
                          ("ntop", "Top_algaas", 400e-9, tm_band(ALGAAS, g, T, n_bg=N_TE))], name=tag)
        b = _builder(d, tag, heterointerface="thermionic")
        try:
            b.solve(_bias(0.0), rel_tol=1e-8)
            prof = {r: (column(b, r, "Potential"), column(b, r, "Electrons")) for r in ("ngaas", "ntop")}
            b.solve(_bias(+V_LR), rel_tol=1e-8)
            jp = current_density(b, None, "cathode")
            b.solve(_bias(-V_LR), rel_tol=1e-8)
            jm = current_density(b, None, "cathode")
            out[(T, g)] = dict(prof=prof, g=abs(jp - jm) / (2 * V_LR), ifaces=b.heterointerfaces())
        finally:
            b.teardown()
    return out


@pytest.mark.slow       # ~20 s on the Windows dev box (two isotype stacks, three solves each)
@pytest.mark.parametrize("T,g", TE_CASES)
def test_thermionic_conductance_at_T_with_an_independent_barrier(isotype_band, T, g):
    """GATE H2-e. The low-bias TE conductance of n-GaAs / N-Al0.3Ga0.7As (bipolar, band models) equals
    1/(R_bulk + kT/(q A* T^2 exp(-phi_B/kT))) with phi_B from the oracle's OWN band diagram (isotype_barrier:
    chi(T) from the re-typed Varshni + gamma_chi, exact-FD Poisson first integral) -- not from the solve (the
    shipped T-a takes phi_B from the solved densities). R_bulk = int dz/(q mu n_eq) of the solved profile (the
    interface resistance dominates). Tolerance 2e-3 (D1a gate C). MEASURED: +9.5e-5 (gamma 1) / +9.7e-5 (1/2).
    DOPING, DERIVED: the FD-SG bulk uses the rational g fit, whose Delta(u) differs from exact Fermi-Dirac by
    +2.8 % at u = 0.2 and +3.6-3.9 % at u = 0.5-2 (MEASURED vs quadrature), so an exact-FD barrier is offset by
    up to ~2 max|dDelta| kT. At 1e17 cm^-3 the GaAs accumulation layer reaches u ~ 2 and the offset is a
    mesh-converged 2.3-2.6e-4 eV (G off by 0.84-0.93 % at 300 K; MEASURED at 0.5 / 0.25 / 0.125 nm interface
    spacing) -- a statistics limit of the bulk model, not of this feature (audit note finding). At 1e16 cm^-3
    and 400 K the peak u is <= 0.1 (asserted; MEASURED 0.092; 0.13 at 350 K, so 350 K is outside this budget), where
    |dDelta| <= 6.9e-4 kT: one for the bulk reference, one for the local activity, |dG/G| <= 1.4e-3 < 2e-3.
    LOAD-BEARING: the oracle with chi unchanged (the v0.14 route) differs from it by > 5x the tolerance.
    Adversarial: the other convention's barrier (gamma 1 <-> 1/2) must miss the gate."""
    vt = thermal_voltage(T)
    r = isotype_band[(T, g)]
    assert r["ifaces"] == {"ngaas__ntop": "thermionic"}
    (zg, _), (_, ng) = r["prof"]["ngaas"]
    (zt, _), (_, nt) = r["prof"]["ntop"]
    r_bulk = _trap(1.0 / (Q_E * GAAS["mu_n"] * ng), zg) + _trap(1.0 / (Q_E * ALGAAS["mu_n"] * nt), zt)
    a_star = richardson(ALGAAS["me"])

    def g_of(phi):
        return 1.0 / (r_bulk + vt / (a_star * T ** 2 * math.exp(-phi / vt)))
    chi_g, chi_a = band_oracle(GAAS, v_gaas, g, T)[1], band_oracle(ALGAAS, v_algaas, g, T)[1]
    phi_b, u_peak = isotype_barrier(T, chi_g, chi_a)
    assert u_peak <= 0.1                                           # the fit-accurate regime (docstring)
    assert 2 * _FIT_DELTA_ERR_KT_U01 < TOL_G
    assert vt / (a_star * T ** 2 * math.exp(-phi_b / vt)) > 2.0 * r_bulk      # interface-dominated
    assert r["g"] == pytest.approx(g_of(phi_b), rel=TOL_G)
    phi_old, _ = isotype_barrier(T, GAAS["chi"], ALGAAS["chi"])           # v0.14: chi used unchanged
    assert abs(g_of(phi_old) / g_of(phi_b) - 1.0) > 5 * TOL_G
    chi_gx, chi_ax = band_oracle(GAAS, v_gaas, 1.5 - g, T)[1], band_oracle(ALGAAS, v_algaas, 1.5 - g, T)[1]
    phi_x, _ = isotype_barrier(T, chi_gx, chi_ax)
    assert abs(r["g"] / g_of(phi_x) - 1.0) > TOL_G


# =====================================================================================================
# 3-D: Stacked3DSpec.from_design reads band data through the same kernel
# =====================================================================================================
def _with_materials(layers, extra, rename):
    """validation.carriers_3d_multisemi._design over `layers` (ITO everywhere a semiconductor goes), plus the
    `extra` Materials, with the layers in `rename` ({layer: material}) switched to them (Design validates
    its materials at construction, so the switch is a dataclasses.replace after registering them)."""
    from validation.carriers_3d_multisemi import _design
    d = _design(layers)
    for m in extra:
        d.materials.add(m)
    return dataclasses.replace(d, stack=dataclasses.replace(d.stack, layers=[
        dataclasses.replace(L, background_material=rename[L.name]) if L.name in rename else L
        for L in d.stack.layers]))


def _ito_azo_design(azo_tr):
    from dynameta.materials import DrudeOptical, Material
    from validation.carriers_3d_multisemi import _design
    layers = [("mirror", 70e-9, "Al-Nd"), ("lower_hfo2", 7e-9, "HfO2"), ("ito_bot", 6e-9, "ITO"),
              ("azo_top", 5e-9, "ITO"), ("upper_hfo2", 7e-9, "HfO2"), ("patch", 50e-9, "air")]
    ito = _design(layers).materials.get("ITO")
    itoc = dataclasses.replace(ito, name="ITOc", transport=dataclasses.replace(ito.transport, chi_eV=4.4,
                                                                                 band_gap_eV=3.6))
    azo = Material("AZO", DrudeOptical(eps_inf=3.8, m_opt_kg=0.3 * M_E, gamma_rad_s=1e14), transport=azo_tr)
    return _with_materials(layers, (itoc, azo), {"ito_bot": "ITOc", "azo_top": "AZO"})


def test_3d_from_design_reads_chi_and_n_i_through_the_band_model():
    """Stacked3DSpec.from_design (devsim-free spec construction):
    * an ITO / AZO equilibrium heterostack (3-D heterostacks run at T_REF) whose AZO chi is quoted at 250 K:
      untagged-model-free it refuses (v0.14 guard, unchanged); with a band model it builds, the AZO affinity
      carried into the spec is chi(300 K) = chi_250 - gamma [V(300) - V(250)] (custom triple law, oracle
      written here) to 1e-12 eV, and the model-free ITO's chi is carried unchanged (4.4);
    * a bipolar body at T_K = 250 K with derive_n_i: spec.n_i_m3 = n_i(300) (T/300)^(3/2) exp(-[E_g(T)/kT -
      E_g(300)/kT_300]/2) computed here, rel 1e-13."""
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    from dynameta.carriers.physics_heterojunction import AffinityTemperatureError
    from dynameta.carriers.bipolar_options import BipolarOptions
    from dynameta.materials import TransportModel
    from dynameta.materials.band_temperature import BandTemperature
    law = (3.4, 6.0e-4, 400.0)
    base = TransportModel(n_bg_m3=1e26, eps_static=9.0, dos_mass_kg_of_n_m3=lambda n: 0.25 * M_E,
                          chi_eV=4.3, band_gap_eV=3.3, chi_reference_T_K=250.0)
    with pytest.raises(AffinityTemperatureError, match="AZO"):
        Stacked3DSpec.from_design(_ito_azo_design(base))
    gamma = 0.5
    spec = Stacked3DSpec.from_design(_ito_azo_design(dataclasses.replace(
        base, band_temperature=BandTemperature(law, gamma))))
    assert spec.extra_semiconductors[0][5] == pytest.approx(
        4.3 - gamma * (varshni(300.0, *law) - varshni(250.0, *law)), abs=1e-12)
    assert spec.chi_semi_eV == 4.4
    # bipolar 3-D body with a derived n_i
    from validation.carriers_3d_multisemi import _design
    layers = [("mirror", 70e-9, "Al-Nd"), ("lower_hfo2", 7e-9, "HfO2"), ("ito_bot", 12e-9, "ITO"),
              ("upper_hfo2", 7e-9, "HfO2"), ("patch", 50e-9, "air")]
    ito = _design(layers).materials.get("ITO")
    T = 250.0
    ni300 = n_i_m3(GAAS, T_REF)
    bip = dataclasses.replace(ito.transport, physics="bipolar_dd", n_bg_m3=1e24,
                              mobility_m2Vs_of_n_m3=lambda n: 0.85, hole_mobility_m2Vs_of_n_m3=lambda n: 0.04,
                              tau_srh_s=1e-9, n_i_m3=ni300, dos_mass_p_kg=0.5 * M_E, band_gap_eV=1.424,
                              bipolar_options=BipolarOptions(T_K=T),
                              band_temperature=BandTemperature("GaAs", 1.0, derive_n_i=True))
    d = _with_materials(layers, (dataclasses.replace(ito, name="GaAsB", transport=bip),), {"ito_bot": "GaAsB"})
    spec = Stacked3DSpec.from_design(d)
    eg_t = 1.424 + v_gaas(T) - v_gaas(T_REF)
    expect = ni300 * (T / T_REF) ** 1.5 * math.exp(1.424 / (2 * thermal_voltage(T_REF))
                                                   - eg_t / (2 * thermal_voltage(T)))
    assert spec.n_i_m3 == pytest.approx(expect, rel=1e-13)
    assert abs(spec.n_i_m3 / ni300 - 1.0) > 0.9                  # really moved (n_i(250)/n_i(300) ~ 0.03)
