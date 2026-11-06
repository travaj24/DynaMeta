"""DEVSIM gates of the v0.15 Si bundles (consolidated plan item 4.2.1; user decisions 26-29): the house
band-gap narrowing (Yan & Cuevas 2013, Fermi-Dirac statistics), its n_i pairing check, and the carrier-
statistics switch with FD detailed balance. Audit note: docs/audit/2026-11-06-v0.15-si-bundles.md.
Devsim-free gates: tests/test_semiconductor_presets.py.

ORACLES ARE INDEPENDENT OF THE CODE: Yan & Cuevas's constants are re-typed below with their equation /
table numbers, F_1/2 and its inverse come from scipy quadrature + brentq written here, k/q from
scipy.constants. The code's own rational-fit Delta(u) (physics_bipolar_dd.fd_potential_shift) is read
ONLY to size the error budget, never as the reference: where a gate touches the moderately degenerate
range (eta -3..8) that fit is +1..+3.9 % off the exact F_1/2 offset (physics_bipolar_dd, "ACCURACY OF
Delta"), so each such gate (a) bounds the miss by the fit's measured error and (b) ATTRIBUTES it: the
miss divided by exp(Delta_exact - Delta_fit) must vanish to round-off.

EVERY GATE FAILS ON THE OLD CODE (9ea6500): there BGNModel has no 'power_log' kind or pairing fields
and setup_bipolar_region no `statistics` (TypeError) -- and so that an ignored flag could not pass, each
gate also asserts that the old configuration (the legacy statistics, or a mismatched n_i) misses its
oracle by far more than its budget (measured margins in the docstrings).

Fixtures are 1-D (< 2 s each, the 15 um long diodes ~3-5 s), one 2-D bar and one 3-D cell (~5 s).
"""
import math
import warnings

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
from scipy.constants import e as QE, k as KB                              # noqa: E402  (oracle consts)
from scipy.integrate import quad                                           # noqa: E402
from scipy.optimize import brentq                                          # noqa: E402

VT300 = KB * 300.0 / QE
CM3 = 1e6
# --- Yan & Cuevas, J. Appl. Phys. 114, 044508 (2013), re-typed -------------------------------------
A_FD, NREF_FD, B_FD = 0.042e-3, 1e14, 3.0       # Table I, "Fermi-Dirac based BGN (this work)"; Eq. (3)
NI_YC, NC_YC, NV_YC = 9.65e9, 2.86e19, 3.1e19   # Sec. III.B (n_i, Ref. 32); Sec. I (N_c, N_v, Ref. 3); cm^-3
# --- a generic (unpaired) Si fixture for the statistics-only gates (the v0.14 R27 values) -----------
NI, NC300, NV300 = 1.0e16, 2.8e25, 1.04e25
MU_N, MU_P = 0.135, 0.048


def _F12(eta):
    """Normalised F_1/2(eta) = (2/sqrt(pi)) int_0^inf sqrt(x)/(1 + exp(x - eta)) dx (scipy quad)."""
    v, _ = quad(lambda x: math.sqrt(x) / (1.0 + math.exp(min(x - eta, 700.0))), 0.0, max(eta, 0.0) + 60.0,
                points=[eta] if eta > 0 else None, epsabs=0.0, epsrel=1e-13, limit=400)
    return 2.0 / math.sqrt(math.pi) * v


def _F12inv(u):
    return brentq(lambda t: _F12(t) - u, -60.0, 200.0, xtol=1e-15, rtol=1e-15)


def _delta_exact(u):
    """The exact FD built-in offset eta - ln u at u = n/N (quadrature)."""
    return _F12inv(u) - math.log(u)


def _dEg_fd(N_cm3):
    """Yan & Cuevas Eq. (3), FD row of Table I (C = 0), clamped to 0 at N <= N_ref."""
    return A_FD * math.log(N_cm3 / NREF_FD) ** B_FD if N_cm3 > NREF_FD else 0.0


def _nie2_eq1(N_cm3, n_i=NI_YC, n_c=NC_YC):
    """Yan & Cuevas Eq. (1): p0 n0 = n_i^2 (n0/N_c) exp(-F_1/2^-1(n0/N_c)) exp(dEg/kT), n0 = N_D [cm^-6]."""
    u = N_cm3 / n_c
    return n_i ** 2 * u * math.exp(-_F12inv(u)) * math.exp(_dEg_fd(N_cm3) / VT300)


def _fit_delta(u):
    """The code's rational-fit Delta (BUDGET ONLY, never the reference)."""
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    return float(fd_potential_shift(u))


# =====================================================================================================
# fixture helpers
# =====================================================================================================

def _fresh(dev):
    from dynameta.carriers import eq_registry as _R
    for d in list(devsim.get_device_list()):
        if d == dev:
            devsim.delete_device(device=d)
    for m in list(devsim.get_mesh_list()):
        if m == dev + "_m":
            devsim.delete_mesh(mesh=m)
    _R.clear(dev)
    return dev + "_m"


def _drop(dev):
    from dynameta.carriers import eq_registry as _R
    try:
        devsim.delete_device(device=dev)
        devsim.delete_mesh(mesh=dev + "_m")
    except Exception:
        pass
    _R.clear(dev)


def _v(dev, name, region="bulk"):
    return np.asarray(devsim.get_node_model_values(device=dev, region=region, name=name), dtype=float)


def _solve(abs_err=1e11, rel_err=1e-10, max_iter=80):
    devsim.solve(type="dc", solver_type="direct", absolute_error=abs_err, relative_error=rel_err,
                 maximum_iterations=max_iter)


def _device(dev, lines, net_doping, *, options=None, statistics=None, fd=True, tau=1e-7,
            n_i=NI, n_c=NC300, n_v=NV300, ohmic=("lo", "hi"), majority_contact=(), presolve=False):
    """1-D bipolar device seeded in the frame its contacts pin (the R27 fixture of
    tests/test_bipolar_recombination_bgn_ionization.py, plus `statistics`). `majority_contact`: pin psi and
    the MAJORITY carrier only (zero minority flux: the minority is fixed by detailed balance alone)."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    mesh = _fresh(dev)
    devsim.create_1d_mesh(mesh=mesh)
    for pos, ps, tag in lines:
        if tag:
            devsim.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, tag=tag)
        else:
            devsim.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps)
    devsim.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    devsim.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    devsim.add_1d_region(mesh=mesh, material="Si", region="bulk", tag1="lo", tag2="hi")
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=dev)
    devsim.node_model(device=dev, region="bulk", name="NetDoping", equation=net_doping)
    kw = {} if statistics is None else {"statistics": statistics}
    BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=n_c, n_dos_p_m3=n_v, n_i_m3=n_i,
                            mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=tau, tau_p_s=tau,
                            fd_enhancement=fd, options=options, **kw)
    BP.setup_equilibrium_seed_models(dev, "bulk")
    for c in ohmic:
        BP.setup_contact_ohmic_bipolar(dev, c)
    devsim.node_model(device=dev, region="bulk", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "bulk")))
    for var, src in (("Potential", "_seed_psi"), ("Electrons", "IntrinsicElectrons"),
                     ("Holes", "IntrinsicHoles")):
        devsim.set_node_values(device=dev, region="bulk", name=var, values=_v(dev, src).tolist())
    for c in majority_contact:
        i = 0 if c == "lo" else -1
        PE.setup_contact(dev, c)
        devsim.set_parameter(device=dev, name="{}_bias".format(c), value=float(_v(dev, "_seed_psi")[i]))
        var, eq, cur = (("Electrons", "ElectronContinuityEquation", "ElectronCurrent")
                        if _v(dev, "NetDoping")[i] > 0 else
                        ("Holes", "HoleContinuityEquation", "HoleCurrent"))
        cm = "{}_{}_majority".format(c, var.lower())
        devsim.set_parameter(device=dev, name=cm + "_value", value=float(_v(dev, var)[i]))
        devsim.contact_node_model(device=dev, contact=c, name=cm, equation="{} - {}_value".format(var, cm))
        devsim.contact_node_model(device=dev, contact=c, name="{}:{}".format(cm, var), equation="1")
        _R.record_contact_equation(dev, c, name=eq, node_model=cm, edge_current_model=cur)
    if presolve:
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.delete_by_name(dev, ceq)
        _solve(1e10, 1e-10, 100)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.reapply_by_name(dev, ceq)
        for var, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            devsim.set_node_values(device=dev, region="bulk", name=var, values=_v(dev, src).tolist())


def _house_options(**kw):
    from dynameta.carriers.bipolar_options import BGN_SI_YAN_CUEVAS_2013_FD, BipolarOptions
    return BipolarOptions(bgn=BGN_SI_YAN_CUEVAS_2013_FD, **kw)


def _slab_np(dev, N_cm3, *, statistics, options, n_i=NI_YC * CM3):
    """Uniform n+ slab (100 nm), one majority contact, holes kicked 3x off: the steady state is fixed by
    recombination detailed balance alone (gate R27-a's fixture). Returns max and min of n p [m^-6]."""
    try:
        _device(dev, [(0.0, 5e-9, "lo"), (100e-9, 5e-9, "hi")], "{:.10e}".format(N_cm3 * CM3),
                options=options, statistics=statistics, n_i=n_i, n_c=NC_YC * CM3, n_v=NV_YC * CM3,
                ohmic=(), majority_contact=("lo",))
        devsim.set_node_values(device=dev, region="bulk", name="Holes", values=list(3.0 * _v(dev, "Holes")))
        _solve()
        prod = _v(dev, "Electrons") * _v(dev, "Holes")
        nie = _v(dev, "n_ie")
    finally:
        _drop(dev)
    return prod, nie


# =====================================================================================================
# G1: the house FD bundle reproduces Yan & Cuevas Eq. (1)
# =====================================================================================================

@pytest.mark.parametrize("N_cm3", [1e19, 1e20])
def test_house_fd_slab_relaxes_to_yan_cuevas_eq1(N_cm3):
    """GATE G1. The house pairing (BGN_SI_YAN_CUEVAS_2013_FD, n_i 9.65e9 / N_c 2.86e19 / N_v 3.1e19,
    statistics='fermi_dirac') on a uniform n+ slab whose holes are fixed by detailed balance alone must
    reach n p = n_ie,eff^2 of Yan & Cuevas Eq. (1) -- n_i^2 (n0/N_c) exp(-F_1/2^-1(n0/N_c)) exp(dEg/kT),
    scipy quadrature, Table I constants (dEg = 64.09 / 110.75 meV at 1e19 / 1e20).

    BUDGET: |exp(Delta_exact - Delta_fit) - 1| (the rational fit's documented error: 4.18e-3 at 1e19, eta =
    -0.93; 3.50e-2 at 1e20, eta = 2.43) + 1e-9 (closed-form vs quadrature Delta 2.7e-12, Newton 1e-12).
    MEASURED: -4.179e-3 / -3.502e-2, and the ATTRIBUTION (miss / the fit's factor) is 1 to <= 1e-12 at
    every node (MEASURED 2.6e-15 / 4.9e-15). FAILS ON OLD CODE (no power_log / statistics: TypeError) and on the OLD CONFIGURATION:
    the legacy statistics (None; allowed through allow_unpaired_bgn) relax to the Boltzmann product
    n_ie0^2, +13.09 % / +225.9 % off (exp(Delta_exact)); n_i = 1.0e10 (the old fixture value) is refused,
    and when allowed misses by +7.385 % once the fit's factor is removed (+3.6 % raw at 1e20, where the
    fit's -3.5 % partly hides it -- which is why the attribution, not the raw budget, is the sharp check)."""
    from dynameta.carriers.bipolar_options import BGNPairingError
    u = N_cm3 / NC_YC
    oracle = _nie2_eq1(N_cm3) * CM3 ** 2
    d_err = _delta_exact(u) - _fit_delta(u)
    budget = abs(math.expm1(d_err)) + 1e-9
    prod, nie = _slab_np("g1_" + str(int(math.log10(N_cm3))), N_cm3, statistics="fermi_dirac",
                         options=_house_options())
    miss = prod / oracle - 1.0
    assert np.max(np.abs(miss)) <= budget
    assert np.max(np.abs((1.0 + miss) * math.exp(-d_err) - 1.0)) < 1e-9      # the fit, and only the fit
    assert np.max(np.abs(prod / nie ** 2 - 1.0)) < 1e-12                      # the n_ie node model
    # --- the old configurations miss by far more than the budget ---
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy, _ = _slab_np("g1l", N_cm3, statistics=None,
                             options=_house_options(allow_unpaired_bgn=True))
    assert [w.category.__name__ for w in caught if "PAIRED" in str(w.message)] == ["AdvisoryWarning"]
    leg_miss = float(np.max(legacy)) / oracle - 1.0
    assert leg_miss == pytest.approx(math.expm1(_delta_exact(u)), rel=2e-2) and leg_miss > 3.0 * budget
    with pytest.raises(BGNPairingError, match="1.84 meV"):
        _slab_np("g1n", N_cm3, statistics="fermi_dirac", options=_house_options(), n_i=1e16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrong_ni, _ = _slab_np("g1m", N_cm3, statistics="fermi_dirac",
                               options=_house_options(allow_unpaired_bgn=True), n_i=1e16)
    # attributed (fit factor removed) the mismatch is exactly (n_i/n_i,paired)^2 - 1 = +7.39 %, 7 decades
    # above the 1e-9 attribution floor the house configuration meets
    wrong_attr = float(np.max(wrong_ni)) / oracle * math.exp(-d_err)
    assert wrong_attr == pytest.approx((1e10 / NI_YC) ** 2, rel=1e-9) and wrong_attr - 1.0 > 0.07


# =====================================================================================================
# G2: FD detailed balance -- no zero-bias current next to a degenerate layer
# =====================================================================================================

_XJ, _NA_BASE, _TAU_D = 0.5e-6, 1e23, 1e-9


def _diode_lines(sc=1.0):
    """The R27-c long-diode mesh (p base 0.5 um | n+ emitter 15 um = 13.5 L_p), spacings scaled by sc."""
    return [(0.0, 5e-9 * sc, "lo"), (_XJ, 2e-10 * sc, None), (_XJ + 0.05e-6, 2e-9 * sc, None),
            (_XJ + 5e-6, 20e-9 * sc, None), (_XJ + 15e-6, 100e-9 * sc, "hi")]


def _edge_hole_current(dev):
    """Hole current density on the edge 20 nm into the n+ emitter (past its ~1 nm depletion layer)."""
    x = _v(dev, "x")
    xe = 0.5 * (x[:-1] + x[1:])
    jp = np.asarray(devsim.get_edge_model_values(device=dev, region="bulk", name="HoleCurrent"))
    i = int(np.argmin(np.abs(xe - (_XJ + 20e-9))))
    return float(jp[i]), float(xe[i])


def _zero_bias_jp(tag, ND, statistics, sc=1.0):
    dev = "g2" + tag
    try:
        _device(dev, _diode_lines(sc), "ifelse(x < {}, {}, {})".format(_XJ, -_NA_BASE, ND), tau=_TAU_D,
                statistics=statistics, presolve=True)
        _solve(1e12, 1e-10)
        return _edge_hole_current(dev)[0]
    finally:
        _drop(dev)


@pytest.mark.parametrize("ND", [1e25, 1e26])
def test_fd_detailed_balance_removes_the_zero_bias_current(ND):
    """GATE G2 (the statistics switch). n+ (1e19 / 1e20 cm^-3; eta = -0.89 / 2.6 on N_c = 2.8e19) / p
    (1e17) long diode, NO narrowing, both contacts at 0 V. Thermodynamics: no current at equilibrium.
    The legacy statistics (None: FD g-factor transport, Boltzmann-product recombination and contacts)
    carry a hole current exp(-Delta) - 1 times J0p = q n_i^2 D_p/(N_D L_p) -- MEASURED -0.1198 / -0.6987
    J0p (exp(-Delta_fit) - 1 = -0.1219 / -0.7081: the long but finite emitter and the contact pin make up
    the rest). statistics='fermi_dirac' (n_ie from the solution, the FD root at the contacts) leaves only
    the O(h^2) discretisation residual of the edge-averaged g -- MEASURED +1.59e-6 / +4.86e-7 J0p on the
    R27-c mesh, x4.03 / x4.08 smaller at h/2 and x4.03 / x4.00 larger at 2h (a clean h^2 convergence
    study, 2026-11-06). Gate: the residual shrinks by 4 +/- 0.5 on halving h (it is discretisation, not
    model) and is below 1e-4 J0p on the gate mesh (60x / 200x the measured residual, 3 decades below the
    legacy current)."""
    Dp = MU_P * VT300
    Lp = math.sqrt(Dp * _TAU_D)
    J0 = QE * NI * NI * Dp / (ND * Lp)
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    expected_legacy = math.expm1(-float(fd_potential_shift(ND / NC300)))
    legacy = _zero_bias_jp("l", ND, None) / J0
    fd_h = _zero_bias_jp("f", ND, "fermi_dirac") / J0
    fd_h2 = _zero_bias_jp("h", ND, "fermi_dirac", sc=0.5) / J0
    assert legacy == pytest.approx(expected_legacy, rel=3e-2)                # the defect, as described
    assert abs(fd_h) < 1e-4 and abs(legacy) > 1e3 * abs(fd_h)
    assert 3.5 < fd_h / fd_h2 < 4.5                                          # O(h^2) -> 0


# =====================================================================================================
# G3: the house bundle's emitter saturation current (plan P7)
# =====================================================================================================

def test_house_bundle_long_diode_J0_is_yan_cuevas_eq1():
    """GATE G3 (plan 4.2.1 P7). The R27-c long diode with the house pairing in the emitter: N_D = 1e19
    cm^-3 x 15 um (13.5 L_p), tau = 1 ns, n_i = 9.65e9, N_c 2.86e19, N_v 3.1e19, statistics='fermi_dirac',
    V = 0.3 V. The hole current at the n-side depletion edge, de-rated by exp((x - x_j)/L_p) and
    (exp(V/V_t) - 1), must equal J0p = q D_p n_ie,eff^2/(N_D L_p) with n_ie,eff^2 from Yan & Cuevas
    Eq. (1) (scipy F_1/2; dEg = 64.09 meV).

    BUDGET 2e-3 + the FD fit's error: 2e-3 is R27-c's derived SG / finite-emitter / depletion budget
    (O((h/L_p)^2) <= 3e-4, tanh(15 um/L_p) - 1 ~ 1e-11, low injection ~1e-12; R27-c measured -6.3e-4),
    and |exp(Delta_exact - Delta_fit) - 1| = 4.18e-3 at the emitter's eta = -0.93. MEASURED -5.26e-3 raw,
    -1.08e-3 after removing the fit's factor (the R27-c discretisation error, same sign; R27-c -6.3e-4). The p
    BASE (1e17 cm^-3) is outside the house fit (p-type constants pending): the fixture states
    outside_basis='warn' so the base runs unnarrowed, with the one UnreliableResultWarning asserted --
    J0p depends only on the emitter. LOAD-BEARING: the ratio to the same device without BGN is
    exp(dEg/kT) = 11.932, not 1 -- MEASURED 11.909 (-1.89e-3); tolerance 4e-3, the sum of the two J0
    budgets (the fit factor cancels in the ratio: both emitters have the same degeneracy). FAILS ON OLD CODE (TypeError); the legacy statistics would
    give the right forward J0 too (the transport is FD) but a zero-bias current (G2), and n_i = 1.0e10 is
    refused (G1)."""
    import dataclasses
    from dynameta.carriers.bipolar_options import BGN_SI_YAN_CUEVAS_2013_FD, BipolarOptions
    from dynameta.warning_types import UnreliableResultWarning
    ND, V = 1e25, 0.3
    Dp = MU_P * VT300
    Lp = math.sqrt(Dp * _TAU_D)
    bgn = dataclasses.replace(BGN_SI_YAN_CUEVAS_2013_FD, outside_basis="warn")
    j0 = {}
    for tag, opts in (("bgn", BipolarOptions(bgn=bgn)), ("off", None)):
        dev = "g3" + tag
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _device(dev, _diode_lines(), "ifelse(x < {}, {}, {})".format(_XJ, -_NA_BASE, ND), tau=_TAU_D,
                        options=opts, statistics="fermi_dirac", n_i=NI_YC * CM3, n_c=NC_YC * CM3,
                        n_v=NV_YC * CM3, presolve=True)
            hits = [w for w in caught if issubclass(w.category, UnreliableResultWarning)]
            assert len(hits) == (1 if tag == "bgn" else 0) and len(caught) == len(hits), \
                [str(w.message)[:100] for w in caught]
            _solve(1e12, 1e-10)
            for vb in np.arange(0.05, V + 1e-9, 0.05):
                devsim.set_parameter(device=dev, name="lo_bias", value=float(vb))
                _solve(1e12, 1e-10)
            jp, xe = _edge_hole_current(dev)
        finally:
            _drop(dev)
        j0[tag] = abs(jp) * math.exp((xe - _XJ) / Lp) / math.expm1(V / VT300)
    u = 1e19 / NC_YC
    d_err = _delta_exact(u) - _fit_delta(u)
    oracle = QE * _nie2_eq1(1e19) * CM3 ** 2 * Dp / (ND * Lp)
    assert abs(j0["bgn"] / oracle - 1.0) <= 2e-3 + abs(math.expm1(d_err))
    assert abs(j0["bgn"] / oracle * math.exp(-d_err) - 1.0) <= 2e-3            # fit factor removed
    assert j0["bgn"] / j0["off"] == pytest.approx(math.exp(_dEg_fd(1e19) / VT300), rel=4e-3)
    assert j0["bgn"] / j0["off"] > 5.0


# =====================================================================================================
# G4: the FD detailed-balance root (equilibrium_fields) against Eq. (1); the paper's quoted maximum
# =====================================================================================================

def _np_code(N_cm3):
    from dynameta.carriers.bipolar_options import equilibrium_fields
    f = equilibrium_fields(np.array([N_cm3 * CM3]), n_i_m3=NI_YC * CM3, N_c_m3=NC_YC * CM3, N_v_m3=NV_YC * CM3,
                           options=_house_options(), fd_enhancement=True, statistics="fermi_dirac")
    return float(f["n0"][0] * f["p0"][0]) / CM3 ** 2, float(f["n0"][0]) / CM3


def _argmax(fn, lo=19.0, hi=20.5):
    """N [cm^-3] maximising fn on [10^lo, 10^hi] (bounded scalar search in log10 N)."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lx: -fn(10.0 ** lx), bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
    return 10.0 ** r.x


def test_fd_root_matches_eq1_and_the_papers_maximum():
    """GATE G4 (no solve). The contacts' FD detailed-balance root (equilibrium_fields, statistics=
    'fermi_dirac', house pairing) at N_D = 1e18 .. 2e20 cm^-3: n0 = N_D to 1e-13 (full ionization) and
    n0 p0 = Eq. (1) x exp(Delta_exact - Delta_fit) to 1e-12 (MEASURED attribution <= 1.1e-14; the raw
    misses, i.e. the fit's error, are -7.3e-5 / -4.18e-3 / -2.21e-2 / -3.50e-2 / -4.54e-2).

    THE PAPER'S OWN QUOTED VALUE (Sec. III.A, Fig. 3): with FD statistics and the Sec. IV BGN, p0 n0 has a
    maximum "at a dopant concentration of 9e19 cm^-3". The exact Eq. (1) with Table I and the paper's N_c
    puts it at 7.78e19 (scipy, here); the maximum is FLAT -- p0 n0(9e19) is 0.78 % below it -- so the
    quoted one-figure value off a log plot agrees with the equation only through that plateau (FINDING,
    audit note: a maximum at 9e19 itself would need N_c ~ 3.2e19). Gates: the paper's 9e19 lies on the
    oracle's 1 % plateau; the code's maximum (shifted by the fit's eta-dependent error) lies on the same
    plateau of the ORACLE. Adversarial: the legacy (Boltzmann-product) root has no maximum -- n0 p0 grows
    monotonically through 2e20, the Boltzmann behaviour of the paper's Fig. 3."""
    from dynameta.carriers.bipolar_options import equilibrium_fields
    for N in (1e18, 1e19, 5e19, 1e20, 2e20):
        code, n0 = _np_code(N)
        u = N / NC_YC
        assert abs(n0 / N - 1.0) < 1e-13
        assert abs(code / _nie2_eq1(N) * math.exp(_fit_delta(u) - _delta_exact(u)) - 1.0) < 1e-12
    n_max = _argmax(_nie2_eq1)
    peak = _nie2_eq1(n_max)
    assert 7e19 < n_max < 8.5e19                                           # the oracle's own maximum
    assert _nie2_eq1(9e19) >= 0.99 * peak                                  # the paper's 9e19: on the plateau
    n_code = _argmax(lambda N: _np_code(N)[0])
    assert _nie2_eq1(n_code) >= 0.99 * peak
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = equilibrium_fields(np.array([5e19, 1e20, 2e20]) * CM3, n_i_m3=NI_YC * CM3, N_c_m3=NC_YC * CM3,
                               N_v_m3=NV_YC * CM3, options=_house_options(allow_unpaired_bgn=True),
                               fd_enhancement=True)
    prod = f["n0"] * f["p0"]
    assert prod[0] < prod[1] < prod[2]


# =====================================================================================================
# G5: builders thread carrier_statistics (2-D bar, 3-D cell)
# =====================================================================================================

def _bar_design(name, tr):
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    reg.add(Material("Si_b", ConstantOptical(12.0 + 0j), transport=tr))
    stack = Stack(layers=[Layer("si", 40e-9, "Si_b")], superstrate_material="air", substrate_material="air")
    electrodes = [Electrode("anode", "si", "x_lo", role="biased"),
                  Electrode("cathode", "si", "x_hi", role="ground", fixed_voltage_V=0.0)]
    return Design(name=name, unit_cell=UnitCell.square(300e-9), stack=stack, electrodes=electrodes,
                  materials=reg, mesh_2d=Mesh2DSpec())


def _region_param(dev, region, name):
    try:
        return devsim.get_parameter(device=dev, region=region, name=name)
    except Exception:
        return None


def _bar_solve(tag, tr):
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    b = LayeredDevsimBuilder(_bar_design("sib_" + tag, tr), mesh_name="sibm_" + tag, device_name="sibd_" + tag)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b.solve(BiasPoint({"anode": 0.0}, "eq"))
        models = set(devsim.get_node_model_list(device=b.device, region="si"))
        return dict(models=models, fd=_region_param(b.device, "si", "fd_enhancement"),
                    fdc=_region_param(b.device, "si", "fd_detailed_balance"),
                    n=_v(b.device, "Electrons", "si"), p=_v(b.device, "Holes", "si"),
                    nie=_v(b.device, "n_ie", "si") if "n_ie" in models else None, warnings=list(caught))
    finally:
        try:
            devsim.delete_device(device=b.device)
            devsim.delete_mesh(mesh=b.mesh_name)
        except Exception:
            pass
        _R.clear(b.device)


def test_layered_builder_threads_the_bundle_and_its_statistics():
    """GATE G5a. LayeredDevsimBuilder on a 1e19 cm^-3 n-type bar built by materials.semiconductor_presets
    .silicon_transport: (i) the house FD bundle builds statistics='fermi_dirac' (region parameter
    fd_detailed_balance = 1, the n_ie node model with FDGammaN/P on NieBgn) and solves at 0 V to
    n p = n_ie^2 (1e-9) with NO warning (plan P8: the FD-paired set does not trip the apparent-BGN
    double-count advisory); (ii) the Boltzmann bundle builds fd_enhancement = 0 and no detailed-balance
    models, silently; (iii) the FD set under the legacy statistics (carrier_statistics=None) is REFUSED at
    build (BGNPairingError naming 'fermi_dirac'); (iv) without a paired set, carrier_statistics=None is the
    v0.14 region (no fd_detailed_balance parameter, fd_enhancement = 1). FAILS ON OLD CODE: no
    semiconductor_presets module / carrier_statistics field."""
    import dataclasses
    from dynameta.carriers.bipolar_options import BGNPairingError
    from dynameta.materials.semiconductor_presets import (SI_PAIRING_YC2013_BOLTZMANN, SI_PAIRING_YC2013_FD,
                                                          silicon_transport)
    common = dict(n_bg_m3=1e25, mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_srh_s=1e-7)
    fd = _bar_solve("fd", silicon_transport(pairing=SI_PAIRING_YC2013_FD, **common))
    assert fd["fdc"] == 1.0 and fd["fd"] == 1.0 and {"FDGammaN", "FDGammaP", "NieBgn"} <= fd["models"]
    assert fd["warnings"] == [] and np.max(np.abs(fd["n"] * fd["p"] / fd["nie"] ** 2 - 1.0)) < 1e-9
    bz = _bar_solve("bz", silicon_transport(pairing=SI_PAIRING_YC2013_BOLTZMANN, **common))
    assert bz["fd"] == 0.0 and bz["fdc"] is None and "FDGammaN" not in bz["models"] and bz["warnings"] == []
    legacy_tr = dataclasses.replace(silicon_transport(pairing=SI_PAIRING_YC2013_FD, **common),
                                    carrier_statistics=None)
    with pytest.raises(BGNPairingError, match="fermi_dirac"):
        _bar_solve("lg", legacy_tr)
    plain = _bar_solve("pl", dataclasses.replace(legacy_tr, bipolar_options=None))
    assert plain["fdc"] is None and plain["fd"] == 1.0 and "FDGammaN" not in plain["models"]


def test_3d_cell_threads_statistics_and_the_gate_frame_shares_the_pairing_check():
    """GATE G5b. A 3-D bipolar cell (4 x 4 nm, 12 nm Si body at 1e19 cm^-3, 4 nm SiO2) from the house
    bundle through Stacked3DSpec(carrier_statistics='fermi_dirac'): the body region carries the FD
    detailed-balance models; the gate frame phi_bi (ohmic_built_in_offset_V with the spec's statistics)
    equals the potential the body contact pins (EquilibriumPotential, 1e-12 V) and its closed form
    V_t [ln(N_D/n_i) + Delta_fit(N_D/N_c)] - dEg/2; and (plan Q5) the gate-frame path raises the SAME
    BGNPairingError message as the 1-D region setup for a mismatched n_i (one check site). ~5 s."""
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.bipolar_options import BGNPairingError
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    from dynameta.materials.semiconductor_presets import SI_PAIRING_YC2013_FD as P, silicon_bipolar_options
    from dynameta.sweep import BiasPoint
    opts = silicon_bipolar_options(P)
    spec = Stacked3DSpec(semi_material="Si", oxide_material="SiO2", lateral_m=4e-9, semi_thk_m=12e-9,
                         oxide_thk_m=4e-9, n_bg_m3=1e25, eps_semi=11.7, eps_oxide=3.9,
                         dos_mass_kg=P.dos_mass_kg(P.N_c_m3), dos_mass_p_kg=P.dos_mass_kg(P.N_v_m3),
                         physics="bipolar_dd", n_i_m3=P.n_i_m3, mobility_m2Vs=MU_N, mobility_p_m2Vs=MU_P,
                         tau_srh_s=1e-7, grid_n=(3, 3, 5), mesh_min_nm=1.0, mesh_max_nm=3.0,
                         bipolar_options=opts, carrier_statistics="fermi_dirac")
    s = Devsim3DEquilibrium(spec, device_name="sib3d", mesh_name="sib3m")
    try:
        s.solve(BiasPoint({"gate": 0.0}, "flat"))
        models = set(devsim.get_node_model_list(device=s.device, region="semi"))
        pinned = float(np.max(_v(s.device, "EquilibriumPotential", "semi")))
        fdc = _region_param(s.device, "semi", "fd_detailed_balance")
        nc = devsim.get_parameter(device=s.device, region="semi", name="N_dos")
        nv = devsim.get_parameter(device=s.device, region="semi", name="N_dos_p")
    finally:
        s.teardown()
    assert fdc == 1.0 and {"FDGammaN", "NieBgn"} <= models
    assert nc == pytest.approx(P.N_c_m3, rel=1e-12) and nv == pytest.approx(P.N_v_m3, rel=1e-12)
    phi = BP.ohmic_built_in_offset_V(1e25, n_i_m3=P.n_i_m3, n_dos_m3=nc, n_dos_p_m3=nv, options=opts,
                                     statistics="fermi_dirac")
    assert phi == pytest.approx(pinned, abs=1e-12)
    psi = VT300 * (math.log(1e25 / P.n_i_m3) + _fit_delta(1e25 / nc)) - 0.5 * _dEg_fd(1e19)
    assert phi == pytest.approx(psi, abs=1e-12)
    with pytest.raises(BGNPairingError) as gate_err:
        BP.ohmic_built_in_offset_V(1e25, n_i_m3=1e16, n_dos_m3=nc, n_dos_p_m3=nv, options=opts,
                                   statistics="fermi_dirac")
    try:
        with pytest.raises(BGNPairingError) as body_err:
            _device("g5b", [(0.0, 5e-9, "lo"), (50e-9, 5e-9, "hi")], "1e25", options=opts,
                    statistics="fermi_dirac", n_i=1e16, n_c=nc, n_v=nv)
    finally:
        _drop("g5b")
    gate_msg, body_msg = str(gate_err.value), str(body_err.value)
    assert gate_msg.split(": ", 1)[1] == body_msg.split(": ", 1)[1] and "1.84 meV" in gate_msg


# =====================================================================================================
# G6: refusals of the not-implemented combinations; G7: defaults off (same-tree A/B of the call log)
# =====================================================================================================

def _bare_region(dev, two_region=False, **kw):
    """A 1e18 cm^-3 n-type 1-D region set up with setup_bipolar_region(**kw); two_region adds an SiO2
    region behind an interface 'si_ox' (for the interface-trap refusal)."""
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    mesh = _fresh(dev)
    devsim.create_1d_mesh(mesh=mesh)
    devsim.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=5e-9, tag="lo")
    devsim.add_1d_mesh_line(mesh=mesh, pos=50e-9, ps=1e-9, tag="i")
    if two_region:
        devsim.add_1d_mesh_line(mesh=mesh, pos=55e-9, ps=1e-9, tag="hi")
    devsim.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    devsim.add_1d_contact(mesh=mesh, name="hi", tag="hi" if two_region else "i", material="metal")
    devsim.add_1d_region(mesh=mesh, material="Si", region="bulk", tag1="lo", tag2="i")
    if two_region:
        devsim.add_1d_region(mesh=mesh, material="SiO2", region="ox", tag1="i", tag2="hi")
        devsim.add_1d_interface(mesh=mesh, name="si_ox", tag="i")
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=dev)
    devsim.node_model(device=dev, region="bulk", name="NetDoping", equation="1e24")
    BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=NC300, n_dos_p_m3=NV300, n_i_m3=NI,
                            mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=1e-7, tau_p_s=1e-7, **kw)
    if two_region:
        PE.setup_dielectric_region(dev, "ox", 3.9)
        PE.setup_interface(dev, "si_ox")


def test_fermi_dirac_refuses_btbt_charged_traps_and_interface_traps():
    """GATE G6. statistics='fermi_dirac' makes n_ie a function of the carriers; the paths whose Jacobian or
    equilibrium charge assume a fixed n_ie refuse (NotImplementedError) instead of running wrong: BTBT,
    CHARGED bulk trap levels, interface traps. A NEUTRAL level is accepted (its rate references n_ie, whose
    derivative models DEVSIM picks up). A contradictory (statistics, fd_enhancement) pair raises ValueError.
    The same interface-trap install under the legacy statistics is accepted (the guard is specific)."""
    from dynameta.carriers import physics_traps as PT
    from dynameta.carriers.tunneling import SILICON_HURKX
    charged = PT.TrapLevel(energy_eV=0.1, density_m3=1e21, kind="acceptor", tau_n0_s=1e-9, tau_p0_s=1e-9)
    neutral = PT.TrapLevel(energy_eV=0.1, kind="neutral", tau_n0_s=1e-9, tau_p0_s=1e-9)
    its = PT.InterfaceTraps(dit_eV_m2=1e15, band_gap_eV=1.12, recombination=True)
    try:
        with pytest.raises(NotImplementedError, match="band-to-band"):
            _bare_region("g6a", statistics="fermi_dirac", btbt=SILICON_HURKX)
        with pytest.raises(NotImplementedError, match="CHARGED"):
            _bare_region("g6b", statistics="fermi_dirac", traps=(charged,))
        _bare_region("g6c", statistics="fermi_dirac", traps=(neutral,))
        assert {"n_ie", "n_ie:Electrons", "n_ie:Holes"} <= set(devsim.get_node_model_list(device="g6c",
                                                                                           region="bulk"))
        with pytest.raises(ValueError, match="contradicts"):
            _bare_region("g6d", statistics="boltzmann", fd_enhancement=True)
        with pytest.raises(ValueError, match="contradicts"):
            _bare_region("g6e", statistics="fermi_dirac", fd_enhancement=False)
        _bare_region("g6f", two_region=True, statistics="fermi_dirac")
        with pytest.raises(NotImplementedError, match="interface traps"):
            PT.install_interface_traps("g6f", "si_ox", its)
        _bare_region("g6g", two_region=True)
        PT.install_interface_traps("g6g", "si_ox", its)                     # legacy statistics: accepted
    finally:
        for d in ("g6a", "g6b", "g6c", "g6d", "g6e", "g6f", "g6g"):
            _drop(d)


_RECORDED = ("node_model", "edge_model", "contact_node_model", "equation", "contact_equation",
             "set_parameter", "node_solution", "edge_from_node_model")


def _call_log(dev, fd, **kw):
    """The ordered DEVSIM definition log of a 1-D pn fixture (the recorder of tests/_bipolar_calllog.py,
    in process and without its circuit contact; device names normalised)."""
    from dynameta.carriers import physics_bipolar_dd as BP
    log, saved = [], {}

    def wrap(name, fn):
        def inner(*a, **k):
            log.append([name, sorted((x, repr(y).replace(dev, "<dev>")) for x, y in k.items())])
            return fn(*a, **k)
        return inner

    for nm in _RECORDED:
        saved[nm] = getattr(devsim, nm)
        setattr(devsim, nm, wrap(nm, saved[nm]))
    try:
        mesh = _fresh(dev)
        devsim.create_1d_mesh(mesh=mesh)
        devsim.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=5e-9, tag="lo")
        devsim.add_1d_mesh_line(mesh=mesh, pos=100e-9, ps=1e-9)
        devsim.add_1d_mesh_line(mesh=mesh, pos=200e-9, ps=5e-9, tag="hi")
        devsim.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
        devsim.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
        devsim.add_1d_region(mesh=mesh, material="Si", region="bulk", tag1="lo", tag2="hi")
        devsim.finalize_mesh(mesh=mesh)
        devsim.create_device(mesh=mesh, device=dev)
        devsim.node_model(device=dev, region="bulk", name="NetDoping", equation="ifelse(x < 1e-7, -1e24, 3e25)")
        BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1.0e16,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=2e-7,
                                fd_enhancement=fd, n_dos_p_m3=1.04e25, **kw)
        BP.setup_equilibrium_seed_models(dev, "bulk")
        BP.setup_contact_ohmic_bipolar(dev, "lo")
        BP.setup_contact_ohmic_bipolar(dev, "hi")
        log.append(["phi", repr(BP.ohmic_built_in_offset_V(3e25, n_i_m3=1e16, n_dos_m3=2.8e25,
                                                           n_dos_p_m3=1.04e25, fd_enhancement=fd))])
    finally:
        for nm, fn in saved.items():
            setattr(devsim, nm, fn)
        _drop(dev)
    return log


def test_defaults_off_statistics_unset_equals_absent_and_boltzmann_equals_fd_off():
    """GATE G7 (defaults off, tier (i) of plan section 2 item 5: same-tree A/B). The full ordered DEVSIM
    definition log of a pn fixture (every model / equation / parameter / solution call, as
    tests/_bipolar_calllog.py records it) is IDENTICAL with statistics absent and statistics=None (FD and
    Boltzmann fixtures), and statistics='boltzmann' equals fd_enhancement=False without it -- the switch
    adds nothing unless set to 'fermi_dirac', which must differ (the adversarial half). Tier (ii) is the
    recorded digest gate (tests/test_bipolar_recombination_bgn_ionization.py::test_defaults_off_physics_
    definition_is_byte_identical_to_the_recorded_digest), tier (iii) the numeric cross-version gate of
    tests/test_devsim_sg_flux.py -- both run unchanged on this branch (audit note)."""
    for fd in (True, False):
        assert _call_log("g7a", fd) == _call_log("g7b", fd, statistics=None)
    assert _call_log("g7c", False) == _call_log("g7d", False, statistics="boltzmann")
    fdc = _call_log("g7e", True, statistics="fermi_dirac")
    assert fdc != _call_log("g7f", True) and any("fd_detailed_balance" in repr(c) for c in fdc)
