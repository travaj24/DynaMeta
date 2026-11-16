"""carriers.physics_traps: the devsim-free kernels -- the Shockley-Read-Hall expression builders (evaluated
numerically by the restricted evaluator of tests/_fake_devsim.py), the level-density conversions, the TrapLevel /
InterfaceTraps specs and the D_it quadrature, and the equilibrium-film FilmInterfaceTraps closed form. On the
solver-free legs these were reached only through DEVSIM-gated tests.

ORACLES: detailed balance (at equilibrium the SRH occupancy is the Fermi function of the level for ANY capture
ratio, the net rate vanishes; n1' p1' equals the equilibrium n p on every energy reference, with or without
bandgap narrowing, the Fermi-Dirac emission factors and the density-gradient scale); the kinetic steady state
(R_n = R_p = U at f = f_SS); central finite differences for the closed-form capture derivatives; the level
energies re-derived from the band edges (E_c - E_i = kT ln(N_c/n_i)); exact sums of the D_it bins and the
Gaussian integral; scipy quadrature of the Fermi function over a trap band for the film sheet charge; refusals
that name the field.
"""
import itertools
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

import _fake_devsim as FD
from dynameta.carriers import physics_traps as PT

VT = PT.V_T
NI = 1.0e16
NIE = 3.0e16                 # the narrowed effective intrinsic density (bandgap narrowing on)
NC, NV = 2.8e25, 1.0e25
REFS = ("intrinsic", "conduction", "valence")


def _ev(expr, **v):
    return float(FD.evaluate(expr, v))


def _levels(reference):
    """(energy in that reference, n1, p1) for a level 0.15 eV above E_i (no narrowing), via the library's
    conversion; the energies are re-derived from the band edges (E_c - E_i = kT ln(N_c/n_i))."""
    e_ti = 0.15
    E = {"intrinsic": e_ti, "conduction": VT * math.log(NC / NI) - e_ti,
         "valence": e_ti + VT * math.log(NV / NI)}[reference]
    n1, p1 = PT.level_reference_densities(E, reference, n_i_m3=NI, N_c_m3=NC, N_v_m3=NV)
    return E, n1, p1


def _fd_namespace(n=2e22, p=5e21, gn=1.3, gp=0.8):
    """Values for the Fermi-Dirac emission factors: FDNiN = n_ie0 gamma_n, FDNiP = n_ie0 gamma_p, n_ie^2 = their
    product (bipolar_options' definitions)."""
    nie0 = 2.0e16
    return {"FDGammaN": gn, "FDGammaP": gp, "FDNiN": nie0 * gn, "FDNiP": nie0 * gp,
            "n_ie": math.sqrt(nie0 * gn * nie0 * gp)}


def _combos():
    for bgn, fd, ref, scale in itertools.product((False, True), (False, True), REFS, (None, "s")):
        if fd and not bgn:
            continue
        yield bgn, fd, ref, scale


# =====================================================================================================
# SRH expression builders
# =====================================================================================================
@pytest.mark.parametrize("bgn, fd, ref, scale", list(_combos()))
def test_reference_densities_keep_n1_p1_equal_to_the_equilibrium_np(bgn, fd, ref, scale):
    """srh_reference_densities: on every reference, with / without bandgap narrowing, with the exact Fermi-Dirac
    emission factors and with the density-gradient scale s, n1' p1' equals the third string n_i'^2 (s^2) -- the
    condition for the occupancy to be a Fermi function at equilibrium (budget 1e-13: round-off)."""
    _, n1, p1 = _levels(ref)
    v = dict(n1=n1, p1=p1, n_i=NI, n_ie=NIE, s=1.7)
    if fd:
        v.update(_fd_namespace())
    n1e, p1e, ni2 = PT.srh_reference_densities("n1", "p1", bgn=bgn, reference=ref, scale=scale, fd=fd)
    assert _ev(n1e, **v) * _ev(p1e, **v) == pytest.approx(_ev(ni2, **v), rel=1e-13)
    if not bgn and scale is None:
        assert (n1e, p1e, ni2) == ("n1", "p1", "n_i^2")       # the legacy strings, character for character


def test_reference_densities_refusals():
    with pytest.raises(ValueError, match="energy reference must be one of"):
        PT.srh_reference_densities(reference="midgap")
    with pytest.raises(ValueError, match="fd=True .* needs bgn=True"):
        PT.srh_reference_densities(fd=True)


def _fermi(reference, E, u, n_eq):
    """f of a level quoted as E in `reference`, the Fermi level u kT above the (effective) intrinsic level whose
    density is n_eq: E_t - E_i from the band edges, E_c - E_i = kT ln(N_c/n_eq), E_i - E_v = kT ln(N_v/n_eq)."""
    e_ti = {"intrinsic": E, "conduction": VT * math.log(NC / n_eq) - E, "valence": E - VT * math.log(NV / n_eq)}
    return 1.0 / (1.0 + math.exp(e_ti[reference] / VT - u))


@pytest.mark.parametrize("bgn", [False, True])
@pytest.mark.parametrize("ref", REFS)
def test_srh_occupancy_is_the_fermi_function_at_equilibrium(ref, bgn):
    """At equilibrium (n = n_eq e^u, p = n_eq e^-u, n_eq = n_i or the narrowed n_ie) the SRH occupancy equals the
    Fermi function of the level for capture ratios 1e-3, 1, 1e3 (detailed balance; budget 1e-12); the vacancy
    string is 1 - f, the odds string f/(1 - f) (both written without cancellation), the net rate vanishes against
    the pair rate (budget 1e-12) and the pair rate is 2 n p / D."""
    E, n1, p1 = _levels(ref)
    n_eq = NIE if bgn else NI
    for u in (-3.0, 0.4, 6.0):
        n, p = n_eq * math.exp(u), n_eq * math.exp(-u)
        f_ex = _fermi(ref, E, u, n_eq)
        for ratio in (1e-3, 1.0, 1e3):
            v = dict(Electrons=n, Holes=p, n1=n1, p1=p1, n_i=NI, n_ie=NIE, taun=1e-9, taup=1e-9 * ratio)
            f = _ev(PT.srh_occupancy_expr(bgn=bgn, reference=ref), **v)
            assert f == pytest.approx(f_ex, rel=1e-12, abs=0)
            assert _ev(PT.srh_vacancy_expr(bgn=bgn, reference=ref), **v) == pytest.approx(1.0 - f_ex, rel=1e-12, abs=0)
            assert _ev(PT.srh_occupancy_odds_expr(bgn=bgn, reference=ref), **v) == pytest.approx(
                f_ex / (1.0 - f_ex), rel=1e-12, abs=0)
            U = _ev(PT.srh_rate_expr(bgn=bgn, reference=ref), **v)
            P = _ev(PT.srh_rate_expr(bgn=bgn, reference=ref, pair=True), **v)
            assert abs(U) <= 1e-12 * P
            d = _ev(PT._srh_denominator(*PT.srh_reference_densities(bgn=bgn, reference=ref)[:2], "taun", "taup"),
                    **v)
            assert P == pytest.approx(2.0 * n * p / d, rel=1e-12)


@pytest.mark.parametrize("bgn, fd, ref, scale", list(_combos()))
def test_kinetic_capture_steady_state_is_the_quasi_static_rate(bgn, fd, ref, scale):
    """srh_capture_exprs off equilibrium (n p = 40 n_i'^2, unequal lifetimes): at f = the steady-state occupancy
    the electron and hole capture rates are equal and equal the quasi-static net rate U (budget 1e-12; the
    occupancy-from-capture identity of the kinetic model); the event rates satisfy P + R = 2 x capture term."""
    _, n1, p1 = _levels(ref)
    v = dict(n1=n1, p1=p1, n_i=NI, n_ie=NIE, s=1.3, taun=2e-9, taup=7e-9)
    if fd:
        v.update(_fd_namespace())
    ni2 = _ev(PT.srh_reference_densities(bgn=bgn, reference=ref, scale=scale, fd=fd)[2], **v)
    v.update(Electrons=8.0 * math.sqrt(ni2), Holes=5.0 * math.sqrt(ni2))
    f = _ev(PT.srh_occupancy_expr(bgn=bgn, reference=ref, fd=fd), **v) if scale is None else None
    if f is None:                       # the occupancy builder has no scale: f_SS of the scaled n1', p1'
        n1e, p1e, _ = PT.srh_reference_densities(bgn=bgn, reference=ref, scale=scale, fd=fd)
        N1, P1 = _ev(n1e, **v), _ev(p1e, **v)
        f = (v["taup"] * v["Electrons"] + v["taun"] * P1) / (
            v["taup"] * (v["Electrons"] + N1) + v["taun"] * (v["Holes"] + P1))
    v.update(f=f, g=1.0 - f)
    Rn_s, Rp_s = PT.srh_capture_exprs(bgn=bgn, reference=ref, scale=scale, fd=fd)
    U = _ev(PT.srh_rate_expr(bgn=bgn, reference=ref, scale=scale, fd=fd), **v)
    Rn, Rp = _ev(Rn_s, **v), _ev(Rp_s, **v)
    assert U > 0.0
    assert Rn == pytest.approx(U, rel=1e-12) and Rp == pytest.approx(U, rel=1e-12)
    Pn_s, Pp_s = PT.srh_capture_exprs(bgn=bgn, reference=ref, scale=scale, fd=fd, pair=True)
    assert _ev(Pn_s, **v) + Rn == pytest.approx(2.0 * v["Electrons"] * (1.0 - f) / v["taun"], rel=1e-12)
    assert _ev(Pp_s, **v) + Rp == pytest.approx(2.0 * v["Holes"] * f / v["taup"], rel=1e-12)


@pytest.mark.parametrize("bgn, fd, ref, scale", list(_combos()))
def test_capture_derivatives_match_finite_differences(bgn, fd, ref, scale):
    """srh_capture_derivs against central differences of srh_capture_exprs in n, p and the odds w (f = w/(1+w),
    1 - f = 1/(1+w), df/dw = (1 - f)^2), relative step 1e-6: truncation ~1e-12 and round-off ~1e-10 relative,
    so the budget is 1e-7 of the entry plus 1e-9 of the natural scale |R|/x (for the entries that are exactly
    zero, where the stand-in's n_ie^2 = sqrt(.)^2 leaves round-off noise). With fd=True the emission densities follow the carriers through
    gamma_n = exp(a n), gamma_p = exp(b p) (FDdLnGammaN = a, FDdLnGammaP = b), which the closed form includes."""
    _, n1, p1 = _levels(ref)
    a, b = 2.0e-23, -3.0e-23

    def values(n, p, w):
        v = dict(Electrons=n, Holes=p, n1=n1, p1=p1, n_i=NI, n_ie=NIE, s=1.3, taun=2e-9, taup=7e-9,
                 f=w / (1.0 + w), g=1.0 / (1.0 + w), dfdw=1.0 / (1.0 + w) ** 2)
        if fd:
            v.update(_fd_namespace(gn=math.exp(a * n), gp=math.exp(b * p)))
            v.update(FDdLnGammaN=a, FDdLnGammaP=b)
        return v

    x0 = dict(n=2e22, p=5e21, w=1.7)
    exprs = PT.srh_capture_exprs(bgn=bgn, reference=ref, scale=scale, fd=fd)
    derivs = PT.srh_capture_derivs("n1", "p1", "taun", "taup", "f", "g", "w", "dfdw", bgn=bgn, reference=ref,
                                   scale=scale, fd=fd)
    v0 = values(**x0)
    for expr, der in zip(exprs, derivs):
        got = {k: _ev(der[name], **v0) for k, name in (("n", "Electrons"), ("p", "Holes"), ("w", "w"))}
        r0 = abs(_ev(expr, **v0))
        for k in ("n", "p", "w"):
            h = 1e-6 * x0[k]
            xp, xm = dict(x0), dict(x0)
            xp[k] += h
            xm[k] -= h
            fdv = (_ev(expr, **values(**xp)) - _ev(expr, **values(**xm))) / (2.0 * h)
            assert abs(got[k] - fdv) <= 1e-7 * abs(got[k]) + 1e-9 * r0 / x0[k], (k, expr)


# =====================================================================================================
# level densities
# =====================================================================================================
def test_level_reference_densities_and_equilibrium_n1():
    """n1 p1 = n_i^2 on every reference; a midgap intrinsic level gives (n_i, n_i) exactly; a band-referenced
    level is N exp(-E/kT) from its edge; equilibrium_n1 follows the same conversion (array-broadcast for the
    conduction reference, n_ie^2 / p1 for the valence one). Missing densities of states and unknown references
    are refused."""
    assert PT.n1_p1(0.0, NI) == (NI, NI)
    for ref in REFS:
        E, n1, p1 = _levels(ref)
        assert n1 * p1 == pytest.approx(NI ** 2, rel=1e-13)
        assert n1 == pytest.approx(NI * math.exp(0.15 / VT), rel=1e-12)      # the same physical level
    assert PT.level_reference_densities(0.2, "conduction", n_i_m3=NI, N_c_m3=NC)[0] == pytest.approx(
        NC * math.exp(-0.2 / VT), rel=1e-14)
    assert PT.level_reference_densities(0.2, "valence", n_i_m3=NI, N_v_m3=NV)[1] == pytest.approx(
        NV * math.exp(-0.2 / VT), rel=1e-14)
    with pytest.raises(ValueError, match="needs the band's effective DOS"):
        PT.level_reference_densities(0.2, "conduction", n_i_m3=NI)
    with pytest.raises(ValueError, match="energy reference must be one of"):
        PT.level_reference_densities(0.2, "vacuum", n_i_m3=NI)
    nie = np.array([1e16, 4e16])
    assert np.allclose(PT.equilibrium_n1((0.1, 1e20, "acceptor"), nie, VT), nie * math.exp(0.1 / VT), rtol=1e-14)
    c = PT.equilibrium_n1((0.3, 1e20, "acceptor", "conduction"), nie, VT, N_c_m3=NC)
    assert c.shape == (2,) and np.allclose(c, NC * math.exp(-0.3 / VT), rtol=1e-14)
    vv = PT.equilibrium_n1((0.3, 1e20, "donor", "valence"), nie, VT, N_v_m3=NV)
    assert np.allclose(vv, nie ** 2 / (NV * math.exp(-0.3 / VT)), rtol=1e-13)


# =====================================================================================================
# specs
# =====================================================================================================
def test_trap_level_lifetimes():
    """tau_x0 = 1/(sigma_x v_th,x N_t) from cross sections; explicit lifetimes win; both are validated at
    construction."""
    t = PT.TrapLevel(energy_eV=0.1, density_m3=1e21, kind="acceptor", sigma_n_m2=1e-19, sigma_p_m2=2e-19,
                     v_th_n_ms=2e5, v_th_p_ms=1e5)
    assert t.lifetimes_s() == pytest.approx((1.0 / (1e-19 * 2e5 * 1e21), 1.0 / (2e-19 * 1e5 * 1e21)), rel=1e-14, abs=0)
    t2 = PT.TrapLevel(tau_n0_s=1e-6, sigma_n_m2=1e-19, tau_p0_s=2e-6, density_m3=1e21, kind="donor")
    assert t2.lifetimes_s() == (1e-6, 2e-6)


@pytest.mark.parametrize("kw, exc, match", [
    (dict(kind="trap"), ValueError, "TrapLevel.kind must be one of"),
    (dict(energy_eV=float("inf")), ValueError, "energy_eV must be finite"),
    (dict(kind="acceptor", density_m3=0.0), ValueError, r"charged \(acceptor\) TrapLevel needs density_m3"),
    (dict(dynamic=1), TypeError, "dynamic must be a bool"),
    (dict(dynamic=True), PT.NeutralKineticTrapError, "kind='neutral', dynamic=True"),
    (dict(dynamic=True, kind="donor", density_m3=float("inf")), ValueError, "needs a finite density_m3 > 0"),
    (dict(energy_reference="conduction", energy_eV=-0.1), ValueError, "distance INTO the gap"),
    (dict(energy_reference="midgap"), ValueError, "energy_reference must be one of"),
    (dict(tau_n0_s=None), ValueError, "needs tau_n0_s, or sigma_n_m2"),
    (dict(tau_p0_s=-1.0), ValueError, "tau_p0 must be finite and > 0"),
    (dict(tau_n0_s=None, sigma_n_m2=1e-19, density_m3=0.0), ValueError, "sigma_n_m2 \\* v_th_n_ms \\* density_m3"),
])
def test_trap_level_refusals(kw, exc, match):
    args = dict(energy_eV=0.0, tau_n0_s=1e-6, tau_p0_s=1e-6)
    args.update(kw)
    with pytest.raises(exc, match=match):
        PT.TrapLevel(**args)


@pytest.mark.parametrize("kw, exc, match", [
    (dict(dynamic="yes"), TypeError, "InterfaceTraps.dynamic must be a bool"),
    (dict(kinetic_bin_eV=0.01), ValueError, "set dynamic=True"),
    (dict(dynamic=True, kinetic_bin_eV=0.0), ValueError, "kinetic_bin_eV must be finite and > 0"),
    (dict(dynamic=True, charge="none"), PT.NeutralKineticTrapError, "charge='none', dynamic=True"),
    (dict(dynamic=True, recombination=False), ValueError, "needs recombination=True"),
    (dict(energy_reference="conduction", band_gap_eV=1.1), ValueError, "needs an explicit neutral_level_eV"),
    (dict(energy_reference="valence", neutral_level_eV=0.5, e_lo_eV=-0.1, band_gap_eV=1.1), ValueError,
     "distance INTO the gap"),
    (dict(energy_reference="valence", neutral_level_eV=0.5, distribution="discrete", levels=((-0.2, 1e15),)),
     ValueError, "distance INTO the gap"),
    (dict(distribution="lorentzian"), ValueError, r"uniform\|gaussian\|discrete"),
    (dict(charge="both"), ValueError, "InterfaceTraps.charge must be one of"),
    (dict(v_th_p_ms=0.0), ValueError, "v_th_p_ms must be finite and > 0"),
    (dict(dit_eV_m2=-1.0), ValueError, "dit_eV_m2 must be >= 0"),
    (dict(fixed_charge_C_m2=float("nan")), ValueError, "fixed_charge_C_m2 must be finite"),
    (dict(band_gap_eV=None), ValueError, "uniform InterfaceTraps needs band_gap_eV"),
    (dict(distribution="gaussian", sigma_eV=0.0), ValueError, "gaussian InterfaceTraps needs sigma_eV > 0"),
    (dict(distribution="discrete", levels=((0.1, -1.0),)), ValueError, r"entries are \(energy_eV, N_it_m2 >= 0\)"),
])
def test_interface_traps_refusals(kw, exc, match):
    args = dict(dit_eV_m2=1e16, band_gap_eV=1.12)
    args.update(kw)
    with pytest.raises(exc, match=match):
        PT.InterfaceTraps(**args)


def _check_bins(levels, lo, hi, dmax):
    e = np.array([x[0] for x in levels])
    n = np.array([x[1] for x in levels])
    assert np.all(np.diff(e) > 0.0) and lo < e[0] and e[-1] < hi
    return e, n


def test_uniform_dit_bins_integrate_exactly_and_never_straddle_the_neutral_level():
    """A uniform D_it over the gap, split at a neutral level 0.1 eV above E_i: the bins sum to D x width on each
    side separately (so no bin straddles E_0; exact up to round-off), each bin is at most kT wide (the default;
    a 77 K kT gives proportionally finer bins), states above E_0 are acceptor-like and below donor-like; in the
    E_c - E coordinate "above" means a SMALLER value."""
    D = 1e16
    it = PT.InterfaceTraps(dit_eV_m2=D, band_gap_eV=1.12, neutral_level_eV=0.1)
    lv = it.sheet_levels()
    e, n = _check_bins(lv, -0.56, 0.56, VT)
    kinds = [x[2] for x in lv]
    don, acc = e < 0.1, e > 0.1
    assert don.sum() + acc.sum() == e.size
    assert all(k == "donor" for k, d in zip(kinds, don) if d) and all(k == "acceptor" for k, a in zip(kinds, acc) if a)
    assert n[don].sum() == pytest.approx(D * 0.66, rel=1e-12) and n[acc].sum() == pytest.approx(D * 0.46, rel=1e-12)
    assert np.max(n) / D <= VT * (1 + 1e-12)                    # bin width = N_k / D <= kT
    cold = it.sheet_levels(kT_eV=VT * 77.0 / 300.0)
    assert len(cold) >= 3.5 * len(lv) and max(x[1] for x in cold) / D <= VT * 77.0 / 300.0 * (1 + 1e-12)
    with pytest.raises(ValueError, match="kT_eV must be finite and > 0"):
        it.sheet_levels(kT_eV=0.0)
    fine = PT.InterfaceTraps(dit_eV_m2=D, band_gap_eV=1.12, neutral_level_eV=0.1, bin_eV=0.005)
    lvf = fine.sheet_levels(kT_eV=1.0)                          # an explicit bin_eV overrides kT
    assert len(lvf) == 132 + 92 and sum(x[1] for x in lvf) == pytest.approx(D * 1.12, rel=1e-12)
    ec = PT.InterfaceTraps(dit_eV_m2=D, band_gap_eV=1.12, energy_reference="conduction", neutral_level_eV=0.4)
    lv = ec.sheet_levels()
    assert all(k == ("acceptor" if x < 0.4 else "donor") for x, _n, k in lv)
    assert sum(x[1] for x in lv) == pytest.approx(D * 1.12, rel=1e-12)
    assert PT.InterfaceTraps(dit_eV_m2=D, band_gap_eV=1.12).neutral_level() == 0.0


def test_gaussian_and_discrete_dit():
    """A Gaussian D_it (peak D, width sigma, truncated at +-5 sigma, bins <= sigma/4): the midpoint sum equals
    D sigma sqrt(2 pi) erf(5/sqrt 2) (the truncated integral; MEASURED 3.7e-8 relative, budget 1e-6); cut by the
    band it integrates only the inside part. Discrete levels: empty ones are dropped, kinds by the neutral
    level; a dynamic spec with kinetic_bin_eV uses that width; an empty uniform spec has no levels."""
    D, s = 2e16, 0.08
    g = PT.InterfaceTraps(dit_eV_m2=D, distribution="gaussian", e_peak_eV=-0.1, sigma_eV=s, charge="acceptor")
    lv = g.sheet_levels()
    assert all(k == "acceptor" for *_x, k in lv)
    assert sum(x[1] for x in lv) == pytest.approx(D * s * math.sqrt(2 * math.pi) * erf(5 / math.sqrt(2)), rel=1e-6)
    cut = PT.InterfaceTraps(dit_eV_m2=D, distribution="gaussian", e_peak_eV=-0.1, sigma_eV=s, charge="donor",
                            e_lo_eV=-0.1, e_hi_eV=0.5)
    half = sum(x[1] for x in cut.sheet_levels())
    assert half == pytest.approx(0.5 * D * s * math.sqrt(2 * math.pi) * erf(5 / math.sqrt(2)), rel=1e-6)
    d = PT.InterfaceTraps(distribution="discrete", levels=((-0.2, 1e15), (0.0, 0.0), (0.3, 2e15)), charge="split")
    assert d.sheet_levels() == [(-0.2, 1e15, "donor"), (0.3, 2e15, "acceptor")]
    assert PT.InterfaceTraps(distribution="discrete", levels=((0.3, 1e15),), charge="none").sheet_levels() == [
        (0.3, 1e15, "neutral")]
    k = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.0, dynamic=True, kinetic_bin_eV=0.1, charge="donor")
    lv = k.sheet_levels()
    assert len(lv) == 10 and all(x[1] == pytest.approx(1e15, rel=1e-12) for x in lv)
    assert PT.InterfaceTraps(dit_eV_m2=0.0, band_gap_eV=1.0).sheet_levels() == []


# =====================================================================================================
# FilmInterfaceTraps (equilibrium film)
# =====================================================================================================
def _quad_charge(spec, eta, vt=VT):
    """Q/q by quadrature: -D int f over the acceptor part, +D int (1 - f) over the donor part, f(E) =
    1/(1 + exp((E - E_F)/kT)), E_F - E_c = eta kT; plus the discrete levels."""
    ef = eta * vt
    tot = 0.0
    for a, b, kind in spec.bands():
        if kind == "acceptor":
            tot -= spec.dit_eV_m2 * quad(lambda e: 1.0 / (1.0 + math.exp((e - ef) / vt)), a, b, epsabs=0.0,
                                         epsrel=1e-13, limit=200)[0]
        else:
            tot += spec.dit_eV_m2 * quad(lambda e: 1.0 / (1.0 + math.exp((ef - e) / vt)), a, b, epsabs=0.0,
                                         epsrel=1e-13, limit=200)[0]
    for e, n, kind in spec.level_kinds():
        f = 1.0 / (1.0 + math.exp((e - ef) / vt))
        tot += -n * f if kind == "acceptor" else n * (1.0 - f)
    return tot


@pytest.mark.parametrize("charge, e0", [("split", 0.05), ("acceptor", None), ("donor", None)])
def test_film_trap_charge_matches_quadrature_and_its_derivative(charge, e0):
    """FilmInterfaceTraps.sheet_charge_and_derivative: the closed-form (softplus) band charge and the level
    charge against scipy quadrature of the Fermi function (budget 1e-9 relative to q D V_t), and dQ/deta against
    a central difference (h = 1e-5, budget 1e-8 of max |dQ/deta|), across E_F from deep in the gap to inside the
    band, for each charge mode; a level exactly at the neutral level is donor-like."""
    spec = PT.FilmInterfaceTraps(dit_eV_m2=3e16, e_lo_eV=-0.4, e_hi_eV=0.3, neutral_level_eV=e0, charge=charge,
                                 levels=((-0.1, 2e15), (0.05, 1e15), (0.2, 0.0)))
    if charge == "split":
        assert [k for *_x, k in spec.level_kinds()] == ["donor", "donor"]
        assert spec.bands() == [(-0.4, 0.05, "donor"), (0.05, 0.3, "acceptor")]
    eta = np.linspace(-25.0, 20.0, 31)
    q, dq = spec.sheet_charge_and_derivative(eta)
    ref = np.array([_quad_charge(spec, x) for x in eta]) * PT.Q_E
    assert np.max(np.abs(q - ref)) <= 1e-9 * PT.Q_E * 3e16 * VT
    h = 1e-5
    fd = (spec.sheet_charge_C_m2(eta + h) - spec.sheet_charge_C_m2(eta - h)) / (2 * h)
    assert np.max(np.abs(dq - fd)) <= 1e-8 * np.max(np.abs(dq))


def test_overflow_free_softplus_and_logistic_strings():
    """The model strings of ln(1 + e^x) and 1/(1 + e^-x) that the film-trap installer writes (DEVSIM traps an
    exp overflow even in an unselected branch): evaluated with numpy overflow trapping on, they never
    overflow for |x| up to 800 and equal the reference logaddexp(0, x) and exp(-logaddexp(0, -x)) to round-off
    (budget 1e-14 relative; for the softplus also 1e-15 absolute: log(1 + e^-|x|) has no log1p, so where e^x
    is below the double resolution, x < -37, it returns 0 against e^x -- MEASURED, an absolute error below
    2.2e-16 that is negligible beside the other terms of a trap charge)."""
    x = np.concatenate([np.linspace(-800.0, 800.0, 161), [-1e-3, 0.0, 1e-3]])
    with np.errstate(over="raise"):
        sp = FD.evaluate(PT._softplus_expr("x"), {"x": x})
        lg = FD.evaluate(PT._logistic_expr("x"), {"x": x})
    # atol: softplus(x) -> exp(x) ~ 0 at x = -800 (both sides underflow); MEASURED max diff 9.6e-17
    assert np.allclose(sp, np.logaddexp(0.0, x), rtol=1e-14, atol=1e-15)
    ref = np.exp(-np.logaddexp(0.0, -x))
    assert np.allclose(lg, ref, rtol=1e-14, atol=0.0)


def test_film_trap_neutrality_and_capacitance_limits():
    """With E_F at the neutral level and a band many kT wide on both sides the trapped charge is zero (budget
    1e-12 of q D V_t: the exp(-20) tails cancel by symmetry); deep inside the band the trap capacitance
    -dQ/dpsi = -(dQ/deta)/V_t equals q D (D per eV, the eV/V unit factor 1; budget 1e-8 from the exp(-12) edges);
    an empty spec gives zero charge."""
    D = 1e16
    spec = PT.FilmInterfaceTraps(dit_eV_m2=D, e_lo_eV=-0.6, e_hi_eV=0.6, neutral_level_eV=0.0)
    q, dq = spec.sheet_charge_and_derivative(np.array([0.0, 2.0]))
    assert abs(q[0]) <= 1e-12 * PT.Q_E * D * VT
    assert -dq[1] / VT == pytest.approx(PT.Q_E * D, rel=1e-8, abs=0)
    empty = PT.FilmInterfaceTraps()
    assert empty.is_empty() and empty.bands() == [] and np.all(empty.sheet_charge_C_m2(np.ones(3)) == 0.0)


@pytest.mark.parametrize("kw, match", [
    (dict(charge="none"), "FilmInterfaceTraps.charge must be one of"),
    (dict(dit_eV_m2=float("nan")), "dit_eV_m2 must be finite and >= 0"),
    (dict(dit_eV_m2=1e16, e_lo_eV=0.2, e_hi_eV=-0.2, neutral_level_eV=0.0), "needs finite e_lo_eV < e_hi_eV"),
    (dict(dit_eV_m2=1e16, e_lo_eV=None, e_hi_eV=0.2, neutral_level_eV=0.0), "needs finite e_lo_eV < e_hi_eV"),
    (dict(levels=((0.1, 1e15, 0.0),), charge="donor"), r"levels entries are \(E - E_c \[eV\], N_it"),
    (dict(levels=((float("inf"), 1e15),), charge="donor"), r"levels entries are \(E - E_c \[eV\], N_it"),
    (dict(levels=((0.1, 1e15),)), "needs a finite neutral_level_eV"),
])
def test_film_trap_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        PT.FilmInterfaceTraps(**kw)
