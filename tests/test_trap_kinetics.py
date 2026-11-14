"""Kinetic traps (DynaMeta v0.15, plan item 4.3.2 steps K1-K2; user decisions 39-41): a trap level's electron
occupancy as a SOLUTION VARIABLE with its own capture / emission rate equation (TrapLevel(dynamic=True),
InterfaceTraps(dynamic=True)), next to the shipped quasi-static model (the default, unchanged).

The variable is the occupancy ODDS w = f/(1-f) = exp(u) (u = the plan's logit), DEVSIM "positive" update --
a measured deviation from decision 40 (audit note section 2: the logit's Newton overflows exp(u) on the
first emission step out of a saturated level).

ORACLES (independent of the code under test; constants re-typed here, CODATA 2018):
  K1-1 kernel identity (numpy): with f the steady-state occupancy written out here, R_n = R_p = U_SRH =
       (n p - n_i'^2)/(tau_p0 (n + n1') + tau_n0 (p + p1')) for the 3 energy references x BGN x DG scale,
       the event rates are the gross sums, and at equilibrium the odds is exp((E_F - E_t)/kT) = n/n1'.
  K1-2 DC identity: kinetic and quasi-static solved on the same devices give the same Potential, n, p,
       terminal current and occupancy (the DC limit of N_t df/dt = R_n - R_p IS the quasi-static model).
  K1-3 the D1c oracles re-run with kinetic levels: tau(E_t) against Schroder's closed form, trap
       compensation against a scipy brentq neutrality root (Fermi occupancy).
  K1-4 DLTS-like emission (Lang 1974 rate e_n = c_n N_c exp(-(E_c - E_t)/kT)): f(t) = f_inf + (f0 - f_inf)
       exp(-t (e_n + e_p + c_n n + c_p p)) at a node deep in the reverse-bias depletion region, f0 from the
       0 V DC state (the test's own SRH formula), with the trapezoidal-rule global error bound (t/tau)(h/tau)^2/12
       and its h^2 convergence.
  K1-5 / K1-6 charge conservation: both contacts circuit-driven, I_lo + I_hi = 0 at every transient step and
       Y_lo + Y_hi = 0 in ssac (Kirchhoff on the device).
  K1-7 defaults off: `dynamic` absent vs dynamic=False -- the DEVSIM call log and every solved array
       bit-identical.
Every gate's derivation, its measured value and its failure on the old code are in its docstring. The
interface (K2, conductance method) gates follow below the bulk ones.
"""
import contextlib
import os
import math

import numpy as np
import pytest

from dynameta.carriers import physics_traps as PT
from dynameta.core.numerics import trapz   # floor-safe (np.trapezoid needs numpy>=2.0)

KB, QE = 1.380649e-23, 1.602176634e-19                   # re-typed (CODATA 2018 exact)
V_T = KB * 300.0 / QE
EPS = np.finfo(float).eps


# =====================================================================================================
# devsim-free
# =====================================================================================================

def _ev(expr, **ns):
    """Evaluate a DEVSIM expression string in numpy-free Python (^ is power; log is the only function)."""
    return eval(expr.replace("^", "**"), {"__builtins__": {}, "log": math.log}, ns)


def _oracle_primed(E, reference, *, n_i, n_ie, Nc, Nv, s):
    """(n1', p1', n_i'^2) re-derived: the level energy -> (n1, p1) by its reference (n1 p1 = n_i^2), then the
    BGN rule (intrinsic: both x n_ie/n_i; conduction: n1 kept, p1' = n_ie^2/n1; valence: the mirror), then the
    DG scale s (n1', p1' x s, n_i'^2 x s^2). D1c doc section 9; P3 section 1.3.5."""
    if reference == "intrinsic":
        n1, p1 = n_i * math.exp(E / V_T), n_i * math.exp(-E / V_T)
    elif reference == "conduction":
        n1 = Nc * math.exp(-E / V_T)
        p1 = n_i * n_i / n1
    else:
        p1 = Nv * math.exp(-E / V_T)
        n1 = n_i * n_i / p1
    if n_ie is None:
        n1p, p1p, ni2 = n1, p1, n_i * n_i
    elif reference == "intrinsic":
        n1p, p1p, ni2 = n1 * n_ie / n_i, p1 * n_ie / n_i, n_ie * n_ie
    elif reference == "conduction":
        n1p, p1p, ni2 = n1, n_ie * n_ie / n1, n_ie * n_ie
    else:
        n1p, p1p, ni2 = n_ie * n_ie / p1, p1, n_ie * n_ie
    return n1, p1, n1p * s, p1p * s, ni2 * s * s


@pytest.mark.parametrize("scale", [None, "s"])
@pytest.mark.parametrize("bgn", [False, True])
@pytest.mark.parametrize("reference", ["intrinsic", "conduction", "valence"])
def test_capture_kernel_identity(reference, bgn, scale):
    """K1-1. srh_capture_exprs (R_n, R_p) and (P_n, P_p), srh_capture_derivs and srh_occupancy_odds_expr
    evaluated on a grid (n, p from 1e8 to 1e25 m^-3, 7 level energies, tau_n0/tau_p0 = 60) with f the
    steady-state occupancy WRITTEN OUT HERE: R_n = R_p = U_SRH; P_n, P_p > 0 equal the gross sums
    n (1-f)/tau_n0 + n1' f/tau_n0 and p f/tau_p0 + p1' (1-f)/tau_p0; the closed-form w-derivative equals
    the analytic one; and at equilibrium (n p = n_i'^2) the odds is n/n1' = exp((E_F - E_t)/kT).
    TOLERANCE 1e-13 relative to the gross rates P: double arithmetic on O(1) ratios, ~10 operations (a
    few ulp each); R_n is a difference of two gross terms, so its error is measured against P, not R (at
    equilibrium R = 0 exactly while each term is ~P). MEASURED max 5.5e-16 of P (2.5 eps). OLD CODE: the kernel
    does not exist (AttributeError: srh_capture_exprs) -- a failure, not a skip.
    ADVERSARIAL: dropping the BGN substitution of p1' (p1 instead of p1') breaks it by O(1) at BGN on
    (asserted below)."""
    worst = _kernel_worst(reference, bgn, scale)
    assert worst < 1e-13, worst


def _kernel_worst(reference, bgn, scale):
    n_i, Nc, Nv = 1.0e16, 2.8e25, 1.04e25
    n_ie = 3.2e16 if bgn else None
    s_val = 1.7 if scale else 1.0
    tn, tp = 3e-7, 5e-9
    energies = ([-0.45, -0.3, -0.1, 0.0, 0.12, 0.3, 0.45] if reference == "intrinsic"
                else [0.05, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0])
    rn_e, rp_e = PT.srh_capture_exprs(bgn=bgn, reference=reference, scale=scale)
    pn_e, pp_e = PT.srh_capture_exprs(bgn=bgn, reference=reference, scale=scale, pair=True)
    dn_e, dp_e = PT.srh_capture_derivs("n1", "p1", "taun", "taup", "f", "g", "w", "g^2", bgn=bgn,
                                       reference=reference, scale=scale)
    odds_e = PT.srh_occupancy_odds_expr(bgn=bgn, reference=reference)
    occ_e = PT.srh_occupancy_expr(bgn=bgn, reference=reference)
    worst = 0.0
    dens = np.logspace(8, 25, 9)
    for E in energies:
        n1, p1, n1p, p1p, ni2 = _oracle_primed(E, reference, n_i=n_i, n_ie=n_ie, Nc=Nc, Nv=Nv, s=s_val)
        for n in dens:
            for p in dens:
                D = tp * (n + n1p) + tn * (p + p1p)
                U = (n * p - ni2) / D
                f = (tp * n + tn * p1p) / D
                g = (tp * n1p + tn * p) / D
                ns = dict(Electrons=n, Holes=p, n1=n1, p1=p1, taun=tn, taup=tp, f=f, g=g, n_i=n_i,
                          n_ie=n_ie if bgn else 0.0, s=s_val)
                Pn = (n * g + n1p * f) / tn
                Pp = (p * f + p1p * g) / tp
                for got, want, P in ((_ev(rn_e, **ns), U, Pn), (_ev(rp_e, **ns), U, Pp),
                                     (_ev(pn_e, **ns), Pn, Pn), (_ev(pp_e, **ns), Pp, Pp)):
                    assert P > 0.0
                    worst = max(worst, abs(got - want) / P)
                dfdw = g * g                                     # f = w/(1+w): df/dw = (1 - f)^2
                ns["w"] = f / g
                for got, want, P in ((_ev(dn_e["w"], **ns), -(n + n1p) * dfdw / tn, (n + n1p) * dfdw / tn),
                                     (_ev(dp_e["w"], **ns), (p + p1p) * dfdw / tp, (p + p1p) * dfdw / tp),
                                     (_ev(dn_e["Electrons"], **ns), g / tn, g / tn),
                                     (_ev(dp_e["Holes"], **ns), f / tp, f / tp)):
                    worst = max(worst, abs(got - want) / P)
                assert dn_e["Holes"] == "0" and dp_e["Electrons"] == "0"
                if scale is None:                               # the occupancy strings carry no DG scale
                    w = _ev(odds_e, **ns)
                    worst = max(worst, abs(w / (1 + w) - f), abs(1 / (1 + w) - g),
                                abs(_ev(occ_e, **ns) - f))
        if scale is None:                                       # equilibrium: odds = n/n1' = e^{(E_F-E_t)/kT}
            for eta in (-12.0, -3.0, 0.0, 4.0, 15.0):
                n = math.sqrt(ni2) * math.exp(eta)
                p = ni2 / n
                w = _ev(odds_e, Electrons=n, Holes=p, n1=n1, p1=p1, taun=tn, taup=tp, n_i=n_i,
                        n_ie=n_ie if bgn else 0.0)
                worst = max(worst, abs(w / (n / n1p) - 1.0))
                if reference == "intrinsic":                    # E_F - E_ie = kT eta, E_t - E_ie = E
                    assert math.log(w) == pytest.approx(eta - E / V_T, abs=1e-12)
    if bgn:                                                     # adversarial: lose the p1 substitution
        n1, p1, n1p, p1p, ni2 = _oracle_primed(0.3 if reference != "intrinsic" else 0.1, reference,
                                               n_i=n_i, n_ie=n_ie, Nc=Nc, Nv=Nv, s=s_val)
        n, p = 1e20, 1e12
        D = tp * (n + n1p) + tn * (p + p1p)
        f, g = (tp * n + tn * p1p) / D, (tp * n1p + tn * p) / D
        bad = rp_e.replace("p1*n_ie/n_i", "p1").replace("(n_ie^2/n1)", "p1")
        if bad != rp_e:
            ns = dict(Electrons=n, Holes=p, n1=n1, p1=p1, taun=tn, taup=tp, f=f, g=g, n_i=n_i,
                      n_ie=n_ie, s=s_val)
            U = (n * p - ni2) / D
            assert abs(_ev(bad, **ns) - U) / ((p * f + p1p * g) / tp) > 1e-2
    return worst


def test_kinetic_spec_validation_and_defaults():
    """The new spec fields: TrapLevel(dynamic=True) needs a finite density_m3 > 0 and a CHARGED kind
    (kind='neutral' raises NeutralKineticTrapError, as does InterfaceTraps(charge='none', dynamic=True));
    `dynamic` must be a bool; InterfaceTraps(dynamic=True) needs recombination=True; kinetic_bin_eV only
    with dynamic=True, > 0, and it sets the bin width of a dynamic spec. Defaults: dynamic=False compares
    EQUAL to the shipped spec, and sheet_levels is unchanged for it."""
    with pytest.raises(ValueError, match="density_m3"):
        PT.TrapLevel(energy_eV=0.1, kind="acceptor", tau_n0_s=1e-9, tau_p0_s=1e-9, dynamic=True)
    with pytest.raises(TypeError):
        PT.TrapLevel(energy_eV=0.1, kind="neutral", tau_n0_s=1e-9, tau_p0_s=1e-9, dynamic=1)
    with pytest.raises(PT.NeutralKineticTrapError, match="acceptor"):
        PT.TrapLevel(energy_eV=0.1, density_m3=1e18, kind="neutral", tau_n0_s=1e-9, tau_p0_s=1e-9,
                     dynamic=True)
    with pytest.raises(PT.NeutralKineticTrapError):
        PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, charge="none", dynamic=True)
    lv = PT.TrapLevel(energy_eV=0.1, density_m3=1e18, kind="acceptor", tau_n0_s=1e-9, tau_p0_s=1e-9,
                      dynamic=True)
    assert lv.dynamic is True
    assert PT.TrapLevel(energy_eV=0.1, tau_n0_s=1e-9, tau_p0_s=1e-9) == \
        PT.TrapLevel(energy_eV=0.1, tau_n0_s=1e-9, tau_p0_s=1e-9, dynamic=False)
    with pytest.raises(ValueError, match="recombination=True"):
        PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, dynamic=True, recombination=False)
    with pytest.raises(ValueError, match="dynamic=True"):
        PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, kinetic_bin_eV=0.05)
    with pytest.raises(ValueError):
        PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, dynamic=True, kinetic_bin_eV=0.0)
    base = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12)
    assert base == PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, dynamic=False)
    assert base.sheet_levels() == PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12,
                                                    dynamic=True).sheet_levels()
    coarse = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, dynamic=True,
                               kinetic_bin_eV=0.1).sheet_levels()
    assert len(coarse) == 12                                    # 2 x ceil(0.56/0.1): split at E_i
    assert sum(x[1] for x in coarse) == pytest.approx(1e16 * 1.12, rel=1e-12)
    from dynameta.carriers import extended_precision as XP
    assert XP._QS_ADVISED == PT.QUASI_STATIC_ADVISED_SOLVES   # the one list, repeated for the fast path


# =====================================================================================================
# DEVSIM: bulk levels (K1)
# =====================================================================================================
ds = pytest.importorskip("devsim")
V = pytest.importorskip("validation.devsim_noise")
from dynameta.carriers import physics_bipolar_dd as BP           # noqa: E402
from dynameta.carriers import bipolar_options as BO              # noqa: E402
from dynameta.carriers import eq_registry as _R                  # noqa: E402
from dynameta.warning_types import AdvisoryWarning               # noqa: E402

NI, NC = 1.0e16, 2.8e25


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    yield
    V.fresh()


def _vals(dev, name, region="r"):
    return np.asarray(ds.get_node_model_values(device=dev, region=region, name=name), dtype=float)


def _solve(ae, re=1e-12, it=100, **kw):
    with V.quiet():
        return ds.solve(type=kw.pop("type", "dc"), absolute_error=ae, relative_error=re,
                        maximum_iterations=it, **kw)


def _seed(dev, region="r"):
    ds.node_model(device=dev, region=region, name="_seed",
                  equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, region)))
    for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"),
                 ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region=region, name=v, values=_vals(dev, m, region).tolist())
    PT.seed_kinetic_traps(dev, [region])


def _diode(dev, traps, *, lines, doping, ae, re=1e-12, tau=1e-7, options=None, btbt=None, circuit=(),
           presolve=False):
    """1-D bipolar device on [(pos, ps, tag)] with contacts 'lo' / 'hi' (ohmic; those in `circuit` are
    circuit-driven: 'hi' -> V1 on node vhi, 'lo' -> V2 on node vlo with NO ac part), solved at equilibrium."""
    m = dev + "_m"
    ds.create_1d_mesh(mesh=m)
    for pos, ps, tag in lines:
        ds.add_1d_mesh_line(mesh=m, pos=pos, ps=ps, **({"tag": tag} if tag else {}))
    ds.add_1d_contact(mesh=m, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=m, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="r", tag1="lo", tag2="hi")
    with V.quiet():
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=dev)
    ds.node_model(device=dev, region="r", name="NetDoping", equation=doping)
    with V.quiet():
        BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=NC, n_i_m3=NI, mobility_n_m2Vs=0.135,
                                mobility_p_m2Vs=0.048, tau_n_s=tau, tau_p_s=tau, fd_enhancement=False,
                                traps=traps, options=options, btbt=btbt)
        BP.setup_equilibrium_seed_models(dev, "r")
        for c, node, src in (("lo", "vlo", "V2"), ("hi", "vhi", "V1")):
            if c in circuit:
                BP.setup_contact_ohmic_bipolar_circuit(dev, c, node_name=node, source_name=src)
            else:
                BP.setup_contact_ohmic_bipolar(dev, c)
    if "lo" in circuit:
        ds.circuit_alter(name="V2", param="acreal", value=0.0)      # a grounded, AC-quiet second terminal
    _seed(dev)
    if presolve:                                                     # frozen-carrier Poisson first
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.delete_by_name(dev, ceq)
        _solve(1e-10, 1e-12)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.reapply_by_name(dev, ceq)
        for v, mm in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="r", name=v, values=_vals(dev, mm).tolist())
        PT.seed_kinetic_traps(dev)
    # a circuit node at zero current never meets a relative test below ~1e-9 (its update is round-off
    # over ~0): 1e-8 there; the device rows converge to ~1e-15 regardless (DEVSIM's AND test per equation)
    _solve(ae, max(re, 1e-8) if circuit else re)
    return dev


def _kin(t, dyn=True):
    """The same TrapLevel with dynamic=dyn (a kinetic level needs N_t: 1e18 for the recombination-only
    ones, which does not enter Poisson)."""
    kw = dict(energy_eV=t.energy_eV, density_m3=t.density_m3 or 1e18, kind=t.kind, tau_n0_s=t.tau_n0_s,
              tau_p0_s=t.tau_p0_s, sigma_n_m2=t.sigma_n_m2, sigma_p_m2=t.sigma_p_m2, v_th_n_ms=t.v_th_n_ms,
              v_th_p_ms=t.v_th_p_ms, energy_reference=t.energy_reference)
    return PT.TrapLevel(dynamic=dyn, **kw)


DIODE_LINES = [(0.0, 5e-9, "lo"), (0.5e-6, 1e-9, None), (1e-6, 5e-9, "hi")]
DIODE_DOP = "ifelse(x < 0.5e-6, -1e23, 1e23)"
COMP_LINES = [(0.0, 2e-9, "lo"), (100e-9, 0.2e-9, None), (200e-9, 2e-9, "hi")]
COMP_DOP = "ifelse(x < 1e-7, -5e24, 5e24)"
COMP_OPTS = BO.BipolarOptions(auger=BO.AUGER_SI_DZIEWIOR_SCHMID_1977,
                              radiative=BO.RADIATIVE_SI_TRUPKE_2003, bgn=BO.BGN_SI_SLOTBOOM_1977)
DC_CASES = {
    # forward-biased Si diode, a charged level 20 % of the doping, each charge kind
    "acceptor": dict(trap=PT.TrapLevel(energy_eV=0.15, density_m3=2e22, kind="acceptor", tau_n0_s=3e-9,
                                       tau_p0_s=5e-9), bias=("lo", (-0.1, -0.2, -0.3, -0.4)), ae=1e10),
    "donor": dict(trap=PT.TrapLevel(energy_eV=-0.2, density_m3=2e22, kind="donor", tau_n0_s=3e-9,
                                    tau_p0_s=5e-9), bias=("lo", (-0.1, -0.2, -0.3, -0.4)), ae=1e10),
    # adversarial (P3 1.5): a deep level near E_c (n1/n0 = 1e3 on the n side), tau_n/tau_p = 1e6,
    # BGN on, the VALENCE reference
    "deep_valence_bgn": dict(trap=PT.TrapLevel(energy_eV=1.0, energy_reference="valence", density_m3=1e22,
                                               kind="acceptor", tau_n0_s=1e-3, tau_p0_s=1e-9),
                             bias=("lo", (-0.1, -0.2, -0.3)), ae=1e10,
                             options=BO.BipolarOptions(bgn=BO.BGN_SI_SLOTBOOM_1977)),
    # the composition fixture of tests/test_devsim_phase_composition.py: BGN + Auger + radiative +
    # acceptor level + BTBT, at 0.5 V reverse
    "composition": dict(trap=PT.TrapLevel(energy_eV=0.2, density_m3=1.0e21, kind="acceptor",
                                          tau_n0_s=2e-7, tau_p0_s=1e-7),
                        bias=("hi", (0.1, 0.2, 0.3, 0.4, 0.5)), ae=1e12, lines=COMP_LINES, dop=COMP_DOP,
                        options=COMP_OPTS, btbt=True, presolve=True),
}


def _dc_state(case, dyn):
    from dynameta.carriers.tunneling import SILICON_HURKX
    c = DC_CASES[case]
    dev = _diode("kdc", (_kin(c["trap"], dyn),), lines=c.get("lines", DIODE_LINES),
                 doping=c.get("dop", DIODE_DOP), ae=c["ae"], options=c.get("options"),
                 btbt=SILICON_HURKX if c.get("btbt") else None, presolve=c.get("presolve", False))
    contact, vs = c["bias"]
    for v in vs:
        ds.set_parameter(device=dev, name="{}_bias".format(contact), value=v)
        _solve(c["ae"])
    out = {k: _vals(dev, k) for k in ("Potential", "Electrons", "Holes", "TrapOccupancy0")}
    out["I"] = np.array([sum(ds.get_contact_current(device=dev, contact="lo", equation=e)
                             for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))])
    V.fresh()
    _R.clear(dev)
    return out


@pytest.mark.parametrize("case", sorted(DC_CASES))
def test_dc_identity_kinetic_equals_quasi_static(case):
    """K1-2. The same device solved with the quasi-static and the kinetic level: Potential to 1e-12 V, n, p,
    the terminal current and the occupancy to 1e-10 relative. DERIVATION: both are Newton-converged
    (relative update 1e-12) on the same discrete steady-state equations -- the kinetic row R_n - R_p = 0
    has the quasi-static occupancy as its root, so only the iteration path differs; a condition-number
    amplification of <= 1e2 (the plan's estimate) gives 1e-10. MEASURED max 1.4e-15 (n, p), 1.1e-16 V
    (psi), 1.2e-15 (occupancy), current bit-identical, on every case -- including the adversarial deep near-E_c level with
    tau_n/tau_p = 1e6, BGN and the valence reference, and the composition fixture (BGN + Auger +
    radiative + acceptor + BTBT at 0.5 V reverse). OLD CODE: TrapLevel has no `dynamic` (TypeError)."""
    qs, kn = _dc_state(case, False), _dc_state(case, True)
    assert np.max(np.abs(qs["Potential"] - kn["Potential"])) < 1e-12
    for k in ("Electrons", "Holes", "TrapOccupancy0", "I"):
        rel = np.max(np.abs(qs[k] - kn[k]) / np.maximum(np.abs(qs[k]), 1e-300))
        assert rel < 1e-10, (k, rel)


# ---------------------------------------------------------------------------------------------------
# K1-3: the D1c oracles re-run with kinetic levels (helpers of tests/test_traps.py, which seed kinetic
# levels through their _seed)
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("E", [-0.38, 0.3])
def test_d1c_lifetime_oracle_with_kinetic_levels(E):
    """K1-3 (a). D1c gate (a) -- the n-type slab lifetime against Schroder's closed form tau_p0 (1 + n1/n0)
    + tau_n0 p1/n0 -- with the level KINETIC (an acceptor-like level of N_t = 1e18 m^-3 = 1e-3 n0 -- a kinetic
    level holds charge, NeutralKineticTrapError; its compensation enters through the measured n0 the closed
    form takes -- built-in channel at 1e3 s). The SAME tolerance as D1c
    (1e-4; D1c MEASURED 2.1e-5 at +-0.4 eV, the dp/n0 residual): the physics is identical at DC.
    MEASURED: see the audit note (K1-3)."""
    import test_traps as TT
    tau_n0, tau_p0, G, L = 3e-10, 1e-10, 1e24, 60e-6
    lv = PT.TrapLevel(energy_eV=E, density_m3=1e18, kind="acceptor", tau_n0_s=tau_n0, tau_p0_s=tau_p0,
                      dynamic=True)
    dp, n0, p0 = TT._slab_tau("ktau", tau_n=1e3, tau_p=1e3, G=G, length=L, traps=[lv])
    _tau, tau_full = TT._tau_closed_form(E, tau_n0, tau_p0, n0, p0)
    Lp = math.sqrt(0.048 * V_T * tau_full)
    tau_meas = dp / G / (1.0 - 1.0 / math.cosh(0.5 * L / Lp))
    assert tau_meas == pytest.approx(tau_full, rel=1e-4)


@pytest.mark.parametrize("case", ["acceptor_in_n", "donor_in_p"])
def test_d1c_compensation_oracle_with_kinetic_levels(case):
    """K1-3 (b). D1c gate (b) -- trap compensation of the majority density against the scipy brentq root
    of the charge-neutrality residual with the FERMI occupancy (an oracle that does not use the SRH
    expressions) -- with the CHARGED level kinetic. Same tolerance as D1c (1e-6; D1c MEASURED 0 /
    1.1e-15). The equilibrium odds equals exp((E_F - E_t)/kT) at the brentq E_F (same budget)."""
    from scipy.optimize import brentq
    import test_traps as TT
    if case == "acceptor_in_n":
        nnet, lv = 1e22, PT.TrapLevel(energy_eV=0.33, density_m3=6e21, kind="acceptor", tau_n0_s=1e-9,
                                      tau_p0_s=1e-9, dynamic=True)
    else:
        nnet, lv = -1e22, PT.TrapLevel(energy_eV=-0.30, density_m3=4e21, kind="donor", tau_n0_s=1e-9,
                                       tau_p0_s=1e-9, dynamic=True)
    TT._slab("kq", length=4e-6, net_doping=nnet, tau_n=1e-7, tau_p=1e-7, ps_end=2e-9, ps_mid=5e-8,
             traps=[lv])
    TT._seed("kq", "r")
    TT._solve(1e10)
    x = _vals("kq", "x")
    ic = int(np.argmin(np.abs(x - 2e-6)))
    maj = _vals("kq", "Electrons" if nnet > 0 else "Holes")[ic]

    def neutral(ef):
        n, p = NI * math.exp(ef / V_T), NI * math.exp(-ef / V_T)
        f = 1.0 / (1.0 + math.exp((lv.energy_eV - ef) / V_T))
        qt = -lv.density_m3 * f if lv.kind == "acceptor" else lv.density_m3 * (1.0 - f)
        return (p - n + nnet + qt) / abs(nnet)

    ef = brentq(neutral, -0.6, 0.6, xtol=1e-15)
    ref = NI * math.exp(ef / V_T) if nnet > 0 else NI * math.exp(-ef / V_T)
    assert maj == pytest.approx(ref, rel=1e-6)
    assert abs(maj / abs(nnet) - 1.0) > 0.3                     # the kinetic level really compensates
    w = _vals("kq", "TrapW_trap0")                              # equilibrium odds = exp((E_F - E_t)/kT)
    assert w[ic] == pytest.approx(math.exp((ef - lv.energy_eV) / V_T), rel=1e-6)


# ---------------------------------------------------------------------------------------------------
# K1-7: defaults off (same tree, one process)
# ---------------------------------------------------------------------------------------------------
_LOGGED = ("node_model", "edge_model", "equation", "set_parameter", "node_solution", "contact_equation",
           "delete_node_model", "contact_node_model", "set_node_values")


@contextlib.contextmanager
def _call_log():
    """Every model / equation / parameter / solution-defining DEVSIM call (kwargs repr, device dropped)."""
    log, saved = [], {}

    def wrap(nm, fn):
        def inner(*a, **kw):
            log.append((nm, tuple(sorted((k, repr(v)) for k, v in kw.items() if k != "device"))))
            return fn(*a, **kw)
        return inner
    for nm in _LOGGED:
        saved[nm] = getattr(ds, nm)
        setattr(ds, nm, wrap(nm, saved[nm]))
    try:
        yield log
    finally:
        for nm, fn in saved.items():
            setattr(ds, nm, fn)


def _ab_state(explicit):
    """The forward diode with a charged level AND the 1-D MOS with split D_it, `dynamic` absent or False:
    (call log, solved arrays)."""
    import test_traps as TT
    kw = {"dynamic": False} if explicit else {}
    tr = PT.TrapLevel(energy_eV=0.15, density_m3=2e22, kind="acceptor", tau_n0_s=3e-9, tau_p0_s=5e-9, **kw)
    with _call_log() as log:
        dev = _diode("kab", (tr,), lines=DIODE_LINES, doping=DIODE_DOP, ae=1e10)
        ds.set_parameter(device=dev, name="lo_bias", value=-0.3)
        _solve(1e10)
    arr = {k: _vals(dev, k).tobytes() for k in ("Potential", "Electrons", "Holes", "TrapOccupancy0",
                                                  "PotentialNodeCharge", "ElectronGeneration")}
    V.fresh()
    _R.clear(dev)
    it = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, recombination=True, **kw)
    with _call_log() as log2:
        psib = TT._mos("kabm", it)
        p = TT._mos_sweep("kabm", psib, [0.2])
    arr["mos"] = p.tobytes() + _vals("kabm", "ItCharge0", "si").tobytes()
    TT._clear("kabm")
    return log + log2, arr


def test_defaults_off_same_tree_call_log_and_arrays():
    """K1-7 (i). `dynamic` absent against dynamic=False, in one process, on a forward diode with a charged
    bulk level and on the 1-D MOS with a split D_it (surface SRH on): the logged DEVSIM call sequence
    (every node/edge model, equation, parameter, solution, contact model and node-value call, with its
    arguments) and every solved array are IDENTICAL, and no kinetic model exists. (ii) the call-log
    digest gate (tests/test_bipolar_recombination_bgn_ionization.py) and (iii) the ee0ad04 numeric
    cross-version gate (tests/test_devsim_sg_flux.py) pass unchanged -- run, not re-recorded (audit note)."""
    la, aa = _ab_state(False)
    lb, ab = _ab_state(True)
    assert la == lb
    assert aa == ab
    names = {dict(kw).get("name", "") for _nm, kw in la}
    assert not any(("TrapW" in n or "ItW" in n or "kinetic" in n or "TrapEquation" in n) for n in names)


# ---------------------------------------------------------------------------------------------------
# Jacobian of the kinetic level (closed forms and the assembled matrix)
# ---------------------------------------------------------------------------------------------------
def _matrix(dev):
    import scipy.sparse as sp
    with V.quiet():
        m = ds.get_matrix_and_rhs(format="csc")["static"]
    A = sp.csc_matrix((np.asarray(m["av"], dtype=float), np.asarray(m["ai"]), np.asarray(m["ap"])))
    return A, np.asarray(m["rhs"], dtype=float)


def _numbers(dev, eq=None, var=None, region="r"):
    kw = {"equation": eq} if eq else {"variable": var}
    return np.asarray(ds.get_equation_numbers(device=dev, region=region, **kw), dtype=np.int64)


def test_kinetic_jacobian_w_column_closed_form_and_fd():
    """The w column of the assembled DC matrix on the forward diode with a kinetic ACCEPTOR level, against
    (1) the closed forms re-derived here, per interior node i (V_i from the node positions, g = 1/(1+w)):
        trap row     q V_i [-(n + n1')/tau_n0 - (p + p1')/tau_p0] g^2      electron row  q V_i (n + n1') g^2/tau_n0
        hole row     q V_i (p + p1') g^2/tau_p0                           Poisson row   q V_i N_t g^2
    (DEVSIM: ElectronGeneration = -q R_n, HoleGeneration = +q R_p, PotentialNodeCharge = -q(... - N_t f));
    budget 16 eps; and (2) central differences of the assembled residual in w (relative step h = 1e-3 on
    every node: the rows are linear in f, g and f = w/(1+w) has |f'''| delta^2/6 <= h^2 f' delta, so the
    truncation is <= h^2 of the column entry; round-off 4 eps R_r, R_r = Sum_k |J_rk x_k|), nothing off
    the node diagonal. The trap-row diagonal is strictly negative at EVERY node, contact nodes included
    (the well-posedness invariant). MEASURED: audit note.
    ADVERSARIAL: the mechanism of 4.0.2 without the w derivative (ElectronGeneration:TrapW_trap0 deleted)
    leaves the electron rows' w column empty: the FD gate misses by 100 %."""
    tr = PT.TrapLevel(energy_eV=0.15, density_m3=2e22, kind="acceptor", tau_n0_s=3e-9, tau_p0_s=5e-9,
                      dynamic=True)
    dev = _diode("kjac", (tr,), lines=DIODE_LINES, doping=DIODE_DOP, ae=1e10)
    ds.set_parameter(device=dev, name="lo_bias", value=-0.3)
    _solve(1e10)
    A, _ = _matrix(dev)
    cols = _numbers(dev, var="TrapW_trap0")
    x = _vals(dev, "x")
    vol = 0.5 * (x[2:] - x[:-2])
    n, p, w = _vals(dev, "Electrons"), _vals(dev, "Holes"), _vals(dev, "TrapW_trap0")
    g2 = (1.0 / (1.0 + w)) ** 2
    par = {k: float(ds.get_parameter(device=dev, region="r", name="trap0_" + k))
           for k in ("n1", "p1", "taun", "taup", "Nt")}
    sl = slice(1, -1)
    closed = {"TrapEquation_trap0": QE * vol * (-(n[sl] + par["n1"]) / par["taun"]
                                                - (p[sl] + par["p1"]) / par["taup"]) * g2[sl],
              "ElectronContinuityEquation": QE * vol * (n[sl] + par["n1"]) * g2[sl] / par["taun"],
              "HoleContinuityEquation": QE * vol * (p[sl] + par["p1"]) * g2[sl] / par["taup"],
              "PotentialEquation": QE * vol * par["Nt"] * g2[sl]}
    h = 1e-3
    rhs = {}
    for sgn in (1.0, -1.0):
        ds.set_node_values(device=dev, region="r", name="TrapW_trap0", values=list(w * (1 + sgn * h)))
        rhs[sgn] = _matrix(dev)[1]
    ds.set_node_values(device=dev, region="r", name="TrapW_trap0", values=list(w))
    fd = -(rhs[1.0] - rhs[-1.0]) / 2.0                         # rhs = -F: dF along delta = h w
    xs = np.zeros(A.shape[0])
    for v in ("Potential", "Electrons", "Holes", "TrapW_trap0"):
        xs[_numbers(dev, var=v)] = _vals(dev, v)
    R = abs(A).dot(np.abs(xs))
    delta = np.zeros(A.shape[0])
    delta[cols] = h * w
    Jd = A.dot(delta)
    absJd = abs(A).dot(np.abs(delta))
    for eq, cf in closed.items():
        rows = _numbers(dev, eq=eq)
        blk = A[rows[sl]][:, cols].toarray()
        j = np.arange(blk.shape[0])
        diag = blk[j, j + 1].copy()
        blk[j, j + 1] = 0.0
        assert np.max(np.abs(blk)) == 0.0, eq
        closed_eps = float(np.max(np.abs(diag - cf) / np.abs(cf)) / EPS)
        assert closed_eps <= 16.0, (eq, closed_eps)
        budget = h * h * absJd[rows] + 4.0 * EPS * R[rows]
        assert np.all(np.abs(fd[rows] - Jd[rows]) <= budget), eq
    tdiag = A[_numbers(dev, eq="TrapEquation_trap0")][:, cols].diagonal()
    assert np.all(tdiag < 0.0)                                  # pinned at every node (contacts too)
    # adversarial: version-1 composites (no w derivative of ElectronGeneration)
    ds.delete_node_model(device=dev, region="r", name="ElectronGeneration:TrapW_trap0")
    A2, _ = _matrix(dev)
    rows = _numbers(dev, eq="ElectronContinuityEquation")
    assert np.max(np.abs(A2[rows][:, cols].toarray())) == 0.0
    miss = np.abs(fd[rows] - A2.dot(delta)[rows])
    assert np.max(miss / (h * h * absJd[rows] + 4.0 * EPS * R[rows] + 1e-300)) > 1e3


# ---------------------------------------------------------------------------------------------------
# guards, refusals and the decision-39 advisory
# ---------------------------------------------------------------------------------------------------
def test_kinetic_noise_inventory_and_seed_range():
    """The inventory lists the electron / hole branches with their partner row, mechanism and (since K3)
    their EVENT rates, so the default noise_spectrum includes trap_n / trap_p (their FDT gates are
    tests/test_trap_noise.py); no fast-trap pair term exists for the kinetic level. A seed with |ln w| >
    U_LIMIT raises TrapOccupancyRangeError. REVISED 2026-11-06 (v0.15 FD-mode integration): fermi_dirac
    statistics no longer refuse a kinetic level -- it installs with the exact FD emission densities (its
    physics gates: tests/test_fd_mode_complete.py, the K-FD block)."""
    from dynameta.carriers import noise_analysis as NA
    tr = PT.TrapLevel(energy_eV=0.15, density_m3=2e21, kind="acceptor", tau_n0_s=3e-7, tau_p0_s=5e-7,
                      dynamic=True)
    dev = _diode("knoise", (tr,), lines=DIODE_LINES, doping=DIODE_DOP, ae=1e10, circuit=("hi",))
    terms = BP.recombination_terms(dev, "r")
    tn, tp = terms["node_electron"]["trap0"], terms["node_hole"]["trap0"]
    assert (tn.partner_equation, tn.partner_coeff, tn.mechanism) == ("TrapEquation_trap0", 1.0, "trap_n")
    assert (tp.partner_equation, tp.partner_coeff, tp.mechanism) == ("TrapEquation_trap0", -1.0, "trap_p")
    assert (tn.pair, tp.pair) == ("PTermN_trap0", "PTermP_trap0") and "trap0" not in terms["node"]
    assert {"trap_n", "trap_p"} <= set(NA.DEFAULT_MECHANISMS)
    with V.quiet():
        s = NA.noise_spectrum(dev, [1e3])
    assert s.by_mechanism["trap_n"][0] > 0.0 and s.by_mechanism["trap_p"][0] > 0.0
    assert PT.kinetic_trap_variables(dev, "r") == ["TrapW_trap0"]
    ds.set_parameter(device=dev, region="r", name="trap0_p1", value=1e300)     # w ~ tau_n0 p1/(tau_n0 p): |ln w| > 600
    with pytest.raises(PT.TrapOccupancyRangeError):
        PT.seed_kinetic_traps(dev)
    V.fresh()
    _R.clear(dev)
    m = "kfd_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=1e-8, tag="lo")
    ds.add_1d_mesh_line(mesh=m, pos=1e-7, ps=1e-8, tag="hi")
    ds.add_1d_contact(mesh=m, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=m, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="r", tag1="lo", tag2="hi")
    with V.quiet():
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device="kfd")
    ds.node_model(device="kfd", region="r", name="NetDoping", equation="1e24")
    lvn = PT.TrapLevel(energy_eV=0.1, density_m3=1e18, kind="acceptor", tau_n0_s=1e-9, tau_p0_s=1e-9,
                       dynamic=True)
    with V.quiet():
        BP.setup_bipolar_region("kfd", "r", eps_static=11.7, n_dos_m3=NC, n_i_m3=NI, mobility_n_m2Vs=0.135,
                                mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=True,
                                statistics="fermi_dirac", traps=(lvn,))
    assert PT.kinetic_trap_variables("kfd", "r") == ["TrapW_trap0"]
    assert "FDNiN" in ds.get_node_model_list(device="kfd", region="r")


def test_quasi_static_charged_trap_advisory():
    """Decision 39: an ssac / noise / transient solve with QUASI-STATIC CHARGED traps raises ONE
    AdvisoryWarning naming the region (it is not charge-conserving there); the same device with the level
    kinetic, or a recombination-only (neutral) quasi-static level, raises nothing (the suite runs with
    warnings as errors), and DC solves never do. The interface twin (split D_it advises, a fixed charge
    Q_f alone does not) is in the K2 section."""
    from dynameta.carriers import ac_analysis as AC
    from dynameta.carriers import extended_precision as XP     # transient_step's (and every analysis') solve
    qs = PT.TrapLevel(energy_eV=0.15, density_m3=2e21, kind="acceptor", tau_n0_s=3e-7, tau_p0_s=5e-7)
    dev = _diode("kadv", (qs,), lines=DIODE_LINES, doping=DIODE_DOP, ae=1e10, circuit=("hi",))
    assert PT.quasi_static_charged_traps(dev) == [("r", "QTerm_bulk_traps")]
    with pytest.warns(AdvisoryWarning, match="QUASI-STATIC CHARGED traps"):
        with V.quiet():
            AC.ssac_admittance([1e6])
    with pytest.warns(AdvisoryWarning, match="QUASI-STATIC CHARGED traps"):
        with V.quiet():
            XP.solve(type="transient_dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=30)
    V.fresh()
    _R.clear(dev)
    for traps in ((_kin(qs),), (PT.TrapLevel(energy_eV=0.15, tau_n0_s=3e-7, tau_p0_s=5e-7),)):
        dev = _diode("kadv2", traps, lines=DIODE_LINES, doping=DIODE_DOP, ae=1e10, circuit=("hi",))
        assert PT.quasi_static_charged_traps(dev) == []
        with V.quiet():
            AC.ssac_admittance([1e6])                           # no warning (errors in this suite)
        V.fresh()
        _R.clear(dev)


# ---------------------------------------------------------------------------------------------------
# K1-4 / K1-5 / K1-6: DLTS-like emission, terminal charge conservation in transient and in ssac
# ---------------------------------------------------------------------------------------------------
# p+ n diode, n side N_D = 1e22 m^-3; an ACCEPTOR level at E_c - 0.3 eV (conduction reference, the DLTS
# convention) with N_t = 1e20 (1 % of N_D) and c_n = c_p chosen so e_n = c_n N_c exp(-0.3/kT) = 1e6 /s.
# Bulk SRH off (1e3 s). Both contacts circuit-driven: 'hi' (n side) on V1, 'lo' on V2 (no AC part).
DL_ND, DL_NA, DL_NT, DL_EA, DL_EN = 1e22, 1e23, 1e20, 0.3, 1e6
DL_N1 = NC * math.exp(-DL_EA / V_T)
DL_CN = DL_EN / DL_N1
DL_LINES = [(0.0, 50e-9, "lo"), (1e-6, 4e-9, None), (4e-6, 50e-9, "hi")]
DL_DOP = "ifelse(x < 1e-6, {!r}, {!r})".format(-DL_NA, DL_ND)
DL_VR = 5.0


def _dl_device(dev, traps):
    return _diode(dev, traps, lines=DL_LINES, doping=DL_DOP, ae=1e13, re=1e-10, tau=1e3, circuit=("lo", "hi"),
                  presolve=True)


def _occ(dev, k):
    have = ds.get_node_model_list(device=dev, region="r")
    return float(_vals(dev, "TrapOccupancy0")[k]) if "TrapOccupancy0" in have else float("nan")


def _dl_level(dyn):
    return PT.TrapLevel(energy_eV=DL_EA, energy_reference="conduction", density_m3=DL_NT, kind="acceptor",
                        tau_n0_s=1.0 / (DL_CN * DL_NT), tau_p0_s=1.0 / (DL_CN * DL_NT), dynamic=dyn)


def _dl_bias(v_from, v_to, step=0.25):
    n = max(1, int(round(abs(v_to - v_from) / step)))
    for k in range(1, n + 1):
        ds.circuit_alter(name="V1", value=v_from + (v_to - v_from) * k / n)
        _solve(1e13, 1e-10)


def _currents():
    return [float(ds.get_circuit_node_value(node=s, solution="dcop")) for s in ("V1.I", "V2.I")]


def _dl_oracle(dev):
    """From the final (DL_VR reverse) DC state: the node k deep in the depletion region ((c_n n + c_p p)
    below 1e-6 of e_n + e_p), its relaxation rate e_n + e_p + c_n n + c_p p and f_inf; then back at 0 V,
    f0 = the steady-state SRH occupancy at k, written out here."""
    _dl_bias(0.0, DL_VR)
    n, p = _vals(dev, "Electrons"), _vals(dev, "Holes")
    p1 = NI * NI / DL_N1
    ok = np.where((DL_CN * n + DL_CN * p) / (DL_CN * DL_N1 + DL_CN * p1) < 1e-6)[0]
    k = int(ok[len(ok) // 2])
    rate = DL_CN * (n[k] + DL_N1) + DL_CN * (p[k] + p1)
    finf = (DL_CN * n[k] + DL_CN * p1) / rate
    _dl_bias(DL_VR, 0.0)
    n0, p0 = _vals(dev, "Electrons")[k], _vals(dev, "Holes")[k]
    f0 = (DL_CN * n0 + DL_CN * p1) / (DL_CN * (n0 + DL_N1) + DL_CN * (p0 + p1))
    return k, 1.0 / rate, finf, f0


def _dl_transient(dev, k, tau, h_frac, *, n_tau=5.0, t_r_frac=1e-5, n_r=20, steps=None, stall_ok=False):
    """transient_dc at 0 V; a BDF1 ramp of V1 to DL_VR over t_r = t_r_frac tau (n_r steps); BDF1 damping
    steps doubling from 1e-6 tau to 5e-4 tau (L-stable: they settle the dielectric-relaxation modes, ~5e-13 s,
    that the trapezoidal rule would carry undamped; each adds a local error (dt/tau)^2/2 <= 1.3e-7 on the
    slow trap mode); then the trapezoidal rule (DEVSIM transient_tr; DEVSIM 2.11.2's transient_bdf2 aborts
    with an FPE in the factorization on this device, MEASURED) at h = h_frac tau to n_tau tau. Returns
    (t, f_k, I1, I2, kind) arrays, t from the end of the ramp, kind 0 ramp / 1 damping / 2 TR. stall_ok:
    stop (instead of raising) at the first step whose Newton cannot meet the relative test -- the trap-free
    twin's terminal current decays to ~0, where a relative test is round-off over zero (transient.py class 2)."""
    # relative 1e-8: a circuit node near zero current never meets a tighter relative test; the Newton is
    # quadratic, so the device rows sit at round-off after its last iteration anyway
    def step(kind, dt):
        with V.quiet():
            r = ds.solve(type=kind, tdelta=dt, charge_error=1e30, absolute_error=1e30, relative_error=1e-8,
                         maximum_iterations=60, info=True)
        if not r["converged"] and not stall_ok:
            raise RuntimeError("{} step dt={} did not converge".format(kind, dt))
        return r["converged"]

    _solve(1e13, 1e-8, type="transient_dc")
    t_r = t_r_frac * tau
    t = -t_r
    rec = [(t, _occ(dev, k), *_currents(), 0)]
    for i in range(1, n_r + 1):
        ds.circuit_alter(name="V1", value=DL_VR * i / n_r)
        if not step("transient_bdf1", t_r / n_r):
            return np.array(rec).T
        t += t_r / n_r
        rec.append((t, _occ(dev, k), *_currents(), 0))
    dd = 1e-6 * tau
    while dd < 5.1e-4 * tau:
        if not step("transient_bdf1", dd):
            return np.array(rec).T
        t += dd
        rec.append((t, _occ(dev, k), *_currents(), 1))
        dd *= 2.0
    h = h_frac * tau
    for i in range(int(round(n_tau / h_frac)) if steps is None else steps):
        if not step("transient_tr", h):
            break
        t += h
        rec.append((t, _occ(dev, k), *_currents(), 2))
    return np.array(rec).T


@pytest.fixture(scope="module")
def dlts():
    """The kinetic DLTS device, its oracle quantities and two transients (h = tau/25, tau/50)."""
    V.fresh()
    dev = _dl_device("kdl", (_dl_level(True),))
    k, tau, finf, f0 = _dl_oracle(dev)
    runs = {}
    for i, hf in enumerate((1.0 / 25.0, 1.0 / 50.0)):
        if i:
            _dl_bias(DL_VR, 0.0)                                 # back to the 0 V DC state
        runs[hf] = _dl_transient(dev, k, tau, hf)
    yield k, tau, finf, f0, runs
    V.fresh()
    _R.clear(dev)


def _tr_error(t, h_frac, tau):
    """The trapezoidal rule's global relative error on y' = -y/tau after t/h steps: its amplification
    (1 - x/2)/(1 + x/2) = exp(-x - x^3/12 - O(x^5)), x = h/tau, so (f - f_or)/(f_or - f_inf) =
    exp(-(t/tau) x^2/12) - 1 (TR decays FASTER than the exponential: negative)."""
    return np.expm1(-(t / tau) * h_frac * h_frac / 12.0)


def test_dlts_emission_transient(dlts):
    """K1-4. The trap occupancy at a node deep in the reverse-bias depletion region after a 0 -> 5 V step
    relaxes as f(t) = f_inf + (f0 - f_inf) exp(-t/tau), 1/tau = e_n + e_p + c_n n + c_p p (Shockley-Read
    kinetics with Lang's e_n = c_n N_c exp(-(E_c - E_t)/kT)), f0 the 0 V steady state, f_inf and the rate
    from the 5 V DC state (all written out here). Error budget of the signed relative error
    err(t) = (f - f_or)/(f_or - f_inf):
      * the ramp: the bias rises over t_r = 1e-5 tau; the node empties for at most t_r of it, so the
        deficit at t = 0, off = (f(0) - f0)/(f0 - f_inf), lies in [-t_r/tau, 0] -- gated;
      * the trapezoidal rule's own global error E_TR(t) = expm1(-(t/tau)(h/tau)^2/12) (_tr_error;
        higher orders (t/tau)(h/tau)^4/80 < 1e-6), so err - off = E_TR to 2 % of E_TR (the off term is
        carried exactly, not bounded: it is common to every later time);
      * 1e-6: the BDF1 damping steps ((dt/tau)^2/2 <= 1.3e-7 each) and the capture terms c_n n + c_p p
        left at the node (< 1e-6 of the rate by the node choice).
    Convergence sub-gate (plan): (err - off) at t = 5 tau scales as h^2: ratio 4 +- 20 % for h = tau/25,
    tau/50. MEASURED: off = -9.14e-6; err(5 tau) - off = -6.666e-4 / -1.666e-4 (E_TR -6.664e-4 /
    -1.667e-4); ratio 4.00. OLD CODE: dynamic does not exist; the quasi-static level is at f_inf the
    moment the ramp ends (MEASURED f(0) = 1.53e-9 = f_inf against f0 = 0.617 -- a 100 % miss; asserted in
    the K1-5 test, which runs the quasi-static twin). ADVERSARIAL: the oracle with e_n 0.1 % high misses by
    5.0e-3 at 5 tau, 400x the budget there."""
    _k, tau, finf, f0, runs = dlts
    tail = {}
    for hf, (t, f, _i1, _i2, kind) in runs.items():
        off = (f[kind == 0][-1] - f0) / (f0 - finf)
        assert -1e-5 <= off <= 0.0, off
        post = kind == 2
        ora = finf + (f0 - finf) * np.exp(-t[post] / tau)
        err = (f[post] - ora) / (ora - finf) - off
        e_tr = _tr_error(t[post] - t[kind == 1][-1], hf, tau)
        budget = 0.02 * np.abs(e_tr) + 1e-6
        assert np.all(np.abs(err - e_tr) <= budget), (hf, np.max(np.abs(err - e_tr) / budget))
        tail[hf] = err[-1]
        bad = finf + (f0 - finf) * np.exp(-t[post] / tau * 1.001)       # adversarial: e_n + 0.1 %
        assert np.max(np.abs((f[post] - bad) / (bad - finf) - off - e_tr) / budget) > 10.0
    ratio = tail[1.0 / 25.0] / tail[1.0 / 50.0]
    assert abs(ratio / 4.0 - 1.0) < 0.2, ratio


def test_transient_kirchhoff_kinetic_vs_quasi_static(dlts):
    """K1-5. Terminal charge conservation in transient: with both contacts circuit-driven, I_V1 + I_V2 = 0
    at every step (Kirchhoff on the device: d/dt of the total device charge is zero because Poisson ties
    the contact charges to the space charge, and every space-charge change is a carrier flux -- including
    the trap's, through its capture terms). Normalised by max |I| over the run (the ramp's displacement
    current). TOLERANCE: 10x the floor measured on the trap-free twin over the same ramp + damping + TR
    steps (round-off of the converged circuit rows). The quasi-static charged level must miss by > 10x
    the tolerance (its trapped charge changes with no capture current). MEASURED: kinetic 8.7e-14
    (5 tau, both h), trap-free floor 8.7e-14 (tolerance 8.7e-13), quasi-static 9.6e-3 (1.1e10 x tol)."""
    k, tau, _f, _f0, runs = dlts
    t, f, i1, i2, _kind = runs[1.0 / 50.0]
    kin = float(np.max(np.abs(i1 + i2)) / np.max(np.abs(i1)))
    floor, qs = [], None
    for traps in (None, (_dl_level(False),)):
        dev = _dl_device("kdl2", traps)
        tt, ff, j1, j2, kd = _dl_transient(dev, k, tau, 1.0 / 50.0, steps=20, stall_ok=True)
        r = float(np.max(np.abs(j1 + j2)) / np.max(np.abs(j1)))
        if traps is None:
            floor.append(r)
        else:
            qs = r
            assert abs(ff[kd == 0][-1] - _f) < 1e-6                 # K1-4 old code: empty at once
        V.fresh()
        _R.clear("kdl2")
    tol = 10.0 * max(floor[0], 1e-15)
    assert kin <= tol, (kin, tol)
    assert qs > 10.0 * tol, (qs, tol)


def test_ssac_kirchhoff_kinetic_vs_quasi_static():
    """K1-6. Small-signal charge conservation: on the DLTS device at 5 V reverse with V1 driven (V2 AC-quiet),
    Y_V1 + Y_V2 = 0 from 1 Hz to 1 GHz, relative to |Y_V1|. TOLERANCE 10x the floor measured on the trap-
    free twin at the same frequencies (ssac round-off: one double factorization). The quasi-static charged
    level must miss by > 10x the tolerance near omega tau ~ 1 (its trapped charge is the steady-state
    function of n, p, so the ssac response sees no capture current). MEASURED (1 Hz ... 1 GHz): trap-free
    2.6e-2, 2.6e-5, 3.2e-7, 1.8e-8, 3.6e-9, 3.3e-11 (the ssac floor of a reverse-biased junction's tiny
    |Y| at low f, D2b); kinetic 4.4e-2, 4.0e-5, 3.0e-7, 2.7e-8, 2.9e-9, 1.9e-11 (<= 0.17 of the
    tolerance); quasi-static 9.8e-3 at every f >= 1 kHz (3.1e3 x the tolerance at 100 kHz)."""
    freqs = [1.0, 1e3, 1e5, 1e6, 1e7, 1e9]

    def kcl(dev):
        out = []
        for fr in freqs:
            with V.quiet():
                from dynameta.carriers import extended_precision as XP
                XP.solve(type="ac", frequency=fr)
            y = [complex(ds.get_circuit_node_value(node=s, solution="ssac_real"),
                         ds.get_circuit_node_value(node=s, solution="ssac_imag")) for s in ("V1.I", "V2.I")]
            out.append(abs(y[0] + y[1]) / abs(y[0]))
        return np.array(out)

    res = {}
    for name, traps in (("kin", (_dl_level(True),)), ("free", None), ("qs", (_dl_level(False),))):
        d2 = _dl_device("kdl3", traps)
        _dl_bias(0.0, DL_VR)
        with pytest.warns(AdvisoryWarning) if name == "qs" else contextlib.nullcontext():
            res[name] = kcl(d2)
        V.fresh()
        _R.clear("kdl3")
    tol = 10.0 * np.maximum(res["free"], 1e-15)
    assert np.all(res["kin"] <= tol), (res["kin"], tol)
    assert np.max(res["qs"] / tol) > 10.0, (res["qs"], tol)


# =====================================================================================================
# DEVSIM: interface bins (K2) -- the conductance method (Nicollian & Goetzberger, Bell Syst. Tech. J. 46,
# 1055 (1967), eqs. 4-6, 15-17, verified from the scanned issue by the plan)
# =====================================================================================================
# 1-D p-Si (N_A = 1e17 cm^-3, 0.3 um body, grounded ohmic) / 10 nm SiO2 / circuit-driven gate (V1).
# Bulk SRH off (1e3 s). One discrete interface level, majority (hole) exchange dominant by the bias and
# sigma_n / sigma_p = 1e-3.
EPS0 = 8.8541878128e-12                                  # re-typed (CODATA 2018)
MOS_NA, MOS_TSI, MOS_TOX, MOS_EPSSI, MOS_EPSOX = 1e23, 0.3e-6, 10e-9, 11.7, 3.9
MOS_COX = MOS_EPSOX * EPS0 / MOS_TOX
MOS_MUP = 0.048
IT_E, IT_N, IT_SN, IT_SP, IT_VTH = -0.20, 1e15, 1e-22, 1e-19, 1e5


def _mos_ac(dev, it, *, ps_i=0.05e-9):
    """The MOS above with interface traps `it` (or None) on si_ox; seeded and solved at flat band
    (gate source at psi_b). Returns psi_b."""
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import ac_analysis as AC
    m = dev + "_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=20e-9, tag="body")
    ds.add_1d_mesh_line(mesh=m, pos=MOS_TSI, ps=ps_i, tag="i")
    ds.add_1d_mesh_line(mesh=m, pos=MOS_TSI + MOS_TOX, ps=0.5e-9, tag="gate")
    ds.add_1d_contact(mesh=m, name="body", tag="body", material="metal")
    ds.add_1d_contact(mesh=m, name="gate", tag="gate", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="si", tag1="body", tag2="i")
    ds.add_1d_region(mesh=m, material="SiO2", region="ox", tag1="i", tag2="gate")
    ds.add_1d_interface(mesh=m, name="si_ox", tag="i")
    with V.quiet():
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=dev)
    ds.node_model(device=dev, region="si", name="NetDoping", equation=repr(-MOS_NA))
    with V.quiet():
        BP.setup_bipolar_region(dev, "si", eps_static=MOS_EPSSI, n_dos_m3=NC, n_i_m3=NI, mobility_n_m2Vs=0.135,
                                mobility_p_m2Vs=MOS_MUP, tau_n_s=1e3, tau_p_s=1e3, fd_enhancement=False)
        BP.setup_equilibrium_seed_models(dev, "si")
        PE.setup_dielectric_region(dev, "ox", MOS_EPSOX)
        PE.setup_interface(dev, "si_ox")
        BP.setup_contact_ohmic_bipolar(dev, "body")
        AC.setup_circuit_contact(dev, "gate")
        if it is not None:
            PT.install_interface_traps(dev, "si_ox", it)
    psib = -V_T * math.log(MOS_NA / NI)
    nox = len(_vals(dev, "x", "ox"))
    ds.set_node_values(device=dev, region="ox", name="Potential", values=[psib] * nox)
    ds.circuit_alter(name="V1", value=psib)
    _seed(dev, "si")
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(dev, ceq)
    _solve(1e-10, 1e-12)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(dev, ceq)
    _solve(1e12, 1e-10)
    return psib


def _mos_vg(psib, v_from, v_to, step=0.02):
    n = max(1, int(math.ceil(abs(v_to - v_from) / step - 1e-9)))
    for k in range(1, n + 1):
        ds.circuit_alter(name="V1", value=psib + v_from + (v_to - v_from) * k / n)
        _solve(1e12, 1e-10)


def _mos_surface(dev):
    """(psi_s, p_s, n_s) at the interface node, from the solved state."""
    pot = _vals(dev, "Potential", "si")
    return pot[-1] - pot[0], _vals(dev, "Holes", "si")[-1], _vals(dev, "Electrons", "si")[-1]


def _y_sc(freqs):
    """The semiconductor-side admittance per area, Y_sc = (1/Y_m - 1/(j omega C_ox))^-1 (N&G eq. 15
    inverted, C_ox = eps_ox/t_ox exact: the FV Laplace in a uniform oxide is exact), from the ssac Y_m."""
    from dynameta.carriers import ac_analysis as AC
    with V.quiet():
        f, C, G = AC.ssac_admittance(freqs)
    w = 2.0 * math.pi * np.asarray(f)
    ym = G + 1j * w * C
    return w, 1.0 / (1.0 / ym - 1.0 / (1j * w * MOS_COX))


def _ng_single_level(ps, ns, *, e_t=IT_E, n_it=IT_N, sn=IT_SN, sp=IT_SP):
    """N&G (4)-(6) mirrored for majority HOLES (the hole mirror tau = (1 - f0)/(c_p p_s0) is derived by
    the same linearisation, P3 1.1): (C_s, tau, f0, minority ratio), from the solved surface densities
    and the level's n1, p1 written out here."""
    n1, p1 = NI * math.exp(e_t / V_T), NI * math.exp(-e_t / V_T)
    cn, cp = sn * IT_VTH, sp * IT_VTH
    f0 = (cn * ns + cp * p1) / (cn * (ns + n1) + cp * (ps + p1))
    cs = QE * QE * n_it * f0 * (1.0 - f0) / (KB * 300.0)
    tau = (1.0 - f0) / (cp * ps)
    return cs, tau, f0, cn * (ns + n1) / (cp * (ps + p1))


def _sze_cd(psi):
    """Depletion capacitance -dQ_s/dpsi_s of p-Si (Sze's exact Boltzmann Q_s), per area."""
    p0, n0 = MOS_NA, NI * NI / MOS_NA
    ld = math.sqrt(MOS_EPSSI * EPS0 * V_T / (QE * p0))

    def qs(x):
        b = x / V_T
        F = math.sqrt(max(0.0, (math.expm1(-b) + b) + (n0 / p0) * (math.expm1(b) - b)))
        return -math.copysign(1.0, x) * math.sqrt(2.0) * MOS_EPSSI * EPS0 * V_T / ld * F
    h = 1e-6
    return -(qs(psi + h) - qs(psi - h)) / (2.0 * h)


def _hole_path_resistance(dev):
    """R_h = Int_body^interface dx/(q mu_p p(x)) [Ohm m^2] from the SOLVED hole profile (trapezoid on the
    nodes; the 0.05 nm interface mesh resolves the V_t/E_s-thick surface layer): the series resistance of
    the path that feeds the interface trap with holes (neutral body + depletion layer)."""
    x, p = _vals(dev, "x", "si"), _vals(dev, "Holes", "si")
    return float(trapz(1.0 / (QE * MOS_MUP * p), x))


def _conductance_budget(w, psi_s, ps, cs, tau, minority, r_h):
    """The K2-1 error budget per frequency (relative to G_p/omega), first order in each loss:
    (i)  the minority (electron) exchange the majority-carrier formula neglects: `minority`;
    (ii) the neutral-body series resistance R_b <= t_si/(q mu_p N_A), in series with the WHOLE stack:
         Y_sc -> Y_sc - R_b Y_sc^2, so Re is off by <= R_b |Y_sc|^2 <= R_b w^2 (C_D + C_s)^2 (C_D from
         Sze's closed form);
    (iii) the hole path to the trap, R_h (_hole_path_resistance, >= the depletion-layer part), in series
         with the TRAP branch only: Y_s -> Y_s - R_h Y_s^2, Re off by R_h |Y_s|^2 = R_h C_s Re(Y_s)/tau,
         i.e. R_h C_s/tau relative (MEASURED to dominate at omega tau << 1: 1.27e-5 of the 1.4e-5 total);
    (iv) dielectric relaxation, w eps/(q mu_p N_A) (< 1e-8 here)."""
    return _budget_ys(w, 1j * w * cs / (1.0 + 1j * w * tau), psi_s, minority, r_h)


def _budget_ys(w, ys, psi_s, minority, r_h):
    """The same budget for any trap admittance oracle Y_s (a single level, a binned sum, the continuum):
    minority + [R_b |j w C_D + Y_s|^2 + R_h |Y_s|^2]/Re(Y_s) + w tau_d (first order in R_b and R_h; for a
    single level |Y_s|^2/Re(Y_s) = C_s/tau)."""
    cd = _sze_cd(psi_s)
    rb = MOS_TSI / (QE * MOS_MUP * MOS_NA)
    taud = MOS_EPSSI * EPS0 / (QE * MOS_MUP * MOS_NA)
    return minority + (rb * abs(1j * w * cd + ys) ** 2 + r_h * abs(ys) ** 2) / ys.real + w * taud


def _it_single():
    """The kinetic single level (a function, so the module still collects on code without `dynamic`)."""
    return PT.InterfaceTraps(distribution="discrete", levels=((IT_E, IT_N),), charge="acceptor",
                             sigma_n_m2=IT_SN, sigma_p_m2=IT_SP, v_th_n_ms=IT_VTH, v_th_p_ms=IT_VTH,
                             dynamic=True)
MOS_VG = 0.45                                             # gate above flat band: depletion, psi_s ~ 0.22 V
WT = np.array([0.1, 0.3, 1.0, 3.0, 10.0])


@pytest.fixture(scope="module")
def mos_single():
    """The kinetic single-level MOS at V_G = 0.45 V: (psi_s, p_s, n_s, oracle tuple, omega, Y_sc) over
    omega tau = WT and the C_D-extraction point omega tau = 1e3, plus the LF point 1e-4."""
    V.fresh()
    psib = _mos_ac("kmos", _it_single())
    _mos_vg(psib, 0.0, MOS_VG)
    psi_s, ps, ns = _mos_surface("kmos")
    orc = _ng_single_level(ps, ns)
    tau = orc[1]
    wt = np.concatenate([WT, [1e3, 1e-4]])
    w, y = _y_sc(wt / tau / (2.0 * math.pi))
    occ = float(_vals("kmos", "ItOccupancy0_0", "si")[-1])
    r_h = _hole_path_resistance("kmos")
    V.fresh()
    _R.clear("kmos")
    yield psi_s, ps, ns, orc, w, y, occ, r_h


def test_conductance_method_single_level(mos_single):
    """K2-1 (+ K2-2). Single discrete interface level, kinetic: G_p/omega = Re(Y_sc)/omega against N&G
    eq. (17) mirrored for holes, C_s omega tau/(1 + omega^2 tau^2) with C_s = q^2 N_it f0 (1-f0)/kT and
    tau = (1 - f0)/(c_p p_s0) from the solved surface densities -- peak C_s/2 at omega tau = 1 -- at
    omega tau = 0.1 ... 10; and the dispersion C_p - C_D = C_s/(1 + omega^2 tau^2) (N&G eq. 16), C_p =
    Im(Y_sc)/omega, C_D = C_p at omega tau = 1e3 (less the oracle's C_s 1e-6 left there), C_D also against
    Sze's closed-form depletion capacitance to 1e-3 (mesh-limited, D1c slope gate 2.3e-4). TOLERANCE per
    point: 10x the computed budget (_conductance_budget: minority exchange, body and depletion-layer
    series resistance, dielectric relaxation); the C_p gate uses the same budget relative to C_s.
    MEASURED (V_G = 0.45 V: psi_s = 0.207 V, f0 = 0.405, C_s = 1.493e-3 F/m^2, tau = 1.77 us, minority
    1.3e-10): G_p/omega rel error 1.39e-5, 1.21e-5, 2.1e-6, -3.5e-6, 4.6e-5 at omega tau = 0.1, 0.3, 1, 3,
    10, against budgets 1.47e-5, 1.48e-5, 1.67e-5, 3.24e-5, 2.11e-4 (every point INSIDE its budget; the
    hole-path term R_h C_s/tau = 1.27e-5 dominates below the peak -- it predicts the omega -> 0 error to
    1 %; R_b dominates above). C_p dispersion rel error <= 1.9e-5. C_D = 2.1416e-3 F/m^2, Sze 2.2e-4 off
    (mesh). The level's occupancy diagnostic equals the oracle f0.
    OLD CODE: dynamic does not exist; the QUASI-STATIC level gives G_p = 0 to the ssac floor (a 100 %
    miss at the peak) and raises the decision-39 advisory: test_conductance_method_quasi_static_misses."""
    psi_s, ps, ns, (cs, tau, f0, minority), w, y, occ, r_h = mos_single
    assert occ == pytest.approx(f0, rel=1e-10)
    assert 0.05 < f0 < 0.95 and minority < 1e-6                 # the regime the formula is for
    wt = w * tau
    k = slice(0, WT.size)
    gpw = y.real[k] / w[k]
    ora = cs * wt[k] / (1.0 + wt[k] ** 2)
    budget = np.array([_conductance_budget(ww, psi_s, ps, cs, tau, minority, r_h) for ww in w[k]])
    rel = np.abs(gpw / ora - 1.0)
    assert np.all(rel <= 10.0 * budget), (rel, budget)
    assert gpw[2] == pytest.approx(0.5 * cs, rel=10.0 * budget[2])      # the peak: C_s/2 at omega tau = 1
    for j in (0, WT.size - 1):                                  # adversarial: the oracle's c_p + 1 %
        t1 = tau / 1.01
        bad = cs * w[j] * t1 / (1.0 + (w[j] * t1) ** 2)
        assert abs(gpw[j] / bad - 1.0) > 10.0 * budget[j], (j, gpw[j] / bad - 1.0, budget[j])   # breaks it
    cp = y.imag / w
    cd = cp[WT.size] - cs / (1.0 + wt[WT.size] ** 2)
    assert cd == pytest.approx(_sze_cd(psi_s), rel=1e-3)
    disp = (cp[k] - cd) / (cs / (1.0 + wt[k] ** 2)) - 1.0
    assert np.all(np.abs(disp) <= 10.0 * budget * (1.0 + wt[k] ** 2)), (disp, budget)


def test_conductance_low_frequency_limit_equals_quasi_static(mos_single):
    """K2-4 (the brief: the quasi-static limit equals today's result at omega -> 0). At omega tau = 1e-4 the
    kinetic level's C_p - C_D equals C_s (1 - (omega tau)^2) -- the LF trap capacitance q^2 N f0(1-f0)/kT --
    and the kinetic Y_sc equals the QUASI-STATIC model's ssac Y_sc at the same bias: the imaginary parts to
    (omega tau)^2 C_s/C_p + 1e-9 (ssac round-off of a capacitive Y; the DC states are identical, K1-2);
    the quasi-static Re Y is the R_b term only. MEASURED: C_p(kinetic)/C_p(quasi-static) - 1 = -4.108e-9 =
    -(omega tau)^2 C_s/C_p (4.11e-9) exactly; C_p - C_D = C_s to 1.2e-6 (budget 1.46e-5); the quasi-static
    G_p/omega at the peak is 2.1e-6 of C_s/2 (the old code's 100 % miss). The quasi-static run raises the
    decision-39 advisory (asserted)."""
    psi_s, ps, ns, (cs, tau, f0, minority), w, y, _occ, r_h = mos_single
    j = WT.size + 1
    V.fresh()
    qs_it = PT.InterfaceTraps(distribution="discrete", levels=((IT_E, IT_N),), charge="acceptor",
                              sigma_n_m2=IT_SN, sigma_p_m2=IT_SP, v_th_n_ms=IT_VTH, v_th_p_ms=IT_VTH)
    psib = _mos_ac("kmosq", qs_it)
    _mos_vg(psib, 0.0, MOS_VG)
    assert PT.quasi_static_charged_traps("kmosq") == [("si", "QTerm_it0")]
    with pytest.warns(AdvisoryWarning, match="QUASI-STATIC CHARGED traps"):
        wq, yq = _y_sc(np.array([w[j], w[2]]) / (2.0 * math.pi))
    cp_k, cp_q = y.imag[j] / w[j], yq.imag[0] / wq[0]
    assert abs(cp_k / cp_q - 1.0) <= (w[j] * tau) ** 2 * cs / cp_k + 1e-9, (cp_k, cp_q)
    cd = y.imag[WT.size] / w[WT.size] - cs / (1.0 + (w[WT.size] * tau) ** 2)
    lf = _conductance_budget(w[j], psi_s, ps, cs, tau, minority, r_h)   # the K2-1 budget at this omega
    assert (cp_k - cd) == pytest.approx(cs / (1.0 + (w[j] * tau) ** 2), rel=10.0 * lf)
    # old code at the peak (omega tau = 1): G_p/omega of the quasi-static level vs the N&G peak C_s/2
    assert abs(yq.real[1] / wq[1] / (0.5 * cs)) < 1e-2                # a 100 % miss: no trap conductance
    V.fresh()
    _R.clear("kmosq")


def test_conductance_minority_channel_is_starved():
    """FINDING (the plan's K2-1 adversarial, P3 1.5, does not refute): with sigma_n raised to 1.5e5 sigma_p,
    so that the electron channel's rate c_n (n_s + n1) is 3e-2 of the hole channel's, the majority-carrier
    formula STILL holds at omega tau = 0.1 (off the peak, where a tau error shows at first order) to the
    K2-1 tolerance: in depletion the minority electrons cannot follow the gate signal (no supply -- bulk
    SRH is off and the electron density at the surface is set by the trap exchange itself), so the
    electron channel adds no parallel conduction path. N&G's two-carrier time constant 1/tau = c_n (n_s +
    n1) + c_p (p_s + p1) (both channels in detailed balance, derived here) would predict a 3 % miss;
    MEASURED 1.4e-5. A refutation of K2-1 that DOES break it: the oracle's c_p perturbed by 1 % misses by
    ~1e-2 at omega tau = 0.1 and 10 (test_conductance_method_single_level, adversarial)."""
    V.fresh()
    sn = 1.5e5 * IT_SP
    it = PT.InterfaceTraps(distribution="discrete", levels=((IT_E, IT_N),), charge="acceptor",
                           sigma_n_m2=sn, sigma_p_m2=IT_SP, v_th_n_ms=IT_VTH, v_th_p_ms=IT_VTH, dynamic=True)
    psib = _mos_ac("kmosw", it)
    _mos_vg(psib, 0.0, MOS_VG)
    psi_s, ps, ns = _mos_surface("kmosw")
    cs, tau, f0, minority = _ng_single_level(ps, ns, sn=sn)
    assert 1e-2 < minority < 1e-1
    w, y = _y_sc(np.array([0.1]) / tau / (2.0 * math.pi))
    rel = abs(y.real[0] / w[0] / (cs * 0.1 / 1.01) - 1.0)
    clean = _conductance_budget(w[0], psi_s, ps, cs, tau, 0.0, _hole_path_resistance("kmosw"))
    assert rel <= 10.0 * clean, (rel, clean)
    tau2 = 1.0 / (1.0 / tau * (1.0 + minority))                      # the two-carrier time constant
    two = cs * w[0] * tau2 / (1.0 + (w[0] * tau2) ** 2)
    assert abs(y.real[0] / w[0] / two - 1.0) > 0.3 * minority       # ... which the device does NOT follow
    V.fresh()
    _R.clear("kmosw")


def test_kinetic_interface_budget_refusal_and_advisory_scope():
    """The decision-41 budget: (kinetic bins) x (region nodes) above max_kinetic_unknowns (default
    KINETIC_UNKNOWN_BUDGET) is REFUSED with the count, before any DEVSIM model is created; kinetic_bin_eV
    coarsens it. A FIXED charge alone is not a quasi-static trap (no advisory); the masked variables are
    1 off the interface and the off rows are diagonal."""
    V.fresh()
    big = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, dynamic=True)
    from dynameta.carriers import physics_equilibrium as PE
    m = "kbud_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=20e-9, tag="body")
    ds.add_1d_mesh_line(mesh=m, pos=MOS_TSI, ps=1e-9, tag="i")
    ds.add_1d_mesh_line(mesh=m, pos=MOS_TSI + MOS_TOX, ps=1e-9, tag="gate")
    for nm in ("body", "gate"):
        ds.add_1d_contact(mesh=m, name=nm, tag=nm, material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="si", tag1="body", tag2="i")
    ds.add_1d_region(mesh=m, material="SiO2", region="ox", tag1="i", tag2="gate")
    ds.add_1d_interface(mesh=m, name="si_ox", tag="i")
    with V.quiet():
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device="kbud")
    ds.node_model(device="kbud", region="si", name="NetDoping", equation=repr(-MOS_NA))
    with V.quiet():
        BP.setup_bipolar_region("kbud", "si", eps_static=MOS_EPSSI, n_dos_m3=NC, n_i_m3=NI,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=MOS_MUP, tau_n_s=1e3, tau_p_s=1e3,
                                fd_enhancement=False)
        PE.setup_dielectric_region("kbud", "ox", MOS_EPSOX)
        PE.setup_interface("kbud", "si_ox")
    nnode = len(_vals("kbud", "x", "si"))
    nbin = len(big.sheet_levels())
    before = sorted(ds.get_node_model_list(device="kbud", region="si"))
    with pytest.raises(ValueError, match=r"{} kinetic interface bins x {} nodes".format(nbin, nnode)):
        PT.install_interface_traps("kbud", "si_ox", big, max_kinetic_unknowns=nbin * nnode - 1)
    assert sorted(ds.get_node_model_list(device="kbud", region="si")) == before   # nothing was created
    coarse = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, dynamic=True, kinetic_bin_eV=0.2,
                               fixed_charge_C_m2=1e-4)
    info = PT.install_interface_traps("kbud", "si_ox", coarse, max_kinetic_unknowns=nbin * nnode - 1)
    assert len(info["kinetic"]) == len(coarse.sheet_levels()) == 6
    mask = _vals("kbud", "ItOffMask0", "si")
    area = _vals("kbud", "ItArea0", "si")
    assert np.array_equal(mask, (area == 0.0).astype(float))
    w0 = _vals("kbud", info["kinetic"][0], "si")
    assert np.all(w0[area == 0.0] == 1.0)                        # inert placeholder off the interface
    assert PT.quasi_static_charged_traps("kbud") == []             # kinetic bins + Q_f: no advisory
    V.fresh()
    _R.clear("kbud")


# ---------------------------------------------------------------------------------------------------
# K2-3: a uniform D_it continuum (N&G eqs. 18-20)
# ---------------------------------------------------------------------------------------------------
DIT, EG = 1e16, 1.12                                      # 1e12 cm^-2 eV^-1 across the Si gap, split at E_i
WTM = np.array([0.3, 1.0, 1.98, 5.0, 20.0])              # omega tau_m


def _continuum_oracles(w, ps, ns, levels):
    """For holes (majority), with E_F at the surface from p_s (E_F - E_i = -kT ln(p_s/n_i)):
    (a) the BINNED sum of N&G (17) over the simulated bins `levels` (E_k, N_k): Y_s = Sum_k j w C_k/(1 + j w
        tau_k), C_k = q^2 N_k f_k (1-f_k)/kT, tau_k = (1 - f_k)/(c_p p_s), f_k the SRH steady state;
    (b) the CONTINUUM integral of N&G (18) over the gap by scipy quad (exact Fermi f(E), tau(E));
    (c) N&G (20) mirrored for holes: G_p/w = (q D/(2 w tau_m)) ln(1 + w^2 tau_m^2), tau_m = 1/(c_p p_s)
        (derived here: with x = 1 - f, f (1-f) dE/kT = dx, so Y = j w q D Int_0^1 dx/(1 + j w tau_m x) =
        (q D/tau_m) ln(1 + j w tau_m)); exact for an infinite band, so it differs from (b) by the tails
        beyond the band ends, <= exp(-(E_F - E_v)/kT) + exp(-(E_c - E_F)/kT) of C_it.
    Returns (Y_a complex, G_b/w, G_c/w, tail fraction)."""
    from scipy.integrate import quad
    cn, cp = IT_SN * IT_VTH, IT_SP * IT_VTH
    efi = -V_T * math.log(ps / NI)
    tau_m = 1.0 / (cp * ps)
    ya = np.zeros(w.size, dtype=complex)
    for e, nk, _kind in levels:
        n1, p1 = NI * math.exp(e / V_T), NI * math.exp(-e / V_T)
        f = (cn * ns + cp * p1) / (cn * (ns + n1) + cp * (ps + p1))
        ck = QE * QE * nk * f * (1.0 - f) / (KB * 300.0)
        tk = (1.0 - f) / (cp * ps)
        ya += 1j * w * ck / (1.0 + 1j * w * tk)
    gb = []
    for ww in w:
        def integrand(e):
            f = 1.0 / (1.0 + math.exp((e - efi) / V_T))
            t = (1.0 - f) * tau_m
            return QE * DIT * f * (1.0 - f) / V_T * ww * t / (1.0 + (ww * t) ** 2)
        gb.append(quad(integrand, -0.5 * EG, 0.5 * EG, points=[efi], epsabs=0.0, epsrel=1e-12, limit=400)[0])
    gc = QE * DIT / (2.0 * w * tau_m) * np.log1p((w * tau_m) ** 2)
    tail = math.exp(-(efi + 0.5 * EG) / V_T) + math.exp(-(0.5 * EG - efi) / V_T)
    return ya, np.array(gb), gc, tail


@pytest.mark.slow
def test_conductance_method_continuum():
    """K2-3 (slow, ~15 s). Uniform D_it = 1e12 cm^-2 eV^-1 over the gap, every kinetic bin at kT (44 region-
    wide variables): G_p/omega = Re(Y_sc)/omega at omega tau_m = 0.3 ... 20 against
    (a) the binned sum of N&G (17) over the simulated bins -- exact for the discretised model: tolerance 10x
        the K2-1 budget with Y_s = the binned oracle;
    (b) bin convergence, devsim-free: the binned sum at kT, kT/2, kT/4 (InterfaceTraps.sheet_levels) against
        the continuum integral of N&G (18) by scipy quad (exact Fermi f(E), tau(E) = (1-f) tau_m) must
        converge at least as h^2 (midpoint): error ratio >= 4 x 0.8 per halving (plan). FINDING: it
        converges exponentially, but the CONDUCTANCE integrand's rate is half the occupancy's: the factor
        omega tau/(1 + omega^2 tau^2), tau = (1-f) tau_m, has poles where 1 - f = +-i/(omega tau_m), i.e. at
        Im E = (pi/2) kT (the Fermi function's own poles sit at pi kT), so the midpoint error is ~exp(-pi^2
        kT/dE) = 5e-5 at dE = kT and 3e-9 at kT/2 (the D1c estimate exp(-2 pi^2 kT/dE) holds for the LF
        charge only). MEASURED 1.2e-5 .. 4.0e-5 at kT, < 1e-9 at kT/2 and kT/4;
    (c) N&G (20) mirrored for holes: |quad - closed form| <= the band-end tail bound (pure arithmetic), and
        the simulated against (20) to 10x budget + the measured kT quadrature error (b) + the tail.
    The peak of (20) sits at omega tau_m = 1.98 with 0.4024 q D (scipy, checked). OLD CODE: quasi-static bins
    give G_p = 0. MEASURED (44 bins, psi_s = 0.264 V): device vs binned 1.3e-5, 9.8e-6, 5.3e-6, -2.5e-6,
    -1.1e-5 (budget 1.4e-5 .. 3.0e-5); quad vs (20) <= 2e-12 (tail bound 1.5e-7 of C_it); G_p/omega at
    omega tau_m = 1.98 = 0.40238 q D."""
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda x: -math.log1p(x * x) / (2.0 * x), bounds=(0.5, 5.0), method="bounded",
                          options={"xatol": 1e-10})
    assert res.x == pytest.approx(1.98, abs=5e-3) and -res.fun == pytest.approx(0.402, abs=5e-4)
    V.fresh()
    spec = dict(dit_eV_m2=DIT, band_gap_eV=EG, sigma_n_m2=IT_SN, sigma_p_m2=IT_SP, v_th_n_ms=IT_VTH,
                v_th_p_ms=IT_VTH, dynamic=True)
    it = PT.InterfaceTraps(**spec)
    psib = _mos_ac("kcont", it)
    _mos_vg(psib, 0.0, MOS_VG)
    psi_s, ps, ns = _mos_surface("kcont")
    r_h = _hole_path_resistance("kcont")
    tau_m = 1.0 / (IT_SP * IT_VTH * ps)
    w, y = _y_sc(WTM / tau_m / (2.0 * math.pi))
    V.fresh()
    _R.clear("kcont")
    ya, gb, gc, tail = _continuum_oracles(w, ps, ns, it.sheet_levels())
    efi = -V_T * math.log(ps / NI)
    minority = max(IT_SN * (ns + NI * math.exp(e / V_T)) / (IT_SP * ps)
                   for e, _n, _k in it.sheet_levels() if e < efi + 10.0 * V_T)
    budget = np.array([_budget_ys(ww, yy, psi_s, minority, r_h) for ww, yy in zip(w, ya)])
    gpw = y.real / w
    assert np.all(np.abs(gpw / (ya.real / w) - 1.0) <= 10.0 * budget), (gpw / (ya.real / w) - 1, budget)   # (a)
    errs = []
    for h in (V_T, 0.5 * V_T, 0.25 * V_T):                                                          # (b)
        lv = PT.InterfaceTraps(**dict(spec, kinetic_bin_eV=h)).sheet_levels()
        errs.append(np.abs(_continuum_oracles(w, ps, ns, lv)[0].real / w / gb - 1.0))
    for e1, e2 in zip(errs[:-1], errs[1:]):
        assert np.all((e1 / e2 >= 0.8 * 4.0) | (e1 < 1e-10)), (e1, e2)
    tail_rel = QE * DIT * tail * 0.5 / gc
    assert np.all(np.abs(gb / gc - 1.0) <= tail_rel), (gb / gc - 1.0, tail_rel)                    # (c)
    assert np.all(np.abs(gpw / gc - 1.0) <= 10.0 * budget + errs[0] + tail_rel)
    assert gpw[2] == pytest.approx(-res.fun * QE * DIT, rel=10.0 * budget[2] + errs[0][2] + tail_rel[2])


# ---------------------------------------------------------------------------------------------------
# K2-6: 2-D wiring (masked region-wide variables on a laterally uniform 2-D MOS) and the builders
# ---------------------------------------------------------------------------------------------------
def _mos_2d(dev, it, ylines, *, lx=60e-9):
    """The 1-D MOS above extruded to 2-D: 3 x-columns (lines at 0, lx/2, lx), the SAME y mesh lines as the
    1-D x mesh (every 1-D node a y line, ps = its larger neighbouring gap, so DEVSIM adds none: its 2-D
    line generator does NOT land on the graded 1-D positions by itself -- MEASURED, the line at t_si
    came out 0.1 nm low), body contact on the bottom edge, circuit gate (V1) on the top edge (DEVSIM's 2-D contacts
    sit on the boundary to inert pad regions, as validation/devsim_noise.bipolar_2d_resistor). Solved at
    flat band. Returns psi_b."""
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import ac_analysis as AC
    m = dev + "_m"
    pad = 5e-9
    ds.create_2d_mesh(mesh=m)
    for xx in (0.0, 0.5 * lx, lx):
        ds.add_2d_mesh_line(mesh=m, dir="x", pos=xx, ps=0.5 * lx)
    yl = np.concatenate([[-pad], np.asarray(ylines, dtype=float), [MOS_TSI + MOS_TOX + pad]])
    gap = np.diff(yl)
    for i, pos in enumerate(yl):
        ps = max(gap[i - 1] if i > 0 else 0.0, gap[i] if i < gap.size else 0.0)
        ds.add_2d_mesh_line(mesh=m, dir="y", pos=float(pos), ps=float(ps) * 1.01)
    ds.add_2d_region(mesh=m, material="Si", region="si", xl=0.0, xh=lx, yl=0.0, yh=MOS_TSI)
    ds.add_2d_region(mesh=m, material="SiO2", region="ox", xl=0.0, xh=lx, yl=MOS_TSI, yh=MOS_TSI + MOS_TOX)
    ds.add_2d_region(mesh=m, material="Si", region="pad_b", xl=0.0, xh=lx, yl=-pad, yh=0.0)
    ds.add_2d_region(mesh=m, material="SiO2", region="pad_t", xl=0.0, xh=lx, yl=MOS_TSI + MOS_TOX,
                     yh=MOS_TSI + MOS_TOX + pad)
    ds.add_2d_interface(mesh=m, name="si_ox", region0="si", region1="ox", xl=0.0, xh=lx, yl=MOS_TSI,
                        yh=MOS_TSI, bloat=1e-12)
    ds.add_2d_contact(mesh=m, name="body", region="si", material="metal", xl=0.0, xh=lx, yl=0.0, yh=0.0,
                      bloat=1e-10)
    ds.add_2d_contact(mesh=m, name="gate", region="ox", material="metal", xl=0.0, xh=lx,
                      yl=MOS_TSI + MOS_TOX, yh=MOS_TSI + MOS_TOX, bloat=1e-10)
    with V.quiet():
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=dev)
    ds.node_model(device=dev, region="si", name="NetDoping", equation=repr(-MOS_NA))
    with V.quiet():
        BP.setup_bipolar_region(dev, "si", eps_static=MOS_EPSSI, n_dos_m3=NC, n_i_m3=NI, mobility_n_m2Vs=0.135,
                                mobility_p_m2Vs=MOS_MUP, tau_n_s=1e3, tau_p_s=1e3, fd_enhancement=False)
        BP.setup_equilibrium_seed_models(dev, "si")
        PE.setup_dielectric_region(dev, "ox", MOS_EPSOX)
        PE.setup_interface(dev, "si_ox")
        BP.setup_contact_ohmic_bipolar(dev, "body")
        AC.setup_circuit_contact(dev, "gate")
        PT.install_interface_traps(dev, "si_ox", it)
    psib = -V_T * math.log(MOS_NA / NI)
    nox = len(_vals(dev, "x", "ox"))
    ds.set_node_values(device=dev, region="ox", name="Potential", values=[psib] * nox)
    ds.circuit_alter(name="V1", value=psib)
    _seed(dev, "si")
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(dev, ceq)
    _solve(1e-10, 1e-12)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(dev, ceq)
    _solve(1e12, 1e-10)
    return psib


def test_conductance_method_2d_wiring():
    """K2-6 (2-D; ~2 s). The single-level kinetic MOS (interface spacing 0.5 nm) and its extrusion to a
    laterally uniform 2-D device (3 columns, every 1-D node a y line): the masked region-wide bin variable,
    the half-edge interface weights (ItArea = A_node/NodeVolume) and the masked off-interface rows must
    reproduce the 1-D Y_sc per unit area at omega tau = 0.1, 1, 10 to 1e-6 relative (on this tensor mesh
    the per-node areas are exact and each column IS the 1-D problem: the difference is round-off; 1e-6 is
    the plan's figure). FINDING: DEVSIM's 2-D mesher drops a mesh line ~0.1 nm from its neighbour (the
    0.05 nm-graded 1-D mesh of K2-1 cannot be extruded: the t_si line vanished and with it the interface),
    hence the 0.5 nm spacing here. The 3-D (Stacked3DSpec) ssac twin is NOT built (audit note, open items).
    MEASURED: audit note (K2-6)."""
    from dynameta.carriers import ac_analysis as AC
    V.fresh()
    psib = _mos_ac("kmos1", _it_single(), ps_i=0.5e-9)
    _mos_vg(psib, 0.0, MOS_VG)
    nodes = np.unique(np.concatenate([_vals("kmos1", "x", "si"), _vals("kmos1", "x", "ox")]))
    _psi, ps, ns = _mos_surface("kmos1")
    tau = _ng_single_level(ps, ns)[1]
    freqs = np.array([0.1, 1.0, 10.0]) / tau / (2.0 * math.pi)
    w1, y1 = _y_sc(freqs)
    V.fresh()
    _R.clear("kmos1")
    lx = 60e-9
    psib = _mos_2d("kmos2", _it_single(), list(nodes), lx=lx)
    assert len(_vals("kmos2", "x", "si")) == 3 * int(np.sum(nodes <= MOS_TSI))
    _mos_vg(psib, 0.0, MOS_VG)
    pot = _vals("kmos2", "Potential", "si")
    yy = _vals("kmos2", "y", "si")
    assert np.max(pot[yy == yy.max()]) - np.min(pot[yy == yy.max()]) < 1e-12        # laterally uniform
    with V.quiet():
        f, C, G = AC.ssac_admittance(freqs)
    wm = 2.0 * math.pi * np.asarray(f)
    ym = (G + 1j * wm * C) / lx                                       # per unit area
    ysc = 1.0 / (1.0 / ym - 1.0 / (1j * wm * MOS_COX))
    assert np.all(np.abs(ysc / y1 - 1.0) < 1e-6), ysc / y1 - 1.0
    assert len(PT.kinetic_trap_variables("kmos2", "si")) == 1
    V.fresh()
    _R.clear("kmos2")


def test_builders_pass_kinetic_traps_through():
    """The builders carry the new flags with no new keyword (P3 1.3.1): LayeredDevsimBuilder(interface_traps=
    {...: InterfaceTraps(dynamic=True)}) + TransportModel(srh_traps=(TrapLevel(dynamic=True),)) build the
    kinetic variables and equations, the builder seeds them (seed_kinetic_traps after its carrier seed), and
    the zero-bias DC state equals the quasi-static build's to 1e-10 relative (K1-2's derivation)."""
    import test_traps as TT
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    out = {}
    for dyn in (False, True):
        lv = (PT.TrapLevel(energy_eV=0.2, density_m3=1e21, kind="acceptor", tau_n0_s=1e-8, tau_p0_s=1e-8,
                           dynamic=dyn),)
        # 0.1 eV bins (12): the builder's 5307-node region x 44 kT bins = 2.3e5 unknowns is above the
        # default budget (the refusal fires -- MEASURED); the quasi-static twin uses the SAME bins
        it = (PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, dynamic=True, kinetic_bin_eV=0.1) if dyn
              else PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, bin_eV=0.1))
        b = LayeredDevsimBuilder(TT._layered_bipolar_cap(lv), mesh_name="klay_m", device_name="klay",
                                 interface_traps={("ox", "si"): it})
        try:
            b.solve(BiasPoint({}, "eq"), rel_tol=1e-10)
            out[dyn] = {k: _vals("klay", k, "si") for k in ("Potential", "Electrons", "Holes")}
            if dyn:
                kv = PT.kinetic_trap_variables("klay", "si")
                assert "TrapW_trap0" in kv and sum(v.startswith("ItW") for v in kv) == len(it.sheet_levels())
                assert "TrapEquation_trap0" in ds.get_equation_list(device="klay", region="si")
        finally:
            b.teardown()
    for k in out[False]:
        rel = np.max(np.abs(out[True][k] - out[False][k]) / np.maximum(np.abs(out[False][k]), 1e-300))
        assert rel < 1e-10 if k != "Potential" else np.max(np.abs(out[True][k] - out[False][k])) < 1e-12, (k, rel)


# ---------------------------------------------------------------------------------------------------
# K2-5: small-step gate transient = the N&G Fig. 6 circuit
# ---------------------------------------------------------------------------------------------------
def _gate_step(vg0, dv, taus, *, h_frac=0.02, n_tau=5.0, stall_ok=False):
    """From the DC state at gate source vg0: transient_dc, a BDF1 ramp to vg0 + dv over 1e-6 tau*, BDF1
    damping steps (1e-6 .. 5e-4 tau*, as _dl_transient), then the trapezoidal rule at h = h_frac tau* to
    n_tau tau*. Returns (t_damp_end, t, i_g) of the TR samples with i_g = -I(V1) the current INTO the gate,
    or, stall_ok, (None, t, i_g) of the damping samples up to the first step whose Newton cannot meet the
    relative test (a current decayed to round-off -- the quasi-static twin)."""
    ds.circuit_alter(name="V1", value=vg0)
    _solve(1e12, 1e-10)
    _solve(1e12, 1e-8, type="transient_dc")

    def step(kind, dt):
        with V.quiet():
            r = ds.solve(type=kind, tdelta=dt, charge_error=1e30, absolute_error=1e30, relative_error=1e-6,
                         maximum_iterations=60, info=True)
        assert r["converged"] or stall_ok, (kind, dt)
        return r["converged"]

    t, tr = 0.0, 1e-6 * taus
    for i in range(1, 11):
        ds.circuit_alter(name="V1", value=vg0 + dv * i / 10)
        step("transient_bdf1", tr / 10)
        t += tr / 10
    dd, damp = 1e-6 * taus, []
    while dd < 5.1e-4 * taus:
        if not step("transient_bdf1", dd):
            break
        t += dd
        dd *= 2.0
        damp.append((t, -float(ds.get_circuit_node_value(node="V1.I", solution="dcop"))))
    if stall_ok:
        return (None,) + tuple(np.array(damp).T)
    t_d, rec = t, []
    for _ in range(int(round(n_tau / h_frac))):
        step("transient_tr", h_frac * taus)
        t += h_frac * taus
        rec.append((t, -float(ds.get_circuit_node_value(node="V1.I", solution="dcop"))))
    return (t_d,) + tuple(np.array(rec).T)


def test_gate_step_transient_is_the_ng_circuit():
    """K2-5 (~7 s). A small gate step on the kinetic single-level MOS (V_G = 0.45 V) against the N&G
    Fig. 6 circuit (C_ox in series with C_D || (R_s + C_s), R_s = tau/C_s; derived in the plan from N&G):
    after the step the trap branch relaxes with tau* = tau C_th/(C_s + C_th), C_th = C_ox + C_D, and
        i_g(t) = C_ox dV C_ox C_s / (C_th (C_th + C_s) tau*) exp(-t/tau*)
    (C_D from the ssac at omega tau = 1e3 less the oracle's C_s/(1 + 1e6) -- K2-2's extraction, the Sze
    check is there; C_s, tau from the DC surface state). The response is taken ANTISYMMETRIC, (i(+dV) -
    i(-dV))/2 with dV = 0.1 mV: FINDING, a single step's first-order nonlinearity is ~2.4 (dV/2V_t) --
    MEASURED symmetric/antisymmetric 4.7e-3 at 0.1 mV -- so the plan's single 1 mV step could not meet 2e-3;
    the antisymmetric part leaves (dV/V_t)^2 = 1.5e-5. GATE over [tau*/10, 5 tau*]: err = i/i_or - 1 equals
    the trapezoidal rule's own error E_TR(t) = expm1(-(t/tau*)(h/tau*)^2/12) to 2 % of it plus
    10 x [(1 + t/tau*)(R_h C_s/tau + R_b (C_D + C_s)^2/(C_s tau)) + (dV/V_t)^2 + 1e-6] (the series resistances
    of K2-1 shift tau* -- an error growing with t -- and the amplitude; the ramp 1e-6). MEASURED: max |err -
    E_TR| = 7.9e-5 (3.5e-5 at 5 tau*, against a 1e-3 tolerance there). OLD CODE: the quasi-static level has
    no R_s-C_s branch, so there is no tail: MEASURED 5.8 i_or at 3e-5 tau* (the instant trap capacitance
    charging through the hole path), 5.8e-2 i_or at the last damping step that converges (< 1e-3 tau*), then
    round-off -- a > 90 % miss (asserted < 10 %).
    ADVERSARIAL: the oracle's C_s + 1 % breaks it."""
    V.fresh()
    psib = _mos_ac("k25", _it_single())
    _mos_vg(psib, 0.0, MOS_VG)
    psi_s, ps, ns = _mos_surface("k25")
    cs, tau, _f0, _mino = _ng_single_level(ps, ns)
    r_h = _hole_path_resistance("k25")
    w, y = _y_sc(np.array([1e3]) / tau / (2.0 * math.pi))
    cd = y.imag[0] / w[0] - cs / (1.0 + 1e6)
    vg0, dv = psib + MOS_VG, 1e-4

    def oracle(cs_):
        cth = MOS_COX + cd
        taus_ = tau * cth / (cs_ + cth)
        return taus_, MOS_COX * dv * MOS_COX * cs_ / (cth * (cth + cs_)) / taus_

    taus, amp = oracle(cs)
    td, tp, ip = _gate_step(vg0, +dv, taus)
    _td, _tm, im = _gate_step(vg0, -dv, taus)
    ia = 0.5 * (ip - im)
    assert np.max(np.abs(0.5 * (ip + im) / ia)[tp >= 0.1 * taus]) < 1e-2   # odd response dominates
    sel = tp >= 0.1 * taus
    e_tr = _tr_error(tp - td, 0.02, taus)
    rb = MOS_TSI / (QE * MOS_MUP * MOS_NA)
    tol = 0.02 * np.abs(e_tr) + 10.0 * ((1.0 + tp / taus) * (r_h * cs / tau + rb * (cd + cs) ** 2 / (cs * tau))
                                        + (dv / V_T) ** 2 + 1e-6)
    err = ia / (amp * np.exp(-tp / taus)) - 1.0
    assert np.all(np.abs(err - e_tr)[sel] <= tol[sel]), np.max((np.abs(err - e_tr) / tol)[sel])
    t2, a2 = oracle(1.01 * cs)                                   # adversarial: C_s + 1 %
    bad = ia / (a2 * np.exp(-tp / t2)) - 1.0
    assert np.max((np.abs(bad - e_tr) - tol)[sel]) > 0.0
    V.fresh()
    _R.clear("k25")
    # old code: the quasi-static level (no capture branch) -- no exponential tail
    qs_it = PT.InterfaceTraps(distribution="discrete", levels=((IT_E, IT_N),), charge="acceptor",
                              sigma_n_m2=IT_SN, sigma_p_m2=IT_SP, v_th_n_ms=IT_VTH, v_th_p_ms=IT_VTH)
    psib = _mos_ac("k25q", qs_it)
    _mos_vg(psib, 0.0, MOS_VG)
    _tq, tq, iq = _gate_step(vg0, +dv, taus, stall_ok=True)
    _t2, tq2, iq2 = _gate_step(vg0, -dv, taus, stall_ok=True)
    n = min(tq.size, tq2.size)
    # the quasi-static trap charge is an INSTANT capacitance charged through the hole path (tau_RC = (R_h +
    # R_b)(C_s + C_D) ~ 5e-11 s): its current is gone within the damping steps -- its last converged sample
    # (the Newton then stalls on a current at round-off) is < 10 % of i_or while i_or(t) ~ amp there
    last = 0.5 * (iq[n - 1] - iq2[n - 1]) / (amp * math.exp(-tq[n - 1] / taus))
    assert tq[n - 1] < 1e-3 * taus and abs(last) < 0.1, (tq[n - 1] / taus, last)
    V.fresh()
    _R.clear("k25q")


# ---------------------------------------------------------------------------------------------------
# DC well-posedness (the floating-body hook, plan 4.4.2): the trap rows are candidates that are PINNED
# ---------------------------------------------------------------------------------------------------
def test_kinetic_trap_rows_are_pinned_candidates_of_the_well_posedness_check():
    """The invariant of K1-K2 made a gate (well_posedness LOCAL candidates, v0.15 K3/K4): every free node-only
    equation -- the kinetic trap rows TrapEquation_* (bulk) and ItEquation* (interface bins, masked) -- is a
    static candidate checked row by row, and the numeric pass REFUTES it (no all-zero equilibrated row: each
    row is pinned by its own capture and emission, d(R_n - R_p)/dw < 0, or by the off-interface mask), so a
    kinetic device passes. ADVERSARIAL: the same device with the trap row's rate terms removed (TrapNode_trap0
    and its derivatives = 0: no capture, no emission) raises IllPosedDCError naming TrapEquation_trap0 and
    every node; and a FLOATING kinetic body (Potential-only contacts: no carrier is pinned) raises with the
    trap rows in the conserved sum (the left null vector carries electron, hole and trap rows)."""
    import test_traps as TT
    from dynameta.carriers import well_posedness as WP
    from dynameta.carriers import physics_equilibrium as PE
    tr = PT.TrapLevel(energy_eV=0.15, density_m3=2e22, kind="acceptor", tau_n0_s=3e-9, tau_p0_s=5e-9,
                      dynamic=True)
    dev = _diode("kwp", (tr,), lines=DIODE_LINES, doping=DIODE_DOP, ae=1e10)
    rep = WP.check_dc_well_posed(dev)
    rows = [g for g in rep.candidates if g[0].side == "row"]
    assert [g[0].equation for g in rows] == ["TrapEquation_trap0"]
    assert rep.well_posed and rep.assembled
    ref = [m for m in rep.refuted if m.side == "row"]
    assert ref and ref[0].r > 1e6 * WP.TOL_SINGULAR                  # refuted by many decades
    nnode = len(_vals(dev, "x"))
    for m in [m for m in ds.get_node_model_list(device=dev, region="r") if m.startswith("TrapNode_trap0")]:
        ds.node_model(device=dev, region="r", name=m, equation="0")   # the row's rate terms removed
    with pytest.raises(WP.IllPosedDCError, match="TrapEquation_trap0") as ei:
        WP.check_dc_well_posed(dev)
    assert "at {} node(s)".format(nnode) in str(ei.value)
    V.fresh()
    _R.clear(dev)
    # the interface bins (masked region-wide rows) of the K2 MOS
    psib = TT._mos("kwpm", PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, dynamic=True, kinetic_bin_eV=0.2))
    rep = WP.check_dc_well_posed("kwpm")
    assert rep.well_posed and sum(g[0].equation.startswith("ItEquation") for g in rep.candidates) == 6
    assert psib < 0.0
    TT._clear("kwpm")
    # a floating kinetic body: Potential-only contacts, nothing pins a carrier
    m = "kfl_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=1e-8, tag="lo")
    ds.add_1d_mesh_line(mesh=m, pos=2e-7, ps=1e-8, tag="hi")
    ds.add_1d_contact(mesh=m, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=m, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="r", tag1="lo", tag2="hi")
    with V.quiet():
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device="kfl")
    ds.node_model(device="kfl", region="r", name="NetDoping", equation="1e22")
    with V.quiet():
        BP.setup_bipolar_region("kfl", "r", eps_static=11.7, n_dos_m3=NC, n_i_m3=NI, mobility_n_m2Vs=0.135,
                                mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=False,
                                traps=(tr,))
        BP.setup_equilibrium_seed_models("kfl", "r")
        PE.setup_contact("kfl", "lo")
        PE.setup_contact("kfl", "hi")
    _seed("kfl")
    with pytest.raises(WP.IllPosedDCError, match="TrapEquation_trap0") as ei:
        WP.check_dc_well_posed("kfl")
    assert "conserved" in str(ei.value)
    V.fresh()
    _R.clear("kfl")


# ---------------------------------------------------------------------------------------------------
# the DC form (physics_traps.kinetic_dc_form, inside dc_solve.solve_dc)
# ---------------------------------------------------------------------------------------------------
def test_dc_form_has_the_same_root_and_makes_the_gated_builder_converge(monkeypatch):
    """dc_solve.solve_dc runs its Newton with every kinetic level in the DC FORM (quasi-static rates and
    charge, w slaved to w_SS); every other solve sees the physical rows. GATES: (a) the root is the same --
    the forward diode solved by solve_dc and by a raw ds.solve (physical form) agree to K1-2's 1e-10 / 1e-12 V,
    and the switch is back at 0 afterwards; (b) the gated bipolar GaAs LayeredDevsimBuilder with kinetic interface bins
    (tests/_trap_noise_child.design) converges and its zero-bias state equals the quasi-static build's to
    1e-10. ADVERSARIAL: with the DC form disabled (kinetic_dc_form a no-op) the same builder's Newton FAILS
    (MEASURED: w of the bins near the valence edge reached 1e190 and the carriers diverged in 60 iterations;
    without the slaving, in the earlier linear (1 + w) form, the factorization hit an FPE)."""
    import contextlib as _cl
    import sys as _sys
    from dynameta.carriers import dc_solve as DC
    tr = PT.TrapLevel(energy_eV=0.15, density_m3=2e22, kind="acceptor", tau_n0_s=3e-9, tau_p0_s=5e-9,
                      dynamic=True)
    out = {}
    for how in ("raw", "solve_dc"):
        dev = _diode("kdf", (tr,), lines=DIODE_LINES, doping=DIODE_DOP, ae=1e10)
        for v in (-0.1, -0.2, -0.3):
            ds.set_parameter(device=dev, name="lo_bias", value=v)
            if how == "raw":
                _solve(1e10)
            else:
                with V.quiet():
                    DC.solve_dc(dev, abs_tol=1e10, rel_tol=1e-12, max_iter=100)
        assert float(ds.get_parameter(device=dev, region="r", name=PT.KINETIC_DC_PARAMETER)) == 0.0
        out[how] = {k: _vals(dev, k) for k in ("Potential", "Electrons", "Holes", "TrapOccupancy0")}
        V.fresh()
        _R.clear(dev)
    assert np.max(np.abs(out["raw"]["Potential"] - out["solve_dc"]["Potential"])) < 1e-12
    for k in ("Electrons", "Holes", "TrapOccupancy0"):
        assert np.max(np.abs(out["raw"][k] / out["solve_dc"][k] - 1.0)) < 1e-10, k
    # (b) the gated builder
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _trap_noise_child as TN
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    states = {}
    for dyn in (False, True):
        it = (PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.42, sigma_n_m2=1e-19, sigma_p_m2=1e-19,
                                dynamic=True, kinetic_bin_eV=0.2) if dyn else
              PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.42, sigma_n_m2=1e-19, sigma_p_m2=1e-19,
                                bin_eV=0.2))
        b = LayeredDevsimBuilder(TN.design(), mesh_name="kdfb_m", device_name="kdfb",
                                 interface_traps={("gaas", "upper_hfo2"): it})
        try:
            with V.quiet():
                b.solve(BiasPoint({}, "eq"), rel_tol=1e-10)
            states[dyn] = {k: _vals("kdfb", k, "gaas") for k in ("Potential", "Electrons", "Holes")}
        finally:
            b.teardown()
    assert np.max(np.abs(states[True]["Potential"] - states[False]["Potential"])) < 1e-12
    for k in ("Electrons", "Holes"):
        assert np.max(np.abs(states[True][k] / states[False][k] - 1.0)) < 1e-10, k
    monkeypatch.setattr(PT, "kinetic_dc_form", lambda *a, **k: _cl.nullcontext())
    b = LayeredDevsimBuilder(TN.design(), mesh_name="kdfc_m", device_name="kdfc",
                             interface_traps={("gaas", "upper_hfo2"): PT.InterfaceTraps(
                                 dit_eV_m2=1e16, band_gap_eV=1.42, sigma_n_m2=1e-19, sigma_p_m2=1e-19,
                                 dynamic=True, kinetic_bin_eV=0.2)})
    try:
        with pytest.raises(Exception, match="onvergence|factoriz"):
            with V.quiet():
                b.solve(BiasPoint({}, "eq"))
    finally:
        b.teardown()


def test_transient_integrated_charge_through_the_n_contact():
    """K1-5, the integrated-charge half. The charge that flows through the n-side contact (V1) during the 0 -> 5 V
    DLTS transient, integrated with the integrators' own rules (rectangle over the BDF1 ramp / damping steps,
    trapezoid over the TR steps -- exact for the discrete scheme, so no quadrature error enters), against
    q (Delta N_e + Delta N_t) from TWO INDEPENDENT DC solves (0 V and 5 V): every electron removed from the
    device -- free electrons of the widened depletion region AND the electrons the traps emit -- leaves through
    the n contact (the holes through the p contact). Budget: the emission not finished at t_end = 12 tau,
    e^-12 |q Delta N_t| = 6e-8 of the total, the minority hole current at the n contact (bulk SRH off) and the
    contact-node displacement (quasi-neutral ohmic contact), both below that; gated at 1e-6. MEASURED: equal
    to the 7 printed digits (-8.475073e-4 C/m^2). OLD CODE: the quasi-static level passes q Delta N_e only
    (-8.392464e-4): its trap electrons never cross the contact -- a 9.7e-3 miss (the K1-5 Kirchhoff miss)."""
    q = 1.602176634e-19
    res = {}
    for dyn in (True, False):
        V.fresh()
        dev = _dl_device("kqi", (_dl_level(dyn),))
        vol = _vals(dev, "NodeVolume")
        k, tau, _finf, _f0 = (_dl_oracle(dev) if dyn else (res["k"], res["tau"], 0.0, 0.0))
        s0 = (_vals(dev, "Electrons"), _vals(dev, "TrapOccupancy0"))
        _dl_bias(0.0, DL_VR)
        s1 = (_vals(dev, "Electrons"), _vals(dev, "TrapOccupancy0"))
        _dl_bias(DL_VR, 0.0)
        t, _f, i1, _i2, kind = _dl_transient(dev, k, tau, 1.0 / 25.0, n_tau=12.0, stall_ok=not dyn)
        dt = np.diff(t)
        integ = float(np.sum(np.where(kind[1:] < 2, dt * i1[1:], 0.5 * dt * (i1[1:] + i1[:-1]))))
        dne = float(np.sum(vol * (s1[0] - s0[0])))
        dnt = DL_NT * float(np.sum(vol * (s1[1] - s0[1])))
        res.update({"k": k, "tau": tau, dyn: (integ, q * (dne + dnt), q * dne)})
        V.fresh()
        _R.clear(dev)
    integ, oracle, free_only = res[True]
    assert abs(integ / oracle - 1.0) < 1e-6, (integ, oracle)
    assert abs(free_only / oracle - 1.0) > 1e-3                   # the traps' share is load-bearing
    q_integ = res[False][0]
    assert abs(q_integ / res[False][1] - 1.0) > 1e-3, (q_integ, res[False][1])


def _gate_to_circuit(dev, gate_v):
    """Repoint the Potential-only contact 'gate' to a circuit-driven one (V1 on node 'vac') at gate_v."""
    from dynameta.carriers import ac_analysis as AC
    ds.delete_contact_equation(device=dev, contact="gate", name="PotentialEquation")
    _R.forget(dev, "PotentialEquation", loc="gate", scope="contact")
    AC.setup_circuit_contact(dev, "gate")
    ds.circuit_alter(name="V1", value=gate_v)


def test_conductance_method_3d_stacked_spec_equals_1d():
    """K2-6 in 3-D. Stacked3DSpec(physics='bipolar_dd', interface_traps=<the K2-1 kinetic single level>) -- the
    builder's own tensor (Kuhn) mesh, bipolar region setup, interface install and seed -- a p-Si 1e17 cm^-3,
    200 nm / 10 nm SiO2 cap of 4 x 4 nm, solved at V_G = 0.45 V, its gate then made circuit-driven at the
    same potential: (a) Y_sc per area equals that of a 1-D device built on the 3-D mesh's z-lines with the
    SAME region physics (Devsim3DEquilibrium._setup_bipolar_semi) and trap spec, at omega tau = 0.1, 1, 10, to
    1e-6 (on the Kuhn mesh a laterally uniform stack reduces to the 1-D discretization on the same z-lines,
    devsim_3d._build_mesh_tensor; the circumcentric interface shares are exact: the difference is round-off,
    the plan's 1e-6); (b) G_p/omega against N&G (17) to 10x the K2-1 budget (R_h from the 3-D column's hole
    profile). MEASURED: audit note."""
    import types
    from dynameta.carriers import ac_analysis as AC
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    from dynameta.constants import M_E
    from dynameta.sweep import BiasPoint
    spec = Stacked3DSpec(semi_material="Si", oxide_material="SiO2", lateral_m=4e-9, semi_thk_m=200e-9,
                         oxide_thk_m=MOS_TOX, n_bg_m3=MOS_NA, eps_semi=MOS_EPSSI, eps_oxide=MOS_EPSOX,
                         dos_mass_kg=1.08 * M_E, physics="bipolar_dd", n_i_m3=NI, mobility_m2Vs=0.135,
                         mobility_p_m2Vs=MOS_MUP, tau_srh_s=1e3, acceptor=True, interface_traps=_it_single(),
                         mesh_min_nm=0.5, mesh_max_nm=5.0)
    sol = Devsim3DEquilibrium(spec, mesh_name="k3d_m", device_name="k3d")
    try:
        sol.build_device()
        with V.quiet():
            sol.solve(BiasPoint(label="g", voltages={"gate": MOS_VG}))
        dev = sol.device
        gate_v = float(ds.get_parameter(device=dev, name="gate_bias"))
        _gate_to_circuit(dev, gate_v)
        _solve(1e12, 1e-8)
        z = np.concatenate([_vals(dev, "z", "semi"), _vals(dev, "z", "oxide")])
        x, yv = _vals(dev, "x", "semi"), _vals(dev, "y", "semi")
        col = (x == x.min()) & (yv == yv.min())
        zs = _vals(dev, "z", "semi")[col]
        o = np.argsort(zs)
        p_col = _vals(dev, "Holes", "semi")[col][o]
        pot_col = _vals(dev, "Potential", "semi")[col][o]
        psi_s, ps, ns = pot_col[-1] - pot_col[0], p_col[-1], _vals(dev, "Electrons", "semi")[col][o][-1]
        cs, tau, _f0, mino = _ng_single_level(ps, ns)
        r_h = float(trapz(1.0 / (QE * MOS_MUP * p_col), zs[o]))
        freqs = np.array([0.1, 1.0, 10.0]) / tau / (2 * math.pi)
        with V.quiet():
            f3, C3, G3 = AC.ssac_admittance(freqs)
        w = 2 * math.pi * np.asarray(f3)
        ym3 = (G3 + 1j * w * C3) / spec.lateral_m ** 2
    finally:
        sol.teardown()
        V.fresh()
        _R.clear("k3d")
    zl = np.unique(np.round(z, 18))
    zi = float(np.max(np.unique(np.round(zs, 18))))
    # the 1-D twin on the same z-lines, with the 3-D builder's own bipolar region setup
    m = "k1d_m"
    ds.create_1d_mesh(mesh=m)
    for i, pos in enumerate(zl):
        tag = "body" if i == 0 else ("i" if pos == zi else ("gate" if i == zl.size - 1 else None))
        ds.add_1d_mesh_line(mesh=m, pos=float(pos), ps=1e-6, **({"tag": tag} if tag else {}))
    ds.add_1d_contact(mesh=m, name="body", tag="body", material="metal")
    ds.add_1d_contact(mesh=m, name="gate", tag="gate", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="semi", tag1="body", tag2="i")
    ds.add_1d_region(mesh=m, material="SiO2", region="oxide", tag1="i", tag2="gate")
    ds.add_1d_interface(mesh=m, name="semi_oxide", tag="i")
    with V.quiet():
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device="k1d")
        Devsim3DEquilibrium._setup_bipolar_semi(types.SimpleNamespace(spec=spec, device="k1d"), "semi",
                                                MOS_NA, MOS_EPSSI)
        PE.setup_dielectric_region("k1d", "oxide", MOS_EPSOX)
        PE.setup_interface("k1d", "semi_oxide")
        PT.install_interface_traps("k1d", "semi_oxide", _it_single())
        BP.setup_contact_ohmic_bipolar("k1d", "body")
        AC.setup_circuit_contact("k1d", "gate")
    assert len(_vals("k1d", "x", "semi")) * len(set(zip(x.tolist(), yv.tolist()))) == len(x)   # same z-lines
    psib = float(np.min(pot_col))
    ds.set_node_values(device="k1d", region="oxide", name="Potential",
                       values=[psib] * len(_vals("k1d", "x", "oxide")))
    ds.circuit_alter(name="V1", value=psib)
    _seed("k1d", "semi")
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name("k1d", ceq)
    _solve(1e-10, 1e-12)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name("k1d", ceq)
    _solve(1e12, 1e-10)
    for v in np.linspace(psib, gate_v, 30)[1:]:
        ds.circuit_alter(name="V1", value=float(v))
        _solve(1e12, 1e-10)
    with V.quiet():
        f1, C1, G1 = AC.ssac_admittance(freqs)
    ym1 = G1 + 1j * w * C1
    V.fresh()
    _R.clear("k1d")
    assert np.all(np.abs(ym3 / ym1 - 1.0) < 1e-6), ym3 / ym1 - 1.0
    ysc = 1.0 / (1.0 / ym3 - 1.0 / (1j * w * MOS_COX))
    wt = w * tau
    ora = cs * wt / (1 + wt ** 2)
    budget = np.array([_budget_ys(ww, 1j * ww * cs / (1 + 1j * ww * tau), psi_s, mino, r_h) for ww in w])
    assert np.all(np.abs(ysc.real / w / ora - 1.0) <= 10.0 * budget), (ysc.real / w / ora - 1.0, budget)


def test_a_kinetic_level_holds_charge_and_conserves_it_in_ssac():
    """A kinetic level is acceptor- or donor-like: kind='neutral' with dynamic=True is REFUSED
    (NeutralKineticTrapError), because a level whose occupancy is a dynamic variable changes its charge by -q per
    captured electron. GATES at 1 MHz (the DLTS device of K1-4, 0 V, both contacts circuit-driven, its level at
    E_c - 0.3 eV with e_n = 1e6 /s, so it responds at 1 MHz): (a) Kirchhoff Y_V1 + Y_V2 = 0 to 10x the trap-free
    ssac floor for the acceptor-like AND the donor-like kinetic level; (b) the FDT on the gate-B diode with the
    same charge kinds kinetic, S_I = 4kT Re Y to 1e-4 at 1 Hz and 1 MHz. MEASURED: floor 1.3e-8; acceptor
    3.5e-10, donor 4.3e-9; FDT acceptor 3.3e-8 at 1 MHz. FAILS ON 862493a (the refusal does not exist; the
    neutral kinetic level it accepted broke Kirchhoff by 3.5e-3 of Y at 1 MHz -- 2.8e5 x the floor -- and the
    FDT by +0.39: it stored N_t f in its trap row's time term but not in Poisson). Both the DC state and 1 Hz were
    unaffected, which is why only the small-signal / noise gates saw it.)"""
    from dynameta.carriers import extended_precision as XP
    with pytest.raises(PT.NeutralKineticTrapError):
        PT.TrapLevel(energy_eV=DL_EA, energy_reference="conduction", density_m3=DL_NT, kind="neutral",
                     tau_n0_s=1.0 / (DL_CN * DL_NT), tau_p0_s=1.0 / (DL_CN * DL_NT), dynamic=True)

    def kcl(traps):
        V.fresh()
        dev = _dl_device("kch", traps)
        with V.quiet():
            XP.solve(type="ac", frequency=1e6)
        y = [complex(ds.get_circuit_node_value(node=s, solution="ssac_real"),
                     ds.get_circuit_node_value(node=s, solution="ssac_imag")) for s in ("V1.I", "V2.I")]
        V.fresh()
        _R.clear(dev)
        return abs(y[0] + y[1]) / abs(y[0])
    floor = kcl(None)
    for kind in ("acceptor", "donor"):
        lv = PT.TrapLevel(energy_eV=DL_EA, energy_reference="conduction", density_m3=DL_NT, kind=kind,
                          tau_n0_s=1.0 / (DL_CN * DL_NT), tau_p0_s=1.0 / (DL_CN * DL_NT), dynamic=True)
        k = kcl((lv,))
        assert k <= 10.0 * floor, (kind, k, floor)
        V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, traps=(dataclasses_replace(
            lv, energy_eV=0.15, energy_reference="intrinsic", density_m3=2e21, tau_n0_s=3e-7, tau_p0_s=5e-7),))
        sn = V.noise([1.0, 1e6])
        Y = V.ssac_Y([1.0, 1e6])
        r = sn.S / (4 * 1.380649e-23 * 300.0 * Y.real) - 1.0
        assert np.max(np.abs(r)) < 1e-4, (kind, r)
        V.fresh()


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)
