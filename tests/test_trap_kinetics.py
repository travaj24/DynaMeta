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
import math

import numpy as np
import pytest

from dynameta.carriers import physics_traps as PT

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
    """The new spec fields: TrapLevel(dynamic=True) needs a finite density_m3 > 0 (also when neutral);
    `dynamic` must be a bool; InterfaceTraps(dynamic=True) needs recombination=True; kinetic_bin_eV only
    with dynamic=True, > 0, and it sets the bin width of a dynamic spec. Defaults: dynamic=False compares
    EQUAL to the shipped spec, and sheet_levels is unchanged for it."""
    with pytest.raises(ValueError, match="density_m3"):
        PT.TrapLevel(energy_eV=0.1, kind="neutral", tau_n0_s=1e-9, tau_p0_s=1e-9, dynamic=True)
    with pytest.raises(TypeError):
        PT.TrapLevel(energy_eV=0.1, kind="neutral", tau_n0_s=1e-9, tau_p0_s=1e-9, dynamic=1)
    lv = PT.TrapLevel(energy_eV=0.1, density_m3=1e18, kind="neutral", tau_n0_s=1e-9, tau_p0_s=1e-9,
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
    "neutral": dict(trap=PT.TrapLevel(energy_eV=0.3, density_m3=1e21, kind="neutral", tau_n0_s=3e-9,
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
    + tau_n0 p1/n0 -- with the level KINETIC (a recombination-only TrapLevel, N_t = 1e18 m^-3 = 1e-3 n0
    so its stored charge stays out of the balance, built-in channel at 1e3 s). The SAME tolerance as D1c
    (1e-4; D1c MEASURED 2.1e-5 at +-0.4 eV, the dp/n0 residual): the physics is identical at DC.
    MEASURED: see the audit note (K1-3)."""
    import test_traps as TT
    tau_n0, tau_p0, G, L = 3e-10, 1e-10, 1e24, 60e-6
    lv = PT.TrapLevel(energy_eV=E, density_m3=1e18, kind="neutral", tau_n0_s=tau_n0, tau_p0_s=tau_p0,
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
    digest gate (tests/test_bipolar_recombination_bgn_ionization.py) and (iii) the 6595bde numeric
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
def test_kinetic_noise_refused_until_k3_and_seed_range():
    """noise_spectrum REFUSES a device with a kinetic level while the trap g-r sources are not validated
    (v0.15 K3): the single-carrier terms carry no event rate, so the shipped "never silently noiseless" rule
    fires for the default mechanisms -- and passes when trap_n / trap_p are left out knowingly. The
    inventory lists the electron / hole branches with their partner row and mechanism (the K3 hook). A
    seed with |ln w| > U_LIMIT raises TrapOccupancyRangeError, and fermi_dirac statistics refuse."""
    from dynameta.carriers import noise_analysis as NA
    tr = PT.TrapLevel(energy_eV=0.15, density_m3=2e21, kind="acceptor", tau_n0_s=3e-7, tau_p0_s=5e-7,
                      dynamic=True)
    dev = _diode("knoise", (tr,), lines=DIODE_LINES, doping=DIODE_DOP, ae=1e10, circuit=("hi",))
    terms = BP.recombination_terms(dev, "r")
    tn, tp = terms["node_electron"]["trap0"], terms["node_hole"]["trap0"]
    assert (tn.partner_equation, tn.partner_coeff, tn.mechanism) == ("TrapEquation_trap0", 1.0, "trap_n")
    assert (tp.partner_equation, tp.partner_coeff, tp.mechanism) == ("TrapEquation_trap0", -1.0, "trap_p")
    assert tn.pair is None and tp.pair is None and "trap0" not in terms["node"]   # no fast-trap pair
    with pytest.raises(RuntimeError, match="without an event rate"):
        with V.quiet():
            NA.noise_spectrum(dev, [1e3])
    with V.quiet():
        s = NA.noise_spectrum(dev, [1e3], mechanisms=("diffusion_n", "diffusion_p", "gr"))
    assert s.S[0] > 0.0
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
    lvn = PT.TrapLevel(energy_eV=0.1, density_m3=1e18, tau_n0_s=1e-9, tau_p0_s=1e-9, dynamic=True)
    with pytest.raises(NotImplementedError, match="fermi_dirac"):
        with V.quiet():
            BP.setup_bipolar_region("kfd", "r", eps_static=11.7, n_dos_m3=NC, n_i_m3=NI, mobility_n_m2Vs=0.135,
                                    mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=True,
                                    statistics="fermi_dirac", traps=(lvn,))


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
