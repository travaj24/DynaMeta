"""Gates for the opt-in bipolar physics of audit R26 (Auger + radiative recombination) and R27 (bandgap
narrowing + incomplete ionization), dynameta.carriers.bipolar_options, DynaMeta v0.14.0.

EVERY ORACLE IS INDEPENDENT OF THE CODE UNDER TEST: the literature constants are re-typed here in CGS
from the papers, the BGN formulas re-typed, the charge-neutrality roots solved with scipy brentq on a
hand-written residual, and k/q taken from scipy.constants (NOT dynameta.constants -- the 10-digit
KB_EV_K literal alone shifts the 50 K freeze-out root by 8.6e-11). No gate reads a value the module
computed and compares it with itself; the DEVSIM solution (or, for the contact roots, the pinned field
the solution must match in the interior) is always the thing measured.

EVERY GATE FAILS ON THE v0.13.0 CODE (main 5cbd998): there `setup_bipolar_region` has no `options`
keyword (TypeError) -- and, so that a silently-ignored flag could not pass either, each physics gate
also asserts that its answer differs from the SRH-only / full-ionization / unnarrowed answer by far more
than its own tolerance (the "ON is load-bearing" asserts, with the measured margins in the docstrings).

The DEFAULTS-OFF gate hashes the ordered DEVSIM call log of two fixtures built through the public API
without `options` (tests/_bipolar_calllog.py) against the digest recorded on v0.13.0 and re-recorded
2026-11-04 for the two-Bernoulli SG flux (numeric cross-version half: tests/test_devsim_sg_flux.py).

Fixtures are tiny 1-D devices (< 1 s each except the 15 um long diode, ~2-4 s); devsim self-skips.
"""
import math

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
from scipy.constants import e as QE, k as KB                              # noqa: E402  (oracle consts)
from scipy.optimize import brentq                                          # noqa: E402

VT300 = KB * 300.0 / QE
NI = 1.0e16                                     # Si n_i at 300 K, m^-3 (1e10 cm^-3)
NC300, NV300 = 2.8e25, 1.04e25                  # Si N_c, N_v at 300 K, m^-3
MU_N, MU_P = 0.135, 0.048

# --- literature constants, re-typed in the units the papers quote ---------------------------------
CN_DS, CP_DS = 2.8e-31 * 1e-12, 0.99e-31 * 1e-12   # Dziewior & Schmid 1977, cm^6/s -> m^6/s
B_TRUPKE = 9.5e-15 * 1e-6                           # Trupke et al. 2003, cm^3/s -> m^3/s


# =====================================================================================================
# fixture helpers
# =====================================================================================================

def _fresh(dev):
    from dynameta.carriers import eq_registry as _R
    mesh = dev + "_m"
    for d in list(devsim.get_device_list()):
        if d == dev:
            devsim.delete_device(device=d)
    for m in list(devsim.get_mesh_list()):
        if m == mesh:
            devsim.delete_mesh(mesh=m)
    _R.clear(dev)
    return mesh


def _drop(dev):
    from dynameta.carriers import eq_registry as _R
    try:
        devsim.delete_device(device=dev)
        devsim.delete_mesh(mesh=dev + "_m")
    except Exception:
        pass
    _R.clear(dev)


def _device(dev, lines, net_doping, *, options=None, tau=1e-7, fd=False, ohmic=("lo", "hi"),
            majority_contact=(), n_i=NI, n_c=NC300, n_v=NV300, donors=None, acceptors=None,
            presolve=False):
    """1-D bipolar device on mesh `lines` [(pos, ps, tag)], seeded in the frame its contacts pin.

    `majority_contact` contacts (the uniform-G / zero-flux slabs) pin Potential AND the MAJORITY
    carrier at their equilibrium seed values and leave the MINORITY carrier's row natural (zero flux):
    a minority-blocking ohmic contact. WHY NOT A POTENTIAL-ONLY CONTACT (the fixture until 2026-11-03):
    with no carrier pinned anywhere the slab is a FLOATING BODY -- in 1-D steady state with zero flux at
    both ends the sum of ALL electron- and hole-continuity rows is identically 0 (the fluxes telescope,
    q(G - U) enters the two with opposite signs: charge conservation), so the DC Jacobian is rank-
    deficient by exactly one at EVERY G (MEASURED on the R26-a slab: |column sums of the continuity
    rows| / |column| <= 1.1e-16 over all 94 Newton matrices of the G = 1e33 ramp; the physically scaled
    smallest singular value 1.3-1.6e-15, its vector = the body potential with the carriers re-shuffled
    at the contact node). The body's charge, i.e. its potential against the contact, is then set by
    nothing but round-off: a 0.1 V body shift re-converged in place at G = 0, 1e27 and 1e30 (far-end n, p
    unchanged to 1e-13), the far-end potential differed by 0.16 mV between the scipy and Elmer
    OpenBLAS at G = 1e30, R26-b(p) moved from -4.87e-8 to -5.27e-8, and at G = 1.2e29 of the R26-a ramp
    UMFPACK hit an exact zero pivot ("matrix is singular" -> umfpack numeric failed) on scipy's
    OpenBLAS only. Pinning the majority density closes the null mode (the body's Fermi level = the
    contact's) without changing the steady state: the minority rows stay zero-flux, so their sum
    forces sum_nodes vol q (G - U) = 0 and the majority contact exchanges no NET carriers (measured
    3e-17 of q G L). After the change the same ramp's scaled smallest singular value is 1.5e-5 (scaled
    condition 2e5, from 2e15), its 93 Newton matrices are BIT-IDENTICAL on the two OpenBLAS builds
    (they differed by 2.4e-11 after the first step and 6.5e-4 by the failing one), and so are the
    results -- test_slab_is_well_posed."""
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
    if donors is not None:
        devsim.node_model(device=dev, region="bulk", name="Donors", equation=donors)
        devsim.node_model(device=dev, region="bulk", name="Acceptors", equation=acceptors)
    kw = {} if options is None else {"options": options}
    BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=n_c, n_dos_p_m3=n_v, n_i_m3=n_i,
                            mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=tau, tau_p_s=tau,
                            fd_enhancement=fd, **kw)
    BP.setup_equilibrium_seed_models(dev, "bulk")
    for c in ohmic:
        BP.setup_contact_ohmic_bipolar(dev, c)
    devsim.node_model(device=dev, region="bulk", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "bulk")))
    for var, src in (("Potential", "_seed_psi"), ("Electrons", "IntrinsicElectrons"),
                     ("Holes", "IntrinsicHoles")):
        devsim.set_node_values(device=dev, region="bulk", name=var,
                               values=devsim.get_node_model_values(device=dev, region="bulk", name=src))
    for c in majority_contact:                   # psi + the majority carrier at their seed values
        i = 0 if c == "lo" else -1
        PE.setup_contact(dev, c)
        devsim.set_parameter(device=dev, name="{}_bias".format(c),
                             value=float(_v(dev, "_seed_psi")[i]))
        var, eq, cur = (("Electrons", "ElectronContinuityEquation", "ElectronCurrent")
                        if _v(dev, "NetDoping")[i] > 0 else
                        ("Holes", "HoleContinuityEquation", "HoleCurrent"))
        cm = "{}_{}_majority".format(c, var.lower())
        devsim.set_parameter(device=dev, name=cm + "_value", value=float(_v(dev, var)[i]))
        devsim.contact_node_model(device=dev, contact=c, name=cm,
                                  equation="{} - {}_value".format(var, cm))
        devsim.contact_node_model(device=dev, contact=c, name="{}:{}".format(cm, var), equation="1")
        _R.record_contact_equation(dev, c, name=eq, node_model=cm, edge_current_model=cur)
    if presolve:                                 # the 1-D diode recipe: frozen-carrier Poisson first
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.delete_by_name(dev, ceq)
        _solve(1e10, 1e-10, 100)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.reapply_by_name(dev, ceq)
        for var, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            devsim.set_node_values(device=dev, region="bulk", name=var,
                                   values=devsim.get_node_model_values(device=dev, region="bulk",
                                                                       name=src))


def _solve(abs_err=1e11, rel_err=1e-10, max_iter=80):
    devsim.solve(type="dc", solver_type="direct", absolute_error=abs_err, relative_error=rel_err,
                 maximum_iterations=max_iter)


def _v(dev, name):
    return np.asarray(devsim.get_node_model_values(device=dev, region="bulk", name=name), dtype=float)


def _slab(dev, nd, options, tau, length=None):
    """Uniform slab with ONE minority-blocking majority contact (_device `majority_contact`): the
    minority carrier sees zero flux on every boundary and the majority contact exchanges no net
    carriers, so the steady state is the zero-flux one -- with the body's potential now FIXED (until
    2026-11-03 a Potential-only contact left it floating: _device docstring, test_slab_is_well_posed).

    The contact pins psi and the majority density at their EQUILIBRIUM values, so under injection a
    one-node Debye layer forms there (measured on R26-a at G = 1e33, dp ~ 0.9 n0: n(0) = 0.53 n_far,
    p(0) = 1.87 p_far, both within 1.5 % of the far value at the next node, 5 nm in). Two regimes:
      * length=None, a 200 nm slab << the diffusion length: both quasi-Fermi levels are flat, so n p is
        uniform (the layer included) but n and p separately are not -- fine for U ~ (n p - n_i^2) at low
        injection (the lifetime gates), NOT for an Auger/SRH root at high injection (the layer's extra
        n shifted the slab-averaged balance by 5e-4: the first version of R26-a measured exactly that);
      * length given (>> the diffusion length, graded mesh): the far end is LOCAL, G = U(n, p) holds
        pointwise there with n - p = N, which is what the analytic root assumes."""
    if length is None:
        lines = [(0.0, 5e-9, "lo"), (200e-9, 5e-9, "hi")]
    else:
        lines = [(0.0, 5e-9, "lo"), (2e-6, 100e-9, None), (length, length / 40.0, "hi")]
    _device(dev, lines, "{:.10e}".format(nd), options=options, tau=tau, ohmic=(),
            majority_contact=("lo",))
    _solve()


def _ramp_G(dev, G_target, n_steps=24):
    for g in np.logspace(math.log10(G_target) - 10.0, math.log10(G_target), n_steps):
        devsim.set_parameter(device=dev, region="bulk", name="G_ext", value=float(g))
        _solve()


@pytest.mark.parametrize("nd,length,G", [(1e25, 150e-6, 1e30), (-1e25, None, 1e24)])
def test_slab_is_well_posed(nd, length, G):
    """GATE (2026-11-03): the uniform-G slab fixture has ONE DC solution, so its answer cannot depend on
    round-off. The converged slab's BODY is shifted by +20 mV (psi at every node but the contact's,
    carriers kept) and re-solved: it must come back to the reference state (|d psi| < 1e-9 V, carriers
    to 1e-9; measured 0 V and <= 2.2e-16 on both the scipy and the Elmer OpenBLAS). On the potential-
    only fixture (a floating body: _device docstring) this gate FAILS under BOTH libraries: with Elmer's
    the n slab re-converged IN PLACE 1.56e-2 V off, every other case/library hit the singular Jacobian
    ("umfpack numeric failed") -- the null mode that made R26-a[1e33] fail on scipy's OpenBLAS and pass
    on Elmer's. Also pinned: the majority contact exchanges no NET carriers (sum over nodes of
    vol q (G - U) = 0 to 1e-12 of q G L; measured 3.0e-17 / 1.2e-16), i.e. the zero-flux physics the
    oracles assume is unchanged."""
    from scipy.constants import e as qe
    from dynameta.carriers.bipolar_options import (AUGER_SI_DZIEWIOR_SCHMID_1977, BipolarOptions,
                                                   RADIATIVE_SI_TRUPKE_2003)
    opts = BipolarOptions(auger=AUGER_SI_DZIEWIOR_SCHMID_1977, radiative=RADIATIVE_SI_TRUPKE_2003,
                          generation_m3s=0.0)
    dev = "wellposed"
    names = ("Potential", "Electrons", "Holes")
    try:
        _slab(dev, nd, opts, 1e-6, length=length)
        _ramp_G(dev, G, n_steps=8)
        ref = {v: _v(dev, v) for v in names}
        x, vol, eg = _v(dev, "x"), _v(dev, "NodeVolume"), _v(dev, "ElectronGeneration")
        psi = ref["Potential"].copy()
        psi[1:] += 0.02
        for v, vals in (("Potential", psi), ("Electrons", ref["Electrons"]), ("Holes", ref["Holes"])):
            devsim.set_node_values(device=dev, region="bulk", name=v, values=list(vals))
        _solve()
        new = {v: _v(dev, v) for v in names}
    finally:
        _drop(dev)
    assert np.max(np.abs(new["Potential"] - ref["Potential"])) < 1e-9
    for v in ("Electrons", "Holes"):
        assert np.max(np.abs(new[v] / ref[v] - 1.0)) < 1e-9, v
    assert abs(np.sum(vol * eg)) < 1e-12 * qe * G * (x.max() - x.min())


# =====================================================================================================
# R26: Auger + radiative
# =====================================================================================================

def _U_oracle(n, p, *, tau, cn=0.0, cp=0.0, b=0.0, ni=NI):
    """SRH (midgap, tau_n = tau_p) + Auger + radiative, re-typed."""
    srh = (n * p - ni * ni) / (tau * (n + ni) + tau * (p + ni))
    return srh + (cn * n + cp * p) * (n * p - ni * ni) + b * (n * p - ni * ni)


@pytest.mark.parametrize("G", [1e24, 1e30, 1e33])
def test_uniform_generation_root_srh_auger_radiative(G):
    """GATE R26-a. n-type 1e25 m^-3 slab, SRH (tau 1 us) + Dziewior-Schmid Auger + Trupke radiative,
    uniform G from 1e24 (low injection, dp ~ 3e-9 n0) to 1e33 m^-3/s (HIGH injection, dp ~ 0.9 n0,
    where the C_p p and B terms and the injection dependence of every term all matter). 150 um slab
    (>= 23 diffusion lengths, so the far end is local). The excess density at the far node must equal
    the scipy brentq root of G = U_SRH + U_A + U_rad to 1e-9 (measured 0, 2.2e-16, -1.1e-16 at the three
    G, bit-identical on the scipy and the Elmer OpenBLAS: the far end is exactly local). The old
    "1e6x headroom for BLAS changes" was true of the COMPARISON only: on the floating-body fixture
    (before 2026-11-03, _device docstring) the 1e33 ramp itself died at G = 1.2e29 on scipy's OpenBLAS
    (exact zero pivot, "umfpack numeric failed"), so no tolerance headroom could have protected it.
    ON is load-bearing: the SRH-only root is 29x / 32x / 229x larger."""
    from dynameta.carriers.bipolar_options import (AUGER_SI_DZIEWIOR_SCHMID_1977, BipolarOptions,
                                                   RADIATIVE_SI_TRUPKE_2003)
    nd, tau = 1e25, 1e-6
    dev = "r26a"
    opts = BipolarOptions(auger=AUGER_SI_DZIEWIOR_SCHMID_1977, radiative=RADIATIVE_SI_TRUPKE_2003,
                          generation_m3s=0.0)
    try:
        _slab(dev, nd, opts, tau, length=150e-6)
        _ramp_G(dev, G)
        p = _v(dev, "Holes")
        n = _v(dev, "Electrons")
    finally:
        _drop(dev)
    n0 = 0.5 * nd + math.sqrt(0.25 * nd * nd + NI * NI)
    p0 = NI * NI / n0
    root = brentq(lambda d: _U_oracle(n0 + d, p0 + d, tau=tau, cn=CN_DS, cp=CP_DS, b=B_TRUPKE) - G,
                  0.0, 1e28, xtol=1e-300, rtol=1e-15, maxiter=500)
    dp = p[-1] - p0
    assert dp == pytest.approx(root, rel=1e-9)
    assert n[-1] - n0 == pytest.approx(root, rel=1e-6)       # neutrality dn = dp (n0 cancellation)
    srh_only = brentq(lambda d: _U_oracle(n0 + d, p0 + d, tau=tau) - G, 0.0, 1e40, rtol=1e-15)
    assert srh_only / root > 20.0                              # ON is load-bearing


@pytest.mark.parametrize("polarity", ["n", "p"])
def test_auger_low_injection_lifetime(polarity):
    """GATE R26-b. Auger-only slab (SRH tau = 1e3 s, i.e. 1/tau 11 orders under C n0^2), low injection:
    dp/G (n-type) must equal tau_A = 1/(C_n n0^2), and dn/G (p-type) 1/(C_p p0^2), to 1e-6 -- the
    neglected terms are the injection share dp/n0 (3.6e-9 n, 1.0e-8 p, entering Auger x2-3) and the
    1/tau SRH share (3.6e-11 / 1e-10). Measured rel -8.4e-9 (n) / -4.9e-8 (p) on both OpenBLAS builds:
    exactly that size (the floating-body fixture gave -3.3e-9 (n) and a library-dependent -4.87e-8 /
    -5.27e-8 (p): the free body potential moved the contact layer's share of the balance). ON is
    load-bearing: the SRH-only lifetime here is 1e3 s, >= 9.9 orders longer."""
    from dynameta.carriers.bipolar_options import AUGER_SI_DZIEWIOR_SCHMID_1977, BipolarOptions
    N0, G = 1e25, 1e24
    dev = "r26b" + polarity
    opts = BipolarOptions(auger=AUGER_SI_DZIEWIOR_SCHMID_1977, generation_m3s=0.0)
    try:
        _slab(dev, N0 if polarity == "n" else -N0, opts, 1e3)
        _ramp_G(dev, G, n_steps=8)
        minority = _v(dev, "Holes" if polarity == "n" else "Electrons")[-1]
    finally:
        _drop(dev)
    eq_min = NI * NI / (0.5 * N0 + math.sqrt(0.25 * N0 * N0 + NI * NI))
    tau_meas = (minority - eq_min) / G
    tau_A = 1.0 / ((CN_DS if polarity == "n" else CP_DS) * N0 * N0)
    assert tau_meas == pytest.approx(tau_A, rel=1e-6)
    assert 1e3 / tau_meas > 1e9                                # ON is load-bearing


def test_coulomb_enhanced_auger_lifetime_altermatt_1997():
    """GATE R26-c. Altermatt 1997 Coulomb-enhanced Auger at n0 = 1e17 cm^-3, where the enhancement is
    large: g_eeh = 1 + 13 [1 - tanh((n0/3.3e17)^0.66)] (re-typed here) = 8.465, so the low-injection
    lifetime is 1/(g_eeh C_n n0^2) -- the plain Dziewior-Schmid value is 8.5x longer. Gate 1e-6
    (measured -1.39e-7 = the dp/n0 injection term at G = 1e20; -1.45e-7 on the floating-body fixture)."""
    from dynameta.carriers.bipolar_options import AUGER_SI_ALTERMATT_1997, BipolarOptions
    N0, G = 1e23, 1e20
    dev = "r26c"
    opts = BipolarOptions(auger=AUGER_SI_ALTERMATT_1997, generation_m3s=0.0)
    try:
        _slab(dev, N0, opts, 1e3)
        _ramp_G(dev, G, n_steps=8)
        p = _v(dev, "Holes")[-1]
    finally:
        _drop(dev)
    n0 = 0.5 * N0 + math.sqrt(0.25 * N0 * N0 + NI * NI)
    g_eeh = 1.0 + 13.0 * (1.0 - math.tanh((n0 / 1e6 / 3.3e17) ** 0.66))
    tau_meas = (p - NI * NI / n0) / G
    assert g_eeh == pytest.approx(8.4646, abs=1e-3)
    assert tau_meas == pytest.approx(1.0 / (g_eeh * CN_DS * n0 * n0), rel=1e-6)
    assert tau_meas * CN_DS * n0 * n0 < 0.2                    # vs un-enhanced: 8.5x shorter


def test_radiative_slab_rate_is_B_np_minus_ni2():
    """GATE R26-d. Radiative-DOMINATED slab, n-type 1e25 (B n0 = 9.5e4 /s vs SRH 1/tau = 1e4 /s: 90 %
    radiative), G = 1e28 (dp ~ 1 % of n0), 3 mm slab (~27 diffusion lengths: the far end is local).
    (i) The URad node model equals B (n p - n_i^2) re-evaluated from the SOLVED n, p at every node to
    1e-12; (ii) at the far node URad + U_SRH(oracle) = G to 1e-9; (iii) the excess density there is the
    brentq root of G = B((n0+d)(p0+d) - n_i^2) + U_SRH to 1e-9 (measured 0 / 0 / 2.2e-16). ON is
    load-bearing: SRH-only gives an 11.6x larger excess density.
    WHY SRH IS KEPT ON (tau 1e-4 s): the first radiative-only version (tau_SRH = 1e3 s) died with UMFPACK
    declaring the Jacobian singular mid-ramp (G = 6e22, 1e24 m^-3, 200 nm). That was NOT the weak
    dU/dp = B n0 pinning it was attributed to but the floating-body null mode of the then potential-only
    fixture (_device docstring): re-measured 2026-11-03, the radiative-only 200 nm ramp to 1e24 still
    fails on that fixture under both OpenBLAS builds and lands on the brentq root to 0 (and the 3 mm one
    at 1e28 to 2.2e-16) on the majority-contact fixture. (i) is the radiative-only statement at every
    node either way."""
    from dynameta.carriers.bipolar_options import BipolarOptions, RADIATIVE_SI_TRUPKE_2003
    nd, G, tau = 1e25, 1e28, 1e-4
    dev = "r26d"
    try:
        _slab(dev, nd, BipolarOptions(radiative=RADIATIVE_SI_TRUPKE_2003, generation_m3s=0.0), tau,
              length=3e-3)
        _ramp_G(dev, G)
        n, p, urad = _v(dev, "Electrons"), _v(dev, "Holes"), _v(dev, "URad")
    finally:
        _drop(dev)
    assert np.allclose(urad, B_TRUPKE * (n * p - NI * NI), rtol=1e-12, atol=0.0)
    assert urad[-1] + _U_oracle(n[-1], p[-1], tau=tau) == pytest.approx(G, rel=1e-9)
    n0 = 0.5 * nd + math.sqrt(0.25 * nd * nd + NI * NI)
    p0 = NI * NI / n0
    root = brentq(lambda d: _U_oracle(n0 + d, p0 + d, tau=tau, b=B_TRUPKE) - G, 0.0, 1e28,
                  xtol=1e-300, rtol=1e-15)
    assert p[-1] - p0 == pytest.approx(root, rel=1e-9)
    srh_only = brentq(lambda d: _U_oracle(n0 + d, p0 + d, tau=tau) - G, 0.0, 1e30, rtol=1e-15)
    assert srh_only / root > 9.0


# =====================================================================================================
# R27: bandgap narrowing
# =====================================================================================================

def _dEg_slotboom(N, e_ref=9e-3, n_ref=1e17, c=0.5):
    """Slotboom & de Graaff 1976/77, N in cm^-3 (re-typed)."""
    x = math.log(N / n_ref)
    return e_ref * (x + math.sqrt(x * x + c))


def _dEg_del_alamo(N, e_ref=18.7e-3, n_ref=7e17):
    """del Alamo, Swirhun & Swanson 1985, N in cm^-3 (re-typed)."""
    return e_ref * math.log(N / n_ref) if N > n_ref else 0.0


def test_bgn_model_formulas_match_the_papers():
    """Pure numpy: the shipped presets evaluate the papers' formulas (CGS, re-typed above); the light-
    doping branch of Slotboom is computed WITHOUT cancellation (x + sqrt(x^2 + C) at x = -30 would lose
    every digit; the C/(sqrt - x) form is exact)."""
    from dynameta.carriers.bipolar_options import (BGN_SI_DEL_ALAMO_1985, BGN_SI_KLAASSEN_1992,
                                                   BGN_SI_SLOTBOOM_1977)
    for Ncm in (1e15, 1e17, 1e18, 1e19, 1e20):
        assert float(BGN_SI_SLOTBOOM_1977.delta_Eg_eV(Ncm * 1e6)) == pytest.approx(
            _dEg_slotboom(Ncm), rel=1e-12)
        assert float(BGN_SI_KLAASSEN_1992.delta_Eg_eV(Ncm * 1e6)) == pytest.approx(
            _dEg_slotboom(Ncm, 6.92e-3, 1.3e17), rel=1e-12)
        assert float(BGN_SI_DEL_ALAMO_1985.delta_Eg_eV(Ncm * 1e6)) == pytest.approx(
            _dEg_del_alamo(Ncm), rel=1e-12, abs=0.0)
    tiny = 1e17 * math.exp(-30.0) * 1e6                              # x = -30
    exact = 9e-3 * 0.5 / (math.sqrt(900.5) + 30.0)
    assert float(BGN_SI_SLOTBOOM_1977.delta_Eg_eV(tiny)) == pytest.approx(exact, rel=1e-13)
    assert float(BGN_SI_SLOTBOOM_1977.delta_Eg_eV(0.0)) == 0.0


def test_bgn_zero_flux_slab_relaxes_to_n_ie_squared():
    """GATE R27-a. Uniform n+ slab (1e19 cm^-3) with ZERO hole flux on every boundary (one minority-
    blocking majority contact: _device), started with the holes 3x off equilibrium: the only thing that
    can fix n p is detailed balance of the recombination, so the steady state must sit at
    n p = n_ie^2 = n_i^2 exp(dEg/kT) with del Alamo's
    dEg = 49.73 meV -- to 1e-12 (measured 2.2e-16). ON is load-bearing: n_ie^2/n_i^2 = 6.845 (an SRH
    still built on n_i^2 would relax to n_i^2, off by 5.8)."""
    from dynameta.carriers.bipolar_options import BGN_SI_DEL_ALAMO_1985, BipolarOptions
    dev = "r27a"
    try:
        _device(dev, [(0.0, 5e-9, "lo"), (100e-9, 5e-9, "hi")], "1e25",
                options=BipolarOptions(bgn=BGN_SI_DEL_ALAMO_1985), ohmic=(), majority_contact=("lo",))
        devsim.set_node_values(device=dev, region="bulk", name="Holes",
                               values=list(3.0 * _v(dev, "Holes")))
        _solve()
        n, p = _v(dev, "Electrons"), _v(dev, "Holes")
    finally:
        _drop(dev)
    nie2 = NI * NI * math.exp(_dEg_del_alamo(1e19) / VT300)
    assert nie2 / (NI * NI) == pytest.approx(6.845, abs=1e-3)
    assert np.max(np.abs(n * p / nie2 - 1.0)) < 1e-12


@pytest.mark.parametrize("gamma", [0.3, 0.5, 1.0])
def test_bgn_graded_equilibrium_is_current_free(gamma):
    """GATE R27-b: the BAND-EDGE SPLIT. Exponentially graded n-type doping 1e20 -> 1e17 cm^-3 over 1 um
    (Slotboom 1977 dEg 6.4..124.7 meV across it), both ohmic contacts at 0 V, the narrowing split gamma into
    E_c. A correct implementation gives true equilibrium: n p = n_ie(x)^2 at EVERY node (1e-12; measured
    <= 4.6e-15) and no current anywhere (|J| < 1e-12 of the doping-gradient diffusion scale
    q D_n N/L_grad = 3.9e11 A/m^2; measured <= 0.1 A/m^2 = 2.5e-13, the float floor of the SG flux).
    NEGATIVE CONTROL (the classic error): drop the conduction-band quasi-field from the transport while
    keeping the BGN contacts -- the zero-bias device then carries a spurious current of 5e8..1.8e9 A/m^2
    (gamma 0.3..1), 9-10 orders above the correct implementation's float floor."""
    from dynameta.carriers.bipolar_options import BGNModel, BipolarOptions
    bgn = BGNModel(kind="slotboom", E_ref_eV=9e-3, N_ref_m3=1e23, C=0.5, conduction_fraction=gamma)
    dev = "r27b"
    lines = [(0.0, 2e-9, "lo"), (1e-6, 2e-9, "hi")]
    expr = "1e26*exp(-x/1e-6*log(1000))"
    try:
        _device(dev, lines, expr, options=BipolarOptions(bgn=bgn))
        _solve(1e10, 1e-10)
        x, n, p = _v(dev, "x"), _v(dev, "Electrons"), _v(dev, "Holes")
        jn = np.asarray(devsim.get_edge_model_values(device=dev, region="bulk",
                                                     name="ElectronCurrent"))
        jp = np.asarray(devsim.get_edge_model_values(device=dev, region="bulk", name="HoleCurrent"))
        # negative control: zero D_c (the electron quasi-field) and re-solve
        devsim.set_node_values(device=dev, region="bulk", name="BgnDc", values=[0.0] * len(x))
        _solve(1e10, 1e-10)
        jn_bad = np.asarray(devsim.get_edge_model_values(device=dev, region="bulk",
                                                         name="ElectronCurrent"))
    finally:
        _drop(dev)
    N = 1e26 * np.exp(-x / 1e-6 * math.log(1000.0))
    dEg = np.array([_dEg_slotboom(v / 1e6) for v in N])
    nie2 = NI * NI * np.exp(dEg / VT300)
    J_scale = QE * MU_N * VT300 * 1e26 / (1e-6 / math.log(1000.0))
    assert np.max(np.abs(n * p / nie2 - 1.0)) < 1e-12
    assert np.max(np.abs(jn)) < 1e-12 * J_scale and np.max(np.abs(jp)) < 1e-12 * J_scale
    assert np.max(np.abs(jn_bad)) > 1e7


def test_bgn_long_diode_saturation_current():
    """GATE R27-c. n+/p long diode: emitter N_D = 1e19 cm^-3 x 15 um (13.5 L_p), base N_A = 1e17 cm^-3,
    tau = 1 ns, del Alamo BGN (49.73 meV in the emitter, exactly 0 in the base), Boltzmann statistics,
    V = 0.3 V (low injection: dp/N_D ~ 1e-12). The hole current at the n-side depletion edge, de-rated
    by exp((x - x_j)/L_p) to the edge and by (exp(V/V_t) - 1), must equal the analytic emitter saturation
    current J0p = q n_ie^2 D_p/(N_D L_p) within 2e-3. Measured -6.3e-4 (BGN) and +6.7e-4 (no BGN against
    its own n_i^2 formula). Error budget: SG discretization O((h/L_p)^2) <= 3e-4 at h <= 20 nm; the
    finite emitter tanh(15um/L_p) - 1 ~ 1e-11; n-side depletion width ~1 nm << L_p; low-injection terms
    ~1e-12. The BGN/no-BGN RATIO must equal exp(dEg/kT) = 6.845 within 2e-3 (measured 1.3e-3; the two
    discretization errors are of opposite sign). ON is load-bearing: the ratio is 6.8, not 1."""
    from dynameta.carriers.bipolar_options import BGN_SI_DEL_ALAMO_1985, BipolarOptions
    NA, ND, XJ, tau, V = 1e23, 1e25, 0.5e-6, 1e-9, 0.3
    lines = [(0.0, 5e-9, "lo"), (XJ, 2e-10, None), (XJ + 0.05e-6, 2e-9, None),
             (XJ + 5e-6, 20e-9, None), (XJ + 15e-6, 100e-9, "hi")]
    expr = "ifelse(x < {}, {}, {})".format(XJ, -NA, ND)
    Dp = MU_P * VT300
    Lp = math.sqrt(Dp * tau)
    j0 = {}
    for tag, opts in (("bgn", BipolarOptions(bgn=BGN_SI_DEL_ALAMO_1985)), ("off", None)):
        dev = "r27c" + tag
        try:
            _device(dev, lines, expr, options=opts, tau=tau, presolve=True)
            _solve(1e12, 1e-10)
            for vb in np.arange(0.05, V + 1e-9, 0.05):
                devsim.set_parameter(device=dev, name="lo_bias", value=float(vb))
                _solve(1e12, 1e-10)
            x = _v(dev, "x")
            xe = 0.5 * (x[:-1] + x[1:])
            Jp = np.asarray(devsim.get_edge_model_values(device=dev, region="bulk", name="HoleCurrent"))
        finally:
            _drop(dev)
        i = int(np.argmin(np.abs(xe - (XJ + 20e-9))))
        j0[tag] = abs(Jp[i]) * math.exp((xe[i] - XJ) / Lp) / math.expm1(V / VT300)
    nie2 = NI * NI * math.exp(_dEg_del_alamo(1e19) / VT300)
    assert j0["bgn"] == pytest.approx(QE * nie2 * Dp / (ND * Lp), rel=2e-3)
    assert j0["off"] == pytest.approx(QE * NI * NI * Dp / (ND * Lp), rel=2e-3)
    assert j0["bgn"] / j0["off"] == pytest.approx(nie2 / (NI * NI), rel=2e-3)


@pytest.mark.parametrize("ND,NA", [(0.0, 1e25), (1e25, 1e24)])
def test_del_alamo_on_p_type_and_compensated_slabs_uses_donors_only(ND, NA):
    """GATE 3.2-4 (DEVSIM twin of tests/test_bipolar_materials_corrections.py; consolidated plan row
    3.2-4). del Alamo's fit (IEEE TED ED-34, 1580 (1987) Eq. (11), re-typed in _dEg_del_alamo) is an
    n-type fit in N_D. The R27-a fixture -- uniform slab, one minority-blocking majority contact, the
    minority carrier kicked 3x -- relaxes to n p = n_i^2 exp(dEg(N_D)/kT) to 1e-12 (R27-a measured
    2.2e-16): p+ 1e19 cm^-3 -> dEg = 0, n p = n_i^2; compensated N_D = 1e19 / N_A = 1e18 -> 49.73 meV,
    n_ie^2/n_i^2 = 6.845. PRE-FIX (N_D + N_A basis) these were 6.845 (5.8 off) and 7.334 (+7.1 %).
    Building the region emits exactly ONE UnreliableResultWarning naming it (N_A > N_ref = 7e17: the fit has
    no basis there). ADVERSARIAL: the pre-fix answer is asserted to be off by far more than 1e-12."""
    import warnings
    from dynameta.carriers.bipolar_options import BGN_SI_DEL_ALAMO_1985, BipolarOptions
    from dynameta.warning_types import UnreliableResultWarning
    dev = "r27dap" if ND == 0.0 else "r27dac"
    minority = "Electrons" if ND < NA else "Holes"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _device(dev, [(0.0, 5e-9, "lo"), (100e-9, 5e-9, "hi")], "{:.10e}".format(ND - NA),
                    options=BipolarOptions(bgn=BGN_SI_DEL_ALAMO_1985), ohmic=(),
                    majority_contact=("lo",), donors="{:.10e}".format(ND),
                    acceptors="{:.10e}".format(NA))
        devsim.set_node_values(device=dev, region="bulk", name=minority,
                               values=list(3.0 * _v(dev, minority)))
        _solve()
        n, p = _v(dev, "Electrons"), _v(dev, "Holes")
    finally:
        _drop(dev)
    nie2 = NI * NI * math.exp(_dEg_del_alamo(ND / 1e6) / VT300)
    assert np.max(np.abs(n * p / nie2 - 1.0)) < 1e-12
    pre_fix = NI * NI * math.exp(_dEg_del_alamo((ND + NA) / 1e6) / VT300)
    assert abs(pre_fix / nie2 - 1.0) > 0.07                        # adversarial: the old basis fails
    assert [type(w.message) for w in caught] == [UnreliableResultWarning],         [str(w.message)[:100] for w in caught]
    assert "'bulk'" in str(caught[0].message) and "N_D only" in str(caught[0].message)


# =====================================================================================================
# R27: incomplete ionization (freeze-out)
# =====================================================================================================

def _si_at(T):
    """Si inputs at T, the SAME numbers handed to the model (they are inputs, not under test):
    N_c, N_v ~ T^1.5 and a Varshni gap 1.17 - 4.73e-4 T^2/(T + 636) eV."""
    nc = NC300 * (T / 300.0) ** 1.5
    nv = NV300 * (T / 300.0) ** 1.5
    eg = 1.17 - 4.73e-4 * T * T / (T + 636.0)
    return nc, nv, math.sqrt(nc * nv) * math.exp(-eg * QE / (2.0 * KB * T))


def _freeze_out_root(T, ND, NA, dED=0.045, dEA=0.045):
    """Charge neutrality n - p = N_D+ - N_A- with Boltzmann occupancy (g_D = 2, g_A = 4), brentq in ln n."""
    nc, nv, ni = _si_at(T)
    kT = KB * T / QE
    n1, p1 = nc * math.exp(-dED / kT), nv * math.exp(-dEA / kT)

    def f(ln_n):
        n = math.exp(ln_n)
        p = ni * ni / n
        return n - p - ND / (1.0 + 2.0 * n / n1) + NA / (1.0 + 4.0 * p / p1)
    lo = math.log(ni * ni / (ND + NA + ni)) - 1.0
    hi = math.log(ND + NA + ni) + 1.0
    return math.exp(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500))


@pytest.mark.parametrize("ND,NA", [(1e22, 0.0), (1e22, 3e21)])
def test_freeze_out_matches_charge_neutrality_root(ND, NA):
    """GATE R27-d. Si:P (45 meV, g_D = 2) at 1e16 cm^-3, uncompensated and 30 % boron-compensated, at
    T = 50, 77, 100, 150, 200, 300 K. The electron density at the MIDDLE of a 2 um bar (13+ Debye
    lengths from either contact, so it is set by the DEVSIM Poisson charge with N_D+(n) -- the interior
    was deliberately kicked by 50 mV before the solve) must equal the brentq root to 1e-11 (measured
    <= 7.1e-15 over all 12 points; an oracle built on the 10-digit KB_EV_K literal instead of scipy's
    exact k/e would sit 8.6e-11 off at 50 K), and so must the field the contacts pin. ON is
    load-bearing: the ionized fraction is 0.051 (uncomp.) / 0.009 (comp.) at 50 K and 0.996 at 300 K
    -- full ionization would be off by 19x / 111x at 50 K. No Mott transition is modelled (not needed
    at 1e16 cm^-3, far below Si:P's ~3.7e18)."""
    from dynameta.carriers.bipolar_options import BipolarOptions, IONIZATION_SI_P_B
    out = []
    for T in (50.0, 77.0, 100.0, 150.0, 200.0, 300.0):
        nc, nv, ni = _si_at(T)
        dev = "r27d"
        don = acc = None
        if NA > 0.0:
            don, acc = "{:.10e}".format(ND), "{:.10e}".format(NA)
        try:
            _device(dev, [(0.0, 10e-9, "lo"), (1e-6, 50e-9, None), (2e-6, 10e-9, "hi")],
                    "{:.10e}".format(ND - NA), n_i=ni, n_c=nc, n_v=nv, donors=don, acceptors=acc,
                    options=BipolarOptions(ionization=IONIZATION_SI_P_B, T_K=T))
            x = _v(dev, "x")
            psi = _v(dev, "Potential") + 0.05 * np.sin(np.pi * x / 2e-6)
            devsim.set_node_values(device=dev, region="bulk", name="Potential", values=list(psi))
            _solve(1e8, 1e-12)
            n_mid = _v(dev, "Electrons")[int(np.argmin(np.abs(x - 1e-6)))]
            n_pin = _v(dev, "NeutralElectrons")[0]
        finally:
            _drop(dev)
        root = _freeze_out_root(T, ND, NA)
        assert n_mid == pytest.approx(root, rel=1e-11), T
        assert n_pin == pytest.approx(root, rel=1e-11), T
        out.append(n_mid / (ND - NA))
    assert out[0] < (0.06 if NA == 0.0 else 0.01) and out[-1] > 0.99


def test_acceptor_freeze_out_p_type():
    """GATE R27-e: the acceptor branch (g_A = 4). Si:B 1e16 cm^-3 at 77 K: hole density at mid-bar =
    brentq root to 1e-11; the ionized fraction is < 0.5 (full ionization would be off by > 2x)."""
    from dynameta.carriers.bipolar_options import BipolarOptions, IONIZATION_SI_P_B
    T, NA = 77.0, 1e22
    nc, nv, ni = _si_at(T)
    dev = "r27e"
    try:
        _device(dev, [(0.0, 10e-9, "lo"), (1e-6, 50e-9, None), (2e-6, 10e-9, "hi")],
                "{:.10e}".format(-NA), n_i=ni, n_c=nc, n_v=nv,
                options=BipolarOptions(ionization=IONIZATION_SI_P_B, T_K=T))
        _solve(1e8, 1e-12)
        x = _v(dev, "x")
        p_mid = _v(dev, "Holes")[int(np.argmin(np.abs(x - 1e-6)))]
    finally:
        _drop(dev)
    root_p = ni * ni / _freeze_out_root(T, 0.0, NA)
    assert p_mid == pytest.approx(root_p, rel=1e-11)
    assert p_mid / NA < 0.5


@pytest.mark.parametrize("N,expect", [(1e25, 1), (-1e25, 1), (1e24, 0)])
def test_mott_warning_once_per_region_above_the_mott_density(N, expect):
    """GATE 3.2-11c (DEVSIM half; P2 item 4 I3). The legacy single-level model has no Mott transition:
    building a region doped above the model's Mott density (Si:P 3.74e18, Si:B proxy 4.5e18 cm^-3) with
    incomplete ionization on emits exactly ONE UnreliableResultWarning naming the region; 1e18 cm^-3 is silent.
    Pre-fix no warning existed (the 1e19 regions built silently at 39 % ionized). The occupancy itself
    is unchanged (the R27-d/e gates and the call-log digest are untouched)."""
    import warnings
    from dynameta.carriers.bipolar_options import BipolarOptions, IONIZATION_SI_P_B
    from dynameta.warning_types import UnreliableResultWarning
    dev = "mott"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _device(dev, [(0.0, 5e-9, "lo"), (100e-9, 5e-9, "hi")], "{:.10e}".format(N),
                    options=BipolarOptions(ionization=IONIZATION_SI_P_B))
    finally:
        _drop(dev)
    assert [type(w.message) for w in caught] == [UnreliableResultWarning] * expect, \
        [str(w.message)[:100] for w in caught]
    if expect:
        assert "'bulk'" in str(caught[0].message) and "Mott" in str(caught[0].message)


# =====================================================================================================
# defaults OFF: byte-identical to v0.13.0
# =====================================================================================================

# sha256 of tests/_bipolar_calllog.record_call_log(). Recorded on v0.13.0 (main 5cbd998, 217 calls:
# ac63536e...b0c6aa) and RE-RECORDED 2026-11-04 (237 calls) for the two-Bernoulli SG flux (sg_flux; user
# decision): the logs differ ONLY in ElectronCurrent / HoleCurrent and their derivatives plus the new
# Bern_g_m / Bern_gp_m models (measured diff of the two logs). The cross-version half -- the same
# fixtures SOLVED, every field and terminal current vs 6595bde (= v0.13.0's flux) to the change's
# measured round-off -- is tests/test_devsim_sg_flux.py::test_defaults_off_fixtures_equal_6595bde_*.
_CALLLOG_DIGEST = "3542731548315655dff28454e48cca4a2dcf69389e22881dad88a879bb205f80"
# v0.15.0 (2026-11-07): the DEFAULT statistics became 'fermi_dirac'. The digest above is now that of
# record_call_log(statistics="legacy") (the pre-v0.15.0 default, kept explicit: UNCHANGED, gated); the new
# default's log is RE-RECORDED below. Cross-check (measured, this tree): the default log equals the explicit
# statistics='fermi_dirac' log, and differs from the legacy one only in the FD fixture (cl_fd): the Boltzmann
# fixture's rows are identical.
_CALLLOG_DIGEST_V0150 = "271e5d87f247959b3a3876a9d0d94ab73fed790f3e5868b98e6954556e37e18d"
_CALLLOG_CALLS_V0150 = 302


def test_defaults_off_physics_definition_is_byte_identical_to_the_recorded_digest():
    """GATE R26/R27-off-a. The ordered log of every DEVSIM model / equation / parameter / solution call
    the bipolar API makes for two fixtures (FD + DC contacts; Boltzmann + a circuit contact), with no
    `options`, hashes to the recorded digest -- every expression string and value character-for-
    character (v0.13.0's up to the 2026-11-04 flux re-record, see _CALLLOG_DIGEST). The recorder is
    SENSITIVE: the same fixtures with one option on change the digest."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import _bipolar_calllog as CL
    finally:
        sys.path.pop(0)
    legacy = CL.record_call_log(statistics="legacy")
    assert len(legacy) == 237
    assert CL.digest(legacy) == _CALLLOG_DIGEST
    log = CL.record_call_log()
    assert (len(log), CL.digest(log)) == (_CALLLOG_CALLS_V0150, _CALLLOG_DIGEST_V0150)
    assert log == CL.record_call_log(statistics="fermi_dirac")


@pytest.mark.devsim_isolated("bitwise comparison of separate DEVSIM solves: a device left live by another test is iterated by the same joint Newton and moves the answer by ulps")
def test_inactive_options_bundle_is_bitwise_identical_to_none():
    """GATE R26/R27-off-b. options=BipolarOptions() (every field None) must be the SAME device as
    options=None: same node-model list, and a forward-biased pn diode solves to BITWISE-identical
    Potential/Electrons/Holes and contact currents."""
    from dynameta.carriers.bipolar_options import BipolarOptions
    res = {}
    for tag, opts in (("none", None), ("inactive", BipolarOptions())):
        dev = "off" + tag
        try:
            _device(dev, [(0.0, 5e-9, "lo"), (1e-6, 5e-10, None), (2e-6, 5e-9, "hi")],
                    "ifelse(x < 1e-6, -1e24, 1e24)", options=opts, fd=True, presolve=True)
            _solve(1e18, 1e-6)
            for vb in (0.1, 0.2, 0.3, 0.4):
                devsim.set_parameter(device=dev, name="lo_bias", value=vb)
                _solve(1e18, 1e-6)
            res[tag] = (sorted(devsim.get_node_model_list(device=dev, region="bulk")),
                        [_v(dev, nm).tobytes() for nm in ("Potential", "Electrons", "Holes")],
                        [devsim.get_contact_current(device=dev, contact="lo", equation=eq)
                         for eq in ("ElectronContinuityEquation", "HoleContinuityEquation")])
        finally:
            _drop(dev)
    assert res["none"][0] == res["inactive"][0]
    assert res["none"][1] == res["inactive"][1]
    assert res["none"][2] == res["inactive"][2]


# =====================================================================================================
# wiring: one shared kernel for the 3-D gate frame, the layered 2-D builder, TransportModel / Stacked3DSpec
# =====================================================================================================

def test_ohmic_offset_with_options_equals_the_pinned_contact_potential():
    """The 3-D gate reference phi_bi (ohmic_built_in_offset_V) and the potential the 1-D ohmic contact
    pins come from ONE kernel: with BGN + ionization + T_K = 200 K both polarities agree to 1e-14 V; and
    the converged contact node sits there. Without options it is the legacy value exactly."""
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.bipolar_options import (BGN_SI_KLAASSEN_1992, BipolarOptions,
                                                   IONIZATION_SI_P_B)
    T = 200.0
    nc, nv, ni = _si_at(T)
    opts = BipolarOptions(bgn=BGN_SI_KLAASSEN_1992, ionization=IONIZATION_SI_P_B, T_K=T)
    for N in (3e24, -3e24):
        dev = "wire1"
        try:
            _device(dev, [(0.0, 5e-9, "lo"), (200e-9, 5e-9, "hi")], "{:.10e}".format(N),
                    options=opts, n_i=ni, n_c=nc, n_v=nv)
            _solve(1e8, 1e-12)
            psi_contact = _v(dev, "Potential")[0]
            pinned = _v(dev, "EquilibriumPotential")[0]
        finally:
            _drop(dev)
        phi = BP.ohmic_built_in_offset_V(N, n_i_m3=ni, n_dos_m3=nc, n_dos_p_m3=nv,
                                         fd_enhancement=False, options=opts)
        assert phi == pytest.approx(pinned, abs=1e-14)
        assert psi_contact == pytest.approx(pinned, abs=1e-12)
    legacy = BP.ohmic_built_in_offset_V(1e24, n_i_m3=NI, n_dos_m3=NC300)
    assert BP.ohmic_built_in_offset_V(1e24, n_i_m3=NI, n_dos_m3=NC300,
                                      options=BipolarOptions()) == legacy


def test_transport_model_and_3d_spec_validate_and_thread_options():
    """Solver-free: TransportModel / Stacked3DSpec accept a BipolarOptions only on physics='bipolar_dd'
    (anything else would silently ignore it) and only of the right type; from_design threads it."""
    from dynameta.carriers.bipolar_options import AUGER_SI_DZIEWIOR_SCHMID_1977, BipolarOptions
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    from dynameta.constants import M_E
    from dynameta.materials import TransportModel
    opts = BipolarOptions(auger=AUGER_SI_DZIEWIOR_SCHMID_1977)
    common = dict(n_bg_m3=1e24, eps_static=11.7,
                  dos_mass_kg_of_n_m3=lambda n: np.full_like(np.asarray(n, float), 1.08 * M_E),
                  mobility_m2Vs_of_n_m3=lambda n: np.full_like(np.asarray(n, float), 0.135))
    with pytest.raises(ValueError, match="bipolar_dd"):
        TransportModel(physics="drift_diffusion", bipolar_options=opts, **common)
    with pytest.raises(TypeError):
        TransportModel(physics="bipolar_dd", n_i_m3=1e16, tau_srh_s=1e-7,
                       hole_mobility_m2Vs_of_n_m3=common["mobility_m2Vs_of_n_m3"],
                       bipolar_options={"auger": 1}, **common)
    tr = TransportModel(physics="bipolar_dd", n_i_m3=1e16, tau_srh_s=1e-7,
                        hole_mobility_m2Vs_of_n_m3=common["mobility_m2Vs_of_n_m3"],
                        bipolar_options=opts, **common)
    assert tr.bipolar_options is opts
    with pytest.raises(ValueError, match="bipolar_dd"):
        Stacked3DSpec(physics="equilibrium", bipolar_options=opts)
    s = Stacked3DSpec(physics="bipolar_dd", n_i_m3=1e16, tau_srh_s=1e-7, bipolar_options=opts)
    assert s.bipolar_options is opts


def test_layered_builder_threads_options_into_the_region():
    """The 2-D LayeredDevsimBuilder passes TransportModel.bipolar_options to setup_bipolar_region: a
    bipolar bar with Auger + BGN + ionization builds the option models on its region, its contacts pin
    EquilibriumPotential, and the zero-bias solve converges with n p = n_ie^2 in the region."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.bipolar_options import (AUGER_SI_DZIEWIOR_SCHMID_1977,
                                                   BGN_SI_DEL_ALAMO_1985, BipolarOptions,
                                                   IONIZATION_SI_P_B)
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.constants import M_E
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel
    from dynameta.sweep import BiasPoint

    def const(v):
        return lambda n: np.full_like(np.asarray(n, dtype=float), v)
    opts = BipolarOptions(auger=AUGER_SI_DZIEWIOR_SCHMID_1977, bgn=BGN_SI_DEL_ALAMO_1985,
                          ionization=IONIZATION_SI_P_B)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    reg.add(Material("Si_np", ConstantOptical(12.0 + 0j), transport=TransportModel(
        n_bg_m3=1e25, eps_static=11.7, dos_mass_kg_of_n_m3=const(1.08 * M_E), band_gap_eV=1.12,
        chi_eV=4.05, physics="bipolar_dd", mobility_m2Vs_of_n_m3=const(MU_N),
        hole_mobility_m2Vs_of_n_m3=const(MU_P), n_i_m3=NI, tau_srh_s=1e-7, dos_mass_p_kg=0.81 * M_E,
        bipolar_options=opts)))
    stack = Stack(layers=[Layer("si", 40e-9, "Si_np")], superstrate_material="air",
                  substrate_material="air")
    electrodes = [Electrode("anode", "si", "x_lo", role="biased"),
                  Electrode("cathode", "si", "x_hi", role="ground", fixed_voltage_V=0.0)]
    design = Design(name="r27_layered", unit_cell=UnitCell.square(300e-9), stack=stack,
                    electrodes=electrodes, materials=reg, mesh_2d=Mesh2DSpec())
    b = LayeredDevsimBuilder(design, mesh_name="r27l_m", device_name="r27l_d")
    try:
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cf = b.solve(BiasPoint({"anode": 0.0}, "eq"))
        models = set(devsim.get_node_model_list(device=b.device, region="si"))
        n = np.asarray(devsim.get_node_model_values(device=b.device, region="si", name="Electrons"))
        p = np.asarray(devsim.get_node_model_values(device=b.device, region="si", name="Holes"))
        nie = np.asarray(devsim.get_node_model_values(device=b.device, region="si", name="n_ie"))
    finally:
        try:
            devsim.delete_device(device=b.device)
            devsim.delete_mesh(mesh=b.mesh_name)
        except Exception:
            pass
        _R.clear(b.device)
    # ONE warning, the Mott-density UnreliableResultWarning (v0.14 row 3.2-11): with incomplete ionization the 1e19 cm^-3 bar
    # holds only n0 = 3.9e24 (39 % -- the no-Mott limit, above Si:P's 3.74e18) -- so u = n0/N_c = 0.14
    # and an FD built-in correction of 0.047 V_t, just under the BGN+FD double-counting advisory
    # (0.05 V_t; that one is gated separately below). n-type only: no del Alamo basis warning.
    assert [(w.category.__name__, "Mott" in str(w.message), "'si'" in str(w.message))
            for w in caught] == [("UnreliableResultWarning", True, True)], [str(w.message)[:120] for w in caught]
    assert {"UAuger", "n_ie", "IonizedNetDoping", "EquilibriumPotential", "SeedPsiShift"} <= models
    assert np.max(np.abs(n * p / nie ** 2 - 1.0)) < 1e-9
    assert "si" in cf.regions


def test_apparent_bgn_on_a_degenerate_fd_region_warns_once():
    """Apparent BGN (Boltzmann-extracted) on a DEGENERATE region with the FD g-factor on double-counts
    part of the degeneracy: setup emits ONE AdvisoryWarning naming the region (n0/N_c = 3.6 at 1e20
    cm^-3, FD built-in correction ~1 V_t). The same doping with fd_enhancement=False is silent."""
    import warnings
    from dynameta.carriers.bipolar_options import BGN_SI_DEL_ALAMO_1985, BipolarOptions
    from dynameta.warning_types import AdvisoryWarning
    for fd, expect in ((True, 1), (False, 0)):
        dev = "advfd"
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _device(dev, [(0.0, 5e-9, "lo"), (100e-9, 5e-9, "hi")], "1e26", fd=fd,
                        options=BipolarOptions(bgn=BGN_SI_DEL_ALAMO_1985))
        finally:
            _drop(dev)
        hits = [w for w in caught if issubclass(w.category, AdvisoryWarning)
                and "APPARENT bandgap-narrowing" in str(w.message) and "'bulk'" in str(w.message)]
        assert len(hits) == expect and len(caught) == expect, [str(w.message)[:100] for w in caught]
