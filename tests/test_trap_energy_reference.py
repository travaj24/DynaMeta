"""Per-level trap ENERGY REFERENCE (v0.14 D1c follow-up): TrapLevel / InterfaceTraps
energy_reference in {"intrinsic" (default), "conduction", "valence"}, the Sentaurus-style
fromMidBandGap / fromCondBand / fromValBand choice. A band-referenced level keeps its distance to the
NARROWED band edge under bandgap narrowing: n1 = N_c exp(-(E_c - E_t)/kT), p1 = n_ie^2/n1 (valence:
the mirror), while the default "intrinsic" level rides the narrowed effective intrinsic level,
n1 = n_ie exp((E_t - E_ie)/kT).

Gates (independent closed forms / scipy; every one FAILS on f7c82db, which has no energy_reference):
  (a) BGN slab lifetime with a level at E_c - E_t = 0.25 eV == the closed form with n1 from the
      narrowed conduction edge, for conduction_fraction gamma = 0.2 and 0.8; the "intrinsic" reading of
      the same DLTS number (E_t - E_i = kT ln(N_c/n_i) - 0.25) is off by the factor quantified here.
  (b) no BGN, or a symmetric split: the three references give the same n1/p1 for equivalent energies.
  (c) n1 p1 = n_ie^2 on every reference (node-evaluated, BGN on).
  (d) the default is byte-identical -- tests/test_traps.py and tests/test_devsim_phase_composition.py
      run unchanged (and the kernel strings are pinned below).
  (e) equilibrium trapped charge with BGN + a band-referenced level == a scipy neutrality root.

FINDING (stated in docs/audit/2026-11-03-devsim-traps.md section 9): in this Boltzmann model the
split gamma does NOT enter SRH at all. The narrowed band diagram has E_c - E_ie = kT ln(N_c/n_ie) and
E_ie - E_v = kT ln(N_v/n_ie) whatever gamma is (gamma only moves both edges relative to psi), so for
either reference n1/p1 -- hence tau -- is gamma-independent; the two references differ by
exp(dEg/2kT) in n1 (and p1), not by a gamma-dependent factor. Gates (a)/(b) pin that.
"""
import contextlib
import math
import os
import sys

import numpy as np
import pytest

from dynameta.carriers import physics_traps as PT

KT = 1.380649e-23 * 300.0 / 1.602176634e-19                     # written out (oracle independence)
Q = 1.602176634e-19
NI = 1.0e16
NC = NV = 2.8e25
# Slotboom (Si) apparent BGN, written out here independently of BGNModel.delta_Eg_eV
SLOT = dict(E_ref_eV=6.92e-3, N_ref_m3=1.3e23, C=0.5)


def _slotboom(N):
    x = math.log(N / SLOT["N_ref_m3"])
    return SLOT["E_ref_eV"] * (x + math.sqrt(x * x + SLOT["C"]))


# ------------------------------------------------------------------------------------------------
# devsim-free: kernel strings and the numeric conversion
# ------------------------------------------------------------------------------------------------
def test_default_kernel_strings_unchanged():
    """(d) the intrinsic reference IS the shipped kernel, character for character."""
    assert PT.srh_rate_expr() == \
        "(Electrons*Holes - n_i^2)/(taup*(Electrons + n1) + taun*(Holes + p1))"
    for bgn in (False, True):
        assert PT.srh_rate_expr(bgn=bgn, reference="intrinsic") == PT.srh_rate_expr(bgn=bgn)
        assert PT.srh_occupancy_expr(bgn=bgn, reference="intrinsic") == PT.srh_occupancy_expr(bgn=bgn)
    assert PT.srh_rate_expr(bgn=True) == ("(Electrons*Holes - n_ie^2)/(taup*(Electrons + n1*n_ie/n_i)"
                                          " + taun*(Holes + p1*n_ie/n_i))")
    assert PT.level_reference_densities(0.13, "intrinsic", n_i_m3=NI) == PT.n1_p1(0.13, NI)
    lv = PT.TrapLevel(energy_eV=0.2, tau_n0_s=1e-9, tau_p0_s=1e-9)
    assert lv.energy_reference == "intrinsic"
    assert PT.equilibrium_trap_levels([PT.TrapLevel(energy_eV=0.2, density_m3=1e21, kind="donor",
                                                    tau_n0_s=1e-9, tau_p0_s=1e-9)]) == \
        ((0.2, 1e21, "donor"),)                                  # the shipped 3-tuple


@pytest.mark.parametrize("ref", ["intrinsic", "conduction", "valence"])
def test_n1_p1_product_numeric(ref):
    """(c, numeric half) n1 p1 = n_i^2 on every reference, and the band references are N exp(-E/kT)."""
    n1, p1 = PT.level_reference_densities(0.25, ref, n_i_m3=NI, N_c_m3=NC, N_v_m3=NV)
    assert n1 * p1 == pytest.approx(NI * NI, rel=1e-13)
    if ref == "conduction":
        assert n1 == pytest.approx(NC * math.exp(-0.25 / KT), rel=1e-13)
    if ref == "valence":
        assert p1 == pytest.approx(NV * math.exp(-0.25 / KT), rel=1e-13)


def test_no_bgn_references_equivalent_numeric():
    """(b, no BGN) E_t - E_i = E  <=>  E_c - E_t = kT ln(N_c/n_i) - E  <=>  E_t - E_v = kT ln(N_v/n_i) + E."""
    for E in (-0.3, 0.0, 0.17):
        a = PT.level_reference_densities(E, "intrinsic", n_i_m3=NI)
        b = PT.level_reference_densities(KT * math.log(NC / NI) - E, "conduction", n_i_m3=NI, N_c_m3=NC)
        c = PT.level_reference_densities(KT * math.log(NV / NI) + E, "valence", n_i_m3=NI, N_v_m3=NV)
        for x in (b, c):
            assert x[0] == pytest.approx(a[0], rel=1e-12) and x[1] == pytest.approx(a[1], rel=1e-12)


def test_reference_validation():
    with pytest.raises(ValueError, match="energy_reference"):
        PT.TrapLevel(energy_eV=0.2, tau_n0_s=1e-9, tau_p0_s=1e-9, energy_reference="midgap")
    with pytest.raises(ValueError, match=">= 0"):
        PT.TrapLevel(energy_eV=-0.1, tau_n0_s=1e-9, tau_p0_s=1e-9, energy_reference="conduction")
    with pytest.raises(ValueError, match="neutral_level_eV"):
        PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, energy_reference="conduction")
    with pytest.raises(ValueError, match="N_c / N_v"):
        PT.level_reference_densities(0.2, "valence", n_i_m3=NI)
    # a band-referenced uniform D_it spans [0, E_g]; "above the neutral level" is toward E_c
    it = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, energy_reference="conduction",
                           neutral_level_eV=0.56)
    lv = it.sheet_levels()
    assert min(e for e, _, _ in lv) > 0.0 and max(e for e, _, _ in lv) < 1.12
    assert all(k == ("acceptor" if e < 0.56 else "donor") for e, _, k in lv)
    assert sum(n for _, n, _ in lv) == pytest.approx(1e16 * 1.12, rel=1e-12)


# ------------------------------------------------------------------------------------------------
# DEVSIM
# ------------------------------------------------------------------------------------------------
ds = pytest.importorskip("devsim")
from dynameta.carriers import physics_bipolar_dd as BP           # noqa: E402
from dynameta.carriers import bipolar_options as BO              # noqa: E402
from dynameta.carriers import eq_registry as _R                  # noqa: E402


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush()
    saved = os.dup(1)
    dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)


def _clear(dev):
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    _R.clear(dev)


def _vals(dev, name, region="r"):
    return np.array(ds.get_node_model_values(device=dev, region=region, name=name))


def _opts(gamma):
    if gamma is None:
        return None
    return BO.BipolarOptions(bgn=BO.BGNModel(kind="slotboom", conduction_fraction=gamma, **SLOT))


def _slab(dev, *, net_doping, gamma, traps, tau_n=1e-3, tau_p=1e-3, length=60e-6, ps_end=5e-8,
          ps_mid=1e-6):
    _clear(dev)
    m = dev + "_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=ps_end, tag="a")
    ds.add_1d_mesh_line(mesh=m, pos=0.5 * length, ps=ps_mid)
    ds.add_1d_mesh_line(mesh=m, pos=length, ps=ps_end, tag="b")
    ds.add_1d_contact(mesh=m, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=m, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="r", tag1="a", tag2="b")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)
    ds.node_model(device=dev, region="r", name="NetDoping", equation=repr(float(net_doping)))
    BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=NC, n_dos_p_m3=NV, n_i_m3=NI,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=tau_n, tau_p_s=tau_p,
                            fd_enhancement=False, options=_opts(gamma), traps=traps)
    BP.setup_equilibrium_seed_models(dev, "r")
    for c in ("a", "b"):
        BP.setup_contact_ohmic_bipolar(dev, c)
    ds.set_parameter(device=dev, region="r", name="G0", value=0.0)
    BP.add_recombination_term(dev, "r", "gen", "-G0")
    ds.node_model(device=dev, region="r", name="_seed",
                  equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "r")))
    for v, mm in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"),
                  ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region="r", name=v, values=_vals(dev, mm).tolist())


def _solve(abs_e):
    with _quiet():
        ds.solve(type="dc", absolute_error=abs_e, relative_error=1e-12, maximum_iterations=80)


# ---- (a) BGN slab lifetime, conduction-referenced level -----------------------------------------
NA_A, G_A, L_A = 1e25, 1e24, 60e-6                 # p-type 1e19 cm^-3: Slotboom dEg = 60.5 meV
TN_A, TP_A, EC_A = 1e-10, 1e-6, 0.25               # level E_c - 0.25 eV; tau_p0 >> tau_n0 so n1 matters


def _lifetime(gamma, level):
    dev = "ref_tau"
    _slab(dev, net_doping=-NA_A, gamma=gamma, traps=[level])
    _solve(1e12)
    x = _vals(dev, "x")
    ic = int(np.argmin(np.abs(x - 0.5 * L_A)))
    n0 = _vals(dev, "Electrons")[ic]
    ds.set_parameter(device=dev, region="r", name="G0", value=G_A)
    _solve(1e12)
    dn = _vals(dev, "Electrons")[ic] - n0
    _clear(dev)
    return dn


def _tau_closed(n1, p1, nie):
    p0 = 0.5 * NA_A + math.sqrt(0.25 * NA_A * NA_A + nie * nie)
    n0 = nie * nie / p0
    return (TP_A * (n0 + n1) + TN_A * (p0 + p1)) / (n0 + p0)


def _measured_tau(dn, tau):
    Ln = math.sqrt(0.135 * KT * tau)
    return dn / G_A / (1.0 - 1.0 / math.cosh(0.5 * L_A / Ln))


@pytest.fixture(scope="module")
def bgn_lifetimes():
    dEg = _slotboom(NA_A)
    nie = NI * math.exp(dEg / (2.0 * KT))
    out = {"dEg": dEg, "nie": nie}
    cond = PT.TrapLevel(energy_eV=EC_A, energy_reference="conduction", tau_n0_s=TN_A, tau_p0_s=TP_A)
    e_int = KT * math.log(NC / NI) - EC_A          # the same DLTS number read against the UNdoped E_i
    intr = PT.TrapLevel(energy_eV=e_int, tau_n0_s=TN_A, tau_p0_s=TP_A)
    for g in (0.2, 0.8):
        out[("cond", g)] = _lifetime(g, cond)
        out[("intr", g)] = _lifetime(g, intr)
    return out


@pytest.mark.parametrize("gamma", [0.2, 0.8])
def test_bgn_lifetime_conduction_reference(bgn_lifetimes, gamma):
    """(a) closed form with n1 = N_c exp(-(E_c - E_t)/kT) from the NARROWED conduction edge and
    p1 = n_ie^2/n1, n_ie = n_i exp(dEg/2kT) (dEg = Slotboom at N_A = 1e25: 60.5 meV). MEASURED rel err
    -1.8e-7 at both gamma (tau = 2.767e-10 s)."""
    nie = bgn_lifetimes["nie"]
    n1 = NC * math.exp(-EC_A / KT)
    tau = _tau_closed(n1, nie * nie / n1, nie)
    assert _measured_tau(bgn_lifetimes[("cond", gamma)], tau) == pytest.approx(tau, rel=1e-4)


def test_bgn_lifetime_intrinsic_reading_differs(bgn_lifetimes):
    """(a) the default "intrinsic" reading of the same quoted level puts n1 (and p1) a factor
    exp(dEg/2kT) = 3.22 higher (the level rides E_ie, which sits dEg/2 closer to the narrowed E_c), so
    tau rises by the closed-form ratio: MEASURED 6.695e-10 / 2.767e-10 = 2.419 (closed form matched to
    -5.7e-7); and gamma changes NEITHER lifetime (the finding in the module docstring)."""
    nie, dEg = bgn_lifetimes["nie"], bgn_lifetimes["dEg"]
    n1c = NC * math.exp(-EC_A / KT)
    tau_c = _tau_closed(n1c, nie * nie / n1c, nie)
    s = math.exp(dEg / (2.0 * KT))
    n1i = n1c * s
    tau_i = _tau_closed(n1i, nie * nie / n1i, nie)
    for g in (0.2, 0.8):
        ti = _measured_tau(bgn_lifetimes[("intr", g)], tau_i)
        tc = _measured_tau(bgn_lifetimes[("cond", g)], tau_c)
        assert ti == pytest.approx(tau_i, rel=1e-4)
        assert ti / tc == pytest.approx(tau_i / tau_c, rel=2e-4)
        assert ti / tc > 2.0                                     # measurably different
    for kind in ("cond", "intr"):                                # gamma-independent (finding)
        a, b = bgn_lifetimes[(kind, 0.2)], bgn_lifetimes[(kind, 0.8)]
        assert abs(a / b - 1.0) < 1e-8


# ---- (b) + (c) node-evaluated n1', p1' under BGN ---------------------------------------------------
def _node_n1p1(dev, k, reference):
    p = "trap{}_".format(k)
    n1e, p1e, _ = PT.srh_reference_densities(p + "n1", p + "p1", bgn=True, reference=reference)
    ds.node_model(device=dev, region="r", name="_n1e", equation=n1e)
    ds.node_model(device=dev, region="r", name="_p1e", equation=p1e)
    return _vals(dev, "_n1e"), _vals(dev, "_p1e")


@pytest.mark.parametrize("gamma", [None, 0.5, 0.2, 0.8])
def test_references_equivalent_and_product(gamma):
    """(b) equivalent quoted energies (in the NARROWED band diagram: E_c - E_t = kT ln(N_c/n_ie) - E,
    E_t - E_v = kT ln(N_v/n_ie) + E) give the same node n1'/p1' on all three references -- with no BGN,
    the symmetric split, and (finding) the asymmetric splits too; (c) n1' p1' = n_ie^2 on every
    reference, node by node."""
    dev = "ref_eq"
    nd = 1e25
    dEg = _slotboom(nd) if gamma is not None else 0.0
    nie = NI * math.exp(dEg / (2.0 * KT))
    E = 0.12
    lv = [PT.TrapLevel(energy_eV=E, tau_n0_s=1e-9, tau_p0_s=1e-9),
          PT.TrapLevel(energy_eV=KT * math.log(NC / nie) - E, energy_reference="conduction",
                       tau_n0_s=1e-9, tau_p0_s=1e-9),
          PT.TrapLevel(energy_eV=KT * math.log(NV / nie) + E, energy_reference="valence",
                       tau_n0_s=1e-9, tau_p0_s=1e-9)]
    _slab(dev, net_doping=nd, gamma=gamma, traps=lv, length=2e-6, ps_mid=2e-7)
    if gamma is None:
        ni2 = NI * NI
        got = [(ds.get_parameter(device=dev, region="r", name="trap{}_n1".format(k)),
                ds.get_parameter(device=dev, region="r", name="trap{}_p1".format(k))) for k in range(3)]
        for n1, p1 in got:
            assert n1 * p1 == pytest.approx(ni2, rel=1e-12)
            assert n1 == pytest.approx(got[0][0], rel=1e-12) and p1 == pytest.approx(got[0][1], rel=1e-12)
    else:
        node_nie = _vals(dev, "n_ie")
        assert node_nie[0] == pytest.approx(nie, rel=1e-12)       # the oracle's n_ie is the region's
        ref = _node_n1p1(dev, 0, "intrinsic")
        for k, r in ((1, "conduction"), (2, "valence")):
            n1e, p1e = _node_n1p1(dev, k, r)
            assert np.allclose(n1e, ref[0], rtol=1e-11) and np.allclose(p1e, ref[1], rtol=1e-11)
        for k, r in enumerate(("intrinsic", "conduction", "valence")):
            n1e, p1e = _node_n1p1(dev, k, r)
            assert np.allclose(n1e * p1e, node_nie ** 2, rtol=1e-12, atol=0.0)
    _clear(dev)


# ---- (e) equilibrium trapped charge, BGN + band-referenced levels --------------------------------
@pytest.mark.parametrize("case", ["acceptor_conduction_n", "donor_valence_p"])
def test_bgn_trapped_charge_band_reference(case):
    """(e) Slotboom BGN (gamma = 0.2), N = 1e24 m^-3 (dEg = 29 meV, n_ie = 1.75 n_i). acceptor_conduction_n:
    an acceptor level at E_c - 0.10 eV, N_t = 5e23, in N_D = 1e24; donor_valence_p: the mirror at
    E_v + 0.10 eV in N_A = 1e24. Oracle: scipy brentq on p - n + N + Q_t/q = 0 with n p = n_ie^2 and
    the FERMI occupancy written in the narrowed band diagram, f = 1/(1 + exp((E_t - E_F)/kT)) with
    E_c - E_F = kT ln(N_c/n). Checked at the slab centre AND at the contact nodes (the trap-aware
    contact rule uses the same reference). The intrinsic reading of the same quoted level compensates
    measurably differently (asserted). MEASURED: majority 0.724 N, centre/contacts rel <= 1.1e-15; the
    intrinsic reading is 8.3 % off."""
    from scipy.optimize import brentq
    N0, NT, DE = 1e24, 5e23, 0.10
    ntype = case.startswith("acceptor")
    dEg = _slotboom(N0)
    nie = NI * math.exp(dEg / (2.0 * KT))
    lv = PT.TrapLevel(energy_eV=DE, density_m3=NT, kind="acceptor" if ntype else "donor",
                      energy_reference="conduction" if ntype else "valence",
                      tau_n0_s=1e-9, tau_p0_s=1e-9)
    nnet = N0 if ntype else -N0

    def neutral(ln_n):
        n = math.exp(ln_n)
        p = nie * nie / n
        if ntype:                                     # E_t - E_F = (E_c - E_F) - DE = kT ln(Nc/n) - DE
            f = 1.0 / (1.0 + math.exp(math.log(NC / n) - DE / KT))
            qt = -NT * f
        else:                                         # E_t - E_F = -(E_F - E_v) + DE = -kT ln(Nv/p) + DE
            f = 1.0 / (1.0 + math.exp(-math.log(NV / p) + DE / KT))
            qt = NT * (1.0 - f)
        return (p - n + nnet + qt) / N0

    n_ref = math.exp(brentq(neutral, math.log(1e0), math.log(1e27), xtol=1e-15))
    maj_ref = n_ref if ntype else nie * nie / n_ref
    dev = "ref_q"
    _slab(dev, net_doping=nnet, gamma=0.2, traps=[lv], length=4e-6, ps_end=2e-9, ps_mid=5e-8)
    _solve(1e10)
    x = _vals(dev, "x")
    maj = _vals(dev, "Electrons" if ntype else "Holes")
    ic = int(np.argmin(np.abs(x - 2e-6)))
    for k in (0, ic, x.size - 1):
        assert maj[k] == pytest.approx(maj_ref, rel=1e-6), k
    assert abs(maj_ref / N0 - 1.0) > 0.1                          # the trap really compensates
    _clear(dev)
    # the intrinsic reading of the same quoted level (against the undoped E_i) lands elsewhere
    e_int = (KT * math.log(NC / NI) - DE) if ntype else -(KT * math.log(NV / NI) - DE)
    li = PT.TrapLevel(energy_eV=e_int, density_m3=NT, kind=lv.kind, tau_n0_s=1e-9, tau_p0_s=1e-9)
    _slab(dev, net_doping=nnet, gamma=0.2, traps=[li], length=4e-6, ps_end=2e-9, ps_mid=5e-8)
    _solve(1e10)
    maj_i = _vals(dev, "Electrons" if ntype else "Holes")[ic]
    _clear(dev)
    assert abs(maj_i / maj_ref - 1.0) > 0.02


def test_reference_composes_with_dg_scale():
    """The density-gradient `scale` (00e2c39) composes in the ONE kernel AFTER the reference
    conversion: n1' p1' = n_i'^2 s^2 on every reference, and scale=None is the unscaled output
    character for character. Checked symbolically on the strings and numerically by evaluation."""
    for bgn in (False, True):
        for ref in ("intrinsic", "conduction", "valence"):
            base = PT.srh_reference_densities("A", "B", bgn=bgn, reference=ref)
            assert PT.srh_reference_densities("A", "B", bgn=bgn, reference=ref, scale=None) == base
            got = PT.srh_reference_densities("A", "B", bgn=bgn, reference=ref, scale="S")
            assert got == ("{}*S".format(base[0]), "{}*S".format(base[1]), "{}*S^2".format(base[2]))
            env = dict(A=3.7e21, B=1e32 / 3.7e21, n_ie=4.1e16, n_i=1e16, n_i_sq=None, S=1.9)
            ev = [eval(x.replace("^", "**"), {}, env) for x in got]          # noqa: S307 (test strings)
            ni2 = (4.1e16 if bgn else 1e16) ** 2 * 1.9 ** 2
            if ref == "intrinsic" and bgn:
                ni2 = (4.1e16 ** 2) * 1.9 ** 2
            assert ev[2] == pytest.approx(ni2, rel=1e-14)
            assert ev[0] * ev[1] == pytest.approx(ev[2], rel=1e-14)
    assert PT.srh_rate_expr(bgn=True, scale="DgNieFactor", reference="intrinsic") == \
        PT.srh_rate_expr(bgn=True, scale="DgNieFactor")
