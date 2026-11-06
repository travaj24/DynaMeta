"""D1c TRAPS (v0.14.0 DEVSIM phase): bulk SRH at an arbitrary trap energy, several discrete levels,
trapped charge in Poisson, and interface traps (D_it) at a semiconductor/insulator interface --
physics_traps + the physics_bipolar_dd hooks. Every oracle is INDEPENDENT of the code under test
(closed forms written out here, scipy roots), and every gate is one the pre-D1c code cannot pass:

  (a) slab lifetime vs E_t   == tau_p0 (1 + n1/n0) + tau_n0 p1/n0   (midgap-only SRH gives ONE tau)
  (b) trap-compensated n, p  == charge-neutrality root (scipy)       (no trapped charge before)
  (c) MOS LF C-V stretch-out == psi_s - (Q_s + Q_it)/C_ox, d psi_s/dV_G = C_ox/(C_ox + C_s + C_it),
      flat-band shift        == -Q_f/C_ox                           (no interface charge before)
  (d) surface recombination  == sigma v_th N_it from the analytic minority profile (S = 0 before)
  (e) defaults OFF           == byte-identical solution arrays and model strings.

MEASURED 2026-11-03 (DEVSIM 2.11.2, box A) -- the numbers each gate's tolerance is set from are in the
test docstrings and in docs/audit/2026-11-03-devsim-traps.md."""
import contextlib
import math
import os
import sys

import numpy as np
import pytest

from dynameta.carriers import physics_traps as PT

V_T = 1.380649e-23 * 300.0 / 1.602176634e-19        # written out: the oracle does not import V_T
Q = 1.602176634e-19
EPS0 = 8.8541878128e-12
NI = 1.0e16


# ------------------------------------------------------------------------------------------------
# devsim-free
# ------------------------------------------------------------------------------------------------
def test_srh_kernel_is_the_legacy_string():
    """The shared kernel with default names IS the pre-D1c USRH string, character for character --
    the midgap channel reduces EXACTLY (not just to the same formula)."""
    assert PT.srh_rate_expr() == \
        "(Electrons*Holes - n_i^2)/(taup*(Electrons + n1) + taun*(Holes + p1))"


def test_midgap_n1_p1_bit_exact():
    n1, p1 = PT.n1_p1(0.0, 1.2345e16)
    assert n1 == 1.2345e16 and p1 == 1.2345e16          # exp(0.0) == 1.0 -> bit-identical legacy values
    n1, p1 = PT.n1_p1(0.2, NI)
    assert n1 * p1 == pytest.approx(NI * NI, rel=1e-14)


def test_sheet_levels_uniform_split():
    """Uniform D_it over the gap: bins of width <= kT, none straddling the neutral level, total
    density D_it * E_g, acceptor-like above E_0 and donor-like below (Nicollian & Brews)."""
    it = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, neutral_level_eV=0.1)
    lv = it.sheet_levels()
    e = np.array([x[0] for x in lv])
    n = np.array([x[1] for x in lv])
    assert n.sum() == pytest.approx(1e16 * 1.12, rel=1e-12)
    assert all(k == ("acceptor" if ee > 0.1 else "donor") for ee, _, k in lv)
    assert np.all(np.diff(e) <= V_T * (1 + 1e-9) + 1e-15)
    g = PT.InterfaceTraps(dit_eV_m2=1e16, distribution="gaussian", e_peak_eV=0.2, sigma_eV=0.05)
    tot = sum(x[1] for x in g.sheet_levels())
    assert tot == pytest.approx(1e16 * 0.05 * math.sqrt(2 * math.pi), rel=1e-5)   # +-5 sigma


def test_legacy_trapspec_enabled_refuses_and_points_to_the_working_api():
    """materials.TrapSpec was read by nothing: TransportModel(traps=TrapSpec(enabled=True)) built and solved
    with the trap SILENTLY ignored (user decision 2026-11-04). enabled=True now raises at construction
    AND on later mutation (TrapSpec.__setattr__), naming TrapLevel / InterfaceTraps and the keywords that
    work. Fails on 6595bde (no raise). enabled=False (the default) is unchanged."""
    from dynameta.materials import TransportModel, TrapSpec
    with pytest.raises(NotImplementedError, match="TrapLevel") as ei:
        TrapSpec(enabled=True)
    msg = str(ei.value)
    for kw in ("srh_traps", "srh_trap_energy_eV", "InterfaceTraps", "interface_traps"):
        assert kw in msg, kw
    off = TrapSpec()                                              # default: inert, as before
    assert off.enabled is False and off.N_trap_acceptor_m3 == 1.0e24
    with pytest.raises(NotImplementedError, match="not implemented"):
        off.enabled = True                                        # mutation cannot bypass it
    assert off.enabled is False
    off.g_trap = 2.0                                              # other fields stay writable
    tm = TransportModel(n_bg_m3=1e23, eps_static=11.7, dos_mass_kg_of_n_m3=lambda n: 1e-31,
                        traps=TrapSpec(enabled=False))
    assert tm.traps.enabled is False


def test_spec_validation():
    with pytest.raises(ValueError):
        PT.TrapLevel(energy_eV=0.1, kind="acceptor", tau_n0_s=1e-9, tau_p0_s=1e-9)   # no density
    with pytest.raises(ValueError):
        PT.TrapLevel(energy_eV=0.1, kind="neutral")                                   # no lifetimes
    with pytest.raises(ValueError):
        PT.TrapLevel(kind="weird", tau_n0_s=1e-9, tau_p0_s=1e-9)
    t = PT.TrapLevel(density_m3=1e21, sigma_n_m2=1e-19, sigma_p_m2=2e-19, v_th_n_ms=2e5,
                     v_th_p_ms=1e5, kind="donor")
    assert t.lifetimes_s() == pytest.approx((1 / (1e-19 * 2e5 * 1e21), 1 / (2e-19 * 1e5 * 1e21)))
    with pytest.raises(ValueError):
        PT.InterfaceTraps(dit_eV_m2=1e16)                       # uniform needs a band
    with pytest.raises(ValueError):
        PT.InterfaceTraps(sigma_n_m2=0.0)


# ------------------------------------------------------------------------------------------------
# DEVSIM harness (1-D devices; each test builds and deletes its own device)
# ------------------------------------------------------------------------------------------------
ds = pytest.importorskip("devsim")
from dynameta.carriers import physics_bipolar_dd as BP           # noqa: E402
from dynameta.carriers import physics_equilibrium as PE          # noqa: E402
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


def _solve(abs_e, rel=1e-12, it=80):
    with _quiet():
        ds.solve(type="dc", absolute_error=abs_e, relative_error=rel, maximum_iterations=it)


def _vals(dev, region, name):
    return np.array(ds.get_node_model_values(device=dev, region=region, name=name))


def _seed(dev, region):
    ds.node_model(device=dev, region=region, name="_seed",
                  equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, region)))
    for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"),
                 ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region=region, name=v, values=_vals(dev, region, m).tolist())
    PT.seed_kinetic_traps(dev, [region])      # v0.15 kinetic levels (tests/test_trap_kinetics.py); else no-op


def _slab(dev, *, length, net_doping, tau_n, tau_p, ps_end=5e-8, ps_mid=1e-6, **kw):
    """Uniform 1-D bipolar slab between two ohmic contacts + a G0 generation knob (RTerm_gen)."""
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
    BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=NI,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=tau_n,
                            tau_p_s=tau_p, fd_enhancement=False, **kw)
    BP.setup_equilibrium_seed_models(dev, "r")
    for c in ("a", "b"):
        BP.setup_contact_ohmic_bipolar(dev, c)


# ------------------------------------------------------------------------------------------------
# (e) defaults OFF: byte-identical
# ------------------------------------------------------------------------------------------------
def _diode_state(dev, **kw):
    _clear(dev)
    m = dev + "_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=2e-9, tag="a")
    ds.add_1d_mesh_line(mesh=m, pos=0.5e-6, ps=5e-10)
    ds.add_1d_mesh_line(mesh=m, pos=1e-6, ps=2e-9, tag="b")
    ds.add_1d_contact(mesh=m, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=m, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="r", tag1="a", tag2="b")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)
    ds.node_model(device=dev, region="r", name="NetDoping",
                  equation="ifelse(x < 0.5e-6, -1e23, 1e23)")
    BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=NI,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7,
                            tau_p_s=1e-7, **kw)
    BP.setup_equilibrium_seed_models(dev, "r")
    for c in ("a", "b"):
        BP.setup_contact_ohmic_bipolar(dev, c)
    _seed(dev, "r")
    _solve(1e17, 1e-8)
    for v in (0.1, 0.2, 0.3):
        ds.set_parameter(device=dev, name="a_bias", value=v)
        _solve(1e17, 1e-8)
    models = sorted(ds.get_node_model_list(device=dev, region="r"))
    out = {k: _vals(dev, "r", k).tobytes() for k in ("Potential", "Electrons", "Holes", "USRH",
                                                    "PotentialNodeCharge", "ElectronGeneration")}
    out["I"] = repr(ds.get_contact_current(device=dev, contact="a",
                                           equation="ElectronContinuityEquation"))
    _clear(dev)
    return models, out


def test_defaults_off_byte_identical():
    """traps=None / traps=() / srh_trap_energy_eV=0.0 against the bare legacy call on a forward-biased
    1-D Si diode: the node-model inventory and every solved array (Potential, Electrons, Holes, USRH,
    PotentialNodeCharge, ElectronGeneration, contact current) are BYTE-identical. The cross-VERSION
    half (this branch vs main 5cbd998, same fixture) is in validation/traps_oracles.py --old-vs-new."""
    ref_models, ref = _diode_state("trp_off0")
    for kw in ({"traps": None}, {"traps": ()}, {"srh_trap_energy_eV": 0.0},
               {"srh_trap_energy_eV": 0.0, "traps": []}):
        models, out = _diode_state("trp_off1", **kw)
        assert models == ref_models, kw
        assert out == ref, kw
    assert not any(m.startswith(("QTerm_", "RTerm_", "Trap", "It")) for m in ref_models)


# ------------------------------------------------------------------------------------------------
# (a) SRH lifetime vs E_t (uniform slab, uniform generation, low injection)
# ------------------------------------------------------------------------------------------------
def _slab_tau(dev, *, tau_n, tau_p, nd=1e21, G=1e24, length=60e-6, **kw):
    """tau measured at the centre of a long uniformly-generating n-type slab. Returns (tau_meas,
    n0, p0); the 1/cosh(L/2Lp) contact correction uses the oracle's own tau."""
    _slab(dev, length=length, net_doping=nd, tau_n=tau_n, tau_p=tau_p, **kw)
    ds.set_parameter(device=dev, region="r", name="G0", value=0.0)
    BP.add_recombination_term(dev, "r", "gen", "-G0")
    _seed(dev, "r")
    _solve(1e6)
    x = _vals(dev, "r", "x")
    ic = int(np.argmin(np.abs(x - 0.5 * length)))
    n0, p0 = _vals(dev, "r", "Electrons")[ic], _vals(dev, "r", "Holes")[ic]
    ds.set_parameter(device=dev, region="r", name="G0", value=G)
    _solve(1e6)
    dp = _vals(dev, "r", "Holes")[ic] - p0
    _clear(dev)
    return dp, n0, p0


def _tau_closed_form(E, tau_n0, tau_p0, n0, p0):
    n1, p1 = NI * math.exp(E / V_T), NI * math.exp(-E / V_T)
    tau = tau_p0 * (1 + n1 / n0) + tau_n0 * (p1 / n0)          # the requested n-type form
    tau_full = (tau_p0 * (n0 + n1) + tau_n0 * (p0 + p1)) / (n0 + p0)
    return tau, tau_full


@pytest.mark.parametrize("E", [-0.38, -0.2, 0.0, 0.3, 0.4])
def test_srh_lifetime_vs_trap_energy(E):
    """Gate (a). n-type slab N_D = 1e21 m^-3 (E_F - E_i = +0.298 eV), tau_n0 = 3e-10, tau_p0 = 1e-10 s
    (asymmetric so E_t and -E_t differ), G = 1e24 m^-3/s (low injection, dp/n0 < 1e-4), L = 60 um.
    The closed form spans 1.0e-10 .. 5.6e-9 s over this E_t range; midgap-only SRH gives 1.0e-10 s at
    every E_t, i.e. fails the -0.38/+0.3/+0.4 points by 2-55x. MEASURED (validation/traps_oracles.py,
    9 levels -0.4..+0.4 eV): |rel err| <= 2.1e-5, the largest at +-0.4 eV (the low-injection residual
    dp/n0); 3e-7 near midgap."""
    tau_n0, tau_p0, G, L = 3e-10, 1e-10, 1e24, 60e-6
    dp, n0, p0 = _slab_tau("trp_tau", tau_n=tau_n0, tau_p=tau_p0, G=G, length=L,
                           srh_trap_energy_eV=E)
    tau, tau_full = _tau_closed_form(E, tau_n0, tau_p0, n0, p0)
    Lp = math.sqrt(0.048 * V_T * tau_full)
    tau_meas = dp / G / (1.0 - 1.0 / math.cosh(0.5 * L / Lp))
    assert tau_meas == pytest.approx(tau, rel=1e-4)
    assert tau_meas == pytest.approx(tau_full, rel=1e-4)


def test_multilevel_lifetime():
    """Two recombination-only levels at +0.35 and -0.3 eV (TrapLevel list), the built-in channel
    pushed to tau = 1e3 s: low-injection 1/tau = sum_k 1/tau_k with each tau_k from the closed form.
    MEASURED rel err -1.7e-7 (tau = 4.504e-10 s)."""
    lv = [PT.TrapLevel(energy_eV=0.35, tau_n0_s=2e-10, tau_p0_s=1e-10),
          PT.TrapLevel(energy_eV=-0.30, tau_n0_s=5e-10, tau_p0_s=4e-10)]
    G, L = 1e24, 60e-6
    dp, n0, p0 = _slab_tau("trp_ml", tau_n=1e3, tau_p=1e3, G=G, length=L, traps=lv)
    inv = sum(1.0 / _tau_closed_form(t.energy_eV, t.tau_n0_s, t.tau_p0_s, n0, p0)[1] for t in lv)
    tau = 1.0 / inv
    Lp = math.sqrt(0.048 * V_T * tau)
    tau_meas = dp / G / (1.0 - 1.0 / math.cosh(0.5 * L / Lp))
    assert tau_meas == pytest.approx(tau, rel=1e-4)


# ------------------------------------------------------------------------------------------------
# (b) trapped charge at equilibrium: compensation vs a scipy neutrality root
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["acceptor_in_n", "donor_in_p"])
def test_trap_charge_compensation(case):
    """Gate (b). Equilibrium slab with a CHARGED level; the centre density must equal the root of
       n - p + N_A^- ... :  n - n_i^2/n - N_net - Q_trap(E_F)/q = 0,  f = 1/(1 + exp((E_t - E_F)/kT))
    solved by scipy.brentq on E_F (an independent Fermi-statistics oracle -- not the SRH occupancy
    expression the code uses). acceptor_in_n: N_D = 1e22, N_t = 6e21 at E_t - E_i = +0.33 eV (partly
    filled: n drops to ~4.1e21); donor_in_p: N_A = 1e22, N_t = 4e21 at -0.30 eV. Pre-D1c the centre
    sits at the trap-free N_D / N_A (+62 % / +52 %). MEASURED rel err 0 / 1.1e-15."""
    from scipy.optimize import brentq
    if case == "acceptor_in_n":
        nnet, lv = 1e22, PT.TrapLevel(energy_eV=0.33, density_m3=6e21, kind="acceptor",
                                      tau_n0_s=1e-9, tau_p0_s=1e-9)
    else:
        nnet, lv = -1e22, PT.TrapLevel(energy_eV=-0.30, density_m3=4e21, kind="donor",
                                       tau_n0_s=1e-9, tau_p0_s=1e-9)
    _slab("trp_q", length=4e-6, net_doping=nnet, tau_n=1e-7, tau_p=1e-7, ps_end=2e-9, ps_mid=5e-8,
          traps=[lv])
    _seed("trp_q", "r")
    _solve(1e10)                                               # ~1e-12 of the density scale
    x = _vals("trp_q", "r", "x")
    ic = int(np.argmin(np.abs(x - 2e-6)))
    n_sim, p_sim = _vals("trp_q", "r", "Electrons")[ic], _vals("trp_q", "r", "Holes")[ic]
    _clear("trp_q")

    def neutral(ef):                                        # E_F - E_i [eV]
        n, p = NI * math.exp(ef / V_T), NI * math.exp(-ef / V_T)
        f = 1.0 / (1.0 + math.exp((lv.energy_eV - ef) / V_T))
        qt = -lv.density_m3 * f if lv.kind == "acceptor" else lv.density_m3 * (1.0 - f)
        return (p - n + nnet + qt) / abs(nnet)

    ef = brentq(neutral, -0.6, 0.6, xtol=1e-15)
    n_ref, p_ref = NI * math.exp(ef / V_T), NI * math.exp(-ef / V_T)
    maj_sim, maj_ref = (n_sim, n_ref) if nnet > 0 else (p_sim, p_ref)
    assert maj_sim == pytest.approx(maj_ref, rel=1e-6)
    assert n_sim * p_sim == pytest.approx(NI * NI, rel=1e-6)
    assert abs(maj_sim / abs(nnet) - 1.0) > 0.3                 # the trap really compensates


# ------------------------------------------------------------------------------------------------
# (c) MOS capacitor: interface-trap LF C-V stretch-out and fixed-charge flat-band shift
# ------------------------------------------------------------------------------------------------
MOS_NA, MOS_TSI, MOS_TOX, MOS_EPSSI, MOS_EPSOX = 1e23, 0.6e-6, 10e-9, 11.7, 3.9
MOS_COX = MOS_EPSOX * EPS0 / MOS_TOX


def _mos(dev, it):
    """1-D p-Si (N_A = 1e17 cm^-3, 0.6 um) / SiO2 (10 nm) MOS cap: ohmic body, Potential-only gate.
    Interface traps (if any) on the si_ox interface. Returns psi_b (the body's pinned potential)."""
    _clear(dev)
    m = dev + "_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=20e-9, tag="body")
    ds.add_1d_mesh_line(mesh=m, pos=MOS_TSI, ps=0.05e-9, tag="i")
    ds.add_1d_mesh_line(mesh=m, pos=MOS_TSI + MOS_TOX, ps=0.5e-9, tag="gate")
    ds.add_1d_contact(mesh=m, name="body", tag="body", material="metal")
    ds.add_1d_contact(mesh=m, name="gate", tag="gate", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="si", tag1="body", tag2="i")
    ds.add_1d_region(mesh=m, material="SiO2", region="ox", tag1="i", tag2="gate")
    ds.add_1d_interface(mesh=m, name="si_ox", tag="i")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)
    ds.node_model(device=dev, region="si", name="NetDoping", equation=repr(-MOS_NA))
    BP.setup_bipolar_region(dev, "si", eps_static=MOS_EPSSI, n_dos_m3=2.8e25, n_i_m3=NI,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7,
                            tau_p_s=1e-7, fd_enhancement=False)
    BP.setup_equilibrium_seed_models(dev, "si")
    PE.setup_dielectric_region(dev, "ox", MOS_EPSOX)
    PE.setup_interface(dev, "si_ox")
    BP.setup_contact_ohmic_bipolar(dev, "body")
    PE.setup_contact(dev, "gate")
    if it is not None:
        PT.install_interface_traps(dev, "si_ox", it)
    psib = -V_T * math.log(MOS_NA / NI)
    _seed(dev, "si")
    nox = len(_vals(dev, "ox", "x"))
    ds.set_node_values(device=dev, region="ox", name="Potential", values=[psib] * nox)
    ds.set_parameter(device=dev, name="gate_bias", value=psib)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(dev, ceq)
    _solve(1e-10, 1e-12, 100)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(dev, ceq)
    _solve(1e12, 1e-10)
    return psib


def _mos_sweep(dev, psib, vgs):
    out = []
    for vg in vgs:
        ds.set_parameter(device=dev, name="gate_bias", value=psib + vg)
        _solve(1e12, 1e-10)
        pot = _vals(dev, "si", "Potential")
        out.append(pot[-1] - pot[0])
    return np.array(out)


def _mos_cv(dev, it):
    """V_G (relative to the trap-free flat band) vs psi_s, from -0.6 V (accumulation) to +1.2 V."""
    psib = _mos(dev, it)
    up = np.round(np.arange(0.0, 1.2001, 0.02), 10)
    dn = np.round(-np.arange(0.02, 0.6001, 0.02), 10)
    p_up = _mos_sweep(dev, psib, up)
    _mos_sweep(dev, psib, up[::-1])                            # walk back to 0 V
    p_dn = _mos_sweep(dev, psib, dn)
    _clear(dev)
    return np.concatenate([dn[::-1], up]), np.concatenate([p_dn[::-1], p_up])


# --- the independent analytic MOS (Sze closed-form Q_s; Nicollian-Brews uniform-D_it Q_it) ---------
_P0 = MOS_NA
_N0 = NI * NI / MOS_NA
_LD = math.sqrt(MOS_EPSSI * EPS0 * V_T / (Q * _P0))
_PHIB = V_T * math.log(MOS_NA / NI)


def _Qs(psi):
    b = psi / V_T
    F = math.sqrt(max(0.0, (math.expm1(-b) + b) + (_N0 / _P0) * (math.expm1(b) - b)))
    return -math.copysign(1.0, psi) * math.sqrt(2.0) * MOS_EPSSI * EPS0 * V_T / _LD * F


def _Cs(psi):
    h = 1e-6
    return -(_Qs(psi + h) - _Qs(psi - h)) / (2.0 * h)


def _int_f(u, a, b):
    """int_a^b dE / (1 + exp((E - u)/kT)), closed form."""
    g = lambda E: V_T * np.logaddexp(0.0, (u - E) / V_T)       # noqa: E731
    return float(g(a) - g(b))


def _Qit(psi, dit, eg):
    u = psi - _PHIB                                            # E_F - E_i at the surface [eV]
    return Q * dit * ((0.5 * eg - _int_f(u, -0.5 * eg, 0.0)) - _int_f(u, 0.0, 0.5 * eg))


def _Cit(psi, dit, eg):
    u = psi - _PHIB
    f = lambda E: 1.0 / (1.0 + math.exp((E - u) / V_T))        # noqa: E731
    return Q * dit * (f(-0.5 * eg) - f(0.5 * eg))


DIT, EG = 1e16, 1.12                                           # 1e12 cm^-2 eV^-1 across the Si gap


@pytest.fixture(scope="module")
def mos_dit_cv():
    return _mos_cv("trp_mos", PT.InterfaceTraps(dit_eV_m2=DIT, band_gap_eV=EG, recombination=False))


def test_mos_dit_lf_cv_stretchout(mos_dit_cv):
    """Gate (c-i). Uniform D_it = 1e12 cm^-2 eV^-1 (C_it/C_ox = 0.46), split at E_i. The simulated
    (V_G, psi_s) must lie on the analytic V_G = psi_s - (Q_s(psi_s) + Q_it(psi_s))/C_ox (Sze's exact
    Boltzmann Q_s; N&B Q_it by the closed-form Fermi integral) -- accumulation -> depletion -> weak
    inversion (psi_s from -0.07 to +0.66 V). MEASURED max |dV| = 7.4e-5 V (accumulation edge); the
    trap-free device misses this relation by up to 0.24 V."""
    V, P = mos_dit_cv
    va = np.array([p - (_Qs(p) + _Qit(p, DIT, EG)) / MOS_COX for p in P])
    assert np.max(np.abs(V - va)) < 5e-4
    va0 = np.array([p - _Qs(p) / MOS_COX if abs(p) > 0 else 0.0 for p in P])
    assert np.max(np.abs(V - va0)) > 0.1                        # the D_it is what moves the curve


def test_mos_dit_lf_cv_slope(mos_dit_cv):
    """Gate (c-i'). d psi_s/dV_G (centred differences of the sweep) vs the Nicollian-Brews LF
    relation C_ox/(C_ox + C_s + C_it), C_it = q^2 D_it (times the band-end Fermi factors). MEASURED
    max rel err 2.3e-4 (median 6.2e-5); the trap-free device misses it by up to 35 %."""
    V, P = mos_dit_cv
    slope = np.gradient(P, V)
    ana = np.array([MOS_COX / (MOS_COX + _Cs(p) + _Cit(p, DIT, EG)) for p in P])
    ana0 = np.array([MOS_COX / (MOS_COX + _Cs(p)) for p in P])
    keep = (np.abs(P) > 2e-3)
    keep[0] = keep[-1] = False                                 # one-sided ends
    assert np.max(np.abs(slope / ana - 1.0)[keep]) < 1e-3
    assert np.min(np.abs(slope / ana0 - 1.0)[keep]) > 0.01


def test_mos_fixed_charge_flatband_shift():
    """Gate (c-ii). A fixed interface charge Q_f = +q * 5e15 m^-2 (5e11 cm^-2) shifts the flat band by
    dV_FB = -Q_f/C_ox = -0.2321 V. V_FB is located where psi_s = 0 by interpolating each sweep. The
    trap-free device's V_FB is 0 by construction (checked). A linear interpolation of the 20 mV
    sweep is 1.3e-4 V off on curvature alone, so the crossing is a cubic through the six nearest
    points. MEASURED |error| 1.9e-6 V."""
    qf = Q * 5e15
    vgs = np.round(np.arange(-0.4, 0.1001, 0.02), 10)

    def vfb(it):
        psib = _mos("trp_fb", it)
        start = np.round(np.arange(0.0, -0.4001, -0.02), 10)
        _mos_sweep("trp_fb", psib, start)                      # walk down from 0 V
        p = _mos_sweep("trp_fb", psib, vgs)
        _clear("trp_fb")
        k = int(np.argmin(np.abs(p)))                           # cubic through the 6 nearest points
        sl = slice(max(0, k - 3), max(0, k - 3) + 6)
        c = np.polyfit(p[sl], vgs[sl], 3)
        return float(np.polyval(c, 0.0))

    v0 = vfb(None)
    v1 = vfb(PT.InterfaceTraps(fixed_charge_C_m2=qf, recombination=False))
    assert abs(v0) < 1e-5
    assert (v1 - v0) == pytest.approx(-qf / MOS_COX, abs=1e-4)


# ------------------------------------------------------------------------------------------------
# (d) surface recombination velocity
# ------------------------------------------------------------------------------------------------
SRV_ND, SRV_L, SRV_TAU, SRV_G = 1e21, 20e-6, 1e-8, 1e23


def _srv_profile(dev, it):
    """n-type slab (ohmic contact at x = 0, Si/SiO2 interface at x = L, flat-band Potential gate on the
    oxide), uniform generation G. Returns (x, dp)."""
    _clear(dev)
    m = dev + "_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=10e-9, tag="c")
    ds.add_1d_mesh_line(mesh=m, pos=0.5 * SRV_L, ps=50e-9)
    ds.add_1d_mesh_line(mesh=m, pos=SRV_L, ps=2e-9, tag="i")
    ds.add_1d_mesh_line(mesh=m, pos=SRV_L + 20e-9, ps=2e-9, tag="top")
    ds.add_1d_contact(mesh=m, name="c", tag="c", material="metal")
    ds.add_1d_contact(mesh=m, name="top", tag="top", material="metal")
    ds.add_1d_region(mesh=m, material="Si", region="si", tag1="c", tag2="i")
    ds.add_1d_region(mesh=m, material="SiO2", region="ox", tag1="i", tag2="top")
    ds.add_1d_interface(mesh=m, name="si_ox", tag="i")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)
    ds.node_model(device=dev, region="si", name="NetDoping", equation=repr(SRV_ND))
    BP.setup_bipolar_region(dev, "si", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=NI,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=SRV_TAU,
                            tau_p_s=SRV_TAU, fd_enhancement=False)
    BP.setup_equilibrium_seed_models(dev, "si")
    PE.setup_dielectric_region(dev, "ox", 3.9)
    PE.setup_interface(dev, "si_ox")
    BP.setup_contact_ohmic_bipolar(dev, "c")
    PE.setup_contact(dev, "top")
    psib = V_T * math.log(SRV_ND / NI)
    ds.set_parameter(device=dev, name="top_bias", value=psib)  # flat band
    ds.set_parameter(device=dev, region="si", name="G0", value=0.0)
    BP.add_recombination_term(dev, "si", "gen", "-G0")
    if it is not None:
        PT.install_interface_traps(dev, "si_ox", it)
    _seed(dev, "si")
    nox = len(_vals(dev, "ox", "x"))
    ds.set_node_values(device=dev, region="ox", name="Potential", values=[psib] * nox)
    _solve(1e6)
    p0 = _vals(dev, "si", "Holes")
    ds.set_parameter(device=dev, region="si", name="G0", value=SRV_G)
    _solve(1e6)
    x, dp = _vals(dev, "si", "x"), _vals(dev, "si", "Holes") - p0
    _clear(dev)
    return x, dp


def _dp_analytic(x, S):
    """D_p dp'' - dp/tau + G = 0, dp(0) = 0 (ohmic), -D_p dp'(L) = S dp(L) (surface sink)."""
    Dp = 0.048 * V_T
    Lp = math.sqrt(Dp * SRV_TAU)
    a = -SRV_G * SRV_TAU
    ch, sh = math.cosh(SRV_L / Lp), math.sinh(SRV_L / Lp)
    b = -(S * (SRV_G * SRV_TAU + a * ch) + Dp / Lp * a * sh) / (Dp / Lp * ch + S * sh)
    return SRV_G * SRV_TAU + a * np.cosh(x / Lp) + b * np.sinh(x / Lp)


@pytest.mark.parametrize("n_it", [1e15, 1e16, 1e17])
def test_surface_recombination_velocity(n_it):
    """Gate (d). One midgap interface level N_it, sigma_n = sigma_p = 1e-15 cm^2, v_th = 1e5 m/s,
    charge='none' (flat bands): S = sigma v_th N_it = 10 / 100 / 1000 m/s against D_p/L_p = 354 m/s.
    S is MEASURED by inverting the analytic surface value dp(L; S) for the simulated dp(L), and the
    whole profile is checked against dp(x; sigma v_th N_it). Pre-D1c there is no surface sink:
    S_meas = 0. G = 1e23 m^-3/s keeps dp/n0 ~ 1e-6: at G = 1e25 the majority-current field and the
    injection level bias dp by ~1e-4, which the weak S = 10 m/s sensitivity (d ln dp_s/d ln S ~ 0.03)
    amplifies to a 0.56 % miss. MEASURED (G = 1e23): S rel err -7.7e-4 / -1.1e-4 / -3.9e-5, profile
    max err <= 2.0e-5 of G tau."""
    from scipy.optimize import brentq
    sig, vth = 1e-19, 1e5
    it = PT.InterfaceTraps(distribution="discrete", levels=((0.0, n_it),), charge="none",
                           sigma_n_m2=sig, sigma_p_m2=sig, v_th_n_ms=vth, v_th_p_ms=vth)
    x, dp = _srv_profile("trp_srv", it)
    s_th = sig * vth * n_it
    s_meas = brentq(lambda S: _dp_analytic(np.array([SRV_L]), S)[0] - dp[-1], 0.0, 1e8, xtol=1e-12)
    assert s_meas == pytest.approx(s_th, rel=2e-3)
    assert np.max(np.abs(dp - _dp_analytic(x, s_th))) / (SRV_G * SRV_TAU) < 2e-3


def test_no_surface_traps_reflecting():
    """The oracle's other end: without interface traps the surface is reflecting (S = 0)."""
    x, dp = _srv_profile("trp_srv0", None)
    assert np.max(np.abs(dp - _dp_analytic(x, 0.0))) / (SRV_G * SRV_TAU) < 2e-3


# ------------------------------------------------------------------------------------------------
# plumbing: closed-form Jacobian entries, interface area, guards
# ------------------------------------------------------------------------------------------------
def test_interface_closed_form_derivatives_match_numeric():
    """The hand-written Electrons/Holes derivatives of the interface charge and rate terms (the Newton
    Jacobian entries) equal centred finite differences of the term models themselves at a
    non-equilibrium state (4 levels, acceptor + donor bins, a fixed charge, recombination on).
    Every term is node-local, so all nodes are perturbed at once. MEASURED max rel dev 1.4e-8 at
    h = 1e-4 (h = 1e-6 is roundoff-limited at 9.5e-7, h = 1e-3 truncation-limited at 9.8e-7)."""
    dev = "trp_jac"
    _mos(dev, None)
    it = PT.InterfaceTraps(distribution="discrete", charge="split", neutral_level_eV=0.0,
                           levels=((-0.3, 2e15), (-0.05, 1e15), (0.1, 3e15), (0.4, 1e15)),
                           sigma_n_m2=3e-19, sigma_p_m2=1e-19, fixed_charge_C_m2=1e-4)
    PT.install_interface_traps(dev, "si_ox", it)
    base = {w: _vals(dev, "si", w) for w in ("Electrons", "Holes")}
    base["Electrons"] = base["Electrons"] * 1e3 + 1e18          # off equilibrium: U != 0
    for w in base:
        ds.set_node_values(device=dev, region="si", name=w, values=base[w].tolist())
    mask = _vals(dev, "si", "ItArea0") > 0
    for term in ("QTerm_it0", "RTerm_it0"):
        for w in ("Electrons", "Holes"):
            mine = _vals(dev, "si", "{}:{}".format(term, w))[mask]
            h = 1e-4 * base[w]                                  # truncation/roundoff optimum
            vals = []
            for sgn in (+1.0, -1.0):
                ds.set_node_values(device=dev, region="si", name=w, values=(base[w] + sgn * h).tolist())
                vals.append(_vals(dev, "si", term)[mask])
            ds.set_node_values(device=dev, region="si", name=w, values=base[w].tolist())
            fd = (vals[0] - vals[1]) / (2.0 * h[mask])
            assert np.all(np.abs(fd) > 0), (term, w)
            assert np.max(np.abs(mine / fd - 1.0)) < 1e-6, (term, w)
    _clear(dev)


def test_interface_node_area_2d():
    """2-D: the per-node interface area sums to the interface length, end nodes own half an edge, and
    it is PER INTERFACE, whereas DEVSIM's built-in SurfaceArea is the SUM over all of a region's
    interfaces (node by node, MEASURED equal to a + b below) -- using it would put the top
    interface's D_it on the side interface's nodes too. DEVSIM gives a node shared by two interfaces
    to the FIRST one only (the (100, 50) nm corner is in 'if'), so 'side' owns 45 of its 50 nm."""
    dev, m = "trp_a2", "trp_a2_m"
    _clear(dev)
    ds.create_2d_mesh(mesh=m)
    for xx in (0.0, 30e-9, 100e-9, 120e-9):
        ds.add_2d_mesh_line(mesh=m, dir="x", pos=xx, ps=10e-9)
    for yy in (0.0, 50e-9, 60e-9):
        ds.add_2d_mesh_line(mesh=m, dir="y", pos=yy, ps=5e-9)
    ds.add_2d_region(mesh=m, region="si", material="Si", xl=0, xh=100e-9, yl=0, yh=50e-9)
    ds.add_2d_region(mesh=m, region="ox", material="SiO2", xl=0, xh=100e-9, yl=50e-9, yh=60e-9)
    ds.add_2d_region(mesh=m, region="ox2", material="SiO2", xl=100e-9, xh=120e-9, yl=0, yh=50e-9)
    ds.add_2d_interface(mesh=m, name="if", region0="si", region1="ox", xl=0, xh=100e-9,
                        yl=50e-9, yh=50e-9, bloat=1e-12)
    ds.add_2d_interface(mesh=m, name="side", region0="si", region1="ox2", xl=100e-9, xh=100e-9,
                        yl=0, yh=50e-9, bloat=1e-12)
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)
    a = PT.interface_node_area(dev, "si", "if")
    b = PT.interface_node_area(dev, "si", "side")
    y = _vals(dev, "si", "y")
    x = _vals(dev, "si", "x")
    assert a.sum() == pytest.approx(100e-9, rel=1e-12)
    assert b.sum() == pytest.approx(45e-9, rel=1e-12)
    assert np.all(a[np.abs(y - 50e-9) > 1e-15] == 0.0)
    corner = int(np.argmin(np.abs(x - 100e-9) + np.abs(y - 50e-9)))
    sa = _vals(dev, "si", "SurfaceArea")
    assert a[corner] == pytest.approx(5e-9, rel=1e-12) and b[corner] == 0.0
    assert np.all(a[b > 0] == 0.0)
    assert np.allclose(sa, a + b, rtol=1e-12, atol=0.0)           # same weights as DEVSIM's own
    _clear(dev)


def test_interface_trap_guards():
    _mos("trp_g", None)
    with pytest.raises(TypeError):
        PT.install_interface_traps("trp_g", "si_ox", [object()])
    _clear("trp_g")
    dev, m = "trp_g2", "trp_g2_m"                               # dielectric/dielectric -> ValueError
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=1e-9, tag="a")
    ds.add_1d_mesh_line(mesh=m, pos=5e-9, ps=1e-9, tag="i")
    ds.add_1d_mesh_line(mesh=m, pos=10e-9, ps=1e-9, tag="b")
    ds.add_1d_contact(mesh=m, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=m, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=m, material="ox", region="r0", tag1="a", tag2="i")
    ds.add_1d_region(mesh=m, material="ox", region="r1", tag1="i", tag2="b")
    ds.add_1d_interface(mesh=m, name="if", tag="i")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)
    PE.setup_dielectric_region(dev, "r0", 3.9)
    PE.setup_dielectric_region(dev, "r1", 3.9)
    with pytest.raises(ValueError):
        PT.install_interface_traps(dev, "if", PT.InterfaceTraps(fixed_charge_C_m2=1e-3))
    _clear(dev)


# ------------------------------------------------------------------------------------------------
# builder wiring (layered 2-D, 3-D spec, TransportModel)
# ------------------------------------------------------------------------------------------------
def _layered_bipolar_cap(srh_traps=()):
    from dynameta.constants import M_E
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel
    const = lambda v: (lambda n: np.full_like(np.asarray(n, float), v))      # noqa: E731
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cm", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    reg.add(Material("ox", ConstantOptical(2.1 + 0j), eps_static_dc=3.9))
    reg.add(Material("si", ConstantOptical(12.0 + 0j),
                     transport=TransportModel(n_bg_m3=1e23, eps_static=11.7,
                                              dos_mass_kg_of_n_m3=const(1.08 * M_E),
                                              physics="bipolar_dd", acceptor=True,
                                              mobility_m2Vs_of_n_m3=const(0.135),
                                              hole_mobility_m2Vs_of_n_m3=const(0.048),
                                              n_i_m3=NI, tau_srh_s=1e-7, srh_traps=srh_traps)))
    stack = Stack(layers=[Layer("si", 40e-9, "si"), Layer("ox", 10e-9, "ox")],
                  superstrate_material="air", substrate_material="air")
    el = [Electrode("bl", "si", "x_lo", role="ground", fixed_voltage_V=0.0),
          Electrode("br", "si", "x_hi", role="ground", fixed_voltage_V=0.0)]
    return Design(name="trp_lay", unit_cell=UnitCell.square(100e-9), stack=stack, electrodes=el,
                  materials=reg)


def test_layered_builder_installs_interface_and_bulk_traps():
    """LayeredDevsimBuilder(interface_traps={("si", "ox"): ...}) + TransportModel.srh_traps: the
    semiconductor region carries the trap terms, the interface weights equal DEVSIM's SurfaceArea on
    the single si/ox interface, and an unknown key raises."""
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    lv = (PT.TrapLevel(energy_eV=0.2, density_m3=1e21, kind="acceptor", tau_n0_s=1e-8,
                       tau_p0_s=1e-8),)
    it = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12)
    b = LayeredDevsimBuilder(_layered_bipolar_cap(lv), mesh_name="trp_lay_m", device_name="trp_lay",
                             interface_traps={("ox", "si"): it})
    try:
        b.build_device()
        models = ds.get_node_model_list(device="trp_lay", region="si")
        for m in ("QTerm_bulk_traps", "QTerm_it0", "RTerm_it0", "TrapOccupancy0", "ItArea0"):
            assert m in models, m
        w = _vals("trp_lay", "si", "ItArea0") * _vals("trp_lay", "si", "NodeVolume")
        # si has ONE interface (si/ox; the edge-metal strips are contacts), so the per-interface area
        # must equal DEVSIM's own SurfaceArea node-for-node. It is 95.4 of the 97 nm overlap: DEVSIM
        # gives the two end nodes to the edge contacts, which pin them, so they own no interface edge.
        assert np.allclose(w, _vals("trp_lay", "si", "SurfaceArea"), rtol=1e-9, atol=0.0)
        assert 0.95 * 97e-9 < w.sum() <= 97e-9
    finally:
        b.teardown()
    bad = LayeredDevsimBuilder(_layered_bipolar_cap(), mesh_name="trp_lay_m", device_name="trp_lay",
                               interface_traps={("si", "nope"): it})
    try:
        with pytest.raises(ValueError, match="matches no interface"):
            bad.build_device()
    finally:
        bad.teardown()


def test_spec_guards_need_bipolar():
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    from dynameta.materials import TransportModel
    with pytest.raises(ValueError, match="bipolar_dd"):
        Stacked3DSpec(interface_traps=PT.InterfaceTraps(fixed_charge_C_m2=1e-3))
    with pytest.raises(ValueError, match="bipolar_dd"):
        Stacked3DSpec(srh_trap_energy_eV=0.1)
    with pytest.raises(ValueError, match="bipolar_dd"):
        TransportModel(n_bg_m3=1e23, eps_static=11.7, dos_mass_kg_of_n_m3=lambda n: 1e-30,
                       srh_traps=(PT.TrapLevel(tau_n0_s=1e-9, tau_p0_s=1e-9),))
    s = Stacked3DSpec(physics="bipolar_dd", n_i_m3=1e16,
                      interface_traps=PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12))
    assert s.srh_traps == () and s.srh_trap_energy_eV == 0.0
