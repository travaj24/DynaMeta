"""Trap generation-recombination noise of KINETIC traps (DynaMeta v0.15, plan item 4.3.2 step K3; P3 section 1.5
K3 rows). Each kinetic level / interface bin is two Poissonian processes, electron exchange (capture +
emission, event rate P_n = [n (1-f) + n1' f]/tau_n0) and hole exchange (P_p = [p f + p1' (1-f)]/tau_p0),
each a two-point event between its carrier row and the trap row (noise_analysis single-carrier sources):

    S = q^2 V [2 P_n |y_t - y_n|^2 + 2 P_p |y_p - y_t|^2]          (mechanisms "trap_n", "trap_p")

ORACLES (independent of the code under test):
  * the FLUCTUATION-DISSIPATION THEOREM S_I = 4kT Re Y on equilibrium devices (Re Y from DEVSIM's ssac, a
    different solve from the noise sum), 1e-4 as the D2a gate B -- for a CHARGED bulk level on the gate-B
    long diode (the plan's xfail flips) and for kinetic interface bins on the conductance-method MOS;
  * the closed-form ambipolar GR Lorentzian (gate D1) for a nearly-empty-of-charge neutral kinetic level;
  * an equilibrated scipy adjoint of ds.get_matrix_and_rhs for DEVSIM's native field on the trap rows.
Load-bearing perturbations: drop the hole process, flip the trap-row sign, double-count a fast-trap pair,
count the contact-node events, each must break the FDT.
"""
import math

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
V = pytest.importorskip("validation.devsim_noise")

from dynameta.carriers import noise_analysis as NA            # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP        # noqa: E402
from dynameta.carriers import physics_traps as PT             # noqa: E402
from dynameta.warning_types import AdvisoryWarning            # noqa: E402

ds = V.ds
KT = 1.380649e-23 * 300.0                                     # re-typed (CODATA 2018)
FREQS = list(np.logspace(0, 9, 10))                           # 1 Hz .. 1 GHz


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()


def _charged(dynamic=True, **kw):
    """test_devsim_phase_composition's CHARGED_TRAP (an acceptor, N_t = 20 % of the gate-B doping)."""
    base = dict(energy_eV=0.15, density_m3=2e21, kind="acceptor", tau_n0_s=3e-7, tau_p0_s=5e-7)
    base.update(kw)
    return PT.TrapLevel(dynamic=dynamic, **base)


def _fdt_diode(traps, freqs=FREQS, mechanisms=None):
    """The gate-B long diode (validation/devsim_noise.py) with `traps`: S_I / 4kT Re Y - 1 per frequency,
    the spectrum and Y."""
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, traps=traps,
                 srh_trap_energy_eV=-0.1)
    kw = {} if mechanisms is None else {"mechanisms": mechanisms}
    sn = V.noise(freqs, **kw)
    Y = V.ssac_Y(freqs)
    return sn.S / (4 * KT * Y.real) - 1.0, sn, Y


# =====================================================================================================
# K3-1 / K3-2: the FDT with a charged kinetic level, and what breaks it
# =====================================================================================================

def test_fdt_with_a_charged_kinetic_level_all_frequencies():
    """K3-1 (the plan's xfail flips). The gate-B diode with the CHARGED acceptor level made KINETIC: S_I =
    4kT Re Y from 1 Hz to 1 GHz to 1e-4 (the D2a gate-B tolerance; ssac-limited). MEASURED max 1.4e-6.
    The quasi-static level misses by 9.9e-2 at 1 MHz (and raises the decision-39 advisory); the kinetic
    sources are a real share of S (trap_n + trap_p = 35 % at 100 kHz). The trap rows sit at the circuit
    contact's node (the trap row is kept there): the refusal is restricted to equations whose contact row
    is replaced (test_contact_row_refusal_both_sides)."""
    r, sn, _ = _fdt_diode((_charged(),))
    assert np.max(np.abs(r)) < 1e-4, r
    share = (sn.by_mechanism["trap_n"] + sn.by_mechanism["trap_p"]) / sn.S
    assert share[5] > 0.2
    with pytest.warns(AdvisoryWarning, match="QUASI-STATIC CHARGED traps"):
        rq, _, _ = _fdt_diode((_charged(dynamic=False),), freqs=[1e6])
    assert abs(rq[0]) > 1e-2


def _perturbed(monkeypatch, how):
    """Re-run K3-1 with ONE deliberate change to the trap sources (noise side only, the device is the same)."""
    real = NA._single_carrier_sources

    def drop_hole(device, region, eqs, mechanisms):
        return [s for s in real(device, region, eqs, mechanisms) if s.mechanism != "trap_p"]

    def flip_trap_row(device, region, eqs, mechanisms):
        out = []
        for s in real(device, region, eqs, mechanisms):
            rows = tuple(NA.SourceRow(r.region, r.equation, -r.coeff if r.equation.startswith("TrapEquation")
                                      else r.coeff, r.nodes) for r in s.rows)
            out.append(NA.TwoPointSource(s.mechanism, s.region, s.rate, rows, s.map_nodes))
        return out

    if how in ("drop_hole", "flip"):
        monkeypatch.setattr(NA, "_single_carrier_sources", drop_hole if how == "drop_hole" else flip_trap_row)
        r, _, _ = _fdt_diode((_charged(),), freqs=[1.0, 1e5, 1e7])
        return r
    V.bipolar_1d("fdt", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, traps=(_charged(),),
                 srh_trap_energy_eV=-0.1)
    if how == "double":            # the fast-trap pair of the SAME level next to its kinetic events
        args = ("trap0_n1", "trap0_p1", "trap0_taun", "trap0_taup")
        BP.add_recombination_term("fdt", "bulk", "dbl", "0", pair=PT.srh_rate_expr(*args, pair=True),
                                  derivs={"Electrons": "0", "Holes": "0"})
    sn = V.noise([1.0, 1e5, 1e7])
    Y = V.ssac_Y([1.0, 1e5, 1e7])
    return sn.S / (4 * KT * Y.real) - 1.0


@pytest.mark.parametrize("how", ["drop_hole", "flip", "double"])
def test_fdt_load_bearing(monkeypatch, how):
    """K3-2. Each perturbation must break the K3-1 FDT by > 1e-2 at some frequency (the plan's figure):
    (a) drop the hole-exchange source; (b) flip the trap-row coefficient (a wrong relative sign); (c) ALSO
    install the fast-trap pair source of the same level (double counting). MEASURED: audit note."""
    r = _perturbed(monkeypatch, how)
    assert np.max(np.abs(r)) > 1e-2, (how, r)


def test_fdt_breaks_with_the_contact_node_events(monkeypatch):
    """The contact-node rule (noise_analysis._single_carrier_sources, v0.15 K3): a kinetic level's events at
    an ohmic contact node are invisible to the DC / ssac system and carry no source. Counting them (the
    4.0.2 behaviour) misses the FDT by the two contact nodes' share: MEASURED +2.3e-3 at 1 Hz -- 23x the
    K3-1 tolerance (the per-node map at the 'hi' node: trap_n 8.2e-4 + trap_p 1.5e-3 of S)."""
    real = NA._single_carrier_sources

    def keep_contacts(device, region, eqs, mechanisms):
        saved = ds.get_contact_list
        try:
            ds.get_contact_list = lambda **kw: ()           # the contact-node zeroing sees no contact
            return real(device, region, eqs, mechanisms)
        finally:
            ds.get_contact_list = saved
    monkeypatch.setattr(NA, "_single_carrier_sources", keep_contacts)
    r, _, _ = _fdt_diode((_charged(),), freqs=[1.0, 1e5])
    assert np.max(np.abs(r)) > 10 * 1e-4, r


# =====================================================================================================
# the circuit-contact-node refusal, both sides
# =====================================================================================================

def test_contact_row_refusal_both_sides():
    """noise_spectrum refuses an event row on a circuit-contact node ONLY for an equation whose row that
    contact replaces (it carries a contact equation of that name) and that has no fix-up. (i) A kinetic
    trap row at the circuit contact's node -- kept by DEVSIM, no contact equation -- is accepted (K3-1 runs
    on it; asserted here by the inventory and a solve). (ii) A single-carrier term whose partner row is
    PotentialEquation (replaced by the circuit contact, no fix-up) is still refused, naming the row."""
    V.bipolar_1d("ref", *V._fdt_lines(), n_i=V.FDT_NI, tau=V.FDT_TAU, ae=1e8, traps=(_charged(),),
                 srh_trap_energy_eV=-0.1)
    hi = NA._contact_nodes("ref", "bulk", "hi")
    assert hi.size and "TrapEquation_trap0" not in ds.get_contact_equation_list(device="ref", contact="hi")
    sn = V.noise([1e3])
    assert sn.by_mechanism["trap_n"][0] > 0 and sn.by_mechanism["trap_p"][0] > 0
    BP.add_recombination_term("ref", "bulk", "badpartner", "0", carrier="electron", pair="1e10",
                              derivs={"Electrons": "0", "Holes": "0"},
                              partner=("PotentialEquation", 1.0), mechanism="trap_n")
    with pytest.raises(NotImplementedError, match="PotentialEquation.*replaces"):
        V.noise([1e3])
