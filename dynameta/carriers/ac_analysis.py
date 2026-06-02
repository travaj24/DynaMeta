"""
Small-signal AC analysis (DEVSIM ssac) -- the device's complex admittance Y(f) = G + i*omega*C at a
DC operating point, hence the intrinsic modulator capacitance and (with an access resistance) the
RC bandwidth f_3dB. This is the rigorous-linearization companion to the quasi-static dQ/dV in
analysis.gate_cv: ssac solves the device's small-signal response directly, and on a drift-diffusion
device it is frequency-RESOLVED (the carrier-dynamics roll-off), where the equilibrium Poisson
device gives a frequency-flat geometric/depletion C.

Two pieces:
  * setup_circuit_contact(device, contact): wire a contact to a circuit AC voltage source so an
    ssac solve can excite it (use INSTEAD of physics_equilibrium.setup_contact on the driven gate;
    the other contacts stay ordinary grounded contacts).
  * ssac_admittance(frequencies): after a DC solve, sweep frequency and extract Y(f) -> C(f), G(f)
    from the source current. In 1-D, C/G are per unit area (F/m^2, S/m^2).

DEVSIM ssac convention (validated against an exact parallel-plate cap, C = eps0 eps_r / d, in
validation/ac_capacitance.py): the voltage source 'V1' carries the device current at circuit node
'V1.I'; with a unit AC excitation the admittance is Y = -I_source / V_ac, so
C = -Im(I)/(omega*V_ac) (> 0 for a capacitor) and G = -Re(I)/V_ac. Requires DEVSIM.
"""

from __future__ import annotations

import math

import numpy as np
import devsim as ds

from dynameta.carriers import eq_registry as _R


def setup_circuit_contact(device: str, contact: str, *,
                          edge_charge_model: str = "PotentialEdgeFlux",
                          source_name: str = "V1", node_name: str = "vac") -> tuple:
    """Make `contact` a circuit-driven Dirichlet (Potential = circuit node `node_name`) with an AC
    voltage source `source_name` from that node to ground, so a small-signal AC (ssac) solve can
    excite it. Use this on the gate you want to AC-probe INSTEAD of the bias-parameter
    physics_equilibrium.setup_contact; other contacts stay ordinary grounded contacts. The
    `edge_charge_model` is the contact's displacement-flux model (the charge whose AC derivative is
    the small-signal current). Returns (source_name, node_name)."""
    ds.add_circuit_node(name=node_name, variable_update="default")
    ds.circuit_element(name=source_name, n1=node_name, n2="0", value=0.0, acreal=1.0, acimag=0.0)
    cn = "{}_circuit_dirichlet".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cn,
                          equation="Potential - {}".format(node_name))
    ds.contact_node_model(device=device, contact=contact, name="{}:Potential".format(cn),
                          equation="1")
    ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cn, node_name),
                          equation="-1")
    # Record via the equation registry (not a raw ds.contact_equation) so a Gummel / staged solve
    # that freezes Potential (eq_registry.delete_by_name -> reapply_by_name) restores this circuit
    # contact too -- a raw contact_equation would be deleted and never re-applied, silently dropping
    # the AC drive. (Parity with setup_contact_ohmic_bipolar_circuit.)
    _R.record_contact_equation(device, contact, name="PotentialEquation", node_model=cn,
                               edge_charge_model=edge_charge_model, circuit_node=node_name)
    return source_name, node_name


def ssac_admittance(frequencies, *, source_name: str = "V1", v_ac: float = 1.0):
    """Small-signal admittance Y(f) = G + i*omega*C of the device at its CURRENT DC operating point,
    via DEVSIM's ssac. The caller must have (1) attached an AC source with setup_circuit_contact and
    (2) solved DC (ds.solve(type='dc', ...)) first. Sweeps `frequencies` (Hz) and returns
    (freqs, C, G): C = -Im(I_source)/(omega*v_ac) [F or F/m^2 in 1-D], G = -Re(I_source)/v_ac
    [S or S/m^2]. A passive capacitor gives C > 0, G ~ 0 (validated)."""
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if np.any(freqs <= 0.0):
        raise ValueError("frequencies must be > 0 Hz")
    if not (float(v_ac) > 0.0):
        raise ValueError("v_ac must be > 0 (the excitation scale for Y = -I/V_ac); got "
                         "{!r}".format(v_ac))
    src_i = "{}.I".format(source_name)
    C = np.empty(freqs.size)
    G = np.empty(freqs.size)
    for i, f in enumerate(freqs):
        ds.solve(type="ac", frequency=float(f))
        i_re = float(ds.get_circuit_node_value(node=src_i, solution="ssac_real"))
        i_im = float(ds.get_circuit_node_value(node=src_i, solution="ssac_imag"))
        w = 2.0 * math.pi * float(f)
        C[i] = -i_im / (w * float(v_ac))             # Y = -I_source/V_ac ; C = Im(Y)/omega
        G[i] = -i_re / float(v_ac)                   # G = Re(Y)
    return freqs, C, G
