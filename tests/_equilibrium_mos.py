"""1-D equilibrium MOS fixture for tests/test_equilibrium_charges.py (DEVSIM, built directly from the
physics_equilibrium primitives, no builder):

    x = 0          gate contact (Potential Dirichlet, PE.setup_contact)
    0 .. T_OX      region "ox"   (dielectric, EPS_OX)
    T_OX           interface "ox_semi"
    T_OX .. +L     region "semi" (equilibrium semiconductor, N_BG, M_REL, EPS_S)
    x = T_OX + L   contact "body": declared (a 1-D DEVSIM mesh needs a contact at each end) but by
                   default WITHOUT a contact equation, i.e. a zero-field (Neumann) end. The equilibrium
                   closure n = N_c F_1/2((psi - [EF_ref] - Phi_c0)/V_t) fixes the potential reference by
                   itself, which is exactly the situation of an equilibrium film whose edge electrode is
                   a Fermi-level reference instead of a Dirichlet column.

The semiconductor is 200 nm (about 50 screening lengths at N_BG), so its far end is neutral bulk. Mesh
spacing 0.01 nm at the interface. Every helper here imports dynameta lazily, so this module can also
be exec'd against an older tree.
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

N_BG = 1.0e24            # m^-3 (eta_bg ~ -1.3: neither Boltzmann nor degenerate)
M_REL = 0.30
EPS_S = 9.5
EPS_OX = 3.9
T_OX = 10.0e-9
L_SEMI = 200.0e-9
EPS0 = 8.8541878128e-12
C_OX = EPS_OX * EPS0 / T_OX

# the definition calls whose keyword strings make up a "model string" log
LOGGED = ("node_model", "edge_model", "contact_node_model", "interface_model", "equation",
          "contact_equation", "interface_equation", "node_solution", "edge_from_node_model")


@contextlib.contextmanager
def quiet():
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


def clear(ds, dev):
    from dynameta.carriers import eq_registry as _R
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    _R.clear(dev)


def build_mesh(ds, dev, *, h_if=0.01e-9, t_ox=T_OX, l_semi=L_SEMI):
    m = dev + "_m"
    ds.create_1d_mesh(mesh=m)
    ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=0.05e-9, tag="gate")
    ds.add_1d_mesh_line(mesh=m, pos=t_ox, ps=h_if, tag="i")
    ds.add_1d_mesh_line(mesh=m, pos=t_ox + l_semi, ps=2.0e-9, tag="body")
    ds.add_1d_contact(mesh=m, name="gate", tag="gate", material="metal")
    ds.add_1d_contact(mesh=m, name="body", tag="body", material="metal")
    ds.add_1d_region(mesh=m, material="ox", region="ox", tag1="gate", tag2="i")
    ds.add_1d_region(mesh=m, material="semi", region="semi", tag1="i", tag2="body")
    ds.add_1d_interface(mesh=m, name="ox_semi", tag="i")
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=dev)


def build_mos(ds, dev, *, fermi_ref_param=None, body_contact=False, **mesh_kw):
    """The physics of the fixture on a fresh device. `fermi_ref_param` is passed to
    setup_semiconductor_region only when given (so the call is valid on trees without it)."""
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.constants import M_E
    clear(ds, dev)
    build_mesh(ds, dev, **mesh_kw)
    kw = {} if fermi_ref_param is None else {"fermi_ref_param": fermi_ref_param}
    with quiet():
        PE.setup_dielectric_region(dev, "ox", EPS_OX)
        PE.setup_semiconductor_region(dev, "semi", n_bg_m3=N_BG, eps_static=EPS_S,
                                      dos_mass_kg=M_REL * M_E, **kw)
        PE.setup_interface(dev, "ox_semi")
        PE.setup_contact(dev, "gate")
        if body_contact:
            PE.setup_contact(dev, "body")


def solve(ds, *, abs_err=1e-14, rel_err=1e-6, it=80):
    with quiet():
        ds.solve(type="dc", absolute_error=abs_err, relative_error=rel_err, maximum_iterations=it)


def vals(ds, dev, region, name):
    return np.asarray(ds.get_node_model_values(device=dev, region=region, name=name), dtype=np.float64)


def set_gate(ds, dev, v, *, steps=1):
    """Walk the gate to v in `steps` equal steps from its present value (Newton continuation)."""
    v0 = float(ds.get_parameter(device=dev, name="gate_bias"))
    for k in range(1, steps + 1):
        ds.set_parameter(device=dev, name="gate_bias", value=v0 + (v - v0) * k / steps)
        solve(ds)


def psi_s(ds, dev):
    """Potential at the oxide/semiconductor interface node (first node of 'semi')."""
    x = vals(ds, dev, "semi", "x")
    return float(vals(ds, dev, "semi", "Potential")[int(np.argmin(x))])


def float_free_repr(v):
    """repr() of a definition-call keyword with every float masked (model STRINGS only: the numbers come
    from libm transcendentals whose last bit is platform-dependent)."""
    if isinstance(v, float):
        return "<float>"
    return repr(v)
