"""Degenerate 1-D fixtures for the v0.15 density-scaled absolute tolerance (plan 4.4.1), shared by
tests/test_dc_solve_abs_tol.py and its measurement notes (docs/audit/2026-11-06-v0.15-tolerance-
wellposedness.md). Each builder takes `extended` (opt the device into DEVSIM extended precision right
after create_device, before any physics) and returns the step list to solve; `run_steps` drives the
DEVSIM Newton with info=True and records the iteration count of every continuation step.

  mos_accumulation  GATE C's classical leg (validation/bipolar_density_gradient.mos): n-type 1e24 m^-3
                    Fermi-Dirac bipolar DD, the wall potential ramped +0.15 V in 5 steps -> an
                    electron accumulation peak of 7.5e25 m^-3 at the wall while the bulk holes sit at
                    n_i^2/N = 1e8 m^-3 (the minority the relative criterion must still protect).
  ito_gate          validation/gated_dd's ITO bar: unipolar DD, n_bg = 4e26, 10 nm oxide, gate ramped
                    0 -> 1 V in 0.25 V steps (accumulation to ~1.4e27 m^-3).
  r27a_slab         the R27-a n+ slab (1e25 m^-3, one minority-blocking majority contact, holes kicked
                    3x off equilibrium; tests/test_bipolar_recombination_bgn_ionization.py) -- here
                    with the SRH-only physics, a single relaxation solve.
"""
import contextlib
import os
import sys

import numpy as np

from dynameta.carriers.devsim_env import import_devsim

ds = import_devsim()

from dynameta.carriers import eq_registry as _R                     # noqa: E402
from dynameta.carriers import extended_precision as XP              # noqa: E402
from dynameta.constants import HBAR, KB, M_E, T_REF                 # noqa: E402

NI = 1.0e16
MN, MP = 0.26 * M_E, 0.49 * M_E                 # the GATE C model masses (isotropic)


def nc_model(m, gv, T=T_REF):
    return 2.0 * gv * (m * KB * T / (2.0 * np.pi * HBAR ** 2)) ** 1.5


@contextlib.contextmanager
def quiet():
    sys.stdout.flush()
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def drop(dev):
    with quiet():
        for d in list(ds.get_device_list()):
            if d == dev:
                ds.delete_device(device=d)
        for m in list(ds.get_mesh_list()):
            if m == dev + "_m":
                ds.delete_mesh(mesh=m)
    _R.clear(dev)


def _mesh_1d(dev, lines, regions, interfaces=()):
    mesh = dev + "_m"
    drop(dev)
    with quiet():
        ds.create_1d_mesh(mesh=mesh)
        for pos, ps, tag in lines:
            if tag:
                ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, tag=tag)
            else:
                ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps)
        tags = [t for _, _, t in lines if t]
        ds.add_1d_contact(mesh=mesh, name=tags[0], tag=tags[0], material="metal")
        ds.add_1d_contact(mesh=mesh, name=tags[-1], tag=tags[-1], material="metal")
        for reg, mat, t1, t2 in regions:
            ds.add_1d_region(mesh=mesh, material=mat, region=reg, tag1=t1, tag2=t2)
        for name, tag in interfaces:
            ds.add_1d_interface(mesh=mesh, name=name, tag=tag)
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)


def mos_accumulation(dev, *, extended=False, N=1.0e24, delta=0.15, steps=5):
    """Returns (steps, regions): steps = list of callables, each setting one continuation value."""
    from dynameta.carriers import physics_bipolar_dd as BP
    _mesh_1d(dev, [(0.0, 0.05e-9, "wall"), (5e-9, 0.1e-9, None), (25e-9, 0.5e-9, None),
                   (80e-9, 2e-9, "sub")], [("s", "Si", "wall", "sub")])
    if extended:
        XP.enable(dev)
    nc, nv = nc_model(MN, 6), nc_model(MP, 2)
    with quiet():
        ds.node_model(device=dev, region="s", name="NetDoping", equation="{:.17e}".format(N))
        BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=nc, n_i_m3=NI,
                                mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-7,
                                tau_p_s=1e-7, fd_enhancement=True, n_dos_p_m3=nv)
        BP.setup_equilibrium_seed_models(dev, "s")
        BP.setup_contact_ohmic_bipolar(dev, "sub")
        for nm, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="s", name=nm, init_from=src)
        psi_b = BP.ohmic_built_in_offset_V(N, n_i_m3=NI, n_dos_m3=nc, n_dos_p_m3=nv,
                                           fd_enhancement=True)
        ds.set_parameter(device=dev, name="wall_bias", value=psi_b)
        ds.contact_node_model(device=dev, contact="wall", name="wall_psi",
                              equation="Potential - wall_bias")
        ds.contact_node_model(device=dev, contact="wall", name="wall_psi:Potential", equation="1")
        _R.record_contact_equation(dev, "wall", name="PotentialEquation", node_model="wall_psi")
        nn = len(ds.get_node_model_values(device=dev, region="s", name="x"))
        ds.set_node_values(device=dev, region="s", name="Potential", values=[psi_b] * nn)

    def setter(v):
        return lambda: ds.set_parameter(device=dev, name="wall_bias", value=v)
    return ([lambda: None] + [setter(psi_b + delta * (k + 1) / steps) for k in range(steps)]), ("s",)


def ito_gate(dev, *, extended=False, vmax=1.0, dv=0.25):
    from dynameta.carriers import physics_drift_diffusion as DD
    from dynameta.carriers import physics_equilibrium as EQ
    _mesh_1d(dev, [(0.0, 1e-9, "gate"), (10e-9, 2e-10, "mid"), (22e-9, 1e-9, "back")],
             [("ox", "oxide", "gate", "mid"), ("ito", "ito", "mid", "back")], [("ox_ito", "mid")])
    if extended:
        XP.enable(dev)
    with quiet():
        EQ.setup_dielectric_region(dev, "ox", eps_static=9.0)
        DD.setup_semiconductor_region_dd(dev, "ito", n_bg_m3=4e26, eps_static=9.5,
                                         dos_mass_kg=0.35 * M_E, mobility_m2Vs=30e-4)
        EQ.setup_interface(dev, "ox_ito")
        EQ.setup_contact(dev, "gate")
        DD.setup_contact_ohmic_dd(dev, "back")
        for reg in ("ox", "ito"):
            nx = len(ds.get_node_model_values(device=dev, region=reg, name="x"))
            ds.set_node_values(device=dev, region=reg, name="Potential", values=[0.0] * nx)
        nx = len(ds.get_node_model_values(device=dev, region="ito", name="x"))
        ds.set_node_values(device=dev, region="ito", name="Electrons", values=[4e26] * nx)
        ds.set_parameter(device=dev, name="gate_bias", value=0.0)
        ds.set_parameter(device=dev, name="back_bias", value=0.0)
        # the gated_dd recipe's step (1): a Poisson-only pre-solve (continuity frozen), Potential in V
        _R.delete_by_name(dev, "ElectronContinuityEquation")
        XP.solve(type="dc", solver_type="direct", absolute_error=1e10, relative_error=1e-10,
                 maximum_iterations=60)
        _R.reapply_by_name(dev, "ElectronContinuityEquation")

    def setter(v):
        return lambda: ds.set_parameter(device=dev, name="gate_bias", value=v)
    n = int(round(vmax / dv))
    return [lambda: None] + [setter(dv * (k + 1)) for k in range(n)], ("ito",)


def r27a_slab(dev, *, extended=False, nd=1.0e25):
    """SRH-only n+ slab, one majority contact (psi + Electrons pinned, holes zero-flux), holes x3."""
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    _mesh_1d(dev, [(0.0, 5e-9, "lo"), (100e-9, 5e-9, "hi")], [("bulk", "Si", "lo", "hi")])
    if extended:
        XP.enable(dev)
    with quiet():
        ds.node_model(device=dev, region="bulk", name="NetDoping", equation="{:.10e}".format(nd))
        BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=2.8e25, n_dos_p_m3=1.04e25,
                                n_i_m3=NI, mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048,
                                tau_n_s=1e-7, tau_p_s=1e-7)
        BP.setup_equilibrium_seed_models(dev, "bulk")
        ds.node_model(device=dev, region="bulk", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "bulk")))
        for var, src in (("Potential", "_seed_psi"), ("Electrons", "IntrinsicElectrons"),
                         ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="bulk", name=var,
                               values=ds.get_node_model_values(device=dev, region="bulk", name=src))
        PE.setup_contact(dev, "lo")
        ds.set_parameter(device=dev, name="lo_bias",
                         value=float(ds.get_node_model_values(device=dev, region="bulk",
                                                              name="_seed_psi")[0]))
        n0 = float(ds.get_node_model_values(device=dev, region="bulk", name="Electrons")[0])
        ds.set_parameter(device=dev, name="lo_n_value", value=n0)
        ds.contact_node_model(device=dev, contact="lo", name="lo_n", equation="Electrons - lo_n_value")
        ds.contact_node_model(device=dev, contact="lo", name="lo_n:Electrons", equation="1")
        _R.record_contact_equation(dev, "lo", name="ElectronContinuityEquation", node_model="lo_n",
                                   edge_current_model="ElectronCurrent")
        p = np.asarray(ds.get_node_model_values(device=dev, region="bulk", name="Holes"))
        ds.set_node_values(device=dev, region="bulk", name="Holes", values=list(3.0 * p))
    return [lambda: None], ("bulk",)


FIXTURES = {"mos_accumulation": (mos_accumulation, 1e-7),
            "ito_gate": (ito_gate, 1e-5),
            "r27a_slab": (r27a_slab, 1e-10)}


def run_steps(steps, *, abs_tol, rel_tol, max_iter=100):
    """Solve every continuation step with info=True. abs_tol: a float, or the callable
    dc_solve.default_abs_tol (resolved on each step's PRE-solve state over the live devices -- the
    fixture's own, as solve_dc(device) does).
    Returns [(iterations, converged, resolved_abs_tol)] -- stops at the first failure."""
    out = []
    for step in steps:
        step()
        a = abs_tol() if callable(abs_tol) else abs_tol
        with quiet():
            info = XP.solve(type="dc", solver_type="direct", absolute_error=a, relative_error=rel_tol,
                            maximum_iterations=max_iter, info=True)
        its = len(info.get("iterations") or ())
        out.append((its, bool(info.get("converged")), a))
        if not info.get("converged"):
            break
    return out


def snapshot(dev, regions, names=("Potential", "Electrons", "Holes")):
    out = {}
    for r in regions:
        have = ds.get_node_model_list(device=dev, region=r)
        for v in names:
            if v in have:
                out[(r, v)] = np.asarray(ds.get_node_model_values(device=dev, region=r, name=v),
                                         dtype=np.float64)
    return out
