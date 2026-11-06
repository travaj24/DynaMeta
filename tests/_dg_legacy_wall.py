"""The RETIRED unipolar density-gradient hard-wall rows (physics_density_gradient.setup_dg_hard_wall
before v0.15), kept as a TEST FIXTURE only -- the library name is now a deprecated shim onto
setup_dg_insulator_wall (plan 4.4.4, decision 53; docs/audit/2026-11-06-v0.15-dg-wall-strong-inversion.md).

Two users:
  * tests/test_density_gradient.py -- the "fails on the old code" half of the wall gates (the old rows
    are a carrier sink and converge to the wide-depletion branch) and the negative control that
    re-installs ONLY the pinned carrier row on the insulator wall (pin_carrier_row).
  * tests/_sgflux_reference.py's `dg` fixture -- the cross-version Scharfetter-Gummel gate compares
    solved arrays with tests/data/sgflux_6595bde.json, recorded on 6595bde WITH these rows; the gate
    is about the flux, so its wall stays the recorded one.

install_legacy_hard_wall makes exactly the DEVSIM calls of the pre-v0.15 setup_dg_hard_wall, in order,
with the same strings (copied from 48cfbc9). set_pin_fraction co-ramps the Lambda pin with the gamma
fraction (what set_dg_gamma did; it still does for one release, so calling both is the same value)."""
from __future__ import annotations

import numpy as np


def install_legacy_hard_wall(device: str, contact: str, *, lambda_pin_factor: float = 15.0,
                             carrier_row: bool = True) -> None:
    """carrier_row=False leaves out ONLY the pinned carrier row (pin_carrier_row): the negative
    control that isolates it as the sink (the u and Lambda pins stay; the Electrons row is natural)."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.constants import V_T
    ds = import_devsim()
    ds.set_parameter(device=device, name="wall_lambda_pin", value=0.0)
    ds.set_parameter(device=device, name="wall_lambda_pin_full", value=float(lambda_pin_factor) * V_T)
    region = ds.get_region_list(device=device, contact=contact)[0]
    n_d = float(ds.get_parameter(device=device, region=region, name="N_D"))
    u_floor = 1.0e-6 * np.sqrt(n_d)
    cu = "{}_qsqrtn_wall".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cu,
                          equation="QSqrtN - {:.16e}".format(u_floor))
    ds.contact_node_model(device=device, contact=contact, name="{}:QSqrtN".format(cu), equation="1")
    _R.record_contact_equation(device, contact, name="QSqrtNEquation", node_model=cu)
    cl = "{}_qlambda_wall".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=cl,
                          equation="QLambda + wall_lambda_pin")
    ds.contact_node_model(device=device, contact=contact, name="{}:QLambda".format(cl), equation="1")
    _R.record_contact_equation(device, contact, name="QLambdaEquation", node_model=cl)
    if carrier_row:
        pin_carrier_row(device, contact)


def pin_carrier_row(device: str, contact: str) -> None:
    """The old wall's CARRIER row, n = u^2 on the Electrons equation of the wall node -- the row that
    made the node a sink. (It cannot be added to setup_dg_insulator_wall, whose Lambda row is the same
    tie c - u^2: the two rows would be identical, a singular matrix -- measured, UMFPACK fails.)"""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    ce = "{}_electrons_wall".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=ce, equation="Electrons - QSqrtN*QSqrtN")
    ds.contact_node_model(device=device, contact=contact, name="{}:Electrons".format(ce), equation="1")
    ds.contact_node_model(device=device, contact=contact, name="{}:QSqrtN".format(ce),
                          equation="-2*QSqrtN")
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation", node_model=ce)


def set_pin_fraction(device: str, frac: float) -> None:
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    full = float(ds.get_parameter(device=device, name="wall_lambda_pin_full"))
    ds.set_parameter(device=device, name="wall_lambda_pin", value=frac * full)
