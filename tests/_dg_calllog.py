"""Recorder for the defaults-OFF identity gate of the UNIPOLAR in-Newton density-gradient model
(tests/test_bipolar_density_gradient.py), the D2d twin of tests/_bipolar_calllog.py.

It builds a small unipolar DG fixture through the PUBLIC physics_density_gradient API exactly as a
v0.14-pre caller does (setup_dg_quantum_correction, setup_contact_dg, setup_dg_hard_wall,
seed_dg_from_solution, set_dg_gamma) while intercepting every DEVSIM model-, equation-, parameter- and
solution-defining call, and returns that ordered call log. The gate hashes it and compares to the
digest recorded on the integration base e46bd25 (BEFORE the bipolar DG landed): equal digests mean the
unipolar DG physics DEFINITION is character-for-character the pre-D2d one. No solve is run (solution
values depend on the BLAS build); seed_dg_from_solution reads the setup-time Electrons values.

Re-record (only when a sibling workstream changes the unipolar DG physics ON PURPOSE):
    cd <a checkout of the reference code> && python <this file>
"""
from __future__ import annotations

import hashlib
import json

_RECORDED = ("node_model", "edge_model", "contact_node_model", "equation", "contact_equation",
             "set_parameter", "node_solution", "edge_from_node_model", "set_node_values",
             "delete_equation", "delete_contact_equation")


def record_call_log():
    import devsim as ds
    from dynameta.constants import M_E
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_density_gradient as DG
    from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                           setup_semiconductor_region_dd)

    log = []
    saved = {}

    def _wrap(name, fn):
        def inner(*a, **kw):
            log.append([name, sorted((k, repr(v)) for k, v in kw.items())])
            return fn(*a, **kw)
        return inner

    for nm in _RECORDED:
        saved[nm] = getattr(ds, nm)
        setattr(ds, nm, _wrap(nm, saved[nm]))
    dev, mesh = "dgcl", "dgcl_m"
    try:
        for d in list(ds.get_device_list()):
            if d == dev:
                ds.delete_device(device=d)
        for m in list(ds.get_mesh_list()):
            if m == mesh:
                ds.delete_mesh(mesh=m)
        _R.clear(dev)
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="wall")
        ds.add_1d_mesh_line(mesh=mesh, pos=20e-9, ps=2e-9, tag="back")
        ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
        ds.add_1d_contact(mesh=mesh, name="back", tag="back", material="metal")
        ds.add_1d_region(mesh=mesh, material="ITO", region="bar", tag1="wall", tag2="back")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        mstar = 0.35 * M_E
        setup_semiconductor_region_dd(dev, "bar", n_bg_m3=4.0e26, eps_static=9.5,
                                      dos_mass_kg=mstar, mobility_m2Vs=0.004)
        setup_contact_ohmic_dd(dev, "back")
        DG.setup_dg_quantum_correction(dev, "bar", m_eff_kg=mstar, gamma=1.0)
        DG.setup_contact_dg(dev, "back", 4.0e26)
        DG.setup_dg_hard_wall(dev, "wall", lambda_pin_factor=8.0)
        DG.seed_dg_from_solution(dev, "bar")
        for fr in (0.25, 1.0):
            DG.set_dg_gamma(dev, "bar", fr)
        log.append(["dg_b_coefficient", repr(DG.dg_b_coefficient(mstar, 0.7))])
        ds.delete_device(device=dev)
        ds.delete_mesh(mesh=mesh)
        _R.clear(dev)
    finally:
        for nm, fn in saved.items():
            setattr(ds, nm, fn)
    return log


def digest(log) -> str:
    return hashlib.sha256(json.dumps(log, sort_keys=False).encode("utf-8")).hexdigest()


if __name__ == "__main__":                                  # pragma: no cover - re-recording aid
    import dynameta
    lg = record_call_log()
    print("dynameta:", dynameta.__file__)
    print("calls:", len(lg))
    print("digest:", digest(lg))
