"""Recorder for the defaults-OFF identity gate of the R26/R27 bipolar options
(tests/test_bipolar_recombination_bgn_ionization.py).

It builds two small bipolar fixtures through the PUBLIC bipolar API WITHOUT the new `options` keyword
(so it runs unchanged on v0.13.0 / main 5cbd998) while intercepting every DEVSIM model-, equation-,
parameter- and solution-defining call, and returns that ordered call log. The gate hashes it and
compares to the digest recorded on v0.13.0: equal digests mean the defaults-off physics DEFINITION is
character-for-character the pre-v0.14 one (every expression string, every parameter value, every
equation binding, in order). Solution values are NOT part of the log (they depend on the BLAS build).

Re-record (only when a sibling workstream changes the base bipolar physics ON PURPOSE):
    cd <a checkout of the reference code> && python <this file>
"""
from __future__ import annotations

import hashlib
import json

_RECORDED = ("node_model", "edge_model", "contact_node_model", "equation", "contact_equation",
             "set_parameter", "node_solution", "edge_from_node_model", "add_circuit_node",
             "circuit_element")


def record_call_log():
    import devsim as ds
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP

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
    try:
        for dev, fd, circuit in (("cl_fd", True, False), ("cl_bz", False, True)):
            mesh = dev + "_m"
            for d in list(ds.get_device_list()):
                if d == dev:
                    ds.delete_device(device=d)
            for m in list(ds.get_mesh_list()):
                if m == mesh:
                    ds.delete_mesh(mesh=m)
            _R.clear(dev)
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=5e-9, tag="lo")
            ds.add_1d_mesh_line(mesh=mesh, pos=100e-9, ps=1e-9)
            ds.add_1d_mesh_line(mesh=mesh, pos=200e-9, ps=5e-9, tag="hi")
            ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
            ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="bulk", tag1="lo", tag2="hi")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            ds.node_model(device=dev, region="bulk", name="NetDoping",
                          equation="ifelse(x < 1e-7, -1e24, 3e25)")
            BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1.0e16,
                                    mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7,
                                    tau_p_s=2e-7, fd_enhancement=fd, n_dos_p_m3=1.04e25)
            BP.setup_equilibrium_seed_models(dev, "bulk")
            BP.setup_contact_ohmic_bipolar(dev, "lo")
            if circuit:
                BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="cl_n", source_name="Vcl")
            else:
                BP.setup_contact_ohmic_bipolar(dev, "hi")
            ds.node_model(device=dev, region="bulk", name="_seed_psi",
                          equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "bulk")))
            log.append(["ohmic_built_in_offset_V", repr(BP.ohmic_built_in_offset_V(
                3e25, n_i_m3=1e16, n_dos_m3=2.8e25, n_dos_p_m3=1.04e25, fd_enhancement=fd))])
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
