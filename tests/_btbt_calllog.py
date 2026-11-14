"""Recorder for the BTBT defaults-off identity gate (v0.15 plan 4.1.2, P7; tests/test_devsim_btbt_presets.py).

It builds a 1-D slab and a 2-D tensor-mesh slab through the PUBLIC bipolar API with btbt=<one set>
(SILICON_HURKX by default -- so it runs unchanged on the pre-v0.15 tree) while intercepting every DEVSIM
model-, equation-, parameter- and solution-defining call, and returns the ordered call log. The gate
hashes it and compares to the digest recorded on 66f6185 (the v0.15 integration commit this work was cut
from): equal digests mean a single BTBT set is installed character-for-character as before (every model
name, expression string, parameter value and equation binding, in order). No solve, no circuit.

CANONICAL FORM (2026-11-10). By default the log is recorded with correctly rounded transcendentals
(tests/_calllog.py, tests/_cr_libm.py), so its digest no longer depends on the platform's C library;
canonical=False records the raw log (the platform's libm) the pre-2026-11-10 digest was taken of.

Re-record (only on an intended change of the BTBT definition):
    cd <a checkout of the reference code> && PYTHONPATH=. python <this file>
(on an export of an older tree, copy this file, tests/_calllog.py and tests/_cr_libm.py into its tests/)
"""
from __future__ import annotations

import json
import sys

from _calllog import digest, intercepting, recording, skeleton  # noqa: F401  (re-exported for the gates)

_RECORDED = ("node_model", "edge_model", "element_model", "contact_node_model", "equation",
             "contact_equation", "set_parameter", "node_solution", "edge_from_node_model",
             "element_from_edge_model", "element_from_node_model")


def record_call_log(btbt=None, tag="btcl", canonical=True):
    """The ordered call log of setup_bipolar_region(..., btbt=btbt) on a 1-D and a 2-D slab (devices
    <tag>_1d / <tag>_2d, deleted afterwards). btbt None -> SILICON_HURKX. canonical (2026-11-10): record with
    correctly rounded transcendentals (module docstring); False: with the platform's libm."""
    with recording(canonical):
        return _record(btbt, tag)


def _record(btbt, tag):
    import numpy as np
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.tunneling import SILICON_HURKX
    ds = import_devsim()
    prm = SILICON_HURKX if btbt is None else btbt
    log = []
    devs = []
    try:
        with intercepting(ds, _RECORDED, log):
            for dim in (1, 2):
                dev = "{}_{}d".format(tag, dim)
                mesh = dev + "_m"
                devs.append((dev, mesh))
                _R.clear(dev)
                if dim == 1:
                    ds.create_1d_mesh(mesh=mesh)
                    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="a")
                    ds.add_1d_mesh_line(mesh=mesh, pos=8e-9, ps=1e-9, tag="b")
                    ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
                    ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
                    ds.add_1d_region(mesh=mesh, material="Si", region="r", tag1="a", tag2="b")
                else:
                    n = 3
                    g = np.linspace(0.0, 6e-9, n + 1)
                    X, Y = np.meshgrid(g, g, indexing="ij")
                    coords = np.column_stack([X.ravel(), Y.ravel(), 0.0 * X.ravel()])
                    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
                    el = []
                    for i in range(n):
                        for j in range(n):
                            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                            el += [2, 0, int(a), int(b), int(c), 2, 0, int(a), int(c), int(d)]
                    ds.create_gmsh_mesh(mesh=mesh, coordinates=[float(c) for c in coords.ravel()],
                                        physical_names=["r"], elements=el)
                    ds.add_gmsh_region(mesh=mesh, gmsh_name="r", region="r", material="Si")
                ds.finalize_mesh(mesh=mesh)
                ds.create_device(mesh=mesh, device=dev)
                ds.node_model(device=dev, region="r", name="NetDoping", equation="1e22")
                BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                                        mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7,
                                        tau_p_s=1e-7, fd_enhancement=False, btbt=prm)
    finally:
        for dev, mesh in devs:
            if dev in ds.get_device_list():
                ds.delete_device(device=dev)
            if mesh in ds.get_mesh_list():
                ds.delete_mesh(mesh=mesh)
            _R.clear(dev)
    return log


if __name__ == "__main__":                                    # pragma: no cover - re-record helper
    lg = record_call_log()
    print(len(lg), digest(lg))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            json.dump(lg, f, indent=0)
