"""Recorder for the defaults-OFF identity gate of the v0.15.1 valley-resolved density gradient
(tests/test_valley_density_gradient.py): the BIPOLAR density-gradient definition with no electron_valleys --
setup_bipolar_dg on both carriers (+ the quantum SRH term), setup_contact_bipolar_dg, setup_dg_insulator_wall,
seed_bipolar_dg(walls=...), the gamma ramp and the quasi-equilibrium closure on both carriers
(physics_bipolar_dd.setup_quasi_equilibrium_carrier) -- on a small p-type MOS bar, for statistics
'fermi_dirac' (the default) and 'boltzmann'. No Newton solve runs: the carriers are seeded from the
charge-neutral models and Potential from the equilibrium seed, so every recorded value is a deterministic
function of the code (no BLAS in the log).

It uses only the public API of 6f9044d-era trees (no electron_valleys keyword), so it runs unchanged on an
export of the base commit aed0aa9; the digest pinned in the test was recorded there AND on the valley tree
(equal). Canonical recording (correctly rounded transcendentals, tests/_calllog.py).

Re-record: cd <a checkout> && python tests/_bipolar_dg_calllog.py (copy this file, _calllog.py and
_cr_libm.py into an older tree's tests/).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from _calllog import digest, intercepting, recording, skeleton  # noqa: F401  (re-exported for the gates)

_RECORDED = ("node_model", "edge_model", "contact_node_model", "equation", "contact_equation",
             "set_parameter", "node_solution", "edge_from_node_model", "set_node_values",
             "delete_equation", "delete_contact_equation")


def record_call_log(canonical: bool = True):
    """The call log, recorded in a CHILD process on the caller's sys.path (a fresh DEVSIM: no device or
    registry state of the caller can leak in)."""
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
        subprocess.run([sys.executable, os.path.abspath(__file__), "--emit", out,
                        "--canonical" if canonical else "--raw"], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.remove(out)


def _record_in_this_process(canonical=True):
    with recording(canonical):
        return _record()


def _record():
    import devsim as ds
    from dynameta.constants import M_E
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_density_gradient as DG

    log = []
    with intercepting(ds, _RECORDED, log):
        for dev, statistics in (("bdgcl_fd", None), ("bdgcl_bz", "boltzmann")):
            mesh = dev + "_m"
            for d in list(ds.get_device_list()):
                if d == dev:
                    ds.delete_device(device=d)
            for m in list(ds.get_mesh_list()):
                if m == mesh:
                    ds.delete_mesh(mesh=m)
            _R.clear(dev)
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=0.2e-9, tag="wall")
            ds.add_1d_mesh_line(mesh=mesh, pos=5e-9, ps=1e-9)
            ds.add_1d_mesh_line(mesh=mesh, pos=40e-9, ps=4e-9, tag="sub")
            ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
            ds.add_1d_contact(mesh=mesh, name="sub", tag="sub", material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="s", tag1="wall", tag2="sub")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            ds.node_model(device=dev, region="s", name="NetDoping", equation="-3e23")
            fd = statistics is None
            BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=3.2e25, n_i_m3=1.0e16,
                                    mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-7, tau_p_s=1e-7,
                                    fd_enhancement=fd, n_dos_p_m3=1.8e25,
                                    **({} if statistics is None else {"statistics": statistics}))
            BP.setup_equilibrium_seed_models(dev, "s")
            BP.setup_contact_ohmic_bipolar(dev, "sub")
            for nm, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
                ds.set_node_values(device=dev, region="s", name=nm, init_from=src)
            ds.node_model(device=dev, region="s", name="_seed_psi",
                          equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "s")))
            ds.set_node_values(device=dev, region="s", name="Potential", init_from="_seed_psi")
            DG.setup_bipolar_dg(dev, "s", m_eff_n_kg=0.26 * M_E, m_eff_p_kg=0.49 * M_E,
                                gamma_n=DG.GAMMA_WALL, gamma_p=DG.GAMMA_WALL)
            DG.setup_contact_bipolar_dg(dev, "sub")
            DG.setup_dg_insulator_wall(dev, "wall")
            DG.set_dg_gamma(dev, "s", 0.1)
            DG.seed_bipolar_dg(dev, "s", walls=("wall",))
            for fr in (0.3, 1.0):
                DG.set_dg_gamma(dev, "s", fr)
            for c in ("electrons", "holes"):
                BP.setup_quasi_equilibrium_carrier(dev, "s", c, contact="sub")
            ds.delete_device(device=dev)
            ds.delete_mesh(mesh=mesh)
            _R.clear(dev)
    return log


if __name__ == "__main__" and sys.argv[1:2] == ["--emit"]:   # the child of record_call_log()
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(_record_in_this_process(canonical=sys.argv[3] == "--canonical"), f)
elif __name__ == "__main__":                                # pragma: no cover - re-recording aid
    import dynameta
    lg = record_call_log()
    print("dynameta:", dynameta.__file__)
    print("calls:", len(lg))
    print("digest:", digest(lg))
