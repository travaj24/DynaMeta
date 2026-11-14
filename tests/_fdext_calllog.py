"""Recorder for the defaults-off identity gate of the v0.15 Fermi-Dirac-mode completion
(tests/test_fd_mode_complete.py; docs/audit/2026-11-06-v0.15-fermi-dirac-mode-complete.md).

It builds, through the PUBLIC bipolar API and with the LEGACY statistics (statistics unset), every path the
FD-mode work touches -- so it runs unchanged on the pre-change tree (2fb4b76):
  * 1-D pn region, fd_enhancement=True, bandgap narrowing (the n_ie node path), the built-in SRH channel at
    a shifted level, charged bulk levels (an intrinsic-referenced acceptor, a conduction-referenced donor,
    a valence-referenced neutral), a TUPLE of BTBT sets (edge assembly with n_ie@n0/@n1), and an SiO2 region
    behind interface traps (split charge, recombination, a fixed charge);
  * 2-D tensor-mesh slab, fd_enhancement=True, bandgap narrowing, one BTBT set (element assembly with
    n_ie@en<k>).
It intercepts every DEVSIM model-, equation-, parameter- and solution-defining call and returns the ordered
log. The gate hashes it and compares to the digest recorded on 2fb4b76: equal digests mean the legacy
definition is character-for-character unchanged. No solve, no circuit.

CANONICAL FORM (2026-11-08). By default the log is recorded inside tests/_cr_libm.correctly_rounded(): every
math / numpy transcendental the physics calls returns its correctly rounded value, so the log -- and its
digest -- no longer depends on which C library rounded a hard case (the Windows UCRT and glibc differ in
the last digit of one interface-trap n1: docs/audit/2026-11-08-v0.15-linux-ci-exactness.md). Everything
else is still recorded bit for bit. canonical=False records with the platform's own libm (the raw log
the pre-2026-11-08 digest was taken of). The interceptor, the canonical context, digest() and skeleton() are
the shared ones of tests/_calllog.py (2026-11-10).

Re-record (only on an intended change of the legacy definition):
    cd <a checkout of the reference code> && PYTHONPATH=. python <this file>
"""
from __future__ import annotations

import json
import sys

from _calllog import digest, intercepting, recording, skeleton  # noqa: F401  (re-exported for the gates)

_RECORDED = ("node_model", "edge_model", "element_model", "contact_node_model", "equation",
             "contact_equation", "set_parameter", "node_solution", "edge_from_node_model",
             "element_from_edge_model", "element_from_node_model", "set_node_values", "interface_model",
             "interface_equation")


def _tensor_2d(ds, mesh, n=3, L=6e-9):
    import numpy as np
    g = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel(), 0.0 * X.ravel()])
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    el = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            el += [2, 0, int(a), int(b), int(c), 2, 0, int(a), int(c), int(d)]
    ds.create_gmsh_mesh(mesh=mesh, coordinates=[float(c) for c in coords.ravel()], physical_names=["r"],
                        elements=el)
    ds.add_gmsh_region(mesh=mesh, gmsh_name="r", region="r", material="Si")


def record_call_log(tag="fdcl", canonical=True, **region_kw):
    """The ordered call log of the two legacy fixtures (devices <tag>_1d / <tag>_2d, deleted afterwards).
    `region_kw` are extra setup_bipolar_region keywords for BOTH fixtures (the gate passes
    statistics='legacy'). canonical=True (default): recorded with correctly rounded transcendentals (module
    docstring); False: with the platform's libm."""
    with recording(canonical):
        return _record(tag, **region_kw)


def _record(tag, **region_kw):
    import warnings
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_traps as PT
    from dynameta.carriers.bipolar_options import BGN_SI_KLAASSEN_1992, BipolarOptions
    from dynameta.carriers.tunneling import GERMANIUM_KAO, SILICON_HURKX
    ds = import_devsim()
    bgn = BipolarOptions(bgn=BGN_SI_KLAASSEN_1992)
    traps = (PT.TrapLevel(energy_eV=0.12, density_m3=3e21, kind="acceptor", tau_n0_s=1e-8, tau_p0_s=3e-8),
             PT.TrapLevel(energy_eV=0.2, density_m3=2e21, kind="donor", tau_n0_s=2e-8, tau_p0_s=1e-8,
                          energy_reference="conduction"),
             PT.TrapLevel(energy_eV=0.3, kind="neutral", tau_n0_s=1e-8, tau_p0_s=1e-8,
                          energy_reference="valence"))
    its = PT.InterfaceTraps(dit_eV_m2=2e15, band_gap_eV=1.12, fixed_charge_C_m2=1e-4, sigma_n_m2=2e-20)
    log = []
    devs = []
    try:
        with intercepting(ds, _RECORDED, log), warnings.catch_warnings():
            warnings.simplefilter("ignore")            # the BGN basis / FD-overlap advisories (not logged)
            for dim in (1, 2):
                dev = "{}_{}d".format(tag, dim)
                mesh = dev + "_m"
                devs.append((dev, mesh))
                _R.clear(dev)
                if dim == 1:
                    ds.create_1d_mesh(mesh=mesh)
                    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="a")
                    ds.add_1d_mesh_line(mesh=mesh, pos=8e-9, ps=1e-9, tag="i")
                    ds.add_1d_mesh_line(mesh=mesh, pos=10e-9, ps=1e-9, tag="b")
                    ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
                    ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
                    ds.add_1d_region(mesh=mesh, material="Si", region="r", tag1="a", tag2="i")
                    ds.add_1d_region(mesh=mesh, material="SiO2", region="ox", tag1="i", tag2="b")
                    ds.add_1d_interface(mesh=mesh, name="si_ox", tag="i")
                else:
                    _tensor_2d(ds, mesh)
                ds.finalize_mesh(mesh=mesh)
                ds.create_device(mesh=mesh, device=dev)
                ds.node_model(device=dev, region="r", name="NetDoping",
                              equation="ifelse(x < 4e-9, -2e25, 4e25)" if dim == 1 else "3e25")
                kw = dict(region_kw)
                if dim == 1:
                    kw.update(traps=traps, btbt=GERMANIUM_KAO, srh_trap_energy_eV=0.07)
                else:
                    kw.update(btbt=SILICON_HURKX)
                BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=2.8e25, n_dos_p_m3=1.04e25,
                                        n_i_m3=1e16, mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048,
                                        tau_n_s=1e-7, tau_p_s=2e-7, fd_enhancement=True, options=bgn, **kw)
                if dim == 1:
                    BP.setup_equilibrium_seed_models(dev, "r")
                    BP.setup_contact_ohmic_bipolar(dev, "a")
                    PE.setup_dielectric_region(dev, "ox", 3.9)
                    PE.setup_interface(dev, "si_ox")
                    PT.install_interface_traps(dev, "si_ox", its)
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
