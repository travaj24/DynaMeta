"""Recorder for the defaults-OFF identity gate of the UNIPOLAR in-Newton density-gradient model
(tests/test_bipolar_density_gradient.py), the D2d twin of tests/_bipolar_calllog.py.

It builds a small unipolar DG fixture through the PUBLIC physics_density_gradient API
(setup_dg_quantum_correction, setup_contact_dg, setup_dg_insulator_wall, seed_dg_from_solution with the
wall-aware seed, set_dg_gamma) while intercepting every DEVSIM model-, equation-, parameter- and
solution-defining call, and returns that ordered call log. The gate hashes it and compares to the
recorded digest: equal digests mean the unipolar DG physics DEFINITION is character-for-character the
recorded one (solved cross-version comparison: tests/_sgflux_reference.py). No solve is run (solution
values depend on the BLAS build); the seed reads the setup-time Electrons values.

History: digest recorded on e46bd25 (BEFORE the bipolar DG landed); re-recorded 2026-11-04 for the
two-Bernoulli SG flux (sg_flux); re-recorded 2026-11-06 for v0.15 item 4.4.4 -- the fixture's wall is
now setup_dg_insulator_wall (the retired setup_dg_hard_wall rows are gone; that name is a deprecated
shim onto it, `hard_wall_shim=True` records the shim's log, gated equal) and the seed passes
walls=("wall",).

CANONICAL FORM (2026-11-10). By default the log is recorded with correctly rounded transcendentals
(tests/_calllog.py, tests/_cr_libm.py), so its digest no longer depends on the platform's C library;
canonical=False records the raw log (the platform's libm) the pre-2026-11-10 digests were taken of.

Re-record (only when a sibling workstream changes the unipolar DG physics ON PURPOSE):
    cd <a checkout of the reference code> && python <this file>
(on an export of an older tree, copy this file, tests/_calllog.py and tests/_cr_libm.py into its tests/)
"""
from __future__ import annotations

from _calllog import digest, intercepting, recording, skeleton  # noqa: F401  (re-exported for the gates)

_RECORDED = ("node_model", "edge_model", "contact_node_model", "equation", "contact_equation",
             "set_parameter", "node_solution", "edge_from_node_model", "set_node_values",
             "delete_equation", "delete_contact_equation")


def record_call_log(*, hard_wall_shim: bool = False, statistics=None, canonical: bool = True):
    """The unipolar in-Newton DG definition's call log. `statistics` (v0.15.0): None passes no keyword (this
    tree's DEFAULT: the FD quasi-field transport); 'legacy' reproduces the pre-v0.15.0 averaged-g region.
    canonical (2026-11-10): record with correctly rounded transcendentals (module docstring); False: with the
    platform's libm."""
    with recording(canonical):
        return _record(hard_wall_shim, statistics)


def _record(hard_wall_shim, statistics):
    import devsim as ds
    from dynameta.constants import M_E
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_density_gradient as DG
    from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                           setup_semiconductor_region_dd)

    log = []
    dev, mesh = "dgcl", "dgcl_m"
    with intercepting(ds, _RECORDED, log):
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
                                      dos_mass_kg=mstar, mobility_m2Vs=0.004,
                                      **({} if statistics is None else {"statistics": statistics}))
        setup_contact_ohmic_dd(dev, "back")
        DG.setup_dg_quantum_correction(dev, "bar", m_eff_kg=mstar, gamma=1.0)
        DG.setup_contact_dg(dev, "back", 4.0e26)
        if hard_wall_shim:              # v0.15: the deprecated name (a shim; the CALLER handles its warning)
            DG.setup_dg_hard_wall(dev, "wall", lambda_pin_factor=8.0)
        else:
            DG.setup_dg_insulator_wall(dev, "wall")
        DG.seed_dg_from_solution(dev, "bar", walls=("wall",))
        for fr in (0.25, 1.0):
            DG.set_dg_gamma(dev, "bar", fr)
        log.append(["dg_b_coefficient", repr(DG.dg_b_coefficient(mstar, 0.7))])
        ds.delete_device(device=dev)
        ds.delete_mesh(mesh=mesh)
        _R.clear(dev)
    return log


if __name__ == "__main__":                                  # pragma: no cover - re-recording aid
    import dynameta
    lg = record_call_log()
    print("dynameta:", dynameta.__file__)
    print("calls:", len(lg))
    print("digest:", digest(lg))
