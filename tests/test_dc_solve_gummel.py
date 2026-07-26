"""Contract gates for the Gummel (decoupled) DC solve path -- audit C-5 and C-6.

The Gummel route freezes a variable by DELETING its equations and re-adding them afterwards
(DEVSIM's ds.solve is coupled-Newton only), which makes the freeze/thaw sequence a resource the
solver must return even on the failure path. The correctness of the converged fixed point is the
job of validation/gummel_vs_newton.py; these tests only pin the two contracts the audit found
broken:

  C-5  a failed inner Newton must NOT leave the live DEVSIM device missing an equation while
       eq_registry still records it (probed on DEVSIM 2.10.0: pre-fix the device came back with
       PotentialEquation only, and every later solve on it -- including a caller's fall-back to
       method='newton' -- silently solved a different, under-determined problem).
  C-6  the convergence test must not be VACUOUS: with semiconductor_regions=() the snapshot is {}
       and _max_rel_change({}, {}) == 0.0 < gummel_tol, so the solver reported success after one
       outer pass having checked nothing.

Self-skips without devsim, like the rest of the DEVSIM-backed suite.
"""
import warnings

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")

LEN, N_D, MU = 400e-9, 4.0e25, 0.004
_SEQ = [0]


def _build():
    """The smallest proven device: the uniform unipolar n-bar of validation/gummel_vs_newton.py."""
    from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                           setup_semiconductor_region_dd)
    _SEQ[0] += 1
    tag = "g{}".format(_SEQ[0])
    mesh, dev, reg = "dcm_" + tag, "dcd_" + tag, "bar"
    devsim.create_1d_mesh(mesh=mesh)
    devsim.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=4e-9, tag="left")
    devsim.add_1d_mesh_line(mesh=mesh, pos=LEN, ps=4e-9, tag="right")
    devsim.add_1d_contact(mesh=mesh, name="left", tag="left", material="metal")
    devsim.add_1d_contact(mesh=mesh, name="right", tag="right", material="metal")
    devsim.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="left", tag2="right")
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=dev)
    setup_semiconductor_region_dd(dev, reg, n_bg_m3=N_D, eps_static=9.5,
                                  dos_mass_kg=0.35 * 9.1093837015e-31, mobility_m2Vs=MU)
    for c in ("left", "right"):
        setup_contact_ohmic_dd(dev, c)
    devsim.set_node_values(
        device=dev, region=reg, name="Electrons",
        values=[N_D] * len(devsim.get_node_model_values(device=dev, region=reg, name="Electrons")))
    return mesh, dev, reg


def _teardown(dev, mesh):
    from dynameta.carriers import eq_registry as _R
    try:
        _R.clear(dev)
        devsim.delete_device(device=dev)
        devsim.delete_mesh(mesh=mesh)
    except Exception:
        pass


def test_c5_failed_inner_newton_leaves_every_equation_on_the_device():
    """audit C-5, on the live solver. Force the inner Newton to fail (1 iteration at an
    unreachable relative error under a 5 V bias) and require that when the exception surfaces the
    LIVE equation list still matches what eq_registry claims. Pre-fix the device came back holding
    PotentialEquation only, with ElectronContinuityEquation deleted for good."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.dc_solve import solve_dc
    mesh, dev, reg = _build()
    try:
        recorded = set(_R.equation_names(dev))
        assert recorded == {"PotentialEquation", "ElectronContinuityEquation"}
        assert set(devsim.get_equation_list(device=dev, region=reg)) == recorded
        devsim.set_parameter(device=dev, name="right_bias", value=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(Exception):                    # devsim.error: Convergence failure!
                solve_dc(dev, method="gummel", abs_tol=1e-30, rel_tol=1e-30, max_iter=1,
                         gummel_inner_iter=1, semiconductor_regions=[reg])
        live = set(devsim.get_equation_list(device=dev, region=reg))
        assert live == recorded, "equations lost on the failure path: {}".format(recorded - live)
        assert set(_R.equation_names(dev)) == recorded        # registry untouched either way
    finally:
        _teardown(dev, mesh)


def test_c5_freeze_context_manager_rethaws_on_any_exception():
    """The same contract at the unit level, with no solver involved: whatever happens inside the
    block, _frozen must delete on entry and re-add on exit. (The DEVSIM-level test above proves
    the solver actually routes through it.)"""
    from dynameta.carriers import dc_solve
    seen = []

    class _Rec:
        @staticmethod
        def delete_by_name(device, name):
            seen.append(("del", name))

        @staticmethod
        def reapply_by_name(device, name):
            seen.append(("add", name))

    old = dc_solve.R
    dc_solve.R = _Rec
    try:
        with pytest.raises(RuntimeError):
            with dc_solve._frozen("dev", ("EqA", "EqB")):
                raise RuntimeError("inner Newton blew up")
    finally:
        dc_solve.R = old
    assert seen == [("del", "EqA"), ("del", "EqB"), ("add", "EqA"), ("add", "EqB")]


def test_c6_empty_semiconductor_regions_raises_instead_of_succeeding_vacuously():
    """audit C-6: the signature default semiconductor_regions=() made the convergence test
    vacuous -- _max_rel_change({}, {}) == 0.0 < gummel_tol -- so the solve reported SUCCESS after
    one outer pass. It must refuse instead, naming the knob."""
    from dynameta.carriers.dc_solve import solve_dc
    mesh, dev, reg = _build()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="semiconductor_regions"):
                solve_dc(dev, method="gummel")                # the trap: the signature default
    finally:
        _teardown(dev, mesh)


def test_c6_regions_that_carry_none_of_the_tracked_variables_also_raise():
    """The second mode of the same trap: _snapshot swallows the per-variable lookup error, so a
    MISSPELLED region name leaves the convergence test just as vacuous as an empty tuple."""
    from dynameta.carriers.dc_solve import solve_dc
    mesh, dev, reg = _build()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="tracked solution variables"):
                solve_dc(dev, method="gummel", semiconductor_regions=["nosuchregion"])
    finally:
        _teardown(dev, mesh)


def test_gummel_still_converges_on_the_ohmic_bar():
    """The fixes must not disturb the validated path: the same 10 mV unipolar ohmic bar
    validation/gummel_vs_newton.py gates still reaches the Newton fixed point."""
    from dynameta.carriers.dc_solve import solve_dc
    fields = {}
    for method in ("newton", "gummel"):
        mesh, dev, reg = _build()
        try:
            devsim.set_parameter(device=dev, name="right_bias", value=0.01)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solve_dc(dev, method=method, abs_tol=1.0e16, rel_tol=1.0e-7, max_iter=100,
                         semiconductor_regions=[reg])
            fields[method] = np.asarray(
                devsim.get_node_model_values(device=dev, region=reg, name="Potential"))
        finally:
            _teardown(dev, mesh)
    a, b = fields["newton"], fields["gummel"]
    assert float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-6))) < 1e-6
