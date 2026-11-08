"""D0 (2026-11-03): the equilibrium semiconductor region's Fermi-integral derivative comes from a
PRECOMPUTED DEVSIM simplify() result instead of a live simplify(diff(...)).

physics_equilibrium.setup_semiconductor_region spent essentially all of its time (MEASURED on devsim
2.11.2: 152 s of the 152 s test_devsim_smoke::test_devsim_equilibrium_smoke; 253 s under load) in ONE
node_model call: simplify(diff(N_c * F_1/2[(Potential - Phi_c0)/V_t], Potential)) -- symdiff's
simplify() of the Aymerich-Humet fit. diff() alone takes milliseconds. The simplified string is a
constant (only parameter NAMES appear in it), so it is computed once and stored in
physics_equilibrium._SIMPLIFIED, keyed by the exact input string.

Gates:
  * the key is exactly the string setup_semiconductor_region builds (a coefficient edit then misses
    the cache and falls back to a live simplify -- correct, only slow -- instead of reusing a stale
    derivative), devsim-free;
  * the cached derivative agrees with DEVSIM's own UNSIMPLIFIED diff() (an independent symbolic
    route) to <= 1e-13 relative over eta in [-40, 100] (fast, needs devsim);
  * @slow: DEVSIM 2.11.2's live simplify(diff(...)) prints EXACTLY the cached string and evaluates
    bit-identically to it on every node of that sweep (the cache is a lossless replay).
"""
import importlib.util

import numpy as np
import pytest

from dynameta.carriers import physics_equilibrium as PE

HAVE_DEVSIM = importlib.util.find_spec("devsim") is not None
_ETA = "(Potential - Phi_c0) / V_t"


def _key():
    return "simplify(diff({}, Potential))".format("N_c * " + PE.F12_aymerich_humet_expr(_ETA))


def test_the_cache_key_is_what_setup_semiconductor_region_builds():
    assert _key() in PE._SIMPLIFIED, "F_1/2 fit or eta string changed: regenerate the cached simplify()"
    assert PE.simplified(_key()) == PE._SIMPLIFIED[_key()]
    assert PE.simplified("simplify(diff(x*x, x))") == "simplify(diff(x*x, x))"   # miss -> passthrough


def _sweep_region(ds, tag):
    """1-D region with Potential spanning eta = -40 .. 100 (log-degenerate to strongly degenerate)."""
    dev, mesh = "f12c_d_" + tag, "f12c_m_" + tag
    nn = 2001
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="a")
    ds.add_1d_mesh_line(mesh=mesh, pos=(nn - 1) * 1e-9, ps=1e-9, tag="b")
    ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=mesh, material="x", region="r", tag1="a", tag2="b")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    phi_c0 = -0.1234567
    for k, v in (("N_c", 2.8e25), ("Phi_c0", phi_c0), ("V_t", PE.V_T)):
        ds.set_parameter(device=dev, region="r", name=k, value=v)
    ds.node_solution(device=dev, region="r", name="Potential")
    n = len(ds.get_node_model_values(device=dev, region="r", name="x"))
    ds.set_node_values(device=dev, region="r", name="Potential",
                       values=list(phi_c0 + np.linspace(-40.0, 100.0, n) * PE.V_T))
    return dev, mesh


def _values(ds, dev, name, equation):
    ds.node_model(device=dev, region="r", name=name, equation=equation)
    return np.asarray(ds.get_node_model_values(device=dev, region="r", name=name), dtype=float)


@pytest.mark.skipif(not HAVE_DEVSIM, reason="needs devsim")
def test_cached_derivative_matches_the_unsimplified_symbolic_derivative():
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    dev, mesh = _sweep_region(ds, "fast")
    try:
        cached = _values(ds, dev, "dn_cached", PE.simplified(_key()))
        exact = _values(ds, dev, "dn_diff", _key()[len("simplify("):-1])       # diff(...) only
        assert np.all(np.isfinite(cached)) and np.all(cached > 0.0)
        rel = np.max(np.abs(cached - exact) / np.abs(exact))
        assert rel <= 1e-13, rel
    finally:
        ds.delete_device(device=dev)
        ds.delete_mesh(mesh=mesh)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DEVSIM, reason="needs devsim")
def test_cached_derivative_is_devsim_simplify_output_bit_for_bit():
    """~2.5-4 min: one live simplify() of the F_1/2 derivative."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    dev, mesh = _sweep_region(ds, "slow")
    try:
        live_str = ds.node_model(device=dev, region="r", name="dn_live", equation=_key())
        assert live_str == PE._SIMPLIFIED[_key()]
        live = np.asarray(ds.get_node_model_values(device=dev, region="r", name="dn_live"), dtype=float)
        cached = _values(ds, dev, "dn_cached", PE.simplified(_key()))
        assert np.array_equal(live.view(np.int64), cached.view(np.int64))
    finally:
        ds.delete_device(device=dev)
        ds.delete_mesh(mesh=mesh)
