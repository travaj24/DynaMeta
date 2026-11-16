"""v0.16 carrier wiring: an edge electrode on an EQUILIBRIUM semiconductor region of
LayeredDevsimBuilder is the region's Fermi-level reference (no DEVSIM contact; E_F = -qV through the closure
parameter EF_ref_V), and Electrode(pin_potential=True) restores the v0.15.1 potential-pinned edge contact.

Device: tests/_wiring_fixture.py (the generic gated 6 nm degenerate film between two oxide-insulated metal gates,
40 nm cell, default Mesh2DSpec, x lines every 4 nm), solved with abs_tol = 1e-14 V (a Potential-only device: the
tight target is absolute, LayeredDevsimBuilder.solve docstring).

ORACLES. Without any edge electrode the device is laterally uniform, so every node column must carry the
same profile; and the closure n = N_c F((psi - V - Phi_c)/V_t) makes the solution with the reference at V
and both gates at Vg the solution without a reference at Vg - V, every potential shifted by V (exact
algebra of the closure, physics_equilibrium.setup_semiconductor_region docstring).

In the code before this change the edge electrode was a Potential Dirichlet with E_F = 0 on
the corner nodes of the edge column (the film's two interface nodes, where the accumulation forms): at +5 V the
column profile deviates from the centre column by 0.696 (relative to its maximum) at the contact, 0.162 at 4 nm,
0.0397 at 8 nm (MEASURED on the pin_potential=True restore below, whose build calls are that code's, digests
recorded on that code), the 0.3 V reference case was 0.3 V of band bending at the contact column instead of a shift, the
two-electrode refusal did not exist, and Electrode.pin_potential was ignored.
"""
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")
from dynameta.carriers.devsim_env import import_devsim     # noqa: E402

ds = import_devsim()

from dynameta.carriers import physics_equilibrium as PE                 # noqa: E402
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder        # noqa: E402
from dynameta.sweep import BiasPoint                                     # noqa: E402
from dynameta.warning_types import AdvisoryWarning                       # noqa: E402

import _wiring_fixture as W                                              # noqa: E402

ABS_TOL = 1e-14          # volts


@pytest.fixture(autouse=True)
def _clean():
    yield
    from dynameta.carriers import eq_registry as _R
    for dv in list(ds.get_device_list()):
        _R.clear(dv)
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)


def _solve(design, voltages, tag):
    """Solve, snapshot, tear down (W.Snapshot: DEVSIM solves every live device jointly)."""
    b = LayeredDevsimBuilder(design, mesh_name="m_" + tag, device_name="d_" + tag)
    b.solve(BiasPoint(voltages, tag), abs_tol=ABS_TOL)
    s = W.Snapshot(ds, b)
    b.teardown()
    return s


def _max_column_deviation(cols, x=None):
    """max over z of |n(x) - n(centre)| / max n(centre), for the column nearest x (every column if None)."""
    xs = sorted(cols)
    ref = cols[xs[len(xs) // 2]]
    pick = xs if x is None else [xs[int(np.argmin(np.abs(np.array(xs) - x)))]]
    return max(float(np.max(np.abs(cols[c] - ref)) / np.max(ref)) for c in pick)


def test_edge_ground_is_a_fermi_reference_and_the_profile_is_laterally_uniform():
    """+5 V on the top gate, edge ground at x = 0: every node column equals the centre column to 1e-12
    relative (MEASURED 2.4e-15), and the ground makes no DEVSIM contact. The pin_potential=True restore
    keeps the v0.15.1 distortion (MEASURED 0.696 at the contact column, 0.0397 at 8 nm)."""
    s = _solve(W.film_design(edges=[W.edge()]), {"top": 5.0, "bot": 0.0}, "u")
    cols = s.cols()
    assert len(cols) == 11
    assert _max_column_deviation(cols) < 1e-12
    assert "gnd" not in s.contacts
    assert s.fermi_references == {"gnd": ["film"]} and s.ef_ref == {"film": 0.0}

    sp = _solve(W.film_design(edges=[W.edge(pin=True)]), {"top": 5.0, "bot": 0.0}, "p")
    colp = sp.cols()
    assert "gnd" in sp.contacts and sp.fermi_references == {}
    # MEASURED 0.696 / 0.0397 / +1.29e-3: windows -1.1 % .. +1.7 % and -17 % .. +18 % around them, bar 0.54x
    assert 0.688 < _max_column_deviation(colp, 0.0) < 0.708
    assert 0.033 < _max_column_deviation(colp, 8e-9) < 0.047
    # the pinned columns also bias the CENTRE of the 40 nm cell (MEASURED +1.29e-3 in n at the top face)
    assert abs(sp.top(name="Electrons") / s.top(name="Electrons") - 1.0) > 7e-4


@pytest.mark.parametrize("v_ref", [0.3, -0.3])
def test_reference_at_v_equals_the_unreferenced_device_with_gates_shifted(v_ref):
    """Edge ground at V: the solution equals the device without the ground at gate biases shifted by -V,
    with every potential shifted by +V (MEASURED, 0.3 V: n to 6.7e-16 relative, potentials to 1.7e-16 V)."""
    s1 = _solve(W.film_design(edges=[W.edge(v=v_ref)]), {"top": 5.0, "bot": 0.0}, "r")
    n1, p1 = s1.cols(), s1.cols(name="Potential")
    assert s1.ef_ref == {"film": v_ref}
    s0 = _solve(W.film_design(), {"top": 5.0 - v_ref, "bot": -v_ref}, "z")
    n0, p0 = s0.cols(), s0.cols(name="Potential")
    assert sorted(n0) == sorted(n1)
    assert max(float(np.max(np.abs(n1[x] / n0[x] - 1.0))) for x in n0) < 1e-12
    assert max(float(np.max(np.abs(p1[x] - p0[x] - v_ref))) for x in n0) < 1e-12


def test_two_edge_electrodes_on_one_equilibrium_film_must_agree():
    """Two edge electrodes of one equilibrium region at different voltages raise (no current flows in
    equilibrium physics); at equal voltages both are references of the same region (a y-edge electrode
    too). All-pinned electrodes keep the v0.15.1 behaviour (no new refusal); a pinned and an unpinned one
    at different voltages raise."""
    edges = [W.edge(), W.edge("right", side="x_hi", role="biased"), W.edge("back", side="y_lo", role="biased")]
    b = LayeredDevsimBuilder(W.film_design(edges=edges), mesh_name="m_t", device_name="d_t")
    with pytest.raises(ValueError, match="different voltages"):
        b.solve(BiasPoint({"top": 1.0, "bot": 0.0, "right": 0.1}, "bad"), abs_tol=ABS_TOL)
    assert b.fermi_references() == {"gnd": ["film"], "right": ["film"], "back": ["film"]}
    b.solve(BiasPoint({"top": 1.0, "bot": 0.0, "right": 0.0, "back": 0.0}, "ok"), abs_tol=ABS_TOL)
    for E in ("right", "back"):
        with pytest.raises(ValueError, match="Fermi-level reference"):
            b.set_ssac_gate(E)
    b.teardown()

    pinned = [W.edge(pin=True), W.edge("right", side="x_hi", role="biased", pin=True)]
    bp = LayeredDevsimBuilder(W.film_design(edges=pinned), mesh_name="m_q", device_name="d_q")
    bp.solve(BiasPoint({"top": 0.5, "bot": 0.0, "right": 0.05}, "legacy"), abs_tol=ABS_TOL)
    assert bp.fermi_references() == {}
    bp.teardown()

    mixed = [W.edge(pin=True), W.edge("right", side="x_hi", role="biased")]
    bm = LayeredDevsimBuilder(W.film_design(edges=mixed), mesh_name="m_x", device_name="d_x")
    with pytest.raises(ValueError, match="different voltages"):
        bm.solve(BiasPoint({"top": 0.5, "bot": 0.0, "right": 0.05}, "mixed"), abs_tol=ABS_TOL)


def test_reference_covers_every_connected_equilibrium_region():
    """The film split into two equilibrium layers (one interface between them), the edge electrode on the
    lower one only: both share the reference (one Fermi level for a connected film), and the solution at a
    0.2 V reference equals the unreferenced split film with both gates shifted by -0.2 V."""
    v = 0.2
    s = _solve(W.film_design(split_film=True, edges=[W.edge(layer="film_a", v=v)]), {"top": 3.0, "bot": 0.0},
               "s")
    assert s.fermi_references == {"gnd": ["film_a", "film_b"]} and s.ef_ref == {"film_a": v, "film_b": v}
    got = {r: s.cols(r) for r in ("film_a", "film_b")}
    s0 = _solve(W.film_design(split_film=True), {"top": 3.0 - v, "bot": -v}, "s0")
    for r in ("film_a", "film_b"):
        ref = s0.cols(r)
        assert max(float(np.max(np.abs(got[r][x] / ref[x] - 1.0))) for x in ref) < 1e-12


# ------------------------------------------------------------------------------------------------
# restore paths: the build call log is the one from before the edge reference
# ------------------------------------------------------------------------------------------------
_NAMES = ("node_model", "edge_model", "contact_node_model", "interface_model", "equation",
          "contact_equation", "interface_equation", "node_solution", "edge_from_node_model",
          "create_2d_mesh", "add_2d_mesh_line", "add_2d_region", "add_2d_interface", "add_2d_contact",
          "set_parameter", "set_node_values")
# sha256 of the build_device call log (_NAMES, every keyword repr'd, canonical transcendentals:
# tests/_calllog.recording(True)), (digest, calls), RECORDED with the library before the edge
# reference and the same recorder, on the fixture of tests/_wiring_fixture.py; refmod_pinned is the shared
# validation cell of validation/_reference_device.py, recorded the same way (that library, this recorder and this
# cell; the library's own edge grounds are pinned).
PRE_EDGE_LOGS = {
    "film_pinned": ("67e9b141d024355374eeb259c8a8ca55ebabf12d8100f56df47ef61ae1e9ce15", 336),
    "film_no_edge": ("ac8dda7a9eb032498d4fe321b1171cfe8137a0a2f5aa81fd9145aadfc8678225", 331),
    "refmod_pinned": ("b072c37f5f133166ec424c8754015fdc5b2dc415bef5fadd5525d5fc3648d64b", 350),
}


def _film(pin):
    return W.film_design(x_lines=None, edges=[] if pin is None else [W.edge(pin=pin)],
                         mats=_gated_film_materials())


def _gated_film_materials():
    # the pre-edge-reference logs were recorded on tests/_gated_film_fixture.materials() (no electron affinity)
    import _gated_film_fixture as G
    return G.materials()


def _refmod(pin):
    import dataclasses
    from validation._reference_device import build_reference_modulator
    d = build_reference_modulator()
    d.electrodes = [dataclasses.replace(e, pin_potential=pin) if e.is_edge else e for e in d.electrodes]
    return d


def _build_log(design):
    from _calllog import digest, intercepting, recording
    log = []
    with recording(True):
        with intercepting(ds, _NAMES, log):
            b = LayeredDevsimBuilder(design, mesh_name="ms_mesh", device_name="ms_device")
            b.build_device()
    b.teardown()
    return digest(log), len(log), log


def test_restore_paths_build_the_pre_edge_reference_device():
    """pin_potential=True, and a film without any edge electrode, make exactly the pre-edge-reference build
    calls (same model strings, mesh, contacts, parameters and seeds, bit for bit; digests recorded on the
    library before the edge reference),
    also on the shared validation device of validation/_reference_device.py (two pinned edge grounds; its
    density-dependent DOS mass now warns once, the DOS-mass-at-n_bg guard). That device with its default
    unpinned grounds is not the pre-edge-reference device: no edge contacts, and its film closure carries EF_ref_V."""
    assert _build_log(_film(True))[:2] == PRE_EDGE_LOGS["film_pinned"]
    assert _build_log(_film(None))[:2] == PRE_EDGE_LOGS["film_no_edge"]
    with pytest.warns(AdvisoryWarning, match="density-dependent"):
        assert _build_log(_refmod(True))[:2] == PRE_EDGE_LOGS["refmod_pinned"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*density-dependent.*", category=AdvisoryWarning)
        d_new, n_new, log = _build_log(_refmod(False))
    assert d_new != PRE_EDGE_LOGS["refmod_pinned"][0]
    contacts = [dict(kw)["name"] for nm, kw in log if nm == "add_2d_contact"]
    assert contacts == ["'bot_contact'", "'top_contact'"]
    closures = [dict(kw)["equation"] for nm, kw in log
                if nm == "node_model" and dict(kw).get("name") == "'Electrons'"]
    assert len(closures) == 1 and PE.FERMI_REF_PARAM in closures[0]
