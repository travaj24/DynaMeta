"""AUDIT V-8: the polarization-vocabulary map is EXECUTABLE, not prose.

The repo speaks five spellings of "which polarization" across ~18 modules (`{'s','p'}`,
OpticalSpec's `{'x','y','p'}`, the grating bridge's `{'te','tm'}`, the differentiable forwards'
integer `row`, and hydro_fem's 2-D in-plane `pol_axis`). `dynameta/core/polarization.py` documents
which is which and converts between them; THIS file is what stops that document from rotting.

Three gates:

1. THE MAP IS EXECUTABLE. Every entry point listed in `VOCABULARIES` is driven twice through the
   SAME probe: once with its own vocabulary (must NOT be rejected by the vocabulary guard) and
   once with a value that is valid in a SIBLING vocabulary (must raise, and the message must name
   the vocabulary this entry point actually speaks AND point at the map). The negative leg is what
   proves the guard is REACHED with those arguments, so the positive leg is meaningful: the probes
   deliberately pass the cheapest arguments that get past the signature, and only polarization
   REJECTION is asserted, never physics.

2. BYTE-IDENTITY OF THE VALID PATH. Six representative valid calls, one per family (two for the
   two biggest), pinned to goldens captured BEFORE the V-8 edits. V-8 was validation + doc only;
   a valid call that moves is a bug in the guard, not a new answer. (The full 35-call bit-exact
   before/after sweep lives in the audit record; these six are the shipped regression.)

3. THE NORMAL-INCIDENCE x/y <-> s/p CORRESPONDENCE IS NUMERIC. One stack, solved through an s/p
   API (tmm_reference) and an x/y API (the lumenairy Berreman bridge, via OpticalSpec ->
   _common.pol_row -> lab Jones rows), must agree to 1e-12 in R -- and must DISAGREE at oblique
   incidence, which is why the correspondence is documented as normal-incidence-only.

Run: python -m pytest tests/test_polarization_vocabulary.py -q
"""
import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from dynameta.core.polarization import (VOCABULARIES, PolarizationConversionError,
                                        PolarizationVocabularyError, all_entry_points,
                                        normalize_pol)

C = 299792458.0
LAM = 1300e-9
OMEGA = 2.0 * math.pi * C / LAM


# Third-party packages a probe may need. A missing one is a SKIP (the repo's `_OPTIONAL`
# convention, see tests/test_api_surface.py): the fast CI legs install neither ngsolve nor devsim,
# so the three FEM-backed entry points are exercised on the solvers leg -- which is why this file
# is listed there in .github/workflows/ci.yml.
_OPTIONAL = {"ngsolve", "netgen", "devsim", "gmsh", "tmm", "numba", "jax", "jaxlib", "h5py",
             "matplotlib", "lumenairy"}


def _m(name):
    """Import a module LAZILY. Probes must not import anything at COLLECTION time -- the map has
    to be walkable on a bare install, and only the probes that need a solver may pay for one."""
    return importlib.import_module(name)


def _resolve(ep):
    """The live object an EntryPoint names (``qualname`` may be dotted, e.g. ``Class.method``)."""
    obj = importlib.import_module(ep.module)
    for part in ep.qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _call(ep, value):
    """Drive one probe, turning a missing OPTIONAL extra into a skip rather than a failure."""
    try:
        return PROBES[ep.dotted](value)
    except ImportError as exc:
        pkg = (getattr(exc, "name", "") or "").split(".")[0]
        if pkg in _OPTIONAL:
            pytest.skip("optional extra {!r} not installed".format(pkg))
        raise


# ------------------------------------------------------------------------------------------------
# The probes.  One per map entry point; `probe(value)` calls it with that polarization label.
#
# CONTRACT: a probe must reach the polarization guard.  Everything else may be junk -- the
# assertions below only ever look at whether a PolarizationVocabularyError is raised, and the
# NEGATIVE leg (same probe, wrong label) is what proves the guard is reached at all.  Probes that
# can run a real physical solve cheaply do so; the expensive ones (FDTD marches, FEM assembles)
# pass None for the physical arguments and die downstream, which is fine and deliberate.
# ------------------------------------------------------------------------------------------------
def _probes():
    def _stack():
        L = _m("dynameta.core.layered")
        return L.LayeredStack(n_super=1.0 + 0j, n_sub=1.5 + 0j,
                              slabs=[L.LayeredSlab(eps=4.0 + 0j, thickness_m=250e-9)])

    def _nl_layers():
        return [_m("dynameta.optics.nonlocal_tmm").DielectricLayer(4.0 + 0j, 100e-9)]

    r_layers = [(4.0 + 0.1j, 200e-9)]
    hs = dict(n_substrate=1.5 + 0j, n_superstrate=1.0 + 0j)

    def _fake_optical(v):
        return SimpleNamespace(polarization=v, incidence_angle_deg=0.0, azimuth_deg=0.0,
                               incidence_side="top")

    def _sym_design(v):
        d = SimpleNamespace(optical=_fake_optical(v))
        d.device_symmetry = lambda: "c4v"
        return d

    P = {}
    # ---- {'s','p'} -----------------------------------------------------------------------------
    P["dynameta.optics.tmm_reference.stack_rta"] = \
        lambda v: _m("dynameta.optics.tmm_reference").stack_rta(
            1.0, [(2.0 + 0j, 250e-9)], 1.5, LAM, pol=v)
    P["dynameta.optics.tmm_reference.layered_rta"] = \
        lambda v: _m("dynameta.optics.tmm_reference").layered_rta(_stack(), LAM, pol=v)
    P["dynameta.optics.tmm_reference.layered_per_layer_absorption"] = \
        lambda v: _m("dynameta.optics.tmm_reference").layered_per_layer_absorption(
            _stack(), LAM, pol=v)
    P["dynameta.optics.resonance.layered_smatrix_complex"] = \
        lambda v: _m("dynameta.optics.resonance").layered_smatrix_complex(
            OMEGA, r_layers, pol=v)
    P["dynameta.optics.resonance.smatrix_pole_func"] = \
        lambda v: _m("dynameta.optics.resonance").smatrix_pole_func(r_layers, pol=v)
    P["dynameta.optics.resonance._admittance"] = \
        lambda v: _m("dynameta.optics.resonance")._admittance(4.0 + 0j, 1.2e7 + 0j, v)
    P["dynameta.optics.nonlocal_tmm.stack_rt"] = \
        lambda v: _m("dynameta.optics.nonlocal_tmm").stack_rt(
            OMEGA, _nl_layers(), pol=v)
    P["dynameta.optics.nonlocal_tmm.pole_function"] = \
        lambda v: _m("dynameta.optics.nonlocal_tmm").pole_function(
            _nl_layers(), pol=v)
    P["dynameta.optics.nonlocal_tmm._admittance"] = \
        lambda v: _m("dynameta.optics.nonlocal_tmm")._admittance(4.0 + 0j, 1.2e7 + 0j, v)
    P["dynameta.optics.shg_fem.rudnick_stern_flat_shg"] = \
        lambda v: _m("dynameta.optics.shg_fem").rudnick_stern_flat_shg(
            800e-9, 45.0, -20 + 1j, -5 + 0.5j, 1e-20, polarization=v)
    P["dynameta.optics.shg_fem.rudnick_stern_flat_sfg"] = \
        lambda v: _m("dynameta.optics.shg_fem").rudnick_stern_flat_sfg(
            2.4e15, 2.1e15, 40.0, 50.0, -20 + 1j, -18 + 1.1j, -5 + 0.5j, 1e-20,
            polarization=v)
    P["dynameta.optics.fdtd_nd.solve2d.solve_fdtd_2d_oblique"] = \
        lambda v: _m("dynameta.optics.fdtd_nd.solve2d").solve_fdtd_2d_oblique(
            None, period_x_m=4e-7, angle_deg=20.0, lambda_min_m=1e-6, lambda_max_m=1.6e-6,
            pol=v)
    P["dynameta.optics.fdtd_nd.oblique2d._run_oblique"] = \
        lambda v: _m("dynameta.optics.fdtd_nd.oblique2d")._run_oblique(
            "numpy", None, None, None, None, None, None, None, None, None, None, None,
            None, None, pol=v)

    # ---- {'x','y','p'} -------------------------------------------------------------------------
    P["dynameta.geometry.specs.OpticalSpec"] = \
        lambda v: _m("dynameta.geometry.specs").OpticalSpec(polarization=v)
    P["dynameta.optics.solver.background_probe_pol"] = \
        lambda v: _m("dynameta.optics.solver").background_probe_pol(v, 0.3, 0.2)
    P["dynameta.optics.solver._layered_background"] = \
        lambda v: _m("dynameta.optics.solver")._layered_background(
            v, None, None, None, None, None, None, None, None, None, None, None)
    P["dynameta.optics.ngsolve_layered.LayeredOpticalBuilder._check_symmetry_supported"] = \
        lambda v: getattr(_m("dynameta.optics.ngsolve_layered").LayeredOpticalBuilder,
                          "_check_symmetry_supported")(None, _sym_design(v), "quarter")
    P["dynameta.optics.fdtd_seam._guard_optical_spec"] = \
        lambda v: _m("dynameta.optics.fdtd_seam")._guard_optical_spec(
            SimpleNamespace(optical=_fake_optical(v)), structured=False)
    P["dynameta.optics.tmm_reference._pol_for"] = \
        lambda v: _m("dynameta.optics.tmm_reference")._pol_for(_fake_optical(v))
    P["dynameta.optics.lumenairy_bridge._common.pol_row"] = \
        lambda v: _m("dynameta.optics.lumenairy_bridge._common").pol_row(_fake_optical(v))
    P["dynameta.optics.lumenairy_bridge._common.p_basis_conversion"] = \
        lambda v: _m("dynameta.optics.lumenairy_bridge._common").p_basis_conversion(
            v, 0.4, 1.0 + 0j, 1.5 + 0j)
    P["dynameta.optics.lumenairy_bridge._common.pol_tangential_unit"] = \
        lambda v: _m("dynameta.optics.lumenairy_bridge._common").pol_tangential_unit(
            v, 0.3)
    P["dynameta.optics.fdtd_mo.solve_fdtd_mo_1d"] = \
        lambda v: _m("dynameta.optics.fdtd_mo").solve_fdtd_mo_1d(
            None, lambda_min_m=1e-6, lambda_max_m=1.6e-6, pol=v)
    P["dynameta.optics.fdtd_nd.solve3d.solve_fdtd_3d_mo"] = \
        lambda v: _m("dynameta.optics.fdtd_nd.solve3d").solve_fdtd_3d_mo(
            None, period_x_m=4e-7, period_y_m=4e-7, lambda_min_m=1e-6,
            lambda_max_m=1.6e-6, pol=v)
    P["dynameta.optics.fdtd_nd.kernels3d._run_3d_mo"] = \
        lambda v: _m("dynameta.optics.fdtd_nd.kernels3d")._run_3d_mo(
            None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, v)

    # ---- {'te','tm'} ---------------------------------------------------------------------------
    P["dynameta.optics.lumenairy_bridge.rcwa_design.rcwa_grating_RT"] = \
        lambda v: _m("dynameta.optics.lumenairy_bridge.rcwa_design").rcwa_grating_RT(
            600e-9, 4.0 + 0j, 1.0 + 0j, 250e-9, 0.5, 1.31e-6, polarization=v,
            n_orders=3, **hs)

    # ---- integer lab `row` ---------------------------------------------------------------------
    P["dynameta.optics.lumenairy_bridge.berreman_design.berreman_RT"] = \
        lambda v: _m("dynameta.optics.lumenairy_bridge.berreman_design").berreman_RT(
            [(np.diag([4.0 + 0j, 4.4 + 0j, 4.0 + 0j]), 250e-9)], 1.31e-6, row=v,
            **hs)
    P["dynameta.optics.lumenairy_bridge.rcwa_design.rcwa_stack_RT"] = \
        lambda v: _m("dynameta.optics.lumenairy_bridge.rcwa_design").rcwa_stack_RT(
            [(4.0 + 0j, 250e-9)], 1.31e-6, period_x=600e-9, n_orders=3, row=v, **hs)
    P["dynameta.optics.lumenairy_bridge.rcwa_design.pmm_stack_RT"] = \
        lambda v: _m("dynameta.optics.lumenairy_bridge.rcwa_design").pmm_stack_RT(
            [([(0.5, 4.0 + 0j), (0.5, 1.0 + 0j)], 250e-9)], 1.31e-6, period=600e-9,
            n_orders=3, row=v, **hs)
    P["dynameta.optics.lumenairy_bridge.berreman_backend.berreman_result_to_optical_result"] = \
        lambda v: getattr(_m("dynameta.optics.lumenairy_bridge.berreman_backend"),
                          "berreman_result_to_optical_result")(None, None, None, None, v,
                                                               t0=0.0)
    P["dynameta.optics.lumenairy_bridge.rcwa_backend.rcwa_result_to_optical_result"] = \
        lambda v: _m("dynameta.optics.lumenairy_bridge.rcwa_backend").rcwa_result_to_optical_result(
            None, v, t0=0.0)

    # ---- `pol_axis` ----------------------------------------------------------------------------
    P["dynameta.optics.hydro_fem.gap_enhancement_2d"] = \
        lambda v: _m("dynameta.optics.hydro_fem").gap_enhancement_2d(
            None, OMEGA, -12 + 1j, 1.0 + 0j, pol_axis=v)
    P["dynameta.optics.hydro_fem.scattering_2d"] = \
        lambda v: _m("dynameta.optics.hydro_fem").scattering_2d(
            None, OMEGA, None, pol_axis=v)
    return P


PROBES = _probes()
_DRIVEN = [(v, ep) for v, ep in all_entry_points() if ep.delegates_to is None]


# ------------------------------------------------------------------------------------------------
# Gate 1 -- the map is executable
# ------------------------------------------------------------------------------------------------
def test_every_map_entry_point_resolves():
    """A map entry that names a function the repo no longer has is a rotted map."""
    missing, skipped = [], []
    for _v, ep in all_entry_points():
        try:
            _resolve(ep)
        except ImportError as exc:
            pkg = (getattr(exc, "name", "") or "").split(".")[0]
            (skipped if pkg in _OPTIONAL else missing).append("{}: {}".format(ep.dotted, exc))
        except AttributeError as exc:
            missing.append("{}: {}".format(ep.dotted, exc))
    assert not missing, missing
    assert len(skipped) <= 3, skipped        # only the ngsolve-importing modules may skip


def test_every_map_entry_point_has_a_probe_and_vice_versa():
    """The map and the probe table must stay in lockstep -- adding an entry point without a
    driver (or leaving a stale driver behind) is exactly how the previous vocabulary census went
    out of date."""
    mapped = {ep.dotted for _v, ep in _DRIVEN}
    assert mapped - set(PROBES) == set(), "map entries with no probe: {}".format(
        sorted(mapped - set(PROBES)))
    assert set(PROBES) - mapped == set(), "probes for nothing in the map: {}".format(
        sorted(set(PROBES) - mapped))


def test_delegating_entry_points_name_a_target_that_is_itself_in_the_map():
    """The two sites that take a label and forward it unvalidated are marked as such, and their
    validator has to be a mapped entry point -- otherwise "it delegates" is an unchecked claim."""
    mapped = {ep.dotted for _v, ep in all_entry_points()}
    for _v, ep in all_entry_points():
        if ep.delegates_to is not None:
            assert ep.delegates_to in mapped, "{} delegates to unmapped {}".format(
                ep.dotted, ep.delegates_to)


@pytest.mark.parametrize("vocab,ep", _DRIVEN, ids=[ep.dotted for _v, ep in _DRIVEN])
def test_entry_point_accepts_its_own_vocabulary(vocab, ep):
    """Every value the entry point is documented to accept must get past its vocabulary guard.
    (Downstream failures from the deliberately-minimal probe arguments are not our business --
    only a PolarizationVocabularyError would be.)"""
    voc = VOCABULARIES[vocab]
    for good in ep.accepted_values(voc):
        try:
            _call(ep, good)
        except PolarizationVocabularyError as exc:
            pytest.fail("{} rejected its OWN vocabulary value {!r}: {}".format(
                ep.dotted, good, exc))
        except Exception:                             # noqa: BLE001 -- physics is out of scope here
            pass


@pytest.mark.parametrize("vocab,ep", _DRIVEN, ids=[ep.dotted for _v, ep in _DRIVEN])
def test_entry_point_rejects_a_sibling_vocabulary_and_names_its_own(vocab, ep):
    """The V-8 deliverable: a label from a SIBLING family must raise (not fall through to an
    `else` branch), and the message must name the vocabulary this entry point speaks and point at
    the shared map. Raising here also proves the probe REACHES the guard, which is what makes the
    positive leg above meaningful."""
    voc = VOCABULARIES[vocab]
    wrong = voc.wrong_value()
    assert wrong not in ep.accepted_values(voc), "probe value is not actually wrong here"
    with pytest.raises(PolarizationVocabularyError) as excinfo:
        _call(ep, wrong)
    msg = str(excinfo.value)
    assert "dynameta.core.polarization" in msg, msg
    assert repr(voc.name) in msg or all(repr(v) in msg for v in voc.values), msg
    assert ep.parameter in msg, msg


def test_subset_entry_points_refuse_the_unimplemented_member_by_name():
    """The three magneto-optic engines speak the lab vocabulary but implement only its 'x'/'y'
    subset (normal incidence -- no plane of incidence, so no distinct p-pol mode). 'p' used to
    take the 'x' branch SILENTLY; it must now say which subset it implements, and say so
    differently from a genuine wrong-vocabulary label."""
    subset = [(v, ep) for v, ep in _DRIVEN if ep.values is not None]
    assert len(subset) == 3, [ep.dotted for _v, ep in subset]
    for vocab, ep in subset:
        unimplemented = [x for x in VOCABULARIES[vocab].values if x not in ep.values]
        for bad in unimplemented:
            with pytest.raises(PolarizationVocabularyError) as excinfo:
                _call(ep, bad)
            msg = str(excinfo.value)
            assert "SUBSET" in msg, msg
            assert "dynameta.core.polarization" in msg, msg


def test_the_map_covers_the_modules_the_audit_counted():
    """V-8's census was "~18 modules, four vocabularies (plus a fifth)". Pin that the map did not
    quietly shrink to the easy half."""
    modules = {ep.module for _v, ep in all_entry_points()}
    assert len(modules) >= 18, sorted(modules)
    assert len(VOCABULARIES) == 5, sorted(VOCABULARIES)
    # every vocabulary is actually spoken somewhere
    for name, voc in VOCABULARIES.items():
        assert voc.entry_points, name


# ------------------------------------------------------------------------------------------------
# Gate 2 -- byte-identity of the valid path (goldens captured BEFORE the V-8 edits)
# ------------------------------------------------------------------------------------------------
def test_six_representative_valid_calls_are_unchanged():
    """Six valid calls, one per family (the two biggest get two), against goldens captured from
    the PRE-V-8 tree. V-8 added guards and docs only, so every one of these must still be the
    same number. Compared at 1e-13 relative rather than `==` because two of the engines route
    through BLAS/LAPACK, where bit-equality is a platform claim, not a correctness one -- the
    bit-exact 35-call before/after sweep was run out of band and is recorded in the audit ledger.
    """
    import dynameta.optics.nonlocal_tmm as nt
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge._common import pol_row
    from dynameta.optics.resonance import layered_smatrix_complex
    from dynameta.optics.shg_fem import rudnick_stern_flat_shg
    from dynameta.optics.tmm_reference import stack_rta

    pytest.importorskip("tmm")
    rel = 1e-13
    layers = [(2.0 + 0j, 250e-9), (1.6 + 0.02j, 90e-9)]

    # (1) {'s','p'} -- tmm_reference at oblique incidence
    R, T, A = stack_rta(1.0, layers, 1.5, LAM, theta_deg=20.0, pol="p")
    assert R == pytest.approx(0.09263059330420648, rel=rel)
    assert T == pytest.approx(0.8907963770833686, rel=rel)
    assert A == pytest.approx(0.016573029612425016, rel=rel)

    # (2) {'s','p'} -- the nonlocal/hydrodynamic twin (the V-3 _admittance pair)
    res = nt.stack_rt(OMEGA, [nt.DielectricLayer(4.0 + 0.1j, 100e-9)], pol="s", n_sub=1.5,
                      theta_rad=0.4)
    assert res.R == pytest.approx(0.18095391367699734, rel=rel)
    assert res.T == pytest.approx(0.7954599176077697, rel=rel)
    assert res.r == pytest.approx(-0.4111491838484168 + 0.10913414817451986j, rel=rel)

    # (3) {'s','p'} -- the complex-omega S-matrix
    sm = layered_smatrix_complex(OMEGA, [(4.0 + 0.1j, 200e-9)], pol="p", theta_rad=0.4, n_sub=1.5)
    assert sm.R == pytest.approx(0.1612626538609237, rel=rel)
    assert sm.r == pytest.approx(-0.39439740946364943 - 0.07558662096751108j, rel=rel)

    # (4) {'s','p'} -- the surface-SHG closed form (the silent p-pol fallthrough site)
    shg = rudnick_stern_flat_shg(800e-9, 45.0, -20 + 1j, -5 + 0.5j, 1e-20, polarization="p")
    assert shg["S_up"] == pytest.approx(3.049357811774525e-34, rel=rel)
    assert shg["E_perp_in"] == pytest.approx(-0.06255598764049232 - 0.023343669336714817j,
                                             rel=rel)
    assert rudnick_stern_flat_shg(800e-9, 45.0, -20 + 1j, -5 + 0.5j, 1e-20,
                                  polarization="s")["S_up"] == 0.0

    # (5) {'x','y','p'} -- the lab-axis -> row bridge (_POL_ROW, non-injective on row 0)
    assert [pol_row(OpticalSpec(polarization=p)) for p in ("x", "y", "p")] == [0, 1, 0]

    # (6) {'te','tm'} and integer `row` -- the two lumenairy-bridge families
    lum = pytest.importorskip("lumenairy")
    del lum
    from dynameta.optics.lumenairy_bridge.berreman_design import berreman_RT
    from dynameta.optics.lumenairy_bridge.rcwa_design import rcwa_grating_RT

    Rg, Tg = rcwa_grating_RT(600e-9, 4.0 + 0j, 1.0 + 0j, 250e-9, 0.5, 1.31e-6,
                             n_substrate=1.5 + 0j, n_superstrate=1.0 + 0j,
                             polarization="te", n_orders=7)
    assert float(Rg) == pytest.approx(0.10697476345509854, rel=1e-11)
    assert float(Tg) == pytest.approx(0.8930252365449003, rel=1e-11)

    Rb, Tb = berreman_RT([(np.diag([4.0 + 0j, 4.4 + 0j, 4.0 + 0j]), 250e-9)], 1.31e-6,
                         n_substrate=1.5 + 0j, n_superstrate=1.0 + 0j, angle=0.2, row=0)
    assert float(Rb) == pytest.approx(0.12139127031149236, rel=1e-11)
    assert float(Tb) == pytest.approx(0.8786087296885077, rel=1e-11)


# ------------------------------------------------------------------------------------------------
# Gate 3 -- the normal-incidence x/y <-> s/p correspondence, numerically
# ------------------------------------------------------------------------------------------------
def _lab_R(pol, theta_deg):
    """R of the reference stack through an x/y (OpticalSpec) API: the lumenairy BERREMAN bridge.
    The path is OpticalSpec.polarization -> _common.pol_row -> the lab-basis Jones ROW of
    berreman_jones_1d -- i.e. two of the five vocabularies (lab_xyp and row) end to end, and a
    completely different engine from the s/p side."""
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.lumenairy_bridge.berreman_backend import BerremanLayeredSolver

    stk = LayeredStack(n_super=1.0 + 0j, n_sub=1.5 + 0j,
                       slabs=[LayeredSlab(eps=4.0 + 0j, thickness_m=250e-9),
                              LayeredSlab(eps=2.56 + 0j, thickness_m=90e-9)])
    return BerremanLayeredSolver().solve(
        stk, LAM, OpticalSpec(polarization=pol, incidence_angle_deg=theta_deg)).R


def _sp_R(pol, theta_deg):
    """R of the SAME stack through an s/p API (tmm_reference / Byrnes-tmm)."""
    from dynameta.optics.tmm_reference import stack_rta
    return stack_rta(1.0, [(2.0 + 0j, 250e-9), (1.6 + 0j, 90e-9)], 1.5, LAM,
                     theta_deg=theta_deg, pol=pol)[0]


def test_normal_incidence_xy_and_sp_agree_to_1e_12():
    """THE documented correspondence: at theta = 0 the plane of incidence is undefined, so the
    stack is polarization-DEGENERATE and every spelling gives the same R.

    MAPPING UNDER TEST (dynameta.core.polarization): 'y' <-> 's' and 'p' <-> 'p' at azimuth 0,
    with 'x' joining them only at normal incidence. All four labels are solved here, through two
    independent engines (Byrnes-tmm on the s/p side, lumenairy's Berreman 4x4 on the lab side),
    and the spread across ALL of them must be <= 1e-12."""
    pytest.importorskip("lumenairy")
    pytest.importorskip("tmm")
    vals = {"s": _sp_R("s", 0.0), "p": _sp_R("p", 0.0),
            "x": _lab_R("x", 0.0), "y": _lab_R("y", 0.0)}
    spread = max(vals.values()) - min(vals.values())
    assert spread <= 1e-12, vals
    # and the map's own converter agrees with what was just solved
    assert normalize_pol("y", "lab_xyp", to="sp", azimuth_deg=0.0, theta_deg=0.0) == "s"
    assert normalize_pol("x", "lab_xyp", to="sp", azimuth_deg=0.0, theta_deg=0.0) == "s"
    assert normalize_pol("p", "lab_xyp", to="sp", azimuth_deg=0.0, theta_deg=0.0) == "p"


def test_the_correspondence_is_normal_incidence_only_discrimination():
    """Discrimination: if s and p agreed everywhere, the gate above would be vacuous. At 40 deg
    the two are far apart, and 'y'/'p' track their s/p partners rather than each other -- which is
    exactly why normalize_pol refuses 'x' -> 's' off normal and demands the azimuth for 'y'."""
    pytest.importorskip("lumenairy")
    pytest.importorskip("tmm")
    th = 40.0
    Rs, Rp = _sp_R("s", th), _sp_R("p", th)
    assert abs(Rs - Rp) > 0.05 * max(Rs, Rp), (Rs, Rp)
    assert _lab_R("y", th) == pytest.approx(Rs, abs=1e-12)
    assert _lab_R("p", th) == pytest.approx(Rp, abs=1e-12)


# ------------------------------------------------------------------------------------------------
# normalize_pol: the conversions, and every refusal the map promises
# ------------------------------------------------------------------------------------------------
def test_normalize_pol_validates_without_widening_any_accepted_set():
    """The converter must not become a back door that admits spellings no module accepts."""
    for name, voc in VOCABULARIES.items():
        for good in voc.values:
            assert normalize_pol(good, name) == good
    assert normalize_pol("TE", "tetm") == "te"        # lumenairy's own alias, case-insensitive
    assert normalize_pol("s", "tetm") == "te"
    for bad, vocab in [("S", "sp"), ("TE", "sp"), ("y", "sp"), ("s", "lab_xyp"),
                       ("X", "lab_xyp"), ("te", "lab_xyp"), ("x", "tetm"), (2, "row"),
                       ("0", "row"), (True, "row"), ("s", "axis_xy"), ("p", "axis_xy")]:
        with pytest.raises(PolarizationVocabularyError):
            normalize_pol(bad, vocab)


def test_normalize_pol_refuses_every_ambiguous_conversion():
    """The V-8 design point: an ambiguous conversion is REFUSED, never defaulted."""
    # 'y' <-> 's' needs the azimuth, and is refused at conical
    with pytest.raises(PolarizationConversionError, match="azimuth"):
        normalize_pol("y", "lab_xyp", to="sp")
    with pytest.raises(PolarizationConversionError, match="conical"):
        normalize_pol("y", "lab_xyp", to="sp", azimuth_deg=30.0)
    with pytest.raises(PolarizationConversionError):
        normalize_pol("s", "sp", to="lab_xyp", azimuth_deg=30.0)
    # 'x' needs the polar angle and has no image off normal (sibling Q-24)
    with pytest.raises(PolarizationConversionError):
        normalize_pol("x", "lab_xyp", to="sp", azimuth_deg=0.0)
    with pytest.raises(PolarizationConversionError, match="transverse"):
        normalize_pol("x", "lab_xyp", to="sp", azimuth_deg=0.0, theta_deg=30.0)
    # row 0 -> a label is not a function ('x' and 'p' both map to 0)
    with pytest.raises(PolarizationConversionError, match="AMBIGUOUS"):
        normalize_pol(0, "row", to="lab_xyp")
    assert normalize_pol(1, "row", to="lab_xyp") == "y"
    # conical p-pol carries no single row
    with pytest.raises(PolarizationConversionError):
        normalize_pol("p", "lab_xyp", to="row", azimuth_deg=25.0)
    # te/tm is a classical-mount spelling
    with pytest.raises(PolarizationConversionError, match="CLASSICAL"):
        normalize_pol("te", "tetm", to="sp", azimuth_deg=25.0)
    # the 2-D in-plane family is a different GEOMETRY, not a different spelling
    for other in ("sp", "lab_xyp", "tetm", "row"):
        with pytest.raises(PolarizationConversionError, match="IN-PLANE"):
            normalize_pol("x", "axis_xy", to=other, azimuth_deg=0.0, theta_deg=0.0)
        with pytest.raises(PolarizationConversionError, match="IN-PLANE"):
            normalize_pol(VOCABULARIES[other].values[0], other, to="axis_xy",
                          azimuth_deg=0.0, theta_deg=0.0)


def test_normalize_pol_round_trips_where_the_map_says_it_can():
    at0 = dict(azimuth_deg=0.0, theta_deg=0.0)
    assert normalize_pol("s", "sp", to="lab_xyp", **at0) == "y"
    assert normalize_pol("y", "lab_xyp", to="sp", **at0) == "s"
    assert normalize_pol("s", "sp", to="tetm", **at0) == "te"
    assert normalize_pol("tm", "tetm", to="sp", **at0) == "p"
    assert normalize_pol("s", "sp", to="row", **at0) == 1          # chained via lab_xyp
    assert normalize_pol("te", "tetm", to="lab_xyp", **at0) == "y"  # chained via sp
    assert normalize_pol("p", "lab_xyp", to="row", **at0) == 0


def test_the_two_admittance_twins_still_share_one_vocabulary_and_one_message():
    """AUDIT V-3/V-8: the duplicated _admittance helpers had MIRROR-IMAGE silent fallbacks. Both
    raise now, and (the V-8 half) both raise the SAME shared text, so the twins cannot drift
    apart in their error contract the way they drifted apart in their defaults."""
    from dynameta.optics.nonlocal_tmm import _admittance as adm_nl
    from dynameta.optics.resonance import _admittance as adm_res

    for bad in ("TE", "TM", "te", "tm", "S", "P", "x", "y", "", "sp"):
        msgs = []
        for fn in (adm_res, adm_nl):
            with pytest.raises(PolarizationVocabularyError) as exc:
                fn(4.0 + 0.1j, 1.2e7 + 0j, bad)
            msgs.append(str(exc.value))
            assert "pol must be" in msgs[-1]              # the long-standing matched phrase
            assert "dynameta.core.polarization" in msgs[-1]
        head = [m.split(" -- ")[0].split(": ", 1)[1] for m in msgs]
        assert head[0] == head[1], msgs                   # one vocabulary, one sentence
