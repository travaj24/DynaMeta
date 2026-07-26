"""THE polarization-vocabulary map (audit V-8) -- one documented home for the five spellings of
"which polarization" that coexist across ~18 DynaMeta modules, plus the validator/converter and
the shared error text every one of those modules now raises through.

SCOPE (read this first). This module is VALIDATION + DOCUMENTATION only. Every entry point in the
map keeps accepting EXACTLY the vocabulary it accepted before -- no parameter was renamed, no new
spelling was made acceptable anywhere, and no valid call changed by a single bit. What changed is
that an INVALID label now fails with a message that names the sibling vocabulary the label came
from and points here, instead of falling through to "whatever the ``else`` branch was".
UNIFICATION-OF-ACCEPTANCE (making, say, ``smatrix_pole_func`` take ``'TE'`` or ``OpticalSpec``
take ``'s'``) is a BREAKING design decision and a deliberate FOLLOW-ON, not part of this change:
it would silently re-point existing call sites at a different physical mode.


THE MAP
=======

Five vocabularies, three geometries. The vocabulary is a property of the GEOMETRY the entry point
solves, which is why they did not (and still do not) collapse into one.

+------------+---------------------+--------------------------------------------------------+
| vocabulary | values              | what the label names                                   |
+============+=====================+========================================================+
| ``sp``     | ``'s'``, ``'p'``    | orientation of E relative to the PLANE OF INCIDENCE    |
|            | (case-sensitive)    | (the plane through the layer normal z and k_par).      |
|            |                     | s = E perpendicular to it (TE); p = E in it (TM).      |
|            |                     | The physics-standard spelling; undefined at theta = 0  |
|            |                     | (no plane of incidence) where s and p degenerate.      |
+------------+---------------------+--------------------------------------------------------+
| ``lab_xyp``| ``'x'``, ``'y'``,   | ``OpticalSpec.polarization``: the LAB AXIS the incident|
|            | ``'p'``             | E lies along, for a stack whose normal is z. ``'y'`` is|
|            | (case-sensitive)    | s-pol (E perpendicular to the x-z plane), ``'p'`` is   |
|            |                     | p-pol, and ``'x'`` is E along lab x -- transverse ONLY |
|            |                     | at normal incidence, which is why ``OpticalSpec``      |
|            |                     | rejects ``'x'`` at theta != 0 or phi != 0.             |
+------------+---------------------+--------------------------------------------------------+
| ``tetm``   | ``'te'``, ``'tm'``  | lumenairy's 1-D grating spelling: te = E along the     |
|            | (+ lumenairy's own  | grooves (= s for a classical mount), tm = p. The ONE   |
|            | case-insensitive    | family that also accepts ``'s'``/``'p'`` -- not a      |
|            | ``'s'``/``'p'``     | DynaMeta choice: ``lumenairy.rcwa_efficiency_1d``      |
|            | aliases)            | normalizes them upstream and always has.               |
+------------+---------------------+--------------------------------------------------------+
| ``row``    | ``0``, ``1``        | an INDEX, not a label: the row of a lab-basis Jones /  |
|            | (int)               | power pair, ``0 = incident E_x``, ``1 = incident E_y``.|
|            |                     | The differentiable Berreman/RCWA/PMM forwards return   |
|            |                     | ``(2,)`` arrays and select with it.                    |
+------------+---------------------+--------------------------------------------------------+
| ``axis_xy``| ``'x'``, ``'y'``    | ``hydro_fem.pol_axis``: a 2-D IN-PLANE geometry (a     |
|            |                     | cylinder / dimer cross-section). E lies along the named|
|            |                     | in-plane axis and the wave PROPAGATES along the other. |
|            |                     | There is no layer normal here, so s/p and the lab      |
|            |                     | x/y/p of a stack do not apply at all.                  |
+------------+---------------------+--------------------------------------------------------+


THE NORMAL-INCIDENCE x/y <-> s/p CORRESPONDENCE (and its azimuth caveat)
=======================================================================

At theta = 0 the plane of incidence is UNDEFINED, so a scalar isotropic layered stack is
polarization-DEGENERATE: ``'x'``, ``'y'``, ``'s'``, ``'p'``, ``'te'``, ``'tm'``, row 0 and row 1
all give the SAME R and T, to machine precision. That degeneracy is gated numerically in
``tests/test_polarization_vocabulary.py``: one stack solved through an s/p API (Byrnes-``tmm``)
and an x/y API (lumenairy's Berreman 4x4, reached through ``OpticalSpec -> _common.pol_row -> lab
Jones row``) spreads 1.8e-16 across {'s','p','x','y'}, against a 1e-12 bound. That degeneracy is
the whole reason the vocabularies were able to drift apart unnoticed: at the DEFAULT incidence
every spelling gives the same number. The same gate's discrimination leg at 40 deg has R_s and
R_p 67% apart, with 'y' still tracking 's' and 'p' tracking 'p' to 1.5e-16.

Away from theta = 0 the correspondence is:

  * ``'y'`` <-> ``'s'`` and ``'p'`` <-> ``'p'`` **at azimuth phi = 0 only**;
  * ``'x'`` has NO s/p image at theta != 0 -- E along lab x is not transverse to an oblique x-z
    wavevector (``OpticalSpec`` refuses it; ``tmm_reference._pol_for`` maps it to ``'s'`` and is
    therefore valid only on the normal-incidence branch, sibling finding Q-24);
  * at phi != 0 (CONICAL) the label ``'y'`` is genuinely ambiguous across the repo. The FEM
    solver and ``OpticalSpec`` read it as the ROTATED s-hat = ``(-sin phi, cos phi, 0)``
    (``optics/solver.py``, ``_common.pol_tangential_unit``), while the lumenairy bridge's
    ``_POL_ROW`` reads it as lab row 1 = E_y, which at phi != 0 is a phi-dependent s/p MIXTURE.
    The bridge refuses conical incidence outright for exactly this reason
    (``_common.guard_conical_ppol``, audit C4-2), and :func:`normalize_pol` likewise REFUSES the
    ``'y'`` <-> ``'s'`` conversion at phi != 0 rather than pick a reading.

So: every conversion out of ``lab_xyp`` needs the azimuth, and the ``'x'`` ones need the polar
angle too. :func:`normalize_pol` demands them and raises when they are missing -- an ambiguous
conversion is never resolved by a default here.

``row`` is NOT injective back to ``lab_xyp``: ``_POL_ROW = {'x': 0, 'y': 1, 'p': 0}`` sends both
``'x'`` and ``'p'`` to row 0 (p-pol's transverse E points along lab x at phi = 0). Converting row
0 -> a label is therefore refused; row 1 -> ``'y'`` is fine.


THE FIXED-k_par 2-D TE/TM SUBTLETY
==================================

``optics/fdtd_nd/solve2d.solve_fdtd_2d_oblique`` and its kernels in ``fdtd_nd/oblique2d.py``
speak ``sp``, and their docstrings gloss ``pol='s'`` as "TE" and ``pol='p'`` as "TM". Three things
hide in that gloss:

1. In a 2-D (x, z) simulation, TE/TM conventionally name which field is OUT of the simulation
   plane. DynaMeta uses TE = E_y out of plane, TM = H_y out of plane. That coincides with s/p
   ONLY because the simulation plane IS the plane of incidence (these solvers are in-plane;
   there is no azimuth). The OPPOSITE 2-D convention (TE = E in-plane) is common in waveguide
   literature -- do not carry a TE label across from there.
2. The solve holds k_par FIXED across the band (it is a complex-envelope Bloch march), so the
   PHYSICAL angle sweeps with frequency: ``theta(f) = asin(k_par c / omega)``. The s/p LABEL is
   frequency-independent -- s/p is defined by the plane of incidence, which does not move -- but
   the angle it refers to is not. Read R(theta) off ``result.theta_deg``, never off the
   ``angle_deg`` you passed in (that is the band-CENTRE angle only).
3. Below the light line (k_par > omega/c) the incident wave is evanescent and no s/p decomposition
   of a propagating beam exists at all; the result's ``band`` mask marks those frequencies.


WHERE EACH VOCABULARY IS SPOKEN
===============================

The authoritative list is the executable :data:`VOCABULARIES` table below -- every entry point in
it is called by ``tests/test_polarization_vocabulary.py`` with its OWN vocabulary (must not raise
a polarization error) and with a WRONG one (must raise, naming the right vocabulary). Adding an
entry point without a caller fails that test, so this map cannot silently rot.

Two entries are marked ``delegates_to``: they TAKE a label and forward it unvalidated to a
sibling that does the checking (``translate.rcwa_stack_to_design`` -> the ``OpticalSpec`` it
builds; ``Design.detect_symmetry_reduction`` READS an already-validated ``OpticalSpec`` field).
They have no guard of their own to drive, so the test checks that their named validator is itself
in the map instead of pretending to exercise one.


IMPORT DISCIPLINE
=================

Stdlib only -- no numpy, no scipy, nothing from the rest of DynaMeta. Consumer modules import it
LAZILY, inside the failure branch, so the valid path pays nothing and no import edge is added to
the package graph (the byte-identity contract above is trivially true when the happy path never
touches this file).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PolarizationVocabularyError",
    "PolarizationConversionError",
    "EntryPoint",
    "Vocabulary",
    "VOCABULARIES",
    "MAP_REFERENCE",
    "UNIFICATION_FOLLOW_ON",
    "DOCSTRING_XREF",
    "ANGLE_TOL_DEG",
    "normalize_pol",
    "vocabulary_of",
    "all_entry_points",
    "pol_vocabulary_error",
]


class PolarizationVocabularyError(ValueError):
    """A polarization label is not in the vocabulary the entry point speaks (audit V-8).

    A ``ValueError`` subclass so every pre-existing ``except ValueError`` / ``pytest.raises(
    ValueError)`` around these guards keeps working unchanged; the subclass exists so a test can
    tell "rejected BECAUSE of the vocabulary" apart from "failed for some other reason"."""


class PolarizationConversionError(PolarizationVocabularyError):
    """A conversion between two polarization vocabularies is AMBIGUOUS or undefined for the
    geometry supplied (missing azimuth, oblique ``'x'``, conical ``'y'``, row 0 -> label, or any
    conversion involving the 2-D in-plane ``axis_xy`` family). Never resolved by a default."""


ANGLE_TOL_DEG = 1e-6
"""Angle below which theta / phi count as zero -- the same 1e-6 deg tolerance ``OpticalSpec``
uses for its own oblique / conical branch decisions, so the two never disagree about whether a
configuration is 'normal' or 'in-plane'."""

MAP_REFERENCE = ("The repo's polarization-vocabulary map -- which spelling means what in which "
                 "geometry, and normalize_pol() to convert between them -- is "
                 "dynameta.core.polarization (audit V-8).")

UNIFICATION_FOLLOW_ON = ("This entry point still accepts EXACTLY its own vocabulary; unifying "
                         "which spellings are ACCEPTED across the repo is a breaking change and "
                         "a deliberate follow-on, not part of the V-8 map.")

DOCSTRING_XREF = ("POLARIZATION VOCABULARY: this module speaks {names}. It is one of five "
                  "spellings in the repo; the map is dynameta.core.polarization (audit V-8).")


@dataclass(frozen=True)
class EntryPoint:
    """One place a polarization vocabulary is spoken. ``module``/``qualname`` are import strings;
    ``parameter`` is the argument (or dataclass field) carrying the label.

    ``values`` is a SUBSET override: a few entry points speak a vocabulary but implement only part
    of it (the normal-incidence magneto-optic FDTD engines take ``'x'``/``'y'`` and have no
    distinct p-pol mode). ``None`` means "the whole vocabulary".

    ``delegates_to`` names another entry point (``module.qualname``) that does the validating: a
    couple of sites take a label and hand it straight on without inspecting it, so they have no
    guard of their own to drive. The executable-map test checks that the delegation target is
    itself in the map rather than pretending to exercise a guard that is not there."""
    module: str
    qualname: str
    parameter: str
    public: bool = True
    note: str = ""
    values: Optional[Tuple] = None
    delegates_to: Optional[str] = None

    @property
    def dotted(self) -> str:
        return "{}.{}".format(self.module, self.qualname)

    def accepted_values(self, voc: "Vocabulary") -> Tuple:
        return voc.values if self.values is None else self.values


@dataclass(frozen=True)
class Vocabulary:
    """One polarization vocabulary: what it accepts, what the label MEANS, and every entry point
    that speaks it."""
    name: str
    values: Tuple
    geometry: str
    meaning: str
    entry_points: Tuple[EntryPoint, ...]
    aliases: Tuple[str, ...] = ()
    case_insensitive: bool = False

    @property
    def accepted(self) -> str:
        """Human-readable accepted set, for error text."""
        vals = [repr(v) for v in self.values]
        core = (vals[0] if len(vals) == 1
                else "{} or {}".format(", ".join(vals[:-1]), vals[-1]))
        if self.aliases:
            core += " (or the {} aliases)".format(
                ", ".join(repr(a) for a in self.aliases))
        if not self.case_insensitive and all(isinstance(v, str) for v in self.values):
            core += " (case-sensitive)"
        return core

    def wrong_value(self):
        """A value from a SIBLING vocabulary that this one must reject -- the executable map's
        negative probe."""
        return _WRONG_PROBE[self.name]


# ------------------------------------------------------------------------------------------------
# The table.  Order matters only for the error text (siblings are listed in this order).
# ------------------------------------------------------------------------------------------------
VOCABULARIES: Dict[str, Vocabulary] = {}


def _add(v: Vocabulary) -> Vocabulary:
    VOCABULARIES[v.name] = v
    return v


_add(Vocabulary(
    name="sp",
    values=("s", "p"),
    geometry="planar stack, orientation of E relative to the plane of incidence",
    meaning="s = E perpendicular to the plane of incidence (TE); p = E in it (TM)",
    entry_points=(
        EntryPoint("dynameta.optics.tmm_reference", "stack_rta", "pol"),
        EntryPoint("dynameta.optics.tmm_reference", "layered_rta", "pol"),
        EntryPoint("dynameta.optics.tmm_reference", "layered_per_layer_absorption", "pol"),
        EntryPoint("dynameta.optics.resonance", "layered_smatrix_complex", "pol"),
        EntryPoint("dynameta.optics.resonance", "smatrix_pole_func", "pol"),
        EntryPoint("dynameta.optics.resonance", "_admittance", "pol", public=False,
                   note="duplicated twin of nonlocal_tmm._admittance (Q-19); the two used to "
                        "have MIRROR-IMAGE silent fallbacks (V-3)"),
        EntryPoint("dynameta.optics.nonlocal_tmm", "stack_rt", "pol"),
        EntryPoint("dynameta.optics.nonlocal_tmm", "pole_function", "pol"),
        EntryPoint("dynameta.optics.nonlocal_tmm", "_admittance", "pol", public=False,
                   note="the other half of the V-3 twin pair"),
        EntryPoint("dynameta.optics.shg_fem", "rudnick_stern_flat_shg", "polarization"),
        EntryPoint("dynameta.optics.shg_fem", "rudnick_stern_flat_sfg", "polarization"),
        EntryPoint("dynameta.optics.fdtd_nd.solve2d", "solve_fdtd_2d_oblique", "pol",
                   note="'s' is glossed TE and 'p' TM -- see the fixed-k_par subtlety above"),
        EntryPoint("dynameta.optics.fdtd_nd.oblique2d", "_run_oblique", "pol", public=False,
                   note="the 2-D complex-envelope kernel dispatcher behind solve_fdtd_2d_oblique"),
    ),
))

_add(Vocabulary(
    name="lab_xyp",
    values=("x", "y", "p"),
    geometry="planar stack, lab axis of the incident E (stack normal = z)",
    meaning="'y' = s-pol, 'p' = p-pol, 'x' = E along lab x (transverse only at theta = phi = 0)",
    entry_points=(
        EntryPoint("dynameta.geometry.specs", "OpticalSpec", "polarization",
                   note="the dataclass field every lab_xyp consumer reads"),
        EntryPoint("dynameta.geometry.design", "Design.detect_symmetry_reduction", "polarization",
                   delegates_to="dynameta.geometry.specs.OpticalSpec",
                   note="ADVISORY reader, not a taker: it reads self.optical.polarization (already "
                        "validated by OpticalSpec) and returns 'none' for 'p', because the "
                        "symmetry wall type is keyed to a linear x/y E axis"),
        EntryPoint("dynameta.optics.solver", "background_probe_pol", "polarization"),
        EntryPoint("dynameta.optics.solver", "_layered_background", "polarization", public=False),
        EntryPoint("dynameta.optics.ngsolve_layered",
                   "LayeredOpticalBuilder._check_symmetry_supported", "polarization",
                   public=False,
                   note="reads OpticalSpec.polarization for the symmetry-wall type; the "
                        "reduction is x/y only ('p' has no single wall type)"),
        EntryPoint("dynameta.optics.fdtd_seam", "_guard_optical_spec", "polarization",
                   public=False),
        EntryPoint("dynameta.optics.tmm_reference", "_pol_for", "polarization", public=False,
                   note="the lab_xyp -> sp bridge; 'x' -> 's' is the normal-incidence branch "
                        "only (Q-24)"),
        EntryPoint("dynameta.optics.lumenairy_bridge._common", "pol_row", "polarization",
                   public=False, note="the lab_xyp -> row bridge (_POL_ROW)"),
        EntryPoint("dynameta.optics.lumenairy_bridge._common", "p_basis_conversion", "pol",
                   public=False),
        EntryPoint("dynameta.optics.lumenairy_bridge._common", "pol_tangential_unit", "pol",
                   public=False),
        EntryPoint("dynameta.optics.lumenairy_bridge.translate", "rcwa_stack_to_design",
                   "polarization", delegates_to="dynameta.geometry.specs.OpticalSpec",
                   note="the label is handed straight to the OpticalSpec it builds, which "
                        "validates it -- note the asymmetry with rcwa_design.rcwa_grating_RT in "
                        "the SAME package, which speaks {'te','tm'}"),
        EntryPoint("dynameta.optics.fdtd_mo", "solve_fdtd_mo_1d", "pol", values=("x", "y"),
                   note="NORMAL incidence only: the source is a linear lab axis and there is no "
                        "distinct p-pol mode, so 'p' is refused as an unimplemented subset "
                        "member (it used to take the 'x' branch silently)"),
        EntryPoint("dynameta.optics.fdtd_nd.solve3d", "solve_fdtd_3d_mo", "pol",
                   values=("x", "y"), note="same normal-incidence subset as fdtd_mo"),
        EntryPoint("dynameta.optics.fdtd_nd.kernels3d", "_run_3d_mo", "pol", public=False,
                   values=("x", "y"), note="the 3-D magneto-optic kernel behind solve_fdtd_3d_mo"),
    ),
))

_add(Vocabulary(
    name="tetm",
    values=("te", "tm"),
    aliases=("s", "p"),
    case_insensitive=True,
    geometry="1-D grating, classical (in-plane) mount",
    meaning="te = E along the grooves (= s); tm = p",
    entry_points=(
        EntryPoint("dynameta.optics.lumenairy_bridge.rcwa_design", "rcwa_grating_RT",
                   "polarization",
                   note="the only family that also takes 's'/'p', case-insensitively -- "
                        "lumenairy.rcwa_efficiency_1d normalizes them upstream"),
    ),
))

_add(Vocabulary(
    name="row",
    values=(0, 1),
    geometry="planar stack, index into a lab-basis (2,) result",
    meaning="0 = incident E_x, 1 = incident E_y (an INDEX, not a label)",
    entry_points=(
        EntryPoint("dynameta.optics.lumenairy_bridge.berreman_design", "berreman_RT", "row"),
        EntryPoint("dynameta.optics.lumenairy_bridge.rcwa_design", "rcwa_stack_RT", "row"),
        EntryPoint("dynameta.optics.lumenairy_bridge.rcwa_design", "pmm_stack_RT", "row"),
        EntryPoint("dynameta.optics.lumenairy_bridge.berreman_backend",
                   "berreman_result_to_optical_result", "row", public=False),
        EntryPoint("dynameta.optics.lumenairy_bridge.rcwa_backend",
                   "rcwa_result_to_optical_result", "row", public=False),
    ),
))

_add(Vocabulary(
    name="axis_xy",
    values=("x", "y"),
    geometry="2-D IN-PLANE cross-section (cylinder / dimer); no layer normal",
    meaning="E along the named in-plane axis, propagation along the other",
    entry_points=(
        EntryPoint("dynameta.optics.hydro_fem", "gap_enhancement_2d", "pol_axis"),
        EntryPoint("dynameta.optics.hydro_fem", "scattering_2d", "pol_axis"),
    ),
))

# One value per vocabulary that is VALID somewhere else in the repo and must be rejected here --
# the negative probe the executable-map test drives, and the reason the error text can always
# name a sibling.
_WRONG_PROBE = {
    "sp": "y",        # lab_xyp
    "lab_xyp": "s",   # sp
    "tetm": "y",      # lab_xyp ('s'/'p' ARE accepted here, so the probe must be a lab axis)
    "row": "x",       # lab_xyp
    "axis_xy": "s",   # sp
}


# ------------------------------------------------------------------------------------------------
# Lookup / introspection
# ------------------------------------------------------------------------------------------------
def all_entry_points() -> List[Tuple[str, EntryPoint]]:
    """``[(vocabulary_name, EntryPoint), ...]`` over the whole map -- what the executable-map
    test iterates."""
    return [(v.name, ep) for v in VOCABULARIES.values() for ep in v.entry_points]


def vocabulary_of(value, exclude: Sequence[str] = ()) -> List[str]:
    """Names of the vocabularies that WOULD accept ``value``, excluding ``exclude``. Used to build
    an error that says where a rejected label actually came from."""
    out = []
    for name, voc in VOCABULARIES.items():
        if name in exclude:
            continue
        try:
            _canonical(value, voc)
        except PolarizationVocabularyError:
            continue
        out.append(name)
    return out


def _canonical(value, voc: Vocabulary):
    """The canonical form of ``value`` in ``voc``, or raise. This is the ONLY acceptance test in
    the repo, and it accepts exactly what the modules accepted before V-8 -- nothing more."""
    if voc.name == "row":
        if isinstance(value, bool) or not isinstance(value, int) or value not in voc.values:
            raise PolarizationVocabularyError("not in {}".format(voc.name))
        return int(value)
    if not isinstance(value, str):
        raise PolarizationVocabularyError("not in {}".format(voc.name))
    cand = value.lower() if voc.case_insensitive else value
    if cand in voc.values:
        return cand
    if voc.case_insensitive and cand in voc.aliases:
        return {"s": "te", "p": "tm"}[cand]
    raise PolarizationVocabularyError("not in {}".format(voc.name))


# ------------------------------------------------------------------------------------------------
# Error text -- the ONE place these messages are written
# ------------------------------------------------------------------------------------------------
def pol_vocabulary_error(value, vocabulary: str, *, where: Optional[str] = None,
                         param: str = "pol", extra: str = "",
                         allowed: Optional[Sequence] = None) -> PolarizationVocabularyError:
    """Build (do not raise) the shared "wrong vocabulary" error. ``where`` is the calling function
    name, ``param`` the argument name as the caller spells it (``pol`` / ``polarization`` /
    ``pol_axis`` / ``row``) so the message quotes the user's own signature.

    ``allowed`` marks an entry point that implements only a SUBSET of its vocabulary. A value
    inside the vocabulary but outside ``allowed`` gets the "this engine implements only ..."
    message instead of the (false) "wrong vocabulary" one."""
    voc = VOCABULARIES[vocabulary]
    head = "{}: ".format(where) if where else ""
    if allowed is not None:
        try:
            in_vocab = _canonical(value, voc) in tuple(allowed)
        except PolarizationVocabularyError:
            in_vocab = None
        if in_vocab is False:
            msg = ["{}{} must be {} here -- this entry point implements only that SUBSET of the "
                   "{!r} vocabulary {}; got {!r}.".format(
                       head, param, " or ".join(repr(v) for v in allowed), voc.name,
                       list(voc.values), value)]
            if extra:
                msg.append(extra)
            msg.append(MAP_REFERENCE)
            return PolarizationVocabularyError(" ".join(msg))
    msg = ["{}{} must be {} -- the {!r} vocabulary ({}); got {!r}.".format(
        head, param, voc.accepted, voc.name, voc.geometry, value)]
    siblings = vocabulary_of(value, exclude=(vocabulary,))
    if siblings:
        other = VOCABULARIES[siblings[0]]
        msg.append("{!r} is the {!r} vocabulary ({}: {}), spoken by {}.".format(
            value, other.name, other.geometry, other.meaning,
            ", ".join(ep.dotted for ep in other.entry_points[:3])))
        msg.append("This entry point does NOT accept it. Convert explicitly: "
                   "normalize_pol({!r}, {!r}, to={!r}, theta_deg=..., azimuth_deg=...).".format(
                       value, other.name, voc.name))
    else:
        msg.append("It belongs to no DynaMeta polarization vocabulary; the others are {}.".format(
            ", ".join("{} {}".format(v.name, list(v.values))
                      for v in VOCABULARIES.values() if v.name != vocabulary)))
    if extra:
        msg.append(extra)
    msg.append(MAP_REFERENCE)
    return PolarizationVocabularyError(" ".join(msg))


def _conversion_error(value, frm: str, to: str, why: str) -> PolarizationConversionError:
    return PolarizationConversionError(
        "normalize_pol: cannot convert {!r} from the {!r} vocabulary to {!r} -- {} {}".format(
            value, frm, to, why, MAP_REFERENCE))


# ------------------------------------------------------------------------------------------------
# normalize_pol
# ------------------------------------------------------------------------------------------------
def _zero(angle: Optional[float]) -> bool:
    return angle is not None and abs(float(angle)) <= ANGLE_TOL_DEG


def _need_azimuth(value, frm, to, azimuth_deg):
    if azimuth_deg is None:
        raise _conversion_error(
            value, frm, to,
            "the s/p <-> lab-axis correspondence holds only at azimuth phi = 0 and this call did "
            "not say what phi is. Pass azimuth_deg explicitly (azimuth_deg=0.0 asserts an "
            "in-plane mount); at phi != 0 the label 'y' means the ROTATED s-hat to the FEM solver "
            "and OpticalSpec, but lab row 1 = E_y (a phi-dependent s/p MIXTURE) to the lumenairy "
            "bridge, which is why the bridge refuses conical incidence outright (audit C4-2).")


def _lab_to_sp(value, theta_deg, azimuth_deg):
    if value == "p":
        return "p"
    _need_azimuth(value, "lab_xyp", "sp", azimuth_deg)
    if not _zero(azimuth_deg):
        raise _conversion_error(
            value, "lab_xyp", "sp",
            "at azimuth phi = {:g} deg (conical) the lab axes are not the s/p eigen-basis: 'y' "
            "reads as the rotated s-hat in the FEM/OpticalSpec and as lab row 1 in the lumenairy "
            "bridge, and those disagree. Solve conical incidence with the FEM path, or set "
            "azimuth_deg = 0.".format(float(azimuth_deg)))
    if value == "y":
        return "s"
    # value == 'x'
    if theta_deg is None:
        raise _conversion_error(
            value, "lab_xyp", "sp",
            "'x' (E along lab x) has an s/p image ONLY at normal incidence, where a scalar "
            "layered stack is polarization-degenerate; at theta != 0 it is not even transverse to "
            "the wavevector (OpticalSpec rejects it there). Pass theta_deg to assert which branch "
            "you are on.")
    if not _zero(theta_deg):
        raise _conversion_error(
            value, "lab_xyp", "sp",
            "'x' (E along lab x) is not transverse to an oblique x-z-plane wavevector "
            "(theta = {:g} deg), so it has no s/p image; OpticalSpec rejects 'x' at theta != 0 "
            "for the same reason (sibling finding Q-24).".format(float(theta_deg)))
    return "s"


def _sp_to_lab(value, theta_deg, azimuth_deg):
    if value == "p":
        return "p"
    _need_azimuth(value, "sp", "lab_xyp", azimuth_deg)
    if not _zero(azimuth_deg):
        raise _conversion_error(
            value, "sp", "lab_xyp",
            "at azimuth phi = {:g} deg the s-hat is the ROTATED (-sin phi, cos phi) direction, "
            "which is lab 'y' to the FEM solver but a mixture of lab rows to the lumenairy "
            "bridge; pick the geometry explicitly rather than let this function choose."
            .format(float(azimuth_deg)))
    return "y"


def _sp_to_tetm(value, theta_deg, azimuth_deg):
    if azimuth_deg is not None and not _zero(azimuth_deg):
        raise _conversion_error(
            value, "sp", "tetm",
            "te/tm is defined for the CLASSICAL (in-plane) grating mount; at azimuth "
            "phi = {:g} deg (conical mount) neither diffracted order is purely te or tm."
            .format(float(azimuth_deg)))
    return {"s": "te", "p": "tm"}[value]


def _tetm_to_sp(value, theta_deg, azimuth_deg):
    if azimuth_deg is not None and not _zero(azimuth_deg):
        raise _conversion_error(
            value, "tetm", "sp",
            "te/tm is defined for the CLASSICAL (in-plane) grating mount; at azimuth "
            "phi = {:g} deg (conical mount) it has no single s/p image."
            .format(float(azimuth_deg)))
    return {"te": "s", "tm": "p"}[value]


def _lab_to_row(value, theta_deg, azimuth_deg):
    if value in ("x", "y"):
        return {"x": 0, "y": 1}[value]
    _need_azimuth(value, "lab_xyp", "row", azimuth_deg)
    if not _zero(azimuth_deg):
        raise _conversion_error(
            value, "lab_xyp", "row",
            "p-pol's transverse E points along lab x ONLY at phi = 0; at phi = {:g} deg it splits "
            "over BOTH lab rows as (cos phi, sin phi), so no single row carries it (this is why "
            "_common.guard_conical_ppol refuses conical p-pol).".format(float(azimuth_deg)))
    return 0


def _row_to_lab(value, theta_deg, azimuth_deg):
    if value == 1:
        return "y"
    raise _conversion_error(
        value, "row", "lab_xyp",
        "row 0 is AMBIGUOUS: _POL_ROW = {'x': 0, 'y': 1, 'p': 0} maps BOTH 'x' and 'p' onto row "
        "0 (p-pol's transverse E lies along lab x at phi = 0), so the inverse is not a function. "
        "Say which one you mean.")


def _refuse_axis(value, frm, to):
    raise _conversion_error(
        value, frm, to,
        "the {!r} vocabulary describes a 2-D IN-PLANE cross-section (hydro_fem): the wave "
        "propagates INSIDE the x-y plane and there is no layer normal, so there is no plane of "
        "incidence to define s/p and no stack axis to define the lab x/y/p of OpticalSpec. The "
        "two geometries are not convertible -- they are different problems that happen to share "
        "the letters 'x' and 'y'.".format("axis_xy"))


_CONVERSIONS = {
    ("lab_xyp", "sp"): _lab_to_sp,
    ("sp", "lab_xyp"): _sp_to_lab,
    ("sp", "tetm"): _sp_to_tetm,
    ("tetm", "sp"): _tetm_to_sp,
    ("lab_xyp", "row"): _lab_to_row,
    ("row", "lab_xyp"): _row_to_lab,
}


def normalize_pol(value, vocabulary: str = "sp", *, to: Optional[str] = None,
                  theta_deg: Optional[float] = None, azimuth_deg: Optional[float] = None,
                  where: Optional[str] = None, param: str = "pol"):
    """Validate ``value`` against ``vocabulary`` and, when ``to`` is given, convert it.

    With ``to=None`` (the default) this is a pure GUARD: it returns the canonical form of
    ``value`` in ``vocabulary`` and raises :class:`PolarizationVocabularyError` -- naming the
    sibling vocabulary the label came from -- otherwise. The accepted set is exactly what the
    corresponding modules accepted before audit V-8: no new spelling is admitted anywhere, so a
    call that worked keeps working, bit for bit.

    With ``to=<other vocabulary>`` it converts, and REFUSES rather than guess whenever the
    conversion depends on geometry the caller did not supply:

      * ``lab_xyp`` <-> ``sp`` needs ``azimuth_deg`` (the 'y' <-> 's' identity holds only at
        phi = 0; at phi != 0 the repo itself carries two incompatible readings of 'y');
      * ``'x'`` -> ``'s'`` additionally needs ``theta_deg`` and is valid only at theta = 0;
      * ``lab_xyp`` -> ``row`` needs ``azimuth_deg`` for ``'p'``;
      * ``row`` 0 -> a label is refused outright (``{'x': 0, 'y': 1, 'p': 0}`` is not injective);
      * ``sp`` <-> ``tetm`` refuses a conical mount;
      * anything involving ``axis_xy`` (the 2-D in-plane hydro_fem geometry) is refused -- it is
        a different problem, not a different spelling.

    Parameters
    ----------
    value : str or int
        The label as the caller spelled it.
    vocabulary : str
        Key of :data:`VOCABULARIES` -- the vocabulary ``value`` is claimed to be in.
    to : str, optional
        Target vocabulary. ``None`` (or equal to ``vocabulary``) = validate only.
    theta_deg, azimuth_deg : float, optional
        Polar / azimuthal incidence angle in DEGREES. Required by the conversions listed above;
        never defaulted.
    where : str, optional
        Calling function name, quoted at the head of the error.
    param : str
        The parameter name as the caller spells it, so the message reads like their signature.

    Returns
    -------
    str or int
        The canonical value in ``to`` (or in ``vocabulary`` when ``to`` is None).
    """
    if vocabulary not in VOCABULARIES:
        raise KeyError("unknown polarization vocabulary {!r}; known: {}".format(
            vocabulary, sorted(VOCABULARIES)))
    try:
        canon = _canonical(value, VOCABULARIES[vocabulary])
    except PolarizationVocabularyError:
        raise pol_vocabulary_error(value, vocabulary, where=where, param=param) from None
    if to is None or to == vocabulary:
        return canon
    if to not in VOCABULARIES:
        raise KeyError("unknown polarization vocabulary {!r}; known: {}".format(
            to, sorted(VOCABULARIES)))
    if "axis_xy" in (vocabulary, to):
        _refuse_axis(canon, vocabulary, to)
    fn = _CONVERSIONS.get((vocabulary, to))
    if fn is not None:
        return fn(canon, theta_deg, azimuth_deg)
    # Two hops through the pivot the pair shares.
    for pivot in ("sp", "lab_xyp"):
        first = _CONVERSIONS.get((vocabulary, pivot))
        second = _CONVERSIONS.get((pivot, to))
        if first is not None and second is not None:
            return second(first(canon, theta_deg, azimuth_deg), theta_deg, azimuth_deg)
    raise _conversion_error(canon, vocabulary, to, "there is no defined mapping between them.")
