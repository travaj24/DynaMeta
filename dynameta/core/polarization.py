"""THE polarization-vocabulary map (audit V-8) -- one documented home for the five spellings of
"which polarization" that coexist across ~18 DynaMeta modules, plus the validator/converter and
the shared error text every one of those modules now raises through.

SCOPE (read this first). This module is VALIDATION + CONVERSION + DOCUMENTATION. No parameter was
renamed and NO VALID CALL CHANGED BY A SINGLE BIT: a canonical label (``'s'``, ``'p'``, ``'x'``,
``0`` ...) still travels the exact path it always did -- every guard is a fast ``not in`` test that
fires only on labels that used to be REJECTED, and the shared error text below is imported LAZILY
inside that failure branch. What changed, in two steps:

  * V-8 (wave 5) made an INVALID label fail with a message that names the sibling vocabulary the
    label came from and points here, instead of falling through to "whatever the ``else`` branch
    was".
  * ACCEPTANCE UNIFICATION, option (b) -- the V-8 follow-on, implemented here. The ``sp`` family
    (the 13 planar-stack entry points listed below) additionally accepts ``'te'``/``'tm'`` and
    MIXED CASE, normalized to ``'s'``/``'p'`` at the boundary. That widening is UNCONDITIONAL
    because the identity behind it is geometry-INDEPENDENT and lossless: in a planar stack TE is
    s and TM is p BY DEFINITION of the plane of incidence -- at every theta, every azimuth, in
    every material. Nothing that used to work stops working, nothing that used to work changes
    answer; only previously-RAISING spellings now succeed, and they succeed with the mode their
    name already meant. ``'TE'`` is exactly ``'s'``, bit for bit, because it IS the same call
    after a two-character normalization at the door.

What was deliberately NOT unified, and why -- these stay STRICT, and an ``sp`` API handed ``'y'``
still raises:

  * ``lab_xyp`` <-> ``sp`` is AZIMUTH-DEPENDENT (``'y'`` is s-pol only at phi = 0, and at phi != 0
    the repo itself carries two incompatible readings of ``'y'`` -- see below). Auto-accepting
    ``'y'`` in an s/p API would silently re-point a call site at a different physical mode at
    exactly the geometries where it matters.
  * ``axis_xy`` <-> anything: a different GEOMETRY (a 2-D in-plane cross-section, no layer normal),
    so no conversion exists to widen.
  * ``row`` -> a label: ``{'x': 0, 'y': 1, 'p': 0}`` is not injective, so the reverse is not a
    function.

The route across those three is EXPLICIT and yours to authorize: :func:`normalize_pol`, which
DEMANDS the geometry the conversion depends on and refuses rather than default. See
HOW TO CONVERT below; every strict guard's error message prints the exact call for you.


THE MAP
=======

Five vocabularies, three geometries. The vocabulary is a property of the GEOMETRY the entry point
solves, which is why they did not (and still do not) collapse into one.

+------------+---------------------+--------------------------------------------------------+
| vocabulary | values              | what the label names                                   |
+============+=====================+========================================================+
| ``sp``     | ``'s'``, ``'p'``    | orientation of E relative to the PLANE OF INCIDENCE    |
|            | (+ the ``'te'`` /   | (the plane through the layer normal z and k_par).      |
|            | ``'tm'`` aliases,   | s = E perpendicular to it (TE); p = E in it (TM).      |
|            | case-INsensitive)   | The physics-standard spelling; undefined at theta = 0  |
|            |                     | (no plane of incidence) where s and p degenerate.      |
|            |                     | TE == s and TM == p unconditionally here, so those two |
|            |                     | spellings are accepted and normalized at the door      |
|            |                     | (acceptance unification (b)); ``'x'``/``'y'`` are NOT. |
+------------+---------------------+--------------------------------------------------------+
| ``lab_xyp``| ``'x'``, ``'y'``,   | ``OpticalSpec.polarization``: the LAB AXIS the incident|
|            | ``'p'``             | E lies along, for a stack whose normal is z. ``'y'`` is|
|            | (case-sensitive)    | s-pol (E perpendicular to the x-z plane), ``'p'`` is   |
|            |                     | p-pol, and ``'x'`` is E along lab x -- transverse ONLY |
|            |                     | at normal incidence, which is why ``OpticalSpec``      |
|            |                     | rejects ``'x'`` at theta != 0 or phi != 0.             |
+------------+---------------------+--------------------------------------------------------+
| ``tetm``   | ``'te'``, ``'tm'``  | lumenairy's 1-D grating spelling: te = E along the     |
|            | (+ lumenairy's own  | grooves (= s for a classical mount), tm = p. Accepts   |
|            | case-insensitive    | ``'s'``/``'p'`` case-insensitively -- not a DynaMeta   |
|            | ``'s'``/``'p'``     | choice: ``lumenairy.rcwa_efficiency_1d._normalize_pol``|
|            | aliases)            | normalizes them upstream and always has, and this      |
|            |                     | guard's accepted set is byte-for-byte that one. The    |
|            |                     | ``sp`` family is now its mirror image (b).             |
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


HOW TO CONVERT
==============

Two routes, and the split between them is the whole design:

1. UNCONDITIONAL aliases -- just pass them. The entry point normalizes at the door because the
   identity holds for every geometry::

       stack_rta(..., pol='TE')    # == pol='s',  bit for bit
       stack_rt(...,  pol='tm')    # == pol='p',  bit for bit
       stack_rta(..., pol='S')     # == pol='s'   (mixed case is accepted for the sp family)

   The set is exactly {s, p, te, tm}, case-insensitive, on the 13 ``sp`` entry points below, and
   {te, tm, s, p}, case-insensitive, on the ``tetm`` grating entry point. Nothing else.

2. CONDITIONAL conversions -- the API stays STRICT and YOU convert, supplying the geometry::

       from dynameta.core.polarization import normalize_pol

       pol = normalize_pol('y', 'lab_xyp', to='sp', azimuth_deg=0.0)      # -> 's'
       R, T, A = stack_rta(n_super, layers, n_sub, lam, theta_deg=40.0, pol=pol)

   The azimuth is a REQUIRED argument, not a defaulted one, and that is the point: it is the
   caller who knows the mount, and the answer changes with it. ``azimuth_deg=0.0`` is an
   ASSERTION ("this is an in-plane mount"), not a fallback::

       normalize_pol('y', 'lab_xyp', to='sp')                    # raises: says "pass azimuth_deg"
       normalize_pol('y', 'lab_xyp', to='sp', azimuth_deg=30.0)  # raises: conical, 'y' is ambiguous

   THE phi != 0 HAZARD, which is why this cannot be automatic: at conical incidence the label
   ``'y'`` means the ROTATED s-hat ``(-sin phi, cos phi, 0)`` to ``optics/solver.py`` and
   ``OpticalSpec``, but lab row 1 = E_y -- a phi-dependent s/p MIXTURE -- to the lumenairy bridge
   (``_POL_ROW``). Those two readings disagree, the bridge refuses conical incidence outright for
   exactly that reason (audit C4-2), and an implicit ``'y' -> 's'`` inside an s/p entry point
   would have silently picked one of them for you at every azimuth.

   The same shape applies to the other conditional legs: ``'x' -> 's'`` additionally needs
   ``theta_deg`` and is valid only at theta = 0 (E along lab x is not transverse to an oblique
   wavevector); ``'p' -> row`` needs ``azimuth_deg``; ``row 0`` -> a label is refused outright;
   and anything touching ``axis_xy`` is refused because it is a different problem, not a
   different spelling.

Every strict guard prints the conversion call for you: the rejection message for a cross-family
label ends with the exact ``normalize_pol(...)`` line, listing only the context arguments THAT
value actually needs, and why you are the one supplying them.


THE FIXED-k_par 2-D TE/TM SUBTLETY
==================================

``optics/fdtd_nd/solve2d.solve_fdtd_2d_oblique`` and its kernels in ``fdtd_nd/oblique2d.py``
speak ``sp``, and their docstrings gloss ``pol='s'`` as "TE" and ``pol='p'`` as "TM". Three things
hide in that gloss:

1. In a 2-D (x, z) simulation, TE/TM conventionally name which field is OUT of the simulation
   plane. DynaMeta uses TE = E_y out of plane, TM = H_y out of plane. That coincides with s/p
   ONLY because the simulation plane IS the plane of incidence (these solvers are in-plane;
   there is no azimuth). The OPPOSITE 2-D convention (TE = E in-plane) is common in waveguide
   literature -- do not carry a TE label across from there. These two solvers DO accept
   ``'te'``/``'tm'`` under acceptance unification (b), and read them as THIS module's s/p; the
   alias is sound for the plane-of-incidence identity, and it is not a licence to import a label
   from a convention that means something else by it.
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
    "HOW_TO_CONVERT",
    "UNCONDITIONAL_ALIASES",
    "DOCSTRING_XREF",
    "ANGLE_TOL_DEG",
    "accept_pol",
    "conversion_example",
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

HOW_TO_CONVERT = ("HOW TO CONVERT: an UNCONDITIONAL alias (te/tm and mixed case for the s/p "
                  "family) is simply accepted and normalized at the door; a geometry-DEPENDENT "
                  "one (lab x/y/p <-> s/p, row -> label) is never implicit -- convert it "
                  "yourself with normalize_pol(), which demands the azimuth (and, for 'x', the "
                  "polar angle) and refuses rather than guess. See the HOW TO CONVERT section of "
                  "dynameta.core.polarization.")

UNCONDITIONAL_ALIASES = ("ACCEPTANCE UNIFICATION (b): this entry point also takes 'te'/'tm' and "
                         "mixed case, normalized to 's'/'p' at the boundary -- in a planar stack "
                         "TE is s and TM is p by definition of the plane of incidence, at every "
                         "angle, so the widening is unconditional and lossless. The "
                         "geometry-DEPENDENT spellings (OpticalSpec's lab 'x'/'y', the integer "
                         "row) are still REFUSED: convert those explicitly with normalize_pol().")

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
    that speaks it.

    ``aliases`` is the UNCONDITIONAL widening (acceptance unification (b)): pairs of
    ``(spelling, canonical)`` where the spelling names the SAME physical mode as the canonical
    value in EVERY geometry this vocabulary covers, so accepting it is lossless and needs no
    context. A geometry-DEPENDENT correspondence is NOT an alias and must never be listed here --
    it lives in :func:`normalize_pol`, which demands the geometry and refuses rather than guess."""
    name: str
    values: Tuple
    geometry: str
    meaning: str
    entry_points: Tuple[EntryPoint, ...]
    aliases: Tuple[Tuple[str, str], ...] = ()
    case_insensitive: bool = False

    @property
    def alias_map(self) -> Dict[str, str]:
        """``{spelling: canonical}`` for the unconditional aliases."""
        return dict(self.aliases)

    @property
    def accepted(self) -> str:
        """Human-readable accepted set, for error text."""
        vals = [repr(v) for v in self.values]
        core = (vals[0] if len(vals) == 1
                else "{} or {}".format(", ".join(vals[:-1]), vals[-1]))
        if self.aliases:
            core += " (or the {} aliases)".format(
                ", ".join(repr(a) for a, _ in self.aliases))
        if all(isinstance(v, str) for v in self.values):
            core += " (case-insensitive)" if self.case_insensitive else " (case-sensitive)"
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
    aliases=(("te", "s"), ("tm", "p")),
    case_insensitive=True,
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
    aliases=(("s", "te"), ("p", "tm")),
    case_insensitive=True,
    geometry="1-D grating, classical (in-plane) mount",
    meaning="te = E along the grooves (= s); tm = p",
    entry_points=(
        EntryPoint("dynameta.optics.lumenairy_bridge.rcwa_design", "rcwa_grating_RT",
                   "polarization",
                   note="takes 's'/'p' case-insensitively, byte-for-byte the set "
                        "lumenairy.rcwa_efficiency_1d._normalize_pol accepts (it normalizes them "
                        "upstream and always has); the sp family is now the mirror image"),
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
# name a sibling.  Under acceptance unification (b) the two families that share the PLANE-OF-
# INCIDENCE identity (sp and tetm) accept each other's spellings, so BOTH of their probes have to
# be a lab axis: the surviving strict boundary is the geometry-dependent one, and that is exactly
# what these probes now measure.
_WRONG_PROBE = {
    "sp": "y",        # lab_xyp ('te'/'tm' ARE accepted here now, so the probe must be a lab axis)
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
    an error that says where a rejected label actually came from.

    A vocabulary in which ``value`` is CANONICAL is listed before one where it is only an
    unconditional alias, so the error names ``'te'`` as the grating family's word rather than the
    s/p family's borrowing of it."""
    canonical, alias = [], []
    for name, voc in VOCABULARIES.items():
        if name in exclude:
            continue
        try:
            _canonical(value, voc)
        except PolarizationVocabularyError:
            continue
        (canonical if _is_canonical_spelling(value, voc) else alias).append(name)
    return canonical + alias


def _is_canonical_spelling(value, voc: Vocabulary) -> bool:
    if not isinstance(value, str):
        return True
    cand = value.lower() if voc.case_insensitive else value
    return cand in voc.values


def _canonical(value, voc: Vocabulary):
    """The canonical form of ``value`` in ``voc``, or raise. This is the ONLY acceptance test in
    the repo: the vocabulary's own values, plus the UNCONDITIONAL aliases declared in the table
    (acceptance unification (b)) -- and nothing else. Geometry-dependent correspondences are not
    reachable from here; they need :func:`normalize_pol` and an explicit azimuth."""
    if voc.name == "row":
        if isinstance(value, bool) or not isinstance(value, int) or value not in voc.values:
            raise PolarizationVocabularyError("not in {}".format(voc.name))
        return int(value)
    if not isinstance(value, str):
        raise PolarizationVocabularyError("not in {}".format(voc.name))
    cand = value.lower() if voc.case_insensitive else value
    if cand in voc.values:
        return cand
    alias = voc.alias_map.get(cand)
    if alias is not None:
        return alias
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
        msg.append("This entry point does NOT accept it: the two are not the same word for the "
                   "same thing, so there is no unconditional alias to widen to. "
                   + conversion_example(value, other.name, voc.name))
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
# The ONE acceptance boundary the strict entry points call (acceptance unification (b))
# ------------------------------------------------------------------------------------------------
def accept_pol(value, vocabulary: str = "sp", *, where: Optional[str] = None, param: str = "pol",
               extra: str = "", allowed: Optional[Sequence] = None):
    """Accept-and-normalize ONE polarization label at an entry point's boundary, or raise the
    shared vocabulary error.

    This is the single implementation of acceptance unification (b): it returns the CANONICAL
    value of ``value`` in ``vocabulary`` -- so an ``sp`` entry point handed ``'TE'`` gets back
    ``'s'`` and then runs the identical code path a ``pol='s'`` call runs, bit for bit -- and it
    admits ONLY the unconditional aliases declared in :data:`VOCABULARIES` (same physical mode in
    every geometry the vocabulary covers). A geometry-DEPENDENT spelling (lab ``'x'``/``'y'`` into
    an s/p API, a ``row`` index, ...) is still REFUSED here, with the explicit
    :func:`normalize_pol` call printed in the message.

    Callers keep their own cheap ``if value not in (canonical values)`` test in front of this, so
    the valid path neither imports this module nor pays a function call:

        if pol not in ("s", "p"):
            pol = _accept_pol(pol, "stack_rta")     # normalizes an alias, or raises

    ``allowed`` marks an entry point that implements only a SUBSET of its vocabulary (see
    :func:`pol_vocabulary_error`); the subset is tested AFTER normalization, so a subset entry
    point cannot be entered through an alias either."""
    voc = VOCABULARIES[vocabulary]
    try:
        canon = _canonical(value, voc)
    except PolarizationVocabularyError:
        raise pol_vocabulary_error(value, vocabulary, where=where, param=param, extra=extra,
                                   allowed=allowed) from None
    if allowed is not None and canon not in tuple(allowed):
        raise pol_vocabulary_error(value, vocabulary, where=where, param=param, extra=extra,
                                   allowed=allowed)
    return canon


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
            "not say what phi is. Pass the azimuth_deg argument explicitly -- e.g. "
            "normalize_pol({!r}, {!r}, to={!r}, azimuth_deg=0.0), where azimuth_deg=0.0 ASSERTS "
            "an in-plane mount rather than defaulting to one. At phi != 0 the label 'y' means the "
            "ROTATED s-hat to the FEM solver and OpticalSpec, but lab row 1 = E_y (a phi-dependent "
            "s/p MIXTURE) to the lumenairy bridge, which is why the bridge refuses conical "
            "incidence outright (audit C4-2).".format(value, frm, to))


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
            "the wavevector (OpticalSpec rejects it there). Pass the theta_deg argument to assert "
            "which branch you are on -- e.g. normalize_pol('x', 'lab_xyp', to='sp', "
            "azimuth_deg=0.0, theta_deg=0.0).")
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

# Deterministic pivot order for the multi-hop search below.  `axis_xy` is absent on purpose: it is
# not a spelling of the other four, so it is never a pivot and never a destination.
_PIVOT_ORDER = ("sp", "lab_xyp", "tetm", "row")


def _conversion_path(frm: str, to: str) -> Optional[List[Tuple[str, str]]]:
    """The shortest chain of DEFINED hops from ``frm`` to ``to``, or None.

    Only the six hops in ``_CONVERSIONS`` are primitive; every other convertible pair is a
    composition of them (``tetm -> row`` is te -> s -> 'y' -> 1, three hops). Composing rather
    than adding direct entries keeps ONE implementation of each physical correspondence, so every
    refusal on the way -- the azimuth demand, the conical refusal, the non-injective row 0 -- fires
    for the composed pairs too instead of being re-derived and forgotten."""
    if (frm, to) in _CONVERSIONS:
        return [(frm, to)]
    seen = {frm}
    queue: List[Tuple[str, List[Tuple[str, str]]]] = [(frm, [])]
    while queue:
        node, path = queue.pop(0)
        for nxt in _PIVOT_ORDER:
            if (node, nxt) not in _CONVERSIONS or nxt in seen:
                continue
            if nxt == to:
                return path + [(node, nxt)]
            seen.add(nxt)
            queue.append((nxt, path + [(node, nxt)]))
    return None


def _convert(canon, vocabulary: str, to: str, theta_deg, azimuth_deg):
    """Run the (possibly multi-hop) conversion of an ALREADY-CANONICAL value. Raises
    :class:`PolarizationConversionError` for anything ambiguous or undefined."""
    if "axis_xy" in (vocabulary, to):
        _refuse_axis(canon, vocabulary, to)
    path = _conversion_path(vocabulary, to)
    if path is None:
        raise _conversion_error(canon, vocabulary, to, "there is no defined mapping between them.")
    out = canon
    for frm_i, to_i in path:
        out = _CONVERSIONS[(frm_i, to_i)](out, theta_deg, azimuth_deg)
    return out


# ------------------------------------------------------------------------------------------------
# The explicit-conversion example every STRICT guard prints (part 2b)
# ------------------------------------------------------------------------------------------------
_CONTEXT_SYMBOL = {"azimuth_deg": "phi", "theta_deg": "theta"}

_WHY_YOU_SUPPLY_IT = {
    ("azimuth_deg",): ("YOU supply the azimuth because the mapping DEPENDS on it: 'y' is s-pol "
                       "only at phi = 0, and at phi != 0 the repo carries two incompatible "
                       "readings of it (rotated s-hat vs lab row 1)"),
    ("theta_deg",): ("YOU supply the polar angle because the mapping DEPENDS on it: the lab axes "
                     "coincide with s/p only where the stack is polarization-degenerate"),
    ("azimuth_deg", "theta_deg"): ("YOU supply BOTH angles because the mapping depends on both: "
                                   "'x' is transverse to the wavevector only at theta = 0, and "
                                   "the lab axes are the s/p basis only at phi = 0"),
}


def _required_context(value, frm: str, to: str) -> Tuple[str, ...]:
    """Which of ``azimuth_deg`` / ``theta_deg`` this particular conversion actually CONSULTS,
    measured by running it once with each omitted at the in-plane normal-incidence geometry.

    Probed rather than tabulated on purpose: the example printed in an error can then never drift
    away from what the converter really demands (``'y' -> 's'`` needs only the azimuth, ``'x' ->
    's'`` needs both, ``'te' -> 's'`` needs neither).

    A conversion that fails even WITH both angles is refused outright, not context-hungry (row 0
    -> a label), so it needs nothing and reports nothing -- the caller prints the refusal itself."""
    try:
        _convert(value, frm, to, 0.0, 0.0)
    except PolarizationConversionError:
        return ()
    except Exception:                                      # noqa: BLE001 -- probing only
        return ()
    needed = []
    for name in ("azimuth_deg", "theta_deg"):
        kw = {"azimuth_deg": 0.0, "theta_deg": 0.0}
        kw[name] = None
        try:
            _convert(value, frm, to, kw["theta_deg"], kw["azimuth_deg"])
        except PolarizationConversionError:
            needed.append(name)
        except Exception:                                  # noqa: BLE001 -- probing only
            pass
    return tuple(needed)


def conversion_example(value, frm: str, to: str) -> str:
    """The one-line "convert it yourself" sentence for a label of vocabulary ``frm`` that reached
    an entry point speaking ``to``. Written HERE, once, so all 19 strict guards say the same thing
    and say it correctly for the value in hand (see :func:`_required_context`)."""
    if "axis_xy" in (frm, to):
        return ("There is no conversion to offer: {!r} is a 2-D IN-PLANE cross-section geometry "
                "(hydro_fem) and {!r} is a planar stack -- different problems that happen to share "
                "the letters 'x' and 'y', not different spellings. {}".format(
                    "axis_xy", to if frm == "axis_xy" else frm, HOW_TO_CONVERT))
    try:
        canon = _canonical(value, VOCABULARIES[frm])
    except PolarizationVocabularyError:                    # pragma: no cover -- caller checked
        canon = value
    needed = _required_context(canon, frm, to)
    call = "normalize_pol({!r}, {!r}, to={!r}{})".format(
        value, frm, to, "".join(", {}={}".format(n, _CONTEXT_SYMBOL[n]) for n in needed))
    if not needed:
        try:
            _convert(canon, frm, to, 0.0, 0.0)
        except PolarizationConversionError as exc:
            why = str(exc).split(" -- ", 1)[-1].replace(MAP_REFERENCE, "").strip()
            return "normalize_pol REFUSES this conversion outright, at every geometry: {}".format(
                why)
        return "Convert explicitly: {} -- this pair is geometry-independent.".format(call)
    return "Convert explicitly: {} -- {}. normalize_pol REFUSES rather than pick a reading when "\
           "the geometry it needs is missing or makes the label ambiguous.".format(
               call, _WHY_YOU_SUPPLY_IT[tuple(sorted(needed))])


def normalize_pol(value, vocabulary: str = "sp", *, to: Optional[str] = None,
                  theta_deg: Optional[float] = None, azimuth_deg: Optional[float] = None,
                  where: Optional[str] = None, param: str = "pol"):
    """Validate ``value`` against ``vocabulary`` and, when ``to`` is given, convert it.

    With ``to=None`` (the default) this is a pure GUARD: it returns the canonical form of
    ``value`` in ``vocabulary`` and raises :class:`PolarizationVocabularyError` -- naming the
    sibling vocabulary the label came from -- otherwise. The accepted set is the vocabulary's own
    values plus its UNCONDITIONAL aliases (``'te'``/``'tm'`` and mixed case for ``sp``,
    ``'s'``/``'p'`` and mixed case for ``tetm``), exactly what the corresponding entry points
    accept -- this function is never a back door to a wider set than the modules take.

    With ``to=<other vocabulary>`` it converts. COVERAGE -- every ordered pair over the four
    stack vocabularies is defined, six of them primitively and the rest by composition
    (:func:`_conversion_path`), and each one REFUSES rather than guess when the geometry it
    depends on is missing:

      ============  ============  ==================================================================
      from          to            required context / refusal
      ============  ============  ==================================================================
      ``lab_xyp``   ``sp``        ``azimuth_deg`` always ('y' <-> 's' holds only at phi = 0, and at
                                  phi != 0 the repo carries two incompatible readings of 'y');
                                  ``'x'`` additionally needs ``theta_deg`` and is valid only at
                                  theta = 0 (sibling finding Q-24)
      ``sp``        ``lab_xyp``   ``azimuth_deg``; refused at phi != 0
      ``lab_xyp``   ``row``       ``azimuth_deg`` for ``'p'`` only (conical p-pol carries no single
                                  row); ``'x'``/``'y'`` are context-free
      ``row``       ``lab_xyp``   row 1 -> ``'y'``; row 0 is REFUSED outright
                                  (``{'x': 0, 'y': 1, 'p': 0}`` is not injective)
      ``sp``        ``tetm``      context-free at a classical mount; refused at phi != 0
      ``tetm``      ``sp``        context-free at a classical mount; refused at phi != 0
      ``sp``        ``row``       composed via ``lab_xyp`` (inherits the azimuth demand)
      ``row``       ``sp``        composed via ``lab_xyp`` (row 0 still refused)
      ``tetm``      ``lab_xyp``   composed via ``sp``
      ``lab_xyp``   ``tetm``      composed via ``sp``
      ``tetm``      ``row``       composed via ``sp`` -> ``lab_xyp`` (three hops)
      ``row``       ``tetm``      composed via ``lab_xyp`` -> ``sp`` (row 0 still refused)
      ``axis_xy``   anything      REFUSED, both directions: the 2-D in-plane hydro_fem geometry is
                                  a different problem, not a different spelling
      ============  ============  ==================================================================

    Note that ``theta_deg`` enters ONLY through the ``'x'`` leg: at phi = 0 the s-hat is the lab
    y-axis for every polar angle, so ``'y' <-> 's'`` is theta-independent and does not ask for it.

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
    return _convert(canon, vocabulary, to, theta_deg, azimuth_deg)
