"""Shared plumbing for the lumenairy-bridge backends (audit 6.3 / section 8.2 step 2).

rcwa_backend historically doubled as the bridge's unofficial common module: the PMM and
Berreman backends imported its underscore helpers, and bor_backend copy-pasted the version
gate (drifting to its own floor). The shared surface now lives HERE under public names;
rcwa_backend keeps underscore aliases for back-compat.

ONE version floor (VERSION_FLOOR): the bridge is VERIFIED against lumenairy 5.21.x only --
every audited path (the C5-1 asymmetric-profile gates, C4-2 conical guards, per-layer
absorption, BOR, Berreman OOP) was exercised against the installed 5.21 source, and nothing
below it was ever tested. The old per-backend floors (5.14.2 / 5.14.4 / 5.16.0) predate all
of that and advertised support that was never demonstrated. parse_version tolerates
pre/post-release suffixes ('5.21.0rc1' -> (5, 21, 0)); the previous tuple(int(p) ...)
parse -- copy-pasted x3 -- crashed on them.
"""

from __future__ import annotations

import re
import warnings
from typing import Tuple

import numpy as np

__all__ = ["VERSION_FLOOR", "parse_version", "require_lumenairy", "pol_row", "angles_rad",
           "guard_incidence_side", "guard_conical_ppol", "p_basis_conversion",
           "stack_layer_records"]

# The single bridge-wide floor (see module docstring). Bumping it is CORRECTNESS work:
# raise it to whatever version the validation gates were actually re-run against.
VERSION_FLOOR = (5, 21, 0)

# The highest lumenairy MINOR the private-surface reads below (stack_layer_records) were
# verified against; a newer install only elicits a warning there (never a hard stop).
_TESTED_CEILING = (5, 21)

_POL_ROW = {"x": 0, "y": 1, "p": 0}


def parse_version(vstr) -> Tuple[int, int, int]:
    """(major, minor, patch) from a version string, tolerating rc/dev/post suffixes and
    short forms: '5.21.0rc1' -> (5, 21, 0), '5.22' -> (5, 22, 0). Each dot-part contributes
    its LEADING digits; the first part without any ends the parse (missing parts are 0)."""
    parts = []
    for p in str(vstr).split("."):
        m = re.match(r"\d+", p)
        if m is None:
            break
        parts.append(int(m.group()))
        if len(parts) == 3:
            break
    return tuple(parts + [0] * (3 - len(parts)))


def require_lumenairy():
    """Import lumenairy and enforce the single bridge floor. Every backend's lazy entry
    point calls this (bor_backend's former copy-paste gate included)."""
    try:
        import lumenairy
    except ImportError as exc:
        raise ImportError(
            "the Lumenairy backend needs lumenairy>={} (a REQUIRED dependency of "
            "dynameta -- this environment is missing it): pip install lumenairy".format(
                ".".join(str(v) for v in VERSION_FLOOR))) from exc
    if parse_version(lumenairy.__version__) < VERSION_FLOOR:
        raise ImportError(
            "lumenairy >= {} required (found {}); the bridge's audited fixes (graded-profile "
            "slab order, conical guards, per-layer absorption, Berreman OOP-oblique, BOR) "
            "were validated against the 5.21 surface only -- older releases were never "
            "exercised. pip install -U lumenairy".format(
                ".".join(str(v) for v in VERSION_FLOOR), lumenairy.__version__))
    return lumenairy


def pol_row(optical) -> int:
    pol = getattr(optical, "polarization", "y") or "y"
    if pol not in _POL_ROW:
        raise ValueError("lumenairy bridge: polarization must be 'x', 'y' or 'p' "
                         "(got {!r})".format(pol))
    return _POL_ROW[pol]


def angles_rad(optical) -> Tuple[float, float]:
    theta = float(np.radians(getattr(optical, "incidence_angle_deg", 0.0) or 0.0))
    phi = float(np.radians(getattr(optical, "azimuth_deg", 0.0) or 0.0))
    return theta, phi


def guard_incidence_side(optical) -> None:
    """The bridges build the stack superstrate-side first from the Design's super/substrate, so they
    model TOP incidence only (matching the FEM solver, solver.py). incidence_side='bottom' is a legal
    OpticalSpec value but on an asymmetric stack it physically differs (different incidence medium,
    reversed layer order); silently solving top would be a wrong number, so raise."""
    side = getattr(optical, "incidence_side", "top") or "top"
    if side != "top":
        raise NotImplementedError(
            "lumenairy bridge: incidence_side={!r} is not supported -- the bridges solve TOP "
            "incidence only. For bottom incidence, swap the superstrate/substrate materials in the "
            "Design (and reverse the layer order) and keep incidence_side='top'.".format(side))


def guard_conical_ppol(optical, phi_rad: float) -> None:
    """CONICAL incidence (azimuth != 0) is unsupported for EVERY polarization (audit C4-2):
    the bridges map 'x'/'y' to the lumenairy LAB rows, which at phi != 0 are phi-dependent
    s/p MIXTURES -- not the rotated s-hat = (-sin phi, cos phi) eigen-polarization the FEM
    implements and OpticalSpec documents (probe: theta=30, phi=45 on bare glass returned
    R=0.0392 vs the s-pol truth 0.0578, 32% low, silently; at phi=90 exactly the ORTHOGONAL
    polarization); the 'p' lab-basis conversion likewise assumes the x-z plane of incidence.
    (The previous guard covered only 'p', with a comment wrongly claiming 'x'/'y' are fine.)
    True conical s/p through the bridges needs per-order Jones synthesis -- a follow-on;
    until then refuse rather than return a wrong-polarization number."""
    if abs(float(phi_rad)) > 1e-12:
        pol = getattr(optical, "polarization", "y") or "y"
        raise NotImplementedError(
            "lumenairy bridge: conical incidence (azimuth != 0) is not supported for "
            "polarization {!r} -- the bridge's lab-basis rows are phi-dependent s/p mixtures, "
            "not the FEM's rotated s/p eigen-polarizations (audit C4-2). Set azimuth_deg=0 or "
            "use the FEM solver for conical incidence.".format(pol))


def p_basis_conversion(pol: str, theta_rad: float, n_super: complex,
                       n_sub: complex) -> Tuple[complex, complex]:
    """(r_factor, t_factor) mapping Lumenairy's LAB-BASIS Jones x-components onto the
    Byrnes-tmm p-hat-basis Fresnel amplitudes -- DynaMeta's incumbent convention (the FEM
    p-pol phases were validated against tmm). Measured: at 30 deg p-pol the Jones r_xx is
    EXACTLY -r_p(tmm) (R/T agree to machine precision, phase off by pi) and
    t_xx = t_p * cos(theta_t)/cos(theta_i) (the lab-x projection of the p-hat fields).
    s-pol ('y') and 'x' need no conversion (the bases coincide)."""
    if pol != "p":
        return 1.0, 1.0
    cos_i = np.sqrt(1.0 + 0j - np.sin(theta_rad) ** 2)
    sin_t = complex(n_super) * np.sin(theta_rad) / complex(n_sub)
    cos_t = np.sqrt(1.0 + 0j - sin_t ** 2)
    return -1.0, complex(cos_i / cos_t)


def stack_layer_records(stack):
    """The per-layer records of a lumenairy RCWAStack (thickness/.kind/.data/.dispersive).
    lumenairy exposes no public accessor (checked through 5.21.3), so this is the ONE place
    the bridge reads the private `_layers` slot -- with a version ceiling: a `layers`
    attribute is preferred if a future release adds one, and reading the private slot on a
    lumenairy NEWER than the tested _TESTED_CEILING line warns (the record shape is pinned
    by validation/lumenairy_translate.py, not by upstream contract)."""
    pub = getattr(stack, "layers", None)
    if pub is not None:
        return list(pub)
    import lumenairy
    ver = parse_version(lumenairy.__version__)
    if (ver[0], ver[1]) > _TESTED_CEILING:
        warnings.warn(
            "lumenairy bridge: reading RCWAStack._layers (no public accessor) on lumenairy "
            "{} -- newer than the {}.{}.x line this private surface was verified against; "
            "if translation output looks wrong, re-run validation/lumenairy_translate.py "
            "and update the bridge.".format(
                lumenairy.__version__, _TESTED_CEILING[0], _TESTED_CEILING[1]),
            RuntimeWarning, stacklevel=2)
    return list(getattr(stack, "_layers"))
