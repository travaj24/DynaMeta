"""Shared plumbing for the lumenairy-bridge backends (audit 6.3 / section 8.2 step 2).

rcwa_backend historically doubled as the bridge's unofficial common module: the PMM and
Berreman backends imported its underscore helpers, and bor_backend copy-pasted the version
gate (drifting to its own floor). The shared surface now lives HERE under public names;
rcwa_backend keeps underscore aliases for back-compat.

ONE version floor (VERSION_FLOOR): the bridge is VERIFIED against lumenairy 5.22.x only --
every audited path (the C5-1 asymmetric-profile gates, C4-2 conical guards, per-layer
absorption, BOR, Berreman OOP, the D1 JAX-twin traced-source/uniform-eps surface, the A2
public RCWAStack.layers accessor) was exercised against the installed 5.22 source, and
nothing below it was ever tested. The old per-backend floors (5.14.2 / 5.14.4 / 5.16.0)
predate all of that and advertised support that was never demonstrated. parse_version
tolerates pre/post-release suffixes ('5.22.0rc1' -> (5, 22, 0)); the previous
tuple(int(p) ...) parse -- copy-pasted x3 -- crashed on them.
"""

from __future__ import annotations

import re
from typing import Tuple

import numpy as np

__all__ = ["VERSION_FLOOR", "parse_version", "require_lumenairy", "pol_row", "angles_rad",
           "guard_incidence_side", "guard_conical_ppol", "p_basis_conversion",
           "stack_layer_records", "conical_synthesis", "pol_tangential_unit"]

# The single bridge-wide floor (see module docstring). Bumping it is CORRECTNESS work:
# raise it to whatever version the validation gates were actually re-run against.
VERSION_FLOOR = (5, 22, 0)

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
            "slab order, conical guards, per-layer absorption, Berreman OOP-oblique, BOR, the "
            "D1 JAX-twin traced-source/uniform-eps surface, the A2 public RCWAStack.layers "
            "accessor) were validated against the 5.22 surface only -- older releases were "
            "never exercised. pip install -U lumenairy".format(
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
    """The per-layer records of a lumenairy RCWAStack (thickness/.kind/.data/.dispersive),
    in stack order (superstrate side first). lumenairy 5.22 added the public read-only
    `RCWAStack.layers` tuple property (AUDIT_DYNAMETA_CONSUMER_API_GAPS A2), so the bridge
    reads it directly -- no private-slot access, no version ceiling (the record shape is
    exercised by validation/lumenairy_translate.py)."""
    return list(stack.layers)


_CONICAL_PROP_TOL = 1e-8   # propagating-order selector (|Im kz| small, Re kz > tol), normalized by k0


def pol_tangential_unit(pol: str, phi: float) -> "np.ndarray":
    """Incident tangential (lab x, y) unit vector of the rotated s/p eigen-polarization the FEM
    solver and OpticalSpec use: 'y' = s-hat = (-sin phi, cos phi) (perpendicular to the plane of
    incidence); 'x'/'p' = the in-plane transverse direction (cos phi, sin phi). At phi = 0 these
    reduce to lab y and lab x, so the synthesis is continuous with the in-plane fast path."""
    if pol == "y":
        return np.array([-np.sin(phi), np.cos(phi)], dtype=float)
    return np.array([np.cos(phi), np.sin(phi)], dtype=float)          # 'x' or 'p'


def _zeroth_order_index(orders) -> int:
    """Row of the specular (all-zero) diffraction order, for a 1-D (N,) integer-order array or a
    2-D (N, 2) (m, n)-order array. Returns -1 if absent."""
    o = np.asarray(orders)
    mask = (o == 0) if o.ndim == 1 else np.all(o == 0, axis=1)
    hit = np.where(mask)[0]
    return int(hit[0]) if hit.size else -1


def conical_synthesis(amp_source, pol: str, theta: float, phi: float, n_super: complex):
    """Rotated s/p (R, T, r, t) for a CONICAL (azimuth != 0) solve, synthesized from the per-order
    complex amplitudes (audit 8.1-1 / consumer-gap B). ENGINE-AGNOSTIC: amp_source is any object
    exposing per_order_amplitudes(port) in the shared RCWA/PMM contract (dict of Ex/Ey (2, N)
    keyed to incident LAB E_x/E_y, kx/ky/kz normalized by k0, `orders`, wavelength) -- so a
    PMMStack, an RCWAResult (1-D or 2-D lattice), or a PMM2D result all feed it.

    At phi != 0 the physical s/p eigen-polarization is a SUPERPOSITION of the lab rows, so the
    total efficiency carries cross terms the per-order POWERS cannot provide. For the incident
    tangential unit u the per-order response is u . (row0, row1); an order's efficiency is
    (|Ex|^2 + |Ey|^2 + |Ez|^2) Re(kz/kz_inc) / |E_inc|^2 with the longitudinal Ez = -(kx Ex + ky
    Ey)/kz and |E_inc|^2 = 1 + |Ez_inc|^2 (a unit-tangential p wave carries its own Ez, so
    |E_inc,p|^2 = sec^2 theta -- the s-vs-p normalization the split hinges on). Summed over
    PROPAGATING orders in the (lossless) end medium -> R (reflection) and T (transmission, kz_inc
    still the superstrate value). r/t = the zeroth-order co-pol complex amplitudes: the specular
    order's FULL 3-D field projected onto the outgoing eigen-direction e_hat and normalized by
    sqrt(|E_inc|^2), so |r|^2 is the zeroth-order co-pol reflectance and the phase is the modulator
    observable. s: e_hat = (-sin phi, cos phi, 0) (tangential; Ez drops). p: e_hat = the outgoing
    p-hat = (k_out x s_hat)/|k_out|, which carries a z-component -- omitting Ez there undercounts
    |r| (probed 0.40 vs the true sqrt(R) = 0.46). All k's are k0-normalized, so scale-free."""
    u = pol_tangential_unit(pol, phi)
    ns = float(np.real(n_super))
    kz_inc = ns * np.cos(theta)                                       # normalized incident kz
    kx0i = ns * np.sin(theta) * np.cos(phi)
    ky0i = ns * np.sin(theta) * np.sin(phi)
    ez_inc = -(kx0i * u[0] + ky0i * u[1]) / kz_inc
    e_inc2 = float(u[0] ** 2 + u[1] ** 2 + abs(ez_inc) ** 2)          # 1 (s) or sec^2 theta (p)

    def _port_total(port):
        a = amp_source.per_order_amplitudes(port=port)
        kx, ky, kz = np.asarray(a["kx"]), np.asarray(a["ky"]), np.asarray(a["kz"])
        ex = u[0] * a["Ex"][0] + u[1] * a["Ex"][1]
        ey = u[0] * a["Ey"][0] + u[1] * a["Ey"][1]
        prop = (np.abs(kz.imag) < _CONICAL_PROP_TOL) & (kz.real > _CONICAL_PROP_TOL)
        ez = np.zeros_like(ex)
        nz = np.abs(kz) > 1e-300
        ez[nz] = -(kx[nz] * ex[nz] + ky[nz] * ey[nz]) / kz[nz]
        w = kz.real / kz_inc
        eff = (np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2) * w
        return float(np.sum(eff[prop])) / e_inc2, a, ex, ey

    R, ar, exr, eyr = _port_total("reflection")
    T, at, ext, eyt = _port_total("transmission")

    def _co_amp(a, ex, ey):
        i0 = _zeroth_order_index(a["orders"])
        if i0 < 0:
            return complex(0.0)
        kx0, ky0, kz0 = float(np.asarray(a["kx"])[i0].real), float(np.asarray(a["ky"])[i0].real), \
            np.asarray(a["kz"])[i0]
        exz, eyz = ex[i0], ey[i0]
        ez = -(kx0 * exz + ky0 * eyz) / kz0 if abs(kz0) > 1e-300 else 0.0
        kpar = np.hypot(kx0, ky0)
        if pol == "y" or kpar < 1e-12:                    # s-hat (tangential) or normal fallback
            eh = np.array([-np.sin(phi), np.cos(phi), 0.0])
        else:                                             # p-hat = (k_out x s_hat)/|k_out|
            shat = np.array([-ky0, kx0, 0.0]) / kpar
            eh = np.cross(np.array([kx0, ky0, float(kz0.real)]), shat)
            eh = eh / np.linalg.norm(eh)
        return complex((eh[0] * exz + eh[1] * eyz + eh[2] * ez) / np.sqrt(e_inc2))

    return R, T, _co_amp(ar, exr, eyr), _co_amp(at, ext, eyt)
