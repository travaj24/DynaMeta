"""Far-field polarization vortices and topological charge (roadmap item 1.4).

Bound states in the continuum (BICs) of a periodic photonic slab are the centres of
POLARIZATION VORTICES on the in-plane wavevector (k_par = (kx, ky)) plane: the far-field
radiation of the leaky resonance carries a well-defined linear polarization at every k EXCEPT
at the BIC, where the radiation vanishes and the polarization direction is undefined. Winding
the polarization angle around a closed loop that encircles the BIC returns an integer (or, for
generic C-points, half-integer) topological charge

    q = (1 / 2 pi) * closed_integral d phi ,

with phi the orientation (major-axis) angle of the far-field polarization ellipse. BICs are the
V-points (the field itself vanishes) of this director field and are protected by the quantized
charge (Zhen, Hsu, Lu, Stone, Soljacic, "Topological Nature of Optical Bound States in the
Continuum", Phys. Rev. Lett. 113, 257401 (2014)).

This module is PURE POST-PROCESSING: it consumes a field of zeroth-order far-field Jones
vectors J(kx, ky) = (Ex, Ey) (complex, lab basis) sampled on a (kx, ky) grid -- however the
caller obtained them -- and returns the polarization-angle field, the winding/charge around a
contour, a candidate-singularity finder, and a small-loop charge map. It carries no units
(everything is dimensionless in kx/ky and the Jones amplitudes) and no simulation state, so it
is independent of any solver. SI / exp(-i omega t) conventions govern the Jones vectors that
feed it; the orientation angle and the charge use only S1, S2 (they are invariant to the
handedness sign that the time convention fixes on S3), so the results are convention-robust.

------------------------------------------------------------------------------------------------
THE mod-pi TOPOLOGY (why we double the angle before winding)
------------------------------------------------------------------------------------------------
A linear polarization is a HEADLESS director: phi and phi + pi denote the same physical
orientation, so phi lives on the projective line RP1, not the circle S1. A naive winding of phi
is ill-defined (it depends on the arbitrary branch chosen at each sample). The fix is to wind the
DOUBLED angle 2 phi, which IS a genuine S1-valued (mod 2 pi) quantity:

    N = (1 / 2 pi) * closed_integral d(2 phi)     (an integer -- the winding of the S1 field 2 phi)
    q = N / 2                                      (the STANDARD charge, quantized in HALF-integers)

A director that rotates by pi over the loop (one full RP1 turn) has 2 phi rotate by 2 pi, i.e.
N = 1 and q = 1/2. Symmetry-protected BICs of a C2/C4v slab carry the integer charges q = +-1,
+-2 (2 phi winds by +-2, +-4 * 2 pi). We ALWAYS unwrap 2 phi (never phi) along the contour, which
is what makes the mod-pi identification exact; topological_charge() then reports q = N / 2 as a
signed multiple of 1/2.

Sign/orientation convention: phi_field[i, j] must be indexed so axis 0 and axis 1 increase with
the two in-plane momentum coordinates (kx along axis 0, ky along axis 1); a rectangle contour is
traversed COUNTER-CLOCKWISE in that (kx, ky) plane, so a field phi = +atan2(ky, kx) yields
q = +1. Reverse either axis mapping and the sign flips (as it must).

------------------------------------------------------------------------------------------------
BRIDGE HOOKUP (lumenairy conical RCWA -> this module) -- worked sketch
------------------------------------------------------------------------------------------------
The zeroth-order far-field Jones field comes from a conical (kx, ky) sweep of the RCWA bridge.
The conical surface this depends on is lumenairy's RCWAStack.jones_reflection() (the (2, 2)
zeroth-order lab-basis Jones, columns keyed to incident Ex / Ey) and, if a specific incident
polarization or the resonant radiation eigenvector is wanted, res.per_order_amplitudes(port=...)
(the shared contract the dynameta bridge already builds on -- see
dynameta.optics.lumenairy_bridge._common.conical_synthesis). A minimal end-to-end sketch::

    import numpy as np
    from lumenairy import RCWAStack           # or dynameta...design_to_rcwa_stack(design, lam)

    lam = 1.30e-6
    n_super = 1.0
    k0 = 2.0 * np.pi / lam
    # symmetry-protected BIC lives at the Gamma point (kx = ky = 0); sample a small k-window
    kxs = np.linspace(-0.02, 0.02, 41) * k0
    kys = np.linspace(-0.02, 0.02, 41) * k0
    J = np.zeros((kxs.size, kys.size, 2), dtype=complex)
    for i, kx in enumerate(kxs):
        for j, ky in enumerate(kys):
            kpar = np.hypot(kx, ky)
            theta = np.arcsin(np.clip(kpar / (k0 * n_super), -1.0, 1.0))
            phi_inc = np.arctan2(ky, kx)
            stack = build_2d_slab_stack()          # a 2-D-lattice RCWAStack (the BIC design)
            stack.set_source(lam, theta=theta, phi=phi_inc)
            res = stack.solve()
            Jr = np.asarray(res.jones_reflection())  # (2, 2), columns = incident Ex / Ey
            # far-field polarization of the leaky resonance: the eigenvector of Jr with the
            # LARGEST |eigenvalue| (the resonantly-radiating channel); a fixed incident column
            # Jr[:, 0] also works away from accidental degeneracies.
            w, V = np.linalg.eig(Jr)
            J[i, j, :] = V[:, int(np.argmax(np.abs(w)))]

    from dynameta.optics.bic import (polarization_angle_field, find_vortex_candidates,
                                     topological_charge, charge_map)
    phi = polarization_angle_field(J)
    cands = find_vortex_candidates(J)                 # C-point / V-point singularities
    for c in cands:
        i, j = c["index"]
        q = topological_charge(phi, (i - 4, i + 4, j - 4, j + 4))   # loop around the candidate
        print("BIC candidate at k-cell", (i, j), "charge", q)
    qmap = charge_map(phi, radius=2)                   # per-loop local charge map

Recovering a REAL symmetry-protected BIC vortex from a designed photonic-crystal slab (a high-Q
C4v/C2 cell, mode isolation, a fine Gamma-point k-window) is heavier than these synthetic gates
warrant and is left as the documented stretch goal; the synthetic gates below pin the algorithm
exactly, and the guarded integration test exercises this bridge Jones surface end to end.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "stokes_parameters",
    "polarization_angle_field",
    "rectangle_contour",
    "contour_winding",
    "topological_charge",
    "find_vortex_candidates",
    "charge_map",
]

_TWO_PI = 2.0 * np.pi


# ------------------------------------------------------------------------------------------------
# Jones -> Stokes / orientation angle
# ------------------------------------------------------------------------------------------------
def _split_jones(jones) -> Tuple[np.ndarray, np.ndarray]:
    """(Ex, Ey) complex components from a Jones field with the polarization on the LAST axis
    (shape (..., 2)). Returns two arrays of shape jones.shape[:-1]."""
    j = np.asarray(jones)
    if j.shape[-1] != 2:
        raise ValueError("bic: Jones field must have shape (..., 2) with the (Ex, Ey) "
                         "components on the last axis (got last-axis size {})".format(j.shape[-1]))
    j = j.astype(complex, copy=False)
    return j[..., 0], j[..., 1]


def stokes_parameters(jones) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(S0, S1, S2, S3) Stokes fields of a Jones field (shape (..., 2), (Ex, Ey) last).

        S0 = |Ex|^2 + |Ey|^2      (total intensity)
        S1 = |Ex|^2 - |Ey|^2      (0/90 linear preference)
        S2 = 2 Re(Ex conj(Ey))    (+/-45 linear preference)
        S3 = 2 Im(Ex conj(Ey))    (circular / handedness)

    The ellipse ORIENTATION uses only (S1, S2); S3 fixes the handedness and its sign follows the
    exp(-i omega t) convention, but it does NOT enter phi or the charge. The linear-polarization
    magnitude L = sqrt(S1^2 + S2^2) = sqrt(S0^2 - S3^2) (fully polarized) vanishes at BOTH kinds
    of polarization singularity -- V-points (S0 -> 0, the radiation dies: the BIC) and C-points
    (S1 = S2 = 0 at S3 != 0: circular polarization, orientation undefined) -- which is why
    find_vortex_candidates keys on L."""
    ex, ey = _split_jones(jones)
    axy = ex * np.conj(ey)
    s0 = (np.abs(ex) ** 2 + np.abs(ey) ** 2).real
    s1 = (np.abs(ex) ** 2 - np.abs(ey) ** 2).real
    s2 = 2.0 * axy.real
    s3 = 2.0 * axy.imag
    return s0, s1, s2, s3


def polarization_angle_field(jones) -> np.ndarray:
    """Orientation (major-axis) angle phi of the far-field polarization ellipse, from a field of
    Jones vectors (shape (..., 2), (Ex, Ey) on the last axis). Returns phi in radians in the
    half-open interval (-pi/2, pi/2] -- i.e. MOD PI, because a linear polarization is a headless
    director (phi == phi + pi). This is the standard Stokes orientation

        phi = 1/2 * atan2(S2, S1),   S1 = |Ex|^2 - |Ey|^2,  S2 = 2 Re(Ex conj(Ey))

    (Zhen et al. PRL 113, 257401 (2014), Eq. for the far-field polarization vector). For a
    linearly polarized Jones vector (cos a, sin a) this returns a mod pi exactly, so a synthetic
    field built from phi = q * atan2(ky, kx) round-trips. Do NOT wind this angle directly -- feed
    it to topological_charge / contour_winding, which wind the doubled angle 2 phi (see the module
    docstring on the mod-pi topology)."""
    _, s1, s2, _ = stokes_parameters(jones)
    return 0.5 * np.arctan2(s2, s1)


# ------------------------------------------------------------------------------------------------
# Contours and winding
# ------------------------------------------------------------------------------------------------
def rectangle_contour(i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
    """Ordered (P, 2) integer index array tracing the boundary of the index rectangle
    [i0..i1] x [j0..j1] (inclusive corners) COUNTER-CLOCKWISE in the (axis0, axis1) = (kx, ky)
    plane, WITHOUT repeating the closing vertex (the winding routines close the loop themselves).
    The four edges are bottom (+i), right (+j), top (-i), left (-j)."""
    i0, i1, j0, j1 = int(i0), int(i1), int(j0), int(j1)
    if i1 <= i0 or j1 <= j0:
        raise ValueError("rectangle_contour: need i0 < i1 and j0 < j1 (got i in [{}, {}], "
                         "j in [{}, {}])".format(i0, i1, j0, j1))
    bottom = [(i, j0) for i in range(i0, i1)]          # (i0,j0) -> (i1,j0)  exclusive of corner
    right = [(i1, j) for j in range(j0, j1)]           # (i1,j0) -> (i1,j1)
    top = [(i, j1) for i in range(i1, i0, -1)]         # (i1,j1) -> (i0,j1)
    left = [(i0, j) for j in range(j1, j0, -1)]        # (i0,j1) -> (i0,j0)
    return np.array(bottom + right + top + left, dtype=int)


def _as_contour(contour, shape) -> np.ndarray:
    """Normalize a contour spec to an ordered (P, 2) index array. Accepts either a 4-tuple
    rectangle spec (i0, i1, j0, j1) or an explicit (P, 2) ordered index array (auto-closed by the
    winding routines). Validates that indices lie inside `shape`."""
    arr = np.asarray(contour)
    if arr.shape == (4,) and arr.dtype.kind in "iu":
        idx = rectangle_contour(*[int(v) for v in arr])
    elif arr.ndim == 1 and arr.size == 4 and not hasattr(contour, "shape"):
        idx = rectangle_contour(*[int(v) for v in contour])
    elif arr.ndim == 2 and arr.shape[1] == 2:
        idx = arr.astype(int)
    else:
        # last resort: a plain length-4 sequence of scalars is a rectangle spec
        seq = list(contour)
        if len(seq) == 4 and all(np.isscalar(v) for v in seq):
            idx = rectangle_contour(*[int(v) for v in seq])
        else:
            raise ValueError("bic: contour must be a rectangle spec (i0, i1, j0, j1) or an "
                             "ordered (P, 2) index array")
    nx, ny = shape
    if idx[:, 0].min() < 0 or idx[:, 0].max() >= nx or idx[:, 1].min() < 0 or idx[:, 1].max() >= ny:
        raise ValueError("bic: contour indices fall outside the phi field of shape "
                         "{} (contour i in [{}, {}], j in [{}, {}])".format(
                             shape, idx[:, 0].min(), idx[:, 0].max(),
                             idx[:, 1].min(), idx[:, 1].max()))
    return idx


def _winding_of_angle(theta_closed: np.ndarray) -> float:
    """(1 / 2 pi) * total change of an S1-valued angle sequence around a CLOSED loop, via robust
    wrapped-difference unwrapping (each adjacent step wrapped into (-pi, pi], plus the closing
    step from the last sample back to the first). Returns a real number that is an integer up to
    the discretization; the caller rounds. Valid provided no adjacent step exceeds pi in
    magnitude (the sampling-density precondition -- see charge_map's Nyquist note)."""
    theta = np.asarray(theta_closed, dtype=float)
    d = np.diff(theta)
    d = (d + np.pi) % _TWO_PI - np.pi
    close = (theta[0] - theta[-1] + np.pi) % _TWO_PI - np.pi
    return float((np.sum(d) + close) / _TWO_PI)


def contour_winding(phi_field, contour) -> float:
    """RAW (unrounded) winding N = (1 / 2 pi) * closed_integral d(2 phi) of the DOUBLED angle
    around `contour` -- the winding of the genuine S1 field 2 phi (see module docstring). N is an
    integer up to discretization noise; the topological charge is q = N / 2. Exposed for
    diagnostics (e.g. asserting the field is cleanly topological before rounding)."""
    phi = np.asarray(phi_field, dtype=float)
    if phi.ndim != 2:
        raise ValueError("bic: phi_field must be a 2-D (kx, ky) array (got ndim {})".format(phi.ndim))
    idx = _as_contour(contour, phi.shape)
    theta = 2.0 * phi[idx[:, 0], idx[:, 1]]
    return _winding_of_angle(theta)


def topological_charge(phi_field, contour) -> float:
    """Topological charge q = (1 / 2 pi) * closed_integral d phi enclosed by `contour`, computed
    from the winding of the DOUBLED angle 2 phi and reported in the STANDARD convention:

        N = (1 / 2 pi) * closed_integral d(2 phi)   (integer; the S1 winding of 2 phi)
        q = N / 2                                    (signed multiple of 1/2)

    Integer q for symmetry-protected BIC/V-point vortices (phi winds by q * 2 pi -> 2 phi winds by
    2 q -> N = 2 q); half-integer q = +-1/2 for generic C-points (phi winds by pi -> N = +-1).
    `contour` is a rectangle spec (i0, i1, j0, j1) (inclusive corners, traversed CCW) or an ordered
    (P, 2) index array. phi_field is the (kx, ky) orientation field from polarization_angle_field
    (axis 0 = kx, axis 1 = ky). The doubled winding is an integer, so we round N to the nearest
    integer before halving -- this makes q exact and topologically robust (small Jones noise cannot
    move it) as long as the contour samples 2 phi densely enough (no adjacent step > pi)."""
    n_raw = contour_winding(phi_field, contour)
    return float(round(n_raw)) / 2.0


# ------------------------------------------------------------------------------------------------
# Singularity (vortex-centre) detection
# ------------------------------------------------------------------------------------------------
def find_vortex_candidates(field, *, rel_threshold: float = 0.5, exclude_border: int = 1,
                           neighborhood: int = 1, contour_radius: int = 2,
                           max_candidates: Optional[int] = None) -> List[Dict[str, object]]:
    """Locate polarization-singularity candidates (BIC V-points and C-points) on a (kx, ky) grid.

    `field` is EITHER a Jones field (shape (Nx, Ny, 2), (Ex, Ey) last -- the preferred input) OR a
    precomputed orientation field phi (shape (Nx, Ny), real). With a Jones field the detector keys
    on the linear-polarization magnitude

        L = sqrt(S1^2 + S2^2)

    which vanishes at BOTH singularity types: V-points (S0 -> 0, radiation dies -- the BIC) and
    C-points (S1 = S2 = 0 at finite S3 -- circular polarization, orientation undefined). A grid
    cell is a candidate when L is a strict local minimum over its (2*neighborhood + 1) window AND
    is pronounced (L < rel_threshold * median(L)). With only a phi field (no amplitudes) the
    detector falls back to a plaquette-winding scan: cells whose smallest enclosing loop carries a
    nonzero charge.

    Returns a list of dicts sorted by ascending indicator (strongest singularity first), each with
    keys: 'index' (i, j), 'indicator' (L, or |charge| in the phi fallback), and 'charge' (the
    local charge on a rectangle contour of half-size contour_radius, or None if the contour does
    not fit inside the grid). C-point detection localizes a synthetic vortex centre to its grid
    cell (the unique L minimum)."""
    arr = np.asarray(field)
    is_jones = arr.ndim == 3 and arr.shape[-1] == 2
    if is_jones:
        _, s1, s2, _ = stokes_parameters(arr)
        indicator = np.sqrt(s1 ** 2 + s2 ** 2)            # L: zero at V-points AND C-points
        phi = 0.5 * np.arctan2(s2, s1)
        cand = _strict_local_minima(indicator, neighborhood, exclude_border, rel_threshold)
        cand_scores = {ij: float(indicator[ij]) for ij in cand}
    elif arr.ndim == 2:
        # phi-only fallback: no amplitudes, so key on the plaquette winding instead of L
        phi = arr.astype(float)
        qmap = charge_map(phi, radius=1)
        cand = [(i, j) for i in range(phi.shape[0]) for j in range(phi.shape[1])
                if np.isfinite(qmap[i, j]) and abs(qmap[i, j]) > 1e-9]
        cand_scores = {ij: -abs(float(qmap[ij])) for ij in cand}     # strongest |q| first
    else:
        raise ValueError("bic: find_vortex_candidates needs a Jones field (Nx, Ny, 2) or an "
                         "orientation field (Nx, Ny) (got shape {})".format(arr.shape))

    nx, ny = phi.shape
    r = int(contour_radius)
    out: List[Dict[str, object]] = []
    for (i, j) in cand:
        charge = None
        if i - r >= 0 and i + r < nx and j - r >= 0 and j + r < ny:
            charge = topological_charge(phi, (i - r, i + r, j - r, j + r))
        out.append({"index": (int(i), int(j)), "indicator": cand_scores[(i, j)], "charge": charge})
    out.sort(key=lambda d: d["indicator"])
    if max_candidates is not None:
        out = out[:int(max_candidates)]
    return out


def _strict_local_minima(field: np.ndarray, neighborhood: int, exclude_border: int,
                         rel_threshold: float) -> List[Tuple[int, int]]:
    """Cells that are a strict minimum of `field` over their (2*neighborhood + 1) window and lie
    below rel_threshold * median(field) (a pronounced dip, not a flat plateau)."""
    nx, ny = field.shape
    med = float(np.median(field))
    thr = rel_threshold * med if med > 0 else np.inf
    r = int(neighborhood)
    b = int(exclude_border)
    hits: List[Tuple[int, int]] = []
    for i in range(max(b, r), nx - max(b, r)):
        for j in range(max(b, r), ny - max(b, r)):
            v = field[i, j]
            if v >= thr:
                continue
            win = field[i - r:i + r + 1, j - r:j + r + 1]
            if v <= win.min() and np.count_nonzero(win <= v) == 1:
                hits.append((i, j))
    return hits


def charge_map(field, radius: int = 2) -> np.ndarray:
    """Local topological charge on a small square loop centred at each interior grid vertex,
    returned as an (Nx, Ny) array (np.nan where the loop does not fit inside the grid).

    `field` is a Jones field (Nx, Ny, 2) or an orientation field phi (Nx, Ny). Each interior
    vertex (i, j) gets the charge on the rectangle contour [i-radius, i+radius] x [j-radius,
    j+radius]; a vortex therefore lights up the block of vertices whose loop encloses it (the
    block is centred on the vortex). NYQUIST NOTE: a loop of half-size R has P = 8 R boundary
    samples and resolves |q| <= (P/2 - 1)/2 = 2 R - 1/2 without aliasing, so choose
    R >= (2 |q| + 1) / 4; radius = 2 (16 points, |q| <= 3) is the robust default and resolves the
    q = +-2 charges. Use radius = 1 for the sharpest localization when |q| <= 1."""
    arr = np.asarray(field)
    phi = polarization_angle_field(arr) if (arr.ndim == 3 and arr.shape[-1] == 2) else arr.astype(float)
    if phi.ndim != 2:
        raise ValueError("bic: charge_map needs a Jones field (Nx, Ny, 2) or phi (Nx, Ny)")
    nx, ny = phi.shape
    r = int(radius)
    out = np.full((nx, ny), np.nan)
    for i in range(r, nx - r):
        for j in range(r, ny - r):
            out[i, j] = topological_charge(phi, (i - r, i + r, j - r, j + r))
    return out
