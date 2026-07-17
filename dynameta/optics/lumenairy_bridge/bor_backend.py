"""Lumenairy BOR-PMM (axisymmetric / body-of-revolution PMM) as a DynaMeta optical backend.

BOR-PMM (lumenairy.elements.bor, graduated + released in lumenairy 5.16.0) is the CYLINDRICAL-
coordinate peer of the Cartesian RCWA/PMM solvers: for a structure invariant under rotation about
an axis (a concentric-ring grating, a fiber, an axisymmetric diffractive element) the fields separate
as exp(i m phi + i q z) and the problem reduces to a 1-D RADIAL eigenproblem at each azimuthal order
m, cascaded in z by a Redheffer S-matrix. It is a DIFFERENT symmetry class from the Cartesian
LayeredStackSolver bridges (rcwa/pmm/berreman), so it does NOT plug into that seam -- it carries its
own axisymmetric stack spec (BorStackSpec) and returns per-INCIDENT-MODE R/T (each incident cylindrical
mode is a wave at a discrete polar angle set by the computational radius).

Conventions are IDENTICAL on both sides (exp(-i omega t), Im(eps) > 0 for absorbers, radians); DynaMeta
is SI (metres) at the API while lumenairy is unit-agnostic, so the bridge scales lengths to MICRONS
internally (k0 * length is dimensionless, so this is exact; microns keep the eigensolver in lumenairy's
validated numeric regime). The bridge is a geometry/result adapter, not a translation layer.

Use:
    from dynameta.optics.lumenairy_bridge import BorStackSpec, BorLayer, solve_bor
    spec = BorStackSpec(layers=[BorLayer(thickness_m=0.5e-6,
                                         rings=(3.0e-6, 0.5, 2.45, 1.41))],   # (period, duty, n_r, n_g)
                        azimuthal_order_m=1, r_max_m=48e-6, n_super=1.41, n_sub=1.41)
    res = solve_bor(spec, lambda_m=1.0e-6)     # res.angles / res.R / res.T per incident mode
    opt = res.fundamental_result()             # the near-axis fundamental mode as an OpticalResult
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from dynameta.core.interfaces import OpticalResult

_SCALE = 1.0e6   # m -> um (lumenairy is unit-agnostic; microns match BOR's validated numeric regime)


def _require_bor():
    """lumenairy.elements.bor.BORStack -- the axisymmetric BOR-PMM tier (graduated in 5.16.0,
    covered by the single bridge floor in _common.VERSION_FLOOR, which replaced this backend's
    copy-pasted version gate). Imported lazily so the base dynameta import stays light."""
    from dynameta.optics.lumenairy_bridge._common import require_lumenairy
    require_lumenairy()
    from lumenairy.elements.bor import BORStack
    return BORStack


@dataclass
class BorLayer:
    """One z-layer of an axisymmetric stack (SI metres). EXACTLY one of: `rings` (a concentric BINARY
    ring grating (radial_period_m, duty, n_ridge, n_groove)); `eps_profile` (a callable r_m -> eps, a
    radial permittivity profile, r in METRES); or `eps` (a uniform scalar permittivity). exp(-i omega t),
    so Im(eps) > 0 is loss."""
    thickness_m: float
    rings: Optional[Tuple[float, float, complex, complex]] = None
    eps_profile: Optional[Callable[[np.ndarray], np.ndarray]] = None
    eps: Optional[complex] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        given = [self.rings is not None, self.eps_profile is not None, self.eps is not None]
        if sum(given) != 1:
            raise ValueError("BorLayer needs EXACTLY one of rings / eps_profile / eps")
        if not (self.thickness_m > 0.0):
            raise ValueError("BorLayer: thickness_m must be > 0")
        if self.rings is not None and len(self.rings) != 4:
            raise ValueError("BorLayer.rings = (radial_period_m, duty, n_ridge, n_groove)")


@dataclass
class BorStackSpec:
    """An axisymmetric (body-of-revolution) stack: a z-list of BorLayers between semi-infinite super/
    substrate, solved at azimuthal order m on a radial domain [0, r_max_m] with n_radial points. r_max_m
    (the computational radius) sets the discrete incident-mode set (each mode = a cylindrical wave at a
    quantized polar angle); take it >> the radial feature period so the rings are locally planar. SI."""
    layers: List[BorLayer]
    azimuthal_order_m: int
    r_max_m: float
    n_radial: int = 256
    n_super: complex = 1.0 + 0j
    n_sub: complex = 1.0 + 0j

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("BorStackSpec needs at least one layer")
        if int(self.azimuthal_order_m) < 0:
            raise ValueError("azimuthal_order_m must be >= 0")
        if not (self.r_max_m > 0.0):
            raise ValueError("r_max_m must be > 0")
        if int(self.n_radial) < 16:
            raise ValueError("n_radial must be >= 16")


@dataclass
class BorResult:
    """Per-INCIDENT-MODE BOR solve result. Each incident cylindrical mode j is a wave at polar angle
    angles[j] (rad, in the superstrate); R[j]/T[j] are its total reflected/transmitted power fractions
    (summed over diffracted orders), energy[j] = R[j] + T[j] (1 for a lossless stack). The modes are
    ordered by increasing angle, so index 0 is the FUNDAMENTAL (near-axis ~ normal incidence)."""
    angles_rad: np.ndarray
    R: np.ndarray
    T: np.ndarray
    energy: np.ndarray
    solve_time_s: float
    raw: dict = field(default_factory=dict)
    # Physically-phased per-incident-mode SPECULAR (0-order, mode -> same mode) amplitudes, in the
    # PINNED deterministic gauge of BORStack.per_mode_amplitudes ('dominant field sample real-positive'),
    # aligned to angles_rad (index 0 = fundamental). r_complex[j] is the reflected specular amplitude of
    # incident mode j (|r_complex[j]|**2 == R[j] only for a NON-diffractive stack -- for a diffractor the
    # order-summed R[j] >= the specular |r|**2, so R stays the energy field and r carries the phase).
    # t_complex[j] is the transmitted specular amplitude into the SAME transverse mode in the substrate,
    # or NaN when that channel does not propagate (fall back to the sqrt(T) magnitude, phase 0). None on
    # the empty-mode / JAX paths. (AUDIT C1)
    r_complex: Optional[np.ndarray] = None
    t_complex: Optional[np.ndarray] = None
    # Per-layer absorbed fraction (n_layers, n_inc) from BORStack.layer_absorption(), columns aligned to
    # angles_rad; layer_names labels the rows in BorStackSpec.layers order. Populated only when solve_bor
    # was called with absorption=True (else None). R + T + sum_layers A = 1 per incident mode. (AUDIT B4b)
    layer_absorption: Optional[np.ndarray] = None
    layer_names: Optional[List[str]] = None

    def fundamental_result(self) -> OpticalResult:
        """The fundamental (smallest-angle, near-normal) incident mode as an OpticalResult.

        `r` is the pinned-gauge COMPLEX specular reflection amplitude (physical phase; magnitude
        sqrt of the 0-order specular reflectance) and phase_deg = degrees(angle(r)); `R` remains the
        order-SUMMED reflected power fraction (so |r|**2 <= R, with equality for a non-diffractive
        stack). `t` likewise carries the transmitted specular phase when that substrate channel
        propagates, else falls back to the sqrt(T) magnitude with phase 0. When per-layer absorption
        was retained (absorption=True), A_independent = sum_layers A for this mode and
        per_region_absorption maps each layer name -> its absorbed fraction (R + T + A_independent = 1
        to machine precision); otherwise both are None and A is the 1 - R - T closure."""
        if self.angles_rad.size == 0:
            raise ValueError("BorResult has no propagating incident mode (raise r_max_m / k0)")
        i0 = int(np.argmin(self.angles_rad))
        R = float(self.R[i0])
        T = float(self.T[i0])
        if self.r_complex is not None and np.isfinite(self.r_complex[i0]):
            r = complex(self.r_complex[i0])
            phase_deg = float(np.degrees(np.angle(r)))
        else:
            r = complex(np.sqrt(max(R, 0.0)), 0.0)
            phase_deg = 0.0
        if self.t_complex is not None and np.isfinite(self.t_complex[i0]):
            t = complex(self.t_complex[i0])
        else:
            t = complex(np.sqrt(max(T, 0.0)), 0.0)
        A_ind: Optional[float] = None
        per_region: Optional[dict] = None
        if self.layer_absorption is not None and self.layer_names is not None and self.layer_absorption.size:
            col = np.asarray(self.layer_absorption)[:, i0]
            A_ind = float(np.sum(col))
            per_region = {nm: float(col[k]) for k, nm in enumerate(self.layer_names)}
        return OpticalResult(r=r, R=R, phase_deg=phase_deg, solve_time_s=self.solve_time_s,
                             t=t, T=T, A=float(1.0 - R - T), A_independent=A_ind,
                             per_region_absorption=per_region)


def _scaled_stack(spec: BorStackSpec, lambda_m: float):
    """Build the lumenairy BORStack with lengths scaled m -> um and k0 set from lambda_m (scale-exact:
    k0 * length is dimensionless). Returns the configured, source-set BORStack ready to .solve()."""
    BORStack = _require_bor()
    k0_um = 2.0 * np.pi / (float(lambda_m) * _SCALE)
    s = BORStack(Rbig=float(spec.r_max_m) * _SCALE, m=int(spec.azimuthal_order_m),
                 N=int(spec.n_radial), n_superstrate=complex(spec.n_super), n_substrate=complex(spec.n_sub))
    for L in spec.layers:
        thk_um = float(L.thickness_m) * _SCALE
        if L.rings is not None:
            period_m, duty, n_r, n_g = L.rings
            s.add_layer(thk_um, rings=(float(period_m) * _SCALE, float(duty), complex(n_r), complex(n_g)))
        elif L.eps_profile is not None:
            prof = L.eps_profile
            s.add_layer(thk_um, eps_profile=lambda r_um, _p=prof: np.asarray(_p(np.asarray(r_um) / _SCALE),
                                                                             dtype=complex))
        else:
            s.add_layer(thk_um, eps=complex(L.eps))
    s.set_source(k0=k0_um)
    return s


def _layer_names(spec: BorStackSpec) -> List[str]:
    """Deterministic, unique per-layer labels in BorStackSpec.layers order (the SAME order the stack
    added them, so they line up row-for-row with BORStack.layer_absorption()). Uses BorLayer.name when
    given, else "layer{i}"; a duplicate is disambiguated with a "#{i}" suffix."""
    names: List[str] = []
    seen: set = set()
    for i, L in enumerate(spec.layers):
        nm = (getattr(L, "name", None) or "layer{}".format(i))
        if nm in seen:
            nm = "{}#{}".format(nm, i)
        seen.add(nm)
        names.append(nm)
    return names


def solve_bor(spec: BorStackSpec, lambda_m: float, *, absorption: bool = False) -> BorResult:
    """Solve an axisymmetric BorStackSpec at wavelength lambda_m (metres) -> BorResult (per-incident-mode
    R/T/angle). The incident modes are sorted by increasing polar angle (index 0 = near-axis fundamental).

    The pinned-gauge COMPLEX specular reflection/transmission amplitudes (physical phase) are ALWAYS
    threaded through (r_complex / t_complex) -- they cost only two cheap gauge-pinned reads of the solved
    S-matrix, no extra eigensolve. When `absorption=True` the solve additionally retains the per-layer
    partial cascades and exposes BORStack.layer_absorption() as (layer_absorption, layer_names), so the
    fundamental OpticalResult carries A_independent + a per-layer absorption map that closes R+T+sum A = 1
    to machine precision (AUDIT B4b)."""
    t0 = time.perf_counter()
    stack = _scaled_stack(spec, lambda_m)
    res = stack.solve(retain_internal=bool(absorption))
    dt = time.perf_counter() - t0
    ang = np.asarray(res["angles"], dtype=float)
    order = np.argsort(ang)                                   # near-axis fundamental first
    inc = np.asarray(res["inc"]).ravel()
    out = np.asarray(res["out"]).ravel()
    n_inc = int(inc.size)

    # Physically-phased specular amplitudes in the pinned deterministic gauge (diagonal = incident mode
    # -> same mode; gauge-invariant for reflection, deterministic for transmission). Cheap; always on.
    r_complex: Optional[np.ndarray] = None
    t_complex: Optional[np.ndarray] = None
    if n_inc:
        amp_r = np.asarray(stack.per_mode_amplitudes("reflection")["amplitude"])   # (n_inc, n_inc)
        r_diag = np.diagonal(amp_r).copy()
        amp_t = np.asarray(stack.per_mode_amplitudes("transmission")["amplitude"])  # (n_out, n_inc)
        t_diag = np.full(n_inc, np.nan, dtype=complex)
        for k in range(n_inc):
            pos = np.where(out == inc[k])[0]     # SAME transverse mode transmitted into the substrate
            if pos.size:
                t_diag[k] = amp_t[int(pos[0]), k]
        r_complex = r_diag[order]
        t_complex = t_diag[order]

    layer_abs: Optional[np.ndarray] = None
    layer_names: Optional[List[str]] = None
    if absorption:
        A = np.asarray(stack.layer_absorption(), dtype=float)      # (n_layers, n_inc), inc order
        layer_abs = A[:, order] if (A.ndim == 2 and A.size) else A
        layer_names = _layer_names(spec)

    return BorResult(angles_rad=ang[order], R=np.asarray(res["R"], float)[order],
                     T=np.asarray(res["T"], float)[order], energy=np.asarray(res["energy"], float)[order],
                     solve_time_s=dt, raw=res, r_complex=r_complex, t_complex=t_complex,
                     layer_absorption=layer_abs, layer_names=layer_names)


def bor_result_to_optical_result(res: BorResult) -> OpticalResult:
    """The fundamental (near-normal) incident mode of a BorResult as an OpticalResult (R/T/A)."""
    return res.fundamental_result()


def make_lumenairy_bor_solver(*, n_radial: int = 256, absorption: bool = False):
    """A BOR-PMM optical solver: `solve(spec_without_N, lambda_m) -> BorResult`, with n_radial supplied
    here (so a spec can omit it). Mirrors the make_lumenairy_*_solver factories; the BOR axisymmetric
    geometry is NOT the Cartesian LayeredStackSolver, so this returns a BorResult (per-mode), not a
    single-channel OpticalResult -- call .fundamental_result() for the near-normal R/T. `absorption=True`
    retains the per-layer absorbed-fraction map (A_independent + per_region_absorption on the fundamental,
    R+T+sum A = 1 to machine precision); the reflection/transmission phase is threaded either way."""
    def _solve(spec: BorStackSpec, lambda_m: float) -> BorResult:
        if int(spec.n_radial) != int(n_radial):
            from dataclasses import replace
            spec = replace(spec, n_radial=int(n_radial))
        return solve_bor(spec, lambda_m, absorption=absorption)
    return _solve
