"""
Liquid-crystal director driver for reconfigurable (LC) metasurfaces (roadmap Phase 4b). A nematic
cell of thickness d with PLANAR anchoring (director in the plate plane, tilt theta = 0, at both
surfaces z = 0 and z = d) and a voltage V across it. For a positive dielectric anisotropy
(dEps_static = eps_par - eps_perp > 0) the director tilts toward the field (z) above the
Freedericksz threshold

    V_th = pi * sqrt(K / (eps0 * dEps_static))                 (1-constant elastic, INDEPENDENT of d)

It PRODUCES the equilibrium tilt profile theta(z); a LiquidCrystalModel (core/effects) turns the
director into a UNIAXIAL optical tensor eps along n-hat = (cos theta, 0, sin theta) (reusing the
tensor-eps FEM path, incl. the off-diagonal UPML solve for an intermediate tilt). The bridge does
not yet auto-assemble the director field -- the driver produces theta for the caller to place in the
bundle (a tracked seam; the Freedericksz threshold + uniaxial response are validated in
validation/reconfigurable_modulators.py and tests/test_lc_director.py).

Method: the 1-constant Frank torque balance has a first integral (1/2 K theta'^2 = g(theta) -
g(theta_m), g = D^2/(2 eps0 eps(theta)), eps(theta) = eps_perp + dEps_static sin^2 theta) whose
midplane tilt theta_m fixes the voltage through a single non-singular elliptic quadrature

    V(theta_m) = 2 sqrt(K/eps0) * B(theta_m),
    B = int_0^{pi/2} sqrt( eps(theta_m) / (dEps * eps(theta)) ) / sqrt(1 - sin^2(theta_m) sin^2 phi) dphi,

with sin(theta) = sin(theta_m) sin(phi) (the substitution that removes the turning-point
singularity). V(theta_m) increases monotonically from V_th (theta_m -> 0) to infinity, so theta_m
is recovered by bisection -- robust right through the pitchfork (where a fixed-seed energy
minimizer stalls on the quartic-flat landscape). Below V_th there is no tilted solution
(theta = 0). SI units (K in J/m i.e. N; eps_* are STATIC relative permittivities; V in volts;
d in metres). Pure numpy/scipy; no devsim/ngsolve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from dynameta.constants import EPS0
from dynameta.core.numerics import trapz


def freedericksz_threshold_V(K_elastic: float, dEps_static: float) -> float:
    """Freedericksz threshold voltage V_th = pi sqrt(K/(eps0 dEps)) for a planar nematic cell with
    positive dielectric anisotropy (independent of cell thickness). Raises for dEps <= 0 (no
    threshold instability -- the field does not tilt a negative/zero-anisotropy planar cell)."""
    if not (K_elastic > 0.0):
        raise ValueError("K_elastic must be > 0")
    if not (dEps_static > 0.0):
        raise ValueError("dEps_static must be > 0 (positive anisotropy required for a planar-cell "
                         "Freedericksz transition)")
    return float(np.pi * np.sqrt(float(K_elastic) / (EPS0 * float(dEps_static))))


@dataclass
class DirectorProfile:
    z_m: np.ndarray         # grid through the cell, 0..d
    theta_rad: np.ndarray   # director tilt from the plate plane (0 = planar, pi/2 = homeotropic)
    theta_max_rad: float    # midplane (maximum) tilt
    V_applied: float        # applied voltage
    V_th: float             # Freedericksz threshold


def _B(theta_m: float, dEps: float, ep: float, nphi: int = 1201) -> float:
    """The elliptic quadrature B(theta_m) (see module docstring); bounded for theta_m < pi/2."""
    s2 = np.sin(theta_m) ** 2
    phi = np.linspace(0.0, 0.5 * np.pi, int(nphi))
    sp2 = np.sin(phi) ** 2
    eps_th = ep + dEps * s2 * sp2                          # eps(theta(phi))
    eps_thm = ep + dEps * s2                               # eps(theta_m)
    integrand = np.sqrt(eps_thm / (dEps * eps_th)) / np.sqrt(1.0 - s2 * sp2)
    return float(trapz(integrand, phi))


def _V_of_theta_m(theta_m: float, K: float, dEps: float, ep: float) -> float:
    return 2.0 * np.sqrt(K / EPS0) * _B(theta_m, dEps, ep)


def director_profile(K_elastic: float, dEps_static: float, eps_perp: float, d_m: float,
                      applied_V: float, *, nz: int = 201) -> DirectorProfile:
    """Equilibrium director tilt theta(z) for a planar nematic cell under `applied_V`. Below V_th
    theta = 0 (planar); above, the midplane tilt theta_m is recovered by bisecting the monotone
    V(theta_m) relation, and theta(z) is reconstructed from the parametric turning-point map.
    Returns a DirectorProfile (theta in the PLATE-PLANE convention: 0 = planar, pi/2 = homeotropic).

    DEPRECATED for general use. This is the 1-constant (single K), PLANAR, STATIC, dielectric-only,
    full-voltage-across-the-LC (no fixed-layer division) special case. For real modeling use the
    comprehensive director solver: director_profile_bvp (two-constant K11/K33, planar OR cylindrical,
    Poisson voltage-division through fixed dielectric layers, flexoelectricity) and lc_dynamics.LCDynamics
    (time-domain Erickson-Leslie switching). director_profile is RETAINED only as the exact-planar
    Freedericksz bifurcation reference: it solves the EXACT strong-planar cell (theta_b = 0) right through
    the pitchfork via an elliptic-quadrature bisection with no initial guess and no pretilt -- something
    the general solve_bvp path needs a small pretilt to do. director_profile_bvp at K11=K33 with
    field_model='poisson' (constant-displacement field, no fixed layers) reproduces it to ~8e-3 rad
    through the pi/2 angle bridge (director_to_extra_fields)."""
    K = float(K_elastic); dEps = float(dEps_static); ep = float(eps_perp)
    d = float(d_m); V = float(applied_V)
    if not (K > 0 and dEps > 0 and ep > 0 and d > 0 and int(nz) >= 11):
        raise ValueError("require K>0, dEps>0, eps_perp>0, d>0, nz>=11")
    Vth = freedericksz_threshold_V(K, dEps)
    z = np.linspace(0.0, d, int(nz))
    if V <= Vth * (1.0 + 1e-9):                            # below threshold: planar (no tilt)
        return DirectorProfile(z_m=z, theta_rad=np.zeros_like(z), theta_max_rad=0.0,
                               V_applied=V, V_th=Vth)
    # bisect the monotone V(theta_m) for theta_m in (0, pi/2)
    lo, hi = 1e-6, 0.5 * np.pi - 1e-6
    if _V_of_theta_m(hi, K, dEps, ep) < V:                # extreme over-drive: saturate near homeotropic
        theta_m = hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _V_of_theta_m(mid, K, dEps, ep) < V:
                lo = mid
            else:
                hi = mid
        theta_m = 0.5 * (lo + hi)
    # reconstruct theta(z): the cumulative A(phi) maps phi in [0, pi/2] to z in [0, d/2]
    s = np.sin(theta_m); s2 = s * s
    phi = np.linspace(0.0, 0.5 * np.pi, 4001)
    th_phi = np.arcsin(np.clip(s * np.sin(phi), -1.0, 1.0))
    eps_th = ep + dEps * np.sin(th_phi) ** 2
    eps_thm = ep + dEps * s2
    integ = np.sqrt(eps_th * eps_thm / dEps) / np.sqrt(1.0 - s2 * np.sin(phi) ** 2)
    A = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(phi))])
    z_half = A / A[-1] * (0.5 * d)                        # z from 0 (surface) to d/2 (midplane)
    nh = int(nz) // 2
    zu = np.linspace(0.0, 0.5 * d, nh + 1)
    th_u = np.interp(zu, z_half, th_phi)                  # tilt on the half grid
    theta_full = np.concatenate([th_u, th_u[-2::-1]])     # mirror about the midplane
    z_full = np.concatenate([zu, d - zu[-2::-1]])
    theta = np.interp(z, z_full, theta_full)              # onto the requested uniform z grid
    return DirectorProfile(z_m=z, theta_rad=theta, theta_max_rad=float(np.max(theta)),
                           V_applied=V, V_th=Vth)


# =============================================================================
# Two-constant (K11/K33) statics: electrostatics + flexoelectricity + optics + BVP
# -----------------------------------------------------------------------------
# CONVENTION (important): the functions below use the FIELD-AXIS angle convention
# of the external nematic solver they are ported from -- theta is measured from the
# applied-field axis (theta = 0 -> director ALONG the field / cell normal =
# HOMEOTROPIC; theta = pi/2 -> director in the plate plane = PLANAR), and the LC
# permittivity along the field is eps(theta) = eps_para cos^2 theta + eps_perp
# sin^2 theta. This is the COMPLEMENT of the plate-plane convention used by
# director_profile() / freedericksz_threshold_V() above and by the optical
# LiquidCrystalModel (which reads 'director_angle_rad' measured FROM the plate
# plane). Bridge a field-axis profile to the optics with director_to_extra_fields()
# (theta_optic = pi/2 - theta_field). Two constants: K11 = splay, K33 = bend, with
# the local effective constant K_eff(theta) = K11 sin^2 theta + K33 cos^2 theta and
# the (K11 - K33) sin theta cos theta (theta')^2 saddle term. SI units; numpy/scipy.
# =============================================================================


def eps_along_field(theta, eps_para: float, eps_perp: float):
    """Uniaxial relative permittivity ALONG the field for a FIELD-AXIS theta:
    eps(theta) = eps_para cos^2 theta + eps_perp sin^2 theta."""
    theta = np.asarray(theta, dtype=float)
    c = np.cos(theta); s = np.sin(theta)
    return float(eps_para) * c * c + float(eps_perp) * s * s


@dataclass
class LCGeometry:
    """LC cell geometry: a planar gap or a coaxial (cylindrical) cell, fixed layers excluded."""
    geometry: str            # 'planar' | 'cyl'
    d_lc: float              # LC thickness (m)
    z_m: np.ndarray          # 0..d_lc grid (m)
    r_m: np.ndarray          # radial coordinate (= z planar; a + t_in + z for cyl)
    r_in: float
    r_out: float


def compute_lc_geometry(*, geometry: str = "planar", nz: int = 201,
                        d_planar: "Optional[float]" = None,
                        a: "Optional[float]" = None, b: "Optional[float]" = None,
                        t_in: float = 0.0, t_out: float = 0.0) -> LCGeometry:
    """Planar gap (d_planar metal-to-metal minus fixed layers t_in/t_out) or coaxial cell
    (inner radius a, outer b, fixed layers t_in/t_out). All SI (metres)."""
    geometry = str(geometry).strip().lower()
    nz = int(nz)
    if nz < 11:
        raise ValueError("nz must be >= 11")
    if geometry == "planar":
        if d_planar is None or not (d_planar > 0):
            raise ValueError("planar geometry needs d_planar > 0 (metal-to-metal gap, m)")
        d_lc = float(d_planar) - float(t_in) - float(t_out)
        if d_lc <= 0:
            raise ValueError("planar LC thickness <= 0 (d_planar must exceed t_in + t_out)")
        z = np.linspace(0.0, d_lc, nz)
        return LCGeometry("planar", d_lc, z, z.copy(), 0.0, d_lc)
    if geometry == "cyl":
        if a is None or b is None or not (float(b) - float(t_out) > float(a) + float(t_in)):
            raise ValueError("cyl geometry needs a, b with (b - t_out) > (a + t_in)")
        r_in = float(a) + float(t_in); r_out = float(b) - float(t_out)
        d_lc = r_out - r_in
        z = np.linspace(0.0, d_lc, nz)
        return LCGeometry("cyl", d_lc, z, r_in + z, r_in, r_out)
    raise ValueError("geometry must be 'planar' or 'cyl'")


def flexo_p_along_field(theta, dtheta_dz, e1: float, e3: float, *,
                        geometry: str = "planar", r=None):
    """Flexoelectric polarization along the field (C/m^2), FIELD-AXIS theta. planar:
    P = -(e1 + e3) sin cos dtheta/dz; cyl adds the splay term e1 cos^2 theta / r."""
    theta = np.asarray(theta, dtype=float)
    s = np.sin(theta); c = np.cos(theta); ebar = float(e1) + float(e3)
    P = -ebar * s * c * np.asarray(dtheta_dz, dtype=float)
    if str(geometry).strip().lower() == "cyl":
        rr = np.maximum(np.asarray(r, dtype=float), 1e-30)
        P = P + float(e1) * c * c / rr
    return P


def flexo_direct_torque(theta, E, dE_dz, e1: float, e3: float, *,
                        geometry: str = "planar", r=None):
    """Flexoelectric direct-torque contribution to the director equation, FIELD-AXIS theta.
    planar: (e1 + e3) dE/dz sin cos; cyl adds (e3 - e1) E/r sin cos."""
    theta = np.asarray(theta, dtype=float)
    s = np.sin(theta); c = np.cos(theta); ebar = float(e1) + float(e3)
    tq = ebar * np.asarray(dE_dz, dtype=float) * s * c
    if str(geometry).strip().lower() == "cyl":
        rr = np.maximum(np.asarray(r, dtype=float), 1e-30)
        tq = tq + (float(e3) - float(e1)) * np.asarray(E, dtype=float) / rr * s * c
    return tq


def solve_lc_field_profile(theta, V_app: float, geo: LCGeometry, *,
                           eps_para: float, eps_perp: float, field_model: str = "poisson",
                           t_in: float = 0.0, t_out: float = 0.0,
                           eps_in: float = 7.5, eps_out: float = 7.5,
                           a: "Optional[float]" = None, b: "Optional[float]" = None,
                           e1: float = 0.0, e3: float = 0.0,
                           flexo_self_consistent: bool = False):
    """Return (E(z), V_lc) for a FIELD-AXIS theta(z) and applied voltage. 'uniform': E = V_app/d_lc,
    V_lc = V_app. 'poisson': quasi-static voltage division across the SERIES fixed dielectric layers +
    the theta-dependent LC eps (planar series-C / cyl log-capacitance), with an optional flexoelectric
    self-consistent depolarization field (nonzero E even at V_app = 0)."""
    theta = np.asarray(theta, dtype=float)
    z = geo.z_m; r = geo.r_m; d_lc = geo.d_lc
    V_app = float(V_app)
    if str(field_model).strip().lower() == "uniform":
        if d_lc <= 0:
            return np.zeros_like(theta), 0.0
        return np.full_like(theta, V_app / d_lc), V_app
    flexo_on = bool(flexo_self_consistent and (e1 or e3))
    if abs(V_app) < 1e-30 and not flexo_on:
        return np.zeros_like(theta), 0.0
    eps = eps_along_field(theta, eps_para, eps_perp)
    P_f = np.zeros_like(theta)
    if flexo_on:
        P_f = flexo_p_along_field(theta, np.gradient(theta, z), e1, e3, geometry=geo.geometry, r=r)
    if geo.geometry == "cyl":
        denom = 0.0
        if t_in > 0:
            denom += (1.0 / eps_in) * math.log((a + t_in) / a)
        denom += trapz(1.0 / (r * eps), z)
        if t_out > 0:
            denom += (1.0 / eps_out) * math.log(b / (b - t_out))
        if not math.isfinite(denom) or denom <= 0:
            return np.zeros_like(theta), 0.0
        flexo_v = trapz(P_f / (EPS0 * eps), z) if flexo_on else 0.0
        g = (V_app + flexo_v) / denom
        E = (g / r - P_f / EPS0) / eps
        return E, float(trapz(E, z))
    denom = 0.0
    if t_in > 0:
        denom += t_in / eps_in
    denom += trapz(1.0 / eps, z)
    if t_out > 0:
        denom += t_out / eps_out
    if not math.isfinite(denom) or denom <= 0:
        return np.zeros_like(theta), 0.0
    flexo_v = trapz(P_f / (EPS0 * eps), z) if flexo_on else 0.0
    g = (V_app + flexo_v) / denom
    E = (g - P_f / EPS0) / eps
    return E, float(trapz(E, z))


def n_local_from_theta(theta, n_o: float, n_e: float, model: str = "extra_k_radial"):
    """Local refractive index for a FIELD-AXIS theta. 'ordinary' = n_o; 'extra_k_axis':
    1/n^2 = sin^2/n_e^2 + cos^2/n_o^2; 'extra_k_radial': 1/n^2 = cos^2/n_e^2 + sin^2/n_o^2."""
    theta = np.asarray(theta, dtype=float)
    if model == "ordinary":
        return np.full_like(theta, float(n_o))
    n_o = float(n_o); n_e = float(n_e)
    if n_o <= 0 or n_e <= 0:
        raise ValueError("n_o and n_e must be > 0")
    s = np.sin(theta); c = np.cos(theta)
    if model == "extra_k_axis":
        return 1.0 / np.sqrt(s * s / (n_e * n_e) + c * c / (n_o * n_o))
    if model == "extra_k_radial":
        return 1.0 / np.sqrt(c * c / (n_e * n_e) + s * s / (n_o * n_o))
    raise ValueError("unknown optical model: {!r}".format(model))


def n_eff_from_theta_profile(theta, z, n_o: float, n_e: float, *, model: str = "extra_k_radial",
                             d_lc: "Optional[float]" = None, n_opt_fine: int = 271,
                             include_fixed: bool = False, n_fixed_in: float = 1.0,
                             n_fixed_out: float = 1.0, t_in: float = 0.0, t_out: float = 0.0) -> float:
    """Optical-path-average effective index over a FIELD-AXIS theta(z) (optionally including the
    fixed inner/outer layers' OPL). n_eff = OPL / d_total."""
    theta = np.asarray(theta, dtype=float); z = np.asarray(z, dtype=float)
    if d_lc is None:
        d_lc = float(z[-1] - z[0]) if z.size > 1 else float("nan")
    d_lc = float(d_lc)
    if not (d_lc > 0):
        return float("nan")
    nf = max(25, int(n_opt_fine))
    zf = np.linspace(float(z[0]), float(z[-1]), nf)
    nloc = n_local_from_theta(np.interp(zf, z, theta), n_o, n_e, model)
    opl = trapz(nloc, zf)
    if include_fixed:
        opl += float(n_fixed_in) * float(t_in) + float(n_fixed_out) * float(t_out)
        return float(opl / (d_lc + float(t_in) + float(t_out)))
    return float(opl / d_lc)


@dataclass
class LCStaticResult:
    """Two-constant static director solution at one applied voltage (FIELD-AXIS theta)."""
    V_app: float
    V_lc: float
    n_eff: float
    z_m: np.ndarray
    theta_field_rad: np.ndarray   # tilt from the field axis (0 = homeotropic, pi/2 = planar)
    theta_b_rad: float
    success: bool = True
    message: str = ""


def director_to_extra_fields(theta_field_rad) -> dict:
    """Bridge a FIELD-AXIS director tilt theta(z) to the plate-plane 'director_angle_rad' that the
    optical LiquidCrystalModel (core/effects) consumes: theta_optic = pi/2 - theta_field. Returns
    {'director_angle_rad': ...} ready to drop into the optics field bundle."""
    return {"director_angle_rad": 0.5 * np.pi - np.asarray(theta_field_rad, dtype=float)}


def director_profile_bvp(*, K11: float, K33: float, eps_para: float, eps_perp: float, V_app: float,
                         geo: "Optional[LCGeometry]" = None, geometry: str = "planar",
                         d_planar: "Optional[float]" = None,
                         nz: int = 201, theta_b_rad: float = math.radians(89.9),
                         field_model: str = "uniform", t_in: float = 0.0, t_out: float = 0.0,
                         eps_in: float = 7.5, eps_out: float = 7.5,
                         a: "Optional[float]" = None, b: "Optional[float]" = None,
                         include_cyl_elastic_corr: bool = False,
                         e1: float = 0.0, e3: float = 0.0, include_flexo: bool = False,
                         flexo_self_consistent: bool = False,
                         n_o: "Optional[float]" = None, n_e: "Optional[float]" = None,
                         opt_model: str = "extra_k_radial", rtol: float = 1e-6) -> LCStaticResult:
    """Two-constant (K11 splay / K33 bend) static director BVP at one applied voltage, FIELD-AXIS theta,
    via scipy.integrate.solve_bvp on the scaled coordinate u = z/d_lc. The Euler-Lagrange torque balance
    K_eff theta'' = -[(K11 - K33) sin cos (theta')^2 - eps0 dEps E^2 sin cos + flexo + cyl-corr] with
    K_eff = K11 sin^2 + K33 cos^2 and dEps = eps_para - eps_perp, strong anchoring theta(0)=theta(d)=
    theta_b. Several initial guesses are tried; the branch consistent with the field-driven direction is
    preferred, free energy breaks ties (robust through the Freedericksz pitchfork). Reduces to the
    1-constant elliptic-quadrature director_profile() at K11=K33 (through the pi/2 angle bridge)."""
    from scipy.integrate import solve_bvp
    K11 = float(K11); K33 = float(K33); dEps = float(eps_para) - float(eps_perp)
    if not (K11 > 0 and K33 > 0):
        raise ValueError("K11, K33 must be > 0")
    if geo is None:
        geo = compute_lc_geometry(geometry=geometry, nz=nz, d_planar=d_planar, a=a, b=b,
                                  t_in=t_in, t_out=t_out)
    z = geo.z_m; r = geo.r_m; d_lc = float(geo.d_lc)
    theta_b = float(theta_b_rad)
    e1e = float(e1) if include_flexo else 0.0
    e3e = float(e3) if include_flexo else 0.0
    fsc = bool(include_flexo and flexo_self_consistent)
    fkw = dict(eps_para=eps_para, eps_perp=eps_perp, field_model=field_model, t_in=t_in, t_out=t_out,
               eps_in=eps_in, eps_out=eps_out, a=a, b=b, e1=e1e, e3=e3e, flexo_self_consistent=fsc)
    u = (z - float(z[0])) / d_lc

    def _field(theta):
        return solve_lc_field_profile(theta, float(V_app), geo, **fkw)

    def _prepare(g):
        gg = np.asarray(g, dtype=float).copy()
        if gg.size != z.size:
            gg = np.interp(u, np.linspace(0.0, 1.0, gg.size), gg)
        gg[0] = gg[-1] = theta_b
        return np.clip(gg, -0.499 * math.pi, 0.999 * math.pi)

    # one-hump initial guess: a positive dEps field drives theta DOWN (toward homeotropic, 0)
    V_th = freedericksz_threshold_V(K11, dEps) if dEps > 0 else float("inf")
    try:
        _e0, vlc0 = _field(np.full_like(z, theta_b))
        drive = abs(float(vlc0)) if math.isfinite(vlc0) else abs(float(V_app))
    except Exception:
        drive = abs(float(V_app))
    drive_frac = math.tanh(max(0.0, drive / V_th - 0.85)) if math.isfinite(V_th) and V_th > 0 else 0.35
    sign = -1.0 if dEps >= 0.0 else 1.0
    max_amp = min(theta_b * 0.85, math.radians(82.0)) if theta_b < 0.5 * math.pi else math.radians(82.0)
    base = theta_b + sign * (max_amp * max(0.0, min(1.0, drive_frac))) * np.sin(math.pi * u)
    base[0] = base[-1] = theta_b
    guesses: List[np.ndarray] = []
    for g in (base, theta_b - (base - theta_b), np.full_like(z, theta_b)):
        gg = _prepare(g)
        if not any(o.shape == gg.shape and np.allclose(o, gg, atol=1e-9) for o in guesses):
            guesses.append(gg)

    def _energy(theta):
        try:
            th = np.asarray(theta, float).copy(); th[0] = th[-1] = theta_b
            dth = np.gradient(th, z)
            E, _v = _field(th)
            s = np.sin(th); c = np.cos(th)
            Keff = K11 * s * s + K33 * c * c
            dens = 0.5 * Keff * dth * dth + 0.5 * EPS0 * dEps * (E * E) * (c * c)
            if e1e or e3e:
                dens = dens - flexo_p_along_field(th, dth, e1e, e3e, geometry=geo.geometry, r=r) * E
            val = trapz(dens * (np.maximum(r, 1e-30) if geo.geometry == "cyl" else 1.0), z)
            return float(val) if math.isfinite(val) else float("inf")
        except Exception:
            return float("inf")

    def _solve(guess):
        y0 = np.vstack((guess, np.gradient(guess, u)))

        def fun(x, y):
            uu = np.asarray(x, float); zz = float(z[0]) + uu * d_lc
            rr = (geo.r_in + zz) if geo.geometry == "cyl" else zz
            th = np.asarray(y[0], float); dth_du = np.asarray(y[1], float); dth_dz = dth_du / d_lc
            gloc = LCGeometry(geo.geometry, d_lc, zz, rr, geo.r_in, geo.r_out)
            E, _v = solve_lc_field_profile(th, float(V_app), gloc, **fkw)
            s = np.sin(th); c = np.cos(th)
            Keff = np.where(np.abs(K11 * s * s + K33 * c * c) < 1e-300, 1e-300, K11 * s * s + K33 * c * c)
            t2 = (K11 - K33) * s * c * (dth_dz * dth_dz)
            t3 = -EPS0 * dEps * (E * E) * s * c
            t4 = 0.0
            if e1e or e3e:
                dE = np.gradient(np.asarray(E, dtype=float), np.asarray(zz, dtype=float))
                t4 = flexo_direct_torque(th, E, dE, e1e, e3e, geometry=geo.geometry, r=rr)
            textra = (Keff * (dth_dz / np.maximum(rr, 1e-30))) if (geo.geometry == "cyl"
                                                                   and include_cyl_elastic_corr) else 0.0
            d2 = -((textra + t2 + t3 + t4) / Keff) * (d_lc * d_lc)
            return np.vstack((dth_du, d2))

        def bc(ya, yb):
            return np.array([ya[0] - theta_b, yb[0] - theta_b], dtype=float)

        sol = solve_bvp(fun, bc, u, y0, tol=max(float(rtol), 3e-2), max_nodes=max(5000, int(nz) * 80),
                        verbose=0)
        if not sol.success:
            raise RuntimeError(str(sol.message))
        th = np.asarray(sol.sol(u)[0], float); th[0] = th[-1] = theta_b
        _E, V_lc = _field(th)
        neff = (n_eff_from_theta_profile(th, z, n_o, n_e, model=opt_model, d_lc=d_lc)
                if (n_o is not None and n_e is not None) else float("nan"))
        return LCStaticResult(V_app=float(V_app), V_lc=float(V_lc), n_eff=float(neff), z_m=z,
                              theta_field_rad=th, theta_b_rad=theta_b, success=True, message=str(sol.message))

    cands: List[Tuple[float, float, LCStaticResult]] = []
    direction = -1.0 if dEps >= 0.0 else 1.0
    use_score = abs(float(V_app)) > 1e-14 and abs(dEps) > 1e-14
    last = ""
    for g in guesses:
        try:
            res = _solve(g)
            score = direction * float(np.nanmean(res.theta_field_rad - theta_b)) if use_score else 0.0
            cands.append((score, _energy(res.theta_field_rad), res))
        except Exception as exc:
            last = str(exc)
    if not cands:
        raise RuntimeError("director_profile_bvp failed at V={:.6g}: {}".format(float(V_app),
                                                                                last or "no candidate"))
    cands.sort(key=lambda it: (-it[0], it[1]))
    return cands[0][2]
