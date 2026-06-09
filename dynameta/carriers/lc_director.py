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
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from dynameta.constants import EPS0
from dynameta.core.numerics import trapz

__all__ = [
    "DirectorProfile", "LCGeometry", "LCStaticResult",
    "freedericksz_threshold_V", "director_profile", "director_profile_bvp",
    "haller_order_parameter", "K_of_temperature", "gamma1_of_temperature",
    "compute_lc_geometry", "solve_lc_field_profile", "eps_along_field",
    "flexo_p_along_field", "flexo_direct_torque",
    "n_local_from_theta", "n_eff_from_theta_profile",
    "director_to_extra_fields", "reduce_director",
    "LCChiralResult", "chiral_director_profile_bvp",
    "cholesteric_q0", "gooch_tarry_transmission", "mauguin_number",
]


def freedericksz_threshold_V(K_elastic: float, dEps_static: float) -> float:
    """Freedericksz threshold voltage V_th = pi sqrt(K/(eps0 dEps)) for a planar nematic cell with
    positive dielectric anisotropy (independent of cell thickness). Raises for dEps <= 0 (no
    threshold instability -- the field does not tilt a negative/zero-anisotropy planar cell).

    K-SELECTION (two-constant cells): pass the elastic constant that governs the FIRST deformation of
    the cell -- K11 (splay) for a PLANAR cell (theta_b ~ pi/2), K33 (bend) for a HOMEOTROPIC cell
    (theta_b ~ 0). Using the wrong constant rescales V_th by sqrt(K_used/K_correct). For a VAN cell
    (negative anisotropy, homeotropic anchoring) the magnitude threshold is pi sqrt(K33/(eps0 |dEps|))."""
    if not (K_elastic > 0.0):
        raise ValueError("K_elastic must be > 0")
    if not (dEps_static > 0.0):
        raise ValueError("dEps_static must be > 0 (positive anisotropy required for a planar-cell "
                         "Freedericksz transition)")
    return float(np.pi * np.sqrt(float(K_elastic) / (EPS0 * float(dEps_static))))


# -----------------------------------------------------------------------------
# Temperature dependence of the material constants (Haller order parameter)
# -----------------------------------------------------------------------------
def haller_order_parameter(T_K: float, T_NI_K: float, *, beta: float = 0.22) -> float:
    """Nematic scalar order parameter S(T) via the Haller extrapolation S = (1 - T/T_NI)^beta
    (beta ~ 0.18-0.25 for common nematics; S -> 0 at the nematic-isotropic transition T_NI). Returns
    0 at/above T_NI."""
    T_K = float(T_K); T_NI_K = float(T_NI_K)
    if not (T_NI_K > 0.0):
        raise ValueError("T_NI_K must be > 0")
    if T_K >= T_NI_K:
        return 0.0
    return float((1.0 - T_K / T_NI_K) ** float(beta))


def K_of_temperature(K_ref: float, T_K: float, *, T_ref_K: float = 300.0, T_NI_K: float = 380.0,
                     beta: float = 0.22) -> float:
    """Frank elastic constant at temperature T from its value K_ref at T_ref. The Frank constants scale
    as the SQUARE of the order parameter, K(T) = K_ref [S(T)/S(T_ref)]^2 (the standard mean-field result;
    the dielectric anisotropy dEps and birefringence scale ~ S, so V_th(T) ~ sqrt(K/dEps) ~ S^(1/2),
    but the cleanest single scaling the caller applies per-constant is this K ~ S^2). At/above T_NI K -> 0."""
    S = haller_order_parameter(T_K, T_NI_K, beta=beta)
    S_ref = haller_order_parameter(T_ref_K, T_NI_K, beta=beta)
    if not (S_ref > 0.0):
        raise ValueError("T_ref_K must be below T_NI_K (S(T_ref) > 0)")
    return float(K_ref) * (S / S_ref) ** 2


def gamma1_of_temperature(gamma1_ref: float, T_K: float, *, T_ref_K: float = 300.0,
                          E_a_eV: float = 0.4, T_NI_K: "Optional[float]" = None,
                          beta: float = 0.22) -> float:
    """Rotational viscosity gamma1(T) = gamma1_ref * exp((E_a/kB)(1/T - 1/T_ref)) * [S(T)/S(T_ref)]
    -- an Arrhenius activation (activation energy E_a in eV, ~0.3-0.6 eV for nematics) times the
    order-parameter factor S (gamma1 ~ S exp(E_a/kT), Diogo-Martins). T_NI_K=None drops the S factor
    (pure Arrhenius). gamma1 FALLS with temperature (the exp dominates the modest S decline)."""
    from dynameta.constants import KB, Q_E
    T_K = float(T_K); T_ref_K = float(T_ref_K)
    arr = math.exp((float(E_a_eV) * Q_E / KB) * (1.0 / T_K - 1.0 / T_ref_K))
    s_fac = 1.0
    if T_NI_K is not None:
        S = haller_order_parameter(T_K, float(T_NI_K), beta=beta)
        S_ref = haller_order_parameter(T_ref_K, float(T_NI_K), beta=beta)
        s_fac = (S / S_ref) if S_ref > 0.0 else 1.0
    return float(gamma1_ref) * arr * s_fac


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


def reduce_director(theta_zt_rad, *, t_index: int = -1, reduce: str = "profile") -> dict:
    """Convenience for the dynamics: select a time column from an LCDynamicsResult.theta_zt_rad (nz, nt)
    and turn it into the optics 'director_angle_rad' bundle (applies the same pi/2 field-axis ->
    plate-plane flip as director_to_extra_fields). reduce='profile' returns the full theta(z) (a gridded
    director, the documented (...,)->(...,3,3) convention); 'midplane'/'mean' collapse to a single scalar
    tilt. t_index selects the time slice (-1 = final)."""
    th = np.asarray(theta_zt_rad, dtype=float)
    col = th[:, int(t_index)] if th.ndim == 2 else th
    if reduce == "midplane":
        col = np.asarray(float(col[col.size // 2]))
    elif reduce == "mean":
        col = np.asarray(float(np.mean(col)))
    elif reduce != "profile":
        raise ValueError("reduce must be 'profile', 'midplane' or 'mean'")
    return director_to_extra_fields(col)


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
                         opt_model: str = "extra_k_radial", rtol: float = 1e-6,
                         W_anchor_J_m2: "Optional[float]" = None,
                         theta_easy_rad: "Optional[float]" = None) -> LCStaticResult:
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
    # finite (Rapini-Papoular) surface anchoring: W_anchor_J_m2 = the anchoring strength (J/m^2); the
    # surface director need NOT pin to the easy axis theta_easy (default theta_b). The Euler-Lagrange
    # NATURAL BC is a torque balance K_eff(theta) dtheta/dz = +/- (1/2) W sin(2(theta - theta_easy)) at
    # each plate (extrapolation length b = K/W). W_anchor_J_m2 = None -> STRONG anchoring (Dirichlet
    # theta = theta_b, byte-identical to before); W -> inf recovers it.
    theta_easy = float(theta_easy_rad) if theta_easy_rad is not None else theta_b
    W_anchor = float(W_anchor_J_m2) if (W_anchor_J_m2 is not None and W_anchor_J_m2 > 0.0) else None
    e1e = float(e1) if include_flexo else 0.0
    e3e = float(e3) if include_flexo else 0.0
    fsc = bool(include_flexo and flexo_self_consistent)
    fkw = dict(eps_para=eps_para, eps_perp=eps_perp, field_model=field_model, t_in=t_in, t_out=t_out,
               eps_in=eps_in, eps_out=eps_out, a=a, b=b, e1=e1e, e3=e3e, flexo_self_consistent=fsc)
    u = (z - float(z[0])) / d_lc

    def _prepare(g):
        gg = np.asarray(g, dtype=float).copy()
        if gg.size != z.size:
            gg = np.interp(u, np.linspace(0.0, 1.0, gg.size), gg)
        gg[0] = gg[-1] = theta_b
        return np.clip(gg, -0.499 * math.pi, 0.999 * math.pi)

    # ---- robust branch selection via voltage CONTINUATION + an amplitude-seed ladder. The Freedericksz
    #      pitchfork, high overdrive, high splay/bend contrast (K11>>K33) and negative anisotropy (VAN)
    #      all have a thin-boundary-layer tilted branch that a single drive-scaled guess can miss
    #      (solve_bvp then converges only the trivial theta=theta_b branch -> n_eff silently rails). We
    #      ramp the voltage up from ~1.05 V_th, carrying each converged tilted profile as the next seed.
    Vmag = abs(float(V_app)); sgn_V = 1.0 if float(V_app) >= 0.0 else -1.0
    # MAGNITUDE-based seeding threshold (finite for dEps<0 too; only used for seeding, not a reported #)
    K_eff_mean = 0.5 * (K11 + K33)
    V_th = (math.pi * math.sqrt(K_eff_mean / (EPS0 * abs(dEps)))) if abs(dEps) > 1e-30 else float("inf")
    # tilt direction + the destabilized far anchoring the midplane swings toward: dEps>0 -> toward the
    # field (theta -> 0, homeotropic); dEps<0 -> toward planar (theta -> pi/2).
    sign = -1.0 if dEps >= 0.0 else 1.0
    reach = theta_b if dEps >= 0.0 else (0.5 * math.pi - theta_b)
    max_amp = min(abs(reach) * 0.98, math.radians(89.0))
    direction = -1.0 if dEps >= 0.0 else 1.0
    use_score = Vmag > 1e-14 and abs(dEps) > 1e-14

    def _field_at(theta, Vk):
        return solve_lc_field_profile(theta, float(Vk), geo, **fkw)

    def _energy(theta, Vk):
        try:
            th = np.asarray(theta, float).copy()
            if W_anchor is None:
                th[0] = th[-1] = theta_b                          # strong anchoring; weak keeps surface theta
            dth = np.gradient(th, z)
            E, _v = _field_at(th, Vk)
            s = np.sin(th); c = np.cos(th)
            Keff = K11 * s * s + K33 * c * c
            # FIELD-AXIS dielectric free-energy density is +0.5 eps0 dEps E^2 sin^2(theta) (its gradient
            # +dEps E^2 sin cos = -t3, the torque solved below); sin^2 (NOT cos^2) is the correct sign.
            dens = 0.5 * Keff * dth * dth + 0.5 * EPS0 * dEps * (E * E) * (s * s)
            if e1e or e3e:
                dens = dens - flexo_p_along_field(th, dth, e1e, e3e, geometry=geo.geometry, r=r) * E
            val = trapz(dens * (np.maximum(r, 1e-30) if geo.geometry == "cyl" else 1.0), z)
            return float(val) if math.isfinite(val) else float("inf")
        except Exception:
            return float("inf")

    def _solve_at(guess, Vk):
        y0 = np.vstack((guess, np.gradient(guess, u)))

        def fun(x, y):
            uu = np.asarray(x, float); zz = float(z[0]) + uu * d_lc
            rr = (geo.r_in + zz) if geo.geometry == "cyl" else zz
            th = np.asarray(y[0], float); dth_du = np.asarray(y[1], float); dth_dz = dth_du / d_lc
            gloc = LCGeometry(geo.geometry, d_lc, zz, rr, geo.r_in, geo.r_out)
            E, _v = solve_lc_field_profile(th, float(Vk), gloc, **fkw)
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
            if W_anchor is None:                                  # strong anchoring (Dirichlet)
                return np.array([ya[0] - theta_b, yb[0] - theta_b], dtype=float)
            # Rapini-Papoular torque balance (dtheta/dz = (dtheta/du)/d_lc): at z=0 the bulk elastic
            # torque balances +dF_s/dtheta, at z=d it balances -dF_s/dtheta (F_s = (1/2)W sin^2(theta-easy)).
            Ka = K11 * np.sin(ya[0]) ** 2 + K33 * np.cos(ya[0]) ** 2
            Kb = K11 * np.sin(yb[0]) ** 2 + K33 * np.cos(yb[0]) ** 2
            return np.array([Ka * (ya[1] / d_lc) - 0.5 * W_anchor * np.sin(2.0 * (ya[0] - theta_easy)),
                             Kb * (yb[1] / d_lc) + 0.5 * W_anchor * np.sin(2.0 * (yb[0] - theta_easy))],
                            dtype=float)

        sol = solve_bvp(fun, bc, u, y0, tol=max(float(rtol), 3e-2), max_nodes=max(5000, int(nz) * 80),
                        verbose=0)
        if not sol.success:
            raise RuntimeError(str(sol.message))
        th = np.asarray(sol.sol(u)[0], float)
        if W_anchor is None:
            th[0] = th[-1] = theta_b
        return th

    def _seeds(Vk, carried):
        gs = []
        if carried is not None:
            gs.append(_prepare(carried))                          # CONTINUATION: previous converged tilt
        df = (math.tanh(max(0.0, abs(Vk) / V_th - 0.85)) if (math.isfinite(V_th) and V_th > 0) else 0.5)
        flat = np.full_like(z, theta_b)
        for frac in (df, 0.3, 0.6, 0.85, 0.98):                   # amplitude ladder toward the far anchoring
            gs.append(_prepare(flat + sign * (max_amp * max(0.0, min(1.0, frac))) * np.sin(math.pi * u)))
        gs.append(_prepare(flat))                                 # trivial (untilted) fallback
        uniq: List[np.ndarray] = []
        for g in gs:
            if not any(o.shape == g.shape and np.allclose(o, g, atol=1e-9) for o in uniq):
                uniq.append(g)
        return uniq

    def _best_at(Vk, carried):
        cands: List[Tuple[float, float, np.ndarray]] = []
        last = ""
        for g in _seeds(Vk, carried):
            try:
                th = _solve_at(g, Vk)
                score = direction * float(np.nanmean(th - theta_b)) if use_score else 0.0
                cands.append((score, _energy(th, Vk), th))
            except Exception as exc:
                last = str(exc)
        if not cands:
            raise RuntimeError("no converged candidate (last: {})".format(last))
        cands.sort(key=lambda it: (-it[0], it[1]))                # most field-driven tilt, energy tie-break
        return cands[0][2]

    if (not use_score) or (not math.isfinite(V_th)) or Vmag <= 1.02 * V_th:
        ladder = [Vmag]                                           # below threshold: untilted is correct
    else:
        n_steps = int(min(14, max(2, math.ceil((Vmag - V_th) / max(0.75, 0.5 * V_th)) + 1)))
        ladder = list(np.linspace(min(1.05 * V_th, Vmag), Vmag, n_steps))
    carried = None
    try:
        for Vk in ladder:
            carried = _best_at(sgn_V * float(Vk), carried)
    except RuntimeError as exc:
        raise RuntimeError("director_profile_bvp failed at V={:.6g}: {}".format(float(V_app), exc))

    th = carried
    if W_anchor is None:
        th[0] = th[-1] = theta_b
    _E, V_lc = _field_at(th, float(V_app))
    neff = (n_eff_from_theta_profile(th, z, n_o, n_e, model=opt_model, d_lc=d_lc)
            if (n_o is not None and n_e is not None) else float("nan"))
    # wrong-branch safety net: well above threshold the cell MUST be tilted; a residual ~untilted result
    # is a solver miss, not physics -> flag it (success=False) rather than silently returning a railed n_eff.
    success, message = True, "ok"
    if use_score and math.isfinite(V_th) and Vmag > 1.3 * V_th:
        tilt = direction * float(np.mean(th - theta_b))           # > 0 if correctly tilted toward the field
        if tilt < math.radians(1.0):
            success = False
            message = ("director_profile_bvp: V={:.3g} > 1.3 V_th(~{:.3g}) but the solver stayed on the "
                       "~untilted branch (mean tilt {:.3g} rad); tilted branch not found".format(
                           Vmag, V_th, tilt))
    return LCStaticResult(V_app=float(V_app), V_lc=float(V_lc), n_eff=float(neff), z_m=z,
                          theta_field_rad=th, theta_b_rad=theta_b, success=success, message=message)


# -----------------------------------------------------------------------------
# Chiral / twisted nematic (TN, cholesteric) -- the SECOND director angle phi(z)
# -----------------------------------------------------------------------------
def cholesteric_q0(pitch_m: float) -> float:
    """Natural cholesteric twist wavenumber q0 = 2 pi / pitch (rad/m). A right-handed helix of pitch p0
    has q0 > 0; the director azimuth advances by 2 pi over one pitch."""
    p = float(pitch_m)
    if abs(p) < 1e-30:
        return 0.0
    return 2.0 * math.pi / p


def mauguin_number(d_lc: float, dn: float, wavelength: float, twist_total_rad: float = 0.5 * math.pi):
    """Mauguin parameter u = 2 d dn / (lambda * (2 phi_t / pi)) for a twisted cell -- the polarization
    'follows' the twist adiabatically (waveguiding) when u >> 1. For a 90 deg TN (phi_t = pi/2) this is the
    classic u = 2 d dn / lambda. Returns the dimensionless Mauguin number."""
    phit = abs(float(twist_total_rad))
    norm = (2.0 * phit / math.pi) if phit > 1e-12 else 1.0
    return 2.0 * float(d_lc) * float(dn) / (float(wavelength) * norm)


def gooch_tarry_transmission(d_lc: float, dn: float, wavelength: float, *,
                             twist_total_rad: float = 0.5 * math.pi) -> float:
    """Normalized optical transmission of a 90 deg twisted-nematic cell between crossed polarizers (input
    polarizer along the entrance rubbing), the Gooch-Tarry formula
        T = sin^2( (pi/2) sqrt(1 + u^2) ) / (1 + u^2),   u = 2 d dn / lambda
    valid for a 90 deg twist. T -> 0 at the Gooch-Tarry minima u = sqrt(3), sqrt(15), ... (the first
    transmission minimum), and the cell is dark there (the waveguiding regime). For a general twist this is
    an approximation (the exact result needs the 4x4/Jones twist matrix); a warning regime only."""
    u = mauguin_number(d_lc, dn, wavelength, twist_total_rad=twist_total_rad)
    arg = 0.5 * math.pi * math.sqrt(1.0 + u * u)
    return float(math.sin(arg) ** 2 / (1.0 + u * u))


@dataclass
class LCChiralResult:
    """Static tilt-twist director solution: theta(z) (tilt from the field/z axis) AND phi(z) (azimuthal
    twist about z). Field-axis convention (theta = 0 homeotropic, pi/2 planar)."""
    V_app: float
    V_lc: float
    z_m: np.ndarray
    theta_field_rad: np.ndarray   # tilt from the field axis (0 = homeotropic, pi/2 = planar)
    phi_rad: np.ndarray           # azimuthal twist about z
    theta_b_rad: float
    phi_bottom_rad: float
    phi_top_rad: float
    twist_energy_J_m2: float = 0.0
    n_eff: float = float("nan")
    success: bool = True
    message: str = ""


def chiral_director_profile_bvp(*, K11: float, K22: float, K33: float, eps_para: float, eps_perp: float,
                                V_app: float, d_planar: float, nz: int = 201,
                                theta_b_rad: float = math.radians(89.9),
                                phi_bottom_rad: float = 0.0, phi_top_rad: float = 0.5 * math.pi,
                                pitch_m: "Optional[float]" = None, q0_rad_m: float = 0.0,
                                field_model: str = "uniform", t_in: float = 0.0, t_out: float = 0.0,
                                eps_in: float = 7.5, eps_out: float = 7.5,
                                n_o: "Optional[float]" = None, n_e: "Optional[float]" = None,
                                opt_model: str = "extra_k_radial", rtol: float = 1e-6) -> LCChiralResult:
    """Three-constant (K11 splay / K22 TWIST / K33 bend) static director BVP for a CHIRAL / TWISTED-nematic
    planar cell, solving the COUPLED tilt theta(z) and azimuthal twist phi(z). Field-axis theta; director
    n = (sin th cos ph, sin th sin ph, cos th). The Frank free-energy density for z-only variation is

        f = (1/2) f1(th) th'^2 + (1/2) f2(th) ph'^2 - K22 q0 sin^2(th) ph'
              + (1/2) eps0 dEps E(z)^2 sin^2(th)
        f1 = K11 sin^2 th + K33 cos^2 th        (the SAME tilt K_eff as director_profile_bvp)
        f2 = sin^2 th (K22 sin^2 th + K33 cos^2 th)

    with q0 = 2 pi / pitch the natural cholesteric wavenumber (q0_rad_m, or pitch_m). The dielectric torque
    acts only on theta (the field is along z; twist preserves the polar angle), so phi couples to theta
    purely elastically. Euler-Lagrange (a 4-D first-order BVP on u = z/d):

        d/dz(f1 th') = (1/2) f1'(th) th'^2 - (1/2) f2'(th) ph'^2 + K22 q0 sin(2th) ph' - eps0 dEps E^2 sin th cos th
        d/dz(f2 ph' - K22 q0 sin^2 th) = 0      (=> f2 ph' - K22 q0 sin^2 th = const)

    Strong anchoring: theta(0)=theta(d)=theta_b, phi(0)=phi_bottom, phi(d)=phi_top. Reduces EXACTLY to
    director_profile_bvp when phi_top=phi_bottom and q0=0 (the twist decouples and phi == const). n_eff is
    the on-axis OPL average via theta (NOTE: a real twisted cell rotates the polarization -- use
    gooch_tarry_transmission / a Jones-matrix optic for the true twisted-cell response; n_eff is only the
    tilt-birefringence proxy)."""
    from scipy.integrate import solve_bvp
    K11 = float(K11); K22 = float(K22); K33 = float(K33)
    dEps = float(eps_para) - float(eps_perp)
    if not (K11 > 0 and K22 > 0 and K33 > 0):
        raise ValueError("K11, K22, K33 must be > 0")
    if not (d_planar and d_planar > 0):
        raise ValueError("d_planar must be > 0 (chiral solver is planar-only)")
    geo = compute_lc_geometry(geometry="planar", nz=nz, d_planar=d_planar, t_in=t_in, t_out=t_out)
    z = geo.z_m; d_lc = float(geo.d_lc)
    theta_b = float(theta_b_rad)
    phi_a = float(phi_bottom_rad); phi_b = float(phi_top_rad)
    q0 = float(q0_rad_m) if abs(q0_rad_m) > 0.0 else (cholesteric_q0(pitch_m) if pitch_m else 0.0)
    fkw = dict(eps_para=eps_para, eps_perp=eps_perp, field_model=field_model, t_in=t_in, t_out=t_out,
               eps_in=eps_in, eps_out=eps_out)
    u = (z - float(z[0])) / d_lc

    def _f1(th):
        s = np.sin(th); c = np.cos(th)
        return K11 * s * s + K33 * c * c

    def _f1p(th):                                     # df1/dtheta = (K11-K33) sin(2 theta)
        return (K11 - K33) * np.sin(2.0 * th)

    def _f2(th):
        s2 = np.sin(th) ** 2; c2 = np.cos(th) ** 2
        return s2 * (K22 * s2 + K33 * c2)

    def _f2p(th):                                     # df2/dtheta = sin(2th)[2 K22 sin^2 + K33 cos(2th)]
        return np.sin(2.0 * th) * (2.0 * K22 * np.sin(th) ** 2 + K33 * np.cos(2.0 * th))

    F2_FLOOR = 1e-3 * K33                             # phi ill-defined where the cell is homeotropic (sin th->0)

    def _field_at(theta, Vk):
        return solve_lc_field_profile(theta, float(Vk), geo, **fkw)

    def bc(ya, yb):
        return np.array([ya[0] - theta_b, yb[0] - theta_b,
                         ya[2] - phi_a, yb[2] - phi_b], dtype=float)

    def _make_fun(Vk):
        def fun(x, y):
            uu = np.asarray(x, float); zz = float(z[0]) + uu * d_lc
            th = np.asarray(y[0], float); th_u = np.asarray(y[1], float)
            ph_u = np.asarray(y[3], float)
            th_z = th_u / d_lc; ph_z = ph_u / d_lc
            gloc = LCGeometry(geo.geometry, d_lc, zz, zz, geo.r_in, geo.r_out)
            E, _v = solve_lc_field_profile(th, float(Vk), gloc, **fkw)
            f1 = np.maximum(np.abs(_f1(th)), 1e-300)
            f2 = np.where(np.abs(_f2(th)) < F2_FLOOR, F2_FLOOR, _f2(th))
            s = np.sin(th); c = np.cos(th); s2 = np.sin(2.0 * th)
            # theta EL (d/dz(dL/dth') - dL/dth = 0, L the Frank+dielectric free-energy density):
            #   f1 th'' = -(1/2) f1' th'^2 + (1/2) f2' ph'^2 - K22 q0 sin(2th) ph' + eps0 dEps E^2 s c
            # (reduces to director_profile_bvp's d2 = -(t2+t3)/Keff at ph'=0, q0=0).
            rhs_th = (-0.5 * _f1p(th) * th_z * th_z + 0.5 * _f2p(th) * ph_z * ph_z
                      - K22 * q0 * s2 * ph_z + EPS0 * dEps * (E * E) * s * c)
            th_zz = rhs_th / f1
            # phi EL: f2 ph'' = - f2'(th) th' ph' + K22 q0 sin(2th) th'
            ph_zz = (-_f2p(th) * th_z * ph_z + K22 * q0 * s2 * th_z) / f2
            return np.vstack((th_u, th_zz * d_lc * d_lc, ph_u, ph_zz * d_lc * d_lc))
        return fun

    ph_seed = phi_a + (phi_b - phi_a) * u + (q0 * d_lc) * (u - u * u)   # linear + chiral bow (fixed BCs)
    Vmag = abs(float(V_app)); sgn_V = 1.0 if float(V_app) >= 0.0 else -1.0
    K_eff_mean = 0.5 * (K11 + K33)
    V_th = (math.pi * math.sqrt(K_eff_mean / (EPS0 * abs(dEps)))) if abs(dEps) > 1e-30 else float("inf")
    sign = -1.0 if dEps >= 0.0 else 1.0                                # tilt direction toward the field
    reach = theta_b if dEps >= 0.0 else (0.5 * math.pi - theta_b)
    max_amp = min(abs(reach) * 0.98, math.radians(89.0))
    direction = -1.0 if dEps >= 0.0 else 1.0
    use_score = Vmag > 1e-14 and abs(dEps) > 1e-14

    def _theta_seeds(Vk, carried):
        gs = []
        if carried is not None:
            cc = np.array(carried, float); cc[0] = cc[-1] = theta_b
            gs.append(np.clip(cc, -0.49 * math.pi, 0.999 * math.pi))    # CONTINUATION
        df = (math.tanh(max(0.0, abs(Vk) / V_th - 0.85)) if (math.isfinite(V_th) and V_th > 0) else 0.5)
        flat = np.full_like(z, theta_b)
        for frac in (df, 0.3, 0.6, 0.85, 0.98):
            g = np.clip(flat + sign * (max_amp * max(0.0, min(1.0, frac))) * np.sin(math.pi * u),
                        -0.49 * math.pi, 0.999 * math.pi)
            g[0] = g[-1] = theta_b
            gs.append(g)
        gs.append(flat)                                                # trivial fallback
        uniq: List[np.ndarray] = []
        for g in gs:
            if not any(o.shape == g.shape and np.allclose(o, g, atol=1e-9) for o in uniq):
                uniq.append(g)
        return uniq

    def _twist_energy(th_, ph_):
        ph_z = np.gradient(ph_, z)
        return float(trapz(0.5 * _f2(th_) * ph_z * ph_z - K22 * q0 * np.sin(th_) ** 2 * ph_z, z))

    def _energy(th_, ph_, Vk):
        try:
            dth = np.gradient(th_, z)
            E, _v = _field_at(th_, Vk)
            dens = 0.5 * _f1(th_) * dth * dth + 0.5 * EPS0 * dEps * (E * E) * np.sin(th_) ** 2
            return float(trapz(dens, z)) + _twist_energy(th_, ph_)
        except Exception:
            return float("inf")

    def _solve_at(th_guess, Vk):
        y0 = np.vstack((th_guess, np.gradient(th_guess, u), ph_seed, np.gradient(ph_seed, u)))
        with warnings.catch_warnings(), np.errstate(all="ignore"):   # silence failed-seed Jacobian noise
            warnings.simplefilter("ignore")
            sol = solve_bvp(_make_fun(sgn_V * abs(Vk)), bc, u, y0, tol=max(float(rtol), 3e-2),
                            max_nodes=max(6000, int(nz) * 100), verbose=0)
        if not sol.success:
            raise RuntimeError(str(sol.message))
        th_ = np.asarray(sol.sol(u)[0], float); ph_ = np.asarray(sol.sol(u)[2], float)
        th_[0] = th_[-1] = theta_b; ph_[0] = phi_a; ph_[-1] = phi_b
        return th_, ph_

    def _best_at(Vk, carried):
        cands = []
        last = ""
        for g in _theta_seeds(Vk, carried):
            try:
                th_, ph_ = _solve_at(g, Vk)
                score = direction * float(np.nanmean(th_ - theta_b)) if use_score else 0.0
                cands.append((score, _energy(th_, ph_, sgn_V * abs(Vk)), th_, ph_))
            except Exception as exc:
                last = str(exc)
        if not cands:
            raise RuntimeError("no converged candidate (last: {})".format(last))
        cands.sort(key=lambda it: (-it[0], it[1]))
        return cands[0][2], cands[0][3]

    if (not use_score) or (not math.isfinite(V_th)) or Vmag <= 1.02 * V_th:
        ladder = [Vmag]
    else:
        n_steps = int(min(14, max(2, math.ceil((Vmag - V_th) / max(0.75, 0.5 * V_th)) + 1)))
        ladder = list(np.linspace(min(1.05 * V_th, Vmag), Vmag, n_steps))
    carried = None; success = True; message = "ok"
    try:
        for Vk in ladder:
            carried, ph = _best_at(float(Vk), carried)
        th = carried
    except RuntimeError as exc:
        # last-resort single solve at the requested voltage (keeps a result + flags the miss)
        th, ph = _solve_at(np.full_like(z, theta_b) + sign * 0.5 * max_amp * np.sin(math.pi * u), Vmag)
        success = False; message = str(exc)
    # wrong-branch safety net (same as director_profile_bvp): above 1.3 V_th the cell MUST tilt
    if use_score and math.isfinite(V_th) and Vmag > 1.3 * V_th:
        tilt = direction * float(np.mean(th - theta_b))
        if tilt < math.radians(1.0):
            success = False
            message = ("chiral_director_profile_bvp: V={:.3g} > 1.3 V_th(~{:.3g}) but stayed ~untilted "
                       "(mean tilt {:.3g} rad)".format(Vmag, V_th, tilt))
    _E, V_lc = _field_at(th, float(V_app))
    # twist elastic energy per unit plate area: integral of [ (1/2) f2 ph'^2 - K22 q0 sin^2 th ph' ] dz
    ph_z = np.gradient(ph, z)
    w_twist = float(trapz(0.5 * _f2(th) * ph_z * ph_z - K22 * q0 * np.sin(th) ** 2 * ph_z, z))
    neff = (n_eff_from_theta_profile(th, z, n_o, n_e, model=opt_model, d_lc=d_lc)
            if (n_o is not None and n_e is not None) else float("nan"))
    return LCChiralResult(V_app=float(V_app), V_lc=float(V_lc), z_m=z, theta_field_rad=th, phi_rad=ph,
                          theta_b_rad=theta_b, phi_bottom_rad=phi_a, phi_top_rad=phi_b,
                          twist_energy_J_m2=w_twist, n_eff=float(neff), success=success,
                          message=message)
