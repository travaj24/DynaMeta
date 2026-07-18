"""Large-mode-area (LMA) step-index fiber physics (docs sec.8): the scalar LP-mode solver, the
per-mode field / dopant overlap, the Marcuse macro-bend loss that makes coiling a mode filter,
and the cladding-pump absorption efficiency beyond the naive area ratio (skew rays / two-
population mixing). These are the DEVICE-REALISM inputs a high-power double-clad Yb amplifier
needs on top of the fundamental-mode waveguide of `waveguide.py`:

  * solve_lp_modes -- ALL guided LP_lm modes of a weakly-guiding step-index fiber from the scalar
    (Gloge) dispersion relation U J_{l-1}(U)/J_l(U) = -W K_{l-1}(W)/K_l(W), W = sqrt(V^2-U^2).
    LP01 always exists; the guided-mode count grows as ~V^2/2. This is the mode content that
    a LMA core (V ~ 3-6) supports and that bend loss must strip down to LP01.
  * mode_field / dopant_overlap -- the exact LP field psi(r) (J_l core, K_l cladding) and its
    power overlap with a top-hat dopant of radius b, the per-mode Gamma_lm that feeds a
    mode-competition gain model; the LP01 result is cross-checked against the Gaussian-mode
    overlap_gamma of waveguide.py.
  * marcuse_bend_loss_per_m -- Marcuse's macro-bend power-loss coefficient 2*alpha (Marcuse,
    JOSA 66:216, 1976; reproduction form of Schermer & Cole, IEEE JQE 43:899, 2007), with the
    elasto-optic effective bend radius R_eff = 1.27 R. The bend loss rises steeply with mode
    order (the exponent scales as gamma^3 = W^3), so a coil that gives LP01 a fraction of a dB/m
    gives LP11 tens of dB/m -- the physical basis of coiled-fiber single-mode operation
    (Koplow & Kliner, Opt. Lett. 25:442, 2000).
  * pump_absorption_efficiency / effective_cladding_overlap -- the geometric efficiency factor
    eta_geo that corrects the area-ratio pump overlap for skew (helical) rays that never cross a
    centered circular core; broken symmetry (D-shape, octagon, offset core) or coiling recovers
    the ideal (Kouznetsov & Moloney, JOSA B 19:1259 & 1304, 2002; Bedo/Luthy/Weber, Opt.
    Commun. 99:331, 1993).
  * cladding_absorption_two_population -- the crossing / non-crossing two-population pump model
    with a mode-mixing rate g_mix: the pump absorbs at only the crossing fraction until mixing
    (g_mix L >> 1) redistributes the skew power into the absorbing population.

Pure numpy / scipy; SI units; ASCII only. exp(-i omega t) convention (unused here -- all real
scalar-mode quantities). docs/fiber_amp_model_spec.md sec.8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, kv

from dynameta.optics.fiber_amp.waveguide import FiberSpec, cladding_pump_overlap

__all__ = [
    "LPMode",
    "ModeOverlap",
    "solve_lp_modes",
    "mode_degeneracy",
    "total_mode_count",
    "mode_field",
    "dopant_overlap",
    "second_moment_radius_m",
    "one_over_e_radius_m",
    "effective_area_m2",
    "marcuse_bend_loss_per_m",
    "marcuse_bend_loss_dB_per_m",
    "pump_absorption_efficiency",
    "effective_cladding_overlap",
    "cladding_absorption_two_population",
    "mode_resolved_gain_overlaps",
]

_DB_PER_NEPER = 10.0 / np.log(10.0)  # power 1/m -> dB/m (P = P0 exp(-2 alpha z))


# ============================ LP-mode solver ============================================

@dataclass(frozen=True)
class LPMode:
    """A guided scalar LP_lm mode of a step-index fiber. l = azimuthal order (0, 1, 2, ...),
    m = radial order (1, 2, ...); U = a*kappa = a sqrt(n_core^2 k0^2 - beta^2) is the normalized
    transverse core wavenumber, W = a*gamma = a sqrt(beta^2 - n_clad^2 k0^2) the cladding decay
    parameter, with U^2 + W^2 = V^2; beta = propagation constant [1/m]; V = normalized frequency.
    core_radius_m (a) and lambda_m are retained so the field / overlap / bend-loss helpers need
    only the mode object. LP01 has the largest beta (best confinement); degeneracy: l>=1 modes
    are 2-fold (cos/sin) x 2 polarizations, l=0 is 2-fold (polarization) -- not expanded here."""
    l: int
    m: int
    U: float
    W: float
    beta: float
    V: float
    core_radius_m: float
    lambda_m: float

    @property
    def kappa(self) -> float:
        """Transverse core wavenumber kappa = U / a [1/m]."""
        return self.U / self.core_radius_m

    @property
    def gamma(self) -> float:
        """Cladding field decay rate gamma = W / a [1/m] (amplitude ~ exp(-gamma r))."""
        return self.W / self.core_radius_m

    def n_eff(self, lambda_m=None) -> float:
        """Effective index n_eff = beta / k0 (lies between n_clad and n_core)."""
        lam = self.lambda_m if lambda_m is None else float(lambda_m)
        return float(self.beta / (2.0 * np.pi / lam))


def _characteristic(U: float, l: int, V: float) -> float:
    """Denominator-cleared LP dispersion function whose zeros in 0<U<V are the guided modes:
        g(U) = U J_{l-1}(U) K_l(W) + W K_{l-1}(W) J_l(U),   W = sqrt(V^2 - U^2).
    Multiplying the ratio form U J_{l-1}/J_l = -W K_{l-1}/K_l through by J_l(U) K_l(W) removes the
    poles at zeros of J_l, so g is smooth and its sign changes are exactly the true eigenvalues
    (K_l(W) > 0 has no sign change; J_l(U) keeps a fixed sign between its zeros). scipy's Bessel
    functions honour J_{-1} = -J_1 and K_{-1} = K_1, so l=0 needs no special handling."""
    W = np.sqrt(max(V * V - U * U, 0.0))
    return (U * jv(l - 1, U) * kv(l, W) + W * kv(l - 1, W) * jv(l, U))


def solve_lp_modes(core_radius_m: float, na: float, lambda_m: float,
                   n_clad: float = 1.45) -> List[LPMode]:
    """All guided LP_lm modes of a weakly-guiding step-index fiber, from the scalar dispersion
    relation U J_{l-1}(U)/J_l(U) = -W K_{l-1}(W)/K_l(W) with W = sqrt(V^2 - U^2) and
    V = k0 a NA (Gloge, Appl. Opt. 10:2252, 1971). For each azimuthal order l the ratio form has
    poles at the zeros of J_l, so the roots are bracketed between consecutive Bessel zeros (from
    scipy jn_zeros) and located with brentq on the denominator-cleared characteristic; the radial
    index m = 1, 2, ... numbers the roots by increasing U. LP01 exists for any V > 0; higher
    LP_lm turn on above their cutoff (LP11 at V=2.405, the single-mode boundary), and the total
    count approaches ~V^2/2 for large V. Modes are returned sorted by beta descending (LP01 first).

    Parameters: core_radius_m = core radius a [m]; na = numerical aperture sqrt(n_core^2-n_clad^2);
    lambda_m = vacuum wavelength [m]; n_clad = cladding refractive index (n_core = sqrt(n_clad^2 +
    NA^2)). Returns a list of LPMode."""
    a = float(core_radius_m)
    na = float(na)
    lam = float(lambda_m)
    if not (a > 0.0 and na > 0.0 and lam > 0.0):
        raise ValueError("solve_lp_modes: core_radius_m, na, lambda_m must be > 0")
    k0 = 2.0 * np.pi / lam
    n_core = np.sqrt(n_clad * n_clad + na * na)
    V = k0 * a * na

    modes: List[LPMode] = []
    l = 0
    while True:
        # Zeros of J_l (poles of the ratio form) that bracket the roots in (0, V).
        n_zeros = int(V / np.pi) + 4
        zeros = jn_zeros(l, n_zeros)
        bounds = [0.0] + [float(z) for z in zeros if z < V] + [V]

        found_this_l = 0
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            width = hi - lo
            if width <= 0.0:
                continue
            inset = width * 1e-6
            # Fine sub-scan for the (single) sign change inside the interval; brentq refines it.
            us = np.linspace(lo + inset, hi - inset, 40)
            gs = np.array([_characteristic(u, l, V) for u in us])
            finite = np.isfinite(gs)
            root = None
            for i in range(len(us) - 1):
                if not (finite[i] and finite[i + 1]):
                    continue
                if gs[i] == 0.0:
                    root = us[i]
                    break
                if gs[i] * gs[i + 1] < 0.0:
                    root = brentq(_characteristic, us[i], us[i + 1], args=(l, V),
                                  xtol=1e-12, rtol=1e-13)
                    break
            if root is None:
                continue
            U = float(root)
            W = float(np.sqrt(max(V * V - U * U, 0.0)))
            beta = float(np.sqrt((n_core * k0) ** 2 - (U / a) ** 2))
            found_this_l += 1
            modes.append(LPMode(l=l, m=found_this_l, U=U, W=W, beta=beta, V=float(V),
                                core_radius_m=a, lambda_m=lam))

        # LP01 (l=0) is guaranteed; for l>=1, once no LP_l1 exists (V below its cutoff, the first
        # zero of J_{l-1}) no higher l can either, since cutoffs increase with l.
        if l >= 1 and found_this_l == 0:
            break
        l += 1
        if l > n_zeros + 8:  # hard guard against runaway (never reached in practice)
            break

    modes.sort(key=lambda md: md.beta, reverse=True)
    return modes


def mode_degeneracy(mode: LPMode) -> int:
    """Total degeneracy of an LP_lm label: 2 for l=0 (two polarizations) and 4 for l>=1 (two
    polarizations x the cos(l*phi)/sin(l*phi) orientation pair). Summing this over the guided LP
    modes gives the physical mode volume, which approaches V^2/2 for large V."""
    return 2 if mode.l == 0 else 4


def total_mode_count(modes: List[LPMode]) -> int:
    """Degeneracy-weighted number of guided modes (the mode volume) = sum of mode_degeneracy over
    the distinct LP_lm labels. Approaches V^2/2 for large V (e.g. ~54 vs 50 at V=10)."""
    return int(sum(mode_degeneracy(md) for md in modes))


# ============================ LP field, overlap, effective area =========================

def mode_field(mode: LPMode, r_m) -> np.ndarray:
    """Normalized scalar LP field psi(r) (continuous, psi(a)=1):
        core   r <= a : J_l(U r/a) / J_l(U)
        clad   r  > a : K_l(W r/a) / K_l(W)
    Returns a float for scalar r, else an array. The K_l cladding tail decays as
    exp(-W r/a)/sqrt(r), so the mode is well localized for W not too small."""
    a = mode.core_radius_m
    U, W, l = mode.U, mode.W, mode.l
    r = np.atleast_1d(np.asarray(r_m, dtype=np.float64))
    out = np.empty_like(r)
    core = r <= a
    out[core] = jv(l, U * r[core] / a) / jv(l, U)
    cl = ~core
    out[cl] = kv(l, W * r[cl] / a) / kv(l, W)
    return out if np.ndim(r_m) else float(out[0])


def _overlap_grid(mode: LPMode, extra: float = 12.0) -> np.ndarray:
    """Radial grid dense enough to resolve the core and capture the cladding tail. Extends to
    several 1/e cladding decay lengths a/W past the core (and at least 6a)."""
    a = mode.core_radius_m
    r_max = max(6.0 * a, a * (1.0 + extra / max(mode.W, 0.25)))
    return np.linspace(0.0, r_max, 6000)


def dopant_overlap(mode: LPMode, r_dope_m: float) -> float:
    """Per-mode power overlap Gamma_lm with a top-hat dopant of radius b = r_dope_m:
        Gamma_lm = int_0^b |psi|^2 r dr / int_0^inf |psi|^2 r dr
    by numerical quadrature of the exact LP field (the 2*pi azimuthal factor cancels). This is
    the fraction of the mode power that sees the ions, i.e. the per-mode confinement factor a
    mode-competition gain model consumes. For the LP01 mode with b = a it reproduces the
    Gaussian-approximation overlap_gamma of waveguide.py to a few percent."""
    b = float(r_dope_m)
    r = _overlap_grid(mode)
    if b > r[-1]:  # dopant wider than the grid -> extend so the whole tail is captured
        r = np.linspace(0.0, 1.05 * b, 8000)
    psi = mode_field(mode, r)
    integrand = psi * psi * r
    cum = cumulative_trapezoid(integrand, r, initial=0.0)
    denom = cum[-1]
    numer = float(np.interp(min(b, r[-1]), r, cum))
    return float(numer / denom)


def second_moment_radius_m(mode: LPMode) -> float:
    """Second-moment (Petermann-I-like) field radius w = sqrt(2 <r^2>), with
        <r^2> = int |psi|^2 r^2 (r dr) / int |psi|^2 (r dr).
    For a Gaussian psi = exp(-r^2/w^2) this returns exactly the 1/e field radius w, so it is the
    natural point of comparison with the Marcuse Gaussian mode-field radius."""
    r = _overlap_grid(mode, extra=16.0)
    psi = mode_field(mode, r)
    w0 = psi * psi * r
    r2 = trapezoid(w0 * r * r, r) / trapezoid(w0, r)
    return float(np.sqrt(2.0 * r2))


def one_over_e_radius_m(mode: LPMode) -> float:
    """Radius at which the field amplitude |psi| falls to 1/e of its peak (found by linear
    interpolation on a dense grid). For LP01 the peak is at r=0; higher modes peak off-axis."""
    r = _overlap_grid(mode, extra=16.0)
    psi = np.abs(mode_field(mode, r))
    ipk = int(np.argmax(psi))
    target = psi[ipk] / np.e
    below = np.where(psi[ipk:] <= target)[0]
    if below.size == 0:
        return float(r[-1])
    j = ipk + int(below[0])
    if j == 0:
        return float(r[0])
    r0, r1, p0, p1 = r[j - 1], r[j], psi[j - 1], psi[j]
    return float(r0 + (target - p0) * (r1 - r0) / (p1 - p0))


def effective_area_m2(mode: LPMode) -> float:
    """Nonlinear effective area A_eff = (int |psi|^2 dA)^2 / int |psi|^4 dA
    = 2 pi (int psi^2 r dr)^2 / (int psi^4 r dr) [m^2] (the area that sets the Kerr/damage
    intensity). For the LP01 mode A_eff ~ pi w^2 with w the mode-field radius."""
    r = _overlap_grid(mode, extra=16.0)
    psi = mode_field(mode, r)
    num = (trapezoid(psi * psi * r, r)) ** 2
    den = trapezoid(psi ** 4 * r, r)
    return float(2.0 * np.pi * num / den)


# ============================ Marcuse macro-bend loss ===================================

def marcuse_bend_loss_per_m(mode: LPMode, core_radius_m: float, bend_radius_m: float, *,
                            elasto_optic_factor: float = 1.27) -> float:
    """Marcuse macro-bend POWER loss coefficient 2*alpha [1/m] (P(z) = P0 exp(-2 alpha z)) for a
    step-index LP mode coiled to radius R (Marcuse, JOSA 66:216, 1976; the reproduction form of
    Schermer & Cole, IEEE JQE 43:899, 2007):

        2*alpha = sqrt(pi) kappa^2 exp( -(2/3)(gamma^3/beta^2) R_eff )
                  / [ 2 gamma^(3/2) V^2 sqrt(R_eff) K_{l-1}(gamma a) K_{l+1}(gamma a) ]

    with kappa = U/a, gamma = W/a, and the elasto-optic effective bend radius
    R_eff = elasto_optic_factor * R (chi = (n^2/2)(p12 - nu(p11+p12)) ~ 0.214 for silica ->
    R_eff ~ 1.27 R). LP01 uses K_{-1} = K_{+1} = K_1 -> K_1(gamma a)^2 in the denominator; the
    general l uses K_{l-1} K_{l+1}. The loss is dominated by the EXPONENT ~ exp(-const * W^3 R),
    which grows steeply with mode order (larger W) -- the reason a coil that barely touches LP01
    strips the higher-order modes (Koplow & Kliner, Opt. Lett. 25:442, 2000)."""
    a = float(core_radius_m)
    if not (bend_radius_m > 0.0 and elasto_optic_factor > 0.0):
        raise ValueError("marcuse_bend_loss_per_m: bend_radius_m and elasto_optic_factor must be > 0")
    l, U, W, beta, V = mode.l, mode.U, mode.W, mode.beta, mode.V
    kappa = U / a
    gamma = W / a
    ga = W  # gamma * a
    R_eff = float(elasto_optic_factor) * float(bend_radius_m)
    Km1 = kv(abs(l - 1), ga)     # K_{l-1}; abs() gives K_1 for l=0 since K_{-1} = K_1
    Kp1 = kv(l + 1, ga)          # K_{l+1}
    expo = -(2.0 / 3.0) * (gamma ** 3 / beta ** 2) * R_eff
    num = np.sqrt(np.pi) * kappa ** 2 * np.exp(expo)
    den = 2.0 * gamma ** 1.5 * V ** 2 * np.sqrt(R_eff) * Km1 * Kp1
    return float(num / den)


def marcuse_bend_loss_dB_per_m(mode: LPMode, core_radius_m: float, bend_radius_m: float, *,
                               elasto_optic_factor: float = 1.27) -> float:
    """Marcuse macro-bend loss expressed in dB/m = (2*alpha) * 10/ln(10)."""
    return _DB_PER_NEPER * marcuse_bend_loss_per_m(
        mode, core_radius_m, bend_radius_m, elasto_optic_factor=elasto_optic_factor)


# ============================ Cladding-pump absorption ==================================

# eta_geo: geometric pump-absorption efficiency multiplying the ideal area ratio A_dope/A_clad.
# A perfectly centered circular inner cladding traps skew (helical) rays that never cross the
# core; breaking that symmetry (offset core, D-shape, polygonal cladding) or coiling the fiber
# mixes the rays and recovers the ideal. Values are representative mid-range points; the
# documented range for each geometry is given in the comment (dossier Module 2; Kouznetsov &
# Moloney, JOSA B 19:1259 & 1304, 2002; Opt. Lett. 26:872, 2001; Bedo et al., Opt. Commun.
# 99:331, 1993).
_ETA_GEO = {
    "circular_centered": 0.40,   # 0.3-0.5: worst case, symmetric skew rays never reach the core
    "offset": 0.80,              # ~0.8: core displaced from the cladding axis breaks the symmetry
    "d_shape": 0.95,             # 0.9-1.0: flat truncates the closed skew-ray orbits
    "octagonal": 0.93,           # 0.9-0.97: polygonal cladding scrambles helical rays
    "hexagonal": 0.97,           # 0.95-1.0: polygonal cladding, near-ideal
    "rectangular": 0.90,         # ~0.9: two flats mix both transverse axes
    "circular_coiled": 1.00,     # ->1.0: coiling/twist breaks meridional invariance -> ideal
}


def pump_absorption_efficiency(geometry: str = "octagonal") -> float:
    """Geometric cladding-pump absorption efficiency eta_geo (dimensionless, in (0,1]) for the
    inner-cladding cross-section 'geometry'. It corrects the naive area-ratio pump overlap
    A_dope/A_clad for skew (helical) rays that, in a perfectly centered circular cladding, orbit
    without ever crossing the doped core. Only the centered circle is badly degraded (0.3-0.5);
    breaking the symmetry (offset core ~0.8; D-shape / polygonal 0.9-1.0) or coiling (->1.0)
    recovers the ideal. Supported keys: 'circular_centered', 'offset', 'd_shape', 'octagonal',
    'hexagonal', 'rectangular', 'circular_coiled'."""
    key = str(geometry).lower()
    if key not in _ETA_GEO:
        raise ValueError("pump_absorption_efficiency: unknown geometry {!r}; choose from {}".format(
            geometry, sorted(_ETA_GEO)))
    return float(_ETA_GEO[key])


def effective_cladding_overlap(fiber: FiberSpec, geometry: str = "octagonal") -> float:
    """Effective double-clad pump overlap with the doped region, corrected for cladding geometry:
        Gamma_p_eff = eta_geo(geometry) * (b_dope / clad_radius)^2 = eta_geo * cladding_pump_overlap.
    It matches the convention of waveguide.cladding_pump_overlap exactly -- the area ratio uses
    the DOPANT radius b_dope (not the core radius), the power fraction inside the dopant that the
    solver forms as Gamma_p P / A_dope. With geometry='circular_coiled' (eta_geo = 1.0) it reduces
    identically to cladding_pump_overlap(fiber); any other geometry scales it down by eta_geo. For
    a core-pumped fiber (clad_radius_m is None) cladding_pump_overlap returns 1.0 and this returns
    eta_geo."""
    return float(pump_absorption_efficiency(geometry) * cladding_pump_overlap(fiber))


def cladding_absorption_two_population(alpha_ideal_per_m: float, g_mix_per_m: float,
                                       length_m: float, f_cross: float) -> float:
    """Crossing / non-crossing two-population cladding-pump absorption (dossier Module 2, Model B).
    The pump splits into a CROSSING population (equilibrium fraction f_cross = A_core-crossing
    fraction) that intersects the doped core and a NON-CROSSING (skew) population that misses it;
    a mode-mixing rate g_mix [1/m] (gentle coiling/perturbation) redistributes power between them:

        dPc/dz = -alpha_core Pc - g_mix (Pc - f_cross (Pc+Ps))
        dPs/dz =                  + g_mix (Pc - f_cross (Pc+Ps))

    Only the crossing population is absorbed, at the intrinsic core rate alpha_core =
    alpha_ideal / f_cross, chosen so that in the WELL-MIXED limit (g_mix L >> 1, Pc pinned at its
    equilibrium fraction f_cross of the total) the total decays at the ideal cladding-averaged
    rate alpha_ideal. The pump is launched at the equilibrium split Pc(0)=f_cross, Ps(0)=1-f_cross.
    The 2x2 linear system is solved in closed form by eigen-decomposition (matrix exponential).

    Returns the effective absorbed fraction 1 - (Pc(L)+Ps(L))/P0. Limits:
      * g_mix L >> 1 -> 1 - exp(-alpha_ideal L)  (the ideal absorbed fraction);
      * g_mix = 0    -> f_cross (1 - exp(-alpha_core L)) -> f_cross for alpha_core L >> 1: without
        mixing only the crossing fraction is ever absorbed."""
    alpha_ideal = float(alpha_ideal_per_m)
    g = float(g_mix_per_m)
    L = float(length_m)
    fc = float(f_cross)
    if not (0.0 < fc <= 1.0):
        raise ValueError("cladding_absorption_two_population: f_cross must be in (0, 1]")
    if alpha_ideal < 0.0 or g < 0.0 or L < 0.0:
        raise ValueError("cladding_absorption_two_population: alpha, g_mix, length must be >= 0")

    alpha_core = alpha_ideal / fc
    # d/dz [Pc, Ps]^T = M [Pc, Ps]^T
    M = np.array([
        [-(alpha_core + g * (1.0 - fc)), g * fc],
        [g * (1.0 - fc), -g * fc],
    ], dtype=np.float64)
    p0 = np.array([fc, 1.0 - fc], dtype=np.float64)   # equilibrium launch, total P0 = 1

    # Closed-form exp(M L) p0 via eigen-decomposition (M is a real 2x2 with real eigenvalues here).
    w, Vv = np.linalg.eig(M)
    coeff = np.linalg.solve(Vv, p0)
    pL = Vv @ (np.exp(w * L) * coeff)
    total_L = float(np.real(pL).sum())
    return float(1.0 - total_L)


# ============================ Mode-resolved gain overlaps ===============================

@dataclass(frozen=True)
class ModeOverlap:
    """Per-mode dopant overlap for a mode-competition gain model: azimuthal l, radial m, and the
    power confinement factor gamma = Gamma_lm (fraction of the mode power inside the dopant)."""
    l: int
    m: int
    gamma: float


def mode_resolved_gain_overlaps(fiber_core_radius: float, na: float, lambda_m: float,
                                r_dope: float, n_clad: float = 1.45) -> List[ModeOverlap]:
    """Per-mode confinement factors Gamma_lm for every guided LP mode of a step-index fiber:
    solve_lp_modes followed by dopant_overlap against a top-hat dopant of radius r_dope. Returns
    a list of ModeOverlap in the same (beta-descending, LP01 first) order as solve_lp_modes -- the
    set of per-mode overlaps a downstream mode-competition / transverse-mode-instability model
    needs. Higher-order modes have smaller Gamma_lm (their power spreads further into the
    cladding), which together with their much larger bend loss is what a coiled LMA exploits."""
    modes = solve_lp_modes(fiber_core_radius, na, lambda_m, n_clad=n_clad)
    return [ModeOverlap(l=md.l, m=md.m, gamma=dopant_overlap(md, r_dope)) for md in modes]
