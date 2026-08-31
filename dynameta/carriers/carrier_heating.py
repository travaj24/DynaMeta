"""
Carrier-heating (two-temperature) ENZ nonlinearity (roadmap R9): the intrinsic ultrafast optical
nonlinearity of a transparent-conductor ENZ film. An absorbed optical pump heats the electron gas to
an electron temperature T_e above the lattice T_l; the hot electrons (i) climb the nonparabolic band so
the f-sum (Drude/conductivity) mass m_c(T_e) RISES (the plasma frequency wp^2 ~ n/m_c drops, so Re(eps)
moves toward eps_inf -- through ENZ), and (ii) scatter more so the Drude damping Gamma(T_e) rises. The
energy flow is the two-temperature model (TTM)

    C_e(T_e) dT_e/dt = -G (T_e - T_l) + alpha_abs I(t)        (electron gas heated by absorbed power)
    C_l      dT_l/dt = +G (T_e - T_l)                          (electron-phonon coupling into the lattice)

with the electron heat capacity C_e(T_e) ~ gamma_e T_e (degenerate gas) giving the characteristic
sub-ps rise / few-ps relaxation asymmetry (Alam-Boyd-class ITO ENZ pump-probe). This is a FREE-CARRIER
Drude effect, so it rides the existing DrudeOptical m_opt_kg / gamma_rad_s callable seam (per-instant
closures capturing T_e(t)) feeding the transient_optics loop -- NOT a chi3 and NOT a lattice dn/dT.

Off-switches (byte-identical to the linear, constant-mass result):
  - intensity I(t) == 0 (or alpha_abs == 0)         -> T_e == T_l == T0 for all t.
  - alpha_per_eV == 0 in kane_mass_of_Te            -> m_c == m0 EXACTLY (no band nonparabolicity).
  - p == 0 in gamma_of_Te                           -> Gamma == gamma0 EXACTLY.
Any of these collapses the per-instant Drude to the constant drude0 at every step. The pump-off
switch holds AT EVERY BASE TEMPERATURE (audit C-7): gamma_of_Te's reference temperature is threaded
from carrier_heating_transient and defaults to T0_K, so Gamma == gamma0 whenever T_e == T_l == T0_K
(it used to be pinned at 300 K, which made the promise FALSE off 300 K -- 0.257*gamma0 at 77 K with
the pump off). Pure numpy/scipy; exp(-i omega t), Im(eps) > 0; SI units (C in J/m^3/K, alpha_abs I
in W/m^3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from scipy.integrate import solve_ivp

from dynameta.constants import Q_E, HBAR, KB, T_REF
from dynameta.core.numerics import ZeroInitBDF
from dynameta.materials.optical_model import DrudeOptical

# T_REF imported from dynameta.constants (audit S1-15)


def fermi_energy_J(n_m3, m0_kg: float, alpha_per_eV: float = 0.0, *, g_s: int = 2, g_v: int = 1):
    """Kane Fermi energy E_F(n) [J] (same inversion as carriers.sp_carrier): gamma_F =
    (hbar^2/2 m0)(6 pi^2 n/(g_s g_v))^(2/3); for alpha>0, E_F = (-1 + sqrt(1 + 4 a gamma_F))/(2 a),
    a = alpha/q; alpha=0 -> E_F = gamma_F (parabolic)."""
    n = np.asarray(n_m3, dtype=np.float64)
    gamma_F = (HBAR ** 2 / (2.0 * m0_kg)) * (6.0 * np.pi ** 2 * n / (g_s * g_v)) ** (2.0 / 3.0)
    if alpha_per_eV == 0.0:
        return gamma_F
    a = float(alpha_per_eV) / Q_E                                  # J^-1
    return (-1.0 + np.sqrt(1.0 + 4.0 * a * gamma_F)) / (2.0 * a)


# ---------------------------------------------------------------------------------------------
# f-sum (Drude / conductivity) mass on the Kane band -- audit C-1
#
# wp^2 = n e^2 / (eps0 m) requires the f-SUM-RULE (conductivity) mass, NOT the mass evaluated at
# the mean energy. The f-sum rule for one isotropic band reads
#
#     n / m_c = (1/3) <Lap_k E / hbar^2>_f ,   i.e.   1/m_c = (1/(3 n)) Int g(E) f(E) [Lap_k E/hbar^2] dE
#
# Everything below evaluates that integral DIRECTLY in k (where the integrand is analytic -- the
# E-space form carries a sqrt(E) endpoint singularity that cripples any fixed quadrature). For the
# Kane band  E (1 + a E) = hbar^2 k^2 / (2 m0) == gamma(k),  a = alpha_per_eV / q, the quadratic
# inversion gives the identity
#
#     D(k) == 1 + 2 a E(k) = sqrt(1 + 4 a gamma(k)),
#     Lap_k E / hbar^2 = (1/m0) [ 3/D - 4 a gamma / D^3 ] ,
#
# so with the isotropic measure d^3k -> k^2 dk the whole thing is the OCCUPATION-WEIGHTED RATIO
#
#     m_c(T_e) = m0 * Int k^2 f dk / Int k^2 f [1/D - (4/3) a gamma / D^3] dk .
#
# Written as a ratio the density prefactor g_s g_v/(2 pi^2) cancels, alpha -> 0 collapses the
# bracket to 1 so m_c == m0 EXACTLY at every T_e (the off-switch), and the T=0 limit reduces
# analytically to the Fermi-surface mass m_c = m0 (1 + 2 a E_F) -- the pinned gate, and the same
# expression materials.scattering.KaneOpticalMass returns (an independent in-repo cross-check).
# ---------------------------------------------------------------------------------------------

_GL_N = 64                                       # Gauss-Legendre nodes per quadrature panel
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_N)


def _kane_k_of_E(E_J, m0_kg: float, a: float):
    """k [1/m] on the Kane band E(1+aE) = hbar^2 k^2/(2 m0) (clipped at E <= 0 -> k = 0)."""
    E = np.maximum(np.asarray(E_J, dtype=np.float64), 0.0)
    return np.sqrt(2.0 * float(m0_kg) * E * (1.0 + a * E)) / HBAR


def _gl_panels(edges):
    """Gauss-Legendre nodes/weights over the union of the consecutive panels [edges[i], edges[i+1]].
    Panels of zero (or inverted) width are dropped; the integrand is analytic on each panel, so a
    modest fixed order converges spectrally."""
    ks, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            ks.append(0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo))
            ws.append(0.5 * (hi - lo) * _GL_W)
    if not ks:                                                     # degenerate (n -> 0) fallback
        return np.zeros(1), np.zeros(1)
    return np.concatenate(ks), np.concatenate(ws)


def _kane_quadrature(mu_J: float, kT_J: float, E_F_J: float, m0_kg: float, a: float):
    """(k, w) for the fixed-density Fermi integrals. At kT == 0 the domain is the Fermi sphere
    [0, k_F]; at kT > 0 three panels straddle the +/-12 kT Fermi step and run out to mu + 50 kT
    (f < 2e-22 there), so the SAME rule is accurate from the degenerate to the nondegenerate limit."""
    if kT_J <= 0.0:
        return _gl_panels([0.0, float(_kane_k_of_E(E_F_J, m0_kg, a))])
    E_cut = max(mu_J, 0.0) + 50.0 * kT_J
    E_a = min(max(mu_J - 12.0 * kT_J, 0.0), E_cut)
    E_b = min(max(mu_J + 12.0 * kT_J, 0.0), E_cut)
    return _gl_panels([0.0] + [float(_kane_k_of_E(E, m0_kg, a)) for E in (E_a, E_b, E_cut)])


def _kane_occupancy(k, mu_J: float, kT_J: float, m0_kg: float, a: float):
    """(f, gamma, D) at momentum k: Fermi-Dirac occupancy, the parabolic-form energy
    gamma = hbar^2 k^2/(2 m0), and D = 1 + 2 a E = sqrt(1 + 4 a gamma)."""
    gam = HBAR ** 2 * np.asarray(k, dtype=np.float64) ** 2 / (2.0 * float(m0_kg))
    D = np.sqrt(1.0 + 4.0 * a * gam)
    E = 2.0 * gam / (1.0 + D)                      # == (D-1)/(2a), stable as a -> 0
    if kT_J <= 0.0:
        return np.where(E <= mu_J, 1.0, 0.0), gam, D
    return 1.0 / (1.0 + np.exp(np.clip((E - mu_J) / kT_J, -500.0, 500.0))), gam, D


def _kane_chemical_potential(n: float, m0_kg: float, a: float, E_F_J: float, kT_J: float,
                             g_s: int, g_v: int) -> float:
    """mu(T_e) at FIXED carrier density from the exact Kane Fermi-Dirac integral (brentq). The Kane
    DOS increases with E, so n(mu = E_F, T > 0) > n and the root is bracketed below E_F; the lower
    bracket is expanded geometrically until n(mu) < n (guaranteed, n(mu) -> 0 as mu -> -inf)."""
    from scipy.optimize import brentq
    pref = g_s * g_v / (2.0 * np.pi ** 2)

    def resid(mu):
        k, w = _kane_quadrature(mu, kT_J, E_F_J, m0_kg, a)
        f, _gam, _D = _kane_occupancy(k, mu, kT_J, m0_kg, a)
        return pref * float(np.sum(w * k * k * f)) - n

    hi, step = E_F_J, max(kT_J, 0.05 * E_F_J)
    for _ in range(200):                                   # defensive: raise the top bracket too
        if resid(hi) >= 0.0:
            break
        hi += step
        step *= 2.0
    else:
        raise RuntimeError("kane_mass_of_Te: could not bracket mu(T_e) from above")
    lo, step = E_F_J - max(kT_J, 0.05 * E_F_J), max(kT_J, 0.05 * E_F_J)
    for _ in range(200):
        if resid(lo) <= 0.0:
            break
        step *= 2.0
        lo = E_F_J - step
    else:
        raise RuntimeError("kane_mass_of_Te: could not bracket mu(T_e) from below")
    return float(brentq(resid, lo, hi, xtol=1.0e-14 * E_F_J, rtol=8.9e-16, maxiter=200))


def _kane_conductivity_mass(m0_kg: float, alpha_per_eV: float, n: float, Te: float,
                            g_s: int, g_v: int) -> float:
    """Scalar f-sum (Drude) mass m_c [kg]; see the block comment above for the functional."""
    a = float(alpha_per_eV) / Q_E                                   # J^-1
    E_F = float(fermi_energy_J(n, m0_kg, alpha_per_eV, g_s=g_s, g_v=g_v))
    kT = KB * float(Te)
    mu = E_F if kT <= 0.0 else _kane_chemical_potential(float(n), m0_kg, a, E_F, kT, g_s, g_v)
    k, w = _kane_quadrature(mu, kT, E_F, m0_kg, a)
    f, gam, D = _kane_occupancy(k, mu, kT, m0_kg, a)
    wk = w * k * k * f
    dens = float(np.sum(wk))                                        # ~ n (prefactor cancels)
    inv_m = float(np.sum(wk * (1.0 / D - (4.0 / 3.0) * a * gam / D ** 3)))   # ~ n/m_c, in 1/m0
    if not (dens > 0.0 and inv_m > 0.0):
        raise ValueError("kane_mass_of_Te: the f-sum integral is non-positive (n_m3 = {:g}, "
                         "T_e = {:g} K) -- check n_m3 > 0 and Te_K >= 0".format(n, Te))
    return float(m0_kg) * dens / inv_m


def kane_mass_of_Te(m0_kg: float, alpha_per_eV: float, n_m3, Te_K, *, g_s: int = 2, g_v: int = 1,
                    exponent: float = 1.0):
    """Kane f-sum (Drude / conductivity) mass m_c(T_e) [kg] -- the mass wp^2 = n e^2/(eps0 m) needs.
    Hotter electrons occupy higher, heavier states on the nonparabolic band, so m_c RISES with T_e:

        m_c(T_e) = m0 * [ Int k^2 f dk / Int k^2 f (1/D - (4/3) a gamma/D^3) dk ]^exponent,
        gamma(k) = hbar^2 k^2/(2 m0),   D = 1 + 2 a E = sqrt(1 + 4 a gamma),   a = alpha_per_eV/q,

    with f the Fermi-Dirac occupancy at the chemical potential mu(T_e) that holds the density FIXED
    (solved exactly, no Sommerfeld expansion -- valid degenerate AND nondegenerate).

    audit C-1: this used to return the mass evaluated at the MEAN ENERGY, m0(1 + 2 a <E>) on a
    parabolic (3/5)E_F base -- the DOS-flavoured average, not the f-sum/conductivity mass
    DrudeOptical documents ("m_opt_kg is the OPTICAL (conductivity) effective mass"). At the repo's
    ITO preset (m0 = 0.35 m_e, alpha = 0.5/eV, n = 1e27) that ran 17.2% LOW, wp^2 20.8% high and put
    the cold ENZ crossing 148.8 nm to the blue. The mu(T_e) closed form the auditor proposed instead
    (m0(1 + 2 a mu)) has the WRONG SIGN of the T_e dependence (mu FALLS with T_e at fixed n) -- the
    f-sum integral is the only correct route; the legacy average-energy form is retained (private,
    unused by the Drude path) as _kane_mass_mean_energy_of_Te for regression reference only.

    LIMITS (both pinned in tests/test_carrier_heating.py):
      alpha_per_eV == 0 -> m0 EXACTLY at every T_e (the byte-identical off-switch),
      T_e -> 0          -> m0 (1 + 2 a E_F), the Fermi-surface mass (== materials.KaneOpticalMass).
    `exponent` is a legacy shaping knob applied to the mass RATIO m_c/m0 (1.0 = the exact f-sum
    mass). n_m3 and Te_K broadcast against each other."""
    if alpha_per_eV == 0.0:
        shape = np.broadcast_shapes(np.shape(np.asarray(n_m3, dtype=np.float64)),
                                    np.shape(np.asarray(Te_K, dtype=np.float64)))
        return np.full(shape, float(m0_kg)) if shape else float(m0_kg)
    n_b, Te_b = np.broadcast_arrays(np.asarray(n_m3, dtype=np.float64),
                                    np.asarray(Te_K, dtype=np.float64))
    # NaN/inf FIRST: every comparison against NaN is False, so a NaN slid past the sign checks and
    # died much later inside _kane_chemical_potential as "could not bracket mu(T_e) from above" --
    # a RuntimeError that points at the root finder instead of at the caller's input.
    if not (np.all(np.isfinite(n_b)) and np.all(np.isfinite(Te_b))):
        raise ValueError("kane_mass_of_Te: n_m3 and Te_K must be finite (got NaN or inf)")
    if np.any(n_b <= 0.0) or np.any(Te_b < 0.0):
        raise ValueError("kane_mass_of_Te: require n_m3 > 0 and Te_K >= 0")
    out = np.array([_kane_conductivity_mass(m0_kg, alpha_per_eV, float(nn), float(tt), g_s, g_v)
                    for nn, tt in zip(n_b.ravel(), Te_b.ravel())], dtype=np.float64)
    out = float(m0_kg) * np.power(out / float(m0_kg), exponent)
    return out.reshape(n_b.shape) if n_b.shape else float(out[0])


def _kane_mass_mean_energy_of_Te(m0_kg: float, alpha_per_eV: float, n_m3, Te_K, *,
                                 g_s: int = 2, g_v: int = 1, exponent: float = 1.0):
    """LEGACY (pre-C-1) band-averaged mass m*(<E>) = m0 (1 + 2 alpha <E>/q)^exponent with the
    Sommerfeld mean energy <E>(T_e) = (3/5) E_F [1 + (5 pi^2/12)(kB T_e/E_F)^2 K],
    K = (1 + 2 a E_F)/(1 + a E_F) the Kane-DOS Sommerfeld factor (audit C2-2).

    NOT the Drude mass and NOT used by any wp^2 consumer -- superseded by the f-sum
    kane_mass_of_Te (audit C-1). Retained PRIVATE so the C2-2 Sommerfeld-coefficient regression and
    the C-1/N5 error-sign gate (the legacy form UNDER-states the T_e-induced rise below
    a*E_F ~ 0.2 and OVER-states it above) keep a reference implementation to test against.
    Its Sommerfeld expansion is a degenerate-gas result: it diverges from the truth once
    kB T_e ~ E_F.

    OUTSIDE THAT ENVELOPE IT RETURNS SILENT NONSENSE (fix-verify W1 item 6): the (kT/E_F)^2 term is
    unbounded, so at n = 1e25 m^-3, T_e = 20000 K (kT/E_F = 36.5) it returns m/m0 = 159.7 against
    the f-sum truth 5.640 -- 28x high, with NO warning. It is private and has no callers outside
    the regression gates for exactly this reason; do not resurrect it as a fast path."""
    if alpha_per_eV == 0.0:
        return np.full(np.shape(np.asarray(n_m3, dtype=np.float64)), float(m0_kg)) \
            if np.ndim(n_m3) else float(m0_kg)
    E_F = fermi_energy_J(n_m3, m0_kg, alpha_per_eV, g_s=g_s, g_v=g_v)
    kT = KB * np.asarray(Te_K, dtype=np.float64)
    a_EF = float(alpha_per_eV) * E_F / Q_E                     # a*E_F, dimensionless
    kane_dos_factor = (1.0 + 2.0 * a_EF) / (1.0 + a_EF)        # ->1 exactly at alpha=0 (C2-2)
    mean_E = (3.0 / 5.0) * E_F * (1.0 + (5.0 * np.pi ** 2 / 12.0) * (kT / E_F) ** 2
                                  * kane_dos_factor)
    return float(m0_kg) * np.power(1.0 + 2.0 * float(alpha_per_eV) * mean_E / Q_E, exponent)


def gamma_of_Te(gamma0_rad_s: float, Te_K, *, p: float = 1.0, T_ref_K: float = T_REF):
    """Phonon-scattering Drude damping vs electron temperature: Gamma(T_e) = gamma0 (T_e/T_ref)^p
    (p ~ 1 high-T phonon limit). p == 0 -> gamma0 EXACTLY (off-switch). The physically-resolved
    decomposition (phonon + ionized-impurity + grain-boundary) is materials.scattering.MatthiessenGamma
    (R2); swap that in for a calibrated model -- this is the carrier-heating phonon seam."""
    if p == 0.0:
        return float(gamma0_rad_s) if not np.ndim(Te_K) else \
            np.full(np.shape(np.asarray(Te_K, dtype=np.float64)), float(gamma0_rad_s))
    return float(gamma0_rad_s) * np.power(np.asarray(Te_K, dtype=np.float64) / float(T_ref_K), p)


@dataclass
class TwoTempParams:
    """Two-temperature-model material parameters (SI, volumetric). C_e is the electron heat capacity --
    a float (constant) or a callable C_e(T_e) (e.g. gamma_e*T_e for a degenerate gas, which gives the
    sub-ps-rise / ps-relaxation asymmetry). C_l the lattice heat capacity [J/m^3/K]; G_e_l the
    electron-phonon coupling [W/m^3/K]; alpha_abs the absorbed-power coupling that turns the supplied
    intensity into a volumetric heating rate alpha_abs*I [W/m^3] (set alpha_abs = absorption coeff [1/m]
    if I is irradiance [W/m^2], or 1 if I is already W/m^3)."""
    C_e: Union[float, Callable[[float], float]]
    C_l: float
    G_e_l: float
    alpha_abs: float = 1.0

    def C_e_of(self, Te: float) -> float:
        return float(self.C_e(Te)) if callable(self.C_e) else float(self.C_e)


def two_temperature_response(times_s, intensity_of_t: Callable, params: TwoTempParams, *,
                             T0_K: float = 300.0, max_step: Optional[float] = None):
    """Integrate the TTM from t=0 to times_s[-1], sampled at times_s. `intensity_of_t(t)` is the pump
    drive (units matched to params.alpha_abs so alpha_abs*I is W/m^3). Returns (t_s, Te, Tl). Uses a
    stiff BDF integrator (G/C_e can be 1e12-1e14 1/s -> sub-ps). intensity == 0 -> Te == Tl == T0.

    max_step caps the internal step so the adaptive integrator cannot STEP OVER a narrow pump pulse
    (a classic stiff-solver-misses-the-forcing failure); default = the output sample spacing, so the
    pump is always resolved at least to the requested time resolution.

    Every coefficient is validated: C_l > 0, G_e_l >= 0 and (audit C-11) C_e finite and > 0 -- at
    T0_K before the solve, and at every rhs evaluation with Te >= 0 (which is what a callable
    C_e = gamma_e*Te needs; a setup check alone cannot see a closure that crosses zero later)."""
    t = np.asarray(times_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("times_s must be a 1D strictly increasing array with >= 2 samples")
    if params.C_l <= 0 or params.G_e_l < 0:
        raise ValueError("C_l must be > 0 and G_e_l >= 0")
    # audit C-11: C_e was the ONE unvalidated coefficient. rhs DIVIDES by it, so C_e < 0 flips the
    # sign of the electron-gas energy equation (the hot gas COOLS when heated) and used to integrate
    # SILENTLY to a plausible-looking (t, Te, Tl); C_e == 0 died far from the cause inside solve_ivp
    # as "array must not contain infs or NaNs". Two checks: once at setup (the float case, and the
    # callable at the initial state -- a clean message before any integration), and once per rhs
    # evaluation for the documented CALLABLE usage (C_e = gamma_e*Te), whose zero crossing a setup
    # check cannot see. The rhs check is gated on Te >= 0 so a stiff-solver trial state at negative
    # temperature -- unphysical, and not what the user asked about -- cannot raise spuriously; any
    # C_e that is positive over the physical Te >= 0 axis therefore never trips it.
    C0 = params.C_e_of(float(T0_K))
    if not (np.isfinite(C0) and C0 > 0.0):
        raise ValueError("two_temperature_response: C_e must be finite and > 0 (got C_e({:g} K) = "
                         "{!r}); a non-positive electron heat capacity inverts the sign of the "
                         "electron-gas energy equation (audit C-11)".format(float(T0_K), C0))
    ms = float(max_step) if max_step is not None else float(np.min(np.diff(t)))

    def rhs(tt, y):
        Te, Tl = y
        Ce = params.C_e_of(Te)
        if Te >= 0.0 and not (np.isfinite(Ce) and Ce > 0.0):       # audit C-11
            raise ValueError(
                "two_temperature_response: C_e({:g} K) = {!r} at t = {:g} s -- the electron heat "
                "capacity must stay finite and > 0 over the whole trajectory (a non-positive C_e "
                "inverts the sign of the electron-gas energy equation, so the hot gas would COOL "
                "when heated); check the C_e closure (audit C-11)".format(Te, Ce, tt))
        Q = float(params.alpha_abs) * float(intensity_of_t(tt))    # W/m^3 absorbed power density
        flow = params.G_e_l * (Te - Tl)
        return [(-flow + Q) / Ce, flow / params.C_l]

    # ZeroInitBDF, not "BDF": scipy's BDF reads one uninitialised row of its difference array on
    # the first step (see core.numerics.ZeroInitBDF). Same trajectory, minus the spurious
    # "invalid value encountered in subtract" the recycled heap can plant there.
    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), [float(T0_K), float(T0_K)],
                    method=ZeroInitBDF, t_eval=t, rtol=1e-7, atol=1e-6, max_step=ms)
    if not sol.success:
        raise RuntimeError("two_temperature_response: ODE integration failed ({})".format(sol.message))
    return t, sol.y[0], sol.y[1]


def carrier_heating_transient(times_s, intensity_of_t: Callable, lambda_m: float, *,
                              drude0: DrudeOptical, ttm_params: TwoTempParams, n_m3: float,
                              alpha_per_eV: float, m0_kg: Optional[float] = None, g_s: int = 2,
                              g_v: int = 1, gamma_p: float = 1.0, mass_exponent: float = 1.0,
                              build_stack: Optional[Callable] = None, T0_K: float = 300.0,
                              gamma_T_ref_K: Optional[float] = None):
    """Full carrier-heating ENZ transient: solve the TTM for T_e(t), then feed the existing
    transient_optics loop a per-instant DrudeOptical whose m_opt_kg / gamma_rad_s are CLOSURES capturing
    T_e(t_i) (so m_c(T_e) and Gamma(T_e) evolve with the hot-electron temperature). The density n_m3 is
    held fixed (carrier heating is a same-density, hot-carrier effect, distinct from gate accumulation).
    Returns (t_s, R, T, eps_front, Te, Tl). Reduces to the linear constant-Drude result when the pump or
    the nonparabolicity/phonon knobs are off (see module off-switches).

    `gamma_T_ref_K` (audit C-7) is the temperature at which `drude0.gamma_rad_s` was CALIBRATED, i.e.
    the T_ref of Gamma(T_e) = gamma0 (T_e/T_ref)^gamma_p. It used to be hardwired to T_REF = 300 K
    while the caller owned T0_K, so off 300 K the module's own byte-identical off-switch was FALSE:
    with the pump off T_e == T_l == T0_K exactly, yet Gamma = gamma0 (T0_K/300)^p -- 0.257*gamma0 at
    77 K, 1.667*gamma0 at 500 K (p=1), a DIFFERENT material with every heating knob off. The default
    None now means "gamma0 is the damping at T0_K", which restores the off-switch at every base
    temperature and is byte-identical at the default T0_K = 300 K. Pass gamma_T_ref_K=300.0
    explicitly to keep the legacy reading of a 300 K-calibrated gamma0 at a non-300 K T0_K."""
    from dynameta.transient_optics import optical_transient_response

    # audit C5-10: a CALIBRATED DrudeOptical carrying callable m_opt_kg / gamma_rad_s used
    # to be silently replaced by the bare electron mass / a hardcoded 1e14 rad/s -- the
    # transient was computed for a DIFFERENT material even with every heating knob off,
    # violating the module's own documented byte-identical off-switch. A density-dependent
    # m_opt(n) callable cannot be inverted safely here (evaluating it at n_m3 double-counts
    # the Kane band filling), so require the explicit band-edge scalars instead.
    if m0_kg is None and callable(drude0.m_opt_kg):
        raise ValueError(
            "carrier_heating_transient: drude0.m_opt_kg is a callable (a calibrated "
            "density-dependent mass); pass the explicit band-edge m0_kg= instead -- "
            "silently substituting M_E computed a different material (audit C5-10).")
    if callable(drude0.gamma_rad_s):
        raise ValueError(
            "carrier_heating_transient: drude0.gamma_rad_s is a callable (a calibrated "
            "scattering model); build the transient from its evaluated value at n_m3 "
            "(float(drude0.gamma_rad_s(n_m3))) and pass a scalar-gamma DrudeOptical -- "
            "silently substituting 1e14 rad/s computed a different material (audit C5-10).")
    m0 = float(m0_kg) if m0_kg is not None else float(drude0.m_opt_kg)
    gamma0 = float(drude0.gamma_rad_s)
    # audit C-7: the phonon-damping reference temperature travels WITH gamma0's calibration, so it is
    # threaded (default = T0_K) instead of being pinned at 300 K inside gamma_of_Te.
    gamma_T_ref = float(T0_K) if gamma_T_ref_K is None else float(gamma_T_ref_K)
    # FINITE and > 0, not merely > 0: +inf passes `> 0` and makes Gamma = gamma0 (T_e/inf)^p = 0
    # EXACTLY, i.e. a silently LOSSLESS Drude -- a different material with no diagnostic, which is
    # the same class of defect C-7 exists to close (measured: gamma_T_ref_K=inf returned
    # eps[0] = -1.866+0j, a purely real permittivity, with no raise).
    if not (np.isfinite(gamma_T_ref) and gamma_T_ref > 0.0):
        raise ValueError("carrier_heating_transient: the gamma reference temperature must be FINITE "
                         "and > 0 K (got gamma_T_ref_K={!r}, T0_K={!r}); a non-finite reference makes "
                         "Gamma(T_e) = gamma0 (T_e/T_ref)^p collapse to 0 and returns a silently "
                         "LOSSLESS Drude (audit C-7).".format(gamma_T_ref_K, T0_K))
    # T0_K is only checked THROUGH gamma_T_ref when gamma_T_ref_K is None; with an explicit
    # reference an unphysical T0_K used to reach the TTM unchecked (it then raised, but from
    # two_temperature_response's C_e guard, naming C-11 and a heat capacity rather than the base
    # temperature the caller actually got wrong -- and only for a C_e whose sign happens to flip).
    if not (np.isfinite(float(T0_K)) and float(T0_K) > 0.0):
        raise ValueError("carrier_heating_transient: the base temperature T0_K must be FINITE and "
                         "> 0 K (got T0_K={!r}); it seeds BOTH the two-temperature solve and (with "
                         "gamma_T_ref_K=None) the phonon-damping reference (audit C-7).".format(T0_K))
    t, Te, Tl = two_temperature_response(times_s, intensity_of_t, ttm_params, T0_K=T0_K)
    Te_of_t = lambda tt: float(np.interp(tt, t, Te))

    def drude_of_t(tt):
        Te_i = Te_of_t(tt)                                         # capture by default-arg below
        return DrudeOptical(
            eps_inf=drude0.eps_inf,
            m_opt_kg=(lambda nn, _Te=Te_i: kane_mass_of_Te(m0, alpha_per_eV, nn, _Te, g_s=g_s,
                                                           g_v=g_v, exponent=mass_exponent)),
            gamma_rad_s=(lambda nn, _Te=Te_i: gamma_of_Te(gamma0, _Te, p=gamma_p,
                                                          T_ref_K=gamma_T_ref)))

    n_of_t = lambda tt: float(n_m3)
    tt, R, Tr, eps_front = optical_transient_response(times_s, n_of_t, lambda_m,
                                                      drude_of_t=drude_of_t, build_stack=build_stack)
    return tt, R, Tr, eps_front, Te, Tl
