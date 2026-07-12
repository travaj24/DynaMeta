"""
Carrier-heating (two-temperature) ENZ nonlinearity (roadmap R9): the intrinsic ultrafast optical
nonlinearity of a transparent-conductor ENZ film. An absorbed optical pump heats the electron gas to
an electron temperature T_e above the lattice T_l; the hot electrons (i) climb the nonparabolic band so
the band-averaged optical mass <m*(T_e)> RISES (the plasma frequency wp^2 ~ n/<m*> drops, so Re(eps)
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
  - alpha_per_eV == 0 in kane_mass_of_Te            -> <m*> == m0 EXACTLY (no band nonparabolicity).
  - p == 0 in gamma_of_Te                           -> Gamma == gamma0 EXACTLY.
Any of these collapses the per-instant Drude to the constant drude0 at every step. Pure numpy/scipy;
exp(-i omega t), Im(eps) > 0; SI units (C in J/m^3/K, alpha_abs I in W/m^3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from scipy.integrate import solve_ivp

from dynameta.constants import Q_E, HBAR, KB, M_E
from dynameta.materials.optical_model import DrudeOptical

T_REF = 300.0


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


def kane_mass_of_Te(m0_kg: float, alpha_per_eV: float, n_m3, Te_K, *, g_s: int = 2, g_v: int = 1,
                    exponent: float = 1.0):
    """Band-averaged Kane optical mass <m*(T_e)> [kg]. Hotter electrons sit higher in the nonparabolic
    band where m* is larger, so <m*> RISES with T_e:
        <m*(T_e)> = m0 (1 + 2 alpha_eV <E>(T_e)/q)^exponent,
        <E>(T_e)  = (3/5) E_F [1 + (5 pi^2/12)(kB T_e / E_F)^2 * K],
        K         = (1 + 2 a E_F) / (1 + a E_F),   a = alpha_per_eV / q.
    K is the KANE-DOS Sommerfeld factor (audit C2-2): for ANY DOS the fixed-n shift is
    d<E> = (pi^2/6)(kT)^2 g(E_F)/n, and for the Kane DOS g(E_F)/n =
    (3/2)(1+2aE_F)/[E_F(1+aE_F)] -- the parabolic coefficient (K=1, exact at alpha=0)
    understated the heating-induced <E>/<m*> rise by 11-21% at the validated ITO regime.
    VALIDITY: the Sommerfeld expansion assumes a DEGENERATE gas, kB*T_e << E_F; the quadratic
    correction term grows without bound, so beyond kB*T_e ~ E_F the formula over-states <E> (the
    validated ITO regime peaks at kB*T_e/E_F ~ 0.3). A RuntimeWarning fires past kB*T_e > E_F; the
    nondegenerate closure (full Fermi-Dirac integral) is a documented follow-on.
    alpha_per_eV == 0 -> returns m0 EXACTLY (the off-switch; no float drift through the sqrt branch)."""
    if alpha_per_eV == 0.0:
        return np.full(np.shape(np.asarray(n_m3, dtype=np.float64)), float(m0_kg)) \
            if np.ndim(n_m3) else float(m0_kg)
    E_F = fermi_energy_J(n_m3, m0_kg, alpha_per_eV, g_s=g_s, g_v=g_v)
    kT = KB * np.asarray(Te_K, dtype=np.float64)
    if np.any(kT > E_F):                       # Sommerfeld validity edge (see docstring)
        import warnings
        warnings.warn("kane_mass_of_Te: kB*T_e exceeds E_F -- the Sommerfeld degenerate-gas expansion "
                      "is outside its validity (kB*Te/E_F up to {:.2f}); <m*> is over-stated.".format(
                          float(np.max(kT / E_F))), RuntimeWarning, stacklevel=2)
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
    pump is always resolved at least to the requested time resolution."""
    t = np.asarray(times_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("times_s must be a 1D strictly increasing array with >= 2 samples")
    if params.C_l <= 0 or params.G_e_l < 0:
        raise ValueError("C_l must be > 0 and G_e_l >= 0")
    ms = float(max_step) if max_step is not None else float(np.min(np.diff(t)))

    def rhs(tt, y):
        Te, Tl = y
        Ce = params.C_e_of(Te)
        Q = float(params.alpha_abs) * float(intensity_of_t(tt))    # W/m^3 absorbed power density
        flow = params.G_e_l * (Te - Tl)
        return [(-flow + Q) / Ce, flow / params.C_l]

    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), [float(T0_K), float(T0_K)], method="BDF",
                    t_eval=t, rtol=1e-7, atol=1e-6, max_step=ms)
    if not sol.success:
        raise RuntimeError("two_temperature_response: ODE integration failed ({})".format(sol.message))
    return t, sol.y[0], sol.y[1]


def carrier_heating_transient(times_s, intensity_of_t: Callable, lambda_m: float, *,
                              drude0: DrudeOptical, ttm_params: TwoTempParams, n_m3: float,
                              alpha_per_eV: float, m0_kg: Optional[float] = None, g_s: int = 2,
                              g_v: int = 1, gamma_p: float = 1.0, mass_exponent: float = 1.0,
                              build_stack: Optional[Callable] = None, T0_K: float = 300.0):
    """Full carrier-heating ENZ transient: solve the TTM for T_e(t), then feed the existing
    transient_optics loop a per-instant DrudeOptical whose m_opt_kg / gamma_rad_s are CLOSURES capturing
    T_e(t_i) (so <m*(T_e)> and Gamma(T_e) evolve with the hot-electron temperature). The density n_m3 is
    held fixed (carrier heating is a same-density, hot-carrier effect, distinct from gate accumulation).
    Returns (t_s, R, T, eps_front, Te, Tl). Reduces to the linear constant-Drude result when the pump or
    the nonparabolicity/phonon knobs are off (see module off-switches)."""
    from dynameta.transient_optics import optical_transient_response

    # audit C5-10: a CALIBRATED DrudeOptical carrying callable m_opt_kg / gamma_rad_s used
    # to be silently replaced by the bare electron mass / a hardcoded 1e14 rad/s -- the
    # transient was computed for a DIFFERENT material even with every heating knob off,
    # violating the module's own documented byte-identical off-switch. A band-averaged
    # m*(n) callable cannot be inverted safely here (evaluating it at n_m3 double-counts
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
    t, Te, Tl = two_temperature_response(times_s, intensity_of_t, ttm_params, T0_K=T0_K)
    Te_of_t = lambda tt: float(np.interp(tt, t, Te))

    def drude_of_t(tt):
        Te_i = Te_of_t(tt)                                         # capture by default-arg below
        return DrudeOptical(
            eps_inf=drude0.eps_inf,
            m_opt_kg=(lambda nn, _Te=Te_i: kane_mass_of_Te(m0, alpha_per_eV, nn, _Te, g_s=g_s,
                                                           g_v=g_v, exponent=mass_exponent)),
            gamma_rad_s=(lambda nn, _Te=Te_i: gamma_of_Te(gamma0, _Te, p=gamma_p)))

    n_of_t = lambda tt: float(n_m3)
    tt, R, Tr, eps_front = optical_transient_response(times_s, n_of_t, lambda_m,
                                                      drude_of_t=drude_of_t, build_stack=build_stack)
    return tt, R, Tr, eps_front, Te, Tl
