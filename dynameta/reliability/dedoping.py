"""REL4: ITO thermal de-doping -> ENZ wavelength drift. THE device-specific PARAMETRIC failure of a
TCO ENZ modulator: oxygen-vacancy re-oxidation/diffusion slowly reduces the carrier density n, the
Drude plasma frequency wp^2 ~ n/m falls, and the ENZ crossing (the spectral operating point) drifts.

Carrier decay (first-order Arrhenius kinetics):

    dn/dt = -lambda(T) * (n - n_min),   lambda(T) = lambda0 * exp(-Ea / (kB * T))
    =>  n(t) = n_min + (n0 - n_min) * exp(-lambda(T) * t)        (constant T)

Ea ~ 1.5-2.1 eV (bulk oxygen diffusion; interface paths are faster/lower-Ea -- the parameters are
stoichiometry/process EMPIRICAL, document the source when calibrating). OFF-SWITCH: lambda0 = 0 ->
n(t) == n0 EXACTLY (an explicit branch, no float drift).

ENZ tracking uses the ACTUAL DrudeOptical model (numeric Re(eps) = 0 crossing), so n-dependent Kane
masses and finite damping are handled exactly; the analytic sensitivity
d(lambda_ENZ)/dn = -(1/2)(lambda/n)(1 - dln m/dln n) (constant eps_inf, gamma << wp/sqrt(eps_inf))
is the ORACLE, not the implementation. Pure numpy/scipy; oracles in
validation/reliability_dedoping.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import KB_EV_K   # eV/K, single source (audit 6.3)


@dataclass(frozen=True)
class DedopingParams:
    """First-order de-doping kinetics. lambda0_per_s and Ea_eV are EMPIRICAL (fit from thermal-aging
    data; cite the dataset). lambda0_per_s = 0 is the exact off-switch."""
    lambda0_per_s: float = 0.0
    Ea_eV: float = 2.1
    n_min_m3: float = 0.0

    def rate_per_s(self, T_K):
        if self.lambda0_per_s == 0.0:
            return np.zeros(np.shape(np.asarray(T_K, dtype=np.float64))) if np.ndim(T_K) else 0.0
        T = np.asarray(T_K, dtype=np.float64)
        if np.any(T <= 0.0):
            raise ValueError("Dedoping: T_K must be > 0")
        return self.lambda0_per_s * np.exp(-self.Ea_eV / (KB_EV_K * T))


def carrier_decay(t_s, T_K, *, n0_m3: float, params: DedopingParams):
    """n(t) [m^-3] at constant temperature T_K. params.lambda0_per_s == 0 -> n(t) == n0 EXACTLY."""
    if not (n0_m3 > 0.0):
        raise ValueError("Dedoping: n0_m3 must be > 0")
    if params.Ea_eV < 0.0 or params.lambda0_per_s < 0.0 or params.n_min_m3 < 0.0:
        raise ValueError("Dedoping: lambda0_per_s, Ea_eV, n_min_m3 must be >= 0")
    t = np.asarray(t_s, dtype=np.float64)
    if np.any(t < 0.0):
        raise ValueError("Dedoping: t_s must be >= 0")
    if params.lambda0_per_s == 0.0:                          # exact off-switch (no exp(0) round-trip)
        return np.full(t.shape, float(n0_m3)) if t.ndim else float(n0_m3)
    lam = params.rate_per_s(T_K)
    return params.n_min_m3 + (n0_m3 - params.n_min_m3) * np.exp(-lam * t)


def enz_wavelength_m(drude, n_m3: float, *, lam_lo_m: float = 600e-9, lam_hi_m: float = 4000e-9,
                     rtol: float = 1e-12) -> float:
    """The ENZ crossing wavelength [m]: the Re(eps(lambda; n)) = 0 root of the ACTUAL DrudeOptical
    model (so callable Kane masses and finite gamma are handled exactly). Raises if no crossing lies
    in [lam_lo_m, lam_hi_m] (the density is too low/high for ENZ in the window)."""
    from scipy.optimize import brentq
    f = lambda lam: float(np.real(np.asarray(drude.eps(float(lam), n_m3=n_m3)).ravel()[0]))
    flo, fhi = f(lam_lo_m), f(lam_hi_m)
    if flo * fhi > 0.0:
        raise ValueError("enz_wavelength_m: no Re(eps)=0 crossing in [{:.0f}, {:.0f}] nm at n={:.3e} "
                         "(Re(eps) = {:+.3f} .. {:+.3f})".format(lam_lo_m * 1e9, lam_hi_m * 1e9,
                                                                 n_m3, flo, fhi))
    return float(brentq(f, lam_lo_m, lam_hi_m, rtol=rtol))


def enz_drift_m(drude, n_t_m3, **kw):
    """lambda_ENZ(t) [m] for a carrier-density trajectory n(t) (the de-doping parametric drift)."""
    n = np.atleast_1d(np.asarray(n_t_m3, dtype=np.float64))
    return np.array([enz_wavelength_m(drude, float(v), **kw) for v in n])
