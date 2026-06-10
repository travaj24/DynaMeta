"""Landau-Lifshitz-Gilbert MACROSPIN magnetization dynamics (roadmap R11) -- the magnetic
order-parameter analog of the LC director dynamics: one unit vector m(t) on the sphere, driven by
the effective field and damped toward it.

    dm/dt = -(gamma0 * mu0 / (1 + alpha^2)) * [ m x H_eff  +  alpha * m x (m x H_eff) ]

(the explicit Gilbert form: the first cross product is the conservative PRECESSION, the double cross
product the DAMPING that spirals m toward H_eff). UNITS: H_eff is in A/m everywhere; gamma0 is the
gyromagnetic ratio in rad/(s*T) (free electron 1.760859630e11), so the torque rate carries the
explicit mu0 (B = mu0 H) -- the classic LLG unit trap, resolved once here.

    H_eff = H_applied(t) + H_K (m . u) u - Ms (N m)
    H_K   = 2 K_u / (mu0 Ms)          (uniaxial anisotropy field, easy axis u)
    N     = demagnetization tensor    (diagonal (3,) or full (3,3); thin film ~ diag(0,0,1))

The energy density (the Lyapunov function for alpha > 0; H_eff = -(1/(mu0 Ms)) dU/dm):

    U(m) = -mu0 Ms (m . H_applied) - K_u (m . u)^2 + (1/2) mu0 Ms^2 (m . N m)   [J/m^3]

Exact limits the oracle gates use: alpha = 0 -> pure precession at omega = gamma0 mu0 |H| with
|m| = 1 conserved; for H along z with no anisotropy/demag the FULL nonlinear ring-down obeys
tan(theta/2)(t) = tan(theta0/2) exp(-lambda t) EXACTLY with lambda = alpha gamma0 mu0 H/(1+alpha^2);
Stoner-Wohlfarth switching at 45 deg occurs at EXACTLY H_K/2 (the astroid minimum).

R13 SEAM: each m_t row feeds VectorMagnetoOpticModel via fields['m_vector'] (and a per-instant
drude_of_t-style hook in transient_optics carries the time dependence). Pure numpy/scipy; new module
-> byte-identical-off by construction. solve_ivp pattern mirrors carriers/lc_dynamics.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from dynameta.constants import MU0

GAMMA_ELECTRON_RAD_ST = 1.760859630e11      # free-electron gyromagnetic ratio [rad/(s T)]

__all__ = ["LLGMacrospin", "LLGResult", "GAMMA_ELECTRON_RAD_ST"]


@dataclass
class LLGResult:
    t_s: np.ndarray              # (nt,)
    m_t: np.ndarray              # (nt, 3) unit magnetization trace (rows feed fields['m_vector'])
    energy_J_m3: np.ndarray      # (nt,) U(m(t)) -- monotone non-increasing for alpha > 0, constant H
    precession_rad_s: float      # gamma0 mu0 |H_eff(t0)| (the alpha=0 frequency scale)


@dataclass
class LLGMacrospin:
    """Macrospin LLG integrator (see module docstring for the equation set and units)."""
    Ms_A_m: float
    alpha: float = 0.0
    gamma0_rad_sT: float = GAMMA_ELECTRON_RAD_ST
    K_u_J_m3: float = 0.0
    easy_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    N_demag: Optional[np.ndarray] = None     # None -> no shape anisotropy; (3,) diag or (3,3)
    H_applied_A_m: Optional[Callable[[float], np.ndarray]] = None   # H(t) [A/m]; None -> 0

    def __post_init__(self):
        if not (self.Ms_A_m > 0.0):
            raise ValueError("LLG: Ms_A_m must be > 0")
        if self.alpha < 0.0:
            raise ValueError("LLG: Gilbert damping alpha must be >= 0")
        if not (self.gamma0_rad_sT > 0.0):
            raise ValueError("LLG: gamma0_rad_sT must be > 0")
        if self.K_u_J_m3 < 0.0:
            raise ValueError("LLG: K_u_J_m3 must be >= 0 (easy-axis uniaxial)")
        u = np.asarray(self.easy_axis, dtype=np.float64)
        if u.shape != (3,) or not np.all(np.isfinite(u)) or np.linalg.norm(u) == 0.0:
            raise ValueError("LLG: easy_axis must be a finite nonzero 3-vector")
        self.easy_axis = u / np.linalg.norm(u)
        if self.N_demag is not None:
            N = np.asarray(self.N_demag, dtype=np.float64)
            if N.ndim == 1 and N.size == 3:
                N = np.diag(N)
            if N.shape != (3, 3):
                raise ValueError("LLG: N_demag must be a (3,) diagonal or a (3,3) tensor")
            self.N_demag = N

    # ---- fields + energy --------------------------------------------------------------------
    def _H_app(self, t: float) -> np.ndarray:
        if self.H_applied_A_m is None:
            return np.zeros(3)
        H = np.asarray(self.H_applied_A_m(float(t)), dtype=np.float64)
        if H.shape != (3,):
            raise ValueError("LLG: H_applied_A_m(t) must return a 3-vector, got shape {}".format(
                H.shape))
        return H

    def H_eff_A_m(self, t: float, m: np.ndarray) -> np.ndarray:
        """H_applied + uniaxial anisotropy field + demagnetization field [A/m]."""
        H = self._H_app(t).copy()
        if self.K_u_J_m3 > 0.0:
            H_K = 2.0 * self.K_u_J_m3 / (MU0 * self.Ms_A_m)
            H += H_K * float(np.dot(m, self.easy_axis)) * self.easy_axis
        if self.N_demag is not None:
            H -= self.Ms_A_m * (self.N_demag @ m)
        return H

    def energy_J_m3(self, t: float, m: np.ndarray) -> float:
        """U(m) = -mu0 Ms m.H_app - K_u (m.u)^2 + (1/2) mu0 Ms^2 m.N m (the Lyapunov function)."""
        u = -MU0 * self.Ms_A_m * float(np.dot(m, self._H_app(t)))
        u -= self.K_u_J_m3 * float(np.dot(m, self.easy_axis)) ** 2
        if self.N_demag is not None:
            u += 0.5 * MU0 * self.Ms_A_m ** 2 * float(m @ self.N_demag @ m)
        return u

    # ---- integrator -------------------------------------------------------------------------
    def simulate(self, t_eval, m0, *, method: str = "BDF", rtol: float = 1e-10,
                 atol: float = 1e-12, max_step: Optional[float] = None) -> LLGResult:
        """Integrate the LLG from t_eval[0] to t_eval[-1]. m0 is normalized; |m| = 1 is maintained
        (the rhs renormalizes against drift -- the exact dynamics conserves it). max_step defaults to
        the output spacing so a pulsed H_applied cannot be stepped over (the R9 lesson)."""
        t = np.asarray(t_eval, dtype=np.float64)
        if t.ndim != 1 or t.size < 5 or np.any(np.diff(t) <= 0):
            raise ValueError("LLG: t_eval must be 1D strictly increasing with >= 5 points")
        m0 = np.asarray(m0, dtype=np.float64)
        if m0.shape != (3,) or np.linalg.norm(m0) == 0.0:
            raise ValueError("LLG: m0 must be a nonzero 3-vector")
        m0 = m0 / np.linalg.norm(m0)
        coeff = self.gamma0_rad_sT * MU0 / (1.0 + self.alpha ** 2)   # rad/(s T) * T m/A -> m/(A s)
        a = self.alpha

        def rhs(tt, y):
            m = y / np.linalg.norm(y)
            H = self.H_eff_A_m(tt, m)
            mxH = np.cross(m, H)
            return -coeff * (mxH + a * np.cross(m, mxH))

        ms = float(max_step) if max_step is not None else float(np.min(np.diff(t)))
        sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), m0, t_eval=t, method=method,
                        rtol=rtol, atol=atol, max_step=ms)
        if not sol.success:
            raise RuntimeError("LLG: solve_ivp failed ({})".format(sol.message))
        m_t = sol.y.T.copy()
        m_t /= np.linalg.norm(m_t, axis=1, keepdims=True)            # exact unit-sphere projection
        en = np.array([self.energy_J_m3(float(ti), m_t[i]) for i, ti in enumerate(sol.t)])
        w0 = self.gamma0_rad_sT * MU0 * float(np.linalg.norm(self.H_eff_A_m(float(t[0]), m0)))
        return LLGResult(t_s=np.asarray(sol.t), m_t=m_t, energy_J_m3=en, precession_rad_s=w0)
