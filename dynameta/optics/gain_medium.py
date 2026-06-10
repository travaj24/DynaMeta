"""R20: four-level gain-medium populations + the small-signal gain closed forms.

Level scheme (the standard four-level laser): the pump promotes N0 -> N3 at rate W_p [1/s];
N3 -> N2 fast (tau_32), N2 -> N1 is the lasing transition (tau_21, the metastable lifetime),
N1 -> N0 drains (tau_10). With constant W_p the system is LINEAR,

    dN/dt = A N,   N = (N0, N1, N2, N3),

so the propagator is the EXACT matrix exponential expm(A t) -- no stiffness games (tau_32/tau_21
spans 1e6 in real lasers and would defeat explicit stepping). Every column of A sums to zero,
so sum(N) = N_total is conserved to machine precision by construction. Steady state (weak pump,
N0 ~ N_total): N2_ss = W_p N0 tau_21, N1_ss = W_p N0 tau_10 -> inversion

    dN_ss = N2 - N1 = W_p N0 (tau_21 - tau_10)      (> 0 needs tau_21 > tau_10).

The optical side (optics.fdtd_nd, R20) consumes the CLAMPED inversion via FDTDLayer.gain_dN_m3
with the Lorentz-oscillator gain ADE P'' + dw_a P' + w_a^2 P = -kappa dN E (kappa = q^2/m_eff
[C^2/kg]); its small-signal susceptibility is chi(w) = -kappa dN / (eps0 (w_a^2 - w^2 -
i dw_a w)), giving the line-center intensity gain

    g0 = kappa dN / (n c eps0 dw_a)        [1/m]    (small_signal_gain_per_m below)

(exp(-i w t): Im chi < 0 = gain). DYNAMIC field-population coupling (saturation, lasing) is a
documented follow-on -- here the populations and the field are exact in their own domains and
meet through the clamped inversion. Pure numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import C_LIGHT, EPS0

__all__ = ["FourLevelSystem", "small_signal_gain_per_m"]


def small_signal_gain_per_m(kappa_C2_kg: float, dN_m3: float, n_refr: float,
                            dw_rad_s: float) -> float:
    """Line-center small-signal INTENSITY gain g0 = kappa dN/(n c eps0 dw) [1/m] (negative for
    dN < 0 = absorption)."""
    if not (kappa_C2_kg > 0.0 and n_refr > 0.0 and dw_rad_s > 0.0):
        raise ValueError("gain_medium: kappa, n and dw must be > 0")
    return float(kappa_C2_kg) * float(dN_m3) / (n_refr * C_LIGHT * EPS0 * float(dw_rad_s))


@dataclass(frozen=True)
class FourLevelSystem:
    """Four-level population dynamics under a constant pump (module header). Times in s; the
    pump W_p promotes N0 -> N3."""
    tau_32_s: float
    tau_21_s: float
    tau_10_s: float
    W_p_per_s: float = 0.0
    N_total_m3: float = 1.0

    def __post_init__(self):
        for nm in ("tau_32_s", "tau_21_s", "tau_10_s"):
            if not (getattr(self, nm) > 0.0):
                raise ValueError("FourLevelSystem: {} must be > 0".format(nm))
        if self.W_p_per_s < 0.0 or not (self.N_total_m3 > 0.0):
            raise ValueError("FourLevelSystem: W_p_per_s >= 0 and N_total_m3 > 0 required")

    def rate_matrix(self) -> np.ndarray:
        """A with dN/dt = A N, N = (N0, N1, N2, N3); every column sums to 0 (conservation)."""
        w, t32, t21, t10 = self.W_p_per_s, self.tau_32_s, self.tau_21_s, self.tau_10_s
        return np.array([[-w, 1.0 / t10, 0.0, 0.0],
                         [0.0, -1.0 / t10, 1.0 / t21, 0.0],
                         [0.0, 0.0, -1.0 / t21, 1.0 / t32],
                         [w, 0.0, 0.0, -1.0 / t32]])

    def evolve(self, t_s, N0=None) -> np.ndarray:
        """Populations (nt, 4) at times t_s via the EXACT propagator expm(A dt) step chain
        (uniform or non-uniform t grid). N0 defaults to everything in the ground state."""
        from scipy.linalg import expm
        t = np.asarray(t_s, dtype=np.float64)
        if t.ndim != 1 or t.size < 1 or np.any(np.diff(t) < 0.0):
            raise ValueError("FourLevelSystem: t_s must be 1D non-decreasing")
        N = np.zeros(4) if N0 is None else np.asarray(N0, dtype=np.float64).copy()
        if N0 is None:
            N[0] = self.N_total_m3
        if N.shape != (4,) or np.any(N < 0.0):
            raise ValueError("FourLevelSystem: N0 must be 4 non-negative populations")
        A = self.rate_matrix()
        out = np.empty((t.size, 4))
        t_prev = 0.0
        for i, ti in enumerate(t):
            if ti > t_prev:
                N = expm(A * (ti - t_prev)) @ N
                t_prev = float(ti)
            out[i] = N
        return out

    def steady_state(self) -> np.ndarray:
        """Exact steady state (null vector of A scaled to N_total); W_p = 0 -> all in N0."""
        if self.W_p_per_s == 0.0:
            return np.array([self.N_total_m3, 0.0, 0.0, 0.0])
        w, t32, t21, t10 = self.W_p_per_s, self.tau_32_s, self.tau_21_s, self.tau_10_s
        # chain balance: w N0 = N3/t32 = N2/t21 = N1/t10
        n0 = 1.0
        n3, n2, n1 = w * t32, w * t21, w * t10
        tot = n0 + n1 + n2 + n3
        return self.N_total_m3 * np.array([n0, n1, n2, n3]) / tot

    def inversion_ss_m3(self) -> float:
        """Steady-state inversion N2 - N1 = W_p N0_ss (tau_21 - tau_10) [m^-3]."""
        ss = self.steady_state()
        return float(ss[2] - ss[1])
