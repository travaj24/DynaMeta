"""
Time-dependent STATE drivers for reconfigurable modulators (roadmap Phase-4 REMAINING): a
phase-change-material (PCM) crystallization driver and a liquid-crystal (LC) director relaxation
driver. They PRODUCE the time-dependent state -- crystalline_fraction(t) / director_angle(t) -- that
core.effects.PCMModel / LiquidCrystalModel read to give the switched eps. Pure numpy, SI.

PCM: the crystalline fraction x in [0,1] evolves under a temperature pulse T(t) by the
Johnson-Mehl-Avrami-Kolmogorov (JMAK) isokinetic-additivity rule x = 1 - exp(-theta^n), theta(t) =
INT_0^t K(T) dt' the accumulated kinetic time, K(T) = K0 exp(-E_a / kB T) (Arrhenius). Above the melt
threshold the material melt-quenches (theta -> 0, x -> 0, amorphous); below the glass/crystallization
onset it is frozen. LC: after the field is removed the director relaxes back to planar with the
single-constant time tau = gamma d^2 / (K pi^2); theta(t) = theta0 exp(-t/tau).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import KB


@dataclass
class PCMSwitching:
    """PCM crystallization kinetics under a thermal pulse (GST/Sb2S3/VO2-like). Crystalline fraction
    x in [0,1]:
      * CRYSTALLIZE (T_glass_K < T < T_melt_K): JMAK isokinetic additivity x = 1 - exp(-theta^n),
        theta += K(T) dt, K(T) = K0 exp(-E_a / kB T).
      * MELT-QUENCH (T >= T_melt_K): theta -> 0, x -> 0 (amorphous).
      * FROZEN (T <= T_glass_K): no change (theta held).
    """
    K0_per_s: float          # JMAK rate prefactor [1/s]
    E_a_J: float             # crystallization activation energy [J]
    T_glass_K: float         # crystallization onset
    T_melt_K: float          # melt / amorphization threshold
    avrami_n: float = 3.0    # Avrami exponent

    def __post_init__(self):
        if not (self.K0_per_s > 0 and self.E_a_J > 0 and self.avrami_n > 0):
            raise ValueError("K0_per_s, E_a_J, avrami_n must be > 0")
        if not (self.T_melt_K > self.T_glass_K > 0):
            raise ValueError("require T_melt_K > T_glass_K > 0")

    def rate_K(self, T_K):
        """Arrhenius JMAK rate K(T) = K0 exp(-E_a / kB T) [1/s]."""
        return self.K0_per_s * np.exp(-self.E_a_J / (KB * np.asarray(T_K, dtype=np.float64)))

    def fraction_isothermal(self, t_s, T_K: float, x0: float = 0.0) -> np.ndarray:
        """Closed-form isothermal Avrami curve x(t) = 1 - exp(-(K(T) (t + t0))^n), with t0 the kinetic
        time already accumulated to reach x0. The integrate() result reduces to this at constant T."""
        n = float(self.avrami_n)
        K = float(self.rate_K(T_K))
        t = np.asarray(t_s, dtype=np.float64)
        theta0 = (-np.log(1.0 - min(max(float(x0), 0.0), 1.0 - 1e-15))) ** (1.0 / n) if x0 > 0 else 0.0
        theta = K * t + theta0
        return 1.0 - np.exp(-theta ** n)

    def integrate(self, t_s, T_K, x0: float = 0.0) -> np.ndarray:
        """Integrate x(t) over a temperature pulse: t_s, T_K equal-length arrays. Returns x(t)
        (same length). Isokinetic additivity for crystallization; melt-quench resets to amorphous."""
        t = np.asarray(t_s, dtype=np.float64)
        T = np.asarray(T_K, dtype=np.float64)
        if t.shape != T.shape or t.ndim != 1 or t.size < 2:
            raise ValueError("t_s and T_K must be 1D equal-length arrays of length >= 2")
        n = float(self.avrami_n)
        x = np.empty_like(t)
        x[0] = float(x0)
        theta = (-np.log(1.0 - min(max(float(x0), 0.0), 1.0 - 1e-15))) ** (1.0 / n) if x0 > 0 else 0.0
        for i in range(1, t.size):
            dt = t[i] - t[i - 1]
            if T[i] >= self.T_melt_K:                    # melt-quench -> amorphous
                theta = 0.0
                x[i] = 0.0
            elif T[i] > self.T_glass_K:                  # crystallize (accumulate kinetic time)
                theta += float(self.rate_K(T[i])) * dt
                x[i] = 1.0 - np.exp(-theta ** n)
            else:                                        # frozen below the onset
                x[i] = x[i - 1]
        return x


@dataclass
class LCRelaxation:
    """Liquid-crystal director relaxation (field-off) for the single-elastic-constant nematic cell:
    after the aligning field is removed the midplane tilt relaxes as theta(t) = theta0 exp(-t/tau),
    tau = gamma d^2 / (K pi^2) (rotational viscosity gamma, Frank constant K, cell thickness d)."""
    K_elastic_N: float       # Frank elastic constant [N]
    gamma_visc_Pa_s: float    # rotational viscosity [Pa s]
    d_m: float               # cell thickness [m]

    def __post_init__(self):
        if not (self.K_elastic_N > 0 and self.gamma_visc_Pa_s > 0 and self.d_m > 0):
            raise ValueError("K_elastic_N, gamma_visc_Pa_s, d_m must be > 0")

    def tau_s(self) -> float:
        """Director relaxation time tau = gamma d^2 / (K pi^2) [s]."""
        return float(self.gamma_visc_Pa_s * self.d_m ** 2 / (self.K_elastic_N * np.pi ** 2))

    def relax(self, t_s, theta0_rad: float) -> np.ndarray:
        """Midplane tilt theta(t) = theta0 exp(-t/tau) relaxing from theta0 toward planar (0)."""
        return float(theta0_rad) * np.exp(-np.asarray(t_s, dtype=np.float64) / self.tau_s())
