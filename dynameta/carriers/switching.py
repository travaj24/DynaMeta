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
    """PCM crystallization kinetics under a thermal pulse (amorphous<->crystalline chalcogenides:
    GST, Sb2S3). NOT VO2 -- VO2 is a Mott/Peierls insulator-to-metal transition, not an
    amorphous<->crystalline melt-quench, so it is NOT described by this JMAK model (use a two-endpoint
    Bruggeman blend for VO2). Crystalline fraction x in [0,1]:
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
        rates = np.asarray(self.rate_K(T), dtype=np.float64)   # audit S6-16: one vectorized
        for i in range(1, t.size):                              # Arrhenius eval, not per step
            dt = t[i] - t[i - 1]
            if T[i] >= self.T_melt_K:                    # melt-quench -> amorphous
                theta = 0.0
                x[i] = 0.0
            elif T[i] > self.T_glass_K:                  # crystallize (accumulate kinetic time)
                theta += float(rates[i]) * dt
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


@dataclass
class PCMClassicalNucleation:
    """PCM crystallization resolved into CLASSICAL NUCLEATION THEORY + growth (roadmap R12) -- the
    deeper companion to the fixed-exponent PCMSwitching JMAK above (which stays byte-identical).

        nucleation rate  I(T) = I0 exp(-(W*(T) + Ea_d)/(kB T))   [1/(m^3 s)], zero outside (Tg, Tm)
        barrier          W*(T) = 16 pi sigma^3 / (3 dG_v(T)^2)   (CNT spherical cap)
        driving force    dG_v(T) = dHf_vol (Tm - T)/Tm           [J/m^3] (Thompson-Spaepen linear)
        growth velocity  u(T) = u0 exp(-Ea_g/(kB T)) (1 - exp(-dG_v Omega/(kB T)))  [m/s]

    The crystallized fraction follows the KJMA extended-volume integral (spherical grains):

        X(t) = 1 - exp(-X_ext),  X_ext(t) = (4 pi/3) [ INT_0^t I(t') (U(t) - U(t'))^3 dt'
                                                       + N0 U(t)^3 ]  + X0,
        U(t) = INT_0^t u dt'    (cumulative growth length; N0 = pre-existing nuclei [1/m^3])

    evaluated EXACTLY in O(n) by expanding the cube into four cumulative moments
    Sk = INT I U^k dt (k = 0..3):  X_ext = (4 pi/3)(U^3 S0 - 3 U^2 S1 + 3 U S2 - S3 + N0 U^3).
    (An incremental kernel += I (u dt)^3 dt scheme is WRONG -- every previously-born nucleus keeps
    growing, so the convolution must carry U(t) - U(t').)

    Reduces-to limits the oracle uses: constant T -> X = 1 - exp(-(pi/3) I u^3 t^4) (Avrami n = 4,
    machine vs the moment scheme); I0 = 0 with N0 > 0 -> X = 1 - exp(-(4 pi/3) N0 u^3 t^3) (n = 3,
    growth-only); the equivalent JMAK rate K_eff = ((pi/3) I u^3)^(1/4) fed to PCMSwitching with
    avrami_n = 4 reproduces the same isothermal trajectory. Melt (T >= Tm) resets to amorphous
    (accumulators + x -> 0, matching PCMSwitching); frozen (T <= Tg) holds. enabled=False -> x stays
    x0 EXACTLY (code-path off-switch). Pure numpy."""
    I0_per_m3_s: float
    sigma_J_m2: float
    dHf_J_m3: float
    Omega_m3: float
    u0_m_s: float
    Ea_d_J: float = 0.0
    Ea_g_J: float = 0.0
    T_glass_K: float = 450.0
    T_melt_K: float = 900.0
    N0_per_m3: float = 0.0
    enabled: bool = True

    def __post_init__(self):
        if not self.enabled:
            return
        if not (self.I0_per_m3_s >= 0 and self.sigma_J_m2 > 0 and self.dHf_J_m3 > 0
                and self.Omega_m3 > 0 and self.u0_m_s > 0):
            raise ValueError("PCMClassicalNucleation: I0 >= 0; sigma, dHf, Omega, u0 must be > 0")
        if self.Ea_d_J < 0 or self.Ea_g_J < 0 or self.N0_per_m3 < 0:
            raise ValueError("PCMClassicalNucleation: Ea_d_J, Ea_g_J, N0_per_m3 must be >= 0")
        if not (self.T_melt_K > self.T_glass_K > 0):
            raise ValueError("PCMClassicalNucleation: require T_melt_K > T_glass_K > 0")

    def _dG_v(self, T):
        return self.dHf_J_m3 * (self.T_melt_K - T) / self.T_melt_K          # J/m^3, > 0 below Tm

    def nucleation_rate_I(self, T_K):
        """I(T) [1/(m^3 s)]; EXACTLY zero outside (T_glass, T_melt) (mask, not clip)."""
        T = np.asarray(T_K, dtype=np.float64)
        inside = (T > self.T_glass_K) & (T < self.T_melt_K)
        Ts = np.where(inside, T, 0.5 * (self.T_glass_K + self.T_melt_K))    # safe eval point
        dgv = np.maximum(self._dG_v(Ts), 1e-300)
        W = 16.0 * np.pi * self.sigma_J_m2 ** 3 / (3.0 * dgv * dgv)         # J
        ex = np.clip(-(W + self.Ea_d_J) / (KB * Ts), -700.0, 0.0)
        out = self.I0_per_m3_s * np.exp(ex)
        return np.where(inside, out, 0.0) if out.ndim else (float(out) if inside else 0.0)

    def growth_velocity_u(self, T_K):
        """u(T) [m/s]; EXACTLY zero outside (T_glass, T_melt)."""
        T = np.asarray(T_K, dtype=np.float64)
        inside = (T > self.T_glass_K) & (T < self.T_melt_K)
        Ts = np.where(inside, T, 0.5 * (self.T_glass_K + self.T_melt_K))
        dgv = np.maximum(self._dG_v(Ts), 0.0)
        ex_g = np.clip(-self.Ea_g_J / (KB * Ts), -700.0, 0.0)
        ex_d = np.clip(-dgv * self.Omega_m3 / (KB * Ts), -700.0, 0.0)
        out = self.u0_m_s * np.exp(ex_g) * (1.0 - np.exp(ex_d))
        return np.where(inside, out, 0.0) if out.ndim else (float(out) if inside else 0.0)

    def fraction_isothermal(self, t_s, T_K: float, x0: float = 0.0) -> np.ndarray:
        """Closed-form isothermal KJMA: X = 1 - exp(-(pi/3) I u^3 t^4 - (4 pi/3) N0 u^3 t^3 - X0),
        X0 = -ln(1 - x0) the pre-accumulated extended volume."""
        t = np.asarray(t_s, dtype=np.float64)
        if not self.enabled:
            return np.full(t.shape, float(x0)) if t.ndim else float(x0)
        I = float(self.nucleation_rate_I(float(T_K)))
        u = float(self.growth_velocity_u(float(T_K)))
        X0 = -np.log(1.0 - min(max(float(x0), 0.0), 1.0 - 1e-15)) if x0 > 0 else 0.0
        Xe = (np.pi / 3.0) * I * u ** 3 * t ** 4 + (4.0 * np.pi / 3.0) * self.N0_per_m3 * u ** 3 * t ** 3
        return 1.0 - np.exp(-(Xe + X0))

    def integrate(self, t_s, T_K, x0: float = 0.0) -> np.ndarray:
        """x(t) over a temperature pulse T(t) via the exact O(n) moment scheme (module docstring).
        Melt-quench (T >= Tm) resets to amorphous; frozen (T <= Tg) holds; enabled=False -> x0."""
        t = np.asarray(t_s, dtype=np.float64)
        T = np.asarray(T_K, dtype=np.float64)
        if t.shape != T.shape or t.ndim != 1 or t.size < 2:
            raise ValueError("t_s and T_K must be 1D equal-length arrays of length >= 2")
        if not self.enabled:
            return np.full(t.shape, float(x0))
        x = np.empty_like(t)
        x[0] = float(x0)
        X0 = -np.log(1.0 - min(max(float(x0), 0.0), 1.0 - 1e-15)) if x0 > 0 else 0.0
        U = 0.0
        S0 = S1 = S2 = S3 = 0.0
        I_prev = float(self.nucleation_rate_I(float(T[0])))
        u_prev = float(self.growth_velocity_u(float(T[0])))
        for i in range(1, t.size):
            dt = t[i] - t[i - 1]
            if T[i] >= self.T_melt_K:                      # melt-quench -> amorphous (reset all)
                U = S0 = S1 = S2 = S3 = 0.0
                X0 = 0.0
                x[i] = 0.0
                I_prev, u_prev = 0.0, 0.0
                continue
            I_i = float(self.nucleation_rate_I(float(T[i])))
            u_i = float(self.growth_velocity_u(float(T[i])))
            if T[i] <= self.T_glass_K:                     # frozen: nothing advances
                x[i] = x[i - 1]
                I_prev, u_prev = I_i, u_i
                continue
            U_new = U + 0.5 * (u_prev + u_i) * dt          # trapezoid cumulative growth length
            S0 += 0.5 * (I_prev + I_i) * dt
            S1 += 0.5 * (I_prev * U + I_i * U_new) * dt
            S2 += 0.5 * (I_prev * U ** 2 + I_i * U_new ** 2) * dt
            S3 += 0.5 * (I_prev * U ** 3 + I_i * U_new ** 3) * dt
            U = U_new
            Xe = (4.0 * np.pi / 3.0) * (U ** 3 * S0 - 3.0 * U ** 2 * S1 + 3.0 * U * S2 - S3
                                        + self.N0_per_m3 * U ** 3)
            x[i] = 1.0 - np.exp(-(max(Xe, 0.0) + X0))
            I_prev, u_prev = I_i, u_i
        return x
