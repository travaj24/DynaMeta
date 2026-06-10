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

__all__ = ["FourLevelSystem", "small_signal_gain_per_m",
           "cavity_photon_lifetime_s", "threshold_inversion_m3", "pump_threshold_per_s",
           "relaxation_oscillation_rad_s"]


def small_signal_gain_per_m(kappa_C2_kg: float, dN_m3: float, n_refr: float,
                            dw_rad_s: float) -> float:
    """Line-center small-signal INTENSITY gain g0 = kappa dN/(n c eps0 dw) [1/m] (negative for
    dN < 0 = absorption)."""
    if not (kappa_C2_kg > 0.0 and n_refr > 0.0 and dw_rad_s > 0.0):
        raise ValueError("gain_medium: kappa, n and dw must be > 0")
    return float(kappa_C2_kg) * float(dN_m3) / (n_refr * C_LIGHT * EPS0 * float(dw_rad_s))


def cavity_photon_lifetime_s(L_m: float, n_group: float, R1: float, R2: float,
                             alpha_i_per_m: float = 0.0) -> float:
    """Cold-cavity photon lifetime of a Fabry-Perot resonator [s]:

        tau_p = t_RT / (ln(1/(R1 R2)) + 2 alpha_i L),    t_RT = 2 n_g L / c.

    R1/R2 are INTENSITY reflectances; mirror loss uses ln(1/R), NOT the low-loss shortcut
    (1-R) (56% wrong at R ~ 0.57). n_group is the GROUP index (= the phase index for a
    dispersionless cold host; an inverted line at clamp adds slow light ~ c*alpha_m/dw)."""
    if not (L_m > 0.0 and n_group > 0.0 and 0.0 < R1 <= 1.0 and 0.0 < R2 <= 1.0
            and alpha_i_per_m >= 0.0):
        raise ValueError("cavity_photon_lifetime_s: L, n_g > 0; R1, R2 in (0, 1]; alpha_i >= 0")
    loss = float(np.log(1.0 / (R1 * R2))) + 2.0 * alpha_i_per_m * L_m
    if loss <= 0.0:
        raise ValueError("cavity_photon_lifetime_s: a lossless cavity (R1=R2=1, alpha_i=0) has "
                         "no finite photon lifetime")
    return 2.0 * n_group * L_m / (C_LIGHT * loss)


def threshold_inversion_m3(kappa_C2_kg: float, n_refr: float, dw_rad_s: float, L_m: float,
                           R1: float, R2: float, *, Gamma: float = 1.0,
                           alpha_i_per_m: float = 0.0) -> float:
    """Threshold inversion density of a Fabry-Perot laser [m^-3]: the round-trip condition
    Gamma g_th = alpha_m + alpha_i with the distributed mirror loss alpha_m =
    ln(1/(R1 R2))/(2 L), inverted through the line-center gain g0 = kappa dN/(n c eps0 dw):

        dN_th = (alpha_m + alpha_i) n c eps0 dw / (Gamma kappa).

    n_refr is the PHASE index (it enters through the wave impedance in g0); the group-index
    subtlety cancels in dN_th (Gamma v_g g_th = 1/tau_p uses n_g on both sides)."""
    if not (0.0 < Gamma <= 1.0):
        raise ValueError("threshold_inversion_m3: Gamma in (0, 1]")
    if not (L_m > 0.0 and 0.0 < R1 <= 1.0 and 0.0 < R2 <= 1.0 and alpha_i_per_m >= 0.0):
        raise ValueError("threshold_inversion_m3: L > 0; R1, R2 in (0, 1]; alpha_i >= 0")
    alpha_m = float(np.log(1.0 / (R1 * R2))) / (2.0 * L_m)
    g_th = (alpha_m + alpha_i_per_m) / Gamma
    # invert small_signal_gain_per_m (validates kappa/n/dw)
    g_unit = small_signal_gain_per_m(kappa_C2_kg, 1.0, n_refr, dw_rad_s)
    return g_th / g_unit


def pump_threshold_per_s(dN_th_m3: float, sysm: "FourLevelSystem") -> float:
    """Pump rate W_p [1/s, per ground-state atom] at which the four-level steady-state
    inversion reaches dN_th: inverting dN_ss(W) = N W (tau21 - tau10)/(1 + W Sum(tau)),

        W_p_th = dN_th / ( N (tau21 - tau10) - dN_th (tau10 + tau21 + tau32) ).

    Raises when the threshold is unreachable (denominator <= 0: the chain saturates below
    dN_th at any pump)."""
    if not (dN_th_m3 > 0.0):
        raise ValueError("pump_threshold_per_s: dN_th must be > 0")
    s = sysm
    den = s.N_total_m3 * (s.tau_21_s - s.tau_10_s) - dN_th_m3 * (
        s.tau_10_s + s.tau_21_s + s.tau_32_s)
    if den <= 0.0:
        raise ValueError("pump_threshold_per_s: threshold inversion {:.3e} m^-3 is unreachable "
                         "for this level system (chain saturates below it)".format(dN_th_m3))
    return float(dN_th_m3 / den)


def relaxation_oscillation_rad_s(r_pump: float, tau_p_s: float, tau_21_s: float):
    """Relaxation-oscillation (omega_RO, gamma_RO) of a class-B four-level laser [rad/s, 1/s]:

        gamma_RO = r / (2 tau21),   omega_RO = sqrt( (r-1)/(tau_p tau21) - gamma_RO^2 )

    with r = dN0(W_p)/dN_th the EXACT zero-field inversion ratio (NOT W_p/W_p_th -- the chain
    denominator makes them differ). Returns omega_RO = 0.0 when overdamped. Validity: class-B
    (T2 = 2/dw << tau_p << tau21) and the ideal-4-level reduction (tau10, tau32 << tau21)."""
    if not (r_pump > 1.0 and tau_p_s > 0.0 and tau_21_s > 0.0):
        raise ValueError("relaxation_oscillation_rad_s: r > 1 (above threshold), taus > 0")
    gamma = r_pump / (2.0 * tau_21_s)
    w2 = (r_pump - 1.0) / (tau_p_s * tau_21_s) - gamma * gamma
    return (float(np.sqrt(w2)) if w2 > 0.0 else 0.0), float(gamma)


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
