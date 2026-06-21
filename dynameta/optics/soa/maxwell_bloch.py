"""Coherent Maxwell-Bloch gain ensemble for the QD-SOA -- the sub-T2 COHERENT polarization dynamics the
rate-equation marcher (which adiabatically eliminates the polarization, gain = occupations) cannot
capture: Rabi flopping, photon echo, pulse-area / self-induced-transparency, and the coherent->incoherent
crossover as the dephasing time T2 shrinks. (Research-grade gap from the 2026-06-20 physics-gap audit.)

Each inhomogeneous QD group j is a real 2-level Bloch vector (u_j, v_j, w_j) -- u = in-phase coherence,
v = in-quadrature coherence, w = inversion (2 rho - 1) -- with detuning delta_j = 2 pi (nu_j - nu_s),
driven by the (real) Rabi field Omega(t) = mu E(t)/hbar [rad/s] (Allen-Eberly convention):

    u_j' =  delta_j v_j                 - u_j / T2
    v_j' = -delta_j u_j + Omega(t) w_j  - v_j / T2
    w_j' = -Omega(t) v_j                - (w_j - w_eq_j) / T1.

The macroscopic coherence P(t) = sum_j w_weight_j (u_j + i v_j) radiates the field; v_j (the
in-quadrature part) carries the stimulated energy exchange. In the WEAK-field steady state
v_j = Omega w_eq_j (1/T2) / ((1/T2)^2 + delta_j^2), so the small-signal gain spectrum sum_j w_weight_j
v_j(nu_s) is the Lorentzian-weighted inversion -- EXACTLY the shape of QDGainModel.material_gain_per_m
(the rate-equation gain is the T2->0 / weak-field limit of this coherent model). Working in the Rabi
frequency Omega [rad/s] keeps the coherent dynamics (Rabi area, echo) free of the absolute dipole /
power calibration; the gain SHAPE ties back to the device model. SI; ASCII; exp(-i omega t).
"""
from __future__ import annotations

import numpy as np


class MaxwellBlochEnsemble:
    """Inhomogeneously-broadened 2-level Bloch ensemble (one Bloch vector per QD group). Drive it with a
    prescribed Rabi field Omega(t) [rad/s] via evolve(); read the coherent macroscopic polarization and
    the inversion. from_model() builds it from a QDGainModel steady state (detunings, weights, equilibrium
    inversion, T2 = 1/(pi fwhm_hom), T1 = tau_sp)."""

    def __init__(self, nu_j_Hz, w_weight, w_eq, nu_s_Hz, T1_s, T2_s):
        self.nu_j = np.asarray(nu_j_Hz, dtype=np.float64)
        self.w_weight = np.asarray(w_weight, dtype=np.float64)
        self.w_eq = np.asarray(w_eq, dtype=np.float64)
        if not (self.nu_j.shape == self.w_weight.shape == self.w_eq.shape and self.nu_j.ndim == 1):
            raise ValueError("MaxwellBlochEnsemble: nu_j, w_weight, w_eq must be equal-length 1-D")
        self.ng = self.nu_j.size
        self.delta = 2.0 * np.pi * (self.nu_j - float(nu_s_Hz))     # group detunings [rad/s]
        self.gT1 = 0.0 if (T1_s is None or not np.isfinite(T1_s)) else 1.0 / float(T1_s)
        self.gT2 = 0.0 if (T2_s is None or not np.isfinite(T2_s)) else 1.0 / float(T2_s)

    @classmethod
    def from_model(cls, model, drive, *, nu_s_Hz=None, T1_s=None, T2_s=None):
        """Build from a QDGainModel at injection `drive`: per-group detunings nu_j, weights w_j, the
        equilibrium inversion w_eq = 2 rho_GS_ss - 1 (the unsaturated steady state), T2 = 1/(pi fwhm_hom)
        (the homogeneous-linewidth dephasing time) and T1 = tau_sp. Excitonic models only."""
        nu_s = float(model.p.nu0_Hz if nu_s_Hz is None else nu_s_Hz)
        y = model.steady_state(drive)
        w_eq = 2.0 * np.asarray(model.rho_GS(y), dtype=np.float64) - 1.0
        T2 = (1.0 / (np.pi * float(model.p.fwhm_hom_Hz))) if T2_s is None else T2_s
        T1 = float(model.p.tau_sp_s) if T1_s is None else T1_s
        return cls(model.nu_j, model.w_j, w_eq, nu_s, T1, T2)

    def _rhs(self, u, v, w, Om):
        du = self.delta * v - self.gT2 * u
        dv = -self.delta * u + Om * w - self.gT2 * v
        dw = -Om * v - self.gT1 * (w - self.w_eq)
        return du, dv, dw

    def evolve(self, Omega_t, dt_s, *, w0=None):
        """Integrate the Bloch ensemble under the prescribed REAL Rabi field Omega_t (nt,) [rad/s] at
        time step dt (explicit RK4, Omega held across each step). Initial coherence u = v = 0, inversion
        w = w0 (default the equilibrium w_eq). Returns a dict:
          t [s]; P (nt,) complex macroscopic coherence sum_j w_weight_j (u_j + i v_j);
          w_mean (nt,) the weighted mean inversion sum_j w_weight_j w_j;
          u, v, w the final per-group Bloch components."""
        Om = np.asarray(Omega_t, dtype=np.float64)
        if Om.ndim != 1 or Om.size < 1:
            raise ValueError("evolve: Omega_t must be a 1-D field waveform")
        nt = Om.size
        u = np.zeros(self.ng)
        v = np.zeros(self.ng)
        w = self.w_eq.copy() if w0 is None else np.broadcast_to(
            np.asarray(w0, dtype=np.float64), (self.ng,)).copy()
        P = np.empty(nt, dtype=np.complex128)
        w_mean = np.empty(nt)
        for n in range(nt):
            P[n] = np.sum(self.w_weight * (u + 1j * v))
            w_mean[n] = float(np.sum(self.w_weight * w))
            O = Om[n]                                            # zero-order hold across the step
            k1 = self._rhs(u, v, w, O)
            k2 = self._rhs(u + 0.5 * dt_s * k1[0], v + 0.5 * dt_s * k1[1], w + 0.5 * dt_s * k1[2], O)
            k3 = self._rhs(u + 0.5 * dt_s * k2[0], v + 0.5 * dt_s * k2[1], w + 0.5 * dt_s * k2[2], O)
            k4 = self._rhs(u + dt_s * k3[0], v + dt_s * k3[1], w + dt_s * k3[2], O)
            u = u + dt_s / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            v = v + dt_s / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            w = w + dt_s / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        return {"t": np.arange(nt) * dt_s, "P": P, "w_mean": w_mean, "u": u, "v": v, "w": w}

    def linear_gain_shape(self, nu_grid_Hz):
        """Weak-field steady-state gain SHAPE g(nu) (arb. units) = sum_j w_weight_j w_eq_j (1/T2) /
        ((1/T2)^2 + (2 pi (nu - nu_j))^2) -- the Lorentzian-weighted inversion. Up to a constant this is
        QDGainModel.material_gain_per_m(rho_eq, nu); the rate-equation gain is this coherent model's
        weak-field / fast-dephasing limit. Requires T2 < inf."""
        if self.gT2 <= 0.0:
            raise ValueError("linear_gain_shape: needs a finite T2 (gT2 > 0)")
        nu = np.atleast_1d(np.asarray(nu_grid_Hz, dtype=np.float64))
        d = 2.0 * np.pi * (nu[:, None] - self.nu_j[None, :])
        lor = self.gT2 / (self.gT2 * self.gT2 + d * d)          # (n_nu, ng)
        return lor @ (self.w_weight * self.w_eq)
