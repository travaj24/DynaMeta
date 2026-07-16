"""REL3: electromigration (Black equation + Blech immortality) for the current-carrying metal
(mirror, gate traces, contacts).

    MTTF = A * J^(-n) * exp(Ea / (kB * T)),   n ~ 2 (void-NUCLEATION-limited -- Korhonen
    sigma ~ J sqrt(t) gives t_nuc ~ J^-2; n ~ 1 is the void-GROWTH-limited regime; the labels
    were previously swapped vs Lloyd 1991 / JEP122 -- audit P3),
    Ea ~ 0.55 eV (Cu grain-boundary) .. ~0.9 eV (Al / Cu surface-diffusion capped)

Blech immortality: below the critical current-density-length product the back-stress gradient stops
the void from ever nucleating -- J * L < (J*L)_crit (~1000-3000 A/cm = 1e5-3e5 A/m for Cu) gives an
effectively INFINITE MTTF. The pre-factor A is GEOMETRY-SCALED (a um-scale contact differs from a
mm-scale interconnect by ~1e4-1e5x) -- always anchor it on a representative qualification point via
EmParams.calibrated().

DRIVER NOTE (stale text corrected per audit C6-5/6.3): the contact-current driver HAS shipped --
carriers.contact_current + drivers.reliability_glue feed the solved DEVSIM contact current straight
into these models (validation/contact_current_drivers.py); the drive current I [A] may still be
supplied as an external design parameter. J = I / (w * t) from the trace geometry.
Miner damage accumulation handles time-varying (J, T) duty cycles. Pure numpy; oracles in
validation/reliability_em.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dynameta.constants import KB_EV_K   # eV/K, single source (audit 6.3)


def current_density_A_m2(I_A: float, width_m: float, thickness_m: float) -> float:
    """J = I / (w * t) for a rectangular trace cross-section."""
    if not (width_m > 0.0 and thickness_m > 0.0):
        raise ValueError("EM: trace width and thickness must be > 0")
    if I_A < 0.0:
        raise ValueError("EM: use the current magnitude (I >= 0)")
    return float(I_A / (width_m * thickness_m))


def black_mttf_s(J_A_m2, T_K, *, A_s: float, n_exp: float = 2.0, Ea_eV: float = 0.55,
                 J_ref_A_m2: float = 1.0e9):
    """Black MTTF [s] = A * (J/J_ref)^(-n) * exp(Ea/kBT). J_ref non-dimensionalizes the power law so
    A carries plain seconds (1e9 A/m^2 = 1e5 A/cm^2, a typical stress density). J = 0 -> inf
    (no current, no electromigration). Broadcasts over J/T arrays."""
    if not (A_s > 0.0):
        raise ValueError("EM: A_s must be > 0")
    if not (n_exp > 0.0):
        raise ValueError("EM: current exponent n must be > 0 (1 = nucleation, 2 = growth)")
    if Ea_eV < 0.0:
        raise ValueError("EM: Ea_eV must be >= 0")
    J = np.asarray(J_A_m2, dtype=np.float64)
    if np.any(J < 0.0):
        raise ValueError("EM: J must be >= 0 (use the magnitude)")
    T = np.asarray(T_K, dtype=np.float64)
    if np.any(T <= 0.0):
        raise ValueError("EM: T_K must be > 0")
    with np.errstate(divide="ignore"):
        out = A_s * (J / J_ref_A_m2) ** (-n_exp) * np.exp(Ea_eV / (KB_EV_K * T))
    return out                                              # J = 0 -> inf, the physical limit


def blech_immortal(J_A_m2: float, length_m: float, *, jl_crit_A_m: float = 2.0e5) -> bool:
    """True when J * L < (J*L)_crit -- the back-stress gradient suppresses void nucleation entirely
    (Blech 1976). (J*L)_crit ~ 1e5-3e5 A/m (1000-3000 A/cm) for Cu; short/wide features are the
    design knob for immortality."""
    if not (length_m > 0.0):
        raise ValueError("EM: length_m must be > 0")
    if not (jl_crit_A_m > 0.0):
        raise ValueError("EM: jl_crit_A_m must be > 0")
    if J_A_m2 < 0.0:
        raise ValueError("EM: J must be >= 0")
    return bool(J_A_m2 * length_m < jl_crit_A_m)


@dataclass(frozen=True)
class EmParams:
    """Black-equation parameter set; A_s is geometry-scaled -- prefer .calibrated()."""
    A_s: float = 3.6e6                                      # ~1000 h at (J_ref, 300 K) before Arrhenius
    n_exp: float = 2.0
    Ea_eV: float = 0.55
    J_ref_A_m2: float = 1.0e9
    jl_crit_A_m: float = 2.0e5

    def mttf_s(self, J_A_m2, T_K, *, length_m: float = None):
        """MTTF with the Blech check folded in (length_m given + immortal -> inf). The immortality
        test is ELEMENTWISE over an array J (previously only the scalar-J path was immortal-aware, so
        a vector of current densities silently lost the immortal -> inf elements)."""
        base = black_mttf_s(J_A_m2, T_K, A_s=self.A_s, n_exp=self.n_exp, Ea_eV=self.Ea_eV,
                            J_ref_A_m2=self.J_ref_A_m2)
        if length_m is None:
            return base
        if not (length_m > 0.0):
            raise ValueError("EM: length_m must be > 0")
        if not (self.jl_crit_A_m > 0.0):
            raise ValueError("EM: jl_crit_A_m must be > 0")
        # Blech immortality: J*L < (J*L)_crit -> back-stress gradient suppresses void nucleation -> inf
        immortal = np.asarray(J_A_m2, dtype=np.float64) * float(length_m) < self.jl_crit_A_m
        out = np.where(immortal, np.inf, base)
        return float(out) if np.ndim(J_A_m2) == 0 else out

    @classmethod
    def calibrated(cls, *, J_A_m2: float, T_K: float, mttf_s: float, **kw) -> "EmParams":
        """Anchor A_s on ONE qualified (J, T, MTTF) point for THIS geometry class."""
        base = cls(A_s=1.0, **kw)
        m_unit = float(base.mttf_s(J_A_m2, T_K))
        if not (mttf_s > 0.0 and np.isfinite(m_unit) and m_unit > 0.0):
            raise ValueError("EM: calibration point must give a finite positive MTTF")
        return replace(base, A_s=mttf_s / m_unit)


def miner_time_to_failure_s(t_grid_s, J_of_t, T_of_t, params: EmParams) -> float:
    """Miner damage accumulation for a time-varying duty cycle: failure when
    integral dt / MTTF(J(t), T(t)) = 1. t_grid_s must resolve the J/T waveform; returns inf if the
    accumulated damage never reaches 1 within the grid (caller extends the grid or treats as
    censored). Immortal (J = 0 / Blech) intervals contribute zero damage."""
    t = np.asarray(t_grid_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("EM: t_grid_s must be 1D strictly increasing with >= 2 samples")
    # sample the (scalar-callable) waveforms per grid point, then ONE broadcast Black solve --
    # elementwise identical to the former per-sample mttf_s calls, ~grid-size fewer numpy round
    # trips (audit 6.2)
    J = np.array([float(J_of_t(tt)) for tt in t], dtype=np.float64)
    T = np.array([float(T_of_t(tt)) for tt in t], dtype=np.float64)
    rate = 1.0 / params.mttf_s(J, T)                        # 1/MTTF; inf MTTF -> 0 rate
    # trapezoid cumulative damage
    dmg = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    if dmg[-1] < 1.0:
        return float("inf")
    return float(np.interp(1.0, dmg, t))
