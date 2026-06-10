"""REL8: hot-carrier injection (HCI) -- lucky-electron interface-trap generation where high field
and current coexist. HONESTLY MARGINAL for this vertical MOS-cap (no lateral channel); included for
completeness and for patterned-gate fringing-field layouts.

    interface-trap rate:   dN_it/dt = (A_it / q) * (I_sub / (W * L)) * f(E)       [m^-2 s^-1]
    time-to-failure:       t_HCI = C * (I_sub / W)^(-m) * exp(Ea / (kB * T)),  m ~ 2/3 (Takeda)

NOTE the q factor in the trap rate (current -> particle flux; the audit-corrected dimension) and the
SIGN QUIRK: HCI often WORSENS at LOW temperature (less phonon scattering -> hotter carriers), so Ea
here may legitimately be NEGATIVE (~ -0.1 to -0.2 eV) -- the one mechanism in this package where a
negative activation is physical and therefore allowed.

DRIVER NOTE: the substrate/impact-ionization current I_sub is the roadmap-flagged missing driver (a
DEVSIM impact-ionization model is the follow-on); it is an EXTERNAL parameter here. Pure numpy;
oracles in validation/reliability_hci.py.
"""

from __future__ import annotations

import numpy as np

KB_EV_K = 8.617333262e-5
Q_E = 1.602176634e-19


def trap_generation_rate_per_m2_s(I_sub_A: float, width_m: float, length_m: float, *,
                                  A_it: float = 1.0e-3) -> float:
    """dN_it/dt = (A_it/q) * I_sub/(W*L): the per-area interface-trap generation rate. A_it is the
    dimensionless generation efficiency (traps per injected carrier)."""
    if not (width_m > 0.0 and length_m > 0.0):
        raise ValueError("HCI: width and length must be > 0")
    if I_sub_A < 0.0:
        raise ValueError("HCI: I_sub must be >= 0 (magnitude)")
    if not (A_it > 0.0):
        raise ValueError("HCI: A_it must be > 0")
    return float((A_it / Q_E) * I_sub_A / (width_m * length_m))


def hci_time_to_failure_s(I_sub_A, T_K, *, C_s: float, width_m: float, m_exp: float = 2.0 / 3.0,
                          Ea_eV: float = -0.1, I_ref_A_m: float = 1.0e-3):
    """t_HCI = C * ((I_sub/W)/I_ref)^(-m) * exp(Ea/kBT). I_sub = 0 -> inf (no hot carriers, no
    degradation). Ea may be NEGATIVE (HCI worsens cold -- the documented sign quirk). Broadcasts."""
    if not (C_s > 0.0 and width_m > 0.0 and I_ref_A_m > 0.0):
        raise ValueError("HCI: C_s, width_m, I_ref_A_m must be > 0")
    if not (m_exp > 0.0):
        raise ValueError("HCI: current exponent m must be > 0 (Takeda ~ 2/3)")
    I = np.asarray(I_sub_A, dtype=np.float64)
    if np.any(I < 0.0):
        raise ValueError("HCI: I_sub must be >= 0")
    T = np.asarray(T_K, dtype=np.float64)
    if np.any(T <= 0.0):
        raise ValueError("HCI: T_K must be > 0")
    with np.errstate(divide="ignore"):
        out = C_s * ((I / width_m) / I_ref_A_m) ** (-m_exp) * np.exp(Ea_eV / (KB_EV_K * T))
    return out                                              # I_sub = 0 -> inf
