"""
Drift-diffusion + Poisson physics helpers for DEVSIM Stage 1.

- Universal constants (SI)
- Aymerich-Humet (1981) F_{1/2}(eta), both as Python function and as a
  DEVSIM expression string suitable for `ds.node_model(equation=...)`.
  Accurate to <0.2% on F and <1.3% on dF/d_eta across all eta.
- Phi_c0 calibration: chooses Phi_c0 such that n(Potential=0) = n_bg.
  Convention: Phi_c0 = -eta_bg * V_T. The chi_eV (electron affinity)
  is NOT included -- we work in a Potential frame anchored to V=0 at the
  far-away peripheral ground; absolute band-edge energies don't enter.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import devsim as ds


# ---------------------------------------------------------------------------
# Universal constants (SI)
# ---------------------------------------------------------------------------

Q_E    = 1.602176634e-19       # electron charge [C]
EPS0   = 8.8541878128e-12      # vacuum permittivity [F/m]
KB     = 1.380649e-23          # Boltzmann [J/K]
HBAR   = 1.054571817e-34       # reduced Planck [J*s]
M_E    = 9.1093837015e-31      # electron rest mass [kg]
T_REF  = 300.0                 # reference temperature [K]
V_T    = KB * T_REF / Q_E      # thermal voltage [V] ~ 0.02585 V at 300 K


# ---------------------------------------------------------------------------
# Aymerich-Humet (1981) F_{1/2}(eta)
# ---------------------------------------------------------------------------

def F12_aymerich_humet(eta: float) -> float:
    """High-accuracy F_{1/2}(eta) via Aymerich-Humet (1981).

    Used for offline calibration (Phi_c0 inversion); the DEVSIM
    expression below uses the same formula.
    """
    a = eta**4 + 50.0 + 33.6 * eta * (1.0 - 0.68 * math.exp(-0.17 * (eta + 1.0)**2))
    A = (3.0 * math.sqrt(math.pi) / 4.0) / a**(3.0 / 8.0)
    B = math.exp(-eta)
    return 1.0 / (A + B)


_AH_C = 3.0 * math.sqrt(math.pi) / 4.0    # Aymerich-Humet leading constant


def F12_aymerich_humet_expr(eta_expr: str = "eta") -> str:
    """DEVSIM expression for F_{1/2}(eta) via Aymerich-Humet.

    Args:
      eta_expr : the DEVSIM (sub)expression representing eta. Default
                  "eta" assumes a node_model named "eta" is already
                  defined; pass an inline expression for a single
                  composite equation.

    Notes:
      DEVSIM's expression parser does NOT accept sqrt(...) syntax, so
      the leading constant 3*sqrt(pi)/4 is pre-computed in Python.
    """
    e = "({})".format(eta_expr)
    a_poly = ("({e}*{e}*{e}*{e} + 50.0 + 33.6 * {e} * "
                "(1.0 - 0.68 * exp(-0.17 * ({e} + 1.0) * ({e} + 1.0))))"
                ).format(e=e)
    A = "({c} * pow({a}, -0.375))".format(c=_AH_C, a=a_poly)
    B = "exp(-{})".format(e)
    return "(1.0 / ({} + {}))".format(A, B)


# ---------------------------------------------------------------------------
# Bisection inverse of F_{1/2}
# ---------------------------------------------------------------------------

def invert_F12(target: float, eta_min: float = -20.0, eta_max: float = 80.0,
                tol: float = 1e-10) -> float:
    """Return eta such that F_{1/2}(eta) = target.

    Bisection on the Aymerich-Humet expression. Accurate to ~1e-10
    relative; works across the full range used by our ITO (eta_bg
    from ~5 at low n to ~30 at high n).
    """
    if target <= 0:
        raise ValueError("target must be positive, got {}".format(target))
    f_lo = F12_aymerich_humet(eta_min)
    f_hi = F12_aymerich_humet(eta_max)
    if not (f_lo <= target <= f_hi):
        raise ValueError(
            "target {:.3e} out of F_{{1/2}}([{}, {}]) = "
            "[{:.3e}, {:.3e}]".format(target, eta_min, eta_max, f_lo, f_hi))
    for _ in range(200):
        eta_mid = 0.5 * (eta_min + eta_max)
        f_mid = F12_aymerich_humet(eta_mid)
        if abs(f_mid / target - 1.0) < tol:
            return eta_mid
        if f_mid < target:
            eta_min = eta_mid
        else:
            eta_max = eta_mid
    return 0.5 * (eta_min + eta_max)


# ---------------------------------------------------------------------------
# Phi_c0 calibration
# ---------------------------------------------------------------------------

def setup_phi_c0(device: str, region: str,
                  band_gap_eV: float, chi_eV: float,
                  n_bg_m3: float, m_eff_kg: float) -> float:
    """Set Phi_c0 in a semiconductor region such that n = n_bg at V = 0.

    Convention used:
        eta = (Potential - Phi_c0) / V_T
        n   = N_c * F_{1/2}(eta)

    For n(V=0) = n_bg, we need eta(V=0) = eta_bg, i.e.
        Phi_c0 = -eta_bg * V_T

    The band-gap and electron-affinity arguments are stored as region
    parameters for reference but NOT used in Phi_c0. We work in a local
    Potential frame anchored at V = 0 at the (far-away peripheral)
    ground; absolute band-edge alignment is unnecessary for our
    drift-diffusion solve.

    Args:
        device, region    : DEVSIM handles
        band_gap_eV       : stored as a region parameter (informational)
        chi_eV            : stored as a region parameter (informational)
        n_bg_m3           : background carrier density [m^-3]
        m_eff_kg          : effective mass [kg] used for N_c

    Returns:
        Phi_c0 [V]
    """
    N_c = 2.0 * (m_eff_kg * KB * T_REF / (2.0 * math.pi * HBAR**2))**1.5
    eta_bg = invert_F12(n_bg_m3 / N_c)
    Phi_c0 = -eta_bg * V_T

    ds.set_parameter(device=device, region=region, name="Phi_c0", value=Phi_c0)
    ds.set_parameter(device=device, region=region, name="N_c",    value=N_c)
    ds.set_parameter(device=device, region=region, name="N_D",    value=n_bg_m3)
    # Informational only -- band_gap and chi are not used in equations
    ds.set_parameter(device=device, region=region,
                      name="band_gap_eV_INFO", value=band_gap_eV)
    ds.set_parameter(device=device, region=region,
                      name="chi_eV_INFO",      value=chi_eV)
    return Phi_c0
