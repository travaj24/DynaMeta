"""
Graphene surface conductivity (Kubo) + analytic conductive-sheet reflection (roadmap Phase 4c).

A gated graphene sheet modulates light through its SURFACE conductivity sigma(E_F, omega) -- a 2D
current response, NOT a volume permittivity -- so it enters Maxwell as a surface-current boundary
condition, with the exact normal-incidence reflection/transmission of a conductive sheet between
two media n1, n2:

    r = (n1 - n2 - Z0 sigma) / (n1 + n2 + Z0 sigma),   t = 2 n1 / (n1 + n2 + Z0 sigma),
    R = |r|^2,   T = (Re n2 / Re n1) |t|^2,   A = 1 - R - T   (Z0 = 1/(eps0 c), free-space impedance)

The Fermi level E_F is gate-tunable; the interband term is PAULI-BLOCKED once 2|E_F| > hbar*omega
(the sheet bleaches to near-transparency), which is the gate-tunable graphene electro-absorption
modulator. sigma is the standard finite-temperature Kubo result (Falkovsky 2007): a Drude
intraband term + the universal-conductivity interband term sigma0 = e^2/(4 hbar).

Convention exp(-i omega t): a passive sheet has Re(sigma) > 0 (absorbing). SI units (E_F in J,
lambda in m, sigma in siemens, tau in s). Pure numpy; no devsim/ngsolve. The FEM surface-current
boundary condition (a sheet Robin term in the curl-curl weak form) is IMPLEMENTED --
optics.solver.solve_fem(sheet_bcs={'iface_z<nm>': sigma}), validated against this module's
sheet_rt oracle in validation/graphene_sheet_fem.py (docstring corrected per audit C6-5; the
old text claimed it was a follow-on).
"""

from __future__ import annotations

import warnings

import numpy as np

from dynameta.constants import Q_E, HBAR, KB, EPS0, C_LIGHT
from dynameta.core.backend import array_namespace

Z0 = 1.0 / (EPS0 * C_LIGHT)             # free-space wave impedance, ~376.730 ohm
SIGMA0 = Q_E ** 2 / (4.0 * HBAR)        # universal interband AC conductivity of graphene, ~6.085e-5 S


def graphene_sigma(E_F_J, lambda_m, *, tau_s: float = 1.0e-13, T_K: float = 300.0):
    """Complex sheet conductivity sigma(E_F, omega) [S] (Kubo, finite T) = intraband Drude +
    interband. `E_F_J` is the Fermi level relative to the Dirac point (J), `tau_s` the carrier
    relaxation time, `T_K` the temperature. Re(sigma) > 0 (passive, exp(-i omega t)).

    Backend-agnostic in E_F (numpy / cupy / jax via array_namespace): dispatching on E_F_J -- the
    gate-tunable knob -- keeps the conductivity inside a JAX trace, so d sigma / d E_F is available
    for gate-tuned-modulator inverse design. lambda/tau/T are fixed scalars. (NumPy path returns a
    0-d complex array, which complex()/.real consume exactly as the old python-complex return.)"""
    if not (tau_s > 0 and T_K > 0 and lambda_m > 0):
        raise ValueError("tau_s, T_K, lambda_m must be > 0")
    xp = array_namespace(E_F_J)
    omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
    kT = KB * float(T_K)
    EF = xp.abs(xp.asarray(E_F_J))
    hw = HBAR * omega
    # intraband Drude: i e^2 * W / (pi hbar^2 (omega + i/tau)),  W = E_F + 2 kT ln(1+exp(-E_F/kT))
    W = EF + 2.0 * kT * xp.log1p(xp.exp(-EF / kT))
    sigma_intra = 1j * Q_E ** 2 * W / (np.pi * HBAR ** 2 * (omega + 1j / float(tau_s)))
    # interband (Falkovsky-Varlamov 2007): the EXACT finite-T universal-conductivity real part
    # Re = sigma0 sinh(hw/2kT)/(cosh(E_F/kT)+cosh(hw/2kT)) -- the thermal factor that -> a hard Pauli
    # step (sigma0 for hw>2E_F, 0 below) as T->0, with the correct (small) blocked-tail residual
    # (an arctan smoothing overestimates that tail ~100x; audit GRAPH-1). Im is the matching KK log.
    re = SIGMA0 * xp.sinh(hw / (2.0 * kT)) / (xp.cosh(EF / kT) + xp.cosh(hw / (2.0 * kT)))
    im = -SIGMA0 * (1.0 / (2.0 * np.pi)) * xp.log((hw + 2.0 * EF) ** 2
                                                  / ((hw - 2.0 * EF) ** 2 + (2.0 * kT) ** 2))
    return sigma_intra + (re + 1j * im)


def sheet_rt(n1, n2, sigma, *, theta_deg: float = 0.0):
    """Reflection/transmission of a conductive sheet between semi-infinite media n1 | sheet | n2 at
    NORMAL incidence (theta_deg kept for signature symmetry; only 0 supported). Returns
    (r, t, R, T, A) with A = 1 - R - T the sheet absorbed fraction. sigma -> 0 recovers the bare
    Fresnel (n1 - n2)/(n1 + n2)."""
    if abs(float(theta_deg)) > 1e-9:
        raise ValueError("sheet_rt currently supports normal incidence only (theta_deg = 0)")
    n1, n2, s = complex(n1), complex(n2), complex(sigma)
    if n1.imag != 0.0:
        raise ValueError("incidence medium n1 must be lossless (Im(n1)=0) for the R/T/A budget")
    if s.real < -1e-15:                          # passivity (exp(-i omega t)): Re(sigma) >= 0
        warnings.warn("sheet_rt: Re(sigma) < 0 is an ACTIVE/gain sheet (violates passivity for "
                      "exp(-i omega t)); the returned A = 1 - R - T will be negative (T > 1).",
                      RuntimeWarning, stacklevel=2)
    denom = n1 + n2 + Z0 * s
    r = (n1 - n2 - Z0 * s) / denom
    t = 2.0 * n1 / denom
    R = float(abs(r) ** 2)
    T = float((n2.real / n1.real) * abs(t) ** 2)
    return r, t, R, T, float(1.0 - R - T)


def gate_tuned_reflection(E_F_J, lambda_m, n1, n2, *, tau_s: float = 1.0e-13,
                          T_K: float = 300.0):
    """Convenience: sigma(E_F) -> (R, T, A) for a graphene sheet between n1 and n2 -- the gate-tuned
    graphene modulator response at one Fermi level."""
    s = graphene_sigma(E_F_J, lambda_m, tau_s=tau_s, T_K=T_K)
    _, _, R, T, A = sheet_rt(n1, n2, s)
    return R, T, A
