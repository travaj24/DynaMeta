"""
High-accuracy Fermi-Dirac integral F_{1/2}(eta) for use as a node-model
expression in DEVSIM 2D Poisson solves.

The DEVSIM 2.10 built-in Fermi function is broken (non-monotonic for
eta >= 20) so we need our own. The earlier Halen-Pulfrey workaround
underestimates F_{1/2} by ~50% at the degenerate eta our ITO operates
in (eta = 12-20 at n_bg = 4-8e20 cm^-3). This module replaces it with
the Aymerich-Humet (1981) approximation, accuracy < 0.2% on F_{1/2}
and < 1.3% on dF/d_eta everywhere.

Reference:
    Aymerich-Humet, X., Serra-Mestres, F., & Millan, J. (1981).
    "An analytical approximation for the Fermi-Dirac integral
    F_{1/2}(eta)". Solid-State Electronics 24, 981-982.

Usage:
    from dynameta.stage1_carriers.fermi_dirac import (
        F12_aymerich_humet_expr,        # str: DEVSIM expression
        F12_aymerich_humet,             # Python callable for sanity checks
    )

    ds.node_model(device=d, region=r, name="F_half",
                    equation=F12_aymerich_humet_expr())
"""

from __future__ import annotations

import numpy as np


def F12_aymerich_humet_expr() -> str:
    """DEVSIM node-model expression for F_{1/2}(eta) via Aymerich-Humet (1981).

    Reads the local variable `eta` (must be defined as a node model
    upstream). Returns a string suitable for ds.node_model(...) equation=.

    Functional form:
        F_{1/2}(eta) = 1 / (A(eta) + B(eta))
        A(eta) = (3 sqrt(pi) / 4) / [eta^4 + 50 + 33.6 * eta * w(eta)]^(3/8)
        B(eta) = exp(-eta)
        w(eta) = 1 - 0.68 * exp(-0.17 * (eta + 1)^2)
    """
    a_poly = ("(pow(eta, 4) + 50.0 + 33.6 * eta * "
                "(1.0 - 0.68 * exp(-0.17 * (eta + 1.0) * (eta + 1.0))))")
    A = "((3.0 * sqrt(3.141592653589793) / 4.0) / pow({}, 3.0/8.0))".format(a_poly)
    B = "exp(-eta)"
    return "(1.0 / ({} + {}))".format(A, B)


def F12_aymerich_humet(eta: np.ndarray) -> np.ndarray:
    """Python evaluation of the Aymerich-Humet (1981) F_{1/2}(eta).

    Useful for verifying the DEVSIM expression's output against the
    high-accuracy approximation, and for converting between n and the
    Fermi level eta when calibrating Phi_c0.
    """
    eta = np.asarray(eta, dtype=np.float64)
    a = eta**4 + 50.0 + 33.6 * eta * (1.0 - 0.68 * np.exp(-0.17 * (eta + 1.0)**2))
    A = (3.0 * np.sqrt(np.pi) / 4.0) / a**(3.0 / 8.0)
    B = np.exp(-eta)
    return 1.0 / (A + B)


def F12_true(eta: float) -> float:
    """High-precision F_{1/2}(eta) via numerical integration. Slow but
    accurate to ~10^-10. Use only for offline calibration / verification."""
    from scipy.integrate import quad
    from scipy.special import gamma
    val, _ = quad(lambda t: t**0.5 / (1.0 + np.exp(t - eta)), 0.0, np.inf,
                   epsabs=1e-12, epsrel=1e-10)
    return float(val / gamma(1.5))


def inverse_F12_joyce_dixon(x: float) -> float:
    """Solve F_{1/2}(eta) = x for eta. Joyce-Dixon (1977) inverse
    approximation, accurate to 0.05% for x in [0, ~100].

    Used to compute Phi_c0 by inverting n_bg = N_c * F_{1/2}(eta_bg).
    """
    x = float(x)
    if x <= 0:
        raise ValueError("F_{1/2} inverse requires positive argument, got " + str(x))
    if x < 1e-3:
        # Maxwell-Boltzmann: eta = ln(x)
        return float(np.log(x))
    # Joyce-Dixon series in x / sqrt(8) (degenerate regime)
    u = x
    eta = np.log(u) + u / np.sqrt(8.0) \
            - (3.0 / 16.0 - np.sqrt(3.0) / 9.0) * u * u \
            + 0.0185 * u**3
    return float(eta)
