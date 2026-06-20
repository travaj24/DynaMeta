"""Two-timescale (heterogeneous-rate) dephasing lineshape for the QD-SOA homogeneous line -- the
generalization of the single-rate single-Lorentzian line the gain model and the spectral line-filter
assume. A biexponential dipole-correlation function

    m(t) = w1 exp(-2 pi gamma1 |t|) + (1 - w1) exp(-2 pi gamma2 |t|)

(two dephasing channels: a fast core-broadening rate and a slow one) Fourier-transforms (Wiener-
Khinchin) to a two-component homogeneous line

    L(df) = w1 Lor(df; gamma1) + (1 - w1) Lor(df; gamma2),   Lor(df; g) = (g/pi)/(df^2 + g^2),

a SUPER-Lorentzian shape: a sharper core (the narrow channel) on top of HEAVIER far wings (the broad
channel sits above a single Lorentzian's 1/f^2 tail). w1 = 1 (or gamma1 = gamma2) recovers the single
Lorentzian. SI; ASCII.

TERMINOLOGY (honest): each exp(-2 pi g|t|) channel is itself a MARKOVIAN single-rate dephasing; a
weighted sum of two is a HETEROGENEOUS / two-rate (bi-Markovian) line, not genuine bath-memory non-
Markovianity (which needs a NON-EXPONENTIAL m(t) -- m'(0) = 0, a Gaussian-curved core, e.g. the Kubo
finite-bath-correlation-time line, whose m(t) stays monotone yet is motional-narrowed / SUB-Lorentzian
-- the OPPOSITE direction, a deeper refinement). The 'multi-timescale dephasing -> non-Lorentzian line'
content is the physical deliverable; the single Lorentzian is the single-rate limit.
"""
from __future__ import annotations

import numpy as np


def lorentzian_area(df_Hz, gamma_Hz):
    """AREA-NORMALIZED Lorentzian Lor(df; g) = (g/pi)/(df^2 + g^2) [1/Hz], HWHM = g, integral = 1.
    This is the Fourier transform of exp(-2 pi g |t|)."""
    g = float(gamma_Hz)
    df = np.asarray(df_Hz, dtype=np.float64)
    return (g / np.pi) / (df * df + g * g)


def biexp_memory_kernel(t_s, gamma1_Hz, gamma2_Hz, w1):
    """Biexponential dipole-correlation memory m(t) = w1 exp(-2 pi g1 |t|) + (1-w1) exp(-2 pi g2 |t|)
    (the two-timescale / heterogeneous-rate dephasing memory; each exp channel is itself Markovian, so
    this is rate-heterogeneity, not a bath-memory kernel). m(0) = 1; w1 in [0,1]. Its Fourier transform
    is the two-component lineshape nonmarkovian_lineshape."""
    t = np.abs(np.asarray(t_s, dtype=np.float64))
    return (float(w1) * np.exp(-2.0 * np.pi * float(gamma1_Hz) * t)
            + (1.0 - float(w1)) * np.exp(-2.0 * np.pi * float(gamma2_Hz) * t))


def nonmarkovian_lineshape(df_Hz, gamma1_Hz, gamma2_Hz, w1):
    """Area-normalized two-component (super-Lorentzian) homogeneous line L(df) = w1 Lor(df; g1) +
    (1-w1) Lor(df; g2) [1/Hz] -- the Wiener-Khinchin transform of biexp_memory_kernel (a HETEROGENEOUS
    two-rate line, not genuine bath-memory; the 'nonmarkovian' function name is kept for API stability).
    Reduces to a single Lorentzian when w1 = 1 (or g1 = g2). w1 in [0,1]; gamma in Hz (HWHM each)."""
    if not (0.0 <= float(w1) <= 1.0):
        raise ValueError("nonmarkovian_lineshape: w1 must be in [0, 1]")
    return (float(w1) * lorentzian_area(df_Hz, gamma1_Hz)
            + (1.0 - float(w1)) * lorentzian_area(df_Hz, gamma2_Hz))
