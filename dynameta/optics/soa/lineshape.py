"""Non-Markovian (multi-timescale) dephasing lineshape for the QD-SOA homogeneous line -- the
generalization of the single-Lorentzian (Markovian, single dephasing rate T2) line the gain model and
the spectral line-filter assume. A biexponential dipole-correlation memory

    m(t) = w1 exp(-2 pi gamma1 |t|) + (1 - w1) exp(-2 pi gamma2 |t|)

(two dephasing channels: a fast core-broadening rate and a slow one) Fourier-transforms (Wiener-
Khinchin) to a two-component homogeneous line

    L_NM(df) = w1 Lor(df; gamma1) + (1 - w1) Lor(df; gamma2),   Lor(df; g) = (g/pi)/(df^2 + g^2),

a SUB-Lorentzian / Voigt-like shape (sharper peak + heavier or lighter wings than a single Lorentzian
of the same area). w1 = 1 (or gamma1 = gamma2) recovers the Markovian single-Lorentzian line. SI; ASCII.
(Non-Markovian dephasing / memory-function lineshape; the gain model's single Lorentzian is the
Markovian limit.)
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
    (the non-Markovian dephasing kernel). m(0) = 1; w1 in [0,1]. Its Fourier transform is the
    two-component lineshape nonmarkovian_lineshape."""
    t = np.abs(np.asarray(t_s, dtype=np.float64))
    return (float(w1) * np.exp(-2.0 * np.pi * float(gamma1_Hz) * t)
            + (1.0 - float(w1)) * np.exp(-2.0 * np.pi * float(gamma2_Hz) * t))


def nonmarkovian_lineshape(df_Hz, gamma1_Hz, gamma2_Hz, w1):
    """Area-normalized non-Markovian homogeneous line L_NM(df) = w1 Lor(df; g1) + (1-w1) Lor(df; g2)
    [1/Hz] -- the Wiener-Khinchin transform of biexp_memory_kernel. Reduces to a single Lorentzian when
    w1 = 1 (or g1 = g2). w1 in [0,1]; gamma in Hz (HWHM of each component)."""
    if not (0.0 <= float(w1) <= 1.0):
        raise ValueError("nonmarkovian_lineshape: w1 must be in [0, 1]")
    return (float(w1) * lorentzian_area(df_Hz, gamma1_Hz)
            + (1.0 - float(w1)) * lorentzian_area(df_Hz, gamma2_Hz))
