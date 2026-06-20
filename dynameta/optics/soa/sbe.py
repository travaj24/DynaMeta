"""Reduced k-resolved Semiconductor-Bloch-Equation (SBE) gain spectrum -- the microscopic many-body /
coherent-polarization model the phenomenological QD gain (Phase 14 Haug-Koch) and the single-Lorentzian
line approximate. Steady-state linear-response SBE for the interband polarization p(k):

    (hbar w - e_k + i hbar/T2) p_k + (1 - f_e,k - f_h,k) sum_k' W(k,k') p_k'
        = -(1 - f_e,k - f_h,k) mu E,

    e_k = Eg + hbar^2 k^2 / 2 m_r - sum_k' W(k,k') (f_e,k' + f_h,k')   (band-gap renormalization),

solved per frequency for the complex interband susceptibility chi(w) = (mu / (eps0 d_qw)) sum_k mu p_k
g_k / E, g_k = k dk / (2 pi) the 2-D radial k-sum measure. f_e/f_h are quasi-equilibrium Fermi-Dirac at
the sheet density N_2d. W is a REDUCED 1-D statically-screened Coulomb kernel W(k,k') = V0/(|k-k'|+kappa)
(NOT the full 2-D angular-averaged Coulomb). coulomb_V0 = 0 -> the FREE-CARRIER (diagonal) limit, whose
closed form is the exact oracle. Coulomb ON gives the excitonic ENHANCEMENT + the BGR red-shift.

SCOPE (honest): a REDUCED demonstration SBE -- 1 band pair, parabolic isotropic, a MODEL 1-D screened
kernel, quasi-equilibrium (not kinetically-solved) carriers, steady-state linear response. It captures
the SBE STRUCTURE (Pauli blocking, T2 dephasing, Coulomb enhancement, BGR) at the k-resolved level, not
a full multi-band kinetic SBE. The absolute chi MAGNITUDE carries a parameterized oscillator-strength /
geometric prefactor (mu, d_qw) that is NOT calibrated to an experimental gain value -- the SPECTRAL
SHAPE, the free-carrier/Coulomb contrast, the transparency point, the gain SIGN, and Kramers-Kronig
consistency are the physical content; multiply by a measured oscillator-strength factor for absolute
1/m. SI; ASCII. (Chow-Koch screened-Hartree-Fock SBE, reduced.)
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import EPS0, HBAR, KB, Q_E


def _quasi_fermi_2d(N_2d_m2, m_eff, T_K):
    """2-D quasi-Fermi level E_F [J] (from the band edge) for density N at temperature T, parabolic
    band: N = (m kT / pi hbar^2) ln(1 + exp(E_F/kT)) -> E_F = kT ln(exp(pi hbar^2 N / m kT) - 1)."""
    kT = KB * float(T_K)
    x = np.pi * HBAR ** 2 * float(N_2d_m2) / (float(m_eff) * kT)
    return kT * np.log(np.expm1(x))                          # expm1 keeps the low-density limit clean


def reduced_sbe_susceptibility(hbar_omega_eV, *, Eg_eV=0.95, m_e=0.067, m_h=0.45, N_2d_m2=3.0e16,
                               T_K=300.0, T2_s=100e-15, eps_r=12.5, d_qw_m=8e-9, mu_Cm=5e-29,
                               coulomb_V0=0.0, screen_kappa_m=2.0e8, nk=240, kmax_factor=6.0):
    """Reduced k-resolved SBE interband susceptibility chi(w) over the photon energies hbar_omega_eV.
    m_e, m_h in units of the free-electron mass. Returns (hw_eV, chi_complex). coulomb_V0=0 -> the
    free-carrier diagonal limit. The material gain is g(w) = -(w/(n c)) Im chi, n = sqrt(eps_r)."""
    hw = np.asarray(hbar_omega_eV, dtype=np.float64) * Q_E    # -> J
    me, mh = float(m_e) * 9.1093837015e-31, float(m_h) * 9.1093837015e-31
    mr = me * mh / (me + mh)
    kT = KB * float(T_K)
    # radial k-grid out to a few thermal wavevectors
    k_th = np.sqrt(2.0 * mr * kT) / HBAR
    k = np.linspace(1.0e-3 * k_th, kmax_factor * k_th, int(nk))
    dk = k[1] - k[0]
    gk = k * dk / (2.0 * np.pi)                               # 2-D radial sum measure (1/A) sum_k
    # quasi-equilibrium occupations
    EFe = _quasi_fermi_2d(N_2d_m2, me, T_K)
    EFh = _quasi_fermi_2d(N_2d_m2, mh, T_K)
    f_e = 1.0 / (1.0 + np.exp((HBAR ** 2 * k ** 2 / (2.0 * me) - EFe) / kT))
    f_h = 1.0 / (1.0 + np.exp((HBAR ** 2 * k ** 2 / (2.0 * mh) - EFh) / kT))
    inv = 1.0 - f_e - f_h                                     # Pauli inversion factor (k,)
    # reduced screened-Coulomb kernel W(k,k') = V0/(|k-k'| + kappa)  [J m^2 effective]
    V0 = float(coulomb_V0)
    if V0 != 0.0:
        W = V0 / (np.abs(k[:, None] - k[None, :]) + float(screen_kappa_m))   # (k,k')
        bgr = (W * (f_e + f_h)[None, :]) @ gk                # exchange / band-gap renormalization (k,)
    else:
        W = None
        bgr = np.zeros_like(k)
    e_k = float(Eg_eV) * Q_E + HBAR ** 2 * k ** 2 / (2.0 * mr) - bgr          # renormalized transition
    gamma = HBAR / float(T2_s)                               # homogeneous dephasing [J]
    mu = float(mu_Cm)
    pref = mu / (EPS0 * float(d_qw_m))                       # chi = pref sum_k mu p_k g_k (E=1)
    chi = np.empty(hw.size, dtype=np.complex128)
    src = -inv * mu                                          # source -(1-f_e-f_h) mu E, E=1
    for i, w in enumerate(hw):
        diag = (w - e_k + 1j * gamma)
        if W is None:
            p = src / diag                                   # free-carrier diagonal solve
        else:
            M = np.diag(diag) + (inv[:, None] * W) * gk[None, :]   # SBE matrix (Coulomb folds g_k')
            p = np.linalg.solve(M, src)
        chi[i] = pref * np.sum(mu * p * gk)
    return np.asarray(hbar_omega_eV, dtype=np.float64), chi


def sbe_gain_per_m(hbar_omega_eV, chi, eps_r=12.5):
    """Material gain g(w) [1/m] = -(w/(n c)) Im chi, n = sqrt(eps_r). g > 0 where Im chi < 0
    (amplification, exp(-i w t) convention)."""
    n = np.sqrt(float(eps_r))
    w = np.asarray(hbar_omega_eV, dtype=np.float64) * Q_E / HBAR
    return -(w / (n * 2.99792458e8)) * np.imag(chi)
