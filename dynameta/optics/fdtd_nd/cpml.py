"""CPML absorbing-boundary coefficient builder (z-direction).

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import EPS0, MU0



def _cpml_z(nz, dz, dt, npml, n_super=1.0, n_sub=1.0, m=3.0, ma=1.0, kappa_max=5.0, alpha_max=0.2, R0=1.0e-6):
    """CFS-CPML stretched-coordinate coefficients along z (the propagation axis; x is periodic so needs
    no PML). Returns (kappa, b, c) on the E-grid (z=k*dz) and the H-grid (z=(k+1/2)*dz). Roden-Gedney:
    sigma/kappa graded polynomially over the outer `npml` cells each end, alpha (CFS) graded the other
    way; b=exp(-(sigma/kappa+alpha)dt/eps0), c=sigma/(sigma*kappa+kappa^2*alpha)(b-1). Outside the PML
    sigma=alpha=0 -> b=1,c=0 -> plain FDTD.

    n_super / n_sub (default 1 = vacuum) impedance-match the conductivity to the END MEDIUM each PML
    terminates. audit C3-5-adjacent fix: under this eps0-normalized convention
    (b = exp(-(sig/kap + alp) dt/eps0)) the one-way attenuation is exp(-n eta0 Int(sigma dz)) --
    the exponent ALREADY carries n through the medium's wavevector -- so the MATCHED scaling is
    sigma ~ 1/n (Gedney sigma_opt ~ 1/(eta0 dz n)), NOT the previous x n (which over-drove sigma
    by n^2 and raised the discrete-reflection echo floor 2-14x for n = 1.5-4; the 1/n law holds
    the vacuum floor flat across all n, probe-verified). Low-z PML scales by 1/n_super, high-z by
    1/n_sub; defaults (n=1) are byte-identical to vacuum."""
    eta0 = np.sqrt(MU0 / EPS0)
    sig_max = -(m + 1.0) * np.log(R0) / (2.0 * eta0 * npml * dz)

    def _coeffs(zpos):                                   # zpos: cell-index position along z (nz,)
        d_lo = np.clip(npml - zpos, 0.0, None)           # depth into the low-z PML (cells)
        d_hi = np.clip(zpos - (nz - 1 - npml), 0.0, None)  # depth into the high-z PML
        rho = np.clip(np.maximum(d_lo, d_hi) / npml, 0.0, 1.0)
        nfac = np.where(d_lo >= d_hi, 1.0 / n_super, 1.0 / n_sub)   # matched ~ 1/n (see docstring)
        sig = sig_max * nfac * rho ** m
        kap = 1.0 + (kappa_max - 1.0) * rho ** m
        alp = alpha_max * (1.0 - rho) ** ma
        b = np.exp(-(sig / kap + alp) * dt / EPS0)
        denom = sig * kap + kap ** 2 * alp
        c = np.where(denom > 0.0, sig / np.where(denom > 0.0, denom, 1.0) * (b - 1.0), 0.0)
        return kap, b, c
    ke, be, ce = _coeffs(np.arange(nz, dtype=float))
    kh, bh, ch = _coeffs(np.arange(nz, dtype=float) + 0.5)
    return (ke, be, ce), (kh, bh, ch)
