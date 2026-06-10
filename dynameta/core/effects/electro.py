"""Electro-refractive effects: Pockels, Kerr, Franz-Keldysh.

Split from the former monolithic effects.py; see the package __init__ docstring for
the EffectModel seam contract. Bodies are verbatim. Pure numpy (scipy only lazily for
the Voigt lineshape).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.core.backend import array_namespace, to_backend
from dynameta.core.effects.base import _E_vec, _voigt6_to_full

@dataclass
class PockelsEffect:
    """Linear electro-optic (Pockels) tensor response -- an EffectModel reading fields['E'].
    The impermeability B = eps^-1 shifts LINEARLY with the applied field, dB_I = sum_k r_Ik E_k
    (Voigt I=1..6, k=x,y,z), so eps(E) = (B0 + dB)^-1 with B0 = eps_bg^-1. Crystal principal axes
    are assumed aligned with the lab x,y,z (a rotation can be composed/added later). At E=0 (or
    r=0) eps -> eps_bg exactly. eps_bg is the zero-field permittivity (e.g. diag(no^2,no^2,ne^2)
    for a uniaxial crystal); r_voigt is the 6x3 EO tensor in m/V; E is in V/m."""
    eps_bg: np.ndarray   # (3,3) zero-field permittivity
    r_voigt: np.ndarray  # (6,3) electro-optic tensor [m/V]

    def eps(self, fields: dict, lambda_m: float):
        E = _E_vec(fields)                                           # (...,3)
        xp = array_namespace(E)                                      # dispatch on the runtime field
        eps_bg = to_backend(self.eps_bg, xp)                         # lift stored params to E's backend
        r_voigt = to_backend(self.r_voigt, xp)
        B0 = xp.linalg.inv(xp.asarray(eps_bg) + 0j)                  # (3,3)
        dB6 = xp.tensordot(E, xp.asarray(r_voigt), axes=([-1], [1]))  # (...,6): dB_I = r_Ik E_k
        B = B0 + (_voigt6_to_full(dB6) + 0j)                         # (...,3,3)
        return xp.linalg.inv(B)


@dataclass
class KerrEffect:
    """DC-Kerr (quadratic EO) tensor response -- an EffectModel reading fields['E']. Simplified
    ISOTROPIC quadratic model: the impermeability shifts as dB = s_kerr * |E|^2 * I, so
    eps(E) = (eps_bg^-1 + s_kerr |E|^2 I)^-1. At E=0 or s_kerr=0, eps -> eps_bg. s_kerr in m^2/V^2.
    (A full Kerr uses the rank-4 s_ijkl tensor; this isotropic form is the common scalar-Kerr
    approximation, adequate for centrosymmetric media without an in-plane preferred axis.)"""
    eps_bg: np.ndarray   # (3,3) zero-field permittivity
    s_kerr: float        # scalar quadratic EO coefficient [m^2/V^2]

    def eps(self, fields: dict, lambda_m: float):
        E = _E_vec(fields)
        xp = array_namespace(E)
        eps_bg = to_backend(self.eps_bg, xp)
        e2 = xp.sum(E ** 2, axis=-1)                                  # (...,) |E|^2
        B0 = xp.linalg.inv(xp.asarray(eps_bg) + 0j)
        eye = xp.eye(3) + 0j
        B = B0 + float(self.s_kerr) * e2[..., None, None] * eye       # (...,3,3)
        return xp.linalg.inv(B)


@dataclass
class FranzKeldyshEffect:
    """Franz-Keldysh electro-ABSORPTION -- an EffectModel reading fields['E']. SIMPLIFIED
    phenomenological model: the field opens a sub-bandgap absorption that grows with |E|, added as
    an isotropic Im(eps) bump on a background, Im(eps) += beta * |E|. (A rigorous FK model uses the
    Airy-function below-gap absorption + a Kramers-Kronig Re(eps) shift; this captures the
    field-on -> loss-up trend for a first electro-absorption modulator and reduces to eps_bg at
    E=0.) eps_bg is the (scalar) zero-field permittivity; beta in (eps units)/(V/m)."""
    eps_bg: complex      # zero-field scalar permittivity
    beta: float          # electro-absorption coefficient, Im(eps) per (V/m)

    def eps(self, fields: dict, lambda_m: float):
        E = _E_vec(fields)
        xp = array_namespace(E)
        mag = xp.sqrt(xp.sum(E ** 2, axis=-1))                        # (...,) |E|
        return xp.asarray(self.eps_bg) + 1j * float(self.beta) * mag  # scalar grid (...,)
