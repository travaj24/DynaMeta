"""
EffectModel: the generalized material-response seam (the v0.3 keystone).

Where the original NToEpsMap mapped ONLY carrier density n -> a scalar eps, an EffectModel maps
the full local-field bundle {n, E, T, ...} -> eps, which may be a TENSOR (3x3 per point) for
anisotropic effects (Pockels, liquid-crystal, ...). The bridge assembles the per-region field
bundle on the aligned grid and calls EffectModel.eps(fields, lambda).

  eps(fields, lambda_m) -> ndarray
      scalar response: shape (...,)            (broadcast of the field grids)
      tensor response: shape (..., 3, 3)       (per-point 3x3 permittivity)

The default `OpticalModelEffect` adapts the existing scalar OpticalModels (Drude / Constant /
Tabulated): it reads fields['n'] (None for a density-independent model) and ignores the rest, so
the free-carrier / ENZ path is unchanged. Richer effects -- a PockelsEffect reading fields['E'],
a ThermoOpticEffect reading fields['T'], ... -- implement the SAME interface and can be COMPOSED
(see `ComposedEffect`: a background response + summed delta-eps contributions).

Pure numpy: no devsim/ngsolve. Convention: exp(-i omega t), Im(eps) > 0 for absorbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EffectModel(Protocol):
    def eps(self, fields: dict, lambda_m: float): ...   # (...,) scalar OR (..., 3, 3) tensor


@dataclass
class OpticalModelEffect:
    """Adapt a scalar OpticalModel (Drude / Constant / Tabulated) to the EffectModel field-bundle
    interface: read fields['n'] (None for a density-independent model) and return the scalar eps
    grid. This is the default response for every material until a richer field-dependent / tensor
    effect is attached -- so the carrier/ENZ results are byte-for-byte unchanged."""
    optical: object   # any object exposing eps(lambda_m, *, n_m3=None)

    def eps(self, fields: dict, lambda_m: float):
        return self.optical.eps(lambda_m, n_m3=fields.get("n"))


def as_tensor(eps) -> np.ndarray:
    """Promote a scalar eps (or scalar grid) to a (..., 3, 3) isotropic tensor eps*I, so a scalar
    and a tensor effect can be summed/composed uniformly. A value already shaped (..., 3, 3) is
    returned unchanged."""
    eps = np.asarray(eps, dtype=np.complex128)
    if eps.ndim >= 2 and eps.shape[-2:] == (3, 3):
        return eps
    eye = np.eye(3, dtype=np.complex128)
    return eps[..., None, None] * eye


@dataclass
class ComposedEffect:
    """Compose effects on a background: eps = background.eps + sum(delta.eps). All contributions
    are promoted to (...,3,3) tensors via as_tensor before summing, so a scalar background (e.g.
    a Drude/Constant response) and tensor deltas (e.g. Pockels) add consistently. Used for an
    EO layer with a background index + a field-induced birefringence, or thermo-optic + free
    carrier on the same region."""
    background: EffectModel
    deltas: List[EffectModel]

    def eps(self, fields: dict, lambda_m: float):
        total = as_tensor(self.background.eps(fields, lambda_m))
        for d in self.deltas:
            total = total + as_tensor(d.eps(fields, lambda_m))
        return total


# ---- field-effect electro-optic mechanisms (Phase 1) -------------------------------------

# Voigt index map for a SYMMETRIC 3x3 tensor: (i,j) -> contracted index I in 0..5
# (1=xx,2=yy,3=zz,4=yz,5=xz,6=xy, here 0-based 0..5).
_VOIGT = ((0, 5, 4), (5, 1, 3), (4, 3, 2))


def _voigt6_to_full(b6: np.ndarray) -> np.ndarray:
    """(...,6) Voigt vector -> (...,3,3) symmetric tensor."""
    b6 = np.asarray(b6)
    out = np.empty(b6.shape[:-1] + (3, 3), dtype=b6.dtype)
    for i in range(3):
        for j in range(3):
            out[..., i, j] = b6[..., _VOIGT[i][j]]
    return out


def _E_vec(fields: dict) -> np.ndarray:
    """The applied field from the bundle as (...,3). Accepts a 3-vector (uniform) or a (...,3)
    grid. Raises if absent -- a field-effect model needs E."""
    if "E" not in fields or fields["E"] is None:
        raise ValueError("field-effect EffectModel requires fields['E'] (V/m); none supplied "
                         "(run the electrostatic driver first)")
    E = np.asarray(fields["E"], dtype=np.float64)
    if E.shape[-1] != 3:
        raise ValueError("fields['E'] must have a trailing length-3 axis (Ex,Ey,Ez)")
    return E


@dataclass
class PockelsEffect:
    """Linear electro-optic (Pockels) tensor response -- an EffectModel reading fields['E'].
    The impermeability B = eps^-1 shifts LINEARLY with the applied field, ΔB_I = sum_k r_Ik E_k
    (Voigt I=1..6, k=x,y,z), so eps(E) = (B0 + ΔB)^-1 with B0 = eps_bg^-1. Crystal principal axes
    are assumed aligned with the lab x,y,z (a rotation can be composed/added later). At E=0 (or
    r=0) eps -> eps_bg exactly. eps_bg is the zero-field permittivity (e.g. diag(no^2,no^2,ne^2)
    for a uniaxial crystal); r_voigt is the 6x3 EO tensor in m/V; E is in V/m."""
    eps_bg: np.ndarray   # (3,3) zero-field permittivity
    r_voigt: np.ndarray  # (6,3) electro-optic tensor [m/V]

    def eps(self, fields: dict, lambda_m: float):
        E = _E_vec(fields)                                           # (...,3)
        B0 = np.linalg.inv(np.asarray(self.eps_bg, dtype=np.complex128))   # (3,3)
        dB6 = np.tensordot(E, np.asarray(self.r_voigt, dtype=np.float64),
                           axes=([-1], [1]))                         # (...,6): dB_I = r_Ik E_k
        B = B0 + _voigt6_to_full(dB6).astype(np.complex128)          # (...,3,3)
        return np.linalg.inv(B)


@dataclass
class KerrEffect:
    """DC-Kerr (quadratic EO) tensor response -- an EffectModel reading fields['E']. Simplified
    ISOTROPIC quadratic model: the impermeability shifts as ΔB = s_kerr * |E|^2 * I, so
    eps(E) = (eps_bg^-1 + s_kerr |E|^2 I)^-1. At E=0 or s_kerr=0, eps -> eps_bg. s_kerr in m^2/V^2.
    (A full Kerr uses the rank-4 s_ijkl tensor; this isotropic form is the common scalar-Kerr
    approximation, adequate for centrosymmetric media without an in-plane preferred axis.)"""
    eps_bg: np.ndarray   # (3,3) zero-field permittivity
    s_kerr: float        # scalar quadratic EO coefficient [m^2/V^2]

    def eps(self, fields: dict, lambda_m: float):
        E = _E_vec(fields)
        e2 = np.sum(E ** 2, axis=-1)                                  # (...,) |E|^2
        B0 = np.linalg.inv(np.asarray(self.eps_bg, dtype=np.complex128))
        eye = np.eye(3, dtype=np.complex128)
        B = B0 + float(self.s_kerr) * e2[..., None, None] * eye       # (...,3,3)
        return np.linalg.inv(B)


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
        mag = np.sqrt(np.sum(E ** 2, axis=-1))                        # (...,) |E|
        return complex(self.eps_bg) + 1j * float(self.beta) * mag     # scalar grid (...,)


@dataclass
class ThermoOpticModel:
    """Thermo-optic (dn/dT) SCALAR response -- an EffectModel reading fields['T'] (kelvin). The
    refractive index varies linearly with temperature: eps(T) = (n_ref + dn_dT*(T - T_ref))^2 with
    n_ref = sqrt(eps_ref). At T = T_ref (or dn_dT = 0) eps -> eps_ref exactly. Isotropic (the
    common case for thermo-optic media); an anisotropic dn/dT would use a tensor variant."""
    eps_ref: complex     # permittivity at T_ref
    dn_dT: float         # dn/dT [1/K]
    T_ref: float = 300.0

    def eps(self, fields: dict, lambda_m: float):
        if "T" not in fields or fields["T"] is None:
            raise ValueError("ThermoOpticModel requires fields['T'] (kelvin); none supplied "
                             "(run the thermal driver first)")
        T = np.asarray(fields["T"], dtype=np.float64)
        n = np.sqrt(complex(self.eps_ref)) + float(self.dn_dT) * (T - float(self.T_ref))
        return n ** 2
