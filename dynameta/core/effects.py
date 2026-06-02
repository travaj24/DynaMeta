"""
EffectModel: the generalized material-response seam (the v0.3 keystone).

Where the original NToEpsMap mapped ONLY carrier density n -> a scalar eps, an EffectModel maps
the full local-field bundle {n, E, T, ...} -> eps, which may be a TENSOR (3x3 per point) for
anisotropic effects (Pockels, liquid-crystal, ...). A caller assembles the per-region field bundle
and calls EffectModel.eps(fields, lambda). (Today the bridge auto-assembles only the carrier field
'n'; the field-effect drivers for {E, T} produce their fields for the caller to place in the
bundle -- wiring them through the bridge is a tracked seam. The richer effects are validated
end-to-end at the FEM level by the Phase-1/2 oracles.)

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

from dynameta.constants import HBAR, C_LIGHT


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
    are promoted to (...,3,3) tensors via as_tensor before summing, so a scalar background (e.g. a
    Drude/Constant response) and tensor deltas (e.g. Pockels) add consistently.

    IMPORTANT: each entry in `deltas` MUST be a TRUE delta-eps model -- one that returns ~0 at zero
    drive, i.e. a SHIFT to add on top of `background`. The bundled field-effect models
    (PockelsEffect, KerrEffect, ThermoOpticModel, ...) instead each return the FULL eps (their own
    background PLUS the shift), so composing them DIRECTLY would add a background once per model
    (double-counting). Wrap each such model in a DeltaEffect (which subtracts its zero-drive
    baseline) before composing. Use for an EO layer with a background index + a field-induced
    birefringence, or thermo-optic + free-carrier shifts on the same region."""
    background: EffectModel
    deltas: List[EffectModel]

    def eps(self, fields: dict, lambda_m: float):
        total = as_tensor(self.background.eps(fields, lambda_m))
        for d in self.deltas:
            total = total + as_tensor(d.eps(fields, lambda_m))
        return total


@dataclass
class DeltaEffect:
    """Adapt an absolute-eps EffectModel into a delta-eps contribution for ComposedEffect.

    The bundled field-effect models (PockelsEffect, KerrEffect, ThermoOpticModel, ...) each return
    the FULL permittivity -- their own background PLUS the field/temperature-induced shift -- so
    summing several directly in a ComposedEffect would add a background once per model
    (double-counting). DeltaEffect returns ONLY the shift relative to a zero-drive reference:

        delta_eps(fields) = as_tensor(effect.eps(fields)) - as_tensor(effect.eps(baseline_fields))

    so ComposedEffect(background=base, deltas=[DeltaEffect(pockels, {'E': zeros(3)}), ...]) adds the
    background exactly once and each effect's shift on top. `baseline_fields` is the zero-drive
    reference for THIS effect (e.g. {'E': np.zeros(3)} for Pockels/Kerr, {'T': T_ref} for a
    thermo-optic model)."""
    effect: EffectModel
    baseline_fields: dict

    def eps(self, fields: dict, lambda_m: float):
        return (as_tensor(self.effect.eps(fields, lambda_m))
                - as_tensor(self.effect.eps(self.baseline_fields, lambda_m)))


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
    The impermeability B = eps^-1 shifts LINEARLY with the applied field, dB_I = sum_k r_Ik E_k
    (Voigt I=1..6, k=x,y,z), so eps(E) = (B0 + dB)^-1 with B0 = eps_bg^-1. Crystal principal axes
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
    ISOTROPIC quadratic model: the impermeability shifts as dB = s_kerr * |E|^2 * I, so
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


# ---- QCSE / MQW electro-absorption (Phase 3) ---------------------------------------------

def _photon_energy_J(lambda_m: float) -> float:
    """Photon energy E = h c / lambda = 2 pi hbar c / lambda (J)."""
    return 2.0 * np.pi * HBAR * C_LIGHT / float(lambda_m)


def kramers_kronig_dn(e_grid_J: np.ndarray, dalpha_per_m: np.ndarray) -> np.ndarray:
    """Refractive-index change dn(E) from an absorption-coefficient change dalpha(E) via the
    Kramers-Kronig relation

        dn(E) = (hbar c / pi) P int_0^inf dalpha(E') / (E'^2 - E^2) dE' .

    Evaluated AT each grid point by the Maclaurin (alternate-point) method on a UNIFORM grid: the
    principal value is approximated by summing only grid points of opposite parity to the
    evaluation index (which omits the singular E'=E term), giving an O(h^2) estimate with no
    explicit pole handling. dalpha in 1/m, E in J; returns dn dimensionless on the same grid."""
    E = np.asarray(e_grid_J, dtype=np.float64)
    a = np.asarray(dalpha_per_m, dtype=np.float64)
    if E.ndim != 1 or E.shape != a.shape or E.size < 3:
        raise ValueError("e_grid_J and dalpha_per_m must be 1D arrays of equal length >= 3")
    h = E[1] - E[0]
    if not np.allclose(np.diff(E), h, rtol=1e-6, atol=0.0):
        raise ValueError("e_grid_J must be uniformly spaced (Maclaurin KK assumes it)")
    pref = (HBAR * C_LIGHT / np.pi) * 2.0 * h
    idx = np.arange(E.size)
    dn = np.empty(E.size)
    for i in range(E.size):
        mask = ((idx - i) & 1).astype(bool)             # (j - i) odd -> Maclaurin alternate points
        dn[i] = pref * np.sum(a[mask] / (E[mask] ** 2 - E[i] ** 2))
    return dn


@dataclass
class ElectroAbsorptionModel:
    """QCSE / MQW electro-absorption -- an EffectModel reading fields['E'] (uses the |E_z| field
    across the well). A quantum-well Stark driver (`qw`, any object exposing .solve(F) -> a state
    with .E_transition_J and .overlap) supplies the field-redshifted interband edge E_T(F) and the
    reduced e-h overlap. The model builds an excitonic absorption edge

        alpha(E_photon; F) = alpha0 * (overlap(F)/overlap(0)) * exp(-0.5 ((E_photon - E_T(F))/sigma)^2)

    (a Gaussian exciton line: redshifts with F and weakens with the overlap), forms the field-on
    minus field-off change dalpha on a photon-energy grid, and returns a complex scalar permittivity
    eps = (n + i kappa)^2 with kappa = kappa_bg + dalpha*hbar c/(2 E_photon) and n = n_bg + dn,
    dn the Kramers-Kronig transform of dalpha. At F = 0 (dalpha = 0) eps -> eps_bg exactly.
    Convention exp(-i omega t), Im(eps) > 0 for the field-induced absorption.

    SIMPLIFIED model (a first QCSE electro-absorption modulator): a single excitonic line, no
    band-to-band continuum, and a uniform well field (one Stark solve at the peak |E_z|). eps_bg is
    the zero-field permittivity at the operating wavelength; alpha0_per_m the zero-field peak
    absorption; broadening_J the exciton linewidth (Gaussian sigma); e_grid_J = (lo, hi, N) the
    photon-energy KK grid (J), which must straddle E_T."""
    qw: object                 # QuantumWell-like: .solve(F) -> state(.E_transition_J, .overlap)
    eps_bg: complex            # zero-field permittivity at the operating wavelength
    alpha0_per_m: float        # zero-field peak excitonic absorption [1/m]
    broadening_J: float        # exciton-line Gaussian sigma [J]
    e_grid_J: tuple            # (E_lo_J, E_hi_J, N) photon-energy grid straddling E_T
    field_axis: int = 2        # component of fields['E'] taken as the well field (z by default)

    def _alpha(self, E_eval, E_T, overlap, overlap0):
        g = np.exp(-0.5 * ((np.asarray(E_eval, float) - E_T) / float(self.broadening_J)) ** 2)
        return self.alpha0_per_m * (overlap / overlap0) * g

    def _field_magnitude(self, fields: dict) -> float:
        E = _E_vec(fields)
        return float(np.max(np.abs(np.asarray(E)[..., int(self.field_axis)])))

    def eps(self, fields: dict, lambda_m: float):
        F = self._field_magnitude(fields)
        E_ph = _photon_energy_J(lambda_m)
        s0 = self.qw.solve(0.0)
        sF = self.qw.solve(F)
        ov0 = s0.overlap
        lo, hi, n = self.e_grid_J
        grid = np.linspace(float(lo), float(hi), int(n))
        if not (grid[0] < s0.E_transition_J < grid[-1] and grid[0] < sF.E_transition_J < grid[-1]):
            raise ValueError("e_grid_J must straddle the transition energy E_T(F) for both F=0 and F")
        dalpha_grid = (self._alpha(grid, sF.E_transition_J, sF.overlap, ov0)
                       - self._alpha(grid, s0.E_transition_J, s0.overlap, ov0))
        dn = float(np.interp(E_ph, grid, kramers_kronig_dn(grid, dalpha_grid)))
        dalpha_ph = float(self._alpha(E_ph, sF.E_transition_J, sF.overlap, ov0)
                          - self._alpha(E_ph, s0.E_transition_J, s0.overlap, ov0))
        dkappa = dalpha_ph * HBAR * C_LIGHT / (2.0 * E_ph)
        nb = np.sqrt(complex(self.eps_bg))
        return complex((nb.real + dn) + 1j * (nb.imag + dkappa)) ** 2

    def delta_alpha_per_m(self, fields: dict, lambda_m: float) -> float:
        """Field-induced absorption change dalpha = alpha(F) - alpha(0) [1/m] at the probe photon
        energy -- the electro-absorption-modulator extinction signal (>0 below the F=0 edge)."""
        F = self._field_magnitude(fields)
        E_ph = _photon_energy_J(lambda_m)
        s0 = self.qw.solve(0.0)
        sF = self.qw.solve(F)
        return float(self._alpha(E_ph, sF.E_transition_J, sF.overlap, s0.overlap)
                     - self._alpha(E_ph, s0.E_transition_J, s0.overlap, s0.overlap))
