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

import warnings
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import numpy as np

from dynameta.constants import HBAR, C_LIGHT, Q_E, EPS0, M_E
from dynameta.core.backend import array_namespace, to_backend, is_jax_array, is_numpy_array
from dynameta.core.numerics import trapz


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


def as_tensor(eps):
    """Promote a scalar eps (or scalar grid) to a (..., 3, 3) isotropic tensor eps*I, so a scalar
    and a tensor effect can be summed/composed uniformly. A value already shaped (..., 3, 3) is
    returned unchanged. Backend-agnostic (numpy / cupy / jax via array_namespace)."""
    xp = array_namespace(eps)
    eps = xp.asarray(eps) + 0j                            # promote to complex on any backend
    if eps.ndim >= 2 and tuple(eps.shape[-2:]) == (3, 3):
        return eps
    return eps[..., None, None] * (xp.eye(3) + 0j)


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
        # passivity check (exp(-iwt), Im(eps)>0 = loss): a DeltaEffect that LOWERS Im (bleaching /
        # Franz-Keldysh / QCSE) can push the composed eps into GAIN. Warn at the constitutive seam
        # (the anti-Hermitian part must be >= 0) rather than letting it surface 3 layers down at the
        # FEM energy tripwire. Numpy-only check (skip a traced/cupy array to stay backend-agnostic).
        if is_numpy_array(total):
            herm = 0.5 * (total + np.conjugate(np.swapaxes(total, -1, -2)))   # ((eps + eps^H)/2)
            anti_im = np.linalg.eigvalsh((total - herm) / 1j)                 # eigs of Im-part (Herm)
            if np.min(anti_im) < -1e-6 * (np.max(np.abs(total)) + 1e-30):
                warnings.warn(
                    "ComposedEffect.eps: the composed permittivity has a NEGATIVE imaginary eigenvalue "
                    "(min {:.2e}) -- with exp(-iwt) that is GAIN, not loss. A DeltaEffect is lowering "
                    "Im(eps) below the background absorption; check the bleaching/FK/QCSE delta or the "
                    "background Im.".format(float(np.min(anti_im))), RuntimeWarning, stacklevel=2)
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


def _voigt6_to_full(b6):
    """(...,6) Voigt vector -> (...,3,3) symmetric tensor. Built by stacking (no in-place
    assignment) so it stays inside a JAX trace; backend-agnostic."""
    xp = array_namespace(b6)
    b6 = xp.asarray(b6)
    rows = [xp.stack([b6[..., _VOIGT[i][j]] for j in range(3)], axis=-1) for i in range(3)]
    return xp.stack(rows, axis=-2)


def _E_vec(fields: dict):
    """The applied field from the bundle as (...,3). Accepts a 3-vector (uniform) or a (...,3)
    grid. Raises if absent -- a field-effect model needs E. Backend-agnostic (the returned array's
    backend is whatever fields['E'] is on -- numpy / cupy / jax)."""
    if "E" not in fields or fields["E"] is None:
        raise ValueError("field-effect EffectModel requires fields['E'] (V/m); none supplied "
                         "(run the electrostatic driver first)")
    xp = array_namespace(fields["E"])
    E = xp.asarray(fields["E"])
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
        xp = array_namespace(fields["T"])
        T = xp.asarray(fields["T"])
        n = xp.sqrt(xp.asarray(self.eps_ref) + 0j) + float(self.dn_dT) * (T - float(self.T_ref))
        return n ** 2


@dataclass
class AnisotropicThermoOpticModel:
    """Anisotropic thermo-optic (dn/dT) TENSOR response -- the principal-axis (diagonal) companion to
    ThermoOpticModel for a birefringent heater whose principal indices have DIFFERENT dn/dT (a uniaxial
    crystal: dn_o/dT != dn_e/dT). Reads fields['T'] (kelvin) and returns the DIAGONAL permittivity
    tensor diag( (n_i + dn_dT_i (T - T_ref))^2 ), i = x,y,z, with n_i = sqrt(eps_ref_i). Reduces
    EXACTLY to the scalar ThermoOpticModel * I when the three axes are equal. The tensor is DIAGONAL
    (principal frame); a tilted principal frame (off-diagonal) is ALSO supported by the FEM now -- the
    solver's explicit UPML path solves off-diagonal tensors end-to-end (validated by
    validation/lc_tilted_fem.py)."""
    eps_ref_diag: tuple        # (eps_xx, eps_yy, eps_zz) at T_ref
    dn_dT_diag: tuple          # (dn/dT_x, dn/dT_y, dn/dT_z) [1/K]
    T_ref: float = 300.0

    def eps(self, fields: dict, lambda_m: float):
        if "T" not in fields or fields["T"] is None:
            raise ValueError("AnisotropicThermoOpticModel requires fields['T'] (kelvin); none "
                             "supplied (run the thermal driver first)")
        if len(self.eps_ref_diag) != 3 or len(self.dn_dT_diag) != 3:
            raise ValueError("eps_ref_diag and dn_dT_diag must each have 3 entries (x, y, z)")
        xp = array_namespace(fields["T"])
        dT = xp.asarray(fields["T"]) - float(self.T_ref)
        d = [(xp.sqrt(xp.asarray(er) + 0j) + float(dndt) * dT) ** 2
             for er, dndt in zip(self.eps_ref_diag, self.dn_dT_diag)]
        zero = xp.zeros_like(d[0]) + 0j
        # build with TRAILING (...,3,3) axes (stack rows on -1, rows on -2) so a GRIDDED T of shape
        # (...,) yields (...,3,3) -- the documented convention as_tensor/bridge expect. The old
        # leading-axis xp.stack([xp.stack([...]),...]) produced (3,3,...) and corrupted a gridded T.
        rows = [xp.stack([d[0], zero, zero], axis=-1),
                xp.stack([zero, d[1], zero], axis=-1),
                xp.stack([zero, zero, d[2]], axis=-1)]
        return xp.stack(rows, axis=-2)                       # (...,3,3) diagonal principal-axis tensor


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
    # Maclaurin (alternate-point) rule, VECTORIZED by parity (~8x faster than the original O(N^2)
    # Python loop): dn[i] = pref * sum_{(j-i) ODD} a[j]/(E[j]^2 - E[i]^2). An EVEN index i sums over
    # ODD j (and vice-versa), so splitting into the two parity blocks halves the work and memory vs
    # a full NxN matrix and never touches the singular i=j term (i=j is even parity, excluded).
    # Bit-identical to the loop to ~1e-17.
    E2 = E * E
    even = np.arange(E.size) % 2 == 0
    odd = ~even
    dn = np.empty(E.size)
    dn[even] = pref * (a[odd][None, :] / (E2[odd][None, :] - E2[even][:, None])).sum(axis=1)
    dn[odd] = pref * (a[even][None, :] / (E2[even][None, :] - E2[odd][:, None])).sum(axis=1)
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
    eps = (n + i kappa)^2 with kappa = max(kappa_bg + dalpha*hbar c/(2 E_photon), 0) and n = n_bg +
    dn, dn the Kramers-Kronig transform of dalpha. At F = 0 (dalpha = 0) eps -> eps_bg exactly.
    Convention exp(-i omega t): a passive medium has Im(eps) >= 0, so the total kappa is FLOORED at
    0 (and a warning fires) -- see the eps_bg note for when that floor would otherwise engage.

    SIMPLIFIED model (a first QCSE electro-absorption modulator): a single excitonic line, no
    band-to-band continuum, and a UNIFORM well field -- ONE Stark solve at the PEAK |E_z| over the
    field bundle. It therefore returns a SCALAR eps even if fields['E'] is a grid (it does NOT
    broadcast to a per-point grid -- do not compose it where a pointwise eps grid is required).
    Only the growth-axis (field_axis, z by default) component drives the QCSE; a purely in-plane
    field gives no modulation (a warning fires if |E| > 0 but the selected component is ~0).

    eps_bg is the zero-field permittivity at the operating wavelength. IMPORTANT (bleaching regime):
    for a probe near/above the F=0 exciton -- where the field moves the line AWAY and dalpha < 0 --
    eps_bg's IMAGINARY part must embed the full zero-field excitonic absorption at the probe, so the
    differential dalpha stays a physical reduction of an absorption that is actually present; the
    kappa >= 0 floor enforces passivity regardless. alpha0_per_m is the zero-field peak excitonic
    absorption; broadening_J the exciton linewidth (Gaussian sigma); e_grid_J = (lo, hi, N) the
    photon-energy KK grid (J), which MUST span several broadening_J beyond E_T(0) AND E_T(F) on both
    sides (the KK integral truncates otherwise) and MUST contain the probe photon energy."""
    qw: object                 # QuantumWell-like: .solve(F) -> state(.E_transition_J, .overlap)
    eps_bg: complex            # zero-field permittivity at the operating wavelength
    alpha0_per_m: float        # zero-field peak excitonic absorption [1/m]
    broadening_J: float        # exciton-line Gaussian sigma [J]
    e_grid_J: tuple            # (E_lo_J, E_hi_J, N) photon-energy grid spanning >> sigma around E_T
    field_axis: int = 2        # component of fields['E'] taken as the well field (z by default)
    continuum_alpha0_per_m: float = 0.0   # Elliott band-to-band continuum step strength (0 -> off)
    continuum_binding_J: float = 0.0      # 2D exciton binding = continuum onset above E_T (0 -> off)

    _KK_MARGIN_SIGMA = 6.0     # required grid coverage beyond E_T on each side, in broadening_J

    def _alpha(self, E_eval, E_T, overlap, overlap0):
        E = np.asarray(E_eval, dtype=np.float64)
        ratio = overlap / overlap0
        g = np.exp(-0.5 * ((E - E_T) / float(self.broadening_J)) ** 2)
        a = self.alpha0_per_m * ratio * g                              # 1s excitonic Gaussian line
        if self.continuum_alpha0_per_m > 0.0 and self.continuum_binding_J > 0.0:
            # Elliott band-to-band continuum above the UNBOUND edge E_cont = E_T + E_binding, with the
            # 2D Sommerfeld enhancement S_2D(dE) = 2/(1+exp(-2 pi sqrt(E_b/dE))) -> 2 at the edge and
            # -> 1 far above (a step joint-DOS, edge-enhanced); also scales with the e-h overlap.
            xb = float(self.continuum_binding_J)
            dE = E - (E_T + xb)
            safe = np.where(dE > 0.0, dE, 1.0)                         # avoid sqrt of <=0
            s2d = np.where(dE > 0.0, 2.0 / (1.0 + np.exp(-2.0 * np.pi * np.sqrt(xb / safe))), 0.0)
            a = a + self.continuum_alpha0_per_m * ratio * s2d
        return a

    def _field_magnitude(self, fields: dict) -> float:
        # numpy-ONLY: unlike the other effect models this one wraps scipy eigensolvers (the QCSE
        # Schrodinger solve) + np.interp (the Kramers-Kronig transform), so it is not JAX-traceable.
        # Give a clear error instead of an opaque concretization failure if a JAX array is passed.
        if is_jax_array((fields or {}).get("E")):
            raise TypeError("ElectroAbsorptionModel is numpy-only (it wraps scipy eigensolvers + "
                            "Kramers-Kronig np.interp) and cannot be JAX-traced; pass a numpy E field "
                            "(use jax.pure_callback / a numpy boundary if differentiating around it).")
        E = np.asarray(_E_vec(fields))
        f_axis = float(np.max(np.abs(E[..., int(self.field_axis)])))
        f_tot = float(np.max(np.abs(E)))
        if f_tot > 0.0 and f_axis < 1e-6 * f_tot:
            warnings.warn(
                "ElectroAbsorptionModel: fields['E'] has |E[axis]| ~ 0 but |E| > 0 -- the QCSE "
                "field is the growth-axis (field_axis={}) component only; a transverse field gives "
                "NO modulation. Check field_axis / the field orientation.".format(self.field_axis),
                RuntimeWarning, stacklevel=3)
        return f_axis

    def eps(self, fields: dict, lambda_m: float):
        F = self._field_magnitude(fields)
        E_ph = _photon_energy_J(lambda_m)
        s0 = self.qw.solve(0.0)
        sF = self.qw.solve(F)
        ov0 = s0.overlap
        lo, hi, n = self.e_grid_J
        grid = np.linspace(float(lo), float(hi), int(n))
        # the KK integral needs the grid to COVER the line several sigma beyond E_T on both sides
        # (a center-only straddle silently truncates dn by tens of percent -- audit QC-2):
        margin = self._KK_MARGIN_SIGMA * float(self.broadening_J)
        e_lo = min(s0.E_transition_J, sF.E_transition_J) - margin
        e_hi = max(s0.E_transition_J, sF.E_transition_J) + margin
        if self.continuum_alpha0_per_m:
            # the band-to-band Elliott continuum onset is at E_T + continuum_binding_J (with a slow
            # s2d -> 1 tail above it), which can sit FAR above E_T + margin; require the grid to reach
            # it + a margin so the KK integral does not silently truncate the continuum (audit QC-2b).
            e_hi = max(e_hi, max(s0.E_transition_J, sF.E_transition_J)
                       + float(self.continuum_binding_J) + margin)
        if not (grid[0] <= e_lo and e_hi <= grid[-1]):
            raise ValueError("e_grid_J must span at least {:.0f}*broadening_J below E_T(0)/E_T(F) and "
                             "(when the continuum is on) up to E_T + continuum_binding_J + {:.0f}*"
                             "broadening_J above (the KK integral truncates otherwise)".format(
                                 self._KK_MARGIN_SIGMA, self._KK_MARGIN_SIGMA))
        # E_photon must be IN the grid: np.interp clamps to the edge outside it, diverging from the
        # analytic dkappa path (audit QC-3).
        if not (grid[0] <= E_ph <= grid[-1]):
            raise ValueError("the probe photon energy h c/lambda must lie within e_grid_J")
        dalpha_grid = (self._alpha(grid, sF.E_transition_J, sF.overlap, ov0)
                       - self._alpha(grid, s0.E_transition_J, s0.overlap, ov0))
        dn = float(np.interp(E_ph, grid, kramers_kronig_dn(grid, dalpha_grid)))
        dalpha_ph = float(self._alpha(E_ph, sF.E_transition_J, sF.overlap, ov0)
                          - self._alpha(E_ph, s0.E_transition_J, s0.overlap, ov0))
        dkappa = dalpha_ph * HBAR * C_LIGHT / (2.0 * E_ph)
        nb = np.sqrt(complex(self.eps_bg))
        kappa = nb.imag + dkappa
        if kappa < 0.0:                                   # passivity: no gain (audit QC-1)
            warnings.warn(
                "ElectroAbsorptionModel: kappa_bg + dkappa < 0 -- the differential model implies "
                "GAIN in the bleaching regime; clamping Im to 0. Supply an eps_bg whose Im embeds "
                "the zero-field exciton absorption at the probe.", RuntimeWarning, stacklevel=2)
            kappa = 0.0
        return complex((nb.real + dn) + 1j * kappa) ** 2

    def delta_alpha_per_m(self, fields: dict, lambda_m: float) -> float:
        """Field-induced absorption change dalpha = alpha(F) - alpha(0) [1/m] at the probe photon
        energy -- the electro-absorption-modulator extinction signal (>0 below the F=0 edge)."""
        F = self._field_magnitude(fields)
        E_ph = _photon_energy_J(lambda_m)
        s0 = self.qw.solve(0.0)
        sF = self.qw.solve(F)
        return float(self._alpha(E_ph, sF.E_transition_J, sF.overlap, s0.overlap)
                     - self._alpha(E_ph, s0.E_transition_J, s0.overlap, s0.overlap))


# ---- Burstein-Moss band-filling + bandgap renormalization (R8) -----------------------------

@dataclass
class BursteinMossEdge:
    """Carrier-density-dependent interband absorption edge of a degenerate semiconductor (e.g. ITO):
    band filling pushes the optical gap UP (Burstein-Moss blueshift) while many-body bandgap
    renormalization pulls it down (a redshift). Reads fields['n'] (carrier density m^-3) and returns
    the interband permittivity contribution as a scalar grid (promoted to isotropic by as_tensor):

        Eg_opt(n) = Eg0 - dE_BGR(n) + dE_BM(n),
          dE_BM(n)  = (hbar^2/2)(1/m_vc) (3 pi^2 n)^(2/3)    (band-filling blueshift)
          dE_BGR(n) = bgr_coeff_J_m * n^(1/3)                (renormalization redshift; 0 -> off)
        Im edge (Tauc/parabolic, exp(-i omega t) -> Im >= 0): dimensionless eps2(E; Eg_opt) = alpha_edge
          * ((E - Eg_opt)/Eg_opt)^tauc_exponent * (Eg_opt/E)^2 above Eg_opt, and its Kramers-Kronig
          partner dn(E) (reusing kramers_kronig_dn on alpha = E eps2/(hbar c)). eps = (sqrt(eps_inf) +
          dn + i kappa)^2, kappa = eps2/2 >= 0.

    This is a PURE interband DELTA meant to be composed THROUGH DeltaEffect on top of the bare Drude
    (whose eps_inf already embeds the interband response AT the reference doping). Compose as
    ComposedEffect(background=OpticalModelEffect(DrudeOptical(...)),
                   deltas=[DeltaEffect(BursteinMossEdge(eps_inf=<same eps_inf>, ...), {"n": n_ref})]);
    only the doping-INDUCED change relative to n_ref survives (no eps_inf double-count). Pick n_ref =
    n_bg (the fitted Drude eps_inf stays valid there). enabled=False -> returns eps_inf everywhere
    (delta = 0 through DeltaEffect = byte-identical off-switch). m_vc is the REDUCED joint
    conduction-valence mass (1/m_vc = 1/m_c + 1/m_v), NOT the Drude optical mass. numpy-only (KK uses
    np.interp); exp(-i omega t), Im(eps) >= 0; grid-capable (dn precomputed vs Eg_opt and interpolated).
    """
    eps_inf: float
    Eg0_J: float                  # undoped optical gap [J] (e.g. 3.6 * Q_E for ITO)
    m_vc_kg: float                # reduced joint conduction-valence mass [kg]
    alpha_edge: float             # dimensionless interband edge amplitude (O(1); Im(eps) ~ alpha_edge)
    bgr_coeff_J_m: float = 0.0    # bandgap-renormalization coefficient C in dE_BGR = C n^(1/3) [J*m]; 0 -> off
    tauc_exponent: float = 0.5    # 0.5 = direct-allowed sqrt(E-Eg) edge
    e_grid_J: tuple = None        # (E_lo, E_hi, N) KK grid override; None -> auto around Eg_opt + probe
    enabled: bool = True          # master off-switch: False -> eps_inf everywhere (delta 0)
    _N_EG = 64                    # Eg_opt samples for the grid-capable dn interpolation
    _KK_SPAN_J = 5.0 * 1.602176634e-19   # how far above the highest edge the KK grid extends (~5 eV)
    _KK_N = 3001                  # KK photon-energy grid points

    def gap_shift_J(self, n_m3):
        """Burstein-Moss blueshift dE_BM(n) [J] = (hbar^2/2)(1/m_vc)(3 pi^2 n)^(2/3)."""
        n = np.asarray(n_m3, dtype=np.float64)
        return (HBAR ** 2 / 2.0) * (1.0 / float(self.m_vc_kg)) * (3.0 * np.pi ** 2 * n) ** (2.0 / 3.0)

    def optical_gap_J(self, n_m3):
        """Doping-shifted optical gap Eg_opt(n) = Eg0 - dE_BGR + dE_BM [J]."""
        n = np.asarray(n_m3, dtype=np.float64)
        dE_BGR = float(self.bgr_coeff_J_m) * n ** (1.0 / 3.0)
        return float(self.Eg0_J) - dE_BGR + self.gap_shift_J(n)

    def _eps2(self, E_eval, Eg_opt):
        """Dimensionless interband Im(eps) edge: alpha_edge * ((E-Eg)/Eg)^p * (Eg/E)^2 above Eg, else 0.
        Non-dimensionalized by Eg so alpha_edge is an O(1) amplitude (not a unit-laden prefactor)."""
        E = np.asarray(E_eval, dtype=np.float64)
        x = np.maximum(E - Eg_opt, 0.0) / Eg_opt
        return float(self.alpha_edge) * x ** float(self.tauc_exponent) * (Eg_opt / E) ** 2

    def eps(self, fields: dict, lambda_m: float):
        n_in = (fields or {}).get("n")
        if n_in is None:
            raise ValueError("BursteinMossEdge requires fields['n'] (carrier density m^-3); none "
                             "supplied (run the carrier model first)")
        if is_jax_array(n_in):
            raise TypeError("BursteinMossEdge is numpy-only (Kramers-Kronig np.interp); pass a numpy "
                            "density (omit this delta from a JAX-traced pipeline -- it is additive).")
        n = np.asarray(n_in, dtype=np.float64)
        if not self.enabled:
            return np.full(n.shape, complex(self.eps_inf))         # off-switch: pure eps_inf -> delta 0

        E_ph = _photon_energy_J(lambda_m)
        Eg = self.optical_gap_J(n)                                 # (...,) optical gap per cell
        Eg_lo, Eg_hi = float(np.min(Eg)), float(np.max(Eg))
        # KK photon-energy grid: span below the lowest edge / probe, up to well above the highest edge
        if self.e_grid_J is not None:
            lo, hi, ng = self.e_grid_J
            grid = np.linspace(float(lo), float(hi), int(ng))
        else:
            e_lo = min(Eg_lo, E_ph) - 0.5 * float(self._KK_SPAN_J)
            e_hi = max(Eg_hi, E_ph) + float(self._KK_SPAN_J)
            grid = np.linspace(max(e_lo, 1e-21), e_hi, int(self._KK_N))
        if not (grid[0] <= min(Eg_lo, E_ph) and max(Eg_hi, E_ph) <= grid[-1]):   # no silent KK truncation
            raise ValueError("e_grid_J must span the optical gap range and the probe energy")

        def _dn_at_gap(eg):
            # absorption alpha = E eps2/(hbar c) [1/m]; KK -> dn (refractive index shift) at the probe
            dalpha_grid = grid * self._eps2(grid, eg) / (HBAR * C_LIGHT)         # 1/m
            return float(np.interp(E_ph, grid, kramers_kronig_dn(grid, dalpha_grid)))

        # grid-capable dn: precompute dn(Eg_opt) on a 1D Eg grid, interpolate onto the per-cell gaps
        if Eg_hi - Eg_lo < 1e-30:
            dn = np.full(Eg.shape, _dn_at_gap(Eg_lo))
        else:
            egs = np.linspace(Eg_lo, Eg_hi, int(self._N_EG))
            dn_tab = np.array([_dn_at_gap(e) for e in egs])
            dn = np.interp(Eg.ravel(), egs, dn_tab).reshape(Eg.shape)

        eps2_ph = self._eps2(E_ph, Eg)                                          # dimensionless Im edge (>= 0)
        kappa = 0.5 * eps2_ph                                                   # extinction = eps2/(2 n_re)~eps2/2
        n_re = np.sqrt(complex(self.eps_inf)).real + dn
        return (n_re + 1j * kappa) ** 2                                          # scalar grid (...,)


# ---- reconfigurable: phase-change + liquid-crystal (Phase 4) -------------------------------

@dataclass
class PCMModel:
    """Phase-change-material EffectModel (GST / Sb2S3; also VO2 as a two-endpoint insulator/metal
    blend): a state fraction f in [0, 1] blends the two endpoint permittivities via the Bruggeman
    effective-medium approximation (the standard intermediate-state optical model). NOTE: f here is a
    generic two-endpoint mixing fraction -- for GST/Sb2S3 it is the JMAK crystalline fraction from
    carriers.switching.PCMSwitching, but VO2's insulator->metal transition is NOT that JMAK
    crystallization kinetics (see PCMSwitching); only the optical blend is shared. Reads
    fields['crystalline_fraction'] (scalar in [0, 1]; default 0 = fully amorphous) and returns the
    self-consistent Bruggeman root of

        f (eps_c - eps)/(eps_c + 2 eps) + (1 - f)(eps_a - eps)/(eps_a + 2 eps) = 0
        => 2 eps^2 - b eps - eps_a eps_c = 0,  b = eps_c (3f - 1) + eps_a (2 - 3f),

    taking the passive branch (Im(eps) >= 0 for exp(-i omega t)). At f = 0 eps -> eps_a and at
    f = 1 eps -> eps_c EXACTLY. Scalar (isotropic) response. eps_amorphous/eps_crystalline are the
    two end-state permittivities at the operating wavelength."""
    eps_amorphous: complex
    eps_crystalline: complex

    def eps(self, fields: dict, lambda_m: float):
        f_in = fields.get("crystalline_fraction", 0.0) if fields else 0.0
        xp = array_namespace(f_in)                       # numpy by default; jax if f is a jax scalar
        if not is_jax_array(f_in) and not (0.0 <= float(f_in) <= 1.0):
            raise ValueError("fields['crystalline_fraction'] must be in [0, 1]")
        f = xp.asarray(f_in)
        ea = xp.asarray(self.eps_amorphous) + 0j
        ec = xp.asarray(self.eps_crystalline) + 0j
        b = ec * (3.0 * f - 1.0) + ea * (2.0 - 3.0 * f)
        s = xp.sqrt(b * b + 8.0 * ea * ec)
        e_plus, e_minus = (b + s) / 4.0, (b - s) / 4.0
        eps = xp.where(e_plus.imag >= e_minus.imag, e_plus, e_minus)  # passive branch (Im >= 0)
        # exact end states (xp.where, not a Python `if`, so it also traces under JAX): for a
        # lossless negative-real endpoint the Im>=Im tie-break would otherwise pick the wrong real
        # root at the boundary (audit PCM-1/PCM-2).
        eps = xp.where(f == 0.0, ea, eps)
        eps = xp.where(f == 1.0, ec, eps)
        return eps


@dataclass
class LiquidCrystalModel:
    """Liquid-crystal uniaxial EffectModel -- the optical companion to the lc_director Freedericksz
    driver. A director tilt angle theta (from the plate plane, rotating in the x-z plane) sets the
    optic axis n-hat = (cos theta, 0, sin theta) and the UNIAXIAL permittivity tensor

        eps = n_o^2 I + (n_e^2 - n_o^2) (n-hat (x) n-hat)            (3x3, anisotropic)

    Reads fields['director_angle_rad'] (scalar, default 0 = planar -> optic axis along x). At
    theta = 0 the extraordinary axis is x (eps_xx = n_e^2, eps_yy = eps_zz = n_o^2); rotating to
    theta = pi/2 puts it along z. Reduces EXACTLY to the isotropic n_o^2 I when n_e = n_o.

    FEM NOTE: both the PRINCIPAL orientations -- planar (theta=0) and homeotropic (theta=pi/2),
    DIAGONAL (validation/lc_uniaxial_fem.py) -- AND an INTERMEDIATE tilt (nonzero off-diagonal eps_xz)
    flow correctly through the tensor-eps FEM. The off-diagonal solve is supported end-to-end via the
    solver's explicit UPML path (the earlier failure was mesh.SetPML's coordinate stretch being wrong
    for an anisotropic medium, not an assembly defect); the tilted ordinary wave is tilt-invariant and
    the extraordinary wave matches n_eff(theta) (validation/lc_tilted_fem.py).

    DIRECTOR SOURCE: 'director_angle_rad' is the tilt measured FROM THE PLATE PLANE (0 = planar/in-plane,
    pi/2 = homeotropic/along z). To compute it from an applied voltage (statics) or its time evolution
    (switching), use carriers.lc_director.director_profile / director_profile_bvp (two-constant K11/K33,
    flexo, Poisson voltage-division, planar/cyl) or carriers.lc_dynamics.LCDynamics (Erickson-Leslie),
    then bridge with carriers.lc_director.director_to_extra_fields (which applies the pi/2 field-axis ->
    plate-plane convention) and drop the result into the optics field bundle."""
    n_o: float
    n_e: float

    def eps(self, fields: dict, lambda_m: float):
        th_in = (fields or {}).get("director_angle_rad", 0.0)
        xp = array_namespace(th_in)
        th = xp.asarray(th_in)
        c, s = xp.cos(th), xp.sin(th)
        # trailing-axis build so a GRIDDED director_angle of shape (...,) yields (...,3,3) (the
        # documented convention); xp.stack([...]) + xp.outer flattened a grid to (3N,3N) and raised.
        nhat = xp.stack([c, xp.zeros_like(c), s], axis=-1)            # (...,3) optic axis
        outer = nhat[..., :, None] * nhat[..., None, :]              # (...,3,3) = nhat (x) nhat
        eps = (self.n_o ** 2) * (xp.eye(3) + 0j) + (self.n_e ** 2 - self.n_o ** 2) * outer
        return eps + 0j                                   # complex (Im=0 here)


@dataclass
class MagnetoOpticModel:
    """Magneto-optic (gyrotropic) EffectModel -- a magnetized medium in the polar Faraday geometry
    (magnetization along z, propagation along z). The permittivity is the gyrotropic tensor

        eps = [[eps_r, i g, 0], [-i g, eps_r, 0], [0, 0, eps_r]]        (3x3, Hermitian for real g)

    with `eps_r` the base (isotropic) permittivity and `g` the gyration -- the off-diagonal
    magneto-optic coupling (~ Verdet constant x B). The two normal modes for z-propagation are
    circular polarizations with indices n_pm = sqrt(eps_r +/- g), so a linearly polarized wave through
    thickness L has its plane of polarization rotated by the Faraday angle
    theta_F = (pi L / lambda) Re(n_+ - n_-). Reads an optional fields['magnetization'] in [-1, 1] that
    SCALES g (sign = magnetization direction; default +1), so the same model serves a static film or a
    field-driven magnetization. Reduces EXACTLY to the isotropic eps_r * I when g (or magnetization)
    is 0. Backend-agnostic in the magnetization (numpy / cupy / jax).

    For real eps_r and real g the tensor is HERMITIAN -> the medium is LOSSLESS (energy-conserving,
    consistent with exp(-i omega t), Im(eps) > 0 for absorbers). Validated against an analytic
    circular-eigenmode Faraday-rotation reference in validation/magneto_optic_faraday.py.

    FEM NOTE: the gyrotropic tensor has nonzero (imaginary) OFF-DIAGONAL entries. The off-diagonal
    FEM solve is now SUPPORTED end-to-end -- the earlier failure was NOT an NGSolve assembly defect
    but mesh.SetPML's coordinate stretch being wrong for an anisotropic medium, fixed by the explicit
    UPML in solver.solve_fem. The FEM transmitted field is Faraday-rotated (co- AND cross-polarized),
    conserves energy exactly (Hermitian eps -> R_flux + T_flux = 1, A_independent ~ 0), and matches
    the circular-eigenmode Jones-TMM reference to ~2% (validation/magneto_optic_faraday.py). The
    single-projection R/T (result.R/T) measures the CO-polarized channel; result.R_flux/T_flux
    measure the full (co + cross) power."""
    eps_r: float
    g: float

    def eps(self, fields: dict, lambda_m: float):
        m_in = (fields or {}).get("magnetization", 1.0)
        xp = array_namespace(m_in)
        g = self.g * xp.asarray(m_in) + 0j
        e = xp.asarray(self.eps_r) + 0j + xp.zeros_like(g)   # broadcast e to the field shape
        zero = xp.zeros_like(g)
        # TRAILING (...,3,3) axes so a GRIDDED magnetization of shape (...,) yields (...,3,3) (the
        # documented convention); the old leading-axis stack gave (3,3,...) and raised on a grid.
        rows = [xp.stack([e, 1j * g, zero], axis=-1),
                xp.stack([-1j * g, e, zero], axis=-1),
                xp.stack([zero, zero, e], axis=-1)]
        return xp.stack(rows, axis=-2)                       # (...,3,3) gyrotropic, Hermitian for real g


# ---- Intersubband quantum eps_zz from sub-band wavefunctions (R7) -------------------------

@dataclass
class IntersubbandEffect:
    """Diagonal-anisotropic permittivity from a quantum SubbandResult: the growth-axis (z) response
    carries the INTERSUBBAND transitions whose dipole <psi_i|z|psi_j> lies along z, while the in-plane
    (x, y) response is the ordinary intraband free-carrier Drude. Reads fields['subband'] (a
    carriers.schrodinger_poisson.SubbandResult: energies_J, psi, z_m, sheet_density_m2).

        eps_xx = eps_yy = eps_inf - wp^2 / (w^2 + i w gamma_intra)              (intraband Drude)
        eps_zz = eps_xx + sum_{i<j}  S_ij / (eps0 (w_ij^2 - w^2 - i w gamma_inter))

    with wp^2 = n3d q^2/(eps0 m_opt), n3d = (sum_i n_s,i)/Leff, w_ij = (E_j - E_i)/hbar, and the
    f-sum-rule-CONSISTENT oscillator strength built in:

        S_ij = N_ij q^2 |z_ij|^2 (2 w_ij)/hbar ,   N_ij = (n_s,i - n_s,j)/Leff ,   z_ij = <psi_i|z|psi_j>

    This S_ij is exactly (q^2/eps0) f_ij N_ij / eps0 -> ... with the dimensionless TRK strength
    f_ij = (2 m0 w_ij/hbar)|z_ij|^2: the free mass m0 CANCELS, so the optical mass m_opt enters ONLY
    the intraband Drude -- the intersubband line is mass-free (a common bug is to leave m* in it).
    The denominator uses -i w gamma (exp(-i omega t), Im(eps_zz) > 0 on resonance = absorptive).

    REDUCES to a scalar Drude * I when fewer than two sub-bands are occupied (no i<j pair, the
    Lorentzian sum is empty) -> diag(eps_D, eps_D, eps_D) == as_tensor(DrudeOptical.eps). v1 returns a
    UNIFORM (3,3) slab response (sheet smeared over Leff = z[-1]-z[0]); a z-graded eps_zz(z) via the
    local |psi(z)|^2 weighting is a documented follow-on (Leff sets the LINE STRENGTH, not the position
    or the Leff-free f-sum rule). exp(-i omega t), Im(eps) > 0 for absorbers; pure numpy."""
    eps_inf: float
    m_opt_kg: float            # sub-band-averaged Kane optical mass (intraband Drude only)
    gamma_intra_rad_s: float
    gamma_inter_rad_s: float   # intersubband dephasing (scalar; per-pair callable = follow-on)
    occ_floor_m2: float = 0.0  # min sheet density to count a sub-band as occupied

    def eps(self, fields: dict, lambda_m: float):
        res = (fields or {}).get("subband")
        if res is None:
            raise ValueError("IntersubbandEffect requires fields['subband'] (a SubbandResult); none "
                             "supplied (run the Schrodinger-Poisson solver first)")
        z = np.asarray(res.z_m, dtype=np.float64)
        psi = np.asarray(res.psi, dtype=np.float64)
        E = np.asarray(res.energies_J, dtype=np.float64)
        ns = np.asarray(res.sheet_density_m2, dtype=np.float64)        # m^-2 per sub-band
        if psi.ndim != 2 or psi.shape[1] != E.size or ns.size != E.size:
            raise ValueError("SubbandResult psi/energies_J/sheet_density_m2 are inconsistent in shape")
        Leff = float(z[-1] - z[0])
        if not (Leff > 0.0):
            raise ValueError("SubbandResult z_m must span a positive width (Leff = z[-1]-z[0])")

        omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
        n3d = float(np.sum(ns)) / Leff                                 # m^-3 total volume density
        wp2 = n3d * Q_E * Q_E / (EPS0 * float(self.m_opt_kg))
        eps_intra = complex(self.eps_inf) - wp2 / (omega * omega + 1j * omega * float(self.gamma_intra_rad_s))

        eps_zz = eps_intra
        gam = float(self.gamma_inter_rad_s)
        idx = np.where(ns > float(self.occ_floor_m2))[0]               # occupied sub-bands
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = int(idx[a]), int(idx[b])                        # E[i] < E[j] (sorted ascending)
                w_ij = (E[j] - E[i]) / HBAR                            # > 0
                z_ij = trapz(psi[:, i] * z * psi[:, j], z)             # <psi_i|z|psi_j> [m]
                N_ij = (ns[i] - ns[j]) / Leff                         # >= 0 (lower band more occupied)
                if N_ij < 0.0:                                        # population inversion -> gain
                    warnings.warn("IntersubbandEffect: inverted population (n_s[{}] < n_s[{}]) gives a "
                                  "gain line; clamping to 0".format(i, j))
                    N_ij = 0.0
                S_ij = N_ij * Q_E * Q_E * (z_ij * z_ij) * (2.0 * w_ij) / HBAR
                eps_zz += S_ij / (EPS0 * (w_ij * w_ij - omega * omega - 1j * omega * gam))

        out = np.zeros((3, 3), dtype=complex)
        out[0, 0] = eps_intra
        out[1, 1] = eps_intra
        out[2, 2] = eps_zz
        return out
