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

from dynameta.constants import HBAR, C_LIGHT
from dynameta.core.backend import array_namespace, to_backend, is_jax_array


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

    _KK_MARGIN_SIGMA = 6.0     # required grid coverage beyond E_T on each side, in broadening_J

    def _alpha(self, E_eval, E_T, overlap, overlap0):
        g = np.exp(-0.5 * ((np.asarray(E_eval, float) - E_T) / float(self.broadening_J)) ** 2)
        return self.alpha0_per_m * (overlap / overlap0) * g

    def _field_magnitude(self, fields: dict) -> float:
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
        if not (grid[0] <= e_lo and e_hi <= grid[-1]):
            raise ValueError("e_grid_J must span at least {:.0f}*broadening_J beyond E_T(0) and "
                             "E_T(F) on BOTH sides (the KK integral truncates otherwise)".format(
                                 self._KK_MARGIN_SIGMA))
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


# ---- reconfigurable: phase-change + liquid-crystal (Phase 4) -------------------------------

@dataclass
class PCMModel:
    """Phase-change-material EffectModel (GST / Sb2S3 / VO2): a crystalline volume fraction f in
    [0, 1] blends the amorphous and crystalline permittivities via the Bruggeman effective-medium
    approximation (the standard intermediate-state optical model). Reads
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

    FEM NOTE: the two PRINCIPAL orientations -- planar (theta=0) and homeotropic (theta=pi/2) -- are
    DIAGONAL and flow correctly through the Phase-0b tensor-eps FEM (validation/lc_uniaxial_fem.py).
    An INTERMEDIATE tilt gives a nonzero off-diagonal eps_xz, which the current FEM matrix-CF matvec
    mis-evaluates under PML (a tracked P0b follow-on; assemble_eps_cf RAISES NotImplementedError for
    off-diagonal tensors rather than return a silently-wrong result). The tilted-director ANGULAR
    physics is validated analytically (validation/reconfigurable_modulators.py); only the FEM solve
    of an off-diagonal tensor is deferred."""
    n_o: float
    n_e: float

    def eps(self, fields: dict, lambda_m: float):
        th_in = (fields or {}).get("director_angle_rad", 0.0)
        xp = array_namespace(th_in)
        th = xp.asarray(th_in)
        c, s = xp.cos(th), xp.sin(th)
        nhat = xp.stack([c, xp.zeros_like(c), s])         # (3,) optic axis (no in-place build)
        eps = (self.n_o ** 2) * xp.eye(3) + (self.n_e ** 2 - self.n_o ** 2) * xp.outer(nhat, nhat)
        return eps + 0j                                   # complex (Im=0 here)
