"""EffectModel protocol, composition (ComposedEffect/DeltaEffect), tensor + field helpers, KK transform.

Split from the former monolithic effects.py; see the package __init__ docstring for
the EffectModel seam contract. Bodies are verbatim. Pure numpy (scipy only lazily for
the Voigt lineshape).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import numpy as np

from dynameta.constants import C_LIGHT, HBAR
from dynameta.core.backend import array_namespace, is_numpy_array

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


# The Maclaurin KK transform above is, for a FIXED grid, a constant LINEAR map dn = K @ dalpha
# (K[i, j] = pref/(E[j]^2 - E[i]^2) on opposite-parity i-j, 0 on same-parity). A caller that
# transforms MANY dalpha profiles on ONE grid (BursteinMossEdge: _N_EG = 64 rows per bias) can
# therefore build the two non-zero parity blocks ONCE and push all rows through two BLAS matmuls
# instead of re-running the O(N^2) divide-and-sum per row (audit 6.2 perf). Keyed on the grid
# VALUES (bytes), not object identity -- callers rebuild equal-valued grids per call via linspace.
# Blocks are ~N^2/2 doubles (~36 MB at N=3001), so the cache is kept tiny (LRU-of-2).
_KK_KERNEL_CACHE: dict = {}
_KK_KERNEL_CACHE_MAX = 2


def _kk_parity_kernel(E: np.ndarray):
    """Cached parity blocks of the Maclaurin KK kernel for the (validated-uniform) grid E:
    (even_mask, K_eo, K_oe) with dn[even] = K_eo @ a[odd] and dn[odd] = K_oe @ a[even]."""
    key = E.tobytes()
    hit = _KK_KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    h = E[1] - E[0]
    pref = (HBAR * C_LIGHT / np.pi) * 2.0 * h
    E2 = E * E
    even = np.arange(E.size) % 2 == 0
    odd = ~even
    K_eo = pref / (E2[odd][None, :] - E2[even][:, None])
    K_oe = pref / (E2[even][None, :] - E2[odd][:, None])
    while len(_KK_KERNEL_CACHE) >= _KK_KERNEL_CACHE_MAX:
        _KK_KERNEL_CACHE.pop(next(iter(_KK_KERNEL_CACHE)))       # evict oldest (insertion order)
    _KK_KERNEL_CACHE[key] = (even, K_eo, K_oe)
    return even, K_eo, K_oe


def kramers_kronig_dn_rows(e_grid_J: np.ndarray, dalpha_rows: np.ndarray,
                           e_eval_J: float = None) -> np.ndarray:
    """Batched kramers_kronig_dn: dn for MANY dalpha rows (shape (M, N)) on ONE fixed uniform grid.
    Row-for-row it equals kramers_kronig_dn up to summation reassociation only (BLAS dot vs
    pairwise .sum; ~1e-15 relative) -- the per-element terms pref*a[j]/(E[j]^2-E[i]^2) are the
    same. Same grid validation as kramers_kronig_dn.

    e_eval_J = None: returns the full (M, N) dn rows via the cached parity kernel (two matmuls).
    e_eval_J given (a photon energy INSIDE the grid): returns the (M,) dn AT that energy only --
    np.interp(e_eval, E, dn_row) is local-linear, so only the TWO kernel rows bracketing e_eval
    contribute; those are built directly in O(N) (no N^2 kernel, nothing worth caching) and applied
    as two matvecs. This is the path a per-probe caller (BursteinMossEdge) wants: it does O(M*N)
    work instead of the O(M*N^2) full transform."""
    E = np.asarray(e_grid_J, dtype=np.float64)
    A = np.asarray(dalpha_rows, dtype=np.float64)
    if E.ndim != 1 or E.size < 3 or A.ndim != 2 or A.shape[1] != E.size:
        raise ValueError("e_grid_J must be a 1D array (>= 3 points) and dalpha_rows shaped "
                         "(n_rows, len(e_grid_J))")
    h = E[1] - E[0]
    if not np.allclose(np.diff(E), h, rtol=1e-6, atol=0.0):
        raise ValueError("e_grid_J must be uniformly spaced (Maclaurin KK assumes it)")
    if e_eval_J is None:
        even, K_eo, K_oe = _kk_parity_kernel(E)
        odd = ~even
        dn = np.empty_like(A)
        dn[:, even] = A[:, odd] @ K_eo.T
        dn[:, odd] = A[:, even] @ K_oe.T
        return dn
    e_eval = float(e_eval_J)
    if not (E[0] <= e_eval <= E[-1]):
        raise ValueError("e_eval_J must lie within e_grid_J (np.interp would silently clamp)")
    pref = (HBAR * C_LIGHT / np.pi) * 2.0 * h
    E2 = E * E
    i0 = int(np.clip(np.searchsorted(E, e_eval, side="right") - 1, 0, E.size - 2))
    parity = np.arange(E.size) % 2

    def _dn_at(i):                                # dn[i] rows: one O(N) kernel row, one matvec
        opp = parity != (i % 2)                   # opposite-parity columns (excludes j == i)
        return A[:, opp] @ (pref / (E2[opp] - E2[i]))

    d0, d1 = _dn_at(i0), _dn_at(i0 + 1)
    w = (e_eval - E[i0]) / (E[i0 + 1] - E[i0])    # np.interp's local-linear weight in [0, 1]
    return d0 + (d1 - d0) * w
