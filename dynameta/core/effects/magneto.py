"""Magneto-optic effects: fixed-axis and vector gyrotropic tensors.

Split from the former monolithic effects.py; see the package __init__ docstring for
the EffectModel seam contract. Bodies are verbatim. Pure numpy (scipy only lazily for
the Voigt lineshape).
"""
from __future__ import annotations

from dataclasses import dataclass

from dynameta.core.backend import array_namespace

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


@dataclass
class VectorMagnetoOpticModel:
    """Full VECTOR gyrotropic permittivity (roadmap R13): the magnetization may point ANYWHERE, not
    just along z. With the gyration vector g = g_s * m (m the unit magnetization direction) the
    tensor is, in the SAME convention the validated z-axis MagnetoOpticModel ships
    (eps_xy = +i g_z, Faraday-TMM-validated):

        eps_ij = eps_r * delta_ij + i * epsilon_ijk * g_k
        =>  [[ eps_r,  +i gz,  -i gy],
             [ -i gz,  eps_r,  +i gx],
             [ +i gy,  -i gx,  eps_r]]

    Hermitian (lossless) for real g_s and real m. Reads fields['m_vector'] -- a unit 3-vector, or a
    GRIDDED (...,3) per-point direction field (e.g. the R11 LLG trajectory m(t) or a domain pattern);
    built with trailing (...,3,3) axes so a gridded m yields (...,3,3) per the documented bridge
    convention. m along z REDUCES EXACTLY to MagnetoOpticModel(eps_r, g_s) at magnetization=1 (the
    byte-identical anchor); rotation-equivariant: eps(R m) = R eps(m) R^T."""
    eps_r: float
    g_s: float

    def eps(self, fields: dict, lambda_m: float):
        m_in = (fields or {}).get("m_vector")
        if m_in is None:
            raise ValueError("VectorMagnetoOpticModel requires fields['m_vector'] (a unit 3-vector or "
                             "a (...,3) direction grid); none supplied (run the LLG/domain driver "
                             "first, or use MagnetoOpticModel for fixed z magnetization)")
        xp = array_namespace(m_in)
        m = xp.asarray(m_in)
        if m.shape[-1] != 3:
            raise ValueError("VectorMagnetoOpticModel: fields['m_vector'] must have a trailing length-3 "
                             "axis (got shape {})".format(tuple(m.shape)))
        g = self.g_s * m + 0j                                # (...,3) gyration vector
        gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
        e = xp.asarray(self.eps_r) + 0j + xp.zeros_like(gx)  # broadcast eps_r to the field shape
        rows = [xp.stack([e, 1j * gz, -1j * gy], axis=-1),
                xp.stack([-1j * gz, e, 1j * gx], axis=-1),
                xp.stack([1j * gy, -1j * gx, e], axis=-1)]
        return xp.stack(rows, axis=-2)                       # (...,3,3); Hermitian for real g


# ---- Intersubband quantum eps_zz from sub-band wavefunctions (R7) -------------------------
