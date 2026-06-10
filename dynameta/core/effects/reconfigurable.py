"""Reconfigurable-material effects: PCM blend and liquid-crystal director tensor.

Split from the former monolithic effects.py; see the package __init__ docstring for
the EffectModel seam contract. Bodies are verbatim. Pure numpy (scipy only lazily for
the Voigt lineshape).
"""
from __future__ import annotations

from dataclasses import dataclass

from dynameta.core.backend import array_namespace, is_jax_array

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
