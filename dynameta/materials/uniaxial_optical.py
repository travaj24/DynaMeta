"""UniaxialOptical: a uniaxial optical model built from two scalar models.

A uniaxial crystal (e.g. wurtzite ZnO with a c-axis-textured film) has an ordinary permittivity for fields
perpendicular to the optic axis and an extraordinary one along it. UniaxialOptical composes two existing
scalar OpticalModels -- typically two DrudeOptical sharing the same mass and damping but with the ordinary
and extraordinary background permittivities -- into the diagonal tensor diag(eps_o, eps_o, eps_e) with the
optic axis along z, the stack axis of the permittivity assembler (a c-axis film grown on the substrate).

It plugs into the assembler's existing tensor path (field_response / eps_fields, the path a DrudeOptical
with boundary scattering already uses), so every optical route that takes a tensor permittivity accepts it
and every route that does not refuses it with its existing message. The scalar eps() refuses, as for a
boundary-scattering Drude: a uniaxial response has no single scalar value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.materials.optical_model import OpticalModel

__all__ = ["UniaxialOptical"]


@dataclass(frozen=True)
class UniaxialOptical(OpticalModel):
    """diag(eps_o, eps_o, eps_e) with the optic axis along z (the film normal / stack axis).

    ordinary, extraordinary: OpticalModels with the eps(lambda_m, n_m3=...) contract (scalar models; a
    model that is itself tensor-valued is refused). Each component is evaluated with its own model at the
    same density, so the carrier physics (mass, damping, background) can differ between the two.
    """
    ordinary: OpticalModel
    extraordinary: OpticalModel

    def __post_init__(self):
        for nm in ("ordinary", "extraordinary"):
            m = getattr(self, nm)
            if not hasattr(m, "eps"):
                raise TypeError("UniaxialOptical.{} must be an OpticalModel, got {!r}".format(nm, type(m).__name__))
            if getattr(m, "field_response", False):
                raise TypeError("UniaxialOptical.{}: a tensor-valued model (field_response) cannot be one "
                                "component of a uniaxial tensor".format(nm))

    @property
    def field_response(self) -> bool:
        return True

    @property
    def needs_position(self) -> bool:
        return False

    def eps(self, lambda_m: float, *, n_m3=None):
        raise ValueError("UniaxialOptical.eps: the response is UNIAXIAL (ordinary and extraordinary components "
                         "differ); the scalar eps() cannot represent it. Use eps_tensor(lambda_m, n_m3=...) or the "
                         "permittivity assembler (eps_fields) and a route that accepts a tensor eps; or use the "
                         "ordinary / extraordinary model directly.")

    def eps_tensor(self, lambda_m: float, *, n_m3=None):
        """(..., 3, 3) diagonal tensor at density n_m3 (shape of n_m3; a 0-d result for n_m3=None)."""
        eo = np.asarray(self.ordinary.eps(lambda_m, n_m3=n_m3), dtype=np.complex128)
        ee = np.asarray(self.extraordinary.eps(lambda_m, n_m3=n_m3), dtype=np.complex128)
        shape = np.broadcast_shapes(eo.shape, ee.shape)
        out = np.zeros(shape + (3, 3), dtype=np.complex128)
        out[..., 0, 0] = eo
        out[..., 1, 1] = eo
        out[..., 2, 2] = ee
        return out

    def eps_fields(self, fields: dict, lambda_m: float):
        """The assembler's entry: fields['n'] (None for a density-independent pair)."""
        return self.eps_tensor(lambda_m, n_m3=fields.get("n"))
