"""Thermo-optic effects: scalar and anisotropic dn/dT.

Split from the former monolithic effects.py; see the package __init__ docstring for
the EffectModel seam contract. Bodies are verbatim. Pure numpy (scipy only lazily for
the Voigt lineshape).
"""
from __future__ import annotations

from dataclasses import dataclass

from dynameta.core.backend import array_namespace

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
