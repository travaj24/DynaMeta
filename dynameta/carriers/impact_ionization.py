"""Chynoweth impact ionization + substrate-current driver (D4) -- a POST-PROCESSOR on solved
drift-diffusion fields.

    alpha_n(E) = a_n exp(-b_n/E),  alpha_p(E) = a_p exp(-b_p/E)        [1/m], E in V/m

The local pair-generation density is G = (alpha_n |J_n| + alpha_p |J_p|)/q [1/(m^3 s)], so the
impact-generated (substrate) current is I_sub = q Int G dV = Int (alpha_n |J_n| + alpha_p |J_p|)
dV -- the HCI driver (reliability.hci takes Isub externally; this supplies it) and the avalanche
onset diagnostic (M - 1 ~ I_sub/I in the low-multiplication regime).

SCOPE (honest): post-hoc on the CONVERGED DD solution -- the generated carriers do NOT feed back
into the Newton solve, so results are quantitative only for LOW multiplication (I_sub << I, the
HCI-relevant operating regime) and the model cannot run avalanche breakdown. Off-switch is
trivial: nothing here touches the solve.

QUADRATURE (probed on the 2D layered mesh): DEVSIM's tensor mesh triangulates with ZERO-couple
diagonals (the circumcenter of a right triangle sits on its hypotenuse), so the finite-volume
mesh is effectively rectangular -- the x- and y-aligned edges each own exactly half the region
volume via 2*EdgeNodeVolume, and sum(EdgeCouple*EdgeLength) = 2 * Volume. The directional edge
quadrature

    I_sub = sum_edges alpha(|E_e|) |J_e| * EdgeCouple_e * EdgeLength_e

is then EXACT for transport-aligned fields (verified to closed form on a constant-field bar:
the aligned edges carry weight = the full volume, the perpendicular edges see E ~ 0 -> alpha
masked to 0, the diagonals have zero couple). For a genuinely 2-D current pattern it is a
first-order directional estimate. UNITS follow the D1 contact-current convention: the 2D
layered mesh is in metres with implicit out-of-plane depth, so I_sub is per-unit-depth [A/m] --
pass depth_m (the cell period_y) to scale to amperes.

Si constants: van Overstraeten & de Man, Solid-State Electron. 13, 583 (1970), 300 K, in SI
(beware the literature's cm units: a_n = 7.03e5 1/cm = 7.03e7 1/m, b_n = 1.231e6 V/cm =
1.231e8 V/m; electrons single range, holes low-field range E < 4e7 V/m).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from dynameta.constants import Q_E

__all__ = ["ChynowethParams", "SILICON_VANOVERSTRAETEN", "impact_generation_edges",
           "substrate_current"]


@dataclass(frozen=True)
class ChynowethParams:
    """Chynoweth ionization coefficients, SI: a in 1/m, b in V/m."""
    a_n_per_m: float
    b_n_V_per_m: float
    a_p_per_m: float
    b_p_V_per_m: float

    def __post_init__(self):
        for nm in ("a_n_per_m", "b_n_V_per_m", "a_p_per_m", "b_p_V_per_m"):
            if not (getattr(self, nm) >= 0.0):
                raise ValueError("ChynowethParams: {} must be >= 0".format(nm))

    @staticmethod
    def _alpha(a: float, b: float, E_V_per_m):
        E = np.asarray(E_V_per_m, dtype=np.float64)
        out = np.zeros_like(E)
        pos = E > 0.0
        with np.errstate(under="ignore"):
            out[pos] = a * np.exp(-b / E[pos])               # E <= 0 stays EXACTLY 0
        return out if out.ndim else float(out)

    def alpha_n(self, E_V_per_m):
        """Electron ionization coefficient alpha_n(E) [1/m]; exactly 0 for E <= 0."""
        return self._alpha(self.a_n_per_m, self.b_n_V_per_m, E_V_per_m)

    def alpha_p(self, E_V_per_m):
        """Hole ionization coefficient alpha_p(E) [1/m]; exactly 0 for E <= 0."""
        return self._alpha(self.a_p_per_m, self.b_p_V_per_m, E_V_per_m)


# van Overstraeten-de Man Si, 300 K (SI; electrons single range, holes E < 4e7 V/m range)
SILICON_VANOVERSTRAETEN = ChynowethParams(a_n_per_m=7.03e7, b_n_V_per_m=1.231e8,
                                          a_p_per_m=1.582e8, b_p_V_per_m=2.036e8)


def impact_generation_edges(device: str, region: str,
                            params: ChynowethParams) -> Dict[str, np.ndarray]:
    """Per-edge impact-ionization profile on the SOLVED bipolar-DD region: |E| [V/m], |J_n|/|J_p|
    [A/m^2], the FV weight EdgeCouple*EdgeLength, and q*G = alpha_n|J_n| + alpha_p|J_p| [A/m^3]
    (the spatial driver, e.g. for locating the hot-carrier injection zone). Requires the region
    to carry the bipolar edge models (ElectricField, ElectronCurrent, HoleCurrent)."""
    import devsim as ds

    def em(name):
        return np.asarray(ds.get_edge_model_values(device=device, region=region, name=name),
                          dtype=np.float64)

    E = np.abs(em("ElectricField"))
    Jn, Jp = np.abs(em("ElectronCurrent")), np.abs(em("HoleCurrent"))
    weight = em("EdgeCouple") * em("EdgeLength")
    qG = params.alpha_n(E) * Jn + params.alpha_p(E) * Jp
    return {"E_V_per_m": E, "Jn_A_m2": Jn, "Jp_A_m2": Jp, "weight_volume": weight,
            "qG_A_m3": qG}


def substrate_current(device: str, region: str, params: ChynowethParams, *,
                      depth_m: Optional[float] = None) -> float:
    """Impact-generated (substrate) current I_sub = Int (alpha_n|J_n| + alpha_p|J_p|) dV over the
    solved region, via the directional FV edge quadrature (module header). 2D layered meshes
    return per-unit-depth [A/m]; pass depth_m = period_y_m to scale to [A] (the D1 convention).
    Zero coefficients (or a field everywhere below the exponential's reach) give EXACTLY 0.0."""
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None to keep per-unit-depth units)")
    prof = impact_generation_edges(device, region, params)
    i_sub = float(np.sum(prof["qG_A_m3"] * prof["weight_volume"]))
    return i_sub * (float(depth_m) if depth_m is not None else 1.0)
