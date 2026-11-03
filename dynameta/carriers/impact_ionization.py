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

3-D (D2c, v0.14.0): ELEMENT-BASED. DEVSIM's element_from_edge_model rebuilds, in every tetrahedron, the
vector E, J_n and J_p from their edge projections (the exact P1 gradient for E; exact for any uniform
vector), so |E| and |J| are ISOTROPIC -- no directional masking, unlike the 2-D edge quadrature -- and

    I_sub = sum_{element edges} (alpha_n(|E|) |J_n| + alpha_p(|E|) |J_p|) * 2 ElementNodeVolume

(sum 2*ENV = the region volume). That last identity is what the pre-D2c gmsh mesh broke (DEVSIM sums
the magnitudes of each tet's circumcentric pieces: 2 sum ENV = 1.20-1.52 x the volume there), and what
made the old 3-D edge quadrature read 1.8-3.1x low; so the 3-D path first checks 2 sum ENV against the
true tetrahedron volumes from the element node list and REFUSES a mesh where they differ by > 1e-9 (use
Stacked3DSpec(mesher='tensor'), exact to 1e-15). MEASURED (tests/test_devsim_3d_mesh.py): a 3-D
uniform-field bar matches alpha(E) J Volume to 1e-12, and a laterally uniform 3-D reverse-biased p-n
diode on the tensor mesh equals the 1-D edge quadrature on the same z-lines to 1e-12 -- per Kuhn tet the
element field is the z-cell's edge field and the SG currents on its diagonal edges are exact projections.

Si constants: van Overstraeten & de Man, Solid-State Electron. 13, 583 (1970), 300 K, in SI
(beware the literature's cm units: a_n = 7.03e5 1/cm = 7.03e7 1/m, b_n = 1.231e6 V/cm =
1.231e8 V/m; electrons single range, holes low-field range E < 4e7 V/m).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


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
    [A/m^2], the FV weight EdgeCouple*EdgeLength (3-D: per ELEMENT edge, weight 2*ElementNodeVolume),
    and q*G = alpha_n|J_n| + alpha_p|J_p| [A/m^3]
    (the spatial driver, e.g. for locating the hot-carrier injection zone). Requires the region
    to carry the bipolar edge models (ElectricField, ElectronCurrent, HoleCurrent)."""
    import devsim as ds

    # audit 7.1 regrade (P2): the directional edge quadrature sum(alpha|J| EdgeCouple
    # EdgeLength) is EXACT on the 2D layered tensor mesh (2V weight identity, zero-couple
    # diagonals) but on 3D unstructured tets it read 1.8-3.1x LOW (a 1.6-2.1x OPTIMISTIC REL8
    # lifetime). D2c (v0.14.0): 3-D now takes the d-consistent ELEMENT reconstruction (module
    # header) on a mesh whose element volumes are exact, and refuses any other.
    if int(ds.get_dimension(device=device)) == 3:
        return _impact_generation_elements_3d(device, region, params)

    def em(name):
        return np.asarray(ds.get_edge_model_values(device=device, region=region, name=name),
                          dtype=np.float64)

    E = np.abs(em("ElectricField"))
    Jn, Jp = np.abs(em("ElectronCurrent")), np.abs(em("HoleCurrent"))
    weight = em("EdgeCouple") * em("EdgeLength")
    qG = params.alpha_n(E) * Jn + params.alpha_p(E) * Jp
    return {"E_V_per_m": E, "Jn_A_m2": Jn, "Jp_A_m2": Jp, "weight_volume": weight,
            "qG_A_m3": qG}


def _impact_generation_elements_3d(device: str, region: str,
                                   params: ChynowethParams) -> Dict[str, np.ndarray]:
    """3-D element-based profile (module header): per ELEMENT EDGE the isotropic |E|, |J_n|, |J_p| of its
    tetrahedron and the weight 2*ElementNodeVolume. Element models it creates are deleted again (models
    that already existed, e.g. BTBT's ElectricField_x/y/z, are only read)."""
    import devsim as ds

    def elv(name):
        return np.asarray(ds.get_element_model_values(device=device, region=region, name=name),
                          dtype=np.float64)
    w = 2.0 * elv("ElementNodeVolume")
    xyz = np.column_stack([np.asarray(ds.get_node_model_values(device=device, region=region, name=c),
                                      dtype=np.float64) for c in ("x", "y", "z")])
    tets = np.asarray(ds.get_element_node_list(device=device, region=region), dtype=int)
    d = xyz[tets[:, 1:]] - xyz[tets[:, :1]]
    v_true = float(np.abs(np.linalg.det(d)).sum() / 6.0)
    if not abs(w.sum() / v_true - 1.0) < 1e-9:
        raise NotImplementedError(
            "impact_generation_edges/substrate_current (3-D): DEVSIM's element volumes on region {!r} sum "
            "to {:.4f} x the true tetrahedron volume, so any I_sub quadrature on this mesh is biased (the "
            "pre-D2c gmsh mesh over-counts 1.20-1.52 x; audit 7.1 measured the old edge quadrature 1.8-3.1x "
            "low). Build the device with Stacked3DSpec(mesher='tensor') (exact) -- "
            "docs/audit/2026-11-03-devsim-3d-mesh.md.".format(region, w.sum() / v_true))
    have = set(ds.get_element_model_list(device=device, region=region))
    made = []
    try:
        mags = {}
        for m in ("ElectricField", "ElectronCurrent", "HoleCurrent"):
            comps = [m + "_" + c for c in "xyz"]
            if not all(c in have for c in comps):
                ds.element_from_edge_model(device=device, region=region, edge_model=m)
                made += [c for c in comps if c not in have]
            mags[m] = np.sqrt(sum(elv(c) ** 2 for c in comps))
    finally:
        for m in made:
            ds.delete_element_model(device=device, region=region, name=m)
    E, Jn, Jp = mags["ElectricField"], mags["ElectronCurrent"], mags["HoleCurrent"]
    qG = params.alpha_n(E) * Jn + params.alpha_p(E) * Jp
    return {"E_V_per_m": E, "Jn_A_m2": Jn, "Jp_A_m2": Jp, "weight_volume": w, "qG_A_m3": qG}


def substrate_current(device: str, region: str, params: ChynowethParams, *,
                      depth_m: Optional[float] = None) -> float:
    """Impact-generated (substrate) current I_sub = Int (alpha_n|J_n| + alpha_p|J_p|) dV over the
    solved region, via the directional FV edge quadrature in 2-D and the element reconstruction in 3-D
    (module header). 2D layered meshes return per-unit-depth [A/m]; pass depth_m = period_y_m to scale
    to [A] (the D1 convention). A 3-D mesh is in metres: the result is [A], leave depth_m None.
    Zero coefficients (or a field everywhere below the exponential's reach) give EXACTLY 0.0."""
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None to keep per-unit-depth units)")
    prof = impact_generation_edges(device, region, params)
    i_sub = float(np.sum(prof["qG_A_m3"] * prof["weight_volume"]))
    return i_sub * (float(depth_m) if depth_m is not None else 1.0)
