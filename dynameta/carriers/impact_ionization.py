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

QUADRATURE -- `quadrature=` on impact_generation_edges / substrate_current ("auto" by default):

  "element" (the 2-D and 3-D default since v0.14.0). DEVSIM's element_from_edge_model rebuilds, in every
  triangle / tetrahedron, the vector E, J_n and J_p from their edge projections (the exact P1 gradient
  for E; exact for any uniform vector), so |E| and |J| are ISOTROPIC, and

      I_sub = sum_{element edges} (alpha_n(|E|) |J_n| + alpha_p(|E|) |J_p|) * 2 ElementNodeVolume

  (2 sum ENV = the element volume). That identity is exact only on a mesh whose elements are well
  centred (DEVSIM sums the MAGNITUDES of the circumcentric dual pieces), so the element path first runs
  the shared check element_geometry.check_element_geometry and REFUSES (InexactElementGeometryError, a
  NotImplementedError) any other mesh: the pre-D2c gmsh tets over-count 1.20-1.52 x, one obtuse
  triangle 11 x. Exact meshes: the 2-D rectilinear tensor mesh (LayeredDevsimBuilder; right triangles,
  zero-couple hypotenuses) and the 3-D Kuhn tensor mesh (Stacked3DSpec(mesher='tensor')). On the 2-D
  right-triangle mesh both legs of a triangle take the (leg_x, leg_y) pair vector and the hypotenuse has
  weight 0, so the integrand is alpha(sqrt(E_x^2 + E_y^2)) sqrt(J_x^2 + J_y^2) per triangle, weighted by
  its area. ONE kernel (_impact_generation_elements) serves 2-D and 3-D. MEASURED (tests/
  test_impact_ionization.py): a uniform field at 0/30/45/60 deg to the 2-D mesh matches alpha(E) J Area
  to <= 1e-12; a harmonic (x^2 - y^2) field converges at second order to the Gauss-Legendre integral of
  alpha(|E|)|J|; (tests/test_devsim_3d_mesh.py) a 3-D uniform-field bar matches alpha(E) J Volume to
  1e-12 and a laterally uniform 3-D reverse diode equals the 1-D edge quadrature to 5e-10.

  "edge" (the 1-D default; 2-D legacy, opt-in; refused in 3-D). The directional FV edge quadrature

      I_sub = sum_edges alpha(|E_e|) |J_e| * EdgeCouple_e * EdgeLength_e .

  In 1-D the edge IS the field direction and this is exact. On DEVSIM's 2-D tensor mesh the diagonals
  have ZERO couple and the x- and y-edges each own the whole region volume (sum EdgeCouple*EdgeLength =
  2 Volume), so under refinement it converges to

      I_edge = Int [alpha(|E_x|) |J_x| + alpha(|E_y|) |J_y|] dA      (componentwise, NOT isotropic)

  -- exact ONLY for mesh-aligned current (a constant-field bar, a laterally uniform stack). For a field at
  angle t to the mesh I_edge/I_true = [cos t e^{-b/(E cos t)} + sin t e^{-b/(E sin t)}] / e^{-b/E}: 0.270
  at 45 deg and b/E = 4 (MEASURED to 1e-12 in tests/test_impact_ionization.py), i.e. it under-reads
  junction corners, current crowding at contacts and drain corners. It was the 2-D default before
  v0.14.0; select it explicitly only to reproduce those numbers or for a mesh-aligned 2-D current. On 3-D
  tets it read 1.8-3.1x low (audit 7.1), hence refused there.

UNITS follow the D1 contact-current convention: the 2D layered mesh is in metres with implicit
out-of-plane depth, so I_sub is per-unit-depth [A/m] -- pass depth_m (the cell period_y) to scale to
amperes. A 3-D mesh gives [A].

Si constants: van Overstraeten & de Man, Solid-State Electron. 13, 583 (1970), 300 K, in SI
(beware the literature's cm units: a_n = 7.03e5 1/cm = 7.03e7 1/m, b_n = 1.231e6 V/cm =
1.231e8 V/m; electrons single range, holes low-field range E < 4e7 V/m).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


__all__ = ["ChynowethParams", "SILICON_VANOVERSTRAETEN", "QUADRATURES", "impact_generation_edges",
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


QUADRATURES = ("auto", "edge", "element")


def _resolve_quadrature(dim: int, quadrature: str) -> str:
    if quadrature not in QUADRATURES:
        raise ValueError("impact ionization: quadrature must be one of {} (got {!r})".format(
            QUADRATURES, quadrature))
    if quadrature == "auto":
        return "edge" if dim == 1 else "element"
    if quadrature == "edge" and dim == 3:
        raise ValueError("impact ionization: quadrature='edge' is not available in 3-D (it read 1.8-3.1x "
                         "low on tetrahedra, audit 7.1); use 'element' (the default)")
    if quadrature == "element" and dim == 1:
        raise ValueError("impact ionization: quadrature='element' needs element models, which DEVSIM does "
                         "not have in 1-D; use 'edge' (the default there, and exact in 1-D)")
    return quadrature


def impact_generation_edges(device: str, region: str, params: ChynowethParams, *,
                            quadrature: str = "auto") -> Dict[str, np.ndarray]:
    """Per-edge impact-ionization profile on the SOLVED bipolar-DD region: |E| [V/m], |J_n|/|J_p|
    [A/m^2], the FV volume weight of each entry, and q*G = alpha_n|J_n| + alpha_p|J_p| [A/m^3] (the
    spatial driver, e.g. for locating the hot-carrier injection zone); I_sub = sum(qG * weight_volume).
    Requires the region to carry the bipolar edge models (ElectricField, ElectronCurrent, HoleCurrent).

    quadrature (module header):
      "auto"    (default) 1-D -> "edge" (exact in 1-D); 2-D / 3-D -> "element".
      "element" 2-D / 3-D: per ELEMENT EDGE the isotropic |E|, |J| of its triangle / tetrahedron,
                weight 2*ElementNodeVolume. Correct for ANY current direction; refuses a mesh whose
                element volumes are not exact (element_geometry.check_element_geometry). Use it for every
                2-D / 3-D calculation -- junction corners, current crowding, drain corners, HCI inputs.
      "edge"    1-D / 2-D: per region edge alpha(|E_e|)|J_e|, weight EdgeCouple*EdgeLength. In 2-D it
                converges to Int [alpha(|E_x|)|J_x| + alpha(|E_y|)|J_y|] dA, NOT Int alpha(|E|)|J| dA:
                valid ONLY when the current is mesh-aligned everywhere (a 1-D-like bar or a laterally
                uniform stack); 0.270 of the truth at 45 deg and b/E = 4. Kept to reproduce pre-v0.14 2-D
                numbers. Not available in 3-D."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()

    # audit 7.1 regrade (P2): the directional edge quadrature sum(alpha|J| EdgeCouple EdgeLength) read
    # 1.8-3.1x LOW on 3D unstructured tets, and in 2-D it is directional (row 3.2-12). v0.14.0: 2-D and
    # 3-D take the d-consistent ELEMENT reconstruction (module header) on a mesh whose element volumes are
    # exact, and refuse any other.
    q = _resolve_quadrature(int(ds.get_dimension(device=device)), quadrature)
    if q == "element":
        return _impact_generation_elements(device, region, params)

    def em(name):
        return np.asarray(ds.get_edge_model_values(device=device, region=region, name=name),
                          dtype=np.float64)

    E = np.abs(em("ElectricField"))
    Jn, Jp = np.abs(em("ElectronCurrent")), np.abs(em("HoleCurrent"))
    weight = em("EdgeCouple") * em("EdgeLength")
    qG = params.alpha_n(E) * Jn + params.alpha_p(E) * Jp
    return {"E_V_per_m": E, "Jn_A_m2": Jn, "Jp_A_m2": Jp, "weight_volume": weight,
            "qG_A_m3": qG}


def _impact_generation_elements(device: str, region: str,
                                params: ChynowethParams) -> Dict[str, np.ndarray]:
    """2-D / 3-D element-based profile (module header): per ELEMENT EDGE the isotropic |E|, |J_n|, |J_p| of
    its triangle / tetrahedron and the weight 2*ElementNodeVolume, after the shared exact-volume check.
    Element models it creates are deleted again (models that already existed, e.g. BTBT's
    ElectricField_x/y/z, are only read)."""
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers.element_geometry import check_element_geometry
    ds = import_devsim()
    dim = int(ds.get_dimension(device=device))
    check_element_geometry(device, region, caller="impact_generation_edges/substrate_current ({}-D)"
                           .format(dim))

    def elv(name):
        return np.asarray(ds.get_element_model_values(device=device, region=region, name=name),
                          dtype=np.float64)
    w = 2.0 * elv("ElementNodeVolume")
    have = set(ds.get_element_model_list(device=device, region=region))
    made = []
    try:
        mags = {}
        for m in ("ElectricField", "ElectronCurrent", "HoleCurrent"):
            comps = [m + "_" + c for c in "xyz"[:dim]]
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
                      depth_m: Optional[float] = None, quadrature: str = "auto") -> float:
    """Impact-generated (substrate) current I_sub = Int (alpha_n|J_n| + alpha_p|J_p|) dV over the
    solved region. quadrature as impact_generation_edges: the default "auto" is the edge quadrature in
    1-D (exact) and the element reconstruction in 2-D / 3-D (isotropic, any current direction); "edge"
    in 2-D is the legacy directional estimate, valid only for mesh-aligned current. 2D layered meshes
    return per-unit-depth [A/m]; pass depth_m = period_y_m to scale to [A] (the D1 convention). A 3-D
    mesh is in metres: the result is [A], leave depth_m None. Zero coefficients (or a field everywhere
    below the exponential's reach) give EXACTLY 0.0."""
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None to keep per-unit-depth units)")
    prof = impact_generation_edges(device, region, params, quadrature=quadrature)
    i_sub = float(np.sum(prof["qG_A_m3"] * prof["weight_volume"]))
    return i_sub * (float(depth_m) if depth_m is not None else 1.0)
