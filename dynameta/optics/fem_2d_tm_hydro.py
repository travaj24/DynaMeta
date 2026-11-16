"""
Nonlocal (hydrodynamic) free-carrier response of a z-graded conductor layer inside the 2-D
periodic TM cell of optics.fem_2d_tm.

WHAT IT SOLVES. A laterally uniform conductor slab (one layer of a TMLayer stack, possibly under
or over laterally structured layers such as metal strips) whose equilibrium electron density
n0(z) -- optionally n0(x, z) -- is given on a grid, with effective mass m, background eps_inf,
damping gamma, the hydrodynamic pressure convention ('high_freq' or 'thomas_fermi') and the
optional GNOR diffusion constant D. The free carriers respond nonlocally (a longitudinal wave
with a sub-nanometre wavelength / screening length), with a hard wall J.n = 0 on the slab's top
and bottom faces and Bloch-periodic continuation laterally.

MODEL (the kernel of optics.nonlocal_tmm, reused, not re-derived). Linearised barotropic electron
gas about an equilibrium n0 in force balance under the same pressure:

    (d/dt + gamma) J = eps0 wp(x, z)**2 (E + grad psi),
    psi = c rho1,  c = -beta_eff**2 / (eps0 wp**2)        (nonlocal_tmm.hydro_psi_per_charge)

with wp**2 proportional to n0 and beta = beta_from_density(n0) LOCAL in the density (GNOR:
beta_eff**2 = beta**2 + D (gamma - i omega)). Inside one conductor (E_t, H_t, J.n, psi) are
continuous (nonlocal_tmm.HYDRO_INTERFACE_FIELDS); J.n = 0 only on the conductor boundary.

FORMULATION (unknowns chosen by measurement; see "WHY THIS FORM" below). Three fields on the
product space of the TM solver, H_y first:

    H    in H1 (whole cell, Bloch-periodic)          -- the TM field of fem_2d_tm,
    X    in L2 (vector, conductor only)              -- the free current J,
    phi  in H1 (conductor only, Bloch-periodic)      -- phi = i omega eps0 psi.

All fields are normalised like H (unit incident H_y at the top interface; lengths in nm), X in
units of curl H. With J the in-plane rotation (a, b) -> (-b, a), so that curl(H_y y) = J grad H,
and E = (J grad H - X) / (-i omega eps0 eps_inf) in the conductor, the weak equations (tests v, K,
w of H, X, phi) over the conductor C are

  (M)  int_cell Lambda grad H . grad v - k0**2 H v   -   int_C (1/eps_inf) X . J grad v       = src
  (K)  int_C (1/Delta + 1/eps_inf) X . K - (1/eps_inf) (J grad H) . K + grad phi . K
                                                       = int_C (1/eps_inf) (J grad H_bg) . K
  (W)  int_C -kappa**2 phi w + X . grad w = 0,           kappa**2 = -L0**2 / (eps0 c)

with Lambda = 1/eps_inf in C (the H_y operator sees only the background; the free response is
carried by X) and Delta = eps_T - eps_inf = -wp**2/(omega**2 + i omega gamma)
(nonlocal_tmm.eps_transverse). (K) is J = sigma (E + grad psi) divided by -i omega eps0 sigma; (W)
is rho1 = div J / (i omega) tested with w and integrated by parts, its boundary term being the
normal current. The form is complex SYMMETRIC (the X-H and X-phi couplings appear transposed in
(M)/(K) and (K)/(W)).

Boundary and interface conditions, all without being imposed:
  * J.n = 0 on the conductor's outer boundary is the NATURAL condition of (W): nothing is added.
  * psi continuous inside the conductor: phi is H1-conforming (continuity of the electrochemical
    potential -- not of the pressure term beta**2 n1, which the planar route (nonlocal_tmm) shows to be wrong
    for graded density).
  * J.n continuous inside the conductor: natural condition of (W).
  * tangential E continuous across the conductor faces: natural condition of (M), with E including
    the free-current part.
Eliminating X in a uniform conductor gives  Delta (eps_inf/eps_T) lap phi = kappa**2 phi, i.e.
k_L**2 = (omega**2 + i omega gamma - wp**2/eps_inf)/beta_eff**2 = nonlocal_tmm.kL_squared.
beta -> 0 drives phi -> 0 through the kappa**2 penalty and (K) becomes the local Drude current;
the EXACT local limit (beta_scale = 0, D = 0) is routed to the local solver with eps_T(x, z).

WHY THIS FORM (measured; see docs/audit/2026-11-14-v0.16-nonlocal-2d.md). The alternatives:
  (a) phi alone, J eliminated analytically: the coefficients become 1/eps_T and Delta eps_inf/eps_T,
      singular where Re eps_T crosses zero -- which a graded accumulation/depletion profile does by
      construction -- so the quadrature has to resolve a Lorentzian of width ~ Im(eps_T)/|d eps_T/dz|
      that the smooth nonlocal solution does not have. (K)-(W) contain only 1/Delta (proportional
      to 1/n0) and 1/eps_inf: no eps_T in a denominator anywhere, like the planar generator.
  (b) J in H(div) with J.n = 0 essential: the same physics, but the hard wall must be imposed on
      interior mesh edges (boundary markers the mesh does not carry), ~2x the current DOFs of an
      RT pair, and a Bloch-periodic H(div) space; (K)-(W) needs none of it -- X is element-local.
  (c) the stabilised psi = div J form of optics.hydro_fem: its continuous field is div J (beta**2 n1
      up to a constant), the wrong interface quantity for graded density (it is exact only for a
      uniform conductor).

MESH. The longitudinal length 1/|k_L| is sub-nanometre and the density varies over ~1 nm, so the
conductor carries its own z lines, independent of the optical mesh: rows of height
h_z = h_per_screening / max|k_L| (or h_z_m), the profile samples as lines (when <= the
max_grid_lines cap), and optional geometric grading toward the two faces (face_levels,
face_ratio). The x lines are the optical mesh's (strip edges and their corner grading run through
every layer) plus, optionally, a uniform h_x_m. hydro_z_convergence tabulates r against h_z and the
order.

VALIDITY. The hydrodynamic model is the long-wavelength expansion of the electron-gas response:
with the 'high_freq' beta it needs u = |k_L| v_F/omega << 1 (nonlocal_tmm.hydro_validity, default
limit 0.5). The report is attached to every solve (TMSolution.diagnostics['hydro'][layer]
['validity']) and, unless check_validity=False, an UnreliableResultWarning names the worst
sub-layer. The solve is never blocked.

NOT SUPPORTED (refused with a message): hydrodynamic BLOCKS (strips made of a nonlocal metal) and
claddings; two adjacent hydrodynamic layers (ambiguous wall); HydroParams / HydroLayer records as
layer eps (build a HydroConductor2D); a complex eps_inf (the planar kernel's eps_inf is real);
non-positive densities.

References: the hydrodynamic-model literature cited in optics.nonlocal_tmm (Melnyk and Harris
1970; Barton 1979; Raza et al., J. Phys. Condens. Matter 27, 183204 (2015); Mortensen et al.,
Nat. Commun. 5, 3809 (2014) for GNOR).

Conventions: those of optics.fem_2d_tm (exp(-i omega t), r at the top interface, 'x'/'p').
ASCII only; NGSolve imported lazily.
"""

from __future__ import annotations

import cmath
import dataclasses
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Hashable, List, Optional, Sequence

import numpy as np

from dynameta.constants import C_LIGHT, EPS0
from dynameta.core.lateral import check_periodic_axis, is_x_invariant
from dynameta.optics.nonlocal_tmm import (
    HydroLayer,
    beta_from_density,
    eps_transverse,
    hydro_layers_from_profile,
    hydro_psi_per_charge,
    hydro_validity,
    kL_squared,
)
from dynameta.warning_types import UnreliableResultWarning

__all__ = [
    "HydroConductor2D", "hydro_conductor_from_profile", "hydro_extensions_for_cell",
    "hydro_z_convergence",
]

_L0 = 1.0e-9                     # mesh length unit (nm) in metres
_S = 1.0 / _L0                   # m -> nm
_CONVENTIONS = ("high_freq", "thomas_fermi")


# ================================================================================================
# the material / layer spec
# ================================================================================================

@dataclass(frozen=True, eq=False)              # array fields: identity equality / hashing
class HydroConductor2D:
    """A hydrodynamic conductor LAYER for optics.fem_2d_tm (pass it as a TMLayer's eps).

    n0_m3    : equilibrium electron density [m**-3], shape (Nz,) (z-graded) or (Nz, Nx) (also
               laterally structured; periodic in x). Must be > 0 everywhere.
    z_m      : sample positions RELATIVE TO THE LAYER BOTTOM [m], strictly increasing, spanning
               exactly [0, thickness] (the TMLayer thickness is checked against it). NOTE: the
               planar route (hydro_layers_from_profile) counts z from the TOP face; use
               hydro_conductor_from_profile to drive both routes with the same arguments.
    m_eff_kg : effective mass (parabolic band; wp**2 = n0 e**2/(eps0 m), beta = beta_from_density).
    eps_inf  : background (bound-charge) permittivity, real.
    gamma    : damping rate [1/s].
    convention : 'high_freq' (beta**2 = 3/5 v_F**2, default) or 'thomas_fermi' (v_F**2/3), as in
               nonlocal_tmm.
    D        : GNOR diffusion constant [m**2/s] (beta_eff**2 = beta**2 + D (gamma - i omega)).
    x_m      : lateral sample positions in [0, period) [m] for an (Nz, Nx) density.
    conductor_id : a label (reported in diagnostics; one HydroConductor2D is one conductor).
    beta_scale : multiplies beta (1 = the physical value; 0 with D = 0 is the exact local limit,
               solved by the local solver with eps_T(x, z)).

    Mesh controls of the conductor (independent of the optical mesh):
    h_z_m    : element height in the conductor [m]; None = h_per_screening / max|k_L| (k_L the
               longitudinal wavevector at the solve frequency over the profile).
    h_per_screening : that ratio (default 1.0: at order 4 the planar graded-film fixture of
               tests/test_fem_2d_tm_hydro.py is then at 1.4e-10 of the planar route with the samples
               aligned; in the propagating longitudinal regime a uniform film with k_L d = 44 is at
               6.3e-12 of the closed-form single layer at order 5).
    face_levels / face_ratio : geometric grading of the rows toward the two faces (the
               longitudinal boundary layers sit there).
    h_x_m    : optional cap on the lateral element size [m] (adds uniform x lines to the whole
               cell); None = the optical mesh's x lines.
    align_samples : put mesh lines at the profile's z samples when there are <= max_grid_lines of
               them (the density is piecewise linear between samples).
    coef_points : uniform z points (besides the samples) on which beta(n0(z)) is tabulated for its
               piecewise-linear coefficient (wp**2 is exact: linear in n0).

    Validity: check_validity=True emits an UnreliableResultWarning when the solve is outside the
    hydrodynamic model's validity (nonlocal_tmm.hydro_validity; E_F/k_B T is also checked when T_K
    is given). The report is always in TMSolution.diagnostics['hydro'][layer]['validity']."""
    _TM_HYDRO_CONDUCTOR: ClassVar[bool] = True

    n0_m3: Any
    z_m: Any
    m_eff_kg: float
    eps_inf: float
    gamma: float
    convention: str = "high_freq"
    D: float = 0.0
    x_m: Any = None
    conductor_id: Optional[Hashable] = None
    beta_scale: float = 1.0
    h_z_m: Optional[float] = None
    h_per_screening: float = 1.0
    face_levels: int = 0
    face_ratio: float = 0.5
    h_x_m: Optional[float] = None
    align_samples: bool = True
    coef_points: int = 2049
    check_validity: bool = True
    T_K: Optional[float] = None

    def __post_init__(self):
        n = np.asarray(self.n0_m3, dtype=float)
        z = np.asarray(self.z_m, dtype=float).ravel()
        if n.ndim == 2 and n.shape[1] == 1:
            n = n[:, 0]
        if n.ndim not in (1, 2) or n.shape[0] != z.size or z.size < 2:
            raise ValueError("HydroConductor2D: n0_m3 must be (Nz,) or (Nz, Nx) with Nz = len(z_m) "
                             ">= 2; got n0 {} and z {}".format(n.shape, z.shape))
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(z))):
            raise ValueError("HydroConductor2D: n0_m3 and z_m must be finite")
        if np.any(np.diff(z) <= 0.0):
            raise ValueError("HydroConductor2D: z_m must be strictly increasing (from the layer "
                             "bottom)")
        if np.any(n <= 0.0):
            raise ValueError("HydroConductor2D: n0_m3 must be > 0 everywhere (the hydrodynamic "
                             "current needs free carriers; a fully depleted region is a dielectric "
                             "layer of its own)")
        if n.ndim == 2:
            if self.x_m is None:
                raise ValueError("HydroConductor2D: an (Nz, Nx) density needs x_m")
            x = np.asarray(self.x_m, dtype=float).ravel()
            if x.size != n.shape[1] or np.any(np.diff(x) <= 0):
                raise ValueError("HydroConductor2D: x_m must be strictly increasing with Nx "
                                 "entries")
        if self.convention not in _CONVENTIONS:
            raise ValueError("HydroConductor2D: convention must be 'high_freq' or 'thomas_fermi'; "
                             "got {!r}".format(self.convention))
        if isinstance(self.eps_inf, complex) or np.iscomplexobj(self.eps_inf):
            raise TypeError("HydroConductor2D: eps_inf must be real (the hydrodynamic kernel's "
                            "background is real; put interband loss in a separate layer)")
        for nm in ("m_eff_kg", "eps_inf"):
            if not float(getattr(self, nm)) > 0:
                raise ValueError("HydroConductor2D: {} must be positive".format(nm))
        if not float(self.gamma) >= 0 or not float(self.D) >= 0 or not float(self.beta_scale) >= 0:
            raise ValueError("HydroConductor2D: gamma, D and beta_scale must be >= 0")
        if self.h_z_m is not None and not float(self.h_z_m) > 0:
            raise ValueError("HydroConductor2D: h_z_m must be positive")
        if not float(self.h_per_screening) > 0:
            raise ValueError("HydroConductor2D: h_per_screening must be positive")
        if int(self.face_levels) < 0 or not (0.0 < float(self.face_ratio) < 1.0):
            raise ValueError("HydroConductor2D: face_levels >= 0 and 0 < face_ratio < 1")
        if int(self.coef_points) < 2:
            raise ValueError("HydroConductor2D: coef_points must be >= 2")

    # ---- basic geometry / data ----------------------------------------------------------------
    @property
    def thickness_m(self) -> float:
        z = np.asarray(self.z_m, dtype=float).ravel()
        return float(z[-1] - z[0])

    @property
    def laterally_structured(self) -> bool:
        return self._n2d().shape[1] > 1

    @property
    def is_local_limit(self) -> bool:
        """beta_scale = 0 and D = 0: no longitudinal wave, the local Drude eps_T(x, z)."""
        return float(self.beta_scale) == 0.0 and float(self.D) == 0.0

    def _z_nm(self) -> np.ndarray:
        z = np.asarray(self.z_m, dtype=float).ravel()
        return (z - z[0]) * _S

    def _n2d(self) -> np.ndarray:
        """The density as (Nz, Nx); identical lateral columns collapse to (Nz, 1) -- a laterally uniform
        profile is solved as one (core.lateral's test, shared by every optical route)."""
        n = np.asarray(self.n0_m3, dtype=float)
        n = n.reshape(n.shape[0], -1)
        if n.shape[1] > 1 and is_x_invariant(n[:, None, :]):
            n = n[:, :1]
        return n

    def with_options(self, **kw) -> "HydroConductor2D":
        """A copy with some fields replaced (e.g. mesh controls for a convergence study)."""
        return dataclasses.replace(self, **kw)

    # ---- the planar kernel at points ----------------------------------------------------------
    def _wp2_beta(self, n) -> tuple:
        """(wp**2 [rad**2/s**2], beta [m/s]) at densities n (array): beta from the kernel's
        beta_from_density (times beta_scale); wp**2 proportional to n, anchored on the kernel's
        own HydroLayer of the largest density (hydro_layers_from_profile builds it exactly as the
        planar route does)."""
        n = np.asarray(n, dtype=float)
        n_ref = float(np.max(n))
        ref = hydro_layers_from_profile(np.array([0.0, 1.0]), np.array([n_ref, n_ref]),
                                        m_eff_kg=float(self.m_eff_kg), eps_inf=float(self.eps_inf),
                                        gamma=float(self.gamma), convention=self.convention,
                                        D=float(self.D))[0]
        wp2 = ref.wp * ref.wp * (n / n_ref)
        beta = np.asarray(beta_from_density(n, float(self.m_eff_kg), self.convention),
                          dtype=float) * float(self.beta_scale)
        return wp2, beta

    def _point_layers(self, n) -> List[HydroLayer]:
        wp2, beta = self._wp2_beta(np.ravel(n))
        return [HydroLayer(float(self.eps_inf), math.sqrt(w), float(self.gamma), float(b), 0.0,
                           D=float(self.D)) for w, b in zip(wp2, beta)]

    def planar_layers(self, column: int = 0, n_layers: Optional[int] = None) -> list:
        """This conductor as the planar route's run (nonlocal_tmm.hydro_layers_from_profile,
        listed TOP FIRST, beta scaled by beta_scale) for one lateral column of the density."""
        z = self._z_nm() * _L0
        n = self._n2d()[:, column]
        zt = z[-1] - z[::-1]
        run = hydro_layers_from_profile(zt, n[::-1], m_eff_kg=float(self.m_eff_kg),
                                        eps_inf=float(self.eps_inf), gamma=float(self.gamma),
                                        n_layers=n_layers, convention=self.convention,
                                        D=float(self.D), conductor_id=self.conductor_id or "graded")
        if float(self.beta_scale) != 1.0:
            run = [dataclasses.replace(L, beta=L.beta * float(self.beta_scale)) for L in run]
        return run

    def validity(self, omega) -> Any:
        """nonlocal_tmm.hydro_validity over every column's sub-layers (the worst column's report;
        u is listed for that column)."""
        worst = None
        for j in range(self._n2d().shape[1]):
            rep = hydro_validity(omega, self.planar_layers(column=j), convention=self.convention,
                                 m_eff_kg=float(self.m_eff_kg), T_K=self.T_K)
            u = max([v for v in rep.u if v == v] or [0.0])
            if worst is None or (not rep.ok and worst[0].ok) or (rep.ok == worst[0].ok and u > worst[1]):
                worst = (rep, u)
        return worst[0]

    # ---- what build_tm_cell asks for ----------------------------------------------------------
    def region_setup(self, *, lambda_m: float, thickness_nm: float, period_nm: float,
                     max_grid_lines: int, where: str) -> Dict[str, Any]:
        """Operator eps, sizing samples and mesh lines of this layer (called by build_tm_cell)."""
        from dynameta.optics.fem_2d_tm import GriddedEps2D

        z = self._z_nm()
        if abs(z[-1] - thickness_nm) > 1e-6 * thickness_nm:
            raise ValueError(
                "{}: HydroConductor2D z samples span {:.6g} nm but the layer is {:.6g} nm thick "
                "(z_m is relative to the layer bottom; check the length unit)".format(
                    where, z[-1], thickness_nm))
        omega = 2.0 * math.pi * C_LIGHT / float(lambda_m)
        n = self._n2d()
        lays = self._point_layers(n)
        eps_t = np.array([eps_transverse(omega, L) for L in lays], dtype=np.complex128)
        eps_t = eps_t.reshape(n.shape)
        x_m = None
        if n.shape[1] > 1:
            # x_m is not None here: __post_init__ refuses an (Nz, Nx) density without it and the
            # dataclass is frozen
            x_m = check_periodic_axis(np.asarray(self.x_m, dtype=float).ravel(), period_nm * _L0,
                                      where="{} HydroConductor2D.x_m".format(where))
        nonlocal_ = not self.is_local_limit
        # ---- z lines -------------------------------------------------------------------------
        kmax = None
        if nonlocal_:
            kmax = max(abs(cmath.sqrt(kL_squared(omega, L))) for L in lays) * _L0     # 1/nm
        if self.h_z_m is not None:
            hz = float(self.h_z_m) * _S
        elif kmax:
            hz = float(self.h_per_screening) / kmax
        else:
            hz = thickness_nm / 8.0
        rows = max(2, int(math.ceil(thickness_nm / hz - 1e-9)))
        if not nonlocal_ and self.h_z_m is None:
            # exact local limit: the lines the local solver puts in a GriddedEps2D layer (its
            # samples), so the mesh is the one of the local solve of the same eps_T profile
            rows, zl = z.size - 1, list(z)
        else:
            zl = list(np.linspace(0.0, thickness_nm, rows + 1))
        if self.align_samples and z.size <= max_grid_lines:
            zl += list(z)
        h1 = thickness_nm / rows
        for lv in range(1, int(self.face_levels) + 1):
            d = h1 * float(self.face_ratio) ** lv
            zl += [d, thickness_nm - d]
        zl = _dedupe(zl, 1e-6 * max(h1 * float(self.face_ratio) ** max(int(self.face_levels), 0),
                                    1e-12) + 1e-9)
        # ---- x lines -------------------------------------------------------------------------
        xl: List[float] = []
        if self.h_x_m is not None:
            nxl = max(1, int(math.ceil(period_nm / (float(self.h_x_m) * _S) - 1e-9)))
            xl += list(np.linspace(0.0, period_nm, nxl + 1))
        if n.shape[1] > 1:
            xu, _zu = self._voxel(period_nm, max_grid_lines, where)[:2]
            if xu.size <= max_grid_lines:
                xl += list(xu)
        operator_eps = complex(float(self.eps_inf)) if nonlocal_ else GriddedEps2D(
            values=eps_t if n.shape[1] > 1 else eps_t[:, 0], z_m=z * _L0, x_m=x_m)
        return {"operator_eps": operator_eps, "local_eps_samples": eps_t, "nonlocal": nonlocal_,
                "z_lines_nm": zl, "x_lines_nm": sorted(set(xl)), "h_z_nm": hz, "rows": rows,
                "kL_max_per_nm": kmax}

    def _voxel(self, period_nm, max_grid_lines, where):
        """Uniform (x, z) resample of the density for the laterally structured case: z on
        coef_points uniform points (the piecewise-linear profile evaluated exactly there), x by
        fem_2d_tm's periodic resample. Returns (x_u, z_u [nm], n (Nz_u, Nx_u))."""
        from dynameta.optics.fem_2d_tm import GriddedEps2D, _RegionEps
        n = self._n2d()
        z = self._z_nm()
        zf = np.linspace(0.0, float(z[-1]), int(min(max(self.coef_points, z.size), 4096)))
        nf = np.stack([np.interp(zf, z, n[:, j]) for j in range(n.shape[1])], axis=1)
        reg = _RegionEps(GriddedEps2D(values=nf.astype(np.complex128), z_m=zf * _L0,
                                      x_m=np.asarray(self.x_m, dtype=float).ravel()),
                         where, thickness_nm=float(z[-1]), period_nm=period_nm)
        xu, zu, comps = reg.resampled(period_nm, max_grid_lines)
        return xu, zu, np.ascontiguousarray(comps[0].real)


def _dedupe(vals, tol):
    v = np.unique(np.round(np.asarray(vals, dtype=float), 12))
    keep = [v[0]]
    for a in v[1:]:
        if a - keep[-1] > tol:
            keep.append(a)
    return keep


def hydro_conductor_from_profile(z_m, n_m3, *, m_eff_kg: float, eps_inf: float, gamma: float,
                                 n_layers: Optional[int] = None, convention: str = "high_freq",
                                 D: float = 0.0, conductor_id: Hashable = "graded",
                                 **options) -> HydroConductor2D:
    """The 2-D route's twin of nonlocal_tmm.hydro_layers_from_profile: SAME profile arguments
    (``z_m[0]`` is the face nearest the superstrate, z increasing into the stack; ``n_m3`` > 0 on
    that grid), returning a HydroConductor2D for a TMLayer of thickness ``z_m[-1] - z_m[0]``. The
    density is used as the continuous piecewise-linear profile (the limit the planar staircase
    converges to as ``n_layers`` grows), so ``n_layers`` -- the planar staircase knob -- must be
    None here; the 2-D resolution knobs are the mesh options (``h_z_m``, ``h_per_screening``,
    ``face_levels``, ...), passed through ``options`` together with any other HydroConductor2D
    field (``beta_scale``, ``check_validity``, ``T_K``)."""
    if n_layers is not None:
        raise ValueError("hydro_conductor_from_profile: n_layers is the planar staircase knob; the "
                         "2-D route uses the continuous piecewise-linear profile (its resolution "
                         "knobs are h_z_m / h_per_screening / face_levels)")
    z = np.asarray(z_m, dtype=float)
    nn = np.asarray(n_m3, dtype=float)
    if z.ndim != 1 or nn.shape != z.shape or z.size < 2:
        raise ValueError("z_m and n_m3 must be 1-D arrays of the same length >= 2")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z_m must be strictly increasing")
    if conductor_id is None:
        raise ValueError("conductor_id must not be None")
    zb = z[-1] - z[::-1]                                        # from the layer bottom
    return HydroConductor2D(n0_m3=nn[::-1].copy(), z_m=zb, m_eff_kg=float(m_eff_kg),
                            eps_inf=float(eps_inf), gamma=float(gamma), convention=convention,
                            D=float(D), conductor_id=conductor_id, **options)


# ================================================================================================
# the extension (assembled by fem_2d_tm.solve_tm_cell on its product space)
# ================================================================================================

class _HydroConductorExtension:
    """(X, phi) on one conductor region; see the module docstring for the weak form."""

    condense = True          # X is element-local (L2): solve_tm_cell condenses it out

    def __init__(self, cell, region: str):
        info = cell.regions[region]
        self.region = region
        self.spec: HydroConductor2D = info.eps.hydro
        self.z_lo = float(info.z_lo_nm)
        self.t_nm = float(info.z_hi_nm - info.z_lo_nm)
        self.omega = 2.0 * math.pi * C_LIGHT / float(cell.lambda_m)
        self._cf = None
        self._ndof = None

    # ---- coefficients -------------------------------------------------------------------------
    def _coefficients(self, ctx):
        """(A1, kappa2) CoefficientFunctions: A1 = 1/Delta + 1/eps_inf, kappa2 = -L0**2/(eps0 c),
        evaluated by the planar kernel on CoefficientFunction-valued wp and beta."""
        if self._cf is not None:
            return self._cf
        ng, sp = ctx.ng, self.spec
        n = sp._n2d()
        z = sp._z_nm()
        if n.shape[1] == 1:
            zc = np.union1d(z, np.linspace(0.0, z[-1], int(sp.coef_points)))
            nc = np.interp(zc, z, n[:, 0])
            wp2, beta = sp._wp2_beta(nc)
            knots = [float(self.z_lo + zc[0])] + [float(self.z_lo + v) for v in zc] + [
                float(self.z_lo + zc[-1])]
            wp2_cf = ng.BSpline(2, knots, [float(v) for v in wp2])(ng.y)
            b2_cf = ng.BSpline(2, knots, [float(v) for v in beta * beta])(ng.y)
        else:
            P = ctx.cell.period_nm
            xu, zu, nv = sp._voxel(P, ctx.cell.mesh_spec.max_grid_lines, self.region)
            wp2, beta = sp._wp2_beta(nv.ravel())
            start, end = (0.0, self.z_lo + float(zu[0])), (float(P), self.z_lo + float(zu[-1]))
            wp2_cf = ng.VoxelCoefficient(start, end, np.ascontiguousarray(wp2.reshape(nv.shape)),
                                         linear=True)
            b2_cf = ng.VoxelCoefficient(start, end,
                                        np.ascontiguousarray((beta * beta).reshape(nv.shape)),
                                        linear=True)
        lay = HydroLayer(float(sp.eps_inf), ng.sqrt(wp2_cf), float(sp.gamma), ng.sqrt(b2_cf),
                         self.t_nm * _L0, D=float(sp.D))
        eps_t = eps_transverse(self.omega, lay)
        delta = eps_t - float(sp.eps_inf)
        a1 = 1.0 / delta + 1.0 / float(sp.eps_inf)
        c = hydro_psi_per_charge(self.omega, lay)
        kappa2 = -(_L0 * _L0) / (EPS0 * c)
        self._cf = (a1, kappa2)
        return self._cf

    # ---- the extension protocol ---------------------------------------------------------------
    def space(self, ctx):
        ng = ctx.ng
        mesh = ctx.cell.mesh
        on = mesh.Materials(self.region)
        phase = cmath.exp(1j * ctx.kx * ctx.cell.period_nm)
        V = ng.VectorL2(mesh, order=ctx.order, complex=True, definedon=on)
        Q = ng.Periodic(ng.H1(mesh, order=ctx.order, complex=True, definedon=on), phase=[phase])
        # A subdomain H1 space still carries DOFs on the cell's left/right BOUNDARY segments outside
        # the conductor (boundary elements are not restricted by a volume `definedon`); nothing
        # assembles there, so they would make the matrix singular. Keep free only the DOFs of the
        # conductor's own elements (FreeDofs() is a reference; the product space reads it).
        used = set()
        for el in mesh.Elements(ng.VOL):
            if el.mat == self.region:
                used.update(Q.GetDofNrs(el))
        fd = Q.FreeDofs()
        for i in range(Q.ndof):
            if i not in used:
                fd.Clear(i)
        self._ndof = (V, Q, len(used))
        return (V, Q)

    def add_terms(self, ctx, a, f, h, hv, trial, test):
        ng = ctx.ng
        X, q = trial
        K, w = test
        dxr = ctx.dx(self.region)
        a1, kappa2 = self._coefficients(ctx)
        ie = 1.0 / float(self.spec.eps_inf)
        gq, gw = ng.grad(q), ng.grad(w)
        jgh = ctx.inplane_curl(ng.grad(h))
        jgv = ctx.inplane_curl(ng.grad(hv))
        jgb = ctx.inplane_curl(ctx.grad_H_bg)
        # (M): - (1/eps_inf) X . J grad v
        a += (-ie * (X[0] * jgv[0] + X[1] * jgv[1])) * dxr
        # (K): (1/Delta + 1/eps_inf) X . K - (1/eps_inf) J grad H . K + grad phi . K
        a += (a1 * (X[0] * K[0] + X[1] * K[1]) - ie * (jgh[0] * K[0] + jgh[1] * K[1])
              + gq[0] * K[0] + gq[1] * K[1]) * dxr
        f += (ie * (jgb[0] * K[0] + jgb[1] * K[1])) * dxr
        # (W): - kappa**2 phi w + X . grad w   (J.n = 0 on the conductor boundary is natural)
        a += (-kappa2 * q * w + X[0] * gw[0] + X[1] * gw[1]) * dxr

    def _norm(self, ctx):
        e1 = complex(ctx.cell.n_super) ** 2
        kz0 = cmath.sqrt(e1 * ctx.k0 ** 2 - ctx.kx ** 2)
        return ctx.cell.period_nm * (kz0 / e1).real

    def absorption(self, ctx, comps, grad_h_tot):
        """(1/2) Re(conj(J) . E) over the conductor, in the solver's normalisation (the background
        eps_inf is real, so the local loss term of this region is zero and this is the whole
        absorption, Drude damping and GNOR diffusion included)."""
        ng = ctx.ng
        X = comps[0]
        jg = ctx.inplane_curl(grad_h_tot)
        ie = 1.0 / float(self.spec.eps_inf)
        # omega eps0 E = i (J grad H - X) / eps_inf;  2 omega eps0 * (1/2) Re(conj(X) . E)
        q = ng.Conj(X[0]) * (jg[0] - X[0]) + ng.Conj(X[1]) * (jg[1] - X[1])
        dens = 1j * ie * q
        with ng.TaskManager():
            val = ng.Integrate(dens, ctx.cell.mesh, definedon=ctx.cell.mesh.Materials(self.region),
                               order=2 * ctx.order + 2)
        self._absorbed = float(complex(val).real) / self._norm(ctx)
        return {self.region: self._absorbed}

    def diagnostics(self, ctx, comps):
        sp = self.spec
        rep = sp.validity(self.omega)
        info = sp.region_setup(lambda_m=ctx.cell.lambda_m, thickness_nm=self.t_nm,
                               period_nm=ctx.cell.period_nm,
                               max_grid_lines=ctx.cell.mesh_spec.max_grid_lines, where=self.region)
        u = [v for v in rep.u if v == v]
        out = {"validity": rep, "u_max": max(u) if u else float("nan"),
               "h_z_nm": info["h_z_nm"], "rows": info["rows"],
               "kL_max_per_nm": info["kL_max_per_nm"],
               "ndof_current": int(self._ndof[0].ndof), "ndof_potential": int(self._ndof[2]),
               "absorption": getattr(self, "_absorbed", None), "conductor_id": sp.conductor_id}
        if sp.check_validity and not rep.ok:
            warnings.warn("fem_2d_tm (hydrodynamic layer {!r}): {} (pass check_validity=False to "
                          "silence; the report is in TMSolution.diagnostics['hydro'])".format(
                              self.region, rep.message), UnreliableResultWarning, stacklevel=4)
        return {"hydro": {self.region: out}}


def hydro_extensions_for_cell(cell) -> list:
    """The hydrodynamic extensions of every nonlocal conductor layer of a built TMCell (called by
    fem_2d_tm.solve_tm_cell; exposed for callers assembling their own product spaces)."""
    return [_HydroConductorExtension(cell, nm) for nm, r in cell.regions.items()
            if r.eps.hydro is not None]


# ================================================================================================
# convergence helper
# ================================================================================================

def hydro_z_convergence(period_x_m: float, layers: Sequence[Any], *,
                        h_factors: Sequence[float] = (2.0, 1.0, 0.5, 0.25),
                        orders: Sequence[int] = (4,), **solve_kwargs) -> List[dict]:
    """Solve the cell for every (order, h_factor) and tabulate the zeroth-order r.

    Every HydroConductor2D layer's conductor mesh is scaled by h_factor (h_z_m if it was set,
    else h_per_screening), the optical mesh is left alone. solve_kwargs go to
    fem_2d_tm.solve_fem_2d_tm (lambda_m, theta_deg, n_super, n_sub, polarization, mesh_spec, ...).
    Rows (in order of `orders`, then `h_factors`): {'order', 'h_factor', 'r', 'R',
    'energy_residual', 'ndof', 'time_s', 'd_prev' (|r - previous row of the same order|),
    'd_last' (|r - the last row overall|), 'rate' (observed order in h from three successive
    rows of one order, None if not available), 'diagnostics'}."""
    from dynameta.optics.fem_2d_tm import TMLayer, solve_fem_2d_tm

    def scaled(fac):
        out = []
        for L in layers:
            sp = L.eps
            if getattr(type(sp), "_TM_HYDRO_CONDUCTOR", False):
                if sp.h_z_m is not None:
                    sp = sp.with_options(h_z_m=float(sp.h_z_m) * fac)
                else:
                    sp = sp.with_options(h_per_screening=float(sp.h_per_screening) * fac)
                L = TMLayer(L.name, L.thickness_m, sp, L.blocks, L.maxh_z_m)
            out.append(L)
        return out

    rows = []
    for order in orders:
        for fac in h_factors:
            t0 = time.perf_counter()
            sol = solve_fem_2d_tm(period_x_m, scaled(fac), order=order, return_solution=True,
                                  **solve_kwargs)
            rows.append({"order": int(order), "h_factor": float(fac), "r": complex(sol.result.r),
                         "R": float(sol.result.R), "energy_residual": float(sol.energy_residual),
                         "ndof": int(sol.ndof), "time_s": time.perf_counter() - t0,
                         "diagnostics": sol.diagnostics})
    for k, row in enumerate(rows):
        same = [r_ for r_ in rows[:k] if r_["order"] == row["order"]]
        row["d_prev"] = abs(row["r"] - same[-1]["r"]) if same else None
        row["d_last"] = abs(row["r"] - rows[-1]["r"])
        row["rate"] = None
        if len(same) >= 2:
            d1 = abs(same[-1]["r"] - same[-2]["r"])
            d2 = abs(row["r"] - same[-1]["r"])
            ratio = same[-2]["h_factor"] / same[-1]["h_factor"]
            if d1 > 0 and d2 > 0 and ratio > 1:
                row["rate"] = math.log(d1 / d2) / math.log(ratio)
    return rows
