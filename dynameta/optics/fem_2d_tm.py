"""
2-D periodic TM finite-element solver (local response): a 1-D-periodic layered cell, invariant
along y, lit by a TM plane wave in the x-z plane. Returns the zeroth-order complex reflection /
transmission, the order-summed R/T and a per-region absorbed-power map.

WHY IT EXISTS. It is the independent referee for the modal (PMM) and Fourier (RCWA) routes on
lamellar cells such as metal strips over a gated thin film: it shares no code with either (no
Fourier factorization, no modal basis, no shared geometry translation), so agreement between the
three is evidence, not self-consistency.

FORMULATION (H_y scalar, chosen on purpose -- see "EXTENSION" below). TM means H = H_y(x, z) y-hat
and E = (E_x, E_z). With exp(-i omega t), curl H = -i omega eps0 eps E and curl E = i omega mu0 H
give, for the in-plane permittivity tensor eps = [[exx, exz], [ezx, ezz]] (x-z basis),

    div( Lambda grad H ) + k0^2 H = 0,      Lambda = eps^T / det(eps)            (1)

acting on grad = (d/dx, d/dz). For a DIAGONAL tensor Lambda = diag(1/ezz, 1/exx): the x-derivative
is weighted by 1/ezz and the z-derivative by 1/exx (E_z comes from dH/dx, E_x from dH/dz). The
weak form is  int Lambda grad H . grad v - k0^2 H v = 0  on a Bloch-periodic H1 space

    H(x + P, z) = exp(i kx P) H(x, z),      kx = n_super k0 sin(theta),

whose natural interface condition is continuity of the tangential E (Lambda grad H . n). The
in-plane E is a post-processed quantity: E = J Lambda grad H / (-i omega eps0), J(a, b) = (-b, a).

SCATTERED FIELD + PML. As in the 3-D solver (optics.solver), the unknown is the field scattered off
the analytic two-half-space (superstrate / substrate) Fresnel background H_bg; the source is
-int (Lambda - Lambda_bg) grad H_bg . grad v over the structure layers only. z is truncated by an
explicit stretched-coordinate PML (d/dz -> d/dz / s, dA -> s dA, s = 1 + i alpha xi^2) on both
sides, with alpha set from the requested round-trip attenuation of the slowest propagating order.

MESH. A structured tensor-product quadrilateral mesh: every layer interface and every lamellar
block edge is a mesh line, so thin (1-5 nm) layers cost one element row and no material boundary
cuts an element. Metal corners and high-contrast dielectric corners (permittivity ratio above
TMMeshSpec.corner_contrast) get geometric grading toward the corner in x and z (an hp-type
geometric mesh); metals are resolved to their skin depth in z; gridded (z-graded and laterally
structured) permittivity gets mesh lines at its samples.

r / t CONVENTIONS (the library's, matching the RCWA/PMM bridges and optics.tmm_reference):
  * exp(-i omega t); Im(eps) > 0 is loss.
  * r is referenced to the TOP interface of the stack (superstrate / top layer); t is the field at
    the BOTTOM interface (top of the substrate) over the incident field at the top interface.
  * polarization 'x' (normal incidence only): r = E_x(refl) / E_x(inc), the lab E_x tangential
    amplitude (r = -r_H, r_H the H_y reflection ratio). polarization 'p': the p-hat convention of
    optics.tmm_reference and the bridges (r = r_H, which equals -E_x(refl)/E_x(inc)). In both,
    t = t_H n_super / n_sub (the E amplitude ratio).
  * R and T are ORDER-SUMMED over the propagating diffraction orders (as the bridges return them;
    the 3-D FEM returns the specular order only). The zeroth-order amplitudes are exact Fourier
    projections of the field on two lines in each cladding (no aliasing: the projection is a
    Gauss quadrature over every element along the line), the second line separating the up- and
    down-going parts (the down-going part in the superstrate is the PML reflection, reported).
  * per_region_absorption: int Im(conj(Lambda grad H) . grad H) dA over each region, normalized by
    the incident power P Re(kz0 / eps_super); A_independent is its sum. R + T + A_independent = 1 is
    the energy check (energy_residual on the TMSolution).

LOSSY SUBSTRATE / MIRROR. A substrate in which EVERY diffraction order decays within the vacuum
wavelength (decay length 1/Im kz of the most slowly decaying order below lambda0; a strongly
absorbing substrate, or total internal reflection with no diffracted order propagating into the
substrate) is treated as opaque: t and T are None, the substrate buffer is sized to >= 15 decay
lengths of that order and its absorption is part of the budget (R + A = 1). A substrate into
which a diffracted order propagates is never opaque, even when order 0 is evanescent there
(grating-coupled transmission out of total internal reflection): T is the order sum. A weakly lossy
substrate is treated as transparent, with t referenced at its top interface; its T is the power
entering the substrate at the interface, summed over ALL orders (the buffer's absorption plus the
all-order flux into the PML, _lossy_substrate_flux), so R + T + A = 1 holds for a lossy diffracting
substrate too (T_orders lists the orders above the lossless cut-off at the interface; in a lossy
substrate their sum is below T). Before this rule a lossy substrate's T summed its order 0 only.

EXTENSION (the nonlocal follow-on). The local solver is split into reusable pieces -- geometry and
mesh (build_tm_cell -> TMCell), material coefficients (one _RegionEps per region, constant, tensor
or gridded), port/PML treatment (_pml_profile, _Background) and r/t extraction (_extract_orders) --
and the assembly accepts EXTENSION fields: extra unknowns living on a subdomain, coupled to H_y.
An extension is an object with
    space(ctx)                                  -> an NGSolve FESpace (e.g. defined on a conductor)
    add_terms(ctx, a, f, h, hv, x, xv)          -> adds its bilinear / linear terms
    absorption(ctx, gfx, grad_h_tot) [optional] -> {region: absorbed fraction} it owns
and the solver assembles them on a product space with H_y first (space(ctx) may also return a
tuple of spaces -- several coupled fields -- in which case add_terms / absorption receive tuples,
and an optional diagnostics(ctx, comps) -> dict hook feeds TMSolution.diagnostics). The test suite
exercises the hook with a local Drude current carried as an explicit unknown, which must reproduce
the local permittivity to round-off.

NONLOCAL (HYDRODYNAMIC) CONDUCTOR LAYERS. A layer whose eps is an
optics.fem_2d_tm_hydro.HydroConductor2D (a laterally uniform slab with equilibrium density n0(z)
or n0(x, z)) is solved with the hydrodynamic free-carrier current through that hook: build_tm_cell
gives the layer its own z mesh lines and the background eps_inf as the H_y operator, and
solve_tm_cell adds the extension (free current in L2, electrochemical potential psi in H1 of the
conductor; the hard wall J.n = 0 and the continuity of psi and J.n inside the conductor are
natural). See that module for the weak form, its validation and its limits. (An earlier draft of
this docstring proposed a beta^2 div-div current row; for a graded density its natural interface
condition, continuity of beta^2 n1, is the wrong one -- the continuous quantity is psi, i.e. the
div-div coefficient must be beta^2/wp^2.) Other hydrodynamic input -- HydroParams / HydroLayer /
HydroSlab records, finite-k Mermin models, hydrodynamic blocks or claddings -- is REFUSED with a
message naming the supported route, rather than silently solving the local limit.
"""

from __future__ import annotations

import cmath
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dynameta.core.interfaces import OpticalResult
from dynameta.core.lateral import check_periodic_axis, is_x_invariant, is_y_invariant, tensor_couples_y
from dynameta.core.units import NM_PER_M
# the ONE out-of-plane wavenumber kernel (principal branch; see resonance's BRANCH CHOICE note).
# Every _kz call here takes a CLADDING eps = n**2 of a passive end medium (the superstrate is
# refused unless lossless), so Im(eps) >= +0 -- CPython forms n**2 by multiplying up from 1+0j,
# which also turns a -0.0 imaginary part of n into +0.0 -- and for such arguments the principal
# root IS the passive / outgoing branch (both parts >= 0), bit for bit. (It differs from the
# Im >= 0 flip only for Im(eps) < 0, a gain substrate, where the principal branch keeps
# Re(kz) > 0: the outgoing wave.)
from dynameta.optics.resonance import _kz
from dynameta.warning_types import AdvisoryWarning, UnreliableResultWarning

__all__ = [
    "GriddedEps2D", "TMBlock", "TMLayer", "TMMeshSpec", "TMCell", "TMSolution",
    "build_tm_cell", "solve_tm_cell", "solve_fem_2d_tm", "design_to_tm_layers",
    "solve_design_2d_tm", "make_fem_2d_tm_optical_solver", "tm_convergence_study",
]

S = NM_PER_M                                    # m -> nm (mesh coordinates are nm)
_RESERVED = ("superstrate", "substrate", "pml_top", "pml_bot")
_REL = 1e-9


# ================================================================================================
# material input
# ================================================================================================

_NONLOCAL_GAP = (
    "the 2-D TM FEM takes a nonlocal (hydrodynamic) free-carrier response only as a conductor "
    "LAYER given by optics.fem_2d_tm_hydro.HydroConductor2D (a laterally uniform slab whose "
    "equilibrium density may vary in z and x; see hydro_conductor_from_profile). HydroParams / "
    "HydroLayer / HydroSlab records, finite-k Mermin models, and hydrodynamic BLOCKS or claddings "
    "are not accepted: pass the local permittivity, build a HydroConductor2D layer, or use "
    "optics.nonlocal_tmm / optics.hydro_fem for the planar / isolated nonlocal cases.")


def _is_hydro_conductor(obj) -> bool:
    """True for the hydrodynamic conductor-layer spec of optics.fem_2d_tm_hydro (duck-typed so
    this module does not import it)."""
    return bool(getattr(type(obj), "_TM_HYDRO_CONDUCTOR", False))


def _refuse_nonlocal(obj, where: str) -> None:
    """RAISE on hydrodynamic (nonlocal) material input: refusing beats silently solving the local
    limit of a material the caller asked to be nonlocal. (A HydroConductor2D used as a LAYER
    background never reaches here: build_tm_cell routes it to the hydrodynamic extension.)"""
    if _is_hydro_conductor(obj):
        raise NotImplementedError("{}: a HydroConductor2D is accepted only as a layer's background "
                                  "eps (a laterally uniform conductor slab), not as a block or "
                                  "cladding; {}".format(where, _NONLOCAL_GAP))
    cls = type(obj).__name__
    nonlocal_cls = cls in ("HydroParams", "HydroLayer", "HydroSlab")
    duck = hasattr(obj, "beta") and hasattr(obj, "wp")
    mermin_k = cls == "MerminDrudeOptical" and float(getattr(obj, "k_per_m", 0.0) or 0.0) != 0.0
    if nonlocal_cls or duck or mermin_k:
        raise NotImplementedError("{}: got hydrodynamic/nonlocal material input ({}); {}".format(
            where, cls, _NONLOCAL_GAP))


@dataclass(frozen=True)
class GriddedEps2D:
    """A gridded permittivity for ONE layer (or block): z-graded and optionally laterally
    structured.

    values : complex, shape (Nz,) or (Nz, 1) for a laterally uniform profile, (Nz, Nx) for a
             laterally structured one, or (..., 2, 2) for an in-plane tensor [[xx, xz], [zx, zz]]
             (x-z basis).
    z_m    : sample positions RELATIVE TO THE LAYER BOTTOM [m], strictly monotonic, spanning
             [0, thickness] (checked against the layer; a mismatch raises).
    x_m    : lateral sample positions in the cell [m] (inside [0, period]); None / Nx == 1 for a
             laterally uniform profile. The data is treated as periodic in x.

    Interpolation is linear in z (and bilinear in (x, z)). A laterally uniform profile is used on
    its native (possibly non-uniform) z samples exactly; a laterally structured one is resampled
    onto a uniform node grid (NGSolve VoxelCoefficient assumes uniform axes) with the finest
    native spacing."""
    values: Any
    z_m: Any
    x_m: Any = None

    @staticmethod
    def uniaxial(exx, ezz, z_m, x_m=None) -> "GriddedEps2D":
        """Diagonal in-plane tensor from two sampled components of equal shape."""
        a, b = np.asarray(exx, dtype=np.complex128), np.asarray(ezz, dtype=np.complex128)
        if a.shape != b.shape:
            raise ValueError("GriddedEps2D.uniaxial: exx and ezz shapes differ")
        t = np.zeros(a.shape + (2, 2), dtype=np.complex128)
        t[..., 0, 0], t[..., 1, 1] = a, b
        return GriddedEps2D(values=t, z_m=z_m, x_m=x_m)


@dataclass(frozen=True)
class TMBlock:
    """A lamellar block of a layer: [x_lo_m, x_hi_m] (cell coordinates; a block crossing x = 0 or
    x = period wraps periodically) filled with `eps` (complex, (exx, ezz), a 2x2 array or a
    GriddedEps2D). `name` labels its absorption region (default '<layer>__blk<j>')."""
    x_lo_m: float
    x_hi_m: float
    eps: Any
    name: Optional[str] = None


@dataclass(frozen=True)
class TMLayer:
    """One layer of the cell (layers are listed BOTTOM FIRST, substrate side first, as in
    dynameta.geometry.Stack). `eps` is the background (complex, (exx, ezz), 2x2 array or
    GriddedEps2D); `blocks` are lamellar inclusions; `maxh_z_m` caps the element height in this
    layer (e.g. to resolve a graded film)."""
    name: str
    thickness_m: float
    eps: Any
    blocks: Tuple[TMBlock, ...] = ()
    maxh_z_m: Optional[float] = None


@dataclass(frozen=True)
class TMMeshSpec:
    """Mesh / truncation controls. Every length None means 'automatic'.

    maxh_m             : element size cap in dielectrics (auto: lambda0 / (8 n_max)).
    metal_h_per_skin   : element size in a metal as a multiple of its skin depth 1/(k0 Im n).
    h_scale            : uniform refinement factor applied to every automatic size (<1 = finer).
    corner_levels      : geometric grading levels toward each refined corner (0 disables).
    corner_ratio       : geometric grading ratio (cell size shrinks by this per level).
    corner_mode        : 'contrast' (default: the 'metal' corners plus every corner where a
                         permittivity jump across the block edge meets one across the layer face by
                         more than corner_contrast; see build_tm_cell), 'metal' (only corners where
                         the metal / non-metal boundary bends: Re eps < 0 in one, three or two
                         diagonal quadrants, each region judged at the corner itself, so a film
                         graded to Re eps < 0 near one face and cut into lateral blocks is not a
                         corner at every block edge), 'all' (every block corner), or 'none'.
    corner_contrast    : the permittivity ratio above which corner_mode 'contrast' grades a
                         dielectric corner. MEASURED (order 4; rib of eps_r in air on n = 1.45,
                         P 1600 / W 660 / H 220 nm, 1250 nm, normal incidence; |dr| against order 7,
                         'all', 8 levels, itself <= 3.5e-10 from order 6): ungraded, |dr| = 8.7e-7 /
                         4.6e-6 / 7.4e-6 / 3.2e-5 / 5.0e-5 at eps_r = 1.44 / 2.0 / 2.25 / 4 / 11.56;
                         graded, 2.3e-8 / 4.8e-8 / 5.8e-8 / 2.1e-8 / 2.7e-8 at 3.3 / 3.3 / 3.3 / 2.7 /
                         2.0 x the ndof. An eps 4 / 2.25 grating (contrast 1.78, ungraded) under an
                         n = 1.5 superstrate at 60 deg is 2.1e-5 off (6.4e-9 graded, 3x the ndof).
                         Grading pays most where the error is largest; at a contrast <= 2 the
                         measured ungraded error stays below the metal-strip default's 2.9e-5. 'all'
                         also grades the near-uniform block edges of a laterally structured film:
                         on a 31-block z-graded film under a metal strip it raises the element count
                         from 7695 to 49434 and the order-4 solve from 0.14 GB / 2.8 s to over
                         11 GB, with the 'contrast' mesh identical to the 'metal' one there.
    min_layer_elements : element rows per layer at least.
    superstrate_buffer_m / substrate_buffer_m : cladding buffers between stack and PML (auto:
                         max(lambda0/(2n), 8 decay lengths of the least-evanescent order)).
    pml_thickness_m    : PML thickness (auto: lambda0 / n of the cladding).
    pml_elements       : element rows across each PML.
    pml_attenuation    : requested round-trip amplitude attenuation of the slowest propagating
                         order through the PML.
    max_grid_lines     : cap on mesh lines inserted per axis for a laterally structured grid."""
    maxh_m: Optional[float] = None
    metal_h_per_skin: float = 1.0
    h_scale: float = 1.0
    corner_levels: int = 6
    corner_ratio: float = 0.25
    corner_mode: str = "contrast"
    min_layer_elements: int = 1
    superstrate_buffer_m: Optional[float] = None
    substrate_buffer_m: Optional[float] = None
    pml_thickness_m: Optional[float] = None
    pml_elements: int = 10
    pml_attenuation: float = 1e-10
    max_grid_lines: int = 256
    corner_contrast: float = 2.0


def _eps_tensor(spec, where):
    """complex / (exx, ezz) / 2x2 -> (exx, exz, ezx, ezz) complex constants."""
    if isinstance(spec, (complex, float, int, np.number)):
        e = complex(spec)
        return (e, 0j, 0j, e)
    arr = np.asarray(spec, dtype=np.complex128)
    if arr.shape == (2,):
        return (complex(arr[0]), 0j, 0j, complex(arr[1]))
    if arr.shape == (2, 2):
        return (complex(arr[0, 0]), complex(arr[0, 1]), complex(arr[1, 0]), complex(arr[1, 1]))
    raise TypeError("{}: eps must be a complex scalar, (exx, ezz), a 2x2 x-z tensor or a "
                    "GriddedEps2D; got {!r}".format(where, spec))


def _xz_embed(comps):
    """The four x-z components [(Nz, Nx)] as a (Nz, 1, Nx, 3, 3) tensor (y row / column zero), the
    layout core.lateral tests."""
    xx, xz, zx, zz = (np.asarray(c, dtype=np.complex128) for c in comps)
    t = np.zeros((xx.shape[0], 1, xx.shape[1], 3, 3), dtype=np.complex128)
    t[:, 0, :, 0, 0], t[:, 0, :, 0, 2], t[:, 0, :, 2, 0], t[:, 0, :, 2, 2] = xx, xz, zx, zz
    return t


class _RegionEps:
    """The permittivity of one mesh region: constant (scalar / tensor) or gridded. Builds the
    NGSolve coefficient functions and the scales the mesher needs."""

    def __init__(self, spec, where, *, thickness_nm=None, period_nm=None):
        _refuse_nonlocal(spec, where)
        self.where = where
        self.grid = None
        self.hydro = None                       # HydroConductor2D of a hydrodynamic layer
        self.lines_override = None              # (x lines, z lines rel. to the layer bottom) [nm]
        if isinstance(spec, GriddedEps2D):
            self._init_grid(spec, where, thickness_nm, period_nm)
            comps = self.grid["comps"]
            samples = np.concatenate([np.ravel(c) for c in (comps[0], comps[3])])
            self.const = None
        else:
            self.const = _eps_tensor(spec, where)
            samples = np.array([self.const[0], self.const[3]], dtype=np.complex128)
        if not np.all(np.isfinite(samples)):
            raise ValueError("{}: non-finite permittivity".format(where))
        n = np.sqrt(samples.astype(np.complex128))
        n = np.where(n.imag < 0, -n, n)
        self.n_max = float(np.max(np.abs(n.real)))
        self.kappa_max = float(np.max(np.abs(n.imag)))
        self.is_metal = bool(np.any(samples.real < 0.0))
        self.isotropic = (self.grid is None and self.const[1] == 0 and self.const[2] == 0
                          and self.const[0] == self.const[3])

    def metal_at_face(self, face: str, x_nm: float, period_nm: float) -> bool:
        """Re eps < 0 (either diagonal component) of this region at the point of its `face` ('bottom' or
        'top') nearest the lateral position x_nm: what the corner rule of build_tm_cell asks. A constant
        region answers is_metal; a gridded one its face row at the nearest x sample (periodic), so a film
        that is metallic only near one face is metallic at that face only. A nonlocal hydrodynamic region
        (whose operator permittivity is the background) answers is_metal, the conservative choice."""
        if self.grid is None or self.hydro is not None:
            return self.is_metal
        row = 0 if face == "bottom" else -1
        x = self.grid["x"]
        col = 0
        if x is not None:
            d = np.abs((np.asarray(x) - float(x_nm) + 0.5 * period_nm) % period_nm - 0.5 * period_nm)
            col = int(np.argmin(d))
        c = self.grid["comps"]
        return bool(c[0][row, col].real < 0.0 or c[3][row, col].real < 0.0)

    def eps_at_face(self, face: str, x_nm: float, period_nm: float) -> Tuple[complex, complex]:
        """(exx, ezz) of this region at the point of its `face` ('bottom' or 'top') nearest the
        lateral position x_nm, sampled as metal_at_face samples it (a constant region: its value; a
        gridded one: its face row at the nearest x sample). Feeds the permittivity-contrast corner
        rule of build_tm_cell."""
        if self.grid is None:
            return (self.const[0], self.const[3])
        row = 0 if face == "bottom" else -1
        x = self.grid["x"]
        col = 0
        if x is not None:
            d = np.abs((np.asarray(x) - float(x_nm) + 0.5 * period_nm) % period_nm - 0.5 * period_nm)
            col = int(np.argmin(d))
        c = self.grid["comps"]
        return (complex(c[0][row, col]), complex(c[3][row, col]))

    def _init_grid(self, g, where, thickness_nm, period_nm):
        v = np.asarray(g.values, dtype=np.complex128)
        z = np.asarray(g.z_m, dtype=np.float64).ravel() * S
        tensor = v.ndim >= 2 and v.shape[-2:] == (2, 2) and v.ndim in (3, 4)
        base = v[..., 0, 0] if tensor else v
        if base.ndim == 1:
            base_shape = (base.shape[0], 1)
        elif base.ndim == 2:
            base_shape = base.shape
        else:
            raise ValueError("{}: GriddedEps2D.values must be (Nz,), (Nz, Nx) or (..., 2, 2); got "
                             "shape {}".format(where, v.shape))
        if tensor:
            comps = [v[..., i, j].reshape(base_shape) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
        else:
            b = base.reshape(base_shape)
            comps = [b, np.zeros_like(b), np.zeros_like(b), b]
        nz, nx = base_shape
        if z.size != nz or nz < 2:
            raise ValueError("{}: GriddedEps2D needs >= 2 z samples matching values (got z {} vs "
                             "values {})".format(where, z.size, nz))
        dz = np.diff(z)
        if not (np.all(dz > 0) or np.all(dz < 0)):
            raise ValueError("{}: GriddedEps2D.z_m must be strictly monotonic".format(where))
        if z[0] > z[-1]:
            z = z[::-1]
            comps = [c[::-1] for c in comps]
        if thickness_nm is not None and (abs(z[0]) > 1e-6 * thickness_nm
                                         or abs(z[-1] - thickness_nm) > 1e-6 * thickness_nm):
            raise ValueError(
                "{}: GriddedEps2D z samples span [{:.6g}, {:.6g}] nm but the layer is {:.6g} nm "
                "thick (z_m is relative to the layer bottom and must span exactly [0, thickness]; "
                "check the length unit).".format(where, z[0], z[-1], thickness_nm))
        x = None
        if nx > 1 and is_x_invariant(_xz_embed(comps)):
            # identical lateral columns ARE a laterally uniform profile (the bridge delivers every
            # carrier film with 256 of them). It goes the native-z path the class docstring promises;
            # it used to be resampled at the finest native spacing as a structured voxel grid,
            # overflowed max_grid_lines and lost its mesh lines. The test is core.lateral's, shared
            # by every route.
            comps = [c[:, :1] for c in comps]
            nx = 1
        if nx > 1:
            if g.x_m is None:
                raise ValueError("{}: laterally structured GriddedEps2D needs x_m".format(where))
            xr = np.asarray(g.x_m, dtype=np.float64).ravel()
            if xr.size != nx or np.any(np.diff(xr) <= 0):
                raise ValueError("{}: GriddedEps2D.x_m must be strictly increasing with Nx "
                                 "entries".format(where))
            if period_nm is not None:
                # the lateral unit / span guard every route shares (a metre
                # axis read as nm, or one covering a fraction of the cell, used to give the
                # uniform-film answer silently)
                x = check_periodic_axis(xr, period_nm / S, where=where + " GriddedEps2D.x_m") * S
            else:
                x = xr * S
        self.grid = {"z": z, "x": x, "comps": comps}

    # ---- uniform resample for the laterally structured case ------------------------------------
    def voxel_axes(self, period_nm, max_lines):
        """Uniform node grids (x_u, z_u) a laterally structured grid is resampled onto."""
        z, x = self.grid["z"], self.grid["x"]
        dzm = float(np.min(np.diff(z)))
        nzu = int(min(max(2, round((z[-1] - z[0]) / dzm) + 1), 4096))
        if np.allclose(np.diff(z), dzm, rtol=1e-9, atol=0):
            zu = z.copy()
        else:
            zu = np.linspace(z[0], z[-1], nzu)
        xs = np.concatenate([x, [x[0] + period_nm]])
        dxm = float(np.min(np.diff(xs)))
        nxu = int(min(max(2, round(period_nm / dxm) + 1), 4096))
        xu = np.linspace(0.0, period_nm, nxu)
        return xu, zu

    def resampled(self, period_nm, max_lines):
        """(x_u, z_u, [4 x (Nz_u, Nx_u) complex]) bilinear periodic resample."""
        xu, zu = self.voxel_axes(period_nm, max_lines)
        z, x = self.grid["z"], self.grid["x"]
        out = []
        for c in self.grid["comps"]:
            # x: periodic linear interpolation; z: linear
            xe = np.concatenate([x[-1:] - period_nm, x, x[:1] + period_nm])
            ce = np.concatenate([c[:, -1:], c, c[:, :1]], axis=1)
            cx = np.empty((c.shape[0], xu.size), dtype=np.complex128)
            for i in range(c.shape[0]):
                cx[i] = np.interp(xu, xe, ce[i].real) + 1j * np.interp(xu, xe, ce[i].imag)
            cz = np.empty((zu.size, xu.size), dtype=np.complex128)
            for j in range(xu.size):
                cz[:, j] = np.interp(zu, z, cx[:, j].real) + 1j * np.interp(zu, z, cx[:, j].imag)
            out.append(np.ascontiguousarray(cz, dtype=np.complex128))
        return xu, zu, out

    def mesh_lines(self, z_lo_nm, period_nm, max_lines):
        """(x lines, z lines) [nm] the mesh should carry so the coefficient is smooth per element."""
        if self.lines_override is not None:
            xl, zl = self.lines_override
            return list(xl), [z_lo_nm + v for v in zl]
        if self.grid is None:
            return [], []
        if self.grid["x"] is None:
            return [], list(z_lo_nm + self.grid["z"])
        xu, zu = self.voxel_axes(period_nm, max_lines)
        xl = list(xu) if xu.size <= max_lines else []
        zl = list(z_lo_nm + zu) if zu.size <= max_lines else []
        return xl, zl

    def components_cf(self, ng, z_lo_nm, period_nm, max_lines):
        """(exx, exz, ezx, ezz) as NGSolve CoefficientFunctions (or complex constants)."""
        if self.grid is None:
            return tuple(ng.CoefficientFunction(c) for c in self.const)
        if self.grid["x"] is None:
            zk = list(z_lo_nm + self.grid["z"])
            knots = [zk[0]] + zk + [zk[-1]]
            out = []
            for c in self.grid["comps"]:
                cc = c[:, 0]
                if not np.any(cc != 0):
                    out.append(ng.CoefficientFunction(0j))
                    continue
                re = ng.BSpline(2, knots, [float(v) for v in cc.real])(ng.y)
                im = ng.BSpline(2, knots, [float(v) for v in cc.imag])(ng.y)
                out.append(re + 1j * im)
            return tuple(out)
        xu, zu, comps = self.resampled(period_nm, max_lines)
        start, end = (0.0, z_lo_nm + float(zu[0])), (float(period_nm), z_lo_nm + float(zu[-1]))
        out = []
        for c in comps:
            if not np.any(c != 0):
                out.append(ng.CoefficientFunction(0j))
                continue
            # ONE complex128 array per component (a complex voxel coefficient assembled from
            # separate real/imag voxel objects is not what NGSolve evaluates)
            out.append(ng.VoxelCoefficient(start, end, c, linear=True))
        return tuple(out)


def _hydro_region_eps(spec, where, *, lambda_m, thickness_nm, period_nm,
                      max_grid_lines) -> _RegionEps:
    """The region record of a hydrodynamic conductor layer (optics.fem_2d_tm_hydro). Its H_y
    operator permittivity is the conductor's BACKGROUND (eps_inf; the free carriers enter through
    the extension's current), except in the exact local limit (beta = D = 0), where it is the
    local Drude eps_T(x, z) and no extension is added. Mesh sizing (metal / skin depth / index)
    follows the local eps_T, so the optical mesh outside the conductor is the one the local solve
    of the same profile would build; the conductor's own z (and optional x) lines come from the
    spec."""
    info = spec.region_setup(lambda_m=lambda_m, thickness_nm=thickness_nm, period_nm=period_nm,
                             max_grid_lines=max_grid_lines, where=where)
    r = _RegionEps(info["operator_eps"], where, thickness_nm=thickness_nm, period_nm=period_nm)
    samples = np.asarray(info["local_eps_samples"], dtype=np.complex128).ravel()
    n = np.sqrt(samples)
    n = np.where(n.imag < 0, -n, n)
    r.n_max = float(np.max(np.abs(n.real)))
    r.kappa_max = float(np.max(np.abs(n.imag)))
    r.is_metal = bool(np.any(samples.real < 0.0))
    r.isotropic = False
    r.hydro = spec if info["nonlocal"] else None
    r.lines_override = (list(info["x_lines_nm"]), list(info["z_lines_nm"]))
    return r


# ================================================================================================
# geometry + mesh
# ================================================================================================

@dataclass
class _RegionInfo:
    name: str
    eps: _RegionEps
    z_lo_nm: float
    z_hi_nm: float
    role: str                    # 'layer' | 'block' | 'superstrate' | 'substrate' | 'pml'
    layer: Optional[str] = None


@dataclass
class TMCell:
    """A meshed 2-D periodic cell (output of build_tm_cell): the NGSolve mesh plus everything the
    solver and the extractors need. Lengths are nm (mesh units) unless the name says _m."""
    mesh: Any
    period_nm: float
    lambda_m: float
    theta_deg: float
    n_super: complex
    n_sub: complex
    regions: Dict[str, _RegionInfo]
    x_lines_nm: np.ndarray
    z_lines_nm: np.ndarray
    z_ref_nm: float                         # top interface (r reference plane)
    z_intervals_nm: Dict[str, Tuple[float, float]]
    corners_nm: List[Tuple[float, float]]
    probe_super_nm: Tuple[float, float]
    probe_sub_nm: Tuple[float, float]
    opaque_substrate: bool
    pml_alpha: Tuple[float, float]          # (top, bottom) PML strength
    mesh_spec: TMMeshSpec

    @property
    def n_elements(self) -> int:
        return int(self.mesh.ne)


def _passive_sqrt(z) -> complex:
    w = cmath.sqrt(complex(z))
    if w.imag < 0 or (w.imag == 0 and w.real < 0):
        w = -w
    return w


def _wrap_blocks(layer: TMLayer, P_nm: float, where: str):
    """Block spans [(lo, hi, block_index)] in [0, P], wrapping a boundary-crossing block; refuses
    overlaps and blocks wider than the period."""
    spans = []
    for j, b in enumerate(layer.blocks):
        lo, hi = float(b.x_lo_m) * S, float(b.x_hi_m) * S
        w = hi - lo
        if w <= 0:
            raise ValueError("{}: block {} has non-positive width".format(where, j))
        if w > P_nm * (1 + _REL):
            raise ValueError("{}: block {} is wider than the period".format(where, j))
        if w >= P_nm * (1 - _REL):
            spans.append((0.0, P_nm, j))
            continue
        lo0 = lo - math.floor(lo / P_nm) * P_nm
        if abs(lo0 - P_nm) <= _REL * P_nm:
            lo0 = 0.0
        hi0 = lo0 + w
        if hi0 <= P_nm * (1 + _REL):
            spans.append((lo0, min(hi0, P_nm), j))
        else:
            if P_nm - lo0 > _REL * P_nm:
                spans.append((lo0, P_nm, j))
            spans.append((0.0, hi0 - P_nm, j))
    spans.sort()
    for (a0, a1, _), (b0, b1, _) in zip(spans, spans[1:]):
        if b0 < a1 - _REL * P_nm:
            raise ValueError("{}: blocks overlap in x".format(where))
    return spans


def _graded_axis(breaks, h_of_interval, grade_pts, levels, ratio, tol):
    """1-D node list over sorted `breaks`: interval k gets ceil(L/h_k) uniform cells; at a
    breakpoint in `grade_pts` the adjacent cell is split geometrically (`levels` nodes at
    h1 * ratio^l from the breakpoint)."""
    nodes = []
    gp = list(grade_pts)

    def graded(p):
        return levels > 0 and any(abs(p - g) <= tol for g in gp)

    for k in range(len(breaks) - 1):
        a, b = breaks[k], breaks[k + 1]
        L = b - a
        n = max(1, int(math.ceil(L / max(h_of_interval[k], 1e-12) - 1e-9)))
        ga, gb = graded(a), graded(b)
        if ga and gb and n < 2:
            n = 2
        pts = [a + L * i / n for i in range(n + 1)]
        h1 = L / n
        if ga:
            pts += [a + h1 * ratio ** l for l in range(1, levels + 1)]
        if gb:
            pts += [b - h1 * ratio ** l for l in range(1, levels + 1)]
        nodes.extend(pts)
    nodes = np.unique(np.round(np.asarray(nodes, dtype=np.float64), 12))
    keep = [nodes[0]]
    for v in nodes[1:]:
        if v - keep[-1] > tol:
            keep.append(v)
    keep[-1] = breaks[-1]
    return np.asarray(keep)


def _grid_lines(reps, z_lo, P, spec, where):
    """Mesh lines for a gridded region; advises when a laterally structured grid is too fine to
    align (the coefficient is then integrated across element-interior kinks)."""
    xl, zl = reps.mesh_lines(z_lo, P, spec.max_grid_lines)
    if reps.grid is not None and reps.grid["x"] is not None and (not xl or not zl):
        warnings.warn(
            "{}: the laterally structured grid needs more than max_grid_lines={} mesh lines per "
            "axis, so its samples are not aligned with the mesh; the bilinear coefficient is then "
            "integrated across element-interior kinks (raise TMMeshSpec.max_grid_lines, or coarsen "
            "the grid)".format(where, spec.max_grid_lines), AdvisoryWarning, stacklevel=3)
    return xl, zl


def _auto_buffer(eps, k0, kx, P, lam0_nm, n):
    """Cladding buffer: >= lambda/(2n) and >= 8 decay lengths of the least-evanescent order.

    The decay-length rule is NOT optional: the PML stretches z by a purely imaginary amount, which
    does not attenuate an evanescent order, so an order just past a diffraction cutoff (decay length
    -> infinity) reflects off the outer wall and returns. MEASURED: a lossless 1.28 um grating at
    1.04 um, 10 deg, whose substrate order -2 decays over 2.6 um, closes R + T = 1 to 2.4e-8 with the
    8-decay-length buffer and misses by 1.1e-2 with the buffer capped at 2 lambda/n. Near cutoff the
    buffer (and the cost) therefore grows; it is capped at 30 lambda/n with a warning.

    WHICH ORDERS ARE EVANESCENT (a lossy cladding). An order counts as evanescent when it lies
    beyond the LOSSLESS cut-off, Re(kz^2) = Re(eps) k0^2 - kx_m^2 <= 0, and its decay length is
    1/Im(kz) of the FULL complex kz. This is the exact complement of the PML sizing in build_tm_cell
    (_alpha treats an order as propagating when Re kz > |Im kz|, i.e. Re(kz^2) > 0): a
    complex-coordinate PML absorbs any order with a substantial real kz but does not attenuate a
    decaying one, so every order the PML cannot handle is sized here. The rule is continuous in the
    loss and identical to the lossless one in the lossless limit (there Re(kz^2) <= 0 is exactly
    Re kz = 0). The earlier test |Re kz| < 1e-12 k0 is met by no order of a lossy cladding, so ANY
    loss silently dropped the decay-length rule: MEASURED (a 1.28 um period, 528 nm wide, 160 nm
    tall eps-11.56 rib grating in air at 1.04 um, 10 deg, p, order 4; r against a 4x longer
    buffer): n_sub = 1.45 + 1e-6j shrank the substrate buffer from 21.14 um (the lossless value)
    to 359 nm and moved r by 1.2e-2; for n_sub = 1.45 + 1e-3j the old 359 nm buffer misses r by
    7.3e-3 and the new one (19.96 um) by 4.2e-8; at n_sub = 1.45 + 0.1j the buffer is 3.4 um
    (8 decay lengths of the slowest order, 1/Im kz = 0.42 um) and r is within 1.1e-11
    (tests/test_fem_2d_tm.py, test_auto_buffer_*)."""
    n = max(n, 1e-6)
    buf = 0.5 * lam0_nm / n
    kap = None
    eps = complex(eps)
    for m in range(-200, 201):
        kxm = kx + 2 * math.pi * m / P
        kz = _kz(eps, k0, kxm)
        if (eps * k0 * k0 - kxm * kxm).real <= 0.0 and kz.imag > 0:
            kap = kz.imag if kap is None else min(kap, kz.imag)
    if kap is not None:
        need = 8.0 / kap
        cap = 30.0 * lam0_nm / n
        if need > cap:
            warnings.warn(
                "build_tm_cell: a diffraction order is {:.2g} decay lengths per wavelength from its "
                "cutoff; the cladding buffer is capped at 30 lambda/n, so that slowly decaying order "
                "can reflect off the PML end and bias R/T. Move the wavelength/angle off the cutoff "
                "or set the buffer explicitly.".format(lam0_nm * kap), UnreliableResultWarning,
                stacklevel=3)
        buf = max(buf, min(need, cap))
    return buf


def _slowest_decay(eps, k0, kx, P):
    """Im(kz) of the most slowly decaying diffraction order in a cladding: min over m of
    Im kz(kx + 2 pi m / P), the full complex kz (the decay rate of that order's amplitude away from
    the stack). It is 0 when any order propagates in a lossless cladding.

    This decides whether the substrate is OPAQUE (build_tm_cell). For a passive cladding
    (Im eps >= 0) Im kz = Im sqrt(eps k0^2 - kx_m^2) grows monotonically with kx_m^2, so the
    slowest order is the one with the smallest |kx_m|; it is order 0 only when |kx| <= pi / P.
    Judging opacity from order 0 alone calls a lossless substrate opaque when order 0 is past
    total internal reflection while a diffracted order still propagates into it: T is then
    dropped, the transmitted power is reported as absorption and the bottom buffer / PML are
    sized for the order-0 decay. MEASURED (n_super 1.5, air substrate, 60 deg p, P = lambda/0.9,
    a 250 nm eps 4 / 2.25 grating): the order-0 rule lost T = 0.0282 and moved r by 7.2e-4 from
    the PMM reference; with this rule T and r match PMM (tests/test_fem_2d_tm.py,
    test_grating_coupled_tir_*)."""
    eps = complex(eps)
    mc = -kx * P / (2.0 * math.pi)                    # continuous minimiser of |kx_m|
    kap = None
    for m in range(int(math.floor(mc)) - 1, int(math.ceil(mc)) + 2):
        k = _kz(eps, k0, kx + 2 * math.pi * m / P).imag
        kap = k if kap is None else min(kap, k)
    return float(kap)


def build_tm_cell(period_x_m: float, layers: Sequence[TMLayer], *, lambda_m: float,
                  n_super: complex = 1.0, n_sub: complex = 1.0, theta_deg: float = 0.0,
                  mesh_spec: Optional[TMMeshSpec] = None) -> TMCell:
    """Build the structured, Bloch-periodic quadrilateral mesh of a 1-D-periodic layered cell.

    `layers` are listed BOTTOM FIRST (substrate side first). Lengths in metres; the mesh is built
    in nm. Returns a TMCell (mesh + region table + reference planes + probe lines)."""
    from netgen.meshing import Element1D, Element2D, MeshPoint, Pnt
    from netgen.meshing import Mesh as NGMesh
    import ngsolve as ng

    spec = mesh_spec or TMMeshSpec()
    if spec.corner_mode not in ("contrast", "metal", "all", "none"):
        raise ValueError("TMMeshSpec.corner_mode must be 'contrast', 'metal', 'all' or 'none'")
    if not (float(spec.corner_contrast) >= 1.0):
        raise ValueError("TMMeshSpec.corner_contrast must be >= 1")
    if not (0.0 < spec.corner_ratio < 1.0):
        raise ValueError("TMMeshSpec.corner_ratio must be in (0, 1)")
    if spec.h_scale <= 0:
        raise ValueError("TMMeshSpec.h_scale must be positive")
    P = float(period_x_m) * S
    if not (P > 0):
        raise ValueError("build_tm_cell: period_x_m must be positive")
    lam0 = float(lambda_m) * S
    if not (lam0 > 0):
        raise ValueError("build_tm_cell: lambda_m must be positive")
    if not layers:
        raise ValueError("build_tm_cell: at least one layer is required")
    k0 = 2.0 * math.pi / lam0
    n_super, n_sub = complex(n_super), complex(n_sub)
    _refuse_nonlocal(n_super, "build_tm_cell(n_super)")
    _refuse_nonlocal(n_sub, "build_tm_cell(n_sub)")
    if abs(n_super.imag) > 1e-9:
        raise NotImplementedError(
            "build_tm_cell: R/T and the absorption normalization are defined only for a LOSSLESS "
            "incidence medium (Im(n_super) = 0); got n_super={:.4g}".format(n_super))
    th = float(theta_deg)
    if not (0.0 <= abs(th) < 90.0):
        raise ValueError("build_tm_cell: theta_deg must satisfy |theta| < 90")
    eps_sup, eps_sub = n_super ** 2, n_sub ** 2
    kx = n_super.real * k0 * math.sin(math.radians(th))
    hs = float(spec.h_scale)

    # ---- regions per layer --------------------------------------------------------------------
    names_seen = set(_RESERVED)
    z = 0.0
    lay_info = []                          # (layer, z_lo, z_hi, bg region, [(lo, hi, region)])
    regions: Dict[str, _RegionInfo] = {}
    xbreaks = {0.0, P}
    extra_x, extra_z = set(), set()
    block_owner: Dict[str, Tuple[int, int]] = {}      # block region -> (layer index, block index)
    for li, L in enumerate(layers):
        if not isinstance(L, TMLayer):
            raise TypeError("build_tm_cell: layers must be TMLayer instances")
        where = "build_tm_cell(layer {!r})".format(L.name)
        if L.name in names_seen:
            raise ValueError("{}: duplicate or reserved layer name".format(where))
        names_seen.add(L.name)
        t = float(L.thickness_m) * S
        if not (t > 0):
            raise ValueError("{}: thickness must be positive".format(where))
        if _is_hydro_conductor(L.eps):
            if L.blocks:
                raise NotImplementedError(
                    "{}: lamellar blocks inside a hydrodynamic conductor layer are not supported "
                    "(the conductor must be a laterally uniform slab; put strips in their own "
                    "layer above or below it)".format(where))
            if li > 0 and _is_hydro_conductor(layers[li - 1].eps):
                raise ValueError(
                    "{}: two adjacent hydrodynamic conductor layers: whether a hard wall J.n = 0 "
                    "separates them is ambiguous. Give one connected electron gas ONE "
                    "HydroConductor2D (one density profile), or separate two conductors by a "
                    "dielectric layer".format(where))
            bg = _hydro_region_eps(L.eps, where, lambda_m=float(lambda_m), thickness_nm=t,
                                   period_nm=P, max_grid_lines=spec.max_grid_lines)
        else:
            bg = _RegionEps(L.eps, where, thickness_nm=t, period_nm=P)
        regions[L.name] = _RegionInfo(L.name, bg, z, z + t, "layer", L.name)
        xl, zl = _grid_lines(bg, z, P, spec, where)
        extra_x.update(xl); extra_z.update(zl)
        spans = []
        for lo, hi, j in _wrap_blocks(L, P, where):
            b = L.blocks[j]
            bname = b.name or "{}__blk{}".format(L.name, j)
            if bname in regions and block_owner.get(bname) != (li, j):
                # only the two spans of ONE block split at the seam share a region; a name that is already a
                # layer or another block used to be merged into it silently, with THAT region's eps
                raise ValueError("{}: duplicate or reserved block name {!r} (a layer or another block already "
                                 "has it; give each block its own name)".format(where, bname))
            if bname not in regions:
                block_owner[bname] = (li, j)
                if bname in names_seen:
                    raise ValueError("{}: duplicate or reserved block name {!r}".format(where, bname))
                names_seen.add(bname)
                be = _RegionEps(b.eps, "{} block {!r}".format(where, bname), thickness_nm=t,
                                period_nm=P)
                regions[bname] = _RegionInfo(bname, be, z, z + t, "block", L.name)
                xl, zl = _grid_lines(be, z, P, spec, where)
                extra_x.update(xl); extra_z.update(zl)
            spans.append((lo, hi, bname))
            xbreaks.update((lo, hi))
        lay_info.append((L, z, z + t, L.name, spans))
        z += t
    z_top = z

    # ---- cladding, buffers, PML ---------------------------------------------------------------
    clad_sup = _RegionEps(eps_sup, "superstrate")
    clad_sub = _RegionEps(eps_sub, "substrate")
    kap_sub = _slowest_decay(eps_sub, k0, kx, P)
    opaque = bool(kap_sub > 0 and 1.0 / kap_sub < lam0)
    if spec.superstrate_buffer_m is not None:
        buf_t = float(spec.superstrate_buffer_m) * S
    else:
        buf_t = _auto_buffer(eps_sup, k0, kx, P, lam0, n_super.real)
    if spec.substrate_buffer_m is not None:
        buf_b = float(spec.substrate_buffer_m) * S
    else:
        buf_b = _auto_buffer(eps_sub, k0, kx, P, lam0, max(abs(n_sub.real), 1e-3))
    if opaque:
        buf_b = max(buf_b if spec.substrate_buffer_m is not None else 0.0,
                    15.0 / kap_sub, 4.0 * (1.0 / kap_sub))
    pml_t = float(spec.pml_thickness_m) * S if spec.pml_thickness_m else lam0 / max(n_super.real, 1e-6)
    if spec.pml_thickness_m:
        pml_b = float(spec.pml_thickness_m) * S
    elif opaque:
        pml_b = min(lam0, 5.0 / kap_sub)               # the field is already ~e^-15 here
    else:
        pml_b = lam0 / max(abs(n_sub.real), 1e-6)

    def _alpha(eps_c, d):
        """Quadratic-profile strength giving the requested round-trip attenuation
        exp(-2 kz alpha d / 3) for the slowest propagating order (floored at 0.2 n k0)."""
        kmin = None
        for m in range(-400, 401):
            kz = _kz(eps_c, k0, kx + 2 * math.pi * m / P)
            if kz.real > abs(kz.imag):                  # predominantly propagating
                kmin = kz.real if kmin is None else min(kmin, kz.real)
        n_c = abs(_passive_sqrt(eps_c).real)
        kmin = max(kmin if kmin is not None else 0.0, 0.2 * max(n_c, 0.2) * k0)
        a = 3.0 * math.log(1.0 / spec.pml_attenuation) / (2.0 * kmin * d)
        return float(min(max(a, 1.0), 60.0))

    alpha_t = _alpha(eps_sup, pml_t)
    alpha_b = 1.0 if opaque else _alpha(eps_sub, pml_b)
    z_int = {"pml_bot": (-buf_b - pml_b, -buf_b), "substrate": (-buf_b, 0.0),
             "superstrate": (z_top, z_top + buf_t), "pml_top": (z_top + buf_t, z_top + buf_t + pml_t)}
    regions["substrate"] = _RegionInfo("substrate", clad_sub, -buf_b, 0.0, "substrate")
    regions["pml_bot"] = _RegionInfo("pml_bot", clad_sub, -buf_b - pml_b, -buf_b, "pml")
    regions["superstrate"] = _RegionInfo("superstrate", clad_sup, z_top, z_top + buf_t, "superstrate")
    regions["pml_top"] = _RegionInfo("pml_top", clad_sup, z_top + buf_t, z_top + buf_t + pml_t, "pml")
    for L, zl, zh, _bg, _sp in lay_info:
        z_int[L.name] = (zl, zh)

    # ---- element sizes ------------------------------------------------------------------------
    n_max = max([r.eps.n_max for r in regions.values() if not r.eps.is_metal] + [1.0])
    h_diel = (float(spec.maxh_m) * S if spec.maxh_m else lam0 / (8.0 * n_max)) * hs

    def h_region(r: _RegionInfo):
        if r.eps.is_metal or r.eps.kappa_max > 0.3 * max(r.eps.n_max, 1e-9):
            skin = 1.0 / (k0 * max(r.eps.kappa_max, 1e-9))
            return min(h_diel, spec.metal_h_per_skin * skin * hs)
        return min(h_diel, lam0 / (8.0 * max(r.eps.n_max, 1e-9)) * hs)

    # corners: block edges of structured layers, at the layer's bottom and top faces
    def mat_at(li, x):
        """Region name of layer li at lateral position x (periodic)."""
        if li < 0:
            return "substrate"
        if li >= len(lay_info):
            return "superstrate"
        x = x % P
        for lo, hi, nm in lay_info[li][4]:
            if lo <= x <= hi:
                return nm
        return lay_info[li][3]

    def metal_corner(li, xe, zf, other, left, right):
        """corner_mode='metal': True when the metal / non-metal boundary BENDS at the corner (xe, zf). Each of
        the four quadrants around it is metal or not by its region's Re eps AT the corner (_RegionEps.
        metal_at_face: a z-graded film is judged at that face); the corner carries the field singularity
        the grading resolves when one or three quadrants are metal, or two diagonal ones. No metal, all
        metal, or a straight split (the two quadrants of one layer, or the two on one side of the edge)
        is a flat interface or none: no singularity, no grading. MEASURED: a thin
        film graded to Re eps < 0 near one face and cut into 31 lateral blocks used to get every block
        corner graded (25628 elements instead of 2331, 419 s instead of ~3 s per solve) because the rule
        asked only whether a touching region had Re eps < 0 anywhere."""
        in_face = "bottom" if zf == lay_info[li][1] else "top"
        out_face = "top" if in_face == "bottom" else "bottom"

        def q(name, face, x):
            return regions[name].eps.metal_at_face(face, x % P, P)

        ol, orr = mat_at(other, xe - eta), mat_at(other, xe + eta)
        m = (q(left, in_face, xe - eta), q(right, in_face, xe + eta),
             q(ol, out_face, xe - eta), q(orr, out_face, xe + eta))       # in-L, in-R, out-L, out-R
        k = sum(m)
        if k in (1, 3):
            return True
        if k == 2:                     # straight splits: (in-L, in-R), (out-L, out-R), (in-L, out-L), (in-R, out-R)
            return not ((m[0] and m[1]) or (m[2] and m[3]) or (m[0] and m[2]) or (m[1] and m[3]))
        return False

    def contrast_corner(li, xe, zf, other, left, right):
        """corner_mode='contrast' (with the metal rule): True when a permittivity jump across the
        vertical block edge MEETS one across the horizontal layer face at (xe, zf) by more than
        corner_contrast. With the four quadrant values (each region at the corner itself, as for
        the metal rule) and ratio(a, b) = max(|a|, |b|) / min(|a|, |b|) over the diagonal
        components, the vertical contrast is c_v = max(ratio(in-L, in-R), ratio(out-L, out-R)) and
        the horizontal one c_h = max(ratio(in-L, out-L), ratio(in-R, out-R)); the corner carries
        a field singularity when both jumps are present, so it is graded when min(c_v, c_h) >
        corner_contrast. A straight interface (the edge continuing into the next layer, c_h = 1,
        or a flat face, c_v = 1) is not a corner. See TMMeshSpec for the measured choice."""
        in_face = "bottom" if zf == lay_info[li][1] else "top"
        out_face = "top" if in_face == "bottom" else "bottom"
        ol, orr = mat_at(other, xe - eta), mat_at(other, xe + eta)
        e = [regions[nm].eps.eps_at_face(fc, xx % P, P)
             for nm, fc, xx in ((left, in_face, xe - eta), (right, in_face, xe + eta),
                                (ol, out_face, xe - eta), (orr, out_face, xe + eta))]

        def ratio(a, b):
            out = 1.0
            for u, v in zip(a, b):
                lo, hi = sorted((abs(u), abs(v)))
                out = max(out, math.inf if lo == 0.0 else hi / lo)
            return out
        c_v = max(ratio(e[0], e[1]), ratio(e[2], e[3]))
        c_h = max(ratio(e[0], e[2]), ratio(e[1], e[3]))
        return min(c_v, c_h) > spec.corner_contrast

    def graded_corner(li, xe, zf, other, left, right):
        if spec.corner_mode == "all":
            return True
        if spec.corner_mode == "none":
            return False
        if metal_corner(li, xe, zf, other, left, right):
            return True
        return spec.corner_mode == "contrast" and contrast_corner(li, xe, zf, other, left, right)

    corners, cx, cz = [], set(), set()
    eta = 1e-6 * P
    for li, (L, zl, zh, bgname, spans) in enumerate(lay_info):
        edges = sorted({s[0] for s in spans} | {s[1] for s in spans})
        for xe in edges:
            left, right = mat_at(li, xe - eta), mat_at(li, xe + eta)
            if left == right:
                continue                      # e.g. a block that wraps across the seam
            for zf, other in ((zl, li - 1), (zh, li + 1)):
                if graded_corner(li, xe, zf, other, left, right):
                    corners.append((xe % P, zf))
                    cx.update((0.0, P) if min(xe, P - xe) <= eta else (xe,))
                    cz.add(zf)

    # ---- x axis -------------------------------------------------------------------------------
    xb = sorted(xbreaks | {v for v in extra_x if 0.0 < v < P})
    xb_m = [xb[0]]
    for v in xb[1:]:
        if v - xb_m[-1] > 1e-7 * P:
            xb_m.append(v)
    xb_m[-1] = P
    hx = []
    for a, b in zip(xb_m, xb_m[1:]):
        xm = 0.5 * (a + b)
        h = h_diel
        for (L, zl, zh, bgname, spans) in lay_info:
            for lo, hi, nm in spans:
                if lo <= xm <= hi:
                    h = min(h, h_region(regions[nm]))
            if regions[bgname].eps.grid is not None and regions[bgname].eps.grid["x"] is not None:
                h = min(h, h_region(regions[bgname]))
        hx.append(h)
    xs = _graded_axis(xb_m, hx, sorted(cx), spec.corner_levels, spec.corner_ratio, 1e-7 * P)

    # ---- z axis -------------------------------------------------------------------------------
    zb = [z_int["pml_bot"][0], -buf_b, 0.0]
    hz = []
    n_pml = max(2, int(math.ceil(spec.pml_elements / hs)))
    hz.append(pml_b / n_pml)
    hz.append(min(h_region(regions["substrate"]), buf_b))
    for L, zl, zh, bgname, spans in lay_info:
        inner = sorted(v for v in extra_z if zl + 1e-9 < v < zh - 1e-9)
        pts = [zl] + inner + [zh]
        hreg = min([h_region(regions[bgname])] + [h_region(regions[s[2]]) for s in spans])
        if L.maxh_z_m is not None:
            hreg = min(hreg, float(L.maxh_z_m) * S)
        hreg = min(hreg, (zh - zl) / max(1, spec.min_layer_elements))
        for a, b in zip(pts, pts[1:]):
            zb.append(b)
            hz.append(hreg)
    zb += [z_top + buf_t, z_top + buf_t + pml_t]
    hz.append(min(h_region(regions["superstrate"]), buf_t))
    hz.append(pml_t / n_pml)
    zs = _graded_axis(zb, hz, sorted(cz), spec.corner_levels, spec.corner_ratio,
                      1e-9 * max(lam0, z_top))

    # ---- region of each cell ------------------------------------------------------------------
    def region_at(xm, zm):
        if zm < -buf_b:
            return "pml_bot"
        if zm < 0.0:
            return "substrate"
        if zm > z_top + buf_t:
            return "pml_top"
        if zm > z_top:
            return "superstrate"
        for L, zl, zh, bgname, spans in lay_info:
            if zl <= zm <= zh:
                for lo, hi, nm in spans:
                    if lo <= xm <= hi:
                        return nm
                return bgname
        raise RuntimeError("build_tm_cell: cell centre ({}, {}) in no region".format(xm, zm))

    m = NGMesh(dim=2)
    nx, nz = xs.size - 1, zs.size - 1
    pid = [[m.Add(MeshPoint(Pnt(float(x), float(zz), 0.0))) for x in xs] for zz in zs]
    for i in range(nz + 1):
        # master x = 0, minion x = P: the Bloch phase on the minion side is exp(+i kx P)
        m.AddPointIdentification(pid[i][0], pid[i][nx], identnr=1, type=2)
    idx = {}
    order_names = (["pml_bot", "substrate"] + [n for n in regions if n not in _RESERVED]
                   + ["superstrate", "pml_top"])
    for nm in order_names:
        idx[nm] = m.AddRegion(nm, dim=2)
    used = set()
    xc = 0.5 * (xs[:-1] + xs[1:])
    zc = 0.5 * (zs[:-1] + zs[1:])
    for i in range(nz):
        for j in range(nx):
            nm = region_at(xc[j], zc[i])
            used.add(nm)
            m.Add(Element2D(idx[nm], [pid[i][j], pid[i][j + 1], pid[i + 1][j + 1], pid[i + 1][j]]))
    unused = [n for n in regions if n not in used]
    if unused:
        raise ValueError("build_tm_cell: region(s) {} received no mesh cell (a block narrower than "
                         "the mesh tolerance?)".format(unused))
    ib, it = m.AddRegion("bottom", dim=1), m.AddRegion("top", dim=1)
    il, ir = m.AddRegion("left", dim=1), m.AddRegion("right", dim=1)
    for j in range(nx):
        m.Add(Element1D([pid[0][j], pid[0][j + 1]], index=ib))
        m.Add(Element1D([pid[nz][j + 1], pid[nz][j]], index=it))
    for i in range(nz):
        m.Add(Element1D([pid[i + 1][0], pid[i][0]], index=il))
        m.Add(Element1D([pid[i][nx], pid[i + 1][nx]], index=ir))
    mesh = ng.Mesh(m)

    # probe lines: two per cladding, inside the buffer, separated by ~lambda/(8n)
    dz_t = min(0.25 * buf_t, lam0 / (8.0 * max(n_super.real, 1e-6)))
    p_sup = (z_top + 0.5 * buf_t, z_top + 0.5 * buf_t + dz_t)
    dz_b = min(0.25 * buf_b, lam0 / (8.0 * max(abs(n_sub.real), 1e-6)))
    p_sub = (-0.5 * buf_b, -0.5 * buf_b - dz_b)
    return TMCell(mesh=mesh, period_nm=P, lambda_m=float(lambda_m), theta_deg=th,
                  n_super=n_super, n_sub=n_sub, regions=regions, x_lines_nm=xs, z_lines_nm=zs,
                  z_ref_nm=z_top, z_intervals_nm=z_int, corners_nm=corners, probe_super_nm=p_sup,
                  probe_sub_nm=p_sub, opaque_substrate=opaque, pml_alpha=(alpha_t, alpha_b),
                  mesh_spec=spec)


# ================================================================================================
# background (port) + PML
# ================================================================================================

class _Background:
    """The analytic two-half-space TM background, H_y normalized to a unit incident amplitude at
    the top interface z_ref: H_inc = exp(i kx x - i kz1 (z - z_ref)). The substrate/superstrate
    interface is z = 0 (all structure layers lie above it, so eps_bg = eps_super there)."""

    def __init__(self, cell: TMCell, k0: float, kx: float):
        self.k0, self.kx = k0, kx
        self.e1, self.e2 = complex(cell.n_super) ** 2, complex(cell.n_sub) ** 2
        self.kz1, self.kz2 = _kz(self.e1, k0, kx), _kz(self.e2, k0, kx)
        if abs(self.kz1) <= 1e-6 * k0:
            raise NotImplementedError("solve_tm_cell: grazing incidence (kz ~ 0) is not supported")
        a1, a2 = self.kz1 / self.e1, self.kz2 / self.e2
        self.rF = (a1 - a2) / (a1 + a2)
        self.tF = 1.0 + self.rF
        self.zr = cell.z_ref_nm
        self.A = cmath.exp(1j * self.kz1 * self.zr)
        self.R0_ref = self.rF * cmath.exp(2j * self.kz1 * self.zr)    # bg reflection at z_ref
        self.T0_bot = self.tF * self.A                                 # bg transmission at z = 0

    def cfs(self, ng):
        """(H_bg, grad H_bg) CoefficientFunctions (superstrate branch above z = 0)."""
        iph = ng.exp(1j * self.kx * ng.x)
        e_dn1 = ng.exp(-1j * self.kz1 * ng.y)
        e_up1 = ng.exp(1j * self.kz1 * ng.y)
        e_dn2 = ng.exp(-1j * self.kz2 * ng.y)
        Hs = self.A * (e_dn1 + self.rF * e_up1)
        dHs = self.A * (-1j * self.kz1 * e_dn1 + 1j * self.kz1 * self.rF * e_up1)
        Hb = self.A * self.tF * e_dn2
        dHb = self.A * self.tF * (-1j * self.kz2) * e_dn2
        H = iph * ng.IfPos(ng.y, Hs, Hb)
        gH = ng.CoefficientFunction((1j * self.kx * H, iph * ng.IfPos(ng.y, dHs, dHb)))
        return H, gH


def _pml_profile(ng, cell: TMCell):
    """s(z) = 1 + i alpha xi^2 inside each PML (xi the normalized depth), 1 elsewhere."""
    zt0, zt1 = cell.z_intervals_nm["pml_top"]
    zb0, zb1 = cell.z_intervals_nm["pml_bot"]
    at, ab = cell.pml_alpha
    xt = (ng.y - zt0) / (zt1 - zt0)
    xb = (zb1 - ng.y) / (zb1 - zb0)
    sig = ng.IfPos(ng.y - zt0, at * xt * xt, 0.0) + ng.IfPos(zb1 - ng.y, ab * xb * xb, 0.0)
    return 1.0 + 1j * sig


def _lambda_cfs(ng, cell: TMCell):
    """Per-region Lambda = eps^T / det(eps) components as domain-wise CFs (Lxx, Lxz, Lzx, Lzz),
    plus whether any region is anisotropic."""
    mats = list(cell.mesh.GetMaterials())
    comps = {k: [] for k in ("xx", "xz", "zx", "zz")}
    aniso = False
    for nm in mats:
        r = cell.regions[nm]
        if r.eps.grid is None:
            exx, exz, ezx, ezz = r.eps.const
            det = exx * ezz - exz * ezx
            if det == 0:
                raise ValueError("region {!r}: singular in-plane permittivity".format(nm))
            vals = (exx / det, ezx / det, exz / det, ezz / det)
            comps["xx"].append(ng.CoefficientFunction(vals[0]))
            comps["xz"].append(ng.CoefficientFunction(vals[1]))
            comps["zx"].append(ng.CoefficientFunction(vals[2]))
            comps["zz"].append(ng.CoefficientFunction(vals[3]))
            aniso = aniso or (vals[1] != 0 or vals[2] != 0 or vals[0] != vals[3])
        else:
            exx, exz, ezx, ezz = r.eps.components_cf(ng, r.z_lo_nm, cell.period_nm,
                                                     cell.mesh_spec.max_grid_lines)
            det = exx * ezz - exz * ezx
            comps["xx"].append(exx / det)
            comps["xz"].append(ezx / det)
            comps["zx"].append(exz / det)
            comps["zz"].append(ezz / det)
            aniso = True
    return tuple(ng.CoefficientFunction(comps[k]) for k in ("xx", "xz", "zx", "zz")), aniso


def _flux(L, g):
    """Lambda . g for the 2x2 component tuple L and a 2-vector CF g."""
    Lxx, Lxz, Lzx, Lzz = L
    return (Lxx * g[0] + Lxz * g[1], Lzx * g[0] + Lzz * g[1])


# ================================================================================================
# solve + extraction
# ================================================================================================

@dataclass
class _AssemblyContext:
    """What an extension sees: mesh, wavenumbers, coefficients, background field, helpers."""
    ng: Any
    cell: TMCell
    k0: float
    kx: float
    order: int
    Lambda: Tuple[Any, Any, Any, Any]
    H_bg: Any
    grad_H_bg: Any
    dx: Callable[..., Any]

    @staticmethod
    def inplane_curl(g):
        """J g = (-g_z, g_x): curl of H_y y-hat in the x-z plane (proportional to D)."""
        return (-g[1], g[0])


@dataclass
class TMSolution:
    """Full output of solve_tm_cell: the library-standard OpticalResult plus diagnostics."""
    result: OpticalResult
    r_orders: Dict[int, complex]            # H_y reflection amplitudes of propagating orders
    t_orders: Dict[int, complex]            # H_y transmission amplitudes (at z = 0)
    R_orders: Dict[int, float]
    T_orders: Dict[int, float]
    energy_residual: float                  # 1 - R - T - A_independent
    pml_reflection: float                   # |down-going / up-going| of the scattered 0-order
    ndof: int
    n_elements: int
    assemble_solve_s: float
    cell: TMCell
    gfu: Any = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)   # extension reports (e.g. hydro)


def _gauss_line(x_lines, nq):
    """Gauss-Legendre nodes / weights over every mesh x-cell."""
    g, w = np.polynomial.legendre.leggauss(nq)
    xs, ws = [], []
    for a, b in zip(x_lines[:-1], x_lines[1:]):
        xs.append(0.5 * (b - a) * g + 0.5 * (a + b))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


def _line_values(mesh, gf, xq, z):
    pts = mesh(xq, np.full(xq.size, float(z)))
    if np.any(pts["nr"] < 0):
        raise RuntimeError("probe line z={:.4g} nm left the mesh".format(z))
    return np.asarray(gf(pts), dtype=np.complex128).reshape(-1)


def _extract_orders(cell, gf, k0, kx, eps_c, zlines, z0, sign, orders, nq):
    """Fourier-project the field on two lines z1, z2 and split each order into its outgoing
    (sign=+1 up / -1 down) and incoming parts, both referenced at z0. Returns
    {m: (outgoing, incoming)}."""
    P = cell.period_nm
    xq, wq = _gauss_line(cell.x_lines_nm, nq)
    v1 = _line_values(cell.mesh, gf, xq, zlines[0])
    v2 = _line_values(cell.mesh, gf, xq, zlines[1])
    out = {}
    for m in orders:
        km = kx + 2.0 * math.pi * m / P
        ph = wq * np.exp(-1j * km * xq) / P
        c1, c2 = complex(np.dot(ph, v1)), complex(np.dot(ph, v2))
        kz = _kz(eps_c, k0, km)
        if abs(kz) <= 1e-6 * k0:
            raise NotImplementedError("order {} is at the grazing cutoff in a cladding; R/T are "
                                      "singular there -- nudge the angle/wavelength".format(m))
        M = np.array([[cmath.exp(sign * 1j * kz * (zlines[0] - z0)), cmath.exp(-sign * 1j * kz * (zlines[0] - z0))],
                      [cmath.exp(sign * 1j * kz * (zlines[1] - z0)), cmath.exp(-sign * 1j * kz * (zlines[1] - z0))]])
        a, b = np.linalg.solve(M, np.array([c1, c2]))
        out[m] = (complex(a), complex(b))
    return out


def _propagating(eps_c, k0, kx, P):
    """Propagating diffraction orders in a lossless cladding (always includes 0)."""
    n = abs(_passive_sqrt(eps_c).real)
    mmax = int(math.ceil((n * k0 + abs(kx)) * P / (2 * math.pi))) + 1
    out = []
    for m in range(-mmax, mmax + 1):
        kz = _kz(eps_c, k0, kx + 2 * math.pi * m / P)
        if kz.real > 1e-9 * k0 and abs(kz.imag) <= 1e-9 * k0:
            out.append(m)
    if 0 not in out:
        out.append(0)
    return sorted(out)


def _lossy_substrate_orders(eps_c, k0, kx, P):
    """Orders reported per order in a LOSSY transparent substrate: those above the lossless cut-off
    (Re(kz^2) > 0, the ones a complex-coordinate PML treats as propagating) and always 0."""
    n = abs(_passive_sqrt(eps_c).real)
    mmax = int(math.ceil((n * k0 + abs(kx)) * P / (2 * math.pi))) + 1
    out = [m for m in range(-mmax, mmax + 1)
           if (eps_c * k0 * k0 - (kx + 2 * math.pi * m / P) ** 2).real > 0.0]
    if 0 not in out:
        out.append(0)
    return sorted(out)


def _lossy_substrate_flux(ng, cell, gh_field, gH_bg, Lam, bg, k0, kx, e2, norm, nq, order):
    """Power entering a LOSSY transparent substrate at the interface z = 0, all orders, per unit
    incident power: by Poynting's theorem in the substrate buffer (-b, 0) it is the power absorbed
    in the buffer (the same Im(conj(Lambda grad H) . grad H) integral as per_region_absorption,
    exact on the buffer's mesh region) plus the z-flux leaving the buffer's bottom face z = -b into
    the PML, sum_m |H_m(-b)|^2 Re(kz_m / eps) over EVERY order (cross terms between orders average
    out over a period). The face amplitudes come from the probe lines at -b/2, propagated DOWN to
    -b, so a fast-decaying order is attenuated, never amplified: an order that the per-order sum at
    z = 0 cannot resolve (its amplitude at the probes is below the discretization noise) is counted
    through the absorption integral instead. Only the outgoing (down-going) part enters; the
    incoming part is the PML reflection (pml_attenuation). Lossless limit: the absorption integral
    and every evanescent order's Re(kz / eps) vanish, leaving the per-order sum of the propagating
    orders."""
    P = cell.period_nm
    z_lo = cell.z_intervals_nm["substrate"][0]
    zp = cell.probe_sub_nm
    depth = max(zp) - z_lo                        # probe-to-face distance (> 0)
    n = abs(_passive_sqrt(e2).real)
    mmax = int(math.ceil((n * k0 + abs(kx)) * P / (2 * math.pi))) + 1
    ords = []
    for m in range(-(mmax + 400), mmax + 401):
        kz = _kz(e2, k0, kx + 2 * math.pi * m / P)
        if kz.imag * depth < 40.0:                # below e^-40 of its probe amplitude at the face
            ords.append(m)
    dn = _extract_orders(cell, gh_field, k0, kx, e2, zp, z_lo, -1, ords, nq)
    face = 0.0
    for m in ords:
        kz = _kz(e2, k0, kx + 2 * math.pi * m / P)
        a = dn[m][0] + (bg.T0_bot * cmath.exp(-1j * kz * z_lo) if m == 0 else 0.0)
        face += abs(a) ** 2 * (kz / e2).real
    gHt = gH_bg + ng.grad(gh_field)
    fx, fz = _flux(Lam, gHt)
    loss = (ng.Conj(fx) * gHt[0] + ng.Conj(fz) * gHt[1])
    loss = (loss - ng.Conj(loss)) / 2j
    with ng.TaskManager():
        a_buf = complex(ng.Integrate(loss, cell.mesh, definedon=cell.mesh.Materials("substrate"),
                                     order=2 * order + 2)).real
    return float((face + a_buf / P) / norm)


def solve_tm_cell(cell: TMCell, *, polarization: str = "x", order: int = 4,
                  extensions: Sequence[Any] = (), num_threads: Optional[int] = None,
                  keep_field: bool = False) -> TMSolution:
    """Solve the TM scattering problem on a built cell and extract r/t/R/T/absorption.

    polarization : 'x' (normal incidence: r = E_x ratio) or 'p' (the p-hat convention of
                   optics.tmm_reference; required off normal). 'y' (TE) is not this solver.
    order        : H1 polynomial order. MEASURED accuracy of the default (order 4, default
                   TMMeshSpec), |r - r_converged| at normal incidence: 2.9e-5 on a metal-strip
                   gap-mode cell (metal corners graded; the metal-corner singularity limits it),
                   2.1e-8 .. 5.8e-8 on dielectric rib gratings of contrast 2.25 .. 11.56 (corners
                   graded by the contrast rule), 8.7e-7 .. 4.6e-6 on ribs of contrast 1.44 .. 2.0
                   and 2.1e-5 on a contrast-1.78 grating at 60 deg (below corner_contrast,
                   ungraded); 6.6e-8 on a planar stack. See TMMeshSpec.
    extensions   : extra coupled fields (see the module docstring, EXTENSION).
    num_threads  : NGSolve worker threads (None: DYNAMETA_FEM_THREADS or NGSolve's default)."""
    import ngsolve as ng
    from dynameta.optics.solver import _apply_fem_threads, _free_dof_relres

    if polarization not in ("x", "p"):
        if polarization == "y":
            raise NotImplementedError("solve_tm_cell: polarization 'y' (TE, E along the lines) is "
                                      "not the TM problem this solver implements")
        from dynameta.core.polarization import pol_vocabulary_error
        raise pol_vocabulary_error(polarization, "lab_xyp", where="solve_tm_cell",
                                   param="polarization")
    if abs(cell.theta_deg) > 1e-12 and polarization == "x":
        raise NotImplementedError("solve_tm_cell: oblique incidence requires polarization='p' "
                                  "('x' is not transverse to an oblique wavevector)")
    order = int(order)
    if order < 1:
        raise ValueError("solve_tm_cell: order must be >= 1")
    _apply_fem_threads(type("_T", (), {"num_threads": num_threads})())
    lam0 = cell.lambda_m * S
    k0 = 2.0 * math.pi / lam0
    kx = complex(cell.n_super).real * k0 * math.sin(math.radians(cell.theta_deg))
    P = cell.period_nm
    mesh = cell.mesh
    bg = _Background(cell, k0, kx)
    H_bg, gH_bg = bg.cfs(ng)
    Lam, aniso = _lambda_cfs(ng, cell)
    s = _pml_profile(ng, cell)
    phase = cmath.exp(1j * kx * P)

    t0 = time.perf_counter()
    fes_h = ng.Periodic(ng.H1(mesh, order=order, complex=True), phase=[phase])
    bonus = 2

    def dx(definedon=None):
        if definedon is None:
            return ng.dx(bonus_intorder=bonus)
        return ng.dx(definedon=mesh.Materials(definedon), bonus_intorder=bonus)

    ctx = _AssemblyContext(ng=ng, cell=cell, k0=k0, kx=kx, order=order, Lambda=Lam, H_bg=H_bg,
                           grad_H_bg=gH_bg, dx=dx)
    extensions = list(extensions)
    if any(r.eps.hydro is not None for r in cell.regions.values()):
        from dynameta.optics.fem_2d_tm_hydro import hydro_extensions_for_cell
        extensions += hydro_extensions_for_cell(cell)
    # an extension's space(ctx) is one FESpace, or a tuple of them (several coupled fields; its
    # add_terms / absorption then receive tuples of trial / test / solution components)
    ext_slots, ext_spaces = [], []
    for e in extensions:
        sp = e.space(ctx)
        multi = isinstance(sp, (tuple, list))
        sps = list(sp) if multi else [sp]
        ext_slots.append((multi, list(range(1 + len(ext_spaces), 1 + len(ext_spaces) + len(sps)))))
        ext_spaces += sps
    fes = ng.FESpace([fes_h] + ext_spaces) if ext_spaces else fes_h
    if ext_spaces:
        trials, tests = fes.TrialFunction(), fes.TestFunction()
        h, hv = trials[0], tests[0]
    else:
        h, hv = fes.TnT()

    def _pick(seq, k):
        multi, idx = ext_slots[k]
        return tuple(seq[i] for i in idx) if multi else seq[idx[0]]
    gh, gv = ng.grad(h), ng.grad(hv)
    phys = "|".join(n for n in mesh.GetMaterials() if cell.regions[n].role != "pml")
    pmls = "pml_top|pml_bot"
    # static condensation of element-local DOFs, only when an extension asks for it (the
    # hydrodynamic current lives in L2: condensing it cuts the direct solver's fill several-fold);
    # a cell without such an extension takes exactly the uncondensed path
    condense = any(bool(getattr(e, "condense", False)) for e in extensions)
    a = ng.BilinearForm(fes, condense=True) if condense else ng.BilinearForm(fes)
    fl = _flux(Lam, gh)
    a += (fl[0] * gv[0] + fl[1] * gv[1] - k0 * k0 * h * hv) * dx(phys)
    Lc = Lam[0]                                   # isotropic cladding in the PML
    a += (Lc * (s * gh[0] * gv[0] + (1.0 / s) * gh[1] * gv[1]) - k0 * k0 * s * h * hv) * dx(pmls)
    f = ng.LinearForm(fes)
    struct = "|".join(n for n in mesh.GetMaterials() if cell.regions[n].role in ("layer", "block"))
    e1 = complex(cell.n_super) ** 2
    dL = (Lam[0] - 1.0 / e1, Lam[1], Lam[2], Lam[3] - 1.0 / e1)
    sf = _flux(dL, gH_bg)
    f += (-(sf[0] * gv[0] + sf[1] * gv[1])) * dx(struct)
    for k, e in enumerate(extensions):
        e.add_terms(ctx, a, f, h, hv, _pick(trials, k), _pick(tests, k))

    gfu = ng.GridFunction(fes)
    if not condense:
        with ng.TaskManager():
            a.Assemble()
            f.Assemble()
            inv = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack")
            gfu.vec.data = inv * f.vec
        relres = _free_dof_relres(a, f, gfu, fes)
    else:
        with ng.TaskManager():
            a.Assemble()
            f.Assemble()
            f.vec.data += a.harmonic_extension_trans * f.vec
            cpl = fes.FreeDofs(True)
            inv = a.mat.Inverse(cpl, inverse="umfpack")
            gfu.vec.data = inv * f.vec
            rvec = gfu.vec.CreateVector()
            rvec.data = f.vec - a.mat * gfu.vec          # residual of the condensed system
            free = np.fromiter(cpl, dtype=bool, count=len(rvec))
            bn = float(np.linalg.norm(f.vec.FV().NumPy()[free]))
            relres = float(np.linalg.norm(rvec.FV().NumPy()[free])) / (bn if bn > 0 else 1.0)
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * f.vec
    t_solve = time.perf_counter() - t0
    if not relres <= 1e-6:
        # a direct (UMFPACK) solve on the geometric corner mesh lands at ~1e-9..1e-7 (the
        # grading makes the matrix ill-conditioned, not inaccurate); 1e-6 flags a genuine failure
        warnings.warn("solve_tm_cell: direct solve residual {:.2e} > 1e-6; the field is "
                      "unreliable".format(relres), UnreliableResultWarning, stacklevel=2)
    gh_field = gfu.components[0] if ext_spaces else gfu

    # ---- extraction ---------------------------------------------------------------------------
    nq = order + 6
    e2 = complex(cell.n_sub) ** 2
    ords_sup = _propagating(e1, k0, kx, P)
    up = _extract_orders(cell, gh_field, k0, kx, e1, cell.probe_super_nm, cell.z_ref_nm, +1,
                         ords_sup, nq)
    r_orders = {m: up[m][0] + (bg.R0_ref if m == 0 else 0.0) for m in ords_sup}
    pml_refl = abs(up[0][1]) / max(abs(up[0][0]), 1e-300)
    kz0 = bg.kz1
    norm = (kz0 / e1).real
    R_orders = {m: float(abs(r) ** 2 * (_kz(e1, k0, kx + 2 * math.pi * m / P) / e1).real / norm)
                for m, r in r_orders.items()}
    R = float(sum(R_orders.values()))
    t_orders, T_orders, T, t = {}, {}, None, None
    if not cell.opaque_substrate:
        lossless_sub = abs(e2.imag) < 1e-12
        ords_sub = _propagating(e2, k0, kx, P) if lossless_sub else _lossy_substrate_orders(e2, k0, kx, P)
        dn = _extract_orders(cell, gh_field, k0, kx, e2, cell.probe_sub_nm, 0.0, -1, ords_sub, nq)
        t_orders = {m: dn[m][0] + (bg.T0_bot if m == 0 else 0.0) for m in ords_sub}
        T_orders = {m: float(abs(tt) ** 2 * (_kz(e2, k0, kx + 2 * math.pi * m / P) / e2).real / norm)
                    for m, tt in t_orders.items()}
        if lossless_sub:
            T = float(sum(T_orders.values()))
        else:
            T = _lossy_substrate_flux(ng, cell, gh_field, gH_bg, Lam, bg, k0, kx, e2, norm, nq, order)
        t = complex(t_orders[0]) * complex(cell.n_super) / complex(cell.n_sub)
    rH = complex(r_orders[0])
    r = -rH if polarization == "x" else rH

    # ---- absorption ---------------------------------------------------------------------------
    gHt =gH_bg + ng.grad(gh_field)
    fx, fz = _flux(Lam, gHt)
    loss = (ng.Conj(fx) * gHt[0] + ng.Conj(fz) * gHt[1])
    loss = (loss - ng.Conj(loss)) / 2j
    absorb_regions = [n for n in mesh.GetMaterials()
                      if cell.regions[n].role in ("layer", "block")
                      or (cell.regions[n].role == "substrate" and cell.opaque_substrate)]
    per_region = {}
    if absorb_regions:
        with ng.TaskManager():
            vals = ng.Integrate(loss, mesh, definedon=mesh.Materials("|".join(absorb_regions)),
                                region_wise=True, order=2 * order + 2)
        for nm, v in zip(mesh.GetMaterials(), vals):
            if nm in absorb_regions:
                per_region[nm] = float(complex(v).real) / (P * norm)
    diagnostics: Dict[str, Any] = {}
    for k, e in enumerate(extensions):
        comps = _pick(gfu.components, k)
        hook = getattr(e, "absorption", None)
        if hook is not None:
            extra = hook(ctx, comps, gHt) or {}
            for nm, v in extra.items():
                per_region[nm] = per_region.get(nm, 0.0) + float(v)
        dhook = getattr(e, "diagnostics", None)
        if dhook is not None:
            for key, val in (dhook(ctx, comps) or {}).items():
                if isinstance(val, dict) and isinstance(diagnostics.get(key), dict):
                    diagnostics[key].update(val)
                else:
                    diagnostics[key] = val
    A_ind = float(sum(per_region.values()))
    A = float(1.0 - R - (T or 0.0))
    resid = float(1.0 - R - (T or 0.0) - A_ind)
    if abs(resid) > 1e-2:
        warnings.warn(
            "solve_tm_cell: energy residual 1 - R - T - A = {:.2e} > 1e-2: the field is not "
            "resolved (refine the mesh / raise the order) or the PML/buffers are too thin".format(
                resid), UnreliableResultWarning, stacklevel=2)
    if pml_refl > 1e-3:
        warnings.warn(
            "solve_tm_cell: the scattered zeroth order carries a down-going part {:.2e} of the "
            "up-going one in the superstrate (PML reflection); thicken the PML or raise "
            "TMMeshSpec.pml_elements".format(pml_refl), UnreliableResultWarning, stacklevel=2)
    res = OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))), solve_time_s=t_solve,
                        t=t, T=T, A=A, A_independent=A_ind, R_flux=R, T_flux=T,
                        per_region_absorption=per_region, fit_relres=float(pml_refl),
                        linear_solver="umfpack")
    return TMSolution(result=res, r_orders=r_orders, t_orders=t_orders, R_orders=R_orders,
                      T_orders=T_orders, energy_residual=resid, pml_reflection=float(pml_refl),
                      ndof=int(fes.ndof), n_elements=int(mesh.ne), assemble_solve_s=t_solve,
                      cell=cell, gfu=gfu if keep_field else None, diagnostics=diagnostics)


def _spec_with(mesh_spec, maxh_m, corner_refinement):
    spec = mesh_spec or TMMeshSpec()
    kw = {}
    if maxh_m is not None:
        kw["maxh_m"] = float(maxh_m)
    if corner_refinement is not None:
        if corner_refinement is False or corner_refinement == 0:
            kw["corner_levels"] = 0
        elif isinstance(corner_refinement, (tuple, list)):
            kw["corner_levels"], kw["corner_ratio"] = int(corner_refinement[0]), float(corner_refinement[1])
        else:
            kw["corner_levels"] = int(corner_refinement)
    return _replace(spec, **kw) if kw else spec


def _replace(spec, **kw):
    import dataclasses
    return dataclasses.replace(spec, **kw)


def solve_fem_2d_tm(period_x_m: float, layers: Sequence[TMLayer], *, lambda_m: float,
                    theta_deg: float = 0.0, n_super: complex = 1.0, n_sub: complex = 1.0,
                    polarization: str = "x", order: int = 4, maxh_m: Optional[float] = None,
                    corner_refinement=None, mesh_spec: Optional[TMMeshSpec] = None,
                    extensions: Sequence[Any] = (), num_threads: Optional[int] = None,
                    return_solution: bool = False):
    """Solve a 1-D-periodic layered cell under TM illumination (one call: mesh + solve + extract).

    period_x_m  : lateral period [m].
    layers      : TMLayer list, BOTTOM FIRST (substrate side first).
    lambda_m    : vacuum wavelength [m]; theta_deg the incidence angle in the superstrate (x-z plane).
    n_super/n_sub : semi-infinite cladding indices (lossless superstrate required).
    polarization: 'x' (normal incidence, r = E_x ratio) or 'p' (p-hat convention; oblique).
    order       : H1 order. maxh_m / corner_refinement (levels, or (levels, ratio), or 0) override
                  the TMMeshSpec.
    Returns an OpticalResult (or the full TMSolution with return_solution=True)."""
    spec = _spec_with(mesh_spec, maxh_m, corner_refinement)
    cell = build_tm_cell(period_x_m, layers, lambda_m=lambda_m, n_super=n_super, n_sub=n_sub,
                         theta_deg=theta_deg, mesh_spec=spec)
    sol = solve_tm_cell(cell, polarization=polarization, order=order, extensions=extensions,
                        num_threads=num_threads)
    return sol if return_solution else sol.result


# ================================================================================================
# Design adapter
# ================================================================================================

def _epsfield_to_spec(ef, layer, period_m, where):
    """A dynameta EpsField (axes in nm) -> TM eps spec for one layer."""
    _refuse_nonlocal(ef, where)
    if getattr(ef, "is_uniform", True):
        if getattr(ef, "tensor", None) is not None:
            return _xz_block(np.asarray(ef.tensor, dtype=np.complex128), where)
        return complex(ef.scalar)
    v = np.asarray(ef.values_zyx, dtype=np.complex128)
    tensor = v.ndim == 5
    if v.ndim not in (3, 5):
        raise ValueError("{}: values_zyx must be (Nz,Ny,Nx) or (Nz,Ny,Nx,3,3)".format(where))
    if not is_y_invariant(v):
        raise ValueError("{}: the gridded EpsField varies along y; the 2-D TM solver needs a "
                         "y-invariant cell".format(where))
    v = v[:, 0]                                                  # (Nz, Nx[, 3, 3])
    if tensor:
        if tensor_couples_y(v):
            raise ValueError("{}: tensor couples y to x/z (TE-TM mixing); not a 2-D TM "
                             "problem".format(where))
        v = v[..., [[0, 0], [2, 2]], [[0, 2], [0, 2]]]          # (Nz, Nx, 2, 2) x-z block
    zz = np.asarray(ef.z_axis_u, dtype=np.float64).ravel()
    if zz.size >= 2 and zz[0] > zz[-1]:                          # descending axis: flip both
        zz, v = zz[::-1], v[::-1]
    z = (zz - zz[0]) / S
    x =np.asarray(ef.x_axis_u, dtype=np.float64) / S if v.shape[1] > 1 else None
    if x is not None and (x[-1] - x[0]) >= float(period_m) * (1 - 1e-9):
        # an endpoint-inclusive periodic grid repeats x = 0 at x = P: drop the duplicate
        x, v = x[:-1], v[:, :-1]
    if v.shape[1] == 1 and not tensor:
        v = v[:, 0]
    return GriddedEps2D(values=v, z_m=z, x_m=x)


def _xz_block(t, where):
    if tensor_couples_y(t):
        raise ValueError("{}: tensor couples y to x/z (TE-TM mixing); not a 2-D TM problem"
                         .format(where))
    return np.array([[t[0, 0], t[0, 2]], [t[2, 0], t[2, 2]]], dtype=np.complex128)


def design_to_tm_layers(design, eps_by_region=None, lambda_m: float = None):
    """Translate a 1-D lamellar Design (y-invariant Rectangle inclusions spanning the full y
    period) into (period_x_m, [TMLayer...] bottom first, n_super, n_sub). eps_by_region (keyed by
    design layer, or by FEM mesh region -- collapsed as for the layered bridges) overrides a
    layer's background with a uniform scalar / tensor or a gridded EpsField (axes in nm)."""
    from dynameta.core.layered import collapse_regions_to_layers
    from dynameta.geometry.guards import require_supported_geometry
    from dynameta.optics.tmm_reference import end_media_indices

    where = "design_to_tm_layers"
    if lambda_m is None or not (float(lambda_m) > 0):
        raise ValueError("{}: lambda_m must be a positive wavelength [m]".format(where))
    require_supported_geometry(design, where)
    for ef in (eps_by_region or {}).values():
        _refuse_nonlocal(ef, where)
    for nm in {design.stack.superstrate_material, design.stack.substrate_material} | {
            m for L in design.stack.layers for m in L.materials_used()}:
        mat = design.materials.get(nm)
        _refuse_nonlocal(getattr(mat, "optical", mat), "{} (material {!r})".format(where, nm))
    ebr = collapse_regions_to_layers(design, eps_by_region or {})
    P = float(design.unit_cell.period_x_m)
    Py = float(design.unit_cell.period_y_m)
    n_super, n_sub = end_media_indices(design, lambda_m)
    out = []
    for L in design.stack.layers:
        lw = "{} (layer {!r})".format(where, L.name)
        ef = ebr.get(L.name)
        if ef is not None:
            bg = _epsfield_to_spec(ef, L, P, lw)
        else:
            bg = complex(design.materials.get(L.background_material).eps(lambda_m))
        blocks = []
        for j, inc in enumerate(L.inclusions):
            if getattr(inc.shape, "kind", "") != "rectangle":
                raise ValueError("{}: inclusion {} is a {!r}, not a lamellar Rectangle (the 2-D "
                                 "solver needs y-invariant lines)".format(
                                     lw, j, getattr(inc.shape, "kind", "?")))
            xlo, xhi, ylo, yhi = inc.shape.bbox_m()
            if ylo > 1e-9 * Py or yhi < Py * (1 - 1e-9):
                raise ValueError("{}: Rectangle inclusion {} does not span the full y period "
                                 "(y-invariant lines required)".format(lw, j))
            blocks.append(TMBlock(xlo, xhi, complex(design.materials.get(inc.material).eps(lambda_m)),
                                  name="{}__incl{}".format(L.name, j)))
        out.append(TMLayer(L.name, float(L.thickness_m), bg, tuple(blocks)))
    return P, out, complex(n_super), complex(n_sub)


def _design_incidence(design):
    opt = design.optical
    if getattr(opt, "incidence_side", "top") != "top":
        raise NotImplementedError("2-D TM FEM: top-side incidence only")
    if abs(float(getattr(opt, "azimuth_deg", 0.0) or 0.0)) > 1e-12:
        raise NotImplementedError("2-D TM FEM: conical incidence (azimuth != 0) is not a 2-D "
                                  "problem")
    pol = getattr(opt, "polarization", "x")
    return float(getattr(opt, "incidence_angle_deg", 0.0) or 0.0), pol


def solve_design_2d_tm(design, eps_by_region, lambda_m: float, *, order: int = 4,
                       maxh_m: Optional[float] = None, corner_refinement=None,
                       mesh_spec: Optional[TMMeshSpec] = None, n_super=None, n_sub=None,
                       num_threads: Optional[int] = None, return_solution: bool = False):
    """solve_fem_2d_tm for a 1-D lamellar Design (see design_to_tm_layers); incidence angle and
    polarization come from design.optical ('x' at normal incidence or 'p')."""
    from dynameta.core.eps_field import require_solver_time_convention
    require_solver_time_convention(eps_by_region or {}, "solve_design_2d_tm")
    P, layers, ns, nb = design_to_tm_layers(design, eps_by_region, lambda_m)
    theta, pol = _design_incidence(design)
    return solve_fem_2d_tm(P, layers, lambda_m=lambda_m, theta_deg=theta,
                           n_super=ns if n_super is None else n_super,
                           n_sub=nb if n_sub is None else n_sub, polarization=pol, order=order,
                           maxh_m=maxh_m, corner_refinement=corner_refinement,
                           mesh_spec=mesh_spec, num_threads=num_threads,
                           return_solution=return_solution)


def make_fem_2d_tm_optical_solver(*, order: int = 4, mesh_spec: Optional[TMMeshSpec] = None,
                                  num_threads: Optional[int] = None):
    """An `optical_solver` for run_pipeline (the OpticalSolver seam signature) backed by the 2-D
    TM FEM, for 1-D lamellar designs. `geo` is ignored (the solver meshes its own cell)."""
    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        return solve_design_2d_tm(design, eps_by_region, lambda_m, order=order,
                                  mesh_spec=mesh_spec, n_super=n_super, n_sub=n_sub,
                                  num_threads=num_threads)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        # the claddings passed in are honoured exactly as by _solve (and by the 3-D FEM
        # sweep); the sweep path used to drop them and re-derive the design's, so solve_sweep and the
        # per-wavelength call disagreed for any caller-supplied cladding
        return [solve_design_2d_tm(design, assemble_at(lam), lam, order=order,
                                   mesh_spec=mesh_spec, n_super=n_super, n_sub=n_sub,
                                   num_threads=num_threads) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    _solve.cache_fingerprint = "fem_2d_tm(order={},mesh={!r})".format(order, mesh_spec)
    # a laterally uniform carrier film takes the native-z path of
    # GriddedEps2D (identical lateral columns collapse), so the route declares the capability and
    # run_pipeline hands it the native carrier grid, like every other layered route
    _solve.native_carrier_grid = True
    return _solve


# ================================================================================================
# convergence helper
# ================================================================================================

def tm_convergence_study(solve: Callable[[Any], Any], settings: Sequence[Any]) -> List[dict]:
    """Run `solve(setting)` for each setting (e.g. an order, a TMMeshSpec, an h_scale) and
    tabulate the zeroth-order r with its change from the previous row and from the LAST row (the
    finest setting is taken as the reference). `solve` returns an OpticalResult or a TMSolution.
    Rows: {'setting', 'r', 'R', 'd_prev', 'd_last', 'ndof', 'time_s', 'energy_residual'}."""
    rows = []
    for s_ in settings:
        t0 = time.perf_counter()
        out = solve(s_)
        dt = time.perf_counter() - t0
        res = out.result if isinstance(out, TMSolution) else out
        rows.append({"setting": s_, "r": complex(res.r), "R": float(res.R),
                     "ndof": getattr(out, "ndof", None), "time_s": dt,
                     "energy_residual": getattr(out, "energy_residual", None)})
    for k, row in enumerate(rows):
        row["d_prev"] = abs(row["r"] - rows[k - 1]["r"]) if k else None
        row["d_last"] = abs(row["r"] - rows[-1]["r"])
    return rows
