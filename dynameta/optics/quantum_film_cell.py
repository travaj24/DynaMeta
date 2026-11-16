"""Exact (per lateral harmonic) treatment of a laterally uniform film under a lamellar grating: a 2-D
periodic TM finite-element region for the patterned layers, coupled through Fourier (Dirichlet-to-
Neumann) ports to the laterally uniform stack below it, where every lateral harmonic is solved with
its own exact transfer matrix.

WHY. The film's self-consistent response (optics.quantum_film) is laterally LOCAL (the electronic
response is taken at in-plane wavevector q -> 0) and nonlocal along z. In a laterally uniform film a
laterally local response does not couple lateral harmonics, so each harmonic exp(i q_m x),
q_m = k_x + 2 pi m / P, of the field crosses the film exactly as a plane wave of in-plane wavevector
q_m crosses the planar film: through QuantumFilmResponse.transfer_matrix(omega, q_m), the exact planar
element. That is the q -> 0 response WITHOUT the thin-layer reduction of
QuantumFilmResponse.effective_layer (first order in the thickness, and in q L). This module is the
independent referee for the equivalent layer inside a structured cell: it shares no code with the
local 2-D TM solver (optics.fem_2d_tm) beyond its input records (TMLayer / TMBlock), so agreement
between the two on the SAME uniaxial film is evidence for both.

(Why not the extension hook of optics.fem_2d_tm. A z-nonlocal kernel sigma_zz(z, z') couples every
pair of points of a film column. NGSolve's symbolic forms are pointwise, so the coupling needs either
auxiliary unknowns that depend on x only (the mode or low-rank amplitudes of the polarization) -- the
structured mesh offers no x-only space without a change to the solver core, and the kernel is full rank
(396 of 399 singular values above 1e-10 at 5 nm), so no low-rank shortcut exists -- or a local
Sternheimer system for the perturbed sub-band states, whose finite-element ground state would differ
from the Schrodinger-Poisson finite-difference model at O(h^2) (~1e-5 in r), above the effect to be
measured. The per-harmonic route below is exact within the q -> 0 model for a laterally uniform film.)

FORMULATION (exp(-i omega t); x lateral, z up; H = H_y; nm in the mesh). In the FEM region
(the patterned layers, plus a superstrate buffer above and part of the uppermost uniform layer below),
H = exp(i k_x x) u with u P-periodic, and

    int [ L_xx (d_x + i k_x) u (d_x - i k_x) v + L_zz d_z u d_z v - k0^2 u v ] dA
      - sum_m (i k_m / eps_s) u^t_m b^t_m(v) + sum_m Y_m u^b_m b^b_m(v)
      = -2 (i k_0 / eps_s) a_0 b^t_0(v),

L = (1 / eps_zz, 1 / eps_xx) on (d_x, d_z); b_m(v) = int exp(i 2 pi m x / P) v dx on the top (t) or
bottom (b) line; u_m = (1/P) int u exp(-i 2 pi m x / P) dx; k_m = sqrt(eps_s k0^2 - q_m^2) (Im >= 0:
outgoing / decaying upward); a_0 the incident H amplitude at the top line. The top port is the exact
radiation condition of the superstrate. The bottom port Y_m = [(1/eps_xx) d_z H / H]_m is the exact
admittance of the stack below for harmonic m: the stack's layers are 2 x 2 transfer matrices in the
(E_x, H_y / (omega eps0)) basis and orientation of optics.nonlocal_tmm (superstrate face -> substrate
face; there (1/eps_xx) d_z Ht = -i E_x with z up, so Y = -i E / Ht), built from the substrate's
outgoing wave upward one layer at a time (a layer whose evanescent decay exceeds exp(-35) is taken as a
half-space: what lies below it changes Y by < exp(-70)). Harmonics |m| <= n_harmonics enter both
ports; the FE trace has finite resolution, so n_harmonics is a convergence parameter (gated).

Structure of a solve: one sparse LU of the FEM matrix with the top port per wavelength, then the bottom
port enters as a rank-(2M+1) update (Woodbury): with X = A^-1 B_b and y = A^-1 f, the bottom-line
harmonics are w = (I + C_b^T X diag(Y))^-1 C_b^T y, so ANY stack below (another bias, another film
model) costs only its admittances and one dense (2M+1)^2 solve. FilmCell holds the factorization.

OUTPUTS (FilmCellResult): r (the library convention: 'x' at normal incidence r = -r_H, 'p' r = r_H; at
the top interface of the patterned layers), R and T order-summed over propagating orders, the absorbed
fraction of every FEM region (volume integral) and of every layer of the stack below (difference of
the per-harmonic Poynting fluxes at its faces -- for the film this is exact for the discrete film
element too, whose discrete Poynting theorem holds), the energy residual 1 - R - T - sum A, and the
per-harmonic data of the film (q_m, H and E at its faces, absorbed power per harmonic) for spectral
analysis (film_spectrum).

Validity: TM, lamellar (y-invariant) patterned layers, laterally uniform layers below them, lossless
superstrate, top-side incidence; the film enters through ANY callable (omega, q) -> 2 x 2 matrix
(FilmLayer: exact, equivalent, local graded, or anything else).
"""
from __future__ import annotations

import cmath
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.optics.resonance import _admittance, _kz

__all__ = ["UniformLayer", "FilmLayer", "CellMeshSpec", "FilmCell", "FilmCellResult", "solve_film_cell",
           "film_spectrum", "stack_admittance"]

_NM = 1e9
_HALFSPACE_DECAY = 35.0


# ------------------------------------------------------------------------------------------------
# the laterally uniform stack below the FEM region
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class UniformLayer:
    """A laterally uniform local layer: eps a complex scalar or (eps_xx, eps_zz)."""
    name: str
    thickness_m: float
    eps: Any

    def components(self) -> Tuple[complex, complex]:
        e = self.eps
        if isinstance(e, (tuple, list, np.ndarray)) and len(e) == 2:
            return complex(e[0]), complex(e[1])
        return complex(e), complex(e)

    def matrix(self, omega: float, k_par: complex) -> np.ndarray:
        from dynameta.optics.quantum_film import uniaxial_layer_matrix
        exx, ezz = self.components()
        return uniaxial_layer_matrix(exx, ezz, float(self.thickness_m), complex(omega) / C_LIGHT, k_par)

    def decay(self, omega: float, k_par: complex) -> float:
        """Im(k_z) d: the evanescent attenuation exponent across the layer."""
        exx, ezz = self.components()
        k0 = complex(omega) / C_LIGHT
        kz = cmath.sqrt(exx * (k0 * k0 - complex(k_par) ** 2 / ezz))
        return abs(kz.imag) * float(self.thickness_m)

    def halfspace_admittance(self, omega: float, k_par: complex) -> complex:
        """Ht / E of the downward outgoing (decaying) wave: eps_xx / k_z (nonlocal_tmm basis)."""
        exx, ezz = self.components()
        k0 = complex(omega) / C_LIGHT
        kz = cmath.sqrt(exx * (k0 * k0 - complex(k_par) ** 2 / ezz))
        if kz.imag < 0 or (kz.imag == 0 and kz.real < 0):
            kz = -kz
        return exx / kz


@dataclass(frozen=True)
class FilmLayer:
    """A laterally uniform layer given by its p-pol transfer matrix for any in-plane wavevector:
    matrix_fn(omega, k_par) -> 2 x 2 in the nonlocal_tmm basis and orientation (superstrate face ->
    substrate face). `symmetric_in_q` (True when the matrix depends on k_par^2 only, as for every
    film here) lets a normal-incidence solve evaluate each |q| once."""
    name: str
    thickness_m: float
    matrix_fn: Callable[[float, complex], np.ndarray]
    symmetric_in_q: bool = True
    label: str = ""

    def matrix(self, omega: float, k_par: complex) -> np.ndarray:
        return np.asarray(self.matrix_fn(omega, k_par), dtype=np.complex128)

    @classmethod
    def exact(cls, response, name: str = "film", *, superstrate_face: str = "top") -> "FilmLayer":
        """The exact planar element of a QuantumFilmResponse for every harmonic (q -> 0 electronic
        response, no thin-layer reduction). The response kernel is computed once per frequency."""
        cache: Dict[float, np.ndarray] = {}

        def fn(omega, k_par, _r=response, _c=cache):
            R = _c.get(float(omega))
            if R is None:
                _c.clear()
                R = _r.response_kernel(omega)
                _c[float(omega)] = R
            return _r.transfer_matrix(omega, k_par, superstrate_face=superstrate_face, R=R)
        # the matrix maps the NOMINAL film (an element extended into the oxides is folded back)
        return cls(name, float(response.film.nominal_thickness_m), fn, True, "exact")

    @classmethod
    def equivalent(cls, response, name: str = "film") -> "FilmLayer":
        """The first-order equivalent uniaxial layer (QuantumFilmResponse.effective_layer) -- what the
        pipeline hands the structured solvers."""
        from dynameta.optics.quantum_film import uniaxial_layer_matrix

        def fn(omega, k_par, _r=response):
            exx, ezz, L = _r.effective_layer(omega)
            return uniaxial_layer_matrix(exx, ezz, L, complex(omega) / C_LIGHT, k_par)
        return cls(name, float(response.film.nominal_thickness_m), fn, True, "equivalent")

    @classmethod
    def local_graded(cls, z_m, n_m3, optical, name: str = "film", *, top_is_last: bool = True) -> "FilmLayer":
        """The classical LOCAL film: a cascade of isotropic slabs, one per interval of the density
        profile n(z) (interval mean of the piecewise-linear profile), eps = optical.eps(lambda,
        n_m3=n_mean) (e.g. a DrudeOptical). z_m ascending; top_is_last: z_m[-1] faces the
        superstrate (the dual-gate solver's top face)."""
        from dynameta.optics import nonlocal_tmm as nt
        z = np.asarray(z_m, dtype=np.float64)
        n = np.asarray(n_m3, dtype=np.float64)
        if top_is_last:
            z, n = (z[-1] - z)[::-1], n[::-1]
        nav = 0.5 * (n[:-1] + n[1:])
        dz = np.diff(z)

        def fn(omega, k_par, _cache={}):
            lam = 2.0 * math.pi * C_LIGHT / float(omega)
            key = float(omega)
            if key not in _cache:
                _cache.clear()
                e = np.asarray(optical.eps(lam, n_m3=nav), dtype=np.complex128)
                _cache[key] = [nt.DielectricLayer(complex(ek), float(dk)) for ek, dk in zip(e, dz)]
            k0 = complex(omega) / C_LIGHT
            return nt._stack_matrix(complex(omega), _cache[key], k0, complex(k_par), "p")
        return cls(name, float(z[-1] - z[0]), fn, True, "local")


def _layer_decay_and_halfspace(layer, omega, q):
    if isinstance(layer, UniformLayer):
        return layer.decay(omega, q), layer
    return 0.0, None


def stack_admittance(lower: Sequence, omega: float, q: complex, eps_sub: complex):
    """Y (z-up convention, 1/m) at the top face of `lower` (BOTTOM FIRST) over substrate eps_sub for
    the in-plane wavevector q, and the list of layer matrices top -> bottom (None below a layer taken
    as a half-space). Built upward from the substrate's outgoing wave."""
    k0 = complex(omega) / C_LIGHT
    kz_s = _kz(complex(eps_sub), k0, complex(q))
    Yb = _admittance(complex(eps_sub), kz_s, "p")                        # Ht / E, downgoing
    mats: List[Optional[np.ndarray]] = [None] * len(lower)
    cut = None
    # from the top down: the first layer whose decay exceeds the threshold is a half-space
    for j in range(len(lower) - 1, -1, -1):
        d, _ = _layer_decay_and_halfspace(lower[j], omega, q)
        if d > _HALFSPACE_DECAY:
            cut = j
            break
    start = 0
    if cut is not None:
        Yb = lower[cut].halfspace_admittance(omega, q)
        start = cut + 1
    for i in range(start, len(lower)):                                     # upward
        T = lower[i].matrix(omega, q)
        mats[i] = T
        e_top = T[1, 1] - T[0, 1] * Yb                                     # T^-1 (1, Yb), det T = 1
        h_top = -T[1, 0] + T[0, 0] * Yb
        Yb = h_top / e_top
    return -1j / Yb, mats, start


# ------------------------------------------------------------------------------------------------
# mesh
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CellMeshSpec:
    """Mesh of the FEM region. h_m: element size cap (default lambda / (10 n_max), at most 20 nm);
    corner_levels / corner_ratio: geometric grading toward every block edge and the layer interfaces
    bounding a block (in x and z); buffer_top_m: superstrate buffer above the patterned layers (default
    30 nm); min_rows: element rows per layer at least."""
    h_m: Optional[float] = None
    corner_levels: int = 6
    corner_ratio: float = 0.3
    buffer_top_m: float = 30e-9
    min_rows: int = 1


def _graded_points(breaks, h, grade_pts, levels, ratio):
    """Sorted points covering [breaks[0], breaks[-1]] with spacing <= h, every break kept, and a
    geometric sequence toward each grade point (distances h r, h r^2, ... h r^levels)."""
    pts = set(float(b) for b in breaks)
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(math.ceil((b - a) / h - 1e-9)))
        for k in range(1, n):
            pts.add(a + (b - a) * k / n)
    lo, hi = float(breaks[0]), float(breaks[-1])
    for g in grade_pts:
        for k in range(1, levels + 1):
            d = h * ratio ** k
            for p in (g - d, g + d):
                if lo < p < hi:
                    pts.add(p)
    out = np.array(sorted(pts))
    keep = [out[0]]
    for p in out[1:]:
        if p - keep[-1] > 1e-9 * max(1.0, hi - lo):
            keep.append(p)
    return np.array(keep)


# ------------------------------------------------------------------------------------------------
# results
# ------------------------------------------------------------------------------------------------
@dataclass
class FilmCellResult:
    """One solve. r: library convention ('x': -r_H at normal incidence, 'p': r_H) at the top interface
    of the patterned layers; r_H the H_y reflection there; R, T order-summed; absorption {region or
    layer name: fraction}; energy_residual = 1 - R - T - sum(absorption); film: per-harmonic data of
    the named film layer {'m', 'q_m' [1/m], 'H_top', 'E_top', 'H_bottom', 'E_bottom', 'A_m'} in the
    nonlocal_tmm basis normalized to the incident flux (A_m sums to the film absorption)."""
    r: complex
    r_H: complex
    R: float
    T: Optional[float]
    absorption: Dict[str, float]
    energy_residual: float
    film: Dict[str, Any] = field(default_factory=dict)
    harmonics: int = 0
    ndof: int = 0
    solve_s: float = 0.0


def film_spectrum(result: FilmCellResult) -> Dict[str, np.ndarray]:
    """Lateral-wavevector spectrum of the field in the film from a FilmCellResult: q_m [1/m], the
    normal displacement |q_m H_m|^2 at the film's top and bottom faces (relative, normalized to its
    sum over m), and the film's absorbed power per harmonic A_m (sums to the film absorption)."""
    f = result.film
    q = np.asarray(f["q_m"])
    dz = 0.5 * (np.abs(q * f["H_top"]) ** 2 + np.abs(q * f["H_bottom"]) ** 2)
    tot = float(np.sum(dz))
    return {"q_m": q, "Dz2": dz / tot if tot > 0 else dz, "A_m": np.asarray(f["A_m"])}


# ------------------------------------------------------------------------------------------------
# the cell
# ------------------------------------------------------------------------------------------------
class FilmCell:
    """The FEM region of a lamellar cell with its top (superstrate) port, factorized at one
    wavelength; solve(lower, eps_sub) adds the bottom port of any laterally uniform stack.

    period_m; cell_layers: fem_2d_tm.TMLayer list BOTTOM FIRST (eps scalar or (exx, ezz); blocks
    TMBlock with scalar or (exx, ezz) eps), the region between the bottom port line (z = 0) and the
    patterned layers' top; a superstrate buffer of mesh.buffer_top_m is added above; lambda_m;
    theta_deg (in the superstrate, x-z plane); n_super (lossless); polarization 'x' (normal incidence)
    or 'p'; order (H1 order); n_harmonics M (harmonics -M..M in both ports); mesh: CellMeshSpec."""

    def __init__(self, period_m: float, cell_layers: Sequence, *, lambda_m: float, theta_deg: float = 0.0,
                 n_super: complex = 1.0, polarization: str = "x", order: int = 5, n_harmonics: int = 200,
                 mesh: Optional[CellMeshSpec] = None, num_threads: Optional[int] = 2):
        import ngsolve as ng
        from netgen.meshing import Element1D, Element2D, MeshPoint, Pnt
        from netgen.meshing import Mesh as NGMesh
        if polarization not in ("x", "p"):
            raise ValueError("FilmCell: polarization must be 'x' (normal incidence) or 'p' (TM)")
        if abs(theta_deg) > 1e-12 and polarization == "x":
            raise ValueError("FilmCell: oblique incidence needs polarization='p'")
        ns = complex(n_super)
        if abs(ns.imag) > 0:
            raise ValueError("FilmCell: the superstrate must be lossless")
        if num_threads:
            ng.SetNumThreads(int(num_threads))
        t0 = time.perf_counter()
        self.ng = ng
        self.P = float(period_m) * _NM
        self.lambda_m = float(lambda_m)
        self.omega = 2.0 * math.pi * C_LIGHT / self.lambda_m
        self.k0 = 2.0 * math.pi / (self.lambda_m * _NM)                    # 1/nm
        self.eps_s = ns * ns
        self.kx = ns.real * self.k0 * math.sin(math.radians(theta_deg))
        self.theta_deg = float(theta_deg)
        self.polarization = polarization
        self.order = int(order)
        self.M = int(n_harmonics)
        spec = mesh or CellMeshSpec()
        self.spec = spec

        # ---- geometry: layers bottom first from z = 0, superstrate buffer on top ----
        lay = []
        z = 0.0
        for L in cell_layers:
            d = float(L.thickness_m) * _NM
            blocks = []
            for b in getattr(L, "blocks", ()) or ():
                blocks.append((float(b.x_lo_m) * _NM, float(b.x_hi_m) * _NM, b.eps,
                               b.name or "{}__blk{}".format(L.name, len(blocks))))
            lay.append((L.name, z, z + d, L.eps, blocks))
            z += d
        self.z_struct_top = z
        buf = float(spec.buffer_top_m) * _NM
        self.z_top = z + buf
        lay.append(("superstrate", z, self.z_top, self.eps_s, []))
        self.layers_nm = lay

        def comps(e):
            if isinstance(e, (tuple, list, np.ndarray)) and len(e) == 2:
                return complex(e[0]), complex(e[1])
            return complex(e), complex(e)

        # ---- mesh lines ----
        n_max = max(abs(cmath.sqrt(c).real) for (_n, _a, _b, e, bl) in lay
                    for c in list(comps(e)) + [x for blk in bl for x in comps(blk[2])])
        h = (float(spec.h_m) * _NM if spec.h_m else min(20.0, self.lambda_m * _NM / (10.0 * max(n_max, 1.0))))
        xb = {0.0, self.P}
        gx, gz = set(), set()
        for (_n, za, zb, _e, bl) in lay:
            for (x0, x1, _eb, _nm) in bl:
                for xv in (x0, x1):
                    xw = xv % self.P
                    xb.add(xw)
                    gx.add(xw)
                gz.update((za, zb))
        xb = sorted(xb)
        xs = _graded_points(xb, h, sorted(gx), spec.corner_levels, spec.corner_ratio)
        gxs = sorted(gx)
        for g in gxs:                                   # periodic images of grade points near 0 / P
            for p in (g - self.P, g + self.P):
                for k in range(1, spec.corner_levels + 1):
                    for s in (-1, 1):
                        v = p + s * h * spec.corner_ratio ** k
                        if 0.0 < v < self.P and all(abs(v - u) > 1e-9 * self.P for u in xs):
                            xs = np.sort(np.append(xs, v))
        zb_ = sorted({0.0, self.z_top} | {za for (_n, za, _b, _e, _bl) in lay})
        hz_layers = []
        for (_n, za, zb, _e, _bl) in lay:
            nrow = max(int(spec.min_rows), int(math.ceil((zb - za) / h - 1e-9)))
            hz_layers.append((za, zb, (zb - za) / nrow))
        zs = set(zb_)
        for (za, zb, hz) in hz_layers:
            n = int(round((zb - za) / hz))
            for k in range(1, n):
                zs.add(za + (zb - za) * k / n)
        for g in sorted(gz):
            for k in range(1, spec.corner_levels + 1):
                d = h * spec.corner_ratio ** k
                for p in (g - d, g + d):
                    if 0.0 < p < self.z_top:
                        zs.add(p)
        zs = np.array(sorted(zs))
        zk = [zs[0]]
        for p in zs[1:]:
            if p - zk[-1] > 1e-9 * self.z_top:
                zk.append(p)
        zs = np.array(zk)
        self.x_lines, self.z_lines = xs, zs

        # ---- regions ----
        def region_at(xm, zm):
            for (nm, za, zb, e, bl) in lay:
                if za <= zm <= zb:
                    for (x0, x1, eb, bn) in bl:
                        lo = x0 % self.P
                        w = x1 - x0
                        for sh in (0.0, -self.P):
                            if lo + sh <= xm <= lo + sh + w:
                                return bn, eb
                    return nm, e
            raise RuntimeError("FilmCell: point ({}, {}) in no region".format(xm, zm))

        m = NGMesh(dim=2)
        nx, nz = xs.size - 1, zs.size - 1
        pid = [[m.Add(MeshPoint(Pnt(float(x), float(zz), 0.0))) for x in xs] for zz in zs]
        for i in range(nz + 1):
            m.AddPointIdentification(pid[i][0], pid[i][nx], identnr=1, type=2)
        reg_idx, reg_eps = {}, {}
        xc = 0.5 * (xs[:-1] + xs[1:])
        zc = 0.5 * (zs[:-1] + zs[1:])
        cells = []
        for i in range(nz):
            for j in range(nx):
                nm, e = region_at(xc[j], zc[i])
                if nm not in reg_idx:
                    reg_idx[nm] = m.AddRegion(nm, dim=2)
                    reg_eps[nm] = comps(e)
                cells.append((reg_idx[nm], i, j))
        for (ri, i, j) in cells:
            m.Add(Element2D(ri, [pid[i][j], pid[i][j + 1], pid[i + 1][j + 1], pid[i + 1][j]]))
        ib, it = m.AddRegion("bottom", dim=1), m.AddRegion("top", dim=1)
        il, ir = m.AddRegion("left", dim=1), m.AddRegion("right", dim=1)
        for j in range(nx):
            m.Add(Element1D([pid[0][j], pid[0][j + 1]], index=ib))
            m.Add(Element1D([pid[nz][j + 1], pid[nz][j]], index=it))
        for i in range(nz):
            m.Add(Element1D([pid[i + 1][0], pid[i][0]], index=il))
            m.Add(Element1D([pid[i][nx], pid[i + 1][nx]], index=ir))
        self.mesh = ng.Mesh(m)
        self.region_eps = reg_eps
        mats = list(self.mesh.GetMaterials())
        self.Lxx = ng.CoefficientFunction([1.0 / reg_eps[nm][1] for nm in mats])     # 1 / eps_zz on d_x
        self.Lzz = ng.CoefficientFunction([1.0 / reg_eps[nm][0] for nm in mats])     # 1 / eps_xx on d_z

        # ---- space, volume form, ports ----
        fes = ng.Periodic(ng.H1(self.mesh, order=self.order, complex=True))
        self.fes = fes
        u, v = fes.TnT()
        kx, k0 = self.kx, self.k0
        a = ng.BilinearForm(fes)
        a += (self.Lxx * (ng.grad(u)[0] + 1j * kx * u) * (ng.grad(v)[0] - 1j * kx * v)
              + self.Lzz * ng.grad(u)[1] * ng.grad(v)[1] - k0 * k0 * u * v) * ng.dx(bonus_intorder=2)
        with ng.TaskManager():
            a.Assemble()
        import scipy.sparse as sp
        rows, cols, vals = a.mat.COO()
        nfull = fes.ndof
        free = np.fromiter(fes.FreeDofs(), dtype=bool, count=nfull)    # periodic minion DOFs are not free
        self.free = np.flatnonzero(free)
        remap = -np.ones(nfull, dtype=np.int64)
        remap[self.free] = np.arange(self.free.size)
        self._remap = remap
        rows, cols = remap[np.asarray(rows, dtype=np.int64)], remap[np.asarray(cols, dtype=np.int64)]
        vals = np.asarray(vals, dtype=np.complex128)
        ok = (rows >= 0) & (cols >= 0)
        n = int(self.free.size)
        A = sp.csr_matrix((vals[ok], (rows[ok], cols[ok])), shape=(n, n))
        ms = np.arange(-self.M, self.M + 1)
        self.ms = ms
        self.q_nm = kx + 2.0 * math.pi * ms / self.P

        def line_dofs(bc):
            d = set()
            for el in self.mesh.Boundaries(bc).Elements():
                d.update(int(i) for i in fes.GetDofNrs(el) if int(i) >= 0 and free[int(i)])
            return np.array(sorted(d), dtype=np.int64)
        full_t, full_b = line_dofs("top"), line_dofs("bottom")
        self._full_t, self._full_b = full_t, full_b
        self.dofs_t, self.dofs_b = remap[full_t], remap[full_b]          # compressed numbering
        # port integrals b_m(phi_j) = int exp(i 2 pi m x / P) phi_j dx: the trace of every port-line
        # basis function sampled at Gauss points of each line element (enough points per element
        # for the highest harmonic), then one matrix product for all harmonics
        gfb = ng.GridFunction(fes)
        vec = gfb.vec.FV().NumPy()

        def port_matrix(z_line, dofs):
            xq, wq = [], []
            for a_, b_ in zip(xs[:-1], xs[1:]):
                nq = max(self.order + 2, int(math.ceil(math.pi * self.M * (b_ - a_) / self.P)) + 8)
                g, wg = np.polynomial.legendre.leggauss(nq)
                xq.append(0.5 * (b_ - a_) * g + 0.5 * (a_ + b_))
                wq.append(0.5 * (b_ - a_) * wg)
            xq, wq = np.concatenate(xq), np.concatenate(wq)
            pts = self.mesh(xq, np.full(xq.size, float(z_line)))
            V = np.zeros((xq.size, dofs.size))
            for c, d in enumerate(dofs):
                vec[:] = 0.0
                vec[d] = 1.0
                V[:, c] = np.asarray(gfb(pts), dtype=np.complex128).reshape(-1).real
            vec[:] = 0.0
            E = np.exp(1j * (2.0 * math.pi / self.P) * np.outer(xq, ms)) * wq[:, None]
            return V.T @ E
        Bt = port_matrix(self.z_top, full_t)                                # port-line rows only
        Bb = port_matrix(0.0, full_b)
        self.Bt, self.Bb = Bt, Bb
        self.Ct, self.Cb = np.conj(Bt) / self.P, np.conj(Bb) / self.P        # u_m = C^T u (real basis)
        km = np.array([_kz(self.eps_s, k0, q) for q in self.q_nm])          # 1/nm, Im >= 0
        self.k_m = km
        Kt = -(1j * km / self.eps_s)
        blk = (Bt * Kt[None, :]) @ self.Ct.T                                # dense top-port block
        ii, jj = np.meshgrid(self.dofs_t, self.dofs_t, indexing="ij")
        A1 = (A + sp.csr_matrix((blk.ravel(), (ii.ravel(), jj.ravel())), shape=(n, n))).tocsc()
        from scipy.sparse.linalg import splu
        self.lu = splu(A1)
        self.a0 = 1.0 + 0j                                                  # incident H at the top line
        i0 = int(np.flatnonzero(ms == 0)[0])
        self.i0 = i0
        f = np.zeros(n, dtype=np.complex128)
        f[self.dofs_t] = -2.0 * (1j * km[i0] / self.eps_s) * self.a0 * Bt[:, i0]
        self.f = f
        self.y = self.lu.solve(f)
        # X = A1^-1 B_b, kept only on the two port lines, in column chunks
        Xb = np.zeros((self.dofs_b.size, ms.size), dtype=np.complex128)
        Xt = np.zeros((self.dofs_t.size, ms.size), dtype=np.complex128)
        chunk = 64
        for c0 in range(0, ms.size, chunk):
            c1 = min(ms.size, c0 + chunk)
            rhs = np.zeros((n, c1 - c0), dtype=np.complex128)
            rhs[self.dofs_b, :] = Bb[:, c0:c1]
            sol = self.lu.solve(rhs)
            Xb[:, c0:c1] = sol[self.dofs_b, :]
            Xt[:, c0:c1] = sol[self.dofs_t, :]
        self.CbX = self.Cb.T @ Xb
        self.CtX = self.Ct.T @ Xt
        self.Cby = self.Cb.T @ self.y[self.dofs_b]
        self.Cty = self.Ct.T @ self.y[self.dofs_t]
        self.ndof = int(n)
        self.setup_s = time.perf_counter() - t0

    # ---- one stack below ----
    def admittances(self, lower: Sequence, eps_sub: complex, film_harmonics: Optional[int] = None,
                    film_fallback=None):
        """Per-harmonic Y [1/nm] of the stack `lower` (BOTTOM FIRST); with film_harmonics = Mf, every
        layer that is a FilmLayer uses film_fallback (a FilmLayer, e.g. the equivalent layer) for
        |m| > Mf (its field there is negligible: check convergence in Mf)."""
        Y = np.zeros(self.ms.size, dtype=np.complex128)
        mats = [None] * self.ms.size
        starts = [0] * self.ms.size
        memo: Dict[float, Tuple] = {}
        sym = abs(self.kx) == 0.0 and all(getattr(lay, "symmetric_in_q", True) for lay in lower)
        for k, mm in enumerate(self.ms):
            q = complex(self.q_nm[k]) * _NM                                 # 1/m
            use = list(lower)
            if film_harmonics is not None and abs(int(mm)) > int(film_harmonics) and film_fallback is not None:
                use = [film_fallback if isinstance(lay, FilmLayer) else lay for lay in lower]
            key = (abs(q.real) if sym else q.real, abs(int(mm)) > int(film_harmonics or 10 ** 9))
            if sym and key in memo:
                Y[k], mats[k], starts[k] = memo[key]
                continue
            y, Ts, st = stack_admittance(use, self.omega, q, eps_sub)
            Y[k], mats[k], starts[k] = y / _NM, Ts, st
            if sym:
                memo[key] = (Y[k], Ts, st)
        return Y, mats, starts

    def solve(self, lower: Sequence, eps_sub: complex, *, film: Optional[str] = None,
              film_harmonics: Optional[int] = None, film_fallback=None,
              absorption: bool = True) -> FilmCellResult:
        """Solve with the stack `lower` (BOTTOM FIRST: UniformLayer / FilmLayer) over substrate eps_sub.
        film: the name of the layer whose per-harmonic data is returned (default: the first FilmLayer)."""
        t0 = time.perf_counter()
        lower = list(lower)
        if film is None:
            film = next((lay.name for lay in lower if isinstance(lay, FilmLayer)), None)
        Y, mats, starts = self.admittances(lower, eps_sub, film_harmonics, film_fallback)
        S = np.eye(self.ms.size, dtype=np.complex128) + self.CbX * Y[None, :]
        w = np.linalg.solve(S, self.Cby)                                    # bottom-line harmonics
        ht = self.Cty - self.CtX @ (Y * w)                                  # top-line harmonics
        i0 = self.i0
        km = self.k_m
        b = ht.copy()
        b[i0] -= self.a0
        norm = (km[i0] / self.eps_s).real
        prop = (km.real > 1e-9 * self.k0) & (np.abs(km.imag) <= 1e-9 * self.k0)
        R = float(np.sum(np.abs(b[prop]) ** 2 * (km[prop] / self.eps_s).real) / (abs(self.a0) ** 2 * norm))
        rH_top = b[i0] / self.a0
        rH = rH_top * cmath.exp(2j * km[i0] * (self.z_struct_top - self.z_top))
        r = -rH if self.polarization == "x" else rH
        # ---- the stack below: per-harmonic fields and fluxes (nonlocal_tmm basis, Ht = H) ----
        flux_inc = (km[i0] * _NM / self.eps_s).real * abs(self.a0) ** 2      # Re(E Ht*) of the incident wave
        names = [lay.name for lay in lower]
        absorb = {nm: 0.0 for nm in names}
        Tsum = 0.0
        fdat = {"m": self.ms.copy(), "q_m": self.q_nm * _NM, "H_top": np.zeros(self.ms.size, complex),
                "E_top": np.zeros(self.ms.size, complex), "H_bottom": np.zeros(self.ms.size, complex),
                "E_bottom": np.zeros(self.ms.size, complex), "A_m": np.zeros(self.ms.size)}
        for k in range(self.ms.size):
            vec = np.array([1j * Y[k] * _NM * w[k], w[k]], dtype=np.complex128)   # (E, Ht) at the port line
            st = starts[k]
            flux_prev = (vec[0] * np.conj(vec[1])).real
            for i in range(len(lower) - 1, st - 1, -1):                        # top -> down
                T = mats[k][i]
                if lower[i].name == film:
                    fdat["E_top"][k], fdat["H_top"][k] = vec[0], vec[1]
                vec = T @ vec
                fl = (vec[0] * np.conj(vec[1])).real
                absorb[lower[i].name] += (flux_prev - fl) / flux_inc
                if lower[i].name == film:
                    fdat["E_bottom"][k], fdat["H_bottom"][k] = vec[0], vec[1]
                    fdat["A_m"][k] = (flux_prev - fl) / flux_inc
                flux_prev = fl
            if st > 0:                                                          # the half-space layer
                absorb[lower[st - 1].name] += flux_prev / flux_inc
            else:
                Tsum += flux_prev / flux_inc
        if absorption:
            ng = self.ng
            rhs = self.f.copy()
            rhs[self.dofs_b] -= self.Bb @ (Y * w)                           # A1 u = f - B_b diag(Y) w
            uvec = self.lu.solve(rhs)
            gf = ng.GridFunction(self.fes)
            full = np.zeros(self.fes.ndof, dtype=np.complex128)
            full[self.free] = uvec
            gf.vec.FV().NumPy()[:] = full
            gx = ng.grad(gf)[0] + 1j * self.kx * gf
            gz = ng.grad(gf)[1]
            mats_ = list(self.mesh.GetMaterials())
            ixx = ng.CoefficientFunction([(-1.0 / self.region_eps[nm][1]).imag for nm in mats_])
            izz = ng.CoefficientFunction([(-1.0 / self.region_eps[nm][0]).imag for nm in mats_])
            dens = (ixx * (gx * ng.Conj(gx)).real + izz * (gz * ng.Conj(gz)).real)
            with ng.TaskManager():
                vals = ng.Integrate(dens, self.mesh, region_wise=True, order=2 * self.order + 4)
            pre = (self.eps_s.real / (km[i0].real * self.P)) / abs(self.a0) ** 2
            for nm, val in zip(mats_, vals):
                if nm == "superstrate":
                    continue
                absorb[nm] = absorb.get(nm, 0.0) + float(np.real(val)) * pre
        T = float(Tsum) if Tsum != 0.0 else 0.0
        resid = float(1.0 - R - T - sum(absorb.values())) if absorption else float("nan")
        return FilmCellResult(r=complex(r), r_H=complex(rH), R=R, T=T, absorption=absorb,
                              energy_residual=resid, film=fdat, harmonics=int(self.M),
                              ndof=self.ndof, solve_s=time.perf_counter() - t0)


def solve_film_cell(period_m: float, cell_layers: Sequence, lower: Sequence, *, lambda_m: float,
                    eps_sub: complex, theta_deg: float = 0.0, n_super: complex = 1.0,
                    polarization: str = "x", order: int = 5, n_harmonics: int = 200,
                    mesh: Optional[CellMeshSpec] = None, film_harmonics: Optional[int] = None,
                    film_fallback=None, num_threads: Optional[int] = 2) -> FilmCellResult:
    """One-call FilmCell(...).solve(lower, eps_sub): the FEM region `cell_layers` (TMLayer list, bottom
    first) over the laterally uniform stack `lower` (UniformLayer / FilmLayer, bottom first) on a
    substrate of permittivity eps_sub. See the module docstring."""
    cell = FilmCell(period_m, cell_layers, lambda_m=lambda_m, theta_deg=theta_deg, n_super=n_super,
                    polarization=polarization, order=order, n_harmonics=n_harmonics, mesh=mesh,
                    num_threads=num_threads)
    return cell.solve(lower, eps_sub, film_harmonics=film_harmonics, film_fallback=film_fallback)
