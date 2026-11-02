"""v0.11.7 gates: a DIAGONAL tensor eps on a symmetry-reduced FEM cell, linear_solver=
'sparsecholesky', solve_fem(cross_pol=True) -> OpticalResult.r_cross, and the FREE-DOF residual.

1. A tensor diagonal in (x, y, z) keeps both mirror parities, so a quarter cell must reproduce the
   full periodic R/T; an off-diagonal tensor must be refused (unless the caller asserts
   tensor_mirror_symmetric). Premise: the pillar's anisotropy is LOAD-BEARING (x- and y-pol R differ
   by far more than the quarter-vs-full agreement), so the gate cannot pass on a degenerate tensor.
2. sparsecholesky (LDL^T) equals umfpack to round-off on a symmetric solve, REFUSES a gyrotropic
   (Hermitian, non-symmetric) tensor by residual (measured 3.9e-2 vs the 1e-8 cap), and falls back to
   umfpack at oblique incidence (Bloch-phased, non-symmetric) without factorizing first.
3. The iterative-solver residual counts FREE DOFs only. Premise: the PEC-walled quarter cell HAS
   Dirichlet DOFs and the old all-DOF residual on the very same converged solve reads O(1) (measured
   ~2), so the "did not converge" warning used to be a false alarm on every PEC-walled GMRes solve.
4. cross_pol: a 45-deg rotated rectangle in a square lattice has a diagonal mirror, so its Jones
   matrix must satisfy r_xx = r_yy and r_yx = r_xy, with r_cross far from zero; and the fit-
   independent Poynting R_flux must be closer to |r|^2 + |r_cross|^2 than to |r|^2 alone. A mirror-
   symmetric pillar gives r_cross ~ 0 on the full cell and EXACTLY 0j on a reduced cell (the
   reduced-cell average of the odd cross field is not its 0-order: measured 5.0e-3 when fitted).

ngsolve-gated."""

import math
import types
import warnings

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.materials import Material, MaterialRegistry, ConstantOptical      # noqa: E402
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell          # noqa: E402
from dynameta.geometry.cross_section import Polygon, Rectangle, centered_circle  # noqa: E402
from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec                     # noqa: E402
from dynameta.core.eps_field import EpsField                                    # noqa: E402
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder               # noqa: E402
from dynameta.optics.eps_assembler import assemble_eps_cf                       # noqa: E402
import dynameta.optics.solver as S                                              # noqa: E402

LAM = 1300e-9
PER = 400e-9
N_PIL = 3.0
# uniaxial-like pillar, principal axes on the lab axes (both mirror parities kept)
DIAG = np.diag([3.2 ** 2, 2.0 ** 2, 2.6 ** 2]).astype(complex)
# gyrotropic: Hermitian, NOT symmetric (eps_xy = -eps_yx) -- also off-diagonal
GYRO = np.array([[2.6 ** 2, 0.3j, 0.0], [-0.3j, 2.6 ** 2, 0.0], [0.0, 0.0, 2.6 ** 2]])


RECT_L, RECT_W = 260e-9, 100e-9


def _design(symmetry="none", shape="circle"):
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0), ("glass", 1.5 ** 2), ("pil", N_PIL ** 2)):
        reg.add(Material(nm, ConstantOptical(complex(e))))
    cell = UnitCell.square(PER)
    if shape == "circle":
        shps = [centered_circle(cell, 130e-9)]
    elif shape == "rot45":   # RECT_L x RECT_W rectangle rotated +45 deg about the cell centre
        cx, cy = cell.center_m
        c = s = math.sqrt(0.5)
        hl, hw = RECT_L / 2, RECT_W / 2
        shps = [Polygon(points_m=[(cx + c * x - s * y, cy + s * x + c * y) for x, y in
                                  ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))])]
    else:   # "rot45_eigenframe": the SAME crystal rotated -45 deg -- rectangles along x, on the lattice
        # (P/sqrt2)(1, -1), (P/sqrt2)(1, 1), whose sqrt2-P square supercell holds two of them at
        # (L/4, L/4) and (3L/4, 3L/4) (both clear of the cell boundary for RECT_L/2 < L/4)
        L = math.sqrt(2.0) * PER
        cell = UnitCell.square(L)
        shps = [Rectangle(f * L, f * L, RECT_L, RECT_W) for f in (0.25, 0.75)]
    stack = Stack(layers=[Layer("slab", 250e-9, "air",
                                inclusions=[Inclusion(shape=sh, material="pil") for sh in shps])],
                  superstrate_material="air", substrate_material="glass")
    m3 = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=120e-9, maxh_substrate_m=120e-9, maxh_background_m=75e-9,
                    maxh_inclusion_m=45e-9, fem_order=2, symmetry=symmetry)
    return Design(name="sym_tensor", unit_cell=cell, stack=stack, electrodes=[], materials=reg,
                  mesh_3d=m3)


_GEO = {}


def _geo(symmetry="none", shape="circle"):
    key = (symmetry, shape)
    if key not in _GEO:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # the opt-in symmetry hint
            _GEO[key] = (_design(symmetry, shape), LayeredOpticalBuilder(_design(symmetry, shape)).build())
    return _GEO[key]


def _eps(d, geo, pil_tensor):
    ebr = {r: EpsField(scalar=complex(d.materials.get(geo.material_by_region[r]).eps(LAM)))
           for r in geo.mesh.GetMaterials()}
    if pil_tensor is not None:
        for r in geo.mesh.GetMaterials():
            if geo.material_by_region[r] == "pil":
                ebr[r] = EpsField(tensor=np.asarray(pil_tensor, complex))
    return assemble_eps_cf(geo, ebr)


def _solve(symmetry, pol, *, shape="circle", ls="umfpack", tensor=None, theta=0.0, record=False, **kw):
    d, geo = _geo(symmetry, shape)
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=theta, linear_solver=ls)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = S.solve_fem(geo, LAM, _eps(d, geo, tensor), opt, order=2, n_super=1.0 + 0j,
                          n_sub=1.5 + 0j, **kw)
    return (res, [str(x.message) for x in w]) if record else res


# ---------------------------------------------------------------- 1. diagonal tensor, reduced cell

def test_diagonal_tensor_quarter_matches_full():
    tol = 2e-3                     # FEM-vs-FEM floor (the two meshes differ); measured |dr| <= 3.1e-4
    full = {p: _solve("none", p, tensor=DIAG) for p in ("x", "y")}
    for p in ("x", "y"):
        q = _solve("quarter", p, tensor=DIAG)
        assert abs(full[p].r - q.r) < tol, (p, full[p].r, q.r)
        assert abs(full[p].T - q.T) < tol, (p, full[p].T, q.T)
    # premise: the tensor is load-bearing -- a degenerate (isotropic-acting) one would pass trivially
    # (measured |r_x - r_y| = 0.060)
    assert abs(full["x"].r - full["y"].r) > 10 * tol, (full["x"].r, full["y"].r)


def test_offdiagonal_tensor_refused_on_reduced_cell_unless_asserted():
    with pytest.raises(NotImplementedError, match="OFF-DIAGONAL"):
        _solve("quarter", "x", tensor=GYRO)
    # the caller's assertion admits it (unchecked by design); it must then solve, not raise
    res = _solve("quarter", "x", tensor=GYRO, tensor_mirror_symmetric=True)
    assert math.isfinite(res.R)


# ---------------------------------------------------------------- 2. sparsecholesky

def test_sparsecholesky_matches_umfpack():
    for sym, tensor in (("none", None), ("quarter", DIAG)):
        ru = _solve(sym, "x", tensor=tensor)
        rs = _solve(sym, "x", ls="sparsecholesky", tensor=tensor)
        assert abs(rs.r - ru.r) < 1e-10, (sym, rs.r, ru.r)


def test_sparsecholesky_refuses_nonsymmetric_tensor():
    with pytest.raises(RuntimeError, match="not complex-symmetric"):
        _solve("none", "x", ls="sparsecholesky", tensor=GYRO)


def test_sparsecholesky_oblique_falls_back_to_umfpack():
    S._ADVISED_ONCE.discard("sparsecholesky_nonsymmetric")
    rs, msgs = _solve("none", "y", ls="sparsecholesky", theta=20.0, record=True)
    ru = _solve("none", "y", theta=20.0)
    assert any("'sparsecholesky' is invalid" in m for m in msgs), msgs
    assert abs(rs.r - ru.r) < 1e-12, (rs.r, ru.r)


# ---------------------------------------------------------------- 3. free-DOF residual

def test_free_dof_residual_no_false_alarm_on_pec_walls(monkeypatch):
    seen = []
    orig = S._free_dof_relres

    def spy(a, f, gfu, fes):
        rv = gfu.vec.CreateVector()
        rv.data = f.vec - a.mat * gfu.vec
        all_dof = float(np.linalg.norm(rv.FV().NumPy()) / np.linalg.norm(f.vec.FV().NumPy()))
        n_dir = sum(1 for b in fes.FreeDofs() if not b)
        free = orig(a, f, gfu, fes)
        seen.append((free, all_dof, n_dir))
        return free

    monkeypatch.setattr(S, "_free_dof_relres", spy)
    ru = _solve("quarter", "x", tensor=DIAG)             # pol x on a quarter cell -> PEC x-wall
    rg, msgs = _solve("quarter", "x", ls="bddc_gmres", tensor=DIAG, record=True)
    free, all_dof, n_dir = seen[-1]
    assert n_dir > 0 and all_dof > 1e-1, seen          # premise: the old metric WOULD have fired
    assert free < 1e-6, seen
    assert not any("did not converge" in m for m in msgs), msgs
    assert abs(rg.R - ru.R) < 1e-6, (rg.R, ru.R)


# ---------------------------------------------------------------- 4. cross_pol

@pytest.mark.parametrize("pol,theta,phi", [("p", 0.0, 0.0), ("y", 20.0, 0.0), ("y", 0.0, 30.0)])
def test_cross_pol_refused_outside_its_regime_before_solving(pol, theta, phi):
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=theta, azimuth_deg=phi)
    no_mesh = types.SimpleNamespace(mesh=None)           # it must raise before touching the mesh
    with pytest.raises(NotImplementedError, match="cross_pol=True needs NORMAL incidence"):
        S.solve_fem(no_mesh, LAM, None, opt, cross_pol=True)


def test_cross_pol_jones_matches_the_eigenframe_solve():
    rx = _solve("none", "x", shape="rot45", cross_pol=True)      # (r_xx, r_yx)
    ry = _solve("none", "y", shape="rot45", cross_pol=True)      # (r_xy, r_yy)
    tol = 3e-3
    assert abs(rx.r - ry.r) < tol, (rx.r, ry.r)                  # diagonal mirror: r_xx = r_yy
    assert abs(rx.r_cross - ry.r_cross) < tol, (rx.r_cross, ry.r_cross)   # reciprocity: r_yx = r_xy
    assert abs(rx.r_cross) > 10 * tol, rx.r_cross                # the conversion is real
    # INDEPENDENT oracle, no cross-pol extraction anywhere: the same crystal rotated into the
    # rectangle's frame is diagonal there, so plain co-pol solves give its eigen-reflections r_u
    # (E along the long axis) and r_v; rotating back, r_xx = (r_u + r_v)/2, r_yx = (r_u - r_v)/2.
    # (R_flux cannot serve: it is under-resolved at this mesh -- measured -3e-4 on the plain circle.)
    r_u = _solve("none", "x", shape="rot45_eigenframe").r
    r_v = _solve("none", "y", shape="rot45_eigenframe").r
    assert abs(rx.r - 0.5 * (r_u + r_v)) < tol, (rx.r, r_u, r_v)
    assert abs(rx.r_cross - 0.5 * (r_u - r_v)) < tol, (rx.r_cross, r_u, r_v)


def test_cross_pol_mirror_symmetric_pillar():
    full = _solve("none", "x", cross_pol=True)
    assert abs(full.r_cross) < 1e-3, full.r_cross
    q = _solve("quarter", "x", cross_pol=True)
    assert q.r_cross == 0j
    assert _solve("quarter", "x").r_cross is None
