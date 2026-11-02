"""v0.12.0 FEM speed + PML gates.

  * FEMFactorCache / solve_fem_pols: x and y (normal incidence) and s and p (one oblique angle) share
    ONE factorization and match separate solves; a nearby wavelength is solved by GMRes
    preconditioned with the stored factorization, and a stalled one refactorizes; a symmetry cell
    (polarization-dependent walls) does NOT share.
  * linear_solver='auto' picks sparsecholesky / pardiso-or-umfpack / bddc_gmres by symmetry and size.
  * 'pardiso' without MKL falls back to umfpack; with MKL it matches umfpack.
  * gmres_rtol reaches ngsolve's GMRes as a RELATIVE tolerance (reltol=), not tol= (absolute there).
  * num_threads / DYNAMETA_FEM_THREADS pin ngsolve.SetNumThreads.
  * the default 1000 nm z-PML matches exact TMM on a flat metal mirror, where the old 400 nm did not.
"""
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("ngsolve")

import ngsolve as ng                                                            # noqa: E402

from dynameta.materials import Material, MaterialRegistry, ConstantOptical      # noqa: E402
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell          # noqa: E402
from dynameta.geometry.cross_section import centered_circle                      # noqa: E402
from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec                     # noqa: E402
from dynameta.core.eps_field import EpsField                                    # noqa: E402
from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder               # noqa: E402
from dynameta.optics.eps_assembler import assemble_eps_cf                       # noqa: E402
from dynameta.optics.tmm_reference import stack_rta                             # noqa: E402
import dynameta.optics.solver as S                                              # noqa: E402

LAM = 1300e-9
PER = 400e-9
GYRO = np.array([[2.6 ** 2, 0.3j, 0.0], [-0.3j, 2.6 ** 2, 0.0], [0.0, 0.0, 2.6 ** 2]])


def _design(symmetry="none"):
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0), ("glass", 1.5 ** 2), ("pil", 3.0 ** 2)):
        reg.add(Material(nm, ConstantOptical(complex(e))))
    cell = UnitCell.square(PER)
    stack = Stack(layers=[Layer("slab", 250e-9, "air",
                                inclusions=[Inclusion(shape=centered_circle(cell, 130e-9), material="pil")])],
                  superstrate_material="air", substrate_material="glass")
    m3 = Mesh3DSpec(pml_thk_m=500e-9, superstrate_buffer_m=500e-9, substrate_buffer_m=500e-9,
                    maxh_superstrate_m=120e-9, maxh_substrate_m=120e-9, maxh_background_m=75e-9,
                    maxh_inclusion_m=45e-9, fem_order=2, symmetry=symmetry)
    return Design(name="fem_speed", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)


_GEO = {}


def _geo(symmetry="none"):
    if symmetry not in _GEO:
        d = _design(symmetry)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # the opt-in symmetry hint
            _GEO[symmetry] = (d, LayeredOpticalBuilder(d).build())
    return _GEO[symmetry]


def _eps(d, geo, lam=LAM, pil_tensor=None):
    ebr = {r: EpsField(scalar=complex(d.materials.get(geo.material_by_region[r]).eps(lam)))
           for r in geo.mesh.GetMaterials()}
    if pil_tensor is not None:
        for r in geo.mesh.GetMaterials():
            if geo.material_by_region[r] == "pil":
                ebr[r] = EpsField(tensor=np.asarray(pil_tensor, complex))
    return assemble_eps_cf(geo, ebr)


def _opt(pol="x", ls="umfpack", theta=0.0, **kw):
    return OpticalSpec(polarization=pol, incidence_angle_deg=theta, linear_solver=ls, **kw)


def _fem(geo, eps, opt, lam=LAM, **kw):
    return S.solve_fem(geo, lam, eps, opt, order=2, n_super=1.0 + 0j, n_sub=1.5 + 0j,
                       diagnostics=False, **kw)


# ---------------------------------------------------------------- 1. one factorization, several pols

@pytest.mark.parametrize("ls", ["umfpack", "sparsecholesky"])
def test_xy_pols_share_one_factorization(ls):
    d, geo = _geo()
    eps = _eps(d, geo)
    cache = S.FEMFactorCache()
    out = S.solve_fem_pols(geo, LAM, eps, _opt(ls=ls), ("x", "y"), factor_cache=cache, order=2,
                           n_super=1.0 + 0j, n_sub=1.5 + 0j, diagnostics=False, cross_pol=True)
    assert cache.stats == {"factorizations": 1, "exact_reuses": 1, "preconditioned": 0, "stalled": 0}
    for p in ("x", "y"):
        ref = _fem(geo, eps, _opt(pol=p, ls=ls), cross_pol=True)
        assert abs(out[p].r - ref.r) <= 1e-12
        assert abs(out[p].r_cross - ref.r_cross) <= 1e-12
        assert out[p].linear_solver == ls


def test_oblique_s_and_p_share_one_factorization():
    d, geo = _geo()
    eps = _eps(d, geo)
    cache = S.FEMFactorCache()
    out = S.solve_fem_pols(geo, LAM, eps, _opt(pol="y", theta=20.0), ("y", "p"), factor_cache=cache,
                           order=2, n_super=1.0 + 0j, n_sub=1.5 + 0j, diagnostics=False)
    assert cache.stats["factorizations"] == 1 and cache.stats["exact_reuses"] == 1
    for p in ("y", "p"):
        ref = _fem(geo, eps, _opt(pol=p, theta=20.0))
        assert abs(out[p].r - ref.r) <= 1e-12


def test_symmetry_cell_pols_do_not_share_a_factorization():
    # the PEC/PMC walls follow the polarization, so x and y are different spaces -- sharing would be wrong
    d, geo = _geo("quarter")
    eps = _eps(d, geo)
    cache = S.FEMFactorCache()
    out = S.solve_fem_pols(geo, LAM, eps, _opt(), ("x", "y"), factor_cache=cache, order=2,
                           n_super=1.0 + 0j, n_sub=1.5 + 0j, diagnostics=False)
    assert cache.stats["factorizations"] == 2 and cache.stats["exact_reuses"] == 0
    for p in ("x", "y"):
        assert abs(out[p].r - _fem(geo, eps, _opt(pol=p)).r) <= 1e-12


def test_solve_fem_pols_rejects_duplicate_or_empty_pols():
    d, geo = _geo()
    for bad in ((), ("x", "x")):
        with pytest.raises(ValueError):
            S.solve_fem_pols(geo, LAM, _eps(d, geo), _opt(), bad)


# ---------------------------------------------------------------- 2. factorization as preconditioner

@pytest.mark.parametrize("ls", ["umfpack", "sparsecholesky"])
def test_nearby_wavelengths_solve_preconditioned_by_the_stored_factorization(ls):
    d, geo = _geo()
    cache = S.FEMFactorCache()
    lams = (LAM, LAM + 10e-9, LAM + 50e-9)
    got = [_fem(geo, _eps(d, geo, lam), _opt(ls=ls), lam=lam, factor_cache=cache).r for lam in lams]
    assert cache.stats["factorizations"] == 1 and cache.stats["preconditioned"] == 2
    for lam, r in zip(lams, got):
        assert abs(r - _fem(geo, _eps(d, geo, lam), _opt(ls=ls), lam=lam).r) <= 1e-9


def test_stalled_preconditioner_refactorizes():
    d, geo = _geo()
    cache = S.FEMFactorCache(max_its=1)          # one GMRes step cannot reach the 1e-8 acceptance
    for lam in (LAM, LAM + 200e-9):
        r = _fem(geo, _eps(d, geo, lam), _opt(), lam=lam, factor_cache=cache).r
    assert cache.stats["stalled"] == 1 and cache.stats["factorizations"] == 2
    assert abs(r - _fem(geo, _eps(d, geo, LAM + 200e-9), _opt(), lam=LAM + 200e-9).r) <= 1e-12


def test_cache_clears_on_a_different_mesh():
    d, geo = _geo()
    dq, geoq = _geo("quarter")
    cache = S.FEMFactorCache()
    _fem(geo, _eps(d, geo), _opt(), factor_cache=cache)
    _fem(geoq, _eps(dq, geoq), _opt(), factor_cache=cache)
    assert cache.stats["factorizations"] == 1 and cache._geo is geoq   # stats restart with the new mesh


def test_sweep_solver_reuses_the_factorization():
    d, geo = _geo()
    d.optical = _opt(ls="sparsecholesky")
    solver = S.make_fem_optical_solver(diagnostics=False)
    lams = [LAM, LAM + 20e-9]

    def assemble_at(lam):
        return {r: EpsField(scalar=complex(d.materials.get(geo.material_by_region[r]).eps(lam)))
                for r in geo.mesh.GetMaterials()}
    res = solver.solve_sweep(d, geo, assemble_at, lams, 1.0 + 0j, 1.5 + 0j)
    assert solver.factor_cache.stats["factorizations"] == 1
    assert solver.factor_cache.stats["preconditioned"] == 1
    for lam, rr in zip(lams, res):
        assert abs(rr.r - _fem(geo, _eps(d, geo, lam), _opt(ls="sparsecholesky"), lam=lam).r) <= 1e-9


# ---------------------------------------------------------------- 3. linear_solver='auto'

def test_auto_picks_by_symmetry_and_size():
    d, geo = _geo()
    eps = _eps(d, geo)
    ref = _fem(geo, eps, _opt()).r
    r_sym = _fem(geo, eps, _opt(ls="auto"))
    assert r_sym.linear_solver == "sparsecholesky" and abs(r_sym.r - ref) <= 1e-10
    r_big = _fem(geo, eps, _opt(ls="auto", auto_direct_max_dofs=0))
    assert r_big.linear_solver == "bddc_gmres" and abs(r_big.r - ref) <= 1e-6
    # oblique (Bloch-phased, Hermitian-not-symmetric) and a gyrotropic tensor are NOT complex-symmetric
    nonsym = "pardiso" if S.pardiso_available() else "umfpack"
    r_obl = _fem(geo, eps, _opt(pol="y", ls="auto", theta=20.0))
    assert r_obl.linear_solver == nonsym
    assert abs(r_obl.r - _fem(geo, eps, _opt(pol="y", theta=20.0)).r) <= 1e-10
    eps_g = _eps(d, geo, pil_tensor=GYRO)
    r_gyro = _fem(geo, eps_g, _opt(ls="auto"))
    assert r_gyro.linear_solver == nonsym
    assert abs(r_gyro.r - _fem(geo, eps_g, _opt()).r) <= 1e-10


def test_auto_umfpack_cap_without_mkl(monkeypatch):
    monkeypatch.setattr(S, "pardiso_available", lambda: False)
    opt = _opt(ls="auto")
    assert S._auto_linear_solver(30_000, False, opt) == "umfpack"
    assert S._auto_linear_solver(60_000, False, opt) == "bddc_gmres"     # umfpack's ~81 kB/DOF
    assert S._auto_linear_solver(60_000, True, opt) == "sparsecholesky"
    assert S._auto_linear_solver(90_000, True, opt) == "bddc_gmres"


# ---------------------------------------------------------------- 4. pardiso

def test_pardiso_without_mkl_falls_back_to_umfpack(monkeypatch):
    monkeypatch.setattr(S, "pardiso_available", lambda: False)
    S._ADVISED_ONCE.discard("pardiso_unavailable")
    d, geo = _geo()
    eps = _eps(d, geo)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = _fem(geo, eps, _opt(ls="pardiso"))
    assert res.linear_solver == "umfpack"
    assert any("needs Intel MKL" in str(x.message) for x in w)
    assert abs(res.r - _fem(geo, eps, _opt()).r) <= 1e-13


@pytest.mark.skipif(not S.pardiso_available(), reason="MKL-PARDISO not installed (pip install mkl)")
def test_pardiso_matches_umfpack():
    d, geo = _geo()
    eps = _eps(d, geo)
    for opt_kw in (dict(), dict(pol="y", theta=20.0)):
        res = _fem(geo, eps, _opt(ls="pardiso", **opt_kw))
        assert res.linear_solver == "pardiso"
        assert abs(res.r - _fem(geo, eps, _opt(**opt_kw)).r) <= 1e-10


# ---------------------------------------------------------------- 5. GMRes tolerance + threads

def test_gmres_rtol_is_passed_as_a_relative_tolerance(monkeypatch):
    seen = []
    real = ng.solvers.GMRes

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(ng.solvers, "GMRes", spy)
    d, geo = _geo()
    _fem(geo, _eps(d, geo), _opt(ls="bddc_gmres", gmres_rtol=1e-7))
    assert seen and seen[0].get("reltol") == 1e-7 and "tol" not in seen[0]


def test_num_threads_pins_ngsolve(monkeypatch):
    calls = []
    monkeypatch.setattr(ng, "SetNumThreads", lambda n: calls.append(n))
    S._apply_fem_threads(_opt(num_threads=3))
    monkeypatch.setenv("DYNAMETA_FEM_THREADS", "5")
    S._apply_fem_threads(_opt())
    S._apply_fem_threads(_opt(num_threads=2))            # the spec wins over the environment
    assert calls == [3, 5, 2]
    monkeypatch.setenv("DYNAMETA_FEM_THREADS", "zero")
    with pytest.raises(ValueError):
        S._apply_fem_threads(_opt())
    monkeypatch.delenv("DYNAMETA_FEM_THREADS")
    S._apply_fem_threads(_opt())                         # nothing set -> NGSolve's default, no call
    assert calls == [3, 5, 2]


def test_optical_spec_validates_new_fields():
    for bad in (dict(num_threads=0), dict(auto_direct_max_dofs=-1), dict(gmres_rtol=0.0)):
        with pytest.raises(ValueError):
            OpticalSpec(**bad)
    assert OpticalSpec().linear_solver == "auto"


# ---------------------------------------------------------------- 6. the default PML vs exact TMM

def _mirror_dR(pml_thk_m, lam):
    """|R_fem - R_tmm| for a flat 100 nm metal film (eps -90+10j, R ~ 0.977) on glass, x-pol, with
    every Mesh3DSpec default except the PML thickness (a flat cell: no evanescent orders, so this
    isolates the PML's back-reflection)."""
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0), ("glass", 1.5 ** 2), ("film", -90 + 10j)):
        reg.add(Material(nm, ConstantOptical(complex(e))))
    st = Stack(layers=[Layer("film", 100e-9, "film")], superstrate_material="air", substrate_material="glass")
    d = Design(name="pml", unit_cell=UnitCell.square(300e-9), stack=st, electrodes=[], materials=reg,
               mesh_3d=Mesh3DSpec(pml_thk_m=pml_thk_m))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geo = LayeredOpticalBuilder(d).build()
    ebr = {r: EpsField(scalar=complex(reg.get(geo.material_by_region[r]).eps(lam)))
           for r in geo.mesh.GetMaterials()}
    res = S.solve_fem(geo, lam, assemble_eps_cf(geo, ebr), _opt(), order=2, n_super=1.0 + 0j,
                      n_sub=1.5 + 0j, diagnostics=False)
    R_tmm = stack_rta(1.0, [(complex(np.sqrt(-90 + 10j)), 100e-9)], 1.5, lam)[0]
    return abs(res.R - R_tmm)


def test_default_pml_matches_tmm_and_the_old_default_did_not():
    assert Mesh3DSpec().pml_thk_m == pytest.approx(1000e-9)
    lam = 1550e-9
    new = _mirror_dR(Mesh3DSpec().pml_thk_m, lam)
    old = _mirror_dR(400e-9, lam)
    # the old default is LOAD-BEARINGLY wrong here (5.2e-2 measured), the new one clean (7.5e-6)
    assert old > 1e-2, old
    assert new < 2e-3, new
    assert math.isfinite(new)
