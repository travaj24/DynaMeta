"""Unit coverage for the solver anti-silent-failure GUARDS (the hardening pass that mirrors the
refractiveindex.info out-of-range guard): each checks that a solver RAISES / WARNS on an input
that previously returned a silently-wrong / NaN / unphysical number, and that a valid input is
unaffected (no false-fire). Run: python -m pytest tests/test_solver_guards.py -q

The Schrodinger-Poisson guards are pure numpy/scipy and always run. The TMM-oracle and DEVSIM-
physics guards need the optional `tmm` / `devsim` solvers and skip when those are absent (CI).
"""
import warnings

import numpy as np
import pytest

from dynameta.constants import Q_E


# ----------------------------- Schrodinger-Poisson (no optional dep) -----------------------------

def test_sp_constructor_rejects_nonpositive_mass_and_temperature():
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
    z = np.linspace(0.0, 10e-9, 64)
    with pytest.raises(ValueError):
        SchrodingerPoisson1D(z, -0.2 * 9.109e-31)          # negative mass inverts the kinetic op
    with pytest.raises(ValueError):
        SchrodingerPoisson1D(z, 0.2 * 9.109e-31, T_K=0.0)  # T=0 -> 0*inf NaN occupation
    SchrodingerPoisson1D(z, 0.2 * 9.109e-31)               # valid: no raise


def test_sp_solve_rejects_nonpositive_node_mass():
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
    z = np.linspace(0.0, 10e-9, 64)
    sp = SchrodingerPoisson1D(z, 0.2 * 9.109e-31)
    bad = np.full_like(z, 0.2 * 9.109e-31); bad[10] = -1.0   # one node negative
    with pytest.raises(ValueError):
        sp.solve_schrodinger(np.zeros_like(z), m_eff_z_kg=bad)


def test_sp_density_warns_on_eigenstate_truncation():
    """A deep box with E_F far above the n_states-th sub-band silently UNDER-counts the density;
    the completeness guard must warn. A large n_states (top state above E_F) must NOT warn."""
    from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
    z = np.linspace(0.0, 150e-9, 200)                       # wide slab -> many sub-bands below E_F
    sp = SchrodingerPoisson1D(z, 0.2 * 9.109e-31)
    U = np.zeros_like(z)
    E_F = 0.5 * Q_E                                         # ~0.5 eV: ~100 sub-bands occupied

    with pytest.warns(UserWarning, match="truncates"):
        sp.density(U, E_F, n_states=20, bound_tol=1e9)      # only 20 solved -> top state << E_F

    with warnings.catch_warnings(record=True) as rec:       # enough states -> top above E_F
        warnings.simplefilter("always")
        sp.density(U, E_F, n_states=150, bound_tol=1e9)
    assert not any("truncates" in str(w.message) for w in rec)


# ----------------------------- TMM reference oracle (needs `tmm`) -----------------------------

def test_tmm_interior_gain_slab_raises():
    pytest.importorskip("tmm")
    from dynameta.optics.tmm_reference import stack_rta
    # interior slab with Im(eps)<0 == GAIN (a sign-convention mistake): tmm only guards the END
    # media, so it returns T>1, A<0 silently -> the energy-budget guard must raise.
    n_gain = np.sqrt(complex(2.0, -0.1))                    # Im(eps)<0
    with pytest.raises(ValueError, match="energy budget"):
        stack_rta(1.0, [(n_gain, 250e-9)], 1.0, 1550e-9)


def test_tmm_lossy_superstrate_and_theta_raise():
    pytest.importorskip("tmm")
    from dynameta.optics.tmm_reference import stack_rta
    with pytest.raises(ValueError, match="LOSSLESS incidence"):
        stack_rta(complex(1.5, 0.05), [(2.0, 250e-9)], 1.0, 1550e-9)
    with pytest.raises(ValueError, match=r"\[0, 90\)"):
        stack_rta(1.0, [(2.0, 250e-9)], 1.0, 1550e-9, theta_deg=95.0)


def test_tmm_passive_stack_unaffected():
    pytest.importorskip("tmm")
    from _rta_oracles import abeles_rta
    from dynameta.optics.tmm_reference import stack_rta
    # a normal passive (lossy-but-physical) stack must still return a clean, PHYSICAL R/T/A --
    # i.e. the gain guard does not fire and the numbers are right.
    layers = [(complex(1.5, 0.02), 200e-9)]
    R, T, A = stack_rta(1.0, layers, 1.5, 1550e-9)
    assert R >= 0 and T >= 0 and A >= -1e-9
    # AUDIT T-1: `R + T + A == 1` was an IDENTITY here (stack_rta returns A := 1 - R - T), so it
    # passed for any R/T whatsoever -- exactly the silently-wrong number this module exists to
    # catch. The stack is LOSSY, so there is no `abs(A) < tol` physics anchor; gate all three
    # against the INDEPENDENT Abeles TMM (tests/_rta_oracles.py, not the `tmm` package) instead.
    R_ref, T_ref, A_ref = abeles_rta(1.0, layers, 1.5, 1550e-9)
    assert R == pytest.approx(R_ref, abs=1e-12)
    assert T == pytest.approx(T_ref, abs=1e-12)
    assert A == pytest.approx(A_ref, abs=1e-12)
    assert A_ref > 1e-3                                    # the oracle agrees it genuinely absorbs


# ----------------------------- DEVSIM carrier physics (needs `devsim`) -----------------------------

def test_invert_F12_out_of_bracket_and_convergence():
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_equilibrium import invert_F12, F12_aymerich_humet
    f_hi = F12_aymerich_humet(80.0)
    with pytest.raises(ValueError, match="outside the solver bracket"):
        invert_F12(2.0 * f_hi)                              # target > F_1/2(eta_max) -> raise
    # a normal degenerate target (ITO: n/N_c ~ 77) still inverts to its known eta ~ 21.9
    eta = invert_F12(77.0)
    assert 21.0 < eta < 23.0


def test_require_positive_guard():
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_equilibrium import require_positive
    require_positive(eps_static=4.0, n_bg_m3=4e26)          # valid: no raise
    for bad in (-1.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            require_positive(tau_n_s=bad)


def test_solve_fem_sourced_rejects_lossy_superstrate():
    """AUDIT F-3. solve_fem_sourced's radiated power p_up = |a|^2 (Re kz/k0)/(2 Z0) A is the
    plane-wave flux of a LOSSLESS medium -- its own docstring calls that "the real limit of the
    power read-out" -- but it never enforced it: a lossy superstrate inflated p_up by 5.03x at
    n_super = 1+0.5j and 66.5x at 1+2j. solve_fem RAISES on exactly that input
    (_incidence_geometry), so the sourced sibling must too. A DENSE but lossless superstrate is
    unaffected (it must get past the guard).

    A bare geo object is used deliberately: the guard has to fire BEFORE any mesh/assembly work,
    so the lossy calls must raise NotImplementedError while the lossless one gets far enough to
    trip over the missing geometry."""
    pytest.importorskip("ngsolve")             # optics.solver imports ngsolve at module load
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.solver import solve_fem_sourced

    class _BareGeo:
        pass

    opt = OpticalSpec(polarization="p", incidence_angle_deg=0.0, linear_solver="umfpack")
    for n_super in (complex(1.0, 0.5), complex(1.0, 2.0), complex(1.5, 1e-3)):
        with pytest.raises(NotImplementedError, match="LOSSLESS superstrate"):
            solve_fem_sourced(_BareGeo(), 1e-6, None, opt, n_super=n_super)
    # lossless (even dense) superstrate: past the guard, into the geometry
    with pytest.raises(AttributeError):
        solve_fem_sourced(_BareGeo(), 1e-6, None, opt, n_super=complex(1.5, 0.0))


def test_shg_structured_fundamental_inherits_lossy_superstrate_guard():
    """AUDIT F-3, part 2: the accidental-protection chain. shg_two_step / sfg_two_step call
    solve_fem first (which raises on a lossy superstrate), but shg_structured_two_step does NOT --
    its fundamental goes through _reconstruct_fundamental_field -> solve_fem_sourced, so a lossy
    n_super used to pass end to end. With the guard hoisted into solve_fem_sourced the structured
    driver's reconstruction inherits it. (Stub geometry: the analytic Fresnel background is built
    from scalars/CoefficientFunctions only, so this reaches the sourced call without a mesh.)"""
    pytest.importorskip("ngsolve")
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.shg_fem import _reconstruct_fundamental_field

    class _StubGeo:                            # only what the reconstruction touches pre-solve
        z_intervals_nm = {"substrate": (0.0, 100.0)}
        z_sub_interface_nm = 100.0

    opt = OpticalSpec(polarization="p", incidence_angle_deg=20.0, linear_solver="umfpack")
    with pytest.raises(NotImplementedError, match="LOSSLESS superstrate"):
        _reconstruct_fundamental_field(_StubGeo(), None, 1000e-9, opt, None, 2,
                                       complex(1.0, 0.5), complex(1.0, 0.0), 20.0, "p")


def test_bloch_z_samples_cover_every_region():
    """AUDIT F-6. _detect_bloch_dirs classifies each periodic identification by toggling a marker
    phase and probing the x-/y-boundary jump at a list of z samples. With 18 GLOBAL stack fractions
    only, a thin layer gets ZERO samples (a 5 nm layer in a 2.3 um stack falls between two global
    samples) -- its idnr is then classified from numerical noise and the x/y-count assertion fires
    ('resolved N x / M y, expected ...') on a supported stack. The cure (per-region sampling) used
    to live in shg_fem._ensure_bloch_dirs, wired into exactly ONE of 5+ call sites; it now lives in
    the solver's own sampler, so every call site gets it."""
    pytest.importorskip("ngsolve")
    from dynameta.optics.solver import _bloch_z_samples

    class _StubGeo:                            # air/PML + a 5 nm device layer, in nm
        z_intervals_nm = {"pml_bot": (0.0, 500.0), "substrate": (500.0, 1100.0),
                          "thin": (1100.0, 1105.0), "lb": (1105.0, 1225.0),
                          "superstrate": (1225.0, 1825.0), "pml_top": (1825.0, 2325.0)}

    geo = _StubGeo()
    zs = _bloch_z_samples(geo)
    for name, (a, b) in geo.z_intervals_nm.items():
        assert any(a < z < b for z in zs), "region {!r} gets no z sample".format(name)
    # the trigger is real: the legacy global-fraction list misses BOTH device layers
    legacy = [0.0 + f * 2325.0 for f in np.linspace(0.03, 0.97, 18)]
    assert not any(1100.0 < z < 1105.0 for z in legacy)
    assert not any(1105.0 < z < 1225.0 for z in legacy)


def test_ensure_bloch_dirs_delegates_to_the_solver_detector(monkeypatch):
    """AUDIT F-6, part 2: shg_fem._ensure_bloch_dirs is now a thin pre-cache over the SOLVER's
    detector (one implementation, every call site), and it keeps its pre-existing forgiving
    semantics -- an inconclusive detection is swallowed HERE and left for the solve path to
    diagnose."""
    pytest.importorskip("ngsolve")
    from dynameta.optics import shg_fem, solver

    seen = []

    def _fake(geo):
        seen.append(geo)
        geo._bloch_dirs = ["x", "y"]
        return ["x", "y"]

    monkeypatch.setattr(solver, "_detect_bloch_dirs", _fake)

    class _G:
        pass

    g = _G()
    shg_fem._ensure_bloch_dirs(g)
    assert seen and g._bloch_dirs == ["x", "y"]
    shg_fem._ensure_bloch_dirs(g)                       # already cached -> no second detection
    assert len(seen) == 1

    def _boom(geo):
        raise RuntimeError("detection inconsistent")

    monkeypatch.setattr(solver, "_detect_bloch_dirs", _boom)
    g2 = _G()
    shg_fem._ensure_bloch_dirs(g2)                      # must NOT raise here
    assert getattr(g2, "_bloch_dirs", None) is None


def test_probe_grid_sizes_kill_aliased_orders():
    # audit C3-1: the N-point cell-centred grid aliases orders m = 0 (mod N) with weight
    # (-1)^(m/N); the size helper must make the first aliased order evanescent, and the
    # alias-weight identity itself is pinned here in pure numpy
    pytest.importorskip("ngsolve")             # optics.solver imports ngsolve at module load
    import numpy as np
    from dynameta.optics.solver import _probe_grid_sizes

    # alias-weight identity D(m): 0 for m != 0 (mod N), (-1)^(m/N) otherwise
    def D(m, N):
        j = np.arange(N)
        return np.mean(np.exp(2j * np.pi * m * (j + 0.5) / N))
    for N in (6, 9):
        for m in range(1, N):
            assert abs(D(m, N)) < 1e-12
        assert D(N, N) == pytest.approx((-1.0) ** 1, abs=1e-12)
        assert D(2 * N, N) == pytest.approx(1.0, abs=1e-12)

    lam, n_sub = 1550.0, 3.48                                 # nm, silicon substrate
    k0 = 2.0 * np.pi / lam
    # sub-wavelength cell: legacy 6x6 (byte-identical envelope)
    assert _probe_grid_sizes(300.0, 300.0, 0.0, 0.0, n_sub * k0) == (6, 6)
    # the audit failure case: Px = 3 um on Si at 1550 -- order 6 PROPAGATES; the sized
    # grid must exceed Px*n/lam ~ 6.7 so its first alias (m = N) is evanescent
    nx, ny = _probe_grid_sizes(3000.0, 300.0, 0.0, 0.0, n_sub * k0)
    assert nx >= 7 and ny == 6
    assert nx * 2.0 * np.pi / 3000.0 > n_sub * k0             # first alias evanescent
    # oblique: |kx| tightens the bound
    kx = k0 * np.sin(np.radians(60.0))
    nx_o, _ = _probe_grid_sizes(3000.0, 300.0, kx, 0.0,
                                np.sqrt((n_sub * k0) ** 2 - kx ** 2))
    assert nx_o > nx


# ------------------- FEM solve-path guards (audit F-8 / F-9 / F-15; needs ngsolve) -------------------
# One small gold/air cell drives all the gates below.

def _fem_cell(theta=0.0, pol="y"):
    """A 2-layer gold/air cell on a coarse-but-CONVERGED mesh (the audit's own p20 census cell at
    0.6x maxh: R = 0.9595 there vs 0.9590 at 0.4x, and it emits zero quality warnings). Returns
    (geo, eps_cf, OpticalSpec)."""
    pytest.importorskip("ngsolve")
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Design
    from dynameta.geometry.specs import Mesh3DSpec, OpticalSpec
    from dynameta.core.eps_field import EpsField
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
    from dynameta.optics.eps_assembler import assemble_eps_cf
    eps_m = complex(-40.0, 2.5)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("gold", ConstantOptical(eps_m)))
    stack = Stack(layers=[Layer("metalL", 200e-9, "gold"), Layer("capL", 300e-9, "air")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=400e-9, superstrate_buffer_m=400e-9, substrate_buffer_m=300e-9,
                    maxh_superstrate_m=132e-9, maxh_substrate_m=132e-9, maxh_pml_m=240e-9,
                    maxh_inclusion_m=108e-9, maxh_background_m=108e-9, maxh_metal_m=108e-9)
    opt = OpticalSpec(polarization=pol, incidence_angle_deg=theta, linear_solver="umfpack")
    d = Design(name="guards", unit_cell=UnitCell.square(400e-9), stack=stack, electrodes=[],
               materials=reg, mesh_3d=m3, optical=opt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geo = LayeredOpticalBuilder(d).build()
    ebr = {rg: EpsField(scalar={"air": 1.0 + 0j, "gold": eps_m}[geo.material_by_region[rg]])
           for rg in geo.mesh.GetMaterials()}
    return geo, assemble_eps_cf(geo, ebr), opt


def test_f8_unmatched_sheet_bc_name_raises_and_lists_the_interfaces():
    """A sheet_bcs key matching no boundary used to assemble ||f|| = 0 -- the solve silently
    returned the SHEET-FREE answer. It must raise, naming the interfaces that DO exist."""
    import re as _re
    geo, eps_cf, opt = _fem_cell()
    from dynameta.optics.solver import solve_fem, _validate_sheet_bcs
    ifaces = [b for b in geo.mesh.GetBoundaries() if b.startswith("iface_z")]
    assert ifaces, "the cell must expose at least one named interior interface"

    with pytest.raises(ValueError, match="match no boundary"):
        solve_fem(geo, 1200e-9, eps_cf, opt, order=1, sheet_bcs={"iface_z999": 1e-4})
    # the message must be actionable: it lists the real names
    with pytest.raises(ValueError, match=_re.escape(ifaces[0])):
        _validate_sheet_bcs(geo.mesh, {"iface_z999": 1e-4})
    # an alternation must not pass on the strength of its good half
    with pytest.raises(ValueError, match="iface_z999"):
        _validate_sheet_bcs(geo.mesh, {ifaces[0] + "|iface_z999": 1e-4})
    # no false fire: a real name, an alternation of real names, and a regex all validate
    _validate_sheet_bcs(geo.mesh, {ifaces[0]: 1e-4})
    _validate_sheet_bcs(geo.mesh, {"|".join(ifaces): 1e-4})
    _validate_sheet_bcs(geo.mesh, {"iface_z.*": 1e-4})


def test_f8_unmatched_boundary_really_assembles_to_zero():
    """Negative control: NGSolve itself does not object -- a linear form over an unknown boundary
    assembles cleanly with ||f|| = 0, which is why the guard has to exist."""
    import ngsolve as ng
    geo, _eps, _opt = _fem_cell()
    fes = ng.Periodic(ng.HCurl(geo.mesh, order=1, complex=True, dirichlet=""))
    v = fes.TestFunction()
    f = ng.LinearForm(fes)
    f += (ng.CoefficientFunction((1.0, 0.0, 0.0)) * v.Trace()) * \
        ng.ds(definedon=geo.mesh.Boundaries("iface_z999"))
    f.Assemble()
    assert float(np.linalg.norm(f.vec.FV().NumPy())) == 0.0


def test_f15_oblique_pml_advisory_is_emitted_once_per_process():
    """The advisory describes the PML, not the solve: a sweep must not repeat it per wavelength.
    The per-solve REGIME GUARD (the 50 deg raise) stays unconditional."""
    from dynameta.optics import solver as S
    S._ADVISED_ONCE.discard("oblique_pml")
    opt = type("O", (), {"incidence_angle_deg": 30.0, "azimuth_deg": 0.0,
                         "polarization": "y", "incidence_side": "top"})()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(5):
            S._incidence_geometry(opt, 1.0 + 0j)
    hits = [w for w in rec if "not angle-aware" in str(w.message)]
    assert len(hits) == 1, "advisory fired {} times for 5 solves".format(len(hits))
    assert issubclass(hits[0].category, S.FEMDiagnosticWarning)
    # the GUARD is not de-duplicated: every over-cap solve still raises
    opt.incidence_angle_deg = 60.0
    for _ in range(2):
        with pytest.raises(NotImplementedError, match="validated envelope"):
            S._incidence_geometry(opt, 1.0 + 0j)


def test_f15_fit_warnings_are_aggregated_into_one_per_solve():
    """Four bad bands in one solve used to emit four near-identical warnings. They must arrive as
    ONE warning naming every band; outside a solve the immediate warning is preserved."""
    from dynameta.optics import solver as S
    M = np.column_stack([np.ones(7), np.linspace(0.0, 1.0, 7)])
    Es = np.linspace(0.0, 1.0, 7) ** 3 + 0.7                    # cubic: not a two-wave field

    S._fit_ctx.acc = acc = []                                   # inside a solve: accumulate only
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for band in ("reflection", "transmission", "p-pol reflection", "p-pol transmission"):
                S._lstsq_2wave(M, Es, where=band)
        assert not rec, "no warning may escape while a solve is accumulating"
    finally:
        S._fit_ctx.acc = None
    bad = [(w, v) for w, v in acc if v > S._FIT_RELRES_WARN]
    assert len(bad) == 4
    text = S._fit_warn_text(bad)
    assert "4 bands" in text and all(b in text for b, _ in bad)

    with warnings.catch_warnings(record=True) as rec:            # outside a solve: warns at once
        warnings.simplefilter("always")
        S._lstsq_2wave(M, Es, where="reflection")
    assert len(rec) == 1 and issubclass(rec[0].category, S.FEMDiagnosticWarning)


def test_f15_ordinary_solve_is_quiet_and_reports_its_fit_quality():
    """The threshold's "validated cases do not false-fire" claim, MEASURED: a converged coarse
    solve emits ZERO diagnostics and carries a fit residual far below the threshold."""
    from dynameta.optics import solver as S
    geo, eps_cf, opt = _fem_cell()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = S.solve_fem(geo, 1200e-9, eps_cf, opt, order=2)
    diag = [w for w in rec if issubclass(w.category, S.FEMDiagnosticWarning)]
    assert not diag, [str(w.message) for w in diag]
    assert res.R == pytest.approx(0.959, abs=0.015)              # opaque 200 nm gold film
    assert 0.0 < res.fit_relres < 0.1 * S._FIT_RELRES_WARN


def test_f9_quasiperiodic_matrix_is_hermitian_not_symmetric():
    """The fact behind F-9: ng.Periodic(..., phase=...) conjugates the phase on the TEST side, so
    even a REAL symmetric integrand assembles Hermitian-not-symmetric -- `symmetric=True` on that
    space is a false statement (inert in ngsolve 6.2.2604, but it must not be asserted)."""
    import ngsolve as ng
    from dynameta.optics import solver as S
    geo, _eps, _opt = _fem_cell(theta=30.0, pol="y")
    kx = 2.0 * np.pi / 1200.0 * np.sin(np.radians(30.0))
    fes = ng.Periodic(ng.HCurl(geo.mesh, order=1, complex=True, dirichlet=""),
                      phase=S._bloch_phase_list(geo, kx, 0.0))
    u, v = fes.TrialFunction(), fes.TestFunction()
    a = ng.BilinearForm(fes, symmetric=False)
    a += (ng.curl(u) * ng.curl(v) + u * v) * ng.dx               # REAL symmetric integrand
    a.Assemble()
    A = np.array(a.mat.ToDense())
    scale = float(np.abs(A).max())
    assert np.abs(A - A.T).max() / scale > 1e-3, "expected NON-symmetric"
    assert np.abs(A - A.conj().T).max() / scale < 1e-10, "expected Hermitian"


def test_f9_bddc_cg_falls_back_on_a_bloch_phased_space():
    """`ng.solvers.CGSolver` runs COCG (complex-SYMMETRIC pseudo inner product), which has no
    convergence theory on the non-symmetric oblique matrix. The documented option must fall back
    to the GMRes route (once per process) rather than iterate an invalid Krylov method."""
    from dynameta.optics import solver as S
    geo, eps_cf, opt = _fem_cell(theta=30.0, pol="y")
    S._ADVISED_ONCE.discard("bddc_cg_nonsymmetric")
    S._ADVISED_ONCE.discard("oblique_pml")
    opt.linear_solver = "bddc_cg"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = S.solve_fem(geo, 1200e-9, eps_cf, opt, order=1)
    hits = [w for w in rec if "bddc_cg" in str(w.message)]
    assert len(hits) == 1 and issubclass(hits[0].category, S.FEMDiagnosticWarning)
    assert res.R is not None and np.isfinite(res.R)
