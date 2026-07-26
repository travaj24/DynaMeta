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


def test_f10_probe_grid_alias_is_DECAYED_not_merely_evanescent():
    """AUDIT F-10. The C3-1 sizing only made the first aliased order EVANESCENT, with a margin of at
    most one grid step: for P (n k0 + kx)/(2 pi) just under an integer the margin is ~0.2%, so
    kappa ~ 0.06 n k0 and at lambda = 1550 nm the alias survives the 50 nm probe standoff at
    exp(-0.012) = 0.988 -- i.e. it is aliased into the reported 0-order coefficient essentially
    UNDAMPED, and "evanescent" was a vacuous guarantee there.

    This pins the failure case AND the fix: the sizing now takes the standoff and picks the
    smallest N that suppresses the alias by e^-3 at the nearest probe plane -- up to the
    `_PROBE_GRID_MAX` cost cap, past which it warns instead (see the residual test below)."""
    pytest.importorskip("ngsolve")
    import numpy as np
    import warnings as _w
    from dynameta.optics.solver import (_ALIAS_DECAY_NEPERS, _PROBE_GRID_MAX, _probe_grid_sizes)

    lam, standoff = 1550.0, 50.0                              # nm
    k0 = 2.0 * np.pi / lam                                    # vacuum, n = 1

    def kappa(N, Px, kx=0.0):                                 # decay const of the first alias
        k_lat = N * 2.0 * np.pi / Px - abs(kx)
        return float(np.sqrt(max(k_lat ** 2 - k0 ** 2, 0.0)))

    # the near-cutoff cell: P n k0 / (2 pi) a hair under the integer 7
    Px = (7.0 - 1e-3) * 2.0 * np.pi / k0
    n_old, _ = _probe_grid_sizes(Px, Px, 0.0, 0.0, k0)                       # legacy rule
    assert kappa(n_old, Px) > 0.0                                            # evanescent: yes
    assert np.exp(-kappa(n_old, Px) * standoff) > 0.98                       # decayed: NO (0.988)

    # This 10.8 um cell at a 50 nm standoff is exactly the runaway the cap exists for: the criterion
    # asks for ~111 points per direction (12000+ point evaluations per probe plane), so it is CAPPED
    # and says so. The alias is then NOT suppressed to e^-3, which is the documented tradeoff.
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        n_cap, _ = _probe_grid_sizes(Px, Px, 0.0, 0.0, k0, standoff_nm=standoff)
    assert n_cap == _PROBE_GRID_MAX and n_cap > n_old
    assert any("F-10" in str(x.message) for x in caught)

    # ... and where the requirement FITS under the cap, the criterion is met exactly and minimally.
    # (Thickening the buffer is what makes it fit -- the requirement goes as ~1/standoff.)
    s_big = 2200.0
    with _w.catch_warnings():
        _w.simplefilter("error")
        n_new, _ = _probe_grid_sizes(Px, Px, 0.0, 0.0, k0, standoff_nm=s_big)
    assert n_old < n_new <= _PROBE_GRID_MAX
    assert kappa(n_new, Px) * s_big >= _ALIAS_DECAY_NEPERS
    assert np.exp(-kappa(n_new, Px) * s_big) <= np.exp(-_ALIAS_DECAY_NEPERS)
    # SMALLEST such N (the sizing must not over-refine: one step down misses the criterion)
    assert kappa(n_new - 1, Px) * s_big < _ALIAS_DECAY_NEPERS

    # oblique tightens it further (the alias sits at N*2pi/P - |kx|), and a lossy/evanescent probe
    # medium (complex kz_med) is handled through the same |n k0| recovery
    kx = k0 * np.sin(np.radians(40.0))
    with _w.catch_warnings():
        _w.simplefilter("error")
        n_ob, _ = _probe_grid_sizes(Px, Px, kx, 0.0, np.sqrt(k0 ** 2 - kx ** 2),
                                    standoff_nm=s_big)
    assert n_ob > n_new
    assert kappa(n_ob, Px, kx) * s_big >= _ALIAS_DECAY_NEPERS

    # NO regression of the validated envelope: at the nominal 50 nm standoff the decay term asks
    # for a probe pitch of ~105 nm, so cells up to ~600 nm stay at the legacy 6x6 floor -- and
    # standoff=None still reproduces the old sizing exactly
    for P in (220.0, 300.0, 400.0, 600.0):
        assert _probe_grid_sizes(P, P, 0.0, 0.0, 3.48 * k0, standoff_nm=standoff) == (6, 6)
    assert _probe_grid_sizes(Px, Px, 0.0, 0.0, k0, standoff_nm=None) == (n_old, n_old)
    # a THIN buffer is what actually bites (the standoff is then the 10% pad, not 50 nm): a 400 nm
    # cell probed 30 nm off the structure needs 7 -- the sizing must respond to the standoff, not
    # just the period
    assert _probe_grid_sizes(400.0, 400.0, 0.0, 0.0, k0, standoff_nm=30.0) == (7, 7)


def test_f10_probe_grid_is_capped_and_the_subwavelength_6x6_claim_is_bounded():
    """AUDIT F-10 residual. Two things the wave-5 note got wrong.

    (1) "Sub-wavelength cells stay at the legacy 6x6" is FALSE as a general claim. It holds for the
        400 nm/50 nm case it was measured on, but 163 of 576 sub-wavelength (P < lambda/n)
        configurations at a 50 nm standoff already exceed 6 -- up to N = 11 at lambda = 1064 nm.
    (2) Cost runs away on a THIN pad, which is the DEFAULT transmission-side geometry (the standoff
        there is 0.1*substrate_buffer, i.e. 10 nm at the 100 nm default): a 1000 nm cell asks for
        (48, 48) instead of (6, 6), 64x the point evaluations. N is now capped, with a warning that
        points at the real fix (thicken the buffer -- the requirement goes as ~1/standoff)."""
    pytest.importorskip("ngsolve")
    import numpy as np
    import warnings as _w
    from dynameta.optics.solver import _PROBE_GRID_MAX, _probe_grid_sizes

    lam = 1550.0
    k0 = 2.0 * np.pi / lam

    # (1) the claim, bounded: count the sub-wavelength configurations that leave the 6x6 floor
    over = 0
    worst = 6
    for lam_ in (400.0, 633.0, 800.0, 1064.0, 1310.0, 1550.0, 2000.0, 3000.0):
        for nmed in (1.0, 1.45, 2.0, 3.5):
            kk = 2.0 * np.pi / lam_
            for frac in (0.1, 0.25, 0.5, 0.75, 0.95, 0.999):
                P = (lam_ / nmed) * frac                    # sub-wavelength by construction
                for th in (0.0, 30.0, 50.0):
                    kx = kk * np.sin(np.radians(th))
                    n, _ = _probe_grid_sizes(P, P, kx, 0.0, np.sqrt(max((nmed * kk) ** 2 - kx ** 2,
                                                                       0.0)), standoff_nm=50.0)
                    if n > 6:
                        over += 1
                        worst = max(worst, n)
    assert over > 100 and worst >= 11, (over, worst)         # measured 163 and 11

    # (2) the cap: the thin-pad case that motivated it (P = 1000 nm, 10 nm transmission pad)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        nx, ny = _probe_grid_sizes(1000.0, 1000.0, 0.0, 0.0, k0, standoff_nm=10.0)
    assert (nx, ny) == (_PROBE_GRID_MAX, _PROBE_GRID_MAX)
    msgs = [str(x.message) for x in caught]
    assert any("F-10" in m and "48" in m and "thicken" in m.lower() for m in msgs), msgs
    # thickening the buffer is the documented fix and takes the SAME cell back to the 6x6 floor,
    # silently (a 900 nm substrate_buffer -> 90 nm pad)
    with _w.catch_warnings():
        _w.simplefilter("error")
        assert _probe_grid_sizes(1000.0, 1000.0, 0.0, 0.0, k0, standoff_nm=90.0) == (6, 6)
    # below the cap nothing warns and nothing is capped (no regression of the F-10 sizing)
    with _w.catch_warnings():
        _w.simplefilter("error")
        assert _probe_grid_sizes(400.0, 400.0, 0.0, 0.0, k0, standoff_nm=30.0) == (7, 7)
    # the cap is per-direction: a cell long in x only caps x
    with _w.catch_warnings(record=True):
        _w.simplefilter("always")
        nx2, ny2 = _probe_grid_sizes(1000.0, 200.0, 0.0, 0.0, k0, standoff_nm=10.0)
    assert nx2 == _PROBE_GRID_MAX and ny2 == 10


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


def test_f13_sourced_tensor_eps_is_refused_with_the_upml_reason():
    """AUDIT F-13. solve_fem routes a TENSOR eps through an explicit UPML because mesh.SetPML's
    scalar stretch is wrong for an anisotropic medium; solve_fem_sourced has no such branch and
    calls SetPML unconditionally. It used to fail deep inside NGSolve ("Dimensions don't match,
    op = -") with no hint that the sourced path simply cannot do tensors."""
    import ngsolve as ng
    from dynameta.optics import solver as S
    geo, _eps, opt = _fem_cell()
    tensor = ng.CoefficientFunction((2.0, 0.1, 0, -0.1, 2.0, 0, 0, 0, 2.0), dims=(3, 3))
    with pytest.raises(NotImplementedError, match="TENSOR"):
        S.solve_fem_sourced(geo, 1200e-9, tensor, opt, order=1)
    # the message must name the actual reason (no UPML branch), not just "unsupported"
    with pytest.raises(NotImplementedError, match="UPML"):
        S.solve_fem_sourced(geo, 1200e-9, tensor, opt, order=1)


def test_f13_sourced_path_enforces_the_same_oblique_pml_envelope():
    """AUDIT F-13. The sourced path carried NONE of solve_fem's incidence guards: no 50 deg cap and
    no PML advisory, so shg_structured_two_step at 70 deg ran straight past the envelope where
    solve_fem refuses. The source carries k_par rather than an angle, so the guard recovers
    sin(theta) = |k_par|/(n_super k0) -- and must NOT false-fire at exactly the cap."""
    import numpy as np
    from dynameta.optics import solver as S
    from dynameta.optics.ngsolve_layered import S as _S
    geo, eps_cf, opt = _fem_cell()
    lam = 1200e-9
    k0 = 2.0 * np.pi / (lam * _S)                                  # nm^-1
    S._ADVISED_ONCE.discard("oblique_pml")

    with pytest.raises(NotImplementedError, match="validated envelope"):
        S.solve_fem_sourced(geo, lam, eps_cf, opt, order=1,
                            k_par_per_nm=(k0 * np.sin(np.radians(70.0)), 0.0))
    # NO false fire at the cap itself (and the advisory arrives, once). A source-free sourced solve
    # is a legal, cheap probe: the RHS is zero, so it exercises every guard and returns p_up = 0.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = S.solve_fem_sourced(geo, lam, eps_cf, opt, order=1,
                                  k_par_per_nm=(k0 * np.sin(np.radians(50.0)), 0.0))
    assert res.p_up == 0.0 and res.a_up == 0j
    hits = [w for w in rec if "not angle-aware" in str(w.message)]
    assert len(hits) == 1 and issubclass(hits[0].category, S.FEMDiagnosticWarning)
    # ... and it is the SAME once-per-process advisory solve_fem emits (same tag, not a second one)
    assert "oblique_pml" in S._ADVISED_ONCE
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        S._incidence_geometry(type("O", (), {"incidence_angle_deg": 30.0, "azimuth_deg": 0.0,
                                             "polarization": "y", "incidence_side": "top"})(),
                              1.0 + 0j)
    assert not [w for w in rec2 if "not angle-aware" in str(w.message)]


def test_f13_sourced_grazing_port_is_refused_but_deep_evanescent_is_not():
    """AUDIT F-13. A port at the grazing cutoff makes the two-wave fit basis exp(+-i kz z) collapse
    to a rank-1 DC fit, so the up/down split is arbitrary -- solve_fem refuses exactly this on its
    substrate order. A DEEPLY evanescent port (|kz| large and imaginary) is a different, legitimate
    case: decaying vs growing exponential, well separated."""
    from dynameta.optics.solver import _refuse_grazing_port
    k0 = 2.0 * np.pi / 1200.0
    with pytest.raises(NotImplementedError, match="grazing cutoff"):
        _refuse_grazing_port(complex(1e-9 * k0, 0.0), k0, "superstrate", 1.0, k0, 0.0)
    with pytest.raises(NotImplementedError, match="grazing cutoff"):
        _refuse_grazing_port(complex(0.0, 1e-9 * k0), k0, "substrate", 1.0, k0, 0.0)
    _refuse_grazing_port(complex(0.0, 0.9 * k0), k0, "substrate", 1.0, 1.5 * k0, 0.0)  # evanescent
    _refuse_grazing_port(complex(0.7 * k0, 0.0), k0, "superstrate", 1.0, 0.7 * k0, 0.0)  # propagating


# ------------------- shared layered background (audit F-12) / surface sheet (F-11) -------------------

def _bg_args(pol, *, kx=0.0, z_int=100.0, n_super=1.0 + 0j, n_sub=1.5 + 0j, k0=2 * np.pi / 1200.0):
    """Positional args for solver._layered_background at azimuth 0."""
    kz_s = np.sqrt(max((complex(n_super).real * k0) ** 2 - kx ** 2, 0.0))
    kz_sub = complex(np.sqrt(complex((complex(n_sub) * k0) ** 2 - kx ** 2)))
    return (pol, k0, kx, 0.0, 0.0, complex(kz_s), kz_sub, z_int,
            complex(n_super) ** 2, complex(n_sub) ** 2, n_super, n_sub)


def test_f12_shared_background_keeps_all_three_polarization_branches():
    """AUDIT F-12. shg_fem carried a private copy of solve_fem's background declared "byte-for-byte
    the construction solve_fem uses" -- and it had drifted to TWO branches (`if pol == 'p' ... else
    s-pol E along y`), so polarization='x', the OpticalSpec DEFAULT, was built as the ORTHOGONAL
    mode. There is now ONE implementation with all three branches; this pins the mode of each."""
    from dynameta.optics import solver as S
    geo, _eps, _opt = _fem_cell()
    z0, z1 = geo.z_intervals_nm["superstrate"]
    pt = geo.mesh(geo.period_x_nm / 3.0, geo.period_y_nm / 4.0, 0.5 * (z0 + z1))

    ex = [complex(c) for c in S._layered_background(*_bg_args("x"))[0](pt)]
    assert abs(ex[0]) > 0.1 and ex[1] == 0j and ex[2] == 0j            # E along x
    ey = [complex(c) for c in S._layered_background(*_bg_args("y"))[0](pt)]
    assert abs(ey[1]) > 0.1 and ey[0] == 0j and ey[2] == 0j            # E along y
    assert abs(ex[0] - ey[1]) < 1e-12 * abs(ex[0])                     # same scalar, rotated
    ep = [complex(c) for c in S._layered_background(*_bg_args("p", kx=0.3 * 2 * np.pi / 1200.0))[0](pt)]
    assert abs(ep[0]) > 0.0 and abs(ep[2]) > 0.0 and ep[1] == 0j       # E in the x-z plane
    # the projection helper must track the branch it is paired with
    assert S.background_probe_pol("x", 0.4) == (1.0, 0.0, 0.0)
    assert S.background_probe_pol("y", 0.4) == (0.0, 1.0, 0.0)
    assert S.background_probe_pol("p", 0.0) == (1.0, 0.0, 0.0)


def test_f12_solve_fem_and_shg_build_the_background_from_the_SAME_helper(monkeypatch):
    """AUDIT F-12, the anti-drift gate: both entry points must go through solver._layered_background
    (a duplicate is what drifted). Also pins the consequence -- for polarization='x' the SHG
    reconstruction now gets an x-directed field probed along x, where it used to get E-along-y
    probed with (0,1,0)."""
    from dynameta.optics import shg_fem
    from dynameta.optics import solver as S
    geo, eps_cf, opt = _fem_cell(pol="x")

    calls = []
    real = S._layered_background

    def spy(*a, **k):
        calls.append(a[0])                                   # the polarization branch taken
        return real(*a, **k)

    monkeypatch.setattr(S, "_layered_background", spy)
    S.solve_fem(geo, 1200e-9, eps_cf, opt, order=1)
    assert calls == ["x"], "solve_fem must build its background through the shared helper"

    # the SHG reconstruction: capture what it hands to the sourced solve (no solve needed)
    grabbed = {}

    def fake_sourced(_geo, _lam, _eps, _opt, **kw):
        grabbed.update(kw)
        return S.SourcedResult(fes=None, gfu=S.ng.CoefficientFunction((0j, 0j, 0j)),
                               bg_field=kw["bg_field"], a_up=0j, a_down=None, p_up=0.0,
                               p_down=None, solve_time_s=0.0, relres=0.0)

    monkeypatch.setattr(S, "solve_fem_sourced", fake_sourced)
    del calls[:]
    shg_fem._reconstruct_fundamental_field(geo, None, 1200e-9, opt, eps_cf, 1,
                                           1.0 + 0j, 1.0 + 0j, 0.0, "x")
    assert calls == ["x"], "shg_fem must build its background through the shared helper"
    assert grabbed["probe_pol"] == (1.0, 0.0, 0.0)            # was (0,1,0): the orthogonal mode
    z0, z1 = geo.z_intervals_nm["superstrate"]
    E = [complex(c) for c in grabbed["bg_field"](geo.mesh(geo.period_x_nm / 3.0,
                                                          geo.period_y_nm / 4.0,
                                                          0.5 * (z0 + z1)))]
    assert abs(E[0]) > 0.1 and E[1] == 0j                    # x-polarized, as solve_fem builds it


def test_f12_conical_incidence_is_refused_by_the_shg_reconstruction():
    """AUDIT F-12 (the other half of the drift): the private copy had no ky, so a conical azimuth
    was silently dropped rather than refused."""
    pytest.importorskip("ngsolve")
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.optics.shg_fem import _reconstruct_fundamental_field
    opt = OpticalSpec(polarization="p", incidence_angle_deg=20.0, azimuth_deg=15.0,
                      linear_solver="umfpack")
    with pytest.raises(NotImplementedError, match="conical"):
        _reconstruct_fundamental_field(None, None, 1000e-9, opt, None, 2,
                                       1.0 + 0j, 1.0 + 0j, 20.0, "p")


def _sheet_stub(eps_above, *, cap_nm=None, eps_cap_top=1.0 + 0j):
    """A flat metal at z = 0 under a dielectric `eps_above`, sampled by pure-python stubs (no mesh):
    an analytic upward E_z wave in the dielectric. cap_nm makes the dielectric a CAP of that
    thickness with eps_cap_top above it (the standoff-straddle case)."""
    eps_m = complex(-40.0, 2.5)
    k0 = 2.0 * np.pi / 1550.0                                  # nm^-1
    kz = complex(np.sqrt(complex(eps_above) * k0 ** 2))        # normal incidence in the dielectric
    E0 = 3.0 + 0j

    class _Geo:
        period_x_nm, period_y_nm = 400.0, 100.0
        z_intervals_nm = {"metal": (-200.0, 0.0)}

    def mesh(x, y, z):
        return (float(x), float(y), float(z))

    def eps_cf(pt):
        if pt[2] <= 0.0:
            return eps_m
        if cap_nm is not None and pt[2] > cap_nm:
            return complex(eps_cap_top)
        return complex(eps_above)

    def field(pt):
        return (0.0 + 0j, 0.0 + 0j, E0 * np.exp(1j * kz * pt[2]))

    return _Geo(), mesh, eps_cf, field, eps_m, k0, E0


def test_f11_sheet_uses_the_MEASURED_eps_above_for_normal_D_continuity():
    """AUDIT F-11. The extraction wrote E_perp,in = d_perp / eps_metal, i.e. it identified the
    extrapolated E_z of the medium above the metal with D_perp -- true only in VACUUM. Normal-D
    continuity is eps_above E_z,above = eps_metal E_perp,in, so the driving field was short by
    eps_above and the sheet P_z ~ E_perp^2 by eps_above^2 (a factor 5 for n = 1.5), while n_super
    is an exposed parameter of shg_structured_two_step. The standoff two-wave fit used the vacuum
    dispersion for the same reason."""
    pytest.importorskip("ngsolve")
    from dynameta.constants import EPS0
    from dynameta.optics.shg_fem import _extract_surface_sheet_profile
    chi = 1e-20 + 0j
    for eps_d in (1.0 + 0j, 2.25 + 0j, complex(4.0, 0.2)):
        geo, mesh, eps_cf, field, eps_m, k0, E0 = _sheet_stub(eps_d)
        sheet = _extract_surface_sheet_profile(mesh, geo, field, eps_cf, 0.0, chi, eps_m,
                                               "metal", 2.0 * k0, k0, nx=8, ny=1, standoff_nm=4.0)
        want = EPS0 * chi * (complex(eps_d) * E0 / eps_m) ** 2       # normal-D continuity
        assert sheet["eps_above"] == pytest.approx(complex(eps_d), rel=1e-12)
        assert abs(sheet["c0"] - want) < 1e-6 * abs(want)
        # the pre-fix value is exactly eps_above^2 smaller -- the bug the ledger quantified
        stale = EPS0 * chi * (E0 / eps_m) ** 2
        assert abs(sheet["c0"] / stale - complex(eps_d) ** 2) < 1e-6 * abs(eps_d) ** 2
    # a standoff band that straddles an interface cannot be one medium: say so
    geo, mesh, eps_cf, field, eps_m, k0, _E0 = _sheet_stub(2.25 + 0j, cap_nm=10.0)
    with pytest.warns(UserWarning, match="MATERIAL INTERFACE"):
        _extract_surface_sheet_profile(mesh, geo, field, eps_cf, 0.0, chi, eps_m, "metal",
                                       2.0 * k0, k0, nx=4, ny=1, standoff_nm=4.0)


def test_f11_straddle_detection_covers_a_thin_cap_and_ignores_a_graded_film():
    """AUDIT F-11 residual, two holes in the straddle test.

    (a) A cap THINNER than the FIRST standoff was invisible: every sample sat above the cap, the
        band read perfectly uniform, and eps_above came back as the medium ABOVE the cap with no
        warning at all (measured on a real mesh: a 2 nm cap under a 4..24 nm band -> eps_above =
        1.0, i.e. the sheet wrong by eps_cap^2 = 5x for a mere n = 1.5 encapsulant).
    (b) The "any sample differs from eps_above by > 1 %" test is a GRADING-RATE test wearing an
        interface's warning text: a smoothly graded film (a VoxelCoefficient interpolates linearly
        across the band) warned "crosses a MATERIAL INTERFACE" at 1.5 %, 5.0 % and 7.8 % of grading
        across the band, though none of them does. The test is now a SHAPE test -- one step much
        larger than the rest -- which separates the two cleanly."""
    pytest.importorskip("ngsolve")
    from dynameta.optics.shg_fem import _straddles_interface, _extract_surface_sheet_profile
    chi = 1e-20 + 0j

    # (b) the shape discriminator, on the exact profiles that used to false-fire
    for pct in (0.5, 1.5, 5.0, 7.8, 20.0):                    # linear grade across the band
        col = [2.25 * (1.0 + pct / 100.0 * i / 5.0) for i in range(6)]
        assert not _straddles_interface(col, col[0]), pct
    for lo_n in (1, 2, 3, 4, 5):                              # a genuine step, anywhere in the band
        col = [2.25] * lo_n + [1.0] * (6 - lo_n)
        assert _straddles_interface(col, col[0]), lo_n
    # a graded film that ALSO crosses an interface is still a straddle
    grad_step = [2.25, 2.26, 2.27, 1.0, 1.01, 1.02]
    assert _straddles_interface(grad_step, grad_step[0])
    # noise-level scatter is not an interface; a 2-sample band falls back to the magnitude test
    assert not _straddles_interface([2.25, 2.2500001, 2.2499998], 2.25)
    assert _straddles_interface([2.25, 1.0], 2.25)

    # (a) the thin cap, end to end on the stub: cap 2 nm under a 4..24 nm band
    geo, mesh, eps_cf, field, eps_m, k0, _E0 = _sheet_stub(2.25 + 0j, cap_nm=2.0)
    with pytest.warns(UserWarning, match="MATERIAL INTERFACE"):
        _extract_surface_sheet_profile(mesh, geo, field, eps_cf, 0.0, chi, eps_m, "metal",
                                       2.0 * k0, k0, nx=4, ny=1, standoff_nm=4.0)
    # ... and a UNIFORM dielectric (no cap at all) must still be silent -- no false fire from the
    # extra sub-standoff probe
    geo, mesh, eps_cf, field, eps_m, k0, _E0 = _sheet_stub(2.25 + 0j)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sheet = _extract_surface_sheet_profile(mesh, geo, field, eps_cf, 0.0, chi, eps_m, "metal",
                                               2.0 * k0, k0, nx=4, ny=1, standoff_nm=4.0)
    assert sheet["eps_above"] == pytest.approx(2.25 + 0j, rel=1e-12)


def test_f11_non_vacuum_superstrate_is_refused_by_every_surface_shg_driver():
    """AUDIT F-11, the exposed parameter: n_super is a kwarg of all three drivers while every
    radiator in the module (the p-pol Fresnel inside-field, the sheet's vacuum radiation +
    eps_ref=1.0, the Sipe denominator, the order extraction) is a VACUUM construction. The guard
    fires before any mesh is built, so `design` is never touched."""
    pytest.importorskip("ngsolve")
    from dynameta.optics.shg_fem import (_require_vacuum_superstrate, sfg_two_step,
                                         shg_structured_two_step, shg_two_step)
    _require_vacuum_superstrate(1.0 + 0j, "unit")                      # no false fire
    _require_vacuum_superstrate(complex(1.0, 0.0), "unit")
    with pytest.raises(NotImplementedError, match="non-vacuum superstrate"):
        _require_vacuum_superstrate(1.5 + 0j, "unit")
    w = 2.0 * np.pi * 3e8 / 1550e-9
    with pytest.raises(NotImplementedError, match="non-vacuum superstrate"):
        shg_two_step(None, lambda_fund_m=1550e-9, chi_zzz=1e-20, n_super=1.5)
    with pytest.raises(NotImplementedError, match="non-vacuum superstrate"):
        shg_structured_two_step(None, lambda_fund_m=1550e-9, chi_zzz=1e-20, n_super=1.5)
    with pytest.raises(NotImplementedError, match="non-vacuum superstrate"):
        sfg_two_step(None, omega1_rad_s=w, omega2_rad_s=0.7 * w, chi_zzz=1e-20, n_super=1.5)


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
