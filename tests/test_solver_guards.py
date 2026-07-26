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
