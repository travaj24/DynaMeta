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
    from dynameta.optics.tmm_reference import stack_rta
    # a normal passive (lossy-but-physical) stack must still return a clean R+T+A=1, A>=0.
    R, T, A = stack_rta(1.0, [(complex(1.5, 0.02), 200e-9)], 1.5, 1550e-9)
    assert R >= 0 and T >= 0 and A >= -1e-9
    assert abs(R + T + A - 1.0) < 1e-9


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
