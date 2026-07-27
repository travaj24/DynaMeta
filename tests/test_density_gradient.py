"""Tests for the R19 density-gradient post-hoc quantum correction. The Schrodinger-Poisson
dead-layer oracle lives in validation/density_gradient_dead_layer.py."""
import numpy as np
import pytest

from dynameta.constants import HBAR, M_E, Q_E as Q
from dynameta.carriers.density_gradient import (dg_correct_density_1d, dg_length_m,
                                                quantum_potential_V)

MSTAR = 0.35 * M_E


def test_quantum_potential_gaussian_closed_form():
    """audit C-3/N3: the `inner = |z| < 8e-9` mask this test used to carry is GONE -- the closed
    form is now met at EVERY node including the four the old np.gradient^2 composition got wrong.
    Measured on this fixture: max relative error 3.1e-5 over the full array (it was 4.96e-1, i.e.
    496x over the 1e-3 budget, at the end nodes before the fix)."""
    s = 3e-9
    z = np.linspace(-12e-9, 12e-9, 1601)
    n = 1e26 * np.exp(-z ** 2 / (2.0 * s ** 2))
    lam = quantum_potential_V(z, n, MSTAR)
    b = HBAR ** 2 / (6.0 * MSTAR * Q)
    cf = b * (z ** 2 / (4.0 * s ** 4) - 1.0 / (2.0 * s ** 2))
    assert np.max(np.abs(lam - cf)) < 1e-3 * np.max(np.abs(cf))       # NO mask


def test_quantum_potential_is_second_order_at_the_four_end_nodes():
    """audit C-3 / N3 (= 2026-07-17 ledger S1-6), the load-bearing gate.

    `np.gradient(np.gradient(u, z), z)` is non-convergent at FOUR nodes: edge_order=1 makes the
    inner pass one-sided-first-order at 0 and N-1, and the outer centred pass then mixes that
    biased slope into 1 and N-2. The limits are exactly (1/2) f'' at 0/N-1 and (3/4) f'' at
    1/N-2 -- measured on the Gaussian fixture above as ratios 0.5036 and 0.7518 -- and they do
    NOT improve with h. So refinement alone is the discriminator: this gate measures the OBSERVED
    order p = log2(e(h)/e(h/2)) at each of the four nodes (plus one interior node as a control)
    and requires p ~ 2 everywhere. Under the old composition p would be ~0 at all four."""
    k = 3.0e8                                                     # ~5 oscillations over 20 nm
    zmax = 20e-9

    def n_of(z):                                                  # u = sqrt(n) = exp(sin(k z))
        return np.exp(2.0 * np.sin(k * z))

    def lam_exact(z):                                             # b (u''/u) = b (s'' + s'^2)
        return b_ref * (-k ** 2 * np.sin(k * z) + (k * np.cos(k * z)) ** 2)

    b_ref = HBAR ** 2 / (6.0 * MSTAR * Q)
    errs = {}
    for nn in (201, 401, 801, 1601):
        z = np.linspace(0.0, zmax, nn)
        got = quantum_potential_V(z, n_of(z), MSTAR)
        ex = lam_exact(z)
        scale = np.max(np.abs(ex))
        errs[nn] = {tag: abs(got[i] - ex[i]) / scale
                    for tag, i in (("0", 0), ("1", 1), ("mid", nn // 2),
                                   ("N-2", nn - 2), ("N-1", nn - 1))}
    grids = sorted(errs)
    for tag in ("0", "1", "mid", "N-2", "N-1"):
        for coarse, fine in zip(grids[:-1], grids[1:]):
            p = np.log2(errs[coarse][tag] / errs[fine][tag])
            assert p > 1.7, "node {}: observed order {:.2f} between n={} and n={} (want ~2)".format(
                tag, p, coarse, fine)
    # and the absolute error at the two hardest nodes is small on the finest grid
    assert errs[1601]["0"] < 1e-4 and errs[1601]["N-1"] < 1e-4


def test_quantum_potential_matches_a_quadratic_exactly_at_every_node():
    """The stencils' exactness contract: interior is the 3-point non-uniform second difference
    (exact for quadratics) and the ends are 4-point one-sided (exact for cubics). On u = 1 + a z
    + c z^2 the discrete u'' must equal 2c to round-off AT EVERY NODE -- including on a
    NON-UNIFORM mesh, which is the case the end weights are solved for."""
    for z in (np.linspace(0.0, 12e-9, 61),
              np.sort(np.concatenate([np.linspace(0.0, 4e-9, 25),      # deliberately graded
                                      np.linspace(4.2e-9, 12e-9, 20)]))):
        a, c = 3.0e7, 5.0e15
        u = 1.0 + a * z + c * z ** 2
        lam = quantum_potential_V(z, u ** 2, MSTAR)
        b = HBAR ** 2 / (6.0 * MSTAR * Q)
        assert np.allclose(lam, b * 2.0 * c / u, rtol=1e-8)


def test_quantum_potential_rejects_a_non_monotonic_grid():
    z = np.array([0.0, 1e-9, 3e-9, 2e-9, 4e-9, 5e-9])             # not sorted
    with pytest.raises(ValueError, match="monotonic"):
        quantum_potential_V(z, np.full_like(z, 1e26), MSTAR)


def test_c8_degeneracy_factor_shortens_l_q_and_defaults_to_the_boltzmann_limit():
    """audit C-8: the BVP closes with BOLTZMANN statistics for a material this module declares
    degenerate, and the in-Newton twin (physics_density_gradient) uses the FD g-form. The
    FD-consistent length is sqrt(g) shorter -- pinned here against the auditor's exact numbers --
    so the ~1.2 nm the header quotes is a CALIBRATION that the fitted `gamma` absorbs, not a
    derivation. degeneracy_g=1 is the byte-identical off-switch."""
    from dynameta.carriers.einstein import g_einstein
    lq_b = dg_length_m(MSTAR)
    assert lq_b == pytest.approx(1.184731e-9, rel=1e-6)               # the Boltzmann value
    assert dg_length_m(MSTAR, degeneracy_g=1.0) == lq_b               # explicit off-switch
    assert dg_length_m(MSTAR, degeneracy_g=13.4) == pytest.approx(3.236438e-10, rel=1e-6)
    for g in (6.8, 13.4):                                             # eta ~ 10 and ~ 20
        assert lq_b / dg_length_m(MSTAR, degeneracy_g=g) == pytest.approx(np.sqrt(g), rel=1e-12)
    assert 2.6 < lq_b / dg_length_m(MSTAR, degeneracy_g=6.8) < 3.7    # the reported 2.6-3.7x band
    # the factor is the repo's own g(n/N_c), not a magic constant
    assert dg_length_m(MSTAR, degeneracy_g=float(g_einstein(30.0))) < 0.5 * lq_b
    with pytest.raises(ValueError):
        dg_length_m(MSTAR, degeneracy_g=0.0)


def test_dg_off_switch_and_dead_layer():
    z = np.linspace(0.0, 15e-9, 901)
    n = np.full_like(z, 2e26)
    assert np.array_equal(dg_correct_density_1d(z, n, MSTAR, gamma=0.0), n)
    n_dg = dg_correct_density_1d(z, n, MSTAR)
    lq = dg_length_m(MSTAR)
    assert n_dg[0] < 1e-3 * 2e26                          # hard wall
    assert abs(n_dg[-1] / 2e26 - 1.0) < 1e-4              # bulk recovered
    assert 0.8e-9 < lq < 1.8e-9                           # the ~1 nm ITO dead-layer scale


def test_dg_guards():
    z = np.linspace(0.0, 10e-9, 301)
    with pytest.raises(ValueError):
        dg_correct_density_1d(z, np.zeros_like(z), MSTAR)
    with pytest.raises(ValueError):
        dg_correct_density_1d(z, np.full_like(z, 1e26), MSTAR, hard_wall="middle")
    with pytest.raises(ValueError):
        quantum_potential_V(z, np.full(300, 1e26), MSTAR)


# ---- R19 follow-on: in-Newton DG-DD (module-level guards; the 4-variable Newton oracle is
# validation/dg_dd_in_newton.py) -----------------------------------------------------------------

def test_dg_b_coefficient_and_guards():
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_density_gradient import dg_b_coefficient, set_dg_gamma
    b = dg_b_coefficient(MSTAR, 1.0)
    assert b == pytest.approx(HBAR ** 2 / (6.0 * MSTAR * Q), rel=1e-14)
    assert dg_b_coefficient(MSTAR, 0.0) == 0.0
    with pytest.raises(ValueError):
        dg_b_coefficient(-1.0)
    with pytest.raises(ValueError):
        set_dg_gamma("d", "r", 1.5)                      # frac outside [0, 1]


def test_dg_contact_guard():
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_density_gradient import setup_contact_dg
    with pytest.raises(ValueError):
        setup_contact_dg("dev", "c", 0.0)


# ---- audit X-6: setup_dg_hard_wall is a shipped __all__ export whose only caller was an
# underscore-prefixed validation that run_all deliberately skips, so NOTHING ever machine-checked
# it. The gates below are its EXECUTABLE contract; the parked WIP is gone (see the note in
# test_dg_hard_wall_pins_the_wall_rows for what it measured and why it is not a physics gate).

def _hard_wall_bar(tag, *, pin=8.0, gamma=1.0, frozen_psi=True,
                   fracs=(0.05, 0.1, 0.25, 0.5, 1.0)):
    """Build the smallest 1-D DG hard-wall device that converges (44 nodes, ~1 s): a uniform n-type
    ITO-like bar with the LEFT contact carrying setup_dg_hard_wall (DG rows only, an insulating
    boundary) and the RIGHT a bulk-ohmic DG contact, ramped to full gamma. Returns
    (z, n, u, Lambda, N_D) sorted by z; the device is torn down before returning."""
    import contextlib
    import os
    import sys

    import devsim as ds

    from dynameta.constants import M_E
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.physics_density_gradient import (seed_dg_from_solution, set_dg_gamma,
                                                            setup_contact_dg, setup_dg_hard_wall,
                                                            setup_dg_quantum_correction)
    from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                           setup_semiconductor_region_dd)

    @contextlib.contextmanager
    def _quiet():                                    # DEVSIM chatters on stdout at C level
        sys.stdout.flush()
        saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(devnull)
            os.close(saved)

    def _solve():
        with _quiet():                               # REL 1e-5: the wall node's n ~ 1e-12 N_D makes
            ds.solve(type="dc", absolute_error=1.0e16,   # per-node RELATIVE updates floor out below
                     relative_error=1.0e-5, maximum_iterations=200)

    mstar, length, n0 = 0.35 * M_E, 100e-9, 4.0e26
    mesh, dev, reg = "x6m_" + tag, "x6d_" + tag, "bar"
    try:
        with _quiet():
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=0.2e-9, tag="wall")   # resolve L_q = 1.18 nm
            ds.add_1d_mesh_line(mesh=mesh, pos=10e-9, ps=1.0e-9)
            ds.add_1d_mesh_line(mesh=mesh, pos=length, ps=10e-9, tag="back")
            ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
            ds.add_1d_contact(mesh=mesh, name="back", tag="back", material="metal")
            ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="wall", tag2="back")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            setup_semiconductor_region_dd(dev, reg, n_bg_m3=n0, eps_static=9.5,
                                          dos_mass_kg=mstar, mobility_m2Vs=0.004)
            setup_contact_ohmic_dd(dev, "back")          # the wall contact: DG equations only
            nn = len(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
            ds.set_node_values(device=dev, region=reg, name="Electrons", values=[n0] * nn)
        _solve()                                         # classical equilibrium (flat)
        with _quiet():
            setup_dg_quantum_correction(dev, reg, m_eff_kg=mstar, gamma=gamma)
            setup_contact_dg(dev, "back", n0)
            setup_dg_hard_wall(dev, "wall", lambda_pin_factor=pin)
            seed_dg_from_solution(dev, reg)
            # wall-aware seed: taper u (and n = u^2) to ~0 over L_q at the wall -- seeding the FLAT
            # bulk profile against the wall pin overflows the first ramp step's Newton transients
            z0 = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="x"))
            u0 = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="QSqrtN"))
            taper = np.maximum(np.tanh(z0 / 1.2e-9), 1e-6)
            ds.set_node_values(device=dev, region=reg, name="QSqrtN", values=list(u0 * taper))
            ds.set_node_values(device=dev, region=reg, name="Electrons",
                               values=list(np.maximum((u0 * taper) ** 2, 1e14)))
            if frozen_psi:
                _R.delete_by_name(dev, "PotentialEquation")     # freeze psi at the flat solution
        for fr in fracs:
            set_dg_gamma(dev, reg, fr)
            _solve()
        get = (lambda name: np.asarray(ds.get_node_model_values(device=dev, region=reg, name=name)))
        z = get("x")
        order = np.argsort(z)
        out = (z[order], get("Electrons")[order], get("QSqrtN")[order], get("QLambda")[order],
               float(ds.get_parameter(device=dev, region=reg, name="N_D")))
    finally:
        with _quiet():
            try:
                _R.clear(dev)
                ds.delete_device(device=dev)
                ds.delete_mesh(mesh=mesh)
            except Exception:                            # nothing to tear down (build failed)
                pass
    return out


def test_dg_hard_wall_input_guard():
    """audit X-6: the only guard the export declares, never executed by anything."""
    pytest.importorskip("devsim")
    from dynameta.carriers.physics_density_gradient import setup_dg_hard_wall
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="lambda_pin_factor"):
            setup_dg_hard_wall("dev", "wall", lambda_pin_factor=bad)


def test_dg_hard_wall_pin_is_a_ramped_device_parameter():
    """audit X-6: setup_dg_hard_wall's Lambda pin is a DEVICE parameter so the gamma ramp can
    co-ramp it (a full-depth pin at the first small-gamma step overflows the wall-edge Bernoulli
    during Newton transients). set_dg_gamma must scale BOTH b_dg and the pin, and must stay a no-op
    on the pin for a device with no hard wall."""
    ds = pytest.importorskip("devsim")
    from dynameta.constants import M_E, V_T
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.physics_density_gradient import (dg_b_coefficient, set_dg_gamma,
                                                            setup_dg_hard_wall,
                                                            setup_dg_quantum_correction)
    from dynameta.carriers.physics_drift_diffusion import setup_semiconductor_region_dd
    mstar, n0, pin = 0.35 * M_E, 4.0e26, 8.0
    mesh, dev, reg = "x6pm", "x6pd", "bar"
    try:
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="wall")
        ds.add_1d_mesh_line(mesh=mesh, pos=20e-9, ps=2e-9, tag="back")
        ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
        ds.add_1d_contact(mesh=mesh, name="back", tag="back", material="metal")
        ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="wall", tag2="back")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        setup_semiconductor_region_dd(dev, reg, n_bg_m3=n0, eps_static=9.5,
                                      dos_mass_kg=mstar, mobility_m2Vs=0.004)
        setup_dg_quantum_correction(dev, reg, m_eff_kg=mstar, gamma=1.0)
        b_full = dg_b_coefficient(mstar, 1.0)
        # no hard wall yet: set_dg_gamma must not invent a pin
        set_dg_gamma(dev, reg, 0.25)
        assert ds.get_parameter(device=dev, region=reg, name="b_dg") == pytest.approx(0.25 * b_full)
        with pytest.raises(Exception):
            ds.get_parameter(device=dev, name="wall_lambda_pin")
        setup_dg_hard_wall(dev, "wall", lambda_pin_factor=pin)
        # installed at ZERO depth (the ramp turns it on), full depth recorded separately
        assert ds.get_parameter(device=dev, name="wall_lambda_pin") == 0.0
        assert ds.get_parameter(device=dev, name="wall_lambda_pin_full") == pytest.approx(pin * V_T)
        for frac in (0.05, 0.5, 1.0):
            set_dg_gamma(dev, reg, frac)
            assert ds.get_parameter(device=dev, name="wall_lambda_pin") == pytest.approx(
                frac * pin * V_T, rel=1e-12)
            assert ds.get_parameter(device=dev, region=reg, name="b_dg") == pytest.approx(
                frac * b_full, rel=1e-12)
    finally:
        try:
            _R.clear(dev)
            ds.delete_device(device=dev)
            ds.delete_mesh(mesh=mesh)
        except Exception:
            pass


def test_dg_hard_wall_pins_the_wall_rows():
    """audit X-6, the load-bearing gate: `setup_dg_hard_wall` shipped in `__all__` with ZERO
    executable coverage -- its only caller was `validation/_dg_hard_wall_wip.py`, and
    `run_all.py` skips every `_`-prefixed file in EVERY tier, so its four declared gates had never
    been machine-checked. This runs the real 4-variable Newton on a coarse bar and pins the DISCRETE
    MECHANICS the docstring claims (the part that IS validated):

      * u(wall) is pinned to the documented floor u_floor = 1e-6 sqrt(N_D) -- not an exact zero,
        which DEVSIM's variable_update='positive' forbids;
      * the wall's ELECTRON row is the bulk constraint n = u^2 (NOT a Boltzmann quasi-equilibrium
        pin, which would evaluate the REGULARIZATION Lambda-pin as a physical wall Lambda), so
        n(wall) = u_floor^2 = 1e-12 N_D -- the dead-layer endpoint;
      * Lambda(wall) sits at the ramped regularization depth -lambda_pin_factor V_t;
      * the bulk is unperturbed (n -> N_D) and the profile rises monotonically off the wall.

    NOT a physics gate, deliberately. The WIP's GATE A (frozen-psi in-Newton == the validated
    post-hoc BVP dg_correct_density_1d) FAILS -- re-measured 2026-07-26 at max |dn|/N0 = 0.80 on
    its own fine mesh, and its self-consistent leg (GATE B) raises a DEVSIM convergence failure.
    The converged Newton settles on a spurious WIDE-depletion branch: the dead-layer deficit here
    is tens of N0 L_q where the BVP closure gives ~1. That is exactly why setup_dg_hard_wall is
    documented EXPERIMENTAL and why the post-hoc closure remains THE dead-layer tool. The width is
    pinned below as a CHARACTERIZATION so a continuation fix cannot land silently."""
    pytest.importorskip("devsim")
    from dynameta.constants import M_E, V_T
    pin = 8.0
    z, n, u, lam, n_d = _hard_wall_bar("rows", pin=pin)
    u_floor = 1.0e-6 * np.sqrt(n_d)

    assert u[0] == pytest.approx(u_floor, rel=1e-9)             # u -> the documented floor
    assert n[0] == pytest.approx(u_floor ** 2, rel=1e-9)        # electron row: n = u^2
    assert n[0] == pytest.approx(1e-12 * n_d, rel=1e-9)         # ... = 1e-12 N_D, the endpoint
    assert lam[0] == pytest.approx(-pin * V_T, rel=1e-9)        # Lambda: the ramped pin depth
    assert n[-1] == pytest.approx(n_d, rel=1e-6)                # bulk unperturbed
    assert np.all(np.diff(n) >= -1e-6 * n_d)                    # rises monotonically off the wall
    assert np.all(np.isfinite(u)) and np.all(u > 0.0)           # variable_update='positive' held

    from dynameta.core.numerics import trapz          # audit X-1: np.trapezoid needs numpy>=2.0
    deficit = float(trapz(n_d - n, z))
    lq = dg_length_m(0.35 * M_E)
    assert deficit > 0.0                                        # a dead layer exists at all
    assert 3.0 < deficit / (n_d * lq) < 1000.0, (
        "dead-layer deficit {:.1f} N0 L_q. This CHARACTERIZES the known-open in-Newton wall "
        "convergence (the post-hoc BVP closure gives ~1 N0 L_q). If a Newton-continuation fix "
        "landed, REPLACE this with the real physics gate against dg_correct_density_1d and "
        "promote setup_dg_hard_wall out of EXPERIMENTAL.".format(deficit / (n_d * lq)))
