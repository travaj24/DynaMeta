"""Tests for the R19 density-gradient post-hoc quantum correction. The Schrodinger-Poisson
dead-layer oracle lives in validation/density_gradient_dead_layer.py."""
import numpy as np
import pytest

from dynameta.constants import HBAR, KB, M_E, Q_E as Q, T_REF, V_T
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


def test_gamma_wall_is_the_exact_free_gas_wall_gamma():
    """GAMMA_WALL (user decision 2026-11-04; fails on ee0ad04, where the name does not exist) =
    (3 pi/4)/I^2, I = int_0^1 (1 - t^2)/sqrt(t^2 ln t^2 - t^2 + 1) dt (DG audit doc, GATE B).

    Pinned two independent ways: (1) I by scipy quadrature here (not the mpmath value the literal was
    rounded from; F(t) cancels near t = 1, so double-precision quad lands 4.0e-14 low -- MEASURED
    1.1872068904949138 vs 1.18720689049496104 -- hence rel 1e-12 on gamma, not 1e-15), (2) the PHYSICS: dg_correct_density_1d's own BVP on a flat profile at
    gamma = GAMMA_WALL must carry the exact Schrodinger free-gas deficit sqrt(pi)/(2 sqrt 2) c
    hbar/sqrt(m kT), and at gamma = 1 the known 0.7734 of it. MEASURED 2026-11-04 (2001 nodes over
    30 L_q, tol 1e-6): -2.1e-10 relative at GAMMA_WALL, ratio 0.773430 at gamma = 1; deficit/(c L_q)
    = 1.1872068902 = I at both gammas (it is gamma-free by construction)."""
    from scipy.integrate import quad
    from dynameta.carriers.density_gradient import GAMMA_WALL
    from dynameta.constants import KB, T_REF
    i_quad = quad(lambda t: (1.0 - t * t) / np.sqrt(t * t * np.log(t * t) - t * t + 1.0), 0.0, 1.0,
                  epsabs=1e-13, epsrel=1e-13, limit=400)[0]   # tighter -> quad's roundoff warning
    assert i_quad == pytest.approx(1.187207, abs=5e-7)                  # the doc's I = 1.187207
    assert GAMMA_WALL == pytest.approx((3.0 * np.pi / 4.0) / i_quad ** 2, rel=1e-12)
    assert round(GAMMA_WALL, 4) == 1.6717                                 # the doc's quoted value
    exact = np.sqrt(np.pi) / (2.0 * np.sqrt(2.0))                         # 0.626657 c hbar/sqrt(m kT)
    c = 1e24
    unit = c * HBAR / np.sqrt(MSTAR * KB * T_REF)
    ratio = {}
    for g in (1.0, GAMMA_WALL):
        z = np.linspace(0.0, 30.0 * dg_length_m(MSTAR, gamma=g), 2001)
        n = dg_correct_density_1d(z, np.full_like(z, c), MSTAR, gamma=g)
        f = c - n
        ratio[g] = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(z))) / unit / exact
    assert abs(ratio[GAMMA_WALL] - 1.0) < 1e-8
    assert ratio[1.0] == pytest.approx(0.7734, abs=1e-4)
    # the in-Newton module re-exports the SAME object (one constant, two import homes)
    pytest.importorskip("devsim")
    from dynameta.carriers import physics_density_gradient as PDG
    assert PDG.GAMMA_WALL is GAMMA_WALL and "GAMMA_WALL" in PDG.__all__


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




# ---- v0.15 item 4.4.4 (decision 53): the unipolar DG WALL is setup_dg_insulator_wall --------------------
# setup_dg_hard_wall (audit X-6's export) pinned the carrier row (n = u^2) and Lambda (-k V_t): the wall
# node was a carrier SINK and the solve converged to a spurious wide-depletion branch (the deficit was
# pinned here as a CHARACTERIZATION, 3-1000 n0 L_q). It is now a deprecated shim onto the conserving
# insulator wall; the old rows survive only as the fixture tests/_dg_legacy_wall.py, which these gates use
# for their "fails on the old code" halves. Fixture: a uniform unipolar bar (m* = 0.35 m0, mu 0.004),
# the wall contact carries the DG wall rows only, the right contact is bulk-ohmic; psi FROZEN flat
# (PotentialEquation deleted), so the DG closes in closed form (D2d audit doc, GATE B):
#     z(t) = L_q int_0^t ds/sqrt(F(s)),  F(t) = t^2 ln t^2 - t^2 + 1,  deficit = I n0 L_q,  I = 1.187207.
# The unipolar region always carries the Fermi-Dirac g-factor, so the EXACT continuum answer of the model
# is the FD first integral (_fd_first_integral_deficit, written here from the zero-current SG relation
# dLambda = V_t g dln n and b u'' = Lambda u); it equals I in the Boltzmann limit and sits 2.4e-5 below I
# at n0 = 1e21 m^-3 (n0/N_c = 1.9e-4). Mesh: nodes x_k = L (e^{beta k/N} - 1)/(e^beta - 1), last/first
# spacing 40, so halving h0 doubles N and NESTS the meshes (DEVSIM's graded 1-D lines gave orders 1.2-1.5
# from their segment joints; measured 2026-11-06).

_WB_MSTAR, _WB_MU = 0.35 * M_E, 0.004


def _wall_bar(tag, *, n0, ps_wall=0.05e-9, gamma=1.0, wall="insulator", u_floor_rel=1e-6,
              bipolar=False, ext=False, rel=1e-8, b_scale=1.0, length=100e-9,
              fracs=(0.05, 0.1, 0.25, 0.5, 1.0)):
    """Solve the frozen-psi DG wall bar; returns dict(x, Electrons, QSqrtN, QLambda (sorted by x),
    J = ElectronCurrentDG on every edge, I_back, its, L_q). wall: "insulator" | "shim" (the deprecated
    name, warning caught) | "legacy" (the old rows) | "legacy_nocrow" (the old rows minus the pinned
    carrier row). bipolar: the same bar as a physics_bipolar_dd region (n-type, n_i 1e10) with
    setup_bipolar_dg. ext: extended-precision Newton. b_scale: multiply b_dg_full (a refutation knob)."""
    import contextlib
    import os
    import sys
    import warnings

    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import extended_precision as XP
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_density_gradient as DG
    from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                           setup_semiconductor_region_dd)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _dg_legacy_wall as LW

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

    its = []

    def _solve():
        # abs criterion off (1e30): the RELATIVE update test binds; REL 1e-8 is attainable in double on
        # the Boltzmann-regime bars, the degenerate 4e26 bar needs the extended Newton (D2d sec. 1.6)
        with _quiet():
            info = (XP.solve if ext else ds.solve)(type="dc", absolute_error=1e30, relative_error=rel,
                                                   maximum_iterations=60, info=True)
        its.append(len(info["iterations"]))
        assert info["converged"], "{}: Newton did not converge ({})".format(tag, its)

    mesh, dev, reg = "wbm_" + tag, "wbd_" + tag, "bar"
    lq = dg_length_m(_WB_MSTAR, gamma=gamma)
    try:
        with _quiet():
            ds.create_1d_mesh(mesh=mesh)
            beta = np.log(40.0)
            nseg = int(round(length * beta / (ps_wall * np.expm1(beta))))
            xs = length * np.expm1(beta * np.arange(nseg + 1) / nseg) / np.expm1(beta)
            gaps = np.diff(xs)
            for k, xk in enumerate(xs):
                ds.add_1d_mesh_line(mesh=mesh, pos=float(xk), ps=1.01 * float(gaps[min(k, nseg - 1)]),
                                    **({"tag": "wall"} if k == 0 else
                                       {"tag": "back"} if k == nseg else {}))
            ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
            ds.add_1d_contact(mesh=mesh, name="back", tag="back", material="metal")
            ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="wall", tag2="back")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            if ext:
                XP.enable(dev)
            if not bipolar:
                setup_semiconductor_region_dd(dev, reg, n_bg_m3=n0, eps_static=9.5,
                                              dos_mass_kg=_WB_MSTAR, mobility_m2Vs=_WB_MU)
                setup_contact_ohmic_dd(dev, "back")
            else:
                nc = _nc_model()                         # = the unipolar region's N_c (same mass, T)
                ds.node_model(device=dev, region=reg, name="NetDoping", equation="{:.17e}".format(n0))
                BP.setup_bipolar_region(dev, reg, eps_static=9.5, n_dos_m3=nc, n_i_m3=1e10,
                                        mobility_n_m2Vs=_WB_MU, mobility_p_m2Vs=_WB_MU, tau_n_s=1e-7,
                                        tau_p_s=1e-7, fd_enhancement=True, n_dos_p_m3=nc)
                BP.setup_equilibrium_seed_models(dev, reg)
                BP.setup_contact_ohmic_bipolar(dev, "back")
                for nm, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
                    ds.set_node_values(device=dev, region=reg, name=nm, init_from=src)
                ds.node_model(device=dev, region=reg, name="_seed_psi",
                              equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, reg)))
                ds.set_node_values(device=dev, region=reg, name="Potential", init_from="_seed_psi")
        _solve()                                         # classical equilibrium (flat)
        with _quiet():
            if not bipolar:
                DG.setup_dg_quantum_correction(dev, reg, m_eff_kg=_WB_MSTAR, gamma=gamma)
                DG.setup_contact_dg(dev, "back", n0)
            else:
                DG.setup_bipolar_dg(dev, reg, m_eff_n_kg=_WB_MSTAR, gamma_n=gamma)
                DG.setup_contact_bipolar_dg(dev, "back")
            if b_scale != 1.0:
                b_full = float(ds.get_parameter(device=dev, region=reg, name="b_dg_full"))
                ds.set_parameter(device=dev, region=reg, name="b_dg_full", value=b_scale * b_full)
            if wall.startswith("legacy"):
                LW.install_legacy_hard_wall(dev, "wall", lambda_pin_factor=8.0,
                                            carrier_row=(wall == "legacy"))
            elif wall == "shim":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    DG.setup_dg_hard_wall(dev, "wall", lambda_pin_factor=8.0)
            else:
                DG.setup_dg_insulator_wall(dev, "wall", u_floor_rel=u_floor_rel)
            _R.delete_by_name(dev, "PotentialEquation")     # freeze psi at the flat solution
        for k, fr in enumerate(fracs):
            DG.set_dg_gamma(dev, reg, fr)
            if wall.startswith("legacy"):
                LW.set_pin_fraction(dev, fr)
            if k == 0:
                if wall.startswith("legacy"):            # the old fixture's taper recipe, as shipped
                    DG.seed_dg_from_solution(dev, reg)
                    z0 = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="x"))
                    u0 = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="QSqrtN"))
                    taper = np.maximum(np.tanh(z0 / 1.2e-9), 1e-6)
                    ds.set_node_values(device=dev, region=reg, name="QSqrtN", values=list(u0 * taper))
                    ds.set_node_values(device=dev, region=reg, name="Electrons",
                                       values=list(np.maximum((u0 * taper) ** 2, 1e14)))
                elif bipolar:
                    DG.seed_bipolar_dg(dev, reg, walls=("wall",), u_floor_rel=u_floor_rel)
                else:
                    DG.seed_dg_from_solution(dev, reg, walls=("wall",), u_floor_rel=u_floor_rel)
            _solve()
        get = (lambda name: np.asarray(ds.get_node_model_values(device=dev, region=reg, name=name)))
        o = np.argsort(get("x"))
        out = {k: get(k)[o] for k in ("x", "Electrons", "QSqrtN", "QLambda")}
        out["J"] = np.asarray(ds.get_edge_model_values(device=dev, region=reg, name="ElectronCurrentDG"))
        out["I_back"] = float(ds.get_contact_current(device=dev, contact="back",
                                                     equation="ElectronContinuityEquation"))
        out["its"], out["L_q"] = its, lq
    finally:
        with _quiet():
            try:
                _R.clear(dev)
                ds.delete_device(device=dev)
                ds.delete_mesh(mesh=mesh)
            except Exception:                            # nothing to tear down (build failed)
                pass
    return out


def _closed_form_I():
    from scipy.integrate import quad
    return quad(lambda t: (1.0 - t * t) / np.sqrt(t * t * np.log(t * t) - t * t + 1.0) if t > 0 else 1.0,
                0.0, 1.0, limit=400, epsabs=1e-13, epsrel=1e-12)[0]


def _closed_form_depth(t):
    """z(t)/L_q = int_0^t ds/sqrt(F(s)) of the Boltzmann closed form."""
    from scipy.integrate import quad
    return quad(lambda s: 1.0 / np.sqrt(s * s * np.log(s * s) - s * s + 1.0) if s > 0 else 1.0, 0.0, t,
                limit=400, epsabs=1e-13, epsrel=1e-12)[0]


def _fd_first_integral_deficit(w):
    """The EXACT continuum dark-space deficit / (n0 L_q) of the frozen-flat-psi DG wall with the Fermi-
    Dirac g-factor, w = n0/N_c -- derived here, independent of the DEVSIM code: zero current gives
    dLambda = V_t g(n/N_c) dln n, so Lambda(t)/V_t = 2 ln t + Delta(w t^2) - Delta(w), t = u/u_b,
    Delta(x) = int_0^x (g - 1) dy/y (scipy quad of the einstein g fit, in s = y^(1/3)); b u'' = Lambda u
    integrates once to (u')^2 = (2 V_t u_b^2/b) G(t), G(t) = int_t^1 (-Lambda/V_t) tau dtau, so
    deficit/(n0 L_q) = int_0^1 (1 - t^2)/sqrt(2 G(t)) dt. Boltzmann (w -> 0): 2G = F, the closed form I."""
    from scipy.integrate import quad
    from dynameta.carriers.einstein import g_einstein

    def delta(x):
        if x <= 0.0:
            return 0.0
        return quad(lambda s: 3.0 * (g_einstein(s ** 3) - 1.0) / s if s > 0 else 0.0, 0.0, x ** (1.0 / 3.0),
                    epsabs=1e-15, epsrel=1e-13, limit=200)[0]

    dw = delta(w)
    lam = (lambda tau: -(2.0 * np.log(tau) + delta(w * tau * tau) - dw))
    big_g = (lambda t: quad(lambda tau: lam(tau) * tau, t, 1.0, epsabs=0.0, epsrel=1e-12, limit=200)[0])
    return quad(lambda t: (1.0 - t * t) / np.sqrt(2.0 * big_g(t)), 0.0, 1.0, epsabs=0.0, epsrel=1e-10,
                limit=200)[0]


def _nc_model():
    return 2.0 * (_WB_MSTAR * KB * T_REF / (2.0 * np.pi * HBAR ** 2)) ** 1.5


def _deficit(r, n0):
    from dynameta.core.numerics import trapz
    return float(trapz(n0 - r["Electrons"], r["x"])) / (n0 * r["L_q"])


def _flux_scale(n0, lq):
    return Q * _WB_MU * n0 * V_T / lq


def test_dg_hard_wall_input_guard():
    """audit X-6: the export's guard -- still validated by the shim, BEFORE any warning."""
    pytest.importorskip("devsim")
    import warnings
    from dynameta.carriers.physics_density_gradient import setup_dg_hard_wall
    for bad in (0.0, -1.0):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="lambda_pin_factor"):
                setup_dg_hard_wall("dev", "wall", lambda_pin_factor=bad)




def test_dg_hard_wall_is_a_deprecated_shim_onto_the_insulator_wall():
    """Decision 53: setup_dg_hard_wall warns (DeprecationWarning naming the removal release and the
    MEASURED change) and installs exactly setup_dg_insulator_wall: the whole unipolar DG definition call
    log through the shim equals the one through setup_dg_insulator_wall, call for call (tests/
    _dg_calllog.py; same-tree A/B). It creates no wall_lambda_pin parameter, so set_dg_gamma moves b_dg
    only. On 9003030 the two logs differed (the pin parameters and their co-ramp, the Lambda row
    'QLambda + wall_lambda_pin', the pinned carrier row) and no warning was issued."""
    pytest.importorskip("devsim")
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _dg_calllog as CL
    with pytest.warns(DeprecationWarning, match="removed in v0.16") as rec:
        shim = CL.record_call_log(hard_wall_shim=True)
    msg = str(rec[0].message)
    for needle in ("setup_dg_insulator_wall", "1.187207", "sink", "23-28 n0 L_q", "walls=(contact,)"):
        assert needle in msg, needle
    assert shim == CL.record_call_log()
    assert not any("wall_lambda_pin" in repr(c) for c in shim)


def test_set_dg_gamma_legacy_pin_branch_still_co_ramps_and_refuses_a_bad_pin():
    """set_dg_gamma keeps the hard wall's pin co-ramp for ONE release (decision 53: a device built by an
    old script may still carry `wall_lambda_pin_full`): with no pin parameter it moves b_dg only; with
    the legacy rows (tests/_dg_legacy_wall.py) it co-ramps the pin; audit C-17's tail stays -- a pin that
    EXISTS but cannot be read raises instead of being taken for "no hard wall"."""
    pytest.importorskip("devsim")
    import os
    import sys
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.physics_density_gradient import (dg_b_coefficient, set_dg_gamma,
                                                            setup_dg_insulator_wall,
                                                            setup_dg_quantum_correction)
    from dynameta.carriers.physics_drift_diffusion import setup_semiconductor_region_dd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _dg_legacy_wall as LW
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
        setup_semiconductor_region_dd(dev, reg, n_bg_m3=4e26, eps_static=9.5, dos_mass_kg=_WB_MSTAR,
                                      mobility_m2Vs=_WB_MU)
        setup_dg_quantum_correction(dev, reg, m_eff_kg=_WB_MSTAR, gamma=1.0)
        setup_dg_insulator_wall(dev, "wall")
        b_full = dg_b_coefficient(_WB_MSTAR, 1.0)
        set_dg_gamma(dev, reg, 0.25)
        assert ds.get_parameter(device=dev, region=reg, name="b_dg") == pytest.approx(0.25 * b_full)
        assert "wall_lambda_pin" not in ds.get_parameter_list(device=dev)
        LW.install_legacy_hard_wall(dev, "wall", lambda_pin_factor=8.0)     # an old script's rows
        for frac in (0.05, 0.5, 1.0):
            set_dg_gamma(dev, reg, frac)
            assert ds.get_parameter(device=dev, name="wall_lambda_pin") == pytest.approx(
                frac * 8.0 * V_T, rel=1e-12)
        ds.set_parameter(device=dev, name="wall_lambda_pin_full", value="not-a-number")
        with pytest.raises(ValueError):
            set_dg_gamma(dev, reg, 0.5)
    finally:
        try:
            _R.clear(dev)
            ds.delete_device(device=dev)
            ds.delete_mesh(mesh=mesh)
        except Exception:
            pass


def test_insulator_wall_reproduces_the_closed_form_dark_space_and_conserves():
    """G2-1 + G2-5 (plan P4 sec. 2.5), the load-bearing gates of 4.4.4. Bar n0 = 1e21 m^-3 (n0/N_c =
    1.9e-4: Boltzmann holds to the 2.4e-5 FD residue), h0 = 0.05 nm (L_q = 1.18 nm), gamma = 1.
      * deficit/(n0 L_q) vs the closed form I (scipy quadrature here): rel <= 2e-4 -- the plan's budget
        (bipolar GATE B measured 3e-6..2e-4 at similar h/L_q, plus the FD residue). MEASURED 1.7e-5.
      * profile: max |z_node - z(t_node)|/L_q <= 1e-3 for 0.05 < t < 0.95 (MEASURED 2.8e-4).
      * conservation: max |ElectronCurrentDG| over every edge and the back-contact current <= 1e-12 of
        q mu n0 V_t/L_q, the size of one diffusion term (MEASURED 3e-15 / 1e-17: round-off of the
        zero-current balance).
    FAILS on the old code (the retired rows, tests/_dg_legacy_wall.py, run on the same bar): deficit
    22.85 n0 L_q -- the wide-depletion branch, inside the 3-1000 band the old characterization pinned --
    and the wall node a SINK, edge and terminal current 6.25e-3 of the scale (measured).
    Refutation: b_dg x 1.001 (L_q x 1.0005) moves the deficit by 5e-4 and must break the 2e-4 gate."""
    pytest.importorskip("devsim")
    n0 = 1e21
    ii = _closed_form_I()
    assert ii == pytest.approx(1.187207, abs=5e-7)
    r = _wall_bar("g21", n0=n0)
    d = _deficit(r, n0)
    assert abs(d / ii - 1.0) <= 2e-4, d
    t = np.sqrt(r["Electrons"] / n0)
    sel = np.flatnonzero((t > 0.05) & (t < 0.95))
    zerr = max(abs(r["x"][i] / r["L_q"] - _closed_form_depth(t[i])) for i in sel)
    assert zerr <= 1e-3, zerr
    scale = _flux_scale(n0, r["L_q"])
    assert np.max(np.abs(r["J"])) <= 1e-12 * scale
    assert abs(r["I_back"]) <= 1e-12 * scale
    assert r["QLambda"][0] / V_T == pytest.approx(np.log(1e-12), rel=1e-3)   # V_t ln(u_floor^2): -27.6
    # the OLD rows on the same bar: wide branch + sink
    old = _wall_bar("g21old", n0=n0, wall="legacy")
    assert 3.0 < _deficit(old, n0) < 1000.0
    assert np.max(np.abs(old["J"])) > 1e-3 * scale and abs(old["I_back"]) > 1e-3 * scale
    # refutation
    bad = _wall_bar("g21b", n0=n0, b_scale=1.001)
    assert abs(_deficit(bad, n0) / ii - 1.0) > 2e-4


def test_insulator_wall_dark_space_converges_at_second_order_to_the_fd_first_integral():
    """G2-1's refinement leg: on nested meshes h0 = 0.1 / 0.05 / 0.025 nm the deficit converges to the
    EXACT continuum value of the model -- the FD first integral (_fd_first_integral_deficit; the closed
    form I itself is not the limit: the unipolar g-factor puts the limit 2.4e-5 below it, MEASURED
    1.1871784 vs 1.1872069) -- at observed order >= 1.7 (box method O(h^2); MEASURED 2.12 / 2.16). The
    oracle reproduces I at w = 0 (Boltzmann) to 1e-9. On the old rows the deficit does not converge to
    it at all (22.85, the test above)."""
    pytest.importorskip("devsim")
    n0 = 1e21
    assert _fd_first_integral_deficit(0.0) == pytest.approx(_closed_form_I(), rel=1e-9)
    exact = _fd_first_integral_deficit(n0 / _nc_model())
    errs = [abs(_deficit(_wall_bar("o%d" % k, n0=n0, ps_wall=h), n0) / exact - 1.0)
            for k, h in enumerate((0.1e-9, 0.05e-9, 0.025e-9))]
    orders = [np.log2(errs[k] / errs[k + 1]) for k in range(2)]
    assert min(orders) >= 1.7, (errs, orders)


def test_dark_space_scales_as_sqrt_gamma_and_ignores_the_floor():
    """G2-2: the deficit (in m^-2, not per L_q) scales as L_q ~ sqrt(gamma): deficit(1)/deficit(1/4)
    = 2 +- 1e-3 (MEASURED 2.00016; the residue is the O(h^2/L_q^2) mesh error at the halved L_q).
    G2-6: u_floor_rel 1e-6 -> 1e-4 moves it by <= 1e-4 relative -- the floor shifts the effective wall
    by ~u_floor L_q, so |dD|/D <~ u_floor (MEASURED 8.3e-5, as the bipolar wall). Old code: the pinned
    rows give 22.85 at gamma = 1 (a wide branch with no sqrt(gamma) law)."""
    pytest.importorskip("devsim")
    from dynameta.core.numerics import trapz
    n0 = 1e21
    a = {}
    for g in (1.0, 0.25):
        fr = tuple(sorted({min(1.0, f / g) for f in (0.05, 0.1, 0.25, 0.5, 1.0)}))
        r = _wall_bar("gs%g" % g, n0=n0, gamma=g, fracs=fr)
        a[g] = float(trapz(n0 - r["Electrons"], r["x"]))
    assert a[1.0] / a[0.25] == pytest.approx(2.0, abs=1e-3)
    d6 = _deficit(_wall_bar("fl6", n0=n0), n0)
    d4 = _deficit(_wall_bar("fl4", n0=n0, u_floor_rel=1e-4), n0)
    assert abs(d4 / d6 - 1.0) <= 1e-4


def test_unipolar_wall_equals_the_bipolar_dg_on_the_same_bar():
    """G2-4: the unipolar DG (setup_dg_quantum_correction) and the bipolar DG (setup_bipolar_dg, electrons
    only, n-type, n_i = 1e10 so holes are ~1e-3 m^-3) with the SAME insulator wall on the SAME frozen-psi
    bar are one discrete system (one kernel, one SG flux): <= 1e-10 relative in Electrons / QSqrtN and
    in QLambda (to its max). Extended Newton to REL 1e-12 (double cannot reach 1e-10 here: the Lambda
    tail's relative criterion, D2d sec. 1.6). MEASURED: densities bit-identical, QLambda 9e-19. So the
    bipolar GATE C Schrodinger-Poisson validation covers the unipolar wall too."""
    pytest.importorskip("devsim")
    n0 = 1e23
    ru = _wall_bar("ub_u", n0=n0, ext=True, rel=1e-12)
    rb = _wall_bar("ub_b", n0=n0, ext=True, rel=1e-12, bipolar=True)
    for k in ("Electrons", "QSqrtN"):
        assert np.max(np.abs(rb[k] / ru[k] - 1.0)) <= 1e-10, k
    assert np.max(np.abs(rb["QLambda"] - ru["QLambda"])) <= 1e-10 * np.max(np.abs(ru["QLambda"]))


def test_degenerate_ito_bar_converges_to_the_fd_first_integral():
    """G2-7, the old fixture (4e26 m^-3 ITO, n0/N_c = 77, g = 13): no closed form, so the reference is
    the FD first integral above (0.350170 n0 L_q). FINDING (2026-11-06): the plan's band "deficit in
    [0.7, 1.5] n0 L_q" does NOT hold -- the g-factor shortens the dark space ~sqrt(g) (audit C-8), and
    the model's exact answer is 0.350; the band was written without it. Gated instead: nested meshes
    h0 = 0.1 / 0.05 / 0.025 nm converge to the first integral at observed order >= 1.8 (MEASURED 1.94 /
    1.98; errors 3.6e-3 / 9.5e-4 / 2.4e-4), and the Richardson extrapolation of the last two lies
    within 2e-5 of it (budget: the next-order term, ~ the observed error ratio's departure from 4 times
    the finest error; MEASURED 1.1e-6 with the 0.0125 nm mesh). Extended Newton (REL 1e-10 is not
    attainable in double at 4e26: measured 60-iteration stalls). On the old rows the same bar sits at
    28.0 n0 L_q (measured on the old test's 0.2 nm mesh)."""
    pytest.importorskip("devsim")
    n0 = 4e26
    exact = _fd_first_integral_deficit(n0 / _nc_model())
    assert exact == pytest.approx(0.35017, abs=1e-5)
    d = [_deficit(_wall_bar("ito%d" % k, n0=n0, ps_wall=h, ext=True, rel=1e-10), n0)
         for k, h in enumerate((0.1e-9, 0.05e-9, 0.025e-9))]
    e = [abs(x / exact - 1.0) for x in d]
    orders = [np.log2(e[k] / e[k + 1]) for k in range(2)]
    assert min(orders) >= 1.8, (e, orders)
    rich = d[2] + (d[2] - d[1]) / 3.0
    assert abs(rich / exact - 1.0) <= 2e-5, rich


def test_the_pinned_carrier_row_alone_is_the_sink():
    """G2-7's negative control: the old rows MINUS only the pinned carrier row (u and Lambda still
    pinned, the carrier row natural) conserve and give the closed-form dark space (deficit within 2e-4
    of I, MEASURED 3.5e-5 vs the FD integral; wall current <= 1e-12 of the scale) -- so the carrier row,
    not the Lambda pin, was the wide-branch cause. (Adding that row to setup_dg_insulator_wall instead is
    impossible: its Lambda row is the same tie c - u^2, a singular matrix -- measured, UMFPACK fails.)"""
    pytest.importorskip("devsim")
    n0 = 1e21
    r = _wall_bar("nocrow", n0=n0, wall="legacy_nocrow")
    assert abs(_deficit(r, n0) / _closed_form_I() - 1.0) <= 2e-4
    assert np.max(np.abs(r["J"])) <= 1e-12 * _flux_scale(n0, r["L_q"])


def test_seed_dg_from_solution_without_walls_is_the_old_seed():
    """D2-b: seed_dg_from_solution now delegates to the ONE seed kernel (_seed_dg, shared with
    seed_bipolar_dg); walls=() writes exactly the pre-v0.15 values -- QSqrtN = sqrt(max(n, 0)) and
    QLambda = 0.0, bit for bit -- on a non-uniform density (a doping step bar)."""
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.physics_density_gradient import (seed_dg_from_solution,
                                                            setup_dg_quantum_correction)
    from dynameta.carriers.physics_drift_diffusion import setup_semiconductor_region_dd
    mesh, dev, reg = "seedm", "seedd", "bar"
    try:
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="l")
        ds.add_1d_mesh_line(mesh=mesh, pos=20e-9, ps=1e-9, tag="r")
        ds.add_1d_contact(mesh=mesh, name="l", tag="l", material="metal")
        ds.add_1d_contact(mesh=mesh, name="r", tag="r", material="metal")
        ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="l", tag2="r")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        setup_semiconductor_region_dd(dev, reg, n_bg_m3=4e26, eps_static=9.5, dos_mass_kg=_WB_MSTAR,
                                      mobility_m2Vs=_WB_MU)
        setup_dg_quantum_correction(dev, reg, m_eff_kg=_WB_MSTAR)
        nn = len(ds.get_node_model_values(device=dev, region=reg, name="x"))
        n = np.geomspace(3e25, 4e26, nn) * (1.0 + 0.1 * np.sin(np.arange(nn)))
        ds.set_node_values(device=dev, region=reg, name="Electrons", values=list(n))
        seed_dg_from_solution(dev, reg)
        u = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="QSqrtN"))
        lam = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="QLambda"))
        n_back = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
        assert np.array_equal(u, np.sqrt(np.maximum(n_back, 0.0)))
        assert np.array_equal(lam, np.zeros(nn)) and not np.any(np.signbit(lam))
        assert np.array_equal(n_back, np.asarray(n))                  # walls=() leaves Electrons alone
    finally:
        try:
            _R.clear(dev)
            ds.delete_device(device=dev)
            ds.delete_mesh(mesh=mesh)
        except Exception:
            pass
