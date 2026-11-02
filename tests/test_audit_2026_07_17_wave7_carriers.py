"""Gates for the still-open carriers rows of the 2026-07-17 exhaustive audit, closed in wave 7
(2026-11-02): S1-4, S1-5, S1-8, S1-9, S1-10, S1-11, S1-13, S1-14, S1-18, S5-16.

Every CORRECTNESS gate here fails on the pre-wave-7 tree (the number it measured then is quoted next
to it); the COVERAGE gates (S1-8, S1-18) pin a previously-untested path against an independent
oracle -- a closed-form modal series, the exact theta-method recursion, an energy balance, a 40-digit
polylog -- tight enough that a load applied one step late, or a branch off by an ulp-scale
cancellation, fails them. Solver-dependent gates importorskip their solver.

Run: python -m pytest -p no:benchmark tests/test_audit_2026_07_17_wave7_carriers.py -q
"""
import ast
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import dynameta

PKG = pathlib.Path(dynameta.__file__).resolve().parent

# a child interpreter in which `import devsim` fails exactly as on a devsim-free install
_NO_DEVSIM = "import sys; sys.modules['devsim'] = None; "


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(PKG.parent), timeout=300)


# ============================================ S1-4 ================================================

def test_s1_4_star_import_is_eager_and_the_plain_import_is_not():
    """S1-4 (= 2026-07-25 V-13): `__all__` names the DEVSIM builder, so `from dynameta.carriers
    import *` is eager BY DEFINITION -- documented, not trimmed (trimming would un-declare the
    headline builder). This pins the documented truth from both sides on a devsim-FREE install: the
    plain import and a named pure-numpy submodule import work, the star form raises devsim's
    ImportError (it does not half-succeed)."""
    out = _run(_NO_DEVSIM + "import dynameta.carriers as c; from dynameta.carriers import "
               "schrodinger_poisson; print('plain-ok');\n"
               "try:\n    exec('from dynameta.carriers import *')\n"
               "except ImportError:\n    print('star-raises')\n")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["plain-ok", "star-raises"]


# ============================================ S1-10 ===============================================

@pytest.mark.parametrize("mod", ["devsim_3d", "devsim_layered"])
def test_s1_10_devsim_free_install_gets_the_friendly_hint(mod):
    """S1-10: devsim_3d carried four in-method `import devsim as ds` (a lazy-import ceremony) but
    imported physics_drift_diffusion / physics_bipolar_dd / dc_solve -- all devsim-at-top -- so a
    devsim-free install got the raw `import of devsim halted; None in sys.modules` from a sibling,
    not devsim_layered's install hint. Both builders now fail the same, actionable way."""
    out = _run(_NO_DEVSIM + "import dynameta.carriers.{}".format(mod))
    assert out.returncode != 0
    assert "requires the optional 'devsim' package" in out.stderr, out.stderr[-600:]
    assert "pip install dynameta[solvers]" in out.stderr


def test_s1_10_no_in_method_devsim_imports_remain():
    tree = ast.parse((PKG / "carriers" / "devsim_3d.py").read_text(encoding="utf-8"))
    nested = [n.lineno for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
              for n in ast.walk(f) if isinstance(n, ast.Import)
              and any(a.name == "devsim" for a in n.names)]
    assert nested == [], "in-method `import devsim` at lines {}".format(nested)


# ============================================ S5-16 ===============================================

def test_s5_16_no_aliased_warn_call_escapes_the_v17_census():
    """S5-16: the V-17 census (test_audit_2026_07_25_conventions) matches the literal
    `warnings.warn`; three carriers sites called it through `import warnings as _w; _w.warn(...)`
    with no category and no stacklevel (devsim_3d x1, devsim_layered x2), and one of them sat inside
    `try: ... except Exception: pass`, which SWALLOWED the warning whenever it was escalated to an
    error. Census: no `warn` call through an alias of the warnings module anywhere in dynameta/."""
    bad = []
    for p in sorted(PKG.rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mod_alias = {a.asname for n in ast.walk(tree) if isinstance(n, ast.Import)
                     for a in n.names if a.name == "warnings" and a.asname}
        fn_alias = {(a.asname or a.name) for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) and n.module == "warnings"
                    for a in n.names if a.name == "warn"}
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if (isinstance(f, ast.Attribute) and f.attr == "warn" and isinstance(f.value, ast.Name)
                    and f.value.id in mod_alias) or (isinstance(f, ast.Name) and f.id in fn_alias):
                bad.append("{}:{}".format(p.relative_to(PKG).as_posix(), n.lineno))
    assert not bad, "aliased warn() calls (invisible to the V-17 census): {}".format(bad)


# ============================================ S1-5 ================================================

def _dg_step_bar(tag, *, x_step, length=40e-9, bias=0.01):
    """1-D unipolar ITO-like bar, 4e26 -> 4e25 m^-3 doping step at x_step, ohmic DG contacts, gamma
    ramped to 1, `bias` V on the right contact. Returns (I_left, I_right, J_DG edges, recorded
    contact edge_current_models); the device is torn down."""
    import contextlib
    import os

    ds = pytest.importorskip("devsim")
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.physics_density_gradient import (seed_dg_from_solution, set_dg_gamma,
                                                            setup_contact_dg,
                                                            setup_dg_quantum_correction)
    from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                           setup_semiconductor_region_dd)
    from dynameta.constants import M_E

    @contextlib.contextmanager
    def _quiet():
        sys.stdout.flush()
        saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(dn, 1)
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(dn)
            os.close(saved)

    def _solve():
        with _quiet():
            ds.solve(type="dc", absolute_error=1e16, relative_error=1e-6, maximum_iterations=200)

    mstar, nhi, nlo = 0.35 * M_E, 4e26, 4e25
    mesh, dev, reg = "s15m_" + tag, "s15d_" + tag, "bar"
    try:
        with _quiet():
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="left")
            ds.add_1d_mesh_line(mesh=mesh, pos=x_step, ps=0.12e-9)
            ds.add_1d_mesh_line(mesh=mesh, pos=length, ps=0.3e-9, tag="right")
            for c in ("left", "right"):
                ds.add_1d_contact(mesh=mesh, name=c, tag=c, material="metal")
            ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="left", tag2="right")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            setup_semiconductor_region_dd(dev, reg, n_bg_m3=nhi, eps_static=9.5, dos_mass_kg=mstar,
                                          mobility_m2Vs=0.004)
            prof = ("({hi:.8e}*0.5*(1.0-tanh((x-{xj:.8e})/3.0e-10)) + "
                    "{lo:.8e}*0.5*(1.0+tanh((x-{xj:.8e})/3.0e-10)))").format(hi=nhi, lo=nlo, xj=x_step)
            ds.node_model(device=dev, region=reg, name="DopingProf", equation=prof)
            ds.node_model(device=dev, region=reg, name="PotentialNodeCharge",
                          equation="ElectronCharge * (Electrons - DopingProf)")
            ds.node_model(device=dev, region=reg, name="PotentialNodeCharge:Electrons",
                          equation="ElectronCharge")
            ds.node_model(device=dev, region=reg, name="PotentialNodeCharge:Potential", equation="0")
            for c in ("left", "right"):
                setup_contact_ohmic_dd(dev, c)
            ds.contact_node_model(device=dev, contact="right", name="right_electrons_dirichlet",
                                  equation="Electrons - {:.8e}".format(nlo))
            nd = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="DopingProf"))
            ds.set_node_values(device=dev, region=reg, name="Electrons", values=list(nd))
        _solve()
        with _quiet():
            setup_dg_quantum_correction(dev, reg, m_eff_kg=mstar, gamma=1.0)
            for c, nc in (("left", nhi), ("right", nlo)):
                setup_contact_dg(dev, c, nc)
            seed_dg_from_solution(dev, reg)
            ds.set_parameter(device=dev, name="right_bias", value=float(bias))
        for fr in (0.0, 0.25, 0.5, 0.75, 1.0):
            set_dg_gamma(dev, reg, fr)
            _solve()
        iL = float(ds.get_contact_current(device=dev, contact="left",
                                          equation="ElectronContinuityEquation"))
        iR = float(ds.get_contact_current(device=dev, contact="right",
                                          equation="ElectronContinuityEquation"))
        jdg = np.asarray(ds.get_edge_model_values(device=dev, region=reg, name="ElectronCurrentDG"))
        # read the registry directly (not via the new eq_registry.recorded) so this gate runs -- and
        # fails on the conservation numbers -- against the pre-fix tree too
        models = {c: [e["kwargs"].get("edge_current_model") for e in _R._REG.get(dev, [])
                      if e["scope"] == "contact" and e["loc"] == c
                      and e["name"] == "ElectronContinuityEquation"]
                  for c in ("left", "right")}
    finally:
        with _quiet():
            try:
                _R.clear(dev)
                ds.delete_device(device=dev)
                ds.delete_mesh(mesh=mesh)
            except Exception:
                pass
    return iL, iR, jdg, models


def test_s1_5_dg_terminal_current_is_the_density_gradient_flux():
    """S1-5: the ohmic CONTACT equation kept edge_current_model='ElectronCurrent' (classical SG on
    psi) after the in-Newton DG re-pointed the region continuity at ElectronCurrentDG, so the
    terminal current omitted the q mu n grad(Lambda) term on the contact edge. With the doping step
    3 nm from the right contact (max |Lambda| 10 mV) at 10 mV: pre-fix |I_L + I_R|/|I_L| = 2.95e-2
    (the right contact read the classical flux), post-fix 4.4e-16 -- and both terminals equal the
    divergence-free interior DG current."""
    iL, iR, jdg, models = _dg_step_bar("a", x_step=37e-9)
    j = float(np.mean(jdg))
    assert (np.max(jdg) - np.min(jdg)) / abs(j) < 1e-10         # steady 1-D: J_DG uniform
    assert abs(iL + iR) / abs(iL) < 1e-10, abs(iL + iR) / abs(iL)
    assert abs(abs(iR) - abs(j)) / abs(j) < 1e-10
    assert abs(abs(iL) - abs(j)) / abs(j) < 1e-10
    assert models == {"left": ["ElectronCurrentDG"], "right": ["ElectronCurrentDG"]}


def test_s1_5_contact_dg_before_the_quantum_correction_refuses():
    """The re-point needs ElectronCurrentDG on the contact's region: the documented order (quantum
    correction FIRST) is now enforced instead of silently leaving the classical current bound."""
    ds = pytest.importorskip("devsim")
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.physics_density_gradient import setup_contact_dg
    from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                           setup_semiconductor_region_dd)
    from dynameta.constants import M_E
    mesh, dev = "s15om", "s15od"
    try:
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="l")
        ds.add_1d_mesh_line(mesh=mesh, pos=20e-9, ps=2e-9, tag="r")
        ds.add_1d_contact(mesh=mesh, name="l", tag="l", material="metal")
        ds.add_1d_contact(mesh=mesh, name="r", tag="r", material="metal")
        ds.add_1d_region(mesh=mesh, material="ITO", region="bar", tag1="l", tag2="r")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        setup_semiconductor_region_dd(dev, "bar", n_bg_m3=4e26, eps_static=9.5,
                                      dos_mass_kg=0.35 * M_E, mobility_m2Vs=0.004)
        setup_contact_ohmic_dd(dev, "l")
        with pytest.raises(ValueError, match="setup_dg_quantum_correction"):
            setup_contact_dg(dev, "l", 4e26)
    finally:
        _R.clear(dev)
        for dv in list(ds.get_device_list()):
            if dv == dev:
                ds.delete_device(device=dv)
        if mesh in ds.get_mesh_list():
            ds.delete_mesh(mesh=mesh)


# ============================================ S1-8 ================================================

def _modal_mean_rise(t, *, L, k, rc, Q_rate=0.0, F_rate=0.0, n_modes=20000):
    """Exact layer-mean temperature rise of the 1-D slab rho c T_t = k T_zz + Q(t), T(0) = 0 (sink),
    k T_z(L) = F(t) (top flux in), T(z, 0) = 0, for RAMPS Q = Q_rate t, F = F_rate t: eigenmodes
    sin(k_n z), k_n = (n + 1/2) pi / L, each driven by (2/L)[Q/k_n + F (-1)^n] / (rho c)."""
    n = np.arange(n_modes)[:, None]
    kn = (n + 0.5) * np.pi / L
    lam = (k / rc) * kn ** 2
    b = (2.0 / L) * (Q_rate / kn + F_rate * (-1.0) ** n)
    tt = np.asarray(t)[None, :]
    cn = (b / rc) * (tt / lam - (1.0 - np.exp(-lam * tt)) / lam ** 2)
    return np.sum(cn / (L * kn), axis=0)


@pytest.mark.parametrize("hook", ["joule_of_t", "flux_of_t"])
def test_s1_8_single_T_time_dependent_load_matches_the_modal_series(hook):
    """S1-8: the flux_of_t / joule_of_t paths (a LinearForm rebuilt and assembled every step, its
    .vec carried as f_old across steps) had no test. Oracle: the exact eigen-series of the slab for a
    RAMPED load, through half a diffusion time. Measured: Crank-Nicolson 100 steps 2.3e-5 (joule) /
    2.6e-5 (flux) of the final rise; backward Euler 200 steps 2.1e-3 / 2.3e-3 (first order). A load
    evaluated one step late is an O(dt/t_end) = 1e-2 error -- 400x the CN gate."""
    pytest.importorskip("ngsolve")
    from dynameta.carriers.thermal_fem import ThermalLayer, solve_thermal_transient_fem
    L, k, rho, cp, T0 = 200e-9, 1.0, 2.0e3, 1.0e3, 300.0
    rc = rho * cp
    tau = L * L * rc / k
    t_end = 0.5 * tau
    if hook == "joule_of_t":
        kw, rates = dict(joule_of_t=lambda t: 1e15 * t / tau), dict(Q_rate=1e15 / tau)
    else:
        kw, rates = dict(flux_of_t=lambda t: 1e8 * t / tau), dict(F_rate=1e8 / tau)
    for theta, ns, mh, tol in ((0.5, 100, 50e-9, 1e-4), (1.0, 200, 25e-9, 5e-3)):
        r = solve_thermal_transient_fem([ThermalLayer("m", L, k, rho, cp)], period_x_m=50e-9,
                                        period_y_m=50e-9, t_end_s=t_end, dt_s=t_end / ns,
                                        T_sink_K=T0, theta=theta, maxh_m=mh, order=2, **kw)
        ref = T0 + _modal_mean_rise(r.t_s, L=L, k=k, rc=rc, **rates)
        rise = ref[-1] - T0
        assert rise > 1.0
        err = np.max(np.abs(r.mean_T_per_layer_t[:, 0] - ref)) / rise
        assert err < tol, (hook, theta, err)
        assert r.steady_limit_T is None                       # C-9: a hook-driven run has no closed form


def test_s1_8_two_temp_source_hooks_reproduce_the_exact_theta_recursion():
    """S1-8: source_e_of_t / source_l_of_t (two-temperature transient). Insulated box + uniform
    sources: a uniform field is in the FE space and the stiffness annihilates it, so the FEM march
    must equal the 2x2 lumped theta-method recursion EXACTLY (measured 4.4e-14 CN / 1.2e-14 BE of the
    68 K rise) -- the gate that pins WHICH times each step's load is sampled at -- and approach the
    continuous ODE at second order under CN (3.4e-4 at 100 steps)."""
    pytest.importorskip("ngsolve")
    from scipy.integrate import solve_ivp

    from dynameta.carriers.thermal_fem import ThermalLayerTwoTemp, solve_thermal_transient_twotemp_fem
    L, ce, rho, cp, G, T0 = 100e-9, 1.0e4, 2.0e3, 1.0e3, 1.0e16, 300.0
    cl = rho * cp
    t_end = 5.0 / (G * (1.0 / ce + 1.0 / cl))
    w = 2.0 * np.pi / t_end
    se = lambda t: 1e18 * np.sin(w * t) ** 2                            # noqa: E731
    sl = lambda t: 3e17 * t / t_end                                       # noqa: E731
    lay = ThermalLayerTwoTemp(name="m", thickness_m=L, k_thermal=1.0, rho_kg_m3=rho, Cp_J_kgK=cp,
                              G_e_l=G, C_e_J_m3K=ce, k_electron=10.0)
    for theta, cont_tol in ((0.5, 1e-3), (1.0, None)):
        ns = 100
        dt = t_end / ns
        r = solve_thermal_transient_twotemp_fem([lay], period_x_m=100e-9, period_y_m=100e-9,
                                                t_end_s=t_end, dt_s=dt, T_sink_K=T0, theta=theta,
                                                maxh_m=50e-9, bottom_bc="insulated",
                                                source_e_of_t=se, source_l_of_t=sl)
        M = np.diag([ce, cl])
        Km = G * np.array([[1.0, -1.0], [-1.0, 1.0]])
        A, B = M + theta * dt * Km, M - (1.0 - theta) * dt * Km
        u, tr = np.array([T0, T0]), [np.array([T0, T0])]
        for n in range(ns):
            t, tn = n * dt, (n + 1) * dt
            f0, f1 = np.array([se(t), sl(t)]), np.array([se(tn), sl(tn)])
            u = np.linalg.solve(A, B @ u + dt * ((1.0 - theta) * f0 + theta * f1))
            tr.append(u)
        tr = np.asarray(tr)
        rise = float(np.max(tr - T0))
        assert rise > 10.0
        assert np.max(np.abs(r.mean_Te_per_layer_t[:, 0] - tr[:, 0])) < 1e-10 * rise
        assert np.max(np.abs(r.mean_Tl_per_layer_t[:, 0] - tr[:, 1])) < 1e-10 * rise
        if cont_tol is not None:
            sol = solve_ivp(lambda t, y: [(-G * (y[0] - y[1]) + se(t)) / ce,
                                          (G * (y[0] - y[1]) + sl(t)) / cl],
                            (0.0, float(r.t_s[-1])), [T0, T0], t_eval=r.t_s, rtol=1e-12,
                            atol=1e-12, method="DOP853")
            assert np.max(np.abs(r.mean_Te_per_layer_t[:, 0] - sol.y[0])) < cont_tol * rise


def test_s1_8_two_temp_flux_hook_conserves_energy_exactly():
    """S1-8: flux_of_t on the two-temperature solver (top-face flux into the ELECTRON field). Into
    an insulated box the discrete energy obeys E^{n+1} - E^n = dt [(1-theta) F^n + theta F^{n+1}]
    exactly (constants are in the space; stiffness and coupling annihilate them). Measured 1.1e-10 /
    1.9e-10 (CN / BE) -- the field is NOT uniform here (Te - Tl = 0.14 K at the end), so this pins the
    boundary-load path the uniform-source gate cannot."""
    pytest.importorskip("ngsolve")
    from dynameta.carriers.thermal_fem import ThermalLayerTwoTemp, solve_thermal_transient_twotemp_fem
    L, ce, rho, cp, G, T0 = 100e-9, 1.0e4, 2.0e3, 1.0e3, 1.0e16, 300.0
    cl = rho * cp
    t_end = 5.0 / (G * (1.0 / ce + 1.0 / cl))
    w = 2.0 * np.pi / t_end
    flux = lambda t: 1e8 * (1.0 + np.cos(w * t))                          # noqa: E731
    lay = ThermalLayerTwoTemp(name="m", thickness_m=L, k_thermal=1.0, rho_kg_m3=rho, Cp_J_kgK=cp,
                              G_e_l=G, C_e_J_m3K=ce, k_electron=10.0)
    for theta in (0.5, 1.0):
        ns = 50
        dt = t_end / ns
        r = solve_thermal_transient_twotemp_fem([lay], period_x_m=100e-9, period_y_m=100e-9,
                                                t_end_s=t_end, dt_s=dt, T_sink_K=T0, theta=theta,
                                                maxh_m=50e-9, bottom_bc="insulated", flux_of_t=flux)
        E = (ce * (r.mean_Te_per_layer_t[:, 0] - T0) + cl * (r.mean_Tl_per_layer_t[:, 0] - T0)) * L
        t = r.t_s
        e_ref = np.concatenate([[0.0], np.cumsum(dt * ((1.0 - theta) * flux(t[:-1])
                                                        + theta * flux(t[1:])))])
        assert np.max(np.abs(E - e_ref)) < 1e-8 * e_ref[-1]
        assert abs(r.mean_Te_per_layer_t[-1, 0] - r.mean_Tl_per_layer_t[-1, 0]) > 1e-2


# ============================================ S1-11 ===============================================

def test_s1_11_k_electron_default_is_deprecated_and_the_parallel_limit_is_exact():
    """S1-11: k_electron=None silently set the ELECTRON conductivity to the LATTICE k_thermal, so the
    documented G -> inf reduction carried k = 2 k_thermal (measured: 5.00 K mean rise vs the 10.0 K
    single-T answer for k_thermal = 1, 200 nm, 1e8 W/m^2). None now warns (value unchanged); an
    explicit split k_e + k_l = k reproduces the single-T solve."""
    pytest.importorskip("ngsolve")
    from dynameta.carriers.thermal_fem import (ThermalLayer, ThermalLayerTwoTemp, solve_thermal_fem,
                                               solve_thermal_twotemp_fem)
    from dynameta.warning_types import DynaMetaDeprecationWarning
    base = dict(name="m", thickness_m=200e-9, k_thermal=0.5, G_e_l=1e20, C_e_J_m3K=1e4)
    with pytest.warns(DynaMetaDeprecationWarning, match="k_electron"):
        legacy = ThermalLayerTwoTemp(**base)
    assert legacy.k_e() == 0.5                                     # legacy value kept
    with pytest.raises(ValueError, match="k_electron"):
        ThermalLayerTwoTemp(**base, k_electron=-1.0)
    with pytest.raises(ValueError, match="k_electron"):
        ThermalLayerTwoTemp(**base, k_electron=float("nan"))
    lay = ThermalLayerTwoTemp(**base, k_electron=0.5)              # explicit: no warning (filterwarnings=error)
    r2 = solve_thermal_twotemp_fem([lay], period_x_m=100e-9, period_y_m=100e-9, flux_W_m2=1e8,
                                   maxh_m=50e-9)
    r1 = solve_thermal_fem([ThermalLayer("m", 200e-9, 1.0)], period_x_m=100e-9, period_y_m=100e-9,
                           flux_W_m2=1e8, maxh_m=50e-9)
    rise1 = float(r1.mean_T_per_layer()[0] - 300.0)
    assert rise1 == pytest.approx(10.0, rel=1e-9)
    assert float(r2.mean_Te_per_layer()[0] - 300.0) == pytest.approx(rise1, rel=1e-5)
    assert float(r2.mean_Tl_per_layer()[0] - 300.0) == pytest.approx(rise1, rel=1e-5)


# ============================================ S1-13 ===============================================

_EPS_PARA, _EPS_PERP = 18.7, 4.0


def test_s1_13_onset_threshold_closed_forms():
    from dynameta.carriers.lc_director import _onset_threshold_V, cholesteric_q0
    from dynameta.constants import EPS0
    K11, K22, K33, dE = 20e-12, 7e-12, 2e-12, _EPS_PARA - _EPS_PERP
    h = 0.5 * math.pi
    # planar (field-axis theta_b = pi/2): the SPLAY constant, not (K11 + K33)/2
    assert _onset_threshold_V(K11, K33, dE, h, 1.0) == pytest.approx(
        math.pi * math.sqrt(K11 / (EPS0 * dE)), rel=1e-14)
    # homeotropic + negative anisotropy (VAN): the BEND constant
    assert _onset_threshold_V(K11, K33, -dE, 0.0, 1.0) == pytest.approx(
        math.pi * math.sqrt(K33 / (EPS0 * dE)), rel=1e-14)
    # the field STABILISES planar for dEps < 0 and homeotropic for dEps > 0: no onset
    assert _onset_threshold_V(K11, K33, -dE, h, 1.0) == math.inf
    assert _onset_threshold_V(K11, K33, dE, 0.0, 1.0) == math.inf
    # twisted nematic (Leenhouts-Schadt): pi sqrt([K11 + (K33-2K22)(Phi/pi)^2 + 4 K22 (d/p)(Phi/pi)]/(eps0 dEps))
    d, p, phi = 4e-6, 16e-6, 0.5 * math.pi
    tn = math.pi * math.sqrt((K11 + (K33 - 2 * K22) * (phi / math.pi) ** 2
                              + 4 * K22 * (d / p) * (phi / math.pi)) / (EPS0 * dE))
    assert _onset_threshold_V(K11, K33, dE, h, 1.0, K22=K22, twist_total_rad=phi,
                              q0_d=cholesteric_q0(p) * d) == pytest.approx(tn, rel=1e-13)
    # a series-capacitance divider scales the APPLIED onset by 1 / (V_lc / V_app)
    assert _onset_threshold_V(K11, K33, dE, h, 0.8) == pytest.approx(
        _onset_threshold_V(K11, K33, dE, h, 1.0) / 0.8, rel=1e-14)


@pytest.mark.parametrize("case", ["uniform", "poisson", "poisson_layers", "one_constant"])
def test_s1_13_onset_brackets_the_bvp_bifurcation(case):
    """The threshold the solver now seeds from is the bifurcation the BVP ITSELF exhibits, incl.
    the voltage division through fixed layers (field_model='poisson', t_in/t_out): with exact planar
    anchoring the midplane tilt is < 0.05 deg at 0.99 V_th and > 2 deg at 1.01 V_th (measured 0.001-
    0.010 deg / 4.7-33.7 deg over these four cells). The old (K11+K33)/2, layer-blind estimate is
    0.74x of it for the contrast cells."""
    from dynameta.carriers.lc_director import (_onset_threshold_V, compute_lc_geometry,
                                               director_profile_bvp, solve_lc_field_profile)
    K11, K33 = (11e-12, 11e-12) if case == "one_constant" else (20e-12, 2e-12)
    fm = "uniform" if case == "uniform" else "poisson"
    lay = (dict(t_in=0.5e-6, t_out=0.3e-6, eps_in=7.5, eps_out=3.0)
           if case in ("poisson_layers", "one_constant") else {})
    d, h = 4e-6, 0.5 * math.pi
    geo = compute_lc_geometry(geometry="planar", nz=161, d_planar=d, t_in=lay.get("t_in", 0.0),
                              t_out=lay.get("t_out", 0.0))
    v = solve_lc_field_profile(np.full_like(geo.z_m, h), 1.0, geo, eps_para=_EPS_PARA,
                               eps_perp=_EPS_PERP, field_model=fm, **lay)[1]
    vth = _onset_threshold_V(K11, K33, _EPS_PARA - _EPS_PERP, h, v)
    mids = []
    for f in (0.99, 1.01):
        r = director_profile_bvp(V_app=f * vth, K11=K11, K33=K33, eps_para=_EPS_PARA,
                                 eps_perp=_EPS_PERP, d_planar=d, field_model=fm, theta_b_rad=h,
                                 nz=161, rtol=1e-4, **lay)
        assert r.success, r.message
        mids.append(math.degrees(h - r.theta_field_rad[r.theta_field_rad.size // 2]))
    assert mids[0] < 0.05 and mids[1] > 2.0, mids


def test_s1_13_tn_onset_brackets_the_chiral_bvp():
    from dynameta.carriers.lc_director import (_onset_threshold_V, chiral_director_profile_bvp,
                                               cholesteric_q0)
    K11, K22, K33, d, h = 11e-12, 7e-12, 18e-12, 4e-6, 0.5 * math.pi
    for q0 in (0.0, cholesteric_q0(16e-6)):
        vth = _onset_threshold_V(K11, K33, _EPS_PARA - _EPS_PERP, h, 1.0, K22=K22,
                                 twist_total_rad=0.5 * math.pi, q0_d=q0 * d)
        mids = []
        for f in (0.97, 1.03):
            r = chiral_director_profile_bvp(V_app=f * vth, K11=K11, K22=K22, K33=K33,
                                            eps_para=_EPS_PARA, eps_perp=_EPS_PERP, d_planar=d,
                                            nz=161, theta_b_rad=h, phi_bottom_rad=0.0,
                                            phi_top_rad=0.5 * math.pi, q0_rad_m=q0, rtol=1e-4)
            assert r.success, r.message
            mids.append(math.degrees(h - r.theta_field_rad[r.theta_field_rad.size // 2]))
        assert mids[0] < 0.05 and mids[1] > 10.0, (q0, mids)


def test_s1_13_safety_net_no_longer_flags_correct_solutions():
    """The two false alarms the mean-K estimate raised. (a) K11 = 20 pN, K33 = 2 pN planar cell at
    1.20 V = 0.974 of the TRUE 1.232 V onset (1.31x the old 0.913 V estimate): the correct, untilted
    answer was flagged success=False (old mean tilt 0.23 deg, a ladder artefact; now 0.006 deg).
    (b) dEps < 0 with planar anchoring cannot tilt at any voltage, but got a finite 1.64 V 'threshold'
    and was flagged at 5 V."""
    from dynameta.carriers.lc_director import director_profile_bvp
    r = director_profile_bvp(V_app=1.20, K11=20e-12, K33=2e-12, eps_para=_EPS_PARA,
                             eps_perp=_EPS_PERP, d_planar=4e-6, field_model="uniform", nz=121,
                             theta_b_rad=0.5 * math.pi)
    assert r.success, r.message
    assert math.degrees(float(np.max(np.abs(r.theta_field_rad - 0.5 * math.pi)))) < 0.05
    r = director_profile_bvp(V_app=5.0, K11=11e-12, K33=18e-12, eps_para=4.0, eps_perp=10.0,
                             d_planar=4e-6, field_model="uniform", nz=121)
    assert r.success, r.message
    assert float(np.max(np.abs(r.theta_field_rad - r.theta_b_rad))) <= math.radians(0.1) + 1e-12
    # no regression above threshold: K11 = K33 planar cell well over the onset still tilts
    r = director_profile_bvp(V_app=2.5, K11=20e-12, K33=2e-12, eps_para=_EPS_PARA,
                             eps_perp=_EPS_PERP, d_planar=4e-6, field_model="uniform", nz=121)
    assert r.success and math.degrees(float(np.mean(r.theta_b_rad - r.theta_field_rad))) > 30.0


# ============================================ S1-14 ===============================================

def test_s1_14_twist_energy_is_the_complete_square():
    """S1-14: twist_energy_J_m2 dropped the (1/2) K22 q0^2 d of the Frank twist term, i.e. was
    measured from an UNTWISTED cell (negative at the natural pitch). Complete-square convention: a
    planar cell (theta = pi/2, no field) with total twist Phi stores (1/2) K22 (Phi/d - q0)^2 d --
    zero at the natural pitch, positive both sides of it (the legacy reading was this minus
    (1/2) K22 q0^2 d, e.g. -2.16e-6 J/m^2 at Phi = q0 d here)."""
    from dynameta.carriers.lc_director import chiral_director_profile_bvp, cholesteric_q0
    K = dict(K11=11e-12, K22=7e-12, K33=18e-12, eps_para=_EPS_PARA, eps_perp=_EPS_PERP,
             d_planar=4e-6, field_model="uniform", nz=121, theta_b_rad=0.5 * math.pi)
    d, q0 = K["d_planar"], cholesteric_q0(16e-6)
    for phi in (0.0, 0.5 * q0 * d, q0 * d, 2.0 * q0 * d):
        r = chiral_director_profile_bvp(V_app=0.0, phi_bottom_rad=0.0, phi_top_rad=phi, q0_rad_m=q0, **K)
        want = 0.5 * K["K22"] * (phi / d - q0) ** 2 * d
        scale = 0.5 * K["K22"] * q0 * q0 * d
        assert abs(r.twist_energy_J_m2 - want) < 1e-6 * scale, (phi, r.twist_energy_J_m2, want)
        assert r.twist_energy_J_m2 >= -1e-6 * scale
    # q0 = 0 is unchanged (the constant vanishes)
    r = chiral_director_profile_bvp(V_app=0.0, phi_bottom_rad=0.0, phi_top_rad=0.5 * math.pi,
                                    q0_rad_m=0.0, **K)
    assert r.twist_energy_J_m2 == pytest.approx(0.5 * K["K22"] * (0.5 * math.pi) ** 2 / d, rel=1e-6)


# ============================================ S1-18 ===============================================

def test_s1_18_fermi_integrals_match_40_digit_polylog_on_every_branch():
    """S1-18: _fd1 = F_1 and _fermi_log = F_0 vs 40-digit mpmath over x in [-700, 1e4], every
    branch and both branch edges (x = -1, 40). The gate found a defect, not just a gap: the spence
    branch ran to x = -700 and its argument 1 + e^x cancels -- 1.0e-3 relative error at x = -30 and
    F_1 == 0 exactly below x = -36.7. Now <= 1.3e-15 everywhere (series below x = -1)."""
    mp = pytest.importorskip("mpmath")
    from dynameta.carriers.schrodinger_poisson import _fd1, _fermi_log
    mp.mp.dps = 40
    xs = np.concatenate([np.linspace(-700.0, -1.0001, 120), [-1.0, -0.9999999],
                         np.linspace(-0.99, 39.99, 160), [40.0, 40.0000001],
                         np.linspace(40.01, 1.0e4, 40)])
    f1 = _fd1(xs)
    f0 = _fermi_log(xs)
    for x, a, b in zip(xs, f1, f0):
        xm = mp.mpf(float(x))
        ref1 = (-mp.polylog(2, -mp.e ** xm) if x < 200.0
                else xm ** 2 / 2 + mp.pi ** 2 / 6 + mp.polylog(2, -mp.e ** (-xm)))
        ref0 = mp.log1p(mp.e ** xm)
        assert abs(a - ref1) <= 4e-15 * ref1, (x, a, float(ref1))
        assert abs(b - ref0) <= 4e-15 * ref0, (x, b, float(ref0))
    # shape preserved on the 2-D (n_int, n_states) arrays the Kane sheet density feeds it
    grid = np.array([[-50.0, 0.3], [45.0, -2.0]])
    assert _fd1(grid).shape == (2, 2) and _fermi_log(grid).shape == (2, 2)


def test_s1_18_dF1_dx_equals_F0_across_the_branch_edges():
    """The identity the Kane density relies on, in integrated form (exact, no FD step):
    F_1(b) - F_1(a) = int_a^b F_0(x) dx, on intervals straddling each branch edge, by adaptive
    quadrature of the INDEPENDENT log1p closed form."""
    from scipy.integrate import quad

    from dynameta.carriers.schrodinger_poisson import _fd1
    f0 = lambda x: math.log1p(math.exp(x)) if x < 30 else x + math.log1p(math.exp(-x))   # noqa: E731
    for a, b in ((-60.0, -20.0), (-3.0, 2.0), (35.0, 45.0), (38.0, 300.0), (-2.0, 60.0)):
        lhs = float(_fd1(np.array([b]))[0] - _fd1(np.array([a]))[0])
        rhs, err = quad(f0, a, b, epsabs=0.0, epsrel=1e-13, limit=200)
        assert abs(lhs - rhs) <= 1e-12 * abs(rhs) + 2.0 * err, (a, b, lhs, rhs)


# ============================================ S1-9 ================================================

def _spec(**kw):
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    from dynameta.constants import M_E
    base = dict(semi_material="ITO", oxide_material="HfO2", lateral_m=4e-9, semi_thk_m=12e-9,
                oxide_thk_m=4e-9, n_bg_m3=4e26, eps_semi=9.5, eps_oxide=18.0, dos_mass_kg=0.35 * M_E,
                grid_n=(3, 3, 9), mesh_min_nm=1.0, mesh_max_nm=3.0)
    base.update(kw)
    return Stacked3DSpec(**base)


def test_s1_9_band_alignment_is_the_work_function_difference():
    """S1-9: each region's Phi_c0 was calibrated to make THAT region neutral at psi = 0 -- equal work
    functions, so psi = 0 was the exact equilibrium of every heterostack (no built-in potential even
    for an n+/n homojunction). band_alignment() (Anderson rule) must give: (a) each region neutral at
    its psi_neutral under its OWN closure n = N_c F_1/2((psi - Phi_c)/V_t); (b) the Boltzmann limit
    psi_bi = V_t ln(n_2/n_1) for a same-material nondegenerate junction; (c) an affinity step shifts
    psi_bi by exactly -(chi_2 - chi_1); (d) the primary keeps its legacy (Phi_c0, 0)."""
    from dynameta.carriers.physics_equilibrium import (V_T, F12_aymerich_humet, effective_dos_m3,
                                                       invert_F12)
    from dynameta.constants import M_E
    m1, m2 = 0.35 * M_E, 0.25 * M_E
    # (a) + (d): degenerate n+/n ITO, then a heterojunction with its own mass and affinity
    for extra, chi_semi in (([("xs", "ITO", 12e-9, 9.5, 4e25)], None),
                            ([("xs", "AZO", 12e-9, 9.0, 1e26, 4.62, m2)], 4.50)):
        s = _spec(extra_semiconductors=extra, chi_semi_eV=chi_semi)
        ba = s.band_alignment()
        assert ba["semi"] == (pytest.approx(-invert_F12(4e26 / effective_dos_m3(m1)) * V_T, rel=1e-15),
                              0.0)
        for (nm, *_r) in s.extra_semiconductor_entries():
            phi_c, psi_n = ba[nm]
            nc = effective_dos_m3(s.semiconductor_dos_mass_kg(nm))
            assert nc * F12_aymerich_humet((psi_n - phi_c) / V_T) == pytest.approx(
                s.semiconductor_nbg(nm), rel=1e-9)
    # (b) Boltzmann limit (eta ~ -11): psi_bi -> V_t ln(n2/n1)
    s = _spec(n_bg_m3=1e22, extra_semiconductors=[("xs", "ITO", 12e-9, 9.5, 1e21)])
    assert s.band_alignment()["xs"][1] == pytest.approx(V_T * math.log(1e21 / 1e22), rel=1e-3)
    # (c) the affinity step enters additively and with the work-function sign
    base = _spec(extra_semiconductors=[("xs", "AZO", 12e-9, 9.5, 4e25, 4.5, m1)], chi_semi_eV=4.5)
    step = _spec(extra_semiconductors=[("xs", "AZO", 12e-9, 9.5, 4e25, 4.6, m1)], chi_semi_eV=4.5)
    d = step.band_alignment()["xs"][1] - base.band_alignment()["xs"][1]
    assert d == pytest.approx(-0.1, abs=1e-12)
    # same material: identical to the explicit equal-affinity form
    same = _spec(extra_semiconductors=[("xs", "ITO", 12e-9, 9.5, 4e25)])
    assert same.band_alignment()["xs"][1] == pytest.approx(base.band_alignment()["xs"][1], abs=1e-15)
    assert same.band_alignment()["xs"][1] < -0.3                # the n side sits ~0.45 V below the n+ side


def test_s1_9_heterojunction_without_affinities_is_refused():
    with pytest.raises(ValueError, match="electron affinities"):
        _spec(extra_semiconductors=[("xs", "AZO", 12e-9, 9.5, 4e25)])
    with pytest.raises(ValueError, match="electron affinities"):
        _spec(extra_semiconductors=[("xs", "AZO", 12e-9, 9.5, 4e25, 4.6)])      # chi_semi missing
    with pytest.raises(ValueError):
        _spec(extra_semiconductors=[("xs", "ITO", 12e-9, 9.5)])                  # short tuple
    with pytest.raises(ValueError, match="dos_mass_kg"):
        _spec(extra_semiconductors=[("xs", "ITO", 12e-9, 9.5, 4e25, None, 0.0)])
    _spec(extra_semiconductors=[("xs", "ITO", 12e-9, 9.5, 4e25)])               # legacy 5-tuple ok


def test_s1_9_build_device_installs_the_aligned_bands(monkeypatch, tmp_path):
    """build_device must hand DEVSIM the aligned Phi_c for every extra region (and seed its Potential
    at the neutral level) while leaving the primary's calibration alone. The FD-closure setup of a
    real region costs ~6 min of DEVSIM symbolic differentiation (measured 375 s per region on DEVSIM
    2.10), so the region physics is stubbed with exactly the parameters/solution it creates; the
    3-D SOLVE of the junction is validation/carriers_3d_multisemi.py GATE C."""
    ds = pytest.importorskip("devsim")
    pytest.importorskip("gmsh")
    from dynameta.carriers import devsim_3d as D3
    from dynameta.carriers import eq_registry as _R

    def _stub(device, region, *, n_bg_m3, eps_static, dos_mass_kg):
        D3.PE.setup_dielectric_region(device, region, eps_static)        # Potential + Poisson
        D3.PE.setup_phi_c0(device, region, n_bg_m3, dos_mass_kg)          # the legacy calibration

    monkeypatch.setattr(D3.PE, "setup_semiconductor_region", _stub)
    s = _spec(extra_semiconductors=[("xs", "ITO", 6e-9, 9.5, 4e25)], semi_thk_m=6e-9)
    solver = D3.Devsim3DEquilibrium(s, device_name="s19d", mesh_name="s19m",
                                    msh_path=str(tmp_path / "s19.msh"))
    try:
        solver.build_device()
        ba = s.band_alignment()
        assert ds.get_parameter(device="s19d", region="semi", name="Phi_c0") == pytest.approx(
            ba["semi"][0], rel=1e-14)
        assert ds.get_parameter(device="s19d", region="xs", name="Phi_c0") == pytest.approx(
            ba["xs"][0], rel=1e-14)
        pot = np.asarray(ds.get_node_model_values(device="s19d", region="xs", name="Potential"))
        assert np.all(pot == ba["xs"][1]) and ba["xs"][1] < -0.3
    finally:
        _R.clear("s19d")
        solver.teardown()
