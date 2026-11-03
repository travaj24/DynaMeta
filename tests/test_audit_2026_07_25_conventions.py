"""Gates for the 2026-07-25 v0.9.0 audit's P3 conventions / dead-code batch handled in wave 6 (H6a,
2026-11-02): V-10, V-11, V-13, V-16, V-17, V-21, V-25, X-12, X-14, X-23, X-24.

Most of these are SOURCE-CENSUS gates (they walk dynameta/ with `ast`) because the defect they
close is structural -- a warning without a category, a public definition missing from `__all__`,
a duplicated body, a literal that should be a shared constant -- and a census is the only gate that
fails on the NEXT instance, not just the one that was fixed. Every census carries an explicit
allow-list with its reason; an entry that is no longer needed is reported as stale, so the lists
cannot rot. The behavioural gates (a guard that now raises, a renamed alias that still resolves, a
dead public method that now has a reader) each fail on the pre-wave-6 tree.
"""
import ast
import importlib
import math
import pathlib
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

import dynameta

PKG = pathlib.Path(dynameta.__file__).resolve().parent


def _modules():
    for p in sorted(PKG.rglob("*.py")):
        yield p.relative_to(PKG).as_posix(), p, ast.parse(p.read_text(encoding="utf-8"))


def _warn_calls(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "warnings.warn":
            yield n


# ============================================ V-17 ================================================

# Sites in files another workstream owned during wave 6 (fixed there or in a follow-up). Matched by
# (file, message prefix) so a line shift does not break the gate. EMPTY since 2026-11-02: the one
# entry (eryb.py's C_up-override note) was fixed by the eryb_fix workstream (AdvisoryWarning).
_V17_PENDING = set()


def test_v17_category_hierarchy_keeps_every_old_category_as_a_base():
    from dynameta.warning_types import (AdvisoryWarning, DynaMetaDeprecationWarning,
                                        DynaMetaWarning, UnreliableResultWarning)
    assert issubclass(DynaMetaWarning, UserWarning)
    # the sites that adopted UnreliableResultWarning were UserWarning OR RuntimeWarning before:
    # an existing pytest.warns / filter on either must keep matching
    assert issubclass(UnreliableResultWarning, UserWarning)
    assert issubclass(UnreliableResultWarning, RuntimeWarning)
    assert issubclass(AdvisoryWarning, UserWarning) and not issubclass(AdvisoryWarning, RuntimeWarning)
    assert issubclass(DynaMetaDeprecationWarning, DeprecationWarning)
    from dynameta.optics.soa.metrics import TransferFitWarning
    assert issubclass(TransferFitWarning, UnreliableResultWarning)
    assert issubclass(TransferFitWarning, RuntimeWarning)                  # A-23's contract
    import dynameta.core as core
    for name in ("DynaMetaWarning", "UnreliableResultWarning", "AdvisoryWarning",
                 "DynaMetaDeprecationWarning"):
        assert getattr(core, name) is getattr(importlib.import_module("dynameta.warning_types"), name)


def test_v17_fem_category_is_rebased_without_breaking_its_filter():
    pytest.importorskip("ngsolve")
    from dynameta.optics.solver import FEMDiagnosticWarning
    from dynameta.warning_types import DynaMetaWarning
    assert issubclass(FEMDiagnosticWarning, DynaMetaWarning)
    assert issubclass(FEMDiagnosticWarning, UserWarning)


def test_v17_every_warn_site_names_a_category_and_a_stacklevel():
    """Before wave 6: 55 of 103 sites relied on the implicit UserWarning and 12 had no stacklevel
    (so the warning pointed INSIDE dynameta, at the warn line, instead of at the caller)."""
    bad, used = [], set()
    for rel, _p, tree in _modules():
        for n in _warn_calls(tree):
            has_cat = len(n.args) >= 2 or any(k.arg == "category" for k in n.keywords)
            has_sl = len(n.args) >= 3 or any(k.arg == "stacklevel" for k in n.keywords)
            msg = ast.unparse(n.args[0]) if n.args else ""
            pend = any(rel == f and msg.startswith(m) for f, m in _V17_PENDING)
            if pend:
                used.add(next((f, m) for f, m in _V17_PENDING if rel == f and msg.startswith(m)))
            if not (has_cat and has_sl) and not pend:
                bad.append("{}:{} (category={}, stacklevel={})".format(rel, n.lineno, has_cat, has_sl))
    assert not bad, "warnings.warn without an explicit category/stacklevel:\n" + "\n".join(bad)
    # an allow-listed site that was FIXED still matches (and is harmless); one that VANISHED is stale
    assert not (_V17_PENDING - used), "stale V-17 allow-list entries: {}".format(_V17_PENDING - used)


def test_v17_v10_intersubband_gain_clamp_is_flagged_as_an_unreliable_result():
    """The inverted-population clamp ALTERS the returned eps; before wave 6 it warned as a bare
    UserWarning (no stacklevel), indistinguishable from an advisory. V-10 row (2)."""
    from dynameta.carriers.schrodinger_poisson import SubbandResult
    from dynameta.constants import M_E
    from dynameta.core.effects import IntersubbandEffect
    from dynameta.warning_types import UnreliableResultWarning
    L = 2e-9
    z = np.linspace(0.0, L, 160)
    h = z[1] - z[0]
    p1, p2 = np.sin(np.pi * z / L), np.sin(2 * np.pi * z / L)
    p1, p2 = p1 / np.sqrt(np.sum(p1 ** 2) * h), p2 / np.sqrt(np.sum(p2 ** 2) * h)
    res = SubbandResult(energies_J=np.array([0.0, 0.95 * 1.602e-19]), psi=np.stack([p1, p2], 1),
                        z_m=z, sheet_density_m2=np.array([1e17, 5e17]))      # INVERTED
    m = IntersubbandEffect(4.25, 0.225 * M_E, 1.1e14, 1.0e13)
    with pytest.warns(UnreliableResultWarning, match="inverted population"):
        m.eps({"subband": res}, 1300e-9)


def test_v10_policy_is_written_down_once():
    import dynameta.core.eps_field as ef
    doc = ef.__doc__
    assert "ANTI-GAIN POLICY" in doc and "V-10" in doc
    for row in ("(1) RAISE", "(2) CLAMP", "(3) WARN, NO CLAMP", "(4) ACCEPT"):
        assert row in doc


# ============================================ V-11 ================================================

def test_v11_layer_aliases_are_renamed_and_the_old_spelling_is_deprecated():
    import dynameta.optics.nonlocal_tmm as nt
    import dynameta.optics.resonance as rs
    from dynameta.geometry.stack import Layer as DesignLayer
    from dynameta.warning_types import DynaMetaDeprecationWarning
    assert dynameta.Layer is DesignLayer
    assert "Layer" not in vars(rs) and "Layer" not in vars(nt)     # no module-level rebinding
    with pytest.warns(DynaMetaDeprecationWarning, match="LayerSpec"):
        assert rs.Layer is rs.LayerSpec
    with pytest.warns(DynaMetaDeprecationWarning, match="NonlocalLayer"):
        assert nt.Layer is nt.NonlocalLayer
    with pytest.raises(AttributeError):
        rs.NoSuchName  # noqa: B018


# ============================================ V-13 ================================================

def test_v13_plain_import_is_solver_free_and_the_star_caveat_is_documented():
    code = ("import sys, dynameta.carriers, dynameta.optics; "
            "print(sorted(m for m in ('devsim', 'ngsolve', 'netgen') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(PKG.parent), timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"
    import dynameta.carriers as c
    import dynameta.optics as o
    assert "STAR IMPORT IS EAGER" in c.__doc__ and "STAR IMPORT IS EAGER" in o.__doc__


# ============================================ V-16 ================================================

def test_v16_hydro_fem_refuses_metres_for_nanometres():
    hf = pytest.importorskip("dynameta.optics.hydro_fem")
    p = hf.HydroParams(eps_inf=1.0, wp=1.37e16, gamma=1.0e14, beta=1.1e6)
    w = hf.bulk_plasmon_omega(1, p, 5.0)
    assert w == math.sqrt(p.wp ** 2 / p.eps_inf + p.beta ** 2 * (math.pi / 5e-9) ** 2)
    with pytest.raises(ValueError, match="NANOMETRES"):
        hf.bulk_plasmon_omega(1, p, 5e-9)                # metres passed to the nm sibling
    with pytest.raises(ValueError, match="NANOMETRES"):
        hf.cylinder_blueshift_raza(p, 3e-9)
    with pytest.raises(ValueError):
        hf.bulk_plasmon_omega(1, p, 0.0)


# ============================================ V-21 ================================================

_V21_ALLOWED = {
    # the ENGINE seam dynamics.simulate_transient(_eryb) calls as march_ase.<name> (documented at
    # its __all__); deliberately not user API and not on the exhaustive fiber_amp facade
    "optics/fiber_amp/march_ase.py": {"check_ase_mode", "residual_views",
                                      "solve_self_consistent_step", "predicted_step_ase_error",
                                      "needs_ase_probe", "auto_switch"},
    # qd_gain.py was another workstream's file during wave 6: add both to its __all__ there
    "optics/soa/qd_gain.py": {"Leakage", "ManyBody"},
}


def test_v21_public_definitions_are_declared_in_all():
    missing = []
    for rel, _p, tree in _modules():
        allv = None
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__"
                                                 for t in n.targets):
                try:
                    allv = set(ast.literal_eval(n.value))
                except ValueError:
                    allv = None
        if allv is None:
            continue
        for n in tree.body:
            if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")
                    and n.name not in allv and n.name not in _V21_ALLOWED.get(rel, ())):
                missing.append("{}: {}".format(rel, n.name))
    assert not missing, "public definitions missing from __all__:\n" + "\n".join(missing)


def test_v21_return_types_are_reachable_from_the_facade():
    from dynameta.optics import fiber_amp
    from dynameta.optics.fiber_amp.calibration import CalibrationReport
    from dynameta.optics.fiber_amp.pulse import PropagationResult
    assert fiber_amp.CalibrationReport is CalibrationReport
    assert fiber_amp.PropagationResult is PropagationResult


# ============================================ V-25 ================================================

def test_v25_differentiable_entry_points_refuse_a_degree_angle():
    pytest.importorskip("lumenairy")
    from dynameta.optics.lumenairy_bridge.berreman_design import berreman_RT
    from dynameta.optics.lumenairy_bridge.rcwa_design import pmm_stack_RT, rcwa_stack_RT
    layers = [(2.25 + 0j, 100e-9)]
    kw = dict(n_substrate=1.45, n_superstrate=1.0)
    R, T = berreman_RT(layers, 1.31e-6, angle=0.3, **kw)          # radians: runs
    assert 0.0 < float(R) < 1.0
    with pytest.raises(ValueError, match="RADIANS"):
        berreman_RT(layers, 1.31e-6, angle=30.0, **kw)            # the incidence_angle_deg habit
    with pytest.raises(ValueError, match="RADIANS"):
        berreman_RT(layers, 1.31e-6, angle=0.3, phi=45.0, **kw)
    with pytest.raises(ValueError, match="RADIANS"):
        rcwa_stack_RT(layers, 1.31e-6, period_x=0.8e-6, theta=30.0, **kw)
    with pytest.raises(ValueError, match="RADIANS"):
        pmm_stack_RT([([(0.5, 4.0 + 0j), (0.5, 1.0 + 0j)], 100e-9)], 1.31e-6, period=0.8e-6,
                     angle=-45.0, **kw)


def test_v25_guard_passes_values_it_cannot_read():
    from dynameta.optics.lumenairy_bridge._common import require_angles_rad
    require_angles_rad("t", polar=(("angle", 0.3 + 0.01j),))       # complex: not checkable
    require_angles_rad("t", polar=(("angle", math.pi / 2),), azimuth=(("phi", -2 * math.pi),))

    class _Tracer:                                                 # float() refuses, like a tracer
        def __float__(self):
            raise TypeError("abstract")
    require_angles_rad("t", polar=(("angle", _Tracer()),))


# ======================================== X-12 / X-14 =============================================

def test_x12_dead_private_helpers_are_gone():
    src = {rel: p.read_text(encoding="utf-8") for rel, p, _t in _modules()}
    for rel, name in (("carriers/devsim_layered.py", "def _meshed("),
                      ("optics/fiber_amp/steady_state.py", "def _nbar2("),
                      ("optics/fiber_amp/steady_state.py", "def _dP_full("),
                      ("optics/fiber_amp/transverse.py", "def _fundamental_psi2(")):
        assert name not in src[rel], (rel, name)


def test_x12_cached_optical_solver_builds_the_cache_it_names(tmp_path):
    from dynameta.cache import OpticalSolverCache, cached_optical_solver

    def inner(design, geo, eps_by_region, lambda_m, n_super, n_sub):   # pragma: no cover
        raise AssertionError("not called")
    a = cached_optical_solver(inner, str(tmp_path / "c.h5"), tag="x12", autosave=False)
    b = OpticalSolverCache(inner, str(tmp_path / "c.h5"), tag="x12", autosave=False)
    assert isinstance(a, OpticalSolverCache)
    assert (a.inner, a.path, a.tag, a._solver_id) == (b.inner, b.path, b.tag, b._solver_id)


def test_x12_gate_tuned_reflection_closes_and_tracks_the_fermi_level():
    from dynameta.constants import Q_E
    from dynameta.core.graphene import gate_tuned_reflection, graphene_sigma, sheet_rt
    lam = 1.55e-6
    R0, T0, A0 = gate_tuned_reflection(0.05 * Q_E, lam, 1.0, 1.45)
    R1, T1, A1 = gate_tuned_reflection(0.60 * Q_E, lam, 1.0, 1.45)
    for R, T, A in ((R0, T0, A0), (R1, T1, A1)):
        assert abs(R + T + A - 1.0) < 1e-12 and A >= 0.0
    # Pauli blocking: E_F > hbar w / 2 (0.40 eV at 1550 nm) switches the interband loss OFF
    assert A1 < 0.2 * A0
    _, _, R, T, A = sheet_rt(1.0, 1.45, graphene_sigma(0.05 * Q_E, lam))
    assert (R, T, A) == (R0, T0, A0)
    # bare-interface limit: sigma -> 0 recovers Fresnel
    r_fresnel = ((1.0 - 1.45) / (1.0 + 1.45)) ** 2
    assert abs(sheet_rt(1.0, 1.45, 0.0)[2] - r_fresnel) < 1e-15


def test_x12_x14_unit_scale_conversion_api_is_live_and_exact():
    from dynameta.core.units import NM, NM_PER_M, SI
    x = np.array([0.0, 1.5e-7, 3.7e-7, 1.0e-6])
    assert np.array_equal(NM.to_units(x), x / 1e-9)               # the bridge's exact operation
    assert np.array_equal(SI.to_units(x), x)
    assert np.allclose(NM.to_metres(NM.to_units(x)), x, rtol=1e-15, atol=0.0)
    assert NM.to_metres(370.0) == 370.0 * 1e-9
    # the documented reason NM_PER_M stays a literal: the two spellings differ in the last bit
    assert 1.0 / NM.metres_per_unit != NM_PER_M
    src = (PKG / "core" / "bridge.py").read_text(encoding="utf-8")
    assert "x_remap_m / mpp" not in src and "unit_scale.to_units" in src


def test_x12_semiconductor_layers_selects_by_role():
    from dynameta import (ConstantOptical, Design, DrudeOptical, Layer, Material,
                          MaterialRegistry, Stack, UnitCell)
    from dynameta.materials import M_E
    from dynameta.materials.transport_model import TransportModel
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("HfO2", ConstantOptical(4.0 + 0j), eps_static_dc=18.0))
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14),
                     transport=TransportModel(n_bg_m3=4e26, eps_static=9.0,
                                              dos_mass_kg_of_n_m3=lambda n: 0.35 * M_E)))
    stack = Stack(layers=[Layer("ox", 10e-9, "HfO2"), Layer("ito", 20e-9, "ITO"),
                          Layer("gap", 30e-9, "air")],
                  superstrate_material="air", substrate_material="air")
    d = Design(name="x12", unit_cell=UnitCell.square(370e-9), stack=stack, electrodes=[],
               materials=reg)
    assert [L.name for L in d.semiconductor_layers()] == ["ito"]


def test_x12_bti_time_to_inverts_the_drift_law():
    from dynameta.reliability.bti import BtiParams
    b = BtiParams(A_V=2e-3)
    for dv in (1e-3, 2e-2):
        t = b.time_to_s(dv, 5e8, 358.0)
        assert abs(float(b.dvth_V(t, 5e8, 358.0)) - dv) < 1e-12 * dv


def test_x12_ase_spectrum_total_power_is_the_binned_sum():
    from dynameta.optics.fiber_amp.noise import AseSpectrum
    p = np.array([1e-6, 2e-6, 4e-6])
    s = AseSpectrum(lambda_m=np.array([1.53e-6, 1.54e-6, 1.55e-6]), power_W=p,
                    psd_1pol=np.zeros(3), n_sp=np.full(3, np.nan), direction="fwd")
    assert s.total_power_W == 7e-6


def test_x12_lc_reduce_director_slices_and_collapses():
    from dynameta.carriers.lc_director import director_to_extra_fields, reduce_director
    th = np.linspace(0.1, 0.9, 11)[:, None] * np.array([[0.5, 1.0]])          # (nz, nt)
    prof = reduce_director(th)                                              # final column
    assert np.array_equal(prof["director_angle_rad"],
                          director_to_extra_fields(th[:, -1])["director_angle_rad"])
    mid = reduce_director(th, t_index=0, reduce="midplane")["director_angle_rad"]
    assert np.array_equal(mid, director_to_extra_fields(np.asarray(th[5, 0]))["director_angle_rad"])
    mean = reduce_director(th, reduce="mean")["director_angle_rad"]
    assert np.array_equal(mean, director_to_extra_fields(np.asarray(np.mean(th[:, -1])))
                          ["director_angle_rad"])
    with pytest.raises(ValueError):
        reduce_director(th, reduce="median")


def test_x12_qd_self_heating_time_constant_and_eh_conservation_diagnostic():
    """QDGainModel.total_electron_density / total_hole_density were a written-but-never-run
    conservation diagnostic for the e/h ladder (audit X-12, relevant to A-15). Both are LINEAR in
    the state, so d(n_tot)/dt = n_tot(rhs(y)): with injection, recombination, spontaneous and
    stimulated emission off, every internal transition (capture, escape, relaxation, back-
    transfer) must cancel -- measured 0 / 8e-17 of the gross flux; with current on it must equal
    the injection rate I/(q V) exactly (measured 7e-15 relative)."""
    import dataclasses as dc

    from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams, SelfHeating
    sh = SelfHeating(Rth_K_W=250.0, Cth_J_K=4e-9)
    assert sh.tau_th_s == 250.0 * 4e-9
    p = dc.replace(QDGainParams(n_groups=5, eh_split=True), tau_sp_s=1e300, B_wl_m3_s=0.0,
                   C_wl_m6_s=0.0)
    m = QDGainModel(p)
    y = np.concatenate([[3e23, 2e23], np.random.default_rng(0).uniform(0.1, 0.9, 4 * m.ng)])
    d = m.rhs(y, 0.0, 0.0, 2.3e14)
    for tot, w_sel, y0 in ((m.total_electron_density, (m.f_c_ES, m.f_c_GS), d[0]),
                           (m.total_hole_density, (m.f_v_ES, m.f_v_GS), d[1])):
        gross = abs(y0) + p.N_q_m3 * np.sum(m.w_j * (p.mu_ES * np.abs(w_sel[0](d))
                                                     + p.mu_GS * np.abs(w_sel[1](d))))
        assert abs(tot(d)) <= 1e-13 * gross
    I_A = 0.05
    d_on = m.rhs(y, I_A, 0.0, 2.3e14)
    inj = I_A / m._qVa
    assert abs(m.total_electron_density(d_on) - inj) <= 1e-12 * inj
    assert abs(m.total_hole_density(d_on) - inj) <= 1e-12 * inj


def test_x12_resolved_eryb_density_and_transfer_accessors():
    from dynameta.optics.fiber_amp import FiberSpec, Pump, Signal, erbium, ytterbium
    from dynameta.optics.fiber_amp.transverse_eryb import ResolvedErYbAmplifier
    a = 3.0e-6
    fib = FiberSpec(core_radius_m=a, na=0.20, n_t_m3=1e25, length_m=1.0, clad_radius_m=62.5e-6)
    amp = ResolvedErYbAmplifier(erbium("aluminosilicate"), ytterbium("phosphosilicate"), fib,
                                [Pump(1.0, 0.976e-6, "fwd", cladding=True)],
                                [Signal(1e-3, 1.550e-6)], None, n_yb_m3=8e25, k_tr_m3_s=2e-22)
    r = amp.grid.r_m
    yb, er = amp.yb_density_r_m3, amp.er_density_r_m3
    assert np.array_equal(yb, np.where(r < a, 8e25, 0.0))         # uniform core doping
    K = amp.channel_plan().lambda_m.size
    P = np.full(K, 1e-3)
    P[0] = 1.0
    f2, b2c, _bn, _bb = amp.local_populations(P)
    rate = amp.local_transfer_rate_per_m3(P)
    ref = 2e-22 * yb * b2c * er * (1.0 - f2)                      # k_tr n_Yb2 n_Er1 (all coupled)
    assert np.max(np.abs(rate - ref)) <= 1e-14 * np.max(np.abs(ref))
    assert np.all(rate[r >= a] == 0.0) and np.max(rate) > 0.0


def test_x12_two_temperature_layer_means_match_the_point_profile():
    pytest.importorskip("ngsolve")
    from dynameta.carriers.thermal_fem import ThermalLayerTwoTemp, solve_thermal_twotemp_fem
    from dynameta.core.numerics import trapz
    L = ThermalLayerTwoTemp(name="m", thickness_m=1e-6, k_thermal=1.0, rho_kg_m3=2.0e3,
                            Cp_J_kgK=1.0e3, G_e_l=1.0e12, C_e_J_m3K=1.0e4, k_electron=10.0)
    r = solve_thermal_twotemp_fem([L], period_x_m=100e-9, period_y_m=100e-9,
                                  source_e_W_m3=1.0e15, T_sink_K=300.0, maxh_m=100e-9)
    z = np.linspace(0.0, 1e-6, 401)
    for mean_fn, at in ((r.mean_Te_per_layer, r.Te_at), (r.mean_Tl_per_layer, r.Tl_at)):
        prof = np.array([at(50e-9, 50e-9, zz) for zz in z])
        assert abs(float(mean_fn()[0]) - trapz(prof, z) / 1e-6) < 1e-4 * (prof.max() - 300.0)
    assert r.mean_Te_per_layer()[0] > r.mean_Tl_per_layer()[0] > 300.0


# ============================================ X-23 ================================================

def test_x23_duplicate_bodies_have_one_home():
    src = {rel: p.read_text(encoding="utf-8") for rel, p, _t in _modules()}
    cs = src["geometry/cross_section.py"]
    assert cs.count("def center_m(self):\n        return (self.cx_m, self.cy_m)") == 1
    assert cs.count("return _polygon_contains(self.vertices_m(), x, y)") == 1
    shg = src["optics/shg_fem.py"]
    assert shg.count("_region_epsfield(design, geo, rg, lam_m)") == 1
    hm = src["optics/harmonics.py"]
    assert hm.count("truncated = bool(hi > f[-1] + 0.5 * df)") == 1
    assert hm.count("fbar = float((fw * sw).sum()) / wtot") == 1
    assert src["carriers/physics_equilibrium.py"].count("class _DevsimShim") == 0
    import dynameta.carriers.eq_registry as er
    import dynameta.carriers.physics_equilibrium as pe
    assert pe.ds is er.ds


def test_x23_shape_methods_unchanged():
    from dynameta.geometry.cross_section import (Circle, Ellipse, Polygon, Rectangle,
                                                 RegularPolygon)
    x, y = np.array([1.0, 0.2, 9.0]), np.array([2.0, 0.2, 9.0])
    assert Rectangle(1, 2, 3, 4).center_m() == (1, 2)
    assert Circle(1, 2, 3).center_m() == (1, 2) and Ellipse(1, 2, 3, 4).center_m() == (1, 2)
    hexa = RegularPolygon(0.0, 0.0, 1.0, 6)
    assert hexa.center_m() == (0.0, 0.0)
    assert np.allclose(hexa.bbox_m(), (-1.0, 1.0, -math.sqrt(3) / 2, math.sqrt(3) / 2),
                       rtol=0, atol=1e-15)
    tri = Polygon([(0, 0), (1, 0), (0, 1)])
    assert tri.bbox_m() == (0.0, 1.0, 0.0, 1.0)
    assert tri.contains_m(x, y).tolist() == [False, True, False]
    assert repr(tri) == "Polygon(points_m=[(0, 0), (1, 0), (0, 1)], kind='polygon')"


# ============================================ X-24 ================================================

_X24_TNAME = re.compile(r"^T(_K|_ref|_ref_K|0_K|_sink_K|_coolant_K)$")
_X24_PENDING = {("optics/soa/qd_gain.py", "T_K"),
                ("optics/soa/qd_gain.py", "T0_K")}          # another workstream's files in wave 6


def test_x24_temperature_defaults_use_the_one_reference_constant():
    from dynameta.constants import T_REF
    bad = []
    for rel, _p, tree in _modules():
        for n in ast.walk(tree):
            pairs = []
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                pos = a.posonlyargs + a.args
                pairs = [(k.arg, d) for k, d in zip(pos[len(pos) - len(a.defaults):], a.defaults)]
                pairs += [(k.arg, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value:
                pairs = [(n.target.id, n.value)]
            for name, d in pairs:
                if (_X24_TNAME.match(name) and isinstance(d, ast.Constant) and d.value == T_REF
                        and (rel, name) not in _X24_PENDING):
                    bad.append("{}:{} {}".format(rel, d.lineno, name))
    assert not bad, "literal 300.0 temperature defaults (use constants.T_REF):\n" + "\n".join(bad)
    # bit-identity: the constant IS the literal it replaced
    assert T_REF == 300.0
    from dynameta.core.graphene import graphene_sigma
    import inspect
    assert inspect.signature(graphene_sigma).parameters["T_K"].default == 300.0


def test_warning_category_adoption_does_not_change_default_visibility():
    """UnreliableResultWarning is shown by the default filters (it is NOT a DeprecationWarning)."""
    from dynameta.warning_types import UnreliableResultWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.resetwarnings()
        warnings.simplefilter("default")
        warnings.warn("probe", UnreliableResultWarning, stacklevel=1)
    assert len(rec) == 1
