"""Solver-free gates for optics/fem_2d_tm_hydro.py: the conductor spec, its planar-kernel coefficients,
the conductor mesh plan it hands build_tm_cell, the validity report and the convergence tabulation.

The weak form itself is assembled by NGSolve (solver leg, tests/test_fem_2d_tm_hydro.py). What runs on
numpy is checked here against closed forms written in this file:

  * wp^2 = n e^2 / (eps0 m), beta = sqrt(3/5) v_F ('high_freq') or sqrt(1/3) v_F ('thomas_fermi'),
    v_F = hbar (3 pi^2 n)^(1/3) / m;
  * eps_T = eps_inf - wp^2 / (omega^2 + i omega gamma);
  * k_L^2 = (omega^2 + i omega gamma - wp^2 / eps_inf) / (beta^2 + D (gamma - i omega));
  * the planar route (nonlocal_tmm.hydro_layers_from_profile) on the SAME arguments, which the 2-D
    spec promises to reproduce (hydro_conductor_from_profile is its twin);
  * observed convergence order log(d1/d2)/log(h1/h2) of a synthetic r = r_inf + C h^p.

Tolerances: closed forms evaluated in double precision agree at round-off; 1e-12 relative is used
throughout (the CODATA constants enter both sides identically).
"""
import cmath
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, EPS0, HBAR, M_E, Q_E
from dynameta.core.numerics import trapz
from dynameta.optics import fem_2d_tm as F
from dynameta.optics import nonlocal_tmm as nt
from dynameta.optics.fem_2d_tm import GriddedEps2D, TMLayer, _hydro_region_eps, _RegionEps
from dynameta.optics.fem_2d_tm_hydro import (HydroConductor2D, _dedupe, hydro_conductor_from_profile,
                                             hydro_extensions_for_cell, hydro_z_convergence)
from dynameta.warning_types import UnreliableResultWarning

LAM = 1200e-9
OMEGA = 2.0 * math.pi * C_LIGHT / LAM
MEFF = 0.30 * M_E
KW = dict(m_eff_kg=MEFF, eps_inf=4.0, gamma=1.2e14)
T_NM = 8.0                                             # film thickness [nm]
ZP = np.linspace(0.0, T_NM * 1e-9, 11)                 # from the TOP face (planar convention)
NP = 3e26 + 1.2e27 * np.exp(-ZP / 1e-9)


def _cond(**kw):
    opts = dict(KW)
    opts.update(check_validity=False)
    opts.update(kw)
    return hydro_conductor_from_profile(ZP, NP, **opts)


def _wp2(n):
    return np.asarray(n, float) * Q_E ** 2 / (EPS0 * MEFF)


def _beta(n, conv="high_freq"):
    vf = HBAR * np.cbrt(3.0 * math.pi ** 2 * np.asarray(n, float)) / MEFF
    return math.sqrt(3.0 / 5.0 if conv == "high_freq" else 1.0 / 3.0) * vf


def _kL(n, D=0.0, scale=1.0, eps_inf=4.0, gamma=1.2e14):
    w = OMEGA
    b2 = (scale * _beta(n)) ** 2 + D * (gamma - 1j * w)
    return np.sqrt((w * w + 1j * w * gamma - _wp2(n) / eps_inf) / b2 + 0j)


# ------------------------------------------------------------------------------------------------
# the spec
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw, exc, msg", [
    (dict(n0_m3=np.ones(3) * 1e26, z_m=[0.0, 1e-9]), ValueError, "Nz = len"),
    (dict(n0_m3=[1e26, np.nan], z_m=[0.0, 1e-9]), ValueError, "finite"),
    (dict(n0_m3=[1e26, 1e26], z_m=[1e-9, 0.0]), ValueError, "strictly increasing"),
    (dict(n0_m3=[1e26, -1.0], z_m=[0.0, 1e-9]), ValueError, "> 0 everywhere"),
    (dict(n0_m3=np.ones((2, 3)) * 1e26, z_m=[0.0, 1e-9]), ValueError, "needs x_m"),
    (dict(n0_m3=np.ones((2, 3)) * 1e26, z_m=[0.0, 1e-9], x_m=[0.0, 2.0, 1.0]), ValueError,
     "x_m must be strictly increasing"),
    (dict(convention="lindhard"), ValueError, "convention"),
    (dict(eps_inf=4.0 + 0.1j), TypeError, "eps_inf must be real"),
    (dict(m_eff_kg=0.0), ValueError, "m_eff_kg must be positive"),
    (dict(eps_inf=-1.0), ValueError, "eps_inf must be positive"),
    (dict(gamma=-1.0), ValueError, ">= 0"),
    (dict(D=-1.0), ValueError, ">= 0"),
    (dict(beta_scale=-0.5), ValueError, ">= 0"),
    (dict(h_z_m=0.0), ValueError, "h_z_m must be positive"),
    (dict(h_per_screening=0.0), ValueError, "h_per_screening"),
    (dict(face_levels=-1), ValueError, "face_ratio"),
    (dict(face_ratio=1.0), ValueError, "face_ratio"),
    (dict(coef_points=1), ValueError, "coef_points"),
])
def test_spec_refusals_name_the_offending_field(kw, exc, msg):
    base = dict(n0_m3=[1e26, 2e26], z_m=[0.0, 1e-9], m_eff_kg=MEFF, eps_inf=4.0, gamma=1.2e14)
    base.update(kw)
    with pytest.raises(exc, match=msg):
        HydroConductor2D(**base)


def test_profile_constructor_is_the_planar_route_turned_over():
    """hydro_conductor_from_profile takes the planar route's arguments (z from the superstrate face):
    the spec stores them from the layer bottom (density reversed), and its planar_layers() returns
    exactly nonlocal_tmm.hydro_layers_from_profile of the same arguments (sheet density conserved)."""
    c = _cond(conductor_id="g7")
    assert np.array_equal(np.asarray(c.n0_m3), NP[::-1])
    assert np.allclose(c.z_m, ZP[-1] - ZP[::-1], rtol=0, atol=1e-24)
    assert abs(c.thickness_m - T_NM * 1e-9) < 1e-24 and not c.laterally_structured and not c.is_local_limit
    run = c.planar_layers()
    ref = nt.hydro_layers_from_profile(ZP, NP, conductor_id="g7", **KW)
    assert len(run) == len(ref) == ZP.size - 1
    for a, b in zip(run, ref):
        for f in ("eps_inf", "wp", "gamma", "beta", "thickness_m", "D"):
            # abs=1e-30: D is exactly 0 here; the floor admits only that zero, far below every other field
            assert getattr(a, f) == pytest.approx(getattr(b, f), rel=1e-12, abs=1e-30), f
        assert a.conductor_id == "g7"
    sheet = sum(_wp2(1.0) ** -1 * L.wp ** 2 * L.thickness_m for L in run)
    assert sheet == pytest.approx(trapz(NP, ZP), rel=1e-12)
    half = c.with_options(beta_scale=0.5).planar_layers()
    assert all(h.beta == pytest.approx(0.5 * r.beta, rel=1e-15) for h, r in zip(half, run))
    # refusals of the twin constructor
    with pytest.raises(ValueError, match="staircase"):
        hydro_conductor_from_profile(ZP, NP, n_layers=64, **KW)
    with pytest.raises(ValueError, match="same length"):
        hydro_conductor_from_profile(ZP, NP[:-1], **KW)
    with pytest.raises(ValueError, match="strictly increasing"):
        hydro_conductor_from_profile(ZP[::-1], NP, **KW)
    with pytest.raises(ValueError, match="None"):
        hydro_conductor_from_profile(ZP, NP, conductor_id=None, **KW)


@pytest.mark.parametrize("conv", ["high_freq", "thomas_fermi"])
def test_point_coefficients_are_the_closed_form_drude_and_fermi_values(conv):
    """_wp2_beta: wp^2 = n e^2/(eps0 m) and beta = c_conv v_F (times beta_scale), at every density."""
    c = _cond(convention=conv, beta_scale=0.7)
    n = np.array([3e26, 8e26, 1.6e27])
    wp2, beta = c._wp2_beta(n)
    assert np.allclose(wp2, _wp2(n), rtol=1e-12, atol=0)
    assert np.allclose(beta, 0.7 * _beta(n, conv), rtol=1e-12, atol=0)
    lays = c._point_layers(n)
    assert [L.thickness_m for L in lays] == [0.0] * 3
    assert np.allclose([L.wp for L in lays], np.sqrt(_wp2(n)), rtol=1e-12, atol=0)


def test_identical_columns_are_a_laterally_uniform_conductor():
    """An (Nz, Nx) density whose columns are identical collapses to (Nz, 1): it is laterally uniform
    (and is planned exactly like the 1-D profile); differing columns stay structured."""
    z = np.array([0.0, 2e-9, 5e-9])
    n1 = np.array([1e27, 6e26, 4e26])
    same = HydroConductor2D(np.repeat(n1[:, None], 4, axis=1), z, x_m=[0, 25e-9, 50e-9, 75e-9], **KW)
    one = HydroConductor2D(n1, z, **KW)
    assert not same.laterally_structured
    a = same.region_setup(lambda_m=LAM, thickness_nm=5.0, period_nm=100.0, max_grid_lines=256, where="w")
    b = one.region_setup(lambda_m=LAM, thickness_nm=5.0, period_nm=100.0, max_grid_lines=256, where="w")
    assert a["z_lines_nm"] == b["z_lines_nm"] and a["x_lines_nm"] == b["x_lines_nm"] == []
    assert np.array_equal(a["local_eps_samples"], b["local_eps_samples"])
    col = np.repeat(n1[:, None], 4, axis=1)
    col[:, 2] *= 1.5
    assert HydroConductor2D(col, z, x_m=[0, 25e-9, 50e-9, 75e-9], **KW).laterally_structured
    assert HydroConductor2D(n1[:, None], z, **KW).n0_m3.shape == (3, 1)     # (Nz, 1) accepted


# ------------------------------------------------------------------------------------------------
# the conductor mesh plan (region_setup) and its use by build_tm_cell's region record
# ------------------------------------------------------------------------------------------------

def test_region_setup_sizes_rows_from_the_longitudinal_wavevector():
    """Nonlocal conductor: the H_y operator sees eps_inf only; the local eps_T samples are the closed
    form; h_z = h_per_screening / max|k_L| over the profile; rows = ceil(t / h_z); the profile samples
    and the geometric face grading h1 r^l are mesh lines."""
    c = _cond(h_per_screening=0.8, face_levels=3, face_ratio=0.4)
    info = c.region_setup(lambda_m=LAM, thickness_nm=T_NM, period_nm=100.0, max_grid_lines=256,
                          where="film")
    n = NP[::-1]
    assert info["nonlocal"] and info["operator_eps"] == complex(4.0)
    eps_t = 4.0 - _wp2(n) / (OMEGA ** 2 + 1j * OMEGA * 1.2e14)
    assert np.allclose(info["local_eps_samples"][:, 0], eps_t, rtol=1e-12, atol=0)
    kmax = float(np.max(np.abs(_kL(n)))) * 1e-9
    assert info["kL_max_per_nm"] == pytest.approx(kmax, rel=1e-12, abs=0)
    assert info["h_z_nm"] == pytest.approx(0.8 / kmax, rel=1e-12, abs=0)
    rows = math.ceil(T_NM / (0.8 / kmax))
    assert info["rows"] == rows
    zl = np.asarray(info["z_lines_nm"])
    assert np.all(np.diff(zl) > 0) and zl[0] == 0.0 and zl[-1] == pytest.approx(T_NM, abs=1e-12)
    want = list(np.linspace(0.0, T_NM, rows + 1)) + list((ZP[-1] - ZP[::-1]) * 1e9)
    h1 = T_NM / rows
    want += [h1 * 0.4 ** lv for lv in (1, 2, 3)] + [T_NM - h1 * 0.4 ** lv for lv in (1, 2, 3)]
    for w in want:
        assert np.min(np.abs(zl - w)) < 1e-9, w
    assert info["x_lines_nm"] == []
    # explicit h_z_m and lateral h_x_m; samples not aligned when there are more than the cap
    info2 = c.with_options(h_z_m=0.5e-9, h_x_m=30e-9, face_levels=0).region_setup(
        lambda_m=LAM, thickness_nm=T_NM, period_nm=100.0, max_grid_lines=4, where="film")
    assert info2["h_z_nm"] == pytest.approx(0.5) and info2["rows"] == 16
    assert np.allclose(info2["z_lines_nm"], np.linspace(0.0, T_NM, 17), rtol=0, atol=1e-12)
    assert np.allclose(info2["x_lines_nm"], np.linspace(0.0, 100.0, 5), rtol=0, atol=1e-12)
    # GNOR diffusion enters k_L through beta_eff^2 = beta^2 + D (gamma - i omega)
    infoD = _cond(D=3e-4).region_setup(lambda_m=LAM, thickness_nm=T_NM, period_nm=100.0,
                                       max_grid_lines=256, where="film")
    assert infoD["kL_max_per_nm"] == pytest.approx(float(np.max(np.abs(_kL(n, D=3e-4)))) * 1e-9,
                                                   rel=1e-12, abs=0)


def test_local_limit_plans_the_local_gridded_layer():
    """beta_scale = 0 and D = 0: no longitudinal wave, so the operator is the local GriddedEps2D of
    eps_T on the profile samples, the mesh lines are those samples (what the local solver puts in a
    GriddedEps2D layer) and _hydro_region_eps attaches no extension."""
    c = _cond(beta_scale=0.0)
    assert c.is_local_limit
    info = c.region_setup(lambda_m=LAM, thickness_nm=T_NM, period_nm=100.0, max_grid_lines=256,
                          where="film")
    assert not info["nonlocal"] and info["kL_max_per_nm"] is None
    op = info["operator_eps"]
    assert isinstance(op, GriddedEps2D)
    eps_t = 4.0 - _wp2(NP[::-1]) / (OMEGA ** 2 + 1j * OMEGA * 1.2e14)
    assert np.allclose(op.values, eps_t, rtol=1e-12, atol=0)
    assert info["rows"] == ZP.size - 1
    assert np.allclose(info["z_lines_nm"], (ZP[-1] - ZP[::-1]) * 1e9, rtol=0, atol=1e-9)
    reg = _hydro_region_eps(c, "film", lambda_m=LAM, thickness_nm=T_NM, period_nm=100.0,
                            max_grid_lines=256)
    assert reg.hydro is None and reg.grid is not None
    assert reg.is_metal == bool(np.any(eps_t.real < 0)) is True     # the accumulation is metallic
    # the nonlocal twin: constant eps_inf operator, extension attached, sizing from the local eps_T
    regn = _hydro_region_eps(_cond(), "film", lambda_m=LAM, thickness_nm=T_NM, period_nm=100.0,
                             max_grid_lines=256)
    assert regn.hydro is not None and regn.const == (4.0 + 0j, 0j, 0j, 4.0 + 0j)
    nloc = np.sqrt(eps_t)
    assert regn.n_max == pytest.approx(float(np.max(np.abs(nloc.real))), rel=1e-12, abs=0)
    assert regn.kappa_max == pytest.approx(float(np.max(np.abs(nloc.imag))), rel=1e-12, abs=0)
    assert regn.is_metal == bool(np.any(eps_t.real < 0)) and not regn.isotropic
    zl = regn.mesh_lines(9.0, 100.0, 256)[1]
    assert zl[0] == pytest.approx(9.0) and zl[-1] == pytest.approx(9.0 + T_NM)


def test_region_setup_refuses_a_thickness_or_lateral_axis_mismatch():
    c = _cond()
    with pytest.raises(ValueError, match="check the length unit"):
        c.region_setup(lambda_m=LAM, thickness_nm=T_NM + 1.0, period_nm=100.0, max_grid_lines=256, where="f")
    z = np.array([0.0, 5e-9])
    lat = HydroConductor2D(np.array([[1e27, 5e26], [6e26, 4e26]]), z, x_m=[0.0, 50.0], **KW)  # nm, not m
    with pytest.raises(ValueError, match="outside the cell"):
        lat.region_setup(lambda_m=LAM, thickness_nm=5.0, period_nm=100.0, max_grid_lines=256, where="f")


def test_laterally_structured_density_is_resampled_exactly():
    """(Nz, Nx) density: the voxel resample of a density that is linear in z and piecewise linear
    (kinks at the samples only) in periodic x reproduces that function exactly on the uniform grid;
    its x nodes become mesh lines (when within the cap) and k_L is maximized over all of it."""
    P = 100.0
    xs = np.array([0.0, 20.0, 50.0, 75.0])                 # nm, non-uniform
    def g(x):                                              # triangle wave, kinks at 0 and 50
        x = np.mod(x, P)
        return np.where(x <= 50.0, 1.0 + x / 50.0, 2.0 - (x - 50.0) / 50.0)
    z = np.array([0.0, 1.5, 5.0])
    def n_of(zz, xx):
        return 3e26 * (1.0 + 0.5 * zz / 5.0) * g(xx)
    dens = n_of(z[:, None], xs[None, :])
    c = HydroConductor2D(dens, z * 1e-9, x_m=xs * 1e-9, coef_points=11, **KW)
    xu, zu, nv = c._voxel(P, 256, "f")
    assert np.allclose(xu, np.linspace(0.0, P, 6), rtol=0, atol=1e-12)    # finest native dx = 20
    assert np.allclose(zu, np.linspace(0.0, 5.0, 11), rtol=0, atol=1e-12)
    assert np.allclose(nv, n_of(zu[:, None], xu[None, :]), rtol=1e-12, atol=0)
    info = c.region_setup(lambda_m=LAM, thickness_nm=5.0, period_nm=P, max_grid_lines=256, where="f")
    assert np.allclose(info["x_lines_nm"], xu, rtol=0, atol=1e-12)
    assert info["kL_max_per_nm"] == pytest.approx(float(np.max(np.abs(_kL(dens.ravel())))) * 1e-9,
                                                  rel=1e-12, abs=0)
    capped = c.region_setup(lambda_m=LAM, thickness_nm=5.0, period_nm=P, max_grid_lines=3, where="f")
    assert capped["x_lines_nm"] == []


def test_dedupe_merges_points_closer_than_the_tolerance():
    assert _dedupe([0.0, 1.0, 1.0 + 1e-13, 0.5, 2.0, 2.0 + 5e-4], 1e-3) == [0.0, 0.5, 1.0, 2.0]


# ------------------------------------------------------------------------------------------------
# validity
# ------------------------------------------------------------------------------------------------

def _u(n):
    """u = |k_L| v_F / omega (v_F = beta / sqrt(3/5) for the 'high_freq' beta)."""
    return float(np.abs(_kL(n)) * _beta(n) / math.sqrt(0.6) / OMEGA)


def test_validity_is_the_closed_form_u_and_reports_the_worst_column():
    """u = |k_L| v_F / omega per sub-layer. u ~ sqrt(|omega^2 - wp^2/eps_inf|)/omega/sqrt(3/5) is small
    only near the screened plasma frequency: a film at the epsilon-near-zero density is inside the
    'high_freq' limit 0.5, the 1.6e27 accumulation is not; for an (Nz, 2) density the report is the
    worse column's (the failing one, or the larger u when both pass)."""
    z = np.array([0.0, 5e-9])
    n_enz = OMEGA ** 2 * EPS0 * MEFF * 4.0 / Q_E ** 2
    lo = HydroConductor2D([n_enz, n_enz], z, **KW).validity(OMEGA)
    assert lo.ok and lo.u[0] == pytest.approx(_u(n_enz), rel=1e-12, abs=0) and _u(n_enz) < 0.5
    hi = HydroConductor2D([1.6e27, 1.6e27], z, **KW).validity(OMEGA)
    assert not hi.ok and hi.u[0] == pytest.approx(_u(1.6e27), rel=1e-12, abs=0) and hi.u[0] > 0.5
    assert "u = |k_L| v_F/omega" in hi.message
    two = HydroConductor2D(np.array([[n_enz, 1.6e27], [n_enz, 1.6e27]]), z, x_m=[0.0, 50e-9], **KW)
    rep = two.validity(OMEGA)
    assert not rep.ok and rep.u == hi.u
    n2 = 1.1 * n_enz
    both = HydroConductor2D(np.array([[n_enz, n2], [n_enz, n2]]), z, x_m=[0.0, 50e-9], **KW)
    rep2 = both.validity(OMEGA)
    assert rep2.ok and rep2.u[0] == pytest.approx(max(_u(n_enz), _u(n2)), rel=1e-12, abs=0)
    assert _u(n2) > _u(n_enz)                                # the gate discriminates the columns


# ------------------------------------------------------------------------------------------------
# the extension objects (their NGSolve parts are on the solver leg)
# ------------------------------------------------------------------------------------------------

def _fake_cell(spec, P=100.0):
    reg = {"lo": SimpleNamespace(eps=_RegionEps(2.0, "lo"), z_lo_nm=0.0, z_hi_nm=9.0),
           "f": SimpleNamespace(eps=SimpleNamespace(hydro=spec), z_lo_nm=9.0, z_hi_nm=9.0 + T_NM),
           "up": SimpleNamespace(eps=_RegionEps(2.0, "up"), z_lo_nm=9.0 + T_NM, z_hi_nm=18.0 + T_NM)}
    return SimpleNamespace(regions=reg, lambda_m=LAM, period_nm=P, n_super=1.0 + 0j,
                           mesh_spec=F.TMMeshSpec())


def test_extensions_are_created_for_nonlocal_conductor_regions_only():
    """hydro_extensions_for_cell: one extension per region carrying a hydrodynamic spec, with its z
    window, the solve frequency and condensation of the element-local current requested; its incident
    power normalization is P Re(kz0 / eps_super) = P k0 cos(theta) in vacuum."""
    spec = _cond()
    cell = _fake_cell(spec)
    exts = hydro_extensions_for_cell(cell)
    assert len(exts) == 1
    e = exts[0]
    assert e.region == "f" and e.spec is spec and e.condense is True
    assert e.z_lo == 9.0 and e.t_nm == T_NM
    assert e.omega == pytest.approx(OMEGA, rel=1e-15)
    k0 = 2.0 * math.pi / 1200.0
    kx = k0 * math.sin(math.radians(40.0))
    ctx = SimpleNamespace(cell=cell, k0=k0, kx=kx)
    assert e._norm(ctx) == pytest.approx(100.0 * k0 * math.cos(math.radians(40.0)), rel=1e-12, abs=0)


def test_diagnostics_report_validity_and_warn_without_blocking():
    """The diagnostics hook returns the validity report, u_max and the conductor mesh plan; it warns
    (UnreliableResultWarning naming the layer) when outside validity unless check_validity=False.
    (space() -- NGSolve -- records the DOF counts it reports; they are stubbed here.)"""
    spec_warn = _cond(check_validity=True, conductor_id="acc")
    cell = _fake_cell(spec_warn)
    e = hydro_extensions_for_cell(cell)[0]
    e._ndof = (SimpleNamespace(ndof=12), None, 5)
    ctx = SimpleNamespace(cell=cell)
    with pytest.warns(UnreliableResultWarning, match="hydrodynamic layer 'f'"):
        out = e.diagnostics(ctx, None)["hydro"]["f"]
    rep = spec_warn.validity(OMEGA)
    got = out["validity"]                                    # (degeneracy is NaN without T_K)
    assert not rep.ok and (got.u, got.ok, got.message) == (rep.u, rep.ok, rep.message)
    assert out["u_max"] == max(v for v in rep.u if v == v)
    plan = spec_warn.region_setup(lambda_m=LAM, thickness_nm=T_NM, period_nm=100.0,
                                  max_grid_lines=cell.mesh_spec.max_grid_lines, where="f")
    assert out["h_z_nm"] == plan["h_z_nm"] and out["rows"] == plan["rows"]
    assert (out["ndof_current"], out["ndof_potential"], out["conductor_id"]) == (12, 5, "acc")
    assert out["absorption"] is None
    quiet = hydro_extensions_for_cell(_fake_cell(_cond(check_validity=False)))[0]
    quiet._ndof = (SimpleNamespace(ndof=1), None, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet.diagnostics(SimpleNamespace(cell=_fake_cell(quiet.spec)), None)


# ------------------------------------------------------------------------------------------------
# convergence tabulation
# ------------------------------------------------------------------------------------------------

def test_z_convergence_scales_only_the_conductor_mesh_and_measures_the_order(monkeypatch):
    """hydro_z_convergence scales h_z_m when it is set and h_per_screening otherwise, leaves other
    layers untouched, and reports the observed order: for a synthetic r = r_inf + C_p h^p the rate
    from three successive rows of one order is exactly p."""
    seen = []
    r_inf = 0.3 + 0.1j

    def fake_solve(period, layers, *, order, return_solution, **kw):
        assert return_solution and kw == {"lambda_m": LAM}
        seen.append(layers)
        a, b = layers[1].eps, layers[3].eps
        h = a.h_per_screening                               # equal factors on both conductors
        assert b.h_z_m == pytest.approx(2e-10 * h / 1.0, rel=1e-12, abs=0)
        p = {4: 4, 2: 2}[order]
        r = r_inf + (0.02 if order == 4 else -0.05j) * h ** p
        return SimpleNamespace(result=SimpleNamespace(r=r, R=abs(r) ** 2), energy_residual=1e-9,
                               ndof=1000 * order, diagnostics={"h": h})

    monkeypatch.setattr(F, "solve_fem_2d_tm", fake_solve)
    diel = TMLayer("lo", 9e-9, 2.0)
    layers = [diel, TMLayer("f1", T_NM * 1e-9, _cond()), TMLayer("mid", 4e-9, 2.0),
              TMLayer("f2", T_NM * 1e-9, _cond(h_z_m=2e-10)), TMLayer("up", 9e-9, 2.0)]
    rows = hydro_z_convergence(100e-9, layers, h_factors=(2.0, 1.0, 0.5, 0.25), orders=(4, 2),
                               lambda_m=LAM)
    assert [(r_["order"], r_["h_factor"]) for r_ in rows] == [(o, f) for o in (4, 2)
                                                              for f in (2.0, 1.0, 0.5, 0.25)]
    assert all(s[0] is diel and s[2] is layers[2] for s in seen)
    for k, row in enumerate(rows):
        same = [x for x in rows[:k] if x["order"] == row["order"]]
        assert row["d_prev"] == (abs(row["r"] - same[-1]["r"]) if same else None)
        assert row["d_last"] == abs(row["r"] - rows[-1]["r"])
        if len(same) >= 2:
            assert row["rate"] == pytest.approx(row["order"], rel=1e-9)
        else:
            assert row["rate"] is None
        assert row["diagnostics"]["h"] == row["h_factor"] and row["ndof"] == 1000 * row["order"]
    assert rows[0]["r"] == pytest.approx(r_inf + 0.02 * 2.0 ** 4, rel=1e-15, abs=0)
    assert cmath.isclose(rows[-1]["r"], r_inf - 0.05j * 0.25 ** 2, rel_tol=1e-15)
