"""v0.16: size-effect (boundary) damping and a density-dependent background permittivity for the
free-carrier optical model (materials.scattering.BoundaryScattering / fuchs_sondheimer_factor,
materials.optical_model.DrudeOptical(boundary=, background=), BandFillingBackground, TabulatedBackground),
and their plumbing through the permittivity assembler. Notes:
docs/audit/2026-11-14-v0.16-thin-film-damping-and-background.md.

Every formula gate has an oracle written HERE (scipy quadrature of the defining integral, the defining
closed form re-derived, or an independent dispersion relation); the default-path gates are bit-identity
A/B checks against a frozen copy of the v0.15.1 formula and, when the v0.15.1 release commit BASE_REF is
available in the repository, against that commit's modules loaded side by side (skipped otherwise). On v0.15.1 every test here fails
at import (no BoundaryScattering / BandFillingBackground in dynameta.materials).
"""
import math
import subprocess
import sys
import types
import pathlib

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, EPS0, HBAR, M_E, Q_E
from dynameta.materials import (BandFillingBackground, BoundaryScattering, ConstantOptical, DrudeOptical,
                                KaneOpticalMass, MatthiessenGamma, Material, MaterialRegistry,
                                STACK_AXIS_FIELD, TabulatedBackground, fs_effective_rate,
                                fuchs_sondheimer_factor)

LAM = 1200e-9
W12 = 2.0 * math.pi * C_LIGHT / LAM
M = 0.30 * M_E
REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_REF = "b8a73c0"                       # the v0.15.1 release commit on main


# ------------------------------------------------------------------------------------------------
# oracles
# ------------------------------------------------------------------------------------------------
def _fs_quad(kappa, p):
    """F(kappa, p) from the defining Fuchs-Sondheimer integral by scipy quadrature (independent of the
    E_n series). Real kappa: plain quad; complex kappa: p = 0 via the Fourier-weight (QAWF) routine."""
    from scipy.integrate import quad
    k = complex(kappa)
    if k.imag == 0.0:
        a = k.real
        f = lambda t: (t ** -3 - t ** -5) * (1.0 - math.exp(-a * t)) / (1.0 - p * math.exp(-a * t))  # noqa: E731
        integ = quad(f, 1.0, np.inf, limit=500, epsabs=1e-15, epsrel=1e-13)[0]
        return 1.0 - 1.5 * (1.0 - p) / a * integ
    assert p == 0.0
    a, b = k.real, -k.imag                      # exp(-kappa t) = exp(-a t) exp(+i b t)
    f = lambda t: (t ** -3 - t ** -5) * math.exp(-a * t)  # noqa: E731
    re = quad(f, 1.0, np.inf, weight="cos", wvar=b)[0]
    im = quad(f, 1.0, np.inf, weight="sin", wvar=b)[0]
    return 1.0 - 1.5 / k * (0.25 - (re + 1j * im))


def _vF(n, m):
    return HBAR * (3.0 * math.pi ** 2 * np.asarray(n, float)) ** (1.0 / 3.0) / m


def _drude_v015(eps_inf, m, g, n, lam):
    """Frozen copy of the v0.15.1 DrudeOptical.eps arithmetic (same operation order)."""
    n = np.asarray(n, dtype=np.float64)
    omega = 2.0 * np.pi * C_LIGHT / float(lam)
    omega_p2 = n * Q_E * Q_E / (EPS0 * m)
    return eps_inf - omega_p2 / (omega * omega + 1j * omega * g)


def _base_source(rel):
    try:
        out = subprocess.run(["git", "show", "{}:{}".format(BASE_REF, rel)], cwd=str(REPO),
                             capture_output=True, text=True, encoding="utf-8")
    except (OSError, ValueError):
        return None
    return out.stdout if out.returncode == 0 else None


def _base_module(rel, name):
    src = _base_source(rel)
    if src is None:
        pytest.skip("base commit {} not in this clone (the frozen-formula A/B gates still run)".format(
            BASE_REF))
    mod = types.ModuleType(name)
    mod.__file__ = "<v0.15.1 {}>".format(rel)
    sys.modules[name] = mod            # dataclasses resolve annotations through sys.modules
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


# ------------------------------------------------------------------------------------------------
# Fuchs-Sondheimer kernel
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kappa,p", [(1e-3, 0.0), (0.1, 0.0), (0.901, 0.0), (1.0, 0.0), (10.0, 0.0),
                                     (60.0, 0.0), (1.0, 0.5), (0.3, 0.9), (5.0, 0.3), (100.0, 0.3)])
def test_fs_factor_real_kappa_matches_quadrature(kappa, p):
    assert abs(complex(fuchs_sondheimer_factor(kappa, p)) - _fs_quad(kappa, p)) < 1e-12


@pytest.mark.parametrize("kappa", [0.901 - 11.8j, 0.05 - 3.0j, 2.0 - 0.5j, 0.3 - 40.0j, 1.0 - 70.0j])
def test_fs_factor_complex_kappa_matches_quadrature(kappa):
    """The optical-frequency kappa = d (gamma - i omega) / v_F (the 6 nm generic film: 0.901 - 11.8i). The
    |kappa| = 70 point exercises the asymptotic E_n branch."""
    assert abs(complex(fuchs_sondheimer_factor(kappa, 0.0)) - _fs_quad(kappa, 0.0)) < 1e-10


def test_fs_factor_complex_kappa_with_specularity_matches_quadrature():
    from scipy.integrate import quad
    k, p = 0.8 - 2.5j, 0.4

    def g(t, part):
        x = np.exp(-k * t)
        v = (t ** -3 - t ** -5) * (1.0 - x) / (1.0 - p * x)
        return v.real if part == 0 else v.imag
    integ = sum(f * quad(g, 1.0, 80.0, args=(i,), limit=2000, epsabs=1e-14)[0]
                + f * quad(g, 80.0, np.inf, args=(i,), limit=200, epsabs=1e-14)[0]
                for i, f in ((0, 1.0), (1, 1j)))
    ref = 1.0 - 1.5 * (1.0 - p) / k * integ
    assert abs(complex(fuchs_sondheimer_factor(k, p)) - ref) < 1e-10


def test_fs_limits():
    k = np.array([1e-3, 0.5, 3.0, 0.7 - 9.0j])
    # p = 1: no size effect, exactly
    assert np.array_equal(fuchs_sondheimer_factor(k, 1.0), np.ones(4, complex))
    # thick film: F = 1 - 3 (1 - p) / (8 kappa) up to exponentially small terms (E_n(kappa) ~ e^-kappa)
    for p in (0.0, 0.3):
        for kap in (40.0, 200.0, 1e4):
            assert abs(fuchs_sondheimer_factor(kap, p) - (1.0 - 3.0 * (1.0 - p) / (8.0 * kap))) < 1e-15 + 1e-14
    # thin film, p = 0: F ~ (3/4) kappa (ln(1/kappa) + 0.4228) (Sondheimer 1952)
    for kap in (1e-4, 1e-5):
        ratio = fuchs_sondheimer_factor(kap, 0.0).real / (0.75 * kap * (math.log(1.0 / kap) + 0.4228))
        assert abs(ratio - 1.0) < 1e-5
    with pytest.raises(ValueError):
        fuchs_sondheimer_factor(-1.0)
    with pytest.raises(ValueError):
        fuchs_sondheimer_factor(1.0, 1.5)


def test_fs_effective_rate_limits():
    """Thick film at optical frequency: Re gamma_eff = gamma + 3 (1 - p) v_F / (8 d) (the additive
    Matthiessen rate); d.c.: gamma / F(d gamma / v_F) (real)."""
    vF, g, w = 8.0e5, 1.2e14, W12
    for d, p in ((2e-6, 0.0), (5e-6, 0.4)):
        ge = fs_effective_rate(w, g, vF, d, p)
        want = g + 3.0 * (1.0 - p) * vF / (8.0 * d)
        assert abs(ge.real - want) < 1e-6 * want
        assert abs(ge.imag) < 1e-6 * want
    dc = fs_effective_rate(0.0, g, vF, 6e-9, 0.0)
    assert abs(dc.imag) < 1e-6 and abs(dc.real - g / _fs_quad(6e-9 * g / vF, 0.0).real) < 1e-6 * g


# ------------------------------------------------------------------------------------------------
# DrudeOptical defaults: bit-identical
# ------------------------------------------------------------------------------------------------
def test_default_drude_is_bit_identical_to_the_v0151_formula():
    n = np.array([0.0, 1e24, 3e26, 1.6e27, 3.3e27])
    kane = KaneOpticalMass(m0_kg=0.22 * M_E, alpha_eV=0.4)
    mg = MatthiessenGamma(gamma_const_rad_s=8e13, bh_prefactor_rad_s=3e13, m_opt=kane)
    for m, g in ((M, 1.2e14), (kane, mg), (M, 0.0)):
        d = DrudeOptical(4.0, m, g)
        mm = m(n) if callable(m) else m
        gg = g(n) if callable(g) else g
        assert np.array_equal(d.eps(LAM, n_m3=n), _drude_v015(4.0, mm, gg, n, LAM))
        assert repr(d) == "DrudeOptical(eps_inf={!r}, m_opt_kg={!r}, gamma_rad_s={!r})".format(4.0, m, g)
    assert DrudeOptical(4.0, M, 1.2e14).wp_gamma(3e26) == pytest.approx(
        (math.sqrt(3e26 * Q_E ** 2 / (EPS0 * M)), 1.2e14), rel=1e-15)


def test_default_path_ab_against_v0151_modules():
    """Same-process A/B: the v0.15.1 optical_model / scattering modules (git show of the release commit)
    against this tree, on eps, repr, the Kane mass and the assembler's OpticalModelEffect route."""
    om0 = _base_module("dynameta/materials/optical_model.py", "_v0151_optical_model")
    sc0 = _base_module("dynameta/materials/scattering.py", "_v0151_scattering")
    from dynameta.core.effects import OpticalModelEffect
    n = np.geomspace(1e20, 4e27, 57).reshape(3, 19)
    k0 = sc0.KaneOpticalMass(m0_kg=0.22 * M_E, alpha_eV=0.4)
    k1 = KaneOpticalMass(m0_kg=0.22 * M_E, alpha_eV=0.4)
    assert np.array_equal(k0(n), k1(n))
    for lam in (900e-9, 1200e-9, 1550e-9):
        a = om0.DrudeOptical(4.0, k0, 1.1e14)
        b = DrudeOptical(4.0, k1, 1.1e14)
        assert np.array_equal(a.eps(lam, n_m3=n), b.eps(lam, n_m3=n))
        assert np.array_equal(a.eps(lam, n_m3=n), OpticalModelEffect(b).eps({"n": n}, lam))
        assert repr(om0.DrudeOptical(4.0, M, 1.2e14)) == repr(DrudeOptical(4.0, M, 1.2e14))


def test_default_assembler_output_is_bit_identical_and_gets_no_position_key():
    """assemble_eps on a default Drude film: the EpsField values equal the frozen v0.15.1 formula on the
    lifted density, and the material receives exactly {'n'} (no stack-axis key)."""
    from dynameta.core import NM, assemble_eps
    from dynameta.core.lift import ExtrudeLift
    reg = _registry(DrudeOptical(4.25, 0.225 * M_E, 1.1e14))
    seen = []

    class _Spy:
        materials = reg

        def eps_grid(self, name, fields, lam):
            seen.append(sorted(fields))
            from dynameta.core.n_to_eps import MaterialEpsMap
            return MaterialEpsMap(reg).eps_grid(name, fields, lam)

        def scalar_eps(self, name, lam):
            return complex(reg.get(name).eps(lam))
    x, z, n2 = _film_grid()
    out = assemble_eps(_field(n2, x, z), _alignment(z), _Spy(), ExtrudeLift(period_y_m=300e-9), LAM)
    v = out["film"].values_zyx                                  # (Nz, Ny, Nx)
    assert seen == [["n"]]
    want = _drude_v015(4.25, 0.225 * M_E, 1.1e14, n2.T[:, None, :], LAM)
    assert np.array_equal(v, np.broadcast_to(want, v.shape))


# ------------------------------------------------------------------------------------------------
# boundary scattering: formulas
# ------------------------------------------------------------------------------------------------
def test_boundary_off_switches_reproduce_the_bulk_drude_in_every_component():
    n = np.array([1e25, 3e26, 1.2e27])
    bulk = DrudeOptical(4.0, M, 1.2e14).eps(LAM, n_m3=n)
    T = DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering(specularity=1.0, thickness_m=6e-9)
                     ).eps_tensor(LAM, n_m3=n)
    for i in range(3):
        assert np.array_equal(T[:, i, i], bulk)
    assert not np.any(T[:, 0, 1]) and not np.any(T[:, 0, 2]) and not np.any(T[:, 1, 2])
    T2 = DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering(in_plane=False, thickness_m=6e-9)
                      ).eps_tensor(LAM, n_m3=n)
    assert np.array_equal(T2[:, 0, 0], bulk) and np.array_equal(T2[:, 2, 2], bulk)


def test_in_plane_component_is_the_fuchs_sondheimer_film_conductivity():
    """eps_xx = eps_inf - wp^2 F(kappa) / (omega (omega + i gamma)), kappa = d (gamma - i omega) / v_F,
    F by quadrature in the test (p = 0); the result differs from the bulk by the expected sign (more
    loss)."""
    d, g = 6e-9, 1.2e14
    for n in (2e26, 3e26, 1.2e27):
        T = DrudeOptical(4.0, M, g, boundary=BoundaryScattering(thickness_m=d)).eps_tensor(LAM, n_m3=n)
        wp2 = n * Q_E ** 2 / (EPS0 * M)
        F = _fs_quad(d * (g - 1j * W12) / _vF(n, M), 0.0)
        want = 4.0 - wp2 * F / (W12 * (W12 + 1j * g))
        # budget: the QAWF oracle carries ~1e-12 relative in F, times the Drude term |wp^2 / omega^2| ~ 5
        assert abs(T[0, 0] - want) < 1e-11 * abs(wp2 / W12 ** 2) and T[1, 1] == T[0, 0]
        assert T[0, 0].imag > DrudeOptical(4.0, M, g).eps(LAM, n_m3=n).imag
        assert T[2, 2] == DrudeOptical(4.0, M, g).eps(LAM, n_m3=n)          # A = 0: z untouched


def test_out_of_plane_rate_is_matthiessen_with_the_kreibig_term():
    """eps_zz = eps_b - wp^2 / (omega (omega + i (gamma(n) + A v_F / (2 d)))) with a MatthiessenGamma bulk
    rate; the rate recovered from eps_zz equals the sum of the two channels."""
    d, A = 6e-9, 0.7
    mg = MatthiessenGamma(gamma_const_rad_s=6e13, bh_prefactor_rad_s=4e13, bh_n_ref_m3=3e26)
    n = np.array([1e26, 3e26, 9e26])
    T = DrudeOptical(4.0, M, mg, boundary=BoundaryScattering(in_plane=False, perp_coefficient=A,
                                                             thickness_m=d)).eps_tensor(LAM, n_m3=n)
    wp2 = n * Q_E ** 2 / (EPS0 * M)
    gam = mg(n) + A * _vF(n, M) / (2.0 * d)
    want = 4.0 - wp2 / (W12 * (W12 + 1j * gam))
    assert np.allclose(T[:, 2, 2], want, rtol=1e-13, atol=0)
    rec = (wp2 / (4.0 - T[:, 2, 2])).imag / W12                 # omega (omega + i Gamma) = wp^2 / (eps_b - eps)
    assert np.allclose(rec, gam, rtol=1e-10)


def test_fermi_velocity_follows_the_kane_band():
    """With a KaneOpticalMass the rate uses v_F = (1/hbar) dE/dk at k_F of E (1 + alpha E) = hbar^2 k^2 /
    (2 m0) -- computed here from the dispersion (quadratic root), not from the optical mass."""
    m0, alpha, d, A = 0.22 * M_E, 0.4, 6e-9, 1.0
    kane = KaneOpticalMass(m0_kg=m0, alpha_eV=alpha)
    n = np.array([2e26, 1e27])
    T = DrudeOptical(4.0, kane, 1.2e14, boundary=BoundaryScattering(in_plane=False, perp_coefficient=A,
                                                                  thickness_m=d)).eps_tensor(LAM, n_m3=n)
    kF = (3.0 * math.pi ** 2 * n) ** (1.0 / 3.0)
    g = HBAR ** 2 * kF ** 2 / (2.0 * m0)
    a = alpha / Q_E
    EF = (-1.0 + np.sqrt(1.0 + 4.0 * a * g)) / (2.0 * a)
    vF = HBAR * kF / (m0 * (1.0 + 2.0 * a * EF))                 # dE/dk / hbar
    wp2 = n * Q_E ** 2 / (EPS0 * kane(n))
    want = 4.0 - wp2 / (W12 * (W12 + 1j * (1.2e14 + A * vF / (2.0 * d))))
    assert np.allclose(T[:, 2, 2], want, rtol=1e-12, atol=0)
    assert np.allclose(kane.fermi_energy_J(n), EF, rtol=1e-13, atol=0.0)    # ~1e-19 J: no absolute floor


def test_local_confinement_length():
    """1/L = 1/(2d) + |dn/dz|/n: flat profile -> 2d exactly; exponential accumulation layer n = nb + n1
    exp(-z/w) -> the analytic gradient (second-order np.gradient on a fine grid)."""
    b = BoundaryScattering(perp_coefficient=1.0, perp_length="local")
    z = np.linspace(0.0, 6e-9, 2001)
    # (np.gradient of a constant on a linspace grid is zero to round-off of the non-uniform-spacing weights)
    assert np.allclose(b.confinement_length_m(np.full(z.size, 3e26), 6e-9, z), 1.2e-8, rtol=1e-12, atol=0)
    nb, n1, w = 3e26, 1.2e27, 0.7e-9
    n = nb + n1 * np.exp(-z / w)
    L = b.confinement_length_m(n, 6e-9, z)
    want = 1.0 / (1.0 / 1.2e-8 + (n1 / w) * np.exp(-z / w) / n)
    # np.gradient's interior central difference of n1 exp(-z/w) is the exact derivative times sinh(x)/x, x = h/w,
    # and 1/L = 1/(2d) + |dn/dz|/n, so |L_num/L - 1| <= sinh(x)/x - 1 = x^2/6 + ... (3.06e-6 here; MEASURED 2.85e-6
    # at the first interior node, where the gradient term carries 93 % of 1/L); + 1e-12 for round-off
    x = (z[1] - z[0]) / w
    assert np.max(np.abs(L[1:-1] / want[1:-1] - 1.0)) <= math.sinh(x) / x - 1.0 + 1e-12
    # 2-D grids: the gradient is along the LAST axis
    L2 = b.confinement_length_m(np.tile(n, (3, 1)), 6e-9, z)
    assert np.array_equal(L2[1], L)
    with pytest.raises(ValueError):
        b.confinement_length_m(n, 6e-9, None)


def test_refusals_and_validation():
    d = DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering(perp_coefficient=0.5, thickness_m=6e-9))
    with pytest.raises(ValueError, match="UNIAXIAL"):
        d.eps(LAM, n_m3=3e26)
    with pytest.raises(NotImplementedError):
        d.wp_gamma(3e26)
    with pytest.raises(NotImplementedError):
        DrudeOptical(4.0, M, 1.2e14, background=lambda n, lam: 4.0 + 0 * n).wp_gamma(3e26)
    with pytest.raises(ValueError, match="thickness"):
        DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering()).eps_tensor(LAM, n_m3=3e26)
    with pytest.raises(TypeError):
        DrudeOptical(4.0, M, 1.2e14, boundary=0.5)
    for bad in (dict(specularity=-0.1), dict(perp_coefficient=-1.0), dict(perp_length="wall"),
                dict(thickness_m=0.0)):
        with pytest.raises(ValueError):
            BoundaryScattering(**bad)


def test_carrier_heating_refuses_the_new_options():
    from dynameta.carriers.carrier_heating import carrier_heating_transient
    d = DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering(thickness_m=6e-9))
    with pytest.raises(NotImplementedError, match="boundary scattering"):
        carrier_heating_transient(np.linspace(0, 1e-12, 3), lambda t: 0.0, LAM, drude0=d,
                                  ttm_params=None, n_m3=3e26, alpha_per_eV=0.4, m0_kg=M)


# ------------------------------------------------------------------------------------------------
# assembler plumbing, routes, cache fingerprint
# ------------------------------------------------------------------------------------------------
def _registry(film_optical):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("ox", ConstantOptical(complex(2.00 ** 2))))
    reg.add(Material("glass", ConstantOptical(complex(1.45 ** 2))))
    reg.add(Material("film", film_optical))
    return reg


def _film_grid(nx=6, nz=41, t=6e-9):
    x = np.linspace(0.0, 300e-9, nx)
    z = np.linspace(0.0, t, nz)
    prof = 3e26 + 8e26 * np.exp(-(t - z) / 0.8e-9)              # accumulation at the top face
    return x, z, np.tile(prof, (nx, 1))                          # (Nx, Nz), laterally uniform


def _field(n2, x, z):
    from dynameta.core.carrier_field import CarrierField, CarrierRegion, ELECTRON_DENSITY
    reg = CarrierRegion(name="film", role="semiconductor", material="film", nodes_m=np.zeros((1, 2)),
                        node_fields={}, grid_axes_m={"x": x, "y": z},
                        grid_fields={ELECTRON_DENSITY: n2})
    return CarrierField(bias_label="b", voltages={}, ndim=2, temperature_K=300.0, regions={"film": reg},
                        n_bg_by_region={"film": 3e26}, unit_cell_m=(300e-9, 300e-9))


def _alignment(z, zlo=6e-9):
    from dynameta.core import NM
    from dynameta.core.alignment import GeometryAlignment, RegionAlignment
    t = float(z[-1] - z[0])
    return GeometryAlignment(unit_scale=NM, region_alignments=[RegionAlignment(
        "film", "film", (0.0, 300e-9, 0.0, 300e-9, zlo, zlo + t), stack_axis="y")],
        fixed_eps_regions={"bot": "ox", "top": "ox"})


def _design(reg, t=6e-9):
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    return Design(name="p17", unit_cell=UnitCell.square(300e-9),
                  stack=Stack(layers=[Layer("bot", 6e-9, "ox"), Layer("film", t, "film"),
                                      Layer("top", 6e-9, "ox")],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="p", incidence_angle_deg=60.0))


def _assembled(optical, n_to_eps=None):
    from dynameta.core import MaterialEpsMap, assemble_eps
    from dynameta.core.lift import ExtrudeLift
    reg = _registry(optical)
    x, z, n2 = _film_grid()
    out = assemble_eps(_field(n2, x, z), _alignment(z), n_to_eps or MaterialEpsMap(reg),
                       ExtrudeLift(period_y_m=300e-9), LAM)
    return reg, out, x, z, n2


def test_assembler_carries_the_tensor_and_the_stack_axis():
    bnd = BoundaryScattering(specularity=0.2, perp_coefficient=0.6, perp_length="local")
    opt = DrudeOptical(4.0, M, 1.2e14, boundary=bnd)
    reg, out, x, z, n2 = _assembled(opt)
    ef = out["film"]
    assert ef.is_tensor and ef.values_zyx.shape == (z.size, 2, x.size, 3, 3)
    want = opt.eps_tensor(LAM, n_m3=n2, z_axis_m=z)              # (Nx, Nz, 3, 3), d = the region extent
    got = np.transpose(ef.values_zyx[:, 0], (1, 0, 2, 3))
    assert np.array_equal(got, want)
    assert not np.any(got[..., 0, 1]) and not np.any(got[..., 2, 0])
    assert np.any(got[..., 0, 0] != got[..., 2, 2])
    # the reserved key cannot be overridden by extra_fields for a material that uses it
    from dynameta.core import MaterialEpsMap, assemble_eps
    from dynameta.core.lift import ExtrudeLift
    with pytest.raises(ValueError, match="reserved"):
        assemble_eps(_field(n2, x, z), _alignment(z), MaterialEpsMap(reg), ExtrudeLift(period_y_m=300e-9),
                     LAM, extra_fields={STACK_AXIS_FIELD: z})


def test_byo_map_without_registry_gets_a_clear_error():
    opt = DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering(perp_coefficient=0.5))
    reg = _registry(opt)

    class _Byo:
        def eps_grid(self, name, fields, lam):
            from dynameta.core.effects import OpticalModelEffect
            return OpticalModelEffect(reg.get(name).optical).eps(fields, lam)

        def scalar_eps(self, name, lam):
            return complex(reg.get(name).eps(lam))
    with pytest.raises(ValueError, match="thickness is unknown"):
        _assembled(opt, n_to_eps=_Byo())


def _uniaxial_reference_R(design_eps_film_cells, dz, lam):
    """Independent planar p-pol solve of air | ox 7 nm | film cells | ox 7 nm | glass: the uniaxial 2x2
    matrices of optics.quantum_film (a different code path from the Lumenairy Berreman engine)."""
    from dynameta.optics import nonlocal_tmm as nt
    from dynameta.optics.quantum_film import film_stack_rt, uniaxial_layer_matrix
    w = 2.0 * math.pi * C_LIGHT / lam

    def fm(ww, kp):
        T = np.eye(2, dtype=complex)
        for exx, ezz in design_eps_film_cells:                  # superstrate side first
            T = uniaxial_layer_matrix(exx, ezz, dz, complex(ww) / C_LIGHT, kp) @ T
        return T
    return film_stack_rt(w, fm, [nt.DielectricLayer(2.00 ** 2, 6e-9)], [nt.DielectricLayer(2.00 ** 2, 6e-9)],
                         n_sub=1.45, theta_rad=math.radians(60.0)).R


def test_routes_tensor_supported_or_refused():
    """The scalar layered TMM refuses the tensor with its existing message; the Berreman bridge solves it
    and agrees with an independent uniaxial transfer-matrix solve of the same slabs (cross-route; the
    in-plane / normal components enter p-polarization differently, so a swapped component fails)."""
    pytest.importorskip("lumenairy")
    from dynameta.core.layered import slice_layer_eps_field
    from dynameta.optics.lumenairy_bridge import make_lumenairy_berreman_solver
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    bnd = BoundaryScattering(specularity=0.0, perp_coefficient=1.0, perp_length="film")
    reg, out, x, z, n2 = _assembled(DrudeOptical(4.0, M, 1.2e14, boundary=bnd))
    design = _design(reg)
    with pytest.raises(ValueError, match="TENSOR"):
        make_layered_tmm_solver()(design, None, out, LAM, 1.0 + 0j, 1.45 + 0j)
    rb = make_lumenairy_berreman_solver()(design, None, out, LAM, 1.0 + 0j, 1.45 + 0j)
    slabs = list(reversed(slice_layer_eps_field(out["film"], design.stack.layers[1], 1e-9)))
    cells = [(complex(s.eps_tensor_cell[0, 0, 0, 0]), complex(s.eps_tensor_cell[0, 0, 2, 2])) for s in slabs]
    dz = float(slabs[0].thickness_m)
    assert all(abs(float(s.thickness_m) - dz) < 1e-15 for s in slabs)
    R_ref = _uniaxial_reference_R(cells, dz, LAM)
    R_swap = _uniaxial_reference_R([(c[1], c[0]) for c in cells], dz, LAM)
    assert abs(rb.R - R_ref) < 1e-9
    # the agreement resolves which component is which: swapping eps_xx / eps_zz moves R 1000x more
    # (MEASURED: |R_berreman - R_ref| = 1.7e-17, |R_swap - R_ref| = 9.1e-6)
    assert abs(R_swap - R_ref) > 1e3 * abs(rb.R - R_ref)
    # the RCWA bridge takes the graded tensor too (MEASURED 1.6e-17 from Berreman); since v0.16 so does
    # the PMM bridge (a laterally uniform gridded tensor: one tensor PMM layer per slab), which used to refuse
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver, make_lumenairy_rcwa_solver
    rr = make_lumenairy_rcwa_solver(n_orders=1)(design, None, out, LAM, 1.0 + 0j, 1.45 + 0j)
    assert abs(rr.R - rb.R) < 1e-9
    rp = make_lumenairy_pmm_solver(n_orders=1)(design, None, out, LAM, 1.0 + 0j, 1.45 + 0j)
    assert abs(rp.R - rb.R) < 1e-9


def test_cache_fingerprint_changes_with_the_options():
    from dynameta.cache import _key, _materials_fingerprint
    base = DrudeOptical(4.0, M, 1.2e14)
    variants = [base,
                DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering(thickness_m=6e-9)),
                DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering(thickness_m=6e-9, specularity=0.5)),
                DrudeOptical(4.0, M, 1.2e14, boundary=BoundaryScattering(thickness_m=6e-9, perp_coefficient=1.0)),
                DrudeOptical(4.0, M, 1.2e14, background=BandFillingBackground(3.75, 2.0, 3e26, M)),
                DrudeOptical(4.0, M, 1.2e14, background=BandFillingBackground(3.75, 2.5, 3e26, M)),
                DrudeOptical(4.0, M, 1.2e14, background=TabulatedBackground((1e26, 1e27), (4.0, 3.8)))]
    fps = [_materials_fingerprint(_design(_registry(v)), LAM) for v in variants]
    assert len(set(fps)) == len(fps)
    keys = {_key(_design(_registry(v)), {}, LAM, 1.0, 1.45, "") for v in variants}
    assert len(keys) == len(variants)
    # the default model keeps the key it had (the repr, hence the fingerprint, is the v0.15.1 one)
    assert _materials_fingerprint(_design(_registry(base)), LAM) == _materials_fingerprint(
        _design(_registry(DrudeOptical(4.0, M, 1.2e14))), LAM)


# ------------------------------------------------------------------------------------------------
# background permittivity
# ------------------------------------------------------------------------------------------------
def _chi_quad(A, Eg0, Eg, E, E_lo):
    """(2/pi) int_{E_lo}^inf E' eps2(E') / (E'^2 - E^2) dE', eps2 = A sqrt((E'-Eg)/Eg0) (Eg0/E')^2; all
    energies in eV (scipy's absolute tolerance is then meaningful)."""
    from scipy.integrate import quad
    f = lambda e: e * A * math.sqrt(max(e - Eg, 0.0) / Eg0) * (Eg0 / e) ** 2 / (e * e - E * E)  # noqa: E731
    return (2.0 / math.pi) * (quad(f, E_lo, 50 * Eg, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
                              + quad(f, 50 * Eg, np.inf, limit=200, epsabs=1e-14)[0])


@pytest.mark.parametrize("cond", ["parabolic", "kane"])
def test_band_filling_background_is_the_kk_transform_of_the_blocked_edge(cond):
    conduction = M if cond == "parabolic" else KaneOpticalMass(m0_kg=0.22 * M_E, alpha_eV=0.4)
    bg = BandFillingBackground(gap_eV=3.75, edge_amplitude=2.5, n_ref_m3=1e26, conduction=conduction,
                               valence_mass_kg=0.6 * M_E, bgr_coeff_eV_m=2e-10)
    for n in (1e25, 3e26, 1.2e27):
        for lam in (2000e-9, 1200e-9, 600e-9):
            E = 2 * math.pi * HBAR * C_LIGHT / lam
            Eg = (3.75 - 2e-10 * n ** (1 / 3)) * Q_E
            kF = (3 * math.pi ** 2 * n) ** (1 / 3)
            if cond == "parabolic":
                EF = HBAR ** 2 * kF ** 2 / (2 * M)
            else:
                g = HBAR ** 2 * kF ** 2 / (2 * 0.22 * M_E)
                a = 0.4 / Q_E
                EF = (-1 + math.sqrt(1 + 4 * a * g)) / (2 * a)
            dBM = EF + HBAR ** 2 * kF ** 2 / (2 * 0.6 * M_E)
            want = _chi_quad(2.5, 3.75, Eg / Q_E, E / Q_E, (Eg + dBM) / Q_E)
            got = float(bg.interband_chi(n, lam))
            assert abs(got - want) < 1e-9 * max(abs(want), 1e-3), (n, lam, got, want)


def test_background_anchoring_limits_and_sign():
    bg = BandFillingBackground(3.75, 2.5, 3e26, M, valence_mass_kg=0.6 * M_E)
    n = np.array([1e24, 1e26, 3e26, 8e26, 1.6e27])
    e = bg.eps_background(n, LAM, 4.0)
    assert e[2] == 4.0                                          # exactly the anchor
    assert np.all(np.diff(e) < 0.0)                             # band filling lowers the background
    assert np.array_equal(BandFillingBackground(3.75, 0.0, 3e26, M).eps_background(n, LAM, 4.0),
                          np.full(n.size, 4.0))
    # gap renormalization moves the whole edge down: at the same filling it RAISES the background
    assert (BandFillingBackground(3.75, 2.5, 0.0, M, bgr_coeff_eV_m=3e-10).interband_chi(1e27, LAM)
            > BandFillingBackground(3.75, 2.5, 0.0, M).interband_chi(1e27, LAM))
    # optical gap = Eg0 - BGR + BM
    assert bg.optical_gap_J(0.0) == 3.75 * Q_E
    with pytest.raises(ValueError, match="below the gap"):
        bg.eps_background(3e26, 300e-9, 4.0)
    d = DrudeOptical(4.0, M, 1.2e14, background=bg)
    assert d.eps(LAM, n_m3=3e26) == DrudeOptical(4.0, M, 1.2e14).eps(LAM, n_m3=3e26)
    x = d.eps(LAM, n_m3=n)
    assert np.allclose(x - DrudeOptical(4.0, M, 1.2e14).eps(LAM, n_m3=n), e - 4.0, rtol=0, atol=1e-14)


def test_tabulated_and_callable_backgrounds():
    tb = TabulatedBackground(n_m3=(1e25, 1e26, 1e27), eps_inf=(4.05, 3.95, 3.70))
    n = np.array([1e25, 10 ** 25.5, 1e26, 10 ** 26.25, 1e27])
    want = np.interp(np.log10(n), [25, 26, 27], [4.05, 3.95, 3.70])
    assert np.allclose(tb.eps_background(n, LAM, 0.0), want, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="outside the table"):
        tb.eps_background(1e24, LAM, 0.0)
    hold = TabulatedBackground(n_m3=(1e25, 1e27), eps_inf=(4.05, 3.70), extrapolate="hold")
    assert hold.eps_background(np.array([0.0, 1e28]), LAM, 0.0).tolist() == [4.05, 3.70]
    d = DrudeOptical(9.9, M, 1.2e14, background=tb)               # eps_inf unused by an absolute table
    assert np.allclose(d.eps(LAM, n_m3=n) - DrudeOptical(0.0, M, 1.2e14).eps(LAM, n_m3=n), want, atol=1e-14)
    dc = DrudeOptical(4.0, M, 1.2e14, background=lambda nn, lam: 4.2 - 1e-28 * np.asarray(nn))
    assert np.allclose(dc.eps(LAM, n_m3=n) - DrudeOptical(0.0, M, 1.2e14).eps(LAM, n_m3=n), 4.2 - 1e-28 * n,
                       rtol=0.0, atol=1e-14)
    with pytest.raises(ValueError, match="Im < 0"):
        DrudeOptical(4.0, M, 1.2e14, background=lambda nn, lam: 4.0 - 0.1j).eps(LAM, n_m3=n)
    with pytest.raises(TypeError):
        DrudeOptical(4.0, M, 1.2e14, background=4.0)


def test_background_flows_through_the_assembler_and_shifts_the_zero_crossing():
    """The background enters every component of the assembled field; the density where Re(eps) = 0 at
    1200 nm moves DOWN when band filling lowers eps_b (sign-sensitive)."""
    from scipy.optimize import brentq
    bg = BandFillingBackground(3.75, 2.5, 3e26, M, valence_mass_kg=0.6 * M_E)
    opt = DrudeOptical(4.0, M, 1.2e14, background=bg)
    reg, out, x, z, n2 = _assembled(opt)
    v = np.transpose(out["film"].values_zyx[:, 0], (1, 0))
    assert np.array_equal(v, opt.eps(LAM, n_m3=n2))
    nc0 = brentq(lambda nn: DrudeOptical(4.0, M, 1.2e14).eps(LAM, n_m3=nn).real, 1e26, 1e28, xtol=1e18)
    nc1 = brentq(lambda nn: opt.eps(LAM, n_m3=nn).real, 1e26, 1e28, xtol=1e18)
    assert nc1 < nc0
