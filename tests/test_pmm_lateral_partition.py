"""PMM bridge: laterally structured gridded fields, z-graded films, pmm_kwargs (v0.16).

The PMM bridge refused a laterally structured tunable layer; PMMStack's convergence controls were
unreachable (the bridge passed the `n_orders=` alias); the bridge sliced gridded fields without the
slab-thickness guard.

ON v0.15.1 (commit ebdfe5c, lumenairy 5.50.x) every bridge-level gate below that uses a new keyword
fails with TypeError (unexpected keyword), every laterally structured field raised "carries a
laterally-structured gridded EpsField", the metre-axis field returned r = -0.849411+0.035236i with
no error (degree 24; the nm-axis field gives -0.234953-0.460540i), and the kernel module
`_lateral_partition` did not exist (ImportError). The bit-identity gates are a same-tree A/B check:
they rebuild the v0.15.1 stack inline and compare with np.array_equal.

Fixture: the generic strip cell (as tests/_strip_cell_fixture): a 400 nm cell, 40 nm metal strip
(150 nm, eps -60+4i) over a 2 nm oxA / 6 nm oxB / 6 nm film / 6 nm oxB / 2 nm oxA spacer on an 80 nm
mirror metal (eps -120+20i), glass substrate (n 1.45), 1200 nm, TM (x) at normal incidence; the film
is a Drude conductor whose density is graded in z (and, for the structured cases, differs under and
between the strips).
"""
import importlib.util
import warnings

import numpy as np
import pytest

from dynameta.core.eps_field import EpsField
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import M_E, ConstantOptical, DrudeOptical, Material, MaterialRegistry
from dynameta.optics.lumenairy_bridge import _lateral_partition as lp
from dynameta.warning_types import AdvisoryWarning

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
HAVE_NGS = importlib.util.find_spec("ngsolve") is not None
needs_lum = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")
needs_fem = pytest.mark.skipif(not (HAVE_LUM and HAVE_NGS), reason="lumenairy + ngsolve needed")

LAM, P, W, N_BG = 1200e-9, 400e-9, 150e-9, 3.0e26
T_FILM = 6.0                                    # nm
PN, WN = P * 1e9, W * 1e9                       # nm
XE1, XE2 = (PN - WN) / 2, (PN + WN) / 2         # strip edges [nm]: 125, 275
DRUDE = DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)
FILM = [("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"), ("lo_oxb", 6e-9, "oxB"),
        ("film", 6e-9, "TCO"), ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA")]


def _mats():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(1.45 ** 2 + 0j)))
    reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j), eps_static_dc=9.0))
    reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j), eps_static_dc=20.0))
    reg.add(Material("metal", ConstantOptical(-60.0 + 4.0j), is_metal=True))
    reg.add(Material("mirror_metal", ConstantOptical(-120.0 + 20.0j), is_metal=True))
    reg.add(Material("TCO", optical=DRUDE))
    return reg


def eps_of(n):
    return np.asarray(DRUDE.eps(LAM, n_m3=np.asarray(n, float)), dtype=complex)


def strip_design(pol="x", azimuth=0.0, theta=0.0):
    layers = [Layer(n, t, m) for n, t, m in FILM]
    layers.append(Layer("grating", 40e-9, "air",
                        inclusions=[Inclusion(Rectangle(P / 2, P / 2, W, P), "metal")]))
    return Design(name="strip", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=_mats(),
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta,
                                      azimuth_deg=azimuth))


def density(z_nm, n_top, Ld=1.0, t=T_FILM):
    """Accumulation at the top face of the 6 nm film."""
    return N_BG * (1.0 + n_top * np.exp(-(t - np.asarray(z_nm, float)) / Ld))


def uniform_field(z_nm, n_top=0.8, Ld=1.0):
    e = eps_of(density(z_nm, n_top, Ld))
    return EpsField(x_axis_u=np.array([0.0, PN]), y_axis_u=np.array([0.0]),
                    z_axis_u=np.asarray(z_nm, float),
                    values_zyx=np.repeat(e[:, None, None], 2, axis=2))


def ramp_s(x_nm, ramp_nm):
    """1 under the strip, 0 between strips, linear ramps of width ramp_nm centred on the edges."""
    d = np.abs(np.asarray(x_nm, float) - PN / 2) - WN / 2
    return np.clip(0.5 - d / ramp_nm, 0.0, 1.0)


def structured_field(z_nm, x_nm, s, n_top_under=0.8, n_top_between=0.1, Ld=1.0):
    """Two different z profiles under (s = 1) and between (s = 0) the strips, mixed by s(x)."""
    nA = density(z_nm, n_top_under, Ld)
    nB = density(z_nm, n_top_between, Ld)
    n = nB[:, None] + (nA - nB)[:, None] * np.asarray(s, float)[None, :]
    return EpsField(x_axis_u=np.asarray(x_nm, float), y_axis_u=np.array([0.0]),
                    z_axis_u=np.asarray(z_nm, float), values_zyx=eps_of(n)[:, None, :])


def _solve(design, ebr, **kw):
    from dynameta.optics.lumenairy_bridge.pmm_backend import make_lumenairy_pmm_solver
    return make_lumenairy_pmm_solver(**kw)(design, None, ebr, LAM, None, None)


# =============================================================================================
# kernels (no lumenairy)
# =============================================================================================
def test_harmonic_piece_matches_quadrature():
    """int_0^1 dt / (a + (b - a) t) in closed form vs scipy quadrature written here, incl. a
    lossy pass near zero, a metal/dielectric pair and the Taylor branch."""
    from scipy.integrate import quad
    pairs = [(2 + 0.1j, 1.3 + 0.2j), (0.05 + 0.3j, -1.2 + 0.4j), (-60 + 4j, 1 + 0j),
             (2.0 + 0j, 2.0 + 1e-9j), (1 + 1j, 1 + 1j), (3 + 0.01j, 3.002 + 0.01j)]
    for a, b in pairs:
        f = lambda t: 1.0 / (a + (b - a) * t)             # noqa: E731
        re = quad(lambda t: f(t).real, 0, 1, epsabs=1e-15, epsrel=1e-13, limit=200)[0]
        im = quad(lambda t: f(t).imag, 0, 1, epsabs=1e-15, epsrel=1e-13, limit=200)[0]
        got = complex(lp._harmonic_piece(np.array([a]), np.array([b]))[0])
        assert abs(got - (re + 1j * im)) <= 1e-12 * abs(re + 1j * im), (a, b)


def test_harmonic_piece_refuses_lossless_sign_change():
    with pytest.raises(ValueError, match="passes through zero"):
        lp._harmonic_piece(np.array([2.0 + 0j]), np.array([-1.0 + 0j]))
    with pytest.raises(ValueError, match="exactly zero"):
        lp._harmonic_piece(np.array([0j]), np.array([1.0 + 0j]))


def test_laminate_exact_on_resolved_piecewise_linear_profile():
    """Constant intervals give their value, a ramp interval gives (a+b)/2 and the closed-form
    harmonic mean (b - a)/log(b/a) of the linear ramp. Constant intervals to 1e-14 relative
    (summation round-off over 169 sample pieces, measured 4.4e-15); ramp intervals 5e-14 (the edge fraction -> metre round trip moves a ramp end by ~1e-22 m off
    its sample, measured 1.1e-14)."""
    a, b = 2.2 + 0.1j, 1.3 + 0.2j
    x = np.array([0.0, 100.0, 101.0, 270.0, 271.0]) * 1e-9          # endpoint-exclusive
    v = np.array([a, a, b, b, a])
    edges = np.array([0.0, 100.0, 101.0, 270.0, 271.0, 400.0]) / 400.0
    w, comps = lp.laminate_segments(v[None, :], x, P, edges, "laminate")
    h_ramp = (b - a) / np.log(b / a)                                 # harmonic mean of the ramp
    expect_xx = [a, h_ramp, b, h_ramp, a]
    expect_ar = [a, 0.5 * (a + b), b, 0.5 * (a + b), a]
    assert np.allclose(w, np.diff(edges), rtol=0, atol=1e-16)
    for k in range(5):
        tol = 1e-14 if k % 2 == 0 else 5e-14
        assert abs(comps[0, k, 0] - expect_xx[k]) <= tol * abs(expect_xx[k])
        assert abs(comps[0, k, 1] - expect_ar[k]) <= tol * abs(expect_ar[k])
        assert comps[0, k, 2] == comps[0, k, 1]
    _, comps_a = lp.laminate_segments(v[None, :], x, P, edges, "arithmetic")
    assert np.array_equal(comps_a[..., 0], comps_a[..., 1])


def test_grid_conventions_describe_one_interpolant():
    """The same periodic piecewise-linear profile (kinks at P/4 and 3P/4, linear across x = 0)
    sampled endpoint-inclusive, endpoint-exclusive and cell-centred gives identical means."""
    A, B = 2.0 + 0.1j, 0.9 + 0.3j

    def f(xn):
        u = np.mod(np.asarray(xn) / PN - 0.25, 1.0)                   # 0 at P/4
        return np.where(u <= 0.5, A + (B - A) * u / 0.5, B + (A - B) * (u - 0.5) / 0.5)

    grids = [np.linspace(0.0, PN, 9), np.arange(8) * PN / 8, (np.arange(10) + 0.5) * PN / 10]
    edges = np.array([0.0, 0.1, 0.25, 0.4, 0.75, 0.9, 1.0])
    out = []
    for g in grids:
        xs, vs = lp.periodic_samples(g * 1e-9, f(g)[None, :], P)
        out.append(lp.interval_means(xs, vs, edges, P)[:2])
    for ar, hm in out[1:]:
        assert np.allclose(ar, out[0][0], rtol=1e-14, atol=0)
        assert np.allclose(hm, out[0][1], rtol=1e-14, atol=0)


def test_periodic_samples_refuses_bad_axes():
    with pytest.raises(ValueError, match="outside the cell"):
        lp.periodic_samples(np.array([0.0, 450e-9]), np.ones(2), P)      # past the 400 nm cell
    with pytest.raises(ValueError, match="strictly increasing"):
        lp.periodic_samples(np.array([0.0, 0.0, 1e-9]), np.ones(3), P)
    with pytest.raises(ValueError, match="< half"):
        lp.periodic_samples(np.array([0.0, 100e-9]), np.ones(2), P)


def test_partition_isolates_a_ramp_and_reports():
    """A lateral ramp inside a seeded interval: 'balanced' cuts land on the ramp ends, the report
    is truthful, refine=False keeps the given edges, and a tight cap reports capped."""
    x = np.arange(400) * 1e-9
    s = ramp_s(np.arange(400), 20.0)
    prof = (2.0 + 0.1j) + (-1.0 + 0.1j) * s[None, :]                  # 2.0 -> 1.0 under strip
    seeds = [XE1 / PN, XE2 / PN]
    e_auto, rep = lp.partition_edges(prof, x, P, seeds, rtol=0.1, max_segments=64)
    assert rep["converged"] and not rep["capped"] and rep["max_spread"] <= 0.1
    assert rep["n_segments"] == e_auto.size - 1
    # every cut sits on a sample inside a ramp ([115, 135] and [265, 285] nm): the flat
    # stretches are never cut (measured cuts: 118 121 125 128 131 134 | 266 268 271 275 279 282)
    cuts = e_auto[1:-1] * PN
    assert all((115.0 <= c <= 135.0) or (265.0 <= c <= 285.0) for c in cuts)
    e_bis, rep_b = lp.partition_edges(prof, x, P, seeds, rtol=0.1, max_segments=64, split="bisect")
    assert rep_b["converged"] and rep_b["n_segments"] > rep["n_segments"]
    e_fix, rep_f = lp.partition_edges(prof, x, P, seeds, refine=False)
    assert np.allclose(e_fix, [0.0, XE1 / PN, XE2 / PN, 1.0], atol=0) and not rep_f["converged"]
    _, rep_c = lp.partition_edges(prof, x, P, seeds, rtol=1e-3, max_segments=6)
    assert rep_c["capped"] and rep_c["n_segments"] == 6 and rep_c["max_spread"] > 1e-3


def test_z_interval_harmonic_and_laminate_z():
    """Per-interval harmonic z means in closed form; z-laminate merge = thickness-weighted
    arithmetic (xx, yy) and harmonic (zz); greedy groups respect the tolerance."""
    z = np.array([0.0, 1.0, 3.0])
    v = np.array([2.0 + 0.1j, 1.0 + 0.2j, 1.0 + 0.2j])
    h = lp.z_interval_harmonic(v, z)
    assert abs(h[0] - (v[1] - v[0]) / np.log(v[1] / v[0])) < 1e-15
    assert abs(h[1] - v[1]) < 1e-15
    assert np.allclose(lp.z_interval_harmonic(v[::-1], z[::-1]), h, rtol=1e-15, atol=0)
    c = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=complex)
    t = np.array([1.0, 3.0])
    tt, cc = lp.laminate_z(c, t, [(0, 2)])
    assert tt[0] == 4.0
    assert abs(cc[0, 0] - 2.5) < 1e-15 and abs(cc[0, 1] - 2.5) < 1e-15
    assert abs(cc[0, 2] - 4.0 / (1.0 + 1.0)) < 1e-15                 # 4 / (1/1 + 3/3)
    comps = np.array([[1.0], [1.005], [1.01], [2.0]], dtype=complex)
    assert lp.z_merge_groups(comps, np.ones(4), 1e-2) == [(0, 3), (3, 4)]
    assert lp.z_merge_groups(comps, np.ones(4), None) == [(0, 1), (1, 2), (2, 3), (3, 4)]


# =============================================================================================
# bit-identity with v0.15.1 for every input the old bridge accepted (same-tree A/B)
# =============================================================================================
def _legacy_stack(design, ebr, *, degree=16, n_orders=21, n_slices=None):
    """design_to_pmm_stack as shipped in v0.15.1 (ebdfe5c), rebuilt here."""
    import lumenairy as lum
    from dynameta.core.layered import collapse_regions_to_layers, slice_eps_field
    from dynameta.optics.lumenairy_bridge.pmm_backend import layer_to_pmm_segments
    from dynameta.optics.tmm_reference import end_media_indices
    n_super, n_sub = end_media_indices(design, LAM)
    px, py = float(design.unit_cell.period_x_m), float(design.unit_cell.period_y_m)
    ebr = collapse_regions_to_layers(design, ebr or {})
    st = lum.PMMStack(px, n_substrate=complex(n_sub), n_superstrate=complex(n_super),
                      degree=int(degree), n_orders=int(n_orders))
    for L in reversed(design.stack.layers):
        ef = ebr.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            for slab in reversed(slice_eps_field(ef, 1e-9, n_slices=n_slices)):
                st.add_layer(slab.thickness_m, eps=complex(slab.eps))
            continue
        if ef is not None and getattr(ef, "is_tensor", False):
            st.add_layer(L.thickness_m, eps=np.asarray(ef.tensor, dtype=complex))
            continue
        if L.inclusions:
            st.add_layer(L.thickness_m, segments=layer_to_pmm_segments(
                L, design, LAM, px, py, bg_eps=(ef.scalar if ef is not None else None)))
            continue
        st.add_layer(L.thickness_m, eps=complex(ef.scalar) if ef is not None else
                     complex(design.materials.get(L.background_material).eps(LAM)))
    return st


def _solved(st, theta=0.0, phi=0.0):
    st.set_source(LAM, theta=theta, phi=phi)
    orders, R, T, J = st.solve()
    return [np.asarray(orders), np.asarray(R), np.asarray(T), np.asarray(J),
            np.asarray(st.jones_transmission())]


ZN11 = np.linspace(0.0, T_FILM, 11)
ZNU = np.array([0.0, 0.05, 0.15, 0.4, 1.0, 3.0, 5.0, 5.6, 5.85, 5.95, 6.0])   # non-uniform
OLD_INPUTS = {
    "uniform_scalar": {"film": EpsField(scalar=complex(eps_of(N_BG)))},
    "uniform_tensor": {"film": EpsField(tensor=np.diag([2.1 + 0.1j, 2.2 + 0.1j, 1.9 + 0.12j]))},
    "graded_uniform": {"film": uniform_field(ZN11, n_top=3.0, Ld=0.6)},
    "graded_nonuniform_z": {"film": uniform_field(ZNU, n_top=3.0, Ld=0.6)},
    "graded_descending_z": {"film": uniform_field(ZN11[::-1], n_top=3.0, Ld=0.6)},
}


@needs_lum
@pytest.mark.parametrize("case", sorted(OLD_INPUTS))
@pytest.mark.parametrize("n_slices", [None, 7])
def test_default_stack_bit_identical_to_v0151(case, n_slices):
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    d = strip_design()
    ebr = OLD_INPUTS[case]
    new, _ = design_to_pmm_stack(d, LAM, eps_by_region=ebr, degree=8, n_orders=9,
                                 n_slices=n_slices)
    old = _legacy_stack(d, ebr, degree=8, n_orders=9, n_slices=n_slices)
    for a, b in zip(_solved(new), _solved(old)):
        assert np.array_equal(a, b)
    assert new.dynameta_partition_report == {}


@needs_lum
def test_default_solver_pin_graded_field_v0151():
    """The v0.15.1 PMM-bridge reading of the graded strip cell (degree 24; recorded with the v0.15.1
    code on lumenairy 5.50.2, where this code gives the same r bit for bit) is reproduced, and the
    conical and absorption paths match the legacy stack bit for bit."""
    z = np.linspace(0.0, T_FILM, 11)
    eg = np.asarray([eps_of(N_BG * (1 + 3 * np.exp(-(T_FILM - q) / 0.6))) for q in z], complex)
    ef = EpsField(x_axis_u=np.array([0.0, PN]), y_axis_u=np.array([0.0]), z_axis_u=z,
                  values_zyx=np.repeat(eg[:, None, None], 2, axis=2))
    res = _solve(strip_design(), {"film": ef}, degree=24)
    # v0.15.1 code and this code on 5.50.2: -0.2349526063665708-0.4605400238084049j (bit-identical);
    # the same on 5.22.0, 7.5e-14 away on 5.51.0 (its uniform-layer fast path; bar 1e-11)
    assert abs(res.r - (-0.2349526063665708 - 0.4605400238084049j)) < 1e-11
    a = _solve(strip_design(pol="y", theta=10.0, azimuth=30.0), {"film": ef}, degree=8,
               n_orders=9)
    old = _legacy_stack(strip_design(), {"film": ef}, degree=8, n_orders=9)
    from dynameta.optics.lumenairy_bridge._common import conical_synthesis
    old.set_source(LAM, theta=np.radians(10.0), phi=np.radians(30.0))
    old.solve()
    R, T, r, t = conical_synthesis(old, "y", np.radians(10.0), np.radians(30.0), 1.0 + 0j)
    assert (a.R, a.T, a.r, a.t) == (R, T, r, t)
    b = _solve(strip_design(), {"film": ef}, degree=8, n_orders=9, absorption=True)
    old = _legacy_stack(strip_design(), {"film": ef}, degree=8, n_orders=9)
    old.set_source(LAM, theta=0.0, phi=0.0)
    old.solve(retain_internal=True)
    la = np.asarray(old.layer_absorption())[:, 0]
    assert b.A_independent == float(np.sum(la))
    assert b.per_region_absorption["film"] == float(sum(la[3:13]))     # 10 film slabs


# =============================================================================================
# the slab-thickness guard at this call site; native (non-uniform) z grids
# =============================================================================================
@needs_lum
def test_wrong_units_field_raises_through_guard():
    """v0.15.1: the metre-axis field returned r = -0.849411+0.035236i (degree 24) silently. The
    lateral analogue (x axis in metres on a structured field) is refused by the partition: a
    periodic sampling of >= 2 points spans at least half the period."""
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    x = np.arange(64) * PN / 64
    for ef in (uniform_field(ZN11 * 1e-9),
               structured_field(ZN11 * 1e-9, x, ramp_s(x, 20.0))):
        with pytest.raises(ValueError, match=r"slice_layer_eps_field: layer 'film'.*METRES"):
            design_to_pmm_stack(strip_design(), LAM, eps_by_region={"film": ef}, degree=6)
    x_metres = structured_field(ZN11, x * 1e-9, ramp_s(x, 20.0))      # x axis in metres
    with pytest.raises(ValueError, match=r"layer 'film'.*METRES"):
        design_to_pmm_stack(strip_design(), LAM, eps_by_region={"film": x_metres}, degree=6)
    one_z = structured_field(np.array([0.0]), x, ramp_s(x, 20.0))
    with pytest.raises(ValueError, match="1 z sample"):
        design_to_pmm_stack(strip_design(), LAM, eps_by_region={"film": one_z}, degree=6)


@needs_lum
def test_native_carrier_grid_flag_and_nonuniform_z_consumption():
    """The solver sets native_carrier_grid; a non-uniform z field becomes one PMM layer per native
    interval (thickness = the interval, eps = its endpoint average) -- the hand-built stack, bit
    for bit -- and agrees with a uniform 4x finer sampling of the same profile (staircase
    convergence, not equality)."""
    import lumenairy as lum
    from dynameta.optics.lumenairy_bridge.pmm_backend import (design_to_pmm_stack,
                                                              make_lumenairy_pmm_solver)
    assert make_lumenairy_pmm_solver().native_carrier_grid is True
    d = strip_design()
    znat = T_FILM * (1.0 - (1.0 - np.linspace(0.0, 1.0, 16)) ** 2)  # fine toward the top face
    ef = uniform_field(znat, n_top=0.8)
    st, names = design_to_pmm_stack(d, LAM, eps_by_region={"film": ef}, degree=8, n_orders=9)
    assert names.count("film") == znat.size - 1
    hand = lum.PMMStack(P, n_substrate=1.45 + 0j, n_superstrate=1.0 + 0j, degree=8,
                        far_field_orders=9)
    hand.add_layer(40e-9, segments=[(0.5 - W / P / 2, 1.0 + 0j), (W / P, -60 + 4j),
                                    (0.5 - W / P / 2, 1.0 + 0j)])
    hand.add_layer(2e-9, eps=1.70 ** 2 + 0j)
    hand.add_layer(6e-9, eps=2.00 ** 2 + 0j)
    e = eps_of(density(znat, 0.8))
    zm = znat * 1e-9
    for k in range(znat.size - 2, -1, -1):
        hand.add_layer(float(zm[k + 1] - zm[k]), eps=complex(0.5 * (e[k] + e[k + 1])))
    hand.add_layer(6e-9, eps=2.00 ** 2 + 0j)
    hand.add_layer(2e-9, eps=1.70 ** 2 + 0j)
    hand.add_layer(80e-9, eps=-120 + 20j)
    for a, b in zip(_solved(st), _solved(hand)):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-14)   # strip widths differ by ulps
    r_nu = _solve(d, {"film": ef}, degree=8, n_orders=9).r
    r_fine = _solve(d, {"film": uniform_field(np.linspace(0, T_FILM, 61), n_top=0.8)}, degree=8,
                    n_orders=9).r
    assert abs(r_nu - r_fine) < 2e-3            # measured 9.7e-4 (15 graded vs 60 uniform slabs)


# =============================================================================================
# laterally structured fields
# =============================================================================================
def _two_profile_ramp_field(delta_nm, n_side=6, z=ZN11):
    """Under-strip and between-strip profiles joined by linear ramps of width delta_nm centred on
    the strip edges; samples on the ramp ends, plus n_side samples per flat stretch."""
    xs = [np.linspace(0.0, XE1 - delta_nm / 2, n_side, endpoint=False),
          np.linspace(XE1 - delta_nm / 2, XE1 + delta_nm / 2, 2, endpoint=False),
          np.linspace(XE1 + delta_nm / 2, XE2 - delta_nm / 2, n_side, endpoint=False),
          np.linspace(XE2 - delta_nm / 2, XE2 + delta_nm / 2, 2, endpoint=False),
          np.linspace(XE2 + delta_nm / 2, PN, n_side, endpoint=False)]
    x = np.concatenate(xs)
    return structured_field(z, x, ramp_s(x, delta_nm)), x


def _hand_film_stack(film_layers, *, degree=8):
    """Hand-built lumenairy stack of the strip cell; film_layers = [(thickness, segments)] top first."""
    import lumenairy as lum
    st = lum.PMMStack(P, n_substrate=1.45 + 0j, n_superstrate=1.0 + 0j, degree=degree,
                      far_field_orders=9)
    g = 0.5 - W / P / 2
    st.add_layer(40e-9, segments=[(g, 1.0 + 0j), (W / P, -60 + 4j), (g, 1.0 + 0j)])
    st.add_layer(2e-9, eps=1.70 ** 2 + 0j)
    st.add_layer(6e-9, eps=2.00 ** 2 + 0j)
    for t, segs in film_layers:
        st.add_layer(t, segments=segs)
    st.add_layer(6e-9, eps=2.00 ** 2 + 0j)
    st.add_layer(2e-9, eps=1.70 ** 2 + 0j)
    st.add_layer(80e-9, eps=-120 + 20j)
    return st


@needs_lum
def test_explicit_partition_equals_hand_built_segment_stack():
    """A two-profile field with 2 nm ramps at the strip edges, partitioned on the ramp ends: the
    flat intervals carry their exact values and each ramp interval the closed-form laminate
    (arith of the ramp; harmonic (b - a)/log(b/a) for xx). Segment values agree with the
    hand-built ones to 4e-14 and widths to 1e-16; the solved stacks agree to 5e-9 relative,
    which is the PMM solve's own round-off amplification on this cell: perturbing the hand-built
    eps by 1e-14 relative (or one strip width by 2e-16) moves R by 2-3e-10 (measured), so a
    1e-12 agreement is not reachable for any construction (6 z slabs, arithmetic z)."""
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    z = np.linspace(0.0, T_FILM, 7)
    delta = 2.0
    ef, _ = _two_profile_ramp_field(delta, z=z)
    edges = np.array([XE1 - delta / 2, XE1 + delta / 2, XE2 - delta / 2, XE2 + delta / 2]) / PN
    st, names = design_to_pmm_stack(strip_design(), LAM, eps_by_region={"film": ef}, degree=8,
                                    n_orders=9, lateral_partition=edges, z_average="arithmetic",
                                    z_merge_rtol=None)
    assert names.count("film") == 6
    eA, eB = eps_of(density(z, 0.8)), eps_of(density(z, 0.1))
    wid = np.diff(np.concatenate([[0.0], edges, [1.0]]))
    film = []
    for k in range(5, -1, -1):
        a, b = 0.5 * (eB[k] + eB[k + 1]), 0.5 * (eA[k] + eA[k + 1])     # between / under
        ramp = np.diag([(b - a) / np.log(b / a), 0.5 * (a + b), 0.5 * (a + b)])
        film.append((1e-9, [(wid[0], a), (wid[1], ramp), (wid[2], b), (wid[3], ramp),
                            (wid[4], a)]))
    hand = _hand_film_stack(film)
    for a_, b_ in zip(_solved(st), _solved(hand)):
        assert np.allclose(a_, b_, rtol=5e-9, atol=1e-13)


@needs_lum
def test_sharp_step_limit_converges_linearly_in_ramp_width():
    """Two profiles joined by ramps of width delta at the strip edges vs the SHARP switch on the
    SAME walls (edges at the strip edge and the ramp ends, so both stacks share one union grid
    and only the ramp averaging differs): |r - r_sharp| falls at least linearly with delta.
    MEASURED (degree 8, 10 z slabs, arithmetic z): 4.04e-4 / 1.33e-4 / 4.20e-5 at delta 4 / 2 / 1 nm
    (ratios 3.05 / 3.15 per halving); asserted: each ratio > 1.6 (first order with margin) and
    4.2e-5 < 2e-4."""
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    z = ZN11
    eA, eB = eps_of(density(z, 0.8)), eps_of(density(z, 0.1))
    errs = []
    for delta in (4.0, 2.0, 1.0):
        edges = np.array([XE1 - delta / 2, XE1, XE1 + delta / 2,
                          XE2 - delta / 2, XE2, XE2 + delta / 2]) / PN
        ef, _ = _two_profile_ramp_field(delta)
        st, _ = design_to_pmm_stack(strip_design(), LAM, eps_by_region={"film": ef}, degree=8,
                                    n_orders=9, lateral_partition=edges,
                                    z_average="arithmetic", z_merge_rtol=None)
        wid = np.diff(np.concatenate([[0.0], edges, [1.0]]))
        film = []
        for k in range(9, -1, -1):
            a, b = 0.5 * (eB[k] + eB[k + 1]), 0.5 * (eA[k] + eA[k + 1])
            film.append((0.6e-9, [(wid[0], a), (wid[1], a), (wid[2], b), (wid[3], b),
                                  (wid[4], b), (wid[5], a), (wid[6], a)]))
        r_sharp = complex(_solved(_hand_film_stack(film))[3][0, 0])
        errs.append(abs(complex(_solved(st)[3][0, 0]) - r_sharp))
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[1] > 1.6 and errs[1] / errs[2] > 1.6      # at least first order
    assert errs[2] < 2e-4


def _planar_lamellar(pol, t_film_m, n_lamellae=16, contrast=0.8):
    """Mirror / 8 nm oxB / film / 8 nm oxB, the film's density alternating in n_lamellae
    lamellae (switches as 0.1 nm ramps, laterally uniform in z). Returns (design, field,
    resolved partition)."""
    film = [("mirror", 80e-9, "mirror_metal"), ("lo", 8e-9, "oxB"), ("film", t_film_m, "TCO"),
            ("up", 8e-9, "oxB")]
    d = Design(name="planar", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
               stack=Stack(layers=[Layer(n, t, m) for n, t, m in film],
                           superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=_mats(),
               optical=OpticalSpec(polarization=pol, incidence_angle_deg=0.0))
    z = np.linspace(0.0, t_film_m * 1e9, 3)
    sw = np.arange(1, n_lamellae) * PN / n_lamellae
    x = np.sort(np.concatenate([[0.0], sw - 0.05, sw + 0.05]))
    k = np.floor((x + 0.05) / (PN / n_lamellae)).astype(int)     # lamella index of each sample
    lam = (k % 2 == 1).astype(float)
    lam[np.isin(x, sw - 0.05)] = (np.arange(n_lamellae - 1) % 2 == 1)   # ramp starts
    n = N_BG * (1.0 + contrast * lam)[None, :] * np.ones((z.size, 1))
    ef = EpsField(x_axis_u=x, y_axis_u=np.array([0.0]), z_axis_u=z,
                  values_zyx=eps_of(n)[:, None, :])
    return d, ef, np.sort(np.concatenate([sw - 0.05, sw + 0.05])) / PN


@needs_lum
@pytest.mark.parametrize("pol", ["x", "y"])
def test_lateral_average_follows_the_aspect_ratio(pol):
    """Sixteen 25 nm lamellae (eps 2.72 / 1.69), ONE segment for the whole period vs the resolved
    solve (degree 12; no metal corner, so converged). In a 6 nm film (interval much wider than the
    film) E_x is pinned by its neighbours and the ARITHMETIC mean is the closer one; in a 50 nm film
    (interval narrower than the film) the HARMONIC (laminate) one is: lateral_average='auto' picks
    the closer one in both. TE (E_y along the lamellae) sees the arithmetic mean in every mode.
    Measured (TM): 6 nm film laminate 1.25e-4 / arithmetic 4.5e-5 (ratio 0.36); 50 nm laminate
    1.10e-3 / arithmetic 4.38e-3 (ratio 0.25); bar 0.6. On the premise "harmonic always" the 6 nm
    row would fail."""
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    row = 0 if pol == "x" else 1
    for t_film, closer in ((6e-9, "arithmetic"), (50e-9, "laminate")):
        d, ef, resolved = _planar_lamellar(pol, t_film)
        kw = dict(degree=12, n_orders=9, z_average="arithmetic", z_merge_rtol=None)
        ref, _ = design_to_pmm_stack(d, LAM, eps_by_region={"film": ef},
                                     lateral_partition=resolved, **kw)
        r_ref = complex(_solved(ref)[3][row, row])
        r = {}
        for mode in ("laminate", "arithmetic", "auto"):
            st, _ = design_to_pmm_stack(d, LAM, eps_by_region={"film": ef}, lateral_partition=[],
                                        lateral_average=mode, **kw)
            r[mode] = complex(_solved(st)[3][row, row])
        err = {m: abs(v - r_ref) for m, v in r.items()}
        assert r["auto"] == r[closer]
        if pol == "x":
            other = "laminate" if closer == "arithmetic" else "arithmetic"
            assert err[closer] < 0.6 * err[other]
        else:
            assert abs(r["laminate"] - r["arithmetic"]) < 1e-13 and err["auto"] < 1e-3


@needs_lum
def test_auto_partition_report_warning_and_merge():
    """Defaults on a ramped two-profile film: a converged report, z slabs merged, no warning;
    a tight tolerance with a small cap warns (AdvisoryWarning) and says so in the report."""
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    x = np.arange(200) * PN / 200
    ef = structured_field(np.linspace(0, T_FILM, 41), x, ramp_s(x, 20.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        st, names = design_to_pmm_stack(strip_design(), LAM, eps_by_region={"film": ef},
                                        degree=6)
    rep = st.dynameta_partition_report["film"]
    assert rep["converged"] and rep["max_spread"] <= 0.1 and rep["n_segments"] <= 24
    assert rep["z_average"] == "laminate" and rep["z_merge_rtol"] == 3e-2
    assert rep["lateral_average"] == "auto"
    assert rep["n_slabs"] == 40 and rep["n_layers"] == names.count("film") < 40
    assert XE1 / PN in rep["edges"] or min(abs(np.array(rep["edges"]) - XE1 / PN)) < 1e-12
    with pytest.warns(AdvisoryWarning, match="lateral partition stopped at 5 segments"):
        st, _ = design_to_pmm_stack(strip_design(), LAM, eps_by_region={"film": ef}, degree=6,
                                    partition_rtol=1e-3, max_segments=5)
    assert st.dynameta_partition_report["film"]["capped"]


@needs_lum
def test_structured_refusals():
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    d = strip_design()
    x = np.arange(32) * PN / 32
    ef = structured_field(ZN11, x, ramp_s(x, 20.0))
    with pytest.raises(ValueError, match="lateral_partition=None"):
        design_to_pmm_stack(d, LAM, eps_by_region={"film": ef}, lateral_partition=None)
    with pytest.raises(ValueError, match="lateral_partition must be"):
        design_to_pmm_stack(d, LAM, eps_by_region={"film": ef}, lateral_partition="uniform")
    with pytest.raises(ValueError, match="z_average='laminate'.*n_slices"):
        design_to_pmm_stack(d, LAM, eps_by_region={"film": ef}, z_average="laminate", n_slices=4)
    v = np.repeat(ef.values_zyx, 2, axis=1).copy()
    v[:, 1, :] *= 1.1
    efy = EpsField(x_axis_u=x, y_axis_u=np.array([0.0, 100.0]), z_axis_u=ZN11, values_zyx=v)
    with pytest.raises(ValueError, match="varies along y"):
        design_to_pmm_stack(d, LAM, eps_by_region={"film": efy})
    t = np.zeros((11, 1, 32, 3, 3), complex)
    t[..., 0, 0] = t[..., 1, 1] = t[..., 2, 2] = 2.0
    # a laterally UNIFORM gridded tensor is accepted (tests/test_pmm_tensor_layer.py); a laterally
    # STRUCTURED one is still refused
    eft = EpsField(x_axis_u=x, y_axis_u=np.array([0.0]), z_axis_u=ZN11, values_zyx=t)
    _st, _nm = design_to_pmm_stack(d, LAM, eps_by_region={"film": eft}, degree=6)
    assert _nm.count("film") == 10
    ts = t.copy()
    ts[:, :, 16:, 0, 0] = 2.5
    efts = EpsField(x_axis_u=x, y_axis_u=np.array([0.0]), z_axis_u=ZN11, values_zyx=ts)
    with pytest.raises(ValueError, match="laterally structured gridded TENSOR"):
        design_to_pmm_stack(d, LAM, eps_by_region={"film": efts})
    # n_slices with a structured field: arithmetic z, one PMM layer per uniform slice (no merge)
    st, names = design_to_pmm_stack(d, LAM, eps_by_region={"film": ef}, degree=6, n_slices=4,
                                    z_merge_rtol=None)
    assert names.count("film") == 4
    assert st.dynameta_partition_report["film"]["z_average"] == "arithmetic"


# =============================================================================================
# pmm_kwargs pass-through, far_field_orders, version tolerance
# =============================================================================================
@needs_lum
def test_pmm_kwargs_forwarded_and_change_the_answer():
    """elements_per_region=3 through the bridge equals a direct PMMStack(..., elements_per_region=3)
    and moves r on the strip cell by more than 1e-4 (the h-refinement control works; on v0.15.1
    the bridge had no pmm_kwargs: TypeError)."""
    import lumenairy as lum
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    d = strip_design()
    film = {"film": EpsField(scalar=complex(eps_of(N_BG)))}
    st, _ = design_to_pmm_stack(d, LAM, eps_by_region=film, degree=8, n_orders=9,
                                pmm_kwargs={"elements_per_region": 3})
    base, _ = design_to_pmm_stack(d, LAM, eps_by_region=film, degree=8, n_orders=9)
    hand = lum.PMMStack(P, n_substrate=1.45 + 0j, n_superstrate=1.0 + 0j, degree=8,
                        far_field_orders=9, elements_per_region=3)
    hand.add_layer(40e-9, segments=[(0.5 - W / P / 2, 1.0 + 0j), (W / P, -60 + 4j),
                                    (0.5 - W / P / 2, 1.0 + 0j)])
    for t_, e_ in ((2e-9, 1.70 ** 2), (6e-9, 2.00 ** 2), (6e-9, complex(eps_of(N_BG))),
                   (6e-9, 2.00 ** 2), (2e-9, 1.70 ** 2), (80e-9, -120 + 20j)):
        hand.add_layer(t_, eps=complex(e_))
    a, b, c = _solved(st), _solved(hand), _solved(base)
    # 1e-11, not 1e-14: where the hand-built strip widths differ from layer_to_pmm_segments' by ulps
    # the solve amplifies that to ~3e-13 in R (measured on an earlier cell); on this cell W / P = 3/8 is
    # exact in binary and the two stacks agree bit for bit (measured max |dR| 0)
    for x_, y_ in zip(a, b):
        assert np.allclose(x_, y_, rtol=1e-11, atol=1e-13)
    assert abs(a[3][0, 0] - c[3][0, 0]) > 1e-4


@needs_lum
def test_pmm_kwargs_refusals_and_version_hints():
    import types
    import lumenairy as lum
    from dynameta.optics.lumenairy_bridge._common import (LumenairyFeatureUnavailable,
                                                          lumenairy_has)
    from dynameta.optics.lumenairy_bridge.pmm_backend import (_resolve_pmm_kwargs,
                                                              design_to_pmm_stack)
    d = strip_design()
    film = {"film": EpsField(scalar=complex(eps_of(N_BG)))}
    for key in ("degree", "n_orders", "far_field_orders", "n_substrate", "layer_grids"):
        with pytest.raises(ValueError, match="set by the bridge itself"):
            design_to_pmm_stack(d, LAM, eps_by_region=film, pmm_kwargs={key: 1})
    with pytest.raises(LumenairyFeatureUnavailable, match="does not accept 'no_such_option'"):
        design_to_pmm_stack(d, LAM, eps_by_region=film, pmm_kwargs={"no_such_option": 1})
    with pytest.raises(TypeError, match="dict"):
        design_to_pmm_stack(d, LAM, eps_by_region=film, pmm_kwargs=[("elements_per_region", 2)])
    if not lumenairy_has("pmm_geometric_grade"):        # detected from the installed API
        with pytest.raises(LumenairyFeatureUnavailable, match=r"grade='geometric'.*>= 5\.51\.0"):
            design_to_pmm_stack(d, LAM, eps_by_region=film, pmm_kwargs={"grade": "geometric"})
        with pytest.raises(LumenairyFeatureUnavailable, match=r">= 5\.51\.0"):
            design_to_pmm_stack(d, LAM, eps_by_region=film, pmm_kwargs={"grade": ("geometric", 0.3)})

    class _OldStack:                       # a PMMStack signature from before window_halfwidth
        def __init__(self, period, *, n_substrate=1.0, n_superstrate=1.0, degree=16,
                     elements_per_region=1, grade=True, far_field_orders=21, n_orders=None,
                     factorization="auto", min_feature=None):
            pass

    fake = types.SimpleNamespace(PMMStack=_OldStack, __version__="5.22.0")
    with pytest.raises(LumenairyFeatureUnavailable, match=r"'window_halfwidth' \(needs lumenairy "
                                                          r">= 5\.33\.0\)"):
        _resolve_pmm_kwargs(fake, {"window_halfwidth": 2})
    assert _resolve_pmm_kwargs(fake, {"grade": False}) == {"grade": False}
    assert _resolve_pmm_kwargs(lum, None) == {}


@needs_lum
def test_far_field_orders_keyword_not_the_alias(monkeypatch):
    """The bridge calls PMMStack(far_field_orders=...) and never the n_orders alias."""
    import lumenairy as lum
    from dynameta.optics.lumenairy_bridge.pmm_backend import design_to_pmm_stack
    seen = {}
    real = lum.PMMStack

    class _Spy(real):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(lum, "PMMStack", _Spy)
    design_to_pmm_stack(strip_design(), LAM, degree=6, n_orders=11,
                        eps_by_region={"film": EpsField(scalar=complex(eps_of(N_BG)))},
                        pmm_kwargs={"min_feature": 1e-10})
    assert seen["far_field_orders"] == 11 and "n_orders" not in seen
    assert seen["min_feature"] == 1e-10


@needs_lum
def test_cache_fingerprint_carries_the_new_controls():
    from dynameta.optics.lumenairy_bridge.pmm_backend import make_lumenairy_pmm_solver
    base = make_lumenairy_pmm_solver().cache_fingerprint
    variants = [dict(pmm_kwargs={"elements_per_region": 2}), dict(partition_rtol=0.05),
                dict(max_segments=12), dict(lateral_average="arithmetic"),
                dict(lateral_partition=[0.25, 0.75]), dict(lateral_partition=None),
                dict(z_merge_rtol=None), dict(z_average="laminate")]
    fps = {make_lumenairy_pmm_solver(**v).cache_fingerprint for v in variants}
    assert base not in fps and len(fps) == len(variants)
    a = make_lumenairy_pmm_solver(pmm_kwargs={"grade": False, "elements_per_region": 2})
    b = make_lumenairy_pmm_solver(pmm_kwargs={"elements_per_region": 2, "grade": False})
    assert a.cache_fingerprint == b.cache_fingerprint


# =============================================================================================
# validation against the independent 2-D TM FEM (optics.fem_2d_tm)
# =============================================================================================
@needs_fem
def test_graded_film_under_strips_vs_fem():
    """z-graded (laterally uniform) film under the strip grating, default bridge path, vs the FEM
    on the same 41 samples (the FEM integrates the z-linear interpolant). Error budget: the PMM's
    algebraic TM corner convergence at degree 16 with 3 elements per region on this cell (the plain
    cell, uniform film, against R_FEM_CONVERGED: 4.4e-3) + the arithmetic z staircase (2.5e-4, 41 vs 321 samples at
    41 samples) ~ 4.6e-3 if the two add; bar 4e-3. Measured 3.8e-3 (FEM order 4: 0.354238-0.147728i). The slow ladder
    below shows the gap is the PMM's own convergence."""
    from dynameta.optics.fem_2d_tm import solve_design_2d_tm
    ef = uniform_field(np.linspace(0, T_FILM, 41))
    r_fem = solve_design_2d_tm(strip_design(), {"film": ef}, LAM, order=4).r
    r = _solve(strip_design(), {"film": ef}, degree=16, pmm_kwargs={"elements_per_region": 3}).r
    assert abs(r - r_fem) < 4e-3


@needs_fem
@pytest.mark.slow
def test_graded_film_under_strips_pmm_ladder_converges_to_fem():
    """The same cell with elements_per_region 3 / 6 / 8 (degree 16, laminate z + merge 3e-2): the
    distance to the FEM (order 5) falls monotonically and ends below 6e-4. Measured: 3.7e-3 /
    6.6e-4 / 2.4e-4 (epr 3 / 6 / 8)."""
    from dynameta.optics.fem_2d_tm import solve_design_2d_tm
    ef = uniform_field(np.linspace(0, T_FILM, 41))
    r_fem = solve_design_2d_tm(strip_design(), {"film": ef}, LAM, order=5).r
    d = []
    for epr in (3, 6, 8):
        r = _solve(strip_design(), {"film": ef}, degree=16, z_average="laminate",
                   z_merge_rtol=3e-2, pmm_kwargs={"elements_per_region": epr}).r
        d.append(abs(r - r_fem))
    assert d[0] > d[1] > d[2] and d[2] < 6e-4


@needs_fem
@pytest.mark.slow
def test_laterally_varying_film_vs_fem_and_partition_convergence():
    """Two z profiles under / between the strips joined by 20 nm lateral ramps: the 'auto'
    partition converges with partition_rtol (|dr| falls) and the converged bridge agrees with
    the FEM on the same sampled field (bilinear interpolant). Measured: |dr| between partition_rtol 0.5 / 0.2 / 0.1: 3.5e-2 then 1.3e-2;
    the 0.1 result is 1.5e-3 from the FEM."""
    from dynameta.optics.fem_2d_tm import solve_design_2d_tm
    x = np.arange(200) * PN / 200
    ef = structured_field(np.linspace(0, T_FILM, 41), x, ramp_s(x, 20.0))
    r_fem = solve_design_2d_tm(strip_design(), {"film": ef}, LAM, order=5).r
    rs = []
    for rt in (0.5, 0.2, 0.1):
        rs.append(_solve(strip_design(), {"film": ef}, degree=10, partition_rtol=rt,
                         max_segments=48).r)
    d = [abs(rs[1] - rs[0]), abs(rs[2] - rs[1])]
    assert d[1] < d[0]
    assert abs(rs[-1] - r_fem) < 2e-3
