"""Fast (no-FDTD-run) unit tests for the FDTD OpticalSolver seam helpers: the complex-eps -> FDTDLayer
Drude inversion, the Design -> layer mapping (order + guards), and the vacuum-end-media guard.

Also home to the audit D-3 absorber-thickness / probe-placement gates. Those raise at SETUP (before
any march), so the negative cases stay no-FDTD-run; the single positive "a fully clear config runs
SILENTLY and closes the energy budget" gate is the one small 2-D march in this module."""
import math
import warnings

import numpy as np
import pytest

from dynameta.constants import C_LIGHT
from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Circle
from dynameta.geometry.stack import Inclusion
from dynameta.materials import ConstantOptical, Material, MaterialRegistry
from dynameta.optics.fdtd_seam import (_eps_to_fdtd_layer, design_to_fdtd_layers,
                                       make_fdtd_optical_solver)

LAM = 1300e-9


def _fdtd_layer_eps(layer, lam_m):
    """The analytic eps(lam) the FDTDLayer represents (its convention: eps_inf - wp^2/(w^2 + i gamma w))."""
    w = 2.0 * math.pi * C_LIGHT / lam_m
    e = complex(layer.eps_inf)
    if layer.drude_wp_rad_s > 0.0:
        e = e - layer.drude_wp_rad_s ** 2 / (w ** 2 + 1j * layer.drude_gamma_rad_s * w)
    return e


@pytest.mark.parametrize("eps", [4.0 + 0j, 0.5 + 0j, 3.24 + 0.4j, 3.24 + 1.0j, -5.0 + 2.0j, -20.0 + 0.5j])
def test_drude_inversion_reproduces_eps_at_lambda(eps):
    """The inverted FDTDLayer must reproduce eps EXACTLY at lambda, with a stable background (eps_inf>=1
    except a pure positive-real dielectric, which is represented directly)."""
    L = _eps_to_fdtd_layer(200e-9, eps, LAM)
    assert abs(_fdtd_layer_eps(L, LAM) - eps) < 1e-6 * (abs(eps) + 1.0)
    pure_dielectric = (abs(eps.imag) < 1e-9 and eps.real > 0.0)
    assert pure_dielectric or L.eps_inf >= 1.0 - 1e-9


def test_lossless_dielectric_has_no_drude():
    L = _eps_to_fdtd_layer(100e-9, 4.0 + 0j, LAM)
    assert L.drude_wp_rad_s == 0.0 and abs(L.eps_inf - 4.0) < 1e-12


def _design(layer_specs):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    layers = []
    for k, (eps, th, incl) in enumerate(layer_specs):
        reg.add(Material("m%d" % k, ConstantOptical(complex(eps))))
        layers.append(Layer("s%d" % k, float(th), "m%d" % k, inclusions=list(incl)))
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="air")
    return Design(name="t", unit_cell=UnitCell.square(220e-9), stack=stack, electrodes=[], materials=reg)


def test_layers_superstrate_first_order():
    """Stack lists bottom->top; the FDTD layers must come out superstrate-first (incidence order)."""
    d = _design([(4.0, 100e-9, []), (9.0, 200e-9, [])])       # s0 (eps4) bottom, s1 (eps9) top
    layers = design_to_fdtd_layers(d, LAM)
    assert len(layers) == 2
    assert abs(layers[0].eps_inf - 9.0) < 1e-9 and abs(layers[1].eps_inf - 4.0) < 1e-9  # top (s1) first


def test_gain_eps_raises_instead_of_clamping_to_lossless():
    """AUDIT V-2: a sign-convention slip (Im(eps) < 0 = gain under exp(-i omega t)) used to be
    CLAMPED, unbounded: eps = -180 - 30j was realized as a strictly real -180+0j -- a collisionless
    metal, gamma = 0.0, absorption identically zero, with no warning. It must now raise, at the
    helper AND through the Design seam. Byte-identity for VALID inputs is unaffected: the guard
    runs before untouched arithmetic (`ei = max(0.0, eps.imag)` is kept verbatim)."""
    with pytest.raises(ValueError, match="exp\\(-i omega t\\)"):
        _eps_to_fdtd_layer(100e-9, -180.0 - 30.0j, LAM)
    d = _design([(-180.0 - 30.0j, 40e-9, [])])
    with pytest.raises(ValueError, match="Im\\(eps\\)"):
        design_to_fdtd_layers(d, LAM)
    # the passive counterpart of the same material is unaffected and IS absorbing
    L = _eps_to_fdtd_layer(100e-9, -180.0 + 30.0j, LAM)
    assert L.drude_gamma_rad_s > 0.0 and abs(_fdtd_layer_eps(L, LAM) - (-180.0 + 30.0j)) < 1e-6 * 181.0


def test_inclusions_layer_raises():
    incl = [Inclusion(Circle(0.0, 0.0, 30e-9), "m0")]         # a lateral inclusion -> not laterally uniform
    d = _design([(4.0, 100e-9, incl)])
    with pytest.raises(NotImplementedError):
        design_to_fdtd_layers(d, LAM)


def test_lossy_end_media_raises():
    """Lossless non-vacuum end media are supported; a LOSSY (complex) end medium still raises."""
    d = _design([(4.0, 100e-9, [])])
    solver = make_fdtd_optical_solver(dim=2)
    with pytest.raises(NotImplementedError):
        solver(d, None, {}, LAM, 1.5 + 0.2j, 1.0 + 0j)        # absorbing superstrate -> raise


def test_bad_dim_raises():
    with pytest.raises(ValueError):
        make_fdtd_optical_solver(dim=4)


def test_fdtd_sweep_solver_is_sweep_aware_and_callable():
    """The sweep-aware solver exposes solve_sweep (run_pipeline's fast-path hook) AND is a drop-in
    per-wavelength OpticalSolver (the __call__ fallback)."""
    from dynameta.optics.fdtd_seam import make_fdtd_sweep_optical_solver
    sw = make_fdtd_sweep_optical_solver(dim=2, resolution=16)
    assert hasattr(sw, "solve_sweep") and callable(sw)


def test_sweep_guards():
    from dynameta.optics.fdtd_seam import fdtd_sweep_spectrum
    d = _design([(4.0, 100e-9, [])])
    with pytest.raises(NotImplementedError):                # LOSSY end media -> raise before solving
        fdtd_sweep_spectrum(d, lambda_min_m=1200e-9, lambda_max_m=1400e-9, n_super=1.5 + 0.2j)


def test_fit_drude_recovers_known_drude():
    import numpy as np
    from dynameta.optics.fdtd_seam import fit_drude_to_eps
    Cc = 299792458.0
    einf, wp, g = 3.0, 1.2e15, 3.0e13
    lam = np.linspace(1100e-9, 1700e-9, 9)
    w = 2.0 * np.pi * Cc / lam
    eps = einf - wp ** 2 / (w ** 2 + 1j * w * g)
    fi, fwp, fg = fit_drude_to_eps(lam, eps)
    assert abs(fi - einf) < 1e-2 and abs(fwp - wp) / wp < 1e-3 and abs(fg - g) / g < 1e-2
    model = fi - fwp ** 2 / (w ** 2 + 1j * w * fg)
    assert np.max(np.abs(model - eps)) < 1e-3 * np.max(np.abs(eps))   # reproduces eps across the band


def test_graded_eps_from_carrier_and_layers():
    import numpy as np
    from dynameta.materials import DrudeOptical, M_E
    from dynameta.optics.fdtd_seam import eps_profile_from_carrier, graded_fdtd_layers
    drude = DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * M_E, gamma_rad_s=1.0e14)
    n = np.array([4.0e26, 1.0e27])                         # m^-3
    eps = eps_profile_from_carrier(n, 1500e-9, drude)
    assert eps.shape == (2,) and np.all(eps.imag > 0)      # passive loss
    assert abs(eps[0] - complex(drude.eps(1500e-9, n_m3=4.0e26))) < 1e-12
    layers = graded_fdtd_layers(400e-9, eps, 1500e-9)
    assert len(layers) == 2 and abs(layers[0].thickness_m - 200e-9) < 1e-15
    w = 2.0 * np.pi * 299792458.0 / 1500e-9
    for i, L in enumerate(layers):                         # each sublayer reproduces eps at lambda
        e = L.eps_inf - L.drude_wp_rad_s ** 2 / (w ** 2 + 1j * w * L.drude_gamma_rad_s)
        assert abs(e - eps[i]) < 1e-6 * (abs(eps[i]) + 1.0)


def test_fit_drude_lossless_dielectric():
    import numpy as np
    from dynameta.optics.fdtd_seam import fit_drude_to_eps
    Cc = 299792458.0
    lam = np.linspace(1100e-9, 1700e-9, 7)
    w = 2.0 * np.pi * Cc / lam
    fi, fwp, fg = fit_drude_to_eps(lam, np.full(7, 4.0 + 0j))
    model = fi - fwp ** 2 / (w ** 2 + 1j * w * fg)
    assert np.max(np.abs(model - 4.0)) < 5e-3               # non-dispersive eps=4 reproduced across the band


def test_fit_drude_lorentz_recovers_known_poles():
    import numpy as np
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_seam import fit_drude_lorentz
    Cc = 299792458.0
    L = FDTDLayer(thickness_m=1.0, eps_inf=2.0, drude_wp_rad_s=1.4e15, drude_gamma_rad_s=5.0e13,
                  lorentz_w0_rad_s=1.30e15, lorentz_gamma_rad_s=1.2e14, lorentz_delta_eps=1.0)
    lam = np.linspace(1200e-9, 1800e-9, 13)
    w = 2.0 * np.pi * Cc / lam
    eps = np.array([L.eps_at(wi) for wi in w])
    fit = fit_drude_lorentz(lam, eps)
    model = np.array([FDTDLayer(thickness_m=1.0, **fit).eps_at(wi) for wi in w])
    assert np.max(np.abs(model - eps)) < 1e-2 * np.max(np.abs(eps))   # reproduces eps across the band


def test_fit_drude_lorentz_degenerate_raises():
    import numpy as np
    from dynameta.optics.fdtd_seam import fit_drude_lorentz
    lam = np.linspace(1200e-9, 1800e-9, 5)
    with pytest.raises(RuntimeError):                       # all multi-starts fail -> clear error, not NoneType
        fit_drude_lorentz(lam, np.full(5, np.nan + 0j))


def test_fit_pure_lorentz_no_drude():
    import numpy as np
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_seam import fit_drude_lorentz
    Cc = 299792458.0
    L = FDTDLayer(thickness_m=1.0, eps_inf=2.25, lorentz_w0_rad_s=1.30e15,
                  lorentz_gamma_rad_s=1.2e14, lorentz_delta_eps=1.5)
    lam = np.linspace(1200e-9, 1800e-9, 11)
    w = 2.0 * np.pi * Cc / lam
    eps = np.array([L.eps_at(wi) for wi in w])
    fit = fit_drude_lorentz(lam, eps, with_drude=False)
    assert fit["drude_wp_rad_s"] < 1e12                    # no Drude pole fitted
    model = np.array([FDTDLayer(thickness_m=1.0, **fit).eps_at(wi) for wi in w])
    assert np.max(np.abs(model - eps)) < 1e-2 * np.max(np.abs(eps))


# ---- lateral-inclusion rasterization (structured cells) -------------------------------------------

def test_rasterize_circle_fill_fraction_and_placement():
    import numpy as np
    from dynameta.geometry.cross_section import Circle
    from dynameta.optics.fdtd_seam import _cell_axes, _layer_eps_cell
    P, r = 200e-9, 60e-9
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j))); reg.add(Material("hi", ConstantOptical(9.0 + 0j)))
    L = Layer("s", 100e-9, "air", inclusions=[Inclusion(Circle(P / 2, P / 2, r), "hi")])
    nx = ny = 240
    xs, ys = _cell_axes(nx, ny, P, P)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cell = _layer_eps_cell(L, X, Y, 1300e-9, reg, {})
    fill = float((np.abs(cell.real - 9.0) < 1e-9).mean())
    assert abs(fill - np.pi * r ** 2 / P ** 2) < 5e-3           # area matches the circle, to the grid res
    assert abs(cell[nx // 2, ny // 2].real - 9.0) < 1e-9        # center -> inclusion
    assert abs(cell[0, 0].real - 1.0) < 1e-9                    # corner -> background


def test_rasterize_priority_overlap():
    import numpy as np
    from dynameta.geometry.cross_section import Circle
    from dynameta.optics.fdtd_seam import _cell_axes, _layer_eps_cell
    P = 200e-9
    reg = MaterialRegistry()
    for nm, e in [("air", 1.0), ("a", 4.0), ("b", 9.0)]:
        reg.add(Material(nm, ConstantOptical(complex(e))))
    L = Layer("s", 100e-9, "air", inclusions=[Inclusion(Circle(P / 2, P / 2, 80e-9), "a", priority=0),
                                              Inclusion(Circle(P / 2, P / 2, 40e-9), "b", priority=5)])
    nx = ny = 120
    X, Y = np.meshgrid(*_cell_axes(nx, ny, P, P), indexing="ij")
    cell = _layer_eps_cell(L, X, Y, 1300e-9, reg, {})
    assert abs(cell[nx // 2, ny // 2].real - 9.0) < 1e-9        # higher-priority 'b' wins the overlap


def test_structured_lateral_grid_and_dispatch_guard():
    import numpy as np
    from dynameta.geometry.cross_section import Circle
    from dynameta.optics.fdtd_seam import design_has_inclusions, make_structured_lateral
    P = 220e-9
    d = _design([(4.0, 150e-9, [Inclusion(Circle(P / 2, P / 2, 60e-9), "m0")])])
    # the m0 inclusion sits in an air background layer; give the layer an air bg via a 2nd uniform material
    assert design_has_inclusions(d)
    layers, lateral_fn = make_structured_lateral(d, LAM)
    eps = lateral_fn(40, 40, 60, (np.arange(60) + 0.5) * 10e-9, 100e-9, 150e-9)
    assert eps.shape == (40, 40, 60)
    inb = (((np.arange(60) + 0.5) * 10e-9) >= 100e-9) & (((np.arange(60) + 0.5) * 10e-9) < 250e-9)
    assert eps[:, :, ~inb].max() <= 1.0 + 1e-9                  # vacuum pad outside the structure
    assert eps[:, :, inb].max() > 1.0                          # patterned eps inside the structure band
    # dim=2 + inclusions must raise
    with pytest.raises(NotImplementedError):
        make_fdtd_optical_solver(dim=2)(d, None, {}, LAM, 1.0 + 0j, 1.0 + 0j)


# ---- audit C5-2: the seam used to silently DROP graded/tensor eps_by_region entries ----

def _graded_ef(nz=13, eps_lo=2.0, eps_hi=9.0, thick_nm=120.0):
    """Asymmetric laterally-uniform graded EpsField, nm axes, ascending z (substrate-first)."""
    import numpy as np
    from dynameta.core.eps_field import EpsField
    z = np.linspace(0.0, thick_nm, nz)
    eps = eps_lo + (eps_hi - eps_lo) * (z / thick_nm) ** 2
    return EpsField(z_axis_u=z, y_axis_u=np.zeros(1), x_axis_u=np.zeros(1),
                    values_zyx=eps.reshape(-1, 1, 1).astype(complex))


def test_graded_eps_by_region_is_sliced_incidence_first():
    # a graded entry must produce per-slab FDTDLayers matching the shared slice_eps_field
    # staircase in INCIDENCE (superstrate-first = descending-eps-first here) order --
    # pre-audit this silently fell through to the nominal material eps (zero modulation)
    import numpy as np
    from dynameta.core.layered import slice_eps_field
    from dynameta.optics.tmm_reference import S
    d = _design([(4.0, 120e-9, [])])
    ef = _graded_ef()
    layers = design_to_fdtd_layers(d, LAM, eps_by_region={"s0": ef})
    slabs = list(reversed(slice_eps_field(ef, 1.0 / S)))
    assert len(layers) == len(slabs) and len(layers) == 12
    got = np.array([L.eps_inf for L in layers])
    want = np.array([s.eps.real for s in slabs])
    assert np.allclose(got, want, rtol=1e-12)
    assert got[0] > got[-1]                                   # top (high-eps) side first
    assert abs(sum(L.thickness_m for L in layers) - 120e-9) < 1e-15


def test_tensor_eps_by_region_raises():
    import numpy as np
    from dynameta.core.eps_field import EpsField
    d = _design([(4.0, 120e-9, [])])
    ef = EpsField(tensor=np.diag([4.0 + 0j, 4.0 + 0j, 2.0 + 0j]))
    with pytest.raises(NotImplementedError, match="TENSOR"):
        design_to_fdtd_layers(d, LAM, eps_by_region={"s0": ef})


def test_graded_drude_carrier_region_does_not_crash():
    # pre-audit repro: a DrudeOptical carrier layer + graded bridge field crashed with a
    # MISLEADING "DrudeOptical.eps requires n_m3" from the nominal fallback -- the seam
    # was holding the bias eps it had just discarded
    from dynameta.materials import DrudeOptical
    d = _design([(4.0, 120e-9, [])])
    d.materials.add(Material("ito", DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * 9.109e-31,
                                                 gamma_rad_s=1.6e14)))
    d.stack.layers[0].background_material = "ito"
    layers = design_to_fdtd_layers(d, LAM, eps_by_region={"s0": _graded_ef()})
    assert len(layers) == 12                                  # sliced from the bias field


def test_sweep_solver_graded_bias_raises_not_silent():
    # the broadband one-pole-per-layer path cannot carry a graded profile: it must say so
    # (pre-audit it silently solved the UNMODULATED nominal stack)
    from dynameta.optics.fdtd_seam import make_fdtd_sweep_optical_solver
    import numpy as np
    d = _design([(4.0, 120e-9, [])])
    sw = make_fdtd_sweep_optical_solver(dim=2, resolution=16)
    ef = _graded_ef()
    with pytest.raises(NotImplementedError, match="graded"):
        sw.solve_sweep(d, None, lambda lam: {"s0": ef},
                       np.array([1.25e-6, 1.35e-6]), 1.0 + 0j, 1.0 + 0j)


def test_structured_path_graded_bg_raises():
    from dynameta.optics.fdtd_seam import _layer_bg_eps
    d = _design([(4.0, 120e-9, [])])
    with pytest.raises(NotImplementedError, match="graded"):
        _layer_bg_eps(d.stack.layers[0], LAM, d.materials, {"s0": _graded_ef()})


def test_fdtd_graded_modulation_moves_R():
    # end-to-end sensitivity: the per-wavelength FDTD seam must actually SEE the graded
    # bias (pre-audit: bit-identical R across biases). Coarse grid -- we assert
    # modulation, not oracle-grade accuracy.
    import numpy as np
    d = _design([(4.0, 120e-9, [])])
    solver = make_fdtd_optical_solver(dim=2, resolution=16)
    r_nom = solver(d, None, {}, LAM, 1.0 + 0j, 1.0 + 0j)
    r_mod = solver(d, None, {"s0": _graded_ef()}, LAM, 1.0 + 0j, 1.0 + 0j)
    assert abs(r_mod.R - r_nom.R) > 1e-3


def test_seam_honors_or_raises_design_optical():
    # audit C5-7: the seam used to IGNORE design.optical entirely -- theta/azimuth/
    # incidence_side silently got the normal-incidence top-side answer
    from dynameta.geometry.specs import OpticalSpec
    solver = make_fdtd_optical_solver(dim=2, resolution=16)
    d = _design([(4.0, 120e-9, [])])
    d.optical = OpticalSpec(polarization="y", incidence_angle_deg=30.0)
    with pytest.raises(NotImplementedError, match="oblique"):
        solver(d, None, {}, LAM, 1.0 + 0j, 1.0 + 0j)
    d.optical = OpticalSpec(polarization="y", incidence_angle_deg=0.0, incidence_side="bottom")
    with pytest.raises(NotImplementedError, match="incidence_side"):
        solver(d, None, {}, LAM, 1.0 + 0j, 1.0 + 0j)
    # normal-incidence top-side (any pol on a uniform stack) still solves
    d.optical = OpticalSpec(polarization="x", incidence_angle_deg=0.0)
    assert solver(d, None, {}, LAM, 1.0 + 0j, 1.0 + 0j).R >= 0.0


def test_oblique_solvers_refuse_dropped_nonlinear_terms():
    # audit C5-7: the oblique kernels carry no chi3/chi2/raman/gain ADEs; an amplifying
    # stack at 20 deg used to return R0/T0 BIT-IDENTICAL to the passive layer
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_nd import solve_fdtd_2d_oblique
    lay = FDTDLayer(thickness_m=300e-9, eps_inf=4.0, gain_w_rad_s=1.45e15,
                    gain_dw_rad_s=3.6e14, gain_kappa_C2_kg=2.8e-8, gain_dN_m3=5e23)
    with pytest.raises(NotImplementedError, match="gain_dN_m3"):
        solve_fdtd_2d_oblique([lay], period_x_m=300e-9, angle_deg=20.0,
                              lambda_min_m=1.2e-6, lambda_max_m=1.5e-6, resolution=16)


def test_ring_time_extends_window_for_narrow_poles():
    # audit C3-6: the DFT window must carry the material memory of narrow Lorentz/gain
    # poles (a loaded-Q~600 line rang past the fixed 200*tau window: |dT0|=0.102 vs TMM,
    # silently); passive/broad layers keep the legacy window exactly
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_nd.solve2d import _ring_time_s
    passive = FDTDLayer(thickness_m=200e-9, eps_inf=2.25)
    assert _ring_time_s([passive]) == 0.0
    narrow = FDTDLayer(thickness_m=200e-9, eps_inf=2.25, lorentz_w0_rad_s=1.3e15,
                       lorentz_gamma_rad_s=1e12, lorentz_delta_eps=0.002)
    assert _ring_time_s([narrow]) == pytest.approx(18.4e-12, rel=1e-12)
    gainy = FDTDLayer(thickness_m=200e-9, eps_inf=4.0, gain_w_rad_s=1.45e15,
                      gain_dw_rad_s=2e12, gain_kappa_C2_kg=2.8e-8, gain_dN_m3=5e23)
    assert _ring_time_s([passive, gainy]) == pytest.approx(18.4 / 2e12, rel=1e-12)


# =====================================================================================================
# audit D-3 (wave-3 redesign): the ABSORBER must be thick enough, and both R/T PROBE planes must
# clear it. The source only warns.
# =====================================================================================================
# k_src / k_pL / k_pR are placed as FRACTIONS OF THE Z PAD (0.35 pad, 0.7 pad, 0.3 pad past the
# structure) in five front ends; `npml` is a fixed CELL count. The wave-2 guard demanded
# k_src/k_pL >= npml+2 and k_pR <= nz-1-npml-2. Re-measured 2026-07-26 (guard bypassed, four
# fixtures) that rule was wrong in BOTH directions:
#   * it never checked npml itself, so npml <= 2 passed silently at up to an 84 % (R0+T0 = 1.84) /
#     197 % (R_flux+T_flux = 2.97) energy-budget violation on a LOSSLESS slab;
#   * it rejected on the SOURCE, whose launch attenuation cancels in the two-run DFT ratio, so
#     npml = 5..12 thin-pad configs that measure PERFECT were refused -- including the library
#     default npml=12 and validation/fdtd_oblique_jax.py GATE D (which passed by ONE cell).
# The rule is now (1) npml >= 4, (2) each probe's predicted one-way CPML amplitude attenuation
# <= 0.3 %, (3) a buried source WARNS. Fixture tables below.
_D3_BAND = dict(lambda_min_m=1.2e-6, lambda_max_m=1.8e-6, resolution=20)
_D3_SLAB = dict(thickness_m=200e-9, eps_inf=4.0)

# (label, k_src, k_pL, k_pR, nz) -- the indices the front ends compute for each fixture, and, per
# npml, the measured in-band R_flux+T_flux range on a LOSSLESS eps=4 slab with the guard bypassed
# (backend 'numpy', 2026-07-26) next to the new verdict. 'S' = accepted with the source warning.
# fixture A: thk=300 nm, lam 1.4-1.6 um, resolution=20, n_pad_wave=0.35
_D3_FIX_A = ("A/300nm", 6, 11, 29, 42, {
    1: ("RAISE", 1.0088, 2.9721), 2: ("RAISE", 1.0108, 1.0719), 3: ("RAISE", 1.0049, 1.0319),
    4: ("warn", 0.9950, 0.9986), 5: ("warn", 0.9999, 1.0001), 6: ("pass", 0.9999, 1.0000),
    7: ("S", 1.0000, 1.0000), 8: ("S", 1.0000, 1.0000), 9: ("S", 1.0000, 1.0000),
    10: ("S", 1.0000, 1.0000), 11: ("S", 1.0000, 1.0000), 12: ("S", 0.9990, 0.9991),
    13: ("RAISE", 0.9899, 0.9908), 14: ("RAISE", 0.9604, 0.9633), 15: ("RAISE", 0.9025, 0.9079),
    16: ("RAISE", 0.8165, 0.8244), 20: ("RAISE", 0.3623, 0.3649)})
# fixture B: thk=600 nm, lam 1.4-1.6 um, resolution=20, n_pad_wave=0.35
_D3_FIX_B = ("B/600nm", 6, 11, 38, 50, {
    1: ("RAISE", 0.6600, 1.3955), 2: ("RAISE", 1.0602, 1.4549), 3: ("RAISE", 0.9604, 0.9982),
    4: ("warn", 0.9957, 1.0027), 5: ("warn", 1.0005, 1.0007), 6: ("pass", 0.9997, 0.9997),
    7: ("S", 0.9998, 0.9999), 8: ("S", 0.9999, 0.9999), 9: ("S", 0.9999, 1.0000),
    10: ("S", 1.0000, 1.0000), 11: ("S", 1.0000, 1.0000), 12: ("S", 0.9983, 0.9983),
    13: ("RAISE", 0.9836, 0.9836), 14: ("RAISE", 0.9416, 0.9416), 15: ("RAISE", 0.8669, 0.8670),
    16: ("RAISE", 0.7646, 0.7648), 20: ("RAISE", 0.3052, 0.3096)})
_D3_BUDGET_TOL = 2e-3            # documented lossless-slab tolerance (an ACCEPTED config must meet it)
_D3_BAD_TOL = 5e-3               # a REJECTED config must miss it by at least this much
_D3_WARN_TOL = 2e-2              # accepted-with-a-thin-npml-warning: the warning quotes 0.5-2 %


def _d3_layer():
    from dynameta.optics.fdtd import FDTDLayer
    return FDTDLayer(**_D3_SLAB)


def _d3_verdict(k_src, k_pL, k_pR, nz, npml, p_cells=15.7, npw=0.35, res=20):
    """'RAISE' / 'warn' (thin npml) / 'S' (source buried) / 'pass' from the live guard."""
    import warnings as _w
    from dynameta.optics.fdtd_nd.solve2d import _check_probe_placement
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        try:
            _check_probe_placement("t", k_src, k_pL, k_pR, nz, npml, p_cells * 1e-8, 1e-8, npw, res)
        except ValueError:
            return "RAISE"
    if any("thin CPML" in str(x.message) for x in rec):
        return "warn"
    return "S" if rec else "pass"


@pytest.mark.parametrize("fix", [_D3_FIX_A, _D3_FIX_B], ids=lambda f: f[0])
def test_d3_guard_verdict_matches_the_measured_fixtures(fix):
    """THE calibration gate: on both fixtures, every npml the guard REJECTS measures badly and
    every npml it ACCEPTS measures within tolerance. Tables are the 2026-07-26 re-run above.

    Rejections split into the two real failure modes: npml <= 3 is too thin an absorber (up to
    197 % over budget) and npml >= 13 buries a probe in the graded absorber (1.6 % -> 70 %). The
    accepted band npml = 4..12 spans the library default (12), where the old rule raised.
    """
    label, k_src, k_pL, k_pR, nz, table = fix
    for npml, (want, lo, hi) in sorted(table.items()):
        got = _d3_verdict(k_src, k_pL, k_pR, nz, npml)
        assert got == want, "{} npml={}: guard says {}, table says {}".format(label, npml, got, want)
        off = max(abs(lo - 1.0), abs(hi - 1.0))
        if want == "RAISE":                                  # every rejected config IS bad
            assert off > _D3_BAD_TOL, "{} npml={} rejected but measures {:.4f}..{:.4f}".format(
                label, npml, lo, hi)
        elif want == "warn":                                 # thin absorber: flagged, never silent
            assert off <= _D3_WARN_TOL, "{} npml={} warned but measures {:.4f}..{:.4f}".format(
                label, npml, lo, hi)
        else:                                                # every silently-accepted config IS good
            assert off <= _D3_BUDGET_TOL, "{} npml={} accepted but measures {:.4f}..{:.4f}".format(
                label, npml, lo, hi)
    # and the thin-npml warning earns its keep: npml=4 is the one accepted row that MISSES the
    # silent tolerance (0.4-0.5 %), which is exactly what it warns about.
    assert max(abs(v - 1.0) for v in table[4][1:]) > _D3_BUDGET_TOL


def test_d3_npml_floor_is_a_hard_raise_below_the_measured_cliff():
    """npml itself was never validated -- the mode the wave-2 guard missed entirely, and the worst
    one measured. R_flux+T_flux on a LOSSLESS slab with the probes fully clear:
        npml = 1: 2.9721 (fixture A) / 1.3955 (B)     npml = 2: 1.0719 / 1.4549  (R0+T0 to 1.8387)
        npml = 3: 1.0319 / 0.9604                     npml = 4: 0.9986 / 1.0027
        npml = 5: 1.0001 / 1.0007                     npml >= 8: within 1e-4
    So: hard raise below 4, warn below 6."""
    from dynameta.optics.fdtd_nd.solve2d import _NPML_MIN, _NPML_WARN, _check_probe_placement
    assert (_NPML_MIN, _NPML_WARN) == (4, 6)
    clear = dict(nz=200, k_pL=70, k_pR=130, pad=1.0e-6, dz=1.0e-8, n_pad_wave=1.0, resolution=20)
    for npml in (0, 1, 2, 3):
        with pytest.raises(ValueError) as exc:
            _check_probe_placement("solve_fdtd_2d", 35, npml=npml, **clear)
        assert "npml" in str(exc.value) and "CPML" in str(exc.value)
    with pytest.warns(RuntimeWarning, match="thin CPML"):
        _check_probe_placement("solve_fdtd_2d", 35, npml=4, **clear)
    with warnings.catch_warnings():                          # npml >= 6 with everything clear: silent
        warnings.simplefilter("error")
        _check_probe_placement("solve_fdtd_2d", 35, npml=6, **clear)


def test_d3_source_in_the_cpml_only_warns_because_the_ratio_cancels():
    """The wave-2 guard REJECTED a buried source. It should not: R0/T0/R_flux/T_flux are two-run
    DFT ratios against a vacuum reference injected through the SAME absorber, and the reflected /
    transmitted wave inherits the launch attenuation of the incident wave that produced it, so it
    cancels. Measured with the probes clear and the source 5-6 cells deep (npml=11-12): fixture A
    R_flux+T_flux = [0.9999, 1.0000] and fixture B [1.0000, 1.0000] -- i.e. 2e-4, at a launch
    amplitude 44 % down. It still warns: the SNR loss is real (a broadband 1.2-1.8 um fixture with
    the source 5-8 cells deep scattered its band-edge bins by +-1.2 %), and the terminal case is
    caught by _check_band."""
    from dynameta.optics.fdtd_nd.solve2d import _check_probe_placement, _pml_atten
    clear = dict(nz=200, k_pL=70, k_pR=130, npml=12, pad=1.0e-6, dz=1.0e-8, n_pad_wave=1.0,
                 resolution=20)
    with pytest.warns(RuntimeWarning) as rec:
        _check_probe_placement("solve_fdtd_2d", 6, **clear)  # source 6 cells inside the CPML
    msg = str(rec[0].message)
    assert "k_src=6" in msg and "6 cell(s) inside" in msg and "cancels" in msg
    assert _pml_atten(6, 12) == pytest.approx(0.4443, abs=2e-3)   # the 44 % quoted above
    with warnings.catch_warnings():                          # k_src == npml is already outside
        warnings.simplefilter("error")
        _check_probe_placement("solve_fdtd_2d", 12, **clear)


def test_d3_probe_clearance_is_an_attenuation_budget_not_a_cell_count():
    """cpml_z grades sigma as (depth/npml)^3, so ONE cell of burial is harmless at npml=12
    (0.13 % amplitude) and fatal at npml=4 (10.2 %) -- a fixed cell margin cannot express that.
    Both ends are guarded symmetrically (low-z depth npml-k_pL, high-z depth k_pR-(nz-1-npml))."""
    from dynameta.optics.fdtd_nd.solve2d import (_PROBE_ATTEN_MAX, _check_probe_placement,
                                                 _max_probe_depth, _pml_atten)
    assert _max_probe_depth(12) == 1 and _max_probe_depth(4) == 0 and _max_probe_depth(20) == 2
    assert _pml_atten(0, 12) == 0.0 and _pml_atten(-3, 12) == 0.0
    assert _pml_atten(1, 12) < _PROBE_ATTEN_MAX < _pml_atten(2, 12)
    ok = dict(nz=200, npml=12, pad=1.0e-6, dz=1.0e-8, n_pad_wave=1.0, resolution=20)
    _check_probe_placement("t", 35, 70, 130, **ok)           # both probes far outside -> silent
    _check_probe_placement("t", 35, 11, 200 - 1 - 12 + 1, **ok)   # both exactly 1 cell deep -> ok
    with pytest.raises(ValueError, match="right"):           # 2 cells into the far CPML
        _check_probe_placement("t", 35, 70, 200 - 1 - 12 + 2, **ok)
    with pytest.raises(ValueError, match="left"):            # 2 cells into the near CPML
        _check_probe_placement("t", 35, 10, 130, **ok)


def test_d3_atten_model_uses_cpml_z_defaults():
    """_pml_atten hard-codes the grading exponent m and the target reflection R0. Pin them to
    cpml_z's OWN defaults, which every front end takes (they pass only nz/dz/dt/npml/n_super/
    n_sub) -- if cpml_z's grading is ever retuned the guard must be retuned with it."""
    import inspect
    from dynameta.optics.fdtd_nd.cpml import cpml_z
    from dynameta.optics.fdtd_nd.solve2d import _pml_atten
    p = inspect.signature(cpml_z).parameters
    assert p["m"].default == 3.0 and p["R0"].default == 1.0e-6
    src = inspect.getsource(_pml_atten)
    assert "m=3.0" in src and "R0=1.0e-6" in src
    # closed form: a(d) = 1 - exp(-27.63 (d(d+1)/2)^2 / npml^4) for m=3, R0=1e-6
    for d, n in ((1, 12), (3, 14), (5, 20)):
        want = 1.0 - math.exp(-(-4.0 * math.log(1e-6) / 2.0) * (d * (d + 1) / 2.0) ** 2 / n ** 4)
        assert _pml_atten(d, n) == pytest.approx(want, rel=1e-12)


def test_d3_ledger_silent_violation_config_is_still_rejected():
    """The audit ledger's own trigger (resolution=8, n_pad_wave=0.5, npml=12 -> k_src=4, k_pL=8,
    k_pR=18, nz=28) returned R_flux+T_flux = 0.8625 silently. It is rejected by the NEW rule too,
    and for the right reason: its probes are 4 and 3 cells inside the absorber (12.5 % / 4.7 %
    predicted attenuation; re-measured R_flux+T_flux = [0.7502, 0.7725]). The message names every
    knob that can fix it."""
    from dynameta.optics.fdtd_nd.solve2d import _check_probe_placement
    with pytest.raises(ValueError) as exc:
        _check_probe_placement("solve_fdtd_2d", 4, 8, 18, 28, 12, 10.5e-8, 1e-8, 0.5, 8)
    msg = str(exc.value)
    assert "CPML" in msg and "k_pL=8" in msg and "k_pR=18" in msg
    for knob in ("n_pad_wave", "resolution", "npml"):
        assert knob in msg


def test_d3_oblique_jax_gate_d_config_passes_with_margin():
    """validation/fdtd_oblique_jax.py GATE D (3-D oblique, resolution=9, n_pad_wave=2.5 ->
    k_src=15, k_pL=30, k_pR=58, nz=89, npml=12) passed the wave-2 rule by exactly ONE cell on the
    source. Under the new rule it is clean with 18 cells of probe margin and 3 cells of source
    margin, and stays clean over npml=6..15."""
    from dynameta.optics.fdtd_nd.solve2d import _check_probe_placement
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # not even a warning
        for npml in range(6, 16):
            _check_probe_placement("solve_fdtd_3d_oblique", 15, 30, 58, 89, npml,
                                   43.6e-8, 1e-8, 2.5, 9)


def test_d3_guard_covers_every_front_end():
    """All five front ends that place probes as pad fractions are guarded (audit D-3 listed four
    sites plus the 2-D oblique twin). The trigger is now a PROBE burial -- n_pad_wave=0.35 at
    npml=20 puts both probes 5 cells into the absorber (3.8 % attenuation each; measured
    R_flux+T_flux = [0.8848, 0.9092]). Each raises at SETUP, so no march runs here."""
    from dynameta.optics.fdtd_nd import (solve_fdtd_2d, solve_fdtd_2d_oblique, solve_fdtd_3d,
                                         solve_fdtd_3d_oblique)
    thin = dict(n_pad_wave=0.35, npml=20)
    calls = {
        "solve_fdtd_2d": lambda: solve_fdtd_2d([_d3_layer()], period_x_m=300e-9, **thin, **_D3_BAND),
        "solve_fdtd_2d_oblique": lambda: solve_fdtd_2d_oblique(
            [_d3_layer()], period_x_m=300e-9, angle_deg=10.0, nx=4, **thin, **_D3_BAND),
        "solve_fdtd_3d": lambda: solve_fdtd_3d(
            [_d3_layer()], period_x_m=300e-9, period_y_m=300e-9, nx=4, ny=4, **thin, **_D3_BAND),
        "solve_fdtd_3d_oblique": lambda: solve_fdtd_3d_oblique(
            [_d3_layer()], period_x_m=300e-9, period_y_m=300e-9, angle_deg=10.0, nx=4, ny=4,
            **thin, **_D3_BAND),
    }
    for name, fn in calls.items():
        with pytest.raises(ValueError, match="CPML"):
            fn()


def test_d3_a_fully_clear_config_runs_silently_and_closes():
    """The positive control (the one 2-D march in this module): with BOTH probes and the source
    clear -- resolution=20, n_pad_wave=0.67, npml=12 -> k_src=14 > npml -- the guard is completely
    silent and the lossless slab closes to 2.1e-5. Acting on the source WARNING is what buys that:
    the same geometry at n_pad_wave=0.35 (probes clear, source 5 cells deep, warning only) still
    runs but closes to only 1.2e-2, and at npml=20 (probes 5 cells deep, REJECTED) it would have
    been 1.15e-1."""
    from dynameta.optics.fdtd_nd import solve_fdtd_2d
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = solve_fdtd_2d([_d3_layer()], period_x_m=300e-9, n_pad_wave=0.67, npml=12, **_D3_BAND)
    m = r.band
    assert np.any(m)
    assert float(np.max(np.abs(r.R_flux[m] + r.T_flux[m] - 1.0))) < _D3_BUDGET_TOL
    assert float(np.max(np.abs(r.R0[m] + r.T0[m] - 1.0))) < _D3_BUDGET_TOL


def test_d3_empty_band_mask_raises_instead_of_silent_zeros():
    """audit D-3 sub-mode: an empty well-excited band used to be returned silently, and every
    downstream `result.R0[result.band].min()` then died with an opaque `zero-size array to
    reduction operation minimum` far from the cause. This is now also the HARD backstop for a
    buried source, whose placement only warns."""
    from dynameta.optics.fdtd_nd.solve2d import _check_band
    good = np.array([False, True, True, False])
    assert _check_band("solve_fdtd_2d", good, 1.0e14, 2.5e14) is None
    with pytest.raises(ValueError) as exc:
        _check_band("solve_fdtd_2d", np.zeros(2617, dtype=bool), 1.0e14, 2.5e14)
    msg = str(exc.value)
    assert "EMPTY" in msg and "2617" in msg and "D-3" in msg
    assert "npml" in msg                                     # points at the actual cause


def test_d3_empty_band_guard_is_wired_into_solve_fdtd_1d(monkeypatch):
    """P3 scope gap: the empty-band raise reached the five 2-D/3-D front ends but NOT
    solve_fdtd_1d, which returned the silent all-False mask. The 1-D grid terminates in Mur ABCs
    (no CPML), so there is no placement guard to run alongside it -- only this one. Spy on the
    shared helper to prove the 1-D path calls it with its own entry-point name, then force an
    empty mask and check the raise propagates."""
    from dynameta.optics import fdtd as F
    from dynameta.optics.fdtd_nd import solve2d as S2
    seen = []
    real = S2._check_band
    monkeypatch.setattr(S2, "_check_band",
                        lambda ep, band, f_lo, f_hi: (seen.append((ep, int(np.sum(band)))),
                                                      real(ep, band, f_lo, f_hi))[1])
    kw = dict(lambda_min_m=1.4e-6, lambda_max_m=1.6e-6, resolution=8, n_pad_wave=1.0)
    r = F.solve_fdtd_1d([_d3_layer()], **kw)
    assert seen and seen[0][0] == "solve_fdtd_1d" and seen[0][1] > 0 and np.any(r.band)
    monkeypatch.setattr(S2, "_check_band",
                        lambda ep, band, f_lo, f_hi: real(ep, np.zeros_like(band), f_lo, f_hi))
    with pytest.raises(ValueError, match="EMPTY"):
        F.solve_fdtd_1d([_d3_layer()], **kw)


# =====================================================================================================
# audit D-2: the HALF-TIMESTEP E/H stagger in the Poynting-flux extraction
# =====================================================================================================
# _flux's docstring used to claim the half-cell AND half-step offsets both cancel in the R/T ratio.
# Only the half-cell one does (it is removed inside the kernel). The kernels record E at t=(n+1)dt but
# H at t=(n+1/2)dt, so E conj(H) carries a spurious exp(+i w dt/2): harmless for a propagating order
# (a common cos(w dt/2)) but for an EVANESCENT / near-cutoff order, where E H* is nearly pure
# imaginary, it manufactures a flux Im(E H*) sin(w dt/2) that survives the ratio.
_D2_N, _D2_DT, _D2_K0 = 4096, 1.0e-17, 137


def _d2_staggered(a_e, a_h):
    """(ey, hx) at the kernel's staggering -- E at (n+1)dt, H at (n+1/2)dt -- for one bin-exact mode."""
    n = np.arange(_D2_N)[:, None]
    w = 2.0 * np.pi * _D2_K0 / (_D2_N * _D2_DT)
    ey = np.real(a_e * np.exp(-1j * w * (n + 1.0) * _D2_DT))
    hx = np.real(a_h * np.exp(-1j * w * (n + 0.5) * _D2_DT))
    return ey, hx


def test_d2_evanescent_order_carries_no_spurious_flux():
    """E H* pure imaginary == an evanescent / cutoff order: the true time-averaged S_z is EXACTLY 0.

    Measured on this fixture: uncorrected +7.342e-2; with the shipped exp(+i w dt/2) H de-stagger
    -2.6e-16 (machine zero); with the audit's originally-prescribed exp(-i w dt/2) +1.460e-1, i.e.
    1.989x WORSE (the predicted exactly-2 doubling). That ratio is why the sign matters."""
    from dynameta.optics.fdtd_nd.results import _flux
    ey, hx = _d2_staggered(1.0 + 0j, 0.0 + 0.7j)             # conj(A)*B pure imaginary -> S_z == 0
    P = _flux(ey, hx, _D2_DT) / (_D2_N / 2.0) ** 2
    assert abs(P[_D2_K0]) < 1e-12                            # was 7.34e-2 before the fix


def test_d2_propagating_order_flux_is_the_analytic_value():
    """A forward propagating order (E H* real) must give exactly -Re(conj(A) B) -- the correction
    removes an error there too, it just happens to be second order in sin(w dt/2)."""
    from dynameta.optics.fdtd_nd.results import _flux
    a_e, a_h = 1.0 + 0j, -1.0 / 376.730313668 + 0j
    ey, hx = _d2_staggered(a_e, a_h)
    P = _flux(ey, hx, _D2_DT) / (_D2_N / 2.0) ** 2
    assert P[_D2_K0] == pytest.approx(-np.real(np.conj(a_e) * a_h), rel=1e-9)
    assert P[_D2_K0] > 0.0                                   # +z Poynting sign convention


def test_d2_flux3d_reduces_to_flux_and_de_staggers_identically():
    """_flux3d must carry the SAME correction: with Ex = Hy = 0 it reduces to the 2-D _flux, and the
    evanescent order gives zero flux there too."""
    from dynameta.optics.fdtd_nd.results import _flux, _flux3d
    ey2, hx2 = _d2_staggered(1.0 + 0j, -2.0e-3 + 0j)
    ey = ey2[:, :, None]; hx = hx2[:, :, None]               # (nsteps, nx, ny) with ny = 1
    z = np.zeros_like(ey)
    assert np.allclose(_flux3d(z, ey, hx, z, _D2_DT), _flux(ey2, hx2, _D2_DT), rtol=1e-12, atol=0.0)
    eyE, hxE = _d2_staggered(1.0 + 0j, 0.0 + 0.7j)           # evanescent
    P3 = _flux3d(z, eyE[:, :, None], hxE[:, :, None], z, _D2_DT) / (_D2_N / 2.0) ** 2
    assert abs(P3[_D2_K0]) < 1e-12


@pytest.mark.parametrize("bad", [complex(4.0, float("nan")), complex(4.0, float("inf")),
                                 complex(4.0, float("-inf")), complex(float("nan"), 0.0),
                                 complex(float("inf"), 1.0)])
def test_nonfinite_eps_raises_in_both_twins(bad):
    """AUDIT V-2 follow-on: a NaN/inf permittivity fell straight THROUGH the Im(eps) >= 0 guard
    (every comparison against NaN is False) and the two twins then disagreed -- the scalar path's
    `max(0.0, nan)` returns 0.0, so eps = 4 + nan*1j became an ordinary LOSSLESS eps_inf = 4
    dielectric, while `np.maximum(nan, 0.0)` propagates and the vectorized twin emitted an all-NaN
    layer. Both must refuse, with the SAME error class, keeping the twins' byte-identity contract
    over the guard as well as the arithmetic."""
    from dynameta.optics.fdtd_seam import effect_eps_to_fdtd_grid
    with pytest.raises(ValueError, match="must be FINITE"):
        _eps_to_fdtd_layer(100e-9, bad, LAM)
    with pytest.raises(ValueError, match="must be FINITE"):
        effect_eps_to_fdtd_grid(np.array([[bad]], dtype=np.complex128), LAM)
    # a single bad cell in an otherwise healthy grid is caught too, and located
    grid = np.full((3, 4), 4.0 + 1e-3j, dtype=np.complex128)
    grid[2, 1] = bad
    with pytest.raises(ValueError, match="flat index 9"):
        effect_eps_to_fdtd_grid(grid, LAM)


def test_finite_eps_path_is_unchanged_by_the_finiteness_guard():
    """The guard runs before untouched arithmetic: every VALID input still gives bit-for-bit the
    same layer, and the scalar/vector twins still agree cell-by-cell."""
    from dynameta.optics.fdtd_seam import effect_eps_to_fdtd_grid
    good = [4.0 + 0.0j, complex(4.0, -0.0), 4.0 + 1e-8j, 2.25 + 0.0j, -180.0 + 30.0j,
            1.0 + 5.0j, 12.0 + 3.0j, 0.5 + 0.2j]
    grid = np.array(good, dtype=np.complex128).reshape(2, 4)
    einf, wp, gam = effect_eps_to_fdtd_grid(grid, LAM)
    for i, e in enumerate(good):
        L = _eps_to_fdtd_layer(100e-9, e, LAM)
        j, k = divmod(i, 4)
        assert L.eps_inf == einf[j, k]
        assert L.drude_wp_rad_s == wp[j, k]
        assert L.drude_gamma_rad_s == gam[j, k]


# =====================================================================================================
# audit D-9 / D-10 / D-11 / D-12 (wave-5): flux SIGNS, the DFT-window memory rule, the CFL guard and
# the Kerr/time-varying composition. Only the two D-9 cases march (one small 2-D and one small 1-D);
# everything else is a setup-time raise or a pure helper call.
# =====================================================================================================

_D9_FIX = dict(period_x_m=100e-9, lambda_min_m=1.2e-6, lambda_max_m=1.5e-6, resolution=16)


def _mo_layers():
    """Minimal duck-typed magneto-optic layer list for solve_fdtd_3d_mo's setup-only guards."""
    from types import SimpleNamespace
    return [SimpleNamespace(thickness_m=150e-9, eps_xx=4.0, eps_yy=4.0, eps_zz=4.0,
                            drude_wp_rad_s=0.0, drude_gamma_rad_s=0.0, cyclotron_wc_rad_s=0.0)]


def test_d9_flux_ratios_are_signed_and_bit_identical_for_physical_signs():
    """audit D-9: R_flux/T_flux took abs() of BOTH the signed Poynting integral and the reference,
    so a physically impossible result -- net backward power at the exit plane, or net FORWARD power
    in the scattered field at the entrance plane -- came back positive and plausible. The signed
    convention (R = -P_refl/P_inc, T = P_trans/P_inc) is the SAME NUMBER bit-for-bit when the signs
    are physical (IEEE negation is exact), and negative when they are not."""
    from dynameta.optics.fdtd_nd.solve2d import _flux_ratios
    band = np.ones(5, dtype=bool)
    P_inc = np.full(5, 3.0)
    P_refl = np.full(5, -0.75)                      # physical: the scattered field goes BACKWARD
    P_trans = np.full(5, 2.25)                      # physical: the exit plane goes FORWARD
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # no warning on a healthy run
        R, T = _flux_ratios("probe", P_inc, P_refl, P_trans, band)
    assert np.array_equal(R, np.abs(P_refl) / np.abs(P_inc))     # bit-identical to the legacy form
    assert np.array_equal(T, np.abs(P_trans) / np.abs(P_inc))
    # flip the transmitted flux: the legacy form hid it as +0.75, the signed one shows -0.75 and warns
    with pytest.warns(RuntimeWarning, match="impossible SIGN"):
        _R2, T2 = _flux_ratios("probe", P_inc, P_refl, -P_trans, band)
    assert np.all(T2 < 0) and np.allclose(np.abs(T2), np.abs(T))
    with pytest.warns(RuntimeWarning, match="impossible SIGN"):
        R3, _T3 = _flux_ratios("probe", P_inc, -P_refl, P_trans, band)
    assert np.all(R3 < 0)
    # sign noise on a ~0 bin must NOT warn (the tolerance exists for exactly that)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _flux_ratios("probe", P_inc, np.full(5, 1e-9), P_trans, band)


def test_d9_shipped_2d_flux_is_unchanged_and_carries_the_physical_signs():
    """End-to-end D-9: on a healthy lossless slab every in-band R_flux/T_flux is bit-for-bit what
    the abs()/abs() form returned, recomputed here from the public time trace."""
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_nd import solve_fdtd_2d
    from dynameta.optics.fdtd_nd.results import _flux
    r = solve_fdtd_2d([FDTDLayer(150e-9, eps_inf=4.0)], return_time_trace=True, **_D9_FIX)
    tt = r.time_trace
    P_inc = _flux(tt["incident_left"], tt["incident_left_hx"], tt["dt"])
    P_refl = _flux(tt["reflected"], tt["reflected_hx"], tt["dt"])
    P_trans = _flux(tt["transmitted"], tt["transmitted_hx"], tt["dt"])
    b = r.band
    assert np.all(P_inc[b] > 0.0) and np.all(P_refl[b] < 0.0) and np.all(P_trans[b] > 0.0)
    assert np.array_equal(r.R_flux[b], np.abs(P_refl[b]) / np.abs(P_inc[b]))
    assert np.array_equal(r.T_flux[b], np.abs(P_trans[b]) / np.abs(P_inc[b]))
    assert float(np.max(np.abs(r.R_flux[b] + r.T_flux[b] - 1.0))) < _D3_BUDGET_TOL
    # OUT-OF-BAND (audit D-9 residual, now documented on the public arrays in fdtd_nd/results.py):
    # outside `band` the incident reference carries no power, so these are ratios of two noise
    # numbers and about half of them come back NEGATIVE where abs() used to disguise them as small
    # positives. That is expected -- the noise is unchanged, only its presentation -- and every
    # in-repo consumer masks with `band`. Pin BOTH halves so neither can drift silently.
    oob = (~b) & np.isfinite(r.R_flux) & np.isfinite(r.T_flux)
    assert int(oob.sum()) > 100
    frac = float(np.mean(r.R_flux[oob] < 0.0))
    assert 0.2 < frac < 0.8, frac                    # measured 0.47 on this class of fixture
    assert r.R_flux[b].tobytes() == np.abs(r.R_flux)[b].tobytes()      # in-band: bitwise unchanged
    assert r.T_flux[b].tobytes() == np.abs(r.T_flux)[b].tobytes()


def test_d10_ring_time_omits_raman_and_keeps_both_gain_signs():
    """audit D-10, re-measured 2026-07-26 (see _ring_time_s' docstring for the numbers).

    (1) The RAMAN pole stays out of the window rule ON PURPOSE: its polarization P_R = eps0 chi3R E Q
        is proportional to E, so Q ringing after the pulse has left reaches no probe -- measured
        tail/peak 2.3e-13 at the end of the legacy window on a chi3R E^2 = 0.09 fixture, versus
        1.8e-3 for the narrow Lorentz line the rule exists for.
    (2) The GAIN pole stays in for BOTH signs of gain_dN_m3: the ADE's homogeneous coefficients do
        not contain dN (only the drive does), and extending the window measurably HELPS an
        amplifying line (transmitted tail/peak 1.3e-4 -> 9.0e-8). What the amplifying case gets is
        a WARNING, because 18.4/dw is then not an upper bound (161x the passive residue)."""
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_nd.solve2d import _ring_time_s, _window_memory_s
    raman = FDTDLayer(150e-9, eps_inf=2.0, raman_chi3_m2_V2=1e-22,
                      raman_w_rad_s=2e13, raman_gamma_rad_s=1e13)
    assert _ring_time_s([raman]) == 0.0                      # byte-identical legacy window
    amp = FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=1e25,
                    gain_w_rad_s=1.6e15, gain_dw_rad_s=1e13)
    pas = FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=-1e25,
                    gain_w_rad_s=1.6e15, gain_dw_rad_s=1e13)
    assert _ring_time_s([amp]) == pytest.approx(18.4 / 1e13, rel=1e-12)
    assert _ring_time_s([pas]) == _ring_time_s([amp])        # the sign of dN changes no memory
    # the window-extension warning fires for both; the amplifying caveat only for dN > 0
    tau_small = (18.4 / 1e13) / 400.0                        # 200*tau << the pole memory
    with pytest.warns(RuntimeWarning) as rec:
        assert _window_memory_s("probe", [amp], tau_small) == _ring_time_s([amp])
    msgs = " | ".join(str(w.message) for w in rec)
    assert "C3-6" in msgs and "AMPLIFYING" in msgs and "D-10" in msgs
    with pytest.warns(RuntimeWarning) as rec2:
        _window_memory_s("probe", [pas], tau_small)
    msgs2 = " | ".join(str(w.message) for w in rec2)
    assert "C3-6" in msgs2 and "AMPLIFYING" not in msgs2
    # a pole whose memory is below the source window neither extends the warning nor caveats
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _window_memory_s("probe", [amp], 18.4 / 1e13) == pytest.approx(18.4 / 1e13, rel=1e-12)


def test_d10_amplifying_caveat_does_not_depend_on_which_pole_won_the_window():
    """audit D-10 residual: the amplifying caveat was gated on the gain line's OWN 18.4/dw equalling
    t_ring, so ANOTHER pole with a longer memory -- which is what actually sizes the window --
    silenced it entirely. The caveat is about whether the EXTENDED window can be trusted, and a gain
    line makes it untrustworthy regardless of which pole set the extension."""
    from dynameta.optics.fdtd import FDTDLayer
    from dynameta.optics.fdtd_nd.solve2d import _ring_time_s, _window_memory_s
    amp = FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=1e25,
                    gain_w_rad_s=1.6e15, gain_dw_rad_s=1e13)          # memory 1.84e-12 s
    lorentz = FDTDLayer(150e-9, eps_inf=2.0, lorentz_w0_rad_s=1.4e15,
                        lorentz_gamma_rad_s=1e12, lorentz_delta_eps=0.5)   # memory 1.84e-11 s: WINS
    tau_small = 1e-17
    assert _ring_time_s([lorentz, amp]) == _ring_time_s([lorentz])    # the Lorentz pole sizes it
    with pytest.warns(RuntimeWarning) as rec:
        _window_memory_s("probe", [lorentz, amp], tau_small)
    msgs = " | ".join(str(w.message) for w in rec)
    assert "C3-6" in msgs and "AMPLIFYING" in msgs and "D-10" in msgs
    # an ABSORBING line winning the max likewise must not silence a present amplifier
    absorb = FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=-1e25,
                       gain_w_rad_s=1.6e15, gain_dw_rad_s=1e12)
    with pytest.warns(RuntimeWarning) as rec2:
        _window_memory_s("probe", [absorb, amp], tau_small)
    assert "AMPLIFYING" in " | ".join(str(w.message) for w in rec2)
    # ... and with two amplifiers the message names the NARROWEST (longest-memory) one
    amp2 = FDTDLayer(150e-9, eps_inf=2.0, gain_kappa_C2_kg=1e-8, gain_dN_m3=1e25,
                     gain_w_rad_s=1.6e15, gain_dw_rad_s=1e12)
    with pytest.warns(RuntimeWarning) as rec3:
        _window_memory_s("probe", [amp, amp2], tau_small)
    assert "1.000e+12" in " | ".join(str(w.message) for w in rec3)
    # no gain line anywhere -> still no caveat (no false positive from dropping the filter)
    with pytest.warns(RuntimeWarning) as rec4:
        _window_memory_s("probe", [lorentz], tau_small)
    assert "AMPLIFYING" not in " | ".join(str(w.message) for w in rec4)


def test_d11_courant_guard_is_enforced_at_every_front_end():
    """audit D-11: every solve_* docstring promises `courant` <= 1, but only the nonuniform 1-D
    branch enforced it -- `courant=2.0` marched to overflow-driven garbage. One shared guard
    (spec.courant_guard) now raises at all EIGHT entry points, at SETUP (no march runs here)."""
    from dynameta.optics.fdtd import FDTDLayer, run_uniform_time_boundary, solve_fdtd_1d
    from dynameta.optics.fdtd_mo import MOLayer, solve_fdtd_mo_1d
    from dynameta.optics.fdtd_nd import (solve_fdtd_2d, solve_fdtd_2d_oblique, solve_fdtd_3d,
                                         solve_fdtd_3d_mo, solve_fdtd_3d_oblique)
    from dynameta.optics.fdtd_nd.spec import courant_guard
    lay = [FDTDLayer(150e-9, eps_inf=4.0)]
    band = dict(lambda_min_m=1.2e-6, lambda_max_m=1.5e-6, resolution=8)
    calls = [
        lambda: solve_fdtd_1d(lay, courant=2.0, **band),
        lambda: solve_fdtd_1d(lay, courant=2.0, refine={0: 3}, n_pad_wave=1.0, **band),
        lambda: solve_fdtd_mo_1d([MOLayer(thickness_m=120e-9, eps_xx=4.0, eps_yy=4.0)],
                                 courant=2.0, n_pad_wave=1.0, **band),
        lambda: solve_fdtd_2d(lay, period_x_m=100e-9, courant=2.0, **band),
        lambda: solve_fdtd_2d_oblique(lay, period_x_m=100e-9, angle_deg=10.0, courant=2.0, **band),
        lambda: solve_fdtd_3d(lay, period_x_m=100e-9, period_y_m=100e-9, courant=2.0, **band),
        lambda: solve_fdtd_3d_oblique(lay, period_x_m=100e-9, period_y_m=100e-9, angle_deg=10.0,
                                      courant=2.0, **band),
        lambda: solve_fdtd_3d_mo(_mo_layers(), period_x_m=100e-9, period_y_m=100e-9, courant=2.0,
                                 **band),
        lambda: run_uniform_time_boundary(index_of_t=lambda t: 1.5, n_init=1.5,
                                          lambda_med_m=1e-6, courant=2.0),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="Courant"):
            call()
    # the bound itself: 1.0 is accepted (with rounding slack), 0 / negative / NaN are not
    assert courant_guard("probe", 1.0) == 1.0
    assert courant_guard("probe", 1.0 + 1e-12) == pytest.approx(1.0)
    for bad in (0.0, -0.5, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="courant"):
            courant_guard("probe", bad)


def test_d11_the_eighth_front_end_fdtd_mo_1d_is_guarded_too():
    """audit D-11 residual: `fdtd_mo.solve_fdtd_mo_1d` also takes `courant` and builds
    dt = courant*dz/c, so S = c dt/dz = courant there as well -- but the wave-5 rollout counted six
    sites and missed it. Unguarded, courant=1.05 marched an unstable leapfrog and returned, with NO
    raise, R/T that are 100% NaN over all 2134 frequency bins and an EMPTY band mask; courant=2.0
    exploded far enough that the emptied band surfaced as numpy's confusing 'zero-size array to
    reduction operation fmax' instead of anything naming the CFL bound."""
    from dynameta.optics.fdtd_mo import MOLayer, solve_fdtd_mo_1d
    lay = [MOLayer(thickness_m=120e-9, eps_xx=4.0, eps_yy=4.0)]
    band = dict(lambda_min_m=1.2e-6, lambda_max_m=1.5e-6, resolution=6, n_pad_wave=1.0)
    for bad in (1.05, 2.0, 0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="(?i)courant"):
            solve_fdtd_mo_1d(lay, courant=bad, **band)
    # the guard names THIS entry point, and runs before any march (a marching 1.05 took seconds)
    import time
    t0 = time.time()
    with pytest.raises(ValueError, match="solve_fdtd_mo_1d"):
        solve_fdtd_mo_1d(lay, courant=1.05, **band)
    assert time.time() - t0 < 1.0
    # a legal courant still solves, and gives a finite in-band R/T (no regression)
    r = solve_fdtd_mo_1d(lay, courant=0.5, **band)
    assert np.all(np.isfinite(r.R[r.band])) and np.all(np.isfinite(r.T[r.band]))


def test_d12_kerr_refuses_to_compose_with_the_eps_inf_temporal_boundary():
    """audit D-12: `_run_tv`'s D-preserving rescale E_new = E_old*eps_old/eps_new is exact for
    D = eps0 eps_inf E + P_Drude but ignores the Kerr term -- with chi3 != 0 the true D continuity
    is a CUBIC in E_new, so the linear rescale injects/destroys energy ~ chi3 E^2 per jump. It used
    to compose SILENTLY. drude_of_t is unaffected (J is continuous, no field jump), and so is a
    Kerr layer that is not the time-varying one."""
    from dynameta.optics.fdtd import FDTDLayer, solve_fdtd_1d
    band = dict(lambda_min_m=1.2e-6, lambda_max_m=1.5e-6, resolution=8, n_pad_wave=1.0)
    kerr_layer = FDTDLayer(150e-9, eps_inf=4.0, chi3_m2_V2=1e-20)
    plain = FDTDLayer(150e-9, eps_inf=4.0)
    with pytest.raises(NotImplementedError, match="D-12"):
        solve_fdtd_1d([kerr_layer], kerr=True, eps_inf_of_t=lambda t: 4.5, **band)
    # kerr=True but chi3 == 0 on the time-varying layer is fine (nothing to get wrong)
    solve_fdtd_1d([plain], kerr=True, eps_inf_of_t=lambda t: 4.5, **band)
    # the Kerr layer elsewhere in the stack is fine: no temporal boundary is applied to it
    solve_fdtd_1d([plain, kerr_layer], kerr=True, eps_inf_of_t=lambda t: 4.5,
                  time_varying_layer=0, **band)
    with pytest.raises(NotImplementedError, match="D-12"):
        solve_fdtd_1d([plain, kerr_layer], kerr=True, eps_inf_of_t=lambda t: 4.5,
                      time_varying_layer=1, **band)
    # drude_of_t + kerr still composes (documented as the supported combination)
    solve_fdtd_1d([kerr_layer], kerr=True, drude_of_t=lambda t: (0.0, 0.0), **band)


def test_d12_refusal_fires_on_the_JUMP_not_on_the_mere_presence_of_the_hook():
    """audit D-12 residual: the guard ran once at entry, so it also refused a STATIC eps_inf_of_t --
    the hook that returns the layer's own eps_inf, which this module documents as a strict no-op and
    gate 1 pins as BIT-IDENTICAL to the no-hook march. No temporal boundary is applied on that path,
    so there is nothing to mis-rescale and nothing to refuse. The refusal now lives in the branch
    where the jump actually fires."""
    from dynameta.optics.fdtd import FDTDLayer, solve_fdtd_1d
    band = dict(lambda_min_m=1.2e-6, lambda_max_m=1.5e-6, resolution=8, n_pad_wave=1.0)
    kerr_layer = FDTDLayer(150e-9, eps_inf=4.0, chi3_m2_V2=1e-20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = solve_fdtd_1d([kerr_layer], kerr=True, **band)               # no hook at all
        stat = solve_fdtd_1d([kerr_layer], kerr=True, eps_inf_of_t=lambda t: 4.0, **band)
    assert stat.R.tobytes() == base.R.tobytes()                             # BYTE-identical
    assert stat.T.tobytes() == base.T.tobytes()
    assert np.array_equal(stat.band, base.band)
    # a hook that sits still and then JUMPS still raises -- at the jump, naming both eps values
    t_jump = [None]

    def late(tt):
        if t_jump[0] is None:
            t_jump[0] = tt
        return 4.0 if tt < t_jump[0] + 1e-15 else 4.5

    with pytest.raises(NotImplementedError, match="D-12"):
        solve_fdtd_1d([kerr_layer], kerr=True, eps_inf_of_t=late, **band)
    with pytest.raises(NotImplementedError, match=r"4 -> 4\.5"):
        solve_fdtd_1d([kerr_layer], kerr=True, eps_inf_of_t=lambda t: 4.5, **band)
