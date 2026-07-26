"""Fast (no-FDTD-run) unit tests for the FDTD OpticalSolver seam helpers: the complex-eps -> FDTDLayer
Drude inversion, the Design -> layer mapping (order + guards), and the vacuum-end-media guard.

Also home to the audit D-3 source/probe-vs-CPML clearance gates. Those raise at SETUP (before any
march), so the negative cases stay no-FDTD-run; the single positive "a minimally-padded VALID config
still runs and closes the energy budget" gate is the one small 2-D march in this module."""
import math

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
# audit D-3: the soft source and BOTH R/T probe planes must clear the CPML
# =====================================================================================================
# k_src / k_pL / k_pR are placed as FRACTIONS OF THE Z PAD (0.35 pad, 0.7 pad, 0.3 pad past the
# structure) in five front ends; `npml` is a fixed CELL count that was never compared against them.
# A thin pad slides them into the absorber and R/T degrade SILENTLY -- no raise, no warning.
_D3_BAND = dict(lambda_min_m=1.2e-6, lambda_max_m=1.8e-6, resolution=20)
_D3_SLAB = dict(thickness_m=200e-9, eps_inf=4.0)


def _d3_layer():
    from dynameta.optics.fdtd import FDTDLayer
    return FDTDLayer(**_D3_SLAB)


def test_d3_thin_pad_source_in_cpml_raises_naming_the_knobs():
    """The ledger's silent-violation config now RAISES, naming n_pad_wave / resolution / npml.

    Measured with the guard bypassed (geometry held fixed at k_src=7, k_pL=15, nz=50; npml swept
    alone -- which isolates PML overlap from evanescent near-field contamination): at
    n_pad_wave=0.35, resolution=20 the lossless slab's R_flux+T_flux walks [0.9997, 0.9999]
    (npml=6, source clear) -> [0.9891, 1.0123] (npml=12, the shipped default) -> [0.8849, 0.9092]
    (npml=20, an 11.5% energy-budget violation on a LOSSLESS slab). The violation switches on
    exactly as npml overtakes k_src and grows monotonically -- PML overlap, not near-field
    contamination.
    """
    from dynameta.optics.fdtd_nd import solve_fdtd_2d
    with pytest.raises(ValueError) as exc:
        solve_fdtd_2d([_d3_layer()], period_x_m=300e-9, n_pad_wave=0.35, npml=20, **_D3_BAND)
    msg = str(exc.value)
    assert "solve_fdtd_2d" in msg and "k_src" in msg and "CPML" in msg
    for knob in ("n_pad_wave", "resolution", "npml"):        # every knob that can fix it is named
        assert knob in msg


def test_d3_guard_covers_every_front_end():
    """All five front ends that place probes as pad fractions are guarded (audit D-3 listed four
    sites plus the 2-D oblique twin). Each raises at SETUP, so no march runs here."""
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


def test_d3_right_probe_clearance_is_checked_too():
    """The high-z end is guarded symmetrically: the helper refuses k_pR within 2 cells of the
    upper CPML, not just the source / left probe."""
    from dynameta.optics.fdtd_nd.solve2d import _PROBE_CLEARANCE, _check_probe_placement
    ok = dict(nz=200, npml=12, pad=1.0e-6, dz=1.0e-8, n_pad_wave=1.0, resolution=20)
    _check_probe_placement("t", 35, 70, 130, **ok)           # all three clear -> no raise
    with pytest.raises(ValueError, match="right"):           # k_pR too close to the far CPML
        _check_probe_placement("t", 35, 70, 200 - 1 - 12 - _PROBE_CLEARANCE + 1, **ok)
    with pytest.raises(ValueError, match="left"):
        _check_probe_placement("t", 35, 12 + _PROBE_CLEARANCE - 1, 130, **ok)


def test_d3_minimally_padded_valid_config_still_runs_and_closes():
    """A MINIMALLY padded but valid config is not over-refused, and the minimum the error message
    recommends actually works: at resolution=20, npml=12 the guard asks for n_pad_wave >= 0.67, and
    that run closes max|R_flux+T_flux-1| = 2.1e-5 (documented lossless-slab tolerance: 2e-3)."""
    from dynameta.optics.fdtd_nd import solve_fdtd_2d
    r = solve_fdtd_2d([_d3_layer()], period_x_m=300e-9, n_pad_wave=0.67, npml=12, **_D3_BAND)
    m = r.band
    assert np.any(m)
    assert float(np.max(np.abs(r.R_flux[m] + r.T_flux[m] - 1.0))) < 2e-3
    assert float(np.max(np.abs(r.R0[m] + r.T0[m] - 1.0))) < 2e-3


def test_d3_empty_band_mask_raises_instead_of_silent_zeros():
    """audit D-3 sub-mode: an empty well-excited band used to be returned silently, and every
    downstream `result.R0[result.band].min()` then died with an opaque `zero-size array to
    reduction operation minimum` far from the cause."""
    from dynameta.optics.fdtd_nd.solve2d import _check_band
    good = np.array([False, True, True, False])
    assert _check_band("solve_fdtd_2d", good, 1.0e14, 2.5e14) is None
    with pytest.raises(ValueError) as exc:
        _check_band("solve_fdtd_2d", np.zeros(2617, dtype=bool), 1.0e14, 2.5e14)
    msg = str(exc.value)
    assert "EMPTY" in msg and "2617" in msg and "D-3" in msg
    assert "npml" in msg                                     # points at the actual cause


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
