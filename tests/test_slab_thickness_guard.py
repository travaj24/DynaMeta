"""Slab-thickness guard for gridded EpsFields (v0.16).

`slice_eps_field(ef, metres_per_unit)` multiplies the field's z axis by `metres_per_unit` (the
layered extractors pass 1e-9: axes in nm) and never compares the slab total with the layer it
fills. On v0.15.1 a field whose axes are in metres therefore became slabs 1e9 times too thin and
the layer silently vanished from the optical stack; a laterally structured field with a single z
sample sliced to ZERO slabs (the layer vanished entirely). MEASURED on v0.15.1 (lumenairy 5.50.2):
the generic strip cell below (planar film + 40 nm metal strip grating, 1200 nm, x pol) through the
PMM bridge at degree 24 gives r = -0.234953-0.460540i for the nm-axis graded field and
r = -0.849411+0.035236i, with no warning, for the same field with metre axes (10 slabs totalling
6.0e-18 m for a 6e-9 m layer). The planar film through TMM / Berreman: see the per-test docstrings.

`dynameta.core.layered.slice_layer_eps_field` is the one guard; every layered extractor that slices
a gridded field for a design layer calls it. The PMM and RCWA bridges' own cases live with those
bridges' tests. No DEVSIM.
"""
import importlib.util

import numpy as np
import pytest

from dynameta.core.eps_field import EpsField
from dynameta.core.layered import (LayeredStack, slice_eps_field, slice_layer_eps_field)
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import (M_E, ConstantOptical, DrudeOptical, Material, MaterialRegistry)
from dynameta.optics.fdtd_seam import design_to_fdtd_layers
from dynameta.optics.tmm_reference import (TmmLayeredSolver, end_media_indices,
                                           layered_stack_from_design, make_layered_tmm_solver)

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None
needs_lum = pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")

LAM, P, W, N_BG = 1200e-9, 400e-9, 150e-9, 3.0e26
DRUDE = DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)
FILM = [("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"), ("lo_oxb", 6e-9, "oxB"),
        ("film", 6e-9, "TCO"), ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA")]
ZN = np.linspace(0.0, 6.0, 11)                         # nm: the 6 nm film


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


def _eps_film(n):
    return complex(DRUDE.eps(LAM, n_m3=n))


def planar_design():
    """The film alone (laterally uniform): the TMM / Berreman / FDTD / PMM2D fixture."""
    layers = [Layer(n, t, m) for n, t, m in FILM]
    return Design(name="film", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=_mats(),
                  optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0))


def strip_design():
    """The film under a 40 nm metal strip grating (the PMM / RCWA fixture)."""
    layers = [Layer(n, t, m) for n, t, m in FILM]
    layers.append(Layer("grating", 40e-9, "air",
                        inclusions=[Inclusion(Rectangle(P / 2, P / 2, W, P), "metal")]))
    return Design(name="strip", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=_mats(),
                  optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0))


def graded_field(z):
    """A z-graded, laterally uniform field (an accumulation profile at the top face)."""
    zz = np.asarray(z, dtype=float)
    zn = (zz - zz[0]) / (zz[-1] - zz[0]) * 6.0 if zz.size > 1 else zz * 0.0
    eg = np.asarray([_eps_film(N_BG * (1 + 3 * np.exp(-(6 - q) / 0.6))) for q in zn], complex)
    return EpsField(x_axis_u=np.array([0.0, P * 1e9]), y_axis_u=np.array([0.0]), z_axis_u=zz,
                    values_zyx=np.repeat(eg[:, None, None], 2, axis=2))


def structured_field(z):
    """A laterally structured field (two x samples with different eps) on the given z axis."""
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    v = np.empty((zz.size, 1, 2), dtype=complex)
    v[:, 0, 0] = _eps_film(N_BG)
    v[:, 0, 1] = _eps_film(2.0 * N_BG)
    return EpsField(x_axis_u=np.array([0.0, P * 1e9]), y_axis_u=np.array([0.0]), z_axis_u=zz,
                    values_zyx=v)


FILM_LAYER = Layer("film", 6e-9, "TCO")
GUARD = "slice_layer_eps_field"


# ---------------------------------------------------------------------------------------------
# the helper itself
# ---------------------------------------------------------------------------------------------
def test_helper_returns_slice_eps_field_slabs_unchanged():
    """Consistency check: a correctly dimensioned field gives exactly slice_eps_field's slabs."""
    for ef in (graded_field(ZN), structured_field(ZN)):
        for n in (None, 3):
            a = slice_eps_field(ef, 1e-9, n_slices=n)
            b = slice_layer_eps_field(ef, FILM_LAYER, 1e-9, n_slices=n)
            assert len(a) == len(b)
            for sa, sb in zip(a, b):
                assert sa.thickness_m == sb.thickness_m
                assert sa.eps == sb.eps
                if sa.eps_cell is not None:
                    assert np.array_equal(sa.eps_cell, sb.eps_cell)


def test_metre_axes_raise_with_layer_thicknesses_and_cause():
    with pytest.raises(ValueError) as ei:
        slice_layer_eps_field(graded_field(ZN * 1e-9), FILM_LAYER, 1e-9)
    msg = str(ei.value)
    assert "'film'" in msg and "6e-09 m" in msg and "6e-18 m" in msg
    assert "ratio 1e-09" in msg and "METRES" in msg


def test_reverse_unit_slip_names_the_reverse_cause():
    # nm axes read as metres (metres_per_unit=1): ratio 1e9
    with pytest.raises(ValueError, match=r"ratio 1e\+09.*looks like it is in nm"):
        slice_layer_eps_field(graded_field(ZN), FILM_LAYER, 1.0)


def test_wrong_span_raises_with_generic_cause():
    with pytest.raises(ValueError, match=r"ratio 2,.*does not cover the layer"):
        slice_layer_eps_field(graded_field(2.0 * ZN), FILM_LAYER, 1e-9)


def test_rtol_boundary():
    """1e-7 relative excess passes at the default rtol=1e-6; 1e-5 raises; rtol widens it."""
    slice_layer_eps_field(graded_field(ZN * (1 + 1e-7)), FILM_LAYER, 1e-9)
    with pytest.raises(ValueError, match=GUARD):
        slice_layer_eps_field(graded_field(ZN * (1 + 1e-5)), FILM_LAYER, 1e-9)
    slice_layer_eps_field(graded_field(ZN * (1 + 1e-5)), FILM_LAYER, 1e-9, rtol=1e-4)


def test_n_slices_staircase_passes_and_sums_to_the_span():
    for ef in (graded_field(ZN), structured_field(ZN)):
        sl = slice_layer_eps_field(ef, FILM_LAYER, 1e-9, n_slices=7)
        assert len(sl) == 7
        assert sum(s.thickness_m for s in sl) == pytest.approx(6e-9, rel=1e-12, abs=0)


def test_descending_and_offset_axes_pass():
    # the slicer normalizes a descending axis; an axis offset (z in [100, 106] nm) keeps the span
    slice_layer_eps_field(graded_field(ZN[::-1]), FILM_LAYER, 1e-9)
    slice_layer_eps_field(graded_field(ZN + 100.0), FILM_LAYER, 1e-9)


def test_single_z_sample_raises_for_structured_and_uniform_fields():
    """v0.15.1: the structured field with one z sample sliced to ZERO slabs (no error, layer gone);
    the laterally uniform one raised slice_profile's generic length error."""
    assert slice_eps_field(structured_field([3.0]), 1e-9) == []          # the v0.15.1 hole
    for ef in (structured_field([3.0]), graded_field(np.array([3.0]))):
        with pytest.raises(ValueError, match=r"1 z sample\(s\).*no thickness"):
            slice_layer_eps_field(ef, FILM_LAYER, 1e-9)
        with pytest.raises(ValueError, match=r"1 z sample\(s\)"):
            slice_layer_eps_field(ef, FILM_LAYER, 1e-9, n_slices=4)


def test_uniform_field_still_refused():
    with pytest.raises(ValueError, match="uniform scalar"):
        slice_layer_eps_field(EpsField(scalar=2.0 + 0j), FILM_LAYER, 1e-9)


# ---------------------------------------------------------------------------------------------
# call sites owned here: TMM, Berreman, PMM2D, FDTD seam
# ---------------------------------------------------------------------------------------------
def test_tmm_metre_axis_field_raises():
    """v0.15.1: the metre-axis field solved silently, r = -0.9225736-0.3431813i (the nm-axis field:
    r = -0.8970662-0.4040100i); |dr| = 0.066."""
    d = planar_design()
    with pytest.raises(ValueError, match=GUARD):
        layered_stack_from_design(d, LAM, eps_by_region={"film": graded_field(ZN * 1e-9)})
    with pytest.raises(ValueError, match=GUARD):
        make_layered_tmm_solver()(d, None, {"film": graded_field(ZN * 1e-9)}, LAM, None, None)


def test_tmm_structured_single_z_sample_raises():
    """v0.15.1: zero slabs -- the film layer silently disappeared from the TMM stack."""
    d = planar_design()
    with pytest.raises(ValueError, match=r"1 z sample\(s\)"):
        layered_stack_from_design(d, LAM, eps_by_region={"film": structured_field([3.0])})


def test_tmm_nm_axis_field_bit_identical():
    """The guard only checks: r through make_layered_tmm_solver equals the v0.15.1 construction
    (slice_eps_field slabs, reversed, in a hand-built stack) bit for bit; value recorded on v0.15.1."""
    d = planar_design()
    ef = graded_field(ZN)
    r_new = make_layered_tmm_solver()(d, None, {"film": ef}, LAM, None, None).r
    slabs = []
    for L in reversed(d.stack.layers):
        if L.name == "film":
            slabs.extend(reversed(slice_eps_field(ef, 1e-9)))
        else:
            from dynameta.core.layered import LayeredSlab
            slabs.append(LayeredSlab(float(L.thickness_m),
                                     eps=complex(d.materials.get(L.background_material).eps(LAM))))
    n_sup, n_sub = end_media_indices(d, LAM)
    r_old = TmmLayeredSolver().solve(LayeredStack(n_sup, n_sub, slabs, P, P), LAM, d.optical).r
    assert np.array_equal(np.array([r_new]), np.array([r_old]))
    # recorded on v0.15.1 (lumenairy-independent TMM)
    assert abs(r_new - (-0.8970661516688937 - 0.4040100116590116j)) < 1e-12


@needs_lum
def test_berreman_metre_axis_field_raises():
    """v0.15.1: solved silently, r = -0.9225736-0.3431813i (nm-axis: -0.8970662-0.4040100i)."""
    from dynameta.optics.lumenairy_bridge.berreman_backend import (
        design_to_berreman_layers, make_lumenairy_berreman_solver)
    d = planar_design()
    with pytest.raises(ValueError, match=GUARD):
        design_to_berreman_layers(d, LAM, eps_by_region={"film": graded_field(ZN * 1e-9)})
    with pytest.raises(ValueError, match=GUARD):
        make_lumenairy_berreman_solver()(d, None, {"film": graded_field(ZN * 1e-9)}, LAM, None, None)
    with pytest.raises(ValueError, match=r"1 z sample\(s\)"):
        design_to_berreman_layers(d, LAM, eps_by_region={"film": structured_field([3.0])})


@needs_lum
def test_berreman_nm_axis_field_bit_identical():
    """Berreman layers for the nm-axis field equal the v0.15.1 construction (slice_eps_field slabs,
    reversed) exactly, and r is unchanged; value recorded on v0.15.1. Berreman and TMM agree on
    this isotropic planar stack (cross-route sign/phase check, 1e-9)."""
    from dynameta.optics.lumenairy_bridge.berreman_backend import (
        design_to_berreman_layers, make_lumenairy_berreman_solver)
    d = planar_design()
    ef = graded_field(ZN)
    layers, _, _, names = design_to_berreman_layers(d, LAM, eps_by_region={"film": ef})
    film = [lay for lay, nm in zip(layers, names) if nm == "film"]
    old = [(complex(s.eps), float(s.thickness_m)) for s in reversed(slice_eps_field(ef, 1e-9))]
    assert film == old
    r = make_lumenairy_berreman_solver()(d, None, {"film": ef}, LAM, None, None).r
    # recorded on v0.15.1 with lumenairy 5.50.2
    assert abs(r - (-0.8970661516688953 - 0.4040100116590120j)) < 1e-12
    r_tmm = make_layered_tmm_solver()(d, None, {"film": ef}, LAM, None, None).r
    assert abs(r - r_tmm) < 1e-9


@needs_lum
@pytest.mark.parametrize("engine", ["pure", "hybrid"])
def test_pmm2d_metre_axis_field_raises(engine):
    """v0.15.1: the stack was built with 10 slabs totalling 6e-18 m for the 6 nm layer."""
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import design_to_pmm2d_stack
    with pytest.raises(ValueError, match=GUARD):
        design_to_pmm2d_stack(planar_design(), LAM, engine=engine,
                              eps_by_region={"film": graded_field(ZN * 1e-9)})


@needs_lum
def test_pmm2d_structured_single_z_sample_raises():
    """v0.15.1 (hybrid engine): zero slabs -- the film layer silently disappeared."""
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import design_to_pmm2d_stack
    with pytest.raises(ValueError, match=r"1 z sample\(s\)"):
        design_to_pmm2d_stack(planar_design(), LAM, engine="hybrid",
                              eps_by_region={"film": structured_field([3.0])})


@needs_lum
def test_pmm2d_nm_axis_field_names_unchanged():
    """Consistency check: the nm-axis field still contributes one layer per native interval."""
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import design_to_pmm2d_stack
    _, names = design_to_pmm2d_stack(planar_design(), LAM, engine="hybrid",
                                     eps_by_region={"film": graded_field(ZN)})
    assert names.count("film") == ZN.size - 1


def test_fdtd_seam_metre_axis_field_raises():
    """v0.15.1: 10 FDTDLayers totalling 6e-18 m were emitted for the 6 nm layer."""
    with pytest.raises(ValueError, match=GUARD):
        design_to_fdtd_layers(planar_design(), LAM, eps_by_region={"film": graded_field(ZN * 1e-9)})
    with pytest.raises(ValueError, match=r"1 z sample\(s\)"):
        design_to_fdtd_layers(planar_design(), LAM,
                              eps_by_region={"film": graded_field(np.array([3.0]))})


def test_fdtd_seam_nm_axis_field_unchanged():
    """The nm-axis field gives the same FDTDLayer thicknesses as the v0.15.1 slicing."""
    layers = design_to_fdtd_layers(planar_design(), LAM, eps_by_region={"film": graded_field(ZN)})
    want = [s.thickness_m for s in reversed(slice_eps_field(graded_field(ZN), 1e-9))]
    got = [lay.thickness_m for lay in layers[2:2 + len(want)]]    # up_oxa, up_oxb, then the film
    assert got == want


# ---------------------------------------------------------------------------------------------
# native carrier grid capability flag (contract with the carrier-grid package)
# ---------------------------------------------------------------------------------------------
def test_native_carrier_grid_flag_on_tmm_and_berreman():
    assert getattr(make_layered_tmm_solver(), "native_carrier_grid", False) is True
    if HAVE_LUM:
        from dynameta.optics.lumenairy_bridge.berreman_backend import make_lumenairy_berreman_solver
        assert getattr(make_lumenairy_berreman_solver(), "native_carrier_grid", False) is True
