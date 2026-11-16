"""v0.16 fixes in the boundary-scattering (size-effect) damping: unit slips in the film thickness, and a film
split into two carrier regions.

Unit slips: BoundaryScattering(thickness_m=6) or DrudeOptical.eps_tensor(..., z_axis_m=<nm>) read the film as
metres thick and the size effect vanished silently (eps_xx with d of a few metres equals the bulk value to
1e-9). A thickness outside [1e-11, 1e-3] m is now refused.

Split film: a film split into two contiguous carrier regions of the same material (supported:
tests/_wiring_fixture split_film) gave each half the size effect of a t/2 film, because the assembler supplied
each region's own stack axis. The assembler now supplies the whole film's thickness to the parts of a split film
(FILM_THICKNESS_FIELD); a film that is one region is unchanged bit for bit.
"""
import numpy as np
import pytest

from dynameta.constants import M_E
from dynameta.materials import BoundaryScattering, ConstantOptical, DrudeOptical, Material, MaterialRegistry

LAM = 1200e-9
EPS_INF, M_OPT, GAM, N_BG, T_FILM = 4.0, 0.30 * M_E, 1.2e14, 3e26, 6e-9     # the generic gated film


def test_thickness_unit_slips_are_refused():
    with pytest.raises(ValueError, match="nm passed as metres"):
        BoundaryScattering(thickness_m=6.0)
    m = DrudeOptical(EPS_INF, M_OPT, GAM, boundary=BoundaryScattering(perp_coefficient=0.5))
    n = np.full(11, N_BG)
    with pytest.raises(ValueError, match="nm passed as metres"):
        m.eps_tensor(LAM, n_m3=n, z_axis_m=np.linspace(0.0, 6.0, 11))           # nm read as metres
    with pytest.raises(ValueError, match="converted twice"):
        m.eps_tensor(LAM, n_m3=n, z_axis_m=np.linspace(0.0, 6e-18, 11))
    ok = m.eps_tensor(LAM, n_m3=n, z_axis_m=np.linspace(0.0, T_FILM, 11))
    bulk = complex(DrudeOptical(EPS_INF, M_OPT, GAM).eps(LAM, n_m3=N_BG))
    assert abs(ok[5, 0, 0] - bulk) > 1e-3                                        # the size effect is there


def _field(regions, P=300e-9):
    from dynameta.core.carrier_field import ELECTRON_DENSITY, CarrierField, CarrierRegion
    regs = {}
    for name, z in regions:
        x = np.linspace(0.0, P, 4)
        n = np.tile(N_BG + N_BG * np.exp(-(T_FILM - z) / 0.8e-9), (x.size, 1))
        regs[name] = CarrierRegion(name=name, role="semiconductor", material="film", nodes_m=np.zeros((1, 2)),
                                   node_fields={}, grid_axes_m={"x": x, "y": z}, grid_fields={ELECTRON_DENSITY: n})
    return CarrierField(bias_label="b", voltages={}, ndim=2, temperature_K=300.0, regions=regs,
                        n_bg_by_region={k: N_BG for k in regs}, unit_cell_m=(P, P))


def _alignment(spans, P=300e-9):
    from dynameta.core import NM
    from dynameta.core.alignment import GeometryAlignment, RegionAlignment
    return GeometryAlignment(unit_scale=NM, region_alignments=[
        RegionAlignment(name, name, (0.0, P, 0.0, P, 8e-9 + a, 8e-9 + b), stack_axis="y") for name, a, b in spans],
        fixed_eps_regions={})


@pytest.mark.parametrize("perp_length", ["film", "local"])
def test_a_split_film_gets_the_whole_film_size_effect(perp_length):
    """With each region's own stack axis the two halves of a 6 nm film each use d = 3 nm, so eps_xx at the same
    depth differs from the one-region film by up to 0.080 in eps_xx and 0.082 in eps_zz (MEASURED by evaluating
    each half with its own stack axis). The whole film's thickness reaches both parts."""
    from dynameta.core import MaterialEpsMap, assemble_eps
    from dynameta.core.lift import ExtrudeLift
    reg = MaterialRegistry()
    reg.add(Material("film", DrudeOptical(EPS_INF, M_OPT, GAM, boundary=BoundaryScattering(
        specularity=0.0, perp_coefficient=0.6, perp_length=perp_length))))
    z = np.linspace(0.0, T_FILM, 41)
    lift = ExtrudeLift(period_y_m=300e-9)
    one = assemble_eps(_field([("film", z)]), _alignment([("film", 0.0, T_FILM)]), MaterialEpsMap(reg), lift, LAM)
    za, zb = z[:21], z[20:]
    two = assemble_eps(_field([("film_a", za), ("film_b", zb)]),
                       _alignment([("film_a", 0.0, 3e-9), ("film_b", 3e-9, T_FILM)]), MaterialEpsMap(reg), lift, LAM)
    v1 = np.asarray(one["film"].values_zyx)[:, 0, 0]
    va = np.asarray(two["film_a"].values_zyx)[:, 0, 0]
    vb = np.asarray(two["film_b"].values_zyx)[:, 0, 0]
    split = np.concatenate([va, vb[1:]])
    if perp_length == "film":
        assert np.allclose(split, v1, rtol=1e-12, atol=0.0)
    else:
        # 'local': the in-plane factor and the film length match; dn/dz is taken per region (one-sided at the
        # cut), so only the out-of-plane component may differ at the shared node
        assert np.allclose(split[..., 0, 0], v1[..., 0, 0], rtol=1e-12, atol=0.0)
    # two SEPARATE films of the same material (not contiguous) keep their own thickness
    sep = assemble_eps(_field([("film_a", za), ("film_b", zb)]),
                       _alignment([("film_a", 0.0, 3e-9), ("film_b", 10e-9, 13e-9)]), MaterialEpsMap(reg), lift,
                       LAM)
    assert not np.allclose(np.asarray(sep["film_a"].values_zyx)[:, 0, 0, 0, 0], va[:, 0, 0], rtol=1e-6)
    # a dielectric next to the film does not join it
    reg.add(Material("ox", ConstantOptical(2.1 + 0j)))
    assert np.array_equal(np.asarray(one["film"].values_zyx),
                          np.asarray(assemble_eps(_field([("film", z)]), _alignment([("film", 0.0, T_FILM)]),
                                                  MaterialEpsMap(reg), lift, LAM)["film"].values_zyx))
