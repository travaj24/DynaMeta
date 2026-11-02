"""Round-trip coverage for the CarrierField Zarr serialization (core.carrier_field
dump/load) -- previously exported but exercised by NO test, so a schema drift would have
gone undetected (audit). Skipped automatically where zarr is not installed (e.g. the
default CI extra). Run: python -m pytest tests/test_carrier_field_io.py -q
"""
import numpy as np
import pytest

pytest.importorskip("zarr")

from dynameta.core.carrier_field import (CarrierField, CarrierRegion,
                                         dump_carrier_field, load_carrier_field,
                                         ELECTRON_DENSITY)


def test_carrier_field_zarr_roundtrip(tmp_path):
    z = np.linspace(0.0, 12e-9, 5)
    reg = CarrierRegion(
        name="semi", role="semiconductor", material="ito",
        nodes_m=np.zeros((3, 3)), node_fields={"potential_V": np.array([0.0, 0.1, 0.2])},
        grid_axes_m={"x": np.array([0.0, 3e-7]), "y": np.array([0.0, 3e-7]), "z": z},
        grid_fields={ELECTRON_DENSITY: np.full((2, 2, z.size), 4e26)})
    cf = CarrierField(bias_label="vg1", voltages={"gate": 1.0, "body": 0.0}, ndim=3,
                      temperature_K=300.0, regions={"semi": reg},
                      n_bg_by_region={"semi": 4e26}, unit_cell_m=(3e-7, 3e-7))
    p = dump_carrier_field(cf, tmp_path / "cf.zarr")
    out = load_carrier_field(p)

    assert out.bias_label == "vg1" and out.ndim == 3
    assert out.voltages["gate"] == pytest.approx(1.0)
    assert out.n_bg_by_region["semi"] == pytest.approx(4e26)
    assert out.unit_cell_m == pytest.approx((3e-7, 3e-7))
    r = out.regions["semi"]
    assert r.role == "semiconductor" and r.material == "ito"
    assert np.allclose(r.grid_axes_m["z"], z)
    assert np.allclose(r.grid_fields[ELECTRON_DENSITY], 4e26)
    assert np.allclose(r.node_fields["potential_V"], [0.0, 0.1, 0.2])


def test_s4_20_axis_order_is_honoured_on_load(tmp_path):
    """audit S4-20 (wave-7, 2026-11-02): dump writes the grid axis order as `axis_order`; load
    never read it, so the reloaded dict came back in zarr's key order (measured on zarr 3.1.6:
    {'y', 'x'} -> ['x', 'y']). Now the declared order is restored, and a file whose stored axes
    disagree with its declaration is refused."""
    import json
    import zarr
    reg = CarrierRegion(
        name="semi", role="semiconductor", material="ito", nodes_m=np.zeros((1, 2)),
        node_fields={}, grid_axes_m={"y": np.linspace(0.0, 1e-8, 3), "x": np.array([0.0, 3e-7])},
        grid_fields={ELECTRON_DENSITY: np.full((2, 3), 4e26)})
    cf = CarrierField(bias_label="b", voltages={}, ndim=2, temperature_K=300.0,
                      regions={"semi": reg}, n_bg_by_region={"semi": 4e26},
                      unit_cell_m=(3e-7, 3e-7))
    p = dump_carrier_field(cf, tmp_path / "ord.zarr")
    out = load_carrier_field(p)
    assert list(out.regions["semi"].grid_axes_m) == ["y", "x"]
    assert np.array_equal(out.regions["semi"].grid_axes_m["y"], reg.grid_axes_m["y"])
    zarr.open_group(str(p), mode="r+")["grid"]["semi"].attrs["axis_order"] = json.dumps(["x", "z"])
    with pytest.raises(ValueError, match="axis_order"):
        load_carrier_field(p)
