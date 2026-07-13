"""Skip-gated DEVSIM smoke tests (audit 1.2/6.3): the DEVSIM solver modules previously had
ZERO pytest presence -- an API/install break surfaced only tens of minutes into a manual
`make validate`, unlike the NGSolve drivers which got test_fem_drivers.py. These smokes
build the smallest proven device (the uniform bar of validation/contact_current_drivers)
and run one equilibrium + one unipolar-DD solve end to end; they self-skip without devsim
(matching the CI legs, where the heavy stack is absent) and take seconds where present.
Correctness pins live in the validation gates -- these only guard 'builds, solves,
returns finite fields'."""
import numpy as np
import pytest

devsim = pytest.importorskip("devsim")

P = 300e-9
T_SI = 40e-9
ND = 1.0e24


def _const(v):
    return lambda n: np.full_like(np.asarray(n, dtype=float), v)


def _bar_design(physics):
    from dynameta.constants import M_E
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    reg.add(Material("Si_bar", ConstantOptical(12.0 + 0j),
                     transport=TransportModel(n_bg_m3=ND, eps_static=11.7,
                                              dos_mass_kg_of_n_m3=_const(1.08 * M_E),
                                              band_gap_eV=1.12, chi_eV=4.05, physics=physics,
                                              mobility_m2Vs_of_n_m3=_const(0.135))))
    stack = Stack(layers=[Layer("si", T_SI, "Si_bar")], superstrate_material="air",
                  substrate_material="air")
    electrodes = [Electrode("anode", "si", "x_lo", role="biased"),
                  Electrode("cathode", "si", "x_hi", role="ground", fixed_voltage_V=0.0)]
    return Design(name="smoke_" + physics, unit_cell=UnitCell.square(P), stack=stack,
                  electrodes=electrodes, materials=reg, mesh_2d=Mesh2DSpec())


def _solve(physics, bias_V, tag):
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    b = LayeredDevsimBuilder(_bar_design(physics), mesh_name="smk_m_" + tag,
                             device_name="smk_d_" + tag)
    try:
        return b.solve(BiasPoint({"anode": bias_V}, tag))
    finally:
        try:
            devsim.delete_device(device=b.device)
            devsim.delete_mesh(mesh=b.mesh_name)
            _R.clear(b.device)
        except Exception:
            pass


def _finite_fields(cf):
    assert "si" in cf.regions
    fields = cf.regions["si"].grid_fields
    assert fields, "no grid fields extracted"
    for key, arr in fields.items():
        a = np.asarray(arr, dtype=float)
        assert np.all(np.isfinite(a)), "non-finite values in field {!r}".format(key)
    return fields


def test_devsim_equilibrium_smoke():
    fields = _finite_fields(_solve("equilibrium", 0.0, "eq"))
    if "n_m3" in fields:            # a flat neutral bar sits at its background density
        n = np.asarray(fields["n_m3"], dtype=float)
        assert 0.5 * ND < float(np.median(n)) < 2.0 * ND


def test_devsim_dd_smoke():
    fields = _finite_fields(_solve("drift_diffusion", 0.01, "dd"))
    if "n_m3" in fields:
        assert float(np.max(np.asarray(fields["n_m3"], dtype=float))) > 1e20
