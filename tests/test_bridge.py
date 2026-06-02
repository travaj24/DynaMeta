"""Fast, solver-free (numpy-only) unit tests for the reusable bridge spine:
assemble_eps (2D-lift and native-3D branches), GeometryAlignment.validate_coverage,
choose_lift symmetry gating, the SeparableXYLift einsum + its new preconditions,
the time-convention guard, and analysis.resonance_dip. None of this needs devsim or
ngsolve, so it is the CI gate the heavy validation/ scripts lack (audit cross-cutting
F1/F2). Run: python -m pytest tests/test_bridge.py -q
"""
import numpy as np
import pytest

from dynameta.core import NM, MaterialEpsMap, assemble_eps
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.core.carrier_field import CarrierField, CarrierRegion, ELECTRON_DENSITY
from dynameta.core.lift import IdentityLift, ExtrudeLift, SeparableXYLift, choose_lift
from dynameta.materials import Material, MaterialRegistry, ConstantOptical, DrudeOptical, M_E
from dynameta.analysis import (resonance_dip, gate_cv, sheet_resistance_ohm_sq,
                               lumped_rc_bandwidth, switching_energy_per_area)

N_BG = 4e26
PERIOD = 300e-9


def _registry():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("ito", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14)))
    return reg


def _field_2d(n2d, x_m, y_m, conv="exp(-iwt)"):
    """A minimal 2D CarrierField with one gridded 'ito' region (x lateral, y vertical)."""
    reg = CarrierRegion(name="ito", role="semiconductor", material="ito",
                        nodes_m=np.zeros((1, 2)), node_fields={},
                        grid_axes_m={"x": x_m, "y": y_m},
                        grid_fields={ELECTRON_DENSITY: n2d})
    return CarrierField(bias_label="t", voltages={}, ndim=2, temperature_K=300.0,
                        regions={"ito": reg}, n_bg_by_region={"ito": N_BG},
                        unit_cell_m=(PERIOD, PERIOD), time_convention=conv)


def _field_3d(n3d, x_m, y_m, z_m):
    reg = CarrierRegion(name="semi", role="semiconductor", material="ito",
                        nodes_m=np.zeros((1, 3)), node_fields={},
                        grid_axes_m={"x": x_m, "y": y_m, "z": z_m},
                        grid_fields={ELECTRON_DENSITY: n3d})
    return CarrierField(bias_label="t", voltages={}, ndim=3, temperature_K=300.0,
                        regions={"semi": reg}, n_bg_by_region={"semi": N_BG},
                        unit_cell_m=(PERIOD, PERIOD))


def _align(mesh_region, source, stack_axis="y", fixed=None):
    return GeometryAlignment(
        unit_scale=NM,
        region_alignments=[RegionAlignment(mesh_region, source,
                            (0.0, PERIOD, 0.0, PERIOD, 0.0, 10e-9), stack_axis=stack_axis)],
        fixed_eps_regions=fixed or {})


# ---- assemble_eps: 2D-lift (Extrude) path ----
def test_assemble_eps_2d_extrude_shape_and_bg():
    nx, nv = 16, 5
    x = np.linspace(0.0, PERIOD, nx)
    y = np.linspace(0.0, 10e-9, nv)
    n2d = np.full((nx, nv), N_BG)
    n2d[nx // 2, -1] = 2.0 * N_BG                       # a gate-side accumulation spike
    field = _field_2d(n2d, x, y)
    out = assemble_eps(field, _align("ito", "ito"), MaterialEpsMap(_registry()),
                       ExtrudeLift(period_y_m=PERIOD), 1300e-9, mesh_regions=["ito"])
    ef = out["ito"]
    assert ef.values_zyx.shape == (nv, 2, nx)           # (Nz, Ny, Nx)
    # the n_bg columns reduce to the Drude eps at n_bg (density-dependent material,
    # so the background flows through eps_grid(n_bg), not scalar_eps); the spike differs
    bg = complex(MaterialEpsMap(_registry()).eps_grid("ito", {"n": np.array([N_BG])}, 1300e-9)[0])
    assert np.isclose(ef.values_zyx[0, 0, 0], bg, rtol=1e-9)
    assert ef.values_zyx[-1, 0, nx // 2].real < bg.real  # accumulation lowers Re(eps)


# ---- assemble_eps: native-3D (Identity) path ----
def test_assemble_eps_3d_identity():
    nx, ny, nz = 4, 4, 6
    x = np.linspace(0.0, PERIOD, nx); y = np.linspace(0.0, PERIOD, ny); z = np.linspace(0.0, 10e-9, nz)
    n3d = np.full((nx, ny, nz), N_BG)
    n3d[:, :, -1] = 1.5 * N_BG                            # gate-side accumulation
    out = assemble_eps(_field_3d(n3d, x, y, z), _align("semi", "semi", stack_axis="z"),
                       MaterialEpsMap(_registry()), IdentityLift(), 1300e-9, mesh_regions=["semi"])
    v = out["semi"].values_zyx
    assert v.shape == (nz, ny, nx)
    assert v[-1].real.mean() < v[0].real.mean()          # accumulation toward ENZ at the gate side


def test_assemble_eps_3d_missing_axis_raises():
    nx, ny, nz = 4, 4, 6
    x = np.linspace(0.0, PERIOD, nx); y = np.linspace(0.0, PERIOD, ny); z = np.linspace(0.0, 10e-9, nz)
    field = _field_3d(np.full((nx, ny, nz), N_BG), x, y, z)
    del field.regions["semi"].grid_axes_m["z"]           # break the 3D axis contract
    with pytest.raises(ValueError):
        assemble_eps(field, _align("semi", "semi"), MaterialEpsMap(_registry()),
                     IdentityLift(), 1300e-9, mesh_regions=["semi"])


# ---- time-convention guard (audit F2/F7) ----
def test_assemble_eps_rejects_wrong_time_convention():
    x = np.linspace(0.0, PERIOD, 8); y = np.linspace(0.0, 10e-9, 4)
    field = _field_2d(np.full((8, 4), N_BG), x, y, conv="exp(+iwt)")
    with pytest.raises(ValueError):
        assemble_eps(field, _align("ito", "ito"), MaterialEpsMap(_registry()),
                     ExtrudeLift(period_y_m=PERIOD), 1300e-9, mesh_regions=["ito"])


# ---- validate_coverage ----
def test_validate_coverage_raises():
    al = _align("ito", "ito", fixed={"air": "air"})
    al.validate_coverage(["ito", "air"])                 # exact -> ok
    with pytest.raises(ValueError):
        al.validate_coverage(["ito"])                    # 'air' extra (not in mesh)
    with pytest.raises(ValueError):
        al.validate_coverage(["ito", "air", "pml"])      # 'pml' unmapped
    dup = GeometryAlignment(NM, al.region_alignments, {"ito": "air"})
    with pytest.raises(ValueError):
        dup.validate_coverage(["ito"])                   # 'ito' both spatial and fixed


def test_assemble_eps_internal_dup_without_mesh_regions():
    dup = GeometryAlignment(NM, _align("ito", "ito").region_alignments, {"ito": "air"})
    x = np.linspace(0.0, PERIOD, 8); y = np.linspace(0.0, 10e-9, 4)
    with pytest.raises(ValueError):
        assemble_eps(_field_2d(np.full((8, 4), N_BG), x, y), dup,
                     MaterialEpsMap(_registry()), ExtrudeLift(period_y_m=PERIOD), 1300e-9)


# ---- choose_lift gating ----
def test_choose_lift_gating():
    assert isinstance(choose_lift("c4v", "auto", period_y_m=PERIOD), SeparableXYLift)
    assert isinstance(choose_lift("none", "auto", period_y_m=PERIOD), ExtrudeLift)
    assert isinstance(choose_lift("none", "identity", period_y_m=PERIOD), IdentityLift)
    with pytest.raises(ValueError):
        choose_lift("none", "separable_xy", period_y_m=PERIOD)   # separable needs c4v


# ---- SeparableXYLift: correctness + new preconditions ----
def test_separable_xy_outer_product():
    nx, nv = 32, 3
    x = np.linspace(0.0, PERIOD, nx)
    bump = N_BG + 1e26 * np.exp(-((x - PERIOD / 2) / (PERIOD / 8)) ** 2)
    n2d = np.repeat(bump[:, None], nv, axis=1)
    lift = SeparableXYLift(period_y_m=PERIOD, ny=nx)
    n3d, xo, yo, zo = lift.apply(n2d, x, np.linspace(0, 10e-9, nv), n_bg=N_BG)
    assert n3d.shape == (nx, nx, nv)
    # center peak is the separable product of the 1D peak deviation; corners ~ n_bg
    assert n3d[nx // 2, nx // 2, 0] > n3d[0, 0, 0]
    assert np.isclose(n3d[0, 0, 0], N_BG, rtol=1e-6)


def test_separable_xy_rejects_nonsquare_and_mixed_sign():
    nx, nv = 16, 2
    x = np.linspace(0.0, PERIOD, nx)
    n2d = np.full((nx, nv), N_BG); n2d[nx // 2] = 1.2 * N_BG
    with pytest.raises(ValueError):                       # x-span != period_y -> not square
        SeparableXYLift(period_y_m=2 * PERIOD).apply(n2d, x, np.zeros(nv), n_bg=N_BG)
    mixed = np.full((nx, nv), N_BG)
    mixed[: nx // 2] = 1.3 * N_BG                          # accumulation on one side
    mixed[nx // 2:] = 0.7 * N_BG                           # depletion on the other -> mixed sign
    with pytest.raises(ValueError):
        SeparableXYLift(period_y_m=PERIOD).apply(mixed, x, np.zeros(nv), n_bg=N_BG)


# ---- analysis.resonance_dip: exact on uniform AND non-uniform grids (audit F6) ----
def test_resonance_dip_exact_vertex():
    def parab(xs):
        return [(xx - 1305.0) ** 2 * 1e-5 + 0.02 for xx in xs]
    uni = [1250.0, 1300.0, 1350.0, 1400.0]
    lam_u, _ = resonance_dip(uni, parab(uni))
    assert abs(lam_u - 1305.0) < 1e-6
    nonuni = [1250.0, 1300.0, 1370.0, 1400.0]             # the old code biased this by ~10 nm
    lam_n, val_n = resonance_dip(nonuni, parab(nonuni))
    assert abs(lam_n - 1305.0) < 1e-6
    assert abs(val_n - 0.02) < 1e-9


def test_resonance_dip_handles_nonfinite_spectrum():
    # a NaN/inf sample (e.g. a failed solve point) must NOT crash (was a NameError on the
    # undefined x1/y1 fallback, audit AN-1/CC-1/AD-1) -- drop it and find the dip among the
    # finite samples; an all-nonfinite spectrum is a clear ValueError, not a crash.
    lam = [1250.0, 1300.0, 1350.0, 1400.0]
    dip_nm, dip_val = resonance_dip(lam, [0.5, 0.04, float("nan"), 0.4])
    assert np.isfinite(dip_nm) and np.isfinite(dip_val)
    with pytest.raises(ValueError):
        resonance_dip(lam, [float("nan")] * 4)


# ---- analysis.gate_cv: DC gate charge + capacitance from a synthetic voltage sweep ----
def test_gate_cv():
    Q_E = 1.602176634e-19
    L, nz = 12e-9, 41
    z = np.linspace(0.0, L, nz)
    S = 3.0e17          # excess sheet density per volt (m^-2 / V) -> Q = q*S*Vg, C = q*S
    fields = []
    for vg in (0.0, 0.5, 1.0, 1.5):
        delta = S * vg / L                                # uniform excess so INT(n-n_bg)dz = S*vg
        n3d = np.full((2, 2, nz), N_BG + delta)
        reg = CarrierRegion(name="semi", role="semiconductor", material="ito",
                            nodes_m=np.zeros((1, 3)), node_fields={},
                            grid_axes_m={"x": np.array([0.0, PERIOD]), "y": np.array([0.0, PERIOD]), "z": z},
                            grid_fields={ELECTRON_DENSITY: n3d})
        fields.append(CarrierField(bias_label="vg", voltages={"gate": vg, "body": 0.0}, ndim=3,
                                   temperature_K=300.0, regions={"semi": reg},
                                   n_bg_by_region={"semi": N_BG}, unit_cell_m=(PERIOD, PERIOD)))
    Vg, Q, Vmid, C = gate_cv(fields, "semi", voltage_key="gate")
    assert np.all(np.diff(Q) > 0)                          # Q rises with gate bias (accumulation)
    assert np.allclose(Q, Q_E * S * Vg, rtol=1e-6)         # Q(Vg) = q*S*Vg
    assert np.allclose(C, Q_E * S, rtol=1e-6)              # C = dQ/dVg = q*S (constant here)


def test_gate_cv_rejects_duplicate_bias():
    # two CarrierFields at the SAME gate voltage make dQ/dVg = 0/0 -> NaN; gate_cv must RAISE
    # rather than return a silent NaN capacitance (audit AN-3).
    z = np.linspace(0.0, 12e-9, 9)

    def _cf(vg):
        reg = CarrierRegion(name="semi", role="semiconductor", material="ito",
                            nodes_m=np.zeros((1, 3)), node_fields={},
                            grid_axes_m={"x": np.array([0.0, PERIOD]),
                                         "y": np.array([0.0, PERIOD]), "z": z},
                            grid_fields={ELECTRON_DENSITY: np.full((2, 2, z.size), N_BG)})
        return CarrierField(bias_label="vg", voltages={"gate": vg, "body": 0.0}, ndim=3,
                            temperature_K=300.0, regions={"semi": reg},
                            n_bg_by_region={"semi": N_BG}, unit_cell_m=(PERIOD, PERIOD))
    with pytest.raises(ValueError):
        gate_cv([_cf(0.0), _cf(0.5), _cf(0.5)], "semi", voltage_key="gate")


def test_gate_cv_rejects_transposed_grid():
    # a BYO carrier that lays its density grid out transposed must be REJECTED, not silently
    # integrated against the wrong axis (audit AN-4). Distinct axis lengths make it visible.
    z = np.linspace(0.0, 12e-9, 5)
    x = np.linspace(0.0, PERIOD, 4)
    n_bad = np.full((z.size, x.size, x.size), N_BG)        # (Nz,Ny,Nx) instead of (Nx,Ny,Nz)
    reg = CarrierRegion(name="semi", role="semiconductor", material="ito",
                        nodes_m=np.zeros((1, 3)), node_fields={},
                        grid_axes_m={"x": x, "y": x, "z": z}, grid_fields={ELECTRON_DENSITY: n_bad})
    cf = CarrierField(bias_label="vg", voltages={"gate": 0.0, "body": 0.0}, ndim=3,
                      temperature_K=300.0, regions={"semi": reg},
                      n_bg_by_region={"semi": N_BG}, unit_cell_m=(PERIOD, PERIOD))
    with pytest.raises(ValueError):
        gate_cv([cf], "semi", voltage_key="gate")


# ---- lumped-RC bandwidth + switching energy (ported from Metasurface_Modulator Stage 4) ----
def test_lumped_rc_bandwidth_formula_with_modulator_C():
    # Checks the RC-bandwidth FORMULA against the Metasurface_Modulator's MEASURED areal
    # capacitance C=0.0145 F/m^2 (a fixed INPUT, not a carrier-derived reproduction -- the
    # end-to-end carrier->C->f_3dB chain is exercised in validation/bandwidth_cv.py; audit AN-2).
    # Park-cell numbers: ITO n_bg=4e26 m^-3, mu=30 cm^2/Vs, t=5 nm; 370 nm cell; medium access
    # 5 um path / 1 um pad -> ~15.4 GHz (the Modulator's result).
    rho_s = sheet_resistance_ohm_sq(4e26, 30e-4, 5e-9)
    assert abs(rho_s - 1040.0) < 5.0                       # ~1040 Ohm/sq
    C_area = 0.0145                                         # F/m^2
    cell_area = (370e-9) ** 2
    R, C_cell, f3db = lumped_rc_bandwidth(C_area, rho_s, path_length_m=5e-6,
                                          pad_width_m=1e-6, cell_area_m2=cell_area)
    assert abs(C_cell - 1.985e-15) < 1e-17                 # ~1.99 fF/cell
    assert abs(f3db / 1e9 - 15.4) < 0.3                    # ~15.4 GHz
    assert abs(f3db - 1.0 / (2 * np.pi * R * C_cell)) < 1.0  # closed form
    # tighter access (shorter path) -> higher bandwidth
    _, _, f3db_tight = lumped_rc_bandwidth(C_area, rho_s, path_length_m=0.5e-6,
                                           pad_width_m=1e-6, cell_area_m2=cell_area)
    assert f3db_tight > f3db
    # switching energy 0.5 C V^2 over an 8 V swing
    E = switching_energy_per_area(C_area, 8.0)
    assert np.isclose(E, 0.5 * C_area * 64.0)
