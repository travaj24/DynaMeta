"""Regression gates for the 2-D TM finite-element solver: identical lateral columns, the lateral-axis guard, the
sweep claddings and the native-carrier-grid flag. Needs ngsolve.

Identical lateral columns (the bridge delivers every carrier film with 256 of them) were treated as a
laterally structured grid, resampled at the finest native z spacing and, past max_grid_lines, integrated
without mesh lines (an AdvisoryWarning, and a different r). They now
collapse to the native-z path (core.lateral, the test every route shares).
A lateral axis in the wrong unit (metres read as nm) or covering part of the cell gave the uniform-film answer
silently; it is refused with the guard every route shares (core.lateral.check_periodic_axis).
Sweep claddings: the sweep path ignored the claddings passed in and re-derived the design's, so solve_sweep and
the per-wavelength call disagreed for a caller-supplied cladding.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.optics.fem_2d_tm import GriddedEps2D, TMLayer, solve_fem_2d_tm     # noqa: E402

LAM = 1200e-9


def test_lateral_axis_unit_and_span_are_checked():
    """A Gaussian strip of a lossy metal-like eps across a 400 nm cell. Before the guard the metre axis read as
    nm and the quarter-period axis both returned the uniform-film r, -0.183018-0.003697j at 1200 nm, against -0.197410-0.061054j
    for the correct axis."""
    P = 400e-9
    x = np.arange(64) * P / 64
    prof = 2.0 + (-5.0 + 1.0j) * np.exp(-((x - P / 2) / 60e-9) ** 2)
    vals = np.tile(prof, (5, 1))
    z = np.linspace(0.0, 20e-9, 5)

    def run(xm):
        layers = [TMLayer("spacer", 50e-9, 2.1), TMLayer("film", 20e-9, GriddedEps2D(values=vals, z_m=z, x_m=xm))]
        return solve_fem_2d_tm(P, layers, lambda_m=LAM, n_super=1.0, n_sub=1.45, polarization="x", order=3)
    good = run(x).r
    with pytest.raises(ValueError, match="METRES"):
        run(x * 1e-9)
    with pytest.raises(ValueError, match="span"):
        run(x * 0.25)
    flat = solve_fem_2d_tm(P, [TMLayer("spacer", 50e-9, 2.1), TMLayer("film", 20e-9, complex(prof[0]))],
                           lambda_m=LAM, n_super=1.0, n_sub=1.45, polarization="x", order=3).r
    assert abs(good - flat) > 1e-2                                        # the strip matters


def test_identical_columns_take_the_native_z_path():
    """256 identical columns over a finely graded native z grid (finest spacing 0.005 nm, so the old
    structured resample needed > max_grid_lines lines): same r as the (Nz,) profile, bit for bit, and no
    warning. Before the fix: AdvisoryWarning ('needs more than max_grid_lines') and a different r."""
    tf = 8e-9                                                   # film thickness
    steps = 0.005e-9 * 1.2 ** np.arange(40)
    half = np.concatenate([[0.0], np.cumsum(steps)])
    half = half[half < 0.5 * tf]
    z = np.unique(np.concatenate([half, tf - half[::-1], [0.5 * tf]]))
    eps_z = (4.4 - 4.5 * np.exp(-(tf - z) / 0.8e-9)) + 1j * (0.2 + 0.4 * np.exp(-(tf - z) / 0.8e-9))
    x = np.linspace(0.0, 300e-9, 256)
    common = dict(lambda_m=LAM, theta_deg=60.0, n_super=1.0, n_sub=1.45, polarization="p", order=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r_cols = solve_fem_2d_tm(300e-9, [TMLayer("ox", 10e-9, 4.0), TMLayer(
            "film", tf, GriddedEps2D(values=np.repeat(eps_z[:, None], x.size, axis=1), z_m=z, x_m=x))], **common).r
        r_prof = solve_fem_2d_tm(300e-9, [TMLayer("ox", 10e-9, 4.0), TMLayer(
            "film", tf, GriddedEps2D(values=eps_z, z_m=z))], **common).r
        r_cols_t = solve_fem_2d_tm(300e-9, [TMLayer("ox", 10e-9, 4.0), TMLayer(
            "film", tf, GriddedEps2D.uniaxial(np.repeat(eps_z[:, None], x.size, axis=1),
                                              np.repeat(1.3 * eps_z[:, None], x.size, axis=1), z, x))], **common).r
        r_prof_t = solve_fem_2d_tm(300e-9, [TMLayer("ox", 10e-9, 4.0), TMLayer(
            "film", tf, GriddedEps2D.uniaxial(eps_z, 1.3 * eps_z, z))], **common).r
    assert r_cols == r_prof
    assert r_cols_t == r_prof_t and abs(r_cols_t - r_cols) > 1e-4


def test_sweep_honours_the_passed_claddings():
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.optics.fem_2d_tm import make_fem_2d_tm_optical_solver
    reg = MaterialRegistry()
    for nm, e in (("air", 1.0), ("glass", 1.45 ** 2), ("f", 3.0 + 0.2j)):
        reg.add(Material(nm, ConstantOptical(complex(e))))
    d = Design(name="s", unit_cell=UnitCell.square(300e-9),
               stack=Stack(layers=[Layer("f", 40e-9, "f")], superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=reg, optical=OpticalSpec(polarization="p", incidence_angle_deg=30.0))
    s = make_fem_2d_tm_optical_solver(order=3)
    one = s(d, None, {}, LAM, 1.0, 1.7).r
    swept = s.solve_sweep(d, None, lambda lam: {}, [LAM], 1.0, 1.7)[0].r
    default = s(d, None, {}, LAM, 1.0, 1.45).r
    assert swept == one and abs(one - default) > 1e-3
    assert s.native_carrier_grid is True               # takes the native carrier grid (see the cross-route test)


def test_hydro_conductor_identical_columns_are_laterally_uniform():
    from dynameta.constants import M_E
    from dynameta.optics.fem_2d_tm_hydro import HydroConductor2D
    z = np.linspace(0.0, 8e-9, 21)
    n1 = 3e26 * (1.0 + np.exp(-(8e-9 - z) / 1e-9))
    x = np.linspace(0.0, 300e-9, 16, endpoint=False)
    hc = HydroConductor2D(np.repeat(n1[:, None], x.size, axis=1), z, 0.30 * M_E, 4.0, 1.2e14, x_m=x)
    assert not hc.laterally_structured and hc._n2d().shape == (21, 1)
    hs = HydroConductor2D(np.repeat(n1[:, None], x.size, axis=1) * (1 + 0.1 * np.cos(2 * np.pi * x / 300e-9)),
                          z, 0.30 * M_E, 4.0, 1.2e14, x_m=x)
    assert hs.laterally_structured
