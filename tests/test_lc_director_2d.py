"""Fast unit tests for the 2-D lateral nematic director theta(x,z) in dynameta/carriers/lc_director_2d.py.
The rigorous reduces-to-1-D + fringing-field checks live in validation/lc_director_2d.py."""
import math

import numpy as np
import pytest

from dynameta.carriers.lc_director_2d import director_profile_2d
from dynameta.carriers.lc_director import director_profile_bvp

THB = math.radians(89.9)
KW = dict(K=17e-12, eps_para=18.7, eps_perp=4.0, d_planar=2e-6, theta_b_rad=THB, n_o=1.52, n_e=1.74)


def test_uniform_columns_reduces_to_1d():
    nz = 41
    r = director_profile_2d(V_top=2.0, field="uniform_columns", Lx_m=8e-6, nx=11, nz=nz, **KW)
    st = director_profile_bvp(V_app=2.0, K11=KW["K"], K33=KW["K"], eps_para=KW["eps_para"],
                              eps_perp=KW["eps_perp"], d_planar=KW["d_planar"], nz=nz,
                              theta_b_rad=THB, field_model="uniform")
    # x-independent (each column decoupled) and equal to the 1-D solver to the FD discretization floor
    assert math.degrees(np.max(np.abs(r.theta_field_rad - r.theta_field_rad[0][None, :]))) < 1e-3
    assert math.degrees(np.max(np.abs(r.theta_field_rad - st.theta_field_rad[None, :]))) < 0.05
    assert r.success


def test_laplace_uniform_matches_uniform_columns():
    nz = 41
    ru = director_profile_2d(V_top=1.8, field="uniform_columns", Lx_m=8e-6, nx=11, nz=nz, **KW)
    rl = director_profile_2d(V_top=1.8, field="laplace", Lx_m=8e-6, nx=11, nz=nz, **KW)
    assert abs(np.mean(np.abs(rl.Ez)) - 1.8 / KW["d_planar"]) / (1.8 / KW["d_planar"]) < 1e-3
    assert np.max(np.abs(rl.Ex)) / np.mean(np.abs(rl.Ez)) < 1e-3
    assert math.degrees(np.max(np.abs(rl.theta_field_rad - ru.theta_field_rad))) < 1e-2


def test_anchoring_pinned_and_boundaries():
    r = director_profile_2d(V_top=2.5, field="laplace", Lx_m=8e-6, nx=9, nz=31, **KW)
    assert np.allclose(r.theta_field_rad[:, 0], THB)
    assert np.allclose(r.theta_field_rad[:, -1], THB)
    assert r.theta_field_rad.shape == (9, 31)


def test_patterned_lateral_contrast_and_fringing():
    Lx = 24e-6

    def Vstep(xa):
        return np.where(xa < Lx / 2, 0.5, 3.0)

    r = director_profile_2d(V_top=Vstep, field="laplace", Lx_m=Lx, nx=33, nz=31, **KW)
    ilo, ihi = r.x_m.size // 4, 3 * r.x_m.size // 4
    # high-voltage pixel is tilted (smaller theta = toward homeotropic), low-voltage pixel stays planar
    assert r.theta_field_rad[ihi, r.z_m.size // 2] < r.theta_field_rad[ilo, r.z_m.size // 2] - math.radians(20.0)
    # lateral optical contrast and localized fringing field
    assert abs(r.n_eff_of_x[ihi] - r.n_eff_of_x[ilo]) > 0.1
    ib = r.x_m.size // 2
    assert np.max(np.abs(r.Ex[ib - 1:ib + 2, :])) > 1e6
    assert np.max(np.abs(r.Ex[ilo, :])) < 0.1 * np.max(np.abs(r.Ex[ib - 1:ib + 2, :]))


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        director_profile_2d(V_top=1.0, K=-1.0, eps_para=10.0, eps_perp=4.0, d_planar=2e-6, Lx_m=8e-6)
    with pytest.raises(ValueError):
        director_profile_2d(V_top=1.0, K=1e-12, eps_para=10.0, eps_perp=4.0, d_planar=2e-6, Lx_m=8e-6,
                            field="bogus")
