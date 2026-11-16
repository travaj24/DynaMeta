"""End-to-end smoke of each TCO preset through the gated-film path:

    preset Material --dual_gate_film_kwargs--> carriers.DualGateSchrodingerPoisson (solve at -1, 0, +1 V)
        --> n(z) --> Material.eps(lambda, n(z)) per grid cell --> planar coherent TMM (tmm_reference.stack_rta)

Structure: air | 10 nm preset film | 5 nm gate oxide (eps_s 20, n 2.0) | 100 nm metal back gate (eps -120+20i) |
glass, p-polarized at 60 deg: the thin ENZ film on the metal gate shows the Berreman absorption dip near its
ENZ wavelength. The bottom gate drives accumulation (+V) or depletion (-V) at the film's bottom face; the
top face is ungated.

Physically expected:
  * SIGN: accumulation raises the density -> the dip moves to SHORTER wavelength; depletion to longer.
  * ORDER OF MAGNITUDE: an independent thin-film estimate -- the peak of the energy-loss function
    sum_cells Im(-1/eps) h (the p-polarized absorption of a film much thinner than the wavelength is
    proportional to it) evaluated on the same n(z) -- predicts the shift; the TMM dip shift agrees with it
    within 25 % (measured agreement 1-10 %), and both are a few nm per volt (tens of nm for In:CdO, whose
    high static permittivity and light band give a long screening length). The gated layer is ~1 nm of a
    10 nm film, so the shift is far smaller than a uniform-film estimate would give.
  * DISCRIMINATION: the same chain with the density pinned at n_bg (gate ignored) gives no shift, and a
    mutant optical mass growing faster than the density (omega_p^2 falling with n) reverses the sign.

Run: python -m pytest tests/test_tco_gated_smoke.py -q
"""
import numpy as np
import pytest

from dynameta.materials import tco_presets as TP

T_FILM = 10e-9
T_OX, EPS_OX, N_OX = 5e-9, 20.0, 2.0
EPS_METAL = -120.0 + 20.0j
THETA = 60.0


def _cell_eps(mat, n_z, lam):
    e = np.asarray(mat.eps(lam, n_m3=n_z))
    return 0.5 * (e[1:] + e[:-1])


def _reflectance(mat, n_z, z, lam):
    from dynameta.optics.tmm_reference import stack_rta
    ec = _cell_eps(mat, n_z, lam)[::-1]                      # top of the film first (light from the air side)
    h = np.diff(z)[::-1]
    layers = [(np.sqrt(e + 0j), d) for e, d in zip(ec, h)] + [(N_OX, T_OX), (np.sqrt(EPS_METAL), 100e-9)]
    R, _T, _A = stack_rta(1.0, layers, 1.45, lam, theta_deg=THETA, pol="p")
    return R


def _extremum(lam, y, find_max):
    k = int(np.argmax(y) if find_max else np.argmin(y))
    assert 0 < k < len(y) - 1, "extremum at the edge of the wavelength window"
    y0, y1, y2 = y[k - 1:k + 2]
    return lam[k] + 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2) * (lam[1] - lam[0])


def _dip(mat, n_z, z, lam):
    return _extremum(lam, np.array([_reflectance(mat, n_z, z, L) for L in lam]), find_max=False)


def _loss_peak(mat, n_z, z, lam):
    return _extremum(lam, np.array([np.sum(np.imag(-1.0 / _cell_eps(mat, n_z, L)) * np.diff(z)) for L in lam]),
                     find_max=True)


@pytest.mark.parametrize("factory, n_bg, lam0", [(TP.azo, 5e26, 1.85e-6), (TP.gzo, 6e26, 1.6e-6),
                                                 (TP.in_cdo, 2.8e26, 2.2e-6), (TP.ito, 5e26, 1.55e-6)])
def test_gated_film_dip_shift_sign_and_magnitude(factory, n_bg, lam0):
    pytest.importorskip("tmm")
    from dynameta.carriers.sp_dual_gate import DualGateSchrodingerPoisson, GateStack
    mat = factory(n_bg)
    solver = DualGateSchrodingerPoisson(t_film_m=T_FILM, bottom=GateStack(((T_OX, EPS_OX),)), top=None, nz=401,
                                        **TP.dual_gate_film_kwargs(mat))
    lam_r = np.linspace(0.6 * lam0, 1.5 * lam0, 181)
    lam_l = np.linspace(0.6 * lam0, 1.5 * lam0, 451)
    dip, peak, sheet = {}, {}, {}
    for V in (-1.0, 0.0, 1.0):
        sol = solver.solve_voltages(V, 0.0)
        assert abs(sol.neutrality_residual_C_m2) < 1e-6
        dip[V] = _dip(mat, sol.n_m3, sol.z_m, lam_r)
        peak[V] = _loss_peak(mat, sol.n_m3, sol.z_m, lam_l)
        sheet[V] = sol.sheet_excess_m2
    assert sheet[1.0] > sheet[0.0] > sheet[-1.0]                       # the gate moves charge the right way
    acc, dep = dip[1.0] - dip[0.0], dip[-1.0] - dip[0.0]
    assert acc < -1e-9 and dep > 1e-9                                 # blue on accumulation, red on depletion
    for d_tmm, d_elf in ((acc, peak[1.0] - peak[0.0]), (dep, peak[-1.0] - peak[0.0])):
        assert d_tmm / d_elf == pytest.approx(1.0, rel=0.25, abs=0.0)
    assert 1e-9 < abs(acc) < 0.05 * lam0                               # nm-scale, far below a uniform-film shift
    # discrimination: (a) the gate ignored (density pinned at n_bg) gives no shift; (b) a mutant optical mass
    # that grows faster than n (m ~ n^1.5, so omega_p^2 FALLS with n) reverses the sign the test asserts
    import dataclasses
    z = solver.z_m
    sol_p = solver.solve_voltages(1.0, 0.0)
    assert _dip(mat, np.full(z.shape, n_bg), z, lam_r) - dip[0.0] != pytest.approx(acc, rel=0.5, abs=0.0)
    m_bg = float(mat.optical.m_opt_kg(n_bg))
    bad = dataclasses.replace(mat, optical=dataclasses.replace(
        mat.optical, m_opt_kg=lambda n: m_bg * (np.maximum(np.asarray(n, float), 1e20) / n_bg) ** 1.5))
    sol_0 = solver.solve_voltages(0.0, 0.0)
    assert _dip(bad, sol_p.n_m3, z, lam_r) - _dip(bad, sol_0.n_m3, z, lam_r) > 0.0
