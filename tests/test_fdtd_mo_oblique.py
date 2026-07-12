"""Fast unit tests for the magneto-optic 1-D FDTD (fdtd_mo) and the oblique complex-envelope 2D-TE
solver (low resolution -- the rigorous oracle checks live in validation/)."""
import numpy as np
import pytest

from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_mo import MOLayer, solve_fdtd_mo_1d
from dynameta.optics.fdtd_nd import _HAVE_NUMBA, solve_fdtd_2d_oblique, solve_fdtd_3d_mo

LMIN, LMAX = 1400e-9, 1600e-9


def test_mo_vacuum_is_transparent_and_unrotated():
    r = solve_fdtd_mo_1d([MOLayer(thickness_m=1e-12)], lambda_min_m=LMIN, lambda_max_m=LMAX,
                         resolution=18, pol="y")
    b = r.band
    assert abs(float(np.median(r.T[b])) - 1.0) < 1e-2
    assert float(np.max(np.abs(r.r_co[b]))) < 5e-2                 # no reflection from nothing
    assert float(np.max(np.abs(r.t_cross[b]))) < 1e-6             # no spurious polarization mixing


def test_mo_reduction_wc_zero_no_rotation():
    L = MOLayer(thickness_m=300e-9, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.2e15,
                drude_gamma_rad_s=2.0e13, cyclotron_wc_rad_s=0.0)
    r = solve_fdtd_mo_1d([L], lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=18, pol="y")
    b = r.band
    assert float(np.max(np.abs(r.faraday_deg[b]))) < 1e-2        # no gyration -> no Faraday rotation
    assert float(np.max(np.abs(r.t_cross[b]))) < 1e-3


def test_mo_gyrotropic_rotates():
    L = MOLayer(thickness_m=400e-9, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.2e15,
                drude_gamma_rad_s=2.0e13, cyclotron_wc_rad_s=3.0e14)
    r = solve_fdtd_mo_1d([L], lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=18, pol="y")
    b = r.band
    assert abs(float(np.median(r.faraday_deg[b]))) > 1.0          # gyration -> a real rotation
    assert float(np.median(np.abs(r.t_cross[b]))) > 1e-2          # cross-pol appears


def test_mo_eps_circular_reduces_to_drude_at_wc0():
    L = MOLayer(thickness_m=1.0, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.0e15, drude_gamma_rad_s=1.0e13)
    w = 2.0 * np.pi * 2.0e14
    ep, em = L.eps_circular(w, +1), L.eps_circular(w, -1)
    assert abs(ep - em) < 1e-12                                   # wc=0 -> the two circular modes coincide
    assert ep.imag > 0                                            # passive (loss)


def test_oblique_angle0_reduces_and_conserves_energy():
    r = solve_fdtd_2d_oblique([FDTDLayer(thickness_m=250e-9, eps_inf=4.0)], period_x_m=300e-9,
                              angle_deg=0.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=24)
    b = r.band
    assert np.all(np.abs(r.theta_deg[b]) < 1e-9)                 # angle 0 everywhere
    assert float(np.max(np.abs(r.R0[b] + r.T0[b] - 1.0))) < 2e-2  # lossless energy closes


def test_oblique_angle_is_frequency_dependent():
    r = solve_fdtd_2d_oblique([FDTDLayer(thickness_m=250e-9, eps_inf=4.0)], period_x_m=300e-9,
                              angle_deg=40.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=24)
    b = r.band
    th = r.theta_deg[b]
    assert th.max() > th.min() + 1.0                             # fixed k_par -> theta varies with f
    assert 20.0 < float(np.median(th)) < 60.0                    # near the requested 40 deg


def test_oblique_tm_ppol_energy_and_angle():
    # TM (p-pol: Hy,Ex,Ez) oblique kernel: lossless energy closes + the physical angle varies with f.
    r = solve_fdtd_2d_oblique([FDTDLayer(thickness_m=250e-9, eps_inf=4.0)], period_x_m=300e-9,
                              angle_deg=30.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=20,
                              nx=4, pol="p")
    b = r.band
    assert float(np.max(np.abs(r.R0[b] + r.T0[b] - 1.0))) < 3e-2  # lossless TM energy closes
    th = r.theta_deg[b]
    assert th.max() > th.min() + 1.0                             # fixed k_par -> theta varies with f


def test_oblique_jax_matches_numpy():
    pytest.importorskip("jax")
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0)]
    kw = dict(period_x_m=300e-9, angle_deg=30.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, nx=4)
    rn = solve_fdtd_2d_oblique(ol, backend="numpy", **kw)
    rj = solve_fdtd_2d_oblique(ol, backend="jax", **kw)
    m = rn.band
    assert float(np.max(np.abs(rn.R0[m] - rj.R0[m]))) < 1e-10    # differentiable scan == reference
    assert float(np.max(np.abs(rn.T0[m] - rj.T0[m]))) < 1e-10


def test_oblique_tm_ppol_jax_matches_numpy():
    pytest.importorskip("jax")
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0, drude_wp_rad_s=1.5e15, drude_gamma_rad_s=1.0e14)]
    kw = dict(period_x_m=300e-9, angle_deg=35.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14,
              nx=6, pol="p")
    rn = solve_fdtd_2d_oblique(ol, backend="numpy", **kw)
    rj = solve_fdtd_2d_oblique(ol, backend="jax", **kw)
    m = rn.band
    assert float(np.max(np.abs(rn.R0[m] - rj.R0[m]))) < 1e-9     # differentiable TM scan == reference
    assert float(np.max(np.abs(rn.T0[m] - rj.T0[m]))) < 1e-9


def test_oblique_tm_dispatch_routes_to_requested_backend(monkeypatch):
    # REGRESSION GUARD: pol='p' must honor `backend`. A stale `name='numpy'` guard once forced the TM
    # path to NumPy regardless of backend, so backend='numba'/'jax' silently ran NumPy and every byte-
    # match passed TAUTOLOGICALLY (jax==numpy because both were numpy). Patch the fast/diff kernels to a
    # sentinel and assert solve_fdtd_2d_oblique actually reaches them for pol='p'.
    import dynameta.optics.fdtd_nd as F
    # the oblique dispatcher (_run_oblique) resolves kernels via ITS module globals, so the
    # patch must land on the defining submodule, not the package namespace
    import dynameta.optics.fdtd_nd.oblique2d as FK

    class _Reached(Exception):
        pass

    def _boom(*a, **k):
        raise _Reached()

    ol = [FDTDLayer(thickness_m=200e-9, eps_inf=4.0)]
    kw = dict(period_x_m=300e-9, angle_deg=20.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=6,
              nx=4, pol="p")
    if F._HAVE_NUMBA:
        monkeypatch.setattr(FK, "_tm2d_oblique_numba", _boom)
        with pytest.raises(_Reached):
            solve_fdtd_2d_oblique(ol, backend="numba", **kw)
    if F._have_jax():
        monkeypatch.setattr(FK, "_run_2d_tm_oblique_jax", _boom)
        with pytest.raises(_Reached):
            solve_fdtd_2d_oblique(ol, backend="jax", **kw)


def test_oblique_3d_jax_matches_numpy():
    pytest.importorskip("jax")
    from dynameta.optics.fdtd_nd import solve_fdtd_3d_oblique
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0, drude_wp_rad_s=1.4e15, drude_gamma_rad_s=1.0e14)]
    kw = dict(period_x_m=300e-9, period_y_m=300e-9, angle_deg=25.0, azimuth_deg=20.0,
              lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=9, nx=5, ny=5, settle=8.0, n_pad_wave=2.5)
    rn = solve_fdtd_3d_oblique(ol, backend="numpy", **kw)
    rj = solve_fdtd_3d_oblique(ol, backend="jax", **kw)
    m = rn.band
    assert float(np.max(np.abs(rn.R0[m] - rj.R0[m]))) < 1e-9     # 3D differentiable scan == reference
    assert float(np.max(np.abs(rn.T0[m] - rj.T0[m]))) < 1e-9


def test_mo_3d_reduces_to_1d_faraday():
    # the full-vector 3D MO engine on a laterally-uniform gyrotropic slab reproduces the 1D Faraday
    # rotation. At THIS low resolution the 3D Yee grid carries stronger numerical dispersion near the
    # plasma band, so they only agree to a factor here -- the rigorous quantitative oracle (3D == 1D to
    # 6e-4 deg at resolution=40) lives in validation/fdtd_3d_mo_vs_1d. This fast test pins the physics:
    # a real rotation of the SAME sign and order of magnitude (a broken kernel is off by >10x or sign).
    L = MOLayer(thickness_m=300e-9, eps_xx=4.0, eps_yy=4.0, drude_wp_rad_s=2.0e15,
                drude_gamma_rad_s=1.0e14, cyclotron_wc_rad_s=3.0e14)
    kw = dict(lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, pol="y")
    r1 = solve_fdtd_mo_1d([L], **kw)
    r3 = solve_fdtd_3d_mo([L], period_x_m=300e-9, period_y_m=300e-9, nx=4, ny=4, **kw)
    f1 = r1.freqs_Hz[r1.band]
    far1 = float(np.median(r1.faraday_deg[r1.band]))
    far3 = float(np.median(np.interp(f1, r3.freqs_Hz[r3.band], r3.faraday_deg[r3.band])))
    assert abs(far1) > 1.0                                        # a real rotation to compare
    assert far1 * far3 > 0.0                                      # same sign of rotation
    assert 0.5 < far3 / far1 < 2.0                                # same order of magnitude (low-res floor)


def test_mo_3d_birefringent_no_cross_pol():
    # diagonal-anisotropic (no gyration) -> NO polarization mixing in the 3D engine + energy closes.
    L = MOLayer(thickness_m=300e-9, eps_xx=4.0, eps_yy=2.25, drude_wp_rad_s=0.0,
                drude_gamma_rad_s=0.0, cyclotron_wc_rad_s=0.0)
    r = solve_fdtd_3d_mo([L], period_x_m=300e-9, period_y_m=300e-9, nx=4, ny=4,
                         lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, pol="y")
    b = r.band
    assert float(np.max(np.abs(r.t_cross[b]))) < 1e-6            # no gyro -> no cross-pol
    assert float(np.max(np.abs(r.R[b] + r.T[b] - 1.0))) < 3e-2   # lossless energy closes


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
def test_mo_numba_matches_numpy():
    L = MOLayer(thickness_m=300e-9, eps_xx=4.0, eps_yy=2.25, drude_wp_rad_s=1.6e15,
                drude_gamma_rad_s=8.0e13, cyclotron_wc_rad_s=2.5e14)
    kw = dict(lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, pol="y")
    a = solve_fdtd_mo_1d([L], backend="numpy", **kw)
    b = solve_fdtd_mo_1d([L], backend="numba", **kw)
    m = a.band
    assert float(np.max(np.abs(a.R[m] - b.R[m]))) < 1e-10        # JIT loop == reference
    assert float(np.max(np.abs(a.T[m] - b.T[m]))) < 1e-10
    assert float(np.max(np.abs(a.faraday_deg[m] - b.faraday_deg[m]))) < 1e-8


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
def test_oblique_numba_matches_numpy():
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0)]
    kw = dict(period_x_m=300e-9, angle_deg=30.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14, nx=4)
    a = solve_fdtd_2d_oblique(ol, backend="numpy", **kw)
    b = solve_fdtd_2d_oblique(ol, backend="numba", **kw)
    m = a.band
    assert float(np.max(np.abs(a.R0[m] - b.R0[m]))) < 1e-10      # complex-envelope JIT loop == reference
    assert float(np.max(np.abs(a.T0[m] - b.T0[m]))) < 1e-10


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
def test_oblique_tm_ppol_numba_matches_numpy():
    # the p-pol (TM) complex-envelope oblique numba kernel == the NumPy TM reference (dispersive layer).
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0, drude_wp_rad_s=1.5e15, drude_gamma_rad_s=1.0e14)]
    kw = dict(period_x_m=300e-9, angle_deg=35.0, lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=14,
              nx=6, pol="p")
    a = solve_fdtd_2d_oblique(ol, backend="numpy", **kw)
    b = solve_fdtd_2d_oblique(ol, backend="numba", **kw)
    m = a.band
    assert float(np.max(np.abs(a.R0[m] - b.R0[m]))) < 1e-10      # TM JIT loop == reference
    assert float(np.max(np.abs(a.T0[m] - b.T0[m]))) < 1e-10


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
def test_oblique_3d_numba_matches_numpy():
    # the full-vector 3D oblique complex-envelope numba kernel == the NumPy reference (2D transverse Bloch
    # envelope kx AND ky, dispersive Drude layer, conical azimuth).
    from dynameta.optics.fdtd_nd import solve_fdtd_3d_oblique
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0, drude_wp_rad_s=1.4e15, drude_gamma_rad_s=1.0e14)]
    kw = dict(period_x_m=300e-9, period_y_m=300e-9, angle_deg=25.0, azimuth_deg=20.0,
              lambda_min_m=LMIN, lambda_max_m=LMAX, resolution=10, nx=5, ny=5, settle=8.0, n_pad_wave=2.5)
    a = solve_fdtd_3d_oblique(ol, backend="numpy", **kw)
    b = solve_fdtd_3d_oblique(ol, backend="numba", **kw)
    m = a.band
    assert float(np.max(np.abs(a.R0[m] - b.R0[m]))) < 1e-9       # 3D complex-envelope JIT == reference
    assert float(np.max(np.abs(a.T0[m] - b.T0[m]))) < 1e-9


# --- structured (laterally-patterned) 3D diagonal-tensor: the lateral_tensor override ----------------
def test_lateral_tensor_bad_inputs_raise():
    L = MOLayer(thickness_m=200e-9, eps_xx=4.0, eps_yy=4.0)
    kw = dict(period_x_m=300e-9, period_y_m=300e-9, lambda_min_m=LMIN, lambda_max_m=LMAX,
              resolution=6, nx=4, ny=4, pol="y")
    with pytest.raises(ValueError):                                   # unknown key
        solve_fdtd_3d_mo([L], lateral_tensor={"bogus": np.ones((4, 4, 4))}, **kw)
    with pytest.raises(ValueError):                                   # wrong shape
        solve_fdtd_3d_mo([L], lateral_tensor={"exx": np.ones((3, 3, 3))}, **kw)


def test_lateral_tensor_uniform_equals_layer_fill():
    # a laterally-UNIFORM lateral_tensor (the pillar fills the whole cell) reproduces the per-layer fill.
    L = MOLayer(thickness_m=250e-9, eps_xx=4.0, eps_yy=2.25, drude_wp_rad_s=0.0, drude_gamma_rad_s=0.0,
                cyclotron_wc_rad_s=0.0)
    kw = dict(period_x_m=300e-9, period_y_m=300e-9, lambda_min_m=LMIN, lambda_max_m=LMAX,
              resolution=10, nx=4, ny=4, pol="y")
    ref = solve_fdtd_3d_mo([L], **kw)

    def uniform(nx, ny, nz, zc, pad, z_struct):
        band = ((zc >= pad) & (zc < pad + z_struct))[None, None, :]
        exx = np.where(band, 4.0, 1.0) * np.ones((nx, ny, nz))
        eyy = np.where(band, 2.25, 1.0) * np.ones((nx, ny, nz))
        ezz = np.where(band, 0.5 * (4.0 + 2.25), 1.0) * np.ones((nx, ny, nz))
        return {"exx": exx, "eyy": eyy, "ezz": ezz}
    pat = solve_fdtd_3d_mo([L], lateral_tensor=uniform, **kw)
    m = ref.band
    assert float(np.max(np.abs(ref.R[m] - pat.R[m]))) < 1e-9
    assert float(np.max(np.abs(ref.T[m] - pat.T[m]))) < 1e-9


def test_lateral_tensor_structured_energy_and_diagonal_no_crosspol():
    # a structured anisotropic pillar (no gyration) closes energy and stays co-polarized (diagonal tensor).
    L = MOLayer(thickness_m=250e-9, eps_xx=6.25, eps_yy=6.25)
    kw = dict(period_x_m=600e-9, period_y_m=600e-9, lambda_min_m=LMIN, lambda_max_m=LMAX,
              resolution=8, nx=8, ny=8, pol="y", n_pad_wave=2.0, settle=8.0)

    def pillar(nx, ny, nz, zc, pad, z_struct):
        xs = (np.arange(nx) + 0.5) / nx
        ys = (np.arange(ny) + 0.5) / ny
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        mask = ((np.abs(X - 0.5) <= 0.25) & (np.abs(Y - 0.5) <= 0.25))[:, :, None]
        band = ((zc >= pad) & (zc < pad + z_struct))[None, None, :]
        exx = np.where(band, np.where(mask, 6.25, 1.0), 1.0)
        eyy = np.where(band, np.where(mask, 4.0, 1.0), 1.0)            # in-plane anisotropy
        ezz = exx
        return {"exx": exx, "eyy": eyy, "ezz": ezz}
    r = solve_fdtd_3d_mo([L], lateral_tensor=pillar, **kw)
    b = r.band
    assert np.all(np.isfinite(r.R[b])) and np.all(np.isfinite(r.T[b]))
    # diagonal tensor (no gyration) on an x/y-mirror-symmetric pillar -> mean cross-pol vanishes by
    # symmetry (numerically small at this coarse res; the grating makes it nonzero unlike a uniform slab).
    assert float(np.max(np.abs(r.t_cross[b]))) < 2e-2                 # negligible cross-pol power (~4e-4)
    assert float(np.max(np.abs(r.R[b] + r.T[b] - 1.0))) < 5e-2        # lossless energy closes


def test_mo_grid_sizing_is_wc_sign_invariant():
    # audit C3-3: the sizing bound must see the resonant circular branch for BOTH signs of
    # wc (the old (w,+1)-only max reported n_max=1.28 where the true both-branch peak is
    # 7.51 for wc=-1.28e15 -- dz 5.9x too coarse, silently) and must not undercut the
    # background birefringent indices
    import numpy as np
    from dynameta.constants import C_LIGHT
    from dynameta.optics.fdtd_mo import MOLayer, _mo_band_index_bound
    w_band = 2.0 * np.pi * np.linspace(C_LIGHT / 1.7e-6, C_LIGHT / 1.3e-6, 9)
    mk = lambda wc: MOLayer(thickness_m=300e-9, eps_xx=2.0, eps_yy=2.0, drude_wp_rad_s=1.2e15,
                            drude_gamma_rad_s=2e13, cyclotron_wc_rad_s=wc)
    n_pos = _mo_band_index_bound(mk(+1.28e15), w_band)
    n_neg = _mo_band_index_bound(mk(-1.28e15), w_band)
    assert n_neg == pytest.approx(n_pos, rel=1e-12)              # sign-invariant
    assert n_neg > 5.0                                           # resonance actually seen
    # background floor: a birefringent layer whose Drude term depresses the circular index
    hi_bg = MOLayer(thickness_m=300e-9, eps_xx=9.0, eps_yy=2.0, drude_wp_rad_s=1e14,
                    drude_gamma_rad_s=2e13, cyclotron_wc_rad_s=0.0)
    assert _mo_band_index_bound(hi_bg, w_band) >= 3.0
