"""Wave-7 (2026-11-02) closure gates for the still-open amplifier P3s of the 2026-07-17 exhaustive
audit (docs/audit/2026-07-17-exhaustive-audit.md): S3-19, S3-21, S3-22, S3-23, S3-26, S3-28, S5-14,
plus the H6a leftovers in soa/qd_gain.py (__all__, T_REF defaults, ONE lazy-njit shim).

Every correctness gate here FAILS on the pre-wave-7 code (HEAD 7b9481a); the measured old values
are quoted in each test."""
import ast
import inspect
import pathlib
import warnings
from fractions import Fraction

import numpy as np
import pytest

from dynameta.optics.soa import QDGainModel, QDGainParams, TravelingWaveSOA, TwoLevelSaturableGain


# ---------------------------------------------------------------- S3-19: one photon-density kernel
def test_s3_19_power_to_photon_density_is_photon_density():
    """power_to_photon_density was a second copy of the S_conf formula that wrapped the result in
    float() -- an array power raised TypeError there while photon_density returned an array. Now
    ONE implementation (the alias), scalar -> float, array -> ndarray, values bit-identical to the
    old scalar formula. FAILS on the old code (not the same function; array input raised)."""
    m = QDGainModel(QDGainParams(n_groups=1))
    p, nu = m.p, m.p.nu0_Hz
    assert QDGainModel.power_to_photon_density is QDGainModel.photon_density
    from dynameta.constants import H_PLANCK
    for P in (1e-6, 3.7e-3, 0.25):
        old = float(p.Gamma * P / (p.v_g_m_s * H_PLANCK * nu * p.A_mode_m2))   # verbatim old copy
        S = m.power_to_photon_density(P, nu)
        assert type(S) is float and S == old
        assert m.photon_density(P, nu) == old
    Pa = np.array([1e-6, 3.7e-3, 0.25])
    Sa = m.power_to_photon_density(Pa, nu)                      # old: TypeError (float(array))
    assert isinstance(Sa, np.ndarray) and Sa.shape == (3,)
    assert np.array_equal(Sa, [m.photon_density(float(x), nu) for x in Pa])


# --------------------------------------------------- S3-21: eta_in on the distributed-GVD path
def _gvd_setup(g0):
    tl = TwoLevelSaturableGain(g0_per_m=g0, tau_c_s=1e-9, E_sat_J=1e-12)
    L, nz = 1.0e-3, 64
    soa = TravelingWaveSOA(tl, L, nz)
    nt = 3 * nz
    t = np.arange(nt) * soa.dt
    T0 = nz * soa.dt / 16.0
    beta2 = T0 * T0 / L
    A = 1e-4 * np.exp(-((t - (nt // 2) * soa.dt) ** 2) / (2 * T0 * T0)) + 0j   # 1e-8 W peak: linear
    return soa, A, beta2


@pytest.mark.parametrize("g0", [0.0, 1500.0])
def test_s3_21_eta_in_acts_on_segmented_gvd_path(g0):
    """amplify_coherent(gvd_segments>1, beta2!=0) returned BEFORE the eta_in facet loss was applied,
    so eta_in was silently ignored there: measured old ratio A_out(eta=0.5)/A_out(eta=1) = 1.000
    (passive and with 1500/m gain) instead of sqrt(0.5) = 0.7071. Linear regime (1e-8 W peak) ->
    the device-referred field must scale by exactly sqrt(eta) on BOTH paths, and the reported
    A_in/P_in stay fiber-referred."""
    soa, A, beta2 = _gvd_setup(g0)
    eta = 0.5
    for S in (1, 4):
        r1 = soa.amplify_coherent(A, None, beta2_s2_per_m=beta2, gvd_segments=S)
        re = soa.amplify_coherent(A, None, beta2_s2_per_m=beta2, gvd_segments=S, eta_in=eta)
        k = np.argmax(np.abs(r1["A_out"]))
        assert abs(re["A_out"][k] / r1["A_out"][k] - np.sqrt(eta)) < 1e-6, (S, re["A_out"][k] / r1["A_out"][k])
        assert np.max(np.abs(re["A_out"] - np.sqrt(eta) * r1["A_out"])) < 1e-6 * np.max(np.abs(r1["A_out"]))
        assert np.array_equal(re["A_in"], A) and np.array_equal(re["P_in"], np.abs(A) ** 2)
    # eta_in = 1 stays byte-identical on the segmented path; invalid eta raises there too
    base = soa.amplify_coherent(A, None, beta2_s2_per_m=beta2, gvd_segments=4)["A_out"]
    assert np.array_equal(base, soa.amplify_coherent(A, None, beta2_s2_per_m=beta2, gvd_segments=4,
                                                     eta_in=1.0)["A_out"])
    with pytest.raises(ValueError):
        soa.amplify_coherent(A, None, beta2_s2_per_m=beta2, gvd_segments=4, eta_in=0.0)


def test_s3_21_segmented_path_refuses_langevin_and_state0():
    """The segmented path also silently dropped langevin (returned a NOISELESS field to a noisy
    request) and state0. They now raise instead."""
    soa, A, beta2 = _gvd_setup(1500.0)
    with pytest.raises(NotImplementedError):
        soa.amplify_coherent(A, None, beta2_s2_per_m=beta2, gvd_segments=4, langevin=True, seed=1)
    with pytest.raises(NotImplementedError):
        soa.amplify_coherent(A, None, beta2_s2_per_m=beta2, gvd_segments=4,
                             state0=soa.model.init_slices(soa.nz, None))


# ------------------------------------------------------------ S3-22: tridiagonal fin solves
def _fin_exact(q, dz, kA, Rp, ends):
    """Exact rational (Fraction) Thomas solve of the fin system (the oracle)."""
    n = len(q)
    c = Fraction(kA) / (Fraction(dz) ** 2)
    s = 1 / Fraction(Rp)
    lo = [c] * n; di = [-(2 * c + s)] * n; up = [c] * n; b = [-Fraction(x) for x in q]
    if ends == "sunk":
        di[0] = di[-1] = Fraction(1); up[0] = lo[-1] = Fraction(0); b[0] = b[-1] = Fraction(0)
    else:
        up[0] = lo[-1] = 2 * c
    cp, dp = [Fraction(0)] * n, [Fraction(0)] * n
    for i in range(n):
        den = di[i] - (lo[i] * cp[i - 1] if i else 0)
        cp[i] = up[i] / den
        dp[i] = (b[i] - (lo[i] * dp[i - 1] if i else 0)) / den
    x = [Fraction(0)] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.array([float(v) for v in x])


def _fin_dense_old(q, dz, kA, Rp, ends):
    """VERBATIM algebra of the pre-wave-7 dense steady solve (HEAD 7b9481a)."""
    n = q.size
    c = kA / dz ** 2
    s = 1.0 / Rp
    A = np.zeros((n, n))
    b = -q.copy()
    for k in range(n):
        A[k, k] = -(2.0 * c + s)
        if k > 0:
            A[k, k - 1] = c
        if k < n - 1:
            A[k, k + 1] = c
    if ends == "sunk":
        A[0, :] = 0.0; A[0, 0] = 1.0; b[0] = 0.0
        A[-1, :] = 0.0; A[-1, -1] = 1.0; b[-1] = 0.0
    else:
        A[0, 0] = -(2.0 * c + s); A[0, 1] = 2.0 * c
        A[-1, -1] = -(2.0 * c + s); A[-1, -2] = 2.0 * c
    return np.linalg.solve(A, b)


def test_s3_22_fin_solve_is_tridiagonal_and_exact():
    """The fin solvers built a DENSE (n, n) matrix and solved it densely although documented as a
    tridiagonal solve. Now LAPACK banded LU (O(n)). (a) 'insulated' agrees with the old dense solve
    to rounding (measured 3.5e-15); (b) 'sunk' matches an EXACT rational solve to 5e-14 where the
    old dense solve was off by 3.1e-10 at n = 300 (its partial pivoting swapped the |c| ~ 1e7 row
    above the unit Dirichlet row) -- FAILS on the old code; (c) n = 200000 is solved in O(n) (the
    old code would have allocated a 320 GB dense matrix -> MemoryError)."""
    from dynameta.optics.soa.thermal import thermal_profile_steady_1d, thermal_profile_transient_1d
    rng = np.random.default_rng(0)
    n, L, kA, Rp = 300, 1.0e-3, 2.0e-5, 5.0e-4
    dz = L / n
    q = 2.0e4 * (1.0 + rng.random(n))
    for ends in ("sunk", "insulated"):
        ex = _fin_exact(q, dz, kA, Rp, ends)
        th = thermal_profile_steady_1d(q, dz, kA, Rp, 0.0, ends=ends)
        assert np.max(np.abs(th - ex)) / np.max(np.abs(ex)) < 5e-14, ends   # measured 9.9e-15 / 1.5e-14
        # transient run to steady (400 lumped time constants) lands on the same exact profile
        tt = thermal_profile_transient_1d(q, dz, kA, Rp, 0.0, 1e-3, 1e-3 * Rp, 400, ends=ends)
        assert np.max(np.abs(tt - ex)) / np.max(np.abs(ex)) < 1e-13, ends
    th_ins = thermal_profile_steady_1d(q, dz, kA, Rp, 0.0, ends="insulated")
    old_ins = _fin_dense_old(q, dz, kA, Rp, "insulated")
    assert np.max(np.abs(th_ins - old_ins)) / np.max(np.abs(old_ins)) < 1e-14
    # the old dense 'sunk' solve was NOT exact (documents why sunk is not compared to it)
    old_sunk = _fin_dense_old(q, dz, kA, Rp, "sunk")
    ex_sunk = _fin_exact(q, dz, kA, Rp, "sunk")
    assert np.max(np.abs(old_sunk - ex_sunk)) / np.max(np.abs(ex_sunk)) > 1e-12
    # (c) O(n): uniform q + insulated has the exact solution theta = q Rth' everywhere
    # (a 0.2 m fin keeps c/s = kA Rth'/dz^2 ~ 1e4; at L = 1 mm and n = 2e5 the operator itself has
    # c/s ~ 4e8 and ANY solver loses ~1e-7 K at the far end -- conditioning, not the solver)
    nb = 200_000
    Tb = thermal_profile_steady_1d(np.full(nb, 2.0e4), 0.2 / nb, kA, Rp, 300.0, ends="insulated")
    assert np.max(np.abs(Tb - (300.0 + 2.0e4 * Rp))) < 1e-9          # measured 1.4e-11 K, 18 ms
    # n = 2 (the smallest accepted grid) still solves
    assert np.array_equal(thermal_profile_steady_1d([1e4, 1e4], dz, kA, Rp, 300.0, ends="sunk"),
                          [300.0, 300.0])


# ------------------------------------------------- S3-23: saturation_curve convergence guard
def test_s3_23_saturation_curve_converges_or_warns():
    """saturation_curve read P_out after a bare settle_transits * nz window with no convergence
    check (every validation overrode it with 400). Two-level oracle device whose carrier time is
    85 transits: the OLD default (60 transits) is off by 0.32 % (1 uW) and 16.7 % (0.3 mW = 0.3
    P_sat) vs the 4000-transit reference -- FAILS on the old code. The guard continues the run
    exactly until the last-half change <= rtol (1e-3): measured error 2.2e-5 / 4.5e-11."""
    tl = TwoLevelSaturableGain(g0_per_m=2000.0, tau_c_s=1e-9, E_sat_J=1e-12)
    soa = TravelingWaveSOA(tl, 1e-3, 10)
    Pin = np.array([1e-6, 3e-4])
    ref = np.array([soa.amplify(np.full(4000 * soa.nz, P), None)["P_out"][-1] for P in Pin])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # converged -> no warning
        sc = soa.saturation_curve(None, Pin)
    assert np.all(np.abs(sc["P_out_W"] / ref - 1.0) < 1e-3), sc["P_out_W"] / ref - 1.0
    assert np.all(sc["settle_rel_change"] <= 1e-3) and np.all(sc["settle_transits_used"] > 60)
    # a window capped below convergence WARNS instead of silently returning the unsettled value
    from dynameta.warning_types import UnreliableResultWarning
    with pytest.warns(UnreliableResultWarning, match="not settled"):
        soa.saturation_curve(None, Pin, max_settle_transits=60)
    # the continuation is EXACT: two chained runs == one long run, bit for bit
    a = soa.amplify(np.full(1200, 3e-4), None)
    b1 = soa.amplify(np.full(600, 3e-4), None)
    b2 = soa.amplify(np.full(600, 3e-4), None, state0=b1["state"], _P_node0=b1["P_node"])
    assert np.array_equal(a["P_out"], np.concatenate([b1["P_out"], b2["P_out"]]))
    assert np.array_equal(a["state"], b2["state"])
    # a point converged at the first check is the old single-window value, bit for bit
    sc400 = soa.saturation_curve(None, Pin, settle_transits=4000)
    assert np.array_equal(sc400["settle_transits_used"], [4000, 4000])
    assert np.array_equal(sc400["P_out_W"], ref)


# --------------------------------------------- S3-26: m_pol contract of the bidirectional ASE
def test_s3_26_mean_psd_is_per_pol_and_documented():
    """ase_spectrum_bidirectional's docstring said m_pol was applied 'to the *_out / *_mean
    accumulators'; the code keeps *_mean PER-POL, which is what its consumer ase_self_consistent
    needs (it multiplies by m_pol itself). Code is right; the doc is corrected and both pinned."""
    from dynameta.optics.soa.ase_noise import ase_spectrum_bidirectional
    rng = np.random.default_rng(3)
    g = 200.0 + 50.0 * rng.random((20, 3))
    gsp = 1.5 * g
    nu = np.array([2.29e14, 2.30e14, 2.31e14])
    r1 = ase_spectrum_bidirectional(g, gsp, 1e-4, nu, np.full(3, 1e11), 0.1, m_pol=1)
    r2 = ase_spectrum_bidirectional(g, gsp, 1e-4, nu, np.full(3, 1e11), 0.1, m_pol=2)
    for k in ("S_f_mean", "S_b_mean", "S_f", "S_b"):
        assert np.array_equal(r1[k], r2[k]), k                 # per-pol: m_pol-independent
    assert np.array_equal(r2["S_f_out"], 2.0 * r1["S_f_out"])
    doc = inspect.getdoc(ase_spectrum_bidirectional)
    assert "*_out / *_mean accumulators only" not in doc       # the old (wrong) contract sentence
    assert "PER-POL z-averaged PSD -- NOT scaled by m_pol" in doc


# ------------------------------------------------- S3-28: noise_figure needs Gamma g with loss
def test_s3_28_noise_figure_internal_loss():
    """noise_figure(G, n_sp, alpha_i_per_m > 0) with Gamma_g_per_m=None silently returned the
    LOSSLESS NF (G 100, n_sp 1.5, alpha_i 60/m: 4.74 dB, vs 5.71 dB at Gamma g 300/m) -- FAILS on
    the old code (no raise). With Gamma g given it equals the analytic distributed-loss NF and the
    PSD-propagated NF of a uniform lossy amplifier (ase_output_psd -> nf_from_psd)."""
    from dynameta.optics.amp_noise import nf_from_psd
    from dynameta.optics.soa.ase_noise import ase_output_psd, noise_figure
    with pytest.raises(ValueError, match="Gamma_g_per_m"):
        noise_figure(100.0, 1.5, alpha_i_per_m=60.0)
    with pytest.raises(ValueError):
        noise_figure(100.0, 1.5, Gamma_g_per_m=300.0, alpha_i_per_m=-1.0)
    assert noise_figure(100.0, 1.5, alpha_i_per_m=0.0) == noise_figure(100.0, 1.5)
    # analytic: NF = 2 n_sp Gg/(Gg - a) (G-1)/G + 1/G
    G, nsp, Gg, ai = 100.0, 1.5, 300.0, 60.0
    nf = noise_figure(G, nsp, Gamma_g_per_m=Gg, alpha_i_per_m=ai)
    assert abs(nf - (2 * nsp * Gg / (Gg - ai) * (G - 1) / G + 1 / G)) < 1e-14
    assert abs(10 * np.log10(nf) - 5.7083) < 1e-4 and abs(10 * np.log10(noise_figure(G, nsp)) - 4.7422) < 1e-4
    # PSD route on a uniform lossy profile: rho with n_sp = rho^2/(2 rho - 1)
    Gamma, rho, nu, nz = 0.1, 0.8, 2.3e14, 400
    nsp_u = rho * rho / (2 * rho - 1)
    g = 3000.0                                                 # material -> Gamma g = 300 /m
    L = np.log(G) / (Gamma * g - ai)                          # length giving net gain G exactly
    S = ase_output_psd(np.full(nz, g), np.full(nz, rho), L / nz, nu, Gamma, alpha_i_per_m=ai,
                       per_pol=True)
    assert abs(nf_from_psd(G, S, nu) / noise_figure(G, nsp_u, Gamma_g_per_m=Gamma * g,
                                                    alpha_i_per_m=ai) - 1.0) < 1e-10


# --------------------------------------------------- S5-14: implicit Optional in fiber_amp
def test_s5_14_no_implicit_optional_in_fiber_amp():
    """Every `x: T = None` in optics/fiber_amp spells Optional[T] (the package convention). The
    audit counted 21 implicit-Optional annotations; 16 were left at HEAD 7b9481a (calibration,
    detection x3, pulse x6, rare_earth, spectroscopy x3, transverse_eryb x2) -- FAILS there."""
    import dynameta.optics.fiber_amp as fa
    hits = []
    for path in sorted(pathlib.Path(fa.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                pos = a.posonlyargs + a.args
                pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults))
                pairs += list(zip(a.kwonlyargs, a.kw_defaults))
                pairs = [(x.annotation, d, x.arg) for x, d in pairs]
            elif isinstance(node, ast.AnnAssign):
                pairs = [(node.annotation, node.value, ast.unparse(node.target))]
            for ann, d, name in pairs:
                if ann is None or not (isinstance(d, ast.Constant) and d.value is None):
                    continue
                txt = ast.unparse(ann)
                if "Optional" not in txt and "None" not in txt and txt not in ("Any", "object"):
                    hits.append("{}:{} {}: {}".format(path.name, node.lineno, name, txt))
    assert not hits, hits


# --------------------------------------------------------- H6a leftovers in soa/qd_gain.py
def test_h6a_qd_gain_all_tref_and_single_lazy_shim():
    """(a) Leakage / ManyBody are in qd_gain.__all__; (b) QDGainParams.T_K and SelfHeating.T0_K
    default to dynameta.constants.T_REF (numerically 300.0, unchanged); (c) the lazy njit shim is
    ONE implementation in dynameta.numba_env, used by soa/qd_gain.py and fdtd_nd/backends.py (the
    soa copy was a second, self-contained one)."""
    import dataclasses

    from dynameta import numba_env
    from dynameta.constants import T_REF
    from dynameta.optics.fdtd_nd import backends
    from dynameta.optics.soa import qd_gain
    assert {"Leakage", "ManyBody"} <= set(qd_gain.__all__)
    assert T_REF == 300.0
    assert QDGainParams().T_K == T_REF
    assert [f.default for f in dataclasses.fields(qd_gain.SelfHeating) if f.name == "T0_K"] == [T_REF]
    assert qd_gain._njit is numba_env.njit and backends.njit is numba_env.njit
    assert qd_gain._LazyJit is numba_env._LazyJit is backends._LazyJit
    assert backends.have_numba is numba_env.have_numba and backends.prange is numba_env.prange
    assert isinstance(qd_gain._qd_carrier_rk4_numba, numba_env._LazyJit)
    for mod in (qd_gain, backends):
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert "class _LazyJit" not in src and "def have_numba(" not in src, mod.__name__
