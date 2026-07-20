"""Roadmap 3.1 harmonic diagnostics: SH/TH spectral extraction, conversion efficiency, the
P_2w ~ P_w^2 / P_3w ~ P_w^3 slope gates, and the undepleted-pump validity check
(dynameta.optics.harmonics).

The fast synthetic tests pin the diagnostics as pure functions (exact multi-tone signals). The
FDTD gate tests drive the SAME uniform-medium chi2/chi3 testbed as the coupled-wave oracle
(validation/fdtd_chi2_shg_raman.py) through a module-scoped fixture (the ~10 marches run once),
and reproduce the oracle's already-validated SHG physics THROUGH harmonic_spectrum:

  GATE 1  chi2 SHG closed form: harmonic_spectrum's SHG band power reproduces the undepleted
          coupled-wave closed form e2 = (chi2 w0 L / 2nc) Im[z1^2] (Boyd ch. 2) built from the
          MEASURED pump, within the oracle's own 5% amplitude tolerance.
  GATE 2  slopes: chi2 slab, >= 4 pump amplitudes -> SHG slope 2.00 +/- 0.02, r2 > 0.999; chi3
          Kerr slab -> THG slope 3.00 +/- 0.05 (the instantaneous eps_eff = eps_inf + 3 chi3 E^2
          radiates a genuine 3w line -- VERIFIED to rise ~24 decades above the zero-nl floor).
  GATE 3  zero nonlinearity -> the 2w/3w bands sit > 60 dB below the fundamental (numerical floor).
  GATE 4  diagnostic linearity: doubling the pump POWER quadruples P_2w in the gate-2 run family.
  GATE 5  byte-identity: solve_fdtd_2d(return_time_trace=...) leaves every R/T array identical to
          the legacy path (nan-aware; the out-of-band 0/0 bins are common to both runs).
"""
import numpy as np
import pytest

from dynameta.constants import C_LIGHT
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import run_2d_te, cpml_z, solve_fdtd_2d
from dynameta.optics.harmonics import (conversion_efficiency, harmonic_spectrum,
                                       power_slope, undepleted_validity)

try:
    from scipy.signal import hilbert
    HAVE_SCIPY = True
except Exception:                                            # pragma: no cover
    HAVE_SCIPY = False


# =================================================================================================
# fast synthetic unit tests (no FDTD)
# =================================================================================================
def _tritone(a1, a2, a3, *, k0=40, N=1200, dt=1e-15):
    """Real tri-tone with tones EXACTLY on bins k0/2k0/3k0 (no leakage) at f0 = k0/(N dt)."""
    n = np.arange(N)
    f0 = k0 / (N * dt)
    e = (a1 * np.cos(2 * np.pi * (k0 / N) * n)
         + a2 * np.cos(2 * np.pi * (2 * k0 / N) * n)
         + a3 * np.cos(2 * np.pi * (3 * k0 / N) * n))
    return e, dt, f0


def test_harmonic_spectrum_band_powers_exact():
    a1, a2, a3 = 1.0, 0.1, 0.03
    e, dt, f0 = _tritone(a1, a2, a3)
    hs = harmonic_spectrum((e, dt), f0)
    # a cosine of amplitude a at an exact bin has |rfft|^2 = (a N / 2)^2; only that bin is in-band,
    # so band-power RATIOS equal the amplitude-squared ratios to machine precision.
    assert hs["P_2w"] / hs["P_w"] == pytest.approx((a2 / a1) ** 2, rel=1e-9)
    assert hs["P_3w"] / hs["P_w"] == pytest.approx((a3 / a1) ** 2, rel=1e-9)
    assert hs["power_type"] == "e2_density"
    # band edges follow the documented constant-fractional convention n*f0*(1 +- bw)
    assert hs["bands_hz"][2] == pytest.approx((2 * f0 * 0.85, 2 * f0 * 1.15), rel=1e-12)


def test_harmonic_spectrum_order_resolution_and_parseval():
    # fundamental uniform in x (specular order 0); SHG spatially modulated cos(2 pi x/nx) -> orders +-1
    N, nx, dt, k0 = 1200, 8, 1e-15, 40
    n = np.arange(N)[:, None]
    xcol = np.arange(nx)[None, :]
    f0 = k0 / (N * dt)
    e = (1.0 * np.cos(2 * np.pi * (k0 / N) * n) * np.ones((1, nx))
         + 0.1 * np.cos(2 * np.pi * (2 * k0 / N) * n) * np.cos(2 * np.pi * xcol / nx))
    hs = harmonic_spectrum((e, dt), f0)
    orders = list(hs["orders"])
    p2 = hs["power_by_order"][2]
    # SHG power sits in orders +-1, ~0 in the specular order 0
    assert p2[orders.index(0)] < 1e-6 * p2.sum()
    assert p2[orders.index(1)] > 0.4 * p2.sum()
    assert p2[orders.index(-1)] > 0.4 * p2.sum()
    # fundamental is specular
    p1 = hs["power_by_order"][1]
    assert p1[orders.index(0)] > 0.999 * p1.sum()
    # Parseval: the order sum equals the total band power
    assert hs["power_by_order"][2].sum() == pytest.approx(hs["P_2w"], rel=1e-10)


def test_power_slope_exact_and_guards():
    Pw = np.array([1.0, 2.0, 4.0, 8.0])
    s2 = power_slope(Pw, Pw ** 2)
    s3 = power_slope(Pw, Pw ** 3)
    assert s2["slope"] == pytest.approx(2.0, abs=1e-12) and s2["r2"] == pytest.approx(1.0, abs=1e-12)
    assert s3["slope"] == pytest.approx(3.0, abs=1e-12) and s3["r2"] == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):                          # non-positive power -> no log-log slope
        power_slope(Pw, np.array([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):                          # < 2 samples
        power_slope(np.array([1.0]), np.array([1.0]))


def test_conversion_efficiency_and_undepleted_arithmetic():
    a1, a2, a3 = 1.0, 0.2, 0.05
    e, dt, f0 = _tritone(a1, a2, a3)
    ce = conversion_efficiency((e, dt), f0, normalization="transmitted")
    assert ce["eta_shg"] == pytest.approx((a2 / a1) ** 2, rel=1e-9)
    assert ce["eta_thg"] == pytest.approx((a3 / a1) ** 2, rel=1e-9)
    assert ce["normalization"] == "transmitted"
    # incident normalization from an explicit reference (pure fundamental incident)
    inc, _, _ = _tritone(2.0, 0.0, 0.0)
    ce_i = conversion_efficiency((e, dt), f0, incident=(inc, dt), normalization="incident")
    assert ce_i["normalization"] == "incident"
    assert ce_i["eta_shg"] == pytest.approx((a2 ** 2) / (2.0 ** 2), rel=1e-9)
    uv = undepleted_validity((e, dt), f0, incident=(inc, dt), threshold=0.1)
    assert uv["undepleted"] is True and uv["eta_total"] < 0.1
    uv2 = undepleted_validity((e, dt), f0, incident=(inc, dt), threshold=1e-3)
    assert uv2["undepleted"] is False                        # eta_total ~ 0.011 > 1e-3


def test_input_contract_guards():
    with pytest.raises(ValueError):                          # bare array needs dt
        harmonic_spectrum(np.zeros(100), 1e14)
    with pytest.raises(ValueError):                          # bw must keep bands disjoint
        harmonic_spectrum((np.zeros(100), 1e-15), 1e13, bandwidth_frac=0.3)


def test_pump_bandwidth_guard():
    """The pump-bandwidth guard (module docstring PUMP-BANDWIDTH PRECONDITION): a broadband
    PURE-FUNDAMENTAL pump leaks its own spectral tail into the 2w band and would be mis-read as
    SHG -- a tau = 3 fs Gaussian pulse at f0 = 2.5e14 (sigma_f/f0 ~ 0.20) measures a phantom
    P_2w/P_w ~ 9e-4 with ZERO nonlinearity, and MUST trigger the UserWarning + the
    pump_broadband flag. The tau = 50 fs testbed pump (sigma_f/f0 ~ 0.013, the FDTD gates'
    source) stays clean: no warning, honest floor."""
    f0, dt = 2.5e14, 5e-17

    def pulse(tau):
        t0 = 6.0 * tau
        n = int(round((12.0 * tau + 200e-15) / dt))
        t = np.arange(n) * dt
        return np.cos(2 * np.pi * f0 * (t - t0)) * np.exp(-((t - t0) / tau) ** 2)

    with pytest.warns(UserWarning, match="BROADBAND pump"):
        hs3 = harmonic_spectrum((pulse(3e-15), dt), f0)
    assert hs3["pump_broadband"] is True
    assert hs3["pump_sigma_hz"] > hs3["pump_sigma_max_hz"]
    assert hs3["P_2w"] / hs3["P_w"] > 1e-6                   # the phantom leak the guard flags
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")                      # any warning would fail the test
        hs50 = harmonic_spectrum((pulse(50e-15), dt), f0)
    assert hs50["pump_broadband"] is False
    assert hs50["pump_sigma_hz"] < 0.02 * f0                 # the narrowband testbed regime
    assert hs50["P_2w"] / hs50["P_w"] < 1e-6                 # honest floor holds

    class _Res:                                              # a result missing its opt-in trace
        time_trace = None
    with pytest.raises(ValueError):
        harmonic_spectrum(_Res(), 1e14)


# =================================================================================================
# FDTD gate fixture: the uniform-medium chi2/chi3 testbed (marched ONCE, module-scoped)
# =================================================================================================
N_MED = np.sqrt(2.0)
NX = 4
F0 = 2.5e14                                                  # 1.2 um pump
DZ = 10e-9
NPAD, NSTR = 48, 40                                          # L = 400 nm nonlinear window
L_WIN = NSTR * DZ
TAU = 50e-15
T0 = 6.0 * TAU
A0_SHG = 5.0e8
CHI2 = 2.0e-11                                               # perturbative chi2*E ~ 1e-2 (oracle value)
A0_THG = 1.5e8
CHI3 = 8.0e-20
SHG_MULTS = np.array([1.0, np.sqrt(2.0), 2.0, 2.0 * np.sqrt(2.0)])   # pump POWER doubles each step
THG_MULTS = 2.0 ** (np.arange(4) / 3.0)                     # gentle range -> deep perturbative THG


def _grid():
    nz = 2 * NPAD + NSTR
    dx = 4.0 * DZ
    dt = 0.5 / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / DZ ** 2))
    return nz, dx, dt


def _src(amp):
    _, _, dt = _grid()
    nsteps = int(round((2.0 * T0 + 150e-15) / dt))
    t = np.arange(nsteps) * dt
    return amp * np.exp(-((t - T0) / TAU) ** 2) * np.cos(2.0 * np.pi * F0 * (t - T0)), dt


def _run(src, *, chi2_val=0.0, kerr_val=0.0):
    """Uniform n=sqrt(2) medium (index-matched ends -> Delta_k = 0), nonlinearity in the central
    window. Returns the x-mean transmitted (E_y, H_x) probe series and dt."""
    nz, dx, dt = _grid()
    eps = np.full((NX, nz), N_MED ** 2)
    zeros = np.zeros((NX, nz))
    c3k = zeros
    if kerr_val:
        c3k = np.zeros((NX, nz))
        c3k[:, NPAD:NPAD + NSTR] = kerr_val
    chi2 = None
    if chi2_val:
        chi2 = np.zeros((NX, nz))
        chi2[:, NPAD:NPAD + NSTR] = chi2_val
    cpml = cpml_z(nz, DZ, dt, 12, N_MED, N_MED)
    eyL, hxL, eyR, hxR = run_2d_te(eps, zeros, zeros, c3k, dx, DZ, dt, src.size, 16, 20, nz - 16,
                                   src, cpml, np, None, chi2=chi2)
    return eyR.mean(axis=1), hxR.mean(axis=1), dt


@pytest.fixture(scope="module")
def fdtd_gates():
    src_shg, dt = _src(A0_SHG)
    ey_off, hx_off, _ = _run(src_shg)                        # linear @ A0_SHG (chi2-off pump ref)
    ey_on, hx_on, _ = _run(src_shg, chi2_val=CHI2)           # chi2-on @ A0_SHG (SHG family base)
    # SHG amplitude family (base reused)
    shg_Pw = [harmonic_spectrum((ey_on, dt), F0)["P_w"]]
    shg_P2 = [harmonic_spectrum((ey_on, dt), F0)["P_2w"]]
    for mm in SHG_MULTS[1:]:
        s, _ = _src(mm * A0_SHG)
        ey, _, _ = _run(s, chi2_val=CHI2)
        hs = harmonic_spectrum((ey, dt), F0)
        shg_Pw.append(hs["P_w"])
        shg_P2.append(hs["P_2w"])
    # THG family + linear floor at the THG base amplitude
    ey_k0, _, _ = _run(_src(A0_THG)[0])                      # linear @ A0_THG (THG-off floor)
    thg_Pw, thg_P3 = [], []
    for mm in THG_MULTS:
        s, _ = _src(mm * A0_THG)
        ey, _, _ = _run(s, kerr_val=CHI3)
        hs = harmonic_spectrum((ey, dt), F0)
        thg_Pw.append(hs["P_w"])
        thg_P3.append(hs["P_3w"])
    return {
        "dt": dt, "ey_off": ey_off, "ey_on": ey_on, "hx_on": hx_on,
        "shg_Pw": np.array(shg_Pw), "shg_P2": np.array(shg_P2),
        "thg_Pw": np.array(thg_Pw), "thg_P3": np.array(thg_P3),
        "thg_floor": harmonic_spectrum((ey_k0, dt), F0),
        "off": harmonic_spectrum((ey_off, dt), F0),
    }


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy.signal.hilbert needed for the analytic-pump closed form")
def test_gate1_shg_closed_form(fdtd_gates):
    """GATE 1: reproduce the undepleted coupled-wave SHG closed form THROUGH harmonic_spectrum."""
    g = fdtd_gates
    dt = g["dt"]
    hs_on = harmonic_spectrum((g["ey_on"], dt), F0)
    z1 = hilbert(g["ey_off"])                                # measured pump analytic signal
    e2_pred = (CHI2 * 2.0 * np.pi * F0 * L_WIN / (2.0 * N_MED * C_LIGHT)) * np.imag(z1 ** 2)
    hs_pred = harmonic_spectrum((e2_pred, dt), F0)
    rel_amp = abs(np.sqrt(hs_on["P_2w"]) - np.sqrt(hs_pred["P_2w"])) / np.sqrt(hs_pred["P_2w"])
    assert rel_amp < 5e-2                                    # the oracle's own tolerance


def test_gate2a_shg_slope(fdtd_gates):
    """GATE 2 (chi2): P_2w ~ P_w^2 over 4 pump amplitudes."""
    g = fdtd_gates
    sl = power_slope(g["shg_Pw"], g["shg_P2"])
    assert abs(sl["slope"] - 2.0) < 0.02
    assert sl["r2"] > 0.999


def test_gate2b_thg_slope_and_radiates(fdtd_gates):
    """GATE 2 (chi3): the instantaneous Kerr term radiates a genuine 3w line -> P_3w ~ P_w^3."""
    g = fdtd_gates
    p3_on = g["thg_P3"][0]
    p3_floor = g["thg_floor"]["P_3w"]
    assert p3_on / p3_floor > 1e3                            # THG rises far above the numerical floor
    sl = power_slope(g["thg_Pw"], g["thg_P3"])
    assert abs(sl["slope"] - 3.0) < 0.05
    assert sl["r2"] > 0.999


def test_gate3_zero_nonlinearity_floor(fdtd_gates):
    """GATE 3: with no nonlinearity the 2w/3w bands sit > 60 dB below the fundamental."""
    off = fdtd_gates["off"]
    assert off["P_2w"] / off["P_w"] < 1e-6                   # 60 dB in power
    assert off["P_3w"] / off["P_w"] < 1e-6


def test_gate4_doubling_power_quadruples_shg(fdtd_gates):
    """GATE 4 (diagnostic linearity): the pump POWER doubles each step, so P_2w must quadruple."""
    g = fdtd_gates
    assert np.allclose(g["shg_Pw"][1:] / g["shg_Pw"][:-1], 2.0, rtol=0.02)
    assert np.allclose(g["shg_P2"][1:] / g["shg_P2"][:-1], 4.0, rtol=0.02)


def test_conversion_efficiency_and_undepleted_physical(fdtd_gates):
    """conversion_efficiency + undepleted_validity on the FDTD field: small eta, and the measured
    pump depletion matches the converted fraction (energy conservation, Delta_k = 0)."""
    g = fdtd_gates
    dt = g["dt"]
    ce = conversion_efficiency((g["ey_on"], dt), F0, incident=(g["ey_off"], dt))
    assert 1e-6 < ce["eta_shg"] < 1e-3 and ce["normalization"] == "incident"
    uv = undepleted_validity((g["ey_on"], dt), F0, incident=(g["ey_off"], dt))
    assert uv["undepleted"] is True
    # converted fraction == measured pump loss to within the perturbative-window smearing
    assert uv["depletion_measured"] == pytest.approx(uv["eta_total"], rel=0.15)


# =================================================================================================
# solver-path: additive opt-in kwarg (byte-identity) + order-resolved result path
# =================================================================================================
_SOLVER_KW = dict(period_x_m=100e-9, lambda_min_m=1.05e-6, lambda_max_m=1.35e-6, resolution=18,
                  source_amp=5.0e8, backend="numpy")


@pytest.fixture(scope="module")
def solver_runs():
    lay = [FDTDLayer(200e-9, eps_inf=2.0, chi2_m_V=2.0e-11)]
    r_default = solve_fdtd_2d(lay, **_SOLVER_KW)                              # no kwarg (legacy path)
    r_false = solve_fdtd_2d(lay, **_SOLVER_KW, return_time_trace=False)       # present-but-False
    r_true = solve_fdtd_2d(lay, **_SOLVER_KW, return_time_trace=True)         # probe on
    return r_default, r_false, r_true


def test_gate5_return_time_trace_byte_identity(solver_runs):
    """GATE 5: the opt-in probe changes nothing in the physics arrays (nan-aware: the out-of-band
    0/0 rfft bins are identical in both runs), and only return_time_trace=True attaches a trace."""
    r_default, r_false, r_true = solver_runs
    for nm in ("freqs_Hz", "R0", "T0", "R_flux", "T_flux", "band", "r0", "t0"):
        np.testing.assert_array_equal(getattr(r_default, nm), getattr(r_true, nm))
        np.testing.assert_array_equal(getattr(r_default, nm), getattr(r_false, nm))
    assert r_default.time_trace is None and r_false.time_trace is None
    assert r_true.time_trace is not None


def test_solver_result_harmonic_order_resolved(solver_runs):
    """harmonic_spectrum reads the FDTD2DResult trace: the Poynting-flux SHG band, order-resolved.
    A laterally uniform chi2 slab radiates only the specular (0) order; Parseval holds."""
    _, _, r_true = solver_runs
    f0_solver = C_LIGHT / 1.2e-6
    hs = harmonic_spectrum(r_true, f0_solver)
    assert hs["power_type"] == "poynting_flux"
    orders = list(hs["orders"])
    p2 = hs["power_by_order"][2]
    assert p2[orders.index(0)] > 0.99 * p2.sum()             # specular for a uniform slab
    assert hs["power_by_order"][2].sum() == pytest.approx(hs["P_2w"], rel=1e-10)
    # SHG present and small (undepleted) relative to the transmitted fundamental
    assert 0.0 < hs["P_2w"] / hs["P_w"] < 1e-3
    ce = conversion_efficiency(r_true, f0_solver)            # incident_right from the trace
    assert ce["normalization"] == "incident" and ce["eta_shg"] > 0.0
