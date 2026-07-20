"""Roadmap 4.2 gates: bichromatic (two-color) FDTD source + three-/four-wave mixing diagnostics.

Two deliverables under test (conventions: SI, exp(-i omega t), pure numpy, ASCII):

  (a) solve_fdtd_1d(second_source=...) -- an ADDITIVE opt-in second carrier superposed in the same
      injected soft-source waveform. GATE 1 pins byte-identity of the second_source=None path
      (tobytes incl. the DFT-zero NaN payloads) and re-runs the pre-edit FDTD coverage gates; the
      two-color injection is verified by EXACT linear superposition (vacuum propagation is linear, so
      incident(bichromatic) - incident(single) is a pure second-color wave scaling exactly with
      amplitude_rel). NOTE the 1-D solver ties the pulse bandwidth to the R/T band, so two in-band
      colors overlap spectrally (each color's spectral std ~ half the band); the composite-source
      machinery + per-color incident bookkeeping is the deliverable, and the mixing PHYSICS (which
      needs chi2, carried only by the 2-D-TE kernels) is gated on run_2d_te below.

  (b) harmonics.mixing_spectrum(trace, f1, f2) -- band powers at {f1, f2, f1+f2, |f1-f2|, 2f1, 2f2,
      2f1-f2, 2f2-f1} with a per-band leakage guard generalized from harmonic_spectrum's pump-
      bandwidth guard, band-overlap validation (raise), and degenerate-spacing (f2=2f1) merge/annotate.

  2-6 drive a chi2 slab in the 2-D-TE kernel (run_2d_te) with a two-color source ARRAY (the same
  physical composite the 1-D source builds, here in the chi2-capable engine):
    GATE 2  SFG bilinearity: P(f1+f2) linear in P1 at fixed P2 and vice versa (slope 1.00).
    GATE 3  DFG: the |f1-f2| idler appears with chi2, floors (>60 dB) without.
    GATE 4  OPA/parametric: idler generation + gain-scaling (P_idler linear in P_pump, i.e. idler
            amplitude ~ |A_p|) + Manley-Rowe photon bookkeeping + seed GAIN (dispersive, see below).
    GATE 5  zero-chi2 floors for every mixing band.
    GATE 6  guard behavior: overlapping-band configs raise; a broadband two-color pump warns per band.

  OPA PHASE-MATCHING CAVEAT (documented, gate 4d): in a NON-dispersive isotropic phase-matched chi2
  slab the seed cannot net-gain -- the sum-frequency up-conversion coupling sqrt(w_s w_sum) ALWAYS
  exceeds the parametric coupling sqrt(w_s w_i) (w_sum = w_s + w_p > w_i = w_p - w_s), so SFG drains
  the seed faster than the parametric process replenishes it (verified: G ~ 0.96 < 1, monotone in L).
  Genuine seed gain needs SFG phase-mismatched while the parametric process stays matched -- realized
  here with a Lorentz pole ABOVE the bands (normal dispersion mismatches SFG ~w_p/w_i times more than
  the parametric process). The parametric-gain closed form g = chi2 |A_p| sqrt(w_s w_i)/(2 n c) is an
  order-of-magnitude + SCALING oracle for a thin slab; the magnitude is gated within a factor ~2 and
  the |A_p| scaling within 15%.
"""
import warnings

import numpy as np
import pytest
import scipy.constants as sc

from dynameta.constants import C_LIGHT, EPS0
from dynameta.optics.fdtd import FDTDLayer, solve_fdtd_1d
from dynameta.optics.fdtd_nd import cpml_z, run_2d_te
from dynameta.optics.harmonics import mixing_spectrum, power_slope

try:
    from scipy.signal import hilbert
    HAVE_SCIPY = True
except Exception:                                            # pragma: no cover
    HAVE_SCIPY = False

# Optional cross-check against the concurrent 4.1 coupled-wave reference (graceful skip if not landed).
try:
    from dynameta.optics import twm_reference as _twm      # noqa: F401
    HAVE_TWM = True
except ImportError:
    HAVE_TWM = False

H_PLANCK = sc.h


# =================================================================================================
# Part A -- mixing_spectrum as a pure function (fast, no FDTD): exact-bin tones, guards, degeneracy
# =================================================================================================
def _multitone(amp_by_bin, *, N=1600, dt=1e-15):
    """Real signal with cosines EXACTLY on the given integer FFT bins (no spectral leakage)."""
    n = np.arange(N)
    e = np.zeros(N)
    for k, a in amp_by_bin.items():
        e += a * np.cos(2 * np.pi * (k / N) * n)
    return e, dt


def test_mixing_bands_exact_and_aliases():
    # f1 at bin 30, f2 at bin 41 (ratio ~1.37); place tones at f1, f2 and the SFG line f1+f2 (bin 71).
    N, dt = 1600, 1e-15
    a1, a2, a3 = 1.0, 0.7, 0.1
    f1 = 30 / (N * dt)
    f2 = 41 / (N * dt)
    e, dt = _multitone({30: a1, 41: a2, 71: a3}, N=N, dt=dt)
    ms = mixing_spectrum((e, dt), f1, f2, bandwidth_frac=0.05)
    assert ms["power_type"] == "e2_density"
    # an exact-bin cosine of amplitude a has |rfft|^2 = (a N/2)^2 in its single in-band bin
    assert ms["P_f1"] / ms["P_f2"] == pytest.approx((a1 / a2) ** 2, rel=1e-9)
    assert ms["P_sum"] / ms["P_f1"] == pytest.approx((a3 / a1) ** 2, rel=1e-9)
    # bands with no tone (2f1 at bin 60, |f1-f2| at bin 11, ...) sit at the numerical floor
    assert ms["P_2f1"] / ms["P_f1"] < 1e-12
    assert ms["P_diff"] / ms["P_f1"] < 1e-12
    # alias <-> label bookkeeping and coeff/center metadata
    assert ms["power"]["f1+f2"] == ms["P_sum"]
    assert ms["coeffs"]["2f1-f2"] == (2, -1)
    assert ms["centers_hz"]["f1-f2"] == pytest.approx(f1 - f2, rel=1e-12)
    # narrowband, well-separated -> no contamination
    assert ms["pump_broadband"] is False


def test_mixing_input_guards():
    e, dt = _multitone({30: 1.0}, N=800)
    f0 = 30 / (800 * 1e-15)
    with pytest.raises(ValueError):                          # two DISTINCT colors required
        mixing_spectrum((e, dt), f0, f0)
    with pytest.raises(ValueError):                          # positive colors
        mixing_spectrum((e, dt), -f0, f0)
    with pytest.raises(ValueError):                          # bare array needs dt (reused _extract)
        mixing_spectrum(np.zeros(100), f0, 1.3 * f0)


def test_mixing_overlap_raises():
    # nearly-equal colors: the f1 and f2 bands (distinct centers) overlap -> double-count guard fires
    N, dt = 2000, 1e-15
    f1 = 40 / (N * dt)
    f2 = 41 / (N * dt)                                        # ratio 1.025 -> f1/f2 bands overlap
    e, _ = _multitone({40: 1.0, 41: 1.0}, N=N, dt=dt)
    with pytest.raises(ValueError, match="OVERLAP"):
        mixing_spectrum((e, dt), f1, f2, bandwidth_frac=0.05)
    # a wide band forces overlap even for the SFG colors (2f1-f2 vs 2f2 crowd)
    f1b = 30 / (N * dt)
    f2b = 41 / (N * dt)
    eb, _ = _multitone({30: 1.0, 41: 1.0}, N=N, dt=dt)
    with pytest.raises(ValueError, match="OVERLAP"):
        mixing_spectrum((eb, dt), f1b, f2b, bandwidth_frac=0.2)


def test_mixing_degenerate_f2_is_2f1_merges():
    # f2 = 2 f1: |f1-f2| coincides with f1, f2 with 2f1, f1+f2 with 2f2-f1, 2f1-f2 collapses onto DC.
    # This must MERGE/ANNOTATE, not raise.
    N, dt = 2000, 1e-15
    f1 = 20 / (N * dt)
    f2 = 40 / (N * dt)
    e, _ = _multitone({20: 1.0, 40: 0.5, 60: 0.2}, N=N, dt=dt)
    ms = mixing_spectrum((e, dt), f1, f2, bandwidth_frac=0.05)   # no raise
    # coincident groups flagged
    assert ms["band_degenerate"]["f1"] and ms["band_degenerate"]["f1-f2"]     # both at f1
    assert ms["band_degenerate"]["f2"] and ms["band_degenerate"]["2f1"]       # both at 2f1
    assert ms["band_degenerate"]["2f1-f2"]                                    # DC collapse
    groups = [set(g) for g in ms["degenerate_groups"]]
    assert {"f1", "f1-f2"} in groups
    assert {"f2", "2f1"} in groups


def test_mixing_broadband_guard_warns():
    """The per-band leakage guard (generalized single-pump guard): two BROADBAND (tau = 3 fs) pumps
    leak their spectral tails across the mixing bands -> UserWarning + pump_broadband flag; the
    narrowband (tau = 50 fs) twin stays clean."""
    f1, f2, dt = 1.82e14, 2.5e14, 5e-17

    def two_color(tau):
        t0 = 6.0 * tau
        n = int(round((12.0 * tau + 200e-15) / dt))
        t = np.arange(n) * dt
        env = np.exp(-((t - t0) / tau) ** 2)
        return env * (np.cos(2 * np.pi * f1 * (t - t0)) + np.cos(2 * np.pi * f2 * (t - t0)))

    with pytest.warns(UserWarning, match="BROADBAND"):
        ms = mixing_spectrum((two_color(3e-15), dt), f1, f2, bandwidth_frac=0.04)
    assert ms["pump_broadband"] is True
    assert any(ms["band_contaminated"].values())
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # any warning fails
        ms50 = mixing_spectrum((two_color(50e-15), dt), f1, f2, bandwidth_frac=0.04)
    assert ms50["pump_broadband"] is False


def test_mixing_global_guard_is_conservative():
    """Verifier gate (2026-07-20): the GLOBAL pump_broadband flag is CONSERVATIVE. Across a tau
    sweep of PURE two-tone (zero-nonlinearity) Gaussian pumps -- so EVERY mixing band is phantom
    (pump-skirt leakage, the phantom-SHG hazard) -- whenever ANY mixing band carries phantom power
    > 1e-6 of the pump, pump_broadband must be True. This pins the reliable contract: the per-band
    band_contaminated flags can under-report FAR bands once the two colors MERGE into one broadband
    blob (their skirts cross the half-way point; the per-pump sigma is then ill-defined), but the
    GLOBAL flag still fires there, so strict callers gate on it (see the mixing_spectrum guard note).
    Adversarial hunt found NO config with phantom > 1e-6 yet pump_broadband False."""
    f1, f2, dt = 1.82e14, 2.50e14, 2e-17
    labs = ("f1+f2", "f1-f2", "2f1", "2f2", "2f1-f2", "2f2-f1")

    def two_tone(tau):
        t0 = 7.0 * tau
        n = int(round((14.0 * tau + 300e-15) / dt))
        t = np.arange(n) * dt
        env = np.exp(-((t - t0) / tau) ** 2)
        return env * (np.cos(2 * np.pi * f1 * (t - t0)) + np.cos(2 * np.pi * f2 * (t - t0)))

    for tau in (3e-15, 5e-15, 8e-15, 12e-15, 20e-15, 50e-15):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ms = mixing_spectrum((two_tone(tau), dt), f1, f2, bandwidth_frac=0.05)
        pump = max(ms["P_f1"], ms["P_f2"])
        max_phantom = max(ms["power"][l] / pump for l in labs)
        if max_phantom > 1e-6:
            assert ms["pump_broadband"] is True, (tau, max_phantom)


# =================================================================================================
# Part B -- 1-D bichromatic source: byte-identity + exact two-color superposition + guards
# =================================================================================================
def _byteid_slab():
    return ([FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0)],
            dict(lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30))


def test_gate1_second_source_none_byte_identical():
    layers, kw = _byteid_slab()
    r_default = solve_fdtd_1d(layers, **kw)                           # no kwarg
    r_none = solve_fdtd_1d(layers, **kw, second_source=None)          # present-but-None
    r_none_f = solve_fdtd_1d(layers, **kw, second_source=None, return_time_trace=False)
    for r in (r_none, r_none_f):
        assert r.R.tobytes() == r_default.R.tobytes()                 # incl. the DFT-zero NaN payloads
        assert r.T.tobytes() == r_default.T.tobytes()
        assert r.freqs_Hz.tobytes() == r_default.freqs_Hz.tobytes()
        assert np.array_equal(r.band, r_default.band)
        assert np.array_equal(r.R, r_default.R, equal_nan=True)
        assert np.array_equal(r.T, r_default.T, equal_nan=True)
    assert r_default.time_trace is None and r_none.time_trace is None


def test_gate1_existing_fdtd_infra_gates_still_pass():
    # re-run the pre-edit FDTD coverage gates (same pattern as ringdown/time-varying gate 1); the
    # additive second_source path must not perturb them.
    from test_audit_2026_07_17_infra import (test_fdtd_1d_dielectric_slab_vs_airy,
                                             test_fdtd_1d_drude_slab_absorbs)
    test_fdtd_1d_dielectric_slab_vs_airy()
    test_fdtd_1d_drude_slab_absorbs()


def _bichromatic_1d(amp_rel):
    lam_min, lam_max = 0.9e-6, 1.9e-6
    f_min, f_max = C_LIGHT / lam_max, C_LIGHT / lam_min
    f_c = 0.5 * (f_min + f_max)
    f2 = f_c / 1.37                                           # in-band, well-separated ratio
    lay = [FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0)]
    kw = dict(lambda_min_m=lam_min, lambda_max_m=lam_max, resolution=30, source_amp=1.0,
              return_time_trace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                      # broadband-pump guard fires on the 1-D pulse
        r = solve_fdtd_1d(lay, **kw, second_source=dict(f_hz=f2, amplitude_rel=amp_rel))
    return r, f_c, f2, kw, lay


def test_1d_bichromatic_exact_linear_superposition():
    """Vacuum propagation is LINEAR, so incident(bichromatic) - incident(single) is a PURE second-
    color wave whose amplitude scales EXACTLY with amplitude_rel -- the rigorous proof the second
    carrier injects additively at the requested frequency and relative amplitude."""
    lay = [FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0)]
    lam_min, lam_max = 0.9e-6, 1.9e-6
    kw = dict(lambda_min_m=lam_min, lambda_max_m=lam_max, resolution=30, source_amp=1.0,
              return_time_trace=True)
    f_c = 0.5 * (C_LIGHT / lam_max + C_LIGHT / lam_min)
    f2 = f_c / 1.37
    r_single = solve_fdtd_1d(lay, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_full = solve_fdtd_1d(lay, **kw, second_source=dict(f_hz=f2, amplitude_rel=1.0))
        r_half = solve_fdtd_1d(lay, **kw, second_source=dict(f_hz=f2, amplitude_rel=0.5))
    inc_s = np.asarray(r_single.time_trace["incident_right"], float)
    d_full = np.asarray(r_full.time_trace["incident_right"], float) - inc_s
    d_half = np.asarray(r_half.time_trace["incident_right"], float) - inc_s
    # exact linear scaling: d_full == 2 * d_half to machine precision
    assert np.max(np.abs(d_full - 2.0 * d_half)) / np.max(np.abs(d_full)) < 1e-10
    # the isolated second-color wave lives at f2 (its spectral peak is far closer to f2 than to f_c)
    dt = float(r_single.time_trace["dt"])
    f = np.fft.rfftfreq(d_full.size, dt)
    fpk = f[np.argmax(np.abs(np.fft.rfft(d_full)) ** 2)]
    assert abs(fpk - f2) < abs(fpk - f_c)
    assert abs(fpk - f2) / f2 < 0.10
    # the carrier frequencies are exposed on the trace
    assert r_full.time_trace["f_carrier_hz"] == pytest.approx(f_c, rel=1e-12)
    assert r_full.time_trace["f_carrier2_hz"] == pytest.approx(f2, rel=1e-12)


def test_1d_bichromatic_per_color_bookkeeping_and_guard():
    """The band extractor reads per-color incident powers; and because the 1-D pulse bandwidth ~ the
    color separation, the raw two-color incident is (honestly) flagged broadband by the guard."""
    r, f_c, f2, _kw, _lay = _bichromatic_1d(0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = mixing_spectrum(r.time_trace, f_c, f2, field="incident_right", bandwidth_frac=0.03)
    assert ms["P_f1"] > 0.0 and ms["P_f2"] > 0.0             # both colors present
    assert ms["pump_broadband"] is True                     # 1-D fixed-bandwidth pulse -> overlapping colors


def test_second_source_guards():
    lay = [FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0)]
    kw = dict(lambda_min_m=1.2e-6, lambda_max_m=1.6e-6, resolution=24)
    with pytest.raises(ValueError):                          # need exactly one of f_hz / lambda0_m
        solve_fdtd_1d(lay, **kw, second_source={})
    with pytest.raises(ValueError):
        solve_fdtd_1d(lay, **kw, second_source=dict(f_hz=2e14, lambda0_m=1.4e-6))
    f_out = C_LIGHT / 1.0e-6                                 # below lambda_min -> above f_max -> out of band
    with pytest.warns(UserWarning, match="OUTSIDE"):
        solve_fdtd_1d(lay, **kw, second_source=dict(f_hz=f_out, amplitude_rel=0.1))


# =================================================================================================
# Part C -- 2-D chi2 mixing: SFG bilinearity / DFG idler / zero-chi2 floors
# =================================================================================================
_NX = 4


def _grid(dz, npad, nstr, courant=0.5):
    nz = 2 * npad + nstr
    dx = 4.0 * dz
    dt = courant / (C_LIGHT * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))
    return nz, dx, dt


def _run2d(a1, f1, a2, f2, *, dz, npad, nstr, n_med, tau, chi2_val=0.0,
           d_eps=0.0, f0_lor=0.0, extra_s=180e-15):
    """One 2-D-TE pass with a TWO-COLOR source array over an (optionally Lorentz-dispersive) chi2 slab.
    Returns the x-mean transmitted (E_y, H_x) probe series + dt (poynting-flux mixing diagnostics)."""
    nz, dx, dt = _grid(dz, npad, nstr)
    t0 = 6.0 * tau
    nsteps = int(round((2.0 * t0 + extra_s + 4.0 * nz * dz / C_LIGHT) / dt))
    t = np.arange(nsteps) * dt
    env = np.exp(-((t - t0) / tau) ** 2)
    src = a1 * env * np.cos(2 * np.pi * f1 * (t - t0)) + a2 * env * np.cos(2 * np.pi * f2 * (t - t0))
    eps = np.full((_NX, nz), float(n_med) ** 2)
    zeros = np.zeros((_NX, nz))
    sl = slice(npad, npad + nstr)
    lor = None
    if d_eps != 0.0:
        w0 = 2 * np.pi * f0_lor
        lw = np.zeros((_NX, nz)); lg = np.zeros((_NX, nz)); ld = np.zeros((_NX, nz))
        lw[:, sl] = w0; ld[:, sl] = d_eps
        den = 1.0 + lg * dt / 2.0
        lor = ((2.0 - lw ** 2 * dt ** 2) / den, (lg * dt / 2.0 - 1.0) / den,
               (EPS0 * ld * lw ** 2 * dt ** 2) / den)
    chi2 = None
    if chi2_val:
        chi2 = np.zeros((_NX, nz)); chi2[:, sl] = chi2_val
    cpml = cpml_z(nz, dz, dt, 14, n_med, n_med)
    eyL, hxL, eyR, hxR = run_2d_te(eps, zeros, zeros, zeros, dx, dz, dt, nsteps,
                                   16, 22, nz - 16, src, cpml, np, lor, chi2=chi2)
    return eyR.mean(axis=1), hxR.mean(axis=1), dt


# SFG/DFG/floor config (non-dispersive, index-matched -> Delta_k = 0, all mixing phase-matched)
_SF = dict(dz=10e-9, npad=48, nstr=40, n_med=np.sqrt(2.0), tau=50e-15)
F1, F2 = 1.82e14, 2.50e14                                    # ratio ~1.37, well separated, non-harmonic
CHI2 = 2.0e-11
A0 = 5.0e8
BW = 0.05


@pytest.fixture(scope="module")
def sfg_runs():
    ey_on, hx_on, dt = _run2d(A0, F1, A0, F2, chi2_val=CHI2, **_SF)
    ey_off, hx_off, _ = _run2d(A0, F1, A0, F2, chi2_val=0.0, **_SF)
    # SFG bilinearity sweeps: vary A1 at fixed A2, then A2 at fixed A1 (measure from the SAME trace)
    mults = np.array([1.0, np.sqrt(2.0), 2.0, 2.0 * np.sqrt(2.0)]) * 0.8 * A0
    P1, Psum_1, P2, Psum_2 = [], [], [], []
    for a in mults:
        e1, h1, _ = _run2d(a, F1, A0, F2, chi2_val=CHI2, **_SF)
        m1 = mixing_spectrum((e1, h1, dt), F1, F2, bandwidth_frac=BW)
        P1.append(m1["P_f1"]); Psum_1.append(m1["P_sum"])
        e2, h2, _ = _run2d(A0, F1, a, F2, chi2_val=CHI2, **_SF)
        m2 = mixing_spectrum((e2, h2, dt), F1, F2, bandwidth_frac=BW)
        P2.append(m2["P_f2"]); Psum_2.append(m2["P_sum"])
    return {"dt": dt,
            "on": mixing_spectrum((ey_on, hx_on, dt), F1, F2, bandwidth_frac=BW),
            "off": mixing_spectrum((ey_off, hx_off, dt), F1, F2, bandwidth_frac=BW),
            "P1": np.array(P1), "Psum_1": np.array(Psum_1),
            "P2": np.array(P2), "Psum_2": np.array(Psum_2)}


def test_gate2_sfg_bilinearity(sfg_runs):
    """GATE 2: P(f1+f2) is linear in P1 at fixed P2 (slope 1.00) and vice versa."""
    g = sfg_runs
    s1 = power_slope(g["P1"], g["Psum_1"])
    s2 = power_slope(g["P2"], g["Psum_2"])
    assert abs(s1["slope"] - 1.0) < 0.05 and s1["r2"] > 0.999
    assert abs(s2["slope"] - 1.0) < 0.05 and s2["r2"] > 0.999


def test_gate3_dfg_idler_appears(sfg_runs):
    """GATE 3: the DFG idler at |f1-f2| appears with chi2 on and floors (> 60 dB down) with chi2 off."""
    on, off = sfg_runs["on"], sfg_runs["off"]
    # NOTE the chi2-off floor is a SIGNED Poynting band power at machine noise (~1e-10 of the
    # pump) whose SIGN is platform-dependent (observed negative on CI BLAS builds) -- always
    # ratio against its MAGNITUDE.
    assert on["P_diff"] / on["P_f1"] > 1e-8                  # a real difference-frequency band
    assert on["P_diff"] / abs(off["P_diff"]) > 1e6          # >> 60 dB above the chi2-off floor
    assert abs(off["P_diff"]) / off["P_f1"] < 1e-6          # chi2 off: idler at the numerical floor


def test_gate5_zero_chi2_floors(sfg_runs):
    """GATE 5: with chi2 off EVERY generated mixing band sits > 60 dB below the pump bands."""
    off = sfg_runs["off"]
    pump = max(off["P_f1"], off["P_f2"])
    for lab in ("f1+f2", "f1-f2", "2f1", "2f2", "2f1-f2", "2f2-f1"):
        assert abs(off["power"][lab]) / pump < 1e-6, lab     # |.|: signed Poynting noise floor
    # and with chi2 ON the sum / SHG bands are genuinely radiated
    on = sfg_runs["on"]
    for lab in ("f1+f2", "2f1", "2f2"):
        assert on["power"][lab] / on["P_f1"] > 1e-8, lab


# =================================================================================================
# Part D -- OPA / parametric amplification
# =================================================================================================
# non-dispersive OPA config: strong pump f_p, weak seed f_s, idler f_i = f_p - f_s
_OPA = dict(dz=10e-9, npad=48, nstr=120, n_med=np.sqrt(2.0), tau=60e-15)
F_S, F_P = 1.5e14, 2.5e14
F_I = F_P - F_S
F_SUM = F_S + F_P
L_OPA = _OPA["nstr"] * _OPA["dz"]
AS = 4.0e8


@pytest.fixture(scope="module")
def opa_runs():
    # idler generation + Manley-Rowe: strong pump (undepleted but with measurable conversion)
    Ap_mr = 4.0e9
    ey_on, hx_on, dt = _run2d(AS, F_S, Ap_mr, F_P, chi2_val=CHI2, **_OPA)
    ey_po, hx_po, _ = _run2d(AS, F_S, 0.0, F_P, chi2_val=CHI2, **_OPA)      # pump OFF (seed reference)
    ey_c0, hx_c0, _ = _run2d(AS, F_S, Ap_mr, F_P, chi2_val=0.0, **_OPA)     # chi2 OFF (idler floor)
    m_on = mixing_spectrum((ey_on, hx_on, dt), F_S, F_P, bandwidth_frac=BW)
    m_po = mixing_spectrum((ey_po, hx_po, dt), F_S, F_P, bandwidth_frac=BW)
    m_c0 = mixing_spectrum((ey_c0, hx_c0, dt), F_S, F_P, bandwidth_frac=BW)
    # gain scaling: idler power vs pump power at fixed seed (perturbative range)
    Aps = np.array([6e8, 9e8, 1.2e9, 1.5e9])
    Ppump, Pidler = [], []
    for ap in Aps:
        e, h, _ = _run2d(AS, F_S, ap, F_P, chi2_val=CHI2, **_OPA)
        m = mixing_spectrum((e, h, dt), F_S, F_P, bandwidth_frac=BW)
        Ppump.append(m["P_f2"]); Pidler.append(m["P_diff"])
    # pump field for the closed-form magnitude (pump-only run at a reference amplitude)
    Ap_ref = 1.2e9
    ey_p, hx_p, _ = _run2d(0.0, F_S, Ap_ref, F_P, chi2_val=CHI2, **_OPA)
    e_ref, h_ref, _ = _run2d(AS, F_S, Ap_ref, F_P, chi2_val=CHI2, **_OPA)
    m_ref = mixing_spectrum((e_ref, h_ref, dt), F_S, F_P, bandwidth_frac=BW)
    return {"dt": dt, "on": m_on, "po": m_po, "c0": m_c0,
            "Ppump": np.array(Ppump), "Pidler": np.array(Pidler),
            "Ap_field": (float(np.max(np.abs(hilbert(ey_p)))) if HAVE_SCIPY else None),
            "gL_ref": (np.sqrt(m_ref["P_diff"] / m_ref["P_f1"]) * np.sqrt(F_S / F_I))}


def test_gate4a_idler_generation(opa_runs):
    """GATE 4a: the parametric idler at f_p - f_s appears far above the chi2-off floor."""
    g = opa_runs
    assert g["on"]["P_diff"] > 0.0                          # a real, positive parametric idler band
    assert g["on"]["P_diff"] > 1e6 * abs(g["c0"]["P_diff"])  # far above the chi2-off floor (|.|: floor is ~0 noise)
    assert g["on"]["P_diff"] / g["on"]["P_f1"] > 1e-6


def test_gate4b_parametric_gain_scaling(opa_runs):
    """GATE 4b: parametric gain SCALING -- the idler POWER is linear in the pump POWER at fixed seed
    (idler amplitude ~ |A_p| = sqrt(P_pump), the g ~ kappa |A_p| law), and the gL magnitude matches
    the undepleted closed form within a factor ~2 (order-of-magnitude, thin-slab caveat)."""
    g = opa_runs
    s = power_slope(g["Ppump"], g["Pidler"])
    assert abs(s["slope"] - 1.0) < 0.15 and s["r2"] > 0.99  # idler amplitude linear in |A_p|
    if HAVE_SCIPY and g["Ap_field"]:
        ws, wi = 2 * np.pi * F_S, 2 * np.pi * F_I
        gL_pred = (CHI2 * g["Ap_field"] / (2.0 * _OPA["n_med"] * C_LIGHT)) * np.sqrt(ws * wi) * L_OPA
        ratio = g["gL_ref"] / gL_pred
        assert 0.5 < ratio < 2.0, ratio                     # closed form to a factor ~2


def test_gate4c_manley_rowe_photon_bookkeeping(opa_runs):
    """GATE 4c: Manley-Rowe from band powers (photon flux = P_band / (h f)). In the non-dispersive
    phase-matched slab the parametric down-conversion (pump -> signal + idler) COMPETES with sum-
    frequency up-conversion (signal + pump -> sum) and signal SHG; the exact photon bookkeeping
    dN_signal = dN_idler - dN_sum - 2 dN_2fs (each DFG makes a signal+idler pair, each SFG destroys a
    signal for a sum, each signal-SHG destroys two signals) closes from the measured band powers."""
    g = opa_runs
    on, po = g["on"], g["po"]

    def dN(lab, f):
        return (on["power"][lab] - po["power"][lab]) / (H_PLANCK * f)

    dN_s = dN("f1", F_S)
    dN_i = dN("f1-f2", F_I)
    dN_sum = dN("f1+f2", F_SUM)
    dN_2s = dN("2f1", 2 * F_S)
    predicted = dN_i - dN_sum - 2 * dN_2s
    assert dN_s < 0.0                                        # non-dispersive: SFG drains the seed net
    assert abs(dN_s / predicted - 1.0) < 0.2                # Manley-Rowe photon accounting closes


def test_gate4c_independent_ode_confirms_seed_drain_mechanism():
    """INDEPENDENT (non-FDTD) coupled-wave oracle for the module's OPA PHASE-MATCHING CAVEAT
    (verifier 2026-07-20). A dense scipy ODE integrates the chi2 frequency comb A_m at w_m = m*w0
    (SVEA, Boyd ch.2) for the exact non-dispersive _OPA slab (n const => every sum/difference is
    phase-matched, Delta_k = 0). This is an INDEPENDENT code path from the FDTD -- it refutes-or-
    confirms the headline claim 'a non-dispersive phase-matched chi2 slab cannot net-gain a seed'
    without re-using the FDTD kernel. It asserts (i) the SIGN of the net seed change is NEGATIVE
    (seed net-LOSES), and (ii) disabling ONLY the s+p->sum SFG term flips the seed to GAIN --
    proving SFG (coupling ~sqrt(w_s w_sum)) is the drain that beats the parametric process
    (~sqrt(w_s w_i)), exactly the mechanism gate 4c/4d rely on."""
    from scipy.integrate import solve_ivp
    w0 = 2.0 * np.pi * 0.5e14
    n_med, chi2, L = np.sqrt(2.0), CHI2, L_OPA
    modes = list(range(1, 13))
    idx = {m: k for k, m in enumerate(modes)}
    S_M, P_M, SUM_M = 3, 5, 8                                # s=1.5e14, p=2.5e14, sum=4.0e14
    sfg = {m: [(a, b, 1.0 if a == b else 2.0)
               for a in modes for b in modes if a <= b and a + b == m] for m in modes}
    dfg = {m: [(a, b) for a in modes for b in modes if a > b and a - b == m] for m in modes}

    def rhs(z, y, kill_sfg_sum):
        A = y[:len(modes)] + 1j * y[len(modes):]
        dA = np.zeros(len(modes), complex)
        for m in modes:
            NL = 0j
            for (a, b, g) in sfg[m]:
                if kill_sfg_sum and m == SUM_M and {a, b} == {S_M, P_M}:
                    continue                                 # remove the s+p->sum up-conversion only
                NL += g * A[idx[a]] * A[idx[b]]
            for (a, b) in dfg[m]:
                NL += 2.0 * A[idx[a]] * np.conj(A[idx[b]])
            dA[idx[m]] = 1j * (m * w0 * chi2 / (2.0 * n_med * C_LIGHT)) * NL
        return np.concatenate([dA.real, dA.imag])

    def seed_gain(kill):
        A0 = np.zeros(len(modes), complex)
        A0[idx[S_M]] = 4e6
        A0[idx[P_M]] = 3e8
        sol = solve_ivp(rhs, [0.0, L], np.concatenate([A0.real, A0.imag]),
                        rtol=1e-9, atol=1e-3, max_step=L / 1500, args=(kill,))
        Af = sol.y[:len(modes), -1] + 1j * sol.y[len(modes):, -1]
        return abs(Af[idx[S_M]]) ** 2 / abs(A0[idx[S_M]]) ** 2

    assert seed_gain(False) < 1.0                            # non-dispersive phase-matched: seed NET-LOSES
    assert seed_gain(True) > 1.0                             # kill s+p->sum SFG => seed GAINS (SFG is the drain)


def test_gate4d_seed_gains_with_sfg_phase_mismatch():
    """GATE 4d: the seed BAND GAINS vs the pump-off run when the competing sum-frequency up-conversion
    is dispersively PHASE-MISMATCHED (a Lorentz pole above the bands; see the module PHASE-MATCHING
    CAVEAT). The non-dispersive slab cannot show this (SFG always wins) -- documented + gated at 4c."""
    disp = dict(dz=8e-9, npad=60, nstr=160, n_med=np.sqrt(2.0), tau=70e-15,
                d_eps=2.0, f0_lor=6.0e14, extra_s=220e-15)
    fs, fp = 2.0e14, 2.5e14
    ap = 3.0e9
    ey_on, hx_on, dt = _run2d(4e8, fs, ap, fp, chi2_val=1.5e-11, **disp)
    ey_po, hx_po, _ = _run2d(4e8, fs, 0.0, fp, chi2_val=1.5e-11, **disp)
    assert np.all(np.isfinite(ey_on)) and np.all(np.isfinite(ey_po))     # stably below the chi2 threshold
    m_on = mixing_spectrum((ey_on, hx_on, dt), fs, fp, bandwidth_frac=0.04)
    m_po = mixing_spectrum((ey_po, hx_po, dt), fs, fp, bandwidth_frac=0.04)
    G = m_on["P_f1"] / m_po["P_f1"]
    assert G > 1.0002                                        # seed GAINS (measured ~1.0008, SFG suppressed)
    # the SFG sum band is strongly suppressed by the phase mismatch (a phase-matched non-dispersive
    # slab at this drive radiates a sum band orders of magnitude larger relative to the seed)
    assert m_on["P_sum"] / m_on["P_f1"] < 1e-3


@pytest.mark.skipif(not HAVE_TWM, reason="twm_reference (roadmap 4.1) not landed yet")
def test_twm_reference_crosscheck(opa_runs):
    """Cross-check the measured parametric gL against the 4.1 coupled-wave reference
    (twm_reference.opa_gain): same slab (d_eff = chi2/2 convention), same measured pump field,
    dk = 0 exactly (equal constant indices). The oracle's gL must equal the in-test closed form
    (independent code path, same physics) and bound the FDTD-measured gL to the gate-4b factor."""
    g = opa_runs
    if not (HAVE_SCIPY and g["Ap_field"]):
        pytest.skip("needs scipy hilbert for the pump-field magnitude")
    ws, wi = 2 * np.pi * F_S, 2 * np.pi * F_I
    spec = _twm.TWMSpec(omega1=ws, omega2=wi, d_eff=CHI2 / 2.0, length=L_OPA,
                        n1=_OPA["n_med"], n2=_OPA["n_med"], n3=_OPA["n_med"])
    ref = _twm.opa_gain(spec, amp_pump=g["Ap_field"], amp_signal=1.0)
    assert ref["above_threshold"]
    gL_twm = abs(ref["gL"])
    gL_selfform = (CHI2 * g["Ap_field"] / (2.0 * _OPA["n_med"] * C_LIGHT)) * np.sqrt(ws * wi) * L_OPA
    assert abs(gL_twm / gL_selfform - 1.0) < 1e-12          # oracle == closed form, dk = 0
    assert 0.3 < g["gL_ref"] / gL_twm < 3.0                 # FDTD-measured gL vs the oracle
