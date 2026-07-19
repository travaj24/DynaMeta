"""Gates for optics/ringdown.py (roadmap 1.2 -- ringdown harmonic inversion by the matrix-pencil
method, Hua & Sarkar IEEE TAP 38:814 (1990)) and the additive opt-in FDTD time-trace probe.

Convention under test (see ringdown.py docstring): a real field trace decays as exp(-gamma t/2)
in amplitude (energy ~ exp(-gamma t)); q = omega_0 / gamma is the ENERGY Q and equals the
pole-finder convention Re(omega_t)/(2|Im omega_t|) for the same resonance.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks, hilbert

from dynameta.constants import C_LIGHT
from dynameta.optics.ringdown import (Mode, matrix_pencil, ringdown_q,
                                      fdtd_etalon_ringdown)


# ---- Gate 1: SYNTHETIC EXACT (3 well-separated damped cosines) -----------------------------

def test_gate1_three_damped_cosines_exact():
    dt, N = 2.0e-3, 500
    t = np.arange(N) * dt
    # (omega_rad_s, gamma_rad_s, amplitude A, phase phi): y_k = A e^{-gamma t/2} cos(omega t + phi)
    truth = [(2 * np.pi * 40.0, 20.0, 1.0, 0.4),
             (2 * np.pi * 95.0, 8.0, 0.6, -1.1),
             (2 * np.pi * 160.0, 45.0, 0.8, 2.0)]
    y = np.zeros(N)
    for w, g, A, ph in truth:
        y += A * np.exp(-g * t / 2.0) * np.cos(w * t + ph)

    modes = sorted(matrix_pencil(y, dt), key=lambda m: m.omega_rad_s)
    assert len(modes) == 3
    for m, (w, g, A, ph) in zip(modes, truth):
        # real-signal amplitude convention: y ~ Re(amplitude e^{-i omega_t t}) => amplitude = A e^{-i phi}
        a_true = A * np.exp(-1j * ph)
        assert abs(m.omega_rad_s - w) / w < 1e-8
        assert abs(m.q - w / g) / (w / g) < 1e-6
        assert abs(m.amplitude - a_true) < 1e-6
        assert m.gamma_rad_s > 0.0 and m.omega_rad_s > 0.0


# ---- Gate 2: CLOSE PAIR (half a linewidth) -- the beyond-FFT claim -------------------------

def test_gate2_close_pair_beats_fft():
    # Two modes separated by HALF A LINEWIDTH. The power spectrum of e^{-gamma t/2} cos is a
    # Lorentzian of FWHM = gamma (rad/s); the two lines sit gamma/2 apart -- below the Rayleigh
    # limit, so the FFT merges them into a single resolvable peak while the matrix pencil (which
    # fits poles, not bins) still separates them.
    g0 = 40.0
    w1 = 2 * np.pi * 80.0
    w2 = w1 + 0.5 * g0                     # separation = half the linewidth gamma
    N, dt = 2500, 1.0e-3                   # long record: 1/T = 0.4 Hz << 3.18 Hz sep (NOT record-limited)
    t = np.arange(N) * dt
    y = (np.exp(-g0 * t / 2.0) * np.cos(w1 * t)
         + np.exp(-g0 * t / 2.0) * np.cos(w2 * t + 0.7))

    # FFT: a well-sampled (zero-padded) power spectrum; count PROMINENT peaks (a line that is not
    # separated by at least a 10%-prominence valley is not resolved). The FFT sees ONE peak.
    P = np.abs(np.fft.rfft(y, n=16 * N)) ** 2
    P = P / P.max()
    fft_peaks, _ = find_peaks(P, prominence=0.1)
    assert len(fft_peaks) == 1             # FFT FAILS to resolve the pair

    # Matrix pencil: recovers BOTH modes, each to omega rtol < 1e-4 (here ~1e-14, noise-free).
    modes = sorted([m for m in matrix_pencil(y, dt, amp_floor=1e-2) if m.omega_rad_s > 0.0],
                   key=lambda m: m.omega_rad_s)
    assert len(modes) == 2                 # MATRIX PENCIL resolves the pair
    assert abs(modes[0].omega_rad_s - w1) / w1 < 1e-4
    assert abs(modes[1].omega_rad_s - w2) / w2 < 1e-4


# ---- Gate 3: NOISE (1% white) -- accuracy + no hallucinated modes --------------------------

def test_gate3_noise_1pct_no_hallucination():
    rng = np.random.default_rng(0)
    dt, N = 1.0e-3, 2000
    t = np.arange(N) * dt
    truth = [(2 * np.pi * 40.0, 3.0, 1.0),
             (2 * np.pi * 95.0, 5.0, 0.9),
             (2 * np.pi * 160.0, 8.0, 0.8)]
    y = np.zeros(N)
    for w, g, A in truth:
        y += A * np.exp(-g * t / 2.0) * np.cos(w * t)
    y = y + 0.01 * np.max(np.abs(y)) * rng.standard_normal(N)   # 1% white noise

    modes = sorted(matrix_pencil(y, dt, svd_tol=1e-6, amp_floor=5e-2),
                   key=lambda m: m.omega_rad_s)
    assert len(modes) == 3                                       # model-order selection: no extras
    for m, (w, g, A) in zip(modes, truth):
        assert abs(m.omega_rad_s - w) / w < 1e-4
        assert abs(m.q - w / g) / (w / g) < 0.05                 # Q within 5%


# ---- Gate 4: REAL-SIGNAL convention (Q = omega/gamma vs measured energy half-life) ---------

def test_gate4_real_signal_energy_decay_convention():
    dt, N = 1.0e-4, 6000
    w, g = 2 * np.pi * 300.0, 50.0
    t = np.arange(N) * dt
    y = np.exp(-g * t / 2.0) * np.cos(w * t)                     # pure REAL decaying cosine

    modes = matrix_pencil(y, dt)
    phys = [m for m in modes if m.omega_rad_s > 0.0]
    assert len(phys) == 1                                        # ONE physical mode
    m = phys[0]
    assert m.omega_rad_s > 0.0 and m.gamma_rad_s > 0.0           # omega>0, gamma>0

    # documented Q convention: q == omega/gamma exactly
    assert abs(m.q - m.omega_rad_s / m.gamma_rad_s) < 1e-9
    assert abs(m.q - w / g) / (w / g) < 1e-6

    # NUMERICALLY verify gamma is the ENERGY decay rate: measure the energy half-life from the
    # sampled envelope and check q == omega * (t_half / ln2)  (i.e. Q = omega/gamma with
    # gamma = ln2 / t_half). Envelope via the analytic signal; skip the ends (edge transients).
    env = np.abs(hilbert(y))
    energy = env ** 2
    i0 = 50                                                      # a few samples in (avoid edge)
    half = 0.5 * energy[i0]
    below = np.where(energy[i0:] <= half)[0]
    assert below.size > 0
    t_half = below[0] * dt                                       # measured energy half-life
    gamma_meas = np.log(2.0) / t_half
    q_from_halflife = m.omega_rad_s / gamma_meas
    assert abs(m.q - q_from_halflife) / q_from_halflife < 0.02   # matches within 2%

    # ringdown_q convenience returns the same dominant mode
    f0, Q = ringdown_q(y, dt)
    assert abs(f0 - w / (2 * np.pi)) / (w / (2 * np.pi)) < 1e-6
    assert abs(Q - m.q) < 1e-9


# ---- Gate 5: FDTD BYTE-IDENTITY (default path unchanged by the additive probe) -------------

def _slab_kwargs():
    from dynameta.optics.fdtd import FDTDLayer
    return ([FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0)],
            dict(lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30))


def test_gate5_fdtd_default_path_byte_identical():
    from dynameta.optics.fdtd import solve_fdtd_1d
    layers, kw = _slab_kwargs()
    r_default = solve_fdtd_1d(layers, **kw)                          # no kwarg at all
    r_false = solve_fdtd_1d(layers, **kw, return_time_trace=False)   # kwarg present-but-False
    r_true = solve_fdtd_1d(layers, **kw, return_time_trace=True)     # probe on

    # Byte-identity of every legacy output across all three calls. NOTE: R/T carry NaN at the
    # DFT-divide-by-zero bins, so np.array_equal(equal_nan=True) is required (bare np.array_equal
    # treats NaN != NaN and would even fail an array against itself) -- and .tobytes() gives the
    # literal bit-for-bit identity the gate asks for (same NaN payloads, same everywhere).
    for a, b in ((r_default, r_false), (r_default, r_true)):
        assert a.R.tobytes() == b.R.tobytes()
        assert a.T.tobytes() == b.T.tobytes()
        assert a.freqs_Hz.tobytes() == b.freqs_Hz.tobytes()
        assert np.array_equal(a.band, b.band)               # bool mask: no NaN, plain equality
        assert np.array_equal(a.R, b.R, equal_nan=True)
        assert np.array_equal(a.T, b.T, equal_nan=True)

    # default / present-but-False attach NOTHING; only return_time_trace=True populates the probe
    assert r_default.time_trace is None
    assert r_false.time_trace is None
    tt = r_true.time_trace
    assert tt is not None
    for key in ("dt", "t", "reflected", "transmitted", "incident_left", "incident_right"):
        assert key in tt
    n = tt["t"].size
    assert tt["reflected"].shape == (n,) and tt["transmitted"].shape == (n,)
    assert tt["dt"] > 0.0


def test_gate5_existing_fdtd_infra_gates_still_pass():
    # explicitly re-run the pre-edit FDTD coverage gates; they must be unaffected by the probe
    from test_audit_2026_07_17_infra import (test_fdtd_1d_dielectric_slab_vs_airy,
                                             test_fdtd_1d_drude_slab_absorbs)
    test_fdtd_1d_dielectric_slab_vs_airy()
    test_fdtd_1d_drude_slab_absorbs()


# ---- Gate 6: FDTD ETALON (matrix-pencil Q vs Fabry-Perot closed form) ----------------------

def test_gate6_fdtd_etalon_ringdown_matches_fabry_perot():
    n_slab, L = 3.5, 1.0e-6
    er = fdtd_etalon_ringdown(n_slab, L, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                              resolution=30)
    assert er.modes, "no ringdown modes extracted"
    assert er.q > 0.0 and np.isfinite(er.f0_Hz)

    # Identify the etalon mode order m from the extracted frequency: FP resonances sit at
    # omega_m = m pi c / (n L)  =>  m = omega_0 n L / (pi c).
    w0 = 2 * np.pi * er.f0_Hz
    m_order = w0 * n_slab * L / (np.pi * C_LIGHT)
    m = int(round(m_order))
    assert m >= 1

    # Symmetric-slab Fabry-Perot Q closed form (roadmap 1.1): Q = -m pi / (2 ln|r12|),
    # r12 = (n1 - n2)/(n1 + n2) the slab/vacuum amplitude reflection.
    r12 = (n_slab - 1.0) / (n_slab + 1.0)
    q_closed = -m * np.pi / (2.0 * np.log(abs(r12)))
    assert abs(er.q - q_closed) / q_closed < 0.10                  # within 10%


# ---- module hygiene ------------------------------------------------------------------------

def test_mode_fields_and_all():
    import dynameta.optics.ringdown as rd
    for name in ("Mode", "matrix_pencil", "ringdown_q", "EtalonRingdown", "fdtd_etalon_ringdown"):
        assert name in rd.__all__
    m = Mode(omega_rad_s=10.0, gamma_rad_s=2.0, q=5.0, amplitude=1 + 0j, snr_est=np.inf)
    assert abs(m.f_hz - 10.0 / (2 * np.pi)) < 1e-12
