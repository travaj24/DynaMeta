"""Gates for optics/ringdown.py (roadmap 1.2 -- ringdown harmonic inversion by the matrix-pencil
method, Hua & Sarkar IEEE TAP 38:814 (1990)) and the additive opt-in FDTD time-trace probe.

Convention under test (see ringdown.py docstring): a real field trace decays as exp(-gamma t/2)
in amplitude (energy ~ exp(-gamma t)); q = omega_0 / gamma is the ENERGY Q and equals the
pole-finder convention Re(omega_t)/(2|Im omega_t|) for the same resonance.

Gate 7 (finding Q-1) covers the data-driven fit window: it must not collapse on a HIGH-Q trace
that is still ringing at the end of the record (n_slab = 3.5/5/7/10 against the Fabry-Perot
closed form, with the 2026-07-20 cross-platform Q anchor pinned), must fail LOUDLY on an
all-noise trace, and must survive a 100x-mis-specified band centre. Gate 8 (finding Q-2) covers
the VARPRO refinement: every pencil mode enters the design matrix, and a refit that does not beat
its own seed is rejected.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks, hilbert

from dynameta.constants import C_LIGHT
from dynameta.optics.ringdown import (Mode, matrix_pencil, ringdown_q,
                                      fdtd_etalon_ringdown, _ringdown_window,
                                      _nls_refine_real, _real_reconstruction)


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


# ---- Gate 7 (finding Q-1): the data-driven window must not collapse on a HIGH-Q trace -----

def _fp_closed_form_q(n_slab, m):
    r12 = (n_slab - 1.0) / (n_slab + 1.0)
    return -m * np.pi / (2.0 * np.log(abs(r12)))


@pytest.mark.parametrize("n_slab,rel_tol", [(3.5, 0.03), (5.0, 0.03), (7.0, 0.03), (10.0, 0.03)])
def test_gate7_high_q_etalon_window_does_not_collapse(n_slab, rel_tol):
    """REGRESSION (finding Q-1). The 2026-07-20 data-driven window ended the fit where the
    envelope reached `floor_margin` x `median(env[last 10%])`. On a trace that is still ringing
    at the end of the record that "floor" IS the live signal, so the end threshold landed ABOVE
    the envelope at the window start, the window collapsed to the hardcoded 8-sample minimum and
    `matrix_pencil` raised "need at least 4 samples after t_start". Measured break-in point:
    n_slab = 7 (Q ~ 50); n_slab = 10 too. A longer record did NOT rescue it (settle = 12/30/60/120
    all raised) because the median tracks a decay that lengthens with the record.

    The fixed window drives the END off the ENVELOPE DECAY (a fixed number of decades below the
    window start) and only trusts the late-time floor when it is a genuine PLATEAU, falling back
    to the full remaining record otherwise. All four indices must now invert against the
    symmetric-slab Fabry-Perot closed form Q = -m pi / (2 ln|r12|).

    Measured at the fix (n_slab: Q, closed form, error):
      3.5:  13.351015 vs 13.361960  (-0.082%)      <- the CI configuration, unchanged
      5.0:  27.291776 vs 27.118423  (+0.639%)
      7.0:  54.970929 vs 54.601815  (+0.676%)      <- pre-fix: ValueError
     10.0: 110.302227 vs 109.588241 (+0.652%)      <- pre-fix: ValueError
    """
    L = 1.0e-6
    er = fdtd_etalon_ringdown(n_slab, L, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6, resolution=30)
    assert er.modes and np.isfinite(er.q) and np.isfinite(er.f0_Hz)
    w0 = 2.0 * np.pi * er.f0_Hz
    m = int(round(w0 * n_slab * L / (np.pi * C_LIGHT)))
    q_cf = _fp_closed_form_q(n_slab, m)
    assert abs(er.q / q_cf - 1.0) < rel_tol
    # the window must be a real window, not the 8-sample terminal state
    i0, i1 = er.window
    assert i1 - i0 > 1000
    assert er.signal_used.size >= 200

    # finding Q-2's safety gate: the returned modes never reconstruct WORSE than the pencil seed.
    seed = matrix_pencil(er.signal_used, er.dt_used, pencil_frac=0.4, svd_tol=1e-6,
                         amp_floor=5e-2, real_signal=True)
    y = er.signal_used
    rms_seed = np.sqrt(np.mean((_real_reconstruction(y.size, er.dt_used, seed) - y) ** 2))
    rms_out = np.sqrt(np.mean((_real_reconstruction(y.size, er.dt_used, er.modes) - y) ** 2))
    assert rms_out <= rms_seed * (1.0 + 1e-12), (rms_out, rms_seed, er.refine_note)
    assert isinstance(er.refine_note, str)


def test_gate9_contaminated_window_is_flagged_not_silent():
    """FIX-VERIFY W1 kill 2.  ``_nls_refine_real``'s safety gate is RELATIVE -- it only asks whether
    the refit beats its own pencil seed -- so on a window that still contains the driven pulse BOTH
    fits are garbage, the comparison passes, and the public path returned ``refine_note = ''`` with
    ZERO warnings.  Measured on the documented escape hatch ``start_frac = 0.02``: Q = 3.592 vs the
    Fabry-Perot 10.69 at n_slab = 3.5 (-66%) and Q = 4.098 vs 49.14 at n_slab = 7 (-92%), with the
    reconstruction RMS at 4.2e-2 / 3.0e-2 of the window's peak-to-peak and the summed |amplitude|
    at 1575x / 4430x the data peak.  Two ABSOLUTE tells now fire, plus a window-vs-envelope-peak
    check on the fixed-fraction path itself."""
    from dynameta.optics.ringdown import _fit_quality, _FIT_RESID_TOL, _FIT_AMP_SANITY

    for n_slab, min_err in ((3.5, 0.50), (7.0, 0.50)):
        with pytest.warns(RuntimeWarning) as rec:
            er = fdtd_etalon_ringdown(n_slab, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                      resolution=30, start_frac=0.02)
        msgs = [str(w.message) for w in rec]
        # it really is wrong (so the flag is not decoration)
        m = int(round(2.0 * np.pi * er.f0_Hz * n_slab * 1.0e-6 / (np.pi * C_LIGHT)))
        assert abs(er.q / _fp_closed_form_q(n_slab, m) - 1.0) > min_err, (n_slab, er.q)
        # ... and it is flagged, twice: the window rule and the fit itself
        assert any("PEAKS at sample" in s for s in msgs), msgs
        assert any("does NOT describe the window" in s for s in msgs), msgs
        assert "UNRELIABLE FIT" in er.refine_note, er.refine_note
        q = _fit_quality(er.signal_used, er.dt_used, er.modes)
        assert q["rms_rel"] > _FIT_RESID_TOL or q["amp_rel"] > _FIT_AMP_SANITY, q

    # a start_frac PAST the driven transient is accepted silently and is accurate
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        ok = fdtd_etalon_ringdown(3.5, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                  resolution=30, start_frac=0.2)
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)], \
        [str(w.message) for w in rec]
    assert "UNRELIABLE" not in ok.refine_note
    m = int(round(2.0 * np.pi * ok.f0_Hz * 3.5 * 1.0e-6 / (np.pi * C_LIGHT)))
    assert abs(ok.q / _fp_closed_form_q(3.5, m) - 1.0) < 0.03

    # start_frac is validated (it was an unchecked multiply into an index before)
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="0 < start_frac < 1"):
            fdtd_etalon_ringdown(3.5, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                 resolution=30, start_frac=bad)


def test_gate9_fit_quality_tells_are_silent_on_every_shipped_config():
    """The companion to the gate above: the tells must not cry wolf.  On the DEFAULT (data-driven)
    window across the whole gate-7 index set the reconstruction RMS is 9.1e-4..2.4e-3 of the
    peak-to-peak and the summed |amplitude| is 1.2..2.1x the data peak -- both an order of
    magnitude inside their limits."""
    from dynameta.optics.ringdown import _fit_quality, _FIT_RESID_TOL, _FIT_AMP_SANITY

    for n_slab in (3.5, 5.0, 7.0, 10.0):
        er = fdtd_etalon_ringdown(n_slab, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                  resolution=30)
        q = _fit_quality(er.signal_used, er.dt_used, er.modes)
        assert q["rms_rel"] < 0.5 * _FIT_RESID_TOL, (n_slab, q)
        assert q["amp_rel"] < 0.25 * _FIT_AMP_SANITY, (n_slab, q)
        assert "UNRELIABLE" not in er.refine_note, (n_slab, er.refine_note)


def test_gate9_high_q_window_span_scope_limit_warns():
    """FIX-VERIFY W1 item 9.  ``max_fit_samples = 1200`` at 20 samples/period is 60 carrier
    periods, i.e. only ``60 pi / Q`` amplitude e-foldings: 13.7 at n_slab = 3.5 but 1.1 at
    n_slab = 20.  That is where the extraction breaks -- Q = 169.9 against the Fabry-Perot 455.2
    (-62.7%) -- and it used to do so silently.  The span guard (and, here, the RMS tell too) now
    says so."""
    with pytest.warns(RuntimeWarning) as rec:
        er = fdtd_etalon_ringdown(20.0, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                  resolution=30)
    msgs = [str(w.message) for w in rec]
    m = int(round(2.0 * np.pi * er.f0_Hz * 20.0 * 1.0e-6 / (np.pi * C_LIGHT)))
    assert er.q < 0.6 * _fp_closed_form_q(20.0, m)              # it IS biased low
    assert any("e-foldings" in s for s in msgs), msgs
    nepers = 0.5 * er.modes[0].gamma_rad_s * float(er.t_used[-1])
    assert nepers < 3.0

    # the shipped low-index configs are comfortably inside the envelope
    for n_slab, lo in ((3.5, 10.0), (7.0, 3.0)):
        e2 = fdtd_etalon_ringdown(n_slab, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                  resolution=30)
        assert 0.5 * e2.modes[0].gamma_rad_s * float(e2.t_used[-1]) > lo, n_slab


def test_gate7_cross_platform_q_anchor_unchanged():
    """The 2026-07-20 CI fix (data-driven window + NLS refinement) pinned the n_slab = 3.5
    dominant-mode Q at 13.351015 on both the Windows dev box and the CI linux wheels. The Q-1 /
    Q-2 remediation must NOT move it: the window is byte-identical there (the late-time floor IS
    a genuine plateau, floor/env_start = 9.95e-13, so the plateau test passes and the decade rule
    is not binding) and the refinement's design matrix is unchanged (all 6 pencil modes are
    oscillatory, so `rest` was empty even pre-fix)."""
    er = fdtd_etalon_ringdown(3.5, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                              resolution=30)
    assert er.q == pytest.approx(13.351015, rel=1e-4)
    assert er.window == (9840, 31980)


def _synthetic_ringdown(n_samp=200000, dt=1.9e-17, f0=2.13e14, q=300.0, pulse_at=3000,
                        pulse_w=800.0, amp_ring=0.1):
    """A driven pulse followed by a high-Q ringdown that is STILL RINGING at the end of the
    record -- the exact configuration that collapsed the pre-fix window."""
    n = np.arange(n_samp)
    t = n * dt
    w = 2.0 * np.pi * f0
    gam = w / q
    y = np.exp(-((n - pulse_at) / pulse_w) ** 2) * np.cos(w * t)
    y = y + amp_ring * np.exp(-0.5 * gam * t) * np.cos(w * t + 0.3)
    return y, dt, f0, w, gam


def test_gate7_synthetic_still_ringing_trace_uses_full_remaining_record():
    """Synthetic trace whose ringdown never reaches a numeric floor inside the record (the
    late-time "floor" is the LIVE SIGNAL, ~8% of the ringdown start): the fixed window must fall
    back to the FULL REMAINING RECORD and the pencil must recover Q, instead of collapsing to the
    pre-fix 8-sample stub (finding Q-1)."""
    y, dt, f0, w, gam = _synthetic_ringdown()
    i0, i1 = _ringdown_window(y, dt, f0)
    assert i1 - i0 > 0.9 * (y.size - i0)          # the whole tail, not a truncated stub
    assert i0 > 3000                              # the driven pulse was skipped
    stride = max(1, int(round((1.0 / f0) / (20 * dt))))
    tail = y[i0:i1][::stride][:1200]
    modes = matrix_pencil(tail - tail.mean(), dt * stride, svd_tol=1e-6, amp_floor=5e-2,
                          real_signal=True)
    assert modes
    m = modes[0]
    assert abs(m.omega_rad_s / w - 1.0) < 1e-3
    assert abs(m.q / (w / gam) - 1.0) < 0.05


def test_gate7_all_noise_trace_fails_honestly():
    """An all-noise trace has no ringdown at all. The pre-fix window silently returned the same
    8-sample stub it returned for every other malformed input; the fix must RAISE and name the
    reason (finding Q-1's anti-silent-failure requirement)."""
    rng = np.random.default_rng(11)
    y = rng.standard_normal(140000)
    with pytest.raises(ValueError, match="no ringdown decay detected"):
        _ringdown_window(y, 1.9e-17, 2.13e14)


@pytest.mark.parametrize("f_scale", [100.0, 0.01])
def test_gate7_mis_specified_fc_block_width_guard(f_scale):
    """A badly mis-specified band centre used to be fatal: f_c 100x too HIGH gave sub-carrier-
    period blocks whose "envelope" dives into every zero crossing (terminal 8-sample window), and
    100x too LOW gave a handful of huge blocks and a silently wrong window. The block width is now
    clamped to [1, 4] carrier periods of the frequency MEASURED from the trace itself, so the
    window stays within a factor ~2 of the correctly-specified one (finding Q-1)."""
    y, dt, f0, w, gam = _synthetic_ringdown()
    i0_ref, i1_ref = _ringdown_window(y, dt, f0)
    i0, i1 = _ringdown_window(y, dt, f0 * f_scale)
    assert i1 - i0 > 0.5 * (i1_ref - i0_ref)
    assert i1 - i0 < 2.0 * (i1_ref - i0_ref) + 4000
    assert i0 > 2000                              # still past the driven pulse


# ---- Gate 8 (finding Q-2): every pencil mode must enter the VARPRO design matrix -----------

def _dc_contaminated_trace():
    """7 damped cosines (more than max_refine = 6) plus a STRONG slow pure-decay drift. Pre-fix,
    `modes[:max_refine]` was sliced BEFORE the frequency test, so the DC mode (the largest
    |amplitude|, hence modes[0]) and the two weakest oscillators were dropped from the design
    matrix entirely; their energy then aliased into the refined oscillators, dragging one to
    omega -> 0 with a spurious amplitude that the post-hoc |amplitude| sort promoted to
    modes[0] -- i.e. straight into the reported (f0, Q).

    Measured on THIS trace with the pre-fix `_nls_refine_real` (re-implemented verbatim):
    reconstruction RMS 3.694e-2 against the pencil seed's 3.224e-12 -- 1.1e10 times WORSE, and
    0.69% of the data peak-to-peak -- with the dominant line's gamma off by 2.5% (2.9244 vs 3.0)
    and the weaker lines corrupted. Nothing in the pre-fix path compared residuals before
    accepting that."""
    dt, N = 1.0e-3, 2000
    t = np.arange(N) * dt
    truth = [(2 * np.pi * 40.0, 3.0, 1.00, 0.3),
             (2 * np.pi * 60.0, 4.0, 0.85, -0.7),
             (2 * np.pi * 95.0, 5.0, 0.70, 1.1),
             (2 * np.pi * 130.0, 6.0, 0.55, 0.2),
             (2 * np.pi * 160.0, 8.0, 0.45, -1.4),
             (2 * np.pi * 200.0, 9.0, 0.35, 0.9),
             (2 * np.pi * 240.0, 11.0, 0.30, 2.0)]
    y = 2.5 * np.exp(-0.6 * t)                                   # DC / slow-drift contamination
    for w, g, A, ph in truth:
        y = y + A * np.exp(-g * t / 2.0) * np.cos(w * t + ph)
    return y, dt, truth


def test_gate8_dc_contaminated_refinement_matches_truth():
    y, dt, truth = _dc_contaminated_trace()
    seed = matrix_pencil(y, dt, svd_tol=1e-8, amp_floor=1e-3, real_signal=True)
    # the scenario must actually exercise the bug: more modes than max_refine, DC dominant
    assert len(seed) > 6
    assert seed[0].omega_rad_s * dt <= 1e-9, "the DC drift must be the largest-|A| pencil mode"

    refined, note = _nls_refine_real(y, dt, seed, max_refine=6)
    rms_seed = np.sqrt(np.mean((_real_reconstruction(y.size, dt, seed) - y) ** 2))
    rms_ref = np.sqrt(np.mean((_real_reconstruction(y.size, dt, refined) - y) ** 2))
    # the cross-cutting safety gate: never worse than the seed
    assert rms_ref <= rms_seed, (rms_ref, rms_seed, note)
    assert rms_ref < 1e-6 * np.ptp(y)                            # and in fact essentially exact

    osc = [m for m in refined if m.omega_rad_s * dt > 1e-9]
    # NO omega -> 0 invention: every oscillatory mode sits on a real line, none near DC
    for m in osc:
        assert m.omega_rad_s > 2 * np.pi * 10.0, "spurious near-DC oscillator invented"
    # every truth line is recovered, including the 0.55-amplitude one the pre-fix code lost
    for w_t, g_t, A_t, _ph in truth:
        got = min(osc, key=lambda m, ww=w_t: abs(m.omega_rad_s - ww))
        assert abs(got.omega_rad_s / w_t - 1.0) < 1e-3, (w_t, got.omega_rad_s)
        assert abs(got.gamma_rad_s / g_t - 1.0) < 1e-2, (w_t, got.gamma_rad_s, g_t)
        assert abs(abs(got.amplitude) / A_t - 1.0) < 1e-2
    # the dominant OSCILLATORY mode is the A = 1.0 line, not an invented DC one
    dom = max(osc, key=lambda m: abs(m.amplitude))
    assert abs(dom.omega_rad_s / truth[0][0] - 1.0) < 1e-3
    # the pure-decay drift is still represented (its own real-exponential column)
    dc = [m for m in refined if m.omega_rad_s * dt <= 1e-9]
    assert dc and abs(abs(dc[0].amplitude) / 2.5 - 1.0) < 1e-2


def test_gate8_refine_returns_seed_with_a_note_when_it_cannot_help():
    """The refinement returns (modes, note); the note is the debug tell that the PENCIL SEED was
    returned instead of a refit (finding Q-2's safety gate)."""
    y, dt, _truth = _dc_contaminated_trace()
    seed = matrix_pencil(y, dt, svd_tol=1e-8, amp_floor=1e-3, real_signal=True)
    out, note = _nls_refine_real(y, dt, seed, max_refine=0)
    assert note and out == seed                                  # nothing to refine -> seed kept
    out2, note2 = _nls_refine_real(y, dt, seed, max_refine=6)
    assert note2 == ""                                           # accepted on this trace


# ---- module hygiene ------------------------------------------------------------------------

def test_mode_fields_and_all():
    import dynameta.optics.ringdown as rd
    for name in ("Mode", "matrix_pencil", "ringdown_q", "EtalonRingdown", "fdtd_etalon_ringdown"):
        assert name in rd.__all__
    m = Mode(omega_rad_s=10.0, gamma_rad_s=2.0, q=5.0, amplitude=1 + 0j, snr_est=np.inf)
    assert abs(m.f_hz - 10.0 / (2 * np.pi)) < 1e-12


# ------------------------------------------------------------------------------------------------
# Gate 10 (findings Q-8 / Q-9): the DISCRETE-POLE DOMAIN -- Nyquist branch and growing poles
# ------------------------------------------------------------------------------------------------
def test_gate10a_nyquist_pole_is_reported_not_nan(recwarn):
    """finding Q-8: `np.linalg.eigvals` returns a REAL array when every eigenvalue is real, and
    `np.log` of a NEGATIVE float64 is nan -- a pole at exactly Nyquist (z < 0) escaped the public
    API as omega_rad_s = 0 with gamma_rad_s = nan and q = nan, plus a RuntimeWarning from numpy.
    z is now cast to complex128 (correct branch ln|z| + i pi) and the self-conjugate Nyquist pole
    is reported ONCE with an undoubled real residue."""
    dt, n = 1e-15, np.arange(400)
    y = ((-1.0) ** n) * np.exp(-0.01 * n)                  # z = -exp(-0.01): |z| < 1, arg = pi
    modes = matrix_pencil(y, dt)
    assert modes, "the Nyquist mode must not be dropped by the conjugate collapse"
    for m in modes:
        assert np.isfinite(m.omega_rad_s) and np.isfinite(m.gamma_rad_s) and np.isfinite(m.q)
    dom = modes[0]
    assert dom.omega_rad_s * dt == pytest.approx(np.pi, abs=1e-9)        # omega = pi/dt exactly
    assert dom.gamma_rad_s * dt == pytest.approx(0.02, abs=1e-9)         # gamma = -2 ln|z| / dt
    assert dom.q == pytest.approx(np.pi / 0.02, rel=1e-9)
    assert abs(dom.amplitude) == pytest.approx(1.0, rel=1e-9)            # NOT doubled
    assert not [w for w in recwarn if "invalid value encountered in log" in str(w.message)]
    # CONTROL: an ordinary near-Nyquist mode is unaffected (0.95 pi, not pi)
    ctl = matrix_pencil(np.cos(0.95 * np.pi * n) * np.exp(-0.01 * n), dt)[0]
    assert ctl.omega_rad_s * dt == pytest.approx(0.95 * np.pi, rel=1e-9)
    assert ctl.gamma_rad_s * dt == pytest.approx(0.02, rel=1e-6)


def test_gate10b_growing_poles_are_dropped_and_do_not_overflow():
    """finding Q-9: nothing rejected |z| > 1, so a growing trace returned gamma < 0 and q < 0
    against the documented Mode contract -- and `z**n` overflowed to inf at large N, poisoning the
    residue lstsq and the |amplitude| sort for EVERY mode. Growing poles are now dropped with a
    RuntimeWarning (opt back in with allow_gain=True, which uses an overflow-free log-space
    Vandermonde)."""
    dt, n = 1e-15, np.arange(400)
    y_grow = np.exp(+0.02 * n) * np.cos(0.3 * n)
    with pytest.warns(RuntimeWarning, match="GROWING pole"):
        modes = matrix_pencil(y_grow, dt)
    assert all(m.gamma_rad_s >= 0.0 and m.q >= 0.0 for m in modes)       # contract restored
    kept = matrix_pencil(y_grow, dt, allow_gain=True)
    assert kept and kept[0].gamma_rad_s < 0.0                            # opt-in returns them
    assert kept[0].omega_rad_s * dt == pytest.approx(0.3, rel=1e-6)
    assert np.isfinite(abs(kept[0].amplitude)) and np.isfinite(kept[0].q)
    # The Vandermonde z**n_idx is built from z ALONE, so a growing pole overflows float64 once
    # |z|**(N-1) passes ~1.8e308 -- for |z| = 1.5 that is N ~ 1750, NOT the N = 1200 the audit
    # note quotes (1.5**1200 = 2.04e211, comfortably finite; the arithmetic in finding Q-9 is
    # wrong on that point even though the mechanism is real). The log-space column-normalized
    # Vandermonde makes the allow_gain path immune to it at any N.
    with np.errstate(over="ignore"):
        assert np.isfinite(1.5 ** 1200) and not np.isfinite(np.float64(1.5) ** 1800)
    nl = np.arange(1400)
    y_long = np.exp(0.1 * nl) * np.cos(0.3 * nl)
    y_long = y_long / np.max(np.abs(y_long))               # keep the DATA in range
    long_grow = matrix_pencil(y_long, dt, allow_gain=True)
    assert long_grow
    assert all(np.isfinite(abs(m.amplitude)) and np.isfinite(m.q) for m in long_grow)
    assert long_grow[0].omega_rad_s * dt == pytest.approx(0.3, rel=1e-6)
    assert long_grow[0].gamma_rad_s * dt == pytest.approx(-0.2, rel=1e-6)


def test_gate10c_decaying_traces_take_the_unchanged_fast_path():
    """The Q-9 machinery must be INERT on every physical ringdown: an ordinary decaying trace
    still goes through the original z**n Vandermonde, so the shipped numbers are untouched."""
    dt, n = 1e-15, np.arange(600)
    y = (np.exp(-0.004 * n) * np.cos(0.7 * n) + 0.4 * np.exp(-0.02 * n) * np.cos(1.9 * n))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no growing-pole warning may fire
        modes = matrix_pencil(y, dt)
    got = sorted((m.omega_rad_s * dt, m.gamma_rad_s * dt) for m in modes)
    assert len(got) == 2
    assert got[0][0] == pytest.approx(0.7, rel=1e-8) and got[0][1] == pytest.approx(0.008, rel=1e-6)
    assert got[1][0] == pytest.approx(1.9, rel=1e-8) and got[1][1] == pytest.approx(0.04, rel=1e-6)
