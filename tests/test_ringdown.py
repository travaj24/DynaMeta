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
# audit V-17 leftover (wave-7, 2026-11-02): the ringdown unreliable-result warns are now
# UnreliableResultWarning (a RuntimeWarning subclass); these pytest.warns pin the category.
from dynameta.warning_types import UnreliableResultWarning
from dynameta.optics.ringdown import (Mode, matrix_pencil, ringdown_q,
                                      fdtd_etalon_ringdown, _ringdown_window,
                                      _nls_refine_real, _real_reconstruction)

# audit T-13 (`filterwarnings = error`): fitting a REAL FDTD etalon trace routinely turns up a
# marginal pole at |z| = 1 + O(1e-4) (dropped, with the "GROWING pole" advisory) and, at high Q,
# a short-window note -- data-dependent, e.g. n_slab 3.5 and 10.0 warn here while 5.0 and 7.0 do
# not, so pinning them per test would be brittle. Both diagnostics have their OWN dedicated gates
# in this file (test_gate10b_growing_poles_are_dropped_and_do_not_overflow,
# test_gate9_default_fits_the_whole_window_not_a_1200_sample_cut), which use `pytest.warns` and are
# unaffected by these marks -- as is test_gate10c's local `simplefilter("error")`.
pytestmark = [pytest.mark.filterwarnings("ignore:matrix_pencil"),
              pytest.mark.filterwarnings("ignore:fdtd_etalon_ringdown")]


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


# Gate 5 ("existing FDTD infra still passes") is GONE (audit T-16, 2026-11-02): it re-invoked
# test_audit_2026_07_17_infra's two FDTD test functions by cross-module import, so the suite ran the
# same two FDTD marches again for zero coverage (pytest already collects them in their own module)
# and the bare `from test_audit_... import` breaks under --import-mode=importlib. The probe's
# non-perturbation is gated directly above (default-off trace, bit-identical R/T).


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
    (Those are pencil values: the refinement was inert until 2026-11-02. With it live: 13.227533
    (-1.006%), 27.202982 (+0.312%), 55.205207 (+1.105%), 110.408333 (+0.748%). With the WHOLE
    data-driven window fit by default (wave-6 H6b, 2026-11-02; was a 1200-sample cut): 13.227438
    (-1.007%), 27.202936 (+0.312%), 55.196766 (+1.090%), 110.349763 (+0.695%) -- 3.5 unchanged to
    7e-6, the other three closer to Fabry-Perot.)
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
        with pytest.warns(UnreliableResultWarning) as rec:
            er = fdtd_etalon_ringdown(n_slab, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                      resolution=30, start_frac=0.02)
        msgs = [str(w.message) for w in rec]
        # it really is wrong (so the flag is not decoration)
        m = int(round(2.0 * np.pi * er.f0_Hz * n_slab * 1.0e-6 / (np.pi * C_LIGHT)))
        assert abs(er.q / _fp_closed_form_q(n_slab, m) - 1.0) > min_err, (n_slab, er.q)
        # ... and it is flagged, twice: the window rule and the fit itself
        assert any("PEAKS at sample" in s for s in msgs), msgs
        assert any("does NOT describe the window" in s for s in msgs), msgs
        # V-17 (wave-7): BOTH flags carry the unreliable-result category, not a bare RuntimeWarning
        assert all(issubclass(w.category, UnreliableResultWarning) for w in rec
                   if str(w.message).startswith("fdtd_etalon_ringdown")), [w.category for w in rec]
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


def _memoize_fdtd_march(monkeypatch):
    """Run each distinct etalon march ONCE per test: fdtd_etalon_ringdown imports solve_fdtd_1d
    from dynameta.optics.fdtd at call time and only READS the result, so a memoizing wrapper lets
    one test compare several fit settings on the very same trace."""
    import dynameta.optics.fdtd as fdtd_mod
    real, cache = fdtd_mod.solve_fdtd_1d, {}

    def memo(layers, **kw):
        key = (tuple((ly.eps_inf, ly.thickness_m) for ly in layers), tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = real(layers, **kw)
        return cache[key]
    monkeypatch.setattr(fdtd_mod, "solve_fdtd_1d", memo)


def test_gate9_default_fits_the_whole_window_not_a_1200_sample_cut():
    """wave-6 H6b (2026-11-02). ``max_fit_samples`` defaulted to 1200, a hard TRUNCATION of the
    decimated data-driven window: at n_slab = 10 that is 1200 of 4937 samples, 1.67 amplitude
    e-foldings, and the span guard fired on a shipped gate-7 config. The default (None) now fits the
    whole window: 4937 samples, 6.88 nepers, no span warning, Q 110.4083 -> 110.3498 (+0.748% ->
    +0.695% vs Fabry-Perot 109.5882). FAILS on the old default (1200 samples, 1.67 nepers, warns).
    The explicit legacy cap still truncates -- and its warning now says the cap did it. (This
    replaces the n_slab = 20 span test that ran a ~55 s march in the PR tier: n_slab = 10 is 21 s,
    and the n_slab = 20 case is the @slow gate below.)"""
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        er = fdtd_etalon_ringdown(10.0, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                  resolution=30)
    assert not [str(w.message) for w in rec if "e-foldings" in str(w.message)]
    assert er.signal_used.size > 4000, er.signal_used.size   # 4937 measured; the old cut was 1200
    nep = 0.5 * er.modes[0].gamma_rad_s * float(er.t_used[-1])
    assert nep > 5.0, nep                                     # 6.88 measured (1.67 at the old cut)
    m = int(round(2.0 * np.pi * er.f0_Hz * 10.0 * 1.0e-6 / (np.pi * C_LIGHT)))
    assert abs(er.q / _fp_closed_form_q(10.0, m) - 1.0) < 0.03, (er.q, m)

    with pytest.warns(UnreliableResultWarning, match="TRUNCATED to max_fit_samples = 1200"):
        e1 = fdtd_etalon_ringdown(10.0, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                  resolution=30, max_fit_samples=1200)
    assert e1.signal_used.size == 1200
    assert 0.5 * e1.modes[0].gamma_rad_s * float(e1.t_used[-1]) < 3.0      # 1.67 measured
    # the flag is conservative: at 1.67 nepers the extraction still holds (+0.748%)
    m1 = int(round(2.0 * np.pi * e1.f0_Hz * 10.0 * 1.0e-6 / (np.pi * C_LIGHT)))
    assert abs(e1.q / _fp_closed_form_q(10.0, m1) - 1.0) < 0.03

    # the shipped low-index configs are comfortably inside the envelope at the default
    for n_slab, lo in ((3.5, 15.0), (7.0, 10.0)):            # 21.3 / 14.0 nepers measured
        e2 = fdtd_etalon_ringdown(n_slab, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                                  resolution=30)
        assert 0.5 * e2.modes[0].gamma_rad_s * float(e2.t_used[-1]) > lo, n_slab

    with pytest.raises(ValueError, match="max_fit_samples must be None"):
        fdtd_etalon_ringdown(3.5, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                             resolution=30, max_fit_samples=8)


def test_gate9_over_budget_window_coarsens_the_stride_before_cutting(monkeypatch):
    """The default's pencil budget (_FIT_SAMPLES_BUDGET = 6000) never binds on a shipped config
    (largest window 4937 samples), so it is exercised here by LOWERING it on the n_slab = 7 trace
    (4838 samples at 20 samples/period, 14.0 nepers, Q 55.1968, +1.090% vs Fabry-Perot 54.6018):
      budget 2000: the stride is coarsened (~20 -> ~8 samples/period) instead of cutting the
                   window: the whole 14.0 nepers kept, Q inside the 3% gate (Q lives in the decay
                   SPAN, not in the samples per period);
      budget  300: even 8 samples/period overflows it, so the window is cut, and the span warning
                   names the budget -- not max_fit_samples -- as the reason."""
    import dynameta.optics.ringdown as rd_mod
    _memoize_fdtd_march(monkeypatch)
    kw = dict(lambda_min_m=1.2e-6, lambda_max_m=1.7e-6, resolution=30)
    ref = fdtd_etalon_ringdown(7.0, 1.0e-6, **kw)
    assert ref.signal_used.size > 4000                     # premise: over the 2000 budget below
    nep_ref = 0.5 * ref.modes[0].gamma_rad_s * float(ref.t_used[-1])

    monkeypatch.setattr(rd_mod, "_FIT_SAMPLES_BUDGET", 2000)
    er = fdtd_etalon_ringdown(7.0, 1.0e-6, **kw)
    assert er.signal_used.size <= 2000
    assert er.dt_used > 2.0 * ref.dt_used                  # coarser stride, not a cut
    assert er.window == ref.window
    nep = 0.5 * er.modes[0].gamma_rad_s * float(er.t_used[-1])
    assert nep == pytest.approx(nep_ref, rel=0.05), (nep, nep_ref)
    m = int(round(2.0 * np.pi * er.f0_Hz * 7.0 * 1.0e-6 / (np.pi * C_LIGHT)))
    assert abs(er.q / _fp_closed_form_q(7.0, m) - 1.0) < 0.03, (er.q, m)

    monkeypatch.setattr(rd_mod, "_FIT_SAMPLES_BUDGET", 300)
    with pytest.warns(UnreliableResultWarning, match="pencil budget"):
        e3 = fdtd_etalon_ringdown(7.0, 1.0e-6, **kw)
    assert e3.signal_used.size == 300


@pytest.mark.slow
def test_gate9_n20_default_matches_fabry_perot(monkeypatch):
    """n_slab = 20 (Q ~ 440) at the DEFAULT settings (wave-6 H6b, 2026-11-02). Until then the
    default fit was cut at max_fit_samples = 1200 of the 3176 decimated samples the data-driven
    window holds -- 0.42 TRUE e-foldings, over which a comb of ~29 near-equal-amplitude etalon modes (the top
    eight within 25% of each other) cannot be separated -- and Q came out 252.8 against Fabry-Perot
    455.2 (-44.5%, m 29; RMS tell 3.5e-2, UNRELIABLE FIT). The wave-2 hand-off asked: record too
    short for Q ~ 450, or more pencil modes than max_refine? NEITHER. Measured on this one trace:
        max_fit_samples 1200 (old default)   Q 252.83 vs 455.15 (m 29)  -44.5%   UNRELIABLE FIT
        max_fit_samples 2400                 Q 442.18 vs 439.46 (m 28)  +0.6%
        None = whole window, 3176 (default)  Q 442.24 vs 439.46 (m 28)  +0.6%
        settle 24 (2x record), cap 1200      Q 5408   (window later, 0.04 nepers) -- WORSE
        settle 24, None (whole window)       Q 441.29 vs 439.46 (m 28)  +0.4%  (SAME 3176 samples)
        max_refine 20 (pencil: 34 modes)     Q 442.35 -- max_refine is not the lever
    The cut was the problem, not the record length. The pencil cost of 3176 vs 1200 samples is
    invisible next to the ~55-70 s FDTD march. The span guard still warns (1.10 < 3 nepers -- the
    window ends at the march's numeric floor, so a longer record does not lengthen it): a
    CONSERVATIVE flag this case over-calls. FAILS on the old default
    (Q -44.5%). @slow: one march, memoized for the legacy-cap contrast; nightly (a) runs it."""
    _memoize_fdtd_march(monkeypatch)
    kw = dict(lambda_min_m=1.2e-6, lambda_max_m=1.7e-6, resolution=30)
    with pytest.warns(UnreliableResultWarning, match="WHOLE data-driven window"):
        er = fdtd_etalon_ringdown(20.0, 1.0e-6, **kw)
    assert "UNRELIABLE" not in er.refine_note, er.refine_note
    assert er.signal_used.size > 3000                       # the whole window (3176), not 1200
    m = int(round(2.0 * np.pi * er.f0_Hz * 20.0 * 1.0e-6 / (np.pi * C_LIGHT)))
    assert abs(er.q / _fp_closed_form_q(20.0, m) - 1.0) < 0.03, (er.q, m)
    # the contrast on the SAME trace: the legacy cap IS the -44.5% bias, and it is flagged
    with pytest.warns(RuntimeWarning) as rec:
        e1 = fdtd_etalon_ringdown(20.0, 1.0e-6, max_fit_samples=1200, **kw)
    m1 = int(round(2.0 * np.pi * e1.f0_Hz * 20.0 * 1.0e-6 / (np.pi * C_LIGHT)))
    assert e1.q < 0.6 * _fp_closed_form_q(20.0, m1)
    assert "UNRELIABLE" in e1.refine_note
    assert any("TRUNCATED" in str(w.message) for w in rec)


def test_gate7_cross_platform_q_anchor_unchanged():
    """The 2026-07-20 CI fix (data-driven window + NLS refinement) pinned the n_slab = 3.5
    dominant-mode Q at 13.351015 on both the Windows dev box and the CI linux wheels. The Q-1 /
    Q-2 remediation must NOT move it: the window is byte-identical there (the late-time floor IS
    a genuine plateau, floor/env_start = 9.95e-13, so the plateau test passes and the decade rule
    is not binding) and the refinement's design matrix is unchanged (all 6 pencil modes are
    oscillatory, so `rest` was empty even pre-fix).

    RE-PINNED 2026-11-02 (wave-6): 13.351015 was the RAW PENCIL Q -- the refinement was inert
    (nfev = 1 in SI units, see test_nls_refinement_moves_optical_poles_toward_truth), so the
    cross-platform agreement of the anchor came from the pencil alone. With the refinement live
    the dominant Q is the NLS optimum 13.227533 (-0.92% from the pencil; -1.01% vs the
    Fabry-Perot 13.361960, inside gate 7's 3%), refit RMS 2.44e-3 -> 1.93e-3 of the window's
    peak-to-peak, and independent of the seed to <= 8e-10 (test_nls_refinement_makes_the_fdtd_
    q_seed_independent) -- which is what makes it a platform-stable anchor.

    RE-PINNED 2026-11-02 (wave-6 H6b): the default fit is now the WHOLE data-driven window (1845
    decimated samples, 21.3 nepers) instead of its first 1200 (13.9): 13.227533 -> 13.227438
    (-7.2e-6 relative; -1.007% vs Fabry-Perot). Same window, same refinement: the extra 645
    samples sit deep in a decay the first 1200 had already pinned, so the anchor barely moves.

    RE-PINNED 2026-11-02 (audit S2-14 siblings, docs/audit/2026-11-02-fdtd-registration.md):
    solve_fdtd_1d now registers materials at the Yee E nodes. Here pad/dz = 892.5 and d/dz = 87.5
    exactly, so the slab keeps its 88 cells under both rules and the change is a RIGID one-cell
    translation (the old cell-centre rule put both effective faces dz = 11.4 nm early; the node rule
    puts them on the true faces). Q is translation-invariant physically; the FITTED Q moves through
    the data-driven window's timing against the probe: 13.227438 -> 13.207623 (-0.15%; -1.007% ->
    -1.155% vs Fabry-Perot 13.361960, inside gate 7's 3%)."""
    er = fdtd_etalon_ringdown(3.5, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                              resolution=30)
    assert er.q == pytest.approx(13.207623, rel=1e-4)
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
    with pytest.warns(UnreliableResultWarning, match="GROWING pole"):
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


# ---- audit Q-22: ringdown_q and fdtd_etalon_ringdown are ONE definition of "the" ringdown Q --

def test_q22_ringdown_q_matches_the_etalon_extraction_on_the_same_trace():
    """audit Q-22 (2026-11-02). ringdown_q returned the RAW pencil dominant mode while
    fdtd_etalon_ringdown refined (VARPRO) and re-ranked by the re-solved amplitudes, so the same
    samples gave two different "ringdown Q"s. Measured pre-fix on two equal-amplitude modes (Q 40
    and Q 25, 600 samples) plus 5% Gaussian noise (rng seed 106): ringdown_q 24.85 vs the
    etalon path 40.18. Both now go through _pencil_modes."""
    from dynameta.optics.ringdown import _nls_refine_real
    dt = 1e-16
    t = np.arange(600) * dt
    w1, w2 = 2 * np.pi * 2.0e14, 2 * np.pi * 2.3e14
    y = (np.exp(-w1 / 40.0 * t / 2) * np.cos(w1 * t + 0.3)
         + np.exp(-w2 / 25.0 * t / 2) * np.cos(w2 * t - 1.0)
         + 0.05 * np.random.default_rng(106).standard_normal(t.size))
    kw = dict(svd_tol=1e-6, amp_floor=5e-2, real_signal=True)
    seed = matrix_pencil(y, dt, **kw)
    refined, note = _nls_refine_real(y, dt, seed)
    assert note == ""
    # premise: on this trace the two orderings really do disagree (else the gate proves nothing)
    assert seed[0].q != refined[0].q
    f0, q = ringdown_q(y, dt, **kw)
    assert (f0, q) == (refined[0].f_hz, refined[0].q)
    assert abs(q / 40.0 - 1.0) < 0.02                       # and it is the Q-40 mode
    # refine=False is the raw pencil, explicitly
    assert ringdown_q(y, dt, refine=False, **kw) == (seed[0].f_hz, seed[0].q)
    # a complex trace has no real-trace refinement: ringdown_q falls back to the pencil
    yc = np.exp((-w1 / 80.0 - 1j * w1) * t)
    assert ringdown_q(yc, dt) == (matrix_pencil(yc, dt)[0].f_hz, matrix_pencil(yc, dt)[0].q)


def test_q22_ringdown_q_reproduces_fdtd_etalon_ringdown_bitwise():
    """The FDTD side of Q-22: hand ringdown_q the very window fdtd_etalon_ringdown inverted, with
    the pencil kwargs it used, and get its (f0, Q) back bit-for-bit."""
    er = fdtd_etalon_ringdown(3.5, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                              resolution=30)
    assert ringdown_q(er.signal_used, er.dt_used, pencil_frac=0.4, svd_tol=1e-6,
                      amp_floor=5e-2, real_signal=True) == (er.f0_Hz, er.q)


# ---- wave-6 (2026-11-02): the VARPRO refinement must actually MOVE poles ----------------------

def _one_mode_optical_trace(rng_seed, sigma=0.05, n_samp=600, dt=1e-16, f0=2.0e14, q=40.0):
    """One damped optical-frequency cosine (omega ~ 1.3e15, gamma ~ 3e13 rad/s -- the SI scale
    of every FDTD trace) plus white noise of std `sigma` (the peak is 1)."""
    t = np.arange(n_samp) * dt
    w = 2.0 * np.pi * f0
    g = w / q
    y = (np.exp(-0.5 * g * t) * np.cos(w * t + 0.3)
         + sigma * np.random.default_rng(rng_seed).standard_normal(n_samp))
    return y, dt, complex(-0.5 * g, w), g


def _pole_err(m, s_true, g):
    """|s_fit - s_true| / gamma for the mode nearest the true line (s = -gamma/2 + i omega)."""
    return abs(complex(-0.5 * m.gamma_rad_s, m.omega_rad_s) - s_true) / g


def test_nls_refinement_moves_optical_poles_toward_truth():
    """wave-1 Q-22 finding, fixed 2026-11-02: `_nls_refine_real` was INERT on every optical trace.
    In SI units (omega ~ 1e15, gamma/2 ~ 1e13 rad/s) the raw gradient was under gtol = 1e-14, so
    least_squares stopped at nfev = 1 (status 1, dx = 0) and the "refined" pole was the pencil
    pole bit-for-bit. The optimizer now works in sample time on the peak-normalized trace.

    Truth-anchored, over 24 noise realizations of a known pole (Q 40, 5% noise): the refined
    (maximum-likelihood) pole must be closer to the truth than the pencil pole ON AVERAGE.
    Measured at the fix: mean |ds|/gamma 1.173e-2 pencil -> 1.102e-2 refined (ratio 0.939; the
    TLS pencil at L = 0.4 N is already near-efficient, so the margin is honest, not large).
    Pre-fix the ratio is EXACTLY 1.0 (no pole moves), which fails the < 0.97 bar."""
    err_seed, err_ref = [], []
    for sd in range(24):
        y, dt, s_true, g = _one_mode_optical_trace(sd)
        seed = matrix_pencil(y, dt, svd_tol=1e-6, amp_floor=5e-2, real_signal=True)
        refined, note = _nls_refine_real(y, dt, seed)
        assert note == "", note
        err_seed.append(_pole_err(seed[0], s_true, g))
        err_ref.append(_pole_err(refined[0], s_true, g))
    ratio = float(np.mean(err_ref) / np.mean(err_seed))
    assert ratio < 0.97, (ratio, np.mean(err_seed), np.mean(err_ref))


def test_nls_refinement_forgets_its_seed():
    """The documented purpose of the refinement (platform stability): the NLS optimum is a
    property of the DATA, so a perturbed pencil seed -- the stand-in for a different LAPACK build
    -- must land on the same refined pole, and far closer to the truth than that seed. Measured
    at the fix: seed off by +0.2% in omega and +20% in gamma (|ds|/gamma 0.11) refines to within
    ~1e-9 relative of the pencil-seeded optimum. Pre-fix the perturbed seed came back unchanged."""
    import dataclasses
    y, dt, s_true, g = _one_mode_optical_trace(7)
    seed = matrix_pencil(y, dt, svd_tol=1e-6, amp_floor=5e-2, real_signal=True)
    ref0, note0 = _nls_refine_real(y, dt, seed)
    assert note0 == ""
    bad = [dataclasses.replace(m, omega_rad_s=m.omega_rad_s * 1.002,
                               gamma_rad_s=m.gamma_rad_s * 1.2,
                               q=m.omega_rad_s * 1.002 / (m.gamma_rad_s * 1.2)) for m in seed]
    ref1, note1 = _nls_refine_real(y, dt, bad)
    assert note1 == "", note1
    assert ref1[0].omega_rad_s == pytest.approx(ref0[0].omega_rad_s, rel=1e-8)
    assert ref1[0].gamma_rad_s == pytest.approx(ref0[0].gamma_rad_s, rel=1e-6)
    assert _pole_err(ref1[0], s_true, g) < 0.2 * _pole_err(bad[0], s_true, g)


def test_nls_refinement_makes_the_fdtd_q_seed_independent():
    """On a real FDTD etalon trace (n_slab = 3.5, all 6 pencil modes refined) the refined
    dominant Q is independent of the seed: a 1e-4 / 1e-2 relative perturbation of every seed
    gamma (the pencil's BLAS-dependent digits, exaggerated) moved the refined Q by <= 8e-10
    (measured), against 1e-4 / 8e-3 for the seed itself. Pre-fix the refined Q moved one-for-one
    with the seed (nfev = 1)."""
    import dataclasses
    er = fdtd_etalon_ringdown(3.5, 1.0e-6, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                              resolution=30)
    y, dt = er.signal_used, er.dt_used
    seed = matrix_pencil(y, dt, pencil_frac=0.4, svd_tol=1e-6, amp_floor=5e-2, real_signal=True)
    assert 1 <= len(seed) <= 6                       # premise: every mode is a refined mode
    for eps in (1e-4, 1e-2):
        bad = [dataclasses.replace(m, gamma_rad_s=m.gamma_rad_s * (1.0 + eps * (-1) ** i),
                                   q=m.omega_rad_s / (m.gamma_rad_s * (1.0 + eps * (-1) ** i)))
               for i, m in enumerate(seed)]
        out, note = _nls_refine_real(y, dt, bad)
        assert note == "", note
        assert out[0].q == pytest.approx(er.q, rel=1e-7), (eps, out[0].q, er.q)



def test_q21_degenerate_modes_are_split_by_index_not_by_value():
    """audit Q-21 (fixed in wave 1, d602866, without a gate): _nls_refine_real split refined vs
    fixed modes with `m not in osc`, i.e. dataclass VALUE equality, so a mode whose exact duplicate
    was among the refined ones vanished from the design matrix AND from the output. Two
    value-identical seed modes, max_refine = 1: both must come back (one refined, one fixed)."""
    dt, n = 1e-3, np.arange(800)
    t = n * dt
    w = 2 * np.pi * 40.0
    y = (np.exp(-2.0 * t) * np.cos(w * t + 0.3)
         + 0.01 * np.random.default_rng(21).standard_normal(n.size))   # so a refit can improve
    m = matrix_pencil(y, dt, svd_tol=1e-6, amp_floor=5e-2, real_signal=True)[0]
    half = Mode(omega_rad_s=m.omega_rad_s, gamma_rad_s=m.gamma_rad_s, q=m.q,
                amplitude=0.5 * m.amplitude, snr_est=m.snr_est)
    out, note = _nls_refine_real(y, dt, [half, half], max_refine=1)
    assert note == "", note
    assert len(out) == 2
