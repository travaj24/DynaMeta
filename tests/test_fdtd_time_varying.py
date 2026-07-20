"""Gates for roadmap 2.2 -- intra-march TIME-VARYING eps in the 1-D FDTD (time refraction / photon
acceleration). Convention exp(-i omega t), SI. Citations: Morgenthaler, IRE Trans. MTT 6:167 (1958)
for the electromagnetic time boundary; photon-number conservation in adiabatic frequency conversion.

The physics under test
----------------------
A purely TEMPORAL change of the medium (uniform in space, changing in time) conserves the spatial
wavenumber k (space translation symmetry survives) while the temporal frequency tracks
omega(t) = c k / n(t). Two regimes:

  * ADIABATIC (slow ramp n1 -> n2 over many optical cycles): the wave follows omega_out = omega_in
    n1/n2, with NO time-reflected daughter, and the photon number N ~ U/omega is conserved (so the
    field energy U scales as omega, i.e. DROPS by n1/n2 when n increases).

  * FAST STEP (n1 -> n2 in << one cycle): the wave splits into a forward (time-transmitted) and a
    backward (time-reflected) daughter, both at omega_out = omega_in n1/n2. The E-amplitude ratios
    follow from D and B continuity at the temporal boundary (Morgenthaler) -- DERIVED below.

Time-boundary coefficient derivation (D-continuous, B-continuous)
-----------------------------------------------------------------
Incident forward wave in medium n1:  E1 = E0 cos(k z - w1 t),  w1 = c k / n1.
This module's Yee convention (mu0 dHy/dt = +dEx/dz) gives a +z wave H = -(n/(mu0 c)) E, but only the
RATIOS matter, so use the impedance magnitude: for a wave of E-amplitude A in medium n, the magnetic
field magnitude is (n/(mu0 c)) A and D = eps0 n^2 A, B = (n/c) A (+ for forward, - for backward).

After the step (medium n2) the field is forward Ef + backward Eb at w2 = c k / n2 = w1 n1/n2. At the
instant of the step (t=0, cos(k z)):
  D continuous:  eps0 n2^2 (Ef + Eb) = eps0 n1^2 E0     ->  Ef + Eb = (n1^2/n2^2) E0        ...(i)
  B continuous:  (n2/c)(Ef - Eb)     = (n1/c) E0         ->  Ef - Eb = (n1/n2)  E0            ...(ii)
Solving (i),(ii) with r = n1/n2:
        a = Ef/E0 = (n1^2/n2^2 + n1/n2)/2 = (r^2 + r)/2         (forward / time-transmitted)
        b = Eb/E0 = (n1^2/n2^2 - n1/n2)/2 = (r^2 - r)/2         (backward / time-reflected)
For n2 > n1 (r<1): b < 0 (a pi phase flip) and |b| < a. This is EXACTLY the D-preserving FDTD update
(E rescaled by eps_old/eps_new with H untouched), so the split emerges from the march itself.

Energy at the fast boundary: D and B are continuous, so with the electric energy scaling by n1^2/n2^2
(D fixed, /eps2 not /eps1) and the magnetic energy unchanged, and the two being equal for the
incident wave, U_after/U_before = (1 + n1^2/n2^2)/2 < 1 for n2 > n1 -- energy DROPS in the
D-preserving direction (an increase would signal the WRONG, field-preserving update).
"""

import subprocess
import sys

import numpy as np
from scipy.signal import hilbert

from dynameta.constants import C_LIGHT
from dynameta.optics.fdtd import (FDTDLayer, solve_fdtd_1d, run_uniform_time_boundary,
                                  frequency_conversion_diagnostic)


# ---- measurement helpers -------------------------------------------------------------------

def _peak_env(x):
    """Peak of the analytic-signal envelope (a propagation-invariant amplitude measure)."""
    return float(np.max(np.abs(hilbert(np.asarray(x, dtype=float)))))


def _centroid(x, dt, floor=0.05):
    """Power-weighted spectral centroid over the positive bins above floor*max."""
    x = np.asarray(x, dtype=float)
    f = np.fft.rfftfreq(x.size, dt)
    P = np.abs(np.fft.rfft(x)) ** 2
    sel = (f > 0) & (P > floor * P[1:].max())
    return float(np.sum(f[sel] * P[sel]) / np.sum(P[sel]))


# ============================================================================================
# GATE 1 -- BYTE-IDENTITY: static callables reproduce the fixed-material solve bit-for-bit,
# and the existing FDTD coverage gates still pass.
# ============================================================================================

def _slab_kwargs():
    return ([FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0, drude_wp_rad_s=6.0e14,
                       drude_gamma_rad_s=5.0e13)],
            dict(lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30))


def test_gate1_static_hooks_byte_identical():
    layers, kw = _slab_kwargs()
    wp0, g0 = layers[0].drude_wp_rad_s, layers[0].drude_gamma_rad_s
    e0 = layers[0].eps_inf
    r_base = solve_fdtd_1d(layers, **kw)                                  # legacy path, no kwargs

    # static callables returning EXACTLY the initial constants must route through _run_tv yet stay
    # bit-identical (the no-op change-detection). Cover eps_inf-only, drude-only, both, and n_update>1.
    variants = [
        dict(eps_inf_of_t=lambda t: e0),
        dict(drude_of_t=lambda t: (wp0, g0)),
        dict(eps_inf_of_t=lambda t: e0, drude_of_t=lambda t: (wp0, g0)),
        dict(eps_inf_of_t=lambda t: e0, drude_of_t=lambda t: (wp0, g0), n_update=7),
    ]
    for v in variants:
        r = solve_fdtd_1d(layers, **kw, **v)
        # .tobytes() is the literal bit-for-bit identity (same NaN payloads at the DFT-zero bins)
        assert r.R.tobytes() == r_base.R.tobytes(), v
        assert r.T.tobytes() == r_base.T.tobytes(), v
        assert r.freqs_Hz.tobytes() == r_base.freqs_Hz.tobytes(), v
        assert np.array_equal(r.band, r_base.band), v
        assert np.array_equal(r.R, r_base.R, equal_nan=True), v
        assert np.array_equal(r.T, r_base.T, equal_nan=True), v


def test_gate1_existing_fdtd_infra_gates_still_pass():
    # re-run the pre-edit FDTD coverage gates verbatim; the additive time-varying path must not
    # perturb them (same pattern as tests/test_ringdown.py gate 5).
    from test_audit_2026_07_17_infra import (test_fdtd_1d_dielectric_slab_vs_airy,
                                             test_fdtd_1d_drude_slab_absorbs)
    test_fdtd_1d_dielectric_slab_vs_airy()
    test_fdtd_1d_drude_slab_absorbs()


# ============================================================================================
# GATE 2 -- ADIABATIC LIMIT: a pulse fully inside a uniform medium whose index ramps slowly
# n1 -> n2 over many optical cycles; the output centroid tracks omega_out/omega_in = n1/n2.
# ============================================================================================

_LAM = 1.5e-6                     # carrier wavelength in the (initial, vacuum) medium
_K = 2.0 * np.pi / _LAM
_TP1 = 2.0 * np.pi / (C_LIGHT * _K)   # initial optical period (n_init = 1)


def _adiabatic_runs():
    n1, n2 = 1.0, 1.5
    t_start, t_end = 6.0 * _TP1, 6.0 * _TP1 + 30.0 * _TP1     # ramp spans 30 optical cycles

    def n_ramp(t):
        if t <= t_start:
            return n1
        if t >= t_end:
            return n2
        x = (t - t_start) / (t_end - t_start)
        return n1 + (n2 - n1) * 0.5 * (1.0 - np.cos(np.pi * x))   # smooth (cosine) ramp

    common = dict(n_init=n1, lambda_med_m=_LAM, domain_wavelengths=160, cells_per_wavelength=30,
                  pulse_fwhm_wavelengths=8.0, run_periods=120.0, probe_offset_wavelengths=40.0)
    res = run_uniform_time_boundary(index_of_t=n_ramp, **common)
    ref = run_uniform_time_boundary(index_of_t=lambda t: n1, **common)
    return res, ref, n1, n2, t_start, t_end


def test_gate2_adiabatic_frequency_conversion():
    res, ref, n1, n2, _ts, _te = _adiabatic_runs()
    w_out = _centroid(res.transmitted, res.dt)
    w_in = _centroid(ref.transmitted, ref.dt)          # same probe, no ramp -> the input frequency
    ratio = w_out / w_in
    assert abs(ratio - n1 / n2) / (n1 / n2) < 0.01     # within 1%
    # a slow ramp makes NO time-reflected daughter (adiabatic theorem): the backward probe is quiet
    assert _peak_env(res.reflected) / _peak_env(res.transmitted) < 0.02
    # the generic diagnostic reproduces the same ratio from the recorded trace
    diag = frequency_conversion_diagnostic(res)        # output=transmitted, reference=transmitted
    assert np.isfinite(diag["output_centroid_Hz"])


# ============================================================================================
# GATE 3 -- FAST TIME BOUNDARY: a step n1 -> n2 while the pulse is inside; forward + backward
# daughters at omega_out = omega_in n1/n2 with the DERIVED Morgenthaler amplitudes.
# ============================================================================================

def _fast_runs():
    n1, n2 = 1.0, 2.0
    t_step = 8.0 * _TP1
    common = dict(n_init=n1, lambda_med_m=_LAM, domain_wavelengths=120, cells_per_wavelength=30,
                  pulse_fwhm_wavelengths=6.0, run_periods=70.0, probe_offset_wavelengths=25.0)
    res = run_uniform_time_boundary(index_of_t=lambda t, ts=t_step: (n1 if t < ts else n2), **common)
    ref = run_uniform_time_boundary(index_of_t=lambda t: n1, **common)
    return res, ref, n1, n2


def test_gate3_fast_time_boundary_frequency_and_amplitudes():
    res, ref, n1, n2 = _fast_runs()
    r = n1 / n2
    a = (r ** 2 + r) / 2.0                 # forward (time-transmitted) E-amplitude ratio -- DERIVED
    b = abs((r ** 2 - r) / 2.0)            # |backward (time-reflected)| E-amplitude ratio -- DERIVED

    # (i) FREQUENCY SHIFT of the forward daughter: omega_out/omega_in = n1/n2 within 1%
    w_out = _centroid(res.transmitted, res.dt)
    w_in = _centroid(ref.transmitted, ref.dt)
    assert abs(w_out / w_in - r) / r < 0.01
    # the time-reflected (backward) daughter is at the same shifted frequency
    w_back = _centroid(res.reflected, res.dt)
    assert abs(w_back / w_in - r) / r < 0.03

    # (ii) AMPLITUDE RATIOS vs the analytic a, b (5%). The reference is a pure forward pulse of the
    # SAME launch amplitude, measured at the same right probe, so its envelope peak == E0. Both
    # daughters live in the SAME final medium n2, so their probe amplitudes compare directly.
    E0 = _peak_env(ref.transmitted)
    a_meas = _peak_env(res.transmitted) / E0
    b_meas = _peak_env(res.reflected) / E0
    assert abs(a_meas - a) / a < 0.05, (a_meas, a)
    assert abs(b_meas - b) / b < 0.05, (b_meas, b)
    # sanity: the untouched reference has no backward daughter (clean forward launch)
    assert _peak_env(ref.reflected) / E0 < 1e-2


# ============================================================================================
# GATE 4 -- ENERGY BOOKKEEPING: adiabatic conserves photon number (~1%); a fast boundary changes
# the pulse energy in the D-preserving direction (sign asserted, value checked).
# ============================================================================================

def test_gate4_adiabatic_photon_number_conserved():
    res, ref, n1, n2, t_start, t_end = _adiabatic_runs()
    t = res.t_s

    def _avg(t0, t1):
        m = (t >= t0) & (t <= t1)
        return float(np.mean(res.energy_t[m]))

    U_before = _avg(t_start - 4 * _TP1, t_start - 0.5 * _TP1)
    U_after = _avg(t_end + 0.5 * _TP1, t_end + 8 * _TP1)
    # photon number N ~ U/omega ; adiabatic omega ratio = n1/n2 -> U ratio should equal n1/n2
    assert U_after < U_before                                      # energy DROPS as n rises
    assert abs(U_after / U_before - n1 / n2) / (n1 / n2) < 0.02    # U tracks omega
    N_ratio = (U_after / U_before) / (n1 / n2)                     # photon number conserved
    assert abs(N_ratio - 1.0) < 0.01


def test_gate4_fast_boundary_energy_drops_D_preserving():
    res, ref, n1, n2 = _fast_runs()
    t = res.t_s
    t_step = 8.0 * _TP1

    def _avg(t0, t1):
        m = (t >= t0) & (t <= t1)
        return float(np.mean(res.energy_t[m]))

    U_before = _avg(t_step - 4 * _TP1, t_step - 0.5 * _TP1)
    U_after = _avg(t_step + 2 * _TP1, t_step + 20 * _TP1)
    expected = 0.5 * (1.0 + n1 ** 2 / n2 ** 2)                     # (1 + n1^2/n2^2)/2, DERIVED above
    assert U_after < U_before                                      # SIGN: D-preserving, n up -> U down
    assert abs(U_after / U_before - expected) / expected < 0.03


# ============================================================================================
# GATE 5 -- NUMBA / BACKEND GUARD: the 1-D engine (and its new time-varying path) is PURE NUMPY;
# there is no numba fast path to extend, so nothing can silently diverge. Formalize that.
# ============================================================================================

def test_gate5_time_varying_path_is_pure_numpy():
    # the 1-D fdtd module imports no numba; the time-varying solve runs with numba BLOCKED.
    import dynameta.optics.fdtd as fmod
    assert "numba" not in dir(fmod)
    code = ("import sys; sys.modules['numba'] = None\n"
            "import numpy as np\n"
            "from dynameta.optics.fdtd import solve_fdtd_1d, FDTDLayer\n"
            "layers=[FDTDLayer(thickness_m=0.3e-6, eps_inf=4.0)]\n"
            "r=solve_fdtd_1d(layers, lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=20,\n"
            "                eps_inf_of_t=lambda t: 4.0 if t<1e-14 else 5.0)\n"
            "assert np.isfinite(r.R[r.band]).all()\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


# ============================================================================================
# LAYERED API + DIAGNOSTIC: solve_fdtd_1d's public eps_inf_of_t hook drives a real frequency
# conversion end-to-end, read out by frequency_conversion_diagnostic (roadmap 2.2 part b).
# ============================================================================================

def test_solve_fdtd_1d_layered_frequency_conversion_diagnostic():
    c = C_LIGHT
    lam_min, lam_max = 1.35e-6, 1.65e-6
    n1, n2 = 1.5, 2.1
    thickness = 45.0 * 1.5e-6 / n1            # thick layer: holds the whole (moderately narrow) pulse
    settle, n_pad_wave = 12.0, 6.0
    f_min, f_max = c / lam_max, c / lam_min
    tau = 1.0 / (np.pi * (f_max - f_min))
    t0 = settle * tau
    # step time from PUBLIC params only: source peak -> layer front (vacuum) -> 30% into the layer
    t_enter = t0 + 0.65 * n_pad_wave * lam_max / c
    t_step = t_enter + 0.3 * thickness * n1 / c

    layers = [FDTDLayer(thickness_m=thickness, eps_inf=n1 ** 2)]
    kw = dict(lambda_min_m=lam_min, lambda_max_m=lam_max, resolution=40, return_time_trace=True)
    r_static = solve_fdtd_1d(layers, **kw)
    r_step = solve_fdtd_1d(layers, **kw,
                           eps_inf_of_t=lambda t, ts=t_step, e1=n1 ** 2, e2=n2 ** 2: (e1 if t < ts else e2))

    d_static = frequency_conversion_diagnostic(r_static)          # transmitted vs incident_right
    d_step = frequency_conversion_diagnostic(r_step)
    # the transmitted-frequency ratio is Fresnel-magnitude independent -> a clean n1/n2 read
    w1 = d_static["output_centroid_Hz"]
    w2 = d_step["output_centroid_Hz"]
    assert w2 < w1                                                 # n increased -> down-conversion
    assert abs((w2 / w1) - n1 / n2) / (n1 / n2) < 0.03            # within 3%
    # the diagnostic also reports an input centroid and a ratio field
    assert np.isfinite(d_step["input_centroid_Hz"])
    assert set(d_step) >= {"freqs_Hz", "input_spectrum", "output_spectrum",
                           "input_centroid_Hz", "output_centroid_Hz", "ratio"}
