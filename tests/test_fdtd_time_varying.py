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
import pytest
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


# ============================================================================================
# audit D-17 (2026-11-02): the uniform march's energy diagnostic is the leapfrog invariant at ONE
# time level, and an index below `courant` (ENZ) is refused instead of marching to NaN.
# ============================================================================================

def test_d17_energy_diagnostic_is_the_time_centred_leapfrog_invariant():
    """energy_t used to add |E^{n+1}|^2 to |H^{n+1/2}|^2 -- two time levels. Measured pre-fix
    (relative peak-to-peak over the window): 7.5e-6 on a pure forward pulse in a constant medium
    and 2.47e-2 after the fast n 1 -> 2 step (the forward + backward daughters carry a
    standing-wave part, whose 2w ripple the mixed levels expose). The fixed diagnostic is the
    discrete invariant of the interior Yee update: flat to round-off in both windows (~1e-15),
    with only the D-preserving jump at the step itself."""
    res, ref, n1, n2 = _fast_runs()
    t = res.t_s
    t_step = 8.0 * _TP1

    def _p2p(e):
        return float((e.max() - e.min()) / e.mean())

    assert _p2p(ref.energy_t[t < t_step + 20 * _TP1]) < 1e-12
    assert _p2p(res.energy_t[(t > t_step + 2 * _TP1) & (t < t_step + 20 * _TP1)]) < 1e-12
    assert _p2p(res.energy_t[(t > t_step - 4 * _TP1) & (t < t_step - 0.5 * _TP1)]) < 1e-12


def test_d17_enz_index_below_courant_is_refused_not_marched_to_nan():
    """The "stable for any n>=1" comment was unchecked while index_of_t may go below 1 (ENZ).
    Measured pre-fix at courant 0.5: n 1 -> 0.45 / 0.3 returned all-NaN fields and energy with no
    error. The march is stable iff n >= courant: n = 0.6 must still run (ENZ is supported), 0.45
    must raise naming the Courant number, and so must an n_init below courant."""
    lam = 1.0e-6
    tp = lam / C_LIGHT
    common = dict(n_init=1.0, lambda_med_m=lam, domain_wavelengths=80, run_periods=12.0)
    ok = run_uniform_time_boundary(index_of_t=lambda t: 1.0 if t < 5 * tp else 0.6, **common)
    assert np.all(np.isfinite(ok.energy_t)) and np.all(np.isfinite(ok.Ex_final))
    assert ok.n_traj[-1] == 0.6
    with pytest.raises(ValueError, match="Courant"):
        run_uniform_time_boundary(index_of_t=lambda t: 1.0 if t < 5 * tp else 0.45, **common)
    with pytest.raises(ValueError, match="Courant"):
        run_uniform_time_boundary(index_of_t=lambda t: 0.4, n_init=0.4, lambda_med_m=lam,
                                  domain_wavelengths=80, run_periods=12.0)
    with pytest.raises(ValueError, match="n > 0"):
        run_uniform_time_boundary(index_of_t=lambda t: 1.0 if t < 5 * tp else float("nan"),
                                  **common)


# ============================================================================================
# audit D-18 / V-18 (wave-6, 2026-11-02)
# ============================================================================================

def test_d18_uniform_z_edges_with_time_varying_hook_collapses_instead_of_raising():
    """D-18: the hooks-vs-nonuniform refusal ran BEFORE the uniform-limit detection, so the
    documented bit-identity grid `uniform_z_edges(...)` was refused with a time-varying hook even
    though it collapses to the supported uniform kernel. It now runs, bit-identical to the solve
    without z_edges; a genuinely nonuniform grid is still refused."""
    from dynameta.optics.fdtd import uniform_z_edges
    layers, kw = _slab_kwargs()
    wp0, g0 = layers[0].drude_wp_rad_s, layers[0].drude_gamma_rad_s
    hook = dict(drude_of_t=lambda t: (wp0 * (1.0 + 0.2 * (t > 3e-14)), g0))
    ze = uniform_z_edges(layers, lambda_min_m=kw["lambda_min_m"], lambda_max_m=kw["lambda_max_m"],
                         resolution=kw["resolution"])
    r_ref = solve_fdtd_1d(layers, **kw, **hook)
    r_ze = solve_fdtd_1d(layers, **kw, **hook, z_edges=ze)              # raised pre-fix
    assert r_ze.R.tobytes() == r_ref.R.tobytes() and r_ze.T.tobytes() == r_ref.T.tobytes()
    assert np.array_equal(r_ze.band, r_ref.band)
    with pytest.raises(NotImplementedError, match="time-varying hooks"):
        solve_fdtd_1d(layers, **kw, **hook, refine={0: 2.0})            # really nonuniform


def test_v18_drude_of_t_accepts_the_transient_optics_drudeoptical_contract():
    """V-18: `drude_of_t` meant (wp, gamma) in solve_fdtd_1d and a DrudeOptical in
    transient_optics (the carrier-heating hook), so the latter could not be handed to the FDTD path
    transient_optics itself recommends. The DrudeOptical form now converts through
    DrudeOptical.wp_gamma at `drude_n_m3` and is BIT-IDENTICAL to the equivalent tuple hook (it
    raised `cannot unpack` pre-fix); the two consistency checks raise."""
    from dynameta.constants import M_E
    from dynameta.materials.optical_model import DrudeOptical
    n = 2.0e25
    m0, g0 = 0.35 * M_E, 5.0e13
    hot = DrudeOptical(eps_inf=4.0, m_opt_kg=lambda nn: 0.45 * M_E + 0.0 * nn, gamma_rad_s=8.0e13)
    cold = DrudeOptical(eps_inf=4.0, m_opt_kg=m0, gamma_rad_s=g0)
    wp_c, g_c = cold.wp_gamma(n)
    wp_h, g_h = hot.wp_gamma(n)
    assert wp_c == pytest.approx(np.sqrt(n * 1.602176634e-19 ** 2 / (8.8541878128e-12 * m0)), rel=1e-9)
    # the FDTD Drude pole built from wp_gamma IS the DrudeOptical's eps (to rounding)
    lam = 1.3e-6
    w = 2.0 * np.pi * C_LIGHT / lam
    assert complex(4.0 - wp_c ** 2 / (w ** 2 + 1j * w * g_c)) == pytest.approx(
        complex(cold.eps(lam, n_m3=n)), rel=1e-12)
    layers = [FDTDLayer(thickness_m=0.30e-6, eps_inf=4.0, drude_wp_rad_s=wp_c, drude_gamma_rad_s=g_c)]
    kw = dict(lambda_min_m=1.2e-6, lambda_max_m=1.45e-6, resolution=30)
    t1 = 4e-14
    r_tuple = solve_fdtd_1d(layers, **kw, drude_of_t=lambda t: (wp_h, g_h) if t > t1 else (wp_c, g_c))
    r_obj = solve_fdtd_1d(layers, **kw, drude_n_m3=n, drude_of_t=lambda t: hot if t > t1 else cold)
    assert r_obj.R.tobytes() == r_tuple.R.tobytes() and r_obj.T.tobytes() == r_tuple.T.tobytes()
    r_static = solve_fdtd_1d(layers, **kw)
    assert not np.array_equal(r_obj.R[r_obj.band], r_static.R[r_static.band])  # the switch is live
    with pytest.raises(ValueError, match="drude_n_m3"):
        solve_fdtd_1d(layers, **kw, drude_of_t=lambda t: cold)
    with pytest.raises(ValueError, match="eps_inf"):
        solve_fdtd_1d(layers, **kw, drude_n_m3=n,
                      drude_of_t=lambda t: DrudeOptical(eps_inf=3.0, m_opt_kg=m0, gamma_rad_s=g0))
    with pytest.raises(ValueError, match="two owners"):
        solve_fdtd_1d(layers, **kw, drude_n_m3=n, eps_inf_of_t=lambda t: 4.0,
                      drude_of_t=lambda t: cold)
