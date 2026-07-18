"""GNLSE completion-term gates (delayed Raman + self-steepening; dossier Topic 5).

Sign convention note: the Raman convolution and the shock d/dT are TIME-domain operators, so
they carry no numpy-vs-physical spectral sign flip (unlike beta3, which is written directly in
numpy-w variables). The gates below pin the physics anyway: SSFS must be RED (physical
frequency down = numpy-w centroid up, since the exp(-i omega t) repo convention puts a physical
detuning Delta at numpy w = -Delta), and the shock must steepen the TRAILING edge.

Conservation contract (Blow & Wood, IEEE JQE 25:2665 (1989)): pure Kerr conserves energy and
photon number exactly; Raman-only (no shock prefactor) conserves ENERGY exactly (real phase)
while red-shifting; Raman + shock conserves PHOTON NUMBER while energy decreases (phonons).
Gordon rate (Opt. Lett. 11:662 (1986)): d nu/dz = -(4/(15 pi)) |beta2| T_R / T0^4 with T_R the
FIRST MOMENT f_R int t h_R dt of the response model itself (~1.46 fs for Blow-Wood -- the
often-quoted 3 fs belongs to the measured-gain-slope convention, not this h_R)."""

import numpy as np

from dynameta.optics.fiber_amp.pulse import (propagate_gnlse, raman_response_freq, sech_pulse)

T0 = 57e-15                       # ~100 fs FWHM fundamental soliton
B2 = -2e-26                       # -20 ps^2/km
GAM = 2e-3
P0 = abs(B2) / (GAM * T0 ** 2)    # N = 1
LAM0 = 1.06e-6


def _soliton(n=4096, t0=T0, p0=None):
    t = np.linspace(-40 * t0, 40 * t0, n)
    return sech_pulse(t, t0_s=t0, peak_power_W=(P0 if p0 is None else p0), lambda0_m=LAM0)


def _centroid_shift_hz_per_m(p, r, L):
    w, S0 = p.spectrum()
    _, S1 = r.output.spectrum()
    c0 = np.sum(w * S0) / np.sum(S0)
    c1 = np.sum(w * S1) / np.sum(S1)
    # physical detuning = -w_numpy: numpy centroid UP = physical RED
    return -(c1 - c0) / (2.0 * np.pi) / L


def _photon_number(pulse_or_out, t):
    w = 2.0 * np.pi * np.fft.fftfreq(t.size, t[1] - t[0])
    om0 = 2.0 * np.pi * 3.0e8 / LAM0
    S = np.abs(np.fft.fft(pulse_or_out.field)) ** 2
    mask = np.abs(w) < 0.3 * om0                 # pulse band (avoid 1/(om0-w) junk far out)
    return float(np.sum(S[mask] / (om0 - w[mask])))


def _gordon_rate_hz_m(model, t0):
    f_R, _ = raman_response_freq(8, 1e-15, model)
    dt = 1e-16
    tt = np.arange(200000) * dt
    tau1, tau2 = 12.2e-15, 32e-15
    ha = (tau1 ** 2 + tau2 ** 2) / (tau1 * tau2 ** 2) * np.exp(-tt / tau2) * np.sin(tt / tau1)
    if model == "blow_wood":
        h = ha
    else:
        tau_b = 96e-15
        hb = (2.0 * tau_b - tt) / tau_b ** 2 * np.exp(-tt / tau_b)
        h = 0.79 * ha + 0.21 * hb
    T_R = f_R * np.trapezoid(tt * h, tt) / np.trapezoid(h, tt)
    return 4.0 / (15.0 * np.pi) * abs(B2) * T_R / t0 ** 4


def test_default_path_matches_pure_kerr_reference():
    # byte-level contract: raman=None, self_steepening=False is the ORIGINAL split-step
    p = _soliton(n=2048)
    r = propagate_gnlse(p, 1.0, beta2_s2_m=B2, gamma_W_m=GAM, n_steps=400)
    # independent inline re-implementation of the legacy pure-Kerr symmetric split-step
    A = p.field.astype(np.complex128).copy()
    w = p.omega_rad_s()
    D = 1j * (B2 / 2.0 * w ** 2)
    h = 1.0 / 400
    half = np.exp(D * (h / 2.0))
    for _ in range(400):
        A = np.fft.ifft(half * np.fft.fft(A))
        A = A * np.exp(1j * GAM * np.abs(A) ** 2 * h)
        A = np.fft.ifft(half * np.fft.fft(A))
    assert np.allclose(r.output.field, A, rtol=0.0, atol=1e-12 * np.max(np.abs(A)))


def test_raman_only_red_shift_at_gordon_rate_and_energy_conserved():
    p = _soliton()
    L = 2.0
    r = propagate_gnlse(p, L, beta2_s2_m=B2, gamma_W_m=GAM, raman="blow_wood", n_steps=1200)
    shift = _centroid_shift_hz_per_m(p, r, L)
    assert shift < 0.0                                        # RED
    gord = _gordon_rate_hz_m("blow_wood", T0)
    assert 0.8 * gord < -shift < 1.7 * gord, (-shift, gord)   # rate ~ Gordon (model T_R)
    assert abs(r.output.energy_J / p.energy_J - 1.0) < 1e-9   # real phase: energy exact


def test_raman_rate_scales_as_inverse_T0_fourth():
    L = 2.0
    s = []
    for t0 in (T0, 1.3 * T0):
        p0 = abs(B2) / (GAM * t0 ** 2)
        p = _soliton(t0=t0, p0=p0)
        r = propagate_gnlse(p, L, beta2_s2_m=B2, gamma_W_m=GAM, raman="blow_wood", n_steps=1200)
        s.append(-_centroid_shift_hz_per_m(p, r, L))
    ratio = s[0] / s[1]
    assert abs(ratio / 1.3 ** 4 - 1.0) < 0.35, ratio          # Gordon 1/T0^4 scaling


def test_lin_agrawal_model_shifts_at_least_as_fast():
    p = _soliton()
    L = 2.0
    r_bw = propagate_gnlse(p, L, beta2_s2_m=B2, gamma_W_m=GAM, raman="blow_wood", n_steps=1200)
    r_la = propagate_gnlse(p, L, beta2_s2_m=B2, gamma_W_m=GAM, raman="lin_agrawal", n_steps=1200)
    s_bw = -_centroid_shift_hz_per_m(p, r_bw, L)
    s_la = -_centroid_shift_hz_per_m(p, r_la, L)
    assert s_la > 0.95 * s_bw                                 # bigger f_R (0.245) at least keeps pace
    f_R, H = raman_response_freq(4096, 1e-15, "lin_agrawal")
    assert abs(f_R - 0.245) < 1e-12 and abs(H[0] - 1.0) < 1e-9


def test_self_steepening_trailing_edge_and_conservation():
    p = _soliton(p0=5.0 * P0)                                 # strong SPM, no dispersion
    t = p.t_s
    r = propagate_gnlse(p, 0.5, gamma_W_m=GAM, self_steepening=True, n_steps=1500)
    pw0, pw1 = p.power_W, r.output.power_W
    tc0 = np.sum(t * pw0) / np.sum(pw0)
    tc1 = np.sum(t * pw1) / np.sum(pw1)
    assert tc1 > tc0 + 1e-15                                  # peak drifts LATER (trailing shock)
    dp = np.gradient(pw1, t)
    i_pk = int(np.argmax(pw1))
    assert np.max(np.abs(dp[i_pk:])) > 1.3 * np.max(np.abs(dp[:i_pk]))   # trailing steepening
    assert abs(r.output.energy_J / p.energy_J - 1.0) < 1e-6   # energy conserved (RK4)
    assert abs(_photon_number(r.output, t) / _photon_number(p, t) - 1.0) < 1e-3


def test_raman_plus_shock_conserves_photons_not_energy():
    p = _soliton()
    t = p.t_s
    r = propagate_gnlse(p, 2.0, beta2_s2_m=B2, gamma_W_m=GAM, raman="blow_wood",
                        self_steepening=True, n_steps=1200)
    e_ratio = r.output.energy_J / p.energy_J
    ph_ratio = _photon_number(r.output, t) / _photon_number(p, t)
    assert e_ratio < 0.9995                                   # energy DOWN (phonons)
    assert abs(ph_ratio - 1.0) < 2e-3                         # photon number conserved
