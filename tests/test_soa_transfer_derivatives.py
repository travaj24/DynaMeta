"""Audit A-23 (2026-11-02): the sampled transfer-curve derivatives report their own accuracy, and a
callable transfer gets finite-difference derivatives.

Oracle: T(P) = G0 P / (1 + P/Psat), whose derivatives are exact:
    T' = G0 Ps^2/(Ps+P)^2,  T'' = -2 G0 Ps^2/(Ps+P)^3,  T''' = 6 G0 Ps^2/(Ps+P)^4.
"""
import warnings

import numpy as np
import pytest

from dynameta.optics.soa import metrics as M

G0, PS = 100.0, 1e-3


def _T(P):
    return G0 * P / (1.0 + P / PS)


def _exact(P):
    return (G0 * PS ** 2 / (PS + P) ** 2, -2.0 * G0 * PS ** 2 / (PS + P) ** 3,
            6.0 * G0 * PS ** 2 / (PS + P) ** 4)


def _log_grid(n):
    P = np.logspace(np.log10(PS) - 3, np.log10(PS) + 2, n)     # 5 decades around Psat
    return P, _T(P)


def _old_transfer_derivatives(P_in_grid, P_out_grid, P0):
    """The pre-A-23 estimator, verbatim."""
    P_in = np.asarray(P_in_grid, dtype=np.float64)
    P_out = np.asarray(P_out_grid, dtype=np.float64)
    order = np.argsort(P_in)
    P_in, P_out = P_in[order], P_out[order]
    i = int(np.clip(np.searchsorted(P_in, P0), 3, P_in.size - 4))
    sl = slice(i - 3, i + 4)
    c = np.polyfit(P_in[sl] - P0, P_out[sl], 3)
    return float(c[2]), float(2.0 * c[1]), float(6.0 * c[0])


def test_sampled_derivatives_are_bit_identical_to_the_old_estimator():
    P, Y = _log_grid(41)
    rng = np.random.default_rng(3)
    for P0 in (2e-5, 1e-4, 1e-3, 3e-3, 5e-2):
        Yn = Y * (1.0 + 1e-6 * rng.standard_normal(Y.size))
        new = M.transfer_derivatives(P, Yn, P0, warn_rtol=None)
        assert new == _old_transfer_derivatives(P, Yn, P0)


def test_sampled_path_flags_an_unresolved_grid_and_is_quiet_on_a_resolved_one():
    # 41 log points: T''' is 35% low at Psat -- the indicator must say so
    P, Y = _log_grid(41)
    with pytest.warns(M.TransferFitWarning, match="not resolved"):
        Tp, Tpp, Tppp, err = M.transfer_derivatives(P, Y, PS, return_error=True)
    true3 = _exact(PS)[2]
    assert abs(Tppp - true3) / true3 > 0.3          # premise: the estimate really is bad here
    assert err > 0.1
    # a dense LINEAR grid around P0 resolves it: accurate, small err, no warning
    Pd = np.linspace(0.5 * PS, 1.5 * PS, 201)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Tp, Tpp, Tppp, err = M.transfer_derivatives(Pd, _T(Pd), PS, return_error=True)
    assert abs(Tppp - true3) / true3 < 1e-3 and err < 1e-3
    # fewer than 9 samples: no estimate (nan), no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(M.transfer_derivatives(P[18:26], Y[18:26], PS, return_error=True)[3])


def test_finite_difference_derivatives_match_the_closed_form():
    for P0 in (1e-4, 1e-3, 3e-3):
        d = M.transfer_derivatives_fd(_T, P0, return_error=True)
        ex = _exact(P0)
        rel = [abs(d[j] - ex[j]) / abs(ex[j]) for j in range(3)]
        assert max(rel) < 3e-3, (P0, rel)
        # on a clean curve the estimate is conservative (>= the true error) yet below the
        # default warning threshold, so an accurate callable does not false-alarm
        assert max(rel) <= d[3] < 0.05, (P0, d[3])
    with pytest.raises(ValueError):
        M.transfer_derivatives_fd(_T, PS, rel_step=0.5)
    with pytest.raises(ValueError):
        M.transfer_derivatives_fd(_T, 0.0)


def _sfdr_exact(P0, mod=0.3):
    c1, c2, c3 = M.harmonic_amplitudes(*_exact(P0), P0 * mod)
    return 10.0 * np.log10((0.5 * c1 ** 2) / max(0.5 * c2 ** 2, 0.5 * c3 ** 2))


def test_sfdr_from_a_callable_is_accurate_where_the_sampled_grid_is_not():
    P, Y = _log_grid(41)
    P0 = 3.0 * PS
    exact = _sfdr_exact(P0)
    fd = M.sfdr_dB(None, None, P0, 0.0, transfer=_T)
    assert abs(fd - exact) < 0.02
    with pytest.warns(M.TransferFitWarning):
        sampled = M.sfdr_dB(P, Y, P0, 0.0)
    assert abs(sampled - exact) > 0.2                 # the sampled path really is off here


def test_sndr_vs_drive_warns_once_per_call_and_takes_a_callable():
    P, Y = _log_grid(41)
    P0s = np.array([0.3, 1.0, 2.0, 3.0]) * PS

    def noise(_P_out):
        return 1e-12

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        M.sndr_vs_drive(P, Y, noise, P0s)
    hits = [x for x in w if issubclass(x.category, M.TransferFitWarning)]
    assert len(hits) == 1                              # one summary, naming the worst P0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s_fd, _, _ = M.sndr_vs_drive(None, None, noise, P0s, transfer=_T)
    assert np.all(np.isfinite(s_fd))


def test_a_noisy_callable_is_flagged():
    rng = np.random.default_rng(5)

    def noisy(P):
        return _T(P) * (1.0 + 1e-4 * rng.standard_normal())
    with pytest.warns(M.TransferFitWarning, match="finite-difference"):
        M.sfdr_dB(None, None, PS, 0.0, transfer=noisy)
