"""Gates for materials.lattice_optical: the factorized TO/LO lattice permittivity, the Lorentz-sum and
Gaussian-oscillator models, the ONE Lorentz kernel shared with optics.resonance / optics.fdtd_seam,
passivity and Lyddane-Sachs-Teller checks, plasmon-LO-phonon coupling, Kramers-Kronig consistency, FDTD
single-pole compatibility and a Berreman-mode check through the planar optical solve.

Oracles are closed forms written here; each check is shown to be discriminating (a mutant misses it).

Run: python -m pytest tests/test_lattice_optical.py -q
"""
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from dynameta.constants import C_LIGHT, EPS0, M_E, Q_E
from dynameta.materials import check_kk
from dynameta.materials.lattice_optical import (GaussianOscillatorOptical, LorentzOscillatorOptical,
                                                PolarLatticeOptical, coupled_mode_frequencies, lorentz_susceptibility,
                                                rad_s_to_wavenumber, wavenumber_to_rad_s, with_free_carriers)


def _lam(nu_cm):
    return 0.01 / nu_cm


def test_one_lorentz_kernel_is_shared_bit_for_bit():
    """optics.resonance.lorentzian_eps is eps_inf + lorentz_susceptibility, equal BIT FOR BIT to the closed
    form written out here (same operation order), for real and complex omega."""
    from dynameta.optics.resonance import lorentzian_eps
    w = np.array([1e13, 2.3e14, 5e14 + 3e12j, 1.1e15 - 2e13j])
    ei, de, w0, g = 2.25, 1.7, 2.1e14, 3.3e12
    wc = np.asarray(w, dtype=np.complex128)
    old = ei + de * w0 * w0 / (w0 * w0 - wc * wc - 1j * wc * g)
    assert np.array_equal(lorentzian_eps(w, ei, de, w0, g), old)
    assert np.array_equal(ei + lorentz_susceptibility(w, de, w0, g), old)


def test_factorized_single_mode_limits_and_lorentz_identity():
    """One mode: zero at w_LO and pole at w_TO (undamped); eps(0) = eps_inf (w_LO / w_TO)^2 (LST); eps(inf) =
    eps_inf; with g_TO = g_LO it IS the Lorentz sum with d_eps = eps_inf (w_LO^2 - w_TO^2) / w_TO^2 (exact
    identity). Mutant: d_eps without the 1 / w_TO^2 misses."""
    T, L = wavenumber_to_rad_s(795.9), wavenumber_to_rad_s(972.3)
    und = PolarLatticeOptical(eps_inf=6.49, modes=((T, L, 0.0, 0.0),))
    assert abs(und.eps_omega(L)) < 1e-9
    assert abs(und.eps_omega(T * (1 + 1e-9))) > 1e8
    assert und.static_permittivity() == pytest.approx(6.49 * (972.3 / 795.9) ** 2, rel=1e-13, abs=0.0)
    assert und.eps_omega(1e-6 * T).real == pytest.approx(und.static_permittivity(), rel=1e-9, abs=0.0)
    assert und.eps_omega(1e6 * L).real == pytest.approx(6.49, rel=1e-9, abs=0.0)
    g = wavenumber_to_rad_s(5.0)
    fac = PolarLatticeOptical(eps_inf=6.49, modes=((T, L, g, g),))
    de = 6.49 * (L * L - T * T) / (T * T)
    lor = LorentzOscillatorOptical(eps_inf=6.49, oscillators=((de, T, g),))
    w = wavenumber_to_rad_s(np.linspace(300.0, 1500.0, 241))
    assert fac.eps_omega(w) == pytest.approx(lor.eps_omega(w), rel=1e-12, abs=0.0)
    bad = LorentzOscillatorOptical(eps_inf=6.49, oscillators=((6.49 * (L * L - T * T), T, g),))
    assert not np.allclose(fac.eps_omega(w), bad.eps_omega(w), rtol=0.1)


def test_factorized_passivity_window_is_enforced():
    """Im(eps) of one factorized mode has the sign of g_TO w_LO^2 - g_LO w_TO^2 + w^2 (g_LO - g_TO): passive iff
    g_TO <= g_LO <= g_TO (w_LO/w_TO)^2. Both edges are refused outside, accepted inside; a direct evaluation
    confirms the refused sets really are active (Im < 0 somewhere). Multi-mode sets are checked numerically."""
    T, L = wavenumber_to_rad_s(666.4), wavenumber_to_rad_s(909.9)
    gT = wavenumber_to_rad_s(3.1)
    hi = gT * (L / T) ** 2
    PolarLatticeOptical(eps_inf=4.08, modes=((T, L, gT, gT),))
    PolarLatticeOptical(eps_inf=4.08, modes=((T, L, gT, 0.999 * hi),))
    for gL in (0.5 * gT, 1.2 * hi):
        with pytest.raises(ValueError, match="gain"):
            PolarLatticeOptical(eps_inf=4.08, modes=((T, L, gT, gL),))
        w = np.geomspace(1e-3 * T, 1e3 * L, 20001)
        num = gT * L ** 2 - gL * T ** 2 + w ** 2 * (gL - gT)
        assert np.min(num) < 0.0                                      # the refused set is genuinely active
    two = tuple(tuple(wavenumber_to_rad_s(v) for v in m) for m in ((400.0, 500.0, 5.0, 5.0), (800.0, 1000.0, 8.0, 2.0)))
    with pytest.raises(ValueError, match="gain"):
        PolarLatticeOptical(eps_inf=3.0, modes=two)
    with pytest.raises(ValueError, match="interlace"):
        PolarLatticeOptical.from_wavenumbers(3.0, ((800.0, 1000.0, 5.0, 5.0), (900.0, 1100.0, 5.0, 5.0)))


def test_lst_check():
    """check_lst returns the relative deviation and refuses beyond its tolerance."""
    lat = PolarLatticeOptical.from_wavenumbers(6.56, ((796.6, 968.7, 2.0, 2.0),))
    assert lat.check_lst(lat.static_permittivity()) == pytest.approx(0.0, abs=1e-14)
    assert lat.check_lst(9.9, rtol=0.05) == pytest.approx(lat.static_permittivity() / 9.9 - 1.0, rel=1e-12, abs=0.0)
    with pytest.raises(ValueError, match="LST"):
        lat.check_lst(8.0, rtol=0.05)


def test_plasmon_phonon_coupled_modes():
    """Doped polar crystal = DrudeOptical with the lattice as background (with_free_carriers): the zeros of
    Re(eps) with negligible damping are the closed-form coupled modes w_-, w_+ (roots of w^4 - w^2 (w_LO^2 +
    wp^2/eps_inf) + wp^2 w_TO^2/eps_inf = 0) to 1e-6; limits w_+ -> w_LO, w_- -> 0 for wp -> 0 and w_- -> w_TO,
    w_+ -> wp/sqrt(eps_inf) for large wp. The background refuses a DrudeOptical with a different eps_inf.
    Mutant: adding the Drude term to eps_inf alone (no lattice) misses w_-."""
    ei, T, L = 6.78, wavenumber_to_rad_s(783.6), wavenumber_to_rad_s(964.2)
    tiny = 1e-6 * T
    lat = PolarLatticeOptical(eps_inf=ei, modes=((T, L, tiny, tiny),))
    n, m = 3e24, 0.29 * M_E
    wp = math.sqrt(n * Q_E ** 2 / (EPS0 * m))
    doped = with_free_carriers(lat, m_opt_kg=m, gamma_rad_s=tiny)
    re = lambda w: complex(doped.eps(2 * math.pi * C_LIGHT / w, n_m3=n)).real   # noqa: E731
    wm, wpl = coupled_mode_frequencies(ei, T, L, wp)
    assert brentq(re, 0.5 * wm, 0.999 * T, xtol=1e-3) == pytest.approx(wm, rel=1e-6, abs=0.0)
    assert brentq(re, 1.001 * T, 3.0 * L, xtol=1e-3) == pytest.approx(wpl, rel=1e-6, abs=0.0)
    a, b = coupled_mode_frequencies(ei, T, L, 1e-6 * T)
    assert b == pytest.approx(L, rel=1e-9, abs=0.0) and a < 1e-5 * T
    a, b = coupled_mode_frequencies(ei, T, L, 1e3 * L)
    assert a == pytest.approx(T, rel=1e-5, abs=0.0) and b == pytest.approx(1e3 * L / math.sqrt(ei), rel=1e-5, abs=0.0)
    from dynameta.materials import DrudeOptical
    with pytest.raises(ValueError, match="eps_inf"):
        DrudeOptical(eps_inf=6.5, m_opt_kg=m, gamma_rad_s=tiny, background=lat).eps(10e-6, n_m3=n)
    plain = DrudeOptical(eps_inf=ei, m_opt_kg=m, gamma_rad_s=tiny)
    re2 = lambda w: complex(plain.eps(2 * math.pi * C_LIGHT / w, n_m3=n)).real   # noqa: E731
    assert brentq(re2, 0.1 * wm, 3.0 * L, xtol=1e-3) != pytest.approx(wm, rel=0.2, abs=0.0)


def test_gaussian_oscillator_is_odd_passive_and_causal_in_closed_form():
    """Gaussian oscillator: Im(eps) >= 0 for nu > 0 and odd in nu; Re(eps) - eps_inf even; the Dawson-function
    real part equals a direct principal-value Kramers-Kronig integral of the Gaussian Im part (numerical
    quadrature here) to 1e-4. Mutant: Re part with the sign of the Dawson difference flipped misses."""
    from scipy.integrate import quad
    g = GaussianOscillatorOptical(2.0, ((1.5, 1000.0, 80.0),))
    nu = np.linspace(10.0, 3000.0, 300)
    e = g.eps_wavenumber(nu)
    assert np.all(e.imag >= 0.0)
    em = g.eps_wavenumber(-nu)
    assert em.imag == pytest.approx(-e.imag, abs=1e-14) and em.real == pytest.approx(e.real, abs=1e-14)
    for nu0 in (700.0, 980.0, 1100.0, 1600.0):
        im = lambda x: float(g.eps_wavenumber(x).imag)                   # noqa: E731
        pv = quad(lambda x: x * im(x) / (x + nu0), 0.0, 6000.0, weight="cauchy", wvar=nu0, limit=400)[0]
        re_kk = 2.0 + (2.0 / math.pi) * pv
        assert float(g.eps_wavenumber(nu0).real) == pytest.approx(re_kk, abs=1e-4)
        mutant = 2.0 - (float(g.eps_wavenumber(nu0).real) - 2.0)
        assert mutant != pytest.approx(re_kk, abs=1e-2)


@pytest.mark.parametrize("which", ["factorized", "lorentz_drude", "gaussian"])
def test_kramers_kronig_consistency(which):
    """check_kk on each model (broadened lines so the shipped grid resolves them): rms_norm < 1e-3 (measured
    3.2e-4 / 6.8e-5 / 4.7e-4 for the three models at N = 8000; the residual is set by the truncated band, not
    the grid, so it plateaus under refinement: N = 16000 must not raise it by more than 5 %, measured -7 % ..
    0 %); the conjugated (gain) mutant sits at > 0.1."""
    if which == "factorized":
        m = PolarLatticeOptical.from_wavenumbers(6.49, ((795.9, 972.3, 20.0, 20.0),))
        wmax = wavenumber_to_rad_s(10000.0)
    elif which == "lorentz_drude":
        m = LorentzOscillatorOptical.from_ev(4.74, ((2.183, 4.1, 1.45), (1.134, 1.85, 1.33)), (7.01, 0.23))
        wmax = 20.0 * 7.01 * 1.519e15
    else:
        m = GaussianOscillatorOptical(2.1232, ((3.7998, 1089.7, 31.454), (7.8147, 1058.2, 63.153)))
        wmax = wavenumber_to_rad_s(12000.0)

    class _Flip:
        def eps(self, lam, n_m3=None):
            return np.conj(m.eps(lam))
    res = {}
    for N in (8000, 16000):
        w = np.linspace(wmax / N, wmax, N)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res[N] = (check_kk(m, w, self_calib=False)["rms_norm"], check_kk(_Flip(), w, self_calib=False)["rms_norm"])
    assert res[8000][0] < 1e-3 and res[16000][0] <= res[8000][0] * 1.05
    assert res[8000][1] > 0.1 and res[16000][1] > 0.5 * res[8000][1]


def test_fdtd_single_pole_fit_recovers_a_single_mode_lattice():
    """FDTD route: optics.fdtd_seam.fit_drude_lorentz (with_drude=False) on eps(lambda) sampled from a single-
    mode factorized lattice with g_TO = g_LO -- exactly a Lorentz pole -- over a band wide enough to pin eps_inf
    (300-4000 cm^-1; a band confined to the Reststrahlen region leaves eps_inf and d_eps degenerate) recovers
    eps_inf, d_eps, w_TO and the damping within 1 %."""
    from dynameta.optics.fdtd_seam import fit_drude_lorentz
    lat = PolarLatticeOptical.from_wavenumbers(6.49, ((795.9, 972.3, 6.0, 6.0),))
    lam = _lam(np.linspace(300.0, 4000.0, 81))
    fit = fit_drude_lorentz(lam, [lat.eps(L) for L in lam], with_drude=False)
    T, L = wavenumber_to_rad_s(795.9), wavenumber_to_rad_s(972.3)
    assert fit["eps_inf"] == pytest.approx(6.49, rel=0.01, abs=0.0)
    assert fit["lorentz_delta_eps"] == pytest.approx(6.49 * (L * L - T * T) / (T * T), rel=0.01, abs=0.0)
    assert fit["lorentz_w0_rad_s"] == pytest.approx(T, rel=0.01, abs=0.0)
    assert fit["lorentz_gamma_rad_s"] == pytest.approx(wavenumber_to_rad_s(6.0), rel=0.01, abs=0.0)


def _absorption(eps_film, d, lam, pol, theta=60.0):
    from dynameta.optics.tmm_reference import stack_rta
    layers = [(np.sqrt(eps_film + 0j), d), (np.sqrt(-3000.0 + 1500.0j), 200e-9)]
    R, T, A = stack_rta(1.0, layers, 1.0, lam, theta_deg=theta, pol=pol)
    return A


@pytest.mark.parametrize("model, nu_lo, d", [
    (PolarLatticeOptical.from_wavenumbers(6.49, ((795.9, 972.3, 2.0, 2.0),)), 972.3, 50e-9),
    (GaussianOscillatorOptical(2.1232, ((3.7998, 1089.7, 31.454), (0.46089, 1187.7, 100.46), (1.2520, 797.78, 91.601),
                                        (7.8147, 1058.2, 63.153), (1.0313, 446.13, 275.111), (5.3757, 443.00, 45.220),
                                        (6.3305, 465.80, 22.680), (1.2948, 1026.7, 232.14))), 1253.15, 50e-9)])
def test_berreman_mode_of_a_thin_film_on_metal(model, nu_lo, d):
    """A film much thinner than the wavelength on a metal, p-polarized at 60 deg: the absorption peaks at the
    LO / ENZ frequency (the Berreman mode, which needs a field normal to the film); s-polarized light shows no
    such peak. Through the planar coherent TMM (optics.tmm_reference.stack_rta). 3C-SiC (LO 972.3 cm^-1) and
    fused silica (energy-loss peak 1253 cm^-1; thermal oxide's LO 1256 cm^-1, Kirk 1988): p peak within 0.5 %
    of the LO; s absorption at that frequency < 1/5 of p. Mutant: the same scan at normal incidence has no
    such peak (p = s)."""
    pytest.importorskip("tmm")
    nu = np.linspace(0.85 * nu_lo, 1.12 * nu_lo, 541)
    Ap = np.array([_absorption(complex(model.eps(_lam(v))), d, _lam(v), "p") for v in nu])
    k = int(np.argmax(Ap))
    assert 0 < k < len(nu) - 1
    assert nu[k] == pytest.approx(nu_lo, rel=0.005, abs=0.0)
    As = _absorption(complex(model.eps(_lam(nu[k]))), d, _lam(nu[k]), "s")
    assert As < 0.2 * Ap[k]
    A0 = _absorption(complex(model.eps(_lam(nu[k]))), d, _lam(nu[k]), "p", theta=0.0)
    assert A0 < 0.2 * Ap[k]


def test_conversions_and_validation():
    assert rad_s_to_wavenumber(wavenumber_to_rad_s(1234.5)) == pytest.approx(1234.5, rel=1e-14, abs=0.0)
    with pytest.raises(ValueError):
        LorentzOscillatorOptical(eps_inf=2.0, oscillators=((-1.0, 1e14, 1e12),))
    with pytest.raises(ValueError):
        GaussianOscillatorOptical(2.0, ((1.0, 1000.0, 0.0),))
    lat = PolarLatticeOptical.from_wavenumbers(6.49, ((795.9, 972.3, 2.0, 2.0),), wavelength_range_m=(5e-6, 50e-6))
    with pytest.raises(ValueError, match="validity range"):
        lat.eps(1.55e-6)
    assert np.shape(lat.eps(10e-6, n_m3=np.ones((2, 3)))) == (2, 3)
