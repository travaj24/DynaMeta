"""Fast unit tests for the carrier-heating two-temperature ENZ driver (carriers.carrier_heating, R9).
Pure numpy/scipy (no devsim/ngsolve/fdtd). The rigorous oracle is validation/carrier_heating_enz.py."""
import numpy as np
import pytest

from dynameta.constants import M_E, KB
from dynameta.materials import DrudeOptical
from dynameta.carriers.carrier_heating import (TwoTempParams, two_temperature_response,
                                               carrier_heating_transient, kane_mass_of_Te,
                                               gamma_of_Te, fermi_energy_J)
from dynameta.transient_optics import optical_transient_response

M0, ALPHA_EV, GAMMA0, N = 0.35 * M_E, 0.5, 1.0e14, 1.0e27
DRUDE0 = DrudeOptical(eps_inf=3.9, m_opt_kg=M0, gamma_rad_s=GAMMA0)
GAMMA_E = (np.pi ** 2 / 2.0) * N * KB ** 2 / float(fermi_energy_J(N, M0, ALPHA_EV))
PARAMS = TwoTempParams(C_e=lambda Te: GAMMA_E * Te, C_l=2.4e6, G_e_l=6.0e15, alpha_abs=1.0)


def test_kane_mass_off_switch_and_monotone():
    # alpha=0 -> m0 EXACTLY (the byte-identical off-switch), at EVERY T_e and on arrays (audit C-1:
    # the f-sum bracket collapses to 1 identically, so this is exact, not merely close)
    assert kane_mass_of_Te(M0, 0.0, N, 5000.0) == M0
    assert float(kane_mass_of_Te(M0, 0.0, N, 300.0)) == M0
    for Te in (0.0, 1.0, 300.0, 3000.0, 12000.0):
        assert kane_mass_of_Te(M0, 0.0, N, Te) == M0
    assert np.all(kane_mass_of_Te(M0, 0.0, N, np.linspace(300.0, 6000.0, 7)) == M0)
    # m_c rises with Te (hot electrons occupy higher, heavier states on the nonparabolic band)
    m_cold = float(kane_mass_of_Te(M0, ALPHA_EV, N, 300.0))
    m_hot = float(kane_mass_of_Te(M0, ALPHA_EV, N, 3000.0))
    assert m_hot > m_cold > M0


def test_gamma_off_switch():
    assert gamma_of_Te(GAMMA0, 5000.0, p=0.0) == GAMMA0          # p=0 -> gamma0 exactly
    assert gamma_of_Te(GAMMA0, 600.0, p=1.0) == pytest.approx(GAMMA0 * 2.0)   # linear in Te


def test_two_temperature_no_pump_stays_at_T0():
    t = np.linspace(0.0, 2e-12, 200)
    _t, Te, Tl = two_temperature_response(t, lambda tt: 0.0, PARAMS, T0_K=300.0)
    assert np.max(np.abs(Te - 300.0)) < 1e-9 and np.max(np.abs(Tl - 300.0)) < 1e-9


def test_two_temperature_monotone_rise_then_fall():
    t = np.linspace(0.0, 3e-12, 400)
    pump = lambda tt: 3e20 * np.exp(-((tt - 0.4e-12) / 6e-14) ** 2)
    _t, Te, _Tl = two_temperature_response(t, pump, PARAMS, T0_K=300.0)
    ipk = int(np.argmax(Te))
    assert Te[ipk] > 800.0                                       # the pump heats the electrons
    assert np.all(np.diff(Te[:ipk + 1]) >= -1e-6)                # monotone up to the peak
    assert np.all(np.diff(Te[ipk:]) <= 1e-6)                     # monotone cooling after


def test_two_temperature_energy_conservation():
    # no pump after the pulse: total thermal energy = integral of absorbed power (no loss term)
    t = np.linspace(0.0, 4e-12, 800)
    pump = lambda tt: 2e20 * np.exp(-((tt - 0.4e-12) / 5e-14) ** 2)
    _t, Te, Tl = two_temperature_response(t, pump, PARAMS, T0_K=300.0)
    U_in = np.trapezoid(np.array([pump(tt) for tt in t]), t) if hasattr(np, "trapezoid") else \
        np.trapz(np.array([pump(tt) for tt in t]), t)
    U_e = 0.5 * GAMMA_E * (Te[-1] ** 2 - 300.0 ** 2)             # electron energy (C_e = gamma_e Te)
    U_l = PARAMS.C_l * (Tl[-1] - 300.0)
    assert abs((U_e + U_l) - U_in) / U_in < 0.05                # conserved (G only redistributes)


def test_carrier_heating_reduces_to_fixed_drude():
    # alpha_per_eV=0, gamma_p=0 -> per-instant Drude collapses to drude0 -> byte-identical R(t)
    t = np.linspace(0.0, 2e-12, 150)
    pump = lambda tt: 3e20 * np.exp(-((tt - 0.4e-12) / 6e-14) ** 2)
    n_of_t = lambda tt: N
    _t, R_fix, _T, _e = optical_transient_response(t, n_of_t, 1500e-9, drude_model=DRUDE0)
    _th, R_h, _Th, _eh, _Te, _Tl = carrier_heating_transient(t, pump, 1500e-9, drude0=DRUDE0,
                                                             ttm_params=PARAMS, n_m3=N,
                                                             alpha_per_eV=0.0, gamma_p=0.0)
    assert np.max(np.abs(R_h - R_fix)) < 1e-12


def test_optical_transient_requires_exactly_one_drude():
    t = np.linspace(0.0, 1e-12, 10)
    with pytest.raises(ValueError):
        optical_transient_response(t, lambda tt: N, 1500e-9)                       # neither
    with pytest.raises(ValueError):
        optical_transient_response(t, lambda tt: N, 1500e-9, drude_model=DRUDE0,
                                   drude_of_t=lambda tt: DRUDE0)                    # both


def test_legacy_mean_energy_mass_sommerfeld_coefficient_vs_exact_fd():
    # audit C2-2: the (5 pi^2/12) Sommerfeld coefficient must carry the Kane-DOS factor
    # (1+2aE_F)/(1+aE_F). Every prior gate pinned limits/scaling only and was blind to
    # it (parabolic coefficient understated the heating SHIFT d<E> by 18-25% here).
    # Reference: EXACT Fermi-Dirac mean energy over the Kane DOS at fixed n -- an
    # independent numeric path with no Sommerfeld expansion. The pinned quantity is the
    # Te-EXCURSION dm(Te) = m(Te) - m(Te->0) (the modulation observable); the T=0
    # baseline itself keeps the parabolic (3/5)E_F convention (a static offset, out of C2-2 scope).
    # audit C-1: the MEAN-ENERGY mass this gate pins is no longer the Drude mass -- it moved to the
    # private _kane_mass_mean_energy_of_Te and the wp^2 path now uses the f-sum mass. Re-targeted
    # here (rather than deleted) so the C2-2 coefficient regression keeps a live gate.
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from dynameta.carriers.carrier_heating import fermi_energy_J, _kane_mass_mean_energy_of_Te
    from dynameta.constants import HBAR, KB, M_E, Q_E

    m0, alpha, n = 0.35 * M_E, 0.5, 1.0e27
    a = alpha / Q_E
    pref = (2.0 * m0) ** 1.5 / (2.0 * np.pi ** 2 * HBAR ** 3)   # spin-2 3D DOS prefactor

    def g(E):
        return pref * (1.0 + 2.0 * a * E) * np.sqrt(np.maximum(E * (1.0 + a * E), 0.0))

    def fd(E, mu, kT):
        return 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -60.0, 60.0)))

    E_F = float(fermi_energy_J(n, m0, alpha))
    Emax = 12.0 * E_F

    def mean_E_exact(Te):
        kT = KB * Te
        n_of = lambda mu: quad(lambda E: g(E) * fd(E, mu, kT), 0.0, Emax, limit=300)[0]
        mu = brentq(lambda m: n_of(m) - n, -E_F, 3.0 * E_F, xtol=1e-26)
        return quad(lambda E: E * g(E) * fd(E, mu, kT), 0.0, Emax, limit=300)[0] / n

    T0 = 1.0                                                    # ~T=0 baseline
    m_base_code = float(_kane_mass_mean_energy_of_Te(m0, alpha, n, T0))
    m_base_ex = m0 * (1.0 + 2.0 * a * mean_E_exact(T0))
    for Te in (600.0, 1000.0, 1500.0):
        dm_code = float(_kane_mass_mean_energy_of_Te(m0, alpha, n, Te)) - m_base_code
        dm_ex = m0 * (1.0 + 2.0 * a * mean_E_exact(Te)) - m_base_ex
        # corrected coefficient tracks the exact shift to a few % (Sommerfeld O(x^4)
        # truncation); the pre-fix parabolic coefficient missed by 18-25% -> 5x margin
        assert abs(dm_code / dm_ex - 1.0) < 0.05, (Te, dm_code / dm_ex)


def test_kane_mass_rejects_non_finite_inputs():
    """FIX-VERIFY W1 item 6. Every comparison against NaN is False, so ``np.any(n <= 0)`` and
    ``np.any(Te < 0)`` both passed a NaN straight through; it then died deep inside the chemical-
    potential root finder as ``RuntimeError: kane_mass_of_Te: could not bracket mu(T_e) from
    above`` -- a message that points at brentq rather than at the caller's argument."""
    import numpy as np
    import pytest as _pytest
    from dynameta.carriers.carrier_heating import kane_mass_of_Te
    from dynameta.constants import M_E

    m0, alpha, n = 0.35 * M_E, 0.5, 1.0e27
    for bad_n, bad_te in ((np.nan, 300.0), (n, np.nan), (np.inf, 300.0), (n, np.inf)):
        with _pytest.raises(ValueError, match="finite"):
            kane_mass_of_Te(m0, alpha, bad_n, bad_te)
    # arrays too (the guard runs after broadcasting)
    with _pytest.raises(ValueError, match="finite"):
        kane_mass_of_Te(m0, alpha, np.array([n, np.nan]), 300.0)
    # the existing sign guards still fire, with their own message
    with _pytest.raises(ValueError, match="n_m3 > 0"):
        kane_mass_of_Te(m0, alpha, -1.0, 300.0)
    with _pytest.raises(ValueError, match="Te_K >= 0"):
        kane_mass_of_Te(m0, alpha, n, -1.0)
    # and a valid call is untouched
    assert float(kane_mass_of_Te(m0, alpha, n, 300.0)) > m0


# ===================== audit C-1: the f-sum (Drude/conductivity) mass =====================
# wp^2 = n e^2/(eps0 m) is an f-SUM-RULE quantity, so kane_mass_of_Te must return the conductivity
# mass 1/m_c = (1/(3n)) Int g(E) f(E) [Lap_k E/hbar^2] dE -- NOT the mass evaluated at the mean
# energy (which the module used to return: 17.2% low at the repo ITO preset). The auditor's proposed
# closed form m0(1 + 2 a mu(T_e)) is NOT the fix -- mu FALLS with T_e at fixed n, so it inverts the
# sign of the whole R9 observable; only the direct integral is correct.

def test_c1_fsum_mass_T0_identity_is_fermi_surface_mass():
    """GATE 1 (C-1). At T_e = 0 the f-sum integral reduces ANALYTICALLY (divergence theorem over the
    Fermi sphere) to the Fermi-surface mass m_c = m0 (1 + 2 alpha E_F). That closed form is the gate,
    never the implementation -- and it is independently the value materials.KaneOpticalMass returns,
    so the two entry points must agree to the same tolerance."""
    from dynameta.constants import Q_E
    from dynameta.materials import KaneOpticalMass

    for alpha in (0.05, 0.2, 0.5, 1.0, 2.0):
        for n in (1.0e24, 1.0e25, 1.0e26, 1.0e27, 5.0e27):
            for m0 in (0.20 * M_E, 0.35 * M_E):
                m_c = float(kane_mass_of_Te(m0, alpha, n, 0.0))
                E_F = float(fermi_energy_J(n, m0, alpha))
                closed = m0 * (1.0 + 2.0 * (alpha / Q_E) * E_F)
                assert m_c == pytest.approx(closed, rel=1e-10), (alpha, n, m0)
                assert m_c == pytest.approx(float(KaneOpticalMass(m0_kg=m0, alpha_eV=alpha)(n)),
                                            rel=1e-10), (alpha, n, m0)
    # continuity into the T=0 limit: 1 K is indistinguishable from 0 K at these E_F
    assert float(kane_mass_of_Te(M0, ALPHA_EV, N, 1.0)) == pytest.approx(
        float(kane_mass_of_Te(M0, ALPHA_EV, N, 0.0)), rel=1e-8)


def test_c1_fsum_mass_vs_independent_energy_space_quadrature():
    """GATE (C-1) INDEPENDENT ORACLE: re-evaluate the f-sum integral in ENERGY space with adaptive
    scipy.quad and a scipy.brentq mu(T_e) solve -- a different variable, a different quadrature rule
    and a different mu bracket from the shipped fixed-order k-space Gauss-Legendre implementation.
    Measured agreement is ~1e-12; gated at 1e-8."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from dynameta.constants import HBAR, Q_E

    def m_cond_energy_space(m0, alpha, n, Te):
        a = alpha / Q_E
        pref = (2.0 * m0) ** 1.5 / (2.0 * np.pi ** 2 * HBAR ** 3)        # g_s=2, g_v=1
        g = lambda E: pref * (1.0 + 2.0 * a * E) * np.sqrt(max(E * (1.0 + a * E), 0.0))
        # (1/3) Lap_k E / hbar^2 for E(1+aE) = hbar^2 k^2/(2 m0)
        w = lambda E: (1.0 / (3.0 * m0)) * (3.0 / (1.0 + 2.0 * a * E)
                                            - 4.0 * a * E * (1.0 + a * E) / (1.0 + 2.0 * a * E) ** 3)
        E_F = float(fermi_energy_J(n, m0, alpha))
        kT = KB * Te
        f = lambda E, mu: 1.0 / (1.0 + np.exp(np.clip((E - mu) / kT, -500.0, 500.0)))
        Emax = E_F + 80.0 * kT
        n_of = lambda mu: quad(lambda E: g(E) * f(E, mu), 0.0, Emax, limit=400)[0]
        lo = E_F - max(kT, 0.05 * E_F)
        while n_of(lo) > n:
            lo -= max(kT, 0.05 * E_F)
        mu = brentq(lambda m: n_of(m) - n, lo, E_F + 10.0 * kT, xtol=1e-16 * E_F, rtol=8.9e-16)
        dens = quad(lambda E: g(E) * f(E, mu), 0.0, Emax, limit=400)[0]
        inv = quad(lambda E: g(E) * f(E, mu) * w(E), 0.0, Emax, limit=400)[0]
        return dens / inv

    for n in (1.0e25, 1.0e26, 1.0e27):
        for Te in (300.0, 1000.0, 3000.0, 6000.0):
            got = float(kane_mass_of_Te(M0, ALPHA_EV, n, Te))
            ref = m_cond_energy_space(M0, ALPHA_EV, n, Te)
            assert got == pytest.approx(ref, rel=1e-8), (n, Te, got / ref - 1.0)


def test_c1_n5_rise_sign_in_both_aEF_regimes():
    """GATE 3 (C-1 + re-grade N5). Two statements, both keyed on a*E_F:
      (a) the TRUE f-sum mass rises monotonically with T_e in BOTH regimes -- the R9 sign
          (wp DROPS, Re(eps) moves toward eps_inf) never flips; and
      (b) the pre-fix mean-energy mass gets the SIZE of that rise wrong with a PARAMETER-DEPENDENT
          SIGN of error: it UNDER-states below a*E_F ~ 0.15 (0.876x at a*E_F = 0.102, 600 K) and
          OVER-states above it (1.219x at the repo ITO preset a*E_F = 0.378, 600 K). N5's point is
          exactly that C-1 cannot be stated as a universal over-statement."""
    from dynameta.constants import Q_E
    from dynameta.carriers.carrier_heating import _kane_mass_mean_energy_of_Te

    cases = {}
    for n in (1.0e26, 1.0e27):                      # a*E_F = 0.1019 (low) and 0.3780 (high)
        a_EF = ALPHA_EV * float(fermi_energy_J(n, M0, ALPHA_EV)) / Q_E
        Tes = np.linspace(0.0, 6000.0, 61)
        m = np.asarray(kane_mass_of_Te(M0, ALPHA_EV, n, Tes), dtype=float)
        assert np.all(np.diff(m) > 0.0), (n, a_EF)                       # (a) monotone rise
        m_t0, m_t6 = float(m[0]), float(kane_mass_of_Te(M0, ALPHA_EV, n, 600.0))
        l_t0 = float(_kane_mass_mean_energy_of_Te(M0, ALPHA_EV, n, 0.0))
        l_t6 = float(_kane_mass_mean_energy_of_Te(M0, ALPHA_EV, n, 600.0))
        cases[n] = (a_EF, (l_t6 / l_t0 - 1.0) / (m_t6 / m_t0 - 1.0))
    a_lo, ratio_lo = cases[1.0e26]
    a_hi, ratio_hi = cases[1.0e27]
    assert a_lo == pytest.approx(0.10188, rel=1e-3) and a_hi == pytest.approx(0.37802, rel=1e-3)
    assert ratio_lo < 1.0 and ratio_lo == pytest.approx(0.8757, rel=1e-3)   # (b) UNDER-states
    assert ratio_hi > 1.0 and ratio_hi == pytest.approx(1.2193, rel=1e-3)   # (b) OVER-states


def test_c1_ito_preset_cold_mass_and_enz_crossing_goldens():
    """GATE 4 (C-1) -- the repo ITO preset (m0 = 0.35 m_e, alpha = 0.5/eV, n = 1e27, a*E_F = 0.378).
    RE-BASELINED at C-1: the cold (300 K) conductivity mass is 1.7587969739 m0, +20.7629% above the
    pre-fix mean-energy value 1.4564050460 m0 (equivalently, the old mass was 17.193% LOW and every
    wp^2 built from it 20.763% HIGH), and the cold ENZ crossing moves 148.802 nm to the RED,
    1493.4033 -> 1642.2053 nm. New values pinned at rtol 1e-8 (BLAS-safe)."""
    from dynameta.constants import EPS0, C_LIGHT, Q_E
    from dynameta.carriers.carrier_heating import _kane_mass_mean_energy_of_Te

    m_new = float(kane_mass_of_Te(M0, ALPHA_EV, N, 300.0))
    m_old = float(_kane_mass_mean_energy_of_Te(M0, ALPHA_EV, N, 300.0))
    assert m_new / M0 == pytest.approx(1.7587969739, rel=1e-8)             # NEW golden (C-1)
    assert m_old / M0 == pytest.approx(1.4564050460, rel=1e-8)             # pre-fix, for the delta
    assert m_old / m_new - 1.0 == pytest.approx(-0.171931, rel=1e-5)       # mass was 17.19% low
    assert m_new / m_old - 1.0 == pytest.approx(+0.207629, rel=1e-5)       # wp^2 was 20.76% high
    # hot points on the same preset (the R9 modulation observable), also re-baselined
    assert float(kane_mass_of_Te(M0, ALPHA_EV, N, 600.0)) / M0 == pytest.approx(1.7670575837, rel=1e-8)
    assert float(kane_mass_of_Te(M0, ALPHA_EV, N, 3000.0)) / M0 == pytest.approx(2.0014807657, rel=1e-8)

    # cold ENZ crossing: Re(eps) = eps_inf - wp^2/(w^2 + gamma^2) = 0 for the module's own Drude
    def lam_enz(m, eps_inf=3.9, gamma=GAMMA0):
        wp2 = N * Q_E ** 2 / (EPS0 * m)
        return 2.0 * np.pi * C_LIGHT / np.sqrt(wp2 / eps_inf - gamma ** 2)

    lam_new, lam_old = lam_enz(m_new), lam_enz(m_old)
    assert lam_new * 1e9 == pytest.approx(1642.2053088, rel=1e-8)          # NEW golden (C-1)
    assert lam_old * 1e9 == pytest.approx(1493.4032846, rel=1e-8)          # pre-fix
    assert (lam_new - lam_old) * 1e9 == pytest.approx(148.802, rel=1e-5)   # 148.8 nm to the RED
    # the crossing is a genuine Re(eps) = 0 of the shipped DrudeOptical, not just of the closed form
    eps = complex(DrudeOptical(eps_inf=3.9, m_opt_kg=m_new,
                               gamma_rad_s=GAMMA0).eps(lam_new, n_m3=N))
    assert abs(eps.real) < 1e-9 * 3.9


def test_transient_rejects_silent_callable_substitution():
    # audit C5-10: a calibrated DrudeOptical (callable m_opt/gamma) used to be silently
    # replaced by M_E / 1e14 rad/s -- a different material, violating the off-switch
    import numpy as np
    import pytest
    from dynameta.carriers.carrier_heating import TwoTempParams, carrier_heating_transient
    from dynameta.constants import M_E
    from dynameta.materials import DrudeOptical, KaneOpticalMass, MatthiessenGamma
    ttm = TwoTempParams(C_e=3.0e4, C_l=2.5e6, G_e_l=3.0e16)
    t = np.linspace(0.0, 1e-12, 8)
    I = lambda tt: 0.0
    d_mass = DrudeOptical(eps_inf=3.9, m_opt_kg=KaneOpticalMass(m0_kg=0.3 * M_E, alpha_eV=0.5),
                          gamma_rad_s=1.6e14)
    with pytest.raises(ValueError, match="m0_kg"):
        carrier_heating_transient(t, I, 1.5e-6, drude0=d_mass, ttm_params=ttm,
                                  n_m3=6e26, alpha_per_eV=0.5)
    d_gam = DrudeOptical(eps_inf=3.9, m_opt_kg=0.3 * M_E,
                         gamma_rad_s=MatthiessenGamma(gamma_const_rad_s=1.6e14))
    with pytest.raises(ValueError, match="gamma_rad_s"):
        carrier_heating_transient(t, I, 1.5e-6, drude0=d_gam, ttm_params=ttm,
                                  n_m3=6e26, alpha_per_eV=0.5, m0_kg=0.3 * M_E)
