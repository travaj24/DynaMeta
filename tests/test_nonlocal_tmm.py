"""Gates for the hydrodynamic (nonlocal) Drude layered TMM (roadmap item 2.4).

Physics recap (see ``dynameta/optics/nonlocal_tmm.py`` for the full derivation):
  * the transverse response is the ORDINARY local Drude ``eps_T``; the hydrodynamic pressure
    adds a LONGITUDINAL wave with ``k_L**2 = (omega**2 + i gamma omega - wp**2/eps_inf)/beta**2``;
  * p-pol couples to it through the ABC ``J_normal = 0`` at each metal face (s-pol does not);
  * a thin film shows bulk-plasmon standing-wave absorption above ``omega_p`` at ``k_L d = m*pi``
    and a 1/d ENZ/Berreman blueshift -- both absent in the local model.

Gates:
  1. beta -> 0 == the LOCAL result (tmm_reference AND an internal local slab) to <1e-8, metal
     film + dielectric spacer, s and p, normal + 45 deg.
  2. bulk-plasmon standing-wave absorption peaks land within 1% of the k_L d = m*pi predictions
     (a symmetric thin film couples to the ODD modes m = 1, 3, 5 -- see the selection-rule note).
  3. those peaks VANISH in the local limit (beta -> 0).
  4. energy budget: lossless (gamma = D = 0) => A = 0 to 1e-10; lossy => R, T, A all >= 0.
  5. GNOR: the diffusion knob D broadens the m = 1 resonance (its FWHM = 2|Im(pole)| grows
     monotonically with D) without moving the centre by more than the broadening.
  6. optics.resonance.find_poles on the pole hook locates the m = 1 bulk-plasmon pole within 2%
     of the real-axis absorption peak, decaying (Im < 0), finite Q.
  7. ENZ/Berreman feature of an ITO-like film BLUESHIFTS in the nonlocal model vs local, more so
     for a thinner film (1/d): direction + monotonicity.

Run: python -m pytest tests/test_nonlocal_tmm.py -q
"""
import dataclasses
import math

import numpy as np
import pytest
from scipy.signal import find_peaks

from dynameta.optics import nonlocal_tmm as nt
from dynameta.optics.tmm_reference import stack_rta
from dynameta.optics.resonance import find_poles, newton_refine, pole_q


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _passive_n(eps):
    n = np.sqrt(complex(eps))
    return n if n.imag >= 0 else -n


def _bulk_plasmon_omega(m, layer):
    """Undamped bulk-plasmon standing-wave frequency from k_L d = m*pi:
    omega_m = sqrt(omega_p**2/eps_inf + beta**2 (m pi/d)**2)  (gamma -> 0, D -> 0)."""
    return math.sqrt(layer.wp ** 2 / layer.eps_inf
                     + layer.beta ** 2 * (m * math.pi / layer.thickness_m) ** 2)


def _spectrum(layer, ws, *, pol="p", n_super=1.0, n_sub=1.0, theta_rad=0.0, k_par_m=None):
    return np.array([nt.rta(w, [layer], pol=pol, n_super=n_super, n_sub=n_sub,
                            theta_rad=theta_rad, k_par_m=k_par_m)[2] for w in ws])


# ------------------------------------------------------------------------------------------------
# unit sanity
# ------------------------------------------------------------------------------------------------
def test_beta_from_vf_conventions():
    vF = 1.39e6
    assert nt.beta_from_vf(vF, "high_freq") == pytest.approx(math.sqrt(0.6) * vF, rel=1e-12)
    assert nt.beta_from_vf(vF, "thomas_fermi") == pytest.approx(math.sqrt(1.0 / 3.0) * vF, rel=1e-12)
    assert nt.beta_from_vf(vF) == nt.beta_from_vf(vF, "high_freq")   # default = high frequency
    with pytest.raises(ValueError):
        nt.beta_from_vf(vF, "nonsense")


def test_eps_transverse_passive_and_enz():
    lay = nt.HydroLayer(eps_inf=1.0, wp=2.0e15, gamma=1.0e14, beta=1.0e6, thickness_m=20e-9)
    eps = nt.eps_transverse(1.0e15, lay)
    assert eps.imag > 0.0                                          # exp(-i w t) => passive absorber
    enz = lay.wp / math.sqrt(lay.eps_inf)
    assert nt.eps_transverse(enz, lay).real == pytest.approx(0.0, abs=5e-2)


def test_kL_squared_and_gnor_knob():
    lay = nt.HydroLayer(eps_inf=1.0, wp=8.0e15, gamma=1.0e13, beta=8.0e5, thickness_m=5e-9)
    w = 9.0e15
    kL2 = nt.kL_squared(w, lay)
    expect = (w ** 2 + 1j * lay.gamma * w - lay.wp ** 2 / lay.eps_inf) / lay.beta ** 2
    assert kL2 == pytest.approx(expect, rel=1e-12)
    # GNOR: beta_eff**2 = beta**2 + D(gamma - i w); a real w adds a negative imaginary part.
    layD = nt.HydroLayer(1.0, 8.0e15, 1.0e13, 8.0e5, 5e-9, D=2.0e-4)
    be2 = nt.beta_eff_squared(w, layD)
    assert be2 == pytest.approx(layD.beta ** 2 + layD.D * (layD.gamma - 1j * w), rel=1e-12)
    assert be2.imag < 0.0


# ------------------------------------------------------------------------------------------------
# Gate 1: beta -> 0 == local TMM (tmm_reference and an internal local slab), s/p, 0/45 deg
# ------------------------------------------------------------------------------------------------
def test_gate1_local_limit_matches_tmm_and_internal():
    lam = 700e-9
    from dynameta.constants import C_LIGHT
    omega = 2.0 * math.pi * C_LIGHT / lam
    # A metallic film (eps_T < 0 here) below its own local dielectric spacer -- a real stack.
    metal = nt.HydroLayer(eps_inf=1.0, wp=1.2e16, gamma=1.1e14, beta=1e-3, thickness_m=25e-9)
    spacer = nt.DielectricLayer(eps=(1.46) ** 2, thickness_m=60e-9)
    eps_T = complex(nt.eps_transverse(omega, metal))
    n_metal = _passive_n(eps_T)
    n_super, n_sub = 1.0, 1.5
    layers_tmm = [(n_metal, metal.thickness_m), (1.46, spacer.thickness_m)]
    layers_nl = [metal, spacer]
    # internal local reference: treat the metal as a plain dielectric slab at eps_T.
    layers_local = [nt.DielectricLayer(eps_T, metal.thickness_m), spacer]
    worst = 0.0
    for theta_deg in (0.0, 45.0):
        for pol in ("s", "p"):
            R_t, T_t, A_t = stack_rta(n_super, layers_tmm, n_sub, lam,
                                      theta_deg=theta_deg, pol=pol)
            R, T, A = nt.rta(omega, layers_nl, pol=pol, n_super=n_super, n_sub=n_sub,
                             theta_rad=math.radians(theta_deg))
            Rl, Tl, Al = nt.rta(omega, layers_local, pol=pol, n_super=n_super, n_sub=n_sub,
                                theta_rad=math.radians(theta_deg))
            assert R == pytest.approx(R_t, abs=1e-8) and T == pytest.approx(T_t, abs=1e-8)
            assert A == pytest.approx(A_t, abs=1e-8)
            # internal local consistency is far tighter than the beta=1e-3 residual to tmm.
            assert R == pytest.approx(Rl, abs=1e-8) and T == pytest.approx(Tl, abs=1e-8)
            worst = max(worst, abs(R - R_t), abs(T - T_t))
    assert worst < 1e-8


# ------------------------------------------------------------------------------------------------
# sodium-like film shared by gates 2, 3, 5, 6
# ------------------------------------------------------------------------------------------------
def _sodium_film(gamma=3.0e12, d=3.0e-9, D=0.0):
    # Na-like: omega_p ~ 5.7 eV, v_F = 1.07e6 m/s -> beta = sqrt(3/5) v_F.
    return nt.HydroLayer(eps_inf=1.0, wp=8.65e15, gamma=gamma,
                         beta=nt.beta_from_vf(1.07e6), thickness_m=d, D=D)


# ------------------------------------------------------------------------------------------------
# Gate 2: bulk-plasmon standing-wave peaks at k_L d = m*pi (odd m: symmetric-film selection rule)
# ------------------------------------------------------------------------------------------------
def test_gate2_bulk_plasmon_standing_waves():
    lay = _sodium_film(gamma=3.0e12, d=3.0e-9)
    wp = lay.wp
    theta = math.radians(45.0)                                    # oblique p-pol: nonzero k_par
    ws = np.linspace(1.0002 * wp, 1.16 * wp, 40000)
    A = _spectrum(lay, ws, pol="p", theta_rad=theta)
    idx, _ = find_peaks(A, prominence=1e-3)
    assert idx.size >= 3, "expected at least three bulk-plasmon absorption peaks above omega_p"
    peak_ws = ws[idx]
    # A symmetric thin film (uniform in-film drive, H_y ~ const since k_z_T d << 1) couples only
    # to the SYMMETRIC longitudinal standing waves -> the ODD orders m = 1, 3, 5.  Each observed
    # peak must land within 1% of its k_L d = m*pi prediction, and above omega_p.
    for m in (1, 3, 5):
        wm = _bulk_plasmon_omega(m, lay)
        got = peak_ws[np.argmin(np.abs(peak_ws - wm))]
        assert got > wp, "bulk-plasmon peak must sit above omega_p"
        assert abs(got - wm) / wm < 0.01, (
            "m={} peak {:.5e} not within 1% of k_L d = m*pi prediction {:.5e}".format(m, got, wm))


# ------------------------------------------------------------------------------------------------
# Gate 3: the peaks VANISH in the local limit (beta -> 0)
# ------------------------------------------------------------------------------------------------
def test_gate3_peaks_vanish_in_local_limit():
    lay = _sodium_film(gamma=3.0e12, d=3.0e-9)
    lay_local = nt.HydroLayer(lay.eps_inf, lay.wp, lay.gamma, 1e-3, lay.thickness_m)  # beta -> 0
    wp = lay.wp
    theta = math.radians(45.0)
    ws = np.linspace(1.0002 * wp, 1.16 * wp, 40000)
    A = _spectrum(lay, ws, pol="p", theta_rad=theta)
    A_loc = _spectrum(lay_local, ws, pol="p", theta_rad=theta)
    # The local film has NO bulk-plasmon structure above omega_p (a featureless, weak, monotone-ish
    # tail); the peaks exist ONLY with nonlocality.
    idx_loc, _ = find_peaks(A_loc, prominence=1e-3)
    idx_nl, _ = find_peaks(A, prominence=1e-3)
    assert idx_loc.size == 0, "local (beta->0) film must have no bulk-plasmon peaks above omega_p"
    assert idx_nl.size >= 3, "nonlocal film must show bulk-plasmon peaks that the local one lacks"
    # the strong (higher-order) resonances dwarf the smooth local background outright.
    for m in (3, 5):
        wm = _bulk_plasmon_omega(m, lay)
        j = int(np.argmin(np.abs(ws - wm)))
        assert A[j] > 3.0 * A_loc[j], (
            "nonlocal absorption at m={} must dwarf the local background".format(m))


# ------------------------------------------------------------------------------------------------
# Gate 4: energy budget R + T + A = 1 to 1e-10 (lossless => A = 0; lossy => non-negative)
# ------------------------------------------------------------------------------------------------
def test_gate4_energy_budget():
    beta = nt.beta_from_vf(1.07e6)
    wp = 8.65e15
    lossless = nt.HydroLayer(1.0, wp, 0.0, beta, 3e-9)            # gamma = D = 0 => no loss channel
    theta = math.radians(40.0)
    # (a) Lossless p-pol BELOW omega_p: metallic (eps_T < 0), no bulk-plasmon resonances and no
    #     eps_T = 0 ENZ singularity there -- A must be 0 (R + T = 1) to machine precision.
    for f in np.linspace(0.50, 0.95, 40):
        w = f * wp
        R, T, A = nt.rta(w, [lossless], pol="p", n_super=1.0, n_sub=1.3, theta_rad=theta)
        assert math.isfinite(A)
        assert R + T + A == pytest.approx(1.0, abs=1e-12)         # A := 1 - R - T (identity)
        assert abs(A) < 1e-10                                     # lossless => no absorption
    # (b) Lossless s-pol has NO longitudinal singularity anywhere -- energy conserved across a
    #     wide band including above omega_p.
    for f in np.linspace(0.50, 1.9, 60):
        w = f * wp
        R, T, A = nt.rta(w, [lossless], pol="s", n_super=1.0, n_sub=1.3, theta_rad=theta)
        assert math.isfinite(A)
        assert R + T + A == pytest.approx(1.0, abs=1e-12)
        assert abs(A) < 1e-10
    # (c) Lossy + GNOR: R, T, A must all be physical (>= 0) -- no spurious interior gain.
    #
    # AUDIT T-1: `R + T + A == 1` is an IDENTITY for this producer (stack_rt returns
    # A := 1 - R - T), and unlike (a)/(b) a LOSSY layer has no `abs(A) < tol` physics anchor to
    # rescue it -- the old assertion here passed for a halved or sign-flipped T. nonlocal_tmm
    # exposes no internal-field/Joule-integral absorption, so there is no independent A to compare
    # against directly; the independent gate is the LOCAL limit, where the very same code path is
    # pinned to the `tmm` package (a wholly separate implementation):
    #   * s-pol carries NO longitudinal physics at all -- eps_T is the whole story, GNOR included --
    #     so the lossy s-pol result must equal the local tmm slab to machine precision;
    #   * p-pol is genuinely nonlocal (it deviates from local by up to ~0.5 in R/T at the m=1 bulk
    #     plasmon, so a local tolerance would be meaningless), so its independent anchor is the
    #     beta -> 0, D = 0 twin of the same lossy film, which must reproduce the local tmm slab.
    lossy = nt.HydroLayer(1.0, wp, 5e13, beta, 3e-9, D=2e-4)
    lossy_local = nt.HydroLayer(1.0, wp, 5e13, 1e-3, 3e-9)        # beta -> 0, D = 0 twin
    from dynameta.constants import C_LIGHT as _C
    for f in np.linspace(0.53, 1.21, 60):
        w = f * wp
        lam = 2.0 * math.pi * _C / w
        n_T = _passive_n(complex(nt.eps_transverse(w, lossy)))    # the transverse (local) index
        for pol in ("s", "p"):
            R, T, A = nt.rta(w, [lossy], pol=pol, n_super=1.0, n_sub=1.3, theta_rad=theta)
            assert R >= -1e-12 and T >= -1e-12 and A >= -1e-12
            assert A > 1e-4                                       # a lossy film genuinely absorbs
            R_t, T_t, A_t = stack_rta(1.0, [(n_T, lossy.thickness_m)], 1.3, lam,
                                      theta_deg=40.0, pol=pol)
            if pol == "s":
                assert R == pytest.approx(R_t, abs=1e-12)         # s-pol IS the local result
                assert T == pytest.approx(T_t, abs=1e-12)
                assert A == pytest.approx(A_t, abs=1e-12)
            else:
                R0, T0, A0 = nt.rta(w, [lossy_local], pol="p", n_super=1.0, n_sub=1.3,
                                    theta_rad=theta)
                assert R0 == pytest.approx(R_t, abs=1e-8)         # p-pol beta->0 == local tmm
                assert T0 == pytest.approx(T_t, abs=1e-8)
                assert A0 == pytest.approx(A_t, abs=1e-8)


# ------------------------------------------------------------------------------------------------
# Gate 6 (built first: gate 5 reuses the pole): find_poles locates the m=1 bulk-plasmon pole
# ------------------------------------------------------------------------------------------------
def test_gate6_pole_finder_locates_m1():
    lay = _sodium_film(gamma=1.0e13, d=3.0e-9)
    wp = lay.wp
    theta = math.radians(45.0)
    w1 = _bulk_plasmon_omega(1, lay)
    k_par = nt.k_par_from_angle(1.0, w1, theta)                   # hold k_par fixed (QNM convention)

    # real-axis absorption peak of the m = 1 mode (same fixed k_par)
    ws = np.linspace(1.0001 * wp, 1.02 * wp, 8000)
    A = _spectrum(lay, ws, pol="p", k_par_m=k_par)
    w_peak = ws[int(np.argmax(A))]

    D = nt.pole_function([lay], pol="p", n_super=1.0, n_sub=1.0, k_par_m=k_par)
    poles = find_poles(D, complex(w1, -0.02 * w1), complex(0.03 * w1, 0.03 * w1),
                       n_grid=40, refine_tol=1e-10)
    decaying = [p for p in poles if p.imag < 0.0 and p.real > 0.0]
    assert decaying, "find_poles found no decaying bulk-plasmon pole"
    # the m = 1 mode is the decaying pole nearest the real-axis absorption peak
    pole = min(decaying, key=lambda p: abs(p.real - w_peak))
    assert pole.imag < 0.0                                        # decaying (exp(-i w t))
    assert abs(pole.real - w_peak) / w_peak < 0.02               # within 2% of the real-axis peak
    Q = pole_q(pole)
    assert math.isfinite(Q) and 0.0 < Q < 1e6                     # finite Q


# ------------------------------------------------------------------------------------------------
# Gate 5: GNOR broadening -- the m = 1 resonance FWHM (= 2|Im(pole)|) grows monotonically with D
# ------------------------------------------------------------------------------------------------
def test_gate5_gnor_broadens_m1():
    theta = math.radians(45.0)
    fwhms = []
    centers = []
    # Realistic GNOR scale: D*omega ~ beta**2 near D ~ 1e-4 m**2/s (Mortensen et al. 2014).
    D_values = (0.0, 5e-5, 1e-4, 2e-4, 4e-4)
    for Dg in D_values:
        lay = _sodium_film(gamma=3.0e12, d=3.0e-9, D=Dg)
        w1 = _bulk_plasmon_omega(1, lay)
        k_par = nt.k_par_from_angle(1.0, w1, theta)
        Dfun = nt.pole_function([lay], pol="p", n_super=1.0, n_sub=1.0, k_par_m=k_par)
        pole = newton_refine(Dfun, complex(w1, -0.01 * w1), tol=1e-11)
        assert pole.imag < 0.0
        fwhms.append(2.0 * abs(pole.imag))                        # mode linewidth (FWHM)
        centers.append(pole.real)
    # FWHM strictly increases with D.
    for a, b in zip(fwhms, fwhms[1:]):
        assert b > a, "GNOR D must broaden the m=1 pole: FWHM not monotonic ({})".format(fwhms)
    # The centre moves by far less than the total broadening (a shift, not a re-tuning).
    center_shift = abs(centers[-1] - centers[0])
    broadening = fwhms[-1] - fwhms[0]
    assert center_shift < broadening


# ------------------------------------------------------------------------------------------------
# Gate 7: ENZ / Berreman blueshift, nonlocal vs local, scaling like 1/d
# ------------------------------------------------------------------------------------------------
def _enz_feature_omega(layer, theta_rad, w_enz):
    """Frequency of the p-pol ENZ/Berreman absorption peak near omega_ENZ (parabola-refined)."""
    ws = np.linspace(0.75 * w_enz, 1.35 * w_enz, 24000)
    A = _spectrum(layer, ws, pol="p", theta_rad=theta_rad)
    i0 = int(np.argmax(A))
    if 0 < i0 < len(ws) - 1:
        y0, y1, y2 = A[i0 - 1], A[i0], A[i0 + 1]
        den = y0 - 2.0 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0.0 else 0.0
        return ws[i0] + off * (ws[1] - ws[0])
    return ws[i0]


def test_gate7_enz_blueshift_one_over_d():
    # TCO-like: eps_inf ~ 4, ENZ in the near-IR; degenerate-semiconductor v_F ~ 1e6 m/s.
    eps_inf, wp, gamma = 4.0, 2.40e15, 1.2e14
    beta = nt.beta_from_vf(1.0e6)
    w_enz = wp / math.sqrt(eps_inf)
    theta = math.radians(60.0)
    shifts = {}
    for d in (10e-9, 50e-9):
        local = nt.HydroLayer(eps_inf, wp, gamma, 1e-3, d)        # beta -> 0
        nonloc = nt.HydroLayer(eps_inf, wp, gamma, beta, d)
        w_loc = _enz_feature_omega(local, theta, w_enz)
        w_nl = _enz_feature_omega(nonloc, theta, w_enz)
        shifts[d] = w_nl - w_loc
    # direction: the nonlocal ENZ/Berreman feature BLUESHIFTS (higher frequency) vs local.
    assert shifts[10e-9] > 0.0 and shifts[50e-9] > 0.0
    # monotonicity / 1/d: a thinner film blueshifts MORE.
    assert shifts[10e-9] > shifts[50e-9]


# ------------------------------------------------------------------------------------------------
# AUDIT V-3: the polarization vocabulary is EXACTLY {'p', 's'} here too (the twin gate lives in
# tests/test_resonance.py::test_admittance_twins_agree_on_off_vocabulary)
# ------------------------------------------------------------------------------------------------
def test_public_entry_points_validate_pol():
    """stack_rt / pole_function already raised; the private _admittance they share with
    optics.resonance did NOT -- it silently returned the S-POL admittance for any unrecognized
    string, the MIRROR of resonance's p-pol fallthrough. All three now reject the same set."""
    layers = [nt.DielectricLayer(4.0 + 0j, 100e-9)]
    w = 2.0 * math.pi * 3.0e8 / 1300e-9
    # ACCEPTANCE UNIFICATION (b): 'TE'/'TM'/'te'/'tm'/'S'/'P' are ACCEPTED now (unconditional
    # aliases of 's'/'p') and moved to the alias leg below; the azimuth-DEPENDENT lab axes, the
    # integer row and non-labels stay off-vocabulary. Map + full gate: core/polarization.py,
    # tests/test_polarization_vocabulary.py.
    for bad in ("x", "y", "X", 0, 1, "", "sp", "tem"):
        with pytest.raises(ValueError, match="pol must be"):
            nt.stack_rt(w, layers, pol=bad)
        with pytest.raises(ValueError, match="pol must be"):
            nt.pole_function(layers, pol=bad)
        with pytest.raises(ValueError, match="pol must be"):
            nt._admittance(4.0 + 0j, 1.2e7 + 0j, bad)
    for good in ("p", "s"):                                   # canonical: bit-identical path
        assert nt.stack_rt(w, layers, pol=good) is not None
        assert callable(nt.pole_function(layers, pol=good))
    for alias, canonical in (("TE", "s"), ("te", "s"), ("S", "s"),
                             ("TM", "p"), ("tm", "p"), ("P", "p")):
        assert nt.stack_rt(w, layers, pol=alias).r == nt.stack_rt(w, layers, pol=canonical).r
        assert (nt._admittance(4.0 + 0j, 1.2e7 + 0j, alias)
                == nt._admittance(4.0 + 0j, 1.2e7 + 0j, canonical))


# ------------------------------------------------------------------------------------------------
# wave-6 hygiene (2026-11-02): audit Q-18 (beta = 0 local limit), Q-19 (one _kz / _admittance /
# k_par_from_angle) and Q-23 (one energy-budget policy)
# ------------------------------------------------------------------------------------------------
def _q18_layer(beta, D=0.0):
    return nt.HydroLayer(eps_inf=1.0, wp=1.37e16, gamma=1.0e14, beta=beta, thickness_m=5.0e-9,
                         D=D)


def test_q18_beta_zero_is_the_local_drude_limit_not_a_zero_division():
    """audit Q-18: HydroLayer(beta=0) promised the local Drude metal but every p-pol call raised
    ZeroDivisionError (kL_squared divides by beta_eff**2 = 0). It is now routed to the local
    Abeles matrix at eps_T -- the exact beta -> 0 limit of the ABC matrix: bit-identical to a
    DielectricLayer(eps_T) stack, and the beta -> 0 sequence converges onto it linearly
    (measured |dr| 1.2e-7 / 1.2e-6 / 1.2e-5 at beta 1e2 / 1e3 / 1e4 m/s)."""
    w = 1.0e16
    spacer = nt.DielectricLayer(2.25, 1.0e-7)
    kw = dict(pol="p", theta_rad=0.5, n_sub=1.5)
    r0 = nt.stack_rt(w, [spacer, _q18_layer(0.0)], **kw)
    eps_t = nt.eps_transverse(w, _q18_layer(0.0))
    loc = nt.stack_rt(w, [spacer, nt.DielectricLayer(eps_t, 5.0e-9)], **kw)
    assert (r0.r, r0.t, r0.D_pole) == (loc.r, loc.t, loc.D_pole)
    d = [abs(nt.stack_rt(w, [spacer, _q18_layer(b)], **kw).r - r0.r) for b in (1e2, 1e3, 1e4)]
    assert d[0] < d[1] < d[2] < 1e-4
    assert d[1] / d[0] == pytest.approx(10.0, rel=0.05)
    assert d[2] / d[1] == pytest.approx(10.0, rel=0.05)
    # the pole function: no csc clearing for a local-limit layer -- it IS the local stack's
    wc = 1.0e16 - 1.0e14j
    f0 = nt.pole_function([_q18_layer(0.0)], k_par_m=1.0e6)
    fl = nt.pole_function([nt.DielectricLayer(nt.eps_transverse(wc, _q18_layer(0.0)), 5.0e-9)],
                          k_par_m=1.0e6)
    assert f0(wc) == fl(wc)
    # GNOR diffusion alone (beta = 0, D > 0) is still a genuine nonlocal layer
    assert nt.stack_rt(w, [spacer, _q18_layer(0.0, D=1.0e-4)], **kw).r != r0.r


def test_q19_one_kz_admittance_kpar():
    """audit Q-19: `_kz`, `_admittance` and `k_par_from_angle` were verbatim copies in
    nonlocal_tmm and resonance (the `_admittance` twins had already diverged once, V-3/V-8). ONE
    implementation now: census (one `def` of each under dynameta/), identity of the objects, and
    the wrapper keeps nonlocal_tmm's `omega` keyword. NOT bit-identical, by a stated tolerance:
    the square root itself is (pinned below against the pre-Q-19 np.sqrt kernel), but kz is now a
    Python complex so eps/kz runs CPython's complex division -- measured 647 of 2400 stack_rt /
    pole_function values moved, max relative 2.4e-14."""
    import pathlib
    import re
    from dynameta.optics import resonance as rz
    assert nt._kz is rz._kz and nt._admittance is rz._admittance
    assert (nt.k_par_from_angle(1.0, omega=2.0e15, theta_rad=0.3)
            == rz.k_par_from_angle(1.0, 2.0e15, 0.3))
    root = pathlib.Path(__file__).resolve().parents[1] / "dynameta"
    defs = {"_kz": 0, "_admittance": 0, "k_par_from_angle": 0}
    for f in root.rglob("*.py"):
        txt = f.read_text(encoding="cp1252", errors="replace")
        for name in defs:
            defs[name] += len(re.findall("^def " + name + "[(]", txt, flags=re.M))
    # k_par_from_angle: the implementation in resonance + the keyword-keeping delegate here
    assert defs == {"_kz": 1, "_admittance": 1, "k_par_from_angle": 2}, defs
    rng = np.random.default_rng(19)
    for _ in range(200):
        eps = complex(rng.uniform(-5, 10), rng.uniform(0, 2))
        k0 = complex(rng.uniform(1e6, 2e7), rng.uniform(-1e5, 0))
        kp = complex(rng.uniform(0, 3e7))
        assert complex(np.sqrt(eps * k0 * k0 - kp * kp + 0j)) == nt._kz(eps, k0, kp)


def test_q23_stack_rt_refuses_a_gain_layer_on_the_real_axis():
    """audit Q-23 (see test_resonance's twin): stack_rt now shares tmm_reference's energy-budget
    rule on the real axis -- a gamma < 0 hydrodynamic film or an Im(eps) < 0 dielectric (a gain
    sign slip) used to come back with a NEGATIVE absorptance A and no signal."""
    gain = nt.HydroLayer(eps_inf=1.0, wp=1.37e16, gamma=-2.0e14, beta=1.07e6,
                         thickness_m=5.0e-9)
    with pytest.raises(ValueError, match="energy budget"):
        nt.stack_rt(1.5e16, [gain], pol="p", theta_rad=0.6)
    with pytest.raises(ValueError, match="energy budget"):
        nt.stack_rt(2.0e15, [nt.DielectricLayer(2.25 - 0.3j, 1.0e-6)])
    res = nt.stack_rt(1.5e16, [gain], pol="p", theta_rad=0.6, check_energy=False)
    assert res.A < 0.0                                     # what used to be returned silently
    nt.stack_rt(1.5e16 - 1.0e14j, [gain], pol="p", k_par_m=1.0e7)   # complex omega: no check


# ================================================================================================
# v0.16: RUNS of hydrodynamic sub-layers of ONE conductor (graded density; walls only at conductor
# boundaries)
# ================================================================================================
# Fixture (the generic gated film): air | 6 nm oxide (n 2.00) | film | 6 nm oxide | glass (n 1.45), p-pol,
# 60 deg, 1200 nm; film eps_inf 4.0, parabolic band m = 0.30 m_e, gamma 1.2e14 1/s, beta = sqrt(3/5) v_F(n0)
# LOCAL in the density, n_bg 3e26 m^-3, 6 nm.
from dynameta.constants import EPS0 as _EPS0, HBAR as _HBAR, M_E as _M_E, Q_E as _Q_E  # noqa: E402
from dynameta.constants import C_LIGHT as _C  # noqa: E402

_MEFF = 0.30 * _M_E
_W1200 = 2.0 * math.pi * _C / 1200e-9
_TH60 = math.radians(60.0)
_OXB = nt.DielectricLayer(2.00 ** 2, 6e-9)
_NGLASS = 1.45


def _film(n, t, cid="film", gamma=1.2e14, beta_scale=1.0):
    wp = math.sqrt(n * _Q_E ** 2 / (_EPS0 * _MEFF))
    vf = _HBAR * (3.0 * math.pi ** 2 * n) ** (1.0 / 3.0) / _MEFF
    return nt.HydroLayer(4.0, wp, gamma, beta_scale * nt.beta_from_vf(vf), t, conductor_id=cid)


def _clad_r(layers, w=_W1200, **kw):
    return nt.stack_rt(w, [_OXB, *layers, _OXB], pol=kw.pop("pol", "p"), n_sub=_NGLASS,
                       theta_rad=_TH60, **kw).r


def _old_cascade_r(layers, w=_W1200, pol="p"):
    """The v0.15.1 code path, transcribed: every layer its own 2x2 matrix (walls everywhere)."""
    k0 = w / _C
    kp = k0 * math.sin(_TH60)
    M = np.eye(2, dtype=np.complex128)
    for lay in [_OXB, *layers, _OXB]:
        M = nt._layer_matrix(w, lay, k0, kp, pol) @ M
    Ys = nt._admittance(1.0 + 0j, nt._kz(1.0 + 0j, k0, kp), pol)
    eb = complex(_NGLASS) ** 2
    Yb = nt._admittance(eb, nt._kz(eb, k0, kp), pol)
    return nt._rt_from_matrix(M, Ys, Yb)[0]


def test_hydro_run_uniform_split_is_the_single_layer_to_roundoff():
    """A uniform 6 nm film cut into N identical sub-layers of one conductor IS the 6 nm film.

    v0.15.1 (a hard wall J_z = 0 between every pair of sub-layers) FAILED this by 4.19e-6 (N = 2)
    and 2.91e-3 (N = 10) in r -- at N = 10 twenty times the whole nonlocal correction (local vs
    nonlocal: 1.35e-4). Those numbers are re-measured below through the explicit opt-in to the old
    physics (allow_walls=True with unlabelled layers). Measured now: |dr| <= 4.7e-14 up to N = 1000.
    """
    one = _clad_r([_film(3e26, 6e-9)])
    assert abs(one - (0.045905 + 0.025745j)) < 1e-6                 # the fixture value
    for N in (2, 10, 100, 1000):
        run = [_film(3e26, 6e-9 / N)] * N
        assert abs(_clad_r(run) - one) < 1e-12, N
    walls2 = _clad_r([_film(3e26, 3e-9, cid=None)] * 2, allow_walls=True)
    walls10 = _clad_r([_film(3e26, 0.6e-9, cid=None)] * 10, allow_walls=True)
    assert abs(walls2 - one) == pytest.approx(4.190e-6, abs=0.015e-6)
    assert abs(walls10 - one) == pytest.approx(2.910e-3, abs=0.01e-3)


def test_hydro_run_single_member_run_is_bit_identical_to_v0151():
    """A run of ONE layer goes through the unchanged closed-form ABC matrix: labelling a lone
    layer changes nothing, bit for bit (consistency + restore check). The general run solver
    applied to one layer agrees with the closed form to round-off (consistency check of two
    different routes: closed-form elimination vs banded global solve)."""
    lay = _film(3e26, 6e-9)
    bare = _film(3e26, 6e-9, cid=None)
    assert _clad_r([lay]) == _clad_r([bare]) == _old_cascade_r([bare])
    w, k0 = _W1200, _W1200 / _C
    for kp in (0.0, 0.5 * k0, 0.866 * k0, 3.0 * k0):
        Mrun = nt._hydro_run_matrix_p(w, [lay], k0, kp)
        Mcf = nt._hydro_char_matrix_p(w, lay, k0, kp)
        assert np.max(np.abs(Mrun - Mcf)) <= 1e-13 * np.max(np.abs(Mcf)), kp


def _bvp_oracle_r(n0_of_z, t_film, *, tol, w=_W1200, regions=None):
    """INDEPENDENT ORACLE: the continuously graded film solved as a two-point boundary-value
    problem with scipy.integrate.solve_bvp. Shares no code with the solver: the equations are
    transcribed here from Maxwell + linearized hydrodynamics with a CONTINUOUS n0(z),

        curl E = i w mu0 H,   curl H = -i w eps0 eps_inf E + J,   div J = i w rho,
        J = sigma(z) (E + grad psi),   psi = -beta(z)**2 rho / (eps0 wp(z)**2),
        sigma = eps0 wp**2/(gamma - i w),  wp**2 = n0 e**2/(eps0 m),  beta**2 = (3/5) v_F(n0)**2,

    integrated as a first-order system in (E_x, H_y, J_z, psi) with no closed-form modes and no
    sub-layers. The two oxide claddings are separate regions of the same BVP (each region mapped onto
    s in [0, 1], states stacked, region couplings as boundary conditions). Unknowns are scaled:
    z in nm, h = Z0 H_y, j = J_z/(w eps0), p = psi/(1 nm). Boundary conditions: incident unit
    tangential E from air, J_z = 0 at the two film faces (conductor boundaries), outgoing in glass.
    regions (optional) = [(n0_of_z, thickness), ...] top first: the ONE conductor as several BVP regions
    (each its own density function; e.g. a density step), coupled by continuity of E_x, H_y, J_z and
    psi -- the interface conditions of one electron gas, written here independently of the solver."""
    from scipy.integrate import solve_bvp
    regions = [(n0_of_z, t_film)] if regions is None else list(regions)
    K = len(regions)
    k0 = w / _C
    kp = k0 * math.sin(_TH60)
    q = kp / k0
    L0 = 1e-9
    eps_h = 2.00 ** 2
    t_h = 6e-9
    gam = 1.2e14
    kzs = np.sqrt(k0 ** 2 - kp ** 2 + 0j)
    ys = k0 / kzs                                   # eps_sup k0 / kz_sup, eps_sup = 1
    eb = _NGLASS ** 2
    kzb = np.sqrt(eb * k0 ** 2 - kp ** 2 + 0j)
    yb = eb * k0 / kzb

    def diel(e, h, eps, scale):
        ez = (-1j * q * h) / (1j * eps)
        return scale * L0 * (1j * k0 * h + 1j * kp * ez), scale * L0 * (1j * k0 * eps * e)

    def film(s, e, h, j, p, n0_of_z, t_film):
        n = n0_of_z(s * t_film)
        wp2 = n * _Q_E ** 2 / (_EPS0 * _MEFF)
        vf = _HBAR * np.cbrt(3.0 * np.pi ** 2 * n) / _MEFF
        b2 = 0.6 * vf ** 2
        st = wp2 / (w * (gam - 1j * w))             # sigma/(w eps0)
        ez = (j - 1j * q * h) / (1j * 4.0)
        jx = st * (e + 1j * kp * L0 * p)            # J_x/(w eps0)
        sc = t_film / L0
        de = sc * L0 * (1j * k0 * h + 1j * kp * ez)
        dh = sc * L0 * (1j * k0 * 4.0 * e - k0 * jx)
        dj = sc * (L0 * (-1j * kp * jx) - 1j * (wp2 / b2) * L0 * L0 * p)
        dp = sc * (j / st - ez)
        return de, dh, dj, dp

    def pack(cs):
        return np.array([part for c in cs for part in (np.real(c), np.imag(c))])

    def fun(s, Y):
        C = Y[0::2] + 1j * Y[1::2]       # states: oxide (e, h) | film regions (e, h, j, p) | oxide (e, h)
        out = list(diel(C[0], C[1], eps_h, t_h / L0))
        for k, (nk, tk) in enumerate(regions):
            out += film(s, *C[2 + 4 * k: 6 + 4 * k], nk, tk)
        out += diel(C[2 + 4 * K], C[3 + 4 * K], eps_h, t_h / L0)
        return pack(out)

    def bc(ya, yb_):
        A = ya[0::2] + 1j * ya[1::2]
        B = yb_[0::2] + 1j * yb_[1::2]
        cond = [A[1] + ys * A[0] - 2.0 * ys,                # air side: unit incident tangential E
                B[0] - A[2], B[1] - A[3],                   # oxide | film: E_x, H_y continuous
                A[4], B[4 * K]]                             # J_z = 0 at both conductor faces
        for k in range(K - 1):                              # inside the conductor: E_x, H_y, J_z, psi
            cond += [B[2 + 4 * k + i] - A[6 + 4 * k + i] for i in range(4)]
        cond += [B[4 * K - 2] - A[2 + 4 * K], B[4 * K - 1] - A[3 + 4 * K],   # film | oxide
                 B[3 + 4 * K] - yb * B[2 + 4 * K]]          # glass side: outgoing only
        return pack(cond)
    s = np.linspace(0.0, 1.0, 401)
    Y0 = np.zeros((8 + 8 * K, s.size))
    Y0[0] = 1.0
    sol = solve_bvp(fun, bc, s, Y0, tol=tol, max_nodes=100000, bc_tol=1e-12)
    assert sol.status == 0, sol.message
    y = sol.sol(0.0)
    return complex(y[0], y[1]) - 1.0


_NB, _NS, _LD, _TF = 3e26, 1.6e27, 1.0e-9, 6e-9


def _accum(z):
    """A smooth accumulation layer: 1.6e27 m^-3 at the top face decaying over 1 nm to 3e26."""
    return _NB + (_NS - _NB) * np.exp(-np.asarray(z) / _LD)


def test_hydro_run_graded_run_converges_to_the_bvp_oracle_at_second_order():
    """The sub-layer staircase of a continuously graded film converges to the INDEPENDENT
    solve_bvp oracle at second order (cell-averaged density, hydro_layers_from_profile).

    MEASURED (oracle r = 0.0511472975392+0.0203706038213i; oracle at tol 1e-8 vs 1e-10: 1.4e-13):
      N =   8  |dr| 3.378e-4
      N =  16  |dr| 8.681e-5   rate 1.96
      N =  32  |dr| 2.181e-5   rate 1.99
      N =  64  |dr| 5.461e-6   rate 2.00
    v0.15.1 FAILED this: with a wall between every sub-layer (the old physics, allow_walls=True)
    the N = 64 staircase is off by 7.01e-3 and does not converge. The REJECTED alternative interface
    condition (continuity of the pressure beta**2 n1 instead of psi) converges to a DIFFERENT
    answer: |dr| plateaus at 1.3e-3 (measured N = 8..256: 2.0e-3, 1.5e-3, 1.3e-3, 1.3e-3, ...).
    The N = 64 bar and the walls bar keep their ratios to the measured values (1.17x and 0.94x).
    """
    r_or = _bvp_oracle_r(_accum, _TF, tol=1e-8)
    r_or_fine = _bvp_oracle_r(_accum, _TF, tol=1e-10)
    assert abs(r_or - r_or_fine) < 1e-11                        # oracle self-convergence
    z = np.linspace(0.0, _TF, 4001)
    nz = _accum(z)
    errs = []
    for N in (8, 16, 32, 64):
        run = nt.hydro_layers_from_profile(z, nz, m_eff_kg=_MEFF, eps_inf=4.0, gamma=1.2e14,
                                           n_layers=N)
        errs.append(abs(_clad_r(run) - r_or))
    rates = [math.log2(a / b) for a, b in zip(errs, errs[1:])]
    assert all(1.9 < r < 2.1 for r in rates), (errs, rates)
    assert errs[-1] < 6.4e-6, errs
    # the rejected pressure-continuity condition: a plateau, not convergence
    run64 = nt.hydro_layers_from_profile(z, nz, m_eff_kg=_MEFF, eps_inf=4.0, gamma=1.2e14,
                                         n_layers=64)
    r_p = _clad_r_with_interface(run64, "pressure")
    assert abs(r_p - r_or) > 100 * errs[-1]
    # the old physics (walls everywhere) on the same staircase
    walls = [nt.HydroLayer(x.eps_inf, x.wp, x.gamma, x.beta, x.thickness_m) for x in run64]
    assert abs(_clad_r(walls, allow_walls=True) - r_or) > 6.6e-3


def _clad_r_with_interface(run, interface, w=_W1200, n_sub=_NGLASS, clad=True):
    """Cascade with the run matrix built under a chosen interface condition (private hook)."""
    k0 = w / _C
    kp = k0 * math.sin(_TH60)
    M = np.eye(2, dtype=np.complex128)
    outer = [_OXB] if clad else []
    for lay in outer:
        M = nt._layer_matrix(w, lay, k0, kp, "p") @ M
    M = nt._hydro_run_matrix_p(w, run, k0, kp, interface=interface) @ M
    for lay in outer:
        M = nt._layer_matrix(w, lay, k0, kp, "p") @ M
    Ys = nt._admittance(1.0 + 0j, nt._kz(1.0 + 0j, k0, kp), "p")
    eb = complex(n_sub) ** 2
    Yb = nt._admittance(eb, nt._kz(eb, k0, kp), "p")
    r, t, _ = nt._rt_from_matrix(M, Ys, Yb)
    return r if clad else (r, t, Ys, Yb)


def _profile_run(N, beta_scale=1.0, gamma=1.2e14, cid="graded"):
    z = np.linspace(0.0, _TF, 4001)
    run = nt.hydro_layers_from_profile(z, _accum(z), m_eff_kg=_MEFF, eps_inf=4.0, gamma=gamma,
                                       n_layers=N, conductor_id=cid)
    if beta_scale != 1.0:
        run = [dataclasses.replace(x, beta=x.beta * beta_scale) for x in run]
    return run


def test_hydro_run_local_limit_of_a_run_is_the_local_graded_cascade():
    """beta -> 0 in every sub-layer: the run converges to the LOCAL graded-layer transfer matrix
    (each sub-layer a plain slab at its eps_T), linearly in beta (the longitudinal boundary layers
    have width ~ beta/wp). MEASURED (16 sub-layers of the accumulation profile) |dr| = 1.50e-4 /
    1.50e-5 / 1.50e-6 at beta x 1e-2 / 1e-3 / 1e-4; exact beta = D = 0 in every member is the local cascade bit for bit."""
    run = _profile_run(16)
    local = [nt.DielectricLayer(nt.eps_transverse(_W1200, x), x.thickness_m) for x in run]
    r_loc = _clad_r(local)
    exact0 = [dataclasses.replace(x, beta=0.0) for x in run]
    assert _clad_r(exact0) == r_loc
    d = [abs(_clad_r(_profile_run(16, beta_scale=s)) - r_loc) for s in (1e-2, 1e-3, 1e-4)]
    assert d[0] > d[1] > d[2]
    assert d[0] / d[1] == pytest.approx(10.0, rel=0.1)
    assert d[1] / d[2] == pytest.approx(10.0, rel=0.1)
    assert d[2] < 1e-5


def test_hydro_run_energy_balance_lossless_and_lossy():
    """Energy is a physics anchor that DISCRIMINATES the interface condition: the conserved flux in
    a hydrodynamic conductor is S_z - (1/2) Re(psi* J_z), continuous only when psi and J_z are.
    A LOSSLESS (gamma = D = 0) graded run below the plasma frequency (2500 nm: every sub-layer
    metallic, longitudinal waves evanescent, no absorption channel) must give A = 0. MEASURED:
    |A| <= 2.0e-15 (psi continuity, this solver; with or without the oxide claddings) vs, for the bare
    run in air|run|glass, A = -3.3e-6 / -5.8e-6 / -6.0e-6 (N = 4 / 16 / 64) under the rejected
    pressure continuity (spurious gain at every interface, not vanishing with refinement). A lossy
    run gives R, T, A >= 0."""
    w = 2.0 * math.pi * _C / 2500e-9
    for N in (4, 16, 64):
        run = _profile_run(N, gamma=0.0)
        res = nt.stack_rt(w, [_OXB, *run, _OXB], pol="p", n_sub=_NGLASS, theta_rad=_TH60)
        assert abs(res.A) < 1e-12, (N, res.A)
        r, t, Ys, Yb = _clad_r_with_interface(run, "pressure", w=w, clad=False)
        # same stack without claddings under the rejected condition, for the gain measurement
        A_p = 1.0 - abs(r) ** 2 - abs(t) ** 2 * Yb.real / Ys.real
        res0 = nt.stack_rt(w, run, pol="p", n_sub=_NGLASS, theta_rad=_TH60)
        assert abs(res0.A) < 1e-12
        assert A_p < -1e-6, (N, A_p)
    lossy = nt.stack_rt(_W1200, [_OXB, *_profile_run(32), _OXB], pol="p", n_sub=_NGLASS,
                        theta_rad=_TH60)
    assert lossy.R > 0.0 and lossy.T > 0.0 and lossy.A > 1e-3


def test_hydro_run_te_is_unaffected_by_runs():
    """s-pol has no longitudinal coupling: a run, the same layers walled (old path, opt-in) and the
    local cascade at eps_T are the SAME numbers, bit for bit."""
    run = _profile_run(16)
    walled = [dataclasses.replace(x, conductor_id=None) for x in run]
    local = [nt.DielectricLayer(nt.eps_transverse(_W1200, x), x.thickness_m) for x in run]
    a = _clad_r(run, pol="s")
    b = _clad_r(walled, pol="s", allow_walls=True)
    c = _clad_r(local, pol="s")
    assert a == b == c == _old_cascade_r(walled, pol="s")


def test_hydro_run_adjacency_rules_and_walls_opt_in_is_v0151_bit_for_bit():
    """Walls only at conductor boundaries. Two adjacent unlabelled HydroLayers (or an unlabelled one
    against a labelled one) are refused; allow_walls=True restores the v0.15.1 physics EXACTLY (A/B
    against the v0.15.1 per-layer cascade transcribed in _old_cascade_r). Different ids = two
    conductors in contact = a wall. A dielectric between two same-id layers keeps a wall on both of
    its sides (each layer is a run of one). A run mixing local-limit and nonlocal members is refused."""
    a1, a2 = _film(3e26, 3e-9, cid=None), _film(6e26, 3e-9, cid=None)
    with pytest.raises(ValueError, match="allow_walls"):
        _clad_r([a1, a2])
    with pytest.raises(ValueError, match="allow_walls"):
        _clad_r([a1, _film(6e26, 3e-9, cid="x")])
    with pytest.raises(ValueError, match="allow_walls"):
        nt.pole_function([a1, a2])
    old = _old_cascade_r([a1, a2])
    assert _clad_r([a1, a2], allow_walls=True) == old
    # different conductors in contact: walls, no error, identical to the opt-in old path
    b1, b2 = _film(3e26, 3e-9, cid="A"), _film(6e26, 3e-9, cid="B")
    assert _clad_r([b1, b2]) == old
    # a dielectric between same-id layers: each side walled
    spacer = nt.DielectricLayer(2.25, 2e-9)
    c1, c2 = _film(3e26, 3e-9, cid="A"), _film(6e26, 3e-9, cid="A")
    assert _clad_r([c1, spacer, c2]) == _old_cascade_r([a1, spacer, a2])
    # and the same pair WITHOUT the spacer is a run, which differs
    assert abs(_clad_r([c1, c2]) - old) > 1e-4
    with pytest.raises(ValueError, match="mixes local-limit"):
        _clad_r([c1, dataclasses.replace(c2, beta=0.0)])
    with pytest.raises(ValueError, match="wp > 0"):
        _clad_r([c1, dataclasses.replace(c2, wp=0.0)])


# r of the step run (top 0.5 nm at n_top, 5.5 nm at 3e26, one conductor) at 1020 nm, the labelled
# resonance case below, for n_top = 3e26 / 6e26 / 9e26 / 1.2e27 (regression values of this solver)
_RUN_R = (0.044450960573 + 0.032580900141j, 0.044551591039 + 0.031606786741j,
          0.044624098379 + 0.030795234935j, 0.044701859519 + 0.030008192087j)
_W1020 = 2.0 * math.pi * _C / 1020e-9
_N_TOP = (3e26, 6e26, 9e26, 1.2e27)


def _step_run_r(n_top, w, walls=False):
    """Top 0.5 nm at n_top over 5.5 nm at 3e26 m^-3: one conductor (a run), or with a wall between."""
    cid = None if walls else "film"
    return _clad_r([_film(n_top, 0.5e-9, cid=cid), _film(3e26, 5.5e-9, cid=cid)], w=w, allow_walls=walls)


def _longitudinal_order(lam_m):
    """Re(k_z,L) d / pi of the uniform 6 nm film at 3e26 m^-3 (60 deg): the order of the longitudinal
    standing wave across the film."""
    w = 2.0 * math.pi * _C / lam_m
    k0 = w / _C
    kp = k0 * math.sin(_TH60)
    kz = np.sqrt(complex(nt.kL_squared(w, _film(3e26, 6e-9))) - kp ** 2)
    return float((kz * 6e-9 / math.pi).real)


def test_hydro_run_restores_the_normal_field_response_of_a_thin_top_sublayer():
    """Top 0.5 nm of the 6 nm film at density n_top, the remaining 5.5 nm at 3e26 m^-3. As ONE conductor
    (a run: E_x, H_y, J_z and psi continuous at the density step) the sub-layer acts through its normal
    field too; with a wall between the two (v0.15.1, J_z = 0 on both sides of the step) only through its
    tangential permittivity.

    (a) Independent route, across the band: the run equals the two-region solve_bvp oracle (the same
    one-conductor interface conditions, transcribed independently) at 950 / 1150 / 1350 nm to 1e-11
    (MEASURED <= 2.3e-14; the oracle's own tol 1e-8 vs 1e-10 change is <= 2.3e-14; the run is exact per
    uniform sub-layer, so the bar is the oracle's convergence with a 100x margin). The walled model
    misses the same oracle by >= 3.5e-5 (MEASURED 6.1e-4 / 3.5e-5 / 1.3e-3).
    (b) The normal-field part of the response to n_top, Delta = |(r_run(1.2e27) - r_run(3e26)) -
    (r_wall(1.2e27) - r_wall(3e26))| (the wall's own effect at n_top = bulk subtracted), is > 4e-6 at
    every 50 nm from 950 to 1350 nm (MEASURED 4.6e-5 / 2.1e-4 / 8.8e-5 / 1.7e-5 / 8.2e-6 / 9.1e-6 /
    1.4e-5 / 2.4e-5 / 4.9e-5; the bar is half the smallest).
    (c) Labelled resonance case: the response of the run to n_top is clearly NON-LINEAR only near a
    longitudinal standing-wave resonance of the film, Re(k_z,L) d = m pi with m odd (m = 4 at 1198 nm
    leaves it linear, ptp/max of the Im steps 0.0035: the even orders couple to the normal field only
    through the film's asymmetry). The test locates m = 5 (1018.9 nm) and there the run's Im r steps are non-linear
    (ptp / max > 0.1, MEASURED 0.19) while the walled ones stay linear (< 0.01, MEASURED 0.0012). At
    1020 nm (within 5 nm of it) the run is pinned to 1e-9 and the walled table to 2e-6 (regression
    values); the n_top = bulk point is the uniform film to round-off."""
    from scipy.optimize import brentq
    # (a) the run is the one-conductor physics at every wavelength, the walls are not
    for lam in (950e-9, 1150e-9, 1350e-9):
        w = 2.0 * math.pi * _C / lam
        regions = [(lambda z: np.full(np.shape(z), 1.2e27), 0.5e-9), (lambda z: np.full(np.shape(z), 3e26), 5.5e-9)]
        r_or = _bvp_oracle_r(None, None, tol=1e-8, w=w, regions=regions)
        assert abs(_step_run_r(1.2e27, w) - r_or) < 1e-11, (lam, abs(_step_run_r(1.2e27, w) - r_or))
        assert abs(_step_run_r(1.2e27, w, walls=True) - r_or) > 3.0e-5, lam
    # (b) the normal-field part of the n_top response is present across the band
    for lam_nm in range(950, 1351, 50):
        w = 2.0 * math.pi * _C / (lam_nm * 1e-9)
        delta = abs((_step_run_r(1.2e27, w) - _step_run_r(3e26, w))
                    - (_step_run_r(1.2e27, w, walls=True) - _step_run_r(3e26, w, walls=True)))
        assert delta > 4e-6, (lam_nm, delta)
    # (c) the labelled resonance case, located physically
    lam5 = brentq(lambda x: _longitudinal_order(x) - 5.0, 950e-9, 1100e-9, xtol=1e-13)
    assert abs(lam5 - 1020e-9) < 5e-9, lam5
    w5 = 2.0 * math.pi * _C / lam5
    di = np.diff([_step_run_r(n, w5).imag for n in _N_TOP])
    dw = np.diff([_step_run_r(n, w5, walls=True).imag for n in _N_TOP])
    assert np.ptp(di) > 0.1 * abs(di).max()                         # run: clearly non-linear
    assert np.ptp(dw) < 0.01 * abs(dw).max()                        # walled: linear
    w = _W1020
    rs = [_step_run_r(n, w) for n in _N_TOP]
    assert abs(rs[0] - _clad_r([_film(3e26, 6e-9)], w=w)) < 1e-12   # n_top = bulk: uniform film
    walls = [_step_run_r(n, w, walls=True) for n in _N_TOP]
    for got, want in zip(walls, (0.043875 + 0.032154j, 0.043961 + 0.031369j,
                                 0.044047 + 0.030584j, 0.044131 + 0.029799j)):
        assert abs(got - want) < 2e-6                               # the walled table
    for got, want in zip(rs, _RUN_R):
        assert abs(got - want) < 1e-9


def test_hydro_run_profile_constructor():
    """hydro_layers_from_profile: exact cell means of the piecewise-linear profile (sheet density
    conserved), local beta(n) = beta_from_vf(v_F(n)), one conductor_id, input validation."""
    z = np.array([0.0, 1e-9, 3e-9, 6e-9])
    n = np.array([2e27, 1e27, 6e26, 3e26])
    run = nt.hydro_layers_from_profile(z, n, m_eff_kg=_MEFF, eps_inf=4.0, gamma=1.2e14)
    assert len(run) == 3 and all(x.conductor_id == "graded" for x in run)
    nk = [_EPS0 * _MEFF * x.wp ** 2 / _Q_E ** 2 for x in run]
    assert nk == pytest.approx([1.5e27, 8e26, 4.5e26], rel=1e-12)
    from dynameta.core.numerics import trapz
    sheet = trapz(n, z)
    for N in (1, 7, 64):
        rr = nt.hydro_layers_from_profile(z, n, m_eff_kg=_MEFF, eps_inf=4.0, gamma=1.2e14,
                                          n_layers=N)
        tot = sum(_EPS0 * _MEFF * x.wp ** 2 / _Q_E ** 2 * x.thickness_m for x in rr)
        assert tot == pytest.approx(sheet, rel=1e-12)
        assert sum(x.thickness_m for x in rr) == pytest.approx(6e-9, rel=1e-12, abs=0)   # metres: no abs floor
    x0 = run[0]
    vf = _HBAR * (3 * math.pi ** 2 * 1.5e27) ** (1 / 3) / _MEFF
    assert x0.beta == pytest.approx(math.sqrt(0.6) * vf, rel=1e-12)
    assert nt.beta_from_density(1.5e27, _MEFF, "thomas_fermi") == pytest.approx(vf / math.sqrt(3),
                                                                                 rel=1e-12)
    for bad in (dict(z_m=z[::-1], n_m3=n), dict(z_m=z, n_m3=-n), dict(z_m=z[:1], n_m3=n[:1]),
                dict(z_m=z, n_m3=n[:3])):
        with pytest.raises(ValueError):
            nt.hydro_layers_from_profile(m_eff_kg=_MEFF, eps_inf=4.0, gamma=1.2e14, **bad)
    with pytest.raises(ValueError):
        nt.hydro_layers_from_profile(z, n, m_eff_kg=_MEFF, eps_inf=4.0, gamma=1.2e14,
                                     conductor_id=None)


def test_hydro_run_kernel_generator_modes_and_constitutive_law():
    """The importable kernel is self-consistent (consistency checks): the closed-form modes are
    eigenvectors of the generator with eigenvalues i*kz, kz_T^2 = eps_T k0^2 - k_par^2 and kz_L^2 =
    k_L^2 - k_par^2; sigma = -i w eps0 (eps_T - eps_inf); psi = c rho with c = -beta_eff^2/(eps0
    wp^2) (GNOR included, complex omega included)."""
    for lay, w in ((_film(3e26, 1e-9), _W1200), (_film(1.6e27, 1e-9), _W1200 * (1 - 0.01j)),
                   (dataclasses.replace(_film(6e26, 1e-9), D=2e-4), _W1200)):
        for kp in (0.0, 0.7 * _W1200 / _C, 2.5 * _W1200 / _C):
            K = nt.hydro_state_generator(w, lay, kp)
            Phi, kz = nt.hydro_modes(w, lay, kp)
            res = K @ Phi - Phi @ np.diag(1j * kz)
            assert np.max(np.abs(res)) < 1e-12 * np.max(np.abs(K)) * np.max(np.abs(Phi))
            k0 = w / _C
            assert kz[0] ** 2 == pytest.approx(nt.eps_transverse(w, lay) * k0 ** 2 - kp ** 2,
                                               rel=1e-12)
            assert kz[2] ** 2 == pytest.approx(nt.kL_squared(w, lay) - kp ** 2, rel=1e-12)
            assert kz[0].imag >= 0.0 and kz[2].imag >= 0.0
        sig = nt.hydro_conductivity(w, lay)
        assert sig == pytest.approx(-1j * w * _EPS0 * (nt.eps_transverse(w, lay) - lay.eps_inf),
                                    rel=1e-12)
        c = nt.hydro_psi_per_charge(w, lay)
        assert c == pytest.approx(-nt.beta_eff_squared(w, lay) / (_EPS0 * lay.wp ** 2), rel=1e-15, abs=0)   # ~1e-8
    assert nt.HYDRO_INTERFACE_FIELDS == ("E_x", "H_y", "J_z", "psi")
    with pytest.raises(ValueError):
        nt.hydro_state_generator(_W1200, _film(3e26, 1e-9, gamma=1.2e14, beta_scale=0.0), 1e6)


def test_hydro_run_run_solver_is_stable_for_thin_and_strongly_evanescent_sublayers():
    """Split identity in the hard regimes (a run of N identical sub-layers vs the closed-form
    single layer, relative to |r|): a Na-like 3 nm film below omega_p (|k_z,L| ~ 9e9 1/m, decay
    0.1 nm) cut into 300 sub-layers of 0.01 nm; a 40 nm film where exp(-|k_z,L| d) underflows
    (|k_z,L| d ~ 360) cut into 40 and into 4000; GNOR; complex omega. A 4x4 transfer-matrix product
    would carry exp(+330) here. MEASURED worst relative |dr| = 1.9e-13 (bar 1e-12)."""
    na = nt.HydroLayer(1.0, 8.65e15, 3e12, nt.beta_from_vf(1.07e6), 3e-9, conductor_id="na")
    cases = [(na, 0.6 * 8.65e15, 300), (na, 1.05 * 8.65e15, 300),
             (dataclasses.replace(na, thickness_m=40e-9), 0.6 * 8.65e15, 40),
             (dataclasses.replace(na, thickness_m=40e-9), 0.6 * 8.65e15, 4000),
             (dataclasses.replace(na, D=2e-4), 1.02 * 8.65e15, 50),
             (na, (0.95 - 0.01j) * 8.65e15, 50)]
    worst = 0.0
    for lay, w, N in cases:
        kp = 0.7 * w.real / _C
        one = nt.stack_rt(w, [lay], pol="p", k_par_m=kp, n_sub=1.3)
        sub = dataclasses.replace(lay, thickness_m=lay.thickness_m / N)
        many = nt.stack_rt(w, [sub] * N, pol="p", k_par_m=kp, n_sub=1.3)
        rel = abs(many.r - one.r) / abs(one.r)
        worst = max(worst, rel)
        assert rel < 1e-12, (N, w, rel)


def test_hydro_run_validity_diagnostic():
    """hydro_validity: u = |k_L| v_F / omega per layer. The near-infrared degenerate-oxide fixture
    is OUTSIDE the high-frequency hydrodynamic validity (u = 1.07 at 3e26, 1200 nm): flagged, and
    stack_rt(check_validity=True) warns. A Na-like film just above omega_p (u ~ 0.1) is inside.
    The degeneracy check flags a depletion-level density at 300 K."""
    from dynameta.warning_types import UnreliableResultWarning
    rep = nt.hydro_validity(_W1200, [_OXB, _film(3e26, 6e-9), _OXB])
    assert not rep.ok and "Landau" in rep.message
    assert math.isnan(rep.u[0]) and rep.u[1] == pytest.approx(1.07, abs=0.01)
    with pytest.warns(UnreliableResultWarning, match="validity"):
        _clad_r([_film(3e26, 6e-9)], check_validity=True)
    na = nt.HydroLayer(1.0, 8.65e15, 3e12, nt.beta_from_vf(1.07e6), 3e-9)
    ok = nt.hydro_validity(1.01 * 8.65e15, [na])
    assert ok.ok and ok.u[0] < 0.2 and ok.message == ""
    nt.stack_rt(1.01 * 8.65e15, [na], pol="p", theta_rad=0.5, check_validity=True)   # no warning
    tf = nt.hydro_validity(1.01 * 8.65e15, [na], convention="thomas_fermi")
    assert not tf.ok                                              # static limit needs u >> 1
    dep = nt.hydro_validity(_W1200, [_film(1e25, 1e-9)], m_eff_kg=_MEFF, T_K=300.0,
                            u_limit=1e9)
    assert not dep.ok and "degenerate" in dep.message and dep.degeneracy[0] < 10.0
    assert nt.hydro_validity(_W1200, [_film(1.6e27, 1e-9)], m_eff_kg=_MEFF, T_K=300.0,
                             u_limit=1e9).ok


def test_hydro_run_pole_function_refuses_nonlocal_runs_only():
    """pole_function: a p-pol nonlocal run of >= 2 sub-layers has no analytic clearing factor ->
    NotImplementedError. A labelled single layer is bit-identical to v0.15.1; s-pol runs and
    local-limit runs work."""
    run = _profile_run(4)
    with pytest.raises(NotImplementedError):
        nt.pole_function(run, k_par_m=1e6)
    f_lab = nt.pole_function([_film(3e26, 6e-9)], k_par_m=1e6)
    f_old = nt.pole_function([_film(3e26, 6e-9, cid=None)], k_par_m=1e6)
    wc = _W1200 * (1 - 0.02j)
    assert f_lab(wc) == f_old(wc)
    assert np.isfinite(nt.pole_function(run, pol="s", k_par_m=1e6)(wc))
    loc = [dataclasses.replace(x, beta=0.0) for x in run]
    assert np.isfinite(nt.pole_function(loc, k_par_m=1e6)(wc))
