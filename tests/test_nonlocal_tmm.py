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
    lossy = nt.HydroLayer(1.0, wp, 5e13, beta, 3e-9, D=2e-4)
    for f in np.linspace(0.53, 1.21, 60):
        w = f * wp
        for pol in ("s", "p"):
            R, T, A = nt.rta(w, [lossy], pol=pol, n_super=1.0, n_sub=1.3, theta_rad=theta)
            assert R >= -1e-12 and T >= -1e-12 and A >= -1e-12
            assert R + T + A == pytest.approx(1.0, abs=1e-12)


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
    # ITO-like: eps_inf ~ 3.9, ENZ in the near-IR; degenerate-semiconductor v_F ~ 1e6 m/s.
    eps_inf, wp, gamma = 3.9, 2.40e15, 1.0e14
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
