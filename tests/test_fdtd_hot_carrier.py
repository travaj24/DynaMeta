"""Gates for the opt-in per-cell hot-carrier two-temperature ADE in the 2D-TE FDTD (roadmap 2.1).

The physics owner is carriers.carrier_heating (the 0-D Alam/De Leon/Boyd chain, Science 352:795
(2016) class); this tier makes it LOCAL inside optics.fdtd_nd.solve_fdtd_2d: an absorbed optical
pump p_abs = J_drude . E heats each Drude cell's electron gas, dropping wp (via the Kane m*(T_e))
and raising gamma (via gamma(T_e)) through precomputed per-material lookup tables. Numbers behind
these gates were pinned interactively; the quantitative oracle is carrier_heating itself.

Gates:
  1  BIT-IDENTITY   -- hot_carrier=None is deterministic AND matches the (untouched) numba path.
  2  UNIFORMITY     -- a uniform Drude film's per-cell T_e(t) matches two_temperature_response driven
                       with the SAME extracted absorbed-power history (the load-bearing oracle).
  3  ZERO-INTENSITY -- a vanishing source leaves T_e == T_ref and outputs == the hot-off run.
  4  PHYSICS DIR    -- below-ENZ transmission INCREASES with pump fluence (wp drops as m* rises);
                       the T_e transient shows the sub-ps rise / few-ps relax asymmetry (rise < relax).
  5  LOCALITY       -- a Drude stripe heats where |E|^2 concentrates, not uniformly.
  6  NUMBA GUARD    -- backend='numba' + hot carrier raises ValueError (numpy reference path only).
Plus the J.E dissipation identity and the HotCarrierParams / off-switch guards.
"""
import numpy as np
import pytest

from dynameta.constants import C_LIGHT, EPS0, M_E, T_REF
from dynameta.optics.fdtd import FDTDLayer
from dynameta.optics.fdtd_nd import HAVE_NUMBA, solve_fdtd_2d
from dynameta.optics.hot_carrier import HotCarrierParams, build_hot_carrier_tables
from dynameta.carriers.carrier_heating import TwoTempParams, two_temperature_response

needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed (CI numpy-only leg)")

# --- shared material: an ITO-class ENZ Drude film. wp places Re(eps)=0 (ENZ) at ~1.18 um (in band);
#     n_m3/m0/alpha set the Kane band average that governs the m*(T_e) -> wp(T_e) shift. ---
EPS_INF, WP, GAM = 4.0, 3.2e15, 1.2e14
N_M3, M0, ALPHA = 1.0e27, 0.30 * M_E, 0.5
GAMMA_E = 200.0                                              # C_e = gamma_e*T_e [J/m^3/K^2] (degenerate gas)
LAMBDA_ENZ = 2.0 * np.pi * C_LIGHT / (WP / np.sqrt(EPS_INF))  # Re(eps)=0 wavelength
BAND = dict(lambda_min_m=1.0e-6, lambda_max_m=1.4e-6)
GRID = dict(resolution=12, n_pad_wave=2.5, npml=8, backend="numpy")


def _ttm(G, C_l=1.0e30, alpha_abs=1.0):
    # C_l -> huge pins the lattice as a FIXED bath (this tier's C_l -> infinity limit of the TTM).
    return TwoTempParams(C_e=lambda Te: GAMMA_E * Te, C_l=C_l, G_e_l=G, alpha_abs=alpha_abs)


def _hc(G=8.0e16, gamma_p=1.0, n_update=1):
    return HotCarrierParams(ttm=_ttm(G), n_m3=N_M3, m0_kg=M0, alpha_per_eV=ALPHA,
                            T_l_K=T_REF, T_e0_K=T_REF, gamma_p=gamma_p, n_update=n_update)


def _cold_layer(thick=120e-9):
    return FDTDLayer(thick, eps_inf=EPS_INF, drude_wp_rad_s=WP, drude_gamma_rad_s=GAM)


def _hot_layer(thick=120e-9, **hckw):
    return FDTDLayer(thick, eps_inf=EPS_INF, drude_wp_rad_s=WP, drude_gamma_rad_s=GAM,
                     hot_carrier=_hc(**hckw))


def _solve(layers, period_x_m=120e-9, **kw):
    opts = dict(GRID); opts.update(kw)                      # let callers override backend/resolution/...
    return solve_fdtd_2d(layers, period_x_m=period_x_m, **BAND, **opts)


# ============================ the J.E dissipation identity ============================

def test_je_dissipation_equals_drude_absorption():
    """p_abs = J.E is the local Drude Joule dissipation: its cycle average must equal the analytic
    Drude absorbed power 0.5 eps0 w Im(eps) |E0|^2. Drive the kernel's EXACT semi-implicit Drude ADE
    (J^{n+1}=aJ J^n + bJ(E^{n+1}+E^n)) with a monochromatic E and compare the midpoint J.E average."""
    w, E0 = 1.5e15, 1.0
    T = 2.0 * np.pi / w
    dt = T / 200.0                                          # 200 steps/optical-period (well resolved)
    aJ = (1.0 - GAM * dt / 2.0) / (1.0 + GAM * dt / 2.0)
    bJ = EPS0 * WP ** 2 * dt / 2.0 / (1.0 + GAM * dt / 2.0)
    J, Eprev, acc, cnt = 0.0, E0, 0.0, 0
    nsteps = 200 * 60
    for n in range(nsteps):
        En = E0 * np.cos(w * (n + 1) * dt)
        Jn = aJ * J + bJ * (En + Eprev)
        pab = 0.5 * (J + Jn) * 0.5 * (Eprev + En)          # the kernel's midpoint p_abs = J.E
        if n > nsteps - 200 * 20:                          # average over the last 20 settled cycles
            acc += pab; cnt += 1
        J, Eprev = Jn, En
    eps = EPS_INF - WP ** 2 / (w ** 2 + 1j * GAM * w)       # exp(-i w t): Im(eps) > 0 = loss
    p_analytic = 0.5 * EPS0 * w * eps.imag * E0 ** 2
    assert p_analytic == pytest.approx(0.5 * EPS0 * WP ** 2 * GAM / (w ** 2 + GAM ** 2) * E0 ** 2, rel=1e-12)
    assert acc / cnt == pytest.approx(p_analytic, rel=3e-3)  # 2nd-order convergent to the Drude absorption


# ============================ gate 1: bit-identity ============================

def test_gate1_hot_off_byte_identical_and_matches_numba():
    r0 = _solve([_cold_layer()])
    r1 = _solve([_cold_layer()])                            # hot_carrier=None everywhere
    for a, b in ((r0.R0, r1.R0), (r0.T0, r1.T0), (r0.R_flux, r1.R_flux), (r0.T_flux, r1.T_flux)):
        assert a.tobytes() == b.tobytes()                  # the None path is untouched -> deterministic


@needs_numba
def test_gate1_numpy_hot_off_matches_untouched_numba():
    lay = [_cold_layer()]
    r_np = _solve(lay)                                      # numpy reference (edited file)
    r_nb = _solve(lay, backend="numba")                    # numba kernel (untouched by 2.1)
    m = r_np.band
    assert np.max(np.abs(r_np.R0[m] - r_nb.R0[m])) < 1e-12  # the additive edit did not perturb R/T
    assert np.max(np.abs(r_np.T0[m] - r_nb.T0[m])) < 1e-12


# ============================ gate 3: zero-intensity ============================

def test_gate3_zero_intensity_leaves_Tref_and_matches_hot_off():
    tiny = 1e-12
    ho = {}
    r_off = _solve([_cold_layer()], source_amp=tiny)
    r_on = _solve([_hot_layer()], source_amp=tiny, hot_out=ho)
    m = r_off.band
    # T_e never leaves T_ref (no pump; T_e0 == T_l so no cooling drive either)
    assert np.max(np.abs(ho["Te_final"][ho["mask"]] - T_REF)) == 0.0
    # outputs equal the hot-off run to rtol 1e-9 (here byte-exact: wp/gamma stay at the cold anchors)
    assert np.allclose(r_on.R0[m], r_off.R0[m], rtol=1e-9, atol=0.0)
    assert np.allclose(r_on.T0[m], r_off.T0[m], rtol=1e-9, atol=0.0)


# ============================ gate 2: the uniformity oracle ============================

def test_gate2_uniform_film_matches_two_temperature_oracle():
    """A uniform Drude film under a plane-wave pulse: the spatially-averaged T_e(t) must reproduce
    carriers.carrier_heating.two_temperature_response driven with the SAME absorbed-power-density
    history extracted from the run. p_abs is the MIDPOINT-in-time J.E, so it corresponds to the
    (n+1/2) dt sample grid -- placing it there is what makes the FDTD forward-Euler integration agree
    with the independent BDF oracle."""
    G = 8.0e16
    ho = {}
    _solve([_hot_layer(G=G, gamma_p=1.0)], source_amp=3e8, hot_out=ho)

    # (i) transverse (x) uniformity of the T_e map is machine-exact (laterally uniform, normal incidence)
    Te2 = ho["Te_final"]
    film_cols = ho["mask"].any(axis=0)
    rows = Te2[:, film_cols]                                # (nx, n_film_z)
    trans = (rows.max(axis=0) - rows.min(axis=0)) / np.maximum(rows.mean(axis=0), 1e-30)
    assert trans.max() < 1e-6

    # (ii) the mask-averaged T_e(t) matches the 0-D two-temperature ODE on the SAME extracted p_abs
    t, pabs, Te_run = ho["t"], ho["p_abs_mean"], ho["Te_mean"]
    dt = float(t[1] - t[0])
    _, Te_oracle, Tl_oracle = two_temperature_response(
        t, lambda tt: float(np.interp(tt, t + 0.5 * dt, pabs)), _ttm(G), T0_K=T_REF)
    peak = max(Te_run.max() - T_REF, 1e-9)
    rel = np.max(np.abs(Te_oracle - Te_run)) / peak
    assert peak > 5.0                                       # a real, resolvable heating transient
    assert Tl_oracle.max() - T_REF < 1e-3                   # the fixed-bath limit held (C_l huge)
    assert rel < 2e-2                                       # a few % (interactively ~1e-3)


# ============================ gate 4: physics direction + asymmetry ============================

def test_gate4_below_enz_transmission_increases_with_fluence():
    """Heating raises <m*(T_e)> -> wp DROPS -> Re(eps) moves toward eps_inf (up through ENZ): below the
    ENZ point a metallic film becomes LESS reflective, so transmission INCREASES. gamma_p=0 isolates
    the plasma shift (the sign the carrier_heating chain itself dictates, per the roadmap)."""
    lay = [_hot_layer(gamma_p=0.0)]
    r_lo = _solve(lay, source_amp=1e3)                     # linear reference (negligible heating)
    ho = {}
    r_hi = _solve(lay, source_amp=8e9, hot_out=ho)         # strong pump -> real wp drop
    lam = C_LIGHT / r_lo.freqs_Hz[r_lo.band]
    below = lam > 1.20e-6                                   # safely below the ENZ point (~1.18 um)
    assert below.sum() > 3 and LAMBDA_ENZ == pytest.approx(1.177e-6, rel=1e-2)
    assert ho["Te_final"][ho["mask"]].max() - T_REF > 500.0  # the film genuinely heated
    # the pinned direction: below-ENZ transmission rises with the pump
    assert r_hi.T0[r_hi.band][below].mean() > r_lo.T0[r_lo.band][below].mean() + 0.05


def test_gate4_transient_rise_faster_than_relaxation():
    """The degenerate-gas C_e(T_e) = gamma_e T_e gives the Alam-class sub-ps rise / few-ps relax
    asymmetry. Gate it qualitatively: the 10-90% rise time is shorter than the peak->1/e relax time."""
    ho = {}
    _solve([_hot_layer(G=8.0e16, gamma_p=1.0)], source_amp=3e8, hot_out=ho)
    t = ho["t"]
    dTe = ho["Te_mean"] - T_REF
    ip = int(np.argmax(dTe)); pk = dTe[ip]
    assert 0 < ip < len(t) - 1 and pk > 5.0
    t_rise = t[np.searchsorted(dTe[:ip + 1], 0.9 * pk)] - t[np.searchsorted(dTe[:ip + 1], 0.1 * pk)]
    ie = ip + int(np.searchsorted(-dTe[ip:], -(pk / np.e)))
    assert ie < len(t)                                     # the film relaxed past 1/e within the window
    t_relax = t[ie] - t[ip]
    assert t_rise < t_relax                                # rise faster than relaxation


# ============================ gate 5: locality discrimination ============================

def test_gate5_stripe_heats_at_field_hotspot():
    """A Drude stripe filling HALF the unit cell (via lateral_wp) under uniform plane-wave illumination
    heats where |E|^2 concentrates -- NOT uniformly. The T_e map must correlate with the |E|^2 map and
    share its argmax; only the stripe columns heat."""
    def latwp(nx, nz, zc, pad, z_struct):
        a = np.zeros((nx, nz))
        film = (zc >= pad) & (zc < pad + z_struct)
        a[: nx // 2, :] = np.where(film, WP, 0.0)          # left half x = Drude stripe, right half dielectric
        return a

    ho = {}
    _solve([_hot_layer(thick=240e-9)], period_x_m=400e-9, nx=16, source_amp=4e8,
           lateral_wp=latwp, hot_out=ho)
    dTe = ho["Te_final"] - T_REF
    E2 = ho["E2_int"]
    heated = ho["mask"] & (dTe > 1e-3)
    cols = np.unique(np.where(heated)[0])
    assert heated.sum() > 4 and cols.max() < 8             # only the left-half (stripe) columns heat
    assert np.corrcoef(dTe[heated], E2[heated])[0, 1] > 0.9  # T_e tracks the field intensity
    aiT = np.unravel_index(np.argmax(np.where(heated, dTe, -1.0)), dTe.shape)
    aiE = np.unravel_index(np.argmax(np.where(heated, E2, -1.0)), E2.shape)
    assert np.hypot(aiT[0] - aiE[0], aiT[1] - aiE[1]) <= 2.0  # co-located hot spot (within a cell or two)
    d = dTe[heated]
    assert (d.max() - d.min()) / d.mean() > 0.1            # a field-structured gradient, not uniform heating


# ============================ gate 6: numba guard ============================

@needs_numba
def test_gate6_numba_backend_refuses_hot_carrier():
    with pytest.raises(ValueError):
        _solve([_hot_layer()], backend="numba")


# ============================ params + off-switch guards ============================

def test_hot_carrier_param_guards():
    ttm = _ttm(8e16)
    with pytest.raises(ValueError):
        HotCarrierParams(ttm=ttm, n_m3=-1.0, m0_kg=M0)             # density must be > 0
    with pytest.raises(ValueError):
        HotCarrierParams(ttm=ttm, n_m3=N_M3, m0_kg=M0, Te_max_K=200.0)  # Te_max <= T_e0
    with pytest.raises(ValueError):
        HotCarrierParams(ttm=ttm, n_m3=N_M3, m0_kg=M0, n_update=0)      # cadence >= 1


def test_table_off_switches_are_exact():
    # alpha_per_eV == 0 -> m* == m0 constant -> wp ratio == 1 everywhere (no plasma shift)
    _, _, wpr, _ = build_hot_carrier_tables(
        HotCarrierParams(ttm=_ttm(8e16), n_m3=N_M3, m0_kg=M0, alpha_per_eV=0.0, gamma_p=1.0))
    assert np.array_equal(wpr, np.ones_like(wpr))
    # gamma_p == 0 -> gamma ratio == 1 everywhere (no damping shift)
    _, _, _, gmr = build_hot_carrier_tables(
        HotCarrierParams(ttm=_ttm(8e16), n_m3=N_M3, m0_kg=M0, alpha_per_eV=ALPHA, gamma_p=0.0))
    assert np.array_equal(gmr, np.ones_like(gmr))
    # the wp/gamma ratios are anchored EXACTLY at the cold T_e0
    Te, U, wpr2, gmr2 = build_hot_carrier_tables(
        HotCarrierParams(ttm=_ttm(8e16), n_m3=N_M3, m0_kg=M0, alpha_per_eV=ALPHA, gamma_p=1.0))
    assert Te[0] == T_REF and U[0] == 0.0
    assert wpr2[0] == pytest.approx(1.0, abs=1e-12) and gmr2[0] == pytest.approx(1.0, abs=1e-12)
