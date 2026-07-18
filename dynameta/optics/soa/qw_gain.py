"""Connelly bulk / bulk-like quantum-well SOA gain core (roadmap SOA generality phase): the
non-QD path that makes optics.soa cover InGaAsP / InGaAs travelling-wave amplifiers, not only
self-assembled quantum dots. Where qd_gain.py carries group-resolved WL -> ES -> GS occupations,
this module carries a single bulk carrier density N and derives the spectral gain from the
band-to-band joint density of states weighted by the quasi-Fermi occupations (Connelly, IEEE
J. Quantum Electron. 37, 439 (2001), "Wideband semiconductor optical amplifier steady-state
numerical model"). Companion of qd_gain.QDGainModel: the same device-level consumers (the
z-integrated saturation curve, the ASE / noise-figure algebra in optics.amp_noise) drive either
core.

PHYSICS (SI; exp(-i omega t)):

Quasi-Fermi levels. For an undoped active region under injection the electron and hole densities
are equal by charge neutrality, N = P. Each is inverted from the normalized Fermi-Dirac integral
of order 1/2 (Aymerich-Humet fit, reused from carriers.physics_equilibrium; the SAME normalized
convention F_1/2(eta) -> exp(eta) nondegenerate, N = N_c F_1/2(eta)):

    N_c = 2 (m_e   kT / (2 pi hbar^2))^{3/2},  eta_c = F_1/2^{-1}(N / N_c),  F_c = eta_c kT
    N_v = 2 (m_dh  kT / (2 pi hbar^2))^{3/2},  eta_v = F_1/2^{-1}(P / N_v),  F_v = eta_v kT

F_c is the electron quasi-Fermi level measured UP from the conduction-band edge E_c; F_v is the
hole quasi-Fermi level measured DOWN from the valence-band edge E_v (both are "depths" -- positive
and large when that band is degenerately populated). The quasi-Fermi SEPARATION is then
Delta_F = E_c + F_c - (E_v - F_v) = E_g + F_c + F_v.

Vertical (k-conserving) transitions. A photon h nu > E_g connects a conduction state at kinetic
energy E1 above E_c to a valence state at kinetic energy E2 below E_v, with E1 + E2 = h nu - E_g
and E1/E2 = m_dh/m_e (equal k):

    E1 = (h nu - E_g) m_dh / (m_e + m_dh)     [electron kinetic above E_c]
    E2 = (h nu - E_g) m_e  / (m_e + m_dh)     [hole    kinetic below E_v]

The occupation of the conduction state by an electron and of the valence state by an electron:

    f_c = 1 / (1 + exp((E1 - F_c)/kT))     (occupied when E1 < F_c, i.e. degenerate electrons)
    f_v = 1 / (1 + exp((F_v - E2)/kT))     (occupied when E2 < F_v, i.e. degenerate holes empty it)

THE SIGN TRAP (verified by the transparency gate): net gain requires f_c > f_v (inversion). With
the forms above, g = 0 (f_c = f_v) forces E1 - F_c = F_v - E2, i.e. E1 + E2 = F_c + F_v, i.e.
h nu = E_g + F_c + F_v = Delta_F. So at transparency the photon energy EQUALS the quasi-Fermi
separation -- the classic identity. A flipped f_v sign would put the g = 0 crossing somewhere that
does NOT satisfy h nu = Delta_F; tests/test_soa_qw.py asserts the identity to < 2 meV.

Material gain (Connelly eq., intensity gain per metre; no confinement factor -- the consumer
applies Gamma once, modal = Gamma x this):

    g_m(nu, N) = C^2 / (4 sqrt(2) pi^{3/2} n^2 nu^2 tau_rad)
                 * (2 m0 m_e m_dh / (hbar (m_e + m_dh)))^{3/2}
                 * sqrt(nu - E_g/h) * (f_c(nu) - f_v(nu))            [1/m]

for nu > E_g/h (identically 0 below the gap; (f_c - f_v) < 0 above the gap gives ABSORPTION, which
is how the sub-transparency device behaves). tau_rad folds the momentum matrix element |M|^2 into an
effective radiative recombination time (Connelly's approach: recast |M|^2 via the spontaneous
radiative lifetime rather than evaluate Kane's element). It is a NANOSECOND-scale time, NOT the
~0.1 ps intraband dephasing time (tau_intraband_s below): substituting 1e-13 s into this prefactor
overstates the gain by ~1e4 (yielding an unphysical ~1e9 /m) -- the parameter separation is
deliberate and is pinned by the log-gain-fit and device-gain gates.

Spontaneous-emission (ASE source) gain -- Topic 6, the finite-through-transparency source:

    g_e(nu, N) = [same prefactor] * sqrt(nu - E_g/h) * f_c (1 - f_v)   [1/m, emission-only]

with g_m = g_e - g_a and g_a = [pref] sqrt(.) (1 - f_c) f_v the absorption; g_e stays > 0 and smooth
through the g_m = 0 crossing, so the ASE source R_sp ~ Gamma g_e h nu d(nu) never hits the 0*inf of
n_sp*g_m. The inversion factor n_sp = g_e / (g_e - g_a) = f_c (1 - f_v) / (f_c - f_v) feeds the
noise figure (via optics.amp_noise.nf_from_nsp) but is never used as a source.

Carrier rate equation (single bulk reservoir; Topic 7):

    dN/dt = I/(q V_active) - (A N + B N^2 + C N^3)
            - sum_j Gamma v_g g_m(nu_j, N) S_conf,j,     S_conf,j = Gamma P_j/(v_g h nu_j A_xsec)

so the stimulated term is Gamma g_m P/(h nu A_xsec) (the v_g cancels; it is written through S_conf to
mirror qd_gain's confined-photon-density convention). A = SRH, B = bimolecular radiative, C = Auger.

Device gain. A travelling-wave SOA saturates ALONG z because the local carrier density is depleted
by the local (growing) power. dP/dz = (Gamma g_m(N_loc) - alpha_i) P is RK4-integrated over the chip
length, and at each z-slice N_loc is the steady-state root of the carrier equation carrying the
LOCAL photon term (one brentq per slice) -- the honest z-resolved local-N amplifier, not a lumped
single-N approximation. The RK4 z-marcher mirrors the pattern in optics.soa.calibration
(device_saturation_curve); it is NOT imported from there (that module is QD-specific).

Pure numpy/scipy; no FDTD, no DEVSIM, no metasurface seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from dynameta.constants import C_LIGHT, H_PLANCK, HBAR, KB, M_E, Q_E
from dynameta.carriers.physics_equilibrium import invert_F12
from dynameta.optics.amp_noise import nf_from_nsp

__all__ = [
    "BulkGainParams",
    "band_edge_Eg_J",
    "quasi_fermi_levels",
    "quasi_fermi_separation_eV",
    "material_gain_per_m",
    "emission_gain_per_m",
    "n_sp_inversion",
    "gain_peak",
    "transparency_density",
    "carrier_recombination_per_m3_s",
    "carrier_rhs",
    "steady_state_N",
    "small_signal_gain_spectrum",
    "device_gain_dB",
    "device_output_power_W",
    "saturation_output_power_dbm",
    "noise_figure_db",
]


@dataclass(frozen=True)
class BulkGainParams:
    """Frozen material + minimal-geometry parameter set for the Connelly bulk / bulk-QW gain core.

    Band structure (In0.53Ga0.47As lattice-matched / InGaAsP 1550 nm effective; Vurgaftman
    JAP 89, 5815 (2001) Varshni row for In0.53Ga0.47As):
      Eg0_eV               T = 0 K band gap [eV].
      varshni_alpha_eV_K   Varshni alpha [eV/K]:  Eg(T) = Eg0 - alpha T^2 / (T + beta).
      varshni_beta_K       Varshni beta  [K].
      bgs_coeff_eV_m       optional band-gap shrinkage: dEg = -bgs_coeff * N^(1/3) [eV]
                           (N in m^-3; default 0 -> OFF, opt-in many-body renormalization).
      m_e, m_dh            electron / (density-of-states) heavy-hole effective masses in units of m0.
      n_index              background refractive index in the gain prefactor.
      n_group              group index; group velocity v_g = C_LIGHT / n_group.

    Optical matrix element / lifetimes:
      tau_rad_s            EFFECTIVE RADIATIVE recombination time [s] folding |M|^2 into the Connelly
                           prefactor (nanosecond scale; see the module header -- NOT tau_intraband_s).
      tau_intraband_s      intraband scattering / dephasing time [s] (~0.1 ps). Sets the homogeneous
                           linewidth; the Connelly base lineshape is the sqrt() band edge modulated by
                           the Fermi factors, so this is reserved (documented, not folded into the base
                           magnitude) -- kept for an opt-in homogeneous broadening.

    Recombination (Topic 7; InGaAsP 1550 band):
      A_srh_per_s          Shockley-Read-Hall coefficient A [1/s].
      B_rad_m3_s           bimolecular radiative coefficient B [m^3/s].
      C_aug_m6_s           Auger coefficient C [m^6/s].

    Confinement + minimal geometry:
      Gamma                optical confinement factor (modal gain = Gamma * material gain).
      V_active_m3          active volume for the injection term I/(q V_active) [m^3].
      A_xsec_m2            active cross-section for the photon-density / stimulated term [m^2].
                           ENERGY CONSISTENCY pins A_xsec = V_active / L_device: dP/dz = Gamma g P
                           must equal (carrier stimulated loss) * A_xsec * h nu, so the default pair
                           (8.5e-16 m^3, 1.4167e-12 m^2) is consistent at the reference L = 600 um.
      T_K                  lattice temperature [K].

    Geometry-sizing note: this steady-state core does NOT self-clamp the carrier density with ASE.
    The default V_active places the 130 mA / 600 um reference operating point at N0 ~ 1.4 x
    transparency, where the UNCLAMPED small-signal chip gain is ~27 dB (the datasheet band). A
    physically smaller active cross-section at 130 mA would drive N0 far above transparency and, in
    the absence of ASE gain-clamping, over-predict the gain -- so the effective volume is the honest
    knob here (equivalently an injection efficiency eta_i < 1); a full ASE-clamped device solve is
    the QDGainModel-style extension.
    """

    Eg0_eV: float = 0.816
    varshni_alpha_eV_K: float = 2.9e-4
    varshni_beta_K: float = 193.0
    bgs_coeff_eV_m: float = 0.0
    m_e: float = 0.045
    m_dh: float = 0.46
    n_index: float = 3.5
    n_group: float = 3.7
    tau_rad_s: float = 1.0e-9
    tau_intraband_s: float = 1.0e-13
    A_srh_per_s: float = 2.5e8
    B_rad_m3_s: float = 1.0e-16
    C_aug_m6_s: float = 5.0e-41
    Gamma: float = 0.3
    V_active_m3: float = 8.5e-16
    A_xsec_m2: float = 1.4167e-12
    T_K: float = 298.0

    @property
    def v_g_m_s(self) -> float:
        """Group velocity C_LIGHT / n_group [m/s]."""
        return C_LIGHT / self.n_group


def _band_dos(m_rel: float, T_K: float) -> float:
    """Effective band density of states N_c or N_v [m^-3] in the normalized-F_1/2 convention
    (N = N_dos * F_1/2(eta)); m_rel is the DOS mass in units of m0."""
    return 2.0 * (m_rel * M_E * KB * T_K / (2.0 * np.pi * HBAR ** 2)) ** 1.5


def band_edge_Eg_J(N_m3: float, params: BulkGainParams) -> float:
    """Temperature- and (optionally) density-dependent band gap [J]:
    E_g = Eg0 - alpha T^2/(T + beta) - bgs_coeff N^(1/3) (Varshni + opt-in shrinkage)."""
    p = params
    eg_eV = p.Eg0_eV - p.varshni_alpha_eV_K * p.T_K ** 2 / (p.T_K + p.varshni_beta_K)
    if p.bgs_coeff_eV_m != 0.0 and N_m3 > 0.0:
        eg_eV -= p.bgs_coeff_eV_m * N_m3 ** (1.0 / 3.0)
    return eg_eV * Q_E


def quasi_fermi_levels(N_m3: float, T_K: float, params: BulkGainParams):
    """(F_c, F_v) [J], the electron and hole quasi-Fermi levels measured from their band edges
    (F_c up from E_c, F_v down from E_v), for carrier density N (charge neutrality N = P).
    Inverts the normalized F_1/2 (Aymerich-Humet) reused from carriers.physics_equilibrium."""
    kT = KB * T_K
    N_c = _band_dos(params.m_e, T_K)
    N_v = _band_dos(params.m_dh, T_K)
    eta_c = invert_F12(N_m3 / N_c)
    eta_v = invert_F12(N_m3 / N_v)
    return eta_c * kT, eta_v * kT


def quasi_fermi_separation_eV(N_m3: float, params: BulkGainParams) -> float:
    """Quasi-Fermi separation Delta_F = E_g + F_c + F_v [eV] at density N (transparency photon
    energy)."""
    F_c, F_v = quasi_fermi_levels(N_m3, params.T_K, params)
    return (band_edge_Eg_J(N_m3, params) + F_c + F_v) / Q_E


def _prefactor(nu, params: BulkGainParams):
    """Connelly gain prefactor C^2/(4 sqrt2 pi^1.5 n^2 nu^2 tau_rad) * (2 m0 m_e m_dh/(hbar(m_e+m_dh)))^1.5
    [m^-1 Hz^0.5] (still to be multiplied by sqrt(nu - Eg/h) and (f_c - f_v))."""
    p = params
    mass_term = (2.0 * M_E * p.m_e * p.m_dh / (HBAR * (p.m_e + p.m_dh))) ** 1.5
    return (C_LIGHT ** 2 / (4.0 * np.sqrt(2.0) * np.pi ** 1.5 * p.n_index ** 2
                            * nu ** 2 * p.tau_rad_s)) * mass_term


def _occupations(nu, N_m3: float, params: BulkGainParams):
    """(f_c, f_v, sqrt(nu - Eg/h), mask) at frequencies nu for density N. f_c / f_v are the
    conduction / valence electron occupations of the vertical transition; the sqrt term is the
    joint-DOS band edge (0 below the gap); mask marks nu > Eg/h."""
    p = params
    T = p.T_K
    kT = KB * T
    Eg_J = band_edge_Eg_J(N_m3, params)
    F_c, F_v = quasi_fermi_levels(N_m3, T, params)
    nu = np.asarray(nu, dtype=np.float64)
    E_ph = H_PLANCK * nu
    dE = E_ph - Eg_J
    mask = dE > 0.0
    dE_pos = np.where(mask, dE, 0.0)
    E1 = dE_pos * p.m_dh / (p.m_e + p.m_dh)
    E2 = dE_pos * p.m_e / (p.m_e + p.m_dh)
    f_c = expit((F_c - E1) / kT)
    f_v = expit((E2 - F_v) / kT)
    sqrt_edge = np.sqrt(np.where(mask, nu - Eg_J / H_PLANCK, 0.0))
    return f_c, f_v, sqrt_edge, mask


def material_gain_per_m(nu_Hz, N_m3: float, params: BulkGainParams):
    """Connelly MATERIAL intensity gain g_m(nu, N) [1/m] (no confinement). Positive = gain,
    negative = absorption; identically 0 below the band gap. Returns a float for scalar nu, else
    an array matching nu_Hz."""
    nu = np.asarray(nu_Hz, dtype=np.float64)
    f_c, f_v, sqrt_edge, mask = _occupations(nu, N_m3, params)
    g = _prefactor(nu, params) * sqrt_edge * (f_c - f_v)
    g = np.where(mask, g, 0.0)
    return float(g) if np.ndim(nu_Hz) == 0 else g


def emission_gain_per_m(nu_Hz, N_m3: float, params: BulkGainParams):
    """Emission-only (spontaneous / ASE-source) gain g_e(nu, N) ~ f_c (1 - f_v) [1/m], SAME
    prefactor. Finite and > 0 through transparency (never n_sp * g); g_m = g_e - g_a."""
    nu = np.asarray(nu_Hz, dtype=np.float64)
    f_c, f_v, sqrt_edge, mask = _occupations(nu, N_m3, params)
    g_e = _prefactor(nu, params) * sqrt_edge * f_c * (1.0 - f_v)
    g_e = np.where(mask, g_e, 0.0)
    return float(g_e) if np.ndim(nu_Hz) == 0 else g_e


def n_sp_inversion(nu_Hz: float, N_m3: float, params: BulkGainParams) -> float:
    """Population-inversion factor n_sp = g_e/(g_e - g_a) = f_c(1 - f_v)/(f_c - f_v) at (nu, N).
    -> 1 at full inversion, -> +inf approaching transparency from above (checked, never used as a
    source term)."""
    f_c, f_v, _sq, _m = _occupations(np.asarray(nu_Hz, dtype=np.float64), N_m3, params)
    f_c = float(f_c)
    f_v = float(f_v)
    denom = f_c - f_v
    if denom == 0.0:
        return float("inf")
    return f_c * (1.0 - f_v) / denom


def gain_peak(N_m3: float, params: BulkGainParams, *, n_grid: int = 4001,
              span_eV: float = 0.25):
    """(nu_peak_Hz, g_peak_per_m) of the material gain at density N, searched from just above the
    band edge over span_eV. Returns the argmax on a dense frequency grid (adequate for the smooth
    Connelly lineshape)."""
    Eg_J = band_edge_Eg_J(N_m3, params)
    nu_lo = Eg_J / H_PLANCK
    nu_hi = (Eg_J + span_eV * Q_E) / H_PLANCK
    nu = np.linspace(nu_lo, nu_hi, int(n_grid))
    g = material_gain_per_m(nu, N_m3, params)
    k = int(np.argmax(g))
    return float(nu[k]), float(g[k])


def transparency_density(nu_Hz: float, params: BulkGainParams,
                         N_lo: float = 1.0e22, N_hi: float = 1.0e25) -> float:
    """Carrier density N_tr [m^-3] at which the material gain at nu_Hz crosses zero (transparency),
    by brentq. g(nu, N) is monotone increasing in N at fixed nu (more inversion), so the root is
    unique in [N_lo, N_hi]."""
    f = lambda N: material_gain_per_m(nu_Hz, N, params)
    return float(brentq(f, N_lo, N_hi, xtol=1.0e19, rtol=1.0e-10))


def carrier_recombination_per_m3_s(N_m3: float, params: BulkGainParams) -> float:
    """Non-stimulated recombination A N + B N^2 + C N^3 [m^-3 s^-1] (SRH + radiative + Auger)."""
    p = params
    return p.A_srh_per_s * N_m3 + p.B_rad_m3_s * N_m3 ** 2 + p.C_aug_m6_s * N_m3 ** 3


def _stimulated_per_m3_s(N_m3: float, P_W, nu_Hz, params: BulkGainParams) -> float:
    """Net stimulated recombination sum_j Gamma g_m(nu_j, N) P_j/(h nu_j A_xsec) [m^-3 s^-1]
    (v_g cancels via S_conf = Gamma P/(v_g h nu A_xsec)). g_m < 0 (absorption) contributes a
    NEGATIVE term -> a sub-transparency signal generates carriers."""
    p = params
    P = np.atleast_1d(np.asarray(P_W, dtype=np.float64))
    nu = np.atleast_1d(np.asarray(nu_Hz, dtype=np.float64))
    g = material_gain_per_m(nu, N_m3, params)
    return float(np.sum(p.Gamma * g * P / (H_PLANCK * nu * p.A_xsec_m2)))


def carrier_rhs(N_m3: float, I_A: float, P_W, nu_Hz, params: BulkGainParams) -> float:
    """dN/dt [m^-3 s^-1] = I/(q V_active) - (A N + B N^2 + C N^3) - stimulated(N, P, nu).
    P_W / nu_Hz are the local optical powers and their frequencies (scalars or matching arrays;
    P_W = 0 gives the small-signal reservoir)."""
    p = params
    pump = I_A / (Q_E * p.V_active_m3)
    recomb = carrier_recombination_per_m3_s(N_m3, params)
    stim = 0.0 if np.all(np.asarray(P_W) == 0.0) else _stimulated_per_m3_s(N_m3, P_W, nu_Hz, params)
    return pump - recomb - stim


def steady_state_N(I_A: float, params: BulkGainParams, P_W=0.0, nu_Hz=None,
                   N_lo: float = 1.0e18, N_hi: float = 1.0e26) -> float:
    """Steady-state carrier density N [m^-3] solving carrier_rhs = 0 by brentq. With P_W = 0 this is
    the unsaturated reservoir I/(qV) = A N + B N^2 + C N^3; with a local power it is the saturated
    local density (the per-z-slice solve of the travelling-wave amplifier)."""
    if nu_Hz is None:
        nu_Hz = C_LIGHT / 1.55e-6
    f = lambda N: carrier_rhs(N, I_A, P_W, nu_Hz, params)
    return float(brentq(f, N_lo, N_hi, xtol=1.0e16, rtol=1.0e-10))


def small_signal_gain_spectrum(I_A: float, nu_grid_Hz, params: BulkGainParams):
    """Unsaturated MATERIAL gain spectrum g_m(nu) [1/m] at the reservoir density N0(I) (P -> 0).
    Multiply by Gamma for modal gain; net device dB = (10/ln10)(Gamma g - alpha_i) L."""
    N0 = steady_state_N(I_A, params, P_W=0.0)
    return material_gain_per_m(np.asarray(nu_grid_Hz, dtype=np.float64), N0, params)


def _local_saturated_gain(I_A: float, nu_Hz: float, params: BulkGainParams, P_grid):
    """Local steady-state MATERIAL gain g_m(P) [1/m] on a power grid: at each P, solve the carrier
    density carrying that local photon term, then evaluate the gain. This is the g(P) the z-marcher
    interpolates (mirrors optics.soa.calibration.device_saturation_curve's saturation_curve step)."""
    g = np.empty_like(P_grid)
    for i, P in enumerate(P_grid):
        N = steady_state_N(I_A, params, P_W=P, nu_Hz=nu_Hz)
        g[i] = material_gain_per_m(nu_Hz, N, params)
    return g


def device_output_power_W(I_A: float, nu_Hz: float, L_m: float, alpha_i_per_m: float,
                          P_in_W, params: BulkGainParams, nz: int = 200):
    """Absolute CW output power P_out(P_in) [W] by z-resolved local-N integration:
    dP/dz = (Gamma g_m(N_loc(P)) - alpha_i) P, RK4 over L. N_loc(P) is the local saturated density
    (one brentq per power sample; precomputed on a grid and interpolated for the RK4, so each step
    is elementwise). Returns (P_in_W, P_out_W), vectorized across the P_in array."""
    p = params
    P_in = np.atleast_1d(np.asarray(P_in_W, dtype=np.float64))
    P_grid = np.logspace(np.log10(P_in.min()) - 1.0,
                         np.log10(P_in.max() * 1.0e5) + 1.0, 400)
    g_loc = _local_saturated_gain(I_A, nu_Hz, params, P_grid)
    g_of_P = lambda P: np.interp(P, P_grid, g_loc)
    dz = L_m / int(nz)
    gam = p.Gamma
    P = P_in.copy()
    for _ in range(int(nz)):
        k1 = (gam * g_of_P(P) - alpha_i_per_m) * P
        k2 = (gam * g_of_P(P + 0.5 * dz * k1) - alpha_i_per_m) * (P + 0.5 * dz * k1)
        k3 = (gam * g_of_P(P + 0.5 * dz * k2) - alpha_i_per_m) * (P + 0.5 * dz * k2)
        k4 = (gam * g_of_P(P + dz * k3) - alpha_i_per_m) * (P + dz * k3)
        P = P + dz / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return P_in, P


def device_gain_dB(I_A: float, nu_Hz: float, L_m: float, alpha_i_per_m: float,
                   P_in_W=1.0e-9, params: BulkGainParams = None, nz: int = 200):
    """Net single-pass device gain [dB] = 10 log10(P_out/P_in) at input power P_in_W (default 1 nW
    -> the small-signal chip gain). For an array of P_in returns an array of gains (the saturation
    curve in dB)."""
    if params is None:
        raise ValueError("device_gain_dB: params (BulkGainParams) is required")
    P_in, P_out = device_output_power_W(I_A, nu_Hz, L_m, alpha_i_per_m, P_in_W, params, nz=nz)
    G_dB = 10.0 * np.log10(P_out / P_in)
    return float(G_dB[0]) if np.ndim(P_in_W) == 0 else G_dB


def saturation_output_power_dbm(I_A: float, nu_Hz: float, L_m: float, alpha_i_per_m: float,
                                params: BulkGainParams, nz: int = 200):
    """Output-referred -3 dB saturation power P_sat,out [dBm]: the output power where the device
    gain has dropped 3 dB below its small-signal value, read off the steady-state saturation curve
    (mirrors optics.soa.calibration._psat_out_dBm). Returns (Psat_out_dBm, G0_dB)."""
    P_in = np.logspace(-9.0, 0.5, 48)
    P_in, P_out = device_output_power_W(I_A, nu_Hz, L_m, alpha_i_per_m, P_in, params, nz=nz)
    G_dB = 10.0 * np.log10(P_out / P_in)
    G0 = float(G_dB[0])
    target = G0 - 3.0
    if np.nanmin(G_dB) > target:
        return float("nan"), G0
    log_pout = np.log10(P_out)
    log_pout_sat = float(np.interp(target, G_dB[::-1], log_pout[::-1]))
    return float(10.0 * np.log10((10.0 ** log_pout_sat) / 1.0e-3)), G0


def noise_figure_db(I_A: float, nu_Hz: float, L_m: float, alpha_i_per_m: float,
                    params: BulkGainParams, nz: int = 200) -> float:
    """Amplifier noise figure [dB] at the small-signal operating point. n_sp = f_c(1-f_v)/(f_c-f_v)
    is evaluated at the reservoir density N0(I) and the gain-peak frequency; the internal-loss
    inversion degradation loss_factor = Gamma g_pk/(Gamma g_pk - alpha_i) and the net linear gain G
    feed the shared optics.amp_noise.nf_from_nsp (NF = 2 n_sp loss_factor (G-1)/G + 1/G)."""
    p = params
    N0 = steady_state_N(I_A, params, P_W=0.0)
    nu_pk, g_pk = gain_peak(N0, params)
    modal = p.Gamma * g_pk
    if modal <= alpha_i_per_m:
        raise ValueError("noise_figure_db: modal gain Gamma*g_pk <= alpha_i (no net gain at peak)")
    n_sp = n_sp_inversion(nu_pk, N0, params)
    loss_factor = modal / (modal - alpha_i_per_m)
    G_dB = device_gain_dB(I_A, nu_pk, L_m, alpha_i_per_m, P_in_W=1.0e-9, params=params, nz=nz)
    G_lin = 10.0 ** (G_dB / 10.0)
    nf_lin = nf_from_nsp(G_lin, n_sp, loss_factor=loss_factor)
    return float(10.0 * np.log10(nf_lin))
