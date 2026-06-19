"""ASE noise, noise figure, and detector beat-noise for the QD-SOA (roadmap SOA Phase 4).

Amplified spontaneous emission sets the amplifier's noise floor -- without it there is no
SFDR/ENOB. The spontaneous-emission factor (population inversion factor) for the EXCITONIC
QD ground state (a single occupation rho per state, charge-neutral, the convention of
optics.soa.qd_gain) is, from the Henry free-carrier form n_sp = f_c(1-f_v)/(f_c-f_v) with
f_c = rho and the valence electron occupation f_v = 1 - rho (hole occ = rho):

    n_sp = rho^2 / (2 rho - 1)              (rho > 1/2; the gain regime)

-- note the rho^2 numerator (the spontaneous emission rate scales as f_c(1-f_v) = rho^2, the
SAME quadratic Pauli factor as the rho^2/tau_sp spontaneous term in the rate equations, NOT
the linear rho the original spec Section 6 wrote). At full inversion rho -> 1, n_sp -> 1 and
the noise figure -> 2 (the 3 dB quantum limit); approaching transparency rho -> 1/2,
n_sp -> infinity.

Forward ASE along z (per polarization, per Hz): dS/dz = Gamma g S + Gamma g n_sp h nu, so the
output spectral density is S_ASE = h nu * integral_0^L Gamma g(z) n_sp(z) G(z->L) dz, which
collapses to the textbook S_ASE = n_sp h nu (G - 1) for a uniform inversion. The z-resolved
form (fed the saturated inversion profile from the traveling-wave engine) captures the
longitudinal gain/inversion variation a single-number formula misses.

Noise figure with the internal-loss + input-coupling degradation (the high-gain ideal 2 n_sp
omits both):

    NF = (1/eta_in) [ 2 n_sp (Gamma g)/(Gamma g - alpha_i) (G-1)/G + 1/G ]

Detector beat-noise variances (direct detection, responsivity R, electrical bandwidth B,
optical bandwidth dnu_o, m_pol ASE polarizations; Olsson JLT 7:1071 1989):

    shot         sigma^2 = 2 q R (P_sig + P_ASE) B + 2 q I_dark B
    signal-spont sigma^2 = 4 R^2 P_sig S_ASE B
    spont-spont  sigma^2 = 2 m_pol R^2 S_ASE^2 (2 dnu_o - B) B

Pure numpy; SI units. exp(-i omega t); h nu energy per photon.
"""

from __future__ import annotations

import numpy as np

from dynameta.constants import HBAR, Q_E

H_PLANCK = 2.0 * np.pi * HBAR

__all__ = ["inversion_factor_nsp", "single_pass_gain", "ase_output_psd", "noise_figure",
           "detector_noise_variances"]


def inversion_factor_nsp(rho_GS):
    """Excitonic spontaneous-emission / population-inversion factor n_sp = rho^2/(2 rho - 1)
    (array-safe). Returns +inf at/below transparency (rho <= 1/2) where there is net
    absorption, not amplification."""
    rho = np.asarray(rho_GS, dtype=np.float64)
    inv = 2.0 * rho - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        nsp = np.where(inv > 1e-12, rho * rho / inv, np.inf)
    return nsp if nsp.ndim else float(nsp)


def single_pass_gain(g_slices, dz_m, Gamma, alpha_i_per_m=0.0):
    """Net single-pass POWER gain G = exp(integral (Gamma g - alpha_i) dz) over the per-slice
    material gain g_slices [1/m]."""
    g = np.asarray(g_slices, dtype=np.float64)
    return float(np.exp(np.sum((Gamma * g - alpha_i_per_m) * dz_m)))


def ase_output_psd(g_slices, rho_GS_slices, dz_m, nu_Hz, Gamma, alpha_i_per_m=0.0,
                   m_pol=2):
    """Forward ASE spectral density at the output [W/Hz], integrating dS/dz = (Gamma g -
    alpha_i) S + Gamma g n_sp h nu over the z-resolved gain + inversion profile (per-slice
    g_slices, rho_GS_slices). m_pol counts ASE polarizations collected (2 for an unpolarized
    receiver). Reduces to n_sp h nu (G - 1) for a uniform inversion."""
    g = np.asarray(g_slices, dtype=np.float64)
    rho = np.asarray(rho_GS_slices, dtype=np.float64)
    nsp = inversion_factor_nsp(rho)
    hnu = H_PLANCK * nu_Hz
    S = 0.0
    for k in range(g.size):
        # exact slice solution of dS/dz = a S + q (constant a, q over the slice):
        # S <- S exp(a dz) + (q/a)(exp(a dz) - 1)  -> S exp + q dz as a -> 0 (no O(dz) bias).
        a = Gamma * g[k] - alpha_i_per_m
        amp = np.exp(a * dz_m)
        q = Gamma * g[k] * nsp[k] * hnu                      # spontaneous source per length
        if not np.isfinite(q):                               # sub-transparency slice (n_sp inf):
            q = 0.0                                           # negligible NET forward ASE -> guard
        emit = q * dz_m if abs(a * dz_m) < 1e-12 else q * (amp - 1.0) / a
        S = S * amp + emit
    return float(m_pol) * S


def noise_figure(G, n_sp, *, Gamma_g_per_m=None, alpha_i_per_m=0.0, eta_in=1.0):
    """Amplifier noise figure (linear, not dB). G the net power gain, n_sp the inversion
    factor. The internal-loss factor (Gamma g)/(Gamma g - alpha_i) and the input-coupling
    efficiency eta_in degrade the ideal high-gain 2 n_sp; with alpha_i = 0 and eta_in = 1,
    NF = 2 n_sp (G-1)/G + 1/G -> 2 n_sp at high gain (3 dB at full inversion, n_sp = 1)."""
    if not (G > 0.0 and 0.0 < eta_in <= 1.0):
        raise ValueError("noise_figure: G > 0 and eta_in in (0, 1]")
    loss = 1.0
    if Gamma_g_per_m is not None and alpha_i_per_m > 0.0:
        if Gamma_g_per_m <= alpha_i_per_m:
            raise ValueError("noise_figure: net gain requires Gamma g > alpha_i")
        loss = Gamma_g_per_m / (Gamma_g_per_m - alpha_i_per_m)
    return float((2.0 * n_sp * loss * (G - 1.0) / G + 1.0 / G) / eta_in)


def detector_noise_variances(P_sig_W, S_ASE_W_Hz, *, R_A_W=1.0, B_Hz=1e10, dnu_opt_Hz=1e12,
                             m_pol=2, I_dark_A=0.0):
    """Photodetector noise variances [A^2] for direct detection of an amplified signal with
    ASE: shot (signal + ASE + dark), signal-spontaneous beat, spontaneous-spontaneous beat.
    P_sig is the detected signal power, S_ASE the one-sided ASE PSD per polarization, dnu_opt
    the optical filter bandwidth, B the electrical bandwidth."""
    P_ASE = float(m_pol) * S_ASE_W_Hz * dnu_opt_Hz
    sh = 2.0 * Q_E * R_A_W * (P_sig_W + P_ASE) * B_Hz + 2.0 * Q_E * I_dark_A * B_Hz
    ssp = 4.0 * R_A_W ** 2 * P_sig_W * S_ASE_W_Hz * B_Hz
    spsp = 2.0 * float(m_pol) * R_A_W ** 2 * S_ASE_W_Hz ** 2 * max(2.0 * dnu_opt_Hz - B_Hz,
                                                                  0.0) * B_Hz
    return {"shot": sh, "sig_spont": ssp, "spont_spont": spsp,
            "total": sh + ssp + spsp, "P_ASE": P_ASE}
