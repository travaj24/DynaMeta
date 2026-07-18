"""Shared optical-amplifier photodetection-noise algebra: the ONE implementation of the
square-law-detector beat-noise variances and the amplifier noise-figure forms, consumed by BOTH
amplifier packages (optics.soa.ase_noise and optics.fiber_amp.detection/noise). The 2026-07-17
audit found the classic duplication failure here: the sp-sp polarization fix (C4-3) was applied
to the soa copy and NOT to the fiber_amp copy (S3-2/S5-1) -- two reviewers independently
rediscovered the same 2x. One implementation makes that class of drift structurally impossible;
the cross-package parity gates in tests/ remain as the contract.

Physics (Olsson, JLT 7:1071 (1989); Desurvire ch.2; Agrawal): a signal at power P_s plus ASE
with per-polarization (one-sided) PSD rho within an optical filter B_o falls on a square-law
photodiode of responsivity R and electrical bandwidth B_e. The photocurrent variances are

    shot      sigma^2 = 2 e (I_sig + I_ase + I_dark) B_e,      I_sig = R P_s, I_ase = R m rho B_o
    sig-sp    sigma^2 = 4 R^2 P_s rho B_e                      (signal beats CO-polarized ASE only)
    sp-sp     sigma^2 = m R^2 rho^2 (2 B_o - B_e) B_e          (each pol beats itself; m pols ADD)

with m the number of independent ASE polarization modes. The sp-sp form is the triangle
autoconvolution of a flat band integrated over +-B_e, PER polarization, times m (audit C4-3,
Monte-Carlo pinned; discriminating limit B_e = B_o, m = 2: thermal-light var/mean^2 = 1/2).

Noise figure (linear): NF = 2 n_sp L (G-1)/G + 1/G with L an internal-loss degradation factor,
equivalently NF = 2 rho_1pol/(h nu G) + 1/G with rho_1pol the forward per-pol ASE PSD at the
signal (n_sp = rho/(h nu (G-1))). Pure numpy; SI units.
"""

from __future__ import annotations

from dynameta.constants import H_PLANCK, Q_E

__all__ = ["beat_noise_variances", "nf_from_nsp", "nf_from_psd"]


def beat_noise_variances(P_sig_W, rho_1pol_W_Hz, *, responsivity_A_W, electrical_bw_Hz,
                         optical_bw_Hz, m_pol=2, I_dark_A=0.0) -> dict:
    """Square-law-detector noise variances [A^2] for a signal + ASE field (module header for the
    formulas). rho_1pol_W_Hz is the PER-POLARIZATION one-sided ASE PSD at the signal wavelength;
    m_pol the number of independent ASE polarization modes reaching the detector. Returns
    {'shot', 'sig_spont', 'spont_spont', 'total', 'P_ASE', 'I_sig', 'I_ase'} -- P_ASE = m rho B_o
    is the detected ASE power, I_* the mean photocurrents."""
    R = float(responsivity_A_W)
    B_e, B_o = float(electrical_bw_Hz), float(optical_bw_Hz)
    P_s, rho, m = float(P_sig_W), float(rho_1pol_W_Hz), float(m_pol)
    P_ase = m * rho * B_o
    I_sig = R * P_s
    I_ase = R * P_ase
    shot = 2.0 * Q_E * (I_sig + I_ase + float(I_dark_A)) * B_e
    sig_sp = 4.0 * R ** 2 * P_s * rho * B_e
    sp_sp = m * R ** 2 * rho ** 2 * max(2.0 * B_o - B_e, 0.0) * B_e
    return {"shot": shot, "sig_spont": sig_sp, "spont_spont": sp_sp,
            "total": shot + sig_sp + sp_sp, "P_ASE": P_ase, "I_sig": I_sig, "I_ase": I_ase}


def nf_from_nsp(G, n_sp, *, loss_factor=1.0, eta_in=1.0) -> float:
    """Amplifier noise figure (LINEAR) from the inversion factor n_sp and net gain G:
        NF = (2 n_sp loss_factor (G-1)/G + 1/G) / eta_in.
    loss_factor is the internal-loss inversion degradation (e.g. Gamma g/(Gamma g - alpha_i) for
    a distributed internal loss); eta_in the input coupling efficiency. Ideal high-gain limit
    2 n_sp (3 dB at full inversion)."""
    if not (G > 0.0 and 0.0 < eta_in <= 1.0):
        raise ValueError("nf_from_nsp: G > 0 and eta_in in (0, 1]")
    return float((2.0 * float(n_sp) * float(loss_factor) * (G - 1.0) / G + 1.0 / G) / eta_in)


def nf_from_psd(G, rho_1pol_W_Hz, nu_Hz) -> float:
    """Amplifier noise figure (LINEAR) from the forward per-polarization ASE PSD at the signal:
        NF = 2 rho / (h nu G) + 1/G.
    The PSD form of nf_from_nsp (substitute rho = n_sp h nu (G-1)); it needs no explicit
    loss_factor because a PSD propagated with the NET gain coefficient already carries the
    loss-degraded inversion."""
    G = float(G)
    if not (G > 0.0):
        raise ValueError("nf_from_psd: G must be > 0")
    return float(2.0 * float(rho_1pol_W_Hz) / (H_PLANCK * float(nu_Hz) * G) + 1.0 / G)
