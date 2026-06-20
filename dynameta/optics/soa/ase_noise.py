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

__all__ = ["inversion_factor_nsp", "inversion_factor_nsp_eh", "single_pass_gain",
           "ase_output_psd", "noise_figure", "detector_noise_variances",
           "spectral_noise_figure", "ase_spectrum_bidirectional", "ase_self_consistent"]


def inversion_factor_nsp(rho_GS):
    """Excitonic spontaneous-emission / population-inversion factor n_sp = rho^2/(2 rho - 1)
    (array-safe). Returns +inf at/below transparency (rho <= 1/2) where there is net
    absorption, not amplification."""
    rho = np.asarray(rho_GS, dtype=np.float64)
    inv = 2.0 * rho - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        nsp = np.where(inv > 1e-12, rho * rho / inv, np.inf)
    return nsp if nsp.ndim else float(nsp)


def inversion_factor_nsp_eh(f_c, f_v):
    """Electron/hole-split inversion factor n_sp = f_c f_v/(f_c + f_v - 1) (Bernard-Duraffourg /
    Henry; f_c electron occupation of the conduction state, f_v hole occupation of the valence
    state). Reduces to the excitonic rho^2/(2 rho - 1) at f_c = f_v = rho. +inf at/below
    transparency (f_c + f_v <= 1)."""
    fc = np.asarray(f_c, dtype=np.float64)
    fv = np.asarray(f_v, dtype=np.float64)
    inv = fc + fv - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        nsp = np.where(inv > 1e-12, fc * fv / inv, np.inf)
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


def spectral_noise_figure(S_f_per_pol, G_nu, nu_grid_Hz, *, eta_in=1.0):
    """Spectral noise figure NF(nu_k) (linear) from the per-pol FORWARD output ASE PSD S_f(nu_k, L)
    and the net single-pass gain G(nu_k):
        n_sp_eff(nu_k) = S_f(nu_k,L) / (h nu_k (G-1)),   NF = (2 n_sp_eff (G-1)/G + 1/G)/eta_in.
    Internal loss needs NO separate factor here: S_f is propagated with the NET coefficient
    (Gamma g - alpha_i), so n_sp_eff already equals n_sp * Gamma_g/(Gamma_g - alpha_i) (the
    loss-degraded inversion). Hence NF == noise_figure(G, n_sp_bare, Gamma_g, alpha_i, eta_in)
    term-for-term, and -> 2 n_sp_eff at high gain. (Multiplying by loss again would double-count
    it.)"""
    Sf = np.asarray(S_f_per_pol, dtype=np.float64)
    G = np.asarray(G_nu, dtype=np.float64)
    nu = np.atleast_1d(np.asarray(nu_grid_Hz, dtype=np.float64))
    hnu = H_PLANCK * nu
    with np.errstate(divide="ignore", invalid="ignore"):
        nsp_eff = np.where(G > 1.0 + 1e-12, Sf / (hnu * (G - 1.0)), 0.0)
    return (2.0 * nsp_eff * (G - 1.0) / G + 1.0 / G) / eta_in


def ase_spectrum_bidirectional(g_slices_nu, gsp_slices_nu, dz_m, nu_grid_Hz, dnu_grid_Hz, Gamma, *,
                               alpha_i_per_m=0.0, m_pol=2, direction="both", return_profile=False):
    """Bidirectional, spectrally-resolved ASE on a GIVEN (frozen) z-resolved gain profile. Inputs
    g_slices_nu, gsp_slices_nu of shape (N_slices, K) are the net modal gain g(nu_k, z_m) and the
    spontaneous-EMISSION gain g_sp(nu_k, z_m) [1/m] (q = Gamma g_sp h nu is the per-pol source;
    pole-free, and = Gamma g n_sp h nu so it reduces to ase_output_psd). Propagates the per-pol ASE
    PSD [W/Hz] with the exact per-slice emit of
        dS/dz = +(Gamma g - alpha_i) S + Gamma g_sp h nu   forward  (BC S_f(.,0)=0, sweep 0..N-1)
        dS/dz = -(Gamma g - alpha_i) S - Gamma g_sp h nu   backward (BC S_b(.,L)=0, sweep N-1..0).
    No gain clamping (ASE-induced saturation lives in ase_self_consistent). m_pol is applied ONCE,
    to the *_out / *_mean accumulators only. Returns a dict: S_f, S_b (per-pol, K) at the output /
    input facet; S_f_out, S_b_out = m_pol * those; S_f_mean, S_b_mean (per-pol z-averaged PSD, for
    the lumped self-consistency); G, Gg (net / gain-only single-pass gain per nu); NF (spectral
    noise figure, forward); nu, dnu. Unlike ase_output_psd it correctly emits at sub-transparency
    slices (g_sp > 0 there); the two agree exactly for an above-transparency profile."""
    g = np.atleast_2d(np.asarray(g_slices_nu, dtype=np.float64))
    gsp = np.atleast_2d(np.asarray(gsp_slices_nu, dtype=np.float64))
    if g.shape[0] == 1 and g.shape[1] != 1 and gsp.shape == g.shape:
        pass                                                 # (1,K) is a single slice, K bands
    nu = np.atleast_1d(np.asarray(nu_grid_Hz, dtype=np.float64))
    dnu = np.atleast_1d(np.asarray(dnu_grid_Hz, dtype=np.float64))
    if g.shape[1] != nu.size:                                # accept (N,) for K=1 column form
        g = g.reshape(-1, nu.size)
        gsp = gsp.reshape(-1, nu.size)
    N = g.shape[0]
    hnu = H_PLANCK * nu                                      # (K,)
    a = Gamma * g - alpha_i_per_m                            # (N,K) net coeff
    q = Gamma * gsp * hnu[None, :]                           # (N,K) per-pol source per length
    amp = np.exp(a * dz_m)
    small = np.abs(a * dz_m) < 1e-12
    emit = np.where(small, q * dz_m, q * (amp - 1.0) / np.where(small, 1.0, a))
    Sf_z = np.zeros((N, nu.size)) if return_profile else None
    Sb_z = np.zeros((N, nu.size)) if return_profile else None
    Sf = np.zeros(nu.size)
    Sf_acc = np.zeros(nu.size)
    for m in range(N):                                       # forward sweep 0..N-1
        Sf = Sf * amp[m] + emit[m]
        Sf_acc += Sf
        if return_profile:
            Sf_z[m] = Sf                                     # forward PSD leaving slice m
    Sb = np.zeros(nu.size)
    Sb_acc = np.zeros(nu.size)
    for m in range(N - 1, -1, -1):                           # backward sweep N-1..0
        Sb = Sb * amp[m] + emit[m]
        Sb_acc += Sb
        if return_profile:
            Sb_z[m] = Sb                                     # backward PSD leaving slice m (toward 0)
    if direction == "forward":
        Sb = np.zeros(nu.size)
        Sb_acc = np.zeros(nu.size)
    elif direction == "backward":
        Sf = np.zeros(nu.size)
        Sf_acc = np.zeros(nu.size)
    G = np.exp(np.sum(a, axis=0) * dz_m)                     # net single-pass gain per nu
    Gg = np.exp(np.sum(Gamma * g, axis=0) * dz_m)            # gain-only (for reference)
    NF = spectral_noise_figure(Sf, G, nu)                    # loss already carried by net-prop S_f
    out = {"S_f": Sf, "S_b": Sb, "S_f_out": float(m_pol) * Sf, "S_b_out": float(m_pol) * Sb,
           "S_f_mean": Sf_acc / N, "S_b_mean": Sb_acc / N, "G": G, "Gg": Gg, "NF": NF,
           "nu": nu, "dnu": dnu}
    if return_profile:
        out["S_f_z"] = Sf_z                                  # (N, K) per-slice forward / backward PSD
        out["S_b_z"] = Sb_z
    return out


def ase_self_consistent(model, I_A, S_conf_signal_m3, nu_s_Hz, nu_grid_Hz, dnu_grid_Hz, L_m, *,
                        n_slices=40, alpha_i_per_m=0.0, m_pol=2, ase_saturation=True,
                        beta=0.5, tol=1e-6, max_iter=60, ase_strength=1.0):
    """Lumped self-consistent ASE: the integrated bidirectional ASE photon density saturates the
    (single-section, uniform-profile) carrier state alongside the signal -- the 'backward ASE
    depletes the inversion' physics. Iterates
      y = model.steady_state(I, S_conf = S_conf_signal + ase_strength*S_ase, nu_s) -> g, g_sp(nu)
      -> ase_spectrum_bidirectional (N uniform slices) -> S_ase (z-averaged confined density,
         S_ase = Gamma/(v_g A_mode) m_pol sum_k (S_f_mean+S_b_mean) dnu_k/(h nu_k)) -> repeat,
    damped (beta), to a NEGATIVE-feedback fixed point (more ASE -> lower inversion -> less gain).
    ase_saturation=False -> ONE pass with no ASE load (the unsaturated spectrum on the signal-only
    carriers; the OFF/reduction path). Returns the ase_spectrum_bidirectional dict plus S_ase
    [m^-3], g_sat (nu grid), g_unsat (nu grid), n_iter, converged. Raises on non-convergence.
    Lumped (single S_conf port) is the minimal well-posed scope; the z-resolved forward/backward+
    carrier BVP is the rigorous future refinement (a coupled marcher in traveling_wave.py)."""
    p = model.p
    nu = np.atleast_1d(np.asarray(nu_grid_Hz, dtype=np.float64))
    dnu = np.atleast_1d(np.asarray(dnu_grid_Hz, dtype=np.float64))
    nz = int(n_slices)
    dz = L_m / nz
    hnu = H_PLANCK * nu
    conv = p.Gamma / (p.v_g_m_s * p.A_mode_m2)               # PSD-power -> confined photon density

    def spectrum(S_ase_load):
        y = model.steady_state(I_A, S_conf_m3=S_conf_signal_m3 + ase_strength * S_ase_load,
                               nu_s_Hz=nu_s_Hz)
        rho = model.rho_GS(y)
        g_nu = model.material_gain_per_m(rho, nu)
        gsp_nu = model.emission_gain_per_m(rho, nu)
        res = ase_spectrum_bidirectional(np.tile(g_nu, (nz, 1)), np.tile(gsp_nu, (nz, 1)), dz,
                                         nu, dnu, p.Gamma, alpha_i_per_m=alpha_i_per_m, m_pol=m_pol)
        res["g_sat"] = g_nu
        return res

    g_unsat = model.material_gain_per_m(model.rho_GS(
        model.steady_state(I_A, S_conf_m3=S_conf_signal_m3, nu_s_Hz=nu_s_Hz)), nu)
    if not ase_saturation:
        res = spectrum(0.0)
        res.update({"S_ase": 0.0, "g_unsat": g_unsat, "n_iter": 0, "converged": True})
        return res
    S_ase = 0.0
    for it in range(max_iter):
        res = spectrum(S_ase)
        S_new = conv * m_pol * float(np.sum((res["S_f_mean"] + res["S_b_mean"]) * dnu / hnu))
        S_d = (1.0 - beta) * S_ase + beta * S_new
        done = abs(S_d - S_ase) <= tol * max(S_d, 1e-300) + 1e-300
        S_ase = S_d
        if done:
            res = spectrum(S_ase)
            res.update({"S_ase": S_ase, "g_unsat": g_unsat, "n_iter": it + 1, "converged": True})
            return res
    raise RuntimeError("ase_self_consistent: ASE fixed point not converged in {} iterates "
                       "(raise max_iter or lower beta)".format(max_iter))


def ase_self_consistent_zresolved(model, I_A, S_conf_signal_m3, nu_s_Hz, nu_grid_Hz, dnu_grid_Hz,
                                  L_m, *, n_slices=40, alpha_i_per_m=0.0, m_pol=2,
                                  ase_saturation=True, beta=0.5, tol=1e-6, max_iter=80,
                                  ase_strength=1.0):
    """Z-RESOLVED self-consistent ASE: refines the lumped ase_self_consistent so EACH slice's gain is
    saturated by its OWN local ASE photon density (not one device-averaged S_ase). The coupled
    fixed point, iterated to convergence:
      g(z, nu), g_sp(z, nu)  from  model.steady_state(I, S_conf = S_signal + ase_strength S_ase(z))
      -> ase_spectrum_bidirectional(return_profile) -> S_f(z, nu), S_b(z, nu)
      -> local confined ASE density  S_ase(z) = Gamma/(v_g A_mode) m_pol sum_nu (S_f+S_b) dnu/(h nu)
      -> repeat (damped beta).
    For a uniform unsaturated device the forward ASE grows toward z=L and the backward toward z=0, so
    S_ase(z) and the gain depression carry a real z-PROFILE (U/dome shaped, deepest where the
    bidirectional ASE flux peaks). SCOPE / MAGNITUDE: in the typical (stiff-QD-gain) regime the ASE
    back-action is WEAK -- the local gain depression is ~1e-4 relative and its spatial spread ~1e-4 --
    so the DEVICE-INTEGRATED output depends essentially only on mean(S_ase(z)) and is unchanged vs the
    lumped ase_self_consistent (which this reduces to whenever S_ase(z) is ~uniform); the refinement is
    the spatial PROFILE itself, not the aggregate output. The refinement is purely LONGITUDINAL: the
    per-slice ASE load enters steady_state as a SCALAR S_conf saturating the carriers through the
    signal-frequency line filter L(nu_s), so spectral saturation stays lumped (an 8 THz ASE comb is
    treated as monochromatic at nu_s; a spectrally-resolved version would weight each band by its own
    L(nu_k)). Reduces to the frozen ase_spectrum_bidirectional on the signal-only gain when
    ase_saturation=False. Returns the ase_spectrum_bidirectional dict (with S_f_z/S_b_z) plus S_ase_z
    (n_slices,), g_sat_z (n_slices, K), g_unsat (K), n_iter, converged. Raises on non-convergence."""
    p = model.p
    nu = np.atleast_1d(np.asarray(nu_grid_Hz, dtype=np.float64))
    dnu = np.atleast_1d(np.asarray(dnu_grid_Hz, dtype=np.float64))
    nz = int(n_slices)
    dz = L_m / nz
    hnu = H_PLANCK * nu
    conv = p.Gamma / (p.v_g_m_s * p.A_mode_m2)

    def gains(S_ase_load):                                   # per-slice gain at the local ASE load
        rows_g, rows_gsp = [], []
        for s in np.atleast_1d(S_ase_load):
            y = model.steady_state(I_A, S_conf_m3=S_conf_signal_m3 + ase_strength * float(s),
                                   nu_s_Hz=nu_s_Hz)
            rho = model.rho_GS(y)
            rows_g.append(model.material_gain_per_m(rho, nu))
            rows_gsp.append(model.emission_gain_per_m(rho, nu))
        return np.atleast_2d(np.array(rows_g)), np.atleast_2d(np.array(rows_gsp))

    g_unsat = model.material_gain_per_m(model.rho_GS(
        model.steady_state(I_A, S_conf_m3=S_conf_signal_m3, nu_s_Hz=nu_s_Hz)), nu)

    def spectrum(S_ase_z):
        g_z, gsp_z = gains(S_ase_z)
        res = ase_spectrum_bidirectional(g_z, gsp_z, dz, nu, dnu, p.Gamma,
                                         alpha_i_per_m=alpha_i_per_m, m_pol=m_pol,
                                         return_profile=True)
        res["g_sat_z"] = g_z
        return res

    if not ase_saturation:
        res = spectrum(np.zeros(nz))
        res.update({"S_ase_z": np.zeros(nz), "g_unsat": g_unsat, "n_iter": 0, "converged": True})
        return res
    S_ase_z = np.zeros(nz)
    for it in range(max_iter):
        res = spectrum(S_ase_z)
        S_new = conv * m_pol * np.sum((res["S_f_z"] + res["S_b_z"]) * dnu / hnu, axis=1)   # (nz,)
        S_d = (1.0 - beta) * S_ase_z + beta * S_new
        done = bool(np.max(np.abs(S_d - S_ase_z)) <= tol * max(float(np.max(S_d)), 1e-300) + 1e-300)
        S_ase_z = S_d
        if done:
            res = spectrum(S_ase_z)
            res.update({"S_ase_z": S_ase_z, "g_unsat": g_unsat, "n_iter": it + 1, "converged": True})
            return res
    raise RuntimeError("ase_self_consistent_zresolved: profile not converged in {} iterates "
                       "(raise max_iter or lower beta)".format(max_iter))
