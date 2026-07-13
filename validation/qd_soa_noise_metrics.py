"""QD-SOA ASE noise, noise figure, and analog SFDR/ENOB (roadmap SOA Phase 4) vs analytic /
known-limit oracles. This closes the chain to the figures of merit the incoherent OVMM gain
leg is judged by.

GATE A (n_sp + noise-figure limits): the excitonic inversion factor n_sp = rho^2/(2 rho - 1)
        -> 1 at full inversion (rho = 1) and -> infinity at transparency (rho = 1/2); the
        noise figure -> 2 n_sp at high gain, i.e. the 3.01 dB quantum limit at full inversion,
        and DEGRADES with internal loss / input-coupling -- the corrected NF the ideal form
        omits.
GATE B (ASE power vs the analytic limit): the z-resolved ASE integrator reduces to the
        textbook S_ASE = n_sp h nu (G - 1) for a uniform inversion profile (< 1%), and scales
        with (G - 1).
GATE C (detector beat-noise regime): at high gain the output noise is signal-spontaneous-beat
        dominated (sigma^2_ssp >> shot), with spont-spont present -- the ASE-beat noise floor
        that sets the SNR, matching the closed forms.
GATE D (SFDR/ENOB optimal drive -- "window, not a wall"): combining the gain-compression
        distortion (transfer-curve curvature) with the ASE beat-noise floor, the SNDR vs
        drive power has an INTERIOR maximum -- noise-limited below, distortion-limited above
        -- so there is an optimal analog operating point and a peak ENOB.

Run: python -m validation.qd_soa_noise_metrics
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR
from dynameta.optics.soa.ase_noise import (ase_output_psd, detector_noise_variances,
                                           inversion_factor_nsp, noise_figure)
from dynameta.optics.soa.metrics import enob, sndr_vs_drive
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams

from dynameta.constants import H_PLANCK   # single source (audit 6.3)


def main():
    print("[nm] === QD-SOA ASE noise + noise figure + SFDR/ENOB vs analytic oracles ===",
          flush=True)
    ok = True
    nu0 = 1.934e14

    # ---- GATE A: n_sp + NF limits ----
    nsp_full = inversion_factor_nsp(1.0)                     # full inversion
    nsp_tr = inversion_factor_nsp(0.5001)                    # near transparency
    NF_full_highG = noise_figure(1.0e3, nsp_full)            # 3 dB quantum limit
    NF_noloss = noise_figure(100.0, 1.5)
    NF_loss = noise_figure(100.0, 1.5, Gamma_g_per_m=300.0, alpha_i_per_m=60.0)
    g_a = bool(abs(nsp_full - 1.0) < 1e-12 and nsp_tr > 1e3
               and abs(10 * np.log10(NF_full_highG) - 3.01) < 0.05 and NF_loss > NF_noloss)
    ok = ok and g_a
    print("[nm] GATE A: n_sp(full)={:.3f}, n_sp(transp)>1e3={}, NF(full,highG)={:.2f} dB "
          "(3.01 limit), NF degrades w/ loss ({:.2f}>{:.2f}) -> {}".format(
              nsp_full, nsp_tr > 1e3, 10 * np.log10(NF_full_highG), NF_loss, NF_noloss,
              "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: z-resolved ASE reduces to n_sp h nu (G-1) for uniform inversion ----
    rho, Nz, L, Gamma = 0.95, 200, 1.0e-3, 0.06
    dz = L / Nz
    g_mat = 8000.0                                           # uniform material gain [1/m]
    g_slices = np.full(Nz, g_mat)
    rho_slices = np.full(Nz, rho)
    S_dist = ase_output_psd(g_slices, rho_slices, dz, nu0, Gamma, m_pol=1)
    G = float(np.exp(Gamma * g_mat * L))
    S_analytic = inversion_factor_nsp(rho) * H_PLANCK * nu0 * (G - 1.0)
    relB = abs(S_dist - S_analytic) / S_analytic
    # (G-1) scaling: half the length -> the ASE tracks (G'-1)
    S_half = ase_output_psd(g_slices[:Nz // 2], rho_slices[:Nz // 2], dz, nu0, Gamma, m_pol=1)
    Gh = float(np.exp(Gamma * g_mat * (L / 2)))
    relB2 = abs(S_half - inversion_factor_nsp(rho) * H_PLANCK * nu0 * (Gh - 1.0)) / \
        (inversion_factor_nsp(rho) * H_PLANCK * nu0 * (Gh - 1.0))
    g_b = bool(relB < 1e-2 and relB2 < 1e-2)
    ok = ok and g_b
    print("[nm] GATE B: z-resolved ASE == n_sp h nu (G-1) for uniform inversion "
          "(rel {:.2e}, half-length {:.2e}; G={:.0f}) -> {}".format(
              relB, relB2, G, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: detector beat-noise regime (ASE-beat-dominated at HIGH gain) ----
    # the sig-spont beat dominates shot only at high gain (large S_ASE); use a 30 dB amplifier
    G_hi = 1.0e3
    S_ASE = inversion_factor_nsp(rho) * H_PLANCK * nu0 * (G_hi - 1.0)
    var = detector_noise_variances(1.0e-4, S_ASE, R_A_W=1.0, B_Hz=2e10, dnu_opt_Hz=1e12,
                                   m_pol=2, I_dark_A=1e-9)
    g_c = bool(var["sig_spont"] > 10.0 * var["shot"] and var["spont_spont"] > 0.0
               and var["total"] > var["sig_spont"])
    ok = ok and g_c
    print("[nm] GATE C: beat noise -- sig-spont {:.2e} > shot {:.2e}, spont-spont {:.2e} "
          "(ASE-beat-limited) -> {}".format(var["sig_spont"], var["shot"],
                                            var["spont_spont"], "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: SFDR/ENOB optimal drive (window not a wall) ----
    qd = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    I, Ld, Gam = 40.0e-3, 0.6e-3, qd.p.Gamma
    y0 = qd.steady_state(I)                                  # unsaturated
    g0 = qd.material_gain_per_m(qd.rho_GS(y0), nu0)
    Gss = float(np.exp(Gam * g0 * Ld))
    nsp = inversion_factor_nsp(float(qd.rho_GS(y0)[0]))
    S_ASE_d = nsp * H_PLANCK * nu0 * (Gss - 1.0)
    P_in = np.logspace(-5, -1.2, 30)                         # 0.01 .. ~63 mW

    def P_out(Pin):                                          # lumped saturated transfer curve
        S = qd.photon_density(Pin, nu0)
        g = qd.material_gain_per_m(qd.rho_GS(qd.steady_state(I, S_conf_m3=S, nu_s_Hz=nu0)), nu0)
        return Pin * float(np.exp(Gam * g * Ld))
    P_out_grid = np.array([P_out(p) for p in P_in])

    def noise_var(Pout):
        return detector_noise_variances(Pout, S_ASE_d, R_A_W=1.0, B_Hz=2e10,
                                        dnu_opt_Hz=1e12, m_pol=2, I_dark_A=1e-9)["total"]
    P0s = np.logspace(-4, -2.0, 24)                          # sweep drive 0.1 .. 10 mW
    sndr, eno, iopt = sndr_vs_drive(P_in, P_out_grid, noise_var, P0s, mod_index=0.3)
    interior = bool(0 < iopt < P0s.size - 1)
    peaked = bool(sndr[iopt] > sndr[0] + 1.0 and sndr[iopt] > sndr[-1] + 1.0)
    g_d = bool(interior and peaked and np.isfinite(eno[iopt]))
    ok = ok and g_d
    print("[nm] GATE D: SNDR peaks at an INTERIOR drive P0={:.2f} mW (SNDR {:.1f} dB, "
          "ENOB {:.1f}); ends {:.1f}/{:.1f} dB -> {}".format(
              P0s[iopt] * 1e3, sndr[iopt], eno[iopt], sndr[0], sndr[-1],
              "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: predistortion linearizes the compressed channel ----
    from dynameta.optics.soa.metrics import (optimal_drive_power, pattern_penalty_dB,
                                             predistort, sfdr_dB, thermal_drift_budget_K)
    # a fine, well-conditioned transfer curve over the operating range (linear P_in spacing,
    # away from the flat-saturation tail where inversion is ill-conditioned)
    P_in_fine = np.linspace(2e-4, 1.0e-2, 80)
    P_out_fine = np.array([P_out(p) for p in P_in_fine])
    # request equally-spaced output levels; predistort finds the inputs that produce them
    targets = np.linspace(P_out_fine[3], P_out_fine[-4], 9)
    P_in_req = predistort(P_in_fine, P_out_fine, targets)
    achieved = np.array([P_out(p) for p in P_in_req])
    lin_resid = np.max(np.abs(achieved - targets)) / (targets[-1] - targets[0])
    # vs the uncorrected (equally-spaced INPUT) ladder, whose output is compressed
    P_in_lin = np.linspace(P_in_req[0], P_in_req[-1], 9)
    out_uncorr = np.array([P_out(p) for p in P_in_lin])
    fit = np.polyfit(np.arange(9), out_uncorr, 1)
    uncorr_resid = np.max(np.abs(out_uncorr - np.polyval(fit, np.arange(9)))) / \
        (out_uncorr[-1] - out_uncorr[0])
    g_e = bool(lin_resid < 5e-3 and lin_resid < 0.1 * uncorr_resid)
    ok = ok and g_e
    print("[nm] GATE E: predistortion linearizes -- residual {:.2e} vs uncorrected {:.2e} "
          "({:.0f}x better) -> {}".format(lin_resid, uncorr_resid,
                                          uncorr_resid / max(lin_resid, 1e-12),
                                          "PASS" if g_e else "FAIL"), flush=True)

    # ---- GATE F: analysis helpers (optimal drive, SFDR, pattern penalty, thermal budget) ----
    P0_opt, sndr_opt, enob_opt = optimal_drive_power(P_in, P_out_grid, noise_var, P0s,
                                                     mod_index=0.3)
    sfdr_at_opt = sfdr_dB(P_in, P_out_grid, P0_opt, noise_var(P_out(P0_opt)), mod_index=0.3)
    pp = pattern_penalty_dB(np.array([1.00, 0.92, 0.97, 0.90]))   # example mark peaks
    dT8 = thermal_drift_budget_K(8, 0.02)                        # 8-bit, 0.02 dB/K gain drift
    dT4 = thermal_drift_budget_K(4, 0.02)
    g_f = bool(abs(P0_opt - P0s[iopt]) < 1e-15 and sfdr_at_opt > 0.0 and pp > 0.0
               and dT8 < dT4 and np.isfinite(dT8))
    ok = ok and g_f
    print("[nm] GATE F: helpers -- optimal drive {:.2f} mW (ENOB {:.1f}), SFDR {:.1f} dB, "
          "pattern penalty {:.2f} dB, thermal budget dT8={:.3f}<dT4={:.3f} K -> {}".format(
              P0_opt * 1e3, enob_opt, sfdr_at_opt, pp, dT8, dT4, "PASS" if g_f else "FAIL"),
          flush=True)

    print("[nm] *** QD-SOA NOISE + METRICS: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
