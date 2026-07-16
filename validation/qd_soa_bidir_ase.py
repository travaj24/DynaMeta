"""QD-SOA bidirectional, spectrally-resolved ASE (the final gain-leg ceiling) vs oracles. The
forward-only single-band ase_output_psd is generalized to a frequency grid with the z-resolved
gain profile, forward + backward propagation, the spectral noise figure, and ASE-induced gain
saturation (the integrated bidirectional ASE photon density depletes the inversion).

GATE A (reduction to ase_output_psd): single nu, forward-only, above transparency -> identical to
        the existing forward integrator (the source q = Gamma g_sp h nu == Gamma g n_sp h nu).
GATE B (uniform spectral sum): for a uniform inversion profile, S_f(nu_k, L) == n_sp(nu_k) h nu_k
        (G(nu_k)-1) per frequency (n_sp(nu_k) = g_sp/g, the spectral inversion factor).
GATE C (spectral noise figure): NF(nu_k) -> 2 n_sp at high gain and equals noise_figure at band
        centre.
GATE D (bidirectional symmetry): a uniform device emits the SAME forward (z=L) and backward (z=0)
        ASE per frequency, S_f(nu_k, L) == S_b(nu_k, 0).
GATE E (ASE-induced gain saturation): the self-consistent integrated bidirectional ASE depletes
        the inversion, so the saturated gain is strictly below the unsaturated gain, deepening
        monotonically with the ASE load; ase_saturation=False reproduces the unsaturated propagator
        exactly (the OFF/reduction path).

Run: python -m validation.qd_soa_bidir_ase
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import HBAR
from dynameta.optics.soa.ase_noise import (ase_output_psd, ase_self_consistent,
                                           ase_spectrum_bidirectional, noise_figure)
from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams

from dynameta.constants import H_PLANCK   # single source (audit 6.3)


def main():
    print("[ba] === QD-SOA bidirectional spectrally-resolved ASE vs oracles ===", flush=True)
    ok = True
    N, dz = 60, 1.0e-5

    # ---- GATE A: reduction to ase_output_psd (single nu, forward, above transparency) ----
    m1 = QDGainModel(QDGainParams(n_groups=1).with_detailed_balance_taus())
    nu0, Gamma = m1.p.nu0_Hz, m1.p.Gamma
    worst_a = 0.0
    for rho_val, ai in [(0.8, 0.0), (0.95, 0.0), (0.75, 200.0)]:
        rho = np.full(1, rho_val)
        g = float(m1.material_gain_per_m(rho, nu0))
        gsp = float(m1.emission_gain_per_m(rho, nu0))
        g_sl = np.full(N, g)
        anchor = ase_output_psd(g_sl, np.full(N, rho_val), dz, nu0, Gamma, ai, m_pol=2)
        got = ase_spectrum_bidirectional(np.full((N, 1), g), np.full((N, 1), gsp), dz,
                                         np.array([nu0]), np.array([1e10]), Gamma,
                                         alpha_i_per_m=ai, m_pol=2, direction="forward")["S_f_out"][0]
        worst_a = max(worst_a, abs(got - anchor) / abs(anchor))
    g_a = bool(worst_a < 1e-13)
    ok = ok and g_a
    print("[ba] GATE A: single-nu forward == ase_output_psd (max rel {:.1e}) -> {}".format(
        worst_a, "PASS" if g_a else "FAIL"), flush=True)

    # ---- spectral grid + multi-group profile ----
    m = QDGainModel(QDGainParams(n_groups=21).with_detailed_balance_taus())
    nu = np.linspace(nu0 - 4e12, nu0 + 4e12, 41)
    dnu = np.gradient(nu)
    y = m.steady_state(40.0e-3)
    rhoGS = m.rho_GS(y)
    g_nu = m.material_gain_per_m(rhoGS, nu)
    gsp_nu = m.emission_gain_per_m(rhoGS, nu)
    res = ase_spectrum_bidirectional(np.tile(g_nu, (N, 1)), np.tile(gsp_nu, (N, 1)), dz, nu, dnu,
                                     Gamma, m_pol=2)

    # ---- GATE B: uniform spectral sum ----
    with np.errstate(divide="ignore", invalid="ignore"):
        nsp_spec = np.where(g_nu > 0.0, gsp_nu / g_nu, 0.0)
    amp = res["G"] > 1.001                                   # the amplifying band
    S_ref = nsp_spec * H_PLANCK * nu * (res["G"] - 1.0)
    rel_b = float(np.max(np.abs(res["S_f"][amp] - S_ref[amp]) / np.abs(S_ref[amp])))
    g_b = bool(rel_b < 1e-12)
    ok = ok and g_b
    print("[ba] GATE B: S_f(nu_k,L) == n_sp(nu_k) h nu (G-1) over gain band (max rel {:.1e}) -> "
          "{}".format(rel_b, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: spectral NF -> 2 n_sp high gain + == noise_figure at centre ----
    # long device (2 cm-equiv slices) so the band centre reaches G > 1e3 for the 2 n_sp limit
    resHG = ase_spectrum_bidirectional(np.tile(g_nu, (2000, 1)), np.tile(gsp_nu, (2000, 1)), dz, nu,
                                       dnu, Gamma, m_pol=2)
    k0 = int(np.argmin(np.abs(nu - nu0)))
    nf_ref = noise_figure(float(resHG["G"][k0]), float(nsp_spec[k0]))
    rel_centre = abs(float(resHG["NF"][k0]) - nf_ref) / nf_ref
    hi = resHG["G"] > 1e3
    to_2nsp = (float(np.max(np.abs(resHG["NF"][hi] - 2.0 * nsp_spec[hi]) / (2.0 * nsp_spec[hi])))
               if hi.any() else 1.0)
    # lossy device: NF must NOT double-count internal loss (it already lives in the net-propagated
    # S_f via a = Gamma g - alpha_i) -- spectral NF @centre == noise_figure(bare n_sp, Gg, alpha_i)
    ai = 200.0
    resL = ase_spectrum_bidirectional(np.tile(g_nu, (N, 1)), np.tile(gsp_nu, (N, 1)), dz, nu, dnu,
                                      Gamma, alpha_i_per_m=ai, m_pol=2)
    Gg_per_m = np.log(float(resL["Gg"][k0])) / (N * dz)
    nf_lossy = noise_figure(float(resL["G"][k0]), float(nsp_spec[k0]), Gamma_g_per_m=Gg_per_m,
                            alpha_i_per_m=ai)
    rel_lossy = abs(float(resL["NF"][k0]) - nf_lossy) / nf_lossy
    g_c = bool(rel_centre < 1e-10 and hi.any() and to_2nsp < 1e-2 and rel_lossy < 1e-10)
    ok = ok and g_c
    print("[ba] GATE C: spectral NF @centre == noise_figure (rel {:.1e}); -> 2 n_sp at high gain "
          "(rel {:.1e}) -> {}".format(rel_centre, to_2nsp, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: bidirectional symmetry (uniform device) ----
    rel_d = float(np.max(np.abs(res["S_f"] - res["S_b"]) / np.maximum(res["S_f"], 1e-300)))
    g_d = bool(rel_d < 1e-13)
    ok = ok and g_d
    print("[ba] GATE D: uniform device S_f(L) == S_b(0) per nu (max rel {:.1e}) -> {}".format(
        rel_d, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: ASE-induced gain saturation ----
    off = ase_self_consistent(m, 40.0e-3, 0.0, nu0, nu, dnu, 0.6e-3, n_slices=N, m_pol=2,
                              ase_saturation=False)
    clamps = []
    for s in (1.0, 5.0, 20.0):                               # increasing ASE load
        on = ase_self_consistent(m, 40.0e-3, 0.0, nu0, nu, dnu, 0.6e-3, n_slices=N, m_pol=2,
                                 ase_saturation=True, ase_strength=s)
        clamps.append(float(np.max(off["g_sat"] - on["g_sat"])))   # unsat - sat per nu
    off_exact = bool(np.array_equal(off["g_sat"], g_nu))    # OFF == unsaturated on signal-only carriers
    monotone = bool(clamps[0] > 0.0 and clamps[1] > clamps[0] and clamps[2] > clamps[1])
    g_e = bool(off_exact and monotone)
    ok = ok and g_e
    print("[ba] GATE E: ASE clamps gain (unsat-sat {:.2e}/{:.2e}/{:.2e} /m at strength 1/5/20, "
          "monotone {}); OFF==unsaturated {} -> {}".format(
              clamps[0], clamps[1], clamps[2], monotone, off_exact, "PASS" if g_e else "FAIL"),
          flush=True)

    print("[ba] *** QD-SOA BIDIRECTIONAL ASE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
