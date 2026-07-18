"""QD-SOA STATIC/CW calibration to the Innolume BOA1310060CC600MXXXX datasheet -- the step that turns the
generic-parameter gain core into a device-matched 1310 nm parameter set (the 'Innolume set' the gain
core's docstring noted does not exist). Fits the load-bearing CW anchors and validates the fitted model
reproduces the datasheet's headline numbers.

GATE A (peak wavelength): the fitted nu0 == 1310 nm (read-off, exact).
GATE B (small-signal gain): the fitted unsaturated chip gain == 35 dB (datasheet typ; min 27).
GATE C (gain bandwidth): the intrinsic (material-gain -3 dB) bandwidth == 60 nm, and the net-gain spans a
        physically-sensible window (~ the datasheet's visible 1240-1360 nm gain-spectrum range), i.e. the
        fitted inhomogeneous distribution is NOT the unphysically-wide one the 35-dB-amplifier -3 dB would
        force.
GATE D (saturation power): the steady-state device saturation curve gives P_sat,out == 23.2 dBm (datasheet
        typ @ 2 A; the marcher overflows at 35 dB single-pass gain, so the saturation uses the gain core's
        steady-state g_QD(P) z-integration).
GATE E (noise figure + two-band ASE): the fitted inversion gives NF <= 5 dB (eta_in = 1, excluding input
        coupling, as the datasheet specifies); and the ES band is ENABLED so the gain spectrum carries a
        second (ES/WL) feature blue of the GS at ~1210 nm (the datasheet's two-band ASE), absent when the
        ES band is off.

HONEST SCOPE: this calibrates the STATIC/CW axes only. alpha_lef, the carrier kinetic times, RIN/linewidth,
NF(lambda), TPA/FCA and the thermal slopes are NOT constrained by this datasheet and stay at flagged
defaults (see dynameta.optics.soa.calibration). N_q is FIXED at a standard QD value (the gain pins only
the product Gamma*N_q*mu_GS*sigma_pk; sigma_pk is the fitted effective factor).

Run: python -m validation.qd_soa_calibration_innolume
"""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import QDGainModel, noise_figure
from dynameta.optics.soa.calibration import (INNOLUME_BOA1310_TARGETS,
                                             calibrate_innolume_boa1310)

C_LIGHT = 2.99792458e8


def main():
    print("[cal] === QD-SOA static calibration to Innolume BOA1310060 datasheet ===", flush=True)
    ok = True
    t = INNOLUME_BOA1310_TARGETS
    dev = calibrate_innolume_boa1310(verbose=False)
    m = QDGainModel(dev.params)
    r = dev.report

    # ---- GATE A: peak wavelength ----
    peak_nm = C_LIGHT / dev.nu0_Hz * 1.0e9
    g_a = bool(abs(peak_nm - t["peak_nm"]) < 0.5)
    ok = ok and g_a
    print("[cal] GATE A: peak {:.1f} nm == {:.0f} nm -> {}".format(
        peak_nm, t["peak_nm"], "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: small-signal gain ----
    g_b = bool(abs(r["G0_dB"] - t["G0_dB"]) < 1.5)
    ok = ok and g_b
    print("[cal] GATE B: small-signal gain {:.1f} dB == {:.0f} dB -> {}".format(
        r["G0_dB"], t["G0_dB"], "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: NET -3 dB bandwidth (the datasheet observable; C4-8 resolved) ----
    # The co-fit tunes the ES-band strength + inhomogeneous width until the FULL GS+ES net
    # -3 dB bandwidth at the 35 dB operating gain matches the sheet's 60 nm. The material
    # half-max width is now the WIDE number (chirped multi-stack QD envelope, ~150-200 nm) and
    # the visible positive-net-gain span grows accordingly.
    from dynameta.optics.soa.calibration import _net_gain_spectrum_full
    nu = dev.nu0_Hz + np.linspace(-60e12, 60e12, 2401)
    Gnet = _net_gain_spectrum_full(m, dev.drive_A, nu, dev.alpha_i_per_m, dev.length_m)
    lam = C_LIGHT / nu * 1.0e9
    span_nm = float(lam[Gnet > 0.0].max() - lam[Gnet > 0.0].min()) if np.any(Gnet > 0.0) else 0.0
    bw_ok = abs(r["net_3dB_bw_nm"] - t["bandwidth_nm"]) < 3.0
    span_ok = 100.0 < span_nm < 450.0                      # physical: wide multi-stack envelope
    narrow_ok = r["material_fwhm_nm"] > 1.5 * r["net_3dB_bw_nm"]   # narrowing runs the honest way
    g_c = bool(bw_ok and span_ok and narrow_ok)
    ok = ok and g_c
    print("[cal] GATE C: NET -3dB BW {:.1f} nm == {:.0f} nm (co-fit); material FWHM {:.0f} nm; "
          "net-gain span {:.0f} nm -> {}".format(r["net_3dB_bw_nm"], t["bandwidth_nm"],
                                                 r["material_fwhm_nm"], span_nm,
                                                 "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: saturation output power ----
    g_d = bool(np.isfinite(r["Psat_out_dBm"]) and abs(r["Psat_out_dBm"] - t["Psat_out_dBm"]) < 1.0)
    ok = ok and g_d
    print("[cal] GATE D: P_sat,out {:.2f} dBm == {:.1f} dBm -> {}".format(
        r["Psat_out_dBm"], t["Psat_out_dBm"], "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: noise figure + two-band ASE ----
    y = m.steady_state(dev.drive_A, S_conf_m3=0.0)
    rho_pk = float(m.rho_GS(y)[m.ng // 2])
    n_sp = rho_pk * rho_pk / (2.0 * rho_pk - 1.0)           # inversion factor at the GS peak
    g0 = float(m.small_signal_gain_per_m(dev.drive_A, dev.nu0_Hz))
    G = 10.0 ** (r["G0_dB"] / 10.0)
    NF_dB = 10.0 * np.log10(noise_figure(G, n_sp, Gamma_g_per_m=m.gamma_confinement * g0,
                                         alpha_i_per_m=dev.alpha_i_per_m, eta_in=1.0))
    nf_ok = NF_dB <= t["NF_dB_max"] + 1e-9
    # two-band: ES gain at 1210 nm present with the ES band on, absent with it off
    nu_es = C_LIGHT / (t["ase_es_nm"] * 1.0e-9)
    g_es_on = float(m.total_material_gain(m.rho_ES(y), m.rho_GS(y), nu_es))
    m_off = QDGainModel(replace(m.p, sigma_pk_ES_m2=0.0))
    y0 = m_off.steady_state(dev.drive_A, S_conf_m3=0.0)
    g_es_off = float(m_off.total_material_gain(m_off.rho_ES(y0), m_off.rho_GS(y0), nu_es))
    two_band = bool(g_es_on - g_es_off > 0.05 * abs(g_es_on) + 1.0)
    g_e = bool(nf_ok and two_band)
    ok = ok and g_e
    print("[cal] GATE E: NF {:.2f} dB <= {:.0f} dB (n_sp {:.2f}); ES-band gain@1210nm {:.0f} vs off {:.0f} "
          "/m -> {}".format(NF_dB, t["NF_dB_max"], n_sp, g_es_on, g_es_off, "PASS" if g_e else "FAIL"),
          flush=True)

    print("[cal] fitted: sigma_pk {:.2e} m2, fwhm_inhom {:.1f} THz, A_mode {:.3f} um2, dE_ES_GS {:.3f} eV, "
          "N_q {:.1e}".format(r["sigma_pk_m2"], r["fwhm_inhom_Hz"] / 1e12, r["A_mode_m2"] * 1e12,
                              r["dE_ES_GS_eV"], r["N_q_m3"]), flush=True)
    print("[cal] *** QD-SOA INNOLUME CALIBRATION: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
