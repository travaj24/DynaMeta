"""QD-SOA dynamic parameters INFERRED from the static/CW Innolume calibration -- physically-motivated
ESTIMATES (NOT measurements) that exploit the physical LINKS between CW observables and dynamic
quantities, each carrying an explicit confidence. (See dynameta.optics.soa.calibration.infer_dynamics_
from_cw.) This validation checks the inferences are self-consistent and physical, and that the honest
labelling holds. It uses the Phase-34 fitted parameters DIRECTLY (a fast standalone build) -- the
calibration fit itself is validated by qd_soa_calibration_innolume.

GATE A (mode area from divergence): A_eff from the 6 deg / 27 deg far-field is a physical guided-mode area
        (a few-to-tens of um^2 for this low-confinement broad-area 600 mW BOA), consistent with the
        Gaussian far-field formula.
GATE B (gain-clamping diagnostic): the GS differential gain dg/dN ~ 0 at the 2 A operating point because
        the GS is ~fully inverted (2 rho_GS - 1 > 0.99) -- the QD gain is CLAMPED and the saturation is
        reservoir-limited, which is WHY tau_eff uses the cross-section form, not dg/dN.
GATE C (effective recovery time): tau_eff = A_eff h nu / (Gamma sigma_pk P_sat) lands in the physical QD
        gain-recovery range (~10-1000 ps) and round-trips the calibrated P_sat exactly (arithmetic check).
GATE D (slow modulation bandwidth): f_3dB ~ 1/(2 pi tau_eff) is in a sane range (0.1-100 GHz) -- the
        slow-envelope estimate (the true high-speed response is set by the unmeasured ps dynamics).
GATE E (honest labelling): alpha is flagged NOT-inferable and retained at its default BECAUSE the fitted
        gain is SYMMETRIC about nu0 (so its Kramers-Kronig index change is ~0 at the peak); and every
        inferred entry carries a non-empty confidence + method string.

Run: python -m validation.qd_soa_inferred_dynamics
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import QDGainModel, QDGainParams
from dynameta.optics.soa.calibration import CalibratedDevice, infer_dynamics_from_cw

C_LIGHT = 2.99792458e8
from dynameta.constants import H_PLANCK   # single source (audit 6.3)


def _innolume_device():
    """The Phase-34 fitted Innolume BOA1310060 set, built directly (the slow fit is validated separately)."""
    nu0 = C_LIGHT / 1310e-9
    p = QDGainParams(n_groups=41, nu0_Hz=nu0, fwhm_inhom_Hz=1.01e13, sigma_pk_m2=1.60e-18,
                     A_mode_m2=0.215e-12, T_K=298.0, sigma_pk_ES_m2=4.8e-19, dE_ES_GS_eV=0.078,
                     N_q_m3=5.0e22).with_detailed_balance_taus()
    return CalibratedDevice(params=p, length_m=8.0e-3, alpha_i_per_m=300.0, drive_A=2.0, nu0_Hz=nu0,
                            name="Innolume BOA1310060 (fitted)", report={"Psat_out_dBm": 23.2})


def main():
    print("[inf] === QD-SOA dynamics INFERRED from the CW Innolume calibration ===", flush=True)
    ok = True
    dev = _innolume_device()
    m = QDGainModel(dev.params)
    inf = infer_dynamics_from_cw(dev)

    # ---- GATE A: mode area from divergence ----
    A_um2 = inf["A_eff_m2"].value * 1e12
    g_a = bool(2.0 < A_um2 < 40.0)
    ok = ok and g_a
    print("[inf] GATE A: A_eff {:.1f} um2 (physical guided-mode area) -> {}".format(
        A_um2, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: gain-clamping diagnostic ----
    inv_pk = float(2.0 * m.rho_GS(m.steady_state(dev.drive_A, S_conf_m3=0.0))[m.ng // 2] - 1.0)
    dgdN = inf["dg_dN_diagnostic_m2"].value
    clamped = bool(inv_pk > 0.99 and abs(dgdN) < 1e-22)     # near-full inversion + ~0 differential gain
    g_b = bool(clamped)
    ok = ok and g_b
    print("[inf] GATE B: GS clamped (inversion {:.3f} > 0.99, dg/dN {:.1e} ~ 0) -> {}".format(
        inv_pk, dgdN, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: effective recovery time + P_sat round-trip ----
    tau = inf["tau_eff_s"].value
    Psat_W = 10.0 ** (dev.report["Psat_out_dBm"] / 10.0) * 1.0e-3
    Psat_rt = inf["A_eff_m2"].value * H_PLANCK * dev.nu0_Hz / (
        m.gamma_confinement * dev.params.sigma_pk_m2 * tau)   # invert the tau_eff relation
    rt = abs(Psat_rt - Psat_W) / Psat_W
    g_c = bool(10e-12 < tau < 1000e-12 and rt < 1e-9)
    ok = ok and g_c
    print("[inf] GATE C: tau_eff {:.0f} ps in [10,1000] ps; P_sat round-trip rel {:.1e} -> {}".format(
        tau * 1e12, rt, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: slow modulation bandwidth ----
    f3 = inf["f_3dB_slow_Hz"].value
    g_d = bool(0.1e9 < f3 < 100e9 and abs(f3 - 1.0 / (2 * np.pi * tau)) / f3 < 1e-6)
    ok = ok and g_d
    print("[inf] GATE D: f_3dB(slow) {:.2f} GHz (consistent with tau_eff) -> {}".format(
        f3 / 1e9, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: honest labelling (alpha not inferable from the symmetric gain) ----
    d = 0.3 * dev.params.fwhm_inhom_Hz
    y = m.steady_state(dev.drive_A, S_conf_m3=0.0)
    g_plus = float(m.material_gain_per_m(m.rho_GS(y), dev.nu0_Hz + d))
    g_minus = float(m.material_gain_per_m(m.rho_GS(y), dev.nu0_Hz - d))
    symmetric = abs(g_plus - g_minus) / abs(g_plus) < 1e-3   # symmetric -> KK index ~0 at peak -> alpha n/a
    alpha_flagged = ("NOT inferable" in inf["alpha_lef"].confidence
                     and inf["alpha_lef"].value == dev.params.alpha_lef)
    labels_ok = all(v.confidence and v.method for v in inf.values())
    g_e = bool(symmetric and alpha_flagged and labels_ok)
    ok = ok and g_e
    print("[inf] GATE E: gain symmetric (KK alpha ~0) {}, alpha flagged-not-inferable {}, all labelled {} "
          "-> {}".format(symmetric, alpha_flagged, labels_ok, "PASS" if g_e else "FAIL"), flush=True)

    print("[inf] *** QD-SOA CW-INFERRED DYNAMICS: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
