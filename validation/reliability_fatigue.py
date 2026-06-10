"""REL6 thermal-cycling fatigue oracle (ductile Coffin-Manson vs brittle Weibull -- the corrected
split).

GATE A (reduces-to-closed-form): the biaxial mismatch stress matches the hand evaluation (Cu-on-Si at
        dT = 100 K lands in the literature 100-300 MPa band); dT = 0 -> zero stress; zero plastic
        strain -> Nf = inf; the Coffin-Manson exponent is 1/c ON the strain (slope check, the
        audit-caught inversion).
GATE B (Norris-Landzberg closed form + direction): AF matches the closed form to machine and is > 1
        for a harsher test (bigger dT, hotter Tmax).
GATE C (brittle-vs-ductile split): a brittle low-sigma_crit film cracks at the FIRST excursion
        (0 cycles) at a dT where the ductile metal still has finite Coffin-Manson life; Weibull
        survival matches its closed form and sigma = 0 -> survival 1 exactly.

Run: python -m validation.reliability_fatigue
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.reliability.fatigue import (MechanicalProps, biaxial_stress_Pa, coffin_manson_nf,
                                          plastic_strain_range, norris_landzberg_af,
                                          brittle_survival, cycles_to_failure)

CU = MechanicalProps(E_Pa=110e9, nu=0.34, cte_per_K=16.5e-6)            # ductile (sigma_crit = inf)
ITO = MechanicalProps(E_Pa=115e9, nu=0.35, cte_per_K=6.0e-6, sigma_crit_Pa=0.3e9)   # brittle
CTE_SI = 2.6e-6


def main():
    print("[rf] === REL6 thermal-cycling fatigue (Coffin-Manson + brittle Weibull) ===", flush=True)
    ok = True

    # ---- GATE A ----
    sig = float(biaxial_stress_Pa(CU, CTE_SI, 100.0))
    sig_hand = 110e9 / (1 - 0.34) * (2.6e-6 - 16.5e-6) * 100.0          # -231.7 MPa
    zero = float(biaxial_stress_Pa(CU, CTE_SI, 0.0))
    nf_inf = coffin_manson_nf(0.0)
    eps = np.array([1e-4, 1e-3, 1e-2])
    nf = coffin_manson_nf(eps, C=0.5, c_ductility=0.6)
    slope = float(np.mean(np.diff(np.log(nf)) / np.diff(np.log(eps))))
    g_a = (abs(sig - sig_hand) < 1.0 and 1e8 < abs(sig) < 3e8 and zero == 0.0
           and np.isinf(nf_inf) and abs(slope + 1.0 / 0.6) < 1e-12)
    ok = ok and bool(g_a)
    print("[rf] GATE A: Cu-on-Si dT=100K sigma = {:.1f} MPa (hand {:.1f}; literature 100-300); "
          "dT=0 -> 0; d_eps=0 -> Nf=inf; log-slope = {:.4f} == -1/c -> {}".format(
              sig / 1e6, sig_hand / 1e6, slope, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Norris-Landzberg ----
    af = norris_landzberg_af(f_use_Hz=1.0e-4, f_test_Hz=5.6e-4, dT_use_K=40.0, dT_test_K=165.0,
                             Tmax_use_K=358.0, Tmax_test_K=398.0)
    af_an = (5.6e-4 / 1.0e-4) ** (1 / 3) * (165.0 / 40.0) ** 2 * np.exp(1414.0 * (1 / 358.0 - 1 / 398.0))
    g_b = bool(abs(af / af_an - 1) < 1e-12 and af > 1.0)
    ok = ok and g_b
    print("[rf] GATE B: Norris-Landzberg AF = {:.1f} == closed form, > 1 for the harsher test "
          "-> {}".format(af, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: brittle-vs-ductile split ----
    dT = 600.0                                                          # ITO stress 0.18 GPa/100K *6
    sig_ito = abs(float(biaxial_stress_Pa(ITO, CTE_SI, dT)))
    n_ito = cycles_to_failure(ITO, CTE_SI, dT)                          # over sigma_crit -> 0 cycles
    n_cu = cycles_to_failure(CU, CTE_SI, dT)                           # ductile -> finite CM life
    surv = float(brittle_survival(0.2e9, sigma0_Pa=0.4e9, m_weibull=8.0))
    surv_an = float(np.exp(-(0.5) ** 8))
    g_c = bool(sig_ito >= ITO.sigma_crit_Pa and n_ito == 0.0 and 0.0 < n_cu < np.inf
               and abs(surv - surv_an) < 1e-15 and brittle_survival(0.0, sigma0_Pa=1e9,
                                                                    m_weibull=8.0) == 1.0)
    ok = ok and g_c
    print("[rf] GATE C: ITO at dT={:.0f}K sigma {:.2f} GPa >= sigma_crit -> cracks at cycle 0; Cu "
          "ductile Nf = {:.3g} cycles; Weibull survival == closed form -> {}".format(
              dT, sig_ito / 1e9, n_cu, "PASS" if g_c else "FAIL"), flush=True)

    print("[rf] *** REL6 THERMAL-CYCLING FATIGUE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
