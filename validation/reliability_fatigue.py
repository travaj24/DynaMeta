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

    # ---- GATE A (de-tautologized per audit 7.3: the old reference re-typed the module's
    # own E/(1-nu) biaxial-modulus expression. The stress reference is now an INDEPENDENT
    # derivation path: solve isotropic Hooke's law numerically -- build the 3x3 compliance
    # matrix from (E, nu) primitives, impose equibiaxial stress s*[1,1,0] with sigma_z = 0,
    # and solve the linear constraint that the in-plane ELASTIC strain equals the mismatch
    # strain (CTE_sub - CTE_film)*dT. A wrong biaxial modulus -- 1-nu^2, 1-2nu, plain E --
    # fails this pin while sitting comfortably inside the literature band.) ----
    sig = float(biaxial_stress_Pa(CU, CTE_SI, 100.0))
    E, nu = CU.E_Pa, CU.nu
    S_compl = np.array([[1.0, -nu, -nu], [-nu, 1.0, -nu], [-nu, -nu, 1.0]]) / E
    eps_mm = (CTE_SI - CU.cte_per_K) * 100.0                # in-plane elastic mismatch strain
    sig_hooke = eps_mm / float((S_compl @ np.array([1.0, 1.0, 0.0]))[0])
    zero = float(biaxial_stress_Pa(CU, CTE_SI, 0.0))
    # property pins: linearity in dT and CTE-swap antisymmetry (sign/direction bugs)
    lin = float(biaxial_stress_Pa(CU, CTE_SI, 200.0))
    swapped = MechanicalProps(E_Pa=CU.E_Pa, nu=CU.nu, cte_per_K=CTE_SI)
    sig_swap = float(biaxial_stress_Pa(swapped, CU.cte_per_K, 100.0))
    nf_inf = coffin_manson_nf(0.0)
    eps = np.array([1e-4, 1e-3, 1e-2])
    nf = coffin_manson_nf(eps, C=0.5, c_ductility=0.6)
    slope = float(np.mean(np.diff(np.log(nf)) / np.diff(np.log(eps))))
    g_a = (abs(sig / sig_hooke - 1.0) < 1e-12 and 1e8 < abs(sig) < 3e8 and zero == 0.0
           and lin == 2.0 * sig and sig_swap == -sig
           and np.isinf(nf_inf) and abs(slope + 1.0 / 0.6) < 1e-12)
    ok = ok and bool(g_a)
    print("[rf] GATE A: Cu-on-Si dT=100K sigma = {:.1f} MPa == Hooke compliance solve {:.1f} "
          "(literature 100-300); dT=0 -> 0; 2x-dT + CTE-swap pins; d_eps=0 -> Nf=inf; "
          "log-slope = {:.4f} == -1/c -> {}".format(
              sig / 1e6, sig_hooke / 1e6, slope, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: Norris-Landzberg (de-tautologized per audit C4-1: the old reference
    # re-typed the AF expression itself, so the inverted frequency ratio passed by
    # construction. The reference is now the RATIO OF THE PRIMITIVE LAW
    # Nf(f, dT, Tmax) = f^m dT^-n exp(Ea/Tmax) -- an independent derivation path --
    # plus a direction-sensitive frequency-only leg.) ----
    def _nf(f, dT, Tmax, m=1.0 / 3.0, n=2.0, Ea=1414.0):
        return f ** m * dT ** (-n) * np.exp(Ea / Tmax)

    af = norris_landzberg_af(f_use_Hz=1.0e-4, f_test_Hz=5.6e-4, dT_use_K=40.0, dT_test_K=165.0,
                             Tmax_use_K=358.0, Tmax_test_K=398.0)
    af_an = _nf(1.0e-4, 40.0, 358.0) / _nf(5.6e-4, 165.0, 398.0)
    # direction: a test differing ONLY by cycling 8x faster is LESS damaging per cycle
    # (shorter creep dwell -> more cycles to failure in test), so AF = (1/8)^(1/3) = 0.5
    # exactly -- this leg FAILS on the pre-audit inverted ratio (which returned 2.0).
    af_freq_only = norris_landzberg_af(f_use_Hz=1.0e-4, f_test_Hz=8.0e-4, dT_use_K=100.0,
                                       dT_test_K=100.0, Tmax_use_K=360.0, Tmax_test_K=360.0)
    g_b = bool(abs(af / af_an - 1) < 1e-12 and af > 1.0
               and abs(af_freq_only - 0.5) < 1e-12)
    ok = ok and g_b
    print("[rf] GATE B: Norris-Landzberg AF = {:.2f} == primitive-law Nf ratio ({:.2f}), > 1 for "
          "the harsher test; frequency-only 8x-faster test AF = {:.3f} == 0.5 exactly "
          "-> {}".format(af, af_an, af_freq_only, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: brittle-vs-ductile split (Weibull leg de-tautologized per audit 7.3:
    # the old reference re-typed exp(-(s/s0)^m). Now pinned by Weibull PROPERTIES that
    # the code does not spell out: (1) the distribution-free anchor S(sigma0) = 1/e for
    # ANY shape m; (2) the functional equation S(s2) = S(s1)^((s2/s1)^m) -- ln S must
    # scale as s^m, a telescoping pin that only the Weibull family satisfies. Together
    # with S(0) = 1 these fully determine exp(-(s/s0)^m) without re-typing it.) ----
    dT = 600.0                                                          # ITO stress 0.18 GPa/100K *6
    sig_ito = abs(float(biaxial_stress_Pa(ITO, CTE_SI, dT)))
    n_ito = cycles_to_failure(ITO, CTE_SI, dT)                          # over sigma_crit -> 0 cycles
    n_cu = cycles_to_failure(CU, CTE_SI, dT)                           # ductile -> finite CM life
    anchor_ok = all(abs(float(brittle_survival(0.4e9, sigma0_Pa=0.4e9, m_weibull=m))
                        - float(np.exp(-1.0))) < 1e-15 for m in (2.0, 8.0))
    s1_v = float(brittle_survival(0.2e9, sigma0_Pa=0.4e9, m_weibull=8.0))
    s2_v = float(brittle_survival(0.3e9, sigma0_Pa=0.4e9, m_weibull=8.0))
    func_rel = abs(s1_v ** ((0.3e9 / 0.2e9) ** 8.0) / s2_v - 1.0)
    g_c = bool(sig_ito >= ITO.sigma_crit_Pa and n_ito == 0.0 and 0.0 < n_cu < np.inf
               and anchor_ok and func_rel < 1e-12 and 0.0 < s2_v < s1_v < 1.0
               and brittle_survival(0.0, sigma0_Pa=1e9, m_weibull=8.0) == 1.0)
    ok = ok and g_c
    print("[rf] GATE C: ITO at dT={:.0f}K sigma {:.2f} GPa >= sigma_crit -> cracks at cycle 0; Cu "
          "ductile Nf = {:.3g} cycles; Weibull 1/e anchor (m=2,8) + functional pin "
          "(rel {:.1e}) -> {}".format(
              dT, sig_ito / 1e9, n_cu, func_rel, "PASS" if g_c else "FAIL"), flush=True)

    print("[rf] *** REL6 THERMAL-CYCLING FATIGUE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
