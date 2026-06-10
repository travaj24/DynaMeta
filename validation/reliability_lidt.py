"""REL5 optical-damage / thermal-runaway oracle (lumped CW node + LIDT fluence scaling).

GATE A (reduces-to-closed-form): I = 0 -> T == T_sink EXACTLY; constant absorbed fraction A0 -> the
        steady state is the algebraic T_sink + A0 I S R_th (brentq vs algebra, machine) and the
        transient is the exact exponential approach with tau = R_th C_th (solve_ivp vs closed form).
GATE B (EXACT runaway bifurcation): for a LINEAR absorbed(T) = a0 + a1 (T - T_sink) the balance has
        the closed solution dT = a0 I S R/(1 - a1 I S R) -- runaway detection must trip at the exact
        analytic threshold (accounting for the T_max truncation: detection occurs where the root
        reaches T_max, x_detect = K/(1+K), K = a1 (T_max - T_sink)/a0 -- ALSO closed-form).
GATE C (fluence law + the real-stack feedback premise): F_th ratios follow sqrt(tau); the ACTUAL
        ENZ-stack absorbed fraction built from layered_rta + MatthiessenGamma(T_K=T) stays in [0, 1]
        and the lumped solve converges at modest intensity (INFO: its runaway threshold).

Run: python -m validation.reliability_lidt
"""
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import M_E
from dynameta.reliability.lidt import (ThermalNode, lidt_fluence_J_m2, cw_steady_temperature_K,
                                       cw_critical_intensity_W_m2, cw_transient_K,
                                       stack_absorbed_of_T)

NODE = ThermalNode(R_th_K_W=1.0e4, C_th_J_K=1.0e-9, area_m2=1.0e-10, T_sink_K=300.0)


def main():
    print("[rl] === REL5 optical damage / CW thermal runaway ===", flush=True)
    ok = True

    # ---- GATE A: zero drive + constant absorption closed forms ----
    g_zero = cw_steady_temperature_K(lambda T: 0.3, 0.0, NODE) == 300.0
    A0, I = 0.3, 2.0e9
    T_ss = cw_steady_temperature_K(lambda T: A0, I, NODE)
    T_ss_an = 300.0 + A0 * I * NODE.area_m2 * NODE.R_th_K_W              # +60 K
    tau = NODE.R_th_K_W * NODE.C_th_J_K
    t = np.linspace(0.0, 5.0 * tau, 40)
    T_t = cw_transient_K(t, lambda T: A0, I, NODE)
    T_t_an = 300.0 + (T_ss_an - 300.0) * (1.0 - np.exp(-t / tau))
    rel_tr = float(np.max(np.abs(T_t - T_t_an) / (T_ss_an - 300.0)))
    g_a = bool(g_zero and abs(T_ss - T_ss_an) < 1e-9 and rel_tr < 1e-6)
    ok = ok and g_a
    print("[rl] GATE A: I=0 -> T_sink exact; steady {:.4f} K == algebra {:.4f} K; transient vs "
          "exponential rel {:.1e} -> {}".format(T_ss, T_ss_an, rel_tr, "PASS" if g_a else "FAIL"),
          flush=True)

    # ---- GATE B: exact linear-absorption bifurcation ----
    a0, a1, T_max = 0.01, 1.0e-4, 2000.0
    absorbed_lin = lambda T: a0 + a1 * (T - 300.0)
    SR = NODE.area_m2 * NODE.R_th_K_W
    I_crit_true = 1.0 / (a1 * SR)                                       # the pole of dT(I)
    K = a1 * (T_max - 300.0) / a0
    I_detect_an = I_crit_true * (K / (1.0 + K))                         # root hits T_max here (exact)
    I_detect = cw_critical_intensity_W_m2(absorbed_lin, NODE, I_lo=1e6, I_hi=1e13, T_max_K=T_max,
                                          rel_tol=1e-7)
    relB = abs(I_detect - I_detect_an) / I_detect_an
    # below the detected threshold the closed-form dT matches the solver
    I_test = 0.5 * I_detect_an
    dT_an = a0 * I_test * SR / (1.0 - a1 * I_test * SR)
    dT_num = cw_steady_temperature_K(absorbed_lin, I_test, NODE, T_max_K=T_max) - 300.0
    g_b = bool(relB < 1e-3 and abs(dT_num - dT_an) / dT_an < 1e-9)
    ok = ok and g_b
    print("[rl] GATE B: runaway detected at {:.6e} W/m^2 vs analytic {:.6e} (rel {:.1e}); below-"
          "threshold dT {:.3f} K == closed form -> {}".format(
              I_detect, I_detect_an, relB, dT_num, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: sqrt(tau) fluence law + the real ENZ stack absorbed(T) ----
    r = float(lidt_fluence_J_m2(4e-9, F_ref_J_m2=1.0e4, tau_ref_s=1e-9)
              / lidt_fluence_J_m2(1e-9, F_ref_J_m2=1.0e4, tau_ref_s=1e-9))
    from dynameta.core.layered import LayeredSlab, LayeredStack
    from dynameta.materials.optical_model import DrudeOptical
    from dynameta.materials.scattering import MatthiessenGamma

    gam300 = MatthiessenGamma(gamma_const_rad_s=5.0e13, gamma_phonon_300K_rad_s=6.0e13)
    LAM, N_ITO = 1300e-9, 9.0e26

    def build_stack_at_T(T_K):
        g = dataclasses.replace(gam300, T_K=T_K)            # the T-dependent phonon channel
        eps_ito = complex(DrudeOptical(eps_inf=4.0, m_opt_kg=0.3 * M_E,
                                       gamma_rad_s=float(g(N_ITO))).eps(LAM, n_m3=N_ITO))
        return LayeredStack(1.0 + 0j, np.sqrt(complex(-120.0 + 3.0j)),
                            [LayeredSlab(10e-9, eps=eps_ito), LayeredSlab(120e-9, eps=4.0 + 0j)])

    absorbed = stack_absorbed_of_T(build_stack_at_T, LAM)
    Avals = np.array([absorbed(T) for T in (300.0, 400.0, 500.0, 700.0)])
    T_op = cw_steady_temperature_K(absorbed, 5.0e8, NODE)
    g_c = bool(abs(r - 2.0) < 1e-12 and np.all(Avals >= 0.0) and np.all(Avals <= 1.0)
               and T_op > 300.0)
    ok = ok and g_c
    try:
        I_cr = cw_critical_intensity_W_m2(absorbed, NODE, I_lo=1e6, I_hi=1e15)
        print("[rl] INFO: real ENZ stack A(300..700K) = {} ; runaway at I ~ {:.2e} W/m^2".format(
            np.round(Avals, 3), I_cr), flush=True)
    except ValueError as e:
        print("[rl] INFO: real ENZ stack A(300..700K) = {} ; {}".format(np.round(Avals, 3), e),
              flush=True)
    print("[rl] GATE C: F_th(4 ns)/F_th(1 ns) = {:.1f} == sqrt(4); stack A in [0,1]; steady at "
          "5e8 W/m^2 = {:.2f} K -> {}".format(r, T_op, "PASS" if g_c else "FAIL"), flush=True)

    print("[rl] *** REL5 OPTICAL DAMAGE / RUNAWAY: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
