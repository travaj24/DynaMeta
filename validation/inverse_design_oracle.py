"""Reduces-to-TMM oracle for the differentiable inverse-design forward model (audit round-1 inv-des-1,
re-confirmed in round 2: Fdtd2dDesignProblem.spectrum had NO independent cross-check). At a UNIFORM
projected density the designable slab is a homogeneous film, so the differentiable FDTD spectrum must
reproduce the coherent-TMM stack:

GATE A (void limit): rho = 0 -> eps = eps_lo = 1 everywhere -> R ~ 0 and T ~ 1 at every target bin.
GATE B (solid slab): rho = 1 -> a homogeneous eps_hi film of thickness n_des*dz (the designable region
        is n_des grid CELLS; slab_thickness_m only sizes the domain) -> R/T match stack_rta to a few %
        at each target wavelength.

Requires JAX (x64); prints SKIP and exits 0 when absent. Run: python -m validation.inverse_design_oracle
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

C = 299792458.0


def main():
    try:
        import jax  # noqa: F401
    except ImportError:
        print("[ido] SKIP: jax not installed (the differentiable forward is jax-only)", flush=True)
        return True
    from dynameta.optics.inverse_design import Fdtd2dDesignProblem
    from dynameta.optics.tmm_reference import stack_rta

    print("[ido] === inverse-design forward (Fdtd2dDesignProblem.spectrum) vs TMM ===", flush=True)
    lams = [1300e-9, 1500e-9]
    eps_hi, n_des, resolution = 6.25, 4, 16
    prob = Fdtd2dDesignProblem(period_x_m=300e-9, lambdas_m=lams, slab_thickness_m=400e-9,
                               n_des=n_des, eps_lo=1.0, eps_hi=eps_hi, resolution=resolution)
    dz = prob.args[1]
    t_slab = n_des * dz                                    # the ACTUAL designable-film thickness
    nx = int(prob.eps_base.shape[0])

    # GATE A: void (rho = 0) -> vacuum -> R ~ 0, T ~ 1
    R0, T0 = prob.spectrum(np.zeros((nx, n_des)))
    R0 = np.asarray(R0); T0 = np.asarray(T0)
    g_a = bool(np.all(R0 < 5e-3) and np.all(np.abs(T0 - 1.0) < 5e-2))
    print("[ido] GATE A void: R={} T={} -> {}".format(np.round(R0, 5), np.round(T0, 4),
                                                      "PASS" if g_a else "FAIL"), flush=True)

    # GATE B: solid (rho = 1) -> homogeneous eps_hi slab of thickness n_des*dz vs coherent TMM
    R1, T1 = prob.spectrum(np.ones((nx, n_des)))
    R1 = np.asarray(R1); T1 = np.asarray(T1)
    ok_b = True
    for k, lam in enumerate(lams):
        Rt, Tt, _A = stack_rta(1.0, [(np.sqrt(eps_hi), t_slab)], 1.0, lam)
        dR, dT = abs(float(R1[k]) - Rt), abs(float(T1[k]) - Tt)
        ok_b = ok_b and dR < 4e-2 and dT < 4e-2
        print("[ido]   lam={:.0f}nm slab {:.1f}nm: R {:.4f}/{:.4f} T {:.4f}/{:.4f} |dR|={:.3f} "
              "|dT|={:.3f}".format(lam * 1e9, t_slab * 1e9, float(R1[k]), Rt, float(T1[k]), Tt,
                                   dR, dT), flush=True)
    print("[ido] GATE B solid slab vs TMM -> {}".format("PASS" if ok_b else "FAIL"), flush=True)

    ok = g_a and ok_b
    print("[ido] *** INVERSE-DESIGN FORWARD ORACLE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
