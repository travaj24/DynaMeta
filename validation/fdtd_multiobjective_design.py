"""MULTI-OBJECTIVE / MULTI-WAVELENGTH inverse design over the differentiable JAX-FDTD. A SINGLE broadband
FDTD solve yields R/T at several target wavelengths (Fdtd2dDesignProblem.spectrum), and weighted_objective
combines per-wavelength goals into one differentiable loss that topology_optimize drives to a binary design
-- the capability that turns the differentiable forward into a real design tool.

DEMO: a wavelength-selective reflector (dichroic) -- MAXIMISE reflectance at lambda1 while MINIMISING it at
lambda2, by shaping a free-standing patterned high-index slab. Two competing objectives, one adjoint solve
per step.

GATE: starting from a gray slab, the optimizer INCREASES R(lambda1) and DECREASES R(lambda2) (the spectral
separation R1 - R2 grows), and the design BINARISES. Skipped if no JAX.

Run: python -m validation.fdtd_multiobjective_design
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.fdtd_nd import have_jax

LAM1, LAM2 = 1500e-9, 1300e-9          # maximise R at LAM1, minimise R at LAM2


def main():
    print("[mo2] === Multi-objective / multi-wavelength inverse design (dichroic reflector) ===", flush=True)
    if not have_jax():
        print("[mo2] JAX not installed -> SKIP (exit 0)", flush=True)
        return True
    import jax.numpy as jnp
    from dynameta.optics.inverse_design import Fdtd2dDesignProblem, weighted_objective
    from dynameta.optics.topology_opt import binarization, topology_optimize

    prob = Fdtd2dDesignProblem(period_x_m=320e-9, lambdas_m=[LAM1, LAM2], slab_thickness_m=220e-9,
                               n_des=5, eps_hi=12.0, resolution=11)

    def R1(p):  # reflectance at LAM1 (index 0)
        return prob.R(p)[0]

    def R2(p):  # reflectance at LAM2 (index 1)
        return prob.R(p)[1]

    loss = weighted_objective([{"value": R1, "sense": "max", "weight": 1.0},
                               {"value": R2, "sense": "min", "weight": 1.0}])

    nx = int(prob.eps_base.shape[0])
    rho0 = 0.5 * np.ones((nx, prob.n_des))
    R_init = np.asarray(prob.R(jnp.asarray(rho0)))
    sep_init = float(R_init[0] - R_init[1])

    rho, rho_p, hist = topology_optimize(loss, rho0, filter_radius=2.0, periodic_axes=(0,),
                                         betas=(2.0, 8.0, 16.0), steps_per_beta=8, lr=0.12)
    R_fin = np.asarray(prob.R(jnp.asarray(rho_p)))
    sep_fin = float(R_fin[0] - R_fin[1])
    binar = binarization(rho_p)

    print("[mo2] R(lambda1={:.0f}nm) {:.3f} -> {:.3f} (maximise) ; R(lambda2={:.0f}nm) {:.3f} -> {:.3f} "
          "(minimise)".format(LAM1 * 1e9, R_init[0], R_fin[0], LAM2 * 1e9, R_init[1], R_fin[1]), flush=True)
    print("[mo2] spectral separation R1-R2: {:+.3f} -> {:+.3f} ; binarization={:.2f}".format(
        sep_init, sep_fin, binar), flush=True)

    g_sep = sep_fin > sep_init + 0.05                       # the dichroic separation genuinely grew
    g_dir = (R_fin[0] > R_init[0] - 1e-3) and (R_fin[1] < R_init[1] + 1e-3)   # each objective moved the right way
    g_bin = binar > 0.7
    ok = bool(g_sep and g_dir and g_bin)
    print("[mo2] GATE (separation grows + both objectives improve + binarises): {}".format(
        "PASS" if ok else "FAIL"), flush=True)
    print("[mo2] *** MULTI-OBJECTIVE INVERSE DESIGN: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
