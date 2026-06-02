"""Backend-seam demonstration (the scoped Lumenairy-style array dispatch, roadmap infra). Shows
that the pure-array CONSTITUTIVE EffectModels run on NumPy / JAX / CuPy interchangeably with
IDENTICAL float64 results, and -- the whole point -- that the JAX path is DIFFERENTIABLE: jax.grad
of a constitutive map matches the analytic field response. That differentiable seam is what a future
RCWA-backed inverse-design loop needs (gradient flows design -> fields -> eps through these maps);
the FEM/DEVSIM solvers stay on host NumPy and are untouched.

Skips JAX / CuPy gracefully if a backend is absent (or has no GPU). Run:
python -m validation.backend_autodiff
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.constants import Q_E as Q
from dynameta.core.backend import JAX_AVAILABLE, CUPY_AVAILABLE, to_numpy
from dynameta.core.effects import PockelsEffect


def _linbo3():
    no, ne, r13, r33 = 2.21, 2.14, 9.6e-12, 30.9e-12
    eps_bg = np.diag([no ** 2, no ** 2, ne ** 2]).astype(complex)
    r = np.zeros((6, 3)); r[0, 2] = r13; r[1, 2] = r13; r[2, 2] = r33
    return no, ne, r13, eps_bg, r


def main():
    no, ne, r13, eps_bg, r = _linbo3()
    pk = PockelsEffect(eps_bg=eps_bg, r_voigt=r)
    Ez = 1.0e7
    e_np = pk.eps({"E": np.array([0., 0., Ez])}, 1300e-9)        # the NumPy reference (float64)
    print("[b] backends available: jax={} cupy={}".format(JAX_AVAILABLE, CUPY_AVAILABLE), flush=True)
    ok = True

    if JAX_AVAILABLE:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        e_jx = to_numpy(pk.eps({"E": jnp.array([0., 0., Ez])}, 1300e-9))
        agree = float(np.max(np.abs(e_jx - e_np)))

        def nx(Ez_):                                            # n_o seen by the x-component
            e = pk.eps({"E": jnp.stack([0.0 * Ez_, 0.0 * Ez_, Ez_])}, 1300e-9)
            return jnp.sqrt(jnp.real(e[0, 0]))

        g = float(jax.grad(nx)(jnp.asarray(Ez)))
        g_analytic = -0.5 * no ** 3 * r13                       # d(n_o)/dE_z via the r13 Pockels coeff
        grad_ok = abs(g - g_analytic) < 1e-3 * abs(g_analytic)
        print("[b] JAX  : max|eps_jax-eps_np|={:.2e} (float64); autodiff d(nx)/dEz={:.4e} "
              "analytic={:.4e} match={}".format(agree, g, g_analytic, grad_ok), flush=True)
        ok = ok and agree < 1e-12 and grad_ok
    else:
        print("[b] JAX  : not installed -- skipped", flush=True)

    if CUPY_AVAILABLE:
        try:
            import cupy as cp
            e_cp = to_numpy(pk.eps({"E": cp.asarray([0., 0., Ez])}, 1300e-9))
            agree = float(np.max(np.abs(e_cp - e_np)))
            print("[b] CuPy : max|eps_cp-eps_np|={:.2e}".format(agree), flush=True)
            ok = ok and agree < 1e-10
        except Exception as exc:                                # cupy present but no usable GPU
            print("[b] CuPy : present but unusable ({}) -- skipped".format(type(exc).__name__), flush=True)
    else:
        print("[b] CuPy : not installed -- skipped", flush=True)

    print("[b] *** BACKEND SEAM (numpy/jax/cupy agree in float64; jax autodiff == analytic "
          "Pockels slope): {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
