"""INDEPENDENT R/T/A oracles for the test suite -- hand-written, NOT the `tmm` package and not
any dynameta solver, so an energy/ordering gate built on them is never self-referential.

Why this module exists (audit T-1): every R/T/A producer in the repo computes ``A := 1 - R - T``,
so ``assert R + T + A == 1`` is an algebraic identity that passes for a halved, doubled or
sign-flipped T. A gate on the energy budget therefore has to compare against a quantity computed
some OTHER way -- either an independent solve (this module) or an independent absorption
(``tmm.absorp_in_each_layer`` / a volume integral / a Poynting deficit). One home for the
independent TMM so the three modules that need it (``test_tmm_reference``, ``test_solver_guards``,
``test_mermin``) share exactly one implementation.

Convention exp(-i omega t): forward wave ~ exp(+i kz z), Im(eps) >= 0 = loss, so the per-layer
characteristic matrix mapping (E_tan, H_tan) from the layer's TOP face to its BOTTOM face is
  m = [[cos q, -i sin q / Y], [-i Y sin q, cos q]],  q = kz d,  Y = kz (s-pol) or eps/kz (p-pol).
"""
import numpy as np


def abeles_rta(n_super, layers_super_first, n_sub, lambda_m, *, theta_deg=0.0, pol="s"):
    """(R, T, A) for super | layers | sub via the Abeles characteristic-matrix TMM.

    ``layers_super_first`` is an ordered [(n, thickness_m), ...] from the superstrate side to the
    substrate side -- the same order ``tmm_reference.stack_rta`` takes. ``n_super`` must be real
    (a lossless incidence medium) for R/T/A to be defined. Here A IS ``1 - R - T``, but R and T
    come from a wholly independent solve, so comparing (R, T, A) against a dynameta producer's
    (R, T, A) is a genuine three-quantity gate.
    """
    k0 = 2.0 * np.pi / float(lambda_m)
    eps_s, eps_b = complex(n_super) ** 2, complex(n_sub) ** 2
    k_par = complex(n_super) * k0 * np.sin(np.radians(float(theta_deg)))

    def kz(eps):
        return np.sqrt(complex(eps) * k0 * k0 - k_par * k_par + 0j)

    def adm(eps):
        return kz(eps) if pol == "s" else complex(eps) / kz(eps)

    M = np.eye(2, dtype=np.complex128)
    for n_j, d_j in layers_super_first:
        eps_j = complex(n_j) ** 2
        q = kz(eps_j) * float(d_j)
        Y = adm(eps_j)
        M = M @ np.array([[np.cos(q), -1j * np.sin(q) / Y],
                          [-1j * Y * np.sin(q), np.cos(q)]], dtype=np.complex128)
    Ys, Yb = adm(eps_s), adm(eps_b)
    B = M[0, 0] + M[0, 1] * Yb                       # [B; C] = M @ [1; Y_sub]
    C = M[1, 0] + M[1, 1] * Yb
    r = (Ys * B - C) / (Ys * B + C)
    t = 2.0 * Ys / (Ys * B + C)
    R = float(abs(r) ** 2)
    T = float((Yb.real / Ys.real) * abs(t) ** 2)
    return R, T, 1.0 - R - T
