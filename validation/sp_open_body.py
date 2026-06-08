"""Validate the OPEN / TRANSPARENT body boundary (bulk_buffer_m) of the self-consistent
Schrodinger-Poisson solver. A degenerate semiconductor body is a contact into the semi-infinite bulk,
so its local density of states is the 3D bulk LDOS and the carrier density there must recover the bulk
n_bg with NO boundary feature. The two hard boundaries both FAIL this:
  * Dirichlet body (psi=0) forces a node -> a ~1 nm DEAD LAYER (density dips to ~0 at the body),
  * Neumann body (dpsi/dz=0) forces an antinode -> a density PILE-UP at the body.
The open BC prepends a field-free bulk buffer and solves on the extended grid (Dirichlet far end = bulk
reference), so the physical body node is interior, far from any wall, and recovers n_bg flat.

E_F is set to the BULK Fermi level for n_bg (eta_bg = invert_F12(n_bg/N_c), the same 3D degenerate
relation the classical reference uses), so a correct solve gives the field-free interior density = n_bg.

GATE A (open body recovers n_bg, no boundary layer): on a field-free degenerate slab the interior
        density equals n_bg (< 5%) AND the near-body node density equals the interior (|ratio-1| < 0.10)
        -- no pile-up, no dead layer.
GATE B (it fixes BOTH known pathologies -- bidirectional proof): the SAME slab with a Neumann body shows
        a pile-up (near-body/interior > 1.3) and with a Dirichlet body shows a dead layer (< 0.5); the
        open body removes both.

Run: python -m validation.sp_open_body
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D
from dynameta.carriers.physics_equilibrium import invert_F12
from dynameta.constants import KB, HBAR, M_E

M = 0.35 * M_E
T = 300.0
N_BG = 1.0e26
EPS_R = 9.5
L = 16e-9
NZ = 65


def main():
    warnings.filterwarnings("ignore")          # the SP non-convergence warning is not under test here
    print("[ob] === Schrodinger-Poisson OPEN/TRANSPARENT body BC (bulk buffer) ===", flush=True)
    kT = KB * T
    N_c = 2.0 * (M * kT / (2.0 * np.pi * HBAR ** 2)) ** 1.5
    E_F = invert_F12(N_BG / N_c) * kT            # bulk Fermi level for n_bg (3D degenerate)
    z = np.linspace(0.0, L, NZ)
    dop = np.full(NZ, N_BG)
    sp = SchrodingerPoisson1D(z, M, T_K=T, g_s=2, g_v=1)
    kw = dict(eps_r=EPS_R, doping_m3=dop, E_F_J=E_F, phi_left_V=0.0, phi_right_V=0.0,
              max_outer=40, tol_V=1e-4, bound_tol=1e9, relax=0.8)
    interior = (z > 6e-9) & (z < 10e-9)

    def near_body_ratio(n):
        ni = float(np.median(n[interior]))
        return float(n[1] / ni), ni              # node[1] is the first node off the body wall

    _, nD, _ = sp.solve_self_consistent(**kw)                          # Dirichlet body (dead layer)
    _, nN, _ = sp.solve_self_consistent(neumann_left=True, **kw)       # Neumann body (pile-up)
    _, nO, _ = sp.solve_self_consistent(bulk_buffer_m=10e-9, **kw)     # OPEN body (bulk buffer)
    rD, _ = near_body_ratio(nD)
    rN, _ = near_body_ratio(nN)
    rO, niO = near_body_ratio(nO)

    g_a = (abs(niO / N_BG - 1.0) < 0.05) and (abs(rO - 1.0) < 0.10)
    print("[ob] A open body: interior/n_bg={:.3f} (->1) ; near-body/interior={:.3f} (->1, no "
          "pile-up/dead-layer) -> {}".format(niO / N_BG, rO, "OK" if g_a else "FAIL"), flush=True)

    g_b = (rN > 1.3) and (rD < 0.5) and (abs(rO - 1.0) < 0.10)
    print("[ob] B fixes both: Neumann near-body/interior={:.3f} (>1.3 PILE-UP), Dirichlet={:.3f} "
          "(<0.5 DEAD LAYER), OPEN={:.3f} (flat) -> {}".format(rN, rD, rO, "OK" if g_b else "FAIL"),
          flush=True)

    ok = g_a and g_b
    print("[ob] *** SCHRODINGER-POISSON OPEN BODY BC: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
