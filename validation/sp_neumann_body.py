"""Validate the optional NEUMANN BODY boundary for the Schrodinger-Poisson solve. The body side
(z=0) of the accumulation slab is, by default, a Dirichlet hard wall (psi=0), which forces an
unphysical ~0.4nm DEAD LAYER (n -> 0) at the body -- but the body is a contact into the neutral bulk,
not an infinite barrier. neumann_left=True makes the SCHRODINGER body a zero-flux (Neumann) boundary
(psi need not vanish) while the POISSON keeps its Dirichlet body reference (so the body density is the
bulk n_bg, not a floating value).

GATE A: both BCs CONVERGE.
GATE B: the Dirichlet body dead layer (n[0] ~ 0) is REMOVED by Neumann (n[0] is O(n_bg)).
GATE C: the GATE-side accumulation is UNCHANGED (the body BC does not perturb the gate-relevant
        physics) -- |dn|/peak < 1e-3 over the interior+gate region.
GATE D: the bulk (mid-slab) recovers n_bg for BOTH.

HONEST NOTE (printed): a HARD Neumann wall has its own quantum boundary layer -- the box eigenmodes
peak at a Neumann end, so n is somewhat ENHANCED right at z=0 (a pile-up) rather than exactly n_bg. So
Neumann trades the Dirichlet depletion DIP for a Neumann pile-up; both are confined to ~the Fermi
wavelength at the body and are far from the gate-side ENZ. A truly artifact-free body needs an OPEN
(transparent) boundary -- a further refinement. Run: python -m validation.sp_neumann_body
"""
import sys, os
import warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.schrodinger_poisson import SchrodingerPoisson1D, Q, HBAR, M_E

N_BG = 4e26
M = 0.35 * M_E
GS, GV = 2, 1
PSI_S = 0.5            # gate-side surface potential
EPS_R = 9.5


def _solve(sp, nz, neu, psi_s):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gammaF = (HBAR ** 2 / (2.0 * M)) * (6.0 * np.pi ** 2 * N_BG / (GS * GV)) ** (2.0 / 3.0)
        phi, n, res = sp.solve_self_consistent(
            eps_r=EPS_R, doping_m3=np.full(nz, N_BG), E_F_J=gammaF, phi_left_V=0.0, phi_right_V=psi_s,
            n_states=80, bound_tol=1e9, max_outer=80, tol_V=1e-5, neumann_left=neu)
    return n, bool(res.converged)


def main():
    print("[t] === Schrodinger-Poisson NEUMANN BODY boundary ===", flush=True)
    L, nz = 14e-9, 281
    sp = SchrodingerPoisson1D(np.linspace(0.0, L, nz), M, T_K=300.0, g_s=GS, g_v=GV)
    nD, cD = _solve(sp, nz, False, PSI_S)
    nN, cN = _solve(sp, nz, True, PSI_S)
    mid = nz // 2

    g_a = bool(cD and cN)
    g_b = bool(nD[0] < 0.05 * N_BG and nN[0] > 0.5 * N_BG)        # Dirichlet dip removed by Neumann
    # The body boundary layer spans ~the Fermi wavelength (~3 nm here). The GATE-side accumulation
    # (top of the slab, well away from the body) must be unperturbed -- compare the gate half z>0.6 L.
    gate = slice(int(0.6 * nz), nz)
    gate_rel = float(np.max(np.abs(nN[gate] - nD[gate])) / max(nN.max(), nD.max()))
    g_c = gate_rel < 1e-3
    g_d = bool(abs(nD[mid] / N_BG - 1.0) < 0.02 and abs(nN[mid] / N_BG - 1.0) < 0.02)

    print("[t] body n[0]/n_bg: Dirichlet={:.3f} (dead layer)  Neumann={:.3f} (dead layer removed)".format(
        nD[0] / N_BG, nN[0] / N_BG), flush=True)
    print("[t] GATE-side (z>0.6L) max|dn|/peak = {:.2e} (Neumann body does not perturb the gate): {}".format(
        gate_rel, "OK" if g_c else "TOO BIG"), flush=True)
    print("[t] bulk(mid)/n_bg: Dirichlet={:.3f}  Neumann={:.3f}".format(nD[mid] / N_BG, nN[mid] / N_BG),
          flush=True)
    print("[t] NOTE: Neumann n[0]/n_bg={:.2f} > 1 is the hard-Neumann-wall pile-up (cos modes peak at "
          "the Neumann end) -- trades the Dirichlet dip for a pile-up; both ~1 Fermi-wavelength, far "
          "from the gate ENZ.".format(nN[0] / N_BG), flush=True)
    ok = bool(g_a and g_b and g_c and g_d)
    print("[t] GATE A converges: {} ; B dead-layer removed: {} ; C gate unchanged: {} ; D bulk recovers: {}"
          .format(cD and cN, g_b, g_c, g_d), flush=True)
    print("[t] *** SP NEUMANN BODY BOUNDARY: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
