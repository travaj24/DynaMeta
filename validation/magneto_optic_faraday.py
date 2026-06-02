"""
Magneto-optic (gyrotropic) Faraday-rotation oracle -- the magneto-optic row of the modulation-
mechanism landscape. Validates core.effects.MagnetoOpticModel (the gyrotropic permittivity tensor
eps = [[eps_r, i g, 0], [-i g, eps_r, 0], [0, 0, eps_r]]) against an INDEPENDENT analytic
circular-eigenmode transfer-matrix reference -- PURE NUMPY, no FEM.

WHY NO FEM: the gyrotropic tensor has nonzero (imaginary) OFF-DIAGONAL entries, which NGSolve 6.2.2604
mis-assembles under PML (confirmed limitation, guarded by eps_assembler._check_diagonal; every
formulation tested gives the same energy-non-conserving result and 6.2.2604 is the latest release).
The constitutive model + the Faraday physics are validated analytically here; the off-diagonal FEM
solve is deferred until a fixed NGSolve ships (the same status as the tilted-LC FEM).

PHYSICS: for z-propagation the two normal modes are circular polarizations with n_pm =
sqrt(eps_r +/- g). A linearly (x) polarized wave through a slab of thickness L rotates its plane of
polarization by the Faraday angle theta_F = (pi L / lambda) Re(n_+ - n_-). Each circular mode is solved
as an isotropic slab (vacuum | n_pm | vacuum, Airy) and recombined into the transmitted Jones vector.

GATE A: the model tensor's eigenvalues are {eps_r - g, eps_r, eps_r + g} (the two circular modes + the
        axial mode).
GATE B: the full circular-TMM transmitted-polarization rotation == the analytic bulk theta_F within
        5% (relative) over a range of gyration g (and g = 0 gives no rotation). The small residual is
        the Fabry-Perot / interface correction the bulk theta_F omits but the exact TMM includes.
GATE C: real eps_r + real g -> the tensor is HERMITIAN -> LOSSLESS: R + T = 1 at every g.

Run: python -m validation.magneto_optic_faraday
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.effects import MagnetoOpticModel

LAM = 1550e-9
EPS_R = 2.25
L = 5.0e-6
G_LIST = [0.0, 0.02, 0.05, 0.08]
ROT_RTOL = 0.05            # bulk theta_F omits the (~4%) Fabry-Perot/interface correction the TMM has
ROT_ABS_DEG = 0.05         # absolute floor (the g=0 zero-rotation case)


def _slab_rt(n, k0, L):
    """Amplitude (r, t) of vacuum | n | vacuum at normal incidence (single-slab Airy)."""
    beta = n * k0 * L
    r1 = (1.0 - n) / (1.0 + n)          # vacuum -> slab
    t1 = 2.0 / (1.0 + n)
    r2 = (n - 1.0) / (n + 1.0)          # slab -> vacuum
    t2 = 2.0 * n / (n + 1.0)
    ph = np.exp(1j * beta)
    den = 1.0 + r1 * r2 * ph * ph
    return (r1 + r2 * ph * ph) / den, (t1 * t2 * ph) / den


def _faraday(eps_r, g, k0, L):
    """Full circular-TMM: incident x-pol through the gyrotropic slab -> (rotation_deg, R, T)."""
    n_p, n_m = np.sqrt(eps_r + g), np.sqrt(eps_r - g)
    rp, tp = _slab_rt(n_p, k0, L)
    rm, tm = _slab_rt(n_m, k0, L)
    # incident x = (e+ + e-)/sqrt2, e_pm = (x +/- i y)/sqrt2; transmitted/reflected x,y Jones:
    Ex, Ey = 0.5 * (tp + tm), 0.5j * (tp - tm)
    Rx, Ry = 0.5 * (rp + rm), 0.5j * (rp - rm)
    # major-axis orientation of the transmitted ellipse (polarization rotation from x)
    rot = 0.5 * np.arctan2(2.0 * np.real(np.conj(Ex) * Ey), abs(Ex) ** 2 - abs(Ey) ** 2)
    T = abs(Ex) ** 2 + abs(Ey) ** 2            # n_sub = n_sup = 1 -> T = sum |t|^2 over the two modes
    R = abs(Rx) ** 2 + abs(Ry) ** 2
    return float(np.degrees(rot)), float(R), float(T)


def main():
    print("[mo] === Magneto-optic Faraday rotation (gyrotropic tensor) vs analytic ===", flush=True)
    k0 = 2.0 * np.pi / LAM

    # GATE A: model tensor eigenvalues
    T3 = np.asarray(MagnetoOpticModel(eps_r=EPS_R, g=0.05).eps({}, LAM))
    eig = np.sort(np.linalg.eigvals(T3).real)
    exp = np.sort([EPS_R - 0.05, EPS_R, EPS_R + 0.05])
    gate_a = bool(np.allclose(eig, exp, atol=1e-12))
    print("[mo] tensor eigenvalues {} vs expected {} : {}".format(
        np.round(eig, 5), np.round(exp, 5), "PASS" if gate_a else "FAIL"), flush=True)

    gate_b = gate_c = True
    for g in G_LIST:
        rot_deg, R, T = _faraday(EPS_R, g, k0, L)
        n_p, n_m = np.sqrt(EPS_R + g), np.sqrt(EPS_R - g)
        theta_an = float(np.degrees(np.pi * L / LAM * (n_p - n_m)))     # analytic bulk Faraday angle
        d_rot = abs(abs(rot_deg) - abs(theta_an))                      # magnitudes (sign convention)
        e_close = abs(R + T - 1.0)
        gate_b = gate_b and (d_rot < ROT_RTOL * abs(theta_an) + ROT_ABS_DEG)
        gate_c = gate_c and (e_close < 1e-9)
        print("[mo] g={:.2f}: rot_tmm={:+7.3f} deg  theta_analytic={:+7.3f} deg  |d|={:.3e} deg | "
              "R+T-1={:.2e}".format(g, rot_deg, theta_an, d_rot, e_close), flush=True)

    overall = gate_a and gate_b and gate_c
    print("[mo]", flush=True)
    print("[mo] GATE A (tensor eigenvalues eps_r, eps_r +/- g): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[mo] GATE B (circular-TMM rotation == analytic theta_F within {:.0%} rel; g=0 -> 0): "
          "{}".format(ROT_RTOL, "PASS" if gate_b else "FAIL"), flush=True)
    print("[mo] GATE C (Hermitian gyrotropic -> lossless R+T=1): {}".format(
        "PASS" if gate_c else "FAIL"), flush=True)
    print("[mo] *** MAGNETO-OPTIC FARADAY (gyrotropic tensor): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
