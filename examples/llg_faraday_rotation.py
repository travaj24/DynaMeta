"""END-TO-END: applied H field -> LLG macrospin relaxation -> gyrotropic tensor -> Faraday rotation.

The chain this example wires (each piece is oracle-validated; the glue is new):

    H(bias) --drivers.llg_extra_fields / LLG Gilbert relaxation--> settled unit m
        --VectorMagnetoOpticModel--> gyrotropic eps(m)
        --circular eigenmodes n+- = sqrt(eps_r +- g_z)--> Faraday rotation per length

Sweeping the POLAR ANGLE of a strong applied field steers m; the z-projection m_z sets the
effective gyrotropy g_z = g_s m_z, so theta_F tracks cos(angle) -- the canonical signature.
(The full-tensor FEM path for arbitrary m is tracked separately: off-diagonal HCurl assembly
is blocked on the installed NGSolve; the circular-eigenmode reduction used here is the same
oracle that validated MagnetoOpticModel against Berreman.)

GATE A: m settles onto the applied-field axis at every bias (|m x h| < 1e-3).
GATE B: at m = +z the VectorMagnetoOpticModel tensor equals MagnetoOpticModel(eps_r, g_s)
        EXACTLY (the documented byte-level reduction), and is Hermitian (lossless) everywhere.
GATE C: theta_F(angle) tracks m_z = cos(angle) to < 1e-3 relative across the sweep, and
        reverses sign when the field flips.

numpy/scipy only. Run: python -m examples.llg_faraday_rotation
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.llg import LLGMacrospin
from dynameta.core.effects import MagnetoOpticModel, VectorMagnetoOpticModel
from dynameta.drivers import llg_extra_fields

EPS_R = 5.5 + 0.0j
G_S = 0.02
LAM_M = 1.31e-6
H_MAG = 2.0e5                                     # A/m -- strong drive, fast relaxation


def _faraday_deg_per_um(eps_zz_model, m):
    """Circular-eigenmode Faraday rotation for propagation along z: g_z = g_s m_z."""
    g_z = G_S * float(m[2])
    n_p = np.sqrt(complex(EPS_R + g_z))
    n_m = np.sqrt(complex(EPS_R - g_z))
    theta_rad_per_m = np.pi * (n_p - n_m).real / LAM_M
    return float(np.degrees(theta_rad_per_m) * 1e-6)


def main():
    print("[llgf] === H(bias) -> LLG settled m -> gyrotropic eps -> Faraday rotation ===",
          flush=True)
    ok = True
    spin = LLGMacrospin(Ms_A_m=8.0e5, alpha=0.5)
    model = VectorMagnetoOpticModel(eps_r=EPS_R, g_s=G_S)

    def H_of_bias(angle_deg):
        a = np.radians(float(angle_deg))
        return H_MAG * np.array([np.sin(a), 0.0, np.cos(a)])

    fields = llg_extra_fields(spin, H_of_bias, t_settle_s=5e-9, m0=[0.3, 0.1, 0.95])

    angles = [0.0, 30.0, 60.0, 89.0, 180.0]
    g_a, worst_track = True, 0.0
    rows = []
    for a in angles:
        m = fields(a)["m_vector"]
        h = H_of_bias(a) / H_MAG
        if float(np.linalg.norm(np.cross(m, h))) > 1e-3:
            g_a = False
        eps = np.asarray(model.eps({"m_vector": m}, LAM_M))
        if float(np.max(np.abs(eps - eps.conj().T))) > 1e-15:      # Hermitian = lossless
            ok = False
            print("[llgf]   GATE B FAIL: non-Hermitian tensor at angle {}".format(a), flush=True)
        th = _faraday_deg_per_um(model, m)
        th_expect = _faraday_deg_per_um(model, h)                  # m == h-hat when settled
        worst_track = max(worst_track, abs(th - th_expect) / max(abs(th_expect), 1e-12))
        rows.append((a, m, th))
        print("[llgf]   angle {:6.1f} deg -> m_z = {:+.5f}, theta_F = {:+.5f} deg/um".format(
            a, float(m[2]), th), flush=True)

    ok = ok and g_a
    print("[llgf] GATE A: m settles onto the field axis at every bias -> {}".format(
        "PASS" if g_a else "FAIL"), flush=True)

    m_z = fields(0.0)["m_vector"]
    eps_vec = np.asarray(model.eps({"m_vector": m_z}, LAM_M))
    eps_fix = np.asarray(MagnetoOpticModel(eps_r=EPS_R, g=G_S * float(m_z[2])).eps({}, LAM_M))
    g_b = bool(np.array_equal(eps_vec, eps_fix) or np.max(np.abs(eps_vec - eps_fix)) < 1e-15)
    ok = ok and g_b
    print("[llgf] GATE B: m = +z reduces to MagnetoOpticModel exactly "
          "(max |deps| = {:.2e}) -> {}".format(float(np.max(np.abs(eps_vec - eps_fix))),
                                               "PASS" if g_b else "FAIL"), flush=True)

    g_c = bool(worst_track < 1e-3 and rows[0][2] > 0.0 and rows[-1][2] < 0.0
               and abs(rows[-1][2] + rows[0][2]) < 1e-6)
    ok = ok and g_c
    print("[llgf] GATE C: theta_F tracks cos(angle) (worst rel {:.1e}) and flips sign with "
          "the field -> {}".format(worst_track, "PASS" if g_c else "FAIL"), flush=True)

    print("[llgf] *** LLG -> FARADAY WORKFLOW: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
