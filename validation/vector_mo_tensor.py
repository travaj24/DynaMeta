"""R13 full-vector gyrotropic tensor oracle. VectorMagnetoOpticModel builds
eps_ij = eps_r delta_ij + i epsilon_ijk g_k for an ARBITRARY gyration vector g = g_s m.

GATE A (reduces-to-known-limit): m = z-hat reproduces the shipped Faraday-TMM-validated
        MagnetoOpticModel(eps_r, g_s) tensor EXACTLY (0.0 -- same formula, same convention).
GATE B (independent construction): every component matches a brute-force Levi-Civita sum
        eps_ij = eps_r d_ij + i sum_k epsilon_ijk g_k built from an explicitly-tabulated
        epsilon_ijk, for random unit m (machine).
GATE C (rotation equivariance, the vector-physics gate): for rotations R (z->x, z->y, and a random
        axis-angle), eps(R m) == R eps(m) R^T to machine -- the tensor transforms as a rank-2 tensor
        under the SAME rotation as the magnetization, which the component formula cannot fake.
GATE D (Hermiticity + grid): real g -> eps Hermitian (lossless, Im eigenvalues 0); a GRIDDED (...,3)
        m field yields (...,3,3) with each point matching the scalar evaluation.

Run: python -m validation.vector_mo_tensor
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.effects import MagnetoOpticModel, VectorMagnetoOpticModel

EPS_R, GS = 4.84, 0.07


def _rot_axis_angle(axis, theta):
    a = np.asarray(axis, dtype=float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def main():
    print("[vm] === R13 full-vector gyrotropic tensor ===", flush=True)
    ok = True
    vm = VectorMagnetoOpticModel(eps_r=EPS_R, g_s=GS)

    # ---- GATE A: z-hat anchor == the shipped z-axis model EXACTLY ----
    Tz = np.asarray(vm.eps({"m_vector": np.array([0.0, 0.0, 1.0])}, 1550e-9))
    T0 = np.asarray(MagnetoOpticModel(eps_r=EPS_R, g=GS).eps({"magnetization": 1.0}, 1550e-9))
    dA = float(np.max(np.abs(Tz - T0)))
    g_a = bool(dA == 0.0)
    ok = ok and g_a
    print("[vm] GATE A: m=z-hat == MagnetoOpticModel max|d| = {:.1e} -> {}".format(
        dA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: brute-force Levi-Civita construction ----
    lc = np.zeros((3, 3, 3))
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        lc[i, j, k] = 1.0
        lc[j, i, k] = -1.0
    rng = np.random.default_rng(11)
    dB = 0.0
    for _ in range(8):
        m = rng.normal(size=3); m /= np.linalg.norm(m)
        T = np.asarray(vm.eps({"m_vector": m}, 1550e-9))
        T_ref = EPS_R * np.eye(3) + 1j * np.einsum("ijk,k->ij", lc, GS * m)
        dB = max(dB, float(np.max(np.abs(T - T_ref))))
    g_b = bool(dB < 1e-15)
    ok = ok and g_b
    print("[vm] GATE B: component-wise == eps_r I + i eps_ijk g_k (8 random m): max|d| = {:.1e} "
          "-> {}".format(dB, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: rotation equivariance eps(R m) == R eps(m) R^T ----
    dC = 0.0
    m0 = np.array([0.0, 0.0, 1.0])
    rots = [_rot_axis_angle([0, 1, 0], np.pi / 2),          # z -> x
            _rot_axis_angle([1, 0, 0], -np.pi / 2),         # z -> y
            _rot_axis_angle([1, 2, 3], 0.7)]                # generic
    for R in rots:
        T_rot_m = np.asarray(vm.eps({"m_vector": R @ m0}, 1550e-9))
        T_conj = R @ np.asarray(vm.eps({"m_vector": m0}, 1550e-9)) @ R.T
        dC = max(dC, float(np.max(np.abs(T_rot_m - T_conj))))
    g_c = bool(dC < 1e-14)
    ok = ok and g_c
    print("[vm] GATE C: eps(R m) == R eps(m) R^T over 3 rotations: max|d| = {:.1e} -> {}".format(
        dC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: Hermitian + gridded m ----
    m = rng.normal(size=3); m /= np.linalg.norm(m)
    T = np.asarray(vm.eps({"m_vector": m}, 1550e-9))
    herm = float(np.max(np.abs(T - T.conj().T)))
    mg = rng.normal(size=(4, 5, 3))
    mg /= np.linalg.norm(mg, axis=-1, keepdims=True)
    Tg = np.asarray(vm.eps({"m_vector": mg}, 1550e-9))
    d_grid = max(float(np.max(np.abs(np.asarray(vm.eps({"m_vector": mg[i, j]}, 1550e-9)) - Tg[i, j])))
                 for i in range(4) for j in range(5))
    g_d = bool(herm < 1e-15 and Tg.shape == (4, 5, 3, 3) and d_grid == 0.0)
    ok = ok and g_d
    print("[vm] GATE D: Hermitian |T - T^dag| = {:.1e}; gridded (4,5,3) -> (4,5,3,3) pointwise exact "
          "-> {}".format(herm, "PASS" if g_d else "FAIL"), flush=True)

    print("[vm] *** R13 VECTOR GYROTROPIC TENSOR: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
