"""C(T): temperature-dependent volumetric heat capacity in the transient k(T) FEM
(solve_thermal_transient_kt_fem rhoCp_of_T_by, the R21 follow-on closing the roadmap's
'C(T) out of scope' note).

Testbed: one 200 nm layer, INSULATED bottom (the new bottom_bc='insulated'), constant top
flux. rhoCp(T) = a + b (T - T0) grows ~30% over the ~100 K rise; k(T) = k0 (1 + alpha (T-T0)).

GATE A (constant-C reduction): a constant rhoCp callable reproduces the rhoCp_of_T_by=None
        path to < 1e-10 rel (solver roundoff -- the same gate class as the constant-k
        reduction; NOT byte-level, since the None path assembles M once while the C(T) path
        reassembles per step).
GATE B (chord enthalpy balance, the conservation oracle): on the insulated domain the stored
        CHORD enthalpy integral( a (T-T0) + b (T-T0)^2 / 2 ) dV must equal the injected
        flux * area * t. The lagged C(T^n) tangent scheme closes this O(dt): the residual at
        dt/4 must shrink by >= 3x vs dt and sit below 1% -- this is ALSO the gate that catches
        a frozen-at-T0 C (which would never converge to the chord balance).
GATE C (independent reference): mean temperature at t_end vs a fine explicit 1D
        finite-difference solution of C(T) dT/dt = d/dz (k(T) dT/dz) with the same BCs
        (rel-to-rise < 5e-3; budgets the L2 element-mean lag + theta error).
GATE D (guards): non-positive rhoCp(T) raises; bad bottom_bc raises.

Run: python -m validation.thermal_ct_transient
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ngsolve as ng

from dynameta.carriers.fem_mesh import _S
from dynameta.carriers.thermal_fem import ThermalLayer, solve_thermal_transient_kt_fem

T0 = 300.0
D = 200e-9
PER = 100e-9
K0, ALPHA = 1.5, 2.0e-3            # k(T) = K0 (1 + ALPHA (T - T0))   [W/(m K)]
A_C, B_C = 2.0e6, 6.0e3            # rhoCp(T) = A_C + B_C (T - T0)    [J/(m^3 K)]
FLUX = 2.0e8                       # W/m^2 -> ~100 K rise by T_END
T_END = 2.0e-7

LAYER = [ThermalLayer("slab", D, K0, rho_kg_m3=2000.0, Cp_J_kgK=1000.0)]  # rho*Cp = A_C


def k_of_T(T):
    return K0 * (1.0 + ALPHA * (np.asarray(T) - T0))


def c_of_T(T):
    return A_C + B_C * (np.asarray(T) - T0)


def _solve(dt_s, *, rhoCp=None, k=k_of_T):
    return solve_thermal_transient_kt_fem(
        LAYER, k, period_x_m=PER, period_y_m=PER, t_end_s=T_END, dt_s=dt_s,
        flux_W_m2=FLUX, T_sink_K=T0, bottom_bc="insulated", rhoCp_of_T_by=rhoCp,
        store_every=10 ** 9)


def _stored_chord_enthalpy_J(res):
    """integral over the domain of the chord enthalpy density a (T-T0) + b (T-T0)^2/2; the
    nm-mesh volume integral carries _S^3, so divide it back out."""
    dT = res.T_final - T0
    return float(ng.Integrate(A_C * dT + 0.5 * B_C * dT * dT, res.mesh).real) / _S ** 3


def _fd_reference(t_end, nz=401):
    """Explicit 1D FD of C(T) dT/dt = d/dz(k(T) dT/dz), insulated bottom, flux top."""
    dz = D / (nz - 1)
    dt = 0.2 * dz * dz * A_C / (2.0 * K0 * (1.0 + ALPHA * 150.0))      # stability w/ margin
    nst = int(np.ceil(t_end / dt))
    dt = t_end / nst
    T = np.full(nz, T0)
    for _ in range(nst):
        k_face = 0.5 * (k_of_T(T[1:]) + k_of_T(T[:-1]))                # interior faces
        q = k_face * (T[1:] - T[:-1]) / dz                             # heat flux up the grid
        div = np.zeros(nz)
        div[1:-1] = (q[1:] - q[:-1]) / dz
        div[0] = q[0] / (0.5 * dz)                                     # insulated bottom
        div[-1] = (FLUX - q[-1]) / (0.5 * dz)                          # injected flux top
        T = T + dt * div / c_of_T(T)
    # volume mean with half-weight end nodes (trapezoid)
    w = np.ones(nz); w[0] = w[-1] = 0.5
    return float(np.sum(w * T) / np.sum(w))


def main():
    print("[ct] === C(T) transient thermal FEM vs reduction + enthalpy + FD oracles ===",
          flush=True)
    ok = True
    dt = 2.0e-9

    # ---- GATE A: constant-C callable reduces to the None path ----
    r_none = _solve(dt, rhoCp=None)
    r_const = _solve(dt, rhoCp=lambda T: A_C * np.ones_like(np.asarray(T, dtype=np.float64)))
    dA = float(np.max(np.abs(r_const.mean_T_per_layer_t - r_none.mean_T_per_layer_t))
               / np.max(r_none.mean_T_per_layer_t - T0))
    g_a = bool(dA < 1e-10)
    ok = ok and g_a
    print("[ct] GATE A: constant-C callable == rhoCp_of_T_by=None path (rel {:.1e}) -> {}"
          .format(dA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: chord enthalpy balance closes O(dt) on the insulated domain ----
    injected = FLUX * PER * PER * T_END
    errs = []
    for dt_b in (dt, dt / 4.0):
        r = _solve(dt_b, rhoCp=c_of_T)
        stored = _stored_chord_enthalpy_J(r)
        errs.append(abs(stored - injected) / injected)
        print("[ct]   dt = {:.1e} s: stored/injected = {:.6f} (residual {:.2e})".format(
            dt_b, stored / injected, errs[-1]), flush=True)
    g_b = bool(errs[1] < 0.01 and errs[0] / max(errs[1], 1e-12) > 3.0)
    ok = ok and g_b
    print("[ct] GATE B: chord-enthalpy residual {:.2e} -> {:.2e} under dt/4 (ratio {:.1f}) "
          "-> {}".format(errs[0], errs[1], errs[0] / max(errs[1], 1e-12),
                         "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: independent fine-FD reference at t_end ----
    r_ct = _solve(dt / 4.0, rhoCp=c_of_T)
    T_fem = float(r_ct.mean_T_per_layer_t[-1, 0])
    T_fd = _fd_reference(T_END)
    rise = T_fd - T0
    dC = abs(T_fem - T_fd) / rise
    g_c = bool(dC < 5e-3)
    ok = ok and g_c
    print("[ct] GATE C: mean T(t_end) FEM {:.3f} K vs FD {:.3f} K (rel-to-rise {:.1e}) -> {}"
          .format(T_fem, T_fd, dC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: guards ----
    g_d = False
    try:
        _solve(dt, rhoCp=lambda T: A_C - 1e9 * np.ones_like(np.asarray(T)))
    except ValueError:
        g_d = True
    g_d2 = False
    try:
        solve_thermal_transient_kt_fem(LAYER, k_of_T, period_x_m=PER, period_y_m=PER,
                                       t_end_s=T_END, dt_s=dt, bottom_bc="adiabatic-typo")
    except ValueError:
        g_d2 = True
    g_d = g_d and g_d2
    ok = ok and g_d
    print("[ct] GATE D: non-positive rhoCp raises; bad bottom_bc raises -> {}".format(
        "PASS" if g_d else "FAIL"), flush=True)

    print("[ct] *** C(T) TRANSIENT THERMAL: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
