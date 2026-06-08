"""COUPLED carrier <-> optics transient: a gate voltage step drives the ITO carrier accumulation n(t)
(the access-RC charging model), which sweeps the free-carrier Drude eps(t) through ENZ, which modulates the
reflective stack -> the modulator's optical turn-on WAVEFORM R(t). This validates the dynamic loop the
steady-state pipeline cannot produce: not just the DC OFF/ON contrast, but the time response.

GATES:
  1  SETTLES: R(t) relaxes to the independent ON steady-state R(n_on) (the transient is consistent with the
     DC endpoints R_off at t=0 and R_on at t->inf), with a real ON/OFF contrast.
  2  RESPONSE TIME: the optical 10-90% rise time is of order the carrier RC time constant tau (the device
     speed is set by the carrier dynamics, not instantaneous) and the waveform is monotonic.
  3  ENZ: the front-ITO permittivity crosses toward ENZ (Re(eps) drops below ~1) during the turn-on.

Run: python -m validation.transient_optics_response
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.materials import DrudeOptical, M_E
from dynameta.transient_optics import enz_reflector_stack, optical_transient_response, rc_accumulation
from dynameta.optics.tmm_reference import layered_rta

LAM = 1550e-9
ITO = DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * M_E, gamma_rad_s=1.0e14)
N_OFF, N_ON = 4.0e26, 1.5e27          # m^-3: background -> strong accumulation (ENZ-crossing)
TAU = 12e-12                          # access-RC time constant (12 ps)


def _R_steady(n_m3):
    eps = complex(ITO.eps(LAM, n_m3=n_m3))
    R, _T, _A = layered_rta(enz_reflector_stack(eps, LAM), LAM)
    return R


def main():
    print("[tr] === Coupled carrier->optics transient (RC accumulation -> R(t)) ===", flush=True)
    times = np.linspace(0.0, 6.0 * TAU, 80)

    def n_of_t(t):
        return rc_accumulation(t, N_OFF, N_ON, TAU)        # homogeneous ITO density at time t

    t, R, T, eps_front = optical_transient_response(times, n_of_t, LAM, drude_model=ITO)
    R_off, R_on = _R_steady(N_OFF), _R_steady(N_ON)
    contrast = abs(R_on - R_off)

    # GATE 1: settle to the DC endpoints + a real contrast
    settled = abs(R[-1] - R_on) < 0.02 * max(contrast, 1e-3) + 5e-3 and abs(R[0] - R_off) < 5e-3
    g1 = bool(settled and contrast > 0.05)

    # GATE 2: 10-90% optical rise time is of order tau, waveform monotonic
    lo, hi = R_off + 0.1 * (R_on - R_off), R_off + 0.9 * (R_on - R_off)
    sgn = 1.0 if R_on > R_off else -1.0
    cross = (lambda lev: float(np.interp(sgn * lev, sgn * R, t)))
    t_rise = cross(hi) - cross(lo)
    monotonic = bool(np.all(np.diff(sgn * R) >= -1e-9))
    g2 = bool(monotonic and 0.5 * TAU < t_rise < 6.0 * TAU)

    # GATE 3: the front eps crosses toward ENZ on accumulation
    g3 = bool(eps_front[0].real > 1.0 and eps_front[-1].real < 1.0)

    print("[tr]   R_off={:.4f}  R_on={:.4f}  contrast={:.4f} ; R(0)={:.4f} R(end)={:.4f}".format(
        R_off, R_on, contrast, R[0], R[-1]), flush=True)
    print("[tr]   front eps {:.3f} -> {:.3f} (ENZ crossing) ; optical 10-90%% rise = {:.1f} ps "
          "(tau = {:.1f} ps)".format(eps_front[0], eps_front[-1], t_rise * 1e12, TAU * 1e12), flush=True)
    print("[tr]   GATE1 settles to DC endpoints + contrast: {}".format("PASS" if g1 else "FAIL"), flush=True)
    print("[tr]   GATE2 rise time ~ RC tau + monotonic: {}".format("PASS" if g2 else "FAIL"), flush=True)
    print("[tr]   GATE3 front eps crosses ENZ: {}".format("PASS" if g3 else "FAIL"), flush=True)
    ok = g1 and g2 and g3
    print("[tr] *** COUPLED CARRIER->OPTICS TRANSIENT (turn-on waveform): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
