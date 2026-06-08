"""Fast unit tests for the coupled carrier->optics transient (TMM-based, no FDTD/DEVSIM)."""
import numpy as np

from dynameta.materials import DrudeOptical, M_E
from dynameta.transient_optics import (enz_reflector_stack, optical_transient_response, rc_accumulation)

ITO = DrudeOptical(eps_inf=3.9, m_opt_kg=0.35 * M_E, gamma_rad_s=1.0e14)
LAM = 1550e-9


def test_rc_accumulation_endpoints_and_monotonic():
    t = np.linspace(0.0, 1e-10, 50)
    n = rc_accumulation(t, 4e26, 1.5e27, 12e-12)
    assert abs(n[0] - 4e26) < 1e20                          # starts at n_off
    assert abs(n[-1] - 1.5e27) < 0.02 * 1.5e27              # ~saturated to n_on
    assert np.all(np.diff(n) >= -1e18)                      # monotonic charging
    assert abs(float(rc_accumulation(0.0, 4e26, 1.5e27, 12e-12)) - 4e26) < 1e20   # scalar t works


def test_enz_reflector_stack_grades_ito():
    eps = np.array([-2.0 + 0.5j, 1.0 + 0.3j, 2.5 + 0.2j])   # a 3-sublayer ITO depth profile
    stack = enz_reflector_stack(eps, LAM, t_ito_m=12e-9)
    assert len(stack.slabs) == 4                            # 3 ITO sublayers + oxide
    assert abs(stack.slabs[0].thickness_m - 4e-9) < 1e-15   # 12nm / 3


def test_optical_transient_settles_and_crosses_enz():
    tau = 12e-12
    times = np.linspace(0.0, 6 * tau, 60)
    t, R, T, eps_front = optical_transient_response(
        times, lambda ti: rc_accumulation(ti, 4e26, 1.5e27, tau), LAM, drude_model=ITO)
    # endpoints == the DC steady states
    R_off = enz_reflector_stack(complex(ITO.eps(LAM, n_m3=4e26)), LAM)
    from dynameta.optics.tmm_reference import layered_rta
    R_off = layered_rta(R_off, LAM)[0]
    R_on = layered_rta(enz_reflector_stack(complex(ITO.eps(LAM, n_m3=1.5e27)), LAM), LAM)[0]
    assert abs(R[0] - R_off) < 5e-3 and abs(R[-1] - R_on) < 1e-2
    assert abs(R_on - R_off) > 0.03                         # a real modulation contrast
    assert eps_front[0].real > 1.0 and eps_front[-1].real < 1.0   # ENZ crossing on accumulation
    assert np.all(np.diff(np.sign(R_on - R_off) * R) >= -1e-9)    # monotonic turn-on
