"""Validate the remaining LC director-physics extensions in dynameta/carriers/lc_director.py against the
external solver and the optical LiquidCrystalModel: (A) the CYLINDRICAL (coaxial) geometry + Poisson
voltage-division, (B) FLEXOELECTRICITY (e1, e3 polarization + direct torque + self-consistent field),
and (C) the end-to-end director -> eps -> optics WIRING (the angle-convention bridge).

Golden = the external lc_statics_base solver:
  * CYL (a=51.5nm, b=181.5nm, t_in=t_out=10nm, eps_in=eps_out=7.5, poisson, theta_b=89.9deg, K11=17pN,
    K33=18pN, eps_para=18.7, eps_perp=4.0, n_o=1.56, n_e=1.92, extra_k_radial):
      V_app -> (V_lc, n_eff, theta_center_deg): 2->(1.6372,1.6414,48.08), 3->(2.2589,1.7540,23.40),
      5->(3.5136,1.8409,5.20).
  * FLEXO (planar, poisson, theta_b=80deg, V=1.0): no-flexo (n_eff=1.5991, theta=62.836deg) vs
    e1=e3=10pC/m self-consistent (n_eff=1.5967, theta=63.839deg).

GATE A (cylindrical): director_profile_bvp(geometry='cyl', poisson) matches the cyl golden (V_lc to <1%,
        theta_center to <1.5 deg, n_eff to <5e-3).
GATE B (flexoelectric): no-flexo and with-flexo match the external golden (theta to <1.5 deg); the flexo
        term shifts theta in the external direction (more tilt-from-field here); and include_flexo=False
        is byte-identical to e1=e3=0 (gated passivity).
GATE C (optics wiring): a field-axis director, bridged via director_to_extra_fields, drives the optical
        LiquidCrystalModel to a UNIAXIAL eps tensor with eigenvalues {n_o^2, n_o^2, n_e^2} and the
        extraordinary axis along (cos theta_optic, 0, sin theta_optic), theta_optic = pi/2 - theta_field
        (planar field-axis -> director along x; homeotropic -> along z) -- the convention is correct.

Run: python -m validation.lc_cyl_flexo_optics
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.lc_director import director_profile_bvp, director_to_extra_fields
from dynameta.core.effects import LiquidCrystalModel

K11, K33, EPS_PARA, EPS_PERP = 17e-12, 18e-12, 18.7, 4.0
N_O, N_E = 1.56, 1.92
CYL = {2.0: (1.6372, 1.6414, 48.0847), 3.0: (2.2589, 1.7539671, 23.399), 5.0: (3.5136, 1.8408747, 5.20477)}


def _ctr(res):
    return math.degrees(float(res.theta_field_rad[res.theta_field_rad.size // 2]))


def main():
    print("[cf] === LC cylindrical geometry + flexoelectricity + optics wiring ===", flush=True)

    # GATE A: cylindrical (coaxial) cell, Poisson voltage division
    dvl = dth = dne = 0.0
    for V, (vlc_g, neff_g, thc_g) in CYL.items():
        r = director_profile_bvp(V_app=V, K11=K11, K33=K33, eps_para=EPS_PARA, eps_perp=EPS_PERP,
                                 geometry="cyl", a=51.5e-9, b=181.5e-9, t_in=10e-9, t_out=10e-9,
                                 eps_in=7.5, eps_out=7.5, theta_b_rad=math.radians(89.9),
                                 field_model="poisson", nz=161, n_o=N_O, n_e=N_E,
                                 opt_model="extra_k_radial")
        dvl = max(dvl, abs(r.V_lc - vlc_g) / vlc_g); dth = max(dth, abs(_ctr(r) - thc_g))
        dne = max(dne, abs(r.n_eff - neff_g))
    g_a = (dvl < 1e-2) and (dth < 1.5) and (dne < 5e-3)
    print("[cf] A cylindrical (coax) poisson: max rel|dV_lc|={:.2e}, max|d theta_ctr|={:.3f} deg, "
          "max|d n_eff|={:.2e} -> {}".format(dvl, dth, dne, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: flexoelectricity (planar, poisson, theta_b=80deg, V=1.0)
    base = dict(K11=K11, K33=K33, eps_para=EPS_PARA, eps_perp=EPS_PERP, d_planar=1e-6,
                theta_b_rad=math.radians(80.0), field_model="poisson", nz=201,
                n_o=N_O, n_e=N_E, opt_model="extra_k_radial")
    r_no = director_profile_bvp(V_app=1.0, **base)
    r_fx = director_profile_bvp(V_app=1.0, include_flexo=True, flexo_self_consistent=True,
                                e1=10e-12, e3=10e-12, **base)
    r_gate = director_profile_bvp(V_app=1.0, include_flexo=True, flexo_self_consistent=True,
                                  e1=0.0, e3=0.0, **base)   # flexo on but zero coeffs -> passive
    d_no = abs(_ctr(r_no) - 62.836); d_fx = abs(_ctr(r_fx) - 63.839)
    shift = _ctr(r_fx) - _ctr(r_no)                          # external: +1.0 deg (more tilt-from-field)
    passive = abs(_ctr(r_gate) - _ctr(r_no)) < 1e-6
    g_b = (d_no < 1.5) and (d_fx < 1.5) and (0.3 < shift < 2.0) and passive
    print("[cf] B flexo: no-flexo theta={:.3f} (gold 62.836), flexo theta={:.3f} (gold 63.839); "
          "shift={:+.3f} deg (gold +1.0); e1=e3=0 passive={} -> {}".format(
              _ctr(r_no), _ctr(r_fx), shift, passive, "OK" if g_b else "FAIL"), flush=True)

    # GATE C: end-to-end director -> bridge -> LiquidCrystalModel eps tensor (convention)
    lcm = LiquidCrystalModel(n_o=N_O, n_e=N_E)
    th_field = np.array([0.0, math.radians(30.0), 0.5 * math.pi])     # homeotropic, 30deg, planar
    fields = director_to_extra_fields(th_field)
    eps = np.asarray(lcm.eps(fields, 1.5e-6))                          # (3,3,3) complex
    okC = True
    for i, thf in enumerate(th_field):
        evals = np.sort(np.linalg.eigvalsh(eps[i].real))
        # uniaxial: two ordinary n_o^2, one extraordinary n_e^2
        okC = okC and np.allclose(evals, np.sort([N_O ** 2, N_O ** 2, N_E ** 2]), atol=1e-9)
        # extraordinary eigenvector along (cos th_optic, 0, sin th_optic), th_optic = pi/2 - th_field
        th_opt = 0.5 * math.pi - float(thf)
        w, V = np.linalg.eigh(eps[i].real)
        ext = V[:, int(np.argmax(w))]                                 # eigenvector of the largest (n_e^2)
        want = np.array([math.cos(th_opt), 0.0, math.sin(th_opt)])
        okC = okC and (abs(abs(float(np.dot(ext, want))) - 1.0) < 1e-6)
    print("[cf] C optics wiring: bridged director -> LiquidCrystalModel eps is uniaxial "
          "{{n_o^2,n_o^2,n_e^2}} with extraordinary axis at pi/2 - theta_field -> {}".format(
              "OK" if okC else "FAIL"), flush=True)

    ok = g_a and g_b and okC
    print("[cf] *** LC CYL + FLEXO + OPTICS WIRING: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
