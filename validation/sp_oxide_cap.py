"""Validate the SP CarrierSolver gate-oxide CALIBRATION. The old map applied the
full gate voltage as the semiconductor surface potential (psi_s = Vg), grossly over-
estimating accumulation. With a gate oxide the carrier solves Gauss's law at the oxide
face for psi_s; once the channel accumulates, most of Vg drops across a thin high-k
oxide -> psi_s << Vg. Two maps are checked, each against ITS OWN Gauss law:
  - gate_map="face_field" (default): Vg = psi_s + F_face/C_ox, F_face = eps_s eps0 dphi/dz at
    the oxide face (here re-evaluated independently by a second-order one-sided difference of
    the returned potential plus the half-cell charge, not by the library's flux kernel);
  - gate_map="flat_band_excess": Vg = psi_s + q*N_exc/C_ox, N_exc the sheet excess relative
    to the psi_s = 0 profile.
At Vg=1V check: (1) psi_s << Vg; (2) accumulation (vs the same map's 0 V profile) is
positive and well below the naive psi_s=Vg map's; (3) each Gauss law holds.
Run: python -m validation.sp_oxide_cap

Since v0.16 the default map is face_field and the flat-band map's residual carries the sign of
the charge term (both maps solve to 1e-9 V by brentq). A single check "psi_s + q N_exc/C_ox = Vg"
(the flat-band relation, 0.05 Vg budget) does not apply to the face-field map, whose 0 V profile is not flat band (psi_s(0 V) = +0.2160 V
here, the field of the gate-side dead layer). Film: the generic gated film (12 nm here, n_bg
3e26 m^-3, m* 0.30 m_e, eps_s 9.0; oxide 8 nm / eps 20). Measured v0.16: face_field
psi_s(1 V) = 0.4422 V, Gauss 1.000047 V by the independent flux; flat_band_excess psi_s(1 V) =
0.2363 V, Gauss 1.000000 V (v0.15.1 default on the same film: psi_s 0.2363 V, flat-band check
1.0003 V -- the Vg/256 bisection, here landing close to the root by chance).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
from dynameta.carriers.schrodinger_poisson import Q, EPS0, M_E
from dynameta.sweep import BiasPoint
from dynameta.core.carrier_field import ELECTRON_DENSITY, POTENTIAL

N_BG, T_SEMI, T_OX, EPS_OX, EPS_S = 3e26, 12e-9, 8e-9, 20.0, 9.0      # the generic gated film
FILM = dict(n_bg_m3=N_BG, m_eff_kg=0.30 * M_E, eps_static=EPS_S)
VG = 1.0


def _profile(cf):
    reg = cf.regions["semi"]
    return (reg.grid_axes_m["z"], reg.grid_fields[POTENTIAL][0, 0, :],
            reg.grid_fields[ELECTRON_DENSITY][0, 0, :])


def sheet_excess(cf, cf0):
    """net excess sheet density of cf RELATIVE TO the profile cf0 (trapezoid)."""
    z, _, v = _profile(cf)
    _, _, v0 = _profile(cf0)
    return float(np.sum(0.5 * ((v[:-1] + v[1:]) - (v0[:-1] + v0[1:])) * np.diff(z)))


def face_flux_independent(cf):
    """eps_s eps0 dphi/dz at the oxide face: second-order one-sided difference of the potential
    (an independent estimate; it differs from the conservative flux by O(h^2))."""
    z, phi, _ = _profile(cf)
    h = z[1] - z[0]
    return EPS_S * EPS0 * (3.0 * phi[-1] - 4.0 * phi[-2] + phi[-3]) / (2.0 * h)


def main():
    C_ox = EPS_OX * EPS0 / T_OX
    print("[t] C_ox = {:.4e} F/m^2".format(C_ox), flush=True)
    naive = SchrodingerPoissonCarrier(semi_thk_m=T_SEMI, lateral_m=12e-9, **FILM)  # psi_s = Vg
    cf_n0 = naive.solve(BiasPoint({"gate": 0.0, "body": 0.0}, "naive0"))
    cf_n = naive.solve(BiasPoint({"gate": VG, "body": 0.0}, "naive"))
    psi_n = cf_n.extras["surface_potential_V"]
    ne_n = sheet_excess(cf_n, cf_n0)
    print("[t] naive (psi_s=Vg):     psi_s={:.4f} V  N_exc={:.3e} m^-2".format(psi_n, ne_n), flush=True)
    ok = True
    for gm in ("face_field", "flat_band_excess"):
        phys = SchrodingerPoissonCarrier(semi_thk_m=T_SEMI, lateral_m=12e-9, oxide_thk_m=T_OX,
                                         eps_oxide=EPS_OX, gate_map=gm, **FILM)
        cf_0 = phys.solve(BiasPoint({"gate": 0.0, "body": 0.0}, "0V"))
        cf_p = phys.solve(BiasPoint({"gate": VG, "body": 0.0}, "phys"))
        psi_0 = cf_0.extras["surface_potential_V"]
        psi_p = cf_p.extras["surface_potential_V"]
        ne_p = sheet_excess(cf_p, cf_0)
        if gm == "face_field":
            gauss = psi_p + face_flux_independent(cf_p) / C_ox        # = Vg (+ O(h^2))
            budget = 1e-3
        else:
            gauss = psi_p + Q * ne_p / C_ox                            # = Vg (root tolerance)
            budget = 1e-6
        print("[t] {:<17s} psi_s(0V)={:+.4f} V  psi_s(1V)={:.4f} V  N_exc(1V vs 0V)={:.3e} m^-2  "
              "Gauss: {:.6f} V (target {:.3f})".format(gm, psi_0, psi_p, ne_p, gauss, VG), flush=True)
        divides = psi_p < 0.9 * VG                                   # meaningful oxide drop
        less_accum = 0.0 < ne_p < 0.8 * ne_n                         # positive but smaller than naive
        consistent = abs(gauss - VG) < budget                        # this map's Gauss law
        print("[t]   psi_s<Vg={} less_accumulation={} gauss_ok={}".format(divides, less_accum, consistent),
              flush=True)
        ok = ok and divides and less_accum and consistent
    print("[t] *** SP OXIDE-CAP CALIBRATION (face-field + flat-band maps): {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
