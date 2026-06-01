"""Validate the SP CarrierSolver gate-oxide CALIBRATION fix. The old map applied the
full gate voltage as the semiconductor surface potential (psi_s = Vg), grossly over-
estimating accumulation. The fix divides Vg across the oxide series capacitance:
Vg = psi_s + q*N_excess(psi_s)/C_ox. With a thin high-k oxide, once the channel
accumulates, most of Vg drops across the oxide -> psi_s << Vg. Compare the two maps at
Vg=1V and check: (1) physical psi_s << Vg; (2) physical accumulation << naive; (3) the
self-consistency Vg = psi_s + q*N_exc/C_ox holds. Run: python -m validation.sp_oxide_cap
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
from dynameta.carriers.schrodinger_poisson import Q, EPS0
from dynameta.sweep import BiasPoint
from dynameta.core.carrier_field import ELECTRON_DENSITY

N_BG, T_SEMI, T_OX, EPS_OX = 4e26, 12e-9, 8e-9, 18.0
VG = 1.0


def peak_and_sheet(cf):
    v = cf.regions["semi"].grid_fields[ELECTRON_DENSITY][0, 0, :]
    z = cf.regions["semi"].grid_axes_m["z"]
    n_exc = float(np.sum(0.5 * ((v[:-1] + v[1:]) - 2.0 * N_BG) * np.diff(z)))  # net excess sheet (m^-2)
    return v.max() / N_BG, n_exc


def main():
    naive = SchrodingerPoissonCarrier(semi_thk_m=T_SEMI, n_bg_m3=N_BG, lateral_m=12e-9)  # psi_s = Vg
    phys = SchrodingerPoissonCarrier(semi_thk_m=T_SEMI, n_bg_m3=N_BG, lateral_m=12e-9,
                                      oxide_thk_m=T_OX, eps_oxide=EPS_OX)               # oxide-cap
    cf_n = naive.solve(BiasPoint({"gate": VG, "body": 0.0}, "naive"))
    cf_p = phys.solve(BiasPoint({"gate": VG, "body": 0.0}, "phys"))
    psi_n = cf_n.extras["surface_potential_V"]
    psi_p = cf_p.extras["surface_potential_V"]
    pk_n, ne_n = peak_and_sheet(cf_n)
    pk_p, ne_p = peak_and_sheet(cf_p)
    C_ox = EPS_OX * EPS0 / T_OX
    consistency = psi_p + Q * ne_p / C_ox                       # should equal Vg
    print("[t] C_ox = {:.4e} F/m^2".format(C_ox), flush=True)
    print("[t] naive (psi_s=Vg):  psi_s={:.3f} V  peak n/n_bg={:.3f}  N_exc={:.3e} m^-2".format(
        psi_n, pk_n, ne_n), flush=True)
    print("[t] physical (ox-cap): psi_s={:.3f} V  peak n/n_bg={:.3f}  N_exc={:.3e} m^-2".format(
        psi_p, pk_p, ne_p), flush=True)
    print("[t] self-consistency: psi_s + q*N_exc/C_ox = {:.3f} V  (target Vg={:.3f})".format(
        consistency, VG), flush=True)

    divides = psi_p < 0.9 * VG                                   # meaningful oxide drop (here ~36%)
    less_accum = ne_p < 0.5 * ne_n                               # physical accumulation much smaller
    consistent = abs(consistency - VG) < 0.05 * VG               # series-cap relation satisfied
    ok = divides and less_accum and consistent
    print("[t] *** SP OXIDE-CAP CALIBRATION: psi_s<Vg(oxide divides)={} less_accumulation={} "
          "self_consistent={} -> {} ***".format(
        bool(divides), bool(less_accum), bool(consistent), "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
