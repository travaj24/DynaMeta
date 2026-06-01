"""Validate the Schrodinger-Poisson CarrierSolver on the degenerate ITO accumulation
layer: (1) the slab-mode solve recovers the bulk density n_bg away from the interface;
(2) a +gate surface potential accumulates electrons; (3) the QUANTUM signature -- the
accumulation density peak is DISPLACED from the oxide interface (the ~1nm quantum dead
layer; the wavefunction vanishes at the interface), where a classical solve peaks AT
it; (4) it emits a CarrierField the bridge turns into an ENZ-shifted eps.
Run:  python -m validation.sp_carrier
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.sp_carrier import SchrodingerPoissonCarrier
from dynameta.sweep import BiasPoint
from dynameta.materials import Material, MaterialRegistry, DrudeOptical, M_E
from dynameta.core.alignment import GeometryAlignment, RegionAlignment
from dynameta.core import NM, MaterialEpsMap, assemble_eps
from dynameta.core.carrier_field import ELECTRON_DENSITY
from dynameta.core.lift import IdentityLift

N_BG = 4e26
T_SEMI = 12e-9


def profile(cf):
    v = cf.regions["semi"].grid_fields[ELECTRON_DENSITY]   # (nx, ny, nz), laterally uniform
    z = cf.regions["semi"].grid_axes_m["z"]
    return z, v[0, 0, :]


def main():
    solver = SchrodingerPoissonCarrier(semi_thk_m=T_SEMI, n_bg_m3=N_BG, lateral_m=12e-9)
    print("[t] ITO bulk E_F - E_c = {:.4f} eV (degenerate)".format(solver.E_F_J / 1.602176634e-19), flush=True)
    cf0 = solver.solve(BiasPoint({"gate": 0.0, "body": 0.0}, "0V"))
    cfV = solver.solve(BiasPoint({"gate": 0.3, "body": 0.0}, "+0.3V"))
    z, n0 = profile(cf0)
    _, nV = profile(cfV)

    mid = len(z) // 2
    bulk0 = n0[mid] / N_BG                                  # bulk recovery (flat band)
    # accumulation: the peak in the gate half (NOT the dead-layer edge, where psi->0)
    gate_half = slice(mid, None)
    ipk = mid + int(np.argmax(nV[gate_half]))
    peak_ratio = nV[ipk] / N_BG
    peak_nm = (z[-1] - z[ipk]) * 1e9                        # accumulation-peak setback from interface
    print("[t] (1) bulk recovery: n_mid(0V)/n_bg = {:.3f}".format(bulk0), flush=True)
    print("[t] (2) gate accumulation (+0.3V): peak n/n_bg = {:.3f}".format(peak_ratio), flush=True)
    print("[t] (3) quantum dead layer: accumulation peak {:.2f} nm from the oxide interface "
          "(n->0 AT the interface)".format(peak_nm), flush=True)

    # (4) bridge -> eps (ITO Drude); the accumulation peak must push Re(eps) toward ENZ
    # (more carriers -> lower Re(eps)) relative to the unbiased profile. Compare the
    # MINIMUM Re(eps) over the stack (the most-accumulated / most-ENZ point).
    reg = MaterialRegistry()
    reg.add(Material("ITO", DrudeOptical(eps_inf=4.25, m_opt_kg=0.225 * M_E, gamma_rad_s=1.1e14)))
    align = GeometryAlignment(unit_scale=NM, fixed_eps_regions={},
        region_alignments=[RegionAlignment("semi", "semi", (0.0, 12e-9, 0.0, 12e-9, 0.0, T_SEMI), "z")])
    nmap = MaterialEpsMap(reg)
    re0 = np.real(assemble_eps(cf0, align, nmap, IdentityLift(), 1300e-9, mesh_regions=["semi"])
                   ["semi"].values_zyx).mean(axis=(1, 2))
    reV = np.real(assemble_eps(cfV, align, nmap, IdentityLift(), 1300e-9, mesh_regions=["semi"])
                   ["semi"].values_zyx).mean(axis=(1, 2))
    print("[t] (4) eps via bridge: min Re(eps) 0V={:+.3f}  +0.3V={:+.3f} (accumulation deepens ENZ)".format(
        float(re0.min()), float(reV.min())), flush=True)

    ok = (0.7 < bulk0 < 1.3) and (peak_ratio > 1.1) and (0.2 < peak_nm < 4.0) and (reV.min() < re0.min() - 1e-3)
    print("[t] *** SP CARRIER SOLVER: bulk_recover={} accumulates={} quantum_dead_layer={} eps_ENZ={} -> {} ***".format(
        bool(0.7 < bulk0 < 1.3), bool(peak_ratio > 1.1), bool(0.2 < peak_nm < 4.0),
        bool(reV.min() < re0.min() - 1e-3), "PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
