"""Validate the BIPOLAR drift-diffusion (Potential + Electrons + Holes + SRH) path of the native 3D
gated-capacitor builder (Devsim3DEquilibrium, Stacked3DSpec.physics='bipolar_dd') -- the 3D analogue of
the 2D LayeredDevsimBuilder bipolar path. The independent oracle is the SAME 3D builder run with the two
already-validated physics models on the SAME mesh:
  * the EQUILIBRIUM solve (single-variable Poisson + Aymerich-Humet F_1/2, the true Fermi-Dirac statistics),
  * the UNIPOLAR electron drift-diffusion (FD-enhanced Scharfetter-Gummel), validated to ~0.8% vs equilibrium.
For an n-type gated cap in ACCUMULATION the holes are negligible, so the bipolar electron profile must
REDUCE to both references; and the bipolar solve must additionally carry a physical hole population
(charge-neutral bulk: the mass-action n p = n_i^2 holds in the field-free region near the body contact).

This reduces-to-known-limit check is load-bearing: it caught a potential-reference-frame bug (the bipolar
body contact pins the intrinsic-referenced built-in phi_bi = +V_t ln(n0/n_i) while the gate pins a raw
Dirichlet, so without offsetting the gate by phi_bi the cap under-accumulated 7x even though the coupled
Newton CONVERGED -- convergence does not prove correctness).

GATE A (reduces to the validated references): bipolar areal electron density matches BOTH the equilibrium
        and the unipolar-DD areal density to < 2%, and the electron z-profile matches unipolar to < 3%.
GATE B (genuinely bipolar + sign-correct): holes are present everywhere (p > 0), the field-free bulk near
        the body satisfies mass-action n p ~ n_i^2 (within a factor of 3), and the cap ACCUMULATES under
        +Vg (n at the oxide interface > n at the body).

A moderately-doped n-type semiconductor (n_bg=1e23, n_i=1e17) is used so the electron/hole dynamic range
stays well-conditioned; a degenerate ITO-like cap (n_bg~1e26) in DEEP accumulation drives the holes ~18
decades below the electrons and can stall the coupled Newton -- use the unipolar electron DD path there
(holes are physically irrelevant for a degenerate n-type accumulation layer).

Run: python -m validation.carriers_3d_bipolar
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.devsim_3d import Stacked3DSpec, Devsim3DEquilibrium
from dynameta.carriers.physics_equilibrium import M_E
from dynameta.sweep import BiasPoint

VG = 0.6
N_I = 1.0e17
COMMON = dict(semi_material="Si", oxide_material="HfO2", lateral_m=12e-9, semi_thk_m=16e-9,
              oxide_thk_m=6e-9, n_bg_m3=1.0e23, eps_semi=11.7, eps_oxide=18.0, dos_mass_kg=1.08 * M_E,
              mobility_m2Vs=0.05, grid_n=(6, 6, 21), mesh_min_nm=1.0, mesh_max_nm=4.0)


def _solve(physics, extra, dev):
    sp = Stacked3DSpec(physics=physics, **extra, **COMMON)
    s = Devsim3DEquilibrium(sp, device_name=dev, mesh_name=dev + "_m")
    cf = s.solve(BiasPoint({"gate": VG, "body": 0.0}, physics))
    reg = cf.regions[sp.field_region_name]
    prof = reg.grid_fields["electron_density_m3"].mean(axis=(0, 1))     # z-profile (mean over x,y)
    zax = np.asarray(reg.grid_axes_m["z"])
    areal = float(np.trapezoid(prof, zax))
    import devsim as ds
    raw = {nm: np.asarray(ds.get_node_model_values(device=dev, region="semi", name=nm))
           for nm in ("z", "Electrons", "Holes")} if physics == "bipolar_dd" else None
    s.teardown()
    return prof, areal, zax, raw


def main():
    print("[b3] === 3D BIPOLAR gated cap vs equilibrium + unipolar DD (reduces-to-known-limit) ===",
          flush=True)
    pe, ae, _ze, _re = _solve("equilibrium", {}, "eq3d")
    pu, au, zu, _ru = _solve("drift_diffusion", {}, "uni3d")
    pb, ab, zb, raw = _solve("bipolar_dd", dict(n_i_m3=N_I, mobility_p_m2Vs=0.02, tau_srh_s=1.0e-7), "bip3d")

    # GATE A: reduces to equilibrium AND unipolar
    d_eq = abs(ab - ae) / ae
    d_uni = abs(ab - au) / au
    prof_rel = float(np.max(np.abs(pb - pu) / np.maximum(pu, 1e20)))
    g_a = (d_eq < 2e-2) and (d_uni < 2e-2) and (prof_rel < 3e-2)
    print("[b3] A reduces: areal bip={:.3e} eq={:.3e} uni={:.3e} m^-2 ; |bip-eq|/eq={:.3f} "
          "|bip-uni|/uni={:.3f} ; profile max-rel(bip,uni)={:.3f} -> {}".format(
              ab, ae, au, d_eq, d_uni, prof_rel, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: genuinely bipolar + sign-correct
    zr, en, ep = raw["z"], raw["Electrons"], raw["Holes"]
    zmax = zr.max()
    n_ox = en[zr > zmax - 1e-9].mean(); n_body = en[zr < zr.min() + 1e-9].mean()
    holes_pos = bool(np.all(ep > 0.0))
    # mass-action in the field-free bulk near the body (away from the accumulation layer): n p ~ n_i^2
    body = zr < zr.min() + 2e-9
    np_bulk = float(np.median((en * ep)[body]))
    mass_action = (np_bulk / (N_I ** 2))
    g_b = holes_pos and (n_ox > 1.2 * n_body) and (1.0 / 3.0 < mass_action < 3.0)
    print("[b3] B bipolar: holes>0 everywhere={} ; accumulation n(ox)/n(body)={:.2f} ; "
          "bulk n*p / n_i^2={:.2f} (->1) -> {}".format(holes_pos, n_ox / n_body, mass_action,
                                                       "OK" if g_b else "FAIL"), flush=True)

    ok = g_a and g_b
    print("[b3] *** 3D BIPOLAR GATED CAP: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
