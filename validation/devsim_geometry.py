"""DEVSIM exact-geometry workaround on DynaMeta's REAL unstructured meshes (devsim_3d mesher='gmsh').

dynameta.carriers.devsim_geometry.apply_exact_geometry replaces DEVSIM 2.11.2's unsigned circumcentric
box-method geometry by the signed one (module doc; docs/audit/2026-11-06-devsim-geometry-workaround.md).
tests/test_devsim_geometry.py gates it on hand-built meshes; this script measures it where it matters --
the legacy gmsh 3-D stack (semi 30 nm / oxide 8 nm, 12 nm cell), which has ~45 % non-well-centred tets
AND ~1.3 % negative assembled edge couples (gmsh's 3-D meshes are not Delaunay), so the default
negative='raise' refuses it and the choice is between 'warn' (exact) and 'clip'.

  GATE G1  geometry: DEVSIM's sum NodeVolume over-counts (> 1.1 x, MEASURED 1.196 semi / 1.293 oxide);
           after the correction (warn) sum NodeVolume, 2 sum ElementNodeVolume and sum EdgeCouple L / 3
           equal the region volume to 1e-12, and SurfaceArea (gate oxide interface) the cell area.
  GATE G2  carrier-free series capacitor (semi as an eps = 11.7 dielectric, oxide 3.9, plates gate /
           body): C = L^2 / (t_si/eps_si + t_ox/eps_ox). DEVSIM's geometry misses by > 5 %; with the exact
           geometry (warn) the piecewise-linear potential is reproduced and C is exact to 1e-9 (budget:
           solver tolerance 1e-12 x cond ~1e3). 'clip' is MEASURED (no closed-form budget: it drops the
           patch test on the clipped edges).
  GATE G3  bipolar MOS cap (FD-SG Newton) at V_g = 0.5 V on the corrected mesh (warn): it converges, and
           Gauss Q_semi + Q_body + Q_gate(geometric) = 0 is MEASURED against DEVSIM's own geometry (the
           D2c gate M3; 20 % off on this mesh pre-correction). The geometric gate charge uses the mean
           surface potential, so a lateral psi_s spread (an unstructured-mesh discretization error, not a
           geometry error) bounds how close it can get. Bound (derived): Q_gate(geometric) replaces the
           laterally varying psi_s under the oxide by its mean, which moves eps_ox (psi_g - psi_s)/t_ox L^2
           by at most eps_ox * spread / t_ox * L^2 (the 'spread term'); gate = residual <= that term.
           DEVSIM's own geometry must FAIL the same bound.
"""
import os
import sys
import tempfile
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

Q, EPS0 = 1.602176634e-19, 8.8541878128e-12


def _apply_hook(sol, policy):
    """Wrap the builder's _create_mesh_device so the correction runs right after create_device and
    before ANY physics setup -- the integration point a builder option would use."""
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers import devsim_geometry as G
    ds = import_devsim()
    orig = sol._create_mesh_device
    reps = {}

    def patched():
        orig()
        for r in ds.get_region_list(device=sol.device):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                reps[r] = G.apply_exact_geometry(sol.device, r, negative=policy)
            reps[r].warned = [str(x.category.__name__) for x in w]
    sol._create_mesh_device = patched
    return reps


def _builder(tag, physics="drift_diffusion", policy=None, **kw):
    import validation.devsim_3d_mesh as M
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium
    spec = M._spec(physics=physics, mesher="gmsh", **kw)
    tmp = tempfile.mkdtemp(prefix="geo_{}_".format(tag))
    sol = Devsim3DEquilibrium(spec, device_name=tag, mesh_name=tag + "m", msh_path=os.path.join(tmp, tag + ".msh"))
    reps = _apply_hook(sol, policy) if policy else {}
    return sol, spec, reps


def geometry(policy):
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers import devsim_geometry as G
    import validation.devsim_3d_mesh as M
    ds = import_devsim()
    M._clear()
    sol, spec, reps = _builder("g1" + (policy or "none"), policy=policy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol._create_mesh_device()
    out = {}
    for (r, _m, _ro, zl, zh, _e) in spec.layer_stack_nm():
        V = M.L * M.L * (zh - zl) * 1e-9
        nvv = M._v(sol.device, r, "NodeVolume")
        ec = np.asarray(ds.get_edge_model_values(device=sol.device, region=r, name="EdgeCouple"))
        el = np.asarray(ds.get_edge_model_values(device=sol.device, region=r, name="EdgeLength"))
        env = np.asarray(ds.get_element_model_values(device=sol.device, region=r, name="ElementNodeVolume"))
        sa = M._v(sol.device, r, "SurfaceArea").sum() / (M.L * M.L)
        rep = reps.get(r) or G.measure_dual_geometry(sol.device, r)
        out[r] = {"nv": nvv.sum() / V, "env": 2 * env.sum() / V, "ec": (ec * el).sum() / 3 / V, "sa": sa,
                  "n_el": rep.n_elements, "nwc": rep.n_not_well_centred, "neg_ec": rep.n_negative_couples,
                  "n_edges": el.size, "min_ec": rep.min_couple_rel, "neg_nv": rep.n_negative_volumes,
                  "devsim_ratio": rep.devsim_volume_ratio, "clip_ec": rep.clipped_couple_fraction,
                  "clip_nv": rep.clipped_volume_fraction}
    sol.teardown()
    return out


def capacitor(policy):
    """G2: carrier-free two-dielectric capacitor on the gmsh mesh; returns C / C_exact."""
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.carriers import physics_equilibrium as PE
    import validation.devsim_3d_mesh as M
    ds = import_devsim()
    M._clear()
    sol, spec, reps = _builder("g2" + (policy or "none"), policy=policy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol._create_mesh_device()
    d = sol.device
    PE.setup_dielectric_region(d, "semi", 11.7)
    PE.setup_dielectric_region(d, "oxide", 3.9)
    for itf in ds.get_interface_list(device=d):
        PE.setup_interface(d, itf)
    PE.setup_contact(d, "gate")
    PE.setup_contact(d, "body")
    ds.set_parameter(device=d, name="gate_bias", value=1.0)
    ds.solve(type="dc", absolute_error=1e-14, relative_error=1e-12, maximum_iterations=10)
    Cg = abs(ds.get_contact_charge(device=d, contact="gate", equation="PotentialEquation"))
    C_exact = M.L ** 2 / (M.TSI / (11.7 * EPS0) + M.TOX / (3.9 * EPS0))
    # the exact potential: linear in each layer, psi_i at the interface
    psi_i = (M.TSI / 11.7) / (M.TSI / 11.7 + M.TOX / 3.9)       # body at z = 0 held at 0, gate at 1 V
    z_s, p_s = M._v(d, "semi", "z"), M._v(d, "semi", "Potential")
    z_o, p_o = M._v(d, "oxide", "z"), M._v(d, "oxide", "Potential")
    zi = M.TSI
    err = max(np.max(np.abs(p_s - psi_i * z_s / zi)),
              np.max(np.abs(p_o - (psi_i + (1 - psi_i) * (z_o - zi) / M.TOX))))
    sol.teardown()
    return {"C_ratio": Cg / C_exact, "phi_err": float(err)}


def gauss(policy, vg=0.5):
    """G3: the D2c M3 bipolar MOS cap on the gmsh mesh."""
    from dynameta.carriers.devsim_env import import_devsim
    from dynameta.sweep import BiasPoint
    import validation.devsim_3d_mesh as M
    ds = import_devsim()
    M._clear()
    sol, spec, reps = _builder("g3" + (policy or "none"), physics="bipolar_dd", policy=policy, n_i_m3=1e17,
                               mobility_p_m2Vs=0.048, tau_srh_s=1e-7)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol.build_device()
        sol.solve(BiasPoint({"gate": vg}, "g"))
    d = sol.device
    z, pot = M._v(d, "semi", "z"), M._v(d, "semi", "Potential")
    rho = Q * (M._v(d, "semi", "Holes") - M._v(d, "semi", "Electrons") + M._v(d, "semi", "NetDoping"))
    q_semi = float((rho * M._v(d, "semi", "NodeVolume")).sum())
    zo, poto = M._v(d, "oxide", "z"), M._v(d, "oxide", "Potential")
    top = z >= z.max() - 1e-15
    psi_s, psi_g = float(pot[top].mean()), float(poto[zo >= zo.max() - 1e-15].mean())
    q_gate_geo = 3.9 * EPS0 * (psi_g - psi_s) / M.TOX * M.L ** 2
    q_body = float(ds.get_contact_charge(device=d, contact="body", equation="PotentialEquation"))
    spread = float(np.ptp(pot[top]))
    out = {"gauss_rel": abs(q_semi + q_gate_geo + q_body) / abs(q_gate_geo), "psi_s_spread": spread,
           "spread_term": 3.9 * EPS0 * spread / M.TOX * M.L ** 2 / abs(q_gate_geo), "psi_s": psi_s,
           "q_semi_C_m2": q_semi / M.L ** 2, "s": time.time() - t0}
    sol.teardown()
    return out


def main():
    import importlib.util
    from validation._harness import Harness
    H = Harness("geo", "DEVSIM exact (signed) box-method geometry on the gmsh 3-D stack")
    H.require(importlib.util.find_spec("devsim") is not None, "devsim not installed")
    H.require(importlib.util.find_spec("gmsh") is not None, "gmsh not installed (the legacy mesher)")
    g0, g1, gc = geometry(None), geometry("warn"), geometry("clip")
    for r in g0:
        print("[geo]   {:5s} {} tets, {} not well-centred, {}/{} edges with a negative exact couple (worst "
              "{:.3g} L^2), {} negative node volumes; DEVSIM sum NV {:.4f} x, SurfaceArea {:.4f} x; clip drops "
              "{:.2e} of sum|EC|".format(r, g1[r]["n_el"], g1[r]["nwc"], g1[r]["neg_ec"], g1[r]["n_edges"],
                                          g1[r]["min_ec"], g1[r]["neg_nv"], g0[r]["nv"], g0[r]["sa"],
                                          gc[r]["clip_ec"]), flush=True)
    H.gate("G1 DEVSIM over-counts (> 1.1 x)", min(g0[r]["nv"] for r in g0) > 1.1,
           ", ".join("{} {:.4f}".format(r, g0[r]["nv"]) for r in g0))
    w = max(abs(g1[r][k] - 1.0) for r in g1 for k in ("nv", "env", "ec"))
    H.gate("G1 corrected volumes exact (1e-12)", w < 1e-12, "worst |sum/V - 1| {:.1e}".format(w))
    H.gate("G1 corrected SurfaceArea = cell area (1e-12)", abs(g1["semi"]["sa"] - 1.0) < 1e-12,
           "semi {:.15f} (DEVSIM {:.6f})".format(g1["semi"]["sa"], g0["semi"]["sa"]))
    c0, c1, cc = capacitor(None), capacitor("warn"), capacitor("clip")
    H.gate("G2 DEVSIM capacitor misses (> 5 %)", abs(c0["C_ratio"] - 1.0) > 0.05,
           "C/C_exact {:.6f}, max|phi - exact| {:.2e} V".format(c0["C_ratio"], c0["phi_err"]))
    H.gate("G2 exact geometry: C and phi exact (1e-9)", abs(c1["C_ratio"] - 1.0) < 1e-9 and c1["phi_err"] < 1e-9,
           "C/C_exact - 1 {:.2e}, max|phi - exact| {:.2e} V".format(c1["C_ratio"] - 1.0, c1["phi_err"]))
    print("[geo]   G2 clip (MEASURED): C/C_exact - 1 {:.3e}, max|phi - exact| {:.3e} V".format(
        cc["C_ratio"] - 1.0, cc["phi_err"]), flush=True)
    b0 = gauss(None)
    H.gate("G3 DEVSIM geometry fails the Gauss bound", b0["gauss_rel"] > b0["spread_term"],
           "Gauss residual {:.3e} vs spread term {:.2e} (psi_s spread {:.2e} V, {:.1f} s)".format(
               b0["gauss_rel"], b0["spread_term"], b0["psi_s_spread"], b0["s"]))
    for pol in ("warn", "clip"):
        def g3(pol=pol):
            b = gauss(pol)
            return (b["gauss_rel"] <= b["spread_term"],
                    "{}: Gauss residual {:.3e} (DEVSIM geometry {:.3e}), psi_s {:.6f} V, spread {:.2e} V "
                    "(term {:.2e}), {:.1f} s".format(pol, b["gauss_rel"], b0["gauss_rel"], b["psi_s"],
                                                     b["psi_s_spread"], b["spread_term"], b["s"]))
        H.run("G3 bipolar MOS cap converges, Gauss within the lateral-spread term ({})".format(pol), g3)
    return H.summary()


if __name__ == "__main__":
    raise SystemExit(main())
