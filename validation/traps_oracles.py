"""
D1c TRAPS -- the heavy / cross-version half of the trap oracles (the fast gates are tests/test_traps.py,
whose 1-D device builders and closed forms this script reuses so the two cannot drift).

  GATE A  slab SRH lifetime vs E_t (9 levels, -0.4..+0.4 eV) == tau_p0 (1 + n1/n0) + tau_n0 p1/n0
  GATE B  trap-compensated majority density == scipy charge-neutrality root (acceptor/n, donor/p)
  GATE C  MOS LF C-V with uniform D_it: V_G(psi_s) and d psi_s/dV_G vs Nicollian-Brews; V_FB shift
          from a fixed Q_f vs -Q_f/C_ox
  GATE D  surface recombination velocity S == sigma v_th N_it (three decades of N_it)
  GATE E  OLD PHYSICS FAILS every one of A-D: the same oracles run with the traps OFF (which is the
          pre-D1c code, byte-for-byte -- gate F) miss by the margins printed; and the pre-D1c MODULE
          (main, 5cbd998, via a subprocess on its own tree) cannot even express them (ImportError /
          TypeError).
  GATE F  CROSS-VERSION BYTE IDENTITY: a 1-D Si diode (FD on, forward ramp), a 1-D MOS cap (gate
          sweep into inversion onset) and a 2-D LayeredDevsimBuilder bipolar bar, solved on main and
          on this branch with the traps OFF: sha256-identical Potential/Electrons/Holes/USRH/
          PotentialNodeCharge arrays.
  GATE G  (--3d) 3-D: G1 the Stacked3DSpec(interface_traps=...) weights are exact (sum = lateral^2,
          only on the semi/oxide face); G2 the D_it psi_s shift equals the 1-D
          cap's. G2 FAILS (3-D 0.80 x the 1-D shift) for a PRE-EXISTING 3-D mesh defect: gmsh's
          tetrahedra are not Delaunay, DEVSIM's Voronoi node volumes over-count the region volume
          (1.20 x semi, 1.29 x oxide) and the discrete capacitances with them, so a geometric sheet
          charge is diluted. A pure-Laplace 3-D sheet test (no carriers) shows the same (0.86); a
          3-D gate shift of -Q/C_ox reproduces the 1-D answer, and mesh_min 0.25 / 0.5 / 1 nm leaves
          the ratio at 0.79 / 0.79 / 0.77 -- not a discretization error that refinement removes.

Run:  python -m validation.traps_oracles [--3d] [--main <path to a main checkout>]
"""
import argparse
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from validation._harness import Harness                                 # noqa: E402

_spec = importlib.util.spec_from_file_location("_trap_tests", os.path.join(HERE, "tests",
                                                                          "test_traps.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
PT = T.PT
DEFAULT_MAIN = os.path.join(os.path.dirname(os.path.dirname(HERE)), "DynaMeta")


# ------------------------------------------------------------------------------------------------
def gate_a(H, legacy=False):
    tn, tp, G, L = 3e-10, 1e-10, 1e24, 60e-6
    worst = 0.0
    rows = []
    for E in (-0.4, -0.35, -0.3, -0.2, 0.0, 0.2, 0.3, 0.35, 0.4):
        dp, n0, p0 = T._slab_tau("v_tau", tau_n=tn, tau_p=tp, G=G, length=L,
                                 srh_trap_energy_eV=(0.0 if legacy else E))
        tau, tau_full = T._tau_closed_form(E, tn, tp, n0, p0)
        Lp = math.sqrt(0.048 * T.V_T * (T._tau_closed_form(0.0 if legacy else E, tn, tp, n0, p0)[1]))
        tm = dp / G / (1.0 - 1.0 / math.cosh(0.5 * L / Lp))
        rel = tm / tau - 1.0
        worst = max(worst, abs(rel))
        rows.append((E, tau, tm, rel))
    for E, tau, tm, rel in rows:
        print("[trp]   E_t-E_i={:+.2f} eV  tau_closed={:.5e}  tau_meas={:.5e}  rel={:+.2e}".format(
            E, tau, tm, rel), flush=True)
    return worst


def gate_b(legacy=False):
    from scipy.optimize import brentq
    out = []
    for nnet, lv in ((1e22, PT.TrapLevel(energy_eV=0.33, density_m3=6e21, kind="acceptor",
                                         tau_n0_s=1e-9, tau_p0_s=1e-9)),
                     (-1e22, PT.TrapLevel(energy_eV=-0.30, density_m3=4e21, kind="donor",
                                          tau_n0_s=1e-9, tau_p0_s=1e-9))):
        T._slab("v_q", length=4e-6, net_doping=nnet, tau_n=1e-7, tau_p=1e-7, ps_end=2e-9,
                ps_mid=5e-8, traps=(None if legacy else [lv]))
        T._seed("v_q", "r")
        T._solve(1e10)
        x = T._vals("v_q", "r", "x")
        ic = int(np.argmin(np.abs(x - 2e-6)))
        maj = (T._vals("v_q", "r", "Electrons") if nnet > 0 else T._vals("v_q", "r", "Holes"))[ic]
        T._clear("v_q")

        def neutral(ef):
            n, p = T.NI * math.exp(ef / T.V_T), T.NI * math.exp(-ef / T.V_T)
            f = 1.0 / (1.0 + math.exp((lv.energy_eV - ef) / T.V_T))
            qt = -lv.density_m3 * f if lv.kind == "acceptor" else lv.density_m3 * (1.0 - f)
            return (p - n + nnet + qt) / abs(nnet)

        ef = brentq(neutral, -0.6, 0.6, xtol=1e-15)
        ref = T.NI * math.exp((ef if nnet > 0 else -ef) / T.V_T)
        out.append((lv.kind, maj, ref, maj / ref - 1.0))
        print("[trp]   {:8s} N_net={:+.0e}: majority sim={:.6e} root={:.6e} rel={:+.2e}".format(
            lv.kind, nnet, maj, ref, maj / ref - 1.0), flush=True)
    return max(abs(r[3]) for r in out)


def gate_c(legacy=False):
    it = PT.InterfaceTraps(dit_eV_m2=T.DIT, band_gap_eV=T.EG, recombination=False)
    V, P = T._mos_cv("v_mos", None if legacy else it)
    va = np.array([p - (T._Qs(p) + T._Qit(p, T.DIT, T.EG)) / T.MOS_COX for p in P])
    dv = float(np.max(np.abs(V - va)))
    slope = np.gradient(P, V)
    ana = np.array([T.MOS_COX / (T.MOS_COX + T._Cs(p) + T._Cit(p, T.DIT, T.EG)) for p in P])
    keep = np.abs(P) > 2e-3
    keep[0] = keep[-1] = False
    rel = np.abs(slope / ana - 1.0)[keep]
    for v, p, a in list(zip(V, P, va))[::10]:
        print("[trp]   V_G={:+.2f} V  psi_s={:+.5f} V  V_G(analytic at psi_s)={:+.5f}".format(v, p, a),
              flush=True)
    qf = T.Q * 5e15
    vgs = np.round(np.arange(-0.4, 0.1001, 0.02), 10)

    def vfb(spec):
        psib = T._mos("v_fb", spec)
        T._mos_sweep("v_fb", psib, np.round(np.arange(0.0, -0.4001, -0.02), 10))
        p = T._mos_sweep("v_fb", psib, vgs)
        T._clear("v_fb")
        k = int(np.argmin(np.abs(p)))
        sl = slice(max(0, k - 3), max(0, k - 3) + 6)
        return float(np.polyval(np.polyfit(p[sl], vgs[sl], 3), 0.0))

    dvfb = vfb(None if legacy else PT.InterfaceTraps(fixed_charge_C_m2=qf, recombination=False)) - vfb(None)
    print("[trp]   dV_FB = {:+.6f} V  (analytic -Q_f/C_ox = {:+.6f} V)".format(dvfb, -qf / T.MOS_COX),
          flush=True)
    return dv, float(rel.max()), float(np.median(rel)), dvfb - (-qf / T.MOS_COX)


def gate_d(legacy=False):
    from scipy.optimize import brentq
    worst_s, worst_p = 0.0, 0.0
    for n_it in (1e15, 1e16, 1e17):
        sig, vth = 1e-19, 1e5
        it = PT.InterfaceTraps(distribution="discrete", levels=((0.0, n_it),), charge="none",
                               sigma_n_m2=sig, sigma_p_m2=sig, v_th_n_ms=vth, v_th_p_ms=vth)
        x, dp = T._srv_profile("v_srv", None if legacy else it)
        s_th = sig * vth * n_it
        s_meas = brentq(lambda S: T._dp_analytic(np.array([T.SRV_L]), S)[0] - dp[-1], -1.0, 1e8,
                        xtol=1e-12)
        prof = float(np.max(np.abs(dp - T._dp_analytic(x, s_th))) / (T.SRV_G * T.SRV_TAU))
        print("[trp]   N_it={:.0e} m^-2: S = sigma v_th N_it = {:.1f} m/s, measured {:.4f} m/s "
              "(rel {:+.2e}); profile max err {:.2e} of G tau".format(
                  n_it, s_th, s_meas, s_meas / s_th - 1.0, prof), flush=True)
        worst_s, worst_p = max(worst_s, abs(s_meas / s_th - 1.0)), max(worst_p, prof)
    return worst_s, worst_p


# ------------------------------------------------------------------------------------------------
_FIXTURE = textwrap.dedent('''
    import hashlib, sys, math
    import numpy as np
    import devsim as ds
    from dynameta.carriers import physics_bipolar_dd as BP, physics_equilibrium as PE
    from dynameta.carriers import eq_registry as R
    def clear(d):
        for dv in list(ds.get_device_list()): ds.delete_device(device=dv)
        for m in list(ds.get_mesh_list()): ds.delete_mesh(mesh=m)
        R.clear(d)
    def h(dev, reg):
        out = hashlib.sha256()
        for k in ("Potential", "Electrons", "Holes", "USRH", "PotentialNodeCharge"):
            out.update(np.asarray(ds.get_node_model_values(device=dev, region=reg, name=k)).tobytes())
        return out.hexdigest()
    def seed(dev, reg):
        ds.node_model(device=dev, region=reg, name="_s",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, reg)))
        for v, m in (("Potential", "_s"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region=reg, name=v,
                               values=list(ds.get_node_model_values(device=dev, region=reg, name=m)))
    # 1) 1-D Si diode, FD g-factor ON, forward bias ramp
    clear("fx"); ds.create_1d_mesh(mesh="fm")
    ds.add_1d_mesh_line(mesh="fm", pos=0.0, ps=2e-9, tag="a")
    ds.add_1d_mesh_line(mesh="fm", pos=0.5e-6, ps=5e-10)
    ds.add_1d_mesh_line(mesh="fm", pos=1e-6, ps=2e-9, tag="b")
    ds.add_1d_contact(mesh="fm", name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh="fm", name="b", tag="b", material="metal")
    ds.add_1d_region(mesh="fm", material="Si", region="r", tag1="a", tag2="b")
    ds.finalize_mesh(mesh="fm"); ds.create_device(mesh="fm", device="fx")
    ds.node_model(device="fx", region="r", name="NetDoping", equation="ifelse(x < 0.5e-6, -1e24, 1e24)")
    BP.setup_bipolar_region("fx", "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7)
    BP.setup_equilibrium_seed_models("fx", "r")
    for c in ("a", "b"): BP.setup_contact_ohmic_bipolar("fx", c)
    seed("fx", "r")
    ds.solve(type="dc", absolute_error=1e18, relative_error=1e-8, maximum_iterations=80)
    for v in (0.1, 0.2, 0.3, 0.4):
        ds.set_parameter(device="fx", name="a_bias", value=v)
        ds.solve(type="dc", absolute_error=1e18, relative_error=1e-8, maximum_iterations=80)
    print("HASH diode", h("fx", "r"))
    # 2) 1-D MOS cap (p-Si / SiO2), Boltzmann, gate at +0.6 V from flat band
    clear("fx"); ds.create_1d_mesh(mesh="fm")
    ds.add_1d_mesh_line(mesh="fm", pos=0.0, ps=20e-9, tag="body")
    ds.add_1d_mesh_line(mesh="fm", pos=0.6e-6, ps=0.05e-9, tag="i")
    ds.add_1d_mesh_line(mesh="fm", pos=0.61e-6, ps=0.5e-9, tag="gate")
    ds.add_1d_contact(mesh="fm", name="body", tag="body", material="metal")
    ds.add_1d_contact(mesh="fm", name="gate", tag="gate", material="metal")
    ds.add_1d_region(mesh="fm", material="Si", region="si", tag1="body", tag2="i")
    ds.add_1d_region(mesh="fm", material="SiO2", region="ox", tag1="i", tag2="gate")
    ds.add_1d_interface(mesh="fm", name="si_ox", tag="i")
    ds.finalize_mesh(mesh="fm"); ds.create_device(mesh="fm", device="fx")
    ds.node_model(device="fx", region="si", name="NetDoping", equation="-1e23")
    BP.setup_bipolar_region("fx", "si", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7,
                            fd_enhancement=False)
    BP.setup_equilibrium_seed_models("fx", "si")
    PE.setup_dielectric_region("fx", "ox", 3.9); PE.setup_interface("fx", "si_ox")
    BP.setup_contact_ohmic_bipolar("fx", "body"); PE.setup_contact("fx", "gate")
    psib = -1.380649e-23 * 300.0 / 1.602176634e-19 * math.log(1e23 / 1e16)
    seed("fx", "si")
    ds.set_node_values(device="fx", region="ox", name="Potential",
                       values=[psib] * len(ds.get_node_model_values(device="fx", region="ox", name="x")))
    for vg in np.arange(0.0, 0.601, 0.05):
        ds.set_parameter(device="fx", name="gate_bias", value=psib + vg)
        ds.solve(type="dc", absolute_error=1e12, relative_error=1e-10, maximum_iterations=80)
    print("HASH mos", h("fx", "si"))
    clear("fx")
    # 3) 2-D LayeredDevsimBuilder bipolar bar (n-type Si, anode/cathode edge contacts), +0.05 V
    from dynameta.constants import M_E
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    cst = lambda v: (lambda n: np.full_like(np.asarray(n, dtype=float), v))
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cm", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    reg.add(Material("si", ConstantOptical(12.0 + 0j), transport=TransportModel(
        n_bg_m3=1e24, eps_static=11.7, dos_mass_kg_of_n_m3=cst(1.08 * M_E), physics="bipolar_dd",
        mobility_m2Vs_of_n_m3=cst(0.135), hole_mobility_m2Vs_of_n_m3=cst(0.048), n_i_m3=1e16,
        tau_srh_s=1e-7)))
    d = Design(name="fx2", unit_cell=UnitCell.square(300e-9),
               stack=Stack(layers=[Layer("si", 40e-9, "si")], superstrate_material="air",
                           substrate_material="air"),
               electrodes=[Electrode("an", "si", "x_lo", role="biased"),
                           Electrode("ca", "si", "x_hi", role="ground", fixed_voltage_V=0.0)],
               materials=reg, mesh_2d=Mesh2DSpec())
    b = LayeredDevsimBuilder(d, mesh_name="fx2m", device_name="fx2")
    b.solve(BiasPoint({"an": 0.05}, "b"))
    print("HASH layered", h("fx2", "si"))
    b.teardown()
''')


def _run_fixture(tree):
    env = dict(os.environ, PYTHONPATH=tree)
    r = subprocess.run([sys.executable, "-c", _FIXTURE], cwd=tree, env=env, capture_output=True,
                       text=True, timeout=1800)
    hashes = dict(line.split()[1:3] for line in r.stdout.splitlines() if line.startswith("HASH "))
    return r.returncode, hashes, r.stderr[-400:]


def _old_tree_rejects(tree):
    probe = ("import sys\ntry:\n    import dynameta.carriers.physics_traps\n    print('IMPORT_OK')\n"
             "except ImportError as e:\n    print('IMPORT_FAIL', e)\n"
             "import inspect\nfrom dynameta.carriers import physics_bipolar_dd as BP\n"
             "print('ET_KW', 'srh_trap_energy_eV' in inspect.signature(BP.setup_bipolar_region).parameters)\n"
             "print('FILE', BP.__file__)\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=tree, env=dict(os.environ, PYTHONPATH=tree),
                       capture_output=True, text=True, timeout=600)
    return r.stdout


# ------------------------------------------------------------------------------------------------
def gate_g(it=None, mesh_min_nm=0.5):
    """3-D full-gate bipolar cap vs the 1-D cap with the same stack: the trap-induced psi_s shift."""
    import devsim as ds
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.constants import M_E
    from dynameta.sweep import BiasPoint
    vg, nb, ni, tsi, tox = 0.5, 1e23, 1e17, 30e-9, 8e-9
    if it is None:
        it = PT.InterfaceTraps(dit_eV_m2=1e16, band_gap_eV=1.12, recombination=False)

    def psi3d(spec_it):
        spec = Stacked3DSpec(semi_material="Si", oxide_material="SiO2", lateral_m=12e-9,
                             semi_thk_m=tsi, oxide_thk_m=tox, n_bg_m3=nb, eps_semi=11.7, eps_oxide=3.9,
                             dos_mass_kg=1.08 * M_E, mobility_m2Vs=0.135, physics="bipolar_dd",
                             n_i_m3=ni, mobility_p_m2Vs=0.048, tau_srh_s=1e-7,
                             interface_traps=spec_it, grid_n=(4, 4, 33), mesh_min_nm=mesh_min_nm)
        sol = Devsim3DEquilibrium(spec, device_name="v3d", mesh_name="v3m")
        sol.build_device()
        sol.solve(BiasPoint({"gate": vg}, "g"))
        z = T._vals("v3d", "semi", "z")
        pot = T._vals("v3d", "semi", "Potential")
        top, bot = z >= z.max() - 1e-15, z <= z.min() + 1e-15
        out = float(pot[top].mean() - pot[bot].mean())
        if spec_it is not None:                                 # wiring + mesh diagnostics
            vol = T._vals("v3d", "semi", "NodeVolume")
            a_it = T._vals("v3d", "semi", "ItArea0") * vol
            diag["area_rel"] = abs(a_it.sum() / 12e-9 ** 2 - 1.0)
            diag["on_top"] = bool(np.all(top[a_it > 0])) and bool(np.all(a_it[top] > 0))
            sa = T._vals("v3d", "semi", "SurfaceArea")
            diag["vs_SurfaceArea"] = float(np.max(np.abs(a_it - sa)) / np.max(sa))
            diag["vol_semi"] = vol.sum() / (12e-9 ** 2 * tsi)
            diag["vol_ox"] = T._vals("v3d", "oxide", "NodeVolume").sum() / (12e-9 ** 2 * tox)
        sol.teardown()
        return out

    def psi1d(spec_it):
        dev, m = "v1d", "v1m"
        T._clear(dev)
        ds.create_1d_mesh(mesh=m)
        ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=1e-9, tag="body")
        ds.add_1d_mesh_line(mesh=m, pos=tsi, ps=0.05e-9, tag="i")
        ds.add_1d_mesh_line(mesh=m, pos=tsi + tox, ps=0.5e-9, tag="gate")
        ds.add_1d_contact(mesh=m, name="body", tag="body", material="metal")
        ds.add_1d_contact(mesh=m, name="gate", tag="gate", material="metal")
        ds.add_1d_region(mesh=m, material="Si", region="si", tag1="body", tag2="i")
        ds.add_1d_region(mesh=m, material="SiO2", region="ox", tag1="i", tag2="gate")
        ds.add_1d_interface(mesh=m, name="si_ox", tag="i")
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=dev)
        ds.node_model(device=dev, region="si", name="NetDoping", equation=repr(nb))
        nc = PE.effective_dos_m3(1.08 * M_E)
        BP.setup_bipolar_region(dev, "si", eps_static=11.7, n_dos_m3=nc, n_i_m3=ni,
                                mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7,
                                tau_p_s=1e-7, fd_enhancement=True, n_dos_p_m3=nc)
        BP.setup_equilibrium_seed_models(dev, "si")
        PE.setup_dielectric_region(dev, "ox", 3.9)
        PE.setup_interface(dev, "si_ox")
        BP.setup_contact_ohmic_bipolar(dev, "body")
        PE.setup_contact(dev, "gate")
        if spec_it is not None:
            PT.install_interface_traps(dev, "si_ox", spec_it)
        phi_bi = BP.ohmic_built_in_offset_V(nb, n_i_m3=ni, n_dos_m3=nc, n_dos_p_m3=nc)
        T._seed(dev, "si")
        nox = len(T._vals(dev, "ox", "x"))
        ds.set_node_values(device=dev, region="ox", name="Potential", values=[phi_bi] * nox)
        for k in range(0, 11):
            ds.set_parameter(device=dev, name="gate_bias", value=phi_bi + vg * k / 10.0)
            T._solve(1e12, 1e-10)
        pot = T._vals(dev, "si", "Potential")
        T._clear(dev)
        return float(pot[-1] - pot[0])

    diag = {}
    a3, b3 = psi3d(None), psi3d(it)
    a1, b1 = psi1d(None), psi1d(it)
    print("[trp]   psi_s(3-D) {:.6f} -> {:.6f} V (shift {:+.6f});  psi_s(1-D) {:.6f} -> {:.6f} V "
          "(shift {:+.6f})".format(a3, b3, b3 - a3, a1, b1, b1 - a1), flush=True)
    print("[trp]   3-D interface area: sum/lateral^2 - 1 = {:.1e}, all on the top face = {}, "
          "per-node partition vs DEVSIM SurfaceArea max {:.1%} of a node (barycentric thirds vs DEVSIM's own split; same total); 3-D mesh sum(NodeVolume)/geometric volume: "
          "semi {:.3f}, oxide {:.3f}".format(diag["area_rel"], diag["on_top"], diag["vs_SurfaceArea"],
                                             diag["vol_semi"], diag["vol_ox"]), flush=True)
    return (b3 - a3), (b1 - a1), diag


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--3d", dest="three_d", action="store_true")
    ap.add_argument("--main", default=DEFAULT_MAIN)
    a = ap.parse_args()
    H = Harness("trp", "D1c TRAPS: bulk SRH(E_t), multi-level, trap charge, interface traps (D_it)")

    print("[trp] --- A: slab lifetime vs E_t (n-type 1e21, tau_n0 = 3e-10, tau_p0 = 1e-10 s)", flush=True)
    wa = gate_a(H)
    H.gate("A tau(E_t) == closed form", wa < 1e-4, "max |rel| {:.2e} < 1e-4".format(wa))
    print("[trp] --- B: trapped charge at equilibrium", flush=True)
    wb = gate_b()
    H.gate("B compensated majority == neutrality root", wb < 1e-6, "max |rel| {:.2e} < 1e-6".format(wb))
    print("[trp] --- C: MOS LF C-V, uniform D_it = 1e12 cm^-2 eV^-1; fixed Q_f = 5e11 q cm^-2", flush=True)
    dv, smax, smed, efb = gate_c()
    H.gate("C1 V_G(psi_s) stretch-out", dv < 5e-4, "max |dV| {:.2e} V < 5e-4".format(dv))
    H.gate("C2 d psi_s/dV_G = Cox/(Cox+Cs+Cit)", smax < 1e-3,
           "max rel {:.2e} (median {:.1e}) < 1e-3".format(smax, smed))
    H.gate("C3 dV_FB = -Q_f/C_ox", abs(efb) < 1e-4, "|err| {:.2e} V < 1e-4".format(abs(efb)))
    print("[trp] --- D: surface recombination velocity", flush=True)
    ws, wp = gate_d()
    H.gate("D S_meas == sigma v_th N_it", ws < 2e-3 and wp < 2e-3,
           "max |rel S| {:.2e}, profile {:.2e} < 2e-3".format(ws, wp))

    print("[trp] --- E: the same oracles with the traps OFF (== the pre-D1c code, gate F)", flush=True)
    def _legacy():
        la = gate_a(H, legacy=True)
        lb = gate_b(legacy=True)
        ldv, lsmax, _lsmed, lefb = gate_c(legacy=True)
        lws, _lwp = gate_d(legacy=True)
        fails = (la > 1e-4, lb > 1e-6, ldv > 5e-4, lsmax > 1e-3, abs(lefb) > 1e-4, lws > 2e-3)
        return all(fails), ("A {:.2f}  B {:.2f}  C1 {:.3f} V  C2 {:.3f}  C3 {:.3f} V  D {:.2f}  (each "
                            "must exceed its gate)".format(la, lb, ldv, lsmax, abs(lefb), lws))

    H.run("E old physics fails A/B/C1/C2/C3/D", _legacy)
    if os.path.isdir(os.path.join(a.main, "dynameta")):
        out = _old_tree_rejects(a.main)
        print("[trp]   old tree probe: " + " | ".join(out.split("\n")), flush=True)
        H.gate("E' old module cannot express traps", "IMPORT_FAIL" in out and "ET_KW False" in out,
               "physics_traps ImportError + no srh_trap_energy_eV keyword on {}".format(a.main))
        print("[trp] --- F: cross-version byte identity (main vs this branch, traps OFF)", flush=True)
        rc0, h0, e0 = _run_fixture(a.main)
        rc1, h1, e1 = _run_fixture(HERE)
        for k in sorted(set(h0) | set(h1)):
            print("[trp]   {:6s} main {}  branch {}".format(k, h0.get(k, "-")[:16], h1.get(k, "-")[:16]),
                  flush=True)
        H.gate("F byte-identical vs main", rc0 == 0 and rc1 == 0 and h0 == h1 and len(h0) == 3,
               "rc {} / {}, {} fixtures".format(rc0, rc1, len(h0)) + ("" if rc0 == rc1 == 0 else
                                                                       " | " + e0 + " | " + e1))
    else:
        H.skip("E'/F cross-version", "no main checkout at {} (--main)".format(a.main))
    if a.three_d:
        print("[trp] --- G: 3-D Stacked3DSpec(interface_traps) vs the 1-D cap", flush=True)
        s3, s1, dg = gate_g()
        H.gate("G1 3-D interface wiring", dg["area_rel"] < 1e-12 and dg["on_top"],
               "total area exact ({:.0e}), every weighted node on the semi/oxide face".format(
                   dg["area_rel"]))
        # G2 FAILS on the current 3-D mesh, for a PRE-EXISTING reason (docs/audit/2026-11-03-devsim-
        # traps.md section 6): gmsh's tetrahedra are not Delaunay, so DEVSIM's Voronoi finite volumes
        # over-count (sum NodeVolume = 1.20 x the semi volume, 1.29 x the oxide's) and the discrete
        # capacitances with them; a GEOMETRIC sheet charge then acts at ~0.8 of its 1-D effect. Kept
        # as a real gate (not relaxed) so it turns green when the 3-D mesh is fixed.
        H.gate("G2 3-D D_it shift == 1-D", abs(s3 - s1) < 0.05 * abs(s1) and abs(s1) > 5e-3,
               "{:+.5f} vs {:+.5f} V (ratio {:.3f}; 3-D mesh volume excess semi {:.3f} / oxide {:.3f})"
               .format(s3, s1, s3 / s1, dg["vol_semi"], dg["vol_ox"]))
    return H.summary("D1c TRAPS")


if __name__ == "__main__":
    raise SystemExit(main())
