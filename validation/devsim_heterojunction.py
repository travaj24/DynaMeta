"""Validate the DEVSIM semiconductor/semiconductor HETEROINTERFACES (v0.14.0, workstream D1a;
dynameta.carriers.physics_heterojunction, wired into LayeredDevsimBuilder and Devsim3DEquilibrium).
Design note + measured numbers: docs/audit/2026-11-03-devsim-heterojunctions.md.

Devices are laterally uniform vertical stacks in the 2-D LayeredDevsimBuilder (a 20 nm cell, three
mesh columns -- the 1-D limit), contacted through metal layers on the top and bottom faces (full-line
ohmic contacts). Materials are GaAs and Al0.3Ga0.7As at 300 K with the Anderson offsets
dE_c = 0.243 eV, dE_v = 0.131 eV (65:35 split of dE_g = 0.374 eV); n_i is derived from each gap,
n_i = sqrt(N_c N_v) exp(-E_g / 2kT), as the bipolar heterojunction guard requires.

Every oracle is INDEPENDENT of the code under test -- band-diagram algebra, Fermi-Dirac statistics by
direct quadrature, the depletion approximation, the Richardson law -- and each one FAILS on v0.13.0
(the builder refused every carrier-transporting semiconductor/semiconductor interface; the 2-D
equilibrium builder put every region at psi = 0 in its own frame, i.e. zero built-in potential).

  GATE A  homojunction limit: a p/n GaAs junction cut into two layers (zero offsets -> 'auto' picks the
          continuous model) vs the SAME junction as an in-region net_doping_expr step in one layer.
  GATE B  abrupt N-AlGaAs / p-GaAs equilibrium: built-in potential, potential split, charge-equivalent
          depletion widths (depletion approximation with the 2kT/q majority-tail correction), and the
          conduction-band spike measured from the SOLVED carrier densities = dE_c.
  GATE C  isotype n-GaAs / N-AlGaAs: low-bias thermionic-emission conductance vs
          G = q A* T^2 exp(-phi_B/kT) / kT (phi_B = solved barrier top - E_F), and the
          continuous-quasi-Fermi model must MISS it (the discriminating gate).
  GATE D  charge neutrality + the discrete Gauss law across the interface (no interface charge).
  GATE E  continuous vs thermionic converge as the barrier vanishes (zero offset, A* -> large).
  GATE F  (--3d, heavy: the 3-D equilibrium build is ~9 min) a gated ITO / AZO-like 3-D heterostack as
          equilibrium vs drift-diffusion (thermionic and continuous): no DC path, so DD must reproduce the
          equilibrium profile. MEASURED 2026-11-03: TE vs continuous 1.5e-9, DD vs equilibrium 2.8e-3
          (the FD g fit vs the Aymerich-Humet closure).

Run: python -m validation.devsim_heterojunction [--3d]
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import EPS0, KB, M_E, Q_E, T_REF, V_T          # noqa: E402

# ---- materials (300 K) ------------------------------------------------------------------------------
GAAS = dict(chi=4.07, eg=1.424, me=0.067, mh=0.50, eps=12.9, mu_n=0.85, mu_p=0.04)
ALGAAS = dict(chi=4.07 - 0.65 * 0.374, eg=1.424 + 0.374, me=0.092, mh=0.55, eps=12.0, mu_n=0.20,
              mu_p=0.01)
DEC = GAAS["chi"] - ALGAAS["chi"]               # 0.2431 eV conduction-band offset (AlGaAs higher)
DEV = (ALGAAS["eg"] - GAAS["eg"]) - DEC         # 0.1309 eV valence-band offset (AlGaAs lower)


def nc_m3(m_rel: float) -> float:
    """Effective DOS 2 (m kT / 2 pi hbar^2)^(3/2) -- written out here (oracle side), not imported."""
    hbar = 1.054571817e-34
    return 2.0 * (m_rel * M_E * KB * T_REF / (2.0 * math.pi * hbar ** 2)) ** 1.5


def n_i_m3(mat: dict) -> float:
    return math.sqrt(nc_m3(mat["me"]) * nc_m3(mat["mh"])) * math.exp(-mat["eg"] / (2.0 * V_T))


def richardson(m_rel: float) -> float:
    """A* = 4 pi q m* k^2 / h^3 (h exact SI)."""
    return 4.0 * math.pi * Q_E * m_rel * M_E * KB ** 2 / 6.62607015e-34 ** 3


def fd_eta(u: float) -> float:
    """eta with F_1/2(eta) = u (normalised, F = (2/sqrt(pi)) int sqrt(e)/(1+exp(e-eta)) de) by direct
    quadrature + bisection -- the oracle's own Fermi-Dirac statistics. Below u = 1e-10 the Boltzmann
    limit eta = ln(u) is exact to |F/e^eta - 1| < u/2^1.5 (minority carriers reach u ~ 1e-22 here)."""
    if u < 1e-10:
        return math.log(u)
    e = np.linspace(0.0, 80.0, 160001)

    def F(eta):
        f = np.sqrt(e) / (1.0 + np.exp(np.clip(e - eta, -700, 700)))
        return 2.0 / math.sqrt(math.pi) * float(np.sum((f[1:] + f[:-1]) * 0.5 * (e[1] - e[0])))
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if F(mid) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def const_fn(v):
    return lambda n: np.full_like(np.asarray(n, dtype=float), v)


def transport(mat: dict, *, n_bg: float, physics: str, acceptor: bool = False, net_doping_expr=None,
        richardson_n=None):
    from dynameta.materials import TransportModel
    kw = dict(n_bg_m3=n_bg, eps_static=mat["eps"], dos_mass_kg_of_n_m3=const_fn(mat["me"] * M_E),
              band_gap_eV=mat["eg"], chi_eV=mat["chi"], physics=physics,
              mobility_m2Vs_of_n_m3=const_fn(mat["mu_n"]))
    if richardson_n is not None:                    # (only when set: keeps the factory usable on v0.13.0,
        kw["richardson_n_Am2K2"] = richardson_n     #  where the fail-on-old check must reach the builder)
    if physics == "bipolar_dd":
        kw.update(hole_mobility_m2Vs_of_n_m3=const_fn(mat["mu_p"]), tau_srh_s=1.0,
                  n_i_m3=n_i_m3(mat), dos_mass_p_kg=mat["mh"] * M_E, acceptor=acceptor,
                  net_doping_expr=net_doping_expr)
    return TransportModel(**kw)


P_CELL = 20e-9                                  # 3 mesh columns -> the 1-D limit
T_METAL = 20e-9


def stack_design(layers, *, name="hj", ps_if=0.5e-9, coarse=4e-9, zone=150e-9):
    """layers: [(layer_name, material_name, thickness_m, TransportModel)] bottom -> top. Adds a metal
    layer below ('anode', biased) and above ('cathode', ground) contacting the outer faces."""
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    seen = set()
    for _ln, mname, _t, tm in layers:
        if mname not in seen:
            reg.add(Material(mname, ConstantOptical(12.0 + 0j), transport=tm))
            seen.add(mname)
    st = [Layer("mbot", T_METAL, "cmetal")] + [Layer(ln, t, m) for ln, m, t, _ in layers] + \
         [Layer("mtop", T_METAL, "cmetal")]
    electrodes = [Electrode("anode", "mbot", "full", role="biased"),
                  Electrode("cathode", "mtop", "full", role="ground", fixed_voltage_V=0.0)]
    m2 = Mesh2DSpec(x_spacing_edge_m=P_CELL / 2, x_spacing_feature_mid_m=P_CELL / 2,
                    x_spacing_feature_edge_m=P_CELL / 2, coarse_ps_m=coarse, interface_ps_m=ps_if,
                    interface_zone_m=zone)
    return Design(name=name, unit_cell=UnitCell.square(P_CELL),
                  stack=Stack(layers=st, superstrate_material="air", substrate_material="air"),
                  electrodes=electrodes, materials=reg, mesh_2d=m2)


def column(builder, region, name):
    """(z, value) along the x = 0 mesh column of `region`, sorted by z."""
    import devsim as ds
    x = np.asarray(ds.get_node_model_values(device=builder.device, region=region, name="x"))
    y = np.asarray(ds.get_node_model_values(device=builder.device, region=region, name="y"))
    v = np.asarray(ds.get_node_model_values(device=builder.device, region=region, name=name))
    sel = np.abs(x) < 1e-12
    o = np.argsort(y[sel])
    return y[sel][o], v[sel][o]


def teardown(builder):
    builder.teardown()


def current_density(builder, cf, contact="cathode"):
    """Terminal current density [A/m^2] through `contact` (the 2-D contact current is per unit depth;
    divide by the cell width)."""
    import devsim as ds
    j = 0.0
    for eq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        if eq in ds.get_contact_equation_list(device=builder.device, contact=contact):
            j += float(ds.get_contact_current(device=builder.device, contact=contact, equation=eq))
    return j / P_CELL


# ---- GATE F (3-D, heavy): drift-diffusion heterostack == equilibrium heterostack at zero current ----
def spec_3d(physics: str, heterointerface: str = "auto"):
    """A gated ITO / AZO-like degenerate heterostack (the library's 3-D use case): primary 'semi'
    4e26 m^-3 (chi 4.50 eV, m* 0.35), gate-ward 'xs' 1e26 m^-3 (chi 4.62 eV, m* 0.25), 4 nm HfO2."""
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    return Stacked3DSpec(semi_material="ITO", oxide_material="HfO2", lateral_m=4e-9, semi_thk_m=6e-9,
                         oxide_thk_m=4e-9, n_bg_m3=4e26, eps_semi=9.5, eps_oxide=18.0,
                         dos_mass_kg=0.35 * M_E, grid_n=(3, 3, 9), mesh_min_nm=1.0, mesh_max_nm=3.0,
                         physics=physics, heterointerface=heterointerface, chi_semi_eV=4.50,
                         extra_semiconductors=[("xs", "AZO", 6e-9, 9.0, 1e26, 4.62, 0.25 * M_E)])


def gate_f_3d(vgs=(0.0, 0.5), msh_dir=None):
    """Solve the 3-D gated heterostack as EQUILIBRIUM and as DRIFT-DIFFUSION with both interface models.
    A gate on an oxide draws no DC current, so the quasi-Fermi level is flat and both DD models must
    reproduce the equilibrium profile -- up to the FD g-fit vs Aymerich-Humet difference of the two bulk
    closures (validation/carriers_3d_dd.py: 0.8 % at +1 V on ITO). Returns {(physics, model, Vg): n}
    of the emitted gate-adjacent region (node values), plus build/solve timings."""
    import tempfile
    import time

    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium
    from dynameta.core.carrier_field import ELECTRON_DENSITY
    from dynameta.sweep import BiasPoint
    msh_dir = msh_dir or tempfile.mkdtemp(prefix="hj3d_")
    out, timing = {}, {}
    for phys, model in (("equilibrium", "auto"), ("drift_diffusion", "thermionic"),
                        ("drift_diffusion", "continuous")):
        tag = "{}_{}".format(phys[:2], model[:4])
        t0 = time.time()
        sv = Devsim3DEquilibrium(spec_3d(phys, model), mesh_name="hj3m_" + tag, device_name="hj3d_" + tag,
                                 msh_path=os.path.join(msh_dir, tag + ".msh"))
        try:
            sv.build_device()
            timing[(phys, model, "build_s")] = time.time() - t0
            out[(phys, model, "ifaces")] = sv.heterointerfaces()
            for vg in vgs:
                cf = sv.solve(BiasPoint(label="g", voltages={"gate": float(vg)}))
                reg = next(iter(cf.regions.values()))
                out[(phys, model, float(vg))] = np.asarray(reg.node_fields[ELECTRON_DENSITY])
            timing[(phys, model, "total_s")] = time.time() - t0
        finally:
            sv.teardown()
    return out, timing


def report_3d():                                          # pragma: no cover - CLI
    out, timing = gate_f_3d()
    for k, v in timing.items():
        print("timing", k, "{:.1f} s".format(v))
    for vg in (0.0, 0.5):
        eq = out[("equilibrium", "auto", vg)]
        te = out[("drift_diffusion", "thermionic", vg)]
        ct = out[("drift_diffusion", "continuous", vg)]
        print("Vg={:+.2f}  max|TE-cont|/n = {:.3e}   max|DD-eq|/n = {:.3e}   n range {:.3e}..{:.3e}".format(
            vg, float(np.max(np.abs(te - ct) / ct)), float(np.max(np.abs(te - eq) / eq)), eq.min(), eq.max()))
    print("ifaces", out[("drift_diffusion", "thermionic", "ifaces")], out[("drift_diffusion", "continuous",
                                                                          "ifaces")])
    return out, timing


def main() -> int:                                        # pragma: no cover - CLI
    import pytest
    here = os.path.dirname(os.path.abspath(__file__))
    rc = pytest.main(["-q", os.path.join(os.path.dirname(here), "tests",
                                         "test_devsim_heterojunction.py"), "-p", "no:cacheprovider"])
    if "--3d" in sys.argv:
        report_3d()
    return rc


if __name__ == "__main__":                               # pragma: no cover
    with warnings.catch_warnings():
        sys.exit(main())
