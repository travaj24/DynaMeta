"""Shared fixture of validation/equilibrium_density_gradient.py and tests/test_equilibrium_density_gradient.py:
a degenerate film (equilibrium physics, parabolic band, DOS mass = quantization mass m) gated through an oxide
stack on BOTH faces by metal gates ("bot" under, "top" over), in a narrow laterally uniform cell (one x cell
by default), and the matching dual-gate Schrodinger-Poisson oracle (carriers.sp_dual_gate). Pure design
construction: nothing here imports DEVSIM.

Stacks (each listed from the film outward): "std" = 6 nm eps 20 + 2 nm eps 9 (C = 1.70e-2 F/m^2), "thin" =
2 nm eps 25 (C = 0.111 F/m^2). Film: eps_s 9.0, Drude eps_inf 4.0, damping 1.2e14 rad/s (optics unused)."""
import numpy as np

STACKS = {"std": ((6e-9, 20.0), (2e-9, 9.0)), "thin": ((2e-9, 25.0),)}
EPS_S = 9.0


def materials(n_bg, m_kg, ox, band=None):
    from dynameta.materials import (ConstantOptical, DrudeOptical, Material, MaterialRegistry,
                                    TransportModel)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(1.45 ** 2 + 0j)))
    for i, (_t, e) in enumerate(STACKS[ox]):
        reg.add(Material("OX{}".format(i), ConstantOptical(3.0 + 0j), eps_static_dc=e))
    reg.add(Material("mirror_metal", ConstantOptical(-120.0 + 20.0j), is_metal=True))
    if band is None:                                  # parabolic: the DOS mass is the quantization mass
        tr = TransportModel(n_bg_m3=n_bg, eps_static=EPS_S, physics="equilibrium",
                            dos_mass_kg_of_n_m3=lambda n, m=m_kg: m * np.ones_like(np.asarray(n, float)))
    else:                                             # a Kane band (refused by the density gradient)
        tr = TransportModel(n_bg_m3=n_bg, eps_static=EPS_S, physics="equilibrium", band=band)
    reg.add(Material("FILM", optical=DrudeOptical(eps_inf=4.0, m_opt_kg=m_kg, gamma_rad_s=1.2e14), transport=tr))
    return reg


def design(t_film, n_bg, m_kg, ox="std", *, cell=4e-9, x_spacing=None, electrodes=None, mesh=None, band=None):
    """The film design. cell: the lateral period; x_spacing: the x mesh spacing (default: one cell); electrodes:
    a list of geometry.Electrode replacing the two full gates; mesh: extra Mesh2DSpec keywords."""
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    lo = [("lo{}".format(i), t, "OX{}".format(i)) for i, (t, _e) in enumerate(STACKS[ox])][::-1]
    up = [("up{}".format(i), t, "OX{}".format(i)) for i, (t, _e) in enumerate(STACKS[ox])]
    stack = [("mirror", 20e-9, "mirror_metal")] + lo + [("film", t_film, "FILM")] + up + [("gate", 20e-9, "mirror_metal")]
    xs = cell if x_spacing is None else x_spacing
    ms = dict(x_spacing_edge_m=xs, x_spacing_feature_edge_m=xs, x_spacing_feature_mid_m=xs)
    ms.update(mesh or {})
    if electrodes is None:
        electrodes = [Electrode("bot", "mirror", "full"), Electrode("top", "gate", "full")]
    return Design(name="eqdg", unit_cell=UnitCell.square(cell),
                  stack=Stack(layers=[Layer(n, t, mat) for n, t, mat in stack],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=electrodes, materials=materials(n_bg, m_kg, ox, band), mesh_2d=Mesh2DSpec(**ms))


def oracle(t_film, n_bg, m_kg, ox="std", nz=None):
    """The dual-gate Schrodinger-Poisson solver of the same film (h <= 0.0125 nm)."""
    from dynameta.carriers.sp_dual_gate import DualGateSchrodingerPoisson, GateStack
    gs = GateStack(STACKS[ox])
    nz = nz or max(401, int(round(t_film / 0.0125e-9)) + 1)
    return DualGateSchrodingerPoisson(t_film_m=t_film, bottom=gs, top=gs, n_bg_m3=n_bg, m_eff_kg=m_kg,
                                      eps_static=EPS_S, nz=nz)


def column(device, region="film", name="Electrons", x=None):
    """(z from the film's lower face, values) of the node column nearest x (default: the first column)."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    X = np.array(ds.get_node_model_values(device=device, region=region, name="x"))
    Y = np.array(ds.get_node_model_values(device=device, region=region, name="y"))
    V = np.array(ds.get_node_model_values(device=device, region=region, name=name))
    xc = np.unique(np.round(X, 15))
    xc = xc[0] if x is None else xc[np.argmin(abs(xc - x))]
    k = abs(X - xc) < 1e-13
    o = np.argsort(Y[k])
    return Y[k][o] - Y[k][o][0], V[k][o]


def compare(z, n, sol, n_bg, t_film):
    """DG profile (z, n) vs a DualGateSolution. Errors relative to S = max(n_bg t, the SP carrier sheet) (the
    film's carrier sheet, or its background sheet when it is depleted): dsheet = sheet-excess error / S, l1 =
    int |n - n_SP| dz / S; the same over n_bg t (dsheet_nt, l1_nt); peak positions."""
    from dynameta.core.numerics import trapz
    nsp = np.interp(z, sol.z_m, sol.n_m3)
    exc = float(trapz(n - n_bg, z))
    exc_sp = float(sol.sheet_excess_m2)
    nt = n_bg * t_film
    s = max(nt, nt + exc_sp)
    l1 = float(trapz(abs(n - nsp), z))
    return dict(exc=exc, exc_sp=exc_sp, dsheet=(exc - exc_sp) / s, l1=l1 / s, dsheet_nt=(exc - exc_sp) / nt,
                l1_nt=l1 / nt, zpeak=float(z[np.argmax(n)]), zpeak_sp=float(sol.z_m[np.argmax(sol.n_m3)]))


def body_design(t_film, n_bg, m_kg, *, t_ox=8e-9, eps_ox=20.0, cell=4e-9):
    """The 3-D builder's default topology in the 2-D builder: a metal under the film (the "bot" electrode's contact
    lands on the film: an ohmic body, Potential = bias, Fermi level 0), one oxide (t_ox, eps_ox) and the "top" gate.
    Used to cross-check Stacked3DSpec(density_gradient=...)."""
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material
    reg = materials(n_bg, m_kg, "std")
    reg.add(Material("OXB", ConstantOptical(3.0 + 0j), eps_static_dc=eps_ox))
    stack = [("mirror", 20e-9, "mirror_metal"), ("film", t_film, "FILM"), ("ox", t_ox, "OXB"),
             ("gate", 20e-9, "mirror_metal")]
    return Design(name="eqdg_body", unit_cell=UnitCell.square(cell),
                  stack=Stack(layers=[Layer(n, t, mat) for n, t, mat in stack],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[Electrode("bot", "mirror", "full"), Electrode("top", "gate", "full")],
                  materials=reg, mesh_2d=Mesh2DSpec(x_spacing_edge_m=cell, x_spacing_feature_edge_m=cell,
                                                    x_spacing_feature_mid_m=cell))
