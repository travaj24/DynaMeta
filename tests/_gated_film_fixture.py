"""Shared fixture for tests/test_carrier_native_grid.py and tests/test_mesh_screening_grading.py: the generic
gated-film cell. A degenerate semiconductor film ("TCO": n_bg 3e26 m^-3, DOS mass 0.30 m_e, eps_s 9.0,
equilibrium physics; its screening length is 0.777 nm) 6 nm thick between two-layer gate oxides (oxA 2 nm eps 9
outside, oxB 6 nm eps 20 next to the film, on both faces), on a metal back gate ("bot", on the 80 nm mirror
metal) under a metal top gate ("top", 40 nm), in a 40 nm cell. Laterally uniform and without an edge ground, so
every node column carries the same 1-D profile. Pure design construction; nothing here imports DEVSIM."""
import numpy as np

N_BG = 3.0e26
T_FILM = 6e-9
# an explicit fine mesh (legacy line rule, so the screening grading does not alter it): 189 nodes across the
# 6 nm film
FINE = dict(interface_ps_m=1.25e-11, coarse_ps_m=5e-11, interface_zone_m=2.25e-9,
            semiconductor_refinement="legacy")
X_LINES = dict(x_spacing_edge_m=4e-9, x_spacing_feature_edge_m=4e-9, x_spacing_feature_mid_m=4e-9)


def materials(n_bg=N_BG):
    from dynameta.materials import (M_E, ConstantOptical, DrudeOptical, Material, MaterialRegistry,
                                    TransportModel)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(1.45 ** 2 + 0j)))
    reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j), eps_static_dc=9.0))
    reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j), eps_static_dc=20.0))
    reg.add(Material("metal", ConstantOptical(-60.0 + 4.0j), is_metal=True))
    reg.add(Material("mirror_metal", ConstantOptical(-120.0 + 20.0j), is_metal=True))
    reg.add(Material("TCO", optical=DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14),
                     transport=TransportModel(
                         n_bg_m3=n_bg, eps_static=9.0, physics="equilibrium",
                         dos_mass_kg_of_n_m3=lambda n: 0.30 * M_E * np.ones_like(np.asarray(n, float)))))
    return reg


def film_design(t_film=T_FILM, x_lines=None, n_bg=N_BG, **mesh):
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    stack = [("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"), ("lo_oxb", 6e-9, "oxB"),
             ("film", t_film, "TCO"), ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA"),
             ("gate", 40e-9, "metal")]
    m = dict(X_LINES if x_lines is None else x_lines)
    m.update(mesh)
    return Design(name="film", unit_cell=UnitCell.square(40e-9),
                  stack=Stack(layers=[Layer(n, t, mat) for n, t, mat in stack],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[Electrode("bot", "mirror", "full"), Electrode("top", "gate", "full")],
                  materials=materials(n_bg), mesh_2d=Mesh2DSpec(**m))


def column(device, x, region="film", name="Electrons"):
    """(y, values) of the raw DEVSIM nodes of the node column nearest x, ascending in y."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    X = np.array(ds.get_node_model_values(device=device, region=region, name="x"))
    Y = np.array(ds.get_node_model_values(device=device, region=region, name="y"))
    V = np.array(ds.get_node_model_values(device=device, region=region, name=name))
    xc = np.unique(np.round(X, 15))
    xc = xc[np.argmin(abs(xc - x))]
    k = abs(X - xc) < 1e-13
    o = np.argsort(Y[k])
    return Y[k][o], V[k][o]
