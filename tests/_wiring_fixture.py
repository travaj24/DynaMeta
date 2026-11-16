"""Shared fixture for the v0.16 carrier-wiring tests (tests/test_layered_edge_reference.py,
tests/test_layered_charges.py, tests/test_pipeline_native_tmm.py): the generic gated-film cell of
tests/_gated_film_fixture.py (film "TCO": n_bg 3e26 m^-3, DOS mass 0.30 m_e, eps_s 9.0, equilibrium physics,
screening length 0.777 nm, 6 nm thick, between 6 nm oxB (eps 20) + 2 nm oxA (eps 9) on both faces, oxB next to
the film, metal back gate "bot" and top gate "top", 40 nm cell, x mesh lines every 4 nm), with the options the
wiring needs: an edge electrode on the film, gate work functions, the film's electron affinity, the film split
into two equilibrium layers, and an optically transparent top gate (a metal for the carrier solve, a weakly
absorbing dielectric constant for the optics). Pure design construction; nothing here imports DEVSIM."""
import numpy as np

N_BG = 3.0e26
T_FILM = 6e-9
M_DOS_REL = 0.30
EPS_S = 9.0
CHI_EV = 4.5
X_LINES = dict(x_spacing_edge_m=4e-9, x_spacing_feature_edge_m=4e-9, x_spacing_feature_mid_m=4e-9)
# the top dielectric stack between the film and the top gate: (thickness, eps_r), film side first
TOP_STACK = ((6e-9, 20.0), (2e-9, 9.0))
EPS0 = 8.8541878128e-12
C_TOP = EPS0 / sum(t / e for t, e in TOP_STACK)          # series capacitance per area [F/m^2]


def materials(n_bg=N_BG, chi_eV=CHI_EV, dos_mass_fn=None):
    from dynameta.materials import (M_E, ConstantOptical, DrudeOptical, Material, MaterialRegistry,
                                    TransportModel)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(1.45 ** 2 + 0j)))
    reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j), eps_static_dc=9.0))
    reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j), eps_static_dc=20.0))
    reg.add(Material("metal", ConstantOptical(-60.0 + 4.0j), is_metal=True))
    reg.add(Material("mirror_metal", ConstantOptical(-120.0 + 20.0j), is_metal=True))
    # a gate metal for the carrier solve that is a thin, weakly absorbing film for the optics
    reg.add(Material("TGATE", ConstantOptical(3.8 + 0.05j), is_metal=True))
    fn = dos_mass_fn or (lambda n: M_DOS_REL * M_E * np.ones_like(np.asarray(n, float)))
    reg.add(Material("TCO", optical=DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14),
                     transport=TransportModel(n_bg_m3=n_bg, eps_static=EPS_S, physics="equilibrium",
                                              dos_mass_kg_of_n_m3=fn, chi_eV=chi_eV)))
    return reg


def film_design(*, t_film=T_FILM, edges=(), wf_top=None, wf_bot=None, split_film=False,
                transparent_gate=False, x_lines=None, mats=None, **mesh):
    """The gated film. `edges`: extra Electrode objects (edge electrodes on 'film' / 'film_a' / 'film_b');
    split_film: the film as two equilibrium layers 'film_a' (lower half) and 'film_b' (upper half)."""
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    film = ([("film_a", 0.5 * t_film, "TCO"), ("film_b", 0.5 * t_film, "TCO")] if split_film
            else [("film", t_film, "TCO")])
    gate = ("gate", 10e-9, "TGATE") if transparent_gate else ("gate", 40e-9, "metal")
    stack = ([("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"), ("lo_oxb", 6e-9, "oxB")] + film
             + [("up_oxb", TOP_STACK[0][0], "oxB"), ("up_oxa", TOP_STACK[1][0], "oxA"), gate])
    m = dict(X_LINES if x_lines is None else x_lines)
    m.update(mesh)
    el = [Electrode("bot", "mirror", "full", work_function_eV=wf_bot),
          Electrode("top", "gate", "full", work_function_eV=wf_top)] + list(edges)
    return Design(name="film", unit_cell=UnitCell.square(40e-9),
                  stack=Stack(layers=[Layer(n, t, mat) for n, t, mat in stack],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=el, materials=mats if mats is not None else materials(), mesh_2d=Mesh2DSpec(**m))


def edge(name="gnd", layer="film", side="x_lo", *, v=0.0, role="ground", pin=False):
    from dynameta.geometry import Electrode
    return Electrode(name, layer, side, role=role, fixed_voltage_V=v, pin_potential=pin)


def node_columns(ds, device, region="film", name="Electrons"):
    """{x: values ascending in y} for every node column of `region` (raw DEVSIM nodes)."""
    X = np.array(ds.get_node_model_values(device=device, region=region, name="x"))
    Y = np.array(ds.get_node_model_values(device=device, region=region, name="y"))
    V = np.array(ds.get_node_model_values(device=device, region=region, name=name))
    kx = np.round(X / 1e-15).astype(np.int64)
    out = {}
    for k in np.unique(kx):
        sel = kx == k
        o = np.argsort(Y[sel])
        out[float(k) * 1e-15] = V[sel][o]
    return out


class Snapshot:
    """What a test reads from a solved LayeredDevsimBuilder, taken so the device can be torn down before the
    next one is solved (DEVSIM's Newton solves EVERY live device jointly, so a second live device would be
    iterated -- and its convergence tested -- together with the first)."""

    def __init__(self, ds, b):
        from dynameta.carriers import physics_equilibrium as PE
        self.raw = {}
        for r in ds.get_region_list(device=b.device):
            models = ds.get_node_model_list(device=b.device, region=r)
            for nm in ("x", "y", "Potential", "Electrons"):
                if nm in models:
                    self.raw[(r, nm)] = np.array(ds.get_node_model_values(device=b.device, region=r, name=nm))
        self.contacts = list(ds.get_contact_list(device=b.device))
        self.fermi_references = b.fermi_references()
        self.frames = b.band_frames()
        self.wf_offset = dict(b._wf_offset)
        self.ef_ref = {r: float(ds.get_parameter(device=b.device, region=r, name=PE.FERMI_REF_PARAM))
                       for g in self.fermi_references.values() for r in g}

    def cols(self, region="film", name="Electrons"):
        X, Y, V = self.raw[(region, "x")], self.raw[(region, "y")], self.raw[(region, name)]
        kx = np.round(X / 1e-15).astype(np.int64)
        out = {}
        for k in np.unique(kx):
            sel = kx == k
            o = np.argsort(Y[sel])
            out[float(k) * 1e-15] = V[sel][o]
        return out

    def top(self, region="film", name="Potential", x=None):
        cols = self.cols(region, name)
        xs = np.array(sorted(cols))
        xc = xs[len(xs) // 2] if x is None else xs[np.argmin(np.abs(xs - x))]
        return float(cols[xc][-1])


def top_face_value(ds, device, region="film", name="Potential", x=None):
    """The value at the film's top face (largest y) on the column nearest x (the cell centre by default)."""
    cols = node_columns(ds, device, region, name)
    xs = np.array(sorted(cols))
    xc = xs[len(xs) // 2] if x is None else xs[np.argmin(np.abs(xs - x))]
    return float(cols[xc][-1])
