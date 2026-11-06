"""1-D fixtures for the v0.15 DC well-posedness gates (plan 4.4.2), tests/test_well_posedness.py.

  slab           the R26 uniform slab (tests/test_bipolar_recombination_bgn_ionization.py _device /
                 _slab) with a choice of contact on its 'lo' end -- 'potential' (the pre-2026-11-03
                 FLOATING fixture: Potential-only), 'majority' (the fix: psi + the majority density),
                 'electrons' (psi + Electrons on a p-type slab: the minority pinned, holes free),
                 'carriers' (both densities pinned, the potential NOT), or 'ohmic'
                 (setup_contact_ohmic_bipolar). The 'hi' end carries no contact equation unless
                 `hi_contact` names one of the same kinds.
  mos_inversion  plan P4 1.1's near-floating MOS: p-type 3e23, 150 nm, wall mesh 0.05 nm, classical
                 Boltzmann DD, Dirichlet psi at the wall, ohmic substrate.
"""
import numpy as np

from _abs_tol_fixtures import NI, _mesh_1d, drop, ds, quiet   # noqa: F401  (re-exported helpers)
from dynameta.carriers import eq_registry as _R


def _v(dev, region, name):
    return np.asarray(ds.get_node_model_values(device=dev, region=region, name=name), dtype=np.float64)


def slab(dev, *, nd=1e25, contact="potential", tau=1e-7, length=200e-9, options=None,
         hi_contact=None, extended=False):
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    if length > 2e-6:
        lines = [(0.0, 5e-9, "lo"), (2e-6, 100e-9, None), (length, length / 40.0, "hi")]
    else:
        lines = [(0.0, 5e-9, "lo"), (length, 5e-9, "hi")]
    _mesh_1d(dev, lines, [("bulk", "Si", "lo", "hi")])
    if extended:
        from dynameta.carriers import extended_precision as XP
        XP.enable(dev)
    kw = {} if options is None else {"options": options}
    with quiet():
        ds.node_model(device=dev, region="bulk", name="NetDoping", equation="{:.10e}".format(nd))
        BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=2.8e25, n_dos_p_m3=1.04e25,
                                n_i_m3=NI, mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048,
                                tau_n_s=tau, tau_p_s=tau, **kw)
        BP.setup_equilibrium_seed_models(dev, "bulk")
        ds.node_model(device=dev, region="bulk", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "bulk")))
        for var, src in (("Potential", "_seed_psi"), ("Electrons", "IntrinsicElectrons"),
                         ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="bulk", name=var, values=list(_v(dev, "bulk", src)))
        for c, kind in (("lo", contact), ("hi", hi_contact)):
            if kind is None:
                continue
            if kind == "ohmic":
                BP.setup_contact_ohmic_bipolar(dev, c)
                continue
            i = 0 if c == "lo" else -1
            if kind != "carriers":                   # 'carriers': both densities, NO potential
                PE.setup_contact(dev, c)
                ds.set_parameter(device=dev, name="{}_bias".format(c),
                                 value=float(_v(dev, "bulk", "_seed_psi")[i]))
            if kind == "potential":
                continue
            if kind == "majority":
                pins = ["Electrons" if nd > 0 else "Holes"]
            elif kind == "carriers":
                pins = ["Electrons", "Holes"]
            else:
                pins = [{"electrons": "Electrons", "holes": "Holes"}[kind]]
            for var in pins:
                eq = "ElectronContinuityEquation" if var == "Electrons" else "HoleContinuityEquation"
                cur = "ElectronCurrent" if var == "Electrons" else "HoleCurrent"
                cm = "{}_{}_pin".format(c, var.lower())
                ds.set_parameter(device=dev, name=cm + "_value", value=float(_v(dev, "bulk", var)[i]))
                ds.contact_node_model(device=dev, contact=c, name=cm,
                                      equation="{} - {}_value".format(var, cm))
                ds.contact_node_model(device=dev, contact=c, name="{}:{}".format(cm, var), equation="1")
                _R.record_contact_equation(dev, c, name=eq, node_model=cm, edge_current_model=cur)
    return "bulk"


def mos_inversion(dev, *, N_A=3e23, length=150e-9, extended=False):
    """Returns (set_band_bending(delta_V), psi_b)."""
    from dynameta.carriers import extended_precision as XP
    from dynameta.carriers import physics_bipolar_dd as BP
    _mesh_1d(dev, [(0.0, 0.05e-9, "wall"), (5e-9, 0.1e-9, None), (25e-9, 0.5e-9, None),
                   (length, 2e-9, "sub")], [("s", "Si", "wall", "sub")])
    if extended:
        XP.enable(dev)
    with quiet():
        ds.node_model(device=dev, region="s", name="NetDoping", equation="{:.17e}".format(-N_A))
        BP.setup_bipolar_region(dev, "s", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=NI,
                                mobility_n_m2Vs=0.1, mobility_p_m2Vs=0.04, tau_n_s=1e-7,
                                tau_p_s=1e-7, n_dos_p_m3=1.04e25)
        BP.setup_equilibrium_seed_models(dev, "s")
        BP.setup_contact_ohmic_bipolar(dev, "sub")
        for nm, src in (("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="s", name=nm, init_from=src)
        psi_b = BP.ohmic_built_in_offset_V(-N_A, n_i_m3=NI, n_dos_m3=2.8e25, n_dos_p_m3=1.04e25)
        ds.set_parameter(device=dev, name="wall_bias", value=psi_b)
        ds.contact_node_model(device=dev, contact="wall", name="wall_psi",
                              equation="Potential - wall_bias")
        ds.contact_node_model(device=dev, contact="wall", name="wall_psi:Potential", equation="1")
        _R.record_contact_equation(dev, "wall", name="PotentialEquation", node_model="wall_psi")
        nn = _v(dev, "s", "x").size
        ds.set_node_values(device=dev, region="s", name="Potential", values=[psi_b] * nn)

    def band_bending(delta):
        ds.set_parameter(device=dev, name="wall_bias", value=psi_b + delta)
    return band_bending, psi_b
