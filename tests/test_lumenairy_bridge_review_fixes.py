"""v0.16 fixes in the lumenairy bridges: the tensor-film stack topology, the per-point factorization rule
and the Li-companion advisory. Needs lumenairy (a core dependency); no solver stack.

Tensor-film topology. A tensor film made the RCWA bridge build a 2-D stack (and so pick Laurent's rule)
whatever the tensor was: on the strip cell of tests/_strip_cell_fixture the same graded film passed as a
scalar field runs Li's rule on a 1-D stack (|dr| 0.147 against the FEM at n_orders 6), and as an isotropic
diagonal tensor it ran Laurent on a 2-D stack at (2N+1)^2 cost. A y-invariant tensor that does not couple y is
1-D-compatible now; a tensor that couples y keeps the 2-D stack.
"""
import warnings

import numpy as np
import pytest

import _strip_cell_fixture as F


def _as_tensor(ef, zz_scale=1.0):
    from dynameta.core.eps_field import EpsField
    v = ef.values_zyx
    t = np.zeros(v.shape + (3, 3), dtype=complex)
    t[..., 0, 0] = t[..., 1, 1] = v
    t[..., 2, 2] = v * zz_scale
    return EpsField(x_axis_u=ef.x_axis_u, y_axis_u=ef.y_axis_u, z_axis_u=ef.z_axis_u, values_zyx=t)


def test_rcwa_tensor_film_keeps_the_1d_rule():
    """Before the fix the tensor film's topology was (is_2d=True, ...) and the rule 'laurent'. Now the isotropic
    tensor gives the scalar field's r to round-off (both Li's rule on a 1-D stack)."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    from dynameta.optics.lumenairy_bridge.rcwa_backend import _stack_topology
    design = F.strip_design()
    ef = F.graded_film_field(nz=21)
    ef_t = _as_tensor(ef)
    assert _stack_topology(design, {"film": ef_t})[0] is False
    out = {}
    for lab, field in (("scalar", ef), ("tensor", ef_t)):
        s = make_lumenairy_rcwa_solver(n_orders=6)
        with F.li_companion_floor():
            res = s(design, None, {"film": field}, F.LAM, None, None)
        out[lab] = (complex(res.r), s.last_formulation)
    assert out["scalar"][1] == out["tensor"][1] == "li"
    assert abs(out["scalar"][0] - out["tensor"][0]) < 1e-10
    # sign-sensitive discrimination: a genuinely uniaxial film (eps_zz != eps_xx) moves r
    s = make_lumenairy_rcwa_solver(n_orders=6)
    with F.li_companion_floor():
        r_u = complex(s(design, None, {"film": _as_tensor(ef, zz_scale=1.5)}, F.LAM, None, None).r)
    assert abs(r_u - out["scalar"][0]) > 1e-4


def test_rcwa_tensor_coupling_y_stays_2d():
    """A tensor with an xy (or yz) component couples TE and TM: the stack stays 2-D (Laurent rule)."""
    from dynameta.optics.lumenairy_bridge.rcwa_backend import _stack_topology
    design = F.strip_design()
    ef_t = _as_tensor(F.graded_film_field(nz=5))
    v = np.array(ef_t.values_zyx)
    v[..., 0, 1] = v[..., 1, 0] = 0.05
    from dynameta.core.eps_field import EpsField
    ef_c = EpsField(x_axis_u=ef_t.x_axis_u, y_axis_u=ef_t.y_axis_u, z_axis_u=ef_t.z_axis_u, values_zyx=v)
    assert _stack_topology(design, {"film": ef_c})[0] is True
    assert _stack_topology(design, {"film": EpsField(tensor=np.diag([2.0, 2.0, 3.0]).astype(complex))})[0] is False
    assert _stack_topology(design, {"film": EpsField(tensor=np.array(
        [[2.0, 0, 0], [0, 2.0, 0.1], [0, 0.1, 3.0]], dtype=complex))})[0] is True


@pytest.mark.parametrize("pol,theta,phi", [("x", 0.0, 0.0), ("p", 40.0, 0.0), ("p", 40.0, 30.0),
                                           ("y", 40.0, 30.0)])
def test_rcwa_uniform_uniaxial_stack_on_1d_matches_berreman(pol, theta, phi):
    """A planar stack holding a uniform uniaxial tensor is now a 1-D (clamped) stack; it must still equal the
    Berreman 4x4 oracle, including at conical incidence (the 1-D tensor path's conical branch)."""
    from dynameta.core.eps_field import EpsField
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    from dynameta.optics.lumenairy_bridge import (design_to_rcwa_stack, make_lumenairy_berreman_solver,
                                                  make_lumenairy_rcwa_solver)
    reg = MaterialRegistry()
    for n, e in (("air", 1.0), ("glass", 2.25), ("film", 2.4), ("ox", 3.0)):
        reg.add(Material(n, ConstantOptical(complex(e))))
    d = Design(name="u", unit_cell=UnitCell.square(500e-9),
               stack=Stack(layers=[Layer("ox", 20e-9, "ox"), Layer("film", 30e-9, "film")],
                           superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=reg,
               optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta, azimuth_deg=phi))
    ebr = {"film": EpsField(tensor=np.diag([-0.4 + 0.3j, -0.4 + 0.3j, 2.0 + 0.1j]))}
    stk, _ = design_to_rcwa_stack(d, 1.2e-6, eps_by_region=ebr, n_orders=11)
    assert stk.n_orders_x == 1 and bool(getattr(stk, "is_1d", True))
    r = make_lumenairy_rcwa_solver()(d, None, ebr, 1.2e-6, None, None)
    b = make_lumenairy_berreman_solver()(d, None, ebr, 1.2e-6, None, None)
    assert max(abs(r.R - b.R), abs(r.T - b.T), abs(r.r - b.r)) < 1e-12



def test_rcwa_auto_rule_is_a_function_of_the_point_only(tmp_path):
    """The factorization rule is a function of the point. A strip of a Drude metal whose Re(eps) changes sign
    inside the band (0.43 at 1000 nm, -5.1 at 1600 nm): before the fix a single solve at 1000 nm used Laurent
    while the sweep [1000, 1600] used Li for both, and the same solve_sweep request through a cold cache and
    through one pre-warmed at 1600 nm gave different r(1000). Now 'auto' is resolved per point on every path:
    single == sweep == either cache, bit for bit."""
    from dynameta.cache import cached_optical_solver
    from dynameta.constants import M_E
    from dynameta.core.eps_field import EpsField
    from dynameta.materials import ConstantOptical, DrudeOptical, Material
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    dr = DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)

    class DMetal(ConstantOptical):
        def __init__(self):
            super().__init__(1.0 + 0j)

        def eps(self, lambda_m, **kw):
            return complex(dr.eps(lambda_m, n_m3=1.2e27))

    def E(lam):
        return {"film": EpsField(scalar=F.eps_film(lam=lam))}
    lams = [1000e-9, 1600e-9]
    d = F.strip_design(strip_material="DM", extra=[Material("DM", DMetal())])
    s = make_lumenairy_rcwa_solver(n_orders=11)
    r_single = complex(s(d, None, E(lams[0]), lams[0], 1.0, F.N_GLASS).r)
    f_single = s.last_formulation
    with F.li_companion_floor():                                        # 1600 nm runs Li
        r_sweep = [complex(o.r) for o in s.solve_sweep(d, None, E, lams, 1.0, F.N_GLASS)]
    assert r_sweep[0] == r_single and f_single == "laurent"
    assert s.last_formulation == "li"                                   # 1600 nm: the metal is hard there
    c1 = cached_optical_solver(make_lumenairy_rcwa_solver(n_orders=11), str(tmp_path / "a.h5"))
    c2 = cached_optical_solver(make_lumenairy_rcwa_solver(n_orders=11), str(tmp_path / "b.h5"))
    with F.li_companion_floor():
        cold = c1.solve_sweep(d, None, E, lams, 1.0, F.N_GLASS)
    with F.li_companion_floor():
        c2.solve_sweep(d, None, E, [lams[1]], 1.0, F.N_GLASS)
    warm = c2.solve_sweep(d, None, E, lams, 1.0, F.N_GLASS)
    assert complex(cold[0].r) == complex(warm[0].r) == r_single


def test_li_without_its_inverse_rule_companion_is_said(monkeypatch):
    """Li's rule on an area raster without the harmonic companion (lumenairy < 5.31, emulated here; or a
    lossless sign change inside a pixel) is 3.6e-3 / 1.0e-2 / 4.1e-3 off in r on the strip cell at the default
    8192-pixel lattice (n_orders 11 / 30 / 50, against the analytic layer), against 2.9e-5 / 1.0e-4 / 1.6e-4
    with it.
    Before the fix it ran silently; now one AdvisoryWarning per solver names the cause. With the companion (lumenairy >= 5.31) the path stays silent."""
    import dynameta.optics.lumenairy_bridge.rcwa_backend as RB
    from dynameta.core.eps_field import EpsField
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    from dynameta.warning_types import AdvisoryWarning
    d, ebr = F.strip_design(), F.ebr()
    if RB._lumenairy_has("rcwa_eps_cell_normal"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            make_lumenairy_rcwa_solver(n_orders=11, inclusion_model="raster")(d, None, ebr, F.LAM, None, None)
    real = RB._lumenairy_has
    monkeypatch.setattr(RB, "_lumenairy_has", lambda n: False if n == "rcwa_eps_cell_normal" else real(n))
    s = make_lumenairy_rcwa_solver(n_orders=11, inclusion_model="raster")
    with pytest.warns(AdvisoryWarning, match="companion.*5.31"):
        s(d, None, ebr, F.LAM, None, None)
    monkeypatch.setattr(RB, "_lumenairy_has", real)
    if not real("rcwa_eps_cell_normal"):
        return
    # a lossless gridded film crossing zero laterally: no harmonic mean inside the crossing pixel
    x = np.linspace(0.0, F.P * 1e9, 64, endpoint=False)
    prof = 2.0 - 4.0 * np.exp(-((x - 0.5 * F.P * 1e9) / 40.0) ** 2)            # real, through zero
    v = np.tile(prof, (2, 1, 1))
    film = EpsField(x_axis_u=x, y_axis_u=np.array([0.0]), z_axis_u=np.array([0.0, F.T_FILM_NM]), values_zyx=v)
    with pytest.warns(AdvisoryWarning, match="lossless sign change"):
        make_lumenairy_rcwa_solver(n_orders=11)(d, None, {"film": film}, F.LAM, None, None)
