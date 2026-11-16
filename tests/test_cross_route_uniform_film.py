"""One laterally uniform graded film, every optical route, one answer (v0.16: the shared lateral-uniformity
test, the lateral-axis guard, the tensor-film topology and the native-carrier-grid flag).

Every optical route now asks core.lateral whether a gridded field is laterally uniform (one rule, one
tolerance) and checks a lateral axis with the same unit / span guard. This file runs the SAME laterally
uniform, z-graded film -- once as a scalar field (Drude film) and once as a uniaxial tensor field (the same
film with boundary-scattering damping) -- through run_pipeline with the bridge's 256 identical lateral samples
on a native non-uniform z grid (geometric grading toward both faces, like the DEVSIM mesh), and through every
route that takes it:

    route                      scalar   uniaxial
    layered TMM                  yes     refuses tensors (by design)
    Berreman bridge              yes     yes
    RCWA (Fourier) bridge        yes     yes
    PMM (modal) bridge           yes     yes
    PMM2D (2-D modal) bridge     yes     yes
    2-D TM finite elements       yes     yes

ORACLE. A laterally uniform stack has one answer; the planar routes (TMM, Berreman) are exact for the sliced
stack, so the Fourier and modal routes must agree with them to round-off (budget 1e-10: measured 2.7e-16 RCWA,
3.2e-13 PMM, 2.3e-14 PMM2D, 1.6e-16 TMM).
The finite-element route solves the piecewise-LINEAR interpolant of the native samples while the layered routes
slice one constant slab per native interval (the slab rule), so it sits at that model difference: measured
2.8e-7 / 2.7e-7 (scalar / uniaxial; the slab rule's O(h^2) on the native grid, not FEM error); budget 3e-6,
10x the measured difference and 280x under the film's bias modulation (8.3e-4), which the test also asserts.
The test is sign-sensitive: r is complex and p-polarized at 60 deg, where the film's normal response matters.
The PMM2D route has a test of its own: a TENSOR on the PMM2D pure engine needs lumenairy >= 5.43, so on an older
lumenairy its uniaxial case is reported as SKIPPED with that reason rather than dropped from a route list.

Before the fix it FAILED: the 2-D FEM took the 256 identical columns as a laterally structured grid,
resampled it at the finest native spacing, overflowed max_grid_lines and dropped the mesh lines (an
AdvisoryWarning, an error under the suite's filter); the RCWA bridge sent the uniaxial film to a 2-D stack
(correct here,
since the stack is planar, but 2-D at (2N+1)^2 cost and a Laurent rule on structured cells: see
test_rcwa_tensor_film_keeps_the_1d_rule in tests/test_lumenairy_bridge_review_fixes.py).
"""
import numpy as np
import pytest

pytest.importorskip("ngsolve")

from dynameta.constants import M_E                                              # noqa: E402

P = 300e-9
T_FILM = 6e-9
LAM_NM = 1200.0
NX = 256


def _native_z(t=T_FILM, n_half=40, h0=0.005e-9, ratio=1.2):
    """A non-uniform z grid graded geometrically toward both faces (like the carrier mesh)."""
    steps = h0 * ratio ** np.arange(n_half)
    half = np.concatenate([[0.0], np.cumsum(steps)])
    half = half[half < 0.5 * t]
    z = np.unique(np.concatenate([half, t - half[::-1], [0.5 * t]]))
    return z


class _Carrier:
    """A 2-D carrier field like the layered builder's: x on 256 samples across the period (endpoint
    inclusive), the stack axis on the native graded grid; accumulation at the top face."""

    def __init__(self):
        self.z = _native_z()
        self.x = np.linspace(0.0, P, NX)

    def solve(self, bp, **kw):
        from dynameta.core.carrier_field import ELECTRON_DENSITY, CarrierField, CarrierRegion
        z = self.z
        prof = 3e26 + float(bp.voltages["top"]) * 3e26 * np.exp(-(T_FILM - z) / 0.8e-9)
        n2 = np.tile(prof, (self.x.size, 1))                                   # (Nx, Nz)
        reg = CarrierRegion(name="semi", role="semiconductor", material="film", nodes_m=np.zeros((1, 2)),
                            node_fields={}, grid_axes_m={"x": self.x, "y": z},
                            grid_fields={ELECTRON_DENSITY: n2})
        return CarrierField(bias_label=bp.label, voltages=dict(bp.voltages), ndim=2, temperature_K=300.0,
                            regions={"semi": reg}, n_bg_by_region={"semi": 3e26}, unit_cell_m=(P, P))


class _Optics:
    def __init__(self, design):
        self.design = design

    def build(self):
        class G:
            class mesh:
                ne = nv = 0
        return G()

    def alignment(self):
        from dynameta.core import NM
        from dynameta.core.alignment import GeometryAlignment, RegionAlignment
        zlo, zhi = self.design.z_intervals()["film"]
        return GeometryAlignment(unit_scale=NM, region_alignments=[
            RegionAlignment("film", "semi", (0.0, P, 0.0, P, zlo, zhi), stack_axis="y")],
            fixed_eps_regions={})

    def mesh_regions(self):
        return ["film"]


def _design(uniaxial):
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import BoundaryScattering, ConstantOptical, DrudeOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("ox", ConstantOptical(complex(2.00 ** 2))))
    reg.add(Material("glass", ConstantOptical(complex(1.45 ** 2))))
    bnd = BoundaryScattering(specularity=0.2, perp_coefficient=0.6) if uniaxial else None
    reg.add(Material("film", DrudeOptical(4.0, 0.30 * M_E, 1.2e14, boundary=bnd)))
    return Design(name="xroute", unit_cell=UnitCell.square(P),
                  stack=Stack(layers=[Layer("bot", 6e-9, "ox"), Layer("film", T_FILM, "film"),
                                      Layer("top", 6e-9, "ox")],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="p", incidence_angle_deg=60.0))


def _routes(uniaxial, names=None):
    from dynameta.optics.fem_2d_tm import make_fem_2d_tm_optical_solver
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_berreman_solver, make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_pmm_solver, make_lumenairy_rcwa_solver)
    from dynameta.optics.tmm_reference import make_layered_tmm_solver
    out = [] if uniaxial else [("tmm", make_layered_tmm_solver)]
    out += [("berreman", make_lumenairy_berreman_solver),
            ("rcwa", lambda: make_lumenairy_rcwa_solver(n_orders=5)),
            ("pmm", make_lumenairy_pmm_solver),
            ("pmm2d", make_lumenairy_pmm2d_solver),
            ("fem2d", lambda: make_fem_2d_tm_optical_solver(order=6))]
    return [(k, mk()) for k, mk in out if names is None or k in names]


def _r_by_route(uniaxial, vt=1.0, names=None):
    import warnings
    from dynameta.pipeline import run_pipeline
    from dynameta.sweep import BiasPoint, Sweep
    design = _design(uniaxial)
    sweep = Sweep(bias_points=[BiasPoint({"top": vt}, "b")], wavelengths_nm=[LAM_NM])
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, solver in _routes(uniaxial, names):
            rows = run_pipeline(design, sweep, verbose=False, carrier_solver=_Carrier(),
                                optical_builder=_Optics(design), optical_solver=solver)
            out[name] = complex(rows[0].result.r)
    return out


_ALL_BUT_PMM2D = ("tmm", "berreman", "rcwa", "pmm", "fem2d")


@pytest.mark.parametrize("uniaxial", [False, True], ids=["scalar", "uniaxial"])
def test_every_route_gives_one_answer_for_a_laterally_uniform_graded_film(uniaxial):
    r = _r_by_route(uniaxial, names=_ALL_BUT_PMM2D)
    ref = r["berreman"]
    if not uniaxial:
        assert abs(r["tmm"] - ref) < 1e-12
    for name in ("rcwa", "pmm"):
        assert abs(r[name] - ref) < 1e-10, (name, r[name], ref)
    assert abs(r["fem2d"] - ref) < 3e-6, ("fem2d", r["fem2d"], ref)
    # discrimination: the film matters at this budget (bias modulation 8.3e-4 / 8.5e-4 >> every budget)
    r0 = _r_by_route(uniaxial, vt=0.0, names=("berreman",))["berreman"]
    assert abs(ref - r0) > 3e-4


@pytest.mark.parametrize("uniaxial", [False, True], ids=["scalar", "uniaxial"])
def test_pmm2d_route_gives_the_same_answer_for_a_laterally_uniform_graded_film(uniaxial):
    """The PMM2D (2-D modal) route on the same film against the Berreman route (budget 1e-10, as for the other
    modal routes; measured 2.3e-14). On a lumenairy without a tensor on the pure engine the uniaxial case is
    SKIPPED, naming the floor."""
    from dynameta.optics.lumenairy_bridge._common import FEATURE_FLOORS, lumenairy_has
    if uniaxial and not lumenairy_has("pmm2d_pure_tensor"):
        floor, what = FEATURE_FLOORS["pmm2d_pure_tensor"]
        pytest.skip("the uniaxial film on the PMM2D route needs lumenairy >= {} ({}); the bridge refuses it "
                    "by name below that".format(".".join(map(str, floor)), what))
    r = _r_by_route(uniaxial, names=("berreman", "pmm2d"))
    assert abs(r["pmm2d"] - r["berreman"]) < 1e-10, r


if __name__ == "__main__":                                                   # measurement helper
    for u in (False, True):
        r = _r_by_route(u)
        ref = r["berreman"]
        print("uniaxial" if u else "scalar", {k: "{:.3e}".format(abs(v - ref)) for k, v in r.items()}, ref)
