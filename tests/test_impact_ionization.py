"""Tests for the Chynoweth impact-ionization driver (D4). The first block is solver-free; the second runs
DEVSIM on hand-built 2-D devices whose state is SET to an exact discrete solution (no Newton), so the gates
see only the I_sub quadrature. The device-level oracles (constant-field bar closed form, reverse-bias
junction growth, mesh convergence) live in validation/impact_ionization_isub.py.

2-D QUADRATURE (v0.14.0, consolidated plan row 3.2-12). Until v0.14.0 the 2-D I_sub was the directional
edge quadrature sum_e alpha(|E_e|)|J_e| c_e L_e, which on DEVSIM's tensor mesh converges to
Int [alpha(|E_x|)|J_x| + alpha(|E_y|)|J_y|] dA instead of Int alpha(|E|)|J| dA. The default is now the
isotropic element quadrature (the 3-D kernel, shared); quadrature="edge" reproduces the old code path
verbatim (the edge branch is unchanged), so every gate below is also run on it as "the old code".

  U1  uniform field at 0/30/45/60 deg to the mesh: element == alpha(E) J Area (closed form, here) to 1e-12;
      edge == the componentwise closed form [alpha(E cos t) J cos t + alpha(E sin t) J sin t] Area to
      1e-12. MEASURED, b_n/E = 4: edge/truth = 1.000000 / 0.475590 / 0.269744 / 0.475590 -- the old code
      FAILS the element gate by 0.52 / 0.73 / 0.52 at 30 / 45 / 60 deg; |element/truth - 1| <= 7e-16.
  H1  harmonic psi = V0 (x^2 - y^2)/L^2 on [L, 2L]^2 (field direction 27-63 deg to the mesh; the box
      method is exact for a quadratic, so the set state IS the discrete solution): element converges at
      order 2.000 to the 400 x 400 Gauss-Legendre integral of alpha(|E|)|J| (Richardson limit 6e-10);
      edge converges at order 2.000 to the componentwise integral, 0.359 of the truth -- the old code
      FAILS the limit gate by 0.64.
  R1  ValueErrors: quadrature='element' in 1-D, 'edge' in 3-D, an unknown name.
  G1  side-effect hygiene: element models created are deleted; pre-existing ElectricField_x/y are only
      read (values bit-identical).
  X1  refusal: a 2-D mesh with an obtuse triangle raises InexactElementGeometryError (the shared check;
      tests/test_devsim_element_geometry.py has the full matrix).
"""
import contextlib
import math
import os
import sys

import numpy as np
import pytest

from dynameta.carriers.impact_ionization import ChynowethParams, SILICON_VANOVERSTRAETEN


def test_alpha_closed_form_and_masking():
    si = SILICON_VANOVERSTRAETEN
    for E in (3.0e7, 6.0e7, 1.2e8):
        assert si.alpha_n(E) == pytest.approx(7.03e7 * math.exp(-1.231e8 / E), rel=1e-12)
        assert si.alpha_p(E) == pytest.approx(1.582e8 * math.exp(-2.036e8 / E), rel=1e-12)
    assert si.alpha_n(0.0) == 0.0
    assert si.alpha_p(-1.0) == 0.0
    arr = si.alpha_n(np.array([0.0, 5.0e7, -2.0]))
    assert arr[0] == 0.0 and arr[2] == 0.0 and arr[1] == si.alpha_n(5.0e7)


def test_si_constants_are_si_not_cgs():
    # the literature cm trap is a factor 100 either way; alpha_n(3e7 V/m) ~ 1.2e6 1/m
    a = SILICON_VANOVERSTRAETEN.alpha_n(3.0e7)
    assert 1.0e5 < a < 1.0e7


def test_alpha_monotone_increasing_in_field():
    si = SILICON_VANOVERSTRAETEN
    E = np.linspace(2.0e7, 2.0e8, 50)
    assert np.all(np.diff(si.alpha_n(E)) > 0.0)
    assert np.all(np.diff(si.alpha_p(E)) > 0.0)


def test_guards():
    with pytest.raises(ValueError):
        ChynowethParams(a_n_per_m=-1.0, b_n_V_per_m=1.0, a_p_per_m=0.0, b_p_V_per_m=1.0)
    pytest.importorskip("devsim")
    from dynameta.carriers.impact_ionization import substrate_current
    with pytest.raises(ValueError):
        substrate_current("dev", "reg", SILICON_VANOVERSTRAETEN, depth_m=0.0)


# ------------------------------------------------------------------------------------------------
# 2-D quadrature gates (DEVSIM, state SET to an exact discrete solution -- see the module docstring)
# ------------------------------------------------------------------------------------------------
Q = 1.602176634e-19
ND, NI, MU_N, MU_P = 1e22, 1e16, 0.135, 0.048
PSI_EQ = 0.357                       # ~ V_t ln(N_D/n_i); any offset: only differences enter E and J
L_SLAB = 100e-9


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)


def _ds():
    pytest.importorskip("devsim")
    from dynameta.carriers.devsim_env import import_devsim
    return import_devsim()


def _drop(dev):
    ds = _ds()
    from dynameta.carriers import eq_registry as _R
    if dev in ds.get_device_list():
        ds.delete_device(device=dev)
    if dev + "_m" in ds.get_mesh_list():
        ds.delete_mesh(mesh=dev + "_m")
    _R.clear(dev)


def _setup_region(ds, dev):
    from dynameta.carriers import physics_bipolar_dd as BP
    ds.node_model(device=dev, region="si", name="NetDoping", equation=repr(ND))
    BP.setup_bipolar_region(dev, "si", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=NI, mobility_n_m2Vs=MU_N,
                            mobility_p_m2Vs=MU_P, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=False)


def _slab(dev, x0, lx, y0, ly, nx, ny):
    """Uniform n-type (1e22) bipolar region on DEVSIM's own 2-D tensor mesh (create_2d_mesh: right
    triangles), no contacts -- the state is set, never solved."""
    ds = _ds()
    _drop(dev)
    m = dev + "_m"
    with _quiet():
        ds.create_2d_mesh(mesh=m)
        for x in (x0, x0 + lx):
            ds.add_2d_mesh_line(mesh=m, dir="x", pos=x, ps=lx / nx)
        for y in (y0, y0 + ly):
            ds.add_2d_mesh_line(mesh=m, dir="y", pos=y, ps=ly / ny)
        ds.add_2d_region(mesh=m, material="Si", region="si")
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=dev)
        _setup_region(ds, dev)


def _set_state(dev, psi_fn):
    """Potential = psi_fn(x, y), n = N_D, p = n_i^2/N_D everywhere. With n uniform the SG edge current is
    exactly q mu n E_e (n (B(-v) - B(v)) = n v), and for psi linear (U1) or a quadratic harmonic (H1) the
    box-method divergence of J and the discrete Laplacian of psi vanish at every node (exact on a tensor
    grid), while the charge and n p - n_i^2 are 0: the set state solves the discrete Poisson and both
    continuity equations exactly, for Dirichlet data equal to it on the boundary."""
    ds = _ds()
    x = np.asarray(ds.get_node_model_values(device=dev, region="si", name="x"))
    y = np.asarray(ds.get_node_model_values(device=dev, region="si", name="y"))
    ds.set_node_values(device=dev, region="si", name="Potential", values=psi_fn(x, y).tolist())
    ds.set_node_values(device=dev, region="si", name="Electrons", values=[ND] * x.size)
    ds.set_node_values(device=dev, region="si", name="Holes", values=[NI * NI / ND] * x.size)


def _j_closed(pars, E):
    """alpha(E) |J| summed over carriers, J = q mu n E for the uniform state (the oracle integrand)."""
    return (pars.alpha_n(E) * Q * MU_N * ND * E) + (pars.alpha_p(E) * Q * MU_P * (NI * NI / ND) * E)


E0 = 1.0e7                                                   # V/m; b_n / E0 = 4 (the plan's 0.270 point)
PARS_U = ChynowethParams(a_n_per_m=1.0e6, b_n_V_per_m=4.0 * E0, a_p_per_m=2.0e6, b_p_V_per_m=6.0 * E0)


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 45.0, 60.0])
def test_u1_uniform_field_at_any_angle_element_exact_edge_componentwise(theta_deg):
    """U1. Bound 1e-12 (error budget): E_e = (psi_0 - psi_1)/L_e carries eps |psi|/|d psi| ~ 2e-16 x 7 =
    1.6e-15 relative (psi ~ 0.36 V, 50 mV per edge); alpha amplifies it by 1 + b/E <= 7; summed over ~480
    triangles with random signs -> ~1e-14. MEASURED <= 1.3e-15 (both forms). The old (edge) code fails the
    element gate by 1 - 0.269744 at 45 deg (module docstring)."""
    from dynameta.carriers.impact_ionization import substrate_current
    t = math.radians(theta_deg)
    dev = "ii2d_u{:.0f}".format(theta_deg)
    _slab(dev, 0.0, L_SLAB, 0.0, 0.6 * L_SLAB, 20, 12)
    try:
        _set_state(dev, lambda x, y: PSI_EQ - E0 * (x * math.cos(t) + y * math.sin(t)))
        ds = _ds()
        Ee = np.abs(np.asarray(ds.get_edge_model_values(device=dev, region="si", name="ElectricField")))
        Je = np.abs(np.asarray(ds.get_edge_model_values(device=dev, region="si", name="ElectronCurrent")))
        on = Ee > 0.0
        # the fixture's premise on the DISCRETE device: J_e = q mu N E_e on every edge
        assert np.max(np.abs(Je[on] / (Q * MU_N * ND * Ee[on]) - 1.0)) < 1e-13
        el = substrate_current(dev, "si", PARS_U)                 # default: element in 2-D
        el_x = substrate_current(dev, "si", PARS_U, quadrature="element")
        ed = substrate_current(dev, "si", PARS_U, quadrature="edge")
    finally:
        _drop(dev)
    area = L_SLAB * 0.6 * L_SLAB
    truth = _j_closed(PARS_U, E0) * area
    comp = (_j_closed(PARS_U, E0 * abs(math.cos(t))) + _j_closed(PARS_U, E0 * abs(math.sin(t)))) * area
    assert el == el_x
    assert abs(el / truth - 1.0) < 1e-12, (el, truth)
    assert abs(ed / comp - 1.0) < 1e-12, (ed, comp)              # the old form IS the componentwise one
    # adversarial: a 1e-9 change of |E| moves the truth by (1 + b/E) 1e-9 = 5e-9 >> 1e-12 -- the gate
    # resolves it (so the 1e-12 agreement is not a tautology of a flat integrand)
    assert abs(el / (_j_closed(PARS_U, E0 * (1.0 + 1e-9)) * area) - 1.0) > 1e-9
    if theta_deg in (30.0, 45.0, 60.0):
        assert abs(ed / truth - 1.0) > 0.5                        # the old code fails the element gate
    else:
        assert abs(ed / truth - 1.0) < 1e-12                      # mesh-aligned: the edge form is exact


def _gauss_legendre_2d(f, a, b, n):
    g, w = np.polynomial.legendre.leggauss(n)
    X = a + (b - a) * (g + 1.0) / 2.0
    W = w * (b - a) / 2.0
    XX, YY = np.meshgrid(X, X, indexing="ij")
    return float(np.sum(np.outer(W, W) * f(XX, YY)))


def test_h1_harmonic_field_element_converges_to_isotropic_integral_edge_to_componentwise():
    """H1. psi = PSI_EQ + V0 (x^2 - y^2)/L^2 on [L, 2L]^2: E = (2 V0/L^2)(-x, y), |E| = 2 V0 |r|/L^2, at
    27-63 deg to the mesh. Reference: tensor Gauss-Legendre 400 x 400 of alpha(|E|) q mu N |E| (cross-
    checked against 200 x 200: < 1e-13). Bounds (error budget): observed order within 0.05 of 2 (the
    quadrature of a smooth integrand with a centred two-leg rule; MEASURED 2.000 / 2.000); the Richardson
    limit of h/2, h/4, h/8 carries the O(h^4) remainder, ~ e(h)^2 with e(L/10) = 1.6e-3 -> ~2.6e-6 at
    L/10 and 1.6e-7 at L/20 -> bound 1e-6 (MEASURED 6.1e-10 element, 7.0e-11 edge vs its own limit). The
    old (edge) code converges to 0.359 of the reference: it fails the element limit gate by 0.64."""
    from dynameta.carriers.impact_ionization import substrate_current
    v0 = 1.0
    k = 2.0 * v0 / L_SLAB ** 2
    e_c = k * math.hypot(1.5 * L_SLAB, 1.5 * L_SLAB)
    pars = ChynowethParams(a_n_per_m=1.0e6, b_n_V_per_m=4.0 * e_c, a_p_per_m=0.0, b_p_V_per_m=1.0)

    def f_iso(x, y):
        e = k * np.hypot(x, y)
        return pars.alpha_n(e) * Q * MU_N * ND * e

    def f_comp(x, y):
        return (pars.alpha_n(k * x) * Q * MU_N * ND * k * x) + (pars.alpha_n(k * y) * Q * MU_N * ND * k * y)
    i_ref = _gauss_legendre_2d(f_iso, L_SLAB, 2.0 * L_SLAB, 400)
    i_comp = _gauss_legendre_2d(f_comp, L_SLAB, 2.0 * L_SLAB, 400)
    assert abs(_gauss_legendre_2d(f_iso, L_SLAB, 2.0 * L_SLAB, 200) / i_ref - 1.0) < 1e-13
    res = {}
    for n in (10, 20, 40, 80):
        dev = "ii2d_h{}".format(n)
        _slab(dev, L_SLAB, L_SLAB, L_SLAB, L_SLAB, n, n)
        try:
            _set_state(dev, lambda x, y: PSI_EQ + v0 * (x * x - y * y) / L_SLAB ** 2)
            res[n] = (substrate_current(dev, "si", pars), substrate_current(dev, "si", pars, quadrature="edge"))
            if n == 20:                                           # refutation (i): x <-> y changes nothing
                _set_state(dev, lambda x, y: PSI_EQ + v0 * (y * y - x * x) / L_SLAB ** 2)
                rot = substrate_current(dev, "si", pars)
        finally:
            _drop(dev)
    assert abs(rot / res[20][0] - 1.0) < 1e-12

    def order_and_limit(kk, a, b, c):
        d1, d2 = res[a][kk] - res[b][kk], res[b][kk] - res[c][kk]
        p = math.log2(abs(d1 / d2))
        return p, res[c][kk] - d2 / (2.0 ** p - 1.0)
    for kk, ref in ((0, i_ref), (1, i_comp)):
        for a, b, c in ((10, 20, 40), (20, 40, 80)):
            p, _lim = order_and_limit(kk, a, b, c)
            assert abs(p - 2.0) < 0.05, (kk, a, p)
        _p, lim = order_and_limit(kk, 20, 40, 80)
        assert abs(lim / ref - 1.0) < 1e-6, (kk, lim, ref)
    # the old (edge) form converges, but to the componentwise integral: 0.359 of the truth
    assert abs(i_comp / i_ref - 1.0) > 0.5
    assert abs(order_and_limit(1, 20, 40, 80)[1] / i_ref - 1.0) > 0.5


def _diode_column(dev, x_lines, width_m=None, tau=1e-10):
    """Abrupt p+ (x < 100 nm) / n+ 1e24 Si diode between ohmic contacts 'bot' (x = 0) and 'top' (x end):
    1-D when width_m is None, else a 2-D rectilinear strip (2 cells across, right triangles, contacts the
    two end lines) on the SAME x-lines -- hand-built arrays, in memory."""
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.constants import M_E
    ds = _ds()
    _drop(dev)
    m = dev + "_m"
    xl = np.asarray(x_lines, dtype=float)
    with _quiet():
        if width_m is None:
            ds.create_1d_mesh(mesh=m)
            for k, xk in enumerate(xl):
                tag = {0: "bot", len(xl) - 1: "top"}.get(k)
                ds.add_1d_mesh_line(mesh=m, pos=float(xk), ps=1.0, **({"tag": tag} if tag else {}))
            ds.add_1d_contact(mesh=m, name="bot", tag="bot", material="metal")
            ds.add_1d_contact(mesh=m, name="top", tag="top", material="metal")
            ds.add_1d_region(mesh=m, material="Si", region="si", tag1="bot", tag2="top")
            ds.finalize_mesh(mesh=m)
        else:
            yl = np.linspace(0.0, width_m, 3)
            nx, ny = len(xl), len(yl)
            X, Y = np.meshgrid(xl, yl, indexing="ij")
            idx = np.arange(nx * ny).reshape(nx, ny)
            el = []
            for j in range(ny - 1):                                     # contact lines (gmsh type 1)
                el += [1, 0, int(idx[0, j]), int(idx[0, j + 1]), 1, 1, int(idx[-1, j]), int(idx[-1, j + 1])]
            for i in range(nx - 1):
                for j in range(ny - 1):
                    a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                    el += [2, 2, int(a), int(b), int(c), 2, 2, int(a), int(c), int(d)]
            ds.create_gmsh_mesh(mesh=m, coordinates=[float(v) for p in zip(X.ravel(), Y.ravel()) for v in (*p, 0.0)],
                                physical_names=["bot", "top", "si"], elements=el)
            ds.add_gmsh_region(mesh=m, gmsh_name="si", region="si", material="Si")
            for c in ("bot", "top"):
                ds.add_gmsh_contact(mesh=m, gmsh_name=c, region="si", name=c, material="metal")
            ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=dev)
        ds.node_model(device=dev, region="si", name="NetDoping", equation="1e24 * (2 * step(x - 100e-9) - 1)")
        nc = PE.effective_dos_m3(1.08 * M_E)
        BP.setup_bipolar_region(dev, "si", eps_static=11.7, n_dos_m3=nc, n_i_m3=NI, mobility_n_m2Vs=MU_N,
                                mobility_p_m2Vs=MU_P, tau_n_s=tau, tau_p_s=tau, fd_enhancement=True,
                                n_dos_p_m3=nc)
        BP.setup_equilibrium_seed_models(dev, "si")
        for c in ("bot", "top"):
            BP.setup_contact_ohmic_bipolar(dev, c)
        ds.node_model(device=dev, region="si", name="_seed",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "si")))
    for var, mod in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region="si", name=var,
                           values=list(ds.get_node_model_values(device=dev, region="si", name=mod)))


def test_u2_laterally_uniform_2d_diode_equals_1d():
    """U2 ("does not break 1-D-like cases"; passes on the old (edge) code too, by design). The reverse-
    biased (3 V) abrupt diode on a 2-D rectilinear strip and on the 1-D mesh with the SAME x-lines are the
    same discrete system: MEASURED psi, n, p agree to 1.4e-14 and the contact current to 0.
      (a) GATE, 1e-12: for this mesh-aligned current the 2-D element quadrature equals the 2-D edge
          quadrature (each right triangle's vector is (E_x-edge, E_y-edge) with E_y ~ 2e-8 V/m, so the
          magnitudes agree to ~(E_y/E_x)^2); MEASURED 0.
      (b) CONSISTENCY CHECK (not an independent oracle), 1e-6: 2-D element / width == 1-D edge. Its floor
          is not quadrature but the SG current's cancellation: per-edge |J| is a residue of drift and
          diffusion terms, and the two solutions' |J| differ by up to 4.8e-4 (MEASURED) on near-zero-J
          edges from their 1e-14 state difference; weighted by alpha, I_sub differs by 1.7e-8 (MEASURED,
          both quadratures equally). That floor is solver noise this test cannot derive a priori, so the
          1e-6 bound (60x the measured floor) makes this a consistency check, not a gate."""
    from dynameta.carriers import devsim_3d as D3
    from dynameta.carriers.impact_ionization import substrate_current
    ds = _ds()
    x = D3._graded_axis([0.0, 100.0, 200.0], [100.0], 1.0, 8.0, dist_min=30.0, dist_max=80.0) * 1e-9
    width = 20e-9
    res = {}
    for name, w in (("1d", None), ("2d", width)):
        dev = "ii2d_col" + name
        _diode_column(dev, x, w)
        try:
            with _quiet():
                for k in range(31):
                    ds.set_parameter(device=dev, name="top_bias", value=3.0 * k / 30)
                    ds.solve(type="dc", absolute_error=1e10, relative_error=1e-11, maximum_iterations=60)
            res[name] = substrate_current(dev, "si", SILICON_VANOVERSTRAETEN)
            if w is not None:
                res["2d_edge"] = substrate_current(dev, "si", SILICON_VANOVERSTRAETEN, quadrature="edge")
            res[name + "_I"] = ds.get_contact_current(device=dev, contact="top",
                                                      equation="ElectronContinuityEquation")
        finally:
            _drop(dev)
    assert abs(res["2d_I"] / width / res["1d_I"] - 1.0) < 1e-12
    assert res["1d"] > 0.0
    assert abs(res["2d"] / res["2d_edge"] - 1.0) < 1e-12, (res["2d"], res["2d_edge"])
    assert abs(res["2d"] / width / res["1d"] - 1.0) < 1e-6, (res["2d"] / width, res["1d"])


def test_r1_quadrature_value_errors():
    from dynameta.carriers.impact_ionization import substrate_current, impact_generation_edges
    ds = _ds()
    dev = "ii2d_r1"
    _drop(dev)
    m = dev + "_m"
    with _quiet():
        ds.create_1d_mesh(mesh=m)
        ds.add_1d_mesh_line(mesh=m, pos=0.0, ps=1e-9, tag="a")
        ds.add_1d_mesh_line(mesh=m, pos=10e-9, ps=1e-9, tag="b")
        ds.add_1d_contact(mesh=m, name="a", tag="a", material="metal")
        ds.add_1d_contact(mesh=m, name="b", tag="b", material="metal")
        ds.add_1d_region(mesh=m, material="Si", region="si", tag1="a", tag2="b")
        ds.finalize_mesh(mesh=m)
        ds.create_device(mesh=m, device=dev)
        _setup_region(ds, dev)
    try:
        with pytest.raises(ValueError, match="1-D"):
            substrate_current(dev, "si", SILICON_VANOVERSTRAETEN, quadrature="element")
        with pytest.raises(ValueError, match="quadrature must be one of"):
            impact_generation_edges(dev, "si", SILICON_VANOVERSTRAETEN, quadrature="isotropic")
        # 1-D "auto" == "edge" (exact in 1-D)
        assert substrate_current(dev, "si", SILICON_VANOVERSTRAETEN) == substrate_current(
            dev, "si", SILICON_VANOVERSTRAETEN, quadrature="edge")
    finally:
        _drop(dev)
    from dynameta.carriers import devsim_3d as D3
    dev = "ii2d_r1t"
    _drop(dev)
    with _quiet():
        D3.kuhn_bar_mesh(dev + "_m", (20.0, 10.0, 10.0), 5.0, region="si", contacts=("a", "b"))
        ds.add_gmsh_region(mesh=dev + "_m", gmsh_name="si", region="si", material="Si")
        ds.finalize_mesh(mesh=dev + "_m")
        ds.create_device(mesh=dev + "_m", device=dev)
        _setup_region(ds, dev)
    try:
        with pytest.raises(ValueError, match="3-D"):
            substrate_current(dev, "si", SILICON_VANOVERSTRAETEN, quadrature="edge")
    finally:
        _drop(dev)


def test_g1_element_models_cleaned_and_preexisting_ones_only_read():
    from dynameta.carriers.impact_ionization import substrate_current
    ds = _ds()
    dev = "ii2d_g1"
    _slab(dev, 0.0, L_SLAB, 0.0, 0.5 * L_SLAB, 10, 5)
    try:
        _set_state(dev, lambda x, y: PSI_EQ - E0 * (0.6 * x + 0.8 * y))
        before = sorted(ds.get_element_model_list(device=dev, region="si"))
        i1 = substrate_current(dev, "si", PARS_U)
        assert sorted(ds.get_element_model_list(device=dev, region="si")) == before
        ds.element_from_edge_model(device=dev, region="si", edge_model="ElectricField")   # as 2-D BTBT
        pre = {c: list(ds.get_element_model_values(device=dev, region="si", name=c))
               for c in ("ElectricField_x", "ElectricField_y")}
        listed = sorted(ds.get_element_model_list(device=dev, region="si"))
        i2 = substrate_current(dev, "si", PARS_U)
        assert sorted(ds.get_element_model_list(device=dev, region="si")) == listed
        for c, v in pre.items():
            assert list(ds.get_element_model_values(device=dev, region="si", name=c)) == v
    finally:
        _drop(dev)
    assert i2 == i1


def test_x1_obtuse_triangle_mesh_is_refused():
    from dynameta.carriers.element_geometry import InexactElementGeometryError
    from dynameta.carriers.impact_ionization import substrate_current
    ds = _ds()
    dev = "ii2d_x1"
    _drop(dev)
    P = [[0.0, 0.0], [2e-8, 0.0], [1e-8, 3e-9], [1e-8, -4e-8]]   # Delaunay pair, one 147-deg triangle
    with _quiet():
        ds.create_gmsh_mesh(mesh=dev + "_m", coordinates=[float(c) for p in P for c in (p[0], p[1], 0.0)],
                            physical_names=["si"], elements=[2, 0, 0, 1, 2, 2, 0, 0, 3, 1])
        ds.add_gmsh_region(mesh=dev + "_m", gmsh_name="si", region="si", material="Si")
        ds.finalize_mesh(mesh=dev + "_m")
        ds.create_device(mesh=dev + "_m", device=dev)
        _setup_region(ds, dev)
    try:
        _set_state(dev, lambda x, y: PSI_EQ - E0 * x)
        with pytest.raises(InexactElementGeometryError, match="impact_generation_edges") as ei:
            substrate_current(dev, "si", PARS_U)
        assert ei.value.ratios["max_element_ratio"] == pytest.approx(100.0 / 9.0, rel=1e-12)  # 11.1x
        assert isinstance(ei.value, NotImplementedError)
    finally:
        _drop(dev)
