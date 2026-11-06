"""D1d (v0.14.0): DEVSIM tunneling -- band-to-band tunneling (BTBT) generation in the bipolar
drift-diffusion Newton, and the post-processed gate-oxide tunneling current (carriers/tunneling.py).

Every reference here is independent of the code under test: the Kane/Hurkx rate, the depletion-
approximation field profile, the FN coefficients and the Gauss-law oxide field are re-coded in this file
from their textbook forms; DEVSIM supplies only the solved potential/carrier arrays.

Gates (each FAILS on the pre-D1d code -- there is no BTBT term, no setup_bipolar_region(btbt=...), no
oxide post-processor; the physical contrast is gated too: Z4 shows the no-BTBT reverse current is ~1e-9 of
the Zener current):
  Z0  equilibrium: with the Hurkx D factor U_BTBT vanishes at V = 0 (|U|/Kane < 1e-12), although the
      Kane prefactor at the built-in field is ~1e26 m^-3 s^-1.
  Z1  Zener I-V vs the integrated analytic estimate over the depletion profile (|dev| <= 1 % + delta_D).
  Z2  the SHAPE: d ln J / dV agrees with the analytic estimate to 1.5 %.
  Z3  terminal current == q Int G dV (continuity) within the SI roundoff floor of the contact current.
  Z4  old physics (no BTBT) is ~1e-9 below: Z1 fails on the old code by 9 orders of magnitude.
  Z5  the installed rate == a numpy evaluation of A E^P exp(-B/E) D(n, p) on the solved arrays (1e-12).
  J1  the assembled BTBT Jacobian == central finite differences of DEVSIM's own residual (1-D, 2-D, 3-D).
  I1  ISOTROPY: |E| exact (< 1e-14) for an oblique uniform field on 2-D triangles and 3-D tets, and the
      element-volume quadrature integrates a uniform field exactly.
  L1  the 2-D layered builder (TransportModel.btbt, element path) CONVERGES to the 1-D edge path on the
      same x nodes at second order in h (order >= 1.7, |gap| <= 2 C_inf h^2).
  F1  MOS cap, accumulation: post-processed E_ox == Gauss-law field from the semiconductor charge.
  F2  ln(J/E^2) vs 1/E is a straight line of slope -B_FN; J == closed form at the Gauss field.
  F3  FN below the barrier warns; tsu_esaki/schuegraf_hu dispatch == the kernels.
  W1  layered builder oxide_tunneling extras == the direct post-processor; default adds no key.
  D1  defaults OFF: no BTBT models / equation terms unless asked; params validation; registry refusals.
"""
import contextlib
import math
import os
import sys
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

from dynameta.carriers import eq_registry as _R                              # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP                       # noqa: E402
from dynameta.carriers import physics_equilibrium as PE                      # noqa: E402
from dynameta.carriers.tunneling import (BTBTParams, OxideTunnelingParams,   # noqa: E402
                                         SILICON_HURKX, btbt_generation_current,
                                         oxide_tunneling_current)

Q = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
VT = KB * 300.0 / Q
# Hurkx et al., IEEE TED 39, 331 (1992), Si: A = 4e14 cm^-1/2 V^-5/2 s^-1, B = 1.9e7 V/cm, P = 2.5,
# converted HERE by hand: A_SI = 4e14 * 1e6 * 100^-2.5 = 4e15, B_SI = 1.9e9 V/m
A_SI, B_SI, P_SI = 4.0e15, 1.9e9, 2.5


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
        os.close(saved)


def _wipe(device):
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    _R.clear(device)


def _solve(abs_err=1e12, rel_err=1e-10):
    with _quiet():
        ds.solve(type="dc", absolute_error=abs_err, relative_error=rel_err, maximum_iterations=100)


# ------------------------------------------------------------------------------------------------
# 1-D abrupt p+n+ Zener diode
# ------------------------------------------------------------------------------------------------
ZN = 5.0e24            # Na = Nd [m^-3] (5e18 cm^-3)
ZNI = 1.0e16
ZMU_N, ZMU_P = 0.135, 0.048
ZL, ZXJ = 200e-9, 100e-9


def _build_zener(device, btbt):
    _wipe(device)
    mesh = device + "_m"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="pc")
    ds.add_1d_mesh_line(mesh=mesh, pos=ZXJ, ps=0.2e-9)
    ds.add_1d_mesh_line(mesh=mesh, pos=ZL, ps=2e-9, tag="nc")
    ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
    ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=device)
    ds.node_model(device=device, region="b", name="NetDoping",
                  equation="ifelse(x < {}, {}, {})".format(ZXJ, -ZN, ZN))
    BP.setup_bipolar_region(device, "b", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=ZNI,
                            mobility_n_m2Vs=ZMU_N, mobility_p_m2Vs=ZMU_P, tau_n_s=1e-3, tau_p_s=1e-3,
                            fd_enhancement=False, btbt=btbt)
    BP.setup_equilibrium_seed_models(device, "b")
    for c in ("pc", "nc"):
        BP.setup_contact_ohmic_bipolar(device, c)
    ds.node_model(device=device, region="b", name="_seed", equation=BP.equilibrium_seed_psi_expr(None))
    for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=device, region="b", name=v,
                           values=ds.get_node_model_values(device=device, region="b", name=m))
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(device, ceq)
    _solve(1e10, 1e-8)                                      # Poisson-only pre-solve
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(device, ceq)
    _solve()


def _terminal(device):
    return -sum(ds.get_contact_current(device=device, contact="pc", equation=e)
                for e in ("ElectronContinuityEquation", "HoleContinuityEquation"))


def _zener_analytic(Vr):
    """q Int A E^P exp(-B/E) (-D) dx over the depletion-approximation profile of the abrupt symmetric
    junction: E(x) = Emax - (qN/eps)|x|, with the exact first-integral Emax^2 = (qN/eps)(Vbi + Vr - 2Vt)
    (the majority-carrier Debye tails), flat quasi-Fermi levels split by Vr, and the Hurkx D factor from
    n = ni exp((psi - phi_n)/Vt), p = ni exp((phi_p - psi)/Vt)."""
    from scipy.integrate import quad
    eps = 11.7 * EPS0
    vbi = 2.0 * VT * math.log(ZN / ZNI)
    emax = math.sqrt(Q * ZN / eps * (vbi + Vr - 2.0 * VT))
    a = Q * ZN / eps
    xn = emax / a

    def g(x):
        E = emax - a * x
        dpsi = emax * x - 0.5 * a * x * x
        n = ZNI * math.exp((dpsi - 0.5 * Vr) / VT)
        p = ZNI * math.exp((-0.5 * Vr - dpsi) / VT)
        D = (n * p - ZNI ** 2) / ((n + ZNI) * (p + ZNI))
        return A_SI * E ** P_SI * math.exp(-B_SI / E) * (-D)
    return 2.0 * Q * quad(g, 0.0, xn, limit=500, epsrel=1e-11)[0], emax


@pytest.fixture(scope="module")
def zener_sweep():
    dev = "tz_zener"
    _build_zener(dev, SILICON_HURKX)
    out = {}
    # Z0 data at equilibrium
    psi = np.asarray(ds.get_node_model_values(device=dev, region="b", name="Potential"))
    x = np.asarray(ds.get_node_model_values(device=dev, region="b", name="x"))
    E = np.abs(np.diff(psi) / np.diff(x))
    with np.errstate(under="ignore"):
        kane = A_SI * E ** P_SI * np.exp(-B_SI / np.maximum(E, 1.0))
    out["eq"] = {"U": np.asarray(ds.get_edge_model_values(device=dev, region="b", name="BTBT_U")),
                 "kane": kane, "I": btbt_generation_current(dev, "b"), "J": _terminal(dev),
                 "kane_int": Q * float(np.sum(kane * np.diff(x)))}
    rows = []
    for vt in np.round(np.arange(-0.1, -1.21, -0.1), 3):
        ds.set_parameter(device=dev, name="pc_bias", value=float(vt))
        _solve()
        if -vt < 0.29:
            continue
        x = np.asarray(ds.get_node_model_values(device=dev, region="b", name="x"))
        n = np.asarray(ds.get_node_model_values(device=dev, region="b", name="Electrons"))
        p = np.asarray(ds.get_node_model_values(device=dev, region="b", name="Holes"))
        psi = np.asarray(ds.get_node_model_values(device=dev, region="b", name="Potential"))
        rows.append({"Vr": -vt, "I": btbt_generation_current(dev, "b"), "J": _terminal(dev),
                     "x": x, "n": n, "p": p, "psi": psi})
    out["rows"] = rows
    yield out
    _wipe(dev)


def test_z0_btbt_vanishes_at_equilibrium(zener_sweep):
    eq = zener_sweep["eq"]
    assert float(np.max(eq["kane"])) > 1e25                      # the Kane prefactor is NOT small here
    assert float(np.max(np.abs(eq["U"]))) < 1e-12 * float(np.max(eq["kane"]))   # MEASURED 1.8e-15
    assert abs(eq["I"]) < 1e-12 * eq["kane_int"]


def test_z1_z2_zener_iv_matches_integrated_analytic_estimate(zener_sweep):
    """delta_D: the analytic estimate assumes flat quasi-Fermi levels, i.e. n, p << n_i in the tunneling
    zone. The GENERATED carriers violate that: they drift out at v = mu E, so n ~ J_n/(q mu_n E) and
    p ~ J_p/(q mu_p E), and the Hurkx D factor loses (n + p)/n_i <= J/(q mu_p Emax n_i) = delta_D --
    a physical self-limitation of the local model, not an error (at -3 V, J ~ 1e4 A/m^2, it costs 40 %).
    The extra 1 % is the 1-D edge-mean-D quadrature (MEASURED 0.2-0.6 % between 0.2 and 0.1 nm meshes)."""
    rows = zener_sweep["rows"]
    Vr = np.array([r["Vr"] for r in rows])
    I = np.array([r["I"] for r in rows])
    ana = [_zener_analytic(v) for v in Vr]
    Ja = np.array([a[0] for a in ana])
    delta = Ja / (Q * ZMU_P * np.array([a[1] for a in ana]) * ZNI)
    assert I[-1] / I[0] > 100.0                                   # > 2 decades of Zener current
    dev = I / Ja - 1.0
    assert np.all(np.abs(dev) <= 0.01 + delta), (dev, delta)       # MEASURED |dev| <= 7.9e-3
    s_dev = np.diff(np.log(I)) / np.diff(Vr)
    s_an = np.diff(np.log(Ja)) / np.diff(Vr)
    assert np.all(np.abs(s_dev / s_an - 1.0) <= 0.015)             # MEASURED <= 0.59 %


def test_z3_terminal_current_is_the_generation_integral(zener_sweep):
    """Steady-state continuity: the terminal current IS q Int G dV (SRH with tau = 1 ms and the minority
    diffusion current are < 1e-7 A/m^2). The SI contact current is a difference of majority Scharfetter-
    Gummel terms of q mu_n Vt N / h = 1.4e12 A/m^2 at these 5e24 m^-3 contacts (h = 2 nm), i.e. one ulp is
    3e-4 A/m^2: MEASURED |J - I| = 0.3-3e-3 A/m^2 (~10 ulps, sign-random). The floor allowed is 32 ulps."""
    floor = 32.0 * np.finfo(float).eps * Q * ZMU_N * VT * ZN / 2e-9
    assert 5e-3 < floor < 2e-2
    for r in zener_sweep["rows"]:
        assert abs(r["J"] - r["I"]) <= floor + 1e-4 * abs(r["I"])


def test_z4_no_btbt_is_nine_orders_below():
    dev = "tz_zener_off"
    try:
        _build_zener(dev, None)
        for vt in np.arange(-0.1, -1.01, -0.1):
            ds.set_parameter(device=dev, name="pc_bias", value=float(vt))
            _solve()
        j_off = _terminal(dev)
        assert "BTBT_U" not in ds.get_edge_model_list(device=dev, region="b")
    finally:
        _wipe(dev)
    j_an, _ = _zener_analytic(1.0)
    assert abs(j_off) < 1e-6 * j_an                                # MEASURED 4.7e-8 vs 38.9 A/m^2


def test_z5_installed_rate_equals_numpy_formula_on_solved_arrays(zener_sweep):
    for r in zener_sweep["rows"]:
        x, n, p, psi = r["x"], r["n"], r["p"], r["psi"]
        E = np.sqrt((np.diff(psi) / np.diff(x)) ** 2 + 1.0)
        D = (n * p - ZNI ** 2) / ((n + ZNI) * (p + ZNI))
        Dm = 0.5 * (D[1:] + D[:-1])
        with np.errstate(under="ignore"):
            G = -A_SI * E ** P_SI * np.exp(-B_SI / E) * Dm
        assert r["I"] == pytest.approx(Q * float(np.sum(G * np.diff(x))), rel=1e-12)


# ------------------------------------------------------------------------------------------------
# Jacobian + isotropy on hand-built meshes (no contacts; models + one assembly only)
# ------------------------------------------------------------------------------------------------
def _mesh_device(device, dim, n=4, L=8e-9, btbt=SILICON_HURKX):
    _wipe(device)
    mesh = device + "_m"
    g = np.linspace(0.0, L, n + 1)
    if dim == 1:
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=L / (4 * n), tag="a")
        ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=L / (4 * n), tag="b")
        ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")   # DEVSIM: 1-D needs both
        ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="r", tag1="a", tag2="b")
    else:
        if dim == 2:
            X, Y = np.meshgrid(g, g, indexing="ij")
            coords = np.column_stack([X.ravel(), Y.ravel(), 0.0 * X.ravel()])
            idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
            el = []
            for i in range(n):
                for j in range(n):
                    a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                    el += [2, 0, int(a), int(b), int(c), 2, 0, int(a), int(c), int(d)]
        else:
            X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
            coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
            el = []
            perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for pm in perms:                        # Kuhn split: 6 tets per cube
                            v = [0, 0, 0]
                            path = [idx[i, j, k]]
                            for ax in pm:
                                v[ax] += 1
                                path.append(idx[i + v[0], j + v[1], k + v[2]])
                            el += [3, 0] + [int(t) for t in path]
        ds.create_gmsh_mesh(mesh=mesh, coordinates=[float(c) for c in coords.ravel()],
                            physical_names=["r"], elements=el)
        ds.add_gmsh_region(mesh=mesh, gmsh_name="r", region="r", material="Si")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=device)
    ds.node_model(device=device, region="r", name="NetDoping", equation="1e22")
    with _quiet():
        BP.setup_bipolar_region(device, "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=ZNI,
                                mobility_n_m2Vs=ZMU_N, mobility_p_m2Vs=ZMU_P, tau_n_s=1e-7,
                                tau_p_s=1e-7, fd_enhancement=False, btbt=btbt)
    return device


def _set_state(device, dim, E0=1.5e8, seed=1):
    def nm(c):
        return np.asarray(ds.get_node_model_values(device=device, region="r", name=c))
    x, y, z = nm("x"), nm("y"), nm("z")
    u = {1: np.array([1.0, 0, 0]), 2: np.array([math.cos(0.5236), math.sin(0.5236), 0.0]),
         3: np.array([0.48, 0.6, 0.64]) / np.linalg.norm([0.48, 0.6, 0.64])}[dim]
    rng = np.random.default_rng(seed)
    st = {"Potential": -E0 * (u[0] * x + u[1] * y + u[2] * z),
          "Electrons": 10 ** rng.uniform(13, 17, x.size), "Holes": 10 ** rng.uniform(13, 17, x.size)}
    for v, val in st.items():
        ds.set_node_values(device=device, region="r", name=v, values=[float(a) for a in val])
    return st, rng


def _fd_residual_error(device, dim):
    """|| d rhs - J d || / || d rhs || on the continuity rows, central differences (DEVSIM's rhs is the
    NEGATED residual, so d rhs = -J d)."""
    import scipy.sparse as sp
    st, rng = _set_state(device, dim)
    try:                                                     # one assembly to number the equations
        with _quiet():
            ds.solve(type="dc", absolute_error=1e30, relative_error=1e30, maximum_iterations=1)
    except ds.error:
        pass

    def assemble():
        m = ds.get_matrix_and_rhs(format="csc")["static"]
        return (sp.csc_matrix((np.asarray(m["av"]), np.asarray(m["ai"]), np.asarray(m["ap"]))),
                np.asarray(m["rhs"]))

    def put(sign, dl):
        for v in st:
            ds.set_node_values(device=device, region="r", name=v,
                               values=[float(a) for a in st[v] + sign * dl[v]])

    put(0.0, {v: 0.0 for v in st})
    J0, _ = assemble()
    nums = {v: np.asarray(ds.get_equation_numbers(device=device, region="r", variable=v)) for v in st}
    dl = {"Potential": 1e-7 * rng.standard_normal(st["Potential"].size),
          "Electrons": 1e-6 * st["Electrons"] * rng.standard_normal(st["Electrons"].size),
          "Holes": 1e-6 * st["Holes"] * rng.standard_normal(st["Holes"].size)}
    vec = np.zeros(J0.shape[0])
    for v in st:
        vec[nums[v]] = dl[v]
    put(+1.0, dl)
    rp = assemble()[1]
    put(-1.0, dl)
    rm = assemble()[1]
    rows = np.concatenate([nums["Electrons"], nums["Holes"]])
    jd = (J0 @ vec)[rows]
    dr = (0.5 * (rp - rm))[rows]
    return jd, dr


@pytest.mark.floating_ok("assembly only: the contact-free device is never solved -- its one-iteration "
                         "ds.solve just numbers the equations for get_matrix_and_rhs (v0.15 4.4.2 hook)")
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_j1_btbt_jacobian_matches_finite_differences(dim):
    """The BTBT share of the Jacobian, isolated as (with BTBT) - (without), on the same state."""
    dev = "tz_jac{}".format(dim)
    try:
        _mesh_device(dev, dim, btbt=SILICON_HURKX)
        jd_on, dr_on = _fd_residual_error(dev, dim)
        _mesh_device(dev, dim, btbt=None)
        jd_off, dr_off = _fd_residual_error(dev, dim)
    finally:
        _wipe(dev)
    jb, rb = jd_on - jd_off, dr_on - dr_off
    assert np.linalg.norm(jb) > 1e-3 * np.linalg.norm(jd_on)       # BTBT is a visible share (~1 %)
    assert np.linalg.norm(rb + jb) <= 1e-6 * np.linalg.norm(rb)    # MEASURED 6e-9 (2-D), 1e-8 (3-D)
    assert np.linalg.norm(dr_on + jd_on) <= 1e-6 * np.linalg.norm(dr_on)


# an integral A AND an integral B above 2^31 - 1: DEVSIM 2.11.2 clamps INTEGER literals to 2147483647, and the
# pre-integration "%.17g" printed integral floats as integers (integration note 2.6). SILICON_HURKX is
# non-integral (A_SI = 4000000000000000.5) and B < 2^31, so it never exercised the clamp; this set does.
_INTEGRAL_BIG = BTBTParams(A_SI=4.0e15, B_V_per_m=3.0e9, P=2.5)


@pytest.mark.parametrize("prm", [SILICON_HURKX, _INTEGRAL_BIG], ids=["hurkx", "integral_A_B_over_2^31"])
@pytest.mark.parametrize("dim", [2, 3])
def test_i1_element_field_is_isotropic_and_integrates_exactly(dim, prm):
    dev = "tz_iso{}".format(dim)
    try:
        _mesh_device(dev, dim, btbt=prm)
        st, _ = _set_state(dev, dim, E0=1.5e8)

        def em(name):
            return np.asarray(ds.get_element_model_values(device=dev, region="r", name=name))
        assert np.max(np.abs(em("BTBT_Emag") / 1.5e8 - 1.0)) < 1e-14   # MEASURED 8.9e-16 at 30 deg / 3-D oblique
        n, p = st["Electrons"], st["Holes"]
        D = (n * p - ZNI ** 2) / ((n + ZNI) * (p + ZNI))
        nv = np.asarray(ds.get_node_model_values(device=dev, region="r", name="NodeVolume"))
        # the coefficients are taken from prm only as the INTENDED values; a clamped A (2147483647 instead
        # of 4e15) or B (2147483647 instead of 3e9) misses this by a factor 1.9e6 or e^(-5.7)
        ref = prm.A_SI * 1.5e8 ** prm.P * math.exp(-prm.B_V_per_m / 1.5e8) * float(np.sum(D * nv))
        got = float(np.sum((em("BTBT_U0") + em("BTBT_U1")) * em("ElementNodeVolume")))
        assert got == pytest.approx(ref, rel=1e-12)                   # MEASURED 5e-15
        assert btbt_generation_current(dev, "r") == pytest.approx(-Q * ref, rel=1e-12)
    finally:
        _wipe(dev)


# ------------------------------------------------------------------------------------------------
# 2-D layered builder (element path) vs 1-D (edge path) on the same x nodes: a CONVERGENCE gate
# ------------------------------------------------------------------------------------------------
# h -> |2-D/1-D - 1| of the BTBT current at -0.5 V, A = 4e15, B = 4e8 V/m. MEASURED 2026-11-03 on a UNIFORM
# x mesh (gmsh-array triangles vs 1-D, identical nodes): 2 nm 3.247e-4, 1 nm 6.796e-5, 0.5 nm 1.611e-5,
# 0.25 nm 3.972e-6 -> orders 2.26, 2.08, 2.02; gap/h^2 = 8.1e-5, 6.8e-5, 6.4e-5, 6.35e-5 nm^-2.
_L1_C_INF_NM2 = 6.35e-5        # asymptotic gap / h^2 [nm^-2] (finest uniform pair)
_L1_K_NM2 = 2.0 * _L1_C_INF_NM2


def _l1_currents(h):
    """BTBT current [A/m^2] at V = 0 and -0.5 V through the 2-D LayeredDevsimBuilder (x spacing h) and
    through a 1-D twin built on the builder's exact x nodes."""
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.constants import M_E
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel
    from dynameta.sweep import BiasPoint
    P, T, N, W = 120e-9, 4e-9, 1e24, 5e-9
    # the builder's coupled Newton converges for the validation diode (1e24, 5 nm tanh grade), where the
    # Si field scale B = 1.9e9 V/m gives negligible BTBT; the test-only B = 4e8 V/m makes it ~2.3e3 A/m^2
    prm = BTBTParams(A_SI=A_SI, B_V_per_m=4e8, P=2.5)
    expr = "{:.8e}*tanh((x - {:.8e})/{:.8e})".format(N, P / 2, W)

    def c(v):
        return lambda nn: np.full_like(np.asarray(nn, dtype=float), v)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50 + 5j), is_metal=True))
    reg.add(Material("Si", ConstantOptical(12 + 0j), transport=TransportModel(
        n_bg_m3=N, eps_static=11.7, dos_mass_kg_of_n_m3=c(1.08 * M_E), band_gap_eV=1.12, chi_eV=4.05,
        physics="bipolar_dd", mobility_m2Vs_of_n_m3=c(ZMU_N), hole_mobility_m2Vs_of_n_m3=c(ZMU_P),
        tau_srh_s=1e-3, n_i_m3=ZNI, net_doping_expr=expr, dos_mass_p_kg=0.81 * M_E, btbt=prm)))
    d = Design(name="tz_l1", unit_cell=UnitCell.square(P),
               stack=Stack(layers=[Layer("si", T, "Si")], superstrate_material="air",
                           substrate_material="air"),
               electrodes=[Electrode("anode", "si", "x_lo", role="biased"),
                           Electrode("cathode", "si", "x_hi", role="ground", fixed_voltage_V=0.0)],
               materials=reg, mesh_2d=Mesh2DSpec(x_spacing_feature_mid_m=h, coarse_ps_m=2e-9))
    b = LayeredDevsimBuilder(d, mesh_name="tz_l1m", device_name="tz_l1d")
    i2, i1 = {}, {}
    try:
        for V in (0.0, -0.5):
            with _quiet():
                b.solve(BiasPoint({"anode": V}, "v"))
            i2[V] = btbt_generation_current(b.device, "si") / T     # per depth / thickness -> A/m^2
            if V == 0.0:                                          # the Kane prefactor integral (no D)
                em_ = np.asarray(ds.get_element_model_values(device=b.device, region="si", name="BTBT_Emag"))
                env = np.asarray(ds.get_element_model_values(device=b.device, region="si",
                                                             name="ElementNodeVolume"))
                with np.errstate(under="ignore"):
                    i2["kane0"] = Q * float(np.sum(2.0 * env * prm.A_SI * em_ ** prm.P
                                                   * np.exp(-prm.B_V_per_m / em_))) / T
        assert "BTBT_U0" in ds.get_element_model_list(device=b.device, region="si")
        xs = np.unique(np.round(np.asarray(ds.get_node_model_values(device=b.device, region="si",
                                                                    name="x")), 15))
    finally:
        _wipe(b.device)
    dev = "tz_l1_1d"
    try:
        mesh = dev + "_m"
        ds.create_1d_mesh(mesh=mesh)
        for k, xx in enumerate(xs):
            ds.add_1d_mesh_line(mesh=mesh, pos=float(xx), ps=1.0,
                                **({"tag": "pc"} if k == 0 else {"tag": "nc"} if k == xs.size - 1 else {}))
        ds.add_1d_contact(mesh=mesh, name="pc", tag="pc", material="metal")
        ds.add_1d_contact(mesh=mesh, name="nc", tag="nc", material="metal")
        ds.add_1d_region(mesh=mesh, material="Si", region="b", tag1="pc", tag2="nc")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        ds.node_model(device=dev, region="b", name="NetDoping", equation=expr)
        Nc, Nv = PE.effective_dos_m3(1.08 * M_E), PE.effective_dos_m3(0.81 * M_E)
        BP.setup_bipolar_region(dev, "b", eps_static=11.7, n_dos_m3=Nc, n_i_m3=ZNI, mobility_n_m2Vs=ZMU_N,
                                mobility_p_m2Vs=ZMU_P, tau_n_s=1e-3, tau_p_s=1e-3, fd_enhancement=True,
                                n_dos_p_m3=Nv, btbt=prm)
        BP.setup_equilibrium_seed_models(dev, "b")
        for cc in ("pc", "nc"):
            BP.setup_contact_ohmic_bipolar(dev, cc)
        ds.node_model(device=dev, region="b", name="_seed",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "b")))
        for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
            ds.set_node_values(device=dev, region="b", name=v,
                               values=ds.get_node_model_values(device=dev, region="b", name=m))
        _solve(1e13, 1e-10)
        for V in (0.0, -0.5):
            v0 = ds.get_parameter(device=dev, name="pc_bias")
            for s_ in np.linspace(v0, V, 11)[1:]:
                ds.set_parameter(device=dev, name="pc_bias", value=float(s_))
                _solve(1e13, 1e-10)
            i1[V] = btbt_generation_current(dev, "b")
    finally:
        _wipe(dev)
    return i2, i1


def test_l1_layered_builder_btbt_converges_to_the_1d_twin():
    """The 2-D element path and the 1-D edge path are two CONSISTENT discretisations of the same
    continuum; their BTBT currents must converge to each other at second order in the mesh spacing h.

    WHY THEY DIFFER AT ALL (diagnosed 2026-11-03; integration note 2.9). On identical fields the two
    INTEGRATED quadratures are algebraically equal on this y-invariant tensor mesh -- each right-triangle
    cell's quadrant of node i lies wholly in that cell and carries the cell's field, so
    sum_i D_i (U_l L_l + U_r L_r)/2 == sum_e U_e (D_0 + D_1)/2 L_e (MEASURED 6e-16 on the solved fields),
    and the 2-D solution is y-invariant to 1e-15. What differs is the NODAL source: node i receives
    sum_j pre_ij (L_ij/2) D_i (2-D, node-local) versus sum_j pre_ij (L_ij/2) (D_i + D_j)/2 (1-D, edge-mean).
    The difference, delta_i = sum_j pre_ij L_ij (D_j - D_i)/4 = (h^2/4) d/dx(pre dD/dx) + O(h^4) on a
    uniform mesh, is a conservative redistribution (it sums to zero), so it changes the current only
    through the solution's response to it: O(h^2). With the true A the junction runs in the self-limited
    regime (generated n, p ~ 0.1-0.3 n_i, D ~ -0.5) where D varies by orders of magnitude per node, so the
    response is visible: the 1-D/2-D solutions differ by 0.2-0.8 % in n, p at the junction and 2x in the
    minority tails, while WITHOUT BTBT they agree to 7.7e-7. The former 1e-6 gate passed only because
    DEVSIM clamped the integral literal A = 4e15 to 2147483647 (a rate 1.9e6x lower: D ~ -1, no D
    variation to redistribute).

    GATE: (a) the observed order between h = 2 and 1 nm is >= 1.7 (MEASURED 2.21 on the builder; 2.26 ->
    2.02 on a clean uniform mesh); (b) |gap(h)| <= K h^2 with K = 2 C_inf, C_inf = 6.35e-5 nm^-2 the
    asymptotic constant of the clean uniform study (header above). The factor 2 covers the measured
    preasymptotic excess (C = 1.28 C_inf at h = 2 nm, uniform) and the builder's non-uniform junction
    spacing (its own constants: 7.0e-5 at 2 nm, 6.1e-5 nm^-2 at 1 nm). MEASURED gaps: 2.81e-4 (2 nm),
    6.09e-5 (1 nm); bounds 5.1e-4, 1.27e-4. (c) the current is a REAL one (> 1e3 A/m^2: a clamped A
    would give ~1e-3); (d) at EQUILIBRIUM the D factor suppresses the Kane prefactor integral to
    |I(0)| <= 1e-4 of it and the residual SHRINKS with h. It is not 0 to roundoff (as in Z0) because this
    fixture runs the Fermi-Dirac g-factor: the discrete FD equilibrium (edge-averaged g) puts n p off
    n_i^2 at O(h^2) in the junction -- MEASURED on the 1-D twin I(0)/Kane = 6.4e-5 (2 nm), 1.6e-5 (1 nm),
    order 1.97, vs 3.2e-16 with fd_enhancement=False; 2-D and 1-D carry the same residual (-6.01e-3 vs
    -6.01e-3 A/m^2 at 2 nm), so it is the FD statistics, not the BTBT assembly."""
    gaps, eq = {}, {}
    for h_nm in (2.0, 1.0):
        i2, i1 = _l1_currents(h_nm * 1e-9)
        assert i1[-0.5] > 1e3 and i2[-0.5] > 1e3                      # MEASURED 2.26e3 / 2.31e3 A/m^2
        eq[h_nm] = abs(i2[0.0]) / i2["kane0"]
        assert eq[h_nm] <= 1e-4                                       # MEASURED 4.7e-5 (2 nm), 1.6e-5 (1 nm)
        gaps[h_nm] = i2[-0.5] / i1[-0.5] - 1.0
        assert abs(gaps[h_nm]) <= _L1_K_NM2 * h_nm ** 2, (h_nm, gaps[h_nm])
    assert eq[1.0] < 0.5 * eq[2.0]                                    # the equilibrium residual converges
    order = math.log(abs(gaps[2.0] / gaps[1.0])) / math.log(2.0)
    assert order >= 1.7, (gaps, order)


# ------------------------------------------------------------------------------------------------
# gate-oxide tunneling: 1-D MOS capacitor in accumulation
# ------------------------------------------------------------------------------------------------
T_OX, EPS_OX, ND_MOS = 5e-9, 3.9, 1e24


@pytest.fixture(scope="module")
def mos_cap():
    dev = "tz_mos"
    _wipe(dev)
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="body")
    ds.add_1d_mesh_line(mesh=mesh, pos=60e-9, ps=0.05e-9, tag="if")
    ds.add_1d_mesh_line(mesh=mesh, pos=60e-9 + T_OX, ps=0.5e-9, tag="gate")
    ds.add_1d_contact(mesh=mesh, name="body", tag="body", material="metal")
    ds.add_1d_contact(mesh=mesh, name="gate", tag="gate", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="semi", tag1="body", tag2="if")
    ds.add_1d_region(mesh=mesh, material="SiO2", region="oxide", tag1="if", tag2="gate")
    ds.add_1d_interface(mesh=mesh, name="semi_ox", tag="if")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="semi", name="NetDoping", equation="{:.6e}".format(ND_MOS))
    BP.setup_bipolar_region(dev, "semi", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=ZNI, mobility_n_m2Vs=ZMU_N,
                            mobility_p_m2Vs=ZMU_P, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=False)
    BP.setup_equilibrium_seed_models(dev, "semi")
    PE.setup_dielectric_region(dev, "oxide", EPS_OX)
    PE.setup_interface(dev, "semi_ox")
    BP.setup_contact_ohmic_bipolar(dev, "body")
    PE.setup_contact(dev, "gate")
    ds.node_model(device=dev, region="semi", name="_seed", equation=BP.equilibrium_seed_psi_expr(None))
    for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"), ("Holes", "IntrinsicHoles")):
        ds.set_node_values(device=dev, region="semi", name=v,
                           values=ds.get_node_model_values(device=dev, region="semi", name=m))
    phi_bi = VT * math.log(ND_MOS / ZNI)
    nox = len(ds.get_node_model_values(device=dev, region="oxide", name="x"))
    ds.set_node_values(device=dev, region="oxide", name="Potential", values=[phi_bi] * nox)
    ds.set_parameter(device=dev, name="gate_bias", value=phi_bi)
    _solve()
    pts = []
    for vg in np.arange(0.25, 9.01, 0.25):
        ds.set_parameter(device=dev, name="gate_bias", value=float(phi_bi + vg))
        _solve()
        if vg < 4.0:
            if abs(vg - 1.0) < 1e-9:
                with pytest.warns(Warning) as rec:              # FN below the barrier must warn
                    low = oxide_tunneling_current(dev, "oxide", "semi", OxideTunnelingParams(model="fn"))
                pts.append(("low", low, rec))
            continue
        n = np.asarray(ds.get_node_model_values(device=dev, region="semi", name="Electrons"))
        p = np.asarray(ds.get_node_model_values(device=dev, region="semi", name="Holes"))
        nv = np.asarray(ds.get_node_model_values(device=dev, region="semi", name="NodeVolume"))
        e_gauss = Q * float(np.sum((n - p - ND_MOS) * nv)) / (EPS_OX * EPS0)   # discrete Gauss law
        r = oxide_tunneling_current(dev, "oxide", "semi", OxideTunnelingParams(model="fn"))
        rd = oxide_tunneling_current(dev, "oxide", "semi",
                                     OxideTunnelingParams(model="tsu_esaki", E_F_eV=0.1, m_supply_over_m0=0.5))
        rs = oxide_tunneling_current(dev, "oxide", "semi", OxideTunnelingParams(model="schuegraf_hu"))
        pts.append((vg, e_gauss, r, rd, rs))
    yield pts
    _wipe(dev)


def test_f1_oxide_field_matches_gauss_law(mos_cap):
    for vg, e_gauss, r, _, _ in [p for p in mos_cap if p[0] != "low"]:
        assert r["E_ox_V_m"].size == 1 and r["area"][0] == 1.0 and r["t_ox_m"] == pytest.approx(T_OX)
        # MEASURED 2.4e-8 .. 5.4e-8 (the residual is the body-contact flux: 60 nm = 15 Debye lengths)
        assert r["E_ox_V_m"][0] == pytest.approx(e_gauss, rel=1e-6)
        assert r["V_ox_V"][0] == pytest.approx(e_gauss * T_OX, rel=1e-6)


def test_f2_fn_line_slope_and_closed_form(mos_cap):
    phi, mox = 3.1 * Q, 0.42 * 9.1093837015e-31
    h = 6.62607015e-34
    B = 8 * math.pi * math.sqrt(2 * mox) * phi ** 1.5 / (3 * Q * h)
    A = Q ** 3 * 9.1093837015e-31 / (8 * math.pi * h * mox * phi)
    pts = [p for p in mos_cap if p[0] != "low"]
    E = np.array([p[1] for p in pts])
    J = np.array([p[2]["I"] for p in pts])
    assert E.min() > 7e8 and E.max() > 1.7e9 and np.all(J > 0)    # accumulation, FN regime (V_ox > phi_b)
    slope, icpt = np.polyfit(1.0 / E, np.log(J / E ** 2), 1)
    assert slope == pytest.approx(-B, rel=1e-6)                    # 1e-6: field error 5e-8 x b ~ 25
    assert np.exp(icpt) == pytest.approx(A, rel=1e-5)
    assert np.allclose(J, A * E ** 2 * np.exp(-B / E), rtol=5e-6)


def test_f3_fn_warns_below_barrier_and_model_dispatch(mos_cap):
    from dynameta.reliability.leakage import schuegraf_hu_current, tsu_esaki_current
    low = [p for p in mos_cap if p[0] == "low"][0]
    assert any("below the barrier" in str(w.message) for w in low[2])
    assert low[1]["V_ox_V"][0] < 3.1
    for vg, e_gauss, r, rd, rs in [p for p in mos_cap if p[0] != "low"][::8]:
        v = r["V_ox_V"][0]
        assert rd["I"] == pytest.approx(tsu_esaki_current(v, T_OX, phi_b_eV=3.1, E_F_eV=0.1,
                                                          m_supply_over_m0=0.5), rel=1e-12)
        assert rs["I"] == pytest.approx(schuegraf_hu_current(v, T_OX), rel=1e-12)
        assert rs["I"] == pytest.approx(r["I"], rel=1e-12)       # V_ox > phi_b: S-H == FN


def test_f4_sign_follows_the_gate_polarity():
    """Reverse the gate: the current changes sign (electrons now injected from the gate)."""
    dev = "tz_sign"
    _wipe(dev)
    try:
        mesh = dev + "_m"
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="g0")
        ds.add_1d_mesh_line(mesh=mesh, pos=5e-9, ps=1e-9, tag="g1")
        ds.add_1d_mesh_line(mesh=mesh, pos=9e-9, ps=1e-9, tag="g2")
        for t in ("g0", "g2"):
            ds.add_1d_contact(mesh=mesh, name=t, tag=t, material="metal")
        ds.add_1d_region(mesh=mesh, material="A", region="a", tag1="g0", tag2="g1")
        ds.add_1d_region(mesh=mesh, material="B", region="ox", tag1="g1", tag2="g2")
        ds.add_1d_interface(mesh=mesh, name="ab", tag="g1")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        PE.setup_dielectric_region(dev, "a", 3.9)                  # a dielectric 'semiconductor' stand-in
        PE.setup_dielectric_region(dev, "ox", 3.9)
        PE.setup_interface(dev, "ab")
        for c in ("g0", "g2"):
            PE.setup_contact(dev, c)
        out = {}
        for vg in (+8.0, -8.0):
            ds.set_parameter(device=dev, name="g2_bias", value=vg)
            _solve(1e-10, 1e-12)
            out[vg] = oxide_tunneling_current(dev, "ox", "a", OxideTunnelingParams(model="fn"))
    finally:
        _wipe(dev)
    # linear potential: E_ox = 8 V / 9 nm, V_ox = E_ox * 4 nm, independent of the mesh
    for vg in (+8.0, -8.0):
        assert out[vg]["E_ox_V_m"][0] == pytest.approx(8.0 / 9e-9, rel=1e-12)
        assert out[vg]["V_ox_V"][0] == pytest.approx(8.0 * 4.0 / 9.0, rel=1e-12)
    assert out[+8.0]["I"] > 0.0 and out[-8.0]["I"] == pytest.approx(-out[+8.0]["I"], rel=1e-12)


# ------------------------------------------------------------------------------------------------
# builder wiring + defaults OFF
# ------------------------------------------------------------------------------------------------
def _gated_design():
    from dynameta.constants import M_E
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel

    def c(v):
        return lambda nn: np.full_like(np.asarray(nn, dtype=float), v)
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50 + 5j), is_metal=True))
    reg.add(Material("SiO2", ConstantOptical(2.1 + 0j), eps_static_dc=3.9))
    reg.add(Material("Si", ConstantOptical(12 + 0j), transport=TransportModel(
        n_bg_m3=1e24, eps_static=11.7, dos_mass_kg_of_n_m3=c(1.08 * M_E), band_gap_eV=1.12, chi_eV=4.05,
        physics="bipolar_dd", mobility_m2Vs_of_n_m3=c(ZMU_N), hole_mobility_m2Vs_of_n_m3=c(ZMU_P),
        tau_srh_s=1e-7, n_i_m3=ZNI, dos_mass_p_kg=0.81 * M_E)))
    stack = Stack(layers=[Layer("si", 20e-9, "Si"), Layer("ox", 5e-9, "SiO2"), Layer("gm", 5e-9, "cmetal")],
                  superstrate_material="air", substrate_material="air")
    return Design(name="tz_gated", unit_cell=UnitCell.square(40e-9), stack=stack,
                  electrodes=[Electrode("body", "si", "x_lo", role="ground", fixed_voltage_V=0.0),
                              Electrode("gate", "gm", "full", role="biased")],
                  materials=reg,
                  mesh_2d=Mesh2DSpec(coarse_ps_m=1e-9, x_spacing_edge_m=5e-9, x_spacing_feature_mid_m=5e-9))


def test_w1_layered_oxide_tunneling_extras_and_default_off():
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    from dynameta.warning_types import UnreliableResultWarning
    prm = OxideTunnelingParams(model="fn")
    res = {}
    for tag, ot in (("off", None), ("on", prm)):
        b = LayeredDevsimBuilder(_gated_design(), mesh_name="tz_w1m" + tag, device_name="tz_w1d" + tag,
                                 oxide_tunneling=ot)
        try:
            with pytest.warns(UnreliableResultWarning, match="under-accumulates"), _quiet():
                cf = b.solve(BiasPoint({"gate": 5.0}, "g"))
            res[tag] = cf
            if ot is not None:
                direct = oxide_tunneling_current(b.device, "ox", "si", prm, depth_m=40e-9)
            psi_off = np.asarray(ds.get_node_model_values(device=b.device, region="si", name="Potential"))
            res[tag + "_psi"] = psi_off
        finally:
            _wipe(b.device)
    assert "oxide_tunneling" not in res["off"].extras                # default: no key at all
    ex = res["on"].extras["oxide_tunneling"]["ox"]
    assert ex["I_A"] == direct["I"] and ex["I_A"] > 0.0
    assert ex["semiconductor_region"] == "si" and ex["model"] == "fn" and ex["V_ox_max_V"] > 3.1
    assert np.array_equal(res["off_psi"], res["on_psi"])            # a post-processor: the solve is untouched


def test_d1_defaults_off_and_guards():
    from dynameta.constants import M_E
    from dynameta.materials import TransportModel
    dev = "tz_d1"
    try:
        _build_zener(dev, None)
        assert not [m for m in ds.get_edge_model_list(device=dev, region="b") if m.startswith("BTBT")]
        for eq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            kw = _R.recorded(dev, eq, scope="region", loc="b")[0]["kwargs"]
            assert set(kw) == {"name", "variable_name", "edge_model", "node_model", "time_node_model",
                               "variable_update"}                    # the legacy equation, term for term
        # amend: a DEVSIM rejection leaves the live equation AND the record as they were ...
        with pytest.raises(ds.error):
            _R.amend_region_equation(dev, "b", "ElectronContinuityEquation", edge_volume_model="NoSuch")
        assert "edge_volume_model" not in _R.recorded(dev, "ElectronContinuityEquation")[0]["kwargs"]
        _solve()                                                     # the equation is still live
        # ... and it refuses to silently replace a term another module wired
        ds.edge_model(device=dev, region="b", name="X1", equation="0")
        _R.amend_region_equation(dev, "b", "ElectronContinuityEquation", edge_volume_model="X1")
        with pytest.raises(ValueError, match="refusing to overwrite"):
            _R.amend_region_equation(dev, "b", "ElectronContinuityEquation", edge_volume_model="X2")
    finally:
        _wipe(dev)
    with pytest.raises(ValueError, match="bipolar"):                 # BTBT needs a bipolar region
        _mesh_device("tz_d1b", 1, btbt=None)
        try:
            ds.delete_equation(device="tz_d1b", region="r", name="HoleContinuityEquation")
            _R.forget("tz_d1b", "HoleContinuityEquation")
            from dynameta.carriers.tunneling import setup_btbt
            setup_btbt("tz_d1b", "r", SILICON_HURKX)
        finally:
            _wipe("tz_d1b")

    def c(v):
        return lambda nn: np.full_like(np.asarray(nn, dtype=float), v)
    with pytest.raises(ValueError, match="bipolar_dd"):
        TransportModel(n_bg_m3=1e24, eps_static=11.7, dos_mass_kg_of_n_m3=c(M_E), physics="equilibrium",
                       btbt=SILICON_HURKX)
    assert TransportModel(n_bg_m3=1e24, eps_static=11.7, dos_mass_kg_of_n_m3=c(M_E)).btbt is None
    from dynameta.carriers.devsim_3d import Stacked3DSpec
    assert Stacked3DSpec().btbt is None and Stacked3DSpec().oxide_tunneling is None
    with pytest.raises(ValueError, match="bipolar_dd"):
        Stacked3DSpec(btbt=SILICON_HURKX)
    with pytest.raises(ValueError, match="E_F_eV"):
        OxideTunnelingParams(model="tsu_esaki")
    with pytest.raises(ValueError):
        OxideTunnelingParams(model="wkb")
    # the Hurkx Si set in SI (hand conversion above)
    assert SILICON_HURKX.A_SI == pytest.approx(A_SI, rel=1e-15)
    assert SILICON_HURKX.B_V_per_m == pytest.approx(B_SI, rel=1e-15) and SILICON_HURKX.P == P_SI


@pytest.mark.slow
def test_w3_stacked3d_btbt_and_oxide_tunneling_wiring(tmp_path):
    """3-D builder: btbt + oxide_tunneling on a laterally-uniform Si/SiO2 stack (~35 s)."""
    pytest.importorskip("gmsh")
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    from dynameta.constants import M_E
    from dynameta.sweep import BiasPoint
    prm = OxideTunnelingParams(model="fn")
    spec = Stacked3DSpec(semi_material="Si", oxide_material="SiO2", lateral_m=4e-9, semi_thk_m=20e-9,
                         oxide_thk_m=4e-9, n_bg_m3=1e24, eps_semi=11.7, eps_oxide=3.9, dos_mass_kg=1.08 * M_E,
                         dos_mass_p_kg=0.81 * M_E, physics="bipolar_dd", n_i_m3=ZNI, mobility_m2Vs=ZMU_N,
                         mobility_p_m2Vs=ZMU_P, tau_srh_s=1e-7, grid_n=(3, 3, 9), mesh_min_nm=0.5,
                         mesh_max_nm=2.0, btbt=SILICON_HURKX, oxide_tunneling=prm)
    s = Devsim3DEquilibrium(spec, device_name="tz_w3d", mesh_name="tz_w3m", msh_path=str(tmp_path / "w3.msh"))
    try:
        with _quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore")                         # FD-ceiling advisories of the 3-D solve
            cf = s.solve(BiasPoint({"gate": 4.0}, "g"))
        assert "BTBT_U0" in ds.get_element_model_list(device="tz_w3d", region="semi")
        u = np.abs(np.asarray(ds.get_element_model_values(device="tz_w3d", region="semi", name="BTBT_U0")))
        em = np.asarray(ds.get_element_model_values(device="tz_w3d", region="semi", name="BTBT_Emag"))
        r = oxide_tunneling_current("tz_w3d", "oxide", "semi", prm)
    finally:
        s.teardown()
    ex = cf.extras["oxide_tunneling"]["oxide"]
    assert ex["I_A"] == r["I"] and ex["I_A"] > 0.0
    assert r["area"].sum() == pytest.approx(4e-9 ** 2, rel=1e-3)   # MEASURED 6e-4 (tet face shares)
    assert np.ptp(r["V_ox_V"]) / np.mean(r["V_ox_V"]) < 0.03       # MEASURED 1.5 % (the 3-D FV solution)
    # an insulated gate: the stack is in EQUILIBRIUM (n p = n_i^2), so the D factor zeroes BTBT even where
    # the accumulation field (~3e8 V/m in the Si) gives a Kane prefactor of ~1e34 m^-3 s^-1. MEASURED
    # max|U| = 3.3e22 = 2e-12 of it -- the 3-D solve's rel_tol 1e-5 on n, p, not physics.
    with np.errstate(under="ignore"):
        kane = A_SI * em ** P_SI * np.exp(-B_SI / em)
    assert float(kane.max()) > 1e30
    assert float(u.max()) < 1e-8 * float(kane.max())
