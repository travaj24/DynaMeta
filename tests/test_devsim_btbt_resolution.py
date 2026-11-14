"""v0.15 follow-up -- the runtime resolution check of the local BTBT current (carriers.tunneling
btbt_quadrature_error, and the UnreliableResultWarning btbt_generation_current emits above 1 %). Audit note:
docs/audit/2026-11-06-v0.15-followups.md, item 1; the hazard itself: docs/audit/2026-11-06-v0.15-btbt-presets.md
("Root cause").

Gates
  R1  the ESTIMATOR on frozen 1-D states whose edge fields sample a known continuum field at the edge midpoints
      (what the FV solve does, Gauss's law): the estimate equals the true relative quadrature error
      Sum_e h_e f(E_e) / Int f(E(s)) ds - 1, with the integral by scipy.integrate.quad (independent oracle),
      to 1e-10 absolute. Budget: quad epsrel 1e-13 on an integral of magnitude 1, the 16-point Gauss rule per
      half edge (integrand log-variation <= 0.6 per half edge: error < 1e-20), ~200 terms of roundoff 1e-16 x
      B/E ~ 20 (f's amplification of a field roundoff) -> < 1e-12; 1e-10 leaves 100x. Three fields:
        uniform   the abrupt-junction triangle E_m - a|s|, peak at an edge midpoint, uniform h = 0.05 nm;
        long      the same triangle with ONE 0.128 nm edge on the peak between 0.0125 nm edges (the old
                  ps-mesh's junction edge);
        parabola  a graded-junction peak E_m - c s^2 centred on a NODE, uniform 0.05 nm (true error ~0: the
                  midpoint rule is spectrally accurate on a smooth bell; the estimate must say so).
      ADVERSARIAL: two cruder continuum models miss the oracle by >= 1e-5 (asserted): ln f interpolated linearly
      between midpoints (the triangle's ln f is concave: MEASURED +7 % high on the uniform case) and E
      interpolated linearly between midpoints (misses the parabola's top between the two node-adjacent edges).
  R2  the WARNING on solved 1-D Si Kao direct Zener diodes (the P4c fixture of tests/test_devsim_btbt_presets.py,
      0.3 V reverse): the old mesh with the fixed 0.128 nm edge on the field peak warns, naming the region and
      that edge ([99.87173, 100] nm); the uniform junction mesh at h = 0.025 nm is silent (warnings are
      errors); the uniform h = 0.05 nm mesh warns (its MEASURED error vs the analytic estimate is +1.02 %, the
      estimate +1.50 %: sensitivity at the budget).

FAILS ON THE OLD CODE (d3c807c): btbt_quadrature_error does not exist (AttributeError, R1) and
btbt_generation_current emits no warning (R2's pytest.warns fails; MEASURED error on that mesh +11.0 %).
Runtime ~25 s.
"""
import math
import re
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")

import test_devsim_btbt_presets as P                                       # noqa: E402  (the Zener fixture)
from dynameta.carriers import physics_bipolar_dd as BP                       # noqa: E402
from dynameta.carriers import tunneling as TN                                # noqa: E402
from dynameta.warning_types import UnreliableResultWarning                   # noqa: E402

PRM = TN.SILICON_KAO_DIRECT
EM, A_TRI = 5.0e8, 1.73e17          # triangle peak [V/m] and slope [V/m^2]: k = B a / E_m^2 = 7.0e9 /m
C_PAR = 0.8 * EM / (2.5e-9) ** 2    # parabola curvature: E falls to 0.2 E_m at 2.5 nm


def _f(E):
    return np.exp(PRM.P * np.log(E) - PRM.B_V_per_m / E)          # f / A


def _field(kind, s):
    return EM - C_PAR * s * s if kind == "parabola" else EM - A_TRI * np.abs(s)


def _nodes(kind):
    half = 2.5e-9
    if kind == "uniform":
        h = 0.05e-9
        return h * (np.arange(-50, 51) + 0.5)                        # the peak (s = 0) is an edge MIDPOINT
    if kind == "long":
        h, he = 0.0125e-9, 0.128e-9
        right = 0.5 * he + h * np.arange(0, int(round((half - 0.5 * he) / h)) + 1)
        return np.concatenate([-right[::-1], right])
    return 0.05e-9 * np.arange(-50, 51)                              # parabola: the peak on a NODE


def _frozen(dev, xs, kind):
    """A 1-D bipolar slab on the nodes xs (no solve), with node potentials such that every edge field equals the
    continuum field at the edge midpoint, and n = p = 1e6 m^-3 (D = -1 + 2e-10, the same on every edge)."""
    P._wipe()
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    for k, x in enumerate(xs):
        ds.add_1d_mesh_line(mesh=mesh, pos=float(x), ps=1.0,
                            **({"tag": "a"} if k == 0 else {"tag": "b"} if k == xs.size - 1 else {}))
    ds.add_1d_contact(mesh=mesh, name="a", tag="a", material="metal")
    ds.add_1d_contact(mesh=mesh, name="b", tag="b", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="r", tag1="a", tag2="b")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="r", name="NetDoping", equation="1e22")
    BP.setup_bipolar_region(dev, "r", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16, mobility_n_m2Vs=0.135,
                            mobility_p_m2Vs=0.048, tau_n_s=1e-7, tau_p_s=1e-7, fd_enhancement=False, btbt=PRM)
    x = np.asarray(ds.get_node_model_values(device=dev, region="r", name="x"))
    s_mid = 0.5 * (x[1:] + x[:-1])
    phi = np.concatenate([[0.0], -np.cumsum(np.diff(x) * _field(kind, s_mid))])   # E = -dphi/dx on each edge
    for v, val in (("Potential", phi), ("Electrons", np.full(x.size, 1e6)), ("Holes", np.full(x.size, 1e6))):
        ds.set_node_values(device=dev, region="r", name=v, values=[float(a) for a in val])
    return x


def _oracle(x, kind):
    from scipy.integrate import quad
    s, h = 0.5 * (x[1:] + x[:-1]), np.diff(x)
    disc = float(np.sum(h * _f(_field(kind, s))))
    ref = quad(lambda t: float(_f(_field(kind, np.array(t)))), x[0], x[-1], points=[0.0], limit=1000,
               epsabs=0.0, epsrel=1e-13)[0]
    return disc / ref - 1.0, s, h


def _crude(s, h, kind, model):
    """ADVERSARIAL continuum models (half edges toward each neighbour, 64-point Gauss): 'logf' = ln f linear
    between midpoints, 'linE' = E linear between midpoints."""
    t, w = np.polynomial.legendre.leggauss(64)
    t = 0.5 * (t + 1.0)
    E = _field(kind, s)
    cont = 0.0
    for e in range(s.size):
        for j in (e - 1, e + 1):
            if not 0 <= j < s.size:
                cont += 0.5 * h[e] * _f(E[e])
                continue
            frac = 0.5 * h[e] / abs(s[j] - s[e]) * t
            if model == "logf":
                vals = np.exp(np.log(_f(E[e])) + (np.log(_f(E[j])) - np.log(_f(E[e]))) * frac)
            else:
                vals = _f(E[e] + (E[j] - E[e]) * frac)
            cont += 0.5 * h[e] * 0.5 * float(vals @ w)
    return float(np.sum(h * _f(E))) / cont - 1.0


@pytest.mark.parametrize("kind", ["uniform", "long", "parabola"])
def test_r1_estimator_equals_the_quadrature_error_on_a_known_field(kind):
    dev = "tbr_" + kind
    try:
        x = _frozen(dev, _nodes(kind), kind)
        rep = TN.btbt_quadrature_error(dev, "r")
    finally:
        P._wipe()
    eps, s, h = _oracle(x, kind)
    assert abs(rep["error"] - eps) <= 1e-10, (kind, rep["error"], eps)
    # MEASURED (oracle / estimate - oracle): uniform +1.3289e-2 / -9e-16 (cf. the pure exponential's
    # (kh/2)coth(kh/2) - 1 = 1.26e-2 at the peak's k h = 0.39), long +1.1286e-1 / -1e-13 (cf. k h_e/2 - 1 +
    # exp(-k h_e/2) = 1.05e-1 at k h_e = 1.0), parabola +2e-16 / +9e-15 (a smooth bell on a uniform mesh: the
    # midpoint rule is spectrally accurate, and the estimate must not invent an error).
    # Adversarial misses: logf +1.21e-3 (uniform), +1.13e-3 (long); linE +1.23e-3 (parabola).
    assert eps > 0.0 if kind != "parabola" else abs(eps) < 1e-12
    if kind == "long":
        assert rep["worst_edge_m"][1] - rep["worst_edge_m"][0] == pytest.approx(0.128e-9, rel=1e-9)
    # ADVERSARIAL: the cruder models of the continuum miss the oracle by far more than the tolerance
    bad = "linE" if kind == "parabola" else "logf"
    assert abs(_crude(s, h, kind, bad) - eps) >= 1e-5, (kind, bad, _crude(s, h, kind, bad), eps)


def _si_direct_current(h, junction_edge=None):
    prm, N, _hs = P.P4C_CASES["si_direct"]
    epsr, ni = P.SI
    W = math.ceil(P._analytic((prm,), N, ni, epsr, 0.4)[3] / 0.2e-9) * 0.2e-9
    try:
        P._zener("tbr_z", prm, N, ni, epsr, W, hj=h, junction_edge=junction_edge)
        P._ramp("tbr_z", 0.0, -0.3)
        rep = TN.btbt_quadrature_error("tbr_z", "b")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            I = TN.btbt_generation_current("tbr_z", "b")
        quiet = TN.btbt_generation_current("tbr_z", "b", check_resolution=False)
    finally:
        P._wipe()
    assert I == quiet                                               # the check never changes the value
    return I, rep, [x for x in w if issubclass(x.category, UnreliableResultWarning)], w


def test_r2_old_fixed_junction_edge_warns_and_the_uniform_mesh_does_not():
    J = P._analytic((PRM,), P.P4C_CASES["si_direct"][1], P.SI[1], P.SI[0], 0.3)[0]
    # the old mesh: one 0.128 nm edge on the peak (MEASURED error vs the analytic estimate +11.0 %)
    I, rep, unrel, _w = _si_direct_current(0.025e-9, junction_edge=0.12827e-9)
    assert I / J - 1.0 > 0.10
    assert len(unrel) == 1, [str(x.message) for x in unrel]
    msg = str(unrel[0].message)
    assert "region 'b'" in msg and "estimated quadrature error {:+.2%}".format(rep["error"]) in msg, msg
    assert re.search(r"Worst edge \[99\.8717\d*, 100\] nm", msg), msg          # the fixed junction edge
    assert rep["error"] > 10.0 * TN.BTBT_QUADRATURE_TOL
    # the uniform junction mesh (h = 0.025 nm; MEASURED error +0.16 %, estimate +0.37 %): silent
    I, rep, unrel, w = _si_direct_current(0.025e-9)
    assert abs(I / J - 1.0) < 0.01 and not w, [str(x.message) for x in w]
    assert 0.0 < rep["error"] < TN.BTBT_QUADRATURE_TOL
    # SENSITIVITY at the budget: the uniform 0.05 nm mesh (MEASURED error +1.02 %, estimate +1.50 %) warns
    I, rep, unrel, _w = _si_direct_current(0.05e-9)
    assert I / J - 1.0 > 0.01 and len(unrel) == 1
