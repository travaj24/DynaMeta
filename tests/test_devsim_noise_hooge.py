"""Hooge 1/f noise OFF the mesh axes (v0.14.0 release blocker, 2026-11-04).

The Hooge source is a fluctuation of the local conductivity, delta j = xi J with ONE scalar xi per
point: a RANK-ONE source tensor alpha_H J J^T / (n f) (noise_analysis module doc, item 2). The 1-D gate
F (tests/test_devsim_noise.py) cannot see how that tensor is discretized -- in 1-D every current is
along the only edge. Until this fix noise_analysis summed it edge by edge as if every edge were an
independent source, which drops the cross terms between the edges of one element: for a uniform field
at angle theta to the mesh axes the edge sum is (cos^4 + sin^4) of the truth.

THE FIXTURE has a closed-form answer at ANY orientation: a uniform n-type DD resistor, S_I = alpha_H
I^2 / (f N), N = n x area (per unit depth). It is built on a right-isosceles triangulation of the
integer lattice -- every element is well centred, so DEVSIM's 2-D volumes and couples are exact (the
D2c lesson) and the discrete potential of a uniform field is exactly linear -- in two orientations:
  aligned  rectangle, contacts on x = 0 / x = L: current ALONG the x edges (the control);
  diagonal the same resistor rotated 45 deg (contacts on x + y = const, insulating walls on
           x - y = const, both exactly unions of triangle hypotenuses): current at 45 deg to EVERY edge.
The fixture's own exactness is asserted first (area, uniform n, I = sigma V w / L), so the analytic
reference is valid on the discrete device, not just in the continuum.

MEASURED (dev box, 2026-11-04; S f N / (alpha I^2)):   aligned      diagonal
  6595bde (edge-independent Hooge)                      0.99999999964   0.49999999982
  this fix (element-assembled, one xi per element)      0.99999999964   0.99999999964
The 3.6e-10 residual is the same in both orientations (the 1 mV drift term of the DD resistor); the
diagonal - aligned difference is 2.3e-13. On the old code the diagonal gate fails by 0.5.
"""
import os

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
# NOT importorskip: a missing noise_analysis must FAIL these gates, not skip them
import validation.devsim_noise as V                                         # noqa: E402
from dynameta.carriers import physics_drift_diffusion as DD                 # noqa: E402
from dynameta.carriers.devsim_env import import_devsim                      # noqa: E402
from dynameta.carriers.physics_equilibrium import M_E                       # noqa: E402

ds = import_devsim()

H = 5e-9                   # lattice step [m]
U, W = 40, 12              # contact separation / width in lattice units (even: lattice-point corners)
N_BG, MU = 4e26, 0.004     # uniform donor density [m^-3], mobility [m^2/Vs]
ALPHA, V_BIAS = 2e-3, 1e-3


def _lattice_mesh(inside, on_lo, on_hi, rng_i, rng_j):
    """Right-isosceles triangulation of the lattice region `inside(i, j)`: a unit square with all four
    corners inside -> two triangles, with three -> that half-square (its hypotenuse lies on a
    diagonal boundary). Contacts = boundary segments with both ends on_lo / on_hi."""
    idx, P = {}, []

    def node(q):
        if q not in idx:
            idx[q] = len(P)
            P.append((q[0] * H, q[1] * H))
        return idx[q]
    tris = []
    for i in rng_i:
        for j in rng_j:
            c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            cin = [inside(*q) for q in c]
            if all(cin):
                tris += [(c[0], c[1], c[3]), (c[1], c[2], c[3])]
            elif sum(cin) == 3:
                tris.append(tuple(q for q, k in zip(c, cin) if k))
    count = {}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = tuple(sorted((a, b)))
            count[key] = count.get(key, 0) + 1
    lines = {"lo": [], "hi": []}
    for (a, b), k in count.items():
        if k == 1 and on_lo(*a) and on_lo(*b):
            lines["lo"].append((node(a), node(b)))
        elif k == 1 and on_hi(*a) and on_hi(*b):
            lines["hi"].append((node(a), node(b)))
    return P, [tuple(node(q) for q in t) for t in tris], lines


def _write_msh(path, P, tris, lines):
    names = [(1, "lo"), (1, "hi"), (2, "bulk")]
    tag = {n: k + 1 for k, (_d, n) in enumerate(names)}
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    out += ['{} {} "{}"'.format(d, tag[n], n) for d, n in names]
    out += ["$EndPhysicalNames", "$Nodes", str(len(P))]
    out += ["{} {!r} {!r} 0.0".format(k + 1, float(x), float(y)) for k, (x, y) in enumerate(P)]
    out += ["$EndNodes", "$Elements"]
    body = []
    for typ, grp, conn in ((1, "lo", lines["lo"]), (1, "hi", lines["hi"]), (2, "bulk", tris)):
        for row in conn:
            body.append("{} {} 2 {} {} {}".format(len(body) + 1, typ, tag[grp], tag[grp],
                                                  " ".join(str(v + 1) for v in row)))
    out += [str(len(body))] + body + ["$EndElements", ""]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out))


def _resistor(kind, tmp_path):
    """Build + bias the uniform resistor; returns (S f N / (alpha I^2) at 1 Hz and 1 kHz, fixture
    checks, the NoiseSpectrum)."""
    if kind == "aligned":
        mesh = _lattice_mesh(lambda i, j: 0 <= i <= U and 0 <= j <= W,
                             lambda i, j: i == 0, lambda i, j: i == U,
                             range(-1, U + 1), range(-1, W + 1))
        L, width = U * H, W * H
    else:
        mesh = _lattice_mesh(lambda i, j: 0 <= i + j <= U and abs(i - j) <= W,
                             lambda i, j: i + j == 0, lambda i, j: i + j == U,
                             range(-W, U + W + 1), range(-U - W, U + W + 1))
        L, width = U * H / np.sqrt(2.0), 2 * W * H / np.sqrt(2.0)
    path = os.path.join(str(tmp_path), "hooge_{}.msh".format(kind))
    _write_msh(path, *mesh)
    ds.create_gmsh_mesh(mesh="hg_m", file=path)
    ds.add_gmsh_region(mesh="hg_m", gmsh_name="bulk", region="bulk", material="ITO")
    ds.add_gmsh_contact(mesh="hg_m", gmsh_name="lo", region="bulk", name="lo", material="metal")
    ds.add_gmsh_contact(mesh="hg_m", gmsh_name="hi", region="bulk", name="hi", material="metal")
    ds.finalize_mesh(mesh="hg_m")
    ds.create_device(mesh="hg_m", device="hg")
    with V.quiet():
        DD.setup_semiconductor_region_dd("hg", "bulk", n_bg_m3=N_BG, eps_static=9.5,
                                         dos_mass_kg=0.35 * M_E, mobility_m2Vs=MU)
        DD.setup_contact_ohmic_dd("hg", "lo")
        DD.setup_contact_ohmic_dd_circuit("hg", "hi", node_name="vhi", source_name="V1")
    ae = max(1e10, N_BG * 1e-12)
    V._solve(ae, 1e-9)
    V.set_bias(V_BIAS, ae)
    I = V.source_current()
    n = np.asarray(ds.get_node_model_values(device="hg", region="bulk", name="Electrons"))
    vol = np.asarray(ds.get_node_model_values(device="hg", region="bulk", name="NodeVolume"))
    N = float(np.sum(n * vol))
    ff = np.array([1.0, 1.0e3])
    sn = V.noise(ff, hooge_alpha=ALPHA, maps=True)
    checks = {"area": float(vol.sum() / (L * width) - 1.0), "n_spread": float(n.max() / n.min() - 1.0),
              "ohm": float(I / (V.QE * MU * N_BG * V_BIAS * width / L) - 1.0)}
    return sn.by_mechanism["hooge_n"] * ff * N / (ALPHA * I * I), checks, sn


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()


def test_hooge_uniform_resistor_is_orientation_independent(tmp_path):
    """S_I = alpha_H I^2 / (f N) for the SAME uniform resistor along the mesh axes and at 45 deg to
    every edge (module doc above). Bound 1e-8: measured 3.6e-10 in both orientations (the DD drift
    term at 1 mV); the edge-independent assembly gives 0.5 at 45 deg."""
    r_al, chk_al, _ = _resistor("aligned", tmp_path)
    V.fresh()
    V.set_bias._v = {}
    r_dg, chk_dg, sn = _resistor("diagonal", tmp_path)
    for chk in (chk_al, chk_dg):            # the analytic reference holds on the DISCRETE fixture
        assert abs(chk["area"]) < 1e-12 and abs(chk["n_spread"]) < 1e-12, chk
        assert abs(chk["ohm"]) < 1e-9, chk  # measured 2e-16: a uniform field, no contact distortion
    assert np.max(np.abs(r_al - 1.0)) < 1e-8, r_al
    assert np.max(np.abs(r_dg - 1.0)) < 1e-8, r_dg          # 6595bde: 0.4999999998
    assert np.max(np.abs(r_dg - r_al)) < 1e-11, (r_dg, r_al)  # measured 2.3e-13
    # the per-node map (element shares split equally among its nodes) sums to the total
    tot = sn.maps["bulk"]["hooge_n"].sum(axis=1)
    assert np.allclose(tot, sn.by_mechanism["hooge_n"], rtol=1e-12, atol=0.0)
