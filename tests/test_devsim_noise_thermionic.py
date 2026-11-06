"""Gates for the THERMIONIC-INTERFACE SHOT NOISE of dynameta.carriers.noise_analysis (DynaMeta v0.15, plan item
4.3.1; P3 section 2; user decisions 43 -- on by default -- and 44 -- a flux interface without a pair model
refuses). Audit note: docs/audit/2026-11-06-v0.15-thermionic-shot-noise.md.

Physics. A D1a thermionic-emission (TE) interface carries two opposed Richardson currents J_01, J_10; each
is an independent Poisson stream, so the interface current noise is S_J = 2q (J_01 + J_10) / A per node
pair -- 4kT G_int at zero bias, 2qJ coth(qV/2kT) under bias. noise_analysis evaluates it through the
general two-point source (rows +1 on r0, -1 on r1), with the pair rate written from the SAME strings as the
flux (physics_heterojunction.thermionic_flux_and_pair) and each pair weighted by the area DEVSIM's fluxterm
assembly multiplies the flux by.

Oracles (none shares code with noise_analysis):
  * the fluctuation-dissipation theorem S_I = 4kT Re Y (ac_analysis ssac) and 4kT G_0 (DC dI/dV), budget
    1e-4 (the D2a gate B budget);
  * the coth law S = 2qJ coth(qV/2kT) (van der Ziel 1955, the D2a oracle C1) from the DC current J and the
    applied V only, on an interface-dominated isotype heterojunction, with a tolerance derived from the
    solved bulk share;
  * DEVSIM's own ASSEMBLED residual (get_matrix_and_rhs with the flux model set to a unit constant) for the
    per-node area (T0) -- the quantity the source must reproduce;
  * an equilibrated scipy solve of the adjoint system for the fields on the two interface copies.
OLD CODE (ebd9e65, MEASURED by running this file against an export of it, with this tree's
validation/devsim_noise.py fixtures): all 14 fail (the FDT misses by -0.694 / -0.9977 / -0.9986 in 1-D /
2-D / 3-D, the coth law by -1.0; the pair model, by_interface and the te source do not exist).
Runtime ~33 s serial (Windows dev box).
"""
import dataclasses
import math

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
V = pytest.importorskip("validation.devsim_noise")
from scipy.constants import e as QE, k as KB                  # noqa: E402

from dynameta.carriers import noise_analysis as NA            # noqa: E402
from dynameta.carriers import physics_heterojunction as HJ    # noqa: E402
from dynameta.carriers import physics_traps as PT             # noqa: E402

ds = devsim
KT = KB * 300.0
EPS = np.finfo(float).eps
EQ_N, EQ_P = "ElectronContinuityEquation", "HoleContinuityEquation"
FF = [1.0, 1e5, 1e7, 1e9]
FDT_TOL = 1e-4                                                # D2a gate B budget


@pytest.fixture(autouse=True)
def _clean():
    V.fresh()
    V.set_bias._v = {}
    yield
    V.fresh()


def _fdt(freqs):
    sn = V.noise(freqs)
    Y = V.ssac_Y(freqs)
    return sn, sn.S / (4 * KT * Y.real) - 1.0


def _te(sn):
    return sum(v for k, v in sn.by_mechanism.items() if k in ("te_n", "te_p"))


# =====================================================================================================
# T1: the pair model is the flux's own strings
# =====================================================================================================

def test_pair_model_is_written_from_the_flux_strings():
    """thermionic_flux_and_pair: the flux string is BYTE-identical to the pre-v0.15 one (re-typed from
    ebd9e65 physics_heterojunction: "{p}*(exp({w0}) - exp({w1}))" with p = repr(qsign A* T^2)), and the pair
    string is |p| (exp(w0) + exp(w1)) of the SAME w0 / w1 -- evaluated here with math.exp: pair >= |flux|
    everywhere, pair = 2 |p| e^w and flux = 0 at w0 = w1 (zero bias: J_01 = J_10 = J_s, pair = 2 J_s)."""
    rng = np.random.default_rng(3)
    for pref in (-1.234e5, 7.5e-1):
        w0, w1 = "(L@r0 - 0.5)", "(L@r1 - 0.25)"
        flux, pair = HJ.thermionic_flux_and_pair(pref, w0, w1)
        assert flux == "{p}*(exp({w0}) - exp({w1}))".format(p="{!r}".format(pref), w0=w0, w1=w1)
        assert HJ.pair_model_name("jn_HJn") == "jn_HJn_pair"
        for a, b in rng.uniform(-30, 5, (50, 2)).tolist():
            env = {"exp": math.exp}
            f = eval(flux.replace("L@r0", repr(a)).replace("L@r1", repr(b)), env)     # noqa: S307
            p = eval(pair.replace("L@r0", repr(a)).replace("L@r1", repr(b)), env)     # noqa: S307
            assert p >= abs(f) and p == pytest.approx(abs(pref) * (math.exp(a - 0.5) + math.exp(b - 0.25)),
                                                       rel=4 * EPS)
        f, p = (eval(s.replace("L@r0", "0.0").replace("L@r1", "-0.25"), {"exp": math.exp})  # noqa: S307
                for s in (flux, pair))
        assert f == 0.0 and p == 2 * abs(pref) * math.exp(-0.5)


# =====================================================================================================
# T-b: the A* sweep (1-D, the xfail fixture's diode) -- the FDT holds at every A*, the TE share falls
# monotonically and the total tends to the CONTINUOUS-quasi-Fermi-level model
# =====================================================================================================

ASTAR_SWEEP = (1e-6, 1e-4, 1e-2, 1.0, 1e2)


def test_astar_sweep_fdt_and_continuous_limit():
    """validation.devsim_noise.hetero_two_region('thermionic', astar_scale) for A* x 1e-6 .. 1e2: S_I = 4kT Re Y
    to 1e-4 at 1 Hz - 1 GHz at EVERY scale (MEASURED <= 3.0e-5, the ssac low-frequency floor at 1 Hz;
    1.7e-6 against DC G_0 on the isotype device below); the TE share by_mechanism[te]/S at 1 Hz falls
    monotonically (MEASURED 0.69, 2.2e-2, 2.3e-4, 2.3e-6, 2.3e-8); |S/S_cont - 1| falls monotonically to the
    continuous model, and A* x 1e6 equals it to 1e-4 (MEASURED 2.3e-12; D1a gate E reached -1.1e-5 in DC).
    The interface source of a resistance r_i in a chain of total R adds (r_i/R)^2 4kT/r_i -> 0 as r_i -> 0:
    the continuous model is its limit, not a separate physics. OLD CODE: the FDT misses by -0.69 / -2.2e-2
    / -2.3e-4 at the three smallest scales (no source)."""
    V.hetero_two_region("continuous")
    s_cont = V.noise(FF).S
    shares, devs = [], []
    for a in ASTAR_SWEEP + (1e6,):
        V.hetero_two_region("thermionic", astar_scale=a)
        sn, r = _fdt(FF)
        assert np.max(np.abs(r)) < FDT_TOL, (a, r)
        shares.append(float(_te(sn)[0] / sn.S[0]))
        devs.append(float(np.max(np.abs(sn.S / s_cont - 1.0))))
    assert all(x > y for x, y in zip(shares, shares[1:])), shares
    assert all(x > y for x, y in zip(devs, devs[1:])), devs
    assert shares[0] > 0.5                                      # interface-dominated at the small end
    assert devs[-1] < 1e-4, devs


# =====================================================================================================
# T-c: forward / reverse bias on the interface-dominated isotype n-GaAs / N-Al0.3Ga0.7As heterojunction
# =====================================================================================================

def _bulk_resistance(dev):
    """Series bulk resistance int dz / (q (mu_n n + mu_p p)) of both layers [Ohm m^2] from the SOLVED
    densities, with the materials' own mobilities (validation.devsim_heterojunction) -- trapezoid rule."""
    from validation.devsim_heterojunction import ALGAAS, GAAS
    r = 0.0
    for reg, m in (("ngaas", GAAS), ("nalgaas", ALGAAS)):
        x = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="x"))
        n = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
        p = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="Holes"))
        rho = 1.0 / (QE * (m["mu_n"] * n + m["mu_p"] * p))
        r += float(np.sum(0.5 * (rho[1:] + rho[:-1]) * np.diff(x)))
    return r


ISO_M = (1, 2, 4, 6, 8)


def test_isotype_heterojunction_obeys_the_coth_law_under_bias():
    """D1a gate C's isotype n-GaAs / N-Al0.3Ga0.7As (1e17 cm^-3, 400 nm each; validation.devsim_noise.
    hetero_isotype_1d) with A* x 1e-6, so the interface carries all but ~1e-7 of the resistance: at
    V = +-1 .. 8 kT/q, S = 2 q J coth(qV/2kT) from the DC current J and the applied V alone (J_01/J_10 =
    exp(qV_int/kT) EXACTLY for the quasi-Fermi-level flux, so S = 2q(J_01 + J_10) = 2qJ coth). MEASURED:
    |S/oracle - 1| <= 1.3e-9; S/2qJ = 2.164 / 1.313 / 1.037 / 1.0050 / 1.00067 (coth(m/2)).
    TOLERANCE, derived per bias from the solve: the series chain gives S = Sum_k (r_k/R)^2 S_k, so with the
    bulk share eps = R_b/R (R_b = int dz/(q mu n), R = dV/dJ by +-0.1 mV central difference) the interface
    term is weighted (1 - eps)^2 (2 eps), the bulk adds its own ~eps, and the bulk drop J R_b shifts the coth
    argument by J R_b/V_T (relative effect delta = J R_b / (V_T sinh v)): budget 3 eps + delta, gated at 3x;
    the plan's regime (budget x 3 <= 3e-3) is asserted. At zero bias pair = 2 J_s: the DEVSIM pair model
    equals 2 (kT/q) G_0 (G_0 from DC) to the same budget.
    FINDING vs the plan: the plan's form S = 2q(J + 2 J_s) with ONE J_s = (kT/q) G_0 misses here by -44 ..
    -57 % forward and -0.3 .. -6 % reverse (MEASURED): the isotype junction's reverse stream J_10 is bias-
    dependent (the barrier-side interface density moves 4.6e19 -> 7e16 m^-3 over +8 kT/q), so the J-V is
    not ideal; the coth form in the APPLIED V is exact for an interface-dominated device and is the gate.
    OLD CODE: S misses the oracle by -1.0 (the bulk share only, ~1e-7)."""
    dev = V.hetero_isotype_1d(astar_scale=1e-6)
    VT = KT / QE
    G0 = V.dc_conductance(0.0, V.ISO_AE, dv=1e-4)
    eps0 = _bulk_resistance(dev) * G0
    assert eps0 < 1e-3
    pv = np.asarray(ds.get_interface_model_values(device=dev, interface="jn", name="jn_HJn_pair"))
    assert pv[0] == pytest.approx(2.0 * VT * G0, rel=3 * 3 * eps0 + 1e-6)   # + the +-0.1 mV FD truncation
    for sign in (+1, -1):
        V.set_bias._v = {}
        V.hetero_isotype_1d(astar_scale=1e-6)
        for m in ISO_M:
            v = sign * m * VT
            V.set_bias(v, V.ISO_AE, step=0.01)
            J = V.source_current()
            sn = V.noise([1.0])
            Rd = 1.0 / V.dc_conductance(v, V.ISO_AE, dv=1e-4)
            eps = _bulk_resistance(dev) / Rd
            delta = abs(J) * _bulk_resistance(dev) / (VT * math.sinh(m))
            tol = 3.0 * (3.0 * eps + delta)
            assert tol <= 3e-3, (m, tol)
            oracle = 2.0 * QE * abs(J) / math.tanh(m / 2.0)
            assert abs(sn.S[0] / oracle - 1.0) <= max(tol, 1e-8), (sign, m, sn.S[0] / oracle - 1.0, tol)
            assert float(_te(sn)[0] / sn.S[0]) > 0.999


# =====================================================================================================
# T0: the per-node area DEVSIM's fluxterm assembly ACTUALLY uses, against the source's own weights
# =====================================================================================================

def _assembled_flux_weights(dev, iface, flux_model, var="Electrons", eq=EQ_N):
    """MEASURE the per-node weight DEVSIM's fluxterm assembly gives a unit flux: the interface model (and
    its derivatives) set to 1.0 vs 0.0 at the same state, residual difference -(rhs_1 - rhs_0) per region row
    (get_matrix_and_rhs returns rhs = -F). Also a round-off bound per row: both assemblies sum the row's
    other terms identically, so the difference is the unit flux up to the running sum's rounding,
    <= 16 eps (P_row + |w|) with P_row = Sum_edges |ElectronCurrent| EdgeCouple + |ElectronGeneration|
    NodeVolume of the node (the magnitude of every term the row adds). DESTRUCTIVE: the flux model stays
    replaced -- measure last."""
    regs = list(ds.get_region_list(device=dev, interface=iface))
    scale = {}
    for r in regs:
        cur = np.asarray(ds.get_edge_model_values(device=dev, region=r, name="ElectronCurrent"))
        ec = np.asarray(ds.get_edge_model_values(device=dev, region=r, name="EdgeCouple"))
        ds.edge_from_node_model(device=dev, region=r, node_model="node_index")
        i0 = np.asarray(ds.get_edge_model_values(device=dev, region=r, name="node_index@n0")).astype(int)
        i1 = np.asarray(ds.get_edge_model_values(device=dev, region=r, name="node_index@n1")).astype(int)
        nv = np.asarray(ds.get_node_model_values(device=dev, region=r, name="NodeVolume"))
        gen = np.asarray(ds.get_node_model_values(device=dev, region=r, name="ElectronGeneration"))
        P = np.abs(gen) * nv
        np.add.at(P, i0, np.abs(cur) * ec)
        np.add.at(P, i1, np.abs(cur) * ec)
        scale[r] = P
    out = {}
    for val in ("0.0", "1.0"):
        ds.interface_model(device=dev, interface=iface, name=flux_model, equation=val)
        for k in (0, 1):
            ds.interface_model(device=dev, interface=iface, name="{}:{}@r{}".format(flux_model, var, k),
                               equation="0.0")
        with V.quiet():
            out[val] = np.asarray(ds.get_matrix_and_rhs(format="csc")["static"]["rhs"], dtype=float)
    d = -(out["1.0"] - out["0.0"])
    res = {}
    for r in regs:
        rows = np.asarray(ds.get_equation_numbers(device=dev, region=r, equation=eq), dtype=np.int64)
        w = np.zeros(rows.size)
        ok = rows >= 0
        w[ok] = d[rows[ok]]
        res[r] = (w, 16 * EPS * (scale[r] + np.abs(w)))
    return regs, res


def _code_weights(dev, iface, regs, mech="te_n"):
    """The per-node area the te source uses: A_i = P_i q / pair_i, mapped to both copies (+A on r0, -A on
    r1, the sign the assembly gives); 0 at nodes without a source."""
    srcs = [s for s in NA._interface_sources(dev, set(ds.get_region_list(device=dev)), {mech})
            if s.interface == iface]
    assert len(srcs) == 1
    s = srcs[0]
    pv = np.asarray(ds.get_interface_model_values(device=dev, interface=iface, name=iface + "_HJn_pair"))
    i0, _i1 = NA._interface_pairs(dev, iface, *regs)
    pmap = dict(zip(i0.tolist(), pv.tolist()))
    A = s.rate * QE / np.asarray([pmap[int(i)] for i in s.rows[0].nodes])
    out = {}
    for row, sgn in zip(s.rows, (1.0, -1.0)):
        n = len(ds.get_node_model_values(device=dev, region=row.region, name="x"))
        w = np.zeros(n)
        w[row.nodes] = sgn * A
        out[row.region] = w
    return out


def _check_t0(dev, ifaces):
    """Assert the source's areas == DEVSIM's assembled weights (within the round-off bound) on every flux
    interface, BEFORE the destructive measurement; returns {(iface, region): (w, SurfaceArea, ia)}."""
    code = {i: _code_weights(dev, i, list(ds.get_region_list(device=dev, interface=i))) for i in ifaces}
    got = {}
    for i in ifaces:
        regs, res = _assembled_flux_weights(dev, i, i + "_HJn")
        for r in regs:
            w, tol = res[r]
            assert np.all(np.abs(w - code[i][r]) <= tol), (i, r, float(np.max(np.abs(w - code[i][r]))))
            sa = np.asarray(ds.get_node_model_values(device=dev, region=r, name="SurfaceArea"))
            got[(i, r)] = (np.abs(w), sa, PT.interface_node_area(dev, r, i))
    return got


def test_t0_1d_area_is_one():
    """1-D: DEVSIM loads the unit flux as +1 on the r0 copy and -1 on the r1 copy (the +J A0 / -J A1 rule read in
    DEVSIM 2.11 InterfaceEquation.cc NodeVolumeType2Assemble), and the source uses exactly that."""
    dev, _fr = V.hetero_two_region("thermionic", astar_scale=1e-6)
    got = _check_t0(dev, ["jn"])
    for (_i, _r), (w, _sa, _ia) in got.items():
        assert sorted(w[w != 0].tolist()) == [1.0]


def test_t0_2d_nonuniform_and_shared_corner():
    """2-D z-stacked heterostack (validation.devsim_noise.hetero_2d_stack: lateral spacing 10 -> 50 nm), plain
    and with the top layer split so two TE interfaces meet the vertical one at a corner node: the source's
    per-node areas equal DEVSIM's assembled weights within the derived round-off bound (MEASURED <= 3.3e-16
    of the largest share, <= 0.093 of the bound). On these rectilinear meshes DEVSIM's SurfaceArea (at the
    interface's own nodes -- in 2-D it also holds the region's CONTACT faces) and
    physics_traps.interface_node_area agree with them node for node, and the 2-D mesher gives the corner to
    the FIRST interface ('jn' owns it, 'jn2' does not -- and jn2 loses its half-edge there: 0.99 um of 1 um
    of interface is loaded)."""
    for split in (None, 0.6e-6):
        dev = V.hetero_2d_stack(split_x=split, dev="t0_2d" if split is None else "t0_2ds")
        ifaces = ["jn"] if split is None else ["jn", "jn2"]
        got = _check_t0(dev, ifaces)
        for (_i, _r), (w, sa, ia) in got.items():
            on = ia > 0                                     # this interface's nodes (SurfaceArea also holds
            assert np.all(sa[w > 0] > 0)                    # the CONTACT faces of the region in 2-D)
            assert np.max(np.abs(w - ia)) <= 1e-12 * sa[on].max()
            assert np.max(np.abs(w[on] - sa[on])) <= 1e-12 * sa[on].max()
        if split is not None:
            x = np.asarray(ds.get_node_model_values(device=dev, region="r_lo", name="x"))
            y = np.asarray(ds.get_node_model_values(device=dev, region="r_lo", name="y"))
            k = int(np.argmin(np.abs(x - split) + np.abs(y - V.H2D_YJ)))
            assert got[("jn", "r_lo")][0][k] > 0 and got[("jn2", "r_lo")][0][k] == 0.0


T3D = [("kuhn", dict()), ("jitter", dict(jitter=0.3)), ("jitter_exact", dict(jitter=0.3, exact=("r_lo", "r_hi"))),
       ("split", dict(split_i=3))]


@pytest.mark.parametrize("tag,kw", T3D, ids=[t for t, _ in T3D])
def test_t0_3d_area_is_devsims_assembled_area(tag, kw):
    """3-D two-region Kuhn-tensor stack (validation.devsim_noise.hetero_3d_stack, graded lateral axes): the
    source's per-node areas equal DEVSIM's assembled weights within the derived round-off bound (MEASURED
    <= 3.2e-16 of the largest share, <= 0.15 of the bound) on the tensor mesh, on a laterally JITTERED
    (non-well-centred) mesh with DEVSIM's own geometry, with devsim_geometry.apply_exact_geometry on both
    regions (the assembly reads the OVERWRITTEN SurfaceArea: sum = the true 300 x 300 nm^2 to round-off, vs
    DEVSIM's own 1.00598 x), and on the SPLIT stack where two TE interfaces share a line of nodes with the
    vertical one. These show what the area is NOT:
    jittered, physics_traps.interface_node_area misses the assembled weight by 11.0 % (DEVSIM geometry) /
    4.6 % (exact geometry) of the largest share; split, DEVSIM's SurfaceArea is the SUM over the region's
    interfaces at the shared nodes and DEVSIM SKIPS those pairs (another interface carries the same equation
    there: RemoveInterfaceNodesFromList) -- only 18 of 24 'jn' pairs and 12 of 18 'jn2' pairs are loaded."""
    dev = V.hetero_3d_stack(dev="t0_3d_" + tag, **kw)
    ifaces = [i for i in ds.get_interface_list(device=dev) if i != "jv"]
    got = _check_t0(dev, ifaces)
    for (i, r), (w, sa, ia) in got.items():
        m = sa.max()
        if tag == "kuhn":
            assert np.max(np.abs(w - ia)) <= 1e-12 * m
        if tag.startswith("jitter"):
            assert np.max(np.abs(w - ia)) > 0.04 * m                  # not the geometric split
            assert np.max(np.abs(w - sa)) <= 1e-12 * m
        if tag == "jitter_exact":
            assert w.sum() == pytest.approx(9e-14, rel=1e-12)          # exact 300 x 300 nm^2
        if tag == "jitter":
            assert w.sum() / 9e-14 - 1 > 5e-3                          # DEVSIM's unsigned over-count
        if tag == "split":
            assert np.max(np.abs(w - sa)) > 0.4 * m                   # SurfaceArea alone is wrong here
            assert np.count_nonzero(w) == {"jn": 18, "jn2": 12}[i]


# =====================================================================================================
# T-d: FDT on non-uniform 2-D / 3-D interfaces; equal area shares (and an unmasked pair set) must break it
# =====================================================================================================

def _rescaled(fn):
    orig = NA._interface_sources

    def wrapped(device, region_names, mech):
        return [dataclasses.replace(s, rate=fn(device, s)) for s in orig(device, region_names, mech)]
    return wrapped


def _equal_shares(device, s):
    sa = np.asarray(ds.get_node_model_values(device=device, region=s.region, name="SurfaceArea"))[s.rows[0].nodes]
    return s.rate * sa.mean() / sa


def test_fdt_2d_nonuniform_interface_and_equal_shares_break_it(monkeypatch):
    """2-D: S_I = 4kT Re Y to 1e-4 at 1 Hz and 1 MHz (MEASURED <= 5.1e-11 plain and split; TE share 0.998) on
    the non-uniform-lateral-mesh stack whose partial top contact makes the adjoint field vary along the
    interface. ADVERSARIAL: the same total area spread in EQUAL shares over the pairs breaks it (MEASURED
    +2.27 % plain), i.e. the per-node area is load-bearing. OLD CODE: -0.9977."""
    for split in (None, 0.6e-6):
        V.hetero_2d_stack(split_x=split, dev="fdt_2d" if split is None else "fdt_2ds")
        sn, r = _fdt([1.0, 1e6])
        assert np.max(np.abs(r)) < FDT_TOL, (split, r)
        assert float(_te(sn)[0] / sn.S[0]) > 0.99
        assert set(sn.by_interface) == ({"jn"} if split is None else {"jn", "jn2"})
        if split is None:
            Y = V.ssac_Y([1.0, 1e6])
            monkeypatch.setattr(NA, "_interface_sources", _rescaled(_equal_shares))
            bad = V.noise([1.0, 1e6]).S / (4 * KT * Y.real) - 1.0
            monkeypatch.undo()
            assert np.min(np.abs(bad)) > 1e-3, bad


def test_fdt_3d_and_adversarials(monkeypatch):
    """3-D: S_I = 4kT Re Y to 1e-4 at 1 Hz (MEASURED <= 1.2e-11) on the Kuhn stack, the jittered mesh (DEVSIM's
    own geometry: the source follows DEVSIM's SurfaceArea, so the discrete FDT still holds), the jittered mesh
    with exact geometry, and the split stack with shared interface nodes. ADVERSARIAL: equal area shares
    break it (MEASURED +1.30 % on the Kuhn stack), and on the split stack loading EVERY pair DEVSIM's assembly
    skips (no active-pair mask, SurfaceArea as the area) breaks it by +151 %. (physics_traps.
    interface_node_area instead of the assembled area moves the FDT by only -2.3e-6 on the jittered exact-
    geometry stack -- below the budget, so the per-node T0 gate above is the one that pins it.) OLD CODE:
    -0.9986 / -0.9986 / -0.9990."""
    orig = NA._interface_sources

    def no_mask(device, region_names, mech):
        out = []
        for s in orig(device, region_names, mech):
            r0, r1 = s.rows[0].region, s.rows[1].region
            i0, i1 = NA._interface_pairs(device, s.interface, r0, r1)
            tag = "n" if s.rows[0].equation == EQ_N else "p"
            pv = np.asarray(ds.get_interface_model_values(device=device, interface=s.interface,
                                                          name="{}_HJ{}_pair".format(s.interface, tag)))
            sa = np.asarray(ds.get_node_model_values(device=device, region=r0, name="SurfaceArea"))[i0]
            n = i0.size
            out.append(NA.TwoPointSource(s.mechanism, r0, pv * sa / QE,
                                         (NA.SourceRow(r0, s.rows[0].equation, np.ones(n), i0),
                                          NA.SourceRow(r1, s.rows[1].equation, -np.ones(n), i1)), i0,
                                         interface=s.interface))
        return out
    for tag, kw in T3D:
        V.hetero_3d_stack(dev="fdt_3d_" + tag, **kw)
        sn, r = _fdt([1.0])
        assert np.max(np.abs(r)) < FDT_TOL, (tag, r)
        assert float(_te(sn)[0] / sn.S[0]) > 0.99
        Y = V.ssac_Y([1.0])
        if tag == "kuhn":
            monkeypatch.setattr(NA, "_interface_sources", _rescaled(_equal_shares))
            bad = V.noise([1.0]).S / (4 * KT * Y.real) - 1.0
            monkeypatch.undo()
            assert abs(bad[0]) > 1e-3, bad
        if tag == "split":
            monkeypatch.setattr(NA, "_interface_sources", no_mask)
            bad = V.noise([1.0]).S / (4 * KT * Y.real) - 1.0
            monkeypatch.undo()
            assert abs(bad[0]) > 0.5, bad


def test_exact_geometry_on_one_region_only_refuses():
    """devsim_geometry.apply_exact_geometry on ONE region of a jittered 3-D stack: DEVSIM then multiplies the flux
    by different areas on the two copies (+J A0 / -J A1), so its interface current is not conserved; the noise
    refuses rather than pick one (RuntimeError naming both regions)."""
    V.hetero_3d_stack(dev="t3d_one", jitter=0.3, exact=("r_lo",))
    with pytest.raises(RuntimeError, match="not conserved"):
        V.noise([1.0])


# =====================================================================================================
# T-f: refusals, defaults, bookkeeping
# =====================================================================================================

def test_flux_interface_without_pair_model_refuses():
    """Decision 44: a fluxterm interface on a continuity equation WITHOUT its pair model makes noise_spectrum raise
    (it would be silently noiseless). Excluding te_n / te_p from `mechanisms` skips the check, and then S is
    exactly the region sources. A 'hybrid' interface equation has no noise model: NotImplementedError.
    OLD CODE: no raise (the interface was silently noiseless)."""
    dev, _fr = V.hetero_two_region("thermionic", astar_scale=1e-6)
    ds.delete_interface_model(device=dev, interface="jn", name="jn_HJn_pair")
    with pytest.raises(RuntimeError, match="without a pair-rate"):
        V.noise([1.0])
    sn = V.noise([1.0], mechanisms=("diffusion_n", "diffusion_p", "gr", "te_p"))
    assert set(sn.by_mechanism) == {"diffusion_n", "diffusion_p", "gr", "te_p"}
    ds.delete_interface_equation(device=dev, interface="jn", name=EQ_P)
    ds.interface_equation(device=dev, interface="jn", name=EQ_P, interface_model="jn_HJp", type="hybrid")
    with pytest.raises(NotImplementedError, match="hybrid"):
        V.noise([1.0], mechanisms=("diffusion_n", "diffusion_p", "gr"))


def test_bookkeeping_by_interface_maps_and_inert_default():
    """S = Sum by_region + Sum by_interface = Sum by_mechanism (to 1e-14); by_region holds no te key; with
    maps=True each interface event is split half to each node copy (the map sums equal by_mechanism); a
    device without a TE interface (the continuous model) has by_interface == {} and no te key (T-f (i): the
    new path is inert there -- the cross-version float.hex identity is in the audit note)."""
    dev, _fr = V.hetero_two_region("thermionic", astar_scale=1e-4)
    sn = V.noise(FF, maps=True)
    tot_r = sum(sn.by_region.values())
    tot_i = sum(v for d in sn.by_interface.values() for v in d.values())
    np.testing.assert_allclose(tot_r + tot_i, sn.S, rtol=1e-14)
    np.testing.assert_allclose(sum(sn.by_mechanism.values()), sn.S, rtol=1e-14)
    assert set(sn.by_interface) == {"jn"} and set(sn.by_interface["jn"]) == {"te_n", "te_p"}
    assert all("te_n" not in d for d in sn.by_region_mechanism.values())
    for m in ("te_n", "te_p"):
        halves = [float(sn.maps[r][m][0].sum()) for r in ("r_n", "r_p")]
        assert halves[0] == pytest.approx(halves[1], rel=1e-14)
        assert sum(halves) == pytest.approx(float(sn.by_mechanism[m][0]), rel=1e-14)
        k_r = int(np.argmax(sn.maps["r_n"][m][0]))
        assert k_r == len(sn.maps["r_n"]["x"]) - 1                  # the interface copy of r_n
    V.hetero_two_region("continuous")
    sc = V.noise(FF)
    assert sc.by_interface == {} and not ({"te_n", "te_p"} & set(sc.by_mechanism))
