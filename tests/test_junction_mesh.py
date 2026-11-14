"""The devsim-free kernels of the opt-in junction mesh band (v0.15.1, dynameta.carriers.junction_mesh; user decision
2026-11-08). Audit note: docs/audit/2026-11-08-v0.15-junction-mesh-band.md. The DEVSIM side (the builder option,
the junction probe, the defaults-off identity and the accuracy gate) is tests/test_devsim_junction_band.py.

Gates
  S1 abrupt_junction_scales == the textbook depletion approximation RE-DERIVED here (scipy.constants): symmetric
     homojunction x_n = x_p = sqrt(eps V_bi / (q N)), E_0 = sqrt(2 q V_bi N_eff / eps); a one-sided junction.
  S2 the two-permittivity (heterojunction) form, checked against its DEFINITION instead of a formula: the field
     it implies satisfies D continuity at the junction and integrates (scipy quad) to V_bi.
  S3 the BTBT width: k_0 = (B / E^2) |dE/dx| evaluated by a central difference of the implied field on the
     heavier side; width = sqrt(12 tol) / k_0 (tunneling's MESH RESOLUTION rule (k h)^2 / 12 = tol).
  B1-B6 band_lines on hand-built line sets: uniform lattice anchored at the junction (equal edges both sides),
     non-fixed lines inside dropped, fixed lines kept and clipping the band (remainder < h/4 merged), snapping
     onto an existing line, two overlapping junctions = one cluster with both junctions as nodes, duplicates.
  Z1 sign_changes: brackets, an undoped sliver, an n-p-n pair, side medians.
FAILS ON THE OLD CODE (098553f): the module does not exist (ImportError at collection).
Runtime < 1 s.
"""
import math

import numpy as np
import pytest
from scipy.constants import e as QE, epsilon_0 as EPS0, k as KB
from scipy.integrate import quad

from dynameta.carriers import junction_mesh as JM

VT = KB * 300.0 / QE


def test_s1_homojunction_scales_match_the_textbook_depletion_approximation():
    N, ni, er = 1e24, 1e16, 11.7
    sc = JM.abrupt_junction_scales(N, N, eps_r_p=er, eps_r_n=er, n_i_p=ni, n_i_n=ni, T_K=300.0)
    eps = er * EPS0
    vbi = VT * math.log(N * N / ni ** 2)
    xn = math.sqrt(eps * vbi / (QE * N))                       # symmetric: W = 2 sqrt(eps V_bi / (q N))
    e0 = math.sqrt(2.0 * QE * vbi * (N / 2.0) / eps)           # N_eff = N_A N_D / (N_A + N_D) = N / 2
    # budget: the library constants vs scipy's (CODATA 2018 eps0 vs 2022 differ by 6.8e-10) -> 1e-9 relative
    for got, want in ((sc.V_bi, vbi), (sc.x_n, xn), (sc.x_p, xn), (sc.W0, 2 * xn), (sc.E0, e0),
                      (sc.peak_width, VT / e0)):
        assert got == pytest.approx(want, rel=1e-9)
    assert sc.btbt_width is None and sc.spacing == sc.peak_width
    # one-sided p+ n: the depletion is (almost) all on the n side, W -> sqrt(2 eps V_bi / (q N_D))
    s1 = JM.abrupt_junction_scales(1e26, 1e22, eps_r_p=er, eps_r_n=er, n_i_p=ni, n_i_n=ni, T_K=300.0)
    vb1 = VT * math.log(1e26 * 1e22 / ni ** 2)
    assert s1.x_n == pytest.approx(math.sqrt(2 * eps * vb1 / (QE * 1e22) * 1e26 / (1e26 + 1e22)), rel=1e-9)
    assert s1.x_p / s1.x_n == pytest.approx(1e22 / 1e26, rel=1e-12)
    # a triangle field: E_0 = 2 V_bi / W_0 (the "4 V_bi / V_t lines per band" of the module docstring)
    assert sc.E0 == pytest.approx(2.0 * sc.V_bi / sc.W0, rel=1e-12)


def test_s2_two_permittivity_form_satisfies_its_definition():
    NA, ND, ep_r, en_r, nip, nin = 3e23, 1e23, 12.9, 12.0, 2.1e12, 2e10
    sc = JM.abrupt_junction_scales(NA, ND, eps_r_p=ep_r, eps_r_n=en_r, n_i_p=nip, n_i_n=nin, T_K=300.0)
    ep, en = ep_r * EPS0, en_r * EPS0

    def E(x):                                                # the depletion field, junction at x = 0
        if -sc.x_p <= x <= 0.0:
            return QE * NA * (x + sc.x_p) / ep
        if 0.0 < x <= sc.x_n:
            return QE * ND * (sc.x_n - x) / en
        return 0.0
    assert NA * sc.x_p == pytest.approx(ND * sc.x_n, rel=1e-12)                    # charge balance
    assert ep * E(-1e-30) == pytest.approx(en * E(1e-30), rel=1e-9)                 # D continuity
    v = quad(E, -sc.x_p, 0.0, epsrel=1e-12)[0] + quad(E, 0.0, sc.x_n, epsrel=1e-12)[0]
    assert v == pytest.approx(VT * math.log(NA * ND / (nip * nin)), rel=1e-9)      # int E dx = V_bi
    assert sc.E0 == pytest.approx(QE * ND * sc.x_n / en, rel=1e-9)
    # ADVERSARIAL: the one-permittivity (homojunction) formula with eps_p on both sides, put through the same
    # definition with the TRUE eps, misses int E dx = V_bi by ~ +5.6 % (the n side carries 3/4 of the drop and
    # its field is eps_p/eps_n = 1.075x too large)
    nv = JM.abrupt_junction_scales(NA, ND, eps_r_p=ep_r, eps_r_n=ep_r, n_i_p=nip, n_i_n=nin, T_K=300.0)
    vn = (QE * NA * nv.x_p ** 2 / (2 * ep) + QE * ND * nv.x_n ** 2 / (2 * en))
    assert vn / sc.V_bi - 1.0 > 0.05


def test_s3_btbt_width_is_the_tunnelling_resolution_rule():
    from dynameta.carriers.tunneling import BTBT_QUADRATURE_TOL
    NA, ND, er, ni, B = 1e26, 2e25, 11.7, 1e16, 1.01e10
    sc = JM.abrupt_junction_scales(NA, ND, eps_r_p=er, eps_r_n=er, n_i_p=ni, n_i_n=ni, T_K=300.0, btbt_B_V_per_m=B)
    eps = er * EPS0
    # k = (B / E^2) |dE/dx| at the field peak, steepest side (the heavier N_A): |dE/dx| = q N_A / eps there,
    # taken here by a central difference of E(x) = E_0 + (q N_A / eps) x on the p side
    h = 1e-13
    dEdx = ((sc.E0 - QE * NA / eps * h) - (sc.E0 - QE * NA / eps * 3 * h)) / (2 * h)
    k = B / sc.E0 ** 2 * abs(dEdx)
    assert sc.btbt_width == pytest.approx(math.sqrt(12.0 * BTBT_QUADRATURE_TOL) / k, rel=1e-6)
    assert sc.btbt_width * k == pytest.approx(0.3464, abs=1e-4)                    # "h <= 0.35 / k"
    # symmetric limit: k = B / V_bi (tunneling's MESH RESOLUTION paragraph)
    ss = JM.abrupt_junction_scales(1e26, 1e26, eps_r_p=er, eps_r_n=er, n_i_p=ni, n_i_n=ni, T_K=300.0,
                                   btbt_B_V_per_m=B)
    assert ss.btbt_width == pytest.approx(math.sqrt(12 * BTBT_QUADRATURE_TOL) * ss.V_bi / B, rel=1e-12)
    assert ss.spacing == min(ss.btbt_width, ss.peak_width) == ss.btbt_width


def test_s4_validation():
    with pytest.raises(ValueError, match="N_D"):
        JM.abrupt_junction_scales(1e24, 0.0, eps_r_p=1, eps_r_n=1, n_i_p=1, n_i_n=1, T_K=300)
    with pytest.raises(ValueError, match="spacing_m"):
        JM.JunctionBand(spacing_m=-1e-9)
    assert JM.resolve_junction_band(None) is None and JM.resolve_junction_band(False) is None
    assert JM.resolve_junction_band(True) == JM.JunctionBand()
    jb = JM.JunctionBand(half_width_m=5e-9)
    assert JM.resolve_junction_band(jb) is jb
    with pytest.raises(TypeError, match="junction_band"):
        JM.resolve_junction_band("on")
    # V_bi floor: an under-doped "junction" (N_A N_D < n_i^2) still gives finite, positive lengths
    lo = JM.abrupt_junction_scales(1e10, 1e10, eps_r_p=11.7, eps_r_n=11.7, n_i_p=1e16, n_i_n=1e16, T_K=300.0)
    assert lo.V_bi == pytest.approx(VT, rel=1e-9) and lo.W0 > 0.0


# -------------------------------------------------------------------------------------------- band_lines
def _edges(lines):
    xs = np.array(sorted(lines))
    return xs, np.diff(xs)


def test_b1_single_junction_uniform_lattice_with_the_junction_a_node():
    lines = {0.0: (20e-9, 20e-9), 1.5e-9: (20e-9, 20e-9), 1e-6: (1e-9, 1e-9), 2e-6 - 1.5e-9: (20e-9, 20e-9),
             2e-6: (20e-9, 20e-9)}
    fixed = {0.0, 1.5e-9, 2e-6 - 1.5e-9, 2e-6}
    h, w = 1.05e-9, 30e-9
    new = JM.band_lines(lines, fixed, [(1e-6 + 2e-16, w, h)])                # snapped onto the 1e-6 line
    xs, d = _edges(new)
    assert 1e-6 in new and new[1e-6] == (h, h)
    m = math.ceil(w / h)                                                      # the band reaches >= w
    band = (xs[:-1] >= 1e-6 - m * h - 1e-15) & (xs[1:] <= 1e-6 + m * h + 1e-15)
    assert d[band].size == 2 * m and np.allclose(d[band], h, rtol=1e-9, atol=0)
    for f in fixed:
        assert new[f] == lines[f]                                             # fixed lines untouched
    assert all(new[q] == (h, h) for q in xs if 1e-6 - m * h - 1e-15 <= q <= 1e-6 + m * h + 1e-15)
    # the old mid line's 1 nm spacing is gone; nothing else changed outside the band
    assert len(new) == len(lines) + 2 * math.ceil(w / h)


def test_b2_non_fixed_lines_inside_are_dropped_fixed_ones_clip_the_band():
    lines = {0.0: (5e-9, 5e-9), 10e-9: (1e-9, 1e-9), 47.3e-9: (2e-9, 2e-9), 52.6e-9: (0.5e-9, 0.5e-9),
             100e-9: (5e-9, 5e-9)}
    fixed = {0.0, 10e-9, 100e-9}
    xj, h, w = 30e-9, 1e-9, 40e-9
    new = JM.band_lines(lines, fixed, [(xj, w, h)])
    assert 47.3e-9 not in new and 52.6e-9 not in new                          # non-fixed, inside: dropped
    xs, d = _edges(new)
    assert new[10e-9] == lines[10e-9]                                         # fixed: kept, clips the left end
    left = d[(xs[:-1] >= 10e-9 - 1e-18) & (xs[1:] <= xj + 1e-18)]
    assert left.size == 20 and np.allclose(left, h, rtol=1e-9, atol=0)        # 20 nm / 1 nm: no remainder
    right = d[(xs[:-1] >= xj - 1e-18) & (xs[1:] <= xj + w + 1e-18)]
    assert right.size == 40 and np.allclose(right, h, rtol=1e-9, atol=0)
    # remainder at a fixed clip line: < h/4 is merged into the last cell, >= h/4 is kept as one short edge
    for f, want in ((10.9e-9, 1.1e-9), (10.4e-9, 0.6e-9)):                  # 0.1 h: merged; 0.6 h: kept
        lf = dict(lines)
        lf.pop(10e-9)
        lf[f] = (1e-9, 1e-9)
        xs2, d2 = _edges(JM.band_lines(lf, {0.0, f, 100e-9}, [(xj, w, h)]))
        k = int(np.searchsorted(xs2, f))
        assert d2[k] == pytest.approx(want, rel=1e-6)
        assert np.allclose(d2[k + 1:k + 5], h, rtol=1e-9, atol=0)


def test_b3_two_overlapping_junctions_are_one_cluster_with_both_as_nodes():
    lines = {0.0: (10e-9, 10e-9), 200e-9: (10e-9, 10e-9)}
    j1, j2, h, w = 80e-9, 93.3e-9, 1e-9, 20e-9                               # 13.3 nm apart, bands overlap
    new = JM.band_lines(lines, {0.0, 200e-9}, [(j2, w, h), (j1, w, h)])
    xs, d = _edges(new)
    assert j1 in new and j2 in new
    inner = d[(xs[:-1] >= j1 - 1e-18) & (xs[1:] <= j2 + 1e-18)]
    assert inner.size == 14 and np.allclose(inner, (j2 - j1) / 14, rtol=1e-9, atol=0)   # ceil(13.3) cells
    s = (j2 - j1) / 14
    outl = d[(xs[:-1] >= j1 - w - s) & (xs[1:] <= j1 + 1e-18)]
    assert np.allclose(outl, s, rtol=1e-9, atol=0)                            # the ends continue that spacing
    assert new[j1] == (s, s) and new[j2] == (s, s)
    # well-separated junctions -> two independent bands, each with spacing exactly h
    new2 = JM.band_lines(lines, {0.0, 200e-9}, [(40e-9, 10e-9, h), (150e-9, 10e-9, h)])
    xs2, d2 = _edges(new2)
    for j in (40e-9, 150e-9):
        k = int(np.argmin(np.abs(xs2 - j)))
        assert xs2[k] == j and d2[k - 1] == pytest.approx(h, rel=1e-9) and d2[k] == pytest.approx(h, rel=1e-9)


def test_b4_duplicates_and_no_junction():
    lines = {0.0: (10e-9, 10e-9), 100e-9: (10e-9, 10e-9)}
    a = JM.band_lines(lines, {0.0, 100e-9}, [(50e-9, 10e-9, 1e-9)])
    b = JM.band_lines(lines, {0.0, 100e-9}, [(50e-9, 10e-9, 1e-9), (50e-9 + 1e-16, 10e-9, 1e-9)])
    assert a == b                                                             # the same junction seen twice
    assert JM.band_lines(lines, {0.0, 100e-9}, []) == lines
    with pytest.raises(ValueError, match="outside the domain"):
        JM.band_lines(lines, {0.0, 100e-9}, [(150e-9, 10e-9, 1e-9)])


def test_b5_junction_on_a_fixed_line_bands_both_sides():
    """A z-stacked junction IS a layer interface (a fixed line): the band runs into both layers, clipped by
    the next interfaces, and the interface keeps its position."""
    lines = {0.0: (0.5e-9, 0.5e-9), 0.8e-9: (4e-9, 4e-9), 200e-9: (4e-9, 4e-9), 399.2e-9: (4e-9, 4e-9),
             400e-9: (0.5e-9, 0.5e-9), 400.8e-9: (4e-9, 4e-9), 600e-9: (4e-9, 4e-9), 799.2e-9: (4e-9, 4e-9),
             800e-9: (0.5e-9, 0.5e-9)}
    fixed = {0.0, 400e-9, 800e-9}
    new = JM.band_lines(lines, fixed, [(400e-9, 190e-9, 0.5e-9)])
    xs, d = _edges(new)
    band = (xs[:-1] >= 210e-9 - 1e-15) & (xs[1:] <= 590e-9 + 1e-15)
    assert np.allclose(d[band], 0.5e-9, rtol=1e-9, atol=0) and d[band].size == 760
    assert 399.2e-9 not in new and 400.8e-9 not in new and 0.8e-9 in new and 799.2e-9 in new
    assert new[400e-9] == (0.5e-9, 0.5e-9)


# -------------------------------------------------------------------------------------------- sign_changes
def test_z1_sign_changes():
    x = np.linspace(0.0, 10.0, 11)
    v = np.where(x < 4.5, -3.0, 2.0)
    assert JM.sign_changes(x, v) == [(4.0, 5.0, -1.0, 3.0, 2.0)]
    v2 = np.array([-1, -1, -1, 0, 0, 2, 2, 2, 2, 2, 2], dtype=float)           # undoped sliver: skipped
    assert JM.sign_changes(x, v2) == [(2.0, 5.0, -1.0, 1.0, 2.0)]
    v3 = np.array([5, 5, 5, -1, -1, -1, -1, 4, 4, 4, 4], dtype=float)          # n-p-n
    assert JM.sign_changes(x, v3) == [(2.0, 3.0, 1.0, 5.0, 1.0), (6.0, 7.0, -1.0, 1.0, 4.0)]
    g = np.array([-9, -4, -1, 1, 2, 3, 5, 8, 9, 9, 9], dtype=float)            # graded: medians
    assert JM.sign_changes(x, g) == [(2.0, 3.0, -1.0, 4.0, 6.5)]             # median of 1 2 3 5 8 9 9 9
    assert JM.sign_changes(x, np.ones(11)) == []
