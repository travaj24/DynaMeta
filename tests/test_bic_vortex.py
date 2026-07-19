"""Gates for the far-field polarization-vortex / topological-charge tooling (roadmap 1.4).

Physics + convention: Zhen, Hsu, Lu, Stone, Soljacic, "Topological Nature of Optical Bound
States in the Continuum", Phys. Rev. Lett. 113, 257401 (2014). The charge q = (1/2pi) closed
integral d phi is quantized in HALF-integers because phi (the far-field ellipse orientation)
lives on RP1 (phi == phi + pi); we wind the DOUBLED angle 2 phi (a genuine S1 field) and report
q = N/2 with N the integer 2-phi winding. Symmetry-protected BIC vortices carry integer q.

The synthetic gates (built from KNOWN phi = q * atan2(ky, kx)) pin the algorithm exactly; the
lumenairy-guarded test exercises the real conical-RCWA zeroth-order Jones surface end to end.
"""
import importlib.util

import numpy as np
import pytest

from dynameta.optics.bic import (charge_map, contour_winding, find_vortex_candidates,
                                  polarization_angle_field, rectangle_contour,
                                  stokes_parameters, topological_charge)

HAVE_LUM = importlib.util.find_spec("lumenairy") is not None


# ------------------------------------------------------------------------------------------------
# Synthetic-field builders
# ------------------------------------------------------------------------------------------------
def _grid(n=41, half=1.0):
    """(kx, ky) meshgrid with axis 0 = kx, axis 1 = ky, origin ON the centre cell (odd n)."""
    ax = np.linspace(-half, half, n)
    KX, KY = np.meshgrid(ax, ax, indexing="ij")
    return ax, KX, KY


def _linear_jones(phi):
    """Linearly polarized Jones field (cos phi, sin phi) with major-axis angle phi (real)."""
    return np.stack([np.cos(phi), np.sin(phi)], axis=-1).astype(complex)


def _vortex_jones(q, n=41, half=1.0, const=0.0, taper=False):
    """Jones field of a single charge-q polarization vortex at the origin, phi = q*atan2 + const.
    taper=True multiplies by the radial amplitude r (radiation vanishes at the centre: a V-point),
    so |E| has a detectable minimum at the vortex cell."""
    ax, KX, KY = _grid(n, half)
    phi = q * np.arctan2(KY, KX) + const
    J = _linear_jones(phi)
    if taper:
        r = np.hypot(KX, KY)
        J = J * r[..., None]
    return ax, J


# ------------------------------------------------------------------------------------------------
# (a) orientation field + mod-pi round-trip
# ------------------------------------------------------------------------------------------------
def test_polarization_angle_roundtrip_mod_pi():
    # a linear Jones (cos a, sin a) must return a mod pi, in (-pi/2, pi/2]
    a = np.array([0.0, 0.3, np.pi / 2 - 0.01, np.pi / 2 + 0.2, np.pi - 0.1, 1.9 * np.pi])
    phi = polarization_angle_field(_linear_jones(a))
    assert np.all(phi > -np.pi / 2 - 1e-12) and np.all(phi <= np.pi / 2 + 1e-9)
    # phi == a modulo pi
    diff = (phi - a) % np.pi
    diff = np.minimum(diff, np.pi - diff)
    assert np.max(diff) < 1e-9


def test_stokes_linear_polarization_degree():
    # a purely linear field has |S3| = 0 and S0^2 == S1^2 + S2^2
    _, J = _vortex_jones(1, n=21)
    s0, s1, s2, s3 = stokes_parameters(J)
    assert np.max(np.abs(s3)) < 1e-12
    assert np.allclose(s0 ** 2, s1 ** 2 + s2 ** 2, atol=1e-12)


# ------------------------------------------------------------------------------------------------
# (b) topological charge: exact recovery for q = +1, -1, +2, -2
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1, -1, 2, -2])
def test_charge_recovered_exactly(q):
    _, J = _vortex_jones(q, n=61, half=1.0, const=0.37)
    phi = polarization_angle_field(J)
    # a large CCW rectangle around the single centred vortex
    contour = (5, 55, 5, 55)
    n_raw = contour_winding(phi, contour)           # doubled-angle winding N
    assert abs(n_raw - 2 * q) < 1e-6                 # cleanly topological before rounding
    assert topological_charge(phi, contour) == float(q)


def test_charge_sign_and_orientation():
    # +q gives +q, mirrored axis flips the sign (CCW convention)
    _, Jp = _vortex_jones(1, n=61)
    phi = polarization_angle_field(Jp)
    assert topological_charge(phi, (5, 55, 5, 55)) == 1.0
    # reverse ky axis -> the winding reverses
    phi_rev = phi[:, ::-1]
    assert topological_charge(phi_rev, (5, 55, 5, 55)) == -1.0


# ------------------------------------------------------------------------------------------------
# charge invariance under contour deformation (3 sizes)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1, -1, 2, -2])
def test_charge_invariant_under_contour_deformation(q):
    _, J = _vortex_jones(q, n=81)
    phi = polarization_angle_field(J)
    charges = [topological_charge(phi, (40 - s, 40 + s, 40 - s, 40 + s)) for s in (8, 20, 35)]
    assert charges == [float(q)] * 3


# ------------------------------------------------------------------------------------------------
# two-vortex field: contour around both = sum; around each = individual
# ------------------------------------------------------------------------------------------------
def test_two_vortex_sum_of_charges():
    n = 81
    ax = np.linspace(-1.0, 1.0, n)
    KX, KY = np.meshgrid(ax, ax, indexing="ij")
    # two vortices offset along kx: q1 = +1 at (-0.4, 0), q2 = +1 at (+0.4, 0)
    c1, c2 = -0.4, 0.4
    phi = (1 * np.arctan2(KY, KX - c1) + 1 * np.arctan2(KY, KX - c2))
    J = _linear_jones(phi)
    phi_f = polarization_angle_field(J)
    # index of the two centres
    i1 = int(np.argmin(np.abs(ax - c1)))
    i2 = int(np.argmin(np.abs(ax - c2)))
    jc = int(np.argmin(np.abs(ax - 0.0)))
    # small loop around each -> +1 each
    assert topological_charge(phi_f, (i1 - 6, i1 + 6, jc - 6, jc + 6)) == 1.0
    assert topological_charge(phi_f, (i2 - 6, i2 + 6, jc - 6, jc + 6)) == 1.0
    # big loop around BOTH -> +2
    assert topological_charge(phi_f, (6, n - 7, 6, n - 7)) == 2.0


def test_opposite_two_vortex_net_zero():
    n = 81
    ax = np.linspace(-1.0, 1.0, n)
    KX, KY = np.meshgrid(ax, ax, indexing="ij")
    phi = (1 * np.arctan2(KY, KX + 0.4) + (-1) * np.arctan2(KY, KX - 0.4))
    phi_f = polarization_angle_field(_linear_jones(phi))
    # loop enclosing a +1 and a -1 -> net 0
    assert topological_charge(phi_f, (6, n - 7, 6, n - 7)) == 0.0


# ------------------------------------------------------------------------------------------------
# vortex-free field = 0
# ------------------------------------------------------------------------------------------------
def test_vortex_free_field_zero_charge():
    ax, KX, KY = _grid(41, 1.0)
    # a smooth, non-singular orientation field
    phi = 0.2 * KX + 0.1 * KY + 0.3
    phi_f = polarization_angle_field(_linear_jones(phi))
    assert topological_charge(phi_f, (5, 35, 5, 35)) == 0.0
    # a globally constant polarization
    Jc = np.broadcast_to(np.array([1.0 + 0j, 0.4 + 0j]), (41, 41, 2)).copy()
    assert topological_charge(polarization_angle_field(Jc), (5, 35, 5, 35)) == 0.0


# ------------------------------------------------------------------------------------------------
# noise robustness: 1% Jones noise, charge unchanged
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1, -1, 2, -2])
def test_charge_robust_to_jones_noise(q):
    rng = np.random.default_rng(1234 + q)
    _, J = _vortex_jones(q, n=81, half=1.0)
    scale = 0.01 * np.sqrt(np.mean(np.abs(J) ** 2))
    noise = scale * (rng.standard_normal(J.shape) + 1j * rng.standard_normal(J.shape))
    phi = polarization_angle_field(J + noise)
    # a comfortably large contour: the integer winding is topologically protected
    assert topological_charge(phi, (10, 70, 10, 70)) == float(q)


# ------------------------------------------------------------------------------------------------
# (c) C-point / V-point detector localizes the synthetic centre to the grid cell
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1, -1, 2, -2])
def test_find_vortex_localizes_centre_cell(q):
    n = 61
    _, J = _vortex_jones(q, n=n, half=1.0, taper=True)   # radiation vanishes at centre (V-point)
    centre = n // 2
    cands = find_vortex_candidates(J, contour_radius=3, max_candidates=3)
    assert len(cands) >= 1
    # the strongest singularity is the centre cell
    assert cands[0]["index"] == (centre, centre)
    assert cands[0]["charge"] == float(q)


def test_find_vortex_none_on_smooth_field():
    ax, KX, KY = _grid(41, 1.0)
    phi = 0.2 * KX + 0.1 * KY
    J = _linear_jones(phi)                                # |E| = 1 everywhere, no singularity
    assert find_vortex_candidates(J) == []


# ------------------------------------------------------------------------------------------------
# (d) charge_map: local charge on small loops localizes the vortex
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", [1, -1, 2, -2])
def test_charge_map_localizes_and_signs(q):
    n = 41
    _, J = _vortex_jones(q, n=n, half=1.0)
    phi = polarization_angle_field(J)
    r = 2
    qmap = charge_map(phi, radius=r)                     # 16-pt loops resolve |q| <= 3
    centre = n // 2
    # the loop CENTRED on the vortex reads the full charge q (vortex well inside the loop)
    assert qmap[centre, centre] == float(q)
    # a loop far from the centre encloses nothing -> 0
    assert qmap[5, 5] == 0.0
    # LOCALIZATION: the nonzero-charge loops form a compact block CENTRED on the vortex (only
    # loops that enclose or touch the single vortex read nonzero; faster |q|=2 winding spreads
    # the touched region by one extra cell, so bound the extent by 2*(r+1))
    ii, jj = np.where(np.nan_to_num(qmap) != 0.0)
    assert round(float(ii.mean())) == centre and round(float(jj.mean())) == centre
    assert (ii.max() - ii.min()) <= 2 * (r + 1) and (jj.max() - jj.min()) <= 2 * (r + 1)


def test_charge_map_two_vortices_distinct_blocks():
    n = 61
    ax = np.linspace(-1.0, 1.0, n)
    KX, KY = np.meshgrid(ax, ax, indexing="ij")
    phi = (1 * np.arctan2(KY, KX + 0.45) + (-1) * np.arctan2(KY, KX - 0.45))
    qmap = charge_map(polarization_angle_field(_linear_jones(phi)), radius=1)
    i_pos = int(np.argmin(np.abs(ax + 0.45)))            # +1 vortex column
    i_neg = int(np.argmin(np.abs(ax - 0.45)))            # -1 vortex column
    jc = int(np.argmin(np.abs(ax - 0.0)))
    assert qmap[i_pos, jc] == 1.0
    assert qmap[i_neg, jc] == -1.0


def test_rectangle_contour_is_closed_ccw():
    idx = rectangle_contour(2, 6, 3, 8)
    assert idx.shape[1] == 2
    # perimeter length = 2*((6-2)+(8-3)) = 18 unique boundary points
    assert idx.shape[0] == 2 * ((6 - 2) + (8 - 3))
    # no repeated closing vertex
    assert not np.array_equal(idx[0], idx[-1])


# ------------------------------------------------------------------------------------------------
# INTEGRATION (guarded): real conical-RCWA zeroth-order Jones surface -> tool runs end to end
# ------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_LUM, reason="lumenairy not installed")
def test_lumenairy_conical_jones_field_end_to_end():
    """Exercise the bridge's conical Jones surface (RCWAStack.jones_reflection over a (kx, ky)
    grid) end to end: build the far-field Jones field, run the full tool, and check it yields a
    well-defined orientation field and a quantized (here trivial) charge. A symmetric uniform
    2-D slab has NO polarization vortex, so every contour charge must be 0 and no spurious
    candidate should be found -- a deterministic real-field zero gate. Recovering a genuine
    symmetry-protected BIC vortex from a designed high-Q slab is the documented stretch goal."""
    import lumenairy

    lam = 1.30e-6
    n_super = 1.0
    k0 = 2.0 * np.pi / lam
    px = py = 500e-9
    ax = np.linspace(-0.05, 0.05, 9) * k0                 # small conical k-window, avoid Gamma-deg.
    J = np.zeros((ax.size, ax.size, 2), dtype=complex)
    for i, kx in enumerate(ax):
        for j, ky in enumerate(ax):
            kpar = np.hypot(kx, ky)
            theta = float(np.arcsin(np.clip(kpar / (k0 * n_super), -1.0, 1.0)))
            phi_inc = float(np.arctan2(ky, kx))
            st = lumenairy.RCWAStack(px, period_y=py, n_superstrate=complex(n_super),
                                     n_substrate=1.5 + 0j, n_orders=2, n_orders_y=2)
            st.add_layer(220e-9, eps=complex(3.5))        # uniform slab (fast, symmetric)
            st.set_source(lam, theta=theta, phi=phi_inc)
            res = st.solve()
            Jr = np.asarray(res.jones_reflection())        # (2, 2), columns = incident Ex/Ey
            J[i, j, :] = Jr[:, 0]                           # far-field response to incident x

    phi = polarization_angle_field(J)
    assert np.all(np.isfinite(phi))
    assert np.all(phi > -np.pi / 2 - 1e-9) and np.all(phi <= np.pi / 2 + 1e-9)
    q = topological_charge(phi, (1, ax.size - 2, 1, ax.size - 2))
    assert q == 0.0                                        # uniform symmetric slab: no vortex
    assert q == round(q * 2) / 2.0                         # a valid half-integer
    assert find_vortex_candidates(J) == []                 # no spurious singularity
