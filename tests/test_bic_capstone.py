"""Fast pytest subset of the real-BIC end-to-end capstone (roadmap item 5.6).

The runnable design study with the full-resolution PASS/FAIL gates lives in
validation/bic_capstone.py (run: python -m validation.bic_capstone). This module re-exercises
the THREE BIC signatures + the internal-consistency check on the SAME landed design at REDUCED
resolution (coarser adaptive sweeps, a coarser conical k-grid), so the whole file runs in a few
seconds on the live lumenairy RCWA bridge.

Landed design (see the validation module docstring): a 700 nm-pitch, 50%-fill, 500 nm-thick
Si-like (n = 3.48) grating suspended as a symmetric vacuum membrane; the TE/s second-band
symmetry-protected BIC sits at Gamma near 1387 nm.

Signatures (Koshelev PRL 121:193903 (2018); Hsu/Zhen Nature 499:188 (2013); Zhen PRL
113:257401 (2014)):
  (1) DRIVEN-SPECTRUM (Fano): resonance absent at theta = 0, Q ~ theta^-2 over theta = 1..4 deg.
  (2) POLE (AAA): Im(omega_tilde) -> 0 as theta -> 0, |Im| ~ theta^2, Q diverges toward Gamma.
  (3) VORTEX: the conical Jones map around Gamma carries topological charge |q| = 1.
  (4) CONSISTENCY: the Fano and pole resonance frequencies agree to << 1% at every common angle.

Guarded by importorskip -- lumenairy is a required dynameta dependency but the fast CI legs that
lack it skip cleanly. Every leg is well under the ~120 s slow-marker threshold (measured ~7 s
total), so no @pytest.mark.slow is needed here (contrast tests/test_soa.py, whose full validation
wrappers ARE marked slow).
"""
import math

import numpy as np
import pytest

pytest.importorskip("lumenairy")

from dynameta.analysis import quasi_bic_scaling                       # noqa: E402
from dynameta.constants import C_LIGHT                                # noqa: E402
from validation.bic_capstone import (DESIGN, run_angle_sweeps, vortex_charge,     # noqa: E402
                                     window_contrast)

THETAS = [1.0, 2.0, 3.0, 4.0]
N_ORDERS = 12


@pytest.fixture(scope="module")
def sweeps():
    """One reduced-resolution adaptive complex-r sweep per angle (shared by the Fano, pole and
    consistency gates). Every angle must yield a BIC pole."""
    res = run_angle_sweeps(DESIGN, THETAS, n_orders=N_ORDERS, n_initial=81, max_samples=193)
    assert all(r is not None for r in res), "pole extraction failed at some angle"
    return res


# ------------------------------------------------------------------------------------------------
# Signature 1: driven-spectrum Fano Q ~ theta^-2 + theta=0 absence
# ------------------------------------------------------------------------------------------------
def test_driven_spectrum_fano_scaling(sweeps):
    fano_Q = [r.fano_Q for r in sweeps]
    exponent, _pref, r2 = quasi_bic_scaling(THETAS, fano_Q)
    assert abs(exponent + 2.0) <= 0.3, "Fano Q ~ theta^{:.3f}, expected -2".format(exponent)
    assert r2 > 0.98, "poor power-law fit (r2={:.4f})".format(r2)
    # every angle's Fano Q tracks the AAA pole Q within a few % (same resonance, two instruments)
    for r in sweeps:
        assert abs(r.fano_Q - r.pole_Q) <= 0.05 * r.pole_Q


def test_theta0_resonance_absent(sweeps):
    # over a window matched to the theta = 1 deg resonance the symmetry-protected mode is ABSENT
    # at normal incidence: the contrast collapses to a smooth background far below the oblique case
    w0 = sweeps[0].pole_omega0
    gamma1 = 2.0 * sweeps[0].pole_Im
    contrast0 = window_contrast(DESIGN, 0.0, w0, gamma1, n_orders=N_ORDERS, n_pts=81)
    contrast1 = window_contrast(DESIGN, 1.0, w0, gamma1, n_orders=N_ORDERS, n_pts=81)
    assert contrast0 < 0.10
    assert contrast0 < 0.2 * contrast1


# ------------------------------------------------------------------------------------------------
# Signature 2: AAA pole tracking -- Im ~ theta^2, Q divergence toward Gamma
# ------------------------------------------------------------------------------------------------
def test_pole_tracking_scaling(sweeps):
    pole_Im = [r.pole_Im for r in sweeps]
    pole_Q = [r.pole_Q for r in sweeps]
    # decaying poles (Im < 0 under exp(-i omega t)); Im magnitude collapses toward Gamma, monotone
    assert all(pole_Im[i] < pole_Im[i + 1] for i in range(len(pole_Im) - 1))
    im_exp = float(np.polyfit(np.log(THETAS), np.log(pole_Im), 1)[0])
    assert abs(im_exp - 2.0) <= 0.4, "|Im| ~ theta^{:.3f}, expected 2".format(im_exp)
    # Q diverges as theta -> 0: the smallest-angle Q exceeds 10x the 4 deg Q
    assert pole_Q[0] > 10.0 * pole_Q[-1]


# ------------------------------------------------------------------------------------------------
# Signature 3: far-field polarization vortex, |q| = 1 (coarse conical k-grid)
# ------------------------------------------------------------------------------------------------
def test_vortex_charge_unity():
    # coarse 13x13 (kx, ky) grid around Gamma at the BIC wavelength; a radius-4 contour is
    # Nyquist-safe for |q| = 1 and does not hug the vortex core, so the undersampling guard
    # must NOT fire and the quantized charge is exactly +/-1
    q, n_raw, guard = vortex_charge(DESIGN, n_grid=13, k_frac=0.03, contour_radius=4,
                                    n_orders=N_ORDERS)
    assert not guard, "bic undersampling guard fired on the chosen k-grid/contour"
    assert abs(abs(q) - 1.0) < 1e-9, "topological charge |q|={} (expected 1)".format(abs(q))
    assert abs(round(n_raw) - n_raw) < 0.15, "raw 2-phi winding not cleanly integer (N={:.3f})".format(n_raw)


# ------------------------------------------------------------------------------------------------
# Signature 4: the three instruments agree on the resonance frequency
# ------------------------------------------------------------------------------------------------
def test_instrument_consistency(sweeps):
    # Fano vs AAA-pole resonance frequency agree to << 1% at every common angle
    for r in sweeps:
        assert abs(r.fano_omega0 - r.pole_omega0) <= 0.01 * r.pole_omega0
    # the resonance quadratically extrapolated to Gamma matches the vortex-map BIC wavelength
    theta2 = np.array(THETAS) ** 2
    omega0 = np.array([r.pole_omega0 for r in sweeps])
    omega0_gamma = float(np.polyfit(theta2, omega0, 1)[1])
    omega_vtx = 2.0 * math.pi * C_LIGHT / DESIGN.lam_bic_m
    assert abs(omega0_gamma - omega_vtx) <= 0.01 * omega_vtx
