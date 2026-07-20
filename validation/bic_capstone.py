"""Real-BIC end-to-end capstone (roadmap item 5.6): design an actual symmetry-protected
bound state in the continuum (BIC) in a 1-D high-index grating slab and close the loop with the
ENTIRE Phase-1 + Phase-5 resonance-instrument stack (Fano tooling, AAA rational pole extraction,
and the far-field polarization-vortex topological charge) on the LIVE lumenairy RCWA bridge.

-------------------------------------------------------------------------------------------------
PHYSICS
-------------------------------------------------------------------------------------------------
A 1-D-periodic slab of high-index bars (here Si-like n = 3.48) suspended as a symmetric membrane
in vacuum supports, at the Gamma point (exactly normal incidence, k_par = 0) of its second
(folded) leaky TE band, a SYMMETRY-PROTECTED BIC: the leaky resonance is ODD under the in-plane
mirror sigma_x (x -> -x) of the grating, while the even zeroth-order radiation continuum (the
specular plane wave) is EVEN, so by symmetry the mode CANNOT radiate -- its radiative Q is
infinite and it is a truly bound state embedded in the continuum. Tilting the incidence by an
angle theta in the plane of periodicity breaks the sigma_x symmetry LINEARLY in the in-plane
momentum k_x ~ (omega/c) sin(theta), so the radiative coupling amplitude ~ theta and the
radiative decay rate gamma_rad ~ theta^2. The resonance therefore reappears as a QUASI-BIC whose
quality factor DIVERGES as

        Q ~ 1 / theta^2          (the canonical symmetry-protected quasi-BIC law).

References: Koshelev, Lepeshov, Liu, Bogdanov, Kivshar, "Asymmetric Metasurfaces with High-Q
Resonances Governed by Bound States in the Continuum", Phys. Rev. Lett. 121, 193903 (2018) (the
Q ~ 1/delta^2 scaling); Hsu, Zhen, Lee, Chua, Johnson, Joannopoulos, Soljacic, "Observation of
trapped light within the radiation continuum", Nature 499, 188 (2013) (the grating-slab design
class); Zhen, Hsu, Lu, Stone, Soljacic, "Topological Nature of Optical Bound States in the
Continuum", Phys. Rev. Lett. 113, 257401 (2014) (the polarization-vortex topological charge).

-------------------------------------------------------------------------------------------------
THE LANDED DESIGN (located by the coarse wavelength x angle scan in GATE 0)
-------------------------------------------------------------------------------------------------
    period p            = 700 nm            (1-D grating pitch, x)
    fill factor f       = 0.50              (high-index fraction of the period)
    bar thickness t     = 500 nm
    n_hi (bars)         = 3.48              (Si-like, lossless)
    n_lo (grooves)      = 1.00              (air)
    superstrate/substrate = 1.00 / 1.00     (symmetric suspended membrane)
    polarization        = TE / s            (incident E along the bars, lumenairy row 1 = E_y)
    band                = 2nd (folded) leaky TE band
    BIC wavelength      ~ 1387 nm           (the Gamma-point resonance, k_par -> 0)

Below the first diffraction threshold (lambda / p = 1.98 > 1, both half-spaces n = 1) ONLY the
zeroth order propagates, so the order-summed reflectance equals the co-polarized |r|^2 and the
zeroth-order complex amplitude r = jones_reflection()[1, 1] is the clean analytic observable the
pole finder consumes.

-------------------------------------------------------------------------------------------------
THE THREE SIGNATURES (the gates)
-------------------------------------------------------------------------------------------------
GATE 0  DESIGN LOCATION: a coarse wavelength x angle RCWA scan locates the BIC -- the resonance
        is ABSENT from the normal-incidence spectrum and APPEARS + SHARPENS (Q grows) as theta
        shrinks toward Gamma.

GATE 1  DRIVEN-SPECTRUM (Fano) SIGNATURE: at theta = 0 the resonance feature is absent from the
        zeroth-order spectrum (contrast << the oblique case); at theta = 1, 2, 3, 4 deg a Fano
        feature appears whose fano_fit Q follows Q ~ theta^-2 -- quasi_bic_scaling exponent
        = -2.0 +/- 0.3 over the four angles (analysis.fano_fit / analysis.quasi_bic_scaling).

GATE 2  POLE SIGNATURE: AAA rational pole extraction (aaa_poles.sweep_and_extract on the complex
        zeroth-order amplitude r(omega)) tracks the pole vs theta; Im(omega_tilde) -> 0 as
        theta -> 0 with |Im| ~ theta^2 (exponent 2.0 +/- 0.4), and Q at the smallest angle
        exceeds 10x the Q at 4 deg.

GATE 3  VORTEX SIGNATURE: the conical (k_x, k_y) Jones map around Gamma at the BIC wavelength
        carries topological charge |q| = 1 via bic.topological_charge (the far-field polarization
        director winds once around the V-point); the undersampling guard does NOT fire on the
        chosen k-grid / contour radius.

GATE 4  INTERNAL CONSISTENCY: the three instruments' resonance frequencies agree to ~1% at the
        common angles (Fano vs pole per angle; the Gamma-extrapolated resonance vs the vortex-map
        wavelength).

Conventions: SI units, exp(-i omega t) (decaying poles have Im(omega_tilde) < 0), lossless
(real-index) structure so every Q is purely radiative. lumenairy 5.25 is the live RCWA engine.

Run: python -m validation.bic_capstone
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import C_LIGHT
from dynameta.analysis import fano_fit, quasi_bic_scaling
from dynameta.optics.aaa_poles import sweep_and_extract
from dynameta.optics.bic import polarization_angle_field, topological_charge, contour_winding


# ------------------------------------------------------------------------------------------------
# The landed design
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BICDesign:
    """The symmetry-protected-BIC grating design (SI units).  `row` is the lumenairy incident
    polarization row (1 = E_y = TE/s, along the bars).  `band_lo_m` / `band_hi_m` bracket the
    Gamma-point resonance for the resonance-centered sweeps; `lam_bic_m` is the Gamma BIC
    wavelength used for the conical vortex map."""

    period_m: float = 700e-9
    fill: float = 0.50
    thickness_m: float = 500e-9
    n_hi: float = 3.48
    n_lo: float = 1.0
    n_super: float = 1.0
    n_sub: float = 1.0
    row: int = 1
    band_lo_m: float = 1.383e-6
    band_hi_m: float = 1.393e-6
    lam_bic_m: float = 1.3870e-6


DESIGN = BICDesign()


# ------------------------------------------------------------------------------------------------
# lumenairy RCWA forward (built on the same live bridge surface as tests/test_bic_vortex.py and
# tests/test_aaa_poles.py gate 5)
# ------------------------------------------------------------------------------------------------
def build_stack(design: BICDesign, n_orders: int):
    """A concrete 1-D binary-grating RCWAStack for the design at the given truncation order.
    The eps cell is a lamellar high/low profile sampled at the lumenairy bound Sx = 4*n_orders+1
    (the binary-grating recipe of tests/test_aaa_poles.py's GMR gate)."""
    import lumenairy

    sx = 4 * int(n_orders) + 1
    cell = np.full(sx, complex(design.n_lo) ** 2, dtype=complex)
    cell[: int(round(design.fill * sx))] = complex(design.n_hi) ** 2
    st = lumenairy.RCWAStack(design.period_m, n_superstrate=complex(design.n_super),
                             n_substrate=complex(design.n_sub), n_orders=int(n_orders))
    st.add_layer(design.thickness_m, eps_cell=cell, formulation="laurent")
    return st


def solve_at(design: BICDesign, lam_m: float, theta_deg: float, phi_deg: float, n_orders: int):
    """Solve once and return the RCWAResult (superstrate-side incidence)."""
    st = build_stack(design, n_orders)
    st.set_source(float(lam_m), theta=math.radians(theta_deg), phi=math.radians(phi_deg))
    return st.solve()


def r_of_omega(design: BICDesign, omega: float, theta_deg: float, n_orders: int) -> complex:
    """Complex zeroth-order co-polarized reflection amplitude r(omega) at in-plane tilt theta
    (phi = 0).  This is the analytic observable AAA reads the pole off (jones_reflection is the
    lab-basis (2, 2); at phi = 0 the TE row stays pure TE so r = jones_r[row, row])."""
    lam = 2.0 * math.pi * C_LIGHT / float(omega)
    res = solve_at(design, lam, theta_deg, 0.0, n_orders)
    jr = np.asarray(res.jones_reflection())
    return complex(jr[design.row, design.row])


def R_of_omega(design: BICDesign, omega: float, theta_deg: float, n_orders: int) -> float:
    """Order-summed reflectance R(omega) for the incident-row polarization at tilt theta (phi=0).
    Below the diffraction threshold this equals |r_of_omega|^2, but it is the independent
    Poynting quantity used for the design-location scan."""
    lam = 2.0 * math.pi * C_LIGHT / float(omega)
    res = solve_at(design, lam, theta_deg, 0.0, n_orders)
    _o, Reff, _T = res.efficiencies()
    return float(Reff[design.row].sum())


# ------------------------------------------------------------------------------------------------
# Shared resonance sweep: ONE adaptive complex-r sweep per angle feeds BOTH the pole (GATE 2) and
# the Fano (GATE 1) instruments (the zeroth-order amplitude and its modulus-squared spectrum).
# ------------------------------------------------------------------------------------------------
@dataclass
class AngleResult:
    theta_deg: float
    pole_omega0: float
    pole_Q: float
    pole_Im: float
    fano_omega0: float
    fano_Q: float
    n_samples: int


def _select_bic_pole(resonances, q_floor: float = 50.0):
    """The BIC is the SHARPEST (highest-Q) physical resonance in the band; broad background poles
    and any Froissart survivors are low-Q.  Prefer Q > q_floor, else fall back to the max-Q
    resonance, else None (nothing found)."""
    if not resonances:
        return None
    sharp = [r for r in resonances if r.Q > q_floor]
    pool = sharp if sharp else list(resonances)
    return max(pool, key=lambda r: r.Q)


def sweep_angle(design: BICDesign, theta_deg: float, *, n_orders: int, n_initial: int,
                max_samples: int, tol: float = 1e-9) -> Optional[AngleResult]:
    """Adaptively sweep the complex r(omega) across the resonance band at one tilt angle, extract
    the BIC pole (AAA) and the Fano fit of |r|^2 (same samples)."""
    w_lo = 2.0 * math.pi * C_LIGHT / design.band_hi_m
    w_hi = 2.0 * math.pi * C_LIGHT / design.band_lo_m
    sw = sweep_and_extract(lambda w: r_of_omega(design, w, theta_deg, n_orders), w_lo, w_hi,
                           n_initial=int(n_initial), max_samples=int(max_samples), tol=tol)
    pole = _select_bic_pole(sw.resonances)
    if pole is None:
        return None
    R = np.abs(sw.response) ** 2
    ff = fano_fit(sw.omega, R)
    return AngleResult(theta_deg=theta_deg, pole_omega0=float(pole.omega_tilde.real),
                       pole_Q=float(pole.Q), pole_Im=float(abs(pole.omega_tilde.imag)),
                       fano_omega0=float(ff.omega0), fano_Q=float(ff.Q), n_samples=int(sw.omega.size))


def run_angle_sweeps(design: BICDesign, thetas: List[float], *, n_orders: int = 12,
                     n_initial: int = 121, max_samples: int = 385) -> List[AngleResult]:
    return [sweep_angle(design, t, n_orders=n_orders, n_initial=n_initial,
                        max_samples=max_samples) for t in thetas]


# ------------------------------------------------------------------------------------------------
# theta = 0 absence (matched to the theta_ref resonance window)
# ------------------------------------------------------------------------------------------------
def window_contrast(design: BICDesign, theta_deg: float, omega0: float, gamma: float, *,
                    half_widths: float = 12.0, n_pts: int = 121, n_orders: int = 12) -> float:
    """Peak-to-peak contrast of R(omega) over [omega0 - k*gamma, omega0 + k*gamma] at tilt
    theta_deg.  At theta = 0 the symmetry-protected resonance is absent, so the contrast collapses
    to the smooth-background variation over the (narrow) window."""
    w = np.linspace(omega0 - half_widths * gamma, omega0 + half_widths * gamma, int(n_pts))
    R = np.array([abs(r_of_omega(design, wi, theta_deg, n_orders)) ** 2 for wi in w])
    return float(R.max() - R.min())


# ------------------------------------------------------------------------------------------------
# Vortex map (conical (kx, ky) around Gamma at the BIC wavelength)
# ------------------------------------------------------------------------------------------------
def vortex_jones_field(design: BICDesign, *, n_grid: int, k_frac: float,
                       n_orders: int) -> np.ndarray:
    """Far-field zeroth-order Jones field J(kx, ky) on an (n_grid, n_grid) grid spanning
    +/- k_frac * k0 around Gamma at the BIC wavelength.  The polarization director at each k is
    the eigenvector of the reflection Jones matrix with the largest |eigenvalue| -- the
    resonantly-radiating channel (the module-docstring hookup in optics/bic.py).  Returns a
    (n_grid, n_grid, 2) complex array with axis 0 = kx, axis 1 = ky, (Ex, Ey) last."""
    lam = design.lam_bic_m
    k0 = 2.0 * math.pi / lam
    ax = np.linspace(-k_frac, k_frac, int(n_grid)) * k0
    J = np.zeros((int(n_grid), int(n_grid), 2), dtype=complex)
    for i, kx in enumerate(ax):
        for j, ky in enumerate(ax):
            kpar = math.hypot(kx, ky)
            theta = math.asin(min(kpar / (k0 * design.n_super), 1.0))
            phi = math.atan2(ky, kx)
            res = solve_at(design, lam, math.degrees(theta), math.degrees(phi), n_orders)
            M = np.asarray(res.jones_reflection())
            w, V = np.linalg.eig(M)
            J[i, j, :] = V[:, int(np.argmax(np.abs(w)))]
    return J


def vortex_charge(design: BICDesign, *, n_grid: int = 21, k_frac: float = 0.03,
                  contour_radius: int = 5, n_orders: int = 12) -> Tuple[float, float, bool]:
    """(charge, raw_winding, guard_fired): topological charge of the conical Jones field around
    Gamma on a rectangle contour of half-size `contour_radius` centered on the grid (= Gamma).
    guard_fired is True iff the bic undersampling guard rejected the contour (returns NaN charge)."""
    J = vortex_jones_field(design, n_grid=n_grid, k_frac=k_frac, n_orders=n_orders)
    phi = polarization_angle_field(J)
    c = int(n_grid) // 2
    r = int(contour_radius)
    try:
        n_raw = contour_winding(phi, (c - r, c + r, c - r, c + r))
        q = topological_charge(phi, (c - r, c + r, c - r, c + r))
        return float(q), float(n_raw), False
    except ValueError:
        return float("nan"), float("nan"), True


# ------------------------------------------------------------------------------------------------
# GATE 0: design-location coarse scan
# ------------------------------------------------------------------------------------------------
def gate_design_location(design: BICDesign, *, n_orders: int = 12) -> bool:
    """Locate the BIC: over the resonance band the normal-incidence (theta = 0) spectrum is
    featureless, while at oblique incidence a resonance appears and SHARPENS (Q grows) toward
    Gamma.  A coarse three-angle Fano-Q trend + the theta = 0 flatness is the location proof."""
    w_lo = 2.0 * math.pi * C_LIGHT / design.band_hi_m
    w_hi = 2.0 * math.pi * C_LIGHT / design.band_lo_m

    # normal incidence: sweep R over the band, must be smooth (no resonance dip/peak)
    w = np.linspace(w_lo, w_hi, 121)
    R0 = np.array([R_of_omega(design, wi, 0.0, n_orders) for wi in w])
    normal_contrast = float(R0.max() - R0.min())

    # oblique: the pole sharpens as theta shrinks (Q grows)
    qs = []
    for theta in (4.0, 3.0, 2.0):
        ar = sweep_angle(design, theta, n_orders=n_orders, n_initial=81, max_samples=257)
        qs.append(ar.pole_Q if ar is not None else float("nan"))
    grows = all(np.isfinite(qs)) and qs[2] > qs[1] > qs[0]
    ok = bool(normal_contrast < 0.10 and grows)
    print("[bic] GATE 0 DESIGN LOCATION: normal-incidence band contrast={:.4f} (<0.10); "
          "pole Q(4,3,2 deg)=({:.0f}, {:.0f}, {:.0f}) sharpening toward Gamma -> {}".format(
              normal_contrast, qs[0], qs[1], qs[2], "PASS" if ok else "FAIL"), flush=True)
    return ok


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main(*, n_orders: int = 12) -> bool:
    print("[bic] ===== REAL-BIC END-TO-END CAPSTONE (roadmap 5.6) =====", flush=True)
    print("[bic] design: p={:.0f}nm fill={:.2f} t={:.0f}nm n_hi={:.2f} membrane(n={:.1f}); "
          "TE/s; BIC ~{:.0f}nm".format(DESIGN.period_m * 1e9, DESIGN.fill,
          DESIGN.thickness_m * 1e9, DESIGN.n_hi, DESIGN.n_super, DESIGN.lam_bic_m * 1e9), flush=True)
    ok = True

    # -------- GATE 0: design location --------
    g0 = gate_design_location(DESIGN, n_orders=n_orders)
    ok = ok and g0

    # -------- shared angle sweeps (feed GATE 1, 2, 4) --------
    thetas = [1.0, 2.0, 3.0, 4.0]
    results = run_angle_sweeps(DESIGN, thetas, n_orders=n_orders, n_initial=121, max_samples=385)
    if any(r is None for r in results):
        print("[bic] FATAL: pole extraction failed at some angle {}".format(
            [t for t, r in zip(thetas, results) if r is None]), flush=True)
        return False
    print("[bic] per-angle Q(theta) table (theta_deg | fano Q | pole Q | Im(omega) | lam0 nm):", flush=True)
    for r in results:
        print("[bic]   {:>4.1f} | {:>9.0f} | {:>9.0f} | {:.3e} | {:.3f}".format(
            r.theta_deg, r.fano_Q, r.pole_Q, r.pole_Im,
            2 * math.pi * C_LIGHT / r.pole_omega0 * 1e9), flush=True)

    fano_Q = [r.fano_Q for r in results]
    pole_Q = [r.pole_Q for r in results]
    pole_Im = [r.pole_Im for r in results]

    # -------- GATE 1: driven-spectrum Fano Q ~ theta^-2 + theta=0 absence --------
    exp_f, _pref, r2_f = quasi_bic_scaling(thetas, fano_Q)
    w0_1 = results[0].pole_omega0
    gamma_1 = 2.0 * results[0].pole_Im
    contrast0 = window_contrast(DESIGN, 0.0, w0_1, gamma_1, n_orders=n_orders)
    contrast1 = window_contrast(DESIGN, 1.0, w0_1, gamma_1, n_orders=n_orders)
    g1 = bool(abs(exp_f + 2.0) <= 0.3 and r2_f > 0.98 and contrast0 < 0.10
              and contrast0 < 0.2 * contrast1)
    ok = ok and g1
    print("[bic] GATE 1 DRIVEN-SPECTRUM (Fano): quasi_bic_scaling exponent={:.3f} (want -2.0+-0.3) "
          "r2={:.4f}; theta=0 window contrast={:.4f} vs theta=1 contrast={:.3f} (resonance absent "
          "at normal) -> {}".format(exp_f, r2_f, contrast0, contrast1, "PASS" if g1 else "FAIL"),
          flush=True)

    # -------- GATE 2: pole |Im| ~ theta^2 + Q divergence --------
    lt = np.log(np.array(thetas))
    im_exp = float(np.polyfit(lt, np.log(np.array(pole_Im)), 1)[0])
    q_ratio = pole_Q[0] / pole_Q[-1]
    g2 = bool(abs(im_exp - 2.0) <= 0.4 and q_ratio > 10.0
              and all(pole_Im[i] < pole_Im[i + 1] for i in range(len(pole_Im) - 1)))
    ok = ok and g2
    print("[bic] GATE 2 POLE: |Im(omega)| ~ theta^{:.3f} (want 2.0+-0.4), Im -> 0 as theta -> 0 "
          "(monotone); Q(1deg)/Q(4deg)={:.1f} (>10) -> {}".format(
              im_exp, q_ratio, "PASS" if g2 else "FAIL"), flush=True)

    # -------- GATE 3: polarization-vortex charge --------
    q_vtx, n_raw, guard = vortex_charge(DESIGN, n_grid=21, k_frac=0.03, contour_radius=5,
                                        n_orders=n_orders)
    g3 = bool((not guard) and abs(abs(q_vtx) - 1.0) < 1e-9)
    ok = ok and g3
    print("[bic] GATE 3 VORTEX: conical Jones map around Gamma at {:.0f}nm -> topological charge "
          "q={:+.1f} (|q|=1; raw 2-phi winding N={:.3f}); undersampling guard fired={} -> {}".format(
              DESIGN.lam_bic_m * 1e9, q_vtx, n_raw, guard, "PASS" if g3 else "FAIL"), flush=True)

    # -------- GATE 4: internal consistency of the three instruments --------
    # (a) Fano vs pole omega0 agree per angle; (b) Gamma-extrapolated resonance vs vortex wavelength
    dev = [abs(r.fano_omega0 - r.pole_omega0) / r.pole_omega0 for r in results]
    max_dev = max(dev)
    theta2 = np.array(thetas) ** 2
    a_ext = np.polyfit(theta2, np.array([r.pole_omega0 for r in results]), 1)   # omega0 = a1*theta^2 + a0
    omega0_gamma = float(a_ext[1])
    omega_vtx = 2.0 * math.pi * C_LIGHT / DESIGN.lam_bic_m
    dev_vtx = abs(omega0_gamma - omega_vtx) / omega_vtx
    g4 = bool(max_dev < 0.01 and dev_vtx < 0.01)
    ok = ok and g4
    print("[bic] GATE 4 CONSISTENCY: max |fano-pole|/omega0 over angles={:.2e} (<1%); "
          "Gamma-extrapolated resonance {:.2f}nm vs vortex-map {:.2f}nm dev={:.2e} (<1%) -> {}".format(
              max_dev, 2 * math.pi * C_LIGHT / omega0_gamma * 1e9, DESIGN.lam_bic_m * 1e9,
              dev_vtx, "PASS" if g4 else "FAIL"), flush=True)

    print("[bic] *** REAL-BIC END-TO-END CAPSTONE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
