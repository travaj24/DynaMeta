"""
Measurement averaging: turn ideal plane-wave, perfectly periodic optical solves into what a real
reflectance measurement of a fabricated grating device records -- so that broadening caused by the
MEASUREMENT (finite illumination/collection cone, instrument resolution) and by FABRICATION SCATTER
(slowly varying width / period / thickness / density) can be separated from broadening that is
intrinsic material loss.

A thin, opt-in layer over the existing optical-solver interface: no solver lives here. Everything is
an average over plane-wave solves supplied by a caller's MODEL callable (any solver: the layered
TMM, the 2-D TM FEM, the 3-D FEM, the RCWA/PMM bridges, or a closed form), followed by a detection
model and, last, the spectrometer. Nothing in the library calls this module, so no existing result
changes.

PUBLIC API
----------
  pupil_quadrature / PupilQuadrature     the cone of incidence directions (item 1)
  plane_wave_incidence                   a single direction (the plane-wave limit)
  parameter_quadrature / combine_parameters / ParameterQuadrature
                                         fabrication-scatter distributions (item 2)
  patch_average / PatchAverage           the coherent / incoherent patch average of r (item 2)
  instrument_convolve / InstrumentFunction
                                         spectrometer line shape on a non-uniform grid (item 3)
  normalize_to_reference                 ratio to a reference mirror under the same averaging
  MeasurementSetup / measure / MeasurementPlan / MeasuredSpectrum
                                         the composition, with its cost reported first (item 5)
  scalar_model / design_plane_wave_model adapters that build a MODEL callable
  fit_broadening / BroadeningFit         separate broadening parameters from intrinsic damping,
                                         with covariance, profile scans and a degeneracy verdict
                                         (item 6)

THE MODEL CALLABLE
------------------
    model(wavelengths_m, theta_deg=..., phi_deg=..., pol='s'|'p', **params) -> r

returns the SPECULAR (zeroth-order) complex reflection amplitude for an incident plane wave of
polarization `pol` at polar angle theta (in the incidence medium) and azimuth phi (0 = plane of
incidence x-z, x = the grating vector; the lines of a 1-D grating run along y), one value per
wavelength. `params` are the fabrication parameters the scatter average varies (e.g. width_m=...).
It may instead return a tuple (r_co, r_cross): r_cross is the specular amplitude converted into the
ORTHOGONAL polarization (input s -> output p is r_ps, input p -> output s is r_sp). Without it the
conversion terms are taken as zero and the result says so (exact in-plane for any y-invariant
grating with isotropic materials, and for every planar isotropic stack at any angle).

Basis and signs are the library's (optics.tmm_reference, Byrnes `tmm`; the FEM, the 2-D TM FEM and
the RCWA/PMM bridges all report this convention): s-hat = (-sin phi, cos phi, 0) for the incident
and the reflected wave; incident p-hat = (cos th cos phi, cos th sin phi, sin th) (wave travelling
in -z); reflected p-hat = (-cos th cos phi, -cos th sin phi, sin th). Hence r_p = -r_s at normal
incidence, and the LAB co-polarized field ratio of an in-plane TM wave is -r_p.

ITEM 1: FINITE ILLUMINATION / COLLECTION CONE
---------------------------------------------
Directions are labelled by the normalized transverse wavevector s = k_par / k0 = n sin(theta) (n the
incidence-medium index, so |s| <= NA). The solid-angle Jacobian is

    d^2 k_par = (n k0)^2 cos(theta) sin(theta) dtheta dphi = (n k0)^2 cos(theta) dOmega.

* Uniformly filled circular pupil of numerical aperture NA (an aplanatic objective obeying the sine
  condition, or a Koehler condenser): the power per unit d^2 s is constant for NA_inner <= |s| <=
  NA, i.e. the power per unit SOLID ANGLE is proportional to cos(theta). (A source uniform per solid
  angle would weight by 1/cos(theta) relative to this; pass a callable profile for it.)
* Gaussian beam: an aplanatic lens with a Gaussian pupil fill gives power per unit d^2 s
  proportional to exp(-2 |s|^2 / s0^2). s0 = na_e2 is the 1/e^2 pupil radius in s units (fixed in
  the pupil, so the focal waist scales with wavelength); or give the paraxial focal waist w0 (1/e^2
  intensity radius) and s0 = lambda / (pi w0) per wavelength (the angular spectrum of a Gaussian of
  waist w0 is exp(-k_par^2 w0^2 / 4) in field). Truncated at the objective NA.

Polarization (Richards-Wolf). A linearly polarized input with lab direction e_in = (cos a, sin a) in
the pupil, a = polarization_deg from x, decomposes on the ray of azimuth phi into
    a_p = e_in . rho_hat = cos(a - phi)      (radial  -> p in focus)
    a_s = e_in . phi_hat = sin(a - phi)      (azimuthal -> s in focus).
The reflected ray returns through the same lens to the pupil, where (tracking the meridional
rotation) its s part lies along phi_hat and its p part along -rho_hat:
    E_ret = (a_s r_ss + a_p r_sp) phi_hat + (a_p r_pp + a_s r_ps) (-rho_hat),
    E_par  = e_in . E_ret    = a_s^2 r_ss - a_p^2 r_pp + a_s a_p (r_sp - r_ps),
    E_perp = e_perp . E_ret  = a_s a_p (r_ss + r_pp) + a_p^2 r_sp + a_s^2 r_ps.
At normal incidence E_par = r_s (the lab field ratio) and E_perp = 0, so NA -> 0 returns the
plane-wave result exactly; a perfect mirror gives E_par = -1 on every ray.

Which solvers can do the conical (phi != 0) part: the 3-D FEM (`optics.solver.solve_fem`, rotated
s/p at any azimuth), the RCWA bridge (2-D lattice; rotated s/p synthesized from per-order
amplitudes, lossless end media) and the PMM bridge (same synthesis, lossless end media), and for
planar stacks the layered TMM and Berreman solvers (azimuth-invariant). NONE of them returns the
cross-polarized specular amplitude at conical incidence through OpticalResult (the 3-D FEM's
r_cross exists at normal incidence only), so the conical path with them neglects the s <-> p
conversion terms; the result carries that note. In-plane only: the 2-D TM FEM (`optics.fem_2d_tm`,
TM only) and the PMM2D bridge. For those, mode='in_plane' is the labelled IN-PLANE-ONLY
APPROXIMATION: each ray is replaced by the in-plane ray with the SAME k_x (n sin th' = s_x), k_y is
dropped, and the input splits into TM (E across the lines, 'p') with amplitude cos(a) and TE (E
along the lines, 's') with amplitude sin(a). The 2-D pupil integral then reduces EXACTLY to a 1-D
integral over s_x with the marginal weight W_x(s_x) = INT W ds_y (the semicircle 2 sqrt(NA^2 -
s_x^2) for a uniform pupil). Keeping k_x and not the polar angle is the right first-order choice:
a grating-coupled resonance obeys k_x + m G = beta(k_y) with beta even in k_y, so the dropped k_y
and the O(theta^2) focal depolarization both enter at O(NA^2); the matching-|theta| alternative is
wrong at first order. The approximation is exact at NA -> 0.

DETECTION MODELS (do not conflate them):
  (a) detection='power' for an EXTENDED INCOHERENT SOURCE (lamp in a microscope, a spectrometer):
      every illumination direction reflects independently and the detector adds intensities,
          S = sum_k W_k P_k,  P_k = |E_par|^2 + |E_perp|^2 (or one of them behind an analyzer).
  (b) A COHERENT FOCUSED BEAM, specularly reflected and collected:
      (b1) integrated over the return pupil on a bucket detector: by Parseval the collected power is
           INT |B(s)|^2 d^2 s with B(s) = J(s) A(s) the reflected angular spectrum, i.e. the SAME
           sum of intensities as (a) with W = |A|^2 -- detection='power' again. This holds for a
           laterally uniform periodic sample only while no non-specular order enters the collection
           pupil (lambda / period > 2 NA); otherwise a diffracted order from direction s lands on
           the specular ray of s + m G and the two INTERFERE. Pass `period_m` and the setup warns.
      (b2) interfered with a reference beam, or coupled into a single mode matched to the
           illumination (confocal fiber, homodyne): the detected complex amplitude is the mode
           overlap INT A_ref^* . B d^2 s, the COHERENT AMPLITUDE AVERAGE
               r_coh = sum_k W_k E_par,k        (detection='amplitude'; signal = |r_coh|^2)
           relative to the same overlap for the incident field (a perfect mirror gives -1). This is
           what a phase measurement records; strong phase dispersion across the cone makes
           |r_coh|^2 < S even for a lossless sample (dephasing, not absorption).

ITEM 2: FABRICATION SCATTER
---------------------------
For a parameter p (strip width, period, layer thickness, film density) that varies SLOWLY across
the illuminated area, each patch is locally periodic and reflects with r(p). The specularly
reflected FIELD of all patches adds coherently in the far field, so a specular detector whose
acceptance is small compared with the diffuse cone (half-angle ~ lambda / (n L), L the correlation
length) records |<r>|^2 (scatter_collection='specular'); the remainder <|r|^2> - |<r>|^2 = Var(r)
>= 0 is scattered into that diffuse cone and is recorded only by a detector that collects it (an
objective of NA >> lambda / L, an integrating sphere: scatter_collection='all', signal <|r|^2>).
The diffuse part is returned as a diagnostic. Coherent mode-matched detection (b2) always sees the
patch-averaged amplitude <r>: the overlap INT E_in^* r(x) E_in d^2 x is the intensity-weighted
spatial average of r.
VALIDITY (the local-patch picture): L >> lambda, L >> period, and L >> the lateral propagation
length of the resonant mode (a high-Q grating mode averages r over its own decay length); the
illuminated area must contain many correlation lengths (a spot smaller than L measures one patch:
the scatter then shows up as spot-to-spot variation, not as broadening). Variation on a scale
smaller than the wavelength, or period-to-period disorder, is NOT this picture -- it couples into
non-specular directions and into the resonance itself and needs a supercell solve; it is not
modelled here and no option pretends to.

ITEM 3: SPECTRAL RESOLUTION AND NORMALIZATION
---------------------------------------------
A spectrometer convolves the DETECTED POWER spectrum with its instrument line shape (Gaussian,
Lorentzian, triangular for equal entrance/exit slits, rectangular for a single slit), on wavelength
(grating instruments) or on wavenumber (FTIR). It is applied LAST, to detected signals, never to a
complex amplitude, and a reflectance is formed as conv(S_sample) / conv(S_reference) x R_reference
(normalize_to_reference), with the reference mirror measured under the same cone. A coherent
amplitude is normalized the same way, amplitude_sample / amplitude_reference x r_reference, which
needs the reference's COMPLEX amplitude r_reference: a power reflectance R_reference fixes |r| but
not the phase (a metal mirror has r close to -1, not +1), so MeasurementSetup takes it from the
reference model ('plane_wave') or from reference_amplitude, and never from sqrt(R_reference).

ITEM 4: FINITE ARRAY SIZE
-------------------------
The leading effect of a finite number N of periods on a specular measurement is angular broadening
by diffraction from the array aperture: the plane-wave response is averaged over a k_x spread of
order 2 pi / (N period), i.e. an effective cone of half-width s ~ lambda / (N period) along x --
item 1 with mode='in_plane' and a profile equal to the aperture's far-field intensity (sinc^2 for
uniform illumination). No separate model is provided. When the resonant mode's lateral propagation
length exceeds N period, edge leakage (a loss channel, not an angular average) dominates and needs
a finite-array solve.

ITEM 5: COMPOSITION ORDER
-------------------------
Angles and scatter are averages over solves performed at each wavelength (fields first, then the
detection model, then the angle sum); the instrument acts last on the detected signal; the
reference ratio is formed after both signals are convolved. MeasurementSetup.plan() reports the
number of solves before anything runs; run() takes n_workers (a thread pool) or map_fn (any
map-like callable, e.g. a process pool's map with a picklable model).
"""

from __future__ import annotations

import itertools
import math
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dynameta.warning_types import UnreliableResultWarning

__all__ = [
    "PupilQuadrature", "pupil_quadrature", "plane_wave_incidence",
    "ParameterQuadrature", "parameter_quadrature", "combine_parameters",
    "PatchAverage", "patch_average",
    "InstrumentFunction", "instrument_convolve", "normalize_to_reference",
    "MeasurementPlan", "MeasuredSpectrum", "MeasurementSetup", "measure",
    "scalar_model", "design_plane_wave_model",
    "BroadeningFit", "fit_broadening",
]

_ZERO_AMP = 1e-14            # |a_s| or |a_p| below this on every ray: that input column is not needed
_KEY_DIGITS = 12             # rounding of node coordinates when deduplicating solves


def _gl(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


# ================================================================================================
# Item 1: the cone of incidence directions
# ================================================================================================

@dataclass(frozen=True, eq=False)
class PupilQuadrature:
    """Quadrature over the illumination/collection cone (see the module docstring, ITEM 1).

    Nodes are normalized transverse wavevectors (s_x, s_y) = n sin(theta) (cos phi, sin phi) in the
    incidence medium. `measure` holds the quadrature weights of the integration measure WITHOUT the
    pupil profile (d^2 s in mode 'conical'; d s_x of the marginal in mode 'in_plane'); weights(lam)
    multiplies in the profile (or its marginal) and normalizes to unit sum per wavelength, so a
    profile that depends on wavelength (a Gaussian of fixed waist) is handled exactly.

    mode 'conical'  : full 2-D cone; every node is a (theta, phi) solve, both input polarizations.
    mode 'in_plane' : the labelled in-plane-only approximation (same k_x, k_y dropped); nodes on
                      the s_x axis with the exact marginal weight; theta' is signed (k_x < 0 -> a
                      negative angle at phi = 0, which is the same physical wave as phi = 180 deg).
    fold_symmetric  : in_plane only: the structure is mirror-symmetric under x -> -x, so
                      r(-theta') = r(theta') and only theta' >= 0 is solved.
    azimuthal_invariant : conical only: the sample's s/p Jones matrix does not depend on phi
                      (planar isotropic stacks): one solve per ring serves every phi node.

    The default instance is the normal-incidence plane wave polarized along x."""
    mode: str = "conical"
    na: float = 0.0
    na_inner: float = 0.0
    n_medium: float = 1.0
    profile: Any = "uniform"
    na_e2: Optional[float] = None
    waist_m: Optional[float] = None
    polarization_deg: float = 0.0
    s_x: np.ndarray = field(default_factory=lambda: np.zeros(1))
    s_y: np.ndarray = field(default_factory=lambda: np.zeros(1))
    measure: np.ndarray = field(default_factory=lambda: np.ones(1))
    fold_symmetric: bool = False
    azimuthal_invariant: bool = False
    n_marginal: int = 24

    def __post_init__(self):
        if self.mode not in ("conical", "in_plane"):
            raise ValueError("PupilQuadrature: mode must be 'conical' or 'in_plane', got {!r}"
                             .format(self.mode))
        if self.fold_symmetric and self.mode != "in_plane":
            raise ValueError("PupilQuadrature: fold_symmetric applies to mode='in_plane' only")
        if self.azimuthal_invariant and self.mode != "conical":
            raise ValueError("PupilQuadrature: azimuthal_invariant applies to mode='conical' only")

    @property
    def n_rays(self) -> int:
        return int(np.asarray(self.s_x).size)

    @property
    def theta_deg(self) -> np.ndarray:
        """Polar angle in the incidence medium per node (signed in mode 'in_plane')."""
        n = float(self.n_medium)
        if self.mode == "in_plane":
            return np.degrees(np.arcsin(np.asarray(self.s_x) / n))
        return np.degrees(np.arcsin(np.hypot(self.s_x, self.s_y) / n))

    @property
    def phi_deg(self) -> np.ndarray:
        """Azimuth per node (0 in mode 'in_plane'; 0 for the axial ray)."""
        if self.mode == "in_plane":
            return np.zeros(self.n_rays)
        return np.degrees(np.arctan2(self.s_y, self.s_x))

    @property
    def polarization_amplitudes(self) -> Tuple[np.ndarray, np.ndarray]:
        """(a_s, a_p) per node: the input field's components along s-hat and p-hat (Richards-Wolf
        decomposition; in mode 'in_plane' the constants (sin a, cos a): TE and TM)."""
        a = math.radians(self.polarization_deg)
        phi = np.radians(self.phi_deg)
        return np.sin(a - phi), np.cos(a - phi)

    @property
    def approximation(self) -> Optional[str]:
        if self.mode == "in_plane" and self.na > 0.0:
            return ("in-plane-only approximation: each ray replaced by the in-plane ray with the "
                    "same k_x (k_y dropped); error O(NA^2)")
        return None

    def _s0(self, wavelengths_m: np.ndarray) -> np.ndarray:
        if self.waist_m is not None:
            return np.asarray(wavelengths_m, dtype=float) / (math.pi * float(self.waist_m))
        return np.full(np.shape(wavelengths_m), float(self.na_e2))

    def _profile(self, s: np.ndarray, s0: float) -> np.ndarray:
        if callable(self.profile):
            return np.asarray(self.profile(s), dtype=float) * np.ones_like(s)
        if self.profile == "uniform":
            return np.ones_like(s)
        return np.exp(-2.0 * s * s / (s0 * s0))

    def _marginal(self, sx: np.ndarray, s0: float) -> np.ndarray:
        """W_x(s_x) = INT W(s_x, s_y) ds_y over the annulus NA_inner <= |s| <= NA."""
        y_out = np.sqrt(np.maximum(self.na ** 2 - sx * sx, 0.0))
        y_in = np.sqrt(np.maximum(self.na_inner ** 2 - sx * sx, 0.0))
        if self.profile == "uniform" and not callable(self.profile):
            return 2.0 * (y_out - y_in)
        if self.profile == "gaussian" and not callable(self.profile):
            from scipy.special import erf
            k = math.sqrt(2.0) / s0
            return (np.exp(-2.0 * sx * sx / (s0 * s0)) * s0 * math.sqrt(math.pi / 2.0)
                    * (erf(k * y_out) - erf(k * y_in)))
        out = np.empty_like(sx)
        for i, (x, lo, hi) in enumerate(zip(sx, y_in, y_out)):
            if hi <= lo:
                out[i] = 0.0
                continue
            # s_y = lo + (hi - lo) sin^2-like map keeps the sqrt end at hi smooth
            t, wt = _gl(self.n_marginal, 0.0, 0.5 * math.pi)
            sy = lo + (hi - lo) * np.sin(t)
            jac = (hi - lo) * np.cos(t)
            out[i] = 2.0 * float(np.sum(wt * jac * self._profile(np.hypot(x, sy), s0)))
        return out

    def weights(self, wavelengths_m) -> np.ndarray:
        """Normalized power weights, shape (n_lambda, n_rays); each row sums to 1."""
        lam = np.atleast_1d(np.asarray(wavelengths_m, dtype=float))
        if self.na == 0.0:
            return np.ones((lam.size, 1))
        lam_dep = self.profile == "gaussian" and self.waist_m is not None
        s0s = self._s0(lam) if (self.profile == "gaussian" and not callable(self.profile)) \
            else np.ones(lam.size)
        rows = []
        cache = None
        for s0 in s0s:
            if cache is not None and not lam_dep:
                rows.append(cache)
                continue
            if self.mode == "conical":
                prof = self._profile(np.hypot(self.s_x, self.s_y), s0)
            else:
                prof = self._marginal(np.asarray(self.s_x, dtype=float), s0)
            w = np.asarray(self.measure, dtype=float) * prof
            tot = float(np.sum(w))
            if not tot > 0.0:
                raise ValueError("PupilQuadrature: the profile has no weight inside the pupil")
            cache = w / tot
            rows.append(cache)
        return np.vstack(rows)


def pupil_quadrature(na: float, *, n_medium: float = 1.0, profile: Any = "uniform",
                     na_e2: Optional[float] = None, waist_m: Optional[float] = None,
                     na_inner: float = 0.0, polarization_deg: float = 0.0,
                     mode: str = "conical", n_radial: int = 4, n_azimuth: int = 16,
                     n_kx: int = 9, fold_symmetric: bool = False,
                     azimuthal_invariant: bool = False) -> PupilQuadrature:
    """Build the cone quadrature for an objective / focused beam (module docstring, ITEM 1).

    na        : numerical aperture n sin(theta_max) in the incidence medium (0 -> the exact plane
                wave at normal incidence: one node, weight 1). Must be < n_medium.
    profile   : 'uniform' (filled pupil: power per d^2 s constant, i.e. per solid angle ~ cos
                theta), 'gaussian' (power per d^2 s ~ exp(-2 s^2 / s0^2); give exactly one of na_e2
                = s0 fixed in the pupil, or waist_m = the paraxial focal 1/e^2 waist, s0 = lambda /
                (pi waist) per wavelength), or a callable profile(s) -> power per d^2 s.
    na_inner  : central obscuration (reflective / Cassegrain objectives), annulus na_inner..na.
    polarization_deg : lab direction a of the linear input polarization in the pupil, from x (the
                grating vector); 0 = E across the lines of a 1-D grating along y.
    mode      : 'conical' (2-D cone: Gauss-Legendre in s^2 x n_radial, uniform phi x n_azimuth;
                2 n_radial n_azimuth solves per spectrum) or 'in_plane' (the labelled same-k_x
                approximation: n_kx nodes on s_x with the exact marginal weight, Gauss-Legendre
                after s_x = NA sin(t), which removes the semicircle's square-root end point; with
                an obscuration the inner and outer pieces are integrated separately).
    fold_symmetric / azimuthal_invariant : see PupilQuadrature.

    Accuracy (a 300 nm n = 2 film on n = 1.5 at 600-700 nm, NA 0.5, polarization 30 deg, against a 2-D
    adaptive integral; max over the three wavelengths): the uniform (filled) pupil agrees to 2e-11 in
    power and 1e-11 in amplitude with the default n_radial = 4, n_azimuth = 16 (6e-17 at n_radial = 8).
    A Gaussian pupil is harder, because the Gauss-Legendre rule in s^2 must also integrate the weight
    exp(-2 s^2 / s0^2), whose variation over the pupil is set by 2 NA^2 / s0^2: at n_radial = 4 the
    error is 1.2e-6 (power) / 2.4e-6 (amplitude) for s0 = 0.35 (2 NA^2 / s0^2 = 4.1), 1.2e-8 / 2.4e-8 for
    s0 = 0.5 and 3e-12 for s0 = 1.0; n_radial = 8 brings all three to <= 1e-14. Use n_radial >= 8 for a
    Gaussian pupil with s0 <= NA (or check by doubling n_radial)."""
    na = float(na)
    na_inner = float(na_inner)
    n_medium = float(n_medium)
    if mode not in ("conical", "in_plane"):
        raise ValueError("pupil_quadrature: mode must be 'conical' or 'in_plane'")
    if not (0.0 <= na < n_medium):
        raise ValueError("pupil_quadrature: need 0 <= na < n_medium (got na={}, n_medium={}); a "
                         "grazing ray has no propagating reflection to average".format(na, n_medium))
    if not (0.0 <= na_inner < max(na, 1e-300)) and not (na == 0.0 and na_inner == 0.0):
        raise ValueError("pupil_quadrature: need 0 <= na_inner < na")
    if not (callable(profile) or profile in ("uniform", "gaussian")):
        raise ValueError("pupil_quadrature: profile must be 'uniform', 'gaussian' or a callable")
    if profile == "gaussian" and not callable(profile):
        if (na_e2 is None) == (waist_m is None):
            raise ValueError("pupil_quadrature: profile='gaussian' needs exactly one of na_e2 (pupil "
                             "1/e^2 radius in s units) or waist_m (focal 1/e^2 waist)")
        if (na_e2 is not None and not na_e2 > 0) or (waist_m is not None and not waist_m > 0):
            raise ValueError("pupil_quadrature: na_e2 / waist_m must be > 0")
    elif na_e2 is not None or waist_m is not None:
        raise ValueError("pupil_quadrature: na_e2 / waist_m apply to profile='gaussian' only")
    common = dict(mode=mode, na=na, na_inner=na_inner, n_medium=n_medium, profile=profile,
                  na_e2=na_e2, waist_m=waist_m, polarization_deg=float(polarization_deg),
                  fold_symmetric=bool(fold_symmetric), azimuthal_invariant=bool(azimuthal_invariant))
    if na == 0.0:
        return PupilQuadrature(s_x=np.zeros(1), s_y=np.zeros(1), measure=np.ones(1), **common)
    if mode == "conical":
        if int(n_radial) < 1 or int(n_azimuth) < 1:
            raise ValueError("pupil_quadrature: n_radial and n_azimuth must be >= 1")
        u, wu = _gl(int(n_radial), na_inner ** 2, na ** 2)           # u = s^2, d^2 s = du dphi / 2
        s = np.sqrt(u)
        phi = 2.0 * math.pi * np.arange(int(n_azimuth)) / int(n_azimuth)
        S, PHI = np.meshgrid(s, phi, indexing="ij")
        W = np.repeat(0.5 * wu[:, None] * (2.0 * math.pi / int(n_azimuth)), phi.size, axis=1)
        return PupilQuadrature(s_x=(S * np.cos(PHI)).ravel(), s_y=(S * np.sin(PHI)).ravel(),
                               measure=W.ravel(), **common)
    n_kx = int(n_kx)
    if n_kx < 2:
        raise ValueError("pupil_quadrature: n_kx must be >= 2")
    if na_inner == 0.0:
        t, wt = _gl(n_kx, -0.5 * math.pi, 0.5 * math.pi)
        sx, meas = na * np.sin(t), na * np.cos(t) * wt
    else:
        n_in = max(3, n_kx // 3)
        n_out = max(2, (n_kx - n_in) // 2)
        u, wu = _gl(n_in, -0.5 * math.pi, 0.5 * math.pi)             # inner: s_x = NA_in sin u
        t_in = math.asin(na_inner / na)
        t, wt = _gl(n_out, t_in, 0.5 * math.pi)                      # outer: s_x = NA sin t
        sx = np.concatenate([-na * np.sin(t[::-1]), na_inner * np.sin(u), na * np.sin(t)])
        meas = np.concatenate([na * np.cos(t[::-1]) * wt[::-1], na_inner * np.cos(u) * wu,
                               na * np.cos(t) * wt])
    return PupilQuadrature(s_x=sx, s_y=np.zeros_like(sx), measure=meas, **common)


def plane_wave_incidence(theta_deg: float = 0.0, phi_deg: float = 0.0, *,
                         polarization_deg: float = 0.0, n_medium: float = 1.0,
                         mode: str = "conical") -> PupilQuadrature:
    """A single incidence direction (weight 1). polarization_deg is the lab angle a of the input
    as for a pupil; on this ray a_p = cos(a - phi), a_s = sin(a - phi): a = phi is pure p, a = phi +
    90 deg pure s. mode='in_plane' needs phi_deg = 0 (a signed theta selects k_x < 0)."""
    th = math.radians(float(theta_deg))
    ph = math.radians(float(phi_deg))
    if mode == "in_plane" and abs(phi_deg) > 1e-12:
        raise ValueError("plane_wave_incidence: mode='in_plane' needs phi_deg=0")
    s = float(n_medium) * math.sin(th)
    return PupilQuadrature(mode=mode, na=0.0, n_medium=float(n_medium),
                           polarization_deg=float(polarization_deg),
                           s_x=np.array([s * math.cos(ph)]), s_y=np.array([s * math.sin(ph)]),
                           measure=np.ones(1))


# ================================================================================================
# Item 2: fabrication-scatter distributions
# ================================================================================================

@dataclass(frozen=True, eq=False)
class ParameterQuadrature:
    """Nodes and weights over one or several independent fabrication parameters.

    names   : parameter names (the keyword the model receives).
    nodes   : (n, d) parameter values; weights (n,) summing to 1 (a sparse rule has negative
              weights, see combine_parameters).
    rule    : 'gauss-hermite' (normal; exact for polynomials of degree <= 2n-1 against the
              Gaussian), 'gauss-legendre' (uniform; same degree), 'samples' (equal weights, Monte
              Carlo error ~ std / sqrt(n)), 'user', 'tensor(...)' or 'sparse(level=L)'.
    specs   : per-dimension (distribution, mean, scale) for rebuilding at another order (None for
              samples / user nodes).
    The default instance is 'no scatter' (one node, weight 1, no names)."""
    names: Tuple[str, ...] = ()
    nodes: np.ndarray = field(default_factory=lambda: np.zeros((1, 0)))
    weights: np.ndarray = field(default_factory=lambda: np.ones(1))
    rule: str = "none"
    specs: Optional[Tuple[Tuple[str, float, float], ...]] = None

    @property
    def n_nodes(self) -> int:
        return int(np.asarray(self.weights).size)

    def values(self, i: int) -> Dict[str, float]:
        """The keyword arguments of node i."""
        return {nm: float(self.nodes[i, j]) for j, nm in enumerate(self.names)}

    def refine(self, n_nodes: int) -> "ParameterQuadrature":
        """The same distribution(s) with n_nodes per dimension (tensor product for d > 1)."""
        if self.specs is None:
            raise ValueError("ParameterQuadrature.refine: samples / user nodes cannot be rebuilt "
                             "at another order")
        quads = [_rule_1d(nm, dist, mean, scale, n_nodes)
                 for nm, (dist, mean, scale) in zip(self.names, self.specs)]
        return quads[0] if len(quads) == 1 else combine_parameters(*quads, method="tensor")


def _std_rule(dist: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Standardized 1-D rule: probabilists' Gauss-Hermite (unit normal) or Gauss-Legendre on
    [-1, 1] (unit uniform), weights summing to 1."""
    if dist == "normal":
        x, w = np.polynomial.hermite_e.hermegauss(int(n))
    else:
        x, w = np.polynomial.legendre.leggauss(int(n))
    return x, w / np.sum(w)


def _rule_1d(name, dist, mean, scale, n) -> ParameterQuadrature:
    x, w = _std_rule(dist, n)
    return ParameterQuadrature(names=(str(name),), nodes=(mean + scale * x)[:, None], weights=w,
                               rule="gauss-hermite" if dist == "normal" else "gauss-legendre",
                               specs=((dist, float(mean), float(scale)),))


def parameter_quadrature(name: str, distribution: str = "normal", *, mean: float = 0.0,
                         std: Optional[float] = None, half_width: Optional[float] = None,
                         n_nodes: int = 7, samples: Optional[Sequence[float]] = None,
                         nodes: Optional[Sequence[float]] = None,
                         weights: Optional[Sequence[float]] = None) -> ParameterQuadrature:
    """One fabrication parameter's distribution as a quadrature.

    distribution 'normal'  (mean, std)        -> Gauss-Hermite, n_nodes points
                 'uniform' (mean, half_width) -> Gauss-Legendre on [mean - hw, mean + hw]
                 'samples' (samples=...)      -> equal weights (measured values, Monte Carlo)
                 'user'    (nodes=, weights=) -> as given (normalized to unit sum)
    Convergence depends on how close the model's complex singularities sit to the real parameter
    axis, in units of the spread. A resonance whose centre moves with p has a pole at a distance
    (linewidth / centre spread) std from the real axis, and Gauss-Hermite then converges only
    root-exponentially: measured for a Lorentzian of HWHM 0.83 x the centre spread, the error in
    <|r|^2> is 1e-2, 7e-5, 9e-7, 2e-9 at n = 10, 40, 80, 160. A line much broader than the spread
    converges in a handful of nodes. Check with MeasurementSetup(convergence_nodes=...) or
    ParameterQuadrature.refine. The Hermite nodes reach +-3.75 std at n = 7 (+-13 std at n = 100):
    make sure the model accepts them (e.g. a width stays > 0)."""
    if distribution == "normal":
        if std is None or not std > 0:
            raise ValueError("parameter_quadrature: 'normal' needs std > 0")
        return _rule_1d(name, "normal", float(mean), float(std), n_nodes)
    if distribution == "uniform":
        if half_width is None or not half_width > 0:
            raise ValueError("parameter_quadrature: 'uniform' needs half_width > 0")
        return _rule_1d(name, "uniform", float(mean), float(half_width), n_nodes)
    if distribution == "samples":
        v = np.asarray(samples, dtype=float).ravel()
        if v.size < 1:
            raise ValueError("parameter_quadrature: 'samples' needs at least one sample")
        return ParameterQuadrature(names=(str(name),), nodes=v[:, None],
                                   weights=np.full(v.size, 1.0 / v.size), rule="samples")
    if distribution == "user":
        x = np.asarray(nodes, dtype=float).ravel()
        w = np.asarray(weights, dtype=float).ravel()
        if x.shape != w.shape or x.size < 1 or not np.sum(w) > 0:
            raise ValueError("parameter_quadrature: 'user' needs matching nodes and weights with a "
                             "positive sum")
        return ParameterQuadrature(names=(str(name),), nodes=x[:, None], weights=w / np.sum(w),
                                   rule="user")
    raise ValueError("parameter_quadrature: distribution must be 'normal', 'uniform', 'samples' or "
                     "'user', got {!r}".format(distribution))


def combine_parameters(*quads: ParameterQuadrature, method: str = "tensor",
                       level: Optional[int] = None) -> ParameterQuadrature:
    """Several INDEPENDENT parameters.

    method='tensor': the product rule; cost = prod(n_i) nodes, exact for every polynomial whose
        degree in each variable is <= 2 n_i - 1.
    method='sparse': Smolyak combination of the 1-D Gauss rules with l points at level l (needs
        every input built from a 'normal'/'uniform' distribution). Level L >= d (d = number of
        parameters): exact for every polynomial of TOTAL degree <= 2 (L - d) + 1; cost = the number
        of distinct nodes, reported in .n_nodes. Measured node counts for total degree 5 (sparse
        vs the 3^d tensor rule of the same exactness): d=2 13 vs 9, d=3 25 vs 27, d=4 41 vs 81,
        d=5 61 vs 243, d=6 85 vs 729 -- with these non-nested Gauss rules the sparse rule pays off
        only from about four parameters; use 'tensor' below that. Weights can be negative, so a
        sparse <|r|^2> - |<r>|^2 can dip slightly below zero."""
    if not quads:
        raise ValueError("combine_parameters: no quadratures given")
    names = tuple(nm for q in quads for nm in q.names)
    if len(set(names)) != len(names):
        raise ValueError("combine_parameters: duplicate parameter names {}".format(names))
    if method == "tensor":
        nodes, weights = [], []
        for combo in itertools.product(*[range(q.n_nodes) for q in quads]):
            nodes.append(np.concatenate([q.nodes[i] for q, i in zip(quads, combo)]))
            weights.append(float(np.prod([q.weights[i] for q, i in zip(quads, combo)])))
        specs = (tuple(s for q in quads for s in q.specs)
                 if all(q.specs is not None for q in quads) else None)
        return ParameterQuadrature(names=names, nodes=np.asarray(nodes), weights=np.asarray(weights),
                                   rule="tensor({})".format(" x ".join(q.rule for q in quads)),
                                   specs=specs)
    if method != "sparse":
        raise ValueError("combine_parameters: method must be 'tensor' or 'sparse'")
    if any(q.specs is None for q in quads):
        raise ValueError("combine_parameters: method='sparse' needs normal/uniform inputs")
    specs = tuple(s for q in quads for s in q.specs)
    d = len(specs)
    L = int(level) if level is not None else d + 2
    if L < d:
        raise ValueError("combine_parameters: sparse level must be >= the number of parameters")
    acc: Dict[tuple, list] = {}
    for ls in itertools.product(range(1, L + 1), repeat=d):
        q_ = sum(ls)
        if q_ < L - d + 1 or q_ > L:
            continue
        coef = (-1) ** (L - q_) * math.comb(d - 1, L - q_)
        rules = [_std_rule(dist, l) for (dist, _m, _s), l in zip(specs, ls)]
        for combo in itertools.product(*[range(len(r[0])) for r in rules]):
            x = tuple(float(r[0][i]) for r, i in zip(rules, combo))
            w = coef * float(np.prod([r[1][i] for r, i in zip(rules, combo)]))
            key = tuple(round(v, _KEY_DIGITS) for v in x)
            if key in acc:
                acc[key][1] += w
            else:
                acc[key] = [x, w]
    xs = np.asarray([v[0] for v in acc.values()])
    ws = np.asarray([v[1] for v in acc.values()])
    keep = np.abs(ws) > 1e-15
    xs, ws = xs[keep], ws[keep]
    means = np.asarray([m for _d, m, _s in specs])
    scales = np.asarray([s for _d, _m, s in specs])
    return ParameterQuadrature(names=names, nodes=means + scales * xs, weights=ws / np.sum(ws),
                               rule="sparse(level={})".format(L), specs=None)


@dataclass(frozen=True, eq=False)
class PatchAverage:
    """Patch average of a reflection amplitude over a parameter distribution (ITEM 2).
    mean = <r>; coherent = |<r>|^2 (small-aperture specular detector); incoherent = <|r|^2>
    (detector collecting the diffuse cone too); diffuse = incoherent - coherent = Var(r)."""
    mean: Any = 0j
    coherent: Any = 0.0
    incoherent: Any = 0.0
    diffuse: Any = 0.0


def _patch_moments(values: np.ndarray, weights: np.ndarray):
    """THE patch-average kernel: (<v>, <|v|^2>) over axis 0 with the given weights."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values)
    return np.tensordot(w, v, axes=(0, 0)), np.tensordot(w, np.abs(v) ** 2, axes=(0, 0))


def patch_average(values, weights) -> PatchAverage:
    """Coherent and incoherent patch averages of complex amplitudes `values` (axis 0 = the
    parameter nodes, any trailing shape) with quadrature `weights` (n,). With non-negative weights
    |<r>|^2 <= <|r|^2> and the diffuse part is the weighted variance of r."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values)
    if v.shape[0] != w.size:
        raise ValueError("patch_average: values.shape[0]={} != len(weights)={}".format(v.shape[0], w.size))
    m, second = _patch_moments(v, w)
    coh = np.abs(m) ** 2
    return PatchAverage(mean=m, coherent=coh, incoherent=second, diffuse=second - coh)


# ================================================================================================
# Item 3: instrument line shape and reference normalization
# ================================================================================================

_SHAPES = ("gaussian", "lorentzian", "triangle", "rectangle")


def _kernel_antiderivatives(shape: str, fwhm: float):
    """(F, M): F(t) = INT_{-inf}^t K, M(t) = INT^t u K(u) du (up to a constant) for a unit-area
    symmetric kernel K of the given FWHM."""
    if shape == "gaussian":
        from scipy.special import erf
        sig = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return (lambda t: 0.5 * (1.0 + erf(t / (sig * math.sqrt(2.0)))),
                lambda t: -sig / math.sqrt(2.0 * math.pi) * np.exp(-0.5 * (t / sig) ** 2))
    if shape == "lorentzian":
        g = 0.5 * fwhm
        return (lambda t: 0.5 + np.arctan(t / g) / math.pi,
                lambda t: g / (2.0 * math.pi) * np.log(t * t + g * g))
    if shape == "rectangle":
        a = fwhm
        return (lambda t: np.clip((t + 0.5 * a) / a, 0.0, 1.0),
                lambda t: (np.clip(t, -0.5 * a, 0.5 * a) ** 2 - 0.25 * a * a) / (2.0 * a))
    if shape == "triangle":
        b = fwhm

        def F(t):
            tc = np.clip(t, -b, b)
            return np.where(tc < 0.0, (tc + b) ** 2 / (2 * b * b), 1.0 - (b - tc) ** 2 / (2 * b * b))

        def M(t):
            tc = np.clip(t, -b, b)
            neg = (b * tc * tc / 2.0 + tc ** 3 / 3.0 - b ** 3 / 6.0) / (b * b)
            pos = -b / 6.0 + (b * tc * tc / 2.0 - tc ** 3 / 3.0) / (b * b)
            return np.where(tc < 0.0, neg, pos)
        return F, M
    raise ValueError("instrument line shape must be one of {}, got {!r}".format(_SHAPES, shape))


def _convolve_pl(x: np.ndarray, y: np.ndarray, F, M, edge: str, block: int = 256) -> np.ndarray:
    """Exact convolution of the piecewise-linear interpolant of (x, y) (x ascending, any spacing)
    with a symmetric unit-area kernel given by its antiderivatives F, M, evaluated at every x."""
    dx = np.diff(x)
    m = np.diff(y) / dx
    out = np.empty(x.size)
    for i0 in range(0, x.size, block):
        xi = x[i0:i0 + block, None]
        t0 = x[None, :-1] - xi
        t1 = x[None, 1:] - xi
        out[i0:i0 + block] = np.sum((y[None, :-1] - m[None, :] * t0) * (F(t1) - F(t0))
                                    + m[None, :] * (M(t1) - M(t0)), axis=1)
    if edge == "renormalize":
        out /= F(x[-1] - x) - F(x[0] - x)
    return out


def instrument_convolve(wavelengths_m, spectrum, *, shape: str = "gaussian",
                        fwhm_m: Optional[float] = None, fwhm_per_cm: Optional[float] = None,
                        source=None, edge: str = "renormalize") -> np.ndarray:
    """Spectrometer line-shape convolution of a DETECTED POWER spectrum on a non-uniform grid.

    out(x_i) = INT K(x_i - x') S(x') dx', computed EXACTLY for the piecewise-linear interpolant of
    the samples (closed-form kernel antiderivatives on every grid interval), so no resampling and no
    requirement that the grid resolve the kernel.
    shape  : 'gaussian' | 'lorentzian' | 'triangle' (equal entrance and exit slits; FWHM = half
             base) | 'rectangle' (a single slit; FWHM = full width). Unit area.
    fwhm_m : constant bandpass in WAVELENGTH (grating instrument), x = lambda; or
    fwhm_per_cm : constant resolution in WAVENUMBER (FTIR), x = 1/lambda in cm^-1.
    source : optional source spectral density on the same grid, per unit of x. The convolved
             quantity is then S * source and the result is divided by the convolved source --
             the ratio a spectrometer reports when the lamp is not flat across the bandpass.
    edge   : 'renormalize' (divide by the kernel mass inside the grid: constants are preserved up
             to the edges) or 'zero' (the spectrum is zero outside the grid: the integral of the
             interpolant is conserved exactly when the kernel lies inside the grid).
    Returns the convolved spectrum in the input order."""
    lam = np.asarray(wavelengths_m, dtype=float).ravel()
    S = np.asarray(spectrum)
    if np.iscomplexobj(S):
        raise TypeError("instrument_convolve: a spectrometer convolves DETECTED POWER; got a complex "
                        "spectrum (convolve |r|^2 or the detected signal, not r)")
    S = np.asarray(S, dtype=float).ravel()
    if S.shape != lam.shape or lam.size < 2:
        raise ValueError("instrument_convolve: need matching 1-D wavelength and spectrum arrays")
    if (fwhm_m is None) == (fwhm_per_cm is None):
        raise ValueError("instrument_convolve: give exactly one of fwhm_m or fwhm_per_cm")
    if edge not in ("renormalize", "zero"):
        raise ValueError("instrument_convolve: edge must be 'renormalize' or 'zero'")
    if fwhm_m is not None:
        x, fw = lam, float(fwhm_m)
    else:
        x, fw = 0.01 / lam, float(fwhm_per_cm)
    if not fw > 0:
        raise ValueError("instrument_convolve: the FWHM must be > 0")
    order = np.argsort(x)
    xs = x[order]
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("instrument_convolve: duplicate wavelengths")
    F, M = _kernel_antiderivatives(shape, fw)
    if source is None:
        res = _convolve_pl(xs, S[order], F, M, edge)
    else:
        src = np.asarray(source, dtype=float).ravel()
        if src.shape != lam.shape:
            raise ValueError("instrument_convolve: source must match the wavelength grid")
        res = _convolve_pl(xs, (S * src)[order], F, M, edge) / _convolve_pl(xs, src[order], F, M, edge)
    out = np.empty_like(res)
    out[order] = res
    return out


@dataclass(frozen=True)
class InstrumentFunction:
    """Spectrometer description for MeasurementSetup; apply() calls instrument_convolve with these
    arguments (see there for the physics). Give exactly one of fwhm_m / fwhm_per_cm."""
    shape: str = "gaussian"
    fwhm_m: Optional[float] = None
    fwhm_per_cm: Optional[float] = None
    source: Any = None
    edge: str = "renormalize"

    def apply(self, wavelengths_m, spectrum) -> np.ndarray:
        return instrument_convolve(wavelengths_m, spectrum, shape=self.shape, fwhm_m=self.fwhm_m,
                                   fwhm_per_cm=self.fwhm_per_cm, source=self.source, edge=self.edge)


def normalize_to_reference(signal, reference_signal, reference_reflectance=1.0) -> np.ndarray:
    """Reflectance as a spectrometer forms it: signal / reference_signal x the reference's known
    reflectance. Both signals must have been detected under the SAME cone and convolved with the
    SAME instrument before the ratio (the ratio of convolutions is not the convolution of the
    ratio)."""
    return (np.asarray(signal, dtype=float) / np.asarray(reference_signal, dtype=float)
            * np.asarray(reference_reflectance, dtype=float))


# ================================================================================================
# Item 5: composition
# ================================================================================================

@dataclass(frozen=True)
class MeasurementPlan:
    """Cost of a MeasurementSetup on a wavelength grid, known before any solve.
    n_model_calls: calls of the model (each returns a whole spectrum); n_solves = n_model_calls x
    n_wavelengths plane-wave solves. Broken down into the sample, the convergence re-run and the
    reference mirror."""
    n_wavelengths: int = 0
    n_rays: int = 0
    n_ray_solves: int = 0
    polarizations: Tuple[str, ...] = ()
    n_parameter_nodes: int = 1
    n_sample_calls: int = 0
    n_convergence_calls: int = 0
    n_reference_calls: int = 0
    n_model_calls: int = 0
    n_solves: int = 0

    def describe(self) -> str:
        return ("measurement plan: {} rays ({} distinct solve directions) x pols {} x {} parameter "
                "nodes = {} sample calls (+{} convergence, +{} reference) = {} model calls x {} "
                "wavelengths = {} plane-wave solves".format(
                    self.n_rays, self.n_ray_solves, "/".join(self.polarizations) or "-",
                    self.n_parameter_nodes, self.n_sample_calls, self.n_convergence_calls,
                    self.n_reference_calls, self.n_model_calls, self.n_wavelengths, self.n_solves))


@dataclass(frozen=True, eq=False)
class MeasuredSpectrum:
    """What the measurement records.

    signal     : the final measured spectrum (detected -> instrument -> reference ratio).
    detected   : the detected signal before the instrument (power; |amplitude|^2 for amplitude
                 detection).
    amplitude  : coherent mode-matched amplitude sum_k W_k <E_par> (always computed; it IS the
                 measurement for detection='amplitude', not convolved by the instrument). With a
                 reference it is amplitude / (the reference's amplitude under the same cone) x the
                 reference's known complex amplitude (MeasurementSetup.reference_reflectance and
                 reference_amplitude); NaN where that complex amplitude is not known (a numeric
                 power reflectance without reference_amplitude, power detection only).
    specular / incoherent : power detected with the patch-coherent field |<E>|^2 and with <|E|^2>;
                 diffuse = incoherent - specular (fraction of incident power scattered out of the
                 specular direction by the parameter variation). Equal (diffuse 0) without scatter.
    reference_detected : the reference mirror's detected signal (None without a reference).
    convergence_delta : max |detected(n) - detected(m)| between the parameter rule and its re-run
                 at convergence_nodes (None when not requested).
    notes      : approximations in force (in-plane-only, cross-polarization neglected, ...)."""
    wavelengths_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    signal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    detected: np.ndarray = field(default_factory=lambda: np.zeros(0))
    amplitude: np.ndarray = field(default_factory=lambda: np.zeros(0, complex))
    specular: np.ndarray = field(default_factory=lambda: np.zeros(0))
    incoherent: np.ndarray = field(default_factory=lambda: np.zeros(0))
    diffuse: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reference_detected: Optional[np.ndarray] = None
    convergence_delta: Optional[float] = None
    plan: MeasurementPlan = field(default_factory=MeasurementPlan)
    notes: Tuple[str, ...] = ()
    elapsed_s: float = 0.0


class _Call:
    """One model call; a class (not a closure) so a process-pool map can pickle it."""

    def __init__(self, model, wavelengths_m):
        self.model = model
        self.wavelengths_m = wavelengths_m

    def __call__(self, job):
        theta, phi, pol, params = job
        return self.model(self.wavelengths_m, theta_deg=theta, phi_deg=phi, pol=pol, **params)


def _ray_keys(pupil: PupilQuadrature) -> List[Tuple[float, float]]:
    """(theta, phi) actually solved for each ray (after folding / azimuthal invariance)."""
    th = pupil.theta_deg
    ph = pupil.phi_deg
    keys = []
    for t, p in zip(th, ph):
        t = float(t)
        p = float(p)
        if pupil.fold_symmetric:
            t = abs(t)
        if pupil.azimuthal_invariant:
            p = 0.0
        keys.append((round(t, _KEY_DIGITS), round(p, _KEY_DIGITS)))
    return keys


def _pols_needed(pupil: PupilQuadrature) -> Tuple[str, ...]:
    a_s, a_p = pupil.polarization_amplitudes
    pols = []
    if np.any(np.abs(a_s) > _ZERO_AMP):
        pols.append("s")
    if np.any(np.abs(a_p) > _ZERO_AMP):
        pols.append("p")
    return tuple(pols)


def _ray_fields(a_s, a_p, J):
    """Return-pupil field (E_par, E_perp) per ray from the s/p Jones entries (module docstring)."""
    Es = a_s * J["ss"] + a_p * J["sp"]
    Ep = a_p * J["pp"] + a_s * J["ps"]
    return a_s * Es - a_p * Ep, a_p * Es + a_s * Ep


def _parse_output(out, n_lam):
    if isinstance(out, tuple) and len(out) == 2:
        co, cross = out
    else:
        co, cross = out, None
    co = np.broadcast_to(np.asarray(co, dtype=complex).reshape(-1), (n_lam,)).copy()
    if cross is not None:
        cross = np.broadcast_to(np.asarray(cross, dtype=complex).reshape(-1), (n_lam,)).copy()
    return co, cross


@dataclass(frozen=True, eq=False)
class MeasurementSetup:
    """A measurement description, applied to any model (module docstring, items 1-5).

    pupil      : PupilQuadrature (None -> the normal-incidence plane wave polarized along x).
    parameters : ParameterQuadrature of the fabrication scatter (None -> no scatter).
    detection  : 'power' (incoherent source, or a coherent beam on a bucket detector) or
                 'amplitude' (coherent mode-matched / interferometric detection; signal = |r_coh|^2).
    analyzer   : None (no polarizer: |E_par|^2 + |E_perp|^2), 'parallel' or 'crossed' (power only).
    scatter_collection : REQUIRED with parameters for power detection: 'specular' (small-aperture
                 specular detector: |<E>|^2) or 'all' (detector collecting the diffuse cone:
                 <|E|^2>). Amplitude detection always uses <E>.
    instrument : InstrumentFunction applied last to the detected signals.
    reference  : model of the reference mirror, detected under the same cone (no scatter); the
                 signal becomes conv(S) / conv(S_ref) x reference_reflectance.
    reference_reflectance : 'plane_wave' (the reference model's own normal-incidence plane-wave
                 reflectance and complex amplitude -- the tabulated value a user multiplies by), None
                 (raw ratios: signal / reference signal, amplitude / reference amplitude, i.e. the
                 phase relative to the reference), or a real POWER reflectance (array / float).
    reference_amplitude : the reference's known complex amplitude reflection coefficient at normal
                 incidence (array / complex; library basis, e.g. close to -1 for a metal mirror),
                 used to normalize the amplitude; only together with a numeric
                 reference_reflectance equal to abs(reference_amplitude)**2. A power reflectance alone
                 does not fix the phase of r, so with a numeric reference_reflectance and no
                 reference_amplitude, detection='amplitude' raises ValueError and power detection
                 returns amplitude = NaN (the power signal is unaffected).
    period_m   : grating period; when given, warns where a non-specular order enters the
                 collection pupil (lambda / period <= 2 NA), which this specular model omits.
    convergence_nodes : re-run the parameter rule with this many nodes per dimension and record /
                 warn on the change (UnreliableResultWarning above convergence_tol)."""
    pupil: Optional[PupilQuadrature] = None
    parameters: Optional[ParameterQuadrature] = None
    detection: str = "power"
    analyzer: Optional[str] = None
    scatter_collection: Optional[str] = None
    instrument: Optional[InstrumentFunction] = None
    reference: Optional[Callable] = None
    reference_reflectance: Any = "plane_wave"
    period_m: Optional[float] = None
    convergence_nodes: Optional[int] = None
    convergence_tol: float = 1e-3
    reference_amplitude: Any = None

    def __post_init__(self):
        rr, ra = self.reference_reflectance, self.reference_amplitude
        if isinstance(rr, str) and rr != "plane_wave":
            raise ValueError("MeasurementSetup: reference_reflectance must be 'plane_wave', None "
                             "or a real power reflectance")
        numeric_rr = rr is not None and not isinstance(rr, str)
        if numeric_rr:
            rv = np.asarray(rr)
            if np.iscomplexobj(rv) or not np.all(np.isfinite(rv)):
                raise ValueError("MeasurementSetup: a numeric reference_reflectance is a real, finite "
                                 "POWER reflectance; give the complex amplitude as reference_amplitude")
        if ra is not None:
            if not numeric_rr:
                raise ValueError("MeasurementSetup: reference_amplitude goes with a numeric "
                                 "reference_reflectance (= abs(reference_amplitude)**2); with "
                                 "'plane_wave' the amplitude comes from the reference model and with "
                                 "None the ratios are raw")
            av = np.asarray(ra, dtype=complex)
            rv = np.asarray(rr, dtype=float)
            # the same number written twice: equal to round-off (a few ulp of R <= 1)
            if not np.allclose(np.abs(av) ** 2, rv, rtol=1e-12, atol=0.0):
                raise ValueError("MeasurementSetup: reference_reflectance must equal "
                                 "abs(reference_amplitude)**2")
        elif numeric_rr and self.reference is not None and self.detection == "amplitude":
            raise ValueError(
                "MeasurementSetup: amplitude detection normalized to a numeric power reflectance: "
                "R fixes abs(r) of the reference but not its phase (a metal mirror has r close to -1), "
                "so the normalized amplitude would carry an unknown phase. Give the reference's "
                "complex amplitude (reference_amplitude=r_ref with reference_reflectance=abs(r_ref)**2), "
                "or reference_reflectance='plane_wave' (from the reference model), or None (the "
                "phase relative to the reference)")
        if self.detection not in ("power", "amplitude"):
            raise ValueError("MeasurementSetup: detection must be 'power' or 'amplitude'")
        if self.analyzer not in (None, "parallel", "crossed"):
            raise ValueError("MeasurementSetup: analyzer must be None, 'parallel' or 'crossed'")
        if self.analyzer is not None and self.detection == "amplitude":
            raise ValueError("MeasurementSetup: an analyzer applies to power detection; amplitude "
                             "detection already projects onto the input polarization")
        if self.scatter_collection not in (None, "specular", "all"):
            raise ValueError("MeasurementSetup: scatter_collection must be 'specular' or 'all'")
        if (self.parameters is not None and self.parameters.n_nodes > 1
                and self.detection == "power" and self.scatter_collection is None):
            raise ValueError(
                "MeasurementSetup: with fabrication scatter, say what the detector collects: "
                "scatter_collection='specular' (small aperture: |<r>|^2, the diffuse part is lost) "
                "or 'all' (objective / integrating detector: <|r|^2>)")
        if self.convergence_nodes is not None:
            if self.parameters is None or self.parameters.specs is None:
                raise ValueError("MeasurementSetup: convergence_nodes needs a parameter rule built "
                                 "from a 'normal' / 'uniform' distribution")

    def _pupil(self) -> PupilQuadrature:
        return self.pupil if self.pupil is not None else PupilQuadrature()

    def _ref_known_calls(self) -> int:
        if self.reference is None or not isinstance(self.reference_reflectance, str):
            return 0
        return len(_pols_needed(self._pupil_normal()))

    def _pupil_normal(self) -> PupilQuadrature:
        p = self._pupil()
        return PupilQuadrature(mode=p.mode, n_medium=p.n_medium, polarization_deg=p.polarization_deg)

    def plan(self, wavelengths_m) -> MeasurementPlan:
        """Cost of run() on this grid, without solving."""
        n_lam = int(np.atleast_1d(np.asarray(wavelengths_m)).size)
        p = self._pupil()
        n_ray_solves = len(set(_ray_keys(p)))
        pols = _pols_needed(p)
        n_par = self.parameters.n_nodes if self.parameters is not None else 1
        sample = n_ray_solves * len(pols) * n_par
        conv = 0
        if self.convergence_nodes is not None:
            conv = n_ray_solves * len(pols) * self.parameters.refine(self.convergence_nodes).n_nodes
        ref = (n_ray_solves * len(pols) + self._ref_known_calls()) if self.reference is not None else 0
        calls = sample + conv + ref
        return MeasurementPlan(n_wavelengths=n_lam, n_rays=p.n_rays, n_ray_solves=n_ray_solves,
                               polarizations=pols, n_parameter_nodes=n_par, n_sample_calls=sample,
                               n_convergence_calls=conv, n_reference_calls=ref,
                               n_model_calls=calls, n_solves=calls * n_lam)

    # -- the averaging core -------------------------------------------------------------------
    def _evaluate(self, model, lam, pupil, params: Optional[ParameterQuadrature], runner):
        """Solve every distinct (ray, pol, parameter node) and reduce to per-ray patch moments."""
        keys = _ray_keys(pupil)
        uniq = sorted(set(keys))
        pols = _pols_needed(pupil)
        pq = params if params is not None else ParameterQuadrature()
        jobs = [(t, ph, pol, pq.values(i)) for (t, ph) in uniq for pol in pols
                for i in range(pq.n_nodes)]
        outs = runner(_Call(model, lam), jobs)
        n_lam = lam.size
        table = {}
        any_cross = False
        for job, out in zip(jobs, outs):
            co, cross = _parse_output(out, n_lam)
            any_cross = any_cross or cross is not None
            table[(job[0], job[1], job[2], tuple(sorted(job[3].items())))] = (co, cross)
        n_rays, n_par = pupil.n_rays, pq.n_nodes
        zero = np.zeros(n_lam, dtype=complex)
        J = {k: np.zeros((n_par, n_rays, n_lam), dtype=complex) for k in ("ss", "pp", "sp", "ps")}
        for k, (t, ph) in enumerate(keys):
            for i in range(n_par):
                pk = tuple(sorted(pq.values(i).items()))
                if "s" in pols:
                    co, cross = table[(t, ph, "s", pk)]
                    J["ss"][i, k] = co
                    J["ps"][i, k] = cross if cross is not None else zero
                if "p" in pols:
                    co, cross = table[(t, ph, "p", pk)]
                    J["pp"][i, k] = co
                    J["sp"][i, k] = cross if cross is not None else zero
        a_s, a_p = pupil.polarization_amplitudes
        E_par, E_perp = _ray_fields(a_s[None, :, None], a_p[None, :, None], J)
        m_par, q_par = _patch_moments(E_par, pq.weights)          # (n_rays, n_lam)
        m_perp, q_perp = _patch_moments(E_perp, pq.weights)
        return m_par, q_par, m_perp, q_perp, any_cross

    def _detect(self, lam, pupil, moments):
        m_par, q_par, m_perp, q_perp, _ = moments
        W = pupil.weights(lam)                                     # (n_lam, n_rays)

        def power(par, perp):
            if self.analyzer == "parallel":
                return par
            if self.analyzer == "crossed":
                return perp
            return par + perp

        spec_k = power(np.abs(m_par) ** 2, np.abs(m_perp) ** 2)    # (n_rays, n_lam)
        inc_k = power(q_par, q_perp)
        specular = np.einsum("lk,kl->l", W, spec_k)
        incoherent = np.einsum("lk,kl->l", W, inc_k)
        amplitude = np.einsum("lk,kl->l", W, m_par)
        if self.detection == "amplitude":
            detected = np.abs(amplitude) ** 2
        elif self.scatter_collection == "all":
            detected = incoherent
        else:
            detected = specular
        return detected, amplitude, specular, incoherent

    def run(self, model, wavelengths_m, *, n_workers: Optional[int] = 1,
            map_fn: Optional[Callable] = None, verbose: bool = False,
            max_solves: Optional[int] = None) -> MeasuredSpectrum:
        """Average the model's plane-wave solves into the measured spectrum.

        n_workers : threads for the model calls (1 = serial; results are identical either way --
                    every call is independent and the reduction order is fixed). NGSolve's task
                    manager is process-global: for NGSolve-backed models keep n_workers = 1 and use
                    map_fn with a process pool and a picklable model.
        map_fn    : any map-like callable map_fn(f, jobs) -> results in order (overrides n_workers).
        max_solves: refuse (ValueError) before solving if the plan exceeds this many solves.
        verbose   : print the plan before solving."""
        lam = np.atleast_1d(np.asarray(wavelengths_m, dtype=float)).ravel()
        plan = self.plan(lam)
        if verbose:
            print("[measurement] " + plan.describe(), flush=True)
        if max_solves is not None and plan.n_solves > int(max_solves):
            raise ValueError("MeasurementSetup.run: the plan needs {} solves > max_solves={} ({})"
                             .format(plan.n_solves, max_solves, plan.describe()))
        pupil = self._pupil()
        if self.period_m is not None and pupil.na > 0.0:
            bad = lam / float(self.period_m) <= 2.0 * pupil.na
            if np.any(bad):
                warnings.warn(
                    "MeasurementSetup: at {} of {} wavelengths lambda/period <= 2 NA, so a "
                    "non-specular diffraction order of some illuminated direction enters the "
                    "collection pupil; this specular model omits it (and for a coherent beam it would "
                    "interfere with the specular light)".format(int(bad.sum()), lam.size),
                    UnreliableResultWarning, stacklevel=2)
        t0 = time.perf_counter()
        runner = self._runner(n_workers, map_fn)
        mom = self._evaluate(model, lam, pupil, self.parameters, runner)
        detected, amplitude, specular, incoherent = self._detect(lam, pupil, mom)
        notes = []
        if pupil.approximation:
            notes.append(pupil.approximation)
        if pupil.mode == "conical" and pupil.na > 0.0 and not pupil.azimuthal_invariant and not mom[4]:
            notes.append("cross-polarization (s<->p conversion) neglected: the model returned "
                         "co-polarized amplitudes only")
        if self.parameters is not None and self.parameters.n_nodes > 1:
            notes.append("fabrication scatter in the local-patch picture ({}; {} nodes)".format(
                self.parameters.rule, self.parameters.n_nodes))
        conv_delta = None
        if self.convergence_nodes is not None:
            mom2 = self._evaluate(model, lam, pupil, self.parameters.refine(self.convergence_nodes),
                                  runner)
            det2 = self._detect(lam, pupil, mom2)[0]
            conv_delta = float(np.max(np.abs(det2 - detected)))
            if conv_delta > self.convergence_tol:
                warnings.warn(
                    "MeasurementSetup: the parameter quadrature is not converged: changing to {} "
                    "nodes per dimension moves the detected signal by {:.3g} > {:.3g}".format(
                        self.convergence_nodes, conv_delta, self.convergence_tol),
                    UnreliableResultWarning, stacklevel=2)
        signal = detected
        ref_det = None
        if self.instrument is not None:
            signal = self.instrument.apply(lam, detected)
            if self.detection == "amplitude":
                notes.append("instrument applied to the detected power |r_coh|^2 only; the complex "
                             "amplitude is not convolved")
        if self.reference is not None:
            rmom = self._evaluate(self.reference, lam, pupil, None, runner)
            ref_det, ref_amp, _, _ = self._detect(lam, pupil, rmom)
            ref_sig = self.instrument.apply(lam, ref_det) if self.instrument is not None else ref_det
            known, known_amp = self._reference_known(lam, runner)
            signal = normalize_to_reference(signal, ref_sig, known)
            amplitude = amplitude / ref_amp * known_amp
        return MeasuredSpectrum(wavelengths_m=lam, signal=np.asarray(signal, dtype=float),
                                detected=np.asarray(detected, dtype=float), amplitude=amplitude,
                                specular=specular, incoherent=incoherent,
                                diffuse=incoherent - specular, reference_detected=ref_det,
                                convergence_delta=conv_delta, plan=plan, notes=tuple(notes),
                                elapsed_s=time.perf_counter() - t0)

    def _reference_known(self, lam, runner):
        rr = self.reference_reflectance
        if rr is None:
            return 1.0, 1.0
        if isinstance(rr, str):                     # 'plane_wave' (validated in __post_init__)
            pn = self._pupil_normal()
            mom = self._evaluate(self.reference, lam, pn, None, runner)
            setup_pw = replace(self, pupil=pn, parameters=None, detection="power",
                               convergence_nodes=None)
            det, amp, _, _ = setup_pw._detect(lam, pn, mom)
            return det, amp
        # a power reflectance fixes abs(r) only: the amplitude needs the complex reference_amplitude
        # (__post_init__ refuses amplitude detection without it; power detection reports NaN)
        if self.reference_amplitude is not None:
            return np.asarray(rr, dtype=float), np.asarray(self.reference_amplitude, dtype=complex)
        return np.asarray(rr, dtype=float), complex(np.nan, np.nan)

    @staticmethod
    def _runner(n_workers, map_fn):
        if map_fn is not None:
            return lambda f, jobs: list(map_fn(f, jobs))
        if n_workers is None or int(n_workers) <= 1:
            return lambda f, jobs: [f(j) for j in jobs]
        from concurrent.futures import ThreadPoolExecutor

        def _threaded(f, jobs):
            with ThreadPoolExecutor(max_workers=int(n_workers)) as pool:
                return list(pool.map(f, jobs))
        return _threaded


def measure(model, wavelengths_m, *, n_workers: Optional[int] = 1, map_fn: Optional[Callable] = None,
            verbose: bool = False, max_solves: Optional[int] = None, dry_run: bool = False,
            **setup_kwargs):
    """One-call form: MeasurementSetup(**setup_kwargs).run(model, wavelengths_m, ...). With
    dry_run=True it returns the MeasurementPlan (the number of solves) and solves nothing."""
    setup = MeasurementSetup(**setup_kwargs)
    if dry_run:
        return setup.plan(wavelengths_m)
    return setup.run(model, wavelengths_m, n_workers=n_workers, map_fn=map_fn, verbose=verbose,
                     max_solves=max_solves)


# ================================================================================================
# Model adapters
# ================================================================================================

class _ScalarModel:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, wavelengths_m, theta_deg=0.0, phi_deg=0.0, pol="s", **params):
        r = np.asarray(self.fn(wavelengths_m, theta_deg=theta_deg, phi_deg=phi_deg, **params),
                       dtype=complex)
        if pol == "s":
            return r
        if pol == "p":
            return -r
        raise ValueError("scalar_model: pol must be 's' or 'p'")


def scalar_model(fn: Callable) -> Callable:
    """Wrap a POLARIZATION-INDEPENDENT reflection fn(wavelengths_m, theta_deg, phi_deg, **params)
    -> r (the lab field ratio, e.g. a closed-form resonance) into the model protocol: r_s = r and
    r_p = -r (the library's p sign), so every ray's co-polarized return field is r and the
    cross-polarized one is 0 for any input polarization."""
    return _ScalarModel(fn)


class _DesignModel:
    def __init__(self, design, optical_solver, eps_by_region, geo, design_for):
        self.design = design
        self.optical_solver = optical_solver
        self.eps_by_region = eps_by_region
        self.geo = geo
        self.design_for = design_for

    def __call__(self, wavelengths_m, theta_deg=0.0, phi_deg=0.0, pol="s", **params):
        from dynameta.optics.tmm_reference import end_media_indices
        if pol not in ("s", "p"):
            raise ValueError("design_plane_wave_model: pol must be 's' or 'p'")
        if params:
            if self.design_for is None:
                raise ValueError("design_plane_wave_model: parameters {} given but no design_for "
                                 "callable to rebuild the Design".format(sorted(params)))
            d = self.design_for(**params)
        else:
            d = self.design
        opt = replace(d.optical, incidence_angle_deg=float(theta_deg), azimuth_deg=float(phi_deg),
                      polarization="y" if pol == "s" else "p")
        d = replace(d, optical=opt)
        out = []
        for lam in np.atleast_1d(np.asarray(wavelengths_m, dtype=float)):
            n_super, n_sub = end_media_indices(d, float(lam))
            eps = self.eps_by_region(float(lam), d) if callable(self.eps_by_region) else self.eps_by_region
            res = self.optical_solver(d, self.geo, eps, float(lam), n_super, n_sub)
            out.append(complex(res.r))
        return np.asarray(out)


def design_plane_wave_model(design, optical_solver: Callable, *, eps_by_region=None, geo=None,
                            design_for: Optional[Callable] = None) -> Callable:
    """A model callable backed by any run_pipeline optical solver
    fn(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult.

    Each call sets design.optical.incidence_angle_deg / azimuth_deg / polarization ('s' -> 'y', the
    rotated s-hat; 'p' -> 'p') and returns OpticalResult.r per wavelength (co-polarized; the
    solvers do not expose the conical cross-polarized amplitude). End media come from
    end_media_indices per wavelength. eps_by_region: None, a dict, or a callable
    (lambda_m, design) -> dict. design_for(**params) -> Design rebuilds the device for a
    fabrication-parameter value (e.g. a strip width). A solver that cannot do the requested
    incidence raises (the 2-D TM FEM refuses azimuth != 0 and 's': use mode='in_plane' with
    polarization_deg=0 there)."""
    return _DesignModel(design, optical_solver, eps_by_region, geo, design_for)


# ================================================================================================
# Item 6: fitting the broadening parameters
# ================================================================================================

@dataclass(frozen=True, eq=False)
class BroadeningFit:
    """Result of fit_broadening.

    params / stderr : best-fit values and 1-sigma standard errors (inf when a parameter sits in an
               exactly degenerate direction). covariance / correlation : over `names` (the varied
               parameters). chi2, dof, reduced_chi2 : with the given sigma (if none was given the
               covariance is scaled by reduced_chi2, i.e. the noise is estimated from the residual).
    null_combinations : parameter sets spanning a numerically singular Fisher direction: EXACTLY
               degenerate, only a combination is measured, at any signal-to-noise.
    undetermined : parameters not resolved from zero (|value| <= 2 stderr).
    degenerate : pairs (a, b, corr) with |corr| >= the threshold of which at least one is
               undetermined -- for THESE data the two cannot be told apart.
    correlated : pairs with |corr| >= the threshold that are nevertheless both resolved (the
               separation rests on the line SHAPE -- wings versus core -- so a model error there
               moves both).
    null_tests : {name: delta_chi2} of refitting with that parameter held at its null value (e.g.
               extra damping = 0); delta_chi2 < 4 means the data are consistent with the null (the
               observed broadening does not require that mechanism).
    profiles : {name: (values, delta_chi2)} profile-likelihood scans (other parameters refitted;
               delta_chi2 in units of the noise variance; < 1 = inside the 1-sigma interval).
    message  : the verdict in words."""
    names: Tuple[str, ...] = ()
    params: Dict[str, float] = field(default_factory=dict)
    stderr: Dict[str, float] = field(default_factory=dict)
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    correlation: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    chi2: float = float("nan")
    dof: int = 0
    reduced_chi2: float = float("nan")
    residual_rms: float = float("nan")
    null_combinations: Tuple[Tuple[str, ...], ...] = ()
    undetermined: Tuple[str, ...] = ()
    degenerate: Tuple[Tuple[str, str, float], ...] = ()
    correlated: Tuple[Tuple[str, str, float], ...] = ()
    null_tests: Dict[str, float] = field(default_factory=dict)
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    success: bool = False
    message: str = ""


def _least_squares(resid, u0, lo, hi, max_nfev):
    from scipy.optimize import least_squares
    # gtol disabled (None) for the reason documented at analysis._fano_varpro: scipy's gtol is an
    # absolute bound and would stop early on small-amplitude spectra.
    return least_squares(resid, u0, bounds=(lo, hi), method="trf", ftol=1e-12, xtol=1e-12,
                         gtol=None, max_nfev=max_nfev)


def fit_broadening(wavelengths_m, measured, model: Callable, p0: Mapping[str, float], *,
                   sigma=None, vary: Optional[Sequence[str]] = None,
                   bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
                   scales: Optional[Mapping[str, float]] = None, profile: Sequence[str] = (),
                   null_values: Optional[Mapping[str, float]] = None,
                   profile_points: int = 9, profile_span: float = 3.0,
                   degeneracy_threshold: float = 0.95, max_nfev: int = 4000) -> BroadeningFit:
    """Fit broadening parameters (NA, scatter width, instrument width, ...) together with an
    intrinsic damping parameter, and say whether the data can tell them apart.

    model(wavelengths_m, **params) -> predicted measured spectrum (typically a MeasurementSetup run,
    or a fast surrogate of one). p0: start values of ALL parameters; vary: the subset fitted (others
    held at p0). sigma: per-point (or scalar) noise standard deviation (None: estimated from the
    residual). bounds: {name: (lo, hi)}. scales: typical magnitudes (default |p0| or 1) -- the fit
    and the finite-difference Jacobian run in scaled variables so nm widths and O(1) NAs mix.

    The verdict, from the Fisher matrix F = J^T J / sigma^2 at the optimum (covariance F^-1):
      * a direction with eigenvalue < 1e-10 x the largest is an EXACT degeneracy (only a
        combination is measured; those parameters get infinite errors);
      * a parameter with |value| <= 2 stderr is not resolved;
      * a pair with |correlation| >= degeneracy_threshold is DEGENERATE FOR THESE DATA when at
        least one of the two is unresolved, and is reported as correlated-but-resolved otherwise.
        Correlation alone is not the verdict: a Gaussian and a Lorentzian width are typically
        correlated at |corr| ~ 0.95 on a single dip, yet both can be pinned to a few percent when
        the wings are sampled at good signal-to-noise (tests/test_measurement.py);
      * null_values={name: value} refits with that parameter held at the value (e.g. extra
        intrinsic damping = 0) and reports delta chi2 -- the direct answer to "do the data require
        this mechanism, or can the other broadening explain the line?" (delta chi2 < 4: no);
      * `profile` names get a profile-likelihood scan over +- profile_span standard errors.
    Why this is not analysis.fano_fit / lorentzian_fit: those reduce a line to ONE width, so by
    construction they cannot separate broadening mechanisms; they remain the right tool to quote
    the measured linewidth itself."""
    lam = np.asarray(wavelengths_m, dtype=float).ravel()
    y = np.asarray(measured, dtype=float).ravel()
    if lam.shape != y.shape:
        raise ValueError("fit_broadening: wavelengths and measured spectrum differ in length")
    names_all = list(p0)
    vary = list(names_all) if vary is None else list(vary)
    unknown = [v for v in vary if v not in p0]
    if unknown:
        raise ValueError("fit_broadening: vary names {} are not in p0".format(unknown))
    k = len(vary)
    if k < 1 or lam.size <= k:
        raise ValueError("fit_broadening: need 1 <= n_varied < n_points")
    sig = np.ones_like(y) if sigma is None else np.broadcast_to(np.asarray(sigma, float), y.shape)
    if np.any(~(sig > 0)):
        raise ValueError("fit_broadening: sigma must be > 0")
    scale = np.array([float((scales or {}).get(n, abs(p0[n]) if p0[n] != 0 else 1.0)) for n in vary])
    lo = np.array([((bounds or {}).get(n, (-np.inf, np.inf))[0]) for n in vary], float) / scale
    hi = np.array([((bounds or {}).get(n, (-np.inf, np.inf))[1]) for n in vary], float) / scale
    fixed = {n: float(p0[n]) for n in names_all if n not in vary}

    def unpack(u, held=None):
        p = dict(fixed)
        p.update({n: float(ui * s) for n, ui, s in zip(vary, u, scale)})
        if held:
            p.update(held)
        return p

    def make_resid(held=None, idx=None):
        def resid(u):
            uu = u if idx is None else np.insert(u, idx, 0.0)
            return (np.asarray(model(lam, **unpack(uu, held)), float).ravel() - y) / sig
        return resid

    u0 = np.clip(np.array([float(p0[n]) for n in vary]) / scale, lo, hi)
    sol = _least_squares(make_resid(), u0, lo, hi, max_nfev)
    r = sol.fun
    chi2 = float(np.sum(r * r))
    dof = lam.size - k
    s2 = 1.0 if sigma is not None else chi2 / dof

    def refit_with(name, value):
        """delta chi2 (noise-variance units) with `name` held at `value`, the others refitted."""
        i = vary.index(name)
        if k == 1:
            rr = make_resid({name: float(value)})(np.array([0.0]))
        else:
            res_i = _least_squares(make_resid({name: float(value)}, i), np.delete(sol.x, i),
                                   np.delete(lo, i), np.delete(hi, i), max_nfev)
            rr = res_i.fun
        return (float(np.sum(rr * rr)) - chi2) / s2

    J = np.asarray(sol.jac, dtype=float)
    Fm = J.T @ J
    ev, V = np.linalg.eigh(Fm)
    cut = 1e-10 * max(float(ev.max()), 1e-300)
    good = ev > cut
    inv = np.zeros_like(Fm)
    for e, v, g in zip(ev, V.T, good):
        if g:
            inv += np.outer(v, v) / e
    cov = inv * s2 * np.outer(scale, scale)
    null = []
    for e, v, g in zip(ev, V.T, good):
        if not g:
            null.append(tuple(n for n, c in zip(vary, v) if abs(c) > 0.2))
    in_null = {n for combo in null for n in combo}
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    stderr = {n: (float("inf") if n in in_null else float(s)) for n, s in zip(vary, std)}
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)
    corr[~np.isfinite(corr)] = np.nan
    best = unpack(sol.x)
    undetermined = tuple(n for n in vary if not (abs(best[n]) > 2.0 * stderr[n]))
    degenerate, correlated = [], []
    for i in range(k):
        for j in range(i + 1, k):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) >= degeneracy_threshold:
                pair = (vary[i], vary[j], float(c))
                if vary[i] in undetermined or vary[j] in undetermined:
                    degenerate.append(pair)
                else:
                    correlated.append(pair)
    null_tests = {}
    for name, value in (null_values or {}).items():
        if name not in vary:
            raise ValueError("fit_broadening: null_values name {!r} is not varied".format(name))
        null_tests[name] = refit_with(name, value)
    profiles = {}
    for name in profile:
        if name not in vary:
            raise ValueError("fit_broadening: profile name {!r} is not varied".format(name))
        span = stderr[name] if np.isfinite(stderr[name]) and stderr[name] > 0 else 0.5 * abs(best[name])
        vals = best[name] + np.linspace(-profile_span, profile_span, int(profile_points)) * span
        profiles[name] = (vals, np.asarray([refit_with(name, v) for v in vals]))
    lines = []
    if null:
        lines.append("EXACTLY DEGENERATE: the data constrain only a combination of {} -- these "
                     "cannot be separated by any amount of signal-to-noise with this model".format(
                         "; ".join("(" + ", ".join(c) + ")" for c in null)))
    for a, b, c in degenerate:
        lines.append("{} and {} are degenerate for these data (correlation {:+.3f}, not both "
                     "resolved): they cannot be told apart".format(a, b, c))
    for a, b, c in correlated:
        lines.append("{} and {} are correlated ({:+.3f}) but both resolved ({:.4g} +- {:.2g}, {:.4g} "
                     "+- {:.2g}); the separation rests on the line shape".format(
                         a, b, c, best[a], stderr[a], best[b], stderr[b]))
    rest = [n for n in undetermined if n not in in_null]
    if rest:
        lines.append("not resolved from zero (|value| <= 2 sigma): {}".format(", ".join(rest)))
    for name, d in null_tests.items():
        if d < 4.0:
            lines.append("the data are consistent with {} = {:g} (delta chi2 {:.2f} < 4): the line "
                         "does not require it".format(name, null_values[name], d))
        else:
            lines.append("{} = {:g} is excluded (delta chi2 {:.1f}, ~{:.1f} sigma)".format(
                name, null_values[name], d, math.sqrt(d)))
    if not (null or degenerate or rest):
        lines.insert(0, "all {} parameters are distinguished".format(k))
    if sigma is not None and dof > 0 and chi2 / dof > 4.0:
        lines.append("reduced chi2 = {:.2f}: the model does not describe the data at the stated "
                     "noise".format(chi2 / dof))
    return BroadeningFit(names=tuple(vary), params=best, stderr=stderr, covariance=cov,
                         correlation=corr, chi2=chi2, dof=dof, reduced_chi2=chi2 / dof,
                         residual_rms=float(np.sqrt(np.mean((r * sig) ** 2))),
                         null_combinations=tuple(null), undetermined=undetermined,
                         degenerate=tuple(degenerate), correlated=tuple(correlated),
                         null_tests=null_tests, profiles=profiles, success=bool(sol.success),
                         message="; ".join(lines))
