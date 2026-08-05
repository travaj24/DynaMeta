"""Scalar gain-BPM: a split-step paraxial beam-propagation solver for the SIGNAL FIELD E(x,y,z)
inside a doped fiber, gain-coupled to the radially-resolved local balance of `transverse.py` and
(opt-in, quasi-statically) to `thermal.py`'s heat / lens model. Feature 2 of
docs/fiber_transverse_grounding_2026_07_29.md; spec section 13 of docs/fiber_amp_model_spec.md.

WHY A FIELD SOLVER. `steady_state.py` and `transverse.py` both propagate POWERS: each channel
carries a FIXED transverse profile i_k(r,phi) and the gain it sees is an overlap integral. A BPM
carries no such assumption. Quoting the design statement this module is built on (A. V. Smith and
J. J. Smith, "Increased mode instability thresholds of fiber amplifiers by gain saturation", Opt.
Express 21(13):15168 (2013), sec. 1):

    "In the model of Ward et al. the increase of signal power in a mode due to laser gain is found
    from the overlap of the local gain g(x,y,z) with the field of that mode. In our BPM model laser
    gain increases the total signal field locally and is then apportioned among the modes
    automatically by diffraction in the presence of the core index step."

So the confinement factor Gamma is an OUTPUT here, never an input -- there is no Gamma anywhere in
(F2.7) below -- and gain guiding, mode reshaping and diffractive redistribution are computed rather
than assumed.

GOVERNING EQUATIONS (dossier numbering).

  (F2.2)  scalar paraxial envelope, in the repo's exp(-i omega t) convention with the carrier
          exp(+i k0 n_ref z) factored out:

              dE/dz = (i/(2k)) (d2/dx2 + d2/dy2) E + (i k0^2/(2k)) (n^2 - n_ref^2) E + (g/2) E,
              k = k0 n_ref,   k0 = 2 pi / lambda.

          |E|^2 is an INTENSITY [W m^-2]: the field normalization contract of this module is
          INT |E|^2 dA = P_signal [W], i.e. [E] = sqrt(W)/m. launch_field() enforces it and
          power_W() reads it back.

  (F2.4)/(F2.5)  the symmetrized (Strang) split D(dz/2) N(dz) D(dz/2):

              D(dz):  E_hat(kx,ky) *= exp( -i (kx^2 + ky^2) dz / (2 k0 n_ref) )
              N(dz):  E(x,y)      *= exp( i k0^2 (n^2 - n_ref^2) dz / (2 k0 n_ref) ) exp( g dz/2 )

          D is EXACTLY unitary (it is a pure phase in Fourier space), so every power change in the
          march lives in N and in the boundary mask -- which is what makes the accounting contract
          below auditable. The sign of the `i` in D is the repo's, NOT the canonical BPM papers':
          M. D. Feit and J. A. Fleck, Appl. Opt. 17(24):3990 (1978) and K. Okamoto, "Fundamentals
          of Optical Waveguides" ch. 7 write the same operator in the exp(+i omega t) engineering
          convention, which flips the sign of every i. Their equation numbers were NOT re-verified
          against the primary text for this build (both are paywalled / JS-gated to the fetch tool
          -- dossier sec. 2.1 and open item 2), so they are cited here for STRUCTURE only. The
          binding internal anchor is soa/transverse_bpm.py, whose diffraction leg is
          `A_k *= exp(-1j kx^2 dz/(2k))`; this module's D matches it term for term, and gates
          C/D (split order, quadratic-duct period) pin the factor and the sign numerically.

  GAIN CONVENTION -- the factor-of-2 trap. With (F2.2), d|E|^2/dz = g |E|^2, so `g` is the
  INTENSITY (power) gain per metre and the per-step AMPLITUDE factor is exp(g dz/2). The
  sigma-form g = sigma_e n2 - sigma_a (nt - n2) already IS that intensity coefficient (it is
  alpha/g* divided by Gamma), so there is no second factor of 2 hiding in it; the 2 belongs in
  the amplitude exponent and nowhere else. Gate B of validation/fiber_gain_bpm.py pins
  P_out/P_in = exp(g L) on a uniform-gain fixture.

  (F2.7)  local saturable gain, from the LOCAL intensities via the local balance:

              nbar2(x,y) from (F1.1) with I_s(x,y) = |E(x,y)|^2 and the pump intensity I_p(x,y)
              g(x,y) = nt(x,y) [ (sigma_a_s + sigma_e_s) nbar2(x,y) - sigma_a_s ] - l_s

          (F1.1) itself is NOT re-implemented here: this module calls
          transverse.ResolvedFiberAmplifier._nbar2_nodes, the package's single home for the
          balance, with the BPM's per-pixel intensities in place of its quadrature-node ones.

  (F2.8)  pump depletion, marched on the same z-step as the field (forward pumps only):

              dP_p/dz = -P_p { INT [ sigma_a_p (nt - n2) - sigma_e_p n2 ] i_p dA + l_p }

          Note the asymmetry that IS transverse spatial hole burning: the pump integral is
          AREA-weighted by its own (flat, for a cladding pump) profile, while the signal is
          weighted by its own field. A cladding pump is flat, I_p = P_p/A_clad over r <= R_clad,
          taken ANALYTICALLY rather than renormalized on the grid -- so a window that does not
          reach the inner cladding does not inflate the pump (it would in transverse.py, whose
          grid normalization is why that solver refuses such a grid).

  (F2.9)  quantum-defect heat source [PINNED: Smith & Smith Eq. (2)]:

              Q(x,y) = nt(x,y) [(nu_p - nu_s)/nu_p] [ sigma_a_p - (sigma_a_p+sigma_e_p) nbar2 ] I_p
                     = (quantum defect) x (net absorbed pump power density)          [W m^-3]

          (nu_p - nu_s)/nu_p is exactly thermal.quantum_defect_fraction. Background loss and
          untracked spontaneous emission are NOT in Q here (thermal.heat_load_per_m owns that
          bookkeeping for the power solvers); with photodarkening on, its dissipated power would
          have to be added -- in Yb it can EXCEED the quantum-defect heat (Jauregui/Limpert/
          Tunnermann 2020 sec. 4), and this module does not model it.

  (F2.10)-(F2.15) thermal lens -> quadratic duct. The in-core temperature rise of
          thermal.radial_temperature_rise is exactly parabolic, so dn_th = (dn/dT) dT(r) is a
          parabolic index perturbation whose dioptric power per unit length is
          thermal.thermal_lens_focal_power_per_m's D'. Reading (F2.2) as a 2-D harmonic
          oscillator gives alpha = sqrt(D'), matched 1/e field radius w_m = sqrt(2/(k0 n0 alpha)),
          self-imaging period z_p = 2 pi/alpha, first spot-size extremum of a mismatched launch
          at z_p/4 (quadratic_duct_radius_m / quadratic_duct_period_m below).

ABSORBING BOUNDARY AND THE POWER-ACCOUNTING CONTRACT (dossier sec. 2.5). The FFT grid is periodic,
so radiated light must be absorbed before it wraps. A super-Gaussian amplitude mask
M = exp(-[(rho - rho_abs)/w_abs]^m_sg) outside rho_abs is applied after each full step, and the
power it eats is ACCUMULATED AND REPORTED, never silently dropped:

    P_signal(L) = P_signal(0) + medium_exchange - absorbed_boundary        (to marcher round-off)

with medium_exchange = SUM_steps INT (exp(g dz) - 1) |E|^2 dA. That identity is not a tautology:
the increments are computed from g and the entering field, so it closes only if D really is
unitary and the index operator really is a pure phase (gates A and G).

The dossier's bar for a converged window is absorbed_boundary/P_signal(0) < 1e-6, and on a smooth
profile that is easy to meet. On a STEP-INDEX fiber it usually is not, and the reason is worth
knowing before reading the number as a window-size verdict: most of what the boundary eats there
is radiation shed by the DISCRETIZED index step (dossier sec. 2.3 -- the discontinuity's high-|k|
tail is propagated with the parabolic dispersion instead of sqrt(k^2-K^2) and re-aliased every
step), not the guided mode leaking to the window edge. MEASURED on the validation fixture over
2 cm: 3.6e-2 of the launch with a hard step, 1.2e-4 with a 2-cell erf taper at dx = a/16, and
3.2e-3 with the SAME 2 cells at dx = a/32 -- i.e. refining dx while holding the taper in CELLS
makes it worse, because the taper is an anti-aliasing filter whose physical width is s*dx. The
absorbed fraction is always reported (`meta["absorbed_fraction"]`); it is a diagnostic to read,
not a knob the solver silently turns.

NUMBERED SCOPE LIMITATIONS OF THE NUMERICS (the physics scope is further down). These are not
tuning advice; they are places where the discretization is not a discretization of anything.

  1. THE ABSORBER IS A PER-STEP MASK, so it is a dz-DEPENDENT operator with NO dz -> 0 limit.
     The mask multiplies the field once per full step, which makes its strength per-STEP rather
     than per-metre: halving dz doubles how often it acts, and a converged-in-dz answer is not
     what the sequence approaches. MEASURED (verifier, 2026-08-04) by parking light in the
     absorber annulus and refining only dz: P_out/P_in fell 0.612 -> 0.521 -> 0.439 -> 0.372 ->
     0.317 -> 0.271 over nz = 2 / 4 / 8 / 16 / 32 / 64 on a FIXED physical length, monotonically
     and with no sign of a limit. That is the honest worst case, not the shipped one: with the
     defaults and a window that holds the mode, the absorbed fractions of every gate in
     validation/fiber_gain_bpm.py are <= 1.2e-4 and dz-stable, and the dz-refinement gate (C) runs
     with absorber=False so the order measurement never touches this operator at all. A run whose
     absorbed_fraction crosses the dossier's converged-window bar of 1e-6 WARNS (UserWarning) for
     exactly this reason -- read it as "your window is too small", not as a knob.
  2. dx AND dz ARE NOT INDEPENDENT KNOBS: refining the transverse grid TIGHTENS the longitudinal
     step. The highest transverse frequency the grid carries, kx_max = pi/dx, accumulates a
     diffraction phase pi^2 dz/(2 k dx^2) per step, so the D leg aliases at the corner of the
     k-grid unless dz <= 2 k dx^2/pi -- and the N leg re-scatters into that corner every step, so
     the aliased phase does not average out. suggested_step_m() carries the bound as its third
     criterion and sampling_report() reports it as `transverse_nyquist_dz_m`. Stated honestly, it
     is a CONSERVATIVE GUARD rather than a fit to a measured failure: on the gate-F step-index
     fixture, holding the erf taper at a fixed PHYSICAL width and refining N at fixed n_steps = 400
     moves the cross-solver deviation not at all (-0.00932 / -0.00957 / -0.00926 / -0.00939 /
     -0.00931 dB at N = 64 / 96 / 128 / 192 / 256, verifier re-measurement 2026-08-04), because
     that fixture's guided field never populates the corner of the k-grid. The apparent "refining
     dx makes it worse" that this criterion was proposed to explain is a DIFFERENT effect: it only
     appears when the taper is held at a fixed number of CELLS, i.e. when refining dx physically
     REMOVES the anti-aliasing smoothing (limitation 3).
  3. THE erf TAPER IS A PHYSICAL WIDTH, NOT A CELL COUNT. index_taper_cells is s in
     s*dx, so holding s while refining dx shrinks the smoothing and the discretized index step
     sheds more: boundary-absorbed fraction 1.21e-4 -> 1.37e-4 -> 3.23e-3 at N = 96 / 128 / 192
     with s = 2. Every dx-refinement study in this module therefore scales s as 1/dx; a series run
     with s fixed measures the taper being removed, not the grid converging.

SCOPE (honest). STEADY-STATE, SCALAR, PARAXIAL, ONE FORWARD-PROPAGATING SIGNAL FIELD, forward
pumps only. No time axis, no polarization, no ASE, no counter-propagation. Every one of those is
REFUSED at construction with the physics reason rather than silently ignored (see _refuse).

TMI (transverse mode instability) IS OUT OF SCOPE, and not by accident of implementation. Per
C. Jauregui, J. Limpert, A. Tunnermann, "Transverse mode instability", Adv. Opt. Photon.
12(2):429 (2020) sec. 3, the accepted mechanism is: two modes beat -> the modal interference
pattern (MIP) is imprinted on the inversion -> quasi-periodic quantum-defect heating -> a thermally
induced refractive index grating (RIG) with exactly the right period to couple the two modes. But
"a second condition needs to be fulfilled to actually enable energy transfer between the transverse
modes: a phase shift between the MIP and the RIG". A STEADY-STATE model computes the temperature as
the instantaneous steady solution of the heat equation driven by the instantaneous irradiance
pattern, i.e. it puts the RIG exactly IN PHASE with the MIP -- and an in-phase index grating
transfers EXACTLY ZERO net energy between the modes. The missing ingredient is a time axis (an
assumed inter-mode frequency offset, or the finite thermal response of the core to a fluctuating
MIP); the experimental fluctuation band is 100 Hz to a few kHz with ms build-up times, none of
which a z-only model represents. This module therefore CANNOT predict a TMI threshold, onset
frequency, or the chaotic beam fluctuations, and any apparent "mode coupling" it shows is gain
competition, not TMI. What it CAN deliver is the input every TMI model needs: the saturated,
transversally resolved inversion and the resulting heat profile. For threshold ESTIMATES see
nonlinear_limits.tmi_threshold_W (a scaling-law estimator, calibrated, not a first-principles
solve).

READING TSHB OUT OF A COHERENT SOLVER -- why the obvious test is the wrong one (verifier,
2026-08-04). Transverse spatial hole burning is usually stated as "the saturated gain favours the
mode that overlaps the burnt hole least", and the tempting way to look for it in a field solver is
to launch LP01 + LP11 and watch the LP11 POWER FRACTION grow. That expectation is NOT valid for a
coherent solver, and the reason is structural rather than numerical. A coherent field carries a
MODAL-INTERFERENCE cross term 2 Re(c01 c11*) psi01 psi11 in |E|^2 that no power (incoherent) model
has, and the saturable medium turns it into a GAIN GRATING delta_g. Because psi11 is odd in the
azimuth and psi01 even, delta_g is odd at FIRST order in the cross-term amplitude, so its diagonal
projections <psi_m| delta_g |psi_m> vanish there and it lives in the OFF-DIAGONAL element
G_x = <psi01| delta_g |psi11>, which couples the two amplitudes. Feeding that back gives a cross
contribution G_x sqrt(P01 P11) to d P01/dz and to d P11/dz -- the SAME ABSOLUTE power rate for both
modes, hence fractionally larger for whichever one is WEAKER. Its effect on the LP11 power FRACTION
is therefore proportional to (P01 - P11): it VANISHES EXACTLY at a 50/50 split and REVERSES sign
across it. MEASURED on the gate-J fixture at 200 W total, LP01 fraction 0.10 / 0.30 / 0.50 / 0.70 /
0.90: d(f11)/dz from the cross term = +1.43e-1 / +1.74e-1 / 0.0 / -1.84e-1 / -1.58e-1 per metre,
with G_x itself never small (-0.60 to -1.08 1/m). A growing or shrinking LP11 fraction therefore
measures the launch ratio at least as much as it measures hole burning. (The SECOND-order, even
part of delta_g does shift the diagonals once the two mode powers are comparable, and by a lot: at
a 70/30 split the coherent field's modal gains differ from the channel model's by up to 1.1 1/m,
against 2.4e-3 for a single-mode field. That is the same statement from the other side -- a
coherent two-mode field's modal gains are not the incoherent model's either.)

The statement of TSHB that IS well posed here, and the one gate J of validation/fiber_gain_bpm.py
pins, is the ordering of the LOCAL MODAL GAINS seen by a WEAK LP11 probe: project the solver's own
local_gain_per_m field (F2.7) onto the mode intensities and g01 - g11 must REVERSE SIGN as the LP01
power saturates the medium. Measured there: +0.0920 1/m unsaturated -> -0.2431 at 100 W -> -0.6673
at 1200 W, against the channel model's resolved modal gains to 1.4e-3 1/m at every point.

Pure numpy/scipy; SI units; ASCII only; exp(-i omega t) with propagation exp(+i beta z).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
# scipy.fft, not numpy.fft: the operator is identical (same normalization, same sign convention as
# soa/transverse_bpm.py -- see _diffract), but pocketfft through scipy measured 2.0x faster than
# numpy.fft on the 64^2-128^2 grids this module marches thousands of times (0.93 vs 1.88 ms per
# round trip at 128^2, this box). Nothing else changes; the gates below are the proof.
import scipy.fft as _sfft
from scipy.special import erf

from dynameta.constants import H_PLANCK
from dynameta.optics.fiber_amp.lma import (LPMode, mode_field, one_over_e_radius_m,
                                           solve_lp_modes)
from dynameta.optics.fiber_amp.rare_earth import ChannelSet
from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
from dynameta.optics.fiber_amp.steady_state import AseBand, Pump, Signal
from dynameta.optics.fiber_amp.thermal import (DN_DT_SILICA, ThermalModel,
                                               quantum_defect_fraction, radial_temperature_rise)
# The local balance (F1.1) has EXACTLY ONE implementation in this package -- Feature 1's
# _nbar2_nodes -- and this module reuses it instead of re-typing the formula (which is how the
# two would silently diverge). Its signature is grid-specific (a coefficient bundle + a channel
# power vector), so it is WRAPPED, not copied: _medium() below hands it a bundle whose `idope`
# rows are the BPM's per-pixel INTENSITIES with a unit power vector, which is exact because
# I_k = P_k i_k. The function never touches `self`, so it is bound here as a plain function; if
# that ever stops being true this call fails loudly on None rather than answering wrongly.
from dynameta.optics.fiber_amp.transverse import V_LP11_CUTOFF, ResolvedFiberAmplifier
from dynameta.optics.fiber_amp.waveguide import FiberSpec, mode_field_radius_m

__all__ = ["GainBPM", "BPMResult", "ThermalLoop", "quadratic_duct_radius_m",
           "quadratic_duct_period_m"]

_LOCAL_BALANCE = ResolvedFiberAmplifier._nbar2_nodes

# Amplitude-exponent ceiling for the saturable-gain leg (the soa/transverse_bpm.py guard): an
# unsaturated fixture with an absurd dz would otherwise overflow exp() into a RuntimeWarning,
# which the repo's filterwarnings=["error"] turns into a test failure far from the real cause.
_EXP_CEILING = 100.0

# The dossier's B11 converged-window bar. Above it the per-STEP absorber (scope limitation 1 in
# the module docstring) is doing enough work that the answer carries a dz-dependent operator, so
# the run WARNS instead of quietly reporting the number.
_ABSORBED_FRACTION_BAR = 1e-6


# ============================ quadratic-duct oracles (F2.11)-(F2.15) ====================

def quadratic_duct_radius_m(focal_power_per_m2, lambda_m: float, n0: float):
    """Matched (propagation-invariant) 1/e FIELD radius of a parabolic index duct -- dossier
    (F2.12): w_m = sqrt(2 / (k0 n0 alpha)) with alpha = sqrt(D') and D' the dioptric power per
    unit length that thermal.thermal_lens_focal_power_per_m returns [1/m^2].

    DERIVED in the dossier by reading (F2.2) with n^2 - n0^2 = -n0^2 alpha^2 r^2 as the 2-D
    quantum harmonic oscillator (z <-> t, "mass" k0 n0, angular frequency alpha); cross-checked
    against the textbook GRIN ray matrix [[cos aL, sin(aL)/a], [-a sin aL, cos aL]], whose focal
    power alpha sin(alpha L) -> alpha^2 L reproduces D' identically in the thin limit. A Gaussian
    launched at exactly w_m propagates without breathing; any other width breathes with period
    z_p/2 (quadratic_duct_period_m)."""
    alpha = np.sqrt(np.asarray(focal_power_per_m2, float))
    k0 = 2.0 * np.pi / float(lambda_m)
    out = np.sqrt(2.0 / (k0 * float(n0) * alpha))
    return out if out.ndim else float(out)


def quadratic_duct_period_m(focal_power_per_m2):
    """Field self-imaging period of a parabolic duct -- dossier (F2.13): z_p = 2 pi / sqrt(D').
    The SPOT-SIZE (breathing) period is z_p/2 and the first spot-size extremum of a mismatched
    launch is at z_p/4 (F2.14)/(F2.15). Depends on the duct alone -- not on the wavelength, not on
    the launch -- which is what makes it such a tight oracle for the diffraction and index legs
    simultaneously."""
    out = 2.0 * np.pi / np.sqrt(np.asarray(focal_power_per_m2, float))
    return out if out.ndim else float(out)


# ============================ opt-in thermal loop =======================================

@dataclass(frozen=True)
class ThermalLoop:
    """Opt-in quasi-static thermal feedback for GainBPM (default: OFF -- pass thermal=None and the
    solve is byte-identical to an athermal one).

    The loop is  E -> n2 -> Q (F2.9) -> dT(r) -> dn = (dn/dT) dT -> n(x,y) -> E,  iterated to
    convergence. dT(r) comes from thermal.radial_temperature_rise (Brown & Hoffman, IEEE JQE
    37:207 (2001)) -- the SAME machinery the power solvers use, not a re-derivation: heat is
    generated in the core, conducts out through core and cladding and leaves by convection at
    b_outer_m. Because that solution is LINEAR in the heat load, the z-dependent profile is stored
    as one shape function [K per (W/m)] times the per-step heat Q(z) [W/m], so a z-varying thermal
    lens costs one extra array, not one per step.

    WHAT THE COUPLING THROWS AWAY, stated because it is the loop's real approximation and not
    visible in its inputs. Only the z-RESOLVED heat survives: (F2.9)'s radial heat density
    Q(x,y,z) [W/m^3] is integrated to a single scalar Q(z) [W/m] before it is handed to
    radial_temperature_rise, whose source term is UNIFORM over the heated disc. The RADIAL SHAPE
    of the heat is therefore discarded, and it is not flat -- on the shipped amplifier fixture
    (1 kW cladding pump, 20 W signal, N = 96) the quantum-defect heat density measures 2.73x the
    disc mean on axis and 0.10x at the dopant edge, a factor of 26 across the disc (verifier,
    2026-08-04). The second discarded fact is geometric:
    the heated radius passed to the temperature solution is fiber.core_radius_m regardless of
    where the dopant actually is, so a CONFINED dopant (b_dope < a) is still modelled as heating
    the full core. Both errors live in the SHAPE of dT inside the core, which is exactly where the
    duct parameter alpha^2 = 2 dn_drop/(a^2 n) is read -- so treat the lens strength as a
    core-averaged estimate, not a resolved thermal profile. Neither is a numerical shortcut that a
    finer grid removes; fixing them means a radially resolved source in thermal.py.

    b_outer_m = outer (glass/coating) radius of the conduction stack [m]; model = the conductivity
    / convection / coolant stack; dndt_per_K = thermo-optic coefficient (fused silica 1.2e-5);
    max_iter / tol_K = iteration cap and convergence bar on the peak temperature change between
    iterates; relax = under-relaxation on the heat profile (1.0 = none).

    SCOPE: quasi-static and z-local. There is no transverse heat FLOW along z, no transient, and
    the radial solution assumes azimuthal symmetry of the heat load -- so an azimuthally structured
    MIP is averaged into its radial profile. That last one is not a numerical shortcut, it is the
    same steady-state boundary that makes TMI unreachable (module docstring)."""
    b_outer_m: float
    model: ThermalModel = ThermalModel()
    dndt_per_K: float = DN_DT_SILICA
    max_iter: int = 8
    tol_K: float = 0.5
    relax: float = 1.0

    def __post_init__(self):
        if not (self.b_outer_m > 0.0):
            raise ValueError("ThermalLoop: b_outer_m must be > 0")
        if int(self.max_iter) < 1:
            raise ValueError("ThermalLoop: max_iter must be >= 1")
        if not (0.0 < self.relax <= 1.0):
            raise ValueError("ThermalLoop: relax must be in (0, 1]")


# ============================ result ====================================================

@dataclass
class BPMResult:
    """One gain-BPM march. Powers are in W, lengths in m, the field in sqrt(W)/m (so
    INT |E|^2 dA = P_signal -- the normalization contract of the module docstring).

    The energy ledger (dossier sec. 2.5) is the contract to read first:
        P_signal_W = P_signal_W[0] + medium_exchange_W - absorbed_boundary_W
    holds at EVERY sample to marcher round-off; energy_residual_W is that identity's residual,
    reported rather than asserted so a caller can see it. medium_exchange_W is the cumulative
    stimulated exchange (net of background loss) the medium put INTO the field;
    absorbed_boundary_W is the cumulative power the super-Gaussian boundary mask removed."""
    z_m: np.ndarray                      # (M,) sample positions
    P_signal_W: np.ndarray               # (M,) power in the field
    P_pump_W: np.ndarray                 # (K_p, M) pump powers
    E_out: np.ndarray                    # (N, N) complex output field [sqrt(W)/m]
    x_m: np.ndarray                      # (N,) transverse axis (both x and y)
    absorbed_boundary_W: np.ndarray      # (M,) cumulative boundary-absorbed power
    medium_exchange_W: np.ndarray        # (M,) cumulative net stimulated exchange
    energy_residual_W: np.ndarray        # (M,) ledger residual (should be ~0)
    spot_radius_m: np.ndarray            # (M,) sqrt(2 <r^2>) of the intensity (= w for a Gaussian)
    axis_intensity_W_m2: np.ndarray      # (M,) |E|^2 on axis
    heat_per_m_W: np.ndarray             # (M,) quantum-defect heat load Q(z) [W/m]
    mode_amp_sqrtW: Optional[np.ndarray] # (n_modes, M) complex modal amplitudes, |c|^2 = W
    fields: Optional[np.ndarray]         # (M, N, N) complex fields, if store_fields
    bpm: "GainBPM"                       # the solver (grid, profiles, decomposition helpers)
    meta: dict = field(default_factory=dict)

    @property
    def gain_dB(self) -> float:
        """End-to-end signal gain 10 log10(P(L)/P(0)) [dB]. It is the DIFFRACTION-apportioned
        gain: no overlap integral was assumed anywhere to produce it."""
        return float(10.0 * np.log10(self.P_signal_W[-1] / self.P_signal_W[0]))

    def mode_powers_W(self, modes: Sequence[LPMode], orientations=None, field_2d=None):
        """Power [W] in each supplied LP mode -- |INT psi_m* E dA|^2 with psi_m normalized to unit
        power on the SAME grid. Defaults to the output field. This is the gain-per-mode read-out
        (compare against the launch decomposition) and the cross-solver bridge to the modal
        pictures of transverse.py / lma.py. The modes are only orthogonal to the extent the grid
        resolves them, so sum(mode_powers) <= P_signal with the deficit being radiation plus
        discretization -- both visible, neither hidden."""
        E = self.E_out if field_2d is None else field_2d
        return np.abs(self.bpm.mode_overlaps(E, modes, orientations)) ** 2


# ============================ the BPM ===================================================

class GainBPM:
    """Scalar paraxial split-step BPM for the signal field of a doped fiber, with saturable gain
    from the resolved local balance and a co-marched pump.

    Constructor mirrors steady_state.FiberAmplifier / transverse.ResolvedFiberAmplifier --
    (ion, fiber, pumps, signals, ase) -- so the same plan can be handed to all three and the
    answers compared (that comparison is validation gate F). Exactly ONE Signal (one scalar field)
    and forward pumps only; everything else is refused with its physics reason.

    Grid keywords:
      n_grid            square FFT grid size N (default 256).
      extent_m          full window width [m]. Default: covers the inner cladding when a cladding
                        pump is present, else 8x the fundamental's 1/e field radius w (and at
                        least 6a / 3b). The window must hold the mode AND the absorber; unlike
                        transverse.py's quadrature grid, truncating it does NOT renormalize the
                        cladding pump (that intensity is analytic here), so the only cost of a
                        small window is boundary absorption -- which is measured and reported.
      n_clad            cladding index (default 1.45); n_core = sqrt(n_clad^2 + NA^2).
      n_ref             paraxial carrier index. Default n_eff(LP01) -- dossier sec. 2.3 measured
                        it best or near-best, by a wide margin at high contrast (1-cm fidelity
                        0.693 for n_ref = n_clad vs 0.6929... n_eff at NA = 0.14; the two are much
                        closer at LMA contrast). With an explicit index_profile the default is
                        max(n) instead (the duct axis).
      index_taper_cells s of the (F2.5a) erf taper n = n_clad + dn 0.5[1 - erf((r-a)/(s dx))].
                        DEFAULT 0.0 = a HARD step. The taper is a NUMERICAL smoothing of a PHYSICAL
                        discontinuity, so it is opt-in and always reported in the result metadata.
                        Dossier sec. 2.4 measured why it exists: a hard step's transverse spectral
                        tail is propagated with the parabolic dispersion k - K^2/2k instead of
                        sqrt(k^2 - K^2) and re-aliased by the periodic grid every step, costing
                        phase fidelity that s = 2 buys back (0.9007 -> 0.9964 over 2 mm at
                        NA = 0.14; 0.9988 -> 0.9999 at NA = 0.054). It shifts n_eff far below the
                        discretization error but it is still a changed problem.
      index_profile     callable f(X, Y) -> n(x,y) that REPLACES the step-index build (GRIN ducts,
                        smooth test fixtures, custom profiles). The a/8 sampling refusal is then
                        skipped -- there is no core radius to sample.
      absorber          super-Gaussian boundary absorber (F2.6), on by default; the layer sits at
                        rho > absorber_inner_frac * (extent/2) with width absorber_width_frac *
                        (extent/2) and order absorber_order (0.85 / 0.10 / 4 per the dossier).
      gain_model        callable f(I) -> g [1/m] that REPLACES the rate-equation medium gain
                        (the soa/transverse_bpm.qd_gain_table precedent). With it, `ion` may be
                        None and there must be no pumps; the fiber's background loss still applies.
                        I arrives as a FLAT array of intensities over the gain support (which is
                        the whole grid in this branch, C-ordered) and the return must broadcast to
                        it -- a scalar or a same-shape array. It is a function of intensity only:
                        an x,y-dependent model belongs in index_profile or in a subclass.
      thermal           ThermalLoop for the opt-in quasi-static thermal lens (default None = off).

    All physics-side quantities are SI; the field convention is exp(-i omega t) with the carrier
    exp(+i k0 n_ref z) factored out, matching soa/transverse_bpm.py."""

    def __init__(self, ion: Optional[RareEarthIon], fiber: FiberSpec, pumps: List[Pump],
                 signals: List[Signal], ase: Optional[AseBand] = None, *,
                 n_grid: int = 256, extent_m: Optional[float] = None, n_clad: float = 1.45,
                 n_ref: Optional[float] = None, index_taper_cells: float = 0.0,
                 index_profile: Optional[Callable] = None, absorber: bool = True,
                 absorber_inner_frac: float = 0.85, absorber_width_frac: float = 0.10,
                 absorber_order: float = 4.0, gain_model: Optional[Callable] = None,
                 thermal: Optional[ThermalLoop] = None,
                 upconversion_C_up: float = 0.0, concentration=None, raman=None):
        self.ion, self.fiber = ion, fiber
        self.pumps, self.signals = list(pumps), list(signals)
        self.ase = ase
        self.gain_model = gain_model
        self.thermal = thermal
        self.n_clad = float(n_clad)
        self._refuse(upconversion_C_up, concentration, raman)

        self.signal = self.signals[0]
        self.lambda_m = float(self.signal.lambda_m)
        self.k0 = 2.0 * np.pi / self.lambda_m
        self.n_core = float(np.sqrt(self.n_clad ** 2 + float(fiber.na) ** 2))

        # ---- grid ------------------------------------------------------------------------
        self.n_grid = int(n_grid)
        if self.n_grid < 8:
            raise ValueError("GainBPM: n_grid must be >= 8")
        self.extent_m = float(extent_m) if extent_m is not None else self._default_extent()
        self.dx = self.extent_m / self.n_grid
        self.x_m = (np.arange(self.n_grid) - self.n_grid // 2) * self.dx      # x = 0 is a NODE
        self.i_axis = self.n_grid // 2
        X, Y = np.meshgrid(self.x_m, self.x_m, indexing="ij")
        self.X, self.Y = X, Y
        self.R = np.sqrt(X * X + Y * Y)
        self.dA = self.dx * self.dx
        kx = 2.0 * np.pi * np.fft.fftfreq(self.n_grid, d=self.dx)
        self._K2 = kx[:, None] ** 2 + kx[None, :] ** 2
        if index_profile is None and self.dx > 0.125 * fiber.core_radius_m:
            # dossier sec. 2.4: dx <= a/16 is the practical rule and the index STEP, not the mode,
            # is what drives it. Half of that is the refusal bar -- a window auto-sized to a wide
            # pump cladding around a small core silently lands here, which is precisely the
            # accident worth refusing rather than answering.
            raise ValueError(
                "GainBPM: dx = {:.4g} m is coarser than a/8 = {:.4g} m -- the core index step "
                "would be represented by fewer than 8 cells per core RADIUS (the dossier's "
                "practical rule is dx <= a/16). This happens by default when the window is sized "
                "to a wide pump cladding around a small core: raise n_grid, or pass a smaller "
                "extent_m (the cladding pump intensity is analytic here, so a window inside the "
                "inner cladding does NOT inflate it), or pass index_profile for a core-free "
                "fixture.".format(self.dx, 0.125 * fiber.core_radius_m))

        # ---- index profile ----------------------------------------------------------------
        self.index_taper_cells = float(index_taper_cells)
        if index_profile is not None:
            self.n_xy = np.asarray(index_profile(X, Y), dtype=np.float64)
            if self.n_xy.shape != (self.n_grid, self.n_grid):
                raise ValueError("GainBPM: index_profile(X, Y) must return an (n_grid, n_grid) "
                                 "array (got {})".format(self.n_xy.shape))
            self._n_ref_default = float(np.max(self.n_xy))
        else:
            self.n_xy = self._step_index()
            self._n_ref_default = self._n_eff_lp01()
        self.n_ref = float(n_ref) if n_ref is not None else self._n_ref_default
        self.k = self.k0 * self.n_ref
        # (F2.5) index leg, as a PHASE RATE [rad/m]: k0^2 (n^2 - n_ref^2)/(2 k0 n_ref).
        self._phase_rate = self.k0 * (self.n_xy ** 2 - self.n_ref ** 2) / (2.0 * self.n_ref)

        # ---- absorber (F2.6) ---------------------------------------------------------------
        self.absorber = bool(absorber)
        half = 0.5 * self.extent_m
        self.absorber_inner_m = float(absorber_inner_frac) * half
        self.absorber_width_m = float(absorber_width_frac) * half
        self.absorber_order = float(absorber_order)
        if self.absorber and not (self.absorber_width_m > 0.0 and self.absorber_inner_m > 0.0):
            raise ValueError("GainBPM: absorber_inner_frac and absorber_width_frac must be > 0")
        self._mask = self._absorber_mask()
        self._one_minus_m2 = 1.0 - self._mask ** 2

        # ---- medium ------------------------------------------------------------------------
        self._build_medium()
        self._build_thermal()

    # ---- scope refusals --------------------------------------------------------------------
    def _refuse(self, upconversion_C_up, concentration, raman):
        """Every v1 scope boundary, refused at CONSTRUCTION with its physics reason (the
        transverse.py convention). Silently ignoring any of them would report a number this model
        did not compute."""
        if len(self.signals) != 1:
            raise ValueError(
                "GainBPM: exactly ONE Signal is supported (got {}). This is a SCALAR single-field "
                "BPM: a second signal at a different wavelength is a second field with its own "
                "diffraction operator and its own n_ref, sharing the inversion -- a genuinely "
                "different solver. Use transverse.ResolvedFiberAmplifier for a multi-channel "
                "plan, or run one BPM per wavelength if they do not saturate each "
                "other.".format(len(self.signals)))
        for p in self.pumps:
            if p.direction != "fwd":
                raise ValueError(
                    "GainBPM: backward (counter-propagating) pumps are not supported -- this is a "
                    "one-directional marcher, and a backward pump makes the problem a two-point "
                    "BVP that needs the relaxation sweeps of steady_state/transverse. Pump "
                    "direction was {!r}.".format(p.direction))
            if p.cladding and self.fiber.clad_radius_m is None:
                raise ValueError("GainBPM: Pump(cladding=True) needs FiberSpec.clad_radius_m")
        if self.ase is not None and self.ase.n_bins > 0:
            raise ValueError(
                "GainBPM: ASE is not supported. Spontaneous emission seeds every guided AND "
                "radiation mode in both directions with a random phase -- it is not a field this "
                "coherent one-way marcher can carry, and faking it as extra power on the signal "
                "field would put spontaneous photons into a single coherent mode. ASE stays in "
                "the channel models (steady_state.FiberAmplifier / "
                "transverse.ResolvedFiberAmplifier, which resolve it per bin and per direction); "
                "run one of those for noise figures and use this solver for the field.")
        if self.gain_model is not None:
            if self.pumps:
                raise ValueError(
                    "GainBPM: gain_model REPLACES the rate-equation medium entirely -- do not "
                    "also pass pumps (they would be marched but could not affect the gain). Pass "
                    "one gain model or the other.")
            return                       # a supplied gain model needs no spectroscopy at all
        if not isinstance(self.ion, RareEarthIon):
            raise ValueError(
                "GainBPM: ion must be a RareEarthIon (or pass gain_model for a non-rate-equation "
                "medium). Er:Yb co-doping (ErYbAmplifier) has no BPM form here: it needs TWO "
                "coupled local balances with a Yb->Er transfer term, and the transfer is a "
                "density-density coupling that the field solver would have to carry per pixel.")
        if self.fiber.overlap_override is not None:
            raise ValueError(
                "GainBPM: FiberSpec.overlap_override is not supported. A BPM PRODUCES the "
                "confinement factor by diffraction; an override would replace the very quantity "
                "the solver exists to compute (it is the right tool for the mean-field "
                "FiberAmplifier, e.g. Giles-calibrated fibers).")
        if raman is not None:
            raise ValueError(
                "GainBPM: RamanStokes is not supported. The SRS exchange is a coupling between "
                "TWO fields at different wavelengths (g_R/A_eff with a Manley-Rowe photon "
                "balance); with the transverse structure resolved, A_eff would have to become an "
                "overlap of both fields with the resolved gain, and the Stokes field needs its "
                "own marcher.")
        if concentration is not None or float(upconversion_C_up) != 0.0:
            raise ValueError(
                "GainBPM: concentration effects (ConcentrationModel, upconversion_C_up != 0) are "
                "not supported. Cooperative upconversion adds -C_up N2^2 to the balance, so "
                "(F1.1) becomes QUADRATIC in nbar2 at every pixel and the linear-fractional form "
                "this module shares with transverse.py is gone. Photodarkening additionally "
                "dissipates ~100% of what it absorbs, so with it on the heat source (F2.9) would "
                "be a serious under-estimate (Jauregui et al. 2020 sec. 4).")
        lam = np.array([p.lambda_m for p in self.pumps] + [self.signals[0].lambda_m], float)
        if self.ion.sigma_esa is not None:
            esa = np.asarray(self.ion.sigma_esa_of(lam), float)
            if float(np.max(esa)) > 0.0:
                raise ValueError(
                    "GainBPM: an ion with NONZERO excited-state absorption (max sigma_esa = "
                    "{:.3e} m^2 over this plan) is not supported. ESA enters (F2.7) as an extra "
                    "-nt sigma_esa nbar2 term; it is transcribed in the docs and deliberately not "
                    "wired, exactly as in transverse.py.".format(float(np.max(esa))))

    def set_temperature_profile(self, *args, **kwargs):
        """REFUSED. steady_state's axial T(z) McCumber scaling has no meaning here: the field
        solver's own thermal object is a T(r, z) (ThermalLoop), and applying a z-only rescaling
        uniformly across a transverse profile whose whole point is that it is not uniform would
        misreport what was solved."""
        raise ValueError(
            "GainBPM: set_temperature_profile is not supported -- an axial T(z) McCumber scaling "
            "is not meaningful once the field is resolved across the mode. Pass thermal=ThermalLoop"
            "(...) for the quasi-static T(r, z) lens, or use steady_state.FiberAmplifier for the "
            "mean-field thermal-feedback loop.")

    # ---- geometry / index ------------------------------------------------------------------
    def _mode_radius_m(self) -> float:
        """1/e FIELD radius of the fundamental at the signal wavelength: the exact LP01 radius
        above the LP11 cutoff, the Marcuse Gaussian below it -- the same branch transverse.py
        takes, so the two modules size themselves off the same mode."""
        a, na = self.fiber.core_radius_m, self.fiber.na
        V = 2.0 * np.pi * a * na / self.lambda_m
        if V > V_LP11_CUTOFF:
            return float(one_over_e_radius_m(solve_lp_modes(a, na, self.lambda_m)[0]))
        return float(mode_field_radius_m(a, na, self.lambda_m))

    def _default_extent(self) -> float:
        """Full window width. Covers the inner cladding when a cladding pump is present (so the
        pump-illuminated cross-section is actually represented), else 8 w / 6 a / 3 b -- at 4 w
        from the axis a Gaussian is down to exp(-32) = 1e-14 in intensity, which leaves the
        absorber nothing to eat."""
        a, b = self.fiber.core_radius_m, self.fiber.b_dope_m
        half = max(4.0 * self._mode_radius_m(), 3.0 * a, 1.5 * b)
        if any(p.cladding for p in self.pumps) and self.fiber.clad_radius_m is not None:
            half = max(half, float(self.fiber.clad_radius_m))
        return 2.0 * half

    def _step_index(self) -> np.ndarray:
        """Step-index n(x,y) from the FiberSpec, optionally with the (F2.5a) erf taper."""
        a = float(self.fiber.core_radius_m)
        dn = self.n_core - self.n_clad
        if self.index_taper_cells > 0.0:
            s = self.index_taper_cells * self.dx
            return self.n_clad + dn * 0.5 * (1.0 - erf((self.R - a) / s))
        return np.where(self.R <= a, self.n_core, self.n_clad)

    def _n_eff_lp01(self) -> float:
        """n_eff of LP01 at the signal wavelength (dossier sec. 2.3's recommended n_ref). Falls
        back to the core index if the scalar solver finds nothing guided, which cannot happen for
        a real fiber (LP01 exists for any V > 0) but keeps the constructor total."""
        try:
            modes = solve_lp_modes(self.fiber.core_radius_m, self.fiber.na, self.lambda_m,
                                   n_clad=self.n_clad)
        except ValueError:                                  # pragma: no cover - guarded upstream
            return self.n_core
        return float(modes[0].n_eff()) if modes else self.n_core

    def _absorber_mask(self) -> np.ndarray:
        """(F2.6) super-Gaussian amplitude mask, circular rho = sqrt(x^2+y^2) (the fiber's own
        symmetry). Identically 1 inside rho_abs, so a run whose field never reaches the layer is
        bit-identical to an absorber-free one."""
        if not self.absorber:
            return np.ones_like(self.R)
        out = np.ones_like(self.R)
        far = self.R > self.absorber_inner_m
        arg = ((self.R[far] - self.absorber_inner_m) / self.absorber_width_m) ** self.absorber_order
        out[far] = np.exp(-np.minimum(arg, 700.0))          # 700: exp underflow floor, no warning
        return out

    # ---- medium bookkeeping -----------------------------------------------------------------
    def _build_medium(self):
        """Hoist everything the per-step gain needs: cross-sections (via ChannelSet, so the
        spectroscopy path is byte-identical to steady_state/transverse), the gain SUPPORT mask,
        the normalized pump profiles, and the coefficient bundle handed to the shared local
        balance.

        The support is the DOPED disc for the rate-equation medium: outside it nt = 0, so the gain
        there is exactly the (scalar) background loss and the pump integrand vanishes -- evaluating
        the balance on ~9% of the pixels instead of all of them is the difference between the
        marcher being FFT-bound (which it should be) and balance-bound. A supplied gain_model is
        a function of intensity with no dopant geometry attached, so its support is the whole
        grid."""
        self.loss_s = 0.0
        self._dope = self.R <= float(self.fiber.b_dope_m)
        self.n_pumps = len(self.pumps)
        self._P_p0 = np.array([p.power_W for p in self.pumps], float)
        if self.gain_model is not None:
            self.loss_s = float(self.fiber.loss_per_m(np.array([self.lambda_m]))[0])
            self._support = np.ones_like(self._dope)
            self._ip_dope = np.zeros((0, self.n_grid * self.n_grid))
            self._qd = np.zeros(0)
            self._bundle = None
            return
        self._support = self._dope
        lam = np.array([p.lambda_m for p in self.pumps] + [self.lambda_m], float)
        u = np.ones_like(lam)
        ch = ChannelSet.build(self.ion, self.fiber, lam, u)
        # ch.gamma is deliberately UNUSED: in a BPM the confinement factor is an output.
        self.sigma_a, self.sigma_e = ch.sigma_a, ch.sigma_e
        self.loss_per_m = ch.loss_per_m
        self.loss_s = float(ch.loss_per_m[-1])
        self.nt = float(self.fiber.n_t_m3)
        self._qd = np.array([quantum_defect_fraction(p.lambda_m, self.lambda_m)
                             for p in self.pumps], float)
        self._ip_dope = np.array([self._pump_profile(p)[self._dope] for p in self.pumps]) \
            if self.pumps else np.zeros((0, int(np.count_nonzero(self._dope))))
        n_dope = int(np.count_nonzero(self._dope))
        self._I_buf = np.empty((lam.size, n_dope))
        self._ones = np.ones(lam.size)
        # The bundle keys are transverse._nbar2_nodes's contract: fa/fe are sigma/(h nu) so that
        # (fa * P) @ idope is the per-node absorption rate [1/s]; with idope carrying INTENSITIES
        # and P = 1 the same expression is SUM_k sigma_a_k I_k/(h nu_k), which is (F1.1) exactly.
        h_nu = H_PLANCK * np.asarray(ch.nu_hz, float)
        self._bundle = {"fa": ch.sigma_a / h_nu, "fe": ch.sigma_e / h_nu,
                        "tau": float(ch.tau_s), "idope": self._I_buf}

    def _pump_profile(self, pump: Pump) -> np.ndarray:
        """Normalized transverse profile i_p(x,y) [m^-2], INT i_p dA = 1.

        A CLADDING pump is flat over the inner cladding, i_p = 1/(pi R_clad^2), taken
        ANALYTICALLY -- not renormalized on the grid. That is the deliberate difference from
        transverse.py: there the flat profile is normalized by its own quadrature integral, so a
        grid inside R_clad inflates the pump overlap by (R_clad/r_max)^2 and the solver refuses
        such a grid; here the intensity is a closed form, so a small window is merely a small
        window. A CORE-guided pump uses the exact LP01 above the LP11 cutoff and the Marcuse
        Gaussian below it (transverse.py's branch), normalized ON the grid because its tail is
        what the normalization is about."""
        if pump.cladding:
            rc = float(self.fiber.clad_radius_m)
            return np.where(self.R <= rc, 1.0 / (np.pi * rc * rc), 0.0)
        a, na = self.fiber.core_radius_m, self.fiber.na
        lam = float(pump.lambda_m)
        V = 2.0 * np.pi * a * na / lam
        if V > V_LP11_CUTOFF:
            psi = mode_field(solve_lp_modes(a, na, lam, n_clad=self.n_clad)[0], self.R)
            raw = psi * psi
        else:
            w = float(mode_field_radius_m(a, na, lam))
            raw = np.exp(-2.0 * self.R * self.R / (w * w))
        norm = float(np.sum(raw)) * self.dA
        if not (norm > 0.0):
            raise ValueError("GainBPM: the core pump profile integrates to zero on this grid")
        return raw / norm

    def _build_thermal(self):
        """Pre-compute the radial temperature SHAPE [K per (W/m)]. thermal.radial_temperature_rise
        is linear in the heat load, so one shape array plus the per-step Q(z) is the whole
        z-dependent thermal lens. Radii beyond b_outer are clamped to b_outer (the coolant
        boundary), which only affects grid corners outside the conduction stack."""
        self._dn_shape = None
        if self.thermal is None:
            return
        b_out = float(self.thermal.b_outer_m)
        _, dT = radial_temperature_rise(1.0, self.fiber.core_radius_m, b_out,
                                        self.thermal.model, r_m=np.minimum(self.R, b_out))
        self._dn_shape = float(self.thermal.dndt_per_K) * np.asarray(dT, float)

    # ---- field helpers -----------------------------------------------------------------------
    def power_W(self, E) -> float:
        """INT |E|^2 dA [W] -- the field normalization contract."""
        return float(np.sum(np.abs(np.asarray(E)) ** 2)) * self.dA

    def spot_radius_m(self, I) -> float:
        """sqrt(2 <r^2>) of an intensity profile [m]. For a Gaussian exp(-2 r^2/w^2) this returns
        exactly the 1/e FIELD radius w, matching lma.second_moment_radius_m's convention (and the
        dossier's w_m), so the duct oracles are read in the same units they are stated in."""
        I = np.asarray(I, float)
        tot = float(np.sum(I))
        return float(np.sqrt(2.0 * float(np.sum(I * self.R ** 2)) / tot))

    def mode_field_2d(self, mode: LPMode, orientation: str = "cos",
                      power_W: float = 1.0) -> np.ndarray:
        """The LP_lm field on this grid, normalized to `power_W` (default unit power, i.e.
        INT psi^2 dA = 1). l >= 1 modes carry the azimuthal factor cos(l phi) or sin(l phi):
        unlike transverse.py, which only ever needs the INTENSITY cos^2(l phi) and therefore
        cannot spell the sin partner, a field solver carries the orientation explicitly, so the
        degenerate pair is available here."""
        if orientation not in ("cos", "sin"):
            raise ValueError("GainBPM.mode_field_2d: orientation must be 'cos' or 'sin'")
        psi = np.asarray(mode_field(mode, self.R), float)
        if mode.l >= 1:
            phi = np.arctan2(self.Y, self.X)
            psi = psi * (np.cos(mode.l * phi) if orientation == "cos" else np.sin(mode.l * phi))
        norm = float(np.sum(psi * psi)) * self.dA
        if not (norm > 0.0):
            raise ValueError("GainBPM.mode_field_2d: the mode integrates to zero on this grid")
        return psi * np.sqrt(float(power_W) / norm)

    def mode_overlaps(self, E, modes: Sequence[LPMode], orientations=None) -> np.ndarray:
        """(n_modes,) complex modal amplitudes c_m = INT psi_m* E dA with psi_m at unit power, so
        |c_m|^2 is the power [W] in that mode. The LP fields are real, so no conjugation of psi is
        actually needed; the form is written as the projection it is."""
        E = np.asarray(E)
        orients = ["cos"] * len(modes) if orientations is None else list(orientations)
        if len(orients) != len(modes):
            raise ValueError("GainBPM.mode_overlaps: orientations must be one per mode")
        out = np.empty(len(modes), dtype=np.complex128)
        for i, (md, orient) in enumerate(zip(modes, orients)):
            psi = self.mode_field_2d(md, orient)
            out[i] = np.sum(psi * E) * self.dA
        return out

    def launch_field(self, power_W: Optional[float] = None, mode: Optional[LPMode] = None,
                     orientation: str = "cos", w_m: Optional[float] = None) -> np.ndarray:
        """The default launch: the fundamental at the Signal's power, normalized to
        INT |E|^2 dA = P. `mode` launches a specific LPMode instead; `w_m` launches a Gaussian of
        1/e field radius w (the duct fixtures). Superpositions are built by adding launch_field
        calls -- for orthogonal modes the powers then add, which mode_powers_W verifies rather
        than assumes."""
        P = float(self.signal.power_W if power_W is None else power_W)
        if mode is not None and w_m is not None:
            raise ValueError("GainBPM.launch_field: pass a mode OR a Gaussian w_m, not both")
        if mode is not None:
            return self.mode_field_2d(mode, orientation, P).astype(np.complex128)
        if w_m is not None:
            raw = np.exp(-(self.R / float(w_m)) ** 2)
        else:
            a, na = self.fiber.core_radius_m, self.fiber.na
            V = 2.0 * np.pi * a * na / self.lambda_m
            if V > V_LP11_CUTOFF:
                raw = np.asarray(mode_field(solve_lp_modes(a, na, self.lambda_m,
                                                           n_clad=self.n_clad)[0], self.R), float)
            else:
                w = float(mode_field_radius_m(a, na, self.lambda_m))
                raw = np.exp(-(self.R / w) ** 2)
        norm = float(np.sum(raw * raw)) * self.dA
        return (raw * np.sqrt(P / norm)).astype(np.complex128)

    # ---- operators ----------------------------------------------------------------------------
    def _diffract(self, E, kernel):
        """(F2.4) EXACT unitary paraxial diffraction: E_hat *= exp(-i (kx^2+ky^2) dz/(2k)),
        k = k0 n_ref, with the kernel precomputed once per march (dz is fixed there). Byte-for-byte
        the 2-D form of soa/transverse_bpm._diffract, which is the repo's convention anchor
        (exp(-i omega t): the minus sign in the exponent is NOT the Feit-Fleck engineering-
        convention one). Being a pure phase in Fourier space it is exactly unitary, which is what
        the energy ledger leans on."""
        return _sfft.ifft2(_sfft.fft2(E) * kernel, overwrite_x=True)

    def diffraction_kernel(self, dz) -> np.ndarray:
        """The (F2.4) propagator phase for a step dz -- exposed so a caller can inspect the exact
        operator the march applies."""
        return np.exp(-1j * self._K2 * float(dz) / (2.0 * self.k))

    def _medium(self, I_s, P_p):
        """Local INTENSITY gain g [1/m] (F2.7) ON THE SUPPORT, the pump log-derivative rates [1/m]
        (F2.8), and the heat load [W/m] (F2.9), for the support-restricted field intensity I_s and
        the pump vector P_p. g is NET of the background loss.

        The balance (F1.1) itself is transverse._nbar2_nodes -- see the import comment; the bundle
        handed to it carries INTENSITIES as its profile rows against a unit power vector, which is
        exact because I_k = P_k i_k."""
        if self.gain_model is not None:
            g = np.asarray(self.gain_model(I_s), float) - self.loss_s
            return (g if g.shape == I_s.shape else np.broadcast_to(g, I_s.shape),
                    np.zeros(0), 0.0)
        self._I_buf[:self.n_pumps] = P_p[:, None] * self._ip_dope
        self._I_buf[self.n_pumps] = I_s
        nb = _LOCAL_BALANCE(None, self._bundle, self._ones)
        g = (self.nt * ((self.sigma_a[-1] + self.sigma_e[-1]) * nb - self.sigma_a[-1])
             - self.loss_s)
        if self.n_pumps == 0:
            return g, np.zeros(0), 0.0
        n2 = self.nt * nb
        # net absorbed pump power density per unit pump intensity [1/m], per pump channel
        absorb = (self.sigma_a[:self.n_pumps, None] * (self.nt - n2)[None, :]
                  - self.sigma_e[:self.n_pumps, None] * n2[None, :])
        rate = -(np.sum(absorb * self._ip_dope, axis=1) * self.dA
                 + self.loss_per_m[:self.n_pumps])
        heat = float(np.sum(self._qd[:, None] * absorb * self._I_buf[:self.n_pumps])) * self.dA
        return g, rate, heat

    def local_gain_per_m(self, E, pump_powers_W=None) -> np.ndarray:
        """(N, N) local intensity gain coefficient g(x,y) [1/m] for a field E and pump powers --
        the public diagnostic behind (F2.7). Its overlap with |E|^2/P is the confinement factor
        this solver never needs as an input. Off the gain support g is the background loss."""
        P_p = (self._P_p0 if pump_powers_W is None
               else np.asarray(pump_powers_W, float).reshape(self.n_pumps))
        E = np.asarray(E)
        out = np.full((self.n_grid, self.n_grid), -self.loss_s)
        out[self._support] = self._medium((E.real ** 2 + E.imag ** 2)[self._support], P_p)[0]
        return out

    def _nonlinear(self, E, I, P_p, dz, n_phase):
        """The N leg over a full dz: the index phase (pure real, so energy-neutral) and the
        SATURABLE gain, MIDPOINT-corrected exactly as soa/transverse_bpm._gain does it -- predictor
        half-step at the entry gain, full step at the gain of the half-advanced field. Freezing g
        at the step entry would drop the whole Strang split to O(dz), which gate C catches. The
        pump is advanced on the same predictor/corrector, so field and pump stay a consistent pair
        at the midpoint. The predictor advances the INTENSITY only: the index phase does not change
        |E|^2 and the gain depends on nothing else, so the complex field is touched exactly once.

        `n_phase` is exp(i k0 (n^2 - n_ref^2) dz/(2 n_ref)) times the scalar off-support amplitude
        factor exp(-l_s dz/2), precomputed for the step."""
        I_s = I[self._support]
        g_a, rate_a, _ = self._medium(I_s, P_p)
        I_half = I_s * np.exp(np.minimum(g_a * (0.5 * dz), _EXP_CEILING))
        P_half = P_p * np.exp(rate_a * (0.5 * dz)) if self.n_pumps else P_p
        g_m, rate_m, heat = self._medium(I_half, P_half)
        amp = np.exp(np.minimum(0.5 * g_m * dz, _EXP_CEILING))
        # (dossier sec. 2.5) the medium's power increment, from g and the ENTERING field -- a
        # closed form, not a power difference, so the ledger it feeds is a real test of the other
        # operators rather than a restatement of them. Off the support the only medium term is the
        # background loss, hence the two-piece sum.
        amp_off2 = float(np.exp(-self.loss_s * dz))
        d_exchange = ((amp_off2 - 1.0) * float(np.sum(I)) * self.dA
                      + float(np.sum(I_s * (amp * amp - amp_off2))) * self.dA)
        E = E * n_phase
        # `n_phase` already carried exp(-l_s dz/2) across the WHOLE grid, and `g_m` (F2.7) is
        # already NET of l_s -- so multiplying the support by `amp` alone would apply the
        # background loss TWICE there (on-support intensity factor exp(g_med dz - 2 l_s dz)).
        # The exp(+l_s dz/2) undoes the whole-grid factor exactly where the medium's own gain
        # already accounts for it; off the support n_phase's loss stands alone. Gate I pins both
        # halves (pure loss and gain+loss) against exp(-l_s L) at 1e-12.
        E[self._support] *= amp * np.exp(0.5 * self.loss_s * dz)
        P_p = P_p * np.exp(rate_m * dz) if self.n_pumps else P_p
        return E, P_p, d_exchange, heat

    # ---- step / grid guidance -------------------------------------------------------------------
    def transverse_nyquist_step_m(self) -> float:
        """The TRANSVERSE-Nyquist longitudinal step: dz <= 2 k dx^2 / pi.

        The split's two legs are diagonal in different bases, so the commutator error is carried by
        the highest transverse frequency the grid holds, kx_max = pi/dx. Its diffraction phase per
        step is kx_max^2 dz/(2k) = pi^2 dz/(2 k dx^2), and once that exceeds ~pi the corner of the
        k-grid is advanced by an aliased phase -- the index/gain leg then re-scatters into it every
        step and the error stops behaving like a discretization error at all.

        The consequence is counter-intuitive: halving dx QUARTERS this bound, so transverse
        refinement TIGHTENS the longitudinal step and the two are not independent knobs. It is a
        conservative guard, not a fit to an observed failure -- module docstring, scope limitation
        2, carries the re-measurement that shows the shipped step-index fixture does not populate
        the corner of the k-grid and so does not feel it."""
        return float(2.0 * self.k * self.dx * self.dx / np.pi)

    def suggested_step_m(self) -> float:
        """Longitudinal step from the dossier's independent criteria (sec. 2.4), whichever is
        smallest:
          1. the N operator must not alias -- keep the index phase per step below 0.3 rad;
          2. >= 20 steps per shortest guided-mode BEAT length L_beat = lambda/(n_eff1 - n_eff2),
             the scale on which the split's commutator error accumulates;
          3. the TRANSVERSE-Nyquist bound dz <= 2 k dx^2/pi (transverse_nyquist_step_m) -- so
             refining the grid in x,y TIGHTENS this step rather than leaving it alone.
        Single-mode fibers (and index_profile fixtures) only see criteria 1 and 3."""
        pmax = float(np.max(np.abs(self._phase_rate)))
        dz = 0.3 / pmax if pmax > 0.0 else float(self.fiber.length_m)
        dz = min(dz, self.transverse_nyquist_step_m())
        try:
            modes = solve_lp_modes(self.fiber.core_radius_m, self.fiber.na, self.lambda_m,
                                   n_clad=self.n_clad)
        except ValueError:                                  # pragma: no cover - guarded upstream
            modes = []
        if len(modes) >= 2:
            dbeta = abs(modes[0].beta - modes[1].beta)
            if dbeta > 0.0:
                dz = min(dz, (2.0 * np.pi / dbeta) / 20.0)
        return float(dz)

    def sampling_report(self) -> dict:
        """Grid / sampling diagnostics: dx in units of the core radius and the mode radius, the
        Nyquist propagation angle theta_max = lambda/(2 n_ref dx), the window in mode radii, the
        transverse-Nyquist step bound, and the suggested step. Everything the dossier's sec. 2.4
        rules are stated in, plus the dx-dz coupling of transverse_nyquist_step_m."""
        w = self._mode_radius_m()
        return {"dx_m": self.dx, "dx_over_core_radius": self.dx / self.fiber.core_radius_m,
                "dx_over_mode_radius": self.dx / w,
                "half_window_over_mode_radius": 0.5 * self.extent_m / w,
                "theta_max_rad": self.lambda_m / (2.0 * self.n_ref * self.dx),
                "transverse_nyquist_dz_m": self.transverse_nyquist_step_m(),
                "suggested_dz_m": self.suggested_step_m()}

    # ---- the march ------------------------------------------------------------------------------
    def solve(self, *, n_steps: Optional[int] = None, dz_m: Optional[float] = None,
              E_in=None, n_samples: int = 101, track_modes: Optional[Sequence[LPMode]] = None,
              track_orientations=None, store_fields: bool = False,
              max_auto_steps: int = 20000) -> BPMResult:
        """March the field over the fiber length and return a BPMResult.

        n_steps / dz_m: pass one (dz_m wins if both are given). With neither, the step comes from
        suggested_step_m(); if that would need more than max_auto_steps steps the call is REFUSED
        rather than silently running for minutes -- pass n_steps explicitly (and then the split
        error is yours to check, which is what the dz-refinement inside every gate is for).

        E_in defaults to launch_field() (the fundamental at the Signal power). track_modes records
        the complex modal amplitude of each supplied LPMode at every sample, which is the
        gain-per-mode read-out; store_fields keeps the full field at every sample (M N^2 complex,
        so opt-in).

        With thermal=ThermalLoop(...) the march is repeated to self-consistency: heat -> dT(r) ->
        dn -> field -> heat, under-relaxed, until the peak temperature change between iterates
        falls below tol_K or max_iter is reached. meta carries iterations / converged / max_dT_K."""
        L = float(self.fiber.length_m)
        if dz_m is not None:
            # dz_m <= 0 used to be answered rather than refused: a negative step fell through
            # max(ceil(L/dz), 1) = 1 and ran a single L-long step (a plausible-looking number from
            # a typo), and dz_m = 0.0 raised a bare ZeroDivisionError from inside the marcher.
            if not (float(dz_m) > 0.0):
                raise ValueError(
                    "GainBPM.solve: dz_m must be > 0 (got {!r}). This marcher only runs FORWARD, "
                    "and a zero or negative step has no meaning here -- pass a positive dz_m, or "
                    "n_steps, or neither (the step then comes from "
                    "suggested_step_m()).".format(dz_m))
            n = max(int(np.ceil(L / float(dz_m))), 1)
        elif n_steps is not None:
            n = int(n_steps)
        else:
            n = max(int(np.ceil(L / self.suggested_step_m())), 1)
            if n > int(max_auto_steps):
                raise ValueError(
                    "GainBPM.solve: the automatic step (dz = {:.4g} m, from the 0.3-rad index "
                    "phase and L_beat/20 criteria) needs {} steps for L = {:.4g} m, above "
                    "max_auto_steps "
                    "= {}. Pass n_steps or dz_m explicitly (and check the answer against a halved "
                    "step), or raise max_auto_steps if you really want the long "
                    "run.".format(L / n, n, L, int(max_auto_steps)))
        if n < 1:
            raise ValueError("GainBPM.solve: n_steps must be >= 1")
        dz = L / n
        E0 = self.launch_field() if E_in is None else np.asarray(E_in, dtype=np.complex128).copy()
        if E0.shape != (self.n_grid, self.n_grid):
            raise ValueError("GainBPM.solve: E_in must have shape (n_grid, n_grid)")
        idx = np.unique(np.round(np.linspace(0, n, max(int(n_samples), 2))).astype(int))

        psi_track = None
        if track_modes:
            orients = (["cos"] * len(track_modes) if track_orientations is None
                       else list(track_orientations))
            psi_track = [self.mode_field_2d(md, o) for md, o in zip(track_modes, orients)]

        Q_prev = np.zeros(n)
        info = {"thermal_iterations": 0, "thermal_converged": self.thermal is None,
                "thermal_max_dT_K": 0.0}
        rec = None
        n_iter = 1 if self.thermal is None else int(self.thermal.max_iter)
        for it in range(n_iter):
            rec = self._march(E0, n, dz, idx, psi_track, store_fields, Q_prev)
            if self.thermal is None:
                break
            Q_new = rec["Q_step"]
            peak = float(np.max(self._dn_shape)) / float(self.thermal.dndt_per_K)
            dT = float(np.max(np.abs(Q_new - Q_prev))) * peak
            Q_prev = Q_prev + self.thermal.relax * (Q_new - Q_prev)
            info.update(thermal_iterations=it + 1, thermal_max_dT_K=dT)
            if dT < float(self.thermal.tol_K):
                info["thermal_converged"] = True
                break

        z = idx * dz
        P_sig = rec["P"]
        residual = (P_sig[0] + rec["exchange"] - rec["absorbed"]) - P_sig
        meta = {"n_steps": n, "dz_m": dz, "n_grid": self.n_grid, "extent_m": self.extent_m,
                "dx_m": self.dx, "n_ref": self.n_ref, "index_taper_cells": self.index_taper_cells,
                "absorber": self.absorber, "absorber_inner_m": self.absorber_inner_m,
                "absorber_width_m": self.absorber_width_m, "absorber_order": self.absorber_order,
                "gain_model": self.gain_model is not None,
                "absorbed_fraction": float(rec["absorbed"][-1] / P_sig[0]) if P_sig[0] > 0 else 0.0,
                "sampling": self.sampling_report() if self.gain_model is None else None}
        meta.update(info)
        if meta["absorbed_fraction"] > _ABSORBED_FRACTION_BAR:
            warnings.warn(
                "GainBPM: the boundary absorber removed {:.3e} of the launched power, above the "
                "converged-window bar {:.0e}. The super-Gaussian mask is applied once per STEP, so "
                "it is a dz-DEPENDENT operator with no dz -> 0 limit (module docstring, scope "
                "limitation 1): at this absorption level the answer moves when you refine dz, and "
                "the direction is not a convergence. Enlarge extent_m (with a cladding pump the "
                "pump intensity is analytic here, so a wider window costs only grid), move the "
                "absorber out (absorber_inner_frac), or -- on a step-index fiber, where most of "
                "what the boundary eats is radiation shed by the DISCRETIZED index step rather "
                "than the guided mode -- raise index_taper_cells.".format(
                    meta["absorbed_fraction"], _ABSORBED_FRACTION_BAR), UserWarning, stacklevel=2)
        return BPMResult(z, P_sig, rec["Pp"], rec["E"], self.x_m, rec["absorbed"],
                         rec["exchange"], residual, rec["w"], rec["Iax"], rec["Q"],
                         rec["amp"], rec["fields"], self, meta)

    def _march(self, E0, n, dz, idx, psi_track, store_fields, Q_step_in):
        """One pass of the split-step march. The step is D(dz/2) N(dz) D(dz/2) followed by the
        boundary mask -- the same ordering soa/transverse_bpm.py uses."""
        E = E0.copy()
        P_p = self._P_p0.copy()
        absorbed = 0.0
        exchange = 0.0
        m = idx.size
        out = {"P": np.empty(m), "Pp": np.empty((self.n_pumps, m)), "absorbed": np.empty(m),
               "exchange": np.empty(m), "w": np.empty(m), "Iax": np.empty(m), "Q": np.empty(m),
               "Q_step": np.empty(n),
               "amp": None if psi_track is None else np.empty((len(psi_track), m), complex),
               "fields": np.empty((m, self.n_grid, self.n_grid), complex) if store_fields else None}
        sample_of = {int(v): i for i, v in enumerate(idx)}
        thermal_on = self.thermal is not None

        def record(slot, heat):
            I = E.real ** 2 + E.imag ** 2
            out["P"][slot] = float(np.sum(I)) * self.dA
            if self.n_pumps:
                out["Pp"][:, slot] = P_p
            out["absorbed"][slot] = absorbed
            out["exchange"][slot] = exchange
            out["w"][slot] = self.spot_radius_m(I)
            out["Iax"][slot] = float(I[self.i_axis, self.i_axis])
            out["Q"][slot] = heat
            if psi_track is not None:
                for i, psi in enumerate(psi_track):
                    out["amp"][i, slot] = np.sum(psi * E) * self.dA
            if store_fields:
                out["fields"][slot] = E

        if 0 in sample_of:
            # the entry-plane heat is the medium evaluated at the LAUNCH state, not zero: recording
            # a 0 there would put a spurious step into Q(z) that any dQ/dz consumer would believe.
            record(sample_of[0], self._medium((E.real ** 2 + E.imag ** 2)[self._support], P_p)[2])
        # dz is fixed over a march, so both diagonal operators are precomputed ONCE: the
        # half-step diffraction kernel and (when there is no z-dependent thermal lens) the
        # combined index-phase / background-loss factor of the N leg. Rebuilding either per step
        # is two extra transcendental passes over the grid, which on the profiled box was ~40% of
        # the marcher.
        kernel = self.diffraction_kernel(0.5 * dz)
        amp_off = float(np.exp(-0.5 * self.loss_s * dz))
        n_phase = np.exp(1j * self._phase_rate * dz) * amp_off
        for j in range(n):
            if thermal_on:
                # written as the SAME expression as self._phase_rate so that a zero heat load is
                # bitwise identical to an athermal march (the first loop iterate is exactly that,
                # and a test pins it) -- reassociating the arithmetic here would silently make
                # "thermal loop, zero heat" a different number from "no thermal loop".
                n_tot = self.n_xy + Q_step_in[j] * self._dn_shape
                rate = self.k0 * (n_tot ** 2 - self.n_ref ** 2) / (2.0 * self.n_ref)
                n_phase = np.exp(1j * rate * dz) * amp_off
            E = self._diffract(E, kernel)
            I = E.real ** 2 + E.imag ** 2
            E, P_p, d_ex, heat = self._nonlinear(E, I, P_p, dz, n_phase)
            E = self._diffract(E, kernel)
            exchange += d_ex
            out["Q_step"][j] = heat
            if self.absorber:
                I = E.real ** 2 + E.imag ** 2
                absorbed += float(np.sum(I * self._one_minus_m2)) * self.dA
                E = E * self._mask
            slot = sample_of.get(j + 1)
            if slot is not None:
                record(slot, heat)
        out["E"] = E
        return out
