"""Radially-resolved inversion n2(r, phi, z) -- the PRE-REDUCTION Giles-Desurvire fiber-amplifier
model, i.e. transverse spatial hole burning (TSHB). `steady_state.py` solves the same physics in
its MEAN-FIELD reduction: every channel's intensity is replaced by its area average over the
dopant, `<I_k> = Gamma_k P_k / A_dope`, so the metastable fraction is ONE number per z. Here the
local balance is evaluated at every quadrature node of the fiber cross-section and the
propagation coefficients are radial integrals against each channel's own normalized modal
intensity. Feature 1 of docs/fiber_transverse_grounding_2026_07_29.md; spec section 12 of
docs/fiber_amp_model_spec.md.

GOVERNING EQUATIONS (dossier numbering; sources in the docstrings of each routine).

  (F1.1)  local two-level balance, PINNED verbatim from A. V. Smith and J. J. Smith, "Increased
          mode instability thresholds of fiber amplifiers by gain saturation", Opt. Express
          21(13):15168 (2013), Eq. (1), generalized to the repo's K-channel abstraction:

              nbar2(r,phi,z) = SUM_k sigma_a_k I_k / (h nu_k)
                             / [ 1/tau + SUM_k (sigma_a_k + sigma_e_k) I_k / (h nu_k) ]

          with I_k(r,phi,z) = P_k(z) i_k(r,phi) and n2 = nt(r) * nbar2. NOTHING transports
          excitation laterally (ion diffusion is nil in a glass host), so the balance is strictly
          pointwise -- unlike the carrier-diffusion-smoothed hole burning of soa/transverse_bpm.py
          there is NO L_diff analogue, which is what makes TSHB a sharp effect in fibers.

  (F1.0)  normalization of every transverse profile:  INT i_k(r,phi) dA = 1  [i_k] = m^-2.

  (F1.2)  per-channel propagation (DERIVED in the dossier; reduces EXACTLY to the pinned Giles
          alpha/g*/Gamma ODE of spec sec.1 when nbar2 is r-independent):

              dP_k/dz = u_k P_k INT [ sigma_e_k n2 - sigma_a_k (nt - n2) ] i_k dA - u_k l_k P_k
                      + u_k m h nu_k dnu_k sigma_e_k INT n2 i_k dA        (ASE channels only)

          The spontaneous seed carries the SAME overlap integral as the stimulated term: one
          photon per mode, same matrix element (Giles & Desurvire, JLT 9(2):271 (1991), the
          `m h nu dnu` form already pinned in spec sec.1).

  (F1.5)-(F1.8)  the Gaussian-mode + top-hat-dopant closed forms and the saturation-power
          correction kappa(b/w) = tanh(x^2)/x^2 are exported below as gate oracles
          (tshb_closed_form_J / tshb_mean_field_J / saturation_correction_kappa).

WHY IT MATTERS. The mean-field closure needs the variation of nbar2 ACROSS the mode to be small,
i.e. I_peak/I_sat << 1. Every efficient high-power amplifier violates that: on the dossier's
kW-class Yb LMA benchmark the mean-field model over-predicts the local gain by 4-18% (multi-dB
over a metre). For a UNIFORM (cladding) pump the deficit is strictly signed -- resolved gain <=
mean-field gain for every saturation level and every confinement (dossier sec.1.4, swept) --
because a real mode burns a hole where it is intense while the area average never sees the peak.

  SIGN EXCEPTION -- a CORE-GUIDED pump (dossier sec.1.4 caveat (b), which excludes exactly this
  case from the closed forms). The signed statement above is about the LOCAL gain of one
  saturating signal in a uniform pump field; an END-TO-END gain is not that quantity. When the
  pump is core-guided its OWN absorption saturates the same way -- it bleaches the ions hardest
  where it is brightest, on axis -- so the resolved pump absorption coefficient comes out SMALLER
  than the mean-field one, more pump survives downstream, and the resolved end-to-end gain
  EXCEEDS the mean-field gain. MEASURED here on a V = 5.41 core-pumped Yb fiber (a = 7 um,
  NA = 0.12, L = 1 m, P_s = 1 mW): resolved - mean-field = +0.335 / +0.798 / +0.207 dB at
  0.05 / 0.2 / 1.0 W of pump. Both signs are gated -- validation gate E(iii) pins the cladding-
  pump deficit, E(iv) the core-pump excess.

MODE PROFILES (dossier sec.1.5, a BINDING correction). A core-guided channel uses the EXACT LP
field of lma.mode_field whenever V > 2.405, and the Marcuse Gaussian of
waveguide.mode_field_radius_m only below that cutoff (where Marcuse is in range). At V = 8.2193 the
Gaussian's Gamma is deceptively good (1.332% off: 0.979180 against the exact LP01 0.992394) but its
SATURATION integral is wrong by up to 13% -- 0.85 dB/m of gain -- because the true LP01 at high V
is far flatter than a Gaussian.
Cladding pumps are flat, 1/A_clad over r <= R_clad, which reproduces
waveguide.cladding_pump_overlap = A_dope/A_clad by quadrature instead of by assumption.

MULTI-MODE COMPETITION (dossier sec.1.6). Several signals may carry explicit LPModes; they are
mutually incoherent (power-only, interference averaged away) and share the ONE nbar2(r,phi,z).
That is what makes the LP01/LP11 modal-gain ordering REVERSE with saturation -- LP11 peaks
off-axis where the inversion is undepleted -- which a mean-field model (constant g_11 - g_01 < 0
at every power) can never show. Originating reference: Jiang & Marciante, JOSA B 25(2):247
(2008); its equations could not be retrieved for the dossier (paywalled), so the crossover here
is DERIVED and gated numerically, not transcribed.

SCOPE / REFUSALS (v1; each raises ValueError with the reason -- see _refuse_unsupported):
  * RamanStokes coupling                       -- the SRS exchange is a channel-power coupling
                                                  that has no transverse statement yet.
  * ConcentrationModel / upconversion_C_up != 0 -- upconversion makes the local balance QUADRATIC
                                                  in nbar2 per node (dossier sec.1.4 caveat).
  * set_temperature_profile                    -- a T(z) McCumber scaling would have to become
                                                  T(r,z) to be meaningful here.
  * an ion with a NONZERO sigma_esa at any channel wavelength -- the ESA term of (F1.2) is
                                                  transcribed but deliberately not wired.
  * Er:Yb co-doping (ErYbAmplifier)            -- two coupled ion species, two balances.
  * FiberSpec.overlap_override                 -- an override REPLACES the overlap this solver
                                                  computes from the mode profile; honouring it
                                                  silently would be a lie about what was solved.
Plain two-level Er or Yb, core or cladding pumped, forward and/or backward, with an ASE band,
is SUPPORTED.

Pure numpy/scipy; SI units; ASCII only. Power-only model, so the repo's exp(-i omega t) field
convention does not enter (every quantity here is a real intensity or a real power).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp.lma import LPMode, mode_field, solve_lp_modes
from dynameta.optics.fiber_amp.rare_earth import ChannelSet
from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
# _frozen_profile_interp is the package's SINGLE HOME for the uniform-mesh relaxation
# interpolator (steady_state audit X-3: it was the largest verbatim cross-file duplicate and was
# consolidated there; eryb.py imports it the same way). Importing it is the composition this
# module is supposed to do -- re-typing it would re-open exactly that defect.
from dynameta.optics.fiber_amp.steady_state import (AseBand, FiberAmplifier, Pump, Signal,
                                                    _frozen_profile_interp,
                                                    _relaxation_residuals)
from dynameta.optics.fiber_amp.waveguide import FiberSpec, mode_field_radius_m, overlap_gamma

__all__ = ["RadialGrid", "ResolvedResult", "ResolvedFiberAmplifier", "mean_field_equivalent",
           "tshb_closed_form_J", "tshb_mean_field_J", "saturation_correction_kappa"]

# LP11 cutoff. Above it the Marcuse fit (valid ~1.2 < V < 2.4) is out of range and the exact LP
# field must be used instead -- dossier sec.1.5, the binding correction.
V_LP11_CUTOFF = 2.405

# The spelling of the synthetic MEAN-FIELD profile (see _profile_flat). A per-signal entry of
# signal_modes may be this string.
FLAT = "flat"

ModeSpec = Union[None, LPMode, str]


# ============================ radial / azimuthal quadrature =============================

def _gauss_legendre_panels(breakpoints: Sequence[float], n_nodes: int):
    """Composite Gauss-Legendre nodes + weights over the consecutive panels of `breakpoints`.

    Dossier sec.1.7 (measured): the integrand of (F1.2) has two breakpoints -- r = b, where
    nt(r) STEPS to zero, and r = a, where the exact LP field switches from J_l to K_l (psi and
    psi' continuous, psi'' not). Panel edges placed there make each panel's integrand analytic,
    and GL then converges spectrally: 16 nodes/panel already reached 9e-16 relative on the
    dossier's test integral, versus O(h) -- 3.3e-5 at 16385 nodes -- for the naive alternative of
    a step MASK on a uniform grid. That alternative is never used here."""
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    r_parts, w_parts = [], []
    for lo, hi in zip(breakpoints[:-1], breakpoints[1:]):
        if not (hi > lo):
            continue
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        r_parts.append(mid + half * x)
        w_parts.append(half * w)
    return np.concatenate(r_parts), np.concatenate(w_parts)


@dataclass(frozen=True)
class RadialGrid:
    """Quadrature grid over the fiber cross-section for the resolved balance (F1.1)/(F1.2).

    Nodes are FLATTENED (r, phi) pairs so a channel profile is one (N,) vector and every
    transverse integral is a single dot product with dA_m2:  INT f dA = sum(dA_m2 * f).

    Radially: composite Gauss-Legendre with panel edges at the geometric breakpoints
    (r = min(a,b), r = max(a,b), the inner-cladding radius when a cladding pump is present, and
    r_max), PLUS a geometric (radius-doubling) refinement of everything beyond max(a,b). That
    refinement is not cosmetic: the outer region carries only the evanescent tail, but it carries
    it into the NORMALIZATION (F1.0), and one 3 um -> 50 um panel cannot resolve a Gaussian that
    decays on 3.5 um -- MEASURED, it puts 2.9e-8 of relative error into every confinement factor
    and hence into every gain coefficient. With the doubling it is exact to round-off, at zero
    cost to the hot path (the RHS only ever touches the DOPED nodes, which are unchanged).
    Azimuthally: n_azimuthal = 1 (the whole cross-section is circularly symmetric, weight
    2 pi r dr) unless a mode with l >= 1 is present, in which case Gauss-Legendre on [0, pi/2]
    with a symmetry factor of 4. That quarter-plane reduction is exact for cos^2(l phi) profiles
    of ANY integer l -- cos^2(l phi) is invariant under phi -> -phi and phi -> pi - phi -- and the
    integrand is analytic in phi, so GL converges spectrally there too (dossier sec.1.7 item 4).
    Only cos-oriented LP modes are represented -- NOT because a sin-oriented partner would break
    the quarter-plane symmetry (it would not: sin^2(l phi) has the SAME two reflection symmetries,
    and this grid integrates it to 6.7e-16 relative for l = 1, 2, 3, MEASURED). The real
    limitation is that _build_profiles hardcodes cos(l phi)^2 and LPMode carries no orientation
    field, so the sin partner cannot be SPELLED; passing the same LPMode twice would silently
    model cos + cos, which is a different physical state (-16.5% modal gain at 200 W, measured).
    _check_modes refuses that spelling rather than answering it."""
    r_m: np.ndarray                              # (N,) node radii [m]
    phi_rad: np.ndarray                          # (N,) node azimuths [rad]
    dA_m2: np.ndarray                            # (N,) area weights [m^2], sum = pi r_max^2
    breakpoints_m: Tuple[float, ...]
    n_nodes_per_panel: int
    n_azimuthal: int

    @property
    def size(self) -> int:
        return int(self.r_m.size)

    @property
    def r_max_m(self) -> float:
        return float(self.breakpoints_m[-1])

    def integrate(self, f) -> float:
        """INT f(r,phi) dA over the whole cross-section [f] * m^2."""
        return float(np.dot(self.dA_m2, np.asarray(f, float)))

    @staticmethod
    def build(breakpoints: Sequence[float], *, n_nodes_per_panel: int = 24,
              n_azimuthal: int = 1) -> "RadialGrid":
        """Composite-GL grid over the panels defined by the sorted unique `breakpoints`
        (breakpoints[0] must be 0). n_nodes_per_panel = 24 is the dossier's ship default (16
        already machine-precision on the smooth pieces; 32 is the convergence-check setting)."""
        bp = tuple(sorted({float(b) for b in breakpoints}))
        if len(bp) < 2 or bp[0] != 0.0:
            raise ValueError("RadialGrid.build: breakpoints must start at 0.0 and span a range")
        if int(n_nodes_per_panel) < 2:
            raise ValueError("RadialGrid.build: n_nodes_per_panel must be >= 2")
        if int(n_azimuthal) < 1:
            raise ValueError("RadialGrid.build: n_azimuthal must be >= 1")
        r, wr = _gauss_legendre_panels(bp, n_nodes_per_panel)
        if int(n_azimuthal) == 1:
            return RadialGrid(r, np.zeros_like(r), 2.0 * np.pi * r * wr, bp,
                              int(n_nodes_per_panel), 1)
        xp, wp = np.polynomial.legendre.leggauss(int(n_azimuthal))
        phi = 0.25 * np.pi * (xp + 1.0)                       # [0, pi/2]
        wphi = 0.25 * np.pi * wp
        # symmetry factor 4: the quarter plane tiles [0, 2pi) for cos^2(l phi) profiles
        dA = 4.0 * np.outer(r * wr, wphi)
        return RadialGrid(np.repeat(r, phi.size), np.tile(phi, r.size), dA.ravel(), bp,
                          int(n_nodes_per_panel), int(n_azimuthal))


# ============================ closed-form TSHB oracles (F1.3)-(F1.8) ====================

def tshb_closed_form_J(n2_0, n2_inf, s0, gamma):
    """The EXACT radial integral J = INT_0^b nbar2(r) i(r) 2 pi r dr for a Gaussian LP01 mode
    i(r) = (2/(pi w^2)) exp(-2 r^2/w^2), a top-hat dopant of radius b, and a nbar2 that is a
    linear-fractional (Moebius) function of the single saturating signal intensity -- dossier
    (F1.5)/(F1.6), DERIVED there and verified against adaptive quadrature to 3.6e-13:

        J = n2_inf * Gamma + (n2_0 - n2_inf) * Phi(s0, Gamma)
        Phi(s0, Gamma) = (1/s0) * ln[ (1 + s0) / (1 + s0 (1 - Gamma)) ]

    n2_0 = the signal-free inversion, n2_inf = sa_s/(sa_s+se_s) the infinite-signal floor,
    s0 = I_peak/I_sat = 2 P_s/(pi w^2 I_sat) the ON-AXIS saturation parameter, and
    Gamma = 1 - exp(-2 b^2/w^2) = waveguide.overlap_gamma. s0 -> 0 gives Phi -> Gamma (the
    unsaturated mean-field answer), so resolved and mean-field agree exactly in the small-signal
    limit. Valid ONLY while nbar2 is Moebius in one intensity: a uniform (cladding) pump plus one
    signal band, no upconversion. Otherwise use the quadrature (this is the GATE, not the
    solver)."""
    s = np.asarray(s0, float)
    g = np.asarray(gamma, float)
    # (1/s) ln[(1+s)/(1+s(1-g))] -> g as s -> 0; guard the removable singularity explicitly
    # (a bare 0/0 would raise under the repo's filterwarnings=["error"]).
    safe = np.where(s > 0.0, s, 1.0)
    phi = np.where(s > 0.0, (np.log1p(safe) - np.log1p(safe * (1.0 - g))) / safe, g)
    return np.asarray(n2_inf, float) * g + (np.asarray(n2_0, float)
                                            - np.asarray(n2_inf, float)) * phi


def tshb_mean_field_J(n2_0, n2_inf, s0, gamma):
    """The SAME integral under the mean-field closure -- dossier (F1.7):

        J_MF = Gamma * [ n2_inf + (n2_0 - n2_inf) / (1 + s_area) ],
        s_area = <I_s>/I_sat = s0 * Gamma / (-ln(1 - Gamma)) = P_s Gamma / (pi b^2 I_sat)

    i.e. exactly what steady_state's `flux_a`/`flux_e` grouping computes. The entire
    resolved-vs-mean-field difference is the bracket Phi(s0,Gamma) - Gamma/(1+s_area), which the
    dossier sweeps (400 x 12 points, Gamma in [0.02, 0.9999] x s0 in [1e-6, 1e8]) and finds
    NON-POSITIVE everywhere: in THIS setting -- a uniform (cladding) pump plus one saturating
    signal -- mean-field never under-predicts the LOCAL gain. That is a statement about this
    bracket, NOT about an end-to-end solve. With a core-guided pump the pump's own bleaching is
    resolved too, the Moebius structure is gone (dossier sec.1.4 caveat (b)), and the resolved
    end-to-end gain can EXCEED mean-field by tenths of a dB -- see the module docstring for the
    measured numbers."""
    s = np.asarray(s0, float)
    g = np.asarray(gamma, float)
    s_area = s * g / (-np.log1p(-g))
    return g * (np.asarray(n2_inf, float)
                + (np.asarray(n2_0, float) - np.asarray(n2_inf, float)) / (1.0 + s_area))


def saturation_correction_kappa(x):
    """Effective saturation-power correction kappa(x) = tanh(x^2)/x^2 for x = b/w -- dossier
    (F1.8), DERIVED as the ratio of the two models' INITIAL saturation slopes:

        kappa = 2 Gamma / [ (2 - Gamma) (-ln(1 - Gamma)) ] == tanh(x^2)/x^2,
        Gamma = 1 - exp(-2 x^2)      ==>   P_sat_resolved = kappa * P_sat_meanfield.

    kappa < 1 for every x > 0, so the resolved model saturates EARLIER and yields LESS saturated
    gain -- the direction of the whole feature, fixed by construction rather than by argument.
    Limits: kappa -> 1 as x -> 0 (uniform intensity over a tiny dopant: mean-field is exact),
    kappa -> 1/x^2 -> 0 as x -> inf (a dopant far wider than the mode: the area average never
    saturates while the real mode burns a hole). kappa(1) = tanh(1) = 0.7615942 is the w = a
    limit; the Marcuse branch never quite reaches it, because w/a = 0.65 + 1.619 V^-1.5
    + 2.879 V^-6 >= 1.0990 for every V <= 2.405, so with the repo's default b = a the correction
    available there is up to ~18% at the V = 2.405 edge (x = 0.9099, kappa = 0.8205 -- MEASURED).
    The exact-LP branch above cutoff has no such floor."""
    xx = np.asarray(x, float) ** 2
    safe = np.where(xx > 0.0, xx, 1.0)
    return np.where(xx > 0.0, np.tanh(safe) / safe, 1.0)


# ============================ result ====================================================

@dataclass
class ResolvedResult:
    """Steady-state result of a resolved solve. The first eight fields and the `meta` keys mirror
    steady_state.SteadyStateResult exactly (so the noise.* / metrics.* consumers that duck-type a
    result can read it), plus the resolved extras.

    nbar2_z is the DOPANT-AREA-AVERAGED fraction INT_dope nbar2 dA / A_dope -- the honest scalar
    reduction of a quantity that is no longer a scalar. Anything that wants the real thing reads
    nbar2_rz / n2_rz on the grid.

    nbar2_mode_z is the OTHER scalar reduction: J_k/Gamma_k for the FIRST signal channel, i.e.
    the inversion weighted by that signal's own normalized mode intensity -- the average the gain
    coefficient of (F1.2) actually integrates. It is NaN-filled when the plan has no signal. The
    two differ exactly because the mode burns its hole where it is brightest, so
    nbar2_mode_z <= nbar2_z once anything saturates.

    ADVISORY NUMBERS. noise.local_inversion_factor -- and hence NoiseResult.n_sp_local_min /
    n_sp_local_in -- reads the DOPANT-AVERAGE nbar2_z, because that is the SteadyStateResult
    contract it duck-types. Under saturation those n_sp values therefore sit BELOW the ones the
    mode-weighted inversion would give: MEASURED on a 0.5 m, 3 um-core Yb fixture (5 W cladding
    pump, 4 ASE bins) the input-end n_sp is -0.2% off at P_s = 10 mW, -9.1% at 1 W (worst -10.3%
    along z) and -13.8% at 5 W; it is exact in the small-signal limit. Read nbar2_mode_z if the
    mode-weighted figure is what is wanted. NoiseResult.nf_dB is NOT affected -- it is built from
    the ASE power spectral density the resolved solve itself produced, not from nbar2_z."""
    z_m: np.ndarray                            # (M,) mesh
    power_W: np.ndarray                        # (K, M) power of each channel along z
    lambda_m: np.ndarray                       # (K,)
    u: np.ndarray                              # (K,) +1/-1
    is_ase: np.ndarray                         # (K,)
    kind: List[str]                            # (K,) 'pump'|'signal'|'ase'
    nbar2_z: np.ndarray                        # (M,) dopant-area-averaged metastable fraction
    signal_gain_dB: np.ndarray                 # per signal channel (= per mode, when modes given)
    grid: RadialGrid                           # the quadrature grid
    nbar2_rz: np.ndarray                       # (M, N) resolved metastable FRACTION
    nt_r_m3: np.ndarray                        # (N,) dopant density nt(r) (top-hat)
    gain_per_m: np.ndarray                     # (K, M) local NET gain coefficient g_k(z) [1/m]
    gamma_quad: np.ndarray                     # (K,) quadrature overlap INT_dope i_k dA
    nbar2_mode_z: np.ndarray                   # (M,) J/Gamma of the FIRST signal (NaN if none)
    meta: dict = field(default_factory=dict)

    @property
    def n2_rz(self) -> np.ndarray:
        """(M, N) resolved excited-ion DENSITY n2 = nt(r) * nbar2(r,z) [m^-3]."""
        return self.nbar2_rz * self.nt_r_m3[None, :]


# ============================ the resolved amplifier ====================================

class ResolvedFiberAmplifier:
    """A rare-earth fiber amplifier solved with the inversion RESOLVED across the mode.

    Constructor mirrors FiberAmplifier -- (ion, fiber, pumps, signals, ase) -- so a mean-field
    model can be re-solved with TSHB by swapping the class (and mean_field_equivalent() maps back,
    for the delta). The opt-ins FiberAmplifier accepts are accepted here TOO, purely so that the
    port fails with a physics message instead of a TypeError; every one of them is refused (see
    the module docstring).

    Extra keyword arguments:
      signal_modes  -- per-signal transverse profile, one entry per Signal:
                         None    the fundamental (exact LP01 if V > 2.405, else Marcuse Gaussian);
                         LPMode  that exact LP_lm field (lma.solve_lp_modes), the multi-mode
                                 competition case -- all signals share the ONE nbar2(r,phi,z);
                         "flat"  the synthetic MEAN-FIELD profile (see below).
      n_quad        -- Gauss-Legendre nodes per radial panel (default 24; dossier ship value).
      n_azimuthal   -- GL nodes in phi on [0, pi/2], used only when some mode has l >= 1
                       (default 32). MEASURED convergence on the hardest azimuthal case the
                       module supports -- an LP11 + LP21 pair BOTH saturating (200 W each, 1 kW
                       cladding pump, Smith & Smith LMA fiber), dP_k/dz relative to n_az = 128:
                         8 -> 4.10e-04    16 -> 4.98e-07    24 -> 2.90e-10
                        32 -> 1.51e-13    48 -> 2.16e-15    64 -> 2.87e-15
                       The repo's convergence bar for this feature is 1e-10 relative (gate G),
                       which 16 misses by three orders and 24 by a factor of ~3; 32 is the first
                       value that clears it with margin, so it is the default. A single l >= 1
                       mode needs far fewer (its integrand is one cos^2), but the default cannot
                       know that, and the axis is off entirely when every mode has l = 0.
      r_max_m       -- outer quadrature radius; default max(6a, the LP/Gaussian tail reach, and
                       the inner-cladding radius when a cladding pump is present). It may NOT be
                       set below the inner-cladding radius while a cladding pump is present --
                       the flat pump profile is renormalized over the truncated disc, which
                       INFLATES its overlap (4x / 16x / 100x at r_max = 100 / 50 / 20 um on a
                       200 um cladding, measured), so that is refused.

    The "flat" profile is i_k = Gamma_k/A_dope inside the dopant and (1 - Gamma_k)/(A_max -
    A_dope) outside, with Gamma_k = waveguide.overlap_gamma -- i.e. the mean-field closure of
    dossier sec.1.2 ("replace I_k by its area average over the doped region") written AS a
    transverse profile. With it the resolved solver reproduces steady_state.FiberAmplifier at
    every saturation level; it exists to make that degeneracy a runnable gate rather than an
    assertion, and it is also the right profile for a channel whose real mode shape is unknown.

    solve() returns a ResolvedResult. The ODE STATE is still P(z) per channel -- the balance
    (F1.1) is algebraic at every node -- so the numerics are the same relaxation (alternating
    forward / backward IVP sweeps) that steady_state.solve uses, only with the RHS gain
    coefficients evaluated by radial quadrature instead of by one confinement factor."""

    def __init__(self, ion: RareEarthIon, fiber: FiberSpec, pumps: List[Pump],
                 signals: List[Signal], ase: Optional[AseBand] = None, *,
                 signal_modes: Optional[Sequence[ModeSpec]] = None,
                 n_quad: int = 24, n_azimuthal: int = 32, r_max_m: Optional[float] = None,
                 upconversion_C_up: float = 0.0, concentration=None, raman=None):
        self.ion, self.fiber = ion, fiber
        self.pumps, self.signals = list(pumps), list(signals)
        self.ase = ase
        self.n_quad = int(n_quad)
        self.n_azimuthal = int(n_azimuthal)
        self.r_max_m = None if r_max_m is None else float(r_max_m)
        if signal_modes is None:
            self.signal_modes: List[ModeSpec] = [None] * len(self.signals)
        else:
            self.signal_modes = list(signal_modes)
            if len(self.signal_modes) != len(self.signals):
                raise ValueError(
                    "ResolvedFiberAmplifier: signal_modes has {} entries for {} signals -- it is "
                    "a PARALLEL list (one profile per Signal, None = the fundamental)".format(
                        len(self.signal_modes), len(self.signals)))
        self._check_modes()
        self._refuse_unsupported(upconversion_C_up, concentration, raman)
        self._cache = None

    # ---- scope refusals ------------------------------------------------------------------
    def _refuse_unsupported(self, upconversion_C_up, concentration, raman):
        """Every v1 scope boundary, refused at CONSTRUCTION with the physics reason. Silently
        ignoring any of these would report a number the model did not compute."""
        if not isinstance(self.ion, RareEarthIon):
            raise ValueError(
                "ResolvedFiberAmplifier: ion must be a RareEarthIon. Er:Yb co-doping "
                "(ErYbAmplifier: two ion species with a Yb->Er transfer term) has no resolved "
                "form in v1 -- it needs TWO coupled local balances, one per species.")
        if self.fiber.overlap_override is not None:
            raise ValueError(
                "ResolvedFiberAmplifier: FiberSpec.overlap_override is not supported. This "
                "solver COMPUTES the overlap by quadrature of the mode profile; an override "
                "would replace the very quantity being resolved (it is the right tool for the "
                "mean-field FiberAmplifier, e.g. Giles-calibrated fibers).")
        if raman is not None:
            raise ValueError(
                "ResolvedFiberAmplifier: RamanStokes is not supported in v1. The SRS exchange is "
                "a channel-power coupling (g_R/A_eff) with no transverse statement here -- A_eff "
                "would have to become an overlap of THREE profiles with the resolved gain.")
        if concentration is not None or float(upconversion_C_up) != 0.0:
            raise ValueError(
                "ResolvedFiberAmplifier: concentration effects (ConcentrationModel, "
                "upconversion_C_up != 0) are not supported in v1. Cooperative upconversion adds "
                "-C_up N2^2 to the balance, so (F1.1) becomes QUADRATIC in nbar2 at every "
                "quadrature node and the Moebius structure the closed-form gates rely on is "
                "gone (dossier sec.1.4 caveat).")
        lam = self._channel_table()[0]
        if self.ion.sigma_esa is not None:
            esa = np.asarray(self.ion.sigma_esa_of(lam), float)
            if float(np.max(esa)) > 0.0:
                raise ValueError(
                    "ResolvedFiberAmplifier: an ion with NONZERO excited-state absorption "
                    "(max sigma_esa = {:.3e} m^2 over the channel plan) is not supported in v1. "
                    "ESA enters (F1.2) as an extra -sigma_esa_k n2 inside the integrand; the "
                    "term is transcribed in the docs but deliberately not wired.".format(
                        float(np.max(esa))))

    def set_temperature_profile(self, *args, **kwargs):
        """REFUSED in v1. steady_state's axial T(z) McCumber scaling multiplies every
        sigma_e-proportional coefficient by mcc(z); with the inversion resolved the physically
        meaningful object is a T(r, z) (the core is hotter on axis -- that IS the thermal-lens
        radial profile of thermal.radial_temperature_rise), so a z-only profile would be applied
        uniformly across a mode whose whole point is that it is not uniform."""
        raise ValueError(
            "ResolvedFiberAmplifier: set_temperature_profile is not supported in v1 -- an axial "
            "T(z) McCumber scaling is not meaningful once the inversion is resolved across the "
            "mode; it would need to become T(r, z). Use steady_state.FiberAmplifier for the "
            "mean-field thermal-feedback loop.")

    # ---- channel plan --------------------------------------------------------------------
    def _channel_table(self):
        """(lam, u, is_ase, dnu, kind, bc, cladding) for the channel plan.

        DUPLICATED (deliberately) from steady_state.FiberAmplifier._plan -- the pump / signal /
        forward-ASE / backward-ASE ordering, the boundary values, and the FREQUENCY bin widths
        (the spontaneous term is per unit optical frequency, so the wavelength edges are mapped
        through nu = c/lambda and differenced there) are byte-for-byte the same construction.
        Reusing _plan itself is not possible without refactoring steady_state, which is out of
        scope for this feature; the two must stay in parity, and the gates compare a resolved and
        a mean-field solve channel-by-channel, which detects any divergence immediately. The
        Raman channel and the ConcentrationModel branches of _plan are absent here because both
        are refused."""
        lam, u, is_ase, dnu, kind, bc, cladding = [], [], [], [], [], [], []
        for p in self.pumps:
            lam.append(p.lambda_m); u.append(+1.0 if p.direction == "fwd" else -1.0)
            is_ase.append(False); dnu.append(0.0); kind.append("pump")
            bc.append(p.power_W); cladding.append(p.cladding)
        for s in self.signals:
            lam.append(s.lambda_m); u.append(+1.0); is_ase.append(False); dnu.append(0.0)
            kind.append("signal"); bc.append(s.power_W); cladding.append(False)
        if self.ase is not None and self.ase.n_bins > 0:
            edges = np.linspace(self.ase.lambda_min_m, self.ase.lambda_max_m,
                                self.ase.n_bins + 1)
            centres = 0.5 * (edges[:-1] + edges[1:])
            nu_edges = C_LIGHT / edges
            dnu_bins = np.abs(nu_edges[:-1] - nu_edges[1:])
            for direction in (+1.0, -1.0):
                for c, dv in zip(centres, dnu_bins):
                    lam.append(float(c)); u.append(direction); is_ase.append(True)
                    dnu.append(float(dv)); kind.append("ase"); bc.append(0.0)
                    cladding.append(False)
        return (np.asarray(lam, float), np.asarray(u, float), np.asarray(is_ase, bool),
                np.asarray(dnu, float), kind, np.asarray(bc, float), cladding)

    # ---- transverse profiles -------------------------------------------------------------
    def _fundamental_psi2(self, lam: float, r: np.ndarray) -> np.ndarray:
        """|psi(r)|^2 of the FUNDAMENTAL guided mode at lam, unnormalized.

        V > 2.405 -> the exact LP01 field of lma.mode_field (J_0 core / K_0 cladding), obtained
        from lma.solve_lp_modes (which returns the modes beta-descending, so [0] IS LP01).
        Otherwise the Marcuse Gaussian exp(-2 r^2/w^2) of waveguide.mode_field_radius_m -- which
        keeps this solver in EXACT correspondence with the mean-field FiberAmplifier in the
        single-mode regime, where both then use the same profile. Above cutoff the Gaussian is
        not merely out of its fit range, it mis-states the SATURATION integral by up to 13%
        (dossier sec.1.5) -- which is the whole reason this switch exists."""
        a, na = self.fiber.core_radius_m, self.fiber.na
        V = 2.0 * np.pi * a * na / float(lam)
        if V > V_LP11_CUTOFF:
            md = solve_lp_modes(a, na, float(lam))[0]
            psi = mode_field(md, r)
            return psi * psi
        w = float(mode_field_radius_m(a, na, float(lam)))
        return np.exp(-2.0 * r * r / (w * w))

    def _profile_flat(self, lam: float, r: np.ndarray, r_max: float) -> np.ndarray:
        """The synthetic MEAN-FIELD profile: Gamma_k/A_dope inside the dopant, the residual
        (1 - Gamma_k) spread uniformly outside it (where nt = 0, so it only carries the
        normalization). Reproduces the mean-field intensity <I_k> = Gamma_k P_k/A_dope and the
        mean-field overlap Gamma_k simultaneously, hence reproduces steady_state EXACTLY."""
        b = self.fiber.b_dope_m
        gam = float(overlap_gamma(self.fiber, np.asarray([lam], float))[0])
        a_dope = np.pi * b * b
        a_out = np.pi * (r_max * r_max - b * b)
        inside = r <= b
        out = np.where(inside, gam / a_dope, 0.0)
        if a_out > 0.0:
            out = out + np.where(inside, 0.0, (1.0 - gam) / a_out)
        return out

    def _build_profiles(self, lam, cladding, grid: RadialGrid):
        """(K, N) NORMALIZED transverse intensity profiles i_k, one row per channel, satisfying
        (F1.0) INT i_k dA = 1 on the grid EXACTLY (each row is divided by its own quadrature
        integral, so the truncation at r_max is absorbed rather than left as a silent leak; with
        r_max at 6a / 5w the discarded Gaussian tail is ~1e-26 of the power)."""
        r = grid.r_m
        K = lam.size
        n_p, n_s = len(self.pumps), len(self.signals)
        prof = np.empty((K, grid.size))
        for k in range(K):
            if k < n_p and cladding[k]:
                # dossier (F2.8) convention: a multimode cladding pump is FLAT over the inner
                # cladding. INT_dope i_p dA = A_dope/A_clad = waveguide.cladding_pump_overlap by
                # quadrature -- the pump overlap becomes an OUTPUT here, not an input.
                rc = float(self.fiber.clad_radius_m)
                raw = np.where(r <= rc, 1.0 / (np.pi * rc * rc), 0.0)
            elif n_p <= k < n_p + n_s and self.signal_modes[k - n_p] is not None:
                spec = self.signal_modes[k - n_p]
                if isinstance(spec, str):
                    raw = self._profile_flat(float(lam[k]), r, grid.r_max_m)
                else:
                    psi = mode_field(spec, r)
                    raw = psi * psi
                    if spec.l >= 1:                       # LP_lm, l>=1: cos^2(l phi) azimuthal
                        raw = raw * np.cos(spec.l * grid.phi_rad) ** 2
            else:
                raw = self._fundamental_psi2(float(lam[k]), r)
            norm = float(np.dot(grid.dA_m2, raw))
            if not (norm > 0.0):
                raise ValueError("ResolvedFiberAmplifier: channel {} profile integrates to {} -- "
                                 "the quadrature grid does not reach the mode".format(k, norm))
            prof[k] = raw / norm
        return prof

    def _check_modes(self):
        """Validate the user-supplied LPModes against the fiber and their own signal."""
        seen_lm = {}
        for i, spec in enumerate(self.signal_modes):
            if spec is None:
                continue
            if isinstance(spec, str):
                if spec != FLAT:
                    raise ValueError("ResolvedFiberAmplifier: signal_modes[{}] = {!r}; the only "
                                     "string spelling is {!r}".format(i, spec, FLAT))
                continue
            if not isinstance(spec, LPMode):
                raise ValueError("ResolvedFiberAmplifier: signal_modes[{}] must be None, an "
                                 "LPMode, or {!r} (got {!r})".format(i, FLAT, type(spec)))
            a = self.fiber.core_radius_m
            if abs(spec.core_radius_m - a) > 1e-9 * a:
                raise ValueError("ResolvedFiberAmplifier: signal_modes[{}] was solved for core "
                                 "radius {:.6e} m but the fiber has {:.6e} m".format(
                                     i, spec.core_radius_m, a))
            lam_s = self.signals[i].lambda_m
            if abs(spec.lambda_m - lam_s) > 1e-6 * lam_s:
                raise ValueError("ResolvedFiberAmplifier: signal_modes[{}] was solved at "
                                 "{:.6e} m but its Signal is at {:.6e} m".format(
                                     i, spec.lambda_m, lam_s))
            # V ties the mode to the fiber's NA, which nothing else here does: a and lambda_m
            # already match by the two checks above, so V = 2 pi a NA / lambda disagreeing can
            # ONLY be an NA mismatch. Left unchecked, an LP01 solved on NA = 0.20 and used on an
            # NA = 0.12 fiber is a tighter, more confined field than that fiber supports and ships
            # +1.00 dB of gain silently (measured, a = 3 um, 0.5 m, 5 W cladding pump).
            V_fiber = 2.0 * np.pi * a * self.fiber.na / spec.lambda_m
            if abs(spec.V - V_fiber) > 1e-9 * V_fiber:
                raise ValueError(
                    "ResolvedFiberAmplifier: signal_modes[{}] has V = {:.9f} but the fiber gives "
                    "V = 2 pi a NA / lambda = {:.9f} at that wavelength -- the mode was solved "
                    "for a different NA ({:.6f} implied, fiber has {:.6f}) and its field is not "
                    "a mode of this fiber".format(i, spec.V, V_fiber,
                                                  spec.V * spec.lambda_m / (2.0 * np.pi * a),
                                                  self.fiber.na))
            if spec.l >= 1 and (spec.l, spec.m) in seen_lm:
                raise ValueError(
                    "ResolvedFiberAmplifier: signal_modes[{}] and signal_modes[{}] are both "
                    "LP{}{} -- the degenerate cos/sin PAIR cannot be spelled in v1. LPMode "
                    "carries no orientation field and _build_profiles hardcodes cos(l phi)^2, so "
                    "passing the same mode twice would model cos + cos, which piles both signals "
                    "into the SAME azimuthal lobes: measured -16.5% modal gain versus the correct "
                    "cos + sin pair at 200 W each on the Smith & Smith LMA fixture. (The "
                    "quarter-plane grid would integrate sin^2 exactly -- 6.7e-16 relative for "
                    "l = 1, 2, 3 -- so this is a missing spelling, not a quadrature "
                    "limitation.)".format(seen_lm[(spec.l, spec.m)], i, spec.l, spec.m))
            seen_lm[(spec.l, spec.m)] = i

    def _plan(self):
        """Build (and cache) everything a solve needs: the channel table, the ChannelSet
        spectroscopy, the quadrature grid, the normalized profiles, and the hoisted per-channel
        coefficient bundle. The bundle mirrors steady_state._coeffs (same regrouping, same hot
        path) except that the flux and gain coefficients are per-NODE rather than per-channel."""
        if self._cache is not None:
            return self._cache
        lam, u, is_ase, dnu, kind, bc, cladding = self._channel_table()
        if lam.size == 0:
            raise ValueError("ResolvedFiberAmplifier: the channel plan is empty (no pumps, no "
                             "signals, no ASE band)")
        ch = ChannelSet.build(self.ion, self.fiber, lam, u, is_ase=is_ase, dnu_hz=dnu)

        a, b = self.fiber.core_radius_m, self.fiber.b_dope_m
        r_max = self.r_max_m if self.r_max_m is not None else self._default_r_max(lam, cladding)
        if r_max <= max(a, b):
            raise ValueError("ResolvedFiberAmplifier: r_max_m = {:.6e} m does not exceed "
                             "max(core, dopant) radius {:.6e} m -- the quadrature would truncate "
                             "the doped region".format(r_max, max(a, b)))
        if any(cladding) and r_max < float(self.fiber.clad_radius_m):
            # A cladding pump is FLAT over the inner cladding and _build_profiles normalizes it on
            # the grid, so truncating the grid inside r = R_clad does not clip the pump -- it
            # RENORMALIZES it over the smaller disc, silently inflating Gamma_p by
            # (R_clad/r_max)^2 (measured 4x / 16x / 100x at r_max = 100 / 50 / 20 um on a 200 um
            # cladding). Refuse rather than answer a different fiber.
            raise ValueError(
                "ResolvedFiberAmplifier: r_max_m = {:.6e} m is inside the inner-cladding radius "
                "{:.6e} m while a cladding pump is present. The flat pump profile would be "
                "renormalized over the truncated disc instead of over the cladding, inflating "
                "its overlap by (R_clad/r_max)^2 = {:.4g}x. Raise r_max_m to at least the "
                "inner-cladding radius (the default already does).".format(
                    r_max, float(self.fiber.clad_radius_m),
                    (float(self.fiber.clad_radius_m) / r_max) ** 2))
        breaks = {0.0, min(a, b), max(a, b), r_max}
        if any(cladding):
            breaks.add(float(self.fiber.clad_radius_m))
        x = max(a, b)                             # radius-doubling refinement of the tail region
        while x * 2.0 < r_max:
            x *= 2.0
            breaks.add(x)
        breaks = [p for p in breaks if p <= r_max]
        n_phi = self.n_azimuthal if any(
            isinstance(s, LPMode) and s.l >= 1 for s in self.signal_modes) else 1
        grid = RadialGrid.build(breaks, n_nodes_per_panel=self.n_quad, n_azimuthal=n_phi)

        prof = self._build_profiles(lam, cladding, grid)
        nt_r = np.where(grid.r_m <= b, self.fiber.n_t_m3, 0.0)
        w_dope = np.where(grid.r_m <= b, grid.dA_m2, 0.0)
        keep = w_dope > 0.0                       # only doped nodes enter ANY integral (nt = 0
        idope = prof[:, keep]                     # outside, so both n2 and (nt - n2) vanish)
        wdope = w_dope[keep]
        hnu = H_PLANCK * ch.nu_hz
        m = self.ase.m_modes if self.ase is not None else 2
        c = {
            "u": u,
            "tau": float(ch.tau_s),
            "sa": ch.sigma_a, "se": ch.sigma_e,
            "loss": ch.loss_per_m,
            "fa": ch.sigma_a / hnu,                       # R_a = SUM_k fa_k P_k i_k(node) [1/s]
            "fe": ch.sigma_e / hnu,
            "gam": idope @ wdope,                         # (K,) Gamma_k = INT_dope i_k dA
            "s_pref": np.where(is_ase, m * H_PLANCK * ch.nu_hz * ch.dnu_hz * ch.sigma_e
                               * self.fiber.n_t_m3, 0.0),
            "m_modes": m,
            "nt": float(self.fiber.n_t_m3),
            "idope": idope, "wdope": wdope, "prof": prof, "keep": keep,
        }
        self._cache = (ch, bc, u, is_ase, kind, grid, nt_r, c)
        return self._cache

    def _default_r_max(self, lam, cladding) -> float:
        """Outer quadrature radius: at least 6a (the lma._overlap_grid reach), plus each guided
        channel's own tail -- a(1 + 12/W) for an exact LP mode, 5w for a Gaussian (exp(-50) of
        the peak) -- and the inner cladding when a cladding pump has to be normalized."""
        a, na = self.fiber.core_radius_m, self.fiber.na
        r_max = max(6.0 * a, 1.2 * self.fiber.b_dope_m)
        for k, lm in enumerate(lam):
            if k < len(cladding) and k < len(self.pumps) and cladding[k]:
                continue
            V = 2.0 * np.pi * a * na / float(lm)
            if V > V_LP11_CUTOFF:
                md = solve_lp_modes(a, na, float(lm))[0]
                r_max = max(r_max, a * (1.0 + 12.0 / max(md.W, 0.25)))
            else:
                r_max = max(r_max, 5.0 * float(mode_field_radius_m(a, na, float(lm))))
        for spec in self.signal_modes:
            if isinstance(spec, LPMode):
                r_max = max(r_max, a * (1.0 + 12.0 / max(spec.W, 0.25)))
        if any(cladding):
            r_max = max(r_max, float(self.fiber.clad_radius_m))
        return float(r_max)

    # ---- pointwise physics ----------------------------------------------------------------
    def _nbar2_nodes(self, c, P):
        """(F1.1) at every DOPED quadrature node from the local channel powers P (K,):
        nbar2 = tau R_a / (1 + tau (R_a + R_e)) with R_a, R_e the per-node absorption and
        stimulated-emission rates. This is steady_state._nbar2_c with the single area-averaged
        intensity replaced by the true local one -- the entire content of the feature."""
        R_a = (c["fa"] * P) @ c["idope"]
        R_e = (c["fe"] * P) @ c["idope"]
        tau = c["tau"]
        return tau * R_a / (1.0 + tau * (R_a + R_e))

    def _overlaps(self, c, P):
        """(K,) J_k = INT_dope nbar2(r,phi) i_k dA, the resolved replacement for Gamma_k nbar2."""
        return c["idope"] @ (self._nbar2_nodes(c, P) * c["wdope"])

    def _gain_from_J(self, c, J):
        """(K,) net gain coefficient [1/m] from the resolved overlaps -- the bracket of (F1.2):
        g_k = nt [ (sigma_a_k + sigma_e_k) J_k - sigma_a_k Gamma_k ] - l_k. With nbar2 constant
        over the dopant, J_k = Gamma_k nbar2 and this is IDENTICALLY the Giles form
        (alpha_k + g*_k) nbar2 - alpha_k - l_k of spec sec.1. (ESA would add -nt sigma_esa_k J_k
        here; it is refused in v1.)"""
        return c["nt"] * ((c["sa"] + c["se"]) * J - c["sa"] * c["gam"]) - c["loss"]

    def _dP(self, c, P):
        """dP_k/dz [W/m] for every channel from the local power vector P (K,) -- (F1.2). The
        spontaneous seed m h nu dnu sigma_e nt J_k carries the SAME overlap J_k as the stimulated
        term (one photon per mode), and is nonzero only on ASE channels."""
        P = np.maximum(P, 0.0)
        J = self._overlaps(c, P)
        return c["u"] * (self._gain_from_J(c, J) * P + c["s_pref"] * J)

    # ---- public diagnostics ---------------------------------------------------------------
    def local_gain_per_m(self, powers_W) -> np.ndarray:
        """(K,) local NET gain coefficient g_k [1/m] at one z, given the full channel power
        vector. For a plan whose signals carry explicit LPModes these ARE the per-mode modal
        gains -- the quantity whose LP01/LP11 ordering reverses with saturation."""
        _, _, _, _, _, _, _, c = self._plan()
        P = np.maximum(np.asarray(powers_W, float), 0.0)
        return self._gain_from_J(c, self._overlaps(c, P))

    def local_nbar2(self, powers_W) -> np.ndarray:
        """(N,) resolved metastable FRACTION at every grid node for one local power vector.
        Nodes outside the dopant carry the balance evaluated with the local intensities there
        (nt = 0, so no ions actually occupy them and nothing downstream uses the value)."""
        _, _, _, _, _, _, _, c = self._plan()
        P = np.maximum(np.asarray(powers_W, float), 0.0)
        R_a = (c["fa"] * P) @ c["prof"]
        R_e = (c["fe"] * P) @ c["prof"]
        return c["tau"] * R_a / (1.0 + c["tau"] * (R_a + R_e))

    def dP_dz(self, powers_W) -> np.ndarray:
        """(K,) dP_k/dz [W/m] at one z from the full local power vector -- (F1.2)."""
        _, _, _, _, _, _, _, c = self._plan()
        return self._dP(c, np.asarray(powers_W, float))

    @property
    def grid(self) -> RadialGrid:
        """The quadrature grid this amplifier integrates on."""
        return self._plan()[5]

    @property
    def profiles(self) -> np.ndarray:
        """(K, N) normalized transverse intensity profiles i_k(r,phi) [m^-2], INT i_k dA = 1."""
        return self._plan()[7]["prof"]

    @property
    def gamma_quad(self) -> np.ndarray:
        """(K,) confinement factors BY QUADRATURE, Gamma_k = INT_dope i_k dA. For a cladding
        pump this equals waveguide.cladding_pump_overlap by construction; for a single-mode
        core channel it equals waveguide.overlap_gamma."""
        return self._plan()[7]["gam"]

    # ---- the relaxation solve --------------------------------------------------------------
    def solve(self, *, n_nodes: int = 201, max_iter: int = 200, tol: float = 1e-6,
              method: str = "LSODA", relax: float = 1.0) -> ResolvedResult:
        """Steady-state solve by the SAME relaxation as steady_state.solve: alternately integrate
        the forward channels 0->L and the backward channels L->0 as initial-value problems
        (scipy.integrate.solve_ivp), freezing the other direction's z-profile each half-sweep,
        until the endpoint powers AND the full interior profiles converge. The state is P(z)
        alone -- (F1.1) is algebraic at every node -- so the ODE set stays first order and the
        only difference from the mean-field solver is that each RHS evaluation does a radial
        quadrature instead of one multiplication by Gamma_k.

        UNDER-RELAXATION (`relax` in (0, 1], audit F-14). Because this is the same iteration, it
        has the same failure mode, and it was shipped without the remedy: on a COUNTER-PUMPED
        fiber the Gauss-Seidel oscillation stops decaying and the sweep can settle on a spurious,
        essentially unpumped branch. Measured on a 3 um-core / NA 0.12 / 6e25 m^-3 / 6 m Yb fiber,
        2 W counter-pumped at 976 nm with 20 mW in at 1060 nm: relax = 1 returns -30.2 dB with
        `converged` False, while relax <= 0.3 converges to the same +17.9 dB fixed point the
        mean-field twin finds -- a 48 dB gap that the resolved solver had no knob to close.
        `relax` blends each pass with the previous profile,
        `profile <- relax * (this pass) + (1 - relax) * (previous)`, which changes the PATH to the
        fixed point and not the fixed point itself. relax = 1.0 (the default) skips the blend
        entirely, so it is byte-identical to this method's pre-fix behaviour.

        `meta` carries `endpoint_residual` / `profile_residual` / `relax` / `min_power_W`, the same
        four diagnostics steady_state.solve reports, so a caller can see HOW converged a solve is
        rather than only whether it tripped `tol`."""
        from scipy.integrate import solve_ivp
        if not (0.0 < float(relax) <= 1.0):
            raise ValueError("solve: relax must be in (0, 1]; got %r" % (relax,))
        relax = float(relax)
        ch, bc, u, is_ase, kind, grid, nt_r, c = self._plan()
        L = self.fiber.length_m
        z = np.linspace(0.0, L, n_nodes)
        fwd = np.where(u > 0)[0]
        bwd = np.where(u < 0)[0]
        P_bwd = (np.repeat(bc[bwd][:, None], n_nodes, axis=1) if bwd.size
                 else np.zeros((0, n_nodes)))
        P_fwd = np.repeat(bc[fwd][:, None], n_nodes, axis=1)

        def _assemble(Pf, Pb):
            P = np.empty(ch.lambda_m.size)
            P[fwd] = Pf
            if bwd.size:
                P[bwd] = Pb
            return P

        last_out = None
        last_prof = None
        converged = False
        end_resid = prof_resid = float("nan")     # final residuals, reported on meta (audit F-14)
        it = 0
        for it in range(max_iter):
            bwd_of = _frozen_profile_interp(P_bwd, z, L, n_nodes) if bwd.size else None

            def rhs_f(zz, Pf):
                Pb = bwd_of(zz) if bwd.size else np.zeros(0)
                return self._dP(c, _assemble(Pf, Pb))[fwd]

            sf = solve_ivp(rhs_f, (0.0, L), bc[fwd], t_eval=z, method=method,
                           rtol=1e-7, atol=1e-15)
            # Under-relaxation (audit F-14), mirroring steady_state._solve_once: the relax == 1.0
            # branch is taken verbatim so the default path performs NO arithmetic on the arrays
            # and stays byte-identical -- a `1.0 * new + 0.0 * old` blend would not, because
            # 0.0 * inf is NaN and -0.0 + 0.0 flips a sign.
            P_fwd = sf.y if relax == 1.0 else relax * sf.y + (1.0 - relax) * P_fwd

            if bwd.size:
                fwd_of = _frozen_profile_interp(P_fwd, z, L, n_nodes)

                def rhs_b(zz, Pb):
                    return self._dP(c, _assemble(fwd_of(zz), Pb))[bwd]

                sb = solve_ivp(rhs_b, (L, 0.0), bc[bwd], t_eval=z[::-1], method=method,
                               rtol=1e-7, atol=1e-15)
                new_bwd = sb.y[:, ::-1]
                P_bwd = new_bwd if relax == 1.0 else relax * new_bwd + (1.0 - relax) * P_bwd

            # SAME convergence test as steady_state.solve -- literally the same function now
            # (audit F-13: this was a byte-identical COPY, so the endpoint-denominator defect
            # existed in triplicate and fixing one solver would have desynchronised the iteration
            # counts that the mean-field-vs-resolved 1e-9 reduction gates compare).
            out = np.concatenate([P_fwd[:, -1], (P_bwd[:, 0] if bwd.size else [])])
            prof = np.concatenate([P_fwd, P_bwd], axis=0) if bwd.size else P_fwd.copy()
            if last_out is not None:
                end_resid, prof_resid = _relaxation_residuals(out, last_out, prof, last_prof)
                if end_resid < tol and prof_resid < tol:
                    converged = True
                    break
            last_out = out
            last_prof = prof

        P = np.empty((ch.lambda_m.size, n_nodes))
        P[fwd] = P_fwd
        if bwd.size:
            P[bwd] = P_bwd

        # resolved inversion, its dopant-area average, and the local modal gains along z
        nbar2_rz = np.empty((n_nodes, grid.size))
        gain = np.empty((ch.lambda_m.size, n_nodes))
        a_dope = self.fiber.a_dope_m2
        nbar2_z = np.empty(n_nodes)
        sig_idx = [i for i, k in enumerate(kind) if k == "signal"]
        # the MODE-weighted reduction J_k/Gamma_k of the first signal -- the average the gain
        # coefficient of (F1.2) integrates, as opposed to the dopant-area average nbar2_z that
        # the SteadyStateResult contract (and hence noise.local_inversion_factor) reads.
        k_mode = sig_idx[0] if sig_idx else -1
        nbar2_mode_z = np.full(n_nodes, np.nan)
        for j in range(n_nodes):
            Pj = np.maximum(P[:, j], 0.0)
            R_a = (c["fa"] * Pj) @ c["prof"]
            R_e = (c["fe"] * Pj) @ c["prof"]
            nb = c["tau"] * R_a / (1.0 + c["tau"] * (R_a + R_e))
            nbar2_rz[j] = nb
            nbar2_z[j] = float(np.dot(c["wdope"], nb[c["keep"]]) / a_dope)
            J = c["idope"] @ (nb[c["keep"]] * c["wdope"])
            gain[:, j] = self._gain_from_J(c, J)
            if k_mode >= 0:
                nbar2_mode_z[j] = float(J[k_mode] / c["gam"][k_mode])

        gains_dB = np.array([10.0 * np.log10(P[i, -1] / bc[i]) for i in sig_idx])
        return ResolvedResult(
            z, P, ch.lambda_m, u, is_ase, kind, nbar2_z, gains_dB, grid, nbar2_rz, nt_r, gain,
            c["gam"].copy(), nbar2_mode_z,
            meta={"converged": converged, "iterations": it + 1,
                  # audit F-14, lifted here from steady_state: the ACTUAL final residuals, the
                  # relax value that produced them, and the most negative returned power (a fully
                  # absorbed channel sitting in integrator noise -- physically zero, but it will
                  # surprise a log or a sqrt).
                  "endpoint_residual": end_resid,
                  "profile_residual": prof_resid,
                  "relax": float(relax),
                  "min_power_W": float(np.min(P)),
                  "dnu_hz": ch.dnu_hz.copy(),
                  "sigma_a": ch.sigma_a.copy(),
                  "sigma_e": ch.sigma_e.copy(),
                  "sigma_esa": ch.sigma_esa.copy(),
                  # `gamma` is the QUADRATURE overlap actually used, not overlap_gamma -- the
                  # SteadyStateResult key name is kept so the noise.* consumers read the right
                  # thing. Those consumers combine it with the SCALAR nbar2_z, i.e. they see the
                  # mean-field reduction of this solve, which is the honest read of an n_sp that
                  # is itself a per-mode average.
                  "gamma": c["gam"].copy(),
                  "mcc": None,
                  "m_modes": c["m_modes"],
                  "resolved": True,
                  "n_quad": self.n_quad,
                  "n_azimuthal": grid.n_azimuthal,
                  "r_max_m": grid.r_max_m,
                  "n_grid": grid.size})


def mean_field_equivalent(amp: ResolvedFiberAmplifier) -> FiberAmplifier:
    """The Gamma-folded steady_state.FiberAmplifier for the SAME channel plan -- the mean-field
    twin of a resolved amplifier. Use it to read the TSHB delta (resolved gain minus mean-field
    gain, <= 0 for a UNIFORM (cladding) pump; a core-guided pump bleaches its OWN absorption
    non-uniformly and flips the end-to-end sign -- see the module docstring) and as the oracle for
    the reduction gates: in the small-signal limit, and for a flat-intensity plan at ANY
    saturation level, the two solvers must agree. Every opt-in of FiberAmplifier is absent by
    construction because ResolvedFiberAmplifier refuses all of them."""
    return FiberAmplifier(amp.ion, amp.fiber, amp.pumps, amp.signals, amp.ase)
