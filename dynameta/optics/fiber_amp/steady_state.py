"""Steady-state z-resolved fiber-amplifier solve: the two-point boundary value problem for the
coupled pump / signal / forward+backward-ASE powers along the fiber (docs sec.1-2). Forward
channels are seeded at z=0, backward channels at z=L; the local metastable fraction nbar2(z) is
algebraic in the local powers, so the state is P(z) alone and the ODE set is first order.

Solved by RELAXATION (the standard EDFA numerical method): alternately integrate the
forward-propagating channels 0->L and the backward-propagating channels L->0 as initial-value
problems (scipy.integrate.solve_ivp), freezing the other direction's z-profile each half-step,
until the endpoint powers converge. Each pass is a stable IVP -- unlike a single two-point
solve_bvp over all channels, whose Newton iteration overflows on the ASE that grows from the
spontaneous floor through tens of dB of gain. Pure numpy/scipy; SI units.
docs/fiber_amp_model_spec.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK, KB
from dynameta.optics.fiber_amp.rare_earth import ChannelSet
from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
from dynameta.optics.fiber_amp.waveguide import FiberSpec, mode_field_radius_m

__all__ = ["Pump", "Signal", "AseBand", "RamanStokes", "FiberAmplifier", "SteadyStateResult",
           "ChannelPlan"]

# "carry the current value through the clone" sentinel: distinguishes "not supplied" from an
# explicit None (ase=None DROPS the ASE band -- see FiberAmplifier.without_ase).
_KEEP = object()

# Relative floor for the ENDPOINT convergence test, as a fraction of each channel's OWN peak power
# (audit 2026-08-04 F-13). The endpoint test used to floor its denominator at an ABSOLUTE 1e-15 W,
# which made meta['converged'] a permanent FALSE NEGATIVE on every amplifier whose pump is fully
# absorbed -- i.e. on every efficient design. A fully absorbed pump reaches z = L attenuated by
# thousands of dB, so its endpoint is pure integrator noise of order 1e-16 W (measured on the Yb
# reference: -1.215e-16 W after 3003 dB, i.e. NEGATIVE), and dividing a 1e-16-level difference by
# the 1e-15 floor gives a residual of ~0.1 forever against tol = 1e-6. Measured consequence: the
# flag never tripped through 3000 iterations while the signal gain was stable to 5 decimals, and
# the solve burned 200 iterations instead of ~60.
#
# The fix makes the endpoint test consistent with the INTERIOR-profile test beside it, which
# already normalises each channel by its own peak and works correctly. 1e-9 sits three decades
# below the default tol = 1e-6, so any channel carrying real information is still fully tested;
# only channels that have decayed nine decades below their own peak -- which carry no physical
# information -- are neutralised.
_ENDPOINT_FLOOR_FRAC = 1e-9

# The escalation ladder used by solve(relax="auto") -- the DEFAULT. Try the undamped iteration
# first, because it is 7-12x faster when it works (measured: 4 iterations vs 29-50 on the Er and Yb
# boosters, for gains identical to 3 decimals), and only damp when it does not converge.
#
# Why a ladder rather than a fixed damped value: under-relaxation costs iterations on EVERY problem
# but is only NEEDED on strongly counter-coupled ones. Measured on a 3 um-core Yb fiber at 1060 nm
# with 20 mW in: co-pumped converges at relax = 1 in 4 iterations, while the same fiber
# COUNTER-pumped needs relax <= 0.3 (relax = 1 returns -30.8 dB, wrong by 49 dB, and relax = 0.5
# merely wanders). A fixed 0.5 would therefore be simultaneously too slow for the common case and
# too weak for the hard one. The ladder is as fast as the first entry when that works and as robust
# as the last when it does not.
_RELAX_LADDER = (1.0, 0.5, 0.25)


def _relaxation_residuals(out, last_out, prof, last_prof):
    """(endpoint_residual, profile_residual) for one relaxation step -- the SINGLE HOME of the
    convergence test shared by FiberAmplifier.solve, ErYbAmplifier.solve and
    ResolvedFiberAmplifier.solve (audit X-3's pattern, applied to the third verbatim duplicate in
    this package). All three carried a byte-identical copy of this algebra, so the audit-F-13
    defect existed in triplicate and a fix to one would have silently desynchronised the
    iteration counts that the cross-solver 1e-9 reduction gates compare.

    Both residuals normalise each channel by ITS OWN PEAK power along z:
      * profile : max_k,z |prof - last_prof| / peak_k       (unchanged; audit S3-34)
      * endpoint: max_k   |out - last_out|  / max(|out_k|, _ENDPOINT_FLOOR_FRAC * peak_k)
    The endpoint denominator used to be floored at an ABSOLUTE 1e-15 W, which is why `converged`
    could never become True once any channel was attenuated below the integrator's own noise
    level -- see _ENDPOINT_FLOOR_FRAC.

    `prof` rows and `out` entries must share one channel ordering ([fwd..., bwd...] in all three
    callers), so that peak_k lines up with out_k.
    """
    ch_peak = np.maximum(np.max(np.abs(prof), axis=1, keepdims=True), 1e-300)
    denom = np.maximum(np.abs(out), _ENDPOINT_FLOOR_FRAC * ch_peak[:, 0])
    end_resid = float(np.max(np.abs(out - last_out) / denom))
    prof_resid = float(np.max(np.abs(prof - last_prof) / ch_peak))
    return end_resid, prof_resid


def _frozen_profile_interp(Y, z, L: float, n_nodes: int):
    """Lean frozen-profile interpolator on the UNIFORM mesh z = linspace(0, L, n_nodes): returns
    f(zz) -> Y[:, zz] by piecewise-linear lookup, for the relaxation sweeps that propagate one
    beam direction through a frozen profile of the other.

    Why not scipy.interpolate.interp1d (audit S6-2): the mesh is uniform, so interp1d's per-call
    validation machinery is ~43% of the solve runtime -- pure overhead here. The endpoint clamps
    reproduce interp1d's fill_value=(left, right) EXACTLY; the interior uses the identical linear
    form, and LSODA samples strictly inside (0, L), so results are unchanged.

    SINGLE HOME (audit X-3): this was the repo's largest verbatim cross-file duplicate -- copied
    into eryb.py (which already imports from this module), provenance comment and all, so the
    endpoint-clamp / uniform-mesh correctness contract was carried twice. Both solvers now call
    this one function; the arithmetic is unchanged, so both remain bit-identical to the copies."""
    inv_dz = (n_nodes - 1) / L
    slopes = (Y[:, 1:] - Y[:, :-1]) * inv_dz
    ncap = n_nodes - 2

    def f(zz):
        if zz <= 0.0:
            return Y[:, 0]
        if zz >= L:
            return Y[:, -1]
        j = int(zz * inv_dz)
        if j > ncap:
            j = ncap
        return Y[:, j] + slopes[:, j] * (zz - z[j])
    return f


@dataclass(frozen=True)
class Pump:
    """A pump beam: power [W], wavelength [m], direction 'fwd' (co, seeded at z=0) or 'bwd'
    (counter, seeded at z=L). For double-clad pumping the FiberSpec carries clad_radius_m and
    the overlap is A_DOPE/A_clad (audit S3-9 -- A_core/A_clad only for uniform core doping) --
    set cladding=True to use it for this pump."""
    power_W: float
    lambda_m: float
    direction: str = "fwd"
    cladding: bool = False


@dataclass(frozen=True)
class Signal:
    """A forward signal: power [W] at z=0, wavelength [m]."""
    power_W: float
    lambda_m: float


@dataclass(frozen=True)
class AseBand:
    """The spectral band over which ASE is resolved (both propagation directions): [lambda_min,
    lambda_max] in n_bins bins. m_modes = 2 (both polarizations). Set n_bins=0 to disable ASE
    (a fast ASE-free gain estimate)."""
    lambda_min_m: float
    lambda_max_m: float
    n_bins: int = 40
    m_modes: int = 2


@dataclass(frozen=True)
class RamanStokes:
    """Opt-in stimulated-Raman-scattering Stokes channel coupled INTO the steady-state solve
    (the amplified signal is the Raman pump). Adds one channel (kind 'stokes') at the silica
    Stokes shift below the chosen signal, propagating 'fwd' (co, threshold exponent ~16) or
    'bwd' (counter, ~20), with the coupled-power exchange
        dP_S/dz  = [rare-earth gain/loss at lambda_S] P_S + (g_R/A_eff) P_sig (P_S + q_seed)
        dP_sig/dz -= (nu_sig/nu_S) (g_R/A_eff) P_sig (P_S + q_seed)
    where the (nu_sig/nu_S) factor is Manley-Rowe photon conservation (each Raman event converts
    ONE signal photon into ONE Stokes photon + a phonon) and q_seed = h nu_S dnu_eff (n_th + 1)
    is the distributed spontaneous-Raman seed (one photon per mode over the lumped Stokes bucket
    bandwidth dnu_eff, times the phonon thermal factor n_th ~ 0.14 at the 13.2 THz shift, 300 K).
    The Stokes channel ALSO sees the rare-earth medium at its own wavelength (cross-sections,
    background loss) -- the reason to couple it into the solver rather than post-process.
    Refs: Smith, Appl. Opt. 11:2489 (1972); Kobyakov/Sauer/Chowdhury, Adv. Opt. Photon. 2:1
    (2010); Agrawal, Nonlinear Fiber Optics ch. 8; Bromage, JLT 22:79 (2004).

    g_r_m_w None -> the silica peak 1.0e-13 * (1 um / lambda_signal) [m/W] (Stolen-Ippen scaling);
    a_eff_m2 None -> pi w^2 at the signal (Gaussian LP01); dnu_eff_hz = the lumped bucket width
    (Raman gain FWHM ~5 THz -- a stated modeling convention); shift_hz = 13.2 THz silica peak."""
    signal_index: int = 0
    direction: str = "fwd"
    g_r_m_w: Optional[float] = None
    a_eff_m2: Optional[float] = None
    dnu_eff_hz: float = 5.0e12
    shift_hz: float = 13.2e12
    T_K: float = 300.0


@dataclass(frozen=True)
class ChannelPlan:
    """The PUBLIC view of an amplifier's channel structure (audit 2026-08-04 F-3).

    `ChannelSet` and `ChannelSet.build` are public but the ASSEMBLY of them was not: `_plan()` is
    private, so an external consumer that needed the channel table had to re-derive the ASE bin
    edges, the wavelength->FREQUENCY bin-width conversion (`dnu = |c/edge_i - c/edge_(i+1)|`, easy
    to get wrong as a wavelength width), the forward/backward channel duplication and the
    cladding-pump overlap override -- a copy that then silently rots whenever `_plan` changes.

    Every array is indexed by channel and shares one ordering: pumps, then signals, then the ASE
    band (all forward bins, then all backward bins), then any Raman Stokes channel. That is the
    same ordering as `SteadyStateResult.power_W`'s rows, so a plan index is a result index.

      lambda_m     (K,) channel wavelengths [m]
      direction    (K,) +1 forward (seeded at z=0) / -1 backward (seeded at z=L)
      is_ase       (K,) bool -- the channels that carry a spontaneous-emission source
      dnu_hz       (K,) ASE bin width in FREQUENCY [Hz]; 0 for pump/signal/Stokes
      kind         (K,) 'pump' | 'signal' | 'ase' | 'stokes'
      launched_W   (K,) boundary power [W]: forward at z=0, backward at z=L, 0 for ASE
      gamma        (K,) mode/dopant overlap, with the cladding-pump override already applied
      loss_per_m   (K,) background loss [1/m]
      m_modes      (K,) spontaneous-emission mode count (2 = both polarizations) for ASE channels
      channels     the single-ion ChannelSet (cross-sections, lifetime). None for a CO-DOPED
                   amplifier, where each channel carries two ions' cross-sections and there is no
                   single ChannelSet to hand back -- use ErYbAmplifier._plan() for those.
    """
    lambda_m: np.ndarray
    direction: np.ndarray
    is_ase: np.ndarray
    dnu_hz: np.ndarray
    kind: List[str]
    launched_W: np.ndarray
    gamma: np.ndarray
    loss_per_m: np.ndarray
    m_modes: np.ndarray
    channels: Optional[ChannelSet] = None

    @property
    def nu_hz(self) -> np.ndarray:
        """Channel optical frequencies [Hz]."""
        return C_LIGHT / self.lambda_m

    def indices(self, kind: str, direction: Optional[str] = None) -> np.ndarray:
        """Channel indices of a given kind, optionally restricted to 'fwd' or 'bwd'.

        `plan.indices('ase', 'fwd')` is the forward ASE band; because the plan ordering matches
        `SteadyStateResult.power_W`'s rows, `result.power_W[plan.indices('ase', 'fwd'), -1]` is the
        output forward-ASE spectrum.
        """
        sel = np.array([k == kind for k in self.kind], bool)
        if direction is not None:
            if direction not in ("fwd", "bwd"):
                raise ValueError("direction must be 'fwd', 'bwd' or None; got %r" % (direction,))
            sel &= (self.direction > 0.0) if direction == "fwd" else (self.direction < 0.0)
        return np.where(sel)[0]


@dataclass
class SteadyStateResult:
    z_m: np.ndarray                            # (M,) mesh
    power_W: np.ndarray                        # (K, M) power of each channel along z
    lambda_m: np.ndarray                       # (K,)
    u: np.ndarray                              # (K,) +1/-1
    is_ase: np.ndarray                         # (K,)
    kind: List[str]                            # (K,) 'pump'|'signal'|'ase'
    nbar2_z: np.ndarray                        # (M,) metastable fraction profile
    signal_gain_dB: np.ndarray                 # per signal channel
    meta: dict = field(default_factory=dict)


class FiberAmplifier:
    """A rare-earth fiber amplifier: an ion + fiber + a channel plan (pumps, signals, an ASE
    band). solve() returns the steady-state z-profiles, signal gain, and the output ASE
    spectrum. Concentration/degradation effects (upconversion, pair-induced quenching,
    photodarkening -- Phase 5, via a ConcentrationModel or the raw upconversion_C_up) and
    cladding pumping (via FiberSpec.clad_radius_m + Pump.cladding) are opt-in; with
    concentration=None and upconversion_C_up=0 the solve is the ideal model."""

    def __init__(self, ion: RareEarthIon, fiber: FiberSpec, pumps: List[Pump],
                 signals: List[Signal], ase: Optional[AseBand] = None, *,
                 upconversion_C_up: float = 0.0, concentration=None,
                 raman: Optional[RamanStokes] = None):
        self.ion, self.fiber = ion, fiber
        self.pumps, self.signals = list(pumps), list(signals)
        self.ase = ase
        self.raman = raman
        self._raman_map = None                      # filled by _plan when raman is active
        self._Tz = None                             # optional axial T profile (set_temperature_profile)
        if raman is not None and not (0 <= raman.signal_index < len(self.signals)):
            raise ValueError("RamanStokes.signal_index out of range")
        # an all-default (identity) model collapses to the None path: truly byte-identical
        if concentration is not None and getattr(concentration, "is_identity", False):
            concentration = None
        self.concentration = concentration
        if concentration is not None:
            if float(upconversion_C_up) != 0.0 and \
                    float(upconversion_C_up) != float(concentration.c_up_m3_s):
                import warnings
                warnings.warn("FiberAmplifier: upconversion_C_up=%.3g is OVERRIDDEN by "
                              "ConcentrationModel.c_up_m3_s=%.3g (the model wins; pass "
                              "C_up through the ConcentrationModel only -- audit B1)"
                              % (float(upconversion_C_up), concentration.c_up_m3_s),
                              stacklevel=2)
            self.upconversion_C_up = float(concentration.c_up_m3_s)
            self._n_active = concentration.active_density(fiber.n_t_m3)
            self._n_dark = concentration.dark_density(fiber.n_t_m3)
        else:
            self.upconversion_C_up = float(upconversion_C_up)
            self._n_active = fiber.n_t_m3
            self._n_dark = 0.0

    # ---- type-preserving clone protocol ---------------------------------------------------
    def _clone(self, *, pumps: Optional[List[Pump]] = None,
               signals: Optional[List[Signal]] = None, ase=_KEEP) -> "FiberAmplifier":
        """Clone this amplifier with the pump list, signal list and/or ASE band swapped, carrying
        EVERY opt-in through: the ASE band, the ConcentrationModel, the normalised
        upconversion_C_up (audit S3-1/A-1: the raw-C_up spelling was dropped by clones that only
        forwarded the concentration model), the RamanStokes coupling, and the axial temperature
        profile set by set_temperature_profile / solve_with_thermal_feedback (audit A-5: every
        clone used to run the COLD model, so metrics.*(amp) disagreed with amp.solve() by ~2.4 dB
        on a hot fiber). The _Tz tuple is immutable in practice (set_temperature_profile copies
        its inputs and nothing mutates it in place), so it is shared rather than re-copied.

        FiberAmplifier-PRIVATE. Anything that must also work on an ErYbAmplifier -- metrics.py
        and chain.py, i.e. everything the user hands an amplifier to -- uses the three PUBLIC
        protocol methods below (with_signals / with_pumps / without_ase) instead; reaching for
        _clone from metrics.py is what made every metric raise AttributeError on an ErYbAmplifier
        (audit A-3 follow-on). dynamics._amp_with_boundary keeps using _clone deliberately: the
        transient march is FiberAmplifier-only anyway (it reads _n_active / _mcc_matrix /
        concentration, none of which ErYbAmplifier has) and it needs pumps AND signals swapped in
        ONE construction."""
        new = FiberAmplifier(self.ion, self.fiber,
                             self.pumps if pumps is None else list(pumps),
                             self.signals if signals is None else list(signals),
                             self.ase if ase is _KEEP else ase,
                             upconversion_C_up=self.upconversion_C_up,
                             concentration=self.concentration, raman=self.raman)
        new._Tz = self._Tz
        return new

    # ---- PUBLIC amplifier re-seed protocol -------------------------------------------------
    # with_signals / with_pumps / without_ase are THE contract every amplifier class in this
    # package implements (FiberAmplifier and ErYbAmplifier both do) and the ONLY route
    # AmplifierChain and metrics.* use to rebuild a stage. Each returns a copy of the SAME class
    # carrying every opt-in.
    def with_signals(self, signals: List[Signal]) -> "FiberAmplifier":
        """Re-seed the amplifier with a new signal list, preserving its TYPE and every opt-in
        (see _clone). ErYbAmplifier implements the same method, so a chain can hold either class
        (audit A-3: the chain used to rebuild every stage as a FiberAmplifier from amp.ion, which
        hard-crashed on an ErYbAmplifier)."""
        return self._clone(signals=signals)

    def with_pumps(self, pumps: List[Pump]) -> "FiberAmplifier":
        """Re-seed the amplifier with a new pump list, preserving its TYPE and every opt-in
        (see _clone). Used by metrics.slope_efficiency to sweep the launched pump."""
        return self._clone(pumps=pumps)

    def without_ase(self) -> "FiberAmplifier":
        """A copy with the ASE band DROPPED (ase=None) and everything else -- including the axial
        temperature profile (audit A-5) -- carried through. Used by metrics.gain_spectrum, whose
        small-signal probe is ASE-independent and much faster without the band."""
        return self._clone(ase=None)

    # ---- PUBLIC channel plan --------------------------------------------------------------
    def channel_plan(self) -> ChannelPlan:
        """This amplifier's channel structure -- see ChannelPlan (audit 2026-08-04 F-3).

        The same content `_plan()` builds internally, in a documented public record. Exposed
        because an external consumer previously had to reimplement the ASE bin-edge and
        frequency-bin-width bookkeeping to reach it. `_plan` stays private as the solver hot path.
        """
        ch, bc, u, is_ase, kind = self._plan()
        m = float(self.ase.m_modes) if self.ase is not None else 2.0
        return ChannelPlan(lambda_m=ch.lambda_m.copy(), direction=u.copy(),
                           is_ase=is_ase.copy(), dnu_hz=ch.dnu_hz.copy(), kind=list(kind),
                           launched_W=bc.copy(), gamma=ch.gamma.copy(),
                           loss_per_m=ch.loss_per_m.copy(),
                           m_modes=np.where(is_ase, m, 0.0), channels=ch)

    # ---- channel plan --------------------------------------------------------------------
    def _plan(self) -> Tuple[ChannelSet, np.ndarray, np.ndarray, np.ndarray, List[str]]:
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
            # bin width in FREQUENCY (the spontaneous term is per unit optical frequency)
            nu_edges = C_LIGHT / edges
            dnu_bins = np.abs(nu_edges[:-1] - nu_edges[1:])
            for direction in (+1.0, -1.0):
                for c, dv in zip(centres, dnu_bins):
                    lam.append(float(c)); u.append(direction); is_ase.append(True)
                    dnu.append(float(dv)); kind.append("ase"); bc.append(0.0)
                    cladding.append(False)
        if self.raman is not None:                   # SRS Stokes channel (RamanStokes docstring)
            rs = self.raman
            lam_sig = self.signals[rs.signal_index].lambda_m
            nu_st = C_LIGHT / lam_sig - rs.shift_hz
            lam.append(C_LIGHT / nu_st); u.append(+1.0 if rs.direction == "fwd" else -1.0)
            is_ase.append(False); dnu.append(0.0); kind.append("stokes"); bc.append(0.0)
            cladding.append(False)
        lam = np.asarray(lam); u = np.asarray(u); is_ase = np.asarray(is_ase, bool)
        ch = ChannelSet.build(self.ion, self.fiber, lam, u, is_ase=is_ase, dnu_hz=np.asarray(dnu))
        # cladding pumps: replace the (core) overlap by A_dope/A_clad (audit S3-9)
        gamma = ch.gamma.copy()
        for k, cl in enumerate(cladding):
            if cl:
                from dynameta.optics.fiber_amp.waveguide import cladding_pump_overlap
                gamma[k] = cladding_pump_overlap(self.fiber)
        ch = ChannelSet(ch.lambda_m, ch.u, ch.is_ase, ch.dnu_hz, ch.sigma_a, ch.sigma_e,
                        gamma, ch.loss_per_m, ch.tau_s, ch.sigma_esa)
        if self.raman is not None:                   # the coupled-exchange bundle (RHS hot path)
            rs = self.raman
            lam_sig = self.signals[rs.signal_index].lambda_m
            nu_sig = C_LIGHT / lam_sig
            nu_st = nu_sig - rs.shift_hz
            g_r = (rs.g_r_m_w if rs.g_r_m_w is not None
                   else 1.0e-13 * (1.0e-6 / lam_sig))              # Stolen-Ippen 1/lambda scaling
            a_eff = (rs.a_eff_m2 if rs.a_eff_m2 is not None
                     else float(np.pi * mode_field_radius_m(self.fiber.core_radius_m,
                                                            self.fiber.na, lam_sig) ** 2))
            n_th = 1.0 / np.expm1(H_PLANCK * rs.shift_hz / (KB * rs.T_K))
            self._raman_map = {
                "i_sig": len(self.pumps) + rs.signal_index,
                "i_st": lam.size - 1,
                "k_r": g_r / a_eff,
                "q_seed_W": H_PLANCK * nu_st * rs.dnu_eff_hz * (n_th + 1.0),
                "ratio": nu_sig / nu_st,
            }
        else:
            self._raman_map = None
        return ch, np.asarray(bc), u, is_ase, kind

    # ---- distributed temperature profile (McCumber z-scaling) ----------------------------
    def set_temperature_profile(self, z_m, T_K, *, T_ref_K: float = 300.0):
        """Impose an axial temperature profile T(z) on the gain medium. Every
        sigma_e-proportional coefficient (local emission gain, stimulated-emission rate, ASE
        spontaneous source) is scaled per-z by the McCumber factor
            mcc_k(z) = exp[(eps - h nu_k) (1/(kB T(z)) - 1/(kB T_ref))],
        eps = `self.ion.eps_J`, i.e. a fitted `mccumber_eps_J` when the ion carries one and
        h c / zero_line_m otherwise (audit F-12; the code below reads the property, which is what
        this line used to state as an unconditional h c / zero_line) -- the z-resolved form of
        spectroscopy.at_temperature (whose
        GLOBAL scaling this reproduces exactly for a uniform profile; gated in tests).
        sigma_a and tau are held (their T-dependence is second order at these Delta-T; module
        docstring of spectroscopy). Cleared with clear_temperature_profile(). Used by
        thermal.solve_with_thermal_feedback for the self-consistent Q(z) -> T(z) loop."""
        z = np.asarray(z_m, float)
        T = np.asarray(T_K, float)
        if z.shape != T.shape or z.ndim != 1 or z.size < 2:
            raise ValueError("set_temperature_profile: z_m and T_K must be equal-length 1-D")
        self._Tz = (z.copy(), T.copy(), float(T_ref_K))

    def clear_temperature_profile(self):
        self._Tz = None

    def _mcc_matrix(self, ch: ChannelSet, z):
        """(K, M) McCumber sigma_e scale factors on the solver mesh z, or None."""
        if getattr(self, "_Tz", None) is None:
            return None
        zt, Tt, T_ref = self._Tz
        T = np.interp(z, zt, Tt)
        # Honour a fitted McCumber eps if the ion carries one (audit F-12); identical to the old
        # h c / zero_line_m when it does not.
        eps = self.ion.eps_J
        slope = eps - H_PLANCK * ch.nu_hz                       # (K,)
        from dynameta.constants import KB as _KB
        return np.exp(np.outer(slope, 1.0 / (_KB * T) - 1.0 / (_KB * T_ref)))

    # ---- pointwise physics used by the IVP passes ----------------------------------------
    # Per-channel constants (overlap/cross-section/frequency products) are hoisted into a
    # coefficient bundle ONCE per solve (audit S6-13: they were recomputed on each of the
    # ~25k RHS calls). Same algebra, tolerance-neutral regrouping.
    def _coeffs(self, ch: ChannelSet):
        A = self.fiber.a_dope_m2
        na = self._n_active
        m = self.ase.m_modes if self.ase else 2
        c = {
            "flux_a": ch.gamma * ch.sigma_a / (H_PLANCK * ch.nu_hz * A),   # R_a = sum(flux_a P)
            "flux_e": ch.gamma * ch.sigma_e / (H_PLANCK * ch.nu_hz * A),
            "g_e": ch.gamma * na * ch.sigma_e,
            "g_a": ch.gamma * na * ch.sigma_a,
            "g_esa": ch.gamma * na * ch.sigma_esa,
            "loss": ch.loss_per_m.copy(),
            "s_pref": np.where(ch.is_ase, ch.gamma * na * ch.sigma_e * m
                               * H_PLANCK * ch.nu_hz * ch.dnu_hz, 0.0),
            "m_modes": m,
        }
        if self.concentration is not None:
            c["loss"] = c["loss"] + ch.gamma * self._n_dark * ch.sigma_a   # unbleachable PIQ
        return c

    def _nbar2_c(self, c, P, mcc=None):
        """Metastable fraction from the local power vector P (K,) via the coefficient bundle.
        mcc (K,) scales every sigma_e-proportional coefficient (set_temperature_profile)."""
        R_a = float(np.dot(c["flux_a"], P))
        R_e = float(np.dot(c["flux_e"] if mcc is None else c["flux_e"] * mcc, P))
        tau = self._tau_s
        # The quadratic (upconversion) branch divides by C_up * n_active, so it needs BOTH to be
        # positive. n_active == 0 is an undoped fiber (audit F-10) -- there are no ions to
        # upconvert, so the linear form is not merely safe but exact.
        if self.upconversion_C_up <= 0.0 or self._n_active <= 0.0:
            return tau * R_a / (1.0 + tau * (R_a + R_e))
        A2 = self.upconversion_C_up * self._n_active     # upconversion among active excited ions
        B = 1.0 / tau + R_a + R_e
        return (-B + np.sqrt(B * B + 4.0 * A2 * R_a)) / (2.0 * A2)

    def _dP_full_c(self, c, u, P, mcc=None):
        """dP_k/dz [W/m] for every channel from the local power vector P (K,). mcc (K,) is the
        optional per-z McCumber sigma_e scale (set_temperature_profile)."""
        P = np.maximum(P, 0.0)
        n2 = self._nbar2_c(c, P, mcc)
        ge = c["g_e"] if mcc is None else c["g_e"] * mcc
        sp = c["s_pref"] if mcc is None else c["s_pref"] * mcc
        g = ge * n2 - c["g_a"] * (1.0 - n2) - c["g_esa"] * n2 - c["loss"]
        if self.concentration is not None:
            g = g - self.concentration.photodarkening_loss_per_m(n2)   # inversion-dependent gray
        dP = u * (g * P + sp * n2)
        if self._raman_map is not None:              # SRS exchange (RamanStokes docstring):
            rm = self._raman_map                     # Stokes gains along ITS direction, signal
            ex = rm["k_r"] * P[rm["i_sig"]] * (P[rm["i_st"]] + rm["q_seed_W"])
            dP[rm["i_st"]] += u[rm["i_st"]] * ex     # loses Manley-Rowe-weighted power
            dP[rm["i_sig"]] -= u[rm["i_sig"]] * rm["ratio"] * ex
        return dP

    # back-compat single-call forms (reference/diagnostic surface)
    def _nbar2(self, ch: ChannelSet, P):
        """Metastable fraction from the full local power vector P (K,) at one z (scalar)."""
        self._tau_s = ch.tau_s
        return self._nbar2_c(self._coeffs(ch), np.asarray(P, float))

    def _dP_full(self, ch: ChannelSet, P):
        """dP_k/dz [W/m] for every channel from the full local power vector P (K,)."""
        self._tau_s = ch.tau_s
        return self._dP_full_c(self._coeffs(ch), ch.u, np.asarray(P, float))

    def _nbar2_profile(self, ch: ChannelSet, P, mcc_mat=None):
        """nbar2 at each z given the full power profile P (K, M); mcc_mat (K, M) optional."""
        self._tau_s = ch.tau_s
        c = self._coeffs(ch)
        return np.array([self._nbar2_c(c, P[:, j],
                                       None if mcc_mat is None else mcc_mat[:, j])
                         for j in range(P.shape[1])])

    def solve(self, *, n_nodes: int = 201, max_iter: int = 200, tol: float = 1e-6,
              method: str = "LSODA", relax="auto") -> SteadyStateResult:
        """Steady-state relaxation solve. See the module docstring for the method.

        relax = "auto" (THE DEFAULT) walks the _RELAX_LADDER -- (1.0, 0.5, 0.25) -- and returns the
        first attempt that CONVERGES, or the last attempt if none does. That makes a converged
        answer the default outcome without the caller needing to know the knob exists, at no cost on
        the problems where the undamped iteration already works (it is tried first, and it is 7-12x
        cheaper when it succeeds). `meta['relax']` reports which value produced the returned result
        and `meta['relax_attempts']` the ladder entries tried.

        Pass a NUMBER to pin it and take exactly one attempt. relax = 1.0 is the plain Gauss-Seidel
        iteration: the BLEND is skipped entirely rather than multiplied by 1.0, so no arithmetic
        touches the arrays and each ITERATE is bit-for-bit the pre-2026-08 one. That is a statement
        about the BLEND ONLY -- it is NOT a claim that a relax = 1.0 solve reproduces the old
        RESULT, because the audit-F-13 fix changed the STOPPING TEST in the same release, so the
        same trajectory is now left at a different point. Measured on the Yb reference at 0.243 mW
        of input: the old endpoint test never converged (400 iterations, `converged` False, gain
        34.722740 dB) where the new one stops at 118 (34.722738 dB). Same path, different exit; the
        returned profile differs by whatever the iteration still had left to converge.

        WHAT UNDER-RELAXATION IS. The solve alternates: integrate the forward channels through a
        frozen backward profile, then the backward channels through the new forward profile, and
        repeat. Each pass therefore OVERSHOOTS, because it treats the other direction as fixed while
        changing its own. relax blends each new pass with the previous one,

            profile <- relax * (this pass) + (1 - relax) * (previous),

        so the iteration approaches the fixed point instead of lunging past it. The fixed point
        itself is unchanged -- only the path to it -- which is why every converged relax value
        returns the same answer (verified to 1e-3 dB at 0.3, 0.2, 0.15 and 0.1 on the counter-pumped
        case below). Standard damped-fixed-point practice; thermal.solve_with_thermal_feedback
        already exposes the identical parameter under the same name.

        WHEN IT IS NEEDED. When the counter-propagating power is comparable to the co-propagating
        power, the Gauss-Seidel oscillation stops decaying and the sweep can wander or fall onto a
        spurious, essentially UNPUMPED branch. Two MEASURED regimes on a 3 um-core / NA 0.12 /
        6e25 m^-3 Yb fiber at 1060 nm, both unrecoverable by further iteration (checked to 2000-3000):

          * LOW SIGNAL, strong pump (backward ASE comparable to the pump). 2 W at 976 nm co-pumped,
            0.05-0.1 mW of signal in: relax = 1 gives -12.6 dB with nbar2 = 0.036; relax = 0.5
            converges to +37.1 dB in 64 iterations.
          * A COUNTER-PROPAGATING PUMP, at any signal level. 2 W at 976 nm counter-pumped, 20 mW
            of signal in, L = 6 m: relax = 1 gives -30.8 dB (endpoint residual 6.5e6) and
            relax = 0.5 still wanders; relax <= 0.3 converges to +17.9 dB, identically at 0.3,
            0.2, 0.15 and 0.1 -- i.e. that IS the fixed point. Bidirectional pumping is milder
            (relax = 1 lands ~1 dB low and non-converged; relax <= 0.5 converges). Co-pumping the
            same fiber converges at relax = 1 in 4 iterations, so this is specifically about the
            counter-propagating coupling, not about the amplifier being hard.

        PRACTICAL GUIDANCE: leave it at "auto". Pin a number only to reproduce an old result
        (relax = 1.0), to save the failed first rung on a fiber you already know is counter-pumped
        (relax = 0.3), or to go below the ladder's floor.

        ALWAYS CHECK THE RESULT. "auto" makes a converged answer likely, not certain --
        `meta['converged']` is still the stopping flag;
        `meta['endpoint_residual']` and `meta['profile_residual']` are the actual final residuals,
        so a caller can see HOW converged a solve is rather than only whether it tripped a
        threshold. `meta['min_power_W']` exposes the most negative power in the returned array: a
        channel attenuated far below the integrator's absolute tolerance (a fully absorbed pump)
        can come out as a small negative number, which is physically zero but will surprise
        anything that takes a log or a square root of it.
        """
        if isinstance(relax, str):
            if relax != "auto":
                raise ValueError("solve: relax must be a number in (0, 1] or \"auto\"; got %r"
                                 % (relax,))
            attempts = []
            res = None
            for rx in _RELAX_LADDER:
                attempts.append(rx)
                res = self._solve_once(n_nodes=n_nodes, max_iter=max_iter, tol=tol, method=method,
                                      relax=rx)
                if res.meta.get("converged"):
                    break
            res.meta["relax_attempts"] = tuple(attempts)
            return res
        res = self._solve_once(n_nodes=n_nodes, max_iter=max_iter, tol=tol, method=method,
                               relax=relax)
        res.meta["relax_attempts"] = (float(relax),)
        return res

    def _solve_once(self, *, n_nodes: int = 201, max_iter: int = 200, tol: float = 1e-6,
                    method: str = "LSODA", relax: float = 1.0) -> SteadyStateResult:
        """ONE relaxation solve at a fixed `relax`. solve() is the ladder wrapper over this."""
        from scipy.integrate import solve_ivp
        if not (0.0 < float(relax) <= 1.0):
            raise ValueError("solve: relax must be in (0, 1]; got %r" % (relax,))
        relax = float(relax)
        ch, bc, u, is_ase, kind = self._plan()
        self._tau_s = ch.tau_s
        c = self._coeffs(ch)
        L = self.fiber.length_m
        z = np.linspace(0.0, L, n_nodes)
        fwd = np.where(u > 0)[0]
        bwd = np.where(u < 0)[0]
        # backward-channel profiles (K_bwd, M), initialised to their z=L seed everywhere
        P_bwd = np.repeat(bc[bwd][:, None], n_nodes, axis=1) if bwd.size else np.zeros((0, n_nodes))
        P_fwd = np.repeat(bc[fwd][:, None], n_nodes, axis=1)

        def _assemble(Pf, Pb):
            P = np.empty(ch.lambda_m.size)
            P[fwd] = Pf
            if bwd.size:
                P[bwd] = Pb
            return P

        def _make_interp(Y):                         # audit X-3: ONE implementation, module level
            return _frozen_profile_interp(Y, z, L, n_nodes)

        mcc_mat = self._mcc_matrix(ch, z)            # (K, M) sigma_e T-scaling or None
        mcc_of = _make_interp(mcc_mat) if mcc_mat is not None else None

        last_out = None
        last_prof = None
        converged = False
        end_resid = prof_resid = float("nan")     # final residuals, reported on meta (audit F-14)
        for it in range(max_iter):
            bwd_of = _make_interp(P_bwd) if bwd.size else None

            def rhs_f(zz, Pf):
                Pb = bwd_of(zz) if bwd.size else np.zeros(0)
                m = mcc_of(zz) if mcc_of is not None else None
                return self._dP_full_c(c, u, _assemble(Pf, Pb), m)[fwd]

            sf = solve_ivp(rhs_f, (0.0, L), bc[fwd], t_eval=z, method=method,
                           rtol=1e-7, atol=1e-15)
            # Under-relaxation (audit F-14). The relax == 1.0 branch is taken verbatim so the
            # default path performs NO arithmetic on the arrays and stays byte-identical -- a
            # `1.0 * new + 0.0 * old` blend would not, because 0.0 * inf is NaN and -0.0 + 0.0
            # flips a sign.
            P_fwd = sf.y if relax == 1.0 else relax * sf.y + (1.0 - relax) * P_fwd

            if bwd.size:
                fwd_of = _make_interp(P_fwd)

                def rhs_b(zz, Pb):
                    m = mcc_of(zz) if mcc_of is not None else None
                    return self._dP_full_c(c, u, _assemble(fwd_of(zz), Pb), m)[bwd]

                sb = solve_ivp(rhs_b, (L, 0.0), bc[bwd], t_eval=z[::-1], method=method,
                               rtol=1e-7, atol=1e-15)
                new_bwd = sb.y[:, ::-1]
                P_bwd = new_bwd if relax == 1.0 else relax * new_bwd + (1.0 - relax) * P_bwd

            # convergence: endpoint powers AND the full interior profile, each channel measured
            # against its own peak power (audit S3-34: the old endpoint-only test with a 1e-15 W
            # floor could declare victory while the interior was still moving, and amplified
            # noise on strongly-absorbed channels).
            out = np.concatenate([P_fwd[:, -1], (P_bwd[:, 0] if bwd.size else [])])
            prof = np.concatenate([P_fwd, P_bwd], axis=0) if bwd.size else P_fwd.copy()
            if last_out is not None:
                # Residuals kept for meta so a caller can see HOW converged the solve is, not
                # merely whether it tripped tol (audit F-14: a run that lands on a spurious branch
                # shows a large residual here, which is the honest diagnostic).
                end_resid, prof_resid = _relaxation_residuals(out, last_out, prof, last_prof)
                end_ok = end_resid < tol
                prof_ok = prof_resid < tol
                if end_ok and prof_ok:
                    converged = True
                    last_out = out
                    break
            last_out = out
            last_prof = prof

        P = np.empty((ch.lambda_m.size, n_nodes))
        P[fwd] = P_fwd
        if bwd.size:
            P[bwd] = P_bwd
        n2 = self._nbar2_profile(ch, P, mcc_mat)
        sig_idx = [i for i, k in enumerate(kind) if k == "signal"]
        gains_dB = np.array([10.0 * np.log10(P[i, -1] / bc[i]) for i in sig_idx])
        return SteadyStateResult(z, P, ch.lambda_m, u, is_ase, kind, n2, gains_dB,
                                 meta={"converged": converged, "iterations": it + 1,
                                       # audit F-14: the ACTUAL final residuals, so a caller can
                                       # judge how converged a solve is instead of trusting a
                                       # boolean; and the most negative returned power, which is a
                                       # fully-absorbed channel sitting in integrator noise
                                       # (physically zero, but it will surprise a log or a sqrt).
                                       "endpoint_residual": end_resid,
                                       "profile_residual": prof_resid,
                                       "relax": float(relax),
                                       "min_power_W": float(np.min(P)),
                                       "dnu_hz": ch.dnu_hz.copy(),
                                       "sigma_a": ch.sigma_a.copy(),
                                       "sigma_e": ch.sigma_e.copy(),
                                       "sigma_esa": ch.sigma_esa.copy(),
                                       "gamma": ch.gamma.copy(),
                                       # audit A-6: the sigma_* above are the T_ref ChannelSet
                                       # values. Under a temperature profile the solve actually
                                       # used sigma_e * mcc(z), so noise.local_inversion_factor
                                       # -- which advertises "the cross-sections the solve
                                       # cached" -- mixed a T_ref sigma_e with the HOT nbar2_z.
                                       # Cache the (K, M) McCumber scaling so it can undo that;
                                       # None on an isothermal solve (byte-identical path).
                                       "mcc": None if mcc_mat is None else mcc_mat.copy(),
                                       "m_modes": c["m_modes"]})
