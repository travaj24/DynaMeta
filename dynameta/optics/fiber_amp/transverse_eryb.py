"""Er:Yb co-doped fiber amplifier with the inversion RESOLVED ACROSS THE MODE -- the co-doped
counterpart of transverse.ResolvedFiberAmplifier and the pre-reduction form of eryb.ErYbAmplifier.

WHAT IS NEW HERE AND WHY IT IS NOT A DETAIL. `eryb.py` solves ONE pair (or triple) of coupled
populations per z from AREA-AVERAGED intensities <I_k> = Gamma_k P_k / A_dope. That closure is
harmless for a single ion whose pump and signal share the same mode, and it is NOT harmless for a
sensitized co-dope, because the Yb -> Er energy transfer is a LOCAL, BIMOLECULAR event:

    R_tr(r) = k_tr n_Yb2(r) n_Er1(r)                                             (E1.1)

an excited Yb donor must meet a GROUND Er acceptor AT THE SAME POINT. Mean-field replaces that
product of two radial profiles by the product of their two averages, and the two differ by the
covariance

    <n_Yb2 n_Er1> - <n_Yb2><n_Er1>                                               (E1.2)

which is exactly what this module computes instead of discarding. The covariance is not small in a
real EYDFA: a cladding pump is FLAT across the core, so the Yb it excites is (nearly) flat, while
the C-band signal is a tightly confined LP01 that burns the erbium inversion hardest on axis. The
Yb-absorbed pump profile and the Er inversion profile therefore peak in DIFFERENT places, and the
transfer integral sees the mismatch. Whether that matters at a given operating point is a MEASURED
question, not an assumed one -- section 5 of docs/audit/2026-09-15-eryb-transverse.md reports both
signs on the study's two reference points.

GOVERNING EQUATIONS (numbering continues the transverse dossier's F1.x with an E for co-doped;
the population algebra is eryb.py's, evaluated per node rather than per z).

  (E1.0)  every channel carries a NORMALIZED transverse intensity profile, INT i_k dA = 1, and
          I_k(r,phi,z) = P_k(z) i_k(r,phi). Core-guided channels take the exact LP field above
          V = 2.405 and the Marcuse Gaussian below it; a cladding pump is FLAT over the inner
          cladding. Shared with transverse.py -- literally the same functions.

  (E1.3)  per-node rates, one pair per ION, from the SAME local intensities:

              R_{a/e}_Er(r) = SUM_k sigma_{a/e}_Er,k P_k i_k(r) / (h nu_k)
              R_{a/e}_Yb(r) = SUM_k sigma_{a/e}_Yb,k P_k i_k(r) / (h nu_k)

          eryb.py's per-ion rates are these with i_k -> Gamma_k/A_dope.

  (E1.4)  the z-local coupled balances, NODE BY NODE, with NODE-LOCAL densities n_Er(r), n_Yb(r):

     Er:   R_a_Er (1-f2) - R_e_Er f2 - f2/tau_Er + k_tr b2c f n_Yb(r) (1-f2) phi
           - C_up n_Er(r) f2^2                                                   = 0
     Ybc:  R_a_Yb (1-b2c) - R_e_Yb b2c - b2c/tau_Yb - k_tr b2c n_Er(r) (1-f2) phi
           - K2 b2c n_Er(r) f2 - W_mig (1-f) (b2c - b2nc)                        = 0
     Ybnc: R_a_Yb (1-b2nc) - R_e_Yb b2nc - b2nc/tau_Yb + W_mig f (b2c - b2nc)    = 0

          THE DENSITIES ARE INSIDE THE BALANCE, not outside it. That is the whole content of
          (E1.1): the transfer-in coefficient is k_tr f n_Yb(r) and the transfer-out coefficient
          is k_tr n_Er(r), both evaluated at the node, so a ring where the Yb is excited but the
          Er is already inverted transfers nothing, and a ring where the Er is hungry but the Yb
          is dark transfers nothing either. Mean-field cannot express either statement.

  (E1.5)  propagation -- the same (F1.2) with BOTH ions' populations under the integral:

     dP_k/dz = u_k P_k { sigma_e_Er,k Je_Er,k - sigma_a_Er,k Ja_Er,k
                       + sigma_e_Yb,k Je_Yb,k - sigma_a_Yb,k Ja_Yb,k
                       - sigma_a_Er,k Jdark,k - l_k }
             + u_k m h nu_k dnu_k [ sigma_e_Er,k Je_Er,k + sigma_e_Yb,k Je_Yb,k ]   (ASE only)

          Je_Er,k = INT n_Er(r) f2(r) i_k dA,  Ja_Er,k = INT n_Er(r)(1-f2(r)) i_k dA, and the
          Yb pair the same with the POPULATION-WEIGHTED bbar(r) = f b2c(r) + (1-f) b2nc(r) (the
          field sees n6 = n6c + n6nc, Dong 2020). Jdark is the unbleachable pair-quenched erbium.
          With every population constant over a top-hat dopant these collapse to Gamma_k n f2 and
          (E1.5) IS eryb.py's ODE, term for term.

SEPARATE ER AND YB RADIAL PROFILES (`er_profile` / `yb_profile`, default BOTH top-hat to
fiber.b_dope_m, i.e. the scalar model's geometry). Real Er:Yb preforms do not co-locate the two
ions perfectly -- Yb is usually the more abundant and more widely distributed species -- and a
confined-Er / wide-Yb fiber is a standard way to suppress 1-um parasitic gain without losing pump
absorption. Because the transfer is local, moving the Yb out of the Er is NOT free: the Yb sitting
beyond the Er radius absorbs pump, stores it, and can only lose it to fluorescence, 1-um ASE or
K2. This module is the only place in the package where that trade can be computed at all.

CLADDING PUMP. Flat over the inner cladding, exactly as the single-ion resolved solver -- which
is the mean-field bracket for that channel, since a flat intensity has no hole to burn. So a
cladding-pumped resolved solve differs from its mean-field twin ONLY through the core-guided
channels (the signal, the two ASE bands, and a core pump if present) -- and through the transfer
covariance (E1.2) that those channels' hole burning creates.

NUMERICS. The ODE state is still P(z) per channel -- (E1.4) is algebraic at every node -- so the
solve is the SAME alternating forward/backward relaxation eryb.solve and transverse.solve use,
with the same under-relaxation knob and the same `_relaxation_residuals` convergence test. The
node balance is a VECTORIZED safeguarded Newton/bisection on the same scalar residual H(f2) that
`ErYbAmplifier._solve_fbb` brackets (b2c and b2nc are closed forms in f2 at every node; the 2x2
Yb system's determinant collapses analytically, see `_pools`), stepped in lockstep across nodes.

SCOPE / REFUSALS (v1; each raises ValueError with the physics reason -- see `_refuse`):
  * FiberSpec.overlap_override WITHOUT uniform_illumination=True -- an override REPLACES the
    overlap this solver computes. With uniform_illumination=True it is honoured, because the
    solver is then deliberately NOT resolving anything and the override is the mean-field Gamma.
  * set_temperature_profile -- a z-only McCumber scaling is not meaningful once the inversion is
    resolved; it would have to become T(r, z).
  * rate_temperature (RateTemperatureLaw) and yb_stark_thermal (YbStarkThermal) -- eryb.py's two
    2026-09-15 temperature opt-ins, which act ONLY through that same axial T(z) profile, so they
    are refused for the same reason AND because carrying them with no profile set would be inert.
    An all-zero RateTemperatureLaw is the exact identity and is accepted.
  * ConcentrationModel.pd_loss_per_m != 0 -- eryb.py applies the photodarkening equilibrium gray
    loss as a CHANNEL-UNIFORM background (no overlap factor); resolved, it is a per-node property
    of the glass and picks up the confinement factor. The two are different models and v1 refuses
    rather than silently answering the other one. C_up and pair-induced quenching ARE supported:
    both are strictly local and both reduce EXACTLY to eryb.py in the uniform limit.
  * an ion with NONZERO sigma_esa at any channel wavelength -- eryb.py drops the ESA term
    silently; this class refuses rather than inherit that.
Everything else ErYbAmplifier accepts is accepted: one or two Yb pools, K2, migration, k_back,
C_up, pair quenching, both ASE bands, core or cladding pumps, forward and backward.

RAM. The ring-resolved state is (z-node x ring) per population plus (channel x z-node) powers and
(channel x ring) profiles; `state_bytes` reports it and `solve` REFUSES above the package's 2 GiB
bar (dynamics._STORE_PROFILES_MAX_BYTES -- the same constant, not a second one).

References: eryb.py's (the Er:Yb rate model, the two-population split, K2, migration);
transverse.py's (Smith & Smith Eq. 1 for the local balance, Giles & Desurvire for the m h nu dnu
seed); docs/fiber_transverse_grounding_2026_07_29.md sec.1 for the quadrature. Audit trail:
docs/audit/2026-09-15-eryb-transverse.md; docs/fiber_amp_model_spec.md sec.17.
Pure numpy/scipy; SI units; ASCII only. Power-only model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.core.numerics import trapz         # audit X-1: floor-safe (np.trapezoid needs >=2.0)
# The 2 GiB ring-state bar is the package's EXISTING one, imported rather than re-declared.
from dynameta.optics.fiber_amp.dynamics import _STORE_PROFILES_MAX_BYTES
from dynameta.optics.fiber_amp.eryb import ErYbAmplifier
from dynameta.optics.fiber_amp.lma import LPMode
from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
from dynameta.optics.fiber_amp.steady_state import (AseBand, Pump, Signal, SteadyStateResult,
                                                    _KEEP, _RELAX_LADDER,
                                                    _frozen_profile_interp,
                                                    _relaxation_residuals)
# The transverse geometry kernel -- ONE home for the grid, the profiles and the mode checks
# (transverse.py), used verbatim so a co-doped solve and a single-ion solve integrate on
# identical quadrature and can be compared node for node.
from dynameta.optics.fiber_amp.transverse import (RadialGrid, build_normalized_profiles,
                                                  check_signal_modes, default_r_max_m,
                                                  flat_intensity_profile, fundamental_psi2,
                                                  quadrature_breakpoints)
from dynameta.optics.fiber_amp.waveguide import FiberSpec

__all__ = ["RadialDopant", "ResolvedErYbAmplifier", "eryb_mean_field_equivalent"]

# The wavelength the 1-um parasitic-gain diagnostic is evaluated at -- eryb.py's own choice, kept
# so the resolved and the mean-field numbers are the same quantity.
LAMBDA_PARASITIC_M = 1.030e-6


# ============================ dopant radial profiles ====================================

@dataclass(frozen=True)
class RadialDopant:
    """One ion's radial DENSITY profile: n(r) = N * shape(r) for r <= radius_m, 0 beyond.

    `N` is the amplifier's nominal density for that ion (fiber.n_t_m3 for Er, n_yb_m3 for Yb), so
    `shape` is dimensionless and `shape = None` (the default, a TOP HAT) reproduces the scalar
    model's geometry exactly. The convention is deliberately NOT number-conserving: a wider dopant
    region at the same density holds MORE ions, which is what `FiberSpec.n_t_m3` +
    `dopant_radius_m` already means everywhere else in this package. `radius_m` is load-bearing
    beyond the shape -- it is a quadrature panel edge, so the density step is resolved spectrally
    instead of being smeared across a panel."""
    radius_m: float
    shape: Optional[Callable] = None

    def __post_init__(self):
        if not (float(self.radius_m) > 0.0):
            raise ValueError("RadialDopant: radius_m must be > 0 (got %r)" % (self.radius_m,))
        if self.shape is not None and not callable(self.shape):
            raise ValueError("RadialDopant: shape must be None (top hat) or a callable "
                             "shape(r_m) -> dimensionless density factor (got %r)"
                             % (type(self.shape),))

    def density(self, n_nominal_m3: float, r_m) -> np.ndarray:
        """(N,) ion density [m^-3] at the grid radii."""
        r = np.asarray(r_m, float)
        inside = r <= float(self.radius_m)
        if self.shape is None:
            return np.where(inside, float(n_nominal_m3), 0.0)
        s = np.asarray(self.shape(r), float)
        if s.shape != r.shape:
            raise ValueError("RadialDopant.shape returned shape %r for %r radii -- it must be "
                             "elementwise" % (s.shape, r.shape))
        if not np.all(np.isfinite(s)) or float(np.min(s)) < 0.0:
            raise ValueError("RadialDopant.shape must return finite, non-negative factors "
                             "(min %r)" % (float(np.min(s)),))
        return np.where(inside, float(n_nominal_m3) * s, 0.0)


DopantSpec = Union[None, float, RadialDopant]


def _as_dopant(spec: DopantSpec, default_radius_m: float, name: str) -> RadialDopant:
    """None -> a top hat at the fiber's own dopant radius; a number -> a top hat at that radius;
    a RadialDopant -> itself. ONE place the three spellings are resolved."""
    if spec is None:
        return RadialDopant(float(default_radius_m))
    if isinstance(spec, RadialDopant):
        return spec
    try:
        rad = float(spec)
    except (TypeError, ValueError):
        raise ValueError("%s must be None (top hat to the fiber dopant radius), a positive "
                         "radius in metres, or a RadialDopant (got %r)" % (name, type(spec))
                         ) from None
    return RadialDopant(rad)


# ============================ the per-node population kernel ============================

def _bracketed_newton_nodes(hdh, n_nodes: int, max_iter: int = 80, tol: float = 1e-13):
    """Safeguarded Newton/bisection for the residual H(f2) at EVERY node at once.

    The scalar original is `ErYbAmplifier._bracketed_newton`: same bracket [0, 1], same guarantee
    (H(0) >= 0 because absorption and transfer drive the erbium up, H(1) < 0 because a fully
    inverted erbium has only decay left), same update rule (Newton where dH < 0, bisection
    otherwise, bisection again when the Newton step leaves the bracket), same convergence test.
    Every node is stepped in LOCKSTEP and the loop exits when all of them have converged.

    ONE DELIBERATE DEPARTURE, and it is a fix rather than a shortcut. The scalar clamps an
    out-of-bracket Newton step to the bisection midpoint BEFORE testing convergence. Once Newton
    has landed on the root, the next pass sets the bracket end TO that root (lo = f or hi = f),
    so the following step -- of size ~1e-17, i.e. below an ulp of f -- is no longer strictly
    inside (lo, hi), is discarded, and the routine then BISECTS ~43 more times to re-derive an
    answer it already had. Harmless for one scalar root; ruinous here, where one straggling node
    holds every other node in the vector (MEASURED on the gate-2 fixture: 44.8 residual
    evaluations per call instead of 3, a 14x wall-clock cost on the whole solve). So a step
    already smaller than `tol` is ACCEPTED instead of clamped. The fixed point is identical and
    the residual is if anything smaller -- Newton leaves |H| at round-off, whereas the bisection
    path only guarantees the 1e-13 bracket -- and the two routines are gated against each other
    to 1e-13 in f2 over a sweep of operating points in tests/test_fiber_transverse_eryb.py.

    `hdh(f)` returns (H, dH/df) as (N,) arrays. Nodes with no drive at all (H(0) <= 0: no pump, no
    signal, no transfer) are pinned at f2 = 0, as in the scalar."""
    n = int(n_nodes)
    H0, _ = hdh(np.zeros(n))
    drive = H0 > 0.0
    lo = np.zeros(n)
    hi = np.ones(n)
    f = np.where(drive, 0.5, 0.0)
    for _ in range(int(max_iter)):
        H, dH = hdh(f)
        pos = H > 0.0
        lo = np.where(pos, f, lo)
        hi = np.where(pos, hi, f)
        mid = 0.5 * (lo + hi)
        # Newton only where the slope is usable; the guarded divisor keeps the discarded branch
        # away from 0 so the repo's filterwarnings=["error"] never sees a divide-by-zero.
        f_new = np.where(dH < 0.0, f - H / np.where(dH < 0.0, dH, -1.0), mid)
        moved = np.abs(f_new - f)
        # clamp only a step that is BOTH out of bracket AND larger than the tolerance (above)
        f_new = np.where(((lo < f_new) & (f_new < hi)) | (moved <= tol), f_new, mid)
        moved = np.abs(f_new - f)
        f = np.where(drive, f_new, 0.0)
        if np.all((moved <= tol) | ((hi - lo) <= tol) | ~drive):
            break
    return np.clip(f, 0.0, 1.0)


def _solve_populations_nodes(Ra_Er, Re_Er, Ra_Yb, Re_Yb, p):
    """(f2, b2c, b2nc) at every node -- the NODE-LOCAL generalization of
    `ErYbAmplifier._solve_fbb`, with the two ion densities INSIDE the coefficients (E1.4).

    THE REDUCTION is unchanged from the scalar: at fixed f2 the two Yb balances are LINEAR in
    (b2c, b2nc), the 2x2 determinant collapses because the migration cross terms cancel
    identically,

        D   = R_a_Yb + R_e_Yb + 1/tau_Yb
        Dc  = D + k_tr n_Er(r) (1 - f2) phi + K2 n_Er(r) f2
        det = Dc D + W (f Dc + (1 - f) D)  > 0 always
        b2c = R_a_Yb (D + W)/det,   b2nc = R_a_Yb (Dc + W)/det

    and substituting b2c into the Er residual leaves ONE bracketed scalar equation per node. What
    is new is that k_tr n_Er, K2 n_Er, C_up n_Er and k_tr f n_Yb are now (N,) arrays: THE
    TRANSFER IS LOCAL. The one-pool configuration (f = 1, K2 = 0, W = 0) is not special-cased --
    it is the same expression with det = Dc D, which returns b2c = R_a_Yb/Dc to one ulp of the
    scalar `_solve_fb`'s own arithmetic.

    `p` is the parameter bundle built by `_plan` (per-node densities, rate constants, lifetimes).
    """
    n_er, n_yb = p["n_er"], p["n_yb"]
    fc, k_tr, k_tr2, w_mig = p["fc"], p["k_tr"], p["k_tr2"], p["w_mig"]
    k_back, a32 = p["k_back"], p["a32"]
    inv_tE, inv_tY = p["inv_tau_er"], p["inv_tau_yb"]
    cupn = p["c_up"] * n_er
    c_in = k_tr * fc * n_yb                       # transfer-IN coefficient  (x b2c)
    s_out = k_tr * n_er                           # coupled-Yb drain coeff   (x (1-f2))
    k2n = k_tr2 * n_er
    D = Ra_Yb + Re_Yb + inv_tY
    Da = Ra_Er + Re_Er + inv_tE

    def _pools(f):
        # phi = 1 seed, then the same three-pass inner fixed point ErYbAmplifier._solve_fbb runs,
        # in the SAME order, so every value returned belongs to the same phi.
        phi = np.ones_like(D)
        Dc = D + s_out * (1.0 - f) + k2n * f
        det = Dc * D + w_mig * (fc * Dc + (1.0 - fc) * D)
        bc = Ra_Yb * (D + w_mig) / det
        if k_back > 0.0:
            for _ in range(3):
                phi = a32 / (a32 + k_back * fc * (1.0 - bc) * n_yb)
                Dc = D + s_out * (1.0 - f) * phi + k2n * f
                det = Dc * D + w_mig * (fc * Dc + (1.0 - fc) * D)
                bc = Ra_Yb * (D + w_mig) / det
        return bc, Ra_Yb * (Dc + w_mig) / det, phi, det

    def _HdH(f):
        bc, _bn, phi, det = _pools(f)
        dDc = k2n - s_out * phi
        dbdf = -bc * (D + w_mig * fc) * dDc / det      # > 0: less drain as f rises
        G = c_in * phi * bc
        H = Ra_Er * (1.0 - f) - Re_Er * f - f * inv_tE + G * (1.0 - f) - cupn * f * f
        dH = -Da + c_in * phi * (dbdf * (1.0 - f) - bc) - 2.0 * cupn * f
        return H, dH

    f2 = _bracketed_newton_nodes(_HdH, D.size)
    bc, bn, _phi, _det = _pools(f2)
    return f2, np.clip(bc, 0.0, 1.0), np.clip(bn, 0.0, 1.0)


# ============================ the resolved co-doped amplifier ===========================

class ResolvedErYbAmplifier:
    """An Er:Yb co-doped fiber amplifier solved with BOTH ions' populations resolved per radial
    ring. The constructor mirrors `eryb.ErYbAmplifier` exactly -- same positional arguments, same
    keywords, same defaults -- so a scalar study can be re-solved with the transverse physics by
    swapping the class, and `eryb_mean_field_equivalent()` maps back for the delta.

    Extra keyword arguments (all default to the scalar model's geometry):
      er_profile, yb_profile  -- each None (top hat to fiber.b_dope_m), a positive radius in
                                 metres (top hat to it), or a RadialDopant for a graded profile.
                                 The two ions may differ; see the module docstring for why that
                                 is a real design knob and not decoration.
      signal_modes            -- per-signal transverse profile, one entry per Signal: None (the
                                 fundamental), an LPMode (multi-mode competition; all signals
                                 share the ONE resolved inversion), or "flat" (that channel's
                                 mean-field profile). Same spelling and same refusals as
                                 ResolvedFiberAmplifier.
      uniform_illumination    -- force EVERY core-guided channel to its mean-field profile
                                 Gamma_k/A_dope. The solver then computes the scalar model by a
                                 different route, which is what makes the reduction a runnable
                                 gate rather than an assertion. A cladding pump is already flat
                                 and is untouched. This is also the ONLY mode in which
                                 FiberSpec.overlap_override is accepted.
      n_quad, n_azimuthal, r_max_m -- the quadrature, exactly as ResolvedFiberAmplifier.

    solve() returns a `steady_state.SteadyStateResult` -- the SAME type ErYbAmplifier returns, so
    every noise.* / metrics.* / efficiency.* consumer reads it unchanged. `nbar2_z` is the
    DOPANT-POPULATION-WEIGHTED Er metastable fraction INT n_Er f2 dA / INT n_Er dA (the honest
    scalar reduction of a quantity that is no longer a scalar; identical to the plain dopant-area
    average for a top-hat profile), `meta['beta_yb_z']` the same reduction of the
    population-weighted Yb inversion, and the RING-RESOLVED populations live on `meta['f2_rz']`,
    `meta['b2c_rz']`, `meta['b2nc_rz']`, `meta['bbar_rz']` with the grid on `meta['grid']`."""

    def __init__(self, er_ion: RareEarthIon, yb_ion: RareEarthIon, fiber: FiberSpec,
                 pumps: List[Pump], signals: List[Signal], ase: Optional[AseBand] = None, *,
                 n_yb_m3: float, k_tr_m3_s: float = 2.0e-22, k_back_m3_s: float = 0.0,
                 a32_per_s: float = 5.0e5, yb_ase: Optional[AseBand] = None,
                 upconversion_C_up: float = 0.0, yb_coupled_fraction: float = 1.0,
                 k_tr2_m3_s: float = 0.0, yb_migration_rate_per_s: float = 0.0,
                 concentration=None, rate_temperature=None, yb_stark_thermal=None,
                 er_profile: DopantSpec = None, yb_profile: DopantSpec = None,
                 signal_modes: Optional[Sequence] = None, uniform_illumination: bool = False,
                 n_quad: int = 24, n_azimuthal: int = 32, r_max_m: Optional[float] = None):
        # THE mean-field twin, built FIRST: it owns every constructor validation (n_yb > 0,
        # a32 > 0, the coupled fraction in [0, 1], the C_up-override warning, the active/dark
        # erbium split) so those rules keep ONE home, and it IS the oracle the
        # uniform-illumination reduction gate compares against.
        self._mf = ErYbAmplifier(er_ion, yb_ion, fiber, pumps, signals, ase,
                                 n_yb_m3=n_yb_m3, k_tr_m3_s=k_tr_m3_s,
                                 k_back_m3_s=k_back_m3_s, a32_per_s=a32_per_s, yb_ase=yb_ase,
                                 upconversion_C_up=upconversion_C_up,
                                 yb_coupled_fraction=yb_coupled_fraction,
                                 k_tr2_m3_s=k_tr2_m3_s,
                                 yb_migration_rate_per_s=yb_migration_rate_per_s,
                                 concentration=concentration)
        # The two TEMPERATURE-DRIVEN opt-ins of eryb.py (2026-09-15) are accepted so a port fails
        # with a physics message instead of a TypeError, and are then refused by name in
        # `_refuse` -- they are functions of an axial T(z) this class cannot take.
        self.rate_temperature = (None if rate_temperature is not None
                                 and getattr(rate_temperature, "is_identity", False)
                                 else rate_temperature)
        self.yb_stark_thermal = yb_stark_thermal
        self.er_ion, self.yb_ion, self.fiber = er_ion, yb_ion, fiber
        self.pumps, self.signals = list(pumps), list(signals)
        self.ase, self.yb_ase = ase, yb_ase
        self.concentration = self._mf.concentration
        self.upconversion_C_up = self._mf.upconversion_C_up
        self.uniform_illumination = bool(uniform_illumination)
        self.n_quad, self.n_azimuthal = int(n_quad), int(n_azimuthal)
        self.r_max_m = None if r_max_m is None else float(r_max_m)
        self.er_profile = _as_dopant(er_profile, fiber.b_dope_m, "er_profile")
        self.yb_profile = _as_dopant(yb_profile, fiber.b_dope_m, "yb_profile")
        if signal_modes is None:
            self.signal_modes = [None] * len(self.signals)
        else:
            self.signal_modes = list(signal_modes)
            if len(self.signal_modes) != len(self.signals):
                raise ValueError(
                    "ResolvedErYbAmplifier: signal_modes has {} entries for {} signals -- it is "
                    "a PARALLEL list (one profile per Signal, None = the fundamental)".format(
                        len(self.signal_modes), len(self.signals)))
        check_signal_modes(self.fiber, self.signals, self.signal_modes,
                           where="ResolvedErYbAmplifier")
        self._refuse()
        self._cache = None

    # ---- scope refusals ------------------------------------------------------------------
    def _refuse(self):
        """Every v1 scope boundary, refused at CONSTRUCTION with its physics reason. Silently
        ignoring any of these would report a number the model did not compute."""
        if self.fiber.overlap_override is not None and not self.uniform_illumination:
            raise ValueError(
                "ResolvedErYbAmplifier: FiberSpec.overlap_override is not supported. This solver "
                "COMPUTES the overlap by quadrature of the mode profile; an override would "
                "replace the very quantity being resolved (it is the right tool for the "
                "mean-field ErYbAmplifier, e.g. Giles-calibrated fibers). It IS accepted with "
                "uniform_illumination=True, where the solver deliberately reproduces the "
                "mean-field model and the override is exactly the Gamma that model uses.")
        for nm, opt in (("rate_temperature", self.rate_temperature),
                        ("yb_stark_thermal", self.yb_stark_thermal)):
            if opt is None:
                continue
            raise ValueError(
                "ResolvedErYbAmplifier: {} is not supported in v1. Both of eryb.py's 2026-09-15 "
                "temperature opt-ins act ONLY through an axial T(z) profile -- "
                "RateTemperatureLaw Arrhenius-scales k_tr / K2 / W_mig at T(z), and "
                "YbStarkThermal depopulates the Yb lower Stark manifold at T(z) -- and this "
                "class refuses set_temperature_profile for the reason that makes them wrong "
                "here: once the inversion is resolved across the mode the meaningful field is "
                "T(r, z), not T(z). Carrying them silently would also be inert (with no profile "
                "set they scale nothing), which is the worse failure. Use eryb.ErYbAmplifier for "
                "the mean-field temperature model; an all-zero RateTemperatureLaw is the exact "
                "identity and IS accepted.".format(nm))
        if self.concentration is not None and float(self.concentration.pd_loss_per_m) > 0.0:
            raise ValueError(
                "ResolvedErYbAmplifier: ConcentrationModel.pd_loss_per_m = {:.6g} (Yb "
                "photodarkening) is not supported in v1. ErYbAmplifier subtracts the equilibrium "
                "gray loss from every channel's gain with NO overlap factor -- a channel-uniform "
                "background. Resolved, the gray loss is a per-node property of the doped glass "
                "and a channel picks up INT pd(bbar(r)) i_k(r) dA, i.e. the confinement factor. "
                "Those are two different models, and answering the second one under the name of "
                "the first would be silent. C_up and pair-induced quenching ARE supported (both "
                "are strictly local and both reduce exactly).".format(
                    float(self.concentration.pd_loss_per_m)))
        lam = np.asarray(self._mf._plan()["lam"], float)
        for nm, ion in (("er_ion", self.er_ion), ("yb_ion", self.yb_ion)):
            if ion.sigma_esa is None:
                continue
            esa = np.asarray(ion.sigma_esa_of(lam), float)
            if float(np.max(esa)) > 0.0:
                raise ValueError(
                    "ResolvedErYbAmplifier: {} has NONZERO excited-state absorption (max "
                    "sigma_esa = {:.3e} m^2 over the channel plan), which is not supported in "
                    "v1. ESA enters (E1.5) as an extra -sigma_esa_k Je_k inside the integrand. "
                    "Note ErYbAmplifier drops the same term SILENTLY; this class refuses rather "
                    "than inherit that.".format(nm, float(np.max(esa))))

    def set_temperature_profile(self, *args, **kwargs):
        """REFUSED in v1 -- see ResolvedFiberAmplifier.set_temperature_profile for the argument,
        which the second ion does not weaken but sharpen: the transfer defect is the dominant
        heat source in an EYDFA, so T is MOST radially structured exactly here."""
        raise ValueError(
            "ResolvedErYbAmplifier: set_temperature_profile is not supported in v1 -- an axial "
            "T(z) McCumber scaling is not meaningful once the inversion is resolved across the "
            "mode; it would need to become T(r, z), and in a co-doped fiber the transfer defect "
            "makes that radial structure larger, not smaller. Use eryb.ErYbAmplifier for the "
            "mean-field thermal-feedback loop.")

    # ---- PUBLIC amplifier re-seed protocol -------------------------------------------------
    # The SAME three methods FiberAmplifier and ErYbAmplifier expose, so metrics.* (gain_spectrum,
    # slope_efficiency) and chain.AmplifierChain can rebuild a stage without knowing which class
    # they hold. Without them this class would raise AttributeError there -- the audit A-3 failure
    # mode, one class later.
    def _clone(self, *, pumps=None, signals=None, ase=_KEEP, yb_ase=_KEEP
               ) -> "ResolvedErYbAmplifier":
        """Clone through THIS class's own constructor, carrying every opt-in of BOTH layers: the
        co-doped physics (which the mean-field twin holds and is re-read from it, so the two lists
        cannot drift) and the resolved geometry (profiles, modes, quadrature). The single place
        that lists what a resolved-ErYb clone must carry."""
        mf = self._mf
        return ResolvedErYbAmplifier(
            self.er_ion, self.yb_ion, self.fiber,
            list(self.pumps) if pumps is None else list(pumps),
            list(self.signals) if signals is None else list(signals),
            self.ase if ase is _KEEP else ase,
            n_yb_m3=mf._n_yb, k_tr_m3_s=mf._k_tr, k_back_m3_s=mf._k_back, a32_per_s=mf._a32,
            yb_ase=self.yb_ase if yb_ase is _KEEP else yb_ase,
            upconversion_C_up=self.upconversion_C_up, yb_coupled_fraction=mf._fc,
            k_tr2_m3_s=mf._k_tr2, yb_migration_rate_per_s=mf._w_mig,
            concentration=self.concentration,
            # both are None on any amplifier that exists (they are refused at construction), but
            # they are carried so that relaxing the refusal later needs ONE edit, not two
            rate_temperature=self.rate_temperature, yb_stark_thermal=self.yb_stark_thermal,
            er_profile=self.er_profile, yb_profile=self.yb_profile,
            signal_modes=(list(self.signal_modes) if signals is None else None),
            uniform_illumination=self.uniform_illumination,
            n_quad=self.n_quad, n_azimuthal=self.n_azimuthal, r_max_m=self.r_max_m)

    def with_signals(self, signals: List[Signal]) -> "ResolvedErYbAmplifier":
        """Re-seed with a new signal list, preserving this class and every opt-in. The per-signal
        MODE specs are DROPPED (reset to the fundamental), because they are solved for a specific
        wavelength and core and `check_signal_modes` would refuse them against a re-seeded signal
        anyway -- silently carrying them would be the wrong kind of preservation."""
        return self._clone(signals=signals)

    def with_pumps(self, pumps: List[Pump]) -> "ResolvedErYbAmplifier":
        """Re-seed with a new pump list, preserving this class and every opt-in. Used by
        metrics.slope_efficiency to sweep the launched pump."""
        return self._clone(pumps=pumps)

    def without_ase(self) -> "ResolvedErYbAmplifier":
        """A copy with BOTH ASE bands dropped -- the ASE-free configuration
        metrics.gain_spectrum probes the small-signal gain in."""
        return self._clone(ase=None, yb_ase=None)

    # ---- channel plan --------------------------------------------------------------------
    def _cladding_flags(self, n_channels: int) -> List[bool]:
        """Per-channel cladding-pump flags in the plan's own order (pumps first, then signals,
        then both ASE bands) -- ErYbAmplifier._plan's ordering, which this class inherits by
        CALLING that method rather than restating it."""
        flags = [bool(p.cladding) for p in self.pumps]
        return flags + [False] * (int(n_channels) - len(flags))

    def channel_plan(self):
        """This amplifier's channel structure -- see steady_state.ChannelPlan. Identical to the
        mean-field twin's EXCEPT that its `gamma` is the analytic overlap; read `gamma_quad`
        here for the quadrature overlaps this solver actually integrates."""
        return self._mf.channel_plan()

    def _plan(self):
        """Build (and cache) everything a solve needs: the mean-field twin's channel table, the
        quadrature grid, the normalized profiles, the two ion densities per node, and the hoisted
        coefficient bundle. Mirrors ResolvedFiberAmplifier._plan; the channel table itself is
        ErYbAmplifier._plan, CALLED not copied, so the two classes cannot drift."""
        if self._cache is not None:
            return self._cache
        pl = self._mf._plan()
        lam = np.asarray(pl["lam"], float)
        if lam.size == 0:
            raise ValueError("ResolvedErYbAmplifier: the channel plan is empty (no pumps, no "
                             "signals, no ASE band)")
        cladding = self._cladding_flags(lam.size)
        a = self.fiber.core_radius_m
        b_er, b_yb = float(self.er_profile.radius_m), float(self.yb_profile.radius_m)
        b_max = max(b_er, b_yb)
        if self.r_max_m is not None:
            r_max = float(self.r_max_m)
        else:
            r_max = max(default_r_max_m(self.fiber, lam, cladding, self.signal_modes),
                        1.2 * b_max)
        if r_max <= max(a, b_max):
            raise ValueError(
                "ResolvedErYbAmplifier: r_max_m = {:.6e} m does not exceed max(core, Er dopant, "
                "Yb dopant) radius {:.6e} m -- the quadrature would truncate the doped "
                "region".format(r_max, max(a, b_max)))
        if any(cladding) and r_max < float(self.fiber.clad_radius_m):
            # A cladding pump is FLAT over the inner cladding and the profile builder normalizes
            # it ON the grid, so truncating the grid inside R_clad does not clip the pump -- it
            # RENORMALIZES it over the smaller disc, silently inflating Gamma_p.
            raise ValueError(
                "ResolvedErYbAmplifier: r_max_m = {:.6e} m is inside the inner-cladding radius "
                "{:.6e} m while a cladding pump is present. The flat pump profile would be "
                "renormalized over the truncated disc instead of over the cladding, inflating "
                "its overlap by (R_clad/r_max)^2 = {:.4g}x. Raise r_max_m to at least the "
                "inner-cladding radius (the default already does).".format(
                    r_max, float(self.fiber.clad_radius_m),
                    (float(self.fiber.clad_radius_m) / r_max) ** 2))
        breaks = quadrature_breakpoints(self.fiber, r_max, cladding_present=any(cladding),
                                        extra_m=(b_er, b_yb))
        n_phi = self.n_azimuthal if any(
            isinstance(s, LPMode) and s.l >= 1 for s in self.signal_modes) else 1
        grid = RadialGrid.build(breaks, n_nodes_per_panel=self.n_quad, n_azimuthal=n_phi)

        n_p, n_s = len(self.pumps), len(self.signals)

        def mode_of(k):
            return self.signal_modes[k - n_p] if n_p <= k < n_p + n_s else None

        prof = build_normalized_profiles(self.fiber, grid, lam, cladding, mode_of,
                                         flat_all=self.uniform_illumination,
                                         where="ResolvedErYbAmplifier")
        n_er_r = self.er_profile.density(self._mf._n_er, grid.r_m)
        n_yb_r = self.yb_profile.density(self._mf._n_yb, grid.r_m)
        n_dark_r = self.er_profile.density(self._mf._n_er_dark, grid.r_m)
        keep = (n_er_r > 0.0) | (n_yb_r > 0.0)      # only DOPED nodes enter any integral
        if not np.any(keep):
            raise ValueError("ResolvedErYbAmplifier: no quadrature node carries either ion -- "
                             "check er_profile / yb_profile against the grid")
        wdope = grid.dA_m2[keep]
        idope = prof[:, keep]
        nu = C_LIGHT / lam
        hnu = H_PLANCK * nu
        is_ase = np.asarray(pl["is_ase"], bool)
        c = {
            "u": np.asarray(pl["u"], float),
            "is_ase": is_ase,
            "kind": list(pl["kind"]),
            "bc": np.asarray(pl["bc"], float),
            "lam": lam, "nu": nu,
            "sa_er": np.asarray(pl["sa_er"], float), "se_er": np.asarray(pl["se_er"], float),
            "sa_yb": np.asarray(pl["sa_yb"], float), "se_yb": np.asarray(pl["se_yb"], float),
            "loss": np.asarray(pl["loss"], float),
            "fa_er": np.asarray(pl["sa_er"], float) / hnu,   # rate per unit LOCAL intensity
            "fe_er": np.asarray(pl["se_er"], float) / hnu,
            "fa_yb": np.asarray(pl["sa_yb"], float) / hnu,
            "fe_yb": np.asarray(pl["se_yb"], float) / hnu,
            "s_pref": np.where(is_ase, np.asarray(pl["m"], float) * hnu
                               * np.asarray(pl["dnu"], float), 0.0),
            "gam": idope @ wdope,                            # quadrature overlaps Gamma_k
            "idope": idope, "wdope": wdope, "prof": prof, "keep": keep,
            "n_er": n_er_r[keep], "n_yb": n_yb_r[keep], "n_dark": n_dark_r[keep],
            "n_er_full": n_er_r, "n_yb_full": n_yb_r, "n_dark_full": n_dark_r,
            # the dark erbium is power-independent, so its unbleachable absorption integral is
            # hoisted out of the hot path entirely
            "jdark": idope @ (n_dark_r[keep] * wdope),
            # the node-local population parameters of (E1.4)
            "fc": self._mf._fc, "k_tr": self._mf._k_tr, "k_tr2": self._mf._k_tr2,
            "w_mig": self._mf._w_mig, "k_back": self._mf._k_back, "a32": self._mf._a32,
            "inv_tau_er": 1.0 / self._mf._tau_er, "inv_tau_yb": 1.0 / self._mf._tau_yb,
            "c_up": self.upconversion_C_up,
        }
        self._cache = (pl, grid, c)
        return self._cache

    # ---- pointwise physics ----------------------------------------------------------------
    @staticmethod
    def _params_at(c, nodes: str):
        """The (E1.4) parameter bundle on either the DOPED nodes (the hot path) or the FULL grid
        (the reporting pass). ONE helper, so the two can never drift apart."""
        p = {k: c[k] for k in ("fc", "k_tr", "k_tr2", "w_mig", "k_back", "a32",
                               "inv_tau_er", "inv_tau_yb", "c_up")}
        if nodes == "doped":
            p["n_er"], p["n_yb"] = c["n_er"], c["n_yb"]
        else:
            p["n_er"], p["n_yb"] = c["n_er_full"], c["n_yb_full"]
        return p

    def _populations(self, c, P, nodes: str = "doped"):
        """(f2, b2c, b2nc, bbar) at every node for one local channel-power vector P (K,) --
        (E1.3) followed by (E1.4). `bbar = f b2c + (1-f) b2nc` is the population-weighted Yb
        inversion the optical field sees."""
        prof = c["idope"] if nodes == "doped" else c["prof"]
        Ra_Er = (c["fa_er"] * P) @ prof
        Re_Er = (c["fe_er"] * P) @ prof
        Ra_Yb = (c["fa_yb"] * P) @ prof
        Re_Yb = (c["fe_yb"] * P) @ prof
        f2, b2c, b2nc = _solve_populations_nodes(Ra_Er, Re_Er, Ra_Yb, Re_Yb,
                                                 self._params_at(c, nodes))
        fc = c["fc"]
        bbar = b2c if fc == 1.0 else fc * b2c + (1.0 - fc) * b2nc
        return f2, b2c, b2nc, bbar

    @staticmethod
    def _overlaps(c, f2, bbar):
        """The four (K,) population integrals of (E1.5) on the doped nodes:
        (Je_Er, Ja_Er, Je_Yb, Ja_Yb) = INT n(r) x(r) i_k(r) dA."""
        w = c["wdope"]
        ner, nyb = c["n_er"], c["n_yb"]
        idope = c["idope"]
        return (idope @ (ner * f2 * w), idope @ (ner * (1.0 - f2) * w),
                idope @ (nyb * bbar * w), idope @ (nyb * (1.0 - bbar) * w))

    @staticmethod
    def _gain_from_J(c, Je_er, Ja_er, Je_yb, Ja_yb):
        """(K,) net gain coefficient [1/m] -- the brace of (E1.5). With every population constant
        over a top-hat dopant this is IDENTICALLY eryb.py's
        Gamma [N_Er(se f2 - sa(1-f2)) + N_Yb(se b2 - sa(1-b2))] - l - Gamma sa_Er n_dark."""
        return (c["se_er"] * Je_er - c["sa_er"] * Ja_er
                + c["se_yb"] * Je_yb - c["sa_yb"] * Ja_yb
                - c["sa_er"] * c["jdark"] - c["loss"])

    def _dP(self, c, P):
        """dP_k/dz [W/m] for every channel from the local power vector P (K,) -- (E1.5). The
        spontaneous seed carries the SAME overlap integrals as the stimulated term (one photon
        per mode) and is cross-section-weighted from BOTH ions, so a C-band ASE bin is seeded
        only by Er and a 1-um bin only by Yb, with no hand-assigned band boundary."""
        P = np.maximum(P, 0.0)
        f2, _b2c, _b2nc, bbar = self._populations(c, P)
        Je_er, Ja_er, Je_yb, Ja_yb = self._overlaps(c, f2, bbar)
        g = self._gain_from_J(c, Je_er, Ja_er, Je_yb, Ja_yb)
        src = c["s_pref"] * (c["se_er"] * Je_er + c["se_yb"] * Je_yb)
        return c["u"] * (g * P + src)

    # ---- public diagnostics ---------------------------------------------------------------
    @property
    def grid(self) -> RadialGrid:
        """The quadrature grid this amplifier integrates on."""
        return self._plan()[1]

    @property
    def profiles(self) -> np.ndarray:
        """(K, N) normalized transverse intensity profiles i_k(r,phi) [m^-2], INT i_k dA = 1."""
        return self._plan()[2]["prof"]

    @property
    def gamma_quad(self) -> np.ndarray:
        """(K,) confinement factors BY QUADRATURE over the doped region, Gamma_k = INT i_k dA.
        For a cladding pump this equals waveguide.cladding_pump_overlap by construction; for a
        single-mode core channel with a top-hat dopant it equals waveguide.overlap_gamma."""
        return self._plan()[2]["gam"]

    @property
    def er_density_r_m3(self) -> np.ndarray:
        """(N,) ACTIVE erbium density on the grid (pair-quenched ions excluded)."""
        return self._plan()[2]["n_er_full"]

    @property
    def yb_density_r_m3(self) -> np.ndarray:
        """(N,) ytterbium density on the grid (both pools)."""
        return self._plan()[2]["n_yb_full"]

    def local_populations(self, powers_W):
        """(f2, b2c, b2nc, bbar) at every grid node for one local channel-power vector -- the
        resolved counterpart of ErYbAmplifier._fbb. Nodes carrying neither ion still return the
        balance evaluated with the local intensities there; nothing downstream uses those."""
        _pl, _grid, c = self._plan()
        return self._populations(c, np.maximum(np.asarray(powers_W, float), 0.0), nodes="full")

    def local_gain_per_m(self, powers_W) -> np.ndarray:
        """(K,) local NET gain coefficient g_k [1/m] at one z, given the full channel power
        vector -- the brace of (E1.5). For a plan whose signals carry explicit LPModes these ARE
        the per-mode modal gains."""
        _pl, _grid, c = self._plan()
        P = np.maximum(np.asarray(powers_W, float), 0.0)
        f2, _bc, _bn, bbar = self._populations(c, P)
        return self._gain_from_J(c, *self._overlaps(c, f2, bbar))

    def dP_dz(self, powers_W) -> np.ndarray:
        """(K,) dP_k/dz [W/m] at one z from the full local power vector -- (E1.5)."""
        _pl, _grid, c = self._plan()
        return self._dP(c, np.asarray(powers_W, float))

    def local_transfer_rate_per_m3(self, powers_W) -> np.ndarray:
        """(N,) the LOCAL Yb -> Er transfer rate density k_tr n_Yb2(r) n_Er1(r) [m^-3 s^-1]
        (E1.1) at one z -- the quantity the mean-field closure replaces by a product of averages,
        and the one the radial mismatch between the Yb-absorbed pump and the Er inversion acts
        on. Integrate it against the grid's area weights for the transfer per unit length."""
        _pl, _grid, c = self._plan()
        f2, b2c, _bn, _bbar = self.local_populations(powers_W)
        n6c = c["fc"] * c["n_yb_full"] * b2c
        n1 = c["n_er_full"] * (1.0 - f2)
        return c["k_tr"] * n6c * n1

    # ---- RAM watchdog ----------------------------------------------------------------------
    def state_bytes(self, n_nodes: int = 201) -> int:
        """float64 bytes the ring-resolved state of a solve on `n_nodes` z-nodes occupies: FOUR
        (z-node x ring) population matrices (f2, b2c, b2nc, bbar) plus the (channel x z-node)
        powers and the (channel x ring) profiles. `solve` refuses above the package's 2 GiB
        bar, so a caller can size a run before launching it."""
        pl, grid, _c = self._plan()
        K = int(np.asarray(pl["lam"], float).size)
        N, M = int(grid.size), int(n_nodes)
        return int(8 * (4 * M * N + K * M + K * N))

    def _check_ram(self, n_nodes: int):
        """Refuse a solve whose ring-resolved state exceeds the package's existing 2 GiB bar,
        naming the ring x z-node x channel shape that produced it (the dynamics.py
        store_profiles guard, reused rather than re-declared)."""
        pl, grid, _c = self._plan()
        K = int(np.asarray(pl["lam"], float).size)
        need = self.state_bytes(n_nodes)
        if need > _STORE_PROFILES_MAX_BYTES:
            raise ValueError(
                "ResolvedErYbAmplifier.solve: the ring-resolved state would allocate %.2f GiB "
                "for %d rings x %d z-nodes x %d channels (four (z, ring) population matrices "
                "plus the powers and the profiles), above the package's %.2f GiB guard "
                "(dynamics._STORE_PROFILES_MAX_BYTES). Coarsen the quadrature (n_quad / "
                "n_azimuthal), shorten the z mesh (n_nodes), or drop ASE bins."
                % (need / 1024.0 ** 3, grid.size, int(n_nodes), K,
                   _STORE_PROFILES_MAX_BYTES / 1024.0 ** 3))

    # ---- the relaxation solve --------------------------------------------------------------
    def solve(self, *, n_nodes: int = 201, max_iter: int = 200, tol: float = 1e-6,
              method: str = "LSODA", relax=1.0) -> SteadyStateResult:
        """Steady-state solve. `relax` is a number in (0, 1] -- 1.0, the plain undamped
        Gauss-Seidel iteration -- or the string "auto", which walks steady_state._RELAX_LADDER
        and returns the first attempt that CONVERGES, exactly as ErYbAmplifier.solve does,
        reporting the ladder entries tried on `meta['relax_attempts']`."""
        self._check_ram(n_nodes)
        if isinstance(relax, str):
            if relax != "auto":
                raise ValueError("solve: relax must be a number in (0, 1] or \"auto\"; got %r"
                                 % (relax,))
            attempts, res = [], None
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
        """ONE relaxation solve at a fixed `relax` -- the SAME alternating forward/backward
        frozen-direction iteration ErYbAmplifier._solve_once and ResolvedFiberAmplifier.solve
        run, sharing their frozen-profile interpolator and their convergence test. The only
        difference from the mean-field co-doped solver is that each RHS evaluation solves the
        coupled balances at EVERY quadrature node and integrates, instead of solving them once
        against an area-averaged intensity."""
        from scipy.integrate import solve_ivp
        if not (0.0 < float(relax) <= 1.0):
            raise ValueError("solve: relax must be in (0, 1]; got %r" % (relax,))
        relax = float(relax)
        pl, grid, c = self._plan()
        u, bc = c["u"], c["bc"]
        K = u.size
        L = self.fiber.length_m
        z = np.linspace(0.0, L, n_nodes)
        fwd = np.where(u > 0)[0]
        bwd = np.where(u < 0)[0]
        P_bwd = (np.repeat(bc[bwd][:, None], n_nodes, axis=1) if bwd.size
                 else np.zeros((0, n_nodes)))
        P_fwd = np.repeat(bc[fwd][:, None], n_nodes, axis=1)

        def _assemble(Pf, Pb):
            P = np.empty(K)
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
            # the relax == 1.0 branch is taken verbatim so the default path performs NO arithmetic
            # on the arrays (audit F-14: a `1.0 * new + 0.0 * old` blend is not a no-op --
            # 0.0 * inf is NaN and -0.0 + 0.0 flips a sign).
            P_fwd = sf.y if relax == 1.0 else relax * sf.y + (1.0 - relax) * P_fwd

            if bwd.size:
                fwd_of = _frozen_profile_interp(P_fwd, z, L, n_nodes)

                def rhs_b(zz, Pb):
                    return self._dP(c, _assemble(fwd_of(zz), Pb))[bwd]

                sb = solve_ivp(rhs_b, (L, 0.0), bc[bwd], t_eval=z[::-1], method=method,
                               rtol=1e-7, atol=1e-15)
                new_bwd = sb.y[:, ::-1]
                P_bwd = new_bwd if relax == 1.0 else relax * new_bwd + (1.0 - relax) * P_bwd

            out = np.concatenate([P_fwd[:, -1], (P_bwd[:, 0] if bwd.size else [])])
            prof = np.concatenate([P_fwd, P_bwd], axis=0) if bwd.size else P_fwd.copy()
            if last_out is not None:
                end_resid, prof_resid = _relaxation_residuals(out, last_out, prof, last_prof)
                if end_resid < tol and prof_resid < tol:
                    converged = True
                    break
            last_out = out
            last_prof = prof

        P = np.empty((K, n_nodes))
        P[fwd] = P_fwd
        if bwd.size:
            P[bwd] = P_bwd

        f2_rz, b2c_rz, b2nc_rz, bbar_rz = self._population_profiles(c, P)
        f2_z, b2c_z, b2nc_z, bbar_z = self._population_reductions(c, f2_rz, b2c_rz, b2nc_rz,
                                                                  bbar_rz)
        keep = c["keep"]
        gain = np.empty((K, n_nodes))
        for j in range(n_nodes):
            gain[:, j] = self._gain_from_J(
                c, *self._overlaps(c, f2_rz[j, keep], bbar_rz[j, keep]))

        sig_idx = [i for i, kd in enumerate(c["kind"]) if kd == "signal"]
        gains_dB = np.array([10.0 * np.log10(P[i, -1] / bc[i]) for i in sig_idx])
        eta_tr = self._transfer_efficiency(c, P, f2_rz, b2c_rz, b2nc_rz, z)
        yb_par_dB = self._yb_parasitic_gain_dB(c, grid, f2_rz, bbar_rz, z)
        m_modes = (self.ase.m_modes if self.ase is not None
                   else (self.yb_ase.m_modes if self.yb_ase is not None else 2))
        meta = {
            "converged": converged, "iterations": it + 1,
            "endpoint_residual": end_resid, "profile_residual": prof_resid,
            "relax": float(relax), "min_power_W": float(np.min(P)),
            "dnu_hz": np.asarray(pl["dnu"], float).copy(), "m_modes": m_modes,
            # `gamma` is the QUADRATURE overlap actually used, not overlap_gamma -- the key name
            # is kept so the noise.* consumers (which duck-type a SteadyStateResult) read the
            # right thing. They combine it with the SCALAR nbar2_z, i.e. they see the mean-field
            # reduction of this solve, which is the honest read of an n_sp that is itself an
            # average over a mode.
            "gamma": c["gam"].copy(),
            "gamma_mean_field": np.asarray(pl["gamma"], float).copy(),
            "sigma_a": c["sa_er"].copy(), "sigma_e": c["se_er"].copy(),
            "sigma_a_er": c["sa_er"].copy(), "sigma_e_er": c["se_er"].copy(),
            "sigma_a_yb": c["sa_yb"].copy(), "sigma_e_yb": c["se_yb"].copy(),
            "beta_yb_z": bbar_z, "eta_transfer": eta_tr,
            "yb_parasitic_gain_dB": yb_par_dB,
            "n_er_m3": self._mf._n_er, "n_yb_m3": self._mf._n_yb,
            "k_tr_m3_s": c["k_tr"], "k_back_m3_s": c["k_back"], "a32_per_s": c["a32"],
            "beta_yb_coupled_z": b2c_z,
            "beta_yb_uncoupled_z": (b2nc_z if self._mf._two_pop else None),
            "yb_coupled_fraction": c["fc"], "k_tr2_m3_s": c["k_tr2"],
            "yb_migration_rate_per_s": c["w_mig"],
            "n_er_active_m3": self._mf._n_er, "n_er_dark_m3": self._mf._n_er_dark,
            "upconversion_C_up": self.upconversion_C_up,
            "mcc": None, "mcc_er": None, "mcc_yb": None,
            # ---- the resolved extras ----
            "resolved": True, "grid": grid, "doped_mask": keep.copy(),
            "f2_rz": f2_rz, "b2c_rz": b2c_rz, "b2nc_rz": b2nc_rz, "bbar_rz": bbar_rz,
            "n_er_r_m3": c["n_er_full"].copy(), "n_yb_r_m3": c["n_yb_full"].copy(),
            "profiles": c["prof"].copy(), "gain_per_m": gain,
            "n_quad": self.n_quad, "n_azimuthal": grid.n_azimuthal,
            "r_max_m": grid.r_max_m, "n_grid": grid.size,
            "uniform_illumination": self.uniform_illumination,
            "er_radius_m": float(self.er_profile.radius_m),
            "yb_radius_m": float(self.yb_profile.radius_m),
            "state_bytes": self.state_bytes(n_nodes),
        }
        return SteadyStateResult(z, P, np.asarray(pl["lam"], float), u,
                                 np.asarray(pl["is_ase"], bool), list(c["kind"]), f2_z,
                                 gains_dB, meta=meta)

    # ---- profiles and their scalar reductions ----------------------------------------------
    def _population_profiles(self, c, P):
        """((M, N) x 4) ring-resolved f2, b2c, b2nc and bbar on the FULL grid."""
        M = P.shape[1]
        N = c["prof"].shape[1]
        f2 = np.empty((M, N)); b2c = np.empty((M, N))
        b2nc = np.empty((M, N)); bbar = np.empty((M, N))
        for j in range(M):
            a, b, cc, d = self._populations(c, np.maximum(P[:, j], 0.0), nodes="full")
            f2[j], b2c[j], b2nc[j], bbar[j] = a, b, cc, d
        return f2, b2c, b2nc, bbar

    @staticmethod
    def _population_reductions(c, f2_rz, b2c_rz, b2nc_rz, bbar_rz):
        """The honest scalar reductions: each population averaged over its OWN ion's radial
        DISTRIBUTION, INT n(r) x(r) dA / INT n(r) dA -- i.e. what a population-weighted
        measurement (a fluorescence decay, a saturation curve) would report. For a top-hat dopant
        that is the plain dopant-area average ResolvedFiberAmplifier reports, and under uniform
        illumination it is the scalar model's own number exactly."""
        w = c["wdope"]
        keep = c["keep"]
        wer = c["n_er"] * w
        wyb = c["n_yb"] * w
        ser, syb = float(np.sum(wer)), float(np.sum(wyb))
        ser = ser if ser > 0.0 else 1.0
        syb = syb if syb > 0.0 else 1.0
        return (f2_rz[:, keep] @ wer / ser, b2c_rz[:, keep] @ wyb / syb,
                b2nc_rz[:, keep] @ wyb / syb, bbar_rz[:, keep] @ wyb / syb)

    # ---- diagnostics ------------------------------------------------------------------------
    def _transfer_efficiency(self, c, P, f2_rz, b2c_rz, b2nc_rz, z) -> float:
        """Fraction of Yb excited-state de-excitations that end as a USEFUL Yb -> Er transfer --
        the RING-SUMMED form of ErYbAmplifier._transfer_efficiency:

            eta = INT_z INT_A k_tr n1(r) n6c(r) dA dz
                / INT_z INT_A [(k_tr n1 + K2 n2 + 1/tau_Yb + R_e_Yb(r)) n6c
                               + (1/tau_Yb + R_e_Yb(r)) n6nc] dA dz

        with n1 = n_Er(r)(1 - f2(r)), n2 = n_Er(r) f2(r), n6c = f n_Yb(r) b2c(r) and
        n6nc = (1-f) n_Yb(r) b2nc(r). Every factor is evaluated at the node BEFORE the area
        integral, so the numerator is the true local transfer (E1.1) and not a product of
        averages; the mean-field expression is what this becomes when every profile is flat, and
        the two agree to round-off there (gated)."""
        keep = c["keep"]
        w, ner, nyb = c["wdope"], c["n_er"], c["n_yb"]
        fc, k_tr, k_tr2 = c["fc"], c["k_tr"], c["k_tr2"]
        inv_tY = c["inv_tau_yb"]
        M = P.shape[1]
        num = np.empty(M)
        den = np.empty(M)
        for j in range(M):
            Pj = np.maximum(P[:, j], 0.0)
            Re_Yb = (c["fe_yb"] * Pj) @ c["idope"]
            f2 = f2_rz[j, keep]
            n1 = ner * (1.0 - f2)
            n2 = ner * f2
            n6c = fc * nyb * b2c_rz[j, keep]
            n6nc = (1.0 - fc) * nyb * b2nc_rz[j, keep]
            idle = inv_tY + Re_Yb
            num[j] = float(np.dot(w, k_tr * n1 * n6c))
            den[j] = float(np.dot(w, (k_tr * n1 + k_tr2 * n2 + idle) * n6c + idle * n6nc))
        n = float(trapz(num, z))
        d = float(trapz(den, z))
        return n / d if d > 0.0 else 0.0

    def transfer_efficiency(self, result: SteadyStateResult) -> float:
        """eta_tr for a solved result (returns the value cached in meta by solve())."""
        return float(result.meta["eta_transfer"])

    def _yb_parasitic_gain_dB(self, c, grid: RadialGrid, f2_rz, bbar_rz, z) -> float:
        """Single-pass 1030 nm parasitic gain [dB] with the RESOLVED populations:

            G_dB = (10/ln10) INT_z INT_A [ n_Yb(r)(se_Yb bbar - sa_Yb (1 - bbar))
                                         + n_Er(r)(se_Er f2 - sa_Er (1 - f2)) ] i_1030(r) dA dz

        against the 1030 nm FUNDAMENTAL mode profile (or the mean-field profile under
        uniform_illumination), which is the resolved statement of the Gamma(1030) factor
        ErYbAmplifier uses. The Er terms are ~0 at 1030 nm. A large positive value flags 1-um
        parasitic-lasing risk (compare to the round-trip cavity loss -ln(R1 R2)/2; the design rule
        is beta_Yb < ~0.05). THIS is the diagnostic the radial profile can move most: the 1-um
        mode is core-guided, so it weights the ON-AXIS Yb inversion, while the cladding pump that
        sets that inversion is flat."""
        lam = LAMBDA_PARASITIC_M
        r = grid.r_m
        raw = (flat_intensity_profile(self.fiber, lam, r, grid.r_max_m)
               if self.uniform_illumination else fundamental_psi2(self.fiber, lam, r))
        i1030 = raw / float(np.dot(grid.dA_m2, raw))
        keep = c["keep"]
        wi = grid.dA_m2[keep] * i1030[keep]
        se_yb = float(self.yb_ion.sigma_e.sigma(lam))
        sa_yb = float(self.yb_ion.sigma_a.sigma(lam))
        se_er = float(self.er_ion.sigma_e.sigma(lam))
        sa_er = float(self.er_ion.sigma_a.sigma(lam))
        ner, nyb = c["n_er"], c["n_yb"]
        g = np.empty(f2_rz.shape[0])
        for j in range(g.size):
            f2, bb = f2_rz[j, keep], bbar_rz[j, keep]
            g[j] = float(np.dot(wi, nyb * (se_yb * bb - sa_yb * (1.0 - bb))
                                + ner * (se_er * f2 - sa_er * (1.0 - f2))))
        return 10.0 / np.log(10.0) * float(trapz(g, z))

    def yb_parasitic_gain_dB(self, result: SteadyStateResult) -> float:
        """1030 nm parasitic gain [dB] for a solved result (cached in meta by solve())."""
        return float(result.meta["yb_parasitic_gain_dB"])

    # ---- energy bookkeeping -----------------------------------------------------------------
    def _rate_balance_dissipation_W(self, result: SteadyStateResult) -> float:
        """Total dissipated optical power [W] from the LOCAL RATE BALANCE -- the hook
        `efficiency._dissipated_power_W` looks for FIRST, so `wall_plug_efficiency` returns a
        FINITE `energy_balance_residual_W` for a ring-resolved co-doped result instead of the NaN
        it reports for ResolvedFiberAmplifier (which defines no hook).

        The net forward flux is F(z) = sum_fwd P - sum_bwd P, so dF/dz = sum_k u_k dP_k/dz --
        THIS amplifier's own right-hand side, with the populations re-solved per ring from the
        returned powers and evaluated ON the returned profile rather than accumulated along it.
        The heat is -INT dF/dz dz, so the residual `launched - exiting - dissipated` measures
        whether the returned P(z) actually satisfies this amplifier's ODEs -- which an endpoint
        flux difference cannot. Returns NaN when the result did not come from this amplifier's
        channel plan."""
        pl, _grid, c = self._plan()
        P, z = result.power_W, result.z_m
        if P.shape[0] != np.asarray(pl["lam"], float).size or P.shape[1] != z.size:
            return float("nan")
        u = c["u"]
        dF = np.array([float(np.sum(u * self._dP(c, P[:, j]))) for j in range(z.size)])
        return -float(trapz(dF, z))

    def energy_terms(self, power_W, f2_rz, b2c_rz, b2nc_rz=None, *, z_m=None) -> dict:
        """Per-z energy bookkeeping [W/m], every term a RING INTEGRAL of the local rate balance.

        Same derivation, same charging convention and the same closure identity as
        `ErYbAmplifier.energy_terms` (see there for eps_Er / eps_Yb and the term-by-term
        argument), with every volumetric quantity INTEGRATED over the cross-section instead of
        multiplied by A_dope:

            q_opt(z) = dU/dt(z) + D_loss + D_Er + D_Yb + D_tr + D_K2,
            U(z)     = INT [eps_Er n_Er(r) f2(r) + eps_Yb n_Yb(r) bbar(r)] dA,
            Phi_tr   = INT k_tr phi n_Er(r)(1 - f2(r)) f n_Yb(r) b2c(r) dA,
            Phi_K2   = INT K2 n_Er(r) f2(r) f n_Yb(r) b2c(r) dA.

        Phi_tr is the point of the module: it is the AREA INTEGRAL OF A PRODUCT, so the transfer
        defect -- the dominant dissipation channel of an EYDFA -- carries the radial covariance
        (E1.2) the mean-field form drops. At the steady state dU/dt = 0 and the whole of q_opt is
        dissipation, so `closure_residual` is a genuine check on the solve rather than an
        identity that closes by construction.

        `power_W` is (K, M); the population arrays are (M, N) on THIS amplifier's grid;
        `b2nc_rz` is REQUIRED whenever an uncoupled pool exists and must be None otherwise -- the
        same rule the scalar class enforces, for the same reason (guessing it would mis-state
        both the stored energy and the Yb fluorescence). Returns the scalar class's key set (all
        (M,) arrays) plus 'transfer_covariance', Phi_tr minus the mean-field product of the same
        two area-averaged densities: exactly what resolving the transfer bought."""
        _pl, _grid, c = self._plan()
        P = np.maximum(np.asarray(power_W, float), 0.0)
        f2 = np.asarray(f2_rz, float)
        b2c = np.asarray(b2c_rz, float)
        N = c["prof"].shape[1]
        if P.ndim != 2 or f2.shape != (P.shape[1], N) or b2c.shape != (P.shape[1], N):
            raise ValueError("energy_terms: power_W must be (K, M) with f2_rz, b2c_rz of shape "
                             "(M, N) on this amplifier's grid (N = %d); got %r, %r, %r"
                             % (N, P.shape, f2.shape, b2c.shape))
        two_pop = self._mf._two_pop
        if two_pop and b2nc_rz is None:
            raise ValueError(
                "energy_terms: this amplifier has an UNCOUPLED ytterbium pool "
                "(yb_coupled_fraction=%g, k_tr2=%g, migration=%g), so the uncoupled "
                "ring-resolved inversion must be passed -- solve() reports it on "
                "meta['b2nc_rz']. Defaulting it would mis-state both the stored energy and the "
                "Yb fluorescence." % (c["fc"], c["k_tr2"], c["w_mig"]))
        if b2nc_rz is not None and not two_pop:
            raise ValueError("energy_terms: this amplifier has ONE ytterbium pool, so b2nc_rz "
                             "must be None (got an array of shape %r)" % (np.shape(b2nc_rz),))
        b2nc = b2c if b2nc_rz is None else np.asarray(b2nc_rz, float)
        if b2nc.shape != b2c.shape:
            raise ValueError("energy_terms: b2nc_rz must have shape %r; got %r"
                             % (b2c.shape, b2nc.shape))
        if z_m is not None and np.asarray(z_m, float).shape != (P.shape[1],):
            raise ValueError("energy_terms: z_m must have shape (M,); got %r" % (np.shape(z_m),))

        keep = c["keep"]
        w, ner, nyb = c["wdope"], c["n_er"], c["n_yb"]
        fc = c["fc"]
        eps_er = float(self.er_ion.eps_J)
        eps_yb = float(self.yb_ion.eps_J)
        inv_h = 1.0 / (H_PLANCK * c["nu"])
        area = float(np.sum(w))
        sum_er = max(float(np.dot(w, ner)), 1e-300)
        sum_yb = max(float(np.dot(w, nyb)), 1e-300)
        M = P.shape[1]
        keys = ("q_optical", "stored_J_per_m", "d_stored_dt", "background_loss",
                "er_dissipation", "yb_dissipation", "transfer_defect", "k2_defect",
                "dissipation", "transfer_rate_per_m", "k2_rate_per_m", "df2_dt", "db2_dt",
                "db2nc_dt", "beta_yb_mean", "transfer_covariance")
        out = {k: np.empty(M) for k in keys}
        for j in range(M):
            Pj = P[:, j]
            f = f2[j, keep]
            bc_ = b2c[j, keep]
            bn_ = b2nc[j, keep]
            bb = bc_ if fc == 1.0 else fc * bc_ + (1.0 - fc) * bn_
            Je_er, Ja_er, Je_yb, Ja_yb = self._overlaps(c, f, bb)
            pw_a_er = c["sa_er"] * Ja_er * Pj
            pw_e_er = c["se_er"] * Je_er * Pj
            pw_sp_er = c["s_pref"] * c["se_er"] * Je_er
            pw_a_yb = c["sa_yb"] * Ja_yb * Pj
            pw_e_yb = c["se_yb"] * Je_yb * Pj
            pw_sp_yb = c["s_pref"] * c["se_yb"] * Je_yb
            # the dark (pair-quenched) erbium absorbs but holds no gain: pure heat, so it sits in
            # the background-loss term exactly as the scalar class puts it in c["loss"].
            loss_W = float(np.sum((c["loss"] + c["sa_er"] * c["jdark"]) * Pj))
            p_a_er, p_e_er, p_sp_er = pw_a_er.sum(), pw_e_er.sum(), pw_sp_er.sum()
            p_a_yb, p_e_yb, p_sp_yb = pw_a_yb.sum(), pw_e_yb.sum(), pw_sp_yb.sum()
            n_a_er, n_e_er = (pw_a_er * inv_h).sum(), (pw_e_er * inv_h).sum()
            n_a_yb, n_e_yb = (pw_a_yb * inv_h).sum(), (pw_e_yb * inv_h).sum()
            q_opt = (p_a_er + p_a_yb + loss_W) - (p_e_er + p_e_yb + p_sp_er + p_sp_yb)

            Ra_er = (c["fa_er"] * Pj) @ c["idope"]
            Re_er = (c["fe_er"] * Pj) @ c["idope"]
            Ra_yb = (c["fa_yb"] * Pj) @ c["idope"]
            Re_yb = (c["fe_yb"] * Pj) @ c["idope"]
            phi = (np.ones_like(f) if c["k_back"] <= 0.0 else
                   c["a32"] / (c["a32"] + c["k_back"] * fc * (1.0 - bc_) * nyb))
            n6c = fc * nyb * bc_
            tr_node = c["k_tr"] * phi * ner * (1.0 - f) * n6c
            k2_node = c["k_tr2"] * ner * f * n6c
            mig = c["w_mig"] * (bc_ - bn_)
            df = (Ra_er * (1.0 - f) - Re_er * f - f * c["inv_tau_er"]
                  + c["k_tr"] * fc * nyb * phi * bc_ * (1.0 - f)
                  - c["c_up"] * ner * f * f)
            dbc = (Ra_yb * (1.0 - bc_) - Re_yb * bc_ - bc_ * c["inv_tau_yb"]
                   - c["k_tr"] * ner * phi * bc_ * (1.0 - f)
                   - c["k_tr2"] * ner * f * bc_ - (1.0 - fc) * mig)
            dbn = (Ra_yb * (1.0 - bn_) - Re_yb * bn_ - bn_ * c["inv_tau_yb"] + fc * mig)
            dbbar = dbc if fc == 1.0 else fc * dbc + (1.0 - fc) * dbn

            tr_rate = float(np.dot(w, tr_node))
            k2_rate = float(np.dot(w, k2_node))
            dec_er = float(np.dot(w, ner * f)) * c["inv_tau_er"]
            dec_yb = float(np.dot(w, nyb * bb)) * c["inv_tau_yb"]
            up_er = c["c_up"] * float(np.dot(w, ner * ner * f * f))
            d_er = ((p_a_er - p_e_er) - eps_er * (n_a_er - n_e_er)
                    + (eps_er * dec_er - p_sp_er) + eps_er * up_er)
            d_yb = ((p_a_yb - p_e_yb) - eps_yb * (n_a_yb - n_e_yb)
                    + (eps_yb * dec_yb - p_sp_yb))
            d_tr = (eps_yb - eps_er) * tr_rate
            d_k2 = eps_yb * k2_rate
            # the mean-field transfer the SAME averaged densities would have produced -- the
            # product of averages, against which tr_rate is the integral of the product
            mf_tr = (c["k_tr"] * float(np.dot(w, phi * ner * (1.0 - f)))
                     * float(np.dot(w, n6c)) / area if area > 0.0 else 0.0)

            out["q_optical"][j] = q_opt
            out["stored_J_per_m"][j] = float(np.dot(w, eps_er * ner * f + eps_yb * nyb * bb))
            out["d_stored_dt"][j] = float(np.dot(w, eps_er * ner * df + eps_yb * nyb * dbbar))
            out["background_loss"][j] = loss_W
            out["er_dissipation"][j] = d_er
            out["yb_dissipation"][j] = d_yb
            out["transfer_defect"][j] = d_tr
            out["k2_defect"][j] = d_k2
            out["dissipation"][j] = loss_W + d_er + d_yb + d_tr + d_k2
            out["transfer_rate_per_m"][j] = tr_rate
            out["k2_rate_per_m"][j] = k2_rate
            out["df2_dt"][j] = float(np.dot(w, ner * df)) / sum_er
            out["db2_dt"][j] = float(np.dot(w, nyb * dbc)) / sum_yb
            out["db2nc_dt"][j] = float(np.dot(w, nyb * dbn)) / sum_yb
            out["beta_yb_mean"][j] = float(np.dot(w, nyb * bb)) / sum_yb
            out["transfer_covariance"][j] = tr_rate - mf_tr
        if not two_pop:
            out.pop("db2nc_dt")
        return out

    def _heat_profile_W_per_m(self, result: SteadyStateResult) -> np.ndarray:
        """Local heat density Q(z) [W/m] from THIS amplifier's own RING-SUMMED rate balance --
        the same hook eryb.ErYbAmplifier offers thermal.py, so `thermal.solve_with_thermal_feedback`
        and every Q(z) consumer read `background loss + Er dissipation + Yb dissipation +
        transfer defect + K2 defect` instead of `np.gradient` of the net flux (which inherits the
        mesh error of the solve and says nothing about WHICH mechanism deposited the heat). For
        an EYDFA the transfer defect dominates -- the 4I11/2 -> 4I13/2 multiphonon step, 36% of
        every transferred 976 nm photon -- and here it carries the radial covariance (E1.2).

        NOTE the thermal FEEDBACK loop still cannot run on this class: it re-solves with
        `set_temperature_profile`, which v1 refuses (T would have to become T(r, z)). The hook is
        for the one-way Q(z) consumers."""
        return np.asarray(self.energy_terms_from_result(result)["dissipation"], float)

    def energy_terms_from_result(self, result: SteadyStateResult) -> dict:
        """`energy_terms` fed straight from a solved result's ring-resolved meta."""
        m = result.meta
        b2nc = m["b2nc_rz"] if self._mf._two_pop else None
        return self.energy_terms(result.power_W, m["f2_rz"], m["b2c_rz"], b2nc, z_m=result.z_m)

    def closure_residual(self, result: SteadyStateResult) -> float:
        """max |q_opt - (dU/dt + dissipation)| / max|q_opt| over z -- the RING-SUMMED energy
        closure. The identity is an algebraic rearrangement of one LOCAL rate balance evaluated
        node by node and then integrated, so a departure beyond round-off is an implementation
        defect, not physics."""
        t = self.energy_terms_from_result(result)
        resid = t["q_optical"] - (t["d_stored_dt"] + t["dissipation"])
        scale = float(np.max(np.abs(t["q_optical"])))
        return float(np.max(np.abs(resid))) / (scale if scale > 0.0 else 1.0)


def eryb_mean_field_equivalent(amp: ResolvedErYbAmplifier) -> ErYbAmplifier:
    """The mean-field `eryb.ErYbAmplifier` for the SAME plan and the SAME opt-ins -- the scalar
    twin of a resolved co-doped amplifier. Use it to read the transverse delta (resolved minus
    mean-field gain, wall-plug, eta_tr and 1-um parasitic gain) and as the oracle for the
    uniform-illumination reduction gate. It is the object the resolved amplifier was BUILT
    around, not a fresh construction, so every keyword is carried by construction and no clone
    protocol can drop one."""
    return amp._mf
