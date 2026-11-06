"""
NONLOCAL band-to-band tunnelling (dynamic-path model) coupled into the bipolar drift-diffusion Newton,
1-D (DynaMeta v0.15, plan item 4.1.3; user decisions 21-25). OPT-IN: importing this module does nothing
to DEVSIM; nothing changes unless setup_btbt_nonlocal() is called on a region.

=====================================================================================================
1. WHEN TO USE IT (and when the local model is enough)
   The LOCAL Kane/Hurkx term (carriers/tunneling.py, setup_btbt) evaluates A F^P exp(-B/F) at every
   point from the LOCAL field and puts the electron and the hole at the same place. That is right for a
   uniform field. It is wrong where
     (i)   the band bending over the tunnelling distance is LESS than the gap: there is no final state,
           but a high local field still "generates" (a 0.8 V step over 2 nm in Si: the local model gives
           a large rate, the nonlocal model exactly zero -- tests/test_btbt_nonlocal_kernels.py N2);
     (ii)  the field varies on the scale of the tunnelling length (graded / abrupt junctions): the
           tunnelling probability is set by the band profile along the whole path, not the field at x0;
     (iii) the carriers must be SEPARATED: the hole appears where the path starts, the electron where it
           ends (tunnel FETs, GIDL, any device whose terminals see the two carriers differently).
   The nonlocal rate is NOT generally lower than the local one (Jiao et al. 2013: about equal in Si
   homojunctions; Vandenberghe et al. 2011: every semiclassical model over-estimates near zero bias).
   Use it when (i)-(iii) matter; keep the local model for speed in smooth, wide junctions.

2. THE MODEL (Synopsys Sentaurus Device User Guide N-2017.09, "Dynamic Nonlocal Path Band-to-Band
   Tunneling Model", pp. 436-441; READ from the page images, see section 6)
   A path starts at every semiconductor node x = 0 (tunnelling energy eps = E_V(0)), runs straight
   along -grad E_V (in 1-D: towards increasing potential) and ends at the first point l where
   E_C(l) = eps. The net HOLE recombination rate at the start (per volume, per second) is
     direct           R_d = |grad E_V(0)| C_d exp(-2 Int_0^l kappa dx) [f_n(l) - f_p(0)]          (Eq. 480)
                      C_d = (g pi / 36 h) (Int_0^l dx/kappa)^-1 [1 - exp(-k_m^2 Int dx/kappa)]     (Eq. 481)
                      kappa = (1/hbar) sqrt(m_r E_g (1 - alpha^2))                                 (Eq. 482)
     phonon-assisted  R_p = |grad E_V(0)| C_p exp(-2 Int_0^x0 kappa_V dx - 2 Int_x0^l kappa_C dx)
                              [f_n(l) - f_p(0)]                                                  (Eq. 490)
                      C_p = Int_0^l [g (1+2N_op) D_op^2 / (2^6 pi^2 rho eps_op E_g)]
                              sqrt(m_V m_C / (h l sqrt(2 m_r E_g))) dx
                              (Int_0^x0 dx/kappa_V)^-1 (Int_x0^l dx/kappa_C)^-1 [1 - e^..][1 - e^..]  (Eq. 491)
                      kappa_V = (1/hbar) sqrt(2 m_V (eps - E_V)),  kappa_C = (1/hbar) sqrt(2 m_C (E_C - eps))
                                                                                         (Eq. 492-493)
                      x0: kappa_V = kappa_C.
   f_n(l) = 1/(exp((eps - E_Fn(l))/kT) + 1), f_p(0) = 1/(exp((eps - E_Fp(0))/kT) + 1). In reverse bias
   f_p(0) -> 1, f_n(l) -> 0: R < 0, i.e. GENERATION. Holes are generated at the start, electrons at the
   end; the integrated rates are equal by construction (charge conservation).

   CHOICES MADE HERE (each documented, each gated):
     * k_m -> infinity: the transverse-momentum cut-off factors [1 - exp(-k_m^2 ...)] are 1.
     * m0 -> infinity in alpha (Eq. 483): alpha = 2 (eps - E_V)/E_g - 1, so
         kappa = (2/hbar) sqrt(m_r (eps - E_V)(E_C - eps)/E_g),
       and with m0 -> infinity Eqs. 488-489 give m_V = m_C = 2 m_r, i.e. mass_ratio_V_over_C = 1 is the
       consistent default of the phonon process.
     With these two choices the model reduces EXACTLY to A F^P exp(-B/F) in a uniform field, with A, B of
     Eqs. 495-498 (re-derived in section 6 and gated to 1e-12 in N1). The finite-m0 alpha and the k_m
     cut-off break that identity; they are later options.
   PARAMETERS. The model takes the SAME fitted set as the local model, a tunneling.BTBTParams (A, B, P):
   P = 2 selects the direct process, P = 2.5 the phonon-assisted one (no other P is defined by the
   model). physical_from_fitted inverts Eqs. 495-496 (direct: m_r, g -- the inverse of
   tunneling.kane_direct_params, THE Kane kernel; fitted_from_physical calls it) or 497-498 (phonon: m_r
   and the combination Lambda = g (1 + 2 N_op) D_op^2 / (rho eps_op), with m_V/m_C from
   mass_ratio_V_over_C) -- Sentaurus' "second parameter set". The Sentaurus Table 90 defaults of this model
   ARE the Hurkx Si numbers (tunneling.SILICON_HURKX, P = 2.5) with eps_op = 0.037 eV.
   THE TUNNELLING GAP AND TEMPERATURE (NonlocalBTBT docstring). (A, B) are inverted at the gap of the
   set's QUOTED temperature (BTBTParams.T_K, None = T_REF); the path uses the gap at the REGION's
   temperature (Sentaurus N-2017.09 p. 440: "E_g(300K) ... is used solely for the definitions of the A and B
   parameters ... The band gap that is used for the WKB integrals features all dependencies (for example
   ... temperature)"). So the physical set (m_r, g / Lambda) is temperature-independent and, in a uniform
   field, B scales as E_g^(3/2) -- exactly the local model's Eq. 478 scaling -- while A scales as
   E_g^(-1/2) (direct) or E_g^(-7/4) (phonon), where Eq. 478 keeps A fixed (gated; the occupation of the
   optical phonons, (1 + 2 N_op(T)) inside Lambda, is not rescaled: eps_op is not a parameter here).
   The gap comes from ONE source: material= (the material's fundamental gap through
   materials.band_temperature.band_data_at_T, the package's one gap kernel -- right for phonon-assisted sets
   in indirect materials and direct sets in direct-gap materials) or Eg_tun_eV= (explicit, for the direct
   process of an indirect material: its Gamma gap is not the fundamental gap and is not tabulated).

3. DISCRETISATION (1-D)
   * Band edges in the region frame: E_V = -q psi + const, E_C = E_V + E_g (E_g = the path gap), linear on
     each edge (psi is P1). In units of the gap, v(x) = (psi(x) - psi(x0))/E_g runs from 0 at the start
     to 1 at the end.
   * START NODES: every interior node of the region that is not a contact node. |grad E_V(0)| is the
     second-order nodal gradient (h_- s_+ + h_+ s_-)/(h_- + h_+) (s_+- the two edge slopes; exact for a
     quadratic potential). The direction is its sign; a zero gradient starts no path.
   * PATH: walk the nodes in that direction; the end is the exact linear root v = 1 in the first edge
     that reaches it. A path is DISCARDED (and counted in last_stats) when it
       - leaves the region / reaches a non-contact region end before v = 1 ("boundary"; decision 24:
         specular reflection at interfaces is deferred),
       - does not rise monotonically (v not increasing from node to node, or a nodal slope <= 0 along
         it -- the band bends back: "reentrant"),
       - ends in an edge touching a CONTACT node ("contact": DEVSIM drops a contact node's bulk row,
         so the electron would be lost and charge not conserved),
       - is longer than max_length_m ("too_long").
     A smooth C^1 taper w(l) = 1 - (3 s^2 - 2 s^3), s = (l - 0.95 L_max)/(0.05 L_max), switches paths off
     over the last 5 % of max_length, so the Newton does not see a jump (P1 3.6); w = 1 EXACTLY for
     l <= 0.95 L_max (the uniform-field identity is untouched there).
   * SEGMENT INTEGRALS IN CLOSED FORM (no quadrature error), two kinds:
       EXPONENT integrals Int kappa ds (direct; Keldysh kappa_V, kappa_C) and the path length on the P1
         band (v linear on each edge): arcsin / sqrt forms, written cancellation-free (kane_segment,
         keldysh_segment: no arcsin of sqrt(v), (x - sin x) by series).
       PREFACTOR integrals Int ds/kappa (C_d, C_p) IN v: Int r(v) K(v) dv with r = ds/dv interpolated
         LINEARLY in v between the second-order nodal slopes of the path nodes (kane_moments,
         keldysh_moments: closed-form zeroth and first moments of K). WHY NOT P1 here too (the plan's first
         design; FINDING, tests/test_btbt_nonlocal_kernels.py N4b): K has an integrable 1/sqrt singularity
         at a path end, where the P1 slope is the edge SECANT, off the true slope by O(h v''); the end
         piece 2 sqrt(h/sigma) then carries an O(h^{3/2}) error (MEASURED order 1.3 against a continuum
         oracle, 2.0 with the slope interpolation), and the secant JUMP between two edges makes the rate a
         sqrt-kinked function of psi whenever a path end crosses a node (MEASURED: a DEVSIM dI/dV that
         moved by 2.5e-4 between +-1 mV and +-0.1 mV differences). With r interpolated the rate is C^1
         there and the scheme O(h^2) throughout. A uniform field has r constant: both forms coincide and
         the uniform-field identity is untouched.
     The phonon path splits at x0 (v = v* = m_C/(m_V + m_C), the exact linear root).
     A path must have v rising strictly from node to node and a positive nodal slope at every path node
     (else it is discarded as "reentrant": a band that bends back is not one tunnelling path).
   * OCCUPANCIES from the transport's own quasi-Fermi levels: the log activities
     L_n = (E_Fn - E_C)/kT and L_p = (E_V - E_Fp)/kT of physics_heterojunction.setup_log_activity (FD
     when the region uses the FD g-factor, Boltzmann otherwise -- the SAME relation the drift-diffusion
     current uses). E_Fn is interpolated LINEARLY to the end point. In the region's statistics frame
     (E_C - E_V = kT ln(N_c N_v / n_i^2) = E_g,stat), f_p(0) = 1/(1 + exp(L_p(0))) and
     f_n(l) = 1/(1 + exp(-L_n(l) - delta)), delta = ln(N_c N_v/n_i^2) - E_g,tun/kT. At a discrete
     equilibrium (flat E_F) f_n(l) = f_p(0) IDENTICALLY, whatever E_g,tun: the rate is zero to round-off
     (gate N3). f_n - f_p is evaluated as sinh((a-b)/2)/(2 cosh(a/2) cosh(b/2)), so this survives
     occupancies near 0 or 1.
   * ASSEMBLY: the hole row of the start node gets +q R V_i (V_i = NodeVolume; DEVSIM's sign: a net
     recombination U enters HoleGeneration = +q U; MEASURED identical to add_recombination_term), and the
     SAME amount goes to the electron rows of the end edge's nodes a, b with the linear weights
     (1 - t), t (ElectronGeneration = -q U). Charge is conserved exactly by construction (gate N5).

4. JACOBIAN (decision 25). In 1-D the direction is a sign, fixed except where the nodal gradient
   crosses zero (measure zero, and there the rate is zero): the Jacobian is EXACT and the term declares
   jacobian_exact = True, so ssac and noise accept it. R depends on psi at every node of the path and at
   the start node's two neighbours (gradient), on psi at the neighbours of every path node (the nodal
   slopes of the prefactor integrals), on the end-point fraction t (implicit: v(l) = 1), on n at the end
   edge's nodes and on p at the start node. The derivative of each closed-form segment integral
   is taken by COMPLEX-STEP differentiation of the closed form (Squire & Trapp 1998: f'(x) = Im f(x+ih)/h,
   exact to round-off -- no subtractive cancellation, no step-size error); the chain rule through the
   path sums, the end point, the taper and the occupancies is written out. Gate N6 compares the
   assembled matrix against central differences of DEVSIM's own residual, mid-path columns included.
   The sparsity pattern follows the paths (pattern_changes = True); DEVSIM re-does the symbolic
   factorisation when the matrix grows (UMFPACK, PARDISO with DEVSIM_NEW_SYMBOLIC != "0";
   custom_terms warns otherwise).

5. NOISE. Each tunnelling event moves ONE electron between two places: a recombination event removes a
   hole at the start node (+1 on its HoleContinuityEquation row) and an electron at the end point (-1
   on the ElectronContinuityEquation rows of a, b with weights (1 - t), t). Its rate counts both
   directions: P = V_i R_pre [f_n (1 - f_p) + f_p (1 - f_n)] (forward f_p (1 - f_n), backward
   f_n (1 - f_p); their difference is the net rate). The source is a noise_analysis.TwoPointSource with
   mechanism key "btbt_nl". Gate N9: S = 4kT Re Y at equilibrium; placing both carriers at the start
   (a same-node pair, the local shortcut) breaks it.

6. VERIFYING THE TRANSCRIPTION OF EQS. 490-491 (decision 22). The page image (lit_nonlocal/img/
   sd-0485.png) shows Eq. 491 as an integral Int_0^l [...] dx whose integrand holds only material
   constants and the path length l -- so for one material it is l x [...]. That reading was verified
   by taking the uniform-field limit of Eqs. 490-493 by hand: x0 = m_C l/(m_V + m_C),
   2 Int kappa = 2^(7/2) pi sqrt(m_r) E_g^(3/2)/(3 q h F) (= Eq. 498 EXACTLY) and
   (Int dx/kappa_V)^-1 (Int dx/kappa_C)^-1 = q F (m_V + m_C)/(2 hbar^2 l), giving
   R = (qF)^(5/2) g (1+2N_op) D_op^2 (m_V m_C)^(3/2) / (2^(21/4) h^(5/2) m_r^(5/4) rho eps_op E_g^(7/4))
   x exp(-B/F) -- Eq. 497 EXACTLY, every power and the 2^(21/4) included. A misplaced l, dx or
   integral changes at least one of those powers. So Sentaurus' phonon form (Eqs. 490-491) is what is
   implemented (decision 22, first option), not our average-field form; gate N10 re-checks the identity
   numerically to 1e-12. The direct form was verified the same way (Eqs. 495-496).

7. SCOPE (1-D, v0.15) AND REFUSALS. 1-D devices only (2-D/3-D raise NotImplementedError: decision 21;
   the walker and the nodal gradient are the only 1-D-specific pieces -- the segment kernels, the
   per-path rate, the chain rule and the assembly are written over generic (path, segment) arrays).
   One region; a path does not cross a region boundary. Refused: the local model in the same region
   (double counting), bandgap narrowing in the region (the gap would vary along the path: not
   implemented), statistics="fermi_dirac" (solution-dependent n_ie) -- and tunneling.setup_btbt refuses a
   region that carries a live nonlocal term (both raise tunneling.BTBTDoubleCountError). A region whose
   temperature is not the set's quoted one warns (AdvisoryWarning) unless the gap follows a temperature law
   (material= with a band-temperature model).

8. CONSTANTS. h = H_PLANCK and hbar = h/(2 pi) (NOT the rounded HBAR literal, which differs by 6e-10 and
   would break the uniform-field identity at ~1e-8 through B/F ~ 30).

PUBLIC API: NonlocalBTBT, setup_btbt_nonlocal, btbt_nonlocal_generation, physical_from_fitted,
fitted_from_physical, kane_segment, keldysh_segment, kane_moments, keldysh_moments, path_rates_1d,
PROCESSES, MECHANISM.
Gates: tests/test_btbt_nonlocal_kernels.py (kernels, no DEVSIM), tests/test_btbt_nonlocal.py (DEVSIM,
child processes: tests/_btbt_nonlocal_child.py). Audit note: docs/audit/2026-11-06-v0.15-nonlocal-btbt-1d.md.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from dynameta.constants import H_PLANCK, M_E, Q_E
from dynameta.carriers.tunneling import BTBTDoubleCountError, BTBTParams, kane_direct_params
from dynameta.warning_types import AdvisoryWarning

__all__ = ["NonlocalBTBT", "NonlocalPhysical", "setup_btbt_nonlocal", "btbt_nonlocal_generation",
           "physical_from_fitted", "fitted_from_physical", "kane_segment",
           "keldysh_segment", "kane_moments", "keldysh_moments", "path_rates_1d", "PROCESSES", "MECHANISM"]

PROCESSES = ("direct", "phonon")
MECHANISM = "btbt_nl"                       # noise_analysis.TWO_POINT_MECHANISMS key (decision 9)
TAPER_START = 0.95                          # w(l) = 1 below 0.95 max_length (module doc, section 3)
_HBAR = H_PLANCK / (2.0 * math.pi)          # section 8: h/(2 pi), not the rounded HBAR literal
_CS_H = 1.0e-30                             # complex-step size (exact for any h << 1; section 4)
_EQ_N, _EQ_P = "ElectronContinuityEquation", "HoleContinuityEquation"


# =====================================================================================================
# parameters
# =====================================================================================================

@dataclass(frozen=True)
class NonlocalBTBT:
    """Options of the nonlocal path model (module doc). `params` is the SAME fitted set the local model
    takes (tunneling.BTBTParams; P = 2 -> direct, P = 2.5 -> phonon-assisted), quoted at
    params.quoted_T_K. mass_ratio_V_over_C = m_V/m_C places the phonon process' x0 (it does not change the
    uniform-field rate); 1.0 = the m0 -> infinity limit of Eqs. 488-489. max_length_m: longer paths are
    switched off (smoothly, over the last 5 %).

    THE TUNNELLING GAP -- give exactly ONE source (module doc, section 2, TEMPERATURE):
      material   a materials.TransportModel: the gap is the material's FUNDAMENTAL gap from the ONE package
                 gap kernel, materials.band_temperature.band_data_at_T -- at the set's quoted temperature
                 to convert (A, B) into the physical set, and at the region's temperature for the path.
                 Use it when the tunnelling gap IS the fundamental gap: a phonon-assisted set in an
                 indirect material (Si, Ge: SILICON_HURKX, *_KAO_INDIRECT) or a direct set in a direct-gap
                 material (GaAs, InAs, In0.53Ga0.47As: IN053GA047AS_KANE). With a band-temperature model
                 on the material the temperature dependence is handled (no warning); without one the gap
                 is band_gap_eV at every T.
      Eg_tun_eV  an explicit gap [eV] AT params.quoted_T_K, used at every temperature. Use it when the
                 tunnelling gap is NOT the fundamental gap: the DIRECT (Gamma) process of an indirect
                 material -- SILICON_KAO_DIRECT, GERMANIUM_KAO_DIRECT, whose E_g is the Gamma gap (Ge
                 ~0.8 eV, not its 0.66 eV L gap) -- which the package's gap law does not tabulate."""
    params: BTBTParams
    Eg_tun_eV: Optional[float] = None
    mass_ratio_V_over_C: float = 1.0
    max_length_m: float = 20e-9
    material: Optional[object] = None

    def __post_init__(self):
        if not isinstance(self.params, BTBTParams):
            raise TypeError("NonlocalBTBT: params must be a tunneling.BTBTParams (got {!r})".format(
                type(self.params)))
        if float(self.params.P) not in (2.0, 2.5):
            raise ValueError("NonlocalBTBT: the nonlocal path model is defined for P = 2 (direct, Eqs. "
                             "480-484) and P = 2.5 (phonon-assisted, Eqs. 490-493) only; got P = {!r}"
                             .format(self.params.P))
        if not (self.params.A_SI > 0.0):
            raise ValueError("NonlocalBTBT: A_SI must be > 0 (it sets the physical prefactor)")
        if (self.Eg_tun_eV is None) == (self.material is None):
            raise ValueError("NonlocalBTBT: give exactly one tunnelling-gap source -- material= (the fundamental "
                             "gap from the package gap law) or Eg_tun_eV= (an explicit gap, e.g. the Gamma gap "
                             "of an indirect material's direct process)")
        names = ("mass_ratio_V_over_C", "max_length_m") + (("Eg_tun_eV",) if self.material is None else ())
        for nm in names:
            v = float(getattr(self, nm))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError("NonlocalBTBT: {} must be finite and > 0, got {!r}".format(nm, v))
        if self.material is not None:
            eg = tunnelling_gaps(self, self.params.quoted_T_K)[0]   # resolvable at the quoted temperature
            if not (math.isfinite(eg) and eg > 0.0):
                raise ValueError("NonlocalBTBT: the material's gap at {} K is {!r}".format(
                    self.params.quoted_T_K, eg))

    @property
    def process(self) -> str:
        return "direct" if float(self.params.P) == 2.0 else "phonon"


def tunnelling_gaps(spec: NonlocalBTBT, T_K: float):
    """(E_g at the set's quoted temperature, E_g of the path at T_K, temperature_handled) [eV]
    (NonlocalBTBT docstring). material: both from materials.band_temperature.band_data_at_T (the one gap
    kernel); handled = the material has a band-temperature model or T_K is the quoted temperature.
    Eg_tun_eV: the same number twice; handled = T_K is the quoted temperature."""
    T_set = spec.params.quoted_T_K
    if spec.material is None:
        g = float(spec.Eg_tun_eV)
        return g, g, float(T_K) == T_set
    from dynameta.materials.band_temperature import band_data_at_T
    e_set = band_data_at_T(spec.material, T_set).E_g_eV
    e_path = band_data_at_T(spec.material, float(T_K)).E_g_eV
    if e_set is None or e_path is None:
        raise ValueError("NonlocalBTBT: material has no band gap (band_gap_eV None and no absolute gap law)")
    handled = getattr(spec.material, "band_temperature", None) is not None or float(T_K) == T_set
    return float(e_set), float(e_path), handled


@dataclass(frozen=True)
class NonlocalPhysical:
    """The physical parameter set of the path model (Sentaurus' "first parameter set"), SI.
    direct: m_r and the degeneracy g; phonon: m_r, m_V, m_C and lam = g (1 + 2 N_op) D_op^2/(rho eps_op).
    m_V = m_C = 2 m_r for the direct process (unused there). Eg_J is the PATH gap (at the device
    temperature); Eg_set_J the gap the fitted (A, B) were converted at (None: = Eg_J)."""
    process: str
    Eg_J: float
    m_r: float
    m_V: float
    m_C: float
    g: float = 0.0
    lam: float = 0.0
    Eg_set_J: Optional[float] = None

    @property
    def v_star(self) -> float:
        """x0 in units of the gap: kappa_V = kappa_C <=> m_V v = m_C (1 - v)."""
        return self.m_C / (self.m_V + self.m_C)

    @property
    def K0(self) -> float:
        """direct: kappa = K0 sqrt(v (1 - v)), K0 = 2 sqrt(m_r E_g)/hbar (m0 -> infinity alpha)."""
        return 2.0 * math.sqrt(self.m_r * self.Eg_J) / _HBAR

    @property
    def KV(self) -> float:
        return math.sqrt(2.0 * self.m_V * self.Eg_J) / _HBAR

    @property
    def KC(self) -> float:
        return math.sqrt(2.0 * self.m_C * self.Eg_J) / _HBAR

    @property
    def Cd0(self) -> float:
        """direct: g pi/(36 h) of Eq. 481."""
        return self.g * math.pi / (36.0 * H_PLANCK)

    @property
    def Kc(self) -> float:
        """phonon: C_p = Kc sqrt(l) (Int dx/kappa_V)^-1 (Int dx/kappa_C)^-1 (Eq. 491, one material)."""
        return (self.lam / (64.0 * math.pi ** 2 * self.Eg_J)
                * math.sqrt(self.m_V * self.m_C / (H_PLANCK * math.sqrt(2.0 * self.m_r * self.Eg_J))))


def physical_from_fitted(spec: NonlocalBTBT, *, T_K: Optional[float] = None) -> NonlocalPhysical:
    """The physical set of `spec` for a device at T_K (None: the set's quoted temperature). (A, B) are
    inverted at the gap of the QUOTED temperature (tunnelling_gaps; Sentaurus: "E_g(300 K) ... is used
    solely for the definitions of A and B"), SI with F0 = 1 V/m:
      direct  B = pi^2 sqrt(m_r) E_g^(3/2)/(q h)                -> m_r;  A = g pi sqrt(m_r) q^2/(9 h^2 sqrt(E_g)) -> g
      phonon  B = 2^(7/2) pi sqrt(m_r) E_g^(3/2)/(3 q h)        -> m_r;  m_V = m_r (1 + r), m_C = m_r (1 + 1/r)
              A = lam (m_V m_C)^(3/2) q^(5/2)/(2^(21/4) h^(5/2) m_r^(5/4) E_g^(7/4))                    -> lam,
    and the path uses the gap at T_K (Sentaurus: "the band gap used for the WKB integrals features all
    dependencies"). The direct pair is the INVERSE of tunneling.kane_direct_params (the one forward kernel,
    gated in tests/test_btbt_nonlocal_kernels.py); the phonon pair (Eqs. 497-498) exists only here."""
    T = spec.params.quoted_T_K if T_K is None else float(T_K)
    eg_set, eg_path, _ = tunnelling_gaps(spec, T)
    Eg = Q_E * eg_set
    A, B = float(spec.params.A_SI), float(spec.params.B_V_per_m)
    h = H_PLANCK
    if spec.process == "direct":
        m_r = (Q_E * h * B / (math.pi ** 2 * Eg ** 1.5)) ** 2
        g = 9.0 * h * h * math.sqrt(Eg) * A / (math.pi * math.sqrt(m_r) * Q_E * Q_E)
        return NonlocalPhysical("direct", Q_E * eg_path, m_r, 2.0 * m_r, 2.0 * m_r, g=g, Eg_set_J=Eg)
    r = float(spec.mass_ratio_V_over_C)
    m_r = (3.0 * Q_E * h * B / (2.0 ** 3.5 * math.pi * Eg ** 1.5)) ** 2
    m_V, m_C = m_r * (1.0 + r), m_r * (1.0 + 1.0 / r)
    lam = (A * 2.0 ** 5.25 * h ** 2.5 * m_r ** 1.25 * Eg ** 1.75
           / ((m_V * m_C) ** 1.5 * Q_E ** 2.5))
    return NonlocalPhysical("phonon", Q_E * eg_path, m_r, m_V, m_C, lam=lam, Eg_set_J=Eg)


def fitted_from_physical(phys: NonlocalPhysical, *, at_path_gap: bool = False):
    """(A_SI, B_V_per_m) of the physical set (the forward direction of physical_from_fitted), at the
    conversion gap Eg_set_J (default) or at the path gap (at_path_gap=True: the uniform-field (A, B) the
    model reproduces at the device temperature). Direct: tunneling.kane_direct_params, THE Kane kernel
    (Eqs. 495-496); phonon: Eqs. 497-498."""
    Eg = phys.Eg_J if (at_path_gap or phys.Eg_set_J is None) else phys.Eg_set_J
    h, m_r = H_PLANCK, phys.m_r
    if phys.process == "direct":
        k = kane_direct_params(Eg / Q_E, m_r / M_E, phys.g, source="btbt_nonlocal.fitted_from_physical")
        return k.A_SI, k.B_V_per_m
    A = (phys.lam * (phys.m_V * phys.m_C) ** 1.5 * Q_E ** 2.5
         / (2.0 ** 5.25 * h ** 2.5 * m_r ** 1.25 * Eg ** 1.75))
    B = 2.0 ** 3.5 * math.pi * math.sqrt(m_r) * Eg ** 1.5 / (3.0 * Q_E * h)
    return A, B


def has_live_nonlocal(device: str, region: str) -> bool:
    """True when `region` of `device` carries a LIVE nonlocal BTBT term (tunneling.setup_btbt refuses the
    local model there). Never imports or registers custom_terms: a process that never imported it has no
    term."""
    import sys
    ct = sys.modules.get("dynameta.carriers.custom_terms")
    if ct is None or _TERM_CLASS is None:
        return False
    return any(isinstance(t, _TERM_CLASS) and t.region == region for t in ct.live_terms(device))


# =====================================================================================================
# closed-form segment integrals (complex-safe: every branch is selected on the REAL part, so the
# complex step of section 4 differentiates them exactly)
# =====================================================================================================

def _atanc(y):
    """arctan(y)/y, series below |y| = 1e-3 (truncation y^8 ~ 1e-24)."""
    y = np.asarray(y)
    small = np.abs(y.real) < 1.0e-3
    yy = np.where(small, 0.5, y)
    y2 = y * y
    ser = 1.0 - y2 * (1.0 / 3.0 - y2 * (1.0 / 5.0 - y2 / 7.0))
    return np.where(small, ser, np.arctan(yy) / yy)


def _sinc(x):
    """sin(x)/x, series below |x| = 1e-3."""
    x = np.asarray(x)
    small = np.abs(x.real) < 1.0e-3
    xx = np.where(small, 0.5, x)
    x2 = x * x
    ser = 1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 / 5040.0))
    return np.where(small, ser, np.sin(xx) / xx)


def _c3(x):
    """(x - sin x)/x^3 = sum_k (-1)^k x^(2k)/(2k+3)!, series below |x| = 0.5 (8 terms: truncation
    < 2e-19 relative); direct above (round-off 6 eps/x^2 < 3e-15)."""
    x = np.asarray(x)
    small = np.abs(x.real) < 0.5
    xx = np.where(small, 1.0, x)
    x2 = x * x
    ser = np.full_like(x2, _C3_COEF[-1])
    for c in _C3_COEF[-2::-1]:
        ser = c + x2 * ser
    return np.where(small, ser, (xx - np.sin(xx)) / (xx * xx * xx))


_C3_COEF = tuple((-1.0) ** k / math.factorial(2 * k + 3) for k in range(8))


def kane_segment(va, vb, L):
    """Int ds / k and Int k ds over a segment of length L on which v runs LINEARLY from va to vb, for
    k(v) = sqrt(v (1 - v)) (the direct-process kappa / K0), 0 <= va, vb <= 1, not both 0 or both 1.
    Closed form: with phi = arcsin(sqrt v), F0 = 2 phi and F1 = (4 phi - sin 4 phi)/16 are the
    antiderivatives (F0(1) = pi, F1(1) = pi/8), and on a segment, with x = 2 (phi_b - phi_a),
      dF0 = x,   dF1 = [2 (x - sin x) + 4 sin x S^2]/16,
      S = sin(phi_a + phi_b) = sqrt(vb (1 - va)) + sqrt(va (1 - vb)),
      C = cos(phi_b - phi_a) = sqrt((1 - va)(1 - vb)) + sqrt(va vb),
      x = 2 arctan(y), y = tan(phi_b - phi_a) = (vb - va)/(S C)    (rationalised sine difference).
    No arcsin of sqrt(v) is ever taken: it is ill-conditioned at v -> 1 (an ulp of v costs sqrt(eps) ~
    1.5e-8 there; MEASURED 4e-11 in Int k before this form), and x - sin x is a series (_c3). Each
    segment returns L dF/dv, exact at va == vb (dF0/dv -> 1/k). Complex-safe (section 4)."""
    va, vb, L = np.asarray(va), np.asarray(vb), np.asarray(L)
    S = np.sqrt(vb * (1.0 - va)) + np.sqrt(va * (1.0 - vb))
    C = np.sqrt((1.0 - va) * (1.0 - vb)) + np.sqrt(va * vb)
    whole = C.real == 0.0                              # va = 0, vb = 1 (a one-segment path): x = pi
    SC = np.where(whole, 1.0, S * C)
    y = (vb - va) / SC
    d0 = np.where(whole, math.pi, 2.0 * _atanc(y) / SC)    # dF0/dv = x/dv (dv = 1 when whole)
    x = np.where(whole, math.pi, d0 * (vb - va))
    d1 = d0 * (2.0 * x * x * _c3(x) + 4.0 * _sinc(x) * S * S) / 16.0
    return L * d0, L * d1


def keldysh_segment(wa, wb, L):
    """Int ds / sqrt(w) and Int sqrt(w) ds over a segment of length L on which w runs linearly from wa
    to wb (w = v for kappa_V / KV, w = 1 - v for kappa_C / KC), rationalised (no cancellation):
      Int ds/sqrt(w) = L 2/(sqrt wa + sqrt wb),  Int sqrt(w) ds = L (2/3)(wa + sqrt(wa wb) + wb)/(sqrt wa + sqrt wb)."""
    wa, wb, L = np.asarray(wa), np.asarray(wb), np.asarray(L)
    sa, sb = np.sqrt(wa), np.sqrt(wb)
    den = sa + sb
    return L * 2.0 / den, L * (2.0 / 3.0) * (wa + sa * sb + wb) / den


def kane_moments(p0, p1):
    """For K(v) = 1/sqrt(v (1 - v)) on a piece p0 -> p1 (0 <= p0 < p1 <= 1, not 0 -> 1 unless whole):
    M0 = Int K dv and M1 = Int (v - p0) K dv / (p1 - p0), so that Int r(v) K dv = r0 M0 + (r1 - r0) M1
    for r linear in v from r0 at p0 to r1 at p1 (section 3: the slope-interpolated prefactor integral).
    With v = sin^2(phi), dphi = phi1 - phi0 = arctan(y) (y as in kane_segment):
      M0 = 2 dphi,   Int (v - p0) K dv = cos(2 phi0) (dphi - sin(2 dphi)/2) + sin(2 phi0) sin^2(dphi),
    divided by p1 - p0 = S sin(dphi), written cancellation-free (dphi - sin(2 dphi)/2 = 4 dphi^3 c3(2 dphi)).
    Complex-safe."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    S = np.sqrt(p1 * (1.0 - p0)) + np.sqrt(p0 * (1.0 - p1))
    C = np.sqrt((1.0 - p0) * (1.0 - p1)) + np.sqrt(p0 * p1)
    whole = C.real == 0.0
    y = (p1 - p0) / np.where(whole, 1.0, S * C)
    dphi = np.where(whole, 0.5 * math.pi, np.arctan(np.where(whole, 0.0, y)))
    sdphi = np.where(whole, 1.0, y / np.sqrt(1.0 + y * y))             # sin(arctan y)
    c2, s2 = 1.0 - 2.0 * p0, 2.0 * np.sqrt(p0 * (1.0 - p0))
    M1 = (c2 * 4.0 * dphi * dphi * _c3(2.0 * dphi) / _sinc(dphi) + s2 * sdphi) / S
    return 2.0 * dphi, M1


def keldysh_moments(w0, w1, rising: bool, dp=None):
    """M0 = Int dv/sqrt(w) and M1 = Int (v - p0)/sqrt(w) dv / (p1 - p0) on a piece p0 -> p1 (p1 > p0) on
    which w runs linearly from w0 to w1 (w = v: rising=True, kappa_V; w = 1 - v: rising=False, kappa_C):
      M0 = 2 dp/(s0 + s1),  M1 = (2/3) dp (s1 + 2 s0)/(s0 + s1)^2 (rising) or (2/3) dp (2 s0 + s1)/(s0 + s1)^2,
    s = sqrt(w), dp = p1 - p0 = |w1 - w0| (rationalised; no cancellation). Pass dp from the v values when
    w = 1 - v (w0 - w1 would lose the relative accuracy of a short piece). Complex-safe."""
    w0, w1 = np.asarray(w0), np.asarray(w1)
    s0, s1 = np.sqrt(w0), np.sqrt(w1)
    if dp is None:
        dp = (w1 - w0) if rising else (w0 - w1)
    den = s0 + s1
    lead = (s1 + 2.0 * s0) if rising else (2.0 * s0 + s1)
    return 2.0 * dp / den, (2.0 / 3.0) * dp * lead / (den * den)


# =====================================================================================================
# paths (1-D walker) and per-path rates
# =====================================================================================================

@dataclass
class Paths:
    """Generic (path, segment) arrays. Per path p: start node, end-edge nodes a, b. Per segment s:
    seg_path[s], the nodes ka -> kb of the edge it lies on, its length L, flags first (va == 0
    identically) and end (contains the path end), and the phonon phase (0: before x0, kappa_V;
    1: contains x0, the first v = v*; 2: after x0, kappa_C; always 0 for the direct process).
    The 1-D walker fills it; a 2-D walker would fill the same arrays with edge-crossing points."""
    start: np.ndarray
    a: np.ndarray
    b: np.ndarray
    seg_path: np.ndarray
    seg_ka: np.ndarray
    seg_kb: np.ndarray
    seg_L: np.ndarray
    seg_first: np.ndarray
    seg_end: np.ndarray
    seg_phase: np.ndarray
    direction: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    stats: Dict[str, int] = field(default_factory=dict)


def nodal_gradient_1d(x, psi):
    """Nodal d(psi)/dx: second order at the interior nodes, (h_- s_+ + h_+ s_-)/(h_- + h_+) (the
    derivative of the quadratic through the node and its two neighbours), the one-sided edge slope at
    the two end nodes; and its weights on psi_{i-1}, psi_i, psi_{i+1} (linear). Returns (g, wm, w0, wp)."""
    x, psi = np.asarray(x, float), np.asarray(psi, float)
    n = x.size
    g = np.zeros(n)
    wm, w0, wp = np.zeros(n), np.zeros(n), np.zeros(n)
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]
    den = hm + hp
    cp = hm / (hp * den)                                # weight of (psi_{i+1} - psi_i)
    cm = hp / (hm * den)                                # weight of (psi_i - psi_{i-1})
    g[1:-1] = cp * (psi[2:] - psi[1:-1]) + cm * (psi[1:-1] - psi[:-2])
    wp[1:-1], wm[1:-1], w0[1:-1] = cp, -cm, cm - cp
    if n >= 2:
        h0, h1 = x[1] - x[0], x[-1] - x[-2]
        g[0], w0[0], wp[0] = (psi[1] - psi[0]) / h0, -1.0 / h0, 1.0 / h0
        g[-1], wm[-1], w0[-1] = (psi[-1] - psi[-2]) / h1, -1.0 / h1, 1.0 / h1
    return g, wm, w0, wp


def trace_paths_1d(x, psi, Eg_V, max_length_m, start_ok, end_ok, v_star=None) -> Paths:
    """The 1-D walker (module doc, section 3). x sorted ascending; start_ok / end_ok boolean masks
    (start_ok: candidate start nodes, interior only; end_ok: nodes an end edge may touch). v_star: the
    phonon split level (None for the direct process)."""
    x, psi = np.asarray(x, float), np.asarray(psi, float)
    n = x.size
    g = nodal_gradient_1d(x, psi)[0]
    stats = {"starts": 0, "paths": 0, "boundary": 0, "reentrant": 0, "contact": 0, "too_long": 0}
    # cheap screen: the potential reachable on each side must rise by Eg somewhere
    sufmax = np.maximum.accumulate(psi[::-1])[::-1]              # max(psi[k:])
    premax = np.maximum.accumulate(psi)                          # max(psi[:k+1])
    starts, dirs, A, B, nodes_list, splits = [], [], [], [], [], []
    for i in np.flatnonzero(np.asarray(start_ok, bool)):
        if i <= 0 or i >= n - 1 or g[i] == 0.0:
            continue
        d = 1 if g[i] > 0.0 else -1
        reach = (sufmax[i + 1] if d > 0 else premax[i - 1]) - psi[i]
        if reach < Eg_V:
            continue
        stats["starts"] += 1
        # nodes in the walking direction, up to one node past max_length (the path may end inside it)
        if d > 0:
            stop = int(np.searchsorted(x, x[i] + max_length_m, side="right"))
            ks = np.arange(i + 1, min(stop + 1, n))
        else:
            stop = int(np.searchsorted(x, x[i] - max_length_m, side="left")) - 1
            ks = np.arange(i - 1, max(stop - 1, -1), -1)
        v = (psi[ks] - psi[i]) / Eg_V                              # band bending in units of the gap
        hit = np.flatnonzero(v >= 1.0)
        if hit.size == 0:
            truncated = (ks[-1] != (n - 1 if d > 0 else 0)) if ks.size else False
            stats["too_long" if truncated else "boundary"] += 1
            continue
        j = int(hit[0])
        # v must rise strictly from node to node, with a positive nodal slope at every path node: the
        # prefactor integrals are taken in v (section 3); a band that bends back is not a single path
        if np.any(np.diff(np.concatenate(([0.0], v[:j + 1]))) <= 0.0) or np.any(d * g[ks[:j + 1]] <= 0.0):
            stats["reentrant"] += 1
            continue
        b = int(ks[j])
        a = int(ks[j - 1]) if j > 0 else int(i)
        va = v[j - 1] if j > 0 else 0.0
        t = (1.0 - va) / (v[j] - va)
        if abs(x[a] - x[i]) + t * abs(x[b] - x[a]) >= max_length_m:
            stats["too_long"] += 1
            continue
        if not (end_ok[a] and end_ok[b]):
            stats["contact"] += 1
            continue
        starts.append(int(i))
        dirs.append(d)
        A.append(a)
        B.append(b)
        nodes_list.append(np.concatenate(([i], ks[:j + 1])))
        # phonon: x0 lies in the first segment whose upper node reaches v* (always <= j: v[j] >= 1 > v*)
        splits.append(int(np.flatnonzero(v[:j + 1] >= v_star)[0]) if v_star is not None else -1)
    stats["paths"] = len(starts)
    if not starts:
        e = np.zeros(0, dtype=np.int64)
        return Paths(e, e, e, e, e, e, np.zeros(0), e.astype(bool), e.astype(bool), e, e, stats)
    nseg = np.array([len(p) - 1 for p in nodes_list])
    seg_path = np.repeat(np.arange(len(starts)), nseg)
    allnodes = np.concatenate(nodes_list)
    offs = np.concatenate(([0], np.cumsum(nseg + 1)[:-1]))
    jloc = np.arange(seg_path.size) - np.repeat(np.concatenate(([0], np.cumsum(nseg)[:-1])), nseg)
    ka = allnodes[np.repeat(offs, nseg) + jloc]
    kb = allnodes[np.repeat(offs, nseg) + jloc + 1]
    first = jloc == 0
    end = jloc == np.repeat(nseg - 1, nseg)
    phase = np.zeros(seg_path.size, np.int64)
    if v_star is not None:
        sp = np.repeat(np.asarray(splits), nseg)
        phase = np.where(jloc < sp, 0, np.where(jloc == sp, 1, 2))
    return Paths(np.asarray(starts, np.int64), np.asarray(A, np.int64), np.asarray(B, np.int64),
                 seg_path, ka.astype(np.int64), kb.astype(np.int64), np.abs(x[kb] - x[ka]), first, end,
                 phase, np.asarray(dirs, np.int64), stats)


def _segment_accumulators(phys: NonlocalPhysical, va, vb, ra, rb, L, end, phase):
    """Per-segment contributions to the path accumulators as functions of the segment's end values
    (va, vb) and inverse slopes (ra, rb) = ds/dv at its nodes; complex-safe (section 4).
    direct: I0 = Int ds/k (k = kappa/K0), I1 = Int k ds, len, t;
    phonon: IV, JV (kappa_V/KV over [0, x0]), IC, JC (kappa_C/KC over [x0, l]), len, t.
    The EXPONENT integrals (I1, JV, JC) and the length are taken on the P1 band (v linear in s); the
    PREFACTOR integrals (I0, IV, IC) in v, Int r(v) K(v) dv with r = ds/dv interpolated linearly in v
    between the nodes' second-order slopes (section 3: O(h^2) and C^1 when a path end crosses a node,
    where the P1 form is O(h^{3/2}) and has a sqrt kink). r = const = L/(vb - va) gives the P1 values.
    An END segment holds the piece from va to v = 1 (fraction t = (1 - va)/(vb - va) of the edge); the
    phonon SPLIT segment (phase 1) holds the V piece up to v* (fraction tau) and the C piece from v* on.
    t is 1 (and constant) off the end segment."""
    one = np.ones_like(va)
    t = np.where(end, (1.0 - va) / np.where(end, vb - va, 1.0), one)
    Lp = L * t
    vhi = np.where(end, one, vb)
    dr = rb - ra
    if phys.process == "direct":
        i1 = kane_segment(va, vhi, Lp)[1]
        m0, m1 = kane_moments(va, vhi)
        i0 = ra * m0 + dr * t * m1
        return {"I0": i0, "I1": i1, "len": Lp, "t": t}
    vs = phys.v_star
    split = phase == 1
    tau = np.where(split, (vs - va) / np.where(split, vb - va, 1.0), np.where(phase == 0, t, 0.0 * t))
    vV = np.where(split, vs + 0.0 * va, vb)
    JV = keldysh_segment(va, vV, tau * L)[1]
    m0, m1 = keldysh_moments(va, vV, rising=True)
    IV = np.where(phase == 2, 0.0 * va, ra * m0 + dr * tau * m1)
    vC0 = np.where(split, vs + 0.0 * va, va)
    JC = keldysh_segment(1.0 - vC0, 1.0 - vhi, (t - tau) * L)[1]
    m0, m1 = keldysh_moments(1.0 - vC0, 1.0 - vhi, rising=False, dp=vhi - vC0)
    rC0 = ra + dr * tau * split
    IC = np.where(phase == 0, 0.0 * va, rC0 * m0 + dr * (t - tau * split) * m1)
    return {"IV": IV, "JV": JV, "IC": IC, "JC": JC, "len": Lp, "t": t}


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, float)))


def _taper(length, max_length_m):
    """w(l) and dw/dl: 1 up to TAPER_START max_length, C^1 smoothstep to 0 at max_length."""
    w0 = TAPER_START * max_length_m
    span = (1.0 - TAPER_START) * max_length_m
    s = np.clip((length - w0) / span, 0.0, 1.0)
    return 1.0 - s * s * (3.0 - 2.0 * s), -6.0 * s * (1.0 - s) / span


def path_rates_1d(x, psi, Ln, Lp, phys: NonlocalPhysical, delta: float, max_length_m: float,
                  start_ok, end_ok, *, derivatives: bool = True) -> dict:
    """The nonlocal rate of every path of a 1-D band profile (module doc, sections 2-4).

    x (sorted ascending) [m], psi [V], Ln = (E_Fn - E_C)/kT and Lp = (E_V - E_Fp)/kT at the nodes,
    delta = ln(N_c N_v/n_i^2) - E_g,tun/kT (the statistics-frame offset of f_n), start_ok / end_ok masks.
    Returns {"paths": Paths, "R": net hole recombination rate density at each path's start [m^-3 s^-1]
    (< 0 = generation), "R_pre": |grad E_V| C exp(-2 Int kappa) w(l) (>= 0), "pair": f_n (1-f_p) +
    f_p (1-f_n), "t": end fraction, "length": l}; with derivatives=True also
    "jac": {"pe", "node", "dR", "dt"} -- dR/dpsi and dt/dpsi at (path pe, node), duplicates to be summed
    by the caller -- and "dR_dLn_a", "dR_dLn_b", "dR_dLp" per path."""
    x, psi = np.asarray(x, float), np.asarray(psi, float)
    Ln, Lp = np.asarray(Ln, float), np.asarray(Lp, float)
    Eg_V = phys.Eg_J / Q_E
    vstar = phys.v_star if phys.process == "phonon" else None
    paths = trace_paths_1d(x, psi, Eg_V, max_length_m, start_ok, end_ok, v_star=vstar)
    P = paths.start.size
    out = {"paths": paths, "R": np.zeros(P), "R_pre": np.zeros(P), "pair": np.zeros(P),
           "t": np.zeros(P), "length": np.zeros(P)}
    if P == 0:
        if derivatives:
            e = np.zeros(0)
            out.update(jac={"pe": e.astype(np.int64), "node": e.astype(np.int64), "dR": e, "dt": e},
                       dR_dLn_a=e, dR_dLn_b=e, dR_dLp=e)
        return out
    sp, i = paths.seg_path, paths.start
    i_s = i[sp]
    va = (psi[paths.seg_ka] - psi[i_s]) / Eg_V
    vb = (psi[paths.seg_kb] - psi[i_s]) / Eg_V
    va[paths.seg_first] = 0.0                                  # identically (not a variable there)
    g, wm, w0, wp = nodal_gradient_1d(x, psi)
    dirs = paths.direction[sp]
    ra = Eg_V / (dirs * g[paths.seg_ka])                       # ds/dv at the nodes (> 0: walker)
    rb = Eg_V / (dirs * g[paths.seg_kb])
    args = (paths.seg_L, paths.seg_end, paths.seg_phase)
    acc = _segment_accumulators(phys, va, vb, ra, rb, *args)
    tot = {k: np.bincount(sp, weights=v, minlength=P) for k, v in acc.items() if k != "t"}
    t = acc["t"][paths.seg_end]                                # one end segment per path, in path order
    length = tot["len"]
    G = np.abs(g[i])
    if phys.process == "direct":
        K0 = phys.K0
        Q = Q_E * G * phys.Cd0 * K0 / tot["I0"] * np.exp(-2.0 * K0 * tot["I1"])
        dlnQ = {"I0": -1.0 / tot["I0"], "I1": np.full(P, -2.0 * K0), "len": np.zeros(P)}
    else:
        KV, KC = phys.KV, phys.KC
        Q = (Q_E * G * phys.Kc * np.sqrt(length) * KV * KC / (tot["IV"] * tot["IC"])
             * np.exp(-2.0 * (KV * tot["JV"] + KC * tot["JC"])))
        dlnQ = {"IV": -1.0 / tot["IV"], "IC": -1.0 / tot["IC"], "JV": np.full(P, -2.0 * KV),
                "JC": np.full(P, -2.0 * KC), "len": 0.5 / length}
    w, dw = _taper(length, max_length_m)
    R_pre = Q * w
    za = Ln[paths.a] * (1.0 - t) + Ln[paths.b] * t + delta    # f_n(l) = sigma(za)
    zb = -Lp[i]                                                # f_p(0) = sigma(zb)
    dfo = np.sinh(0.5 * (za - zb)) / (2.0 * np.cosh(0.5 * za) * np.cosh(0.5 * zb))
    sa, sb = _sigmoid(za), _sigmoid(zb)
    sa_, sb_ = _sigmoid(-za), _sigmoid(-zb)
    out.update(R=R_pre * dfo, R_pre=R_pre, pair=sa * sb_ + sb * sa_, t=t, length=length)
    if not derivatives:
        return out
    # --- d/d(accumulator) of R ---
    dR_dacc = {k: dfo * (R_pre * dlnQ[k] + (Q * dw if k == "len" else 0.0)) for k in dlnQ}
    dR_dacc["t"] = R_pre * sa * sa_ * (Ln[paths.b] - Ln[paths.a])
    # --- complex step through every segment closed form (section 4) ---
    h = _CS_H
    dv_a, dv_b, dr_a, dr_b = {}, {}, {}, {}
    vac = va + 1j * h * (~paths.seg_first)
    z = 0j
    for k, v in _segment_accumulators(phys, vac, vb + z, ra + z, rb + z, *args).items():
        dv_a[k] = v.imag / h
    for k, v in _segment_accumulators(phys, va + z, vb + 1j * h, ra + z, rb + z, *args).items():
        dv_b[k] = v.imag / h
    for k, v in _segment_accumulators(phys, va + z, vb + z, ra + 1j * h, rb + z, *args).items():
        dr_a[k] = v.imag / h
    for k, v in _segment_accumulators(phys, va + z, vb + z, ra + z, rb + 1j * h, *args).items():
        dr_b[k] = v.imag / h

    def contract(dd, keys):
        return sum(dR_dacc[k][sp] * dd[k] for k in keys)
    keys = list(dR_dacc)
    dRa, dRb = contract(dv_a, keys), contract(dv_b, keys)
    dRra, dRrb = contract(dr_a, keys), contract(dr_b, keys)
    dTa, dTb = dv_a["t"], dv_b["t"]

    def per_node(sa_seg, sb_seg):
        """d/d(value) at each segment's upper node (the path nodes k_1..k_m): its b-derivative plus the
        a-derivative of the next segment of the same path."""
        o = sb_seg.copy()
        o[:-1] += np.where(paths.seg_first[1:], 0.0, sa_seg[1:])
        return o
    dvR, dvT = per_node(dRa, dRb), per_node(dTa, dTb)
    drR = per_node(dRra, dRrb)                                 # dR/dr at path nodes k_1..k_m
    drR0 = dRra[paths.seg_first]                               # dR/dr at the start node (per path)
    dRdG = dfo * (R_pre / G)                                   # R is proportional to |grad E_V|
    sg = np.sign(g[i])
    # r_k = Eg_V/(d g_k): dr_k/dpsi_j = -r_k^2 d (dg_k/dpsi_j)/Eg_V, g_k linear in psi_{k-1}, psi_k, psi_{k+1}
    kk = np.concatenate([paths.seg_kb, i])
    rk = np.concatenate([rb, Eg_V / (paths.direction * g[i])])
    dk = np.concatenate([dirs, paths.direction])
    cr = np.concatenate([drR, drR0]) * (-rk * rk * dk / Eg_V)
    pk = np.concatenate([sp, np.arange(P)])
    pe = np.concatenate([sp, np.arange(P), np.arange(P), np.arange(P), pk, pk, pk])
    node = np.concatenate([paths.seg_kb, i, i - 1, i + 1, kk - 1, kk, kk + 1])
    dR = np.concatenate([dvR / Eg_V,
                         -np.bincount(sp, weights=dvR, minlength=P) / Eg_V + dRdG * sg * w0[i],
                         dRdG * sg * wm[i], dRdG * sg * wp[i],
                         cr * wm[kk], cr * w0[kk], cr * wp[kk]])
    dt = np.concatenate([dvT / Eg_V, -np.bincount(sp, weights=dvT, minlength=P) / Eg_V,
                         np.zeros(P), np.zeros(P), np.zeros(3 * kk.size)])
    keep = (node >= 0) & (node < x.size)                       # one-sided ends have a zero weight outside
    pe, node, dR, dt = pe[keep], node[keep], dR[keep], dt[keep]
    s_n = R_pre * sa * sa_
    out.update(jac={"pe": pe, "node": node, "dR": dR, "dt": dt},
               dR_dLn_a=s_n * (1.0 - t), dR_dLn_b=s_n * t, dR_dLp=R_pre * sb * sb_)
    return out


# =====================================================================================================
# the DEVSIM custom term
# =====================================================================================================

def _ct():
    from dynameta.carriers import custom_terms
    return custom_terms


_TERM_CLASS = None


def _term_class():
    """The CustomTerm subclass, built on first use: custom_terms is imported lazily, so importing this
    module never touches DEVSIM."""
    global _TERM_CLASS
    if _TERM_CLASS is not None:
        return _TERM_CLASS
    CT = _ct()

    class NonlocalBTBTTerm(CT.CustomTerm):
        """The nonlocal BTBT custom term of ONE region (module doc). Rows: the region's two continuity
        equations; jacobian_exact (1-D: section 4); a noise model (section 5)."""
        locality = "nonlocal"
        jacobian_exact = True
        pattern_changes = True

        def __init__(self, region: str, spec: NonlocalBTBT, name: str = MECHANISM, T_K: Optional[float] = None):
            super().__init__(name)
            self.region = region
            self.spec = spec
            self.phys = physical_from_fitted(spec, T_K=T_K)   # the region's temperature, fixed at setup
            self.last_stats: Dict[str, int] = {}

        def equations(self, device):
            return [(self.region, _EQ_N), (self.region, _EQ_P)]

        def state(self, device: str) -> dict:
            """Node arrays in x order (perm maps sorted position -> region node index)."""
            from dynameta.carriers.eq_registry import ds
            reg = self.region

            def nv(name):
                return np.asarray(ds.get_node_model_values(device=device, region=reg, name=name), float)

            def par(name):
                return float(ds.get_parameter(device=device, region=reg, name=name))
            x = nv("x")
            perm = np.argsort(x, kind="stable")
            vt = par("V_t")
            ok = np.ones(x.size, bool)
            ok[sorted(CT._contact_node_set(device, reg))] = False
            end_ok = ok[perm]
            start_ok = end_ok.copy()
            start_ok[0] = start_ok[-1] = False
            delta = math.log(par("N_dos") * par("N_dos_p") / par("n_i") ** 2) - self.phys.Eg_J / Q_E / vt
            return {"perm": perm, "x": x[perm], "psi": nv("Potential")[perm],
                    "Ln": nv("HJ_L_Electrons")[perm], "Lp": nv("HJ_L_Holes")[perm],
                    "dLn": nv("HJ_L_Electrons:Electrons")[perm], "dLp": nv("HJ_L_Holes:Holes")[perm],
                    "vol": nv("NodeVolume")[perm], "delta": delta, "start_ok": start_ok, "end_ok": end_ok}

        def evaluate(self, device: str, derivatives: bool = False):
            st = self.state(device)
            res = path_rates_1d(st["x"], st["psi"], st["Ln"], st["Lp"], self.phys, st["delta"],
                                self.spec.max_length_m, st["start_ok"], st["end_ok"],
                                derivatives=derivatives)
            self.last_stats = dict(res["paths"].stats)
            return st, res

        def assemble(self, device, what, timemode):
            if timemode != "DC":
                return [], []                                  # static term: no charge part
            need_j = what in ("MATRIXONLY", "MATRIXANDRHS")
            st, res = self.evaluate(device, derivatives=need_j)
            p = res["paths"]
            if p.start.size == 0:
                return [], []
            perm, reg = st["perm"], self.region
            i, a, b, t, R = p.start, p.a, p.b, res["t"], res["R"]
            qv = Q_E * st["vol"][i]
            rp = CT.node_rows(device, reg, _EQ_P, perm[i])
            ra = CT.node_rows(device, reg, _EQ_N, perm[a])
            rb = CT.node_rows(device, reg, _EQ_N, perm[b])
            rv = []
            for r_, v_ in ((rp, qv * R), (ra, -qv * (1.0 - t) * R), (rb, -qv * t * R)):
                rv += [x_ for pair in zip(r_.tolist(), v_.tolist()) for x_ in pair]
            if not need_j:
                return [], rv
            jac = res["jac"]
            pe, nd, dR, dt = jac["pe"], jac["node"], jac["dR"], jac["dt"]
            cpsi = CT.node_columns(device, reg, "Potential", perm[nd])
            blocks = [(rp[pe], cpsi, qv[pe] * dR),
                      (ra[pe], cpsi, -qv[pe] * ((1.0 - t[pe]) * dR - R[pe] * dt)),
                      (rb[pe], cpsi, -qv[pe] * (t[pe] * dR + R[pe] * dt))]
            cna = CT.node_columns(device, reg, "Electrons", perm[a])
            cnb = CT.node_columns(device, reg, "Electrons", perm[b])
            cph = CT.node_columns(device, reg, "Holes", perm[i])
            for col, dval in ((cna, res["dR_dLn_a"] * st["dLn"][a]),
                              (cnb, res["dR_dLn_b"] * st["dLn"][b]),
                              (cph, res["dR_dLp"] * st["dLp"][i])):
                blocks += [(rp, col, qv * dval), (ra, col, -qv * (1.0 - t) * dval),
                           (rb, col, -qv * t * dval)]
            rcv = []
            for r_, c_, v_ in blocks:
                rcv += [x_ for trip in zip(r_.tolist(), c_.tolist(), v_.tolist()) for x_ in trip]
            return rcv, rv

        def noise_sources(self, device):
            """One TwoPointSource over every path (module doc, section 5)."""
            from dynameta.carriers.noise_analysis import SourceRow, TwoPointSource
            st, res = self.evaluate(device, derivatives=False)
            p = res["paths"]
            perm, reg, t = st["perm"], self.region, res["t"]
            rate = st["vol"][p.start] * res["R_pre"] * res["pair"]
            ni, na_, nb_ = perm[p.start], perm[p.a], perm[p.b]
            return [TwoPointSource(MECHANISM, reg, rate,
                                   (SourceRow(reg, _EQ_P, np.ones(t.size), ni),
                                    SourceRow(reg, _EQ_N, -(1.0 - t), na_),
                                    SourceRow(reg, _EQ_N, -t, nb_)), ni)]

    _TERM_CLASS = NonlocalBTBTTerm
    return _TERM_CLASS


# =====================================================================================================
# public setup / post-processing
# =====================================================================================================

def setup_btbt_nonlocal(device: str, region: str, spec: NonlocalBTBT, *, name: str = MECHANISM):
    """Install the nonlocal path BTBT term (module doc) on an EXISTING 1-D bipolar region (call after
    physics_bipolar_dd.setup_bipolar_region and the contacts). Returns the custom term (its last_stats
    count the paths of the latest assembly). Refuses: a non-1-D device, a region without both continuity
    equations, the local BTBT term in the same region (double counting), bandgap narrowing (BgnDc /
    n_ie in the region) and statistics='fermi_dirac'. Warns (AdvisoryWarning) at T != T_REF.

    WHICH OPTION FOR WHICH CALCULATION. The process is P of spec.params: P = 2.5 (phonon-assisted, the
    Si/Ge indirect gap; tunneling.SILICON_HURKX is Sentaurus' default set of this model) or P = 2
    (direct gap: III-V, the Ge direct process). The gap: material= (fundamental gap from the package gap
    kernel, temperature-aware with a band-temperature model) or Eg_tun_eV= (explicit; the Gamma gap of an
    indirect material's direct process) -- NonlocalBTBT docstring. max_length_m: longer than the longest path that matters (the 20 nm default covers average
    fields above E_g/20 nm = 5.6e7 V/m in Si, where exp(-B/F) < 2e-15). Use the LOCAL model
    (tunneling.setup_btbt) instead for 2-D/3-D, for regions with bandgap narrowing, and where speed
    matters more than the nonlocal effects of the module doc, section 1."""
    from dynameta.carriers.eq_registry import ds
    from dynameta.carriers.tunneling import _require_bipolar
    if not isinstance(spec, NonlocalBTBT):
        raise TypeError("setup_btbt_nonlocal: spec must be a NonlocalBTBT (got {!r})".format(type(spec)))
    dim = int(ds.get_dimension(device=device))
    if dim != 1:
        raise NotImplementedError(
            "setup_btbt_nonlocal: {}-D devices are not implemented in v0.15 (decision 21: 1-D first; the "
            "2-D element walker follows the review of the 1-D results). Use the local model "
            "(tunneling.setup_btbt) in 2-D/3-D.".format(dim))
    _require_bipolar(device, region)
    nodes = set(ds.get_node_model_list(device=device, region=region))
    edges = set(ds.get_edge_model_list(device=device, region=region))
    elems = set(ds.get_element_model_list(device=device, region=region)) if dim > 1 else set()
    if any(m.startswith("BTBT_U") for m in nodes | edges | elems):
        raise BTBTDoubleCountError(
            "setup_btbt_nonlocal: region {!r} already carries the LOCAL BTBT term (setup_btbt / btbt=...); "
            "both would count the same tunnelling twice.".format(region))
    if "BgnDc" in nodes or "n_ie" in nodes:
        raise NotImplementedError(
            "setup_btbt_nonlocal: region {!r} has bandgap narrowing / an effective n_ie; the gap would "
            "vary along the path (the closed-form segment integrals assume a constant gap). Not "
            "implemented in v0.15.".format(region))
    if "fd_detailed_balance" in set(ds.get_parameter_list(device=device, region=region)):
        raise NotImplementedError("setup_btbt_nonlocal: statistics='fermi_dirac' (solution-dependent "
                                  "n_ie) is not implemented with the nonlocal model.")
    from dynameta.carriers.physics_heterojunction import temperature_of_V_t
    T = temperature_of_V_t(float(ds.get_parameter(device=device, region=region, name="V_t")))
    eg_set, eg_path, handled = tunnelling_gaps(spec, T)
    if not handled:
        warnings.warn(
            "setup_btbt_nonlocal: region {!r} runs at T = {:.6g} K but the BTBT set is quoted at {:.6g} K and "
            "the tunnelling gap has no temperature law ({}): A, B and the gap {:.6g} eV are used as given. Give "
            "material= a TransportModel with a band-temperature model (materials.BandTemperature) to follow "
            "E_g(T), or supply A, B and the gap at this temperature.".format(
                region, T, spec.params.quoted_T_K,
                "explicit Eg_tun_eV" if spec.material is None else "the material has no band_temperature",
                eg_path), AdvisoryWarning, stacklevel=2)
    from dynameta.carriers.physics_heterojunction import setup_log_activity
    fd = float(ds.get_parameter(device=device, region=region, name="fd_enhancement")) != 0.0
    setup_log_activity(device, region, "Electrons", "N_dos", fd=fd)
    setup_log_activity(device, region, "Holes", "N_dos_p", fd=fd)
    term = _term_class()(region, spec, name, T_K=T)
    _ct().add_custom_term(device, term)
    return term


def btbt_nonlocal_generation(device: str, region: str, name: str = MECHANISM) -> dict:
    """The nonlocal term's generation at the current state (post-processing; nothing is changed):
    {"G_p", "G_n": per-node GENERATION [1/(m^2 s) in 1-D: rate x NodeVolume of the start node], indexed
    like the region's nodes; "I_p", "I_n": q Sum G [A/m^2] (equal by construction; > 0 = net
    generation); "R": the per-path net rate density; "stats": the path counts; "paths"}."""
    term = next((t for t in _ct().live_terms(device) if t.name == name and
                 getattr(t, "region", None) == region), None)
    if term is None:
        raise ValueError("btbt_nonlocal_generation: no live nonlocal BTBT term {!r} on {!r}/{!r}".format(
            name, device, region))
    st, res = term.evaluate(device, derivatives=False)
    p, perm = res["paths"], st["perm"]
    amt = -res["R"] * st["vol"][p.start]
    Gp = np.zeros(perm.size)
    Gn = np.zeros(perm.size)
    np.add.at(Gp, perm[p.start], amt)
    np.add.at(Gn, perm[p.a], (1.0 - res["t"]) * amt)
    np.add.at(Gn, perm[p.b], res["t"] * amt)
    return {"G_p": Gp, "G_n": Gn, "I_p": Q_E * float(np.sum(Gp)), "I_n": Q_E * float(np.sum(Gn)),
            "R": res["R"], "stats": dict(p.stats), "paths": p}
