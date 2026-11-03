"""
Semiconductor/semiconductor HETEROINTERFACES for the DEVSIM carrier layer (v0.14.0, workstream D1a):
Anderson band alignment, thermionic-emission (TE) carrier transport across the interface, and the
continuous-quasi-Fermi-level (drift-diffusion) limit -- ONE implementation shared by the 2-D
LayeredDevsimBuilder and the 3-D Devsim3DEquilibrium. Design note + measured gates:
docs/audit/2026-11-03-devsim-heterojunctions.md.

FORMULATION (literature, not improvised)
----------------------------------------
1. Band alignment -- Anderson's electron-affinity rule (R. L. Anderson, Solid-State Electron. 5, 341
   (1962)). The electrostatic potential psi is CONTINUOUS across the interface (it is the vacuum level
   up to one constant, -q psi = E_vac + const), and every band edge hangs off the vacuum level by the
   material's affinity: E_c(r) = E_vac - chi_r. In this library's frame (E_F = 0 at zero bias)

       E_c(r) = q (Phi_c(r) - psi),     Phi_c(r) = Phi_c,own(ref) + (chi_ref - chi_r) / q,
       E_v(r) = q (Phi_v(r) - psi),     Phi_v(r) = Phi_c(r) - E_g,eff(r) / q,

   so the offsets are Delta E_c = chi_0 - chi_1 and Delta E_v = Delta E_c + (E_g,1 - E_g,0) (sign per
   side). Phi_c,own(r) is the conduction edge each physics module ALREADY implies in its own legacy
   frame (below); the region's whole internal model is shifted by the constant

       s(r) = Phi_c(r) - Phi_c,own(r)                                                   (frame shift)

   which is the region's neutral-bulk potential in the common frame: an ohmic contact on region r pins
   Potential = bias + s(r) (+ the bipolar built-in offset), and a seed starts there. s = 0 for the
   reference region, so a single-semiconductor device -- and every region of a same-material stack in
   the bipolar/drift-diffusion frames -- is unchanged byte-for-byte. Potential and the D-field are
   continuous (DEVSIM 'continuous' interface equation on PotentialEquation: the two region rows are
   summed, i.e. no interface charge, and Potential@r0 = Potential@r1 replaces one of them).

   Phi_c,own per physics (all in volts, at T_REF):
     equilibrium (n = N_c F_1/2((psi - Phi_c0)/V_t))      : Phi_c0 = -eta_bg V_t (Aymerich-Humet inverse)
     unipolar DD (contact pins n = N_D at psi = bias)     : -V_t [ln(N_D/N_c) + Delta(N_D/N_c)]
     bipolar DD (psi = intrinsic level, contact pins
                 psi = V_t[ln(n0/n_i) + Delta])           : +V_t ln(N_c/n_i);  E_g,eff = kT ln(N_c N_v/n_i^2)
   Delta(u) = int_0^u (g(x) - 1) dx / x is the Fermi-Dirac built-in correction of the FD-enhanced
   Scharfetter-Gummel bulk (physics_bipolar_dd.fd_potential_shift; 0 in the Boltzmann limit).

2. Carrier transport across the interface -- thermionic emission as an interface BOUNDARY CONDITION in
   series with bulk drift-diffusion (K. Horio & H. Yanai, IEEE Trans. Electron Devices 37, 1093 (1990);
   K. Yang, J. R. East & G. I. Haddad, Solid-State Electron. 36, 321 (1993), whose thermionic-field
   emission condition reduces to this one at zero tunnelling; D. Schroeder, "Modelling of Interface
   Carrier Transport for Device Simulation", Springer 1994, ch. 5). The net electron particle flux
   from side 0 to side 1 over the conduction-band step is the difference of the two Richardson fluxes
   over the barrier top E_top = max(E_c,0, E_c,1) at the interface:

       F_n(0->1) = (A*_n T^2 / q) [ exp((E_Fn,0 - E_top)/kT) - exp((E_Fn,1 - E_top)/kT) ]
                 = (A*_n T^2 / q) exp(-(E_top - E_Fn,1)/kT) [ exp((E_Fn,0 - E_Fn,1)/kT) - 1 ]

   -- the Horio-Yanai quasi-Fermi-level form. With Boltzmann statistics (E_Fn - E_c = kT ln(n/N_c)) it
   is exactly the textbook interface condition J = q v_n [n_lo exp(-Delta E_c/kT) N_c,hi/N_c,lo - n_hi]
   with the Richardson (emission) velocity v_n = A*_n T^2/(q N_c,hi) = sqrt(kT/(2 pi m*_hi)), i.e.
   the Sentaurus / Quay (TU Wien thesis 2001, eq. 3.88) form J = q [v_2 n_2 - (m_2/m_1) v_1 n_1 exp(-
   Delta E/kT)] after v_1 (m_2/m_1) = v_2 (m_2/m_1)^(3/2). A*_n = 4 pi q m* k^2 / h^3 is the effective
   Richardson constant of the BARRIER side (the side holding E_top -- the effective mass of the region
   containing the barrier maximum sets A*: Yang/East/Haddad; Horio/Yanai), taken from that side's DOS
   mass unless overridden. Holes mirror it on the valence edge (E_bot = min(E_v,0, E_v,1), hole energy
   counted downward). Zero bias => E_Fn,0 = E_Fn,1 => ZERO flux, for ANY band offset: detailed balance
   holds by construction because the flux is written in the model's own quasi-Fermi levels.

   Quasi-Fermi level from the SOLUTION variable. The bulk transport is FD-enhanced Scharfetter-Gummel,
   whose zero-current limit is d psi = g V_t d ln n, so the quasi-Fermi level it implies is

       (E_Fn - E_c)/kT = L_n(n) = ln(u) + Delta(u),   u = n / N_c,   dL_n/dn = g(u)/n,

   (Boltzmann: Delta = 0, g = 1). Using EXACTLY this L at the interface is what makes the interface
   consistent with the bulk: any other activity (exact F_1/2^-1, or plain ln u on a degenerate side)
   acts as a spurious interface battery that drives a current at zero bias. Delta has a CLOSED FORM for
   the rational g fit (polynomial + log + arctan; carriers.einstein coefficients), evaluated here as a
   DEVSIM node model; DEVSIM has no atan(), so arctan is four half-angle reductions
   z -> z / (1 + sqrt(1 + z^2)) followed by a 7-term Taylor series (|error| < 2e-15 for every z >= 0).
   MEASURED vs the Gauss-Legendre quadrature physics_bipolar_dd.fd_potential_shift over u in
   [1e-8, 1e5]: max |Delta_closed - Delta_quad| = 2.7e-12 (at u = 1e5, Delta = 2342).

3. Continuous quasi-Fermi level (the drift-diffusion limit, Horio-Yanai's "A* -> infinity"): the
   interface carries no resistance and E_Fn (E_Fp) is continuous,

       L_n,0 - (E_top - E_c,0)/kT = L_n,1 - (E_top - E_c,1)/kT,

   imposed as a DEVSIM 'continuous' interface equation (the region continuity rows are summed --
   current continuity -- and this algebraic relation replaces one). It is the A* -> infinity limit of
   (2), the right model when the interface is not a barrier (a same-material split: the 'auto' default
   picks it there), and the comparison model of oracle (e).

DEVSIM mechanism (native): every quantity is an `interface_model` on the region node copies @r0/@r1
with explicit derivative models; TE enters the continuity equations as a `fluxterm` interface equation
(the integrated interface flux enters the two regions' equations with opposite signs; the model is
written as the CONVENTIONAL current density from r0 into r1, the edge-model convention -- MEASURED, not
assumed: on a zero-offset 1-D split it reproduces the analytic series interface resistance and with
A* x 1e6 the continuous model to 1e-6; gates C/E pin it on the builder path), the continuous model as
a `continuous` one. Both are recorded through eq_registry under the continuity-equation names, so the
Gummel / staged freeze-and-reapply machinery sees them.

LIMITS (stated, not hidden): the TE MAGNITUDE uses the Maxwell-Boltzmann Richardson tail -- on a
DEGENERATE emitting side (E_Fn above E_top) the true Fermi-Dirac emission ~ ln(1 + exp(.)) is smaller
than the exp(.) used here; the zero-bias equilibrium is exact either way (it is a quasi-Fermi-level
form). No tunnelling / thermionic-field emission (workstream D1d), no interface traps or dipoles (D1c).
v0.14 INTEGRATION: bandgap narrowing (D1b) composes as an effective band-edge field (Lundstrom &
Schuelke 1983): the interface reads E_F against E_c - q D_c / E_v + q D_v of each node copy
(_bgn_interface_arguments); the contacts compose s(r) with the region's own-frame equilibrium potential
in ONE rule (physics_bipolar_dd.contact_potential_residual). A region at T_K != T_REF on a heterointerface
with BGN is refused (the interface is written at T_REF). The TE fluxterm has no noise source in
noise_analysis (documented limit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from dynameta.carriers import eq_registry as _R
from dynameta.carriers.eq_registry import ds
from dynameta.carriers.einstein import GA, GB, GC, GD, g_expr_devsim
from dynameta.constants import Q_E, KB, HBAR, M_E, T_REF, V_T

__all__ = ["richardson_constant_Am2K2", "A_RICHARDSON_FREE", "HeteroSide", "anderson_alignment",
           "setup_log_activity", "setup_heterointerface_carriers", "interface_models",
           "HETERO_MODELS", "delta_closed_form_coefficients"]

# Interface carrier models: thermionic emission (Horio-Yanai / Yang-East-Haddad) or the continuous
# quasi-Fermi-level drift-diffusion limit.
HETERO_MODELS = ("thermionic", "continuous")


def richardson_constant_Am2K2(m_kg: float) -> float:
    """Effective Richardson constant A* = 4 pi q m* k^2 / h^3 [A m^-2 K^-2] for an (isotropic,
    single-valley) DOS mass m*. h = 2 pi HBAR -- the SAME Planck constant effective_dos_m3 uses, so the
    emission velocity A* T^2/(q N_c) equals sqrt(kT/(2 pi m*)) to round-off (the identity the TE flux
    relies on). Free-electron value A_RICHARDSON_FREE = 1.20173e6 A m^-2 K^-2 (120.17 A cm^-2 K^-2)."""
    m = float(m_kg)
    if not (math.isfinite(m) and m > 0.0):
        raise ValueError("richardson_constant_Am2K2: m_kg must be finite and > 0, got {!r}".format(m_kg))
    h = 2.0 * math.pi * HBAR
    return 4.0 * math.pi * Q_E * m * KB ** 2 / h ** 3


A_RICHARDSON_FREE = richardson_constant_Am2K2(M_E)


# ---- the closed-form Fermi-Dirac built-in correction Delta(u) of the rational g fit -----------------
#   Delta(u) = int_0^{s} 3 t^2 (a + c t) / (1 + b t + d t^2) dt,  s = u^(1/3)
#            = K1 s^2 + K2 s + KLOG ln(1 + b s + d s^2) + KATAN arctan(D s / (2 + b s)),  D = sqrt(4d - b^2)
# (polynomial division of the integrand; the two arctans of the partial fraction merge by
# atan(A) - atan(B) = atan((A - B)/(1 + AB)), valid since A, B > 0). b^2 < 4d for the audited fit, so
# the denominator has complex roots and the arctan term is real.
_D = math.sqrt(4.0 * GD - GB * GB)
_K1 = 3.0 * GC / (2.0 * GD)
_K2 = (3.0 * GA - 3.0 * GC * GB / GD) / GD
_ALPHA = -(3.0 * GC / GD) - _K2 * GB
_BETA = -_K2
_KLOG = _ALPHA / (2.0 * GD)
_KATAN = (_BETA - _ALPHA * GB / (2.0 * GD)) * 2.0 / _D
_ATAN_HALVINGS = 4          # z_max = D/b = 6.72 -> 0.086 after four reductions
_ATAN_TERMS = 7             # Taylor to z^13: |err| < 2^4 * 0.086^15/15 ~ 1e-16


def delta_closed_form_coefficients() -> Dict[str, float]:
    """The closed-form Delta(u) coefficients (exposed for the oracle tests)."""
    return {"D": _D, "K1": _K1, "K2": _K2, "KLOG": _KLOG, "KATAN": _KATAN}


@dataclass(frozen=True)
class HeteroSide:
    """One side of a semiconductor/semiconductor interface, in the common (Anderson-aligned) frame.

    phi_c_V / phi_v_V : E_c = q (phi_c - psi), E_v = q (phi_v - psi) [V]; phi_v None -> no holes.
    n_dos_param / p_dos_param : the DEVSIM region PARAMETER holding that side's band DOS (the
        normalisation of the activity L = ln(c/N) + Delta): 'N_c' (unipolar DD) or 'N_dos'/'N_dos_p'
        (bipolar), so the interface reads exactly the N the bulk g-factor reads.
    richardson_n / richardson_p : A* [A m^-2 K^-2] of this side (used when this side holds the barrier).
    fd : the bulk transport uses the Fermi-Dirac g-factor (then L carries Delta; else L = ln(c/N)).
    """
    region: str
    phi_c_V: float
    n_dos_param: str
    richardson_n: float
    fd: bool = True
    phi_v_V: Optional[float] = None
    p_dos_param: Optional[str] = None
    richardson_p: Optional[float] = None

    @property
    def holes(self) -> bool:
        return self.phi_v_V is not None


def anderson_alignment(own_phi_c_V: Mapping[str, float], chi_eV: Mapping[str, Optional[float]],
                       ref: str) -> Dict[str, Tuple[float, float]]:
    """Anderson electron-affinity rule: {region: (Phi_c_V, s_V)} in the frame of `ref`.

    own_phi_c_V[r] is the conduction edge region r's physics implies in its OWN legacy frame (see the
    module docstring), chi_eV[r] its electron affinity. Phi_c(r) = Phi_c,own(ref) + (chi_ref - chi_r)
    and the frame shift s(r) = Phi_c(r) - Phi_c,own(r) (the neutral-bulk potential of r). The
    reference keeps (Phi_c,own(ref), 0.0). THE single implementation: Stacked3DSpec.band_alignment
    (3-D) and LayeredDevsimBuilder (2-D) both call it; the operation order (phi_ref + d_chi, then
    phi_c - own) is the one band_alignment shipped with in v0.13.0, so its numbers are bit-identical."""
    phi_ref = float(own_phi_c_V[ref])
    out = {ref: (phi_ref, 0.0)}
    for r, own in own_phi_c_V.items():
        if r == ref:
            continue
        cr, c0 = chi_eV.get(r), chi_eV.get(ref)
        if cr is None or c0 is None:
            raise ValueError("anderson_alignment: region {!r} needs both electron affinities (its own and "
                             "the reference {!r}'s); there is no neutral default for a band offset."
                             .format(r, ref))
        d_chi = float(c0) - float(cr)
        phi_c = phi_ref + d_chi
        out[r] = (phi_c, phi_c - float(own))
    return out


# ---- DEVSIM node models: the bulk-consistent log-activity L = (E_F - E_band)/kT of one carrier --------
def _atan_halving_models(device: str, region: str, pfx: str, z_expr: str) -> str:
    """Node models for arctan(z) via _ATAN_HALVINGS half-angle reductions + a Taylor tail; returns the
    name of the model holding arctan(z). Separate node models keep DEVSIM's expression trees small (no
    derivatives are taken of these -- the activity's derivative is the closed form g/c)."""
    ds.node_model(device=device, region=region, name=pfx + "z0", equation=z_expr)
    for k in range(_ATAN_HALVINGS):
        zk = "{}z{}".format(pfx, k)
        ds.node_model(device=device, region=region, name="{}z{}".format(pfx, k + 1),
                      equation="{z}/(1.0 + pow(1.0 + {z}*{z}, 0.5))".format(z=zk))
    zl = "{}z{}".format(pfx, _ATAN_HALVINGS)
    terms = " + ".join("({:+.17g})*pow({}, {})".format((-1.0) ** k / (2 * k + 1), zl, 2 * k + 1)
                       for k in range(_ATAN_TERMS))
    ds.node_model(device=device, region=region, name=pfx + "atan",
                  equation="{:.17g}*({})".format(2.0 ** _ATAN_HALVINGS, terms))
    return pfx + "atan"


def setup_log_activity(device: str, region: str, carrier: str, dos_param: str, *, fd: bool) -> str:
    """Install (idempotently) the node model `HJ_L_<carrier>` = ln(u) + Delta(u), u = carrier/dos_param,
    and its derivative `HJ_L_<carrier>:<carrier>` = g(u)/carrier (exact: d/du [ln u + Delta] = g/u).
    fd=False -> the Boltzmann activity ln(u), derivative 1/carrier. Returns the model name."""
    name = "HJ_L_{}".format(carrier)
    if name in ds.get_node_model_list(device=device, region=region):
        return name
    u = "({}/{})".format(carrier, dos_param)
    if fd:
        pfx = "HJ_{}_".format(carrier)
        s = pfx + "s"
        ds.node_model(device=device, region=region, name=s, equation="pow({}, {!r})".format(u, 1.0 / 3.0))
        at = _atan_halving_models(device, region, pfx,
                                  "({D!r}*{s})/(2.0 + {b!r}*{s})".format(D=_D, s=s, b=GB))
        delta = ("{k1!r}*{s}*{s} + {k2!r}*{s} + {kl!r}*log(1.0 + {b!r}*{s} + {d!r}*{s}*{s}) + {ka!r}*{at}"
                 .format(k1=_K1, k2=_K2, kl=_KLOG, ka=_KATAN, b=GB, d=GD, s=s, at=at))
        ds.node_model(device=device, region=region, name=pfx + "Delta", equation=delta)
        ds.node_model(device=device, region=region, name=name,
                      equation="log({}) + {}Delta".format(u, pfx))
        dexpr = "{}/{}".format(g_expr_devsim(carrier, dos_param, ""), carrier)
    else:
        ds.node_model(device=device, region=region, name=name, equation="log({})".format(u))
        dexpr = "1.0/{}".format(carrier)
    ds.node_model(device=device, region=region, name="{}:{}".format(name, carrier), equation=dexpr)
    return name


def interface_models(side0: HeteroSide, side1: HeteroSide, carrier: str) -> Tuple[float, float, float]:
    """(k0, k1, A*T^2) for `carrier` in {'Electrons', 'Holes'}: k_i = (barrier top - band edge_i)/kT >= 0
    (one of them is 0) and the Richardson prefactor of the BARRIER side (the side holding the top; on a
    zero offset, the smaller of the two -- the conservative choice, and symmetric)."""
    if carrier == "Electrons":
        e0, e1 = side0.phi_c_V, side1.phi_c_V                     # E_c / q + psi
        top = max(e0, e1)
        k0, k1 = (top - e0) / V_T, (top - e1) / V_T
        a0, a1 = side0.richardson_n, side1.richardson_n
        a = a0 if e0 > e1 else (a1 if e1 > e0 else min(a0, a1))
    elif carrier == "Holes":
        e0, e1 = side0.phi_v_V, side1.phi_v_V                     # E_v / q + psi
        bot = min(e0, e1)                                          # hole barrier = LOWER valence edge
        k0, k1 = (e0 - bot) / V_T, (e1 - bot) / V_T
        a0, a1 = side0.richardson_p, side1.richardson_p
        a = a0 if e0 < e1 else (a1 if e1 < e0 else min(a0, a1))
    else:
        raise ValueError("carrier must be 'Electrons' or 'Holes'")
    return k0, k1, float(a) * T_REF ** 2


def _region_has(device: str, region: str, model: str) -> bool:
    return model in ds.get_node_model_list(device=device, region=region)


def _bgn_interface_arguments(device: str, side0: HeteroSide, side1: HeteroSide, carrier: str,
                             L0: str, L1: str):
    """(w0, w1) = (E_F,i - E_top)/kT on the two node copies when bandgap narrowing (D1b,
    bipolar_options) is on in either region, else (None, None).

    v0.14 integration -- the composition rule of Lundstrom & Schuelke (IEEE TED 30, 1151 (1983),
    "Numerical analysis of heterostructure semiconductor devices"): material band offsets (Anderson,
    the constant Phi_c/Phi_v of each side) and heavy-doping band-edge shifts (BGN) are ONE band-edge
    field. The BGN region's bulk statistics are written against the EFFECTIVE edges
    E_c,eff = E_c - q D_c, E_v,eff = E_v + q D_v (BgnDc / BgnDv node fields; vdiff_g / vdiff_gp carry
    them in the bulk), so the activity L = ln(c/N) + Delta of the interface node measures E_F against
    the effective edge, and the barrier top is the larger effective conduction edge (lower effective
    valence edge for holes) of the two node copies. Without this the interface would read E_F with an
    error D_c,i on each side: a spurious battery D_c,0 - D_c,1 at zero bias (gated by
    tests/test_devsim_phase_composition.py::test_heterojunction_with_bgn_*). The BGN fields are fixed,
    so the carrier derivatives are unchanged."""
    if carrier == "Electrons":
        fld, e0, e1, sgn = "BgnDc", side0.phi_c_V, side1.phi_c_V, "-"
    else:
        fld, e0, e1, sgn = "BgnDv", side0.phi_v_V, side1.phi_v_V, "+"
    has0, has1 = _region_has(device, side0.region, fld), _region_has(device, side1.region, fld)
    if not (has0 or has1):
        return None, None
    for sd in (side0, side1):
        vt = float(ds.get_parameter(device=device, region=sd.region, name="V_t"))
        if vt != V_T:
            raise NotImplementedError(
                "setup_heterointerface_carriers: region {!r} runs at V_t = {!r} (BipolarOptions.T_K) but "
                "the heterointerface model is written at T_REF (V_T = {!r}); a temperature-dependent "
                "interface (Richardson T^2, kT in the activities) is not implemented.".format(
                    sd.region, vt, V_T))
    E0 = "({!r} {} {}@r0)".format(e0, sgn, fld) if has0 else "({!r})".format(e0)
    E1 = "({!r} {} {}@r1)".format(e1, sgn, fld) if has1 else "({!r})".format(e1)
    if carrier == "Electrons":                               # barrier top = higher effective E_c
        top = "ifelse({a} > {b}, {a}, {b})".format(a=E0, b=E1)
        k0 = "({} - {})/{!r}".format(top, E0, V_T)
        k1 = "({} - {})/{!r}".format(top, E1, V_T)
    else:                                                    # hole barrier = lower effective E_v
        bot = "ifelse({a} < {b}, {a}, {b})".format(a=E0, b=E1)
        k0 = "({} - {})/{!r}".format(E0, bot, V_T)
        k1 = "({} - {})/{!r}".format(E1, bot, V_T)
    return "({}@r0 - {})".format(L0, k0), "({}@r1 - {})".format(L1, k1)


def setup_heterointerface_carriers(device: str, interface: str, side0: HeteroSide, side1: HeteroSide,
                                   *, model: str = "thermionic") -> None:
    """Wire the carrier-continuity coupling of one semiconductor/semiconductor interface (Potential is
    wired separately by physics_equilibrium.setup_interface -- continuous psi and D). Electrons always;
    Holes when BOTH sides carry them (a unipolar side next to a bipolar one is refused by the builders).

    model='thermionic': fluxterm J(0->1) = -/+ A*T^2 [exp(L_0 - k_0) - exp(L_1 - k_1)] (electrons /
    holes, conventional current), the Horio-Yanai/Yang-East-Haddad interface condition.
    model='continuous': (L_0 - k_0) - (L_1 - k_1) = 0, continuous quasi-Fermi level.
    side0/side1 MUST be region0/region1 of `interface` (checked against DEVSIM)."""
    if model not in HETERO_MODELS:
        raise ValueError("heterointerface model must be one of {}, got {!r}".format(HETERO_MODELS, model))
    regs = list(ds.get_region_list(device=device, interface=interface))
    if regs != [side0.region, side1.region]:
        raise ValueError("setup_heterointerface_carriers: interface {!r} joins {} (region0, region1) but "
                         "the sides were given as ({!r}, {!r})".format(interface, regs, side0.region,
                                                                     side1.region))
    carriers = [("Electrons", "ElectronContinuityEquation", "n", -1.0)]
    if side0.holes and side1.holes:
        carriers.append(("Holes", "HoleContinuityEquation", "p", +1.0))
    for var, eq, tag, qsign in carriers:
        dos0 = side0.n_dos_param if var == "Electrons" else side0.p_dos_param
        dos1 = side1.n_dos_param if var == "Electrons" else side1.p_dos_param
        L0 = setup_log_activity(device, side0.region, var, dos0, fd=side0.fd)
        L1 = setup_log_activity(device, side1.region, var, dos1, fd=side1.fd)
        k0, k1, at2 = interface_models(side0, side1, var)
        w0, w1 = _bgn_interface_arguments(device, side0, side1, var, L0, L1)
        if w0 is None:                                       # no BGN on either side: D1a's strings
            w0 = "({L}@r0 - {k!r})".format(L=L0, k=k0)
            w1 = "({L}@r1 - {k!r})".format(L=L1, k=k1)
        name = "{}_HJ{}".format(interface, tag)
        if model == "thermionic":
            pref = "{!r}".format(qsign * at2)
            ds.interface_model(device=device, interface=interface, name=name,
                               equation="{p}*(exp({w0}) - exp({w1}))".format(p=pref, w0=w0, w1=w1))
            ds.interface_model(device=device, interface=interface, name="{}:{}@r0".format(name, var),
                               equation="{p}*exp({w0})*{L}:{v}@r0".format(p=pref, w0=w0, L=L0, v=var))
            ds.interface_model(device=device, interface=interface, name="{}:{}@r1".format(name, var),
                               equation="-{p}*exp({w1})*{L}:{v}@r1".format(p=pref, w1=w1, L=L1, v=var))
            _R.record_interface_equation(device, interface, name=eq, interface_model=name,
                                         type="fluxterm")
        else:
            ds.interface_model(device=device, interface=interface, name=name,
                               equation="{w0} - {w1}".format(w0=w0, w1=w1))
            ds.interface_model(device=device, interface=interface, name="{}:{}@r0".format(name, var),
                               equation="{L}:{v}@r0".format(L=L0, v=var))
            ds.interface_model(device=device, interface=interface, name="{}:{}@r1".format(name, var),
                               equation="-{L}:{v}@r1".format(L=L1, v=var))
            _R.record_interface_equation(device, interface, name=eq, interface_model=name,
                                         type="continuous")
