"""Tunneling on DEVSIM devices (D1d, v0.14.0): band-to-band tunneling (BTBT) GENERATION coupled into the
bipolar drift-diffusion Newton, and the gate-oxide tunneling CURRENT post-processed from the solved
oxide field. Both are OPT-IN; nothing here runs unless asked, so every existing device is unchanged.

BAND-TO-BAND TUNNELING (local Kane / Hurkx model)
-------------------------------------------------
Net BTBT recombination rate [1/(m^3 s)], added to SRH in BOTH continuity equations:

    U_BTBT = A |E|^P exp(-B/|E|) * D(n, p),       D = (n p - n_i^2) / ((n + n_i)(p + n_i))

  * the field prefactor A |E|^P exp(-B/|E|) is Kane's tunneling rate (E. O. Kane, J. Appl. Phys. 32,
    83 (1961)): P = 2 for a DIRECT gap, P = 2.5 for the phonon-assisted (indirect) transition;
  * D is the Hurkx factor (G. A. M. Hurkx, D. B. M. Klaassen, M. P. G. Knuvers, IEEE Trans. Electron
    Devices 39, 331 (1992)): D = 0 at equilibrium (n p = n_i^2) so U vanishes EXACTLY there, D -> -1
    in a depleted reverse-biased junction (n, p << n_i: net GENERATION, U < 0), D > 0 under forward
    bias (tunneling-assisted recombination). Without D the Kane rate generates at zero bias -- a
    current with no driving force (the gate in tests/test_devsim_tunneling.py shows the magnitude).

PARAMETER SETS (v0.15 plan item 4.1.2). Units: with E in V/cm the literature rate is A E^P cm^-3 s^-1
(equivalently A [cm^-3 s^-1] (E/1 V/cm)^P, the Sentaurus form, or A [cm^-3 s^-1 (V/cm)^-P]; the same
numbers). In SI (BTBTParams.from_cgs): A_SI = A_cgs 1e6 100^-P [m^-3 s^-1 (V/m)^-P], B_SI = 100 B_cgs.
Every set records its `source`, its `process` and how its numbers were checked (`verification`:
"primary" = the original paper read; "secondary" = read in a reprint, the original NOT read; "computed" =
a call of kane_direct_params from cited inputs).

  set                      A (printed)            B (printed)   P    process    verification
  SILICON_HURKX            4e14 cm^-1/2 V^-5/2 s^-1  19 MV/cm    2.5  phonon     secondary (Hurkx 1992 not
                                                                                 read; Sentaurus UG Table 90)
  SILICON_KAO_DIRECT       1.35e20 cm^-3 s^-1     101 MV/cm     2    direct     secondary
  SILICON_KAO_INDIRECT     3.29e15 cm^-3 s^-1     23.8 MV/cm    2.5  phonon     secondary
  GERMANIUM_KAO_DIRECT     1.46e20 cm^-3 s^-1     6.04 MV/cm    2    direct     secondary
  GERMANIUM_KAO_INDIRECT   1.67e15 cm^-3 s^-1     6.55 MV/cm    2.5  phonon     secondary
  GERMANIUM_KAO            (GERMANIUM_KAO_DIRECT, GERMANIUM_KAO_INDIRECT) -- a TUPLE: both processes, summed
  IN053GA047AS_KANE        kane_direct_params(E_g(300 K) = 0.737 eV, m_r = 0.02346)   direct   computed

  * Kao: K.-H. Kao, A. S. Verhulst, W. G. Vandenberghe, B. Soree, G. Groeseneken, K. De Meyer, "Direct and
    indirect band-to-band tunneling in germanium-based TFETs", IEEE TED 59, 292 (2012), Table III, the
    uniform-field [100] values. The primary was NOT readable (closed access; the imec repository copy is
    embargoed, 2026-11-06); the numbers are re-typed from the open-access reprint Jiao et al., J. Semicond.
    34, 092002 (2013), Table 1 ("calculated in Ref. [14]" = Kao 2012), page image read. Kao states that
    Ge needs the direct AND the indirect process (the Gamma valley lies only ~0.14 eV above L), so use
    GERMANIUM_KAO, not either half. For Si the indirect (phonon-assisted) set is the physical one; the
    Si direct set is the Gamma-valley process, negligible below ~1e9 V/m. Ge-direct A as printed is ~10 %
    below the Kane formula at E_g = 0.80 eV, m_r = 0.022 (which reproduce its B); kept as printed (user
    decision 19), see docs/audit/2026-11-06-v0.15-btbt-presets.md.
  * In0.53Ga0.47As: COMPUTED (no traceable published local set exists): E_g(300 K) from the package gap
    law (materials.varshni_eg_ev "InGaAs_LM", Vurgaftman Eq. (4.1), 0.737 eV), m_e = 0.0430 and the [001]
    light-hole mass 0.0516 from Vurgaftman et al., JAP 89, 5815 (2001) (Tables I, III, XIII; Eq. (2.16)
    with Eq. (4.1)); the masses are the paper's LOW-temperature values. Treat it as a Kane estimate, not
    a calibration (a nonlocal calibration exists: Smets et al., JAP 115, 184503 (2014), not read).
  * NOT shipped (decision 18): vendor default sets (Sentaurus E2, the Sentaurus Ge 9.1e16 / 4.9 MV/cm),
    Atlas BBT.STD (units misprinted in its manual), untraceable InGaAs fits.

SEVERAL PROCESSES PER REGION. `btbt=` (setup_bipolar_region, TransportModel.btbt, Stacked3DSpec.btbt,
setup_btbt) takes one BTBTParams OR a tuple of them. A single set is installed exactly as before (term
"btbt", models BTBT_U / BTBT_U0, BTBT_U1 -- byte-identical). A tuple installs term "btbt_<k>" per set,
with models BTBT_U_<k> (1-D) / BTBT_U0_<k>, BTBT_U1_<k> (2-D/3-D); the terms SUM through the one
additive mechanism and each carries its own noise pair rate. btbt_generation_current sums them (or
takes index=k).

FIELD: LOCAL |E| (the nonlocal path-integrated model -- tunneling over the band-bending distance,
Hurkx sec. IV / Schenk -- is a documented follow-up). Discretisation, by dimension:
  * 2-D / 3-D: the ELEMENT field. DEVSIM's element_from_edge_model gives the exact P1 gradient of the
    potential in each triangle/tetrahedron (MEASURED: a uniform 1e8 V/m field at 30 deg is reproduced
    to 2e-16 on a 2-D tensor mesh), so |E| is ISOTROPIC -- unlike the edge-projected field the
    impact-ionization post-processor uses (impact_ionization.py: 1.8-3.1x low on 3-D tets). U is
    assembled per element edge into each end node's share of that element's volume
    (volume_node0_model / volume_node1_model, weight ElementNodeVolume) with D evaluated at THAT node --
    so a uniform field integrates exactly on any mesh whose element volumes are exact (2 sum ENV = the
    element volume). DEVSIM 2.11 breaks that identity on elements that are not well centred (it sums the
    magnitudes of the circumcentric pieces: an obtuse triangle 11x, the old gmsh tets 1.2-1.5x), so
    setup_btbt runs the shared element_geometry.check_element_geometry first and REFUSES such a mesh
    (InexactElementGeometryError). The library meshes (2-D rectilinear tensor, 3-D Kuhn tensor) pass.
  * 1-D: DEVSIM has no element models in 1-D (element_from_edge_model raises), and there the edge
    field IS the field: U on each edge with D = the mean of its two end-node D values, assembled via
    edge_volume_model (EdgeNodeVolume = L/2 to each node).
  |E| carries a 1 V/m floor, |E| = sqrt(E.E + 1 (V/m)^2), so its derivative is finite at E = 0 (and
  exp(-B/1 V/m) = 0 there); the bias it adds is < 1e-14 relative above 1e7 V/m.

COUPLING: SELF-CONSISTENT (inside the coupled Newton): the generation feeds the continuity equations
with full analytic Jacobians (the derivative models are DEVSIM symbolic diffs; checked against finite
differences in the tests).

TEMPERATURE (v0.15, user decision 20). A set's A, B are numbers at the temperature `T_K` it is quoted at
(None = T_REF = 300 K; every shipped set). Which option for which calculation:
  * region at the set's temperature (no BipolarOptions.T_K, or T_K equal to it): nothing to do.
  * region at another T, OPT-IN scaling: set `scale_B_with_gap=True` on the set (dataclasses.replace) and
    give the material a band-temperature model (TransportModel.band_temperature). The 2-D and 3-D builders
    then apply btbt_at_temperature: B -> B (E_g(T)/E_g(T_set))^(3/2), A unchanged (Sentaurus Device UG
    N-2017.09 Eq. 478, read; Hurkx 1992 via that text), with the ratio from materials.band_gap_ratio --
    the ONE gap law of the package. Calling setup_btbt / setup_bipolar_region directly, apply
    btbt_at_temperature(btbt, T, ratio) yourself. The ratio is that of the material's FUNDAMENTAL gap;
    for a direct-process set in an indirect material (Si or Ge direct) the direct gap's own temperature
    law differs (not modelled; stated).
  * region at another T, set fitted at that T: give the set its T_K (BTBTParams(..., T_K=250.0)).
  Otherwise setup_btbt installs the set as given and emits the v0.14 AdvisoryWarning (row 3.2-9): the
  BTBT current is then an estimate at the set's temperature.

GATE-OXIDE TUNNELING CURRENT (post-processed)
---------------------------------------------
oxide_tunneling_current() reads the CONVERGED potential in a dielectric region, takes the oxide-side
field normal to its (planar) interface with a semiconductor region, and evaluates one of the shared
kernels of dynameta.reliability.leakage at each interface node:
  "fn"           Fowler-Nordheim, J = A E^2 exp(-B/E) (Lenzlinger & Snow, J. Appl. Phys. 40, 278 (1969));
  "schuegraf_hu" the full Schuegraf-Hu trapezoid closed form (== FN above the barrier);
  "tsu_esaki"    the Tsu-Esaki/WKB quadrature (both regimes; needs the supply E_F and mass).
At each interface node the oxide drop V_ox is the COLUMN drop: the potential on the opposite (gate-side)
face of the oxide, interpolated to the node's lateral position, minus the node's potential; E_ox =
V_ox/t_ox with t_ox the oxide thickness along the normal. Both are exact for the linear potential of a
charge-free planar oxide on any mesh (a uniform-gate capacitor is exact); under a patch-gate edge they
are the straight-column approximation. The current is integrated over the interface with DEVSIM's
SurfaceArea node weights.

WHY POST-PROCESSED, and its limit: the tunneling current is NOT fed back into the semiconductor
continuity equation, so the carrier profile is that of an ideal (non-leaky) capacitor. That is exact to
first order while the leakage is small against the transport currents (the static-power / hold-bias
question it answers). It is WRONG where the leakage itself sets the carrier state -- a floating or
deep-depleted body whose minority supply is the tunneling current, or a gate stack in heavy FN
injection. A self-consistent interface sink is the follow-up (the decision is the user's -- see
docs/audit/2026-11-03-devsim-tunneling.md).

SIGN: J > 0 when the gate side of the oxide is at the HIGHER potential (electrons tunnel from the
semiconductor into the gate; conventional current flows gate -> semiconductor). The SAME barrier/supply
parameters serve both polarities (an asymmetric gate/semiconductor stack needs two calls).

UNITS follow the D1 contact-current convention: a 1-D device is per unit area (I in A/m^2); the 2-D
layered mesh is per unit depth (pass depth_m = period_y_m to get A); the 3-D gmsh mesh is in metres
(I already in A).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple, Union

import numpy as np

from dynameta.constants import H_PLANCK, KB, M_E, Q_E, T_REF, V_T
from dynameta.warning_types import AdvisoryWarning

__all__ = ["BTBTParams", "SILICON_HURKX", "SILICON_KAO_DIRECT", "SILICON_KAO_INDIRECT", "GERMANIUM_KAO_DIRECT",
           "GERMANIUM_KAO_INDIRECT", "GERMANIUM_KAO", "IN053GA047AS_KANE", "BTBT_PROCESSES",
           "BTBT_VERIFICATIONS", "kane_direct_params", "btbt_at_temperature", "btbt_for_material",
           "btbt_rate_expr", "hurkx_d_expr", "setup_btbt", "btbt_generation_current",
           "OxideTunnelingParams", "oxide_tunneling_current", "TUNNEL_MODELS"]


# ------------------------------------------------------------------------------------------------
# BTBT
# ------------------------------------------------------------------------------------------------

BTBT_PROCESSES = ("direct", "phonon_assisted")
BTBT_VERIFICATIONS = ("primary", "secondary", "computed")


@dataclass(frozen=True)
class BTBTParams:
    """Kane/Hurkx BTBT coefficients in SI: rate A |E|^P exp(-B/|E|) [1/(m^3 s)] with |E| in V/m, so
    A_SI is in m^-3 s^-1 (V/m)^-P and B in V/m. The Hurkx D factor is ALWAYS applied (it is what makes
    the rate vanish at equilibrium); see the module header.

    Physics fields (compared): A_SI, B_V_per_m, P, and
      T_K               the temperature [K] A and B are quoted at; None (default) = T_REF. setup_btbt warns when
                        the region runs at another temperature.
      scale_B_with_gap  OPT-IN (default False) temperature scaling B ~ E_g^(3/2) with A fixed (Sentaurus Eq.
                        478): the builders apply btbt_at_temperature from the material's band-temperature
                        model (module header, TEMPERATURE).
    Provenance fields (NOT compared, no physics): `source` (citation with table/equation), `process`
    (BTBT_PROCESSES: "direct" P = 2, "phonon_assisted" P = 2.5) and `verification` (BTBT_VERIFICATIONS:
    "primary" / "secondary" / "computed"; module header). All new fields are keyword fields with defaults
    appended after P, so BTBTParams(A, B, P) and every v0.14 call are unchanged."""
    A_SI: float
    B_V_per_m: float
    P: float = 2.5
    T_K: Optional[float] = None
    scale_B_with_gap: bool = False
    source: str = field(default="", compare=False)
    process: Optional[str] = field(default=None, compare=False)
    verification: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        if not (self.A_SI >= 0.0 and self.B_V_per_m > 0.0 and self.P >= 0.0):
            raise ValueError("BTBTParams: need A_SI >= 0, B_V_per_m > 0, P >= 0")
        if self.T_K is not None and not (math.isfinite(float(self.T_K)) and float(self.T_K) > 0.0):
            raise ValueError("BTBTParams: T_K must be None (= T_REF) or finite and > 0, got {!r}".format(
                self.T_K))
        if not isinstance(self.scale_B_with_gap, bool):
            raise TypeError("BTBTParams: scale_B_with_gap must be a bool")
        if self.process is not None and self.process not in BTBT_PROCESSES:
            raise ValueError("BTBTParams: process must be None or one of {}, got {!r}".format(
                BTBT_PROCESSES, self.process))
        if self.verification is not None and self.verification not in BTBT_VERIFICATIONS:
            raise ValueError("BTBTParams: verification must be None or one of {}, got {!r}".format(
                BTBT_VERIFICATIONS, self.verification))
        if not isinstance(self.source, str):
            raise TypeError("BTBTParams: source must be a str")

    @classmethod
    def from_cgs(cls, A_cgs: float, B_V_per_cm: float, P: float, **fields) -> "BTBTParams":
        """From the literature's cgs form, rate = A_cgs (E[V/cm])^P [cm^-3 s^-1]:
        A_SI = A_cgs * 1e6 * 100^-P, B_SI = 100 * B_V_per_cm. `fields` are the keyword fields (T_K,
        scale_B_with_gap, source, process, verification)."""
        return cls(A_SI=float(A_cgs) * 1.0e6 * 100.0 ** (-float(P)),
                   B_V_per_m=100.0 * float(B_V_per_cm), P=float(P), **fields)

    @property
    def quoted_T_K(self) -> float:
        """The temperature [K] A and B belong to: T_K, or T_REF when T_K is None."""
        return T_REF if self.T_K is None else float(self.T_K)


def kane_direct_params(Eg_eV: float, m_r_over_m0: float, g: float = 2.0, *, source: str,
                       verification: str = "computed") -> BTBTParams:
    """THE Kane direct-process (P = 2) set from a band gap and a reduced mass -- every computed preset is a
    call of this function, never a pasted number (plan 4.1.2, one function):

        A = g pi m_r^(1/2) q^2 / (9 h^2 E_g^(1/2))   [m^-3 s^-1 (V/m)^-2]
        B = pi^2 m_r^(1/2) E_g^(3/2) / (q h)          [V/m]

    (Synopsys Sentaurus Device UG N-2017.09 Eqs. 495-496, p. 439, with F0 = 1 V/m; read. With g = 2 this is
    Kane's uniform-field rate q^2 F^2 m_r^(1/2)/(18 pi hbar^2 E_g^(1/2)) exp(-pi m_r^(1/2) E_g^(3/2)/(2 q hbar F)),
    E. O. Kane, J. Appl. Phys. 32, 83 (1961) -- Kane itself not read; the same form is Das, Broderick &
    O'Reilly, arXiv:2108.09328 Eq. (13), read.) E_g [eV] is the DIRECT gap at the temperature the set is
    for, 1/m_r = 1/m_c + 1/m_v [m0] (conduction and light-hole masses along the tunnelling direction), g the
    spin degeneracy factor. Constants: exact-SI q and h, CODATA m0 (dynameta.constants). Use it for a direct
    material with cited E_g and masses when no published calibrated set exists; label the result as computed.
    The phonon-assisted A (Eq. 497) needs deformation-potential data and is not a closed form here: ship
    phonon-assisted sets only as published fits."""
    Eg, mr, gg = float(Eg_eV), float(m_r_over_m0), float(g)
    if not (Eg > 0.0 and mr > 0.0 and gg > 0.0):
        raise ValueError("kane_direct_params: need Eg_eV > 0, m_r_over_m0 > 0, g > 0")
    E_J = Eg * Q_E
    sm = math.sqrt(mr * M_E)
    A = gg * math.pi * sm * Q_E * Q_E / (9.0 * H_PLANCK * H_PLANCK * math.sqrt(E_J))
    B = math.pi * math.pi * sm * E_J ** 1.5 / (Q_E * H_PLANCK)
    return BTBTParams(A_SI=A, B_V_per_m=B, P=2.0, source=source, process="direct",
                      verification=verification)


# ---- shipped sets (module header table; plan 4.1.2, user decisions 18-19) ----------------------------------
# Hurkx, Klaassen & Knuvers, IEEE TED 39, 331 (1992): Si, phonon-assisted. A_SI = 4.0e15, B = 1.9e9 V/m.
SILICON_HURKX = BTBTParams.from_cgs(
    4.0e14, 1.9e7, 2.5, process="phonon_assisted", verification="secondary",
    source="Hurkx, Klaassen & Knuvers, IEEE TED 39, 331 (1992): A = 4e14 cm^-1/2 V^-5/2 s^-1, B = 1.9e7 V/cm, "
           "P = 2.5 (the primary was not read; the numbers as the Hurkx defaults of Sentaurus Device UG Table 90 "
           "and the Eq. 478 text)")

_KAO = ("Kao, Verhulst, Vandenberghe, Soree, Groeseneken & De Meyer, IEEE TED 59, 292 (2012), Table III, "
        "uniform-field [100] {} set (primary NOT read: closed access; re-typed from the reprint Jiao et al., "
        "J. Semicond. 34, 092002 (2013), Table 1, 'calculated in Ref. [14]')")
SILICON_KAO_DIRECT = BTBTParams.from_cgs(1.35e20, 101.0e6, 2.0, process="direct", verification="secondary",
                                         source=_KAO.format("Si direct"))
SILICON_KAO_INDIRECT = BTBTParams.from_cgs(3.29e15, 23.8e6, 2.5, process="phonon_assisted",
                                           verification="secondary", source=_KAO.format("Si indirect"))
# Ge direct A as printed (user decision 19): ~10 % below kane_direct_params(0.80, 0.022), which reproduces its B
GERMANIUM_KAO_DIRECT = BTBTParams.from_cgs(1.46e20, 6.04e6, 2.0, process="direct", verification="secondary",
                                           source=_KAO.format("Ge direct"))
GERMANIUM_KAO_INDIRECT = BTBTParams.from_cgs(1.67e15, 6.55e6, 2.5, process="phonon_assisted",
                                             verification="secondary", source=_KAO.format("Ge indirect"))
# Kao: both processes are needed in Ge (Gamma only ~0.14 eV above L) -> a TUPLE, the terms sum
GERMANIUM_KAO = (GERMANIUM_KAO_DIRECT, GERMANIUM_KAO_INDIRECT)


def _vurgaftman_alloy_value(p_A: float, p_B: float, x: float, C: float) -> float:
    """Vurgaftman, Meyer & Ram-Mohan, JAP 89, 5815 (2001) Eq. (4.1) for a temperature-independent band
    PARAMETER of A(1-x)B(x): (1-x) p_A + x p_B - x(1-x) C (used here for the masses; the gap's
    temperature law is materials.varshni_eg_ev)."""
    return (1.0 - x) * p_A + x * p_B - x * (1.0 - x) * C


def _ingaas_kane() -> BTBTParams:
    from dynameta.materials.band_temperature import ALLOY_GAP_PARAMS, varshni_eg_ev
    _a, _b, x, _c = ALLOY_GAP_PARAMS["InGaAs_LM"]                # ("GaAs", "InAs", 0.53, 0.477): x = In
    # Vurgaftman 2001 (re-read): m_e(Gamma) GaAs 0.067 (Table I), InAs 0.026 (Table III), bowing 0.0091
    # (Table XIII) -> 0.0430 = the "low-temperature value of 0.043 m0" of Sec. IV.A.2. Light hole along [001],
    # Eq. (2.16) m0/m_lh = gamma1 + 2 gamma2: GaAs 6.98, 2.06 (Table I), InAs 20.0, 8.5 (Table III), bowing
    # 0.0202 (Table XIII) -> 0.0516 (Alavi's gamma1 = 11.01, gamma2 = 4.18 quoted there give 0.0516).
    m_e = _vurgaftman_alloy_value(0.067, 0.026, x, 0.0091)
    m_lh = _vurgaftman_alloy_value(1.0 / (6.98 + 2.0 * 2.06), 1.0 / (20.0 + 2.0 * 8.5), x, 0.0202)
    m_r = 1.0 / (1.0 / m_e + 1.0 / m_lh)
    return kane_direct_params(
        varshni_eg_ev(T_REF, "InGaAs_LM"), m_r, 2.0,
        source="COMPUTED: kane_direct_params(E_g(300 K) of Ga0.47In0.53As = materials.varshni_eg_ev('InGaAs_LM') "
               "(Vurgaftman 2001 Eq. (4.1), Tables I/III/XIII: 0.737 eV), m_r = (1/m_e + 1/m_lh[001])^-1 with "
               "m_e = 0.0430, m_lh = 0.0516 (Vurgaftman 2001 Tables I/III/XIII, Eqs. (2.16)/(4.1); low-T masses), "
               "g = 2) -- a Kane estimate, not a calibration")


IN053GA047AS_KANE = _ingaas_kane()

_BTBTArg = Union[BTBTParams, Tuple[BTBTParams, ...]]


def _btbt_sets(btbt, who: str) -> Tuple[Tuple[BTBTParams, ...], bool]:
    """(sets, indexed): a single BTBTParams -> ((p,), False); a non-empty tuple/list of them -> (tuple, True)."""
    if isinstance(btbt, BTBTParams):
        return (btbt,), False
    if isinstance(btbt, (tuple, list)) and len(btbt) > 0 and all(isinstance(p, BTBTParams) for p in btbt):
        return tuple(btbt), True
    raise TypeError("{}: btbt must be a BTBTParams or a non-empty tuple of them (one term per process, "
                    "e.g. GERMANIUM_KAO); got {!r}".format(who, btbt))


def _scaled(p: BTBTParams, T_K: float, ratio: float) -> BTBTParams:
    # THE scaling (Sentaurus Eq. 478): B (E_g(T)/E_g(T_set))^(3/2), A and P unchanged; the result is quoted at T_K
    return replace(p, B_V_per_m=p.B_V_per_m * ratio ** 1.5, T_K=float(T_K))


def btbt_at_temperature(btbt: _BTBTArg, T_K: float, gap_ratio: float) -> _BTBTArg:
    """The opt-in temperature scaling of B (user decision 20; module header TEMPERATURE): every set with
    scale_B_with_gap=True becomes B (gap_ratio)^(3/2) with A and P unchanged and T_K = T_K; a set without the
    flag is returned unchanged (the same object). gap_ratio = E_g(T_K) / E_g(T_set), T_set the sets' quoted
    temperature -- e.g. materials.band_gap_ratio(transport_model, T_K) for sets quoted at T_REF. Use it when
    calling setup_btbt / setup_bipolar_region directly; the builders use btbt_for_material. A tuple is mapped
    set by set; its scaled sets must share one quoted temperature (one ratio)."""
    sets, indexed = _btbt_sets(btbt, "btbt_at_temperature")
    T, r = float(T_K), float(gap_ratio)
    if not (math.isfinite(T) and T > 0.0 and math.isfinite(r) and r > 0.0):
        raise ValueError("btbt_at_temperature: need T_K > 0 and gap_ratio > 0 (finite)")
    quoted = {p.quoted_T_K for p in sets if p.scale_B_with_gap}
    if len(quoted) > 1:
        raise ValueError("btbt_at_temperature: the sets to scale are quoted at different temperatures {} -- one "
                         "gap ratio cannot serve them; scale them one by one".format(sorted(quoted)))
    out = tuple(_scaled(p, T, r) if p.scale_B_with_gap else p for p in sets)
    return out if indexed else out[0]


def btbt_for_material(btbt, transport_model):
    """The builders' hook (2-D LayeredDevsimBuilder, 3-D Stacked3DSpec.from_design): `btbt` of a material at
    the temperature its region runs at (materials.transport_temperature_K: BipolarOptions.T_K, else T_REF).
    None -> None. Each set with scale_B_with_gap=True is scaled by btbt_at_temperature's rule with
    E_g(T)/E_g(T_set) from materials.band_gap_ratio -- the material's band-temperature model, the one gap
    law. Without a model (band_gap_ratio None) the set is passed on unscaled and setup_btbt keeps the v0.14
    advisory. A set without the flag is passed on as the SAME object (default: nothing changes)."""
    if btbt is None:
        return None
    sets, indexed = _btbt_sets(btbt, "btbt_for_material")
    if not any(p.scale_B_with_gap for p in sets):
        return btbt
    from dynameta.materials.band_temperature import band_gap_ratio, transport_temperature_K
    T = transport_temperature_K(transport_model)
    out = []
    for p in sets:
        r = band_gap_ratio(transport_model, T, p.quoted_T_K) if p.scale_B_with_gap else None
        out.append(p if r is None else _scaled(p, T, r))
    return tuple(out) if indexed else out[0]


def hurkx_d_expr(n: str, p: str, n_i: str = "n_i", *, pair: bool = False) -> str:
    """The Hurkx factor D = (n p - n_i^2)/((n + n_i)(p + n_i)) as a DEVSIM expression. `n_i` is the
    intrinsic-density symbol (a region parameter by default; pass the node-suffixed effective density,
    e.g. "n_ie@n0", under bandgap narrowing). pair=True gives the PAIR factor (n p + n_i^2)/(...): the
    detailed-balance split R ~ n p, G ~ n_i^2 of the net rate, whose sum G + R is the noise GR source
    (noise_analysis) -- the same split as the SRH kernel's pair rate."""
    return "(({n})*({p}) {s} {ni}^2)/((({n}) + {ni})*(({p}) + {ni}))".format(
        n=n, p=p, ni=n_i, s="+" if pair else "-")


def btbt_rate_expr(params: BTBTParams, *, emag: str, d: str) -> str:
    """THE shared DEVSIM expression for the net BTBT recombination rate U [1/(m^3 s)]:
    A exp(P ln|E| - B/|E|) * D. `emag` names a model/expression for |E| (> 0 -- floored by the callers);
    `d` is the D-factor expression for the node(s) the term is assembled into (hurkx_d_expr at one
    node in 2-D/3-D, the mean of the two end nodes in 1-D). exp(P ln E - B/E) instead of
    E^P exp(-B/E): one exponential, no overflow of E^P and no 0*inf at the floor.

    Coefficients are written with "%.17e" (always a FLOAT literal). v0.14 integration fix: the former
    "%.17g" printed an integral value as an INTEGER literal ("10000000000000000"), and DEVSIM 2.11.2's
    parser CLAMPS integer literals above 2^31 - 1 to 2147483647 (MEASURED: "10000000000000000" ->
    2147483647.0, "1e16" -> 1e16) -- so any A >= 2^31 or B >= 2^31 V/m that happens to be integral
    was silently replaced. The shipped SILICON_HURKX (A = 4000000000000000.5, B = 1.9e9 < 2^31) was not
    affected; the parsed doubles are identical, so its results are unchanged."""
    return "{A:.17e}*exp({P:.17e}*log({E}) - {B:.17e}/({E}))*({D})".format(
        A=float(params.A_SI), P=float(params.P), B=float(params.B_V_per_m), E=emag, D=d)


def _require_bipolar(device: str, region: str):
    from dynameta.carriers import eq_registry as _R
    for eq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        if not _R.recorded(device, eq, scope="region", loc=region):
            raise ValueError("setup_btbt: region {!r} has no {} -- BTBT generation needs a bipolar "
                             "drift-diffusion region (physics_bipolar_dd.setup_bipolar_region)".format(
                                 region, eq))


def _set_V_t(p: BTBTParams) -> float:
    # the thermal voltage of a set's quoted temperature, by BipolarOptions.V_t's own rule (V_T at T_REF)
    return V_T if p.T_K is None else KB * float(p.T_K) / Q_E


def _warn_temperature(region: str, vt: float, sets) -> None:
    """v0.14 row 3.2-9, kept by v0.15 decision 20 for every set NOT at the region's temperature."""
    off = [p for p in sets if _set_V_t(p) != vt]
    if not off:
        return
    from dynameta.carriers.physics_heterojunction import temperature_of_V_t
    quoted = sorted({p.quoted_T_K for p in off})
    at = ", ".join("T_REF = {} K".format(T_REF) if t == T_REF else "T = {:.6g} K".format(t) for t in quoted)
    pending = sum(1 for p in off if p.scale_B_with_gap)
    flag = ("" if not pending else
            " scale_B_with_gap is set on {} of them, but no E_g(T) ratio was applied (the material has no "
            "band-temperature model, or setup_btbt was called directly without btbt_at_temperature).".format(
                pending))
    warnings.warn(
        "setup_btbt: region {!r} runs at T = {:.6g} K (BipolarOptions.T_K) but the BTBT parameters (A, B, P) "
        "of {} of its {} set(s) are used as given, i.e. as at {}: B ~ E_g^(3/2) is not rescaled with E_g(T).{} To "
        "scale B with the gap (A fixed; Sentaurus Eq. 478) set scale_B_with_gap=True and give the material "
        "a band-temperature model (TransportModel.band_temperature; the builders apply btbt_at_temperature), "
        "or apply btbt_at_temperature yourself; or supply A, B fitted at this temperature "
        "(BTBTParams(..., T_K=...)). Otherwise treat the BTBT current as an estimate at the set's "
        "temperature.".format(region, temperature_of_V_t(vt), len(off), len(sets), at, flag),
        AdvisoryWarning, stacklevel=4)


def setup_btbt(device: str, region: str, params, *, n_i: str = "n_i") -> None:
    """Install the BTBT term(s) (module header) into an EXISTING bipolar region: build the field-magnitude
    and rate models (the diagnostics BTBT_Emag, BTBT_U / BTBT_U0, BTBT_U1) and add the rate through the
    ONE additive term mechanism, physics_bipolar_dd.add_recombination_term (assembly "edge" in 1-D,
    "element" in 2-D/3-D), with its noise pair rate. That mechanism wires the continuity equations'
    volume slots once (eq_registry.amend_region_equation, recorded, so a Gummel/staged
    delete_by_name -> reapply_by_name keeps them) and SUMS every term sharing a slot.
    Called by setup_bipolar_region(btbt=...).

    `params`: one BTBTParams (term "btbt", models BTBT_U / BTBT_U0, BTBT_U1 -- the v0.14 names) or a
    non-empty TUPLE of them, one per process (e.g. GERMANIUM_KAO = direct + indirect): term "btbt_<k>" with
    models BTBT_U_<k> / BTBT_U0_<k>, BTBT_U1_<k> per set, sharing one BTBT_Emag; the terms sum, each with its
    own noise pair rate.

    Temperature: a set is installed as given. A region whose V_t is not that of the set's quoted temperature
    (BTBTParams.T_K, None = T_REF) gets the AdvisoryWarning of v0.14 row 3.2-9; scale with
    btbt_at_temperature BEFORE this call (the builders do it through btbt_for_material).

    `n_i`: the intrinsic density in the Hurkx D -- a region PARAMETER name ("n_i", default) or a NODE
    MODEL name ("n_ie" under bandgap narrowing: setup_bipolar_region passes it, so D vanishes at the
    narrowed equilibrium n p = n_ie^2); a node model is evaluated at each D node (@n0/@n1 or
    @en0/@en1)."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    from dynameta.carriers import physics_bipolar_dd as _BP
    from dynameta.carriers.eq_registry import edge_with_derivs

    sets, indexed = _btbt_sets(params, "setup_btbt")
    _require_bipolar(device, region)
    # v0.14 row 3.2-9 (decision D-H4), kept by v0.15 decision 20: A, B are numbers at the set's quoted
    # temperature and are NOT rescaled here; say so instead of running silently.
    vt = float(ds.get_parameter(device=device, region=region, name="V_t"))
    _warn_temperature(region, vt, sets)
    ni_node = n_i in ds.get_node_model_list(device=device, region=region)

    def ni_at(sfx):
        return n_i + sfx if ni_node else n_i

    def suffix(k):                         # "" for a single set: the v0.14 names, byte-identical
        return "_{}".format(k) if indexed else ""

    dim = int(ds.get_dimension(device=device))
    if dim == 1:
        # the edge field IS the field in 1-D; D = mean of the two end-node values
        if ni_node:
            ds.edge_from_node_model(device=device, region=region, node_model=n_i)
        edge_with_derivs(device, region, "BTBT_Emag", "pow(ElectricField^2 + 1.0, 0.5)", ("Potential",))

        def d_avg(pair):
            return "0.5*({} + {})".format(
                hurkx_d_expr("Electrons@n0", "Holes@n0", ni_at("@n0"), pair=pair),
                hurkx_d_expr("Electrons@n1", "Holes@n1", ni_at("@n1"), pair=pair))
        wrt = ("Potential", "Electrons", "Holes")
        for k, prm in enumerate(sets):
            rate = "BTBT_U" + suffix(k)
            edge_with_derivs(device, region, rate, btbt_rate_expr(prm, emag="BTBT_Emag", d=d_avg(False)), wrt)
            _BP.add_recombination_term(device, region, "btbt" + suffix(k), rate, assembly="edge",
                                       pair=btbt_rate_expr(prm, emag="BTBT_Emag", d=d_avg(True)))
        return
    if dim not in (2, 3):                                     # pragma: no cover - devsim dims are 1..3
        raise NotImplementedError("setup_btbt: dimension {}".format(dim))
    # the element assembly weights each element edge's rate by ElementNodeVolume: exact only on a
    # well-centred mesh -- refuse any other (row 3.2-13, the shared check), BEFORE any model is created
    from dynameta.carriers.element_geometry import check_element_geometry
    check_element_geometry(device, region, caller="setup_btbt ({}-D element assembly)".format(dim))
    from dynameta.carriers.eq_registry import element_with_derivs
    ens = ["@en{}".format(k) for k in range(dim + 1)]           # the element's nodes (en0/en1 = the edge)
    comps = ["ElectricField_{}".format(c) for c in "xyz"[:dim]]
    ds.element_from_edge_model(device=device, region=region, edge_model="ElectricField")
    ds.element_from_edge_model(device=device, region=region, edge_model="ElectricField",
                               derivative="Potential")
    if ni_node:
        ds.element_from_node_model(device=device, region=region, node_model=n_i)
    pot = ["Potential" + e for e in ens]
    element_with_derivs(device, region, "BTBT_Emag",
                        "pow({} + 1.0, 0.5)".format(" + ".join(c + "^2" for c in comps)), pot)
    # DEVSIM's element assembler demands the derivative w.r.t. EVERY element node of EVERY solution
    # variable of the equation (MEASURED: "Model: ...:Electrons@en1 MISSING" is FATAL), so the carrier
    # derivatives are created on all ens, most of them simplifying to the literal 0.
    wrt = pot + ["Electrons" + e for e in ens] + ["Holes" + e for e in ens]
    for i, prm in enumerate(sets):
        shares, pairs = [], []
        for k in (0, 1):                                      # node0 / node1 share of each element edge
            nd = "@en{}".format(k)
            u = btbt_rate_expr(prm, emag="BTBT_Emag",
                               d=hurkx_d_expr("Electrons" + nd, "Holes" + nd, ni_at(nd)))
            element_with_derivs(device, region, "BTBT_U{}{}".format(k, suffix(i)), u, wrt)
            shares.append("BTBT_U{}{}".format(k, suffix(i)))
            pairs.append(btbt_rate_expr(prm, emag="BTBT_Emag",
                                        d=hurkx_d_expr("Electrons" + nd, "Holes" + nd, ni_at(nd), pair=True)))
        _BP.add_recombination_term(device, region, "btbt" + suffix(i), tuple(shares), assembly="element",
                                   pair=tuple(pairs))


def _btbt_rate_models(ds, device: str, region: str, dim: int):
    """[(rate model names of term k), ...] installed by setup_btbt on the region, in tuple order."""
    have = set(ds.get_edge_model_list(device=device, region=region) if dim == 1
               else ds.get_element_model_list(device=device, region=region))
    base = ("BTBT_U",) if dim == 1 else ("BTBT_U0", "BTBT_U1")
    if base[0] in have:
        return [base]
    out = []
    while base[0] + "_{}".format(len(out)) in have:
        out.append(tuple(b + "_{}".format(len(out)) for b in base))
    if not out:
        raise ValueError("btbt_generation_current: region {!r} has no BTBT term (setup_btbt)".format(region))
    return out


def btbt_generation_current(device: str, region: str, *, depth_m: Optional[float] = None,
                            index: Optional[int] = None) -> float:
    """q * Int G_BTBT dV over the solved region [A] (G = -U: positive = net GENERATION), with the SAME
    quadrature weights the equations assemble (so in steady state it equals the BTBT share of the
    terminal current). 1-D: per unit area [A/m^2]; 2-D layered: per unit depth unless depth_m is given.
    Requires setup_btbt on the region. With a tuple of sets: the SUM over the terms in tuple order (each
    term's integral as for a single set), or term `index` alone (its position in the tuple; a single set
    is index 0)."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None to keep the mesh's native units)")
    dim = int(ds.get_dimension(device=device))
    terms = _btbt_rate_models(ds, device, region, dim)
    if index is not None:
        if not (isinstance(index, int) and 0 <= index < len(terms)):
            raise ValueError("btbt_generation_current: index must be an int in [0, {}), got {!r}".format(
                len(terms), index))
        terms = [terms[index]]
    if dim == 1:
        w = 2.0 * np.asarray(ds.get_edge_model_values(device=device, region=region,
                                                      name="EdgeNodeVolume"))
    else:
        w = np.asarray(ds.get_element_model_values(device=device, region=region,
                                                   name="ElementNodeVolume"))
    total = 0.0
    for names in terms:                   # the total is the SUM of the per-term currents (index=k each)
        if dim == 1:
            u = np.asarray(ds.get_edge_model_values(device=device, region=region, name=names[0]))
            integral = float(np.sum(u * w))
        else:
            u0 = np.asarray(ds.get_element_model_values(device=device, region=region, name=names[0]))
            u1 = np.asarray(ds.get_element_model_values(device=device, region=region, name=names[1]))
            integral = float(np.sum((u0 + u1) * w))
        total += -Q_E * integral * (float(depth_m) if depth_m is not None else 1.0)
    return total


# ------------------------------------------------------------------------------------------------
# gate-oxide tunneling current (post-processed)
# ------------------------------------------------------------------------------------------------

TUNNEL_MODELS = ("fn", "schuegraf_hu", "tsu_esaki")


@dataclass(frozen=True)
class OxideTunnelingParams:
    """Gate-oxide tunneling post-processor settings (module header). SiO2 defaults: phi_b = 3.1 eV,
    m_ox = 0.42 m0 (the Lenzlinger-Snow fit). model in TUNNEL_MODELS; "tsu_esaki" REQUIRES E_F_eV (the
    cathode Fermi energy above its supply band bottom -- no default, it is a property of the stack)
    and takes m_supply_over_m0 / T_K. "fn" is valid only for V_ox >= phi_b; below it the call warns
    (use "tsu_esaki" for a thin oxide in direct tunneling)."""
    model: str = "fn"
    phi_b_eV: float = 3.1
    m_ox_over_m0: float = 0.42
    E_F_eV: Optional[float] = None
    m_supply_over_m0: float = 1.0
    T_K: float = T_REF

    def __post_init__(self):
        if self.model not in TUNNEL_MODELS:
            raise ValueError("OxideTunnelingParams.model must be one of {} (got {!r})".format(
                TUNNEL_MODELS, self.model))
        if not (self.phi_b_eV > 0.0 and self.m_ox_over_m0 > 0.0 and self.m_supply_over_m0 > 0.0
                and self.T_K >= 0.0):
            raise ValueError("OxideTunnelingParams: phi_b_eV, m_ox_over_m0, m_supply_over_m0 must be "
                             "> 0 and T_K >= 0")
        if self.model == "tsu_esaki" and not (self.E_F_eV is not None and self.E_F_eV > 0.0):
            raise ValueError("OxideTunnelingParams(model='tsu_esaki') needs E_F_eV > 0 (the cathode "
                             "supply Fermi energy; a metal gate ~3-11 eV, an accumulated Si surface "
                             "~0.05-0.2 eV) -- there is no safe default")

    def current_density(self, V_ox_V, t_ox_m: float):
        """|J| [A/m^2] for oxide drop(s) V_ox_V >= 0 across thickness t_ox_m (dispatch to the shared
        leakage kernels)."""
        from dynameta.reliability import leakage as LK
        V = np.asarray(V_ox_V, dtype=np.float64)
        if self.model == "fn":
            return LK.fowler_nordheim_current(V / float(t_ox_m), m_ox_over_m0=self.m_ox_over_m0,
                                              phi_b_eV=self.phi_b_eV)
        if self.model == "schuegraf_hu":
            return LK.schuegraf_hu_current(V, t_ox_m, m_ox_over_m0=self.m_ox_over_m0,
                                           phi_b_eV=self.phi_b_eV)
        return LK.tsu_esaki_current(V, t_ox_m, phi_b_eV=self.phi_b_eV, E_F_eV=float(self.E_F_eV),
                                    m_ox_over_m0=self.m_ox_over_m0,
                                    m_supply_over_m0=self.m_supply_over_m0, T_K=self.T_K)


def _interface_between(device: str, region_a: str, region_b: str) -> str:
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    hits = [i for i in ds.get_interface_list(device=device)
            if set(ds.get_region_list(device=device, interface=i)) == {region_a, region_b}]
    if len(hits) != 1:
        raise ValueError("oxide_tunneling_current: expected exactly one interface between {!r} and "
                         "{!r}, found {}".format(region_a, region_b, hits))
    return hits[0]


def oxide_tunneling_current(device: str, oxide_region: str, semiconductor_region: str,
                            params: OxideTunnelingParams, *,
                            depth_m: Optional[float] = None) -> Dict[str, object]:
    """Post-process the gate-oxide tunneling current through `oxide_region` from the solved potential
    (module header for the method, sign and units). Returns a dict:
      I              total current through the oxide/semiconductor interface (sign: module header)
      J_A_m2         signed current density at each interface node [A/m^2]
      V_ox_V         |oxide drop| through the oxide column above each interface node [V]
      E_ox_V_m       V_ox / t_ox [V/m] (the uniform field of a charge-free planar oxide)
      area           the SurfaceArea weight of each node (1-D: 1; 2-D: m; 3-D: m^2)
      t_ox_m, normal_axis, interface, model.
    Requires a PLANAR interface normal to a coordinate axis (both builders' stacks are)."""
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    if not isinstance(params, OxideTunnelingParams):
        raise TypeError("oxide_tunneling_current: params must be an OxideTunnelingParams")
    if depth_m is not None and not (depth_m > 0.0):
        raise ValueError("depth_m must be > 0 (or None to keep the mesh's native units)")
    dim = int(ds.get_dimension(device=device))
    itf = _interface_between(device, oxide_region, semiconductor_region)
    side = "r0" if ds.get_region_list(device=device, interface=itf)[0] == oxide_region else "r1"
    nm = "_tun_oxide_idx_" + side
    if nm not in ds.get_interface_model_list(device=device, interface=itf):
        ds.interface_model(device=device, interface=itf, name=nm, equation="node_index@" + side)
    iface = np.asarray(ds.get_interface_model_values(device=device, interface=itf, name=nm)).astype(int)

    def node(name):
        return np.asarray(ds.get_node_model_values(device=device, region=oxide_region, name=name),
                          dtype=np.float64)

    X = np.column_stack([node(c) for c in "xyz"[:dim]])
    psi = node("Potential")
    # planar interface: exactly one axis along which every interface node sits at the same coordinate
    span = np.ptp(X, axis=0)
    ispan = np.ptp(X[iface], axis=0)
    flat = [0] if dim == 1 else [k for k in range(dim)
                                 if ispan[k] <= 1e-9 * max(float(span.max()), 1e-30)]
    if len(flat) != 1:
        raise NotImplementedError("oxide_tunneling_current: the {}/{} interface is not planar and "
                                  "normal to a coordinate axis (per-axis spread {}); only layered "
                                  "stacks are supported".format(oxide_region, semiconductor_region,
                                                                ispan.tolist()))
    ax = flat[0]
    z_if = float(np.mean(X[iface, ax]))
    t_ox = float(span[ax])
    far_z = float(X[:, ax].max()) if float(np.mean(X[:, ax])) > z_if else float(X[:, ax].min())
    far = np.nonzero(np.abs(X[:, ax] - far_z) <= 1e-9 * t_ox)[0]  # the opposite (gate-side) face
    # COLUMN DROP: V_col = psi(opposite face, same lateral position) - psi(interface node), with the
    # opposite face interpolated piecewise-linearly in the lateral coordinate(s). Exact for any linear
    # potential (a charge-free planar oxide) on any mesh, and -- unlike the nodal gradient at the
    # interface -- it does not amplify the FV solution's lateral noise on an unstructured 3-D mesh:
    # MEASURED on a laterally-uniform 3-D Si/SiO2 stack (4 nm cell, mesh_min 0.5 nm): the per-node
    # interface gradient varied 17.7 % across the interface, the column drop 1.5 %.
    lat = [k for k in range(dim) if k != ax]
    if not lat:
        psi_far = np.full(iface.size, float(np.mean(psi[far])))
    elif len(lat) == 1:
        o = np.argsort(X[far, lat[0]])
        psi_far = np.interp(X[iface, lat[0]], X[far, lat[0]][o], psi[far][o])
    else:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        pts = X[far][:, lat]
        psi_far = LinearNDInterpolator(pts, psi[far])(X[iface][:, lat])
        miss = ~np.isfinite(psi_far)                         # hull round-off at the cell boundary
        if np.any(miss):
            psi_far[miss] = NearestNDInterpolator(pts, psi[far])(X[iface][miss][:, lat])
    G = psi_far - psi[iface]                                   # > 0: the gate side is at the higher potential
    V_ox = np.abs(G)
    E_ox = V_ox / t_ox
    if params.model == "fn" and bool(np.any((V_ox > 0.0) & (V_ox < params.phi_b_eV))):
        warnings.warn(
            "oxide_tunneling_current(model='fn'): {} of {} interface nodes have V_ox below the barrier "
            "(max V_ox = {:.3g} V < phi_b = {:.3g} eV) -- Fowler-Nordheim is the TRIANGULAR-barrier limit "
            "and under-counts direct tunneling there (by orders of magnitude for a few-nm oxide). Use "
            "model='tsu_esaki' (or 'schuegraf_hu' at moderate V_ox).".format(
                int(np.count_nonzero((V_ox > 0.0) & (V_ox < params.phi_b_eV))), V_ox.size,
                float(V_ox.max()), params.phi_b_eV), AdvisoryWarning, stacklevel=2)
    J = np.sign(G) * np.asarray(params.current_density(V_ox, t_ox), dtype=np.float64)
    area = node("SurfaceArea")[iface]
    total = float(np.sum(J * area)) * (float(depth_m) if depth_m is not None else 1.0)
    return {"I": total, "J_A_m2": J, "E_ox_V_m": E_ox, "V_ox_V": V_ox, "area": area,
            "t_ox_m": t_ox, "normal_axis": ax, "interface": itf, "model": params.model}
