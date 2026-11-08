"""Band data at a temperature: THE Varshni gap law of the package and the opt-in band-temperature model.

ONE KERNEL, ONE TABLE (v0.15 plan 4.0.4). `varshni_eg_ev` / `d_eg_dT_ev_per_K` and the tables
`VARSHNI_PARAMS` / `ALLOY_GAP_PARAMS` live here, devsim-free and optics-free, and are used by
  * optics.soa (re-exported under the same names from optics.soa.temperature: the qw_gain band edge and
    qd_params_at_temperature; moved here verbatim on 2026-11-06, results bit-identical), and
  * the DEVSIM carrier builders, through `BandTemperature` + `band_data_at_T` below (the Anderson frame,
    the bipolar band-gap check, the bipolar n_i), and the later BTBT temperature scaling (`band_gap_ratio`).

THE GAP LAW (Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001); every value re-read from the
paper):
  * binary   Eq. (2.13)  E_g(T) = E_g(0) - alpha T^2 / (T + beta)        (Varshni, Physica 34, 149 (1967))
             GaAs Table I (1.519 eV, 0.5405 meV/K, 204 K), AlAs Table II Gamma (3.099 eV, 0.885 meV/K, 530 K),
             InAs Table III, InP Table VI. Only binaries have Varshni rows.
  * ternary  Eq. (4.1)   E_g(A(1-x)B(x)) = (1-x) E_g,A(T) + x E_g,B(T) - x (1-x) C, with the bowing C
             temperature-INDEPENDENT (the paper's recommended assumption: AlGaAs Sec. IV.A.1 -- "in
             reasonable agreement with our recommended assumption that the bowing parameter is independent
             of temperature"; GaInAs Sec. IV.A.2). AlGaAs Gamma: C = -0.127 + 1.310 x (Table XII, Aspnes et
             al.; the composition-dependent form the paper recommends for this alloy) -> `algaas_gap_law(x)`.
             GaInAs lattice-matched to InP: C = 0.477 eV at x(In) = 0.53 (Table XIII) -> "InGaAs_LM".
  * lattice-matched quaternary  Eq. (5.4)  G(z) = (1-z) G_a + z G_b - z (1-z) C_ab: the SAME closed form
             with two LATTICE-MATCHED end points a, b, each a binary or a ternary (Sec. V, introduction).
             GaInAsP on InP = (InP)(1-z) (Ga0.47In0.53As)(z), i.e. Ga(x)In(1-x)As(z)P(1-z) with x = 0.47 z
             (Sec. V.B.1): C = 0.13 eV "at all temperatures", and the temperature dependence deduced from
             the binary Varshni laws (the paper's recommendation over the two published, mutually
             inconsistent GaInAsP Varshni fits) -> `ingaasp_lm_gap_law(z)` and the C-band SOA preset
             "InGaAsP_LM_C". The paper's printed closed forms "Eg = 1.4236(1-z) + 0.816z + 0.13z^2" (0 K)
             and "1.353(1-z) + 0.737z + 0.13z^2" (300 K) drop the -0.13 z term of -z(1-z)C (at z = 1 they
             give 0.946 / 0.867 eV, not the paper's own Ga0.47In0.53As end points 0.816 / 0.737 eV); the
             code follows Eq. (5.4), as the text says to.
  LATTICE CONSTANTS (300 K, Tables I-IV and VI: GaAs 5.65325, AlAs 5.6611, GaP 5.4505, InAs 6.0583, InP
             5.8697 Angstrom) with Vegard's law decide which end-point pairs are lattice matched (an alloy end
             point is refused otherwise) and give the strain of a GaInAsP composition on InP
             (`ingaasp_lattice_mismatch`).
  Validity: Varshni is an empirical fit; the cited parameter sets are fitted from ~4 K to above room
  temperature, and some sets elsewhere in the literature misbehave below ~50 K -- the gates go no lower
  than 77 K. The Gamma gap of Al(x)Ga(1-x)As is the FUNDAMENTAL gap only below the Gamma-X crossover,
  x = 0.38 at low T and 0.39 at 300 K (Sec. IV.A.1); above it the conduction edge is the X valley (not
  modelled here: give that material its own gap law).

THE BAND-TEMPERATURE MODEL (opt-in; TransportModel.band_temperature, default None = OFF). Users supply 0 K
or 300 K band data and simulate at another T. With a `BandTemperature` on a material, `band_data_at_T`
returns (E_g, chi, n_i) at the region temperature T (BipolarOptions.T_K for a bipolar region, T_REF
otherwise). T_ref = TransportModel.chi_reference_T_K (None = T_REF) is the temperature the material's
band_gap_eV / chi_eV / n_i_m3 are QUOTED at -- the same field the v0.14 affinity guard reads.

  gap, anchor="reference" (DEFAULT; user decision 32):
             E_g(T) = band_gap_eV + [V(T) - V(T_ref)],  V = the gap law above.
             The law contributes only its CHANGE, so the material's own gap is kept: at T = T_ref the result
             IS band_gap_eV (bit-identical, the delta is exactly 0.0). Use it whenever the material's gap is
             a 300 K (or other non-zero T) number -- in particular a gap that is defined by a paired
             statistics bundle, kT ln(N_c N_v / n_i^2): Si with P3's absolute Varshni (1.17 eV, 4.73e-4
             eV/K, 636 K) gives 1.1245 eV at 300 K, 5.2 meV off the bundle's 1.12973 eV, and a derived n_i
             of 1.067e10 instead of 9.65e9 cm^-3 (plan 5.3) -- the anchored form changes neither at 300 K.
  gap, anchor="absolute": E_g(T) = V(T) -- for data quoted at 0 K, where the law's E_g(0) IS the
             material's gap. band_gap_eV must then be None (a second gap would be an unused second source).
  affinity   chi(T) = chi_eV - gamma_chi [V(T) - V(T_ref)],  gamma_chi = `affinity_fraction`, REQUIRED (no
             default; user decision 46). There is no standard; choose by where the gap change goes:
               gamma_chi = 1   VALENCE-ANCHORED: E_v fixed on the absolute scale, the whole gap change moves
                               E_c. This is Vurgaftman et al.'s convention ("in all cases we will take the
                               valence band offsets to be independent of T", Sec. VI, p. 5854), so use 1 with
                               their VBO-based band alignments (III-V heterostacks built from that paper).
               gamma_chi = 1/2 SYMMETRIC: half the change moves each edge. The Sentaurus Device default
                               affinity model chi(T) = chi(0) + alpha T^2 / (2 (T + beta)) (read from a
                               Sentaurus training mirror, PMI section 17.5.4 -- a SECONDARY source; the
                               licensed manual was not accessed). Use it to reproduce Sentaurus results or
                               where no offset data constrain the split.
             Delta E_c(T) of a pair then moves by gamma_chi x the mismatch of the two gap changes
             (Al0.3Ga0.7As / GaAs: 2.15 meV per 100 K in the gap difference, 250 -> 350 K).
  n_i        derive_n_i=False (default): n_i_m3 is used as given -- YOUR value at the device T (the bipolar
             heterojunction band-gap check compares kT ln(N_c N_v/n_i^2) with E_g(T) and refuses a 300 K n_i
             at 250 K). derive_n_i=True: n_i_m3 is the value at T_ref and
                 n_i(T) = n_i(T_ref) (T/T_ref)^(3/2) exp(-[E_g(T)/kT - E_g(T_ref)/kT_ref] / 2),
             i.e. n_i = sqrt(N_c N_v) exp(-E_g/2kT) with N_c N_v ~ T^3 (effective_dos_m3 with T-independent
             DOS masses) anchored at the material's own n_i: bit-identical at T_ref, and the material's
             statistical gap kT ln(N_c N_v/n_i^2) then equals E_g(T) up to the T_ref mismatch scaled by
             T/T_ref (zero for a consistent material). This is the RECOMMENDED "own n_i(T)" for the P2 paired
             band-gap-narrowing override at T != 300 K (plan 5.2 item 3): a BGN fit stays a 300 K statement,
             but this n_i(T) is consistent with the band model the device runs. Not derivable at T_ref = 0 K.
  Not modelled (stated): the temperature dependence of the DOS and Richardson masses (A* T^2 uses the
  constant DOS mass), and of the BGN fits; band data of insulators (dielectrics carry no band model).

INTEGRATION WITH THE v0.14 AFFINITY GUARD. A material that carries a band-temperature model has its chi
EVALUATED at the device T, so physics_heterojunction.check_affinity_temperature counts it as consistent
(`physics_heterojunction.affinity_temperature_tags`) and the builders' on_chi_temperature toggle no longer
applies to it. A material without one is checked exactly as in v0.14.

ASCII only; SI except energies in eV where named _eV / _ev.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union

from dynameta.constants import KB, Q_E, T_REF

__all__ = ["VARSHNI_PARAMS", "ALLOY_GAP_PARAMS", "ALGAAS_GAMMA_BOWING_EV", "varshni_eg_ev",
           "d_eg_dT_ev_per_K", "algaas_gap_law", "LATTICE_CONSTANT_300K_A", "LATTICE_MATCH_RTOL",
           "INGAASP_LM_BOWING_EV", "INGAASP_LM_C_Z", "ingaasp_lm_gap_law", "ingaasp_lattice_mismatch", "BandTemperature", "BandData", "BAND_ANCHORS",
           "reference_temperature_K", "transport_temperature_K", "band_data_at_T", "band_gap_ratio",
           "check_band_temperature"]


# Gamma-gap Varshni (Eg0 [eV], alpha [eV/K], beta [K]) of the BINARIES -- Vurgaftman, Meyer &
# Ram-Mohan, JAP 89, 5815 (2001): GaAs Table I, AlAs Table II, InAs Table III, InP Table VI (alpha re-typed
# from the tables' meV/K). Only a binary has a Varshni row; an alloy is NOT a Varshni triple (ALLOY_GAP_PARAMS).
VARSHNI_PARAMS = {
    "GaAs":       (1.519, 5.405e-4, 204.0),
    "AlAs":       (3.099, 8.85e-4, 530.0),
    "InAs":       (0.417, 2.76e-4, 93.0),
    "InP":        (1.4236, 3.63e-4, 162.0),
}

# Ternary A(1-x) B(x) C alloys: (binary A, binary B, x, Gamma-gap bowing C [eV]) for Vurgaftman
# Eq. (4.1) Eg(T) = (1-x) Eg_A(T) + x Eg_B(T) - x (1-x) C, with C temperature-INDEPENDENT (the
# paper's stated choice for GaInAs, Sec. IV.A.2) and Eg_A, Eg_B the VARSHNI_PARAMS laws.
#   InGaAs_LM: Ga0.47In0.53As lattice-matched to InP; C = 0.477 eV (Table XIII), which the paper
#   fixes from its composite Eg(0) = 816 meV at x = 0.53. Eg(300 K) = 0.737 eV, as the paper itself
#   quotes for this end point in Sec. V.B.1 (GaInAsP on InP). Audit 2026-11-06: this row used to be a
#   Varshni triple (0.816 eV, 2.9e-4 eV/K, 193 K) with no table behind it -- right at 0 K but 26 meV
#   too wide at 300 K.
#   InGaAsP_LM_C: the opt-in C-band SOA active material (v0.15.1, user decision 2026-11-08): the
#   InP-lattice-matched quaternary (InP)(1-z)(Ga0.47In0.53As)(z) = Ga(0.47z)In(1-0.47z)As(z)P(1-z) of
#   Sec. V.B.1 -- Eq. (5.4) with C = 0.13 eV between the InP law and the "InGaAs_LM" law above -- at the
#   composition INGAASP_LM_C_Z (see ingaasp_lm_gap_law).
# GaInAsP on InP: the Eq. (5.4) bowing of Sec. V.B.1 ("C = 0.13 at all temperatures"), T-independent.
INGAASP_LM_BOWING_EV = 0.13
# z (= the As fraction y; Ga fraction 0.47 z) of the C-band preset: solved so that the Connelly material-gain
# peak of optics.soa.qw_gain.BulkGainParams.c_band() sits at 1550.0 nm at the library's reference operating
# point (every other BulkGainParams default: 298 K, 130 mA, P = 0 -> N0 = 1.695e24 m^-3), then rounded to the
# 0.001 a grower specifies (brentq on the peak wavelength: z* = 0.93737, docs/audit/2026-11-08-v0.15-soa-cband.md;
# 0.001 in z moves the peak ~1 nm). At z = 0.937: Eg(300 K) = 0.76837 eV, Ga0.440In0.560As0.937P0.063, Vegard
# mismatch on InP -3.5e-4. The gain peak sits 31 meV above the gap by band filling, so the GAP wavelength is
# 1.614 um ("Q1.61" in grower's notation); a "Q1.55" composition (1.55 um GAP, z = 0.875) peaks at ~1491 nm here.
INGAASP_LM_C_Z = 0.937

ALLOY_GAP_PARAMS = {
    "InGaAs_LM":  ("GaAs", "InAs", 0.53, 0.477),
    "InGaAsP_LM_C": ("InP", "InGaAs_LM", INGAASP_LM_C_Z, INGAASP_LM_BOWING_EV),
}

# Lattice constants [Angstrom] at 300 K: the a_lc rows of Vurgaftman Tables I (GaAs), II (AlAs), III (InAs),
# IV (GaP), VI (InP) (their T-slopes, 2.7-3.9e-5 A/K, are not used). Vegard's law (linear in the binary
# fractions) over these decides lattice matching.
LATTICE_CONSTANT_300K_A = {
    "GaAs": 5.65325,
    "AlAs": 5.6611,
    "GaP":  5.4505,
    "InAs": 6.0583,
    "InP":  5.8697,
}
# Two end points count as lattice matched for Eq. (5.4) when their Vegard lattice constants differ by at most
# 1e-3 (relative): 2x the largest mismatch among the paper's OWN named lattice-matched end points by the same
# rule -- Ga0.51In0.49P / GaAs +4.7e-4, Al0.48In0.52As / InP -3.5e-4, Ga0.47In0.53As / InP -3.0e-4 (Secs.
# V.A.1, V.B.1-2; computed from the table above). A non-matched pair (InGaAs_LM with InAs: 3.2 %) is not an
# Eq. (5.4) quaternary and is refused.
LATTICE_MATCH_RTOL = 1.0e-3

# Al(x)Ga(1-x)As Gamma-gap bowing C(x) = c0 + c1 x [eV], Vurgaftman Table XII (Aspnes et al.), T-independent.
ALGAAS_GAMMA_BOWING_EV = (-0.127, 1.310)


def algaas_gap_law(x_Al: float) -> Tuple[str, str, float, float]:
    """The Gamma-gap law of Al(x)Ga(1-x)As as a varshni_eg_ev ternary spec: ("GaAs", "AlAs", x, C(x)) with
    C(x) = -0.127 + 1.310 x eV (Table XII), i.e. A = GaAs, B = AlAs, x = the Al fraction in Eq. (4.1).
    The Gamma gap is the fundamental one only below the Gamma-X crossover (x = 0.38 at low T, 0.39 at
    300 K; module docstring)."""
    x = float(x_Al)
    if not (0.0 <= x <= 1.0):
        raise ValueError("algaas_gap_law: x must be in [0, 1], got {!r}".format(x_Al))
    c0, c1 = ALGAAS_GAMMA_BOWING_EV
    return ("GaAs", "AlAs", x, c0 + c1 * x)


def ingaasp_lm_gap_law(z: float) -> Tuple[str, str, float, float]:
    """The Gamma-gap law of Ga(x)In(1-x)As(z)P(1-z) lattice-matched to InP, x = 0.47 z, as a varshni_eg_ev
    spec: ("InP", "InGaAs_LM", z, 0.13) = Vurgaftman Eq. (5.4) between InP (Table VI) and Ga0.47In0.53As (the
    "InGaAs_LM" alloy law) with the Sec. V.B.1 bowing C = 0.13 eV, temperature-independent. z is the
    Ga0.47In0.53As fraction, which equals the As fraction y. 300 K gap: 1.353 eV at z = 0 down to 0.737 eV at
    z = 1 (the paper's end points). Use it for any InP-matched GaInAsP layer (active region, barrier, SCH);
    "InGaAsP_LM_C" is the composition behind the library's C-band SOA preset (BulkGainParams.c_band())."""
    z = float(z)
    if not (0.0 <= z <= 1.0):
        raise ValueError("ingaasp_lm_gap_law: z (the As fraction) must be in [0, 1], got {!r}".format(z))
    return ("InP", "InGaAs_LM", z, INGAASP_LM_BOWING_EV)


def ingaasp_lattice_mismatch(z: float, x_Ga: Optional[float] = None) -> float:
    """(a - a_InP) / a_InP of Ga(x)In(1-x)As(z)P(1-z) by Vegard's law over the four binaries at 300 K
    (LATTICE_CONSTANT_300K_A): a = x z a_GaAs + x (1-z) a_GaP + (1-x) z a_InAs + (1-x)(1-z) a_InP. x_Ga
    defaults to the paper's lattice-matched scheme x = 0.47 z (Sec. V.B.1). Negative = smaller than InP
    (tensile on an InP substrate). The exact Vegard match would be
    x = z (a_InAs - a_InP) / (z (a_InAs - a_GaAs) + (1-z) (a_InP - a_GaP))."""
    z = float(z)
    x = 0.47 * z if x_Ga is None else float(x_Ga)
    if not (0.0 <= z <= 1.0 and 0.0 <= x <= 1.0):
        raise ValueError("ingaasp_lattice_mismatch: compositions must be in [0, 1] (z={!r}, x={!r})".format(z, x))
    a = LATTICE_CONSTANT_300K_A
    a_q = (x * z * a["GaAs"] + x * (1.0 - z) * a["GaP"] + (1.0 - x) * z * a["InAs"]
           + (1.0 - x) * (1.0 - z) * a["InP"])
    return (a_q - a["InP"]) / a["InP"]


def _varshni_abc(material) -> Tuple[float, float, float]:
    if isinstance(material, str):
        if material not in VARSHNI_PARAMS:
            raise ValueError("varshni: '{}' is not a binary with a Varshni row (known: {}); an "
                             "alloy end point must be a binary".format(material,
                                                                      sorted(VARSHNI_PARAMS)))
        return VARSHNI_PARAMS[material]
    abc = tuple(float(x) for x in material)
    if len(abc) != 3:
        raise ValueError("varshni: custom binary must be (Eg0_eV, alpha_eV_K, beta_K)")
    return abc  # type: ignore[return-value]


def _lattice_constant_A(spec) -> Optional[float]:
    """Vegard lattice constant [A] at 300 K of a binary name, or of an ALLOY_GAP_PARAMS ternary whose end
    points are binary names; None when it is not known (a custom triple, a quaternary)."""
    if not isinstance(spec, str):
        return None
    if spec in LATTICE_CONSTANT_300K_A:
        return LATTICE_CONSTANT_300K_A[spec]
    alloy = ALLOY_GAP_PARAMS.get(spec)
    if alloy is None:
        return None
    A, B, x = alloy[0], alloy[1], float(alloy[2])
    if A in LATTICE_CONSTANT_300K_A and B in LATTICE_CONSTANT_300K_A:
        return (1.0 - x) * LATTICE_CONSTANT_300K_A[A] + x * LATTICE_CONSTANT_300K_A[B]
    return None


def _alloy_endpoint_terms(name: str, other):
    """Terms of an alloy END POINT of an Eq. (5.4) lattice-matched quaternary: `name` must be an
    ALLOY_GAP_PARAMS ternary of two binaries, lattice matched (LATTICE_MATCH_RTOL) to the `other` end point."""
    A, B = ALLOY_GAP_PARAMS[name][:2]
    if not all(isinstance(e, str) and e in VARSHNI_PARAMS for e in (A, B)):
        raise ValueError("varshni: end point '{}' is not a binary or a ternary of two binaries; Eq. (5.4) "
                         "combines binary or ternary end points only".format(name))
    a_n, a_o = _lattice_constant_A(name), _lattice_constant_A(other)
    if a_n is None or a_o is None:
        raise ValueError("varshni: end point '{}' is not a binary; an alloy end point is accepted only in a "
                         "lattice-matched quaternary (Vurgaftman Eq. (5.4)) whose other end point has a known "
                         "lattice constant (a binary or ternary NAME), got {!r}".format(name, other))
    mis = (a_n - a_o) / a_o
    if abs(mis) > LATTICE_MATCH_RTOL:
        raise ValueError("varshni: end point '{}' is not a binary; an alloy end point is accepted only in a "
                         "lattice-matched quaternary (Vurgaftman Eq. (5.4)), and '{}' ({:.5f} A) and {!r} "
                         "({:.5f} A) differ by {:.3g} % (> {:.3g} %)".format(
                             name, name, a_n, other, a_o, 100.0 * mis, 100.0 * LATTICE_MATCH_RTOL))
    return _gap_terms(name)


def _gap_terms(material) -> Tuple[Tuple[Tuple[float, Tuple[float, float, float]], ...], float]:
    """Resolve a material spec into ((weight, (Eg0, alpha, beta)), ...) and a constant bowing
    offset [eV], so Eg(T) = sum w (Eg0 - alpha T^2/(T + beta)) - offset. A binary / custom triple
    is one term of weight 1 and no offset (bit-identical to the bare Varshni law); a ternary is two
    terms (unchanged since v0.14). A lattice-matched quaternary (Eq. (5.4): an end point that is an
    ALLOY_GAP_PARAMS ternary name) flattens to the binaries of its end points, weights multiplied and
    bowing offsets combined: offset = z (1-z) C_ab + (1-z) offset_a + z offset_b."""
    if isinstance(material, str):
        if material in ALLOY_GAP_PARAMS:
            material = ALLOY_GAP_PARAMS[material]
        elif material not in VARSHNI_PARAMS:
            raise ValueError("varshni: unknown material '{}' (known: {}); pass a custom "
                             "(Eg0_eV, alpha_eV_K, beta_K) binary or (A, B, x, C_eV) ternary "
                             "instead".format(material,
                                              sorted(VARSHNI_PARAMS) + sorted(ALLOY_GAP_PARAMS)))
        else:
            return ((1.0, VARSHNI_PARAMS[material]),), 0.0
    spec = tuple(material)
    if len(spec) == 4:
        A, B, x, C = spec
        x, C = float(x), float(C)
        if not (0.0 <= x <= 1.0):
            raise ValueError("varshni: ternary composition x must be in [0, 1], got {}".format(x))
        a_alloy = isinstance(A, str) and A in ALLOY_GAP_PARAMS
        b_alloy = isinstance(B, str) and B in ALLOY_GAP_PARAMS
        if not (a_alloy or b_alloy):                    # Eq. (4.1) ternary: the v0.14 path, unchanged
            return ((1.0 - x, _varshni_abc(A)), (x, _varshni_abc(B))), x * (1.0 - x) * C
        # Eq. (5.4): a lattice-matched quaternary of two end points (binary or ternary)
        t_a, off_a = _alloy_endpoint_terms(A, B) if a_alloy else (((1.0, _varshni_abc(A)),), 0.0)
        t_b, off_b = _alloy_endpoint_terms(B, A) if b_alloy else (((1.0, _varshni_abc(B)),), 0.0)
        terms = tuple(((1.0 - x) * w, abc) for w, abc in t_a) + tuple((x * w, abc) for w, abc in t_b)
        return terms, x * (1.0 - x) * C + (1.0 - x) * off_a + x * off_b
    return ((1.0, _varshni_abc(spec)),), 0.0


def varshni_eg_ev(T_K: float, material: Union[str, Tuple] = "InAs") -> float:
    """Gamma band gap Eg(T) [eV] -- THE temperature law of the gap for the package (optics.soa's qw_gain
    band_edge_Eg_J and qd_params_at_temperature, and the carriers' BandTemperature, all call it).
    material is:
      * a binary name in VARSHNI_PARAMS ("GaAs", "AlAs", "InAs", "InP") or a custom (Eg0_eV, alpha_eV_K,
        beta_K) triple -> the Varshni law Eg(T) = Eg0 - alpha T^2/(T + beta);
      * a ternary name in ALLOY_GAP_PARAMS ("InGaAs_LM" = Ga0.47In0.53As on InP), algaas_gap_law(x), or a
        custom (A, B, x, C_eV) tuple (A, B binary names or triples) -> Vurgaftman, Meyer & Ram-Mohan,
        JAP 89, 5815 (2001) Eq. (4.1) over the binary Varshni laws with a temperature-independent
        bowing: Eg(T) = (1-x) Eg_A(T) + x Eg_B(T) - x (1-x) C;
      * a lattice-matched quaternary: the same (A, B, z, C_eV) form with an ALLOY_GAP_PARAMS ternary NAME
        as an end point, lattice matched to the other one (Eq. (5.4)) -- ingaasp_lm_gap_law(z) for GaInAsP
        on InP, or the C-band SOA preset "InGaAsP_LM_C" (z = INGAASP_LM_C_Z).
    Use the alloy rule (not a fitted Varshni triple) for any ternary: Vurgaftman tabulates Varshni
    parameters for binaries only (Tables I-IX) and bowings for alloys (Tables XII-XXX). InGaAs_LM:
    GaAs Table I + InAs Table III + C = 0.477 eV Table XIII -> 0.816 eV at 0 K, 0.737 eV at 300 K."""
    terms, offset = _gap_terms(material)
    T = float(T_K)
    eg = 0.0
    for w, (Eg0, a, b) in terms:
        eg += w * (Eg0 - a * T * T / (T + b))
    return float(eg - offset)


def d_eg_dT_ev_per_K(T_K: float, material: Union[str, Tuple] = "InAs") -> float:
    """Temperature slope dEg/dT [eV/K] of varshni_eg_ev (<= 0; the gap narrows as T rises): per
    binary term -alpha T (T + 2 beta)/(T + beta)^2 (hand-derived from d/dT[-a T^2/(T+b)]), weighted
    by the alloy fractions; the bowing is temperature-independent so it does not enter."""
    terms, _offset = _gap_terms(material)
    T = float(T_K)
    d = 0.0
    for w, (_Eg0, a, b) in terms:
        d += w * (-a * T * (T + 2.0 * b) / (T + b) ** 2)
    return float(d)


# ---- the opt-in band-temperature model (v0.15 plan 4.3.3; P3 item H2) -----------------------------------
BAND_ANCHORS = ("reference", "absolute")


@dataclass(frozen=True)
class BandTemperature:
    """Opt-in temperature model of one semiconductor's band data (module docstring has the physics and
    which option to use when). Attach it as TransportModel(band_temperature=...).

    gap_law           : any varshni_eg_ev material spec -- a VARSHNI_PARAMS binary name ("GaAs", "AlAs",
                        ...), a custom (Eg0_eV, alpha_eV_K, beta_K) triple, "InGaAs_LM", algaas_gap_law(x) or
                        a custom (A, B, x, C_eV) ternary.
    affinity_fraction : gamma_chi in [0, 1], REQUIRED (no default): the share of the gap change that moves
                        the conduction edge (chi(T) = chi_ref - gamma_chi dE_g). 1 = valence-anchored
                        (Vurgaftman 2001, Sec. VI: VBOs independent of T), 1/2 = symmetric (Sentaurus default
                        affinity model, secondary source). 0 = the affinity does not move.
    anchor            : "reference" (default, decision 32): E_g(T) = band_gap_eV + V(T) - V(T_ref), bit-identical
                        to band_gap_eV at T_ref; "absolute": E_g(T) = V(T) for 0 K data (band_gap_eV None).
    derive_n_i        : False (default) -> n_i_m3 is used as given (your value AT the device T); True ->
                        n_i_m3 is the value at T_ref and n_i(T) is derived consistently with E_g(T)
                        (bit-identical at T_ref). The recommended "own n_i(T)" for a paired BGN override.
    source            : free-text citation of the parameters (recorded, not read).
    T_ref is TransportModel.chi_reference_T_K (None = T_REF = 300 K): the temperature the material's
    band_gap_eV, chi_eV and n_i_m3 are quoted at."""
    gap_law: Union[str, tuple]
    affinity_fraction: float
    anchor: str = "reference"
    derive_n_i: bool = False
    source: str = ""

    def __post_init__(self):
        law = self.gap_law
        if not isinstance(law, str):
            law = tuple(law)
            object.__setattr__(self, "gap_law", law)
        _gap_terms(law)                                     # resolvable (raises ValueError naming the spec)
        g = self.affinity_fraction
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not (math.isfinite(float(g))
                                                                           and 0.0 <= float(g) <= 1.0):
            raise ValueError("BandTemperature.affinity_fraction (gamma_chi) must be a number in [0, 1] (1 = "
                             "valence-anchored, Vurgaftman 2001; 1/2 = symmetric, Sentaurus); got {!r}".format(g))
        object.__setattr__(self, "affinity_fraction", float(g))
        if self.anchor not in BAND_ANCHORS:
            raise ValueError("BandTemperature.anchor must be one of {}; got {!r}".format(BAND_ANCHORS, self.anchor))
        if not isinstance(self.derive_n_i, bool):
            raise TypeError("BandTemperature.derive_n_i must be a bool, got {!r}".format(self.derive_n_i))

    def gap_law_eV(self, T_K: float) -> float:
        """V(T): the law's own (absolute) gap at T [eV]."""
        return varshni_eg_ev(T_K, self.gap_law)

    def gap_change_eV(self, T_K: float, T_ref_K: float) -> float:
        """V(T) - V(T_ref) [eV] -- exactly 0.0 at T == T_ref (the same float evaluated twice)."""
        return varshni_eg_ev(T_K, self.gap_law) - varshni_eg_ev(T_ref_K, self.gap_law)


class BandData(NamedTuple):
    """Band data of one material at one temperature (eV, eV, m^-3); a field is None where the material
    leaves it None."""
    E_g_eV: Optional[float]
    chi_eV: Optional[float]
    n_i_m3: Optional[float]


def reference_temperature_K(tr) -> float:
    """T_ref [K] of a TransportModel's band data: chi_reference_T_K, None -> T_REF."""
    t = getattr(tr, "chi_reference_T_K", None)
    return T_REF if t is None else float(t)


def transport_temperature_K(tr) -> float:
    """The temperature [K] the builders run a material's region at: BipolarOptions.T_K (bipolar_dd only, the
    ONE temperature source that also sets the region V_t), else T_REF."""
    t = getattr(getattr(tr, "bipolar_options", None), "T_K", None)
    return T_REF if t is None else float(t)


def _thermal_voltage(T: float) -> float:
    return KB * T / Q_E                     # == constants.V_T bit for bit at T_REF


def band_data_at_T(tr, T_K: float) -> BandData:
    """(E_g, chi, n_i) of TransportModel `tr` at temperature T_K -- THE kernel the 2-D and 3-D builders read
    band data through (Anderson frame chi, the bipolar heterojunction band-gap check, the bipolar n_i).

    Without a band-temperature model (tr.band_temperature None, the default) it returns the material's
    (band_gap_eV, chi_eV, n_i_m3) UNTOUCHED at any T -- the v0.14 behaviour, bit-identical. With one, the
    module docstring formulas, anchored at T_ref = reference_temperature_K(tr): at T == T_ref every field
    equals the input bit for bit ("reference" anchor; "absolute" returns the law's gap there)."""
    bt = getattr(tr, "band_temperature", None)
    if bt is None:
        return BandData(tr.band_gap_eV, tr.chi_eV, tr.n_i_m3)
    T = float(T_K)
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError("band_data_at_T: T_K must be finite and > 0, got {!r}".format(T_K))
    T_ref = reference_temperature_K(tr)
    d_eg = bt.gap_change_eV(T, T_ref)
    if bt.anchor == "reference":
        eg_ref = float(tr.band_gap_eV)
        eg = eg_ref + d_eg
    else:
        eg_ref = bt.gap_law_eV(T_ref)
        eg = bt.gap_law_eV(T)
    chi = None if tr.chi_eV is None else float(tr.chi_eV) - bt.affinity_fraction * d_eg
    ni = tr.n_i_m3
    if bt.derive_n_i and ni is not None:
        ni = float(ni) * ((T / T_ref) ** 1.5 * math.exp(eg_ref / (2.0 * _thermal_voltage(T_ref))
                                                        - eg / (2.0 * _thermal_voltage(T))))
    return BandData(float(eg), chi, ni)


def band_gap_ratio(tr, T_K: float, T0_K: float = T_REF) -> Optional[float]:
    """E_g(T) / E_g(T0) from the material's band-temperature model, or None when it has none -- the accessor
    the BTBT temperature scaling reads (plan 5.2 item 2; v0.15 item 4.1.2: carriers.tunneling.btbt_for_material
    for sets with scale_B_with_gap=True; without a model setup_btbt keeps its v0.14 warning). T0_K defaults
    to T_REF, the temperature the shipped BTBT parameter sets are quoted at."""
    if getattr(tr, "band_temperature", None) is None:
        return None
    return band_data_at_T(tr, T_K).E_g_eV / band_data_at_T(tr, T0_K).E_g_eV


def check_band_temperature(tr) -> None:
    """TransportModel.__post_init__ validation of a band_temperature model (raises ValueError / TypeError):
    the anchor's inputs exist and nothing given would be silently unused."""
    bt = tr.band_temperature
    if not isinstance(bt, BandTemperature):
        raise TypeError("TransportModel.band_temperature must be a materials.BandTemperature, got {!r}".format(
            type(bt).__name__))
    T_ref = reference_temperature_K(tr)
    if bt.anchor == "reference" and tr.band_gap_eV is None:
        raise ValueError("BandTemperature(anchor='reference') shifts band_gap_eV by V(T) - V(T_ref); set "
                         "TransportModel.band_gap_eV (eV, at chi_reference_T_K = {:.6g} K), or use "
                         "anchor='absolute' for a gap law quoted at 0 K".format(T_ref))
    if bt.anchor == "absolute" and tr.band_gap_eV is not None:
        raise ValueError("BandTemperature(anchor='absolute') takes the gap from the law itself (E_g(T) = V(T)); "
                         "leave TransportModel.band_gap_eV None (got {!r}: it would be a second, unused gap). "
                         "For a gap quoted at T_ref use anchor='reference'.".format(tr.band_gap_eV))
    if bt.derive_n_i:
        if tr.n_i_m3 is None:
            raise ValueError("BandTemperature(derive_n_i=True) scales n_i_m3 from T_ref; set "
                             "TransportModel.n_i_m3 (the intrinsic density at {:.6g} K)".format(T_ref))
        if T_ref == 0.0:
            raise ValueError("BandTemperature(derive_n_i=True) cannot anchor n_i at T_ref = 0 K (n_i(0) = 0); "
                             "quote n_i_m3 at a finite chi_reference_T_K")
