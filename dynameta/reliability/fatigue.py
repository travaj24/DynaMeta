"""REL6: thermal-cycling fatigue from CTE mismatch. Cyclic dT (ambient swing or pulsed self-heating,
the dT amplitude extractable from an R5 ThermalTransientResult trace) strains the CTE-mismatched
film stack; DUCTILE metal follows Coffin-Manson, BRITTLE films (ITO, gate oxide) follow a
critical-stress / Weibull fracture model -- the two regimes must NOT be conflated (the
roadmap-corrected split):

    biaxial film stress:   sigma = (E_film / (1 - nu_film)) * (CTE_sub - CTE_film) * dT
    ductile Coffin-Manson: Nf = C * (d_eps_p)^(-1/c),  c = fatigue DUCTILITY exponent ~ 0.5-0.7
                           (so the exponent ON the plastic strain range is 1/c ~ 1.4-2.0)
    Norris-Landzberg AF:   from the primitive law Nf = C f^m dT^(-n) exp(Ea_K/Tmax)
                           (slower cycling = longer creep dwell = FEWER cycles to failure):
                           AF = Nf_use/Nf_test = (f_use/f_test)^m * (dT_test/dT_use)^n
                           * exp(Ea_K (1/Tmax_use - 1/Tmax_test)),  m ~ 1/3, n ~ 2,
                           Ea_K ~ 1414 K (SnPb baseline; re-fit per materials system).
                           NOTE the frequency ratio direction: a FASTER test (f_test > f_use)
                           contributes a factor < 1 (audit C4-1 fixed the inversion).
    brittle (Weibull):     P_survive(sigma) = exp(-(sigma/sigma0)^m_w); sigma >= sigma_crit -> cracks
                           on the FIRST excursion (no cycle accumulation)

DRIVER NOTE (D3 RESOLVED): MechanicalProps now lives on the MATERIAL schema
(dynameta/materials/mechanical.py, optional Material.mechanical field) and is RE-EXPORTED here for
backward compatibility -- the fatigue/stress-migration post-processors and the materials registry
share one table. Pure numpy; oracles in validation/reliability_fatigue.py.
"""

from __future__ import annotations

import numpy as np

from dynameta.materials.mechanical import MechanicalProps   # D3: promoted; re-exported (back-compat)

__all__ = ["MechanicalProps", "biaxial_stress_Pa", "coffin_manson_nf", "plastic_strain_range",
           "norris_landzberg_af", "brittle_survival", "cycles_to_failure"]


def biaxial_stress_Pa(film: MechanicalProps, cte_sub_per_K: float, dT_K) -> np.ndarray:
    """Equibiaxial thermal-mismatch film stress sigma = E/(1-nu) (CTE_sub - CTE_film) dT, tension
    positive. On cooling (dT < 0) the film ends up TENSILE when it shrinks MORE than the
    substrate (CTE_film > CTE_sub) -- e.g. metal films on Si; a film that shrinks less than the
    substrate is compressed. (Docstring sign-case corrected per audit; the formula is unchanged
    and was always correct.)"""
    dT = np.asarray(dT_K, dtype=np.float64)
    return (film.E_Pa / (1.0 - film.nu)) * (cte_sub_per_K - film.cte_per_K) * dT


def coffin_manson_nf(d_eps_plastic, *, C: float = 0.5, c_ductility: float = 0.6):
    """Ductile low-cycle fatigue life Nf = C * (d_eps_p)^(-1/c). d_eps_p = 0 -> inf (no plastic
    strain, no low-cycle fatigue). The exponent on strain is 1/c (~1.4-2.0), NOT c -- the
    audit-caught inversion."""
    if not (C > 0.0):
        raise ValueError("Coffin-Manson: C must be > 0")
    if not (0.0 < c_ductility <= 1.0):
        raise ValueError("Coffin-Manson: ductility exponent c must be in (0, 1] (typ. 0.5-0.7)")
    d = np.asarray(d_eps_plastic, dtype=np.float64)
    if np.any(d < 0.0):
        raise ValueError("Coffin-Manson: plastic strain range must be >= 0")
    with np.errstate(divide="ignore"):
        return np.where(d > 0.0, C * d ** (-1.0 / c_ductility), np.inf)


def plastic_strain_range(cte_film_per_K: float, cte_sub_per_K: float, dT_K: float, *,
                         eps_elastic_offset: float = 0.0) -> float:
    """Per-cycle plastic strain range d_eps_p = max(|dCTE| dT - eps_elastic, 0): the mismatch strain
    minus the elastically-accommodated part (offset 0 = fully plastic, the conservative default)."""
    if dT_K < 0.0 or eps_elastic_offset < 0.0:
        raise ValueError("fatigue: dT_K and eps_elastic_offset must be >= 0")
    return float(max(abs(cte_sub_per_K - cte_film_per_K) * dT_K - eps_elastic_offset, 0.0))


def norris_landzberg_af(*, f_use_Hz: float, f_test_Hz: float, dT_use_K: float, dT_test_K: float,
                        Tmax_use_K: float, Tmax_test_K: float, m: float = 1.0 / 3.0,
                        n: float = 2.0, Ea_K: float = 1414.0) -> float:
    """Norris-Landzberg acceleration factor AF = Nf_use / Nf_test, from the primitive law
    Nf = C f^m dT^(-n) exp(Ea_K/Tmax) with m = +1/3: FASTER cycling means SHORTER creep
    dwell per cycle, hence MORE cycles to failure, so a faster test (f_test > f_use)
    contributes a frequency factor (f_use/f_test)^m < 1 to AF. (Audit C4-1: the ratio was
    previously inverted, overestimating field life by (f_test/f_use)^(2m) -- 4x for a
    typical chamber-vs-field cadence -- systematically non-conservative.)"""
    for v, nm in ((f_use_Hz, "f_use"), (f_test_Hz, "f_test"), (dT_use_K, "dT_use"),
                  (dT_test_K, "dT_test"), (Tmax_use_K, "Tmax_use"), (Tmax_test_K, "Tmax_test")):
        if not (v > 0.0):
            raise ValueError("Norris-Landzberg: {} must be > 0".format(nm))
    return float((f_use_Hz / f_test_Hz) ** m * (dT_test_K / dT_use_K) ** n
                 * np.exp(Ea_K * (1.0 / Tmax_use_K - 1.0 / Tmax_test_K)))


def brittle_survival(sigma_Pa, *, sigma0_Pa: float, m_weibull: float):
    """Weibull fracture survival probability of a brittle film at peak stress sigma:
    P = exp(-(sigma/sigma0)^m). sigma = 0 -> 1 exactly."""
    if not (sigma0_Pa > 0.0 and m_weibull > 0.0):
        raise ValueError("brittle_survival: sigma0 and m must be > 0")
    s = np.asarray(sigma_Pa, dtype=np.float64)
    if np.any(s < 0.0):
        raise ValueError("brittle_survival: use the stress magnitude (>= 0)")
    return np.exp(-(s / sigma0_Pa) ** m_weibull)


def cycles_to_failure(film: MechanicalProps, cte_sub_per_K: float, dT_K: float, *,
                      C: float = 0.5, c_ductility: float = 0.6,
                      eps_elastic_offset: float = 0.0) -> float:
    """The film's cycling life at amplitude dT: BRITTLE first (one over-stress excursion cracks it:
    sigma >= sigma_crit -> 0 cycles), else ductile Coffin-Manson on the plastic mismatch strain."""
    sig = abs(float(biaxial_stress_Pa(film, cte_sub_per_K, dT_K)))
    if sig >= film.sigma_crit_Pa:
        return 0.0                                          # cracks on the first excursion
    d_eps = plastic_strain_range(film.cte_per_K, cte_sub_per_K, dT_K,
                                 eps_elastic_offset=eps_elastic_offset)
    return float(coffin_manson_nf(d_eps, C=C, c_ductility=c_ductility))
