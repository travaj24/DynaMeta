"""REL1: gate-oxide time-dependent dielectric breakdown (TDDB). The ITO-ENZ modulator's primary
CATASTROPHIC lifetime limiter: a few volts over a 5-20 nm gate oxide is 1-10 MV/cm, squarely in the
TDDB window, and Joule/optical self-heating shortens tBD through the Arrhenius factor.

Model: the standard SEPARABLE E-model (industry default; McPherson-class field acceleration with a
temperature-independent field term, which avoids the negative-activation pathology of folding the
field into the activation energy):

    tBD(E_ox, T) = tau0 * exp(-gamma_E * E_ox[MV/cm]) * exp(Ea / (kB * T))

with gamma_E the field-acceleration coefficient (literature ~1-2 DECADES per MV/cm for thin SiO2,
i.e. gamma_E = ln(10)*1..2 ~ 2.3-4.6 per MV/cm) and Ea ~ 0.6-0.9 eV the thermal activation. The 1/E
(anode-hole-injection) alternative tBD = tau0 * exp(G / E_ox) * exp(Ea/kBT), G ~ 300-400 MV/cm, is
offered for the high-field/thin-oxide regime. Weibull AREA scaling converts a single-cell median to
an array's characteristic life: t63(A2) = t63(A1) * (A1/A2)^(1/beta), beta ~ 1-2 for thin oxides
(weakest-link percolation statistics).

tau0 is CALIBRATION-bearing: anchor it with one measured/qualified (E, T, tBD) point via
TddbParams.calibrated(); the model then EXTRAPOLATES along the documented acceleration slopes.
Pure numpy; SI in/out except E expressed internally in MV/cm (1 MV/cm = 1e8 V/m). Oracles in
validation/reliability_tddb.py (closed-form acceleration ratios + a percolation-ODE numeric
cross-check + the literature slope band).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dynameta.constants import KB_EV_K   # eV/K, single source (audit 6.3)
_MV_CM = 1.0e8                      # 1 MV/cm in V/m


def _check_common(T_K, tau0_s, Ea_eV):
    if np.any(np.asarray(T_K, dtype=np.float64) <= 0.0):
        raise ValueError("TDDB: T_K must be > 0")
    if not (tau0_s > 0.0):
        raise ValueError("TDDB: tau0_s must be > 0")
    if Ea_eV < 0.0:
        raise ValueError("TDDB: Ea_eV must be >= 0")


def tbd_e_model(E_ox_V_m, T_K, *, tau0_s: float, gamma_E_per_MV_cm: float, Ea_eV: float):
    """Median time-to-breakdown [s] under the separable E-model (see module docstring). E_ox in V/m
    (>= 0); broadcasts over E/T arrays."""
    _check_common(T_K, tau0_s, Ea_eV)
    if gamma_E_per_MV_cm < 0.0:
        raise ValueError("TDDB: gamma_E_per_MV_cm must be >= 0 (field ACCELERATES breakdown)")
    E = np.asarray(E_ox_V_m, dtype=np.float64)
    if np.any(E < 0.0):
        raise ValueError("TDDB: E_ox_V_m must be >= 0 (use the field magnitude)")
    T = np.asarray(T_K, dtype=np.float64)
    return tau0_s * np.exp(-gamma_E_per_MV_cm * (E / _MV_CM)) * np.exp(Ea_eV / (KB_EV_K * T))


def tbd_one_over_e(E_ox_V_m, T_K, *, tau0_s: float, G_MV_cm: float, Ea_eV: float = 0.0):
    """Median time-to-breakdown [s] under the 1/E (anode-hole-injection) model:
    tBD = tau0 * exp(G / E[MV/cm]) * exp(Ea/kBT). Requires E > 0 (the model diverges at E -> 0,
    which is the PHYSICAL immortal limit)."""
    _check_common(T_K, tau0_s, Ea_eV)
    if G_MV_cm <= 0.0:
        raise ValueError("TDDB: G_MV_cm must be > 0")
    E = np.asarray(E_ox_V_m, dtype=np.float64)
    if np.any(E <= 0.0):
        raise ValueError("TDDB 1/E model: E_ox_V_m must be > 0 (E -> 0 is the immortal limit)")
    T = np.asarray(T_K, dtype=np.float64)
    return tau0_s * np.exp(G_MV_cm / (E / _MV_CM)) * np.exp(Ea_eV / (KB_EV_K * T))


def weibull_area_scale(t63_ref_s, area_ref_m2: float, area_m2: float, beta: float):
    """Weibull weakest-link AREA scaling of the characteristic (63.2%) life:
    t63(A) = t63(A_ref) * (A_ref / A)^(1/beta). A larger device (or an N-element array, A = N*A_cell)
    fails sooner; beta ~ 1-2 for thin-oxide percolation."""
    if not (area_ref_m2 > 0.0 and area_m2 > 0.0):
        raise ValueError("TDDB: areas must be > 0")
    if not (beta > 0.0):
        raise ValueError("TDDB: Weibull shape beta must be > 0")
    return np.asarray(t63_ref_s, dtype=np.float64) * (area_ref_m2 / area_m2) ** (1.0 / beta)


@dataclass(frozen=True)
class TddbParams:
    """Separable-E-model parameter set. gamma_E ~ 2.3-4.6 /(MV/cm) (1-2 decades/(MV/cm), thin SiO2);
    Ea ~ 0.6-0.9 eV; beta ~ 1-2 (Weibull shape). tau0 is calibration-bearing -- prefer .calibrated()."""
    tau0_s: float = 1.0
    gamma_E_per_MV_cm: float = 3.0
    Ea_eV: float = 0.7
    beta: float = 1.5

    def tbd_s(self, E_ox_V_m, T_K):
        return tbd_e_model(E_ox_V_m, T_K, tau0_s=self.tau0_s,
                           gamma_E_per_MV_cm=self.gamma_E_per_MV_cm, Ea_eV=self.Ea_eV)

    @classmethod
    def calibrated(cls, *, E_ox_V_m: float, T_K: float, tbd_s: float,
                   gamma_E_per_MV_cm: float = 3.0, Ea_eV: float = 0.7,
                   beta: float = 1.5) -> "TddbParams":
        """Anchor tau0 on ONE measured/qualified stress point (E, T, tBD); the slopes gamma_E/Ea then
        extrapolate to other conditions."""
        base = cls(tau0_s=1.0, gamma_E_per_MV_cm=gamma_E_per_MV_cm, Ea_eV=Ea_eV, beta=beta)
        t_unit = float(base.tbd_s(E_ox_V_m, T_K))            # tBD with tau0 = 1
        if not (tbd_s > 0.0):
            raise ValueError("TDDB: calibration tbd_s must be > 0")
        return replace(base, tau0_s=tbd_s / t_unit)


def oxide_stress_from_electrothermal(et_result, layer_name: str):
    """Adapter: pull the (|E_ox| [V/m], T [K]) stress pair for `layer_name` out of an
    ElectroThermalResult (carriers.electrothermal) -- duck-typed so this module never imports the
    NGSolve-backed solver. Returns (E_ox_V_m, T_K).

    audit C4-9: the stress statistic is the layer-mean |E_z| (mean_absEz_per_layer), NOT
    |mean E_z| -- the signed mean is exactly ZERO for a +/-V split-gate profile and
    generally understates the percolation-driving field for any sign-changing lateral
    profile, silently overstating time-to-breakdown exponentially. The abs-mean still
    understates the true hot-spot peak on nonuniform profiles; a sampled per-layer peak
    statistic is the tracked refinement."""
    names = [L.name for L in et_result.layers]
    if layer_name not in names:
        raise ValueError("oxide_stress_from_electrothermal: layer {!r} not in {}".format(
            layer_name, names))
    i = names.index(layer_name)
    E_res = et_result.E_result
    if hasattr(E_res, "mean_absEz_per_layer"):
        ez = float(E_res.mean_absEz_per_layer()[i])
    else:                                                   # duck-typed stub without the C4-9 stat
        ez = float(abs(E_res.mean_Ez_per_layer()[i]))
    return ez, float(et_result.T_per_layer[i])
