"""The shared rare-earth rate-equation CORE (Giles-Desurvire two-level, extends to the Yb
quasi-three-level): given the local optical powers of every channel (pump / signal / ASE bins)
at one z, compute the steady-state metastable fraction nbar2 = N2/n_t and the per-channel local
gain coefficient + ASE spontaneous source. These are the REFERENCE (ideal-model, unit-tested) forms of the pointwise physics;
the steady-state solver owns its own optimized copy (which additionally carries the opt-in
concentration effects: active/dark ion split, photodarkening). Audit S3-8: the former dP_dz /
ase_source_per_m exports duplicated the solver inline physics and had silently diverged from it
(n_t vs active density); they were removed -- build on FiberAmplifier.solve(), not on these.

docs/fiber_amp_model_spec.md sec.1. Pure numpy; SI units; exp(-i omega t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
from dynameta.optics.fiber_amp.waveguide import FiberSpec, overlap_gamma

__all__ = ["ChannelSet", "metastable_fraction", "gain_coeff_per_m"]


@dataclass(frozen=True)
class ChannelSet:
    """The optical channels sharing the fiber at one z (or precomputed spectra for all z). Each
    channel k has a wavelength lambda_m[k], a propagation direction u[k] = +1/-1, and a flag
    is_ase[k] (True -> carries the spontaneous-emission source). Powers are supplied per-call
    (they change along z); the per-channel wavelength-dependent cross-sections and overlap are
    cached here since they do NOT change along a uniform fiber."""
    lambda_m: np.ndarray                       # (K,) channel wavelengths [m]
    u: np.ndarray                              # (K,) +1 forward / -1 backward
    is_ase: np.ndarray                         # (K,) bool
    dnu_hz: np.ndarray                         # (K,) ASE bin width [Hz] (0 for pump/signal)
    sigma_a: np.ndarray                        # (K,) absorption cross-section [m^2]
    sigma_e: np.ndarray                        # (K,) emission cross-section [m^2]
    gamma: np.ndarray                          # (K,) mode/dopant overlap
    loss_per_m: np.ndarray                     # (K,) background loss l_k [1/m]
    tau_s: float                               # upper-state lifetime [s] (from the ion)
    sigma_esa: np.ndarray = None               # (K,) excited-state-absorption cross-section [m^2]

    def __post_init__(self):
        if self.sigma_esa is None:
            object.__setattr__(self, "sigma_esa", np.zeros_like(self.sigma_a))

    @staticmethod
    def build(ion: RareEarthIon, fiber: FiberSpec, lambda_m, u, *, is_ase=None,
              dnu_hz=None) -> "ChannelSet":
        lam = np.atleast_1d(np.asarray(lambda_m, dtype=np.float64))
        u = np.broadcast_to(np.asarray(u, dtype=np.float64), lam.shape).astype(np.float64).copy()
        is_ase = (np.zeros(lam.shape, bool) if is_ase is None
                  else np.broadcast_to(np.asarray(is_ase, bool), lam.shape).copy())
        dnu = (np.zeros(lam.shape) if dnu_hz is None
               else np.broadcast_to(np.asarray(dnu_hz, float), lam.shape).astype(np.float64).copy())
        return ChannelSet(lam, u, is_ase, dnu,
                          np.asarray(ion.sigma_a.sigma(lam), float),
                          np.asarray(ion.sigma_e.sigma(lam), float),
                          np.asarray(overlap_gamma(fiber, lam), float),
                          fiber.loss_per_m(lam), float(ion.tau_s),
                          np.asarray(ion.sigma_esa_of(lam), float))

    @property
    def nu_hz(self) -> np.ndarray:
        return C_LIGHT / self.lambda_m


def metastable_fraction(ch: ChannelSet, powers_W, fiber: FiberSpec, *,
                        upconversion_C_up: float = 0.0) -> float:
    """Steady-state fractional upper-level population nbar2 = N2/n_t at one z from the local
    channel powers (docs sec.1, finding [1]):

        nbar2 = (tau * R_a) / (1 + tau * (R_a + R_e)),
        R_a = SUM_k sigma_a,k Gamma_k P_k / (h nu_k A_dope)   [1/s]  (absorption/pump rate)
        R_e = SUM_k sigma_e,k Gamma_k P_k / (h nu_k A_dope)   [1/s]  (stimulated-emission rate)

    A_dope = pi b^2 is the doped area; Gamma_k P_k / A_dope is the average intensity the ions
    see. upconversion_C_up > 0 adds a cooperative-upconversion loss -C_up N2^2 to the balance
    (Phase 5): nbar2 then solves the quadratic tau^-1 nbar2 + C_up n_t nbar2^2 = R_a (1-nbar2)
    - R_e nbar2, so the closed form gets a sqrt branch (off by default -> the linear result)."""
    P = np.asarray(powers_W, dtype=np.float64)
    A = fiber.a_dope_m2
    flux = ch.gamma * P / (H_PLANCK * ch.nu_hz * A)   # (K,) overlap photon-flux per ion [1/s]/sigma
    R_a = float(np.sum(ch.sigma_a * flux))
    R_e = float(np.sum(ch.sigma_e * flux))
    tau_s = ch.tau_s
    if upconversion_C_up <= 0.0:
        return tau_s * R_a / (1.0 + tau_s * (R_a + R_e))
    # cooperative upconversion: balance R_a(1-n) = n/tau + R_e n + C_up n_t n^2 -> quadratic in n
    A2 = upconversion_C_up * fiber.n_t_m3
    B = 1.0 / tau_s + R_a + R_e
    disc = B * B - 4.0 * A2 * (-R_a)
    return float((-B + np.sqrt(disc)) / (2.0 * A2))


def gain_coeff_per_m(ch: ChannelSet, nbar2: float, fiber: FiberSpec) -> np.ndarray:
    """Per-channel local NET gain coefficient g_k [1/m] (docs sec.1):
        g_k = Gamma_k n_t [sigma_e,k nbar2 - sigma_a,k (1 - nbar2) - sigma_esa,k nbar2] - l_k.
    Positive = amplification, negative = net absorption. The first bracket terms are the Giles
    g*_k nbar2 - alpha_k (1 - nbar2); the sigma_esa nbar2 term is excited-state absorption (a
    parasitic loss from the excited population, zero unless the ion carries an ESA spectrum)."""
    return (ch.gamma * fiber.n_t_m3
            * (ch.sigma_e * nbar2 - ch.sigma_a * (1.0 - nbar2) - ch.sigma_esa * nbar2)
            - ch.loss_per_m)
