"""Polarization-dependent gain (PDG) / polarization hole burning (PHB) for the fiber amplifier
(dossier Module 6; Mazurczyk & Zyskind, IEEE Photon. Technol. Lett. 6:616 (1994)).

Physics: Er ions carry anisotropic cross-section tensors at random site orientations. A
POLARIZED saturating signal preferentially bleaches the ions aligned with it, so a probe (or
the ASE) sees MORE gain in the ORTHOGONAL polarization. Measured magnitude: PDG ~ 0.026 dB per
dB of gain compression (~0.08 dB at 3 dB compression), growing to 0.2-0.4 dB deep in
saturation; a cascade of N amplifiers random-walks to ~sqrt(N) x that (aligned worst case ~N);
polarization scrambling faster than the ~10 ms Er lifetime averages the anisotropy away.

TWO VIEWS, CONNECTED:
  * pdg_db: the MEASURED slope anchor (eps = 0.026 dB/dB) -- use when you have a gain
    compression number and want the standard engineering estimate.
  * TwoPolSaturation: the lumped two-polarization-mode cross-saturation model, S_a = P_a +
    f P_b + (1+f) P_ase/2 (mirror for b), g_i = g0/(1 + S_i/P_sat). f is the ENSEMBLE
    cross-saturation factor: in the small-compression signal-dominated limit PDG/DeltaG_comp ->
    (1 - f) exactly, so the measured slope pins f ~ 1 - 0.026 = 0.974. NOTE the microscopic
    SAME-ION orthogonal-saturation factor is ~2/3 (an aligned dipole saturates its orthogonal
    response ~2/3 as strongly), but averaging over the random site orientations dilutes the
    ensemble anisotropy by another order of magnitude -- the lumped f is the ORIENTATION-AVERAGED
    number and must be calibrated from the measured slope (f_from_pdg_slope), not set to 2/3.

Bookkeeping contract (adopt everywhere): a coherent signal is fully polarized (one pol mode);
ASE is unpolarized (P_ase/2 in each mode, m_modes = 2 -- the AseBand convention). ASE saturates
both polarizations symmetrically and contributes NO PDG by itself, but the signal's anisotropic
saturation makes the ASE accumulate preferentially ORTHOGONAL to the signal.

Pure numpy; SI units; dB where suffixed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["pdg_db", "pdg_cascade_db", "f_from_pdg_slope", "TwoPolSaturation"]

PDG_SLOPE_DB_PER_DB = 0.026        # Mazurczyk-Zyskind measured slope (0.02-0.03)


def pdg_db(compression_dB: float, *, eps: float = PDG_SLOPE_DB_PER_DB) -> float:
    """PDG [dB] from the gain compression [dB]: the measured Mazurczyk-Zyskind anchor
    PDG = eps * DeltaG_comp (~0.08 dB at 3 dB compression). Maximum gain lies ORTHOGONAL to
    the saturating signal polarization."""
    return float(eps * max(float(compression_dB), 0.0))


def pdg_cascade_db(pdg_single_dB: float, n_amps: int, *, aligned: bool = False) -> float:
    """PDG of an N-amplifier chain [dB]: random inter-stage polarization walks give ~sqrt(N)
    growth; aligned=True is the worst case ~N (all hole-burning axes coincide)."""
    n = max(int(n_amps), 1)
    return float(pdg_single_dB * (n if aligned else np.sqrt(n)))


def f_from_pdg_slope(eps: float = PDG_SLOPE_DB_PER_DB) -> float:
    """Ensemble cross-saturation factor f pinned by the measured PDG slope: in the
    small-compression signal-dominated limit PDG/DeltaG = (1 - f), so f = 1 - eps."""
    return float(1.0 - eps)


@dataclass(frozen=True)
class TwoPolSaturation:
    """Lumped two-polarization-mode saturation model (module header). g0_dB = unsaturated gain;
    P_sat_W = the saturation power of the gain (output-referred, consistent with how the
    compression is measured); f = ensemble cross-saturation factor (default from the measured
    slope). All powers are at the saturating (output) reference plane."""
    g0_dB: float
    P_sat_W: float
    f: float = 1.0 - PDG_SLOPE_DB_PER_DB

    def __post_init__(self):
        if not (self.P_sat_W > 0.0 and 0.0 <= self.f <= 1.0):
            raise ValueError("TwoPolSaturation: P_sat_W > 0 and f in [0, 1]")

    def gains_dB(self, P_sig_W: float, *, P_ase_W: float = 0.0, sig_pol: str = "a"):
        """(G_a_dB, G_b_dB) with the signal polarized in 'a' (or 'b'). The signal saturates its
        own polarization with weight 1 and the orthogonal with weight f; unpolarized ASE splits
        P_ase/2 per mode and saturates both with weight (1 + f)/2 x total."""
        Ps, Pa = max(float(P_sig_W), 0.0), max(float(P_ase_W), 0.0)
        s_own = Ps + (1.0 + self.f) * Pa / 2.0
        s_orth = self.f * Ps + (1.0 + self.f) * Pa / 2.0
        g_own = self.g0_dB / (1.0 + s_own / self.P_sat_W)
        g_orth = self.g0_dB / (1.0 + s_orth / self.P_sat_W)
        return (g_own, g_orth) if sig_pol == "a" else (g_orth, g_own)

    def pdg_dB(self, P_sig_W: float, *, P_ase_W: float = 0.0) -> float:
        """PDG = G_orthogonal - G_co [dB] (positive: the orthogonal pol wins)."""
        ga, gb = self.gains_dB(P_sig_W, P_ase_W=P_ase_W)
        return float(gb - ga)

    def compression_dB(self, P_sig_W: float, *, P_ase_W: float = 0.0) -> float:
        """Co-polarized gain compression DeltaG = g0 - G_co [dB] at this operating point."""
        ga, _ = self.gains_dB(P_sig_W, P_ase_W=P_ase_W)
        return float(self.g0_dB - ga)
