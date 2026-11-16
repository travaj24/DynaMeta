"""
Screening length of a semiconductor transport model, and the pure helpers of the default z mesh grading
of the layered DEVSIM builder (geometry.specs.Mesh2DSpec.semiconductor_refinement="screening").

Nothing here imports DEVSIM: carriers.devsim_layered (which does) re-exports every name, so the mesh rule
and the screening length are importable and testable on an install without the optional solver stack.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dynameta.carriers import physics_equilibrium as PE
from dynameta.materials.band_temperature import transport_temperature_K


def _needs_screening(spec, L: float) -> bool:
    """True when the legacy lines of a semiconductor layer with screening length L are coarser than the
    screening grading asks for (face spacing above screening_interface_factor L, or bulk spacing above
    screening_ps_factor L)."""
    return not (spec.interface_ps_m <= spec.screening_interface_factor * L
                and spec.coarse_ps_m <= spec.screening_ps_factor * L)


def _graded_offsets(h0: float, h1: float, D: float) -> np.ndarray:
    """Offsets 0 = s_0 < s_1 < ... < s_K = D whose steps grow geometrically from ~h0 to ~h1 (constant ratio,
    rescaled so the last offset is exactly D); uniform steps <= h0 when h1 <= h0."""
    if D <= h0:
        return np.array([0.0, D])
    if h1 <= h0 * (1.0 + 1e-12) or D <= h1:
        k = int(np.ceil(D / h0 - 1e-9))
        return np.linspace(0.0, D, k + 1)
    r = (D - h0) / (D - h1)                         # geometric series h0 .. h1 summing to D
    k = max(1, int(round(np.log(h1 / h0) / np.log(r))) + 1)
    steps = h0 * r ** np.arange(k)
    steps *= D / steps.sum()
    return np.r_[0.0, np.cumsum(steps)[:-1], D]


def _interface_resolution(x, y, n) -> Optional[float]:
    """max |ln(n_{k+1} / n_k)| over z-adjacent node pairs of every node column (unique x, binned like
    core.resample) of a region: the largest density ratio across one z edge. The steepest edges sit on
    the faces of an accumulation or depletion layer, and the value scales with the spacing there, so it
    compares meshes (diagnostic only; no threshold is applied). MEASURED on the generic gated degenerate film of
    tests/test_mesh_screening_grading.py: 0.022 (accumulation) / 0.150 (depletion, a steep Boltzmann tail) with
    the default grading, 0.350 / 0.707 with the legacy lines. None when the region has no positive-density z edge."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    kx = np.round(x / 1e-15).astype(np.int64)
    o = np.lexsort((y, kx))                                  # by column, then ascending y
    kx, nn = kx[o], n[o]
    same = kx[1:] == kx[:-1]
    ok = same & (nn[1:] > 0.0) & (nn[:-1] > 0.0)
    if not np.any(ok):
        return None
    return float(np.max(np.abs(np.log(nn[1:][ok] / nn[:-1][ok]))))
# A semiconductor background density below this fraction of its band density is non-degenerate to
# far better than the mesh rule needs: F_1/2'(eta) = F_1/2(eta) = n/N (and invert_F12 would leave its
# bracket for an extremely dilute density).
_NONDEGENERATE_X = 1e-6


def screening_length_m(transport) -> float:
    """Screening length [m] of a semiconductor TransportModel at its background density n_bg:
    L_s^2 = eps_s eps0 kT / (q^2 N F'(eta_bg)), i.e. eps_s eps0 / (q^2 dn/dE_F) of the closure
    n = N F_1/2(eta) the layered builder solves (N the effective band density at the region's
    temperature from the DOS mass at n_bg, eta_bg = F_1/2^-1(n_bg / N), F' = dF_1/2/deta of the same
    Aymerich-Humet F_1/2). Non-degenerate: the Debye length sqrt(eps_s eps0 kT / (q^2 n_bg));
    degenerate: the Thomas-Fermi length. A bipolar acceptor-doped material (uniform doping) uses
    its majority band (N_v from dos_mass_p_kg); carrier_statistics="boltzmann" uses F' = n/N.
    v0.16: with TransportModel.band = KaneBand(m0, alpha) the closure is the band's,
    n = N_c(m0) G(eta, alpha kT) (carriers.kane_band), so F' is dG/deta at eta_bg = G^-1(n_bg/N_c(m0)).
    Used by the default z mesh grading (geometry.specs.Mesh2DSpec.semiconductor_refinement)."""
    from dynameta.constants import EPS0, KB, Q_E
    T = transport_temperature_K(transport)
    n = float(transport.n_bg_m3)
    kb = PE.kane_band_of(getattr(transport, "band", None))
    if kb is not None and float(kb.alpha_per_eV) != 0.0:
        from dynameta.carriers import kane_band as KB_
        N = float(PE.effective_dos_m3(float(kb.m0_kg), T))
        PE.require_positive(n_bg_m3=n, dos_mass_kg=kb.m0_kg, eps_static=transport.eps_static)
        b = KB_.kane_b(kb.alpha_per_eV, T)
        x = n / N
        # non-degenerate: G' = G (every Laguerre term is ~ e^eta) to far better than the mesh rule needs
        dF = x if x < _NONDEGENERATE_X else float(KB_.kane_closure_slope(KB_.invert_kane_closure(x, b), b))
        return float(np.sqrt(float(transport.eps_static) * EPS0 * KB * T / (Q_E * Q_E * N * dF)))
    if (getattr(transport, "physics", None) == "bipolar_dd" and getattr(transport, "acceptor", False)
            and not getattr(transport, "net_doping_expr", None)
            and getattr(transport, "dos_mass_p_kg", None) is not None):
        m = float(transport.dos_mass_p_kg)
    else:
        m = float(np.asarray(transport.dos_mass_kg_of_n_m3(n)))
    N = float(PE.effective_dos_m3(m, T))
    PE.require_positive(n_bg_m3=n, dos_mass_kg=m, eps_static=transport.eps_static)
    x = n / N
    boltz = (getattr(transport, "physics", None) == "bipolar_dd"
             and getattr(transport, "carrier_statistics", None) == "boltzmann")
    if boltz or x < _NONDEGENERATE_X:
        dF = x
    else:
        eta = PE.invert_F12(x)
        h = 1e-4                                            # central difference: rel. error ~1e-9
        dF = (PE.F12_aymerich_humet(eta + h) - PE.F12_aymerich_humet(eta - h)) / (2.0 * h)
    return float(np.sqrt(float(transport.eps_static) * EPS0 * KB * T / (Q_E * Q_E * N * dF)))
