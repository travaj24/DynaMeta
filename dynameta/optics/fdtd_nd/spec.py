"""Shared FDTD input spec: the FDTDLayer material/geometry dataclass.

Owned by the n-D package (audit 2026-07-05 section 5: fdtd_nd used to import its input
spec FROM the legacy 1-D optics/fdtd.py -- an ownership inversion). The 1-D module
re-exports FDTDLayer from here, so `from dynameta.optics.fdtd import FDTDLayer` keeps
working. Deliberately dependency-light (dataclasses only): every engine imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:                                            # keep spec.py dependency-light at runtime
    from dynameta.optics.hot_carrier import HotCarrierParams


@dataclass
class FDTDLayer:
    """One layer of the 1D stack. Non-dispersive eps_inf, plus an optional Drude pole
    (eps -= wp^2/(w^2 + i gamma w)), an optional Lorentz pole (eps += d_eps w0^2/(w0^2 - w^2 - i gl w),
    the bound-electron / interband resonance the bare Drude cannot capture) and an instantaneous Kerr
    chi3 in the STANDARD chi^(3) convention: P_NL = eps0 chi3 E^3, i.e. the update uses
    eps_eff = eps_inf + 3 chi3 E(t)^2 and the fundamental-band shift is
    d_eps = (3/4) chi3 |A|^2 (audit C3-2: the update previously used eps_inf + chi3 E^2 --
    an effective chi^(3) three times weaker than the literature value entered, and 4x weaker
    than this docstring then claimed; chi3_m2_V2 now takes literature chi^(3) values
    directly). All optional terms default off, so a plain dielectric is just eps_inf.

    The 2D/3D engine (fdtd_nd) carries the Drude + Kerr; the Lorentz pole is honored by the 2D-TE kernels
    (numpy/numba) via the central-difference ADE (a second polarization state PL). eps(w) at one lambda is
    typically supplied by inverting to a single Drude pole (the seam); fit_drude_lorentz fits BOTH poles
    across a band for an exact broadband metal/interband representation."""
    thickness_m: float
    eps_inf: float = 1.0
    drude_wp_rad_s: float = 0.0
    drude_gamma_rad_s: float = 0.0
    chi3_m2_V2: float = 0.0
    lorentz_w0_rad_s: float = 0.0          # Lorentz resonance frequency (0 = no Lorentz pole)
    lorentz_gamma_rad_s: float = 0.0       # Lorentz damping rate
    lorentz_delta_eps: float = 0.0         # Lorentz static strength (eps(0) gains d_eps)
    # R15 second-order + dispersive third-order nonlinearities (2D-TE numpy kernel; default off):
    chi2_m_V: float = 0.0                  # SHG chi2 [m/V]: P2 = eps0 chi2 E^2 polarization source
    raman_chi3_m2_V2: float = 0.0          # Raman chi3 strength [m^2/V^2]: P_R = eps0 chiR E Q
    raman_w_rad_s: float = 0.0             # Raman vibrational resonance Omega_R [rad/s]
    raman_gamma_rad_s: float = 0.0         # Raman damping [rad/s] (Q'' + g Q' + W^2 Q = W^2 E^2)
    # R20 CLAMPED-INVERSION gain line (2D-TE numpy kernel; default off). The Lorentz-oscillator
    # gain ADE P'' + dw P' + w^2 P = -kappa dN E: inversion dN = N2 - N1 > 0 -> Im(chi) < 0 = GAIN
    # (exp(-i w t)); dN < 0 reduces EXACTLY to a passive Lorentz pole with delta_eps =
    # kappa |dN|/(eps0 w^2). kappa is the classical coupling q^2/m_eff [C^2/kg]; the small-signal
    # intensity gain at line center is g0 = kappa dN / (n c eps0 dw) [1/m]. The inversion is
    # CLAMPED (no rate dynamics in the field loop -- see optics.laser_gain for the four-level
    # populations; dynamic field-population coupling is a documented follow-on).
    gain_w_rad_s: float = 0.0              # transition frequency w_a [rad/s] (0 = no gain line)
    gain_dw_rad_s: float = 0.0             # FWHM linewidth dw_a [rad/s]
    gain_kappa_C2_kg: float = 0.0          # coupling q^2/m_eff [C^2/kg]
    gain_dN_m3: float = 0.0                # clamped inversion N2 - N1 [m^-3] (sign = gain/loss)
    # roadmap 2.1 -- opt-in per-cell hot-carrier two-temperature ADE (2D-TE NumPy reference kernel only;
    # None -> byte-identical to the passive Drude path). A HotCarrierParams (optics.hot_carrier) whose
    # precomputed m*(T_e)/gamma(T_e) tables drop this layer's cold (wp, gamma) as the absorbed optical
    # power p_abs = J_drude.E heats the local electron gas. ADDED LAST so parallel additive edits to this
    # dataclass cannot collide textually. Typed via TYPE_CHECKING (spec.py stays import-light).
    hot_carrier: Optional[HotCarrierParams] = None

    def eps_at(self, w_rad_s):
        """The complex eps(omega) this layer represents (eps_inf - Drude + Lorentz), convention
        exp(-i w t), Im(eps) > 0 = loss. Excludes the intensity-dependent Kerr term."""
        e = complex(self.eps_inf)
        if self.drude_wp_rad_s > 0.0:
            e = e - self.drude_wp_rad_s ** 2 / (w_rad_s ** 2 + 1j * self.drude_gamma_rad_s * w_rad_s)
        if self.lorentz_delta_eps != 0.0 and self.lorentz_w0_rad_s > 0.0:
            w0 = self.lorentz_w0_rad_s
            e = e + self.lorentz_delta_eps * w0 ** 2 / (w0 ** 2 - w_rad_s ** 2 - 1j * self.lorentz_gamma_rad_s * w_rad_s)
        return e


def hot_carrier_guard(entry_point: str, layers) -> None:
    """audit C5-7 / D-1: refuse a `hot_carrier` layer on an entry point that cannot march it.

    FDTDLayer.hot_carrier (roadmap 2.1, the per-cell two-temperature ADE) is implemented in exactly
    ONE kernel -- fdtd_nd.kernels2d.run_2d_te -- reached only through
    solve_fdtd_2d(..., backend='numpy'). Every other front end fills its material grids from the
    layer's Drude/Lorentz/chi fields and never reads `hot_carrier`, so it marches the layer as a
    plain PASSIVE film and returns a result BIT-IDENTICAL to hot_carrier=None (measured 2026-07-25:
    array_equal True on all four dropping entry points, no raise, no warning). The dropped physics is
    large, not marginal: on the one supported path a drive sweep moves max|dR0| by up to 0.166
    absolute (absorbed fraction 0.025 -> 0.19).

    The repo policy (audit C5-7) is "raise loudly rather than mis-solve", and the per-BACKEND half of
    it already exists one level down (solve2d._dispatch_2d_te refuses numba/jax/cupy with hot != None);
    this is the missing per-ENTRY-POINT half, factored to ONE implementation so the five call sites
    (solve_fdtd_1d, solve_fdtd_2d_oblique, solve_fdtd_3d, solve_fdtd_3d_oblique, solve_fdtd_3d_mo)
    cannot drift. Layers WITHOUT a hot_carrier take a single `any(...)` over None and are otherwise
    byte-identical to the pre-guard path.
    """
    if any(getattr(L, "hot_carrier", None) is not None for L in layers):
        raise NotImplementedError(
            "{}: hot-carrier two-temperature dynamics (FDTDLayer.hot_carrier) are carried ONLY by the "
            "2-D-TE NumPy reference kernel (optics.fdtd_nd.kernels2d.run_2d_te, i.e. "
            "solve_fdtd_2d(..., backend='numpy')) -- this entry point has no hot-carrier path and "
            "would SILENTLY ignore the term, returning the passive-layer result bit-for-bit "
            "(audit C5-7 / D-1). Use solve_fdtd_2d(..., backend='numpy'), or clear "
            "hot_carrier.".format(entry_point))
