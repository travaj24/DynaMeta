"""Carrier/quantum absorption effects: QCSE electro-absorption, Burstein-Moss, intersubband.

Split from the former monolithic effects.py; see the package __init__ docstring for
the EffectModel seam contract. Bodies are verbatim. Pure numpy (scipy only lazily for
the Voigt lineshape).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, HBAR, M_E, Q_E
from dynameta.core.backend import is_jax_array
from dynameta.core.numerics import trapz
from dynameta.core.effects.base import (_E_vec, _photon_energy_J, kramers_kronig_dn,
                                        kramers_kronig_dn_rows)
from dynameta.warning_types import AdvisoryWarning, UnreliableResultWarning


# ---- QCSE / MQW electro-absorption (Phase 3) ---------------------------------------------

@dataclass
class ElectroAbsorptionModel:
    """QCSE / MQW electro-absorption -- an EffectModel reading fields['E'] (uses the |E_z| field
    across the well). A quantum-well Stark driver (`qw`, any object exposing .solve(F) -> a state
    with .E_transition_J and .overlap) supplies the field-redshifted interband edge E_T(F) and the
    reduced e-h overlap. The model builds an excitonic absorption edge

        alpha(E_photon; F) = alpha0 * (overlap(F)/overlap(0)) * exp(-0.5 ((E_photon - E_T(F))/sigma)^2)

    (a Gaussian exciton line: redshifts with F and weakens with the overlap), forms the field-on
    minus field-off change dalpha on a photon-energy grid, and returns a complex scalar permittivity
    eps = (n + i kappa)^2 with kappa = max(kappa_bg + dalpha*hbar c/(2 E_photon), 0) and n = n_bg +
    dn, dn the Kramers-Kronig transform of dalpha. At F = 0 (dalpha = 0) eps -> eps_bg exactly.
    Convention exp(-i omega t): a passive medium has Im(eps) >= 0, so the total kappa is FLOORED at
    0 (and a warning fires) -- see the eps_bg note for when that floor would otherwise engage.

    SIMPLIFIED model (a first QCSE electro-absorption modulator): a single excitonic line, no
    band-to-band continuum, and a UNIFORM well field -- ONE Stark solve at the PEAK |E_z| over the
    field bundle. It therefore returns a SCALAR eps even if fields['E'] is a grid (it does NOT
    broadcast to a per-point grid -- do not compose it where a pointwise eps grid is required).
    Only the growth-axis (field_axis, z by default) component drives the QCSE; a purely in-plane
    field gives no modulation (a warning fires if |E| > 0 but the selected component is ~0).

    eps_bg is the zero-field permittivity at the operating wavelength. IMPORTANT (bleaching regime):
    for a probe near/above the F=0 exciton -- where the field moves the line AWAY and dalpha < 0 --
    eps_bg's IMAGINARY part must embed the full zero-field excitonic absorption at the probe, so the
    differential dalpha stays a physical reduction of an absorption that is actually present; the
    kappa >= 0 floor enforces passivity regardless. alpha0_per_m is the zero-field peak excitonic
    absorption; broadening_J the exciton linewidth (Gaussian sigma); e_grid_J = (lo, hi, N) the
    photon-energy KK grid (J), which MUST span several broadening_J beyond E_T(0) AND E_T(F) on both
    sides (the KK integral truncates otherwise) and MUST contain the probe photon energy."""
    qw: object                 # QuantumWell-like: .solve(F) -> state(.E_transition_J, .overlap)
    eps_bg: complex            # zero-field permittivity at the operating wavelength
    alpha0_per_m: float        # zero-field peak excitonic absorption [1/m]
    broadening_J: float        # exciton-line Gaussian sigma [J]
    e_grid_J: tuple            # (E_lo_J, E_hi_J, N) photon-energy grid spanning >> sigma around E_T
    field_axis: int = 2        # component of fields['E'] taken as the well field (z by default)
    continuum_alpha0_per_m: float = 0.0   # Elliott band-to-band continuum step strength (0 -> off)
    continuum_binding_J: float = 0.0      # 2D exciton binding = continuum onset above E_T (0 -> off)
    # R17 Voigt lineshape (lineshape="gaussian", the default, keeps the ORIGINAL code path --
    # byte-identical off-switch). lineshape="voigt" convolves the Gaussian sigma (broadening_J,
    # inhomogeneous) with a Lorentzian HWHM Gamma0_J [J] (lifetime dephasing) + an optional
    # field-ionization term Gamma_F_func(F_V_m) -> ADDITIONAL Lorentzian HWHM [J] (the QCSE
    # carrier-escape broadening). The Voigt is scaled to UNIT PEAK at Gamma = 0 -- so alpha0_per_m
    # keeps its peak-absorption meaning, the Gamma = 0 limit equals the Gaussian branch, and the
    # line AREA int alpha dE = alpha0 sigma sqrt(2 pi) is conserved for EVERY Gamma (oscillator
    # strength is not magicked in/out by broadening; the peak drops instead). The continuum keeps
    # its Elliott step shape (no Voigt broadening) by design.
    lineshape: str = "gaussian"           # "gaussian" | "voigt"
    Gamma0_J: float = 0.0                 # Lorentzian HWHM [J] (voigt only)
    Gamma_F_func: object = None           # F [V/m] -> additional Lorentzian HWHM [J] (voigt only)
    # R18 many-body density corrections, applied as POST-SOLVE closed forms when fields['n'] is
    # present (peak density, mirroring the peak-|E| convention; all default-off = byte-identical):
    #   bandgap renormalization  dE_BGR = -bgr_coeff_J_m * n^(1/3)  (edge REDSHIFT; published GaAs
    #     coefficient 2.4e-8 eV*cm = 3.8e-29 J*m -> ~24 meV at 1e24 m^-3 -- mind the cm/m trap),
    #   static exciton screening E_b(n) = E_b0 / (1 + n/n_s)  (n_s = screening_density_m3): the
    #     WEAKER binding blueshifts the line by E_b0 - E_b(n) and bleaches its oscillator strength
    #     f ~ E_b^p (screening_exponent p: 2D excitons f ~ |phi(0)|^2 ~ E_b -> p = 1, the default;
    #     3D -> p = 1.5),
    #   Mott transition: the LINE amplitude is EXACTLY 0 at n >= mott_density_m3 (the exciton
    #     unbinds; enable the Elliott continuum so absorption does not vanish unphysically).
    # exciton_binding_J here MUST equal the QuantumWell's (the EAM cannot read it from the driver).
    bgr_coeff_J_m: float = 0.0            # 0 = BGR off
    screening_density_m3: float = 0.0     # 0 = screening off
    exciton_binding_J: float = 0.0        # E_b0 [J]; required > 0 when screening is on
    screening_exponent: float = 1.0       # f ~ E_b^p (2D: 1, 3D: 1.5)
    mott_density_m3: float = 0.0          # 0 = no Mott cutoff

    _KK_MARGIN_SIGMA = 6.0     # required grid coverage beyond E_T on each side, in broadening_J

    def _density_corrections(self, fields: dict):
        """(dE_T_J, line_amplitude) from the R18 closed forms at the PEAK fields['n'] density.
        dE_T = -C n^(1/3) + (E_b0 - E_b(n)) (BGR redshift + screening blueshift); amplitude =
        (E_b(n)/E_b0)^p, EXACTLY 0 past the Mott density. (0.0, 1.0) when off/no density."""
        n_field = (fields or {}).get("n")
        if n_field is None or (self.bgr_coeff_J_m == 0.0 and self.screening_density_m3 == 0.0
                               and self.mott_density_m3 == 0.0):
            return 0.0, 1.0
        n = float(np.max(np.real(np.asarray(n_field))))
        if n < 0.0 or not np.isfinite(n):
            raise ValueError("ElectroAbsorptionModel: fields['n'] must be finite and >= 0")
        if (self.screening_density_m3 > 0.0 or self.mott_density_m3 > 0.0) \
                and not (self.exciton_binding_J > 0.0):
            raise ValueError("ElectroAbsorptionModel: exciton screening needs exciton_binding_J "
                             "> 0 (set it to the QuantumWell's binding energy)")
        d_e = -float(self.bgr_coeff_J_m) * n ** (1.0 / 3.0)
        amp = 1.0
        if self.screening_density_m3 > 0.0:
            eb_ratio = 1.0 / (1.0 + n / float(self.screening_density_m3))
            d_e += float(self.exciton_binding_J) * (1.0 - eb_ratio)   # weaker binding -> blueshift
            amp = eb_ratio ** float(self.screening_exponent)
        if self.mott_density_m3 > 0.0 and n >= self.mott_density_m3:
            amp = 0.0                                                  # the exciton unbinds
        return d_e, amp

    def _gamma_lor_J(self, F: float) -> float:
        """Total Lorentzian HWHM [J] at field F (0.0 on the gaussian path)."""
        if self.lineshape != "voigt":
            if self.Gamma0_J != 0.0 or self.Gamma_F_func is not None:
                raise ValueError("ElectroAbsorptionModel: Gamma0_J/Gamma_F_func are read only by "
                                 "lineshape='voigt' (set it explicitly; the gaussian default "
                                 "would silently ignore them)")
            return 0.0
        g = float(self.Gamma0_J)
        if self.Gamma_F_func is not None:
            g += float(self.Gamma_F_func(float(F)))
        if not (np.isfinite(g) and g >= 0.0):
            raise ValueError("ElectroAbsorptionModel: total Lorentzian HWHM Gamma(F) must be a "
                             "finite value >= 0, got {} at F={}".format(g, F))
        return g

    def _alpha(self, E_eval, E_T, overlap, overlap0, gamma_lor_J: float = 0.0):
        E = np.asarray(E_eval, dtype=np.float64)
        ratio = overlap / overlap0
        if self.lineshape == "voigt":
            from scipy.special import voigt_profile
            sig = float(self.broadening_J)
            # unit-peak-at-Gamma=0 scaling: voigt_profile is AREA-normalized (peak 1/(sig sqrt(2pi))
            # at Gamma=0), so * sig sqrt(2pi) recovers the unit-peak Gaussian at Gamma=0 and
            # conserves the line area for Gamma > 0 (the peak drops as the wings grow).
            g = voigt_profile(E - E_T, sig, float(gamma_lor_J)) * (sig * np.sqrt(2.0 * np.pi))
        elif self.lineshape == "gaussian":
            g = np.exp(-0.5 * ((E - E_T) / float(self.broadening_J)) ** 2)
        else:
            raise ValueError("lineshape must be 'gaussian' or 'voigt', got {!r}".format(
                self.lineshape))
        a = self.alpha0_per_m * ratio * g                              # 1s excitonic line
        if self.continuum_alpha0_per_m > 0.0 and self.continuum_binding_J > 0.0:
            # Elliott band-to-band continuum above the UNBOUND edge E_cont = E_T + E_binding, with the
            # 2D Sommerfeld enhancement S_2D(dE) = 2/(1+exp(-2 pi gamma)), gamma = sqrt(R/dE) with
            # R the effective 3D Rydberg = E_b(2D)/4 (Shinada-Sugano / Haug-Koch), i.e. in terms of
            # the 2D binding the exponent is -pi sqrt(E_b/dE) -> 2 at the edge and -> 1 far above
            # (a step joint-DOS, edge-enhanced). audit 7b P3: the exponent was previously doubled
            # (-2 pi sqrt(E_b/dE)), over-enhancing the continuum by up to ~20% for dE ~ (2-30) E_b. The continuum STRENGTH is set by the
            # interband momentum matrix element and is field-INDEPENDENT (NOT scaled by the 1s-exciton
            # envelope overlap ratio, which governs only the bound-exciton oscillator strength above):
            # the QCSE field acts on the continuum solely through the EDGE REDSHIFT E_T(F) carried in
            # E_T. (A prior ratio*continuum scaling made the continuum plateau spuriously drop with F;
            # it was never exercised because every gate had overlap==overlap0 -- audit fix.)
            xb = float(self.continuum_binding_J)
            dE = E - (E_T + xb)
            safe = np.where(dE > 0.0, dE, 1.0)                         # avoid sqrt of <=0
            s2d = np.where(dE > 0.0, 2.0 / (1.0 + np.exp(-np.pi * np.sqrt(xb / safe))), 0.0)
            a = a + self.continuum_alpha0_per_m * s2d
        return a

    def _solve0(self):
        """The bias-INDEPENDENT zero-field Stark solve, cached per qw object (audit 6.2 perf): a
        bias sweep re-enters eps()/delta_alpha_per_m once per bias but F = 0 never changes, so the
        baseline is reused EXACTLY (the same StarkState object; callers only read it) for ANY
        .solve(F) driver, not just the bundled QuantumWell (which memoizes by field itself). Keyed
        on qw object identity: the drivers treat their well parameters as fixed after construction
        (QuantumWell's own _solve_cache assumes the same) -- swap in a NEW qw object to change the
        well and the baseline re-solves."""
        cached = getattr(self, "_s0_cache", None)
        if cached is not None and cached[0] is self.qw:
            return cached[1]
        s0 = self.qw.solve(0.0)
        self._s0_cache = (self.qw, s0)
        return s0

    def _field_magnitude(self, fields: dict) -> float:
        # numpy-ONLY: unlike the other effect models this one wraps scipy eigensolvers (the QCSE
        # Schrodinger solve) + np.interp (the Kramers-Kronig transform), so it is not JAX-traceable.
        # Give a clear error instead of an opaque concretization failure if a JAX array is passed.
        if is_jax_array((fields or {}).get("E")):
            raise TypeError("ElectroAbsorptionModel is numpy-only (it wraps scipy eigensolvers + "
                            "Kramers-Kronig np.interp) and cannot be JAX-traced; pass a numpy E field "
                            "(use jax.pure_callback / a numpy boundary if differentiating around it).")
        E = np.asarray(_E_vec(fields))
        f_axis = float(np.max(np.abs(E[..., int(self.field_axis)])))
        f_tot = float(np.max(np.abs(E)))
        if f_tot > 0.0 and f_axis < 1e-6 * f_tot:
            warnings.warn(
                "ElectroAbsorptionModel: fields['E'] has |E[axis]| ~ 0 but |E| > 0 -- the QCSE "
                "field is the growth-axis (field_axis={}) component only; a transverse field gives "
                "NO modulation. Check field_axis / the field orientation.".format(self.field_axis),
                RuntimeWarning, stacklevel=3)
        return f_axis

    def eps(self, fields: dict, lambda_m: float):
        F = self._field_magnitude(fields)
        E_ph = _photon_energy_J(lambda_m)
        s0 = self._solve0()
        sF = self.qw.solve(F)
        ov0 = s0.overlap
        gam0, gamF = self._gamma_lor_J(0.0), self._gamma_lor_J(F)
        # R18 many-body corrections (closed forms on the SOLVED states; same density for the
        # F-on and F-off profiles, so the differential stays purely field-driven and the F = 0
        # flat-band reduction eps == eps_bg is preserved at every density)
        dE_n, amp_n = self._density_corrections(fields)
        eT0, eTF = s0.E_transition_J + dE_n, sF.E_transition_J + dE_n
        ovF_eff, ov0_eff = sF.overlap * amp_n, s0.overlap * amp_n
        lo, hi, n = self.e_grid_J
        grid = np.linspace(float(lo), float(hi), int(n))
        # the KK integral needs the grid to COVER the line several sigma beyond E_T on both sides
        # (a center-only straddle silently truncates dn by tens of percent -- audit QC-2). The
        # Voigt's Lorentzian 1/x^2 wings decay far slower than exp(-x^2), so the margin WIDENS
        # with Gamma/sigma (6 sigma pure-Gaussian -> +10 HWHM of Lorentzian reach; ~1% wing
        # truncation at the cap -- R17):
        margin = self._KK_MARGIN_SIGMA * float(self.broadening_J) + 10.0 * max(gam0, gamF)
        e_lo = min(eT0, eTF) - margin
        e_hi = max(eT0, eTF) + margin
        if self.continuum_alpha0_per_m:
            # the band-to-band Elliott continuum onset is at E_T + continuum_binding_J (with a slow
            # s2d -> 1 tail above it), which can sit FAR above E_T + margin; require the grid to reach
            # it + a margin so the KK integral does not silently truncate the continuum (audit QC-2b).
            e_hi = max(e_hi, max(eT0, eTF) + float(self.continuum_binding_J) + margin)
        if not (grid[0] <= e_lo and e_hi <= grid[-1]):
            raise ValueError("e_grid_J must span at least {:.0f}*broadening_J below E_T(0)/E_T(F) and "
                             "(when the continuum is on) up to E_T + continuum_binding_J + {:.0f}*"
                             "broadening_J above (the KK integral truncates otherwise)".format(
                                 self._KK_MARGIN_SIGMA, self._KK_MARGIN_SIGMA))
        # E_photon must be IN the grid: np.interp clamps to the edge outside it, diverging from the
        # analytic dkappa path (audit QC-3).
        if not (grid[0] <= E_ph <= grid[-1]):
            raise ValueError("the probe photon energy h c/lambda must lie within e_grid_J")
        dalpha_grid = (self._alpha(grid, eTF, ovF_eff, ov0, gamF)
                       - self._alpha(grid, eT0, ov0_eff, ov0, gam0))
        dn = float(np.interp(E_ph, grid, kramers_kronig_dn(grid, dalpha_grid)))
        dalpha_ph = float(self._alpha(E_ph, eTF, ovF_eff, ov0, gamF)
                          - self._alpha(E_ph, eT0, ov0_eff, ov0, gam0))
        dkappa = dalpha_ph * HBAR * C_LIGHT / (2.0 * E_ph)
        nb = np.sqrt(complex(self.eps_bg))
        kappa = nb.imag + dkappa
        if kappa < 0.0:                                   # passivity: no gain (audit QC-1)
            warnings.warn(
                "ElectroAbsorptionModel: kappa_bg + dkappa < 0 -- the differential model implies "
                "GAIN in the bleaching regime; clamping Im to 0. Supply an eps_bg whose Im embeds "
                "the zero-field exciton absorption at the probe.", UnreliableResultWarning, stacklevel=2)
            kappa = 0.0
        return complex((nb.real + dn) + 1j * kappa) ** 2

    def delta_alpha_per_m(self, fields: dict, lambda_m: float) -> float:
        """Field-induced absorption change dalpha = alpha(F) - alpha(0) [1/m] at the probe photon
        energy -- the electro-absorption-modulator extinction signal (>0 below the F=0 edge)."""
        F = self._field_magnitude(fields)
        E_ph = _photon_energy_J(lambda_m)
        s0 = self._solve0()
        sF = self.qw.solve(F)
        dE_n, amp_n = self._density_corrections(fields)               # R18 (0, 1 when off)
        return float(self._alpha(E_ph, sF.E_transition_J + dE_n, sF.overlap * amp_n, s0.overlap,
                                 self._gamma_lor_J(F))
                     - self._alpha(E_ph, s0.E_transition_J + dE_n, s0.overlap * amp_n, s0.overlap,
                                   self._gamma_lor_J(0.0)))


# ---- Burstein-Moss band-filling + bandgap renormalization (R8) -----------------------------

@dataclass
class BursteinMossEdge:
    """Carrier-density-dependent interband absorption edge of a degenerate semiconductor (e.g. ITO):
    band filling pushes the optical gap UP (Burstein-Moss blueshift) while many-body bandgap
    renormalization pulls it down (a redshift). Reads fields['n'] (carrier density m^-3) and returns
    the interband permittivity contribution as a scalar grid (promoted to isotropic by as_tensor):

        Eg_opt(n) = Eg0 - dE_BGR(n) + dE_BM(n),
          dE_BM(n)  = (hbar^2/2)(1/m_vc) (3 pi^2 n)^(2/3)    (band-filling blueshift)
          dE_BGR(n) = bgr_coeff_J_m * n^(1/3)                (renormalization redshift; 0 -> off)
        Im edge (Tauc/parabolic, exp(-i omega t) -> Im >= 0): dimensionless eps2(E; Eg_opt) = alpha_edge
          * ((E - Eg_opt)/Eg_opt)^tauc_exponent * (Eg_opt/E)^2 above Eg_opt, and its Kramers-Kronig
          partner dn(E) (reusing kramers_kronig_dn on alpha = E eps2/(hbar c)). eps = (sqrt(eps_inf) +
          dn + i kappa)^2, kappa = eps2/2 >= 0.

    This is a PURE interband DELTA meant to be composed THROUGH DeltaEffect on top of the bare Drude
    (whose eps_inf already embeds the interband response AT the reference doping). Compose as
    ComposedEffect(background=OpticalModelEffect(DrudeOptical(...)),
                   deltas=[DeltaEffect(BursteinMossEdge(eps_inf=<same eps_inf>, ...), {"n": n_ref})]);
    only the doping-INDUCED change relative to n_ref survives (no eps_inf double-count). Pick n_ref =
    n_bg (the fitted Drude eps_inf stays valid there). enabled=False -> returns eps_inf everywhere
    (delta = 0 through DeltaEffect = byte-identical off-switch). m_vc is the REDUCED joint
    conduction-valence mass (1/m_vc = 1/m_c + 1/m_v), NOT the Drude optical mass. numpy-only (KK uses
    np.interp); exp(-i omega t), Im(eps) >= 0; grid-capable (dn precomputed vs Eg_opt and interpolated).

    KK GRID AND ITS TRUNCATION (audit R-2 -- this is what the shipped default got wrong). The Tauc
    edge has NO high-energy cutoff, so its KK integrand decays only algebraically:

        dn(E) = (1/pi) int_0^inf E' eps2(E') / (E'^2 - E^2) dE' ,  E' eps2/(E'^2-E^2) ~ Eg^(2-p) E'^(p-3)

    (p = tauc_exponent). The integral CONVERGES for p < 2, but the neglected E' > E_hi remainder is
    only O(E_hi^(p-2)) -- E_hi^-1.5 for the default p = 0.5. The old default stopped 5 eV above the
    edge, which truncated ~45% of the absolute dn; worse, the old grid was rebuilt PER eps() CALL
    from the per-call Eg_opt(n) and probe, so a DeltaEffect integrated its two halves on DIFFERENT
    grids and the large common truncation did not cancel -- the delta came out 14-16x too large.
    Three things fix it here:

      1. the auto grid is a pure function of the CONFIGURATION (Eg0_J, tauc_exponent, and the class
         constants below) -- not of fields['n'] or lambda -- so it is built ONCE per instance and
         every eps() call, in particular both halves of a DeltaEffect, integrates on the SAME grid;
      2. E_hi is chosen adaptively so the truncated tail is below _KK_TAIL_REL of the dn scale
         (see _kk_min_e_hi: the tail and the scale are both proportional to alpha_edge, so the
         criterion is a pure ratio E_hi/Eg -- 142.3x Eg for p = 0.5 at 1e-3);
      3. what is left of the tail is added back in closed form (_kk_tail_dn), which removes both
         its absolute offset AND its Eg-dependence, so a DeltaEffect delta no longer inherits it.

    Measured against scipy.quad carried to infinity (the oracle the audit prescribes), the absolute
    dn is now correct to ~1e-5 relative and the DeltaEffect delta to <1e-4 of the dn scale, where
    the shipped default was 44-50% low in absolute dn and 14-16x too large in the delta. Because
    the grid no longer moves, alpha_edge is a physical amplitude again rather than a number that
    silently absorbs the grid choice.

    e_grid_J overrides the auto grid; it is validated against the same tail criterion and warns
    (audit R-2: the old guard only checked that the grid BRACKETED the gap, and accepted a grid
    ending 1 meV above Eg). An overriding grid is still used for both DeltaEffect halves, so it is
    pinned too. The AUTO grid is size-capped (_KK_MAX_GRID_BYTES): E_hi ~ tol^(-1/(2-p)) explodes as
    p -> 2 (1.6e9 points = 12.4 GB at p = 1.5), so past the cap _kk_grid RAISES and names e_grid_J
    rather than attempting the allocation.

    tauc_exponent = 0 (the 2-D / constant-joint-DOS edge) is ALLOWED and is a clean STEP: eps2 = 0
    below Eg_opt, alpha_edge (Eg_opt/E)^2 above. That requires masking the shape factor rather than
    evaluating ((E-Eg)/Eg)^p, because IEEE ``0.0 ** 0.0 == 1.0`` otherwise puts alpha_edge (Eg/E)^2
    of absorption EVERYWHERE below the gap (diverging toward E -> 0, and driving Im(eps) NEGATIVE --
    gain -- at a sub-gap probe once the equally corrupted KK dn pushed n_re past zero). See _eps2.
    """
    eps_inf: float
    Eg0_J: float                  # undoped optical gap [J] (e.g. 3.6 * Q_E for ITO)
    m_vc_kg: float                # reduced joint conduction-valence mass [kg]
    alpha_edge: float             # dimensionless interband edge amplitude (O(1); Im(eps) ~ alpha_edge)
    bgr_coeff_J_m: float = 0.0    # bandgap-renormalization coefficient C in dE_BGR = C n^(1/3) [J*m]; 0 -> off
    tauc_exponent: float = 0.5    # 0.5 = direct-allowed sqrt(E-Eg) edge (must be in [0, 2))
    e_grid_J: tuple = None        # (E_lo, E_hi, N) KK grid override; None -> auto (config-pinned)
    enabled: bool = True          # master off-switch: False -> eps_inf everywhere (delta 0)
    _N_EG = 64                    # Eg_opt samples for the grid-capable dn interpolation
    # ---- auto KK grid (audit R-2; replaces the old _KK_SPAN_J = 5 eV / _KK_N = 3001 pair) ----
    _KK_TAIL_REL = 1.0e-3         # target |truncated tail| / dn_scale for the auto E_hi
    _KK_GAP_HEADROOM = 1.5        # auto grid sized for Eg_opt up to this multiple of Eg0_J
    _KK_H_J = 0.01 * Q_E          # target auto-grid spacing [J] (10 meV; sets N with E_hi)
    _KK_E_LO_J = 1.0e-21          # grid start [J] (~0+, the KK lower limit; NOT exactly 0: _eps2
                                  # divides by E, and dalpha is identically 0 below the gap anyway)
    _KK_ROW_CHUNK_BYTES = 8 << 20  # byte budget for one dalpha_rows chunk (the grid is now ~1e5 wide)
    _KK_MAX_GRID_BYTES = 256 << 20  # hard cap on the AUTO grid's own float64 footprint (see _kk_grid)

    def gap_shift_J(self, n_m3):
        """Burstein-Moss blueshift dE_BM(n) [J] = (hbar^2/2)(1/m_vc)(3 pi^2 n)^(2/3)."""
        n = np.asarray(n_m3, dtype=np.float64)
        return (HBAR ** 2 / 2.0) * (1.0 / float(self.m_vc_kg)) * (3.0 * np.pi ** 2 * n) ** (2.0 / 3.0)

    def optical_gap_J(self, n_m3):
        """Doping-shifted optical gap Eg_opt(n) = Eg0 - dE_BGR + dE_BM [J]."""
        n = np.asarray(n_m3, dtype=np.float64)
        dE_BGR = float(self.bgr_coeff_J_m) * n ** (1.0 / 3.0)
        return float(self.Eg0_J) - dE_BGR + self.gap_shift_J(n)

    def _eps2(self, E_eval, Eg_opt):
        """Dimensionless interband Im(eps) edge: alpha_edge * ((E-Eg)/Eg)^p * (Eg/E)^2 above Eg, else 0.
        Non-dimensionalized by Eg so alpha_edge is an O(1) amplitude (not a unit-laden prefactor).

        The shape factor is MASKED on x = (E - Eg)/Eg > 0 rather than evaluated as ``x ** p``:
        numpy (like IEEE-754 ``pow``) defines ``0.0 ** 0.0 == 1.0``, so at p = 0 -- the physical
        2-D / constant-joint-DOS edge, and an ALLOWED exponent -- the unmasked form returned
        alpha_edge * (Eg/E)^2 EVERYWHERE BELOW THE GAP, diverging as (Eg/E)^2 toward E -> 0. That
        put absorption in the transparency window (Im(eps) = 2 n_re kappa at a 0.5 Eg probe came out
        NEGATIVE -- gain -- once the correspondingly corrupted KK dn drove n_re below zero) and it
        poisoned the KK integral itself, since the quadrature grid starts at ~0 and integrated that
        spurious sub-gap tail. Masked, p = 0 is a clean STEP edge (eps2 = 0 for E <= Eg,
        alpha_edge (Eg/E)^2 above) and every p > 0 is BYTE-IDENTICAL to the unmasked form
        (0.0 ** p == 0.0 exactly there)."""
        E = np.asarray(E_eval, dtype=np.float64)
        x = np.maximum(E - Eg_opt, 0.0) / Eg_opt
        shape = np.where(x > 0.0, x ** self._tauc_p(), 0.0)
        return float(self.alpha_edge) * shape * (Eg_opt / E) ** 2

    # ---- KK grid sizing and tail correction (audit R-2) -------------------------------------

    def _tauc_p(self) -> float:
        """tauc_exponent, validated. p >= 2 makes the KK integrand E'^(p-3) non-integrable, i.e.
        dn is INFINITE for this cutoff-free Tauc edge -- the old code returned a finite,
        purely grid-determined number instead.

        p == 0 IS allowed and physical (a 2-D / constant joint-DOS edge: eps2 jumps to
        alpha_edge (Eg/E)^2 at the gap and has no onset shape). _eps2 masks the shape factor so
        that case is a clean step; see its docstring for what the unmasked ``0.0 ** 0.0 == 1.0``
        did below the gap."""
        p = float(self.tauc_exponent)
        if not (0.0 <= p < 2.0):
            raise ValueError("BursteinMossEdge: tauc_exponent must lie in [0, 2) -- the cutoff-free "
                             "Tauc tail eps2 ~ (E/Eg)^(p-2) makes the Kramers-Kronig integral "
                             "divergent at p >= 2 (give the edge a finite bandwidth instead); "
                             "got {}".format(p))
        return p

    def _kk_dn_scale(self) -> float:
        """|dn| in the E -> 0 limit, in CLOSED FORM and independent of Eg:

            dn(0) = (1/pi) int_Eg^inf eps2(E')/E' dE'  --[E' = Eg/t]-->  (alpha/pi) B(p+1, 2-p)

        with B the Euler beta function. Used as the reference magnitude for the relative tail
        criterion in _kk_min_e_hi (0.1875 for the default alpha_edge = 1.5, p = 0.5)."""
        p = self._tauc_p()
        beta = math.gamma(p + 1.0) * math.gamma(2.0 - p) / math.gamma(3.0)
        return abs(float(self.alpha_edge)) / math.pi * beta

    def _kk_min_e_hi(self, Eg_ref: float) -> float:
        """Smallest KK grid upper limit whose neglected tail is below _KK_TAIL_REL of dn_scale.

        The leading tail term is T(E_hi) = (alpha/pi) Eg^(2-p) E_hi^(p-2) / (2-p) (E'^-1.5 for
        p = 0.5, i.e. halving per doubling of E_hi -- that observed halving IS the convergence
        law, not evidence of divergence). Setting T <= tol * (alpha/pi) B(p+1, 2-p) makes alpha
        cancel, leaving a pure ratio

            E_hi >= Eg * [tol (2-p) B(p+1, 2-p)]^(-1/(2-p))       (142.31 x Eg at p=0.5, tol=1e-3).
        """
        p = self._tauc_p()
        beta = math.gamma(p + 1.0) * math.gamma(2.0 - p) / math.gamma(3.0)
        return float(Eg_ref) * (float(self._KK_TAIL_REL) * (2.0 - p) * beta) ** (-1.0 / (2.0 - p))

    def _kk_tail_dn(self, Eg, e_hi: float):
        """Closed-form E' > e_hi remainder of the KK integral, added back to the quadrature so the
        result no longer carries the truncation offset OR its Eg-dependence (which is what made a
        DeltaEffect delta 14-16x too large). Two-term asymptotic in Eg/e_hi:

            (1/pi) int_{e_hi}^inf E' eps2 /(E'^2-E^2) dE'
              ~ (alpha/pi) Eg^(2-p) e_hi^(p-2) [ 1/(2-p) - p (Eg/e_hi)/(3-p) ] ,

        from (E'-Eg)^p = E'^p (1 - p Eg/E' + ...) and 1/(E'^2-E^2) = E'^-2 (1 + E^2/E'^2 + ...);
        the dropped terms are O((Eg/e_hi)^2) and O((E/e_hi)^2), both < 1e-4 of the tail on the auto
        grid. Vectorized over Eg."""
        p = self._tauc_p()
        Eg = np.asarray(Eg, dtype=np.float64)
        return (float(self.alpha_edge) / np.pi) * Eg ** (2.0 - p) * float(e_hi) ** (p - 2.0) * (
            1.0 / (2.0 - p) - p * (Eg / float(e_hi)) / (3.0 - p))

    def _kk_grid(self) -> np.ndarray:
        """The KK photon-energy grid, built ONCE per configuration and cached on the instance.

        Deliberately independent of fields['n'] and of lambda: that is what pins DeltaEffect's two
        eps() calls to the SAME grid (audit R-2). The cache key is the set of dataclass fields the
        grid depends on, so mutating them after construction still rebuilds.

        SIZE GUARD. The auto E_hi grows as ``tol^(-1/(2-p))``, i.e. EXPLOSIVELY as p -> 2 (the
        exponent at which the cutoff-free Tauc KK integral diverges): at the fixed _KK_H_J = 10 meV
        spacing the auto grid is 7.7e4 points at the default p = 0.5, 1.1e6 at p = 1, 1.3e7 at
        p = 1.25 and 1.6e9 -- a 12.4 GB float64 array, before any of the per-row temporaries -- at
        p = 1.5 (direct-FORBIDDEN Tauc). _KK_ROW_CHUNK_BYTES bounds a dalpha_rows CHUNK, but a chunk
        can never be smaller than ONE row of ``grid.size`` doubles, so the row budget is only
        honourable while the grid itself is bounded. This raises instead of attempting the
        allocation, and names the ``e_grid_J`` override (a user-chosen grid is validated against the
        same tail criterion and only WARNS, so the physics stays reachable at any p in [0, 2))."""
        key = (self.e_grid_J, float(self.Eg0_J), float(self.tauc_exponent),
               float(self._KK_TAIL_REL), float(self._KK_GAP_HEADROOM), float(self._KK_H_J),
               float(self._KK_E_LO_J))
        cached = getattr(self, "_kk_grid_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if self.e_grid_J is not None:
            lo, hi, ng = self.e_grid_J
            grid = np.linspace(float(lo), float(hi), int(ng))
        else:
            e_hi = self._kk_min_e_hi(float(self._KK_GAP_HEADROOM) * float(self.Eg0_J))
            npts = int(np.ceil(e_hi / float(self._KK_H_J))) + 1
            npts += 1 - npts % 2                       # force ODD (keeps the Maclaurin midpoint
                                                       # branch endpoint-free; R-3 covers both)
            max_pts = max(3, int(self._KK_MAX_GRID_BYTES) // 8)
            if npts > max_pts:
                raise ValueError(
                    "BursteinMossEdge: the auto Kramers-Kronig grid for tauc_exponent = {:g} needs "
                    "{:.4g} points ({:.3g} GB of float64) out to E_hi = {:.4g} eV -- the cutoff-free "
                    "Tauc tail requires E_hi ~ tol^(-1/(2-p)), which diverges as p -> 2, and the "
                    "{:.4g} meV auto spacing then makes the grid unbounded. The cap is {:.4g} points "
                    "({:.3g} GB, BursteinMossEdge._KK_MAX_GRID_BYTES). Supply an explicit "
                    "e_grid_J=(E_lo, E_hi, N) override (it is validated against the same tail "
                    "criterion and only WARNS, and the tail beyond E_hi is added back in closed "
                    "form by _kk_tail_dn), or raise _KK_H_J / _KK_TAIL_REL.".format(
                        self._tauc_p(), float(npts), 8.0 * npts / 1e9, e_hi / Q_E,
                        float(self._KK_H_J) / Q_E * 1e3, float(max_pts),
                        8.0 * max_pts / 1e9))
            grid = np.linspace(float(self._KK_E_LO_J), e_hi, npts)
        self._kk_grid_cache = (key, grid)
        self._kk_tail_warned = False       # re-arm the tail test for the new grid (see eps())
        return grid

    def eps(self, fields: dict, lambda_m: float):
        n_in = (fields or {}).get("n")
        if n_in is None:
            raise ValueError("BursteinMossEdge requires fields['n'] (carrier density m^-3); none "
                             "supplied (run the carrier model first)")
        if is_jax_array(n_in):
            raise TypeError("BursteinMossEdge is numpy-only (Kramers-Kronig np.interp); pass a numpy "
                            "density (omit this delta from a JAX-traced pipeline -- it is additive).")
        n = np.asarray(n_in, dtype=np.float64)
        if not self.enabled:
            return np.full(n.shape, complex(self.eps_inf))         # off-switch: pure eps_inf -> delta 0

        E_ph = _photon_energy_J(lambda_m)
        Eg = self.optical_gap_J(n)                                 # (...,) optical gap per cell
        Eg_lo, Eg_hi = float(np.min(Eg)), float(np.max(Eg))
        # KK photon-energy grid: PINNED to the configuration (audit R-2), so a DeltaEffect's two
        # eps() calls -- different n, hence different Eg_opt -- integrate on the identical grid.
        grid = self._kk_grid()
        e_hi_grid = float(grid[-1])
        if not (grid[0] <= min(Eg_lo, E_ph) and max(Eg_hi, E_ph) <= e_hi_grid):
            raise ValueError("e_grid_J must span the optical gap range and the probe energy")
        # REAL tail test (audit R-2): the old guard only checked that the grid BRACKETED the gap
        # and the probe, which accepts a grid ending 1 meV above Eg -- i.e. it could not see the
        # defect it was written to prevent. Compare against the criterion that actually sets E_hi.
        # Fires ONCE per grid build (the grid is a configuration property, and a lambda/bias sweep
        # would otherwise emit the same warning thousands of times); _kk_grid re-arms it.
        need_e_hi = self._kk_min_e_hi(Eg_hi)
        if e_hi_grid < need_e_hi and not getattr(self, "_kk_tail_warned", False):
            self._kk_tail_warned = True
            warnings.warn(
                "BursteinMossEdge: the Kramers-Kronig grid ends at {:.3g} eV but the cutoff-free "
                "Tauc tail needs >= {:.3g} eV for a truncation below {:.1e} of dn at Eg_opt = "
                "{:.3g} eV (the residual scales as E_hi^({:+.2f})). dn is tail-corrected in closed "
                "form, so the error is far smaller than the raw truncation, but widen e_grid_J (or "
                "drop the override) if dn matters at that level.".format(
                    e_hi_grid / Q_E, need_e_hi / Q_E, float(self._KK_TAIL_REL), Eg_hi / Q_E,
                    self._tauc_p() - 2.0), UnreliableResultWarning, stacklevel=2)

        # grid-capable dn: precompute dn(Eg_opt) on a 1D Eg grid, interpolate onto the per-cell gaps.
        # absorption alpha = E eps2/(hbar c) [1/m] per Eg row; KK -> dn (index shift) at the probe.
        # The KK transform on the FIXED photon grid is a constant linear map and only its value AT
        # E_ph is consumed, so all _N_EG dalpha rows go through the two probe-bracketing kernel rows
        # as batched matvecs (audit 6.2 perf: the dominant per-bias cost, O(N) per row) instead of
        # _N_EG independent full O(N^2) divide-and-sum transforms. The rows are built in CHUNKS
        # under a byte budget because the R-2 grid is ~1e5 wide (a full (_N_EG, N) block plus the
        # _eps2 temporaries would be a few hundred MB).
        egs = (np.array([Eg_lo]) if Eg_hi - Eg_lo < 1e-30
               else np.linspace(Eg_lo, Eg_hi, int(self._N_EG)))
        per_chunk = max(1, int(self._KK_ROW_CHUNK_BYTES) // (8 * grid.size))
        dn_tab = np.concatenate([
            kramers_kronig_dn_rows(
                grid,
                grid[None, :] * self._eps2(grid[None, :], egs[k:k + per_chunk, None]) / (HBAR * C_LIGHT),
                e_eval_J=E_ph)
            for k in range(0, egs.size, per_chunk)])                 # (_N_EG,) at E_ph
        dn_tab = dn_tab + self._kk_tail_dn(egs, e_hi_grid)           # closed-form tail (audit R-2)
        if egs.size == 1:
            dn = np.full(Eg.shape, float(dn_tab[0]))
        else:
            dn = np.interp(Eg.ravel(), egs, dn_tab).reshape(Eg.shape)

        eps2_ph = self._eps2(E_ph, Eg)                                          # dimensionless Im edge (>= 0)
        kappa = 0.5 * eps2_ph                                                   # extinction = eps2/(2 n_re)~eps2/2
        n_re = np.sqrt(complex(self.eps_inf)).real + dn
        return (n_re + 1j * kappa) ** 2                                          # scalar grid (...,)


# ---- Intersubband quantum eps_zz from sub-band wavefunctions (R7) -------------------------

_Z_RESPONSES = ("subbands", "drude_plus_subbands")
_FSUM_CHECKS = ("warn", "off")
_FSUM_BAND = (0.9, 1.05)
# below this f-sum ratio the sub-band set is refused (fsum_check="warn"): more than half
# of the electrons' z response is missing, so eps_zz is not a degraded approximation but a different material
# (it tends to eps_inf exactly as the ratio tends to 0 -- the default density() output of a 5-10 nm film keeps
# only the occupied sub-band and gave eps_zz = eps_inf + 0j).
_FSUM_HARD_FLOOR = 0.5


def subband_dipole_matrix(psi, z_m) -> np.ndarray:
    """Dipole matrix Z_ij = <psi_i| z - z_c |psi_j> [m] of sub-band states sampled on a UNIFORM grid.

    The quadrature is the one the eigenvectors are normalized with (SchrodingerPoisson1D: the Riemann
    sum, sum_k |psi_k|^2 h = 1, over the grid the states live on), so the discrete states are exactly
    orthonormal under it and Z is invariant to the choice of the z origin: shifting z by a constant
    adds (shift) x (overlap matrix) = (shift) x identity, which only moves the diagonal. z_c is the
    midpoint of z_m (it fixes the diagonal; the off-diagonal elements do not depend on it). One matrix
    product: Z = psi^T diag(z - z_c) psi h. Raises if z_m is not uniform (rtol 1e-6)."""
    z = np.asarray(z_m, dtype=np.float64)
    p = np.asarray(psi, dtype=np.float64)
    if z.ndim != 1 or z.size < 2 or p.ndim != 2 or p.shape[0] != z.size:
        raise ValueError("subband_dipole_matrix: psi must be (len(z_m), n_states) on a 1-D grid z_m")
    h = float(z[1] - z[0])
    if not (h > 0.0) or not np.allclose(np.diff(z), h, rtol=1e-6, atol=0.0):
        raise ValueError("subband_dipole_matrix: z_m must be uniformly spaced (rtol 1e-6): the dipole "
                         "quadrature must be the Riemann sum the eigenvectors are normalized with")
    zc = 0.5 * (z[0] + z[-1])
    return (p.T @ (p * (z - zc)[:, None])) * h


@dataclass
class IntersubbandEffect:
    """Diagonal-anisotropic permittivity of a film described by a quantum SubbandResult (reads
    fields['subband']: energies_J, psi, z_m, sheet_density_m2). The in-plane response is the free
    in-plane (intraband) motion of the sub-band electrons; the out-of-plane (z) response is the sum
    over the INTERSUBBAND transitions, whose dipoles lie along z.

    Default, z_response="subbands" (v0.16):

        eps_xx = eps_yy = eps_inf - wp^2 / (w^2 + i w gamma_intra) + chi_free
        eps_zz = eps_inf + sum_{i<j} S_ij / (eps0 (w_ij^2 - w^2 - i w gamma_inter)) + chi_free

        wp^2 = n3d q^2 / (eps0 m_opt),  n3d = sum_i n_s,i / Leff,  w_ij = (E_j - E_i) / hbar,
        S_ij = N_ij q^2 |z_ij|^2 (2 w_ij) / hbar,  N_ij = (n_s,i - n_s,j) / Leff,
        chi_free = -wp_free^2 / (w^2 + i w gamma_intra),  wp_free^2 = free_density_m3 q^2 / (eps0 m_opt).

    Derivation. The sub-band Hamiltonian is separable: free in-plane motion (E = E_i + hbar^2 k^2/2m)
    times the confined z states psi_i. A field along x accelerates every electron in-plane: the
    intraband Drude term. A uniform field along z cannot accelerate a confined electron (the states have
    no continuous z momentum); it mixes the states, and second-order perturbation theory gives the
    Lorentz-oscillator sum above, each transition weighted by its population difference. The
    Thomas-Reiche-Kuhn rule sum_j f_ij = 1, f_ij = (2 m w_ij / hbar) |z_ij|^2 (m the z mass of the
    Schrodinger solve), makes the oscillator strengths of a COMPLETE set exhaust the free-carrier
    response: sum_{i<j} (n_s,i - n_s,j) f_ij = sum_i n_s,i. So for w >> every w_ij the z sum tends to
    -wp_z^2 / w^2 (wp_z with the Schrodinger mass), i.e. the electrons respond along z like free
    carriers at high frequency WITHOUT any added Drude term, and as the film widens the transitions
    crowd into a continuum and eps_zz -> eps_xx (an isotropic bulk; the difference falls as 1/L).
    The z response therefore carries NO Drude term of the sub-band electrons: adding one (the
    pre-0.16 form, below) counts the same electrons twice, in every regime -- for a complete set by
    the sum rule, and for an isolated well because its confined electrons have no free z motion at all.
    Only a SEPARATE free population that is not in the sub-band set (e.g. continuum electrons of an
    isolated-well solve) has a Drude response, along every axis: give its density as free_density_m3
    (it enters eps_xx and eps_zz alike, with m_opt and gamma_intra).

    Mass. The in-plane Drude uses m_opt; the z response uses the Schrodinger mass that produced psi_i
    and E_i (it enters only through them; the free mass of the dipole form cancels -- inserting a mass
    again in S_ij would be the classic double count). The two differ for a nonparabolic band, so the
    thick-film limit of eps_zz is the plasma response with the PARABOLIC (Schrodinger) mass, and
    eps_zz - eps_xx tends to wp^2 (1/m_opt - 1/m_z)/w^2 there, not to 0.

    Quadrature. z_ij uses the Riemann sum the eigenvectors are normalized with
    (subband_dipole_matrix), measured from the midpoint of z_m, so the states are exactly orthonormal
    under it and eps_zz does not depend on the z origin. z_m must be uniform.

    f-sum diagnostic. Every call stores self.last_fsum_ratio = sum_{i<j} (n_s,i - n_s,j) f_ij / sum_i
    n_s,i with f_ij = (2 m_sp w_ij / hbar) z_ij^2, m_sp = subband_mass_kg (default m_opt; set it to the
    Schrodinger mass when m_opt differs). For a complete state set with m_sp = the Schrodinger mass it
    is 1 up to the O(h^2) discretization (the discrete identity is sum_i n_s,i kappa_i / sum n_s,i,
    kappa_i = h sum_k psi_i,k psi_i,k+1). In "subbands" mode (fsum_check="warn", the default):
      * a ratio below 0.5 RAISES ValueError: more than half of the electrons' z
        response is missing, so eps_zz is not an approximation but a different material -- it tends to
        eps_inf exactly as the ratio tends to 0 (SchrodingerPoisson1D.density's default keeps only the
        states needed for the density; for a 5 or 10 nm film that is the occupied sub-band alone, ratio 0,
        and eps_zz came out as eps_inf + 0j behind an AdvisoryWarning). Why refuse rather than complete
        the sum: the missing strength sits in transitions to the states that were dropped, at frequencies
        the result does not know (near the optical frequency in a few-nm film), so no stated remainder --
        e.g. a Drude term, i.e. all of it at zero frequency -- is right; a complete basis is cheap.
      * a ratio in [0.5, 0.9) or above 1.05 warns (AdvisoryWarning): a mild truncation, or m_sp is not
        the Schrodinger mass.
    fsum_check="off" skips both (model studies: a synthetic 2-3 state well is incomplete by construction).

    Example (a COMPLETE state set: every discrete state of the slab, as optics.quantum_film.QuantumFilm
    builds it; f-sum ratio 0.9997 on a 6 nm film at 0.025 nm):
        sp = SchrodingerPoisson1D(z_m, m_z)
        res = sp.density(U_J, E_F_J, n_states=None, bound_tol=np.inf)
        eps = IntersubbandEffect(4.0, m_z, 1.2e14, 1.2e14).eps({"subband": res}, 1200e-9)

    Legacy, z_response="drude_plus_subbands" (DEPRECATED, removal in 0.17; AdvisoryWarning): the pre-0.16
    eps_zz = eps_xx + sum ISB with the original trapezoid dipole quadrature on the absolute z, reproduced
    bit for bit (free_density_m3 must be 0 there).

    WHICH PAIRS RADIATE. A pair (i, j) contributes iff |n_s,i - n_s,j| > occ_floor_m2 (the
    canonical absorber, a FULL ground sub-band radiating into an EMPTY excited one, has the largest
    N_ij and the strongest line). An inverted pair (N_ij < 0) would be gain: it is clamped to 0 with an
    UnreliableResultWarning ("inverted population"), anti-gain policy row (2) of core.eps_field.

    Leff (the thickness the sheet densities are spread over, so that the effective layer of that
    thickness carries the film's sheet response) is thickness_m when given. Otherwise it is the
    distance between the film's two hard walls, read from the grid convention: a SubbandResult of a
    Dirichlet slab solve lives on the INTERIOR nodes (the wall nodes, where psi = 0, are excluded), so
    each wall sits one spacing h beyond the end node and Leff = z[-1] - z[0] + 2h; an end node at
    which every state vanishes (|psi| <= 1e-10 of max |psi|) is itself the wall and adds nothing (a
    grid written with its wall nodes included gives Leff = z[-1] - z[0]). This is the exact layer
    thickness of the slab, so the sheet response converges as O(h^2) with the grid (that of the states
    themselves); z[-1] - z[0] alone would be off by 2h / L (1 % on a 6 nm film at 201 nodes) and
    converge only as O(h). Pass thickness_m when the states live on a domain that is not the
    optical layer (e.g. a solve extended into finite oxide barriers: give the nominal film
    thickness). The legacy z_response='drude_plus_subbands' keeps its v0.15.1 Leff = z[-1] - z[0].
    This model is the LOCAL (uniform-field) response of the film: the same as the self-
    consistent response of the sub-band system with its depolarization kernel replaced by the
    uniform-field one (see dynameta.optics.quantum_film, which builds the nonlocal response).
    exp(-i omega t), Im(eps) > 0 for absorbers; pure numpy."""
    eps_inf: float
    m_opt_kg: float            # sub-band-averaged optical mass (in-plane Drude and the free population)
    gamma_intra_rad_s: float
    gamma_inter_rad_s: float   # intersubband dephasing (scalar; per-pair callable = follow-on)
    # R-1: this is the minimum |n_s,i - n_s,j| sheet-density DIFFERENCE for a pair to radiate, NOT
    # a per-sub-band occupancy floor (which silently deleted the full-ground/empty-excited absorber).
    occ_floor_m2: float = 0.0  # min |n_s,i - n_s,j| [m^-2] for an i->j line to be kept
    z_response: str = "subbands"            # "subbands" (v0.16) | "drude_plus_subbands" (deprecated)
    free_density_m3: float = 0.0            # a free population NOT in the sub-band set (isotropic Drude)
    subband_mass_kg: Optional[float] = None  # m_sp of the f-sum diagnostic (None -> m_opt_kg)
    fsum_check: str = "warn"                # "warn" | "off"
    thickness_m: Optional[float] = None     # Leff override (None -> wall-to-wall thickness of the grid)
    last_fsum_ratio: Optional[float] = field(default=None, init=False, repr=False, compare=False)

    def _inputs(self, fields):
        res = (fields or {}).get("subband")
        if res is None:
            raise ValueError("IntersubbandEffect requires fields['subband'] (a SubbandResult); none "
                             "supplied (run the Schrodinger-Poisson solver first)")
        z = np.asarray(res.z_m, dtype=np.float64)
        psi = np.asarray(res.psi, dtype=np.float64)
        E = np.asarray(res.energies_J, dtype=np.float64)
        ns = np.asarray(res.sheet_density_m2, dtype=np.float64)        # m^-2 per sub-band
        if psi.ndim != 2 or psi.shape[1] != E.size or ns.size != E.size:
            raise ValueError("SubbandResult psi/energies_J/sheet_density_m2 are inconsistent in shape")
        return z, psi, E, ns

    #: an end node is a hard wall when every state is below this fraction of max |psi| there
    _WALL_REL = 1e-10

    def _leff(self, z, psi=None) -> float:
        """Leff (class docstring): thickness_m, else the wall-to-wall thickness of the grid (psi
        given), else z[-1] - z[0] (the legacy form)."""
        if self.thickness_m is not None:
            Leff = float(self.thickness_m)
        else:
            Leff = float(z[-1] - z[0])
            if psi is not None and z.size >= 2:
                h = float(z[1] - z[0])
                a = np.abs(np.asarray(psi, dtype=np.float64))
                floor = self._WALL_REL * float(np.max(a)) if a.size else 0.0
                for end in (0, -1):                     # interior-node grid: the wall is h beyond
                    if not (a.size and float(np.max(a[end])) <= floor):
                        Leff += h
        if not (Leff > 0.0):
            raise ValueError("SubbandResult z_m must span a positive width, or thickness_m must be > 0")
        return Leff

    def eps(self, fields: dict, lambda_m: float):
        if self.z_response not in _Z_RESPONSES:
            raise ValueError("IntersubbandEffect: z_response must be one of {}, got {!r}".format(
                _Z_RESPONSES, self.z_response))
        if self.fsum_check not in _FSUM_CHECKS:
            raise ValueError("IntersubbandEffect: fsum_check must be one of {}, got {!r}".format(
                _FSUM_CHECKS, self.fsum_check))
        if self.z_response == "drude_plus_subbands":
            return self._eps_legacy(fields, lambda_m)
        z, psi, E, ns = self._inputs(fields)
        Leff = self._leff(z, psi)
        order = np.argsort(E, kind="stable")                           # ascending energies
        E, psi, ns = E[order], psi[:, order], ns[order]
        Z = subband_dipole_matrix(psi, z)                              # Riemann, origin-free

        omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
        drude = 1.0 / (omega * omega + 1j * omega * float(self.gamma_intra_rad_s))
        n3d = float(np.sum(ns)) / Leff
        wp2 = n3d * Q_E * Q_E / (EPS0 * float(self.m_opt_kg))
        nfree = float(self.free_density_m3)
        if not (np.isfinite(nfree) and nfree >= 0.0):
            raise ValueError("IntersubbandEffect: free_density_m3 must be finite and >= 0")
        chi_free = -(nfree * Q_E * Q_E / (EPS0 * float(self.m_opt_kg))) * drude
        eps_intra = complex(self.eps_inf) - wp2 * drude + chi_free

        iu, ju = np.triu_indices(E.size, 1)                            # all pairs i < j
        dns = ns[iu] - ns[ju]
        keep = np.abs(dns) > float(self.occ_floor_m2)
        iu, ju, dns = iu[keep], ju[keep], dns[keep]
        w_ij = (E[ju] - E[iu]) / HBAR                                  # >= 0
        z_ij = Z[iu, ju]
        self._fsum_diagnostic(ns, dns, w_ij, z_ij)
        inv = dns < 0.0
        if np.any(inv):                                                # population inversion -> gain
            k = int(np.flatnonzero(inv)[0])
            warnings.warn("IntersubbandEffect: inverted population (n_s[{}] < n_s[{}]{}) gives a gain "
                          "line; clamping to 0".format(int(iu[k]), int(ju[k]),
                                                       "" if inv.sum() == 1 else
                                                       " and {} more pair(s)".format(int(inv.sum()) - 1)),
                          UnreliableResultWarning, stacklevel=2)
            dns = np.where(inv, 0.0, dns)
        S_ij = (dns / Leff) * Q_E * Q_E * (z_ij * z_ij) * (2.0 * w_ij) / HBAR
        gam = float(self.gamma_inter_rad_s)
        chi_isb = complex(np.sum(S_ij / (EPS0 * (w_ij * w_ij - omega * omega - 1j * omega * gam))))
        out = np.zeros((3, 3), dtype=complex)
        out[0, 0] = eps_intra
        out[1, 1] = eps_intra
        out[2, 2] = complex(self.eps_inf) + chi_isb + chi_free
        return out

    def _fsum_diagnostic(self, ns, dns, w_ij, z_ij) -> None:
        """Store (and in "subbands" mode check) the f-sum ratio; see the class docstring."""
        m_sp = float(self.m_opt_kg if self.subband_mass_kg is None else self.subband_mass_kg)
        total = float(np.sum(ns))
        if not (total > 0.0):
            self.last_fsum_ratio = float("nan")                        # no carriers: nothing to check
            return
        f_ij = (2.0 * m_sp * w_ij / HBAR) * (z_ij * z_ij)
        ratio = float(np.sum(dns * f_ij) / total)
        self.last_fsum_ratio = ratio
        if self.z_response == "subbands" and self.fsum_check == "warn" and ratio < _FSUM_HARD_FLOOR:
            raise ValueError(
                "IntersubbandEffect: f-sum ratio sum (n_i - n_j) f_ij / sum n_i = {:.4g} < {}: the sub-band set "
                "misses most of the electrons' z response, so eps_zz would tend to eps_inf instead of the film's "
                "response. Pass a COMPLETE state set -- SchrodingerPoisson1D.density(U, E_F, n_states=None, "
                "bound_tol=np.inf), or the states of optics.quantum_film.QuantumFilm -- (an isolated-well solve: "
                "give the continuum density as free_density_m3); if subband_mass_kg ({:.4g} m_e) is not the "
                "Schrodinger mass, set it; fsum_check='off' accepts an incomplete set deliberately (model "
                "studies).".format(ratio, _FSUM_HARD_FLOOR, m_sp / M_E))
        lo, hi = _FSUM_BAND
        if (self.z_response == "subbands" and self.fsum_check == "warn"
                and not (lo <= ratio <= hi)):
            warnings.warn(
                "IntersubbandEffect: f-sum ratio sum (n_i - n_j) f_ij / sum n_i = {:.4g} is outside "
                "[{}, {}]: the sub-band set is incomplete (raise n_states / keep all slab states with "
                "a large bound_tol; an isolated-well solve drops the continuum -- give its density as "
                "free_density_m3), or subband_mass_kg ({:.4g} m_e) is not the Schrodinger mass of the "
                "solve. eps_zz then misses part of the free-carrier z response. fsum_check='off' "
                "silences this.".format(ratio, lo, hi, m_sp / M_E),
                AdvisoryWarning, stacklevel=3)

    def _eps_legacy(self, fields: dict, lambda_m: float):
        """The pre-0.16 form, bit for bit: eps_zz = eps_xx(Drude) + sum ISB with the trapezoid dipole
        quadrature on the absolute z. DEPRECATED (double counts the free-carrier z response)."""
        if float(self.free_density_m3) != 0.0:
            raise ValueError("IntersubbandEffect: free_density_m3 is not part of the legacy "
                             "'drude_plus_subbands' form; use z_response='subbands'")
        warnings.warn(
            "IntersubbandEffect(z_response='drude_plus_subbands') is deprecated and will be removed in "
            "0.17: it adds the sub-band electrons' own Drude term to eps_zz on top of the intersubband "
            "sum, which already exhausts their free-carrier response (f-sum rule) -- a double count in "
            "every regime. Use the default z_response='subbands' (and free_density_m3 for a population "
            "outside the sub-band set).", AdvisoryWarning, stacklevel=3)
        z, psi, E, ns = self._inputs(fields)
        Leff = self._leff(z)

        omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
        n3d = float(np.sum(ns)) / Leff                                 # m^-3 total volume density
        wp2 = n3d * Q_E * Q_E / (EPS0 * float(self.m_opt_kg))
        eps_intra = complex(self.eps_inf) - wp2 / (omega * omega + 1j * omega * float(self.gamma_intra_rad_s))

        eps_zz = eps_intra
        gam = float(self.gamma_inter_rad_s)
        floor = float(self.occ_floor_m2)
        # R-1: gate on the POPULATION DIFFERENCE, not on both states being occupied. The INITIAL
        # state supplies the carriers and the FINAL state ranges over every bound sub-band, so a
        # full ground / empty excited pair keeps its (maximal) line and an inverted pair reaches
        # the gain warning below instead of being deleted silently.
        for i in range(E.size):
            for j in range(i + 1, E.size):                             # E[i] < E[j] (ascending)
                dns = float(ns[i] - ns[j])
                if abs(dns) <= floor:                                  # no population difference
                    continue
                w_ij = (E[j] - E[i]) / HBAR                            # > 0
                z_ij = trapz(psi[:, i] * z * psi[:, j], z)             # <psi_i|z|psi_j> [m]
                N_ij = dns / Leff                                      # >= 0 (lower band fuller)
                if N_ij < 0.0:                                        # population inversion -> gain
                    warnings.warn("IntersubbandEffect: inverted population (n_s[{}] < n_s[{}]) gives a "
                                  "gain line; clamping to 0".format(i, j), UnreliableResultWarning, stacklevel=2)
                    N_ij = 0.0
                S_ij = N_ij * Q_E * Q_E * (z_ij * z_ij) * (2.0 * w_ij) / HBAR
                eps_zz += S_ij / (EPS0 * (w_ij * w_ij - omega * omega - 1j * omega * gam))

        out = np.zeros((3, 3), dtype=complex)
        out[0, 0] = eps_intra
        out[1, 1] = eps_intra
        out[2, 2] = eps_zz
        return out
