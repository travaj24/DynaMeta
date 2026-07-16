"""Carrier/quantum absorption effects: QCSE electro-absorption, Burstein-Moss, intersubband.

Split from the former monolithic effects.py; see the package __init__ docstring for
the EffectModel seam contract. Bodies are verbatim. Pure numpy (scipy only lazily for
the Voigt lineshape).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from dynameta.constants import C_LIGHT, EPS0, HBAR, Q_E
from dynameta.core.backend import is_jax_array
from dynameta.core.numerics import trapz
from dynameta.core.effects.base import (_E_vec, _photon_energy_J, kramers_kronig_dn,
                                        kramers_kronig_dn_rows)

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
                "the zero-field exciton absorption at the probe.", RuntimeWarning, stacklevel=2)
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
    """
    eps_inf: float
    Eg0_J: float                  # undoped optical gap [J] (e.g. 3.6 * Q_E for ITO)
    m_vc_kg: float                # reduced joint conduction-valence mass [kg]
    alpha_edge: float             # dimensionless interband edge amplitude (O(1); Im(eps) ~ alpha_edge)
    bgr_coeff_J_m: float = 0.0    # bandgap-renormalization coefficient C in dE_BGR = C n^(1/3) [J*m]; 0 -> off
    tauc_exponent: float = 0.5    # 0.5 = direct-allowed sqrt(E-Eg) edge
    e_grid_J: tuple = None        # (E_lo, E_hi, N) KK grid override; None -> auto around Eg_opt + probe
    enabled: bool = True          # master off-switch: False -> eps_inf everywhere (delta 0)
    _N_EG = 64                    # Eg_opt samples for the grid-capable dn interpolation
    _KK_SPAN_J = 5.0 * Q_E   # how far above the highest edge the KK grid extends (~5 eV)
    _KK_N = 3001                  # KK photon-energy grid points

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
        Non-dimensionalized by Eg so alpha_edge is an O(1) amplitude (not a unit-laden prefactor)."""
        E = np.asarray(E_eval, dtype=np.float64)
        x = np.maximum(E - Eg_opt, 0.0) / Eg_opt
        return float(self.alpha_edge) * x ** float(self.tauc_exponent) * (Eg_opt / E) ** 2

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
        # KK photon-energy grid: span below the lowest edge / probe, up to well above the highest edge
        if self.e_grid_J is not None:
            lo, hi, ng = self.e_grid_J
            grid = np.linspace(float(lo), float(hi), int(ng))
        else:
            e_lo = min(Eg_lo, E_ph) - 0.5 * float(self._KK_SPAN_J)
            e_hi = max(Eg_hi, E_ph) + float(self._KK_SPAN_J)
            grid = np.linspace(max(e_lo, 1e-21), e_hi, int(self._KK_N))
        if not (grid[0] <= min(Eg_lo, E_ph) and max(Eg_hi, E_ph) <= grid[-1]):   # no silent KK truncation
            raise ValueError("e_grid_J must span the optical gap range and the probe energy")

        # grid-capable dn: precompute dn(Eg_opt) on a 1D Eg grid, interpolate onto the per-cell gaps.
        # absorption alpha = E eps2/(hbar c) [1/m] per Eg row; KK -> dn (index shift) at the probe.
        # The KK transform on the FIXED photon grid is a constant linear map and only its value AT
        # E_ph is consumed, so all _N_EG dalpha rows go through the two probe-bracketing kernel rows
        # as batched matvecs (audit 6.2 perf: the dominant per-bias cost, O(N) per row) instead of
        # _N_EG independent full O(N^2) divide-and-sum transforms.
        egs = (np.array([Eg_lo]) if Eg_hi - Eg_lo < 1e-30
               else np.linspace(Eg_lo, Eg_hi, int(self._N_EG)))
        dalpha_rows = grid[None, :] * self._eps2(grid[None, :], egs[:, None]) / (HBAR * C_LIGHT)
        dn_tab = kramers_kronig_dn_rows(grid, dalpha_rows, e_eval_J=E_ph)       # (_N_EG,) at E_ph
        if egs.size == 1:
            dn = np.full(Eg.shape, float(dn_tab[0]))
        else:
            dn = np.interp(Eg.ravel(), egs, dn_tab).reshape(Eg.shape)

        eps2_ph = self._eps2(E_ph, Eg)                                          # dimensionless Im edge (>= 0)
        kappa = 0.5 * eps2_ph                                                   # extinction = eps2/(2 n_re)~eps2/2
        n_re = np.sqrt(complex(self.eps_inf)).real + dn
        return (n_re + 1j * kappa) ** 2                                          # scalar grid (...,)


# ---- reconfigurable: phase-change + liquid-crystal (Phase 4) -------------------------------


@dataclass
class IntersubbandEffect:
    """Diagonal-anisotropic permittivity from a quantum SubbandResult: the growth-axis (z) response
    carries the INTERSUBBAND transitions whose dipole <psi_i|z|psi_j> lies along z, while the in-plane
    (x, y) response is the ordinary intraband free-carrier Drude. Reads fields['subband'] (a
    carriers.schrodinger_poisson.SubbandResult: energies_J, psi, z_m, sheet_density_m2).

        eps_xx = eps_yy = eps_inf - wp^2 / (w^2 + i w gamma_intra)              (intraband Drude)
        eps_zz = eps_xx + sum_{i<j}  S_ij / (eps0 (w_ij^2 - w^2 - i w gamma_inter))

    with wp^2 = n3d q^2/(eps0 m_opt), n3d = (sum_i n_s,i)/Leff, w_ij = (E_j - E_i)/hbar, and the
    f-sum-rule-CONSISTENT oscillator strength built in:

        S_ij = N_ij q^2 |z_ij|^2 (2 w_ij)/hbar ,   N_ij = (n_s,i - n_s,j)/Leff ,   z_ij = <psi_i|z|psi_j>

    Where the mass went: write the standard Lorentz-oscillator susceptibility per transition,
    chi_ij = (N_ij q^2 / (eps0 m0)) * f_ij / (w_ij^2 - w^2 - i w gamma), with the dimensionless TRK
    strength f_ij = (2 m0 w_ij / hbar) |z_ij|^2; substituting f_ij, the free mass m0 CANCELS exactly,
    leaving the dipole form S_ij above. The effective mass is NOT absent from the physics -- it enters
    through the psi_i / E_i of the Schrodinger solve that produce z_ij and w_ij -- but it must not be
    inserted AGAIN in the line strength (the classic double-counting bug). m_opt therefore appears
    ONLY in the intraband Drude term.
    The denominator uses -i w gamma (exp(-i omega t), Im(eps_zz) > 0 on resonance = absorptive).

    REDUCES to a scalar Drude * I when fewer than two sub-bands are occupied (no i<j pair, the
    Lorentzian sum is empty) -> diag(eps_D, eps_D, eps_D) == as_tensor(DrudeOptical.eps). v1 returns a
    UNIFORM (3,3) slab response (sheet smeared over Leff = z[-1]-z[0]); a z-graded eps_zz(z) via the
    local |psi(z)|^2 weighting is a documented follow-on (Leff sets the LINE STRENGTH, not the position
    or the Leff-free f-sum rule). exp(-i omega t), Im(eps) > 0 for absorbers; pure numpy."""
    eps_inf: float
    m_opt_kg: float            # sub-band-averaged Kane optical mass (intraband Drude only)
    gamma_intra_rad_s: float
    gamma_inter_rad_s: float   # intersubband dephasing (scalar; per-pair callable = follow-on)
    occ_floor_m2: float = 0.0  # min sheet density to count a sub-band as occupied

    def eps(self, fields: dict, lambda_m: float):
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
        Leff = float(z[-1] - z[0])
        if not (Leff > 0.0):
            raise ValueError("SubbandResult z_m must span a positive width (Leff = z[-1]-z[0])")

        omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
        n3d = float(np.sum(ns)) / Leff                                 # m^-3 total volume density
        wp2 = n3d * Q_E * Q_E / (EPS0 * float(self.m_opt_kg))
        eps_intra = complex(self.eps_inf) - wp2 / (omega * omega + 1j * omega * float(self.gamma_intra_rad_s))

        eps_zz = eps_intra
        gam = float(self.gamma_inter_rad_s)
        idx = np.where(ns > float(self.occ_floor_m2))[0]               # occupied sub-bands
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = int(idx[a]), int(idx[b])                        # E[i] < E[j] (sorted ascending)
                w_ij = (E[j] - E[i]) / HBAR                            # > 0
                z_ij = trapz(psi[:, i] * z * psi[:, j], z)             # <psi_i|z|psi_j> [m]
                N_ij = (ns[i] - ns[j]) / Leff                         # >= 0 (lower band more occupied)
                if N_ij < 0.0:                                        # population inversion -> gain
                    warnings.warn("IntersubbandEffect: inverted population (n_s[{}] < n_s[{}]) gives a "
                                  "gain line; clamping to 0".format(i, j))
                    N_ij = 0.0
                S_ij = N_ij * Q_E * Q_E * (z_ij * z_ij) * (2.0 * w_ij) / HBAR
                eps_zz += S_ij / (EPS0 * (w_ij * w_ij - omega * omega - 1j * omega * gam))

        out = np.zeros((3, 3), dtype=complex)
        out[0, 0] = eps_intra
        out[1, 1] = eps_intra
        out[2, 2] = eps_zz
        return out
