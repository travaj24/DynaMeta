"""
Graphene surface conductivity (Kubo) + analytic conductive-sheet reflection (roadmap Phase 4c).

A gated graphene sheet modulates light through its SURFACE conductivity sigma(E_F, omega) -- a 2D
current response, NOT a volume permittivity -- so it enters Maxwell as a surface-current boundary
condition, with the exact normal-incidence reflection/transmission of a conductive sheet between
two media n1, n2:

    r = (n1 - n2 - Z0 sigma) / (n1 + n2 + Z0 sigma),   t = 2 n1 / (n1 + n2 + Z0 sigma),
    R = |r|^2,   T = (Re n2 / Re n1) |t|^2,   A = 1 - R - T   (Z0 = 1/(eps0 c), free-space impedance)

The Fermi level E_F is gate-tunable; the interband term is PAULI-BLOCKED once 2|E_F| > hbar*omega
(the sheet bleaches to near-transparency), which is the gate-tunable graphene electro-absorption
modulator. sigma is the finite-temperature Kubo result (Falkovsky & Varlamov, Eur. Phys. J. B 56,
281 (2007), Eqs. (10)-(11); Hanson, J. Appl. Phys. 103, 064302 (2008), Eq. (1)): a Drude intraband
term + the interband term in units of the universal conductivity sigma0 = e^2/(4 hbar). BOTH parts
of the interband term are now EXACT at finite T (audit S4-22, wave-7 2026-11-02):

  * Re: the closed form sigma0 sinh(hw/2kT)/(cosh(E_F/kT) + cosh(hw/2kT)) (exponent-scaled);
  * Im: the principal-value Kubo integral, evaluated exactly (to <= 1e-13 sigma0) by a Maldague
    split -- closed-form digamma terms carry the two log singularities, a fixed-node Gauss-Legendre
    rule integrates the smooth remainder (see _interband_im_exact). It reduces to the T = 0 log
    -(1/2 pi) ln((hw + 2E_F)^2/(hw - 2E_F)^2) as T -> 0 away from 2|E_F| = hw. The previous
    thermally SMOOTHED log (Falkovsky 2008; exact only at T = 0, wrong by up to 0.10 sigma0 at
    300 K / 1.55 um near 2E_F = hw) stays available as graphene_sigma(..., model="T0").

Convention exp(-i omega t): a passive sheet has Re(sigma) > 0 (absorbing). SI units (E_F in J,
lambda in m, sigma in siemens, tau in s). Pure numpy (jax-traceable in E_F); no devsim/ngsolve.
The FEM surface-current boundary condition (a sheet Robin term in the curl-curl weak form) is
IMPLEMENTED -- optics.solver.solve_fem(sheet_bcs={'iface_z<nm>': sigma}), validated against this
module's sheet_rt oracle in validation/graphene_sheet_fem.py (docstring corrected per audit C6-5;
the old text claimed it was a follow-on).
"""

from __future__ import annotations

import warnings

import numpy as np

from dynameta.constants import Q_E, HBAR, KB, C_LIGHT, Z0
from dynameta.core.backend import array_namespace, is_numpy_array
from dynameta.constants import T_REF   # audit X-24: one reference temperature

# Z0 (free-space wave impedance, ~376.730 ohm) is imported from constants (audit F-21); `graphene.Z0`
# stays importable as before.
SIGMA0 = Q_E ** 2 / (4.0 * HBAR)        # universal interband AC conductivity of graphene, ~6.085e-5 S

# ------------------------------------------------------------------------------------------------
# Exact finite-T interband Im(sigma) (audit S4-22 deferred part; wave-7, 2026-11-02).
#
# Kubo (FV 2007 Eq. (11); Hanson 2008 Eq. (1) with Gamma -> 0 and j -> -i), energies in units of
# hbar*omega (so the photon energy is 1 and the resonance sits at e = a = 1/2), mu = |E_F|/hw,
# t = kT/hw, n_F(x) = 1/(e^x + 1):
#     sigma_inter / sigma0 = G(a) + i (4/pi) PV int_0^inf G(e) / (1 - 4 e^2) de,
#     G(e) = f(-e) - f(e) = 1 - n_F((e + mu)/t) - n_F((e - mu)/t).
# MALDAGUE SPLIT. f at (mu, t) is the T = 0 occupation theta(m - e) averaged over a Fermi level m
# with the kernel K(m - mu) = 1/(4t cosh^2((m - mu)/2t)) (= -d n_F/d m, unit area), and the Kubo
# integral is LINEAR in f, so Im_T(mu) = int K(m - mu) Im_0(m) dm with the T = 0 result
#     Im_0(m) = -(1/pi) ln((a + |m|) / |a - |m||)
#             = (1/pi) [ln|m - a| + ln|m + a|] - (2/pi) ln(a + |m|)     (identity, any real m).
# The two log singularities have the CLOSED-FORM thermal average
#     int K(x) ln|x + y| dx = ln(2 pi t) + Re psi(1/2 + i y / (2 pi t))         (psi = digamma)
# (checked: y -> 0 gives ln t - 0.1256 = ln t + ln 2 + [ln(pi/4) - gamma]; |y| >> t gives ln|y|),
# and the bounded kinked remainder integrates by parts (boundary term at m = 0 gives ln a) into
#     int K(m - mu) ln(a + |m|) dm = ln a + int_0^inf [n_F((m - mu)/t) + n_F((m + mu)/t)]/(a + m) dm,
# whose integrand is smooth and bounded by 2/a on m >= 0. Hence, EXACTLY:
#     Im/sigma0 = (1/pi) [2 ln(2 pi t) + Re psi(1/2 + i(mu - a)/(2 pi t))
#                          + Re psi(1/2 + i(mu + a)/(2 pi t))] - (2/pi) [ln a + J],
#     J = int_0^inf [n_F((m - mu)/t) + n_F((m + mu)/t)] / (a + m) dm.
# T -> 0: Re psi(1/2 + i z) -> ln|z| and J -> ln((a + mu)/a), giving -(1/pi) ln((a + mu)/|a - mu|),
# the T = 0 log. At the resonance mu = a the exact value is (1/pi)[ln(pi t/2) - gamma_E] + O(t^2),
# while the old smoothed log reads (1/pi) ln t there: exact - old = (ln(pi/2) - gamma_E)/pi =
# -0.03999 for every T -> 0, the documented 0.040 resonance offset.
#
# J QUADRATURE (fixed nodes => JAX-traceable and differentiable, no adaptive control flow): for
# m < L = max(0, mu - 45 t) the integrand is 1/(a + m) up to e^-45 (n_F((m - mu)/t) = 1 - O(e^-45),
# n_F((m + mu)/t) <= e^-90), integrated analytically to ln((a + L)/a); on [L, L + 90 t] the
# variable s = (m - L)/t in [0, 90] carries the Fermi edge (width 1 in s, n_F poles at height pi)
# and the 1/(a + m) pole at s = -(a + L)/t (distance a/t >= 0.97 from the interval over the
# hw >= 0.1 eV, T <= 600 K design range; >= 0.01 still resolved). Panels: geometric toward s = 0
# (0, .01, .03, .09, .27, .81, 2.43, 6) for that real pole, then width-6 panels to 90; 16-point
# Gauss-Legendre each = 336 nodes. MEASURED (2026-11-02) against a 40-point x 118-panel reference
# over 4700 random (mu in 0..7, t in 1e-4..3, incl. mu ~ a and mu ~ 45 t) : max |dIm| 5.7e-14 (the
# 240-node variant: 8.6e-11; 180 nodes: 2.1e-8). Beyond s = 90 the integrand is < e^-45.
# Re psi(1/2 + i z): upward recurrence by _PSI_SHIFT = 8 then the Stirling series to B_14 at
# |W| >= 8.5 (truncation < 1e-15), all in xp complex arithmetic (traceable, real output).
# ------------------------------------------------------------------------------------------------
_KUBO_PANEL_EDGES = (0.0, 0.01, 0.03, 0.09, 0.27, 0.81, 2.43) + tuple(6.0 * k for k in range(1, 16))
_KUBO_NODES_PER_PANEL = 16
_KUBO_WINDOW = 45.0                     # half-width (in kT) of the integrated Fermi-edge window


def _gauss_legendre_panels(edges, n):
    x, w = np.polynomial.legendre.leggauss(int(n))
    e = np.asarray(edges, dtype=np.float64)
    lo, hi = e[:-1, None], e[1:, None]
    return (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel(), (0.5 * (hi - lo) * w).ravel()


_KUBO_S, _KUBO_W = _gauss_legendre_panels(_KUBO_PANEL_EDGES, _KUBO_NODES_PER_PANEL)   # 336 nodes
_PSI_SHIFT = 8
_PSI_K = np.arange(_PSI_SHIFT, dtype=np.float64) + 0.5
# B_{2n} / (2n), n = 1..7 (Stirling series of digamma: psi(W) ~ ln W - 1/2W - sum c_n W^-2n)
_PSI_STIRLING = (1.0 / 12.0, -1.0 / 120.0, 1.0 / 252.0, -1.0 / 240.0, 1.0 / 132.0,
                 -691.0 / 32760.0, 1.0 / 12.0)
_NUMPY_CHUNK = 8192                     # numpy path: E_F elements per (chunk x 336) node block


def _re_digamma_half(z, xp):
    """Re psi(1/2 + i z) for real z (array, any backend). Recurrence psi(w) = psi(w + N) -
    sum_{k<N} 1/(w + k), whose real parts are (k + 1/2)/((k + 1/2)^2 + z^2), then Stirling at
    W = N + 1/2 + i z. Measured vs mpmath.digamma (2026-11-02): <= 1.8e-15 abs (1.1e-15 rel) over
    z in {0} U [1e-6, 1e4]."""
    s = xp.sum(_PSI_K / (_PSI_K * _PSI_K + (z * z)[..., None]), axis=-1)
    W = (_PSI_SHIFT + 0.5) + 1j * z
    iw2 = 1.0 / (W * W)
    ser = _PSI_STIRLING[-1] * iw2
    for c in _PSI_STIRLING[-2::-1]:         # Horner in W^-2
        ser = (ser + c) * iw2
    return xp.real(xp.log(W) - 0.5 / W - ser) - s


def _nF(x, xp):
    """Fermi function 1/(e^x + 1) in the overflow-free, gradient-safe tanh form."""
    return 0.5 * (1.0 - xp.tanh(0.5 * x))


def _interband_im_exact(mu, t, xp):
    """EXACT finite-T Im(sigma_inter)/sigma0 (block comment above). mu = |E_F|/hw >= 0 (array),
    t = kT/hw > 0 (python float). Fixed 336-node rule: traceable / differentiable in mu."""
    a = 0.5
    L = xp.maximum(0.0, mu - _KUBO_WINDOW * t)
    m = L[..., None] + t * _KUBO_S
    mu_ = mu[..., None]
    J = t * xp.sum(_KUBO_W * (_nF((m - mu_) / t, xp) + _nF((m + mu_) / t, xp)) / (a + m), axis=-1)
    tpt = 2.0 * np.pi * t
    C = (2.0 * np.log(tpt) + _re_digamma_half((mu - a) / tpt, xp)
         + _re_digamma_half((mu + a) / tpt, xp))
    return (C - 2.0 * (xp.log(a + L) + J)) / np.pi


def graphene_sigma(E_F_J, lambda_m, *, tau_s: float = 1.0e-13, T_K: float = T_REF,
                   model: str = "exact"):
    """Complex sheet conductivity sigma(E_F, omega) [S] (Kubo, finite T) = intraband Drude +
    interband. `E_F_J` is the Fermi level relative to the Dirac point (J; only |E_F| enters),
    `tau_s` the carrier relaxation time (intraband Drude only -- the interband term is the
    Gamma -> 0 Kubo limit of FV 2007 / Hanson 2008), `T_K` the temperature. Re(sigma) > 0
    (passive, exp(-i omega t)).

    model = "exact" (default): interband Im is the exact finite-T principal-value Kubo integral
    (audit S4-22, wave-7 2026-11-02; <= 1e-13 sigma0; numpy 48 us per scalar call, 16 us per
    element vectorized -- measured 2026-11-02).
    model = "T0": the previous thermally smoothed T = 0 log
    -(1/2 pi) ln((hw + 2E_F)^2 / ((hw - 2E_F)^2 + (2kT)^2)), bit-identical to the pre-2026-11-02
    code; 7.9 us per scalar call (6x cheaper) but off by up to 0.10 sigma0 at 300 K / 1.55 um
    near 2E_F = hw, by (8 ln2/pi)(kT/hw) at E_F = 0 and by up to 0.56 sigma0 at hw = 0.1 eV /
    600 K. Re and the intraband term are identical in both models.

    Backend-agnostic in E_F (numpy / cupy / jax via array_namespace): dispatching on E_F_J -- the
    gate-tunable knob -- keeps the conductivity inside a JAX trace, so d sigma / d E_F is available
    for gate-tuned-modulator inverse design (both models; the exact one uses a fixed-node rule, no
    adaptive control flow). lambda/tau/T are fixed scalars. (NumPy path returns a 0-d complex
    array for a scalar E_F, which complex()/.real consume exactly as the old python-complex
    return.) Memory: the exact model holds an (n_EF x 336) float block; the numpy path chunks it."""
    if not (tau_s > 0 and T_K > 0 and lambda_m > 0):
        raise ValueError("tau_s, T_K, lambda_m must be > 0")
    if model not in ("exact", "T0"):
        raise ValueError("model must be 'exact' or 'T0', got {!r}".format(model))
    xp = array_namespace(E_F_J)
    omega = 2.0 * np.pi * C_LIGHT / float(lambda_m)
    kT = KB * float(T_K)
    EF = xp.abs(xp.asarray(E_F_J))
    hw = HBAR * omega
    # intraband Drude: i e^2 * W / (pi hbar^2 (omega + i/tau)),  W = E_F + 2 kT ln(1+exp(-E_F/kT))
    W = EF + 2.0 * kT * xp.log1p(xp.exp(-EF / kT))
    sigma_intra = 1j * Q_E ** 2 * W / (np.pi * HBAR ** 2 * (omega + 1j / float(tau_s)))
    # interband (Falkovsky-Varlamov 2007): the EXACT finite-T universal-conductivity real part
    # Re = sigma0 sinh(hw/2kT)/(cosh(E_F/kT)+cosh(hw/2kT)) -- the thermal factor that -> a hard Pauli
    # step (sigma0 for hw>2E_F, 0 below) as T->0, with the correct (small) blocked-tail residual
    # (an arctan smoothing overestimates that tail ~100x; audit GRAPH-1).
    # OVERFLOW-SAFE form (audit S4-22, wave-7 2026-11-02, found by the Kubo-integral oracle):
    # sinh(a) / (cosh(b) + cosh(a)), a = hw/2kT, b = E_F/kT, with numerator and denominator both
    # scaled by 2 exp(-m), m = max(a, b), so every exponent is <= 0. The bare sinh/cosh overflowed
    # once a or b passed ~710 -- at 1.55 um that is T < ~6.5 K -- and returned NaN (measured: T_K 3
    # -> nan + RuntimeWarnings). Same value elsewhere to <= 2 ulp (measured max rel 4.2e-16 over
    # E_F 0..1 eV x T 30..600 K x lambda 0.8..10 um, Re and Im).
    a_ = hw / (2.0 * kT)
    b_ = EF / kT
    m_ = xp.maximum(b_, a_)
    re = SIGMA0 * ((xp.exp(a_ - m_) - xp.exp(-a_ - m_))
                   / (xp.exp(b_ - m_) + xp.exp(-b_ - m_) + xp.exp(a_ - m_) + xp.exp(-a_ - m_)))
    if model == "T0":
        # the Falkovsky (2008) thermally SMOOTHED log, EXACT only as T -> 0 (kept verbatim from the
        # pre-2026-11-02 code). Against the S4-22 Kubo oracle at 1.55 um: |dIm|/sigma0 <= 0.10 at
        # 300 K (2 E_F within ~0.1 eV of hw), <= 0.075 at 77 K, 0.040 at 2 E_F = hw for T <= 30 K,
        # (8 ln2/pi)(kT/hw) ~ 0.058 at E_F = 0; at hw = 0.1 eV / 600 K up to 0.56 (E_F = 0).
        im = -SIGMA0 * (1.0 / (2.0 * np.pi)) * xp.log((hw + 2.0 * EF) ** 2
                                                      / ((hw - 2.0 * EF) ** 2 + (2.0 * kT) ** 2))
    else:
        mu = EF / hw
        t = kT / hw
        if is_numpy_array(mu) and mu.size > _NUMPY_CHUNK:     # bound the (n x 336) node block
            flat = mu.ravel()
            im = np.concatenate([_interband_im_exact(flat[i:i + _NUMPY_CHUNK], t, np)
                                 for i in range(0, flat.size, _NUMPY_CHUNK)]).reshape(mu.shape)
        else:
            im = _interband_im_exact(mu, t, xp)
        im = SIGMA0 * im
    return sigma_intra + (re + 1j * im)


def sheet_rt(n1, n2, sigma, *, theta_deg: float = 0.0):
    """Reflection/transmission of a conductive sheet between semi-infinite media n1 | sheet | n2 at
    NORMAL incidence (theta_deg kept for signature symmetry; only 0 supported). Returns
    (r, t, R, T, A) with A = 1 - R - T the sheet absorbed fraction. sigma -> 0 recovers the bare
    Fresnel (n1 - n2)/(n1 + n2)."""
    if abs(float(theta_deg)) > 1e-9:
        raise ValueError("sheet_rt currently supports normal incidence only (theta_deg = 0)")
    n1, n2, s = complex(n1), complex(n2), complex(sigma)
    if n1.imag != 0.0:
        raise ValueError("incidence medium n1 must be lossless (Im(n1)=0) for the R/T/A budget")
    if s.real < -1e-15:                          # passivity (exp(-i omega t)): Re(sigma) >= 0
        warnings.warn("sheet_rt: Re(sigma) < 0 is an ACTIVE/gain sheet (violates passivity for "
                      "exp(-i omega t)); the returned A = 1 - R - T will be negative (T > 1).",
                      RuntimeWarning, stacklevel=2)
    denom = n1 + n2 + Z0 * s
    r = (n1 - n2 - Z0 * s) / denom
    t = 2.0 * n1 / denom
    R = float(abs(r) ** 2)
    T = float((n2.real / n1.real) * abs(t) ** 2)
    return r, t, R, T, float(1.0 - R - T)


def gate_tuned_reflection(E_F_J, lambda_m, n1, n2, *, tau_s: float = 1.0e-13,
                          T_K: float = T_REF, model: str = "exact"):
    """Convenience: sigma(E_F) -> (R, T, A) for a graphene sheet between n1 and n2 -- the gate-tuned
    graphene modulator response at one Fermi level (`model` as in graphene_sigma)."""
    s = graphene_sigma(E_F_J, lambda_m, tau_s=tau_s, T_K=T_K, model=model)
    _, _, R, T, A = sheet_rt(n1, n2, s)
    return R, T, A
