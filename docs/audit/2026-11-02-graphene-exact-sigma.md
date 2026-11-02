# Graphene interband Im(sigma): exact finite-T Kubo form (release note section)

Date: 2026-11-02. Scope: `dynameta/core/graphene.py` (`graphene_sigma`, `gate_tuned_reflection`).
Closes the deferred part of audit S4-22 (docs/audit/2026-07-17-exhaustive-audit.md), wave-7.

## What changed

1. **Default interband Im is now exact at finite T.** It was the thermally smoothed T = 0 log
   `-(1/2 pi) ln((hw + 2E_F)^2 / ((hw - 2E_F)^2 + (2kT)^2))`, which is exact only as T -> 0.
   It is now the principal-value Kubo integral of Falkovsky & Varlamov (Eur. Phys. J. B 56, 281
   (2007), Eqs. (10)-(11)), which is the Gamma -> 0 limit of Hanson (J. Appl. Phys. 103, 064302
   (2008), Eq. (1)), exp(-i omega t):
   `sigma_inter/sigma0 = G(hw/2) + i (4 hw/pi) PV int_0^inf G(e)/(hw^2 - 4e^2) de`, with
   `G(e) = f(-e) - f(e)`.
2. **Re(sigma_inter) and the intraband Drude term are unchanged.** They are bit-identical in both
   models. `tau_s` still enters the Drude term only.
3. **The old form is kept as an option:** `graphene_sigma(..., model="T0")`, which is
   bit-identical to the pre-2026-11-02 code. `gate_tuned_reflection` forwards `model`. Any other
   value raises `ValueError`.

## How the exact Im is computed

The method is a Maldague split. In units of hw, with mu = |E_F|/hw, t = kT/hw and a = 1/2:

- The Fermi function at (mu, T) is the T = 0 step averaged over a Fermi level m with the kernel
  `1/(4t cosh^2((m - mu)/2t))`. Because the Kubo integral is linear in f, Im_T is the thermal
  average of the T = 0 result `Im_0(m) = -(1/pi) ln((a + |m|)/|a - |m||)`.
- `Im_0 = (1/pi)[ln|m - a| + ln|m + a|] - (2/pi) ln(a + |m|)`. The two log singularities have a
  closed-form thermal average, `ln(2 pi t) + Re psi(1/2 + i y/(2 pi t))`.
- The remainder integrates by parts to `ln a + J`, where
  `J = int_0^inf [n_F((m - mu)/t) + n_F((m + mu)/t)]/(a + m) dm`. Its integrand is smooth and
  bounded.
- J is computed with a FIXED rule: 21 Gauss-Legendre panels of 16 nodes (336 nodes). The panels
  are graded toward the 1/(a + m) pole, and the window is +-45 kT around the Fermi edge. The
  region below the window is integrated analytically. There is no adaptive control flow, so the
  rule is JAX-traceable and differentiable in E_F.
- Re psi is computed by an 8-step recurrence followed by the Stirling series to B_14.
  Measured against `mpmath.digamma`: 1.8e-15.
- Rule accuracy, measured against a 40-point x 118-panel reference at 4700 random
  (mu, t) in [0, 7] x [1e-4, 3]: 5.7e-14.
- Limits:
  - T -> 0 away from resonance: the exact form reduces to the T = 0 log, with an O(T^2)
    residual against the T0 form (ratio 4.000 per halving of T).
  - At the resonance 2E_F = hw it gives `(1/pi)[ln(pi kT/(2hw)) - gamma_E] + O(T^2)`. For the T0 log,
    `exact - T0 = (ln(pi/2) - gamma_E)/pi = -0.03999` at every T, which is the documented
    "0.040 at resonance".

## Measured before/after: Im(sigma_inter)/sigma0 at 1.55 um (hw = 0.800 eV)

The oracle is a 30-digit mpmath quadrature of the subtracted Falkovsky-Varlamov integral
(`tests/test_reconfigurable.py::_kubo_im_mpmath`). It is independent of the Maldague route.

| T (K) | E_F (eV) | Im (oracle) | old (T0) | old - oracle | new - oracle |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.00 | -0.000761 | +0.000000 | +7.6e-04 | -4.4e-16 |
| 4 | 0.35 | -0.862327 | -0.862295 | +3.2e-05 | -2.2e-16 |
| 4 | 0.40 | -2.505299 | -2.463347 | +4.2e-02 | -4.4e-16 |
| 77 | 0.00 | -0.014652 | +0.000044 | +1.5e-02 | +2.3e-16 |
| 77 | 0.35 | -0.873147 | -0.859518 | +1.4e-02 | 0.0 |
| 77 | 0.40 | -1.565335 | -1.525375 | +4.0e-02 | -2.2e-16 |
| 300 | 0.00 | -0.057706 | +0.000664 | +5.8e-02 | +1.8e-16 |
| 300 | 0.20 | -0.359032 | -0.347115 | +1.2e-02 | +2.5e-15 |
| 300 | 0.35 | -0.924521 | -0.824528 | **+1.0e-01** | +5.4e-15 |
| 300 | 0.40 | -1.131930 | -1.092490 | +3.9e-02 | -6.0e-15 |
| 300 | 0.45 | -0.963938 | -0.863859 | **+1.0e-01** | -5.6e-16 |
| 300 | 0.60 | -0.522067 | -0.509567 | +1.2e-02 | -1.1e-16 |
| 600 | 0.00 | -0.120646 | +0.002638 | +1.2e-01 | +4.3e-16 |
| 600 | 0.45 | -0.870872 | -0.785923 | +8.5e-02 | -2.3e-14 |

Measured over the full gate grid (T 4 / 77 / 300 / 600 K x E_F -0.6..0.6 eV x hw 0.1 / 0.8 /
2.0 eV, through the public API):

- New code: max |dIm| = 1.8e-14 sigma0.
- Old code: max |dIm| = **0.556** sigma0 (hw = 0.1 eV, 600 K, E_F = 0).
- Against the wave-7 scipy oracle (`_kubo_interband_oracle`, 1.55 um, 10-300 K): new 6.2e-15,
  old 0.1001.

## Kramers-Kronig, gradients, cost

- **KK consistency of the module's own Re and Im** (Maclaurin PV, `kramers_kronig_dn_rows`,
  4001/8001 nodes, Richardson-extrapolated; 300 K):
  - new: |KK - Im| = 5.1e-10 at E_F = 0 and <= 1.3e-12 at E_F 0.35 / 0.40 / 0.45;
  - old: 0.058 / 0.100 / 0.039 / 0.100.
- **d Im / d E_F**:
  - `jax.jit(jax.grad)` against a central FD of the numpy path: relative 1.6e-8 at 4 K (FD
    truncation) and <= 2.4e-10 at 300 / 600 K.
  - Against the Richardson slope of the mpmath integral at 300 K, E_F 0.35 / 0.40 / 0.45: within
    2.6e-12 per eV. The true slopes are -5.970 / -0.389 / +5.168 per eV. The old slope was off by
    0.52 / 0.015 / 0.52 per eV, i.e. about 9 % of the gate sensitivity at the modulator edge.
  - jax and numpy values agree to <= 3.4e-16 sigma0. The grad at E_F = 0 is finite (it is 0,
    since Im is even in E_F).
- **Cost (numpy, measured on box A):**
  - per scalar call: 48 us exact vs 7.9 us T0, about 6x;
  - vectorized numpy: 16 us per E_F element;
  - JAX jit, 1001-element vector: 1.05 us per element.
  - Memory is an (n_EF x 336) float block. The numpy path chunks it at 8192 elements.

## Moved pins and consumers

- `tests/test_reconfigurable.py::test_s4_22_interband_kubo_vs_direct_quadrature`: the Im bar
  against the scipy oracle tightens from 0.11 / 0.08 / 0.045 (300 / 77 / 30 K) to **1e-12**
  (measured 6.2e-15). The old bars are kept, pinned on `model="T0"`.
- No other pin moved. The tests checked were `test_backend.py` jax/numpy parity (rtol 1e-12, still
  measured |ds| = 0.0 at E_F 0.3 eV), `test_audit_2026_07_25_conventions.py` (X-12 gate-tuned
  reflection, T_REF default), and the remaining graphene tests (Re-only or energy-balance).
- `validation/reconfigurable_modulators.py`: printed output is identical, and it PASSES.
  - A(E_F 0) = 0.0144134 -> 0.0144134 (+4e-9).
  - A(E_F 0.6) moves by +1.1e-10.
  - A depends on Im(sigma) only at second order through Z0 sigma.
- `validation/graphene_sheet_fem.py`: PASSES. Its gate-sweep sigma values move in Im only:

  | E_F (eV) | sigma before (S) | sigma after (S) | dIm/sigma0 | A before | A after |
  | --- | --- | --- | --- | --- | --- |
  | 0.2 | 6.0986e-05 - 1.7495e-06j | 6.0986e-05 - 2.4747e-06j | -0.0119 | 0.022456 | 0.022456 |
  | 0.4 | 3.0715e-05 - 2.7739e-05j | 3.0715e-05 - 3.0139e-05j | -0.0394 | 0.011438 | 0.011438 |
  | 0.6 | 5.0471e-07 + 2.7105e-05j | 5.0471e-07 + 2.6345e-05j | -0.0125 | 0.000190 | 0.000190 |

  The phases move at first order in Z0 dIm(sigma). At E_F 0.4 eV in the vacuum | sheet | vacuum
  cell: arg t changes by +4.5e-4 rad and arg r by -0.041 rad. The arg r shift is large because r
  is small there (R = 1e-4).

## Gates (each fails with the old Im)

Mutation check, 2026-11-02: `_interband_im_exact` was swapped for the old log, and all five gates
below FAIL.

- `test_s4_22_interband_kubo_vs_direct_quadrature`: Im bar 1e-12 against the scipy oracle.
- `test_graphene_exact_interband_im_vs_mpmath_principal_value`: 1e-11 over the T / E_F / hw grid,
  plus tau-independence (spread 1.1e-16).
- `test_graphene_exact_im_reduces_to_T0_closed_form`: the O(T^2) approach and the resonance
  constant.
- `test_graphene_interband_kramers_kronig_consistent`
- `test_graphene_exact_im_jax_grad_matches_fd_and_kubo_slope`
