# P4: DEVSIM bridge v0.15 deferred items: strong-inversion density gradient, unipolar wall retirement, 2-D element impact ionization, floating-body detection, density-scaled abs_tol

Plan only. No repository was modified. Reference tree: `<workspace>\Python_Test_Scripts\DynaMeta_wt\dv_integ`
(branch `feat/devsim-2.11-v0.14` at `6595bde`), devsim 2.11.2, scipy-OpenBLAS pin.

Scratch probes run for this plan (read-only against the tree, `PYTHONDONTWRITEBYTECODE=1`, one process):
`C:\Users\Tesla\AppData\Local\Temp\claude\C--Users-Tesla\2446cd33-1dc4-4fe3-b419-8eafba706074\scratchpad\plan_v015\exp\inv_probe.py` and
`C:\Users\Tesla\AppData\Local\Temp\claude\C--Users-Tesla\2446cd33-1dc4-4fe3-b419-8eafba706074\scratchpad\plan_v015\exp\inv_diag.py`.
Their measured outputs are quoted in section 1.1 and drive the item-1 recommendation.

---------------------------------------------------------------------------------------------------

## 0. Cross-cutting facts that every item below depends on

### 0.1 DEVSIM's Newton convergence test (read from the devsim/devsim `main` source, 2026-11-04)

Read from the raw source. Paraphrase checked against the quoted lines:

* `src/Equation/Equation.cc`, `DefaultUpdate`: per equation, **absolute error = max over nodes of |update|**.
  **Relative error = max over nodes of |update| / (|x| + minError)**, with `defminError = 1.0e-10`.
  `PositiveSolutionUpdate` retries `0.001*old` and then `0.5*old` when a new value is not positive.
  `LogSolutionUpdate` compresses any |update| > 0.0259 to `0.0259*log(1 + |upd|/0.0259)`.
* `src/Geometry/Region.cc`, `Region::Update`: the region's absolute and relative errors are the **sums** over its
  equations.
* `src/Geometry/Device.cc`, `Device::Update`: the device's absolute error is the **max over regions**, and its
  relative error is the **sum over regions**.
* `src/math/Newton.cc`: `converged = converged && (devrerr < relLimit) && (devaerr < absLimit)` for every device. The
  circuit is tested the same way. Divergence is an OR test. There is no damping and no line search.

Consequences used throughout:

* (C1) The absolute criterion is **an AND with the relative one**. Raising `abs_tol` can never accept a state that
  `rel_tol` rejects. It can only remove a criterion that cannot be met.
* (C2) The absolute error of a semiconductor region is `max|dPsi| + max|dn| + max|dp| (+ max|du| + max|dLambda|)`
  in mixed units, so in practice it is the largest density update. An abs_tol of 1e10 m^-3 is therefore an
  absolute density criterion. In a Poisson-only (equilibrium) region it is vacuous, because there it is in volts.
* (C3) The relative criterion sums over every equation and region. A 5-equation DG region must reach about rel/5
  per equation.

### 0.2 Parallel work this plan must sequence with

* (P-a) Every Scharfetter-Gummel flux is being moved to the two-Bernoulli form, and the byte-identity digests
  (`tests/_dg_calllog.py`, `tests/_bipolar_calllog.py`) are being replaced by round-off agreement checks. Items 1
  and 2 must land **after** that. Their "unchanged" gates are written as round-off agreement, not digests.
* (P-b) DEVSIM sums UNSIGNED circumcentric pieces in 2-D and 3-D, so its volumes are exact only on well-centred
  elements. The library meshes qualify: 2-D `create_2d_mesh` right triangles and 3-D Kuhn tets. Item 3's element
  quadrature relies on this and must refuse any other mesh.

### 0.3 Literature access (honest status)

I tried the primary DG numerics sources:

* Wettstein, Schenk & Fichtner, IEEE TED 48, 279 (2001), DOI 10.1109/16.902727;
* Ancona, J. Comput. Electron. 1, 435 (2002), DOI 10.1023/A:1020732515391;
* Garcia-Loureiro et al., IEEE TCAD 30, 841 (2011), DOI 10.1109/TCAD.2011.2107990;
* Odanaka, IEEE TCAD 23, 837 (2004);
* Pinnau & Ruiz (FE analysis of Ancona's nonlinear scheme, kluedo.ub.rptu.de);
* the Silvaco DG application note (Ferron et al., Simulation Standard 2004).

**Full text was unreachable for all of them**: paywalls, an HTTP 403, a scanned PDF with no text layer, and
Enlighten "full text not available". Only abstracts were read:

* Wettstein et al. (2001) validated a dimension-independent unstructured-grid DG against 1-D Schrodinger-Poisson
  (SP) on a MOS diode, a MOSFET and a DG-SOI MOSFET.
* Ancona (2002) gives two finite-difference schemes that improve on the linear discretization.
* Garcia-Loureiro et al. (2011) is a "second-order differential" scheme that "converges at high voltages where
  previous schemes fail".

No equation numbers from these papers are quoted here. **Every formulation claim attributed to them is
UNVERIFIED.** The item-1 recommendation therefore rests on a measured diagnosis (section 1.1), not on a literature
claim. Before the release notes cite them, someone with library access must check the abstract-level statements
and the bibliographic details. Two textbook relations are used as oracles: the MOS surface-charge F-function and
the Fang-Howard centroid. They are written below from standard texts and marked "re-derive in the test" (the
oracle code derives them from Poisson's first integral, not from a citation).

---------------------------------------------------------------------------------------------------

## 1. Strong-inversion density gradient

### 1.1 Diagnosis: measured for this plan, and it changes the recommended fix

Fixture: `validation/bipolar_density_gradient._bar` with these settings:

* 1-D p-type Si, N_A = 3e23 m^-3, n_i = 1e16;
* 150 nm, wall mesh 0.05 nm, 268 nodes;
* tau = 1e-7 s, Boltzmann;
* Dirichlet psi_s at the wall, ohmic substrate.

The probe is the **classical** DD with no DG, ramped in 20 mV steps of psi_s - psi_b, with REL 1e-7 and ABS 1e10.

**Failure point in each precision:**

| run | last converged | fails at | Newton behaviour at failure |
|---|---|---|---|
| double | 0.76 V | 0.78 V | 40 iterations, no quadratic phase |
| extended (`XP.enable`) | 0.78 V | 0.80 V | Electron rel. update wanders between 1e-6 and 1e-1 for 40 iterations, never monotone; the state stays at max abs(phi_n) = 1e-14 V |
| `inv_probe.py`, extended, 16-step ramp to 0.80 | 0.80 V converged | 0.95 and 1.05 V | -- |

Iteration counts before failure (extended run): 6 its at 0.60 V, 9 at 0.70 V, 17 at 0.76 V, 22 at 0.78 V.

**Equilibrated (Ruiz) singular values of DEVSIM's DC Jacobian** (`get_matrix_and_rhs`, dense SVD, 804 unknowns),
double:

| psi_s - psi_b | sigma_min | next sigma | cond | null-vector mass in the Electrons block |
|---|---|---|---|---|
| 0.30 V | 3.8e-11 | 2.1e-5 | 8.6e10 | 1 - 1.5e-22 |
| 0.60 V | 9.0e-15 | 5.6e-6 | 3.8e14 | 1 - 4e-13 |
| 0.70 V | 2.8e-16 | 6.8e-6 | 1.2e16 | 1 - 6e-11 |
| 0.76 V | 2.0e-17 | 7.3e-6 | 1.7e17 | 1 - 2e-9 |

The same 0.70 V / 0.76 V Jacobians were tested under candidate reformulations, applied exactly by the chain rule
(`J_new = J_old T`):

| variables | sigma_min (0.70 V / 0.76 V) | next sigma |
|---|---|---|
| density (psi, n, p): the current code | 4.6e-16 / 6.0e-17 | 1.3e-5 |
| quasi-Fermi (psi, phi_n, phi_p) | **3.0e-16 / 7.7e-18** (no better) | 1.4e-4 |
| quasi-equilibrium electron closure (phi_n fixed, electron rows/cols removed) | **1.45e-4 / 1.43e-4** | 1.5e-4 |

What this shows:

1. **One isolated near-null mode** lives entirely in the electron (minority) block. Its singular value falls
   roughly e-fold per V_t of band bending until cond x eps > 1, and the Newton then fails. This is a physical
   near-floating body, the R26 null mode in near-singular form. In an MOS capacitor the inversion layer's
   electron number is tied to the substrate only through bulk minority diffusion (n0 = 3.3e8 m^-3) and SRH
   generation, whose derivative in the inversion layer is about p/(tau n), i.e. 1e-5 s^-1 or less. Against the
   dielectric-relaxation-scale entries of the rest of the matrix that coupling is 1e-17-weak.
2. **It is not a precision or criterion problem.** Extended precision assembles the residual in 128 bits, but
   DEVSIM factorizes in double (`extended_precision.py` header). Once cond x eps_double > 1 the correction is
   noise along the null vector, whatever the residual precision. Extended precision buys one 20 mV step (0.78 to
   0.80 V). Item 5's abs_tol would not help either: abs is not the binding criterion in the table.
3. **Quasi-Fermi / Slotboom / log-density variables do not fix it.** At the linear level each is a
   (block-triangular or diagonal) change of variables of the same Jacobian. The equilibrated sigma_min is
   unchanged or worse, as measured. This **refutes** the direction the D2d doc's Limits and Open question 4 suggest
   ("a quasi-Fermi / Slotboom-variable formulation"). Those variables can change the Newton path, but not the
   conditioning of the DC problem.
4. **Removing the degree of freedom fixes it.** With phi_n fixed, cond falls from 1e17 to about 1.4e4 (2.0 /
   1.43e-4), for a physically exact reason, given next.

**Why the closure is exact, not an approximation, for the target problem.** A 1-D MOS capacitor (perfect insulator)
has no DC current path. Its DC steady state is therefore **thermodynamic equilibrium**: phi_n = phi_p = the body
contact's Fermi level everywhere. This is what SP solvers and C-V codes assume, and it is exactly what GATE E
measures (max abs(phi_n) = 3e-16 V). The full DD carries that answer through an equation that determines it only
with stiffness ~1e-17.

The same holds for a MOSFET channel slice with an imposed imref split: phi_n = V_ch (the channel quasi-Fermi
level), phi_p = body. Real transport (a 2-D MOSFET with source and drain) pins the channel's phi_n laterally, so
the mode does not exist there and no closure is needed.

### 1.2 Goal and physics

Make the bipolar DG converge from depletion through strong inversion, psi_s - psi_b = 0 to >= 1.3 V. That is
>= 2 phi_F + 0.4 V for N_A = 3e23 (2 phi_F = 0.89 V) and for N_A = 1e24. Quantum accuracy is gated against SP.
Equations (sign convention of `physics_density_gradient`, D2d doc sec. 1.2):

    n = n_i exp((psi + D_c + Lambda_n - phi_n)/V_t)        (Boltzmann; FD: V_t[ln(n/n_i) + Delta(n/N_c)] = psi + D_c + Lambda_n - phi_n)
    Lambda_n = b_n lap(sqrt n)/sqrt n,   b_n = gamma_n hbar^2/(6 m_n q)                    (existing kernel)
    Poisson:  -div(eps grad psi) = q (p - n + N_D+ - N_A-)                                (existing)

**Quasi-equilibrium closure (QE) for carrier c in region R.** The continuity row of c is replaced at every node of R
by the algebraic row

    R_c = V_t [ln(c/n_i) + Delta_c(c/N_c)] - s_c (psi + D_c + s_c Lambda_c) + s_c phi_ref = 0
    s_n = +1, s_p = -1 (holes mirror: V_t[ln(p/n_i)+Delta] + psi - D_v - Lambda_p - phi_ref)

Here phi_ref is a device or region parameter (default: the bias parameter of a named ohmic contact).

* The row is in **volts**, local, and well scaled.
* Its Jacobian is `dR/dc = V_t (1/c + Delta'(c/N_c)/N_c)`, `dR/dpsi = -s_c`, `dR/dLambda = -1`.
* The solution variable stays the DENSITY. So every existing consumer of `Electrons`/`Holes` works unchanged:
  Poisson, the u = sqrt(c) tie, BGN quasi-fields, incomplete ionization, traps, noise inventory.

**Discrete exactness.** On every edge the zero-current SG condition of the full DD,
`n1 B(-v) = n0 B(v)`, gives `n1/n0 = exp(-v)` exactly, with v = vdiff_g_dg (Boltzmann). That is exactly the
closure's ratio. So **where the full DD converges, its discrete equilibrium and the closure's are the same discrete
solution**, and the agreement gate can be at solve tolerance, not at discretization error. (With FD the zero-current
SG ratio uses the edge-mean g, while the closure uses the node-local Delta. They then agree to O(h^2 g'). That is
gated at a derived tolerance, below.)

**FD Delta in DEVSIM form.** The repo's g fit is

    g = 1 + (a x + c x^{4/3})/(1 + b x^{1/3} + d x^{2/3})        (einstein.py, GA..GD)

so

    Delta(u) = int_0^u (g - 1) dx/x = int_0^{u^{1/3}} 3 s^2 (a + c s)/(1 + b s + d s^2) ds

This is rational in s and has a closed form (a polynomial plus log plus arctan). The discriminant is
b^2 - 4d = -0.805 < 0, hence the arctan. Implementation step 1 checks that DEVSIM's symbolic engine supports
`atan`. If it does not, v1 ships the closure for Boltzmann regions only and **refuses** FD (ValueError naming the
limitation). It does not approximate.

Literature position, all UNVERIFIED (section 0.3):

* Equilibrium DG of inversion layers is the setting of Ancona & Tiersten's original MOS work.
* SP inversion references (Stern; Ando, Fowler & Stern) are equilibrium calculations.
* Wettstein et al. (2001) compare a "MOS-diode" against SP, which is an equilibrium comparison.

What their codes do about the minority continuity in a 1-D MOS diode could not be read.

### 1.3 Current code state and reuse (one-kernel rule)

* **Reused unchanged:** `_install_dg_variables` (u, Lambda, tie, Lambda-equation), `setup_bipolar_dg` (fluxes stay
  for the non-closed carrier), `setup_dg_insulator_wall` (its c row is "natural", and on a closed carrier the
  natural row IS the QE row), `seed_bipolar_dg`, `_install_quantum_srh`, `set_dg_gamma`.
* **Reused from bipolar DD:** the equilibrium relation pieces already single-sourced for contacts and seeds
  (`fd_shift_model`, `ohmic_built_in_offset_V`, `equilibrium_fields` for D_c/D_v, n_ie). The QE row must be built
  from the same pieces, not re-derived. Concretely, the Boltzmann part `V_t ln(c/n_i)` and D_c come from the
  region's existing node models. The FD Delta becomes the DEVSIM twin of `physics_bipolar_dd.fd_potential_shift`,
  sharing the einstein coefficients, and is gated against the numpy quadrature.
* **New, small:** `physics_bipolar_dd.setup_quasi_equilibrium_carrier` (QE is DD physics, not DG-specific; DG reads
  the Lambda slot when present) and `delete_quasi_equilibrium_carrier` (restores the recorded continuity equation
  via eq_registry: same mechanism as `_frozen`).
* **Not built:** a parallel (psi, phi, Lambda) variable-set implementation (refuted in 1.1, and a parallel DD
  kernel); pseudo-transient continuation (it regularizes the path but cannot beat cond x eps at the end); Gummel
  splitting (already refused for bipolar, audit C-15).

### 1.4 Design

    physics_bipolar_dd.setup_quasi_equilibrium_carrier(
        device, region, carrier,                    # "electrons" | "holes"
        *, fermi_level: str,                        # name of a device/region PARAMETER (e.g. "sub_bias") holding phi_ref [V]
        lambda_model: str | None = "auto")          # "auto": QLambda / QLambdaP if the region carries that DG carrier
    -> None

* **Placement.** Call after `setup_bipolar_region` (+ options) and after `setup_bipolar_dg` when DG is used. The
  call order is guarded: if `setup_bipolar_dg` runs later on a closed carrier, it raises, because it re-points the
  continuity equation that QE replaced.
* **Equation mechanics.** It records a REGION equation `ElectronContinuityEquation` (same name, so contact
  equations, `_repoint_contact_current_to_dg` and noise refusals keep working) with `node_model="QE_n"`, no
  edge_model, and `variable_update="positive"`. The old kwargs are kept in the registry for
  `delete_quasi_equilibrium_carrier`. Contact rows are unchanged: ohmic contacts pin c and the contact bias, which
  is consistent with QE when fermi_level is that contact's bias.
* **Refusals (ValueError), and why.**
  * A closed carrier's terminal current is undefined (no continuity), so `get_contact_current` on that equation is
    flagged in the docstring.
  * `noise_analysis` refuses a QE region (its `edge_model == ElectronCurrent` check already refuses).
  * ssac IS allowed: QE = infinitely fast minority response = the low-frequency (quasi-static) C-V, which is
    physically right and gated (G1-5).
  * Multi-contact region with differing bias parameters: setup raises unless `allow_imref_split=True`, which states
    the user knows phi_n is a constant parameter (the channel-slice use).
* **Defaults and results.** Opt-in; nothing changes unless called. **No default changes.**
* **Extended precision.** Composes trivially (node model). It is no longer needed for convergence: gate G1-3 is
  run in double AND extended.
* **Dimensions.** The node model is dimension-agnostic. 2-D is gated (G1-7); 3-D gets a wiring test only.

### 1.5 Ordered implementation steps

1. **Probe DEVSIM `atan` and the FD Delta closed form.** Write the DEVSIM string twin of
   `fd_potential_shift`. Gate it against numpy Gauss-Legendre (`fd_potential_shift`) at u = 1e-6..136, rel <= 1e-12.
   Decide FD-or-refuse.
2. **`setup_quasi_equilibrium_carrier` + delete, Boltzmann first.** Unit-gate the Jacobian (node derivatives
   against central finite differences, rel 1e-7).
3. **Classical-closure oracles.** G1-1 and G1-2 (no DG): analytic Q_s(psi_s), and agreement with the full DD
   where it converges.
4. **Turn the 1.1 diagnosis into a test.** The sigma_min of the closed versus full Jacobian (G1-0) prevents a
   silent regression to an ill-posed formulation.
5. **DG + QE.** Run through the existing `_dg_ramp`/`_ramp_param` harness at gamma_n = gamma_p = 1 and gamma_wall,
   from 0 to 1.3 V. Gates G1-2b, G1-3, G1-4.
6. **SP inversion oracle.** Write it as validation code (see the G1-4 oracle), then the C-V (G1-5) and the mesh
   convergence (G1-6).
7. **2-D laterally uniform MOS-C** equals 1-D (G1-7).
8. **Docs.** D2d doc Limits rewritten: the root cause, the closure, and why QFL variables were rejected (with the
   numbers). Release note. Close D2d Open question 4.

### 1.6 Verification gates

| id | gate | oracle | tolerance and its derivation | fails on old code? | refutation attempt |
|---|---|---|---|---|---|
| G1-0 | the near-null mode is gone | equilibrated dense SVD of `get_matrix_and_rhs` on the 1-D MOS-C at 0.70 V (804 dof) | closed sigma_min / full sigma_min >= 1e8 (measured 1.45e-4 / 4.6e-16 = 3e11); closed cond <= 1e6 | yes: the full (old) system gives 2.8e-16 | also assert the QFL change of variables does NOT raise sigma_min by > 10x, so a future "Slotboom" proposal is pinned as no-help |
| G1-1 | classical closure vs analytic MOS surface charge | Q_s(psi_s) from Poisson's first integral, `E_s^2 = (2 q N_A V_t/eps)[(e^{-psi/V_t} + psi/V_t - 1) + (n0/p0)(e^{psi/V_t} - psi/V_t - 1)]`; re-derived in the test (written from the textbook F-function, Kingston & Neustadter 1955 / Sze ch. 4, citation unverified); Q_s = eps E_s by Gauss | box method is O(h^2/l^2), l = inversion-layer thickness (about V_t/E_s, 0.03-0.3 nm). Three meshes (wall spacing 0.05/0.025/0.0125 nm); gate observed order >= 1.8 and Richardson limit within 1e-5 of analytic, at psi_s = 0.2, 0.6, 0.9, 1.1, 1.3 V | yes: no closure means no convergence >= 0.80 V (measured) | the integrated charge (sum of rho vol) and Gauss (wall flux) must agree to 1e-12, so the gate cannot pass on a charge-bookkeeping error |
| G1-2 | closure == full DD where the latter converges (classical, 0.1-0.70 V; and **G1-2b** bipolar DG, gamma = 1, 0.1-0.60 V) | the existing solve | Boltzmann: identical discrete equilibrium (1.2), so <= 1e-10 rel in n, p, Lambda (solve-tolerance level; full DD at REL 1e-10 in extended) | n/a (consistency) | run with tau = 1e-7 and 1e3: the full DD answer may not depend on tau in equilibrium, and the closure has no tau |
| G1-3 | convergence through strong inversion | the solver itself | at N_A = 3e23 and 1e24, gamma in {0.5, 1, 1.672}, psi_s - psi_b up to 1.3 V in 50 mV steps: every step converges in <= 12 Newton iterations, in **double** AND extended, under both OpenBLAS builds (scipy, Elmer). Results across builds/precision agree to <= 1e-12 rel | yes: old fails at 0.70 V (DG) / 0.78 V (classical) | ramp-path independence: 3 different gamma/psi ramps end within 1e-12 |
| G1-4 | inversion charge and centroid vs Schrodinger-Poisson, weak to strong inversion | **new independent SP inversion oracle** (validation code; see the note below) | registered in advance as fidelity (not numerics) tolerances, from GATE C's accumulation precedent (gamma_wall within 5.6 %): at gamma_wall, N_inv within 10 % and centroid within 15 % for N_inv >= 1e15 m^-2; at gamma = 1, close >= 60 % of the classical-to-SP gap. Load-bearing: classical error >= 3x DG error at every point | yes: old DG cannot produce numbers beyond 0.60-0.70 V | (i) the SP oracle's own sanity: its zero-field free-gas deficit (0.626657, existing) and its classical limit (h -> 0 mass scaling) vs G1-1's analytic Q_s; (ii) the trend gate: the threshold shift dV_th (psi_s at N_inv = 1e16 m^-2) from DG and from SP agree within 20 mV, and classical dV_th = 0 |
| G1-5 | quasi-static C-V | C_s = dQ_s/dpsi_s by +-1 mV central differences, with V_g = psi_s + Q_s/C_ox (t_ox 1.5 and 3 nm, algebraic series, exact in 1-D by Gauss) | classical closure vs the analytic F-function C-V: <= 1e-4 after Richardson. DG vs SP C-V: C_inv/C_ox within 5 %, and the depletion-to-inversion transition voltage within 20 mV | yes (as G1-3) | cross-method: ssac at 1 Hz on the closed device == DC dQ/dV to 1e-6 (QE is the LF C-V); ssac on a closure-free 0.5 V device must also match (where both work) |
| G1-6 | detailed balance | -- | max abs(phi_n), abs(phi_p) <= 1e-12 V at every node; abs(ln(np/n_i^2) - (Ln + Lp)/V_t) <= 1e-12; hole flux <= 1e-12 of q mu p V_t/L_D; `quantum_srh=True` -> U = 0 to 1e-12 relative | n/a | negative control: `quantum_srh=False` breaks phi_p flatness by > 1 mV (GATE E's known result) |
| G1-7 | mesh convergence and cross-dimension | Richardson on 3 meshes; 2-D laterally uniform MOS-C (hand-built `create_2d_mesh`, right triangles, 3 columns) | order >= 1.7 for N_inv and centroid; 2-D == 1-D to <= 1e-10 rel (exact geometry, P-b) | n/a | 2-D with a 0.3-1-3 graded lateral spacing must still equal 1-D (no lateral artefact) |
| G1-8 | imref split | classical: Q_s(psi_s, V_ch) analytic with n0 -> n0 exp(-V_ch/V_t) | as G1-1 | n/a | refusal test: a 2-contact pn diode with QE and no `allow_imref_split` raises; with it, the forward current is visibly wrong. This proves QE cannot be a silent transport substitute |
| G1-9 | FD Delta string | numpy `fd_potential_shift` | <= 1e-12 rel, u in [1e-6, 136] | n/a | -- |

**SP inversion oracle (G1-4).** The existing `SchrodingerPoisson1D.solve_self_consistent` is electron-only (doping =
ionized net donors). Inversion needs ionized acceptors plus a **classical Boltzmann hole density in the nonlinear
Poisson**. The plan is a validation-local oracle in `validation/bipolar_density_gradient.py`:

* it calls `SchrodingerPoisson1D.solve_schrodinger` / `density` for the electrons, with the same isotropic model
  masses and g_v as the DG (model consistency, as in GATE C);
* it runs its own predictor-corrector Poisson Newton (Trellakis-type) with p(psi) = N_A exp(-(psi - psi_b)/V_t);
* the electron E_F is the body Fermi level.

Self-checks: the hbar -> 0 limit (mass x 1e4) reproduces G1-1's analytic Q_s to 1e-3, and the free-gas deficit
holds. An anisotropic (m_l = 0.916, m_t = 0.19, two- and four-fold valley) SP run is **reported, not gated**, as
the "real Si" context. An optional sanity check, reported only: the Fang-Howard variational centroid 3/b,
b = (48 pi m_z q^2 N*/(eps hbar^2))^{1/3}, N* = N_dep + 11/32 N_inv. It is a low-T single-subband formula and
indicative at 300 K only; Ando-Fowler-Stern RMP 54, 437 (1982), citation unverified.

### 1.7 Placement and runtime

* `tests/test_quasi_equilibrium.py` (new). Contents: G1-0 (5 s), G1-1 at 2 meshes x 3 psi_s (8 s), G1-2 (6 s), G1-3
  at one gamma and 0 to 1.3 V in double (10 s), G1-6, G1-8, G1-9. About 40 s at `-n 2`.
* `validation/bipolar_density_gradient.py` gains GATE G (strong inversion: G1-3 full matrix, G1-4, G1-5, G1-7).
  Estimate: 60 SP solves at 160 states (~10-20 s each), so **nightly, ~20 min**.

### 1.8 Risks and decisions

* **Risk: DG accuracy in strong inversion may simply be the model's limit.** DG at gamma = 1 under-predicts the dark
  space; at gamma_wall the sign of the residual may flip with triangular wells. G1-4's tolerances are fidelity
  expectations registered before measurement. If they fail, the outcome is a documented model result plus
  possibly a separate inversion-calibrated gamma. It is not a numerics bug, and the tolerances will not be retuned
  silently.
* **Risk: DEVSIM lacks `atan`.** Then FD QE is refused in v1 (decision D1-c).
* **Risk: the wall node's Lambda under QE.** At the wall, `setup_dg_insulator_wall` gives u = floor, Lambda row
  `c - u^2`, and c row = QE. So `Lambda_wall = V_t ln(u_floor^2/c_cl) + ...`, the same -27.6 V_t as now. It is gated
  (floor 1e-6 -> 1e-4 moves N_inv <= 1e-4).

| decision | recommendation |
|---|---|
| D1-a: accept the closure (exact for the DC MOS-C, LF C-V, channel slice) as THE strong-inversion route, and record QFL/Slotboom variables as measured-no-help | **yes** |
| D1-b: home and name: `physics_bipolar_dd.setup_quasi_equilibrium_carrier` | yes (it is DD physics; DG composes) |
| D1-c: if DEVSIM has no `atan`, ship Boltzmann-only QE and refuse FD | yes, refuse; do not approximate |
| D1-d: also gate the realistic anisotropic Si SP (two valley sets) | report only, no gate |
| D1-e: 2-D MOSFET strong-inversion DG transport | separate follow-up (needs a 2-D DG oracle) |

### 1.9 Dependencies and effort

* Depends on P-a (the two-Bernoulli switch moves the classical DD strings) and on item 4's
  `check_dc_well_posed`/sigma_min tooling, which G1-0 reuses.
* Effort 6-9 days: closure plus classical gates 2 days, DG integration 1 day, SP inversion oracle 2-3 days, C-V
  and convergence 1-2 days, docs 1 day.

---------------------------------------------------------------------------------------------------

## 2. Retire the unipolar `setup_dg_hard_wall` rows in favour of `setup_dg_insulator_wall`

### 2.1 Goal and physics

The hard wall has three defects:

* it pins the carrier row (n = u^2) and pins Lambda = -k V_t;
* the wall node is therefore a carrier sink whose SG flux is driven by the pinned Lambda jump;
* the converged profile is the spurious wide-depletion branch: a deficit of 3-1000 N0 L_q against the BVP's ~1.2.

This is characterized, not validated (`test_dg_hard_wall_pins_the_wall_rows`). The insulator wall has u = floor, a
Lambda row `c - u^2 = 0`, and a natural (zero-normal-flux) continuity row. It conserves carriers by construction,
and its Lambda is whatever makes the wall-edge flux vanish.

For a frozen flat potential and Boltzmann statistics the DG closes in closed form (D2d GATE B):

    z(t) = L_q int_0^t ds/sqrt(F(s)),  F(t) = t^2 ln t^2 - t^2 + 1,  deficit = I n0 L_q,  I = 1.187207

The derivation is in the D2d doc, sec. 3. It was verified there against the SP free gas: the ratio 0.7734 and
gamma_wall = 1.6717.

### 2.2 Current code state and reuse

* `setup_dg_insulator_wall` already handles a unipolar region: `_BIPOLAR_DG["electrons"]` keys on `QSqrtN`, which
  the unipolar region carries.
* `seed_bipolar_dg` works on a unipolar region too: `bipolar_dg_carriers` keys on `QSqrtN` and
  `ElectronCurrentDG`, both present.
* The unipolar flux `_JN_DG` is the kahan3 form. A unipolar wall edge sits at |v| ~ 27.6, so the cancellation of
  D2d sec. 1.6 applies (6.5e-5 loss). **After P-a, `_JN_DG` is the two-Bernoulli form, and this item must not land
  before it.**
* `set_dg_gamma` co-ramps `wall_lambda_pin`. That becomes dead once no device creates the parameter.

### 2.3 Design

* `setup_dg_hard_wall(device, contact, *, lambda_pin_factor=15.0)` becomes a **deprecated shim**. It validates
  `lambda_pin_factor` (the existing guard test stays), issues `DeprecationWarning` (category decision D2-a), and
  calls `setup_dg_insulator_wall(device, contact)`.
* `lambda_pin_factor` is accepted and ignored, with the docstring saying so.
* It is removed from `__all__` in v0.16.
* No `wall_lambda_pin*` parameters are created. `set_dg_gamma`'s co-ramp branch stays for one release, because a
  device built by old pickled scripts could still carry the parameter. It is deleted with the shim.
* `seed_dg_from_solution(device, region, *, walls=(), u_floor_rel=1e-6)` gains the wall-aware seed by
  **delegating** to the `seed_bipolar_dg` kernel (renamed internally `_seed_dg`). Both public names call it: one
  kernel. `walls=()` reproduces the old seed to round-off.

**Results that move.**

* Every unipolar DG device with a hard wall moves from the wide-depletion branch to the BVP-consistent profile. In
  the repo that is only `tests/test_density_gradient.py` (the hard-wall bar, its pin test and its rows test) and
  `tests/_dg_calllog.py` (which P-a turns into an agreement check).
* No validation uses the hard wall (grep: none).
* `docs/physics_depth_roadmap.md` and `docs/roadmap_v0.5_integration_photonics.md` mention it and get updated.

### 2.4 Ordered steps

1. After P-a lands, run the existing hard-wall bar through `setup_dg_insulator_wall` and record:
   * the deficit;
   * Lambda_wall;
   * the wall-edge flux;
   * iterations.
2. Write the new gates (2.5) and the shim.
3. Replace `test_dg_hard_wall_pins_the_wall_rows` with the physics gates. Keep the input guard. Rewrite
   `..._pin_is_a_ramped_device_parameter` as "the shim creates no pin parameter and set_dg_gamma is a no-op for
   walls".
4. Re-record or convert the unipolar call-log check, in coordination with P-a.
5. Update docs and release notes.

### 2.5 Verification gates

| id | gate | oracle | tolerance | fails on old code? | refutation |
|---|---|---|---|---|---|
| G2-1 | closed-form dark space, unipolar | z(t), I = 1.187207 (mpmath quadrature in the test); fixture: n0 = 1e21 m^-3 (n0/N_c = 2e-4, so g - 1 ~ 7e-5: Boltzmann holds), frozen flat psi, wall 0.05 nm, L_q = 1.18 nm at m = 0.35 m0 | deficit rel <= 2e-4 (bipolar measured 3e-6..2e-4 at similar h/L_q; plus 7e-5 FD residue); profile max abs(dz)/L_q <= 1e-3 | **yes**: hard wall gives 3-1000 N0 L_q | mesh refinement 0.1 -> 0.05 -> 0.025 nm: observed order >= 1.7 toward I |
| G2-2 | gamma scaling | L_q proportional to sqrt(gamma) | deficit(gamma)/deficit(gamma/4) = 2 +- 1e-3 | yes | -- |
| G2-3 | self-consistent fixed point | `density_gradient.dg_correct_density_1d` on the DG solution's own psi (GATE D twin), n0 = 1e23 | max error <= 2e-3 of peak (bipolar measured 2.7e-4) | yes (the old self-consistent leg raised Convergence failure) | -- |
| G2-4 | unipolar == bipolar DG on the same device (holes negligible: n-type 1e23, n_i 1e10 m^-3) | `setup_bipolar_dg(m_eff_n_kg=)` + the same wall | <= 1e-10 rel in n, Lambda (same kernel and flux after P-a; holes enter Poisson at p/n ~ 1e-26) | n/a | -- |
| G2-5 | conservation | -- | wall-edge SG current <= 1e-12 of q mu n0 V_t/L_q; terminal current at the back contact <= 1e-12 of the same scale | **yes**: the old wall is a sink (measure its flux; expect O(1e-3..1)) | -- |
| G2-6 | floor independence | -- | u_floor_rel 1e-6 -> 1e-4 moves the deficit <= 1e-4 rel (the bipolar measured 8.3e-5) | n/a | -- |
| G2-7 | degenerate ITO bar (the old fixture, 4e26, FD g) | no closed form (Boltzmann BVP not exact under FD, audit C-8) | characterization replaced by: the deficit lies in [0.7, 1.5] N0 L_q, i.e. the order of the closed form, not tens; equals the bipolar DG on the same bar (G2-4 analogue) to 1e-10 | **yes** (old: tens) | negative control: re-installing the old pinned carrier row (`n = u^2` on the c row) on the new wall must reproduce the wide branch, proving the row and not something else is the cause |

### 2.6 Placement and runtime

`tests/test_density_gradient.py`: G2-1, G2-2, G2-4, G2-5, G2-6, G2-7, the shim, and the guard. About 15 s. G2-3 and
the convergence-order sweep go in `validation/dg_dd_in_newton.py` as gate E (nightly, ~1 min).

### 2.7 Risks and decisions

* **Risk:** a user script relying on `lambda_pin_factor` changes silently. Mitigation: the DeprecationWarning, the
  docstring and the release note.
* **Risk:** the degenerate bar converges more slowly. Mitigation: the wall-aware seed. Extended precision is in
  reserve only if criterion attainability shows up (D2d sec. 1.6).

| decision | recommendation |
|---|---|
| D2-a: shim with `DeprecationWarning` (tests escalate it, so the existing direct-call tests must expect it) vs hard removal now | shim for v0.15, remove in v0.16 |
| D2-b: `seed_dg_from_solution` gains `walls=` (delegating to the one seed kernel) | yes |
| D2-c: rename `setup_dg_insulator_wall`? It is no longer "bipolar" in its module section | keep the name; move the docstring section header |

**Dependencies and effort:** P-a is a hard prerequisite. Effort 1.5-2 days. Group it with item 1 (same module and
harness).

---------------------------------------------------------------------------------------------------

## 3. 2-D impact ionization on the element reconstruction

### 3.1 Goal and physics

    I_sub = int (alpha_n(|E|) |J_n| + alpha_p(|E|) |J_p|) dV ,  alpha(E) = a exp(-b/E)    (Chynoweth; vOdM Si constants in module)

**What the current 2-D edge form actually computes.** On DEVSIM's 2-D tensor mesh the hypotenuse couples are zero.
For each triangle leg pair the form `sum_e alpha(|E_e|) |J_e| couple_e L_e` therefore converges, under refinement,
to

    I_edge = int [alpha(|E_x|) |J_x| + alpha(|E_y|) |J_y|] dV        (componentwise, NOT isotropic)

This is a closed-form statement that the test can check exactly. For a field at angle theta to the mesh,

    I_edge/I_true = [cos t e^{-b/(E cos t)} + sin t e^{-b/(E sin t)}] / e^{-b/E}

At b/E = 4 and theta = 45 deg that is 0.27. It is exact only for mesh-aligned transport. So it is wrong for any
genuinely 2-D current pattern: junction corners, current crowding at contacts, the MOSFET drain corner.

**What DEVSIM's 2-D element reconstruction computes.** Read from `src/Geometry/TriangleElementField.cc` (devsim
main).

* For each triangle, every edge pair (i, j) is solved as a 2x2 system from the two edge projections, giving a vector
  at their shared node.
* Element edge i's vector is the average of its two pair vectors, weighted by the **partner** edge's
  ElementEdgeCouple.
* This is exact for any uniform vector field.
* On a right triangle the hypotenuse couple is 0. Both legs then take the (leg_x, leg_y) pair vector
  `(E_x-edge, E_y-edge)`, and the hypotenuse carries weight 2 ElementNodeVolume = 0.

So the element form evaluates `alpha(sqrt(E_x^2 + E_y^2)) sqrt(J_x^2 + J_y^2)` per triangle, weighted by the
triangle area. That is the isotropic integrand. It is the same `element_from_edge_model` reconstruction 2-D BTBT
already uses (`tunneling.py`: a uniform 30 deg field reproduced to 2e-16) and the same one 3-D impact ionization
uses.

### 3.2 Current code state and reuse

`impact_ionization.impact_generation_edges` works as follows:

* 1-D: edge form (exact in 1-D);
* 2-D: edge form;
* 3-D: `_impact_generation_elements_3d`, which runs the exact-volume check and the element reconstruction.

The plan **generalizes the 3-D function to `_impact_generation_elements(device, region, params)` for dim in (2, 3)**:
components `"xyz"[:dim]`, and true simplex volume `|det|/dim!`. That gives one kernel and no 2-D copy. The
exact-volume check becomes a shared helper, `_element_volumes_exact(device, region) -> ratio`, that also covers
2-D (sum of 2 ENV vs the sum of triangle areas, refuse at > 1e-9). A 2-D gmsh or other unstructured mesh with
obtuse triangles is then refused, consistent with P-b.

### 3.3 Design

    impact_generation_edges(device, region, params, *, quadrature: str = "auto")
    substrate_current(device, region, params, *, depth_m=None, quadrature: str = "auto")
      "auto"    -> 1-D: "edge" (exact), 2-D/3-D: "element"
      "edge"    -> allowed in 1-D and 2-D only (legacy directional estimate; documented as such); 3-D raises
      "element" -> 2-D/3-D only; 1-D raises (DEVSIM has no 1-D element models)

**This changes 2-D results by default.** That is user decision D3-a, recommended: element by default. The legacy
`"edge"` value stays available with no warning, because selecting it explicitly is an informed choice and
warnings are errors in the suite.

The returned dict keys are unchanged (`weight_volume` = 2 ENV per element edge). `hci.py`'s stale docstring
("2D layered meshes only, the 3D tet quadrature is guarded") is corrected.

### 3.4 Ordered steps

1. **Baseline on the current code.** Record every 2-D number the validations and tests print: `impact_ionization_isub`
   GATE B, C (-5, -1, 0 V) and D (2 nm vs 1 nm).
2. Generalize the element kernel, add the 2-D exactness check and the `quadrature=` kwarg.
3. Write the oracle fixtures (3.5) and run them on both quadratures.
4. Switch the default and re-pin the moved numbers, each with its evidence (the 3-D D2c doc's "moved pins" table
   format).
5. Docs: module header, D2c doc section 8 item 3 closed, release note.

### 3.5 Verification gates

| id | gate | oracle | tolerance | fails on old (edge) code? | refutation |
|---|---|---|---|---|---|
| G3-1 | uniform-field bar at theta = 0, 30, 45 deg | **exact discrete solution**: a uniformly doped bipolar n-region (1e22) on a 2-D tensor mesh; every boundary node is a contact with `Potential = psi_eq + E0 (x cos t + y sin t)` and n, p pinned to their neutral values; psi linear and n uniform solve the discrete Poisson and SG continuity exactly (SG with uniform n: J_e = q mu N E_e, zero discrete divergence on a tensor grid). I = alpha(E0) q mu N E0 Area | element rel <= 1e-12 at every theta | **yes** at 30 and 45 deg | the edge result must equal the componentwise closed form I_edge (3.1) to 1e-12. This proves the old error is exactly the directional masking, not a bookkeeping slip |
| G3-2 | laterally uniform diode 2-D == 1-D | the 1-D edge quadrature on the same lines (abrupt p+/n+ 1e24, -3 V, tau 1e-10, as the 3-D test) | rel <= 1e-9 (3-D measured 5e-10) for element; edge must also pass (it is exact for aligned fields) | no (both exact): this is the "does not break 1-D-like cases" gate | graded lateral spacing 0.5-5 nm: unchanged |
| G3-3 | a genuinely 2-D field with an independent reference | **harmonic Dirichlet fixture**: same construction as G3-1 with psi = psi_eq + V0 (x^2 - y^2)/L^2 (and a second one, V0 sinh(kx) sin(ky)/sinh(kL)). The box method is exact for quadratics on tensor meshes, so the discrete psi equals the nodal interpolant and the solver error is 0. Reference: I_ref = int alpha(abs(grad psi)) q mu N abs(grad psi) dA by tensor Gauss-Legendre (400 x 400) in numpy, cross-checked at 3 points with mpmath dblquad (rel 1e-12) | element: observed order >= 1.8 on h, h/2, h/4; Richardson limit within 1e-6 of I_ref. Edge: converges to I_edge,inf (computed by the same quadrature of the componentwise integrand) and differs from I_ref by >= 10 % at the chosen V0 | **yes** | (i) rotating the fixture 90 deg (x <-> y) changes nothing (both); (ii) the sinh case (curved field lines, not quadratic) has nonzero discrete error; the gate is then only the order and limit, which proves the order claim is not an artefact of the quadratic's exactness |
| G3-4 | mesh convergence order (real device) | 2-D p+ well corner in n-substrate (junction curvature), -5 V, no closed form | element: successive differences shrink at order >= 1.5 over 3 meshes; the edge result is reported (its limit differs) | -- | the element I_sub must lie above the edge one at the corner (the isotropic |E| >= each component), a sign check that needs no reference |
| G3-5 | cross-dimension | the G3-3 harmonic fixture extruded in y on the 3-D Kuhn tensor mesh (3-D element path) | 3-D and 2-D element both within 1e-5 of I_ref at the finest common mesh, and their Richardson limits agree to 1e-6 | n/a | -- |
| G3-6 | refusal | -- | a 2-D mesh with an obtuse triangle (hand-built via `create_2d_mesh` + `add_2d_region` from a gmsh file) raises `NotImplementedError` naming the exact-mesh requirement | n/a | -- |
| G3-7 | side-effect hygiene | -- | element models created are deleted; pre-existing `ElectricField_x/y` (2-D BTBT) are only read; the BTBT solve before and after is bit-identical | n/a | -- |

**Moved pins, predicted (to be measured in step 1 and step 4):**

* `impact_ionization_isub` GATE B (mesh-aligned bar): unchanged to round-off.
* GATE C and D (2-D diode `validation.bipolar_diode_2d` with edge-metal contacts): change where the current
  crowds near the contacts and the junction ends. The 5 % mesh-convergence tolerance of GATE D is expected to
  tighten. Re-pin from the measured value with the G3-4 order as the evidence.
* 3-D numbers: unchanged (same kernel, same arithmetic order; gate: the 3-D tests bit-identical).

### 3.6 Placement and runtime

* `tests/test_impact_ionization.py` gains G3-1 (3 angles, 21 x 11 mesh, 3 s), G3-2 (5 s), G3-3 at 2 meshes (8 s),
  G3-6 and G3-7. About 25 s.
* `validation/impact_ionization_isub.py` gains GATE E (G3-3 full order, G3-4, G3-5). Nightly; the file is already
  the longest validation at 1463 s, so add <= 5 min and reuse its devices.

### 3.7 Risks and decisions

* **Risk:** the corner diode on `LayeredDevsimBuilder` is not available as a 2-D well geometry. Fallback: a
  hand-built `create_2d_mesh` device (as the G3-1 fixtures).
* **Risk:** J reconstruction for non-uniform SG fluxes is first-order pointwise. The integrated O(h^2) is a
  prediction, gated by G3-3 (ii) and G3-4 at a conservative order >= 1.5.

| decision | recommendation |
|---|---|
| D3-a: switch 2-D default to element (changes 2-D I_sub; HCI lifetimes that consume it move) | **yes** |
| D3-b: keep `quadrature="edge"` without a warning | yes (explicit opt-in) |
| D3-c: also apply the shared exact-volume check to 2-D/3-D BTBT (`tunneling.setup_btbt` relies on sum 2 ENV = volume but does not check) | yes, as a small follow-up; it is not in this item's gates |

**Dependencies and effort:** independent of the other items (P-b only). Effort 2-3 days. Can run in parallel.

---------------------------------------------------------------------------------------------------

## 4. Floating-body detection (singular DC Jacobian from missing Dirichlet data)

### 4.1 Goal and physics

The R26-class null mode: a carrier component with no row of its continuity equations replaced by a contact (or
interface) row. The fluxes telescope and the generation terms enter the electron and hole rows with opposite
signs. So a left null vector exists **exactly**:

    w = 1 on every Electron-continuity row and every Hole-continuity row of the component  =>  w^T F(x) == 0 for all x
    (with U == 0 additionally each species separately:  w_n = 1 on electron rows,  w_p = 1 on hole rows)

and `w^T J = 0`, so the DC Jacobian is rank-deficient by one (or by two). The analogous **right** null vector is
v = 1 on the Potential columns of an electrostatically connected component with no row depending on absolute psi
(no Dirichlet, no gate row, no equilibrium-physics n(psi)). That is a pure Neumann Poisson.

The recombination doc measured the R26 old fixture at `|w^T J|/col-norms <= 1.1e-16` in all 94 matrices, and at a
scaled sigma_min of 1.3-1.6e-15 before the fix and 1.5e-5 after.

**Near-floating** (the item-1 MOS inversion, or a reverse-biased isolated well inside a contacted region) is not
exactly singular. There the scaled sigma_min falls continuously (1.1: 3.8e-11 to 2e-17). It is detectable only
numerically, and it is a convergence-failure diagnosis, not a setup error.

### 4.2 Current code state and reuse

* Nothing checks well-posedness today.
* `eq_registry` records region, contact and interface equations. DEVSIM itself answers
  `get_contact_equation_list` / `get_contact_equation_command` and `get_interface_equation_list`, which also see
  raw `ds.contact_equation` calls made by test fixtures.
* `get_matrix_and_rhs(format="csc")` is already used by noise and tunneling tests. Note the keys `av`/`ai`/`ap`/`rhs`
  (as used in `tests/test_devsim_tunneling.py`).
* `get_equation_numbers(device, region, variable)` gives the row and column blocks (measured: contiguous per
  variable).
* `transient._device_is_converged` already parses `solve(info=True)`.

### 4.3 Design

New module `dynameta/carriers/well_posedness.py`:

    check_dc_well_posed(device, *, numeric: str = "candidates", tol_singular: float = 64 * 2.2e-16) -> WellPosedReport
        # static: components x carriers, candidate left/right null vectors from topology + contact/interface rows
        # numeric "candidates": ONE get_matrix_and_rhs; measure r = |w^T J|_inf / (sum_i |w_i| |J_i,:|_1) per candidate
        # numeric "full": additionally equilibrated sigma_min (dense SVD <= 5000 dof, else splu + inverse iteration on J^T J)
        # raises IllPosedDCError (subclass of ValueError) when a candidate has r <= tol_singular, naming the region,
        # the carrier(s), the conserved quantity, and the fixes (pin the majority at a contact; a floating contact
        # with a zero-current circuit node; allow_floating)
    allow_floating(device, region, *, carriers=("electrons", "holes"), reason: str) -> None
        # an explicit acknowledgement recorded in eq_registry; the check skips that candidate (no warning: explicit)
    diagnose_singular(device) -> str
        # "full" sigma_min + the support of its singular vector (variable block, region, node x-range), for error messages

**Error policy: error only when static AND numeric agree.**

* The static analysis proposes candidates; it might not know about a custom contact row.
* The numeric check confirms them. A custom row that breaks conservation makes r = O(1), so there is no false
  positive.
* A static candidate refuted numerically is reported in the `WellPosedReport` (debug) and does not raise.
* Exact singularity is a structural fact, so the error is not a warning. The test suite's warnings-as-errors policy
  is then irrelevant.

**Hooks.**

* `dc_solve.solve_dc` (method newton or gummel) calls `check_dc_well_posed` once per device per equation-set
  fingerprint: the hash of DEVSIM's equation lists plus contact and interface equation commands, cached. This
  covers `LayeredDevsimBuilder` and `devsim_3d`, which route through `solve_dc`.
* On a Newton convergence failure, `solve_dc` appends `diagnose_singular` to the re-raised error, but only when the
  failure trace shows the stall signature (non-monotone relative updates while abs/rel of the other equations are
  converged). The 1.1 trace has exactly this shape. The message then says "near-singular DC Jacobian
  (sigma_min/sigma_max = 2e-17), mode in Electrons, region s, x in [0, 12] nm: a weakly pinned minority; see
  setup_quasi_equilibrium_carrier".
* **Raw `ds.solve` in test fixtures** (the R26 fixture used it) is not covered by solve_dc. Proposed (D4-a): a
  `tests/conftest.py` autouse, session-scoped wrapper of `devsim.solve` that runs the cached static+candidate check
  for `type="dc"`, with a `@pytest.mark.floating_ok` opt-out.
* Transient solves are never checked: the time term `C/dt` regularizes, and a floating body is legitimate
  transiently.

**Legitimate floating cases, made expressible.**

1. *Floating contact* (SOI body, floating gate on a semiconductor): a contact whose carrier rows are pinned
   relative to an unknown potential, with a circuit node carrying zero current. The static analysis sees a carrier
   row replaced, so it is well-posed and needs no flag. A documented recipe plus a test.
2. *Isolated well with a fixed charge*: DEVSIM has no global-integral constraint row. The honest options are
   `allow_floating(..., reason=)` with the docstring stating that the DC answer along that mode is then set by
   round-off, or a transient solve from a known initial charge. D4-c covers a real charge-constraint row.
3. *Near-floating* (weakly coupled well, MOS inversion): never an error. Diagnosis on failure only.

**Cost.**

* Static: negligible (lists and set unions).
* Numeric: one assembly per fingerprint, and only when the static analysis found a candidate. Measured context: a
  1-D bipolar assembly is milliseconds; the 76k-node 2-D DD Newton iteration is 0.27-0.36 s per iteration (which
  includes the factorization, so assembly is less). In extended precision it is 6-20x more, still one-off.
* "full": dense SVD at 804 dof took under 1 s. The `splu` path for large devices is used only in `diagnose_singular`
  after a failure.

### 4.4 Ordered steps

1. **Static topology.** Carrier components: regions joined by interfaces whose interface equations reference that
   carrier's continuity (continuous or thermionic). Potential components: regions joined by Potential interface
   equations. Rows replaced: contact equations on those equations with a `node_model` (DEVSIM's command query).
2. **Candidate vectors** via `get_equation_numbers`; the numeric r measure.
3. Rebuild the R26 old fixture (Potential-only contact) in a test; measure r for w_n + w_p (expect <= 1e-15).
   Measure the fixed fixture (expect O(1)).
4. Hook into `solve_dc` with the fingerprint cache; write `IllPosedDCError`.
5. `diagnose_singular` plus the failure-message enrichment; the MOS inversion trace from 1.1 as its fixture.
6. `allow_floating`; the floating-contact recipe and test.
7. The conftest wrapper (if D4-a is accepted); run the full devsim test set (976 tests) to find any other fixture
   that was silently singular. Each finding is fixed or marked `floating_ok` with a reason.

### 4.5 Verification gates

| id | gate | tolerance | fails on old code? | refutation |
|---|---|---|---|---|
| G4-1 | R26 old fixture: raises `IllPosedDCError` naming region `bulk`, electrons+holes, "net charge conserved" | r <= 1e-15 measured, threshold 64 eps = 1.4e-14 | yes: the old code solved it silently and the answer depended on the BLAS | the fixed fixture (majority pin) passes with r >= 1e-3 |
| G4-2 | U == 0 slab (tau = 1e30, no Auger/radiative), no carrier pinned: rank deficiency 2 detected (both species separately) | as G4-1 | yes | pin electrons only: still raises for holes alone |
| G4-3 | pure-Neumann Poisson (no Potential Dirichlet, DD physics): right null vector detected | abs(J 1_psi) <= 64 eps | yes | equilibrium (`physics_equilibrium`) region with no Dirichlet: NOT flagged (n(psi) pins the absolute level), proven by its r = O(1) |
| G4-4 | no false positives | the check runs on every device of the devsim test set and the 33 devsim validations; zero raises except the documented `floating_ok` | -- | a custom Robin recombination contact row (written raw, unknown to the registry) on an otherwise floating slab: static candidate, numeric r = O(1), no raise |
| G4-5 | floating contact is well-posed | zero-current circuit node contact on a body: no raise; DC solution independent of BLAS build (bitwise or <= 1e-14) | -- | -- |
| G4-6 | near-floating diagnosis | the 0.80 V MOS failure message contains "near-singular", sigma_min/sigma_max <= 1e-15 and "Electrons" | -- | the same device at 0.30 V converges and produces no message |
| G4-7 | cost | the check adds <= 5 % to `LayeredDevsimBuilder.solve` on the reference metasurface (static only, no candidate) and <= 1 assembly when a candidate exists | -- | -- |

### 4.6 Placement and runtime

`tests/test_well_posedness.py`: G4-1 to G4-6, about 20 s. The conftest wrapper adds about 0 s per test when there
is no candidate. G4-4 is the full-suite run: a one-off during implementation, then implicit.

### 4.7 Risks and decisions

* **Risk:** heterojunction or thermionic interfaces whose equations reference both regions' carriers in forms the
  static parser misreads. Mitigation: the numeric confirmation, so the failure mode is "miss", not "false raise".
  G4-4 measures misses only through "full" spot-checks.
* **Risk:** `get_matrix_and_rhs` on an extended device needs the extended solver context. Use the existing
  `XP.extended_solver()`.
* **Risk:** the conftest wrapper interacts with xdist and the process-global circuit trap (integration note sec.
  7). The wrapper must not create devices or circuits; it only reads.

| decision | recommendation |
|---|---|
| D4-a: conftest autouse wrapper of raw `devsim.solve` in tests (catches fixture-level floating bodies like R26) | **yes**, with a `floating_ok` marker |
| D4-b: exact singularity is an **error** (`IllPosedDCError`), near-singularity a **diagnosis in the failure message** (never a warning) | yes |
| D4-c: a real DC charge-constraint row for an isolated well (needs a global-integral equation DEVSIM lacks; probably a circuit-node trick, unproven) | defer; ship `allow_floating` + the transient route |

**Dependencies and effort:** item 1 reuses `diagnose_singular` for G1-0. Effort 2.5-3.5 days, plus 1 day for the
full-suite sweep. Group it with item 5 (both live in or next to `dc_solve`).

---------------------------------------------------------------------------------------------------

## 5. Density-scaled absolute tolerance in `solve_dc`

### 5.1 Goal and numerics

By 0.1 (C1, C2), DEVSIM's device absolute error is

    A = max_regions sum_eq max_nodes |dx_eq|

and convergence needs A < abs_tol AND R < rel_tol. Two facts follow:

* The attainable floor of A is about kappa u S, where `S = max_r sum_eq max_nodes |x_eq|` and u = unit round-off.
  kappa was measured at 2-4 ulp median and up to 36 ulp in D2d sec. 1.6.
* The default abs_tol = 1e10 m^-3 is below that floor once S > 1e10/(kappa u). For kappa = 36 that is
  S > 1.25e24 m^-3. It is below one ulp for S above ~6e25 (ulp(6e25) = 8.6e9).

Newton then stalls "waiting for a lucky update" (D2d sec. 1.6: 10-18 iterations at relative 1e-15, one step ran
out). By C1, raising abs_tol above the floor **cannot loosen** the accuracy delivered by rel_tol. So the principled
default makes the absolute criterion exactly "attainable", never binding where it cannot be met:

    abs_tol_default(device) = max(1e10, K * u_eff * S_pre),   K = 1024,
    u_eff = 2^-53 (double)  or  2^-113 if extended_precision.any_enabled() (128-bit residual; the floor measured ~1e-33 relative)
    S_pre = the DEVSIM aggregation (max over regions of the sum over that region's equations of max|x|), evaluated
            on the pre-solve solution

**Derivation of K = 1024.** K u = 2^-43 = 1.14e-13, i.e. 512 ulp of S (ulp = 2^-52 relative). That is 14x the
worst measured floor (36 ulp). It also covers growth of S within one continuation step, up to about 14x (a
ln(14) x V_t = 68 mV bias step at the peak). It is still 8.8x tighter than the builders' existing `n_bg * 1e-12`
rule, so it never loosens their criterion.

**Byte-identical region.** The floor of 1e10 binds whenever `S_pre < 1e10/(1024 x 2^-53) = 8.8e22 m^-3`. Every
lightly doped device is then **byte-identical** by construction: the same number reaches DEVSIM. Extended devices
are byte-identical up to S = 1e10/(1024 x 9.6e-35) ~ 1e41, i.e. always.

**Per-variable scaling is not available.** DEVSIM has one absolute and one relative limit per solve (0.1).
Rescaling the variables (densities in 1e24 units) would change every model string. It is rejected.

### 5.2 Current code state and reuse

* `solve_dc(..., abs_tol=1e10, ...)` passes the value through.
* The builders already compute their own:
  * `LayeredDevsimBuilder._dc_abs_tol`: `max(1e10, n_bg x 1e-12)`; bipolar `max(that, 1e13)`;
  * `devsim_3d`: `max(1e13, n_bg x 1e-12)` / `max(1e10, max n_bg x 1e-12)`.
* They pass it explicitly, so they are **unaffected** by a new solve_dc default.
* No library caller relies on the 1e10 default (grep: every `solve_dc` call in `dynameta/` passes abs_tol). Users
  and the DG validations (`ABS_ERR = 1e10` with raw `XP.solve`) are the beneficiaries. **The blast radius inside the
  repo is small, and the plan says so.**

### 5.3 Design

    solve_dc(device, *, method="newton", abs_tol: float | None = None, rel_tol=1e-5, ...)
    dc_solve.default_abs_tol(device=None) -> float        # public, documented; the ONE implementation
    dc_solve.ABS_TOL_FLOOR = 1e10; dc_solve.ABS_TOL_ULPS = 1024

* `abs_tol=None` (new default) resolves through `default_abs_tol`. An explicit float is passed unchanged
  (backward compatible). The resolved value goes into the debug log and into any convergence-failure message.
* **This changes the default only for devices with S > 8.8e22 m^-3** (decision D5-a).
* Gummel sub-solves use the same resolved value.
* The builders' private rules **stay** in this item. Migrating them to `default_abs_tol` would change their numbers
  (their 1e-12 x n_bg vs 1.14e-13 x S). Offered as D5-b.
* Not in scope, noted: the relative criterion's floor `|dx|/(|x| + 1e-10)` for variables near 0 (Lambda tails,
  psi ~ 0). It is a separate unattainability. DEVSIM has `Equation::setMinError` in C++, but **whether Python
  exposes it is unverified**.

### 5.4 Ordered steps

1. **Floor measurement fixture.** A degenerate unipolar ITO bar (4e26) and the GATE C bipolar accumulation
   (7.5e25), in double. Compute an extended-precision reference x* (to 1e-30). Then run the double Newton with
   abs_tol = 0 and max_iter 60 and record max|dx| per iteration and |x_k - x*|. That yields kappa (the
   distribution) for the docstring.
2. Implement `default_abs_tol` (aggregation read from DEVSIM's equation list, `get_equation_command` ->
   `variable_name`, node values) and the `abs_tol=None` default.
3. Gates 5.5. Then grep the tests for `solve_dc(` without abs_tol (the Gummel tests) and confirm they are
   byte-identical (light doping).
4. Docs: the solve_dc docstring (the 0.1 criterion facts), release note.

### 5.5 Verification gates

| id | gate | oracle | tolerance | fails on old default? | refutation |
|---|---|---|---|---|---|
| G5-1 | degenerate devices converge | ITO bar 4e26 (classical DD, 1 V gate ramp), GATE C 7.5e25 accumulation, R27-a n+ 1e25; double, both OpenBLAS builds | each step converges within <= 2 iterations of the extended run's count; zero max_iter exhaustion | yes: measure the stall/exhaustion count on old (D2d recorded 10-18 extra iterations and one exhaustion) | -- |
| G5-2 | converged answers unchanged beyond the criterion | the extended reference x* | for every node: abs(x_new - x*) <= rel_tol x abs(x) + abs_tol_new; and abs(x_new - x_old) <= abs_tol_new where old converged | -- | adversarial: tiny MINORITY densities (1e8 next to a 1e26 majority, item-1 fixture at 0.6 V): the minority must still be accurate to rel_tol against x* (rel is a max over nodes, so abs never protected it); shown explicitly |
| G5-3 | light doping byte-identical | -- | for S < 8.8e22, `default_abs_tol` returns exactly 1e10 and every array is bitwise equal to the explicit `abs_tol=1e10` solve: the 1e22/1e23 diode fixtures, `gummel_vs_newton`, `test_dc_solve_gummel` | -- | at S slightly above the threshold, the iteration count is unchanged and the results agree to rel_tol |
| G5-4 | K derivation | step-1 measurement | measured kappa_max x 10 <= 512 ulp (else K is raised and documented) | -- | inject a 10x density growth within one step (a 60 mV accumulation step): still converges |
| G5-5 | extended devices | -- | `default_abs_tol` == 1e10 on any extended device (byte-identical); the DG validation numbers unchanged | -- | -- |
| G5-6 | equilibrium (Poisson-only) device | -- | unchanged: abs is in V there and 1e10 was already vacuous (documented, gated by bitwise equality) | -- | -- |

### 5.6 Placement and runtime

`tests/test_dc_solve_abs_tol.py`, about 30 s (G5-1 on two fixtures, G5-3 to G5-6). The step-1 kappa measurement is
a validation (`validation/dc_abs_tol_floor.py`, nightly, ~2 min).

### 5.7 Decisions

| decision | recommendation |
|---|---|
| D5-a: `abs_tol=None` -> `max(1e10, 1024 u_eff S)`; changes results (within rel_tol) only for S > 8.8e22 | **yes** |
| D5-b: migrate `LayeredDevsimBuilder._dc_abs_tol` and the devsim_3d rules to `default_abs_tol` (one kernel; their numbers move within rel_tol) | yes, as a separate commit after G5-2 passes on the builders' fixtures |
| D5-c: route the validations' hard-coded `ABS_ERR = 1e10` through `default_abs_tol` | no for the DG validation (it is extended: no change anyway); yes for new fixtures |

**Dependencies and effort:** none (item 4 shares the file). Effort 1.5-2 days.

---------------------------------------------------------------------------------------------------

## 6. Summary of dependencies, effort, grouping

| workstream | items | depends on | effort | CI tests added | nightly added |
|---|---|---|---|---|---|
| WS-N (numerics core) | 5, then 4 | none (4 after 5 in the same file) | 4-5.5 days + 1 day suite sweep | ~50 s | ~2 min |
| WS-Q (quantum) | 2, then 1 | **P-a merged**; WS-N's `diagnose_singular` for G1-0 | 7.5-11 days | ~55 s | ~21 min |
| WS-I (impact ionization) | 3 | P-b (done) | 2-3 days | ~25 s | <= 5 min |

The added CI test time is about 2.2 min, well inside the 150-min cap. WS-N and WS-I can start immediately and run
in parallel. WS-Q waits for P-a.

**What is hard, honestly.**

* Item 1's numerics are now understood and the fix is small. Its **physics** gate against an inversion SP
  (G1-4) is the real work: a new oracle, and DG fidelity that may fall short of the accumulation numbers.
* Item 4's static topology across heterointerfaces is the part most likely to have misses.
* Item 3's integrated order is a prediction until G3-3 runs.

## 7. Sources consulted

* devsim/devsim main, raw source:
  * `src/math/Newton.cc`
  * `src/Equation/Equation.cc`
  * `src/Geometry/Region.cc`
  * `src/Geometry/Device.cc`
  * `src/models/TriangleEdgeFromEdgeModel.cc`
  * `src/Geometry/TriangleElementField.cc`

  All via raw.githubusercontent.com, 2026-11-04. Read and paraphrased (0.1, 3.1).
* Abstract or record pages only, full text unreachable:
  * Wettstein, Schenk & Fichtner, IEEE TED 48(2), 279-284 (2001), DOI 10.1109/16.902727
  * Ancona, J. Comput. Electron. 1, 435-443 (2002), DOI 10.1023/A:1020732515391
  * Garcia-Loureiro et al., IEEE TCAD 30(6), 841-851 (2011), DOI 10.1109/TCAD.2011.2107990 (eprints.gla.ac.uk/53759)
  * Ferron et al., Silvaco Simulation Standard (2004) (silvaco.com/?p=12374)
  * Pinnau & Ruiz (kluedo.ub.rptu.de/files/1864/Pinnau_Ruiz.pdf, HTTP 403)
  * an NTRS scan (no text layer)
* In-tree, read:
  * the D2d, D2c, D1b and integration audit docs;
  * the files listed in the task;
  * `tunneling.py`;
  * `devsim_layered.py` and `devsim_3d.py` (abs_tol rules);
  * `transient.py` (`_device_is_converged`);
  * `schrodinger_poisson.py` and `density_gradient.py` (API).
