# Bipolar density-gradient: a quantum potential for holes as well as electrons (D2d)

DEVSIM phase, release v0.14.0, workstream D2d, branch `wip/dv_dg` (off `feat/devsim-2.11-v0.14` at
`e46bd25`). Before this, the in-Newton density-gradient (DG) model existed only for electrons on a
unipolar region (`physics_density_gradient.setup_dg_quantum_correction`). Its oxide hard wall
(`setup_dg_hard_wall`) was EXPERIMENTAL and known to converge to a wrong, wide-depletion profile.

## What changed

Everything is **opt-in and off by default**:

* Nothing new runs unless `setup_bipolar_dg` is called.
* The existing unipolar DG is byte-identical to `e46bd25`. Its full DEVSIM definition-call log
  (127 calls) hashes to the digest recorded on `e46bd25`.
* The bipolar DD without DG is byte-identical too (the v0.13 digest, 217 calls).

Section 4 lists both gates.

| Piece | Where |
|---|---|
| Shared DG kernel `_install_dg_variables` (u = sqrt(c), Lambda, the tie and the Lambda-equation). The unipolar electron DG now calls it too, with identical calls. | `carriers/physics_density_gradient.py` |
| `setup_bipolar_dg(device, region, *, m_eff_n_kg=None, m_eff_p_kg=None, gamma_n=1.0, gamma_p=1.0, quantum_srh=True)` | same |
| `setup_contact_bipolar_dg(device, contact)`: ohmic rows for the DG carriers | same |
| `setup_dg_insulator_wall(device, contact, *, u_floor_rel=1e-6)`: impenetrable oxide wall that conserves carriers | same |
| `seed_bipolar_dg(device, region, *, walls=(), u_floor_rel=1e-6)`, `bipolar_dg_carriers`, `DG_SRH_TERM` | same |
| `set_dg_gamma` co-ramps the hole `b_dg_p` when present. A unipolar region's calls are unchanged. | same |
| `_repoint_contact_current_to_dg` generalized to any carrier. Its defaults are the old electron call. | same |
| `srh_reference_densities` / `srh_rate_expr` gain `scale=None`, the quantum intrinsic-level factor. `None` returns the old strings character-for-character. | `carriers/physics_traps.py` |
| Oracles | `validation/bipolar_density_gradient.py`, `tests/test_bipolar_density_gradient.py`, `tests/_dg_calllog.py` |

## 1. Formulation

### 1.1 One quantum potential per carrier

The potentials are in volts, with the sign convention of the existing unipolar code:

    Lambda_n = b_n lap(sqrt n) / sqrt n ,   b_n = gamma_n hbar^2 / (6 m_n q)
    Lambda_p = b_p lap(sqrt p) / sqrt p ,   b_p = gamma_p hbar^2 / (6 m_p q)

**Sources.**

* Ancona & Tiersten, Phys. Rev. B 35, 7959 (1987): density-gradient theory of the Si inversion layer,
  where the gas's equation of state gains a gradient term.
* Ancona & Iafrate, Phys. Rev. B 39, 9536 (1989): the coefficient derived from the Wigner (hbar^2)
  expansion.
* Wettstein, Schenk & Fichtner, IEEE TED 48, 279 (2001): one quantum potential per carrier type,
  each with its own effective mass and fit factor. The quantum potential is a separate solution
  variable, and the classical density relation is generalized with Lambda as a band-edge shift.

`gamma = 1` is the high-temperature (Boltzmann) value. It is Ancona's `r = 3` in his convention, as
checked in `docs/audit/2026-07-05-deep-audit.md`.

**Check of gamma = 1 (sec. 1.2).** The first-order Wigner-Kirkwood correction for a Boltzmann gas
in a potential energy V is

    n = n_cl [1 - (hbar^2 beta^2/12m) lap V + (hbar^2 beta^3/24m) |grad V|^2]

Write this as `n = n_cl exp(beta Lambda_E)` and use `V = -kT ln n_cl`. Then

    Lambda_E = (hbar^2/12m) [lap ln n + 1/2 |grad ln n|^2] = (hbar^2/6m) lap(sqrt n)/sqrt n

so `gamma = 1` in `b = gamma hbar^2/(6 m q)`.

**Citations were checked against the code, not the papers.** They were written from memory: the
papers were not available on box A. Each equation used here is derived or verified independently
inside this note or by an oracle below. Please check the bibliographic details before the release
notes quote them.

### 1.2 Modified quasi-Fermi relations: the quantum potentials are band-edge shifts

    E_c,eff = E_c - q Lambda_n ,          E_v,eff = E_v + q Lambda_p
    n = n_i exp((psi + Lambda_n - phi_n)/V_t) ,    p = n_i exp((phi_p - psi + Lambda_p)/V_t)

Near a wall `lap(sqrt c)/sqrt c < 0`. So `Lambda_n, Lambda_p < 0` there, and both carriers are
repelled.

**Equilibrium.**

    n p = n_i^2 exp((Lambda_n + Lambda_p)/V_t)

This equals `n_i^2` only where the quantum potentials vanish.

**The Scharfetter-Gummel flux.** Electrons drift in `psi + Lambda_n` and holes in
`psi - Lambda_p`. That is exactly the Lundstrom-Schuelke quasi-field slot that bandgap narrowing
already uses (`D_c`, `D_v`, in `bipolar_options`). So the DG fluxes are built on top of the region's
own quasi-field: the FD g-factor, the BGN band edges and all.

    vdiff_g_dg  = vdiff_g  + (Lambda_n@n0 - Lambda_n@n1) / (V_t g_enh)
    vdiff_gp_dg = vdiff_gp - (Lambda_p@n0 - Lambda_p@n1) / (V_t g_enh_p)

`ElectronCurrentDG` / `HoleCurrentDG` are the classical bipolar strings with
`vdiff_g -> vdiff_g_dg` and `Bern_g -> Bern_g_dg` (holes: `_gp -> _gp_dg`). The Bernoulli argument is
clamped at +-200, as in the unipolar code; the clamp never binds at convergence.

With `fd_enhancement=True`, Lambda shifts eta before the g-division:

    n = N_c F_1/2(eta + Lambda_n/V_t)

This is the form Wettstein et al. use for Fermi-Dirac DG. `gamma` then carries the degeneracy
calibration discussed in `density_gradient.py` (audit C-8).

### 1.3 Detailed balance: the quantum SRH reference (`quantum_srh=True`, default)

The quantum potentials move the band edges. Equilibrium therefore has
`n p = n_i^2 exp((Ln+Lp)/V_t)`, not `n_i^2`. The SRH kernel's `n_i^2` would then generate pairs in
equilibrium everywhere Lambda != 0, which is mainly the dark space.

The fix is the same substitution BGN uses, applied on top of it:

    n_i'^2 -> n_i'^2 s^2 ,  n1' -> n1' s ,  p1' -> p1' s ,   s = exp((Lambda_n + Lambda_p)/(2 V_t))

It is done in ONE place, `physics_traps.srh_reference_densities(scale=...)`.

**How it enters.** The correction goes in through the one additive mechanism, as
`RTerm_dgsrh = U_SRH(quantum) - USRH`, with its pair rate `PTerm_dgsrh`, so the noise inventory sees
it. Its Lambda derivatives are added to the generation composites' Jacobian. The exponent is clamped
at +-200, which matters only during Newton transients.

**Measured effect (GATE E).** In an inverted MOS:

| | max abs(phi_n) | max abs(phi_p) |
|---|---|---|
| `quantum_srh=True` | 3.3e-16 V (1.7e-10 V before the flux fix, section 1.6) | 5.6e-17 V |
| `quantum_srh=False` | 6 mV off the wall node | 62 V at the wall node (the classical reference is meaningless there) |

### 1.4 Boundary conditions

**Ohmic contacts: `Lambda = 0`, `u^2 = c`.** The one ohmic contact rule
(`contact_potential_residual` and the neutral carrier pins) fixes two things:

* `c` = the CLASSICAL charge-neutral density at the contact's psi;
* `phi_c` = bias.

The modified relation `c = c_cl exp(Lambda_c/V_t)` then forces `Lambda_c = 0`. That is the
bulk-reservoir condition of the DG literature, and it follows from the classical rule rather than
being added to it. The u row is the bulk tie `u^2 - c`, so u follows whatever the contact pins,
including BGN, ionization and traps.

**Semiconductor/insulator: an impenetrable wall.** The existing unipolar code models the oxide as
an infinite barrier on a contact-tagged boundary (`setup_dg_hard_wall`). D2d stays with that choice:
the confined carrier's wavefunction vanishes there, `u -> 0`. The alternative is a penetration
model, where the DG equation is solved into the oxide with the band offset (e.g. Ancona et al.,
IEEE TED 47, 2310 (2000)). It is NOT implemented (Limits).

**Why the wall rows are new.** The existing hard wall pins the CARRIER row (`n = u^2`) and pins
Lambda to a regularization depth. That makes the wall node a carrier sink, driven by the pinned
Lambda jump: this is the spurious wide-depletion branch its docstring records. `setup_dg_insulator_wall`
uses these rows at each wall node instead:

| Row | Equation |
|---|---|
| u | `u = u_floor` (`u_floor_rel x sqrt(c_classical)`, positive because u is `positive`-updated) |
| Lambda | `c - u^2 = 0` (the bulk tie, moved to this row) |
| c | the NATURAL continuity row: zero normal flux through the insulator |

Lambda at the wall is therefore not pinned. It is whatever makes the wall edge's SG flux vanish with
`c(wall) = u_floor^2`, measured `-27.63 V_t = V_t ln(1e-12)` for both carriers. That is the finite,
floor-set stand-in for the physical logarithmic divergence (`u Lambda -> 0`).

The floor only sets the wall node's own dead density. Moving it from 1e-6 to 1e-4 moves the
dark-space deficit by 8.3e-5 (gated).

`setup_dg_hard_wall` is untouched (unipolar byte-identity); see Open questions.

### 1.5 The DEVSIM equation form

Each DG carrier adds two node solutions with their own equations. The pattern is the unipolar
kernel's.

| variable | region equation | ohmic contact | insulator wall |
|---|---|---|---|
| `QSqrtN` / `QSqrtP` (u) | `u*u - c` (node), `positive` | `u*u - c` | `u - u_floor` |
| `QLambda` / `QLambdaP` | edge flux `b (u@n0 - u@n1)/L` + node `Lambda u` (so `b lap u = Lambda u`) | `Lambda` | `c - u*u` |
| `Electrons` / `Holes` | continuity on `ElectronCurrentDG` / `HoleCurrentDG`, generation composite kept | the one ohmic rule; current read from the DG flux | natural (zero flux) |

Parameters are `b_dg` (electrons) and `b_dg_p` (holes), plus `*_full` for the gamma ramp
(`set_dg_gamma`).

### 1.6 Numerics: the wall-edge flux and the validation's precision (2026-11-03)

**The flux is written with two Bernoulli functions.** The bipolar DG currents are

    Jn ~ n@n1 B(-v) - n@n0 B(v) ,   Jp ~ -( p@n1 B(v) - p@n0 B(-v) )

(`_JN_DG_BIPOLAR`, `_JP_DG_BIPOLAR`). The first version reused the classical `kahan3` strings,
which spell B(-v) as B(v) + v. For v << 0 that is the difference of two O(|v|) numbers. Measured
against mpmath on DEVSIM 2.11.2, it loses 1.9e-7 at v = -21, 6.5e-5 at v = -27.6 and 100 % at
v = -40. DEVSIM's own B(-v) stays at 2e-16. A classical edge never sees |v| ~ 20. An insulator-wall
edge does by construction: the wall node holds c = 1e-12 c_cl, so its zero-flux edge sits at
|v| ~ 27.6.

The wall node's Lambda is set only by that edge's flux. With the `kahan3` form it converged
linearly, then sat in a period-3 Newton cycle at a relative update of 1.9-3.8e-7, against
REL_ERR = 1e-7. GATE C (holes) passed or failed on BLAS round-off: it failed with scipy's OpenBLAS
and passed with Elmer's. The same loss was GATE E's "1.7e-10 V" flatness. It is 4.4e-16 V with the
two-Bernoulli form in double. The Bernoulli argument is clamped at +200 only. A two-sided clamp
makes both Bernoullis constant beyond |v| = 200, so a wall edge loses its whole Lambda dependence in
a Newton transient. That gave an exact zero column (GATE B and GATE E with `quantum_srh=False`
failed under both BLAS builds). Gate: `test_dg_edge_flux_is_accurate_at_large_drift`, 1e-13 against
mpmath, measured 3e-16. The unipolar DG keeps `_JN_DG`, because its definition is pinned by a
call-log digest.

**Update 2026-11-04 (user decision): every SG flux now goes through one kernel,
`dynameta.carriers.sg_flux`** -- the unipolar DG and the classical DD included (open item 5 below,
resolved). `_JN_DG_BIPOLAR` / `_JP_DG_BIPOLAR` / `_JN_DG` are gone. The kernel evaluates B(v) and
B(-v) like the form above but combines them sign-selected (the B(x) + x = B(-x) identity only on the
non-negative argument, one shared Bernoulli per branch): the same large-|v| accuracy (this file's
gate now measures 3.3e-16 / 2.2e-16) with ~4x less ssac round-off near equilibrium than the plain
bracket. The upper-only clamp is kept. docs/audit/2026-11-04-sg-flux-two-bernoulli.md.

**The validation runs the extended Newton.** In double, two of its convergence criteria are finer
than the arithmetic:

* DEVSIM's relative update is `|dx|/(|x| + 1e-10)`. That formula reproduces the reported RelError
  node by node. Where the dark-space tail of Lambda passes ~1e-17 V (GATE B, 11-16 nm from the
  wall), REL_ERR = 1e-7 therefore demands `|dLambda| < 1e-17 V`. That is below one ulp of the wall's
  Lambda (-0.71 V).
* ABS_ERR = 1e10 m^-3 is 0.6 ulp of GATE C's 7.5e25 accumulation peak. The converged update floor
  measured 2-4 ulp (median) and up to 36 ulp.

So the solves waited at round-off for a lucky iterate. `extended_precision.enable` on every device
of `validation/bipolar_density_gradient.py` keeps both criteria as written and makes them
attainable. Every gate number then agrees between the two BLAS builds, mostly bit for bit. GATE C's
metrics move by <= 1e-15. The test file takes 63 s (54 s with double solves).

## 2. API (typical 1-D MOS)

    BP.setup_bipolar_region(dev, reg, ...)                  # classical bipolar DD (+ options, traps, BTBT)
    BP.setup_contact_ohmic_bipolar(dev, "sub")
    # wall: Potential row of your choice (Dirichlet psi_s here), carriers natural
    ...classical solve...
    DG.setup_bipolar_dg(dev, reg, m_eff_n_kg=0.26*M_E, m_eff_p_kg=0.49*M_E)   # gamma_n/p = 1
    DG.setup_contact_bipolar_dg(dev, "sub")
    DG.setup_dg_insulator_wall(dev, "wall")                 # after the classical solve (floor)
    for k, fr in enumerate((0.1, 0.3, 1.0)):
        DG.set_dg_gamma(dev, reg, fr)
        if k == 0:
            DG.seed_bipolar_dg(dev, reg, walls=("wall",))
        ds.solve(...)

Pass only `m_eff_p_kg` to give holes alone a quantum potential (and likewise for electrons). The
oracle harness in `validation/bipolar_density_gradient.py` has an adaptive gamma ramp (`_dg_ramp`)
and a surface-potential continuation (`_ramp_param`).

## 3. Oracles (each independent of the code under test)

Run `python -m validation.bipolar_density_gradient`: 42 s on box A, all PASS. The tests in
`tests/test_bipolar_density_gradient.py` take 29 s at `-n 2`.

Model masses are isotropic, Si-like: `m_n = 0.26 m0` and `m_p = 0.49 m0`, which give
`L_q = sqrt(b/V_t) = 1.3746 / 1.0013 nm` at gamma = 1.

### GATE A (task a): classical limit

A 1e23/1e22 Si p-n diode with the bipolar DG ACTIVE at `b = 0` reproduces the classical bipolar DD.
The match covers Potential, Electrons, Holes and the terminal current, at +0.4, +0.2 and -0.5 V.

| Result | Value |
|---|---|
| max rel | **1.2e-12** |
| Lambda | 0 |

### GATE B (task b, the analytic DG offset): closed-form dark space

Setup: an intrinsic bar against the wall, with the potential frozen and flat. The DG equations then
integrate in closed form:

    z(t) = L_q int_0^t ds / sqrt(F(s)),   F(t) = t^2 ln t^2 - t^2 + 1,   t = sqrt(c / c_bulk)
    deficit = int (c_bulk - c) dz = I c_bulk L_q,   I = int_0^1 (1 - t^2)/sqrt(F) dt = 1.187207

| gamma | carrier | L_q [nm] | deficit/(c L_q) rel. error | profile max abs(dz)/L_q |
|---|---|---|---|---|
| 1 | electrons | 1.3746 | -3.2e-6 | 1.4e-4 |
| 1 | holes | 1.0013 | -2.6e-5 | 2.3e-4 |
| 1/4 | electrons | 0.6873 | -8.7e-5 | 4.3e-4 |
| 1/4 | holes | 0.5006 | -1.9e-4 | 7.6e-4 |

**Oracle sanity check.** The Schrodinger-Poisson free gas (zero field, non-degenerate) at the same
infinite wall has the exact profile `c/c_bulk = 1 - exp(-2 m kT z^2/hbar^2)`. Its deficit is measured
`0.626609` against the exact `sqrt(pi)/(2 sqrt 2) = 0.626657`, in units of `c hbar/sqrt(m kT)`.

So the DG(gamma=1)/QM dark-space ratio is **0.7734**. That is the known shortfall of the
first-order gradient theory. It defines

    gamma_wall = (3 pi/4) / I^2 = 1.6717

which is the gamma at which the DG wall deficit equals the exact quantum one. It is fixed a priori by
this limit and used, unfitted, in GATE C.

### GATE C (task b, vs Schrodinger-Poisson; the p-channel is the new physics)

Strong ACCUMULATION:

* electrons on n-type and HOLES on p-type;
* `N = 1e24 m^-3`, `abs(psi_s - psi_b) = 0.15 V`, fixed surface potential;
* FD-consistent DD (`N_c`, `N_v` of the SP model: `g_v = 6 / 2`).

The oracle is the self-consistent `SchrodingerPoisson1D` (bulk buffer, slab mode, 160 states,
converged). Holes are modeled by the exact electron<->hole mirror. The metrics are the excess sheet
density and its centroid within 20 nm of the wall, and the peak depth.

| | SP | classical DD | DG gamma = 1 | DG gamma_wall = 1.672 |
|---|---|---|---|---|
| electrons: sheet | 2.391e16 m^-2 | +177.4 % | +21.3 % | **+5.6 %** |
| electrons: centroid | 3.973 nm | -63.9 % | -17.5 % | **-5.1 %** |
| electrons: peak | 2.40 nm | 0 (AT the wall) | 1.61 nm (0.67x) | 2.02 nm (0.84x) |
| holes: sheet | 2.860e16 m^-2 | +124.7 % | +17.3 % | **+4.0 %** |
| holes: centroid | 3.320 nm | -55.5 % | -14.7 % | **-3.7 %** |
| holes: peak | 1.80 nm | 0 (AT the wall) | 1.23 nm (0.68x) | 1.48 nm (0.82x) |

**Gates and why these tolerances.**

* **gamma = 1** (the textbook value, no calibration) must close at least 70 % of the
  classical-to-SP gap in both sheet and centroid. Measured: 86-88 % (sheet) and 73 % (centroid).
  The peak must lie at 0.5-1.5x the SP depth.
* **gamma_wall** must land within 8 % of SP on both, with the peak at 0.75-1.25x. Measured: 3.7-5.6 %.

The 8 % covers the worst measured 5.6 % with margin. The residual has the sign the free-gas analysis
predicts (DG dark space slightly too thin). It is also of the size expected from the
Fermi-Dirac-vs-Wigner calibration mismatch: the SP peak reaches `c/N_c = 0.37-0.59`.

**Regime scan (gamma = 1, same harness).** The DG residual stays at 7-21 % of sheet and 7-18 % of
centroid while the classical error grows from 28 % to 177 %:

| case | classical error | DG gamma = 1 error |
|---|---|---|
| N = 1e23, 0.075 V | sheet +55 % / +40 %, centroid -35 % / -28 % | sheet +9.7 % / +7.4 %, centroid -8.4 % / -6.8 % |
| N = 3e23, 0.1 V | sheet +100 % / +73 % | sheet +14 % / +11 % |

(Pairs are electrons / holes.) At `gamma_wall` the 1e23 case is within 1.5 % for both carriers
(electrons +1.3 % sheet / -1.5 % centroid, holes +0.8 % / -1.0 %).

### GATE D: fixed point

Boltzmann accumulation (3e23, 0.1 V), both carriers. The converged coupled DG density equals the
post-hoc DG boundary-value problem (`density_gradient.dg_correct_density_1d`, scipy `solve_bvp`)
evaluated on the DG solution's OWN potential.

| carrier | max error (fraction of the peak) |
|---|---|
| electrons | **2.7e-4** |
| holes | **4.0e-4** |

The gate is 2e-3. This verifies the coupled assembly and wall rows against an independent numpy
solver.

### GATE E (task c): equilibrium, bipolar at the wall

Setup: an inverted MOS, p-type 3e23, `psi_s - psi_b = 0.60 V`, so `n_s ~ 1e5 p_s`. Both carriers are
quantum at the wall.

| Quantity | Measured |
|---|---|
| max abs(phi_n), `phi_n = psi + Lambda_n - V_t ln(n/n_i)` | **3.3e-16 V** (4.3e-15 on Elmer's OpenBLAS; 1.7e-10 V before section 1.6) |
| max abs(phi_p), `phi_p = psi - Lambda_p + V_t ln(p/n_i)` | **5.6e-17 V** |
| `n p = n_i^2 exp((Ln+Lp)/V_t)` at every node | 7.1e-15 / 1.6e-13 (6.4e-9 before section 1.6) |
| `n p = n_i^2` in the neutral bulk | 4.2e-7, the 1e-6 solve tolerance |
| Lambda at the wall | -27.6 V_t for both carriers |
| Inversion peak | 1.74 nm off the wall; classical peak AT the wall |

With `quantum_srh=False`, phi drifts by 6 mV (section 1.3).

### GATE F (task d): transport

A 1-D Si diode (1e23 / 1e22, 2 um, tau = 1e-7 s). Forward bias uses the terminal current. Reverse
bias uses the net-recombination current `q int U dx`. That equals the terminal current in steady
state, but the reverse terminal value is a majority-carrier flux about 1e-12 of its drift/diffusion
parts. It jitters by about 1e-3 between equivalent converged solves, classical and DG alike. The
classical `abs(I_a + I_c)/abs(I)` is 6.9e-3 at -0.5 V: that is the classical DD, not DG.

| bias | classical I [A/m^2] | dI/I at gamma = 1 | dI/I at gamma = 0.5 | ratio | max abs(Lambda) |
|---|---|---|---|---|---|
| +0.40 V | 16.72 | -8.28e-5 | -4.15e-5 | 1.997 | 0.020 V_t |
| +0.20 V | 1.901e-2 | +2.24e-3 | +1.12e-3 | 2.000 | 0.014 V_t |
| -0.50 V (q int U dx) | -1.046e-3 | +6.77e-3 | +3.38e-3 | 2.004 | 0.013 V_t |

The gates:

* change < 1e-2;
* change <= 2 max abs(Lambda)/V_t, i.e. first order;
* gamma ratio 2 +- 0.2;
* forward conservation no worse than the classical's.

Measured forward conservation: 3.2e-8 (classical 7.8e-8).

The largest change, 0.7 % at reverse bias, is the field-driven Wigner term in the depletion region,
`Lambda ~ b (E/2V_t)^2`, entering the generation through `quantum_srh`. It is physical, first order
and linear in gamma.

## 4. Defaults off (task e) and the "fails on e46bd25" check

| Guard | Result |
|---|---|
| Unipolar DG call log (`tests/_dg_calllog.py`: `setup_dg_quantum_correction`, `setup_contact_dg`, `setup_dg_hard_wall`, `seed_dg_from_solution`, `set_dg_gamma`; 127 calls) | digest `6bcbe925...` recorded on `e46bd25` BEFORE any change: **identical** on the final tree |
| Bipolar DD call log (`tests/_bipolar_calllog.py`, 217 calls) | v0.13 digest `ac63536e...`: **identical** |
| `srh_rate_expr()` / `srh_reference_densities()` defaults | the legacy strings, pinned |
| `validation/dg_dd_in_newton.py` (unipolar in-Newton DG) | gates A-D PASS with the same numbers as before (5.6e-16 / 8.4e-14 / 7.5e-16 / 0.0) |
| `tests/test_bipolar_density_gradient.py` copied onto a pristine `e46bd25` worktree | **13 failed, 2 passed**. The two passes are the defaults-off digest gates, which must pass there. Every physics gate fails, including both p-channel (`holes`) oracles. |

**Regression.** `test_density_gradient`, `test_traps`, `test_bipolar_recombination_bgn_ionization`,
`test_audit_2026_07_17_wave7_carriers`, `test_api_surface` and `test_validation_runner` run at
`pytest -n 2`: **102 passed**.

## 5. Limits

* **Inversion beyond weak inversion does not converge** in the coupled Newton. The
  minority-carrier inversion layer is pinned only through the substrate contact, so the electron
  chain spans about 13 decades. Measured with N_A = 3e23:
  * 0.60 V converges at full gamma, both carriers;
  * 0.70 V fails at the last ramp step;
  * 0.75-0.80 V fails at the first step.

  It is the same for holes on n-type (mirror), with or without the wall, with psi frozen, without SRH,
  and from an exact BVP seed.

  The classical DD is itself marginal there. It needs SRH coupling to converge at 0.80 V (with
  tau = 1e3 s it fails), and it fails at 0.95 V even with SRH.

  Accumulation and majority depletion converge for both carriers up to the strongest case tried
  (1e24 / 0.15 V, peak about 1e25 m^-3). Strong-inversion DG needs a lateral source (2-D MOSFET) or a
  quasi-Fermi / Slotboom-variable formulation (open).
* **1-D only validated.** The kernel is dimension-agnostic (edge models). `seed_bipolar_dg` uses
  KD-tree wall distances. But no 2-D/3-D oracle was run.
* **The wall is a contact-tagged boundary.** The Potential row is the caller's: Dirichlet psi_s in
  the oracles. A DEVSIM oxide REGION with an interface is not supported for the DG rows. The DG
  variables live in one region only, and DEVSIM's interface equations do not express "replace this
  side's row".
* **No oxide penetration model.** The barrier is infinite. Tunneling or penetration into the oxide
  would need the DG equation solved in the insulator with the band offset.
* **Only the built-in SRH channel follows the quantum reference.** The other channels keep the
  classical `n_i`:
  * Auger and radiative;
  * the discrete trap levels and interface traps;
  * BTBT's Hurkx D.

  They are therefore not in detailed balance inside the dark space. For the built-in channel this is
  measured (GATE E). For the others it is not gated.
* **Noise** refuses a DG region by construction: `noise_analysis` demands `edge_model == ElectronCurrent`.
  DG noise sources are not defined.
* **Gummel / `dc_solve` splitting** of the DG variables was not exercised. Use the coupled Newton.
* **Fermi-Dirac DG**: gamma absorbs the degeneracy (audit C-8). GATE C is FD-consistent and lands
  within 5.6 % at gamma_wall, but that is a regime result, not a derivation.

## 6. Open questions (user decisions)

1. **Default gamma.** The code default is `gamma_n = gamma_p = 1`, the textbook Wigner value. Against
   Schrodinger-Poisson, `gamma_wall = 1.672` is 3-4x more accurate (GATE C). It is fixed a priori by
   the exact free-gas wall, not fitted to MOS data. Should `setup_bipolar_dg` default to it, or should
   a helper expose it?
2. **`quantum_srh=True` by default.** I made the detailed-balance-consistent reference the default,
   because without it equilibrium is wrong in the dark space (GATE E). Keep it, or make it opt-in?
3. **Retire the unipolar `setup_dg_hard_wall` rows?** `setup_dg_insulator_wall` already works on a
   unipolar region carrying `QSqrtN`. Its zero-flux rows fix the known wide-depletion defect. But
   switching the unipolar wall changes `setup_dg_hard_wall`'s call log and its characterization test.
   That was out of scope here.
4. **Strong inversion** (Limits, first item): worth a quasi-Fermi-variable DG formulation?
5. **RESOLVED 2026-11-04** (user decision; docs/audit/2026-11-04-sg-flux-two-bernoulli.md: one
   kernel `sg_flux` for every family, digests re-recorded, numeric cross-version guards). Original
   text: **The `kahan3` flux elsewhere (section 1.6).** The unipolar DG (`_JN_DG`) and the classical DD
   strings (`physics_bipolar_dd`, `physics_drift_diffusion`) still spell B(-v) as B(v) + v. That is
   harmless at classical |v|. A unipolar region given `setup_dg_insulator_wall` or
   `setup_dg_hard_wall` would carry the same wall-edge loss. Switching them changes the v0.13
   bipolar and e46bd25 DG call-log digests and moves every DD result at round-off.
6. **Extended precision in the validation (section 1.6).** The validation's criteria are finer than
   double precision. The alternative to the extended Newton is to change REL_ERR / ABS_ERR to
   attainable values. That would loosen REL_ERR for every variable, because DEVSIM has one criterion
   per solve.

## 7. Merge notes

* The only shared file outside the DG module is `physics_traps.py`: a keyword-only `scale=None` on
  `srh_reference_densities` / `srh_rate_expr`. Default output is unchanged.
* `physics_density_gradient._repoint_contact_current_to_dg` keeps its old positional signature and
  default behavior.
* The bipolar DG reads `vdiff_g` / `vdiff_gp` BY NAME. A sibling that redefines them later (e.g. a
  heterojunction quasi-field) is picked up automatically.

## Release-note lines

- Bipolar density-gradient (opt-in): `physics_density_gradient.setup_bipolar_dg` gives electrons AND
  holes their own quantum potential. Each has its own effective mass and gamma. The potentials are
  folded into the Scharfetter-Gummel quasi-fields like bandgap narrowing, and they compose with the
  FD g-factor and BGN. Ohmic rows (`Lambda = 0`) come from `setup_contact_bipolar_dg`. Default off;
  the unipolar DG and the bipolar DD are byte-identical (call-log digests).
- `setup_dg_insulator_wall`: an impenetrable-oxide DG wall with zero normal flux. It reproduces the
  closed-form DG dark space for both carriers to 3e-6 / 3e-5. Against Schrodinger-Poisson
  accumulation layers (electrons and holes), DG at gamma = 1.672 is within 5.6 % in sheet charge and
  centroid; the classical error is 55-177 %.
- Detailed balance under DG: the built-in SRH channel follows the quantum band edges
  (`quantum_srh=True`, via `physics_traps.srh_reference_densities(scale=)`). Equilibrium
  quasi-Fermi levels are flat to 5e-15 V.
- Limit: DG inversion layers converge only up to weak inversion in 1-D (minority chain pinned at the
  substrate contact). Accumulation and depletion converge for both carriers.
