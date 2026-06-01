# Implementation notes (grounded recipes for the open roadmap items)

Distilled from research; the authoritative sources are cited. These ground the
deep-physics implementations so they don't have to be re-derived.

## Bipolar drift-diffusion + SRH (DEVSIM)

Source of truth (bundled, on disk): `devsim/python_packages/simple_dd.py`
(currents) + `simple_physics.py` (node models, SRH, contacts). The existing
DynaMeta DD is electrons-only with an FD-enhanced Scharfetter-Gummel current;
the FD enhancement only changes the Bernoulli *argument* and is orthogonal to the
holes/SRH/Poisson structure below (apply the same FD factor to the hole current).

DEVSIM sign convention (residual = q*divergence form): `Jn = +q*...`, `Jp = -q*...`;
time-charge `NCharge=-q*n`, `PCharge=+q*p`; SRH source `Gn=-q*USRH`, `Gp=+q*USRH`;
Poisson `-q*(p - n + NetDoping)`. Holes = the electron expression with `q -> -q`
AND the `vdiff` drift term moved from the `@n1` to the `@n0` node:

    vdiff   = (Potential@n0 - Potential@n1)/V_t ;  Bern01 = B(vdiff)   # shared
    ElectronCurrent = +q*mu_n*EdgeInverseLength*V_t*kahan3(Electrons@n1*Bern01, Electrons@n1*vdiff, -Electrons@n0*Bern01)
    HoleCurrent     = -q*mu_p*EdgeInverseLength*V_t*kahan3(Holes@n1*Bern01,    -Holes@n0*Bern01,    -Holes@n0*vdiff)

SRH (one node model into BOTH continuity equations, opposite signs):

    USRH = (n*p - n_i^2) / (taup*(n + n1) + taun*(p + p1))     # n1=p1=n_i defaults
    ElectronGeneration = -q*USRH   (node_model of ElectronContinuityEquation)
    HoleGeneration     = +q*USRH   (node_model of HoleContinuityEquation)

Bipolar Poisson charge: `PotentialNodeCharge = -q*(Holes - Electrons + NetDoping)`.
Bipolar ohmic contact: pin n to `n0=1/2(N+sqrt(N^2+4 n_i^2))`, p to `n_i^2/n0`
(swap on p-type), derivative literal "1"; `contact_equation(... edge_current_model
=ElectronCurrent / HoleCurrent)`. Potential contact adds the built-in offset
`+ifelse(N>0, -V_t*log(n0/n_i), V_t*log(p0/n_i))`.

Staged solve (the convergence path): (1) potential-only pre-solve; (2) seed
Electrons/Holes from the Boltzmann equilibrium node models (set_node_values
init_from=IntrinsicElectrons/Holes); (3) build the 3-variable coupled system
(Potential, Electrons, Holes) -> ds.solve (fully coupled Newton); (4) bias ramp.
`variable_update`: Potential "log_damp", Electrons/Holes "positive".

Validation gate: a p-n diode J-V (monotonic, sign/ideality-correct) + reduce to
the electron-only result in the unipolar limit.

## Quantum confinement (Schrodinger-Poisson + density-gradient)

Two routes; use BOTH (SP as the equilibrium reference, DG for device coupling).

### 1D self-consistent Schrodinger-Poisson (equilibrium reference)
BenDaniel-Duke tridiagonal Schrodinger (mass at half-nodes), symmetrize for a
nonuniform mesh (Tan/Snider Eqs 8-15), Dirichlet psi=0 at the barriers; discard
states whose psi at the mesh edge is not ~0 (not truly bound). DEGENERATE 2D-
subband density (closed form, NOT Boltzmann):

    n(z) = sum_i (g_s g_v m*_DOS kB T / (2 pi hbar^2)) ln[1+exp((E_F - E_i)/kB T)] |psi_i(z)|^2

For ITO: single Gamma valley -> g_v=1, g_s=2 -> prefactor = m*/(pi hbar^2) (no
Si-style x2). E_F is PINNED by the contact (flat quasi-Fermi level), not by
isolated-well neutrality. Overflow guard: ln(1+exp(x)) -> x for x>40 (strongly
degenerate). CAVEAT: ITO band is nonparabolic at 1e26-1e27 m^-3 -> use a
density-dependent m* or expect subband-spacing error.

Convergence: the Trellakis predictor-corrector (NOT naive Picard, which sloshes):
move the nonlinearity into a NONLINEAR Poisson solve with an a-priori quantum
density that rigidly shifts each subband floor with the local potential,
n_tilde(phi)=sum_i (m*/pi hbar^2) |psi_i^(k)|^2 ln[1+exp((E_F - E_i^(k) + q(phi-phi^(k)))/kB T)],
solved by Newton (exact Jacobian = Fermi function), then re-solve Schrodinger.
~10-20 outer iters; get the sign of the q(phi-phi^(k)) shift right.

### Density-gradient (DG) -- the device-coupled route (DEVSIM supports it)
DEVSIM density_gradient example (`dg_common.py`, github.com/devsim/devsim_density_gradient):
add a quantum potential Lambda as an extra nodal variable + elliptic equation;
the density sees a shifted band edge `E_n = E_c + Lambda`, with
`b_n = Gamma_n * hbar^2 / (6 q m*)` the DG coefficient (Gamma_n ~ 1-3 calibration
knob). Couples natively into Poisson+continuity Newton (2D/3D, cheap). Calibrate
Gamma_n so the DG accumulation profile matches the 1D SP reference for ITO.
Recommendation: SP = equilibrium C-V reference; DG = device-level transport.

## Oblique incidence -- envelope (Bloch-transform) formulation [the robust fix]

Source-verified vs NGSolve 6.2.2604 C++ (periodic.cpp, occgeom.cpp, meshaccess.cpp)
+ numerically on-machine. Replaces the fragile phase-in-space route (which no
phase sign made conserve energy).

KEY PITFALL (NGSolve complex CF): `a*b` does NOT conjugate; `InnerProduct(a,b)`
CONJUGATES the 2nd arg. Mixing them in one BilinearForm is invisible at normal
incidence (k_par=0) but breaks energy conservation at oblique. Use `*` UNIFORMLY.

ng.Periodic phase convention (if ever using phase-in-space): for
`master.Identify(minion)`, the space enforces `u(minion)=phase[idnr]*u(master)`,
phase indexed PER identification in creation order. Bloch rule:
`phase[idnr]=exp(i k_par . (r_minion - r_master))`. Tutorial idiom
`Max(X).Identify(Min(X))` -> r_min-r_master = -Px x -> phase=exp(-i kx Px).

Envelope route (PREFERRED -- what we implement): write E = u*exp(i k_par.r) with
u PLAIN-periodic (phase=None space, no idnr fragility). curl(E) =
exp(i k_par.r)*(curl u + i k_par x u). Test carries the CONJUGATE phase
exp(-i k_par.r) (verified from the dispersion tutorial scalar expansion: test
operator is ∇ - i k x, i.e. a MINUS), so the modified-curl operators are:
  trial:  C(u)  = curl(u) + i*k_par x u      = curl(u) + kcross(u)
  test:   Cbar(v) = curl(v) - i*k_par x v    = curl(v) - kcross(v)   # MINUS on test
with kcross(w) = 1j*CF((0, -kx*w[2], kx*w[1])) for k_par=(kx,0,0). Weak form, `*`
uniformly (NO InnerProduct), eps_bg=1 (vacuum incidence medium):
  a += ((curl(u)+kcross(u)) * (curl(v)-kcross(v)) - k0**2*eps*(u*v)) * dx
  f += k0**2*(eps - 1)*(u_inc_env * v) * dx ;  u_inc_env = exp(-i kz_s z)*pol_vec
At k_par=0, kcross=0 and u_inc_env=E_inc -> reduces EXACTLY to the proven
normal-incidence form (the regression anchor). gfu now holds the ENVELOPE u, so
R/T fits average the envelope DIRECTLY (NO exp(-i kx x) demod). PML: ordinary
normal HalfSpace z-stretch, alpha=1j CONSTANT (do NOT scale by 1/cos theta -- the
on-machine check confirmed alpha-rescaling only changes absorption length, not
conservation; matches our earlier "no effect" observation). Make PML >= ~lambda
thick and interface >= lambda/2 above the structure. symmetric=False at oblique
(cross terms break complex symmetry). Validate vs tmm at 0/15/30 deg; if energy
is off, the prime suspect order is: test-sign (flip kcross sign on v) -> PML
thickness -> source envelope. p-pol oblique is a further follow-up.

## Sources
Tan/Snider JAP 68,4071 (1990); Subramanian IWCE (1994); Trellakis et al. JAP
81,7880 (1997); Gao et al. QCAD arXiv:1403.7561 (2014); Gregory et al. APL 105,
181117 (2014, ITO nonparabolicity); DEVSIM simple_dd.py/simple_physics.py +
devsim_density_gradient.
