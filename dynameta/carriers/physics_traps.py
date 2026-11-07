"""
Trap physics for the bipolar drift-diffusion regions (physics_bipolar_dd): bulk SRH levels at an
arbitrary trap energy E_t, several discrete levels each with its own capture parameters and density,
the trapped charge those levels carry into Poisson, and interface traps (D_it) at a
semiconductor/insulator interface -- their sheet charge Q_it(psi_s) and their surface SRH
recombination. Quasi-static (steady-state) occupancy by default; KINETIC occupancy (finite capture and
emission, an occupancy variable per level / bin, v0.15) with TrapLevel(dynamic=True) /
InterfaceTraps(dynamic=True) -- section 5 below. OPT-IN: nothing here runs unless a caller passes traps;
the defaults reproduce the legacy midgap SRH model character-for-character.

FORMULATION (all SI; energies are measured from the intrinsic level E_i, + toward E_c, in eV)

1. Bulk SRH at a trap level E_t  [Shockley & Read, Phys. Rev. 87, 835 (1952); Hall, Phys. Rev. 87,
   387 (1952)]. Electron capture/emission and hole capture/emission at one level of density N_t give,
   in steady state, the single net rate

       U = (n p - n_ie^2) / ( tau_p0 (n + n1) + tau_n0 (p + p1) ),
       n1 = n_ie exp((E_t - E_i)/kT),   p1 = n_ie exp((E_i - E_t)/kT),   n1 p1 = n_ie^2,
       tau_n0 = 1/(sigma_n v_th,n N_t),  tau_p0 = 1/(sigma_p v_th,p N_t).

   E_t = E_i gives n1 = p1 = n_ie -- the legacy model. `srh_rate_expr()` with its default names IS the
   legacy DEVSIM string (pinned character-for-character by tests/test_traps.py), and the legacy channel's
   n1/p1 parameters are now n_i exp(+-E/kT), which for the default E = 0 is n_i * 1.0 == n_i exactly in
   IEEE arithmetic -- so the reduction is BIT-identical, not just formula-identical.
   (A level degeneracy factor g only shifts E_t by kT ln g; fold it into energy_eV.)

   Low-injection lifetime this implies, n-type (n0 >> p0, Delta n = Delta p << n0):
       tau = tau_p0 (1 + n1/n0) + tau_n0 (p1/n0)            [Schroder, Semiconductor Material and
   and in general tau = [tau_p0 (n0 + n1) + tau_n0 (p0 + p1)]/(n0 + p0).  Device Characterization, 3rd
   ed., Ch. 7]. tests/test_traps.py pins the solved slab lifetime to this across E_t.

2. Several discrete levels k (each its own E_t,k, N_t,k and tau/sigma pairs): the levels are
   independent recombination channels, U = sum_k U_k. (Strictly exact when the levels are not
   coupled by inter-level transitions -- the standard multi-level SRH assumption.)

3. Occupancy and trapped charge. The steady-state fraction of level k occupied by an electron is
       f_t = (c_n n + c_p p1) / (c_n (n + n1) + c_p (p + p1)),   c = sigma v_th,
           = (tau_p0 n + tau_n0 p1) / (tau_p0 (n + n1) + tau_n0 (p + p1))   (same denominator as U),
   which reduces to the Fermi function 1/(1 + exp((E_t - E_F)/kT)) in equilibrium (n p = n_ie^2).
   Trapped charge density entering Poisson:
       acceptor-like (neutral when empty) : -q N_t f_t
       donor-like    (neutral when full)  : +q N_t (1 - f_t)
   kind="neutral" keeps a level recombination-only (the legacy SRH assumption N_t -> 0).

4. Interface traps at a semiconductor/insulator interface  [Nicollian & Brews, MOS (Metal Oxide
   Semiconductor) Physics and Technology, Wiley 1982]. A density D_it(E) [eV^-1 m^-2]
   -- uniform over a band of the gap, Gaussian, or discrete levels -- is discretized into sheet levels
   N_it,k = D_it(E_k) dE_k [m^-2] (midpoint bins of width <= kT at the REGION's T, edges aligned to the
   neutral level;
   the midpoint rule on the thermally-smoothed integrand converges like exp(-2 pi^2 kT/dE), a ~1e-8
   ESTIMATED quadrature error at dE = kT away from the band ends -- the measured C-V gates, ~2e-4, are
   mesh-limited, not bin-limited). Each sheet level has the SRH occupancy of item 3
   evaluated with the SURFACE densities n_s, p_s, and
       Q_it = q sum_{donor-like} N_it,k (1 - f_k) - q sum_{acceptor-like} N_it,k f_k  [+ Q_f fixed],
   with the Nicollian-Brews convention "acceptor-like above the neutral level E_0, donor-like below"
   (charge="split", E_0 = E_i by default) or all-acceptor / all-donor. In equilibrium this gives
   C_it = -dQ_it/dpsi_s = q^2 D_it (for a uniform D_it, E_F away from the band ends), and the LF C-V
   stretch-out d psi_s/dV_G = C_ox/(C_ox + C_s + C_it).
   Surface SRH through the same levels: U_s = sum_k (n_s p_s - n_ie^2) / ((n_s + n1,k)/S_p,k +
   (p_s + p1,k)/S_n,k)  [m^-2 s^-1], S_n,k = sigma_n v_th,n N_it,k, S_p,k = sigma_p v_th,p N_it,k
   [Fitzgerald & Grove, Surf. Sci. 9, 347 (1968); Aberle, Glunz & Warta, J. Appl. Phys. 71, 4422
   (1992)]. Low-injection n-type: S_eff = (n0 + p0)/((n0 + n1)/S_p + (p0 + p1)/S_n) -> S_p.

5. Kinetic traps (v0.15, plan 4.3.2; docs/audit/2026-11-06-v0.15-kinetic-traps.md). The occupancy is a
   solution variable -- its odds w = f/(1-f) -- with N_t df/dt = R_n - R_p (srh_capture_exprs: R_n = [n (1-f)
   - n1' f]/tau_n0, R_p = [p f - p1' (1-f)]/tau_p0; emission by detailed balance with the SAME n1', p1' as the
   quasi-static kernel). R_n enters the electron row only, R_p the hole row only (physics_bipolar_dd's
   single-carrier terms), the trap row balances both, and the stored charge -q N_t f IS the Poisson charge:
   DC equals the quasi-static model (measured 1.4e-15), transient and ssac conserve charge (Kirchhoff to the
   trap-free floor), and the conductance method G_p(omega)/omega of D_it follows Nicollian & Goetzberger
   (Bell Syst. Tech. J. 46, 1055 (1967), eqs. 16-20) to the computed series-resistance budget. WHICH TO
   USE: quasi-static for DC, I-V, lifetimes, compensation and the LF C-V; kinetic for any transient, ssac or
   (from v0.15 K3) noise analysis of CHARGED levels -- a quasi-static charged trap in an 'ac' / 'noise' /
   'transient_dc' solve raises an AdvisoryWarning (quasi_static_charged_traps). Kinetic levels need a seed
   (seed_kinetic_traps, after the carriers are seeded; the builders do it); under statistics=
   "fermi_dirac" they take the exact FD emission densities (srh_capture_exprs fd=True; v0.15 integration);
   kinetic interface bins cost (bins) x (region nodes) unknowns, budgeted (KINETIC_UNKNOWN_BUDGET).

DEVSIM REALIZATION. The sheet terms are applied on the SEMICONDUCTOR side of the interface as node
models weighted by ItArea = A_node/NodeVolume, where A_node is the interface area each node owns
(computed from DEVSIM's own interface element list: 1 in 1-D, half of each interface edge in 2-D, a third
of each interface triangle in 3-D). DEVSIM integrates a region node model over NodeVolume, so the
residual contribution is exactly (sheet quantity) * A_node; and the interface's `continuous`
PotentialEquation SUMS the two regions' rows at each interface node, so a sheet charge added to the
semiconductor row is the Gauss-law surface charge of the interface (MEASURED on a two-dielectric
1-D stack: interface potential = sigma/(eps1/d1 + eps2/d2) to the last printed digit). DEVSIM's own
`fluxterm`/`hybrid` interface types are flux-transfer terms (added to region 0, subtracted from region
1): measured on the same stack they move charge ACROSS the interface (-2x and opposite signs), so they
are the right tool for D1a's semiconductor/semiconductor thermionic current, NOT for a trapped sheet.
The per-interface weighting is used instead of DEVSIM's built-in `SurfaceArea` node model because
SurfaceArea is the SUM over ALL of a region's interfaces (MEASURED in 2-D node-by-node equal to the sum
of the per-interface areas, tests/test_traps.py::test_interface_node_area_2d): on a semiconductor touching a
gate oxide on top and a spacer on the side it would put the gate interface's D_it on the spacer's
nodes as well. (DEVSIM assigns a node shared by two interfaces to the first one only; the area here
follows DEVSIM's own element list, so it inherits that convention.)

LIMITS (documented in docs/audit/2026-11-03-devsim-traps.md):
  * Quasi-static occupancy (the default): the trapped charge is an instantaneous function of (n, p). A DC
    sweep (the LF C-V) is exact; a transient or ssac solve is NOT charge-conserving with CHARGED traps (the
    capture current that fills a trap is not in the continuity time term) -- use dynamic=True there (item 5).
  * Kinetic traps' noise: the capture / emission events (trap_n / trap_p); the McWhorter (border-trap) 1/f
    spectrum is out of scope (decision 42).
  * Ohmic contacts (v0.14 integration): with CHARGED bulk levels the contacts, the seed and the 3-D gate
    frame pin the TRAP-AWARE charge-neutral root (bipolar_options.equilibrium_fields with the levels'
    equilibrium Fermi occupancy, n1 = n_ie exp(E_t/kT)) -- the same rule as BGN / incomplete
    ionization. Interface traps do not enter it (a sheet charge, not a bulk neutrality term).
  * Bandgap narrowing (D1b): every rate, occupancy and pair rate here takes the region's n_ie
    substitution through srh_reference_densities, so a level's E_t is measured from the NARROWED
    effective intrinsic level; with an asymmetric band-edge split (conduction_fraction != 1/2) that is a
    modelling choice (see docs/audit/2026-11-03-devsim-phase-integration.md).
  * statistics="fermi_dirac" (v0.15 completion): every rate, occupancy and pair rate here takes the EXACT
    FD emission densities gamma_n n1, gamma_p p1 (srh_reference_densities fd=True; derivation there), so a
    level's equilibrium occupancy is the Fermi function of the FD quasi-Fermi level for any capture ratio,
    charged levels pin the FD charge-neutral root (bipolar_options._neutral_root_fd), and the interface terms'
    closed-form derivatives carry the d ln gamma/dc terms (install_interface_traps). The legacy statistics
    are unchanged (fd=False: the strings character-for-character).
  * Interface traps need a bipolar (n AND p) semiconductor side -- the SRH occupancy uses both.
    A semiconductor/semiconductor interface raises (heterojunctions are D1a's).
  * 3-D: on the devsim_3d tensor (Kuhn) mesh (D2c, v0.14.0 default) the 3-D sheet-charge shift equals
    the 1-D one (validation/traps_oracles.py --3d, gates G2/G3). On the legacy mesher='gmsh' mesh DEVSIM's
    node volumes over-count (sum NodeVolume = 1.20 x the semiconductor volume, 1.29 x the oxide's) and
    a geometric sheet charge acts at ~0.8 of its 1-D effect (docs/audit/2026-11-03-devsim-3d-mesh.md).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dynameta.constants import Q_E, V_T
from dynameta.carriers.eq_registry import ds, node_with_derivs, record_region_equation

_KINDS = ("acceptor", "donor", "neutral")
# Where a level's energy is quoted from (the per-level ENERGY REFERENCE, v0.14 follow-up). Equivalent to
# Sentaurus Device's trap EnergyMid reference keywords fromMidBandGap / fromCondBand / fromValBand
# (Synopsys, "Sentaurus Device User Guide", chapter "Traps and Fixed Charges"):
#   "intrinsic"  : energy_eV = E_t - E_ie, from the (narrowed) effective intrinsic level -- the D1c default
#   "conduction" : energy_eV = E_c - E_t >= 0, below the NARROWED conduction edge (a DLTS "E_c - 0.25 eV")
#   "valence"    : energy_eV = E_t - E_v >= 0, above the NARROWED valence edge (a DLTS "E_v + 0.25 eV")
_REFERENCES = ("intrinsic", "conduction", "valence")
_CHARGE_MODES = ("split", "acceptor", "donor", "none")


# ------------------------------------------------------------------------------------------------
# Shared expression kernels (pure strings -- no DEVSIM needed; ONE implementation for the legacy
# midgap channel, every bulk level and every interface sheet level, with or without bandgap
# narrowing, for the net rate, the occupancy and the noise pair rate G + R).
# ------------------------------------------------------------------------------------------------
def srh_reference_densities(n1: str = "n1", p1: str = "p1", *, bgn: bool = False,
                            reference: str = "intrinsic", scale: "str | None" = None,
                            fd: bool = False) -> Tuple[str, str, str]:
    """(n1', p1', n_i'^2) -- THE single place the bandgap-narrowing substitution of D1b (audit R27)
    enters SRH: with BGN on, n_i^2 -> n_ie^2 and n1 -> n1 n_ie/n_i, p1 -> p1 n_ie/n_i (so
    n1' p1' = n_ie^2 and a level keeps its energy relative to the NARROWED effective intrinsic level:
    n1' = n_ie exp((E_t - E_i)/kT)). `n_ie` is the region's effective-intrinsic-density field
    (bipolar_options). Without BGN the names are returned unchanged.

    `reference` (the per-level energy reference; level_reference_densities sets the matching NUMBERS):
      "intrinsic"  -- as above: both n1 and p1 scale by n_ie/n_i (the level rides the narrowed E_ie);
      "conduction" -- n1 = N_c exp(-(E_c - E_t)/kT) is a distance from the NARROWED conduction edge, which
                      BGN does not change (the narrowing moves the edge, not the DOS), so n1 is kept and
                      p1' = n_ie^2/n1;
      "valence"    -- the mirror: p1 kept, n1' = n_ie^2/p1.
    Every reference keeps n1' p1' = n_ie^2 (n_i^2 without BGN), the condition for the occupancy to be a
    Fermi function at equilibrium. Without BGN all three return the names unchanged.

    `scale` (D2d, bipolar density-gradient, 2026-11-03): the name of a node model s multiplying the
    intrinsic level on top of the BGN substitution -- n1 -> n1 s, p1 -> p1 s, n_i'^2 -> n_i'^2 s^2.
    The density-gradient quantum potentials shift the band edges like BGN does, so detailed balance
    needs n1' p1' = n_i'^2 = n p|_eq = n_i'^2 exp((Lambda_n + Lambda_p)/V_t), i.e.
    s = exp((Lambda_n + Lambda_p)/(2 V_t)) (physics_density_gradient.setup_bipolar_dg). None (the
    default) returns the strings above CHARACTER-FOR-CHARACTER.

    COMPOSITION ORDER (v0.14): the reference conversion first, then the DG scale -- so every reference
    keeps n1' p1' = n_i'^2 s^2 = n p|_eq. (The DG scale is the symmetric (Lambda_n + Lambda_p)/2 split
    for every reference; a band-referenced level that should follow ONE quantum band edge only --
    n1 x exp(Lambda_n/V_t) for "conduction" -- is not modelled; see the D1c doc section 9.)

    `fd` (v0.15, statistics="fermi_dirac"; requires bgn=True, the region's n_ie node path): the EXACT
    Fermi-Dirac emission densities n1' = gamma_n n1_B, p1' = gamma_p p1_B, with n1_B, p1_B the Boltzmann-frame
    values of the reference above (n_ie0 = n_i exp(dEg/2kT) in place of n_ie) and gamma = exp(-Delta(c/N))
    the solution-dependent degeneracy factors of bipolar_options.install_fd_detailed_balance (node models
    FDNiN = n_ie0 gamma_n, FDNiP = n_ie0 gamma_p, FDGammaN, FDGammaP; n_ie^2 = FDNiN FDNiP). DERIVATION
    (energy-independent capture coefficient c_n, each band in quasi-equilibrium at its own FD quasi-Fermi
    level E_Fn): detailed balance per band energy gives the emission e(E) = c exp((E_t - E)/kT), and the
    net capture integrated over the band is c_n [n (1 - f_t) - f_t n exp((E_t - E_Fn)/kT)] -- the Pauli
    factor (1 - f(E)) of the emission is what makes it n exp((E_t - E_Fn)/kT) = gamma_n n1_B, LOCAL in n
    (Sentaurus Device's FD SRH, "gamma_n n1"). Hence
        U = (n p - n_ie^2)/(taup (n + gamma_n n1_B) + taun (p + gamma_p p1_B)),
        f_t = (taup n + taun gamma_p p1_B)/(same)  ->  1/(1 + exp((E_t - E_F)/kT)) at equilibrium EXACTLY,
    for ANY taun/taup. The shared sqrt(gamma_n gamma_p) n1 of the bare n_ie substitution (the pre-completion
    FD mode) has the same numerator but its occupancy is a Fermi function only where gamma_n = gamma_p: a
    CHARGED level on a degenerate layer would then carry a capture-ratio-dependent equilibrium charge
    (docs/audit/2026-11-06-v0.15-fermi-dirac-mode-complete.md). fd=False (default) returns the strings
    above CHARACTER-FOR-CHARACTER."""
    if reference not in _REFERENCES:
        raise ValueError("energy reference must be one of {}; got {!r}".format(_REFERENCES, reference))
    if fd and not bgn:
        raise ValueError("srh_reference_densities: fd=True (statistics='fermi_dirac') needs bgn=True: the FD "
                         "detailed balance lives on the region's n_ie node models")
    if fd:
        if reference == "conduction":
            out = ("{}*FDGammaN".format(n1), "(n_ie^2/({}*FDGammaN))".format(n1), "n_ie^2")
        elif reference == "valence":
            out = ("(n_ie^2/({}*FDGammaP))".format(p1), "{}*FDGammaP".format(p1), "n_ie^2")
        else:
            out = ("{}*FDNiN/n_i".format(n1), "{}*FDNiP/n_i".format(p1), "n_ie^2")
    elif not bgn:
        out = (n1, p1, "n_i^2")
    elif reference == "conduction":
        out = (n1, "(n_ie^2/{})".format(n1), "n_ie^2")
    elif reference == "valence":
        out = ("(n_ie^2/{})".format(p1), p1, "n_ie^2")
    else:
        out = ("{}*n_ie/n_i".format(n1), "{}*n_ie/n_i".format(p1), "n_ie^2")
    if scale is None:
        return out
    return ("{}*{}".format(out[0], scale), "{}*{}".format(out[1], scale),
            "{}*{}^2".format(out[2], scale))


def _srh_denominator(n1: str, p1: str, taun: str, taup: str) -> str:
    return "({tp}*(Electrons + {n1}) + {tn}*(Holes + {p1}))".format(n1=n1, p1=p1, tn=taun, tp=taup)


def srh_rate_expr(n1: str = "n1", p1: str = "p1", taun: str = "taun", taup: str = "taup", *,
                  bgn: bool = False, pair: bool = False, reference: str = "intrinsic",
                  scale: "str | None" = None, fd: bool = False) -> str:
    """The Shockley-Read-Hall net rate U = (n p - n_i^2)/(taup (n + n1) + taun (p + p1)) as a DEVSIM
    expression in the solution variables Electrons/Holes. The arguments are parameter names (or
    parenthesized sub-expressions). The DEFAULTS reproduce the legacy physics_bipolar_dd USRH string
    CHARACTER-FOR-CHARACTER (tests/test_traps.py pins it). For a sheet level pass taun -> 1/S_n,
    taup -> 1/S_p and the result is the surface rate [m^-2 s^-1].

    bgn=True applies the bandgap-narrowing substitution (srh_reference_densities); with the default
    names it is bipolar_options' former srh_expr(True) character-for-character.
    pair=True returns the PAIR rate G + R = (n p + n_i^2)/(...) instead of the net rate R - G: the
    fast-trap detailed-balance split R = n p/D, G = n_i^2/D that the noise module's GR source
    2 (G + R) needs (noise_analysis; one kernel for both).
    reference / scale: the per-level energy reference and the density-gradient intrinsic-level factor
    of srh_reference_densities (defaults: unchanged); fd: its exact Fermi-Dirac emission densities
    (statistics="fermi_dirac": gamma_n n1, gamma_p p1)."""
    n1e, p1e, ni2 = srh_reference_densities(n1, p1, bgn=bgn, reference=reference, scale=scale, fd=fd)
    return "(Electrons*Holes {} {})/{}".format("+" if pair else "-", ni2,
                                               _srh_denominator(n1e, p1e, taun, taup))


def srh_occupancy_expr(n1: str = "n1", p1: str = "p1", taun: str = "taun",
                       taup: str = "taup", *, bgn: bool = False, reference: str = "intrinsic",
                       fd: bool = False) -> str:
    """Steady-state electron occupancy f_t = (taup n + taun p1)/(taup (n + n1) + taun (p + p1)) of an
    SRH level (Shockley & Read 1952) -- the SAME denominator as srh_rate_expr. Equals the Fermi
    function of E_t at equilibrium PROVIDED n1 p1 equals the equilibrium n p -- which is why it takes
    the same `bgn` substitution as the rate (with BGN on n p = n_ie^2 at equilibrium; an occupancy
    left on n_i would depend on taun/taup at equilibrium, i.e. not be a Fermi function). Under
    statistics="fermi_dirac" pass fd=True: only the exact gamma_n n1, gamma_p p1 form keeps it a Fermi
    function of the FD quasi-Fermi level on a degenerate layer (srh_reference_densities)."""
    n1e, p1e, _ = srh_reference_densities(n1, p1, bgn=bgn, reference=reference, fd=fd)
    return "({tp}*Electrons + {tn}*{p1})/{d}".format(tp=taup, tn=taun, p1=p1e,
                                                     d=_srh_denominator(n1e, p1e, taun, taup))


def srh_capture_exprs(n1: str = "n1", p1: str = "p1", taun: str = "taun", taup: str = "taup",
                      f: str = "f", one_minus_f: str = "g", *, bgn: bool = False,
                      reference: str = "intrinsic", scale: "str | None" = None,
                      pair: bool = False, fd: bool = False) -> Tuple[str, str]:
    """(R_n, R_p): the electron and hole CAPTURE-minus-EMISSION rates [m^-3 s^-1] of a level whose
    electron occupancy is a VARIABLE (`f`, and `one_minus_f` = 1 - f spelled separately so a nearly
    full level never cancels) -- the kinetic (finite capture and emission) twin of srh_rate_expr
    [Shockley & Read, Phys. Rev. 87, 835 (1952); Hall, Phys. Rev. 87, 387 (1952)]:

        R_n = [ n (1 - f) - n1' f ] / tau_n0        (net electron capture: c_n n (1-f) - e_n f)
        R_p = [ p f - p1' (1 - f) ] / tau_p0        (net hole capture:     c_p p f - e_p (1-f))
        N_t df/dt = R_n - R_p,

    with emission by detailed balance, e_n = c_n n1', e_p = c_p p1', and n1', p1' EXACTLY the densities
    of srh_reference_densities (energy reference, then BGN, then the DG scale) -- so the steady state
    R_n = R_p gives f = srh_occupancy_expr and R_n = R_p = srh_rate_expr: the DC limit IS the
    quasi-static model (tests/test_trap_kinetics.py, gate K1-1, every reference x BGN x DG scale). With
    reference="conduction" e_n = sigma_n v_th N_c exp(-(E_c - E_t)/kT), the DLTS emission rate [Lang,
    J. Appl. Phys. 45, 3023 (1974)]. For a sheet level pass taun -> 1, taup -> c_n/c_p and multiply by
    N_it c_n (as install_interface_traps does for the quasi-static kernel).

    pair=True returns the EVENT rates (P_n, P_p) instead: both directions of each exchange counted,
    P_n = [n (1 - f) + n1' f]/tau_n0, P_p = [p f + p1' (1 - f)]/tau_p0 -- the Poisson rate of the
    electron- and hole-exchange processes, the noise source of the kinetic trap (each a two-point event
    between its carrier row and the trap row, noise_analysis.TwoPointSource).

    fd=True (statistics="fermi_dirac", v0.15 FD-mode integration): n1' = gamma_n n1_B, p1' = gamma_p p1_B, the
    exact FD emission densities of srh_reference_densities -- e_n = c_n gamma_n n1_B is the band-integrated
    emission into the Pauli-blocked conduction band at the LOCAL electron distribution. Then R_n = 0 <=>
    f/(1-f) = n/n1' = exp((E_F - E_t)/kT) and R_p = 0 <=> f/(1-f) = p1'/p = the same, exactly, at the FD
    equilibrium; and the linear response is c_n n (1 - f) (d eta_n - d u) (the gamma_n derivative supplies
    the g-factor), so the event rates (pair=True) stay Nyquist-consistent."""
    n1e, p1e, _ = srh_reference_densities(n1, p1, bgn=bgn, reference=reference, scale=scale, fd=fd)
    s = "+" if pair else "-"
    return ("(Electrons*{g} {s} {n1}*{f})/{tn}".format(g=one_minus_f, s=s, n1=n1e, f=f, tn=taun),
            "(Holes*{f} {s} {p1}*{g})/{tp}".format(f=f, s=s, p1=p1e, g=one_minus_f, tp=taup))


def srh_capture_derivs(n1: str, p1: str, taun: str, taup: str, f: str, one_minus_f: str, var: str,
                       dfdx: str, *, bgn: bool = False, reference: str = "intrinsic",
                       scale: "str | None" = None, fd: bool = False) -> Tuple[dict, dict]:
    """Closed-form derivatives of srh_capture_exprs for physics_bipolar_dd.add_recombination_term(
    derivs=...): ({Electrons, Holes, var} of R_n, the same of R_p), where `var` is the occupancy variable
    x and `dfdx` the expression of df/dx (the kinetic levels use the odds x = w = f/(1-f): df/dw =
    (1 - f)^2). n1', p1' are fixed fields here (the region's n_ie does not depend on the carriers --
    install_bulk_traps refuses the FD-statistics regions where it does). R_n depends on (n, x), R_p on
    (p, x):
        dR_n/dn = (1 - f)/tau_n0          dR_n/dx = -(n + n1') (df/dx)/tau_n0
        dR_p/dp = f/tau_p0                dR_p/dx = +(p + p1') (df/dx)/tau_p0.
    fd=True: n1' ~ gamma_n(n), p1' ~ gamma_p(p) (srh_reference_densities), so with ln = FDdLnGammaN,
    lp = FDdLnGammaP:  dR_n/dn = ((1 - f) - f n1' ln)/tau_n0,  dR_p/dp = (f - (1 - f) p1' lp)/tau_p0."""
    n1e, p1e, _ = srh_reference_densities(n1, p1, bgn=bgn, reference=reference, scale=scale, fd=fd)
    if fd:
        return ({"Electrons": "({g} - {f}*{n1}*FDdLnGammaN)/{tn}".format(g=one_minus_f, f=f, n1=n1e, tn=taun),
                 "Holes": "0", var: "-(Electrons + {})*{}/{}".format(n1e, dfdx, taun)},
                {"Electrons": "0",
                 "Holes": "({f} - {g}*{p1}*FDdLnGammaP)/{tp}".format(f=f, g=one_minus_f, p1=p1e, tp=taup),
                 var: "(Holes + {})*{}/{}".format(p1e, dfdx, taup)})
    return ({"Electrons": "{}/{}".format(one_minus_f, taun), "Holes": "0",
             var: "-(Electrons + {})*{}/{}".format(n1e, dfdx, taun)},
            {"Electrons": "0", "Holes": "{}/{}".format(f, taup),
             var: "(Holes + {})*{}/{}".format(p1e, dfdx, taup)})


def srh_vacancy_expr(n1: str = "n1", p1: str = "p1", taun: str = "taun", taup: str = "taup", *,
                     bgn: bool = False, reference: str = "intrinsic", fd: bool = False) -> str:
    """1 - f_SS = (tau_p0 n1' + tau_n0 p)/(tau_p0 (n + n1') + tau_n0 (p + p1')) -- the steady-state vacancy of
    srh_occupancy_expr written without the 1 - f subtraction (a nearly full level keeps its digits). fd: as
    srh_reference_densities."""
    n1e, p1e, _ = srh_reference_densities(n1, p1, bgn=bgn, reference=reference, fd=fd)
    return "({tp}*{n1} + {tn}*Holes)/{d}".format(tp=taup, tn=taun, n1=n1e, d=_srh_denominator(n1e, p1e, taun, taup))


def srh_occupancy_odds_expr(n1: str = "n1", p1: str = "p1", taun: str = "taun", taup: str = "taup",
                            *, bgn: bool = False, reference: str = "intrinsic", fd: bool = False) -> str:
    """w_SS = f_SS/(1 - f_SS) of the steady-state occupancy srh_occupancy_expr, written as
    (tau_p0 n + tau_n0 p1')/(tau_p0 n1' + tau_n0 p) (f_SS and 1 - f_SS share the SRH denominator) -- no
    1 - f cancellation, finite and > 0 at n = p = 0. The exact seed of a kinetic level's occupancy
    variable (seed_kinetic_traps); at equilibrium it is exp((E_F - E_t)/kT) -- with fd=True (srh_reference_
    densities) exactly so also at the FD equilibrium of a degenerate layer."""
    n1e, p1e, _ = srh_reference_densities(n1, p1, bgn=bgn, reference=reference, fd=fd)
    return "(({tp}*Electrons + {tn}*{p1})/({tp}*{n1} + {tn}*Holes))".format(
        tp=taup, tn=taun, p1=p1e, n1=n1e)


def n1_p1(energy_eV: float, n_i_m3: float, V_t: float = V_T) -> Tuple[float, float]:
    """(n1, p1) = n_i exp(+-E/kT) for a level at E = E_t - E_i [eV], kT/q = V_t (the region's thermal
    voltage; V_T at T_REF by default, BipolarOptions.T_K otherwise). E = 0 returns (n_i, n_i)
    EXACTLY (exp(0.0) == 1.0), which is what keeps the legacy midgap channel bit-identical."""
    e = float(energy_eV) / float(V_t)
    return float(n_i_m3) * math.exp(e), float(n_i_m3) * math.exp(-e)


def level_reference_densities(energy_eV: float, reference: str, *, n_i_m3: float,
                              V_t: float = V_T, N_c_m3: Optional[float] = None,
                              N_v_m3: Optional[float] = None) -> Tuple[float, float]:
    """THE conversion of a quoted level energy to its (n1, p1) [m^-3], before the BGN substitution of
    srh_reference_densities (which takes the SAME `reference`):
      "intrinsic"  : E = E_t - E_i     ->  n1_p1(E): n1 = n_i exp(E/kT), p1 = n_i exp(-E/kT)
      "conduction" : E = E_c - E_t     ->  n1 = N_c exp(-E/kT), p1 = n_i^2/n1
      "valence"    : E = E_t - E_v     ->  p1 = N_v exp(-E/kT), n1 = n_i^2/p1
    so n1 p1 = n_i^2 on every reference (the BGN step then makes it n_ie^2). N_c / N_v are the
    region's effective densities of states (setup_bipolar_region's n_dos_m3 / n_dos_p_m3). The two band
    references are the DLTS convention: a level measured as an activation energy from a band edge keeps
    that distance to the (narrowed) edge, n1 = N_c exp(-E_a/kT) [Lang, J. Appl. Phys. 45, 3023 (1974);
    Schroder, Semiconductor Material and Device Characterization, 3rd ed., ch. 5]. The intrinsic path
    is n1_p1 itself, so the default is the shipped arithmetic bit for bit."""
    if reference == "intrinsic":
        return n1_p1(energy_eV, n_i_m3, V_t)
    if reference not in _REFERENCES:
        raise ValueError("energy reference must be one of {}; got {!r}".format(_REFERENCES, reference))
    dos = N_c_m3 if reference == "conduction" else N_v_m3
    if dos is None or not (float(dos) > 0.0):
        raise ValueError("a {!r}-referenced level needs the band's effective DOS (N_c / N_v) > 0"
                         .format(reference))
    edge = float(dos) * math.exp(-float(energy_eV) / float(V_t))
    other = float(n_i_m3) * float(n_i_m3) / edge
    return (edge, other) if reference == "conduction" else (other, edge)


def equilibrium_n1(level, n_ie, V_t: float, N_c_m3=None, N_v_m3=None):
    """n1 of a CHARGED level at the charge-neutral equilibrium (n p = n_ie^2) -- the occupancy
    f = n/(n + n1) of bipolar_options' neutrality root. `level` is an equilibrium_trap_levels entry:
    (E, N, kind) for an intrinsic-referenced level (n1 = n_ie exp(E/kT), the shipped arithmetic
    unchanged) or (E, N, kind, reference) for a band-referenced one (n1 = N_c exp(-E/kT), or
    n_ie^2/(N_v exp(-E/kT))). The SAME conversion as level_reference_densities + the BGN step."""
    E = float(level[0])
    ref = level[3] if len(level) > 3 else "intrinsic"
    if ref == "intrinsic":
        return n_ie * math.exp(E / float(V_t))
    n1, p1 = level_reference_densities(E, ref, n_i_m3=1.0, V_t=V_t, N_c_m3=N_c_m3, N_v_m3=N_v_m3)
    if ref == "conduction":
        return n1 * np.ones_like(np.asarray(n_ie, dtype=np.float64))
    return n_ie * n_ie / p1


def _check_reference(owner: str, reference: str, energy_eV: float) -> None:
    if reference not in _REFERENCES:
        raise ValueError("{}.energy_reference must be one of {}; got {!r}".format(owner, _REFERENCES,
                                                                              reference))
    if reference != "intrinsic" and not (float(energy_eV) >= 0.0):
        raise ValueError("{}: a {!r}-referenced energy is a distance INTO the gap from the band edge "
                         "and must be >= 0; got {!r}".format(owner, reference, energy_eV))


# ------------------------------------------------------------------------------------------------
# Specs
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrapLevel:
    """One discrete bulk trap level (Shockley-Read-Hall).

    energy_eV    : the level energy [eV] in `energy_reference`: E_t - E_i (+ toward E_c) for the default
                   "intrinsic"; E_c - E_t >= 0 for "conduction"; E_t - E_v >= 0 for "valence".
    energy_reference : "intrinsic" (default -- the level rides the narrowed effective intrinsic level
                   under BGN, the shipped behaviour), "conduction" or "valence" (the level keeps its
                   distance to the NARROWED band edge: DLTS-quoted levels, E_c - E_t / E_t - E_v, belong
                   here). See level_reference_densities.
    density_m3   : N_t [m^-3] -- needed for a CHARGED level and to turn cross sections into lifetimes.
    kind         : "acceptor" (charge -q N_t f_t), "donor" (+q N_t (1 - f_t)) or "neutral"
                   (recombination only -- the legacy N_t -> 0 SRH assumption).
    tau_n0_s, tau_p0_s : capture lifetimes [s]; OR give sigma_n_m2, sigma_p_m2 [m^2] with v_th
                   [m/s] and density_m3, and tau_x0 = 1/(sigma_x v_th,x N_t). Explicit taus win.
    dynamic      : False (default) -- QUASI-STATIC occupancy: f is the steady-state SRH value, an
                   instantaneous function of (n, p). Exact at DC and for the low-frequency C-V, and the
                   cheapest; the shipped model, character for character. Use it for DC sweeps, I-V,
                   lifetimes, compensation, and for recombination-only (kind="neutral") levels in any
                   analysis whose frequencies are far below the capture rates.
                   True -- KINETIC occupancy (v0.15, plan 4.3.2): f becomes a solution variable (its
                   odds w = f/(1-f) = exp(u), u the level's own quasi-Fermi level in kT, TrapW_trap<k>)
                   with its own rate equation N_t df/dt = R_n - R_p (srh_capture_exprs). The DC result is the
                   quasi-static one (to solver round-off), and transient / ssac become CHARGE-
                   CONSERVING (the capture current enters the continuity rows). Use it for any
                   transient or small-signal analysis of a CHARGED level (DLTS-like emission, C(omega)
                   and G(omega) dispersion, terminal-current balance), and for a neutral level whose
                   capture time is not short against the analysis time scale. Needs density_m3 > 0
                   (N_t sets the stored charge), a bipolar region with the legacy statistics, and a
                   seed (seed_kinetic_traps after the carriers are seeded). Noise: the capture /
                   emission events are the trap g-r sources (mechanisms "trap_n" / "trap_p", in
                   noise_analysis.DEFAULT_MECHANISMS): with a kinetic charged level S_I = 4kT Re Y holds at
                   every frequency (the quasi-static level breaks it above the capture rate).
    """
    energy_eV: float = 0.0
    density_m3: float = 0.0
    kind: str = "neutral"
    tau_n0_s: Optional[float] = None
    tau_p0_s: Optional[float] = None
    sigma_n_m2: Optional[float] = None
    sigma_p_m2: Optional[float] = None
    v_th_n_ms: float = 1.0e5
    v_th_p_ms: float = 1.0e5
    energy_reference: str = "intrinsic"
    dynamic: bool = False

    def __post_init__(self):
        _check_reference("TrapLevel", self.energy_reference, self.energy_eV)
        if self.kind not in _KINDS:
            raise ValueError("TrapLevel.kind must be one of {}; got {!r}".format(_KINDS, self.kind))
        if not math.isfinite(float(self.energy_eV)):
            raise ValueError("TrapLevel.energy_eV must be finite")
        if self.kind != "neutral" and not (float(self.density_m3) > 0.0):
            raise ValueError("a charged ({}) TrapLevel needs density_m3 > 0".format(self.kind))
        if not isinstance(self.dynamic, bool):
            raise TypeError("TrapLevel.dynamic must be a bool; got {!r}".format(self.dynamic))
        if self.dynamic and not (float(self.density_m3) > 0.0 and math.isfinite(float(self.density_m3))):
            raise ValueError("a kinetic TrapLevel (dynamic=True) needs a finite density_m3 > 0: N_t is the "
                             "number of states whose occupancy is the new variable (also for kind="
                             "'neutral')")
        self.lifetimes_s()                                     # validate now, not at device build

    def lifetimes_s(self) -> Tuple[float, float]:
        """(tau_n0, tau_p0) [s] from the explicit lifetimes or from sigma * v_th * N_t."""
        out = []
        for tau, sig, vth, lab in ((self.tau_n0_s, self.sigma_n_m2, self.v_th_n_ms, "n"),
                                   (self.tau_p0_s, self.sigma_p_m2, self.v_th_p_ms, "p")):
            if tau is not None:
                t = float(tau)
            elif sig is not None:
                prod = float(sig) * float(vth) * float(self.density_m3)
                if not (prod > 0.0 and math.isfinite(prod)):
                    raise ValueError("TrapLevel: sigma_{0}_m2 * v_th_{0}_ms * density_m3 must be > 0 "
                                     "(got {1!r})".format(lab, prod))
                t = 1.0 / prod
            else:
                raise ValueError("TrapLevel needs tau_{0}0_s, or sigma_{0}_m2 (+ density_m3)".format(lab))
            if not (t > 0.0 and math.isfinite(t)):
                raise ValueError("TrapLevel: tau_{}0 must be finite and > 0; got {!r}".format(lab, t))
            out.append(t)
        return out[0], out[1]


@dataclass(frozen=True)
class InterfaceTraps:
    """Interface-trap density D_it(E) at a semiconductor/insulator interface (Nicollian & Brews).

    distribution : "uniform" -- D_it = dit_eV_m2 over [e_lo_eV, e_hi_eV] (default +-band_gap_eV/2);
                   "gaussian" -- dit_eV_m2 * exp(-(E - e_peak_eV)^2/(2 sigma_eV^2)), truncated at
                   +-5 sigma and to [e_lo_eV, e_hi_eV] when given;
                   "discrete" -- `levels` = ((E_t - E_i [eV], N_it [m^-2]), ...).
    dit_eV_m2    : D_it [eV^-1 m^-2] (1e12 cm^-2 eV^-1 = 1e16 m^-2 eV^-1).
    charge       : "split" (acceptor-like above neutral_level_eV, donor-like below -- N&B), "acceptor",
                   "donor", or "none" (recombination only).
    sigma_*_m2, v_th_*_ms : capture cross sections / thermal velocities -> S = sigma v_th N_it, and
                   the occupancy ratio c_n/c_p.
    recombination: install the surface SRH term (U_s into both continuity equations).
    fixed_charge_C_m2 : a FIXED sheet charge Q_f [C/m^2] (signed), e.g. oxide fixed charge.
    bin_eV       : maximum quadrature bin width (default kT at the region's temperature): the continuous
                   D_it is integrated by the midpoint rule on bins aligned to the band ends and the neutral
                   level. install_interface_traps passes the region's kT (BipolarOptions.T_K), so the
                   default bins are <= kT at any T (v0.14 row 3.2-8: they were the 300 K kT, 3.9 kT wide at
                   77 K). Set it only to force finer bins; a value above kT coarsens the integral.
    energy_reference : the coordinate EVERY energy of this spec is quoted in (levels, e_lo/e_hi, e_peak,
                   neutral_level): "intrinsic" (default, E - E_i), "conduction" (E_c - E >= 0) or
                   "valence" (E - E_v >= 0); see TrapLevel. With a band reference, band_gap_eV spans
                   [0, E_g] and charge="split" needs an explicit neutral_level_eV.
    dynamic      : False (default) -- quasi-static occupancy (the shipped model, character for
                   character): exact for DC and the LF C-V. True -- KINETIC bins (v0.15, plan 4.3.2): every
                   sheet level of this spec gets its own occupancy variable (odds w, as TrapLevel(
                   dynamic=True)) and rate equation N_it df/dt = R_n - R_p; use it for the conductance
                   method (C(omega), G_p(omega)/omega of D_it -- Nicollian & Goetzberger 1967), interface
                   transients and any ssac / transient analysis with charged D_it. Needs recombination=True
                   (the capture IS the surface SRH). Cost: (bins) x (semiconductor-region nodes) extra
                   unknowns, because DEVSIM 2.11 has no interface solution variable (a region-wide
                   variable, masked off the interface); install_interface_traps refuses above
                   KINETIC_UNKNOWN_BUDGET -- coarsen with kinetic_bin_eV, or raise the budget.
    kinetic_bin_eV : bin width [eV] for a dynamic spec (default: as bin_eV, <= kT). A coarser bin cuts
                   the unknown count. The conductance G_p(omega) of a continuum converges like
                   exp(-pi^2 kT/dE) in it (the omega tau(E) factor has poles at Im E = pi kT/2): MEASURED
                   1e-5 .. 4e-5 at dE = kT, < 1e-9 at kT/2 (tests/test_trap_kinetics.py, K2-3) -- the LF
                   charge alone converges twice as fast (D1c). Use kT/2 for a 1e-8 G_p, kT for 1e-4.
    """
    dit_eV_m2: float = 0.0
    distribution: str = "uniform"
    band_gap_eV: Optional[float] = None
    e_lo_eV: Optional[float] = None
    e_hi_eV: Optional[float] = None
    e_peak_eV: float = 0.0
    sigma_eV: float = 0.1
    levels: Tuple[Tuple[float, float], ...] = ()
    charge: str = "split"
    neutral_level_eV: Optional[float] = None    # None -> 0.0 (E_i) for the intrinsic reference
    sigma_n_m2: float = 1.0e-20
    sigma_p_m2: float = 1.0e-20
    v_th_n_ms: float = 1.0e5
    v_th_p_ms: float = 1.0e5
    recombination: bool = True
    fixed_charge_C_m2: float = 0.0
    bin_eV: Optional[float] = None
    energy_reference: str = "intrinsic"
    dynamic: bool = False
    kinetic_bin_eV: Optional[float] = None

    def __post_init__(self):
        _check_reference("InterfaceTraps", self.energy_reference, 0.0)
        if not isinstance(self.dynamic, bool):
            raise TypeError("InterfaceTraps.dynamic must be a bool; got {!r}".format(self.dynamic))
        if self.kinetic_bin_eV is not None:
            if not self.dynamic:
                raise ValueError("InterfaceTraps.kinetic_bin_eV is the bin width of KINETIC bins; set "
                                 "dynamic=True (or use bin_eV for the quasi-static quadrature)")
            if not (math.isfinite(float(self.kinetic_bin_eV)) and float(self.kinetic_bin_eV) > 0.0):
                raise ValueError("InterfaceTraps.kinetic_bin_eV must be finite and > 0")
        if self.dynamic and not self.recombination:
            raise ValueError("InterfaceTraps(dynamic=True) needs recombination=True: a kinetic bin fills "
                             "and empties by capture from the bands, which IS the surface SRH (without it "
                             "the trapped charge would change with no carrier current -- not "
                             "charge-conserving)")
        if self.energy_reference != "intrinsic":
            for nm, v in (("e_lo_eV", self.e_lo_eV), ("e_hi_eV", self.e_hi_eV),
                          ("neutral_level_eV", self.neutral_level_eV)):
                if v is not None:
                    _check_reference("InterfaceTraps." + nm, self.energy_reference, v)
            for e, _n in self.levels:
                _check_reference("InterfaceTraps.levels", self.energy_reference, e)
            if self.charge == "split" and self.neutral_level_eV is None:
                raise ValueError("InterfaceTraps: charge='split' with energy_reference={!r} needs an "
                                 "explicit neutral_level_eV in that reference (the default 0 is E_i only)"
                                 .format(self.energy_reference))
        if self.distribution not in ("uniform", "gaussian", "discrete"):
            raise ValueError("InterfaceTraps.distribution must be uniform|gaussian|discrete")
        if self.charge not in _CHARGE_MODES:
            raise ValueError("InterfaceTraps.charge must be one of {}".format(_CHARGE_MODES))
        for nm in ("sigma_n_m2", "sigma_p_m2", "v_th_n_ms", "v_th_p_ms"):
            v = float(getattr(self, nm))
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError("InterfaceTraps.{} must be finite and > 0 (the occupancy needs both "
                                 "capture rates); got {!r}".format(nm, v))
        if not (float(self.dit_eV_m2) >= 0.0):
            raise ValueError("InterfaceTraps.dit_eV_m2 must be >= 0")
        if not math.isfinite(float(self.fixed_charge_C_m2)):
            raise ValueError("InterfaceTraps.fixed_charge_C_m2 must be finite")
        if self.distribution == "uniform" and self.dit_eV_m2 > 0.0:
            lo, hi = self._band()
            if lo is None or not (hi > lo):
                raise ValueError("uniform InterfaceTraps needs band_gap_eV or e_lo_eV < e_hi_eV")
        if self.distribution == "gaussian" and not (float(self.sigma_eV) > 0.0):
            raise ValueError("gaussian InterfaceTraps needs sigma_eV > 0")
        for lv in self.levels:
            if len(lv) != 2 or not (float(lv[1]) >= 0.0):
                raise ValueError("InterfaceTraps.levels entries are (energy_eV, N_it_m2 >= 0)")

    def _band(self):
        # the gap in the spec's coordinate: +-E_g/2 about E_i, or [0, E_g] from a band edge
        g = float(self.band_gap_eV) if self.band_gap_eV is not None else None
        dlo, dhi = ((-0.5 * g, 0.5 * g) if self.energy_reference == "intrinsic" else (0.0, g)) \
            if g is not None else (None, None)
        lo = self.e_lo_eV if self.e_lo_eV is not None else dlo
        hi = self.e_hi_eV if self.e_hi_eV is not None else dhi
        return lo, hi

    def neutral_level(self) -> float:
        """The neutral level E_0 in the spec's coordinate (0.0 = E_i for the intrinsic reference)."""
        return float(self.neutral_level_eV) if self.neutral_level_eV is not None else 0.0

    def sheet_levels(self, kT_eV: Optional[float] = None) -> List[Tuple[float, float, str]]:
        """Discretize D_it into sheet levels [(E_k [eV, in the spec's energy_reference], N_it,k [m^-2],
        kind_k)], kind in {acceptor, donor, neutral}. Bins never straddle the neutral level
        (charge='split'), so every bin has ONE charge type. "Above" the neutral level means toward E_c:
        a LARGER value for the intrinsic / valence coordinates, a SMALLER one for E_c - E.
        Pure numpy -- usable devsim-free.

        kT_eV: the thermal energy [eV] that sets the DEFAULT bin width (bin_eV=None) -- pass the
        region's kT/q (V_t) when the device does not run at T_REF; install_interface_traps does. None
        keeps kT(T_REF) = V_T (devsim-free callers at 300 K). Ignored when bin_eV (or, for a dynamic
        spec, kinetic_bin_eV) is given."""
        if self.dynamic and self.kinetic_bin_eV is not None:
            dmax = float(self.kinetic_bin_eV)
        elif self.bin_eV is not None:
            dmax = float(self.bin_eV)
        else:
            dmax = float(kT_eV) if kT_eV is not None else V_T
            if not (math.isfinite(dmax) and dmax > 0.0):
                raise ValueError("InterfaceTraps.sheet_levels: kT_eV must be finite and > 0, got {!r}"
                                 .format(kT_eV))
        e0 = self.neutral_level()
        toward_ec = -1.0 if self.energy_reference == "conduction" else 1.0

        def kind_of(e):
            if self.charge == "none":
                return "neutral"
            if self.charge in ("acceptor", "donor"):
                return self.charge
            return "acceptor" if toward_ec * (e - e0) > 0.0 else "donor"

        if self.distribution == "discrete":
            return [(float(e), float(n), kind_of(float(e))) for e, n in self.levels if float(n) > 0.0]
        if not (self.dit_eV_m2 > 0.0):
            return []
        if self.distribution == "uniform":
            lo, hi = self._band()
            dens = lambda e: np.full_like(e, float(self.dit_eV_m2))       # noqa: E731
        else:
            lo = float(self.e_peak_eV) - 5.0 * float(self.sigma_eV)
            hi = float(self.e_peak_eV) + 5.0 * float(self.sigma_eV)
            blo, bhi = self._band()
            if blo is not None:
                lo, hi = max(lo, blo), min(hi, bhi)
            dmax = min(dmax, 0.25 * float(self.sigma_eV))
            dens = lambda e: float(self.dit_eV_m2) * np.exp(                # noqa: E731
                -0.5 * ((e - float(self.e_peak_eV)) / float(self.sigma_eV)) ** 2)
        edges = [lo, hi]
        if self.charge == "split" and lo < e0 < hi:
            edges = [lo, e0, hi]
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            nb = max(1, int(math.ceil((b - a) / dmax - 1e-12)))
            de = (b - a) / nb
            ec = a + (np.arange(nb) + 0.5) * de
            for e, d in zip(ec, dens(ec)):
                out.append((float(e), float(d) * de, kind_of(float(e))))
        return out


# ------------------------------------------------------------------------------------------------
# DEVSIM installers
# ------------------------------------------------------------------------------------------------
def install_bulk_traps(device: str, region: str, traps: Sequence[TrapLevel], *, n_i_m3: float,
                       V_t: float = V_T, bgn: bool = False, fd: bool = False) -> None:
    """Install discrete bulk SRH levels on a bipolar region set up by
    physics_bipolar_dd.setup_bipolar_region (which calls this with traps=...).

    Per level k: parameters trap<k>_n1/_p1/_taun/_taup/_Nt (n1, p1 = n_i exp(+-E_t/kT) at the region's
    V_t), the diagnostic occupancy TrapOccupancy<k>, and its rate as ONE recombination term
    RTerm_trap<k> of the shared additive mechanism (physics_bipolar_dd.add_recombination_term) WITH
    its pair rate G + R (so the noise GR source sees it -- noise_analysis). The charged levels' trapped
    charge goes in as the single charge term QTerm_bulk_traps. Rate, occupancy and pair rate all come
    from the ONE SRH kernel (srh_rate_expr / srh_occupancy_expr) with the region's BGN substitution
    (bgn=True: n_i^2 -> n_ie^2, n1/p1 scaled by n_ie/n_i) and, under statistics="fermi_dirac" (fd=True),
    the exact FD emission densities gamma_n n1, gamma_p p1 (srh_reference_densities; the charge term's
    Jacobian then follows the solution-dependent gamma through DEVSIM's FDNiN:Electrons-style models) --
    for the quasi-static AND the kinetic levels (v0.15 FD-mode completion, integration).

    A KINETIC level (TrapLevel.dynamic=True; v0.15 plan 4.3.2) gets instead (k = its index, name
    trap<k>): the node solution TrapW_trap<k> -- the occupancy ODDS w = f/(1-f) = exp(u), u = (E_Ft -
    E_t)/kT the level's own quasi-Fermi level -- with f = TrapOccupancy<k> = w/(1+w), 1 - f =
    TrapVacancy<k> = 1/(1+w) (no 1 - f cancellation), the steady-state diagnostics TrapOccupancySS<k> and
    TrapWSS_trap<k> (its odds, the seed), the single-carrier terms RTermN_trap<k> = R_n (electron row
    only) and RTermP_trap<k> = R_p (hole row only) of srh_capture_exprs with closed-form derivatives
    (Electrons, Holes, w) and the trap row as their partner (+1 / -1, mechanism "trap_n" / "trap_p"),
    and the equation TrapEquation_trap<k> (variable TrapW_trap<k>, variable_update "positive", recorded
    in eq_registry):
        time_node_model TrapCharge_trap<k> = -q N_t f       node_model TrapNode_trap<k> = +q (R_n - R_p)
    i.e. N_t df/dt = R_n - R_p in DEVSIM's residual convention (d(time)/dt + node = 0) -- the stored charge
    is exactly the Poisson charge, so a BDF step conserves charge to round-off. No fast-trap pair term
    RTerm_trap<k> is installed for it (never counted twice). Charged kinetic levels carry their charge
    -N_t f / +N_t (1 - f) in the separate charge term QTerm_kinetic_traps (derivatives with respect to w
    only); QTerm_bulk_traps keeps the quasi-static levels, so a trap set without a kinetic level builds
    exactly the shipped models. Each branch carries its EVENT rate (srh_capture_exprs(pair=True)), the
    trap g-r noise source of noise_analysis (v0.15 K3): S = q^2 V [2 P_n |y_t - y_n|^2 + 2 P_p |y_p - y_t|^2].

    WHY THE ODDS, NOT THE LOGIT u (plan decision 40 chose u; deviation MEASURED, docs/audit/2026-11-06-
    v0.15-kinetic-traps.md section 2). The f-row is a sigmoid in u: Newton's linear step out of a
    saturated level (f = 1 - 1e-12 emitting, the DLTS case; or 1e-12 filling) is df/(f(1-f)) ~ 1e10 and
    exp(u) overflows -- DEVSIM aborts, and its only damping (log_damp) is hard-wired to 0.0259 per step.
    In w the same steps converge in 4-16 iterations: DEVSIM's "positive" update bounds a step toward 0
    to 3 decades, and the row is linear in f = w/(1+w) near w -> 0. Every property decision 40 wanted
    is kept: 0 < f < 1 under any update (w > 0), no 1 - f cancellation, an exact seed (w_SS =
    srh_occupancy_odds_expr), the equilibrium value exp((E_F - E_t)/kT).

    WELL-POSEDNESS (for the static floating-body check): every trap row is pinned by its own capture
    and emission -- d(R_n - R_p)/dw = -[(n + n1')/tau_n0 + (p + p1')/tau_p0] (1 - f)^2 < 0 strictly for
    finite w -- so the DC Jacobian of TrapEquation_* has a nonzero diagonal at every node, contact nodes
    included (the trap row is not replaced there: DEVSIM keeps a node-only equation's bulk row at a
    contact node, P3 0.3). The trap rows are therefore DC well-posed candidates like the continuity
    rows; a checker that lists equation variables must include TrapW_* (and the interface ItW*)."""
    from dynameta.carriers import physics_bipolar_dd as _BP   # lazy: BP imports this module

    charges = []
    kin_q, kin_dq = [], {}
    for k, t in enumerate(traps):
        if not isinstance(t, TrapLevel):
            raise TypeError("traps must be TrapLevel instances; got {!r}".format(type(t).__name__))
        p = "trap{}_".format(k)
        ref = t.energy_reference
        if ref == "intrinsic":
            n1, p1 = n1_p1(t.energy_eV, n_i_m3, V_t)           # the shipped arithmetic, unchanged
        else:                                                 # band edge: the region's N_c / N_v
            n1, p1 = level_reference_densities(
                t.energy_eV, ref, n_i_m3=n_i_m3, V_t=V_t,
                N_c_m3=float(ds.get_parameter(device=device, region=region, name="N_dos")),
                N_v_m3=float(ds.get_parameter(device=device, region=region, name="N_dos_p")))
        tn, tp = t.lifetimes_s()
        for nm, v in (("n1", n1), ("p1", p1), ("taun", tn), ("taup", tp),
                      ("Nt", float(t.density_m3))):
            ds.set_parameter(device=device, region=region, name=p + nm, value=v)
        args = (p + "n1", p + "p1", p + "taun", p + "taup")
        fkw = {"fd": True} if fd else {}                        # (the legacy call, unchanged, otherwise)
        occ = srh_occupancy_expr(*args, bgn=bgn, reference=ref, **fkw)
        if t.dynamic:
            var, f, g = _install_kinetic_level(device, region, k, p, args, occ, bgn=bgn, reference=ref, fd=fd)
            if t.kind != "neutral":
                sgn, occm = ("-", f) if t.kind == "acceptor" else ("", g)
                kin_q.append("{}{}Nt*{}".format(sgn, p, occm))
                for v in ("Electrons", "Holes", var):   # through the occupancy models' own derivatives
                    kin_dq.setdefault(v, []).append("{}{}Nt*{}:{}".format(sgn, p, occm, v))
            continue
        ds.node_model(device=device, region=region, name="TrapOccupancy{}".format(k), equation=occ)
        _BP.add_recombination_term(device, region, "trap{}".format(k),
                                   srh_rate_expr(*args, bgn=bgn, reference=ref, **fkw),
                                   pair=srh_rate_expr(*args, bgn=bgn, pair=True, reference=ref, **fkw))
        if t.kind == "acceptor":
            charges.append("-{}Nt*{}".format(p, occ))
        elif t.kind == "donor":
            charges.append("{}Nt*(1 - {})".format(p, occ))
    if charges:
        _BP.add_charge_term(device, region, "bulk_traps", " + ".join(charges))
    if kin_q:
        _BP.add_charge_term(device, region, KINETIC_CHARGE_TERM, " + ".join(kin_q),
                            wrt=tuple(v for v in kin_dq if v not in ("Electrons", "Holes")),
                            derivs={v: " + ".join(e) for v, e in kin_dq.items()})


#: the charge term of the CHARGED kinetic bulk levels (QTerm_<this>); quasi-static ones stay in bulk_traps
KINETIC_CHARGE_TERM = "kinetic_traps"
#: (variable prefix, steady-state-odds seed prefix): every kinetic occupancy variable and its seed model
_KINETIC_VARS = (("TrapW_", "TrapWSS_"), ("ItW", "ItWSS"))
#: |u| = |ln w| ceiling: w = exp(u) leaves double range past |u| ~ 709 (and its Newton steps scale with w),
#: so a seed beyond this is refused (a deep level at low T)
U_LIMIT = 600.0
#: default ceiling on the extra unknowns of KINETIC interface bins (bins x semiconductor-region nodes): the
#: masked region-wide variables cost one unknown per region node per bin (DEVSIM 2.11 has no interface
#: solution variable). Calibrated on box A (docs/audit/2026-11-06-v0.15-kinetic-traps.md, section 4.3): a 2-D
#: MOS DC Newton iteration costs ~0.3 s at 5e4, ~1.1 s at 1e5-3e5, 3.5 s at 6e5 extra unknowns; the install
#: grows ~K^2 in the bin count K (4 s at 64 bins, 228 s at 200). install_interface_traps(max_kinetic_unknowns=)
#: overrides it per call.
KINETIC_UNKNOWN_BUDGET = 200_000


class TrapOccupancyRangeError(ValueError):
    """A kinetic level's |u| = |ln w| = |E_Ft - E_t|/kT exceeds U_LIMIT at its seed: the level is so far
    from the Fermi level (a deep level at low temperature) that its odds w = exp(u) leaves the double
    range in the Newton. Use the quasi-static model (dynamic=False) for it -- its occupancy is then 0 or 1 to double
    precision anyway."""


def _region_fd(device: str, region: str) -> bool:
    """statistics='fermi_dirac' on `region` (bipolar_options.install_fd_detailed_balance's models present):
    every SRH kernel then takes fd=True (the exact FD emission densities). The kinetic levels and bins
    support it since the v0.15 FD-mode integration (their n1', p1' models carry the gamma derivatives)."""
    return "FDGammaN" in ds.get_node_model_list(device=device, region=region)


def _initial_odds(device: str, region: str, seed: str, fd: bool):
    """The value a new occupancy variable holds until seed_kinetic_traps: its seed model w_SS at the current
    carriers -- or, under statistics='fermi_dirac', 1.0 (f = 1/2): the FD seed reads ln(c/N), which is not
    defined on carriers that are not set yet (0 at install; a FATAL FP exception, MEASURED). Either way the
    seed call replaces it."""
    if fd:
        return [1.0] * len(ds.get_node_model_values(device=device, region=region, name="x"))
    return ds.get_node_model_values(device=device, region=region, name=seed)


def _install_kinetic_level(device: str, region: str, k: int, p: str, args, occ: str, *, bgn: bool,
                           reference: str, fd: bool = False):
    """The variable, models, single-carrier terms and equation of one kinetic bulk level (see
    install_bulk_traps). Returns (variable, f model, 1 - f model)."""
    from dynameta.carriers import physics_bipolar_dd as _BP
    name = "trap{}".format(k)
    var, eq = "TrapW_" + name, "TrapEquation_" + name
    f, g = "TrapOccupancy{}".format(k), "TrapVacancy{}".format(k)
    seed = "TrapWSS_" + name
    ds.node_model(device=device, region=region, name="TrapOccupancySS{}".format(k), equation=occ)
    ds.node_model(device=device, region=region, name=seed,
                  equation=srh_occupancy_odds_expr(*args, bgn=bgn, reference=reference, fd=fd))
    ds.node_solution(device=device, region=region, name=var)             # FIRST: the terms need it
    ds.set_node_values(device=device, region=region, name=var,          # finite, > 0 until seeded
                       values=_initial_odds(device, region, seed, fd))
    _occupancy_models(device, region, var, f, g, *args, bgn=bgn, reference=reference, fd=fd)
    rn, rp = srh_capture_exprs(*args, f=f, one_minus_f=g, bgn=bgn, reference=reference, fd=fd)
    pn, pp = srh_capture_exprs(*args, f=f, one_minus_f=g, bgn=bgn, reference=reference, pair=True, fd=fd)
    dn = dp = None          # DEVSIM's diff through the occupancy models (they carry n, p, w derivatives)
    # the event rates P_n, P_p (both directions of each exchange) are the trap g-r noise sources (K3): an
    # electron event changes the electron row by -1 and the trap row by +1, a hole event the hole row by +1
    # and the trap row by -1 (TrapNode = +q (R_n - R_p)) -- noise_analysis's single-carrier sources
    _BP.add_recombination_term(device, region, name, rn, carrier="electron", wrt=(var,), derivs=dn,
                               pair=pn, partner=(eq, +1.0), mechanism="trap_n")
    _BP.add_recombination_term(device, region, name, rp, carrier="hole", wrt=(var,), derivs=dp,
                               pair=pp, partner=(eq, -1.0), mechanism="trap_p")
    _trap_row(device, region, var, eq, "TrapNode_" + name, "TrapCharge_" + name, name, p + "Nt", seed)
    return var, f, g


def _occupancy_models(device, region, var, f, g, n1, p1, taun, taup, *, bgn, reference, fd=False):
    """The occupancy models every kinetic term reads by name, BLENDED by the DC-form switch D
    (KINETIC_DC_PARAMETER): f = D f_SS(n, p) + (1 - D) w/(1+w), 1 - f = D (1 - f_SS) + (1 - D)/(1+w). D = 0 --
    every analysis -- is the kinetic occupancy (0 x f_SS = 0 exactly, so the values are w's); D = 1 -- inside
    kinetic_dc_form only -- is the quasi-static occupancy, so the rates and the charge are the quasi-static
    model. Derivatives in CLOSED FORM, without a 1 - f subtraction (DEVSIM's diff of w/(1+w) is 1/(1+w) -
    w/(1+w)^2, which cancels for a full level: MEASURED 2e4 eps on the Poisson column): df/dw = (1 - D)/(1+w)^2,
    df_SS/dn = tau_p0 (1 - f_SS)/Den, df_SS/dp = -tau_n0 f_SS/Den (Den the SRH denominator), and 1 - f the
    negatives. fd=True (statistics='fermi_dirac'): n1' ~ gamma_n(n), p1' ~ gamma_p(p) also move Den, so
    df_SS/dn = tau_p0 ((1 - f_SS) - f_SS n1' ln)/Den and df_SS/dp = tau_n0 ((1 - f_SS) p1' lp - f_SS)/Den
    (ln, lp = FDdLnGammaN, FDdLnGammaP)."""
    if KINETIC_DC_PARAMETER not in ds.get_parameter_list(device=device, region=region):
        ds.set_parameter(device=device, region=region, name=KINETIC_DC_PARAMETER, value=0.0)
    d = KINETIC_DC_PARAMETER
    n1e, p1e, _ = srh_reference_densities(n1, p1, bgn=bgn, reference=reference, fd=fd)
    den = _srh_denominator(n1e, p1e, taun, taup)
    fss = srh_occupancy_expr(n1, p1, taun, taup, bgn=bgn, reference=reference, fd=fd)
    gss = srh_vacancy_expr(n1, p1, taun, taup, bgn=bgn, reference=reference, fd=fd)
    gw = "(1/(1 + {}))".format(var)
    if fd:
        de = "{d}*{tp}*({g} - {f}*{n1}*FDdLnGammaN)/{den}".format(d=d, tp=taup, g=gss, f=fss, n1=n1e, den=den)
        dh = "{d}*{tn}*({g}*{p1}*FDdLnGammaP - {f})/{den}".format(d=d, tn=taun, g=gss, f=fss, p1=p1e, den=den)
    else:
        de = "{d}*{tp}*{g}/{den}".format(d=d, tp=taup, g=gss, den=den)
        dh = "-{d}*{tn}*{f}/{den}".format(d=d, tn=taun, f=fss, den=den)
    for name, expr, sgn in ((f, "{d}*{s} + (1 - {d})*({w}/(1 + {w}))".format(d=d, s=fss, w=var), ""),
                            (g, "{d}*{s} + (1 - {d})*{gw}".format(d=d, s=gss, gw=gw), "-")):
        ds.node_model(device=device, region=region, name=name, equation=expr)
        for v, e in (("Electrons", de), ("Holes", dh),
                     (var, "(1 - {d})*{gw}^2".format(d=d, gw=gw))):
            ds.node_model(device=device, region=region, name="{}:{}".format(name, v),
                          equation="{}({})".format(sgn, e))


def _trap_row(device, region, var, eq, node, time, term, nt, seed, *, area=None, mask=None):
    """The occupancy equation of one kinetic level / interface bin `term`, variable_update "positive" (w > 0:
    0 < f < 1 under any update), time model -q N w/(1+w) [x weight] (always the kinetic charge). Node model:
        (1 - D) q (R_n - R_p) [x weight]  +  D (w - w_SS)  [+ the off-interface mask row (w - 1)]
    D = 0 (every analysis): the physical row, read through the term models by NAME (DEVSIM's chain rule
    through named models, measured in 4.0.2). D = 1 (kinetic_dc_form, the DC solves of dc_solve.solve_dc):
    the rates are the quasi-static ones (R_n = R_p), and w is SLAVED to its steady state w_SS (the `seed`
    model) -- with NO n, p derivative: w_SS carries no derivative model, which DEVSIM differentiates to 0
    (MEASURED in 4.0.2: a missing model derivative is a silent zero, used here on purpose), so w lags the
    carriers by one Newton iteration and can never feed back. The DC root is the same in both forms."""
    w = "" if area is None else "{}*".format(area)
    d = KINETIC_DC_PARAMETER
    expr = "(1 - {d})*ElectronCharge*(RTermN_{t} - RTermP_{t}) + {d}*({v} - {s})".format(d=d, t=term, v=var,
                                                                                      s=seed)
    if mask is not None:                       # off the interface: NodeVolume (w - 1) = 0 (w > 0 inert)
        expr = "{} + {}*({} - 1)".format(expr, mask, var)
    node_with_derivs(device, region, node, expr, ("Electrons", "Holes", var))
    f = "({v}/(1 + {v}))".format(v=var)
    ds.node_model(device=device, region=region, name=time,
                  equation="-ElectronCharge*{}{}*{}".format(w, nt, f))
    ds.node_model(device=device, region=region, name="{}:{}".format(time, var),
                  equation="-ElectronCharge*{}{}*(1/(1 + {}))^2".format(w, nt, var))
    record_region_equation(device, region, name=eq, variable_name=var, node_model=node,
                           time_node_model=time, variable_update="positive")


#: region parameter switching every kinetic level between its PHYSICAL form (0: the kinetic model, what the
#: transient, ssac and noise solves see) and its DC form (1: the quasi-static model with w slaved to w_SS) --
#: see kinetic_dc_form
KINETIC_DC_PARAMETER = "KineticDCForm"


@contextmanager
def kinetic_dc_form(m=ds):
    """Inside the block every kinetic level of every live device is in its DC FORM: its rates and charge use
    the QUASI-STATIC occupancy f_SS(n, p) and its row slaves w to w_SS (_occupancy_models, _trap_row) -- the DC
    Newton is then the quasi-static model's own, with w following one iteration behind and never feeding
    back. Why: the physical row's linearization in w at a saturated level divides by (1 - f)^2, so a large
    carrier change in one iteration -- a gated 2-D device going straight into the coupled Newton from its
    neutral seed -- flings w past 1e150 ((1 - f)^2 underflows, the factorization fails) or, in a linear (1 + w)
    form, lets the interface charge swing by q N_it per bin and the carriers diverge (both MEASURED on a gated
    bipolar GaAs builder with kinetic interface bins; the quasi-static twin converges). The DC root is the SAME
    (R_n = R_p <=> w = w_SS); only the Newton path changes. dc_solve.solve_dc
    (hence every builder) wraps its Newton in this; a raw ds.solve(type='dc') uses the physical form (it
    converges from a seeded state, tests/test_trap_kinetics.py). Never wrap a transient, ssac or noise solve:
    their rows must be the physical ones. A no-op without kinetic traps (no parameter is touched)."""
    hits = []
    lister = getattr(m, "get_device_list", None)
    for d in (lister() if lister is not None else ()):
        for r in m.get_region_list(device=d):
            if KINETIC_DC_PARAMETER in m.get_parameter_list(device=d, region=r):
                hits.append((d, r))
    for d, r in hits:
        m.set_parameter(device=d, region=r, name=KINETIC_DC_PARAMETER, value=1.0)
    try:
        yield hits
    finally:
        for d, r in hits:
            m.set_parameter(device=d, region=r, name=KINETIC_DC_PARAMETER, value=0.0)


def kinetic_trap_variables(device: str, region: str) -> List[str]:
    """The kinetic occupancy variables (TrapW_*, ItW*) of a region, read back from DEVSIM."""
    have = set(ds.get_node_model_list(device=device, region=region))
    # a variable is <prefix><suffix> WITH its seed model <seed prefix><suffix> (DEVSIM adds helper models
    # such as <var>_prev after a solve; the seed model is what identifies an installed variable)
    return sorted(m for m in have for v, s in _KINETIC_VARS
                  if m.startswith(v) and not m.startswith(s) and s + m[len(v):] in have)


#: the solve types whose answer a QUASI-STATIC charged trap makes wrong (plan decision 39): ssac, the noise
#: solve, and a transient (advised once, at its transient_dc start, not at every step)
QUASI_STATIC_ADVISED_SOLVES = ("ac", "noise", "transient_dc")
QUASI_STATIC_TRAP_MESSAGE = (
    "physics_traps: a {kind} solve with QUASI-STATIC CHARGED traps ({where}): their occupancy follows n and p "
    "instantly, so the capture current that fills them is missing from the continuity time terms -- the "
    "small-signal / transient / noise answer is NOT charge-conserving above the capture rate (measured: the "
    "FDT misses by 6.6e-2 at 10 MHz; G_p of D_it is 0). Use TrapLevel(dynamic=True) / InterfaceTraps("
    "dynamic=True) for the kinetic model; the quasi-static answer is exact only well below the capture "
    "rates (DC, LF C-V).")


def quasi_static_charged_traps(device: str, m=ds) -> List[Tuple[str, str]]:
    """[(region, charge term)] of the device's QUASI-STATIC CHARGED traps -- the bulk levels' QTerm_bulk_traps
    and every interface install's QTerm_it<j> whose charge depends on the carriers (a nonzero d/dn or d/dp
    somewhere: a fixed oxide charge Q_f alone does not count, kinetic levels live in their own terms).
    Read from DEVSIM, so it dies with the device. `m` is the devsim module (a stand-in without
    get_device_list -- a control-flow test double -- has none)."""
    out = []
    try:
        regions = m.get_region_list(device=device)
    except Exception:                                           # pragma: no cover - not a device
        return out
    for r in regions:
        have = set(m.get_node_model_list(device=device, region=r))
        for t in sorted(t for t in have if ":" not in t and (
                t == "QTerm_bulk_traps" or (t.startswith("QTerm_it") and t[8:].isdigit()))):
            for v in ("Electrons", "Holes"):
                d = "{}:{}".format(t, v)
                if d in have and np.any(np.asarray(m.get_node_model_values(
                        device=device, region=r, name=d)) != 0.0):
                    out.append((r, t))
                    break
    return out


def warn_if_quasi_static_traps(kind: str, m=ds, stacklevel: int = 3) -> None:
    """The decision-39 advisory: an AdvisoryWarning when a solve of type `kind` (QUASI_STATIC_ADVISED_SOLVES)
    runs while a live device carries quasi-static charged traps. Called by extended_precision.solve, the
    one solve site every DynaMeta analysis goes through (ssac_admittance, noise_spectrum, transient_step)."""
    if kind not in QUASI_STATIC_ADVISED_SOLVES:
        return
    lister = getattr(m, "get_device_list", None)
    if lister is None:
        return
    hits = ["{}/{}".format(d, r) + ("" if t == "QTerm_bulk_traps" else " (interface)")
            for d in lister() for r, t in quasi_static_charged_traps(d, m)]
    if hits:
        import warnings
        from dynameta.warning_types import AdvisoryWarning
        warnings.warn(QUASI_STATIC_TRAP_MESSAGE.format(kind=kind, where=", ".join(hits)), AdvisoryWarning,
                      stacklevel=stacklevel)


def seed_kinetic_traps(device: str, regions=None) -> dict:
    """Seed every kinetic occupancy variable w (TrapW_*, ItW*) of `regions` (default: all) with its
    steady-state value at the CURRENT carrier densities, w = f_SS/(1 - f_SS) (srh_occupancy_odds_expr)
    -- exactly the quasi-static occupancy, so the first coupled Newton iterate is the quasi-static
    state. Call it after the carriers are seeded and before the first DC solve (the builders do; a
    hand-built device must), and again after any staged pre-solve that re-seeds the carriers. A DC solve
    also converges from other seeds, but slower (the odds of a nearly full level grow ~2x per Newton
    iteration). Refuses (TrapOccupancyRangeError) a seed with |ln w| > U_LIMIT. Returns {(region,
    variable): max |ln w|}. Do NOT call it between transient steps: it would erase the trap memory the
    kinetic model exists for."""
    out = {}
    for r in (ds.get_region_list(device=device) if regions is None else regions):
        for var in kinetic_trap_variables(device, r):
            pre, spre = next((v, s) for v, s in _KINETIC_VARS if var.startswith(v))
            w = np.asarray(ds.get_node_model_values(device=device, region=r,
                                                    name=spre + var[len(pre):]), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                lu = np.abs(np.log(w))
            umax = float(np.max(lu)) if w.size else 0.0
            if not (np.all(w > 0.0) and np.all(np.isfinite(lu)) and umax <= U_LIMIT):
                raise TrapOccupancyRangeError(
                    "seed_kinetic_traps: {!r} on region {!r} has |u| = |ln w| = |E_Ft - E_t|/kT = {:.4g} "
                    "at its seed (limit {}): the occupancy odds would leave the double range in the "
                    "Newton. Use dynamic=False for this level (its occupancy is 0 or 1 to double "
                    "precision).".format(var, r, umax, U_LIMIT))
            ds.set_node_values(device=device, region=r, name=var, values=w.tolist())
            out[(r, var)] = umax
    return out


def equilibrium_trap_levels(traps) -> Tuple[Tuple[float, float, str], ...]:
    """(energy_eV, N_t [m^-3], kind) of the CHARGED bulk levels -- what the charge-neutral equilibrium
    of the ohmic contacts / seed / 3-D gate frame needs (bipolar_options.equilibrium_fields). Neutral
    (recombination-only) levels carry no charge and are left out, so a trap set without charged levels
    returns () and leaves the legacy contact rule in place. A band-referenced level carries its
    reference as a 4th entry, (E, N_t, kind, reference) -- equilibrium_n1 reads it; intrinsic levels
    keep the 3-tuple, so the shipped root is unchanged for them."""
    out = []
    for t in (traps or ()):
        if not isinstance(t, TrapLevel):
            raise TypeError("traps must be TrapLevel instances; got {!r}".format(type(t).__name__))
        if t.kind != "neutral":
            out.append((float(t.energy_eV), float(t.density_m3), t.kind)
                       + (() if t.energy_reference == "intrinsic" else (t.energy_reference,)))
    return tuple(out)


def _semiconductor_side(device: str, interface: str) -> Tuple[str, str]:
    regions = list(ds.get_region_list(device=device, interface=interface))

    def bipolar(r):
        try:
            return "Holes" in ds.get_node_model_list(device=device, region=r) and \
                   "USRH" in ds.get_node_model_list(device=device, region=r)
        except Exception:                                      # pragma: no cover - devsim internals
            return False

    semi = [r for r in regions if bipolar(r)]
    if len(semi) == 2:
        raise NotImplementedError(
            "install_interface_traps: interface {!r} joins two carrier regions {}; interface traps "
            "are implemented for semiconductor/INSULATOR interfaces only (a semiconductor/"
            "semiconductor interface needs the carrier-continuity interface conditions of the "
            "heterojunction work).".format(interface, regions))
    if not semi:
        raise ValueError(
            "install_interface_traps: neither region of interface {!r} ({}) is a bipolar drift-"
            "diffusion region (physics_bipolar_dd.setup_bipolar_region): the SRH occupancy of an "
            "interface trap needs both n and p at the surface.".format(interface, regions))
    other = [r for r in regions if r != semi[0]][0]
    return semi[0], other


def _triangle_node_areas(v: np.ndarray) -> np.ndarray:
    """The three node shares of a triangle v (3, 3): circumcentric (Voronoi) for a non-obtuse triangle,
    barycentric thirds for an obtuse one (see interface_node_area). Sums to the triangle area."""
    a2 = 0.5 * float(np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0])))
    out = np.empty(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        # cot of the angle at k (opposite edge ij) and at j (opposite edge ik)
        ukj, uki = v[j] - v[k], v[i] - v[k]
        cot_k = float(np.dot(ukj, uki)) / (2.0 * a2)
        ujk, uji = v[k] - v[j], v[i] - v[j]
        cot_j = float(np.dot(ujk, uji)) / (2.0 * a2)
        out[i] = (float(np.dot(v[j] - v[i], v[j] - v[i])) * cot_k
                  + float(np.dot(v[k] - v[i], v[k] - v[i])) * cot_j) / 8.0
    if np.any(out < 0.0):                                      # obtuse: the circumcentre is outside
        return np.full(3, a2 / 3.0)
    return out


def interface_node_area(device: str, region: str, interface: str) -> np.ndarray:
    """Interface area [m^2 in 3-D, m in 2-D, 1 in 1-D] OWNED by each node of `region` on `interface`
    (0 elsewhere), from DEVSIM's interface element list: 1-D a point (1), 2-D half of each interface
    edge, 3-D each interface triangle split CIRCUMCENTRICALLY (the Voronoi share of node i is
    (|e_ij|^2 cot theta_k + |e_ik|^2 cot theta_j)/8), i.e. the interface face of node i's box-method
    control volume -- the area the FV equations actually give that node. D2c (v0.14.0): this used to be
    a third of each triangle, which on the tensor (Kuhn) mesh of devsim_3d gives the two nodes on a
    face diagonal 1/3 of the face and the other two 1/6 (a 2x checkerboard in a laterally uniform
    D_it; MEASURED psi_s scatter below); the circumcentric split gives each node exactly its quarter of
    every adjacent rectangle and equals DEVSIM's SurfaceArea node-for-node. An OBTUSE triangle (only
    on an unstructured mesh, e.g. mesher='gmsh') keeps the barycentric thirds, since its circumcentric
    shares go negative. The total is the triangle area either way. Cartesian coordinates only."""
    elems = ds.get_element_node_list(device=device, region=region, interface=interface)
    x = np.asarray(ds.get_node_model_values(device=device, region=region, name="x"))
    y = np.asarray(ds.get_node_model_values(device=device, region=region, name="y"))
    z = np.asarray(ds.get_node_model_values(device=device, region=region, name="z"))
    xyz = np.column_stack([x, y, z])
    area = np.zeros(x.size)
    for e in elems:
        e = [int(i) for i in e]
        if len(e) == 1:
            area[e[0]] += 1.0
        elif len(e) == 2:
            area[e] += 0.5 * float(np.linalg.norm(xyz[e[1]] - xyz[e[0]]))
        elif len(e) == 3:
            area[e] += _triangle_node_areas(xyz[e])
        else:                                                  # pragma: no cover - not a DEVSIM surface
            raise ValueError("unexpected interface element with {} nodes".format(len(e)))
    if not np.any(area > 0.0):
        raise ValueError("interface {!r} has no nodes on region {!r}".format(interface, region))
    return area


def install_interface_traps(device: str, interface: str, traps, *,
                            max_kinetic_unknowns: Optional[int] = None) -> dict:
    """Attach interface traps (one InterfaceTraps or a sequence of them) to a semiconductor/insulator
    `interface` of a built device. The semiconductor side must be a bipolar DD region; its Poisson node
    charge gains the sheet charge Q_it/q (+ the fixed Q_f/q) and both continuity equations gain the
    surface SRH sink, through physics_bipolar_dd.add_charge_term / add_recombination_term (the
    composite models are rebuilt, the equations untouched). Call AFTER the region physics and the
    interface Potential continuity are set up, BEFORE solving.

    Diagnostics installed on the semiconductor region (j = the install index on that region):
      ItArea<j>   : A_node/NodeVolume [1/m] (0 off the interface),
      ItCharge<j> : trapped + fixed sheet charge [C/m^2] at each interface node (meaningful where
                    ItArea<j> > 0),
      ItRecomb<j> : surface SRH rate U_s [m^-2 s^-1] (when recombination is on).
    Returns {'region', 'insulator', 'index', 'levels', 'area'} (+ 'kinetic': the kinetic bins'
    variables, in level order).

    KINETIC bins (InterfaceTraps(dynamic=True); v0.15 plan 4.3.2, decision 41). Bin b of install j (b
    counts the kinetic bins of every spec, in order) is a region-wide odds variable ItW<j>_<b> masked
    off the interface (DEVSIM 2.11 has no interface solution variable: interface_equation only couples
    two region rows, node_solution is region-scoped). Its row ItEquation<j>_<b> is
        node  ItArea<j> q (R_n,b - R_p,b) + ItOffMask<j> (w - 1)       time  -q ItArea<j> N_it,b f_b
    with R_n,b = N_b c_n [n_s (1-f) - n1' f], R_p,b = N_b c_p [p_s f - p1' (1-f)] (srh_capture_exprs on
    the surface densities, tau -> 1/(c N_b)) entering the electron / hole rows as the single-carrier
    terms RTermN_it<j>_<b> / RTermP_it<j>_<b> (partner the bin's row, mechanism trap_n / trap_p, event
    rates ItArea N_b c_n x srh_capture_exprs(pair=True): the noise source weight is then A_node). ItOffMask<j> is 1 where ItArea<j> = 0: there the row is
    NodeVolume (w - 1) = 0, decoupled and diagonal (w = 1 is inert -- every term carries ItArea). The
    trapped charge of the kinetic bins is QTerm_itk<j>; ItCharge<j> then includes it; ItOccupancy<j>_<b>
    = f_b. COST: (kinetic bins) x (region nodes) unknowns; above `max_kinetic_unknowns` (default
    KINETIC_UNKNOWN_BUDGET) the install is REFUSED before any DEVSIM model is created -- coarsen with
    InterfaceTraps(kinetic_bin_eV=) or pass a larger budget knowingly. Seed with seed_kinetic_traps."""
    from dynameta.carriers import physics_bipolar_dd as _BP   # lazy: BP imports this module

    specs = [traps] if isinstance(traps, InterfaceTraps) else list(traps)
    if not specs or not all(isinstance(s, InterfaceTraps) for s in specs):
        raise TypeError("install_interface_traps: traps must be an InterfaceTraps or a sequence of them")
    semi, other = _semiconductor_side(device, interface)
    if any(sp.dynamic for sp in specs):                        # refuse BEFORE any DEVSIM write
        vt0 = float(ds.get_parameter(device=device, region=semi, name="V_t"))
        nbin = sum(len(sp.sheet_levels(kT_eV=vt0)) for sp in specs if sp.dynamic)
        nnode = len(ds.get_node_model_values(device=device, region=semi, name="NodeVolume"))
        budget = KINETIC_UNKNOWN_BUDGET if max_kinetic_unknowns is None else int(max_kinetic_unknowns)
        if nbin * nnode > budget:
            raise ValueError(
                "install_interface_traps: {} kinetic interface bins x {} nodes of region {!r} = {} extra "
                "unknowns exceeds the budget {} (each kinetic bin is a region-wide variable: DEVSIM has "
                "no interface unknowns). Coarsen the kinetic bins (InterfaceTraps(kinetic_bin_eV=...)), "
                "restrict the D_it band, or pass max_kinetic_unknowns= knowingly (cost: docs/audit/"
                "2026-11-06-v0.15-kinetic-traps.md, section K2).".format(
                    nbin, nnode, semi, nbin * nnode, budget))
    existing = [m for m in ds.get_node_model_list(device=device, region=semi)
                if m.startswith("ItArea")]
    j = len(existing)
    area = interface_node_area(device, semi, interface)
    vol = np.asarray(ds.get_node_model_values(device=device, region=semi, name="NodeVolume"))
    weight = np.where(area > 0.0, area / np.where(vol > 0.0, vol, 1.0), 0.0)
    a_name = "ItArea{}".format(j)
    ds.node_solution(device=device, region=semi, name=a_name)
    ds.set_node_values(device=device, region=semi, name=a_name, values=[float(v) for v in weight])

    n_i = float(ds.get_parameter(device=device, region=semi, name="n_i"))
    V_t = float(ds.get_parameter(device=device, region=semi, name="V_t"))   # == V_T unless T_K is set
    bgn = "n_ie" in ds.get_node_model_list(device=device, region=semi)      # BGN on this region (D1b)
    # v0.15 statistics="fermi_dirac": the exact FD emission densities (srh_reference_densities fd=True),
    # whose gamma_n, gamma_p -- and n_ie^2 = n_ie0^2 gamma_n gamma_p -- follow the surface densities; the
    # analytic derivatives below then carry the lambda = d ln gamma/dc terms.
    fd = _region_fd(device, semi)
    fkw = {"fd": True} if fd else {}
    n_c = float(ds.get_parameter(device=device, region=semi, name="N_dos"))   # band-referenced levels
    n_v = float(ds.get_parameter(device=device, region=semi, name="N_dos_p"))
    # Per sheet level k (one spec: shared capture ratio cr = c_n/c_p, c = sigma v_th): the SRH kernel
    # with taun -> 1, taup -> cr (tau -> 1/S scaled by c_n N_k), so with n1', p1', n_i'^2 the region's
    # (BGN-substituted) reference densities and D_k = cr (n + n1'_k) + 1*(p + p1'_k):
    #   f_k  = (cr n + p1'_k)/D_k                    df/dn = cr (cr n1'_k + p)/D_k^2,  df/dp = -(cr n + p1'_k)/D_k^2
    #   U_k  = N_k c_n (n p - n_i'^2)/D_k            dU/dn = N_k c_n (p D_k - cr (n p - n_i'^2))/D_k^2
    #                                                dU/dp = N_k c_n (n D_k - (n p - n_i'^2))/D_k^2
    #   G_k + R_k = N_k c_n (n p + n_i'^2)/D_k       (the noise pair rate; srh_rate_expr(pair=True))
    # The derivatives are written out ANALYTICALLY: DEVSIM's simplify(diff()) of a ~45-level sum took
    # ~25 s per install (MEASURED, 1-D MOS, 2 x 22 bins) -- the closed forms build in well under 1 s.
    # FD statistics (fd): n1'_k = a_k gamma_n, p1'_k = b_k gamma_p and n_i'^2 = n_ie^2 ~ gamma_n gamma_p for
    # EVERY energy reference, so with ln = FDdLnGammaN = d ln gamma_n/dn, lp = FDdLnGammaP = d ln gamma_p/dp:
    #   df/dn = cr (cr n1'_k + p - (cr n + p1'_k) n1'_k ln)/D_k^2
    #   df/dp = ((cr n + p1'_k)(1 + p1'_k lp) - p1'_k lp D_k)/(-D_k^2)
    #   dU/dn = N_k c_n ((p - n_i'^2 ln) D_k - cr (1 + n1'_k ln)(n p - n_i'^2))/D_k^2
    #   dU/dp = N_k c_n ((n - n_i'^2 lp) D_k - (1 + p1'_k lp)(n p - n_i'^2))/D_k^2
    # (ln = lp = 0 is the fixed-n_ie form above).
    q_val, q_dn, q_dp = [], [], []                             # sheet charge [m^-2], units of +q
    r_val, r_dn, r_dp, r_pair = [], [], [], []                 # surface SRH rate [m^-2 s^-1]
    all_levels = []
    kinetic = []                                               # the dynamic specs' bins (installed below)
    q_fixed = 0.0
    for si, s in enumerate(specs):
        q_fixed += float(s.fixed_charge_C_m2)
        pfx = "it{}_{}_".format(j, si)
        cn = float(s.sigma_n_m2) * float(s.v_th_n_ms)
        cp = float(s.sigma_p_m2) * float(s.v_th_p_ms)
        cr, cnp = pfx + "cr", pfx + "cn"
        ds.set_parameter(device=device, region=semi, name=cr, value=cn / cp)
        ds.set_parameter(device=device, region=semi, name=cnp, value=cn)
        ref = s.energy_reference
        for (e, nk, kind) in s.sheet_levels(kT_eV=V_t):         # bins <= kT at the REGION's T (3.2-8)
            all_levels.append((e, nk, kind))
            n1, p1 = (repr(v) for v in level_reference_densities(e, ref, n_i_m3=n_i, V_t=V_t,
                                                                 N_c_m3=n_c, N_v_m3=n_v))
            if s.dynamic:
                kinetic.append((n1, p1, repr(float(nk)), kind, cr, cnp, ref))
                continue
            n1e, p1e, ni2 = srh_reference_densities(n1, p1, bgn=bgn, reference=ref, **fkw)
            d = _srh_denominator(n1e, p1e, "1", cr)
            num = "({cr}*Electrons + 1*{p1})".format(cr=cr, p1=p1e)
            occ = srh_occupancy_expr(n1, p1, "1", cr, bgn=bgn, reference=ref, **fkw)   # == num/d
            nk_s = repr(float(nk))
            if kind in ("acceptor", "donor"):
                # acceptor: -N f ; donor: N (1 - f) -- both carry -N df in the derivatives
                q_val.append(("-{n}*{f}" if kind == "acceptor" else "{n}*(1 - {f})")
                             .format(n=nk_s, f=occ))
                if fd:
                    q_dn.append("-{n}*{cr}*({cr}*{n1} + Holes - {num}*{n1}*FDdLnGammaN)/({d}^2)".format(
                        n=nk_s, cr=cr, n1=n1e, num=num, d=d))
                    q_dp.append("{n}*({num}*(1 + {p1}*FDdLnGammaP) - {p1}*FDdLnGammaP*{d})/({d}^2)".format(
                        n=nk_s, num=num, p1=p1e, d=d))
                else:
                    q_dn.append("-{n}*{cr}*({cr}*{n1} + Holes)/({d}^2)".format(n=nk_s, cr=cr, n1=n1e, d=d))
                    q_dp.append("{n}*{num}/({d}^2)".format(n=nk_s, num=num, d=d))
            if s.recombination:
                k = "{}*{}".format(nk_s, cnp)
                r_val.append("{k}*{u}".format(k=k, u=srh_rate_expr(n1, p1, "1", cr, bgn=bgn,
                                                                     reference=ref, **fkw)))
                r_pair.append("{k}*{u}".format(k=k, u=srh_rate_expr(n1, p1, "1", cr, bgn=bgn,
                                                                      pair=True, reference=ref, **fkw)))
                if fd:
                    r_dn.append("{k}*((Holes - {ni2}*FDdLnGammaN)*{d} - {cr}*(1 + {n1}*FDdLnGammaN)*"
                                "(Electrons*Holes - {ni2}))/({d}^2)".format(k=k, d=d, cr=cr, ni2=ni2, n1=n1e))
                    r_dp.append("{k}*((Electrons - {ni2}*FDdLnGammaP)*{d} - (1 + {p1}*FDdLnGammaP)*"
                                "(Electrons*Holes - {ni2}))/({d}^2)".format(k=k, d=d, ni2=ni2, p1=p1e))
                else:
                    r_dn.append("{k}*(Holes*{d} - {cr}*(Electrons*Holes - {ni2}))/({d}^2)"
                                .format(k=k, d=d, cr=cr, ni2=ni2))
                    r_dp.append("{k}*(Electrons*{d} - (Electrons*Holes - {ni2}))/({d}^2)"
                                .format(k=k, d=d, ni2=ni2))
    if q_fixed != 0.0:
        q_val.append(repr(q_fixed / Q_E))

    def _sum(terms):
        return "(" + " + ".join(terms) + ")" if terms else "0"

    if q_val:
        sheet = _sum(q_val)
        ds.node_model(device=device, region=semi, name="ItCharge{}".format(j),
                      equation="ElectronCharge*{}".format(sheet))
        _BP.add_charge_term(device, semi, "it{}".format(j), "{}*{}".format(a_name, sheet),
                            derivs={"Electrons": "{}*{}".format(a_name, _sum(q_dn)),
                                    "Holes": "{}*{}".format(a_name, _sum(q_dp))})
    if r_val:
        rate = _sum(r_val)
        ds.node_model(device=device, region=semi, name="ItRecomb{}".format(j), equation=rate)
        _BP.add_recombination_term(device, semi, "it{}".format(j), "{}*{}".format(a_name, rate),
                                   pair="{}*{}".format(a_name, _sum(r_pair)),
                                   derivs={"Electrons": "{}*{}".format(a_name, _sum(r_dn)),
                                           "Holes": "{}*{}".format(a_name, _sum(r_dp))})
    out = {"region": semi, "insulator": other, "index": j, "levels": all_levels, "area": area}
    if kinetic:
        out["kinetic"] = _install_kinetic_bins(device, semi, j, a_name, weight, kinetic, bgn, q_val, fd=fd)
    return out


def _install_kinetic_bins(device, semi, j, a_name, weight, kinetic, bgn, q_val, fd=False):
    """The masked region-wide variables, rows, single-carrier terms and charge term of the kinetic
    interface bins of install j (see install_interface_traps). Returns the variables in bin order."""
    from dynameta.carriers import physics_bipolar_dd as _BP
    mask = "ItOffMask{}".format(j)
    ds.node_solution(device=device, region=semi, name=mask)
    ds.set_node_values(device=device, region=semi, name=mask,
                       values=[0.0 if v > 0.0 else 1.0 for v in weight])
    kq, kdq, made = [], {}, []
    for b, (n1, p1, nk, kind, cr, cnp, ref) in enumerate(kinetic):
        name = "it{}_{}".format(j, b)
        var, eq = "ItW{}_{}".format(j, b), "ItEquation{}_{}".format(j, b)
        f, g = "ItOccupancy{}_{}".format(j, b), "ItVacancy{}_{}".format(j, b)
        seed = "ItWSS{}_{}".format(j, b)
        ds.node_model(device=device, region=semi, name=seed, equation="(1 - {m})*{w} + {m}".format(
            m=mask, w=srh_occupancy_odds_expr(n1, p1, "1", cr, bgn=bgn, reference=ref, fd=fd)))
        ds.node_solution(device=device, region=semi, name=var)       # FIRST: the terms need it
        ds.set_node_values(device=device, region=semi, name=var, values=_initial_odds(device, semi, seed, fd))
        _occupancy_models(device, semi, var, f, g, n1, p1, "1", cr, bgn=bgn, reference=ref, fd=fd)
        k = "{}*{}*{}".format(a_name, nk, cnp)                       # ItArea N_b c_n (tau -> 1/S)
        rn, rp = srh_capture_exprs(n1, p1, "1", cr, f=f, one_minus_f=g, bgn=bgn, reference=ref, fd=fd)
        pn, pp = srh_capture_exprs(n1, p1, "1", cr, f=f, one_minus_f=g, bgn=bgn, reference=ref, pair=True,
                                   fd=fd)
        for carrier, r, pr, c, mech in (("electron", rn, pn, +1.0, "trap_n"), ("hole", rp, pp, -1.0, "trap_p")):
            _BP.add_recombination_term(device, semi, name, "{}*{}".format(k, r), carrier=carrier, wrt=(var,),
                                       pair="{}*{}".format(k, pr), partner=(eq, c), mechanism=mech)
        _trap_row(device, semi, var, eq, "ItNode{}_{}".format(j, b), "ItTrapCharge{}_{}".format(j, b),
                  name, nk, seed, area=a_name, mask=mask)
        if kind in ("acceptor", "donor"):
            sgn, occm = ("-", f) if kind == "acceptor" else ("", g)
            kq.append("{}{}*{}".format(sgn, nk, occm))
            for v in ("Electrons", "Holes", var):
                kdq.setdefault(v, []).append("{}{}*{}:{}".format(sgn, nk, occm, v))
        made.append(var)
    if kq:
        _BP.add_charge_term(device, semi, "itk{}".format(j), "{}*({})".format(a_name, " + ".join(kq)),
                            wrt=tuple(v for v in kdq if v not in ("Electrons", "Holes")),
                            derivs={v: "{}*({})".format(a_name, " + ".join(e)) for v, e in kdq.items()})
        ds.node_model(device=device, region=semi, name="ItCharge{}".format(j),
                      equation="ElectronCharge*({})".format(" + ".join(list(q_val) + kq)))
    return made
