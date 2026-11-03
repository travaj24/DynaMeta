"""
Trap physics for the bipolar drift-diffusion regions (physics_bipolar_dd): bulk SRH levels at an
arbitrary trap energy E_t, several discrete levels each with its own capture parameters and density,
the trapped charge those levels carry into Poisson, and interface traps (D_it) at a
semiconductor/insulator interface -- their sheet charge Q_it(psi_s) and their surface SRH
recombination. Quasi-static (steady-state) occupancy throughout. OPT-IN: nothing here runs unless a
caller passes traps; the defaults reproduce the legacy midgap SRH model character-for-character.

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
   N_it,k = D_it(E_k) dE_k [m^-2] (midpoint bins of width <= kT, edges aligned to the neutral level;
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
  * Quasi-static occupancy only: the trapped charge is an instantaneous function of (n, p). A DC sweep
    (the LF C-V) is exact; a transient or ssac solve is NOT charge-conserving with CHARGED traps (the
    capture current that fills a trap is not in the continuity time term). Trap kinetics are a follow-up.
  * Ohmic contacts (v0.14 integration): with CHARGED bulk levels the contacts, the seed and the 3-D gate
    frame pin the TRAP-AWARE charge-neutral root (bipolar_options.equilibrium_fields with the levels'
    equilibrium Fermi occupancy, n1 = n_ie exp(E_t/kT)) -- the same rule as BGN / incomplete
    ionization. Interface traps do not enter it (a sheet charge, not a bulk neutrality term).
  * Bandgap narrowing (D1b): every rate, occupancy and pair rate here takes the region's n_ie
    substitution through srh_reference_densities, so a level's E_t is measured from the NARROWED
    effective intrinsic level; with an asymmetric band-edge split (conduction_fraction != 1/2) that is a
    modelling choice (see docs/audit/2026-11-03-devsim-phase-integration.md).
  * Interface traps need a bipolar (n AND p) semiconductor side -- the SRH occupancy uses both.
    A semiconductor/semiconductor interface raises (heterojunctions are D1a's).
  * On the devsim_3d gmsh mesh (not Delaunay: sum NodeVolume = 1.20 x the semiconductor volume, 1.29 x
    the oxide's) a geometric sheet charge acts at ~0.8 of its 1-D effect -- a PRE-EXISTING 3-D
    discretization defect, not the weighting here (validation/traps_oracles.py --3d, gate G2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dynameta.constants import Q_E, V_T
from dynameta.carriers.eq_registry import ds

_KINDS = ("acceptor", "donor", "neutral")
_CHARGE_MODES = ("split", "acceptor", "donor", "none")


# ------------------------------------------------------------------------------------------------
# Shared expression kernels (pure strings -- no DEVSIM needed; ONE implementation for the legacy
# midgap channel, every bulk level and every interface sheet level, with or without bandgap
# narrowing, for the net rate, the occupancy and the noise pair rate G + R).
# ------------------------------------------------------------------------------------------------
def srh_reference_densities(n1: str = "n1", p1: str = "p1", *, bgn: bool = False
                            ) -> Tuple[str, str, str]:
    """(n1', p1', n_i'^2) -- THE single place the bandgap-narrowing substitution of D1b (audit R27)
    enters SRH: with BGN on, n_i^2 -> n_ie^2 and n1 -> n1 n_ie/n_i, p1 -> p1 n_ie/n_i (so
    n1' p1' = n_ie^2 and a level keeps its energy relative to the NARROWED effective intrinsic level:
    n1' = n_ie exp((E_t - E_i)/kT)). `n_ie` is the region's effective-intrinsic-density field
    (bipolar_options). Without BGN the names are returned unchanged."""
    if not bgn:
        return n1, p1, "n_i^2"
    return "{}*n_ie/n_i".format(n1), "{}*n_ie/n_i".format(p1), "n_ie^2"


def _srh_denominator(n1: str, p1: str, taun: str, taup: str) -> str:
    return "({tp}*(Electrons + {n1}) + {tn}*(Holes + {p1}))".format(n1=n1, p1=p1, tn=taun, tp=taup)


def srh_rate_expr(n1: str = "n1", p1: str = "p1", taun: str = "taun", taup: str = "taup", *,
                  bgn: bool = False, pair: bool = False) -> str:
    """The Shockley-Read-Hall net rate U = (n p - n_i^2)/(taup (n + n1) + taun (p + p1)) as a DEVSIM
    expression in the solution variables Electrons/Holes. The arguments are parameter names (or
    parenthesized sub-expressions). The DEFAULTS reproduce the legacy physics_bipolar_dd USRH string
    CHARACTER-FOR-CHARACTER (tests/test_traps.py pins it). For a sheet level pass taun -> 1/S_n,
    taup -> 1/S_p and the result is the surface rate [m^-2 s^-1].

    bgn=True applies the bandgap-narrowing substitution (srh_reference_densities); with the default
    names it is bipolar_options' former srh_expr(True) character-for-character.
    pair=True returns the PAIR rate G + R = (n p + n_i^2)/(...) instead of the net rate R - G: the
    fast-trap detailed-balance split R = n p/D, G = n_i^2/D that the noise module's GR source
    2 (G + R) needs (noise_analysis; one kernel for both)."""
    n1e, p1e, ni2 = srh_reference_densities(n1, p1, bgn=bgn)
    return "(Electrons*Holes {} {})/{}".format("+" if pair else "-", ni2,
                                               _srh_denominator(n1e, p1e, taun, taup))


def srh_occupancy_expr(n1: str = "n1", p1: str = "p1", taun: str = "taun",
                       taup: str = "taup", *, bgn: bool = False) -> str:
    """Steady-state electron occupancy f_t = (taup n + taun p1)/(taup (n + n1) + taun (p + p1)) of an
    SRH level (Shockley & Read 1952) -- the SAME denominator as srh_rate_expr. Equals the Fermi
    function of E_t at equilibrium PROVIDED n1 p1 equals the equilibrium n p -- which is why it takes
    the same `bgn` substitution as the rate (with BGN on n p = n_ie^2 at equilibrium; an occupancy
    left on n_i would depend on taun/taup at equilibrium, i.e. not be a Fermi function)."""
    n1e, p1e, _ = srh_reference_densities(n1, p1, bgn=bgn)
    return "({tp}*Electrons + {tn}*{p1})/{d}".format(tp=taup, tn=taun, p1=p1e,
                                                     d=_srh_denominator(n1e, p1e, taun, taup))


def n1_p1(energy_eV: float, n_i_m3: float, V_t: float = V_T) -> Tuple[float, float]:
    """(n1, p1) = n_i exp(+-E/kT) for a level at E = E_t - E_i [eV], kT/q = V_t (the region's thermal
    voltage; V_T at T_REF by default, BipolarOptions.T_K otherwise). E = 0 returns (n_i, n_i)
    EXACTLY (exp(0.0) == 1.0), which is what keeps the legacy midgap channel bit-identical."""
    e = float(energy_eV) / float(V_t)
    return float(n_i_m3) * math.exp(e), float(n_i_m3) * math.exp(-e)


# ------------------------------------------------------------------------------------------------
# Specs
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrapLevel:
    """One discrete bulk trap level (Shockley-Read-Hall).

    energy_eV    : E_t - E_i [eV] (+ toward E_c).
    density_m3   : N_t [m^-3] -- needed for a CHARGED level and to turn cross sections into lifetimes.
    kind         : "acceptor" (charge -q N_t f_t), "donor" (+q N_t (1 - f_t)) or "neutral"
                   (recombination only -- the legacy N_t -> 0 SRH assumption).
    tau_n0_s, tau_p0_s : capture lifetimes [s]; OR give sigma_n_m2, sigma_p_m2 [m^2] with v_th
                   [m/s] and density_m3, and tau_x0 = 1/(sigma_x v_th,x N_t). Explicit taus win.
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

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError("TrapLevel.kind must be one of {}; got {!r}".format(_KINDS, self.kind))
        if not math.isfinite(float(self.energy_eV)):
            raise ValueError("TrapLevel.energy_eV must be finite")
        if self.kind != "neutral" and not (float(self.density_m3) > 0.0):
            raise ValueError("a charged ({}) TrapLevel needs density_m3 > 0".format(self.kind))
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
    bin_eV       : maximum quadrature bin width (default kT): the continuous D_it is integrated by the
                   midpoint rule on bins aligned to the band ends and the neutral level.
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
    neutral_level_eV: float = 0.0
    sigma_n_m2: float = 1.0e-20
    sigma_p_m2: float = 1.0e-20
    v_th_n_ms: float = 1.0e5
    v_th_p_ms: float = 1.0e5
    recombination: bool = True
    fixed_charge_C_m2: float = 0.0
    bin_eV: Optional[float] = None

    def __post_init__(self):
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
        lo = self.e_lo_eV if self.e_lo_eV is not None else (
            -0.5 * float(self.band_gap_eV) if self.band_gap_eV is not None else None)
        hi = self.e_hi_eV if self.e_hi_eV is not None else (
            0.5 * float(self.band_gap_eV) if self.band_gap_eV is not None else None)
        return lo, hi

    def sheet_levels(self) -> List[Tuple[float, float, str]]:
        """Discretize D_it into sheet levels [(E_k - E_i [eV], N_it,k [m^-2], kind_k)], kind in
        {acceptor, donor, neutral}. Bins never straddle the neutral level (charge='split'), so every
        bin has ONE charge type. Pure numpy -- usable devsim-free."""
        dmax = float(self.bin_eV) if self.bin_eV is not None else V_T
        e0 = float(self.neutral_level_eV)

        def kind_of(e):
            if self.charge == "none":
                return "neutral"
            if self.charge in ("acceptor", "donor"):
                return self.charge
            return "acceptor" if e > e0 else "donor"

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
                       V_t: float = V_T, bgn: bool = False) -> None:
    """Install discrete bulk SRH levels on a bipolar region set up by
    physics_bipolar_dd.setup_bipolar_region (which calls this with traps=...).

    Per level k: parameters trap<k>_n1/_p1/_taun/_taup/_Nt (n1, p1 = n_i exp(+-E_t/kT) at the region's
    V_t), the diagnostic occupancy TrapOccupancy<k>, and its rate as ONE recombination term
    RTerm_trap<k> of the shared additive mechanism (physics_bipolar_dd.add_recombination_term) WITH
    its pair rate G + R (so the noise GR source sees it -- noise_analysis). The charged levels' trapped
    charge goes in as the single charge term QTerm_bulk_traps. Rate, occupancy and pair rate all come
    from the ONE SRH kernel (srh_rate_expr / srh_occupancy_expr) with the region's BGN substitution
    (bgn=True: n_i^2 -> n_ie^2, n1/p1 scaled by n_ie/n_i)."""
    from dynameta.carriers import physics_bipolar_dd as _BP   # lazy: BP imports this module

    charges = []
    for k, t in enumerate(traps):
        if not isinstance(t, TrapLevel):
            raise TypeError("traps must be TrapLevel instances; got {!r}".format(type(t).__name__))
        p = "trap{}_".format(k)
        n1, p1 = n1_p1(t.energy_eV, n_i_m3, V_t)
        tn, tp = t.lifetimes_s()
        for nm, v in (("n1", n1), ("p1", p1), ("taun", tn), ("taup", tp),
                      ("Nt", float(t.density_m3))):
            ds.set_parameter(device=device, region=region, name=p + nm, value=v)
        args = (p + "n1", p + "p1", p + "taun", p + "taup")
        occ = srh_occupancy_expr(*args, bgn=bgn)
        ds.node_model(device=device, region=region, name="TrapOccupancy{}".format(k), equation=occ)
        _BP.add_recombination_term(device, region, "trap{}".format(k), srh_rate_expr(*args, bgn=bgn),
                                   pair=srh_rate_expr(*args, bgn=bgn, pair=True))
        if t.kind == "acceptor":
            charges.append("-{}Nt*{}".format(p, occ))
        elif t.kind == "donor":
            charges.append("{}Nt*(1 - {})".format(p, occ))
    if charges:
        _BP.add_charge_term(device, region, "bulk_traps", " + ".join(charges))


def equilibrium_trap_levels(traps) -> Tuple[Tuple[float, float, str], ...]:
    """(energy_eV, N_t [m^-3], kind) of the CHARGED bulk levels -- what the charge-neutral equilibrium
    of the ohmic contacts / seed / 3-D gate frame needs (bipolar_options.equilibrium_fields). Neutral
    (recombination-only) levels carry no charge and are left out, so a trap set without charged levels
    returns () and leaves the legacy contact rule in place."""
    out = []
    for t in (traps or ()):
        if not isinstance(t, TrapLevel):
            raise TypeError("traps must be TrapLevel instances; got {!r}".format(type(t).__name__))
        if t.kind != "neutral":
            out.append((float(t.energy_eV), float(t.density_m3), t.kind))
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


def interface_node_area(device: str, region: str, interface: str) -> np.ndarray:
    """Interface area [m^2 in 3-D, m in 2-D, 1 in 1-D] OWNED by each node of `region` on `interface`
    (0 elsewhere), from DEVSIM's interface element list: 1-D a point (1), 2-D half of each interface
    edge, 3-D a third of each interface triangle. Cartesian coordinates only."""
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
            a = 0.5 * float(np.linalg.norm(np.cross(xyz[e[1]] - xyz[e[0]], xyz[e[2]] - xyz[e[0]])))
            area[e] += a / 3.0
        else:                                                  # pragma: no cover - not a DEVSIM surface
            raise ValueError("unexpected interface element with {} nodes".format(len(e)))
    if not np.any(area > 0.0):
        raise ValueError("interface {!r} has no nodes on region {!r}".format(interface, region))
    return area


def install_interface_traps(device: str, interface: str, traps) -> dict:
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
    Returns {'region', 'insulator', 'index', 'levels', 'area'}."""
    from dynameta.carriers import physics_bipolar_dd as _BP   # lazy: BP imports this module

    specs = [traps] if isinstance(traps, InterfaceTraps) else list(traps)
    if not specs or not all(isinstance(s, InterfaceTraps) for s in specs):
        raise TypeError("install_interface_traps: traps must be an InterfaceTraps or a sequence of them")
    semi, other = _semiconductor_side(device, interface)
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
    # Per sheet level k (one spec: shared capture ratio cr = c_n/c_p, c = sigma v_th): the SRH kernel
    # with taun -> 1, taup -> cr (tau -> 1/S scaled by c_n N_k), so with n1', p1', n_i'^2 the region's
    # (BGN-substituted) reference densities and D_k = cr (n + n1'_k) + 1*(p + p1'_k):
    #   f_k  = (cr n + p1'_k)/D_k                    df/dn = cr (cr n1'_k + p)/D_k^2,  df/dp = -(cr n + p1'_k)/D_k^2
    #   U_k  = N_k c_n (n p - n_i'^2)/D_k            dU/dn = N_k c_n (p D_k - cr (n p - n_i'^2))/D_k^2
    #                                                dU/dp = N_k c_n (n D_k - (n p - n_i'^2))/D_k^2
    #   G_k + R_k = N_k c_n (n p + n_i'^2)/D_k       (the noise pair rate; srh_rate_expr(pair=True))
    # The derivatives are written out ANALYTICALLY: DEVSIM's simplify(diff()) of a ~45-level sum took
    # ~25 s per install (MEASURED, 1-D MOS, 2 x 22 bins) -- the closed forms build in well under 1 s.
    q_val, q_dn, q_dp = [], [], []                             # sheet charge [m^-2], units of +q
    r_val, r_dn, r_dp, r_pair = [], [], [], []                 # surface SRH rate [m^-2 s^-1]
    all_levels = []
    q_fixed = 0.0
    for si, s in enumerate(specs):
        q_fixed += float(s.fixed_charge_C_m2)
        pfx = "it{}_{}_".format(j, si)
        cn = float(s.sigma_n_m2) * float(s.v_th_n_ms)
        cp = float(s.sigma_p_m2) * float(s.v_th_p_ms)
        cr, cnp = pfx + "cr", pfx + "cn"
        ds.set_parameter(device=device, region=semi, name=cr, value=cn / cp)
        ds.set_parameter(device=device, region=semi, name=cnp, value=cn)
        for (e, nk, kind) in s.sheet_levels():
            all_levels.append((e, nk, kind))
            n1, p1 = (repr(v) for v in n1_p1(e, n_i, V_t))
            n1e, p1e, ni2 = srh_reference_densities(n1, p1, bgn=bgn)
            d = _srh_denominator(n1e, p1e, "1", cr)
            num = "({cr}*Electrons + 1*{p1})".format(cr=cr, p1=p1e)
            occ = srh_occupancy_expr(n1, p1, "1", cr, bgn=bgn)          # == num/d
            nk_s = repr(float(nk))
            if kind in ("acceptor", "donor"):
                # acceptor: -N f ; donor: N (1 - f) -- both carry -N df in the derivatives
                q_val.append(("-{n}*{f}" if kind == "acceptor" else "{n}*(1 - {f})")
                             .format(n=nk_s, f=occ))
                q_dn.append("-{n}*{cr}*({cr}*{n1} + Holes)/({d}^2)".format(n=nk_s, cr=cr, n1=n1e, d=d))
                q_dp.append("{n}*{num}/({d}^2)".format(n=nk_s, num=num, d=d))
            if s.recombination:
                k = "{}*{}".format(nk_s, cnp)
                r_val.append("{k}*{u}".format(k=k, u=srh_rate_expr(n1, p1, "1", cr, bgn=bgn)))
                r_pair.append("{k}*{u}".format(k=k, u=srh_rate_expr(n1, p1, "1", cr, bgn=bgn,
                                                                      pair=True)))
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
    return {"region": semi, "insulator": other, "index": j, "levels": all_levels, "area": area}
