"""
Bipolar drift-diffusion (electrons + holes + Shockley-Read-Hall recombination),
DEVSIM-native, in SI units. Opt-in extension of the electrons-only
physics_drift_diffusion.py -- mirrors its SI conventions and the FERMI-DIRAC
diffusion-enhancement (FD g-factor), and reimplements the holes / SRH / bipolar
Poisson / charge-neutral contacts in SI (the bundled simple_physics.py is CGS).

Solution variables: Potential, Electrons, Holes (three coupled fields).

DEVSIM sign convention (residual = q*divergence form), per docs/implementation_notes.md:
  Jn = +q mu_n EdgeInvLength V_t [ Electrons@n1 B(-vdiff) - Electrons@n0 B(vdiff) ]   (electrons)
  Jp = -q mu_p EdgeInvLength V_t [ Holes@n1 B(vdiff) - Holes@n0 B(-vdiff) ]           (holes; q->-q
                                   AND the B(+-vdiff) roles swapped)
  -- built by the ONE SG kernel (sg_flux, 2026-11-04): B(vdiff) and B(-vdiff) both, combined
  sign-selected so the Bernoulli identity B(x) + x = B(-x) is only used for x >= 0. DEVSIM's bundled
  kahan3(c@n1 B, c@n1 vdiff, -c@n0 B) spelling (used here until then) cancels for vdiff << 0.
  Poisson node charge : PotentialNodeCharge = -q (Holes - Electrons + NetDoping)
  SRH source          : USRH = (n p - n_i^2)/(taup (n + n1) + taun (p + p1))
                        ElectronGeneration = -q USRH  (into ElectronContinuityEquation)
                        HoleGeneration     = +q USRH  (into HoleContinuityEquation)
                        (+ sum of the opt-in additive terms RTerm_*: see add_recombination_term)

FD enhancement (accurate rational fit of the generalized-Einstein ratio g=F_1/2/F_-1/2
in pow()s of the solution variable -- the only form DEVSIM differentiates without hanging;
see physics_drift_diffusion.py for the fit + coefficients):
  g(x) = 1 + (a x + c x^(4/3))/(1 + b x^(1/3) + d x^(2/3)),  x = c/N_dos
  replace vdiff/V_t by (vdiff/V_t)/g and scale the current by g.
For a non-degenerate Si diode (c/N_dos << 1) g -> 1 and the current reduces EXACTLY to
standard Boltzmann Scharfetter-Gummel; for ITO it is the FD softening (~1.1% peak, <0.5%
over eta>=10, valid to eta~32; vs the old degenerate-asymptote form's 6-35% error; audit F1,
bounds re-measured DD-1/DD-2). The hole g-factor uses its OWN valence DOS N_dos_p (~N_v),
passed via setup_bipolar_region(n_dos_p_m3=...); it defaults to N_dos (N_v == N_c) for the
non-degenerate case, so a degenerate p-region is now correct too (audit DD-5).

Charge-neutral ohmic contact (recipe): pin
  n0 = 1/2 ( NetDoping + sqrt(NetDoping^2 + 4 n_i^2) )   (majority on n-side),
  p0 = n_i^2 / n0,  with the symmetric swap on the p-side, and the Potential
contact carries the built-in offset + ifelse(N>0, -V_t log(n0/n_i), V_t log(p0/n_i)).

FERMI-DIRAC CONTACT OFFSET (audit C-10). That log() offset is the BOLTZMANN built-in potential,
and it used to be pinned even with fd_enhancement=True -- i.e. the boundary condition contradicted
the interior model, by 0.17 V per degenerate contact at eta = 10. Setting the SG current to zero,
  Jn = 0  =>  n1/n0 = exp(-vdiff/g)  =>  dpsi = g V_t dln n,
so the offset the FD interior actually implies is V_t [ ln(n0/n_i) + Delta(n0/N_dos) ] with
  Delta(u) = int_0^u (g(x) - 1) dx/x ,
evaluated with the SAME rational g the edge models use (carriers.einstein) -- so the contact and
the interior agree by construction and the Boltzmann limit (g -> 1, Delta -> 0) is EXACT.
fd_enhancement=False keeps the pure Boltzmann expressions byte-identically.

ACCURACY OF Delta, IN TWO FRAMES (re-measured 2026-07-26; the earlier "0.85 % at eta = 10, <0.5 %
for eta >= 15" was measured only in the AH frame and does NOT hold against exact quadrature).
Comparing Delta(u) against the exact Fermi-Dirac offset eta - ln(F_1/2(eta)):
  * AH frame (u = F12_aymerich_humet(eta), the fit tests/test_devsim_smoke.py uses):
      0.85 % at eta = 10, and < 0.5 % for eta >= 15 -- the claim as originally written.
  * EXACT frame (u = F_1/2(eta) by quadrature, the physical oracle):
      eta = 10: +0.78 %   15: +0.60 %   20: +0.55 %   25: +0.50 %   32 (the ceiling): +0.40 %
    and, in the MODERATELY degenerate range (re-measured 2026-11-06 against scipy quadrature of
    F_1/2; the earlier "under 1 % everywhere degeneracy matters" was wrong there):
      eta = -4: -0.21 %   -3: +1.03 %   -2: +2.30 %   -1: +3.35 %   0: +3.86 %   0.5: +3.88 %
      1: +3.76 %   2: +3.27 %   3: +2.68 %   4: +2.15 %   5: +1.72 %   7: +1.16 %   8: +0.99 %
    i.e. the relative error is ABOVE 1 % from eta ~ -3 to ~ 8 and peaks at +3.9 % near eta ~ 0.3. In
    absolute terms (what enters exp(-Delta) and the built-in potential, in V_t units): 1.8e-4 at
    eta = -3, 3.9e-3 at -1 (Si:P 1e19 cm^-3 sits at eta = -0.93: 4.2e-3), 1.0e-2 at 0, 3.1e-2 at 2
    (Si:P 1e20 cm^-3, eta = 2.43: 3.6e-2), 4.9e-2 at 5, 5.3e-2 at 10. "< 0.5 % for
    eta >= 15" is an artifact of the AH frame; against exact quadrature it does not reach 0.5 % until
    eta ~ 25. A gate that touches eta in [-3, 8] must budget for this error explicitly (the v0.15 Si
    bundle gates do: tests/test_si_bundles_devsim.py).
The sign matters for the ceiling warning: Delta OVERSHOOTS the exact offset (positive error) all
the way to eta ~ 50, where the error crosses zero (+0.009 % at eta = 50, -0.040 % at 52); only past
that does it undershoot (-0.24 % at eta = 60, -1.20 % at 100). The g-FACTOR is the one that crosses
over early -- +0.009 % at eta = 30, -0.041 % at 31, -1.07 % at 50 -- so g undershoots past eta ~ 30,
which is where _FD_U_MAX comes from.

WHO CHECKS THAT CEILING -- two tests, because a doping profile and a converged solution are different
things. (1) SETUP time, `_fd_contact_shift`: the charge-neutral majority density NetDoping implies.
(2) POST-SOLVE, `warn_if_solution_exceeds_fd_ceiling` (audit C10-d), which `dc_solve.solve_dc` runs on
every converged DC/Gummel solve: max(Electrons)/N_dos and max(Holes)/N_dos_p from the SOLUTION. Only
(2) can see a gated accumulation cap, whose body doping sits below the ceiling while the gate drives
the channel far past it -- the devsim_3d `Stacked3DSpec` case exactly.

OPT-IN PHYSICS BEYOND SRH (audit R26/R27, v0.14.0): setup_bipolar_region(options=BipolarOptions(...))
adds Auger / radiative recombination, apparent bandgap narrowing (band-edge quasi-fields in both SG
currents + n_ie in every recombination term and contact), incomplete dopant ionization (N_D+ / N_A- in
Poisson), a lattice temperature for V_t, and a uniform generation G. The formulation, citations and
measured oracles live in carriers/bipolar_options.py and
docs/audit/2026-11-03-devsim-recombination-bgn-ionization.md. options=None is byte-identical.

Staged solve: see solve_bipolar_diode in validation/bipolar_diode.py:
  (1) potential-only pre-solve (freeze carriers); (2) seed Electrons/Holes from the
  Boltzmann equilibrium node models; (3) coupled 3-variable Newton; (4) bias ramp.
  variable_update: Potential "log_damp", Electrons/Holes "positive".
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import numpy as np
from dynameta.carriers.devsim_env import import_devsim

ds = import_devsim()      # D0: devsim via its ONE import home (BLAS pinned before the first import)

from dynameta.carriers.physics_equilibrium import (
    Q_E, EPS0, V_T, _poisson_edge_models, require_positive, invert_F12, psi_offset_term)
from dynameta.carriers import eq_registry as _R
from dynameta.carriers import sg_flux as _SG
from dynameta.carriers.eq_registry import (edge_with_derivs as _edge_with_derivs,
                                           node_with_derivs as _node_with_derivs)
from dynameta.carriers.einstein import g_einstein, g_expr_devsim
from dynameta.carriers import physics_traps as _PT
from dynameta.warning_types import UnreliableResultWarning

if TYPE_CHECKING:                                              # pragma: no cover - typing only
    from dynameta.carriers.bipolar_options import BipolarOptions

# --- audit C-10: the Fermi-Dirac degeneracy correction to the ohmic built-in potential ----------
_FD_GL_N = 96                                                  # Gauss-Legendre nodes for Delta(u)
_FD_GL_X, _FD_GL_W = np.polynomial.legendre.leggauss(_FD_GL_N)
_FD_U_MAX = 136.0          # F_1/2(eta = 32): the rational g fit's validated ceiling (einstein.py)
# audit C-10 (wave-5 residual): peak-memory budget for the (n_u, _FD_GL_N) quadrature block below.
# The rule is evaluated on the UNIQUE u values, which is normally one or two, but a GRADED doping
# profile makes every node unique: a 1e6-unique-value case allocated 5.5 GB (several (1e6, 96)
# float64 temporaries at once inside g_einstein) and could take the process down. Chunking the
# unique axis is EXACTLY arithmetic-preserving -- each row's 96-point sum is independent of every
# other row -- so results are bit-identical to the unchunked form at any chunk size.
# MEASURED (tracemalloc, 2026-07-26): u = linspace(1e-6, 130, 1e6) peaks at 113 MB chunked; the
# unchunked form peaks at 1047 MB for only 2e5 values (10.4x the chunked peak at the same n), i.e.
# ~5.1 GB extrapolated to 1e6 -- and the two results are byte-identical.
_FD_CHUNK_BYTES = 12 * 1024 * 1024                             # ~113 MB peak with the temporaries
_FD_CHUNK = max(1, _FD_CHUNK_BYTES // (8 * _FD_GL_N))          # rows per block (16384 at 96 nodes)


def fd_potential_shift(u):
    """Degeneracy correction Delta(u) = int_0^u (g(x) - 1) dx/x to the built-in potential, in units
    of V_t; u = n0/N_dos (the majority carrier over ITS band DOS). audit C-10.

    Zero-current equilibrium of the FD-enhanced Scharfetter-Gummel current gives dpsi = g V_t dln n
    (see the module docstring), so the ohmic contact's built-in offset is
    V_t [ln(n0/n_i) + Delta(n0/N_dos)] instead of the Boltzmann V_t ln(n0/n_i). Integrating in
    s = x^(1/3) (the fit's natural variable) makes the integrand 3 (g(s^3) - 1)/s, which is
    ANALYTIC and vanishes like 3*GA*s^2 at the origin, so a fixed Gauss-Legendre rule converges
    spectrally and Delta(0) = 0 EXACTLY -- the Boltzmann off-switch. Broadcasts over arrays.

    The (n_u, 96) quadrature block is evaluated in `_FD_CHUNK`-row slices so peak memory stays ~113 MB
    however many values are passed (audit C-10 residual; a 1e6-value graded profile allocated ~5 GB).
    Rows are independent, so the chunked result is BIT-IDENTICAL to the unchunked one."""
    uu = np.asarray(u, dtype=np.float64)
    if np.any(uu < 0.0):
        raise ValueError("fd_potential_shift: u = n/N_dos must be >= 0")
    flat = np.power(uu, 1.0 / 3.0).reshape(-1)                  # upper limit in s, one row per value
    out = np.empty(flat.shape, dtype=np.float64)
    for lo in range(0, flat.size, _FD_CHUNK):
        hi = flat[lo:lo + _FD_CHUNK]
        s = 0.5 * hi[:, None] * (_FD_GL_X + 1.0)                # nodes on [0, u^(1/3)]
        w = 0.5 * hi[:, None] * _FD_GL_W
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(s > 0.0,
                                 3.0 * (g_einstein(s ** 3) - 1.0) / np.where(s > 0.0, s, 1.0), 0.0)
        out[lo:lo + _FD_CHUNK] = np.sum(w * integrand, axis=-1)
    out = out.reshape(uu.shape)
    return float(out) if out.ndim == 0 else out


def ohmic_built_in_offset_V(net_doping_m3: float, *, n_i_m3: float, n_dos_m3: float,
                            n_dos_p_m3: "float | None" = None,
                            fd_enhancement: bool = True,
                            options: "BipolarOptions | None" = None,
                            traps=None, statistics: "str | None" = None) -> float:
    """The built-in potential [V] the bipolar ohmic contact PINS for a node of SIGNED net doping
    `net_doping_m3`, in the contact's own frame (n-type: +V_t ln(n0/n_i) plus the FD correction).

    THE single source for anything that has to reference another terminal to the ohmic contact's
    frame -- notably devsim_3d's gate offset phi_bi, which re-derived the Boltzmann log() and would
    otherwise re-open the very frame mismatch it exists to close (audit C2-1) as soon as the body is
    degenerate (audit C-10: 0.178 V at eta = 10). fd_enhancement=False returns the plain Boltzmann
    value, matching the contact expression exactly.

    `options` (audit R26/R27): an ACTIVE BipolarOptions routes through the SAME numpy kernel the region
    setup uses (bipolar_options.equilibrium_fields) -- BGN band-edge shift, incomplete-ionization
    neutrality, and V_t at options.T_K -- so the 3-D gate frame follows the contact it references.
    None (the default) or an inactive bundle is the legacy expression unchanged.

    `traps` (v0.14 integration): the region's bulk physics_traps.TrapLevel sequence. CHARGED levels move
    the charge-neutral equilibrium the contact pins (the same kernel, with the trapped charge in the
    neutrality root), so a gate referenced to a trap-compensated body follows its contact too.
    None / no charged level is the trap-free value unchanged.

    `statistics` (v0.15): the region's setup_bipolar_region(statistics=...) -- "fermi_dirac" takes the
    FD detailed-balance root, and a PAIRED BGN set in `options` is checked here by the SAME function
    (bipolar_options.check_bgn_pairing, inside equilibrium_fields) the region setup uses, so the gate
    frame refuses exactly what the body refuses. None (v0.15.0) resolves like the region: the FD root
    with fd_enhancement=True (bipolar_options.resolve_statistics); 'legacy' keeps the v0.14 value."""
    eq_traps = _PT.equilibrium_trap_levels(traps)
    from dynameta.carriers.bipolar_options import resolve_statistics
    label = resolve_statistics(statistics, fd_enhancement)       # v0.15.0: None -> 'fermi_dirac' (FD root)
    explicit = statistics is not None and label != "fermi_dirac_legacy"   # 'legacy' = the v0.14 closed form
    if (options is not None and options.active) or eq_traps or explicit or label == "fermi_dirac":
        from dynameta.carriers.bipolar_options import BipolarOptions, equilibrium_fields
        nd = float(n_dos_m3)
        f = equilibrium_fields(np.array([float(net_doping_m3)]), n_i_m3=float(n_i_m3), N_c_m3=nd,
                               N_v_m3=float(n_dos_p_m3) if n_dos_p_m3 else nd,
                               options=options if options is not None else BipolarOptions(),
                               fd_enhancement=fd_enhancement, traps=eq_traps, statistics=statistics)
        return float(f["psi_eq"][0])
    N = float(net_doping_m3)
    ni = float(n_i_m3)
    root = (N * N + 4.0 * ni * ni) ** 0.5
    n0 = 0.5 * (abs(N) + root)                                 # majority density (CELEC / CHOLE)
    sign = 1.0 if N >= 0.0 else -1.0
    dos = float(n_dos_m3) if N >= 0.0 else float(n_dos_p_m3 if n_dos_p_m3 else n_dos_m3)
    shift = float(fd_potential_shift(n0 / dos)) if fd_enhancement else 0.0
    return sign * V_T * (np.log(n0 / ni) + shift)


def _contact_region(device: str, contact: str):
    """The region a contact belongs to (None if DEVSIM cannot say)."""
    try:
        regions = ds.get_region_list(device=device, contact=contact)
    except Exception:                                          # pragma: no cover - devsim internals
        return None
    return regions[0] if regions else None


def _fd_contact_shift(device: str, contact: str):
    """audit C-10: build (once per region) the node model carrying Delta(u) for the ohmic contact's
    built-in offset, and return its name -- or None when the FD correction does not apply (the
    region was set up with fd_enhancement=False, or is not a bipolar region at all), in which case
    the callers keep the pure-Boltzmann expressions BYTE-IDENTICALLY.

    Delta depends only on NetDoping (through the charge-neutral majority density) and on the band
    DOS parameters, i.e. on no solution variable, so it is a fixed field: computed here in numpy
    and installed with node_solution/set_node_values. That also keeps the contact model's
    derivative literal "1" exactly as before.

    TWO DIFFERENT "return None"s, and only one of them is silent (audit C-10 residual). A region with
    no `fd_enhancement` parameter at all is not a bipolar region -- the caller is setting up some
    other physics and the Boltzmann string is simply what it always was, so that path stays quiet.
    But a region that DOES declare fd_enhancement > 0 and then fails a MATERIAL lookup (n_i / N_dos /
    N_dos_p) would silently revert its contact to Boltzmann and re-open the very 0.17 V mismatch this
    function exists to close, so that path WARNS and names the parameter that could not be read."""
    region = _contact_region(device, contact)
    if region is None:
        return None
    try:
        fd_on = float(ds.get_parameter(device=device, region=region,
                                       name="fd_enhancement")) > 0.0
    except Exception:
        return None                                            # not a bipolar region: leave as-is
    if not fd_on:
        return None                                            # Boltzmann interior -> Boltzmann BC
    _param = "n_i"
    try:
        n_i = float(ds.get_parameter(device=device, region=region, name=_param))
        _param = "N_dos"
        N_c = float(ds.get_parameter(device=device, region=region, name=_param))
        _param = "N_dos_p"
        N_v = float(ds.get_parameter(device=device, region=region, name=_param))
    except Exception as exc:                                   # audit C-10: do NOT revert silently
        warnings.warn(
            "setup_contact_ohmic_bipolar: region {!r} declares fd_enhancement > 0 (Fermi-Dirac "
            "transport) but its material parameter {!r} could not be read ({}: {}), so contact {!r} "
            "falls back to the BOLTZMANN built-in potential while the interior keeps the FD "
            "g-factor -- the exact boundary/interior contradiction audit C-10 closed (0.17 V per "
            "degenerate contact at eta = 10). Set the parameter on the region (setup_bipolar_region "
            "does), or set fd_enhancement=0 to choose Boltzmann deliberately.".format(
                region, _param, type(exc).__name__, str(exc)[:80], contact),
            UnreliableResultWarning, stacklevel=3)
        return None
    name = "FDContactShift"
    if name in ds.get_node_model_list(device=device, region=region):
        return name
    nd = np.asarray(ds.get_node_model_values(device=device, region=region, name="NetDoping"),
                    dtype=np.float64)
    root = np.sqrt(nd ** 2 + 4.0 * n_i ** 2)
    n0 = 1.0 + 0.5 * np.abs(nd + root)                         # == CELEC, evaluated in numpy
    p0 = 1.0 + 0.5 * np.abs(-nd + root)                        # == CHOLE
    u = np.where(nd > 0.0, n0 / N_c, p0 / N_v)                 # the branch the contact expression takes
    # CEILING WARNING. It reads NetDoping ONLY -- i.e. the CHARGE-NEUTRAL majority density at the
    # contact -- so it cannot see degeneracy that the SOLUTION produces: a gated/accumulation device
    # (the MOS-cap builders, an ENZ accumulation layer) can drive n/N_dos far past this ceiling in
    # the channel while the contact's own doping sits comfortably below it. The transport g-factor is
    # what degrades in that case, not this offset. That SOLUTION-level half is now covered by
    # `warn_if_solution_exceeds_fd_ceiling` (audit C10-d), which `dc_solve.solve_dc` runs on every
    # converged DC/Gummel solve; this setup-time test remains the one that fires BEFORE any solve.
    _warn_fd_doping_ceiling(u)
    # Delta is a function of u alone, and a doping profile has few distinct values (usually one or
    # two), so evaluate the quadrature on the UNIQUE ratios: a 1e6-node 3-D body region would
    # otherwise allocate an (nnodes, 96) quadrature array.
    uq, inv = np.unique(u, return_inverse=True)
    shift = np.asarray(fd_potential_shift(uq), dtype=np.float64).reshape(-1)[inv]
    ds.node_solution(device=device, region=region, name=name)
    ds.set_node_values(device=device, region=region, name=name,
                       values=[float(v) for v in shift])
    return name


def _warn_fd_doping_ceiling(u, stacklevel: int = 4) -> None:
    """The setup-time FD-ceiling advisory (audit C-10 / DD-2) on the contact region's charge-neutral majority
    ratio u = n0/N_dos (or p0/N_dos_p): one UnreliableResultWarning past _FD_U_MAX. Shared by the legacy
    contact shift (_fd_contact_shift) and, since v0.15.0, the FD-detailed-balance contact path (the default
    statistics), which pins EquilibriumPotential instead and would otherwise have dropped it."""
    if float(np.max(u)) > _FD_U_MAX:
        warnings.warn(
            "setup_contact_ohmic_bipolar: the contact's majority density reaches n/N_dos = {:.3g} "
            "(eta > 32), beyond the validated ceiling of the rational g fit (carriers.einstein). "
            "Past it the transport g-FACTOR undershoots (its error crosses zero near eta ~ 30: "
            "-0.04 % at eta = 31, -1.1 % at 50), while this contact's built-in OFFSET still "
            "overshoots the exact Fermi-Dirac value out to eta ~ 50 (+0.40 % at the ceiling, "
            "+0.009 % at 50) and only undershoots beyond that (-0.24 % at 60) -- so the two do NOT "
            "err in the same direction (audit C-10 / DD-2, re-measured 2026-07-26). NOTE this test "
            "sees NetDoping only; the SOLUTION-level exceedance a gated/accumulation device produces "
            "is caught after each converged solve instead (audit C10-d, "
            "warn_if_solution_exceeds_fd_ceiling). Supply a larger N_dos, or drop fd_enhancement "
            "and accept the Boltzmann model knowingly.".format(float(np.max(u))),
            UnreliableResultWarning, stacklevel=stacklevel)


def _fd_root_ceiling_check(device: str, contact: str) -> None:
    """The setup-time FD-ceiling advisory for a contact pinning the option-equilibrium fields (statistics=
    'fermi_dirac', the v0.15.0 default): u from the region's pinned charge-neutral densities
    (NeutralElectrons / N_dos on n-type nodes, NeutralHoles / N_dos_p on p-type ones), once per region
    (the region parameter fd_ceiling_checked marks it; set only when the check runs on an FD region)."""
    region = _contact_region(device, contact)
    if region is None:
        return
    try:
        if not float(ds.get_parameter(device=device, region=region, name="fd_enhancement")) > 0.0:
            return
    except Exception:
        return
    if "fd_ceiling_checked" in ds.get_parameter_list(device=device, region=region):
        return
    nd = np.asarray(ds.get_node_model_values(device=device, region=region, name="NetDoping"), np.float64)
    n0 = np.asarray(ds.get_node_model_values(device=device, region=region, name="NeutralElectrons"), np.float64)
    p0 = np.asarray(ds.get_node_model_values(device=device, region=region, name="NeutralHoles"), np.float64)
    nc = float(ds.get_parameter(device=device, region=region, name="N_dos"))
    nv = float(ds.get_parameter(device=device, region=region, name="N_dos_p"))
    u = np.where(nd > 0.0, n0 / nc, p0 / nv)
    if float(np.max(u)) > _FD_U_MAX:
        ds.set_parameter(device=device, region=region, name="fd_ceiling_checked", value=1.0)
        _warn_fd_doping_ceiling(u, stacklevel=5)


# --- audit C10-d: the SOLUTION-level counterpart of the ceiling test above ----------------------
# The rational g fit is anchored at the Boltzmann end and NOT at the degenerate one, so it drifts
# once eta runs past ~32 (u = n/N_dos past _FD_U_MAX). MEASURED 2026-07-26 against exact Gauss
# quadrature of F_1/2 and F_-1/2 (scratch probe; the same oracle tests/test_devsim_smoke.py uses):
#
#     eta        32      50      100     200          <- 32 is _FD_U_MAX's eta, the ceiling
#     g fit    -0.09%  -1.07%  -3.24%  -5.61%         <- the TRANSPORT factor: undershoots
#     Delta    +0.40%  +0.01%  -1.20%  -2.96%         <- the built-in OFFSET: crosses over at ~50
#
# so the two halves do not err in the same direction, and quoting one for the other misleads.
# Anchors only: the warning INTERPOLATES linearly in eta between them and clamps outside.
_FD_FIT_ERR_ANCHORS = ((32.0, -0.09, +0.40), (50.0, -1.07, +0.01),
                       (100.0, -3.24, -1.20), (200.0, -5.61, -2.96))
# (carrier solution variable, its band-DOS parameter) -- Holes is absent on a unipolar region, and
# reading it is simply skipped there, so the same check covers both.
_FD_CARRIERS = (("Electrons", "N_dos"), ("Holes", "N_dos_p"))


def _fd_eta_from_u(u: float) -> float:
    """eta = (E_F - E_c)/kT implied by u = n/N_dos, i.e. the inverse of F_1/2 in the SAME
    Aymerich-Humet frame the equilibrium module uses. Diagnostic only (it labels a warning), so any
    inversion failure falls back to the degenerate asymptote u -> (4/(3 sqrt(pi))) eta^(3/2), which
    is good to 0.003 % by eta = 200 -- a diagnostic must never be the thing that raises."""
    u = float(u)
    asym = (0.75 * np.sqrt(np.pi) * max(u, 1e-300)) ** (2.0 / 3.0)
    try:
        return float(invert_F12(u, eta_max=max(100.0, 4.0 * asym), tol=1e-6))
    except Exception:                                          # pragma: no cover - diagnostic guard
        return float(asym)


def _fd_fit_error_pct(eta: float) -> "tuple[float, float]":
    """(g-fit, built-in-offset) relative error [%] at `eta`, linearly interpolated between the
    measured anchors of _FD_FIT_ERR_ANCHORS and clamped to the end anchors outside their range."""
    lo = _FD_FIT_ERR_ANCHORS[0]
    if eta <= lo[0]:
        return lo[1], lo[2]
    for hi in _FD_FIT_ERR_ANCHORS[1:]:
        if eta <= hi[0]:
            t = (eta - lo[0]) / (hi[0] - lo[0])
            return lo[1] + t * (hi[1] - lo[1]), lo[2] + t * (hi[2] - lo[2])
        lo = hi
    return lo[1], lo[2]


def warn_if_solution_exceeds_fd_ceiling(device: str, regions=None, *, stacklevel: int = 3):
    """audit C10-d. After a CONVERGED solve, warn once if the solution carrier density on any
    Fermi-Dirac region has run past the rational g fit's validated ceiling (u = n/N_dos > _FD_U_MAX,
    eta > 32). Returns the largest u seen on an FD region (None if there is no FD region at all).

    WHY THIS EXISTS SEPARATELY FROM THE SETUP-TIME TEST. `_fd_contact_shift` warns when the CHARGE-
    NEUTRAL doping at an ohmic contact is past the ceiling. That is the only degeneracy a doping
    profile can express -- and it is blind to exactly the device this library was built for: a gated
    accumulation cap (devsim_3d's `Stacked3DSpec`, the ENZ modulator) whose body doping sits
    comfortably below the ceiling while the gate drives n/N_dos far past it IN THE CHANNEL. Nothing
    warned there; the transport g-factor was quietly extrapolated. This reads the converged state
    instead, so it sees what the solve actually produced.

    COST: one `get_parameter` per region to find the FD regions, then one node-array read per
    carrier per FD region (two on a bipolar region). No quadrature, no per-node transcendental --
    the eta inversion runs ONCE, on the single worst node, and only when the ceiling is exceeded.

    SILENT when a region carries no `fd_enhancement` parameter (not a bipolar region), when it
    carries `fd_enhancement = 0` (a deliberate Boltzmann choice -- the ceiling is a property of the
    FD FIT, so a Boltzmann region has no ceiling to exceed), or when every u stays under it."""
    try:
        names = list(regions) if regions is not None else list(ds.get_region_list(device=device))
    except Exception:                                          # pragma: no cover - devsim internals
        return None
    seen_fd = False
    worst_u = 0.0
    over = []                                                  # (u, region, carrier) past the ceiling
    for region in names:
        try:
            if not (float(ds.get_parameter(device=device, region=region,
                                           name="fd_enhancement")) > 0.0):
                continue                                       # Boltzmann region, or not bipolar
        except Exception:
            continue                                           # no such parameter: not an FD region
        seen_fd = True
        for var, dos_name in _FD_CARRIERS:
            try:
                dos = float(ds.get_parameter(device=device, region=region, name=dos_name))
                vals = ds.get_node_model_values(device=device, region=region, name=var)
            except Exception:
                continue                                       # carrier/DOS absent (unipolar region)
            if not (dos > 0.0) or len(vals) == 0:
                continue
            u = float(np.max(np.asarray(vals, dtype=np.float64))) / dos
            worst_u = max(worst_u, u)
            if u > _FD_U_MAX:
                over.append((u, region, var))
    if not over:
        return worst_u if seen_fd else None
    over.sort(reverse=True)
    u, region, var = over[0]
    eta = _fd_eta_from_u(u)
    g_pct, d_pct = _fd_fit_error_pct(eta)
    extra = ("; also over the ceiling: " + ", ".join(
        "{} {} = {:.3g}".format(r, v, uu) for uu, r, v in over[1:])) if len(over) > 1 else ""
    warnings.warn(
        "Fermi-Dirac ceiling exceeded BY THE SOLUTION: region {!r} converged with "
        "max({})/{} = {:.4g} (eta = {:.1f}), past the rational g fit's validated ceiling of "
        "{:.4g} (eta = 32; carriers.einstein){}. Interpolating the measured error table (vs exact "
        "quadrature of F_1/2 and F_-1/2, 2026-07-26 -- eta 32/50/100/200: g -0.09/-1.07/-3.24/"
        "-5.61 %, built-in offset +0.40/+0.01/-1.20/-2.96 %), the transport g-factor is off by "
        "about {:+.2f} % here and the ohmic built-in offset by about {:+.2f} % -- they do NOT err "
        "in the same direction. The doping-level test at contact setup cannot see this: it reads "
        "NetDoping only, so a gated/accumulation device whose contact doping sits below the ceiling "
        "gets no warning from it (audit C10-d). Supply a larger N_dos/N_dos_p, reduce the "
        "accumulation, or set fd_enhancement=False and accept the Boltzmann model knowingly.".format(
            region, var, "N_dos" if var == "Electrons" else "N_dos_p", u, eta, _FD_U_MAX, extra,
            g_pct, d_pct),
        UnreliableResultWarning, stacklevel=stacklevel)
    return worst_u


def _g_expr(var: str, s: str, dos: str = "N_dos") -> str:
    """g(var{s}/dos) as a DEVSIM edge expression -- the rational fit + its coefficients live
    in carriers/einstein.g_expr_devsim (the single source, shared with physics_drift_diffusion;
    same coefficients for electrons and holes). `dos` is the band DOS parameter name (N_dos ~ N_c
    for electrons, N_dos_p ~ N_v for holes)."""
    return g_expr_devsim(var, dos, s)


# --- the ONE additive term mechanism (D1c hooks, extended at the v0.14 integration) ------------------
# Every opt-in physics term of a bipolar region enters through NAMED models, never by re-defining a
# composite or an equation slot by hand:
#   QTerm_<name> : a net positive charge DENSITY in units of +q [m^-3] (trapped charge, a sheet charge
#                  weighted onto interface nodes, ...) -> PotentialNodeCharge = -q (p - n + N + sum QTerm)
#                  (N = IonizedNetDoping when incomplete ionization is on, else NetDoping)
#   RTerm_<name> : a net recombination RATE [m^-3 s^-1] (Auger, radiative, a bulk trap level, surface
#                  SRH weighted onto interface nodes, an external generation as a negative rate, ...)
#                  -> ElectronGeneration = -q (USRH + sum RTerm), HoleGeneration = +q (...)
#   PTerm_<name> : the PAIR rate G + R of RTerm_<name> (the noise GR source 2 (G + R) of
#                  noise_analysis; omitted only when the caller gives no pair= -- the noise then
#                  REFUSES the device rather than silently dropping the term)
# A rate whose natural quadrature is not node-local (D1d BTBT: the field lives on edges/elements) uses
# the SAME mechanism with assembly="edge" (1-D: edge models RTermEdge_<name>, composite
# ElectronGenerationEdge into the equations' edge_volume_model) or assembly="element" (2-D/3-D:
# element models RTermElem0_/RTermElem1_<name> = each element edge's node0/node1 share, composites
# ElectronGenerationElem0/1 into volume_node0_model/volume_node1_model). The equation slot is wired
# ONCE through eq_registry.amend_region_equation (which refuses to overwrite a slot another module
# wired); later terms only rebuild the composite, so several terms SUM, never overwrite.
# Composites are REBUILT from whatever terms the region carries (sorted names: deterministic order),
# read back from DEVSIM's own model lists, so they live and die with the device (no module state).
# With NO terms every composite is the legacy string character-for-character and no model, slot or
# DEVSIM call is added (the defaults-off guards).
#
# VERSION 2 (v0.15, plan item 4.0.2; P3 section 1.3.2 step K0, P1 section 0.3):
#   RTermN_<name> / RTermP_<name> : a SINGLE-CARRIER rate (carrier="electron" / "hole"): it enters
#                  ElectronGeneration only / HoleGeneration only (a kinetic trap captures the two carriers
#                  at DIFFERENT rates R_n, R_p). PTermN_ / PTermP_ is its event rate for the noise, and
#                  the region parameter _rtN_<name> / _rtP_<name> = "<partner equation>,<coeff>,<mech>"
#                  records the OTHER row each event changes (or none) -- read back by
#                  recombination_terms, so the noise source is rebuilt from DEVSIM, not module state.
#                    ElectronGeneration = -q (USRH + sum RTerm_ + sum RTermN_)
#                    HoleGeneration     = +q (USRH + sum RTerm_ + sum RTermP_)
#                  With no N/P terms both strings are the version-1 strings character for character.
#   wrt=(<var>, ...): a term (any carrier, and QTerm_ via add_charge_term) that depends on EXTRA node
#                  solution variables (a trap occupancy) carries the derivative models <term>:<var>.
#                  Each composite is then differentiated with respect to Electrons, Holes AND every var
#                  for which one of its terms carries a derivative model (read back from DEVSIM's model
#                  list) plus every var the composite already has a derivative for. MEASURED (devsim
#                  2.11.2, 2026-11-06, docs/audit/2026-11-06-v0.15-term-bookkeeping-v2.md section 2):
#                  simplify(diff(C, U)) of a composite naming model M gives the symbol M:U for ANY node
#                  solution U, and the equation assembly consumes C:U for U's column (2.4e-16 against
#                  the closed form) -- the chain rule through named models works. But a MISSING M:U
#                  evaluates as 0 SILENTLY (the U column was 100 % wrong), and a C:U created BEFORE M:U
#                  keeps a stale cached 0 until a solution it depends on changes. Hence every declared
#                  var gets its <term>:<var> model (closed form, or DEVSIM's own diff) BEFORE the
#                  composite derivatives are rebuilt, and a var that is not a node model is refused.
#   "custom" kind: the region's live Python-assembled terms (custom_terms), with their noise capability
#                  and Jacobian exactness, so the ONE inventory covers every term the noise must see.
_CHARGE_TERM_PREFIX = "QTerm_"
_RECOMB_TERM_PREFIX = "RTerm_"
_PAIR_TERM_PREFIX = "PTerm_"
_EDGE_RTERM_PREFIX, _EDGE_PTERM_PREFIX = "RTermEdge_", "PTermEdge_"
_ELEM_RTERM_PREFIX = ("RTermElem0_", "RTermElem1_")
_ELEM_PTERM_PREFIX = ("PTermElem0_", "PTermElem1_")
_ASSEMBLIES = ("node", "edge", "element")
_CARRIERS = ("both", "electron", "hole")
# single-carrier terms: carrier -> (rate prefix, pair prefix, partner-parameter prefix, own row, own c)
_SINGLE = {"electron": ("RTermN_", "PTermN_", "_rtN_", "ElectronContinuityEquation", -1.0),
           "hole": ("RTermP_", "PTermP_", "_rtP_", "HoleContinuityEquation", +1.0)}
_BASE_VARS = ("Electrons", "Holes")


def _poisson_charge_expr(terms, doping: str = "NetDoping") -> str:
    """Bipolar Poisson node charge -q (p - n + <doping> [+ sum of QTerm_* models])."""
    base = "kahan3(Holes, -Electrons, {})".format(doping)
    if not terms:
        return "-ElectronCharge * " + base                      # legacy string, unchanged
    return "-ElectronCharge * ({} + {})".format(base, " + ".join(terms))


def _generation_exprs(terms, n_terms=(), p_terms=()) -> "tuple[str, str]":
    """(ElectronGeneration, HoleGeneration) = (-q R_e, +q R_h), R_e = USRH [+ sum RTerm_ + sum RTermN_],
    R_h = USRH [+ sum RTerm_ + sum RTermP_]. With no single-carrier terms R_e == R_h and both strings
    are the version-1 strings (no terms: the legacy strings) character for character."""
    def rate(names):
        names = list(names)
        return "(USRH + {})".format(" + ".join(names)) if names else "USRH"
    return ("-ElectronCharge * " + rate(list(terms) + list(n_terms)),
            "+ElectronCharge * " + rate(list(terms) + list(p_terms)))


def _region_terms(device: str, region: str, prefix: str) -> list:
    return sorted(m for m in ds.get_node_model_list(device=device, region=region)
                  if m.startswith(prefix) and ":" not in m)


def _check_wrt(device: str, region: str, wrt, who: str) -> tuple:
    """The extra solution variables of a term, validated: each must already be a node model of the
    region (a node_solution), because DEVSIM differentiates with respect to ANY symbol without
    complaint -- an unknown one silently gives 0 (MEASURED, block comment above)."""
    out = []
    have = None
    for v in (() if wrt is None else ((wrt,) if isinstance(wrt, str) else tuple(wrt))):
        if v in _BASE_VARS or v in out:
            continue
        if have is None:
            have = set(ds.get_node_model_list(device=device, region=region))
        if not isinstance(v, str) or v not in have or ":" in v:
            raise ValueError("{}: wrt variable {!r} is not a node solution of region {!r}; create it "
                             "(ds.node_solution) before adding a term that depends on it".format(
                                 who, v, region))
        out.append(v)
    return tuple(out)


def _derivative_vars(device: str, region: str, models, composite: str) -> tuple:
    """Electrons, Holes + every extra variable v for which one of `models` carries the derivative model
    <model>:v, or for which `composite` already has <composite>:v (a var whose last term was replaced
    keeps a derivative that DEVSIM re-differentiates to 0 instead of a stale one). Read from DEVSIM."""
    have = ds.get_node_model_list(device=device, region=region)
    owners = set(models) | {composite}
    extra = sorted({m.split(":", 1)[1] for m in have if ":" in m and m.split(":", 1)[0] in owners}
                   - set(_BASE_VARS))
    return _BASE_VARS + tuple(extra)


def _term_with_derivs(device: str, region: str, model: str, expr: str, derivs, wrt=()) -> None:
    """A term model + its derivatives with respect to Electrons, Holes and every extra variable of
    `wrt`: DEVSIM's simplify(diff()) by default, or the caller's closed forms `derivs` = {"Electrons":
    ..., "Holes": ..., <each wrt var>: ...} (a long sum -- e.g. ~45 interface trap bins -- takes DEVSIM
    ~25 s to differentiate symbolically, the closed forms well under 1 s). A closed-form dict must cover
    every variable (a missing derivative model is a SILENT zero in DEVSIM). Derivative models of an
    extra variable the term no longer depends on (a re-added term) are deleted, so no stale Jacobian
    entry survives a replacement."""
    wrt_all = _BASE_VARS + tuple(wrt)
    if derivs is not None:
        missing = [w for w in wrt_all if w not in derivs]
        extra = sorted(set(derivs) - set(wrt_all))
        if missing or extra:
            raise ValueError("closed-form derivs of {!r} must give exactly {} (missing {}, not in wrt "
                             "{}): DEVSIM treats a missing derivative model as 0".format(
                                 model, list(wrt_all), missing, extra))
    stale = [m for m in ds.get_node_model_list(device=device, region=region)
             if m.startswith(model + ":") and m.split(":", 1)[1] not in wrt_all]
    for m in stale:
        ds.delete_node_model(device=device, region=region, name=m)
    if derivs is None:
        _node_with_derivs(device, region, model, expr, wrt_all)
        return
    ds.node_model(device=device, region=region, name=model, equation=expr)
    for w in wrt_all:
        ds.node_model(device=device, region=region, name="{}:{}".format(model, w), equation=derivs[w])


def rebuild_poisson_charge(device: str, region: str) -> None:
    """(Re)build PotentialNodeCharge from the region's doping model (IonizedNetDoping when incomplete
    ionization installed it, else NetDoping) plus every QTerm_* charge term, differentiated with respect
    to Electrons, Holes and every extra variable a charge term carries a derivative for (add_charge_term
    wrt=)."""
    doping = ("IonizedNetDoping" if "IonizedNetDoping" in ds.get_node_model_list(
        device=device, region=region) else "NetDoping")
    terms = _region_terms(device, region, _CHARGE_TERM_PREFIX)
    pne = _poisson_charge_expr(terms, doping)
    _node_with_derivs(device, region, "PotentialNodeCharge", pne,
                      _derivative_vars(device, region, terms, "PotentialNodeCharge"))


def add_charge_term(device: str, region: str, name: str, expr: str, *, derivs=None, wrt=None) -> str:
    """Add a net positive space-charge density `expr` [m^-3, units of +q; a DEVSIM node expression in
    Electrons/Holes and parameters] to a bipolar region's Poisson equation, as node model
    QTerm_<name> (+ its Electrons/Holes derivatives), and rebuild PotentialNodeCharge. Re-adding the
    same name REPLACES that term. `derivs` optionally supplies closed-form derivatives (see
    _term_with_derivs). `wrt` names EXTRA node solution variables the charge depends on (a kinetic trap's
    occupancy variable): the term then also carries <model>:<var> (from `derivs` or DEVSIM's diff) and
    the Poisson composite is differentiated with respect to them. Returns the model name."""
    model = _CHARGE_TERM_PREFIX + name
    extra = _check_wrt(device, region, wrt, "add_charge_term")
    _term_with_derivs(device, region, model, expr, derivs, extra)
    rebuild_poisson_charge(device, region)
    return model


def _element_wrt(device: str) -> list:
    dim = int(ds.get_dimension(device=device))
    ens = ["@en{}".format(k) for k in range(dim + 1)]
    return [v + e for v in ("Potential", "Electrons", "Holes") for e in ens]


def _wire_volume_slot(device: str, region: str, slots: dict) -> None:
    """Point the continuity equations' edge/element volume slots at the term composites, ONCE.
    `slots` maps an equation keyword to (electron composite, hole composite). A slot already carrying
    the composite is left alone; one carrying a FOREIGN model makes amend_region_equation refuse."""
    for eq, idx in (("ElectronContinuityEquation", 0), ("HoleContinuityEquation", 1)):
        rec = _R.recorded(device, eq, scope="region", loc=region)
        if len(rec) != 1:
            raise ValueError("add_recombination_term: region {!r} has no recorded {} -- a bipolar "
                             "drift-diffusion region (setup_bipolar_region) is required".format(
                                 region, eq))
        kw = rec[0]["kwargs"]
        need = {k: v[idx] for k, v in slots.items() if kw.get(k) != v[idx]}
        if need:
            _R.amend_region_equation(device, region, eq, **need)


def _rebuild_edge_composites(device: str, region: str) -> None:
    terms = sorted(m for m in ds.get_edge_model_list(device=device, region=region)
                   if m.startswith(_EDGE_RTERM_PREFIX) and ":" not in m)
    r = "({})".format(" + ".join(terms))
    wrt = ("Potential", "Electrons", "Holes")
    _edge_with_derivs(device, region, "ElectronGenerationEdge", "-ElectronCharge*" + r, wrt)
    _edge_with_derivs(device, region, "HoleGenerationEdge", "ElectronCharge*" + r, wrt)
    _wire_volume_slot(device, region, {"edge_volume_model": ("ElectronGenerationEdge",
                                                             "HoleGenerationEdge")})


def _rebuild_element_composites(device: str, region: str) -> None:
    wrt = _element_wrt(device)
    have = ds.get_element_model_list(device=device, region=region)
    slots = {}
    for k, pre in enumerate(_ELEM_RTERM_PREFIX):
        terms = sorted(m for m in have if m.startswith(pre) and ":" not in m)
        r = "({})".format(" + ".join(terms))
        en, hn = "ElectronGenerationElem{}".format(k), "HoleGenerationElem{}".format(k)
        _R.element_with_derivs(device, region, en, "-ElectronCharge*" + r, wrt)
        _R.element_with_derivs(device, region, hn, "ElectronCharge*" + r, wrt)
        slots["volume_node{}_model".format(k)] = (en, hn)
    _wire_volume_slot(device, region, slots)


def _rebuild_generation(device: str, region: str) -> None:
    """(Re)build ElectronGeneration / HoleGeneration (+ derivatives) from the region's node terms: the
    shared RTerm_* in both, RTermN_* in the electron composite only, RTermP_* in the hole one only."""
    both = _region_terms(device, region, _RECOMB_TERM_PREFIX)
    nt = _region_terms(device, region, _SINGLE["electron"][0])
    pt = _region_terms(device, region, _SINGLE["hole"][0])
    gn, gp = _generation_exprs(both, nt, pt)
    _node_with_derivs(device, region, "ElectronGeneration", gn,
                      _derivative_vars(device, region, both + nt, "ElectronGeneration"))
    _node_with_derivs(device, region, "HoleGeneration", gp,
                      _derivative_vars(device, region, both + pt, "HoleGeneration"))


def _single_partner(device: str, region: str, carrier: str, name: str):
    """(partner equation | None, partner coefficient, mechanism) of a single-carrier node term, read back
    from the region parameter add_recombination_term recorded."""
    par = _SINGLE[carrier][2] + name
    if par not in ds.get_parameter_list(device=device, region=region):
        return None, 0.0, "gr"
    eq, coeff, mech = str(ds.get_parameter(device=device, region=region, name=par)).split(",")
    return (eq or None), float(coeff), mech


def add_recombination_term(device: str, region: str, name: str, expr, *, pair=None, derivs=None,
                           assembly: str = "node", carrier: str = "both", wrt=None, partner=None,
                           mechanism: str = "gr") -> str:
    """Add a net recombination rate `expr` [m^-3 s^-1] (negative = generation) to BOTH continuity
    equations of a bipolar region and rebuild the composite it sums into (see the block comment above).
    Re-adding the same name REPLACES that term; distinct names always SUM.

      assembly="node"    : `expr` a node expression -> RTerm_<name> (+ Electrons/Holes derivatives;
                           `derivs` optionally the closed forms, as in add_charge_term).
      assembly="edge"    : `expr` an edge expression (1-D field-driven rates) -> RTermEdge_<name>,
                           assembled with EdgeNodeVolume into both end nodes (edge_volume_model).
      assembly="element" : `expr` = (node0 share, node1 share) element expressions (2-D/3-D) ->
                           RTermElem0_/RTermElem1_<name>, assembled with ElementNodeVolume into the
                           element edge's node0 / node1 (volume_node0_model / volume_node1_model).
      pair               : the PAIR rate G + R of the same process in the same form as `expr` (a
                           2-tuple for "element"); it is the noise GR source (noise_analysis). Pass
                           "0" for a deterministic term. None leaves the term without a pair rate and
                           noise_analysis refuses the region (so no term can be silently noiseless).

    Version 2 (v0.15, plan 4.0.2; node assembly only). Which option for which calculation:
      carrier="both" (default): one process removes (or creates) an electron AND a hole at the same node
                           -- SRH through a fast level, Auger, radiative, external generation. Unchanged.
      carrier="electron" / "hole": the process changes ONE carrier only -- the capture / emission branch
                           of a KINETIC trap (R_n into the electron equation, R_p into the hole one, the
                           trap's own equation balancing both), or a carrier exchange with a reservoir the
                           device does not model. -> RTermN_<name> / RTermP_<name> (ElectronGeneration
                           only / HoleGeneration only). `pair` is then that process's EVENT rate, both
                           directions counted (e.g. c_n n (1 - f) + e_n f) [m^-3 s^-1]; its noise source is
                           a two-point event (noise_analysis.TwoPointSource) on the carrier's own row (an
                           electron event changes the electron row by -1, a hole event the hole row by
                           +1: ElectronGeneration = -q R, HoleGeneration = +q R) and, when given, on
      partner            : (equation, coeff) -- the OTHER row of the same region each event changes; coeff
                           is its change per event in units of q in DEVSIM's sign convention for that row
                           (a trap row with node model +q (R_n - R_p): +1 for the electron branch, -1 for
                           the hole branch). None: the carrier leaves the modelled system (no second row;
                           the terminal currents then do not balance -- a sink to nowhere does not
                           conserve charge). The equation may be created after the term; noise_analysis
                           refuses the region if it is missing when the noise is computed.
      mechanism          : the noise_analysis.MECHANISMS key a single-carrier source is reported under
                           ("gr" by default; a kinetic trap uses "trap_n" / "trap_p").
      wrt                : EXTRA node solution variables the term depends on (a trap occupancy variable;
                           any carrier, node assembly only). The term carries <model>:<var> for each (from
                           `derivs`, which must then give them too, or DEVSIM's simplify(diff())), and the
                           composites are differentiated with respect to them. The variable must exist (a
                           node_solution) before the term is added: DEVSIM silently differentiates an
                           unknown symbol to 0.
    A name is either a pair term (carrier="both") or a single-carrier one; the electron and hole branches
    of one process share it (RTermN_<name> + RTermP_<name>). Mixing is refused: there is no term removal,
    and a fast-trap pair left beside its kinetic branches would double-count silently. Returns the
    (first) term model name."""
    if assembly not in _ASSEMBLIES:
        raise ValueError("add_recombination_term: assembly must be one of {}; got {!r}".format(
            _ASSEMBLIES, assembly))
    if carrier not in _CARRIERS:
        raise ValueError("add_recombination_term: carrier must be one of {}; got {!r}".format(
            _CARRIERS, carrier))
    if assembly != "node" and (carrier != "both" or wrt is not None):
        raise ValueError("add_recombination_term: carrier= and wrt= are supported for node terms only "
                         "(assembly={!r})".format(assembly))
    if (partner is not None or mechanism != "gr") and carrier == "both":
        raise ValueError("add_recombination_term: partner= / mechanism= describe a SINGLE-carrier "
                         "term's noise event; a carrier='both' term is a same-node pair ('gr')")
    if assembly == "node":
        extra = _check_wrt(device, region, wrt, "add_recombination_term")
        if carrier == "both":
            model, pm = _RECOMB_TERM_PREFIX + name, _PAIR_TERM_PREFIX + name
        else:
            model, pm = _SINGLE[carrier][0] + name, _SINGLE[carrier][1] + name
            from dynameta.carriers.noise_analysis import MECHANISMS
            if mechanism not in MECHANISMS:
                raise ValueError("add_recombination_term: mechanism {!r} is not a noise_analysis."
                                 "MECHANISMS key {}".format(mechanism, MECHANISMS))
            p_eq, p_c = ("", 0.0) if partner is None else (str(partner[0]), float(partner[1]))
            if partner is not None and (not p_eq or "," in p_eq or not np.isfinite(p_c) or p_c == 0.0):
                raise ValueError("add_recombination_term: partner must be (equation name, finite nonzero "
                                 "coeff); got {!r}".format(partner))
        have = set(ds.get_node_model_list(device=device, region=region))
        rivals = ((v[0] for v in _SINGLE.values()) if carrier == "both" else (_RECOMB_TERM_PREFIX,))
        clash = [p + name for p in rivals if p + name in have]
        if clash:
            raise ValueError("add_recombination_term: region {!r} already carries {} under the name "
                             "{!r}; a name is either a pair term (carrier='both') or a single-carrier "
                             "one (pick a new name)".format(region, clash, name))
        _term_with_derivs(device, region, model, expr, derivs, extra)
        if pair is not None:
            ds.node_model(device=device, region=region, name=pm, equation=pair)
        elif pm in ds.get_node_model_list(device=device, region=region):
            ds.delete_node_model(device=device, region=region, name=pm)      # no stale pair rate
        if carrier != "both":
            ds.set_parameter(device=device, region=region, name=_SINGLE[carrier][2] + name,
                             value="{},{!r},{}".format(p_eq, p_c, mechanism))
        _rebuild_generation(device, region)
        return model
    if derivs is not None:
        raise ValueError("add_recombination_term: closed-form derivs are supported for node terms only")
    if assembly == "edge":
        model = _EDGE_RTERM_PREFIX + name
        _edge_with_derivs(device, region, model, expr, ("Potential", "Electrons", "Holes"))
        pm = _EDGE_PTERM_PREFIX + name
        if pair is not None:
            ds.edge_model(device=device, region=region, name=pm, equation=pair)
        elif pm in ds.get_edge_model_list(device=device, region=region):
            ds.delete_edge_model(device=device, region=region, name=pm)
        _rebuild_edge_composites(device, region)
        return model
    exprs = tuple(expr)
    pairs = tuple(pair) if pair is not None else (None, None)
    if len(exprs) != 2 or len(pairs) != 2:
        raise ValueError("add_recombination_term(assembly='element'): expr and pair are (node0, node1) "
                         "share pairs")
    wrt = _element_wrt(device)
    have = ds.get_element_model_list(device=device, region=region)
    for k in (0, 1):
        _R.element_with_derivs(device, region, _ELEM_RTERM_PREFIX[k] + name, exprs[k], wrt)
        pm = _ELEM_PTERM_PREFIX[k] + name
        if pairs[k] is not None:
            ds.element_model(device=device, region=region, name=pm, equation=pairs[k])
        elif pm in have:
            ds.delete_element_model(device=device, region=region, name=pm)
    _rebuild_element_composites(device, region)
    return _ELEM_RTERM_PREFIX[0] + name


class CarrierTerm(NamedTuple):
    """One single-carrier node term of recombination_terms()["node_electron" / "node_hole"]."""
    rate: str                         # RTermN_<name> / RTermP_<name>
    pair: Optional[str]               # PTermN_<name> / PTermP_<name>; None = no noise model (refused)
    partner_equation: Optional[str]   # the other row each event changes; None = none
    partner_coeff: float              # its change per event in units of q (0.0 without a partner)
    mechanism: str                    # the noise_analysis.MECHANISMS key of its source
    own_equation: str                 # ElectronContinuityEquation / HoleContinuityEquation
    own_coeff: float                  # -1.0 (electron) / +1.0 (hole): the carrier row's change per event


class CustomTermRecord(NamedTuple):
    """One live Python-assembled term of recombination_terms()["custom"] (carriers/custom_terms)."""
    term: object                      # the custom_terms.CustomTerm
    regions: Tuple[str, ...]          # regions of its declared rows; () = undeclared (listed everywhere)
    noise_model: bool                 # it implements noise_sources (else noise_analysis refuses)
    jacobian_exact: bool              # False: noise_analysis refuses (ssac / noise ARE its Jacobian)


def _custom_records(device: str, region: str) -> dict:
    """The live custom terms whose declared rows touch `region` -- or that declare no region row at all:
    their footprint is unknown, so they are listed in EVERY region (never silently missed). Liveness is
    custom_terms' DEVSIM-marker test; nothing is registered."""
    from dynameta.carriers import custom_terms as _CT
    out = {}
    for t in _CT.live_terms(device):
        regs = tuple(sorted({r for r, _eq in t.equations(device) if r is not None}))
        if regs and region not in regs:
            continue
        out[t.name] = CustomTermRecord(
            t, regs, type(t).noise_sources is not _CT.CustomTerm.noise_sources, bool(t.jacobian_exact))
    return out


def recombination_terms(device: str, region: str) -> dict:
    """Inventory of the region's additive recombination terms, read back from DEVSIM:
    {"node": {name: (rate model, pair model | None)}, "edge": {...}, "element": {name: ((rate0,
    rate1), (pair0, pair1) | None)}, "node_electron": {name: CarrierTerm}, "node_hole": {name:
    CarrierTerm}, "custom": {name: CustomTermRecord}}. The base SRH channel USRH is not listed (it is
    always present and its pair rate is physics_traps.srh_rate_expr(pair=True)). "node_electron" /
    "node_hole" are the single-carrier terms (add_recombination_term carrier=), "custom" the live
    Python-assembled terms (custom_terms) writing rows of this region or declaring none. Used by
    noise_analysis -- which refuses a term of ANY kind without a noise model -- and the composition
    gate."""
    nodes = set(ds.get_node_model_list(device=device, region=region))
    edges = set(ds.get_edge_model_list(device=device, region=region))
    elems = set()
    if int(ds.get_dimension(device=device)) > 1:
        elems = set(ds.get_element_model_list(device=device, region=region))

    def names(models, prefix):
        return sorted(m[len(prefix):] for m in models if m.startswith(prefix) and ":" not in m)

    out = {"node": {}, "edge": {}, "element": {}, "node_electron": {}, "node_hole": {}}
    for nm in names(nodes, _RECOMB_TERM_PREFIX):
        pm = _PAIR_TERM_PREFIX + nm
        out["node"][nm] = (_RECOMB_TERM_PREFIX + nm, pm if pm in nodes else None)
    for nm in names(edges, _EDGE_RTERM_PREFIX):
        pm = _EDGE_PTERM_PREFIX + nm
        out["edge"][nm] = (_EDGE_RTERM_PREFIX + nm, pm if pm in edges else None)
    for nm in names(elems, _ELEM_RTERM_PREFIX[0]):
        pms = tuple(p + nm for p in _ELEM_PTERM_PREFIX)
        out["element"][nm] = (tuple(p + nm for p in _ELEM_RTERM_PREFIX),
                              pms if all(p in elems for p in pms) else None)
    for carrier, key in (("electron", "node_electron"), ("hole", "node_hole")):
        rp, pp, _par, own_eq, own_c = _SINGLE[carrier]
        for nm in names(nodes, rp):
            peq, pc, mech = _single_partner(device, region, carrier, nm)
            out[key][nm] = CarrierTerm(rp + nm, pp + nm if pp + nm in nodes else None, peq, pc, mech,
                                       own_eq, own_c)
    out["custom"] = _custom_records(device, region)
    return out


def setup_bipolar_region(device: str, region: str, *,
                         eps_static: float, n_dos_m3: float, n_i_m3: float,
                         mobility_n_m2Vs: float, mobility_p_m2Vs: float,
                         tau_n_s: float, tau_p_s: float,
                         fd_enhancement: bool = True,
                         n_dos_p_m3: "float | None" = None,
                         options: "BipolarOptions | None" = None,
                         btbt=None,
                         srh_trap_energy_eV: float = 0.0,
                         traps=None,
                         statistics: "str | None" = None) -> None:
    """Attach bipolar drift-diffusion physics to a semiconductor region.

    Parameters (all SI):
      eps_static       : static relative permittivity (dimensionless)
      n_dos_m3         : CONDUCTION-band effective DOS (~N_c, m^-3) for the electron FD g-factor.
                         For a non-degenerate diode the exact value is irrelevant (g~1).
      n_dos_p_m3       : VALENCE-band effective DOS (~N_v, m^-3) for the HOLE FD g-factor; defaults
                         to n_dos_m3 (N_v == N_c) -- supply N_v for a degenerate p-region (audit DD-5).
      n_i_m3           : intrinsic carrier density (m^-3)
      mobility_n_m2Vs  : electron mobility (m^2/(V s))
      mobility_p_m2Vs  : hole mobility (m^2/(V s))
      tau_n_s, tau_p_s : SRH lifetimes (s); must be > 0 -- use a LARGE value to suppress
                         recombination (tau=0 would make the SRH denominator 0/0 -> NaN)
      fd_enhancement   : if True, apply the degenerate FD g-factor to BOTH currents;
                         if False, plain Boltzmann Scharfetter-Gummel (g==1).
      statistics       : carrier statistics (bipolar_options module doc, STATISTICS). v0.15.0 DEFAULT
                         (None with fd_enhancement=True): "fermi_dirac" -- FD transport (the consistent
                         quasi-field) AND FD detailed balance (n_ie carries exp(-Delta/2) per carrier;
                         contacts pin the FD root): use it for every calculation unless you reproduce
                         v0.14 numbers. None with fd_enhancement=False, or "boltzmann" (needs
                         fd_enhancement=False): Boltzmann everywhere -- use it with an APPARENT
                         (Boltzmann-extracted) BGN set. "legacy" (needs fd_enhancement=True): the v0.14
                         default, byte-identical -- FD transport with the edge-averaged g and the
                         Boltzmann product n p = n_ie^2 in recombination and contacts; it carries a
                         zero-bias current next to degenerate layers (-0.12 J0 at 1e19 cm^-3 Si, plus the
                         averaged g's O(h^2) battery). NOT recommended; for reproducing published v0.14
                         results only. v0.15 completion: "fermi_dirac" also
                         carries btbt (the Hurkx D with the FD Fermi functions at E_ie), CHARGED trap levels
                         (exact FD emission densities gamma_n n1, gamma_p p1 -- also for the built-in SRH
                         channel -- and their exact FD equilibrium occupancy in the contacts' neutral root),
                         interface traps and the nonlocal BTBT term, all with exact Jacobians
                         (docs/audit/2026-11-06-v0.15-fermi-dirac-mode-complete.md).
      options          : opt-in BipolarOptions (audit R26/R27: Auger, radiative, bandgap narrowing,
                         incomplete ionization, T_K, uniform generation; see carriers/bipolar_options).
                         None (default) -- or a bundle with every field None -- leaves every model
                         below BYTE-IDENTICAL to the SRH-only physics.
      btbt             : None (default: NO band-to-band tunneling -- the models and equations are
                         exactly the legacy ones) or a carriers.tunneling.BTBTParams: adds the local
                         Kane/Hurkx BTBT net-recombination term to both continuity equations, coupled
                         in the Newton (carriers/tunneling.py; D1d). v0.15: or a tuple of them, one
                         summed term per process (tunneling.GERMANIUM_KAO); the sets are installed as
                         given -- at a T_K region apply tunneling.btbt_at_temperature first.
      srh_trap_energy_eV : E_t - E_i [eV] of the built-in SRH channel (tau_n_s, tau_p_s): n1 =
                         n_i exp(E/kT), p1 = n_i exp(-E/kT) (Shockley-Read 1952 / Hall 1952). The
                         default 0.0 is the legacy midgap level BIT-identically (exp(0) == 1).
      traps            : optional sequence of physics_traps.TrapLevel -- extra discrete SRH levels,
                         each with its own E_t, lifetimes/cross sections, density and charge type.
                         Each level's rate is its own recombination term RTerm_trap<k> (with its noise
                         pair rate) and the trapped charge (acceptor -q N_t f_t, donor +q N_t (1 - f_t),
                         steady-state SRH occupancy) the charge term QTerm_bulk_traps; CHARGED levels
                         also enter the ohmic-contact / seed charge-neutral equilibrium. None/empty
                         (default) leaves every model string exactly as before.

    COMPOSITION (v0.14 integration): every opt-in term -- Auger, radiative, external G (options),
    the trap levels, interface SRH (physics_traps.install_interface_traps), BTBT (btbt) -- enters
    through the ONE additive mechanism (add_recombination_term / add_charge_term), so any combination
    SUMS. Every SRH channel (the built-in one, each TrapLevel, each interface bin) comes from the ONE
    kernel physics_traps.srh_rate_expr with the region's bandgap-narrowing substitution, and BTBT's
    Hurkx D uses n_ie when BGN is on.

    The region must already carry a "NetDoping" node model (signed: +Nd donor,
    -Na acceptor). Build it before calling (the diode mesh sets it per-region).
    """
    require_positive(eps_static=eps_static, n_dos_m3=n_dos_m3, n_i_m3=n_i_m3,
                     mobility_n_m2Vs=mobility_n_m2Vs, mobility_p_m2Vs=mobility_p_m2Vs,
                     tau_n_s=tau_n_s, tau_p_s=tau_p_s)
    ds.set_parameter(device=device, region=region, name="Permittivity",
                     value=eps_static * EPS0)
    ds.set_parameter(device=device, region=region, name="ElectronCharge", value=Q_E)
    ds.set_parameter(device=device, region=region, name="V_t", value=V_T)
    ds.set_parameter(device=device, region=region, name="mu_n", value=mobility_n_m2Vs)
    ds.set_parameter(device=device, region=region, name="mu_p", value=mobility_p_m2Vs)
    ds.set_parameter(device=device, region=region, name="N_dos", value=n_dos_m3)
    ds.set_parameter(device=device, region=region, name="N_dos_p",
                     value=float(n_dos_p_m3) if n_dos_p_m3 is not None else n_dos_m3)
    ds.set_parameter(device=device, region=region, name="n_i", value=n_i_m3)
    ds.set_parameter(device=device, region=region, name="taun", value=tau_n_s)
    ds.set_parameter(device=device, region=region, name="taup", value=tau_p_s)
    # SRH trap level of the built-in channel: n1 = n_i exp(E/kT), p1 = n_i exp(-E/kT) (D1c traps,
    # 2026-11-03). E = 0 (default) -> n_i * 1.0 == n_i exactly: the legacy midgap values, bit for bit.
    # kT of the region: BipolarOptions.T_K when set (V_T otherwise -- the same float), so a level keeps
    # its energy at the simulated temperature.
    from dynameta.carriers.bipolar_options import resolve_statistics
    _label = resolve_statistics(statistics, fd_enhancement)                 # validates the pair
    _fdc = _label == "fermi_dirac"
    _opt = options if (options is not None and options.active) else None
    if _fdc:
        if _opt is None:                       # FD detailed balance alone still needs the option path
            from dynameta.carriers.bipolar_options import BipolarOptions
            _opt = options if options is not None else BipolarOptions()
    _vt = _opt.V_t() if _opt is not None else V_T
    # _bgn: the recombination strings reference the region's n_ie model (BGN, or FD detailed balance)
    _bgn = _opt is not None and (_opt.bgn is not None or _fdc)
    _n1, _p1 = _PT.n1_p1(srh_trap_energy_eV, n_i_m3, _vt)
    ds.set_parameter(device=device, region=region, name="n1", value=_n1)  # SRH trap
    ds.set_parameter(device=device, region=region, name="p1", value=_p1)
    # audit C-10: the contact builders are called later and separately, and the ohmic built-in
    # potential must follow the SAME statistics as the transport in this region. Record the choice
    # here (1.0 = Fermi-Dirac g-factor, 0.0 = Boltzmann) so setup_contact_ohmic_bipolar* can read it.
    ds.set_parameter(device=device, region=region, name="fd_enhancement",
                     value=1.0 if fd_enhancement else 0.0)
    if _fdc:                                   # v0.15: recorded ONLY when set (legacy log unchanged)
        ds.set_parameter(device=device, region=region, name="fd_detailed_balance", value=1.0)

    # --- solution variables ---
    ds.node_solution(device=device, region=region, name="Potential")
    ds.node_solution(device=device, region=region, name="Electrons")
    ds.node_solution(device=device, region=region, name="Holes")
    ds.edge_from_node_model(device=device, region=region, node_model="Potential")
    ds.edge_from_node_model(device=device, region=region, node_model="Electrons")
    ds.edge_from_node_model(device=device, region=region, node_model="Holes")

    # --- bipolar Poisson node charge: -q (p - n + NetDoping) ---
    pne = _poisson_charge_expr(())
    _node_with_derivs(device, region, "PotentialNodeCharge", pne, ("Electrons", "Holes"))
    _poisson_edge_models(device, region)
    _R.record_region_equation(device, region, name="PotentialEquation",
                              variable_name="Potential",
                              node_model="PotentialNodeCharge",
                              edge_model="PotentialEdgeFlux",
                              variable_update="log_damp")

    # --- shared Bernoulli argument vdiff = (psi@n0 - psi@n1)/V_t ---
    _edge_with_derivs(device, region, "vdiff",
                      "(Potential@n0 - Potential@n1)/V_t", ("Potential",))

    # --- FD diffusion enhancement g(c) (accurate rational fit; see module docstring) ---
    # edge value = average of g(c/N_dos) over the edge's two nodes.
    _fdq_n = _fdq_p = ""
    if _fdc:
        # statistics="fermi_dirac": the degeneracy as a QUASI-FIELD (g = 1 in the prefactor, Delta1 - Delta0 in
        # the Bernoulli argument; physics_heterojunction.fd_quasi_field_expr) -- its discrete zero-current
        # state IS the FD equilibrium every detailed-balance term uses, and its edge conductance is the noise
        # model's (exact FDT)
        from dynameta.carriers.physics_heterojunction import fd_quasi_field_expr
        _fdq_n = " + " + fd_quasi_field_expr(device, region, "Electrons", "N_dos")
        _fdq_p = " - " + fd_quasi_field_expr(device, region, "Holes", "N_dos_p")
        ds.edge_model(device=device, region=region, name="g_enh", equation="1.0")
        ds.edge_model(device=device, region=region, name="g_enh_p", equation="1.0")
    elif fd_enhancement:
        g_n = "(0.5*({} + {}))".format(_g_expr("Electrons", "@n0"), _g_expr("Electrons", "@n1"))
        g_p = "(0.5*({} + {}))".format(_g_expr("Holes", "@n0", dos="N_dos_p"),
                                       _g_expr("Holes", "@n1", dos="N_dos_p"))
        _edge_with_derivs(device, region, "g_enh", g_n, ("Electrons",))
        _edge_with_derivs(device, region, "g_enh_p", g_p, ("Holes",))
    else:
        ds.edge_model(device=device, region=region, name="g_enh", equation="1.0")
        ds.edge_model(device=device, region=region, name="g_enh_p", equation="1.0")

    # --- electron / hole currents (FD-scaled Scharfetter-Gummel), the ONE SG kernel (sg_flux) ---
    # vdiff_g = vdiff/g_enh, vdiff_gp = vdiff/g_enh_p; Bern_g(p) = B(v), Bern_g(p)_m = B(-v);
    #   Jn = +q mu_n ... g_enh   ( n@n1 B(-v) - n@n0 B(v) )   (Jn has no Holes dependence)
    #   Jp = -q mu_p ... g_enh_p ( p@n1 B(v)  - p@n0 B(-v) )  (Jp has no Electrons dependence)
    # Sign-selected two-Bernoulli bracket since 2026-11-04: the kahan3 spelling B(-v) = B(v) + v
    # cancelled for v << 0 (1.9e-7 at v = -21, 100 % at -40; sg_flux header, docs/audit/2026-11-04-
    # sg-flux-two-bernoulli.md).
    _edge_with_derivs(device, region, "vdiff_g", "vdiff / g_enh" + _fdq_n,
                      ("Potential", "Electrons"))
    _SG.install_sg_current(device, region, carrier="electrons", density="Electrons",
                           vdiff="vdiff_g", bern="Bern_g", bern_m="Bern_g_m",
                           current="ElectronCurrent",
                           prefactor="ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh",
                           wrt=("Potential", "Electrons"), current_wrt=("Electrons", "Potential"))
    _edge_with_derivs(device, region, "vdiff_gp", "vdiff / g_enh_p" + _fdq_p,
                      ("Potential", "Holes"))
    _SG.install_sg_current(device, region, carrier="holes", density="Holes",
                           vdiff="vdiff_gp", bern="Bern_gp", bern_m="Bern_gp_m",
                           current="HoleCurrent",
                           prefactor="-ElectronCharge*mu_p*EdgeInverseLength*V_t*g_enh_p",
                           wrt=("Potential", "Holes"), current_wrt=("Holes", "Potential"))

    # --- audit R26/R27 fixed fields (BipolarOptions): T_K -> V_t, n_ie + the BGN band-edge quasi-fields
    #     in both SG currents, the incomplete-ionization reference fields + IonizedNetDoping. Installed
    #     BEFORE the recombination models, which reference n_ie. Off -> nothing runs here.
    if _opt is not None:
        from dynameta.carriers.bipolar_options import install_option_fields
        install_option_fields(device, region, _opt, fd_enhancement=fd_enhancement,
                              statistics=_label)

    # --- SRH recombination: one node model into BOTH continuity equations ---
    # srh_rate_expr() with its default names IS the legacy string (pinned in tests/test_traps.py); with
    # BGN the SAME kernel applies the n_ie substitution. Extra discrete levels (traps=...) are
    # independent channels, each its own additive term (physics_traps.install_bulk_traps).
    # statistics="fermi_dirac": the exact FD emission densities gamma_n n1, gamma_p p1 (fd=True; the ONE kernel
    # for this channel, every TrapLevel, every interface bin and the noise pair rate)
    _fkw = {"fd": True} if _fdc else {}
    _node_with_derivs(device, region, "USRH", _PT.srh_rate_expr(bgn=_bgn, **_fkw), ("Electrons", "Holes"))
    gn, gp = _generation_exprs(())
    _node_with_derivs(device, region, "ElectronGeneration", gn, ("Electrons", "Holes"))
    _node_with_derivs(device, region, "HoleGeneration", gp, ("Electrons", "Holes"))
    if traps:
        _PT.install_bulk_traps(device, region, list(traps), n_i_m3=n_i_m3, V_t=_vt, bgn=_bgn, **_fkw)

    # --- audit R26/R27 terms (Auger, radiative, external G) through the same additive mechanism, the
    #     ionized-dopant Poisson charge, and the charge-neutral equilibrium fields the contacts and
    #     the seed pin (BGN / ionization / CHARGED trap levels). Off (and no charged trap) -> nothing.
    _eq_traps = _PT.equilibrium_trap_levels(traps)
    if _opt is not None or _eq_traps:
        from dynameta.carriers.bipolar_options import install_option_terms
        install_option_terms(device, region, _opt, fd_enhancement=fd_enhancement, traps=_eq_traps,
                             statistics=_label)

    # --- transient / AC charge: the d(q n)/dt time term of each continuity equation. NCharge =
    #     -q n (electron charge density), PCharge = +q p. Defined ALWAYS: a steady-state DC solve
    #     ignores time_node_model, while a transient (solve type="transient_*") or a small-signal
    #     ssac solve USES it -- it is what makes the carrier-charge (junction / diffusion)
    #     capacitance appear in the ssac admittance and the large-signal transient response. (Was
    #     absent before, so ssac on a DD device saw only the resistive part; verified the diode
    #     junction C then matches the analytic depletion C.)
    _node_with_derivs(device, region, "NCharge", "-ElectronCharge * Electrons", ("Electrons",))
    _node_with_derivs(device, region, "PCharge", "ElectronCharge * Holes", ("Holes",))

    # --- continuity equations (time_node_model -> transient/AC-ready; ignored by a DC solve) ---
    _R.record_region_equation(device, region, name="ElectronContinuityEquation",
                              variable_name="Electrons", edge_model="ElectronCurrent",
                              node_model="ElectronGeneration", time_node_model="NCharge",
                              variable_update="positive")
    _R.record_region_equation(device, region, name="HoleContinuityEquation",
                              variable_name="Holes", edge_model="HoleCurrent",
                              node_model="HoleGeneration", time_node_model="PCharge",
                              variable_update="positive")
    if btbt is not None:                                       # D1d: opt-in BTBT generation
        from dynameta.carriers.tunneling import setup_btbt
        # the Hurkx D factor vanishes at equilibrium only with the region's OWN n p product: n_ie^2
        # under BGN (D1d merge note), n_i^2 otherwise
        setup_btbt(device, region, btbt, n_i="n_ie" if _bgn else "n_i")


# Charge-neutral equilibrium carrier node models (SI). Used both to seed the
# coupled solve and inside the contact pinning expression. The +1.0 floor keeps
# the sqrt/abs well-conditioned in fully-depleted or undoped cells.
CELEC = "(1.0 + 0.5*abs(NetDoping + (NetDoping^2 + 4 * n_i^2)^(0.5)))"
CHOLE = "(1.0 + 0.5*abs(-NetDoping + (NetDoping^2 + 4 * n_i^2)^(0.5)))"


def setup_equilibrium_seed_models(device: str, region: str) -> None:
    """Create IntrinsicElectrons/IntrinsicHoles node models (charge-neutral
    equilibrium) so the staged solve can seed Electrons/Holes before the coupled
    Newton. n0 = majority on n-side, p0 = n_i^2/n0 (symmetric swap on p-side).

    audit R26/R27: a region whose BipolarOptions change the equilibrium (BGN: n0 p0 = n_ie^2;
    incomplete ionization: neutrality with N_D+ / N_A-) carries the numpy charge-neutral root as the
    fixed fields NeutralElectrons / NeutralHoles, and the seed takes THOSE (the same values its ohmic
    contacts pin). Otherwise the closed-form strings below, unchanged."""
    from dynameta.carriers.bipolar_options import options_contact_models
    if options_contact_models(device, region):
        ds.node_model(device=device, region=region, name="IntrinsicElectrons",
                      equation="NeutralElectrons")
        ds.node_model(device=device, region=region, name="IntrinsicHoles", equation="NeutralHoles")
        return
    n0 = "ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(ce=CELEC, ch=CHOLE)
    p0 = "ifelse(NetDoping < 0, {ch}, n_i^2/{ce})".format(ce=CELEC, ch=CHOLE)
    ds.node_model(device=device, region=region, name="IntrinsicElectrons", equation=n0)
    ds.node_model(device=device, region=region, name="IntrinsicHoles", equation=p0)


def fd_shift_model(device: str, region: str) -> "str | None":
    """The name of the FD contact-shift node model if `region` carries one (installed on the region
    by setup_contact_ohmic_bipolar via _fd_contact_shift), else None. audit C-10.

    Staged solvers seed Potential from the charge-neutral equilibrium and MUST use the same offset
    the contact pins; this is how they ask which frame the contact chose.

    audit R26/R27: a region with option-altered equilibrium carries `SeedPsiShift`, the per-node value s
    for which equilibrium_seed_psi_expr(s) reproduces EXACTLY the potential its contacts pin (BGN band
    edge + ionization neutrality + FD correction, from bipolar_options.equilibrium_fields); it takes
    precedence, so every staged solver seeds in the right frame without a signature change."""
    if "SeedPsiShift" in ds.get_node_model_list(device=device, region=region):
        return "SeedPsiShift"
    return ("FDContactShift"
            if "FDContactShift" in ds.get_node_model_list(device=device, region=region) else None)


def equilibrium_seed_psi_expr(shift: "str | None") -> str:
    """The DEVSIM expression for the charge-neutral equilibrium seed potential, in the SAME frame the
    ohmic contact pins (audit C-10). `shift` is `fd_shift_model(...)`; None reproduces the legacy
    Boltzmann string CHARACTER-FOR-CHARACTER.

    SIGN BY DOPING TYPE -- the residual C-10 defect. IntrinsicElectrons is n0 on the n side and
    n_i^2/p0 on the p side, so V_t ln(IntrinsicElectrons/n_i) is +V_t ln(n0/n_i) on the n side and
    -V_t ln(p0/n_i) on the p side: the log term already flips sign with the doping, and the FD
    correction must flip WITH it. Adding a bare +Delta therefore landed the p-side seed 2 V_t Delta
    ABOVE the contact's pinned potential -- measured +0.355 V at eta = 10 (the n side was exact) --
    i.e. a p-type body seeded the coupled Newton in a frame the boundary condition contradicts."""
    if shift is None:
        return "V_t*log(IntrinsicElectrons/n_i)"
    return ("V_t*(log(IntrinsicElectrons/n_i) + ifelse(NetDoping > 0, {s}, -{s}))".format(s=shift))


def _pot_offset(shift: "str | None", ce: str, ch: str) -> str:
    """The ohmic built-in-potential offset expression (audit C-10). `shift` is the FD degeneracy
    node model from `_fd_contact_shift` or None; None reproduces the legacy Boltzmann string
    CHARACTER-FOR-CHARACTER, so a Boltzmann region's contact equation is unchanged."""
    if shift is None:
        return "ifelse(NetDoping > 0, -V_t*log({ce}/n_i), V_t*log({ch}/n_i))".format(ce=ce, ch=ch)
    return ("ifelse(NetDoping > 0, -V_t*(log({ce}/n_i) + {s}), V_t*(log({ch}/n_i) + {s}))"
            .format(ce=ce, ch=ch, s=shift))


def _options_contact_exprs(device: str, contact: str):
    """audit R26/R27: (electron, hole) contact residual strings pinning the option-equilibrium fields
    when the contact's region carries them (BGN and/or incomplete ionization), else None -- in which
    case every contact string below is the legacy one CHARACTER-FOR-CHARACTER. The Potential residual
    in that mode is `Potential - bias - EquilibriumPotential` (derivative literal 1, as before)."""
    from dynameta.carriers.bipolar_options import options_contact_models
    if not options_contact_models(device, _contact_region(device, contact)):
        return None
    return "Electrons - NeutralElectrons", "Holes - NeutralHoles"


def contact_potential_residual(device: str, contact: str, drive_symbol: str,
                               psi_offset_V: float = 0.0) -> str:
    """THE one ohmic-contact rule for the Potential of a bipolar contact (DC and circuit recipes):

        Potential = <drive> + s(r) + psi_eq,own(r)

    s(r) = psi_offset_V is the region's Anderson frame shift (physics_heterojunction, D1a; the region's
    neutral-bulk potential in the common heterostack frame), and psi_eq,own(r) is the built-in potential
    of the contact node IN THE REGION'S OWN FRAME: the bipolar_options.equilibrium_fields root
    (EquilibriumPotential: BGN band-edge split + incomplete-ionization neutrality + the trapped charge of
    charged levels + the FD correction) when the region carries it, else the legacy closed form
    (Boltzmann or FD, audit C-10). The two pieces are additive and disjoint -- s(r) is a constant of the
    UNNARROWED Anderson alignment, psi_eq,own carries every doping-dependent shift -- so nothing is
    counted twice (v0.14 integration). With psi_offset_V = 0 and no option fields the string is the
    legacy one character-for-character."""
    drv = drive_symbol + psi_offset_term(psi_offset_V)
    if _options_contact_exprs(device, contact):
        _fd_root_ceiling_check(device, contact)                # v0.15.0: the default FD path keeps the advisory
        return "Potential - {} - EquilibriumPotential".format(drv)
    shift = _fd_contact_shift(device, contact)                 # audit C-10 (None -> Boltzmann)
    return "Potential - {} + {}".format(drv, _pot_offset(shift, CELEC, CHOLE))


def setup_contact_ohmic_bipolar(device: str, contact: str, *, psi_offset_V: float = 0.0) -> None:
    """Bipolar ohmic contact: pin Potential (with the built-in offset) and pin
    Electrons/Holes to their charge-neutral equilibrium values. The built-in offset follows the
    region's own statistics (audit C-10): Boltzmann when the region was set up with
    fd_enhancement=False, Fermi-Dirac (+ V_t Delta(n0/N_dos)) when it was set up with the g-factor
    on -- the boundary condition used to be Boltzmann either way, contradicting the interior model
    by 0.17 V per degenerate contact at eta = 10. psi_offset_V is the heterostack frame shift s(r) of
    the contact's region (physics_heterojunction; 0 -> the string is unchanged)."""
    ds.set_parameter(device=device, name="{}_bias".format(contact), value=0.0)
    _opt = _options_contact_exprs(device, contact)             # audit R26/R27 (None -> legacy)

    # Potential: Dirichlet + built-in potential offset (the one contact rule)
    cp = "{}_potential_dirichlet".format(contact)
    pot_eq = contact_potential_residual(device, contact, "{}_bias".format(contact), psi_offset_V)
    ds.contact_node_model(device=device, contact=contact, name=cp, equation=pot_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Potential".format(cp), equation="1")
    _R.record_contact_equation(device, contact, name="PotentialEquation",
                               node_model=cp, edge_charge_model="PotentialEdgeFlux")

    # Electrons pinned to n0 (n-side) or n_i^2/p0 (p-side); derivative literal "1".
    ce = "{}_electrons_dirichlet".format(contact)
    elec_eq = "Electrons - ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(
        ce=CELEC, ch=CHOLE) if not _opt else _opt[0]
    ds.contact_node_model(device=device, contact=contact, name=ce, equation=elec_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Electrons".format(ce), equation="1")
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation",
                               node_model=ce, edge_current_model="ElectronCurrent")

    # Holes pinned to p0 (p-side) or n_i^2/n0 (n-side); derivative literal "1".
    ch = "{}_holes_dirichlet".format(contact)
    hole_eq = "Holes - ifelse(NetDoping < 0, {ch}, n_i^2/{ce})".format(
        ce=CELEC, ch=CHOLE) if not _opt else _opt[1]
    ds.contact_node_model(device=device, contact=contact, name=ch, equation=hole_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Holes".format(ch), equation="1")
    _R.record_contact_equation(device, contact, name="HoleContinuityEquation",
                               node_model=ch, edge_current_model="HoleCurrent")


def _check_drive(drive: str, source_name: str, who: str) -> str:
    """Validate a circuit contact's `drive` ("voltage" | "current") against DEVSIM's name-prefix
    element typing (V... = voltage source, I... = current source). Returns source_name unchanged.
    The voltage drive is not name-checked (the pre-v0.14 behaviour, byte-identical)."""
    want = {"voltage": "V", "current": "I"}.get(drive)
    if want is None:
        raise ValueError("{}: drive must be 'voltage' or 'current'; got {!r}".format(who, drive))
    if drive == "current" and not str(source_name).startswith(want):
        raise ValueError("{}: a {} drive needs a source name starting with {!r} (DEVSIM types circuit "
                         "elements by their first letter); got {!r}".format(who, drive, want,
                                                                            source_name))
    return source_name


def setup_contact_ohmic_bipolar_circuit(device: str, contact: str, *,
                                        node_name: str = "vac",
                                        source_name: str = "V1",
                                        v_ac: float = 1.0, drive: str = "voltage",
                                        psi_offset_V: float = 0.0) -> tuple:
    """Circuit-driven bipolar ohmic contact -- the small-signal-AC (ssac) / transient analogue of
    setup_contact_ohmic_bipolar. The Potential is tied to a CIRCUIT NODE `node_name` (driven by an
    AC voltage source `source_name`) instead of a bias PARAMETER, and the Potential + Electron +
    Hole contact currents all couple into that node, so the source current `<source_name>.I` is the
    TOTAL small-signal terminal current. Use this on the contact you want to AC/transient-probe; the
    OTHER contact stays an ordinary grounded setup_contact_ohmic_bipolar. The region must be
    transient-ready (setup_bipolar_region defines the NCharge/PCharge time-node models -- the
    carrier-charge capacitance the ssac admittance needs).

    DC operating point: set the source DC value with ds.circuit_alter(name=source_name, value=V)
    and ramp; acreal=1 is a unit AC excitation. Extract C(f), G(f) with
    ac_analysis.ssac_admittance(frequencies, source_name=source_name). Returns (source_name,
    node_name). (Validated: a reverse-biased diode's ssac junction C matches the analytic depletion
    C_j; see validation/ac_diode.py.)

    drive="current" (v0.14.0, noise_analysis) makes the source a CURRENT source instead (DEVSIM
    infers the element type from the name, so `source_name` must then start with "I", e.g. "I1"):
    the contact is then open-circuit for the small signal, its circuit node voltage `node_name` is
    the free terminal voltage, and noise_analysis.noise_spectrum(output_node=node_name) is the
    open-circuit voltage noise S_V. DEVSIM refuses an AC part on a current source, so such a contact
    carries NO ssac excitation (v_ac is ignored; take Z = 1/Y from a voltage-driven twin). The DC
    operating point is the source's DC CURRENT (ds.circuit_alter(name=source_name, value=I)); skip
    the frozen-carrier Poisson pre-solve (its KCL row has no DC current and is singular) and go
    straight to the coupled Newton from the equilibrium seed (measured: converges in 15 iterations
    on the 1305-node long diode of validation/devsim_noise.py). The default "voltage" is byte-
    identical to the pre-v0.14 helper (the R26/R27 call-log digest gate covers it).

    psi_offset_V: the heterostack frame shift s(r) of the contact's region (as for the DC twin; the
    shared rule contact_potential_residual). 0 -> the string is unchanged."""
    source_name = _check_drive(drive, source_name, "setup_contact_ohmic_bipolar_circuit")
    ds.add_circuit_node(name=node_name, variable_update="default")
    if not (float(v_ac) > 0.0):
        raise ValueError("setup_contact_ohmic_bipolar_circuit: v_ac must be > 0; got {!r}".format(v_ac))
    # audit C6-2: carry the SAME v_ac ssac_admittance divides by (was hardcoded 1.0)
    if drive == "current":                 # DEVSIM refuses an AC part on a current source
        ds.circuit_element(name=source_name, n1=node_name, n2="0", value=0.0)
    else:
        ds.circuit_element(name=source_name, n1=node_name, n2="0", value=0.0,
                           acreal=float(v_ac), acimag=0.0)
    # Potential: Dirichlet tied to the circuit node + built-in offset (reference n_i; FD-corrected
    # when the region's transport uses the FD g-factor -- audit C-10, same rule as the DC twin).
    _opt = _options_contact_exprs(device, contact)             # audit R26/R27 (None -> legacy)
    cp = "{}_potential_circuit".format(contact)
    pot_eq = contact_potential_residual(device, contact, node_name, psi_offset_V)
    ds.contact_node_model(device=device, contact=contact, name=cp, equation=pot_eq)
    ds.contact_node_model(device=device, contact=contact, name="{}:Potential".format(cp),
                          equation="1")
    ds.contact_node_model(device=device, contact=contact, name="{}:{}".format(cp, node_name),
                          equation="-1")
    _R.record_contact_equation(device, contact, name="PotentialEquation", node_model=cp,
                               edge_charge_model="PotentialEdgeFlux", circuit_node=node_name)
    # Electrons / Holes pinned to charge-neutral equilibrium; their currents couple to the node.
    ce = "{}_electrons_circuit".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=ce,
                          equation=_opt[0] if _opt else
                          "Electrons - ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(
                              ce=CELEC, ch=CHOLE))
    ds.contact_node_model(device=device, contact=contact, name="{}:Electrons".format(ce),
                          equation="1")
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation", node_model=ce,
                               edge_current_model="ElectronCurrent", circuit_node=node_name)
    ch = "{}_holes_circuit".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=ch,
                          equation=_opt[1] if _opt else
                          "Holes - ifelse(NetDoping < 0, {ch}, n_i^2/{ce})".format(
                              ce=CELEC, ch=CHOLE))
    ds.contact_node_model(device=device, contact=contact, name="{}:Holes".format(ch),
                          equation="1")
    _R.record_contact_equation(device, contact, name="HoleContinuityEquation", node_model=ch,
                               edge_current_model="HoleCurrent", circuit_node=node_name)
    return source_name, node_name


# =================================================================================================
# v0.15 (plan 4.4.3, decisions 48/49): the QUASI-EQUILIBRIUM carrier closure.
#
# A 1-D MOS capacitor in strong inversion has no DC current path, so its DC state is thermodynamic
# equilibrium: the minority carrier's quasi-Fermi level is the body contact's everywhere. The full
# drift-diffusion row carries that answer through an equation that pins the inversion layer only via
# bulk minority diffusion and SRH generation -- a near-null mode of the DC Jacobian whose equilibrated
# sigma_min falls ~e-fold per V_t of band bending (3.8e-11 at 0.30 V, 2.0e-17 at 0.76 V on 3e23 p-Si),
# so Newton fails near 0.78-0.80 V in double precision (and DG near 0.70 V). Quasi-Fermi / Slotboom
# variables are a change of variables of the SAME Jacobian and do not help (P4 1.1, measured). The
# closure REMOVES the degree of freedom: the carrier's continuity row becomes the algebraic statement
#     electrons:  R_n = phi_ref - [ (psi + D_c + Lambda_n) - V_t (ln(n/N_c) + Delta(n/N_c) + ln(N_c/n_i)) ]
#     holes:      R_p = [ (psi - D_v - Lambda_p) + V_t (ln(p/N_v) + Delta(p/N_v) + ln(N_v/n_i)) ] - phi_ref
# i.e. the carrier's own quasi-Fermi level (the bracket, the QE_n_imref / QE_p_imref node model) equals
# phi_ref. The activity ln u + Delta(u) is physics_heterojunction.setup_log_activity -- the ONE DEVSIM
# form of the Fermi-Dirac built-in correction (closed form with arctan by half-angle reductions; DEVSIM
# has no atan) whose exact derivative g(u)/c is this row's Jacobian (plan 5.6: reused, not rewritten).
# D_c / D_v are the bandgap-narrowing quasi-fields when the region carries them (BgnDc / BgnDv), Lambda
# the density-gradient quantum potential when the region carries that DG carrier.
#
# Discrete exactness (Boltzmann): the zero-current Scharfetter-Gummel edge relation n1/n0 = exp(-v) of
# the full model IS this row's ratio on every edge, so wherever the full model converges the two give
# the same discrete solution (gated to 1e-10, tests/test_quasi_equilibrium.py). With the FD g-factor the
# full model's edge-mean g makes them differ at O(h^2 g'), a discretization difference.
# =================================================================================================

_QE = {"electrons": dict(var="Electrons", eq="ElectronContinuityEquation", dos="N_dos", lam="QLambda",
                         band="BgnDc", sign="+", model="QE_n"),
       "holes": dict(var="Holes", eq="HoleContinuityEquation", dos="N_dos_p", lam="QLambdaP",
                     band="BgnDv", sign="-", model="QE_p")}
_QE_SAVED: dict = {}      # (device, region, eq) -> (eq_registry generation, recorded continuity kwargs)


def quasi_equilibrium_carriers(device: str, region: str) -> tuple:
    """The carriers of `region` currently closed by setup_quasi_equilibrium_carrier (() if none)."""
    out = []
    for key, sp in _QE.items():
        rec = _R.recorded(device, sp["eq"], scope="region", loc=region)
        if rec and rec[-1]["kwargs"].get("node_model") == sp["model"]:
            out.append(key)
    return tuple(out)


def _ohmic_pins(device: str, region: str, eq: str) -> list:
    """Contacts of `region` whose recorded contact equation `eq` PINS the carrier (a node_model)."""
    out = []
    for c in ds.get_contact_list(device=device):
        if ds.get_region_list(device=device, contact=c)[0] != region:
            continue
        if any(e["kwargs"].get("node_model") for e in _R.recorded(device, eq, scope="contact", loc=c)):
            out.append(c)
    return out


def setup_quasi_equilibrium_carrier(device: str, region: str, carrier: str, *, contact: str,
                                    imref_shift: "str | None" = None,
                                    allow_imref_split: bool = False) -> str:
    """Replace `carrier`'s continuity row on `region` ("electrons" | "holes") by the QUASI-EQUILIBRIUM
    closure: at every node its quasi-Fermi level equals that of the ohmic `contact` (section header
    above). Returns the name of the node model holding the carrier's quasi-Fermi level [V]
    (QE_n_imref / QE_p_imref). Opt-in: nothing changes unless called.

    WHEN TO USE IT. A carrier with no DC current path -- the minority carrier of an MOS capacitor
    (1-D or a laterally uniform 2-D slice) in depletion and inversion, the low-frequency (quasi-static)
    C-V, a MOSFET channel slice with an imposed imref (imref_shift). There the closure is EXACT (the DC
    state is equilibrium) and it removes the near-null mode that makes the full model fail in strong
    inversion: on 3e23 m^-3 p-Si the classical DD fails at 0.78-0.80 V of band bending and the bipolar
    density-gradient at 0.70 V; closed, both reach >= 1.3 V (tests/test_quasi_equilibrium.py,
    validation/bipolar_density_gradient.py GATE G). NOT a transport model: any device that carries a
    current of this carrier (a diode, a MOSFET with source and drain, a photodetector) must keep the
    full continuity row -- the closure then gives a wrong current-free answer.

    What follows from it:
      * DC and ssac: allowed. The row has no time term, i.e. the carrier follows the bias
        instantaneously -- ssac on a closed device IS the low-frequency C-V (gated against dQ/dV).
      * transient: the carrier is quasi-static (infinitely fast generation); fine for slow ramps of a
        capacitor, wrong for anything that resolves the minority response time.
      * noise_analysis REFUSES a closed region (the row has no Scharfetter-Gummel current; the noise
        sources are not defined).
      * get_contact_current on the closed carrier's equation is meaningless (no continuity).
      * Gummel (dc_solve method="gummel") freezes/thaws the closed row like any carrier row.

    contact : the ohmic contact (setup_contact_ohmic_bipolar) whose quasi-Fermi level the carrier
        shares: phi_ref = <contact>_bias + the contact node's own fixed imref offset, READ from the
        contact node at setup (so it includes everything the contact pins: the CELEC/CHOLE +1 floor,
        the FD built-in correction, BGN, a degenerate contact's Boltzmann product). Call AFTER a
        converged DC solve of the region (the contact rows must hold), after any BipolarOptions /
        traps and after physics_density_gradient.setup_bipolar_dg when DG is used (the carrier's
        Lambda then enters; setup_bipolar_dg refuses to run on an already-closed carrier). A
        circuit-driven contact has no bias parameter and is refused.
    imref_shift : optional name of a DEVICE parameter V_ch [V] added to phi_ref -- the imref split of a
        MOSFET channel slice (phi_n = phi_body + V_ch).
    allow_imref_split : the region has more than one ohmic contact pinning this carrier. The closure
        then forces ONE quasi-Fermi level on the whole region while the other contacts pin their own
        (no current flows between them): refused unless you pass True, which states that you know
        (e.g. a source/drain-tied channel slice at equal bias)."""
    if carrier not in _QE:
        raise ValueError("setup_quasi_equilibrium_carrier: carrier must be 'electrons' or 'holes', got "
                         "{!r}".format(carrier))
    sp = _QE[carrier]
    who = "setup_quasi_equilibrium_carrier({!r}, {!r})".format(region, carrier)
    nodes = set(ds.get_node_model_list(device=device, region=region))
    rec = _R.recorded(device, sp["eq"], scope="region", loc=region)
    if "Holes" not in nodes or len(rec) != 1:
        raise ValueError("{}: a bipolar drift-diffusion region (setup_bipolar_region) is "
                         "required".format(who))
    if rec[0]["kwargs"].get("node_model") == sp["model"]:
        raise ValueError("{}: the carrier is already closed".format(who))
    if ds.get_region_list(device=device, contact=contact)[0] != region:
        raise ValueError("{}: contact {!r} is not on region {!r}".format(who, contact, region))
    pins = _ohmic_pins(device, region, sp["eq"])
    if contact not in pins:
        raise ValueError("{}: contact {!r} does not pin {} (an ohmic contact is required)".format(
            who, contact, sp["var"]))
    bias = "{}_bias".format(contact)
    if (_R.circuit_driven_contacts(device).get(contact)
            or bias not in ds.get_parameter_list(device=device)):
        raise ValueError("{}: contact {!r} has no bias parameter {!r} (circuit-driven contacts are not "
                         "supported as the quasi-Fermi reference)".format(who, contact, bias))
    others = [c for c in pins if c != contact]
    if others and not allow_imref_split:
        raise ValueError(
            "{}: region {!r} has further ohmic contacts pinning {} ({}). The closure imposes ONE "
            "quasi-Fermi level (contact {!r}'s) on the whole region, so it cannot carry a current "
            "between them -- it is not a transport model. Keep the continuity row, or pass "
            "allow_imref_split=True if a single imref is what you mean.".format(
                who, region, sp["var"], ", ".join(repr(c) for c in others), contact))
    if imref_shift is not None and imref_shift not in ds.get_parameter_list(device=device):
        raise ValueError("{}: imref_shift {!r} is not a device parameter".format(who, imref_shift))

    from dynameta.carriers.physics_heterojunction import setup_log_activity
    fd = float(ds.get_parameter(device=device, region=region, name="fd_enhancement")) > 0.0
    act = setup_log_activity(device, region, sp["var"], sp["dos"], fd=fd)
    lam = sp["lam"] if sp["lam"] in nodes else None
    band = sp["band"] if sp["band"] in nodes else None
    s = sp["sign"]
    psi_eff = "Potential" + "".join(" {} {}".format(s, t) for t in (band, lam) if t)
    vt_act = "V_t*({} + log({}/n_i))".format(act, sp["dos"])
    imref = sp["model"] + "_imref"
    ds.node_model(device=device, region=region, name=imref,
                  equation="({}) {} {}".format(psi_eff, "-" if s == "+" else "+", vt_act))
    # the fixed offset of the contact node's own quasi-Fermi level against its bias
    from dynameta.carriers.physics_density_gradient import _contact_nodes
    idx = _contact_nodes(device, region, contact)
    phi_c = np.asarray(ds.get_node_model_values(device=device, region=region, name=imref))[idx]
    c_c = np.asarray(ds.get_node_model_values(device=device, region=region, name=sp["var"]))[idx]
    if not (np.all(np.isfinite(phi_c)) and np.all(c_c > 0.0)):
        raise ValueError("{}: the contact node's state is not a converged solution -- solve the region "
                         "first".format(who))
    offset = float(np.mean(phi_c)) - float(ds.get_parameter(device=device, name=bias))
    off_name = sp["model"] + "_offset"
    ds.set_parameter(device=device, region=region, name=off_name, value=offset)
    phi_ref = "({} + {}{})".format(bias, off_name, " + " + imref_shift if imref_shift else "")
    row = ("{} - {}".format(phi_ref, imref) if s == "+" else "{} - {}".format(imref, phi_ref))
    m = sp["model"]
    ds.node_model(device=device, region=region, name=m, equation=row)
    # exact Jacobian, written out: dR/dc = V_t d(activity)/dc (= V_t g(u)/c), dR/dpsi = -/+1,
    # dR/dLambda = -1 (both carriers: R_n = phi_ref - psi - Lambda_n + ..., R_p = psi - Lambda_p + ...)
    ds.node_model(device=device, region=region, name="{}:{}".format(m, sp["var"]),
                  equation="V_t*{}:{}".format(act, sp["var"]))
    ds.node_model(device=device, region=region, name="{}:Potential".format(m),
                  equation="-1" if s == "+" else "1")
    if lam:
        ds.node_model(device=device, region=region, name="{}:{}".format(m, lam), equation="-1")
    _QE_SAVED[(device, region, sp["eq"])] = (_R.generation(device), dict(rec[0]["kwargs"]))
    ds.delete_equation(device=device, region=region, name=sp["eq"])
    _R.forget(device, sp["eq"], loc=region, scope="region")
    _R.record_region_equation(device, region, name=sp["eq"], variable_name=sp["var"], node_model=m,
                              variable_update="positive")
    return imref


def delete_quasi_equilibrium_carrier(device: str, region: str, carrier: str) -> None:
    """Undo setup_quasi_equilibrium_carrier: re-create the carrier's recorded continuity equation
    (every kwarg as it was: flux, generation composite, time term). The QE node models stay defined
    (unused). Re-solve afterwards; the closed solution is the full model's equilibrium seed."""
    if carrier not in _QE:
        raise ValueError("delete_quasi_equilibrium_carrier: carrier must be 'electrons' or 'holes'")
    sp = _QE[carrier]
    saved = _QE_SAVED.get((device, region, sp["eq"]))
    stale = saved is None or saved[0] != _R.generation(device)
    if carrier not in quasi_equilibrium_carriers(device, region) or stale:
        raise ValueError("delete_quasi_equilibrium_carrier: {!r} is not closed on region {!r}".format(
            carrier, region))
    ds.delete_equation(device=device, region=region, name=sp["eq"])
    _R.forget(device, sp["eq"], loc=region, scope="region")
    _R.record_region_equation(device, region, **saved[1])
    del _QE_SAVED[(device, region, sp["eq"])]
