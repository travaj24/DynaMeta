"""
Bipolar drift-diffusion (electrons + holes + Shockley-Read-Hall recombination),
DEVSIM-native, in SI units. Opt-in extension of the electrons-only
physics_drift_diffusion.py -- mirrors its SI conventions and the FERMI-DIRAC
diffusion-enhancement (FD g-factor), and reimplements the holes / SRH / bipolar
Poisson / charge-neutral contacts in SI (the bundled simple_physics.py is CGS).

Solution variables: Potential, Electrons, Holes (three coupled fields).

DEVSIM sign convention (residual = q*divergence form), per docs/implementation_notes.md:
  Jn = +q mu_n EdgeInvLength V_t [ Electrons@n1 B(vdiff) + Electrons@n1 vdiff
                                   - Electrons@n0 B(vdiff) ]            (electrons)
  Jp = -q mu_p EdgeInvLength V_t [ Holes@n1 B(vdiff) - Holes@n0 B(vdiff)
                                   - Holes@n0 vdiff ]                   (holes; q->-q
                                   AND the vdiff drift term on the @n0 node)
  Poisson node charge : PotentialNodeCharge = -q (Holes - Electrons + NetDoping)
  SRH source          : USRH = (n p - n_i^2)/(taup (n + n1) + taun (p + p1))
                        ElectronGeneration = -q USRH  (into ElectronContinuityEquation)
                        HoleGeneration     = +q USRH  (into HoleContinuityEquation)

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
    i.e. the error is under 1 % everywhere degeneracy matters, but "< 0.5 % for eta >= 15" is an
    artifact of the AH frame; against exact quadrature it does not reach 0.5 % until eta ~ 25.
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

Staged solve: see solve_bipolar_diode in validation/bipolar_diode.py:
  (1) potential-only pre-solve (freeze carriers); (2) seed Electrons/Holes from the
  Boltzmann equilibrium node models; (3) coupled 3-variable Newton; (4) bias ramp.
  variable_update: Potential "log_damp", Electrons/Holes "positive".
"""

from __future__ import annotations

import warnings

import numpy as np
import devsim as ds

from dynameta.carriers.physics_equilibrium import (
    Q_E, EPS0, V_T, _poisson_edge_models, require_positive, invert_F12)
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.eq_registry import (edge_with_derivs as _edge_with_derivs,
                                           node_with_derivs as _node_with_derivs)
from dynameta.carriers.einstein import g_einstein, g_expr_devsim

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
                            fd_enhancement: bool = True) -> float:
    """The built-in potential [V] the bipolar ohmic contact PINS for a node of SIGNED net doping
    `net_doping_m3`, in the contact's own frame (n-type: +V_t ln(n0/n_i) plus the FD correction).

    THE single source for anything that has to reference another terminal to the ohmic contact's
    frame -- notably devsim_3d's gate offset phi_bi, which re-derived the Boltzmann log() and would
    otherwise re-open the very frame mismatch it exists to close (audit C2-1) as soon as the body is
    degenerate (audit C-10: 0.178 V at eta = 10). fd_enhancement=False returns the plain Boltzmann
    value, matching the contact expression exactly."""
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
            RuntimeWarning, stacklevel=3)
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
            RuntimeWarning, stacklevel=3)
    # Delta is a function of u alone, and a doping profile has few distinct values (usually one or
    # two), so evaluate the quadrature on the UNIQUE ratios: a 1e6-node 3-D body region would
    # otherwise allocate an (nnodes, 96) quadrature array.
    uq, inv = np.unique(u, return_inverse=True)
    shift = np.asarray(fd_potential_shift(uq), dtype=np.float64).reshape(-1)[inv]
    ds.node_solution(device=device, region=region, name=name)
    ds.set_node_values(device=device, region=region, name=name,
                       values=[float(v) for v in shift])
    return name


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
        RuntimeWarning, stacklevel=stacklevel)
    return worst_u


def _g_expr(var: str, s: str, dos: str = "N_dos") -> str:
    """g(var{s}/dos) as a DEVSIM edge expression -- the rational fit + its coefficients live
    in carriers/einstein.g_expr_devsim (the single source, shared with physics_drift_diffusion;
    same coefficients for electrons and holes). `dos` is the band DOS parameter name (N_dos ~ N_c
    for electrons, N_dos_p ~ N_v for holes)."""
    return g_expr_devsim(var, dos, s)


def setup_bipolar_region(device: str, region: str, *,
                         eps_static: float, n_dos_m3: float, n_i_m3: float,
                         mobility_n_m2Vs: float, mobility_p_m2Vs: float,
                         tau_n_s: float, tau_p_s: float,
                         fd_enhancement: bool = True,
                         n_dos_p_m3: "float | None" = None) -> None:
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
    ds.set_parameter(device=device, region=region, name="n1", value=n_i_m3)  # SRH trap
    ds.set_parameter(device=device, region=region, name="p1", value=n_i_m3)
    # audit C-10: the contact builders are called later and separately, and the ohmic built-in
    # potential must follow the SAME statistics as the transport in this region. Record the choice
    # here (1.0 = Fermi-Dirac g-factor, 0.0 = Boltzmann) so setup_contact_ohmic_bipolar* can read it.
    ds.set_parameter(device=device, region=region, name="fd_enhancement",
                     value=1.0 if fd_enhancement else 0.0)

    # --- solution variables ---
    ds.node_solution(device=device, region=region, name="Potential")
    ds.node_solution(device=device, region=region, name="Electrons")
    ds.node_solution(device=device, region=region, name="Holes")
    ds.edge_from_node_model(device=device, region=region, node_model="Potential")
    ds.edge_from_node_model(device=device, region=region, node_model="Electrons")
    ds.edge_from_node_model(device=device, region=region, node_model="Holes")

    # --- bipolar Poisson node charge: -q (p - n + NetDoping) ---
    pne = "-ElectronCharge * kahan3(Holes, -Electrons, NetDoping)"
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
    if fd_enhancement:
        g_n = "(0.5*({} + {}))".format(_g_expr("Electrons", "@n0"), _g_expr("Electrons", "@n1"))
        g_p = "(0.5*({} + {}))".format(_g_expr("Holes", "@n0", dos="N_dos_p"),
                                       _g_expr("Holes", "@n1", dos="N_dos_p"))
        _edge_with_derivs(device, region, "g_enh", g_n, ("Electrons",))
        _edge_with_derivs(device, region, "g_enh_p", g_p, ("Holes",))
    else:
        ds.edge_model(device=device, region=region, name="g_enh", equation="1.0")
        ds.edge_model(device=device, region=region, name="g_enh_p", equation="1.0")

    # --- electron current (FD-scaled Scharfetter-Gummel) ---
    # vdiff_g = vdiff/g_enh ; Bern_g = B(vdiff_g) ; Jn scaled by g_enh.
    _edge_with_derivs(device, region, "vdiff_g", "vdiff / g_enh",
                      ("Potential", "Electrons"))
    _edge_with_derivs(device, region, "Bern_g", "B(vdiff_g)",
                      ("Potential", "Electrons"))
    jn = ("ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh*"
          "kahan3(Electrons@n1*Bern_g, Electrons@n1*vdiff_g, -Electrons@n0*Bern_g)")
    _edge_with_derivs(device, region, "ElectronCurrent", jn,
                      ("Electrons", "Potential"))            # Jn has no Holes dependence (no dead dJn/dp)

    # --- hole current (q -> -q; vdiff drift term moved to @n0) ---
    # vdiff_gp = vdiff/g_enh_p ; Bern_gp = B(vdiff_gp) ; Jp scaled by g_enh_p.
    _edge_with_derivs(device, region, "vdiff_gp", "vdiff / g_enh_p",
                      ("Potential", "Holes"))
    _edge_with_derivs(device, region, "Bern_gp", "B(vdiff_gp)",
                      ("Potential", "Holes"))
    jp = ("-ElectronCharge*mu_p*EdgeInverseLength*V_t*g_enh_p*"
          "kahan3(Holes@n1*Bern_gp, -Holes@n0*Bern_gp, -Holes@n0*vdiff_gp)")
    _edge_with_derivs(device, region, "HoleCurrent", jp,
                      ("Holes", "Potential"))                # Jp has no Electrons dependence

    # --- SRH recombination: one node model into BOTH continuity equations ---
    usrh = "(Electrons*Holes - n_i^2)/(taup*(Electrons + n1) + taun*(Holes + p1))"
    _node_with_derivs(device, region, "USRH", usrh, ("Electrons", "Holes"))
    gn = "-ElectronCharge * USRH"
    gp = "+ElectronCharge * USRH"
    _node_with_derivs(device, region, "ElectronGeneration", gn, ("Electrons", "Holes"))
    _node_with_derivs(device, region, "HoleGeneration", gp, ("Electrons", "Holes"))

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


# Charge-neutral equilibrium carrier node models (SI). Used both to seed the
# coupled solve and inside the contact pinning expression. The +1.0 floor keeps
# the sqrt/abs well-conditioned in fully-depleted or undoped cells.
CELEC = "(1.0 + 0.5*abs(NetDoping + (NetDoping^2 + 4 * n_i^2)^(0.5)))"
CHOLE = "(1.0 + 0.5*abs(-NetDoping + (NetDoping^2 + 4 * n_i^2)^(0.5)))"


def setup_equilibrium_seed_models(device: str, region: str) -> None:
    """Create IntrinsicElectrons/IntrinsicHoles node models (charge-neutral
    equilibrium) so the staged solve can seed Electrons/Holes before the coupled
    Newton. n0 = majority on n-side, p0 = n_i^2/n0 (symmetric swap on p-side)."""
    n0 = "ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(ce=CELEC, ch=CHOLE)
    p0 = "ifelse(NetDoping < 0, {ch}, n_i^2/{ce})".format(ce=CELEC, ch=CHOLE)
    ds.node_model(device=device, region=region, name="IntrinsicElectrons", equation=n0)
    ds.node_model(device=device, region=region, name="IntrinsicHoles", equation=p0)


def fd_shift_model(device: str, region: str) -> "str | None":
    """The name of the FD contact-shift node model if `region` carries one (installed on the region
    by setup_contact_ohmic_bipolar via _fd_contact_shift), else None. audit C-10.

    Staged solvers seed Potential from the charge-neutral equilibrium and MUST use the same offset
    the contact pins; this is how they ask which frame the contact chose."""
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


def setup_contact_ohmic_bipolar(device: str, contact: str) -> None:
    """Bipolar ohmic contact: pin Potential (with the built-in offset) and pin
    Electrons/Holes to their charge-neutral equilibrium values. The built-in offset follows the
    region's own statistics (audit C-10): Boltzmann when the region was set up with
    fd_enhancement=False, Fermi-Dirac (+ V_t Delta(n0/N_dos)) when it was set up with the g-factor
    on -- the boundary condition used to be Boltzmann either way, contradicting the interior model
    by 0.17 V per degenerate contact at eta = 10."""
    ds.set_parameter(device=device, name="{}_bias".format(contact), value=0.0)
    bias = "{}_bias".format(contact)
    shift = _fd_contact_shift(device, contact)                 # audit C-10 (None -> Boltzmann)

    # Potential: Dirichlet + built-in potential offset (reference n_i; FD-corrected when the
    # region's transport uses the FD g-factor).
    cp = "{}_potential_dirichlet".format(contact)
    pot_eq = "Potential - {b} + {off}".format(b=bias, off=_pot_offset(shift, CELEC, CHOLE))
    ds.contact_node_model(device=device, contact=contact, name=cp, equation=pot_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Potential".format(cp), equation="1")
    _R.record_contact_equation(device, contact, name="PotentialEquation",
                               node_model=cp, edge_charge_model="PotentialEdgeFlux")

    # Electrons pinned to n0 (n-side) or n_i^2/p0 (p-side); derivative literal "1".
    ce = "{}_electrons_dirichlet".format(contact)
    elec_eq = "Electrons - ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(
        ce=CELEC, ch=CHOLE)
    ds.contact_node_model(device=device, contact=contact, name=ce, equation=elec_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Electrons".format(ce), equation="1")
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation",
                               node_model=ce, edge_current_model="ElectronCurrent")

    # Holes pinned to p0 (p-side) or n_i^2/n0 (n-side); derivative literal "1".
    ch = "{}_holes_dirichlet".format(contact)
    hole_eq = "Holes - ifelse(NetDoping < 0, {ch}, n_i^2/{ce})".format(
        ce=CELEC, ch=CHOLE)
    ds.contact_node_model(device=device, contact=contact, name=ch, equation=hole_eq)
    ds.contact_node_model(device=device, contact=contact,
                          name="{}:Holes".format(ch), equation="1")
    _R.record_contact_equation(device, contact, name="HoleContinuityEquation",
                               node_model=ch, edge_current_model="HoleCurrent")


def setup_contact_ohmic_bipolar_circuit(device: str, contact: str, *,
                                        node_name: str = "vac",
                                        source_name: str = "V1",
                                        v_ac: float = 1.0) -> tuple:
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
    C_j; see validation/ac_diode.py.)"""
    ds.add_circuit_node(name=node_name, variable_update="default")
    if not (float(v_ac) > 0.0):
        raise ValueError("setup_contact_ohmic_bipolar_circuit: v_ac must be > 0; got {!r}".format(v_ac))
    # audit C6-2: carry the SAME v_ac ssac_admittance divides by (was hardcoded 1.0)
    ds.circuit_element(name=source_name, n1=node_name, n2="0", value=0.0,
                       acreal=float(v_ac), acimag=0.0)
    # Potential: Dirichlet tied to the circuit node + built-in offset (reference n_i; FD-corrected
    # when the region's transport uses the FD g-factor -- audit C-10, same rule as the DC twin).
    shift = _fd_contact_shift(device, contact)
    cp = "{}_potential_circuit".format(contact)
    pot_eq = "Potential - {n} + {off}".format(n=node_name, off=_pot_offset(shift, CELEC, CHOLE))
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
                          equation="Electrons - ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(
                              ce=CELEC, ch=CHOLE))
    ds.contact_node_model(device=device, contact=contact, name="{}:Electrons".format(ce),
                          equation="1")
    _R.record_contact_equation(device, contact, name="ElectronContinuityEquation", node_model=ce,
                               edge_current_model="ElectronCurrent", circuit_node=node_name)
    ch = "{}_holes_circuit".format(contact)
    ds.contact_node_model(device=device, contact=contact, name=ch,
                          equation="Holes - ifelse(NetDoping < 0, {ch}, n_i^2/{ce})".format(
                              ce=CELEC, ch=CHOLE))
    ds.contact_node_model(device=device, contact=contact, name="{}:Holes".format(ch),
                          equation="1")
    _R.record_contact_equation(device, contact, name="HoleContinuityEquation", node_model=ch,
                               edge_current_model="HoleCurrent", circuit_node=node_name)
    return source_name, node_name
