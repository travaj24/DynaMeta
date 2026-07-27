"""
Default optical solver: scattered-field complex HCurl on the periodic unit cell,
HalfSpace PML in z, BDDC+GMRes (or UMFPACK). solve_fem is the workhorse (it takes an
already-assembled eps CoefficientFunction); the pipeline drives it through the thin
_fem_optical_solver wrapper, which is the callable that satisfies the core.interfaces
OpticalSolver seam.

Incidence: a plane wave in the x-z plane (azimuth phi=0). Oblique incidence
(theta != 0) is supported for s-polarization (E along y) by two selectable
formulations (module global _OBLIQUE_FORMULATION), both reducing EXACTLY to the
proven normal-incidence curl-curl form at theta=0:

  "phase_in_space" (default): solve for the PHYSICAL field E on a QUASI-periodic
      HCurl space ng.Periodic(HCurl, phase=[exp(i*kx*Px)]*n_px + [1]*n_py). The
      identification order is f0(x=0).Identify(p(x=Px), tx=+Px), so master=x=0,
      minion=x=Px and the Bloch rule phase[idnr]=exp(i*k_par.(r_minion-r_master))
      = exp(+i*kx*Px). The curl operator is the GENUINE curl, so the standard
      stretched-coordinate HalfSpace z-PML transforms it exactly (this is the key
      advantage over the envelope route, whose algebraic transverse-phase term the
      z-PML cannot transform consistently). R/T are demodulated by exp(-i*kx*x).

  "envelope": solve for the envelope u with E=u*exp(i*kx*x), u plain-periodic, the
      transverse phase carried by a modified curl curl(u)+i*k_par x u. Cleaner
      periodicity but the z-PML mishandles the algebraic cross term at oblique
      (R is under-captured) -- kept for reference/diagnostics.

p-pol (TM, E in the x-z plane) oblique and conical s-pol (azimuth phi != 0) are also
implemented (each tmm-validated at phi=0 / phi-invariance).

PML is the ordinary normal z-stretch (alpha=1j CONSTANT; it is NOT angle-aware, so
energy conservation degrades with angle -- validated to ~1% through 30 deg). solve_fem
emits a runtime advisory at oblique incidence -- ONCE PER PROCESS, since it describes the
PML and not the individual solve (audit F-15) -- and RAISES above ~50 deg; the OpticalSpec
also caps the polar angle. Every advisory/quality warning this module emits carries the
FEMDiagnosticWarning category, so a sweep can silence exactly that class
(warnings.filterwarnings('ignore', category=FEMDiagnosticWarning)) or re-enable the
de-duplicated ones (warnings.simplefilter('always', FEMDiagnosticWarning)).
Oblique REQUIRES a vacuum/air incidence medium (n_super=1): the in-plane wavevector
kx=k0 sin(theta) uses the vacuum dispersion, so solve_fem RAISES on a non-vacuum
superstrate at angle rather than returning a silently-wrong result. Validated against
the `tmm` library (validation/oblique_vs_tmm.py).

POLARIZATION VOCABULARY (audit V-8): this module speaks {'x', 'y', 'p'} -- the LAB AXIS of the
incident E ('y' = s-pol, 'p' = p-pol, 'x' = E along lab x, transverse only at normal incidence). It
is one of five spellings in the repo -- {'s','p'} is the PLANE-OF-INCIDENCE spelling (tmm_reference,
resonance, nonlocal_tmm, shg_fem's closed forms, the oblique 2-D FDTD), {'te','tm'} the lumenairy
grating bridge's, the integer `row` 0/1 the differentiable Berreman/RCWA/PMM forwards', and
`pol_axis` hydro_fem's 2-D in-plane axis. The map, the `normalize_pol` converter and the
normal-incidence / azimuth caveats live in `dynameta.core.polarization`. The set ACCEPTED here is
UNCHANGED; acceptance unification (b), the V-8 follow-on, widened only the two PLANE-OF-INCIDENCE
families ({'s','p'} and {'te','tm'}), whose aliases name the same physical mode in every geometry
they cover; this vocabulary's crossings depend on the azimuth (or have no image at all), so they
stay STRICT and are made explicitly, through normalize_pol. At azimuth phi != 0 THIS module reads
'y' as the ROTATED s-hat, while the lumenairy bridge reads the same letter as lab row 1 -- the
documented split, and why the bridge refuses conical.
"""

from __future__ import annotations

import cmath
import math
import os
import re
import threading
import time
import warnings

from dataclasses import dataclass

import numpy as np
import ngsolve as ng

from typing import TYPE_CHECKING

from dynameta.constants import C_LIGHT, EPS0
from dynameta.core.interfaces import OpticalResult
from dynameta.optics.ngsolve_layered import OpticalGeometry, S

if TYPE_CHECKING:                       # type-only; OpticalSpec lives in geometry (no runtime dep)
    from dynameta.geometry.specs import OpticalSpec

# Oblique formulation: "phase_in_space" (physical field, genuine curl, standard
# PML -- the validated route) or "envelope" (plain-periodic envelope, modified
# curl -- diagnostic only). Both are identical at normal incidence. This is a module-level
# switch (NOT a solve_fem kwarg): the "envelope" branch is reachable only by editing it here, so it
# is effectively dead on the public path -- kept for reference/diagnostics, not for production use.
_OBLIQUE_FORMULATION = "phase_in_space"
# Sign of the transverse-phase term on the TEST envelope's modified curl (envelope
# route only): trial carries exp(+i k_par.r) (+kcross), test the conjugate (-kcross).
_TEST_KCROSS_SIGN = -1.0
# Relative-residual threshold above which the two-wave (up/down) R/T fit is flagged unreliable: the
# probe band is then not a clean up/down field (super/substrate buffer too thin -> undecayed
# diffraction orders, or PML leak-back). Conservative -- the real failure mode gives O(1) residuals,
# while a clean propagating 0-order fits to far below this.
#
# MEASURED (audit F-15 remediation, 2026-07-26). The audit reported this threshold false-firing at
# 9.7e-2 on "a plain gold/air cell"; re-measured at HEAD on that exact cell (400 nm square period,
# 200 nm gold eps=-40+2.5j under a 300 nm air cap, umfpack, order 2), sweeping the mesh:
#     maxh scale 1.00 (78 mesh vertices):  residual 2.6e-2 / 2.1e-2   -> 0 fit warnings
#     maxh scale 0.60 (294 vertices):      residual 5.8e-4 / 1.3e-3   -> 0 fit warnings
#     maxh scale 0.40 (839 vertices):      residual 2.7e-4 / 4.3e-4   -> 0 fit warnings
# i.e. it does not fire at all any more -- the wave-1/2 periodicity/Bloch-sampling fixes (F-1, F-2,
# F-6) removed the field pollution that produced the 9.7e-2 reading. The threshold is NOT raised:
# 5e-2 still sits an order of magnitude above the worst converged residual, and the scale-1.00 mesh
# above is not "ordinary" but grossly under-resolved -- it returns R=0.30/T=0.257 for a 200 nm gold
# film whose converged answer is R=0.959/T=1e-4, so a warning there is a TRUE positive, not fatigue.
_FIT_RELRES_WARN = 5e-2

# Validated oblique envelope of the fixed-alpha HalfSpace z-PML, shared by solve_fem
# (_incidence_geometry) and solve_fem_sourced (audit F-13: the sourced path used to carry NONE of
# the incidence guards). Above this the PML reflection grows fast enough to VIOLATE energy
# conservation (R+T ~ 1.17 at 60 deg), so both paths RAISE rather than return a silently-wrong
# number. The sourced path only has k_par (not an OpticalSpec angle), so it compares SINES --
# sin(theta) = |k_par| / (n_super k0) -- which avoids an asin round-trip re-raising at exactly the cap.
# Suppression (in nepers) the probe grid must give the FIRST aliased Fourier order at the nearest
# probe plane (audit F-10). e^-3 = 5%: the old rule only required the alias to be evanescent, which
# near cutoff left it at 0.988 of its structure-face amplitude over the 50 nm standoff.
_ALIAS_DECAY_NEPERS = 3.0
# Hard cap on the per-direction probe-grid size the decay criterion may ask for (audit F-10
# residual). Point evaluations cost N^2 and the criterion grows as ~1/standoff, so a thin buffer
# runs away: 10 nm pad + 1000 nm cell -> (48, 48), i.e. 64x the evaluations and ~+40% wall on a
# shipped solve. 12x12 is 4x the legacy 6x6 cost and covers every shipped fixture measured; past it
# _probe_grid_sizes warns and tells the caller to thicken the buffer instead.
_PROBE_GRID_MAX = 12
_FEM_THETA_MAX_DEG = 50.0
_FEM_SIN_THETA_MAX = math.sin(math.radians(_FEM_THETA_MAX_DEG))
# The angle advisory itself (emitted ONCE PER PROCESS under the tag "oblique_pml" from either
# entry point -- it describes the PML, not the solve).
_OBLIQUE_PML_ADVICE = (
    "oblique incidence: the fixed-alpha HalfSpace z-PML is not angle-aware. Measured vs tmm: "
    "~1% through 30 deg, ~3% at 45 deg. Treat oblique FEM R/T as approximate and stay at or "
    "below ~45 deg for quantitative work (the solver raises above ~50 deg). [reported once "
    "per process; re-enable with warnings.simplefilter('always', FEMDiagnosticWarning)]")


class FEMDiagnosticWarning(UserWarning):
    """Category for solve_fem's ADVISORY diagnostics (regime advisories, fit/energy-closure quality
    flags) as distinct from a plain UserWarning about a caller mistake (audit F-15).

    A 40-wavelength oblique sweep used to emit one identical PML advisory per solve with no new
    information after the first. Two things fix that: the advisories are now emitted at most ONCE
    PER PROCESS per distinct advisory (see :data:`_ADVISED_ONCE`), and every remaining per-solve
    quality flag carries THIS category, so a caller who wants them all can re-enable them
    (``warnings.simplefilter('always', FEMDiagnosticWarning)``) and a caller running a sweep can
    silence exactly this class without also silencing NumPy/NGSolve warnings:

        warnings.filterwarnings('ignore', category=FEMDiagnosticWarning)
    """


# Advisories already emitted in this process (audit F-15): keyed by a short tag, NOT by the message
# text, so a per-solve numeric detail cannot defeat the de-duplication. Mirrors the
# `_symmetry_hinted` once-per-process pattern in ngsolve_layered. Exposed (and clearable) so a test
# can assert the advisory fires the first time.
_ADVISED_ONCE = set()
_ADVISED_LOCK = threading.Lock()


def _advise_once(tag: str, message: str, stacklevel: int = 3) -> bool:
    """Emit `message` as a FEMDiagnosticWarning the FIRST time `tag` is seen in this process.
    Returns True if it warned. Thread-safe (a threaded sweep must not emit N copies)."""
    with _ADVISED_LOCK:
        if tag in _ADVISED_ONCE:
            return False
        _ADVISED_ONCE.add(tag)
    warnings.warn(message, FEMDiagnosticWarning, stacklevel=stacklevel)
    return True


# Per-solve accumulator for the two-wave fit residuals (audit F-15): solve_fem opens one, every
# _lstsq_2wave call inside appends, and solve_fem emits at most ONE aggregated warning naming every
# offending band instead of one warning per band (a p-pol solve fits FOUR bands). Thread-local so a
# threaded sweep cannot cross-attribute residuals between concurrent solves; a _lstsq_2wave call
# made OUTSIDE a solve_fem (direct use, solve_fem_sourced) still warns immediately.
_fit_ctx = threading.local()


def _bloch_z_samples(geo: OpticalGeometry):
    """z sample points (nm) for the Bloch-idnr direction probe: 18 fractions of the GLOBAL stack
    span PLUS three fractions of EVERY region z-interval (audit F-6). The global fractions alone
    can give a thin layer ZERO samples (a 5 nm layer in a ~2.5 um stack sits between two global
    samples), leaving its identification classified from numerical noise. Exposed as a helper so
    the per-region coverage is unit-testable without a solve."""
    zvals = [z for iv in geo.z_intervals_nm.values() for z in iv]
    zlo, zhi = min(zvals), max(zvals)
    zpts = [zlo + fz * (zhi - zlo) for fz in np.linspace(0.03, 0.97, 18)]     # global spread
    for (_zl, _zh) in geo.z_intervals_nm.values():                            # + every region
        zpts += [_zl + fr * (_zh - _zl) for fr in (0.2, 0.5, 0.8)]
    return zpts


def _detect_bloch_dirs(geo: OpticalGeometry):
    """Return a per-identification direction list (['x'|'y'] of length n_px+n_py)
    mapping each periodic idnr to its translation axis. CRITICAL: ng.Periodic keys
    its `phase` list per identification in IDNR order, and netgen does NOT keep that
    in creation order -- for a glued multi-layer stack the x- and y-face idnrs come
    out INTERLEAVED (x,y,x,y,...), one x/y pair per z-layer. A wrong mapping silently
    puts phase=1 on the x-faces (plain-periodic in x) -> the solver returns the
    NORMAL-incidence field at every angle (a particularly nasty silent failure, since
    normal incidence still validates). We resolve each idnr's axis by toggling a
    marker phase on that idnr alone and measuring whether it perturbs the x- or the
    y-boundary, then assert the recovered x/y counts (anti-silent-failure).

    Z-SAMPLING (audit F-6): the 18 GLOBAL stack fractions alone MISS a thin layer's periodic faces
    (a sub-100 nm layer in a multi-micron stack falls between two global samples), so its idnr is
    classified from numerical noise and the x/y-count assertion fires -- 'resolved N x / M y,
    expected ...' on an otherwise supported stack. The sample list therefore ALSO carries three
    PER-REGION fractions of every z-interval, so every layer -- however thin -- is hit. This is the
    per-region sampling shg_fem._ensure_bloch_dirs used to apply at ONE call site; folding it in
    here gives it to every caller (solve_fem, solve_fem_sourced, _bloch_phase_list). Purely
    ADDITIVE: extra samples can only strengthen the true axis's violation (the marker phase
    perturbs only its own identification's faces), so stacks that already resolved are unchanged."""
    cached = getattr(geo, "_bloch_dirs", None)
    if cached is not None:
        return cached
    mesh = geo.mesh
    n_px, n_py = geo.n_px, geo.n_py
    N = n_px + n_py
    Px, Py = geo.period_x_nm, geo.period_y_nm
    if N == 0 or n_py == 0:
        dirs = ["x"] * n_px
    elif n_px == 0:
        dirs = ["y"] * n_py
    else:
        th = cmath.exp(1j * 0.7853981634)           # marker phase != 1
        zpts = _bloch_z_samples(geo)                # global spread + every region (audit F-6)

        def viol(i):                                # x- vs y-boundary perturbation of idnr i
            phases = [(th if j == i else 1.0 + 0j) for j in range(N)]
            fes = ng.Periodic(ng.H1(mesh, order=1, complex=True), phase=phases)
            gf = ng.GridFunction(fes)
            gf.Set(ng.exp(0.01j * ng.z) * (1.0 + 0.3 * ng.y / Py + 0.25 * ng.x / Px + 0.2j))
            xv = yv = 0.0
            for z in zpts:
                for fy in (0.3, 0.6):
                    try:
                        a = complex(gf(mesh(0.0, fy * Py, z))); b = complex(gf(mesh(Px, fy * Py, z)))
                        xv = max(xv, abs(b - a))
                    except Exception:
                        pass
                for fx in (0.3, 0.6):
                    try:
                        c = complex(gf(mesh(fx * Px, 0.0, z))); d = complex(gf(mesh(fx * Px, Py, z)))
                        yv = max(yv, abs(d - c))
                    except Exception:
                        pass
            return xv, yv
        dirs = []
        for i in range(N):
            xv, yv = viol(i)
            dirs.append("x" if xv > yv else "y")
        if dirs.count("x") != n_px or dirs.count("y") != n_py:
            raise RuntimeError(
                "Bloch periodic-phase idnr direction detection inconsistent: resolved "
                "{} x / {} y, expected {} / {}. Oblique incidence cannot be trusted.".format(
                    dirs.count("x"), dirs.count("y"), n_px, n_py))
    try:                                       # memoize on the geo object (a frozen/slotted geo just
        geo._bloch_dirs = dirs                 # recomputes each solve -- correct, only slower)
    except (AttributeError, TypeError):
        pass
    return dirs


def _bloch_phase_list(geo: OpticalGeometry, kx_per_nm: float, ky_per_nm: float = 0.0):
    """Floquet-Bloch phase list (one entry per periodic identification, in idnr order):
    exp(i*kx*Px) on x-identifications, exp(i*ky*Py) on y-identifications. ky=0 (x-z-plane
    incidence) gives phase 1 on y; ky!=0 is CONICAL incidence (2D Bloch phase)."""
    dirs = _detect_bloch_dirs(geo)
    px_phase = cmath.exp(1j * kx_per_nm * geo.period_x_nm)
    py_phase = cmath.exp(1j * ky_per_nm * geo.period_y_nm)
    return [(px_phase if d == "x" else py_phase) for d in dirs]


def _incidence_geometry(optical, n_super):
    """Derive (theta, phi, oblique, conical) [radians/bools] from the OpticalSpec and VALIDATE
    the incidence against solve_fem's implemented regime: top-side only; vacuum superstrate if
    oblique; pol constraints for oblique/conical. Raises NotImplementedError on an unsupported
    combination and warns on a non-angle-aware oblique solve. (Audit OPT-1/OPT-3/OPT-4 guards.)"""
    theta = math.radians(float(getattr(optical, "incidence_angle_deg", 0.0) or 0.0))
    phi = math.radians(float(getattr(optical, "azimuth_deg", 0.0) or 0.0))
    oblique = abs(theta) > 1e-9
    conical = abs(phi) > 1e-9
    if abs(complex(n_super).imag) > 1e-9:
        # R/R0/T normalization and the up/down-wave separation assume the incident wave does not
        # itself decay; a lossy incidence medium makes the energy budget A=1-R-T meaningless. The
        # TMM oracle raises on this (LTM-5); mirror it so the FEM path is not silently wrong at
        # NORMAL incidence too (the n_super!=1 guard below is oblique-only). (Audit OPT-1 mirror.)
        raise NotImplementedError(
            "solve_fem: R/T/A and the energy budget A=1-R-T are defined only for a LOSSLESS "
            "incidence medium (Im(n_super)=0); got n_super={:.4g}.".format(complex(n_super)))
    if oblique and optical.polarization == "x":
        raise NotImplementedError(
            "oblique incidence requires polarization='y' (s-pol) or 'p' (p-pol); "
            "'x' (E along x) is not transverse to an oblique x-z-plane wavevector.")
    # conical (azimuth != 0) supports BOTH s-pol ('y', the in-plane Es rotates with phi) and p-pol
    # ('p', the in-plane p-pol component splits over (cos phi, sin phi)); 'x' is caught by the oblique
    # guard above.
    if getattr(optical, "incidence_side", "top") != "top":
        raise NotImplementedError(
            "solve_fem implements top-side incidence only; incidence_side='{}' is not "
            "supported (the source + R/T extraction are hardwired to top illumination)."
            .format(getattr(optical, "incidence_side", "top")))
    if oblique and abs(complex(n_super) - 1.0) > 1e-6:
        # The in-plane wavevector uses kx = k0 sin(theta) (the VACUUM dispersion). A dense
        # incidence medium would need kx = Re(n_super) k0 sin(theta) and a matched
        # T-normalization; without it the result is silently wrong (audit OPT-1).
        raise NotImplementedError(
            "oblique incidence assumes a vacuum/air incidence medium (n_super=1); got "
            "n_super={:.4g}. Use normal incidence for a non-vacuum superstrate, or extend "
            "solve_fem with the dense-incidence dispersion.".format(complex(n_super)))
    if oblique:
        # The fixed-alpha HalfSpace PML is not angle-aware; energy conservation degrades with angle
        # (audit OPT-3 / cross-cutting F5). Measured vs the tmm oracle (validation/oblique_vs_tmm,
        # lossless slab): ~1% through 30 deg, ~3% at 45 deg. ABOVE ~50 deg the PML reflection grows
        # fast enough that the budget VIOLATES energy conservation (R+T = 1.17 at 60 deg) -- a
        # silently-wrong R/T, so RAISE there rather than return it behind a warning (pass-2 audit).
        # The exact RCWA/PMM/Berreman bridges have no PML and keep the full angular range (the
        # OpticalSpec polar cap stays at 60 deg for them); only the FEM backend is bounded here.
        if abs(theta) > math.radians(_FEM_THETA_MAX_DEG):
            raise NotImplementedError(
                "oblique FEM: |theta| = {:.1f} deg exceeds the FEM's validated envelope (~50 deg) -- "
                "the fixed-alpha HalfSpace z-PML is not angle-aware and VIOLATES energy conservation "
                "above ~50 deg (R+T ~ 1.17 at 60 deg), so R/T would be silently wrong. Stay at "
                "|theta| <= 45 deg for quantitative FEM, or use the exact RCWA/PMM/Berreman bridges "
                "(no PML, full angular range).".format(math.degrees(abs(theta))))
        # ONCE PER PROCESS (audit F-15): this advisory is a property of the FEM's PML, not of the
        # individual solve, so a 40-wavelength sweep repeating it 40x is pure noise. The regime
        # GUARD above (the 50 deg raise) is per-solve and unconditional -- only the advisory is
        # de-duplicated, so no angle can slip through unchecked.
        _advise_once("oblique_pml", _OBLIQUE_PML_ADVICE, stacklevel=3)
    return theta, phi, oblique, conical


_SIMPLE_ALT = re.compile(r"^[A-Za-z0-9_.-]+(\|[A-Za-z0-9_.-]+)*$")


def _validate_sheet_bcs(mesh, sheet_bcs) -> None:
    """RAISE if any `sheet_bcs` key matches no boundary of `mesh` (audit F-8).

    ``ng.ds(definedon=mesh.Boundaries(name))`` on an unknown name is NOT an error in NGSolve: the
    linear/bilinear form assembles cleanly with ``||f|| = 0``, so a typo'd or off-by-one graphene
    sheet simply is not there and the solve returns the SHEET-FREE answer -- numerically plausible,
    silently wrong. The builder's interior interfaces are named ``iface_z<int(round(z_nm))>``
    (ngsolve_layered), i.e. the caller has to guess an exact integer nanometre, so an off-by-one is
    the expected mistake, not an exotic one.

    Names are NGSolve boundary PATTERNS (regexes, ``a|b`` alternation allowed), so matching is
    ``re.fullmatch`` against every boundary name. For a pattern that is a plain ``|``-alternation of
    literal names, EACH alternative must match something -- otherwise ``iface_z100|iface_z999``
    would pass on the strength of its good half and drop the typo'd sheet silently."""
    known = list(dict.fromkeys(mesh.GetBoundaries()))
    ifaces = sorted(b for b in known if b.startswith("iface_z"))
    bad = []
    for name in sheet_bcs:
        pat = str(name)
        parts = pat.split("|") if _SIMPLE_ALT.match(pat) else [pat]
        for p in parts:
            try:
                hit = any(re.fullmatch(p, b) for b in known)
            except re.error as e:                       # not a valid regex -> it can match nothing
                raise ValueError("sheet_bcs boundary pattern {!r} is not a valid regex: {}".format(
                    pat, e)) from None
            if not hit:
                bad.append(p)
    if bad:
        raise ValueError(
            "sheet_bcs boundary name(s) {} match no boundary of this mesh, so the conductive-sheet "
            "term would assemble to ZERO and the solve would silently return the SHEET-FREE answer. "
            "Available interface boundaries: {}. All mesh boundaries: {}. (Interfaces are named "
            "'iface_z<z in nm, rounded to the nearest integer>' -- check the layer z you meant.)"
            .format(", ".join(repr(b) for b in bad),
                    ", ".join(ifaces) if ifaces else "<none>", ", ".join(known)))


# ---- polarization vocabulary (audit V-8) --------------------------------------------------------
# This solver speaks the OpticalSpec LAB-AXIS family {'x', 'y', 'p'} everywhere.  The four sibling
# spellings, the conversions, and the normal-incidence / azimuth caveats are documented in
# dynameta.core.polarization; the set accepted here is unchanged -- acceptance unification (b)
# widened only the plane-of-incidence families ({'s','p'}, {'te','tm'}), whose aliases hold at every
# geometry.  The lab-axis crossing does NOT: 'y' is s-pol only at phi = 0, and THIS module reads 'y'
# at phi != 0 as the rotated s-hat while the lumenairy bridge reads it as lab row 1.  So it stays
# strict, and a caller crossing in from {'s','p'} converts explicitly with normalize_pol.
def _reject_pol_unless_lab(polarization, where: str):
    """Guard the lab-axis vocabulary.  LAZY import, failure path only, so the valid path is
    untouched and no import edge is added."""
    if polarization not in ("x", "y", "p"):
        from dynameta.core.polarization import pol_vocabulary_error
        raise pol_vocabulary_error(polarization, "lab_xyp", where=where, param="polarization")


def _layered_background(polarization, k0, kx, ky, phi, kz_s_c, kz_sub, z_int,
                        eps_sup_c, eps_sub_c, n_super, n_sub, *, inc_phase=None):
    """THE analytic layered (two-region Fresnel) background of the scattered-field formulation:
    the bare superstrate/substrate field E_bg the structure scatters off, its eps_bg(z), the
    incident-only field E_inc (the Poynting-flux reference) and, for the scalar polarizations, the
    background amplitudes R0/T0 the extractor adds back.

    eps_bg(z) = superstrate medium above the substrate-top interface z_int, substrate medium below.
    The scattered source k0^2 (eps - eps_bg) E_bg is then nonzero ONLY in the structure layers --
    the substrate carries NO volumetric source (eps == eps_bg there), which is what makes a dense
    (non-vacuum) substrate accurate. Reduces exactly to the plain incident wave when
    n_sub == n_super == 1 (R0 = 0, T0 = 1).

    THREE polarization branches, keyed to OpticalSpec.polarization: 'p' (E in the plane of
    incidence), 'x' (E along x -- the OpticalSpec DEFAULT, normal incidence only) and anything else
    = s-pol along the conical in-plane direction Es = (-sin phi, cos phi, 0).

    SINGLE SOURCE (audit F-12): shg_fem carried its OWN copy of this construction, declared
    "byte-for-byte the construction solve_fem uses" -- and it had already drifted to only TWO
    branches, so polarization='x' was silently solved as E-along-y, the ORTHOGONAL mode, and every
    field/relres/sheet diagnostic built on it described the wrong mode. Both callers now build the
    background here.

    inc_phase overrides the transverse Bloch phase exp(i(kx x + ky y)) (solve_fem's diagnostic
    "envelope" route passes 1.0, carrying the transverse phase in the operator instead).

    AUDIT V-8: `polarization` is the OpticalSpec LAB-AXIS vocabulary {'x', 'y', 'p'} -- see
    dynameta.core.polarization for the map and for why 'y' is s-pol only at azimuth phi = 0 (at
    phi != 0 this builder reads 'y' as the ROTATED s-hat, while the lumenairy bridge reads the
    same letter as lab row 1; the bridge refuses conical for exactly that reason). The "anything
    else = s-pol" fallthrough below is now unreachable for an off-vocabulary label: the accepted
    set is unchanged, the rejection is named."""
    _reject_pol_unless_lab(polarization, "_layered_background")
    iph = ng.exp(1j * (kx * ng.x + ky * ng.y)) if inc_phase is None else inc_phase
    eps_bg = ng.IfPos(ng.z - z_int, eps_sup_c, eps_sub_c)
    # s-pol unit E (perpendicular to the plane of incidence); reduces to +y at phi=0
    es_x, es_y = -math.sin(phi), math.cos(phi)
    # in-plane wavevector magnitude + its azimuthal unit (cos phi, sin phi) = (kx, ky)/|k_par|, used by
    # the p-pol field (the transverse p-pol component points along k_par). At phi=0 -> (1, 0); at normal
    # incidence (|k_par|=0) -> (1, 0) so p-pol reduces to E along x. Conical p-pol splits over this.
    kpar = math.hypot(kx, ky)
    cphi, sphi = (kx / kpar, ky / kpar) if kpar > 1e-12 * k0 else (1.0, 0.0)
    if polarization == "p":
        # p-pol: E in the PLANE OF INCIDENCE (the plane through z and k_par). The transverse part
        # points along (cos phi, sin phi); the z part carries the full in-plane |k_par|. At phi=0 this
        # reduces to (Ex, 0, Ez). The background reflection/transmission E-vector amplitudes (rho, tau)
        # come from the physical interface BCs at z_int -- tangential-E and Hy continuity -- solved
        # NUMERICALLY (no Fresnel sign ambiguity). A 1-D layered medium is rotationally symmetric about
        # z, so M/rhs depend only on the kz's + eps (NOT on phi): identical to the phi=0 problem. Hy ~
        # E_t*eps/qz (qz the z-wavevector: -kz down, +kz up).
        cth, sth = kz_s_c / (complex(n_super) * k0), kpar / (complex(n_super) * k0)
        cth_t, sth_t = kz_sub / (complex(n_sub) * k0), kpar / (complex(n_sub) * k0)
        A = cmath.exp(-1j * kz_s_c * z_int); B = cmath.exp(1j * kz_s_c * z_int)
        C = cmath.exp(-1j * kz_sub * z_int)
        M = np.array([[cth * B,                   -cth_t * C],
                      [cth * B * eps_sup_c / kz_s_c, cth_t * C * eps_sub_c / kz_sub]], dtype=complex)
        rhs = np.array([-cth * A, cth * A * eps_sup_c / kz_s_c], dtype=complex)
        pp_rho, pp_tau = (complex(val) for val in np.linalg.solve(M, rhs))
        # transverse (in-plane) p-pol profile (the old "Ex"); split over (cos phi, sin phi) for x/y.
        et_sup = cth * ng.exp((-1j * kz_s_c) * ng.z) + pp_rho * cth * ng.exp((1j * kz_s_c) * ng.z)
        ez_sup = sth * ng.exp((-1j * kz_s_c) * ng.z) - pp_rho * sth * ng.exp((1j * kz_s_c) * ng.z)
        et_sub = pp_tau * cth_t * ng.exp((-1j * kz_sub) * ng.z)
        ez_sub = pp_tau * sth_t * ng.exp((-1j * kz_sub) * ng.z)
        et = ng.IfPos(ng.z - z_int, et_sup, et_sub)             # transverse magnitude (phi-frame Ex)
        E_bg = ng.CoefficientFunction((iph * cphi * et,
                                        iph * sphi * et,
                                        iph * ng.IfPos(ng.z - z_int, ez_sup, ez_sub)))
        # incident-only field (no background reflection/transmission) for the Poynting-flux reference
        inc_t = cth * ng.exp((-1j * kz_s_c) * ng.z)             # incident transverse profile
        E_inc = ng.CoefficientFunction((iph * cphi * inc_t,
                                         iph * sphi * inc_t,
                                         iph * sth * ng.exp((-1j * kz_s_c) * ng.z)))
        R0 = T0 = None                          # p-pol extracts the total field directly
    else:
        # s-pol (E along y) / x-pol: scalar tangential field. Fresnel R0/T0 (field
        # amplitude), z=0 reference. The extractor returns the scattered amplitude and
        # the caller adds R0/T0 back.
        r_f = (kz_s_c - kz_sub) / (kz_s_c + kz_sub)
        t_f = 2.0 * kz_s_c / (kz_s_c + kz_sub)
        R0 = r_f * cmath.exp(-2j * kz_s_c * z_int)
        T0 = t_f * cmath.exp(-1j * (kz_s_c - kz_sub) * z_int)
        sup_bg = ng.exp((-1j * kz_s_c) * ng.z) + R0 * ng.exp((1j * kz_s_c) * ng.z)
        sub_bg = T0 * ng.exp((-1j * kz_sub) * ng.z)
        bg_field = iph * ng.IfPos(ng.z - z_int, sup_bg, sub_bg)
        inc_only = iph * ng.exp((-1j * kz_s_c) * ng.z)   # incident-only (flux reference)
        if polarization == "x":
            E_bg = ng.CoefficientFunction((bg_field, 0.0, 0.0))
            E_inc = ng.CoefficientFunction((inc_only, 0.0, 0.0))
        else:
            # s-pol along the (conical) in-plane direction E_s = (-sin phi, cos phi, 0);
            # reduces to (0, bg_field, 0) at phi=0.
            E_bg = ng.CoefficientFunction((es_x * bg_field, es_y * bg_field, 0.0))
            E_inc = ng.CoefficientFunction((es_x * inc_only, es_y * inc_only, 0.0))
    return E_bg, E_inc, eps_bg, R0, T0


def background_probe_pol(polarization, theta, phi=0.0):
    """The polarization unit vector to PROJECT a _layered_background field onto: the p-pol unit
    vector (cos th cos phi, cos th sin phi, -sin th) for 'p', xhat for 'x', and the conical s-pol
    unit (-sin phi, cos phi, 0) otherwise. Kept beside the background builder so the projection can
    never drift out of step with the branch that built the field (audit F-12: shg_fem's private
    copy probed (0,1,0) for 'x' as well, matching its own wrong background).

    AUDIT V-8: `polarization` is the OpticalSpec LAB-AXIS vocabulary {'x', 'y', 'p'} (map:
    dynameta.core.polarization). The trailing "otherwise" branch is the 'y' branch, so an
    off-vocabulary label used to be probed as s-pol silently; same accepted set, named
    rejection."""
    _reject_pol_unless_lab(polarization, "background_probe_pol")
    if polarization == "p":
        return (math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), -math.sin(theta))
    if polarization == "x":
        return (1.0, 0.0, 0.0)
    return (-math.sin(phi), math.cos(phi), 0.0)


def solve_fem(geo: OpticalGeometry, lambda_m: float,
                eps_cf: ng.CoefficientFunction, optical: "OpticalSpec",
                *, order: int = 2, n_super: complex = 1.0 + 0j,
                n_sub: complex = 1.0 + 0j, verbose: bool = False,
                sheet_bcs: "dict | None" = None, diagnostics: bool = True,
                _reuse_fes=None) -> OpticalResult:
    """Solve and extract reflection r/R and (if a transmitted wave reaches the
    substrate) transmission t/T. n_super/n_sub are the semi-infinite superstrate/
    substrate refractive indices = sqrt(eps).

    R, T (and r/t) are the SPECULAR 0-ORDER (zeroth diffraction order) only -- the
    _cell_average demodulates and laterally averages on a probe grid SIZED so that every
    aliased Fourier order is evanescent AND DECAYED to e^-3 at the probe planes (audit C3-1: the
    old fixed 6x6 grid aliased orders m = 0 (mod 6) into the coefficient at full weight for
    period > 6*lambda/n cells and 6x1 supercells; propagating non-aliased orders always
    averaged to zero. Audit F-10: "evanescent" alone is vacuous just past cutoff, where the
    alias survives the standoff at 0.988). The grid size is CAPPED at _PROBE_GRID_MAX per
    direction and warns above it -- a thin buffer makes the criterion ask for tens of points
    per direction, which costs N^2 point evaluations and is better fixed by thickening the
    buffer (audit F-10 residual).
    They are therefore the TOTAL reflectance/transmittance only for a SUB-WAVELENGTH cell
    (no propagating higher orders); for a diffracting (period > lambda/n) cell the
    diffracted power is missing from R/T and is mis-attributed to A. result.R_flux/T_flux
    (the reconstructed-field z-Poynting flux) capture ALL propagating orders and are the
    authoritative total there -- prefer them, and use a sub-wavelength cell when relying on
    R/T/A. (RCWA/PMM bridges instead return order-SUMMED R/T, so this 0-order caveat is
    FEM-specific.)

    A = 1 - R - T is the energy-budget CLOSURE (it is identically 1-R-T, not an
    independent measurement). result.A_independent is the INDEPENDENTLY measured
    absorbed fraction (volumetric Im(eps)|E|^2 integral); |A - A_independent| is the
    genuine, non-tautological energy/numerics diagnostic (a large gap with A_independent ~ 0
    but A > 0 flags higher-order diffraction leaving the 0-order R/T, not absorption).

    sheet_bcs (C3): {boundary_name: sigma_S} applies a conductive-SHEET surface-current
    boundary condition (e.g. graphene) on the named internal interface(s). A sheet of
    conductivity sigma (siemens) carries J_s = sigma E_tan, giving the tangential-trace
    Robin term + i k0 Z0 sigma (E_tan . v_tan) over the interface (Z0 = free-space
    impedance); in the scattered-field formulation the sheet-free background E_bg also
    drives it, so the same term enters the RHS on E_bg. Validated vs the analytic
    core.graphene.sheet_rt in validation/graphene_sheet_fem.py.

    diagnostics=False skips the best-effort post-solve diagnostics -- A_independent,
    per_region_absorption, R_flux/T_flux (three volume/flux integrations + two field
    interpolations per solve) -- returning None for all four. The r/R/t/T/A extraction is
    byte-identical. NOTE this also disables the energy-closure mismatch warning (it needs
    A_independent), so only opt out on throughput paths whose configuration a
    diagnostics=True solve has already vetted (audit 6.2 perf)."""
    k0 = 2.0 * math.pi / (lambda_m * S)        # nm^-1
    mesh = geo.mesh

    # ---- incidence: plane wave, polar angle theta, azimuth phi (derived + validated) ----
    theta, phi, oblique, conical = _incidence_geometry(optical, n_super)
    pol_p = optical.polarization == "p"        # p-pol: E in the x-z plane (Ex, Ez)
    envelope = oblique and _OBLIQUE_FORMULATION == "envelope" and not pol_p and not conical
    # in-plane wavevector k_par = (kx, ky) (vacuum dispersion; oblique requires n_super=1).
    kx = k0 * math.sin(theta) * math.cos(phi)
    ky = k0 * math.sin(theta) * math.sin(phi)
    # incidence-medium z-wavevector kz_s = sqrt((n_super k0)^2 - k_par^2). At NORMAL incidence this
    # is n_super*k0 -- the old k0*cos(theta) silently used the VACUUM dispersion and gave wrong R/T for
    # a dense (n_super != 1) superstrate (audit P1). For the oblique path n_super is guaranteed 1
    # (vacuum incidence), so this reduces EXACTLY to k0*cos(theta) (byte-identical for every validated
    # case). n_super is real here (Im(n_super) is screened upstream).
    kz_s = math.sqrt(max((complex(n_super).real * k0) ** 2 - kx ** 2 - ky ** 2, 0.0))
    # s-pol unit E (perpendicular to the plane of incidence); reduces to +y at phi=0
    es_x, es_y = -math.sin(phi), math.cos(phi)
    # in-plane wavevector magnitude + its azimuthal unit (cos phi, sin phi) = (kx, ky)/|k_par|, used by
    # the p-pol field (the transverse p-pol component points along k_par). At phi=0 -> (1, 0); at normal
    # incidence (|k_par|=0) -> (1, 0) so p-pol reduces to E along x. Conical p-pol splits over this.
    kpar = math.hypot(kx, ky)
    cphi, sphi = (kx / kpar, ky / kpar) if kpar > 1e-12 * k0 else (1.0, 0.0)

    # PML: by default the ordinary normal HalfSpace z-stretch (alpha=1j CONSTANT). BUT mesh.SetPML's
    # coordinate stretch is WRONG for an OFF-DIAGONAL anisotropic eps -- it perturbs the physically
    # decoupled component by a resolution-INDEPENDENT ~3% (B-fix diagnosis: with the PML removed the
    # off-diagonal-tensor field is identical to the diagonal one to ~1e-6; SetPML alone breaks it).
    # So for a TENSOR eps we instead use an explicit UPML: the anisotropic PML material tensor
    # Lambda = diag(s_z, s_z, 1/s_z) folded directly into the weak form (curl . Lambda^-1 curl
    # - k0^2 Lambda eps). That is the rigorous stretched-coordinate PML for an arbitrary medium and
    # reduces to the SetPML answer for isotropic/diagonal eps; the scalar path keeps the heavily
    # validated SetPML (zero regression). s_z = 1 + alpha inside either PML region, else 1.
    pml_alpha = 1j
    z_air_top = geo.z_super_interface_nm
    z_sub_top = geo.z_sub_interface_nm
    eps_is_tensor = (tuple(getattr(eps_cf, "dims", ())) == (3, 3))
    use_upml = eps_is_tensor and not envelope
    try:
        mesh.UnSetPML("pml_top"); mesh.UnSetPML("pml_bot")
    except Exception:
        pass
    if not use_upml:
        mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_air_top), normal=(0, 0, 1), alpha=pml_alpha), "pml_top")
        mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_sub_top), normal=(0, 0, -1), alpha=pml_alpha), "pml_bot")
    _pml_mask = ng.IfPos(ng.z - z_air_top, 1.0, 0.0) + ng.IfPos(z_sub_top - ng.z, 1.0, 0.0)
    _s_z = 1.0 + pml_alpha * _pml_mask
    upml_Linv = (1.0 / _s_z, 1.0 / _s_z, _s_z)     # Lambda^-1 diag, for curl . Lambda^-1 curl
    upml_Ldiag = (_s_z, _s_z, 1.0 / _s_z)          # Lambda diag, for the mass term Lambda . eps

    # ---- layered (Fresnel two-region) background ----
    # eps_bg(z) = superstrate medium above the substrate-top interface z_int, substrate
    # medium below. E_bg = the analytic bare air/substrate Fresnel field (incident +
    # background reflection R0 above, background transmission T0 below). The scattered
    # source k0^2 (eps - eps_bg) E_bg is then nonzero ONLY in the structure layers
    # (slab/oxide/patch/carrier) -- the substrate carries NO volumetric source
    # (eps==eps_bg there). This is what makes a dense (non-vacuum) substrate accurate:
    # the old uniform eps_bg=1 drove a huge source through the whole substrate at the
    # WRONG (vacuum) wavevector. Reduces exactly to the plain incident wave when
    # n_sub==n_super==1 (R0=0, T0=1). Incidence medium assumed vacuum (kx=k0 sin th).
    kz_sub = complex(np.sqrt(complex((complex(n_sub) * k0) ** 2 - kx ** 2 - ky ** 2)))
    if abs(kz_sub) <= 1e-6 * k0:
        # At the grazing cutoff the substrate 0-order is non-propagating and |kz_sub| ~ 0. kz_sub
        # is a DIVISOR in the p-pol interface BC matrix and the T power factor (and the s-pol
        # substrate fit degenerates to a rank-deficient DC fit), so R/T blow up to inf/NaN here.
        # Refuse rather than return nonsense. (A deep-evanescent substrate keeps |kz_sub| large --
        # that case returns T~0 sensibly and is NOT caught here.)
        raise NotImplementedError(
            "solve_fem: substrate 0-order is at the grazing cutoff (|kz_sub|={:.3e} ~ 0 nm^-1): no "
            "propagating transmitted order and R/T are singular. n_sub={:.4g}, sin(theta)={:.4g}. "
            "Nudge the angle/wavelength off the exact cutoff.".format(
                abs(kz_sub), complex(n_sub), math.sin(theta)))
    kz_s_c = complex(kz_s)
    z_int = (geo.z_intervals_nm["substrate"][1] if "substrate" in geo.z_intervals_nm
              else geo.z_sub_interface_nm)                       # substrate-top interface (nm)
    eps_sup_c, eps_sub_c = complex(n_super) ** 2, complex(n_sub) ** 2
    # transverse Bloch phase exp(i(kx x + ky y)); reduces to exp(i kx x) at phi=0
    inc_x_phase = (1.0 if envelope else ng.exp(1j * (kx * ng.x + ky * ng.y)))

    E_bg, E_inc, eps_bg_cf, R0, T0 = _layered_background(
        optical.polarization, k0, kx, ky, phi, kz_s_c, kz_sub, z_int,
        eps_sup_c, eps_sub_c, n_super, n_sub, inc_phase=inc_x_phase)

    # ---- periodic HCurl space ----
    if getattr(geo, "sym_x", False) or getattr(geo, "sym_y", False):
        # MIRROR-SYMMETRY reduced cell (NORMAL incidence): the reduced lateral axis carries a symmetry
        # WALL instead of a periodic boundary. The wall whose outward normal is PARALLEL to the incident
        # E is a PEC (perfect electric conductor: tangential E = 0 = the HCurl Dirichlet/essential BC);
        # the wall whose normal is PERPENDICULAR to E is a PMC (perfect magnetic conductor: the NATURAL
        # BC of the curl-curl form -- left unconstrained). At normal incidence pol 'x' -> E||x so the
        # x-walls ('sym_x') are PEC; pol 'y' -> E||y so the y-walls ('sym_y') are PEC. A surviving
        # non-reduced axis stays plain-periodic (Bloch phase 1 at normal incidence).
        if oblique or conical:
            raise NotImplementedError(
                "symmetry-reduced FEM mesh is NORMAL-incidence only (an oblique/conical wavevector "
                "breaks the mirror symmetry); got theta={:.3g} deg.".format(math.degrees(theta)))
        if eps_is_tensor:
            raise NotImplementedError(
                "symmetry-reduced FEM mesh does not support a tensor (anisotropic/gyrotropic) eps -- "
                "off-diagonal coupling breaks the mirror parity. Use the full periodic solve.")
        if optical.polarization not in ("x", "y"):
            raise NotImplementedError(
                "symmetry-reduced FEM mesh requires polarization 'x' or 'y' (got {!r}); the wall "
                "type is keyed to the linear-polarization axis.".format(optical.polarization))
        pec = []
        if getattr(geo, "sym_x", False) and optical.polarization == "x":
            pec.append("sym_x")                              # x-wall normal || E(x) -> PEC
        if getattr(geo, "sym_y", False) and optical.polarization == "y":
            pec.append("sym_y")                              # y-wall normal || E(y) -> PEC
        base = ng.HCurl(mesh, order=order, complex=True, dirichlet="|".join(pec))
        # any surviving periodic axis (half-cell) is plain-periodic at normal incidence (phase 1);
        # a quarter cell has no periodic identification (n_px=n_py=0) -> the bare HCurl is the space.
        fes = ng.Periodic(base) if (geo.n_px or geo.n_py) else base
    elif oblique and not envelope and (geo.n_px or geo.n_py):
        # quasi-periodic: u(minion=x=Px) = exp(+i kx Px) u(master=x=0). The phase
        # list is keyed per identification in idnr order, which netgen does NOT keep
        # in creation order -- _bloch_phase_list resolves + verifies the true order.
        phases = _bloch_phase_list(geo, kx, ky)
        fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""), phase=phases)
    elif _reuse_fes is not None:
        # NORMAL incidence (no oblique Bloch phases) -> the FESpace is wavelength-INDEPENDENT, so a
        # sweep solver may build it ONCE and reuse it here (the eps mass + RHS + factorization still
        # rebuild per wavelength, so R/T are byte-identical). Reuse is REFUSED for oblique above (the
        # Periodic Bloch phases depend on k0). See make_fem_optical_solver.
        fes = _reuse_fes
    else:
        fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""))
    # Is the space QUASI-periodic (Bloch phases != 1)? Only the oblique non-envelope branch above
    # builds one: the symmetry-reduced branch raises at oblique, _reuse_fes is refused at oblique,
    # and the plain-periodic fallbacks carry phase 1. Drives BOTH the BilinearForm(symmetric=) flag
    # and the bddc_cg guard (audit F-9).
    bloch_phased = bool(oblique and not envelope and (geo.n_px or geo.n_py))
    u, v = fes.TrialFunction(), fes.TestFunction()

    if envelope:
        # modified curl: C(w)=curl(w)+i k_par x w, k_par=(kx,0,0) -> kcross(w)=
        # 1j*(0,-kx w_z, kx w_y). Trial +kcross, test -kcross (conjugate phase).
        def kcross(w):
            return 1j * ng.CoefficientFunction((0.0, -kx * w[2], kx * w[1]))
        curlE = ng.curl(u) + kcross(u)
        curlV = ng.curl(v) + _TEST_KCROSS_SIGN * kcross(v)
    else:
        curlE, curlV = ng.curl(u), ng.curl(v)

    # a tensor eps makes the matvec term (eps.u).v non-symmetric in general (e.g. magneto-optic);
    # assemble non-symmetric so NGSolve does not symmetrize it (only the scalar path is symmetric).
    #
    # audit F-9: a BLOCH-PHASED space is not symmetric either. ng.Periodic(..., phase=...) applies
    # conj(phase) on the TEST (row) side and phase on the trial (column) side, so even a REAL
    # symmetric integrand assembles HERMITIAN, not symmetric -- measured max|A-A^T|/|A| = 0.406 for
    # HCurl (0.197 for H1) with a real eps, and with a complex (lossy) eps the matrix is NEITHER
    # (0.411 / 0.038). The flag is inert in ngsolve 6.2.2604 (both builds return the identical
    # SparseMatrix<complex>), so this is a correctness statement, not a behaviour change today; it
    # stops any build that HONOURS the flag from symmetrizing a non-symmetric operator, and it is
    # the same condition the bddc_cg guard below uses.
    a = ng.BilinearForm(fes, symmetric=(not envelope) and not eps_is_tensor and not bloch_phased)
    f = ng.LinearForm(fes)
    if eps_is_tensor:
        # anisotropic eps: (eps . E) . v expanded as the explicit scalar component sum
        # sum_ij eps_ij u[j] v[i] (the matrix-vector CF (eps_cf*u)*v of a per-material domain-list
        # is a known assembly footgun). eps_bg (scalar) subtracts on the diagonal only.
        if use_upml:
            # explicit UPML: curl . Lambda^-1 curl - k0^2 (Lambda . eps); Lambda = I outside the PML.
            curl_term = sum(upml_Linv[i] * curlE[i] * curlV[i] for i in range(3))
            mass = sum(upml_Ldiag[i] * eps_cf[i, j] * u[j] * v[i] for i in range(3) for j in range(3))
            a += (curl_term - k0 ** 2 * mass) * ng.dx
        else:
            a += (curlE * curlV - k0 ** 2 * sum(eps_cf[i, j] * u[j] * v[i]
                                                for i in range(3) for j in range(3))) * ng.dx
        # the scattered source is nonzero only where eps != eps_bg (the structure, Lambda = I there),
        # so no UPML factor is needed on it.
        f += (k0 ** 2 * sum((eps_cf[i, j] - (eps_bg_cf if i == j else 0.0)) * E_bg[j] * v[i]
                            for i in range(3) for j in range(3))) * ng.dx
    else:
        a += (curlE * curlV - k0 ** 2 * eps_cf * (u * v)) * ng.dx
        f += (k0 ** 2 * (eps_cf - eps_bg_cf) * (E_bg * v)) * ng.dx
    # C3: conductive-sheet (graphene) surface-current BC on named internal interface(s). The sheet
    # current J_s = sigma E_tan makes [n x H] = sigma E_tan, contributing the tangential-trace Robin
    # term + i k0 Z0 sigma (E_tan . v_tan) over the interface (Z0 free-space impedance; k0 in nm^-1
    # and ds in nm^2 keep it dimensionally consistent with the k0^2 eps volume term). The sheet-free
    # background E_bg drives the scattered field, so the SAME term enters the RHS on E_bg.
    if sheet_bcs:
        _validate_sheet_bcs(mesh, sheet_bcs)                 # audit F-8 (silent no-op sheet)
        _Z0 = 1.0 / (EPS0 * C_LIGHT)                         # free-space wave impedance (ohm), from constants.py
        for _bnd, _sigma in sheet_bcs.items():
            # sign: with exp(-i omega t) and Im(eps)>0 = loss, a passive sheet (Re sigma > 0) must
            # ABSORB, so the dissipative operator term is - i k0 Z0 sigma (E_tan . v_tan).
            _ds = ng.ds(definedon=mesh.Boundaries(_bnd))
            a += (-1j * k0 * _Z0 * complex(_sigma) * (u.Trace() * v.Trace())) * _ds
            f += (1j * k0 * _Z0 * complex(_sigma) * (E_bg * v.Trace())) * _ds
    # C8: resolve the linear solver. "ams"/"hypre" request a HYPRE auxiliary-space-Maxwell (AMS)
    # preconditioner -- the rung above BDDC for large 3D -- but the standard pip NGSolve wheel is
    # built WITHOUT HYPRE and naively constructing it SEGFAULTS (not a catchable Python error). So by
    # default we DO NOT attempt it: warn and fall back to bddc_gmres. A user whose NGSolve IS built
    # with HYPRE/AMS opts in with DYNAMETA_AMG_OK=1 (see docs/installing_hypre_windows.md).
    _ls = optical.linear_solver
    if _ls == "bddc_cg" and (bloch_phased or eps_is_tensor):
        # audit F-9: ng.solvers.CGSolver is called without conjugate=, i.e. as COCG -- "a pseudo
        # inner product that makes CG work with complex SYMMETRIC matrices" (its own docstring).
        # On a Bloch-phased space the assembled matrix is HERMITIAN-not-symmetric with a real eps
        # and NEITHER with a lossy one (measured max|A-A^T|/|A| = 0.406 / 0.411 for HCurl), and a
        # tensor eps breaks symmetry outright, so COCG has no convergence theory here. The
        # relres > 1e-3 check afterwards only WARNS, so a stalled COCG could still return its
        # iterate as the answer. Fall back to the GMRes route (same BDDC preconditioner, no
        # symmetry assumption) rather than silently iterating an invalid Krylov method.
        _advise_once(
            "bddc_cg_nonsymmetric",
            "linear_solver='bddc_cg' is invalid for this solve: {} makes the assembled matrix "
            "non-symmetric (COCG assumes complex-SYMMETRIC and has no convergence theory here). "
            "Falling back to 'bddc_gmres' (same BDDC preconditioner). Set linear_solver="
            "'bddc_gmres' or 'umfpack' explicitly to silence this. [reported once per process]"
            .format("oblique incidence on a quasi-periodic (Bloch-phased) space"
                    if bloch_phased else "an anisotropic (tensor) eps"))
        _ls = "bddc_gmres"
    if _ls in ("ams", "hypre"):
        if os.environ.get("DYNAMETA_AMG_OK"):
            pre = ng.Preconditioner(a, "hypre_ams")     # HCurl AMS (caller vouches the build has it)
            _ls = "amg_gmres"
        else:
            warnings.warn(
                "linear_solver='{}' needs an NGSolve built with HYPRE/AMS; the standard pip wheel "
                "lacks it and ATTEMPTING it segfaults, so falling back to 'bddc_gmres'. Build a "
                "HYPRE-enabled NGSolve (docs/installing_hypre_windows.md) and set DYNAMETA_AMG_OK=1 "
                "to use it.".format(optical.linear_solver), RuntimeWarning, stacklevel=2)
            _ls, pre = "bddc_gmres", ng.Preconditioner(a, "bddc")
    else:
        pre = ng.Preconditioner(a, "bddc") if _ls.startswith("bddc") else None

    gfu = ng.GridFunction(fes)
    t0 = time.time()
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        if _ls == "umfpack":
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec
        elif _ls == "bddc_cg":
            inv = ng.solvers.CGSolver(mat=a.mat, pre=pre.mat, tol=optical.gmres_rtol,
                                        maxiter=optical.gmres_max_iter)
            gfu.vec.data = inv * f.vec
        else:  # bddc_gmres / amg_gmres
            ng.solvers.GMRes(A=a.mat, b=f.vec, pre=pre.mat, x=gfu.vec,
                                tol=optical.gmres_rtol, maxsteps=optical.gmres_max_iter,
                                printrates=verbose)
    dt = time.time() - t0

    # OS-1: bddc_gmres/bddc_cg can stop at maxsteps without reaching tol (the ill-conditioned
    # ENZ / lossy-metal regime) and silently return a wrong field. Independently measure the
    # relative residual ||b - A x|| / ||b|| and warn so a non-converged solve is not trusted.
    if optical.linear_solver != "umfpack":
        rvec = gfu.vec.CreateVector()
        rvec.data = f.vec - a.mat * gfu.vec
        rn = float(np.linalg.norm(rvec.FV().NumPy()))
        bn = float(np.linalg.norm(f.vec.FV().NumPy()))
        relres = rn / bn if bn > 0.0 else rn
        if relres > 1e-3:
            warnings.warn(
                "solve_fem: iterative solver '{}' did not converge (relative residual {:.2e} "
                "> 1e-3 after {} steps); R/T/A are unreliable. Use linear_solver='umfpack', "
                "raise gmres_max_iter, or refine the mesh.".format(
                    optical.linear_solver, relres, optical.gmres_max_iter), stacklevel=2)

    # Demodulation: phase_in_space holds the physical field E=u exp(i(kx x+ky y)) -> demod
    # by exp(-i(kx x+ky y)) to recover the 0-order Fourier coefficient. envelope already
    # holds u (kx=ky=0 in the demod).
    kx_d = 0.0 if envelope else kx
    ky_d = 0.0 if envelope else ky
    # audit F-15: collect THIS solve's two-wave fit residuals and emit at most ONE warning naming
    # every offending band, instead of one warning per band. try/finally so an exception in the
    # extraction cannot leave a stale accumulator on the thread.
    _fit_ctx.acc = _fit_bands = []
    try:
        if pol_p:
            # p-pol: reconstruct the TOTAL field (E_bg + scattered gfu) and extract from the in-plane
            # p-pol up/down ratio (convention-robust). Project onto the transverse p-pol direction
            # (cos phi, sin phi) and 2D-demodulate by (kx, ky); at phi=0 this is the tangential-Ex
            # extraction. T carries the p-pol Poynting factor (Sz ~ |E_t|^2 eps/kz).
            r, R, t, T = _ppol_extract(mesh, E_bg + gfu, kz_s, kz_sub, kx_d, ky_d, (cphi, sphi, 0.0),
                                         geo, eps_sup_c, eps_sub_c)
            A = None if T is None else float(1.0 - R - T)
        else:
            # project the scattered field onto the extraction polarization: tangential Ex
            # (1,0,0) for x-pol, or the (conical) s-pol unit vector Es=(es_x,es_y,0) for 'y'.
            proj = (1.0, 0.0, 0.0) if optical.polarization == "x" else (es_x, es_y, 0.0)
            # total amplitude = background (analytic R0/T0) + scattered (fitted from gfu)
            r = complex(R0) + _reflection(mesh, gfu, kz_s, proj, kx_d, ky_d, geo)
            R = float(abs(r) ** 2)
            t_scat = _transmission(mesh, gfu, kz_sub, proj, kx_d, ky_d, geo)
            if t_scat is None:
                t = T = A = None
            else:
                t = complex(T0) + t_scat
                kz_sup_med = complex(n_super) * k0 * math.cos(theta)
                T = float(abs(t) ** 2 * (kz_sub.real / max(kz_sup_med.real, 1e-12)))
                A = float(1.0 - R - T)
    finally:
        _fit_ctx.acc = None
    fit_relres = max((v for _w, v in _fit_bands), default=0.0)      # exposed on OpticalResult
    _bad_fits = [(w, v) for w, v in _fit_bands if v > _FIT_RELRES_WARN]
    if _bad_fits:
        warnings.warn(_fit_warn_text(_bad_fits), FEMDiagnosticWarning, stacklevel=2)
    # Independent absorption diagnostic (audit OPT-2): the normalized volumetric loss
    # integral, computed from the reconstructed TOTAL field. Best-effort -- a diagnostic
    # must not break the solve, so a failure warns (not silent) and yields None.
    # diagnostics=False (opt-out, audit 6.2): skip this + per_region + flux R/T entirely.
    try:
        A_independent = (_absorbed_fraction(mesh, E_bg + gfu, eps_cf, k0, theta,
                                            geo.period_x_nm, geo.period_y_nm, n_super=n_super,
                                            roles=getattr(geo, "role_by_region", None))
                         if diagnostics else None)
    except Exception as _e:                                   # noqa: BLE001 (diagnostic)
        warnings.warn("independent absorption diagnostic unavailable: {}".format(_e))
        A_independent = None
    # Per-region absorbed-power map (driver D2): the same loss integral split by material region
    # (sums to A_independent by additivity). Best-effort diagnostic like A_independent.
    try:
        per_region_A = (None if A_independent is None else
                        _per_region_absorption(mesh, E_bg + gfu, eps_cf, k0, theta,
                                               geo.period_x_nm, geo.period_y_nm, n_super=n_super,
                                               roles=getattr(geo, "role_by_region", None)))
    except Exception as _e:                                   # noqa: BLE001 (diagnostic)
        warnings.warn("per-region absorption map unavailable: {}".format(_e))
        per_region_A = None
    # Fit-INDEPENDENT Poynting-flux R/T (audit B-fix): reads the FULL z-power straight from the
    # field, so it is correct even when the transmitted wave is elliptical (off-diagonal /
    # gyrotropic) and the single-projection lstsq fit cannot. Best-effort -- a diagnostic must not
    # break the solve. Skipped for the envelope formulation (its modified curl != mesh curl).
    # The flux band-averages Sz over the cladding buffer, which is z-CONSTANT only for LOSSLESS
    # cladding; a lossy super/substrate makes Sz decay through the buffer and biases the ratio (same
    # caveat _absorbed_fraction carries). Skip it (leave None) for the envelope formulation or a lossy
    # cladding rather than report a silently-biased "independent" R/T.
    lossless_clad = abs(complex(n_super).imag) < 1e-9 and abs(complex(n_sub).imag) < 1e-9
    try:
        if envelope or not lossless_clad or not diagnostics:
            R_flux = T_flux = None
        else:
            R_flux, T_flux = _poynting_flux_rt(fes, gfu, E_bg, E_inc, geo)
    except Exception as _e:                                   # noqa: BLE001 (diagnostic)
        warnings.warn("Poynting-flux R/T diagnostic unavailable: {}".format(_e))
        R_flux = T_flux = None
    # Backstop sanity: a bad fit / near-grazing case can push R or T well outside [0,1] or make
    # A=1-R-T strongly negative (energy created). Thresholds are LOOSE (5e-2): the documented
    # non-angle-aware-PML oblique error and ordinary FEM numerical error put A slightly negative
    # (~1-2%) on VALIDATED cases, so a tight bound would false-fire there. This backstop is for
    # GROSS violations only (a broken fit / grazing blow-up gives A ~ -0.2 or R/T >> 1); the
    # finer-grained signals are the fit-residual and energy-closure warnings above.
    for _nm, _v in (("R", R), ("T", T)):
        if _v is not None and (not math.isfinite(_v) or _v < -5e-2 or _v > 1.0 + 5e-2):
            warnings.warn("solve_fem: unphysical {}={} (well outside [0,1]); the solve/fit is "
                          "unreliable.".format(_nm, _v), FEMDiagnosticWarning, stacklevel=2)
    if A is not None and A < -5e-2:
        warnings.warn("solve_fem: unphysical A=1-R-T={:.4f} << 0 (energy created); R/T are "
                      "unreliable.".format(A), FEMDiagnosticWarning, stacklevel=2)
    # Energy-closure check: the budget A=1-R-T and the INDEPENDENTLY measured volumetric absorption
    # must agree for lossless cladding (audit OS-4). A large gap means the R/T extraction or the
    # field is inconsistent -- surface it on the SOLVE path (previously only validation scripts
    # compared the two). The 5e-2 band tolerates the OS-4 lossy-cladding case without false-firing
    # on the validated lossless ones.
    if (A is not None and A_independent is not None
            and (not math.isfinite(A_independent) or abs(A - A_independent) > 5e-2)):
        warnings.warn(
            "solve_fem: energy-closure mismatch -- budget A=1-R-T={:.4f} vs independently measured "
            "volumetric absorption A_independent={:.4f} (|diff|={:.2e} > 5e-2); R/T are "
            "inconsistent with the field. LIKELY CAUSE: HIGHER-ORDER DIFFRACTION -- R/T are 0-order "
            "(specular) only, so a period > lambda/n cell loses the diffracted power from R/T and it "
            "shows up as A (A_independent ~ 0 here flags exactly this; use R_flux/T_flux for the "
            "all-orders total, or a sub-wavelength cell). Other causes: a lossy cladding (A_independent "
            "qualitative) or an extraction error. Treat R/T/A as suspect.".format(
                A, A_independent, abs(A - A_independent)), FEMDiagnosticWarning, stacklevel=2)
    return OpticalResult(r=r, R=R, phase_deg=float(np.degrees(np.angle(r))),
                          solve_time_s=dt, t=t, T=T, A=A, A_independent=A_independent,
                          R_flux=R_flux, T_flux=T_flux, per_region_absorption=per_region_A,
                          fit_relres=fit_relres)


@dataclass
class SourcedResult:
    """Result of a source-driven FEM solve (solve_fem_sourced): the radiated field plus the
    per-port radiated power. Unlike OpticalResult (an incident-wave R/T), this describes the field
    RADIATED by an imposed current / equivalent-polarization source with NO incident wave.

    fes/gfu:   the periodic HCurl space and the solved SCATTERED GridFunction (equals the TOTAL
               field when bg_field is None).
    bg_field:  the analytic background E0 (a CoefficientFunction) the scattered field is measured
               on top of (the source's radiation in the uniform reference medium), or None.
    a_up/a_down: 0-order outgoing plane-wave amplitude (V/m, projected onto probe_pol) in the
               superstrate (up) and substrate (down); a_down is None with no substrate buffer.
    p_up/p_down: time-averaged radiated power through the top/bottom port INTEGRATED over the unit
               cell (Watts). Both are LOSSLESS-medium plane-wave fluxes: solve_fem_sourced RAISES on
               a lossy n_super, and p_down is None with no substrate buffer, with an evanescent
               0-order, or with a LOSSY n_sub (audit F-3).
    relres:    the iterative-solver relative residual (0.0 for a direct umfpack solve)."""
    fes:          object
    gfu:          object
    bg_field:     object
    a_up:         complex
    a_down:       "complex | None"
    p_up:         float
    p_down:       "float | None"
    solve_time_s: float
    relres:       float


def solve_fem_sourced(geo: OpticalGeometry, lambda_m: float,
                        eps_cf: ng.CoefficientFunction, optical: "OpticalSpec",
                        *, order: int = 2, n_super: complex = 1.0 + 0j,
                        n_sub: complex = 1.0 + 0j,
                        bg_field: "ng.CoefficientFunction | None" = None,
                        eps_ref: "complex | ng.CoefficientFunction | None" = None,
                        volume_current: "ng.CoefficientFunction | None" = None,
                        surface_currents: "dict | None" = None,
                        k_par_per_nm: "tuple[float, float]" = (0.0, 0.0),
                        probe_pol: "tuple[float, float, float]" = (1.0, 0.0, 0.0),
                        verbose: bool = False) -> SourcedResult:
    """Source-driven sibling of solve_fem: solve the SAME periodic layered curl-curl weak form,
    but driven by an imposed current / equivalent-polarization source instead of an incident plane
    wave, and return the radiated field + per-port radiated power. ADDITIVE -- solve_fem is
    untouched; this shares its PML / periodic-space / solver machinery. SI units, exp(-i omega t).

    The physical time-harmonic Maxwell curl-curl equation with a free current density J (SI, A/m^2)
    is (see e.g. Nireekshan Reddy et al., JOSA B 2017, Eq. 7)

        curl curl E - k0^2 eps E = i omega mu0 J = i k0 Z0 J        (Z0 = free-space impedance)

    In the solver's nm-length weak form (mesh coordinates in nm, k0 and ng.curl in nm^-1) the volume
    RHS is  i * k0 * Z0 * (J / S) . v  and a surface current sheet K (A/m) on a named boundary is
    i * k0 * Z0 * K . v_trace  (S = 1e9 nm/m; derived by rescaling curl_phys = S*ng.curl and
    dV_phys = dV_nm / S^3, dA_phys = dA_nm / S^2 -- the surface term carries no 1/S, the volume term
    one).

    TWO source routes:
      * SCATTERED-FIELD (preferred, WELL-CONDITIONED): pass bg_field = E0 (the source's radiation in
        a UNIFORM reference medium eps_ref, a homogeneous solution of curl curl E0 - k0^2 eps_ref E0
        = the source) and eps_ref. The FEM then solves the bounded scattered correction driven by
        k0^2 (eps - eps_ref) E0 -- byte-for-byte the same source structure solve_fem uses, so it
        inherits solve_fem's conditioning. The TOTAL field is E0 + gfu. This is the route the SHG
        two-step (shg_fem) uses: E0 is the analytic radiation of the surface-SHG sheet.
      * DIRECT current (volume_current J and/or surface_currents {bnd: K}): added straight to the
        RHS as above. ACCURACY (measured 2026-07-19, superseding an earlier 'near-null interior
        mode / ~2-2.5x low-loss bias' claim that adversarial verification could NOT reproduce):
        with the direct umfpack solve the extracted p_up matches independent current-sheet closed
        forms to ~0.3% at normal incidence and ~1-5.5% oblique (0-40 deg; the growth is the
        documented fixed-alpha z-PML oblique error) EVEN in an all-vacuum open cell, and to ~2-5%
        with a metal-backed sheet. Iterative solvers (gmres/bddc) may still struggle on low-loss
        curl-curl source problems -- watch relres -- but with a direct factorization both routes
        are quantitatively reliable.

    k_par_per_nm = (kx, ky) is the in-plane (Bloch) wavevector the source carries (0 at normal
    incidence; for surface-SHG at 2*omega it is 2 * k_par of the fundamental). probe_pol projects
    the field onto the polarization of interest for the 0-order amplitude fit -- it must be the
    FULL polarization unit vector of the outgoing wave for p_up to be the true flux: for a p-pol
    wave pass (cos th, 0, -sin th); a tangential-only (1,0,0) projection captures Ex = E cos(th)
    and under-counts the power by cos^2(th). Radiated power REQUIRES a LOSSLESS (vacuum/
    dielectric) super/substrate: p_up = |a|^2 (Re kz / k0)/(2 Z0) A is the plane-wave flux of a
    lossless medium, and a lossy superstrate makes it meaningless (measured: Im(eps_super) = 1
    inflates p_up ~5x vs the analytic sheet power; Im = 5 by ~260x). This lossless-superstrate
    constraint -- not solver conditioning -- is the real limit of the power read-out, and it is now
    ENFORCED (audit F-3): a lossy n_super RAISES NotImplementedError, mirroring solve_fem's
    _incidence_geometry guard (previously only solve_fem raised, so the drivers that reach this
    function WITHOUT a prior solve_fem call -- shg_structured_two_step via
    _reconstruct_fundamental_field -- passed a lossy superstrate end to end). A lossy n_sub is
    treated the way solve_fem treats its lossy-cladding flux diagnostic: p_down is returned as None
    (with a warning) rather than as a silently-wrong number; a_down (an amplitude, not a power) is
    still returned.

    REGIME GUARDS (audit F-13 -- this path used to carry NONE of solve_fem's, so a driver that
    reaches it WITHOUT a prior solve_fem call, i.e. shg_structured_two_step, ran unchecked):
      * TENSOR eps -> NotImplementedError. There is no UPML branch here and mesh.SetPML is called
        unconditionally, which eps_assembler documents as WRONG for an anisotropic medium; NGSolve
        would otherwise raise "Dimensions don't match" from deep inside the form assembly.
      * the radiated 0-order's polar angle, recovered from sin(theta) = |k_par|/(n_super k0), is
        held to the SAME ~50 deg PML envelope solve_fem enforces (and emits the same once-per-
        process non-angle-aware-PML advisory). sin(theta) > 1 (an evanescent 0-order, i.e. a
        near-field source) is not an angle and is left alone.
      * a port whose |kz| ~ 0 (grazing cutoff) -> NotImplementedError: the two-wave fit basis
        degenerates to rank 1 there, so the amplitude split is arbitrary."""
    if abs(complex(n_super).imag) > 1e-9:
        # p_up = |a|^2 (Re kz/k0)/(2 Z0) A is the plane-wave flux of a LOSSLESS medium; in a lossy
        # superstrate the 2-wave fit's "outgoing" amplitude is z-dependent and the flux formula
        # over-counts (measured 5.03x at Im(eps_super)=1, 66.5x at n_super=1+2j). solve_fem raises
        # on exactly this input (see _incidence_geometry) -- match it. (Audit F-3.)
        raise NotImplementedError(
            "solve_fem_sourced: the radiated-power read-out p_up = |a|^2 (Re kz/k0)/(2 Z0) A is the "
            "plane-wave flux of a LOSSLESS superstrate (Im(n_super)=0); got n_super={:.4g}, which "
            "inflates p_up by ~5x at Im(eps_super)=1 and ~260x at Im(eps_super)=5. Use a lossless "
            "(vacuum/dielectric) incidence medium -- solve_fem raises on the same input."
            .format(complex(n_super)))
    if tuple(getattr(eps_cf, "dims", ())) == (3, 3):
        # solve_fem routes a TENSOR eps through an explicit UPML (the anisotropic PML material
        # tensor folded into the weak form) precisely because mesh.SetPML's coordinate stretch is
        # WRONG for an anisotropic medium (eps_assembler: a resolution-INDEPENDENT ~3% error on the
        # decoupled component). This path has no UPML branch and calls mesh.SetPML unconditionally
        # below, and its scalar `eps_cf * (u*v)` mass term does not even ASSEMBLE against a 3x3 CF
        # (NGSolve raises "Dimensions don't match, op = -" from deep inside the form). Refuse HERE,
        # where the reason is visible. (Audit F-13.)
        raise NotImplementedError(
            "solve_fem_sourced: a TENSOR (3x3, anisotropic/gyrotropic) eps is not supported -- the "
            "sourced path has no UPML branch and mesh.SetPML's scalar coordinate stretch is wrong "
            "for an anisotropic medium. Use solve_fem (which builds the explicit UPML for tensor "
            "eps) for an incident-wave solve, or pass a scalar/diagonal eps here.")
    k0 = 2.0 * math.pi / (lambda_m * S)        # nm^-1
    mesh = geo.mesh
    Z0 = 1.0 / (EPS0 * C_LIGHT)                 # free-space wave impedance (ohm)
    kx, ky = float(k_par_per_nm[0]), float(k_par_per_nm[1])
    oblique = math.hypot(kx, ky) > 1e-12 * k0
    if oblique:
        # audit F-13: the sourced path carried NONE of solve_fem's incidence guards, so
        # shg_structured_two_step at 70 deg ran straight past the envelope where the same PML makes
        # solve_fem refuse. The source carries k_par instead of an OpticalSpec angle, so the
        # radiated 0-order's polar angle is recovered from the superstrate dispersion
        # sin(theta) = |k_par| / (n_super k0) and compared as a SINE (no asin round-trip at the cap).
        # sin > 1 means the 0-order is EVANESCENT in the superstrate -- a legitimate near-field
        # source problem, not an angle -- so the PML envelope simply does not apply there.
        n_sup_r = abs(complex(n_super).real)
        sin_th = math.hypot(kx, ky) / (n_sup_r * k0) if n_sup_r * k0 > 0.0 else float("inf")
        if sin_th <= 1.0:
            if sin_th > _FEM_SIN_THETA_MAX * (1.0 + 1e-12):
                raise NotImplementedError(
                    "solve_fem_sourced: the radiated 0-order leaves at |theta| = {:.1f} deg "
                    "(|k_par| = {:.4g} nm^-1, n_super = {:.4g}), beyond the FEM's validated "
                    "envelope (~{:.0f} deg) -- the fixed-alpha HalfSpace z-PML is not angle-aware "
                    "and VIOLATES energy conservation above it (R+T ~ 1.17 at 60 deg), so p_up/"
                    "p_down would be silently wrong. solve_fem raises on the same geometry."
                    .format(math.degrees(math.asin(min(1.0, sin_th))), math.hypot(kx, ky),
                            complex(n_super), _FEM_THETA_MAX_DEG))
            _advise_once("oblique_pml", _OBLIQUE_PML_ADVICE, stacklevel=2)

    # ---- PML: clone solve_fem's scalar HalfSpace z-stretch (this ordering -- UnSetPML then SetPML
    # BEFORE building the FESpace/forms -- matters; a SetPML after the forms silently zeros the RHS).
    z_air_top = geo.z_super_interface_nm
    z_sub_top = geo.z_sub_interface_nm
    try:
        mesh.UnSetPML("pml_top"); mesh.UnSetPML("pml_bot")
    except Exception:
        pass
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_air_top), normal=(0, 0, 1), alpha=1j), "pml_top")
    mesh.SetPML(ng.pml.HalfSpace(point=(0, 0, z_sub_top), normal=(0, 0, -1), alpha=1j), "pml_bot")

    # ---- periodic HCurl space (quasi-periodic Bloch phases if the source carries k_par) ----
    bloch_phased = bool(oblique and (geo.n_px or geo.n_py))       # audit F-9
    if bloch_phased:
        phases = _bloch_phase_list(geo, kx, ky)
        fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""), phase=phases)
    else:
        fes = ng.Periodic(ng.HCurl(mesh, order=order, complex=True, dirichlet=""))
    u, v = fes.TrialFunction(), fes.TestFunction()

    # audit F-9: symmetric=True was a FALSE statement whenever the source carries k_par -- the
    # Bloch-phased space conjugates the phase on the test side, so the assembled matrix is
    # Hermitian-not-symmetric (real eps) or neither (lossy eps). Inert in ngsolve 6.2.2604 but it
    # must not be asserted; see the matching note in solve_fem.
    a = ng.BilinearForm(fes, symmetric=not bloch_phased)
    a += (ng.curl(u) * ng.curl(v) - k0 ** 2 * eps_cf * (u * v)) * ng.dx
    f = ng.LinearForm(fes)
    if bg_field is not None:
        # scattered-field source k0^2 (eps - eps_ref) E0 -- nonzero only where eps != eps_ref (the
        # structure). eps_ref may be a scalar (uniform reference) or a CF.
        eps_ref_cf = eps_ref if isinstance(eps_ref, ng.CoefficientFunction) else \
            ng.CoefficientFunction(complex(eps_ref if eps_ref is not None else 1.0))
        f += (k0 ** 2 * (eps_cf - eps_ref_cf) * (bg_field * v)) * ng.dx
    if volume_current is not None:
        f += (1j * k0 * Z0 * (volume_current * v) / S) * ng.dx
    if surface_currents:
        # audit F-8 (same silent no-op as solve_fem's sheet_bcs): an unknown boundary name here
        # assembles ||f|| = 0, i.e. the imposed sheet current just is not there and the "radiated"
        # power comes back as the source-free answer.
        _validate_sheet_bcs(mesh, surface_currents)
        for _bnd, _K in surface_currents.items():
            f += (1j * k0 * Z0 * (_K * v.Trace())) * ng.ds(definedon=mesh.Boundaries(_bnd))

    _ls = optical.linear_solver
    if _ls == "bddc_cg" and bloch_phased:
        _advise_once(                                          # audit F-9 (see solve_fem)
            "bddc_cg_nonsymmetric",
            "linear_solver='bddc_cg' is invalid for this solve: a k_par-carrying source makes the "
            "space quasi-periodic (Bloch-phased) and the assembled matrix non-symmetric, while "
            "ng.solvers.CGSolver runs COCG, which assumes complex-SYMMETRIC. Falling back to "
            "'bddc_gmres' (same BDDC preconditioner). [reported once per process]")
        _ls = "bddc_gmres"
    pre = ng.Preconditioner(a, "bddc") if _ls.startswith("bddc") else None
    gfu = ng.GridFunction(fes)
    t0 = time.time()
    with ng.TaskManager():
        a.Assemble(); f.Assemble()
        if _ls == "umfpack":
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec
        elif _ls == "bddc_cg":
            inv = ng.solvers.CGSolver(mat=a.mat, pre=pre.mat, tol=optical.gmres_rtol,
                                        maxiter=optical.gmres_max_iter)
            gfu.vec.data = inv * f.vec
        else:
            ng.solvers.GMRes(A=a.mat, b=f.vec, pre=pre.mat, x=gfu.vec,
                                tol=optical.gmres_rtol, maxsteps=optical.gmres_max_iter,
                                printrates=verbose)
    dt = time.time() - t0

    rvec = gfu.vec.CreateVector()
    rvec.data = f.vec - a.mat * gfu.vec
    bn = float(np.linalg.norm(f.vec.FV().NumPy()))
    relres = float(np.linalg.norm(rvec.FV().NumPy())) / bn if bn > 1e-300 else 0.0
    if relres > 1e-3:
        warnings.warn(
            "solve_fem_sourced: solver '{}' did not converge (relative residual {:.2e} > 1e-3). The "
            "curl-curl operator is near-singular for low-loss / open-cavity source problems (the "
            "DIRECT current route especially); use the scattered-field route (bg_field/eps_ref) with "
            "a metal-filled cell, linear_solver='umfpack', or refine.".format(
                optical.linear_solver, relres), stacklevel=2)

    # ---- radiated 0-order amplitudes + per-port power (total field = bg_field + gfu) ----
    total = gfu if bg_field is None else (bg_field + gfu)
    Px, Py = geo.period_x_nm, geo.period_y_nm
    proj = (complex(probe_pol[0]), complex(probe_pol[1]), complex(probe_pol[2]))
    kz_s = complex(cmath.sqrt((complex(n_super) * k0) ** 2 - kx ** 2 - ky ** 2))
    if kz_s.imag < 0:
        kz_s = -kz_s
    _refuse_grazing_port(kz_s, k0, "superstrate", n_super, kx, ky)          # audit F-13
    a_up = _sourced_port_amp(mesh, total, geo, kz_s, proj, kx, ky, "superstrate", +1)
    area_phys = (Px * Py) / S ** 2
    p_up = (abs(a_up) ** 2 * (kz_s.real / k0) / (2.0 * Z0)) * area_phys
    a_down = p_down = None
    if "substrate" in geo.z_intervals_nm:
        kz_b = complex(cmath.sqrt((complex(n_sub) * k0) ** 2 - kx ** 2 - ky ** 2))
        if kz_b.imag < 0:
            kz_b = -kz_b
        _refuse_grazing_port(kz_b, k0, "substrate", n_sub, kx, ky)          # audit F-13
        a_down = _sourced_port_amp(mesh, total, geo, kz_b, proj, kx, ky, "substrate", -1)
        if abs(complex(n_sub).imag) > 1e-9:
            # Same lossless-medium restriction as p_up, on the DOWN port. A metal-backed cell
            # (n_sub = sqrt(eps_metal)) is a legitimate, common configuration whose UP-port power is
            # still valid, so this flags/omits p_down instead of raising (mirroring solve_fem, which
            # leaves its R_flux/T_flux diagnostic None for a lossy cladding). (Audit F-3.)
            # audit T-13: FEMDiagnosticWarning, not a bare UserWarning. This is an ADVISORY about
            # the SOLVE REGIME (a metal-backed cell is a legitimate configuration and the result is
            # correct -- p_down is simply omitted), which is exactly what the category exists for;
            # a plain UserWarning would mean "the caller made a mistake". Categorising it here is
            # what let pyproject's temporary message-scoped exemption be deleted: conftest's
            # conditional `ignore::FEMDiagnosticWarning` now covers it alongside the solver's other
            # advisories, and a caller silencing the class silences this one too.
            warnings.warn(
                "solve_fem_sourced: p_down is undefined for a LOSSY substrate (Im(n_sub)={:.3g} != 0) "
                "-- the plane-wave flux formula assumes a lossless medium -- so it is returned as "
                "None rather than a silently-wrong power. a_up/p_up (lossless superstrate) and "
                "a_down are unaffected.".format(complex(n_sub).imag),
                FEMDiagnosticWarning, stacklevel=2)
        elif kz_b.real > 1e-9 * k0:
            # lossless-substrate z-flux of a tangential-field plane wave (valid for a vacuum/
            # dielectric substrate); the SHG use case radiates into a vacuum superstrate.
            p_down = (abs(a_down) ** 2 * (kz_b.real / k0) / (2.0 * Z0)) * area_phys
    return SourcedResult(fes=fes, gfu=gfu, bg_field=bg_field, a_up=complex(a_up),
                          a_down=(None if a_down is None else complex(a_down)),
                          p_up=float(p_up), p_down=(None if p_down is None else float(p_down)),
                          solve_time_s=dt, relres=relres)


def _refuse_grazing_port(kz, k0, region, n_med, kx, ky):
    """RAISE if a port's 0-order sits at the grazing cutoff, |kz| ~ 0 (audit F-13).

    The port amplitude comes from a two-wave fit onto the basis exp(+i kz z), exp(-i kz z). At
    |kz| ~ 0 BOTH columns collapse to 1, the design matrix is rank-1, and the up/down split lstsq
    returns is arbitrary -- so a_up/a_down (and p_up = |a|^2 Re kz/k0 ...) are numerical noise, not
    a small number. This is the sourced mirror of solve_fem's substrate grazing refusal. A DEEPLY
    evanescent 0-order keeps |kz| LARGE (purely imaginary) and is NOT caught here: there the two
    columns are a decaying and a growing exponential, well separated, and p_up ~ 0 is correct."""
    if abs(kz) <= 1e-6 * k0:
        raise NotImplementedError(
            "solve_fem_sourced: the {} 0-order is at the grazing cutoff (|kz| = {:.3e} nm^-1 ~ 0): "
            "the two-wave port fit basis exp(+-i kz z) degenerates to a rank-1 DC fit, so the "
            "radiated amplitude/power there is arbitrary. n_med = {:.4g}, |k_par| = {:.4g} nm^-1. "
            "Nudge the source k_par / wavelength off the exact cutoff. (solve_fem refuses the same "
            "geometry on its substrate order.)".format(
                region, abs(kz), complex(n_med), math.hypot(kx, ky)))


def _sourced_port_amp(mesh, total_field, geo: OpticalGeometry, kz, proj, kx, ky, region, sgn):
    """0-order outgoing amplitude in a super/substrate buffer: 2-wave fit of the (proj-projected,
    demodulated) TOTAL field to sgn*outgoing exp(sgn i kz z) + residual. sgn=+1 up (superstrate),
    -1 down (substrate). Reuses solve_fem's _cell_average (probe-grid sized for the medium)."""
    z0, z1 = geo.z_intervals_nm[region]
    pad = max(50.0, 0.1 * (z1 - z0))
    zlo, zhi = z0 + pad, z1 - pad
    if zhi <= zlo:
        zlo, zhi = z0 + 0.2 * (z1 - z0), z0 + 0.8 * (z1 - z0)
    zr = np.linspace(zlo, zhi, 7)
    # the band is padded symmetrically, so the structure-to-nearest-probe standoff is that pad
    # whichever port this is (structure BELOW the superstrate band, ABOVE the substrate one).
    Es = _cell_average(mesh, total_field, zr, geo.period_x_nm, geo.period_y_nm, proj, kx, ky,
                        kz_med=kz, standoff_nm=(zlo - z0 if sgn > 0 else z1 - zhi))   # F-10
    M = np.column_stack([np.exp(1j * sgn * kz * zr), np.exp(-1j * sgn * kz * zr)])
    coeffs = _lstsq_2wave(M, Es, where="{} radiated".format(region))
    return complex(coeffs[0])


def _probe_grid_sizes(Px, Py, kx, ky, kz_med, standoff_nm=None,
                      decay_nepers=_ALIAS_DECAY_NEPERS):
    """Per-direction probe-grid sizes so the first aliased Fourier order is DECAYED (not merely
    evanescent) at the probe planes (audit C3-1, tightened by F-10).

    An N-point cell-centred grid aliases orders m = 0 (mod N) into the reported 0-order
    coefficient with weight (-1)^(m/N), so the first surviving alias has in-plane wavevector at
    least `N*2pi/P - |k_lat|` and decay constant kappa = sqrt(that^2 - (n k0)^2). The old rule
    only required kappa > 0: for `P (n k0 + kx)/(2 pi)` just under an integer the margin is ~0.2%,
    kappa ~ 0.06 n k0, and at lambda = 1550 nm the alias survives the 50 nm standoff at
    exp(-0.012) = 0.988 -- essentially UNDAMPED, i.e. the guarantee was vacuous near cutoff
    (audit F-10). The criterion is now a DECAY one: pick the smallest N with

        kappa * standoff >= decay_nepers   <=>   N*2pi/P - |k_lat| >= sqrt((n k0)^2 + (d/s)^2),

    i.e. e^-3 = 5% suppression of the first alias at the NEAREST probe plane by default. n*k0 is
    recovered from the medium dispersion n^2 k0^2 = kz_med^2 + k_par^2; `standoff_nm` is the gap
    between the structure (where the evanescent orders are excited) and the nearest probe plane.
    standoff_nm=None keeps the legacy evanescent-only rule.

    COST. The decay term alone asks for a probe pitch of about (2 pi / decay_nepers) * standoff,
    i.e. ~105 nm at a 50 nm standoff and ~63 nm at 30 nm, INDEPENDENTLY of wavelength -- so N grows
    linearly with the period and the point-evaluation count as N^2. A THIN PAD is what bites, since
    the transmission-side standoff is the 10 % substrate_buffer pad: at the DEFAULT
    substrate_buffer = 100 nm (pad = 10 nm) a 1000 nm cell asks for (48, 48) instead of (6, 6) --
    64x the point evaluations and ~+40 % wall on a shipped solve.

    N IS THEREFORE CAPPED AT `max(_PROBE_GRID_MAX, N_evanescent)`, with a warning (audit F-10
    residual). The cap bounds the DECAY refinement only -- it is floored at the evanescent-only size
    the C3-1 rule asks for, so a supercell that legitimately needs many points because it has many
    PROPAGATING orders is never capped below that (capping there would silently re-open C3-1), and
    the `standoff_nm=None` legacy path is untouched by the cap entirely. The claim
    that "sub-wavelength cells stay at the legacy 6x6" is FALSE as stated -- it holds for a 400 nm
    cell at a 50 nm standoff (byte-identical), but 163 of 576 sub-wavelength (period < lambda/n)
    configurations at that standoff already exceed 6, up to N = 11 at lambda = 1064 nm -- so the
    floor cannot be relied on for cost. Above the cap the criterion is not met and the FIRST alias
    is suppressed by less than e^-3 at the nearest probe plane; the fix is to THICKEN the buffer
    (the standoff enters as ~1/standoff, so 10 nm -> 90 nm takes N from 48 back to 6) rather than to
    pay N^2 point evaluations for a geometry whose probes are too close to the structure anyway.
    Pass `decay_nepers=0` (or `standoff_nm=None`) to opt out of the decay criterion entirely.

    ALIAS SUPPRESSION AT THE CAP: with N capped, the residual alias amplitude is
    exp(-kappa*standoff) with kappa from the CAPPED N -- the caller should treat R/T as
    alias-contaminated at the level the warning quotes."""
    nk0_ev = float(np.hypot(abs(complex(kz_med)), float(np.hypot(kx, ky))))
    decay_on = (standoff_nm is not None and float(standoff_nm) > 0.0 and decay_nepers > 0.0)
    # required lateral wavevector: sqrt((n k0)^2 + kappa_min^2), kappa_min = nepers / standoff
    nk0 = (float(np.hypot(nk0_ev, float(decay_nepers) / float(standoff_nm))) if decay_on else nk0_ev)
    nx_g = max(6, int(np.floor(Px * (nk0 + abs(float(kx))) / (2.0 * np.pi))) + 1)
    ny_g = max(6, int(np.floor(Py * (nk0 + abs(float(ky))) / (2.0 * np.pi))) + 1)
    # The cap bounds the F-10 DECAY refinement only. The C3-1 requirement -- make the first alias
    # EVANESCENT -- is never weakened: a supercell with many propagating orders legitimately needs a
    # large N, and capping that would silently re-open C3-1 (a propagating order aliased into the
    # 0-order coefficient at full weight). So the cap is floored at the evanescent-only size, and the
    # legacy standoff_nm=None path is untouched.
    nx_ev = max(6, int(np.floor(Px * (nk0_ev + abs(float(kx))) / (2.0 * np.pi))) + 1)
    ny_ev = max(6, int(np.floor(Py * (nk0_ev + abs(float(ky))) / (2.0 * np.pi))) + 1)
    cap_x = max(_PROBE_GRID_MAX, nx_ev)
    cap_y = max(_PROBE_GRID_MAX, ny_ev)
    if decay_on and (nx_g > cap_x or ny_g > cap_y):
        warnings.warn(
            "_probe_grid_sizes: the e^-{:.0f} alias-decay criterion asks for a ({}, {}) probe grid "
            "at a {:.3g} nm standoff, above the {}x{} cap -- capping it, so the first aliased order "
            "is NOT suppressed to {:.1%} at the nearest probe plane and R/T carry that alias. This "
            "is the THIN-PAD regime: the requirement grows as ~1/standoff (and the point-evaluation "
            "count as N^2, e.g. 64x and ~+40 % wall for a 1000 nm cell at the default 10 nm "
            "transmission pad). THICKEN the buffer instead -- Mesh3DSpec(substrate_buffer_m=...) / "
            "superstrate_buffer_m -- which fixes the physics rather than paying for it (a 90 nm pad "
            "takes that same cell back to 6x6). audit F-10.".format(
                float(decay_nepers), nx_g, ny_g, float(standoff_nm or 0.0),
                cap_x, cap_y, float(np.exp(-float(decay_nepers)))),
            FEMDiagnosticWarning, stacklevel=3)
        nx_g = min(nx_g, cap_x)
        ny_g = min(ny_g, cap_y)
    return nx_g, ny_g


def _cell_average(mesh, field, z_probes, Px, Py, proj, kx, ky, kz_med=None, standoff_nm=None):
    """Transverse (x,y) cell-average of (proj . field), demodulated by exp(-i(kx x+ky y)),
    at each z. `proj` is a length-3 weight vector projecting the vector field onto the
    polarization of interest (the s-pol unit vector, or (1,0,0) for tangential Ex). The
    demod removes the transverse Bloch phase; on the N x N cell-centred grid the average
    equals the 0-order Fourier coefficient PLUS aliases of orders m = 0 (mod N) (audit
    C3-1: NOT exact orthogonality, as previously claimed -- a propagating substrate
    order-6 amplitude at 30% of r0 corrupted the fit silently at the old fixed 6x6).
    Passing kz_med (the fitted wave's medium z-wavevector) sizes the grid so every
    aliased order is evanescent at the probe planes; None keeps the legacy 6x6.
    standoff_nm (audit F-10) is the gap from the structure to the NEAREST probe plane; passing it
    upgrades that sizing from "evanescent" to "decayed by e^-3 at the probe" (see
    _probe_grid_sizes -- near cutoff the two are not remotely the same guarantee)."""
    # cell-centred probe grid: offset off the x=0 / y=0 periodic-boundary lines (where
    # quasi-periodic point evaluation can fail) by half a step (audit OPT-7).
    if kz_med is not None:
        nx_g, ny_g = _probe_grid_sizes(Px, Py, kx, ky, kz_med, standoff_nm=standoff_nm)
    else:
        nx_g = ny_g = 6
    xs = (np.arange(nx_g) + 0.5) * (Px / nx_g)
    ys = (np.arange(ny_g) + 0.5) * (Py / ny_g)
    p0, p1, p2 = proj
    out = []
    for zv in z_probes:
        vals = []
        for xv in xs:
            for yv in ys:
                try:
                    E = field(mesh(float(xv), float(yv), float(zv)))
                    proj_val = p0 * complex(E[0]) + p1 * complex(E[1]) + p2 * complex(E[2])
                    vals.append(proj_val * np.exp(-1j * (kx * xv + ky * yv)))
                except Exception:
                    pass
        if not vals:
            # A whole z-plane with zero valid samples means the probe points all fell
            # outside the mesh (a geometry/units contract break) -- fail loudly instead
            # of feeding a silent 0+0j into the least-squares R/T fit (audit OPT-7/F6).
            raise RuntimeError(
                "cell-average got no valid field samples at z={:.3f} nm; the probe grid "
                "missed the mesh (check the geometry/units and z-interval bounds).".format(
                    float(zv)))
        out.append(complex(np.mean(vals)))
    return np.array(out)


def _lstsq_2wave(M, Es, *, where):
    """Two-wave (up/down) least-squares fit M @ c = Es WITH a goodness-of-fit guard. lstsq
    silently returns the best 2-parameter projection even when Es is NOT a clean two-wave field
    (buffer too thin -> undecayed diffraction orders, or PML leak-back), giving a silently-wrong
    0-order coefficient. Flag when the relative residual exceeds _FIT_RELRES_WARN so the resulting
    R/T are not trusted. Returns the fit coefficients.

    audit F-15: inside a solve_fem the residual is APPENDED to that solve's accumulator and the
    warning is emitted ONCE for the whole solve, naming every offending band -- a p-pol solve fits
    four bands (reflection/transmission x up/down) and used to emit up to four near-identical
    warnings for one cause. Outside a solve_fem (direct use) it warns immediately, as before."""
    coeffs, *_ = np.linalg.lstsq(M, Es, rcond=None)
    denom = float(np.linalg.norm(Es))
    if denom > 1e-300:
        relres = float(np.linalg.norm(np.asarray(Es) - M @ coeffs) / denom)
        acc = getattr(_fit_ctx, "acc", None)
        if acc is not None:
            acc.append((where, relres))
        elif relres > _FIT_RELRES_WARN:
            warnings.warn(_fit_warn_text([(where, relres)]), FEMDiagnosticWarning, stacklevel=3)
    return coeffs


def _fit_warn_text(bad) -> str:
    return ("solve_fem two-wave 0-order fit residual above {:.0e} on {}: {} -- the probe band(s) "
            "are not a clean up/down field (super/substrate buffer too thin, undecayed diffraction "
            "orders, or PML leak-back); R/T are unreliable. Thicken the buffer or refine the mesh."
            .format(_FIT_RELRES_WARN,
                    "1 band" if len(bad) == 1 else "{} bands".format(len(bad)),
                    ", ".join("{}={:.2e}".format(w, v) for w, v in bad)))


def _reflection(mesh, gfu, kz_s, proj, kx, ky, geo: OpticalGeometry) -> complex:
    """0-order reflection: least-squares fit of the cell-averaged (proj-projected,
    demodulated) scattered field over z-planes in the superstrate buffer."""
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z_struct_top = geo.z_intervals_nm["superstrate"][0]
    z_air_top = geo.z_intervals_nm["superstrate"][1]
    z_lo = z_struct_top + 50.0
    z_hi = z_air_top - 50.0
    if z_hi <= z_lo:
        z_lo = z_struct_top + 0.2 * (z_air_top - z_struct_top)
        z_hi = z_struct_top + 0.8 * (z_air_top - z_struct_top)
    z_probes = np.linspace(z_lo, z_hi, 7)
    Es = _cell_average(mesh, gfu, z_probes, Px, Py, proj, kx, ky, kz_med=kz_s,
                        standoff_nm=(z_lo - z_struct_top))                       # C3-1 / F-10
    # upward (reflected) exp(+i kz_s z) + residual downward exp(-i kz_s z)
    M = np.column_stack([np.exp(+1j * kz_s * z_probes), np.exp(-1j * kz_s * z_probes)])
    coeffs = _lstsq_2wave(M, Es, where="reflection")
    return complex(coeffs[0])


def _transmission(mesh, gfu, kz_sub, proj, kx, ky, geo: OpticalGeometry):
    """0-order SCATTERED transmission amplitude: fit the cell-averaged scattered field
    in the substrate buffer to a downward wave exp(-i kz_sub z). The background
    transmission T0 is added analytically by the caller. Returns None if there is no
    usable substrate buffer."""
    if "substrate" not in geo.z_intervals_nm:
        return None
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z_sub_lo, z_sub_hi = geo.z_intervals_nm["substrate"]
    pad = 0.1 * (z_sub_hi - z_sub_lo)
    z_lo, z_hi = z_sub_lo + pad, z_sub_hi - pad
    if z_hi <= z_lo:
        return None
    z_probes = np.linspace(z_lo, z_hi, 7)
    # the structure sits ABOVE the substrate buffer, so the nearest-probe standoff is the top pad
    Es = _cell_average(mesh, gfu, z_probes, Px, Py, proj, kx, ky, kz_med=kz_sub,
                        standoff_nm=(z_sub_hi - z_hi))                            # C3-1 / F-10
    # downward (transmitted) exp(-i kz_sub z) + any upward residual exp(+i kz_sub z)
    M = np.column_stack([np.exp(-1j * kz_sub * z_probes), np.exp(+1j * kz_sub * z_probes)])
    coeffs = _lstsq_2wave(M, Es, where="transmission")
    return complex(coeffs[0])


def _ppol_extract(mesh, E_tot, kz_s, kz_sub, kx, ky, proj_t, geo: OpticalGeometry, eps_sup, eps_sub):
    """p-pol R/T from the reconstructed TOTAL field (E_bg + scattered), projecting onto the in-plane
    p-pol direction proj_t = (cos phi, sin phi, 0) and 2D-demodulating by exp(-i(kx x + ky y)).
    R = |E_up/E_down|^2 in the superstrate (the eps/kz Poynting factors cancel in the same medium);
    T = |E_down_sub/E_down_super|^2 * Re((eps_sub/eps_sup)(kz_s/kz_sub)) -- the p-pol z-flux Sz ~
    |E_t|^2 eps/kz. proj_t=(1,0,0), ky=0 recovers the phi=0 tangential-Ex extraction. Returns (r,R,t,T)."""
    Px, Py = geo.period_x_nm, geo.period_y_nm
    z0, z1 = geo.z_intervals_nm["superstrate"]
    zlo, zhi = z0 + 50.0, z1 - 50.0
    if zhi <= zlo:
        zlo, zhi = z0 + 0.2 * (z1 - z0), z0 + 0.8 * (z1 - z0)
    zr = np.linspace(zlo, zhi, 7)
    Exr = _cell_average(mesh, E_tot, zr, Px, Py, proj_t, kx, ky, kz_med=kz_s,
                         standoff_nm=(zlo - z0))            # in-plane p-pol (C3-1 / F-10)
    Mr = np.column_stack([np.exp(-1j * kz_s * zr), np.exp(+1j * kz_s * zr)])
    cr = _lstsq_2wave(Mr, Exr, where="p-pol reflection")
    a_d, a_u = complex(cr[0]), complex(cr[1])                     # incident-dir, reflected-dir E_t
    r = a_u / a_d if abs(a_d) > 1e-30 else 0j
    R = float(abs(r) ** 2)
    if "substrate" not in geo.z_intervals_nm:
        return r, R, None, None
    zs_lo, zs_hi = geo.z_intervals_nm["substrate"]
    pad = 0.1 * (zs_hi - zs_lo)
    if zs_hi - pad <= zs_lo + pad:
        return r, R, None, None
    zt = np.linspace(zs_lo + pad, zs_hi - pad, 7)
    Ext = _cell_average(mesh, E_tot, zt, Px, Py, proj_t, kx, ky, kz_med=kz_sub,
                         standoff_nm=pad)                                        # C3-1 / F-10
    Mt = np.column_stack([np.exp(-1j * kz_sub * zt), np.exp(+1j * kz_sub * zt)])
    ct = _lstsq_2wave(Mt, Ext, where="p-pol transmission")
    t = complex(ct[0]) / a_d if abs(a_d) > 1e-30 else 0j          # transmitted Ex / incident Ex
    factor = (eps_sub / eps_sup) * (kz_s / kz_sub)
    T = float(abs(t) ** 2 * factor.real)
    return r, R, t, T


def _non_pml_regions(mesh, roles=None):
    """Mesh material (= region) names to integrate the absorption over: everything EXCEPT the PML.

    Route on the recorded structural ROLE when the geometry carries one (audit F-7): the builder
    tags each region 'pml' | 'substrate' | ... where it creates it, so a user layer legitimately
    named 'pml_calibration_film' stays in the absorption budget. The 'pml_' NAME prefix is only the
    fallback for a geometry built by other means (a subclass builder, a hand-assembled mesh) -- it
    was the sole rule before, and it silently dropped any user layer wearing that prefix from both
    A_independent and the per-region map while still meshing it."""
    mats = list(dict.fromkeys(mesh.GetMaterials()))
    if roles:
        return [m for m in mats if roles.get(m, "") != "pml"]
    return [m for m in mats if not m.startswith("pml_")]


def _absorbed_fraction(mesh, E_tot, eps_cf, k0, theta, Px, Py, n_super=1.0 + 0j, roles=None):
    """Independently measured absorbed fraction (audit OPT-2): the normalized
    volumetric loss integral
        A = k0 * Int_V Im(eps) |E|^2 dV / (Re(n_super) * cos(theta) * cell_area),
    over the PHYSICAL (non-PML) domain, for a unit-amplitude incident plane wave.
    audit C3-7: the incident power in a dense superstrate is ~ n_super cos(theta) area,
    so the normalization carries Re(n_super) (it used to omit it, inflating A -- and the
    D2 per-region deposition -- by exactly Re(n_super) for a dense encapsulant; vacuum
    incidence, the only supported oblique case, is byte-identical). This
    is a genuine measurement (not 1-R-T): comparing it to the budget closure 1-R-T
    catches energy/numerics errors that the R/T extraction alone cannot. The PML
    materials are excluded -- their stretched-coordinate eps would corrupt the integral
    (and a lossy substrate makes the bottom-PML contribution spurious). NOTE: a lossy
    super/substrate BUFFER (between the structure and its PML) is still integrated, so A is
    a clean measurement only for LOSSLESS cladding media (the validated cases); for a lossy
    cladding treat A as qualitative (audit OS-4)."""
    # Exclude PML regions only (see _non_pml_regions: role-routed, name-prefix fallback -- a
    # physical material like 'pmlayer' is NOT dropped, OS-3). re.escape each surviving name so a
    # regex-metacharacter material name (e.g. 'ito.n+') is matched literally rather than silently
    # missed by mesh.Materials' regex (OS-2).
    non_pml = _non_pml_regions(mesh, roles)
    if not non_pml:
        return None
    defon = mesh.Materials("|".join(re.escape(m) for m in non_pml))
    # Use the plain vector product a*b (the sum a_i b_i, NO conjugation) with an explicit Conj(E_tot),
    # NOT ng.InnerProduct(.,Conj(E_tot)): InnerProduct now conjugates its 2nd argument, and the code
    # only worked because NGSolve happens to detect the already-Conj arg and skip the extra conjugate
    # (the "c2 is already a Conjugate" notice). a*Conj(E) makes E^* . (.) explicit and version-robust.
    if tuple(getattr(eps_cf, "dims", ())) == (3, 3):
        # tensor eps: absorbed power density ~ Im(E^* . eps . E) (the scalar Im(eps)|E|^2 analog;
        # reduces to it for eps = eps_scalar * I)
        q = (eps_cf * E_tot) * ng.Conj(E_tot)                 # E^* . eps . E
        loss = (q - ng.Conj(q)) / 2j
    else:
        im_eps = (eps_cf - ng.Conj(eps_cf)) / 2j              # Im(eps) as a real CF
        loss = im_eps * (E_tot * ng.Conj(E_tot))              # Im(eps) |E|^2
    integ = ng.Integrate(loss, mesh, definedon=defon)
    area = float(Px) * float(Py)
    if area <= 0:
        return None
    n_cos = max(complex(n_super).real * math.cos(theta), 1e-12)   # C3-7 incident-power factor
    return float(complex(integ).real * k0 / (n_cos * area))


def _per_region_absorption(mesh, E_tot, eps_cf, k0, theta, Px, Py, n_super=1.0 + 0j, roles=None):
    """Per-region absorbed-power map (driver D2): the _absorbed_fraction integrand evaluated
    region by region -- IDENTICAL loss CF, IDENTICAL normalization, restricted to one material
    domain at a time -- so sum(values) equals A_independent EXACTLY (domain additivity of the
    integral), each value is the fraction of the incident power deposited in that region, and a
    region with Im(eps) = 0 contributes exactly 0. Same caveats as _absorbed_fraction (clean
    only for lossless cladding; PML regions excluded). Returns {region_name: fraction}."""
    non_pml = _non_pml_regions(mesh, roles)
    area = float(Px) * float(Py)
    if not non_pml or area <= 0:
        return None
    if tuple(getattr(eps_cf, "dims", ())) == (3, 3):
        q = (eps_cf * E_tot) * ng.Conj(E_tot)                 # E^* . eps . E (see _absorbed_fraction)
        loss = (q - ng.Conj(q)) / 2j
    else:
        im_eps = (eps_cf - ng.Conj(eps_cf)) / 2j
        loss = im_eps * (E_tot * ng.Conj(E_tot))
    # audit C3-7: same Re(n_super) incident-power factor as _absorbed_fraction (the two must
    # stay IDENTICAL for the sum-equals-A_independent additivity contract)
    scale = k0 / (max(complex(n_super).real * math.cos(theta), 1e-12) * area)
    out = {}
    for m in non_pml:
        integ = ng.Integrate(loss, mesh, definedon=mesh.Materials(re.escape(m)))
        out[m] = float(complex(integ).real * scale)
    return out


def _poynting_flux_rt(fes, gfu, E_bg, E_inc, geo: OpticalGeometry):
    """Fit-INDEPENDENT R/T from the time-averaged z-Poynting flux of the reconstructed TOTAL
    field. Sz = 0.5 Re(Ex Hy* - Ey Hx*) with H = curl(E)/(i omega mu0); the omega*mu0 and the
    nm->m curl-scale constants are COMMON to numerator and the incident reference, so they cancel
    in the flux RATIO and we use the bare mesh curl. This is to the up/down least-squares fit what
    _absorbed_fraction is to 1-R-T: a second, independent measurement.

    Why it matters for the off-diagonal / gyrotropic tensor: the transmitted field is elliptical
    (co- AND cross-polarized), and the single-projection _transmission fit measures only the
    co-pol amplitude -- so it can report T > 1 / energy non-closure. The Poynting flux integrates
    the FULL z-power (every component), so R_flux + T_flux closes for a lossless slab regardless of
    polarization mixing. R = 1 - <Sz>_super/<Sz>_inc, T = <Sz>_sub/<Sz>_inc (incident is down-going,
    Sz < 0; the ratios are positive). Returns (R_flux, T_flux); T_flux is None with no substrate."""
    mesh = geo.mesh
    Etot = ng.GridFunction(fes); Etot.Set(E_bg); Etot.vec.data += gfu.vec      # interp(bg) + scattered
    Einc = ng.GridFunction(fes); Einc.Set(E_inc)                               # bare incident reference

    def _sz(E):
        H = ng.curl(E) / 1j                                  # proxy H (omega*mu0, S cancel in the ratio)
        w = E[0] * ng.Conj(H[1]) - E[1] * ng.Conj(H[0])      # Ex Hy* - Ey Hx*
        return 0.25 * (w + ng.Conj(w))                       # 0.5 * Re(w)

    def _avg_sz(E, zlo, zhi):
        mask = ng.IfPos(ng.z - zlo, 1.0, 0.0) * ng.IfPos(zhi - ng.z, 1.0, 0.0)
        num = complex(ng.Integrate(_sz(E) * mask, mesh)).real
        den = complex(ng.Integrate(mask, mesh)).real         # cell_area * (zhi - zlo)
        return num / den if abs(den) > 1e-30 else 0.0

    z0s, z1s = geo.z_intervals_nm["superstrate"]
    zlo_s, zhi_s = z0s + 50.0, z1s - 50.0
    if zhi_s <= zlo_s:
        zlo_s, zhi_s = z0s + 0.2 * (z1s - z0s), z0s + 0.8 * (z1s - z0s)
    ref = _avg_sz(Einc, zlo_s, zhi_s)                        # incident-only reference (super); < 0
    if abs(ref) < 1e-30:
        return None, None
    R_flux = 1.0 - _avg_sz(Etot, zlo_s, zhi_s) / ref
    if "substrate" not in geo.z_intervals_nm:
        return float(R_flux), None
    z0b, z1b = geo.z_intervals_nm["substrate"]
    pad = 0.1 * (z1b - z0b)
    zlo_b, zhi_b = z0b + pad, z1b - pad
    if zhi_b <= zlo_b:
        return float(R_flux), None
    T_flux = _avg_sz(Etot, zlo_b, zhi_b) / ref
    return float(R_flux), float(T_flux)


def make_fem_optical_solver(*, order=None, diagnostics=True):
    """An `optical_solver` for run_pipeline backed by the FEM, with a SWEEP-AWARE fast path. At NORMAL
    incidence the HCurl FESpace is wavelength-INDEPENDENT (no oblique Bloch phases), so solve_sweep
    builds it ONCE and reuses it across the whole wavelength sweep -- avoiding the redundant FESpace
    construction the per-call default repeats every wavelength (pass-2 audit perf finding). The eps
    mass, RHS, and factorization still rebuild per wavelength, so R/T are BYTE-IDENTICAL to a per-call
    solve_fem (only the k0-independent space build is amortized; the solve itself is not reusable).
    Oblique/conical incidence falls back to a per-wavelength build (the Periodic Bloch phases depend
    on k0). OPT-IN -- pass optical_solver=make_fem_optical_solver() to run_pipeline; the default FEM
    path (pipeline._fem_optical_solver) is unchanged. `order` overrides design.mesh_3d.fem_order;
    `diagnostics=False` threads the solve_fem diagnostics opt-out (A_independent/per-region/flux R/T
    skipped on every solve of the sweep -- see solve_fem's caveat about the energy-closure warning)."""
    from dynameta.optics.eps_assembler import assemble_eps_cf

    def _ord(design):
        return int(order) if order is not None else int(design.mesh_3d.fem_order)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult:
        eps_cf = assemble_eps_cf(geo, eps_by_region)
        return solve_fem(geo, lambda_m, eps_cf, design.optical, order=_ord(design),
                         n_super=n_super, n_sub=n_sub, diagnostics=diagnostics)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        od = _ord(design)
        normal = abs(float(getattr(design.optical, "incidence_angle_deg", 0.0) or 0.0)) <= 1e-9
        # build the wavelength-independent FESpace ONCE for normal incidence (identical to what
        # solve_fem builds internally there); None -> oblique falls back to per-wavelength build.
        fes = ng.Periodic(ng.HCurl(geo.mesh, order=od, complex=True, dirichlet="")) if normal else None
        out = []
        for lam in lams:
            eps_cf = assemble_eps_cf(geo, assemble_at(lam))
            out.append(solve_fem(geo, lam, eps_cf, design.optical, order=od,
                                 n_super=n_super, n_sub=n_sub, diagnostics=diagnostics,
                                 _reuse_fes=fes))
        return out

    _solve.solve_sweep = _solve_sweep
    return _solve
