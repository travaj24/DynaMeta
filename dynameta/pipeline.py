"""
Top-level orchestration on the new bridge architecture:

  carrier solver (DEVSIM) --solve each bias--> CarrierField
        |                                           |
        |                                  core.bridge.assemble_eps
        v                                           v
  optical geometry builder (NGSolve) <--GeometryAlignment--  EpsField per region
        |                                           |
        +-------------- optical solver -------------+ --> OpticalResult per (bias, lambda)

The pipeline is written against the core Protocols, so a caller can swap in a
bring-your-own CarrierSolver / OpticalGeometryBuilder / OpticalSolver. The
defaults are the layered DEVSIM/NGSolve builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from dynameta.core.bridge import assemble_eps
from dynameta.core.lift import choose_lift
from dynameta.core.n_to_eps import MaterialEpsMap, NToEpsMap
from dynameta.core.interfaces import OpticalResult, CarrierSolver, OpticalGeometryBuilder
from dynameta.geometry.design import Design
from dynameta.sweep import Sweep
from dynameta.optics.tmm_reference import end_media_indices   # pure (no ngsolve); shared n_super/n_sub
# NOTE: the FEM trio (LayeredOpticalBuilder / assemble_eps_cf / solve_fem) and the default
# DEVSIM carrier builder are imported LAZILY inside the functions below, so a pure
# layered/TMM path (run_pipeline with carrier_solver + optical_builder + optical_solver all
# supplied) can `import dynameta.pipeline` and run without ngsolve/devsim installed (BLP-2).


@dataclass
class SweepRow:
    bias_label:   str
    lambda_nm:    float
    result:       OpticalResult


def _fem_optical_solver(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult:
    """Default optical solve: assemble the NGSolve VoxelCoefficient + run the FEM."""
    from dynameta.optics.eps_assembler import assemble_eps_cf
    from dynameta.optics.solver import solve_fem
    eps_cf = assemble_eps_cf(geo, eps_by_region)
    return solve_fem(geo, lambda_m, eps_cf, design.optical,
                      order=design.mesh_3d.fem_order, n_super=n_super, n_sub=n_sub)


def run_pipeline(design: Design, sweep: Sweep, *,
                   verbose: bool = True,
                   carrier_solver: Optional[CarrierSolver] = None,
                   optical_builder: Optional[OpticalGeometryBuilder] = None,
                   optical_solver: Optional[Callable[..., OpticalResult]] = None,
                   n_to_eps: Optional[NToEpsMap] = None,
                   extra_fields: Optional[object] = None) -> List[SweepRow]:
    """Run the full Design + Sweep through carriers -> bridge -> optics.

    carrier_solver / optical_builder may be supplied to override the defaults
    (bring-your-own); each must satisfy the corresponding core Protocol.

    optical_solver: an optional callable
    ``fn(design, geo, eps_by_region, lambda_m, n_super, n_sub) -> OpticalResult`` that replaces
    the default per-(bias, wavelength) FEM solve (the default is the NGSolve FEM,
    `_fem_optical_solver`). For a laterally-uniform/graded stack, pass the ready-made
    ``dynameta.optics.tmm_reference.make_layered_tmm_solver()`` to solve with the layered TMM
    instead (a future RCWA backend plugs in the same way). NOTE: run_pipeline still calls
    ``optical_builder.build()`` (the FEM mesh) unless you ALSO supply an ``optical_builder``
    that exposes only ``alignment()`` / ``mesh_regions()`` -- pass such a stub to skip the
    mesh build and the ngsolve dependency entirely.

    n_to_eps: the per-region RESPONSE map (NToEpsMap). Defaults to ``MaterialEpsMap(design.materials)``
    (the carrier/Drude path reading 'n'). Pass an ``EffectEpsMap(design.materials, effects={...})`` to
    drive the field/temperature/state EFFECT MODELS (Pockels / Kerr / Franz-Keldysh / thermo-optic /
    QCSE / PCM / LC / magneto-optic) through the SAME bridge -- this is how the modulation-mechanism
    family is reached from the orchestrator (otherwise it is reachable only by a hand-rolled
    assemble_eps loop).

    extra_fields: the field bundle the effect models read alongside 'n' -- {'E': ..., 'T': ...,
    'director_angle_rad': ..., 'crystalline_fraction': ..., 'magnetization': ...}. Either a static
    dict (same for every bias) or a CALLABLE ``fn(bias_point) -> dict`` (the usual case: the applied
    field/temperature changes with bias -- e.g. run ``carriers.electrostatics_fem.solve_electrostatics_fem``
    / ``thermal_fem.solve_thermal_fem`` per bias and thread the resulting E / T here). CONTRACT: a
    callable must return a dict with the SAME key set at every bias (only the VALUES vary) -- a key
    that appears or disappears mid-sweep would silently switch which effect terms are driven (a
    RuntimeWarning fires on drift). None (the default) leaves the carrier-only path byte-identical.
    """
    if not sweep.bias_points or not len(sweep.wavelengths_nm):     # no silent empty-sweep passthrough
        raise ValueError("run_pipeline: sweep must have at least one bias point and one wavelength "
                         "(got {} bias points, {} wavelengths)".format(
                             len(sweep.bias_points), len(sweep.wavelengths_nm)))
    if carrier_solver is None:
        from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
        carrier_solver = LayeredDevsimBuilder(design)
    carrier = carrier_solver
    # 1) carriers: solve every bias, collect CarrierFields, then free DEVSIM
    fields = {}
    for bp in sweep.bias_points:
        if verbose:
            print("[carriers] bias '{}' ...".format(bp.label), flush=True)
        fields[bp.label] = carrier.solve(bp)
    if hasattr(carrier, "teardown"):
        carrier.teardown()

    # 2) optical geometry: build mesh once + the alignment contract
    if optical_builder is None:
        from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
        optical_builder = LayeredOpticalBuilder(design)
    optical = optical_builder
    geo = optical.build()
    align = optical.alignment()
    mesh_regions = optical.mesh_regions()
    align.validate_coverage(mesh_regions)
    if verbose:
        print("[optics] mesh ne={} nv={}; {} spatial + {} fixed regions".format(
            geo.mesh.ne, geo.mesh.nv, len(align.region_alignments),
            len(align.fixed_eps_regions)), flush=True)

    solve_optics = optical_solver or _fem_optical_solver
    # a SWEEP-AWARE optical solver (e.g. make_fdtd_sweep_optical_solver) exposes solve_sweep: ONE broadband
    # solve per bias serves the whole wavelength list, instead of re-solving each wavelength.
    sweep_aware = hasattr(solve_optics, "solve_sweep")
    # The sweep-aware fast path passes ONE (n_super, n_sub) for the whole band (computed at band-centre
    # below). That is EXACT only when the end media are NON-dispersive; if n_super/n_sub vary with
    # wavelength a band-centre freeze gives wrong R/T off-centre AND disagrees with the per-wavelength
    # path (which re-derives end media per lambda) -- a silent inconsistency for a dispersive cladding/
    # substrate over a broadband sweep. Detect dispersion at the band edges and DISABLE the fast path
    # when found, so optics are solved per-wavelength (end media track lambda; exact -- only the
    # one-broadband-solve-per-bias speedup is lost). Non-dispersive end media (air super / fixed-index
    # substrate -- the common case) keep the fast path, byte-identical.
    if sweep_aware and len(sweep.wavelengths_nm) > 1:
        # audit C5-12: sample end media at EVERY sweep wavelength PLUS the band centre (the
        # value the fast path actually freezes) -- the old two-edge check false-passed a
        # dispersive medium with equal band-edge values and an in-band feature (probe: a
        # tabulated substrate with eps 4->6->4 froze n_sub=2.449 while the true edge
        # values are 2.0).
        _lams_m = [float(w) * 1e-9 for w in sweep.wavelengths_nm]
        _lams_m.append(0.5 * (min(_lams_m) + max(_lams_m)))   # lam_c, the freeze point
        _pairs = [end_media_indices(design, lm) for lm in _lams_m]
        _ns_ref, _nb_ref = _pairs[-1]                         # the frozen band-centre values
        if any(abs(ns - _ns_ref) > 1e-12 * max(abs(_ns_ref), 1.0)
               or abs(nb - _nb_ref) > 1e-12 * max(abs(_nb_ref), 1.0)
               for ns, nb in _pairs[:-1]):
            import warnings
            warnings.warn(
                "run_pipeline: end media (n_super/n_sub) are DISPERSIVE across the sweep band, so the "
                "sweep-aware fast path (which freezes them at band-centre) is DISABLED -- optics are "
                "solved per-wavelength instead so n_super/n_sub track lambda. The per-wavelength result "
                "is exact; only the one-broadband-solve-per-bias speedup is lost.", RuntimeWarning,
                stacklevel=2)
            sweep_aware = False
    if n_to_eps is None:
        n_to_eps = MaterialEpsMap(design.materials)
    lift = choose_lift(design.device_symmetry(), design.optical.lift,
                        period_y_m=design.unit_cell.period_y_m, ny=design.optical.ny_sym)

    # 3) bridge + solve per (bias, wavelength)
    rows: List[SweepRow] = []

    def _emit(label, lam_nm, res):
        rows.append(SweepRow(label, float(lam_nm), res))
        if verbose:
            tstr = ("T={:.4f} A={:+.4f}".format(res.T, res.A) if res.T is not None else "T=n/a")
            print("[optics]   {} lam={:.0f}nm  R={:.4f}  {}  phase={:+.1f}  ({:.1f}s)".format(
                label, lam_nm, res.R, tstr, res.phase_deg, res.solve_time_s), flush=True)

    ef_keys0 = None                            # callable extra_fields key-set stability (see docstring)
    for bp in sweep.bias_points:
        # pop, not read: this loop is the sole consumer, so each bias's grids are freed once its
        # optics finish -- peak memory no longer stacks every CarrierField under the optics solve
        # (duplicate labels can't KeyError: Sweep rejects them at construction)
        cf = fields.pop(bp.label)
        # the field-effect bundle for THIS bias (a callable resolves the per-bias E/T/state; a plain
        # dict is reused; None keeps the carrier-only path byte-identical)
        ef = extra_fields(bp) if callable(extra_fields) else extra_fields
        if callable(extra_fields) and ef is not None:
            keys = frozenset(ef.keys())
            if ef_keys0 is None:
                ef_keys0 = keys
            elif keys != ef_keys0:
                import warnings
                warnings.warn("run_pipeline: extra_fields callable changed its key set at bias '{}' "
                              "({} vs {}) -- the driven effect terms silently differ across the sweep."
                              .format(bp.label, sorted(keys), sorted(ef_keys0)), RuntimeWarning,
                              stacklevel=2)
        if sweep_aware and len(sweep.wavelengths_nm) > 1:
            # FAST PATH: ONE broadband solve for THIS bias serves the whole wavelength list. Hand the solver
            # an eps-assembler closure (so it can sample the bias's per-layer dispersion across the band).
            lams = [float(w) * 1e-9 for w in sweep.wavelengths_nm]
            lam_c = 0.5 * (min(lams) + max(lams))
            n_super, n_sub = end_media_indices(design, lam_c)

            def _assemble_at(lm, _cf=cf, _ef=ef):
                return assemble_eps(_cf, align, n_to_eps, lift, lm,
                                    mesh_regions=mesh_regions, extra_fields=_ef)
            results = list(solve_optics.solve_sweep(design, geo, _assemble_at, lams, n_super, n_sub))
            if len(results) != len(sweep.wavelengths_nm):       # zip would TRUNCATE silently otherwise
                raise ValueError("optical_solver.solve_sweep returned {} results for {} wavelengths -- "
                                 "a sweep-aware solver must return exactly one OpticalResult per "
                                 "requested wavelength, in order.".format(
                                     len(results), len(sweep.wavelengths_nm)))
            for lam_nm, res in zip(sweep.wavelengths_nm, results):
                _emit(bp.label, lam_nm, res)
            continue
        for lam_nm in sweep.wavelengths_nm:
            lambda_m = float(lam_nm) * 1e-9
            n_super, n_sub = end_media_indices(design, lambda_m)
            eps_by_region = assemble_eps(cf, align, n_to_eps, lift, lambda_m,
                                          mesh_regions=mesh_regions, extra_fields=ef)
            res = solve_optics(design, geo, eps_by_region, lambda_m, n_super, n_sub)
            _emit(bp.label, lam_nm, res)
    return rows
