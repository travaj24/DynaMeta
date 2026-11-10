"""
Solver-configuration specs carried by a Design: Stage-1 DEVSIM mesh, Stage-3
NGSolve mesh, and the optical/incidence configuration. Field names are generic
(no 'mirror'/'patch' coupling); the builders key mesh sizing off material role
+ stack position, not off layer names.

POLARIZATION VOCABULARY (audit V-8): ``OpticalSpec.polarization`` is the LAB-AXIS family {'x', 'y',
'p'} -- 'y' = s-pol, 'p' = p-pol, 'x' = E along lab x (transverse only at normal incidence). It is
ONE of five spellings in the repo; the siblings are the plane-of-incidence {'s','p'} (tmm_reference
/ resonance / nonlocal_tmm / shg_fem / the oblique 2-D FDTD), {'te','tm'} (the lumenairy grating
bridge), the integer ``row`` 0/1 of the differentiable Berreman/RCWA/PMM forwards, and ``pol_axis``
(hydro_fem's 2-D in-plane cross-section). The map, the ``normalize_pol`` converter, and the
normal-incidence / azimuth caveats live in ``dynameta.core.polarization``. The set ACCEPTED here is
unchanged -- acceptance unification (b), the V-8 follow-on, widened only the two PLANE-OF-INCIDENCE
families ({'s','p'} and {'te','tm'}), whose aliases name the same physical mode in every geometry
they cover; this vocabulary's crossings depend on the azimuth (or have no image at all), so they
stay STRICT and are made explicitly, through normalize_pol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class Mesh2DSpec:
    """Stage-1 DEVSIM (x, z) mesh refinement, all lengths in metres."""
    x_spacing_edge_m:          float = 30e-9    # near cell edges (peripheral grounds)
    x_spacing_feature_edge_m:  float = 1e-9     # near inclusion lateral edges
    x_spacing_feature_mid_m:   float = 2.5e-9   # inside an inclusion footprint
    coarse_ps_m:               float = 5e-10    # 0.5 nm layer bulk
    interface_ps_m:            float = 5e-11    # 0.05 nm at semiconductor interfaces
    interface_zone_m:          float = 8e-10    # 0.8 nm refinement zone each side
    # DC solve method: "newton" (coupled, default) | "gummel" (decoupled outer
    # iteration -- robust for stiff degenerate gated accumulation that diverges
    # under coupled Newton). Extensible in carriers/dc_solve.py.
    dc_method:                 str   = "newton"


@dataclass
class Mesh3DSpec:
    """Stage-3 NGSolve OCC mesh. Superstrate/substrate MATERIALS come from the
    Stack; these are buffer thicknesses + element sizes only."""
    # z-PML thickness (0.12.0: 400 -> 1000 nm). The fixed-alpha (1+1j stretch) HalfSpace PML ends on a
    # natural (reflecting) face, so what crosses it comes back attenuated by ~exp(-2 k n cos(theta) d).
    # Measured vs exact TMM on a flat 100 nm metal mirror (R=0.977, maxh_pml 200 nm, 2026-11-02):
    # |dR| at 400 nm = 3.0e-2 (1300 nm) / 5.2e-2 (1550 nm) / 6.5e-3 (1060 nm); at 1000 nm <= 4.9e-4
    # at all three. Keep d >~ 0.65 lambda (2kd >~ 8) for longer wavelengths; never thin it for speed
    # (the PML is meshed at maxh_pml_m, so the extra thickness costs few elements).
    pml_thk_m:               float = 1000e-9
    superstrate_buffer_m:    float = 300e-9     # buffer above stack before top PML
    substrate_buffer_m:      float = 100e-9     # buffer below stack before bottom PML
    maxh_superstrate_m:      float = 120e-9
    maxh_substrate_m:        float = 80e-9
    maxh_pml_m:              float = 200e-9
    maxh_metal_m:            float = 30e-9
    # Thick-metal-film skin/bulk split (applied to the first/last metal film):
    metal_skin_thk_m:        float = 0.0         # 0 disables the split
    maxh_metal_skin_m:       Optional[float] = None
    maxh_metal_bulk_m:       float = 35e-9
    # maxh_inclusion_m sizes the INCLUSION solids only (`<layer>__incl<j>`, i.e. a Layer that
    # actually carries an Inclusion). maxh_background_m is the catch-all for everything else: plain
    # full-cell bands, an inclusion layer's background remainder, AND BOTH HALVES OF A CAVITY SPLIT
    # (`_inpatch` + `_outside`) -- a cavity layer follows the BACKGROUND knob, not this one, since
    # the in-patch column is full-layer-thickness background material (see
    # ngsolve_layered.LayeredOpticalBuilder._maxh; putting it on maxh_inclusion_m's 5 nm default
    # cost 5.1M elements / 493 s of meshing on a 250 nm air cap).
    maxh_inclusion_m:        float = 5e-9
    maxh_background_m:       float = 30e-9
    fem_order:               int = 2
    # Optional prismatic boundary layers grown into semiconductor interfaces:
    semi_prism_thk_m:        Optional[List[float]] = None
    # Mirror-symmetry domain reduction (NORMAL incidence only): solve a half ('half_x'/'half_y')
    # or 'quarter' lateral cell with symmetry walls (PEC/PMC) replacing the periodic boundary on
    # the reduced axis, for a mesh-symmetric (centered-inclusion) cell. 'none' (default) builds the
    # full periodic cell -- byte-identical. The reduction is correctness-neutral (same 0-order R/T as
    # the full solve) but uses 1/2 or 1/4 the DOFs; the solver applies the wall types per polarization.
    symmetry:                Literal["none", "half_x", "half_y", "quarter"] = "none"


@dataclass
class OpticalSpec:
    """Incidence + polarization + linear solver + carrier-field lift control."""
    polarization:        Literal["x", "y", "p"] = "x"  # 'y'=s-pol; 'p'=p-pol (E in x-z plane)
    incidence_angle_deg: float = 0.0
    azimuth_deg:         float = 0.0        # in-plane azimuth phi (0 = x-z plane of incidence;
                                             # !=0 = CONICAL incidence, s-pol only)
    incidence_side:      Literal["top", "bottom"] = "top"   # which semi-infinite medium
    # Carrier-field lift (2D DEVSIM -> 3D eps): "auto" picks SeparableXYLift for a
    # square centered c4v device, else ExtrudeLift; explicit values force a choice
    # and are validated against the device symmetry.
    lift:                Literal["auto", "separable_xy", "extrude", "identity"] = "auto"
    ny_sym:              int = 256
    # "ams"/"hypre" request a HYPRE auxiliary-space Maxwell (AMS) preconditioner (large-3D scaling,
    # the rung above BDDC); they require an NGSolve built with HYPRE and otherwise fall back to
    # bddc_gmres (the standard pip wheel lacks HYPRE) -- see docs/installing_hypre_windows.md.
    # "sparsecholesky" is NGSolve's direct LDL^T: 2-3x faster than umfpack for solve_fem's
    # complex-symmetric normal-incidence matrix, refused by residual when the matrix is not symmetric.
    # "pardiso" is MKL-PARDISO (`pip install mkl`; falls back to umfpack without it): any matrix,
    # 1.2-1.6x over umfpack at ~40% of its memory.
    # "auto" (the default since 0.12.0) picks per solve from the FREE-DOF count and the matrix symmetry:
    # a direct solve up to auto_direct_max_dofs (sparsecholesky when symmetric, else pardiso, else
    # umfpack up to 40k DOFs), bddc_gmres above. Measured 2026-11-02 (order-2 HCurl, 6 threads): direct
    # wins below ~80-100k DOFs (63k: 5.6 s vs BDDC 8-11 s), BDDC above (136k: 9.3 s vs 25 s); peak
    # memory per free DOF ~27 kB sparsecholesky / ~34 kB pardiso / ~81 kB umfpack / ~5 kB BDDC.
    linear_solver:       Literal["auto", "umfpack", "sparsecholesky", "pardiso", "bddc_cg", "bddc_gmres",
                                 "ams", "hypre"] = "auto"
    auto_direct_max_dofs: int = 80_000
    # RELATIVE residual target of the Krylov paths (0.12.0: it used to reach ngsolve.solvers.GMRes as
    # `tol=`, which that function treats as an ABSOLUTE tolerance; now passed as `reltol=`). 1e-8 gives
    # |dr| ~3e-9 against the direct solve on a 204k-DOF metal-pillar cell, 1.6x faster than the old setting.
    gmres_rtol:          float = 1e-8
    gmres_max_iter:      int = 800
    # NGSolve worker threads for the FEM solves (None: env DYNAMETA_FEM_THREADS, else NGSolve's default
    # = every core). Process-global (ngsolve.SetNumThreads). Pin it when running several solves in
    # parallel processes: umfpack does not thread at all (1 thread 10.4 s vs 6 threads 11.5 s at 68k DOFs).
    num_threads:         Optional[int] = None

    def __post_init__(self) -> None:
        if self.polarization not in ("x", "y", "p"):
            # audit V-8: same accepted set, but the rejection now comes from the shared
            # vocabulary home and NAMES the sibling family a wrong label came from ('s'/'p' is
            # the plane-of-incidence spelling, 'te'/'tm' the grating bridge's).  LAZY import,
            # failure path only, so this dataclass module keeps its stdlib-only import cost.
            from dynameta.core.polarization import pol_vocabulary_error
            raise pol_vocabulary_error(
                self.polarization, "lab_xyp", where="OpticalSpec", param="polarization",
                extra="'y' is s-pol (E perpendicular to the x-z plane of incidence) and 'p' is "
                      "p-pol; 'x' (E along lab x) is accepted at NORMAL incidence only.")
        # Oblique incidence (Bloch-Floquet) is implemented for s-pol ('y', E along y
        # perpendicular to the x-z plane) and p-pol ('p', E in the x-z plane). 'x'
        # (E along x) is not transverse at oblique incidence.
        if abs(self.incidence_angle_deg) > 1e-6 and self.polarization == "x":
            raise NotImplementedError(
                "oblique incidence requires polarization='y' (s-pol) or 'p' (p-pol); "
                "'x' (E along x) is not transverse to an oblique x-z-plane wavevector.")
        # Polar-angle cap: a SOLVER-AGNOSTIC sanity bound (the spec feeds the FEM, TMM, RCWA and
        # FDTD backends alike), not a validation claim. AUDIT R-28 (2026-11-02): the old comment
        # justified 60 deg with the FEM PML's ~1%-through-30-deg figure, which covers none of
        # 30-60. The per-backend envelopes are enforced where they live: the FEM advises at any
        # oblique angle (~1% to 30 deg, ~3% at 45, measured vs tmm) and RAISES above 50 deg
        # (optics.solver._FEM_THETA_MAX_DEG); the layered TMM is exact at any real angle. The cap
        # only rejects near-grazing input no backend here is validated for (audit conical-verifier).
        if abs(self.incidence_angle_deg) > 60.0:
            raise NotImplementedError(
                "incidence_angle_deg={:.1f} exceeds the supported range (|theta| <= 60 deg, a "
                "near-grazing bound shared by every backend); the FEM solver itself is "
                "validated to ~1% through 30 deg (~3% at 45) and refuses above 50 deg."
                .format(self.incidence_angle_deg))
        # Conical incidence (azimuth != 0) supports s-pol ('y', the in-plane Es rotates with phi)
        # AND p-pol ('p', the in-plane p-pol component splits over (cos phi, sin phi)); only 'x'
        # (E along x) is not a transverse polarization for an oblique/conical wavevector.
        if abs(self.azimuth_deg) > 1e-6 and self.polarization == "x":
            raise NotImplementedError(
                "conical incidence (azimuth != 0) supports s-pol (polarization='y') or p-pol "
                "(polarization='p'); 'x' (E along x) is not transverse to the wavevector.")
        if self.incidence_side not in ("top", "bottom"):
            raise ValueError("incidence_side must be 'top' or 'bottom'")
        if self.ny_sym < 4:
            raise ValueError("ny_sym must be >= 4")
        if self.num_threads is not None and int(self.num_threads) < 1:
            raise ValueError("num_threads must be >= 1 or None, got {!r}".format(self.num_threads))
        if int(self.auto_direct_max_dofs) < 0:
            raise ValueError("auto_direct_max_dofs must be >= 0, got {!r}".format(self.auto_direct_max_dofs))
        if not self.gmres_rtol > 0.0:
            raise ValueError("gmres_rtol must be > 0, got {!r}".format(self.gmres_rtol))


@dataclass
class CarrierSpec:
    """Carrier-solver settings carried by a Design (v0.15, consolidated plan 4.2.6, user decision 37).
    The builders (LayeredDevsimBuilder, Stacked3DSpec.from_design) and therefore run_pipeline read it;
    nothing here changes the optical side or the optical cache key (cache._design_fingerprint never
    hashes it -- extended precision reaches the optics only through the solved eps_by_region).

    extended_precision: Optional[bool] = None  (None = not stated = OFF; byte-identical to v0.14)
      True builds the carrier device with DEVSIM's 128-bit model evaluation + equation assembly and solves
      it with the mixed-precision extended Newton (carriers/extended_precision.py). WHEN TO USE IT:
        * currents far below the SG round-off floor -- wide-gap (GaN-like) generation / recombination /
          leakage currents, where a double solve returns round-off (measured on a GaN-like junction:
          1e5-1e9x off, wrong sign; extended matches the Sah-Noyce-Shockley oracle to < 1e-3), and
        * heavily doped junctions that do not converge at DynaMeta's DC tolerance in double (1e25/1e22
          p+n: converges at abs 1e8 only in extended).
      NOT for: well-conditioned devices (the degenerate ITO reference metasurface agrees to 4e-16 in
      either precision -- extended buys nothing), and NOT for ssac / noise: DEVSIM factorizes the
      small-signal system in DOUBLE even in extended mode, so its floor stays (an AdvisoryWarning says
      so; use the DC dI/dV for the f -> 0 conductance and drive the lightly doped contact).
      COST (measured, docs/audit/2026-11-03-devsim-extended-precision.md): 6.3x per Newton iteration in
      1-D, 18.5-21.8x on the 76k-node 2-D reference metasurface, identical iteration counts. With a
      v0.15 custom term (carriers.custom_terms) the extended Newton converges LINEARLY (MEASURED: ~12
      iterations to 7e-30 for a weak sink, 34 to 4e-31 for a link as strong as its row; budget
      maximum_iterations >= 60; plan 5.9, decision 8: accepted), and where the term DOMINATES its rows
      it stalls at the double floor (deep-depletion oxide tunnelling: rel ~7e-16 for 120 iterations) --
      see carriers.custom_terms, item 5.

    PRECEDENCE: an explicit builder argument (LayeredDevsimBuilder(extended_precision=...),
    Stacked3DSpec.from_design(extended_precision=...)) wins. When it CONTRADICTS a value stated here
    (True vs False, either way) the builder emits ONE AdvisoryWarning naming both; an unstated design
    (None) never warns -- the v0.14 call LayeredDevsimBuilder(design, extended_precision=True) is
    unchanged. carriers.extended_precision.resolve_design_flag is the one resolver.

    PROCESS-GLOBAL DEVSIM STATE (why this is per DESIGN, not per material or per mesh). The opt-in itself
    is device-scoped (DEVICE-level extended_model / extended_equation, cleared with the device), but
    DEVSIM's extended_solver is GLOBAL and DEVSIM solves EVERY live device in ONE Newton. DynaMeta sets it
    only for the duration of each solve, iff a live device opted in. Consequences:
      * two designs that DISAGREE, built side by side in one process (two live builders): the default
        device is solved by the extended Newton too (its answer moves off the double one). Every such
        solve raises an AdvisoryWarning naming both devices. Tear one builder down first -- NOTE that
        LayeredDevsimBuilder.teardown() deletes EVERY DEVSIM device and mesh in the process, not only its
        own (gated: tests/test_extended_precision_design.py).
      * designs run ONE AFTER ANOTHER (run_pipeline tears its builder down after the carrier sweep) are
        independent: a default run after an extended one is bitwise equal to a fresh default run (gated).
      * a per-material flag was REJECTED: regions of one device share one Newton, and mixed-precision
        regions in one Newton have no gate. On the mesh spec (Mesh2DSpec) was REJECTED: the 3-D builder
        would read a "2-D mesh" setting.
    PROVENANCE: a builder with the setting on adds CarrierField.extras["extended_precision"] = True; off
    adds no key (byte-identical CarrierField)."""
    extended_precision: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.extended_precision is not None and not isinstance(self.extended_precision, bool):
            raise TypeError("CarrierSpec.extended_precision must be True, False or None (unstated = off), "
                            "got {!r}".format(self.extended_precision))
