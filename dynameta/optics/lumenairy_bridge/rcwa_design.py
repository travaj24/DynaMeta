"""Differentiable RCWA + PMM forwards for layered/periodic inverse design (audit 8.1-5).

berreman_design.py gives DynaMeta an exact-gradient forward for the PLANAR-anisotropic class;
this module extends the semi-analytic-gradient surface to LAYERED/PERIODIC structures (binary
gratings, patterned metasurface cells, carrier-modulated multilayers) by wrapping Lumenairy's
JAX-differentiable RCWA and PMM twins -- far cheaper per iterate than the JAX-FDTD
topology-opt path (optics.inverse_design.optimize_fdtd), which pays a full space-time march
per gradient. REQUIRES jax double precision (jax.config.update("jax_enable_x64", True)):
lumenairy's traced solves refuse/warn on f32 (the RCWA eigenproblem is ill-conditioned there).

WHAT LUMENAIRY ACTUALLY TRACES (pinned from the 5.22 source + lumenairy's own tests; each
entry point below scopes itself honestly and raises for the rest):

- rcwa_efficiency_1d (binary grating, functional): region indices (re AND im), BOTH half-space
  indices, depth, angle, wavelength. STATIC: period, duty_cycle, n_orders (float()-ed / order
  set). Wrapped by rcwa_grating_RT, which keeps DynaMeta's EPS convention (the materials
  machinery hands out eps; the lumenairy scalar entry natively takes INDICES n, so the wrapper
  lifts eps -> n via the principal sqrt -- differentiable, Im(eps) > 0 maps to Im(n) > 0).
- RCWAStack.solve (multilayer, 1-D or 2-D patterned): TRACED are the eps_cell /
  eps_tensor_cell VALUES, the UNIFORM eps= scalars (lumenairy 5.22 keeps a traced uniform
  permittivity RAW -- layer kind 'uniform' -- so its gradient flows through the analytic
  homogeneous modes, no lifted constant-cell eigensolve), the layer THICKNESSES, and (5.22)
  the source WAVELENGTH / THETA / PHI (set_source keeps them raw). STATIC: the half-space
  indices (complex()-ed in solve -- still sever the gradient), the periods (float()-ed in the
  constructor), the order counts, and the patterned-cell WALLS. Wrapped by rcwa_stack_RT /
  rcwa_stack_jones: a jax-typed static-only argument (half-space index or period) raises
  TypeError instead of silently concretizing. Upstream pins:
  tests/unit/test_v5_10_3_rcwa_2d_autodiff.py (eps-cell + depth AD == FD, vmap forward,
  vmap-of-grad, Hessian).
- PMMStack.solve (1-D lamellar spectral element -- no Fourier-factorization accuracy floor):
  traced segment eps (scalar or in-plane (3, 3), re AND im), layer thicknesses, angle,
  half-space indices. STATIC: period, segment WIDTHS (frozen union grid), degree, order
  count, and -- lumenairy >= 5.30 (W7 F-E) -- the WAVELENGTH: PMM sizes its propagating-order
  set FROM the wavelength (m_prop = floor(n_max*period/wavelength)), which cannot be read
  from a tracer, so the differentiable path raises NotImplementedError on a traced wavelength
  instead of silently solving a smaller order set (lumenairy measured 4.2% forward / 20.7%
  wrong d/d(wavelength) under jit). Wavelength sensitivity: finite differences over concrete
  wavelengths (gate-pinned twin==bridge) or PMMStack.solve_vs_wavelength on the numpy side;
  the RCWAStack twin still traces the wavelength (its order count is the explicit static
  n_orders). Slant / out-of-plane tensors / stabilize / retain_internal raise upstream.
  Upstream pin: tests/unit/test_v5_14_2_jax_stacks.py. The PMM twin is SHIPPED (not skipped):
  it is genuinely usable for 1-D lamellar gratings with gradients, and it traces the angle
  and half-space parameters the RCWAStack twin cannot -- the two are complementary. ANGLE
  CAVEAT (measured on 5.30.0, fix-verification 2026-07-27): the angle gradient is correct
  OFF-normal (AD==FD to ~6e-7 at theta 0.15-0.4 rad) but INVALID at exactly theta = 0 -- the
  +/-m order degeneracy breaks lumenairy's eig VJP there (AD dR/dtheta = 4.1e-3 on a
  mirror-symmetric lossless fixture where symmetry AND energy conservation both force 0;
  clean by theta ~ 1e-4). Differentiate the angle away from exact normal incidence.

Convention: identical on both sides (public exp(-i omega t), Im(eps) > 0 lossy, metres,
radians). Efficiency rows are keyed INCIDENT lab E_x (row 0) / E_y (row 1) -- never relabel
as TE/TM; the zeroth-order Jones (rcwa_stack_jones) carries the phase observable r.

VERSION NOTE on jit/vmap: unlike the Berreman twin (see berreman_design.py -- its eig-VJP
pytree fix landed only post-tagged-5.14.4), the RCWA/PMM twins' gauge-stable custom-VJP eig
already returns a plain (eigvals, eigvecs) tuple on the whole bridge floor (>= 5.22;
lumenairy rcwa/_core.py _jax_eig_stable), so grad-of-vmap / Hessian compose without a version
condition (pinned upstream). Eager jax.grad is the gate-validated bridge path
(validation/lumenairy_rcwa_jax.py: parity vs the non-JAX bridge, AD vs FD of the non-JAX
bridge, descent sanity).

Carrier modulation: drude_eps_jax lifts an existing materials-machinery DrudeOptical into a
jax-traceable carrier-density -> eps closure (same constants, same formula, eagerly validated
static parameters), so a carrier-actuated layer chains n_m3 -> eps -> R/T with one jax.grad.

POLARIZATION VOCABULARY (audit V-8): this module speaks TWO of the five -- {'te', 'tm'} for the 1-D
grating entry point (plus lumenairy's own case-insensitive 's'/'p' aliases) and the integer lab
`row` 0/1 (0 = E_x, 1 = E_y) for the stack entry points. It is one of five spellings in the repo --
{'s','p'} is the PLANE-OF-INCIDENCE spelling, {'x','y','p'} OpticalSpec's LAB AXIS, and `pol_axis`
hydro_fem's 2-D in-plane axis. The map, the `normalize_pol` converter and the normal-incidence /
azimuth caveats live in `dynameta.core.polarization`. NEITHER set changed under acceptance
unification (b): the grating entry point ALREADY took the unconditional 's'/'p' aliases (its
accepted set is byte-for-byte lumenairy's own `_normalize_pol`, and the {'s','p'} family is now the
mirror image of it), while the integer `row` is an INDEX, not a label -- `{'x': 0, 'y': 1, 'p': 0}`
is not injective, so there is no lossless alias to admit and a label crossing into it stays
explicit, through normalize_pol.
"""

from __future__ import annotations

import numpy as np

from dynameta.optics.lumenairy_bridge._common import _REQUIRED
from dynameta.optics.lumenairy_bridge._common import \
    require_halfspace_keywords as _require_halfspace_keywords
from dynameta.optics.lumenairy_bridge._common import require_lumenairy as _require_lumenairy

__all__ = ["rcwa_grating_RT", "rcwa_stack_RT", "rcwa_stack_jones", "pmm_stack_RT",
           "pmm_stack_jones", "drude_eps_jax"]

# HALF-SPACE ARGUMENT ORDER (finding V-4): `n_substrate` / `n_superstrate` are KEYWORD-ONLY on
# every public entry point below. These signatures mirror lumenairy's upstream SUB-FIRST order,
# the inverse of the 46 super-first functions elsewhere in DynaMeta (including
# _common.p_basis_conversion in this same package), and both arguments are bare scalars -- so a
# super-first POSITIONAL call type-checked, ran, and solved the stack upside down in silence.
# Naming them removes the ambiguity; a legacy positional call now raises a TypeError that spells
# out the migration. The MEANINGS are unchanged (nothing is silently transposed).


def _is_jaxish(x) -> bool:
    """True for ANY jax value (concrete device array or tracer) WITHOUT importing jax: the
    bridge stays import-light (hygiene contract) and the check must not concretize a trace."""
    mod = type(x).__module__ or ""
    return mod.split(".")[0] in ("jax", "jaxlib")


def _require_jax_x64(fn_name: str, *values) -> None:
    """audit R-14: this module's docstring says it REQUIRES jax double precision (lumenairy's
    traced RCWA/PMM solves refuse or warn on f32 -- the eigenproblem is ill-conditioned there),
    and it was the one jax entry point in the repo that set `jax_enable_x64` NOWHERE and checked
    it nowhere: `rcwa_stack_jones` under a default-configured jax returned a SILENT float32
    result. Five sibling modules each set the flag themselves; the bridge must not, because
    flipping a global on someone else's traced program is worse than refusing -- so it CHECKS,
    only when a jax value actually arrives, and names the one-line fix.

    Import-light: `jax` is touched only once a jax-typed argument has already been seen, so the
    numpy path never imports it."""
    if not any(_is_jaxish(v) for v in values):
        return
    import jax

    from dynameta.core.backend import require_jax_011
    require_jax_011(jax)
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError(
            "{}: jax x64 is OFF, so this differentiable solve would run in float32 and return a "
            "silently wrong result (the RCWA/PMM eigenproblem is ill-conditioned in single "
            "precision -- lumenairy's traced solves refuse or warn on it). Enable it at program "
            "start, BEFORE any jax array is built: jax.config.update('jax_enable_x64', True) "
            "(or JAX_ENABLE_X64=1). The bridge deliberately does not flip this global for you: "
            "it governs every array in your program, not just this call.".format(fn_name))


def _require_static(fn_name: str, **kwargs) -> None:
    """Raise loudly when a STATIC-only argument arrives as a jax value. The lumenairy stack
    surface would complex()/float() it -- a concrete jax scalar would silently LOSE its
    gradient and a tracer would die with an opaque conversion error deep inside lumenairy;
    honest scoping is the bridge's job (the berreman_design precedent). As of lumenairy 5.22
    the ONLY remaining static-only stack-twin arguments are the half-space indices (complex()
    at solve) and the periods (float() in the constructor) -- the source wavelength / theta /
    phi now trace (set_source keeps them raw), so the call site no longer routes them here."""
    for name, val in kwargs.items():
        if _is_jaxish(val):
            raise TypeError(
                "{}: {} must be a concrete python/numpy number -- lumenairy's RCWAStack twin "
                "concretizes the half-space indices (complex() in solve) and the periods "
                "(float() in the constructor), severing their gradient. It DOES trace the "
                "layer permittivity cells, the uniform eps, the thicknesses, and (5.22) the "
                "source wavelength / theta / phi. For gradients w.r.t. the half-space indices "
                "use rcwa_grating_RT (binary grating) or pmm_stack_RT (1-D lamellar stack), "
                "whose lumenairy twins trace them.".format(fn_name, name))


# ---- polarization vocabularies spoken in THIS file (audit V-8) ----------------------------------
# TWO of the repo's five: rcwa_grating_RT takes {'te','tm'} (lumenairy's grating spelling, which
# ALSO accepts 's'/'p' case-insensitively -- lumenairy.rcwa_efficiency_1d normalizes them upstream
# and always has), while rcwa_stack_RT / pmm_stack_RT take the integer lab `row` (0 = E_x,
# 1 = E_y).  Both accepted sets are UNCHANGED; only the rejections moved to the shared home so a
# label from a sibling family is named instead of dying inside lumenairy or inside a fancy-index.
# Map, converter and caveats: dynameta.core.polarization.
def _reject_grating_pol(polarization, where: str):
    """Guard the {'te','tm'} vocabulary INCLUDING lumenairy's own case-insensitive 's'/'p'
    aliases -- the accepted set is byte-for-byte what `lumenairy.rcwa_efficiency_1d._normalize_pol`
    accepts, so nothing that used to work stops working and the ORIGINAL string is still what gets
    passed downstream.  Only the REJECTION changes: a lab-axis 'x'/'y' now names its own family
    instead of producing a lumenairy error that mentions neither.  LAZY import, failure path
    only."""
    if str(polarization).lower() not in ("te", "tm", "s", "p"):
        from dynameta.core.polarization import pol_vocabulary_error
        raise pol_vocabulary_error(polarization, "tetm", where=where, param="polarization")


def _reject_row(row, where: str):
    """Guard the integer lab-`row` vocabulary (audit V-8): 0 = incident E_x, 1 = incident E_y.
    A sibling label ('x'/'y'/'p', 's'/'p', 'te'/'tm') used to reach the numpy fancy-index and die
    opaquely (or, for a bool, silently index row 0/1).  Accepted set unchanged."""
    if isinstance(row, bool) or not isinstance(row, int) or row not in (0, 1):
        from dynameta.core.polarization import pol_vocabulary_error
        raise pol_vocabulary_error(row, "row", where=where, param="row")


def rcwa_grating_RT(period, eps_ridge, eps_groove, depth, duty_cycle, wavelength=_REQUIRED,
                    *_legacy, n_substrate=_REQUIRED, n_superstrate=_REQUIRED,
                    angle=0.0, polarization="te", n_orders=11, formulation="auto"):
    """Differentiable (R_total, T_total) of a 1-D binary grating for ONE linear polarization
    ('te' = s = E along the grooves, 'tm' = p) -- the scalar FOM ingredients for jax.grad.

    `n_substrate` / `n_superstrate` are KEYWORD-ONLY (audit V-4 -- see the module-level note):
    rcwa_grating_RT(period, eps_ridge, eps_groove, depth, duty_cycle, wavelength,
    n_substrate=..., n_superstrate=...).

    Routes to lumenairy's JAX twin when ANY of eps_ridge / eps_groove / n_substrate /
    n_superstrate / depth / angle / wavelength is a jax array (gradients then flow through all
    of them, real and imaginary parts); plain numbers give the concrete NumPy forward. STATIC:
    period, duty_cycle, n_orders, polarization, formulation. Region eps follows DynaMeta's
    materials convention and is lifted to lumenairy's native refractive INDEX by the principal
    sqrt (differentiable; a passive Im(eps) > 0 maps to Im(n) > 0); the half-spaces are
    indices, as everywhere in the bridge."""
    _require_halfspace_keywords("rcwa_grating_RT", _legacy, wavelength=wavelength,
                                n_substrate=n_substrate, n_superstrate=n_superstrate)
    _reject_grating_pol(polarization, "rcwa_grating_RT")                # audit V-8
    _require_jax_x64("rcwa_grating_RT", eps_ridge, eps_groove, n_substrate, n_superstrate,
                     depth, wavelength, angle)                      # audit R-14
    lum = _require_lumenairy()
    orders, R, T = lum.rcwa_efficiency_1d(
        period, eps_ridge ** 0.5, eps_groove ** 0.5, n_substrate, n_superstrate, depth,
        duty_cycle, wavelength, angle=angle, polarization=polarization,
        n_orders=int(n_orders), formulation=formulation)
    return R.sum(), T.sum()


def _add_stack_layers(stack, layers, is_2d: bool, formulation: str, fn_name: str) -> None:
    """Append [(eps_spec, thickness), ...] (superstrate-side first, the berreman_RT layer
    convention) to a lumenairy RCWAStack, dispatching on the spec's shape:

    - scalar        -> uniform layer; passed straight to add_layer(eps=) whether concrete or
                       a traced JAX scalar (lumenairy 5.22 keeps a traced uniform eps RAW --
                       layer kind 'uniform' -- so its gradient flows through the analytic
                       homogeneous modes, no lifted constant-cell eigensolve);
    - (Sx,)/(Sx,Sy) -> patterned eps_cell (VALUES differentiable when jax; walls static);
    - (3, 3)        -> uniform anisotropic tensor, tiled to an eps_tensor_cell;
    - (Sx,Sy,3,3)   -> patterned eps_tensor_cell.

    Thickness passes through raw (float or traced jax scalar -- lumenairy skips the range
    guard on a trace). Patterned specs must meet lumenairy's sampling bound
    Sx >= 4*n_orders_x + 1 (and y alike on 2-D stacks) -- enforced loudly upstream."""
    smx = 4 * int(stack.n_orders_x) + 1
    smy = (4 * int(stack.n_orders_y) + 1) if is_2d else 1
    for eps, thickness in layers:
        nd = int(np.ndim(eps)) if not _is_jaxish(eps) else int(eps.ndim)
        if nd == 0:
            # uniform layer: a traced jax eps passes STRAIGHT to add_layer(eps=) -- lumenairy
            # 5.22 keeps it raw (kind 'uniform') so the gradient flows through the analytic
            # homogeneous modes (no lifted constant-cell eigensolve); a concrete number takes
            # the complex() uniform path.
            if _is_jaxish(eps):
                stack.add_layer(thickness, eps=eps)
            else:
                stack.add_layer(thickness, eps=complex(eps))
        elif nd in (1, 2) and tuple(np.shape(eps)) != (3, 3):
            stack.add_layer(thickness, eps_cell=eps, formulation=formulation)
        elif tuple(np.shape(eps)) == (3, 3):
            if _is_jaxish(eps):
                import jax.numpy as jnp
                tcell = jnp.broadcast_to(jnp.asarray(eps, dtype=jnp.complex128)[None, None],
                                         (smx, smy, 3, 3))
            else:
                tcell = np.broadcast_to(np.asarray(eps, dtype=complex), (smx, smy, 3, 3)).copy()
            stack.add_layer(thickness, eps_tensor_cell=tcell)
        elif nd == 4:
            stack.add_layer(thickness, eps_tensor_cell=eps)
        else:
            raise ValueError(
                "{}: layer eps spec has unsupported shape {} -- expected a scalar, a (Sx,) / "
                "(Sx, Sy) cell, a (3, 3) tensor, or a (Sx, Sy, 3, 3) tensor cell.".format(
                    fn_name, np.shape(eps)))


def _solve_rcwa_stack(layers, wavelength, *, n_substrate, n_superstrate, period_x, period_y,
                      theta, phi, n_orders, n_orders_y, formulation, fn_name):
    _require_jax_x64(fn_name, wavelength, theta, phi,
                     *[e for e, _t in layers], *[t for _e, t in layers])   # audit R-14
    lum = _require_lumenairy()
    _require_static(fn_name, n_substrate=n_substrate, n_superstrate=n_superstrate,
                    period_x=period_x, period_y=period_y)
    if period_y is None:
        stack = lum.RCWAStack(period_x, n_superstrate=complex(n_superstrate),
                              n_substrate=complex(n_substrate), n_orders=int(n_orders))
        is_2d = False
    else:
        stack = lum.RCWAStack(period_x, period_y=period_y,
                              n_superstrate=complex(n_superstrate),
                              n_substrate=complex(n_substrate), n_orders=int(n_orders),
                              n_orders_y=int(n_orders_y if n_orders_y is not None
                                             else n_orders))
        is_2d = True
    _add_stack_layers(stack, layers, is_2d, formulation, fn_name)
    stack.set_source(wavelength, theta=theta, phi=phi)
    return stack.solve()


def rcwa_stack_RT(layers, wavelength=_REQUIRED, *_legacy, n_substrate=_REQUIRED,
                  n_superstrate=_REQUIRED, period_x, period_y=None,
                  theta=0.0, phi=0.0, n_orders=11, n_orders_y=None, formulation="laurent",
                  row=0):
    """Differentiable order-summed (R, T) of a patterned multilayer for ONE incident lab
    polarization (row 0 = E_x, 1 = E_y) -- a scalar FOM ingredient for jax.grad.

    `n_substrate` / `n_superstrate` are KEYWORD-ONLY (audit V-4 -- see the module-level note).

    `layers` = [(eps_spec, thickness), ...] superstrate-side first (the berreman_RT layer
    convention); see _add_stack_layers for the shape dispatch (a uniform scalar, concrete or
    traced, goes straight to add_layer(eps=)). Gradients flow through every jax-typed eps
    VALUE (cells: values only, walls static; a uniform eps traces raw), every jax-typed
    thickness, AND (lumenairy 5.22) the source wavelength / theta / phi. STATIC here: the
    half-space indices and the periods (TypeError on a jax value -- rcwa_grating_RT and
    pmm_stack_RT trace the half-space indices). A 1-D stack (period_y=None) is genuinely
    cheaper and better-conditioned than a y-degenerate 2-D one -- keep lamellar problems
    1-D."""
    _require_halfspace_keywords("rcwa_stack_RT", _legacy, wavelength=wavelength,
                                n_substrate=n_substrate, n_superstrate=n_superstrate)
    _reject_row(row, "rcwa_stack_RT")                                   # audit V-8
    res = _solve_rcwa_stack(layers, wavelength,
                            n_substrate=n_substrate, n_superstrate=n_superstrate,
                            period_x=period_x, period_y=period_y, theta=theta, phi=phi,
                            n_orders=n_orders, n_orders_y=n_orders_y,
                            formulation=formulation, fn_name="rcwa_stack_RT")
    _orders, R, T = res.efficiencies()
    return R[row].sum(), T[row].sum()


def rcwa_stack_jones(layers, wavelength=_REQUIRED, *_legacy, n_substrate=_REQUIRED,
                     n_superstrate=_REQUIRED, period_x,
                     period_y=None, theta=0.0, phi=0.0, n_orders=11, n_orders_y=None,
                     formulation="laurent"):
    """Differentiable full far field of a patterned multilayer: (orders, R_eff, T_eff,
    jones_r, jones_t) with per-order (2, N) efficiencies and the zeroth-order (2, 2) lab-basis
    Jones matrices -- for phase-bearing FOMs (the modulator observable r = jones_r[row, row])
    and per-order targets. Same layer convention, tracing surface, static-argument policy and
    KEYWORD-ONLY half-space indices (audit V-4) as rcwa_stack_RT."""
    _require_halfspace_keywords("rcwa_stack_jones", _legacy, wavelength=wavelength,
                                n_substrate=n_substrate, n_superstrate=n_superstrate)
    res = _solve_rcwa_stack(layers, wavelength,
                            n_substrate=n_substrate, n_superstrate=n_superstrate,
                            period_x=period_x, period_y=period_y, theta=theta, phi=phi,
                            n_orders=n_orders, n_orders_y=n_orders_y,
                            formulation=formulation, fn_name="rcwa_stack_jones")
    orders, R, T = res.efficiencies()
    return orders, R, T, res.jones_reflection(), res.jones_transmission()


def _solve_pmm_stack(layers, wavelength, *, n_substrate, n_superstrate, period, angle,
                     degree, n_orders, fn_name):
    _require_jax_x64(fn_name, wavelength, angle, n_substrate, n_superstrate,
                     *[t for _s, t in layers])                            # audit R-14
    lum = _require_lumenairy()
    if _is_jaxish(period):
        raise TypeError(
            "{}: period must be a concrete number (the PMM union grid / segment walls are "
            "frozen NumPy geometry; only the segment eps VALUES, thicknesses, angle and "
            "half-space indices trace -- the wavelength is static too on lumenairy >= 5.30, "
            "see the docstring).".format(fn_name))
    st = lum.PMMStack(period, n_substrate=n_substrate, n_superstrate=n_superstrate,
                      degree=int(degree), n_orders=int(n_orders))
    for spec, thickness in layers:
        if isinstance(spec, (list, tuple)):
            for w, _e in spec:
                if _is_jaxish(w):
                    raise TypeError(
                        "{}: segment WIDTHS are static (frozen union grid) -- only the "
                        "segment eps values are differentiable. Reparameterize a moving "
                        "wall as an eps interpolation, or use the FDTD topology-opt "
                        "path.".format(fn_name))
            st.add_layer(thickness, segments=[(float(w), e) for w, e in spec])
        else:
            st.add_layer(thickness, eps=spec)
    st.set_source(wavelength, angle=angle)
    return st.solve()


def pmm_stack_RT(layers, wavelength=_REQUIRED, *_legacy, n_substrate=_REQUIRED,
                 n_superstrate=_REQUIRED, period, angle=0.0,
                 degree=12, n_orders=21, row=0):
    """Differentiable order-summed (R, T) of a 1-D lamellar stack via lumenairy's PMM JAX
    twin (spectral element -- no Fourier-factorization accuracy floor) for ONE incident lab
    polarization (row 0 = E_x, 1 = E_y).

    `n_substrate` / `n_superstrate` are KEYWORD-ONLY (audit V-4 -- see the module-level note).

    `layers` = [(spec, thickness), ...] superstrate-side first; spec is either a segment list
    [(width_fraction, eps), ...] (widths STATIC and summing to 1; eps scalar or in-plane
    (3, 3) tensor) or a bare eps for a uniform layer. Gradients flow through every jax-typed
    eps (re AND im), thickness, AND -- unlike the RCWAStack twin -- the angle (OFF-normal
    only: at exactly angle=0 the eig VJP hits the +/-m degeneracy and the angle gradient is
    invalid, see the module docstring) and the half-space indices. STATIC: period, widths,
    degree, n_orders, and (lumenairy >= 5.30) the
    WAVELENGTH -- PMM sizes its propagating-order set from the wavelength, so the
    differentiable path raises NotImplementedError on a traced one rather than silently
    solving a smaller order set (see the module docstring); use concrete-wavelength finite
    differences for wavelength sensitivity. Out-of-plane tensors / slants raise upstream
    (lumenairy's own guards)."""
    _require_halfspace_keywords("pmm_stack_RT", _legacy, wavelength=wavelength,
                                n_substrate=n_substrate, n_superstrate=n_superstrate)
    _reject_row(row, "pmm_stack_RT")                                    # audit V-8
    _orders, R, T, _jones = _solve_pmm_stack(layers, wavelength, n_substrate=n_substrate,
                                             n_superstrate=n_superstrate,
                                             period=period, angle=angle, degree=degree,
                                             n_orders=n_orders, fn_name="pmm_stack_RT")
    return R[row].sum(), T[row].sum()


def pmm_stack_jones(layers, wavelength=_REQUIRED, *_legacy, n_substrate=_REQUIRED,
                    n_superstrate=_REQUIRED, period, angle=0.0,
                    degree=12, n_orders=21):
    """Differentiable full PMM far field: (orders, R_eff, T_eff, jones_r) -- per-order (2, M)
    efficiencies plus the zeroth-order (2, 2) reflection Jones (PMM exposes NO transmission
    Jones; see pmm_backend). Same layer convention, tracing surface and KEYWORD-ONLY half-space
    indices (audit V-4) as pmm_stack_RT."""
    _require_halfspace_keywords("pmm_stack_jones", _legacy, wavelength=wavelength,
                                n_substrate=n_substrate, n_superstrate=n_superstrate)
    return _solve_pmm_stack(layers, wavelength, n_substrate=n_substrate,
                            n_superstrate=n_superstrate, period=period,
                            angle=angle, degree=degree, n_orders=n_orders,
                            fn_name="pmm_stack_jones")


def drude_eps_jax(model):
    """Lift a materials-machinery DrudeOptical into a jax-traceable closure
    eps_of(n_m3, lambda_m) -> complex eps -- the carrier-density -> permittivity link for
    gradient design THROUGH the carrier actuation (chain: n_m3 -> eps -> rcwa/pmm R/T).

    DrudeOptical.eps itself np.asarray()s its inputs (host-only), so this rebuilds the SAME
    formula (eps_inf - omega_p^2 / (omega^2 + i omega gamma), omega_p^2 = n e^2 / (eps0 m))
    from the model's parameters and dynameta.constants -- byte-identical at concrete inputs
    (pinned in tests). The static parameters are validated eagerly with DrudeOptical's own
    rules; per-density CALLABLE m_opt_kg / gamma_rad_s raise (host-numpy callables cannot
    trace -- reparameterize them jax-side if needed). n_m3 and lambda_m may each be traced."""
    from dynameta.constants import C_LIGHT, EPS0, Q_E
    from dynameta.materials.optical_model import DrudeOptical
    if not isinstance(model, DrudeOptical):
        raise TypeError("drude_eps_jax: expected a DrudeOptical, got {!r}".format(
            type(model).__name__))
    if callable(model.m_opt_kg) or callable(model.gamma_rad_s):
        raise NotImplementedError(
            "drude_eps_jax: callable (per-density) m_opt_kg / gamma_rad_s run host-side numpy "
            "and cannot be traced; use scalar parameters (or supply your own jax closure).")
    eps_inf = float(model.eps_inf)
    m = float(model.m_opt_kg)
    g = float(model.gamma_rad_s)
    if not (np.isfinite(m) and m > 0.0):
        raise ValueError("drude_eps_jax: m_opt_kg must be finite and > 0 (got {!r}).".format(m))
    if not (np.isfinite(g) and g >= 0.0):
        raise ValueError("drude_eps_jax: gamma_rad_s must be finite and >= 0 (negative damping "
                         "is gain under exp(-i omega t); got {!r}).".format(g))
    pref = Q_E * Q_E / (EPS0 * m)

    def eps_of(n_m3, lambda_m):
        omega = 2.0 * np.pi * C_LIGHT / lambda_m
        return eps_inf - (pref * n_m3) / (omega * omega + 1j * omega * g)

    return eps_of
