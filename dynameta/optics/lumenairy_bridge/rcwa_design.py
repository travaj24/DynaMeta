"""Differentiable RCWA + PMM forwards for layered/periodic inverse design (audit 8.1-5).

berreman_design.py gives DynaMeta an exact-gradient forward for the PLANAR-anisotropic class;
this module extends the semi-analytic-gradient surface to LAYERED/PERIODIC structures (binary
gratings, patterned metasurface cells, carrier-modulated multilayers) by wrapping Lumenairy's
JAX-differentiable RCWA and PMM twins -- far cheaper per iterate than the JAX-FDTD
topology-opt path (optics.inverse_design.optimize_fdtd), which pays a full space-time march
per gradient. REQUIRES jax double precision (jax.config.update("jax_enable_x64", True)):
lumenairy's traced solves refuse/warn on f32 (the RCWA eigenproblem is ill-conditioned there).

WHAT LUMENAIRY ACTUALLY TRACES (pinned from the 5.21 source + lumenairy's own tests; each
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
  traced segment eps (scalar or in-plane (3, 3), re AND im), layer thicknesses, wavelength,
  angle, half-space indices. STATIC: period, segment WIDTHS (frozen union grid), degree,
  order count; slant / out-of-plane tensors / stabilize / retain_internal raise upstream.
  Upstream pin: tests/unit/test_v5_14_2_jax_stacks.py. The PMM twin is SHIPPED (not skipped):
  it is genuinely usable for 1-D lamellar gratings with gradients, and it traces exactly the
  source/half-space parameters the RCWAStack twin cannot -- the two are complementary.

Convention: identical on both sides (public exp(-i omega t), Im(eps) > 0 lossy, metres,
radians). Efficiency rows are keyed INCIDENT lab E_x (row 0) / E_y (row 1) -- never relabel
as TE/TM; the zeroth-order Jones (rcwa_stack_jones) carries the phase observable r.

VERSION NOTE on jit/vmap: unlike the Berreman twin (see berreman_design.py -- its eig-VJP
pytree fix landed only post-tagged-5.14.4), the RCWA/PMM twins' gauge-stable custom-VJP eig
already returns a plain (eigvals, eigvecs) tuple on the whole bridge floor (>= 5.21;
lumenairy rcwa/_core.py _jax_eig_stable), so grad-of-vmap / Hessian compose without a version
condition (pinned upstream). Eager jax.grad is the gate-validated bridge path
(validation/lumenairy_rcwa_jax.py: parity vs the non-JAX bridge, AD vs FD of the non-JAX
bridge, descent sanity).

Carrier modulation: drude_eps_jax lifts an existing materials-machinery DrudeOptical into a
jax-traceable carrier-density -> eps closure (same constants, same formula, eagerly validated
static parameters), so a carrier-actuated layer chains n_m3 -> eps -> R/T with one jax.grad.
"""

from __future__ import annotations

import numpy as np

from dynameta.optics.lumenairy_bridge._common import require_lumenairy as _require_lumenairy

__all__ = ["rcwa_grating_RT", "rcwa_stack_RT", "rcwa_stack_jones", "pmm_stack_RT",
           "pmm_stack_jones", "drude_eps_jax"]


def _is_jaxish(x) -> bool:
    """True for ANY jax value (concrete device array or tracer) WITHOUT importing jax: the
    bridge stays import-light (hygiene contract) and the check must not concretize a trace."""
    mod = type(x).__module__ or ""
    return mod.split(".")[0] in ("jax", "jaxlib")


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


def rcwa_grating_RT(period, eps_ridge, eps_groove, n_substrate, n_superstrate, depth,
                    duty_cycle, wavelength, *, angle=0.0, polarization="te", n_orders=11,
                    formulation="auto"):
    """Differentiable (R_total, T_total) of a 1-D binary grating for ONE linear polarization
    ('te' = s = E along the grooves, 'tm' = p) -- the scalar FOM ingredients for jax.grad.

    Routes to lumenairy's JAX twin when ANY of eps_ridge / eps_groove / n_substrate /
    n_superstrate / depth / angle / wavelength is a jax array (gradients then flow through all
    of them, real and imaginary parts); plain numbers give the concrete NumPy forward. STATIC:
    period, duty_cycle, n_orders, polarization, formulation. Region eps follows DynaMeta's
    materials convention and is lifted to lumenairy's native refractive INDEX by the principal
    sqrt (differentiable; a passive Im(eps) > 0 maps to Im(n) > 0); the half-spaces are
    indices, as everywhere in the bridge."""
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


def _solve_rcwa_stack(layers, n_substrate, n_superstrate, wavelength, *, period_x, period_y,
                      theta, phi, n_orders, n_orders_y, formulation, fn_name):
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


def rcwa_stack_RT(layers, n_substrate, n_superstrate, wavelength, *, period_x, period_y=None,
                  theta=0.0, phi=0.0, n_orders=11, n_orders_y=None, formulation="laurent",
                  row=0):
    """Differentiable order-summed (R, T) of a patterned multilayer for ONE incident lab
    polarization (row 0 = E_x, 1 = E_y) -- a scalar FOM ingredient for jax.grad.

    `layers` = [(eps_spec, thickness), ...] superstrate-side first (the berreman_RT layer
    convention); see _add_stack_layers for the shape dispatch (a uniform scalar, concrete or
    traced, goes straight to add_layer(eps=)). Gradients flow through every jax-typed eps
    VALUE (cells: values only, walls static; a uniform eps traces raw), every jax-typed
    thickness, AND (lumenairy 5.22) the source wavelength / theta / phi. STATIC here: the
    half-space indices and the periods (TypeError on a jax value -- rcwa_grating_RT and
    pmm_stack_RT trace the half-space indices). A 1-D stack (period_y=None) is genuinely
    cheaper and better-conditioned than a y-degenerate 2-D one -- keep lamellar problems
    1-D."""
    res = _solve_rcwa_stack(layers, n_substrate, n_superstrate, wavelength,
                            period_x=period_x, period_y=period_y, theta=theta, phi=phi,
                            n_orders=n_orders, n_orders_y=n_orders_y,
                            formulation=formulation, fn_name="rcwa_stack_RT")
    _orders, R, T = res.efficiencies()
    return R[row].sum(), T[row].sum()


def rcwa_stack_jones(layers, n_substrate, n_superstrate, wavelength, *, period_x,
                     period_y=None, theta=0.0, phi=0.0, n_orders=11, n_orders_y=None,
                     formulation="laurent"):
    """Differentiable full far field of a patterned multilayer: (orders, R_eff, T_eff,
    jones_r, jones_t) with per-order (2, N) efficiencies and the zeroth-order (2, 2) lab-basis
    Jones matrices -- for phase-bearing FOMs (the modulator observable r = jones_r[row, row])
    and per-order targets. Same layer convention, tracing surface and static-argument policy
    as rcwa_stack_RT."""
    res = _solve_rcwa_stack(layers, n_substrate, n_superstrate, wavelength,
                            period_x=period_x, period_y=period_y, theta=theta, phi=phi,
                            n_orders=n_orders, n_orders_y=n_orders_y,
                            formulation=formulation, fn_name="rcwa_stack_jones")
    orders, R, T = res.efficiencies()
    return orders, R, T, res.jones_reflection(), res.jones_transmission()


def _solve_pmm_stack(layers, n_substrate, n_superstrate, wavelength, *, period, angle,
                     degree, n_orders, fn_name):
    lum = _require_lumenairy()
    if _is_jaxish(period):
        raise TypeError(
            "{}: period must be a concrete number (the PMM union grid / segment walls are "
            "frozen NumPy geometry; only the segment eps VALUES, thicknesses, wavelength, "
            "angle and half-space indices trace).".format(fn_name))
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


def pmm_stack_RT(layers, n_substrate, n_superstrate, wavelength, *, period, angle=0.0,
                 degree=12, n_orders=21, row=0):
    """Differentiable order-summed (R, T) of a 1-D lamellar stack via lumenairy's PMM JAX
    twin (spectral element -- no Fourier-factorization accuracy floor) for ONE incident lab
    polarization (row 0 = E_x, 1 = E_y).

    `layers` = [(spec, thickness), ...] superstrate-side first; spec is either a segment list
    [(width_fraction, eps), ...] (widths STATIC and summing to 1; eps scalar or in-plane
    (3, 3) tensor) or a bare eps for a uniform layer. Gradients flow through every jax-typed
    eps (re AND im), thickness, AND -- unlike the RCWAStack twin -- wavelength, angle and the
    half-space indices. STATIC: period, widths, degree, n_orders. Out-of-plane tensors /
    slants raise upstream (lumenairy's own guards). NOTE the traced-wavelength caveat
    (lumenairy pmm/_jax_stack.py): the far-field order set is sized from concrete numbers, so
    wavelength gradients are valid BETWEEN order cutoffs (Wood anomalies)."""
    _orders, R, T, _jones = _solve_pmm_stack(layers, n_substrate, n_superstrate, wavelength,
                                             period=period, angle=angle, degree=degree,
                                             n_orders=n_orders, fn_name="pmm_stack_RT")
    return R[row].sum(), T[row].sum()


def pmm_stack_jones(layers, n_substrate, n_superstrate, wavelength, *, period, angle=0.0,
                    degree=12, n_orders=21):
    """Differentiable full PMM far field: (orders, R_eff, T_eff, jones_r) -- per-order (2, M)
    efficiencies plus the zeroth-order (2, 2) reflection Jones (PMM exposes NO transmission
    Jones; see pmm_backend). Same layer convention and tracing surface as pmm_stack_RT."""
    return _solve_pmm_stack(layers, n_substrate, n_superstrate, wavelength, period=period,
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
