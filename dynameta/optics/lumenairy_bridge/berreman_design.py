"""Differentiable Berreman forward for planar-anisotropic inverse design (roadmap v0.5 A6).

DynaMeta's differentiable inverse-design / topology-opt is built ENTIRELY on the JAX-FDTD
backend (optics.inverse_design.optimize_fdtd): a planar anisotropic FOM (an LC retarder,
waveplate, magneto-optic stack, EO modulator, or a Rytov-homogenized sub-wavelength grating) pays
a full space-time march per forward to get a gradient -- and the 3-D vector FEM path carries NO
gradient at all. The Berreman 4x4 method is an analytic cascade with an EXACT JAX adjoint
(Lumenairy CHANGELOG 5.14.4: gradients flow through every layer permittivity tensor real AND
imaginary, thickness, wavelength, angle, phi, half-space indices; AD-vs-FD <= 1e-8; vmap/jit
clean; x64), so it is the fast exactly-differentiable forward for the WHOLE planar-anisotropic
inverse-design class.

This module exposes the JAX twin as a DynaMeta-shaped scalar forward. berreman_jones_1d already
ROUTES to the jnp implementation when ANY input is a traced JAX array, so berreman_RT below is a
thin, dependency-free wrapper (numpy when given floats; the differentiable twin when given a
traced eps tensor / thickness / wavelength / angle). Build a FOM on top and take jax.grad /
jax.value_and_grad; jax.jit and jax.vmap (a wavelength / angle sweep inside one gradient) compose.

Convention: identical on both sides (public exp(-i omega t), Im(eps) > 0, metres, radians, raw
eps); `row` selects the incident lab polarization (0 = E_x, 1 = E_y). R/T are the TOTAL (co+cross)
flux-normalized power for that incident pol -- the natural planar-anisotropic FOM ingredients.
"""

from __future__ import annotations

from dynameta.optics.lumenairy_bridge.berreman_backend import _require_berreman

__all__ = ["berreman_RT", "berreman_jones"]


def berreman_RT(layers, n_substrate, n_superstrate, wavelength, *, angle=0.0, phi=0.0, row=0):
    """Differentiable (R, T) power for ONE incident lab polarization (row 0 = E_x, 1 = E_y) of a
    planar anisotropic multilayer. `layers` = [(eps, thickness), ...] superstrate-side first (eps
    scalar or (3, 3)); a SCALAR FOM ingredient suitable for jax.grad.

    Routes to the Berreman JAX twin automatically when any of layers / thickness / wavelength /
    angle / phi / half-space indices is a traced JAX array (gradients then flow through all of
    them); plain numpy floats give a concrete forward. Use as the analytic-gradient forward for a
    planar-anisotropic inverse-design FOM -- a fast exact alternative to optimize_fdtd's space-time
    march, and the only differentiable path for the anisotropic stacks the FEM cannot grad."""
    lum = _require_berreman()
    R, T, _Jr, _Jt = lum.berreman_jones_1d(layers, n_substrate, n_superstrate, wavelength,
                                           angle=angle, phi=phi)
    return R[row], T[row]


def berreman_jones(layers, n_substrate, n_superstrate, wavelength, *, angle=0.0, phi=0.0):
    """Differentiable full far field (R, T, jones_r, jones_t) of a planar anisotropic multilayer
    -- the (2,) power + (2, 2) lab-basis Jones, for a phase- / cross-pol-bearing FOM (a waveplate
    retardance, a Faraday rotation angle). Same JAX auto-dispatch as berreman_RT."""
    return _require_berreman().berreman_jones_1d(layers, n_substrate, n_superstrate, wavelength,
                                                 angle=angle, phi=phi)
