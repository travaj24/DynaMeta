"""Differentiable RCWA + PMM design twins (audit 8.1-5).

rcwa_design.py wraps Lumenairy's JAX-differentiable RCWA (rcwa_efficiency_1d, RCWAStack) and
PMM (PMMStack) twins as DynaMeta-shaped forwards for gradient design of layered/periodic
structures. These gates pin the two claims that make a design twin trustworthy: the traced
forward is the SAME physics as the production (non-JAX) bridge solve, and the analytic
gradient matches finite differences OF THAT INDEPENDENT ENGINE PATH (numpy RCWA takes the
TE/TM-decoupled / even-parity fast paths; the jax twin runs the general 2N cascade -- so FD
of the non-JAX bridge is a genuinely different code path, the strongest available oracle).

GATE A (twin == non-JAX bridge forward): rcwa_stack_RT on a lamellar grating stack (the SAME
        128-sample rasterized cell the bridge builds) matches make_lumenairy_rcwa_solver's
        R and T < 1e-10; a uniform lossy stack through the LIFTED (constant-cell) path
        matches < 1e-10 too; the functional grating twin (rcwa_grating_RT) jax forward
        equals its own numpy forward < 1e-12.
GATE B (gradient correctness vs the independent path): jax.grad of R through rcwa_stack_RT
        w.r.t. the grating layer THICKNESS and the ridge EPS (real part) matches
        Richardson-extrapolated central finite differences of the NON-JAX bridge solve
        (design rebuilt per FD point) < 1e-5 relative.
GATE C (gradient nonzero + descent sanity): two half-Newton gradient-descent steps on
        |R(eps) - R_target|^2 (eager grad) strictly reduce the objective.
GATE D (PMM design twin): pmm_stack_RT on the same lamellar geometry (analytic segments)
        matches make_lumenairy_pmm_solver < 1e-9, and jax.grad w.r.t. the ridge eps AND the
        WAVELENGTH (a parameter the RCWAStack twin cannot trace) match Richardson FD of the
        non-JAX PMM bridge < 1e-5 relative.

Honest SKIP (exit 0 + banner) when lumenairy or jax is not importable.

Run: python -m validation.lumenairy_rcwa_jax
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAM = 1.31e-6
PER = 600e-9
T0 = 180e-9
EPS_RIDGE = 6.0
N_ORD = 5
SX = 128                                  # the bridge's structured-cell sampling (pinned)


def _design(thick, eps_ridge):
    """The gates' lamellar fixture: air | grating(air bg + ridge lines, duty 0.5) | glass,
    normal incidence, pol 'x' (row 0) -- the same shape as tests' _grating_design, with the
    thickness and ridge eps parameterized for the FD oracle."""
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("ridge", ConstantOptical(complex(eps_ridge))))
    ridge = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 0.5 * PER, PER),
                      material="ridge")
    return Design(name="g", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=[Layer("grating", thick, "air", inclusions=[ridge])],
                              superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0))


def _richardson(f, x, h):
    """Richardson-extrapolated central difference (cancels the O(h^2) truncation)."""
    d1 = (f(x + h) - f(x - h)) / (2.0 * h)
    d2 = (f(x + h / 2.0) - f(x - h / 2.0)) / h
    return (4.0 * d2 - d1) / 3.0


def main():
    if importlib.util.find_spec("lumenairy") is None or importlib.util.find_spec("jax") is None:
        print("[rjx] *** SKIP: lumenairy or jax not installed -- RCWA/PMM JAX gates not run ***",
              flush=True)
        return True
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm_solver,
                                                  make_lumenairy_rcwa_solver, pmm_stack_RT,
                                                  rcwa_grating_RT, rcwa_stack_RT)
    from dynameta.optics.rasterize import cell_axes, layer_eps_cell

    print("[rjx] === Differentiable RCWA + PMM design twins (parity, AD vs FD, descent) ===",
          flush=True)
    ok = True

    # The bridge's own rasterization of the grating layer (identical geometry on both sides:
    # design_to_rcwa_stack paints structured 1-D layers on a (128, 1) cell-centred grid).
    d0 = _design(T0, EPS_RIDGE)
    xs, ys = cell_axes(SX, 1, PER, PER)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cell0 = layer_eps_cell(d0.stack.layers[0], X, Y, LAM, d0.materials, {})
    mask = jnp.asarray(cell0 == complex(EPS_RIDGE))

    def bridge_R(thick, eps_ridge):
        r = make_lumenairy_rcwa_solver(n_orders=N_ORD, cell_samples=SX)(
            _design(float(thick), float(eps_ridge)), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        return r.R, r.T

    def twin_RT(thick, eps_ridge):
        cell = jnp.where(mask, eps_ridge + 0.0j, 1.0 + 0.0j)
        return rcwa_stack_RT([(cell, thick)], 1.5 + 0j, 1.0 + 0j, LAM, period_x=PER,
                             n_orders=N_ORD, row=0)

    # ---- GATE A: twin forward == the non-JAX bridge solve (independent lumenairy path) ----
    Rb, Tb = bridge_R(T0, EPS_RIDGE)
    Rj, Tj = twin_RT(jnp.asarray(T0), jnp.asarray(EPS_RIDGE))
    par_g = max(abs(Rb - float(Rj)), abs(Tb - float(Tj)))

    # uniform lossy stack through the LIFTED constant-cell path vs the bridge's eps= path
    from dynameta.geometry import Design, Layer, Stack, UnitCell
    from dynameta.geometry.specs import OpticalSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(complex(4.0, 0.3))))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    du = Design(name="u", unit_cell=UnitCell.square(300e-9),
                stack=Stack(layers=[Layer("a", 120e-9, "hi")], superstrate_material="air",
                            substrate_material="glass"),
                electrodes=[], materials=reg,
                optical=OpticalSpec(polarization="x", incidence_angle_deg=0.0))
    ru = make_lumenairy_rcwa_solver(n_orders=2)(du, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    Ru, Tu = rcwa_stack_RT([(jnp.asarray(4.0 + 0.3j), 120e-9)], 1.5 + 0j, 1.0 + 0j, LAM,
                           period_x=300e-9, n_orders=2, row=0)
    par_u = max(abs(ru.R - float(Ru)), abs(ru.T - float(Tu)))

    # functional grating twin: jax forward == its own numpy forward (general-2N vs fast path)
    args = (PER, EPS_RIDGE + 0j, 1.0 + 0j, 1.5 + 0j, 1.0 + 0j, T0, 0.5, LAM)
    Rn, Tn = rcwa_grating_RT(*args, angle=0.0, polarization="tm", n_orders=N_ORD)
    Rx, Tx = rcwa_grating_RT(PER, jnp.asarray(EPS_RIDGE + 0j), 1.0 + 0j, 1.5 + 0j, 1.0 + 0j,
                             jnp.asarray(T0), 0.5, jnp.asarray(LAM), angle=0.0,
                             polarization="tm", n_orders=N_ORD)
    par_f = max(abs(float(Rn) - float(Rx)), abs(float(Tn) - float(Tx)))
    g_a = bool(par_g < 1e-10 and par_u < 1e-10 and par_f < 1e-12)
    ok = ok and g_a
    print("[rjx] GATE A: twin==bridge grating {:.2e}, lifted-uniform {:.2e}, "
          "grating-fn jax==numpy {:.2e} -> {}".format(par_g, par_u, par_f,
                                                      "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: jax.grad vs Richardson FD of the NON-JAX bridge (thickness + eps) --------
    g_t = float(jax.grad(lambda t: jnp.real(twin_RT(t, jnp.asarray(EPS_RIDGE))[0]))(
        jnp.asarray(T0)))
    g_e = float(jax.grad(lambda e: jnp.real(twin_RT(jnp.asarray(T0), e)[0]))(
        jnp.asarray(EPS_RIDGE)))
    fd_t = _richardson(lambda t: bridge_R(t, EPS_RIDGE)[0], T0, 1e-9)
    fd_e = _richardson(lambda e: bridge_R(T0, e)[0], EPS_RIDGE, 1e-5)
    rel_t = abs(g_t - fd_t) / (abs(fd_t) + 1e-30)
    rel_e = abs(g_e - fd_e) / (abs(fd_e) + 1e-30)
    g_b = bool(rel_t < 1e-5 and rel_e < 1e-5 and abs(g_t) > 0.0 and abs(g_e) > 0.0)
    ok = ok and g_b
    print("[rjx] GATE B: AD vs bridge-FD rel: d/d(thickness) {:.2e} (AD {:+.4e}), "
          "d/d(eps_ridge) {:.2e} (AD {:+.4e}) -> {}".format(
              rel_t, g_t, rel_e, g_e, "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: two gradient-descent steps reduce |R(eps) - R_target|^2 (eager grad) -----
    R_tgt = float(twin_RT(jnp.asarray(T0), jnp.asarray(6.3))[0])

    def loss(e):
        return (jnp.real(twin_RT(jnp.asarray(T0), e)[0]) - R_tgt) ** 2

    grad_loss = jax.grad(loss)
    e_i = jnp.asarray(EPS_RIDGE)
    losses = [float(loss(e_i))]
    for _ in range(2):
        g = grad_loss(e_i)
        # half-Newton step for a scalar quadratic-in-R objective: lr*g = L / (2 g) * ...
        lr = losses[-1] / (float(g) ** 2 + 1e-30)
        e_i = e_i - lr * g
        losses.append(float(loss(e_i)))
    g_c = bool(losses[1] < losses[0] and losses[2] < losses[1] and losses[0] > 0.0)
    ok = ok and g_c
    print("[rjx] GATE C: descent {:.3e} -> {:.3e} -> {:.3e} (eps {:.4f} toward 6.3) -> {}"
          .format(losses[0], losses[1], losses[2], float(e_i),
                  "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: PMM design twin (parity + AD vs FD incl. the wavelength gradient) --------
    DEG, NOP = 10, 9
    segs0 = [(0.25, 1.0 + 0j), (0.5, EPS_RIDGE + 0j), (0.25, 1.0 + 0j)]

    def bridge_pmm_R(eps_ridge, lam):
        r = make_lumenairy_pmm_solver(degree=DEG, n_orders=NOP)(
            _design(T0, float(eps_ridge)), None, {}, float(lam), 1.0 + 0j, 1.5 + 0j)
        return r.R

    def twin_pmm_R(eps_ridge, lam):
        segs = [(0.25, 1.0 + 0j), (0.5, eps_ridge + 0.0j), (0.25, 1.0 + 0j)]
        R, _T = pmm_stack_RT([(segs, T0)], 1.5 + 0j, 1.0 + 0j, lam, period=PER,
                             degree=DEG, n_orders=NOP, row=0)
        return jnp.real(R)

    Rpb = bridge_pmm_R(EPS_RIDGE, LAM)
    Rpj = float(twin_pmm_R(jnp.asarray(EPS_RIDGE), jnp.asarray(LAM)))
    par_p = abs(Rpb - Rpj)
    gp_e = float(jax.grad(lambda e: twin_pmm_R(e, jnp.asarray(LAM)))(jnp.asarray(EPS_RIDGE)))
    gp_l = float(jax.grad(lambda w: twin_pmm_R(jnp.asarray(EPS_RIDGE), w))(jnp.asarray(LAM)))
    fdp_e = _richardson(lambda e: bridge_pmm_R(e, LAM), EPS_RIDGE, 1e-5)
    fdp_l = _richardson(lambda w: bridge_pmm_R(EPS_RIDGE, w), LAM, 2e-12)
    relp_e = abs(gp_e - fdp_e) / (abs(fdp_e) + 1e-30)
    relp_l = abs(gp_l - fdp_l) / (abs(fdp_l) + 1e-30)
    g_d = bool(par_p < 1e-9 and relp_e < 1e-5 and relp_l < 1e-5
               and abs(gp_e) > 0.0 and abs(gp_l) > 0.0)
    ok = ok and g_d
    print("[rjx] GATE D: PMM twin==bridge {:.2e}; AD vs bridge-FD rel: d/d(eps) {:.2e}, "
          "d/d(wavelength) {:.2e} -> {}".format(par_p, relp_e, relp_l,
                                                "PASS" if g_d else "FAIL"), flush=True)

    print("[rjx] *** RCWA/PMM JAX (DIFFERENTIABLE) DESIGN TWINS: {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
