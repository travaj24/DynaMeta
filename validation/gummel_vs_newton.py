"""Gummel (decoupled) DC solve vs the coupled Newton: the validation the EXPERIMENTAL tag
demanded. Device: a 1D unipolar uniform n-bar (400 nm, N_D = 4e25 m^-3, mu = 0.004 m^2/Vs,
ohmic contacts) -- drift-dominated, so the converged state has an EXACT analytic ohmic limit.

GATE A (same fixed point, fields): from the same doping seed, solve_dc(method='gummel')
        converges to the SAME (Potential, Electrons) as method='newton' at 10 mV bias
        (< 1e-6 max rel difference -- both are fixed points of the same discrete system).
GATE B (same observable + independent oracle): both terminal currents agree (< 1e-8 rel) AND
        match the analytic ohmic density J = q n mu V / L (< 1e-3 rel -- the uniform bar's
        exact small-bias limit; the residual is the discrete SG vs continuum difference).
GATE C (reduces to equilibrium): at zero bias both methods give |J| < 1e-9 * J(10 mV) and a
        flat potential (max |psi| < 1 uV).
GATE D (contract): the EXPERIMENTAL-path warning fires on the gummel route.

Scope note (kept honest): this validates the unipolar OHMIC-transport case. The degenerate
gated-ACCUMULATION case Gummel was originally built for remains unproven (the
physics_drift_diffusion KNOWN LIMITATION); the dc_solve docstring carries the split.

Run: python -m validation.gummel_vs_newton
"""
import contextlib
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import devsim as ds

from dynameta.constants import Q_E
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.contact_current import extract_contact_currents
from dynameta.carriers.dc_solve import solve_dc
from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                       setup_semiconductor_region_dd)

LEN = 400e-9
N_D = 4.0e25
MU = 0.004
V_BIAS = 0.01
ABS_ERR, REL_ERR, MAX_ITER = 1.0e16, 1.0e-7, 100


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _build(tag):
    mesh, dev, reg = "gvm_" + tag, "gvd_" + tag, "bar"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="left")
    ds.add_1d_mesh_line(mesh=mesh, pos=LEN, ps=2e-9, tag="right")
    ds.add_1d_contact(mesh=mesh, name="left", tag="left", material="metal")
    ds.add_1d_contact(mesh=mesh, name="right", tag="right", material="metal")
    ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="left", tag2="right")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    setup_semiconductor_region_dd(dev, reg, n_bg_m3=N_D, eps_static=9.5,
                                  dos_mass_kg=0.35 * 9.1093837015e-31, mobility_m2Vs=MU)
    for c in ("left", "right"):
        setup_contact_ohmic_dd(dev, c)
    ds.set_node_values(device=dev, region=reg, name="Electrons",
                       values=[N_D] * len(ds.get_node_model_values(device=dev, region=reg,
                                                                   name="Electrons")))
    return mesh, dev, reg


def _teardown(dev, mesh):
    try:
        _R.clear(dev)
        ds.delete_device(device=dev)
        ds.delete_mesh(mesh=mesh)
    except Exception:
        pass


def _run(tag, method, bias_V):
    mesh, dev, reg = _build(tag)
    ds.set_parameter(device=dev, name="right_bias", value=float(bias_V))
    kw = dict(abs_tol=ABS_ERR, rel_tol=REL_ERR, max_iter=MAX_ITER,
              semiconductor_regions=[reg])
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        with _quiet():
            solve_dc(dev, method=method, **kw)
    warned = any("EXPERIMENTAL" in str(w.message) for w in wlist)
    psi = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="Potential"))
    n = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
    cc = extract_contact_currents(dev)
    _teardown(dev, mesh)
    return psi, n, cc, warned


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-6)))


def main():
    print("[gn] === Gummel vs Newton: unipolar ohmic bar, {} mV ===".format(
        1e3 * V_BIAS), flush=True)
    ok = True

    psi_n, n_n, cc_n, _ = _run("n", "newton", V_BIAS)
    psi_g, n_g, cc_g, warned = _run("g", "gummel", V_BIAS)

    d_psi, d_n = _rel(psi_n, psi_g), _rel(n_n, n_g)
    g_a = bool(d_psi < 1e-6 and d_n < 1e-6)
    ok = ok and g_a
    print("[gn] GATE A: same fixed point -- max rel d(psi) = {:.2e}, d(n) = {:.2e} -> {}".format(
        d_psi, d_n, "PASS" if g_a else "FAIL"), flush=True)

    J_n, J_g = abs(cc_n["left"]), abs(cc_g["left"])
    J_ohm = Q_E * N_D * MU * V_BIAS / LEN                    # exact uniform-bar drift limit
    dJ = abs(J_n - J_g) / J_n
    dJo = max(abs(J_n - J_ohm), abs(J_g - J_ohm)) / J_ohm
    g_b = bool(dJ < 1e-8 and dJo < 1e-3)
    ok = ok and g_b
    print("[gn] GATE B: J newton/gummel/ohmic = {:.6e}/{:.6e}/{:.6e} A/m^2 "
          "(rel {:.1e}, vs ohmic {:.1e}) -> {}".format(
              J_n, J_g, J_ohm, dJ, dJo, "PASS" if g_b else "FAIL"), flush=True)

    psi_n0, _, cc_n0, _ = _run("n0", "newton", 0.0)
    psi_g0, _, cc_g0, _ = _run("g0", "gummel", 0.0)
    J0 = max(abs(cc_n0["left"]), abs(cc_g0["left"]))
    flat = max(float(np.max(np.abs(psi_n0))), float(np.max(np.abs(psi_g0))))
    g_c = bool(J0 < 1e-9 * J_n and flat < 1e-6)
    ok = ok and g_c
    print("[gn] GATE C: zero bias reduces -- |J| = {:.2e} (vs biased {:.2e}), "
          "max |psi| = {:.2e} V -> {}".format(J0, J_n, flat, "PASS" if g_c else "FAIL"),
          flush=True)

    g_d = bool(warned)
    ok = ok and g_d
    print("[gn] GATE D: experimental-path warning fires on the gummel route -> {}".format(
        "PASS" if g_d else "FAIL"), flush=True)

    print("[gn] *** GUMMEL vs NEWTON: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
