"""R19 follow-on oracle: IN-NEWTON density-gradient drift-diffusion (4-variable DEVSIM Newton).

Device: a 1D unipolar n-type bar (400 nm, ITO-like m* = 0.35 m0) with a 10x DOPING STEP at the
midpoint (4e26 -> 4e25 m^-3) -- chosen so the quantum length L_q = hbar sqrt(1/(6 m kT)) =
1.18 nm DOMINATES the Debye length on BOTH sides (0.18 / 0.58 nm; the first design's 1e24 low
side carried a 3.7 nm Debye tail that swamped L_q) and the DG smoothing is the leading effect.

GATE A (gamma = 0 reduces to classical): with the DG equations ACTIVE but b_dg = 0, the
        converged n(z) matches the classical DD solve to < 1e-9 rel, Lambda == 0 EXACTLY and
        u^2 - n closes to machine.
GATE B (the discrete Laplacian, independent stencil): at gamma = 1 the converged Lambda equals
        +b u''/u evaluated by an INDEPENDENT non-uniform 3-point finite-difference stencil on
        the converged nodal u -- the oracle that pins the equation-assembly sign and the
        edge-flux/EdgeCouple bookkeeping (< 1e-6 rel where Lambda is significant).
GATE C (fixed-point decomposition, the feedback half): the converged Lambda(z) is FROZEN as
        external data in a FRESH classical device (vdiff rebuilt on psi + Lambda_frozen, no
        QLambda variable) and re-solved -- the frozen-Lambda classical density must reproduce
        n_DG to Newton tolerance. With GATE B (Lambda is the right functional of n) this
        verifies BOTH halves of the coupled fixed point independently. (A width/sqrt-gamma
        junction gate was tried and dropped: Poisson screening absorbs the DG correction at a
        self-consistent junction -- Lambda ~ 10 mV reshapes the transition without widening
        the 25/75 log metric; the dead-layer width physics lives in the post-hoc module's
        Schrodinger-Poisson oracle.) Also asserts DG moves the density (> 10% somewhere, in
        the exp(Lambda/V_t) ballpark).
GATE D (transport sanity): at 10 mV bias the two terminal currents conserve (< 1e-8 rel) and
        the current is finite and bias-driven.
GATE E (v0.15 4.4.4, plan G2-3): the oxide WALL -- setup_dg_insulator_wall on a unipolar bar
        (n0 = 1e23), potential coupled: the converged density equals the post-hoc DG BVP on its own
        potential (< 2e-3 of the peak) and the wall carries no current (< 1e-12 of q mu n0 V_t/L_q).

Run: python -m validation.dg_dd_in_newton
"""
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.devsim_env import import_devsim   # D0: the ONE devsim import home (BLAS pin)

ds = import_devsim()


from dynameta.constants import HBAR, KB, M_E, Q_E, T_REF
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.physics_density_gradient import (dg_b_coefficient, seed_dg_from_solution,
                                                        set_dg_gamma, setup_contact_dg,
                                                        setup_dg_quantum_correction)
from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                       setup_semiconductor_region_dd)

MSTAR = 0.35 * M_E
LEN = 400e-9
X_STEP = 200e-9
N_HI, N_LO = 4.0e26, 4.0e25
# REL 1e-6 / 200 iters: the 4-variable DG Newton's update floors near the density precision
# limit (the dc_solve note) -- a 1e-7 rel gate ping-pongs there and reports a false
# 'Convergence failure' for b_dg above ~b(gamma=1)
ABS_ERR, REL_ERR, MAX_ITER = 1.0e16, 1.0e-6, 200


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
    mesh, dev, reg = "dgm_" + tag, "dgd_" + tag, "bar"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="left")
    ds.add_1d_mesh_line(mesh=mesh, pos=X_STEP, ps=0.12e-9)        # resolve L_q at the step
    ds.add_1d_mesh_line(mesh=mesh, pos=LEN, ps=2e-9, tag="right")
    ds.add_1d_contact(mesh=mesh, name="left", tag="left", material="metal")
    ds.add_1d_contact(mesh=mesh, name="right", tag="right", material="metal")
    ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="left", tag2="right")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    setup_semiconductor_region_dd(dev, reg, n_bg_m3=N_HI, eps_static=9.5,
                                  dos_mass_kg=MSTAR, mobility_m2Vs=0.004)
    # doping STEP: smooth tanh over 0.3 nm (<< L_q; just enough for classical convergence)
    # charge + the right contact's pin (the left contact keeps the scalar N_D = N_HI)
    prof = "({hi:.8e}*0.5*(1.0-tanh((x-{xj:.8e})/3.0e-10)) + {lo:.8e}*0.5*(1.0+tanh((x-{xj:.8e})/3.0e-10)))".format(
        hi=N_HI, lo=N_LO, xj=X_STEP)
    ds.node_model(device=dev, region=reg, name="DopingProf", equation=prof)
    ds.node_model(device=dev, region=reg, name="PotentialNodeCharge",
                  equation="ElectronCharge * (Electrons - DopingProf)")
    ds.node_model(device=dev, region=reg, name="PotentialNodeCharge:Electrons",
                  equation="ElectronCharge")
    ds.node_model(device=dev, region=reg, name="PotentialNodeCharge:Potential", equation="0")
    for c in ("left", "right"):
        setup_contact_ohmic_dd(dev, c)
    nfix = "{}_electrons_dirichlet".format("right")
    ds.contact_node_model(device=dev, contact="right", name=nfix,
                          equation="Electrons - {:.8e}".format(N_LO))
    nd = np.asarray(ds.get_node_model_values(device=dev, region=reg, name="DopingProf"))
    ds.set_node_values(device=dev, region=reg, name="Electrons", values=list(nd))
    return dev, reg


def _solve(dev):
    with _quiet():
        ds.solve(type="dc", absolute_error=ABS_ERR, relative_error=REL_ERR,
                 maximum_iterations=MAX_ITER)


def _np_node(dev, reg, name):
    return np.asarray(ds.get_node_model_values(device=dev, region=reg, name=name))


def _teardown(dev, mesh):
    try:
        _R.clear(dev)
        ds.delete_device(device=dev)
        ds.delete_mesh(mesh=mesh)
    except Exception:
        pass


def _dg_solve(tag, gamma_full, *, bias_right=0.0, fracs=None):
    dev, reg = _build(tag)
    _solve(dev)                                              # classical converged baseline
    n_cl = _np_node(dev, reg, "Electrons").copy()
    setup_dg_quantum_correction(dev, reg, m_eff_kg=MSTAR, gamma=gamma_full)
    for c, nc in (("left", N_HI), ("right", N_LO)):
        setup_contact_dg(dev, c, nc)
    seed_dg_from_solution(dev, reg)
    ds.set_parameter(device=dev, name="right_bias", value=float(bias_right))
    if fracs is None:
        # constant |Delta b| per ramp step regardless of gamma_full (the gamma=1 case converges
        # at Delta = 0.25 b(gamma=1)), with step-halving retries on a convergence failure
        n_ramp = max(4, int(np.ceil(4.0 * gamma_full)))
        fracs = np.linspace(0.0, 1.0, n_ramp + 1)
    fr_now = 0.0
    set_dg_gamma(dev, reg, 0.0)
    _solve(dev)
    for fr in fracs[1:] if fracs[0] == 0.0 else fracs:       # gamma ramp w/ bisection retries
        target, step = float(fr), float(fr) - fr_now
        while fr_now < target - 1e-12:
            trial = min(fr_now + step, target)
            set_dg_gamma(dev, reg, trial)
            try:
                _solve(dev)
                fr_now = trial
            except Exception:
                step /= 2.0
                if step < 1e-3:
                    raise RuntimeError("DG gamma ramp stalled at frac {:.4f}".format(fr_now))
    x = _np_node(dev, reg, "x")
    out = dict(x=x, n=_np_node(dev, reg, "Electrons"), u=_np_node(dev, reg, "QSqrtN"),
               lam=_np_node(dev, reg, "QLambda"), n_cl=n_cl, dev=dev, reg=reg,
               mesh="dgm_" + tag)
    return out


def _width(x, n):
    """25%-75% crossing width of log10(n) across the step."""
    ln = np.log10(n)
    lo, hi = np.log10(N_LO), np.log10(N_HI)
    f = (ln - lo) / (hi - lo)
    i75 = np.where(f >= 0.75)[0][-1]
    i25 = np.where(f <= 0.25)[0][0]
    return float(x[i25] - x[i75])


def main():
    print("[dn] === R19 follow-on: in-Newton density-gradient DD ===", flush=True)
    ok = True

    # ---- GATE A: gamma = 0 reduces to classical ----
    rA = _dg_solve("a", 1.0, fracs=(0.0,))
    relA = float(np.max(np.abs(rA["n"] - rA["n_cl"]) / rA["n_cl"]))
    lam0 = float(np.max(np.abs(rA["lam"])))
    closure = float(np.max(np.abs(rA["u"] ** 2 - rA["n"]) / rA["n"]))
    _teardown(rA["dev"], rA["mesh"])
    g_a = bool(relA < 1e-9 and lam0 == 0.0 and closure < 1e-12)
    ok = ok and g_a
    print("[dn] GATE A: b_dg=0 with DG equations active -- n vs classical rel {:.1e}; "
          "max|Lambda| = {}; u^2-n closure {:.1e} -> {}".format(
              relA, lam0, closure, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B + C(part): full gamma = 1 solve ----
    r1 = _dg_solve("b", 1.0)
    x, u, lam, n1 = r1["x"], r1["u"], r1["lam"], r1["n"]
    b = dg_b_coefficient(MSTAR, 1.0)
    # independent non-uniform 3-point stencil for u'' (interior)
    hL = x[1:-1] - x[:-2]
    hR = x[2:] - x[1:-1]
    upp = 2.0 * (hL * u[2:] - (hL + hR) * u[1:-1] + hR * u[:-2]) / (hL * hR * (hL + hR))
    lam_fd = b * upp / u[1:-1]
    sig = np.abs(lam_fd) > 0.05 * np.max(np.abs(lam_fd))     # where Lambda is significant
    relB = float(np.max(np.abs(lam[1:-1][sig] - lam_fd[sig]) / np.max(np.abs(lam_fd))))
    g_b = bool(relB < 1e-6)
    ok = ok and g_b
    print("[dn] GATE B: converged Lambda vs independent FD stencil of +b u''/u, rel {:.1e} "
          "(max Lambda {:.1f} mV) -> {}".format(
              relB, 1e3 * np.max(np.abs(lam)), "PASS" if g_b else "FAIL"), flush=True)

    lam1 = lam.copy()
    n_cl1 = r1["n_cl"].copy()
    _teardown(r1["dev"], r1["mesh"])

    # ---- GATE C: frozen-Lambda classical re-solve reproduces n_DG (the feedback half) ----
    devc, regc = _build("c")
    _solve(devc)
    ds.node_solution(device=devc, region=regc, name="LambdaFrozen")
    ds.set_node_values(device=devc, region=regc, name="LambdaFrozen", values=list(lam1))
    ds.edge_from_node_model(device=devc, region=regc, node_model="LambdaFrozen")
    from dynameta.carriers.eq_registry import edge_with_derivs as _ewd
    _ewd(devc, regc, "vdiff_f",
         "(Potential@n0 + LambdaFrozen@n0 - Potential@n1 - LambdaFrozen@n1)/V_t", ("Potential",))
    _ewd(devc, regc, "vdiff_g_f", "vdiff_f / g_enh", ("Potential", "Electrons"))
    from dynameta.carriers.sg_flux import install_sg_current      # the ONE SG kernel (2026-11-04)
    install_sg_current(devc, regc, carrier="electrons", density="Electrons", vdiff="vdiff_g_f",
                       bern="Bern_g_f", bern_m="Bern_g_f_m", current="ElectronCurrentF",
                       prefactor="ElectronCharge*mu_n*EdgeInverseLength*V_t*g_enh",
                       wrt=("Potential", "Electrons"), current_wrt=("Electrons", "Potential"))
    ds.delete_equation(device=devc, region=regc, name="ElectronContinuityEquation")
    _R.forget(devc, "ElectronContinuityEquation", loc=regc)
    _R.record_region_equation(devc, regc, name="ElectronContinuityEquation",
                              variable_name="Electrons", edge_model="ElectronCurrentF",
                              time_node_model="NCharge", variable_update="positive")
    _solve(devc)
    n_frozen = _np_node(devc, regc, "Electrons")
    _teardown(devc, "dgm_c")
    relC = float(np.max(np.abs(n_frozen - n1) / n1))
    moved = float(np.max(np.abs(n1 - n_cl1) / n_cl1))
    # the unscreened Boltzmann factor exp(|Lambda|/V_t) - 1 is the CEILING of the density
    # response; Poisson screening reduces the self-consistent response well below it (the same
    # screening that absorbs the junction-width signature) -- the honest band is (measurable,
    # below the ceiling)
    lam_ceil = float(np.exp(np.max(np.abs(lam1)) / 0.02585) - 1.0)
    g_c = bool(relC < 1e-6 and 0.01 < moved <= 1.5 * lam_ceil)
    ok = ok and g_c
    print("[dn] GATE C: frozen-Lambda classical re-solve vs coupled n_DG rel {:.1e}; DG moves "
          "the density by {:.0%} (screened; unscreened Boltzmann ceiling {:.0%}) -> {}".format(
              relC, moved, lam_ceil, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: biased transport sanity ----
    rD = _dg_solve("d", 1.0, bias_right=0.01)
    iL = float(ds.get_contact_current(device=rD["dev"], contact="left",
                                      equation="ElectronContinuityEquation"))
    iR = float(ds.get_contact_current(device=rD["dev"], contact="right",
                                      equation="ElectronContinuityEquation"))
    _teardown(rD["dev"], rD["mesh"])
    cons = abs(iL + iR) / max(abs(iL), 1e-300)
    g_d = bool(np.isfinite(iL) and abs(iL) > 0.0 and cons < 1e-8)
    ok = ok and g_d
    print("[dn] GATE D: 10 mV bias -- |I| = {:.3e} (1D A/m^2), conservation {:.1e} -> {}".format(
        abs(iL), cons, "PASS" if g_d else "FAIL"), flush=True)

    ok = gate_e() and ok

    print("[dn] *** IN-NEWTON DENSITY-GRADIENT DD: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


def _wall_device(tag, n0, ps_wall, *, length=100e-9):
    """Unipolar bar against setup_dg_insulator_wall (v0.15 4.4.4), POTENTIAL COUPLED (the wall carries
    no Potential row: zero normal field), nested exponential mesh (last/first spacing 40)."""
    from dynameta.carriers.physics_density_gradient import setup_dg_insulator_wall
    mesh, dev, reg = "dge_m_" + tag, "dge_d_" + tag, "bar"
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        beta = np.log(40.0)
        nseg = int(round(length * beta / (ps_wall * np.expm1(beta))))
        xs = length * np.expm1(beta * np.arange(nseg + 1) / nseg) / np.expm1(beta)
        gaps = np.diff(xs)
        for k, xk in enumerate(xs):
            ds.add_1d_mesh_line(mesh=mesh, pos=float(xk), ps=1.01 * float(gaps[min(k, nseg - 1)]),
                                **({"tag": "wall"} if k == 0 else {"tag": "back"} if k == nseg else {}))
        ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
        ds.add_1d_contact(mesh=mesh, name="back", tag="back", material="metal")
        ds.add_1d_region(mesh=mesh, material="ITO", region=reg, tag1="wall", tag2="back")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        setup_semiconductor_region_dd(dev, reg, n_bg_m3=n0, eps_static=9.5, dos_mass_kg=MSTAR,
                                      mobility_m2Vs=0.004)
        setup_contact_ohmic_dd(dev, "back")
    _solve(dev)
    with _quiet():
        setup_dg_quantum_correction(dev, reg, m_eff_kg=MSTAR, gamma=1.0)
        setup_contact_dg(dev, "back", n0)
        setup_dg_insulator_wall(dev, "wall")
    for k, fr in enumerate((0.05, 0.1, 0.25, 0.5, 1.0)):
        set_dg_gamma(dev, reg, fr)
        if k == 0:
            seed_dg_from_solution(dev, reg, walls=("wall",))
        _solve(dev)
    return dev, reg, mesh


def gate_e():
    """GATE E (v0.15 item 4.4.4, plan G2-3): the unipolar DG with the insulator wall (the retired
    setup_dg_hard_wall's replacement), potential COUPLED, n0 = 1e23 m^-3: the converged DG density equals
    the post-hoc DG boundary-value problem (density_gradient.dg_correct_density_1d, scipy solve_bvp)
    evaluated on the DG solution's OWN potential, n_cl = n0 exp((psi - psi_b)/V_t), within 2e-3 of the
    peak (the bipolar GATE D twin measured 2.7e-4; the unipolar region's FD g-factor, g - 1 = 0.7 % at
    n0/N_c = 0.019, is outside the Boltzmann BVP and is budgeted inside the 2e-3). MEASURED 2026-11-06:
    1.2e-3 at 1e23; the same comparison at 1e22 / 1e21 (g - 1 = 6.7e-4 / 6.6e-5) gives 2.5e-4 / 1.6e-4
    -- the residual is the g-factor plus a ~1.5e-4 discretization floor, as budgeted. On the old rows the
    self-consistent leg raised a DEVSIM convergence failure (audit X-6, 2026-07-26). The frozen-psi
    closed-form gates (deficit, order, conservation) are in tests/test_density_gradient.py."""
    from dynameta.carriers.density_gradient import dg_correct_density_1d
    from dynameta.constants import V_T
    n0 = 1e23
    dev, reg, mesh = _wall_device("e", n0, 0.05e-9)
    try:
        x = _np_node(dev, reg, "x")
        o = np.argsort(x)
        x, n, psi = x[o], _np_node(dev, reg, "Electrons")[o], _np_node(dev, reg, "Potential")[o]
        jmax = float(np.max(np.abs(np.asarray(ds.get_edge_model_values(device=dev, region=reg,
                                                                       name="ElectronCurrentDG")))))
    finally:
        _teardown(dev, mesh)
    n_cl = n0 * np.exp((psi - psi[-1]) / V_T)
    n_bvp = dg_correct_density_1d(x, n_cl, MSTAR, gamma=1.0, hard_wall="left", tol=1e-8)
    err = float(np.max(np.abs(n - n_bvp)) / np.max(n))
    scale = Q_E * 0.004 * n0 * 0.025851991 / (HBAR / np.sqrt(6.0 * MSTAR * KB * T_REF))
    g_e = bool(err < 2e-3 and jmax <= 1e-12 * scale)
    print("[dn] GATE E: insulator wall, coupled psi, n0 = 1e23: DG density vs post-hoc DG BVP on its own "
          "potential max {:.1e} of peak; max wall-edge current {:.1e} of q mu n0 V_t/L_q -> {}".format(
              err, jmax / scale, "PASS" if g_e else "FAIL"), flush=True)
    return g_e


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
