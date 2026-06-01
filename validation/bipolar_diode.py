"""
Validate bipolar drift-diffusion (electrons + holes + SRH) on a 1D Si p-n diode.

Physics: dynameta.carriers.physics_bipolar_dd (3 coupled variables: Potential,
Electrons, Holes), SI units. The mesh is a direct 1D DEVSIM device -- the full
metasurface geometry is NOT needed to validate a diode.

Device: length 2 um, abrupt junction at the midpoint. p-side NetDoping = -Na,
n-side NetDoping = +Nd, Na = Nd = 1e24 m^-3 (= 1e18 cm^-3). Si-like params:
n_i = 1.0e16 m^-3 (=1e10 cm^-3), mu_n = 0.135, mu_p = 0.048 m^2/(V s),
tau_n = tau_p = 1e-7 s, eps_r = 11.7.

Staged solve (the convergence path from docs/implementation_notes.md):
  (1) potential-only pre-solve (freeze carriers via eq_registry);
  (2) seed Electrons/Holes from the Boltzmann equilibrium node models;
  (3) coupled 3-variable Newton at 0 bias;
  (4) bias ramp (forward + reverse), terminal current via get_contact_current.

GATE A: forward/reverse J-V monotonic and rectifying (forward rises ~exp,
        reverse saturates small). A short J-V table is printed.
GATE B: minority-carrier injection under forward bias (the bipolar signature);
        and a Boltzmann-limit cross-check (FD g-factor on/off agree at this low
        degeneracy -> the FD path reduces to standard SG, i.e. toward the
        electron-only behavior in the unipolar/non-degenerate limit).

Run:  python -m validation.bipolar_diode
"""

import contextlib
import os
import sys

import numpy as np
import devsim as ds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers import physics_bipolar_dd as BP
from dynameta.carriers import eq_registry as _R
from dynameta.carriers.physics_equilibrium import V_T

# Newton tolerances. The SI continuity residual scales with the carrier density;
# the absolute Newton update floors near density*machine_eps, so a tight relative
# tol can never be met (it ping-pongs at the precision floor) -- see the dc_solve
# note. We make the ABSOLUTE error the satisfiable gate (generous, well above the
# floor, ~1e-6 of the density-current scale) and keep a moderate relative tol.
ABS_ERR = 1.0e18
REL_ERR = 1.0e-6
MAX_ITER = 80


@contextlib.contextmanager
def _quiet():
    """Suppress DEVSIM's per-iteration stdout (C-level fd redirect) so the
    validation prints only the [t] result lines."""
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

# ---- device / material constants (SI) ----
LEN_M = 2.0e-6
X_JUNC = 1.0e-6
NA_M3 = 1.0e24          # acceptor (p-side), 1e18 cm^-3
ND_M3 = 1.0e24          # donor    (n-side), 1e18 cm^-3
N_I_M3 = 1.0e16         # Si intrinsic, 1e10 cm^-3
N_DOS_M3 = 2.8e25       # Si conduction-band DOS ~2.8e19 cm^-3 (FD g-factor only)
MU_N = 0.135            # m^2/(V s)  (1350 cm^2/Vs)
MU_P = 0.048            # m^2/(V s)  (480  cm^2/Vs)
TAU_N = 1.0e-7
TAU_P = 1.0e-7
EPS_R = 11.7

DEVICE = "diode"
MESH = "diode_mesh"
REGION = "bulk"


def build_diode(fd_enhancement: bool, na_m3: float = NA_M3,
                nd_m3: float = ND_M3) -> None:
    """Create the 1D mesh, device, NetDoping step, physics, and contacts.
    na_m3/nd_m3 let the Boltzmann-limit cross-check use a lower-doped (truly
    non-degenerate) diode where the FD g-factor provably -> 1."""
    for dv in list(ds.get_device_list()):
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    _R.clear(DEVICE)

    ds.create_1d_mesh(mesh=MESH)
    # Fine spacing at the contacts and (especially) the junction; coarser bulk.
    ds.add_1d_mesh_line(mesh=MESH, pos=0.0, ps=2e-9, tag="p_contact")
    ds.add_1d_mesh_line(mesh=MESH, pos=X_JUNC, ps=5e-10)        # 0.5 nm at junction
    ds.add_1d_mesh_line(mesh=MESH, pos=LEN_M, ps=2e-9, tag="n_contact")
    ds.add_1d_contact(mesh=MESH, name="p_contact", tag="p_contact", material="metal")
    ds.add_1d_contact(mesh=MESH, name="n_contact", tag="n_contact", material="metal")
    ds.add_1d_region(mesh=MESH, material="Si", region=REGION,
                     tag1="p_contact", tag2="n_contact")
    ds.finalize_mesh(mesh=MESH)
    ds.create_device(mesh=MESH, device=DEVICE)

    # NetDoping: abrupt step, -Na on p-side (x < junction), +Nd on n-side.
    nd_eq = "ifelse(x < {xj}, {na}, {nd})".format(
        xj=X_JUNC, na=-na_m3, nd=nd_m3)
    ds.node_model(device=DEVICE, region=REGION, name="NetDoping", equation=nd_eq)

    BP.setup_bipolar_region(
        DEVICE, REGION, eps_static=EPS_R, n_dos_m3=N_DOS_M3, n_i_m3=N_I_M3,
        mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P,
        tau_n_s=TAU_N, tau_p_s=TAU_P, fd_enhancement=fd_enhancement)
    BP.setup_equilibrium_seed_models(DEVICE, REGION)

    for c in ("p_contact", "n_contact"):
        BP.setup_contact_ohmic_bipolar(DEVICE, c)


def _solve(abs_err=ABS_ERR, rel_err=REL_ERR, max_iter=MAX_ITER):
    with _quiet():
        ds.solve(type="dc", solver_type="direct", absolute_error=abs_err,
                 relative_error=rel_err, maximum_iterations=max_iter)


def staged_equilibrium_solve(verbose=False):
    """(1) potential-only pre-solve; (2) seed carriers; (3) coupled Newton @ 0 V."""
    # Seed Potential from the built-in (charge-neutral) value: psi = V_t*log(n0/n_i)
    # on the n-side, -V_t*log(p0/n_i) on the p-side (Boltzmann reference n_i).
    n0_eq = "ifelse(NetDoping > 0, {ce}, n_i^2/{ch})".format(ce=BP.CELEC, ch=BP.CHOLE)
    ds.node_model(device=DEVICE, region=REGION, name="_seed_psi",
                  equation="V_t*log({n0}/n_i)".format(n0=n0_eq))
    psi_seed = ds.get_node_model_values(device=DEVICE, region=REGION, name="_seed_psi")
    n_seed = ds.get_node_model_values(device=DEVICE, region=REGION,
                                      name="IntrinsicElectrons")
    p_seed = ds.get_node_model_values(device=DEVICE, region=REGION,
                                      name="IntrinsicHoles")
    ds.set_node_values(device=DEVICE, region=REGION, name="Potential", values=psi_seed)
    ds.set_node_values(device=DEVICE, region=REGION, name="Electrons", values=n_seed)
    ds.set_node_values(device=DEVICE, region=REGION, name="Holes", values=p_seed)

    # (1) potential-only pre-solve: freeze the two continuity equations. This is a
    # Poisson-only solve (V-unit residual) so the tight 1e10 abs / 1e-10 rel apply.
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name(DEVICE, ceq)
    _solve(1e10, 1e-10, 100)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name(DEVICE, ceq)
    if verbose:
        psi = np.array(ds.get_node_model_values(device=DEVICE, region=REGION, name="Potential"))
        print("[t]   pre-solve Potential range = [{:.4f}, {:.4f}] V (Vbi~{:.4f})".format(
            psi.min(), psi.max(), psi.max() - psi.min()), flush=True)

    # (2) re-seed carriers from equilibrium node models (Potential now updated;
    #     IntrinsicElectrons/Holes are charge-neutral, independent of Potential).
    n_seed = ds.get_node_model_values(device=DEVICE, region=REGION, name="IntrinsicElectrons")
    p_seed = ds.get_node_model_values(device=DEVICE, region=REGION, name="IntrinsicHoles")
    ds.set_node_values(device=DEVICE, region=REGION, name="Electrons", values=n_seed)
    ds.set_node_values(device=DEVICE, region=REGION, name="Holes", values=p_seed)

    # (3) coupled 3-variable Newton at 0 bias (density-scaled abs gate).
    _solve()
    if verbose:
        print("[t]   coupled equilibrium solve converged", flush=True)


def terminal_current(contact):
    """Total terminal current = electron + hole contact current (A; 1D -> A/m^2
    numerically since unit cross-section)."""
    jn = ds.get_contact_current(device=DEVICE, contact=contact,
                                equation="ElectronContinuityEquation")
    jp = ds.get_contact_current(device=DEVICE, contact=contact,
                                equation="HoleContinuityEquation")
    return jn + jp, jn, jp


def ramp_to(target_v, v_step=0.025, verbose=False):
    """Ramp p_contact bias to target_v in steps, n_contact grounded at 0.
    Halves the step on a convergence failure (down to a floor)."""
    ds.set_parameter(device=DEVICE, name="n_contact_bias", value=0.0)
    v_now = ds.get_parameter(device=DEVICE, name="p_contact_bias")
    step = abs(v_step)
    sign = 1.0 if target_v >= v_now else -1.0
    min_step = 1e-4
    while abs(v_now - target_v) > 1e-9:
        dv = sign * min(step, abs(target_v - v_now))
        v_try = v_now + dv
        ds.set_parameter(device=DEVICE, name="p_contact_bias", value=v_try)
        try:
            _solve()
        except ds.error as msg:
            if "Convergence failure" not in str(msg):
                raise
            ds.set_parameter(device=DEVICE, name="p_contact_bias", value=v_now)
            step *= 0.5
            if step < min_step:
                raise RuntimeError(
                    "ramp stalled at V={:.4f} (target {:.4f})".format(v_now, target_v))
            continue
        v_now = v_try
    return v_now


def jv_sweep(voltages, verbose=False):
    """Solve equilibrium once, then ramp through the given bias list in order,
    recording the terminal current at each. Returns list of (V, Jtot, Jn, Jp)."""
    out = []
    v_prev = 0.0
    for v in voltages:
        ramp_to(v, verbose=verbose)
        v_prev = v
        jtot, jn, jp = terminal_current("p_contact")
        out.append((v, jtot, jn, jp))
    return out


def carrier_profile():
    x = np.array(ds.get_node_model_values(device=DEVICE, region=REGION, name="x"))
    n = np.array(ds.get_node_model_values(device=DEVICE, region=REGION, name="Electrons"))
    p = np.array(ds.get_node_model_values(device=DEVICE, region=REGION, name="Holes"))
    return x, n, p


def main():
    print("[t] === Bipolar p-n diode (Si-like), staged 3-variable DD solve ===", flush=True)
    print("[t] Na=Nd={:.1e} m^-3, n_i={:.1e} m^-3, mu_n={:.3f} mu_p={:.3f} m^2/Vs, "
          "tau={:.1e} s".format(NA_M3, N_I_M3, MU_N, MU_P, TAU_N), flush=True)
    Vbi_analytic = V_T * np.log(NA_M3 * ND_M3 / N_I_M3**2)
    print("[t] analytic built-in potential Vbi = V_t*ln(Na*Nd/n_i^2) = {:.4f} V".format(
        Vbi_analytic), flush=True)

    # ---------- primary run: FD enhancement ON (the ITO-ready path) ----------
    build_diode(fd_enhancement=True)
    staged_equilibrium_solve(verbose=True)

    # equilibrium minority/majority sanity
    x0, n0, p0 = carrier_profile()
    i_p = np.argmin(np.abs(x0 - 0.25e-6))       # deep p-side
    i_n = np.argmin(np.abs(x0 - 1.75e-6))       # deep n-side
    print("[t] equilibrium: p-side(x=0.25um) n={:.3e} p={:.3e} ; "
          "n-side(x=1.75um) n={:.3e} p={:.3e} m^-3".format(
              n0[i_p], p0[i_p], n0[i_n], p0[i_n]), flush=True)
    p_minority_eq = p0[i_n]                      # equilibrium minority holes, n-side

    # ---------- GATE A: J-V sweep, forward and reverse ----------
    fwd_V = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("[t]", flush=True)
    print("[t] --- FORWARD bias J-V (p_contact swept +, n_contact grounded) ---", flush=True)
    print("[t]   V (V)      J_total (A/m^2)     J_n            J_p", flush=True)
    fwd = jv_sweep(fwd_V)
    for v, jt, jn, jp in fwd:
        print("[t]   {:+6.3f}    {:+14.6e}   {:+12.4e}  {:+12.4e}".format(v, jt, jn, jp),
              flush=True)

    # minority injection at forward bias (n-side holes) -- GATE B signature
    xf, nf, pf = carrier_profile()
    p_minority_fwd = pf[i_n]

    # reverse sweep (rebuild + re-equilibrate so we start clean at 0 V)
    build_diode(fd_enhancement=True)
    staged_equilibrium_solve(verbose=False)
    rev_V = [0.0, -0.1, -0.25, -0.5, -1.0, -2.0]
    print("[t]", flush=True)
    print("[t] --- REVERSE bias J-V ---", flush=True)
    print("[t]   V (V)      J_total (A/m^2)     J_n            J_p", flush=True)
    rev = jv_sweep(rev_V)
    for v, jt, jn, jp in rev:
        print("[t]   {:+6.3f}    {:+14.6e}   {:+12.4e}  {:+12.4e}".format(v, jt, jn, jp),
              flush=True)

    # ---------- verdicts ----------
    # Sign convention: bias on the p-contact. Forward (V>0) -> large positive-growing
    # current; reverse (V<0) -> small saturating current of the opposite sign.
    fwd_j = [jt for (v, jt, jn, jp) in fwd]
    rev_j = [jt for (v, jt, jn, jp) in rev]

    # Monotonic in magnitude with bias magnitude (forward current strictly grows).
    fwd_abs = [abs(j) for j in fwd_j]
    mono_fwd = all(fwd_abs[i + 1] >= fwd_abs[i] - 1e-30 for i in range(len(fwd_abs) - 1))
    rev_abs = [abs(j) for j in rev_j]
    mono_rev = all(rev_abs[i + 1] >= rev_abs[i] - 1e-30 for i in range(len(rev_abs) - 1))
    monotonic = mono_fwd and mono_rev

    # Rectifying: forward current at +0.6 V is orders of magnitude larger than the
    # reverse current at the same |V| (=0.6 V interpolated -> use -0.5 V point), and
    # the two have OPPOSITE sign.
    j_fwd_06 = fwd_j[fwd_V.index(0.6)]
    j_rev_05 = rev_j[rev_V.index(-0.5)]
    opposite_sign = (j_fwd_06 * j_rev_05) < 0.0
    big_ratio = abs(j_fwd_06) > 100.0 * max(abs(j_rev_05), 1e-300)
    rectifying = opposite_sign and big_ratio
    rect_ratio = abs(j_fwd_06) / max(abs(j_rev_05), 1e-300)

    # ~exponential forward rise: successive decades per ~60 mV (ideality ~1-2).
    # Check the slope d(ln|J|)/dV over the clean injection region (0.2 -> 0.5 V).
    v_lo, v_hi = 0.2, 0.5
    j_lo = abs(fwd_j[fwd_V.index(v_lo)])
    j_hi = abs(fwd_j[fwd_V.index(v_hi)])
    n_ideality = (v_hi - v_lo) / (V_T * np.log(j_hi / j_lo)) if j_hi > j_lo else float("inf")

    print("[t]", flush=True)
    print("[t] forward J(+0.6V)  = {:+.4e} A/m^2".format(j_fwd_06), flush=True)
    print("[t] reverse J(-0.5V)  = {:+.4e} A/m^2".format(j_rev_05), flush=True)
    print("[t] rectification ratio |J_fwd(0.6)|/|J_rev(0.5)| = {:.3e}".format(rect_ratio),
          flush=True)
    print("[t] extracted ideality factor (0.2-0.5 V) n = {:.3f}".format(n_ideality),
          flush=True)

    # ---------- GATE B: minority injection + Boltzmann-limit cross-check ----------
    inj_ratio = p_minority_fwd / max(p_minority_eq, 1e-300)
    print("[t]", flush=True)
    print("[t] GATE B minority injection (n-side holes, x=1.75um):", flush=True)
    print("[t]   equilibrium p = {:.3e} ; forward(+0.7V) p = {:.3e} m^-3 ; "
          "injection ratio = {:.3e}".format(
              p_minority_eq, p_minority_fwd, inj_ratio), flush=True)
    minority_injected = inj_ratio > 10.0

    # Boltzmann-limit cross-check. The accurate generalized-Einstein g (the fit in
    # physics_drift_diffusion) gives g(majority) ~ 1.012 at the primary doping
    # (N/N_dos ~ 0.036) -- a small but real ~1.2% FD enhancement, NOT a no-op -- so we
    # test the REDUCTION on a deliberately LOW-doped diode (Na=Nd=1e22 -> N/N_dos ~ 4e-4):
    # there the FD path must collapse onto standard Boltzmann SG. (The old code printed
    # 1.087 here from the degenerate-asymptote g, which was ~7% too high; audit F1.)
    def _g_einstein(x):   # matches the physics_drift_diffusion rational fit
        return 1.0 + (0.33717 * x + 0.14143 * x ** (4.0 / 3.0)) / (
            1.0 + 0.13356 * x ** (1.0 / 3.0) + 0.20570 * x ** (2.0 / 3.0))
    g_majority_primary = _g_einstein(NA_M3 / N_DOS_M3)
    NA_LOW = 1.0e22
    build_diode(fd_enhancement=True, na_m3=NA_LOW, nd_m3=NA_LOW)
    staged_equilibrium_solve(verbose=False)
    j_fd_low = jv_sweep([0.0, 0.3, 0.6])[-1][1]
    build_diode(fd_enhancement=False, na_m3=NA_LOW, nd_m3=NA_LOW)
    staged_equilibrium_solve(verbose=False)
    j_bz_low = jv_sweep([0.0, 0.3, 0.6])[-1][1]
    fd_vs_bz_rel = abs(j_fd_low - j_bz_low) / max(abs(j_bz_low), 1e-300)
    print("[t] FD g-factor is ACTIVE at primary doping: g(majority)={:.4f} "
          "(N/N_dos={:.4f})".format(g_majority_primary, NA_M3 / N_DOS_M3), flush=True)
    print("[t] Boltzmann-limit cross-check (low doping Na=Nd={:.0e}, g~1.004): "
          "J_fwd(0.6) FD-on={:+.4e}  FD-off={:+.4e}  rel-diff={:.3e}".format(
              NA_LOW, j_fd_low, j_bz_low, fd_vs_bz_rel), flush=True)
    boltzmann_reduces = fd_vs_bz_rel < 0.02      # within 2% -> FD path == standard SG

    ideality_sane = 1.0 <= n_ideality <= 2.0        # F6: a real diode, not eyeballed
    gate_a = monotonic and rectifying and ideality_sane
    gate_b = minority_injected and boltzmann_reduces
    overall = gate_a and gate_b

    print("[t]", flush=True)
    print("[t] GATE A (J-V monotonic & rectifying & ideality in [1,2]): monotonic={} "
          "rectifying={} ideality={:.3f}({}) -> {}".format(
        monotonic, rectifying, n_ideality, "ok" if ideality_sane else "BAD",
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[t] GATE B (minority injection & Boltzmann reduction): injected={} "
          "reduces={} -> {}".format(
              minority_injected, boltzmann_reduces, "PASS" if gate_b else "FAIL"), flush=True)
    print("[t] *** BIPOLAR DD: J-V monotonic={} rectifying={} -> {} ***".format(
        monotonic, rectifying, "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
