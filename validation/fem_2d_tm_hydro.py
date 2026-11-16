"""Nonlocal (hydrodynamic) graded conductor layer in the 2-D periodic TM FEM
(optics/fem_2d_tm_hydro.py): convergence tables, limits and an independent structured check.

Fixtures
  planar : the generic gated film: air | 6 nm oxB (n 2.00) | 6 nm graded film | 6 nm oxB | glass
           (n 1.45), p, 60 deg, 1200 nm. Film eps_inf 4.0, m 0.30 m_e, gamma 1.2e14 1/s, 'high_freq'
           beta, n0(z) = 3e26 + 1.2e27 exp(-z / 1 nm) on 11 samples from the top face (Re eps_T
           +2.7 .. -2.4).
  strip  : the generic strip cell: metal strips (eps -60+4i) 150 nm of 400 nm, 40 nm tall, on
           2 nm oxA (n 1.70) / 6 nm oxB / the graded film / 6 nm oxB / 2 nm oxA over an 80 nm mirror
           metal (eps -120+20i) on glass, TM ('x'), normal incidence, 1200 nm.

Gates
  A  planar graded film vs nonlocal_tmm (Richardson-extrapolated staircase): order-4 default
     < 1e-8, best row < 1e-9; table in order x conductor rows (samples aligned / not aligned)
  B  uniform film vs nonlocal_tmm's closed-form single layer: evanescent (8e26 m^-3) and
     propagating (Na-like at 1.05 / 1.2 / 1.5 wp) longitudinal waves, order 5 < 1e-8
  C  beta -> 0: the beta = 0 path is bit-identical to the local solve; |r - r_local| is linear in
     beta (slope 1 +- 0.02 over 1e-1 .. 1e-3) and matches the planar run at each beta (< 1e-8)
  D  energy: lossless graded film absorbs nothing (|A| < 1e-14, 1 - R - T < 1e-8) through a Re eps_T
     zero crossing and in the metallic regime; GNOR (D = 1e-3) matches the planar run and closes
  E  strip over the graded film: order x corner ladder converges monotonically; the conductor
     z-ladder moves r by < 5e-8; energy closes on every row (< 1e-6); the nonlocal correction vs
     the local solve is reported (time / DOFs / peak memory printed)
  F  a structured layer with planar physics (metal sheet containing a block of the same metal)
     over the graded film, 30 deg, vs the planar route: order 5 < 1e-8, order 4 < 5e-8 (on this
     stack order 3 / 4 / 5 / 6 give 5.9e-6 / 1.6e-8 / 1.3e-9 / 1.2e-10, the same without the block)
  G  INDEPENDENT structured check: an E-field solver written here (H(curl) Nedelec E plus a
     potential, NGSolve's Cartesian PML, its own background and extraction; it shares only the
     mesh builder and the material kernel): (G1) planar graded film vs nonlocal_tmm; (G2) strip
     over a UNIFORM film with the potential psi = div J (the optics.hydro_fem stabilised form) and
     with the electrochemical psi; (G3) strip over the GRADED film (electrochemical psi). Both
     solvers converged in order: agreement < 1e-6.
  H  optics.hydro_fem.hydro_layered_1d (free-standing uniform film, its 1-D coupled (E, J) FEM):
     the only existing hydro_fem configuration both can represent. Normal incidence (no
     longitudinal coupling) < 1e-6 in A; 30 deg < 5 % (hydro_fem documents ~0.5-3 % at 30 deg and up
     to ~20 % at 60 deg for that solver's scalar impedance boundary condition; measured 1.3 % at
     30 deg on the generic film, while the 2-D route equals nonlocal_tmm's A to 8 digits).

Run: python -m validation.fem_2d_tm_hydro
"""
import cmath
import dataclasses
import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation._harness import Harness  # noqa: E402

LAM = 1200e-9
HF = 2.00 ** 2
T_F = 6e-9
P_A, W_A = 400e-9, 150e-9
THREADS = 2


def _sq(z):
    w = cmath.sqrt(complex(z))
    return -w if (w.imag < 0 or (w.imag == 0 and w.real < 0)) else w


# ------------------------------------------------------------------------------------------------
# independent E-field periodic TM solver (gate G)
# ------------------------------------------------------------------------------------------------

def _film_coefficients(cond, omega, eps0, nt):
    """CF builder for the E-form: eps_T, Delta, kappa**2 [1/nm**2] (electrochemical psi) and
    beta**2/(Omega L0**2) [nm**2], wp**2/Omega (psi = div J) of a z-graded HydroConductor2D,
    tabulated on 4001 points (the kernel shared by design: eps_transverse, hydro_psi_per_charge)."""
    z = cond._z_nm()
    n = cond._n2d()[:, 0]

    def build(ng, z_lo):
        zc = np.union1d(z, np.linspace(0.0, z[-1], 4001))
        nc = np.interp(zc, z, n)
        wp2, beta = cond._wp2_beta(nc)
        kn = [z_lo + zc[0]] + list(z_lo + zc) + [z_lo + zc[-1]]
        wp2c = ng.BSpline(2, kn, [float(v) for v in wp2])(ng.y)
        b2c = ng.BSpline(2, kn, [float(v) for v in beta * beta])(ng.y)
        lay = nt.HydroLayer(float(cond.eps_inf), ng.sqrt(wp2c), float(cond.gamma), ng.sqrt(b2c),
                            cond.thickness_m)
        et = nt.eps_transverse(omega, lay)
        c = nt.hydro_psi_per_charge(omega, lay)
        om = omega * (omega + 1j * float(cond.gamma))
        return {"eps_t": et, "delta": et - float(cond.eps_inf), "kappa2": -1e-18 / (eps0 * c),
                "bn2": b2c / (om * 1e-18), "wp2_over_omega": wp2c / om}
    return build


def eform_solve(P, layers, film, cond, lam, *, theta=0.0, n1=1.0, n2=1.45, order=4, psi,
                mesh_spec=None, film_h_nm=0.6, pml_alpha=2.0):
    """r_H (H_y reflection at the top interface) of the cell with the hydrodynamic film `film`
    (a HydroConductor2D layer of `layers`) solved in the E-FIELD form:

        int curl E . curl v - k0**2 eps E . v  (+ potential coupling)  = k0**2 int (eps - eps1) E_bg . v

    E in a Bloch-periodic H(curl) (Nedelec) space, scattered with respect to the two-half-space
    background, NGSolve's Cartesian PML, eps = eps_T in the film. psi='electrochemical':
        - k0**2 int_film Delta grad psi . v ;  int_film kappa**2 psi w + Delta (E + grad psi) . grad w
          = - int_film Delta E_bg . grad w      (J = sigma (E + grad psi), J.n = 0 natural)
    psi='div_j' (optics.hydro_fem's stabilised form; uniform density only):
        - k0**2 b (wp**2/Omega) int grad chi . v ;  int -b grad chi . grad w + chi w + E . grad w
          = - int E_bg . grad w                 (chi ~ div J, b = beta**2/Omega)
    The mesh comes from fem_2d_tm.build_tm_cell with the film as a local placeholder whose
    samples set its rows (film_h_nm); r_H from the scattered H_y = (curl E)_y / k0**2 projected on
    two lines in the superstrate."""
    import ngsolve as ng
    from dynameta.constants import C_LIGHT, EPS0
    from dynameta.optics import nonlocal_tmm as nt
    from dynameta.optics.fem_2d_tm import GriddedEps2D, TMLayer, build_tm_cell

    lay2 = []
    for L in layers:
        if L.name == film:
            zl = np.linspace(0.0, L.thickness_m, int(round(L.thickness_m * 1e9 / film_h_nm)) + 1)
            lay2.append(TMLayer(L.name, L.thickness_m, GriddedEps2D(np.full(zl.size, 2.0 + 0.1j), zl)))
        else:
            lay2.append(L)
    cell = build_tm_cell(P, lay2, lambda_m=lam, n_super=n1, n_sub=n2, theta_deg=theta,
                         mesh_spec=mesh_spec)
    mesh = cell.mesh
    Pn = cell.period_nm
    omega = 2 * math.pi * C_LIGHT / lam
    k0 = 2 * math.pi / (lam * 1e9)
    kx = n1 * k0 * math.sin(math.radians(theta))
    e1, e2 = complex(n1) ** 2, complex(n2) ** 2
    kz1, kz2 = _sq(e1 * k0 ** 2 - kx ** 2), _sq(e2 * k0 ** 2 - kx ** 2)
    rF = (kz1 / e1 - kz2 / e2) / (kz1 / e1 + kz2 / e2)
    zr = cell.z_ref_nm
    A = cmath.exp(1j * kz1 * zr)
    iph = ng.exp(1j * kx * ng.x)
    up, dn, dn2 = ng.exp(1j * kz1 * ng.y), ng.exp(-1j * kz1 * ng.y), ng.exp(-1j * kz2 * ng.y)
    Hb = iph * ng.IfPos(ng.y, A * (dn + rF * up), A * (1 + rF) * dn2)
    dzH = iph * ng.IfPos(ng.y, A * (-1j * kz1 * dn + 1j * kz1 * rF * up),
                         A * (1 + rF) * (-1j * kz2) * dn2)
    ebg = ng.IfPos(ng.y, e1, e2)
    Ebg = ng.CoefficientFunction((-dzH / ebg, 1j * kx * Hb / ebg))     # E~ = J grad H / eps
    mesh.SetPML(ng.pml.Cartesian(mins=(-1e7, cell.z_intervals_nm["pml_bot"][1]),
                                 maxs=(1e7, cell.z_intervals_nm["pml_top"][0]),
                                 alpha=1j * pml_alpha), "pml_top|pml_bot")
    co = _film_coefficients(cond, omega, EPS0, nt)(ng, cell.regions[film].z_lo_nm)
    mats = list(mesh.GetMaterials())
    eps = ng.CoefficientFunction([co["eps_t"] if nm == film else
                                  ng.CoefficientFunction(cell.regions[nm].eps.const[0])
                                  for nm in mats])
    phase = cmath.exp(1j * kx * Pn)
    on = mesh.Materials(film)
    V = ng.Periodic(ng.HCurl(mesh, order=order, complex=True), phase=[phase])
    Q = ng.Periodic(ng.H1(mesh, order=order, complex=True, definedon=on), phase=[phase])
    used = set()
    for el in mesh.Elements(ng.VOL):
        if el.mat == film:
            used.update(Q.GetDofNrs(el))
    fd = Q.FreeDofs()
    for i in range(Q.ndof):
        if i not in used:
            fd.Clear(i)
    X = ng.FESpace([V, Q])
    (E, p), (v, w) = X.TnT()
    dxf = ng.dx(definedon=on, bonus_intorder=2)
    a = ng.BilinearForm(X)
    a += (ng.curl(E) * ng.curl(v) - k0 * k0 * eps * (E * v)) * ng.dx(bonus_intorder=2)
    f = ng.LinearForm(X)
    struct = "|".join(n for n in mats if cell.regions[n].role in ("layer", "block"))
    f += (k0 * k0 * (eps - e1) * (Ebg * v)) * ng.dx(definedon=mesh.Materials(struct),
                                                     bonus_intorder=2)
    if psi == "electrochemical":
        d = co["delta"]
        a += (-k0 * k0 * d * (ng.grad(p) * v)) * dxf
        a += (co["kappa2"] * p * w + d * (E * ng.grad(w)) + d * (ng.grad(p) * ng.grad(w))) * dxf
        f += (-d * (Ebg * ng.grad(w))) * dxf
    elif psi == "div_j":
        b, wo = co["bn2"], co["wp2_over_omega"]
        a += (-k0 * k0 * b * wo * (ng.grad(p) * v)) * dxf
        a += (-b * (ng.grad(p) * ng.grad(w)) + p * w + E * ng.grad(w)) * dxf
        f += (-(Ebg * ng.grad(w))) * dxf
    else:
        raise ValueError(psi)
    gfu = ng.GridFunction(X)
    with ng.TaskManager():
        a.Assemble()
        f.Assemble()
        gfu.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    Hs = -ng.curl(gfu.components[0]) / (k0 * k0)          # NGSolve curl = dEz/dx - dEx/dz
    xs = (np.arange(512) + 0.5) * Pn / 512

    def proj(z):
        vals = np.array([complex(np.ravel(Hs(mesh(float(x_), float(z))))[0]) for x_ in xs])
        return complex(np.mean(vals * np.exp(-1j * kx * xs)))
    z1, z2 = cell.probe_super_nm
    M = np.array([[cmath.exp(1j * kz1 * (z1 - zr)), cmath.exp(-1j * kz1 * (z1 - zr))],
                  [cmath.exp(1j * kz1 * (z2 - zr)), cmath.exp(-1j * kz1 * (z2 - zr))]])
    up_a, _dn_a = np.linalg.solve(M, [proj(z1), proj(z2)])
    return complex(up_a + rF * cmath.exp(2j * kz1 * zr)), int(X.ndof)


# ------------------------------------------------------------------------------------------------

def main():
    h = Harness("f2h", "2-D TM FEM: nonlocal graded conductor layer")
    try:
        import ngsolve  # noqa: F401
    except ImportError:
        h.require(False, "ngsolve not installed")
    from dynameta.constants import C_LIGHT
    from dynameta.materials import M_E
    from dynameta.optics import nonlocal_tmm as nt
    from dynameta.optics.fem_2d_tm import (GriddedEps2D, TMBlock, TMLayer, TMMeshSpec,
                                           solve_fem_2d_tm)
    from dynameta.optics.fem_2d_tm_hydro import HydroConductor2D, hydro_conductor_from_profile
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None
    warnings.simplefilter("ignore")                 # validity warnings: the fixture is u ~ 1.06

    KW = dict(m_eff_kg=0.30 * M_E, eps_inf=4.0, gamma=1.2e14)
    ZP = np.linspace(0.0, T_F, 11)
    NP = 3e26 + 1.2e27 * np.exp(-ZP / 1e-9)

    def om(lam):
        return 2 * math.pi * C_LIGHT / lam

    def planar(film, lam=LAM, th=60.0, top=None, bot=None):
        top = [nt.DielectricLayer(HF, 6e-9)] if top is None else top
        bot = [nt.DielectricLayer(HF, 6e-9)] if bot is None else bot
        return -nt.stack_rt(om(lam), top + list(film) + bot, pol="p", n_super=1.0, n_sub=1.45,
                            theta_rad=math.radians(th)).r

    def planar_graded(lam=LAM, th=60.0, kw=None, bs=1.0, D=0.0, top=None, bot=None, table=False):
        kw = KW if kw is None else kw
        rs = {}
        for N in ((256, 512, 1024, 2048, 4096, 8192) if table else (4096, 8192)):
            run = nt.hydro_layers_from_profile(ZP, NP, n_layers=N, D=D, **kw)
            if bs != 1.0:
                run = [dataclasses.replace(L, beta=L.beta * bs) for L in run]
            rs[N] = planar(run, lam, th, top, bot)
        rinf = rs[8192] + (rs[8192] - rs[4096]) / 3.0
        return (rinf, rs) if table else rinf

    def cond(**k):
        o = dict(KW, check_validity=False)
        o.update(k)
        return hydro_conductor_from_profile(ZP, NP, **o)

    def film(c):
        return [TMLayer("lo", 6e-9, HF), TMLayer("f", T_F, c), TMLayer("up", 6e-9, HF)]

    def solve(layers, lam=LAM, th=60.0, P=100e-9, order=4, spec=None):
        return solve_fem_2d_tm(P, layers, lambda_m=lam, theta_deg=th, n_super=1.0, n_sub=1.45,
                               polarization="x" if th == 0 else "p", order=order, mesh_spec=spec,
                               return_solution=True, num_threads=THREADS)

    # ---- A ------------------------------------------------------------------------------------
    r_ref, rs = planar_graded(table=True)
    Ns = sorted(rs)
    print("[f2h] A: planar staircase (nonlocal_tmm run, n_layers N), r_H = -r_planar", flush=True)
    for a_, b_ in zip(Ns, Ns[1:]):
        print("[f2h]   N {:5d}  r = {:.12f}{:+.12f}i  step {:.2e}".format(b_, rs[b_].real, rs[b_].imag,
                                                                       abs(rs[b_] - rs[a_])), flush=True)
    print("[f2h]   Richardson r_ref = {:.12f}{:+.12f}i".format(r_ref.real, r_ref.imag), flush=True)
    print("[f2h] A: FEM  aligned order rows |r - r_ref|   R+T+A-1   ndof   time[s]", flush=True)
    tabA = {}
    for align in (True, False):
        for order in (2, 3, 4, 5, 6):
            for hps in ((4.0, 1.0) if align else (4.0, 2.0, 0.5, 0.25)):
                sol = solve(film(cond(h_per_screening=hps, align_samples=align)), order=order)
                d = abs(sol.result.r - r_ref)
                tabA[(align, order, hps)] = d
                print("[f2h]      {!s:5} {:5d} {:4d}  {:.2e}  {:+.1e}  {:6d}  {:.2f}".format(
                    align, order, sol.diagnostics["hydro"]["f"]["rows"], d, sol.energy_residual,
                    sol.ndof, sol.assemble_solve_s), flush=True)
    loc = solve(film(cond(beta_scale=0.0))).result.r
    print("[f2h]   nonlocal effect in this planar fixture |r_nl - r_loc| = {:.3e}".format(
        abs(loc - r_ref)), flush=True)
    h.gate("A (planar graded vs nonlocal_tmm: default < 1e-8, best < 1e-9)",
           tabA[(True, 4, 1.0)] < 1e-8 and min(tabA.values()) < 1e-9,
           "default {:.1e}, best {:.1e}".format(tabA[(True, 4, 1.0)], min(tabA.values())))

    # ---- B ------------------------------------------------------------------------------------
    worstB = 0.0
    for name, n0, m, ei, f_wp, th in (("flm", 8e26, 0.30 * M_E, 4.0, None, 60.0),
                                      ("na", 2.65e28, M_E, 1.0, 1.05, 45.0),
                                      ("na", 2.65e28, M_E, 1.0, 1.2, 45.0),
                                      ("na", 2.65e28, M_E, 1.0, 1.5, 45.0)):
        lay = nt.hydro_layers_from_profile(np.array([0.0, T_F]), np.array([n0, n0]), m_eff_kg=m,
                                           eps_inf=ei, gamma=1.2e14)[0]
        lam = LAM if f_wp is None else 2 * math.pi * C_LIGHT / (f_wp * lay.wp)
        single = nt.HydroLayer(lay.eps_inf, lay.wp, lay.gamma, lay.beta, T_F)
        rr = planar([single], lam, th)
        kL = abs(cmath.sqrt(nt.kL_squared(om(lam), single)))
        c = HydroConductor2D(np.array([n0, n0]), np.array([0.0, T_F]), m, ei, 1.2e14,
                             check_validity=False)
        for order in (4, 5):
            sol = solve(film(c), lam, th, order=order)
            d = abs(sol.result.r - rr)
            if order == 5:
                worstB = max(worstB, d)
            print("[f2h] B: {:3s} w/wp {}  |k_L| d {:5.1f}  order {}  rows {:3d}  |dr| {:.2e}  "
                  "R+T+A-1 {:+.1e}".format(name, "-" if f_wp is None else f_wp, kL * T_F, order,
                                           sol.diagnostics["hydro"]["f"]["rows"], d,
                                           sol.energy_residual), flush=True)
    h.gate("B (uniform film vs closed form, order 5 < 1e-8)", worstB < 1e-8,
           "worst {:.1e}".format(worstB))

    # ---- C ------------------------------------------------------------------------------------
    lamC = 2500e-9
    c0 = cond(beta_scale=0.0)
    s0 = solve(film(c0), lamC)
    et = np.array([nt.eps_transverse(om(lamC), L) for L in c0._point_layers(c0._n2d()[:, 0])])
    s_loc = solve(film(GriddedEps2D(et, c0._z_nm() * 1e-9)), lamC)
    bit = s0.result.r == s_loc.result.r
    ds, okp = [], True
    for bs in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
        t0 = time.perf_counter()
        sol = solve(film(cond(beta_scale=bs)), lamC)
        dp = abs(sol.result.r - planar_graded(lamC, bs=bs))
        ds.append((bs, abs(sol.result.r - s0.result.r)))
        okp = okp and dp < 1e-8
        print("[f2h] C: beta_scale {:.0e}  |r - r_local| {:.4e}  |r - planar(same beta)| {:.1e}  "
              "rows {:5d}  ndof {:7d}  {:.1f} s".format(bs, ds[-1][1], dp,
                                                         sol.diagnostics["hydro"]["f"]["rows"],
                                                         sol.ndof, time.perf_counter() - t0), flush=True)
    slope = math.log(ds[0][1] / ds[-1][1]) / math.log(ds[0][0] / ds[-1][0])
    h.gate("C (beta -> 0: bit-identical local path; slope 1 +- 0.02; planar at each beta < 1e-8)",
           bit and abs(slope - 1.0) < 0.02 and okp, "bit-identical {}, slope {:.4f}".format(bit, slope))

    # ---- D ------------------------------------------------------------------------------------
    okD, msg = True, []
    for lamD in (1200e-9, 2500e-9):
        kw0 = dict(KW, gamma=0.0)
        sol = solve(film(cond(gamma=0.0)), lamD)
        res = sol.result
        dr = abs(res.r - planar_graded(lamD, kw=kw0))
        okD = okD and abs(res.per_region_absorption["f"]) < 1e-14 and abs(1 - res.R - res.T) < 1e-8 and dr < 1e-8
        msg.append("lossless {:.0f} nm: A {:.1e}, 1-R-T {:.1e}, |dr| {:.1e}".format(
            lamD * 1e9, res.per_region_absorption["f"], 1 - res.R - res.T, dr))
    sol = solve(film(cond(D=1e-3)))
    dr = abs(sol.result.r - planar_graded(D=1e-3))
    okD = okD and dr < 1e-8 and abs(sol.energy_residual) < 1e-8
    msg.append("GNOR D=1e-3: |dr| {:.1e}, R+T+A-1 {:.1e}, A_film {:.6f}".format(
        dr, -sol.energy_residual, sol.result.per_region_absorption["f"]))
    for m_ in msg:
        print("[f2h] D: " + m_, flush=True)
    h.gate("D (energy: lossless no gain, GNOR closes)", okD, "; ".join(msg[:1]))

    # ---- E ------------------------------------------------------------------------------------
    def strip_layers(c):
        return [TMLayer("mirror", 80e-9, -120 + 20j), TMLayer("lo_oxa", 2e-9, 1.70 ** 2),
                TMLayer("lo_oxb", 6e-9, HF), TMLayer("film", T_F, c),
                TMLayer("up_oxb", 6e-9, HF), TMLayer("up_oxa", 2e-9, 1.70 ** 2),
                TMLayer("grating", 40e-9, 1.0 + 0j,
                        (TMBlock(P_A / 2 - W_A / 2, P_A / 2 + W_A / 2, -60 + 4j, "strip"),))]

    def strip(c, order, lev):
        t0 = time.perf_counter()
        sol = solve_fem_2d_tm(P_A, strip_layers(c), lambda_m=LAM, n_super=1.0, n_sub=1.45,
                              order=order, mesh_spec=TMMeshSpec(corner_levels=lev),
                              return_solution=True, num_threads=THREADS)
        return sol, time.perf_counter() - t0

    print("[f2h] E: strip   model    order lev | r                          R+T+A-1   A_film   "
          "ndof    time[s] peak[MB]", flush=True)
    ladder = {}
    for model, bs in (("local", 0.0), ("nonloc", 1.0)):
        for order, lev in ((3, 6), (4, 6), (5, 8), (6, 10), (7, 10)):
            sol, dt = strip(cond(beta_scale=bs), order, lev)
            res = sol.result
            ladder[(model, order)] = sol
            pk = (getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 1e6
                  if proc is not None else float("nan"))
            print("[f2h]           {:7s} {:5d} {:3d} | {:.8f}{:+.8f}i  {:+.1e}  {:.5f}  {:7d}  {:6.2f}  {:6.0f}"
                  .format(model, order, lev, res.r.real, res.r.imag, sol.energy_residual,
                          res.per_region_absorption["film"], sol.ndof, dt, pk), flush=True)
    fin = ladder[("nonloc", 7)].result.r
    dl = [abs(ladder[("nonloc", o)].result.r - fin) for o in (3, 4, 5, 6)]
    loc_fin = ladder[("local", 7)].result.r
    dloc = [abs(ladder[("local", o)].result.r - loc_fin) for o in (3, 4, 5, 6)]
    print("[f2h]   nonlocal |r - r(7,10)| = " + ", ".join("{:.1e}".format(x) for x in dl), flush=True)
    print("[f2h]   local    |r - r(7,10)| = " + ", ".join("{:.1e}".format(x) for x in dloc)
          + "  (the local field is near-singular where Re eps_T = 0 inside the film)", flush=True)
    corr = abs(fin - loc_fin)
    print("[f2h]   NONLOCAL CORRECTION |r_nl - r_loc| = {:.4f} (|r_loc| {:.4f}); film absorption "
          "{:.4f} (nonlocal) vs {:.4f} (local); R {:.4f} vs {:.4f}".format(
              corr, abs(loc_fin), ladder[("nonloc", 7)].result.per_region_absorption["film"],
              ladder[("local", 7)].result.per_region_absorption["film"],
              ladder[("nonloc", 7)].result.R, ladder[("local", 7)].result.R), flush=True)
    rz = []
    for hz in (1.0e-9, 0.5e-9, 0.25e-9, 0.125e-9):
        sol = solve_fem_2d_tm(P_A, strip_layers(cond(h_z_m=hz)), lambda_m=LAM, n_super=1.0,
                              n_sub=1.45, order=4, return_solution=True, num_threads=THREADS)
        rz.append(sol.result.r)
        print("[f2h] E: conductor h_z {:.3f} nm  rows {:3d}  r = {:.10f}{:+.10f}i  ndof {}".format(
            hz * 1e9, sol.diagnostics["hydro"]["film"]["rows"], sol.result.r.real,
            sol.result.r.imag, sol.ndof), flush=True)
    dz = max(abs(r_ - rz[-1]) for r_ in rz)
    worstE = max(abs(s.energy_residual) for s in ladder.values())
    h.gate("E (strip: monotone order ladder, z-ladder < 5e-8, energy < 1e-6)",
           all(a_ > b_ for a_, b_ in zip(dl, dl[1:])) and dl[-1] < 1e-6 and dz < 5e-8 and worstE < 1e-6,
           "ladder {}, z {:.1e}, energy {:.1e}, correction {:.3f}".format(
               ", ".join("{:.1e}".format(x) for x in dl), dz, worstE, corr))

    # ---- F ------------------------------------------------------------------------------------
    em = -60 + 4j
    layF = [TMLayer("mirror", 80e-9, -120 + 20j), TMLayer("lo", 6e-9, HF), TMLayer("f", T_F, cond()),
            TMLayer("up", 6e-9, HF), TMLayer("sheet", 20e-9, em, (TMBlock(100e-9, 250e-9, em, "m_a"),))]
    rF_ref = planar_graded(th=30.0, top=[nt.DielectricLayer(em, 20e-9), nt.DielectricLayer(HF, 6e-9)],
                           bot=[nt.DielectricLayer(HF, 6e-9), nt.DielectricLayer(-120 + 20j, 80e-9)])
    # corner_mode='all': the default rule does not grade the edges of a block of the sheet's own metal
    # (no permittivity jump across the edge: a flat interface), and this gate exercises the corner-graded
    # structured mesh
    dF = [abs(solve(layF, LAM, 30.0, P=P_A, order=o, spec=TMMeshSpec(corner_mode="all")).result.r - rF_ref)
          for o in (4, 5)]
    h.gate("F (structured mesh, planar physics, vs planar: order 5 < 1e-8, order 4 < 5e-8)",
           dF[1] < 1e-8 and dF[0] < 5e-8,
           "order 4 / 5: {:.1e} / {:.1e}".format(*dF))

    # ---- G ------------------------------------------------------------------------------------
    c = cond()
    # film rows ALIGNED with the profile samples (0.6 nm apart): kinks of the piecewise-linear density
    # inside E-form elements cut its convergence to ~first order (0.25 nm rows: 4.9e-7; 0.0625 nm: 2.8e-8)
    rg1, _ = eform_solve(100e-9, film(c), "f", c, LAM, theta=60.0, order=5, psi="electrochemical",
                         film_h_nm=0.3)
    dG1 = abs(rg1 - r_ref)
    print("[f2h] G1: E-form planar graded vs nonlocal_tmm |dr| = {:.2e}".format(dG1), flush=True)
    cu = HydroConductor2D(np.array([8e26, 8e26]), np.array([0.0, T_F]), 0.30 * M_E, 4.0, 1.2e14,
                          check_validity=False)
    resG = {}
    for kind, cc, psis in (("uniform", cu, ("div_j", "electrochemical")),
                           ("graded", c, ("electrochemical",))):
        for order, lev in ((4, 6), (5, 8), (6, 10)):
            if kind == "graded":                                  # r_H of the H-form
                rh = -ladder[("nonloc", order)].result.r
            else:
                rh = -strip(cc, order, lev)[0].result.r
            for psi in psis:
                t0 = time.perf_counter()
                re_, nd = eform_solve(P_A, strip_layers(cc), "film", cc, LAM, order=order, psi=psi,
                                      mesh_spec=TMMeshSpec(corner_levels=lev))
                resG[(kind, psi, order)] = (re_, rh)
                print("[f2h] G: strip {:7s} psi={:15s} order {} | H-form r_H {:.8f}{:+.8f}i  E-form "
                      "{:.8f}{:+.8f}i  |d| {:.2e}  (E-form ndof {}, {:.1f} s)".format(
                          kind, psi, order, rh.real, rh.imag, re_.real, re_.imag, abs(re_ - rh), nd,
                          time.perf_counter() - t0), flush=True)
    dG = {k: abs(v[0] - v[1]) for k, v in resG.items() if k[2] == 6}
    nonloc_u = abs(resG[("uniform", "div_j", 6)][1] - (-strip(HydroConductor2D(
        np.array([8e26, 8e26]), np.array([0.0, T_F]), 0.30 * M_E, 4.0, 1.2e14, beta_scale=0.0,
        check_validity=False), 6, 10)[0].result.r))
    print("[f2h]   nonlocal correction in the uniform-film strip |r_nl - r_loc| = {:.3e}".format(
        nonloc_u), flush=True)
    h.gate("G (independent E-form: planar < 1e-8; strip uniform (div J and electrochemical) and "
           "graded at order 6 < 1e-6)", dG1 < 1e-8 and max(dG.values()) < 1e-6,
           "planar {:.1e}; strip ".format(dG1) + ", ".join("{}/{} {:.1e}".format(k[0], k[1], v)
                                                         for k, v in dG.items()))

    # ---- H ------------------------------------------------------------------------------------
    from dynameta.optics.hydro_fem import HydroParams, hydro_layered_1d
    lay = nt.hydro_layers_from_profile(np.array([0.0, T_F]), np.array([8e26, 8e26]), **KW)[0]
    hp = HydroParams(eps_inf=lay.eps_inf, wp=lay.wp, gamma=lay.gamma, beta=lay.beta)
    okH, msgH = True, []
    for th in (0.0, 30.0):
        r1d = hydro_layered_1d(om(LAM), hp, T_F * 1e9, theta_rad=math.radians(th), buffer_nm=150.0,
                               order=4, metal_cells=200)
        sol = solve_fem_2d_tm(100e-9, [TMLayer("f", T_F, cu)], lambda_m=LAM, theta_deg=th,
                              polarization="x" if th == 0 else "p", order=5, return_solution=True,
                              num_threads=THREADS)
        a2 = sol.result.per_region_absorption["f"]
        tm = nt.stack_rt(om(LAM), [nt.HydroLayer(lay.eps_inf, lay.wp, lay.gamma, lay.beta, T_F)],
                         pol="p", theta_rad=math.radians(th))
        rel = abs(r1d.A_volume - a2) / a2
        okH = okH and (rel < 1e-6 if th == 0 else rel < 0.05)
        msgH.append("{:.0f} deg: A 2-D {:.8f}, hydro_layered_1d A_volume {:.8f} (rel {:.1e}), "
                    "nonlocal_tmm A {:.8f}".format(th, a2, r1d.A_volume, rel, tm.A))
    for m_ in msgH:
        print("[f2h] H: " + m_, flush=True)
    h.gate("H (hydro_fem 1-D film: normal < 1e-6 in A; 30 deg < 5 %, its documented oblique "
           "impedance-BC error)", okH,
           "; ".join(msgH))
    return h.summary()


if __name__ == "__main__":
    raise SystemExit(main())
