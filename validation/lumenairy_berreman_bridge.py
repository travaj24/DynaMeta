"""Lumenairy Berreman 4x4 backend bridge (roadmap v0.5 A4) vs hand-derived ANALYTIC oracles.

The Berreman backend is the PLANAR anisotropic tier (uniform LC/MO/EO/birefringent stacks).
It is validated here against MACHINE-PRECISION analytic oracles -- strictly stronger than the
~2%-accurate 3-D FEM the audit flagged, and needing only lumenairy + tmm + numpy (no NGSolve):

GATE A (isotropic == TMM): a 3-layer lossy stack at normal + 30 deg s/p through the
        BerremanLayeredSolver vs the in-repo TmmLayeredSolver -- R/T/r/t (incl. the p-basis
        conversion reused from the RCWA bridge) match < 1e-9. Exercises the WHOLE 4x4 cascade
        in its scalar limit, where Berreman must reduce to the transfer-matrix coating model.
GATE B (uniaxial waveplate, exact decoupling): a single uniaxial slab, optic axis x at normal
        incidence, DECOUPLES into two independent isotropic problems -- x sees n_e, y sees n_o.
        The full Jones (berreman_jones_1d) must equal the per-axis scalar TMM (Jt[0,0]=t(n_e),
        Jt[1,1]=t(n_o), Jr likewise), with ZERO cross-pol; the two channels must GENUINELY
        differ (n_e != n_o), and the backend's eps_tensor_cell co-pol r must match the diagonal.
        WRONG-MODEL GUARD: an isotropic model (n_e==n_o) would also pass a single-channel check,
        so the gate asserts the channels differ by the birefringence.
GATE C (gyrotropic Faraday, off-diagonal): a magneto-optic gyrotropic slab
        [[e, i g, 0], [-i g, e, 0], [0, 0, e]] at normal incidence decouples in the CIRCULAR
        basis (indices n_pm = sqrt(e -/+ g)). The full transmission Jones must equal the analytic
        two-circular-TMM construction < 1e-9, energy closes (lossless A ~ 0), AND the cross-pol
        is NONZERO -- the WRONG diagonal/g=0 model gives zero Faraday rotation and FAILS this.
        This is the case the RCWA in-plane tensor path explicitly REJECTS; Berreman is its only
        rigorous home.
GATE D (lossy raw-eps split, no T>1): a lossy isotropic slab matches the complex-angle TMM and
        keeps T < 1, 0 <= A <= 1; a lossy uniaxial slab stays energy-physical. ADVERSARIAL: the
        SAME stack with Im(eps) FLIPPED (gain) gives T > 1 / A < 0 -- so the loss gate is
        meaningful and the raw-eps forward/backward split is sign-correct (the standalone
        Berreman oracle that conjugates eps never validated loss).
GATE E (real DynaMeta device path + closure): LiquidCrystalModel and MagnetoOpticModel emit the
        actual (3,3) tensors; routed through the backend they REPRODUCE the GATE-B/C analytic.
        Per-layer absorption closes NON-trivially on a lossy uniaxial slab
        (sum_layers A_i == 1 - R - T < 1e-9), and the lossless gyrotropic film takes A ~ 0.
GATE F (Design-level seam + dispatch boundary + conical): make_lumenairy_berreman_solver +
        design_to_berreman_layers on a real Design (uniform-tensor EpsField override) match the
        LayeredStack path; conical incidence (azimuth != 0, which the PMM bridge cannot do)
        solves and conserves energy; a PATTERNED layer (inclusion / eps_cell) RAISES pointing at
        the RCWA backend -- the planar-tier scope contract.
GATE G (A1 far-field consolidation + C2 OOP-oblique absorption): the bridge now sources the far
        field from ONE BerremanStack CLASS solve (+ jones_transmission()), retiring the old
        functional-berreman_jones_1d-plus-second-retain_internal-solve pattern. The class-solved
        r/t/R/T must equal the functional far field post-processed IDENTICALLY (p-basis + lab-row)
        bit-for-bit (< 1e-13) on an in-plane tensor (p-pol, exercising the p-basis factors) AND a
        conical-rotated case (the _rotate_layers_conical + phi=0 equivalent problem). AND, with
        lumenairy 5.22 reconstructing the generalized-cascade internals (AUDIT C2), a tilted-
        director tensor (out-of-plane, lossy) at OBLIQUE conical incidence now closes its per-layer
        absorption budget (|sum A_i - (1 - R - T)| < 1e-6, A_independent != None) -- previously the
        OOP-tensor-at-oblique case gracefully degraded to A_independent = None.

Honest SKIP (exit 0 + banner) when lumenairy is not importable.

Run: python -m validation.lumenairy_berreman_bridge
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.eps_field import EpsField
from dynameta.core.layered import LayeredSlab, LayeredStack
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

LAM = 1.55e-6
PER = 400e-9


def _scalar_tmm(n_idx, d_m, n_sup, n_sub, lam, theta_deg=0.0, pol="s"):
    """Single isotropic slab (index n_idx, thickness d_m) between n_sup|n_sub: the tmm result
    dict (r, t, R, T). The per-axis / per-circular-eigenmode oracle for the anisotropic gates."""
    import tmm
    n_list = [complex(n_sup), complex(n_idx), complex(n_sub)]
    d_list = [np.inf, float(d_m) * 1e9, np.inf]
    return tmm.coh_tmm(pol, n_list, d_list, np.radians(theta_deg), float(lam) * 1e9)


def _tensor_stack(eps_t, d_m, n_sup, n_sub):
    """A 1-slab LayeredStack carrying a uniform (3,3) tensor (broadcast to a (1,1,3,3) cell)."""
    return LayeredStack(complex(n_sup), complex(n_sub),
                        [LayeredSlab(float(d_m),
                                     eps_tensor_cell=np.broadcast_to(
                                         np.asarray(eps_t, complex), (1, 1, 3, 3)).copy())],
                        period_x_m=PER, period_y_m=PER)


def _registry():
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("hi", ConstantOptical(complex(4.0, 0.3))))
    reg.add(Material("lo", ConstantOptical(complex(2.1, 0.0))))
    reg.add(Material("pillar", ConstantOptical(complex(4.0, 0.0))))
    return reg


def _design(layers, *, pol="y", theta=0.0, phi=0.0, sub="glass"):
    return Design(name="brg-ber", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material=sub),
                  electrodes=[], materials=_registry(),
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta,
                                      azimuth_deg=phi))


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[lbb] *** SKIP: lumenairy not installed -- Berreman bridge gates not run ***",
              flush=True)
        return True
    import lumenairy as lum
    from dynameta.optics.lumenairy_bridge import (BerremanLayeredSolver,
                                                  design_to_berreman_layers,
                                                  make_lumenairy_berreman_solver)
    from dynameta.optics.tmm_reference import TmmLayeredSolver

    print("[lbb] === Lumenairy Berreman bridge vs analytic oracles ===", flush=True)
    ok = True
    n_sup, n_sub = 1.0 + 0j, 1.5 + 0j

    # ---- GATE A: isotropic multilayer == TMM (normal + 30 deg s/p) ----
    slabs = [LayeredSlab(120e-9, eps=complex(4.0, 0.3)),
             LayeredSlab(200e-9, eps=complex(2.1, 0.0))]
    stk = LayeredStack(n_sup, n_sub, slabs)
    worst = 0.0
    for pol, th in (("y", 0.0), ("y", 30.0), ("p", 30.0)):
        opt = OpticalSpec(polarization=pol, incidence_angle_deg=th)
        r_b = BerremanLayeredSolver().solve(stk, LAM, opt)
        r_t = TmmLayeredSolver().solve(stk, LAM, opt)
        worst = max(worst, abs(r_b.R - r_t.R), abs(r_b.T - r_t.T),
                    abs(r_b.r - r_t.r), abs(r_b.t - r_t.t))
    g_a = bool(worst < 1e-9)
    ok = ok and g_a
    print("[lbb] GATE A: isotropic vs TMM (normal + 30deg s/p): worst |d| = {:.2e} -> {}".format(
        worst, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: uniaxial waveplate decouples into per-axis scalar TMM ----
    n_o, n_e, d = 1.50, 1.74, 220e-9
    eps_uni = np.diag([n_e ** 2, n_o ** 2, n_o ** 2]).astype(complex)   # optic axis along x
    R2, T2, Jr, Jt = lum.berreman_jones_1d([(eps_uni, d)], n_sub, n_sup, LAM)
    tx = _scalar_tmm(n_e, d, n_sup, n_sub, LAM)
    ty = _scalar_tmm(n_o, d, n_sup, n_sub, LAM)
    diag_err = max(abs(Jt[0, 0] - tx["t"]), abs(Jt[1, 1] - ty["t"]),
                   abs(Jr[0, 0] - tx["r"]), abs(Jr[1, 1] - ty["r"]))
    cross = max(abs(Jt[0, 1]), abs(Jt[1, 0]), abs(Jr[0, 1]), abs(Jr[1, 0]))
    birefringent = abs(Jt[0, 0] - Jt[1, 1]) > 1e-2            # channels genuinely differ
    # backend eps_tensor_cell co-pol r matches the full-Jones diagonal
    stk_u = _tensor_stack(eps_uni, d, n_sup, n_sub)
    rb_x = BerremanLayeredSolver().solve(stk_u, LAM, OpticalSpec(polarization="x",
                                                                incidence_angle_deg=0.0))
    rb_y = BerremanLayeredSolver().solve(stk_u, LAM, OpticalSpec(polarization="y",
                                                                incidence_angle_deg=0.0))
    backend_err = max(abs(rb_x.r - Jr[0, 0]), abs(rb_y.r - Jr[1, 1]))
    g_b = bool(diag_err < 1e-9 and cross < 1e-12 and birefringent and backend_err < 1e-9)
    ok = ok and g_b
    print("[lbb] GATE B: uniaxial decouples (per-axis TMM {:.1e}, cross-pol {:.1e}, birefringent "
          "{}, backend {:.1e}) -> {}".format(diag_err, cross, birefringent, backend_err,
                                             "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: gyrotropic Faraday == analytic circular two-TMM ----
    e0, g0, dF = 4.0, 0.40, 300e-9
    eps_gyro = np.array([[e0, 1j * g0, 0], [-1j * g0, e0, 0], [0, 0, e0]], dtype=complex)
    Rg, Tg, Jrg, Jtg = lum.berreman_jones_1d([(eps_gyro, dF)], n_sub, n_sup, LAM)
    n_minus = np.sqrt(e0 - g0)               # index of the (1, i) circular eigenmode
    n_plus = np.sqrt(e0 + g0)                # index of the (1, -i) circular eigenmode
    t_m = _scalar_tmm(n_minus, dF, n_sup, n_sub, LAM)["t"]
    t_p = _scalar_tmm(n_plus, dF, n_sup, n_sub, LAM)["t"]
    Ex = 0.5 * (t_m + t_p)                                    # x-incident transmitted lab field
    # the +ig (DynaMeta MagnetoOpticModel) convention prediction -- (1,i) eigenmode carries n_minus,
    # so E_y = +0.5j(t_minus - t_plus). NO sign-twin / min(): the gate must be SIGN-sensitive so a
    # wrong-HANDEDNESS tensor (eps_xy = -ig) fails it (a min-over-+/-Ey twin would pass both).
    analytic = np.array([Ex, 0.5j * (t_m - t_p)])
    col0 = np.asarray(Jtg)[:, 0]
    faraday_err = float(np.max(np.abs(col0 - analytic)))     # SIGN-sensitive match
    hand_sign = float(np.sign((col0[1] / col0[0]).imag))     # rotation SENSE: -1 for the +ig convention
    cross_gyro = abs(Jtg[1, 0])                              # NONZERO Faraday rotation
    g0_lossless = abs(Rg[0] + Tg[0] - 1.0)
    # WRONG-MODEL GUARDS: (a) g=0 (drop the off-diagonal) gives ZERO cross-pol; (b) the SIGN-REVERSED
    # gyrotropic tensor (wrong handedness, eps_xy = -ig) must FAIL the analytic match AND flip the sense.
    _, _, _, Jt0 = lum.berreman_jones_1d([(e0 * np.eye(3, dtype=complex), dF)], n_sub, n_sup, LAM)
    wrong_cross = abs(Jt0[1, 0])
    eps_wrong = np.array([[e0, -1j * g0, 0], [1j * g0, e0, 0], [0, 0, e0]], dtype=complex)
    _, _, _, Jtw = lum.berreman_jones_1d([(eps_wrong, dF)], n_sub, n_sup, LAM)
    wrong_hand_err = float(np.max(np.abs(np.asarray(Jtw)[:, 0] - analytic)))   # must be LARGE
    g_c = bool(faraday_err < 1e-9 and hand_sign < 0.0 and cross_gyro > 1e-2 and g0_lossless < 1e-9
               and wrong_cross < 1e-14 and wrong_hand_err > 1e-2)
    ok = ok and g_c
    print("[lbb] GATE C: gyrotropic Faraday vs circular-TMM {:.1e} (sense {:+.0f}), cross-pol {:.3f} "
          "(g=0 {:.1e}, wrong-handedness err {:.2f}), lossless {:.1e} -> {}".format(
              faraday_err, hand_sign, cross_gyro, wrong_cross, wrong_hand_err, g0_lossless,
              "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE D: lossy raw-eps split, no T>1; gain (flipped Im) violates ----
    eps_lossy = complex(2.2, 0.4)
    dL = 1.6e-6
    rb = BerremanLayeredSolver().solve(LayeredStack(n_sup, n_sub, [LayeredSlab(dL, eps=eps_lossy)]),
                                       LAM, OpticalSpec(polarization="y", incidence_angle_deg=25.0))
    tt = _scalar_tmm(np.sqrt(eps_lossy), dL, n_sup, n_sub, LAM, theta_deg=25.0, pol="s")
    iso_lossy_err = max(abs(rb.R - tt["R"]), abs(rb.T - tt["T"]))
    physical = bool(rb.T < 1.0 and rb.R < 1.0 and -1e-12 <= rb.A <= 1.0 + 1e-9)
    # lossy uniaxial stays physical
    eps_uni_lossy = np.diag([complex(n_e ** 2, 0.3), complex(n_o ** 2, 0.1),
                             complex(n_o ** 2, 0.1)])
    Ru, Tu, _, _ = lum.berreman_jones_1d([(eps_uni_lossy, dL)], n_sub, n_sup, LAM)
    uni_phys = bool(Tu[0] < 1.0 and Tu[1] < 1.0 and (1.0 - Ru[0] - Tu[0]) > -1e-9)
    # ADVERSARIAL: flip Im(eps) -> gain -> T > 1 (the gate is meaningful, the split is sign-correct)
    Rga, Tga, _, _ = lum.berreman_jones_1d([(np.conj(eps_lossy) * np.eye(3, dtype=complex), dL)],
                                           n_sub, n_sup, LAM)
    gain_violates = bool(Tga[0] > 1.0 + 1e-3)
    g_d = bool(iso_lossy_err < 1e-9 and physical and uni_phys and gain_violates)
    ok = ok and g_d
    print("[lbb] GATE D: lossy vs TMM {:.1e}, T<1 & 0<=A<=1 {}, uniaxial physical {}, gain-flip "
          "T={:.2f}>1 {} -> {}".format(iso_lossy_err, physical, uni_phys, Tga[0], gain_violates,
                                       "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: real DynaMeta device tensors + non-trivial energy closure ----
    from dynameta.core.effects import LiquidCrystalModel, MagnetoOpticModel
    lc = LiquidCrystalModel(n_o=1.50, n_e=1.74)
    eps_lc = np.asarray(lc.eps({"director_angle_rad": 0.0}, LAM), dtype=complex)
    stk_lc = _tensor_stack(eps_lc, d, n_sup, n_sub)
    e_lc = 0.0
    for pol, n_idx in (("x", n_e), ("y", n_o)):
        rdev = BerremanLayeredSolver().solve(stk_lc, LAM, OpticalSpec(polarization=pol,
                                                                     incidence_angle_deg=0.0))
        ts = _scalar_tmm(n_idx, d, n_sup, n_sub, LAM)
        e_lc = max(e_lc, abs(rdev.R - ts["R"]), abs(rdev.T - ts["T"]))
    mo = MagnetoOpticModel(eps_r=4.0, g=0.40)
    eps_mo = np.asarray(mo.eps({"magnetization": 1.0}, LAM), dtype=complex)
    Rm, Tm, _, Jtm = lum.berreman_jones_1d([(eps_mo, dF)], n_sub, n_sup, LAM)
    # the MO model emits eps_xy = +ig (magneto.py), so it must match the SAME sign-sensitive analytic
    dev_faraday = float(np.max(np.abs(np.asarray(Jtm)[:, 0] - analytic)))
    # NON-trivial absorption closure on a LOSSY uniaxial slab: sum_layers A_i == 1 - R - T
    st = lum.BerremanStack(n_substrate=complex(n_sub), n_superstrate=complex(n_sup))
    st.add_layer(dL, eps=eps_uni_lossy)
    st.add_layer(150e-9, eps=complex(2.1, 0.0))
    st.set_source(LAM, theta=0.0)
    Rc, Tc, _ = st.solve(retain_internal=True)
    la = np.asarray(st.layer_absorption())
    closure = abs(float(np.sum(la[:, 0])) - (1.0 - Rc[0] - Tc[0]))
    nontrivial = float(np.sum(la[:, 0])) > 1e-3
    g_e = bool(e_lc < 1e-9 and dev_faraday < 1e-9 and closure < 1e-9 and nontrivial)
    ok = ok and g_e
    print("[lbb] GATE E: LC device {:.1e}, MO device Faraday {:.1e}, lossy closure {:.1e} "
          "(A={:.3f} nontrivial {}) -> {}".format(e_lc, dev_faraday, closure,
                                                  float(np.sum(la[:, 0])), nontrivial,
                                                  "PASS" if g_e else "FAIL"), flush=True)

    # ---- GATE F: Design-level seam + conical + dispatch boundary ----
    d_film = _design([Layer("film", d, "lo")], pol="x")
    eps_by_region = {"film": EpsField(tensor=eps_uni)}
    solver = make_lumenairy_berreman_solver()
    r_design = solver(d_film, None, eps_by_region, LAM, n_sup, n_sub)
    # the SAME uniform-tensor film through the LayeredStack path
    r_stack = BerremanLayeredSolver().solve(stk_u, LAM, OpticalSpec(polarization="x",
                                                                   incidence_angle_deg=0.0))
    design_seam_err = max(abs(r_design.R - r_stack.R), abs(r_design.T - r_stack.T),
                          abs(r_design.r - r_stack.r))
    # conical (azimuth != 0): the PMM bridge raises here; Berreman solves. DISCRIMINATOR: the full
    # 2x2 Jones is rotationally COVARIANT -- Jr(eps; theta, phi) == Rot(phi) Jr(Rz(-phi) eps
    # Rz(-phi)^T; theta, 0) Rot(-phi). A phi-IGNORING model (Jones independent of phi) FAILS this
    # (energy closure alone is phi-invariant and proves nothing about conical physics).
    th_c, phi_c = np.radians(20.0), np.radians(30.0)

    def _Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _Rot2(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]], dtype=complex)

    _, _, Jr_c, _ = lum.berreman_jones_1d([(eps_uni, d)], n_sub, n_sup, LAM, angle=th_c, phi=phi_c)
    eps_rot = _Rz(-phi_c) @ eps_uni @ _Rz(-phi_c).T
    _, _, Jr_r, _ = lum.berreman_jones_1d([(eps_rot, d)], n_sub, n_sup, LAM, angle=th_c, phi=0.0)
    covar_err = float(np.max(np.abs(np.asarray(Jr_c)
                                    - _Rot2(phi_c) @ np.asarray(Jr_r) @ _Rot2(-phi_c))))
    _, _, Jr_p0, _ = lum.berreman_jones_1d([(eps_uni, d)], n_sub, n_sup, LAM, angle=th_c, phi=0.0)
    phi_matters = float(np.max(np.abs(np.asarray(Jr_c) - np.asarray(Jr_p0))))   # phi genuinely used
    # audit 8.1-6 / 8.2 step 4: the DESIGN-LEVEL conical solve now SOLVES via the rotated
    # equivalent in-plane problem (rotational covariance -- planar tier only), so its r is
    # keyed to the FEM's rotated s/p eigen-polarizations. END-TO-END ORACLE: the same r via
    # lumenairy's NATIVE conical engine (kx, ky source -- a different code path) with its
    # lab-basis Jones explicitly rotated into the s/p frame, Rot(-phi) Jr_c Rot(phi). The
    # covariance leg above independently pins that identity at 1e-9; a wrong rotation sign,
    # a phi-ignoring bridge, or the old C4-2 s/p-mixture extraction all break the match
    # (pre-fix probe: 32% off the s-pol truth at theta=30, phi=45).
    # LOW-SYMMETRY biaxial tensor for the end-to-end leg: eps_uni's optic axis lies along x
    # (mirror-symmetric about the phi=0 plane of incidence), which makes the CO-polarized r
    # EVEN in phi -- a rotation-sign bug survives it (probed: a deliberate -phi bridge
    # rotation passed a co-pol match on eps_uni). Tilting the principal frame by Rz(25deg)
    # Ry(35deg) breaks every mirror, so the sign flip moves r at the 1e-3 level.
    a_z, b_y = np.radians(25.0), np.radians(35.0)
    _Ry = np.array([[np.cos(b_y), 0.0, np.sin(b_y)], [0.0, 1.0, 0.0],
                    [-np.sin(b_y), 0.0, np.cos(b_y)]])
    frame = _Rz(a_z) @ _Ry
    eps_low = frame @ np.diag([2.2, 2.8, 3.4]).astype(complex) @ frame.T
    d_con = _design([Layer("film", d, "lo")], pol="y", theta=20.0, phi=30.0)
    r_con = make_lumenairy_berreman_solver()(d_con, None, {"film": EpsField(tensor=eps_low)},
                                             LAM, n_sup, n_sub)
    _, _, Jr_cl, _ = lum.berreman_jones_1d([(eps_low, d)], n_sub, n_sup, LAM,
                                           angle=th_c, phi=phi_c)
    Jr_sp = _Rot2(-phi_c) @ np.asarray(Jr_cl) @ _Rot2(phi_c)
    e2e_err = abs(r_con.r - complex(Jr_sp[1, 1]))          # 'y' = rotated s = row 1, rf = 1
    # azimuthal invariance: an ISOTROPIC (scalar-eps) film must reproduce its in-plane
    # R/T/r at ANY azimuth exactly (scalars are rotation-invariant -> bit-identical path)
    d_iso0 = _design([Layer("film", d, "lo")], pol="y", theta=20.0)
    d_iso1 = _design([Layer("film", d, "lo")], pol="y", theta=20.0, phi=137.0)
    r_iso0 = make_lumenairy_berreman_solver()(d_iso0, None, {}, LAM, n_sup, n_sub)
    r_iso1 = make_lumenairy_berreman_solver()(d_iso1, None, {}, LAM, n_sup, n_sub)
    azim_err = max(abs(r_iso1.R - r_iso0.R), abs(r_iso1.T - r_iso0.T),
                   abs(r_iso1.r - r_iso0.r))
    conical_ok = bool(covar_err < 1e-9 and phi_matters > 1e-2 and e2e_err < 1e-9
                      and azim_err < 1e-12)
    # DISPATCH BOUNDARY: a patterned layer must RAISE pointing at RCWA
    pil = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 150e-9, 80e-9), material="pillar")
    d_pat = _design([Layer("slab", 200e-9, "air", inclusions=[pil])], pol="y")
    raised = False
    try:
        design_to_berreman_layers(d_pat, LAM)
    except NotImplementedError as exc:
        raised = "RCWA" in str(exc) or "PMM" in str(exc)
    # and a patterned eps_cell slab through the LayeredStackSolver raises too
    raised_cell = False
    try:
        BerremanLayeredSolver().solve(
            LayeredStack(n_sup, n_sub, [LayeredSlab(d, eps_cell=np.full((4, 1), 4.0 + 0j))],
                         period_x_m=PER), LAM, OpticalSpec(polarization="y",
                                                           incidence_angle_deg=0.0))
    except NotImplementedError:
        raised_cell = True
    g_f = bool(design_seam_err < 1e-9 and conical_ok and raised and raised_cell)
    ok = ok and g_f
    print("[lbb] GATE F: design-seam {:.1e}, conical Jones-covariance {:.1e} (phi matters {:.3f}), "
          "conical E2E bridge-vs-native-rotated {:.1e} (low-sym biaxial; sign-flip probed), "
          "isotropic azimuth-invariance {:.1e}, patterned dispatch raises {}/{} -> {}".format(
              design_seam_err, covar_err, phi_matters, e2e_err, azim_err,
              raised, raised_cell, "PASS" if g_f else "FAIL"), flush=True)

    # ---- GATE G: A1 far-field consolidation (class solve == functional) + C2 OOP absorption ----
    # A1: the bridge sources the far field from ONE BerremanStack CLASS solve (+ jones_transmission)
    # instead of the functional berreman_jones_1d PLUS a second retain_internal solve. Pin that
    # consolidation results-IDENTICAL: the class-solved r/t/R/T must equal the functional far field
    # post-processed identically (p-basis + lab-row extraction) bit-for-bit, on an IN-PLANE tensor
    # (p-pol -> exercises the p-basis factors) AND a CONICAL-rotated case. C2: with the director
    # tilted OUT of plane at OBLIQUE incidence, layer_absorption() now reconstructs the generalized-
    # cascade internals (lumenairy 5.22), so the per-layer budget CLOSES (was A_independent=None).
    from dynameta.optics.lumenairy_bridge._common import (p_basis_conversion as _pbc_g,
                                                          pol_row as _prow_g)
    from dynameta.optics.lumenairy_bridge.berreman_backend import \
        _rotate_layers_conical as _rot_con_g

    def _func_rt(layers_ff, theta_r, phi_r, pol):
        """The far field the bridge USED to source from the functional entry, post-processed
        with the SAME p-basis conversion + lab-row extraction the bridge applies."""
        Rf, Tf, Jrf, Jtf = lum.berreman_jones_1d(layers_ff, complex(n_sub), complex(n_sup),
                                                 LAM, angle=theta_r, phi=phi_r)
        row = _prow_g(SimpleNamespace(polarization=pol))
        rf, tf = _pbc_g(pol, theta_r, n_sup, n_sub)
        return (float(Rf[row]), float(Tf[row]),
                complex(Jrf[row, row]) * rf, complex(Jtf[row, row]) * tf)

    # (a1) in-plane tensor, p-pol at oblique: class LayeredStack path vs functional far field
    rb_ip = BerremanLayeredSolver().solve(
        _tensor_stack(eps_uni, d, n_sup, n_sub), LAM,
        OpticalSpec(polarization="p", incidence_angle_deg=25.0))
    gR, gT, gr, gt = _func_rt([(eps_uni, d)], np.radians(25.0), 0.0, "p")
    ip_err = max(abs(rb_ip.R - gR), abs(rb_ip.T - gT), abs(rb_ip.r - gr), abs(rb_ip.t - gt))
    # (a2) conical (phi != 0): the bridge rotates the layers then solves phi=0; the functional
    # far field of the SAME rotated layers must match the design-seam result bit-for-bit
    r_con_g = make_lumenairy_berreman_solver()(
        _design([Layer("film", d, "lo")], pol="y", theta=20.0, phi=30.0), None,
        {"film": EpsField(tensor=eps_low)}, LAM, n_sup, n_sub)
    lay_rot = _rot_con_g([(eps_low, d)], np.radians(30.0))
    cR, cT, cr, ct = _func_rt(lay_rot, np.radians(20.0), 0.0, "y")
    con_err = max(abs(r_con_g.R - cR), abs(r_con_g.T - cT),
                  abs(r_con_g.r - cr), abs(r_con_g.t - ct))
    consol_err = max(ip_err, con_err)

    # (b) OOP-tensor director tilted ~40deg out of plane (both n_o/n_e lossy) at theta=30 phi=25 ->
    # the C2 absorption budget closes; the lossy iso 'cap' makes the per-region sum non-trivial
    _el, _az = np.radians(40.0), np.radians(30.0)
    _dv = np.array([np.cos(_el) * np.cos(_az), np.cos(_el) * np.sin(_az), np.sin(_el)])
    _dv = _dv / np.linalg.norm(_dv)
    _no, _ne = complex(1.50, 0.05), complex(1.75, 0.08)
    eps_tilt = (_no ** 2) * np.eye(3, dtype=complex) + (_ne ** 2 - _no ** 2) * np.outer(_dv, _dv)
    d_oop = _design([Layer("director", 900e-9, "lo"), Layer("cap", 150e-9, "hi")],
                    pol="y", theta=30.0, phi=25.0)
    r_oop = make_lumenairy_berreman_solver(absorption=True)(
        d_oop, None, {"director": EpsField(tensor=eps_tilt)}, LAM, n_sup, n_sub)
    oop_budget = (abs(r_oop.A_independent - r_oop.A)
                  if r_oop.A_independent is not None else float("inf"))
    oop_keys = (bool(r_oop.per_region_absorption)
                and set(r_oop.per_region_absorption) == {"director", "cap"})
    oop_ok = bool(r_oop.A_independent is not None and oop_budget < 1e-6 and oop_keys)
    g_g = bool(consol_err < 1e-13 and oop_ok)
    ok = ok and g_g
    print("[lbb] GATE G: A1 class==functional far field (in-plane p-pol {:.1e}, conical {:.1e}) "
          "{:.1e}; C2 OOP-oblique absorption budget |sumA-(1-R-T)|={:.1e} (A_ind={}, keys {}) "
          "-> {}".format(ip_err, con_err, consol_err, oop_budget,
                         None if r_oop.A_independent is None else round(r_oop.A_independent, 4),
                         oop_keys, "PASS" if g_g else "FAIL"), flush=True)

    print("[lbb] *** LUMENAIRY BERREMAN BRIDGE: {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
