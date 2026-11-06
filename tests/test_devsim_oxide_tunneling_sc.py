"""v0.15 (plan item 4.1.1): SELF-CONSISTENT gate-oxide tunnelling (carriers.tunneling.setup_oxide_tunneling) on
1-D MOS capacitors. Docs: the tunneling module header + docs/audit/2026-11-06-v0.15-oxide-tunnelling.md.

EVERY DEVICE SCENARIO RUNS IN A CHILD PROCESS (tests/_oxide_tunneling_child.py): the term is a custom
equation (its DEVSIM dispatcher can never be unregistered) and the circuit gates are DEVSIM circuits (they
cannot be removed, and no circuit node can be added after the first solve) -- one circuit per child.

ORACLES (independent of the code under test; DEVSIM supplies only the solved arrays):
  detailed balance  the algebra of the Tsu-Esaki integrand (zero pointwise at E_Fs = E_Fg);
  inversion         the depletion approximation re-derived here (closed-form W(V_g) with the gate charge),
                    the SRH rate with n_1 = p_1 = n_i integrated by scipy quad, plus the short-base minority
                    supply q D_n n_p0 / (L - W) of the ohmic body contact; and the continuity identity
                    I_gate = q Int G dV + (the body contact's electron current);
  conservation      I_body + I_gate = 0 -- in 128-bit (extended_precision), where the floor is the last
                    double digit, and in double against the rigorous residual bound sum |F|;
  ssac              Re Y(f -> 0) vs the DC dI/dV by 4th-order (Richardson) central differences;
  noise             the fluctuation-dissipation theorem S_I = 4 kT Re Y over 1 Hz - 1 GHz;
  Jacobian          central differences (h, 2h) of the term's own residual, and of DEVSIM's global system;
  stack             D-field continuity eps_j E_j across the layers of a charge-free stack; one SiO2 region
                    == the same SiO2 split into two regions.
OLD CODE ("the post-processor", coupling="post", the v0.14 one-sided FN / Schuegraf-Hu / Tsu-Esaki on the
ideal-capacitor solution) is evaluated inside the gates and FAILS each of them by the amounts stated; on the
pre-v0.15 tree (8bbf36c) every test here fails outright (no setup_oxide_tunneling: the child raises).
Runtime: ~85 s serial (children 2-18 s).
"""
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("devsim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _oxide_tunneling_child as CH
finally:
    sys.path.pop(0)

Q = 1.602176634e-19
KB = 1.380649e-23
EPS0 = 8.8541878128e-12
VT = KB * 300.0 / Q
EPS = np.finfo(float).eps


def _child(scenario, **opts):
    return CH.run_child(scenario, **opts)


# =====================================================================================================
# detailed balance
# =====================================================================================================
@pytest.fixture(scope="module")
def balance():
    return _child("balance")


@pytest.mark.parametrize("case", ["n_acc", "p_inv", "p_acc", "n_inv", "n_acc_wf"])
def test_o1_detailed_balance_at_zero_bias_is_exact(balance, case):
    """O1: zero bias with a work-function offset (a built-in oxide field: |V_ox| 0.29-1.17 V over 2 nm, i.e.
    up to 5.8e8 V/m), Boltzmann bulk. |J_net| / J_one-way <= 1e-10. Budget: J/J_one ~ the residual of
    x_s - x_g, i.e. of the discrete equilibrium of ln n - psi/V_t, which the Newton converges to rel 1e-12
    (Boltzmann SG is exactly ln n - psi/V_t = const at zero current); MEASURED <= 1.4e-14.
    OLD CODE: the post-processor at the SAME equilibrium gives |J_post| / J_one-way = 25-66 (Schuegraf-Hu)
    and 1.1-2.6 (Tsu-Esaki with E_F = 0.1 eV) on the accumulated / inverted cases (> 0.5 asserted) --
    a current with no driving force; FN is nonzero too (and 1e-15..1e-9 of J_one: the wrong formula
    below the barrier). An n_s/(n_s + n_ref)-scaled FN sink (the rejected alternative) is nonzero as well.
    REFUTE: a 1 mV error in the gate frame (phi_c) gives |J| / J_one = 1.9-3.8e-2."""
    r = balance[case]
    assert r["J_fwd"] > 0.0 and r["J_bwd"] > 0.0
    assert abs(r["x_s"] - r["x_g"]) <= 1e-9
    assert abs(r["J"]) <= 1e-10 * r["J_fwd"]
    assert abs(r["J_refute"]) > 1e-3 * r["J_fwd"]
    fn = r["post_fn"]
    assert fn != 0.0
    assert abs(fn * r["n_s"] / (r["n_s"] + 1e24)) > 0.0          # the rejected n_s-scaled FN sink
    if case in ("n_acc", "p_inv", "n_acc_wf"):                    # a real one-way current to compare with
        assert abs(r["post_schuegraf_hu"]) / r["J_fwd"] > 0.5
        assert abs(r["post_tsu_esaki"]) / r["J_fwd"] > 0.5


def test_o1_the_built_in_field_and_both_polarities(balance):
    """The O1 fixtures really carry an oxide field at equilibrium, of both signs (gate above / below)."""
    assert balance["n_acc"]["V_ox"] > 0.8 and balance["n_acc_wf"]["V_ox"] > 1.1
    assert balance["p_acc"]["V_ox"] < -0.4 and balance["n_inv"]["V_ox"] < -0.25
    assert balance["p_inv"]["V_ox"] > 0.5


def test_o1_fermi_dirac_equilibrium_converges_at_second_order():
    """O1 with the Fermi-Dirac g-factor (~18 s): the DISCRETE FD-SG equilibrium (edge-averaged g) is off
    the closed-form activity ln u + Delta(u) by the trapezoid error of Int g d ln n across the degenerate
    accumulation layer (x_s = 3.3), so the zero-bias |J| / J_one-way is a mesh-convergence quantity: on the
    exactly uniform 0.1 nm x s surface zone it must fall at order >= 1.7 (the bulk scheme is O(h^2)).
    MEASURED 1.6e-3, 4.0e-4, 1.0e-4, 2.5e-5 at s = 1, 1/2, 1/4, 1/8 (orders 1.99, 2.00, 2.00); with
    DEVSIM's own graded mesh one odd-sized edge AT the interface (2-9 h) held it at ~4e-4 for every h.
    Refute: the same with Boltzmann statistics is at round-off (O1 above)."""
    r = _child("balance_fd", scales=[1.0, 0.5, 0.25, 0.125])
    e = np.array([abs(p["J"]) / p["J_fwd"] for p in r])
    orders = np.log2(e[:-1] / e[1:])
    assert np.all(orders >= 1.7), (e, orders)
    assert e[0] > 1e-4                                       # it is a real effect at a 0.1 nm mesh


# =====================================================================================================
# supply-limited inversion (the load-bearing physics gate)
# =====================================================================================================
NA, NI, TOX, L_BODY = 1e23, 1e16, 2e-9, 1.5e-6
EPS_S, EPS_OX = 11.7 * EPS0, 3.9 * EPS0
MU_N = 0.1


def _deep_depletion_current(Vg, tau, n_s):
    """The analytic deep-depletion gate current: depletion approximation for psi(x) with the gate charge
    (V_g - psi_b = a W^2 + b W, a = q N_A / (2 eps_s), b = q N_A / C_ox, flat band at psi_g = psi_b), holes
    at equilibrium p = N_A exp(-(psi - psi_b)/V_t), the drained surface electron layer n = n_s
    exp(-(psi_s - psi)/V_t) (n_s READ from the solution: it only sets the ~10 nm layer's width,
    logarithmically), SRH generation (n_i^2 - n p) / (tau (n + p + 2 n_i)) integrated by quad, plus the
    short-base minority supply of the ohmic body contact q D_n (n_i^2/p0) / (L - W)."""
    from scipy.integrate import quad
    p0 = NA / 2 + math.sqrt(NA ** 2 / 4 + NI ** 2)
    psib = -VT * math.log(p0 / NI)
    a, b = Q * NA / (2 * EPS_S), Q * NA / (EPS_OX / TOX)
    W = (-b + math.sqrt(b * b + 4 * a * (Vg - psib))) / (2 * a)
    psis = psib + a * W * W

    def psi(y):
        return psib + a * (W - y) ** 2 if y < W else psib

    def G(y):
        p = NA * math.exp(-(psi(y) - psib) / VT)
        n = n_s * math.exp(-(psis - psi(y)) / VT)
        return (NI * NI - n * p) / (tau * (n + p + 2 * NI))
    I_gen = Q * (quad(G, 0, W, limit=400, epsrel=1e-10)[0] + quad(G, W, L_BODY, limit=200)[0])
    I_contact = Q * MU_N * VT * (NI * NI / p0) / (L_BODY - W)
    return I_gen + I_contact, W, psis, Q * NI * W / (2 * tau)


@pytest.fixture(scope="module")
def inversion():
    return _child("inversion", taus=[1e-6, 1e-7], V=[1.5, 3.0])["pts"]


def test_o4_supply_limited_inversion_matches_the_deep_depletion_estimate(inversion):
    """O4: p-Si 1e23 m^-3, 2 nm SiO2, V_g = 1.5 / 3 V, tau = 1e-6 / 1e-7 s. The tunnelling drains the
    inversion layer; the body deep-depletes (psi_s within 1.1 mV of the depletion approximation) and the
    gate current is the generation supply. Within 5 % of the analytic estimate (plan 4.1.1; the depletion
    approximation and the surface electron layer -- MEASURED +1.3..+2.3 %). Without the surface electron
    layer the estimate misses by -5..-16 %, and the textbook q n_i W / (2 tau) by -40..-53 % (reported, the
    p >> n_i edge of the depletion region does not generate). OLD CODE: the post-processor on the ideal
    capacitor (equilibrium inversion at the same V_g) gives 7e5..8e8 x (Schuegraf-Hu) and 3e5..5e7 x
    (Tsu-Esaki, E_F = 0.1 eV) the supply-limited current (> 10x asserted); FN 1.6e6..1.6e7 x at 3 V
    (at 1.5 V, V_ox < phi_b, FN under-counts by 1e9: the wrong formula there in the other direction)."""
    for p in inversion:
        I, W, psis, simple = _deep_depletion_current(p["V"], p["tau"], p["n_s"])
        assert abs(p["psi_s"] - psis) < 2e-3
        assert p["I_gate"] == pytest.approx(I, rel=0.05), (p["I_gate"] / I - 1)
        assert abs(p["I_gate"] / simple - 1) > 0.3                     # the textbook form is not the gate
        assert p["post"]["schuegraf_hu"] > 10 * p["I_gate"] and p["post"]["tsu_esaki"] > 10 * p["I_gate"]
        if p["V"] >= 3.0:
            assert p["post"]["fn"] > 10 * p["I_gate"]


def test_o4_gate_current_is_the_generation_integral_plus_the_contact_supply(inversion):
    """O4 part 1 (continuity): I_gate = q Int G_SRH dV (interior nodes) - I_n(body contact), exactly: the
    electron continuity rows telescope. MEASURED |residual| <= 2.1e-19 A/m^2 (4e-16 of I_gate); budget
    N eps (|I_gate| + q Int |U| dV) with N = 500 rows (the round-off of the identity's terms). The body
    contact's minority supply is 0.06-0.55 % of I_gate here (short-base diffusion from n_p0)."""
    for p in inversion:
        lhs = p["I_gate"] - p["I_gen"] + p["I_n_body"]
        assert abs(lhs) <= 500 * EPS * (abs(p["I_gate"]) + p["I_gen_abs"])
        assert 0.0 < -p["I_n_body"] < 0.01 * p["I_gate"]


def test_o4_tau_refutation_monotone_and_bounded(inversion):
    """REFUTE: shorten tau 10x -> I_gate rises ~10x (generation-limited: 9.56x / 9.82x MEASURED), and
    I_gate stays below the old tunnel-limited post-processor value at every point."""
    by = {(p["tau"], p["V"]): p for p in inversion}
    for V in (1.5, 3.0):
        r = by[(1e-7, V)]["I_gate"] / by[(1e-6, V)]["I_gate"]
        assert 8.0 < r < 10.5
        for tau in (1e-6, 1e-7):
            p = by[(tau, V)]
            assert p["I_gate"] < min(p["post"]["schuegraf_hu"], p["post"]["tsu_esaki"])


# =====================================================================================================
# conservation
# =====================================================================================================
@pytest.fixture(scope="module")
def conserve():
    return {v: _child("conserve", variant=v) for v in
            ("param", "circuit", "param_xp", "circuit_xp", "flipped_xp", "old")}


def _i_gate(p):
    return p["I_gate"] if p["I_src"] is None else -p["I_src"]   # the source branch current is -I(contact)


def test_o2_terminal_currents_balance_exactly_in_extended_precision(conserve):
    """O2: I_body + I_gate = 0 in 128-bit (extended_precision.solve; with the double-valued term the Newton
    converges linearly, 3 iterations to rel 2e-32 here): the floor is the last double digit, gate <= 2 ulp
    of |I_gate|. MEASURED 0 ulp at 0.6 / 1.0 / 1.4 V for the bias-parameter gate (module-reported I_gate)
    AND the circuit gate (the source current). REFUTE: the KCL entries with the wrong sign give
    I_body + I_gate = -2 I (MEASURED -2.000). contact_current.extract_contact_currents carries the gate
    current (DEVSIM's own is 0 in DC) and its dict sums to the same zero."""
    for v in ("param_xp", "circuit_xp"):
        for p in conserve[v]["pts"]:
            Ig = _i_gate(p)
            assert Ig > 0.5
            assert abs(Ig + p["I_body"]) <= 2 * np.spacing(abs(Ig))
            cc = p["cc"]
            assert set(cc) == {"body", "gate"} and abs(cc["gate"] + cc["body"]) <= 2 * np.spacing(abs(Ig))
    for p in conserve["flipped_xp"]["pts"]:
        assert (_i_gate(p) + p["I_body"]) / abs(p["I_body"]) == pytest.approx(-2.0, rel=1e-12)


def test_o2_terminal_balance_in_double_and_the_old_code(conserve):
    """O2 in double: |I_body + I_gate| <= sum |F| over the interior carrier rows and the KCL row (the
    rigorous bound: the rows' signed sum telescopes to the imbalance). MEASURED imbalance 1-2e-5 A/m^2
    (1e-6 of I) against a bound of 5-34 A/m^2 -- loose: the accumulation layer's SG flux terms are ~1e13
    A/m^2 per edge; the extended-precision gate above is the sharp one. OLD CODE: the post-processed
    current is NOT drawn from the body: I_body = 2.8e-6 A/m^2 (the solver floor) against J_post = 1655
    (Schuegraf-Hu) / 63 (Tsu-Esaki) A/m^2 at 1 V -- terminal charge is not conserved, and the gate is
    absent from the terminal currents."""
    for v in ("param", "circuit"):
        for p in conserve[v]["pts"]:
            assert abs(_i_gate(p) + p["I_body"]) <= p["floor"]
    o = conserve["old"]
    assert abs(o["I_body"]) < 1e-3 * o["J_post"] and o["J_post"] > 100.0 and o["J_post_te"] > 10.0
    assert set(o["cc"]) == {"body"}


# =====================================================================================================
# ssac
# =====================================================================================================
@pytest.fixture(scope="module")
def ssac():
    return {v: _child("ssac", variant=v) for v in ("on", "off", "flipped")}


def _dIdV(r):
    I = r["I"]
    d1 = (I["0.0001"] - I["-0.0001"]) / 2e-4
    d2 = (I["0.0002"] - I["-0.0002"]) / 4e-4
    return (4 * d1 - d2) / 3, abs(d1 - d2)


def test_o8_ssac_conductance_is_the_dc_slope(ssac):
    """O8: circuit gate, n-Si accumulation at 1.0 V (J = 15.4 A/m^2). Re Y(1 Hz) == the DC dI/dV
    (4th-order central differences, h = 0.1 / 0.2 mV) to 1e-6 (plan 4.1.1). Budget: the Richardson
    remainder (<= |d1 - d2|, MEASURED 3e-8 relative) + quasi-static corrections (omega tau)^2 ~ 1e-20
    at 1 Hz. MEASURED 1.1e-11 at 1 Hz, 7.9e-9 at 1 kHz. OLD CODE (no term): Re Y = 5.9e-13 S/m^2 (the
    oxide is an insulator to ssac) against dI/dV = 74.6 S/m^2. REFUTE: the KCL entries with the wrong
    sign flip Re Y (-74.58: G/dIdV = -2 against the true slope)."""
    R, rich = _dIdV(ssac["on"])
    G1 = ssac["on"]["G"][0]
    # sign: Y = -I_source / V; the source current is -I_gate, so dI_source/dV = -dI_gate/dV
    assert -R > 10.0
    assert G1 == pytest.approx(-R, rel=1e-6)
    assert rich / abs(R) < 1e-6
    assert abs(ssac["off"]["G"][0]) < 1e-9 * abs(R)
    Rf, _ = _dIdV(ssac["flipped"])
    assert ssac["flipped"]["G"][0] == pytest.approx(-abs(R), rel=1e-6)
    assert Rf == pytest.approx(abs(R), rel=1e-9)               # the flipped KCL reverses the source current
    assert ssac["on"]["C"][0] == pytest.approx(ssac["off"]["C"][0], rel=1e-6)   # the charge is unchanged


# =====================================================================================================
# noise
# =====================================================================================================
@pytest.fixture(scope="module")
def fdt():
    return {v: _child("fdt", variant=v) for v in ("n_acc", "n_dep")}


@pytest.mark.parametrize("case", ["n_acc", "n_dep"])
def test_o10_noise_fluctuation_dissipation_with_the_tunnel_shot_source(fdt, case):
    """O10: circuit gate at EQUILIBRIUM (body and gate both at 0.5 V; a work-function offset 1.5 V gives a
    degenerate accumulation x_s = 2.28 -- 'n_acc' -- or depletion x_s = -15 -- 'n_dep'). S_I = 4 kT Re Y,
    1 Hz - 1 GHz, to 1e-4 (the D2a gate). MEASURED <= 1.6e-9 (n_acc), 2.2e-11 (n_dep). The tunnel source
    (mechanism tunnel_ox, S = 2 q J coth(qV/2kT)) carries 99.99998 % of S at 1 Hz. OLD CODE / REFUTE:
    dropping the source misses by -100 % at low f; the one-sided shot formula S = 2 q |J| is 0 at
    equilibrium (the same -100 %); the 'one-sided sum' 2 q (J_fwd + J_bwd), which drops the Pauli
    partition term, overcounts by +68 % on the degenerate surface (and is right only when non-degenerate:
    +7e-8 on n_dep, which is why the degenerate case is the gate)."""
    r = fdt[case]
    S, G = np.array(r["S"]), np.array(r["G"])
    assert np.all(G > 0.0)
    fdt_err = S / (4 * KB * 300.0 * G) - 1.0
    assert np.all(np.abs(fdt_err) <= 1e-4), fdt_err
    t = np.array(r["by_mechanism"]["tunnel_ox"])
    assert t[0] / S[0] > 0.99
    dropped = (S - t) / (4 * KB * 300.0 * G) - 1.0
    assert dropped[0] < -0.99
    one_sided = 2 * Q * abs(r["J"]) * r["area"]
    assert one_sided == 0.0
    if case == "n_acc":
        alt = t * 2 * Q * (r["J_fwd"] + r["J_bwd"]) / r["S_kernel"]
        assert ((S - t + alt) / (4 * KB * 300.0 * G) - 1.0)[0] > 0.5


# =====================================================================================================
# Jacobian
# =====================================================================================================
@pytest.fixture(scope="module")
def jac():
    return {v: _child("jac", variant=v) for v in ("acc", "inv", "near_eq", "stack")}


@pytest.mark.parametrize("case", ["acc", "inv", "near_eq", "stack"])
def test_o6_the_analytic_jacobian_is_exact(jac, case):
    """O6: (a) the term's OWN Jacobian (its rcv) == central differences of its OWN residual (rv) at the
    same state, every column it touches -- psi_s, n_s, each face potential (the gate face of the oxide is a
    column several elements away: a local model would miss it), the gate circuit node. Budget: |fd(h) -
    fd(2h)| (3x the O(h^2) truncation) + 16 eps |F| / h + 8 eps |J|. MEASURED err/budget <= 0.34, i.e.
    1e-11..3e-9 relative where the entry is resolvable (the 'acc' gate-Fermi column, 1.3e-14, is below
    the double resolution of F: inside its round-off budget). (b) The same share in DEVSIM's GLOBAL matrix
    (on - off) == central differences of DEVSIM's residual (the custom_terms J1 pattern), within the global
    round-off budget 8 eps R / h. 'near_eq' exercises the gate-Fermi column (dJ/dV_g = 1798 S/m^2)."""
    r = jac[case]
    for rec in r["term"]:
        fd, fd2, J, F = (np.asarray(rec[k]) for k in ("fd", "fd2", "J", "F"))
        tol = np.abs(fd - fd2) + 16 * EPS * np.abs(F) / rec["h"] + 8 * EPS * np.abs(J)
        assert np.all(np.abs(fd - J) <= tol), (rec["col"], fd, J, tol)
    for rec in r["global"]:
        fd, fd2, J = (np.asarray(rec[k]) for k in ("fd", "fd2", "J"))
        tol = np.abs(fd - fd2) + 8 * EPS * rec["rhs_scale"] / rec["h"] + 8 * EPS * np.abs(J)
        assert np.all(np.abs(fd + J) <= tol), (rec["col"], fd, J, tol)   # rhs = -F
    if case == "near_eq":
        circ = [rec for rec in r["term"] if rec["col"][0] == "circuit"][0]
        assert abs(circ["J"][0]) > 100.0


def test_o6_refute_a_dropped_face_column_breaks_the_check(jac):
    """REFUTE: the gate-face column carries a full share of dJ (|dJ/dpsi_gate| = |dJ/dpsi_s| in 1-D
    accumulation when the gate Fermi level is far below: J depends on psi_s - psi_g); zeroing it (a local
    model's Jacobian) misses its central difference by 100 %."""
    face = [rec for rec in jac["acc"]["term"] if rec["col"][0] == "oxide"][0]
    fd, J = np.asarray(face["fd"]), np.asarray(face["J"])
    assert np.all(np.abs(fd - 0.0 * J) > 0.99 * np.abs(fd))
    assert np.all(np.abs(fd) > 10.0)


# =====================================================================================================
# dielectric stacks
# =====================================================================================================
@pytest.fixture(scope="module")
def stack():
    return _child("stack", V=1.2)


def test_o12_split_oxide_equals_one_oxide_and_the_stack_drops_follow_d_continuity(stack):
    """O12: (a) the same 2 nm SiO2 as ONE region and as TWO regions (0.8 + 1.2 nm): J equal to 1e-10
    (MEASURED 2.5e-12: the kernel's layer sum, K10, and the two solves). (b) SiO2 1 nm / high-k (eps 20,
    offset 1.5 eV, m 0.18) 3 nm: the face potentials the term reads give the SAME D field eps_j V_j / t_j
    in both layers (charge-free stack; 1e-8, MEASURED ~1e-12) and the layer offsets are the barriers.
    REFUTE: reversing the layer order changes J (5.86 -> 0.95 A/m^2)."""
    a, b = stack["single"], stack["split"]
    assert b["J"] == pytest.approx(a["J"], rel=1e-10)
    assert len(b["faces"]) == 3 and len(a["faces"]) == 2
    for tag in ("sio2_hk", "hk_sio2", "split"):
        r = stack[tag]
        D = [r["eps"][j] * (r["faces"][j + 1] - r["faces"][j]) / r["t"][j] for j in range(len(r["t"]))]
        assert np.allclose(D, D[0], rtol=1e-8)
    assert stack["sio2_hk"]["offsets"] == [3.1, 1.5]
    assert abs(stack["hk_sio2"]["J"] / stack["sio2_hk"]["J"] - 1.0) > 0.5
    for r in stack.values():                                # the module-reported current is the body's
        assert abs(r["I_gate"] + r["I_body"]) <= r["floor"]


# =====================================================================================================
# refusals and the parameter rules (in-process: every refusal happens before anything is registered)
# =====================================================================================================
def test_params_refusals_and_defaults_off():
    """Decision 10: FN and Schuegraf-Hu are REFUSED as self-consistent supplies (one-sided: an equilibrium
    current and an FDT violation), with the reason; E_F_eV / T_K are refused there (the supply is the
    solved n_s, the temperature the region's V_t); barriers and the gate Fermi energy are required. The
    default stays the post-processor (coupling='post'), so every existing caller is unchanged."""
    from dynameta.carriers.tunneling import SIO2_ON_SI, OxideTunnelingParams
    assert OxideTunnelingParams().coupling == "post"
    for m in ("fn", "schuegraf_hu"):
        with pytest.raises(ValueError, match="REFUSED.*ONE-SIDED"):
            OxideTunnelingParams(model=m, coupling="self_consistent", barriers=SIO2_ON_SI, gate_fermi_energy_eV=8.0)
    ok = dict(model="tsu_esaki", coupling="self_consistent", barriers=SIO2_ON_SI, gate_fermi_energy_eV=8.0)
    OxideTunnelingParams(**ok)
    for bad, msg in (({"E_F_eV": 0.1}, "E_F_eV must be None"), ({"T_K": 250.0}, "T_K"),
                     ({"barriers": None}, "needs barriers"), ({"gate_fermi_energy_eV": None}, "gate_fermi"),
                     ({"coupling": "both"}, "coupling")):
        with pytest.raises(ValueError, match=msg):
            OxideTunnelingParams(**dict(ok, **bad))
    with pytest.raises(ValueError, match="E_F_eV"):                 # the post-processor rule is unchanged
        OxideTunnelingParams(model="tsu_esaki")


def test_dielectric_barrier_against_a_named_semiconductor():
    """Decision 13 (plan 5.4): a barrier fitted against Si is used AS GIVEN on Si -- whatever Si's chi is
    (None, 4.05 or 4.17: the fit never silently becomes 3.22 eV) -- and is carried to ANOTHER semiconductor
    by Anderson, offset = chi_s - (chi_Si - barrier), which needs both affinities: a None chi is refused."""
    from dynameta.carriers.tunneling import SIO2_ON_SI, DielectricBarrier
    for chi in ({}, {"Si": None}, {"Si": 4.05}, {"Si": 4.17}):
        assert SIO2_ON_SI.offset_eV("Si", chi) == 3.1
    assert SIO2_ON_SI.offset_eV("GaAs", {"Si": 4.05, "GaAs": 4.07}) == pytest.approx(4.07 - (4.05 - 3.1), abs=1e-15)
    for chi in ({"Si": None, "GaAs": 4.07}, {"Si": 4.05}, {}):
        with pytest.raises(ValueError, match="no neutral default"):
            SIO2_ON_SI.offset_eV("GaAs", chi)
    with pytest.raises(ValueError):
        DielectricBarrier(barrier_eV=3.1, against="", m_tunnel_over_m0=0.42)


def test_gate_offset_for_work_function_round_trip():
    """Decision 12: the barrier is READ from the gate contact offset; the helper converts a work function:
    W_m = chi_s + phi_c - offset (derivation in its docstring), re-coded here."""
    from dynameta.carriers.tunneling import gate_offset_for_work_function
    chi_s, phi_c = 4.05, VT * math.log(2.8e25 / 1e16)
    for W in (4.1, 4.6, 5.1):
        off = gate_offset_for_work_function(W, chi_s, phi_c)
        assert chi_s + phi_c - off == pytest.approx(W, abs=1e-15)


def _bare_1d(dev, regions):
    """A 1-D device of consecutive regions [(name, material, length)]; returns nothing (no physics)."""
    ds = pytest.importorskip("devsim")
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    x = 0.0
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=0.5e-9, tag="t0")
    for k, (_, _, L) in enumerate(regions):
        x += L
        ds.add_1d_mesh_line(mesh=mesh, pos=x, ps=0.5e-9, tag="t{}".format(k + 1))
    ds.add_1d_contact(mesh=mesh, name="body", tag="t0", material="metal")
    ds.add_1d_contact(mesh=mesh, name="gate", tag="t{}".format(len(regions)), material="metal")
    for k, (nm, mat, _) in enumerate(regions):
        ds.add_1d_region(mesh=mesh, material=mat, region=nm, tag1="t{}".format(k), tag2="t{}".format(k + 1))
        if k:
            ds.add_1d_interface(mesh=mesh, name="i{}".format(k), tag="t{}".format(k))
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    return ds


def _bipolar(dev, region, dop):
    from dynameta.carriers import physics_bipolar_dd as BP
    from dynameta.carriers.devsim_env import import_devsim
    ds = import_devsim()
    ds.node_model(device=dev, region=region, name="NetDoping", equation="{:.6e}".format(dop))
    BP.setup_bipolar_region(dev, region, eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1e16, mobility_n_m2Vs=0.1,
                            mobility_p_m2Vs=0.04, tau_n_s=1e-6, tau_p_s=1e-6, fd_enhancement=False)


def test_setup_refuses_non_bipolar_semiconductor_gates_and_registers_nothing():
    """Refused before anything is registered: a semiconductor without carrier rows (the equilibrium /
    Poisson-only physics), a semiconductor (poly-Si) gate (decision 17), and a column whose dielectric
    neighbour is ambiguous."""
    from dynameta.carriers import custom_terms as CT
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import tunneling as TN
    prm = CH.params()
    was = CT.is_registered()
    try:
        ds = _bare_1d("ot_r1", [("semi", "Si", 20e-9), ("oxide", "SiO2", 2e-9)])
        PE.setup_dielectric_region("ot_r1", "semi", 11.7)
        PE.setup_dielectric_region("ot_r1", "oxide", 3.9)
        with pytest.raises(ValueError, match="bipolar"):
            TN.setup_oxide_tunneling("ot_r1", "semi", "gate", prm)
        with pytest.raises(ValueError, match="coupling='self_consistent'"):
            TN.setup_oxide_tunneling("ot_r1", "semi", "gate", TN.OxideTunnelingParams())
        ds.delete_device(device="ot_r1")
        ds.delete_mesh(mesh="ot_r1_m")
        _R.clear("ot_r1")
        ds = _bare_1d("ot_r2", [("semi", "Si", 20e-9), ("oxide", "SiO2", 2e-9), ("poly", "Si", 10e-9)])
        _bipolar("ot_r2", "semi", -1e23)
        _bipolar("ot_r2", "poly", 1e26)
        PE.setup_dielectric_region("ot_r2", "oxide", 3.9)
        with pytest.raises(NotImplementedError, match="poly-Si"):
            TN.setup_oxide_tunneling("ot_r2", "semi", "gate", prm)
    finally:
        from dynameta.carriers.devsim_env import import_devsim
        ds = import_devsim()
        for d in ("ot_r1", "ot_r2"):
            if d in ds.get_device_list():
                ds.delete_device(device=d)
                ds.delete_mesh(mesh=d + "_m")
            _R.clear(d)
    assert CT.is_registered() == was
