"""LayeredDevsimBuilder.noise_spectrum / terminal_noise: the one-call gate noise of the 2-D builder (v0.15
plan 4.2.5, user decision 35; P2 item 6 gates W1-W8).

The scenarios run in CHILD processes (tests/_noise_method_child.py, one per process, at most two at a time):
DEVSIM circuits are process-global and cannot be removed in-process, so a circuit-driven gate would leak
into every later DEVSIM test of the worker (tests/_bipolar_calllog.py). Fixtures:
  * the reference gated-ITO modulator (validation/_reference_device, drift-diffusion; 370 nm cell): gate G's
    device (validation/devsim_noise.py);
  * a gated bipolar n-GaAs / N-Al0.3Ga0.7As heterostack with a THERMIONIC interface (Boltzmann statistics,
    A* x 1e-3, 60 nm cell): the default-mechanism inheritance fixture, where the interface carries 80 % of
    the gate's 1 GHz noise.

ORACLES. W1 / W4 / W8: the documented hand recipe written out call by call in the child (solve,
set_ssac_gate, circuit_alter, the library re-settle dc_solve.solve_dc with the builder's branch, then
noise_analysis.noise_spectrum) -- the same calls, so the gate is bitwise (sha256 of S and every breakdown).
W2: the fluctuation-dissipation theorem S_I = 4 kT Re Y with Y from ac_analysis.ssac_admittance (a separate
linear solve), both per cell. At equilibrium the identity is exact in exact arithmetic for both fixtures (the
flat-band ITO has uniform n, so the FD g-factor cancels -- gate G; the heterostack is Boltzmann, every source
Nyquist-exact, the thermionic one included), so the residual is the round-off of two independent
factorizations; 1e-9 is the plan's W2 tolerance (gate G recorded 1.3e-11). MEASURED: <= 7.8e-12 (modulator),
<= 2.5e-12 (heterostack). W5: a fresh builder solved at 1 V (no circuit at all). W3: S_cell = period_y S_2D
exactly; the independent 3-D check of P2 W3 is NOT run (v0.15 follow-up, audit 2026-11-06-v0.15-followups item
4: the modulator's 3-D twin is not y-invariant -- a 175 nm square patch in a 370 nm cell -- so S_3D != W S_2D by
construction, and Devsim3DEquilibrium has no circuit-driven gate) -- the scaling's exactness argument is in the
method's docstring. W7 (nightly, @slow): the 0 / +1 / 0 sequence on an extended-precision builder.

OLD CODE (aacfe46, MEASURED: this file + the child against `git archive aacfe46`): 11 of 11 fail (2 failed,
9 errors) -- every child dies with ImportError (no OhmicContactNoiseError / noise_spectrum), and the two
in-process tests fail on the missing method.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")

KB, T = 1.380649e-23, 300.0                     # scipy.constants k (exact SI); oracle side
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHILD = os.path.join(_REPO, "tests", "_noise_method_child.py")
SCENARIOS = ("seq_ref", "wrap_ref", "hand_ref", "wrap_te", "hand_te", "plain_1V", "fresh1_ref")


def _run(scenarios):
    """{scenario: result dict}; at most two children at a time (the box is shared). Output goes to temporary
    files, not pipes: DEVSIM is verbose, and a full pipe would stall the child that is not being read."""
    import tempfile
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [_REPO] + [p for p in sys.path if isinstance(p, str) and p]))
    out, todo, live, failed = {}, list(scenarios), [], []
    try:
        while todo or live:
            while todo and len(live) < 2:
                sc = todo.pop(0)
                fo, fe = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
                live.append((sc, fo, fe, subprocess.Popen([sys.executable, "-W", "error", _CHILD, sc], cwd=_REPO,
                                                          env=env, stdout=fo, stderr=fe, text=True)))
            sc, fo, fe, p = live.pop(0)
            try:
                p.wait(timeout=900)
                fo.seek(0)
                fe.seek(0)
                so, se = fo.read(), fe.read()
            finally:
                fo.close()
                fe.close()
            lines = [ln for ln in so.splitlines() if ln.startswith("NOISE_METHOD_RESULT ")]
            if p.returncode != 0 or not lines:
                failed.append((sc, p.returncode, so[-1500:], se[-3000:]))
                todo.clear()                                   # do not start more children after a failure
                continue
            out[sc] = json.loads(lines[-1][len("NOISE_METHOD_RESULT "):])
    finally:
        for _sc, fo, fe, p in live:                            # only after an unexpected exception
            p.kill()
            p.wait()
            fo.close()
            fe.close()
    assert not failed, failed
    return out


@pytest.fixture(scope="module")
def runs():
    """All scenarios: ~50 s wall on the Windows dev box (two children at a time), 2026-11-06."""
    return _run(SCENARIOS)


def _fdt(r):
    return np.array(r["S_I"]) / (4.0 * KB * T * np.array(r["ReY"])) - 1.0


def test_w1_bitwise_equal_to_the_hand_recipe(runs):
    """W1. The method (per_cell=False) equals the hand recipe bit for bit on the same device and state: the
    flat-band modulator at Vg = 0 and the thermionic heterostack at equilibrium. (The gate-G recipe with a
    raw ds.solve(absolute_error=1e18) re-settle gives the same digest on the modulator -- MEASURED; on an
    extended-precision device it would bypass the extended Newton, which the method does not.)"""
    assert runs["wrap_ref"]["w0"]["digest"] == runs["hand_ref"]["digest"]
    assert runs["wrap_te"]["w"]["digest"] == runs["hand_te"]["digest"]
    assert runs["wrap_ref"]["w0"]["units"] == "A^2/Hz/m per unit depth" and runs["wrap_ref"]["w0"]["depth"] == 1.0


def test_w2_fdt_through_the_method_per_cell(runs):
    """W2. S_I = 4 kT Re Y per cell (terminal_noise: S_I and Y both scaled by period_y) to 1e-9 on both
    fixtures (module doc). Adversarial: the per-cell S against the per-depth Re Y (a units slip) misses by
    period - 1 (the factor 3.7e-7)."""
    for sc in ("wrap_ref", "wrap_te"):
        r = _fdt(runs[sc]["tn"])
        assert np.max(np.abs(r)) < 1e-9, (sc, r)
    tn, hand = runs["wrap_ref"]["tn"], runs["hand_ref"]
    assert runs["wrap_ref"]["tn"]["units"] == "A^2/Hz per cell" and tn["S_V_ok"]
    slip = np.array(tn["S_I"]) / (4.0 * KB * T * np.array(hand["G"])) - 1.0
    assert np.all(np.abs(slip - (hand["period"] - 1.0)) < 1e-6 * hand["period"]) and np.all(np.abs(slip) > 0.99)


def test_w3_per_cell_is_the_period_times_the_per_depth_result(runs):
    """W3. per_cell=True multiplies S, every breakdown and every map by UnitCell.period_y_m: S_cell ==
    period * S_depth bitwise (the same multiplication), units "A^2/Hz per cell", and the scaled maps still sum
    to S (to 2.2e-16). vg defaults to the electrode's last converged bias (0 V here) and the result records
    every electrode's bias."""
    w, h = runs["wrap_ref"], runs["hand_ref"]
    wc = w["wc"]
    assert wc["units"] == "A^2/Hz per cell" and wc["depth"] == h["period"] == 370e-9
    assert all(a == b * h["period"] for a, b in zip(wc["S"], w["w0"]["S"]))
    assert wc["breakdown_exact"]
    assert np.max(np.abs(np.array(wc["map_sum"]) / np.array(wc["S"]) - 1.0)) < 1e-14
    assert wc["vg"] == 0.0 and wc["voltages"] == {"bot_contact": 0.0, "top_contact": 0.0, "ito_gnd_left": 0.0,
                                                  "ito_gnd_right": 0.0}


def test_w4_sequence_0_1_0_gives_a_bitwise_repeat_and_leaves_the_state(runs):
    """W4. Calls at 0, +1, 0 V: the third result is bitwise the first, the second differs. Run on the
    modulator with its BOTTOM gate held at +0.5 V: at the plain flat band (psi = 0, n = n_bg, exactly
    representable) any Newton lands on the same bits, so a 0/1/0 test there cannot tell (MEASURED: even the
    no-restore path repeats bitwise). Adversarial: the same sequence WITHOUT the method's state restore
    (settle at 0, ramp to +1 and back in its 0.25 V steps) does NOT repeat (MEASURED S3/S1 - 1 = -5.7e-12 at
    1 GHz). Also: a +1 V call in a FRESH process equals the second call (MEASURED bitwise; the plan's 1e-12 is
    asserted: the next test), and after a call every node solution is bitwise the pre-call state,
    the gate's circuit unknowns back at 0."""
    s = runs["seq_ref"]
    assert s["seq"][2] == s["seq"][0] and s["seq"][1] != s["seq"][0]
    nr = s["no_restore"]
    assert nr["digests"][0] == s["seq"][0]                    # same first call ...
    assert nr["digests"][2] != nr["digests"][0]               # ... but no bitwise repeat without the restore
    assert 0.0 < np.max(np.abs(np.array(nr["S_third_over_first"]) - 1.0)) < 1e-9
    assert runs["wrap_ref"]["w2"]["digest"] == runs["wrap_ref"]["w0"]["digest"]
    assert runs["wrap_ref"]["restored"]
    assert runs["wrap_ref"]["circuit_after"] == {"circuit:V1.I": [0.0], "circuit:vac": [0.0]}


@pytest.mark.slow
def test_w7_extended_precision_sequence_0_1_0_repeats_bitwise():
    """W7 (P2; v0.15 follow-up, nightly: MEASURED 4.1-4.3 min on the loaded Windows dev box -- the extended
    build + solve ~86 s, the +1 V call's extended re-settle ramp ~130 s). seq_ref's 0 / +1 / 0 V sequence (bottom
    gate +0.5 V, so the 0 V state is not the exactly representable flat band) on a builder with
    extended_precision=True: the third result is bitwise the first and the second differs. Not implied by W4:
    the method's restore writes the double-rounded state back into a device whose Newton runs in 128 bits, so
    the third call's re-settle starts from a different representation than the first's (it still lands on the
    same bits: MEASURED). The only warnings are the extended-precision 'noise solve buys nothing' advisory, one
    per frequency per call (P2 W7: the method adds none of its own), and none at the build. Sensitivity: the
    +1 V digest differs from the 0 V one, and the device really is extended (both asserted)."""
    r = _run(("seq_xp",))["seq_xp"]
    assert r["extended"] == [True, True]
    assert r["seq"][2] == r["seq"][0] and r["seq"][1] != r["seq"][0]
    assert r["warnings_build"] == []
    calls = r["warnings_calls"]
    assert len(calls) == 3 * 4 and all(c == "AdvisoryWarning" and "extended_precision: a 'noise' solve" in m
                                       for c, m in calls), calls


def test_w4_a_fresh_process_reproduces_the_second_call(runs):
    """W4, second half. A +1 V call in a FRESH process (the builder solved at
    0, then the method at +1 V) equals the +1 V call of the 0/+1/0 sequence to 1e-12 (the plan's bound for
    different continuation paths; MEASURED bitwise, the paths being the same once the state is restored)."""
    w1, f1 = np.array(runs["wrap_ref"]["w1"]["S"]), np.array(runs["fresh1_ref"]["w1"]["S"])
    assert np.max(np.abs(f1 / w1 - 1.0)) < 1e-12


def test_w5_solve_after_the_method_equals_a_fresh_solve(runs):
    """W5. After the method (and its refusals), builder.solve(top_contact = 1 V) equals a fresh builder
    solved at 1 V (no circuit) to 1e-9 in every node Potential (absolute V) and Electrons (relative):
    MEASURED 0.0 -- bitwise. The circuit node follows (vac = 1 V). (The stale-solve defect itself is fixed
    and gated in tests/test_gated_builder.py.)"""
    w, ref = runs["wrap_ref"], runs["plain_1V"]["fresh_1V"]
    dev = {k: max(abs(a - b) for a, b in zip(w["solve_1V"][k], ref[k]))
           / (max(abs(b) for b in ref[k]) if k.endswith("Electrons") else 1.0) for k in ref}
    assert len(dev) >= 6 and max(dev.values()) < 1e-9, dev
    assert w["solve_1V_voltages"] == {"top_contact": 1.0}
    assert w["solve_1V"]["circuit:vac"] == [1.0]


def test_w6_refusals(runs):
    """W6. An OHMIC carrier contact raises OhmicContactNoiseError (a NotImplementedError; decision 35); the
    gate under a second source name, and another gate under a source / node name already in the process-
    global circuit, raise ValueError -- the latter BEFORE the bottom gate's bias Dirichlet is removed (it
    still has its PotentialEquation); an unknown `hooge` string raises ValueError. None of them moves the
    device."""
    w = runs["wrap_ref"]
    assert w["ohmic"].startswith("OhmicContactNoiseError:") and "ito_gnd_left" in w["ohmic"]
    assert w["second_source"].startswith("ValueError:") and "already circuit-driven" in w["second_source"]
    assert w["name_clash"].startswith("ValueError:") and "already exist" in w["name_clash"]
    assert w["bot_contact_eqs"] == ["PotentialEquation"]
    assert w["bad_hooge"].startswith("ValueError:")
    assert w["after_refusals"]


def test_w6_unbuilt_builder_refuses():
    """The method needs a built AND solved builder (RuntimeError); nothing is built here (no DEVSIM state)."""
    from validation._reference_device import build_reference_modulator
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder, OhmicContactNoiseError
    b = LayeredDevsimBuilder(build_reference_modulator("drift_diffusion"), mesh_name="nm_unb", device_name="nm_unb")
    with pytest.raises(RuntimeError, match="built AND solved"):
        b.noise_spectrum([1e9], gate="top_contact")
    with pytest.raises(RuntimeError, match="built AND solved"):
        b.terminal_noise([1e9], gate="top_contact", hooge="material")
    assert issubclass(OhmicContactNoiseError, NotImplementedError)


def test_default_mechanisms_are_inherited_from_noise_analysis(runs):
    """The [C] gate. With mechanisms=None (the default) the method uses noise_analysis.DEFAULT_MECHANISMS, so
    the thermionic interface's te_n / te_p (default-on since v0.15, decision 43) are in the result
    (by_mechanism and by_interface['gaas__algaas']) and the FDT closes (W2). Adversarial: the plan's P2
    literal default ('diffusion_n', 'diffusion_p', 'gr') silently drops them and misses the FDT by -0.80
    (1 GHz) / -0.056 (100 GHz) -- the interface's share."""
    w = runs["wrap_te"]
    assert runs["hand_te"]["ifaces"] == {"gaas__algaas": "thermionic"}
    assert {"te_n", "te_p"} <= set(w["w"]["mechanisms"]) and set(w["w"]["by_interface"]) == {"gaas__algaas"}
    te = np.array(w["w"]["by_interface"]["gaas__algaas"]["te_n"])
    assert np.all(te > 0.05 * np.array(w["w"]["S"]))
    miss = _fdt(w["tn_plan_literal_default"])
    assert np.all(miss < -0.05), miss


def test_default_mechanisms_are_read_at_call_time(monkeypatch):
    """Structural half of the inheritance gate (no DEVSIM state): the method passes noise_analysis.
    DEFAULT_MECHANISMS as it is AT CALL TIME (a later default-on source, e.g. the trap keys, propagates
    without touching the builder), and an explicit list unchanged."""
    from validation._reference_device import build_reference_modulator
    from dynameta.carriers import devsim_layered as L
    from dynameta.carriers import noise_analysis as NA
    seen = []
    monkeypatch.setattr(NA, "DEFAULT_MECHANISMS", ("diffusion_n", "sentinel_mechanism"))
    monkeypatch.setattr(NA, "noise_spectrum", lambda dev, f, **kw: seen.append(kw["mechanisms"]) or "sn")
    b = L.LayeredDevsimBuilder(build_reference_modulator("drift_diffusion"), mesh_name="nm_spy", device_name="nm_spy")
    monkeypatch.setattr(b, "_at_gate_bias", lambda gate, vg, *a: (a[-1](), 0.0, {}))
    monkeypatch.setattr(L, "_scaled_noise", lambda sn, *a: sn)
    b.noise_spectrum([1e9], gate="top_contact")
    b.noise_spectrum([1e9], gate="top_contact", mechanisms=["gr"])
    assert seen == [("diffusion_n", "sentinel_mechanism"), ("gr",)]


def test_w8_hooge_material_equals_the_explicit_mapping(runs):
    """W8 / H4. hooge='material' resolves each carrier region's TransportModel.hooge_alpha_n / _p: on the
    heterostack (GaAs (1e-4, 3e-5), AlGaAs (2e-5, None)) it is bitwise the explicit mapping, and AlGaAs's
    hooge_p is exactly 0. On the modulator, whose ITO sets no alpha, 'material' is NO Hooge at all: bitwise
    the default call, no hooge_* key. (A dielectric gate draws no DC current, so the Hooge values themselves
    are those of the round-off equilibrium current, ~1e-56 here.)"""
    h = runs["wrap_te"]["hooge"]
    assert h["material"] == h["explicit"] and h["keys"] == ["hooge_n", "hooge_p"] and h["algaas_p_zero"]
    hm = runs["wrap_ref"]["hooge_material_none"]
    assert hm["digest"] == runs["wrap_ref"]["w0"]["digest"]
    assert not any(m.startswith("hooge") for m in hm["mechanisms"])
