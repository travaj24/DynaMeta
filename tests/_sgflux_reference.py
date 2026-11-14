"""Cross-version NUMERIC guard for the defaults-off definition fixtures (tests/test_devsim_sg_flux.py).

The call-log digests of tests/_bipolar_calllog.py and tests/_dg_calllog.py pin the physics DEFINITION
character-for-character; the two-Bernoulli SG flux (2026-11-04) changed it on purpose, so they were
re-recorded. This module keeps the guard's purpose -- an unintended physics change must fail -- by
SOLVING the same fixtures and comparing every field and terminal current with the arrays recorded on
ee0ad04 (the last kahan3 tree; tests/data/sgflux_6595bde.json), at a tolerance set by the measured
round-off of the flux change (TOL below).

Fixtures (the calllog meshes, materials and setup calls, then a solve; public API only, so the same
code runs on ee0ad04):
  fd   cl_fd: Si p+n diode (-1e24 | 3e25), FD g-factor, DC ohmic contacts; 0 -> +0.4 V forward and
       -0.5 V reverse on 'lo'.
  bz   cl_bz: the same diode, Boltzmann, 'hi' on a circuit node (source Vcl); hi at -0.3 V (forward)
       and +0.5 V (reverse), source current read from the circuit.
  dg   dgcl: the unipolar in-Newton DG ITO bar (4e26), DG ohmic 'back', the pre-v0.15 setup_dg_hard_wall
       rows on 'wall' (since v0.15 the test fixture tests/_dg_legacy_wall.py; the library name is a shim
       onto setup_dg_insulator_wall), wall-aware taper seed, gamma ramp to 1.

Re-record (only on purpose, from a checkout of the reference code):
    PYTHONPATH=<reference tree> python tests/_sgflux_reference.py --record
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "sgflux_6595bde.json")

# TOLERANCES, from the change's measured round-off (2026-11-04: ee0ad04 vs this tree, same process
# recipe, scipy OpenBLAS, devsim 2.11.2; per-array numbers in docs/audit/2026-11-04-sg-flux-two-
# bernoulli.md sec. 4):
#  * densities (Electrons, Holes, QSqrtN), entry-wise |new/old - 1|: worst 7.7e-15 (fd_rev:Electrons);
#  * potentials (Potential, QLambda; arbitrary zero), max |new - old| / max |old|: worst 1.9e-16;
#    -> TOL_FIELD = 1e-13 (13x the worst; a 1e-6 change of mu_n moves fd_rev's electrons by 4e-7).
#  * terminal currents: |new - old| <= TOL_I_REL |I_old| + TOL_I_TERM T, T = q mu V_t c / h of the
#    contact edge (the size of EACH of its drift and diffusion terms, recorded as '<key>:T'). A
#    MAJORITY-carrier contact current is the round-off of two such terms: fd's hole current at the
#    p-contact (T = 4.0e10 A/m^2) moved 0.15 / 0.24 eps T (fwd / rev; 2.1e-5 / 5.4e-2 RELATIVE) --
#    neither flux spelling is "right" there, both are eps T noise; the worst entry is 0.43 eps T
#    (fd_rev electrons, 1.1e-15 relative). Every other current moved <= 6.5e-13 relative (bz_rev).
#    bz's current is the circuit source current, a Newton UNKNOWN (its KCL row), not a
#    get_contact_current read-back, and shows no eps T floor (moved 4e-17 A/m^2 with T ~ 3e12 A/m^2
#    at 'hi'), so it is gated relative only (T = 0). TOL_I_REL = 1e-11 (15x), TOL_I_TERM = 4 eps =
#    8.9e-16 (9x the worst 0.43 eps).
TOL_FIELD = 1e-13
TOL_I_REL = 1e-11
TOL_I_TERM = 4.0 * 2.220446049250313e-16
_DENSITIES = ("Electrons", "Holes", "QSqrtN")
_POTENTIALS = ("Potential", "QLambda")
N_COMPARED = 21                                    # arrays + currents gated (the :T scales are not)


def _quiet():
    import contextlib

    @contextlib.contextmanager
    def q():
        sys.stdout.flush()
        saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(devnull)
            os.close(saved)
    return q()


def _perturb(ds, dev, region):
    """SGFLUX_PERTURB=mu_n: multiply the region's mu_n by (1 + 1e-6) -- the sensitivity check."""
    if os.environ.get("SGFLUX_PERTURB") == "mu_n":
        mu = float(ds.get_parameter(device=dev, region=region, name="mu_n"))
        ds.set_parameter(device=dev, region=region, name="mu_n", value=mu * (1.0 + 1e-6))


def solve_fixtures(statistics=None):
    """`statistics` (v0.15.0; default: the SGFLUX_STATISTICS environment variable, else None): None passes no
    keyword (the tree's DEFAULT; on a pre-v0.15.0 tree the legacy statistics); a value goes to the FD
    fixture's setup_bipolar_region and the DG bar's setup_semiconductor_region_dd -- 'legacy' is the path the
    ee0ad04 arrays pin (the v0.15.0 default, 'fermi_dirac', moves fd and dg by physics: audit note
    2026-11-06-v0.15-fermi-dirac-mode-complete.md, section 10)."""
    import numpy as np
    if statistics is None:
        statistics = os.environ.get("SGFLUX_STATISTICS") or None
    st = {} if statistics is None else {"statistics": statistics}
    try:
        from dynameta.carriers.devsim_env import import_devsim
        ds = import_devsim()
    except ImportError:                                 # a tree without devsim_env (never ee0ad04)
        import devsim as ds
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP

    out = {}

    def nv(dev, reg, name):
        return [float(x) for x in ds.get_node_model_values(device=dev, region=reg, name=name)]

    def term(dev, reg, end, *carriers):
        """T = q V_t / h * sum(mu c) at the contact node of the 1-D device's min-x (end=0) or max-x
        (end=-1) end: the magnitude of each drift / diffusion term of that contact edge's flux."""
        x = np.asarray(nv(dev, reg, "x"))
        o = np.argsort(x)
        i, j = (o[0], o[1]) if end == 0 else (o[-1], o[-2])
        par = (lambda nm: float(ds.get_parameter(device=dev, region=reg, name=nm)))
        return (par("ElectronCharge") * par("V_t") / abs(x[j] - x[i])
                * sum(par(mu) * nv(dev, reg, c)[i] for mu, c in carriers))

    def solve(ae, re_, it=80):
        with _quiet():
            ds.solve(type="dc", absolute_error=ae, relative_error=re_, maximum_iterations=it)

    # ---- fd / bz: the _bipolar_calllog fixtures ----
    for dev, fd, circuit in (("cl_fd", True, False), ("cl_bz", False, True)):
        mesh = dev + "_m"
        _R.clear(dev)
        with _quiet():
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=5e-9, tag="lo")
            ds.add_1d_mesh_line(mesh=mesh, pos=100e-9, ps=1e-9)
            ds.add_1d_mesh_line(mesh=mesh, pos=200e-9, ps=5e-9, tag="hi")
            ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
            ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="bulk", tag1="lo", tag2="hi")
            ds.finalize_mesh(mesh=mesh)
            ds.create_device(mesh=mesh, device=dev)
            ds.node_model(device=dev, region="bulk", name="NetDoping",
                          equation="ifelse(x < 1e-7, -1e24, 3e25)")
            BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=2.8e25, n_i_m3=1.0e16,
                                    mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7,
                                    tau_p_s=2e-7, fd_enhancement=fd, n_dos_p_m3=1.04e25,
                                    **(st if fd else {}))
            BP.setup_equilibrium_seed_models(dev, "bulk")
            BP.setup_contact_ohmic_bipolar(dev, "lo")
            if circuit:
                BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="cl_n", source_name="Vcl")
            else:
                BP.setup_contact_ohmic_bipolar(dev, "hi")
            ds.node_model(device=dev, region="bulk", name="_seed_psi",
                          equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "bulk")))
            for v, m in (("Potential", "_seed_psi"), ("Electrons", "IntrinsicElectrons"),
                         ("Holes", "IntrinsicHoles")):
                ds.set_node_values(device=dev, region="bulk", name=v,
                                   values=nv(dev, "bulk", m))
        _perturb(ds, dev, "bulk")
        # equilibrium at REL 1e-6: the circuit node's source current is ~0 there and its relative
        # update dithers at 2e-8 - 2e-7 on round-off in BOTH flux forms (measured 2026-11-04)
        solve(1e18, 1e-6)
        tag = "fd" if fd else "bz"
        if circuit:
            # never back to exactly 0 V at REL 1e-8: the ~0 source current dithers (see above)
            steps = (("fwd", [-0.1, -0.2, -0.3]), ("rev", [-0.2, -0.1, 0.1, 0.25, 0.5]))
        else:
            steps = (("fwd", [0.1, 0.2, 0.3, 0.4]), ("rev", [0.2, 0.0, -0.25, -0.5]))
        for lab, vs in steps:
            for vb in vs:
                if circuit:
                    ds.circuit_alter(name="Vcl", value=vb)
                else:
                    ds.set_parameter(device=dev, name="lo_bias", value=vb)
                solve(1e18, 1e-8)
            key = "{}_{}".format(tag, lab)
            for f in ("Potential", "Electrons", "Holes"):
                out["{}:{}".format(key, f)] = nv(dev, "bulk", f)
            if circuit:                       # Vcl.I: electrons + holes through 'hi' (max x)
                out[key + ":I"] = [float(ds.get_circuit_node_value(solution="dcop", node="Vcl.I"))]
                out[key + ":T"] = [0.0]       # a Newton UNKNOWN, not a flux read-back: no eps T floor
            else:                             # per carrier through 'lo' (min x)
                out[key + ":I"] = [float(ds.get_contact_current(device=dev, contact="lo", equation=e))
                                   for e in ("ElectronContinuityEquation", "HoleContinuityEquation")]
                out[key + ":T"] = [term(dev, "bulk", 0, ("mu_n", "Electrons")),
                                   term(dev, "bulk", 0, ("mu_p", "Holes"))]
        if not circuit:
            with _quiet():
                ds.delete_device(device=dev)
                ds.delete_mesh(mesh=mesh)
            _R.clear(dev)

    # ---- dg: the _dg_calllog fixture ----
    from dynameta.constants import M_E
    from dynameta.carriers import physics_density_gradient as DG
    from dynameta.carriers.physics_drift_diffusion import (setup_contact_ohmic_dd,
                                                           setup_semiconductor_region_dd)
    sys.path.insert(0, HERE)
    import _dg_legacy_wall as LW
    dev, mesh = "dgcl", "dgcl_m"
    _R.clear(dev)
    with _quiet():
        ds.create_1d_mesh(mesh=mesh)
        ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=1e-9, tag="wall")
        ds.add_1d_mesh_line(mesh=mesh, pos=20e-9, ps=2e-9, tag="back")
        ds.add_1d_contact(mesh=mesh, name="wall", tag="wall", material="metal")
        ds.add_1d_contact(mesh=mesh, name="back", tag="back", material="metal")
        ds.add_1d_region(mesh=mesh, material="ITO", region="bar", tag1="wall", tag2="back")
        ds.finalize_mesh(mesh=mesh)
        ds.create_device(mesh=mesh, device=dev)
        mstar = 0.35 * M_E
        setup_semiconductor_region_dd(dev, "bar", n_bg_m3=4.0e26, eps_static=9.5,
                                      dos_mass_kg=mstar, mobility_m2Vs=0.004, **st)
        setup_contact_ohmic_dd(dev, "back")
        DG.setup_dg_quantum_correction(dev, "bar", m_eff_kg=mstar, gamma=1.0)
        DG.setup_contact_dg(dev, "back", 4.0e26)
        # the RETIRED hard-wall rows (v0.15: setup_dg_hard_wall is a shim onto the insulator wall),
        # re-created call-for-call by the test fixture, so this flux gate still compares with the
        # arrays recorded on ee0ad04 WITH those rows (tests/_dg_legacy_wall.py)
        LW.install_legacy_hard_wall(dev, "wall", lambda_pin_factor=8.0)
        DG.seed_dg_from_solution(dev, "bar")
        z0 = np.asarray(nv(dev, "bar", "x"))
        u0 = np.asarray(nv(dev, "bar", "QSqrtN"))
        taper = np.maximum(np.tanh(z0 / 1.2e-9), 1e-6)       # wall-aware seed (_hard_wall_bar)
        ds.set_node_values(device=dev, region="bar", name="QSqrtN", values=list(u0 * taper))
        ds.set_node_values(device=dev, region="bar", name="Electrons",
                           values=list(np.maximum((u0 * taper) ** 2, 1e14)))
    _perturb(ds, dev, "bar")
    for fr in (0.05, 0.1, 0.25, 0.5, 1.0):
        DG.set_dg_gamma(dev, "bar", fr)
        LW.set_pin_fraction(dev, fr)          # the old co-ramp (set_dg_gamma's legacy branch, same value)
        solve(1e16, 1e-8, 200)
    for f in ("Potential", "Electrons", "QSqrtN", "QLambda"):
        out["dg:" + f] = nv(dev, "bar", f)
    out["dg:I"] = [float(ds.get_contact_current(device=dev, contact="back",
                                                equation="ElectronContinuityEquation"))]
    out["dg:T"] = [term(dev, "bar", -1, ("mu_n", "Electrons"))]
    return out


def load_reference():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)["arrays"]


def compare(ref, new):
    """[(key, measure, tolerance, ok)] for every gated array (module TOLERANCES): densities entry-wise
    relative, potentials relative to the array's max |old|, currents as max |dI| / (TOL_I_REL |I| +
    TOL_I_TERM T) against 1."""
    rows = []
    for key in sorted(k for k in ref if not k.endswith(":T")):
        a, b = ref[key], new.get(key)
        if b is None or len(a) != len(b):
            rows.append((key, math.inf, 0.0, False))
            continue
        f = key.split(":")[1]
        if f == "I":
            m = max(abs(y - x) / (TOL_I_REL * abs(x) + TOL_I_TERM * t)
                    for x, y, t in zip(a, b, ref[key[:-1] + "T"]))
            tol = 1.0
        elif f in _DENSITIES:
            m, tol = max(abs(y / x - 1.0) for x, y in zip(a, b)), TOL_FIELD
        elif f in _POTENTIALS:
            m, tol = max(abs(y - x) for x, y in zip(a, b)) / max(abs(x) for x in a), TOL_FIELD
        else:
            raise KeyError(key)
        rows.append((key, m, tol, m <= tol))
    return rows


if __name__ == "__main__" and sys.argv[1:2] == ["--emit"]:       # child of the test: --emit <out.json>
    res = solve_fixtures()
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(res, f)
elif __name__ == "__main__" and sys.argv[1:2] == ["--record"]:  # pragma: no cover - re-recording aid
    import subprocess
    import dynameta
    res = solve_fixtures()
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(
            os.path.dirname(dynameta.__file__)), capture_output=True, text=True).stdout.strip()
    except OSError:
        head = "?"
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as f:
        json.dump({"tree": head, "dynameta": dynameta.__file__, "arrays": res}, f)
    print("recorded", len(res), "arrays from", head, "->", DATA)
