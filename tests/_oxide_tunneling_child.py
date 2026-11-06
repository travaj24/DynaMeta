"""Child-process scenarios for tests/test_devsim_oxide_tunneling_sc.py (DynaMeta v0.15, plan item 4.1.1).

WHY A CHILD. The self-consistent oxide tunnelling is a custom term (carriers/custom_terms.py): its DEVSIM
dispatcher can never be unregistered, and the circuit-driven gates create DEVSIM circuits, which cannot be
removed in-process either (tests/_bipolar_calllog.py). Every scenario therefore runs in a fresh interpreter
and reports plain JSON (floats round-trip exactly through repr).

    python tests/_oxide_tunneling_child.py <scenario> <out.json> [json options]

The devices are 1-D MOS capacitors (semiconductor | dielectric layer(s) | metal gate) built by hand with the
library's bipolar drift-diffusion and dielectric physics. The tests re-derive every oracle; this module only
builds, solves and reads arrays.
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

Q = 1.602176634e-19
KB = 1.380649e-23
EPS0 = 8.8541878128e-12
VT = KB * 300.0 / Q
NI, NDOS = 1.0e16, 2.8e25
MU_N, MU_P = 0.1, 0.04
EPS_SI = 11.7
EQN, EQP = "ElectronContinuityEquation", "HoleContinuityEquation"


def _ds():
    from dynameta.carriers.devsim_env import import_devsim
    return import_devsim()


def _quiet():
    import validation.devsim_noise as V
    return V.quiet()


def _lst(a):
    return [float(x) for x in np.ravel(a)]


def _tensor_mesh(mesh, dim, xs, L, W, n_lat):
    """A laterally uniform 2-D (DEVSIM rectilinear triangles) or 3-D (Kuhn tetrahedra, in memory) mesh whose
    normal-axis node set is EXACTLY xs (semiconductor [0, L], oxide [L, xs[-1]]), lateral width W with n_lat
    nodes per lateral axis. Regions semi / oxide, contacts body (x = 0) / gate (top), interface semi_oxide.
    In 2-D the normal axis is x (DEVSIM lines with ps = the local spacing: no inserted points); in 3-D it is z."""
    ds = _ds()
    top = float(xs[-1])
    if dim == 2:
        ds.create_2d_mesh(mesh=mesh)
        pad = 10.0 * float(np.min(np.diff(xs)))         # DEVSIM 2-D contacts are boxes of mesh OUTSIDE the
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=-pad, ps=pad)    # region they touch: inert pads
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=top + pad, ps=pad)
        for i, x in enumerate(xs):
            ps = min(xs[i] - xs[i - 1] if i > 0 else np.inf, xs[i + 1] - xs[i] if i + 1 < xs.size else np.inf)
            ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=float(x), ps=float(ps))
        for y in np.linspace(0.0, W, n_lat):
            ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=float(y), ps=W / (n_lat - 1))
        b = 1e-3 * float(np.min(np.diff(xs)))
        ds.add_2d_region(mesh=mesh, material="Si", region="pad_lo", xl=-pad, xh=0.0, yl=0.0, yh=W, bloat=b)
        ds.add_2d_region(mesh=mesh, material="metal", region="pad_hi", xl=top, xh=top + pad, yl=0.0, yh=W,
                         bloat=b)
        ds.add_2d_region(mesh=mesh, material="Si", region="semi", xl=0.0, xh=L, yl=0.0, yh=W, bloat=b)
        ds.add_2d_region(mesh=mesh, material="SiO2", region="oxide", xl=L, xh=top, yl=0.0, yh=W, bloat=b)
        ds.add_2d_interface(mesh=mesh, name="semi_oxide", region0="semi", region1="oxide", xl=L, xh=L, yl=0.0,
                            yh=W, bloat=b)
        ds.add_2d_contact(mesh=mesh, name="body", material="metal", region="semi", xl=0.0, xh=0.0, yl=0.0, yh=W,
                          bloat=b)
        ds.add_2d_contact(mesh=mesh, name="gate", material="metal", region="oxide", xl=top, xh=top, yl=0.0,
                          yh=W, bloat=b)
        ds.finalize_mesh(mesh=mesh)
        return
    from dynameta.carriers.devsim_3d import _face_triangles, _kuhn_tets, create_mesh_in_memory
    lat = np.linspace(0.0, W, n_lat)
    ax = (lat, lat, xs)
    P, T, cell = _kuhn_tets(*ax)
    cz = 0.5 * (xs[1:] + xs[:-1])
    nx = ny = n_lat - 1
    rid = np.zeros((nx, ny, cz.size), dtype=int)
    rid[:, :, cz > L] = 1
    tet_rid = rid.ravel()[cell]
    k_if = int(np.nonzero(cz < L)[0].max())
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    I, J = I.ravel(), J.ravel()

    def layer(k):
        return (I, J, np.full(I.size, k))
    tri = {"body": _face_triangles(ax, 2, layer(0), plus=False),
           "gate": _face_triangles(ax, 2, layer(cz.size - 1), plus=True),
           "semi_oxide": _face_triangles(ax, 2, layer(k_if), plus=True)}
    create_mesh_in_memory(mesh, P, {"semi": T[tet_rid == 0], "oxide": T[tet_rid == 1]}, tri)
    ds.add_gmsh_region(mesh=mesh, gmsh_name="semi", region="semi", material="Si")
    ds.add_gmsh_region(mesh=mesh, gmsh_name="oxide", region="oxide", material="SiO2")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="body", region="semi", name="body", material="metal")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="gate", region="oxide", name="gate", material="metal")
    ds.add_gmsh_interface(mesh=mesh, gmsh_name="semi_oxide", region0="semi", region1="oxide", name="semi_oxide")
    ds.finalize_mesh(mesh=mesh)


class Dev:
    """A 1-D MOS (or its laterally uniform 2-D / 3-D tensor-mesh twin) and its gate drive."""

    def __init__(self, name, *, doping=-1e23, L=1.0e-6, t_ox=2e-9, layers=None, gate_offset=0.0,
                 circuit=False, fd=False, tau=1e-6, h_if=0.05e-9, h_body=20e-9, h_ox=0.1e-9, eps_ox=3.9,
                 source=None, node=None, dummy_circuit=False, v_ref=0.0, i_park=None, extended=False,
                 dim=1, x_nodes=None, width=4e-9, n_lat=3):
        ds = _ds()
        from dynameta.carriers import eq_registry as _R
        from dynameta.carriers import physics_bipolar_dd as BP
        from dynameta.carriers import physics_equilibrium as PE
        self.name, self.circuit, self.v = name, circuit, 0.0
        self.source = source or ("V" + name)
        self.node = node or ("n" + name)
        self.layers = layers if layers is not None else [("oxide", t_ox, eps_ox)]
        self.doping, self.tau, self.fd = doping, tau, fd
        dev, mesh = name, name + "_m"
        _R.clear(dev)
        with _quiet():
          if dim == 1:
            ds.create_1d_mesh(mesh=mesh)
            ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=h_body, tag="body")
            # an EXACTLY uniform h_if zone over the 10 nm next to the interface (the accumulation / inversion
            # layer), one mesh line per node: DEVSIM's 1-D grading leaves an odd-sized edge where graded runs
            # meet -- MEASURED: right at the interface, 2-9 h long, and it alone set the FD equilibrium error
            nz = int(round(10e-9 / h_if))
            for k in range(nz, 0, -1):
                ds.add_1d_mesh_line(mesh=mesh, pos=L - k * h_if, ps=h_if)
            ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=h_if, tag="if0")
            x = L
            for k, (reg, t, _) in enumerate(self.layers):
                x += t
                ds.add_1d_mesh_line(mesh=mesh, pos=x, ps=h_ox, tag="if{}".format(k + 1))
            ds.add_1d_contact(mesh=mesh, name="body", tag="body", material="metal")
            ds.add_1d_contact(mesh=mesh, name="gate", tag="if{}".format(len(self.layers)), material="metal")
            ds.add_1d_region(mesh=mesh, material="Si", region="semi", tag1="body", tag2="if0")
            prev = "semi"
            for k, (reg, t, _) in enumerate(self.layers):
                ds.add_1d_region(mesh=mesh, material="SiO2" if reg.startswith("ox") else reg, region=reg,
                                 tag1="if{}".format(k), tag2="if{}".format(k + 1))
                ds.add_1d_interface(mesh=mesh, name="{}_{}".format(prev, reg), tag="if{}".format(k))
                prev = reg
            ds.finalize_mesh(mesh=mesh)
          else:                          # 2-D / 3-D tensor mesh on the SAME normal node set (one oxide)
            _tensor_mesh(mesh, dim, np.asarray(x_nodes, dtype=float), L, width, n_lat)
          with _quiet():
            ds.create_device(mesh=mesh, device=dev)
            if extended:                 # 128-bit models + equations (before any physics)
                from dynameta.carriers import extended_precision as XP
                XP.enable(dev)
            ds.node_model(device=dev, region="semi", name="NetDoping", equation="{:.6e}".format(doping))
            BP.setup_bipolar_region(dev, "semi", eps_static=EPS_SI, n_dos_m3=NDOS, n_i_m3=NI,
                                    mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=tau, tau_p_s=tau,
                                    fd_enhancement=fd)
            BP.setup_equilibrium_seed_models(dev, "semi")
            prev = "semi"
            for reg, t, eps in self.layers:
                PE.setup_dielectric_region(dev, reg, eps)
                PE.setup_interface(dev, "{}_{}".format(prev, reg))
                prev = reg
            BP.setup_contact_ohmic_bipolar(dev, "body")
            if dummy_circuit:            # get_matrix_and_rhs ASSERTS (NodeKeeper.cc:266) with no circuit at all
                ds.add_circuit_node(name="ndum_" + name, variable_update="default")
                ds.circuit_element(name="VDUM_" + name, n1="ndum_" + name, n2="0", value=1.0)
            if circuit:                  # ac_analysis.setup_circuit_contact's rule, plus a work-function offset
                ds.add_circuit_node(name=self.node, variable_update="default")
                ds.circuit_element(name=self.source, n1=self.node, n2="0", value=float(v_ref), acreal=1.0,
                                   acimag=0.0)
                if i_park is not None:
                    # MEASURED: DEVSIM's RELATIVE criterion on a circuit unknown at ~0 (an equilibrium gate's
                    # source current) never converges (round-off 1e-14 over |x| ~ 0). A DC current source in
                    # parallel parks the branch current away from 0; it is an open circuit for ssac / noise.
                    ds.circuit_element(name="I" + self.source, n1=self.node, n2="0", value=float(i_park))
                cn = "gate_circuit_dirichlet"
                ds.contact_node_model(device=dev, contact="gate", name=cn,
                                      equation="Potential - {} - ({!r})".format(self.node, float(gate_offset)))
                ds.contact_node_model(device=dev, contact="gate", name=cn + ":Potential", equation="1")
                ds.contact_node_model(device=dev, contact="gate", name=cn + ":" + self.node, equation="-1")
                _R.record_contact_equation(dev, "gate", name="PotentialEquation", node_model=cn,
                                           edge_charge_model="PotentialEdgeFlux", circuit_node=self.node)
            else:
                PE.setup_contact(dev, "gate", psi_offset_V=gate_offset)
            ds.node_model(device=dev, region="semi", name="_seed", equation=BP.equilibrium_seed_psi_expr(None))
            for v, m in (("Potential", "_seed"), ("Electrons", "IntrinsicElectrons"),
                         ("Holes", "IntrinsicHoles")):
                ds.set_node_values(device=dev, region="semi", name=v,
                                   values=ds.get_node_model_values(device=dev, region="semi", name=m))
        self.regions = ["semi"] + [r for r, _, _ in self.layers]
        # start with the gate at the seed's potential (bias = -offset) and continue to the reference bias
        # v_ref (body and gate both at v_ref = equilibrium; a circuit node at exactly 0 V never meets the
        # relative criterion either, so the equilibrium circuit scenarios sit at v_ref != 0)
        self.v_ref = float(v_ref)
        if self.v_ref != 0.0:
            with _quiet():
                ds.set_parameter(device=dev, name="body_bias", value=0.0)
        self.v = -float(gate_offset)
        self.set_gate(self.v)
        solve()
        if self.v_ref != 0.0:              # body and gate TOGETHER by v_ref first (the field does not change)
            for t in np.linspace(0.0, 1.0, 6)[1:]:
                ds.set_parameter(device=dev, name="body_bias", value=t * self.v_ref)
                self.set_gate(-float(gate_offset) + t * self.v_ref)
                solve()
            self.v = self.v_ref - float(gate_offset)
        self.goto(self.v_ref)

    # ---- drive / solve ----
    def set_gate(self, V):
        ds = _ds()
        if self.circuit:
            ds.circuit_alter(name=self.source, value=float(V))
        else:
            ds.set_parameter(device=self.name, name="gate_bias", value=float(V))

    def _snapshot(self):
        ds = _ds()
        snap = {}
        for r in self.regions:
            for nm in ("Potential", "Electrons", "Holes"):
                if nm in ds.get_node_model_list(device=self.name, region=r):
                    snap[(r, nm)] = ds.get_node_model_values(device=self.name, region=r, name=nm)
        return snap

    def _restore(self, snap):
        ds = _ds()
        for (r, nm), vals in snap.items():
            ds.set_node_values(device=self.name, region=r, name=nm, values=vals)

    def goto(self, target, *, dv_max=0.1):
        """Continuation to `target` in steps <= dv_max, halving a step whose Newton fails."""
        v0, step = self.v, dv_max
        while abs(target - v0) > 1e-15:
            v1 = v0 + max(-step, min(step, target - v0))
            snap = self._snapshot()
            try:
                self.set_gate(v1)
                solve()
                v0 = self.v = v1
                step = min(dv_max, 1.5 * step)
            except Exception as e:                       # noqa: BLE001
                if os.environ.get("OXT_DEBUG"):
                    print("goto {} -> {} failed: {!r}".format(v0, v1, e), file=sys.stderr)
                self._restore(snap)
                self.set_gate(v0)
                step *= 0.5
                if step < 1e-5:
                    raise
        solve()

    # ---- read-outs ----
    def body_current(self):
        ds = _ds()
        return sum(float(ds.get_contact_current(device=self.name, contact="body", equation=e))
                   for e in (EQN, EQP))

    def source_current(self):
        ds = _ds()
        return float(ds.get_circuit_node_value(node=self.source + ".I", solution="dcop"))

    def nv(self, region, name):
        return np.asarray(_ds().get_node_model_values(device=self.name, region=region, name=name),
                          dtype=np.float64)

    def close(self):
        """Delete the device and its mesh (DEVSIM solves EVERY live device together)."""
        ds = _ds()
        from dynameta.carriers import eq_registry as _R
        ds.delete_device(device=self.name)
        ds.delete_mesh(mesh=self.name + "_m")
        _R.clear(self.name)

    def residual_floor(self, eqs=(EQN, EQP)):
        """sum |F| over the semiconductor's interior rows of `eqs` (and the gate KCL row) at the converged
        state (get_matrix_and_rhs returns rhs = -F): the exact bound on the imbalance of the currents those
        rows balance (their signed sum telescopes to it)."""
        from dynameta.carriers import custom_terms as CT
        from dynameta.carriers.noise_analysis import _contact_nodes
        rhs = matrix()[3]
        n = self.nv("semi", "x").size
        cn = set(int(i) for c in ("body", "gate") for i in _contact_nodes(self.name, "semi", c))
        inner = np.array([i for i in range(n) if i not in cn])
        rows = np.concatenate([CT.node_rows(self.name, "semi", e, inner) for e in eqs])
        tot = float(np.sum(np.abs(rhs[rows])))
        if self.circuit:
            tot += abs(float(rhs[CT.circuit_row(self.node)]))
        return tot


def solve(ae=1e14, re_=1e-12, it=80):
    """DC Newton (DEVSIM restores the previous solution when it fails)."""
    ds = _ds()
    with _quiet():
        info = ds.solve(type="dc", absolute_error=ae, relative_error=re_, maximum_iterations=it, info=True)
    if not info["converged"]:
        last = info["iterations"][-1] if info["iterations"] else {}
        raise RuntimeError("Convergence failure (circuit at the last iteration: {})".format(last.get("circuit")))


def xp_solve(re_=1e-28, it=120):
    """The 128-bit (mixed-precision) Newton of carriers.extended_precision; with a custom term it converges
    LINEARLY (custom_terms module doc, section 5) -- returns the per-iteration relative errors."""
    from dynameta.carriers import extended_precision as XP
    with _quiet():
        info = XP.solve(type="dc", absolute_error=1e-3, relative_error=re_, maximum_iterations=it, info=True)
    if not info["converged"]:
        raise RuntimeError("extended-precision Newton did not converge: rel {} abs {}".format(
            [float(i["devices"][0]["relative_error"]) for i in info["iterations"]][-8:],
            [float(i["devices"][0]["absolute_error"]) for i in info["iterations"]][-8:]))
    return [float(i["devices"][0]["relative_error"]) for i in info["iterations"]]


def params(**kw):
    from dynameta.carriers.tunneling import SIO2_ON_SI, OxideTunnelingParams
    base = dict(model="tsu_esaki", coupling="self_consistent", barriers=SIO2_ON_SI, gate_fermi_energy_eV=8.0,
                m_supply_over_m0=0.5)
    base.update(kw)
    return OxideTunnelingParams(**base)


def matrix():
    ds = _ds()
    with _quiet():
        m = ds.get_matrix_and_rhs(format="csc")["static"]
    return (np.asarray(m["av"], dtype=float), np.asarray(m["ai"]), np.asarray(m["ap"]),
            np.asarray(m["rhs"], dtype=float))


def install(d, **kw):
    from dynameta.carriers import tunneling as TN
    return TN.setup_oxide_tunneling(d.name, "semi", "gate", params(**kw.pop("prm", {})), **kw)


def report(d, **kw):
    from dynameta.carriers import tunneling as TN
    return TN.oxide_tunneling_current(d.name, d.layers[-1][0], "semi", params(**kw))


def post(d, model, **kw):
    """The pre-v0.15 post-processor ('old code') on the device's current solution."""
    import warnings
    from dynameta.carriers import tunneling as TN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # FN below the barrier warns (old-code read-out)
        return TN.oxide_tunneling_current(d.name, d.layers[0][0], "semi", TN.OxideTunnelingParams(model=model, **kw))


# =====================================================================================================
# scenarios
# =====================================================================================================

def sc_balance(opts):
    """Detailed balance at zero bias with a built-in oxide field (gate work-function offset)."""
    out = {}
    # the equilibrium does not depend on tau; a short tau makes the inversion-layer Newton well conditioned
    for tag, dop, off, fd, tau in (("n_acc", 1e24, 1.5, False, 1e-6), ("p_inv", -1e23, 1.2, False, 1e-12),
                                   ("p_acc", -1e23, -1.0, False, 1e-6), ("n_inv", 1e24, -0.6, False, 1e-12),
                                   ("n_acc_wf", 1e24, 1.8, False, 1e-6), ("n_acc_fd", 1e24, 1.5, True, 1e-6)):
        d = Dev("b_" + tag, doping=dop, gate_offset=off, fd=fd, tau=tau)
        term = install(d)
        solve()
        st = term.state(d.name)
        te = st["te"]
        r = {"J": float(te.J[0]), "J_fwd": float(te.J_fwd[0]), "J_bwd": float(te.J_bwd[0]),
             "x_s": float(st["x_s"][0]), "x_g": float(st["x_g"][0]), "V_ox": float(-st["d"][0, -1]),
             "n_s": float(st["n_s"][0]), "p_s": float(d.nv("semi", "Holes")[term.iface[0]])}
        for m, kw in (("fn", {}), ("schuegraf_hu", {}), ("tsu_esaki", {"E_F_eV": 0.1, "m_supply_over_m0": 0.5})):
            r["post_" + m] = float(post(d, m, **kw)["I"])
        # refute: the gate frame 1 mV off (phi_c) -> a spurious battery
        phi = term.phi_c
        term.phi_c = phi + 1e-3
        r["J_refute"] = float(term.state(d.name)["te"].J[0])
        term.phi_c = phi
        out[tag] = r
        d.close()
    return out


def sc_balance_fd(opts):
    """Detailed balance with the Fermi-Dirac g-factor: the discrete FD-SG equilibrium is off the closed-form
    activity by O(h^2) (D1d L1 addendum), so the zero-bias current is a CONVERGENCE quantity: the uniform
    surface zone at h = 0.1 nm x s."""
    out = []
    for k, sc in enumerate(opts.get("scales", [1.0, 0.5, 0.25])):
        d = Dev("bfd{}".format(k), doping=1e24, gate_offset=1.5, fd=True, h_if=0.1e-9 * sc)
        term = install(d)
        solve()
        te = term.state(d.name)["te"]
        out.append({"scale": sc, "J": float(te.J[0]), "J_fwd": float(te.J_fwd[0]),
                    "n_nodes": int(d.nv("semi", "x").size)})
        d.close()
    return out


def sc_inversion(opts):
    """Supply-limited inversion: p-Si, 2 nm SiO2, positive gate; the tunnelling drains the inversion
    layer and the gate current is set by the SRH generation in the deep-depleted body. The OLD model (no
    tunnelling feedback, the ideal capacitor in equilibrium inversion) is post-processed at the same bias;
    its DC state does not depend on tau, and a short tau makes that Newton well conditioned."""
    out = {"pts": []}
    L = opts.get("L", 1.5e-6)
    Vs = opts.get("V", [1.5, 3.0])
    ref = Dev("ref", doping=-1e23, tau=1e-12, L=L)
    old = {}
    for V in Vs:
        ref.goto(V)
        old[V] = ({m: float(post(ref, m, **kw)["I"]) for m, kw in
                   (("fn", {}), ("schuegraf_hu", {}), ("tsu_esaki", {"E_F_eV": 0.1, "m_supply_over_m0": 0.5}))},
                  float(post(ref, "fn")["V_ox_V"][0]), float(ref.nv("semi", "Electrons")[-1]))
    ref.close()
    for k, tau in enumerate(opts.get("taus", [1e-6, 1e-7])):
        d = Dev("inv{}".format(k), doping=-1e23, tau=tau, L=L)
        term = install(d)
        solve()
        for V in Vs:
            d.goto(V)
            solve(re_=1e-14)                                   # polish: the conservation floor is the residual
            rep = report(d)
            U = d.nv("semi", "USRH")
            vol = d.nv("semi", "NodeVolume")
            out["pts"].append({
                "tau": tau, "V": V, "I_gate": rep["I_gate"], "I_body": d.body_current(),
                "I_n_body": float(_ds().get_contact_current(device=d.name, contact="body", equation=EQN)),
                "I_gen": float(-Q * np.sum(U[1:] * vol[1:])), "I_gen_abs": float(Q * np.sum(np.abs(U) * vol)),
                "floor": d.residual_floor(), "floor_n": d.residual_floor((EQN,)),
                "psi_s": float(d.nv("semi", "Potential")[term.iface[0]]),
                "psi_bulk": float(d.nv("semi", "Potential")[0]), "n_s": float(rep["n_s_m3"][0]),
                "V_ox": float(rep["V_ox_V"][0]), "J_fwd": float(rep["J_fwd_A_m2"][0]),
                "post": old[V][0], "V_ox_ref": old[V][1], "n_s_ref": old[V][2]})
        d.close()
    return out


def term_jacobian(d, term, *, circ):
    """The term's OWN assembled Jacobian against central differences (h, 2h) of its OWN residual, from
    term.assemble at perturbed states -- no global-matrix cancellation (the global-matrix check is
    jac_records). Returns per column: analytic entries, fd, fd2 on the term's rows."""
    ds = _ds()
    from dynameta.carriers import custom_terms as CT

    def entries():
        rcv, rv = term.assemble(d.name, "MATRIXANDRHS", "DC")
        M = {}
        for r, c, v in zip(rcv[0::3], rcv[1::3], rcv[2::3]):
            M[(int(r), int(c))] = M.get((int(r), int(c)), 0.0) + float(v)
        F = {}
        for r, v in zip(rv[0::2], rv[1::2]):
            F[int(r)] = F.get(int(r), 0.0) + float(v)
        return M, F
    M, F = entries()
    rows = sorted(F)
    out = []
    i0 = int(term.iface[0])
    cols = [("semi", "Potential", i0, 1e-5), ("semi", "Electrons", i0, None)]
    for reg, nodes, W in term.faces:
        cols.append((reg, "Potential", int(nodes[W[0][0][0]]), 1e-5))
    for reg, var, node, h in cols:
        col = int(CT.node_columns(d.name, reg, var, [node])[0])
        x0 = float(d.nv(reg, var)[node])
        hh = h if h is not None else 1e-6 * x0
        rec = {"col": [reg, var, node], "J": [M.get((r, col), 0.0) for r in rows], "h": hh,
               "F": [F[r] for r in rows]}
        for tg, step in (("fd", hh), ("fd2", 2 * hh)):
            vals = []
            for sgn in (1.0, -1.0):
                ds.set_node_value(device=d.name, region=reg, name=var, index=node, value=x0 + sgn * step)
                vals.append(entries()[1])
            rec[tg] = [(vals[0][r] - vals[1][r]) / (2 * step) for r in rows]
        ds.set_node_value(device=d.name, region=reg, name=var, index=node, value=x0)
        out.append(rec)
    if circ:
        col = CT.circuit_row(term.gate_node)
        v0 = float(ds.get_circuit_node_value(node=term.gate_node, solution="dcop"))
        rec = {"col": ["circuit", term.gate_node, 0], "J": [M.get((r, col), 0.0) for r in rows], "h": 1e-5,
               "F": [F[r] for r in rows]}
        for tg, step in (("fd", 1e-5), ("fd2", 2e-5)):
            vals = []
            for sgn in (1.0, -1.0):
                ds.set_circuit_node_value(node=term.gate_node, solution="dcop", value=v0 + sgn * step)
                vals.append(entries()[1])
            rec[tg] = [(vals[0][r] - vals[1][r]) / (2 * step) for r in rows]
        ds.set_circuit_node_value(node=term.gate_node, solution="dcop", value=v0)
        out.append(rec)
    return out


def _flip_kcl(term):
    """The gate KCL entries with the WRONG sign: the deliberate refutation of the conservation and ssac
    gates."""
    orig = term.assemble

    def assemble(device, what, timemode):
        from dynameta.carriers import custom_terms as CT
        rcv, rv = orig(device, what, timemode)
        if term.gate_node is None or (not rcv and not rv):
            return rcv, rv
        g = CT.circuit_row(term.gate_node)
        rcv = list(rcv)
        for i in range(0, len(rcv), 3):
            if rcv[i] == g:
                rcv[i + 2] = -rcv[i + 2]
        rv = list(rv)
        for i in range(0, len(rv), 2):
            if rv[i] == g:
                rv[i + 1] = -rv[i + 1]
        return rcv, rv
    term.assemble = assemble


def sc_conserve(opts):
    """Terminal-current balance I_body + I_gate = 0, n-Si accumulation, ONE variant per process (a DEVSIM
    circuit node cannot be added after the first solve): "param" (bias-parameter gate, module-reported
    I_gate), "circuit" (the source current), "flipped" (circuit, KCL sign flipped), "old" (no term: the
    post-processed current is NOT drawn from the body)."""
    from dynameta.carriers.contact_current import extract_contact_currents
    var = opts.get("variant", "param")
    circ = var.split("_")[0] in ("circuit", "flipped")
    xp = var.endswith("_xp")
    d = Dev("c_" + var, doping=1e24, circuit=circ, extended=xp)
    if var == "old":
        d.goto(1.0)
        return {"I_body": d.body_current(), "J_post": float(post(d, "schuegraf_hu")["I"]),
                "J_post_te": float(post(d, "tsu_esaki", E_F_eV=0.1, m_supply_over_m0=0.5)["I"]),
                "cc": extract_contact_currents(d.name)}
    term = install(d)
    if var.startswith("flipped"):
        _flip_kcl(term)
    solve()
    pts = []
    for V in opts.get("V", [0.6, 1.0, 1.4]):
        d.goto(V)
        its = xp_solve() if xp else (solve(re_=1e-14) or [])
        rep = report(d)
        pts.append({"V": V, "I_gate": rep["I_gate"], "I_body": d.body_current(),
                    "I_src": d.source_current() if circ else None,
                    "floor": None if xp else d.residual_floor(), "xp_iterations": len(its),
                    "xp_rel": its[-3:], "cc": extract_contact_currents(d.name)})
    return {"pts": pts}


def sc_ssac(opts):
    """Re Y(f -> 0) of a circuit gate vs the DC dI/dV (Richardson central differences), ONE variant per
    process: "on", "off" (no term), "flipped" (KCL sign)."""
    from dynameta.carriers import ac_analysis as AC
    V0 = float(opts.get("V0", 1.0))
    var = opts.get("variant", "on")
    d = Dev("s_" + var, doping=1e24, circuit=True)
    if var != "off":
        term = install(d)
        if var == "flipped":
            _flip_kcl(term)
        solve()
    d.goto(V0)
    I = {}
    for h in (-2e-4, -1e-4, 1e-4, 2e-4):
        d.set_gate(V0 + h)
        solve(re_=1e-14)
        I[repr(h)] = d.source_current()
    d.set_gate(V0)
    solve(re_=1e-14)
    ff = opts.get("f", [1.0, 1e3, 1e6])
    f, C, G = AC.ssac_admittance(ff, source_name=d.source)
    return {"I": I, "G": _lst(G), "C": _lst(C), "f": list(ff), "I0": d.source_current()}


def sc_fdt(opts):
    """Equilibrium noise of a circuit gate with a built-in oxide field: S_I = 4 kT Re Y. ONE variant per
    process: "n_acc" (degenerate accumulation at equilibrium: the Pauli term matters), "n_dep"."""
    from dynameta.carriers import ac_analysis as AC
    from dynameta.carriers import noise_analysis as NA
    var = opts.get("variant", "n_acc")
    off = {"n_acc": float(opts.get("offset", 1.5)), "n_dep": 0.0}[var]
    d = Dev("f_" + var, doping=1e24, circuit=True, gate_offset=off, v_ref=0.5, i_park=1.0)
    term = install(d)
    solve()
    ff = opts.get("f", [1.0, 1e3, 1e6, 1e9])
    sn = NA.noise_spectrum(d.name, ff, output_node=d.source + ".I")
    f, C, G = AC.ssac_admittance(ff, source_name=d.source)
    st = term.state(d.name)
    te = st["te"]
    return {"S": _lst(sn.S), "by_mechanism": {k: _lst(v) for k, v in sn.by_mechanism.items()},
            "G": _lst(G), "C": _lst(C), "f": list(ff), "J": float(te.J[0]), "J_fwd": float(te.J_fwd[0]),
            "J_bwd": float(te.J_bwd[0]), "S_kernel": float(te.S[0]), "x_s": float(st["x_s"][0]),
            "area": float(term.area[0])}


def _x_vector(dev, regions):
    ds = _ds()
    from dynameta.carriers import custom_terms as CT
    ncol = matrix()[2].size - 1
    x = np.zeros(ncol)
    for r in regions:
        for v in ("Potential", "Electrons", "Holes"):
            if v not in ds.get_node_model_list(device=dev, region=r):
                continue
            c = np.asarray(ds.get_equation_numbers(device=dev, region=r, variable=v))
            ok = c >= 0
            x[c[ok]] = np.asarray(ds.get_node_model_values(device=dev, region=r, name=v))[ok]
    for cn in ds.get_circuit_node_list():
        x[CT.circuit_row(cn)] = float(ds.get_circuit_node_value(node=cn, solution="dcop"))
    return x


def jac_records(d, term, *, circ):
    """The term's own Jacobian share (on - off at the same state) against central differences (steps h,
    2h) of its own residual share, for every column it touches: psi_s, n_s, each face potential and (circuit
    gate) the gate node. rhs_scale = max_r sum_k |J[r, k] x_k| (the round-off scale of the rows)."""
    import scipy.sparse as sp
    from dynameta.carriers import custom_terms as CT
    ds = _ds()

    def both():
        term.active = False
        off = matrix()
        term.active = True
        on = matrix()
        return on, off
    on, off = both()
    Jon = sp.csc_matrix((on[0], on[1], on[2]))
    Jc = Jon - sp.csc_matrix((off[0], off[1], off[2]))
    rows = list(CT.node_rows(d.name, "semi", EQN, term.iface))
    if circ:
        rows.append(CT.circuit_row(term.gate_node))
    x = _x_vector(d.name, d.regions)
    R = float(np.max(np.abs(Jon[rows, :]) @ np.abs(x)))
    recs = []
    i0 = int(term.iface[0])
    cols = [("semi", "Potential", i0, 1e-4), ("semi", "Electrons", i0, None)]
    for reg, nodes, W in term.faces:
        cols.append((reg, "Potential", int(nodes[W[0][0][0]]), 1e-4))
    for reg, var, node, h in cols:
        col = int(CT.node_columns(d.name, reg, var, [node])[0])
        x0 = float(d.nv(reg, var)[node])
        hh = h if h is not None else 1e-5 * x0
        rec = {"col": [reg, var, node], "h": hh, "J": _lst(Jc[rows, col].toarray()), "rhs_scale": R}
        for tg, step in (("fd", hh), ("fd2", 2 * hh)):
            share = []
            for sgn in (1.0, -1.0):
                ds.set_node_value(device=d.name, region=reg, name=var, index=node, value=x0 + sgn * step)
                a, b = both()
                share.append(a[3][rows] - b[3][rows])
            rec[tg] = _lst((share[0] - share[1]) / (2 * step))
        ds.set_node_value(device=d.name, region=reg, name=var, index=node, value=x0)
        recs.append(rec)
    if circ:
        col = CT.circuit_row(term.gate_node)
        v0 = float(ds.get_circuit_node_value(node=term.gate_node, solution="dcop"))
        rec = {"col": ["circuit", term.gate_node, 0], "h": 1e-4, "J": _lst(Jc[rows, col].toarray()),
               "rhs_scale": R}
        for tg, step in (("fd", 1e-4), ("fd2", 2e-4)):
            share = []
            for sgn in (1.0, -1.0):
                ds.set_circuit_node_value(node=term.gate_node, solution="dcop", value=v0 + sgn * step)
                a, b = both()
                share.append(a[3][rows] - b[3][rows])
            rec[tg] = _lst((share[0] - share[1]) / (2 * step))
        ds.set_circuit_node_value(node=term.gate_node, solution="dcop", value=v0)
        recs.append(rec)
    return recs


def sc_jac(opts):
    """Jacobian records (jac_records), ONE variant per process: "acc" (circuit gate, accumulation), "inv"
    (bias-parameter gate, deep depletion), "stack" (circuit gate, SiO2 / high-k)."""
    from dynameta.carriers.tunneling import DielectricBarrier, SIO2_ON_SI
    var = opts.get("variant", "acc")
    dop, V, circ, layers = {"acc": (1e24, 1.0, True, None), "inv": (-1e23, 2.0, False, None),
                            "near_eq": (1e24, 0.52, True, None),
                            "stack": (1e24, 1.2, True, [("ox_a", 1e-9, 3.9), ("hk", 2e-9, 20.0)])}[var]
    extra = {"gate_offset": 1.5, "v_ref": 0.5, "i_park": 1.0} if var == "near_eq" else {}
    d = Dev("j_" + var, doping=dop, circuit=circ, layers=layers, dummy_circuit=not circ, **extra)
    kw = {}
    if layers is not None:
        kw["prm"] = {"barriers": {"ox_a": SIO2_ON_SI, "hk": DielectricBarrier(1.5, "Si", 0.18)}}
    term = install(d, **kw)
    d.goto(V)
    return {"global": jac_records(d, term, circ=circ), "term": term_jacobian(d, term, circ=circ)}


def sc_stack(opts):
    """Multi-dielectric columns (bias-parameter gates): SiO2 as ONE region vs the same SiO2 split into TWO
    regions; a SiO2/high-k stack and its reversed order."""
    from dynameta.carriers.tunneling import DielectricBarrier, SIO2_ON_SI
    hk = DielectricBarrier(1.5, "Si", 0.18)
    out = {}
    V = float(opts.get("V", 1.2))
    for tag, layers, bars in (
            ("single", [("oxide", 2e-9, 3.9)], SIO2_ON_SI),
            ("split", [("ox_a", 0.8e-9, 3.9), ("ox_b", 1.2e-9, 3.9)], SIO2_ON_SI),
            ("sio2_hk", [("ox_a", 1e-9, 3.9), ("hk", 3e-9, 20.0)], {"ox_a": SIO2_ON_SI, "hk": hk}),
            ("hk_sio2", [("hk", 3e-9, 20.0), ("ox_a", 1e-9, 3.9)], {"ox_a": SIO2_ON_SI, "hk": hk})):
        d = Dev("st_" + tag, doping=1e24, layers=layers)
        term = install(d, prm={"barriers": bars})
        solve()
        d.goto(V)
        solve(re_=1e-14)
        st = term.state(d.name)
        faces = [float(d.nv("semi", "Potential")[term.iface[0]])]
        for reg, nodes, _ in term.faces:
            faces.append(float(d.nv(reg, "Potential")[nodes[0]]))
        out[tag] = {"J": float(st["te"].J[0]), "I_gate": float(np.sum(term.area * st["te"].J)),
                    "I_body": d.body_current(), "faces": faces, "t": [l.thickness_m for l in term.layers],
                    "eps": [e for _, _, e in layers], "offsets": [l.offset_eV for l in term.layers],
                    "floor": d.residual_floor()}
        d.close()
    return out


def sc_dims(opts):
    """2-D / 3-D == 1-D on tensor meshes: the 1-D device is solved first (then deleted -- DEVSIM solves every
    live device together), its normal-axis node set is replicated laterally, and the same bias is solved.
    ONE dimension per process (variant "2d" / "3d"); bias-parameter gates (no circuits)."""
    dim = {"2d": 2, "3d": 3}[opts.get("variant", "2d")]
    V, dop = float(opts.get("V", 1.0)), float(opts.get("doping", 1e24))
    L = float(opts.get("L", 0.3e-6))
    W = float(opts.get("width", 4e-9))
    d1 = Dev("d1", doping=dop, L=L, h_body=float(opts.get("h_body", 10e-9)), h_if=0.1e-9)
    t1 = install(d1)
    d1.goto(V)
    st1 = t1.state(d1.name)
    xs = np.concatenate([d1.nv("semi", "x"), d1.nv("oxide", "x")[1:]])
    ref = {"J": float(st1["te"].J[0]), "V_ox": float(-st1["d"][0, -1]), "n_s": float(st1["n_s"][0]),
           "I_body": d1.body_current(), "n_nodes": int(xs.size)}
    d1.close()
    import time
    t0 = time.time()
    dn = Dev("dn", doping=dop, L=L, dim=dim, x_nodes=xs, width=W, n_lat=int(opts.get("n_lat", 3)),
             dummy_circuit=True)
    tn = install(dn)
    dn.goto(V)
    stn = tn.state(dn.name)
    rep = report(dn)
    out = {"ref": ref, "J": _lst(stn["te"].J), "V_ox": _lst(-stn["d"][:, -1]), "area": _lst(tn.area),
           "W": W, "dim": dim, "I_gate": rep["I_gate"], "I_body": dn.body_current(),
           "n_iface": int(tn.iface.size), "weights": [[int(j) for j in w[0]] for w in tn.faces[0][2]],
           "wvals": [[float(v) for v in w[1]] for w in tn.faces[0][2]], "solve_s": time.time() - t0}
    t1_ = time.time()
    out["term"] = term_jacobian(dn, tn, circ=False)
    out["jac_s"] = time.time() - t1_
    return out


def sc_gmsh(opts):
    """The plan's mesh caveat, MEASURED: a laterally uniform Si / 2 nm SiO2 stack on the Stacked3DSpec
    builder, mesher "gmsh" (DEVSIM's unsigned box geometry), "gmsh" + devsim_geometry.apply_exact_geometry
    (negative='warn', hooked right after create_device as validation/devsim_geometry.py does), and the
    "tensor" (Kuhn) default. The term is installed on the solved ideal-capacitor state and its kernel call
    read (no re-solve): the lateral spread of V_ox and of J over the interface nodes."""
    import tempfile
    import warnings
    from dynameta.carriers import devsim_geometry as G
    from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec
    from dynameta.constants import M_E
    from dynameta.sweep import BiasPoint
    ds = _ds()
    var = opts.get("variant", "gmsh")
    spec = Stacked3DSpec(semi_material="Si", oxide_material="SiO2", lateral_m=4e-9, semi_thk_m=20e-9,
                         oxide_thk_m=2e-9, n_bg_m3=1e24, eps_semi=11.7, eps_oxide=3.9, dos_mass_kg=1.08 * M_E,
                         dos_mass_p_kg=0.81 * M_E, physics="bipolar_dd", n_i_m3=1e16, mobility_m2Vs=0.1,
                         mobility_p_m2Vs=0.04, tau_srh_s=1e-7, grid_n=(3, 3, 9), mesh_min_nm=0.5,
                         mesh_max_nm=2.0, mesher="tensor" if var == "tensor" else "gmsh")
    tmp = tempfile.mkdtemp(prefix="oxg_")
    sol = Devsim3DEquilibrium(spec, device_name="g3", mesh_name="g3m",
                              msh_path=None if var == "tensor" else os.path.join(tmp, "g3.msh"))
    reps = {}
    if var == "gmsh_exact":
        orig = sol._create_mesh_device

        def patched():
            orig()
            for r in ds.get_region_list(device=sol.device):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rep = G.apply_exact_geometry(sol.device, r, negative="warn")
                reps[r] = {"neg_couples": int(rep.n_negative_couples), "n_el": int(rep.n_elements)}
        sol._create_mesh_device = patched
    with _quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # the gmsh mesher's own UnreliableResultWarning etc.
        sol.solve(BiasPoint({"gate": float(opts.get("V", 1.2))}, "g"))
    out = {"variant": var, "geometry": reps}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        term = install(type("D", (), {"name": "g3"})(), dielectric_regions=["oxide"])
    out["setup_warnings"] = [(type(x.message).__name__, str(x.message)[:300]) for x in w]
    st = term.state("g3")
    J = st["te"].J
    Vox = -st["d"][:, -1]
    out.update({"J": _lst(J), "V_ox": _lst(Vox), "area": _lst(term.area), "lateral_m2": 16e-18,
                "J_spread": float(np.ptp(J) / np.mean(J)), "V_ox_spread": float(np.ptp(Vox) / np.mean(Vox)),
                "I_over_area": float(np.sum(term.area * J) / 16e-18)})
    return out


SCENARIOS = {"balance": sc_balance, "balance_fd": sc_balance_fd, "inversion": sc_inversion,
             "conserve": sc_conserve, "ssac": sc_ssac, "fdt": sc_fdt,
             "jac": sc_jac, "stack": sc_stack, "dims": sc_dims, "gmsh": sc_gmsh}


def run_child(scenario, **opts):
    """Run `scenario` in a fresh interpreter on this sys.path; returns its JSON result."""
    import subprocess
    import tempfile
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if isinstance(p, str) and p))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), scenario, out, json.dumps(opts)],
                           env=env, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError("child {} failed:\n{}".format(scenario, r.stderr[-6000:]))
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.remove(out)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error")                       # warnings are errors, as in the suite
    _opts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    _res = SCENARIOS[sys.argv[1]](_opts)
    with open(sys.argv[2], "w", encoding="utf-8") as _f:
        json.dump(_res, _f)
