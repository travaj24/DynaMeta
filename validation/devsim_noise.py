"""
Validate small-signal NOISE (dynameta.carriers.noise_analysis, impedance-field method) on DEVSIM
drift-diffusion devices against oracles that are INDEPENDENT of the code under test: closed-form
physics (Johnson-Nyquist, the fluctuation-dissipation theorem with the ssac admittance, the ideal-
diode shot-noise law, the ambipolar GR Lorentzian, Hooge's 1/f law) evaluated with scipy.constants
and the solved DC/ssac quantities -- never a number noise_analysis computed.

GATE A  Johnson-Nyquist: uniform n-type bar at equilibrium, S_I = 4 kT G with G the solved DC
        conductance (+-1 mV finite difference). Bipolar 1-D (non-uniform mesh), unipolar DD 1-D
        (Fermi-Dirac g-factor ON), bipolar 2-D, unipolar 3-D (gmsh, if installed).
GATE B  Fluctuation-dissipation: long p-n diode at zero bias, 1 Hz - 10 GHz (Re Z spans 4.6
        decades): S_I = 4 kT Re Y(f) [voltage-driven contact] and, through a SEPARATE adjoint
        system, the open-circuit S_V = 4 kT Re(1/Y(f)) [current-driven contact]; the same diode
        split into two regions joined by a carrier-continuous interface.
GATE C  Ideal p-n diode: S_I(V) = 2 q I coth(qV/2kT) (= 2q(I + 2 I_s); full shot noise 2qI at
        forward bias, 4kT G_0 at V -> 0) for V = 0.05 ... 0.55 V on a long Si diode at low
        frequency; S_I / (4 kT G_d) -> 1/2 at forward bias; van der Ziel's long-diode high-frequency
        law S_I(f) = 4 kT Re Y(f) - 2 q I from 1 Hz to 1 MHz (S_I rises 13x).
GATE D  GR noise, uniform SRH slab under bias: the GR part equals the ambipolar Lorentzian (with
        the ohmic contacts' cosh shorting) for f/f_c = 1e-3 ... 100, f_c = 1/(2 pi tau_eff); on a
        400 L_D slab the plain Lorentzian plateau 2(G+R) tau^2 (q(mu_n+mu_p)E)^2/L and the corner.
GATE E  Diffusion vs shot discriminant: a BIASED resistor keeps S = 4kT G (no shot noise; 2qI would
        be 1.93x larger at 0.1 V) while a forward diode gives S = 2qI = (1/2) 4kT G_d.
GATE F  Hooge 1/f (opt-in): biased resistor, S_H = alpha_H I^2 / (N f), N = n L. F3: the same law on
        the 3-D Kuhn-tensor bar (N = n V; it read 1.125 before the P1 element couples, 2026-11-06).
GATE G  2-D LAYERED device (LayeredDevsimBuilder reference metasurface, degenerate FD ITO, 9 regions,
        dielectric interfaces, edge-metal grounds) via set_ssac_gate: S_I = 4kT Re Y, 1 GHz - 1 PHz, at
        Vg = 0 (flat band) and Vg = +2 V (degenerate accumulation, where the FD g-factor SG is only
        approximately Nyquist-consistent).
GATE H  THERMIONIC-interface shot noise (v0.15, plan 4.3.1): the te_n / te_p two-point source S = 2q(J_01 +
        J_10) A per node pair. H1 the FDT on the heterojunction diode at A* x 1e-6 .. 1e2 (the TE share falls
        monotonically, the total tends to the continuous-QFL model); H2 the coth law S = 2qJ coth(qV/2kT) on
        the interface-dominated isotype n-GaAs / N-Al0.3Ga0.7As at +-1 .. 8 kT/q (DC J and V only); H3 the
        FDT on the 2-D stack with a non-uniform lateral mesh, and EQUAL area shares must break it; H4 the
        FDT on 3-D Kuhn stacks, jittered with exact geometry and split with shared interface nodes.

Run: python -m validation.devsim_noise
"""

from __future__ import annotations

import contextlib
import math
import os
import sys

import numpy as np
from scipy.constants import e as QE, k as KB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.carriers.devsim_env import import_devsim  # noqa: E402
ds = import_devsim()
from dynameta.carriers import eq_registry as _R                           # noqa: E402
from dynameta.carriers import noise_analysis as NA                        # noqa: E402
from dynameta.carriers import physics_bipolar_dd as BP                    # noqa: E402
from dynameta.carriers import physics_drift_diffusion as DD               # noqa: E402
from dynameta.core.numerics import trapz                                   # noqa: E402

T = 300.0
KT = KB * T
VT = KT / QE
MU_N, MU_P = 0.135, 0.048                       # Si, m^2/Vs
N_DOS = 2.8e25


@contextlib.contextmanager
def quiet():
    """Silence DEVSIM's C-level stdout (fd redirect)."""
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


def fresh() -> None:
    """Drop every device / mesh / circuit node (circuit nodes are global in DEVSIM)."""
    for d in list(ds.get_device_list()):
        ds.delete_device(device=d)
        _R.clear(d)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)
    try:
        ds.delete_circuit()
    except Exception:
        pass


def _solve(ae, re=1e-5, mi=80):
    with quiet():
        ds.solve(type="dc", solver_type="direct", absolute_error=ae, relative_error=re,
                 maximum_iterations=mi)


#: the CIRCUIT's relative criterion of a DC solve at ZERO terminal current (solve_zero_current)
ZERO_CURRENT_RE_CIRCUIT = 1e-3


def solve_zero_current(ae, re=1e-5, mi=100, re_circuit=ZERO_CURRENT_RE_CIRCUIT):
    """DC solve of a circuit-driven device at an EQUILIBRIUM bias (zero terminal current), with the device
    equations held to `re` and only the circuit's relative test relaxed to `re_circuit`.

    WHY. DEVSIM's relative test is |dx| / (|x| + 1e-10). The circuit unknown here is the source current,
    ~1e-14 A/m^2 of round-off at zero bias, so the test becomes |dI| < re * 1e-10 -- and |dI| at
    convergence is the round-off of the contact's flux sum, a few ulp of the one-way interface stream
    (the isotype junction: J_s = (kT/q) G_0 = 1.6 A/m^2, MEASURED |dI| = 1-7e-15 A/m^2, i.e. a circuit
    relative error that random-walks over 1.3e-5 .. 7e-5 for as long as Newton runs while the device
    equations sit at 1e-15). At re = 1e-5 the exit is decided by the last bits: Windows dipped to 9.3e-6
    at iteration 8; WSL / the CI runner cycled (period 4, 4.3e-5 .. 6.8e-5) to 'Convergence failure!'
    (PR #36, 2026-11-08). DEVSIM has one relative_error for device and circuit, so the solve runs at
    `re_circuit` (14x the worst measured floor) and the DEVICE's relative error at the accepted
    iteration is then required to be below `re` -- the device criterion is the one the caller asked for;
    only the ill-posed circuit test changes. (Away from zero current the circuit test is well posed:
    use _solve.)"""
    with quiet():
        info = ds.solve(type="dc", solver_type="direct", absolute_error=ae, relative_error=re_circuit,
                        maximum_iterations=mi, info=True)
    its = info.get("iterations") or ()
    if not info.get("converged", False):
        raise ds.error("Convergence failure! (solve_zero_current: {} iterations)".format(len(its)))
    dev_rel = max(d["relative_error"] for d in its[-1]["devices"])
    if not dev_rel < re:
        raise ds.error("Convergence failure! (solve_zero_current: the circuit met {:g} but the device "
                       "relative error {:.3g} is not below {:g})".format(re_circuit, dev_rel, re))
    return info


# =====================================================================================================
# device builders
# =====================================================================================================

def bipolar_1d(dev, lines, net_doping, *, n_i, tau, drive="voltage", presolve=True, ae=1e8,
               options=None, **region_kw):
    """1-D bipolar device on [(pos, ps, tag)]; 'lo' grounded ohmic, 'hi' circuit ohmic (source V1
    or current source I1 on node 'vhi'); solved to equilibrium in the frame its contacts pin.
    `region_kw` go to setup_bipolar_region (traps=, srh_trap_energy_eV=, btbt=: v0.14 integration)."""
    fresh()
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    for pos, ps, tag in lines:
        kw = {"tag": tag} if tag else {}
        ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, **kw)
    ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="bulk", tag1="lo", tag2="hi")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="bulk", name="NetDoping", equation=net_doping)
    with quiet():
        BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=N_DOS, n_i_m3=n_i,
                                mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=tau,
                                tau_p_s=tau, fd_enhancement=False,
                                **({} if options is None else {"options": options}), **region_kw)
        BP.setup_equilibrium_seed_models(dev, "bulk")
        BP.setup_contact_ohmic_bipolar(dev, "lo")
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi",
                                               source_name="V1" if drive == "voltage" else "I1",
                                               drive=drive)
        ds.node_model(device=dev, region="bulk", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(BP.fd_shift_model(dev, "bulk")))
    _seed(dev, ("Potential", "Electrons", "Holes"))
    if presolve and drive == "voltage":                 # frozen-carrier Poisson first (1-D recipe)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.delete_by_name(dev, ceq)
        _solve(1e10, 1e-10, 100)
        for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
            _R.reapply_by_name(dev, ceq)
        _seed(dev, ("Electrons", "Holes"))
    _solve(ae)
    return dev


def _seed(dev, names, region="bulk"):
    src = {"Potential": "_seed_psi", "Electrons": "IntrinsicElectrons", "Holes": "IntrinsicHoles"}
    for nm in names:
        ds.set_node_values(device=dev, region=region, name=nm,
                           values=ds.get_node_model_values(device=dev, region=region, name=src[nm]))
    if "Electrons" in names:                    # v0.15 kinetic trap levels follow their carriers (no-op else)
        from dynameta.carriers.physics_traps import seed_kinetic_traps
        seed_kinetic_traps(dev, [region])


def unipolar_1d_resistor(dev="ures", *, L=1e-6, n_bg=5e26):
    """1-D unipolar DD bar (ITO-like, FD g-factor ON, degenerate) with circuit contact 'hi'."""
    from dynameta.carriers.physics_equilibrium import M_E
    fresh()
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=L / 50, tag="lo")
    ds.add_1d_mesh_line(mesh=mesh, pos=0.3 * L, ps=L / 211)
    ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=L / 30, tag="hi")
    ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=mesh, material="ITO", region="bulk", tag1="lo", tag2="hi")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    with quiet():
        DD.setup_semiconductor_region_dd(dev, "bulk", n_bg_m3=n_bg, eps_static=9.5,
                                         dos_mass_kg=0.35 * M_E, mobility_m2Vs=0.004)
        DD.setup_contact_ohmic_dd(dev, "lo")
        DD.setup_contact_ohmic_dd_circuit(dev, "hi", node_name="vhi", source_name="V1")
    _solve(max(1e10, n_bg * 1e-12), 1e-9)
    return dev


def bipolar_2d_resistor(dev="res2d", *, LX=1e-6, LY=0.4e-6, nd=1e23):
    """2-D uniform n-type bipolar bar, contacts on the x = 0 / x = LX faces, non-uniform x/y mesh."""
    fresh()
    mesh = dev + "_m"
    ds.create_2d_mesh(mesh=mesh)
    pad = LX / 40                                  # DEVSIM 2-D contacts are boxes of mesh OUTSIDE
    ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=-pad, ps=pad)          # the region they touch
    ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=0.0, ps=LX / 40)
    ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=0.4 * LX, ps=LX / 90)
    ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=LX, ps=LX / 25)
    ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=LX + pad, ps=pad)
    ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=0.0, ps=LY / 8)
    ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=LY, ps=LY / 13)
    ds.add_2d_region(mesh=mesh, material="Si", region="bulk")                  # (DEVSIM's 2-D
    ds.add_2d_region(mesh=mesh, material="Si", region="pad_lo", xl=-pad, xh=0.0)  # diode recipe:
    ds.add_2d_region(mesh=mesh, material="Si", region="pad_hi", xl=LX, xh=LX + pad)  # inert pads)
    ds.add_2d_contact(mesh=mesh, name="lo", material="metal", region="bulk", xl=0.0, xh=0.0,
                      bloat=1e-10)
    ds.add_2d_contact(mesh=mesh, name="hi", material="metal", region="bulk", xl=LX, xh=LX,
                      bloat=1e-10)
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    ds.node_model(device=dev, region="bulk", name="NetDoping", equation="{:.6e}".format(nd))
    with quiet():
        BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=N_DOS, n_i_m3=1e16,
                                mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=1e-7,
                                tau_p_s=1e-7, fd_enhancement=False)
        BP.setup_equilibrium_seed_models(dev, "bulk")
        BP.setup_contact_ohmic_bipolar(dev, "lo")
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1")
        ds.node_model(device=dev, region="bulk", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(None))
    _seed(dev, ("Potential", "Electrons", "Holes"))
    _solve(1e9)
    return dev


def unipolar_3d_resistor(dev="res3d", *, LX=200.0, LY=80.0, LZ=80.0, n_bg=4e26):
    """3-D unipolar DD bar [nm] on the exact Kuhn-tensor mesh (devsim_3d.kuhn_bar_mesh, 20 nm cells, in
    memory). It was an unstructured gmsh mesh (14-20 nm) until 2026-11-04: MEASURED sum NodeVolume =
    1.225 x the bar volume there (DEVSIM's unsigned circumcentric pieces), so G_dc was not sigma A/L --
    gate A4's S = 4kT G still held, because the same couplings enter S and G (the ratio cancels the
    defect). On this mesh G_dc = sigma A/L as well."""
    from dynameta.carriers import devsim_3d as D3
    from dynameta.carriers.physics_equilibrium import M_E
    fresh()
    mesh = dev + "_m"
    D3.kuhn_bar_mesh(mesh, (LX, LY, LZ), 20.0, region="semi", contacts=("left", "right"))
    ds.add_gmsh_region(mesh=mesh, gmsh_name="semi", region="bulk", material="ITO")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="left", region="bulk", name="lo", material="metal")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="right", region="bulk", name="hi", material="metal")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    with quiet():
        DD.setup_semiconductor_region_dd(dev, "bulk", n_bg_m3=n_bg, eps_static=9.5,
                                         dos_mass_kg=0.35 * M_E, mobility_m2Vs=0.004)
        DD.setup_contact_ohmic_dd(dev, "lo")
        DD.setup_contact_ohmic_dd_circuit(dev, "hi", node_name="vhi", source_name="V1")
    _solve(max(1e10, n_bg * 1e-12), 1e-9)
    return dev


# =====================================================================================================
# measured quantities the oracles use (DC / ssac, never noise_analysis)
# =====================================================================================================

def source_current(src="V1"):
    """DC terminal current INTO the device through the driven contact [A, A/m (2-D), A/m^2 (1-D)]."""
    return -float(ds.get_circuit_node_value(node="{}.I".format(src), solution="dcop"))


def set_bias(v, ae, src="V1", step=0.025):
    """Ramp the voltage source's DC value to v (solving each step); returns v."""
    v0 = float(getattr(set_bias, "_v", {}).get(src, 0.0))
    n = max(1, int(math.ceil(abs(v - v0) / step - 1e-9)))
    for k in range(1, n + 1):
        ds.circuit_alter(name=src, value=v0 + (v - v0) * k / n)
        _solve(ae)
    set_bias._v = {**getattr(set_bias, "_v", {}), src: v}
    return v


def dc_conductance(v0, ae, dv=1e-3, src="V1", zero_current=False):
    """G = dI/dV at v0 by a central difference of solved DC currents; leaves the device at v0.
    zero_current=True: v0 is an equilibrium point (zero terminal current), so the return solve at v0 is
    solve_zero_current (the circuit's relative test is ill-posed there; the +-dv solves are not)."""
    out = []
    for v in (v0 - dv, v0 + dv):
        ds.circuit_alter(name=src, value=v)
        _solve(ae)
        out.append(source_current(src))
    ds.circuit_alter(name=src, value=v0)
    if zero_current:
        solve_zero_current(ae)
    else:
        _solve(ae)
    return (out[1] - out[0]) / (2.0 * dv)


def ssac_Y(freqs, src="V1"):
    from dynameta.carriers.ac_analysis import ssac_admittance
    with quiet():
        f, C, G = ssac_admittance(freqs, source_name=src)
    return G + 1j * 2.0 * np.pi * f * C


def noise(freqs, **kw):
    with quiet():
        return NA.noise_spectrum(**{"device": ds.get_device_list()[0], "frequencies": freqs,
                                    **kw})


# =====================================================================================================
# oracles
# =====================================================================================================

FREQS_A = [1.0, 1.0e6, 1.0e12]


def gate_a_johnson():
    """S_I / (4 kT G_dc) for the four resistors (1 = Johnson-Nyquist)."""
    out = {}
    set_bias._v = {}
    bipolar_1d("res1d", [(0.0, 1e-7, "lo"), (3.3e-7, 2.7e-8, None), (1e-6, 1e-7, "hi")], "1e22",
               n_i=1e16, tau=1e-7, ae=1e8)
    G = dc_conductance(0.0, 1e8)
    out["bipolar_1d"] = (noise(FREQS_A).S / (4 * KT * G), G, QE * MU_N * 1e22 / 1e-6)
    set_bias._v = {}
    unipolar_1d_resistor()
    G = dc_conductance(0.0, 5e14)
    out["unipolar_1d_fd"] = (noise(FREQS_A).S / (4 * KT * G), G, None)
    set_bias._v = {}
    bipolar_2d_resistor()
    G = dc_conductance(0.0, 1e9)
    out["bipolar_2d"] = (noise(FREQS_A).S / (4 * KT * G), G, QE * MU_N * 1e23 * 0.4e-6 / 1e-6)
    set_bias._v = {}
    if unipolar_3d_resistor() is not None:
        G = dc_conductance(0.0, 4e14)
        out["unipolar_3d"] = (noise(FREQS_A).S / (4 * KT * G), G, None)
    return out


# long Ge-like diode for the FDT gate: n_i = 1e19 keeps G_0 ~ 7 S/m^2 resolvable by ssac at 1 Hz
# (on a Si-like n_i = 1e16 p+n diode G_0 ~ 3e-5 S/m^2 is BELOW ssac's round-off floor ~0.3 S/m^2 --
# measured Re Y(1 Hz) = -0.27 S/m^2 there -- while the noise sum, all positive terms, stays exact).
FDT_NI, FDT_TAU, FDT_N = 1e19, 1e-6, 1e22
FREQS_B = list(np.logspace(0, 10, 11))


def _fdt_lines():
    Lp, Ln = math.sqrt(MU_P * VT * FDT_TAU), math.sqrt(MU_N * VT * FDT_TAU)
    xj = 4 * Ln
    return ([(0.0, Ln / 40, "lo"), (xj - 2e-6, 50e-9, None), (xj, 2e-9, None),
             (xj + 2e-6, 50e-9, None), (xj + 4 * Lp, Lp / 40, "hi")],
            "ifelse(x < {}, {}, {})".format(xj, FDT_N, -FDT_N))


def gate_b_fdt(options=None, open_circuit=True, freqs=None, **region_kw):
    """(S_I/4kT Re Y, S_V/4kT Re(1/Y), Re Z) on the zero-bias long diode, 1 Hz - 10 GHz. With
    `options` (BipolarOptions) -- or `region_kw` (traps=, btbt=, ...) -- the same identity tests the
    G + R split of every extra mechanism."""
    freqs = FREQS_B if freqs is None else freqs
    lines, dop = _fdt_lines()
    bipolar_1d("fdt", lines, dop, n_i=FDT_NI, tau=FDT_TAU, ae=1e8, options=options, **region_kw)
    sn = noise(freqs)
    sI = sn.S
    Y = ssac_Y(freqs)
    if not open_circuit:
        return sI / (4 * KT * Y.real), sn, Y
    bipolar_1d("fdt", lines, dop, n_i=FDT_NI, tau=FDT_TAU, ae=1e8, drive="current",
               options=options, **region_kw)
    sV = noise(freqs, output_node="vhi").S
    Z = 1.0 / Y
    return sI / (4 * KT * Y.real), sV / (4 * KT * Z.real), Z.real


def gate_b_two_region(freqs=(1.0, 1e5, 1e7, 1e9)):
    """The FDT diode split into TWO regions (r_n | r_p) joined at the junction by a carrier-
    CONTINUOUS interface (Potential, Electrons, Holes all type='continuous'): DEVSIM permutes one
    region's interface rows into the other's, and the noise must see the merged row on both copies.
    Returns (S_I/4kT Re Y - 1, S_two/S_one) with S_one the single-region gate_b device."""
    freqs = list(freqs)
    lines, _ = _fdt_lines()
    xj = lines[2][0]
    fresh()
    dev, mesh = "two", "two_m"
    ds.create_1d_mesh(mesh=mesh)
    for k, (pos, ps, tag) in enumerate(lines):
        ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, tag=tag or ("mid" if pos == xj else
                                                                  "t{}".format(k)))
    ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="r_n", tag1="lo", tag2="mid")
    ds.add_1d_region(mesh=mesh, material="Si", region="r_p", tag1="mid", tag2="hi")
    ds.add_1d_interface(mesh=mesh, name="jn", tag="mid")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    for r, nd in (("r_n", FDT_N), ("r_p", -FDT_N)):
        ds.node_model(device=dev, region=r, name="NetDoping", equation="{:.6e}".format(nd))
        with quiet():
            BP.setup_bipolar_region(dev, r, eps_static=11.7, n_dos_m3=N_DOS, n_i_m3=FDT_NI,
                                    mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=FDT_TAU,
                                    tau_p_s=FDT_TAU, fd_enhancement=False)
            BP.setup_equilibrium_seed_models(dev, r)
            ds.node_model(device=dev, region=r, name="_seed_psi",
                          equation=BP.equilibrium_seed_psi_expr(None))
    for var, eq in (("Potential", "PotentialEquation"), ("Electrons", "ElectronContinuityEquation"),
                    ("Holes", "HoleContinuityEquation")):
        im = "cont_{}".format(var)
        ds.interface_model(device=dev, interface="jn", name=im,
                           equation="{v}@r0 - {v}@r1".format(v=var))
        ds.interface_model(device=dev, interface="jn", name="{}:{}@r0".format(im, var), equation="1")
        ds.interface_model(device=dev, interface="jn", name="{}:{}@r1".format(im, var),
                           equation="-1")
        _R.record_interface_equation(dev, "jn", name=eq, interface_model=im, type="continuous")
    with quiet():
        BP.setup_contact_ohmic_bipolar(dev, "lo")
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1")
    for r in ("r_n", "r_p"):
        _seed(dev, ("Potential", "Electrons", "Holes"), region=r)
    _solve(1e8)
    sn = noise(freqs)
    Y = ssac_Y(freqs)
    r_two = sn.S / (4 * KT * Y.real) - 1
    _, sn1, _ = gate_b_fdt(open_circuit=False, freqs=freqs)
    return r_two, sn.S / sn1.S, sorted(sn.by_region)


def hetero_two_region(model="continuous", *, ni_p=3e18, dchi_eV=0.05, astar_scale=1.0):
    """The FDT diode as a HETEROjunction (v0.14 integration): r_n keeps the Ge-like statistics
    (n_i 1e19), r_p has n_i = `ni_p` and an affinity `dchi_eV` lower (Anderson offsets from
    physics_heterojunction.anderson_alignment, frame reference r_n), joined by the D1a carrier
    interface `model` ("continuous" quasi-Fermi level or "thermionic" emission with the free-electron
    Richardson constant x `astar_scale`). The 'hi' circuit contact on r_p pins bias + s(r_p) through the
    shared contact rule (psi_offset_V). Solved at zero bias; returns (device, frames)."""
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_heterojunction as HJ
    from dynameta.constants import M_E, V_T as VT_CODE    # the bipolar frame is written at V_T
    lines, _ = _fdt_lines()
    xj = lines[2][0]
    fresh()
    dev, mesh = "het", "het_m"
    ds.create_1d_mesh(mesh=mesh)
    for k, (pos, ps, tag) in enumerate(lines):
        ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, tag=tag or ("mid" if pos == xj else
                                                                  "t{}".format(k)))
    ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=mesh, material="A", region="r_n", tag1="lo", tag2="mid")
    ds.add_1d_region(mesh=mesh, material="B", region="r_p", tag1="mid", tag2="hi")
    ds.add_1d_interface(mesh=mesh, name="jn", tag="mid")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    nis = {"r_n": FDT_NI, "r_p": float(ni_p)}
    for r, nd in (("r_n", FDT_N), ("r_p", -FDT_N)):
        ds.node_model(device=dev, region=r, name="NetDoping", equation="{:.6e}".format(nd))
        with quiet():
            BP.setup_bipolar_region(dev, r, eps_static=11.7, n_dos_m3=N_DOS, n_i_m3=nis[r],
                                    mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=FDT_TAU,
                                    tau_p_s=FDT_TAU, fd_enhancement=False)
            BP.setup_equilibrium_seed_models(dev, r)
    own = {r: float(VT_CODE * math.log(N_DOS / nis[r])) for r in nis}
    fr = HJ.anderson_alignment(own, {"r_n": 4.0, "r_p": 4.0 - float(dchi_eV)}, "r_n")
    a_star = float(astar_scale) * HJ.richardson_constant_Am2K2(M_E)
    sides = [HJ.HeteroSide(r, fr[r][0], "N_dos", a_star, fd=False,
                           phi_v_V=fr[r][0] - float(VT_CODE * math.log(N_DOS * N_DOS / nis[r] ** 2)),
                           p_dos_param="N_dos_p", richardson_p=a_star) for r in ("r_n", "r_p")]
    PE.setup_interface(dev, "jn")
    HJ.setup_heterointerface_carriers(dev, "jn", sides[0], sides[1], model=model)
    with quiet():
        BP.setup_contact_ohmic_bipolar(dev, "lo", psi_offset_V=fr["r_n"][1])
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1",
                                               psi_offset_V=fr["r_p"][1])
    for r in ("r_n", "r_p"):
        ds.node_model(device=dev, region=r, name="_seed_psi", equation="{} + ({!r})".format(
            BP.equilibrium_seed_psi_expr(None), float(fr[r][1])))
        _seed(dev, ("Potential", "Electrons", "Holes"), region=r)
    _solve(1e8)
    return dev, fr


def _hetero_sides(dev, regs, chi, nis, astar_scale):
    """Anderson frames + HeteroSides of bipolar (Boltzmann) regions (shared by the hetero fixtures)."""
    from dynameta.carriers import physics_heterojunction as HJ
    from dynameta.constants import M_E, V_T as VT_CODE    # the bipolar frame is written at V_T
    own = {r: float(VT_CODE * math.log(N_DOS / nis[r])) for r in regs}
    fr = HJ.anderson_alignment(own, chi, regs[0])
    a_star = float(astar_scale) * HJ.richardson_constant_Am2K2(M_E)
    sides = {r: HJ.HeteroSide(r, fr[r][0], "N_dos", a_star, fd=False,
                              phi_v_V=fr[r][0] - float(VT_CODE * math.log(N_DOS * N_DOS / nis[r] ** 2)),
                              p_dos_param="N_dos_p", richardson_p=a_star) for r in regs}
    return fr, sides


#: 1-D isotype n-GaAs / N-Al0.3Ga0.7As (v0.15 gate T-c): D1a gate C's materials and doping, 400 nm each
ISO_ND, ISO_L, ISO_AE = 1e23, 400e-9, 1e10


def hetero_isotype_1d(*, astar_scale=1e-6, dev="iso1d"):
    """D1a gate C's isotype heterojunction (validation/devsim_heterojunction: GAAS / ALGAAS parameters,
    N_D = 1e17 cm^-3 both sides, 400 nm each) as a 1-D BIPOLAR (Boltzmann, so every noise source is
    Nyquist-exact) device: 'lo' grounded on the n-GaAs end, 'hi' (V1, circuit) on the N-AlGaAs end,
    joined by the thermionic interface 'jn' with the barrier-side Richardson constant x astar_scale
    (A* of each side's DOS mass). Solved at zero bias; returns the device name."""
    from validation.devsim_heterojunction import ALGAAS, GAAS, n_i_m3, nc_m3
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_heterojunction as HJ
    from dynameta.constants import M_E, V_T as VT_CODE
    fresh()
    mesh = dev + "_m"
    ds.create_1d_mesh(mesh=mesh)
    for pos, ps, tag in ((0.0, 4e-9, "lo"), (ISO_L, 0.25e-9, "mid"), (2 * ISO_L, 4e-9, "hi")):
        ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, tag=tag)
    ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    ds.add_1d_region(mesh=mesh, material="GaAs", region="ngaas", tag1="lo", tag2="mid")
    ds.add_1d_region(mesh=mesh, material="AlGaAs", region="nalgaas", tag1="mid", tag2="hi")
    ds.add_1d_interface(mesh=mesh, name="jn", tag="mid")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    mats = {"ngaas": GAAS, "nalgaas": ALGAAS}
    nc = {r: nc_m3(m["me"]) for r, m in mats.items()}
    nv = {r: nc_m3(m["mh"]) for r, m in mats.items()}
    nis = {r: n_i_m3(m) for r, m in mats.items()}
    for r, m in mats.items():
        ds.node_model(device=dev, region=r, name="NetDoping", equation="{:.6e}".format(ISO_ND))
        with quiet():
            BP.setup_bipolar_region(dev, r, eps_static=m["eps"], n_dos_m3=nc[r], n_dos_p_m3=nv[r],
                                    n_i_m3=nis[r], mobility_n_m2Vs=m["mu_n"], mobility_p_m2Vs=m["mu_p"],
                                    tau_n_s=1e-6, tau_p_s=1e-6, fd_enhancement=False)
            BP.setup_equilibrium_seed_models(dev, r)
    own = {r: float(VT_CODE * math.log(nc[r] / nis[r])) for r in mats}
    fr = HJ.anderson_alignment(own, {r: m["chi"] for r, m in mats.items()}, "ngaas")
    sides = [HJ.HeteroSide(r, fr[r][0], "N_dos",
                           float(astar_scale) * HJ.richardson_constant_Am2K2(mats[r]["me"] * M_E), fd=False,
                           phi_v_V=fr[r][0] - float(VT_CODE * math.log(nc[r] * nv[r] / nis[r] ** 2)),
                           p_dos_param="N_dos_p",
                           richardson_p=float(astar_scale) * HJ.richardson_constant_Am2K2(mats[r]["mh"] * M_E))
             for r in ("ngaas", "nalgaas")]
    PE.setup_interface(dev, "jn")
    HJ.setup_heterointerface_carriers(dev, "jn", sides[0], sides[1], model="thermionic")
    with quiet():
        BP.setup_contact_ohmic_bipolar(dev, "lo", psi_offset_V=fr["ngaas"][1])
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1",
                                               psi_offset_V=fr["nalgaas"][1])
    for r in mats:
        ds.node_model(device=dev, region=r, name="_seed_psi", equation="{} + ({!r})".format(
            BP.equilibrium_seed_psi_expr(None), float(fr[r][1])))
        _seed(dev, ("Potential", "Electrons", "Holes"), region=r)
    # at zero bias the terminal current is round-off, so the CIRCUIT's relative update never drops reliably
    # below ~1e-5 (MEASURED: it stalls at 1e-6 - 7e-5 while the device reaches 4e-15): solve_zero_current
    # relaxes the circuit test only and holds the device to 1e-5 (the same DEVSIM call as the former
    # _solve(ISO_AE, 1e-3, 100), plus that device check)
    solve_zero_current(ISO_AE)
    return dev


#: 2-D thermionic heterostack (v0.15 gate T-d): lateral width, partial top contact, junction, top
H2D_LX, H2D_XC, H2D_YJ, H2D_YT = 1e-6, 0.3e-6, 0.3e-6, 0.4e-6
H2D_AE = 1e9


def hetero_2d_stack(*, astar_scale=1e-3, dchi_eV=0.1, nd=1e22, hx=10e-9, ratio=5.0, split_x=None,
                    dev="h2d"):
    """A z-stacked 2-D ISOTYPE heterostack (v0.15 plan 4.3.1, gate T-d): n-type (N_D `nd`, n_i 1e16)
    bottom region 'r_lo' (y in [0, YJ]) under a top layer (y in [YJ, YT]) whose affinity is `dchi_eV`
    lower, joined by D1a's THERMIONIC interface 'jn' (free-electron A* x `astar_scale`). The lateral
    mesh is deliberately NON-UNIFORM: spacing hx over the contact (x < XC) growing to ratio*hx at
    x = LX. Grounded contact 'lo' on the whole bottom face, circuit contact 'hi' (V1) only on the top
    face over x < XC, so the current spreads and the adjoint field varies along the interface -- the
    per-node interface areas are then load-bearing. split_x: split the top layer at x = split_x into
    'r_hi' | 'r_hi2' (same material, carrier-continuous interface 'jv'), joined to r_lo by TWO
    thermionic interfaces 'jn' and 'jn2' that meet the vertical one at the corner node (split_x, YJ).
    Solved at zero bias; returns the device name."""
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_heterojunction as HJ
    fresh()
    mesh = dev + "_m"
    LX, XC, YJ, YT = H2D_LX, H2D_XC, H2D_YJ, H2D_YT
    ds.create_2d_mesh(mesh=mesh)
    pad = 20e-9                                    # DEVSIM's 2-D recipe: inert pads outside the contacts
    xlines = [(0.0, hx), (XC, hx), (LX, hx * ratio)]
    if split_x is not None:
        xlines.append((float(split_x), hx))
    for pos, ps in sorted(xlines):
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=pos, ps=ps)
    for pos, ps in ((-pad, pad), (0.0, 20e-9), (YJ, 2e-9), (YT, 10e-9), (YT + pad, pad)):
        ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=pos, ps=ps)
    tops = [("r_hi", 0.0, LX)] if split_x is None else [("r_hi", 0.0, split_x), ("r_hi2", split_x, LX)]
    ds.add_2d_region(mesh=mesh, material="A", region="r_lo", xl=0.0, xh=LX, yl=0.0, yh=YJ)
    for r, xl, xh in tops:
        ds.add_2d_region(mesh=mesh, material="B", region=r, xl=xl, xh=xh, yl=YJ, yh=YT)
    ds.add_2d_region(mesh=mesh, material="pad", region="pad_lo", xl=0.0, xh=LX, yl=-pad, yh=0.0)
    ds.add_2d_region(mesh=mesh, material="pad", region="pad_hi", xl=0.0, xh=LX, yl=YT, yh=YT + pad)
    ifaces = [("jn", "r_lo", tops[0])]
    if split_x is not None:
        ifaces.append(("jn2", "r_lo", tops[1]))
    for name, r0, (r1, xl, xh) in ifaces:
        ds.add_2d_interface(mesh=mesh, name=name, region0=r0, region1=r1, xl=xl, xh=xh, yl=YJ, yh=YJ,
                            bloat=1e-12)
    if split_x is not None:
        ds.add_2d_interface(mesh=mesh, name="jv", region0="r_hi", region1="r_hi2", xl=split_x,
                            xh=split_x, yl=YJ, yh=YT, bloat=1e-12)
    ds.add_2d_contact(mesh=mesh, name="lo", material="metal", region="r_lo", xl=0.0, xh=LX, yl=0.0,
                      yh=0.0, bloat=1e-12)
    ds.add_2d_contact(mesh=mesh, name="hi", material="metal", region="r_hi", xl=0.0, xh=XC, yl=YT,
                      yh=YT, bloat=1e-12)
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    regs = ["r_lo"] + [t[0] for t in tops]
    nis = {r: 1e16 for r in regs}
    for r in regs:
        ds.node_model(device=dev, region=r, name="NetDoping", equation="{:.6e}".format(nd))
        with quiet():
            BP.setup_bipolar_region(dev, r, eps_static=11.7, n_dos_m3=N_DOS, n_i_m3=nis[r],
                                    mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=1e-7,
                                    tau_p_s=1e-7, fd_enhancement=False)
            BP.setup_equilibrium_seed_models(dev, r)
    chi = {r: (4.0 if r == "r_lo" else 4.0 - float(dchi_eV)) for r in regs}
    fr, sides = _hetero_sides(dev, regs, chi, nis, astar_scale)
    for name, r0, (r1, _xl, _xh) in ifaces:
        PE.setup_interface(dev, name)
        HJ.setup_heterointerface_carriers(dev, name, sides[r0], sides[r1], model="thermionic")
    if split_x is not None:
        PE.setup_interface(dev, "jv")
        HJ.setup_heterointerface_carriers(dev, "jv", sides["r_hi"], sides["r_hi2"], model="continuous")
    with quiet():
        BP.setup_contact_ohmic_bipolar(dev, "lo", psi_offset_V=fr["r_lo"][1])
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1",
                                               psi_offset_V=fr["r_hi"][1])
    for r in regs:
        ds.node_model(device=dev, region=r, name="_seed_psi", equation="{} + ({!r})".format(
            BP.equilibrium_seed_psi_expr(None), float(fr[r][1])))
        _seed(dev, ("Potential", "Electrons", "Holes"), region=r)
    _solve(H2D_AE)
    return dev


#: 3-D thermionic heterostack (v0.15 T0 / T-d): graded lateral axis [nm], junction and top z [nm]
H3D_XS = (0.0, 20.0, 50.0, 100.0, 180.0, 300.0)
H3D_ZJ, H3D_ZT = 300.0, 400.0
H3D_AE = 1e8


def hetero_3d_stack(*, astar_scale=1e-3, dchi_eV=0.1, nd=1e22, jitter=0.0, exact=None, split_i=None,
                    seed=1, dev="h3d"):
    """The 3-D twin of hetero_2d_stack on a hand-built Kuhn-tensor mesh (devsim_3d._kuhn_tets, in memory):
    graded lateral axes H3D_XS (spacing ratio 6), bottom region 'r_lo' (z < ZJ) under the top layer
    (affinity dchi_eV lower) through the thermionic interface 'jn' (z = ZJ); grounded 'lo' on the whole
    bottom face, circuit 'hi' (V1) on the top face of the 2 x 2 corner cells only (spreading current).
    jitter: move every interior node column laterally by up to jitter x (the smallest spacing) (seeded),
    which makes the tets and the interface triangles non-well-centred; exact: regions to correct with
    devsim_geometry.apply_exact_geometry(negative='warn') right after create_device. split_i: split the
    top layer at x = H3D_XS[split_i] into 'r_hi' | 'r_hi2' (carrier-continuous interface 'jv'), with two
    thermionic interfaces 'jn' / 'jn2' that SHARE the line of nodes (x_split, y, ZJ) with 'jv' (gmsh
    surface groups are not de-duplicated the way the 2-D mesher assigns a shared node to one interface).
    Solved at zero bias; returns the device name."""
    import warnings
    from dynameta.carriers import devsim_3d as D3
    from dynameta.carriers import devsim_geometry as DG
    from dynameta.carriers import physics_equilibrium as PE
    from dynameta.carriers import physics_heterojunction as HJ
    fresh()
    mesh = dev + "_m"
    xs = np.asarray(H3D_XS, dtype=float)
    ys = xs.copy()
    zs = np.concatenate([np.linspace(0.0, 250.0, 11), np.linspace(250.0, H3D_ZJ, 11)[1:],
                         np.linspace(H3D_ZJ, H3D_ZT, 6)[1:]])
    P, T, cell = D3._kuhn_tets(xs, ys, zs)
    nx, ny, nz = len(xs), len(ys), len(zs)
    if jitter:
        rng = np.random.default_rng(seed)
        I, J, _K = (g.ravel() for g in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                                    indexing="ij"))
        jx, jy = rng.uniform(-jitter, jitter, (nx, ny)), rng.uniform(-jitter, jitter, (nx, ny))
        h = float(np.diff(xs).min())
        P = P.copy()
        ii, jj = (I > 0) & (I < nx - 1), (J > 0) & (J < ny - 1)
        P[ii, 0] += jx[I[ii], J[ii]] * h
        P[jj, 1] += jy[I[jj], J[jj]] * h
    ci = cell // ((ny - 1) * (nz - 1))
    kz = cell % (nz - 1)
    top = 0.5 * (zs[kz] + zs[kz + 1]) > H3D_ZJ
    groups3 = {"g_lo": T[~top]}
    if split_i is None:
        groups3["g_hi"] = T[top]
    else:
        groups3["g_hi"] = T[top & (ci < split_i)]
        groups3["g_hi2"] = T[top & (ci >= split_i)]
    kj = int(np.argmin(np.abs(zs - H3D_ZJ)))
    ax = (xs, ys, zs)
    I2, J2 = (g.ravel() for g in np.indices((nx - 1, ny - 1)))
    full = np.ones(I2.size, dtype=bool) if split_i is None else (I2 < split_i)
    groups2 = {"g_jn": D3._face_triangles(ax, 2, (I2[full], J2[full], np.full(int(full.sum()), kj - 1)),
                                          plus=True)}
    if split_i is not None:
        groups2["g_jn2"] = D3._face_triangles(ax, 2, (I2[~full], J2[~full],
                                                      np.full(int((~full).sum()), kj - 1)), plus=True)
        Jv, Kv = (g.ravel() for g in np.indices((ny - 1, nz - 1)))
        sel = zs[Kv] >= H3D_ZJ - 1e-9
        groups2["g_jv"] = D3._face_triangles(ax, 0, (np.full(int(sel.sum()), split_i - 1), Jv[sel], Kv[sel]),
                                             plus=True)
    groups2["g_c_lo"] = D3._face_triangles(ax, 2, (I2, J2, np.zeros(I2.size, dtype=int)), plus=False)
    sel = (I2 < 2) & (J2 < 2)
    groups2["g_c_hi"] = D3._face_triangles(ax, 2, (I2[sel], J2[sel], np.full(int(sel.sum()), nz - 2)),
                                           plus=True)
    D3.create_mesh_in_memory(mesh, P * 1e-9, groups3, groups2)
    tops = ["r_hi"] if split_i is None else ["r_hi", "r_hi2"]
    for g, r in [("g_lo", "r_lo")] + [("g_" + t[2:], t) for t in tops]:
        ds.add_gmsh_region(mesh=mesh, gmsh_name=g, region=r, material="A" if r == "r_lo" else "B")
    ifaces = [("jn", "r_lo", "r_hi")] + ([] if split_i is None else [("jn2", "r_lo", "r_hi2")])
    for name, r0, r1 in ifaces:
        ds.add_gmsh_interface(mesh=mesh, gmsh_name="g_" + name, region0=r0, region1=r1, name=name)
    if split_i is not None:
        ds.add_gmsh_interface(mesh=mesh, gmsh_name="g_jv", region0="r_hi", region1="r_hi2", name="jv")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="g_c_lo", region="r_lo", name="lo", material="metal")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="g_c_hi", region="r_hi", name="hi", material="metal")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    for r in (exact or ()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # negative shares are reported, not fatal here
            DG.apply_exact_geometry(dev, r, negative="warn")
    regs = ["r_lo"] + tops
    nis = {r: 1e16 for r in regs}
    for r in regs:
        ds.node_model(device=dev, region=r, name="NetDoping", equation="{:.6e}".format(nd))
        with quiet():
            BP.setup_bipolar_region(dev, r, eps_static=11.7, n_dos_m3=N_DOS, n_i_m3=nis[r],
                                    mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=1e-7,
                                    tau_p_s=1e-7, fd_enhancement=False)
            BP.setup_equilibrium_seed_models(dev, r)
    chi = {r: (4.0 if r == "r_lo" else 4.0 - float(dchi_eV)) for r in regs}
    fr, sides = _hetero_sides(dev, regs, chi, nis, astar_scale)
    for name, r0, r1 in ifaces:
        PE.setup_interface(dev, name)
        HJ.setup_heterointerface_carriers(dev, name, sides[r0], sides[r1], model="thermionic")
    if split_i is not None:
        PE.setup_interface(dev, "jv")
        HJ.setup_heterointerface_carriers(dev, "jv", sides["r_hi"], sides["r_hi2"], model="continuous")
    with quiet():
        BP.setup_contact_ohmic_bipolar(dev, "lo", psi_offset_V=fr["r_lo"][1])
        BP.setup_contact_ohmic_bipolar_circuit(dev, "hi", node_name="vhi", source_name="V1",
                                               psi_offset_V=fr["r_hi"][1])
    for r in regs:
        ds.node_model(device=dev, region=r, name="_seed_psi", equation="{} + ({!r})".format(
            BP.equilibrium_seed_psi_expr(None), float(fr[r][1])))
        _seed(dev, ("Potential", "Electrons", "Holes"), region=r)
    _solve(H3D_AE)
    return dev


# long Si p+ n diode for the shot-noise gate: N_A = 1e25 (2 um, short side), N_D = 1e22, n-side
# 5 L_p long (tau = 20 us, L_p = 157 um); low injection (p_n(0)/N_D = 1.7e-3 at 0.55 V).
SHOT_TAU, SHOT_NA, SHOT_ND = 2e-5, 1e25, 1e22
SHOT_V = [0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55]
SHOT_AE = 1e13                 # the p+ continuity residual floors near 1e25 * 1e-16 ~ 1e9


def build_shot_diode():
    Lp = math.sqrt(MU_P * VT * SHOT_TAU)
    xj = 2e-6
    lines = [(0.0, 50e-9, "hi"), (xj, 1e-9, None), (xj + 1e-6, 10e-9, None),
             (xj + 20e-6, 300e-9, None), (xj + 5 * Lp, Lp / 20, "lo")]
    # 'hi' (the circuit contact) is the p+ end at x = 0; 'lo' the grounded n end.
    fresh()
    mesh = "shot_m"
    ds.create_1d_mesh(mesh=mesh)
    for pos, ps, tag in lines:
        kw = {"tag": tag} if tag else {}
        ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, **kw)
    ds.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    ds.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region="bulk", tag1="hi", tag2="lo")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device="shot")
    ds.node_model(device="shot", region="bulk", name="NetDoping",
                  equation="ifelse(x < {}, {}, {})".format(xj, -SHOT_NA, SHOT_ND))
    with quiet():
        BP.setup_bipolar_region("shot", "bulk", eps_static=11.7, n_dos_m3=N_DOS, n_i_m3=1e16,
                                mobility_n_m2Vs=MU_N, mobility_p_m2Vs=MU_P, tau_n_s=SHOT_TAU,
                                tau_p_s=SHOT_TAU, fd_enhancement=False)
        BP.setup_equilibrium_seed_models("shot", "bulk")
        BP.setup_contact_ohmic_bipolar("shot", "lo")
        BP.setup_contact_ohmic_bipolar_circuit("shot", "hi", node_name="vhi", source_name="V1")
        ds.node_model(device="shot", region="bulk", name="_seed_psi",
                      equation=BP.equilibrium_seed_psi_expr(None))
    _seed("shot", ("Potential", "Electrons", "Holes"))
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.delete_by_name("shot", ceq)
    _solve(1e10, 1e-10, 100)
    for ceq in ("ElectronContinuityEquation", "HoleContinuityEquation"):
        _R.reapply_by_name("shot", ceq)
    _seed("shot", ("Electrons", "Holes"))
    _solve(SHOT_AE)
    set_bias._v = {}
    return "shot"


FREQS_C_HF = [1.0, 1.0e3, 1.0e5, 1.0e6]


def gate_c_shot():
    """Per bias: I, S(1 Hz), S/(2qI coth(qV/2kT)), S/(2qI), S/(4kT G_d), the diffusion-only share;
    at 0.5 V the van der Ziel ratio S(f)/(4kT Re Y(f) - 2qI) and S(f)/(2qI) over FREQS_C_HF."""
    build_shot_diode()
    rows = []
    hf = None
    for v in SHOT_V:
        set_bias(v, SHOT_AE)
        Gd = dc_conductance(v, SHOT_AE, dv=1e-3)
        I = source_current()
        sn = noise([1.0])
        S = float(sn.S[0])
        diff_only = float(sn.by_mechanism["diffusion_n"][0] + sn.by_mechanism["diffusion_p"][0])
        rows.append(dict(V=v, I=I, S=S, coth=S / (2 * QE * I / math.tanh(QE * v / (2 * KT))),
                         shot=S / (2 * QE * I), nyq=S / (4 * KT * Gd), diff_share=diff_only / S))
        if abs(v - 0.5) < 1e-9:
            s_hf = noise(FREQS_C_HF).S
            Y = ssac_Y(FREQS_C_HF)
            hf = dict(vdz=s_hf / (4 * KT * Y.real - 2 * QE * I), shot=s_hf / (2 * QE * I))
    return rows, hf


# GR slab: n-type 1e22, n_i = 1e21 (p0 = 1e20 -- a measurable minority), SRH tau = 10 ns
GR_ND, GR_NI, GR_TAU, GR_V = 1e22, 1e21, 1e-8, 0.05


def gr_constants():
    n0 = GR_ND / 2 + math.sqrt(GR_ND ** 2 / 4 + GR_NI ** 2)
    p0 = GR_NI ** 2 / n0
    Dn, Dp = MU_N * VT, MU_P * VT
    Da = (n0 + p0) * Dn * Dp / (n0 * Dn + p0 * Dp)                  # ambipolar diffusivity
    tau_eff = GR_TAU * (n0 + p0 + 2 * GR_NI) / (n0 + p0)            # 1/(dU/dn + dU/dp) at eq
    GR2 = 2 * (n0 * p0 + GR_NI ** 2) / (GR_TAU * (n0 + p0 + 2 * GR_NI))   # 2 (G + R)
    return n0, p0, Da, tau_eff, GR2


def gr_analytic(freqs, L, E):
    """Ambipolar GR Lorentzian of a slab of length L between ohmic contacts (delta p = 0 there):
    s(f) = (q (mu_n + mu_p) E)^2 2(G+R) / L^2  int_0^L |g(x)|^2 dx,
    g = tau~ [1 - cosh((x - L/2)/lam)/cosh(L/(2 lam))], tau~ = tau/(1 + i w tau), lam^2 = D_a tau~;
    and the plain (L -> inf) Lorentzian (q(mu_n+mu_p)E)^2 2(G+R) tau^2 / (L (1 + w^2 tau^2))."""
    n0, p0, Da, tau, GR2 = gr_constants()
    w = 2 * np.pi * np.asarray(freqs, dtype=float)
    taut = tau / (1 + 1j * w * tau)
    lam = np.sqrt(Da * taut)
    x = np.linspace(-L / 2, L / 2, 400001)
    pref = (QE * (MU_N + MU_P) * E) ** 2 * GR2
    ax = np.abs(x)

    def _ratio(ll):            # cosh(x/lam)/cosh(L/2lam) without overflow (Re(1/lam) > 0)
        return (np.exp((ax - L / 2) / ll) * (1 + np.exp(-2 * ax / ll)) / (1 + np.exp(-L / ll)))

    I2 = np.array([trapz(np.abs(t * (1 - _ratio(ll))) ** 2, x) for t, ll in zip(taut, lam)])
    return pref * I2 / L ** 2, pref * tau ** 2 / (L * (1 + (w * tau) ** 2))


def gate_d_gr(L_over_LD=50.0):
    """GR part of S_I on the biased slab vs the analytic Lorentzians at f/f_c = 1e-3 ... 100."""
    n0, p0, Da, tau, GR2 = gr_constants()
    LD = math.sqrt(Da * tau)
    L = L_over_LD * LD
    ps_mid = LD / 10 if L_over_LD <= 100 else LD / 3
    bipolar_1d("gr", [(0.0, LD / 200, "lo"), (L / 2, ps_mid, None), (L, LD / 200, "hi")],
               "{:.10e}".format(GR_ND), n_i=GR_NI, tau=GR_TAU, presolve=False, ae=1e8)
    set_bias._v = {}
    set_bias(GR_V, 1e8)
    fc = 1.0 / (2 * math.pi * tau)
    ff = fc * np.array([1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
    gr = noise(ff).by_mechanism["gr"]
    exact, plain = gr_analytic(ff, L, GR_V / L)
    return ff / fc, gr / exact, gr / plain, gr / gr[0]


def gate_e_resistor_bias(v=0.1, L=1e-5):
    """Biased uniform resistor: (S/4kTG, S/2qI). Diffusion noise = 4kTG; shot would be 2qI.
    The DD answer is 4kTG only up to a drift/Debye-screening correction near the contacts that is
    PHYSICAL (mesh-independent: 1.14e-3 at h = 10 nm vs 1.03e-3 at 2 nm), grows as V^2 and falls
    as ~L^-2.7 -- measured S/4kTG - 1 = 1.1e-3 at L = 1 um, 2.3e-6 at L = 10 um (0.1 V, n = 1e22,
    Debye length 41 nm). The gate uses L = 10 um."""
    bipolar_1d("resb", [(0.0, 2e-8, "lo"), (L, 2e-8, "hi")], "1e22", n_i=1e16, tau=1e-7, ae=1e8)
    set_bias._v = {}
    set_bias(v, 1e8)
    G = dc_conductance(v, 1e8)
    I = source_current()
    S = float(noise([1e3]).S[0])
    return S / (4 * KT * G), S / (2 * QE * I)


def gate_f_hooge(alpha=2e-3, v=0.05, L=1e-5, nd=1e22):
    """S_hooge * f / (alpha I^2 / N) at f = 1, 1e3, 1e6 (N = n L per unit area)."""
    bipolar_1d("hoo", [(0.0, L / 200, "lo"), (L, L / 200, "hi")], "{:.6e}".format(nd), n_i=1e16,
               tau=1e-7, ae=1e8)
    set_bias._v = {}
    set_bias(v, 1e8)
    I = source_current()
    ff = np.array([1.0, 1e3, 1e6])
    sn = noise(ff, hooge_alpha=alpha)
    sh = sn.by_mechanism["hooge_n"] + sn.by_mechanism["hooge_p"]
    return sh / (alpha * I * I / (nd * L * ff))


def gate_f3_hooge(alpha=2e-3, v=1e-3, n_bg=4e26, LX=200e-9, A=80e-9 * 80e-9):
    """3-D Hooge on the default Kuhn-tensor bar (unipolar_3d_resistor, 960 tets): S_hooge f / (alpha
    I^2 / N) - 1 at f = 1, 1e3 Hz with N = n_bg LX A (closed form), and its DERIVED budget. Budget: with
    uniform n and a linear potential the SG current is exact and noise_analysis's SG edge density is
    n_eff = n (v/2) coth(v/2) <= n (1 + v^2/12) (v = vdiff_g), so 1 - sqrt(n/n_eff) <= v^2/24 per edge;
    every P1 element couple of a Kuhn tet is >= 0 (no sign-mixed amplification), hence |r - 1| <=
    max v^2/12, plus 4 |I/I_ohm - 1| through the solved state and a 1e-11 summation floor. MEASURED
    (2026-11-06): -5.80e-9 vs budget 5.81e-9 (the deviation IS the SG density term); be719d5, with
    DEVSIM's circumcentric element couples, +0.125 (the defect this gate exists for)."""
    set_bias._v = {}
    unipolar_3d_resistor()
    ae = max(1e10, n_bg * 1e-12)
    set_bias(v, ae)
    I = source_current()
    dev = ds.get_device_list()[0]
    n = np.asarray(ds.get_node_model_values(device=dev, region="bulk", name="Electrons"))
    vg = np.asarray(ds.get_edge_model_values(device=dev, region="bulk", name="vdiff_g"))
    ohm = I / (QE * 0.004 * n_bg * v * A / LX) - 1.0
    ff = np.array([1.0, 1e3])
    sn = noise(ff, hooge_alpha=alpha)
    r = sn.by_mechanism["hooge_n"] / (alpha * I * I / (n_bg * LX * A * ff)) - 1.0
    budget = float(np.max(vg * vg)) / 12.0 + 4.0 * abs(ohm) + 1e-11
    return r, budget, dict(n_spread=float(n.max() / n.min() - 1.0), ohm=float(ohm))


FREQS_G = [1e9, 1e11, 1e13, 1e15]


def gate_g_layered(vg=0.0, freqs=None):
    """2-D LAYERED device: the reference metasurface (validation/_reference_device: degenerate ITO,
    FD g-factor ON, HfO2/Al2O3 stack, edge-metal grounds, 9 regions) through LayeredDevsimBuilder +
    set_ssac_gate, at gate bias vg (no DC current: equilibrium at every vg). Returns
    (S_I/(4kT Re Y), Re Y [S/m per unit y], per-region split)."""
    from validation._reference_device import build_reference_modulator
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    freqs = FREQS_G if freqs is None else freqs
    fresh()
    b = LayeredDevsimBuilder(build_reference_modulator("drift_diffusion"), mesh_name="lay_m",
                             device_name="lay_d")
    with quiet():
        b.solve(BiasPoint({"top_contact": vg}, "g{:+.1f}".format(vg)))
    b.set_ssac_gate("top_contact", source_name="V1")
    ds.circuit_alter(name="V1", value=float(vg))
    _solve(1e18, 1e-5, 60)
    sn = noise(freqs)
    Y = ssac_Y(freqs)
    return sn.S / (4 * KT * Y.real), Y.real, sn.by_region


def gate_h_thermionic():
    """GATE H (module doc): returns a dict of measured residuals (oracles: ssac Re Y, the coth law)."""
    import dataclasses
    out = {}
    F = [1.0, 1e5, 1e7, 1e9]
    hetero_two_region("continuous")
    s_cont = noise(F).S
    rows = []
    for a in (1e-6, 1e-4, 1e-2, 1.0, 1e2):
        hetero_two_region("thermionic", astar_scale=a)
        sn = noise(F)
        Y = ssac_Y(F)
        te = sn.by_mechanism["te_n"] + sn.by_mechanism["te_p"]
        rows.append((a, float(np.max(np.abs(sn.S / (4 * KT * Y.real) - 1))), float(te[0] / sn.S[0]),
                     float(np.max(np.abs(sn.S / s_cont - 1)))))
    out["H1"] = rows
    coth = []
    for sign in (+1, -1):
        set_bias._v = {}
        hetero_isotype_1d(astar_scale=1e-6)
        for m in (1, 2, 4, 8):
            set_bias(sign * m * VT, ISO_AE, step=0.01)
            J = source_current()
            coth.append((sign * m, float(noise([1.0]).S[0] / (2 * QE * abs(J) / math.tanh(m / 2.0)) - 1)))
    set_bias._v = {}
    out["H2"] = coth
    hetero_2d_stack()
    Y = ssac_Y([1.0, 1e6])
    r2 = noise([1.0, 1e6]).S / (4 * KT * Y.real) - 1
    orig = NA._interface_sources

    def equal(device, rn, mech):
        res = []
        for src in orig(device, rn, mech):
            sa = np.asarray(ds.get_node_model_values(device=device, region=src.region,
                                                     name="SurfaceArea"))[src.rows[0].nodes]
            res.append(dataclasses.replace(src, rate=src.rate * sa.mean() / sa))
        return res
    NA._interface_sources = equal
    try:
        r2e = noise([1.0, 1e6]).S / (4 * KT * Y.real) - 1
    finally:
        NA._interface_sources = orig
    out["H3"] = (float(np.max(np.abs(r2))), float(np.min(np.abs(r2e))))
    r3 = {}
    for tag, kw in (("kuhn", {}), ("jitter_exact", dict(jitter=0.3, exact=("r_lo", "r_hi"))),
                    ("split", dict(split_i=3))):
        hetero_3d_stack(**kw)
        r3[tag] = float(abs(noise([1.0]).S[0] / (4 * KT * ssac_Y([1.0]).real[0]) - 1))
    out["H4"] = r3
    return out


# =====================================================================================================

def main():
    ok = True

    def gate(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print("[t] GATE {}: {}".format(name, "PASS" if cond else "FAIL"), flush=True)

    print("[t] === DEVSIM noise (impedance-field method) vs independent oracles ===", flush=True)
    ja = gate_a_johnson()
    worst = 0.0
    for k, (r, G, Gan) in ja.items():
        worst = max(worst, float(np.max(np.abs(r - 1))))
        print("[t] A {:16s} G_dc={:.6e}{}  S/(4kTG) - 1 = {}".format(
            k, G, "" if Gan is None else " (sigma A/L {:.6e})".format(Gan),
            np.array2string(r - 1, precision=2)), flush=True)
    gate("A Johnson-Nyquist, 1-D/2-D/3-D (|S/4kTG - 1| < 1e-9)", worst < 1e-9)

    rI, rV, ReZ = gate_b_fdt()
    print("[t] B f = 1 Hz..10 GHz  Re Z = {:.3e} .. {:.3e} Ohm m^2".format(ReZ.max(), ReZ.min()),
          flush=True)
    print("[t] B S_I/(4kT Re Y) - 1 = {}".format(np.array2string(rI - 1, precision=1)), flush=True)
    print("[t] B S_V/(4kT Re Z) - 1 = {}".format(np.array2string(rV - 1, precision=1)), flush=True)
    gate("B fluctuation-dissipation, S_I and open-circuit S_V (< 1e-4)",
         np.max(np.abs(rI - 1)) < 1e-4 and np.max(np.abs(rV - 1)) < 1e-4)

    r2, ratio, regs = gate_b_two_region()
    print("[t] B two regions {} + carrier-continuous interface: S/(4kT Re Y) - 1 = {}  "
          "S_two/S_one - 1 = {}".format(regs, np.array2string(r2, precision=1),
                                        np.array2string(ratio - 1, precision=1)), flush=True)
    gate("B multi-region interface (FDT < 1e-4, == single region < 1e-4)",
         np.max(np.abs(r2)) < 1e-4 and np.max(np.abs(ratio - 1)) < 1e-4)

    rows, hf = gate_c_shot()
    for r in rows:
        print("[t] C V={:.2f} I={:.3e} A/m^2  S/(2qI coth)={:.5f}  S/2qI={:.5f}  S/(4kTG_d)={:.4f}"
              "  diffusion share={:.3f}".format(r["V"], r["I"], r["coth"], r["shot"], r["nyq"],
                                                r["diff_share"]), flush=True)
    print("[t] C 0.5 V, f={}: S/(4kT ReY - 2qI)={}  S/2qI={}".format(
        FREQS_C_HF, np.round(hf["vdz"], 4), np.round(hf["shot"], 3)), flush=True)
    gate("C ideal diode S = 2qI coth(qV/2kT) (< 0.3%), shot 2qI at >= 0.4 V (< 0.2%), "
         "S/(4kT G_d) -> 1/2 (< 1%)",
         all(abs(r["coth"] - 1) < 3e-3 for r in rows)
         and all(abs(r["shot"] - 1) < 2e-3 for r in rows if r["V"] >= 0.4)
         and all(abs(r["nyq"] - 0.5) < 1e-2 for r in rows if r["V"] >= 0.45))
    gate("C van der Ziel S(f) = 4kT ReY(f) - 2qI, 1 Hz..1 MHz (< 1.5%)",
         np.max(np.abs(hf["vdz"] - 1)) < 1.5e-2)

    for Lr in (50.0, 400.0):
        x, r_exact, r_plain, shape = gate_d_gr(Lr)
        print("[t] D L={:.0f} L_D  f/f_c={}  GR/exact={}  GR/plain={}  S(f)/S(0)={}".format(
            Lr, x, np.round(r_exact, 5), np.round(r_plain, 4), np.round(shape, 4)), flush=True)
        if Lr == 50.0:
            gate("D GR noise == ambipolar Lorentzian with contact shorting (< 1e-3)",
                 np.max(np.abs(r_exact - 1)) < 1e-3)
        else:
            gate("D plain Lorentzian plateau (< 1.5%) and corner S(f_c)/S(0) = 1/2 (+-0.01)",
                 abs(r_plain[0] - 1) < 1.5e-2 and abs(shape[3] - 0.5) < 1e-2)

    nyq, shot = gate_e_resistor_bias()
    print("[t] E biased resistor 0.1 V: S/(4kTG)={:.6f}  S/(2qI)={:.4f} (shot would be 1)".format(
        nyq, shot), flush=True)
    gate("E diffusion vs shot: resistor S = 4kTG (< 1e-4) != 2qI; diode S = 2qI = 4kTG_d/2",
         abs(nyq - 1) < 1e-4 and abs(shot - 1) > 0.3
         and all(abs(r["nyq"] - 0.5) < 1e-2 for r in rows if r["V"] >= 0.45))

    rh = gate_f_hooge()
    print("[t] F Hooge S_H f/(alpha I^2/N) = {}".format(np.round(rh, 6)), flush=True)
    gate("F Hooge 1/f (< 1e-4)", np.max(np.abs(rh - 1)) < 1e-4)

    r3, b3, chk3 = gate_f3_hooge()
    print("[t] F3 Hooge 3-D Kuhn bar: S_H f/(alpha I^2/N) - 1 = {}  budget {:.3e}  fixture {}".format(
        np.array2string(r3, precision=3), b3, chk3), flush=True)
    gate("F3 Hooge 1/f 3-D (|r - 1| <= derived SG budget; fixture n uniform, I ohmic < 1e-9)",
         np.max(np.abs(r3)) <= b3 and abs(chk3["n_spread"]) < 1e-12 and abs(chk3["ohm"]) < 1e-9)

    for vg, tol in ((0.0, 1e-9), (2.0, 1e-3)):
        rg, ReY, byr = gate_g_layered(vg)
        print("[t] G layered metasurface Vg={:+.1f} V  Re Y={}  S/(4kT ReY) - 1 = {}  regions={}"
              .format(vg, np.array2string(ReY, precision=3), np.array2string(rg - 1, precision=2),
                      sorted(byr)), flush=True)
        gate("G 2-D layered builder FDT at Vg={:+.0f} V (< {:.0e})".format(vg, tol),
             np.max(np.abs(rg - 1)) < tol)

    h = gate_h_thermionic()
    for a, r, share, dc in h["H1"]:
        print("[t] H1 A* x {:g}: |S/(4kT Re Y) - 1| = {:.2e}  TE share {:.3e}  |S/S_cont - 1| = {:.2e}".format(
            a, r, share, dc), flush=True)
    gate("H1 TE interface FDT at every A* (< 1e-4), TE share and |S/S_cont - 1| monotone",
         all(r < 1e-4 for _a, r, _s, _d in h["H1"])
         and all(x[2] > y[2] and x[3] > y[3] for x, y in zip(h["H1"], h["H1"][1:])))
    print("[t] H2 isotype coth law S/(2qJ coth(qV/2kT)) - 1 at V/(kT/q) = {}".format(
        ", ".join("{:+d}: {:.1e}".format(m, r) for m, r in h["H2"])), flush=True)
    gate("H2 isotype n-GaAs/N-AlGaAs coth law (< 3e-3)", all(abs(r) < 3e-3 for _m, r in h["H2"]))
    print("[t] H3 2-D non-uniform interface FDT {:.2e}; equal shares {:.2e}".format(*h["H3"]), flush=True)
    gate("H3 2-D TE FDT (< 1e-4) and equal area shares break it (> 1e-3)",
         h["H3"][0] < 1e-4 and h["H3"][1] > 1e-3)
    print("[t] H4 3-D TE FDT {}".format({k: "{:.1e}".format(v) for k, v in h["H4"].items()}), flush=True)
    gate("H4 3-D TE FDT, Kuhn / jittered + exact geometry / shared nodes (< 1e-4)",
         all(v < 1e-4 for v in h["H4"].values()))

    fresh()
    print("[t] *** DEVSIM NOISE: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
