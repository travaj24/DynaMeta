"""v0.16: the Kane (non-parabolic) band in the DEVSIM equilibrium electrostatics -- hand-built 1-D devices, the 2-D
LayeredDevsimBuilder and the 3-D Stacked3DSpec.

ORACLE. An independent 1-D box-method nonlinear Poisson solve written here (numpy/scipy Newton), with the SAME
band through a DIFFERENT code path: n = N_c(m0) [F_AH(eta) + Delta_q(eta)], F_AH the Aymerich-Humet fit
re-typed below (the parabolic part every equilibrium closure keeps, tests/test_mesh_screening_grading.py does
the same) and Delta_q the Kane increment from scipy QUADRATURE of the defining integral (tabulated every 0.05
in eta, cubic spline; interpolation error ~1e-9), Phi_c0 from bisection on that closure. On the DEVSIM node
positions (1-D and the 3-D tensor column) the two discretizations are identical, so the comparison isolates
the closure, its derivative and the wiring: budget = the increment's evaluation error (<= 1.5e-6 of n at
b = 0.0103, tests/test_kane_band.py) + the Newton tolerances. On the 2-D builder's mesh the oracle uses its own
graded grid and the budget is the mesh error of the default screening grading (tests/test_mesh_screening_grading.py:
~1e-4 at +5 V, ~4e-4 at -3 V on this film).

Band: m0 = 0.22 m_e, alpha = 0.4 /eV (b = 0.0103), n_bg = 3e26 m^-3, eps_s 9.0 (eta_bg = 23.1).
On the library before KaneBand every gate here fails: KaneBand / setup_semiconductor_region(band=) /
TransportModel(band=) / Stacked3DSpec(bands=) do not exist (ImportError / TypeError).
"""
import contextlib
import io
import math
import os
import re
import sys
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")
from dynameta.carriers.devsim_env import import_devsim     # noqa: E402

ds = import_devsim()

from scipy import integrate                                              # noqa: E402
from scipy.interpolate import CubicSpline                                # noqa: E402

from dynameta.carriers import eq_registry as _R                          # noqa: E402
from dynameta.carriers import physics_equilibrium as PE                  # noqa: E402
from dynameta.carriers.devsim_layered import LayeredDevsimBuilder, screening_length_m   # noqa: E402
from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec  # noqa: E402
from dynameta.materials.transport_model import KaneBand, TransportModel  # noqa: E402
from dynameta.sweep import BiasPoint                                      # noqa: E402

import _equilibrium_mos as F                                              # noqa: E402

EPS0, Q, KB, HBAR, M_E, T = 8.8541878128e-12, 1.602176634e-19, 1.380649e-23, 1.054571817e-34, \
    9.1093837015e-31, 300.0
VT = KB * T / Q
N_BG, EPS_S = 3.0e26, 9.0
BAND = KaneBand(0.22 * M_E, 0.4)
B_KT = 0.4 * VT


@pytest.fixture(autouse=True)
def _clean():
    yield
    for dv in list(ds.get_device_list()):
        _R.clear(dv)
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)


# ---------------------------------------------------------------------------------------------------
# the oracle closure: Aymerich-Humet F_1/2 (re-typed) + the QUADRATURE Kane increment
# ---------------------------------------------------------------------------------------------------
def _F_AH(eta):
    eta = np.asarray(eta, float)
    a = eta ** 4 + 50.0 + 33.6 * eta * (1.0 - 0.68 * np.exp(-0.17 * (eta + 1.0) ** 2))
    return 1.0 / ((3.0 * np.sqrt(np.pi) / 4.0) / a ** 0.375 + np.exp(-eta))


def _G_quad(eta, b):
    def dK(x):
        return 1.5 * math.sqrt(x + b * x * x) * (1.0 + 2.0 * b * x)
    top = max(eta, 0.0) + 60.0
    pts = [q for q in (eta - 5.0, eta, eta + 5.0) if 0.0 < q < top]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v, _e = integrate.quad(lambda x: 0.5 * (1.0 - math.tanh(0.5 * (x - eta))) * dK(x), 0.0, top,
                               points=pts or None, limit=400, epsabs=0.0, epsrel=1e-13)
    return v / math.gamma(2.5)


class OracleClosure:
    """n(psi), dn/dpsi of a band for the oracle. kane=False: the parabolic closure with DOS mass m (the
    'frozen-mass' model when m is the band's DOS mass at n_bg)."""
    _table = {}

    def __init__(self, kane=True, m=None, b=B_KT, m0=BAND.m0_kg):
        self.kane = kane
        self.Nc = 2.0 * ((m0 if kane else m) * KB * T / (2.0 * np.pi * HBAR ** 2)) ** 1.5
        if kane:
            key = round(b, 15)
            if key not in self._table:
                eg = np.arange(-40.0, 75.0001, 0.05)
                d = np.array([_G_quad(e, b) - _G_quad(e, 0.0) for e in eg])
                self._table[key] = CubicSpline(eg, d)
            self.spl = self._table[key]
        lo, hi = -20.0, 80.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.G(mid) < N_BG / self.Nc else (lo, mid)
        self.phi_c0 = -0.5 * (lo + hi) * VT

    def G(self, eta):
        return _F_AH(eta) + (self.spl(eta) if self.kane else 0.0)

    def dG(self, eta):
        d = (_F_AH(eta + 1e-6) - _F_AH(eta - 1e-6)) / 2e-6
        return d + (self.spl(eta, 1) if self.kane else 0.0)

    def __call__(self, psi):
        eta = (psi - self.phi_c0) / VT
        if self.kane:
            assert eta.min() > -40.0 and eta.max() < 75.0, (eta.min(), eta.max())
        return self.Nc * self.G(eta), self.Nc * self.dG(eta) / VT


def oracle_1d(z, eps_cell, semi_cell, closure, v_left, v_right=None):
    """Box-method nonlinear Poisson on nodes z: Dirichlet psi = v_left at z[0]; z[-1] Dirichlet v_right, or
    zero-flux when None. eps_cell / semi_cell per cell. Returns psi (all nodes) and the semiconductor node
    mask. Newton with a 0.1 V step cap, to |dpsi| < 1e-14 V."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    h = np.diff(z)
    g = np.asarray(eps_cell) * EPS0 / h
    sc = np.asarray(semi_cell, float)
    vol = np.zeros(z.size)
    vol[:-1] += 0.5 * h * sc
    vol[1:] += 0.5 * h * sc
    semi = vol > 0
    psi = np.full(z.size, 0.0)
    psi[0] = v_left
    if v_right is not None:
        psi[-1] = v_right
    for _ in range(400):
        n, dn = np.zeros(z.size), np.zeros(z.size)
        n[semi], dn[semi] = closure(psi[semi])
        r = np.zeros(z.size)
        fl = g * (psi[1:] - psi[:-1])
        r[:-1] += fl
        r[1:] -= fl
        r -= Q * vol * (n - N_BG)
        main = -np.r_[g, 0.0] - np.r_[0.0, g] - Q * vol * dn
        J = diags([main, g, g], [0, 1, -1], format="lil")
        J[0, :] = 0
        J[0, 0] = 1.0
        r[0] = psi[0] - v_left
        if v_right is not None:
            J[-1, :] = 0
            J[-1, -1] = 1.0
            r[-1] = psi[-1] - v_right
        d = spsolve(J.tocsr(), -r)
        step = np.max(np.abs(d))
        psi += d * min(1.0, 0.1 / step) if step > 0 else d
        if step < 1e-14:
            break
    else:                                                                 # pragma: no cover
        raise RuntimeError("oracle Newton did not converge")
    return psi, semi


# ---------------------------------------------------------------------------------------------------
# hand-built 1-D MOS: gate | oxide 8 nm (eps 20) | film 30 nm | zero-flux end
# ---------------------------------------------------------------------------------------------------
T_OX, EPS_OX, L_FILM = 8e-9, 20.0, 30e-9


def _mos(dev, *, band=BAND, dos_mass_kg=None, fermi_ref_param=None):
    F.clear(ds, dev)
    F.build_mesh(ds, dev, t_ox=T_OX, l_semi=L_FILM)
    kw = {"band": band} if dos_mass_kg is None else {"dos_mass_kg": dos_mass_kg}
    if fermi_ref_param is not None:
        kw["fermi_ref_param"] = fermi_ref_param
    with F.quiet():
        PE.setup_dielectric_region(dev, "ox", EPS_OX)
        PE.setup_semiconductor_region(dev, "semi", n_bg_m3=N_BG, eps_static=EPS_S, **kw)
        PE.setup_interface(dev, "ox_semi")
        PE.setup_contact(dev, "gate")


def _walk(dev, v, steps):
    v0 = float(ds.get_parameter(device=dev, name="gate_bias"))
    for k in range(1, steps + 1):
        ds.set_parameter(device=dev, name="gate_bias", value=v0 + (v - v0) * k / steps)
        F.solve(ds, rel_err=1e-5)


def _mos_arrays(dev):
    xo, xs = F.vals(ds, dev, "ox", "x"), F.vals(ds, dev, "semi", "x")
    return dict(xo=xo, xs=xs, po=F.vals(ds, dev, "ox", "Potential"), ps=F.vals(ds, dev, "semi", "Potential"),
                n=F.vals(ds, dev, "semi", "Electrons"))


def _mos_oracle(a, vg, closure):
    z = np.unique(np.r_[a["xo"], a["xs"]])
    zi = T_OX
    cells_semi = 0.5 * (z[1:] + z[:-1]) > zi
    eps = np.where(cells_semi, EPS_S, EPS_OX)
    psi, semi = oracle_1d(z, eps, cells_semi, closure, vg, None)
    n = np.zeros(z.size)
    n[semi] = closure(psi[semi])[0]
    o = np.argsort(a["xs"])
    zs = z[semi]
    assert np.allclose(zs, a["xs"][o], rtol=0, atol=1e-18)
    return psi[semi], n[semi], o


@pytest.mark.parametrize("vg,steps", [(5.0, 5), (-3.0, 3)])
def test_1d_kane_film_matches_the_independent_poisson_solve(vg, steps):
    """Accumulation (+5 V) and depletion (-3 V) of the 1-D MOS with the Kane band, against the oracle on the
    same nodes: every film node's density to 2e-5, the surface potential to 1e-7 V (MEASURED 4.1e-11 / 1.2e-6
    and 1.0e-10 / 1.1e-8 V at +5 / -3 V; at +5 V every film node is in the Sommerfeld branch). The same device
    with the parabolic band frozen at the band's DOS mass at n_bg (the v0.15.1 model of this material): at +5 V
    the Kane face density is +22.3 % (1.385e27 vs 1.133e27 m^-3) at 0.742 V of band bending instead of 0.853 V,
    the excess sheet density +2.7 %; at -3 V the depleted face (eta ~ 0.2, near-Boltzmann) is +19.8 % (2.445e24
    vs 2.042e24 m^-3; psi_s -0.592 vs -0.607 V): there the Kane closure runs on N_c(m0), the frozen one on the
    heavier DOS mass at n_bg."""
    _mos("k")
    _walk("k", vg, steps)
    a = _mos_arrays("k")
    psi_o, n_o, o = _mos_oracle(a, vg, OracleClosure(kane=True))
    rel = np.max(np.abs(a["n"][o] / n_o - 1.0))
    assert rel < 2e-5, rel
    assert abs(a["ps"][o][0] - psi_o[0]) < 1e-7
    # the frozen-mass parabolic model of the same material, in DEVSIM (dos_mass_kg = band DOS mass at n_bg)
    _mos("f", dos_mass_kg=BAND.dos_mass_kg(N_BG))
    _walk("f", vg, steps)
    nf = _mos_arrays("f")["n"][o]
    face = a["n"][o][0] / nf[0]
    # MEASURED 1.2225 (+5 V) / 1.1975 (-3 V): windows -2.0 % .. +2.2 % and -0.09 % .. +0.21 % around them
    if vg > 0:
        assert 1.198 < face < 1.249, face
    else:
        assert 1.1964 < face < 1.2000, face


def _newton_errors(dev, vg, tmp_path):
    """The per-iteration device AbsError DEVSIM prints for one solve at gate vg."""
    out = tmp_path / "newton.txt"
    buf = io.StringIO()                       # DEVSIM reports through Python's sys.stdout when it is replaced
    sys.stdout.flush()
    saved = os.dup(1)
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.dup2(fd, 1)                        # ... and through the C stdout otherwise
        with contextlib.redirect_stdout(buf):
            ds.set_parameter(device=dev, name="gate_bias", value=vg)
            with contextlib.suppress(Exception):
                ds.solve(type="dc", absolute_error=1e-30, relative_error=1e-30, maximum_iterations=12)
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(fd)
        os.close(saved)
    txt = out.read_text(errors="replace") + buf.getvalue()
    return [float(m) for m in re.findall(r'Device: "%s"\s+RelError: \S+\s+AbsError: (\S+)' % dev, txt)]


def test_newton_converges_quadratically(tmp_path):
    """From the converged +4 V point to +5 V: once below 1e-2 V the potential updates fall quadratically,
    e_k+1 <= 10 e_k^2, until the round-off floor; MEASURED 1.0 (the 1 V contact step) -> 3.8e-2 -> 2.7e-2 -> 2.1e-2
    -> 2.8e-5 -> 1.5e-10 -> 4.5e-16 (floor). The floor is the rounding of a Newton update whose unknowns are
    potentials up to |V| = 5 V: one ulp of 5 V is 1.1e-15 V (MEASURED 0.4 ulp), so "reached the floor" is gated as
    min e <= 64 ulp of 5 V = 7.1e-14 V, independent of the BLAS's last bits. NEGATIVE CONTROL: with the Kane part
    of the Jacobian removed (KaneDelta:Potential = 0) the same step converges only linearly (rate ~0.83; 1.6e-4
    after 12 iterations, 2e9 x that bar), so the gate sees a wrong derivative."""
    _mos("q")
    _walk("q", 4.0, 4)
    e = _newton_errors("q", 5.0, tmp_path)
    pairs = [(a, b) for a, b in zip(e, e[1:]) if 1e-9 < a < 1e-2 and b > 1e-15]
    assert pairs, e
    assert all(b <= 10.0 * a * a for a, b in pairs), e
    floor = 64.0 * np.finfo(float).eps * 5.0                  # 64 ulp of the largest potential (5 V)
    assert min(e) <= floor, e
    _mos("w")
    _walk("w", 4.0, 4)
    ds.node_model(device="w", region="semi", name="KaneDelta:Potential", equation="0")
    e2 = _newton_errors("w", 5.0, tmp_path)
    assert len(e2) == 12 and e2[-1] > 1e-5 and min(e2) > floor, e2
    assert all(b > 0.5 * a for a, b in zip(e2[-6:], e2[-5:])), e2


def test_screening_length_is_the_closure_slope_in_devsim():
    """screening_length_m(TransportModel(band=BAND)) equals sqrt(eps eps0 / (q dn/dpsi)) with dn/dpsi the
    DEVSIM model's analytic derivative Electrons:Potential in the neutral film, and its central difference of
    Electrons at psi = +-1e-6 V (1e-6; MEASURED 6.1e-12 / 4.4e-11). 0.7456 nm against 0.8137 nm for the frozen-mass
    parabolic band."""
    _mos("s")
    dn = F.vals(ds, "s", "semi", "Electrons:Potential")
    assert np.ptp(dn) == 0.0
    nn = F.vals(ds, "s", "semi", "x").size
    ds.set_node_values(device="s", region="semi", name="Potential", values=[1e-6] * nn)
    up = F.vals(ds, "s", "semi", "Electrons")[0]
    ds.set_node_values(device="s", region="semi", name="Potential", values=[-1e-6] * nn)
    dn_fd = (up - F.vals(ds, "s", "semi", "Electrons")[0]) / 2e-6
    Ls = screening_length_m(TransportModel(N_BG, EPS_S, band=BAND))
    for d in (dn[0], dn_fd):
        assert math.sqrt(EPS_S * EPS0 / (Q * d)) == pytest.approx(Ls, rel=1e-6, abs=0)
    assert Ls * 1e9 == pytest.approx(0.7456, abs=1e-4)


def test_kane_closure_follows_the_fermi_reference():
    """The Kane closure is built on equilibrium_eta_expr, so the edge Fermi-level reference (EF_ref_V) applies:
    EF_ref_V = 0.3 V with the gate at 5.3 V equals the unreferenced film at 5 V with every potential shifted by
    0.3 V (closure algebra; MEASURED 6.7e-16 relative in n, 1.7e-16 V)."""
    _mos("r", fermi_ref_param=PE.FERMI_REF_PARAM)
    ds.set_parameter(device="r", region="semi", name=PE.FERMI_REF_PARAM, value=0.3)
    ds.set_node_values(device="r", region="semi", name="Potential", values=[0.3] * F.vals(ds, "r", "semi", "x").size)
    ds.set_node_values(device="r", region="ox", name="Potential", values=[0.3] * F.vals(ds, "r", "ox", "x").size)
    ds.set_parameter(device="r", name="gate_bias", value=0.3)
    F.solve(ds, rel_err=1e-5)
    _walk("r", 5.3, 5)
    a1 = _mos_arrays("r")
    ds.delete_device(device="r")
    _mos("u")
    _walk("u", 5.0, 5)
    a0 = _mos_arrays("u")
    assert np.max(np.abs(a1["n"] / a0["n"] - 1.0)) < 1e-12
    assert np.max(np.abs(a1["ps"] - a0["ps"] - 0.3)) < 1e-12


_LOGGED = ("node_model", "edge_model", "contact_node_model", "interface_model", "equation",
           "contact_equation", "interface_equation", "node_solution", "edge_from_node_model", "set_parameter",
           "set_node_values")


def _setup_log(**kw):
    from _calllog import intercepting
    log = []
    with intercepting(ds, _LOGGED, log):
        _mos("l", **kw)
    return log


def test_alpha_zero_band_is_the_parabolic_device_bit_for_bit():
    """KaneBand(m, 0) makes exactly the definition calls of dos_mass_kg = m (model strings, parameters,
    seeds), so every solution is bit-identical; alpha = 1e-12 /eV (b = 2.6e-14) differs from it by its own
    physical increment only (MEASURED 1.3e-12 relative in n at +5 V; (15/4) b F_3/2/F_1/2 is of that size):
    the closure is continuous at alpha -> 0."""
    m = BAND.m0_kg
    assert _setup_log(band=KaneBand(m, 0.0)) == _setup_log(dos_mass_kg=m)
    out = {}
    for tag, kw in (("p", dict(dos_mass_kg=m)), ("z", dict(band=KaneBand(m, 0.0))),
                    ("t", dict(band=KaneBand(m, 1e-12)))):
        _mos(tag, **kw)
        _walk(tag, 5.0, 5)
        out[tag] = _mos_arrays(tag)
        ds.delete_device(device=tag)
    assert np.array_equal(out["p"]["n"], out["z"]["n"]) and np.array_equal(out["p"]["ps"], out["z"]["ps"])
    assert 0.0 < np.max(np.abs(out["t"]["n"] / out["p"]["n"] - 1.0)) < 1e-10
    with pytest.raises(ValueError, match="exactly one of dos_mass_kg"):
        PE.setup_semiconductor_region("nodev", "semi", n_bg_m3=N_BG, eps_static=EPS_S, dos_mass_kg=m, band=BAND)


# ---------------------------------------------------------------------------------------------------
# 2-D LayeredDevsimBuilder: the gated film (tests/_wiring_fixture.py) with a Kane material
# ---------------------------------------------------------------------------------------------------
def _kane_materials(band, chi_eV=4.5):
    """tests/_wiring_fixture.materials() with the film's transport on `band` (and its Drude mass from it)."""
    from dynameta.materials import ConstantOptical, DrudeOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    for nm, eps, kw in (("air", 1.0, {}), ("glass", 1.45 ** 2, {}), ("oxA", 1.70 ** 2, dict(eps_static_dc=9.0)),
                        ("oxB", 2.00 ** 2, dict(eps_static_dc=20.0)), ("metal", -60.0 + 4.0j, dict(is_metal=True)),
                        ("mirror_metal", -120.0 + 20.0j, dict(is_metal=True)),
                        ("TGATE", 3.8 + 0.05j, dict(is_metal=True))):
        reg.add(Material(nm, ConstantOptical(complex(eps)), **kw))
    reg.add(Material("TCO", optical=DrudeOptical(eps_inf=4.0, m_opt_kg=band.optical_mass(), gamma_rad_s=1.2e14),
                     transport=TransportModel(n_bg_m3=N_BG, eps_static=EPS_S, physics="equilibrium", band=band,
                                              chi_eV=chi_eV)))
    return reg


_NAMES_2D = ("node_model", "edge_model", "contact_node_model", "interface_model", "equation",
             "contact_equation", "interface_equation", "node_solution", "edge_from_node_model",
             "create_2d_mesh", "add_2d_mesh_line", "add_2d_region", "add_2d_interface", "add_2d_contact",
             "set_parameter", "set_node_values")


def test_builder_alpha_zero_band_is_the_constant_mass_device():
    """The gated film whose material carries KaneBand(0.30 m_e, 0) builds the constant-mass device: its build_device
    call log has the digest recorded with the library before the edge reference for the constant-0.30 m_e film
    (tests/test_layered_edge_reference.py PRE_EDGE_LOGS['film_no_edge']), i.e. every default result is unchanged."""
    import _wiring_fixture as W
    from _calllog import digest, intercepting, recording
    reg = _kane_materials(KaneBand(0.30 * M_E, 0.0), chi_eV=None)
    log = []
    with recording(True):
        with intercepting(ds, _NAMES_2D, log):
            b = LayeredDevsimBuilder(W.film_design(x_lines=None, mats=reg), mesh_name="ms_mesh",
                                     device_name="ms_device")
            b.build_device()
    b.teardown()
    assert (digest(log), len(log)) == ("ac8dda7a9eb032498d4fe321b1171cfe8137a0a2f5aa81fd9145aadfc8678225", 331)


def _graded(a, b, h0, ratio, hmax):
    L = b - a
    s, h = [0.0], h0
    while s[-1] < L / 2:
        s.append(s[-1] + h)
        h = min(h * ratio, hmax)
    s = np.array(s)
    s = s[s < L / 2]
    return a + np.unique(np.r_[s, L / 2, L - s[::-1]])


def _film_oracle(vg, closure, t_film=6e-9):
    """Top-face density of the gated film (oxA 2 nm eps 9 / oxB 6 nm eps 20 on both faces, Dirichlet
    gates on the outer faces, bottom gate 0 V) on a fine graded grid (the k = 2 oracle grid of
    tests/test_mesh_screening_grading.py)."""
    layers = [(2e-9, 9.0, False), (6e-9, 20.0, False), (t_film, EPS_S, True), (6e-9, 20.0, False),
              (2e-9, 9.0, False)]
    z, eps, semi, z0 = [np.array([0.0])], [], [], 0.0
    for t, e, s in layers:
        pts = (_graded(z0, z0 + t, 5e-14, 1.01, 2.5e-12) if s
               else np.linspace(z0, z0 + t, max(2, int(np.ceil(t / 5e-11))) + 1))
        z.append(pts[1:])
        eps += [e] * (len(pts) - 1)
        semi += [s] * (len(pts) - 1)
        z0 += t
    z = np.concatenate(z)
    psi, sm = oracle_1d(z, eps, semi, closure, 0.0, vg)
    i = np.where(sm)[0]
    return float(closure(psi[i[-1:]])[0][0]), float(closure(psi[i[:1]])[0][0])


@pytest.mark.parametrize("vg,bar", [(5.0, 3e-4), (-3.0, 2e-3)])
def test_builder_gated_film_with_the_kane_band(vg, bar):
    """The 2-D builder solves the Kane closure: the centre column's top-face density matches the oracle to the
    default mesh's error (bars as tests/test_mesh_screening_grading.py: 3e-4 at +5 V, 2e-3 at -3 V; MEASURED
    -1.44e-4 / +2.42e-4 with 212 z nodes across the film), and no density-dependent-DOS-mass advisory is raised (the
    suite runs with warnings as errors). Against the frozen-mass parabolic model of the same material (oracle) the
    Kane face density is +17.7 % at +5 V and -5.2 % at -3 V."""
    import _wiring_fixture as W
    b = LayeredDevsimBuilder(W.film_design(mats=_kane_materials(BAND)), mesh_name="mk", device_name="dk")
    b.solve(BiasPoint({"top": vg, "bot": 0.0}, "k"), abs_tol=1e-14)
    cols = W.node_columns(ds, b.device)
    xs = sorted(cols)
    top = float(cols[xs[len(xs) // 2]][-1])
    b.teardown()
    n_or, _bot = _film_oracle(vg, OracleClosure(kane=True))
    assert abs(top / n_or - 1.0) < bar, top / n_or - 1.0
    n_fr, _ = _film_oracle(vg, OracleClosure(kane=False, m=BAND.dos_mass_kg(N_BG)))
    # MEASURED 1.1773 (+5 V) / 0.9479 (-3 V): windows -2.0 % .. +2.2 % and -0.13 % .. +0.27 % around them
    if vg > 0:
        assert 1.154 < n_or / n_fr < 1.203, n_or / n_fr
    else:
        assert 0.9467 < n_or / n_fr < 0.9505, n_or / n_fr


# ---------------------------------------------------------------------------------------------------
# 3-D Stacked3DSpec
# ---------------------------------------------------------------------------------------------------
EPS_OX_3D = 20.0
K3D = dict(n_bg_m3=N_BG, eps_semi=EPS_S, eps_oxide=EPS_OX_3D)     # the film/oxide of the 3-D spec (no defaults)


def test_stacked3d_kane_band_matches_the_oracle_on_its_column():
    """Stacked3DSpec(bands={'semi': BAND}) (12 nm film, 8 nm oxide eps 20, full gate, body contact under the
    film; tensor mesh): the laterally uniform solution at +2 V equals the oracle on the mesh's z nodes (Dirichlet
    body and gate) to 2e-5 in n (MEASURED 7.0e-11; every film node is degenerate at +2 V, the Sommerfeld
    branch); the spec reports the band's DOS mass."""
    spec = Stacked3DSpec(**K3D, bands={"semi": BAND})
    assert spec.dos_mass_kg == BAND.dos_mass_kg(N_BG)
    b = Devsim3DEquilibrium(spec)
    b.solve(BiasPoint({"gate": 2.0, "body": 0.0}, "k3"))
    g = lambda r, nm: np.array(ds.get_node_model_values(device=b.device, region=r, name=nm))
    zs, n, zo = g("semi", "z"), g("semi", "Electrons"), g("oxide", "z")
    b.teardown()
    z = np.unique(np.round(np.r_[zs, zo], 18))
    t_semi = zs.max()
    cells_semi = 0.5 * (z[1:] + z[:-1]) < t_semi
    psi, sm = oracle_1d(z, np.where(cells_semi, EPS_S, EPS_OX_3D), cells_semi, OracleClosure(kane=True), 0.0, 2.0)
    n_or = OracleClosure(kane=True)(psi[sm])[0]
    zk = np.round(zs, 18)
    for zz, nn in zip(z[sm], n_or):
        sel = zk == zz
        assert np.all(np.abs(n[sel] / nn - 1.0) < 2e-5), (zz, n[sel][:3], nn)


def test_stacked3d_bands_validation_and_from_design():
    import _wiring_fixture as W
    with pytest.raises(NotImplementedError, match="EQUILIBRIUM closure only"):
        Stacked3DSpec(**K3D, bands={"semi": BAND}, physics="drift_diffusion")
    with pytest.raises(ValueError, match="not a semiconductor region"):
        Stacked3DSpec(**K3D, dos_mass_kg=BAND.dos_mass_kg(N_BG), bands={"oxide": BAND})
    with pytest.raises(TypeError, match="must be a materials.KaneBand"):
        Stacked3DSpec(**K3D, bands={"semi": 0.3})
    with pytest.raises(ValueError, match="disagrees with its KaneBand"):
        Stacked3DSpec(**K3D, bands={"semi": BAND}, dos_mass_kg=0.3 * M_E)
    with pytest.raises(TypeError, match="dos_mass_kg"):
        Stacked3DSpec(**K3D)                            # no band on 'semi': the DOS mass is required
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # no density-dependent-DOS-mass advisory with a band
        spec = Stacked3DSpec.from_design(W.film_design(mats=_kane_materials(BAND)))
    assert spec.bands == {"semi": BAND}
    assert spec.dos_mass_kg == BAND.dos_mass_kg(N_BG)
    assert spec.band_alignment()["semi"][0] == pytest.approx(-BAND.eta_of_density(N_BG) * VT, abs=1e-15)
