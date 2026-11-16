"""v0.16 carrier wiring in the 3-D builder (devsim_3d): the DOS-mass-at-n_bg guard at
Stacked3DSpec.from_design's two call sites, and the equilibrium charge primitives -- gate work function, fixed
sheet charge, dielectric bulk charge and equilibrium-film interface traps -- on a generic gated
capacitor (the Stacked3DSpec geometry defaults: a 12 nm equilibrium semiconductor over a body contact, one 8 nm
oxide, a full gate; tensor Kuhn mesh; the film and oxide parameters of tests/_wiring_fixture.py: n_bg 3e26 m^-3,
eps_s 9.0, DOS mass 0.30 m_e, oxide eps 20). The oracles read the oxide and semiconductor parameters from the spec.

ORACLES (as in tests/test_layered_charges.py, closed forms re-derived here): the oxide holds no charge of its
own, so its potential is linear and C_ox = eps_ox eps0 / t_ox exactly; a sheet Q at the semiconductor/oxide
face is compensated by a gate shift of -Q/C_ox, a uniform rho in the oxide by -rho t_ox^2/(2 eps_ox eps0); a
gate work function phi_m is the bias shift -(phi_m - W_s)/q with W_s = chi + q Phi_c0 of the body; interface
traps add C_it = q D_it (f(e_lo) - f(e_hi)) to the face, so d psi_s/dV_g = C_ox/(C_ox + C_s + C_it) with C_s
from the trap-free device on the same mesh.

On the library before this change none of this exists in 3-D: Stacked3DSpec has no
gate_work_function_eV / fixed_charge / bulk_charge (TypeError), from_design silently dropped an
Electrode.work_function_eV and read the DOS mass without the guard (no warning), and FilmInterfaceTraps on an
equilibrium spec were refused (interface_traps needed physics='bipolar_dd').
"""
import math
import warnings

import numpy as np
import pytest

ds = pytest.importorskip("devsim")
from dynameta.carriers.devsim_env import import_devsim     # noqa: E402

ds = import_devsim()

from dynameta.carriers import physics_equilibrium as PE                 # noqa: E402
from dynameta.carriers import physics_traps as PT                       # noqa: E402
from dynameta.carriers.devsim_3d import Devsim3DEquilibrium, Stacked3DSpec   # noqa: E402
from dynameta.constants import EPS0, M_E, Q_E, V_T                      # noqa: E402
from dynameta.sweep import BiasPoint                                     # noqa: E402
from dynameta.warning_types import AdvisoryWarning                       # noqa: E402

import _wiring_fixture as W                                              # noqa: E402

FILM3D = dict(n_bg_m3=W.N_BG, eps_semi=W.EPS_S, eps_oxide=W.TOP_STACK[0][1], dos_mass_kg=W.M_DOS_REL * M_E)


def _spec(**kw):
    """The generic gated capacitor (FILM3D on the Stacked3DSpec geometry defaults) with `kw` on top."""
    return Stacked3DSpec(**FILM3D, **kw)


_DEFAULT = _spec()
C_OX = _DEFAULT.eps_oxide * EPS0 / _DEFAULT.oxide_thk_m


@pytest.fixture(autouse=True)
def _clean():
    yield
    from dynameta.carriers import eq_registry as _R
    for dv in list(ds.get_device_list()):
        _R.clear(dv)
        ds.delete_device(device=dv)
    for m in list(ds.get_mesh_list()):
        ds.delete_mesh(mesh=m)


def _run(spec, vg):
    """Solve, read the 'semi' node arrays, tear down (DEVSIM solves every live device jointly)."""
    b = Devsim3DEquilibrium(spec)
    b.solve(BiasPoint({"gate": vg, "body": 0.0}, "b"))
    out = {nm: np.array(ds.get_node_model_values(device=b.device, region="semi", name=nm))
           for nm in ("z", "Electrons", "Potential")}
    out["oxide_potential"] = np.array(ds.get_node_model_values(device=b.device, region="oxide", name="Potential"))
    out["offset"] = b._gate_offset_V
    b.teardown()
    return out


def _psi_s(o):
    face = np.abs(o["z"] - o["z"].max()) < 1e-15
    # laterally uniform face: every column solves the same equations, so the face potentials differ only by the
    # rounding of the Newton updates -- a few ulp of the largest potential (bar 64 ulp of max |V|, ~7e-15 V at
    # 0.5 V; any lateral structure is orders above it)
    assert np.ptp(o["Potential"][face]) <= 64.0 * np.finfo(float).eps * max(np.max(np.abs(o["Potential"])), 1.0)
    return float(np.mean(o["Potential"][face]))


# ------------------------------------------------------------------------------------------------
# the DOS-mass-at-n_bg guard in from_design
# ------------------------------------------------------------------------------------------------
def test_from_design_dos_mass_guard():
    """from_design reads each semiconductor's DOS mass through TransportModel.dos_mass_at_bg: one
    AdvisoryWarning naming the layer for a density-dependent callable, the value at n_bg either way; a
    constant mass is silent and gives the same float as before."""
    def varying(n):
        return W.M_DOS_REL * M_E * (1.0 + 0.1 * np.log(np.asarray(n, float) / W.N_BG))
    d = W.film_design(mats=W.materials(dos_mass_fn=varying))
    with pytest.warns(AdvisoryWarning, match="Stacked3DSpec.from_design layer 'film'") as rec:
        spec = Stacked3DSpec.from_design(d)
    assert sum("density-dependent" in str(w.message) for w in rec) == 1
    assert spec.dos_mass_kg == float(varying(W.N_BG))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = Stacked3DSpec.from_design(W.film_design())
    assert spec.dos_mass_kg == W.M_DOS_REL * M_E


# ------------------------------------------------------------------------------------------------
# the equilibrium charge primitives in 3-D
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", [2.0e-3, -2.0e-3])
def test_sheet_charge_shifts_the_gate_by_minus_q_over_cox(q):
    """MEASURED 4.4e-16 relative in n, 2.8e-17 V (bar 1e-9 / 1e-12 V: the 3-D equilibrium solve stops at a
    relative update of 1e-5, after which Newton is quadratic)."""
    a = _run(_spec(), 1.0)
    b = _run(_spec(fixed_charge={("semi", "oxide"): q}), 1.0 - q / C_OX)
    assert np.max(np.abs(b["Electrons"] / a["Electrons"] - 1.0)) < 1e-9
    assert np.max(np.abs(b["Potential"] - a["Potential"])) < 1e-12


def test_bulk_oxide_charge_shifts_the_gate_by_the_centroid_formula():
    """rho = 4e5 C/m^3 in the default 8 nm oxide: gate shift -rho t^2/(2 eps eps0) = -0.0723 V (MEASURED 4.4e-16)."""
    rho = 4.0e5
    dv = -rho * _DEFAULT.oxide_thk_m ** 2 / (2.0 * _DEFAULT.eps_oxide * EPS0)
    a = _run(_spec(), 1.0)
    b = _run(_spec(bulk_charge={"oxide": rho}), 1.0 + dv)
    assert np.max(np.abs(b["Electrons"] / a["Electrons"] - 1.0)) < 1e-9
    assert np.max(np.abs(b["Potential"] - a["Potential"])) < 1e-12


@pytest.mark.devsim_isolated("bitwise comparison of two DEVSIM solves")
def test_gate_work_function_is_a_bias_shift_bit_for_bit():
    """gate_work_function_eV = 4.7 eV at V_g = 0.5 V: node arrays bit-identical to the spec without it at
    V_g + offset, offset = -(phi_m - chi - Phi_c0) with Phi_c0 the body's (MEASURED offset -0.7442 V)."""
    phi_m, chi, vg = 4.7, W.CHI_EV, 0.5
    w = _run(_spec(chi_semi_eV=chi, gate_work_function_eV=phi_m), vg)
    phi_c0 = _spec().band_alignment()["semi"][0]
    assert w["offset"] == -(phi_m - (chi + phi_c0))
    w0 = _run(_spec(), vg + w["offset"])
    assert w0["offset"] is None
    for nm in ("Electrons", "Potential", "oxide_potential"):
        assert np.array_equal(w[nm], w0[nm]), nm


def test_work_function_and_charge_inputs_through_from_design():
    """from_design maps the GATE's work function (and the body chi) and layer-keyed charges onto the spec;
    a work function on another electrode, an unmeshed layer, a drift-diffusion spec with a work function and
    a missing chi are refused."""
    d = W.film_design(wf_bot=4.6)                                # from_design's gate here is 'bot' (below)
    spec = Stacked3DSpec.from_design(d, fixed_charge={("film", "lo_oxb"): 1e-3},
                                     bulk_charge={"lo_oxa": 2e5})
    assert spec.gate_name == "bot" and spec.gate_work_function_eV == 4.6 and spec.chi_semi_eV == W.CHI_EV
    assert spec.fixed_charge == {("semi", "oxide"): 1e-3} and spec.bulk_charge == {"diel1": 2e5}
    with pytest.raises(ValueError, match="only the gate"):
        Stacked3DSpec.from_design(W.film_design(wf_top=4.6))
    with pytest.raises(ValueError, match="not meshed"):
        Stacked3DSpec.from_design(W.film_design(), fixed_charge={("film", "up_oxb"): 1e-3})
    with pytest.raises(NotImplementedError, match="LayeredDevsimBuilder"):
        _spec(physics="drift_diffusion", chi_semi_eV=4.5, gate_work_function_eV=4.6)
    with pytest.raises(ValueError, match="chi_semi_eV"):
        _spec(gate_work_function_eV=4.6)
    with pytest.raises(ValueError, match="matches no interface"):
        _run(_spec(fixed_charge={("semi", "nope"): 1e-3}), 0.0)


def _traps():
    nc = PE.effective_dos_m3(_DEFAULT.dos_mass_kg)
    e0 = PE.invert_F12(_DEFAULT.n_bg_m3 / nc) * V_T              # E_F - E_c of the neutral body
    return PT.FilmInterfaceTraps(dit_eV_m2=5e16, e_lo_eV=e0 - 0.6, e_hi_eV=e0 + 0.6, neutral_level_eV=e0)


def test_film_traps_on_an_equilibrium_spec_gate_efficiency():
    """FilmInterfaceTraps on the equilibrium capacitor: d psi_s/dV_g at flat band = C_ox/(C_ox + C_s + C_it)
    (MEASURED 1.3e-12 at dV = 1e-4 V; 0.1705 -> 0.1606 with the traps). On drift-diffusion they are
    refused."""
    dv = 1e-4

    def eff(spec):
        return (_psi_s(_run(spec, dv)) - _psi_s(_run(spec, -dv))) / (2.0 * dv)
    e0 = eff(_spec())
    c_s = C_OX * (1.0 / e0 - 1.0)
    c_it = Q_E * 5e16 * (2.0 / (1.0 + math.exp(-0.6 / V_T)) - 1.0)
    got = eff(_spec(interface_traps=_traps()))
    assert abs(got / (C_OX / (C_OX + c_s + c_it)) - 1.0) < 1e-8
    assert got < 0.97 * e0
    with pytest.raises(ValueError, match="physics='equilibrium'"):
        _spec(physics="drift_diffusion", interface_traps=_traps())
