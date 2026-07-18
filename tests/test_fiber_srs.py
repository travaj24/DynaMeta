"""SRS Stokes-channel gates (RamanStokes coupled into the steady-state solver).

Independent oracles: on a PASSIVE fiber (negligible dopant, zero background loss) the coupled
system collapses to the textbook two-wave SRS problem with closed forms -- (i) undepleted
distributed-seed growth P_S(L) = q_seed (exp(k_r P0 L) - 1), and (ii) Manley-Rowe photon-flux
conservation P_sig/nu_sig + P_S/nu_S = const once seeding is negligible. The active-amplifier
gate then checks the physical ceiling: above threshold the Stokes drains the amplified signal."""

import numpy as np
import pytest

from dynameta.constants import C_LIGHT, H_PLANCK, KB
from dynameta.optics.fiber_amp.spectroscopy import erbium, ytterbium
from dynameta.optics.fiber_amp.steady_state import (FiberAmplifier, Pump, RamanStokes, Signal)
from dynameta.optics.fiber_amp.waveguide import FiberSpec

LAM_S = 1.06e-6


def _passive_fiber(L):
    # essentially undoped (n_t must be > 0), lossless: pure two-wave SRS
    return FiberSpec(core_radius_m=3.0e-6, na=0.14, n_t_m3=1.0, length_m=L)


def _q_seed(rs, lam_sig):
    nu_st = C_LIGHT / lam_sig - rs.shift_hz
    n_th = 1.0 / np.expm1(H_PLANCK * rs.shift_hz / (KB * rs.T_K))
    return H_PLANCK * nu_st * rs.dnu_eff_hz * (n_th + 1.0)


def test_undepleted_distributed_seed_growth_matches_closed_form():
    L = 30.0
    fib = _passive_fiber(L)
    a_eff = 2.0e-11                       # fixed so the closed form is exact
    g_r = 1.0e-13
    P0 = 6.0
    G = g_r / a_eff * P0 * L              # exponent 0.9: cleanly undepleted
    rs = RamanStokes(g_r_m_w=g_r, a_eff_m2=a_eff)
    amp = FiberAmplifier(erbium(), fib, [], [Signal(P0, LAM_S)], ase=None, raman=rs)
    res = amp.solve(n_nodes=201)
    assert res.meta["converged"]
    i_st = res.kind.index("stokes")
    q = _q_seed(rs, LAM_S)
    expect = q * (np.exp(G) - 1.0)
    got = float(res.power_W[i_st, -1])
    assert abs(got / expect - 1.0) < 0.02, (got, expect)
    # signal essentially undepleted at this exponent
    i_sig = res.kind.index("signal")
    assert res.power_W[i_sig, -1] > 0.999 * P0


def test_depleted_regime_conserves_photon_flux():
    L = 60.0
    fib = _passive_fiber(L)
    a_eff, g_r, P0 = 2.0e-12, 1.0e-13, 8.0        # exponent k_r P0 L ~ 23: deep depletion
    rs = RamanStokes(g_r_m_w=g_r, a_eff_m2=a_eff)
    amp = FiberAmplifier(erbium(), fib, [], [Signal(P0, LAM_S)], ase=None, raman=rs)
    res = amp.solve(n_nodes=301)
    i_sig, i_st = res.kind.index("signal"), res.kind.index("stokes")
    P_sig, P_st = res.power_W[i_sig], res.power_W[i_st]
    assert P_st[-1] > 0.2 * P0                       # substantial transfer happened
    assert P_sig[-1] < 0.8 * P0                      # and the signal is depleted
    nu_sig = C_LIGHT / LAM_S
    nu_st = nu_sig - rs.shift_hz
    flux = P_sig / nu_sig + P_st / nu_st             # Manley-Rowe photon flux (per h)
    assert float(np.max(np.abs(flux / flux[0] - 1.0))) < 2e-3


def test_zero_coupling_matches_baseline_solve():
    fib = FiberSpec(core_radius_m=3.0e-6, na=0.14, n_t_m3=8.0e24, length_m=8.0)
    ion = erbium()
    pumps = [Pump(0.25, 0.98e-6)]
    sigs = [Signal(1.0e-3, 1.55e-6)]
    base = FiberAmplifier(ion, fib, pumps, sigs).solve(n_nodes=101)
    rs = RamanStokes(g_r_m_w=0.0, a_eff_m2=1e-11, dnu_eff_hz=0.0)
    with_r = FiberAmplifier(ion, fib, pumps, sigs, raman=rs).solve(n_nodes=101)
    i_sig = with_r.kind.index("signal")
    assert abs(float(with_r.signal_gain_dB[0]) - float(base.signal_gain_dB[0])) < 1e-9
    assert float(with_r.power_W[with_r.kind.index("stokes"), -1]) == 0.0
    assert np.allclose(with_r.power_W[i_sig], base.power_W[base.kind.index("signal")],
                       rtol=1e-9, atol=0.0)


def test_active_amplifier_srs_ceiling():
    # a high-power Yb amplifier whose amplified signal crosses the SRS threshold in-fiber:
    # the Stokes must drain the signal relative to the raman=None solve
    fib = FiberSpec(core_radius_m=5.0e-6, na=0.07, n_t_m3=6.0e25, length_m=25.0,
                    clad_radius_m=62.5e-6)
    ion = ytterbium()
    pumps = [Pump(250.0, 0.976e-6, cladding=True)]
    sigs = [Signal(2.0, LAM_S)]
    # explicit effective area: a tighter-than-LP01 Raman overlap pushes this device across
    # threshold within 25 m (the default pi w^2 ~ 1.2e-10 m^2 leaves G_R ~ 1 -- correctly
    # sub-threshold, as the passive gates verify the coupling itself)
    rs = RamanStokes(a_eff_m2=1.5e-11)
    base = FiberAmplifier(ion, fib, pumps, sigs).solve(n_nodes=201)
    amp = FiberAmplifier(ion, fib, pumps, sigs, raman=rs).solve(n_nodes=201)
    i_sig = amp.kind.index("signal")
    i_st = amp.kind.index("stokes")
    P_out_base = float(base.power_W[base.kind.index("signal"), -1])
    P_out = float(amp.power_W[i_sig, -1])
    P_stokes = float(amp.power_W[i_st, -1])
    assert amp.meta["converged"] and base.meta["converged"]
    assert P_stokes > 0.02 * P_out_base              # the ceiling is active
    assert P_out < 0.98 * P_out_base                 # signal visibly drained


def test_transient_refuses_raman():
    from dynameta.optics.fiber_amp.dynamics import simulate_transient
    fib = FiberSpec(core_radius_m=3.0e-6, na=0.14, n_t_m3=8.0e24, length_m=5.0)
    amp = FiberAmplifier(erbium(), fib, [Pump(0.1, 0.98e-6)], [Signal(1e-3, 1.55e-6)],
                        raman=RamanStokes())
    with pytest.raises(NotImplementedError):
        simulate_transient(amp, np.linspace(0.0, 1e-3, 5))
