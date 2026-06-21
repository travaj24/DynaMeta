"""QD-SOA coherent MAXWELL-BLOCH gain -- the sub-T2 COHERENT polarization dynamics the rate-equation
marcher (which adiabatically eliminates the polarization) cannot capture: Rabi flopping, photon echo,
pulse-area / self-induced-transparency, and the coherent->incoherent crossover as the dephasing time T2
shrinks (rate equations are the T2->0 / weak-field limit). MaxwellBlochEnsemble integrates a real 2-level
Bloch vector (u, v, w) per inhomogeneous QD group. (Research-grade gap from the 2026-06-20 physics-gap
audit -- the marcher's polarization is adiabatically eliminated.)

GATE A (linear gain == the rate-equation spectrum): the weak-field steady-state gain SHAPE
        linear_gain_shape(nu) equals QDGainModel.material_gain_per_m(rho_eq, nu) up to a constant -- the
        coherent model's weak-field / fast-dephasing limit IS the rate-equation gain (the reduction).
GATE B (coherent -> incoherent crossover): a strong resonant field makes the inversion RABI-FLOP when
        T2 >> the Rabi period (the inversion dives to ~ -w0) but only monotonically SATURATE (no flop)
        when T2 << the Rabi period -- the dephasing washes the coherence into the rate-equation limit.
GATE C (Rabi flopping / pulse area): a resonant pulse of area theta = integral Omega dt flips a single
        group's inversion as w = w0 cos(theta) -- a pi pulse inverts (w -> -w0), a 2 pi pulse returns it
        (w -> +w0). This is purely coherent (no rate-equation analogue).
GATE D (photon echo): an inhomogeneously-broadened ensemble dephases (free-induction decay) after a
        pi/2 pulse, then a pi pulse at t = tau REPHASES it into a macroscopic-polarization ECHO at
        t = 2 tau -- the hallmark coherent transient the rate-equation gain cannot produce.
GATE E (pulse-area energy / self-induced transparency): on resonance a 2 pi pulse returns the inversion
        to its start (the medium gives back the energy it took -- |w_end - w0| ~ 0) while a pi pulse
        leaves it fully flipped (|w_end - w0| = 2 w0, maximal energy exchange); the 2 pi / pi ratio << 1.

Run: python -m validation.qd_soa_maxwell_bloch
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa import MaxwellBlochEnsemble, QDGainModel, QDGainParams


def _square_pulse(nt, dt, i0, width_n, area):
    """A square Rabi pulse of total area `area` [rad] over width_n samples starting at index i0."""
    Om = np.zeros(nt)
    Om[i0:i0 + width_n] = area / (width_n * dt)
    return Om


def main():
    print("[mb] === QD-SOA coherent Maxwell-Bloch (Rabi, echo, pulse-area) ===", flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=31).with_detailed_balance_taus())
    nu0 = m.p.nu0_Hz
    fwhm_inh = float(getattr(m.p, "fwhm_inhom_Hz", 5e12))
    drive = 120e-3

    # ---- GATE A: the DYNAMICAL integrator's weak-field steady-state gain == the rate-eq spectrum ----
    # drive a WEAK CW field through evolve() at a set of probe frequencies and read the steady macroscopic
    # in-quadrature coherence Im(P) = sum_j w_j v_j: it reproduces the Lorentzian-weighted inversion (==
    # QDGainModel.material_gain_per_m), so the rate-equation gain is the weak-field / fast-dephasing limit
    # of the dynamical Bloch model. This exercises the ODE INTEGRATION (a wrong sign / Rabi coupling would
    # fail), not just a static closed-form identity.
    y = m.steady_state(drive)
    T2 = 1.0 / (np.pi * float(m.p.fwhm_hom_Hz))
    nu_probe = nu0 + np.linspace(-1.0, 1.0, 9) * fwhm_inh
    nt_a, dt_a = 8000, 0.01 * T2                            # nt*dt = 80 T2 (steady); dt << 1/detuning
    Om_w = 1.0e-3 / T2                                       # weak (linear-response) CW Rabi field
    g_dyn = np.empty(nu_probe.size)
    for i, nup in enumerate(nu_probe):
        mbp = MaxwellBlochEnsemble.from_model(m, drive, nu_s_Hz=float(nup))
        g_dyn[i] = float(np.imag(mbp.evolve(np.full(nt_a, Om_w), dt_a)["P"][-1]))   # sum_j w_j v_j(ss)
    g_model = m.material_gain_per_m(m.rho_GS(y), nu_probe)
    sh_dyn = g_dyn / np.max(np.abs(g_dyn))
    sh_md = g_model / np.max(np.abs(g_model))
    relA = float(np.max(np.abs(sh_dyn - sh_md)))
    g_a = bool(relA < 2e-2)
    ok = ok and g_a
    print("[mb] GATE A: integrated weak-CW steady gain (via evolve) == rate-eq material_gain_per_m "
          "(rel {:.1e}) -> {}".format(relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE C: Rabi flopping w(theta) = w0 cos(theta) (single resonant group) ----
    one = MaxwellBlochEnsemble(np.array([nu0]), np.array([1.0]), np.array([1.0]), nu0, np.inf, np.inf)
    tp, nt = 1.0e-12, 4000
    dt = tp / nt
    rabi = {}
    for area in (np.pi, 2.0 * np.pi):
        w_end = float(one.evolve(np.full(nt, area / tp), dt, w0=1.0)["w_mean"][-1])
        rabi[area] = w_end
    errC = max(abs(rabi[np.pi] - np.cos(np.pi)), abs(rabi[2.0 * np.pi] - np.cos(2.0 * np.pi)))
    g_c = bool(errC < 5e-3)
    ok = ok and g_c
    print("[mb] GATE C: Rabi pi -> {:.4f} (cos=-1), 2pi -> {:.4f} (cos=+1), err {:.1e} -> {}".format(
        rabi[np.pi], rabi[2.0 * np.pi], errC, "PASS" if g_c else "FAIL"), flush=True)

    # ---- GATE B: coherent -> incoherent crossover (Rabi flop vs dephasing-damped) ----
    # strong resonant field for several Rabi periods; large T2 -> deep flop (min w ~ -1), small T2 ->
    # the coherence dephases and the inversion only saturates (min w stays high).
    Om0 = 6.0 * np.pi / tp                                   # ~3 Rabi periods over tp
    T2_rabi = 2.0 * np.pi / Om0                              # one Rabi period
    minw = {}
    for label, T2 in (("coh", np.inf), ("inc", 0.08 * T2_rabi)):
        mb1 = MaxwellBlochEnsemble(np.array([nu0]), np.array([1.0]), np.array([1.0]), nu0,
                                   np.inf, T2)
        minw[label] = float(np.min(mb1.evolve(np.full(nt, Om0), dt, w0=1.0)["w_mean"]))
    g_b = bool(minw["coh"] < -0.8 and minw["inc"] > -0.3 and minw["coh"] < minw["inc"] - 0.5)
    ok = ok and g_b
    print("[mb] GATE B: coherent min-w {:.3f} (deep flop) vs dephased {:.3f} (no flop) -> {}".format(
        minw["coh"], minw["inc"], "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE D: photon echo (inhomogeneous rephasing at 2 tau) ----
    sig = 0.5 * fwhm_inh                                     # inhomogeneous spread (fast FID)
    nu_j = nu0 + sig * np.linspace(-3.0, 3.0, 121)
    wj = np.exp(-0.5 * ((nu_j - nu0) / sig) ** 2)
    wj = wj / wj.sum()
    mbe = MaxwellBlochEnsemble(nu_j, wj, -np.ones_like(nu_j), nu0, 1.0e-11, np.inf)  # T2=10 ps, ground
    dt_e, nt_e = 1.0e-15, 1400
    tau_n = 400                                             # tau = 0.4 ps
    wpi2 = 20                                               # pulse widths
    Om_e = (_square_pulse(nt_e, dt_e, 0, wpi2, 0.5 * np.pi)          # pi/2 at t=0
            + _square_pulse(nt_e, dt_e, tau_n, wpi2, np.pi))        # pi at t=tau
    P = np.abs(mbe.evolve(Om_e, dt_e)["P"])
    i_echo = 2 * tau_n
    echo_peak = float(np.max(P[i_echo - 40:i_echo + 40]))
    baseline = float(np.median(P[tau_n + 3 * wpi2:i_echo - 60]))     # dephased FID floor between pulses
    g_d = bool(echo_peak > 5.0 * baseline)
    ok = ok and g_d
    print("[mb] GATE D: photon echo peak {:.2e} >> dephased baseline {:.2e} (x{:.1f}) -> {}".format(
        echo_peak, baseline, echo_peak / baseline, "PASS" if g_d else "FAIL"), flush=True)

    # ---- GATE E: pulse-area energy / self-induced transparency ----
    grnd = MaxwellBlochEnsemble(np.array([nu0]), np.array([1.0]), np.array([-1.0]), nu0, np.inf, np.inf)
    w_2pi = float(grnd.evolve(np.full(nt, 2.0 * np.pi / tp), dt, w0=-1.0)["w_mean"][-1])
    w_pi = float(grnd.evolve(np.full(nt, np.pi / tp), dt, w0=-1.0)["w_mean"][-1])
    ret_2pi = abs(w_2pi - (-1.0))                            # 2pi returns -> ~0 (transparent)
    ret_pi = abs(w_pi - (-1.0))                              # pi fully inverts -> ~2 (max absorbed)
    g_e = bool(ret_2pi < 5e-3 and ret_pi > 1.9 and ret_2pi < 0.05 * ret_pi)
    ok = ok and g_e
    print("[mb] GATE E: 2pi returns (|dw|={:.1e}, transparent) vs pi inverts (|dw|={:.3f}) -> {}".format(
        ret_2pi, ret_pi, "PASS" if g_e else "FAIL"), flush=True)

    print("[mb] *** QD-SOA COHERENT MAXWELL-BLOCH: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
