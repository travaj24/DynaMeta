"""Phase-4 reconfigurable-modulator validation (roadmap Phase 4 oracle): the three reconfigurable
mechanism families built on the generalized spine, each checked against an INDEPENDENT analytic
oracle (pure numpy -- no FEM, so this runs fast and solver-free):

  (1) PCM (phase-change, PCMModel): the Bruggeman effective-medium mix reduces EXACTLY to the
      amorphous/crystalline end states at f=0/1, is monotonic in the crystalline fraction, stays
      PASSIVE (Im(eps) >= 0), and lies between the Wiener bounds (series <= Bruggeman <= parallel).
  (2) LIQUID CRYSTAL (the comprehensive director solver -> LiquidCrystalModel): the two-constant
      (K11/K33) static BVP (director_profile_bvp) + the convention bridge (director_to_extra_fields)
      modulate the optical eps from ~n_o (planar, below V_th) toward n_e (tilted, above V_th); the
      Erickson-Leslie dynamics (LCDynamics) switch up and relax back with finite rise/decay; the
      uniaxial tensor's eigenvalues stay the rotation-invariant {n_o^2, n_o^2, n_e^2} for ANY tilt;
      and n_e = n_o reduces to isotropic. (The deeper director-physics oracles live in
      validation/lc_two_constant_bvp.py, lc_director_dynamics.py and lc_cyl_flexo_optics.py.)
  (3) GRAPHENE (graphene_sigma + sheet_rt): the interband conductivity is the universal
      sigma0 = e^2/(4 hbar) well above threshold and is PAULI-BLOCKED (Re(sigma) -> ~0) once
      2|E_F| > hbar*omega -- the gate-tunable absorption modulator; the conductive-sheet reflection
      conserves energy (R+T+A=1) and reduces to the bare Fresnel as sigma -> 0.

Run: python -m validation.reconfigurable_modulators
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.constants import HBAR, C_LIGHT, Q_E as Q
from dynameta.core.effects import PCMModel, LiquidCrystalModel
from dynameta.core.graphene import graphene_sigma, sheet_rt, SIGMA0, Z0
from dynameta.carriers.lc_director import (
    freedericksz_threshold_V, director_profile_bvp, director_to_extra_fields)
from dynameta.carriers.lc_dynamics import LCDynamics

LAM = 1.55e-6


def part1_pcm():
    """Bruggeman PCM: endpoint reduction, monotonicity, passivity, Wiener-bounds containment."""
    ea, ec = complex(16.0, 0.5), complex(36.0, 6.0)            # amorphous / crystalline (GST-like)
    pcm = PCMModel(eps_amorphous=ea, eps_crystalline=ec)
    e0 = pcm.eps({"crystalline_fraction": 0.0}, LAM)
    e1 = pcm.eps({"crystalline_fraction": 1.0}, LAM)
    fs = np.linspace(0.0, 1.0, 11)
    es = np.array([pcm.eps({"crystalline_fraction": f}, LAM) for f in fs])
    endpoints = abs(e0 - ea) < 1e-12 and abs(e1 - ec) < 1e-12
    passive = bool(np.all(es.imag >= -1e-12))
    mono = bool(np.all(np.diff(np.sqrt(es).real) > 0))         # Re(n) rises with crystallinity
    # Wiener bounds (on Re): series (harmonic) <= Bruggeman <= parallel (arithmetic)
    within = True
    for f, e in zip(fs, es):
        par = f * ec + (1 - f) * ea
        ser = 1.0 / (f / ec + (1 - f) / ea)
        if not (ser.real - 1e-9 <= e.real <= par.real + 1e-9):
            within = False
    ok = endpoints and passive and mono and within
    print("[r] (1) PCM: endpoints={} passive={} monotonic={} Wiener-bounded={} (n {:.3f}->{:.3f})".format(
        endpoints, passive, mono, within, np.sqrt(e0).real, np.sqrt(e1).real), flush=True)
    return ok


def part2_liquid_crystal():
    """LC: the COMPREHENSIVE director solver drives the optics. The two-constant (K11/K33) static BVP
    (director_profile_bvp) + the convention bridge (director_to_extra_fields) modulate the optical
    LiquidCrystalModel from ~n_o (planar, below V_th) toward n_e (tilted, above V_th); the Erickson-Leslie
    dynamics (LCDynamics) switch up and relax back with finite rise/decay; the uniaxial tensor's
    eigenvalues stay the rotation-invariant {n_o^2, n_o^2, n_e^2}; and n_e=n_o reduces to isotropic. (The
    legacy 1-constant exact-planar Freedericksz bifurcation sqrt-law -- director_profile -- is the
    retained reference covered in tests/test_lc_director.py + validation/lc_two_constant_bvp.py.)"""
    K11, K33, eps_para, eps_perp, d = 17e-12, 18e-12, 18.7, 4.0, 1e-6
    no, ne = 1.53, 1.71
    Vth = freedericksz_threshold_V(K11, eps_para - eps_perp)
    bkw = dict(K11=K11, K33=K33, eps_para=eps_para, eps_perp=eps_perp, d_planar=d,
               theta_b_rad=math.radians(89.9), field_model="uniform", n_o=no, n_e=ne, nz=121)
    below = director_profile_bvp(V_app=0.5 * Vth, **bkw)
    above = director_profile_bvp(V_app=2.0 * Vth, **bkw)
    # voltage -> tilt -> n_eff: ~planar (n_o) below V_th; rises toward n_e once tilted above V_th
    modulation_ok = bool(abs(below.n_eff - no) < 3e-3 and above.n_eff > no + 0.02 and above.n_eff <= ne + 1e-6)
    # the convention bridge feeds the optical LiquidCrystalModel; eigenvalues are invariant for any tilt
    lc = LiquidCrystalModel(n_o=no, n_e=ne)
    eps = np.asarray(lc.eps(director_to_extra_fields(above.theta_field_rad), LAM))
    eig_ok = all(np.allclose(np.sort(np.linalg.eigvalsh(eps[i].real)),
                             np.sort([no ** 2, no ** 2, ne ** 2]), atol=1e-9) for i in range(eps.shape[0]))
    iso_ok = bool(np.allclose(LiquidCrystalModel(1.6, 1.6).eps({"director_angle_rad": 0.7}, LAM),
                              1.6 ** 2 * np.eye(3)))
    # Erickson-Leslie dynamics: an above-threshold pulse switches n_eff up and relaxes back, finite rise/decay
    dyn = LCDynamics(K11=K11, K33=K33, gamma1=0.085, eps_para=eps_para, eps_perp=eps_perp,
                     theta_b_rad=math.radians(89.9), geometry="planar", d_planar=d,
                     field_model="uniform", n_o=no, n_e=ne, nz=81)
    rdyn = dyn.simulate_pulse(V0=2.0 * Vth, Ton=4e-3, T_end=12e-3, n_t=200, waveform="step")
    switch_ok = bool(np.isfinite(rdyn.rise_10_90_s) and rdyn.rise_10_90_s > 0
                     and np.isfinite(rdyn.decay_90_10_s) and rdyn.decay_90_10_s > 0
                     and float(np.nanmax(rdyn.n_eff)) > no + 0.02 and abs(float(rdyn.n_eff[-1]) - no) < 3e-3)
    ok = modulation_ok and eig_ok and iso_ok and switch_ok
    print("[r] (2) LC: V_th={:.4f} V  n_eff {:.4f}->{:.4f} modulation={} eig-invariant={} iso={} "
          "switching(rise={:.3g}s decay={:.3g}s)={}".format(
              Vth, below.n_eff, above.n_eff, modulation_ok, eig_ok, iso_ok,
              rdyn.rise_10_90_s, rdyn.decay_90_10_s, switch_ok), flush=True)
    return ok


def part3_graphene():
    """Graphene sheet: universal sigma0, Pauli blocking, energy conservation, Fresnel reduction."""
    omega = 2.0 * np.pi * C_LIGHT / LAM
    hw_eV = HBAR * omega / Q                                   # ~0.80 eV at 1.55 um
    n1, n2 = 1.0, 1.5
    # universal interband conductivity well below threshold (E_F=0): Re(sigma) ~ sigma0
    s_lo = graphene_sigma(0.0, LAM)
    universal = abs(s_lo.real / SIGMA0 - 1.0) < 0.05
    # Pauli blocking: Re(sigma) collapses once 2 E_F > hbar omega (E_F > 0.40 eV)
    re_ratio = [graphene_sigma(EF * Q, LAM).real / SIGMA0 for EF in (0.0, 0.3, 0.5, 0.7)]
    pauli = re_ratio[0] > 0.9 and re_ratio[-1] < 0.1 and bool(np.all(np.diff(re_ratio) < 0))
    passive = bool(np.all([graphene_sigma(EF * Q, LAM).real > 0 for EF in (0.0, 0.5, 1.0)]))
    # INDEPENDENT energy balance (NOT the tautological A:=1-R-T, audit F1): the flux deficit
    # 1-R-T must equal the Ohmic SHEET DISSIPATION A_poynting = Z0 Re(sigma) |t|^2 / n1 computed
    # separately from sigma and the transmitted amplitude.
    s_on = graphene_sigma(0.0, LAM)
    r, t, R, T, A_on = sheet_rt(n1, n2, s_on)                  # E_F=0 -> interband ON
    A_poynting = Z0 * s_on.real * abs(t) ** 2 / n1
    energy = abs(A_on - A_poynting) < 1e-9 and A_on >= 0.0     # flux deficit == sheet absorption
    # gate-tunable absorption: ON (E_F=0) >> OFF (Pauli-blocked at E_F=0.6 eV)
    A_off = sheet_rt(n1, n2, graphene_sigma(0.6 * Q, LAM))[4]
    tunable = A_on > 5.0 * A_off and A_on > 0.0
    # sigma -> 0 recovers the bare Fresnel
    fresnel = abs(sheet_rt(n1, n2, 0.0)[0] - (n1 - n2) / (n1 + n2)) < 1e-12
    ok = universal and pauli and passive and tunable and energy and fresnel
    print("[r] (3) graphene (hw={:.2f} eV): sigma0-universal={} Pauli-block={} passive={} "
          "A {:.4f}->{:.4f} (tunable={}) energy(Poynting)={} Fresnel={}".format(
              hw_eV, universal, pauli, passive, A_on, A_off, tunable, energy, fresnel), flush=True)
    return ok


def main():
    ok1, ok2, ok3 = part1_pcm(), part2_liquid_crystal(), part3_graphene()
    ok = ok1 and ok2 and ok3
    print("[r] *** RECONFIGURABLE MODULATORS (PCM Bruggeman; LC Freedericksz+uniaxial; graphene "
          "Pauli-blocked sheet): {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
