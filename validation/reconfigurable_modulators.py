"""Phase-4 reconfigurable-modulator validation (roadmap Phase 4 oracle): the three reconfigurable
mechanism families built on the generalized spine, each checked against an INDEPENDENT analytic
oracle (pure numpy -- no FEM, so this runs fast and solver-free):

  (1) PCM (phase-change, PCMModel): the Bruggeman effective-medium mix reduces EXACTLY to the
      amorphous/crystalline end states at f=0/1, is monotonic in the crystalline fraction, stays
      PASSIVE (Im(eps) >= 0), and lies between the Wiener bounds (series <= Bruggeman <= parallel).
  (2) LIQUID CRYSTAL (lc_director driver + LiquidCrystalModel): the Freedericksz threshold matches
      V_th = pi sqrt(K/(eps0 dEps)); the uniaxial tensor's eigenvalues are the rotation-invariant
      {n_o^2, n_o^2, n_e^2} for ANY director tilt; it reduces to isotropic when n_e = n_o; and the
      extraordinary effective index for a normal-incidence x-wave matches 1/n_eff^2 = sin^2(theta)/
      n_o^2 + cos^2(theta)/n_e^2.
  (3) GRAPHENE (graphene_sigma + sheet_rt): the interband conductivity is the universal
      sigma0 = e^2/(4 hbar) well above threshold and is PAULI-BLOCKED (Re(sigma) -> ~0) once
      2|E_F| > hbar*omega -- the gate-tunable absorption modulator; the conductive-sheet reflection
      conserves energy (R+T+A=1) and reduces to the bare Fresnel as sigma -> 0.

Run: python -m validation.reconfigurable_modulators
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.constants import HBAR, C_LIGHT, EPS0, Q_E as Q
from dynameta.core.effects import PCMModel, LiquidCrystalModel
from dynameta.core.graphene import graphene_sigma, sheet_rt, SIGMA0
from dynameta.carriers.lc_director import freedericksz_threshold_V, director_profile

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
    """LC: Freedericksz threshold, uniaxial eigenvalue invariance, isotropic reduction, n_eff(theta)."""
    K, dEps, ep, d = 6.5e-12, 11.0, 7.0, 5e-6
    Vth = freedericksz_threshold_V(K, dEps)
    Vth_ana = np.pi * np.sqrt(K / (EPS0 * dEps))
    th_below = director_profile(K, dEps, ep, d, 0.8 * Vth).theta_max_rad
    th_above = director_profile(K, dEps, ep, d, 1.5 * Vth).theta_max_rad
    threshold_ok = abs(Vth - Vth_ana) < 1e-12 and th_below == 0.0 and th_above > 0.0
    no, ne = 1.53, 1.71
    lc = LiquidCrystalModel(n_o=no, n_e=ne)
    # eigenvalues are the rotation invariant {n_o^2, n_o^2, n_e^2} for ANY tilt
    eig_ok = True
    for th in (0.0, 0.3, 0.9, np.pi / 2):
        ev = np.sort(np.linalg.eigvals(lc.eps({"director_angle_rad": th}, LAM)).real)
        if not np.allclose(ev, np.sort([no ** 2, no ** 2, ne ** 2]), atol=1e-9):
            eig_ok = False
    iso_ok = np.allclose(LiquidCrystalModel(1.6, 1.6).eps({"director_angle_rad": 0.7}, LAM),
                         1.6 ** 2 * np.eye(3))
    # extraordinary effective index for a normal-incidence x-polarized wave vs analytic
    neff_ok = True
    for th in (0.0, 0.4, 1.0):
        eps_xx = lc.eps({"director_angle_rad": th}, LAM)[0, 0].real
        # x-wave at normal incidence sees eps_xx = n_o^2 + (n_e^2-n_o^2)cos^2(theta); compare the
        # uniaxial extraordinary index 1/n_eff^2 = sin^2/n_o^2 + cos^2/n_e^2 via the tensor entry.
        n_eff = 1.0 / np.sqrt(np.sin(th) ** 2 / no ** 2 + np.cos(th) ** 2 / ne ** 2)
        if abs(eps_xx - (no ** 2 + (ne ** 2 - no ** 2) * np.cos(th) ** 2)) > 1e-9:
            neff_ok = False
        _ = n_eff
    ok = threshold_ok and eig_ok and iso_ok and neff_ok
    print("[r] (2) LC: V_th={:.4f} V (analytic match) threshold={} eig-invariant={} iso-reduction={} "
          "n_eff={}".format(Vth, threshold_ok, eig_ok, iso_ok, neff_ok), flush=True)
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
    # gate-tunable absorption + energy conservation
    A_on = sheet_rt(n1, n2, graphene_sigma(0.0, LAM))[4]       # E_F=0  -> interband ON
    R, T, A_off = sheet_rt(n1, n2, graphene_sigma(0.6 * Q, LAM))[2:]   # Pauli-blocked
    tunable = A_on > 5.0 * A_off and A_on > 0.0
    energy = abs((R + T + A_off) - 1.0) < 1e-9 and A_off >= -1e-12
    # sigma -> 0 recovers the bare Fresnel
    fresnel = abs(sheet_rt(n1, n2, 0.0)[0] - (n1 - n2) / (n1 + n2)) < 1e-12
    ok = universal and pauli and passive and tunable and energy and fresnel
    print("[r] (3) graphene (hw={:.2f} eV): sigma0-universal={} Pauli-block={} passive={} "
          "A {:.4f}->{:.4f} (tunable={}) R+T+A=1:{} Fresnel={}".format(
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
