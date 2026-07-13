"""Phase-3 QCSE / MQW electro-absorption validation (roadmap Phase 3 oracle): a GaAs quantum-well
Stark electro-absorption modulator built from the generalized spine -- a QuantumWell Stark driver
-> ElectroAbsorptionModel (eps as a function of the applied field) -- checked against independent
ANALYTIC oracles (no FEM; the QCSE solver is a 1D BenDaniel-Duke eigenproblem, so this runs fast
and solver-free):

  (1) SOLVER vs ANALYTIC (deep/infinite-barrier well): the electron ground confinement energy
      matches the textbook infinite-square-well E1 = hbar^2 pi^2 / (2 m L^2), and the small-field
      quadratic Stark shift matches the analytic 2nd-order coefficient
          dE1 = - beta q^2 m F^2 L^4 / hbar^2,  beta = (128/pi^6) sum_{n even} n^2/(n^2-1)^5 ~ 2.1944e-3.
  (2) QCSE PHYSICS (physical GaAs well): the interband edge redshifts QUADRATICALLY in F, the
      electron-hole overlap (oscillator strength) DECREASES with F, and there is NO shift at F=0.
  (3) ELECTRO-ABSORPTION (device): at a probe ~2 sigma below the zero-field exciton, the field
      turns ON absorption (d-alpha > 0, Im(eps) rises above the background) with a clear on-state
      maximum at the field whose redshift ~ the probe offset; and at F=0 eps reduces EXACTLY to
      eps_bg (flat-band reduction).

Run: python -m validation.qcse_electroabsorption
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.constants import HBAR, M_E, C_LIGHT, Q_E as Q
from dynameta.carriers.qcse import QuantumWell, INFINITE_WELL_STARK_BETA as BETA
from dynameta.core.effects import ElectroAbsorptionModel

ME = 0.067 * M_E        # GaAs electron effective mass
MHH = 0.34 * M_E        # GaAs heavy-hole effective mass


def _r2(x, y):
    """Coefficient of determination of a straight-line fit y ~ a*x + b."""
    a, b = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (a * x + b)) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0, a


def part1_solver_vs_analytic():
    """Deep-well limit: confinement energy + quadratic Stark coefficient vs analytic infinite well."""
    L = 15e-9
    qw = QuantumWell(well_width_m=L, barrier_e_J=200.0 * Q, barrier_h_J=200.0 * Q,
                     m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q, nz=2001, n_pad=1.5)
    E1 = qw.solve(0.0).E_e1_J
    E1_ana = HBAR ** 2 * np.pi ** 2 / (2.0 * ME * L ** 2)
    Fs = np.array([0.0, 1e6, 2e6, 3e6, 4e6])
    dE = np.array([qw.solve(F).E_e1_J for F in Fs]); dE = dE - dE[0]
    r2, slope = _r2(Fs ** 2, dE)              # dE = -C F^2  -> slope vs F^2
    C_num = -slope
    C_ana = BETA * Q ** 2 * ME * L ** 4 / HBAR ** 2
    e1_ratio, c_ratio = E1 / E1_ana, C_num / C_ana
    print("[q] (1) deep-well: E1/E1_ana={:.4f}  Stark C_num/C_ana={:.4f}  quad-fit R2={:.5f}".format(
        e1_ratio, c_ratio, r2), flush=True)
    ok = (0.97 < e1_ratio < 1.03) and (0.95 < c_ratio < 1.15) and (r2 > 0.999)
    return ok


def _gaas_well():
    # Deep enough barriers (0.30/0.20 eV) that BOTH carriers stay bound (in-well prob > 0.95,
    # redshift n_pad-stable to ~2%) through the on-state fields -- the shallower 0.15 eV hole
    # barrier field-ionizes by ~7e6 V/m (audit QC-2), which the solver now flags with a warning.
    return QuantumWell(well_width_m=10e-9, barrier_e_J=0.30 * Q, barrier_h_J=0.20 * Q,
                       m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q,
                       exciton_binding_J=0.010 * Q, nz=1201, n_pad=2.0)


def part2_qcse_physics():
    """Physical GaAs well: quadratic edge redshift, overlap reduction, F^2-law through the origin."""
    qw = _gaas_well()
    ET0 = qw.transition_energy_J(0.0)
    Fs = np.array([0.0, 1e6, 2e6, 3e6, 4e6, 5e6])
    red = np.array([(ET0 - qw.solve(F).E_transition_J) for F in Fs])    # redshift (J), >=0
    ov = np.array([qw.solve(F).overlap for F in Fs])
    slope, intercept = np.polyfit(Fs ** 2, red, 1)
    r2 = _r2(Fs ** 2, red)[0]
    # Independent zero-field check (NOT the tautological red[0]==0 self-comparison, audit QC-1):
    # the fitted F^2-law must extrapolate THROUGH the origin -- a spurious constant offset in
    # E_T(F) - E_T(0) would surface as a nonzero intercept.
    zero_field_ok = abs(intercept) < 0.05 * red[-1]
    quad = (slope > 0) and (r2 > 0.998) and bool(np.all(red[1:] > 0))
    overlap_drop = bool(np.all(np.diff(ov) < 0) and ov[-1] < ov[0])
    print("[q] (2) GaAs well: redshift@5e6={:.2f} meV  quad-R2={:.5f}  fit-intercept={:.1e} meV  "
          "overlap {:.3f}->{:.3f}".format(red[-1] / Q * 1e3, r2, intercept / Q * 1e3,
                                          ov[0], ov[-1]), flush=True)
    return bool(zero_field_ok and quad and overlap_drop)


def part3_electroabsorption():
    """Device: flat-band reduction (F=0 -> eps_bg) + a clear field-ON absorption on-state."""
    qw = _gaas_well()
    ET0 = qw.transition_energy_J(0.0)
    sigma = 0.006 * Q
    eps_bg = complex(3.6 ** 2, 0.01)
    eam = ElectroAbsorptionModel(qw=qw, eps_bg=eps_bg, alpha0_per_m=1e6, broadening_J=sigma,
                                 e_grid_J=(ET0 - 0.3 * Q, ET0 + 0.3 * Q, 3001))
    lam = 2.0 * np.pi * HBAR * C_LIGHT / (ET0 - 2.0 * sigma)            # probe 2 sigma below edge
    eps0 = eam.eps({"E": np.zeros(3)}, lam)
    # F=0 -> dalpha identically 0 -> eps == eps_bg (a structural reduction-to-known-limit); the
    # 1e-12 tol is ~1000x above the ~1e-15 sqrt-then-square roundoff floor (audit QC-2).
    flat_band = abs(eps0 - eps_bg) < 1e-12
    Fs = [0.0, 3e6, 5e6, 7e6, 9e6]
    da = np.array([eam.delta_alpha_per_m({"E": np.array([0., 0., F])}, lam) for F in Fs])
    im = np.array([eam.eps({"E": np.array([0., 0., F])}, lam).imag for F in Fs])
    kmax = int(np.argmax(da))
    on_state = (da[kmax] > 0.1e6) and (0 < kmax < len(Fs) - 1)         # absorption ON at interior F
    absorb_up = im[kmax] > eps_bg.imag                                # load-bearing: Im(eps) RISES
    passive = bool(np.all(im > 0))                                    # exp(-iwt) absorber-sign guard
    print("[q] (3) EAM: flat-band |eps0-eps_bg|={:.1e}  max d-alpha={:.0f} 1/m @F={:.0e}  "
          "Im(eps) {:.3f}->{:.3f}".format(abs(eps0 - eps_bg), da[kmax], Fs[kmax],
                                          eps_bg.imag, im[kmax]), flush=True)
    return bool(flat_band and on_state and absorb_up and passive)


def _s2d(dE, xb):
    """2-D Sommerfeld enhancement, Shinada-Sugano form: gamma = sqrt(R/dE), R = E_b(2D)/4,
    so the exponent is -pi sqrt(E_b/dE) (audit 7b: the code and this helper both carried a
    doubled exponent; NOTE this helper deliberately mirrors the implementation -- the
    field-independence of the continuum STRENGTH is what part-4 discriminates, not the
    lineshape itself)."""
    return 2.0 / (1.0 + np.exp(-np.pi * np.sqrt(xb / dE))) if dE > 0.0 else 0.0


def part4_continuum_under_field():
    """Elliott band-to-band continuum UNDER FIELD (the previously-dead path): its EA contribution is
    driven by the EDGE REDSHIFT only -- the continuum STRENGTH is field-INDEPENDENT (set by the
    interband matrix element, NOT the 1s-exciton overlap). DISCRIMINATOR: the ratio-scaled continuum a
    prior version carried would give a materially different (overlap-suppressed) value."""
    qw = _gaas_well()
    s0 = qw.solve(0.0)
    ET0 = s0.E_transition_J
    xb = 0.012 * Q
    sigma = 0.006 * Q
    cont = 1.0e6
    eps_bg = complex(3.6 ** 2, 0.05)
    grid = (ET0 - 0.3 * Q, ET0 + xb + 0.4 * Q, 4001)
    kw = dict(qw=qw, eps_bg=eps_bg, alpha0_per_m=1e6, broadening_J=sigma,
              continuum_binding_J=xb, e_grid_J=grid)
    eam = ElectroAbsorptionModel(continuum_alpha0_per_m=cont, **kw)
    eam_noc = ElectroAbsorptionModel(continuum_alpha0_per_m=0.0, **kw)   # continuum OFF (isolator)
    F = 9e6
    sF = qw.solve(F)
    ratioF = sF.overlap / s0.overlap                                    # < 1 under field
    E_ph = ET0 + xb                                                     # at the 0-field continuum onset
    lam = 2.0 * np.pi * HBAR * C_LIGHT / E_ph
    fld = {"E": np.array([0.0, 0.0, F])}
    # with-MINUS-without continuum: the exciton + KK cancel, leaving the PURE continuum EA contribution
    cc = eam.delta_alpha_per_m(fld, lam) - eam_noc.delta_alpha_per_m(fld, lam)
    s2dF = _s2d(E_ph - (sF.E_transition_J + xb), xb)                    # field-on (redshifted onset)
    s2d0 = _s2d(E_ph - (ET0 + xb), xb)                                  # field-off (= 0 at the onset)
    expected_fix = cont * (s2dF - s2d0)                                # FIELD-INDEPENDENT strength
    expected_old = cont * (ratioF * s2dF - 1.0 * s2d0)                 # the ratio-scaled (wrong) model
    match_fix = abs(cc - expected_fix) < 1e-3 * cont                   # continuum strength is field-indep
    discriminates = abs(expected_fix - expected_old) > 0.05 * cont     # old model materially differs
    contributes = cc > 0.05 * cont                                     # the continuum branch is LIVE
    print("[q] (4) continuum under F: cc={:.3e} == field-indep {:.3e} (match {}), ratioF={:.3f}, "
          "old-model would give {:.3e} (discriminates {}) -> {}".format(
              cc, expected_fix, match_fix, ratioF, expected_old, discriminates,
              "PASS" if (match_fix and discriminates and contributes and ratioF < 0.98) else "FAIL"),
          flush=True)
    return bool(match_fix and discriminates and contributes and ratioF < 0.98)


def main():
    ok1 = part1_solver_vs_analytic()
    ok2 = part2_qcse_physics()
    ok3 = part3_electroabsorption()
    ok4 = part4_continuum_under_field()
    ok = ok1 and ok2 and ok3 and ok4
    print("[q] *** QCSE ELECTRO-ABSORPTION (Stark solver == analytic; quadratic redshift; "
          "field-ON absorption; field-indep continuum): {} ***".format("PASS" if ok else "FAIL"),
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
