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
    return QuantumWell(well_width_m=10e-9, barrier_e_J=0.25 * Q, barrier_h_J=0.15 * Q,
                       m_e_kg=ME, m_h_kg=MHH, E_g_J=1.42 * Q,
                       exciton_binding_J=0.010 * Q, nz=1201, n_pad=2.0)


def part2_qcse_physics():
    """Physical GaAs well: quadratic edge redshift, overlap reduction, no shift at F=0."""
    qw = _gaas_well()
    ET0 = qw.transition_energy_J(0.0)
    Fs = np.array([0.0, 1e6, 2e6, 3e6, 4e6, 5e6])
    red = np.array([(ET0 - qw.solve(F).E_transition_J) for F in Fs])    # redshift (J), >=0
    ov = np.array([qw.solve(F).overlap for F in Fs])
    r2, slope = _r2(Fs ** 2, red)
    no_shift = abs(red[0]) < 1e-30 and abs(ov[0] - qw.solve(0.0).overlap) < 1e-12
    quad = (slope > 0) and (r2 > 0.998) and np.all(red[1:] > 0)
    overlap_drop = np.all(np.diff(ov) < 0) and (ov[-1] < ov[0])
    print("[q] (2) GaAs well: redshift@5e6={:.2f} meV  quad-R2={:.5f}  overlap {:.3f}->{:.3f}  "
          "no_shift={}".format(red[-1] / Q * 1e3, r2, ov[0], ov[-1], no_shift), flush=True)
    return bool(no_shift and quad and overlap_drop)


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
    flat_band = abs(eps0 - eps_bg) < 1e-9                              # F=0 reduces to background
    Fs = [0.0, 3e6, 5e6, 7e6, 9e6]
    da = np.array([eam.delta_alpha_per_m({"E": np.array([0., 0., F])}, lam) for F in Fs])
    im = np.array([eam.eps({"E": np.array([0., 0., F])}, lam).imag for F in Fs])
    kmax = int(np.argmax(da))
    on_state = (da[kmax] > 0.1e6) and (0 < kmax < len(Fs) - 1)         # interior on-state max
    absorb_up = (im[kmax] > eps_bg.imag) and np.all(im > 0)            # Im(eps)>0 (absorber)
    print("[q] (3) EAM: flat-band |eps0-eps_bg|={:.1e}  max d-alpha={:.0f} 1/m @F={:.0e}  "
          "Im(eps) {:.3f}->{:.3f}".format(abs(eps0 - eps_bg), da[kmax], Fs[kmax],
                                          eps_bg.imag, im[kmax]), flush=True)
    return bool(flat_band and on_state and absorb_up)


def main():
    ok1 = part1_solver_vs_analytic()
    ok2 = part2_qcse_physics()
    ok3 = part3_electroabsorption()
    ok = ok1 and ok2 and ok3
    print("[q] *** QCSE ELECTRO-ABSORPTION (Stark solver == analytic; quadratic redshift; "
          "field-ON absorption): {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
