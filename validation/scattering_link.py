"""Validate the shared ScatteringModel link (roadmap R3): ONE momentum-relaxation law tau(n;T) drives
BOTH the optical Drude damping gamma(n)=1/tau AND the transport drift mobility mu(n)=q/(m_cond 1/tau),
removing the hidden inconsistency of fitting them independently. The bidirectional check: the SAME tau
must reproduce the reference ITO optical eps AND give a physically sane mobility.

GATE A-OPTICAL (off-switch exactness): with a CONSTANT tau0 = 1/1.1e14 s, the linked DrudeOptical eps
        equals the constant-gamma DrudeOptical eps to < 1e-12 over n,lambda (the link is exact when tau
        is constant -- it reduces to today's behavior).
GATE A-TRANSPORT (sane mobility from the SAME tau): mu = q tau0 / m_cond with m_cond ~ 0.35 m_e lands in
        20-60 cm^2/Vs (reference ITO ~30; the ~1.5x band is the documented DC-vs-optical mass/Hall caveat).
GATE A-FIT (independent code path): re-fit the linked eps with the existing fit_drude_params -> recovers
        gamma ~ 1.1e14 and m_opt ~ 0.225 m_e to the fitter's rms (cross-checks the link via a separate path).
GATE B-MATTHIESSEN (trend): constant tau -> gamma(n) flat; with an ionized-impurity term -> gamma(n)
        MONOTONICALLY increases with n (ENZ loss grows with accumulation, the physically-correct trend).

Run: python -m validation.scattering_link
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.constants import Q_E, M_E
from dynameta.materials import (DrudeOptical, TransportModel, Material, KaneOpticalMass,
                                MatthiessenGamma, ScatteringModel, fit_drude_params)

N = np.geomspace(1e26, 2e27, 9)
LAMS = np.linspace(1200e-9, 2000e-9, 21)
GAMMA0, M_OPT = 1.1e14, 0.225 * M_E
M_COND = 0.35 * M_E


def main():
    print("[sl] === shared ScatteringModel link (one tau -> optical gamma + transport mobility) ===",
          flush=True)

    sm = ScatteringModel(one_over_tau=MatthiessenGamma(gamma_const_rad_s=GAMMA0), m_cond_kg=M_COND)
    opt = DrudeOptical(eps_inf=4.25, m_opt_kg=M_OPT, gamma_rad_s=9.9e9)     # placeholder gamma
    tr = TransportModel(n_bg_m3=6e26, eps_static=9.5, dos_mass_kg_of_n_m3=lambda n: M_COND)
    mat = Material("ito", optical=opt, transport=tr, scattering=sm)
    ref = DrudeOptical(eps_inf=4.25, m_opt_kg=M_OPT, gamma_rad_s=GAMMA0)    # the constant-gamma truth

    dmax = 0.0
    for lam in LAMS:
        dmax = max(dmax, float(np.max(np.abs(mat.optical.eps(lam, n_m3=N) - ref.eps(lam, n_m3=N)))))
    g_opt = dmax < 1e-12
    print("[sl] A-OPTICAL linked eps == constant-gamma Drude: max|d eps|={:.1e} -> {}".format(
        dmax, "OK" if g_opt else "FAIL"), flush=True)

    mu = float(mat.transport.mobility_m2Vs_of_n_m3(6e26)) * 1e4              # cm^2/Vs
    g_tr = 20.0 < mu < 60.0
    print("[sl] A-TRANSPORT mu from same tau = {:.1f} cm^2/Vs (reference ITO ~30; DC-vs-optical band) -> {}".format(
        mu, "OK" if g_tr else "FAIL"), flush=True)

    # independent code path: re-fit the LINKED model's eps (audit 7.3: this used to fit
    # ref.eps -- the reference's own output -- making the leg a generate-then-recover
    # round-trip that never touched the ScatteringModel link it claims to cross-check)
    n_fit = 6e26
    eps = np.array([complex(mat.optical.eps(l, n_m3=n_fit)) for l in LAMS])
    fit = fit_drude_params(n_m3=np.full_like(LAMS, n_fit), lambda_m=LAMS, eps_re=eps.real,
                           eps_im=eps.imag, eps_inf0=4.0, m_eff_ratio0=0.30, gamma0=1.0e14)
    g_fit = (abs(fit["gamma_rad_s"] - GAMMA0) / GAMMA0 < 1e-3
             and abs(fit["m_opt_kg"] - M_OPT) / M_OPT < 1e-3)
    print("[sl] A-FIT re-fit linked eps: gamma={:.3e} (vs {:.3e}), m_opt/m_e={:.4f} (vs {:.4f}) rms={:.1e}"
          " -> {}".format(fit["gamma_rad_s"], GAMMA0, fit["m_opt_kg"] / M_E, M_OPT / M_E,
                          fit["rms_residual"], "OK" if g_fit else "FAIL"), flush=True)

    g_const = float(np.std(sm.gamma_optical_of_n()(N))) == 0.0
    sm_ii = ScatteringModel(one_over_tau=MatthiessenGamma(gamma_const_rad_s=5e13, bh_prefactor_rad_s=6e13,
                                                          bh_n_ref_m3=1e27,
                                                          m_opt=KaneOpticalMass(m0_kg=0.27 * M_E,
                                                                                alpha_eV=0.5)),
                            m_cond_kg=M_COND)
    g_n = sm_ii.gamma_optical_of_n()(N)
    g_trend = bool(np.all(np.diff(g_n) > 0)) and g_const
    print("[sl] B-MATTHIESSEN const-tau flat={}, ionized-impurity gamma(n) increasing with n={} -> {}".format(
        g_const, bool(np.all(np.diff(g_n) > 0)), "OK" if g_trend else "FAIL"), flush=True)

    ok = g_opt and g_tr and g_fit and g_trend
    print("[sl] *** SCATTERING LINK: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
