"""REL7: stress / thermal-gradient migration -- atomic flux driven by mechanical-stress and
temperature GRADIENTS (the electromigration-free companion: it acts even below the Blech threshold,
and the sharp grad T at the ENZ-absorbing ITO / metal-mirror interface drives it).

Korhonen back-stress evolution (1-D line, stress relaxing toward a free surface/via):

    d(sigma)/dt = d/dx [ kappa d(sigma)/dx ],   kappa = D_a * B * Omega / (kB * T),
    D_a = D0 * exp(-Q / (kB * T))

(B = bulk modulus [Pa], Omega = atomic volume [m^3]; kappa has m^2/s). A void nucleates where the
TENSILE stress reaches sigma_crit. The Soret (thermal-gradient) atomic flux is
J_atom = -(D_a C / (kB T)) (Q* / T) grad T (Q* = heat of transport).

DRIVER NOTE: the mechanical residual stress comes from the REL6 reliability-local schema
(biaxial_stress_Pa); a full stress-FIELD solver is the roadmap-flagged follow-on. Pure numpy/scipy;
oracles in validation/reliability_stressmig.py (the constant-kappa PDE has the exact
erf(x / 2 sqrt(kappa t)) solution).
"""

from __future__ import annotations

import numpy as np

from dynameta.constants import KB as KB_J_K    # single-source CODATA (was re-declared here)

KB_EV_K = 8.617333262e-5                       # eV/K Boltzmann (constants.py carries only KB in J/K)


def korhonen_kappa_m2_s(T_K: float, *, D0_m2_s: float, Q_eV: float, B_Pa: float,
                        Omega_m3: float) -> float:
    """kappa = D_a B Omega / (kB T) with the Arrhenius diffusivity D_a = D0 exp(-Q/kBT)."""
    if not (T_K > 0.0 and D0_m2_s > 0.0 and B_Pa > 0.0 and Omega_m3 > 0.0):
        raise ValueError("stress migration: T, D0, B, Omega must be > 0")
    if Q_eV < 0.0:
        raise ValueError("stress migration: Q_eV must be >= 0")
    Da = D0_m2_s * np.exp(-Q_eV / (KB_EV_K * T_K))
    return float(Da * B_Pa * Omega_m3 / (KB_J_K * T_K))


def korhonen_relax(x_m, t_s, *, sigma0_Pa: float, kappa_m2_s: float, n_save: int = 0):
    """Evolve the 1-D Korhonen stress PDE from a uniform residual sigma0 with a stress-FREE end at
    x = 0 (a via/surface relieving stress) and a blocked far end (d sigma/dx = 0 at x = L), by the
    method of lines (scipy LSODA). Returns sigma(x) at t_s (and intermediate saves if n_save > 0:
    (sigma_final, t_saves, sigma_saves)). For kappa t << L^2 the exact semi-infinite solution is
    sigma = sigma0 * erf(x / (2 sqrt(kappa t))) -- the validation gate."""
    from scipy.integrate import solve_ivp
    x = np.asarray(x_m, dtype=np.float64)
    if x.ndim != 1 or x.size < 5 or np.any(np.diff(x) <= 0):
        raise ValueError("stress migration: x_m must be 1D strictly increasing with >= 5 nodes")
    if not (t_s > 0.0 and kappa_m2_s > 0.0):
        raise ValueError("stress migration: t_s and kappa must be > 0")
    h = float(x[1] - x[0])
    if not np.allclose(np.diff(x), h, rtol=1e-8, atol=0.0):
        raise ValueError("stress migration: x_m must be uniformly spaced")
    n = x.size

    def rhs(tt, s):
        d = np.empty(n)
        d[0] = 0.0                                          # Dirichlet sigma(0) = 0 held fixed
        d[1:-1] = kappa_m2_s * (s[2:] - 2.0 * s[1:-1] + s[:-2]) / h ** 2
        d[-1] = kappa_m2_s * 2.0 * (s[-2] - s[-1]) / h ** 2  # Neumann mirror at the blocked end
        return d

    s0 = np.full(n, float(sigma0_Pa))
    s0[0] = 0.0
    t_eval = np.linspace(0.0, float(t_s), max(2, n_save + 2)) if n_save > 0 else [0.0, float(t_s)]
    sol = solve_ivp(rhs, (0.0, float(t_s)), s0, t_eval=t_eval, method="LSODA",
                    rtol=1e-9, atol=abs(sigma0_Pa) * 1e-10 + 1e-12)
    if not sol.success:
        raise RuntimeError("stress migration: PDE integration failed ({})".format(sol.message))
    if n_save > 0:
        return sol.y[:, -1], np.asarray(sol.t), sol.y
    return sol.y[:, -1]


def void_nucleates(sigma_Pa, sigma_crit_Pa: float) -> bool:
    """Threshold criterion: a void nucleates where the tensile stress reaches sigma_crit
    (sigma_crit = inf -> never)."""
    if not (sigma_crit_Pa > 0.0):
        raise ValueError("stress migration: sigma_crit_Pa must be > 0")
    return bool(np.any(np.asarray(sigma_Pa, dtype=np.float64) >= sigma_crit_Pa))


def soret_flux_per_m2_s(C_per_m3: float, T_K: float, gradT_K_m: float, *, D_a_m2_s: float,
                        Qstar_eV: float) -> float:
    """Thermal-gradient (Soret) atomic flux J = -(D_a C / (kB T)) (Q*/T) grad T [atoms/(m^2 s)]
    (down the gradient for Q* > 0)."""
    if not (C_per_m3 > 0.0 and T_K > 0.0 and D_a_m2_s > 0.0):
        raise ValueError("soret_flux: C, T, D_a must be > 0")
    Qstar_J = Qstar_eV * 1.602176634e-19
    return float(-(D_a_m2_s * C_per_m3 / (KB_J_K * T_K)) * (Qstar_J / T_K) * gradT_K_m)
