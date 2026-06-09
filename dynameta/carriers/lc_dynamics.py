"""
Time-domain nematic liquid-crystal director DYNAMICS (Erickson-Leslie relaxation) for reconfigurable
(LC) metasurfaces. This is the SWITCHING-SPEED axis the static director solvers (lc_director.py) lack:
given a time-dependent drive voltage V(t), it evolves the director tilt theta(z, t) by the rotational-
viscosity torque balance

    gamma1 dtheta/dt = d/dz[K_eff dtheta/dz] - (1/2) dK_eff/dtheta (dtheta/dz)^2 ... (Frank elastic)
                       - eps0 (eps_para - eps_perp) E(z)^2 sin theta cos theta        (dielectric)
                       + flexo_direct_torque                                          (flexoelectric)

with K_eff(theta) = K11 sin^2 theta + K33 cos^2 theta (FIELD-AXIS convention: theta = 0 homeotropic,
theta = pi/2 planar, matching lc_director.py's two-constant statics). The standard Frank elastic torque
expands to K_eff theta'' + (K11 - K33) sin theta cos theta (theta')^2. Strong anchoring fixes
theta(0) = theta(d) = theta_b for all t (dtheta/dt = 0 at the plates). The PDE is reduced to ODEs by
method-of-lines (central differences on a uniform z grid) and integrated with scipy.integrate.solve_ivp
(implicit BDF by default -- the dielectric torque is stiff). The electric field E(z, t) is the same
quasi-static voltage-division solve as the statics (lc_director.solve_lc_field_profile), so fixed
dielectric layers (V_lc < V_app), cylindrical geometry and flexoelectric self-consistency all carry over.

Reduces to the analytic single-relaxation time tau = gamma1 d^2 / (K pi^2) for a small field-OFF
perturbation (1-constant limit). Produces rise/decay (10-90 / 90-10) and settle-time switching metrics,
and the n_eff(t) optical trace (via lc_director.n_eff_from_theta_profile). Bridge a director frame to the
optical LiquidCrystalModel with lc_director.director_to_extra_fields. Pure numpy/scipy; SI units.

BACKFLOW (optional, LOCAL effective-viscosity model): set include_backflow=True to couple the director to
the induced shear flow via the Leslie coefficients alpha2/alpha3. The full Leslie-Ericksen problem solves
the director AND the Navier-Stokes flow self-consistently; here we use the standard LOCAL reduction --
the flow is slaved to the director rotation, giving an effective rotational viscosity
gamma1_eff(theta) = gamma1 - g(theta)^2 / eta_shear  with  g(theta) = alpha2 sin^2 theta + alpha3 cos^2
theta. Because gamma1_eff < gamma1, backflow SPEEDS UP reorientation (faster rise and decay). Limitations
of the local model: it neglects the no-slip wall boundary condition on the flow and the director-
reorientation "optical bounce" kickback, so it OVERESTIMATES the speedup relative to a real cell (treat
the magnitude as an upper bound; the direction and the alpha2=alpha3=0 -> gamma1 off-limit are exact).
With include_backflow=False (the default) only gamma1 is used -- the standard single-relaxation model,
thermodynamically valid (free energy decreases monotonically to the static equilibrium) but ~10-20% faster
in switching time than a real cell. Validated: the dynamics steady state matches the static BVP
(lc_director.director_profile_bvp) to < 0.04 deg at fixed V (same torque balance), AND the weak-anchoring
dynamics (finite W_anchor_J_m2 with surface viscosity gamma_s_Pa_s_m) relax to the static weak-anchoring
BVP (Rapini-Papoular natural BC) to < 0.5 deg.

Ported/adapted from the external lc_dynamics_base solver. The 1-constant static planar driver
(lc_director.director_profile) + the thermal-pulse LCRelaxation (switching.py) are complementary and
untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from dynameta.constants import EPS0
from dynameta.carriers.lc_director import (
    LCGeometry, compute_lc_geometry, solve_lc_field_profile, flexo_direct_torque,
    n_eff_from_theta_profile, freedericksz_threshold_V)

__all__ = [
    "LCDynamics", "LCDynamicsResult",
    "v_step", "v_rc_mirrored", "make_three_stage_voltage_func",
    "crossing_time", "step_rise_10_90", "step_decay_90_10",
]


# -----------------------------------------------------------------------------
# Drive waveforms
# -----------------------------------------------------------------------------
def v_step(t: float, V0: float, Ton: float) -> float:
    """Ideal step pulse: V = V0 for 0 <= t < Ton, else 0."""
    if t < 0.0:
        return 0.0
    return float(V0) if t < float(Ton) else 0.0


def v_rc_mirrored(t: float, V0: float, Ton: float, tau_rc: float) -> float:
    """First-order RC rise/fall pulse: rise V0(1 - e^{-t/tau}) for t<Ton, then exp decay from V(Ton)."""
    if t < 0.0:
        return 0.0
    if tau_rc <= 0.0:
        return v_step(t, V0, Ton)
    if t < float(Ton):
        return float(V0) * (1.0 - math.exp(-t / tau_rc))
    v_off = float(V0) * (1.0 - math.exp(-float(Ton) / tau_rc))
    return v_off * math.exp(-(t - float(Ton)) / tau_rc)


def make_three_stage_voltage_func(V_turn: float, V_max: float, V_decay: float,
                                  T_turn: float, T_max: float, waveform: str = "step",
                                  tau_rc: float = 0.0) -> Callable[[float], float]:
    """V(t) for a V_turn -> V_max -> V_decay three-stage drive (step or RC-smoothed stage transitions)."""
    V_turn = float(V_turn); V_max = float(V_max); V_decay = float(V_decay)
    T_turn = float(T_turn); T_max = float(T_max); tau_rc = float(tau_rc)
    t1, t2 = T_turn, T_turn + T_max

    def _exp_to(target, initial, dt):
        if tau_rc <= 0.0:
            return float(target)
        return float(target + (initial - target) * math.exp(-max(0.0, dt) / tau_rc))

    v1 = _exp_to(V_turn, 0.0, T_turn)
    v2 = _exp_to(V_max, v1, T_max)

    def V_func(t):
        tt = float(t)
        if str(waveform).strip().lower() == "step" or tau_rc <= 0.0:
            return V_turn if tt < t1 else (V_max if tt < t2 else V_decay)
        if tt < t1:
            return _exp_to(V_turn, 0.0, tt)
        if tt < t2:
            return _exp_to(V_max, v1, tt - t1)
        return _exp_to(V_decay, v2, tt - t2)

    return V_func


# -----------------------------------------------------------------------------
# Switching-metric helpers (10-90 rise / 90-10 decay / settle time)
# -----------------------------------------------------------------------------
def crossing_time(t, y, level: float, direction: str) -> float:
    """First time y crosses `level` in the given direction ('rising'|'falling') by linear interp."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    if t.size != y.size or t.size < 2:
        return float("nan")
    for i in range(t.size - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if direction == "rising" and y0 < level <= y1:
            return float(t[i] + (level - y0) / (y1 - y0 + 1e-300) * (t[i + 1] - t[i]))
        if direction == "falling" and y0 > level >= y1:
            return float(t[i] + (level - y0) / (y1 - y0 + 1e-300) * (t[i + 1] - t[i]))
    return float("nan")


def step_rise_10_90(t, y, Ton: float) -> float:
    """10-90 (or 90-10) transition time on the ON segment t <= Ton (sign-agnostic on the swing)."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    on = t <= float(Ton)
    if np.count_nonzero(on) < 3:
        return float("nan")
    to, yo = t[on], y[on]
    y0 = float(yo[0]); ymx = float(np.nanmax(yo)); ymn = float(np.nanmin(yo))
    yext = ymx if abs(ymx - y0) >= abs(ymn - y0) else ymn
    amp = yext - y0
    if not math.isfinite(amp) or abs(amp) < 1e-18:
        return float("nan")
    y10, y90 = y0 + 0.1 * amp, y0 + 0.9 * amp
    d = "rising" if amp > 0 else "falling"
    dt = crossing_time(to, yo, y90, d) - crossing_time(to, yo, y10, d)
    return dt if (math.isfinite(dt) and dt >= 0) else float("nan")


def step_decay_90_10(t, y, Ton: float, *, min_swing: float = 1e-3, on_swing_frac: float = 0.1) -> float:
    """90-10 (or 10-90) transition time on the OFF segment t >= Ton. Returns NaN when the cell barely
    switched: the decay amplitude must exceed BOTH an absolute floor `min_swing` AND `on_swing_frac` of
    the ON-segment swing -- otherwise the '90%'/'10%' levels straddle solver noise and the reported decay
    time flips wildly with atol/n_t (a real noise bug the swing guard removes)."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    off = t >= float(Ton)
    if np.count_nonzero(off) < 5:
        return float("nan")
    on = t <= float(Ton)
    on_swing = (float(np.nanmax(y[on])) - float(np.nanmin(y[on]))) if np.count_nonzero(on) >= 2 else 0.0
    to, yo = t[off], y[off]
    y_start = float(yo[0]); n_tail = max(3, int(0.05 * yo.size)); y_end = float(np.nanmean(yo[-n_tail:]))
    amp = y_start - y_end
    if not math.isfinite(amp) or abs(amp) < max(float(min_swing), float(on_swing_frac) * on_swing):
        return float("nan")
    y90, y10 = y_end + 0.9 * amp, y_end + 0.1 * amp
    d = "falling" if amp > 0 else "rising"
    dt = crossing_time(to, yo, y10, d) - crossing_time(to, yo, y90, d)
    return dt if (math.isfinite(dt) and dt >= 0) else float("nan")


# -----------------------------------------------------------------------------
# Director dynamics core
# -----------------------------------------------------------------------------
@dataclass
class LCDynamicsResult:
    t_s: np.ndarray
    theta_zt_rad: np.ndarray     # (nz, nt) FIELD-AXIS tilt
    theta_mid_rad: np.ndarray    # (nt,) midplane tilt
    V_app_V: np.ndarray
    V_lc_V: np.ndarray
    n_eff: np.ndarray            # (nt,) optical-path n_eff (NaN if n_o/n_e not given)
    z_m: np.ndarray
    rise_10_90_s: float = float("nan")
    decay_90_10_s: float = float("nan")


@dataclass
class LCDynamics:
    """Erickson-Leslie director dynamics for a nematic cell (FIELD-AXIS theta). All SI.

    K11/K33 elastic constants (J/m), gamma1 rotational viscosity (Pa s), eps_para/eps_perp static
    permittivities, theta_b strong-anchoring tilt (rad; ~pi/2 = planar cell). Geometry/field-model
    params (d_planar or a/b, fixed layers, eps_in/out) mirror lc_director.solve_lc_field_profile.
    Flexoelectric (e1, e3) and the cyl elastic correction are gated OFF by default."""
    K11: float
    K33: float
    gamma1: float
    eps_para: float
    eps_perp: float
    theta_b_rad: float = math.radians(89.9)
    geometry: str = "planar"
    d_planar: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    t_in: float = 0.0
    t_out: float = 0.0
    eps_in: float = 7.5
    eps_out: float = 7.5
    field_model: str = "uniform"
    include_cyl_elastic_corr: bool = False
    include_flexo: bool = False
    flexo_e1: float = 0.0
    flexo_e3: float = 0.0
    flexo_self_consistent: bool = False
    n_o: Optional[float] = None
    n_e: Optional[float] = None
    opt_model: str = "extra_k_radial"
    nz: int = 121
    # finite surface anchoring (None -> strong/clamped, byte-identical): Rapini-Papoular surface torque
    # balance at the plates with a surface viscosity gamma_s [Pa s m]; theta_easy defaults to theta_b.
    W_anchor_J_m2: Optional[float] = None
    theta_easy_rad: Optional[float] = None
    gamma_s_Pa_s_m: float = 1.0e-10
    # BACKFLOW (Leslie director-flow coupling), LOCAL effective-viscosity model (off -> byte-identical):
    # gamma1_eff(theta) = gamma1 - g(theta)^2 / eta_shear, g(theta) = alpha2 sin^2 + alpha3 cos^2, so the
    # director reorientation is SPED UP (lower effective viscosity) by the induced shear flow. alpha2/
    # alpha3 are Leslie coefficients (Pa s), eta_shear an effective Miesowicz shear viscosity (Pa s).
    include_backflow: bool = False
    alpha2_Pa_s: float = -0.08     # ~5CB (alpha2 < 0)
    alpha3_Pa_s: float = -0.003    # ~5CB (alpha3 small, < 0)
    eta_shear_Pa_s: float = 0.08

    def geometry_obj(self) -> LCGeometry:
        return compute_lc_geometry(geometry=self.geometry, nz=int(self.nz), d_planar=self.d_planar,
                                   a=self.a, b=self.b, t_in=self.t_in, t_out=self.t_out)

    def tau_1const_s(self, K: Optional[float] = None) -> float:
        """Analytic 1-constant relaxation time tau = gamma1 d_lc^2 / (K pi^2) (small-perturbation,
        field-off). K defaults to K11."""
        geo = self.geometry_obj()
        Kc = float(self.K11 if K is None else K)
        return float(self.gamma1) * geo.d_lc ** 2 / (Kc * math.pi ** 2)

    def _field_kwargs(self) -> dict:
        return dict(eps_para=self.eps_para, eps_perp=self.eps_perp, field_model=self.field_model,
                    t_in=self.t_in, t_out=self.t_out, eps_in=self.eps_in, eps_out=self.eps_out,
                    a=self.a, b=self.b,
                    e1=(self.flexo_e1 if self.include_flexo else 0.0),
                    e3=(self.flexo_e3 if self.include_flexo else 0.0),
                    flexo_self_consistent=bool(self.include_flexo and self.flexo_self_consistent))

    def simulate(self, t_eval, V_func: Callable[[float], float],
                 theta0_rad: "Optional[np.ndarray]" = None, *, method: str = "BDF",
                 rtol: float = 1e-6, atol: float = 1e-9, fit_ton_s: Optional[float] = None,
                 max_step_to_eval: bool = False) -> LCDynamicsResult:
        """Evolve theta(z, t) under the drive V_func(t). Returns an LCDynamicsResult with the full
        theta(z,t), the midplane tilt, V_app/V_lc/n_eff traces, and (if fit_ton_s given) the 10-90 rise
        and 90-10 decay switching times of theta_mid about that on-time."""
        from scipy.integrate import solve_ivp
        geo = self.geometry_obj()
        z = geo.z_m; r = geo.r_m; dz = float(z[1] - z[0]); N = z.size
        thb = float(self.theta_b_rad)
        K11 = float(self.K11); K33 = float(self.K33); dEps = float(self.eps_para) - float(self.eps_perp)
        g1 = float(self.gamma1)
        if not (g1 > 0):
            raise ValueError("gamma1 must be > 0")
        fkw = self._field_kwargs()
        flexo_on = bool(self.include_flexo and (self.flexo_e1 or self.flexo_e3))
        cyl_corr = bool(self.geometry == "cyl" and self.include_cyl_elastic_corr)
        # finite-anchoring (None -> strong) + backflow setup
        W_anchor = float(self.W_anchor_J_m2) if (self.W_anchor_J_m2 is not None
                                                 and self.W_anchor_J_m2 > 0.0) else None
        theta_easy = float(self.theta_easy_rad) if self.theta_easy_rad is not None else thb
        gamma_s = float(self.gamma_s_Pa_s_m)
        backflow = bool(self.include_backflow and (self.alpha2_Pa_s or self.alpha3_Pa_s))
        a2, a3, eta_sh = float(self.alpha2_Pa_s), float(self.alpha3_Pa_s), float(self.eta_shear_Pa_s)

        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.size < 5 or not np.all(np.diff(t_eval) > 0):
            raise ValueError("t_eval must be strictly increasing with >= 5 points")
        if theta0_rad is None:
            th0 = np.full(N, thb, dtype=float)
        else:
            th0 = np.asarray(theta0_rad, dtype=float).copy()
            if th0.size != N:
                raise ValueError("theta0_rad length must equal nz")
            if W_anchor is None:
                th0[0] = th0[-1] = thb

        def rhs(t, th_vec):
            th = np.array(th_vec, dtype=float, copy=True)
            if W_anchor is None:
                th[0] = th[-1] = thb                              # strong anchoring; weak lets the surface move
            dth = np.gradient(th, dz)                              # 1st derivative (term2 + cyl correction)
            # 2nd derivative via the proper 3-point tridiagonal stencil (the dz-exact Laplacian), NOT
            # np.gradient(np.gradient(.)) which is a 2*dz-wide stencil ~4-6x less accurate near the walls.
            d2th = np.empty_like(th)
            d2th[1:-1] = (th[2:] - 2.0 * th[1:-1] + th[:-2]) / (dz * dz)
            d2th[0] = d2th[-1] = 0.0
            E, _vlc = solve_lc_field_profile(th, float(V_func(float(t))), geo, **fkw)
            s = np.sin(th); c = np.cos(th)
            Keff = K11 * s * s + K33 * c * c
            term1 = Keff * d2th + (Keff * (dth / np.maximum(r, 1e-30)) if cyl_corr else 0.0)
            term2 = (K11 - K33) * s * c * (dth * dth)
            term3 = -EPS0 * dEps * (E * E) * s * c
            term4 = (flexo_direct_torque(th, E, np.gradient(E, dz), self.flexo_e1, self.flexo_e3,
                                         geometry=geo.geometry, r=r) if flexo_on else 0.0)
            torque = term1 + term2 + term3 + term4
            # BACKFLOW: the induced shear flow lowers the EFFECTIVE rotational viscosity (local model),
            # gamma1_eff(theta) = gamma1 - g(theta)^2/eta_shear, g = alpha2 sin^2 + alpha3 cos^2; floored
            # to stay positive. Speeds up the reorientation. Off -> g1_eff == g1 (byte-identical).
            if backflow:
                g_th = a2 * s * s + a3 * c * c
                g1_eff = np.maximum(g1 - g_th * g_th / eta_sh, 0.05 * g1)
            else:
                g1_eff = g1
            dth_dt = torque / g1_eff
            if W_anchor is None:
                dth_dt[0] = dth_dt[-1] = 0.0
            else:
                # Rapini-Papoular surface torque balance with surface viscosity gamma_s (one-sided
                # theta' at each plate): steady state = the static weak-anchoring BC.
                thp0 = (th[1] - th[0]) / dz
                thpd = (th[-1] - th[-2]) / dz
                dth_dt[0] = (Keff[0] * thp0 - 0.5 * W_anchor * math.sin(2.0 * (th[0] - theta_easy))) / gamma_s
                dth_dt[-1] = (-Keff[-1] * thpd - 0.5 * W_anchor * math.sin(2.0 * (th[-1] - theta_easy))) / gamma_s
            return dth_dt

        kwargs = {}
        if max_step_to_eval:
            d_eval = np.diff(t_eval)
            ms = float(np.nanmin(d_eval)) if d_eval.size else float("inf")
            if math.isfinite(ms) and ms > 0:
                kwargs["max_step"] = ms
        sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), th0, t_eval=t_eval,
                        method=str(method), rtol=float(rtol), atol=float(atol), **kwargs)
        if not sol.success:
            raise RuntimeError("LCDynamics.simulate solve_ivp failed: {}".format(sol.message))

        t_s = np.asarray(sol.t, dtype=float)
        theta_zt = np.asarray(sol.y, dtype=float)
        V_app = np.array([float(V_func(float(tt))) for tt in t_s], dtype=float)
        V_lc = np.empty_like(t_s); neff = np.full_like(t_s, np.nan)
        have_opt = (self.n_o is not None and self.n_e is not None)
        for i in range(t_s.size):
            th = theta_zt[:, i].copy(); th[0] = th[-1] = thb
            _E, V_lc[i] = solve_lc_field_profile(th, float(V_app[i]), geo, **fkw)
            if have_opt:
                neff[i] = n_eff_from_theta_profile(th, z, self.n_o, self.n_e, model=self.opt_model,
                                                   d_lc=geo.d_lc)
        theta_mid = theta_zt[N // 2, :].copy()
        res = LCDynamicsResult(t_s=t_s, theta_zt_rad=theta_zt, theta_mid_rad=theta_mid, V_app_V=V_app,
                               V_lc_V=V_lc, n_eff=neff, z_m=z)
        if fit_ton_s is not None:
            trace = neff if have_opt else theta_mid
            res.rise_10_90_s = step_rise_10_90(t_s, trace, float(fit_ton_s))
            res.decay_90_10_s = step_decay_90_10(t_s, trace, float(fit_ton_s))
        return res

    def simulate_pulse(self, V0: float, Ton: float, T_end: float, *, n_t: int = 400,
                       waveform: str = "step", tau_rc: float = 0.0, method: str = "BDF",
                       rtol: float = 1e-6, atol: float = 1e-9) -> LCDynamicsResult:
        """Convenience: one on/off pulse (step or RC) on a fresh planar/relaxed start, with rise/decay
        switching metrics fit about Ton."""
        t_eval = np.linspace(0.0, float(T_end), int(n_t))
        if str(waveform).strip().lower() == "rc":
            vf = lambda t: v_rc_mirrored(t, V0, Ton, tau_rc)
        else:
            vf = lambda t: v_step(t, V0, Ton)
        return self.simulate(t_eval, vf, theta0_rad=None, method=method, rtol=rtol, atol=atol,
                             fit_ton_s=float(Ton))
