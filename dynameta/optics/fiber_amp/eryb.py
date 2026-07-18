"""Er:Yb co-doped fiber amplifier (EYDFA): the z-resolved coupled-power solve for a
phosphosilicate erbium/ytterbium sensitized amplifier, the sensitized counterpart of the
single-ion FiberAmplifier (steady_state.py). Ytterbium is the pump-absorbing SENSITIZER: it
has a ~30x larger 900-980 nm absorption cross-section than Er, soaks up the (typically
cladding-guided) 915/940/976 nm pump, and hands the excitation to Er by a near-resonant
dipole-dipole ENERGY TRANSFER Yb(2F5/2) + Er(4I15/2) -> Yb(2F7/2) + Er(4I11/2). The Er 4I11/2
pump band then relaxes fast (A_32 ~ 3e5-1e6 1/s in a high-phonon phosphosilicate host) to the
4I13/2 metastable level, from which the 1530-1565 nm C-band signal is amplified exactly as in a
plain EDFA. The point of the co-dope is CLADDING-PUMPABILITY at high power: Yb makes the weak,
narrow Er 980 nm line irrelevant, so a multimode diode pump can be absorbed in a few metres.

LEVELS. Yb: ground n_Yb1 (2F7/2), excited n_Yb2 (2F5/2), N_Yb = n_Yb1 + n_Yb2. Er: n1 (4I15/2
ground), n2 (4I13/2 metastable), n3 (4I11/2 pump band), N_Er = n1 + n2 + n3. fiber.n_t_m3 is
N_Er; n_yb_m3 is N_Yb.

FAST-4I11/2 LIMIT (standard for phosphosilicate; A_32 >> k_back n_Yb1). n3 is adiabatically
eliminated: A_32 n3 -> (transfer + direct Er pump) feeding n2, back-transfer -> 0, and N_Er ~
n1 + n2. The Er block collapses to the two-level EDFA algebra with an EXTRA optical-pump-like
term = the Yb->Er transfer, and the direct Er absorption (which in the two-level treatment also
promotes n1 -> n2, whether it lands on the 4I11/2 980 nm line or the 4I13/2 signal band) is the
usual R_a_Er. With f2 = n2/N_Er (Er metastable fraction) and b2 = n_Yb2/N_Yb (Yb inversion) the
z-local steady state is the coupled pair (per ion, 1/s):

    Er:  R_a_Er (1 - f2) - R_e_Er f2 - f2/tau_Er + k_tr b2 N_Yb (1 - f2) - C_up N_Er f2^2 = 0
    Yb:  R_a_Yb (1 - b2) - R_e_Yb b2 - b2/tau_Yb - k_tr b2 N_Er (1 - f2)               = 0

    R_{a/e}_ion = SUM_k Gamma_k sigma_{a/e}_ion,k P_k / (h nu_k A_dope)     (per-ion rates)

DERIVATION OF THE TRANSFER TERMS (resolves the dossier's flagged eta_tr N_Yb-vs-N_Er
ambiguity). The volumetric forward transfer rate is R_tr = k_tr n_Yb2 n1 (an excited Yb donor
meeting a GROUND Er acceptor). Per Yb ion (divide by N_Yb) this is k_tr n1 = k_tr N_Er (1 - f2):
a DRAIN on the Yb inversion set by the Er GROUND density, NOT by N_Yb -- so the transfer-out
term in the Yb equation is k_tr b2 N_Er (1 - f2). Per Er ground ion (divide the same R_tr by
N_Er) it is k_tr n_Yb2 = k_tr b2 N_Yb, an extra pump acting on the Er ground fraction (1 - f2) --
the transfer-in term in the Er equation. The physical weak-signal transfer efficiency is
therefore eta_tr = k_tr N_Er tau_Yb / (1 + k_tr N_Er tau_Yb) (N_Er, the acceptor density), which
is what transfer_efficiency() reduces to at low power -- see the DISCREPANCY NOTE below.

k_back optional refinement. Retaining back-transfer k_back n3 n_Yb1 in the (eliminated) n3
balance multiplies BOTH transfer terms by phi = A_32 / (A_32 + k_back (1 - b2) N_Yb) (the
fraction of 4I11/2 population that relaxes to 4I13/2 before back-transferring). For literature
k_back ~ 1e-24 and A_32 ~ 5e5 this is phi ~ 1 - 2e-4 (negligible), consistent with the
fast-limit assumption; it is included exactly (phi = 1 when k_back = 0, the default).

PROPAGATION. Every channel sees BOTH ions at its wavelength (a 976 pump interacts almost only
with Yb, a 1550 signal almost only with Er since sigma_Yb(1550) = 0 numerically):

    dP_k/dz = u_k { Gamma_k [ N_Er (sigma_e_Er,k f2 - sigma_a_Er,k (1 - f2))
                            + N_Yb (sigma_e_Yb,k b2 - sigma_a_Yb,k (1 - b2)) ] - l_k } P_k
              + [ASE] u_k Gamma_k m h nu_k dnu_k [ N_Er sigma_e_Er,k f2 + N_Yb sigma_e_Yb,k b2 ]

The spontaneous source is cross-section-weighted from both ions, so a C-band ASE bin is seeded
only by Er and a 1000-1100 nm (yb_ase) bin only by Yb, with no hand-assigned band boundary.

SOLVE. The two-point boundary-value problem is closed by the SAME relaxation scheme as
FiberAmplifier.solve (alternate forward 0->L / backward L->0 initial-value passes with the
other direction's z-profile frozen through the lean uniform-mesh interpolator, to convergence on
endpoints + interior). The z-local (f2, b2) are found by reducing the coupled pair to a single
bracketed scalar equation in f2 -- b2(f2) is a closed form (the Yb equation is LINEAR in b2 at
fixed f2) substituted into the Er residual H(f2), which satisfies H(0) >= 0 and H(1) < 0, so a
safeguarded Newton/bisection on [0, 1] is unconditionally robust. This sidesteps the naive
block fixed point, whose positive Yb<->Er feedback has spectral radius ~ k_tr^2 N_Er N_Yb
tau_Er tau_Yb >> 1 (the "stiff when k_tr N_Er >> 1/tau_Yb" regime) and diverges.

DISCREPANCY NOTE (bidirectional adversarial; dossier ">95% transfer" gate). With the measured
phosphosilicate numbers k_tr ~ 1.1e-22-2e-22 m^3/s, tau_Yb = 1.45 ms and N_Er = 2e25 m^-3, the
weak-signal transfer efficiency is k_tr N_Er tau_Yb ~ 4-6, i.e. eta_tr ~ 0.80-0.85, NOT >0.95.
The >95% figure requires either a larger k_tr N_Er product (higher Er loading or the fast-
transfer end of the k_tr range) or that most of the residual is recovered because a Yb photon
lost to spontaneous decay is largely reabsorbed by Yb and eventually transferred; that
reabsorption cascade is NOT in this z-local model. transfer_efficiency() reports the ACTUAL
model value (a rate-integral consistent with the low-power analytic form), and the model does
NOT force the >95% number.

References: Karasek, IEEE JQE 33(9):1699 (1997) [Er:Yb rate model, k_tr]; Di Pasquale & Federighi,
JOSA B 23(3):195 (2006) [measured k_tr ~ 1.1e-22]; Paschotta et al., IEEE JQE 33(7):1049 (1997)
[Yb lifetime]; Giles & Desurvire, JLT 9(2):271 (1991) [coupled-power EDFA core]. Pure
numpy/scipy; SI units; exp(-i omega t); ASCII-only. docs/fiber_amp_model_spec.md sec.1;
FORMULATION DOSSIER MODULE 1.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from dynameta.constants import C_LIGHT, H_PLANCK
from dynameta.optics.fiber_amp.rare_earth import ChannelSet
from dynameta.optics.fiber_amp.spectroscopy import RareEarthIon
from dynameta.optics.fiber_amp.waveguide import FiberSpec, cladding_pump_overlap, overlap_gamma
from dynameta.optics.fiber_amp.steady_state import AseBand, Pump, Signal, SteadyStateResult

__all__ = ["ErYbAmplifier"]


class ErYbAmplifier:
    """An Er:Yb co-doped fiber amplifier: an Er ion + a Yb sensitizer + a fiber + a channel plan
    (pumps, signals, an Er-band ASE band, and an optional Yb-band ASE band). Architecturally
    parallel to FiberAmplifier so the two can be swapped: it reuses the Pump / Signal / AseBand
    dataclasses and returns the SAME SteadyStateResult. solve() returns the z-profiles, the Er
    metastable fraction (nbar2_z = f2), the Yb inversion (meta['beta_yb_z'] = b2), the signal
    gain, and the transfer diagnostics.

    Parameters
    ----------
    er_ion, yb_ion : RareEarthIon
        The Er and Yb spectroscopy (cross-sections, lifetimes). Every channel is evaluated
        against BOTH ions at its wavelength.
    fiber : FiberSpec
        The doped fiber; fiber.n_t_m3 is the Er density N_Er. Cladding pumping uses
        fiber.clad_radius_m + Pump.cladding, exactly as in steady_state.
    pumps, signals : list
        steady_state.Pump / steady_state.Signal (reused verbatim).
    ase : AseBand, optional
        The Er C-band ASE band (both propagation directions). None -> ASE-free gain estimate.
    n_yb_m3 : float (keyword-only, required)
        The Yb density N_Yb.
    k_tr_m3_s : float
        Yb->Er forward energy-transfer coefficient k_tr [m^3/s] (phosphosilicate ~ 1.1e-22-5e-22;
        default 2e-22).
    k_back_m3_s : float
        Er->Yb back-transfer coefficient k_back [m^3/s] (default 0; see the phi refinement above).
    a32_per_s : float
        Er 4I11/2 -> 4I13/2 relaxation rate A_32 [1/s] (only enters through k_back; default 5e5).
    yb_ase : AseBand, optional
        A second ASE band (typically 1000-1100 nm) that tracks Yb-band / 1-um parasitic ASE.
    upconversion_C_up : float
        Er cooperative-upconversion coefficient C_up [m^3/s] (default 0; adds -C_up N_Er f2^2 to
        the Er balance, as in FiberAmplifier).
    """

    def __init__(self, er_ion: RareEarthIon, yb_ion: RareEarthIon, fiber: FiberSpec,
                 pumps: List[Pump], signals: List[Signal], ase: Optional[AseBand] = None, *,
                 n_yb_m3: float, k_tr_m3_s: float = 2.0e-22, k_back_m3_s: float = 0.0,
                 a32_per_s: float = 5.0e5, yb_ase: Optional[AseBand] = None,
                 upconversion_C_up: float = 0.0):
        if not (n_yb_m3 > 0.0):
            raise ValueError("ErYbAmplifier: n_yb_m3 (N_Yb) must be > 0")
        if not (a32_per_s > 0.0):
            raise ValueError("ErYbAmplifier: a32_per_s must be > 0")
        self.er_ion, self.yb_ion, self.fiber = er_ion, yb_ion, fiber
        self.pumps, self.signals = list(pumps), list(signals)
        self.ase, self.yb_ase = ase, yb_ase
        self._n_er = float(fiber.n_t_m3)
        self._n_yb = float(n_yb_m3)
        self._k_tr = float(k_tr_m3_s)
        self._k_back = float(k_back_m3_s)
        self._a32 = float(a32_per_s)
        self._tau_er = float(er_ion.tau_s)
        self._tau_yb = float(yb_ion.tau_s)
        self.upconversion_C_up = float(upconversion_C_up)

    # ---- channel plan --------------------------------------------------------------------
    def _plan(self):
        """Assemble the channel arrays. Every channel carries BOTH ions' cross-sections; the
        overlap Gamma and background loss are ion-independent (geometry), and a cladding pump has
        its (core) overlap replaced by the doped-fraction cladding overlap A_dope/A_clad."""
        lam, u, is_ase, dnu, kind, bc, cladding, m = [], [], [], [], [], [], [], []
        for p in self.pumps:
            lam.append(p.lambda_m); u.append(+1.0 if p.direction == "fwd" else -1.0)
            is_ase.append(False); dnu.append(0.0); kind.append("pump")
            bc.append(p.power_W); cladding.append(p.cladding); m.append(2.0)
        for s in self.signals:
            lam.append(s.lambda_m); u.append(+1.0); is_ase.append(False); dnu.append(0.0)
            kind.append("signal"); bc.append(s.power_W); cladding.append(False); m.append(2.0)
        for band in (self.ase, self.yb_ase):
            if band is not None and band.n_bins > 0:
                edges = np.linspace(band.lambda_min_m, band.lambda_max_m, band.n_bins + 1)
                centres = 0.5 * (edges[:-1] + edges[1:])
                nu_edges = C_LIGHT / edges                       # bin width in FREQUENCY
                dnu_bins = np.abs(nu_edges[:-1] - nu_edges[1:])
                for direction in (+1.0, -1.0):
                    for cwl, dv in zip(centres, dnu_bins):
                        lam.append(float(cwl)); u.append(direction); is_ase.append(True)
                        dnu.append(float(dv)); kind.append("ase"); bc.append(0.0)
                        cladding.append(False); m.append(float(band.m_modes))
        lam = np.asarray(lam); u = np.asarray(u); is_ase = np.asarray(is_ase, bool)
        dnu = np.asarray(dnu); m = np.asarray(m)
        ch_er = ChannelSet.build(self.er_ion, self.fiber, lam, u, is_ase=is_ase, dnu_hz=dnu)
        ch_yb = ChannelSet.build(self.yb_ion, self.fiber, lam, u, is_ase=is_ase, dnu_hz=dnu)
        gamma = ch_er.gamma.copy()
        for k, cl in enumerate(cladding):
            if cl:
                gamma[k] = cladding_pump_overlap(self.fiber)
        return {"lam": lam, "u": u, "is_ase": is_ase, "dnu": dnu, "m": m, "kind": kind,
                "bc": np.asarray(bc), "gamma": gamma, "loss": ch_er.loss_per_m.copy(),
                "sa_er": ch_er.sigma_a, "se_er": ch_er.sigma_e,
                "sa_yb": ch_yb.sigma_a, "se_yb": ch_yb.sigma_e}

    def _coeffs(self, pl):
        """Hoist the per-channel overlap/cross-section/frequency products into a bundle once per
        solve (mirrors steady_state._coeffs). flux_* are the per-ion pumping/emission rates per
        unit power; g_* are the modal gain coefficients; s_pref_* the ASE spontaneous prefactors."""
        gamma, dnu, m = pl["gamma"], pl["dnu"], pl["m"]
        nu = C_LIGHT / pl["lam"]
        A = self.fiber.a_dope_m2
        inv_hnuA = 1.0 / (H_PLANCK * nu * A)
        N_Er, N_Yb = self._n_er, self._n_yb
        s_er = np.where(pl["is_ase"], gamma * N_Er * pl["se_er"] * m * H_PLANCK * nu * dnu, 0.0)
        s_yb = np.where(pl["is_ase"], gamma * N_Yb * pl["se_yb"] * m * H_PLANCK * nu * dnu, 0.0)
        return {
            "flux_a_er": gamma * pl["sa_er"] * inv_hnuA,
            "flux_e_er": gamma * pl["se_er"] * inv_hnuA,
            "flux_a_yb": gamma * pl["sa_yb"] * inv_hnuA,
            "flux_e_yb": gamma * pl["se_yb"] * inv_hnuA,
            "g_e_er": gamma * N_Er * pl["se_er"], "g_a_er": gamma * N_Er * pl["sa_er"],
            "g_e_yb": gamma * N_Yb * pl["se_yb"], "g_a_yb": gamma * N_Yb * pl["sa_yb"],
            "loss": pl["loss"], "s_er": s_er, "s_yb": s_yb,
        }

    # ---- z-local coupled algebra ---------------------------------------------------------
    def _solve_fb(self, Ra_Er: float, Re_Er: float, Ra_Yb: float, Re_Yb: float
                  ) -> Tuple[float, float]:
        """Steady-state (f2, b2) at one z from the four per-ion rates. b2 is a closed form in f2
        (the Yb equation is linear in b2 for fixed f2); substituting it into the Er residual
        H(f2) leaves a single bracketed scalar equation on [0, 1] solved by a safeguarded
        Newton/bisection. H(0) >= 0 and H(1) < 0 (the transfer/absorption drive vanishes at full
        inversion, leaving only decay), so the root is bracketed and the solve cannot diverge."""
        N_Er, N_Yb = self._n_er, self._n_yb
        k_tr, k_back, A32 = self._k_tr, self._k_back, self._a32
        inv_tE, inv_tY = 1.0 / self._tau_er, 1.0 / self._tau_yb
        Cup = self.upconversion_C_up
        c = k_tr * N_Yb                                   # transfer-IN coefficient (x b2)
        s = k_tr * N_Er                                   # transfer-OUT (Yb drain) coeff (x (1-f2))
        Db0 = Ra_Yb + Re_Yb + inv_tY
        Da = Ra_Er + Re_Er + inv_tE

        def _b_phi(f):
            drain = s * (1.0 - f)
            if k_back <= 0.0:
                denom = Db0 + drain
                b = Ra_Yb / denom
                return b, 1.0, denom, drain
            b = Ra_Yb / (Db0 + drain)                     # phi = 1 seed
            phi = 1.0
            for _ in range(3):                            # inner fixed point (phi ~ 1, converges instantly)
                phi = A32 / (A32 + k_back * (1.0 - b) * N_Yb)
                b = Ra_Yb / (Db0 + drain * phi)
            return b, phi, Db0 + drain * phi, drain

        def _HdH(f):
            b, phi, denom, _ = _b_phi(f)
            dbdf = Ra_Yb * s * phi / (denom * denom)      # >0: less drain as f rises -> more Yb inversion
            G = c * phi * b
            H = Ra_Er * (1.0 - f) - Re_Er * f - f * inv_tE + G * (1.0 - f) - Cup * N_Er * f * f
            dH = -Da + c * phi * (dbdf * (1.0 - f) - b) - 2.0 * Cup * N_Er * f
            return H, dH, b

        H0, _, b0 = _HdH(0.0)
        if H0 <= 0.0:                                     # no drive (unpumped, unseeded)
            return 0.0, b0
        lo, hi, f = 0.0, 1.0, 0.5
        for _ in range(80):
            H, dH, _ = _HdH(f)
            if H > 0.0:
                lo = f
            else:
                hi = f
            f_new = f - H / dH if dH < 0.0 else 0.5 * (lo + hi)
            if not (lo < f_new < hi):
                f_new = 0.5 * (lo + hi)
            if abs(f_new - f) <= 1e-13 or (hi - lo) <= 1e-13:
                f = f_new
                break
            f = f_new
        b, _, _, _ = _b_phi(f)
        return float(min(max(f, 0.0), 1.0)), float(min(max(b, 0.0), 1.0))

    def _dP(self, c, u, P):
        """dP_k/dz [W/m] for every channel from the local power vector P (K,)."""
        P = np.maximum(P, 0.0)
        Ra_Er = float(np.dot(c["flux_a_er"], P)); Re_Er = float(np.dot(c["flux_e_er"], P))
        Ra_Yb = float(np.dot(c["flux_a_yb"], P)); Re_Yb = float(np.dot(c["flux_e_yb"], P))
        f2, b2 = self._solve_fb(Ra_Er, Re_Er, Ra_Yb, Re_Yb)
        g = (c["g_e_er"] * f2 - c["g_a_er"] * (1.0 - f2)
             + c["g_e_yb"] * b2 - c["g_a_yb"] * (1.0 - b2) - c["loss"])
        src = c["s_er"] * f2 + c["s_yb"] * b2
        return u * (g * P + src)

    def _fb_profile(self, c, P):
        """(f2(z), b2(z)) at each z given the full power profile P (K, M)."""
        M = P.shape[1]
        f2 = np.empty(M); b2 = np.empty(M)
        for j in range(M):
            Pj = np.maximum(P[:, j], 0.0)
            Ra_Er = float(np.dot(c["flux_a_er"], Pj)); Re_Er = float(np.dot(c["flux_e_er"], Pj))
            Ra_Yb = float(np.dot(c["flux_a_yb"], Pj)); Re_Yb = float(np.dot(c["flux_e_yb"], Pj))
            f2[j], b2[j] = self._solve_fb(Ra_Er, Re_Er, Ra_Yb, Re_Yb)
        return f2, b2

    # ---- solve (relaxation, mirrors FiberAmplifier.solve) --------------------------------
    def solve(self, *, n_nodes: int = 201, max_iter: int = 200, tol: float = 1e-6,
              method: str = "LSODA") -> SteadyStateResult:
        from scipy.integrate import solve_ivp
        pl = self._plan()
        c = self._coeffs(pl)
        u, is_ase, kind, bc = pl["u"], pl["is_ase"], pl["kind"], pl["bc"]
        K = pl["lam"].size
        L = self.fiber.length_m
        z = np.linspace(0.0, L, n_nodes)
        fwd = np.where(u > 0)[0]
        bwd = np.where(u < 0)[0]

        P_bwd = (np.repeat(bc[bwd][:, None], n_nodes, axis=1) if bwd.size
                 else np.zeros((0, n_nodes)))
        P_fwd = np.repeat(bc[fwd][:, None], n_nodes, axis=1)

        def _assemble(Pf, Pb):
            P = np.empty(K)
            P[fwd] = Pf
            if bwd.size:
                P[bwd] = Pb
            return P

        # Lean frozen-profile interpolator on the uniform mesh (steady_state audit S6-2).
        inv_dz = (n_nodes - 1) / L

        def _make_interp(Y):
            slopes = (Y[:, 1:] - Y[:, :-1]) * inv_dz
            ncap = n_nodes - 2

            def f(zz):
                if zz <= 0.0:
                    return Y[:, 0]
                if zz >= L:
                    return Y[:, -1]
                j = int(zz * inv_dz)
                if j > ncap:
                    j = ncap
                return Y[:, j] + slopes[:, j] * (zz - z[j])
            return f

        last_out = None
        last_prof = None
        converged = False
        it = 0
        for it in range(max_iter):
            bwd_of = _make_interp(P_bwd) if bwd.size else None

            def rhs_f(zz, Pf):
                Pb = bwd_of(zz) if bwd.size else np.zeros(0)
                return self._dP(c, u, _assemble(Pf, Pb))[fwd]

            sf = solve_ivp(rhs_f, (0.0, L), bc[fwd], t_eval=z, method=method,
                           rtol=1e-7, atol=1e-15)
            P_fwd = sf.y

            if bwd.size:
                fwd_of = _make_interp(P_fwd)

                def rhs_b(zz, Pb):
                    return self._dP(c, u, _assemble(fwd_of(zz), Pb))[bwd]

                sb = solve_ivp(rhs_b, (L, 0.0), bc[bwd], t_eval=z[::-1], method=method,
                               rtol=1e-7, atol=1e-15)
                P_bwd = sb.y[:, ::-1]

            out = np.concatenate([P_fwd[:, -1], (P_bwd[:, 0] if bwd.size else [])])
            prof = np.concatenate([P_fwd, P_bwd], axis=0) if bwd.size else P_fwd.copy()
            if last_out is not None:
                denom = np.maximum(np.abs(out), 1e-15)
                end_ok = float(np.max(np.abs(out - last_out) / denom)) < tol
                ch_peak = np.maximum(np.max(np.abs(prof), axis=1, keepdims=True), 1e-300)
                prof_ok = float(np.max(np.abs(prof - last_prof) / ch_peak)) < tol
                if end_ok and prof_ok:
                    converged = True
                    break
            last_out = out
            last_prof = prof

        P = np.empty((K, n_nodes))
        P[fwd] = P_fwd
        if bwd.size:
            P[bwd] = P_bwd
        f2, b2 = self._fb_profile(c, P)
        sig_idx = [i for i, kd in enumerate(kind) if kd == "signal"]
        gains_dB = np.array([10.0 * np.log10(P[i, -1] / bc[i]) for i in sig_idx])

        eta_tr = self._transfer_efficiency(c, P, f2, b2, z)
        yb_par_dB = self._yb_parasitic_gain_dB(P, f2, b2, z)
        m_modes = (self.ase.m_modes if self.ase is not None
                   else (self.yb_ase.m_modes if self.yb_ase is not None else 2))
        meta = {"converged": converged, "iterations": it + 1,
                "dnu_hz": pl["dnu"].copy(), "gamma": pl["gamma"].copy(), "m_modes": m_modes,
                "sigma_a": pl["sa_er"].copy(), "sigma_e": pl["se_er"].copy(),
                "sigma_a_er": pl["sa_er"].copy(), "sigma_e_er": pl["se_er"].copy(),
                "sigma_a_yb": pl["sa_yb"].copy(), "sigma_e_yb": pl["se_yb"].copy(),
                "beta_yb_z": b2, "eta_transfer": eta_tr,
                "yb_parasitic_gain_dB": yb_par_dB,
                "n_er_m3": self._n_er, "n_yb_m3": self._n_yb,
                "k_tr_m3_s": self._k_tr, "k_back_m3_s": self._k_back, "a32_per_s": self._a32}
        return SteadyStateResult(z, P, pl["lam"], u, is_ase, kind, f2, gains_dB, meta=meta)

    # ---- diagnostics ---------------------------------------------------------------------
    def _transfer_efficiency(self, c, P, f2, b2, z) -> float:
        """Fraction of Yb excited-state de-excitations that end as a USEFUL Yb->Er transfer,
        as a rate integral weighted by the excited-Yb density n_Yb2 = b2 N_Yb:

            eta_tr = INT k_tr n1 n_Yb2 dz / INT (k_tr n1 + 1/tau_Yb + R_e_Yb) n_Yb2 dz,

        where n1 = (1 - f2) N_Er is the Er ground (acceptor) density and R_e_Yb(z) is the local
        Yb stimulated-emission rate (a Yb photon re-emitted to the field instead of transferred).
        The denominator is the total Yb* loss rate: transfer + spontaneous decay + stimulated
        emission. At low power (R_e_Yb -> 0, f2 -> 0 so n1 -> N_Er) this collapses to the
        analytic k_tr N_Er tau_Yb / (1 + k_tr N_Er tau_Yb) -- see the module DISCREPANCY NOTE."""
        N_Er, N_Yb = self._n_er, self._n_yb
        n1 = (1.0 - f2) * N_Er
        nYb2 = b2 * N_Yb
        Re_Yb = np.array([float(np.dot(c["flux_e_yb"], np.maximum(P[:, j], 0.0)))
                          for j in range(P.shape[1])])
        transfer = self._k_tr * n1 * nYb2
        total = (self._k_tr * n1 + 1.0 / self._tau_yb + Re_Yb) * nYb2
        num = float(np.trapezoid(transfer, z))
        den = float(np.trapezoid(total, z))
        return num / den if den > 0.0 else 0.0

    def transfer_efficiency(self, result: SteadyStateResult) -> float:
        """eta_tr for a solved result (returns the value cached in meta by solve())."""
        return float(result.meta["eta_transfer"])

    def _yb_parasitic_gain_dB(self, P, f2, b2, z) -> float:
        """Single-pass 1030 nm Yb parasitic gain [dB] from the b2 profile (computed even with no
        yb_ase channels): G_dB = (10/ln10) INT Gamma(1030) [N_Yb (sigma_e_Yb b2 - sigma_a_Yb
        (1 - b2)) + N_Er (sigma_e_Er f2 - sigma_a_Er (1 - f2))] dz. The Er terms are ~0 at 1030
        nm. A large positive value flags 1-um parasitic-lasing risk (compare to the round-trip
        cavity loss -ln(R1 R2)/2; dossier design rule beta_Yb < ~0.05)."""
        lam = 1.030e-6
        gam = float(overlap_gamma(self.fiber, lam))
        se_yb = float(self.yb_ion.sigma_e.sigma(lam)); sa_yb = float(self.yb_ion.sigma_a.sigma(lam))
        se_er = float(self.er_ion.sigma_e.sigma(lam)); sa_er = float(self.er_ion.sigma_a.sigma(lam))
        g = gam * (self._n_yb * (se_yb * b2 - sa_yb * (1.0 - b2))
                   + self._n_er * (se_er * f2 - sa_er * (1.0 - f2)))
        ln_gain = float(np.trapezoid(g, z))
        return 10.0 / np.log(10.0) * ln_gain

    def yb_parasitic_gain_dB(self, result: SteadyStateResult) -> float:
        """1030 nm Yb parasitic gain [dB] for a solved result (cached in meta by solve())."""
        return float(result.meta["yb_parasitic_gain_dB"])
