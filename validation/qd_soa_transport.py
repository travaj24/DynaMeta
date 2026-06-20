"""QD-SOA carrier-transport coupling vs oracles. Two coupling depths beyond the lumped uniform-current
injection: (1) a REDUCED SCH (separate-confinement-heterostructure) carrier-transport stage --
amplify(transport_tau_s=...) low-passes the injection current through an SCH reservoir feeding the
wetting layer (the transport delay -> electrical modulation bandwidth limit); (2) the spatially-
resolved coupling SEAM -- a NON-UNIFORM injection PROFILE I(z) is passed as a per-slice `drive` that
init_slices seeds and rhs_fields carries through the dynamics. The SEAM is the deliverable; a DEVSIM /
drift-diffusion solve SUPPLIES the real I(z) through it (NONE is run here -- the gate drives the seam
with a 1-D current-crowding REFERENCE profile, so this ships the interface + reduced model, not a DD
physics solve).

GATE A (reductions): transport_tau_s=0 is byte-identical to lumped injection; a constant time-drive
        I(t)=I0 == the scalar drive; a uniform injection profile I(z)=I0 == the scalar drive.
GATE B (transport electrical bandwidth): under current MODULATION I(t)=I0(1+m sin 2pi f t), the gain
        modulation depth is ~unrolled below the transport pole f_t=1/(2pi tau_t) and strongly rolled
        off above it -- the SCH transport low-passes the injection (the lumped model has no such pole).
GATE C (DD injection profile -> non-uniform gain): a non-uniform I(z) gives a non-uniform per-slice
        steady gain that follows the profile (monotone with a monotone I(z)); a uniform profile
        reduces to the scalar-drive gain. (DEVSIM / drift-diffusion supplies the real I(z) through the
        same per-slice seam; here a 1-D current-spreading reference profile drives it.)
GATE D (DC invariance): the SCH transport leaves the STEADY-STATE (DC) gain unchanged for any tau_t
        (N_sch -> I tau_t/(qV) -> the same WL feed) -- it adds dynamics, not a DC shift.
GATE E (passivity): finite (no NaN) and the reductions hold.

Run: python -m validation.qd_soa_transport
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.optics.soa.qd_gain import QDGainModel, QDGainParams
from dynameta.optics.soa.traveling_wave import TravelingWaveSOA


def main():
    print("[tr] === QD-SOA carrier-transport (SCH reduced + DD injection profile) vs oracles ===",
          flush=True)
    ok = True
    m = QDGainModel(QDGainParams(n_groups=15).with_detailed_balance_taus())
    L, nz, I0 = 0.5e-3, 40, 40e-3
    soa = TravelingWaveSOA(m, L, nz, nu_s_Hz=m.p.nu0_Hz)
    dt = soa.dt
    nu0 = m.p.nu0_Hz

    # ---- GATE A: reductions ----
    nt = 6000
    Pw = np.full(nt, 1e-5)
    base = soa.amplify(Pw, I0)["P_out"]
    z0 = soa.amplify(Pw, I0, transport_tau_s=0.0)["P_out"]
    tconst = soa.amplify(Pw, np.full(nt, I0))["P_out"]                 # time-drive constant
    uprof = soa.amplify(Pw, np.full(nz, I0))["P_out"]                  # uniform spatial profile
    relA = max(float(np.max(np.abs(z0 - base))), float(np.max(np.abs(tconst - base))),
               float(np.max(np.abs(uprof - base))))
    g_a = bool(relA < 1e-12)
    ok = ok and g_a
    print("[tr] GATE A: transport_tau=0 / const time-drive / uniform profile == lumped (max|d| {:.1e}) "
          "-> {}".format(relA, "PASS" if g_a else "FAIL"), flush=True)

    # ---- GATE B: transport electrical-modulation bandwidth ----
    tau_t = 300e-12
    f_pole = 1.0 / (2.0 * np.pi * tau_t)
    ntm = 40000
    t = np.arange(ntm) * dt
    P = np.full(ntm, 1e-5)
    ratios = {}
    for fm in (0.15e9, 2.5e9):
        drv = I0 * (1.0 + 0.1 * np.sin(2.0 * np.pi * fm * t))
        d0 = soa.amplify(P, drv, transport_tau_s=0.0, return_traces=True)["g_zt"][ntm // 2:, nz // 2]
        dt_ = soa.amplify(P, drv, transport_tau_s=tau_t, return_traces=True)["g_zt"][ntm // 2:, nz // 2]
        ratios[fm] = (dt_.max() - dt_.min()) / (d0.max() - d0.min())
    below, above = ratios[0.15e9], ratios[2.5e9]
    g_b = bool(below > 0.85 and above < 0.5)
    ok = ok and g_b
    print("[tr] GATE B: SCH transport rolls off current modulation -- below pole ({:.2f} GHz) ratio "
          "{:.2f} (~1), above ratio {:.2f} (<<1) -> {}".format(f_pole / 1e9, below, above,
                                                               "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: DD injection profile -> non-uniform gain ----
    Iramp = np.linspace(0.5 * I0, 1.5 * I0, nz)                        # current-crowding profile I(z)
    st = soa.amplify(np.full(2000, 1e-5), Iramp)["state"]
    g_z = m.gain_per_m_slices(st, nu0)
    monotone = bool(np.all(np.diff(g_z) > 0))
    stu = soa.amplify(np.full(2000, 1e-5), np.full(nz, I0))["state"]
    sts = soa.amplify(np.full(2000, 1e-5), I0)["state"]
    uni_ok = float(np.max(np.abs(m.gain_per_m_slices(stu, nu0) - m.gain_per_m_slices(sts, nu0)))) < 1e-9
    spread = (g_z.max() - g_z.min()) / g_z.mean()
    g_c = bool(monotone and uni_ok and spread > 1e-3)
    ok = ok and g_c
    print("[tr] GATE C: DD profile I(z) -> monotone non-uniform gain (spread {:.1e}, monotone {}); "
          "uniform == scalar {} -> {}".format(spread, monotone, uni_ok, "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: DC (steady-state) gain invariant under the transport ----
    ntl = 12000
    g_no = soa.amplify(np.full(ntl, 1e-5), I0, transport_tau_s=0.0,
                       return_traces=True)["g_zt"][-1, nz // 2]
    g_tr = soa.amplify(np.full(ntl, 1e-5), I0, transport_tau_s=tau_t,
                       return_traces=True)["g_zt"][-1, nz // 2]
    relD = abs(g_tr - g_no) / abs(g_no)
    g_d = bool(relD < 1e-6)
    ok = ok and g_d
    print("[tr] GATE D: transport leaves DC gain unchanged ({:.0f} vs {:.0f} /m, rel {:.1e}) -> "
          "{}".format(g_tr, g_no, relD, "PASS" if g_d else "FAIL"), flush=True)

    g_e = bool(g_a and not np.any(np.isnan(base)) and np.isfinite(g_z).all())
    ok = ok and g_e
    print("[tr] GATE E: passivity -> {}".format("PASS" if g_e else "FAIL"), flush=True)

    print("[tr] *** QD-SOA CARRIER TRANSPORT: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
