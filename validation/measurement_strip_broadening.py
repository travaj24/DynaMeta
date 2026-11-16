"""Measurement averaging on a real solver: how much can the measurement cone and strip-width scatter
broaden the dip of a metal-strip grating on a thin spacer over a mirror?  PART 1 of 4: the CONE.

The cell (2-D TM FEM, optics/fem_2d_tm.py, order 4, the validated default): Au strips (Drude: eps_inf
9.84, hbar wp 9.01 eV, hbar gamma 0.072 eV) 200 nm wide and 40 nm tall, period 400 nm, on a 10 nm
spacer (n 1.45) over a 100 nm Au mirror on glass (n 1.45); air above; E across the strips (TM). Its
plane-wave reflectance has a gap-plasmon dip near 1574 nm (R 0.051, FWHM 191 nm). 36 wavelengths 1250-1950 nm.

Measurement models (dynameta.measurement):
  * filled objective pupil NA 0.1 / 0.3 / 0.5, power detection (incoherent source) and coherent
    mode-matched amplitude detection. The 2-D TM FEM solves in-plane TM only, so the cone uses the
    labelled in-plane-only approximation (same k_x, k_y dropped; error O(NA^2)), folded by the
    x -> -x mirror symmetry of the cell.                                       [this script]
  * strip-width scatter, normal distribution with std 2 / 5 / 10 nm (Gauss-Hermite, 9 nodes; 11 at
    10 nm), small-aperture specular detection |<r>|^2 and full collection <|r|^2>.
                                       [measurement_strip_scatter / _scatter_wide / _scatter_conv]
  * a Gaussian spectrometer of 10 / 30 nm FWHM on the plane-wave spectrum (no extra solves).
                                                                               [this script]

For every case the table gives the dip position (analysis.resonance_dip), its shift from the plane
wave, the reflectance at the dip, the linewidth (FWHM of analysis.fano_fit) and, for scatter, the
diffuse fraction at the dip.

Gates (all four scripts together; tolerances as originally written)
  A  NA = 0 through the measurement layer equals the direct plane-wave solve (|dR| < 1e-14)  [here]
  B  cone quadrature converged: NA 0.5 with n_kx 9 vs 11 agree to 2e-3 in R at every wavelength [here]
  C  scatter quadrature converged: std 10 nm with 11 vs 13 Hermite nodes agree to 2e-3 in R
                                                                   [measurement_strip_scatter_conv]
  D  identities at every wavelength: |<r>|^2 <= <|r|^2> (diffuse >= 0) and the coherent amplitude
     power <= the power-detected signal (Cauchy-Schwarz with positive weights)  [every script, over
     the spectra that script computes; their union is every spectrum of the unsplit script]
  E  with full collection the FWHM and R_min grow monotonically with the width std, and the
     specular (coherent) dip is never shallower than the full-collection one. The chain
     plane wave < 2 nm < 5 nm < 10 nm is checked as its consecutive links: E1 (plane wave, 2, 5 nm)
     in measurement_strip_scatter, E2 (5 -> 10 nm) in measurement_strip_scatter_wide; together they
     are exactly the original assertion
  F  independent check of the scatter broadening: Voigt width from the plane-wave FWHM and the
     dip sensitivity d(lambda)/dw (two extra plane-wave spectra), within 5 % at std 2 and 5 nm
                                                                        [measurement_strip_scatter]
  (No gate on the NA rows: this gap-plasmon dip is a localized mode, nearly independent of angle,
  so the cone barely moves it -- that is the measured result, not a prediction to gate.)

WHY FOUR SCRIPTS. Unsplit, this was 2379 plane-wave FEM solves: 1977-2487 s on the 2-thread dev box
and a TIMEOUT at run_all's 2700 s per-script cap in a shard run. Every solve re-meshes (buffers and
PML scale with the wavelength), so no solve can be shared across wavelengths. What CAN be shared is
an IDENTICAL solve: the normal-incidence, nominal-width spectrum is also the k_x = 0 node of every
cone and the centre node of every odd Gauss-Hermite width rule. StripModel(cache=True) memoizes
r(lambda, theta, width) within a process (the solves are deterministic: a repeat is bitwise equal),
which removes those repeats without changing any number. The rest is split by gate group, each gate
unchanged; the width cases needed by two groups are recomputed in both (same solves, same values).

  script                               solves   measured wall, dev box, 2 threads
  measurement_strip_broadening (cone)     651   507 s
  measurement_strip_scatter (2, 5 nm)     684   488 s
  measurement_strip_scatter_wide (5,10)   684   498 s
  measurement_strip_scatter_conv (10)     828   631 s
(0.71-0.78 s per solve on a moderately loaded box; at 1.04 s per solve the slowest part would take
~870 s, still under 1/3 of the cap.)

Run: python -m validation.measurement_strip_broadening
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation._harness import Harness  # noqa: E402

HBAR_EV = 6.582119569e-16
C0 = 299792458.0
P, W, T_STRIP, T_GAP, T_MIRROR = 400e-9, 200e-9, 40e-9, 10e-9, 100e-9
LAMS = np.linspace(1250e-9, 1950e-9, 36)


def eps_au(lam):
    w = 2 * math.pi * C0 / lam * HBAR_EV
    return 9.84 - 9.01 ** 2 / (w * w + 1j * 0.072 * w)


class StripModel:
    """The measurement-layer model: r for (theta, 'p') and a strip width. Counts its solves.

    cache=True memoizes r per (lambda, theta, width) inside this process. A cached value is the
    value a repeat solve would return (the FEM solve is deterministic; checked bitwise), so the
    cache only removes repeated identical solves. direct() always solves (gate A uses it, so the
    measurement-layer result is compared against a FRESH solve, never against its own cache)."""

    def __init__(self, cache=False):
        self.n_solves = 0
        self.n_cache_hits = 0
        self._cache = {} if cache else None

    def _solve_one(self, lam, theta_deg, width_m):
        from dynameta.optics.fem_2d_tm import TMBlock, TMLayer, solve_fem_2d_tm
        e = eps_au(lam)
        layers = [TMLayer("mirror", T_MIRROR, e), TMLayer("spacer", T_GAP, 1.45 ** 2),
                  TMLayer("grating", T_STRIP, 1.0,
                          (TMBlock(P / 2 - width_m / 2, P / 2 + width_m / 2, e, "strip"),))]
        res = solve_fem_2d_tm(P, layers, lambda_m=float(lam), theta_deg=float(theta_deg),
                              n_super=1.0, n_sub=1.45, polarization="p", order=4, num_threads=2)
        self.n_solves += 1
        return res.r

    def direct(self, wavelengths_m, theta_deg=0.0, width_m=W):
        """Always-fresh solves (bypass the cache)."""
        return np.asarray([self._solve_one(lam, theta_deg, width_m) for lam in wavelengths_m])

    def __call__(self, wavelengths_m, theta_deg=0.0, phi_deg=0.0, pol="p", width_m=W):
        if pol != "p" or abs(phi_deg) > 1e-12:
            raise NotImplementedError("the 2-D TM FEM solves in-plane TM ('p') only")
        out = []
        for lam in wavelengths_m:
            if self._cache is None:
                out.append(self._solve_one(lam, theta_deg, width_m))
                continue
            key = (float(lam), float(theta_deg), float(width_m))
            r = self._cache.get(key)
            if r is None:
                r = self._solve_one(lam, theta_deg, width_m)
                self._cache[key] = r
            else:
                self.n_cache_hits += 1
            out.append(r)
        return np.asarray(out)


def dip_metrics(lam_m, R):
    """(dip nm, R at dip, FWHM nm) from the library's resonance utilities."""
    from dynameta.analysis import fano_fit, resonance_dip
    lam_nm = np.asarray(lam_m) * 1e9
    x0, rmin = resonance_dip(lam_nm, R)
    fit = fano_fit(lam_nm, R, x_kind="wavelength")
    return x0, rmin, fit.gamma_fwhm


class Study:
    """Shared driver of the four scripts: one cached model, the computed spectra and table rows."""

    def __init__(self, tag, title):
        self.h = Harness(tag, title)
        self.tag = tag
        try:
            import ngsolve  # noqa: F401
        except ImportError:
            self.h.require(False, "ngsolve not installed")
        import dynameta
        print("[{}] dynameta from {}".format(tag, os.path.dirname(dynameta.__file__)), flush=True)
        self.model = StripModel(cache=True)
        self.t_start = time.time()
        self.rows = []
        self.spectra = {}

    def say(self, msg):
        print("[{}] {}".format(self.tag, msg), flush=True)

    def run(self, label, setup, field="signal"):
        t0 = time.time()
        plan = setup.plan(LAMS)
        self.say("{}: {}".format(label, plan.describe()))
        n0, h0 = self.model.n_solves, self.model.n_cache_hits
        res = setup.run(self.model, LAMS)
        self.spectra[label] = res
        y = getattr(res, field)
        x0, rmin, fw = dip_metrics(LAMS, y)
        self.rows.append((label, x0, rmin, fw, res))
        self.say("  dip {:.1f} nm  R_min {:.4f}  FWHM {:.1f} nm  ({:.0f} s; {} solves, {} identical "
                 "solves reused)".format(x0, rmin, fw, time.time() - t0, self.model.n_solves - n0,
                                         self.model.n_cache_hits - h0))
        return res

    def plane_wave(self):
        from dynameta.measurement import MeasurementSetup
        return self.run("plane wave", MeasurementSetup())

    def width_case(self, sd, convergence):
        """One width-scatter case, exactly as the unsplit script built it. convergence=False runs
        the same primary rule without the refinement re-run (its primary result is identical: the
        refinement is a separate evaluation that only fills convergence_delta)."""
        from dynameta.measurement import MeasurementSetup, parameter_quadrature
        # 9 Hermite nodes; at 10 nm the centre spread (d(lambda)/dw = 6.8 nm/nm: ~70 nm) rivals the
        # linewidth, so that case runs 11 nodes with a 13-node check (11 vs 13: 3.2e-4 in R, gate C)
        q = parameter_quadrature("width_m", "normal", mean=W, std=sd,
                                 n_nodes=11 if sd == 10e-9 else 9)
        kw = dict(convergence_nodes=13, convergence_tol=2e-3) if (sd == 10e-9 and convergence) else {}
        res = self.run("width std {:.0f} nm, all".format(sd * 1e9),
                       MeasurementSetup(parameters=q, scatter_collection="all", **kw))
        x0, rmin, fw = dip_metrics(LAMS, res.specular)
        self.rows.append(("width std {:.0f} nm, specular".format(sd * 1e9), x0, rmin, fw, res))
        self.say("  (specular collection, same solves) dip {:.1f} nm  R_min {:.4f}  FWHM {:.1f} nm"
                 .format(x0, rmin, fw))
        return res

    def byl(self):
        return {r[0]: r for r in self.rows}

    def gate_d(self):
        def gate():
            worst_diff, worst_cs = 0.0, 0.0
            for res in self.spectra.values():
                worst_diff = min(worst_diff, float(np.min(res.diffuse)))
                worst_cs = max(worst_cs, float(np.max(np.abs(res.amplitude) ** 2 - res.incoherent)))
            return (worst_diff > -1e-14 and worst_cs < 1e-14,
                    "min diffuse {:.1e}, max (|r_coh|^2 - power) {:.1e} over {}".format(
                        worst_diff, worst_cs, ", ".join(self.spectra)))
        self.h.run("D", gate)

    def gate_e(self, label, keys_all, spec_keys):
        """Gate E on the consecutive cases keys_all (full collection): FWHM and R_min strictly grow
        along keys_all; for every key in spec_keys the specular dip is never shallower."""
        byl = self.byl()

        def gate():
            fw_sd = [byl[k][3] for k in keys_all]
            rm_sd = [byl[k][2] for k in keys_all]
            spec_le = all(byl[k.replace("all", "specular")][2] <= byl[k][2] + 1e-12 for k in spec_keys)
            ok = all(np.diff(fw_sd) > 0) and all(np.diff(rm_sd) > 0) and spec_le
            return ok, ("full collection ({}): FWHM {} and R_min {} grow with width std; specular dip "
                        "never shallower than full collection: {}".format(
                            " < ".join(k.replace(", all", "") for k in keys_all),
                            ["{:.1f}".format(v) for v in fw_sd], ["{:.4f}".format(v) for v in rm_sd],
                            spec_le))
        self.h.run(label, gate)

    def finish(self):
        byl = self.byl()
        x_pw = byl["plane wave"][1]
        self.say("")
        self.say("RESULT TABLE (dip from analysis.resonance_dip; FWHM from analysis.fano_fit)")
        self.say("{:<28s} {:>9s} {:>9s} {:>8s} {:>9s} {:>10s}".format(
            "case", "dip [nm]", "shift", "R_min", "FWHM", "diffuse"))
        for label, x0, rmin, fw, res in self.rows:
            dif = ""
            if res is not None and "width" in label:
                i = int(np.argmin(np.abs(LAMS * 1e9 - x0)))
                dif = "{:.4f}".format(res.diffuse[i])
            self.say("{:<28s} {:9.1f} {:+9.1f} {:8.4f} {:9.1f} {:>10s}".format(
                label, x0, x0 - x_pw, rmin, fw, dif))
        self.say("{} plane-wave solves ({} identical solves reused) in {:.0f} s".format(
            self.model.n_solves, self.model.n_cache_hits, time.time() - self.t_start))
        return self.h.summary()


def main():
    from dynameta.measurement import InstrumentFunction, MeasurementSetup, pupil_quadrature
    s = Study("msb", "measurement averaging on the strip-on-mirror dip (2-D TM FEM): the cone")
    h, model = s.h, s.model

    pw = s.plane_wave()
    R_pw = pw.signal.copy()

    def gate_a():
        direct = np.abs(model.direct(LAMS[:3])) ** 2           # fresh solves, never the cache
        d = float(np.max(np.abs(direct - R_pw[:3])))
        return d < 1e-14, "|dR| = {:.1e}".format(d)
    h.run("A", gate_a)

    for na in (0.1, 0.3, 0.5):
        p = pupil_quadrature(na, mode="in_plane", n_kx=9, fold_symmetric=True)
        res = s.run("NA {:.1f} power".format(na), MeasurementSetup(pupil=p, period_m=P))
        x0, rmin, fw = dip_metrics(LAMS, np.abs(res.amplitude) ** 2)
        s.rows.append(("NA {:.1f} amplitude".format(na), x0, rmin, fw, res))
        s.say("  (amplitude detection, same solves) dip {:.1f} nm  R_min {:.4f}  FWHM {:.1f} nm"
              .format(x0, rmin, fw))

    def gate_b():
        p11 = pupil_quadrature(0.5, mode="in_plane", n_kx=11, fold_symmetric=True)
        r11 = MeasurementSetup(pupil=p11).run(model, LAMS)
        d = float(np.max(np.abs(r11.signal - s.spectra["NA 0.5 power"].signal)))
        return d < 2e-3, "max |R(9) - R(11)| = {:.2e}".format(d)
    h.run("B", gate_b)

    for fwhm in (10e-9, 30e-9):
        y = InstrumentFunction("gaussian", fwhm_m=fwhm).apply(LAMS, R_pw)
        x0, rmin, fw = dip_metrics(LAMS, y)
        s.rows.append(("instrument {:.0f} nm".format(fwhm * 1e9), x0, rmin, fw, None))

    s.gate_d()
    return s.finish()


if __name__ == "__main__":
    raise SystemExit(main())
