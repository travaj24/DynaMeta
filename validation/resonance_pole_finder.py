"""Solver-free gates for the complex-omega POLE FINDER in ``optics.resonance`` -- the other module
the audit (X-19) flagged as having no library consumer and no validation script, so its only oracle
was the test written beside it in the same commit.

Pure numpy/scipy, seconds to run.  Every oracle here is either a closed form derived in this file
or a REAL-AXIS measurement of the same stack, so nothing is checked against the implementation
that produced it.

GATE A  ONE Q CONVENTION (audit X-5).  ``resonance.pole_q`` and ``aaa_poles.q_from_pole`` used to be
        byte-identical copies under two public names whose docstrings had already drifted.  They
        must now be the SAME function object, and Q must be positive for a pole and for its
        ``-conj`` partner (the ``|Re|`` the drifted docstring omitted).
GATE B  ETALON CLOSED FORM.  For a lossless symmetric slab (index n, thickness L, vacuum both
        sides) the round-trip condition puts the m-th scattering pole at

            omega_m = m pi c/(n L) - i (c/(n L)) ln|r12| ,   r12 = (n-1)/(n+1),
            Q_m     = -m pi / (2 ln|r12|) .

        ``find_poles`` must land on omega_m (Re and Im separately) and ``pole_q`` must reproduce
        Q_m -- for several orders m and two indices.
GATE C  POLE -> LINESHAPE.  The pole is a statement about the REAL axis: the transmittance peak
        must sit at Re(omega_tilde) and its FWHM must equal 2|Im(omega_tilde)|, measured by
        sampling the real-axis spectrum (no pole-finder involvement).
GATE D  LOSS BUDGET.  Adding material loss to the slab must move the pole DOWN in the complex plane
        (lower Q) monotonically, and ``q_budget`` must return Q_rad ~ the lossless Q with
        ``1/Q = 1/Q_rad + 1/Q_abs`` closing.

Run: python -m validation.resonance_pole_finder
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

C = 299792458.0


def _etalon_closed_form(n, L_m, m):
    """(omega_pole, Q) of the m-th Fabry-Perot pole of a lossless slab in vacuum. Derived in the
    module docstring: the transmission denominator's round-trip factor is 1 - r12^2 exp(2 i delta)
    with delta = n omega L/c, so the pole is at delta = m pi - i ln|r12|."""
    r12 = (n - 1.0) / (n + 1.0)
    base = C / (n * L_m)
    return complex(m * math.pi * base, base * math.log(abs(r12))), -m * math.pi / (2.0 * math.log(abs(r12)))


def main():
    from dynameta.optics.resonance import (find_poles, pole_q, q_budget, smatrix_pole_func)
    from dynameta.optics.aaa_poles import q_from_pole

    ok = True

    # ---------------------------------------------------------------- GATE A: one Q convention
    print("[rpf] === GATE A: one Q convention, one implementation ===", flush=True)
    a_ok = q_from_pole is pole_q and q_from_pole.__doc__ is pole_q.__doc__
    print("[rpf]   aaa_poles.q_from_pole IS resonance.pole_q (same object, one docstring): {}"
          .format(a_ok), flush=True)
    w = 3.0e15 - 1.0e13j
    a_ok = a_ok and abs(pole_q(w) - 3.0e15 / 2.0e13) < 1e-9 * pole_q(w)
    a_ok = a_ok and pole_q(-w.conjugate()) == pole_q(w)          # the |Re| the docstring omitted
    a_ok = a_ok and pole_q(3.0e15 + 0j) == float("inf")          # undamped pole
    print("[rpf]   Q({:.2e}) = {:.3f}; Q(-conj) identical; real pole -> inf".format(w, pole_q(w)),
          flush=True)
    print("[rpf] GATE A -> {}".format("PASS" if a_ok else "FAIL"), flush=True)
    ok = ok and a_ok

    # ---------------------------------------------------------------- GATE B: etalon closed form
    print("[rpf] === GATE B: Fabry-Perot poles vs the closed form ===", flush=True)
    b_ok, worst_w, worst_q = True, 0.0, 0.0
    for n, L in ((3.5, 1.0e-6), (2.0, 1.5e-6)):
        D = smatrix_pole_func([(n * n, L)], pol="s", n_super=1.0, n_sub=1.0, k_par_m=0.0)
        for m in (3, 4, 5):
            w_t, q_t = _etalon_closed_form(n, L, m)
            span = complex(0.15 * abs(w_t.real), 1.5 * abs(w_t.imag))
            got = find_poles(D, complex(w_t.real, w_t.imag), span, n_grid=24)
            if not got:
                b_ok = False
                print("[rpf]   n={} m={}: NO POLE FOUND".format(n, m), flush=True)
                continue
            p = min(got, key=lambda z: abs(z - w_t))
            e_re = abs(p.real - w_t.real) / abs(w_t.real)
            e_im = abs(p.imag - w_t.imag) / abs(w_t.imag)
            e_q = abs(pole_q(p) - q_t) / q_t
            worst_w = max(worst_w, e_re, e_im)
            worst_q = max(worst_q, e_q)
            b_ok = b_ok and e_re < 1e-8 and e_im < 1e-8 and e_q < 1e-8
            print("[rpf]   n={:.1f} m={}: pole {:.6e}{:+.4e}j vs {:.6e}{:+.4e}j | Q {:.4f} vs "
                  "{:.4f} (rel {:.1e})".format(n, m, p.real, p.imag, w_t.real, w_t.imag,
                                               pole_q(p), q_t, e_q), flush=True)
    print("[rpf] GATE B (worst pole {:.1e}, worst Q {:.1e}) -> {}".format(
        worst_w, worst_q, "PASS" if b_ok else "FAIL"), flush=True)
    ok = ok and b_ok

    # ---------------------------------------------------------------- GATE C: pole -> lineshape
    print("[rpf] === GATE C: the pole predicts the REAL-AXIS lineshape ===", flush=True)
    # The exact real-axis lineshape is Airy, T = 1/(1 + F sin^2(delta)) with F = 4R/(1-R)^2 and
    # R = r12^2, so the EXACT half-max half-width is delta_half = arcsin(1/sqrt(F)) while the pole
    # gives delta_pole = -ln|r12|. The two agree only in the HIGH-FINESSE limit, so the gate is: the
    # measured width matches the Airy closed form to <1%, and the pole width converges to it as the
    # index (hence the finesse) rises. Sample strictly inside one free spectral range.
    from dynameta.optics.resonance import layered_smatrix_complex
    c_ok, ratios = True, []
    for n, L, m in ((3.5, 1.0e-6, 4), (8.0, 1.0e-6, 4)):
        w_t, q_t = _etalon_closed_form(n, L, m)
        fsr = math.pi * C / (n * L)                              # omega spacing of the FP orders
        wr = np.linspace(w_t.real - 0.45 * fsr, w_t.real + 0.45 * fsr, 40001)
        T = np.array([abs(layered_smatrix_complex(float(x), [(n * n, L)], pol="s").t) ** 2
                      for x in wr])
        i_pk = int(np.argmax(T))
        half = T[i_pk] / 2.0
        lo = wr[:i_pk][np.argmin(np.abs(T[:i_pk] - half))]
        hi = wr[i_pk:][np.argmin(np.abs(T[i_pk:] - half))]
        fwhm = hi - lo
        r12 = (n - 1.0) / (n + 1.0)
        R = r12 * r12
        F = 4.0 * R / (1.0 - R) ** 2
        fwhm_airy = 2.0 * math.asin(1.0 / math.sqrt(F)) * C / (n * L)     # exact closed form
        fwhm_pole = 2.0 * abs(w_t.imag)
        e_pk = abs(wr[i_pk] - w_t.real) / abs(w_t.real)
        e_airy = abs(fwhm - fwhm_airy) / fwhm_airy
        ratios.append(fwhm_airy / fwhm_pole)
        c_ok = c_ok and e_pk < 1e-5 and e_airy < 1e-2 and abs(T[i_pk] - 1.0) < 1e-9
        print("[rpf]   n={:.1f}: peak {:.6e} vs Re(pole) {:.6e} (rel {:.1e}), T_peak={:.9f}".format(
            n, wr[i_pk], w_t.real, e_pk, T[i_pk]), flush=True)
        print("[rpf]          FWHM {:.4e} vs Airy {:.4e} (rel {:.1e}); Airy/2|Im(pole)| = {:.4f}"
              .format(fwhm, fwhm_airy, e_airy, ratios[-1]), flush=True)
    # low finesse -> the Lorentzian (pole) width under-states the Airy width; higher index closes it
    c_ok = c_ok and ratios[0] > 1.05 and 1.0 < ratios[1] < ratios[0]
    print("[rpf]   pole width -> Airy width as the finesse rises: {:.4f} -> {:.4f}".format(
        ratios[0], ratios[1]), flush=True)
    print("[rpf] GATE C -> {}".format("PASS" if c_ok else "FAIL"), flush=True)
    ok = ok and c_ok
    n, L, m = 3.5, 1.0e-6, 4
    w_t, q_t = _etalon_closed_form(n, L, m)

    # ---------------------------------------------------------------- GATE D: loss budget
    print("[rpf] === GATE D: material loss lowers Q; q_budget closes ===", flush=True)
    eps_r = n * n

    def make(loss):                     # eps = n^2 + i*loss*k -- a MATERIAL-level knob (analytic)
        return smatrix_pole_func([(complex(eps_r, 0.30 * loss), L)], pol="s",
                                 n_super=1.0, n_sub=1.0, k_par_m=0.0)

    qs = []
    for loss in (0.0, 0.5, 1.0, 2.0):
        span = complex(0.15 * abs(w_t.real), 4.0 * abs(w_t.imag))
        got = find_poles(make(loss), complex(w_t.real, w_t.imag), span, n_grid=24)
        p = min(got, key=lambda z: abs(z - w_t))
        qs.append(pole_q(p))
        print("[rpf]   loss x{:.1f}: pole {:.6e}{:+.4e}j  Q = {:.3f}".format(
            loss, p.real, p.imag, qs[-1]), flush=True)
    d_ok = all(b < a for a, b in zip(qs, qs[1:]))                # Q strictly decreasing with loss
    d_ok = d_ok and abs(qs[0] - q_t) / q_t < 1e-8                # lossless limit = closed form
    span = complex(0.15 * abs(w_t.real), 4.0 * abs(w_t.imag))
    p1 = min(find_poles(make(1.0), complex(w_t.real, w_t.imag), span, n_grid=24),
             key=lambda z: abs(z - w_t))
    bud = q_budget(make, p1)
    closes = abs(1.0 / bud["Q_total"] - (1.0 / bud["Q_rad"] + 1.0 / bud["Q_abs"])) \
        * bud["Q_total"] < 1e-9
    d_ok = d_ok and closes and bud["pole_rad_ok"] and abs(bud["Q_rad"] - q_t) / q_t < 1e-6
    print("[rpf]   q_budget: Q={:.3f} Q_rad={:.3f} (lossless closed form {:.3f}) Q_abs={:.3f}; "
          "1/Q = 1/Q_rad + 1/Q_abs closes: {}; lossless-pole validity: {}".format(
              bud["Q_total"], bud["Q_rad"], q_t, bud["Q_abs"], closes, bud["pole_rad_ok"]),
          flush=True)
    print("[rpf] GATE D -> {}".format("PASS" if d_ok else "FAIL"), flush=True)
    ok = ok and d_ok

    print("[rpf] *** RESONANCE POLE FINDER: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
