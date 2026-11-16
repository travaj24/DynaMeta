"""2-D periodic TM FEM (optics/fem_2d_tm.py) on the metal-strip cell: convergence table + gates.

The cell: the generic strip cell of tests/_strip_cell_fixture -- metal strips (eps -60+4i), 150 nm
wide, period 400 nm, 40 nm tall, on 2 nm oxA (n 1.70) / 6 nm oxB (n 2.00) / 6 nm Drude film / 6 nm
oxB / 2 nm oxA over an 80 nm mirror metal (eps -120+20i) on glass (n 1.45), air above, TM (E across
the strips), normal incidence, 1200 nm. Reference: the polynomial-modal-method value
r_ref = 0.3352997+0.0171262i (lumenairy 5.50.2 PMMStack, elements-per-region 8, degree 40; its last
refinement step, epr 6 -> 8, moved r by 6.3e-5, so it is good to ~1e-4; Richardson in epr gives
0.335269+0.017197i, 5e-6 from the FEM's converged value 0.335270+0.017192i).

Gates
  A  planar stack (the same cell without strips) vs an independent TM transfer matrix: |dr| < 1e-7
  B  every row of the (order x corner-levels) table closes R + T + A = 1 to 1e-6
  C  the order ladder at 10 corner levels converges monotonically toward its finest member, down to
     a 1e-6 self-consistency floor below which every row must stay (measured 4.2e-4 / 2.7e-5 /
     2.0e-7 / 9.6e-8 / 2.1e-7: orders 6 / 7 are at the floor), and its first rung lies far above the
     floor (> 100 x FLOOR; measured 4.2e-4), so the ladder demonstrably converges rather than
     starting below the floor (an order-insensitive or stalled solver passes the monotonicity alone)
  D  the finest FEM r agrees with r_ref to 1e-4; the default setting (order 4, 6 levels) to 2e-4

Run: python -m validation.fem_2d_tm_strip
"""
import cmath
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation._harness import Harness  # noqa: E402


LAM, P, W = 1200e-9, 400e-9, 150e-9
R_REF = 0.3352997 + 0.0171262j


def _sq(z):
    w = cmath.sqrt(complex(z))
    return -w if (w.imag < 0 or (w.imag == 0 and w.real < 0)) else w


def _tm_tmm(layers, lam, n1, n2):
    """Normal-incidence TM transfer matrix (isotropic layers, bottom first) -> r_H at the top."""
    k0 = 2 * math.pi / lam
    q1, q2 = _sq(n1 ** 2 * k0 ** 2) / n1 ** 2, _sq(n2 ** 2 * k0 ** 2) / n2 ** 2
    v = np.array([1.0, -1j * q2], dtype=complex)
    for e, d in layers:
        kz = _sq(e * k0 ** 2)
        q = kz / e
        c, s = cmath.cos(kz * d), cmath.sin(kz * d)
        v = np.array([[c, s / q], [-q * s, c]]) @ v
    t = 2j * q1 / (1j * q1 * v[0] - v[1])
    return t * v[0] - 1.0


def main():
    h = Harness("f2d", "2-D TM FEM: metal-strip cell vs modal reference")
    try:
        import ngsolve  # noqa: F401
    except ImportError:
        h.require(False, "ngsolve not installed")
    from dynameta.materials import M_E, DrudeOptical
    from dynameta.optics.fem_2d_tm import TMBlock, TMLayer, TMMeshSpec, solve_fem_2d_tm
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:                                   # memory column optional
        proc = None

    e_film = complex(DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14).eps(LAM, n_m3=3.0e26))
    film = [("mirror", 80e-9, -120 + 20j), ("lo_oxa", 2e-9, 1.70 ** 2), ("lo_oxb", 6e-9, 2.00 ** 2),
            ("film", 6e-9, e_film), ("up_oxb", 6e-9, 2.00 ** 2), ("up_oxa", 2e-9, 1.70 ** 2)]
    planar = [TMLayer(n, t, e) for n, t, e in film]
    strip = planar + [TMLayer("grating", 40e-9, 1.0 + 0j,
                              (TMBlock(P / 2 - W / 2, P / 2 + W / 2, -60 + 4j, "strip"),))]

    def solve(layers, order, levels, h_scale=1.0):
        return solve_fem_2d_tm(P, layers, lambda_m=LAM, n_super=1.0, n_sub=1.45, order=order,
                               mesh_spec=TMMeshSpec(corner_levels=levels, h_scale=h_scale),
                               return_solution=True, num_threads=2)

    # A: planar stack
    def gate_a():
        r_fem = solve(planar, 4, 6).result.r
        r_an = -_tm_tmm([(complex(e), t) for _n, t, e in film], LAM, 1.0, 1.45)
        d = abs(r_fem - r_an)
        return d < 1e-7, "|dr| = {:.2e}".format(d)
    h.run("A (planar vs transfer matrix)", gate_a)

    # convergence table
    print("[f2d] order levels h_scale |   r                      |r-r_ref|   R+T+A-1    ndof    "
          "elems   time[s]  rss[MB]", flush=True)
    rows = {}
    for order, levels, hs in ((3, 4, 1.0), (3, 6, 1.0), (4, 4, 1.0), (4, 6, 1.0), (4, 6, 0.5),
                              (4, 8, 1.0), (5, 6, 1.0), (5, 8, 1.0), (6, 6, 1.0), (6, 10, 1.0),
                              (7, 10, 1.0), (8, 10, 1.0)):
        t0 = time.perf_counter()
        sol = solve(strip, order, levels, hs)
        dt = time.perf_counter() - t0
        res = sol.result
        rows[(order, levels, hs)] = sol
        rss = proc.memory_info().rss / 1e6 if proc is not None else float("nan")
        print("[f2d] {:5d} {:6d} {:7.2f} | {:.7f}{:+.7f}i  {:.2e}  {:+.1e}  {:7d}  {:6d}  {:7.2f}  {:7.0f}"
              .format(order, levels, hs, res.r.real, res.r.imag, abs(res.r - R_REF),
                      res.R + res.T + res.A_independent - 1.0, sol.ndof, sol.n_elements, dt, rss),
              flush=True)

    worst = max(abs(s.energy_residual) for s in rows.values())
    h.gate("B (energy R + T + A = 1 on every row)", worst < 1e-6, "worst {:.1e}".format(worst))
    ladder = [rows[(o, 10, 1.0)] if (o, 10, 1.0) in rows else None for o in (6, 7, 8)]
    lad = [rows[(3, 6, 1.0)], rows[(4, 6, 1.0)], rows[(5, 8, 1.0)]] + [s for s in ladder if s]
    fin = lad[-1].result.r
    d = [abs(s.result.r - fin) for s in lad[:-1]]
    # monotone down to the self-consistency floor: above FLOOR (10x the ~1e-7 spread of orders 6-8
    # at 10 levels) every step decreases; from the first row below FLOOR on, every row stays below it.
    # On the generic cell orders 6 / 7 reach the floor (9.6e-8 / 2.1e-7), where order is noise.
    FLOOR = 1e-6
    k = next((i for i, x in enumerate(d) if x < FLOOR), len(d))
    above = d[:k + 1] if k < len(d) else d
    # the first rung (order 3) must lie well above the floor: a ladder that starts at or below it shows
    # no convergence at all (measured 4.2e-4 = 420 x FLOOR)
    ok_c = (d[0] > 100.0 * FLOOR and all(a > b for a, b in zip(above, above[1:]))
            and all(x < FLOOR for x in d[k:]))
    h.gate("C (order ladder converges monotonically down to the {:.0e} floor)".format(FLOOR), ok_c,
           "|r - r_finest| = " + ", ".join("{:.1e}".format(x) for x in d))
    d_fin = abs(fin - R_REF)
    d_def = abs(rows[(4, 6, 1.0)].result.r - R_REF)
    h.gate("D (finest vs r_ref < 1e-4; default vs r_ref < 2e-4)", d_fin < 1e-4 and d_def < 2e-4,
           "finest {:.2e}, default {:.2e}".format(d_fin, d_def))
    return h.summary()


if __name__ == "__main__":
    raise SystemExit(main())
