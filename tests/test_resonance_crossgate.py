"""THE Phase-1 cross-instrument gate: the same physical resonance measured three independent
ways must agree -- (1) the complex-omega pole of the layered response (optics.resonance),
(2) matrix-pencil inversion of an FDTD ringdown (optics.ringdown), and (3) a
Fano/Lorentzian fit of the DRIVEN transmission spectrum (analysis.fano_fit/lorentzian_fit,
spectrum from the real-axis S-matrix evaluator, itself pinned against tmm_reference to 1e-14
in test_resonance). All three probe the SAME symmetric n=3.5 etalon; the FDTD ringdown picks
the mode it rings on (its dominant mode -- the source spectrum's skirts decide, here m=4) and
the other two instruments are aimed at THAT mode, with the Fabry-Perot closed form as the
shared analytic anchor. One Q convention (energy Q = Re(omega)/(2|Im omega|) = omega/gamma),
one mode, three routes."""

import numpy as np

from dynameta.constants import C_LIGHT
from dynameta.analysis import fano_fit, lorentzian_fit
from dynameta.optics.resonance import (find_poles, layered_smatrix_complex, pole_q,
                                       smatrix_pole_func)
from dynameta.optics.ringdown import fdtd_etalon_ringdown

N_SLAB = 3.5
L_SLAB = 1.0e-6


def _closed_form(m):
    """Exact FP pole (omega_m, Q_m) of the symmetric slab in vacuum."""
    r12 = (N_SLAB - 1.0) / (N_SLAB + 1.0)
    om = np.pi * C_LIGHT * m / (N_SLAB * L_SLAB)
    return om, -m * np.pi / (2.0 * np.log(r12))


def test_three_instruments_agree_on_one_etalon():
    # ---- instrument 2 first: FDTD ringdown (fully independent solver) picks the mode.
    # NOTE the excitation band matters: this is the ringdown module's own validated gate-6
    # configuration (1.2-1.7 um). A band centred between modes rings a skirt-excited
    # neighbour whose extracted tail is contaminated by the source turn-off transient
    # (observed: m=4 with a 5x-slow spurious decay) -- band placement is part of the
    # instrument's operating manual, documented here deliberately.
    rd = fdtd_etalon_ringdown(N_SLAB, L_SLAB, lambda_min_m=1.2e-6, lambda_max_m=1.7e-6,
                              resolution=30)
    om_rd = 2.0 * np.pi * rd.f0_Hz
    m = int(round(om_rd * N_SLAB * L_SLAB / (np.pi * C_LIGHT)))   # its etalon order
    om_cf, q_cf = _closed_form(m)
    assert m >= 3
    assert abs(om_rd / om_cf - 1.0) < 0.03                        # FDTD grid-dispersion band
    assert abs(rd.q / q_cf - 1.0) < 0.12

    # ---- instrument 1: complex-omega pole of the SAME mode ----
    layers = [(N_SLAB ** 2 + 0.0j, L_SLAB)]
    f = smatrix_pole_func(layers)
    poles = find_poles(f, om_cf - 0.6j * om_cf / (2.0 * q_cf),
                       0.06 * om_cf + 1.5j * om_cf / (2.0 * q_cf), n_grid=48)
    assert poles, "pole finder found nothing near the closed-form mode"
    pole = min(poles, key=lambda p: abs(p.real - om_cf))
    q_pole = pole_q(pole)
    assert abs(pole.real / om_cf - 1.0) < 1e-8
    assert abs(q_pole / q_cf - 1.0) < 1e-6

    # ---- instrument 3: driven-spectrum lineshape fit (narrow window -> pole Q) ----
    om_grid = om_cf * np.linspace(0.97, 1.03, 1201)
    T = np.array([abs(layered_smatrix_complex(om, layers).t) ** 2 for om in om_grid])
    lf = lorentzian_fit(om_grid, T)
    ff = fano_fit(om_grid, T)
    assert abs(lf.x0 / om_cf - 1.0) < 1e-3
    assert abs(lf.Q / q_cf - 1.0) < 0.03
    assert abs(ff.Q / q_cf - 1.0) < 0.03

    # ---- the contract: all three instruments land on the same (omega, Q) ----
    assert abs(q_pole / lf.Q - 1.0) < 0.04
    assert abs(q_pole / rd.q - 1.0) < 0.12
