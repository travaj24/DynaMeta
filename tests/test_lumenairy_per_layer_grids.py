"""The PMM `layer_grids='per-layer'` seam (lumenairy >= 5.32), bridged 2026-08-31.

WHAT THIS OPTION IS. The 1-D PMM backend is DynaMeta's convergence REFEREE: a subsectional
spectral element with no Fourier-factorization accuracy floor, used to referee the RCWA path's
truncation settings. Its cost driver is the SHARED union grid -- one element grid carrying
every layer's walls, so each layer pays for the walls of all the others. 'per-layer' gives each
layer only its own walls and couples them with a non-conforming mortar.

WHY THESE GATES EXIST. The option is a genuine speed/closure TRADE, not a free win, and the
trade is the whole reason it ships opt-in rather than as the default. So the gates pin BOTH
sides of it: that the two paths agree on the physics, and that the mortar's cost is the loss of
exact energy closure. A gate that only checked the speed-up would be advertising, not evidence.

Run: python -m pytest tests/test_lumenairy_per_layer_grids.py -q
"""
import numpy as np
import pytest

from dynameta.geometry import Design, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.geometry.stack import Inclusion
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

needs_lum = pytest.importorskip("lumenairy")

LAM = 1.31e-6
PERIOD = 600e-9


def _stack_design(theta_deg=30.0):
    """Four lamellar layers whose walls sit at DIFFERENT x -- the case the union grid is
    expensive for (every layer carries all four wall sets) and the per-layer grid is not."""
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(6.25 + 0j)))
    reg.add(Material("glass", ConstantOptical(2.25 + 0j)))
    layers = []
    for k, (x0, w) in enumerate([(0.10, 0.30), (0.22, 0.26), (0.37, 0.21), (0.05, 0.44)]):
        line = Rectangle((x0 + w / 2) * PERIOD, PERIOD / 2, w * PERIOD, PERIOD)
        layers.append(Layer("L{}".format(k), 180e-9, "air",
                            inclusions=[Inclusion(line, "hi")]))
    return Design(name="pl", unit_cell=UnitCell.square(PERIOD),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization="y", incidence_angle_deg=theta_deg))


def _solve(layer_grids, degree):
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver
    solver = make_lumenairy_pmm_solver(degree=degree, n_orders=21, absorption=True,
                                       layer_grids=layer_grids)
    return solver(_stack_design(), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)


def test_default_is_shared_and_unchanged():
    """The default must be the pre-5.32 behaviour, byte-for-byte: this option may not move a
    single existing number. Explicit 'shared' and the default are the SAME call."""
    a = _solve("shared", 8)
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver
    b = make_lumenairy_pmm_solver(degree=8, n_orders=21, absorption=True)(
        _stack_design(), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    assert a.R == b.R and a.T == b.T
    assert a.r == b.r


def test_per_layer_agrees_on_the_physics_and_converges_toward_shared():
    """The two grids are different discretisations of the SAME problem, so they must agree to
    a discretisation error that SHRINKS with degree -- the honest form of 'same answer'. A
    fixed tolerance would hide a per-layer path that was simply wrong at every degree."""
    d8_s, d8_p = _solve("shared", 8), _solve("per-layer", 8)
    d16_s, d16_p = _solve("shared", 16), _solve("per-layer", 16)
    gap8 = abs(float(d8_p.R) - float(d8_s.R))
    gap16 = abs(float(d16_p.R) - float(d16_s.R))
    assert gap8 < 1e-4, gap8                    # same physics already at low degree
    assert gap16 < gap8, (gap8, gap16)          # and CONVERGING, which is the real claim


def test_the_trade_is_real_exact_closure_is_what_is_given_up():
    """The cost side, pinned so nobody 'optimises' the default over to per-layer without
    meeting it. The union grid closes energy to ~1e-13; the non-conforming mortar does not.

    MEASURE |R + T - 1| ON A LOSSLESS STACK, NOT |R + T + A - 1|. The first draft of this gate
    used the latter and both paths returned EXACTLY 0.0 -- because A is reported as 1 - R - T
    (the R+T+A tautology the v0.9.0 audit already named), so that form is arithmetically
    incapable of failing. Every material in this fixture is real-valued, so the stack is
    lossless by construction and |R + T - 1| is the honest closure residual."""
    s = _solve("shared", 8)
    p = _solve("per-layer", 8)
    close_s = abs(float(s.R) + float(s.T) - 1.0)
    close_p = abs(float(p.R) + float(p.T) - 1.0)
    assert close_s < 1e-9, close_s              # the referee-grade closure we are protecting
    assert close_p < 1e-4, close_p              # mortar: looser, but still physical
    assert close_p > close_s, (close_s, close_p)   # ... and this is the trade, not noise


def test_bad_value_is_refused_and_the_cache_key_separates_the_two():
    """Two solvers differing only in layer_grids are DIFFERENT solves (different grids,
    different closure). If they shared a cache fingerprint, a cached 'shared' result could be
    served for a 'per-layer' request -- silently, and with the wrong closure."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm_solver
    with pytest.raises(ValueError, match="layer_grids"):
        make_lumenairy_pmm_solver(layer_grids="union")(
            _stack_design(), None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    fp_s = make_lumenairy_pmm_solver(layer_grids="shared").cache_fingerprint
    fp_p = make_lumenairy_pmm_solver(layer_grids="per-layer").cache_fingerprint
    assert fp_s != fp_p, (fp_s, fp_p)
    assert "layer_grids" in fp_s


def test_absorption_survives_the_mortar():
    """layer_absorption() is the per-region budget consumers read; the audit verified it works
    on the per-layer path, so pin it -- a mortar that silently dropped per-region absorption
    would still pass every R/T gate above."""
    p = _solve("per-layer", 8)
    assert p.per_region_absorption is not None
    assert set(p.per_region_absorption) == {"L0", "L1", "L2", "L3"}
    assert np.isfinite(list(p.per_region_absorption.values())).all()
