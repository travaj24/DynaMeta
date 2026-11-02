"""Lumenairy 2-D crossed-patterned PMM bridge (audit 8.1-4) vs TMM / cross-engine oracles.

GATE A (uniform == TMM, both engines): a 2-layer ASYMMETRIC lossy stack at normal +
        30 deg y/p -- R/T/r AND (since 2026-11-02) the complex transmission t match
        tmm_reference at machine level (pins the stack layer ORDER, the pol-row
        mapping, and the shared lab-basis -> Byrnes p-pol r / t conversion on BOTH the
        pure and hybrid engines).
GATE B (the 2-D REFEREE role): an ITO-like NEAR-ENZ METAL patch (eps = -3 + 1j, the
        modulator regime; harder metals converge too slowly for a validation-scale
        gate on EVERY engine -- measured during bring-up) on a dielectric spacer.
        PMM2DStackPure (no Fourier floor, wall-resolved staggered basis) is the
        reference; the RCWA bridge on the SAME Design must CONVERGE TOWARD it: |R -
        R_pure| strictly shrinking over n_orders 4 -> 8 -> 12 -> 16 and < 1.5e-3 at
        n=16 (direction-sensitive, the validation/lumenairy_pmm_bridge GATE B/E
        pattern), with the pure reference degree-stable (|R(M=6) - R(M=5)| below the
        final RCWA gap).  The hybrid engine must land within 5e-3 of pure -- three
        methods (staggered-modal, Fourier-projected-modal, FMM) on one geometry.
GATE C (slab order, audit C5-1): an asymmetric LOSSY linear-in-z graded profile --
        slice_eps_field returns SUBSTRATE-first slabs and both 2-D PMM engines build
        superstrate-first, so an unreversed translation vertically flips the profile.
        Both engines match TMM and the RCWA bridge on the same design; the FLIPPED
        profile differs by > 1e-3 (the fixture keeps its discriminating power).
GATE D (scope guards raise): conical azimuth, incidence_side='bottom', a Circle
        inclusion under an EXPLICIT pure_grid='union', non-commensurate Rectangle walls
        under pure_grid='union', a laterally-structured gridded EpsField (pure), a
        laterally-structured gridded TENSOR EpsField (pure), a Polygon inclusion on
        the pure engine (no exact image), a bogus engine name and a bogus pure_grid.
        (Circles, non-commensurate rectangles and tensors were refusals in v1; they
        are capabilities since 2026-11-02, gated in F-J below.)
GATE E (per-layer absorption, audit C3): a LOSSY ITO patch (eps -3 + 1j) on a LOSSY
        'hi' spacer (eps 4 + 0.3j).  The PURE engine (internal retention new in
        lumenairy 5.22) must close its own budget |A_independent - (1 - R - T)| <
        1e-6, be genuinely absorbing (A_independent > 1e-3), key
        per_region_absorption by DESIGN layer name, and vanish (< 1e-6) on the
        lossless twin -- and absorption=False must leave per_region_absorption=None
        (byte-identical off-switch).  Hybrid-vs-Pure cross-check: both engines close
        their OWN budget and their TOTAL A_independent agree (staggered-modal vs
        Fourier-projected internal quadrature -- different bases, so gated at the
        convergence-appropriate tolerance measured at bring-up).

GATE F (uniform TENSOR layer == Berreman 4x4, 2026-11-02): an asymmetric lossy
        IN-PLANE tensor (e_xy != e_yx) and an OUT-OF-PLANE one (tilted uniaxial
        director + asymmetric e_xz / e_zy perturbation) as a uniform tensor EpsField,
        pols x / y / p at normal + 30 deg: the bridge's R / T / r / t on BOTH engines
        match make_lumenairy_berreman_solver (an independent planar 4x4 transfer-
        matrix method), AND the bridge-built stack's FULL 2x2 reflection +
        transmission Jones matches BerremanStack's (co-pol entries are TRANSPOSE-
        BLIND by reciprocity -- the off-diagonal is where a transposed mapping shows).
        Discrimination: the transposed tensor moves the full Jones by > 1e-2; the
        x <-> y index-permuted tensor moves the co-pol r by > 1e-3.
GATE G (patterned ANISOTROPIC cell vs RCWA, 2026-11-02): a uniaxial LC background
        (n_o 1.5 / n_e 1.7, director azimuth 30 deg, tilt 0 [block-form] and 40 deg
        [out-of-plane]) as a uniform tensor EpsField on a layer with an eps-4 pillar
        -- a centred square (union grid N = 4, (4, 4, 3, 3) cell) and a circle (curved
        shapes= map with a (3, 3) background_eps).  Oracle: lumenairy RCWAStack with
        an (S, S, 3, 3) eps_tensor_cell painted directly from the shapes (DIRECT, not
        through the RCWA bridge, whose uniform-tensor branch drops inclusions -- module
        docstring of pmm2d_backend).  |R - R_pure| strictly shrinking over n_orders 6
        -> 9 -> 12 for x AND y pol, < 4e-4 at 12, and the first-order (O(1/n),
        Laurent-only tensor path) Richardson value from 9 / 12 within 2e-4; hybrid
        (square) within 1.5e-4.  Discrimination: the MUTANT must fail the same
        acceptance -- the x <-> y index-permuted tensor and (tilted director) the
        out-of-plane entries dropped each land > 2e-4 from the Richardson value.
GATE H (transmission semantics, 2026-11-02): on the uniform stack (y and p, 30 deg)
        and a non-diffracting circle pillar (normal, x): |t|^2 Re(n_sub cos_t) /
        Re(n_sup cos_i) == T (co-pol power closure; Byrnes' p-power factor uses the
        same cosines for lossless media) to 1e-9, and stack.per_order_amplitudes(
        'transmission') at order (0, 0) equals stack.jones_transmission() column-wise
        (the documented per-order route).  Discrimination: dropping the p-basis t
        factor breaks the p-pol closure by > 1e-2.
GATE I (CURVED shapes vs RCWA bracket, 2026-11-02): an eps-4 circle (r = 0.3 p) and an
        axis-aligned ellipse (0.33 p x 0.22 p) pillar in air, 200 nm, 1.31 um, 600 nm
        cell (non-diffracting in both media).  Pure (shapes=, curved map, M = 6) vs the
        RCWA bridge with the FFF (normal-vector) and Laurent formulations on a 600-px
        raster: the FFF distances strictly shrink over n_orders 9 -> 12 -> 15 and end
        < 2e-5, and Laurent (n 15) sits on the OPPOSITE side of pure (a two-sided
        bracket).  Discrimination: a 2 % larger radius moves R by > 1e-4, the ellipse
        with rx / ry swapped by > 1e-3.
GATE J (NON-COMMENSURATE rectangles, 2026-11-02): an off-centre 0.46 p x 0.37 p eps-4
        pillar (walls 0.30 / 0.76 p and 0.295 / 0.665 p -- no uniform N <= 32 fits):
        pure_grid='auto' takes 'shapes' (exact walls); vs the RCWA bridge the FFF
        distances shrink over 9 -> 12 -> 15, end < 4e-5, Laurent brackets from the
        other side; a 6 nm wider pillar moves R by > 2e-4.  'per-layer' on the same
        single layer reproduces 'shapes' to 1e-10 (identical walls); on a two-layer
        stack (x-stripe under the pillar -- segment equalization exercised) per-layer
        (M = 7) lands within 2e-4 of RCWA-FFF (n 15), and an 18 nm wider pillar lands
        > 2e-4 from it (fails the same acceptance).

Honest SKIP (exit 0 + banner) when lumenairy is not importable.

Run: python -m validation.lumenairy_pmm2d_bridge     (~5-8 min on a quiet box: the
     pure staggered eigs dominate; cases are deliberately small, see the budget note
     in each gate)
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.eps_field import EpsField
from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
from dynameta.geometry.cross_section import Circle, Ellipse, Polygon, Rectangle
from dynameta.geometry.specs import OpticalSpec
from dynameta.materials import ConstantOptical, Material, MaterialRegistry

LAM = 1.31e-6
PER = 600e-9


def _design(layers, *, pol="y", theta=0.0, extra=()):
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(complex(1.5 ** 2))))
    reg.add(Material("hi", ConstantOptical(complex(4.0, 0.3))))
    reg.add(Material("lo", ConstantOptical(complex(2.1))))
    # near-ENZ metal (ITO-like at 1.31 um): genuinely Re(eps) < 0 but soft enough
    # that all three engines converge at validation scale (see GATE B header)
    reg.add(Material("ito", ConstantOptical(complex(-3.0, 1.0))))
    for nm, eps in extra:
        reg.add(Material(nm, ConstantOptical(eps)))
    return Design(name="pmm2d", unit_cell=UnitCell.square(PER),
                  stack=Stack(layers=layers, superstrate_material="air",
                              substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def _patch_design(pol="x"):
    """Centered 300 nm ITO patch (walls at PER/4 and 3 PER/4 -> exact on the pure
    (4, 4) union grid AND on the bridge's 128-pixel raster) ON TOP of a lo spacer.
    Design layers are BOTTOM -> TOP, so the spacer comes first (the patch-on-top
    geometry diffracts strongly, R ~ 0.07 -- a robustly discriminating referee
    case; buried-patch R ~ 0.009 left only a 2% monotonicity margin)."""
    patch = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0),
                      material="ito")
    return _design([Layer("spacer", 150e-9, "lo"),
                    Layer("patch", 60e-9, "air", inclusions=[patch])], pol=pol)


def _graded_design_and_eps():
    """Asymmetric LOSSY graded fixture (audit C5-1): eps(top) >> eps(bottom), loss
    concentrated toward the top -- R differs measurably from the flipped copy."""
    d = _design([Layer("a", 120e-9, "hi")])
    z_nm = np.linspace(0.0, 120.0, 25)           # ascending = substrate-first
    u = z_nm / 120.0
    eps_z = 2.0 + 6.7 * u ** 2 + 1.0j * u ** 3
    ef = EpsField(z_axis_u=z_nm, y_axis_u=np.zeros(1), x_axis_u=np.zeros(1),
                  values_zyx=eps_z.reshape(-1, 1, 1).astype(complex))
    return d, {"a": ef}


def _lossy_patch_design(pol="x"):
    """A LOSSY ITO patch (eps -3 + 1j, walls at PER/4 & 3 PER/4 -> pure (4, 4) union
    grid) ON TOP of a LOSSY 'hi' spacer (eps 4 + 0.3j) -- BOTH Im(eps) > 0, the
    per-layer absorption fixture (GATE E).  Design layers are BOTTOM -> TOP."""
    patch = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0),
                      material="ito")
    return _design([Layer("spacer", 150e-9, "hi"),
                    Layer("patch", 60e-9, "air", inclusions=[patch])], pol=pol)


def _lossless_patch_design(pol="x"):
    """The lossless twin of _lossy_patch_design (real-eps materials throughout, same
    geometry): a 'glass' patch (eps 2.25) on a 'lo' spacer (eps 2.1).  A_independent
    must vanish -- the honest zero check for the absorption path."""
    patch = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0),
                      material="glass")
    return _design([Layer("spacer", 150e-9, "lo"),
                    Layer("patch", 60e-9, "air", inclusions=[patch])], pol=pol)


def _uniaxial(no, ne, az_deg, tilt_deg):
    """Uniaxial LC tensor eps = no^2 I + (ne^2 - no^2) d d^T, director d at azimuth
    az (from x, in-plane) and tilt (out of the plane) -- the EpsField.tensor layout
    eps[i, j], i, j in (x, y, z)."""
    a, t = np.radians(az_deg), np.radians(tilt_deg)
    d = np.array([np.cos(t) * np.cos(a), np.cos(t) * np.sin(a), np.sin(t)])
    return (no ** 2 * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)).astype(complex)


def _swap_xy(eps):
    """The x <-> y index-permuted tensor (the classic axis-order mapping slip)."""
    p = [1, 0, 2]
    return np.asarray(eps)[p][:, p].copy()


def _pillar_design(shape, *, bg="air", thk=200e-9, pol="x", under=()):
    """One eps-4 pillar layer (plus optional layers BELOW it) in air on glass, the
    600 nm / 1.31 um non-diffracting cell of GATES G-J (lambda / p = 2.18 > n_glass,
    so every engine's order-summed R / T is the zeroth order)."""
    return _design(list(under) + [Layer("p", thk, bg,
                                        inclusions=[Inclusion(shape=shape,
                                                              material="pil")])],
                   pol=pol, extra=(("pil", complex(4.0)), ("lcm", complex(2.4))))


def _gate_f():
    """Uniform TENSOR layer vs the Berreman 4x4 bridge (module docstring, GATE F)."""
    import lumenairy as lum
    from dynameta.optics.lumenairy_bridge import (design_to_berreman_layers,
                                                  design_to_pmm2d_stack,
                                                  make_lumenairy_berreman_solver,
                                                  make_lumenairy_pmm2d_solver)
    ip = np.array([[3.0 + 0.1j, 0.4 + 0.2j, 0.0], [-0.1 + 0.3j, 2.5 + 0.05j, 0.0],
                   [0.0, 0.0, 2.8 + 0.1j]])
    oop = _uniaxial(1.5, 1.75, 30.0, 40.0) + 0.05j * np.eye(3)
    oop[0, 2] += 0.2
    oop[2, 1] -= 0.15j
    ber = make_lumenairy_berreman_solver()
    solv = {"pure": make_lumenairy_pmm2d_solver(engine="pure", n_modes=5, n_orders=3),
            "hybrid": make_lumenairy_pmm2d_solver(engine="hybrid", degree=7,
                                                  n_orders=3)}
    kw = {"pure": dict(n_modes=5, n_orders=3), "hybrid": dict(degree=7, n_orders=3)}

    def jones(d, ebr, eng):
        st, _ = design_to_pmm2d_stack(d, LAM, engine=eng, eps_by_region=ebr, **kw[eng])
        th = np.radians(d.optical.incidence_angle_deg)
        st.set_source(LAM, theta=th, phi=0.0)
        jr = np.asarray(st.solve()[3])
        return jr, np.asarray(st.jones_transmission())

    def ber_jones(d, ebr):
        layers, n_sup, n_sb, _ = design_to_berreman_layers(d, LAM, eps_by_region=ebr)
        st = lum.BerremanStack(n_substrate=complex(n_sb), n_superstrate=complex(n_sup))
        for e, t in layers:
            st.add_layer(float(t), eps=e)
        st.set_source(LAM, theta=np.radians(d.optical.incidence_angle_deg), phi=0.0)
        return np.asarray(st.solve()[2]), np.asarray(st.jones_transmission())

    worst = {"pure": 0.0, "hybrid": 0.0}
    worst_j = {"pure": 0.0, "hybrid": 0.0}
    disc_t, disc_p = np.inf, np.inf
    for eps in (ip, oop):
        ebr = {"a": EpsField(tensor=eps)}
        for pol, th in (("x", 0.0), ("y", 30.0), ("p", 30.0)):
            d = _design([Layer("a", 200e-9, "lo")], pol=pol, theta=th)
            rb = ber(d, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j)
            for eng in ("pure", "hybrid"):
                rp = solv[eng](d, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j)
                worst[eng] = max(worst[eng], abs(rp.R - rb.R), abs(rp.T - rb.T),
                                 abs(rp.r - rb.r), abs(rp.t - rb.t))
            # the co-pol permutation slip (x <-> y) is visible to the solver's r
            rp_s = solv["pure"](d, None, {"a": EpsField(tensor=_swap_xy(eps))}, LAM,
                                1.0 + 0j, 1.5 + 0j)
            disc_p = min(disc_p, abs(rp_s.r - rb.r))
        # FULL Jones (normal + 30 deg) vs BerremanStack -- the off-diagonal is the
        # only place a TRANSPOSED mapping shows (reciprocity: co-pol is blind)
        for th in (0.0, 30.0):
            d = _design([Layer("a", 200e-9, "lo")], pol="p" if th else "x", theta=th)
            jb = ber_jones(d, ebr)
            for eng in ("pure", "hybrid"):
                jp = jones(d, ebr, eng)
                worst_j[eng] = max(worst_j[eng], np.abs(jp[0] - jb[0]).max(),
                                   np.abs(jp[1] - jb[1]).max())
            jt = jones(d, {"a": EpsField(tensor=eps.T.copy())}, "pure")
            disc_t = min(disc_t, max(np.abs(jt[0] - jb[0]).max(),
                                     np.abs(jt[1] - jb[1]).max()))
    # measured at bring-up 2026-11-02 (M = 5 pure / degree 7 hybrid): pure 1.7e-10
    # worst (the oblique plane wave on the polynomial basis; 6e-15 at normal), hybrid
    # 7e-15; full Jones pure 2.8e-10 / hybrid 1.8e-14; transposed-tensor Jones shift
    # >= 8.5e-3 (min over the four arms); x <-> y permuted co-pol r shift >= 2.5e-2.
    # Full run 2026-11-02: 323 s for GATES A-J on a loaded box (A-E alone ~96 s).
    g = bool(worst["pure"] < 1e-8 and worst["hybrid"] < 1e-8
             and worst_j["pure"] < 1e-8 and worst_j["hybrid"] < 1e-8
             and disc_t > 5e-3 and disc_p > 1e-3)
    print("[lp2b] GATE F: uniform tensor (in-plane asym + OOP) vs Berreman: worst "
          "R/T/r/t |d| pure {:.1e} / hybrid {:.1e}, full Jones pure {:.1e} / hybrid "
          "{:.1e} (< 1e-8); transposed-tensor Jones shift {:.1e} (> 5e-3), x<->y "
          "permuted r shift {:.1e} (> 1e-3) -> {}".format(
              worst["pure"], worst["hybrid"], worst_j["pure"], worst_j["hybrid"],
              disc_t, disc_p, "PASS" if g else "FAIL"), flush=True)
    return g


def _rcwa_tensor_oracle(shape, eps_bg, eps_pil, thk, n_orders, samples=600):
    """DIRECT lumenairy RCWAStack on an (S, S, 3, 3) eps_tensor_cell painted from the
    shape (cell-centred samples, axis 0 = x) -- returns (R_x, R_y) order-summed."""
    import lumenairy as lum
    from dynameta.optics.rasterize import cell_axes
    x, y = cell_axes(samples, samples, PER, PER)
    X, Y = np.meshgrid(x, y, indexing="ij")
    cell = np.broadcast_to(np.asarray(eps_bg, dtype=complex),
                           (samples, samples, 3, 3)).copy()
    cell[np.asarray(shape.contains_m(X, Y), dtype=bool)] = eps_pil * np.eye(3)
    st = lum.RCWAStack(PER, period_y=PER, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=int(n_orders), n_orders_y=int(n_orders))
    st.add_layer(thk, eps_tensor_cell=cell)
    st.set_source(LAM, theta=0.0, phi=0.0)
    _, R, _ = st.solve().efficiencies()
    return float(np.sum(R[0])), float(np.sum(R[1]))


def _gate_g():
    """Patterned anisotropic (LC background + pillar) vs direct-RCWA tensor raster."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import resolve_pure_grid
    pure = make_lumenairy_pmm2d_solver(engine="pure", n_modes=5, n_orders=3)
    hyb = make_lumenairy_pmm2d_solver(engine="hybrid", degree=9, n_orders=9)
    thk = 250e-9
    shapes = {"square": Rectangle(PER / 2.0, PER / 2.0, PER / 2.0, PER / 2.0),
              "circle": Circle(PER / 2.0, PER / 2.0, 0.3 * PER)}
    ok_all, lines = True, []
    for name, shp in shapes.items():
        for tilt in (0.0, 40.0):
            eps_lc = _uniaxial(1.5, 1.7, 30.0, tilt)
            ebr = {"p": EpsField(tensor=eps_lc)}
            dx = _pillar_design(shp, bg="lcm", thk=thk, pol="x")
            dy = _pillar_design(shp, bg="lcm", thk=thk, pol="y")
            route = resolve_pure_grid(dx)
            rx = pure(dx, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j).R
            ry = pure(dy, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j).R
            dist, rc = [], []
            for n in (6, 9, 12):
                ox, oy = _rcwa_tensor_oracle(shp, eps_lc, 4.0, thk, n)
                rc.append((ox, oy))
                dist.append((abs(ox - rx), abs(oy - ry)))
            mono = all(dist[k + 1][c] < dist[k][c] for k in range(2) for c in (0, 1))
            fin = max(dist[-1])
            # first-order Richardson on the 9 / 12 rungs: the Laurent-only tensor
            # path converges O(1/n) (measured d12/d9 = 0.79-0.83 vs 9/12 = 0.75)
            rich = [(12.0 * rc[2][c] - 9.0 * rc[1][c]) / 3.0 for c in (0, 1)]
            d_rich = max(abs(rich[0] - rx), abs(rich[1] - ry))
            d_hyb = None
            if name == "square":     # the staircase circle trips the hybrid's own
                d_hyb = abs(hyb(dx, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j).R - rx)
            # discrimination = the MUTANT must FAIL the acceptance the real mapping
            # passes (|R - R_rich| < 2e-4), not merely move R by some amount
            disc = abs(pure(dx, None, {"p": EpsField(tensor=_swap_xy(eps_lc))}, LAM,
                            1.0 + 0j, 1.5 + 0j).R - rich[0])
            disc_oop = None
            if tilt:
                bf = eps_lc.copy()
                bf[0, 2] = bf[1, 2] = bf[2, 0] = bf[2, 1] = 0.0
                disc_oop = abs(pure(dx, None, {"p": EpsField(tensor=bf)}, LAM,
                                    1.0 + 0j, 1.5 + 0j).R - rich[0])
            g = bool(mono and fin < 4e-4 and d_rich < 2e-4 and disc > 2e-4
                     and (d_hyb is None or d_hyb < 1.5e-4)
                     and (disc_oop is None or disc_oop > 2e-4))
            ok_all = ok_all and g
            lines.append("{} tilt {:.0f} [{}]: RCWA |d| x {:.1e}->{:.1e}->{:.1e}, y "
                         "{:.1e}->{:.1e}->{:.1e}, Richardson {:.1e}{}; mutant vs "
                         "Richardson: x<->y perm {:.1e}{}{}".format(
                             name, tilt, route, dist[0][0], dist[1][0], dist[2][0],
                             dist[0][1], dist[1][1], dist[2][1], d_rich,
                             "" if d_hyb is None else ", hybrid {:.1e}".format(d_hyb),
                             disc, "" if disc_oop is None
                             else ", OOP-dropped {:.1e}".format(disc_oop),
                             "" if g else " FAIL"))
    # measured at bring-up 2026-11-02 (pure M = 5, converged to <= 7e-7 vs M = 7):
    # square tilt 0 x 3.1e-4 -> 1.9e-4 -> 1.6e-4 (y 5.7e-4 -> 2.9e-4), tilt 40 x
    # 4.2e-4 -> 2.1e-4 (y 6.2e-4 -> 3.1e-4); circle tilt 0 x 2.9e-4 -> 1.5e-4,
    # tilt 40 alike -- RCWA's tensor path is Laurent-only (no Li rule for tensor
    # cells), hence the slow O(1/n) approach FROM ABOVE; hybrid 5e-5 / 6e-5 below;
    # x<->y permutation 1.6e-2 (it IS the y-pol answer for the square, by symmetry).
    # The OOP-dropped mutant (tilted director truncated to block form) moves R by
    # only 7.8e-4 (square) / 6.3e-4 (circle) -- inside 2x the raw n = 12 gap, so a
    # raw-gap tolerance could NOT tell it apart (first bring-up run: the gate's
    # "> 1e-3" guess failed on exactly this); the 1/n Richardson value (within
    # 0.3-1.1e-4 of pure on all eight pol x case arms) is what separates it.
    for ln in lines:
        print("[lp2b]   " + ln, flush=True)
    print("[lp2b] GATE G: patterned anisotropic (LC bg, in-plane + tilted) vs direct "
          "RCWA tensor raster: strictly shrinking, < 4e-4 at n 12, Richardson < 2e-4, "
          "hybrid < 1.5e-4, mutants > 2e-4 from Richardson -> {}".format("PASS" if ok_all else "FAIL"), flush=True)
    return ok_all


def _gate_h():
    """t power closure + the per-order amplitude route (GATE H)."""
    from dynameta.optics.lumenairy_bridge import (design_to_pmm2d_stack,
                                                  make_lumenairy_pmm2d_solver)
    from dynameta.optics.lumenairy_bridge._common import p_basis_conversion

    def pfac(th_deg):
        th = np.radians(th_deg)
        ci = np.cos(th)
        ct = np.sqrt(1.0 - (np.sin(th) / 1.5) ** 2)
        return 1.5 * ct / ci

    pure = make_lumenairy_pmm2d_solver(engine="pure", n_modes=5, n_orders=3)
    worst, disc, worst_po = 0.0, 0.0, 0.0
    lays = [Layer("a", 120e-9, "lo"), Layer("b", 200e-9, "glass")]
    for pol, th in (("y", 30.0), ("p", 30.0)):
        d = _design(lays, pol=pol, theta=th)
        r = pure(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        worst = max(worst, abs(abs(r.t) ** 2 * pfac(th) - r.T))
        if pol == "p":       # undo the p-basis factor -> the lab-x amplitude
            _, tf = p_basis_conversion("p", np.radians(th), 1.0, 1.5)
            disc = abs(abs(r.t / tf) ** 2 * pfac(th) - r.T)
    d = _pillar_design(Circle(PER / 2.0, PER / 2.0, 0.3 * PER), pol="x")
    r = pure(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    worst = max(worst, abs(abs(r.t) ** 2 * pfac(0.0) - r.T))
    st, _ = design_to_pmm2d_stack(d, LAM, engine="pure", n_modes=5, n_orders=3)
    st.set_source(LAM, theta=0.0, phi=0.0)
    st.solve()
    po = st.per_order_amplitudes("transmission")
    k0 = int(np.flatnonzero((po["orders"][:, 0] == 0) & (po["orders"][:, 1] == 0))[0])
    jt = np.asarray(st.jones_transmission())
    worst_po = max(np.abs(po["Ex"][:, k0] - jt[0]).max(),
                   np.abs(po["Ey"][:, k0] - jt[1]).max(), abs(jt[0, 0] - r.t))
    # measured at bring-up 2026-11-02: closure <= 4e-13 (uniform) / 2e-12 (circle);
    # per-order route == jones_transmission bit-for-bit; dropping the p factor
    # breaks closure by 1.5e-2 at 30 deg.
    g = bool(worst < 1e-9 and worst_po < 1e-12 and disc > 1e-2)
    print("[lp2b] GATE H: |t|^2 power factor == T worst {:.1e} (< 1e-9; uniform y/p "
          "30deg + circle pillar); per-order (0,0) == jones_transmission == bridge t "
          "{:.1e}; p-factor-dropped closure {:.1e} (> 1e-2) -> {}".format(
              worst, worst_po, disc, "PASS" if g else "FAIL"), flush=True)
    return g


def _bracket(r_pure, fff, lau):
    """(dists, strictly_shrinking, two_sided): FFF distances to the pure value over
    the ladder, whether they strictly shrink, and whether Laurent's last rung sits on
    the OTHER side of pure from FFF's last rung."""
    dist = [abs(v - r_pure) for v in fff]
    shrink = all(dist[k + 1] < dist[k] for k in range(len(dist) - 1))
    two = (fff[-1] - r_pure) * (lau - r_pure) < 0.0
    return dist, shrink, two


def _rcwa_ladder(d, orders=(9, 12, 15)):
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    fff = [make_lumenairy_rcwa_solver(n_orders=n, cell_samples=600, formulation="fff")(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R for n in orders]
    lau = make_lumenairy_rcwa_solver(n_orders=orders[-1], cell_samples=600)(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R
    return fff, lau


def _gate_i():
    """Curved shapes (circle, ellipse) vs the RCWA FFF / Laurent bracket."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import resolve_pure_grid
    pure = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3)
    rp = {}
    ok_all, out = True, []
    cases = {"circle": (Circle(PER / 2.0, PER / 2.0, 0.3 * PER),
                        Circle(PER / 2.0, PER / 2.0, 0.306 * PER), 1e-4),
             "ellipse": (Ellipse(PER / 2.0, PER / 2.0, 0.33 * PER, 0.22 * PER),
                         Ellipse(PER / 2.0, PER / 2.0, 0.22 * PER, 0.33 * PER), 1e-3)}
    for name, (shp, wrong, dmin) in cases.items():
        d = _pillar_design(shp)
        route = resolve_pure_grid(d)
        rp[name] = pure(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R
        fff, lau = _rcwa_ladder(d)
        dist, shrink, two = _bracket(rp[name], fff, lau)
        disc = abs(pure(_pillar_design(wrong), None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R
                   - rp[name])
        g = bool(route == "shapes" and shrink and two and dist[-1] < 2e-5
                 and disc > dmin)
        ok_all = ok_all and g
        out.append("{} [{}]: R_pure {:.6e}; FFF |d| {:.1e}->{:.1e}->{:.1e}, Laurent "
                   "{:+.1e} (other side: {}); wrong-geometry shift {:.1e} (> {:.0e}){}"
                   .format(name, route, rp[name], dist[0], dist[1], dist[2],
                           lau - rp[name], two, disc, dmin, "" if g else " FAIL"))
    # measured at bring-up 2026-11-02: circle FFF 1.25e-5 -> 1.0e-5 -> 8.4e-6 from
    # BELOW, Laurent +7.1e-5 from ABOVE, r + 2 % shift 1.6e-4; ellipse FFF 1.35e-5
    # -> 9.6e-6 -> 7.4e-6 from ABOVE, Laurent -5.3e-5, rx <-> ry swap 2.9e-3.  Pure
    # M = 6 sits 1e-6 (circle M 7) / 2e-7 (ellipse M 7) from its next rung.
    for ln in out:
        print("[lp2b]   " + ln, flush=True)
    print("[lp2b] GATE I: curved shapes (exact map) inside the RCWA FFF/Laurent "
          "bracket, FFF < 2e-5 and shrinking, discriminating -> {}".format(
              "PASS" if ok_all else "FAIL"), flush=True)
    return ok_all


def _gate_j():
    """Non-commensurate rectangles: 'shapes' (auto) + 'per-layer' routes."""
    from dynameta.optics.lumenairy_bridge import make_lumenairy_pmm2d_solver
    from dynameta.optics.lumenairy_bridge.pmm2d_backend import resolve_pure_grid

    def odd(w=0.46 * PER):
        return Rectangle(0.53 * PER, 0.48 * PER, w, 0.37 * PER)

    pure = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3)
    perl6 = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3,
                                        pure_grid="per-layer")
    d = _pillar_design(odd())
    route = resolve_pure_grid(d)
    r_p = pure(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R
    fff, lau = _rcwa_ladder(d)
    dist, shrink, two = _bracket(r_p, fff, lau)
    disc = abs(pure(_pillar_design(odd(0.47 * PER)), None, {}, LAM, 1.0 + 0j,
                    1.5 + 0j).R - r_p)
    same = abs(perl6(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R - r_p)
    g1 = bool(route == "shapes" and shrink and two and dist[-1] < 4e-5
              and disc > 2e-4 and same < 1e-10)
    # two patterned layers on DIFFERENT walls: an x-stripe (walls on x only -> the
    # y axis is bisected up to the same segment count) under the pillar
    stripe = Inclusion(shape=Rectangle(0.375 * PER, PER / 2.0, 0.35 * PER, PER),
                       material="pil")

    def two_layer(w=0.46 * PER):
        return _pillar_design(odd(w), under=[Layer("st", 150e-9, "air",
                                                   inclusions=[stripe])])
    perl7 = make_lumenairy_pmm2d_solver(engine="pure", n_modes=7, n_orders=3,
                                        pure_grid="per-layer")
    r_pl = perl7(two_layer(), None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R
    from dynameta.optics.lumenairy_bridge import make_lumenairy_rcwa_solver
    r_ff = make_lumenairy_rcwa_solver(n_orders=15, cell_samples=600, formulation="fff")(
        two_layer(), None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R
    d_pl = abs(r_pl - r_ff)
    # the mutant must FAIL the same acceptance (|R - R_FFF| < 2e-4) the real one passes
    disc_pl = abs(perl7(two_layer(0.49 * PER), None, {}, LAM, 1.0 + 0j, 1.5 + 0j).R
                  - r_ff)
    g2 = bool(d_pl < 2e-4 and disc_pl > 2e-4)
    # measured at bring-up 2026-11-02: single layer FFF 4.1e-5 -> 2.9e-5 -> 2.7e-5
    # from ABOVE (pure M = 6 itself still 6.7e-6 below M = 7: the dielectric corner
    # singularity caps both methods algebraically), Laurent -4.0e-4; 6 nm wider 4.9e-4;
    # per-layer == shapes to round-off (identical walls).  Two-layer: 'shapes'
    # merges BOTH layers' walls into a 5 x 5 grid (290 s at M = 7 -- the cost case
    # per-layer exists for) while per-layer M = 7 takes ~15 s and lands 6.6e-5 below
    # RCWA-FFF (n 15), whose own ladder moves 3e-6 per rung there; the 18 nm wider
    # mutant moves per-layer R by 4.2e-4 (first bring-up's "> 1e-3" was a guess and
    # failed), landing 3.5e-4 from FFF -- it fails the 2e-4 acceptance.
    g = g1 and g2
    print("[lp2b] GATE J: non-commensurate rect [{}]: FFF |d| {:.1e}->{:.1e}->{:.1e} "
          "(< 4e-5, shrinking {}), Laurent {:+.1e} (other side {}), 6 nm wider {:.1e} "
          "(> 2e-4), per-layer == shapes {:.1e}; two-layer per-layer vs FFF {:.1e} "
          "(< 2e-4), 18 nm wider vs FFF {:.1e} (> 2e-4) -> {}".format(
              route, dist[0], dist[1], dist[2], shrink, lau - r_p, two, disc, same,
              d_pl, disc_pl, "PASS" if g else "FAIL"), flush=True)
    return g


def main():
    if importlib.util.find_spec("lumenairy") is None:
        print("[lp2b] *** SKIP: lumenairy not installed -- PMM2D gates not run ***",
              flush=True)
        return True
    from dynameta.optics.lumenairy_bridge import (make_lumenairy_pmm2d_solver,
                                                  make_lumenairy_rcwa_solver,
                                                  pure_union_grid_n)
    from dynameta.optics.tmm_reference import make_layered_tmm_solver

    print("[lp2b] === Lumenairy PMM2D bridge vs TMM / cross-engine referee ===",
          flush=True)
    ok = True
    tmm = make_layered_tmm_solver()

    # ---- GATE A: unstructured asymmetric lossy stack vs TMM (both engines) ----
    # pure n_modes=6 measured 5.0e-13 worst; hybrid is Rayleigh-exact on uniform
    # films (measured ~1e-15).  Threshold 1e-10 = two decades of margin.  The complex
    # t joined 2026-11-02 (measured at bring-up: pure 4.3e-13 / hybrid 3.5e-16 worst,
    # incl. the p-pol cos_i / cos_t factor).
    lays = [Layer("a", 120e-9, "hi"), Layer("b", 200e-9, "lo")]
    worst = {"pure": 0.0, "hybrid": 0.0}
    solv = {"pure": make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3),
            "hybrid": make_lumenairy_pmm2d_solver(engine="hybrid", degree=7,
                                                  n_orders=3)}
    for pol, th in (("y", 0.0), ("y", 30.0), ("p", 30.0)):
        d = _design(lays, pol=pol, theta=th)
        r_t = tmm(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
        for eng in ("pure", "hybrid"):
            r_p = solv[eng](d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
            worst[eng] = max(worst[eng], abs(r_p.R - r_t.R), abs(r_p.T - r_t.T),
                             abs(r_p.r - r_t.r), abs(r_p.t - r_t.t))
    g_a = bool(worst["pure"] < 1e-10 and worst["hybrid"] < 1e-10)
    ok = ok and g_a
    print("[lp2b] GATE A: uniform vs TMM (normal + 30deg y/p; R, T, r, t): worst |d| pure = "
          "{:.2e}, hybrid = {:.2e} -> {}".format(worst["pure"], worst["hybrid"],
                                                 "PASS" if g_a else "FAIL"),
          flush=True)

    # ---- GATE B: pure PMM2D as the 2-D convergence referee for RCWA ----
    d = _patch_design(pol="x")
    n_grid = pure_union_grid_n(d)
    ref_lo = make_lumenairy_pmm2d_solver(engine="pure", n_modes=5, n_orders=3)(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    ref = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3)(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    self_conv = abs(ref.R - ref_lo.R)
    dists = []
    for n in (4, 8, 12, 16):
        rc = make_lumenairy_rcwa_solver(n_orders=n)(d, None, {}, LAM,
                                                    1.0 + 0j, 1.5 + 0j)
        dists.append(abs(rc.R - ref.R))
        print("[lp2b]   RCWA n_orders {:2d}: R = {:.5f} (pure ref {:.5f}, "
              "|d| = {:.1e})".format(n, rc.R, ref.R, dists[-1]), flush=True)
    r_hyb = make_lumenairy_pmm2d_solver(engine="hybrid", degree=9, n_orders=9)(
        d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    d_hyb = abs(r_hyb.R - ref.R)
    # measured at bring-up: dists 5.7e-3 -> 4.0e-3 -> 1.6e-3 -> 5.8e-4 (strictly
    # decreasing TOWARD the pure value); pure self-convergence 3.1e-4 sits BELOW
    # the final RCWA gap, so the approach direction is resolvable, not noise
    g_b = bool(n_grid == 4
               and dists[3] < dists[2] < dists[1] < dists[0]
               and dists[3] < 1.5e-3 and self_conv < 1e-3
               and self_conv < dists[2] and d_hyb < 5e-3)
    ok = ok and g_b
    print("[lp2b] GATE B: union grid N = {}; RCWA converges TOWARD pure "
          "({:.1e} -> {:.1e} -> {:.1e} -> {:.1e}, final < 1.5e-3); pure degree-"
          "stability {:.1e}; hybrid within {:.1e} (< 5e-3) -> {}".format(
              n_grid, dists[0], dists[1], dists[2], dists[3], self_conv, d_hyb,
              "PASS" if g_b else "FAIL"), flush=True)

    # ---- GATE C: graded slab ORDER (audit C5-1) on an asymmetric lossy profile ----
    d, ebr = _graded_design_and_eps()
    r_t = tmm(d, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j)
    r_rc = make_lumenairy_rcwa_solver(n_orders=2)(d, None, ebr, LAM,
                                                  1.0 + 0j, 1.5 + 0j)
    worst_c = 0.0
    for eng in ("pure", "hybrid"):
        r_p = solv[eng](d, None, ebr, LAM, 1.0 + 0j, 1.5 + 0j)
        worst_c = max(worst_c, abs(r_p.R - r_t.R), abs(r_p.T - r_t.T),
                      abs(r_p.R - r_rc.R))
    # the fixture must actually discriminate the flip: the same profile reversed
    # in z gives a materially different R (else a future regression could hide)
    d2, ebr2 = _graded_design_and_eps()
    ef = ebr2["a"]
    ebr_flip = {"a": EpsField(z_axis_u=ef.z_axis_u, y_axis_u=ef.y_axis_u,
                              x_axis_u=ef.x_axis_u,
                              values_zyx=ef.values_zyx[::-1].copy())}
    r_flip = solv["pure"](d2, None, ebr_flip, LAM, 1.0 + 0j, 1.5 + 0j)
    disc = abs(r_flip.R - r_t.R)
    g_c = bool(worst_c < 1e-8 and disc > 1e-3)
    ok = ok and g_c
    print("[lp2b] GATE C: graded slab order vs TMM + RCWA bridge: worst |d| = "
          "{:.2e} (< 1e-8); flipped-profile discrimination |dR| = {:.2e} "
          "(> 1e-3) -> {}".format(worst_c, disc, "PASS" if g_c else "FAIL"),
          flush=True)

    # ---- GATE D: scope guards raise ----
    g_d = 0
    pure = solv["pure"]
    union = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3,
                                        pure_grid="union")
    # 1) conical azimuth (audit C4-2)
    d = _design(lays, pol="y", theta=30.0)
    object.__setattr__(d.optical, "azimuth_deg", 20.0)
    try:
        pure(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except NotImplementedError:
        g_d += 1
    # 2) bottom incidence
    d = _design(lays, pol="y")
    object.__setattr__(d.optical, "incidence_side", "bottom")
    try:
        pure(d, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except NotImplementedError:
        g_d += 1
    # 3) Circle under an EXPLICIT union grid (the v1 analytic-rectangles-only route;
    # 'auto' / 'shapes' take it exactly since 2026-11-02 -- GATE I)
    disk = Inclusion(shape=Circle(PER / 2.0, PER / 2.0, PER / 4.0), material="ito")
    try:
        union(_design([Layer("d", 60e-9, "air", inclusions=[disk])], pol="y"),
              None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # 4) non-commensurate Rectangle walls under an explicit union grid
    odd = Inclusion(shape=Rectangle(PER / 2.0, PER / 2.0, 0.46 * PER, 0.46 * PER),
                    material="ito")
    try:
        union(_design([Layer("o", 60e-9, "air", inclusions=[odd])], pol="y"),
              None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # 5) laterally-STRUCTURED gridded EpsField on pure (genuinely x-varying values;
    # laterally-uniform grids are in scope -- they slice to uniform slabs, GATE C)
    grid = EpsField(values_zyx=np.broadcast_to(2.0 + 0.1 * np.arange(8.0),
                                               (3, 1, 8)).astype(complex),
                    z_axis_u=np.array([0.0, 50.0, 100.0]),
                    x_axis_u=np.arange(8.0), y_axis_u=np.array([0.0]))
    try:
        pure(_design([Layer("a", 100e-9, "hi")], pol="y"), None, {"a": grid},
             LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # 6) the same for a TENSOR raster (laterally-uniform tensor grids ARE in scope)
    tv = (np.eye(3) * (2.0 + 0.1 * np.arange(8.0))[:, None, None]).astype(complex)
    tgrid = EpsField(values_zyx=np.broadcast_to(tv, (3, 1, 8, 3, 3)).copy(),
                     z_axis_u=np.array([0.0, 50.0, 100.0]),
                     x_axis_u=np.arange(8.0), y_axis_u=np.array([0.0]))
    try:
        pure(_design([Layer("a", 100e-9, "hi")], pol="y"), None, {"a": tgrid},
             LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # 7) a Polygon inclusion on the pure engine (no exact image -> staircase refused)
    tri = Inclusion(shape=Polygon([(100e-9, 100e-9), (500e-9, 120e-9),
                                   (300e-9, 480e-9)]), material="ito")
    try:
        pure(_design([Layer("t", 60e-9, "air", inclusions=[tri])], pol="y"),
             None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    except ValueError:
        g_d += 1
    # 8) bogus engine name; 9) bogus pure_grid
    try:
        make_lumenairy_pmm2d_solver(engine="fmm")
    except ValueError:
        g_d += 1
    try:
        make_lumenairy_pmm2d_solver(pure_grid="mortar")
    except ValueError:
        g_d += 1
    g_d = bool(g_d == 9)
    ok = ok and g_d
    print("[lp2b] GATE D: conical / bottom-incidence / circle-on-union / "
          "non-commensurate-on-union / structured-grid-on-pure / structured-tensor-"
          "grid-on-pure / polygon-on-pure / bad-engine / bad-pure_grid guards "
          "({}/9) -> {}".format(9 if g_d else "<9", "PASS" if g_d else "FAIL"),
          flush=True)

    # ---- GATE E: pure-engine per-layer absorption budget closure (audit C3) ----
    # measured at bring-up (n_modes=6 pure / degree=9 hybrid on the lossy 4x4 patch):
    # pure closes its Gram-flux-vs-far-field budget at 3.7e-15, hybrid at 1.0e-15;
    # A_independent 0.1198 (pure) vs 0.1203 (hybrid) -> cross 4.6e-4 (staggered-modal
    # vs Fourier-projected internal quadrature: two DIFFERENT bases, << the 1e-2 the
    # two-basis total is gated at); lossless twin 5.3e-15; off-switch A_indep None.
    d_abs = _lossy_patch_design(pol="x")
    r_pa = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3,
                                       absorption=True)(d_abs, None, {}, LAM,
                                                        1.0 + 0j, 1.5 + 0j)
    close_p = abs(r_pa.A_independent - r_pa.A)
    keys_ok = (r_pa.per_region_absorption is not None
               and set(r_pa.per_region_absorption) == {"spacer", "patch"})
    r_ll = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3,
                                       absorption=True)(_lossless_patch_design(),
                                                        None, {}, LAM,
                                                        1.0 + 0j, 1.5 + 0j)
    r_off = make_lumenairy_pmm2d_solver(engine="pure", n_modes=6, n_orders=3)(
        d_abs, None, {}, LAM, 1.0 + 0j, 1.5 + 0j)
    off_ok = bool(r_off.per_region_absorption is None
                  and r_off.A_independent is None)
    r_ha = make_lumenairy_pmm2d_solver(engine="hybrid", degree=9, n_orders=9,
                                       absorption=True)(d_abs, None, {}, LAM,
                                                        1.0 + 0j, 1.5 + 0j)
    close_h = abs(r_ha.A_independent - r_ha.A)
    cross = abs(r_pa.A_independent - r_ha.A_independent)
    g_e = bool(close_p < 1e-6 and keys_ok and r_pa.A_independent > 1e-3
               and abs(r_ll.A_independent) < 1e-6 and off_ok
               and close_h < 1e-6 and cross < 1e-2)
    ok = ok and g_e
    print("[lp2b] GATE E: pure per-layer absorption: closure |dA| = {:.1e} (< 1e-6), "
          "A_indep = {:.4f} keyed [{}], lossless A_indep = {:.1e} (< 1e-6), "
          "off-switch None {}; Hybrid-vs-Pure |dA_indep| = {:.1e} (< 1e-2, both "
          "close own budget: pure {:.1e} / hybrid {:.1e}) -> {}".format(
              close_p, r_pa.A_independent,
              "+".join(sorted(r_pa.per_region_absorption)) if keys_ok else "BADKEYS",
              r_ll.A_independent, "OK" if off_ok else "BAD", cross, close_p, close_h,
              "PASS" if g_e else "FAIL"), flush=True)

    # ---- GATES F-J: the 2026-11-02 capabilities (tensor, curved, walls, t) ----
    ok = _gate_f() and ok
    ok = _gate_g() and ok
    ok = _gate_h() and ok
    ok = _gate_i() and ok
    ok = _gate_j() and ok

    print("[lp2b] *** LUMENAIRY PMM2D BRIDGE: {} ***".format(
        "PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
