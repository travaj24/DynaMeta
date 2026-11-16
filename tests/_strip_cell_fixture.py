"""Shared fixtures for the v0.16 RCWA-bridge gates (tests/test_rcwa_auto_formulation.py,
tests/test_rcwa_area_raster.py).

* `strip_design`: the generic strip cell -- a metal-strip grating over a thin oxide / film / oxide
  spacer on a lossy mirror (metal strips eps -60+4i, 150 nm of a 400 nm period, 40 nm tall; 80 nm
  mirror metal eps -120+20i; spacer 2 nm oxA (n 1.70) / 6 nm oxB (n 2.00) / 6 nm Drude film /
  6 nm oxB / 2 nm oxA; glass substrate n 1.45), 1200 nm. In TM (polarization 'x') it is a gap-mode
  resonator (reflection dip near 1190 nm, R ~ 0.113 at 1200 nm) whose metal corners make every
  Fourier method converge algebraically.
* `fem_strip`: the same cell through the independent 2-D TM finite-element solver
  (optics.fem_2d_tm, H_y on a corner-graded quadrilateral mesh; no Fourier factorization).
  Its default (order 4, 6 corner levels) is 2.8e-5 from its own converged value R_FEM_CONVERGED.
  MEASURED convergence study (2 threads; r, then |r - R_FEM_CONVERGED|):
      order 3, 6 levels             0.3354363+0.0167988i   3.8e-4
      order 4, 6 levels (default)   0.3352817+0.0171654i   2.8e-5
      order 5, 8 levels             0.3352712+0.0171903i   2.3e-6
      order 6, 10 levels            0.3352712+0.0171900i   2.6e-6
      order 8, 10 levels            0.3352712+0.0171901i   2.6e-6
      order 9, 12 levels            0.3352707+0.0171914i   1.0e-6
      order 8, 10 levels, ratio .15 0.3352694+0.0171947i   2.6e-6
      order 8, 12 lv, ratio .15, h/2 0.3352708+0.0171910i  1.4e-6
      order 9, 14 lv, ratio .2, h/2  0.3352696+0.0171943i  2.1e-6
  The finest settings scatter by ~5e-6 with the grading ratio and h, not with the order, so the
  converged value is their mean, R_FEM_CONVERGED = 0.335270+0.017192i, uncertain by ~3e-6.
* `v0151_solve`: the v0.15.1 RCWA construction of a 1-D design REBUILT INLINE from lumenairy's
  public API (the same calls in the same order: no `formulation` keyword on an analytic shapes
  layer -- i.e. lumenairy's 5.50.x default 'laurent', named explicitly here --, cell-centred point
  raster, gridded values passed as they come), for the same-process bit-identity A/B of
  formulation='laurent', raster_sampling='point'.
"""
import numpy as np

LAM, P, W, N_BG = 1200e-9, 400e-9, 150e-9, 3.0e26
H_STRIP = 40e-9
EPS_STRIP, EPS_MIRROR, N_GLASS = -60.0 + 4.0j, -120.0 + 20.0j, 1.45
R_FEM_CONVERGED = 0.335270 + 0.017192j


def drude():
    from dynameta.materials import M_E, DrudeOptical
    return DrudeOptical(eps_inf=4.0, m_opt_kg=0.30 * M_E, gamma_rad_s=1.2e14)


def eps_film(n=N_BG, lam=LAM):
    return complex(drude().eps(lam, n_m3=n))


def mats():
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(N_GLASS ** 2 + 0j)))
    reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j)))
    reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j)))
    reg.add(Material("metal", ConstantOptical(EPS_STRIP), is_metal=True))
    reg.add(Material("mirror_metal", ConstantOptical(EPS_MIRROR), is_metal=True))
    reg.add(Material("TCO", optical=drude()))
    reg.add(Material("diel", ConstantOptical(2.25 + 0.01j)))
    return reg


FILM = [("mirror", 80e-9, "mirror_metal"), ("lo_oxa", 2e-9, "oxA"), ("lo_oxb", 6e-9, "oxB"),
        ("film", 6e-9, "TCO"), ("up_oxb", 6e-9, "oxB"), ("up_oxa", 2e-9, "oxA")]
T_FILM_NM = 6.0


def strip_design(width=W, pol="x", theta=0.0, strip_material="metal", cx=P / 2, extra=()):
    from dynameta.geometry import Design, Inclusion, Layer, Stack, UnitCell
    from dynameta.geometry.cross_section import Rectangle
    from dynameta.geometry.specs import OpticalSpec
    layers = [Layer(n, t, m) for n, t, m in FILM]
    layers.append(Layer("grating", H_STRIP, "air",
                        inclusions=[Inclusion(Rectangle(cx, P / 2, width, P), strip_material)]))
    reg = mats()
    for m in extra:
        reg.add(m)
    return Design(name="strip", unit_cell=UnitCell(period_x_m=P, period_y_m=P),
                  stack=Stack(layers=layers, superstrate_material="air", substrate_material="glass"),
                  electrodes=[], materials=reg,
                  optical=OpticalSpec(polarization=pol, incidence_angle_deg=theta))


def ebr():
    from dynameta.core.eps_field import EpsField
    return {"film": EpsField(scalar=eps_film())}


def graded_film_field(nz=21):
    """A laterally uniform, z-graded film (accumulation at the top face), axes in nm."""
    from dynameta.core.eps_field import EpsField
    zn = np.linspace(0.0, T_FILM_NM, nz)
    eg = np.asarray([eps_film(N_BG * (1.0 + 3.0 * np.exp(-(T_FILM_NM - z) / 0.6))) for z in zn])
    return EpsField(x_axis_u=np.array([0.0, P * 1e9]), y_axis_u=np.array([0.0]), z_axis_u=zn,
                    values_zyx=np.repeat(eg[:, None, None], 2, axis=2).astype(complex))


def fem_strip(film=None, order=4, width=W, cx=P / 2):
    """r of the strip cell through optics.fem_2d_tm; `film` = [(thickness_m, eps)] bottom first
    (default: the uniform 6 nm film)."""
    from dynameta.optics.fem_2d_tm import TMBlock, TMLayer, solve_fem_2d_tm
    film = [(T_FILM_NM * 1e-9, eps_film())] if film is None else film
    lays = [TMLayer("mirror", 80e-9, EPS_MIRROR), TMLayer("lo_oxa", 2e-9, 1.70 ** 2),
            TMLayer("lo_oxb", 6e-9, 2.00 ** 2)]
    lays += [TMLayer("film{}".format(k), t, e) for k, (t, e) in enumerate(film)]
    lays += [TMLayer("up_oxb", 6e-9, 2.00 ** 2), TMLayer("up_oxa", 2e-9, 1.70 ** 2),
             TMLayer("grating", H_STRIP, 1.0 + 0j,
                     (TMBlock(cx - width / 2, cx + width / 2, EPS_STRIP, "strip"),))]
    return solve_fem_2d_tm(P, lays, lambda_m=LAM, n_super=1.0, n_sub=N_GLASS, order=order,
                           num_threads=2)


def v0151_solve(design, lam, eps_by_region, *, n_orders, cell_samples=None,
                inclusion_model="auto", n_slices=None):
    """(R, T, r, t) of a 1-D design exactly as v0.15.1's make_lumenairy_rcwa_solver built and
    solved it with its defaults formulation='laurent' (and the given inclusion_model), rebuilt
    from lumenairy's public API."""
    import lumenairy as lum
    from dynameta.core.layered import collapse_regions_to_layers, slice_eps_field
    from dynameta.optics.lumenairy_bridge._common import (angles_rad, lumenairy_has,
                                                          p_basis_conversion, pol_row)
    from dynameta.optics.lumenairy_bridge._shapes import (layer_shapes_payload,
                                                          periodic_layer_eps_cell)
    from dynameta.optics.lumenairy_bridge.rcwa_backend import (_default_raster_samples,
                                                               _meet_sampling, _min_samples)
    from dynameta.optics.rasterize import cell_axes, layer_bg_eps
    from dynameta.optics.tmm_reference import end_media_indices
    n_super, n_sub = end_media_indices(design, lam)
    px, py = float(design.unit_cell.period_x_m), float(design.unit_cell.period_y_m)
    ebr_l = collapse_regions_to_layers(design, eps_by_region or {})
    use_shapes = inclusion_model != "raster" and lumenairy_has("rcwa_analytic_shapes")
    stack = lum.RCWAStack(px, n_superstrate=complex(n_super), n_substrate=complex(n_sub),
                          n_orders=int(n_orders))
    nx_o = stack.n_orders_x
    s_min_x = _min_samples(nx_o)
    sx = (_default_raster_samples(nx_o) if cell_samples is None
          else max(int(cell_samples), s_min_x, 128))
    for L in reversed(design.stack.layers):
        ef = ebr_l.get(L.name)
        if ef is not None and not getattr(ef, "is_uniform", True):
            for slab in reversed(slice_eps_field(ef, 1e-9, n_slices=n_slices)):
                if slab.eps is not None:
                    stack.add_layer(slab.thickness_m, eps=complex(slab.eps))
                else:
                    cell = np.asarray(slab.eps_cell, dtype=complex)[:, :1]
                    stack.add_layer(slab.thickness_m, eps_cell=_meet_sampling(cell, s_min_x, 1),
                                    formulation="laurent")
            continue
        if L.inclusions:
            if use_shapes:
                bg = layer_bg_eps(L, lam, design.materials, ebr_l)
                bg, shapes = layer_shapes_payload(L, lam, design.materials, bg, px, py,
                                                  stack_period_y_m=float(stack.period_y))
                # v0.15.1 passed NO formulation here and so got lumenairy's default, which was
                # 'laurent' through 5.50.x; named explicitly so this rebuild stays the v0.15.1
                # construction on lumenairy 5.51 (whose own default is 'auto')
                stack.add_layer(L.thickness_m, shapes=shapes, eps_background=bg,
                                formulation="laurent")
            else:
                x, y = cell_axes(sx, 1, px, py)
                X, Y = np.meshgrid(x, y, indexing="ij")
                cell = periodic_layer_eps_cell(L, X, Y, lam, design.materials, ebr_l, px, py)
                stack.add_layer(L.thickness_m, eps_cell=np.asarray(cell, dtype=complex),
                                formulation="laurent")
            continue
        if ef is not None:
            stack.add_layer(L.thickness_m, eps=complex(ef.scalar))
        else:
            stack.add_layer(L.thickness_m,
                            eps=complex(design.materials.get(L.background_material).eps(lam)))
    theta, phi = angles_rad(design.optical)
    stack.set_source(lam, theta=theta, phi=phi)
    res = stack.solve(retain_internal=False, stabilize=False)
    row = pol_row(design.optical)
    rf, tf = p_basis_conversion(design.optical.polarization, theta, n_super, n_sub)
    _, R2, T2 = res.efficiencies()
    return (float(np.sum(R2[row])), float(np.sum(T2[row])),
            complex(np.asarray(res.jones_reflection())[row, row]) * complex(rf),
            complex(np.asarray(res.jones_transmission())[row, row]) * complex(tf))


def li_companion_floor():
    """Context for a test that runs Li's rule on an AREA raster: below lumenairy 5.31 (the floor leg) the bridge
    says once per solver that the inverse-rule companion is missing -- pytest.warns there, a null context where
    lumenairy has the companion."""
    import contextlib
    import pytest
    from dynameta.optics.lumenairy_bridge.rcwa_backend import _lumenairy_has
    from dynameta.warning_types import AdvisoryWarning
    if _lumenairy_has("rcwa_eps_cell_normal"):
        return contextlib.nullcontext()
    return pytest.warns(AdvisoryWarning, match="inverse-rule companion")
